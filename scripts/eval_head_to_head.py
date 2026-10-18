"""Head-to-head strength evaluation between checkpoints.

Plays 4-player 1830 games (Rust engine + Rust MCTS) in which two checkpoints
each hold two seats, cycling through all six 2-2 seat arrangements so seat
order and priority deal even out (``--a-seats 1``: side A holds one seat
against three, in all four arrangements; equal strength then scores 0.25). Each checkpoint gets its own inference
server; every worker holds a client for each server, and a seat searches
with its own checkpoint's tree (rebuilt from the current position whenever
the other checkpoint moved in between, kept across its own consecutive moves).

A game scores share-of-winners per seat (1/k for each of k tied leaders on
net worth), so a side's expected score is 0.5 when the checkpoints are equally
strong. Games use the self-play ``auction_unlock`` variant both checkpoints
were trained under; moves are sampled from visit counts before
``--softpick`` decisions and argmax after, with no Dirichlet noise and no
resignation. With ``--start-positions`` games start at the first Stock Round
from self-play's start positions (human auction endings, plus
``--random-start-fraction`` random ones), each start played in all six seat
arrangements. A game still in the private auction after ``--stall``
decisions, or still running after ``--max-length``, is scored on net worth.
Limits count decisions (moves with more than one legal action), as self-play
does. With ``--games-per-task N`` (> 1) each worker plays N games at once
from a shared queue and runs their searches in lockstep -- one request per
round and checkpoint carries every game's leaves, encoded in Rust
(rl18xx/agent/alphazero/lockstep.py) -- the same searches with far less
per-leaf overhead.

    uv run python scripts/eval_head_to_head.py --match 107 7 --games 240
    uv run python scripts/eval_head_to_head.py --match 107 7 --match 57 7 --games 120

A player is a checkpoint -- ``<num>`` (in the current_best session),
``<session>/<num>``, or a path to a ``.pth``; or ``<policy>+<value>``, two
of those, to search with one checkpoint's policy and the other's value
(composite_model.PolicyValueComposite) -- optionally followed by
``@<readouts>`` (default ``--readouts``; the budget at every decision, narrow
ones included -- self-play gives positions with <= 5 legal moves
``min_readouts`` instead), ``/<c_puct_init>`` (applied in every
round: it replaces SelfPlayConfig's per-round c_puct_by_round table) and
``:<option>,<option>...``:

- ``mean`` / ``mean=<fpu_reduction>``: mean-Q selection with first-play
  urgency (SelfPlayConfig.mcts_mean_q / fpu_reduction)
- ``par=<n>``: leaves selected per batch (SelfPlayConfig.parallel_readouts)
- ``value=networth``: leaf values from a net-worth heuristic instead of the
  network (SelfPlayConfig.leaf_value_heuristic), to test the search alone;
  with ``mix=<w>`` the heuristic gets weight w and the network 1 - w
- ``min=<n>``: readouts at positions with <= 5 legal moves (self-play's
  ``min_readouts``; default: the player's own budget)
- ``sample=<t>``: no search -- the policy plays a move sampled from
  softmax(logits / t) over the legal moves, its price drawn from the price
  head, as ``eval_policy_only.py`` plays (``@`` and the other options are
  ignored); ``sample=0.5`` is how policy-gradient checkpoints are evaluated
- ``ptemp=<t>``: priors softmax(logits / t) (SelfPlayConfig.prior_temperature;
  a policy-gradient learner's policy is softmax(logits / learner_temperature))
- ``dedupe``: a leaf picked twice in one batch is backed up once
  (SelfPlayConfig.dedupe_leaves)
- ``gumbel=<k>``: Gumbel root selection over the top k moves, sequential
  halving over exactly the ``@`` budget, the move argmax(g + logits +
  sigma(q)) (no softpick sampling for this player); ``gscale=<s>`` scales the
  Gumbel noise (0: deterministic), ``cvisit=`` / ``cscale=`` set sigma's
  constants (SelfPlayConfig.gumbel_k / gumbel_noise_scale / gumbel_c_visit /
  gumbel_c_scale; engine-rs/src/gumbel.rs)
- ``budget=total``: on a seat's own consecutive moves the tree is kept (the
  chosen move's subtree) and by default the next search adds the ``@`` budget
  to its inherited visits ("additional"); ``total`` stops once the root has
  1 + budget visits, inherited ones included (SelfPlayConfig.readout_budget;
  search learning's ``--reuse-tree total``)

Standard errors treat games as independent; the six arrangements of one start
are correlated, which (over the 21 runs checked 2026-10-06) moved a
start-clustered SE by 0.81-1.11x of the reported one.

    uv run python scripts/eval_head_to_head.py --match 7@200 7@64 --games 120
    uv run python scripts/eval_head_to_head.py --match 7@64/0.6 7@64 --games 120
    uv run python scripts/eval_head_to_head.py --match 7@64/0.2:mean 7@64 --games 120
    uv run python scripts/eval_head_to_head.py --match 7@200:value=networth 7@8:value=networth
"""

import argparse
import itertools
import json
import logging
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

LOGGER = logging.getLogger("eval_head_to_head")
CHECKPOINT_DIR = REPO_ROOT / "model_checkpoints"
NUM_PLAYERS = 4


def arrangements(a_seats: int = 2) -> list:
    """Every way to give side A ``a_seats`` of the four seats (side B the rest)."""
    return [
        tuple("A" if seat in chosen else "B" for seat in range(NUM_PLAYERS))
        for chosen in itertools.combinations(range(NUM_PLAYERS), a_seats)
    ]


# Every way to give each of two checkpoints two of the four seats.
ARRANGEMENTS = arrangements(2)
# Game ids of match m are m * MATCH_STRIDE + g. A multiple of the six (and four)
# arrangements, so every start is played in all six in every match (with
# 1_000_000, matches after the first split some starts 2/4); the start comes
# from g alone, so all matches of one run play the same starts.
MATCH_STRIDE = 1_200_000


def resolve_checkpoint(spec: str) -> Path:
    path = Path(spec)
    if path.suffix == ".pth":
        return path.resolve()
    pointers = [json.loads(p.read_text()) for p in CHECKPOINT_DIR.glob("*/current_best.json")]
    if "/" in spec:
        session, num = spec.rsplit("/", 1)
        matches = list(CHECKPOINT_DIR.glob(f"*/{session}/{int(num)}.pth"))
    else:
        if not pointers:
            raise SystemExit(f"No current_best.json under {CHECKPOINT_DIR}; give <session>/<num> instead of {spec!r}")
        best = max(pointers, key=lambda p: p["session"])
        matches = [CHECKPOINT_DIR / best["arch"] / best["session"] / f"{int(spec)}.pth"]
    if not matches or not matches[0].exists():
        raise SystemExit(f"Checkpoint {spec!r} not found ({matches})")
    return matches[0].resolve()


def parse_player(spec: str, default_readouts: int) -> tuple:
    """``(checkpoint path, SelfPlayConfig overrides)`` of a player spec (see the module docstring)."""
    spec, _, options = spec.partition(":")
    checkpoint, _, search = spec.partition("@")
    readouts, _, c_puct = search.partition("/")
    budget = int(readouts) if readouts else default_readouts
    # min_readouts is self-play's budget at narrow positions (<= 5 legal moves,
    # ~half of all decisions); left at its default 50 it overrode the spec there,
    # so "@200 vs @8" averaged ~119 vs ~31 readouts a decision. A spec's budget
    # applies everywhere unless ``min=<n>`` restores a separate one.
    overrides = {"num_readouts": budget, "min_readouts": budget}
    if c_puct:
        overrides.update(c_puct_init=float(c_puct), c_puct_by_round={})
    for option in filter(None, options.split(",")):
        key, _, value = option.partition("=")
        if key == "mean":
            overrides.update(mcts_mean_q=True, fpu_reduction=float(value) if value else 0.0)
        elif key == "par":
            overrides["parallel_readouts"] = int(value)
        elif key == "value" and value == "networth":
            overrides["leaf_value_heuristic"] = "net_worth"
        elif key == "mix":
            overrides["leaf_value_heuristic_weight"] = float(value)
        elif key == "min":
            overrides["min_readouts"] = int(value)
        elif key == "ptemp":
            overrides["prior_temperature"] = float(value)
        elif key == "dedupe":
            overrides["dedupe_leaves"] = True
        elif key == "gumbel":
            overrides["gumbel_k"] = int(value) if value else 16
        elif key == "gscale":
            overrides["gumbel_noise_scale"] = float(value)
        elif key == "cvisit":
            overrides["gumbel_c_visit"] = float(value)
        elif key == "cscale":
            overrides["gumbel_c_scale"] = float(value)
        elif key == "budget":
            if value not in ("additional", "total"):
                raise SystemExit(f"budget={value!r}: additional or total")
            overrides["readout_budget"] = value
        elif key == "sample":
            overrides = {"policy_temperature": float(value) if value else 1.0}
            break
        else:
            raise SystemExit(f"Unknown player option {option!r} in {spec!r}")
    paths = "+".join(str(resolve_checkpoint(part)) for part in checkpoint.split("+"))
    return paths, overrides


def model_factory(checkpoint_path):
    """Inference-server model loader: one checkpoint, or ``<policy>+<value>``
    paths for a composite (module-level so it pickles)."""
    from rl18xx.agent.alphazero.composite_model import load_policy_value

    return load_policy_value(checkpoint_path)


_CLIENTS: dict = {}


def worker_init(server_queues: dict, jobs=None, results=None):
    """Pool initializer: take one slot on every checkpoint's inference server
    (and, for lockstep tasks, the shared job and result queues)."""
    from rl18xx.agent.alphazero.inference_server import InferenceClient

    global _JOBS, _RESULTS
    _JOBS, _RESULTS = jobs, results
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    for name, (request_q, reply_qs, ticket_q) in server_queues.items():
        worker_id = ticket_q.get()
        _CLIENTS[name] = InferenceClient(request_q=request_q, reply_q=reply_qs[worker_id], worker_id=worker_id)
    # Forked workers inherit the parent's INFO logging; per-move player logs would flood it.
    logging.getLogger().setLevel(logging.WARNING)


def _expand_root(player):
    """Evaluate a fresh root once, as ``SelfPlay.play`` does before its first search."""
    from rl18xx.agent.alphazero.self_play import _slice_price_components

    leaf = player.root.select_leaf()
    leaf.ensure_encoded()
    probs, _, value = player._backend.run_encoded(leaf.encoded_game_state)
    price_components = _slice_price_components(getattr(player._backend, "last_price_components", None), 0)
    leaf.incorporate_results(probs, value, leaf, price_components=price_components)


class PolicyPlayer:
    """A ``sample=<t>`` player: no search. Each move is sampled from softmax(logits
    / t) over the legal moves and its price drawn from the price head, as
    ``eval_policy_only.py`` plays; forced moves are played without the network."""

    def __init__(self, client, temperature: float):
        self.client = client
        self.temperature = temperature
        self.rng = np.random.default_rng(np.random.randint(2**31))
        self.game = None
        self.price_row = None
        self.decisions = 0

    def initialize_game(self, game):
        self.game = game

    def legal(self) -> list:
        return [int(i) for i in self.game._game.factored_legal_indices()]

    def needs_network(self):
        """The legal moves when this move is a decision (several moves, or one
        with an open price), else None: the one move is played as is."""
        from rl18xx.agent.alphazero.policy_selfplay import has_price_choice

        legal = self.legal()
        self.price_row = None
        if len(legal) == 1 and not has_price_choice(self.game._game, legal[0]):
            return None
        return legal

    def suggest_move(self) -> int:
        from rl18xx.agent.alphazero.mcts import _rust_encode

        legal = self.needs_network()
        if legal is None:
            return self.legal()[0]
        probs, _, _ = self.client.run_many_encoded([_rust_encode(self.game)], legal_indices=[legal])
        price = self.client.last_price_components
        price_row = np.asarray(price["price_logits"])[0] if price is not None else None
        return self.choose(legal, np.asarray(probs[0])[legal], price_row)

    def choose(self, legal: list, priors, price_row) -> int:
        """Sample a move from the network's priors of ``legal``; ``price_row`` (the
        price head's output at this position) prices it in :meth:`play_move`."""
        self.price_row = price_row
        p = np.asarray(priors).astype(np.float64)
        if self.temperature != 1.0:
            p = np.power(np.clip(p, 1e-12, None), 1.0 / self.temperature)
        p = p / p.sum() if p.sum() > 0 else np.full(len(legal), 1.0 / len(legal))
        return legal[int(self.rng.choice(len(legal), p=p))]

    def play_move(self, move: int) -> None:
        from rl18xx.agent.alphazero import policy_gradient as pg

        price, _, _ = pg.choose_price(self.game, move, self.price_row, self.rng, 0.0)
        self.game._game.apply_action_index(int(move), price)

    def get_game_state(self):
        return self.game


def _legal_at_root(player) -> list:
    if isinstance(player, PolicyPlayer):
        return player.legal()
    return player._rust_player.legal_action_indices_at_root()


def _setup_game(game_idx: int, seat_names: tuple, settings: dict) -> tuple:
    """``(game, start label, players by name)`` for one game: its start position
    and a player per spec, each on its checkpoint's client."""
    from engine_rs import BaseGame as RustBaseGame

    from rl18xx.agent.alphazero.config import SelfPlayConfig
    from rl18xx.agent.alphazero.rust_mcts_player import RustMCTSPlayer
    from rl18xx.rust_adapter import RustGameAdapter

    start_label = None
    if settings.get("start_positions"):
        # Self-play's starts (first Stock Round). Seeded by the start index so
        # all six seat arrangements of a start play from the same position.
        from rl18xx.agent.alphazero.start_positions import apply_actions, new_game, sample_start_position

        n_arrangements = len(settings.get("arrangements", ARRANGEMENTS))
        start_rng = random.Random(f"{settings['seed']}:{(game_idx % MATCH_STRIDE) // n_arrangements}")
        start_position = sample_start_position(
            NUM_PLAYERS, settings["start_positions"], settings["random_start_fraction"], rng=start_rng
        )
        game = new_game(NUM_PLAYERS)
        apply_actions(game, start_position.actions)
        start_label = start_position.label
    else:
        rust_game = RustBaseGame({i + 1: f"Player {i + 1}" for i in range(NUM_PLAYERS)})
        rust_game.set_auction_unlock(True)
        game = RustGameAdapter(rust_game)

    players = {}
    for name in sorted(set(seat_names)):
        spec = settings["players"][name]
        if "policy_temperature" in spec["overrides"]:
            players[name] = PolicyPlayer(_CLIENTS[spec["server"]], spec["overrides"]["policy_temperature"])
            continue
        config = SelfPlayConfig(
            network=None,
            use_inference_server=True,
            inference_client=_CLIENTS[spec["server"]],
            **spec["overrides"],
            softpick_move_cutoff=settings["softpick"],
            dirichlet_noise_weight=0.0,
            enable_resign=False,
            noresign_holdout_rate=0.0,
            max_game_length=settings["max_length"],
            player_count_distribution={NUM_PLAYERS: 1.0},
            auction_unlock=True,
        )
        players[name] = RustMCTSPlayer(config)
    return game, start_label, players


def _game_record(game_idx, seat_names, game, start_label, decision_count, decisions, termination, start) -> dict:
    from rl18xx.agent.alphazero.self_play import _auction_unlock_discounts

    seat_by_player_id = {player.id: seat for seat, player in enumerate(game.players)}
    net_worth = {seat_by_player_id[pid]: float(v) for pid, v in game.result().items()}
    best = max(net_worth.values())
    winners = [seat for seat, v in net_worth.items() if v >= best - 1e-6]
    total = sum(max(v, 0.0) for v in net_worth.values()) or 1.0
    return {
        "game_idx": game_idx,
        "seats": list(seat_names),
        "termination": termination,
        "engine_moves": int(game.move_number),
        "decision_count": decision_count,
        "start": start_label,
        "decisions": dict(decisions),
        "net_worth": [net_worth[s] for s in range(NUM_PLAYERS)],
        "worth_share": [max(net_worth[s], 0.0) / total for s in range(NUM_PLAYERS)],
        "win_share": [(1.0 / len(winners) if s in winners else 0.0) for s in range(NUM_PLAYERS)],
        "auction_unlock_discounts": _auction_unlock_discounts(game),
        "seconds": round(time.time() - start, 1),
        "raw_actions": list(game.raw_actions),
    }


def play_game(game_idx: int, seat_names: tuple, settings: dict) -> dict:
    """Play one game with ``seat_names[i]`` (a player spec) searching for seat ``i``."""
    seed = settings["seed"] * 100_003 + game_idx
    random.seed(seed)
    np.random.seed(seed % (2**32))
    start = time.time()
    game, start_label, players = _setup_game(game_idx, seat_names, settings)

    seat_by_player_id = {player.id: seat for seat, player in enumerate(game.players)}
    last_mover = None
    decisions = Counter()
    decision_count = 0
    termination = None
    while True:
        if game.finished:
            termination = "finished"
            break
        if decision_count >= settings["max_length"]:
            game.end_game()
            termination = "max_length"
            break
        round_name = game.round.__class__.__name__
        if settings["stall"] and "Auction" in round_name and decision_count >= settings["stall"]:
            termination = "auction_stall"
            break

        seat = seat_by_player_id[game.active_players()[0].id]
        name = seat_names[seat]
        player = players[name]
        if last_mover != name:
            player.initialize_game(game.pickle_clone())
            player.decisions = decision_count  # the softpick cutoff counts the game's decisions
            if not isinstance(player, PolicyPlayer):
                _expand_root(player)
        if len(_legal_at_root(player)) > 1:
            decision_count += 1
            decisions["auction" if "Auction" in round_name else round_name.lower()] += 1
        move = player.suggest_move()
        player.play_move(move)
        game = player.get_game_state()
        last_mover = name

    return _game_record(game_idx, seat_names, game, start_label, decision_count, decisions, termination, start)


class _LockstepGame:
    """One game of a lockstep task (``play_games``): ``play_game``'s loop cut at
    each network need -- a search (``TreeSearch``) or a policy player's move
    (``PolicyRequest``) -- so a worker's games share requests."""

    def __init__(self, game_idx: int, seat_names: tuple, settings: dict):
        self.game_idx, self.seat_names, self.settings = game_idx, seat_names, settings
        self.start = time.time()
        self.game, self.start_label, self.players = _setup_game(game_idx, seat_names, settings)
        self.seat_by_player_id = {player.id: seat for seat, player in enumerate(self.game.players)}
        self.last_mover = None
        self.decisions = Counter()
        self.decision_count = 0
        self.termination = None
        self.player = self.name = None

    def next_need(self):
        """Play moves until one needs the network and return that need (None:
        the game is over)."""
        from rl18xx.agent.alphazero.lockstep import PolicyRequest

        settings = self.settings
        while True:
            game = self.game
            if game.finished:
                self.termination = "finished"
                return None
            if self.decision_count >= settings["max_length"]:
                game.end_game()
                self.termination = "max_length"
                return None
            round_name = game.round.__class__.__name__
            if settings["stall"] and "Auction" in round_name and self.decision_count >= settings["stall"]:
                self.termination = "auction_stall"
                return None
            seat = self.seat_by_player_id[game.active_players()[0].id]
            self.name = self.seat_names[seat]
            self.player = player = self.players[self.name]
            if self.last_mover != self.name:
                player.initialize_game(game.pickle_clone())
                player.decisions = self.decision_count  # the softpick cutoff counts the game's decisions
            if len(_legal_at_root(player)) > 1:
                self.decision_count += 1
                self.decisions["auction" if "Auction" in round_name else round_name.lower()] += 1
            key = settings["players"][self.name]["server"]
            if isinstance(player, PolicyPlayer):
                legal = player.needs_network()
                if legal is not None:
                    return PolicyRequest(player.game._game, legal, key=key)
                self.play(player.legal()[0])
                continue
            # A fresh root is expanded by the search's first round (play_game's _expand_root).
            search = player.start_search(key=key)
            if search is not None:
                return search
            self.play(int(_legal_at_root(player)[0]))

    def answer(self, need) -> None:
        """Play the move a finished need gives."""
        if isinstance(self.player, PolicyPlayer):
            self.play(self.player.choose(need.legal, need.priors, need.price_row))
        else:
            self.play(self.player.pick_move())

    def play(self, move: int) -> None:
        self.player.play_move(move)
        self.game = self.player.get_game_state()
        self.last_mover = self.name

    def record(self) -> dict:
        return _game_record(
            self.game_idx, self.seat_names, self.game, self.start_label, self.decision_count, self.decisions,
            self.termination, self.start,
        )


_JOBS = None  # lockstep: the shared queue of games to play (worker_init)
_RESULTS = None  # lockstep: finished games go back on this queue


def serve_games(settings: dict, concurrency: int) -> int:
    """Pool task with ``--games-per-task`` > 1: play games from the shared job
    queue, ``concurrency`` at a time, until it is empty. Every game's pending
    search or policy move joins one lockstep round, whose requests carry the
    leaves of all of them (rl18xx/agent/alphazero/lockstep.py); a game that
    ends is replaced by the next job at once, so the batch stays full. Each
    finished game goes back on the results queue. Returns the games played."""
    import queue

    from rl18xx.agent.alphazero.lockstep import ClientEvaluator, LockstepSearcher, PolicyRequest

    seed = settings["seed"] * 100_003 + os.getpid()
    random.seed(seed)
    np.random.seed(seed % (2**32))
    searcher = LockstepSearcher(
        {name: ClientEvaluator(client) for name, client in _CLIENTS.items()},
        dedupe_maps=bool(settings.get("dedupe_maps", False)),
    )
    runs, needs, labels = {}, {}, {}
    exhausted, slot, played = False, 0, 0
    while True:
        while not exhausted and len(runs) < concurrency:
            try:
                job = _JOBS.get(timeout=1.0)
            except queue.Empty:
                continue
            if job is None:
                exhausted = True
                break
            match_index, arrangement, game_idx, seat_names = job
            run = _LockstepGame(game_idx, seat_names, settings)
            runs[slot], labels[slot] = run, (match_index, arrangement)
            needs[slot] = run.next_need()
            slot += 1
        for k in [k for k, need in needs.items() if need is None]:
            record = runs.pop(k).record()
            stats = searcher.stats
            record["lockstep"] = {"concurrency": concurrency, "rounds": stats.rounds, "requests": stats.requests,
                                  "leaves": stats.leaves, "policy_states": stats.policy_states}
            _RESULTS.put((*labels.pop(k), record))
            del needs[k]
            played += 1
        if not needs:
            if exhausted:
                return played
            continue
        active = list(needs.values())
        searcher.round(
            [n for n in active if not isinstance(n, PolicyRequest)],
            [n for n in active if isinstance(n, PolicyRequest)],
        )
        for k, need in needs.items():
            if need.done:
                runs[k].answer(need)
                needs[k] = runs[k].next_need()


def _collect_lockstep(pool, servers: dict, args, result_q, total: int, store, results_file, settings: dict) -> None:
    """``--games-per-task`` > 1: one ``serve_games`` task per worker, each
    playing games from the shared job queue; store games as they come back.
    Fails (killing the workers) if an inference server dies, as
    ``wait_checking_servers`` does."""
    import queue

    futures = [pool.submit(serve_games, settings, args.games_per_task) for _ in range(args.workers)]
    received = 0
    while received < total:
        try:
            match_index, arrangement, record = result_q.get(timeout=5.0)
        except queue.Empty:
            dead = [name for name, h in servers.items() if h.process is not None and not h.process.is_alive()]
            if dead:
                for process in list((getattr(pool, "_processes", None) or {}).values()):
                    process.terminate()
                pool.shutdown(wait=False, cancel_futures=True)
                raise RuntimeError(f"inference server died: {', '.join(dead)}")
            if all(f.done() for f in futures):
                break  # every task ended (some crashed: logged below)
            continue
        received += 1
        store(match_index, arrangement, record, results_file, last=received == total)
    for f in futures:
        if f.done() and f.exception() is not None:
            LOGGER.error(f"Lockstep task crashed (its games in play were lost): {f.exception()!r}")
    if received < total:
        LOGGER.error(f"{total - received} of {total} games were lost to crashed tasks")


def summarize(match_label: str, names: dict, results: list) -> dict:
    """Side A's score with a standard error, plus a breakdown."""
    side_scores, side_worth, by_arrangement = [], [], defaultdict(list)
    seat_scores = defaultdict(list)
    for r in results:
        a_seats = [s for s, side in enumerate(r["seats"]) if side == "A"]
        score = sum(r["win_share"][s] for s in a_seats)
        side_scores.append(score)
        side_worth.append(sum(r["worth_share"][s] for s in a_seats))
        by_arrangement["".join(r["seats"])].append(score)
        for s, side in enumerate(r["seats"]):
            seat_scores[(side, s)].append(r["win_share"][s])
    n = len(side_scores)
    a_seat_count = results[0]["seats"].count("A") if results else 2
    mean = float(np.mean(side_scores)) if n else float("nan")
    se = float(np.std(side_scores, ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
    terminations = Counter(r["termination"] for r in results)
    return {
        "match": match_label,
        "A": names["A"],
        "B": names["B"],
        "games": n,
        "A_score": mean,
        "A_score_se": se,
        "A_score_equal": a_seat_count / NUM_PLAYERS,
        "A_worth_share": float(np.mean(side_worth)) if n else float("nan"),
        "by_arrangement": {k: round(float(np.mean(v)), 3) for k, v in sorted(by_arrangement.items())},
        "seat_win_rate": {f"{side}{s}": round(float(np.mean(v)), 3) for (side, s), v in sorted(seat_scores.items())},
        "terminations": dict(terminations),
        "mean_engine_moves": float(np.mean([r["engine_moves"] for r in results])) if n else float("nan"),
        "mean_auction_decisions": (
            float(np.mean([r["decisions"].get("auction", 0) for r in results])) if n else float("nan")
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--match", nargs=2, action="append", metavar=("A", "B"), required=True)
    parser.add_argument("--games", type=int, default=240, help="Games per match (rounded up to a multiple of 6)")
    parser.add_argument("--readouts", type=int, default=64)
    parser.add_argument("--softpick", type=int, default=30, help="Sample from visit counts for this many decisions")
    parser.add_argument("--max-length", type=int, default=1000)
    parser.add_argument(
        "--stall", type=int, default=400, help="End a game still in the auction after this many decisions"
    )
    parser.add_argument("--workers", type=int, default=48)
    parser.add_argument(
        "--games-per-task", type=int, default=1,
        help="Games each worker plays at once with their searches in lockstep: one request per round carries "
        "the leaves of all of them, encoded natively (rl18xx/agent/alphazero/lockstep.py); 1 plays one game "
        "per task through RustMCTSPlayer.suggest_move as before",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--dedupe-maps", action="store_true",
        help="--games-per-task > 1: send each distinct hex map of a request once (the map encoder runs once per map)",
    )
    parser.add_argument(
        "--server-replicas", type=int, default=1,
        help="Inference server processes per checkpoint (workers split between them)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--start-positions", default=None,
        help="Start games at the first Stock Round from this start-position file, as self-play does",
    )
    parser.add_argument("--random-start-fraction", type=float, default=0.2)
    parser.add_argument(
        "--a-seats", type=int, default=2, choices=(1, 2, 3),
        help="Seats side A holds (2: 2v2 in six arrangements; 1 / 3: 1v3 / 3v1 in four); equal strength scores a_seats/4",
    )
    parser.add_argument("--out", type=Path, default=None, help="Output directory (default logs/eval/<timestamp>)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out_dir = args.out or REPO_ROOT / "logs" / "eval" / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.softpick % 2:
        raise SystemExit("--softpick must be even (SelfPlayConfig asserts it)")

    players = {}
    for a, b in args.match:
        for spec in (a, b):
            path, overrides = parse_player(spec, args.readouts)
            players[spec] = {"server": str(path), "overrides": overrides}
    paths = sorted({p["server"] for p in players.values()})
    seatings = arrangements(args.a_seats)
    games_per_match = math.ceil(args.games / len(seatings)) * len(seatings)
    settings = {
        "readouts": args.readouts,
        "softpick": args.softpick,
        "max_length": args.max_length,
        "stall": args.stall,
        "seed": args.seed,
        "players": players,
        "start_positions": args.start_positions,
        "random_start_fraction": args.random_start_fraction,
        "arrangements": seatings,
        "dedupe_maps": args.dedupe_maps,
    }
    (out_dir / "settings.json").write_text(
        json.dumps({**settings, "matches": args.match, "games_per_match": games_per_match}, indent=2)
    )

    from rl18xx.agent.alphazero.inference_server import start_inference_server, wait_checking_servers

    servers = {}
    for path in paths:
        LOGGER.info(f"Starting inference server for {path}")
        servers[path] = start_inference_server(
            num_workers=args.workers,
            model_factory=model_factory,
            checkpoint_path=path,
            batch_size=args.batch_size,
            batch_timeout_ms=2.0,
            autocast_device=None,
            replicas=args.server_replicas,
        )
    server_queues = {path: (h.request_q, h.reply_qs, h.ticket_q) for path, h in servers.items()}

    jobs = []
    for match_index, (a, b) in enumerate(args.match):
        for g in range(games_per_match):
            arrangement = seatings[g % len(seatings)]
            seat_names = tuple(a if side == "A" else b for side in arrangement)
            jobs.append((match_index, arrangement, match_index * MATCH_STRIDE + g, seat_names))

    results = defaultdict(list)
    results_path = out_dir / "games.jsonl"
    started = time.time()
    done_count = 0

    def store(match_index, arrangement, record, results_file, last: bool):
        nonlocal done_count
        record["match"] = match_index
        record["seats"] = list(arrangement)
        raw_actions = record.pop("raw_actions")
        (out_dir / "games").mkdir(exist_ok=True)
        (out_dir / "games" / f"{record['game_idx']}.json").write_text(json.dumps(raw_actions))
        results_file.write(json.dumps(record) + "\n")
        results_file.flush()
        results[match_index].append(record)
        done_count += 1
        if done_count % 12 == 0 or last:
            elapsed = time.time() - started
            lines = []
            for mi, (a, b) in enumerate(args.match):
                if results[mi]:
                    s = summarize(f"{a} vs {b}", {"A": a, "B": b}, results[mi])
                    lines.append(f"{a} vs {b}: {s['games']} games, {a} score {s['A_score']:.3f} "
                                 f"± {s['A_score_se']:.3f}")
            LOGGER.info(f"[{done_count}/{len(jobs)} games, {elapsed / 60:.1f} min] " + "; ".join(lines))

    lockstep = args.games_per_task > 1
    if lockstep and any("leaf_value_heuristic" in p["overrides"] for p in players.values()):
        raise SystemExit("value=networth players need --games-per-task 1 (lockstep search has no leaf heuristic)")
    job_q = result_q = None
    if lockstep:
        import multiprocessing as mp

        job_q, result_q = mp.Queue(), mp.Queue()
        for job in jobs:
            job_q.put(job)
        for _ in range(args.workers):
            job_q.put(None)
    try:
        with ProcessPoolExecutor(
            max_workers=args.workers, initializer=worker_init, initargs=(server_queues, job_q, result_q)
        ) as pool, open(results_path, "a") as results_file:
            if lockstep:
                _collect_lockstep(pool, servers, args, result_q, len(jobs), store, results_file, settings)
            else:
                pending = {
                    pool.submit(play_game, game_idx, seat_names, settings): (match_index, arrangement)
                    for match_index, arrangement, game_idx, seat_names in jobs
                }
                while pending:
                    finished, _ = wait_checking_servers(list(pending), servers, pool)
                    for future in finished:
                        match_index, arrangement = pending.pop(future)
                        try:
                            record = future.result()
                        except Exception as e:
                            LOGGER.error(f"Game crashed: {e}", exc_info=True)
                            continue
                        store(match_index, arrangement, record, results_file, last=not pending)
    finally:
        for handle in servers.values():
            handle.shutdown()

    summaries = [
        summarize(f"{a} vs {b}", {"A": a, "B": b}, results[mi]) for mi, (a, b) in enumerate(args.match)
    ]
    (out_dir / "summary.json").write_text(json.dumps(summaries, indent=2))
    for s in summaries:
        LOGGER.info(json.dumps(s, indent=2))
    LOGGER.info(f"Results in {out_dir}")


if __name__ == "__main__":
    main()
