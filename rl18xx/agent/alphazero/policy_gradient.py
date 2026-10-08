"""Policy-gradient refinement of the supervised policy (AlphaGo's RL-policy stage).

AlphaGo went supervised policy -> policy improved by self-play policy gradient
(REINFORCE against a pool of its earlier versions, no search) -> value network
trained on that policy's games -> search. This module is the second step for
1830, with a few later improvements:

- **Games**: 4-player games from the first Stock Round (``start_positions``).
  The learner holds ``learner_seats`` random seats; the others go to one
  opponent: the supervised policy, or (``1 - sl_opponent_fraction`` of the
  games, once snapshots exist) a snapshot of the learner from the pool.
  Every seat samples its moves from its policy (the learner at
  ``learner_temperature``, the opponent at ``opponent_temperature``) and its
  prices from its price head; forced moves are applied without the network.
- **Temperature**: at temperature 1 every seat often samples weak moves, and
  sharpening alone wins (checkpoint 10 at 0.5 scores 0.87 against itself at
  1), so a learner trained at 1 spends its updates and its KL budget getting
  sharper rather than ranking moves better (2026-10-06: its training score
  rose 0.65 -> 0.73 over updates 20-50 while its score at matched
  temperature 0.5 stayed ~0.54). With ``learner_temperature`` tau the learner's
  policy *is* softmax(logits / tau): moves are sampled from it, the gradient
  and the KL anchor (to the supervised policy at the same tau) are taken
  through it, and opponents at the same tau make the score a matched one.
- **Objective**: each learner decision is credited with its own seat's win
  share (1/k for k tied leaders on net worth) minus a critic's estimate --
  ``A = R - V(s)`` with Monte Carlo returns (``gae_lambda`` 1), or a
  lambda-return over the seat's own decisions. Updates use PPO's clipped
  ratio against the probability the move was sampled with (generation runs
  continuously, so a row may come from a policy a few updates old), price
  draws included in the move's probability.
- **Anchor**: a KL(learner || anchor) penalty over the legal moves (and
  the chosen slot's price cells) limits how far the policy moves; with
  ``kl_target`` the coefficient adapts toward that KL per decision. The anchor
  is the starting policy, or with ``anchor_every`` N the learner itself as of
  every N-th update (as DeepNash's R-NaD resets its regularization policy):
  a fixed anchor caps the total distance -- pg3 plateaued at its 0.08 budget
  around update 350 -- while a moving one caps the speed. Without any anchor,
  noisy updates random-walk (pg2 drifted to 0.19 and got weaker).
- **Starts**: every game begins at the first Stock Round (human or random
  auction endings) or, in ``midgame_fraction`` of them, at a later Stock Round
  of a recorded human game (``midgame_positions``; see ``start_positions``),
  which puts the agents into positions self-play rarely reaches. ``starts`` in
  the history counts the games by kind.
- **Opponents**: the fixed opponent (``opponent_checkpoint``, default the
  starting policy) in ``sl_opponent_fraction`` of the games, the pool of
  learner snapshots (seeded with the starting policy) in the rest. With a
  moving anchor, keeping the supervised policy as the fixed opponent makes
  ``score_vs_sl`` a check that self-play gains still hold against human-like
  play.
- **Critic**: a separate value network (``value_checkpoint``, e.g. the network
  trained on policy-only games) gives the baseline at generation time -- the
  learner's inference server is a ``PolicyValueComposite`` -- and keeps
  training on the learner's games (each position seen once), so it stays
  calibrated as the policy changes and becomes the next value network.

Progress is the learner's score in its games against the supervised policy:
the summed win share of its seats, 0.5 per two seats when equally strong (as
``scripts/eval_head_to_head.py`` scores, but without search). Checkpoints
land in ``<out_dir>/<run>/{learner,critic}/<update>.pth`` (every
``snapshot_every``-th kept; those join the opponent pool), TensorBoard in
``runs/alphazero_runs/<run>``, and one JSON line per update in
``<out_dir>/<run>/history.jsonl``. Touch ``<out_dir>/<run>/STOP`` to end a run
after the current update.

With ``save_game_every`` N (off by default) about one game in N (sampled at
random) also keeps its action log, as ``<out_dir>/<run>/games/u<update>_<id>.json``
(``game_records``) with a line in ``<out_dir>/<run>/games.jsonl``: the update
(updates completed when the game was collected), which seats the learner held,
the opponent and its checkpoint, the start, termination and result -- to
browse in the dashboard's game viewer (``/games``).
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import time
import uuid
from collections import Counter, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

LOGGER = logging.getLogger(__name__)

NUM_PLAYERS = 4


@dataclass
class PGConfig:
    policy_checkpoint: str
    value_checkpoint: str
    out_dir: str = "model_checkpoints_pg"
    run_name: Optional[str] = None
    # Generation
    workers: int = 48
    games_per_task: int = 32
    tasks_in_flight_per_worker: int = 2
    learner_seats: int = 2
    positions_per_seat: int = 32
    sl_opponent_fraction: float = 0.5
    # The fixed opponent ("sl" server); None: the starting policy.
    opponent_checkpoint: Optional[str] = None
    # Games against the pool start at once, the pool seeded with the starting policy.
    seed_pool: bool = True
    # 0: the KL anchor is the starting policy for the whole run. N: every N
    # updates the anchor becomes the current learner (a multiple of snapshot_every).
    anchor_every: int = 0
    # League mode (population set): opponents come from a population of checkpoints
    # rather than the fixed opponent + pool. Each game deals the learner 1-3 seats
    # (learner_seat_weights) and every other seat one of opponent_slots servers, each
    # holding a population member; every slot_refresh_every updates one slot reloads a
    # member sampled by prioritized fictitious self-play (League).
    population: Optional[str] = None
    opponent_slots: int = 3
    learner_seat_weights: Optional[dict] = None
    pfsp_power: float = 2.0
    slot_refresh_every: int = 2
    league_add_snapshots: bool = True
    learner_temperature: float = 1.0
    opponent_temperature: float = 1.0
    opponent_price_eps: float = 0.0
    max_decisions: int = 1000
    start_positions: str = "human_games/start_positions_1830_4p.jsonl"
    random_start_fraction: float = 0.2
    # Mid-game starts (start_positions.sample_start_position): this share of the
    # games begins at a later Stock Round of a recorded human game.
    midgame_positions: Optional[str] = None
    midgame_fraction: float = 0.0
    inference_batch_size: int = 512
    # Updates
    rows_per_update: int = 65536
    minibatch: int = 256  # rows per optimizer step
    microbatch: int = 256  # rows per forward/backward (gradients accumulate up to minibatch)
    ppo_epochs: int = 1
    clip: float = 0.2
    lr: float = 1e-5
    critic_lr: float = 3e-5
    kl_coef: float = 0.02
    kl_target: Optional[float] = None
    kl_coef_min: float = 0.005  # floor for the adaptive coefficient
    entropy_coef: float = 0.0
    gae_lambda: float = 1.0
    normalize_advantages: bool = False
    score_loss_weight: float = 1.0
    max_updates: int = 1000
    snapshot_every: int = 10
    pool_refresh_every: int = 5
    score_window: int = 2000
    # 0: off. N: keep the action log of about one game in N for the game viewer.
    save_game_every: int = 0


# ---------------------------------------------------------------------------
# Worker side: games
# ---------------------------------------------------------------------------

_CLIENTS: dict = {}


def worker_init(server_queues: dict):
    """Pool initializer: take one slot on every inference server ("learner", "sl", "pool")."""
    from rl18xx.agent.alphazero.inference_server import InferenceClient

    for name, (request_q, reply_qs, ticket_q) in server_queues.items():
        worker_id = ticket_q.get()
        _CLIENTS[name] = InferenceClient(request_q=request_q, reply_q=reply_qs[worker_id], worker_id=worker_id)
    logging.getLogger().setLevel(logging.WARNING)


@dataclass
class _PGGame:
    game: object  # RustGameAdapter
    uid: str
    opponent: str  # "sl" or "pool"
    seat_models: list  # server name per seat
    decisions: int = 0
    price_row: Optional[np.ndarray] = None
    termination: Optional[str] = None
    values: dict = field(default_factory=dict)  # learner seat -> critic value at each of its decisions
    seen: dict = field(default_factory=dict)  # learner seat -> decisions so far
    reservoir: dict = field(default_factory=dict)  # learner seat -> [(decision number, row)]
    seat_members: dict = field(default_factory=dict)  # league mode: opponent seat -> member label
    start: Optional[str] = None  # the start position's label


def choose_price(game, index: int, price_row: Optional[np.ndarray], rng: np.random.Generator, eps: float):
    """``(price, (slot, cell, lo, hi) or None, log P(cell))`` for a legal ``index``:
    no price for categorical moves, the engine price for fixed-price ones, and
    for price-head slots a cell drawn from the head (mixed with ``eps``
    uniform) and a price inside it."""
    from rl18xx.agent.alphazero import price_pmf

    rs = game._game
    price_range = rs.price_range_for_index(int(index))
    if price_range is None:
        return None, None, 0.0
    lo, hi = price_range
    slot = rs.price_head_slot_for_index(int(index))
    if lo == hi or slot is None:
        return int(lo), None, 0.0
    slot_index, lo, hi = slot
    cells = price_pmf.price_cells(price_pmf.SLOT_TYPES[slot_index], int(lo), int(hi))
    probs = cells.cell_probs(price_row[slot_index] if price_row is not None else None)
    live = cells.nonempty
    mixed = np.where(live, (1.0 - eps) * probs + eps / live.sum(), 0.0)
    mixed = mixed / mixed.sum()
    cell = int(rng.choice(len(mixed), p=mixed))
    return int(cells.sample_price(cell, rng)), (int(slot_index), cell, int(lo), int(hi)), float(np.log(mixed[cell]))


def seat_advantages(values: list, reward: float, lam: float) -> np.ndarray:
    """Lambda-return advantages over one seat's decisions: ``delta_t = v_{t+1} - v_t``
    (``reward`` after the last), ``A_t = delta_t + lam * A_{t+1}``. With lam 1
    every decision gets ``reward - v_t``."""
    advantages = np.zeros(len(values), dtype=np.float32)
    following = 0.0
    for t in range(len(values) - 1, -1, -1):
        next_value = values[t + 1] if t + 1 < len(values) else reward
        following = (next_value - values[t]) + lam * following
        advantages[t] = following
    return advantages


def outcome(game) -> tuple:
    """``(win share, net-worth fractions)`` per seat (player-id order)."""
    from rl18xx.agent.alphazero.self_play import _compute_net_worth

    net_worth = _compute_net_worth(game)
    worth = np.array([float(net_worth[pid]) for pid in sorted(net_worth)], dtype=np.float64)
    best = worth.max()
    winners = worth >= best - 1e-6
    win_share = winners / winners.sum()
    clipped = worth.clip(min=0)
    fractions = clipped / clipped.sum() if clipped.sum() > 0 else np.full(len(worth), 1.0 / len(worth))
    return win_share.astype(np.float32), fractions.astype(np.float32)


DEFAULT_LEARNER_SEAT_WEIGHTS = {1: 0.4, 2: 0.4, 3: 0.2}


def load_population(path) -> list:
    """Checkpoint paths of a population file: a JSON list of paths or of
    ``{"path": ..., ...}`` objects, or ``{"members": [...]}``."""
    data = json.loads(Path(path).read_text())
    members = data["members"] if isinstance(data, dict) else data
    return [m["path"] if isinstance(m, dict) else str(m) for m in members]


class League:
    """The opponents of a league-mode run, sampled by prioritized fictitious self-play
    (AlphaStar): a member is drawn with weight ``(1 - p) ** power``, ``p`` the
    learner's (decayed) rate of finishing ahead of it, so members it rarely beats are
    played most. A new member starts at ``p`` 0.5 (``prior`` pseudo-games)."""

    def __init__(self, members=(), power: float = 2.0, floor: float = 0.02, decay: float = 0.995, prior: float = 4.0):
        self.power, self.floor, self.decay, self.prior = power, floor, decay, prior
        self.stats: dict = {}
        for member in members:
            self.add(member)

    def add(self, member: str) -> None:
        self.stats.setdefault(str(member), [0.5 * self.prior, self.prior])

    @property
    def members(self) -> list:
        return list(self.stats)

    def record(self, member: str, learner_result: float) -> None:
        """One comparison: 1 if the learner finished ahead of ``member``'s seat, 0.5 tied, 0 behind."""
        if member not in self.stats:
            return
        wins, games = self.stats[member]
        self.stats[member] = [wins * self.decay + learner_result, games * self.decay + 1.0]

    def win_rate(self, member: str) -> float:
        wins, games = self.stats[member]
        return wins / games

    def weights(self) -> np.ndarray:
        rates = np.array([self.win_rate(m) for m in self.stats])
        w = np.maximum((1.0 - rates) ** self.power, self.floor)
        return w / w.sum()

    def sample(self, rng: np.random.Generator) -> str:
        return self.members[int(rng.choice(len(self.stats), p=self.weights()))]

    def state(self) -> dict:
        return {"stats": self.stats}

    @classmethod
    def from_state(cls, state: dict, **kwargs) -> "League":
        league = cls(**kwargs)
        league.stats = {k: list(v) for k, v in state["stats"].items()}
        return league


def _deal_seats(settings: dict, py_rng: random.Random) -> tuple:
    """``(learner seats, seat -> server name, seat -> member label)`` for a new game."""
    league = settings.get("league")
    if not league:
        opponent = "pool" if settings["pool_ready"] and py_rng.random() >= settings["sl_opponent_fraction"] else "sl"
        learner = set(py_rng.sample(range(NUM_PLAYERS), settings["learner_seats"]))
        models = ["learner" if seat in learner else opponent for seat in range(NUM_PLAYERS)]
        return learner, models, {}
    counts, weights = zip(*sorted((int(k), float(v)) for k, v in league["learner_seat_weights"].items()))
    count = py_rng.choices(counts, weights=weights)[0]
    learner = set(py_rng.sample(range(NUM_PLAYERS), count))
    models = ["learner" if seat in learner else py_rng.choice(league["slots"]) for seat in range(NUM_PLAYERS)]
    labels = {seat: league["slot_labels"][models[seat]] for seat in range(NUM_PLAYERS) if seat not in learner}
    return learner, models, labels


def play_pg_games(num_games: int, settings: dict) -> dict:
    """Pool task: play ``num_games`` concurrent games and return the learner's
    sampled decisions (with advantages) and each game's result."""
    from rl18xx.agent.alphazero.mcts import _rust_encode
    from rl18xx.agent.alphazero.policy_selfplay import _advance
    from rl18xx.agent.alphazero.start_positions import apply_actions, new_game, sample_start_position

    rng = np.random.default_rng()
    py_rng = random.Random()
    k = settings["positions_per_seat"]
    start_time = time.time()

    games = []
    for _ in range(num_games):
        start = sample_start_position(
            NUM_PLAYERS,
            settings["start_positions"],
            settings["random_start_fraction"],
            rng=py_rng,
            midgame_path=settings.get("midgame_positions"),
            midgame_fraction=settings.get("midgame_fraction", 0.0),
        )
        game = apply_actions(new_game(NUM_PLAYERS), start.actions)
        learner, seat_models, seat_members = _deal_seats(settings, py_rng)
        opponent = "league" if settings.get("league") else seat_models[min(set(range(NUM_PLAYERS)) - learner)]
        g = _PGGame(game=game, uid=uuid.uuid4().hex, opponent=opponent, seat_models=seat_models, start=start.label)
        g.seat_members = seat_members
        for seat in learner:
            g.values[seat], g.seen[seat], g.reservoir[seat] = [], 0, []
        games.append(g)

    finished, active, decisions_total = [], list(games), 0
    while active:
        pending = []
        for g in active:
            legal = _advance(g, settings, rng)
            if legal is None:
                finished.append(g)
            else:
                pending.append((g, legal))
        active = [g for g, _ in pending]
        if not pending:
            break
        encoded = [_rust_encode(g.game) for g, _ in pending]
        by_model = defaultdict(list)
        for i, (g, _) in enumerate(pending):
            # The encoder's rotation is the mover's seat (player-id order).
            by_model[g.seat_models[int(encoded[i][6])]].append(i)
        # Both servers work at once: send every request before waiting on any.
        sent = {
            name: _CLIENTS[name].send([encoded[i] for i in rows], legal_indices=[pending[i][1] for i in rows])
            for name, rows in by_model.items()
        }
        for name, rows in by_model.items():
            client = _CLIENTS[name]
            probs, _, values = client.receive(sent[name])
            price = client.last_price_components
            price_logits = price["price_logits"].numpy() if price is not None else None
            is_learner = name == "learner"
            temperature = settings["learner_temperature"] if is_learner else settings["opponent_temperature"]
            for j, i in enumerate(rows):
                g, legal = pending[i]
                seat = int(encoded[i][6])
                p = probs[j].numpy()[legal].astype(np.float64)
                if temperature != 1.0:
                    p = np.power(np.clip(p, 1e-12, None), 1.0 / temperature)
                p = p / p.sum() if p.sum() > 0 else np.full(len(legal), 1.0 / len(legal))
                pos = int(rng.choice(len(legal), p=p))
                choice = legal[pos]
                g.price_row = price_logits[j] if price_logits is not None else None
                eps = 0.0 if is_learner else settings["opponent_price_eps"]
                price_value, price_info, cell_logp = choose_price(g.game, choice, g.price_row, rng, eps)
                if is_learner:
                    t = len(g.values[seat])
                    g.values[seat].append(float(values[j][0]))  # canonical slot 0 is the mover
                    row = (encoded[i], np.asarray(legal, dtype=np.int32), int(choice), float(np.log(p[pos]) + cell_logp), price_info)
                    g.seen[seat] += 1
                    if len(g.reservoir[seat]) < k:
                        g.reservoir[seat].append((t, row))
                    else:
                        r = int(rng.integers(0, g.seen[seat]))
                        if r < k:
                            g.reservoir[seat][r] = (t, row)
                g.game._game.apply_action_index(choice, price_value)
                g.decisions += 1
                decisions_total += 1

    records = []
    save_every = settings.get("save_game_every", 0)
    for g in finished:
        win_share, fractions = outcome(g.game)
        rows = []
        for seat, kept in g.reservoir.items():
            advantages = seat_advantages(g.values[seat], float(win_share[seat]), settings["gae_lambda"])
            for t, (enc, legal, choice, logp, price_info) in kept:
                rows.append((enc, legal, choice, logp, price_info, float(advantages[t]), fractions))
        record = {
            "uid": g.uid,
            "opponent": g.opponent,
            "pool_label": settings.get("pool_label"),
            "learner_seats": sorted(g.values),
            "win_share": win_share.tolist(),
            "decisions": g.decisions,
            "learner_decisions": sum(len(v) for v in g.values.values()),
            "seat_decisions": {seat: len(v) for seat, v in g.values.items()},
            "termination": g.termination,
            "start_kind": g.start.split(":")[0] if g.start else None,
            "first_values": {seat: v[0] for seat, v in g.values.items() if v},
            "rows": rows,
        }
        if g.seat_members:
            # League mode: per opponent seat, did the learner's best seat finish ahead of it?
            best = max(float(fractions[seat]) for seat in g.values)
            record["seat_members"] = g.seat_members
            record["pairwise"] = [
                (member, 1.0 if best > float(fractions[seat]) else 0.5 if best == float(fractions[seat]) else 0.0)
                for seat, member in g.seat_members.items()
            ]
        if save_every and py_rng.random() * save_every < 1.0:
            from rl18xx.agent.alphazero.self_play import _compute_net_worth

            net_worth = _compute_net_worth(g.game)
            record["saved_game"] = {
                "start": g.start,
                "net_worth": [float(net_worth[pid]) for pid in sorted(net_worth)],
                "worth_share": fractions.tolist(),
                "raw_actions": list(g.game.raw_actions),
            }
        records.append(record)
    return {"games": records, "decisions": decisions_total, "seconds": time.time() - start_time}


def save_game(run_dir: Path, record: dict, update: int, learner_checkpoint: str, opponent_checkpoint: str, cfg) -> str:
    """Keep a ``save_game_every`` game's action log (``record["saved_game"]``) as
    ``<run_dir>/games/u<update>_<id>.json`` with a line in ``<run_dir>/games.jsonl``;
    returns the file's name."""
    from rl18xx.agent.alphazero.game_records import append_index, make_game_record, save_game_record

    saved = record["saved_game"]
    learner = set(int(seat) for seat in record["learner_seats"])
    name = f"u{update}_{record['uid'][:8]}"
    opponent = {"pool": "pool snapshot", "league": "league member"}.get(record["opponent"], "fixed opponent")
    members = {int(k): v for k, v in (record.get("seat_members") or {}).items()}
    entry = {
        "name": name,
        "kind": "policy_gradient",
        "run": run_dir.name,
        "update": update,
        "learner_checkpoint": learner_checkpoint,
        "learner_seats": sorted(learner),
        "opponent": record["opponent"],
        "opponent_checkpoint": opponent_checkpoint,
        "sides": ["A" if seat in learner else "B" for seat in range(NUM_PLAYERS)],
        "seat_labels": [
            f"learner (update {update})" if seat in learner else members.get(seat, f"{opponent} {opponent_checkpoint}")
            for seat in range(NUM_PLAYERS)
        ],
        "learner_temperature": cfg.learner_temperature,
        "opponent_temperature": cfg.opponent_temperature,
        "start": saved["start"],
        "termination": record["termination"],
        "decisions": record["decisions"],
        "win_share": record["win_share"],
        "net_worth": saved["net_worth"],
        "worth_share": saved["worth_share"],
    }
    meta = {**entry, "description": f"policy-gradient {run_dir.name} update {update}: learner vs {opponent}"}
    entry["game_file"] = save_game_record(run_dir, name, make_game_record(saved["raw_actions"], NUM_PLAYERS, meta=meta))
    append_index(run_dir, entry)
    return name


# ---------------------------------------------------------------------------
# Learner side: losses
# ---------------------------------------------------------------------------


def batch_tensors(rows: list, device) -> dict:
    """Stack learner rows ``(encoded, legal, choice, old_logp, price_info,
    advantage, fractions)`` into device tensors."""
    from rl18xx.agent.alphazero import price_pmf
    from rl18xx.agent.alphazero.mcts import POLICY_SIZE

    n = len(rows)
    game_state = torch.stack([r[0][0].reshape(-1) for r in rows]).float().to(device, non_blocking=True)
    nodes = torch.stack([r[0][1] for r in rows]).float().to(device, non_blocking=True)
    lengths = torch.tensor([len(r[1]) for r in rows])
    row_ids = torch.repeat_interleave(torch.arange(n), lengths)
    col_ids = torch.from_numpy(np.concatenate([r[1] for r in rows]).astype(np.int64))
    legal = torch.zeros(n, POLICY_SIZE, dtype=torch.bool)
    legal[row_ids, col_ids] = True
    price_rows, price_slots, price_cells, price_masks = [], [], [], []
    for b, r in enumerate(rows):
        if r[4] is not None:
            slot, cell, lo, hi = r[4]
            price_rows.append(b)
            price_slots.append(slot)
            price_cells.append(cell)
            price_masks.append(price_pmf.price_cells(price_pmf.SLOT_TYPES[slot], lo, hi).nonempty)
    # Value targets rotated into the encoder's frame (the mover first).
    fractions = torch.stack(
        [torch.roll(torch.as_tensor(r[6]), shifts=-int(r[0][6]), dims=0) for r in rows]
    ).float()
    out = {
        "game_state": game_state,
        "nodes": nodes,
        "legal": legal.to(device, non_blocking=True),
        "choice": torch.tensor([r[2] for r in rows], dtype=torch.long, device=device),
        "old_logp": torch.tensor([r[3] for r in rows], dtype=torch.float32, device=device),
        "advantage": torch.tensor([r[5] for r in rows], dtype=torch.float32, device=device),
        "fractions": fractions.to(device),
        "price_rows": torch.tensor(price_rows, dtype=torch.long, device=device),
        "price_slots": torch.tensor(price_slots, dtype=torch.long, device=device),
        "price_cells": torch.tensor(price_cells, dtype=torch.long, device=device),
        "price_masks": (
            torch.from_numpy(np.stack(price_masks)).to(device)
            if price_masks
            else torch.zeros(0, price_pmf.NUM_CELLS, dtype=torch.bool, device=device)
        ),
    }
    return out


def _masked_log_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """log_softmax over the ``mask``ed entries; 0 (not -inf) elsewhere so products with probabilities stay finite."""
    log_p = torch.log_softmax(logits.float().masked_fill(~mask, float("-inf")), dim=-1)
    return log_p.masked_fill(~mask, 0.0)


def pg_losses(
    policy_logits: torch.Tensor,
    price_logits: Optional[torch.Tensor],
    sl_policy_logits: torch.Tensor,
    sl_price_logits: Optional[torch.Tensor],
    batch: dict,
    clip: float,
    kl_coef: float,
    entropy_coef: float,
    temperature: float = 1.0,
) -> dict:
    """PPO-clipped policy gradient plus ``kl_coef`` * KL(learner || anchor).

    A move's log-probability is its index's (softmax over the legal moves of
    the logits / ``temperature``) plus, for price-head moves, its price cell's
    (softmax over the slot's non-empty cells) -- the probabilities the move was
    sampled with. The KL compares both policies at ``temperature``."""
    legal = batch["legal"]
    log_p = _masked_log_softmax(policy_logits / temperature, legal)
    sl_log_p = _masked_log_softmax(sl_policy_logits / temperature, legal)
    p = log_p.exp() * legal
    new_logp = log_p.gather(1, batch["choice"].unsqueeze(1)).squeeze(1)
    kl = (p * (log_p - sl_log_p)).sum(dim=1)
    entropy = -(p * log_p).sum(dim=1)

    price_kl = torch.zeros((), device=policy_logits.device)
    if price_logits is not None and batch["price_rows"].numel():
        rows, slots, masks = batch["price_rows"], batch["price_slots"], batch["price_masks"]
        cell_log_p = _masked_log_softmax(price_logits[rows, slots], masks)
        sl_cell_log_p = _masked_log_softmax(sl_price_logits[rows, slots], masks)
        new_logp = new_logp.index_add(0, rows, cell_log_p.gather(1, batch["price_cells"].unsqueeze(1)).squeeze(1))
        cell_kl = (cell_log_p.exp() * masks * (cell_log_p - sl_cell_log_p)).sum(dim=1)
        kl = kl.index_add(0, rows, cell_kl)
        price_kl = cell_kl.mean()

    advantage = batch["advantage"]
    log_ratio = new_logp - batch["old_logp"]
    ratio = log_ratio.exp()
    surrogate = torch.minimum(ratio * advantage, ratio.clamp(1.0 - clip, 1.0 + clip) * advantage)
    pg_loss = -surrogate.mean()
    total = pg_loss + kl_coef * kl.mean() - entropy_coef * entropy.mean()
    return {
        "total": total,
        "pg_loss": pg_loss.detach(),
        "kl_anchor": kl.mean().detach(),
        "price_kl_anchor": price_kl.detach(),
        "entropy": entropy.mean().detach(),
        "clip_frac": ((ratio - 1.0).abs() > clip).float().mean().detach(),
        "approx_kl": (-log_ratio).mean().detach(),
        "ratio_max": ratio.max().detach(),
    }


def critic_losses(win_loss_logits: torch.Tensor, score_pred: torch.Tensor, fractions: torch.Tensor) -> dict:
    """Win-share CE and net-worth MSE against the game outcome (encoder frame)."""
    from rl18xx.agent.alphazero.train import _derive_dual_value_targets

    win_target, score_target = _derive_dual_value_targets(fractions)
    pad = win_loss_logits.shape[1] - win_target.shape[1]
    if pad:
        win_target = F.pad(win_target, (0, pad))
        score_target = F.pad(score_target, (0, pad))
    log_v = torch.log_softmax(win_loss_logits.float(), dim=1)
    ce = -(win_target * log_v).sum(dim=1).mean()
    mse = F.mse_loss(score_pred.float(), score_target)
    hit = win_target.gather(1, log_v.argmax(dim=1, keepdim=True)).mean()
    return {"ce": ce, "mse": mse, "winner_hit": hit.detach()}


# ---------------------------------------------------------------------------
# Learner side: run
# ---------------------------------------------------------------------------


def _save(model, path: Path) -> None:
    """A self-describing checkpoint (as ``checkpointer.save_model`` writes) at ``path``."""
    from rl18xx.agent.alphazero.checkpointer import (
        ARCHITECTURE_KEY,
        CONFIG_KEY,
        STATE_DICT_KEY,
        _atomic_save_torch,
    )

    config_payload = model.config.to_json()
    config_payload.pop("device", None)
    config_payload[ARCHITECTURE_KEY] = model.architecture_name()
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_save_torch(
        path,
        {STATE_DICT_KEY: model.state_dict(), CONFIG_KEY: config_payload, ARCHITECTURE_KEY: model.architecture_name()},
    )


def use_tf32() -> None:
    """TF32 matmuls: ~1.6x faster than fp32 and within ~0.01 nats of it on a
    sampled move's log-probability, where bf16 is off by 0.2 on average (up to
    3) -- enough to put a sixth of PPO's ratios outside the clip on their own.
    Generation and training both use it, so the ratio sees one precision."""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def load_server_model(checkpoint_paths: str):
    """Inference-server loader (module-level so it pickles): TF32, then one
    checkpoint or a ``<policy>+<value>`` composite."""
    from rl18xx.agent.alphazero.composite_model import load_policy_value

    use_tf32()
    return load_policy_value(checkpoint_paths)


def _load(path) -> torch.nn.Module:
    from rl18xx.agent.alphazero.policy_selfplay import load_checkpoint_model

    return load_checkpoint_model(str(path))


class _Scores:
    """Rolling learner score per opponent kind (summed win share of its seats)."""

    def __init__(self, window: int):
        self.window = defaultdict(lambda: deque(maxlen=window))
        self.recent = defaultdict(list)
        for kind in ("sl", "pool"):  # reported even before their first game, as before
            self.window[kind] = deque(maxlen=window)
            self.recent[kind] = []

    def add(self, record: dict):
        score = sum(record["win_share"][s] for s in record["learner_seats"])
        if record["opponent"] == "league":
            # The learner holds 1-3 seats: its win share as a multiple of a fair one (1.0 = even).
            score /= len(record["learner_seats"]) / NUM_PLAYERS
        self.window[record["opponent"]].append(score)
        self.recent[record["opponent"]].append(score)

    def summary(self) -> dict:
        out = {}
        for kind, values in self.window.items():
            if len(values) > 1:
                out[f"score_vs_{kind}"] = float(np.mean(values))
                out[f"score_vs_{kind}_se"] = float(np.std(values, ddof=1) / math.sqrt(len(values)))
                out[f"games_vs_{kind}_window"] = len(values)
            if self.recent[kind]:
                out[f"score_vs_{kind}_update"] = float(np.mean(self.recent[kind]))
            self.recent[kind] = []
        return out


def run(cfg: PGConfig, resume: Optional[str] = None) -> Path:
    """Run the policy-gradient stage; returns the run directory."""
    from torch.utils.tensorboard import SummaryWriter

    from rl18xx.agent.alphazero.inference_server import start_inference_server, wait_checking_servers

    use_tf32()
    if resume:
        run_dir = Path(resume)
        state = json.loads((run_dir / "state.json").read_text())
        # The run keeps its own settings; only its length and game saving can change.
        cfg = PGConfig(
            **{
                **state["config"],
                "max_updates": cfg.max_updates,
                "save_game_every": cfg.save_game_every,
                "midgame_positions": cfg.midgame_positions,
                "midgame_fraction": cfg.midgame_fraction,
            }
        )
        update = int(state["snapshot_update"])
        pool = list(state["pool"])
        kl_coef = float(state["kl_coef"])
        anchor_update = int(state.get("anchor_update", 0))
        league_state = state.get("league")
        learner_path = run_dir / "learner" / f"{update}.pth"
        critic_path = run_dir / "critic" / f"{update}.pth"
    else:
        name = cfg.run_name or datetime.now().strftime("pg_%Y%m%d_%H%M%S")
        cfg.run_name = name
        run_dir = Path(cfg.out_dir) / name
        if run_dir.exists():
            raise SystemExit(f"{run_dir} exists; pass --resume to continue it")
        run_dir.mkdir(parents=True)
        update, kl_coef, anchor_update = 0, cfg.kl_coef, 0
        pool = [str(cfg.policy_checkpoint)] if cfg.seed_pool else []
        league_state = None
        learner_path, critic_path = Path(cfg.policy_checkpoint), Path(cfg.value_checkpoint)
    if cfg.anchor_every and cfg.anchor_every % cfg.snapshot_every:
        raise SystemExit("--anchor-every must be a multiple of --snapshot-every (the anchor's checkpoint is kept)")
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    LOGGER.info(f"Policy-gradient run {run_dir} from update {update}: {asdict(cfg)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    anchor_path = run_dir / "learner" / f"{anchor_update}.pth" if anchor_update else Path(cfg.policy_checkpoint)
    learner, critic, anchor = _load(learner_path), _load(critic_path), _load(anchor_path)
    anchor.eval()
    for param in anchor.parameters():
        param.requires_grad_(False)
    learner_opt = torch.optim.AdamW(learner.parameters(), lr=cfg.lr, weight_decay=0.0)
    critic_opt = torch.optim.AdamW(critic.parameters(), lr=cfg.critic_lr, weight_decay=0.0)
    if resume and (run_dir / "optimizer.pth").exists():
        opt_state = torch.load(run_dir / "optimizer.pth", map_location="cpu")
        learner_opt.load_state_dict(opt_state["learner"])
        critic_opt.load_state_dict(opt_state["critic"])

    current_learner = run_dir / "learner" / f"{update}.pth"
    current_critic = run_dir / "critic" / f"{update}.pth"
    if not current_learner.exists():
        _save(learner, current_learner)
        _save(critic, current_critic)
    writer = SummaryWriter(str(Path("runs") / "alphazero_runs" / run_dir.name))
    history = (run_dir / "history.jsonl").open("a")

    rng = np.random.default_rng()
    league = None
    if cfg.population:
        league_kw = {"power": cfg.pfsp_power}
        league = (
            League.from_state(league_state, **league_kw)
            if league_state
            else League(load_population(cfg.population), **league_kw)
        )
        if not league.members:
            raise SystemExit(f"population {cfg.population} has no members")
        servers = {
            "learner": f"{current_learner}+{current_critic}",
            **{f"opp{i}": league.sample(rng) for i in range(cfg.opponent_slots)},
        }
    else:
        servers = {
            "learner": f"{current_learner}+{current_critic}",
            "sl": str(cfg.opponent_checkpoint or cfg.policy_checkpoint),
            "pool": pool[-1] if pool else str(cfg.policy_checkpoint),
        }
    slot_labels = {name: path for name, path in servers.items() if name.startswith("opp")}
    handles = {
        name: start_inference_server(
            num_workers=cfg.workers,
            model_factory=load_server_model,
            checkpoint_path=path,
            batch_size=cfg.inference_batch_size,
            autocast_device=None,
        )
        for name, path in servers.items()
    }
    pool_label = servers.get("pool")
    queues = {name: (h.request_q, h.reply_qs, h.ticket_q) for name, h in handles.items()}
    scores = _Scores(cfg.score_window)
    stop_file = run_dir / "STOP"

    def settings() -> dict:
        return {
            "positions_per_seat": cfg.positions_per_seat,
            "learner_seats": cfg.learner_seats,
            "sl_opponent_fraction": cfg.sl_opponent_fraction,
            "pool_ready": bool(pool),
            "pool_label": pool_label,
            "learner_temperature": cfg.learner_temperature,
            "opponent_temperature": cfg.opponent_temperature,
            "opponent_price_eps": cfg.opponent_price_eps,
            "price_eps": 0.0,  # forced moves (fixed prices only; open-price moves are decisions)
            "max_decisions": cfg.max_decisions,
            "start_positions": cfg.start_positions,
            "random_start_fraction": cfg.random_start_fraction,
            "midgame_positions": cfg.midgame_positions,
            "midgame_fraction": cfg.midgame_fraction,
            "gae_lambda": cfg.gae_lambda,
            "save_game_every": cfg.save_game_every,
            "league": (
                {
                    "slots": sorted(slot_labels),
                    "slot_labels": dict(slot_labels),
                    "learner_seat_weights": cfg.learner_seat_weights or DEFAULT_LEARNER_SEAT_WEIGHTS,
                }
                if league
                else None
            ),
        }

    buffer, gen, total_games = [], Counter(), 0
    gen_started = run_started = time.time()
    terminations, lengths, starts = Counter(), [], Counter()
    critic_first_values = []
    try:
        executor = ProcessPoolExecutor(max_workers=cfg.workers, initializer=worker_init, initargs=(queues,))
        try:
            futures = {
                executor.submit(play_pg_games, cfg.games_per_task, settings())
                for _ in range(cfg.workers * cfg.tasks_in_flight_per_worker)
            }
            while update < cfg.max_updates and not stop_file.exists():
                done, futures = wait_checking_servers(futures, handles, executor, timeout=60)
                for future in done:
                    result = future.result()
                    gen["decisions"] += result["decisions"]
                    for record in result["games"]:
                        gen["games"] += 1
                        scores.add(record)
                        if league:
                            for member, result in record.get("pairwise", []):
                                league.record(member, result)
                        terminations[record["termination"]] += 1
                        starts[record.get("start_kind")] += 1
                        lengths.append(record["decisions"])
                        for seat, value in record["first_values"].items():
                            critic_first_values.append((value, record["win_share"][int(seat)]))
                        buffer.extend(record["rows"])
                        if record.get("saved_game"):
                            opponent_checkpoint = (
                                servers.get("sl") if record["opponent"] == "sl"
                                else record["pool_label"] if record["opponent"] == "pool"
                                else "league"
                            )
                            try:
                                save_game(run_dir, record, update, str(current_learner), str(opponent_checkpoint), cfg)
                            except OSError as e:
                                LOGGER.warning(f"Could not save game {record['uid']}: {e}")
                    futures.add(executor.submit(play_pg_games, cfg.games_per_task, settings()))
                if len(buffer) < cfg.rows_per_update:
                    continue

                t0 = time.time()
                stats = _update(learner, critic, anchor, learner_opt, critic_opt, buffer, cfg, kl_coef, device, rng)
                train_seconds = time.time() - t0
                buffer = []
                update += 1

                previous = (current_learner, current_critic)
                current_learner = run_dir / "learner" / f"{update}.pth"
                current_critic = run_dir / "critic" / f"{update}.pth"
                _save(learner, current_learner)
                _save(critic, current_critic)
                handles["learner"].reload(f"{current_learner}+{current_critic}")
                if cfg.anchor_every and update % cfg.anchor_every == 0:
                    # The KL now measures change since this update: a fresh budget around
                    # a policy the run reached, rather than one fixed reference.
                    anchor.load_state_dict(learner.state_dict())
                    anchor_update = update
                snapshot = update % cfg.snapshot_every == 0
                for path in previous:
                    number = int(path.stem)
                    if number % cfg.snapshot_every != 0 and path.parent.parent == run_dir and path.exists():
                        path.unlink()
                if snapshot:
                    pool.append(str(current_learner))
                    if league and cfg.league_add_snapshots:
                        league.add(str(current_learner))
                    torch.save(
                        {"learner": learner_opt.state_dict(), "critic": critic_opt.state_dict()},
                        run_dir / "optimizer.pth",
                    )
                    (run_dir / "state.json").write_text(
                        json.dumps(
                            {
                                "config": asdict(cfg),
                                "snapshot_update": update,
                                "pool": pool,
                                "kl_coef": kl_coef,
                                "anchor_update": anchor_update,
                                "league": league.state() if league else None,
                            },
                            indent=2,
                        )
                    )
                if not league and pool and update % cfg.pool_refresh_every == 0:
                    pool_label = pool[int(rng.integers(0, len(pool)))]
                    handles["pool"].reload(pool_label)
                if league and update % cfg.slot_refresh_every == 0:
                    slot = f"opp{(update // cfg.slot_refresh_every) % cfg.opponent_slots}"
                    member = league.sample(rng)
                    handles[slot].reload(member)
                    slot_labels[slot] = member

                if cfg.kl_target is not None:
                    if stats["kl_anchor"] > 1.5 * cfg.kl_target:
                        kl_coef *= 1.5
                    elif stats["kl_anchor"] < cfg.kl_target / 1.5:
                        # Floored: far below the target the coefficient would otherwise
                        # decay toward 0 (pg3: 4e-6 by update 21) and take ~20 updates
                        # to matter again once the KL reached it.
                        kl_coef = max(kl_coef / 1.5, cfg.kl_coef_min)

                elapsed = time.time() - gen_started
                total_games += gen["games"]
                record = {
                    "update": update,
                    "time": datetime.now().isoformat(timespec="seconds"),
                    **stats,
                    **scores.summary(),
                    "kl_coef": kl_coef,
                    "anchor_update": anchor_update,
                    "games": gen["games"],
                    "games_total": total_games,
                    # Results arrive in bursts (a task returns 32 games at once), so the
                    # per-update rate is noisy; the run average is the throughput.
                    "games_per_hour": gen["games"] / elapsed * 3600,
                    "games_per_hour_run": total_games / (time.time() - run_started) * 3600,
                    "decisions_per_second": gen["decisions"] / elapsed,
                    "mean_decisions": float(np.mean(lengths)) if lengths else 0.0,
                    "terminations": dict(terminations),
                    "starts": dict(starts),
                    "train_seconds": train_seconds,
                    "pool_size": len(pool),
                    "pool_opponent": pool_label if pool else None,
                }
                if league:
                    hardest = sorted(league.members, key=league.win_rate)[:3]
                    record["league_size"] = len(league.members)
                    record["league_hardest"] = [[member, round(league.win_rate(member), 3)] for member in hardest]
                    record["league_slots"] = dict(slot_labels)
                if critic_first_values:
                    v, r = np.array(critic_first_values).T
                    record["critic_start_brier"] = float(np.mean((v - r) ** 2))
                gen.clear()
                gen_started = time.time()
                terminations, lengths, critic_first_values, starts = Counter(), [], [], Counter()
                history.write(json.dumps(record) + "\n")
                history.flush()
                for key, value in record.items():
                    if isinstance(value, (int, float)) and key not in ("update",):
                        writer.add_scalar(f"PG/{key}", value, update)
                # League mode scores the learner against the league (a multiple of a fair share).
                kind = "league" if league else "sl"
                LOGGER.info(
                    f"update {update}: vs {'league' if league else 'SL'} {record.get(f'score_vs_{kind}', float('nan')):.3f}"
                    f"±{record.get(f'score_vs_{kind}_se', float('nan')):.3f} "
                    f"(this update {record.get(f'score_vs_{kind}_update', float('nan')):.3f}), "
                    f"kl_anchor {stats['kl_anchor']:.4f}, entropy {stats['entropy']:.3f}, clip {stats['clip_frac']:.3f}, "
                    f"critic CE {stats['critic_ce']:.4f}, {record['games_per_hour_run']:.0f} games/h, "
                    f"train {train_seconds:.0f}s"
                )
        finally:
            # Queued tasks are dropped; running ones finish against the still-live servers.
            executor.shutdown(wait=True, cancel_futures=True)
    finally:
        history.close()
        writer.close()
        for handle in handles.values():
            handle.shutdown()
    return run_dir


def _update(learner, critic, anchor, learner_opt, critic_opt, rows, cfg: PGConfig, kl_coef, device, rng) -> dict:
    """One pass of PPO over ``rows`` (``cfg.ppo_epochs`` times) and one critic pass.

    The networks train in eval mode: the economic transformer's layers carry
    dropout (0.1), which in train mode makes the policy being updated a
    different one from the policy that played the games (eval mode, on the
    inference server) and puts noise into every PPO ratio."""
    learner.eval()
    critic.eval()
    advantages =np.array([r[5] for r in rows], dtype=np.float32)
    if cfg.normalize_advantages:
        scale = float(advantages.std()) or 1.0
        rows = [r[:5] + (r[5] / scale,) + r[6:] for r in rows]
    sums, count, first, steps = Counter(), 0, {}, 0
    critic_sums, critic_count = Counter(), 0
    micro = min(cfg.microbatch, cfg.minibatch)
    for epoch in range(cfg.ppo_epochs):
        order = rng.permutation(len(rows))
        for start in range(0, len(rows), cfg.minibatch):
            # One optimizer step per minibatch, its gradient accumulated over
            # microbatches (the policy head's activations bound the microbatch).
            step_rows = order[start : start + cfg.minibatch]
            learner_opt.zero_grad(set_to_none=True)
            critic_opt.zero_grad(set_to_none=True)
            for micro_start in range(0, len(step_rows), micro):
                batch = batch_tensors([rows[i] for i in step_rows[micro_start : micro_start + micro]], device)
                n = len(batch["choice"])
                share = n / len(step_rows)
                with torch.no_grad():
                    sl_logits, _, _, _ = anchor(batch["game_state"], batch["nodes"])
                    sl_price = anchor.last_price_components["price_logits"]
                policy_logits, _, _, _ = learner(batch["game_state"], batch["nodes"])
                price = learner.last_price_components["price_logits"]
                losses = pg_losses(
                    policy_logits, price, sl_logits, sl_price, batch, cfg.clip, kl_coef, cfg.entropy_coef,
                    temperature=cfg.learner_temperature,
                )
                if not torch.isfinite(losses["total"]):
                    raise RuntimeError(
                        f"Policy gradient diverged: non-finite loss ({ {k: float(v) for k, v in losses.items()} })"
                    )
                (losses["total"] * share).backward()
                if count == 0:
                    first = {"clip_frac_first": float(losses["clip_frac"]), "approx_kl_first": float(losses["approx_kl"])}
                count += n
                for key, value in losses.items():
                    if key != "total":
                        sums[key] += float(value) * n

                if epoch == 0:
                    _, win_loss_logits, score_pred, _ = critic(batch["game_state"], batch["nodes"], value_only=True)
                    c = critic_losses(win_loss_logits, score_pred, batch["fractions"])
                    loss = c["ce"] + cfg.score_loss_weight * c["mse"]
                    if not torch.isfinite(loss):
                        raise RuntimeError("Critic diverged: non-finite loss")
                    (loss * share).backward()
                    critic_count += n
                    critic_sums["ce"] += float(c["ce"].detach()) * n
                    critic_sums["mse"] += float(c["mse"].detach()) * n
                    critic_sums["winner_hit"] += float(c["winner_hit"]) * n
            norm = torch.nn.utils.clip_grad_norm_(learner.parameters(), max_norm=1.0)
            if not torch.isfinite(norm):
                raise RuntimeError("Policy gradient diverged: non-finite gradient norm")
            learner_opt.step()
            sums["grad_norm"] += float(norm) * len(step_rows)
            steps += 1
            if epoch == 0:
                norm = torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1.0)
                if not torch.isfinite(norm):
                    raise RuntimeError("Critic diverged: non-finite gradient norm")
                critic_opt.step()
    stats = {key: value / max(count, 1) for key, value in sums.items()}
    stats.update(first)
    stats.update({f"critic_{key}": value / max(critic_count, 1) for key, value in critic_sums.items()})
    stats.update(
        rows=len(rows),
        optimizer_steps=steps,
        adv_mean=float(advantages.mean()),
        adv_std=float(advantages.std()),
        adv_abs_mean=float(np.abs(advantages).mean()),
        price_rows=sum(1 for r in rows if r[4] is not None),
    )
    return stats
