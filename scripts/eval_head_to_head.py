"""Head-to-head strength evaluation between checkpoints.

Plays 4-player 1830 games (Rust engine + Rust MCTS) in which two checkpoints
each hold two seats, cycling through all six 2-2 seat arrangements so seat
order and priority deal even out. Each checkpoint gets its own inference
server; every worker holds a client for each server, and a seat searches
with its own checkpoint's tree (rebuilt from the current position whenever
the other checkpoint moved in between, kept across its own consecutive moves).

A game scores share-of-winners per seat (1/k for each of k tied leaders on
net worth), so a side's expected score is 0.5 when the checkpoints are equally
strong. Games use the self-play ``auction_unlock`` variant both checkpoints
were trained under; moves are sampled from visit counts before
``--softpick`` engine moves and argmax after, with no Dirichlet noise and no
resignation. A game still in the private auction after ``--stall`` engine
moves, or still running at ``--max-length``, is scored on net worth.

    uv run python scripts/eval_head_to_head.py --match 107 7 --games 240
    uv run python scripts/eval_head_to_head.py --match 107 7 --match 57 7 --games 120

A player is a checkpoint -- ``<num>`` (in the current_best session),
``<session>/<num>``, or a path to a ``.pth`` -- optionally with its own search
size, ``<checkpoint>@<readouts>`` (default ``--readouts``), and its own PUCT
constant, ``<checkpoint>@<readouts>/<c_puct_init>`` (default SelfPlayConfig's):

    uv run python scripts/eval_head_to_head.py --match 7@200 7@64 --games 120
    uv run python scripts/eval_head_to_head.py --match 7@64/0.6 7@64 --games 120
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
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

LOGGER = logging.getLogger("eval_head_to_head")
CHECKPOINT_DIR = REPO_ROOT / "model_checkpoints"
NUM_PLAYERS = 4
# Every way to give each of two checkpoints two of the four seats.
ARRANGEMENTS = [
    tuple("A" if seat in pair else "B" for seat in range(NUM_PLAYERS))
    for pair in itertools.combinations(range(NUM_PLAYERS), 2)
]


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
    """``(checkpoint path, readouts, c_puct_init or None)`` of a
    ``<checkpoint>[@<readouts>[/<c_puct_init>]]`` player spec."""
    checkpoint, _, search = spec.partition("@")
    readouts, _, c_puct = search.partition("/")
    return (
        resolve_checkpoint(checkpoint),
        int(readouts) if readouts else default_readouts,
        float(c_puct) if c_puct else None,
    )


def model_factory(checkpoint_path):
    """Inference-server model loader for one fixed checkpoint (module-level so it pickles)."""
    from rl18xx.agent.alphazero.checkpointer import _load_model_from_session_checkpoint

    path = Path(checkpoint_path)
    model = _load_model_from_session_checkpoint(path.parent, path)
    model.eval()
    return model


_CLIENTS: dict = {}


def worker_init(server_queues: dict):
    """Pool initializer: take one slot on every checkpoint's inference server."""
    from rl18xx.agent.alphazero.inference_server import InferenceClient

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


def play_game(game_idx: int, seat_names: tuple, settings: dict) -> dict:
    """Play one game with ``seat_names[i]`` (a player spec) searching for seat ``i``."""
    from engine_rs import BaseGame as RustBaseGame

    from rl18xx.agent.alphazero.config import SelfPlayConfig
    from rl18xx.agent.alphazero.rust_mcts_player import RustMCTSPlayer
    from rl18xx.agent.alphazero.self_play import _auction_unlock_discounts
    from rl18xx.rust_adapter import RustGameAdapter

    seed = settings["seed"] * 100_003 + game_idx
    random.seed(seed)
    np.random.seed(seed % (2**32))
    start = time.time()

    rust_game = RustBaseGame({i + 1: f"Player {i + 1}" for i in range(NUM_PLAYERS)})
    rust_game.set_auction_unlock(True)
    game = RustGameAdapter(rust_game)

    players = {}
    for name in sorted(set(seat_names)):
        spec = settings["players"][name]
        config = SelfPlayConfig(
            network=None,
            use_inference_server=True,
            inference_client=_CLIENTS[spec["server"]],
            num_readouts=spec["readouts"],
            **({"c_puct_init": spec["c_puct"]} if spec["c_puct"] is not None else {}),
            softpick_move_cutoff=settings["softpick"],
            dirichlet_noise_weight=0.0,
            enable_resign=False,
            noresign_holdout_rate=0.0,
            max_game_length=settings["max_length"],
            player_count_distribution={NUM_PLAYERS: 1.0},
            auction_unlock=True,
        )
        players[name] = RustMCTSPlayer(config)

    seat_by_player_id = {player.id: seat for seat, player in enumerate(game.players)}
    last_mover = None
    decisions = Counter()
    termination = None
    while True:
        if game.finished:
            termination = "finished"
            break
        if game.move_number >= settings["max_length"]:
            game.end_game()
            termination = "max_length"
            break
        round_name = game.round.__class__.__name__
        if settings["stall"] and "Auction" in round_name and game.move_number >= settings["stall"]:
            termination = "auction_stall"
            break

        seat = seat_by_player_id[game.active_players()[0].id]
        name = seat_names[seat]
        player = players[name]
        if last_mover != name:
            player.initialize_game(game.pickle_clone())
            _expand_root(player)
        move = player.suggest_move()
        player.play_move(move)
        game = player.get_game_state()
        last_mover = name
        decisions["auction" if "Auction" in round_name else round_name.lower()] += 1

    net_worth = {seat_by_player_id[pid]: float(v) for pid, v in game.result().items()}
    best = max(net_worth.values())
    winners = [seat for seat, v in net_worth.items() if v >= best - 1e-6]
    total = sum(max(v, 0.0) for v in net_worth.values()) or 1.0
    return {
        "game_idx": game_idx,
        "seats": list(seat_names),
        "termination": termination,
        "engine_moves": int(game.move_number),
        "decisions": dict(decisions),
        "net_worth": [net_worth[s] for s in range(NUM_PLAYERS)],
        "worth_share": [max(net_worth[s], 0.0) / total for s in range(NUM_PLAYERS)],
        "win_share": [(1.0 / len(winners) if s in winners else 0.0) for s in range(NUM_PLAYERS)],
        "auction_unlock_discounts": _auction_unlock_discounts(game),
        "seconds": round(time.time() - start, 1),
        "raw_actions": list(game.raw_actions),
    }


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
    parser.add_argument("--softpick", type=int, default=30, help="Sample from visit counts before this engine move")
    parser.add_argument("--max-length", type=int, default=1000)
    parser.add_argument("--stall", type=int, default=400, help="End a game still in the private auction here")
    parser.add_argument("--workers", type=int, default=48)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
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
            path, readouts, c_puct = parse_player(spec, args.readouts)
            players[spec] = {"server": str(path), "readouts": readouts, "c_puct": c_puct}
    paths = sorted({p["server"] for p in players.values()})
    games_per_match = math.ceil(args.games / len(ARRANGEMENTS)) * len(ARRANGEMENTS)
    settings = {
        "readouts": args.readouts,
        "softpick": args.softpick,
        "max_length": args.max_length,
        "stall": args.stall,
        "seed": args.seed,
        "players": players,
    }
    (out_dir / "settings.json").write_text(
        json.dumps({**settings, "matches": args.match, "games_per_match": games_per_match}, indent=2)
    )

    from rl18xx.agent.alphazero.inference_server import start_inference_server

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
        )
    server_queues = {path: (h.request_q, h.reply_qs, h.ticket_q) for path, h in servers.items()}

    jobs = []
    for match_index, (a, b) in enumerate(args.match):
        for g in range(games_per_match):
            arrangement = ARRANGEMENTS[g % len(ARRANGEMENTS)]
            seat_names = tuple(a if side == "A" else b for side in arrangement)
            jobs.append((match_index, arrangement, match_index * 1_000_000 + g, seat_names))

    results = defaultdict(list)
    results_path = out_dir / "games.jsonl"
    started = time.time()
    try:
        with ProcessPoolExecutor(
            max_workers=args.workers, initializer=worker_init, initargs=(server_queues,)
        ) as pool, open(results_path, "a") as results_file:
            pending = {
                pool.submit(play_game, game_idx, seat_names, settings): (match_index, arrangement)
                for match_index, arrangement, game_idx, seat_names in jobs
            }
            done_count = 0
            while pending:
                finished, _ = wait(list(pending), return_when=FIRST_COMPLETED)
                for future in finished:
                    match_index, arrangement = pending.pop(future)
                    try:
                        record = future.result()
                    except Exception as e:
                        LOGGER.error(f"Game crashed: {e}", exc_info=True)
                        continue
                    record["match"] = match_index
                    record["seats"] = list(arrangement)
                    raw_actions = record.pop("raw_actions")
                    (out_dir / "games").mkdir(exist_ok=True)
                    (out_dir / "games" / f"{record['game_idx']}.json").write_text(json.dumps(raw_actions))
                    results_file.write(json.dumps(record) + "\n")
                    results_file.flush()
                    results[match_index].append(record)
                    done_count += 1
                    if done_count % 12 == 0 or not pending:
                        elapsed = time.time() - started
                        lines = []
                        for mi, (a, b) in enumerate(args.match):
                            if results[mi]:
                                s = summarize(f"{a} vs {b}", {"A": a, "B": b}, results[mi])
                                lines.append(f"{a} vs {b}: {s['games']} games, {a} score {s['A_score']:.3f} "
                                             f"± {s['A_score_se']:.3f}")
                        LOGGER.info(f"[{done_count}/{len(jobs)} games, {elapsed / 60:.1f} min] " + "; ".join(lines))
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
