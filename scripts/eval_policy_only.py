"""Search-free head-to-head between two policies, each at its own temperature.

Like ``eval_head_to_head.py`` -- 4-player games from self-play's first-Stock-
Round starts, each start played in all six 2-2 seat arrangements, scored by
share of winners (0.5 = equal) -- but every seat samples its move straight from
its policy (and prices from its price head), so thousands of games take
minutes. A player is a checkpoint spec as in ``eval_head_to_head.py``
optionally followed by ``@<temperature>`` (default 1), which applies to the
move; prices are drawn from the price head as is (as the policy-gradient
learner draws them), mixed with ``--price-eps`` uniform. Lower temperature
sharpens a policy, which by itself beats the same policy at temperature 1, so
compare a trained policy against its starting point at several temperatures to
tell learning from sharpening.

    uv run python scripts/eval_policy_only.py --match model_checkpoints_pg/<run>/learner/20.pth 10 \\
        --match 10@0.5 10 --games 1200

``--save-games N`` (with ``--out``) also keeps the action logs of each match's
first N games (game indices 0..N-1: every seat arrangement of the first N/6
starts) as ``<out>/games/m<match>_<idx>.json`` (``game_records``), each named
by its ``games.jsonl`` record (``game_file``), to browse in the dashboard's
game viewer (``/games``) or with ``main.py replay``. Off by default.
"""

import argparse
import itertools
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

NUM_PLAYERS = 4


def arrangements(a_seats: int) -> list:
    """Every way to give side A ``a_seats`` of the four seats (B the rest)."""
    return [
        tuple("A" if seat in chosen else "B" for seat in range(NUM_PLAYERS))
        for chosen in itertools.combinations(range(NUM_PLAYERS), a_seats)
    ]


ARRANGEMENTS = arrangements(2)


def parse_player(spec: str) -> tuple:
    """``(checkpoint path(s), temperature)`` of ``<checkpoint>[+<value>][@<temperature>]``."""
    from eval_head_to_head import resolve_checkpoint

    checkpoint, _, temperature = spec.partition("@")
    paths = "+".join(str(resolve_checkpoint(part)) for part in checkpoint.split("+"))
    return paths, float(temperature) if temperature else 1.0


def play_games(game_indices: list, settings: dict) -> list:
    """Pool task: play the given games concurrently; returns one record per game."""
    from rl18xx.agent.alphazero import policy_gradient as pg
    from rl18xx.agent.alphazero.mcts import _rust_encode
    from rl18xx.agent.alphazero.policy_selfplay import _advance
    from rl18xx.agent.alphazero.start_positions import apply_actions, new_game, sample_start_position

    rng = np.random.default_rng()
    seatings = [tuple(a) for a in settings.get("arrangements", ARRANGEMENTS)]
    games = []
    for idx in game_indices:
        # Seeded by the start index, so all arrangements of a start share it.
        start_rng = random.Random(f"{settings['seed']}:{idx // len(seatings)}")
        start = sample_start_position(
            NUM_PLAYERS, settings["start_positions"], settings["random_start_fraction"], rng=start_rng
        )
        game = apply_actions(new_game(NUM_PLAYERS), start.actions)
        games.append(
            SimpleNamespace(
                game=game, uid=str(idx), idx=idx, seats=seatings[idx % len(seatings)],
                decisions=0, price_row=None, termination=None, start=start.label,
            )
        )

    finished, active = [], list(games)
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
        by_side = defaultdict(list)
        for i, (g, _) in enumerate(pending):
            by_side[g.seats[int(encoded[i][6])]].append(i)
        sent = {
            side: pg._CLIENTS[settings["servers"][side]].send(
                [encoded[i] for i in rows], legal_indices=[pending[i][1] for i in rows]
            )
            for side, rows in by_side.items()
        }
        for side, rows in by_side.items():
            client = pg._CLIENTS[settings["servers"][side]]
            probs, _, _ = client.receive(sent[side])
            price = client.last_price_components
            price_logits = price["price_logits"].numpy() if price is not None else None
            temperature = settings["temperatures"][side]
            for j, i in enumerate(rows):
                g, legal = pending[i]
                p = probs[j].numpy()[legal].astype(np.float64)
                if temperature != 1.0:
                    p = np.power(np.clip(p, 1e-12, None), 1.0 / temperature)
                p = p / p.sum() if p.sum() > 0 else np.full(len(legal), 1.0 / len(legal))
                choice = legal[int(rng.choice(len(legal), p=p))]
                g.price_row = price_logits[j] if price_logits is not None else None
                price_value, _, _ = pg.choose_price(g.game, choice, g.price_row, rng, settings["price_eps"])
                g.game._game.apply_action_index(choice, price_value)
                g.decisions += 1

    records = []
    for g in finished:
        win_share, fractions = pg.outcome(g.game)
        record = {
            "idx": g.idx,
            "seats": list(g.seats),
            "win_share": win_share.tolist(),
            "worth_share": fractions.tolist(),
            "decisions": g.decisions,
            "termination": g.termination,
        }
        if g.idx < settings.get("save_games", 0):
            # --save-games: the action log, and what the game viewer shows with it.
            from rl18xx.agent.alphazero.self_play import _compute_net_worth

            net_worth = _compute_net_worth(g.game)
            record["start"] = g.start
            record["net_worth"] = [float(net_worth[pid]) for pid in sorted(net_worth)]
            record["raw_actions"] = list(g.game.raw_actions)
        records.append(record)
    return records


def summarize(a: str, b: str, records: list) -> dict:
    scores = [sum(r["win_share"][s] for s, side in enumerate(r["seats"]) if side == "A") for r in records]
    worth = [sum(r["worth_share"][s] for s, side in enumerate(r["seats"]) if side == "A") for r in records]
    by_arrangement = defaultdict(list)
    for r, score in zip(records, scores):
        by_arrangement["".join(r["seats"])].append(score)
    n = len(scores)
    a_seats = records[0]["seats"].count("A") if records else 2
    return {
        "A": a,
        "B": b,
        "games": n,
        "a_seats": a_seats,
        # Side A's summed win share; equally strong sides score a_seats / 4.
        "A_score": float(np.mean(scores)),
        "A_score_equal": a_seats / NUM_PLAYERS,
        "A_score_se": float(np.std(scores, ddof=1) / math.sqrt(n)) if n > 1 else float("nan"),
        "A_worth_share": float(np.mean(worth)),
        "by_arrangement": {k: round(float(np.mean(v)), 3) for k, v in sorted(by_arrangement.items())},
        "terminations": dict(Counter(r["termination"] for r in records)),
        "mean_decisions": float(np.mean([r["decisions"] for r in records])),
    }


def run_match(a_spec: str, b_spec: str, args) -> dict:
    from rl18xx.agent.alphazero import policy_gradient as pg
    from rl18xx.agent.alphazero.inference_server import start_inference_server

    players = {"A": parse_player(a_spec), "B": parse_player(b_spec)}
    paths = sorted({path for path, _ in players.values()})
    handles = {
        path: start_inference_server(
            num_workers=args.workers,
            model_factory=pg.load_server_model,
            checkpoint_path=path,
            batch_size=512,
            autocast_device=None,
        )
        for path in paths
    }
    settings = {
        "servers": {side: path for side, (path, _) in players.items()},
        "temperatures": {side: t for side, (_, t) in players.items()},
        "seed": args.seed,
        "start_positions": args.start_positions,
        "random_start_fraction": args.random_start_fraction,
        "max_decisions": args.max_decisions,
        "price_eps": args.price_eps,
        "save_games": getattr(args, "save_games", 0),
        "arrangements": arrangements(getattr(args, "a_seats", 2)),
    }
    seatings = settings["arrangements"]
    num_games = math.ceil(args.games / len(seatings)) * len(seatings)
    chunks = [list(range(i, min(i + args.games_per_task, num_games))) for i in range(0, num_games, args.games_per_task)]
    queues = {path: (h.request_q, h.reply_qs, h.ticket_q) for path, h in handles.items()}
    records, started = [], time.time()
    try:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=pg.worker_init, initargs=(queues,)) as pool:
            futures = {pool.submit(play_games, chunk, settings) for chunk in chunks}
            while futures:
                done, futures = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    records.extend(future.result())
    finally:
        for handle in handles.values():
            handle.shutdown()
    summary = summarize(a_spec, b_spec, records)
    summary["seconds"] = round(time.time() - started, 1)
    return summary, records


def save_game_logs(out: Path, match_index: int, a: str, b: str, records: list) -> None:
    """Write the records' action logs (``raw_actions``, present for ``--save-games``
    games) to ``<out>/games/m<match>_<idx>.json`` and name each file in its record."""
    from rl18xx.agent.alphazero.game_records import make_game_record, save_game_record

    labels = {"A": a, "B": b}
    for r in sorted(records, key=lambda r: r["idx"]):
        actions = r.pop("raw_actions", None)
        if actions is None:
            continue
        name = f"m{match_index}_{r['idx']}"
        r["seat_labels"] = [labels[side] for side in r["seats"]]
        meta = {
            "name": name,
            "kind": "policy_only",
            "match": f"{a} vs {b}",
            "description": f"eval_policy_only {a} vs {b}, game {r['idx']}",
            "sides": list(r["seats"]),
            **{k: v for k, v in r.items() if k != "seats"},
        }
        r["game_file"] = save_game_record(out, name, make_game_record(actions, NUM_PLAYERS, meta=meta))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--match", nargs=2, action="append", metavar=("A", "B"), required=True)
    parser.add_argument(
        "--games", type=int, default=1200, help="Games per match (rounded up to a multiple of the arrangements)"
    )
    parser.add_argument(
        "--a-seats", type=int, default=2, choices=(1, 2, 3),
        help="Seats side A holds (2: 2v2 in six arrangements; 1 / 3: 1v3 / 3v1 in four); equal strength scores a_seats/4",
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--games-per-task", type=int, default=30)
    parser.add_argument("--max-decisions", type=int, default=1000)
    parser.add_argument("--price-eps", type=float, default=0.0, help="Uniform mixing into both sides' price draws")
    parser.add_argument("--start-positions", type=str, default="human_games/start_positions_1830_4p.jsonl")
    parser.add_argument("--random-start-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default=None, help="Directory for summary.json and games.jsonl")
    parser.add_argument(
        "--save-games", type=int, default=0,
        help="Keep the action logs of each match's first N games in <out>/games/ for the game viewer (needs --out)",
    )
    args = parser.parse_args()
    if args.save_games and not args.out:
        parser.error("--save-games needs --out")

    import multiprocessing

    multiprocessing.set_start_method("spawn", force=True)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    summaries, all_records = [], []
    for match_index, (a, b) in enumerate(args.match):
        summary, records = run_match(a, b, args)
        summaries.append(summary)
        if args.save_games:
            save_game_logs(Path(args.out), match_index, a, b, records)
        all_records.extend({"match": f"{a} vs {b}", **r} for r in records)
        print(
            f"{a} vs {b}: {summary['A_score']:.3f} ± {summary['A_score_se']:.3f} over {summary['games']} games "
            + (f"[{summary['a_seats']}v{NUM_PLAYERS - summary['a_seats']}, equal = {summary['A_score_equal']:.2f}] " if summary["a_seats"] != 2 else "")
            + f"(worth share {summary['A_worth_share']:.3f}, {summary['seconds']:.0f}s)",
            flush=True,
        )
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary.json").write_text(json.dumps(summaries, indent=2))
        with (out / "games.jsonl").open("w") as f:
            for r in all_records:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
