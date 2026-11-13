"""Policy-only head-to-head between two title-generic checkpoints, for any title with an RL layout (RL bridge
milestone C2; ``docs/rl_model.md`` "Policy gradient for the generic model").

``scripts/eval_policy_only.py`` plays 1830 only (the legacy encoder routes moves, the legacy start code makes
the starts) but can seat legacy checkpoints; this one plays generic checkpoints only, on any title: starts from
``generic_play.replayed_start`` (seeded by the start index, so every seat arrangement of a start shares it; the
opening with ``--auction-start``), legal moves from ``generic_play.title_play``, moves and prices from each
side's network exactly as the generic policy-gradient workers draw them (temperature in log space, the price
cell from the head over the legal range, a price inside it). Each side runs in the pool workers (CPU) or, with
``--server``, on a generic inference server (the GPU when there is one). Side A holds ``--a-seats`` of
``--num-players`` seats in every arrangement; equal strength scores ``a_seats / players``.

    uv run python scripts/eval_generic.py --title 1830 --match <a.pth>@0.5 <b.pth>@0.5 --num-players 4 \\
        --a-seats 1 --games 400 --auction-start --out logs/eval/<name>
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def parse_player(spec: str) -> tuple:
    """``(checkpoint path, temperature)`` of ``<checkpoint>[@<temperature>]`` (a ``generic:`` prefix is allowed)."""
    from rl18xx.agent.alphazero.generic_policy import GENERIC_PREFIX

    if spec.startswith(GENERIC_PREFIX):
        spec = spec[len(GENERIC_PREFIX) :]
    path, _, temperature = spec.partition("@")
    if not Path(path).is_file():
        raise SystemExit(f"checkpoint {path!r} not found")
    return str(Path(path)), float(temperature) if temperature else 1.0


_LOCAL: dict = {}


def _policy(path: str):
    """A side's in-process server (``LocalGenericClient`` on the checkpoint, CPU), one per process."""
    from rl18xx.agent.alphazero import model_generic as mg
    from rl18xx.agent.alphazero.generic_inference import LocalGenericClient, LocalGenericServer

    if path not in _LOCAL:
        model, _ = mg.load_checkpoint(path, "cpu")
        _LOCAL[path] = LocalGenericClient(LocalGenericServer(model.eval()))
    return _LOCAL[path]


def play_games(game_indices: list, settings: dict) -> list:
    """Pool task: play the given games concurrently; one record per game."""
    import torch

    from rl18xx.agent.alphazero import generic_pg as gpg
    from rl18xx.agent.alphazero import model_generic as mg
    from rl18xx.agent.alphazero.generic_inference import slice_positions
    from rl18xx.agent.alphazero.generic_play import replayed_start, title_play
    from rl18xx.agent.alphazero.generic_tokens import encode_batch

    torch.set_num_threads(1)
    rng = np.random.default_rng()
    title = settings["title"]
    play = title_play(title)
    seatings = [tuple(a) for a in settings["arrangements"]]
    n = len(seatings[0])
    games = []
    for idx in game_indices:
        start_rng = random.Random(f"{settings['seed']}:{idx // len(seatings)}")
        start, raw, _ = replayed_start(
            title, n, settings["start_positions"], settings["random_start_fraction"], rng=start_rng,
            auction_fraction=1.0 if settings["auction_start"] else 0.0, auction_unlock=False,
        )  # fmt: skip
        seats = list(seatings[idx % len(seatings)])
        g = gpg._Game(raw=raw, uid=str(idx), opponent="", seat_models=seats, start=start.label)
        g.seed, g.idx = start.seed, idx
        games.append(g)

    def client(side):
        if settings["server"]:
            return gpg._CLIENTS[side]
        return _policy(settings["paths"][side])

    finished, active = [], list(games)
    while active:
        pending = []
        for g in active:
            legal = gpg._advance(g, play, settings)
            if legal is None:
                finished.append(g)
            else:
                pending.append((g, legal))
        active = [g for g, _ in pending]
        if not pending:
            break
        enc = encode_batch([g.raw for g, _ in pending])
        rotation = enc["meta"][:, 5]
        by_side = defaultdict(list)
        for i, (g, _) in enumerate(pending):
            by_side[g.seat_models[int(rotation[i])]].append(i)
        sent = {}
        for side, rows in by_side.items():
            arrays = mg.add_legal_inputs(slice_positions(enc, rows), [pending[i][1] for i in rows])
            sent[side] = (client(side).send_inputs(arrays, price_eps=settings["price_eps"]), rows, arrays)
        for side, (request, rows, arrays) in sent.items():
            reply = client(side).receive_outputs(request)
            offsets = arrays["legal_offsets"]
            price_of_row = {int(r): j for j, r in enumerate(arrays["price_rows"])}
            for j, i in enumerate(rows):
                g, legal = pending[i]
                lo, hi = int(offsets[j]), int(offsets[j + 1])
                p = np.exp(gpg.tempered_logp(reply.logp[lo:hi], settings["temperatures"][side]))
                k = int(rng.choice(len(p), p=p / p.sum()))
                if play.open_price(legal, k):
                    cell = int(reply.price_cell[price_of_row[lo + k]])
                    kind = int(legal.rows[k, mg.C_PRICE_KIND])
                    plo, phi, step = (int(x) for x in legal.price_range[k])
                    price = int(mg.price_cells_for(kind, plo, phi, step).sample_price(cell, rng))
                else:
                    price = play.fixed_price(g.raw, legal, k)
                play.apply(g.raw, legal, k, price)
                g.decisions += 1

    records = []
    for g in finished:
        worth, win_share, fractions = gpg.game_outcome(g.raw)
        record = {
            "idx": g.idx,
            "seats": list(g.seat_models),
            "win_share": win_share.tolist(),
            "worth_share": fractions.tolist(),
            "decisions": g.decisions,
            "termination": g.termination,
            "start": g.start,
        }
        if g.idx < settings.get("save_games", 0):
            record["net_worth"] = worth.tolist()
            record["seed"] = g.seed
            record["players"] = [[int(p.id), str(p.name)] for p in g.raw.players]
            record["raw_actions"] = [dict(a) for a in g.raw.raw_actions]
        records.append(record)
    return records


def run_match(a_spec: str, b_spec: str, args) -> tuple:
    from eval_policy_only import arrangements, summarize

    from rl18xx.agent.alphazero import generic_pg as gpg
    from rl18xx.agent.alphazero.generic_inference import start_generic_inference_server
    from rl18xx.agent.alphazero.inference_server import wait_checking_servers

    players = {"A": parse_player(a_spec), "B": parse_player(b_spec)}
    for path, _ in players.values():
        gpg.check_generic_checkpoint(path, args.title)
    handles = {}
    if args.server:
        handles = {
            side: start_generic_inference_server(num_workers=args.workers, checkpoint_path=path, batch_size=1024)
            for side, (path, _) in players.items()
        }
    settings = {
        "title": args.title,
        "paths": {side: path for side, (path, _) in players.items()},
        "temperatures": {side: t for side, (_, t) in players.items()},
        "server": bool(args.server),
        "seed": args.seed,
        "start_positions": args.start_positions,
        "random_start_fraction": args.random_start_fraction,
        "auction_start": args.auction_start,
        "max_decisions": args.max_decisions,
        "auction_stall_decisions": args.auction_stall_decisions,
        "price_eps": args.price_eps,
        "save_games": args.save_games,
        "arrangements": arrangements(args.a_seats, args.num_players),
    }
    seatings = settings["arrangements"]
    num_games = math.ceil(args.games / len(seatings)) * len(seatings)
    chunks = [list(range(i, min(i + args.games_per_task, num_games))) for i in range(0, num_games, args.games_per_task)]
    queues = {side: (h.request_q, h.reply_qs, h.ticket_q) for side, h in handles.items()}
    records, started = [], time.time()
    try:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=gpg.worker_init, initargs=(queues,)) as pool:
            futures = {pool.submit(play_games, chunk, settings) for chunk in chunks}
            while futures:
                done, futures = wait_checking_servers(futures, handles, pool)
                for future in done:
                    records.extend(future.result())
    finally:
        for handle in handles.values():
            handle.shutdown()
    summary = summarize(a_spec, b_spec, records)
    summary["title"] = args.title
    summary["seconds"] = round(time.time() - started, 1)
    return summary, records


def save_game_logs(out: Path, match_index: int, a: str, b: str, title: str, records: list) -> None:
    from rl18xx.agent.alphazero.game_records import GAME_RECORD_VERSION, save_game_record

    labels = {"A": a, "B": b}
    for r in sorted(records, key=lambda r: r["idx"]):
        actions = r.pop("raw_actions", None)
        if actions is None:
            continue
        name = f"m{match_index}_{r['idx']}"
        r["seat_labels"] = [labels[side] for side in r["seats"]]
        game = {
            "id": name,
            "title": title,
            "description": f"eval_generic {a} vs {b}, game {r['idx']}",
            "players": [{"id": pid, "name": pname} for pid, pname in r.pop("players")],
            "settings": {"optional_rules": [], "seed": r.get("seed") or 0},
            "actions": actions,
            "rl18xx": {"version": GAME_RECORD_VERSION, "kind": "eval_generic", "match": f"{a} vs {b}", **r},
        }
        r["game_file"] = save_game_record(out, name, game)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--title", default="1830")
    parser.add_argument("--match", nargs=2, action="append", metavar=("A", "B"), required=True)
    parser.add_argument("--games", type=int, default=400)
    parser.add_argument("--num-players", type=int, default=4)
    parser.add_argument("--a-seats", type=int, default=None, help="Side A's seats (default half the table)")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--games-per-task", type=int, default=30)
    parser.add_argument("--max-decisions", type=int, default=1500)
    parser.add_argument("--price-eps", type=float, default=0.0)
    parser.add_argument("--start-positions", default=None, help="SR1 start file (default: the title's for the count)")
    parser.add_argument("--random-start-fraction", type=float, default=0.2)
    parser.add_argument("--auction-start", action="store_true", help="Every game from the opening (real rules)")
    parser.add_argument("--auction-stall-decisions", type=int, default=150)
    parser.add_argument("--server", action="store_true", help="Each side on a generic inference server")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    parser.add_argument("--save-games", type=int, default=0)
    args = parser.parse_args()
    from rl18xx.agent.alphazero.titles import title_spec

    spec = title_spec(args.title)
    spec.check_players(args.num_players)
    if args.a_seats is None:
        args.a_seats = args.num_players // 2
    if args.save_games and not args.out:
        parser.error("--save-games needs --out")
    if args.start_positions is None and not args.auction_start:
        args.start_positions = spec.start_positions(args.num_players)
    import multiprocessing

    multiprocessing.set_start_method("spawn", force=True)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    summaries, all_records = [], []
    for match_index, (a, b) in enumerate(args.match):
        summary, records = run_match(a, b, args)
        summaries.append(summary)
        if args.save_games:
            save_game_logs(Path(args.out), match_index, a, b, args.title, records)
        all_records.extend({"match": f"{a} vs {b}", **r} for r in records)
        print(
            f"[{args.title} {summary['players']}p {summary['a_seats']}v{summary['players'] - summary['a_seats']}] "
            f"{a} vs {b}: {summary['A_score']:.3f} ± {summary['A_score_se']:.3f} over {summary['games']} games, "
            f"equal = {summary['A_score_equal']:.2f} (worth share {summary['A_worth_share']:.3f}, "
            f"{summary['seconds']:.0f}s)",
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
