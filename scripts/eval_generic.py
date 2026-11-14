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

Optional rules (1822's 1822+ / tax-haven combinations, ``titles.TitleSpec.rules_mix``): ``--rules <name>`` plays
every game under one combination (``base``, ``plus``, ``plus+nr+ss+th``, ...), ``--rules-mix <name>:<weight>,...``
draws each start's combination (seeded by the start index, like the start: every seat arrangement of a start plays
the same rules); the default is the title's mix (1822: the human games' combinations; other titles: base rules).
Records carry the combination (``rules``); the summary splits decisions and scores by it (``by_rules``).

All-pass openings (the 1822 family): a game whose opening Stock Round ends with nothing bought, sold or bid ends by
Ruby's rule with every seat tied (``generic_play.all_pass_opening``); it is not scored but replayed from a fresh
draw of its start index (``<seed>:<start>:r<k>``; at most ``generic_play.MAX_ALL_PASS_REDRAWS``, then it counts
with termination ``all_pass``), so ties do not dilute the score. The summary counts them (``all_pass_redrawn``);
``--keep-all-pass`` scores them as they are.

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
    from rl18xx.agent.alphazero.generic_play import ALL_PASS, MAX_ALL_PASS_REDRAWS, all_pass_opening, replayed_start
    from rl18xx.agent.alphazero.generic_play import title_play
    from rl18xx.agent.alphazero.generic_tokens import encode_batch
    from rl18xx.agent.alphazero.titles import draw_rules, rules_name

    torch.set_num_threads(1)
    rng = np.random.default_rng()
    title = settings["title"]
    play = title_play(title)
    seatings = [tuple(a) for a in settings["arrangements"]]
    n = len(seatings[0])

    def new_game(idx: int, redraws: int = 0):
        # the start index seeds the start (every arrangement of a start shares it); a redrawn all-pass opening
        # takes the k-th redraw of that start index
        key = f"{settings['seed']}:{idx // len(seatings)}" + (f":r{redraws}" if redraws else "")
        start_rng = random.Random(key)
        rules = draw_rules(settings.get("rules_mix") or {"base": 1.0}, start_rng, title)
        start, raw, _ = replayed_start(
            title, n, settings["start_positions"], settings["random_start_fraction"], rng=start_rng,
            auction_fraction=1.0 if settings["auction_start"] else 0.0, auction_unlock=False, optional_rules=rules,
        )  # fmt: skip
        seats = list(seatings[idx % len(seatings)])
        g = gpg._Game(raw=raw, uid=str(idx), opponent="", seat_models=seats, start=start.label)
        g.seed, g.idx = start.seed, idx
        g.rules = rules_name(title, start.optional_rules)
        g.seating = [int(p.id) for p in raw.players]  # (the 1822 family reorders raw.players every Stock Round)
        g.redraws = redraws
        return g

    games = [new_game(idx) for idx in game_indices]
    redraw_all_pass = settings.get("redraw_all_pass", True)

    def client(side):
        if settings["server"]:
            return gpg._CLIENTS[side]
        return _policy(settings["paths"][side])

    finished, active = [], list(games)
    while active:
        pending = []
        queue = list(active)
        while queue:
            g = queue.pop(0)
            legal = gpg._advance(g, play, settings)
            if legal is not None:
                pending.append((g, legal))
            elif redraw_all_pass and all_pass_opening(title, g.raw):
                if g.redraws < MAX_ALL_PASS_REDRAWS:
                    queue.append(new_game(g.idx, g.redraws + 1))  # every seat tied: not scored, replayed
                else:
                    g.termination = ALL_PASS
                    finished.append(g)
            else:
                finished.append(g)
        active = [g for g, _ in pending]
        if not pending:
            break
        enc = encode_batch([g.raw for g, _ in pending])
        rotation = enc["meta"][:, 5]
        by_side = defaultdict(list)
        for i, (g, _) in enumerate(pending):
            by_side[g.seat_models[gpg.mover_seat(g.raw, g.seating, int(rotation[i]))]].append(i)
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
                    price = int(legal.price_cells(k).sample_price(cell, rng))  # on the layout's grid
                else:
                    price = play.fixed_price(g.raw, legal, k)
                g.auction_decisions += g.raw.round.round_type == "Auction"
                if gpg.apply_move(g, play, legal, k, price):  # (a refused move ends the game: "rejected_move")
                    g.decisions += 1

    records = []
    for g in finished:
        worth, win_share, fractions = gpg.game_outcome(g.raw, g.seating)
        record = {
            "idx": g.idx,
            "seats": list(g.seat_models),
            "win_share": win_share.tolist(),
            "worth_share": fractions.tolist(),
            "decisions": g.decisions,
            "auction_decisions": g.auction_decisions,
            "termination": g.termination,
            "start": g.start,
            "rules": g.rules,
            "optional_rules": list(getattr(g.raw, "optional_rules", None) or []),
            "all_pass_redrawn": g.redraws,
        }
        if g.rejected:
            record["rejected_move"] = g.rejected
        if g.idx < settings.get("save_games", 0) or g.rejected:
            record["net_worth"] = worth.tolist()
            record["seed"] = g.seed
            record["players"] = gpg.seating_players(g.raw, g.seating)
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
        "rules_mix": dict(args.rules_mix),
        "redraw_all_pass": not args.keep_all_pass,
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
    summary["rules_mix"] = dict(args.rules_mix)
    summary["by_rules"] = by_rules(records, args.a_seats / args.num_players)
    summary["all_pass_redrawn"] = int(sum(r.get("all_pass_redrawn", 0) for r in records))
    summary["seconds"] = round(time.time() - started, 1)
    return summary, records


def by_rules(records: list, equal: float) -> dict:
    """Per rules combination: games, decisions (median / p90 / max), terminations and side A's score (equal =
    ``equal``)."""
    out: dict = {}
    for name in sorted({r.get("rules", "base") for r in records}):
        rs = [r for r in records if r.get("rules", "base") == name]
        decisions = np.array([r["decisions"] for r in rs])
        a_score = [sum(w for w, side in zip(r["win_share"], r["seats"]) if side == "A") for r in rs]
        terminations: dict = {}
        for r in rs:
            terminations[r["termination"]] = terminations.get(r["termination"], 0) + 1
        out[name] = {
            "games": len(rs),
            "decisions_median": float(np.median(decisions)),
            "decisions_p90": float(np.percentile(decisions, 90)),
            "decisions_max": int(decisions.max()),
            "terminations": terminations,
            "A_score": float(np.mean(a_score)),
            "A_score_equal": equal,
        }
    return out


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
            "settings": {"optional_rules": list(r.get("optional_rules") or []), "seed": r.get("seed") or 0},
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
    parser.add_argument(
        "--keep-all-pass", action="store_true",
        help="The 1822 family: score games whose opening Stock Round ends with nothing bought, sold or bid (every "
        "seat tied) instead of replaying them from a fresh draw",
    )
    rules = parser.add_mutually_exclusive_group()
    rules.add_argument(
        "--rules", default=None, help="Every game under this rules combination (base, plus, plus+nr, ...)"
    )
    rules.add_argument(
        "--rules-mix", default=None,
        help="<name>:<weight>,... each start draws its rules from (default: the title's mix, "
        "titles.TitleSpec.rules_mix)",
    )
    parser.add_argument("--server", action="store_true", help="Each side on a generic inference server")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    parser.add_argument("--save-games", type=int, default=0)
    args = parser.parse_args()
    from rl18xx.agent.alphazero.titles import title_spec

    spec = title_spec(args.title)
    spec.check_players(args.num_players)
    try:
        args.rules_mix = spec.parse_rules_mix(args.rules if args.rules is not None else args.rules_mix)
    except ValueError as e:
        parser.error(str(e))
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
        if args.out and (args.save_games or any(r.get("rejected_move") for r in records)):
            save_game_logs(Path(args.out), match_index, a, b, args.title, records)
        all_records.extend({"match": f"{a} vs {b}", **r} for r in records)
        print(
            f"[{args.title} {summary['players']}p {summary['a_seats']}v{summary['players'] - summary['a_seats']}] "
            f"{a} vs {b}: {summary['A_score']:.3f} ± {summary['A_score_se']:.3f} over {summary['games']} games, "
            f"equal = {summary['A_score_equal']:.2f} (worth share {summary['A_worth_share']:.3f}, "
            f"{summary['seconds']:.0f}s; all-pass openings redrawn: {summary['all_pass_redrawn']})",
            flush=True,
        )
        if len(summary["by_rules"]) > 1 or list(summary["by_rules"]) != ["base"]:
            for name, r in summary["by_rules"].items():
                print(
                    f"    {name:16} {r['games']:4} games, decisions median {r['decisions_median']:.0f} / max "
                    f"{r['decisions_max']}, {r['terminations']}, A {r['A_score']:.3f}",
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
