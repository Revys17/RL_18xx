"""Cost of the 1822 family's legal-move generator (1822, 1822PNW, 1822MX, 1822CA) on human positions: the
per-section profile, and before (the reference generator, the code before the 2026-11-14 speed-ups) vs after
on the SAME positions.

Every human decision of every game of a title's corpus (the completeness script's replay:
``ruby_filtered_actions``, recorded auto-actions are not decisions) is a position. ``--mode profile`` times
``rl_legal_moves_json(policy)`` -- what the self-play workers call per decision (generator + RL indices + wire
dicts) -- and records the generator's per-section profile (``BaseGame.g1822_profile``); ``--mode ab`` times it
through the reference and the fast generator alternately (wall and thread CPU time) and checks the two outputs
are equal. Output: one json line per position; ``--summary`` prints p50 / p90 / p99 / max / mean by step and
the profile's sections.

    CUDA_VISIBLE_DEVICES= taskset -c 24-31,56-63 nice -19 uv run python -P scripts/g1822_generator_profile.py \\
        --title 1822PNW --jobs 8 --every 4 --mode profile --out /tmp/p.jsonl.gz
    uv run python -P scripts/g1822_generator_profile.py --summary /tmp/p.jsonl.gz [--decisions]
"""

from __future__ import annotations

import argparse
import glob as globmod
import gzip
import json
import multiprocessing as mp
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CORPUS = "human_games/scrape_2026_10/{title}/*.json"


def new_game(title: str, game: dict):
    import engine_rs

    from tests.title_replay_harness import SEAT_ID_BASE, game_seed

    base = SEAT_ID_BASE.get(title, 0)
    seat = {p["id"]: base + i + 1 for i, p in enumerate(game["players"])}
    rules = list((game.get("settings") or {}).get("optional_rules") or [])
    g = engine_rs.BaseGame.new_titled(title, {v: f"Player {v - base}" for v in seat.values()}, rules or None, game_seed(game))
    return g, seat


def human_actions(game: dict, seat: dict):
    """(file id, auto?, action) of the replay, players' ids seated."""
    from tests.title_replay_harness import ruby_filtered_actions

    raw = json.loads(json.dumps(game["actions"]))
    for a in raw:
        a["_fid"] = a.get("id")
        for au in a.get("auto_actions") or []:
            au["_fid"] = a.get("id")
    for a in ruby_filtered_actions(raw, mark_auto=True):
        fid = a.pop("_fid", None)
        auto = a.pop("_auto", False)
        if a.get("entity_type") == "player":
            a["entity"] = a["user"] = seat.get(a["entity"], a["entity"])
        yield fid, auto, a


def run_game(args) -> list:
    path, title, mode, policy = args
    import engine_rs

    game = json.load(open(path))
    if game.get("title") != title or game.get("status") not in ("finished", "archived"):
        return []
    try:
        g, seat = new_game(title, game)
    except ValueError:
        return []
    engine_rs.BaseGame.g1822_profile(mode == "profile")
    engine_rs.BaseGame.g1822_profile_take()
    gid = Path(path).stem
    out = []
    for fid, auto, a in human_actions(game, seat):
        if not auto and not g.finished:
            if mode == "ab":
                row = [gid, fid]
                # Alternate which runs first, so neither always meets a warm cache.
                order = (True, False) if fid % 2 else (False, True)
                res = {}
                for ref in order:
                    prev = engine_rs.BaseGame.g1822_set_reference_generator(ref)
                    w0, c0 = time.perf_counter_ns(), time.thread_time_ns()
                    js = g.rl_legal_moves_json(policy)
                    res[ref] = (time.perf_counter_ns() - w0, time.thread_time_ns() - c0, js)
                    engine_rs.BaseGame.g1822_set_reference_generator(prev)
                moves = json.loads(res[False][2])
                steps = "+".join(sorted({m["step"] for m in moves}))
                decision = len(moves) > 1 or any(m["range"] is not None for m in moves)
                same = res[True][2] == res[False][2]
                row += [len(moves), decision, steps, res[True][0], res[True][1], res[False][0], res[False][1], same]
            else:
                w0, c0 = time.perf_counter_ns(), time.thread_time_ns()
                js = g.rl_legal_moves_json(policy)
                ns = time.thread_time_ns() - c0
                wall = time.perf_counter_ns() - w0
                moves = json.loads(js)
                steps = "+".join(sorted({m["step"] for m in moves}))
                types = "+".join(sorted({m["wire"]["type"] for m in moves}))
                decision = len(moves) > 1 or any(m["range"] is not None for m in moves)
                prof = {k: [c, t] for k, c, t in engine_rs.BaseGame.g1822_profile_take()}
                row = [gid, fid, len(moves), decision, steps, ns, prof, types, wall]
            out.append(row)
        try:
            g.process_action(a)
        except BaseException:  # noqa: BLE001 -- a record the replay rejects: stop the game there
            break
    return out


def _q(xs) -> str:
    import numpy as np

    xs = np.asarray(xs, dtype=np.float64) / 1e6
    if not len(xs):
        return "-"
    p50, p90, p99 = np.percentile(xs, [50, 90, 99])
    return f"n={len(xs):7d} p50 {p50:7.3f} p90 {p90:7.3f} p99 {p99:8.2f} max {xs.max():8.1f} mean {xs.mean():7.3f} ms"


def read_rows(path: str) -> list:
    """The rows of an output file (a file still being written: the rows so far)."""
    rows = []
    try:
        for line in gzip.open(path, "rt"):
            rows.append(json.loads(line))
    except (EOFError, json.JSONDecodeError):
        pass
    return rows


def summary(path: str, decisions_only: bool) -> None:
    rows = read_rows(path)
    if decisions_only:
        rows = [r for r in rows if r[3]]
    print(f"{path}: {len(rows)} positions{' (decisions only)' if decisions_only else ''}")
    if len(rows[0]) == 10:  # ab
        print("mismatching outputs:", sum(not r[9] for r in rows))
        for name, i in (("wall", 0), ("cpu", 1)):
            print(f"[{name}] before", _q([r[5 + i] for r in rows]))
            print(f"[{name}] after ", _q([r[7 + i] for r in rows]))
        by = defaultdict(list)
        for r in rows:
            by[r[4]].append(r)
        print("\nby step (thread CPU time, before -> after):")
        for k, rs in sorted(by.items(), key=lambda kv: -sum(r[6] for r in kv[1])):
            print(f"  {k[:30]:30s} before {_q([r[6] for r in rs])}")
            print(f"  {'':30s} after  {_q([r[8] for r in rs])}")
    else:
        tot = sum(r[5] for r in rows)
        print("all (thread CPU)", _q([r[5] for r in rows]))
        by = defaultdict(list)
        for r in rows:
            by[r[4]].append(r[5])
        print("\nby step (share of the total CPU time):")
        for k, xs in sorted(by.items(), key=lambda kv: -sum(kv[1])):
            print(f"  {k[:30]:30s} {100 * sum(xs) / tot:5.1f}%  {_q(xs)}")
        by = defaultdict(list)
        for r in rows:
            by[r[7]].append(r[5])
        print("\nby move family (the position's move types):")
        for k, xs in sorted(by.items(), key=lambda kv: -sum(kv[1]))[:20]:
            print(f"  {k[:50]:50s} {100 * sum(xs) / tot:5.1f}%  {_q(xs)}")
        sec = defaultdict(lambda: [0, 0])
        for r in rows:
            for k, (c, ns) in r[6].items():
                sec[k][0] += c
                sec[k][1] += ns
        print("\nprofile sections (nested sections overlap):")
        for k, (c, ns) in sorted(sec.items(), key=lambda kv: -kv[1][1])[:45]:
            print(f"  {k:58s} calls {c:9d} {ns / 1e9:8.2f} s ({100 * ns / tot:5.1f}%)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--title", default="1822", help="1822, 1822PNW, 1822MX or 1822CA")
    ap.add_argument("--glob", action="append", default=None, help="default: the title's 2026-10 scrape")
    ap.add_argument("--corpus-root", default=".", help="where human_games/ lives")
    ap.add_argument("--out", default=None)
    ap.add_argument("--mode", choices=["ab", "profile"], default="profile")
    ap.add_argument("--policy", default=None, help="None = the title's cross-player policy; 'rules'")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--every", type=int, default=1, help="every k-th file")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--summary", default=None, help="summarise an output file instead")
    ap.add_argument("--decisions", action="store_true", help="summary: decisions only (several moves or a price)")
    args = ap.parse_args()
    if args.summary:
        summary(args.summary, args.decisions)
        return 0
    import engine_rs

    globs = args.glob or [str(Path(args.corpus_root) / CORPUS.format(title=args.title))]
    paths = sorted(p for g in globs for p in globmod.glob(g))[:: args.every][: args.limit]
    # Build the layout before forking (a process's first index call builds it).
    engine_rs.rl_layout_info(args.title)
    t0 = time.time()
    work = [(p, args.title, args.mode, args.policy) for p in paths]
    with mp.get_context("fork").Pool(args.jobs) as pool, gzip.open(args.out, "wt") as f:
        n = 0
        for rows in pool.imap_unordered(run_game, work, chunksize=1):
            for r in rows:
                f.write(json.dumps(r) + "\n")
            n += len(rows)
    print(f"{args.title}: {len(paths)} files, {n} positions, {time.time() - t0:.0f} s -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
