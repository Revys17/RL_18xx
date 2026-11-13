"""Cost of the 1867 legal-move generator on human positions: before (the reference generator, the code
before the 2026-11-13 speed-ups) vs after, on the SAME positions, interleaved, and the per-section profile.

Every human decision of every in-scope game (the importer's replay, `generic_import_1867.replay_export`;
recorded auto-actions are not decisions) is a position. ``--mode ab`` times `rl_legal_moves_json(policy)` --
what the policy-gradient workers call per decision (generator + RL indices + wire dicts) -- through the
reference and the fast generator alternately (wall and thread CPU time); ``--mode profile`` records the
generator's per-section profile (`g1867_profile`) of the fast path. Output: one json line per position;
``--summary`` prints p50 / p90 / p99 / mean by step kind.

    CUDA_VISIBLE_DEVICES= taskset -c 24-31,56-63 nice -19 uv run python -P scripts/g1867_generator_profile.py \\
        --glob 'human_games/1867/*.json' --glob 'human_games/scrape_2026_10/1867/*.json' --jobs 8 --out /tmp/ab.jsonl.gz
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


def run_game(args) -> list:
    path, mode, policy = args
    import engine_rs

    from rl18xx.agent.alphazero.generic_data import new_engine_game
    from rl18xx.agent.alphazero.generic_import_1867 import replay_export, scope_reason
    from tests.title_replay_harness import game_seed

    game = json.load(open(path))
    if scope_reason(game) is not None:
        return []
    actions, auto, _stopped, _ = replay_export(game)
    n = len(game["players"])
    g = new_engine_game("1867", {i + 1: f"Player {i + 1}" for i in range(n)}, (), game_seed(game))
    engine_rs.BaseGame.g1867_profile(mode == "profile")
    engine_rs.BaseGame.g1867_profile_take()
    gid = Path(path).stem
    out = []
    for pos, a in enumerate(actions):
        if not auto[pos] and not g.finished:
            if mode == "ab":
                row = [gid, pos]
                # Alternate which runs first, so neither always meets a warm cache.
                order = (True, False) if pos % 2 else (False, True)
                res = {}
                for ref in order:
                    w0, c0 = time.perf_counter_ns(), time.thread_time_ns()
                    js = g.rl_legal_moves_json_with(policy, ref)
                    res[ref] = (time.perf_counter_ns() - w0, time.thread_time_ns() - c0, js)
                moves = json.loads(res[False][2])
                steps = "+".join(sorted({m["step"] for m in moves}))
                decision = len(moves) > 1 or any(m["range"] is not None for m in moves)
                same = res[True][2] == res[False][2]
                row += [len(moves), decision, steps, res[True][0], res[True][1], res[False][0], res[False][1], same]
            else:
                t0 = time.perf_counter_ns()
                js = g.rl_legal_moves_json(policy)
                ns = time.perf_counter_ns() - t0
                moves = json.loads(js)
                steps = "+".join(sorted({m["step"] for m in moves}))
                decision = len(moves) > 1 or any(m["range"] is not None for m in moves)
                prof = {k: [c, t] for k, c, t in engine_rs.BaseGame.g1867_profile_take()}
                row = [gid, pos, len(moves), decision, steps, ns, prof]
            out.append(row)
        g.process_action(a)
    return out


def _q(xs) -> str:
    import numpy as np

    xs = np.asarray(xs, dtype=np.float64) / 1e6
    if not len(xs):
        return "-"
    p50, p90, p99 = np.percentile(xs, [50, 90, 99])
    return f"n={len(xs):7d} p50 {p50:7.3f} p90 {p90:7.3f} p99 {p99:8.3f} mean {xs.mean():7.3f} ms"


def summary(path: str, decisions_only: bool) -> None:
    rows = [json.loads(line) for line in gzip.open(path, "rt")]
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
        print("\nby step kind (thread CPU time, before -> after):")
        for k, rs in sorted(by.items(), key=lambda kv: -sum(r[6] for r in kv[1])):
            if sum(r[6] for r in rs) < 2e8:
                continue
            print(f"  {k[:38]:38s} before {_q([r[6] for r in rs])}")
            print(f"  {'':38s} after  {_q([r[8] for r in rs])}")
    else:
        sec = defaultdict(lambda: [0, 0])
        for r in rows:
            for k, (c, ns) in r[6].items():
                sec[k][0] += c
                sec[k][1] += ns
        tot = sum(r[5] for r in rows)
        print("all", _q([r[5] for r in rows]))
        for k, (c, ns) in sorted(sec.items(), key=lambda kv: -kv[1][1])[:40]:
            print(f"  {k:50s} calls {c:9d} {ns / 1e9:7.2f} s ({100 * ns / tot:5.1f}%)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--glob", action="append", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--mode", choices=["ab", "profile"], default="ab")
    ap.add_argument("--policy", default=None, help="None = the title's cross-player policy; 'rules'")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--every", type=int, default=1, help="every k-th file")
    ap.add_argument("--summary", default=None, help="summarise an output file instead")
    ap.add_argument("--decisions", action="store_true", help="summary: decisions only (several moves or a price)")
    args = ap.parse_args()
    if args.summary:
        summary(args.summary, args.decisions)
        return 0
    globs = args.glob or ["human_games/1867/*.json", "human_games/scrape_2026_10/1867/*.json"]
    paths = sorted(p for g in globs for p in globmod.glob(g))[:: args.every]
    t0 = time.time()
    with mp.get_context("fork").Pool(args.jobs) as pool, gzip.open(args.out, "wt") as f:
        n = 0
        for rows in pool.imap_unordered(run_game, [(p, args.mode, args.policy) for p in paths], chunksize=1):
            for r in rows:
                f.write(json.dumps(r) + "\n")
            n += len(rows)
    print(f"{len(paths)} files, {n} positions, {time.time() - t0:.0f} s -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
