"""Differential test of the 1867 legal-move generator on the human corpus.

The generator's speed-ups (2026-11-13: one Ruby graph walk per tile-lay
candidate cut short, a validation clone without the JSON action log, ...)
keep the code they replaced callable as the *reference* generator
(`g1867::legal::reference`). At every human decision of every in-scope game
(the importer's replay: `generic_import_1867.replay_export`; recorded
auto-actions are not decisions) this script asks the engine to compare the two
under the title's cross-player policy AND the rules' (`g1867_reference_diff`):
move for move -- step, action, price range, certificate ids, in list order --
and at the RL index level (each move's `1867-v1` index, wire dict and range).
Any difference is a bug in the fast paths; the expected result is zero.

    CUDA_VISIBLE_DEVICES= taskset -c 24-31,56-63 nice -19 uv run python -P scripts/g1867_generator_diff.py \\
        --glob 'human_games/1867/*.json' --glob 'human_games/scrape_2026_10/1867/*.json' --jobs 8

The in-crate half (fuzz walks, 3-6 players, both policies at every state):
`cargo test --release --no-default-features generator_matches_reference` and
the long `g1867_diff_long -- --ignored`.
"""

from __future__ import annotations

import argparse
import glob as globmod
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

POLICIES = (None, "rules")


def diff_game(path: str) -> dict:
    from rl18xx.agent.alphazero.generic_data import new_engine_game
    from rl18xx.agent.alphazero.generic_import_1867 import replay_export, scope_reason
    from tests.title_replay_harness import game_seed

    game = json.load(open(path))
    out = {"id": Path(path).stem, "positions": 0, "compared": 0, "moves": 0, "diffs": [], "skipped": None}
    reason = scope_reason(game)
    if reason is not None:
        out["skipped"] = reason
        return out
    actions, auto, _stopped, _ = replay_export(game)
    n = len(game["players"])
    g = new_engine_game("1867", {i + 1: f"Player {i + 1}" for i in range(n)}, (), game_seed(game))
    for pos, a in enumerate(actions):
        if not auto[pos] and not g.finished:
            out["positions"] += 1
            for p in POLICIES:
                moves, diff = g.g1867_reference_diff(p)
                out["compared"] += 1
                out["moves"] += moves
                if diff is not None:
                    out["diffs"].append({"pos": pos, "policy": p, "diff": diff[:2000]})
        g.process_action(a)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--glob", action="append", default=None, help="raw 18xx.games exports (repeatable)")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="first N files only")
    ap.add_argument("--json", default=None, help="write the per-game results here")
    args = ap.parse_args()
    globs = args.glob or ["human_games/1867/*.json", "human_games/scrape_2026_10/1867/*.json"]
    paths = sorted(p for g in globs for p in globmod.glob(g))
    if args.limit:
        paths = paths[: args.limit]
    t0 = time.time()
    if args.jobs > 1:
        with mp.get_context("fork").Pool(args.jobs) as pool:
            results = pool.map(diff_game, paths, chunksize=1)
    else:
        results = [diff_game(p) for p in paths]
    games = [r for r in results if r["skipped"] is None]
    diffs = [(r["id"], d) for r in games for d in r["diffs"]]
    print(
        f"{len(paths)} files, {len(games)} games in scope, {sum(r['positions'] for r in games)} positions, "
        f"{sum(r['compared'] for r in games)} comparisons (title policy + rules), "
        f"{sum(r['moves'] for r in games)} moves compared, {len(diffs)} differences ({time.time() - t0:.0f} s)"
    )
    for gid, d in diffs[:20]:
        print(f"  {gid}@{d['pos']} policy={d['policy']}: {d['diff'][:600]}")
    if args.json:
        Path(args.json).write_text(json.dumps(results))
    return 1 if diffs else 0


if __name__ == "__main__":
    sys.exit(main())
