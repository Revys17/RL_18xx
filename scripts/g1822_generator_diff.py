"""Differential test of the 1822 family's legal-move generator (1822, 1822PNW, 1822MX, 1822CA) on the human
corpus.

The generator's speed-ups (2026-11-14) keep the code they replaced callable as the *reference* generator
(`g1822::legal::reference`). At every human decision of every game of a title's corpus (the completeness
script's replay; recorded auto-actions are not decisions) this script asks the engine to compare the two under
the title's cross-player policy AND the rules' (`BaseGame.g1822_reference_diff`): the sorted move lists (action,
combo, share price, price range, certificate ids), the RL index rows (each move's `<title>-v1` index, wire dict
and range), and that both calls leave the state as they found it. Any difference is a bug in the fast paths;
the expected result is zero.

    CUDA_VISIBLE_DEVICES= taskset -c 24-31,56-63 nice -19 uv run python -P scripts/g1822_generator_diff.py \\
        --title 1822PNW --jobs 8 [--corpus-root /path/to/checkout]

The in-crate half (fuzz walks at each title's player counts, both policies at every state):
`cargo test --release --no-default-features generator_matches_reference` and the long
`g1822_diff_long -- --ignored`.
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
# --both-once: one reference run per position (see g1822_reference_diff_both)
BOTH = False
CORPUS = "human_games/scrape_2026_10/{title}/*.json"


def diff_game(args) -> dict:
    path, title = args
    from scripts.g1822_generator_profile import human_actions, new_game

    game = json.load(open(path))
    out = {"id": Path(path).stem, "positions": 0, "compared": 0, "moves": 0, "diffs": [], "skipped": None}
    if game.get("title") != title or game.get("status") not in ("finished", "archived"):
        out["skipped"] = "not a finished game of the title"
        return out
    try:
        g, seat = new_game(title, game)
    except ValueError as exc:
        out["skipped"] = f"construction: {exc}"
        return out
    for fid, auto, a in human_actions(game, seat):
        if not auto and not g.finished:
            out["positions"] += 1
            # the fast paths' pure helpers: neighbour map, every tokened corporation's graph
            check = g.g1822_graph_check()
            if check is not None:
                out["diffs"].append({"fid": fid, "policy": "graph", "diff": check[:3000]})
            if BOTH:
                # both policies, the reference run once (its title-policy list derived from its rules' list:
                # BaseGame.g1822_reference_diff_both)
                moves, diff = g.g1822_reference_diff_both()
                out["compared"] += 2
                out["moves"] += moves
                if diff is not None:
                    out["diffs"].append({"fid": fid, "policy": "both", "diff": diff[:3000]})
            else:
                for p in POLICIES:
                    moves, diff = g.g1822_reference_diff(p)
                    out["compared"] += 1
                    out["moves"] += moves
                    if diff is not None:
                        out["diffs"].append({"fid": fid, "policy": p, "diff": diff[:3000]})
        try:
            g.process_action(a)
        except BaseException as exc:  # noqa: BLE001 -- a record the replay rejects (Ruby HEAD rejects it too)
            out["stopped"] = f"file id {fid}: {str(exc)[:200]}"
            break
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--title", default="1822")
    ap.add_argument("--glob", action="append", default=None, help="default: the title's 2026-10 scrape")
    ap.add_argument("--corpus-root", default=".", help="where human_games/ lives")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--every", type=int, default=1, help="every k-th file")
    ap.add_argument("--limit", type=int, default=0, help="first N files only")
    ap.add_argument("--json", default=None, help="write the per-game results here")
    ap.add_argument(
        "--both-once",
        action="store_true",
        help="run the reference generator once per position (under the rules'; its title-policy list derived "
        "from it, BaseGame.g1822_reference_diff_both): about half the time",
    )
    args = ap.parse_args()
    global BOTH
    BOTH = args.both_once
    import engine_rs

    globs = args.glob or [str(Path(args.corpus_root) / CORPUS.format(title=args.title))]
    paths = sorted(p for g in globs for p in globmod.glob(g))[:: args.every]
    if args.limit:
        paths = paths[: args.limit]
    engine_rs.rl_layout_info(args.title)  # build the layout before forking
    t0 = time.time()
    work = [(p, args.title) for p in paths]
    if args.jobs > 1:
        with mp.get_context("fork").Pool(args.jobs) as pool:
            results = list(pool.imap_unordered(diff_game, work, chunksize=1))
    else:
        results = [diff_game(w) for w in work]
    games = [r for r in results if r["skipped"] is None]
    diffs = [(r["id"], d) for r in games for d in r["diffs"]]
    stopped = sum(1 for r in games if r.get("stopped"))
    print(
        f"{args.title}: {len(paths)} files, {len(games)} games ({stopped} stop at a record the replay rejects), "
        f"{sum(r['positions'] for r in games)} positions, {sum(r['compared'] for r in games)} comparisons "
        f"(title policy + rules), {sum(r['moves'] for r in games)} moves compared, {len(diffs)} differences "
        f"({time.time() - t0:.0f} s)"
    )
    for gid, d in diffs[:20]:
        print(f"  {gid}@{d['fid']} policy={d['policy']}: {d['diff'][:800]}")
    if args.json:
        Path(args.json).write_text(json.dumps(results))
    return 1 if diffs else 0


if __name__ == "__main__":
    sys.exit(main())
