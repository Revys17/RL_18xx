"""Full-corpus per-step replay parity audit (parallel, fresh cleaning).

For every game in ``human_games/1830/``, runs the live cleaning pipeline,
then replays the cleaned action stream through both Python and Rust engines
with per-step state comparison via ``compare_state``.

Parallelizes across CPU cores. Writes a CSV + markdown summary.
"""
from __future__ import annotations
import sys
import json
import logging
import multiprocessing as mp
from pathlib import Path
from collections import Counter

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.disable(logging.CRITICAL)

CORPUS_DIR = REPO_ROOT / "human_games" / "1830"
OUT_CSV = REPO_ROOT / "docs" / "full_replay_audit.csv"
OUT_MD = REPO_ROOT / "docs" / "full_replay_audit.md"


def test_one(path_str: str) -> dict:
    from scripts.test_specific_games import test_one as t1
    path = Path(path_str)
    game_id = path.stem
    status, detail = t1(game_id, path.parent)
    return {"game_id": game_id, "status": status, "detail": detail[:200]}


def main():
    paths = sorted(CORPUS_DIR.glob("*.json"))
    print(f"Auditing {len(paths)} games with {mp.cpu_count()} workers")
    with mp.Pool() as pool:
        results = []
        for i, r in enumerate(pool.imap_unordered(test_one, [str(p) for p in paths], chunksize=8)):
            results.append(r)
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(paths)}")
    counts = Counter(r["status"] for r in results)
    print()
    print("Summary:")
    for status, n in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {status}: {n}")
    bad = [r for r in results if r["status"] not in ("PERFECT", "DROPPED")]
    print()
    print(f"Failures ({len(bad)}):")
    for r in bad[:20]:
        print(f"  {r['game_id']}: {r['status']} — {r['detail']}")
    # Write outputs
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    import csv
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["game_id", "status", "detail"])
        w.writeheader()
        for r in sorted(results, key=lambda x: x["game_id"]):
            w.writerow(r)
    with OUT_MD.open("w") as f:
        f.write("# Full per-step replay parity audit\n\n")
        f.write(f"Total games: **{len(paths)}**\n\n")
        f.write("## Status counts\n\n")
        for status, n in sorted(counts.items(), key=lambda x: -x[1]):
            f.write(f"- `{status}`: **{n}**\n")
        if bad:
            f.write("\n## Failures\n\n")
            for r in bad:
                f.write(f"- `{r['game_id']}` — **{r['status']}** — {r['detail']}\n")
    print(f"\nCSV: {OUT_CSV}")
    print(f"MD : {OUT_MD}")


if __name__ == "__main__":
    main()
