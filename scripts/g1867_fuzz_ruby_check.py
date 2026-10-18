"""Ruby HEAD replay of recorded 1867 fuzz games (scripts/g1867_fuzz.py),
classified (RL-readiness contract, item 5).

Each game replays in both engines through the scripts/ruby_diff.py rig;
per game:

  * clean       — Ruby HEAD processes every action with no state divergence;
  * dead_end    — no divergence, and Ruby rejects at or after the action the
                  fuzzer flagged as Ruby's stale-operator dead end
                  (``rl18xx.ruby_dead_end_at``; see g1867_ruby_dead_end_choice);
  * orphan_trains — no divergence or Ruby rejection before the action after
                  which a corporation left play still owning trains
                  (``rl18xx.ruby_orphan_trains_at``): Ruby's reset_corporation
                  keeps those trains on the old object and keeps offering
                  them to other corporations — a Ruby defect the engine does
                  not copy (it drops them);
  * FAIL        — a state divergence, or a Ruby rejection anywhere else:
                  a generator soundness bug or an engine divergence.

    RUBY_DOCKER_CPUSET=40-63 CUDA_VISIBLE_DEVICES="" uv run python \\
        scripts/g1867_fuzz_ruby_check.py logs/fuzz/1867 --jobs 4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ruby_diff  # noqa: E402


def classify(path: Path, rb_file: Path) -> tuple[str, str]:
    game = json.load(open(path))
    rs_states, _, rs_exc = ruby_diff.split(ruby_diff.rust_dump(game))
    rb_rows = [json.loads(line) for line in open(rb_file)] if rb_file.exists() else []
    rb_states, _, rb_exc = ruby_diff.split(rb_rows)
    if not rb_rows:
        return "FAIL", "no Ruby dump"
    first = None
    for fid in sorted(set(rs_states) & set(rb_states)):
        d = ruby_diff.diffs(rs_states[fid], rb_states[fid], "1867")
        if d:
            first = (fid, d)
            break
    dead = (game.get("rl18xx") or {}).get("ruby_dead_end_at")
    orphan = (game.get("rl18xx") or {}).get("ruby_orphan_trains_at")
    rb_stop = rb_exc["id"] if rb_exc else None
    if orphan and (first or rb_stop) and (first is None or first[0] > orphan) and (rb_stop is None or rb_stop > orphan):
        what = f"diverges @{first[0]}" if first else f"Ruby stops @{rb_stop}" if rb_stop else "no divergence"
        return "orphan_trains", f"{what} after orphan trains @{orphan}"
    if first and not (dead and first[0] > dead):
        return "FAIL", f"diverges @{first[0]}: {'; '.join(first[1][:3])}"
    if rs_exc:
        return "FAIL", f"rust rejects @{rs_exc['id']}: {rs_exc['exception'][:100]}"
    if rb_exc:
        if dead and rb_exc["id"] >= dead:
            return "dead_end", f"Ruby stops @{rb_exc['id']} (dead end from @{dead})"
        return "FAIL", f"ruby rejects @{rb_exc['id']}: {rb_exc['exception'][:100]}"
    return "clean", f"{len(game['actions'])} actions"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", help="directory of fuzz games")
    ap.add_argument("--ruby-repo", default="logs/ruby/18xx")
    ap.add_argument("--gems", default="logs/ruby/gems")
    ap.add_argument("--work", default="logs/ruby/dumps")
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()
    paths = sorted(Path(args.dir).glob("*.json"))
    gems = Path(args.gems).resolve()
    ruby_diff.ensure_gems(gems)

    def rb(p: Path) -> Path:
        out = Path(args.work) / f"1867_{Path(args.dir).name}_{p.stem}.rb.jsonl"
        ruby_diff.ruby_dump(p, out, Path(args.ruby_repo), gems)
        return out

    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        rb_files = list(ex.map(rb, paths))
    counts: dict = {}
    for p, f in zip(paths, rb_files):
        kind, note = classify(p, f)
        counts[kind] = counts.get(kind, 0) + 1
        print(f"{p.stem}: {kind} — {note}")
    print(counts)
    return 1 if counts.get("FAIL") else 0


if __name__ == "__main__":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    sys.exit(main())
