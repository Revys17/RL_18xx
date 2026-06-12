"""1867 human-corpus outcome audit (Phase 1 validation strategy, leg 2).

Replays finished 18xx.games 1867 games (human_games/1867/*.json) through the
Rust engine and asserts outcome (final ``result()``) against the recorded
scores — the per-title analogue of the 1830 import-outcome audit. Volume
leg: thousands of games, outcome-level assertions only.

Usage:
    uv run python tests/g1867_corpus_audit.py [--glob 'human_games/1867/*.json']
        [--limit N] [--json /tmp/g1867_audit.json]

Exit codes: 0 = every replayed game matched (drops are reported, not fatal);
1 = at least one mismatch/rejection. While the title is unsupported, exits 2
with a clear message.
"""

from __future__ import annotations

import argparse
import glob as globmod
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.title_replay_harness import ReplayError, TitleUnsupported, replay_game


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="human_games/1867/*.json")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--json", dest="json_out", default=None)
    args = ap.parse_args()

    paths = sorted(globmod.glob(args.glob))[: args.limit]
    counts: Counter[str] = Counter()
    failures: list[dict] = []

    for path in paths:
        game_id = Path(path).stem
        try:
            game = json.load(open(path))
        except Exception as exc:
            counts["unparseable"] += 1
            failures.append({"id": game_id, "kind": "unparseable", "reason": str(exc)})
            continue
        if game.get("status") not in ("finished", "archived"):
            counts["not_finished"] += 1
            continue
        if game.get("settings", {}).get("optional_rules"):
            # grid_market etc. — replay only the base ruleset for now.
            counts["optional_rules_skipped"] += 1
            continue
        try:
            report = replay_game(game)
        except TitleUnsupported as exc:
            print(f"UNSUPPORTED: {exc}")
            return 2
        except ReplayError as exc:
            counts["rejected"] += 1
            failures.append(
                {
                    "id": game_id,
                    "kind": "rejected",
                    "step": exc.step,
                    "action": exc.action,
                    "reason": str(exc)[:300],
                }
            )
            continue
        recorded = {k: int(v) for k, v in (game.get("result") or {}).items()}
        if recorded and report.result != recorded:
            counts["result_mismatch"] += 1
            failures.append(
                {
                    "id": game_id,
                    "kind": "result_mismatch",
                    "engine": report.result,
                    "recorded": recorded,
                }
            )
        else:
            counts["outcome_match"] += 1

    print(f"scanned {len(paths)} files: {dict(counts)}")
    if args.json_out:
        json.dump(
            {"counts": dict(counts), "failures": failures}, open(args.json_out, "w"), indent=1
        )
        print(f"wrote {args.json_out}")
    bad = counts["rejected"] + counts["result_mismatch"] + counts["unparseable"]
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
