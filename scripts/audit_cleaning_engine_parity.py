#!/usr/bin/env python3
"""Audit Python vs. Rust engine parity on the human-game cleaning pipeline.

For every game in ``human_games/1830/``, runs ``get_game_object_for_game``
twice (Python engine and Rust adapter) and records the outcome:

  - ``ok``                — cleaning returned a non-None game.
  - ``dropped(<reason>)`` — cleaning filtered the game with a known reason.
  - ``cleaning_error``    — an exception escaped the cleaning function.

Discrepancy categories:
  - ``both_ok``                       — both engines accepted the game.
  - ``both_dropped_same_reason``      — both engines dropped, same reason.
  - ``both_dropped_diff_reason``      — both dropped, different reasons.
  - ``python_ok_rust_dropped``
  - ``python_dropped_rust_ok``
  - ``python_ok_rust_error``
  - ``python_error_rust_ok``
  - ``python_dropped_rust_error``
  - ``python_error_rust_dropped``
  - ``both_error``                    — both raised exceptions.

Usage:
    uv run python scripts/audit_cleaning_engine_parity.py --sample 50
    uv run python scripts/audit_cleaning_engine_parity.py
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import multiprocessing as mp
import sys
import traceback
from collections import Counter
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.disable(logging.CRITICAL)

CORPUS_DIR = REPO_ROOT / "human_games" / "1830"
OUT_CSV = REPO_ROOT / "docs" / "cleaning_engine_parity.csv"
OUT_MD = REPO_ROOT / "docs" / "cleaning_engine_parity.md"


def run_one(path_str: str) -> dict:
    """Worker function: run cleaning twice and return the result row.

    Imports happen inside the worker so the multiprocessing pool can spin
    up cleanly. Each worker process owns its own Python + Rust engine.
    """
    # Lazy imports — see note above.
    from rl18xx.agent.alphazero.pretraining import _get_game_object_for_game_with_reason

    path = Path(path_str)
    game_id = path.stem
    try:
        raw = json.loads(path.read_text())
    except Exception as e:
        return {
            "game_id": game_id,
            "python_outcome": "load_error",
            "rust_outcome": "load_error",
            "python_reason": f"{type(e).__name__}: {e}",
            "rust_reason": f"{type(e).__name__}: {e}",
            "match": True,
            "category": "load_error",
        }

    # Skip games that aren't finished — they're never cleaned by the real
    # pipeline either, and we want to keep the audit comparable.
    if raw.get("status") != "finished":
        return {
            "game_id": game_id,
            "python_outcome": "skip_not_finished",
            "rust_outcome": "skip_not_finished",
            "python_reason": raw.get("status", ""),
            "rust_reason": raw.get("status", ""),
            "match": True,
            "category": "skip_not_finished",
        }

    py_outcome, py_reason = _run_one(raw, use_rust=False)
    rust_outcome, rust_reason = _run_one(raw, use_rust=True)

    category = _classify(py_outcome, py_reason, rust_outcome, rust_reason)
    match = category in (
        "both_ok",
        "both_dropped_same_reason",
        "both_error",
    )

    return {
        "game_id": game_id,
        "python_outcome": py_outcome,
        "rust_outcome": rust_outcome,
        "python_reason": py_reason or "",
        "rust_reason": rust_reason or "",
        "match": match,
        "category": category,
    }


def _run_one(raw, use_rust: bool) -> tuple[str, Optional[str]]:
    from rl18xx.agent.alphazero.pretraining import _get_game_object_for_game_with_reason

    try:
        game, reason = _get_game_object_for_game_with_reason(raw, use_rust=use_rust)
    except Exception as e:
        msg = f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"
        return "cleaning_error", msg
    if game is None:
        return "dropped", reason or "unknown"
    return "ok", None


def _classify(py_out, py_reason, rust_out, rust_reason) -> str:
    if py_out == "ok" and rust_out == "ok":
        return "both_ok"
    if py_out == "dropped" and rust_out == "dropped":
        if py_reason == rust_reason:
            return "both_dropped_same_reason"
        return "both_dropped_diff_reason"
    if py_out == "cleaning_error" and rust_out == "cleaning_error":
        return "both_error"
    if py_out == "ok" and rust_out == "dropped":
        return "python_ok_rust_dropped"
    if py_out == "dropped" and rust_out == "ok":
        return "python_dropped_rust_ok"
    if py_out == "ok" and rust_out == "cleaning_error":
        return "python_ok_rust_error"
    if py_out == "cleaning_error" and rust_out == "ok":
        return "python_error_rust_ok"
    if py_out == "dropped" and rust_out == "cleaning_error":
        return "python_dropped_rust_error"
    if py_out == "cleaning_error" and rust_out == "dropped":
        return "python_error_rust_dropped"
    return f"other:{py_out}/{rust_out}"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sample", type=int, default=None, help="Cap to first N games (sorted) for a quick run.")
    p.add_argument(
        "--workers", type=int, default=max(1, mp.cpu_count() - 1),
        help="Number of worker processes (default: cpu_count - 1)."
    )
    p.add_argument("--no-csv", action="store_true", help="Do not write the CSV report.")
    p.add_argument("--no-md", action="store_true", help="Do not write the Markdown report.")
    args = p.parse_args()

    paths = sorted(CORPUS_DIR.glob("*.json"))
    if args.sample is not None:
        paths = paths[: args.sample]
    print(f"Auditing {len(paths)} games with {args.workers} worker(s)...")

    results = []
    if args.workers > 1:
        with mp.Pool(args.workers) as pool:
            for i, row in enumerate(pool.imap_unordered(run_one, [str(p) for p in paths], chunksize=4), 1):
                results.append(row)
                if i % 50 == 0 or i == len(paths):
                    print(f"  ... {i}/{len(paths)} processed")
    else:
        for i, path in enumerate(paths, 1):
            results.append(run_one(str(path)))
            if i % 50 == 0 or i == len(paths):
                print(f"  ... {i}/{len(paths)} processed")

    # Sort by game_id for stable output.
    results.sort(key=lambda r: r["game_id"])

    # Summary counts
    cat_counts = Counter(r["category"] for r in results)
    discrepancies = [r for r in results if not r["match"] and not r["category"].startswith("skip_") and r["category"] != "load_error"]

    # Drop reason cross-tab for both_dropped cases
    drop_reason_pairs = Counter(
        (r["python_reason"], r["rust_reason"])
        for r in results if r["category"].startswith("both_dropped")
    )

    print()
    print("=" * 60)
    print("AUDIT SUMMARY")
    print("=" * 60)
    print(f"Total games:        {len(results)}")
    print(f"Discrepancies:      {len(discrepancies)}")
    print(f"Match rate (cleaning outcome): {(len(results) - len(discrepancies)) / max(1, len(results)) * 100:.2f}%")
    print()
    print("Category counts:")
    for cat, n in sorted(cat_counts.items(), key=lambda x: -x[1]):
        print(f"  {cat:<35} {n}")

    if discrepancies:
        print()
        print("Discrepancies (game_id  python_outcome/reason  rust_outcome/reason):")
        for r in discrepancies:
            print(
                f"  {r['game_id']:<10} "
                f"py={r['python_outcome']}({r['python_reason']}) "
                f"rust={r['rust_outcome']}({r['rust_reason']})"
            )

    if not args.no_csv:
        OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        with OUT_CSV.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "game_id", "python_outcome", "rust_outcome",
                "python_reason", "rust_reason", "match", "category",
            ])
            writer.writeheader()
            for r in results:
                writer.writerow(r)
        print(f"\nCSV written: {OUT_CSV}")

    if not args.no_md:
        OUT_MD.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        lines.append("# Cleaning pipeline: Python vs. Rust engine parity audit\n")
        lines.append(f"Total games audited: **{len(results)}**\n")
        lines.append(f"Discrepancies: **{len(discrepancies)}**\n")
        match_rate = (len(results) - len(discrepancies)) / max(1, len(results)) * 100
        lines.append(f"Match rate (cleaning outcome): **{match_rate:.2f}%**\n")
        lines.append("\n## Category counts\n")
        lines.append("| Category | Count |")
        lines.append("|----------|-------|")
        for cat, n in sorted(cat_counts.items(), key=lambda x: -x[1]):
            lines.append(f"| `{cat}` | {n} |")
        lines.append("\n## Drop-reason cross-tab (both engines dropped)\n")
        lines.append("| Python reason | Rust reason | Count |")
        lines.append("|---------------|-------------|-------|")
        for (pr, rr), n in sorted(drop_reason_pairs.items(), key=lambda x: -x[1]):
            lines.append(f"| `{pr}` | `{rr}` | {n} |")
        if discrepancies:
            lines.append("\n## Discrepancies\n")
            lines.append("| Game ID | Python outcome | Python reason | Rust outcome | Rust reason | Category |")
            lines.append("|---------|---------------|--------------|--------------|------------|----------|")
            for r in discrepancies:
                lines.append(
                    f"| {r['game_id']} | {r['python_outcome']} | `{r['python_reason']}` | "
                    f"{r['rust_outcome']} | `{r['rust_reason']}` | {r['category']} |"
                )
        OUT_MD.write_text("\n".join(lines) + "\n")
        print(f"Markdown written: {OUT_MD}")

    return 0 if not discrepancies else 1


if __name__ == "__main__":
    sys.exit(main())
