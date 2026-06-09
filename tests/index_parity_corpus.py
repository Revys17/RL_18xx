"""Parallel full-corpus strict parity gate.

Runs ``cleaning_diff.run(path, strict=True)`` over every human game in
``human_games/1830/`` across a process pool and aggregates the outcomes. The
strict lockstep enforces, at every state of the real cleaning import, all four
parity axes:
  * STATE parity (certs/cash/trains/tokens/privates),
  * factored ENUMERATION parity (``_key`` sets, bidirectional),
  * factored POLICY-INDEX parity (Python ActionMapper training-target indices
    == Rust ``factored_legal_indices`` search indices),
  * native DECODE parity (every legal index decode+applies to the same state via
    the native Rust path as via the Python ActionMapper) — on by default; set
    ``DECODE_CHECK=0`` to skip this (most expensive) axis, and
  * acceptance of the applied action by both engines.

Because the real recorded action stream is replayed, this exercises the rare
paths that synthetic random/round-robin walks miss — emergency-money president
share sales, bankruptcy, and D-train trade-in/exchange. It is the gate to run
before and after engine refactors / architectural simplifications.

Any game whose status is in BAD is a real Rust divergence. Prints a summary and
the first divergence of each bad game; writes the full result array to --json.

Usage::

    uv run python tests/index_parity_corpus.py [--limit N] [--workers W] [--json out.json]
    DECODE_CHECK=0 uv run python tests/index_parity_corpus.py   # skip decode axis (faster)
"""

import argparse
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

BAD = {"rust_error", "state_divergence", "enum_divergence", "index_divergence",
       "decode_divergence", "python_error", "stream_length_mismatch",
       "reason_mismatch", "diagnostic_crash"}


def _one(path):
    # Imported inside the worker so each process initializes its own engines.
    import logging
    logging.disable(logging.CRITICAL)
    from tests import cleaning_diff
    # Native decode parity is part of the gate by default; set DECODE_CHECK=0 to
    # skip it (it is the most expensive axis — it decode+applies every legal
    # index at every state). Decode mismatches surface as ``decode_divergence``,
    # which is in BAD above so they actually fail the gate.
    if os.environ.get("DECODE_CHECK", "1") != "0":
        cleaning_diff.set_check_decode(True)
    try:
        return cleaning_diff.run(path, strict=True)
    except BaseException as exc:  # noqa: BLE001
        import traceback
        return {"game_id": Path(path).stem, "status": "diagnostic_crash",
                "error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()[-1500:]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=0, help="cap number of games (0 = all)")
    ap.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 8))
    ap.add_argument("--json", default="/tmp/index_parity_corpus.json")
    ap.add_argument("--glob", default="human_games/1830/*.json")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.glob))
    if args.limit:
        paths = paths[: args.limit]
    total = len(paths)
    print(f"Running strict parity over {total} games on {args.workers} workers...", flush=True)

    results = []
    counts = {}
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_one, p): p for p in paths}
        for fut in as_completed(futs):
            res = fut.result()
            results.append(res)
            counts[res["status"]] = counts.get(res["status"], 0) + 1
            done += 1
            if res["status"] in BAD:
                print(f"  !! BAD [{res.get('game_id')}] {res['status']} "
                      f"@#{res.get('index')} step={res.get('op_step')} "
                      f"phase={res.get('phase')} entity={res.get('entity')} "
                      f"py_only={res.get('py_only')} rust_only={res.get('rust_only')} "
                      f"{res.get('error','')}", flush=True)
            if done % 250 == 0:
                print(f"  ... {done}/{total}", flush=True)

    Path(args.json).write_text(json.dumps(results, indent=2, default=str))

    print("\n==== SUMMARY ====")
    for status in sorted(counts):
        print(f"  {status}: {counts[status]}")
    bad = [r for r in results if r["status"] in BAD]
    print(f"\n  total={total}  bad={len(bad)}")
    print(f"  wrote {args.json}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
