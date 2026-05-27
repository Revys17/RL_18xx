#!/usr/bin/env python3
"""Quick parity check: run cleaning pipeline against both Python and Rust engines
for a list of game IDs and print per-game outcomes.
"""
from __future__ import annotations
import sys, json, logging
from pathlib import Path

logging.disable(logging.CRITICAL)

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from rl18xx.agent.alphazero.pretraining import _get_game_object_for_game_with_reason


def run(gid: str, raw_dir: Path) -> None:
    path = raw_dir / f"{gid}.json"
    if not path.exists():
        print(f"{gid}: MISSING", flush=True)
        return
    raw = json.loads(path.read_text())
    if raw.get("status") != "finished":
        print(f"{gid}: SKIP not_finished", flush=True)
        return
    # Python
    try:
        py_obj, py_reason = _get_game_object_for_game_with_reason(raw, use_rust=False)
        py_status = "ok" if py_obj is not None else f"dropped:{py_reason}"
    except Exception as e:
        py_status = f"ERROR {type(e).__name__}: {str(e)[:100]}"
    # Rust
    try:
        rust_obj, rust_reason = _get_game_object_for_game_with_reason(raw, use_rust=True)
        rust_status = "ok" if rust_obj is not None else f"dropped:{rust_reason}"
    except Exception as e:
        rust_status = f"ERROR {type(e).__name__}: {str(e)[:100]}"
    print(f"{gid}: py={py_status} | rust={rust_status}", flush=True)


def main() -> int:
    raw_dir = REPO_ROOT / "human_games" / "1830"
    for gid in sys.argv[1:]:
        run(gid, raw_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
