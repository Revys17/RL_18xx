#!/usr/bin/env python3
"""Test engine parity on a specific list of game IDs.

Runs each game through ``get_game_object_for_game`` (Python cleaning), and
for games that survive cleaning, replays the cleaned action stream through
both Python and Rust engines, comparing state at every step.

Usage:
    uv run python scripts/test_specific_games.py 26846 27514 62926

The script prints one line per game: PERFECT, DROPPED <reason>,
DIVERGENCE <kind>, RUST_ERROR <msg>, PYTHON_ERROR <msg>, CLEANING_ERROR <msg>.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.disable(logging.CRITICAL)

from engine_rs import BaseGame as RustGame
from rl18xx.agent.alphazero.pretraining import get_game_object_for_game
from rl18xx.game.gamemap import GameMap

# Reuse the per-step comparator from the existing audit.
from scripts.audit_rust_engine_corpus import compare_state  # type: ignore


def test_one(game_id: str, raw_dir: Path) -> tuple[str, str]:
    """Return (status, detail). status is one of:
    PERFECT, DROPPED, DIVERGENCE, RUST_ERROR, PYTHON_ERROR, CLEANING_ERROR.
    """
    path = raw_dir / f"{game_id}.json"
    if not path.exists():
        return ("MISSING", f"file {path} not found")

    try:
        raw = json.loads(path.read_text())
    except Exception as e:
        return ("MISSING", f"failed to load: {e}")

    if raw.get("status") != "finished":
        return ("DROPPED", f"not_finished (game status: {raw.get('status')})")

    # Cleaning pipeline
    try:
        cleaned = get_game_object_for_game(raw)
    except Exception as e:
        return ("CLEANING_ERROR", f"{type(e).__name__}: {str(e)[:200]}")

    if cleaned is None:
        return ("DROPPED", "filtered by cleaning")

    # Replay cleaned action stream through both engines.
    cleaned_dict = cleaned.to_dict()
    actions = cleaned_dict.get("actions") or []
    names = {p["id"]: p["name"] for p in cleaned_dict["players"]}

    try:
        gm = GameMap()
        py_game = gm.game_by_title("1830")(names)
        rust_game = RustGame(names)
    except Exception as e:
        return ("ENGINE_INIT_ERROR", str(e)[:200])

    for i, action in enumerate(actions):
        a_type = action.get("type", "?")
        try:
            py_game = py_game.process_action(action)
        except Exception as e:
            return (
                "PYTHON_ERROR",
                f"action #{i} ({a_type}): {type(e).__name__}: {str(e)[:120]}",
            )
        try:
            rust_game.process_action(action)
        except Exception as e:
            return (
                "RUST_ERROR",
                f"action #{i} ({a_type}): {type(e).__name__}: {str(e)[:120]}",
            )
        try:
            errors = compare_state(rust_game, py_game)
        except Exception as e:
            return (
                "AUDIT_ERROR",
                f"compare_state at #{i}: {type(e).__name__}: {str(e)[:120]}",
            )
        if errors:
            return (
                "DIVERGENCE",
                f"action #{i} ({a_type}): {errors[0][:160]}",
            )

    return ("PERFECT", f"{len(actions)} actions matched")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("game_ids", nargs="+", help="Game IDs to test (filenames without .json)")
    p.add_argument(
        "--raw-dir",
        default=str(REPO_ROOT / "human_games" / "1830"),
        help="Directory with raw human game JSONs",
    )
    args = p.parse_args()

    raw_dir = Path(args.raw_dir)
    rows = []
    for gid in args.game_ids:
        status, detail = test_one(gid, raw_dir)
        print(f"{gid}: {status}  {detail}")
        rows.append((gid, status, detail))

    # Summary
    print()
    statuses = {}
    for _, s, _ in rows:
        statuses[s] = statuses.get(s, 0) + 1
    print("Summary: " + ", ".join(f"{k}={v}" for k, v in sorted(statuses.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
