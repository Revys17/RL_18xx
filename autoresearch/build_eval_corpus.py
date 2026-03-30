#!/usr/bin/env python3
"""
One-time script to build the evaluation corpus from human games.

Loads all usable clean games from human_games/1830_clean/, splits them
deterministically by sorted filename into training (first 200) and
evaluation (last 50) sets, and writes the game ID lists to JSON files.
"""

import json
import sys
from pathlib import Path

# Default path to clean human games (relative to repo root)
DEFAULT_GAME_DIR = Path(__file__).resolve().parent.parent / "human_games" / "1830_clean"
OUTPUT_DIR = Path(__file__).resolve().parent / "eval_corpus"


def build_eval_corpus(game_dir: Path = DEFAULT_GAME_DIR):
    if not game_dir.exists():
        print(f"ERROR: Game directory not found: {game_dir}", file=sys.stderr)
        sys.exit(1)

    game_files = sorted(game_dir.glob("*.json"))
    if not game_files:
        print(f"ERROR: No JSON files found in {game_dir}", file=sys.stderr)
        sys.exit(1)

    # Filter to usable games (skip those with status "error")
    usable_filenames = []
    skipped = 0
    for gf in game_files:
        with open(gf, "r") as f:
            data = json.load(f)
        if data.get("status") == "error":
            skipped += 1
            continue
        usable_filenames.append(gf.name)

    print(f"Total JSON files: {len(game_files)}")
    print(f"Skipped (status=error): {skipped}")
    print(f"Usable games: {len(usable_filenames)}")

    if len(usable_filenames) < 250:
        print(f"WARNING: Only {len(usable_filenames)} usable games found, need 250 for full split.", file=sys.stderr)

    # Deterministic split by sorted filename: first 200 = training, last 50 = evaluation
    # usable_filenames is already sorted (from sorted glob)
    train_ids = usable_filenames[:200]
    eval_ids = usable_filenames[-50:]

    # Count positions per split (by counting actions in each game)
    train_positions = 0
    for fn in train_ids:
        with open(game_dir / fn, "r") as f:
            data = json.load(f)
        train_positions += len(data.get("actions", []))

    eval_positions = 0
    for fn in eval_ids:
        with open(game_dir / fn, "r") as f:
            data = json.load(f)
        eval_positions += len(data.get("actions", []))

    # Write output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_path = OUTPUT_DIR / "train_game_ids.json"
    eval_path = OUTPUT_DIR / "eval_game_ids.json"

    with open(train_path, "w") as f:
        json.dump(train_ids, f, indent=2)

    with open(eval_path, "w") as f:
        json.dump(eval_ids, f, indent=2)

    print(f"\nTraining split:")
    print(f"  Games: {len(train_ids)}")
    print(f"  Positions (raw actions): {train_positions}")
    print(f"  Written to: {train_path}")

    print(f"\nEvaluation split:")
    print(f"  Games: {len(eval_ids)}")
    print(f"  Positions (raw actions): {eval_positions}")
    print(f"  Written to: {eval_path}")


if __name__ == "__main__":
    game_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_GAME_DIR
    build_eval_corpus(game_dir)
