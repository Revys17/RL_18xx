"""Test RustGameAdapter compatibility with encoder and ActionHelper.

Replays a full game through the adapter to find all missing attributes/methods.
"""

import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import logging

logging.disable(logging.CRITICAL)

from engine_rs import BaseGame as RustGame
from rl18xx.rust_adapter import RustGameAdapter
from rl18xx.agent.alphazero.encoder import Encoder_GNN
from rl18xx.agent.alphazero.action_mapper import ActionMapper

GAME_FILE = "tests/test_games/manual_game.json"


def test_full_game_adapter():
    data = json.load(open(GAME_FILE))
    names = {p.get("id"): p["name"] for p in data["players"]}
    actions = data.get("actions", [])

    rust_game = RustGame(names)
    adapted = RustGameAdapter(rust_game)
    encoder = Encoder_GNN()
    mapper = ActionMapper()

    errors = []
    last_success = 0

    for i, action in enumerate(actions):
        try:
            # Test encoder
            encoded = encoder.encode(adapted)
        except Exception as e:
            err = f"Move {i}: encoder.encode() failed: {e}"
            if err not in [e_msg for e_msg, _ in errors]:
                errors.append((err, traceback.format_exc()))

        try:
            # Test action enumeration
            indices = mapper.get_legal_action_indices(adapted)
        except Exception as e:
            err = f"Move {i}: get_legal_action_indices() failed: {e}"
            if err not in [e_msg for e_msg, _ in errors]:
                errors.append((err, traceback.format_exc()))

        try:
            # Process the action
            adapted.process_action(action)
            last_success = i
        except Exception as e:
            err = f"Move {i}: process_action() failed: {e}"
            errors.append((err, traceback.format_exc()))
            break

    print(f"Processed {last_success + 1}/{len(actions)} actions successfully")

    if errors:
        print(f"\n{len(errors)} unique errors found:\n")
        for msg, tb in errors:
            print(f"  {msg}")
            # Print last 3 lines of traceback for context
            tb_lines = tb.strip().split('\n')
            for line in tb_lines[-4:]:
                print(f"    {line}")
            print()
    else:
        print("ALL actions processed with encoder + action_mapper successfully!")


if __name__ == "__main__":
    test_full_game_adapter()
