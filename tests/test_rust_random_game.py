"""Play random full games through both Python and Rust engines, comparing state at every step.

Uses the Python ActionHelper to generate legal actions, picks one at random,
and processes it through both engines. Validates that all state matches.
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import logging

logging.disable(logging.CRITICAL)

from engine_rs import BaseGame as RustGame
from rl18xx.game.action_helper import ActionHelper
from rl18xx.game.gamemap import GameMap
from tests.validate_rust_engine import compare_state


def play_random_game(seed: int, max_actions: int = 2000, verbose: bool = False) -> bool:
    rng = random.Random(seed)
    names = {1: "P1", 2: "P2", 3: "P3", 4: "P4"}
    game_cls = GameMap().game_by_title("1830")
    py_game = game_cls(names)
    rust_game = RustGame(names)
    helper = ActionHelper()

    action_count = 0
    errors = []

    while not py_game.finished and action_count < max_actions:
        # Get legal actions from Python
        choices = helper.get_all_choices(py_game)
        if not choices:
            break

        # Pick a random action
        action = rng.choice(choices)
        action_dict = action.to_dict()
        action_type = action_dict.get("type", "?")
        entity = action_dict.get("entity", "?")

        if verbose:
            print(f"  #{action_count}: {action_type} entity={entity}")

        # Process in Python
        try:
            py_game = py_game.process_action(action_dict)
        except Exception as e:
            print(f"  #{action_count} Python error: {e}")
            errors.append(f"Python error at #{action_count}: {e}")
            break

        # Process in Rust
        try:
            rust_game.process_action(action_dict)
        except Exception as e:
            print(f"  #{action_count} Rust error: {e}")
            print(f"    action: {action_type} entity={entity}")
            errors.append(f"Rust error at #{action_count}: {e}")
            break

        # Compare state
        mismatches = compare_state(rust_game, py_game)
        if mismatches:
            print(f"  #{action_count} {action_type} entity={entity}: State mismatch:")
            for m in mismatches:
                print(f"     {m}")
            errors.append(f"Mismatch at #{action_count}")
            break

        action_count += 1

    status = "FINISHED" if py_game.finished else f"STOPPED at {action_count}"
    ok = len(errors) == 0
    print(f"  [seed={seed}] {status} after {action_count} actions — {'OK' if ok else 'FAILED'}")
    return ok


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5, help="Number of random games to play")
    parser.add_argument("--start-seed", type=int, default=42, help="Starting seed")
    parser.add_argument("--max-actions", type=int, default=2000, help="Max actions per game")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    all_ok = True
    for i in range(args.seeds):
        seed = args.start_seed + i
        print(f"Game {i + 1}/{args.seeds} (seed={seed}):")
        ok = play_random_game(seed, max_actions=args.max_actions, verbose=args.verbose)
        if not ok:
            all_ok = False
        print()

    if all_ok:
        print(f"All {args.seeds} games passed!")
    else:
        print("Some games failed.")
    sys.exit(0 if all_ok else 1)
