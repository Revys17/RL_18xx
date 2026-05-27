"""Play random full games using the RUST action helper, applied to both engines.

Counterpart to ``test_rust_random_game.py`` (which uses the Python ActionHelper).
Uses Rust's ``get_factored_choices()`` to enumerate legal actions, picks one at
random (sampling a price from the range for price-bearing actions), converts to
a concrete BaseAction via the ActionMapper, and processes it through both
Python and Rust engines. Validates that all state matches at every step.
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import logging

logging.disable(logging.CRITICAL)

from engine_rs import BaseGame as RustGame
from rl18xx.agent.alphazero.action_mapper import ActionMapper
from rl18xx.game.gamemap import GameMap
from rl18xx.rust_adapter import RustGameAdapter
from tests.validate_rust_engine import compare_state


def play_random_game(seed: int, max_actions: int = 2000, verbose: bool = False) -> bool:
    rng = random.Random(seed)
    names = {1: "P1", 2: "P2", 3: "P3", 4: "P4"}
    game_cls = GameMap().game_by_title("1830")
    py_game = game_cls(names)
    rust_game = RustGame(names)
    rust_adapter = RustGameAdapter(rust_game)
    mapper = ActionMapper()

    action_count = 0
    errors = []

    while not py_game.finished and action_count < max_actions:
        indices, price_ranges, _action_types = mapper.get_legal_actions_factored(rust_adapter)
        if not indices:
            break

        idx = rng.choice(indices)
        pr = price_ranges.get(idx)
        if pr is not None:
            lo, hi = pr
            price = rng.randint(lo, hi)
            action = mapper.map_index_to_action_with_price(idx, rust_adapter, price)
        else:
            action = mapper.map_index_to_action(idx, rust_adapter)

        action_dict = action.to_dict()
        action_type = action_dict.get("type", "?")
        entity = action_dict.get("entity", "?")

        if verbose:
            print(f"  #{action_count}: {action_type} entity={entity}")

        try:
            py_game = py_game.process_action(action_dict)
        except Exception as e:
            print(f"  #{action_count} Python error: {e}")
            errors.append(f"Python error at #{action_count}: {e}")
            break

        try:
            rust_game.process_action(action_dict)
        except Exception as e:
            print(f"  #{action_count} Rust error: {e}")
            print(f"    action: {action_type} entity={entity}")
            errors.append(f"Rust error at #{action_count}: {e}")
            break

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
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--start-seed", type=int, default=42)
    parser.add_argument("--max-actions", type=int, default=2000)
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
