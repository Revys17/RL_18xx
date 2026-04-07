"""Test full MCTS flow using only the Rust engine: encode, get_legal_actions, process_action.

For each game:
1. Create Rust game + RustGameAdapter
2. At each step: encode game state, get legal action indices, pick random action, process it
3. Verify no crashes through full game

Usage:
    uv run python tests/test_rust_mcts_flow.py --seeds 500 --start-seed 42
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import logging

logging.disable(logging.CRITICAL)

from engine_rs import BaseGame as RustGame
from rl18xx.rust_adapter import RustGameAdapter
from rl18xx.game.action_helper import ActionHelper
from rl18xx.agent.alphazero.action_mapper import ActionMapper
from rl18xx.agent.alphazero.encoder import Encoder_GNN


def play_mcts_game(seed: int, max_actions: int = 3000, full_mcts: bool = False) -> dict:
    rng = random.Random(seed)
    names = {1: "P1", 2: "P2", 3: "P3", 4: "P4"}
    rust_game = RustGame(names)
    adapted = RustGameAdapter(rust_game)
    helper = ActionHelper()
    mapper = ActionMapper()
    encoder = Encoder_GNN() if full_mcts else None

    for i in range(max_actions):
        # Encode (as MCTS would) — optional for speed
        if encoder is not None:
            try:
                encoded = encoder.encode(adapted)
            except Exception as e:
                return {"status": f"ENCODE_ERROR({e})", "steps": i}

        # Get legal actions via ActionMapper (the MCTS path)
        try:
            indices = mapper.get_legal_action_indices(adapted)
        except Exception as e:
            return {"status": f"LEGAL_ERROR({e})", "steps": i}

        if not indices:
            if rust_game.finished:
                return {"status": "FINISHED", "steps": i}
            return {"status": f"NO_CHOICES({rust_game.active_step_type()})", "steps": i}

        # Pick random action and map back (as MCTS would)
        idx = rng.choice(indices)
        try:
            action = mapper.map_index_to_action(idx, adapted)
            action_dict = action.to_dict()
        except Exception as e:
            return {"status": f"MAP_ERROR({e})", "steps": i}

        # Process action
        try:
            rust_game.process_action(action_dict)
        except Exception as e:
            return {"status": f"PROCESS_ERROR({e})", "steps": i}

    return {"status": "TIMEOUT", "steps": max_actions}


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Test MCTS flow with Rust engine")
    parser.add_argument("--seeds", type=int, default=500, help="Number of games")
    parser.add_argument("--start-seed", type=int, default=42, help="Starting seed")
    parser.add_argument("--max-actions", type=int, default=3000, help="Max actions per game")
    parser.add_argument("--full-mcts", action="store_true", help="Also run encoder at each step (slow)")
    args = parser.parse_args()

    passed = 0
    failed = 0
    errors = {}

    for i in range(args.seeds):
        seed = args.start_seed + i
        result = play_mcts_game(seed, max_actions=args.max_actions, full_mcts=args.full_mcts)
        status = result["status"]

        if status in ("FINISHED", "TIMEOUT"):
            passed += 1
        else:
            failed += 1
            cat = status.split("(")[0]
            errors[cat] = errors.get(cat, 0) + 1
            if failed <= 10:  # Print first 10 failures
                print(f"  FAIL seed={seed}: {status} at step {result['steps']}")

        if (i + 1) % 50 == 0 or i == args.seeds - 1:
            print(f"Progress: {i + 1}/{args.seeds} — {passed} passed, {failed} failed")

    print(f"\n{'=' * 60}")
    print(f"Total: {args.seeds} games")
    print(f"  Passed: {passed} ({passed * 100 / args.seeds:.1f}%)")
    print(f"  Failed: {failed}")
    if errors:
        print(f"\nError breakdown:")
        for cat, count in sorted(errors.items(), key=lambda x: -x[1]):
            print(f"  {cat}: {count}")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
