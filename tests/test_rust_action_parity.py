"""Compare Rust engine's legal_action_types() and query methods against Python's ActionHelper.

For each random game step:
1. legal_action_types() from Rust vs set of action types from Python's ActionHelper
2. Encoder output parity (Rust-adapted game vs Python game)

Usage:
    uv run python tests/test_rust_action_parity.py --seeds 20 --start-seed 42
    uv run python tests/test_rust_action_parity.py --seeds 1 --start-seed 47 -v
"""

import random
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

import logging

logging.disable(logging.CRITICAL)

from engine_rs import BaseGame as RustGame
from rl18xx.game.action_helper import ActionHelper
from rl18xx.game.gamemap import GameMap
from rl18xx.rust_adapter import RustGameAdapter
from tests.validate_rust_engine import compare_state


def _py_action_types(choices):
    """Extract sorted unique action type strings from Python ActionHelper choices."""
    types = set()
    for action in choices:
        d = action.to_dict()
        t = d.get("type", "?")
        types.add(t)
    return sorted(types)


def play_game_with_parity_check(
    seed: int,
    max_actions: int = 2000,
    verbose: bool = False,
    check_encoder: bool = False,
):
    """Play a random game, comparing legal_action_types at every step.

    Returns dict with:
        ok: bool
        action_count: int
        action_type_diffs: list of diff descriptions
        encoder_diffs: list of diff descriptions
        state_diffs: list of diff descriptions
    """
    rng = random.Random(seed)
    names = {1: "P1", 2: "P2", 3: "P3", 4: "P4"}
    game_cls = GameMap().game_by_title("1830")
    py_game = game_cls(names)
    rust_game = RustGame(names)
    helper = ActionHelper()

    result = {
        "ok": True,
        "action_count": 0,
        "action_type_diffs": [],
        "encoder_diffs": [],
        "state_diffs": [],
        "finished": False,
    }

    encoder = None
    if check_encoder:
        from rl18xx.agent.alphazero.encoder import Encoder_GNN

        encoder = Encoder_GNN()

    action_count = 0
    while not py_game.finished and action_count < max_actions:
        # --- Check legal_action_types parity ---
        choices = helper.get_all_choices(py_game)
        if not choices:
            break

        py_types = _py_action_types(choices)
        rust_types = sorted(rust_game.legal_action_types())

        if py_types != rust_types:
            rust_extra = sorted(set(rust_types) - set(py_types))
            rust_missing = sorted(set(py_types) - set(rust_types))
            # Context info
            round_type = rust_game.round.round_type
            step = rust_game.get_or_step() if round_type == "Operating" else ""
            entity = rust_game.current_entity_id
            diff_desc = (
                f"#{action_count} [{round_type}/{step}] entity={entity} "
                f"rust_extra={rust_extra} rust_missing={rust_missing} "
                f"py={py_types} rust={rust_types}"
            )
            result["action_type_diffs"].append(diff_desc)
            if verbose:
                print(f"  ACTION_TYPE_DIFF {diff_desc}")

        # --- Check encoder parity ---
        if encoder is not None:
            try:
                adapted = RustGameAdapter(rust_game)
                rust_encoded = encoder.encode(adapted)
                py_encoded = encoder.encode(py_game)
                # Compare the game_state_vector (last element of the tuple)
                import numpy as np

                rust_vec = rust_encoded[-1]  # game_state_vector
                py_vec = py_encoded[-1]
                if isinstance(rust_vec, np.ndarray) and isinstance(py_vec, np.ndarray):
                    if rust_vec.shape == py_vec.shape:
                        diffs = np.where(np.abs(rust_vec - py_vec) > 1e-5)[0]
                        if len(diffs) > 0:
                            diff_desc = (
                                f"#{action_count} "
                                f"encoder_diffs_at_indices={diffs.tolist()[:10]} "
                                f"max_diff={np.max(np.abs(rust_vec - py_vec)):.6f}"
                            )
                            result["encoder_diffs"].append(diff_desc)
                            if verbose:
                                print(f"  ENCODER_DIFF {diff_desc}")
                    else:
                        diff_desc = f"#{action_count} shape_mismatch rust={rust_vec.shape} py={py_vec.shape}"
                        result["encoder_diffs"].append(diff_desc)
            except Exception as e:
                diff_desc = f"#{action_count} encoder_error: {e}"
                result["encoder_diffs"].append(diff_desc)
                if verbose:
                    print(f"  ENCODER_ERROR {diff_desc}")

        # Pick a random action and process through both engines
        action = rng.choice(choices)
        action_dict = action.to_dict()
        action_type = action_dict.get("type", "?")
        entity = action_dict.get("entity", "?")

        if verbose:
            print(f"  #{action_count}: {action_type} entity={entity}")

        try:
            py_game = py_game.process_action(action_dict)
        except Exception as e:
            result["state_diffs"].append(f"Python error at #{action_count}: {e}")
            result["ok"] = False
            break

        try:
            rust_game.process_action(action_dict)
        except Exception as e:
            result["state_diffs"].append(f"Rust error at #{action_count}: {e}")
            result["ok"] = False
            break

        # Compare state
        mismatches = compare_state(rust_game, py_game)
        if mismatches:
            for m in mismatches:
                result["state_diffs"].append(f"#{action_count} {action_type}: {m}")
            result["ok"] = False
            break

        action_count += 1

    result["action_count"] = action_count
    result["finished"] = py_game.finished
    if result["action_type_diffs"]:
        result["ok"] = False

    return result


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Test Rust vs Python action type parity")
    parser.add_argument("--seeds", type=int, default=20, help="Number of random games")
    parser.add_argument("--start-seed", type=int, default=42, help="Starting seed")
    parser.add_argument("--max-actions", type=int, default=2000, help="Max actions per game")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--encoder", action="store_true", help="Also check encoder parity")
    args = parser.parse_args()

    total_action_type_diffs = 0
    total_encoder_diffs = 0
    total_state_diffs = 0
    total_actions = 0
    diff_summary = defaultdict(int)  # category -> count

    all_ok = True
    for i in range(args.seeds):
        seed = args.start_seed + i
        print(f"Game {i + 1}/{args.seeds} (seed={seed}):", end=" ", flush=True)
        result = play_game_with_parity_check(
            seed,
            max_actions=args.max_actions,
            verbose=args.verbose,
            check_encoder=args.encoder,
        )
        total_actions += result["action_count"]
        status = "FINISHED" if result["finished"] else f"STOPPED@{result['action_count']}"
        n_at = len(result["action_type_diffs"])
        n_enc = len(result["encoder_diffs"])
        n_st = len(result["state_diffs"])
        total_action_type_diffs += n_at
        total_encoder_diffs += n_enc
        total_state_diffs += n_st

        # Categorize diffs
        for diff in result["action_type_diffs"]:
            # Extract rust_extra/rust_missing for summary
            if "rust_extra=" in diff:
                extra = diff.split("rust_extra=")[1].split(" rust_missing=")[0]
                missing = diff.split("rust_missing=")[1].split(" py=")[0]
                if extra != "[]":
                    diff_summary[f"extra:{extra}"] += 1
                if missing != "[]":
                    diff_summary[f"missing:{missing}"] += 1

        ok_str = "OK" if result["ok"] else "FAILED"
        detail = ""
        if n_at:
            detail += f" action_type_diffs={n_at}"
        if n_enc:
            detail += f" encoder_diffs={n_enc}"
        if n_st:
            detail += f" state_diffs={n_st}"
        print(f"{status} {result['action_count']} actions — {ok_str}{detail}")

        if not result["ok"]:
            all_ok = False
            if result["state_diffs"] and not args.verbose:
                for d in result["state_diffs"][:3]:
                    print(f"    STATE: {d}")

    print(f"\n{'=' * 60}")
    print(f"Total: {args.seeds} games, {total_actions} actions checked")
    print(f"  action_type_diffs: {total_action_type_diffs}")
    print(f"  encoder_diffs:     {total_encoder_diffs}")
    print(f"  state_diffs:       {total_state_diffs}")

    if diff_summary:
        print(f"\nAction type diff breakdown:")
        for cat, count in sorted(diff_summary.items(), key=lambda x: -x[1]):
            print(f"  {cat}: {count}")

    print()
    if all_ok:
        print(f"All {args.seeds} games passed!")
    else:
        print("Some games had diffs.")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
