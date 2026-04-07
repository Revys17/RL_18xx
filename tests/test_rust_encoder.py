"""Validate Rust encoder output against Python encoder."""

import json
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
logging.disable(logging.CRITICAL)

from engine_rs import BaseGame as RustGame
from rl18xx.rust_adapter import RustGameAdapter
from rl18xx.agent.alphazero.encoder import Encoder_GNN

GAME_FILE = "tests/test_games/manual_game.json"


def compare_encodings(py_game, rust_game, move_num):
    """Compare Python and Rust encoder outputs."""
    encoder = Encoder_GNN()
    encoder.initialize(py_game)

    # Python encoding
    py_gs, py_nf, py_ei, py_ea, *_ = encoder.encode(py_game)
    py_gs_np = py_gs.squeeze(0).numpy()
    py_nf_np = py_nf.numpy()

    # Rust encoding
    gs_flat, nf_flat, enc_size, num_hexes, num_nf = rust_game.encode_for_gnn()
    rust_gs_np = np.array(gs_flat, dtype=np.float32)
    rust_nf_np = np.array(nf_flat, dtype=np.float32).reshape(num_hexes, num_nf)

    # Compare game state
    gs_diff = np.abs(py_gs_np - rust_gs_np)
    gs_max_diff = gs_diff.max()
    gs_mismatch = np.where(gs_diff > 0.01)[0]

    # Compare node features
    nf_diff = np.abs(py_nf_np - rust_nf_np)
    nf_max_diff = nf_diff.max()
    nf_mismatch_count = (nf_diff > 0.01).sum()

    status = "PASS" if gs_max_diff < 0.02 and nf_max_diff < 0.02 else "FAIL"

    if status == "FAIL" or gs_max_diff > 0.001:
        print(f"  Move {move_num}: {status}  gs_max_diff={gs_max_diff:.4f} ({len(gs_mismatch)} indices), nf_max_diff={nf_max_diff:.4f} ({nf_mismatch_count} cells)")
        if len(gs_mismatch) > 0 and len(gs_mismatch) <= 10:
            for idx in gs_mismatch[:10]:
                print(f"    gs[{idx}]: py={py_gs_np[idx]:.4f} rust={rust_gs_np[idx]:.4f}")
    else:
        print(f"  Move {move_num}: {status}  gs_max_diff={gs_max_diff:.6f}, nf_max_diff={nf_max_diff:.6f}")

    return status == "PASS"


def main():
    data = json.load(open(GAME_FILE))
    names = {p.get("id"): p["name"] for p in data["players"]}
    actions = data.get("actions", [])

    # Advance both games together
    from rl18xx.game.gamemap import GameMap
    game_cls = GameMap().game_by_title("1830")
    py_game = game_cls(names)
    rust_game = RustGame(names)
    adapted = RustGameAdapter(rust_game)

    checkpoints = [0, 10, 20, 50, 100, 200, 300, 400, 500, 600, 690]
    passed = 0
    total = 0

    for i, action in enumerate(actions):
        if i in checkpoints:
            total += 1
            if compare_encodings(adapted, rust_game, i):
                passed += 1

        py_game = py_game.process_action(action)
        rust_game.process_action(action)
        adapted = RustGameAdapter(rust_game)

    # Final state
    total += 1
    if compare_encodings(adapted, rust_game, len(actions)):
        passed += 1

    print(f"\n{passed}/{total} checkpoints matched")


if __name__ == "__main__":
    main()
