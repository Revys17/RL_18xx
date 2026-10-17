"""Per-leaf costs of the Rust search, on human positions (CPU, one core):
selecting and expanding a leaf in Rust, cloning its game out to Python, the
Python adapter encoding (``mcts._rust_encode``) against the native one, and
the price-head / priors hand-off. Complements ``bench_search_harness.py``,
which times whole searches.

    uv run python scripts/bench_leaf_costs.py --positions 32
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import defaultdict  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positions", type=int, default=32)
    parser.add_argument("--leaves", type=int, default=64, help="Leaves expanded per position")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    import engine_rs
    from bench_search_harness import StubNet, build_position, position_prefixes

    from rl18xx.agent.alphazero.mcts import POLICY_SIZE, _native_encode, _rust_encode
    from rl18xx.rust_adapter import RustGameAdapter

    positions = [build_position(p) for p in position_prefixes(args.positions, args.seed)]
    positions = [g for g in positions if not g._game.finished]
    net = StubNet()
    t = defaultdict(float)
    counts = defaultdict(int)
    legal_sizes = []
    for game in positions:
        tree = engine_rs.RustMCTSPlayer(game)
        tree.set_q_config(True, 0.0)
        root = tree.root_idx
        probs = np.full(POLICY_SIZE, 1.0 / POLICY_SIZE, dtype=np.float32)
        value = np.full(6, 0.25, dtype=np.float32)
        tree.incorporate_results(root, probs, value, None)
        for _ in range(args.leaves):
            s = time.perf_counter()
            idx = tree.select_leaf()
            t["select_expand (Rust)"] += time.perf_counter() - s
            if tree.is_terminal(idx):
                tree.backup_value(idx, value.tolist())
                continue
            s = time.perf_counter()
            rust_game = tree.get_game_for_idx(idx)
            t["clone leaf game to Python"] += time.perf_counter() - s
            s = time.perf_counter()
            adapter = RustGameAdapter(rust_game)
            encoded = _rust_encode(adapter)
            t["_rust_encode (adapter, Python)"] += time.perf_counter() - s
            s = time.perf_counter()
            _native_encode(rust_game)
            t["_native_encode (Rust)"] += time.perf_counter() - s
            s = time.perf_counter()
            legal = tree.legal_action_indices_for_idx(idx)
            t["legal indices to Python"] += time.perf_counter() - s
            legal_sizes.append(len(legal))
            s = time.perf_counter()
            dense = np.zeros(POLICY_SIZE, dtype=np.float32)
            dense[legal] = 1.0 / len(legal)
            t["dense prior vector (client)"] += time.perf_counter() - s
            s = time.perf_counter()
            tree.incorporate_results(idx, dense, value, None)
            t["incorporate (dense probs)"] += time.perf_counter() - s
            counts["leaves"] += 1
        del encoded
    n = counts["leaves"]
    out = {k: round(1e6 * v / n, 1) for k, v in t.items()}
    out["leaves"] = n
    out["mean_legal_moves"] = round(float(np.mean(legal_sizes)), 1)
    print(json.dumps({"us_per_leaf": out}, indent=2))
    del net


if __name__ == "__main__":
    main()
