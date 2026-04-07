"""Profile the MCTS hot path: measure time spent in each component.

Usage:
    uv run python tests/profile_mcts.py

Measures per-simulation breakdown:
  - pickle_clone() — game state deep copy
  - process_action() — apply one action
  - encoder.encode() — convert game state to tensors
  - get_legal_action_indices() — enumerate legal actions + map to indices
  - map_index_to_action() — reverse lookup index -> Action
  - Neural net forward pass (model inference)

Uses the Python BaseGame (same as actual MCTS) for encoder/action_mapper,
and the Rust BaseGame for clone/process_action benchmarks.
"""

import json
import sys
import time
import statistics
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import logging

logging.disable(logging.CRITICAL)

import numpy as np

from engine_rs import BaseGame as RustGame
from rl18xx.game.gamemap import GameMap
from rl18xx.rust_adapter import RustGameAdapter
from rl18xx.agent.alphazero.encoder import Encoder_GNN
from rl18xx.agent.alphazero.action_mapper import ActionMapper

GAME_FILE = "tests/test_games/manual_game.json"


def advance_rust_game(names, actions, target_moves):
    game = RustGame(names)
    for action in actions[:target_moves]:
        game.process_action(action)
    return game


def advance_adapted_game(names, actions, target_moves):
    game = RustGameAdapter(RustGame(names))
    for action in actions[:target_moves]:
        game.process_action(action)
    return game


def advance_python_game(names, actions, target_moves):
    game_cls = GameMap().game_by_title("1830")
    game = game_cls(names)
    for action in actions[:target_moves]:
        game = game.process_action(action)
    return game


def profile_rust_clone(game, n=500):
    """Measure Rust pickle_clone (clone_for_search)."""
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        clone = game.pickle_clone()
        t1 = time.perf_counter()
        times.append(t1 - t0)
        del clone
    return times


def profile_python_clone(game, n=200):
    """Measure Python pickle_clone."""
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        clone = game.pickle_clone()
        t1 = time.perf_counter()
        times.append(t1 - t0)
        del clone
    return times


def profile_encoder(game, n=200):
    """Measure encoder.encode() on Python game."""
    encoder = Encoder_GNN()
    encoder.initialize(game)
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        result = encoder.encode(game)
        t1 = time.perf_counter()
        times.append(t1 - t0)
        del result
    return times


def profile_rust_encoder(rust_game, n=500):
    """Measure Rust encode_for_gnn()."""
    import torch
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        gs, nf, enc_size, num_hexes, num_nf = rust_game.encode_for_gnn()
        # Wrap in torch tensors (same as Python would do)
        gs_t = torch.tensor(gs, dtype=torch.float32).unsqueeze(0)
        nf_t = torch.tensor(nf, dtype=torch.float32).reshape(num_hexes, num_nf)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return times


def profile_action_enum(game, n=200):
    """Measure get_legal_action_indices (ActionHelper + index mapping)."""
    mapper = ActionMapper()
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        indices = mapper.get_legal_action_indices(game)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return times


def profile_map_index_to_action(game, n=200):
    """Measure map_index_to_action (reverse lookup)."""
    mapper = ActionMapper()
    indices = mapper.get_legal_action_indices(game)
    if not indices:
        return []
    times = []
    for i in range(n):
        idx = indices[i % len(indices)]
        t0 = time.perf_counter()
        action = mapper.map_index_to_action(idx, game)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return times


def profile_full_node_creation(game, n=50):
    """Measure full MCTSNode creation (clone + process + encode + action enum).
    This simulates maybe_add_child without the neural net."""
    mapper = ActionMapper()
    encoder = Encoder_GNN()
    encoder.initialize(game)
    indices = mapper.get_legal_action_indices(game)
    if not indices:
        return []

    times = []
    for i in range(n):
        idx = indices[i % len(indices)]
        t0 = time.perf_counter()
        clone = game.pickle_clone()
        action = mapper.map_index_to_action(idx, clone)
        clone = clone.process_action(action)
        encoded = encoder.encode(clone)
        legal = mapper.get_legal_action_indices(clone)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return times


def fmt_stats(times, unit="ms"):
    """Format timing statistics."""
    if not times:
        return "N/A"
    scale = 1000 if unit == "ms" else 1_000_000 if unit == "us" else 1
    median = statistics.median(times) * scale
    mean = statistics.mean(times) * scale
    p95 = sorted(times)[int(len(times) * 0.95)] * scale
    return f"median={median:.3f}{unit}  mean={mean:.3f}{unit}  p95={p95:.3f}{unit}  (n={len(times)})"


def main():
    data = json.load(open(GAME_FILE))
    names = {p.get("id"): p["name"] for p in data["players"]}
    actions = data.get("actions", [])

    # Profile at multiple game stages
    stages = [
        ("Early game (move 20)", 20),
        ("Mid game (move 100)", 100),
        ("Late game (move 300)", 300),
    ]

    for stage_name, target_moves in stages:
        if target_moves > len(actions):
            print(f"\n{stage_name}: not enough actions ({len(actions)} < {target_moves}), skipping")
            continue

        print(f"\n{'=' * 70}")
        print(f"  {stage_name} — profiling from move {target_moves}")
        print(f"{'=' * 70}")

        rust_game = advance_rust_game(names, actions, target_moves)
        py_game = advance_python_game(names, actions, target_moves)

        print(f"\n  Rust pickle_clone():      {fmt_stats(profile_rust_clone(rust_game))}")
        print(f"  Python pickle_clone():    {fmt_stats(profile_python_clone(py_game, n=100))}")
        print(f"  encoder.encode() [py]:    {fmt_stats(profile_encoder(py_game))}")
        print(f"  action_enum [py]:         {fmt_stats(profile_action_enum(py_game))}")
        print(f"  map_index_to_action [py]: {fmt_stats(profile_map_index_to_action(py_game))}")
        print(f"  full_node [py]:           {fmt_stats(profile_full_node_creation(py_game))}")

        # Test with Rust adapter
        try:
            adapted_game = advance_adapted_game(names, actions, target_moves)
            print(f"\n  Adapted clone():          {fmt_stats(profile_python_clone(adapted_game, n=500))}")
            print(f"  encoder.encode() [adapt]: {fmt_stats(profile_encoder(adapted_game))}")
            print(f"  Rust encode_for_gnn():    {fmt_stats(profile_rust_encoder(rust_game))}")
            print(f"  action_enum [rust]:       {fmt_stats(profile_action_enum(adapted_game))}")
            print(f"  full_node [rust]:         {fmt_stats(profile_full_node_creation(adapted_game))}")
        except Exception as e:
            print(f"\n  Adapted game FAILED: {e}")

    # Estimate per-MCTS-move cost
    print(f"\n{'=' * 70}")
    print("  Estimated cost per MCTS move (200 readouts)")
    print(f"{'=' * 70}")
    py_game = advance_python_game(names, actions, 100)
    adapted_game = advance_adapted_game(names, actions, 100)

    clone_med = statistics.median(profile_python_clone(py_game, n=100))
    encode_med = statistics.median(profile_encoder(py_game, 200))
    action_enum_med = statistics.median(profile_action_enum(py_game, 200))

    per_sim = clone_med + encode_med + action_enum_med
    per_move = per_sim * 200
    per_game = per_move * 700

    print(f"\n  PYTHON ENGINE:")
    print(f"  Per simulation: {per_sim * 1000:.2f}ms")
    print(f"    clone:       {clone_med * 1000:.3f}ms ({clone_med / per_sim * 100:.0f}%)")
    print(f"    encode:      {encode_med * 1000:.3f}ms ({encode_med / per_sim * 100:.0f}%)")
    print(f"    action_enum: {action_enum_med * 1000:.3f}ms ({action_enum_med / per_sim * 100:.0f}%)")
    print(f"  Per MCTS move (200 sims): {per_move:.2f}s")
    print(f"  Per game (700 moves):     {per_game:.0f}s = {per_game / 3600:.1f}h")

    rust_clone_med = statistics.median(profile_python_clone(adapted_game, n=500))
    rust_encode_med = statistics.median(profile_encoder(adapted_game, 200))
    rust_action_med = statistics.median(profile_action_enum(adapted_game, 200))

    rust_per_sim = rust_clone_med + rust_encode_med + rust_action_med
    rust_per_move = rust_per_sim * 200
    rust_per_game = rust_per_move * 700

    print(f"\n  RUST ADAPTER:")
    print(f"  Per simulation: {rust_per_sim * 1000:.2f}ms")
    print(f"    clone:       {rust_clone_med * 1000:.3f}ms ({rust_clone_med / rust_per_sim * 100:.0f}%)")
    print(f"    encode:      {rust_encode_med * 1000:.3f}ms ({rust_encode_med / rust_per_sim * 100:.0f}%)")
    print(f"    action_enum: {rust_action_med * 1000:.3f}ms ({rust_action_med / rust_per_sim * 100:.0f}%)")
    print(f"  Per MCTS move (200 sims): {rust_per_move:.2f}s")
    print(f"  Per game (700 moves):     {rust_per_game:.0f}s = {rust_per_game / 3600:.1f}h")

    speedup = per_sim / rust_per_sim
    print(f"\n  SPEEDUP: {speedup:.1f}x faster per simulation")

    # With Rust-native encoder
    rust_enc_med = statistics.median(profile_rust_encoder(advance_rust_game(names, actions, 100), 500))
    native_per_sim = rust_clone_med + rust_enc_med + rust_action_med
    native_per_move = native_per_sim * 200
    native_per_game = native_per_move * 700

    print(f"\n  RUST ADAPTER + RUST ENCODER:")
    print(f"  Per simulation: {native_per_sim * 1000:.2f}ms")
    print(f"    clone:       {rust_clone_med * 1000:.3f}ms ({rust_clone_med / native_per_sim * 100:.0f}%)")
    print(f"    encode:      {rust_enc_med * 1000:.3f}ms ({rust_enc_med / native_per_sim * 100:.0f}%)")
    print(f"    action_enum: {rust_action_med * 1000:.3f}ms ({rust_action_med / native_per_sim * 100:.0f}%)")
    print(f"  Per MCTS move (200 sims): {native_per_move:.2f}s")
    print(f"  Per game (700 moves):     {native_per_game:.0f}s = {native_per_game / 3600:.1f}h")
    print(f"  SPEEDUP vs Python: {per_sim / native_per_sim:.0f}x")
    print(f"\n  Note: excludes neural net forward pass and process_action")


if __name__ == "__main__":
    main()
