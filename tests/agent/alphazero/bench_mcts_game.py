"""
Benchmark: full self-play game with MCTS using v1 (GNN) and v2 (Hex Transformer) models.

Runs a complete 1830 game with typical self-play settings (200 readouts) and reports
per-move and per-component timing.

Usage:
    uv run pytest tests/agent/alphazero/bench_mcts_game.py -s -v
    uv run pytest tests/agent/alphazero/bench_mcts_game.py -s -v -k v2
"""

import time
import torch
import numpy as np
from dataclasses import dataclass, field

from rl18xx.game.gamemap import GameMap
from rl18xx.agent.alphazero.config import ModelConfig, ModelV2Config, SelfPlayConfig
from rl18xx.agent.alphazero.model import AlphaZeroGNNModel
from rl18xx.agent.alphazero.model_v2 import AlphaZeroV2Model
from rl18xx.agent.alphazero.self_play import MCTSPlayer
import rl18xx.agent.alphazero.mcts as mcts


@dataclass
class BenchmarkStats:
    """Collects timing statistics for a full self-play game."""

    model_name: str = ""
    total_moves: int = 0
    total_game_time: float = 0.0
    move_times: list = field(default_factory=list)
    tree_search_times: list = field(default_factory=list)
    encode_times: list = field(default_factory=list)
    network_times: list = field(default_factory=list)
    readouts_per_move: list = field(default_factory=list)
    legal_actions_per_move: list = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"\n{'='*70}",
            f"  Benchmark Results: {self.model_name}",
            f"{'='*70}",
            f"  Total moves played:      {self.total_moves}",
            f"  Total game time:         {self.total_game_time:.2f}s",
            f"  Avg time per move:       {np.mean(self.move_times)*1000:.1f}ms",
            f"  Median time per move:    {np.median(self.move_times)*1000:.1f}ms",
            f"  P95 time per move:       {np.percentile(self.move_times, 95)*1000:.1f}ms",
            f"  Max time per move:       {np.max(self.move_times)*1000:.1f}ms",
            f"",
            f"  Avg readouts per move:   {np.mean(self.readouts_per_move):.1f}",
            f"  Avg legal actions:       {np.mean(self.legal_actions_per_move):.1f}",
            f"",
            f"  --- Per tree_search() call ---",
            f"  Total tree_search calls: {len(self.tree_search_times)}",
            f"  Avg tree_search time:    {np.mean(self.tree_search_times)*1000:.2f}ms",
            f"",
            f"  --- Network inference ---",
            f"  Total network calls:     {len(self.network_times)}",
            f"  Avg network time:        {np.mean(self.network_times)*1000:.2f}ms",
            f"",
            f"  --- Encoding (Python encoder only; Rust encode tracked in tree_search) ---",
            f"  Total encode calls:      {len(self.encode_times)}",
            f"  Avg encode time:         {np.mean(self.encode_times)*1000:.2f}ms" if self.encode_times else f"  (Rust encoder used — encode time included in tree_search)",
            f"{'='*70}",
        ]
        return "\n".join(lines)


def _create_game():
    """Create a fresh 1830 game using the best available engine."""
    try:
        from engine_rs import BaseGame as RustGame
        from rl18xx.rust_adapter import RustGameAdapter

        players = {1: "Player 1", 2: "Player 2", 3: "Player 3", 4: "Player 4"}
        return RustGameAdapter(RustGame(players))
    except ImportError:
        game_map = GameMap()
        game_class = game_map.game_by_title("1830")
        players = {1: "Player 1", 2: "Player 2", 3: "Player 3", 4: "Player 4"}
        return game_class(players)


def _run_selfplay_game(model, model_name: str, num_readouts: int = 200, max_moves: int = 0) -> BenchmarkStats:
    """Run a full self-play game with instrumented timing.

    Args:
        model: AlphaZero model (v1 or v2)
        model_name: Name for reporting
        num_readouts: MCTS simulations per move (default: 200, typical for self-play)
        max_moves: Stop after this many moves (0 = play to completion)
    """
    stats = BenchmarkStats(model_name=model_name)
    model.eval()

    config = SelfPlayConfig(
        network=model,
        num_readouts=num_readouts,
        min_readouts=50,
        parallel_readouts=32,
        use_fp16_inference=False,  # CPU benchmarking, no FP16
        use_score_values=True,
        backup_discount=0.995,
    )

    game = _create_game()
    player = MCTSPlayer(config)
    player.initialize_game(game)

    # If v2 model, initialize structural matrices
    if hasattr(model, "_compute_structural_matrices"):
        model._compute_structural_matrices(game)

    # Instrument the encoder for timing — patch the singleton that MCTS nodes use
    from rl18xx.agent.alphazero.encoder import Encoder_1830

    encoder_instance = Encoder_1830.get_encoder_for_model(model)
    original_encode = encoder_instance.encode
    encode_times_local = []

    def timed_encode(game_state):
        t0 = time.perf_counter()
        result = original_encode(game_state)
        encode_times_local.append(time.perf_counter() - t0)
        return result

    encoder_instance.encode = timed_encode

    # Instrument network inference
    original_run_many = model.run_many_encoded
    network_times_local = []

    def timed_run_many(encoded_states):
        t0 = time.perf_counter()
        result = original_run_many(encoded_states)
        network_times_local.append(time.perf_counter() - t0)
        return result

    model.run_many_encoded = timed_run_many

    # Expand root
    first_node = player.root.select_leaf()
    first_node.ensure_encoded()
    with torch.no_grad():
        priors, _, values = model.run_encoded(first_node.encoded_game_state)
    first_node.incorporate_results(priors, values, up_to=player.root)

    game_start = time.perf_counter()
    move_count = 0

    while not player.root.is_done():
        move_start = time.perf_counter()

        # Inject noise
        player.root.inject_noise()

        # Adaptive readouts
        n_legal = player.root.num_legal_actions
        if n_legal <= 1:
            # Forced move — skip MCTS
            move = player.root.legal_action_indices[0] if n_legal == 1 else player.pick_move()
            player.play_move(move)
            move_count += 1
            move_time = time.perf_counter() - move_start
            stats.move_times.append(move_time)
            stats.readouts_per_move.append(0)
            stats.legal_actions_per_move.append(n_legal)
            if max_moves > 0 and move_count >= max_moves:
                break
            continue

        if n_legal <= 5:
            readouts = max(config.min_readouts, config.num_readouts // 4)
        elif n_legal <= 20:
            readouts = max(config.min_readouts, config.num_readouts // 2)
        else:
            readouts = config.num_readouts

        target_readouts = player.root.N + readouts

        # Tree search
        tree_search_calls = 0
        while player.root.N < target_readouts:
            ts_start = time.perf_counter()
            player.tree_search()
            stats.tree_search_times.append(time.perf_counter() - ts_start)
            tree_search_calls += 1

        # Pick and play move
        move = player.pick_move()
        player.play_move(move)

        move_time = time.perf_counter() - move_start
        stats.move_times.append(move_time)
        stats.readouts_per_move.append(readouts)
        stats.legal_actions_per_move.append(n_legal)

        move_count += 1
        if move_count % 50 == 0:
            elapsed = time.perf_counter() - game_start
            print(f"  [{model_name}] Move {move_count}: {elapsed:.1f}s elapsed, avg {elapsed/move_count*1000:.0f}ms/move")

        if max_moves > 0 and move_count >= max_moves:
            break

    stats.total_game_time = time.perf_counter() - game_start
    stats.total_moves = move_count
    stats.encode_times = encode_times_local
    stats.network_times = network_times_local

    # Restore original methods
    encoder_instance.encode = original_encode
    model.run_many_encoded = original_run_many

    return stats


# ──────────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────────


def test_bench_v1_mcts_game():
    """Benchmark: full self-play game with v1 (GNN) model, 200 readouts."""
    model = AlphaZeroGNNModel(ModelConfig())
    stats = _run_selfplay_game(model, "v1 (GNN)", num_readouts=200, max_moves=50)
    print(stats.summary())
    assert stats.total_moves > 0


def test_bench_v2_mcts_game():
    """Benchmark: full self-play game with v2 (Hex Transformer) model, 200 readouts."""
    model = AlphaZeroV2Model(ModelV2Config())
    stats = _run_selfplay_game(model, "v2 (Hex Transformer)", num_readouts=200, max_moves=50)
    print(stats.summary())
    assert stats.total_moves > 0


def test_bench_v1_vs_v2_comparison():
    """Side-by-side comparison of v1 and v2 performance for 20 moves."""
    NUM_MOVES = 20

    print("\n\n" + "=" * 70)
    print("  V1 vs V2 Performance Comparison")
    print("=" * 70)

    v1_model = AlphaZeroGNNModel(ModelConfig())
    v1_stats = _run_selfplay_game(v1_model, "v1 (GNN)", num_readouts=200, max_moves=NUM_MOVES)

    v2_model = AlphaZeroV2Model(ModelV2Config())
    v2_stats = _run_selfplay_game(v2_model, "v2 (Hex Transformer)", num_readouts=200, max_moves=NUM_MOVES)

    print(v1_stats.summary())
    print(v2_stats.summary())

    # Comparison
    v1_avg = np.mean(v1_stats.move_times) * 1000
    v2_avg = np.mean(v2_stats.move_times) * 1000
    ratio = v2_avg / v1_avg if v1_avg > 0 else float("inf")

    v1_params = sum(p.numel() for p in v1_model.parameters())
    v2_params = sum(p.numel() for p in v2_model.parameters())

    print(f"\n  --- Comparison ---")
    print(f"  V1 params: {v1_params:,}   V2 params: {v2_params:,}   Ratio: {v2_params/v1_params:.2f}x")
    print(f"  V1 avg move: {v1_avg:.1f}ms   V2 avg move: {v2_avg:.1f}ms   Ratio: {ratio:.2f}x")

    if v1_stats.network_times and v2_stats.network_times:
        v1_net = np.mean(v1_stats.network_times) * 1000
        v2_net = np.mean(v2_stats.network_times) * 1000
        print(f"  V1 avg network: {v1_net:.2f}ms   V2 avg network: {v2_net:.2f}ms   Ratio: {v2_net/v1_net:.2f}x")

    if v1_stats.encode_times and v2_stats.encode_times:
        v1_enc = np.mean(v1_stats.encode_times) * 1000
        v2_enc = np.mean(v2_stats.encode_times) * 1000
        print(f"  V1 avg encode: {v1_enc:.2f}ms   V2 avg encode: {v2_enc:.2f}ms   Ratio: {v2_enc/v1_enc:.2f}x")

    print()

    assert v1_stats.total_moves > 0
    assert v2_stats.total_moves > 0
