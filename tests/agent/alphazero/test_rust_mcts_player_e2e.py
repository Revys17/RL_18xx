"""Phase 4b end-to-end smoke test for the Rust-MCTS-backed adapter.

Constructs a ``RustMCTSPlayer`` (Python adapter around
``engine_rs.RustMCTSPlayer``), runs a few ``tree_search`` cycles, plays
moves, drives the player through a short stub game, and verifies that
``extract_data`` produces tuples of the expected shape.

Scope (4b): categorical descent only. PW + continuous prices land in 4c.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from rl18xx.agent.alphazero.config import SelfPlayConfig
from rl18xx.agent.alphazero.mcts import POLICY_SIZE, VALUE_SIZE

pytest.importorskip("engine_rs")
from rl18xx.agent.alphazero.rust_mcts_player import RustMCTSPlayer  # noqa: E402


class DummyNet:
    """Drop-in stub for the AlphaZero model that emits uniform prior + zero value."""

    def encoder_type(self) -> str:
        return "GNN"

    def run_encoded(self, encoded_game_state):
        priors = torch.ones(POLICY_SIZE, dtype=torch.float32) / POLICY_SIZE
        value = torch.zeros(VALUE_SIZE, dtype=torch.float32)
        return priors, torch.log(priors), value

    def run_many_encoded(self, encoded_game_states):
        n = len(encoded_game_states)
        priors = torch.ones(POLICY_SIZE, dtype=torch.float32) / POLICY_SIZE
        value = torch.zeros(VALUE_SIZE, dtype=torch.float32)
        return [priors] * n, [torch.log(priors)] * n, [value] * n


def _make_config(**overrides) -> SelfPlayConfig:
    defaults = dict(
        network=DummyNet(),
        use_rust_mcts=True,
        num_readouts=8,
        parallel_readouts=2,
        min_readouts=2,
        backup_discount=1.0,
        use_score_values=False,
        use_fp16_inference=False,
        # Force 4-player games so the test is deterministic across runs.
        player_count_distribution={4: 1.0},
        # Resign is disabled on the Rust path anyway; flip the flag off so
        # check_resign() returns quickly without logging warnings.
        enable_resign=False,
        # Phase 4b: keep PW disabled — categorical only.
        min_price_children=1,
        # Phase 1 tracing off.
        # No tracing -> default TraceConfig with trace_game_rate=0.
        dirichlet_noise_weight=0.0,
    )
    defaults.update(overrides)
    return SelfPlayConfig(**defaults)


def test_rust_mcts_player_short_loop_no_crash():
    """Three ``tree_search`` cycles + a ``pick_move`` + ``play_move`` round trip.

    Just ensures the basic Rust-backed flow works without exceptions and
    that the visit counts on the root accumulate as expected.
    """
    config = _make_config()
    player = RustMCTSPlayer(config)

    n_before = player._rust_player.n_at_root()
    assert n_before == 0.0

    # Drive a few tree-search cycles. Each one runs ``parallel_readouts``
    # leaves through the DummyNet.
    for _ in range(3):
        player.tree_search()

    # The root's visit count should reflect the cycles (each adds at most
    # ``parallel_readouts`` visits; some cycles may collapse on a terminal
    # leaf that backs up directly).
    n_after = player._rust_player.n_at_root()
    assert n_after > 0, "tree_search should have accumulated visits at root"

    # pick_move + play_move should succeed.
    move = player.pick_move()
    assert isinstance(move, int)
    legal_before = player._rust_player.legal_action_indices_at_root()
    assert move in legal_before

    ok = player.play_move(move)
    assert ok is True
    # After play_move, the new root's legal actions may differ.
    legal_after = player._rust_player.legal_action_indices_at_root()
    assert len(legal_after) > 0

    # searches_pi recorded exactly one policy vector per play_move.
    assert len(player.searches_pi) == 1
    pi = player.searches_pi[0]
    assert pi.shape == (POLICY_SIZE,)
    assert abs(pi.sum() - 1.0) < 1e-4 or pi.sum() == 0.0


def test_rust_mcts_player_short_game_extracts_data():
    """Drive ~30 moves through the SelfPlay.play() pattern and verify
    extract_data yields at least one tuple of the expected shape."""
    # max_game_length=30 so the engine truncates quickly enough for CI.
    config = _make_config(num_readouts=4, parallel_readouts=2, max_game_length=30)
    player = RustMCTSPlayer(config)

    # Mimic SelfPlay.play()'s preamble: expand the root via the leaf shim.
    root = player.get_root()
    first_node = root.select_leaf()
    first_node.ensure_encoded()
    probs, _, val = config.network.run_encoded(first_node.encoded_game_state)
    first_node.incorporate_results(probs, val, first_node)

    # Game loop modeled after SelfPlay.play() (no resign hook on Rust path).
    move_counter = 0
    while not player.is_done() and move_counter < 30:
        if root.num_legal_actions == 1:
            move = player.pick_move()
        else:
            current = root.N
            target = current + player.adaptive_readouts()
            while root.N < target:
                player.tree_search()
            move = player.pick_move()
        player.play_move(move)
        move_counter += 1

    # Truncated or naturally finished — either way, set_result and extract_data.
    from rl18xx.rust_adapter import RustGameAdapter

    if not player._is_root_terminal():
        adapter = RustGameAdapter(player._rust_player.root_game_object())
        adapter.end_game()
        player.termination = "max_length"

    # Use the terminal value of the (now-finished) root as the result. The
    # truncated game's net-worth is all-equal (every player has the same
    # starting cash and no shares traded yet in the early game), so the
    # value vector is uniform — that's still a valid non-zero result
    # provided we tweak it to have a unique winner so the assertion in
    # ``extract_data`` (result != zeros) holds.
    num_players = len(player._rust_player.root_game_object().players)
    result_vec = np.ones(num_players, dtype=np.float32) / num_players
    # Bump player 0's slot so the result vector is asymmetric and clearly
    # non-zero per the assertion in ``extract_data``.
    result_vec[0] = result_vec[0] + 0.01
    player.set_result(result_vec)

    tuples = list(player.extract_data())
    assert len(tuples) >= 1, "extract_data should yield at least one tuple"

    sample_game, legal, pi, result, price_targets = tuples[0]
    assert hasattr(sample_game, "raw_actions"), "first element must be a game state"
    assert isinstance(legal, torch.Tensor)
    assert isinstance(pi, torch.Tensor)
    assert pi.shape[-1] == POLICY_SIZE
    assert isinstance(result, torch.Tensor)
    # price_targets is empty unless the move chosen for this state was a
    # price-bearing slot. With softpick (visit-proportional sampling below
    # softpick_move_cutoff, parity with the Python player) the first move CAN
    # be e.g. a Bid, so assert structure rather than emptiness: entries are
    # (head_slot, price, visit_weight, range_min, range_max).
    assert isinstance(price_targets, list)
    for entry in price_targets:
        assert len(entry) == 5
        _slot, price, weight, pmin, pmax = entry
        assert pmin <= price <= pmax
        assert weight > 0


def test_rust_mcts_player_check_resign_window_not_full():
    """With an empty rolling window, ``check_resign`` cannot fire — should
    return ``(False, None)`` regardless of the (here-default zero) Q vector."""
    config = _make_config(enable_resign=True, resign_window=3)
    player = RustMCTSPlayer(config)
    should, info = player.check_resign()
    assert should is False
    assert info is None


def test_rust_mcts_player_check_resign_disabled_when_flag_off():
    """When ``enable_resign=False`` the adapter must never resign nor record."""
    config = _make_config(enable_resign=False)
    player = RustMCTSPlayer(config)
    should, info = player.check_resign()
    assert should is False
    assert info is None
