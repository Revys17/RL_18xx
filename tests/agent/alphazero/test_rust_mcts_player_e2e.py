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


def test_policy_target_is_the_visit_distribution_after_the_softpick_cutoff():
    """Past ``softpick_move_cutoff`` the move is the argmax, but the training
    target stays the full visit distribution, not a one-hot of that move."""
    from rl18xx.agent.alphazero.loop import _create_fresh_game

    player = RustMCTSPlayer(_make_config(num_readouts=16, softpick_move_cutoff=0))
    player.initialize_game(_create_fresh_game(4))
    while player.root.N < 16:
        player.tree_search()
    visits = np.asarray(player._rust_player.child_n_at_root(), dtype=np.float32)
    assert (visits > 0).sum() > 1
    player.play_move(player.pick_move())
    pi = player.searches_pi[0]
    assert np.allclose(np.sort(pi[pi > 0]), np.sort(visits[visits > 0] / visits.sum()))


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


def test_game_result_reads_the_current_root_not_the_starting_position():
    """``advance_root`` moves the root through the arena; slot 0 stays the
    starting position. The self-play value target (``root.game_result()``)
    and the root-terminal check must read the current root — reading slot 0
    gave every game the opening's equal net-worth split as its target."""
    from rl18xx.agent.alphazero.loop import _create_fresh_game

    player = RustMCTSPlayer(_make_config(use_score_values=True))
    player.initialize_game(_create_fresh_game(4))
    start = player.root.game_result()[:4].copy()

    for _ in range(200):
        player.play_move(player.suggest_move())
        root_idx = player._rust_player.root_idx
        current = player._compute_terminal_value(root_idx)[:4]
        if not np.allclose(current, start):
            break
    else:
        pytest.fail("net-worth split never moved off the starting position in 200 moves")

    assert np.allclose(player.root.game_result()[:4], current)
    assert player._is_root_terminal() == player._rust_player.is_terminal(root_idx)


def test_self_play_games_run_the_auction_unlock_variant():
    """Self-play games (and extract_data's replay game, built the same way)
    run the engines' ``auction_unlock`` variant; it can be turned off."""
    assert RustMCTSPlayer(_make_config()).get_new_game_state()._game.auction_unlock is True
    assert RustMCTSPlayer(_make_config(auction_unlock=False)).get_new_game_state()._game.auction_unlock is False


def test_stalled_private_auction_ends_scored_with_training_data(tmp_path, monkeypatch):
    """A game still in the initial private auction after ``auction_stall_moves``
    engine moves ends there, scored on net worth, and its moves are written
    as training data like any finished game."""
    import json

    from rl18xx.agent.alphazero import self_play

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(self_play, "SELF_PLAY_GAMES_STATUS_PATH", tmp_path / "status")
    (tmp_path / "status").mkdir()

    from rl18xx.agent.alphazero.encoder import Encoder_Transformer

    class EvalDummyNet(DummyNet):
        encoder = Encoder_Transformer()

        def eval(self):
            return self

        def get_name(self):
            return "dummy"

    config = _make_config(network=EvalDummyNet(), auction_stall_moves=3, game_id="stall")
    self_play.SelfPlay(config).run_game()

    status = json.loads((tmp_path / "status" / "stall.json").read_text())
    assert status["termination"] == "auction_stall"
    assert status["status"] == "Completed"
    assert sum(status["result_per_player"]) == pytest.approx(1.0)
    assert status["auction_unlock_discounts"] == 0  # recorded for the loop's unlock rate
    assert list((tmp_path / "training_examples").rglob("data.mdb"))


def test_resignation_ends_the_game(tmp_path, monkeypatch):
    """A resign ends the game. ``end_game()`` lands on a clone of the Rust
    root, so the end-of-game block must key off the termination: before, the
    loop re-searched the resigned position forever, growing the tree until
    the OOM killer took the worker."""
    from rl18xx.agent.alphazero import self_play

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(self_play, "SELF_PLAY_GAMES_STATUS_PATH", tmp_path / "status")
    (tmp_path / "status").mkdir()
    calls = []

    def resign_on_third_search(self):
        calls.append(1)
        assert len(calls) <= 3, "kept searching after resigning"
        return len(calls) == 3, {"leader": 2, "q_leader_min": 0.9, "gap_min": 0.5}

    class EvalDummyNet(DummyNet):
        def eval(self):
            return self

    monkeypatch.setattr(RustMCTSPlayer, "check_resign", resign_on_third_search)
    player = self_play.SelfPlay(_make_config(network=EvalDummyNet(), game_id="resign")).play()
    assert player.termination == "resigned"
    assert len(calls) == 3
    assert np.any(player.result != 0)  # scored at the resigned position
    # Three moves in, everyone still holds the same net worth; the player the
    # search was confident in is the sole winner of the stored target.
    winners = np.flatnonzero(player.result >= player.result.max() - 1e-6)
    assert winners.tolist() == [2]


def test_resigned_result_makes_the_resign_leader_the_winner():
    from rl18xx.agent.alphazero.self_play import resigned_result

    shares = np.array([0.30, 0.20, 0.26, 0.24, 0.0, 0.0], dtype=np.float32)
    swapped = resigned_result(shares, leader=2)
    assert int(np.argmax(swapped)) == 2
    assert sorted(swapped.tolist()) == sorted(shares.tolist())  # same shares, reassigned
    assert swapped[0] == np.float32(0.26) and swapped[2] == np.float32(0.30)
    assert np.array_equal(resigned_result(shares, leader=0), shares)  # already the leader

    tied = resigned_result(np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32), leader=1)
    assert np.flatnonzero(tied >= tied.max() - 1e-6).tolist() == [1]
    assert abs(float(tied.sum()) - 1.0) < 1e-6


def test_no_resign_check_before_resign_min_move(tmp_path, monkeypatch):
    """An overconfident early value head must not end games before
    ``resign_min_move`` engine moves: check_resign isn't even consulted."""
    from rl18xx.agent.alphazero import self_play

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(self_play, "SELF_PLAY_GAMES_STATUS_PATH", tmp_path / "status")
    (tmp_path / "status").mkdir()

    def always_resign(self):
        raise AssertionError("check_resign called before resign_min_move")

    class EvalDummyNet(DummyNet):
        def eval(self):
            return self

    monkeypatch.setattr(RustMCTSPlayer, "check_resign", always_resign)
    config = _make_config(network=EvalDummyNet(), game_id="floor", max_game_length=30, resign_min_move=1000)
    player = self_play.SelfPlay(config).play()
    assert player.termination != "resigned"


def test_arena_keeps_only_the_current_subtree():
    """advance_root compacts the arena to the new root's subtree. Every node
    holds a full game clone, so an arena that kept the whole game's history
    grew to ~18 GB per self-play worker."""
    from rl18xx.agent.alphazero.loop import _create_fresh_game

    player = RustMCTSPlayer(_make_config(num_readouts=8))
    player.initialize_game(_create_fresh_game(4))
    sizes = []
    for _ in range(100):
        if player.is_done():
            break
        player.play_move(player.suggest_move())
        sizes.append(player._rust_player.arena_size())
        assert player._rust_player.root_idx == 0
    # Without compaction this grows by >= num_readouts nodes per move.
    assert max(sizes) <= 64, f"arena kept growing: {sizes[-5:]}"


def test_unrotate_value_rotates_only_the_real_seats():
    from rl18xx.agent.alphazero.mcts import unrotate_value

    canonical = np.array([0.4, 0.3, 0.2, 0.1, 0.0, 0.0], dtype=np.float32)  # 4 players, active = seat 2
    absolute = unrotate_value(canonical, rotation=2, num_players=4)
    assert np.allclose(absolute, [0.2, 0.1, 0.4, 0.3, 0.0, 0.0])


def test_terminal_backup_value_is_share_of_winners():
    """Finished games back up on the win-loss head's scale (what the network
    returns for every other leaf), not as raw net-worth fractions."""
    from rl18xx.agent.alphazero.mcts import terminal_backup_value

    fractions = np.array([0.30, 0.25, 0.40, 0.05, 0.0, 0.0], dtype=np.float32)
    assert np.allclose(terminal_backup_value(fractions, 4), [0, 0, 1, 0, 0, 0])
    tied = np.array([0.4, 0.4, 0.2, 0.0, 0.0, 0.0], dtype=np.float32)
    assert np.allclose(terminal_backup_value(tied, 3), [0.5, 0.5, 0, 0, 0, 0])


def test_network_values_are_backed_up_in_absolute_seat_order():
    """The network's value is in the leaf's canonical frame (seat 0 = the
    leaf's active player). A net that always says "the player to move wins"
    must credit the leaves' actual movers — at a fresh game's root, P1 moves
    and the evaluated children mostly have P2 (seat 1) to move."""
    from rl18xx.agent.alphazero.loop import _create_fresh_game

    class MoverWinsNet(DummyNet):
        def run_many_encoded(self, encoded_game_states):
            probs, log_probs, _ = super().run_many_encoded(encoded_game_states)
            value = torch.zeros(VALUE_SIZE, dtype=torch.float32)
            value[0] = 1.0
            return probs, log_probs, [value] * len(encoded_game_states)

    player = RustMCTSPlayer(_make_config(network=MoverWinsNet(), num_readouts=16, parallel_readouts=4))
    player.initialize_game(_create_fresh_game(4))
    for _ in range(4):
        player.tree_search()
    q = np.asarray(player._rust_player.root_q_vector())
    assert q.argmax() == 1, f"root value by seat: {q}"


def test_net_worth_leaf_heuristic_is_a_softmax_over_net_worth():
    player = RustMCTSPlayer(_make_config(leaf_value_heuristic="net_worth", leaf_value_heuristic_scale=100.0))
    game = player.get_game_state()
    value = player._heuristic_leaf_value(game)  # everyone starts with the same net worth
    np.testing.assert_allclose(value[:4], 0.25, atol=1e-6)
    assert np.all(value[4:] == 0)

    class Ahead:
        """Player 2 is $100 ahead of the rest."""

        def result(self):
            return {1: 600, 2: 700, 3: 600, 4: 600}

    value = player._heuristic_leaf_value(Ahead())
    np.testing.assert_allclose(value[1] / value[0], np.e, rtol=1e-5)
    assert abs(float(value.sum()) - 1.0) < 1e-6
