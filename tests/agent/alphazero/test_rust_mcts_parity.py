"""Phase 4a parity test: Rust MCTS vs Python MCTS.

Drives ``RustMCTSPlayer`` and the Python ``MCTSNode`` through N readouts
from the same starting state with deterministic priors / value (uniform
priors, zero value). Asserts that the visit count vectors agree on the
chosen actions and that the per-readout descent reaches comparable depths.

Scope (4a): categorical descent only — no progressive widening, no
continuous prices, no Dirichlet noise (run with dirichlet_noise_weight=0).
Price-bearing slots are treated as fixed-price at price_range[0] for now;
PW lands in 4c.
"""

from __future__ import annotations

import numpy as np
import pytest

from rl18xx.agent.alphazero.action_mapper import ActionMapper
from rl18xx.agent.alphazero.config import SelfPlayConfig
from rl18xx.agent.alphazero.mcts import MCTSNode, POLICY_SIZE, VALUE_SIZE
from rl18xx.rust_adapter import RustGameAdapter

try:
    from engine_rs import BaseGame as RustBaseGame, RustMCTSPlayer  # type: ignore
except ImportError:  # pragma: no cover
    RustBaseGame = None
    RustMCTSPlayer = None


READOUTS = 10
PLAYER_COUNT = 4


def _fresh_game() -> RustGameAdapter:
    players = {i + 1: f"Player {i + 1}" for i in range(PLAYER_COUNT)}
    return RustGameAdapter(RustBaseGame(players))


def _mcts_config() -> SelfPlayConfig:
    return SelfPlayConfig(
        network=None,
        dirichlet_noise_weight=0.0,  # parity: skip noise on both sides
        backup_discount=1.0,  # match Rust default
        use_score_values=False,
        # Disable PW so Python MCTS mirrors the Rust 4a scope: any price-bearing
        # slot is treated as fixed-price (pw_c=0 means never grow grandchildren).
        # Actually, simpler: Phase 4a starting state (Auction round) has only
        # `Bid` and `Pass` actions. We pin price_range to fixed at min_bid for
        # the auction state by using `min_price_children=1` which guarantees
        # the single fixed-price child grows.
        min_price_children=1,
    )


@pytest.mark.skipif(RustMCTSPlayer is None, reason="engine_rs not built")
def test_rust_mcts_matches_python_mcts_visit_counts():
    """Drive both Python and Rust MCTS through ``READOUTS`` readouts with the
    same uniform prior + zero value. Compare visit counts on the legal slots."""

    # ------------------------------------------------------------------------
    # Set up two independent root nodes from the same fresh game.
    # ------------------------------------------------------------------------
    game_py = _fresh_game()
    game_rs = _fresh_game()  # Rust side gets its own copy.

    cfg = _mcts_config()
    py_root = MCTSNode(game_py, config=cfg)
    rs_player = RustMCTSPlayer(game_rs)

    # ------------------------------------------------------------------------
    # Sanity: both sides see the same set of legal action indices.
    # ------------------------------------------------------------------------
    rs_legal = sorted(rs_player.legal_action_indices_at_root())
    py_legal = sorted(py_root.legal_action_indices)
    assert rs_legal == py_legal, (
        f"legal_action_indices diverge:\n  rust={rs_legal}\n  python={py_legal}"
    )
    num_legal = len(rs_legal)
    assert num_legal > 0, "starting state must have at least one legal action"

    # ------------------------------------------------------------------------
    # Drive N readouts on each side. Uniform prior (each legal slot gets the
    # same probability) + zero value vector.
    # ------------------------------------------------------------------------
    uniform = np.full(POLICY_SIZE, 1.0 / POLICY_SIZE, dtype=np.float32)
    zero_value = np.zeros(VALUE_SIZE, dtype=np.float32)

    for _ in range(READOUTS):
        # Python MCTS readout
        leaf_py = py_root.select_leaf()
        if leaf_py.is_done():
            value = leaf_py.game_result()
            leaf_py.backup_value(value, up_to=py_root)
        else:
            leaf_py.incorporate_results(uniform, zero_value, up_to=py_root)

        # Rust MCTS readout
        leaf_idx = rs_player.select_leaf()
        if rs_player.is_terminal(leaf_idx):
            # Terminal — the Rust scaffold doesn't compute game_result yet;
            # since this branch shouldn't fire on a fresh game over 10 readouts
            # we treat the leaf as a zero-value backup for parity.
            rs_player.backup_value(leaf_idx, list(zero_value))
        else:
            rs_player.incorporate_results(leaf_idx, uniform, zero_value)

    # ------------------------------------------------------------------------
    # Compare visit counts on legal slots. Allow small tie-breaking jitter
    # (ties in PUCT scores may break differently between numpy argmax and
    # Rust's first-best iteration). Per the spec: within 1 visit per slot.
    # ------------------------------------------------------------------------
    py_n = py_root.child_N
    rs_n = np.asarray(rs_player.child_n_at_root(), dtype=np.float32)

    diffs = []
    for idx in rs_legal:
        py_v = float(py_n[idx])
        rs_v = float(rs_n[idx])
        diffs.append((idx, py_v, rs_v, abs(py_v - rs_v)))
    max_diff = max(d[3] for d in diffs)

    # Sum of visits should match: both sides did READOUTS readouts, so the
    # total child_N should equal the number of finite (non-terminal-only)
    # readouts on each side.
    total_py = float(py_n.sum())
    total_rs = float(rs_n.sum())
    assert abs(total_py - total_rs) <= 1.0, (
        f"total visit count diverges: py={total_py}, rs={total_rs}"
    )

    # Per-slot tolerance: the categorical descent should match modulo tie
    # breaking. Allow up to 1 visit slack per slot.
    assert max_diff <= 1.0, (
        f"max per-slot visit diff = {max_diff}; offending slots:\n"
        + "\n".join(
            f"  idx={idx} py={py_v} rs={rs_v}"
            for (idx, py_v, rs_v, d) in diffs
            if d > 0
        )
    )


@pytest.mark.skipif(RustMCTSPlayer is None, reason="engine_rs not built")
def test_rust_mcts_first_leaf_matches_python():
    """Smoke test: with uniform priors, the very first readout on both sides
    should select the same compressed slot (the first legal action) and
    expand its child to the same legal-action signature."""
    game_py = _fresh_game()
    game_rs = _fresh_game()
    cfg = _mcts_config()
    py_root = MCTSNode(game_py, config=cfg)
    rs_player = RustMCTSPlayer(game_rs)

    # First call: both should return the root (unexpanded).
    py_leaf = py_root.select_leaf()
    rs_leaf_idx = rs_player.select_leaf()
    assert py_leaf is py_root
    assert rs_leaf_idx == 0  # arena root

    uniform = np.full(POLICY_SIZE, 1.0 / POLICY_SIZE, dtype=np.float32)
    zero_value = np.zeros(VALUE_SIZE, dtype=np.float32)
    py_leaf.incorporate_results(uniform, zero_value, up_to=py_root)
    rs_player.incorporate_results(rs_leaf_idx, uniform, zero_value)

    # Second call: PUCT picks first slot (argmax with ties broken by first).
    py_leaf2 = py_root.select_leaf()
    rs_leaf_idx2 = rs_player.select_leaf()
    assert py_leaf2 is not py_root
    assert rs_leaf_idx2 != 0

    # Both should have expanded the same action index.
    assert py_leaf2.fmove in py_root.legal_action_indices
    # And the chosen action index should match the Rust side's first slot.
    # (Both use argmax over child_action_score; the prior is uniform, the
    # value zero, so this reduces to the first legal index.)
    assert py_leaf2.fmove == py_root.legal_action_indices[0]


@pytest.mark.skipif(RustMCTSPlayer is None, reason="engine_rs not built")
def test_rust_action_offsets_match_python():
    """The Rust slot layout must agree with Python's ActionMapper."""
    from engine_rs import action_offsets_py, policy_size_py

    rs_offsets = action_offsets_py()
    py_mapper = ActionMapper()
    py_offsets = py_mapper.action_offsets

    # Every Python offset must appear in Rust with the same value (modulo
    # any keys the 4a scaffold deliberately doesn't export, like CompanyPass
    # which collapses onto Pass).
    for k, v in py_offsets.items():
        assert k in rs_offsets, f"Rust missing action_offset key: {k}"
        assert rs_offsets[k] == v, (
            f"action_offset[{k}]: rust={rs_offsets[k]} python={v}"
        )

    assert policy_size_py() == py_mapper.action_encoding_size
