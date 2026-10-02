"""Phase 4c parity tests for the Rust MCTS progressive-widening path.

Drives the Rust MCTS through readouts that traverse price-bearing slots
(initial auction Bids) with a deterministic ``DummyNet`` and verifies:

  1. Categorical-level child_N totals roughly match the Python MCTS path.
  2. After enough readouts a PW slot has grown >1 grandchildren with prices
     in the legal range, one per price cell, the price head's most likely
     cell among them.
  3. Both engines' first price proposal is the head's most likely cell.
"""
from __future__ import annotations

import math
import numpy as np
import pytest

from rl18xx.agent.alphazero.action_mapper import ActionMapper
from rl18xx.agent.alphazero.config import SelfPlayConfig
from rl18xx.agent.alphazero import price_pmf
from rl18xx.agent.alphazero.mcts import MCTSNode, POLICY_SIZE, VALUE_SIZE
from rl18xx.rust_adapter import RustGameAdapter

try:
    from engine_rs import BaseGame as RustBaseGame, RustMCTSPlayer  # type: ignore
except ImportError:  # pragma: no cover
    RustBaseGame = None
    RustMCTSPlayer = None


PLAYER_COUNT = 4


def _fresh_game() -> RustGameAdapter:
    players = {i + 1: f"Player {i + 1}" for i in range(PLAYER_COUNT)}
    return RustGameAdapter(RustBaseGame(players))


def _mcts_config(**overrides) -> SelfPlayConfig:
    defaults = dict(
        network=None,
        dirichlet_noise_weight=0.0,
        backup_discount=1.0,
        use_score_values=False,
        # PW knobs: aggressive growth so 30 readouts grows >1 grandchild.
        pw_c=1.0,
        pw_alpha=0.5,
        min_price_children=1,
    )
    defaults.update(overrides)
    return SelfPlayConfig(**defaults)


def _zero_price_components(num_slots: int = 0) -> dict:
    """``price_components`` with all-zero cell logits (uniform price cells)
    for every slot, in the price head's slot layout (``price_pmf.SLOTS``)."""
    return {
        "price_logits": np.zeros((price_pmf.NUM_SLOTS, price_pmf.NUM_CELLS), dtype=np.float32),
        "slot_index": dict(price_pmf.SLOT_INDEX),
        "num_slots": price_pmf.NUM_SLOTS,
    }


def _target_bid_slot():
    """The first price-bearing Bid slot at the initial state, with its range."""
    am = ActionMapper()
    indices, price_ranges, action_types = am.get_legal_actions_factored(_fresh_game())
    target = next(
        i for i in indices
        if action_types.get(i) == "Bid" and price_ranges.get(i) is not None and price_ranges[i][0] != price_ranges[i][1]
    )
    return target, price_ranges[target]


def _bid_components_with_mode(target_slot: int, mode_cell: int) -> dict:
    """Zero logits except a confident ``mode_cell`` for ``target_slot``'s company."""
    pc = _zero_price_components()
    company = ActionMapper().actions[target_slot][1][0]
    pc["price_logits"][price_pmf.SLOT_INDEX[("Bid", (company,))], mode_cell] = 5.0
    return pc


@pytest.mark.skipif(RustMCTSPlayer is None, reason="engine_rs not built")
def test_pw_visit_count_totals_match_python_within_tolerance():
    """Drive Python and Rust MCTS through 30 readouts on the initial auction
    state. Assert that the categorical-level child_N totals agree closely.

    The PW path samples prices stochastically, so we cannot assert per-slot
    equality. Instead we assert that:
      - the *total* visits at the root match (both ran the same #readouts).
      - the per-slot diff stays within K = readouts / 10, mirroring the
        relaxed-tolerance escape clause in the Phase 4c spec.
    """
    readouts = 30
    K_TOLERANCE = max(1, readouts // 10)  # K = readouts/10 per the spec

    game_py = _fresh_game()
    game_rs = _fresh_game()
    cfg = _mcts_config()

    py_root = MCTSNode(game_py, config=cfg)
    rs_player = RustMCTSPlayer(
        game_rs, cfg.pw_c, cfg.pw_alpha, cfg.min_price_children
    )

    # Sanity: both sides see the same legal slots.
    assert sorted(rs_player.legal_action_indices_at_root()) == sorted(
        py_root.legal_action_indices
    )

    # Uniform prior + zero value + zero price components on both sides.
    uniform = np.full(POLICY_SIZE, 1.0 / POLICY_SIZE, dtype=np.float32)
    zero_value = np.zeros(VALUE_SIZE, dtype=np.float32)
    price_components_py = _zero_price_components()
    price_components_rs = _zero_price_components()  # numpy dict for Rust

    for _ in range(readouts):
        # Python MCTS readout
        leaf_py = py_root.select_leaf()
        if leaf_py.is_done():
            value = leaf_py.game_result()
            leaf_py.backup_value(value, up_to=py_root)
        else:
            leaf_py.incorporate_results(
                uniform, zero_value, up_to=py_root, price_components=price_components_py
            )

        # Rust MCTS readout
        leaf_idx = rs_player.select_leaf()
        if rs_player.is_terminal(leaf_idx):
            rs_player.backup_value(leaf_idx, list(zero_value))
        else:
            rs_player.incorporate_results(
                leaf_idx, uniform, zero_value, price_components_rs
            )

    py_n = py_root.child_N
    rs_n = np.asarray(rs_player.child_n_at_root(), dtype=np.float32)
    total_py = float(py_n.sum())
    total_rs = float(rs_n.sum())

    # Both sides should have advanced exactly ``readouts`` non-terminal
    # incorporations (the initial state is far from terminal in 30 readouts).
    assert abs(total_py - total_rs) <= 1.0, (
        f"total visits diverge too far: py={total_py}, rs={total_rs}"
    )

    diffs = []
    for idx in rs_player.legal_action_indices_at_root():
        py_v = float(py_n[idx])
        rs_v = float(rs_n[idx])
        diffs.append((idx, py_v, rs_v, abs(py_v - rs_v)))
    max_diff = max(d[3] for d in diffs)
    # Relaxed tolerance (PW sampling is stochastic; the per-slot routing
    # diverges where the price-grandchild expansion fires).
    assert max_diff <= K_TOLERANCE, (
        f"max per-slot diff = {max_diff} > K={K_TOLERANCE}; "
        + "\n".join(f"  idx={i} py={p} rs={r}" for (i, p, r, _) in diffs)
    )


@pytest.mark.skipif(RustMCTSPlayer is None, reason="engine_rs not built")
def test_pw_grows_one_grandchild_per_price_cell():
    """After enough readouts on a price-bearing slot, the Rust side has more
    than one grandchild under that slot: legal prices, each in its own price
    cell, the head's most likely cell (min bid + $15 here) among them.

    We strongly bias the prior so the search visits a single Bid slot many
    times — that's the slot we'll inspect for grandchildren.
    """
    game_rs = _fresh_game()
    cfg = _mcts_config(pw_c=2.0, pw_alpha=0.7, min_price_children=1)
    rs_player = RustMCTSPlayer(
        game_rs, cfg.pw_c, cfg.pw_alpha, cfg.min_price_children
    )
    target_slot, (p_min, p_max) = _target_bid_slot()

    # Spike-prior on ``target_slot`` and almost zero elsewhere so PUCT
    # routes almost every readout there.
    probs = np.full(POLICY_SIZE, 1e-9, dtype=np.float32)
    probs[target_slot] = 1.0
    probs /= probs.sum()
    zero_value = np.zeros(VALUE_SIZE, dtype=np.float32)
    price_components = _bid_components_with_mode(target_slot, mode_cell=3)

    # Drive ~50 readouts and let PW grow grandchildren.
    READOUTS = 50
    for _ in range(READOUTS):
        idx = rs_player.select_leaf()
        if rs_player.is_terminal(idx):
            rs_player.backup_value(idx, list(zero_value))
        else:
            rs_player.incorporate_results(idx, probs, zero_value, price_components)

    grand = rs_player.price_grandchildren_at_root()
    assert target_slot in grand, (
        f"PW slot {target_slot} did not grow any grandchildren; map={grand}"
    )
    prices_visited = list(grand[target_slot].keys())
    assert len(prices_visited) > 1, (
        f"PW slot {target_slot}: expected >1 grandchildren, got {prices_visited}"
    )
    cells = price_pmf.price_cells("Bid", p_min, p_max)
    for p in prices_visited:
        assert p_min <= p <= p_max, (
            f"PW price {p} outside legal range [{p_min}, {p_max}] for slot {target_slot}"
        )
    assert len({cells.cell_of(p) for p in prices_visited}) == len(prices_visited), "two grandchildren share a cell"
    assert p_min + 15 in prices_visited, "the head's most likely cell was never proposed"

    # ``most_visited_price_for_slot`` should return one of the visited prices.
    best_price = rs_player.most_visited_price_for_slot(target_slot)
    assert best_price in prices_visited, (
        f"most_visited_price_for_slot returned {best_price}, not in {prices_visited}"
    )


@pytest.mark.skipif(RustMCTSPlayer is None, reason="engine_rs not built")
def test_first_price_proposal_is_the_heads_mode_in_both_engines():
    """Readout 1 expands the root; readout 2 descends into the spiked Bid
    slot and widens it once — with the price head's most likely cell, a
    ladder atom, so both engines pick the identical price."""
    target_slot, (p_min, _) = _target_bid_slot()
    cfg = _mcts_config()
    probs = np.full(POLICY_SIZE, 1e-9, dtype=np.float32)
    probs[target_slot] = 1.0
    probs /= probs.sum()
    zero_value = np.zeros(VALUE_SIZE, dtype=np.float32)
    pc = _bid_components_with_mode(target_slot, mode_cell=3)

    py_root = MCTSNode(_fresh_game(), config=cfg)
    rs_player = RustMCTSPlayer(_fresh_game(), cfg.pw_c, cfg.pw_alpha, cfg.min_price_children)
    for _ in range(2):
        py_root.select_leaf().incorporate_results(probs, zero_value, up_to=py_root, price_components=pc)
        rs_player.incorporate_results(rs_player.select_leaf(), probs, zero_value, pc)

    assert list(py_root.price_children[target_slot]) == [p_min + 15]
    assert list(rs_player.price_grandchildren_at_root()[target_slot]) == [p_min + 15]
