"""Phase 4c parity tests for the Rust MCTS progressive-widening path.

Drives the Rust MCTS through readouts that traverse price-bearing slots
(initial auction Bids) with a deterministic ``DummyNet`` and verifies:

  1. Categorical-level child_N totals roughly match the Python MCTS path.
  2. After enough readouts a PW slot has grown >1 grandchildren with prices
     in the legal range, snapped to the right grid (Bid → $5 ticks).
"""
from __future__ import annotations

import math
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


def _zero_price_components(num_slots: int) -> dict:
    """Build a price_components dict with zero μ / log σ for every slot.

    This is what ``DummyNet`` would produce — a head with degenerate
    (deterministic) output. The slot_index map mirrors
    ``ContinuousPriceHead.slot_index``: ``(type, entity_key_tuple) -> int``.
    """
    # Mirror the head's slot layout from
    # ``model_transformer.ContinuousPriceHead.__init__``.
    companies = ("SV", "CS", "DH", "MH", "CA", "BO")
    corporations = ("PRR", "NYC", "CPR", "B&O", "C&O", "ERIE", "NYNH", "B&M")
    train_types = ("2", "3", "4", "5", "6", "D")
    slot_index: dict = {}
    i = 0
    for c in companies:
        slot_index[("Bid", (c,))] = i
        i += 1
    for corp in corporations:
        for t in train_types:
            slot_index[("BuyTrain", (corp, t))] = i
            i += 1
    for c in companies:
        slot_index[("BuyCompany", (c,))] = i
        i += 1
    n = i
    return {
        "price_mean": np.zeros(n, dtype=np.float32),
        "price_log_std": np.zeros(n, dtype=np.float32),
        "slot_index": slot_index,
        "num_slots": n,
    }


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
    price_components_py = _zero_price_components(0)
    # For Python, MCTSNode.incorporate_results expects torch tensors for the
    # head outputs; np float32 is sufficient (it reads via float(...)).
    price_components_py = {
        "price_mean": price_components_py["price_mean"],
        "price_log_std": price_components_py["price_log_std"],
        "slot_index": price_components_py["slot_index"],
        "num_slots": price_components_py["num_slots"],
    }
    price_components_rs = _zero_price_components(0)  # numpy dict for Rust

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
def test_pw_grows_multiple_grandchildren_with_legal_snapped_prices():
    """After enough readouts on a price-bearing slot, the Rust side has more
    than one grandchild under that slot — sampled prices are in the legal
    range and snapped to the right grid (Bid → $5 ticks).

    We strongly bias the prior so the search visits a single Bid slot (CS,
    index 2) many times — that's the slot we'll inspect for grandchildren.
    """
    game_rs = _fresh_game()
    cfg = _mcts_config(pw_c=2.0, pw_alpha=0.7, min_price_children=1)
    rs_player = RustMCTSPlayer(
        game_rs, cfg.pw_c, cfg.pw_alpha, cfg.min_price_children
    )
    legal = rs_player.legal_action_indices_at_root()
    # Confirm slot 2 is a PW Bid slot.
    am = ActionMapper()
    indices, price_ranges, action_types = am.get_legal_actions_factored(
        RustGameAdapter(RustBaseGame({i + 1: f"P{i + 1}" for i in range(PLAYER_COUNT)}))
    )
    target_slot = next(
        i for i in indices
        if action_types.get(i) == "Bid"
        and price_ranges.get(i) is not None
        and price_ranges[i][0] != price_ranges[i][1]
    )
    p_min, p_max = price_ranges[target_slot]
    assert p_min != p_max  # sanity

    # Spike-prior on ``target_slot`` and almost zero elsewhere so PUCT
    # routes almost every readout there.
    probs = np.full(POLICY_SIZE, 1e-9, dtype=np.float32)
    probs[target_slot] = 1.0
    probs /= probs.sum()
    zero_value = np.zeros(VALUE_SIZE, dtype=np.float32)
    price_components = _zero_price_components(0)

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
    # All prices in legal range + snapped to $5 (Bid grid).
    for p in prices_visited:
        assert p_min <= p <= p_max, (
            f"PW price {p} outside legal range [{p_min}, {p_max}] for slot {target_slot}"
        )
        assert p % 5 == 0, (
            f"PW price {p} not snapped to $5 grid for Bid slot {target_slot}"
        )

    # ``most_visited_price_for_slot`` should return one of the visited prices.
    best_price = rs_player.most_visited_price_for_slot(target_slot)
    assert best_price in prices_visited, (
        f"most_visited_price_for_slot returned {best_price}, not in {prices_visited}"
    )
