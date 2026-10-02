"""Unit tests for MCTS price proposals on price-bearing slots.

Progressive widening materializes a price-bearing slot's grandchildren one
price cell at a time (``price_pmf.proposal_order``): the price head's most
likely cell first, then the rest by Gumbel-top-k with an exploration floor,
each at a uniform price inside its cell. These tests drive the
``MCTSNode`` helpers directly on the initial 1830 state (whose legal actions
are all auction bids) without running a search.
"""
from __future__ import annotations

import numpy as np
import pytest

from rl18xx.agent.alphazero import price_pmf
from rl18xx.agent.alphazero.config import SelfPlayConfig
from rl18xx.agent.alphazero.mcts import MCTSNode
from rl18xx.game.gamemap import GameMap


def _fresh_root_node() -> MCTSNode:
    game_class = GameMap().game_by_title("1830")
    game = game_class({1: "P1", 2: "P2", 3: "P3", 4: "P4"})
    node = MCTSNode(game, config=SelfPlayConfig())
    node._pw_rng = np.random.default_rng(12345)
    return node


def _bid_slot(node: MCTSNode):
    """A price-bearing Bid slot at the initial state: (index, type, range)."""
    for idx in node.legal_action_indices:
        pr = node.price_ranges_by_idx.get(idx)
        if pr is not None and pr[0] != pr[1] and node.action_types_by_idx[idx] == "Bid":
            return idx, "Bid", pr
    pytest.fail("no price-bearing Bid slot at the initial state")


def _attach_logits(node: MCTSNode, action_index: int, action_type: str, cell_logits: np.ndarray):
    logits = np.zeros((price_pmf.NUM_SLOTS, price_pmf.NUM_CELLS), dtype=np.float32)
    slot = price_pmf.SLOT_INDEX[(action_type, node._price_head_entity_key(action_index))]
    logits[slot] = cell_logits
    node.price_components = {
        "price_logits": logits,
        "slot_index": price_pmf.SLOT_INDEX,
        "num_slots": price_pmf.NUM_SLOTS,
    }


def test_without_price_head_any_legal_price_can_come_out():
    """No ``price_components`` (GNN model / before ``incorporate_results``):
    cells are uniform, prices stay legal, and off-ladder bids are reachable."""
    node = _fresh_root_node()
    idx, action_type, (lo, hi) = _bid_slot(node)
    prices = [node._sample_price_for_slot(idx, action_type, (lo, hi)) for _ in range(400)]
    assert all(lo <= p <= hi for p in prices)
    assert any((p - lo) % 5 for p in prices), "the residual cell should yield off-ladder bids"


def test_degenerate_range_returns_exact_price():
    node = _fresh_root_node()
    for action_type in ("Bid", "BuyTrain", "BuyCompany"):
        assert node._sample_price_for_slot(0, action_type, (123, 123)) == 123


def test_first_proposal_is_the_heads_most_likely_cell():
    node = _fresh_root_node()
    idx, action_type, (lo, hi) = _bid_slot(node)
    cell_logits = np.zeros(price_pmf.NUM_CELLS)
    cell_logits[2] = 6.0  # min bid + $10
    _attach_logits(node, idx, action_type, cell_logits)
    assert node._next_proposed_price(idx, action_type, (lo, hi)) == lo + 10


def test_proposals_visit_each_nonempty_cell_once_then_stop():
    """Widening never proposes two prices in one cell, reaches every non-empty
    cell (the exploration floor), and returns ``None`` once all are taken."""
    node = _fresh_root_node()
    idx, action_type, (lo, hi) = _bid_slot(node)
    cell_logits = np.full(price_pmf.NUM_CELLS, -30.0)
    cell_logits[0] = 10.0  # a confidently peaked head
    _attach_logits(node, idx, action_type, cell_logits)
    cells = price_pmf.price_cells(action_type, lo, hi)
    seen = []
    while True:
        price = node._next_proposed_price(idx, action_type, (lo, hi))
        if price is None:
            break
        assert lo <= price <= hi
        seen.append(cells.cell_of(price))
        # Register a grandchild at that price the way maybe_add_child does.
        node.price_children.setdefault(idx, {})[price] = object()
    assert seen[0] == 0
    assert sorted(seen) == np.flatnonzero(cells.nonempty).tolist()


def test_non_finite_price_head_still_yields_legal_prices():
    node = _fresh_root_node()
    idx, action_type, (lo, hi) = _bid_slot(node)
    _attach_logits(node, idx, action_type, np.full(price_pmf.NUM_CELLS, np.nan))
    for _ in range(50):
        assert lo <= node._sample_price_for_slot(idx, action_type, (lo, hi)) <= hi
    assert lo <= node._next_proposed_price(idx, action_type, (lo, hi)) <= hi


def test_maybe_add_child_plays_an_explicit_off_ladder_price():
    """Any legal price is playable: an explicit price is used verbatim (not
    snapped to the $5 ladder) and becomes that grandchild's key."""
    node = _fresh_root_node()
    idx, _, (lo, hi) = _bid_slot(node)
    price = lo + 7
    child = node.maybe_add_child(idx, price=price)
    assert child.sampled_price == price
    assert price in node.price_children[idx]
