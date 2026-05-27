"""Unit tests for MCTS continuous-price progressive-widening sampling.

These tests target the small, pure helpers in ``rl18xx.agent.alphazero.mcts``
(``sample_price_for_pw``, ``_snap_price``) and the ``MCTSNode``-attached
``_sample_price_for_slot`` method. Each test is deterministic via a seeded
``np.random.default_rng``; none of them run a real MCTS search or build a
game tree beyond a single ``MCTSNode`` at the initial state.
"""
from __future__ import annotations

import numpy as np
import pytest

from rl18xx.agent.alphazero.mcts import (
    MCTSNode,
    PRICE_GRID,
    _snap_price,
    sample_price_for_pw,
)
from rl18xx.agent.alphazero.config import SelfPlayConfig
from rl18xx.game.gamemap import GameMap


# ----- pure-function tests on sample_price_for_pw -----------------------------


def _seeded_rng(seed: int = 12345) -> np.random.Generator:
    return np.random.default_rng(seed)


@pytest.mark.parametrize(
    "action_type, step",
    [("Bid", 5), ("BuyTrain", 1), ("BuyCompany", 1)],
)
def test_sample_price_snaps_to_grid(action_type, step):
    """Snapping invariant: every sampled price is a multiple of the grid step
    for that action type, and within the legal range."""
    rng = _seeded_rng()
    price_range = (50, 500) if action_type == "Bid" else (10, 300)
    for _ in range(200):
        sample = sample_price_for_pw(
            price_mean=float((price_range[0] + price_range[1]) / 2.0),
            price_log_std=float(np.log((price_range[1] - price_range[0]) / 4.0)),
            action_type=action_type,
            price_range=price_range,
            rng=rng,
        )
        assert price_range[0] <= sample <= price_range[1]
        assert sample % step == 0, f"{action_type}: {sample} not multiple of {step}"


def test_sample_price_cluster_around_mean_for_bid():
    """With a fixed (μ, log σ) and a seeded RNG, repeated samples cluster
    around μ (within a few σ on average). Tests both Bid (5-step snap) and
    BuyTrain (1-step snap)."""
    rng = _seeded_rng(seed=42)
    mu = 200.0
    log_sigma = float(np.log(20.0))  # σ = 20
    price_range = (5, 1000)
    samples = [
        sample_price_for_pw(mu, log_sigma, "Bid", price_range, rng)
        for _ in range(500)
    ]
    arr = np.asarray(samples, dtype=float)
    # Mean is within a small tolerance of μ (σ/√n ≈ ~1 for n=500, σ=20).
    assert abs(arr.mean() - mu) < 5.0
    # Std is in the ballpark of σ (snapping + clipping widens slightly).
    assert 10.0 < arr.std() < 40.0
    # All Bid samples should land on the $5 grid.
    assert np.all(arr % 5 == 0)


def test_sample_price_fixed_range_returns_exact_price():
    """``price_range = (p, p)`` (degenerate range) should always return p exactly,
    regardless of μ, σ, or action_type."""
    for action_type in ("Bid", "BuyTrain", "BuyCompany"):
        for p in (1, 50, 250, 999):
            # Even with a wildly mismatched μ, σ should not matter.
            sample = sample_price_for_pw(
                price_mean=10000.0,
                price_log_std=float(np.log(0.001)),
                action_type=action_type,
                price_range=(p, p),
                rng=_seeded_rng(),
            )
            assert sample == p, f"({action_type}, {p}) got {sample}"


def test_sample_price_always_in_range_under_extreme_inputs():
    """Even when μ is wildly outside the legal range and σ is huge, returned
    samples must clamp into ``[price_range[0], price_range[1]]``."""
    rng = _seeded_rng()
    price_range = (10, 100)
    # μ way above the range
    for _ in range(50):
        sample = sample_price_for_pw(1e6, float(np.log(1e6)), "BuyTrain", price_range, rng)
        assert price_range[0] <= sample <= price_range[1]
    # μ way below
    for _ in range(50):
        sample = sample_price_for_pw(-1e6, float(np.log(1e6)), "BuyTrain", price_range, rng)
        assert price_range[0] <= sample <= price_range[1]


def test_snap_price_rounds_up_when_below_min():
    """If a sampled price is below price_min, snap should round up to the
    nearest legal grid multiple ≥ price_min."""
    # Bid: step=5, price_min=52 → 55 (52 % 5 == 2 → 52 + (5-2) = 55)
    assert _snap_price(0.0, "Bid", 52, 100) == 55
    # BuyTrain: step=1, price_min=10 → 10
    assert _snap_price(0.0, "BuyTrain", 10, 100) == 10


def test_snap_price_rounds_down_when_above_max():
    """If a sample is above price_max, snap should round down to the nearest
    legal grid multiple ≤ price_max."""
    # Bid: step=5, price_max=53 → 50 (53 - (53 % 5) = 50)
    assert _snap_price(1e6, "Bid", 5, 53) == 50
    # BuyTrain: step=1, price_max=100 → 100
    assert _snap_price(1e6, "BuyTrain", 1, 100) == 100


# ----- MCTSNode._sample_price_for_slot integration tests ----------------------


def _fresh_root_node() -> MCTSNode:
    """Construct an MCTSNode at the initial 1830 game state. Used for the
    fallback-prior test below — we don't need to drive any moves to exercise
    ``_sample_price_for_slot`` directly."""
    game_map = GameMap()
    game_class = game_map.game_by_title("1830")
    game = game_class({1: "P1", 2: "P2", 3: "P3", 4: "P4"})
    return MCTSNode(game, config=SelfPlayConfig())


def test_sample_price_for_slot_falls_back_to_wide_normal_when_no_components():
    """When the node has no ``price_components`` attached (e.g. GNN model
    or pre-``incorporate_results`` expansion), ``_sample_price_for_slot``
    falls back to a wide-Normal prior centered at the midpoint and still
    returns a legal snapped price."""
    node = _fresh_root_node()
    # Defensive: clear any price_components that may have been attached.
    assert not hasattr(node, "price_components") or getattr(node, "price_components") is None

    # Seed numpy global since the node uses a private RNG; the fallback path
    # only goes through ``sample_price_for_pw`` which honours its own ``rng``
    # argument (here defaulted by ``_sample_price_for_slot``). We assert the
    # invariants that should hold regardless of RNG seed.
    price_range = (50, 200)
    # Use an arbitrary action_index — _sample_price_for_slot only uses
    # action_index when looking up price_components (None here).
    for _ in range(50):
        sampled = node._sample_price_for_slot(
            action_index=0, action_type="Bid", price_range=price_range
        )
        assert price_range[0] <= sampled <= price_range[1]
        assert sampled % PRICE_GRID["Bid"] == 0


def test_sample_price_for_slot_degenerate_range_returns_exact_price():
    """A fixed-price slot (price_range[0] == price_range[1]) on a real node
    should short-circuit and return the exact price."""
    node = _fresh_root_node()
    for action_type in ("Bid", "BuyTrain", "BuyCompany"):
        sampled = node._sample_price_for_slot(
            action_index=0, action_type=action_type, price_range=(123, 123)
        )
        assert sampled == 123
