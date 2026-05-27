"""Lightweight non-determinism test for gating-arena MCTS root noise.

``loop._play_gate_game`` injects Dirichlet noise on each gating MCTS root
(see ``GATING_DIRICHLET_NOISE_WEIGHT``) so repeated games against the same
opponent don't collapse to a single trajectory. The actual gating loop runs
full games, which is far too expensive for a unit test; instead we directly
exercise ``MCTSNode.inject_noise`` and assert that two consecutive calls on
the same node produce *different* ``child_prior_compressed`` arrays because
``np.random.dirichlet`` is non-deterministic between calls.

This is the smallest assertion that would catch the regression "someone
hard-coded a seed inside inject_noise" — which would silently make every
gating game identical and tank gating signal-to-noise.
"""
from __future__ import annotations

import numpy as np
import pytest

from rl18xx.agent.alphazero.config import SelfPlayConfig
from rl18xx.agent.alphazero.mcts import MCTSNode
from rl18xx.game.gamemap import GameMap


def _root_with_uniform_prior(noise_weight: float = 0.25) -> MCTSNode:
    """Build a root MCTSNode at the initial 1830 state, seed it with a
    uniform prior, and configure Dirichlet noise weight."""
    game_map = GameMap()
    game_class = game_map.game_by_title("1830")
    game = game_class({1: "P1", 2: "P2", 3: "P3", 4: "P4"})
    config = SelfPlayConfig(dirichlet_noise_weight=noise_weight)
    node = MCTSNode(game, config=config)
    # Plant a non-trivial uniform original_prior; otherwise the noise injection
    # only sees zeros.
    n = node.num_legal_actions
    assert n > 1, "test needs a state with multiple legal actions"
    node.original_prior_compressed = np.full(n, 1.0 / n, dtype=np.float32)
    return node


def test_inject_noise_changes_prior_between_calls():
    """Two consecutive ``inject_noise`` calls should give different priors
    because ``np.random.dirichlet`` is reseeded by neither call."""
    node = _root_with_uniform_prior(noise_weight=0.25)

    node.inject_noise()
    first = node.child_prior_compressed.copy()

    node.inject_noise()
    second = node.child_prior_compressed.copy()

    assert first.shape == second.shape
    # Vanishing probability of matching arrays element-wise for a Dirichlet
    # over many legal actions; we want *any* difference.
    assert not np.array_equal(first, second), "inject_noise produced identical samples — non-determinism broken"


def test_inject_noise_preserves_simplex():
    """Each noise injection should leave ``child_prior_compressed`` on the
    probability simplex (non-negative, sums to ~1)."""
    node = _root_with_uniform_prior(noise_weight=0.25)
    for _ in range(5):
        node.inject_noise()
        p = node.child_prior_compressed
        assert np.all(p >= 0)
        # ``original_prior * (1-w) + dirichlet * w`` — both pieces sum to 1.
        assert abs(p.sum() - 1.0) < 1e-5


def test_inject_noise_weight_zero_returns_original_prior():
    """Noise weight 0 disables the perturbation: child_prior must equal the
    original prior (up to float copy)."""
    node = _root_with_uniform_prior(noise_weight=0.0)
    original = node.original_prior_compressed.copy()

    node.inject_noise()
    np.testing.assert_array_equal(node.child_prior_compressed, original)

    node.inject_noise()  # idempotent
    np.testing.assert_array_equal(node.child_prior_compressed, original)


def test_inject_noise_seeded_global_rng_is_deterministic():
    """Documenting the wiring: with the global numpy RNG seeded *between*
    calls (e.g. for snapshot/golden tests), the resulting prior is
    reproducible. If this ever breaks, a downstream test that relies on
    seeded reproducibility will too."""
    node1 = _root_with_uniform_prior(noise_weight=0.25)
    node2 = _root_with_uniform_prior(noise_weight=0.25)

    np.random.seed(7)
    node1.inject_noise()
    np.random.seed(7)
    node2.inject_noise()

    np.testing.assert_array_equal(node1.child_prior_compressed, node2.child_prior_compressed)
