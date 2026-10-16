"""Batched readouts in the Rust MCTS wrapper (``RustMCTSPlayer.tree_search``).

A virtual loss lowers an in-flight child's W but not its N, so within one
batch the search keeps re-selecting unexpanded children; at a fresh node a
batch of 32 picks the same few leaves over and over. Backing up every copy
counted each as a visit carrying one network value (2026-10-16: @64 with
parallel_readouts 32 evaluated ~17 distinct positions). ``dedupe_leaves``
evaluates and backs up each distinct leaf once, as minigo does.
"""

import numpy as np
import pytest

from rl18xx.agent.alphazero.config import SelfPlayConfig
from rl18xx.agent.alphazero.mcts import POLICY_SIZE, VALUE_SIZE
from rl18xx.agent.alphazero.rust_mcts_player import RustMCTSPlayer, _temper_priors
from rl18xx.rust_adapter import RustGameAdapter

engine_rs = pytest.importorskip("engine_rs")


class _FlatNet:
    """Uniform priors and equal values: selection is driven by the batch alone."""

    def run_many_encoded(self, encoded_states, legal_indices=None):
        n = len(encoded_states)
        values = np.zeros((n, VALUE_SIZE), dtype=np.float32)
        values[:, :4] = 0.25
        return np.full((n, POLICY_SIZE), 1.0 / POLICY_SIZE, dtype=np.float32), None, values


def _searched(dedupe: bool, batches: int = 4):
    config = SelfPlayConfig(
        network=_FlatNet(),
        parallel_readouts=32,
        num_readouts=128,
        dedupe_leaves=dedupe,
        dirichlet_noise_weight=0.0,
        enable_resign=False,
        noresign_holdout_rate=0.0,
        player_count_distribution={4: 1.0},
    )
    player = RustMCTSPlayer(config)
    player.initialize_game(RustGameAdapter(engine_rs.BaseGame({i + 1: f"Player {i + 1}" for i in range(4)})))
    for _ in range(batches):
        player.tree_search()
    tree = player._rust_player
    expanded = sum(bool(tree.is_expanded(i)) for i in range(tree.arena_size()))
    return tree.n_at_root(), expanded


def test_duplicate_leaves_are_backed_up_once():
    visits, expanded = _searched(dedupe=True)
    # The root's first expansion plus one backup per distinct leaf.
    assert visits == expanded


def test_without_dedupe_a_batch_counts_copies_as_visits():
    visits, expanded = _searched(dedupe=False)
    assert visits > 2 * expanded


def test_prior_temperature_is_softmax_of_scaled_logits():
    logits = np.array([2.0, 1.0, 0.0, -1.0])
    probs = np.exp(logits) / np.exp(logits).sum()
    config = SelfPlayConfig(network=None, prior_temperature=0.5)
    tempered = _temper_priors(probs.astype(np.float32), config)
    tempered = tempered / tempered.sum()  # the Rust tree renormalizes over the legal moves
    expected = np.exp(logits / 0.5) / np.exp(logits / 0.5).sum()
    np.testing.assert_allclose(tempered, expected, rtol=1e-5)
