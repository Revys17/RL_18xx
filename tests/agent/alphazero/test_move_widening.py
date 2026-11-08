"""The top-k move widening at wide nodes is a switch (SelfPlayConfig.mcts_move_widening, both
engines): on (the default), a node with more than 20 legal moves selects among only the
k = max(1, pw_c * N^pw_alpha) highest-prior ones; off, every legal move competes, as in AlphaZero."""

import math

import numpy as np
import pytest

engine_rs = pytest.importorskip("engine_rs")

from rl18xx.agent.alphazero.lockstep import LockstepSearcher, ModelEvaluator, TreeSearch  # noqa: E402

from .search_test_utils import HashNet, make_player, random_positions  # noqa: E402

SEARCH = dict(
    parallel_readouts=8, num_readouts=200, min_readouts=200, mcts_mean_q=True, fpu_reduction=0.0,
    c_puct_init=0.3125, c_puct_by_round={}, dedupe_leaves=True, prior_temperature=1.0,
)  # fmt: skip


def _wide_position(net):
    for seed in range(8):
        for game in random_positions(2, seed, steps=1500, every=9):
            if len(make_player(net, game, **SEARCH)._rust_player.legal_action_indices_at_root()) > 30:
                return game
    raise AssertionError("no position with more than 30 legal moves")


def _visited(net, game, **overrides):
    player = make_player(net, game, seed=1, **SEARCH, **overrides)
    LockstepSearcher({None: ModelEvaluator(net)}).run([TreeSearch(player, 200)])
    stats = player._rust_player.root_target_stats()
    assert stats["move_widening"] is overrides.get("mcts_move_widening", True)
    return int((np.asarray(stats["child_n"]) > 0).sum()), float(stats["root_n"])


def test_move_widening_is_a_switch():
    net = HashNet()
    game = _wide_position(net)
    assert make_player(net, game, **SEARCH)._rust_player.move_widening_enabled  # on by default
    on, root_n = _visited(net, game)
    assert on <= math.floor(root_n**0.5)  # pw_c 1, pw_alpha 0.5: the top floor(sqrt(N)) moves at most
    off, _ = _visited(net, game, mcts_move_widening=False)
    assert off > math.floor(root_n**0.5)
