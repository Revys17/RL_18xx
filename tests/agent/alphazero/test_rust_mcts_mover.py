"""The Rust search tree's player to move (``RustMCTSPlayer.root_active_player_index``).

During an operating round the active entity is a corporation; its president
decides. The tree read only the active *player*, empty on a corporation's
turn, and fell back to seat 0 -- so until 2026-10-16 every operating-round
node searched for seat 0's value whoever was moving (search evals: seat 0 won
up to half the games). The tree must agree with ``RustGameAdapter.active_players``,
which also sets the encoder's rotation.
"""

import random

import pytest

from rl18xx.agent.alphazero.rust_mcts_player import _active_player_index
from rl18xx.rust_adapter import RustGameAdapter

engine_rs = pytest.importorskip("engine_rs")


def _movers_by_round(seed: int, max_actions: int = 4000):
    """(round name, tree's mover, adapter's mover) at each position of a random game."""
    rng = random.Random(seed)
    game = engine_rs.BaseGame({i + 1: f"Player {i + 1}" for i in range(4)})
    game.set_auction_unlock(True)
    adapter = RustGameAdapter(game)
    out = []
    for _ in range(max_actions):
        if adapter.finished:
            break
        tree = engine_rs.RustMCTSPlayer(adapter)
        out.append((adapter.round.__class__.__name__, tree.root_active_player_index(), _active_player_index(adapter)))
        tree.advance_root(rng.choice(list(tree.legal_action_indices_at_root())))
        adapter = RustGameAdapter(tree.root_game_object())
    return out


def test_operating_round_mover_is_the_presidents_seat():
    movers = _movers_by_round(seed=3)
    operating = [(tree, adapter) for name, tree, adapter in movers if name == "Operating"]
    assert any(adapter != 0 for _, adapter in operating), "no operating turn by a president outside seat 0"
    assert all(tree == adapter for tree, adapter in operating)


def test_every_round_mover_matches_the_adapter():
    assert all(tree == adapter for _, tree, adapter in _movers_by_round(seed=5))
