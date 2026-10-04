"""Game-length limits count decisions; the search trees' terminal check counts
engine actions only as a safety bound.

``SelfPlayConfig.max_game_length`` (and the resign floor, exploration cutoff
and auction-stall limits) count decisions -- moves where the player to act
had more than one legal action -- so actions the engine applies on its own
never use up a game. ``MCTSNode.is_done`` (the Python tree) and the Rust
tree's forced-action collapse stop at ``max_engine_actions`` instead, far
above any real game. Both players (Python ``MCTSPlayer`` and
``RustMCTSPlayer``) report done at the same decision count.
"""

import numpy as np
import pytest
import torch

from rl18xx.agent.alphazero.config import SelfPlayConfig
from rl18xx.agent.alphazero.mcts import POLICY_SIZE, VALUE_SIZE
from rl18xx.agent.alphazero.rust_mcts_player import RustMCTSPlayer
from rl18xx.agent.alphazero.self_play import MCTSPlayer
from rl18xx.game.gamemap import GameMap


class DummyNet:
    """Minimal network stub matching the MCTS inference interface."""

    def encoder_type(self):
        return "GNN"

    def run_many_encoded(self, encoded_game_states):
        n = len(encoded_game_states)
        priors = torch.ones(POLICY_SIZE, dtype=torch.float32) / POLICY_SIZE
        value = torch.zeros(VALUE_SIZE, dtype=torch.float32)
        return [priors] * n, [torch.log(priors)] * n, [value] * n


def _build_python_game():
    game_class = GameMap().game_by_title("1830")
    return game_class({1: "Player 1", 2: "Player 2", 3: "Player 3", 4: "Player 4"})


@pytest.mark.parametrize("max_engine_actions,expected_done", [(0, True), (1, False)])
def test_python_tree_is_done_at_the_engine_action_bound(max_engine_actions, expected_done):
    config = SelfPlayConfig(network=DummyNet(), max_engine_actions=max_engine_actions, use_score_values=False)
    player = MCTSPlayer(config)
    player.initialize_game(_build_python_game())
    assert player.root.game_object.move_number == 0
    assert player.root.is_done() == expected_done


@pytest.mark.parametrize("player_class", [MCTSPlayer, RustMCTSPlayer])
def test_players_are_done_after_max_game_length_decisions(player_class):
    """Opening-auction moves are decisions (several bids or a pass), so after
    max_game_length of them the player is done -- whatever the engine-action
    count."""
    config = SelfPlayConfig(
        network=DummyNet(), max_game_length=3, num_readouts=2, parallel_readouts=1, min_readouts=1,
        dirichlet_noise_weight=0.0, use_score_values=False, player_count_distribution={4: 1.0},
    )
    player = player_class(config)  # both start a fresh Rust-engine game
    for played in range(3):
        assert not player.is_done()
        player.play_move(player.suggest_move())
        assert player.decisions == played + 1
    assert player.is_done()
    assert np.array_equal(player.result, np.zeros_like(player.result))
