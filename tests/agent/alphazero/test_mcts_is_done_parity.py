"""Parity test: ``MCTSPlayer.is_done`` and ``MCTSNode.is_done`` agree on the
``>= max_game_length`` boundary.

Both methods consult ``game_object.move_number >= config.max_game_length``
(MCTSPlayer ORs in a check on whether ``self.result`` has been set, but at
the moment of interest ``result`` is still zero). This test pins that they
agree on both sides of the boundary so that a regression where one uses
``>`` and the other uses ``>=`` (or one reads a different config field)
would be caught.
"""

import numpy as np
import pytest
import torch

from rl18xx.agent.alphazero.config import SelfPlayConfig
from rl18xx.agent.alphazero.mcts import VALUE_SIZE
from rl18xx.agent.alphazero.self_play import MCTSPlayer
from rl18xx.game.gamemap import GameMap


class DummyNet:
    """Minimal network stub matching the MCTS inference interface."""

    def encoder_type(self):
        return "GNN"

    def run_many_encoded(self, encoded_game_states):
        n = len(encoded_game_states)
        priors = torch.ones(26537, dtype=torch.float32) / 26537
        value = torch.zeros(VALUE_SIZE, dtype=torch.float32)
        return [priors] * n, [torch.log(priors)] * n, [value] * n


def _build_python_game():
    """Fresh 4-player 1830 game on the Python engine.

    Python engine usage matches the existing mcts_test.py style so the
    DummyNet is sufficient; we never advance through MCTS readouts here."""
    game_map = GameMap()
    game_class = game_map.game_by_title("1830")
    players = {1: "Player 1", 2: "Player 2", 3: "Player 3", 4: "Player 4"}
    return game_class(players)


@pytest.mark.parametrize(
    "max_game_length,expected_done",
    [
        # move_number == 0 at game start, so >= 0 is True, > 0 is False.
        (0, True),   # 0 >= 0 → root.is_done() True
        (1, False),  # 0 >= 1 → root.is_done() False (and game not finished)
    ],
)
def test_player_and_root_is_done_agree_on_max_game_length_boundary(
    max_game_length, expected_done
):
    """At the ``move_number >= max_game_length`` boundary, both methods
    should agree (since ``MCTSPlayer.result`` is still all-zero at this
    point, the player's extra ``result != 0`` term is False and both
    reduce to ``root.is_done()``)."""
    config = SelfPlayConfig(
        network=DummyNet(),
        max_game_length=max_game_length,
        use_score_values=False,
    )
    player = MCTSPlayer(config)
    # Override the default Rust-backed game with a Python BaseGame so we
    # don't need engine_rs available; move_number == 0 on a fresh game.
    player.initialize_game(_build_python_game())

    # Sanity: the player's result is still zero at this point.
    assert np.array_equal(player.result, np.zeros_like(player.result))

    root_done = player.root.is_done()
    player_done = player.is_done()

    assert root_done == expected_done, (
        f"root.is_done() returned {root_done}, expected {expected_done} "
        f"for max_game_length={max_game_length}, move_number={player.root.game_object.move_number}"
    )
    assert player_done == root_done, (
        f"MCTSPlayer.is_done() and MCTSNode.is_done() disagree at "
        f"move_number={player.root.game_object.move_number}, "
        f"max_game_length={max_game_length}: "
        f"player_done={player_done}, root_done={root_done}"
    )
