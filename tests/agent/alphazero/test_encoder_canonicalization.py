"""Encoder canonicalization invariant tests.

The encoder's contract (``encoder.py``):

- ``encode(game)`` returns ``(... , active_player_idx, rotation, num_players)``.
- After canonicalization, ``active_player_idx`` is always 0; ``rotation`` is
  the absolute (un-canonicalized) index of the active player.
- Player-indexed sections of the flat game-state vector are rotated by
  ``-rotation`` so the active player's slot 0 holds their data.

This module exercises the black-box invariant from a fresh 4-player 1830 game:
take one action as Player 1, then verify Player 2's data ends up at the
"player 0" offset in the canonical state.
"""

import pytest

from rl18xx.agent.alphazero.encoder import Encoder_GNN
from rl18xx.game.action_helper import ActionHelper
from rl18xx.game.gamemap import GameMap


@pytest.fixture
def fresh_4p_game():
    """Build a fresh 4-player 1830 game at the start of the private auction."""
    game_map = GameMap()
    game_class = game_map.game_by_title("1830")
    players = {1: "Player 1", 2: "Player 2", 3: "Player 3", 4: "Player 4"}
    return game_class(players)


def test_initial_canonicalization_active_player_idx_zero(fresh_4p_game):
    """At the start of a fresh game Player 1 is active → rotation == 0,
    active_player_idx == 0, and the state passes through unchanged."""
    encoder = Encoder_GNN()
    result = encoder.encode(fresh_4p_game)
    _, _, _, _, _round_type, active_player_idx, rotation, num_players = result

    assert active_player_idx == 0
    assert rotation == 0
    assert num_players == 4


def test_canonicalization_rotates_after_first_action(fresh_4p_game):
    """After Player 1's first action, the active player shifts to Player 2.
    ``rotation`` therefore equals 1 (Player 2's absolute index) and the
    player-cash section in the canonical state has been rolled left so the
    active player's actual cash sits at slot 0."""
    encoder = Encoder_GNN()
    action_helper = ActionHelper()

    pre = encoder.encode(fresh_4p_game)
    assert pre[6] == 0  # rotation is 0 at start (P1 active)
    assert pre[5] == 0  # active_player_idx always 0 after canonicalization

    # Player 1's first action: bid $20 on SV. This advances the auction to
    # Player 2 (active player index 1 in absolute order).
    choices = action_helper.get_all_choices(fresh_4p_game)
    fresh_4p_game.process_action(choices[0])
    assert fresh_4p_game.active_players()[0].id == 2

    layout, _ = Encoder_GNN.compute_section_layout(4)
    cash_offset, cash_size = layout["player_cash"]
    assert cash_size == 4

    post = encoder.encode(fresh_4p_game)
    post_state = post[0].squeeze(0).numpy()
    assert post[5] == 0, "Canonicalized active_player_idx is always 0"
    assert post[6] == 1, (
        f"Expected rotation == 1 after first action shifts active player to "
        f"P2, got {post[6]}"
    )
    assert post[7] == 4

    # Read actual cash from the game and convert to the encoder's normalized
    # form so we can compare directly against the canonical slot values.
    starting_cash = encoder.starting_cash
    players_by_id = {p.id: p for p in fresh_4p_game.players}
    expected = [
        players_by_id[2].cash / starting_cash,  # P2 now at slot 0 (active)
        players_by_id[3].cash / starting_cash,  # P3 at slot 1
        players_by_id[4].cash / starting_cash,  # P4 at slot 2
        players_by_id[1].cash / starting_cash,  # P1 wrapped to slot 3
    ]
    cash_after = post_state[cash_offset:cash_offset + cash_size]
    for i, want in enumerate(expected):
        assert abs(float(cash_after[i]) - want) < 1e-5, (
            f"cash slot {i}: expected {want}, got {cash_after[i]}"
        )


def test_canonicalize_perspective_is_inverse_of_itself(fresh_4p_game):
    """``canonicalize_perspective`` shifts player-indexed sections left by
    ``rotation``; applying the inverse shift (``-rotation mod N``) must
    recover the original absolute layout. This is the property the
    ``encode_absolute`` helper in encoder_gnn_test.py depends on."""
    encoder = Encoder_GNN()
    encoder.initialize(fresh_4p_game)

    # Pre-canonical state with known values: pretend Player 2 is active.
    layout, _ = Encoder_GNN.compute_section_layout(4)
    cash_offset, _ = layout["player_cash"]
    state_pre = encoder.encode(fresh_4p_game)[0].squeeze(0).numpy().copy()
    # Stamp identifiable per-player values into the cash slots.
    state_pre[cash_offset:cash_offset + 4] = [10.0, 20.0, 30.0, 40.0]

    canonical = encoder.canonicalize_perspective(state_pre, 1)
    # After rotation by -1, slot 0 holds Player 2's cash, slot 3 holds P1.
    assert canonical[cash_offset:cash_offset + 4].tolist() == [20.0, 30.0, 40.0, 10.0]

    # Round-trip: rotating by N-1 (== -1 mod 4) undoes the original shift.
    undone = encoder.canonicalize_perspective(canonical, (-1) % 4)
    assert undone[cash_offset:cash_offset + 4].tolist() == [10.0, 20.0, 30.0, 40.0]
