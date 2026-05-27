"""Tests for the canonical (price-collapsed) ActionMapper round-trip.

The contract:

- ``get_legal_action_indices(state)`` returns the canonical, price-collapsed
  set of legal indices.
- ``map_index_to_action_with_price(idx, state, price)`` materializes an
  ``Action`` from a canonical index, plugging in an explicit price for
  price-bearing slots (and ignoring ``price`` for categorical ones).
- ``canonical_index_for_action(action, state)`` maps that ``Action`` back to
  the same canonical index — including for price-bearing actions, where the
  price dimension must be folded out so the index is independent of the
  sampled price.

This module tests the round trip on a fresh 4-player 1830 game (private
auction, all legal actions are ``Bid``s — i.e. the price-bearing branch is
exercised on every example).
"""

import pytest

from rl18xx.agent.alphazero.action_mapper import ActionMapper
from rl18xx.game.gamemap import GameMap


@pytest.fixture
def fresh_4p_game():
    game_map = GameMap()
    game_class = game_map.game_by_title("1830")
    players = {1: "Player 1", 2: "Player 2", 3: "Player 3", 4: "Player 4"}
    return game_class(players)


def test_canonical_index_for_action_round_trip(fresh_4p_game):
    """For every legal action at game start, decoding the canonical index and
    re-encoding the resulting Action returns the original canonical index."""
    mapper = ActionMapper()
    indices, price_ranges, _action_types = mapper.get_legal_actions_factored(fresh_4p_game)
    assert len(indices) > 0, "Fresh game must have at least one legal action"

    for idx in indices:
        # Pick a price in the legal range when present; otherwise pass 0
        # (ignored by ``map_index_to_action_with_price`` for categorical slots).
        if idx in price_ranges:
            lo, _hi = price_ranges[idx]
            sample_price = lo
        else:
            sample_price = 0

        action = mapper.map_index_to_action_with_price(idx, fresh_4p_game, sample_price)
        round_trip_idx = mapper.canonical_index_for_action(action, fresh_4p_game)
        assert round_trip_idx == idx, (
            f"Round-trip failed for index {idx}: action={action!r} "
            f"-> canonical_index_for_action returned {round_trip_idx}"
        )


def test_canonical_index_is_price_invariant(fresh_4p_game):
    """For every price-bearing legal slot in a fresh game, varying the price
    we plug into ``map_index_to_action_with_price`` must NOT change the
    canonical index we get back from ``canonical_index_for_action``."""
    mapper = ActionMapper()
    indices, price_ranges, _ = mapper.get_legal_actions_factored(fresh_4p_game)

    # Only price-bearing slots matter; at game start these are all ``Bid``
    # slots (one per private company) so we exercise the price-collapse path.
    priced_indices = [i for i in indices if i in price_ranges]
    assert priced_indices, "Expected at least one price-bearing slot in the fresh game"

    for idx in priced_indices:
        lo, hi = price_ranges[idx]
        # Three probe prices spanning the legal range.
        probe_prices = sorted(set([lo, (lo + hi) // 2, hi]))
        canonical_indices = []
        for price in probe_prices:
            action = mapper.map_index_to_action_with_price(idx, fresh_4p_game, price)
            canonical_indices.append(mapper.canonical_index_for_action(action, fresh_4p_game))
        assert len(set(canonical_indices)) == 1, (
            f"canonical_index_for_action must collapse the price dim for "
            f"slot {idx}; got {canonical_indices} for prices {probe_prices}"
        )
        # And the collapsed value matches the original slot index.
        assert canonical_indices[0] == idx
