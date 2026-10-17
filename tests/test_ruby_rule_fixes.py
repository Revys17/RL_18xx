"""Rules where the Ruby engine (tobymao/18xx) is the source of truth and both
our engines were brought in line, found by replaying our own games in Ruby
(scripts/ruby_replay_check.py). The Rust engine pins the same cases in its own
tests (game.rs ``ruby_rule_fixes``).
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
logging.disable(logging.CRITICAL)

from rl18xx.game.gamemap import GameMap  # noqa: E402

PLAYERS = {i: f"Player {i}" for i in range(1, 5)}


def _game():
    return GameMap().game_by_title("1830")(PLAYERS)


def test_erie_reservation_keeps_a_slot_on_the_preprinted_e11_and_blocks_on_a_laid_tile():
    """Ruby Tile#token_blocked_by_reservation?: ERIE's E11 reservation is
    tile-level (two cities, none given). The preprinted tile keeps the default
    ``:never`` -- others may token while a slot stays open for ERIE -- but a
    laid tile takes G1830 TILE_RESERVATION_BLOCKS_OTHERS = ``:always``: nobody
    else may token it until ERIE's home token is down."""
    game = _game()
    nyc, erie = game.corporation_by_id("NYC"), game.corporation_by_id("ERIE")
    hex_ = game.hex_by_id("E11")
    assert hex_.tile.reservation_blocks == "never"
    assert hex_.tile.reservations == [erie]
    assert not hex_.tile.token_blocked_by_reservation(nyc)

    hex_.lay(game.get_available_tile_with_name("59"))
    assert hex_.tile.name == "59" and hex_.tile.reservation_blocks == "always"
    assert hex_.tile.reservations == [erie]
    assert hex_.tile.token_blocked_by_reservation(nyc)
    assert not hex_.tile.token_blocked_by_reservation(erie)


def test_the_bank_breaks_only_when_a_payment_takes_it_below_zero():
    """Ruby Bank#check_cash: paying out its last dollar exactly leaves the
    bank unbroken; the next payment breaks it."""
    bank = _game().bank
    bank.check_cash(bank.cash)
    assert not bank.is_broken()
    bank.check_cash(bank.cash + 1)
    assert bank.is_broken()
