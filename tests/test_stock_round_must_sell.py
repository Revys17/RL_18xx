"""Ruby ``BuySellParShares#must_sell?`` in the Python engine.

Over the cert limit, or holding a corporation above its holding limit THAT THE
PLAYER CAN SELL (``must_sell_corporations`` -> ``can_sell_any_bundles``), the
stock turn offers only SellShares. The Rust engine pins the same cases on the
same rigged states (``rounds::stock::must_sell_tests``).
"""

import logging
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
logging.disable(logging.CRITICAL)

from rl18xx.game.engine.actions import SellShares  # noqa: E402
from rl18xx.game.engine.entities import ShareBundle  # noqa: E402
from rl18xx.game.gamemap import GameMap  # noqa: E402

PLAYERS = {i: f"Player {i}" for i in range(1, 5)}
PAR = (5, 6)


def _rigged(nynh_pool):
    """A 4-player 1830 stock round after the first (sales allowed): NYNH and PRR
    parred at $67 on the par cell. Player 1 is NYNH's president with certs 0-3
    (50%); ``nynh_pool`` of the other five certs are in the market pool, the
    rest in the IPO. Player 2 is PRR's president (cert 0) and player 1 holds PRR
    cert 1. NYNH's holding limit is lowered to 40% so player 1 is over it.
    Player 1 acts. Same rig as the Rust tests."""
    game = GameMap().game_by_title("1830")(PLAYERS)
    game.turn = 2
    game.round = game.new_stock_round()
    for sym in ("NYNH", "PRR"):
        corp = game.corporation_by_id(sym)
        game.stock_market.set_par(corp, game.stock_market.share_price(PAR))
        corp.ipoed = True
    nynh = [game.share_by_id(f"NYNH_{i}") for i in range(9)]
    prr = [game.share_by_id(f"PRR_{i}") for i in range(9)]
    p1, p2 = game.player_by_id(1), game.player_by_id(2)
    game.share_pool.transfer_shares(ShareBundle(nynh[0:4]), p1)
    if nynh_pool:
        game.share_pool.transfer_shares(ShareBundle(nynh[4 : 4 + nynh_pool]), game.share_pool)
    game.share_pool.transfer_shares(ShareBundle(prr[0:1]), p2)
    game.share_pool.transfer_shares(ShareBundle(prr[1:2]), p1)
    game.corporation_by_id("NYNH").max_ownership_percent = 40
    assert game.current_entity.id == 1
    assert game.corporation_by_id("NYNH").owner == p1 and game.corporation_by_id("PRR").owner == p2
    return game


def _step(game):
    return game.round.active_step()


def _sellable_corps(game):
    step = _step(game)
    return {corp.id for corp in game.corporations if step.can_sell_any_bundles(game.player_by_id(1), corp)}


def test_an_unsellable_over_limit_corp_forces_no_sale():
    """NYNH's pool is full (50%) and nobody else could take the presidency:
    player 1 can't sell NYNH, so being over its limit forces nothing."""
    game = _rigged(5)
    assert _sellable_corps(game) == {"PRR"}
    p1 = game.player_by_id(1)
    assert not _step(game).must_sell(p1)
    actions = _step(game).actions(p1)
    assert SellShares in actions and len(actions) > 1


def test_a_sellable_over_limit_corp_forces_a_sale():
    game = _rigged(3)
    assert "NYNH" in _sellable_corps(game)
    p1 = game.player_by_id(1)
    assert _step(game).must_sell(p1)
    assert _step(game).actions(p1) == [SellShares]


def test_over_the_cert_limit_forces_a_sale():
    game = _rigged(5)
    game.cert_limit = lambda *args, **kwargs: 4
    p1 = game.player_by_id(1)
    assert _step(game).must_sell(p1)
    assert _step(game).actions(p1) == [SellShares]


@pytest.mark.parametrize("nynh_pool, must", [(5, False), (3, True)])
def test_must_sell_corporations_are_the_sellable_over_limit_ones(nynh_pool, must):
    """The Ruby rule as a set: the over-limit corporations the player can sell."""
    game = _rigged(nynh_pool)
    corps = [c.id for c in _step(game).must_sell_corporations(game.player_by_id(1))]
    assert corps == (["NYNH"] if must else [])
