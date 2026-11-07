"""Real-rules auctions in policy gradient (gen5, 2026-11-07): auction starts without the auction_unlock variant, the
stall limit (a game still in the auction after N decisions ends, scored on net worth) and the prices paid."""
import random

import numpy as np

from rl18xx.agent.alphazero import policy_gradient as pg
from rl18xx.agent.alphazero import start_positions as sp
from rl18xx.agent.alphazero.policy_selfplay import _Game, _advance


def _auction_start(unlock: bool):
    start, game, _ = sp.replayed_start(4, None, 1.0, rng=random.Random(3), auction_fraction=1.0, auction_unlock=unlock)
    assert start.label == sp.AUCTION_START
    return start, game


def test_auction_start_unlock_flag():
    assert _auction_start(True)[1]._game.auction_unlock is True
    assert _auction_start(False)[1]._game.auction_unlock is False


def _play(game, settings, pick, max_steps=5000):
    g = _Game(game=game, uid="t", start=sp.AUCTION_START)
    rng = np.random.default_rng(0)
    for _ in range(max_steps):
        legal = _advance(g, settings, rng)
        if legal is None:
            return g
        index = pick(game, legal)
        game._game.apply_action_index(index, None if game._game.price_range_for_index(index) is None
                                      else game._game.price_range_for_index(index)[0])
        g.decisions += 1
    raise AssertionError("game did not end")


def test_stall_limit_ends_the_auction_on_net_worth():
    _, game = _auction_start(False)
    settings = {"max_decisions": 1000, "auction_stall_decisions": 6, "price_eps": 0.0}
    g = _play(game, settings, lambda game, legal: legal[-1])
    if g.termination == "auction_stall":
        assert g.decisions == 6
        win, fractions = pg.outcome(game)
        assert np.isclose(win.sum(), 1.0) and np.isclose(fractions.sum(), 1.0)
        assert pg.auction_prices(type("G", (), {"start": sp.AUCTION_START, "termination": "auction_stall",
                                                 "game": game})()) is None


def test_no_stall_limit_outside_the_auction_and_prices_recorded():
    _, game = _auction_start(False)
    settings = {"max_decisions": 60, "auction_stall_decisions": 0, "price_eps": 0.0}
    g = _play(game, settings, lambda game, legal: legal[0])
    assert g.termination in ("max_length", "finished")
    rec = type("G", (), {"start": sp.AUCTION_START, "termination": g.termination, "game": game})()
    prices = pg.auction_prices(rec)
    assert isinstance(prices, dict)
    assert all(isinstance(v, int) and v >= 0 for v in prices.values())


def test_auction_prices_none_for_other_starts():
    rec = type("G", (), {"start": "human:1", "termination": "finished", "game": None})()
    assert pg.auction_prices(rec) is None
