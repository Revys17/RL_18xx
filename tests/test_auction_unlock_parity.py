"""Self-play auction-unlock variant (``auction_unlock``): both engines agree,
and it only acts when no player can afford the next private.

Real 1830 discounts only the SV on an all-pass, so a waterfall auction where
every player's cash is committed to bids and the next private is unaffordable
loops forever. With the variant on, that private's price drops $5 per all-pass
round until someone can afford it (or it is taken free at $0)."""
import engine_rs
import pytest

from rl18xx.game.engine.actions import BaseAction
from rl18xx.game.gamemap import GameMap

PLAYERS = {i: f"Player {i}" for i in range(1, 5)}


def _games(unlock: bool):
    py = GameMap().game_by_title("1830")(PLAYERS)
    py.auction_unlock = unlock
    rs = engine_rs.BaseGame(PLAYERS)
    rs.set_auction_unlock(unlock)
    return py, rs


def _apply(py, rs, action: dict):
    py.process_action(BaseAction.action_from_dict(dict(action), py))
    rs.process_action(dict(action))


def _state(py, rs):
    """(remaining privates, their min bids, player cash, private owners) per engine."""
    step = py.active_step()
    in_auction = bool(rs.auction_companies())
    py_state = (
        [c.sym for c in step.companies] if in_auction else [],
        {c.sym: step.min_bid(c) for c in step.companies} if in_auction else {},
        {p.id: p.cash for p in py.players},
        {c.sym: getattr(c.owner, "id", None) for c in py.companies},
    )
    rs_state = (
        rs.auction_companies(),
        {sym: rs.auction_min_bid(sym) for sym in rs.auction_companies()},
        {p.id: p.cash for p in rs.players},
    )
    return py_state, rs_state


def bid(player, company, price):
    return {"type": "bid", "entity": player, "entity_type": "player", "company": company, "price": price}


def pass_(player):
    return {"type": "pass", "entity": player, "entity_type": "player"}


# P1 buys the SV; every player's cash then goes into bids on other privates,
# leaving the $40 CS next in line and unaffordable.
LOCK = [bid(1, "SV", 20), bid(2, "BO", 600), bid(3, "CA", 600), bid(4, "DH", 600), bid(1, "MH", 580)]
ALL_PASS = [pass_(2), pass_(3), pass_(4), pass_(1)]


@pytest.mark.parametrize("unlock", [False, True])
def test_engines_agree_through_a_locked_auction(unlock):
    py, rs = _games(unlock)
    for action in LOCK + ALL_PASS * 6:
        _apply(py, rs, action)
        py_state, rs_state = _state(py, rs)
        assert py_state[:3] == rs_state, action


def test_unlock_discounts_the_next_private_only_while_nobody_can_afford_it():
    py, rs = _games(True)
    for action in LOCK:
        _apply(py, rs, action)
    prices = []
    for _ in range(6):
        for action in ALL_PASS:
            _apply(py, rs, action)
        prices.append(rs.auction_min_bid("CS"))
    # P1's SV revenue adds $5 of free cash per round: $35, $30, $25, $20, then
    # P1 can afford it and the real rules apply again.
    assert prices == [35, 30, 25, 20, 20, 20]


def test_without_unlock_the_locked_auction_never_moves():
    py, rs = _games(False)
    for action in LOCK + ALL_PASS * 6:
        _apply(py, rs, action)
    assert rs.auction_min_bid("CS") == 40 and py.active_step().min_bid(py.company_by_id("CS")) == 40


def test_unlocked_auction_resolves_every_bid():
    """Even when the SV's owner spends its revenue on raises, the CS gets
    cheap enough for it to buy (the $5 revenue it holds after each payout);
    buying it cascades through the bids waiting behind it and the auction
    ends, identically in both engines."""
    py, rs = _games(True)
    for action in LOCK + ALL_PASS:  # P1 now holds the SV's $5; CS at $35
        _apply(py, rs, action)
    mh_bid = 580
    while rs.auction_min_bid("CS") > 5:
        mh_bid += 5
        for action in [pass_(2), pass_(3), pass_(4), bid(1, "MH", mh_bid)] + ALL_PASS:
            _apply(py, rs, action)
            py_state, rs_state = _state(py, rs)
            assert py_state[:3] == rs_state, action
    for action in [pass_(2), pass_(3), pass_(4), bid(1, "CS", 5)]:
        _apply(py, rs, action)
        py_state, rs_state = _state(py, rs)
        assert py_state[:3] == rs_state, action
    assert not rs.auction_companies(), "buying the CS should resolve every remaining bid"
    owners = _state(py, rs)[0][3]
    assert owners == {"SV": 1, "CS": 1, "DH": 4, "MH": 1, "CA": 3, "BO": 2}
