"""Exhaustive slot-coverage tests for ``ActionMapper.price_head_slot_for_action``.

For every (action_type, entity) combination that the ``ContinuousPriceHead``
models, we assert that ``price_head_slot_for_action`` returns the right slot
index, observed price, and price-range. Mocks stand in for the heavy game
objects — the function only touches a handful of fields on
``action`` / ``state``, and pulling in a full ``BaseGame`` per case would
make the test multi-second.

Slot layout (from ``ActionMapper`` constants):
  - Bid slots:          0 .. 5             (6 companies)
  - BuyTrain slots:     6 .. 53            (8 corps × 6 train types = 48)
  - BuyCompany slots:   54 .. 59           (6 companies)

The function returns ``None`` (skipped here) for:
  - depot-owned BuyTrain (fixed price)
  - market-discarded BuyTrain
  - unknown company / corporation symbols
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from rl18xx.agent.alphazero.action_mapper import ActionMapper


# ----- helpers ----------------------------------------------------------------


def _mock_bid(company_id: str, price: int, *, min_bid: int = 5, max_bid: int = 1000):
    """Build a (action, state) pair shaped just enough for the Bid branch."""
    company = MagicMock()
    company.id = company_id
    entity = MagicMock()
    entity.id = "player1"

    action = MagicMock()
    action.__class__.__name__ = "Bid"
    # ``price_head_slot_for_action`` inspects ``action.__class__.__name__``,
    # which a MagicMock doesn't honour. Use a tiny shim with the right
    # class name instead.
    class _Bid:
        pass

    _Bid.__name__ = "Bid"
    a = _Bid()
    a.company = company
    a.entity = entity
    a.price = price

    step = MagicMock()
    step.min_bid.return_value = min_bid
    step.max_bid.return_value = max_bid
    state = MagicMock()
    state.active_step.return_value = step
    return a, state


def _mock_buy_company(company_id: str, price: int, *, min_price: int = 5, max_price: int = 1000, buying_power: int = 1000):
    class _BuyCompany:
        pass

    _BuyCompany.__name__ = "BuyCompany"

    company = MagicMock()
    company.id = company_id
    company.min_price = min_price
    company.max_price = max_price
    entity = MagicMock()
    entity.id = "ENTITY"
    entity.cash = buying_power

    a = _BuyCompany()
    a.company = company
    a.entity = entity
    a.price = price

    state = MagicMock()
    # buying_power is read as ``getattr(state, 'buying_power', lambda x: x.cash)``,
    # so override with a callable on the mock.
    state.buying_power = lambda e: e.cash
    return a, state


def _mock_buy_train(
    corp_sym: str,
    train_name: str,
    price: int,
    *,
    spend_min: int = 1,
    spend_max: int = 1000,
):
    class _BuyTrain:
        pass

    _BuyTrain.__name__ = "BuyTrain"

    owner = MagicMock()
    owner.sym = corp_sym
    owner.name = corp_sym  # both ``sym`` and ``name`` populated; either works
    owner.id = corp_sym

    train = MagicMock()
    train.owner = owner
    train.name = train_name

    entity = MagicMock()
    entity.id = "BUYER"
    entity.cash = 5000

    a = _BuyTrain()
    a.train = train
    a.entity = entity
    a.price = price

    step = MagicMock()
    step.spend_minmax.return_value = (spend_min, spend_max)
    round_obj = MagicMock()
    round_obj.active_step.return_value = step
    state = MagicMock()
    state.round = round_obj
    return a, state


# ----- fixture ----------------------------------------------------------------


@pytest.fixture(scope="module")
def mapper():
    return ActionMapper()


# ----- Bid exhaustive coverage (6 companies → slots 0..5) ---------------------


@pytest.mark.parametrize(
    "company_id, expected_slot",
    [
        ("SV", 0),
        ("CS", 1),
        ("DH", 2),
        ("MH", 3),
        ("CA", 4),
        ("BO", 5),
    ],
)
def test_bid_slot_for_each_company(mapper, company_id, expected_slot):
    action, state = _mock_bid(company_id, price=120, min_bid=50, max_bid=600)
    result = mapper.price_head_slot_for_action(action, state)
    assert result is not None
    action_type, slot, observed, lo, hi = result
    assert action_type == "Bid"
    assert slot == expected_slot
    assert observed == 120
    assert lo == 50
    assert hi == 600


def test_bid_unknown_company_returns_none(mapper):
    action, state = _mock_bid("ZZZ", price=10, min_bid=5, max_bid=1000)
    assert mapper.price_head_slot_for_action(action, state) is None


def test_bid_with_step_failure_falls_back_to_observed_price(mapper):
    """Bid step.min_bid raising (e.g. wrong active step on Rust adapter) should
    fall back to (price, price) and still return a usable slot."""
    action, state = _mock_bid("SV", price=200)
    state.active_step.return_value.min_bid.side_effect = AttributeError("boom")
    result = mapper.price_head_slot_for_action(action, state)
    assert result is not None
    _, slot, observed, lo, hi = result
    assert slot == 0
    assert observed == 200
    assert lo == 200 and hi == 200


# ----- BuyTrain exhaustive coverage -------------------------------------------
# Cross-corp slot layout (per ActionMapper / price_head_slot_for_action):
#   slot = len(_PRICE_HEAD_COMPANIES)
#        + corp_idx * len(_PRICE_HEAD_TRAIN_TYPES)
#        + train_idx
# With 6 companies, 8 corps, 6 train types → 6 .. 53 inclusive (48 slots).

_CORPS = ("PRR", "NYC", "CPR", "B&O", "C&O", "ERIE", "NYNH", "B&M")
_TRAINS = ("2", "3", "4", "5", "6", "D")


@pytest.mark.parametrize("corp_idx, corp_sym", list(enumerate(_CORPS)))
@pytest.mark.parametrize("train_idx, train_name", list(enumerate(_TRAINS)))
def test_buy_train_slot_for_each_corp_and_train(mapper, corp_idx, corp_sym, train_idx, train_name):
    expected_slot = 6 + corp_idx * 6 + train_idx
    action, state = _mock_buy_train(corp_sym, train_name, price=150, spend_min=1, spend_max=900)
    result = mapper.price_head_slot_for_action(action, state)
    assert result is not None
    action_type, slot, observed, lo, hi = result
    assert action_type == "BuyTrain"
    assert slot == expected_slot, f"({corp_sym}, {train_name}) expected {expected_slot} got {slot}"
    assert 6 <= slot < 6 + 48
    assert observed == 150
    assert lo == 1 and hi == 900


def test_buy_train_depot_returns_none(mapper):
    """Depot-owned trains are fixed price — the head doesn't model them."""
    action, state = _mock_buy_train("PRR", "2", price=80)
    action.train.owner = "The Depot"  # owner can be a bare string per the docstring
    assert mapper.price_head_slot_for_action(action, state) is None


def test_buy_train_unknown_corp_returns_none(mapper):
    """A cross-corp BuyTrain from an unknown corp sym should return None."""
    action, state = _mock_buy_train("UNKNOWN", "2", price=100)
    assert mapper.price_head_slot_for_action(action, state) is None


def test_buy_train_unknown_train_type_returns_none(mapper):
    """A train of an unmodeled type (e.g. "7") should return None."""
    action, state = _mock_buy_train("PRR", "7", price=100)
    assert mapper.price_head_slot_for_action(action, state) is None


def test_buy_train_string_owner_resolves(mapper):
    """When ``train.owner`` is a bare string (Rust adapter shape) for a real
    corp, the slot should still resolve correctly."""
    action, state = _mock_buy_train("NYC", "3", price=200, spend_min=1, spend_max=400)
    action.train.owner = "NYC"  # rust adapter shape
    result = mapper.price_head_slot_for_action(action, state)
    assert result is not None
    _, slot, observed, lo, hi = result
    # NYC = corp_idx 1, "3" = train_idx 1 → slot = 6 + 1*6 + 1 = 13
    assert slot == 13
    assert observed == 200


# ----- BuyCompany exhaustive coverage (6 companies → slots 54..59) ------------


@pytest.mark.parametrize(
    "company_id, expected_offset",
    [
        ("SV", 0),
        ("CS", 1),
        ("DH", 2),
        ("MH", 3),
        ("CA", 4),
        ("BO", 5),
    ],
)
def test_buy_company_slot_for_each_company(mapper, company_id, expected_offset):
    # Base offset = len(companies) + len(corps)*len(trains) = 6 + 8*6 = 54.
    expected_slot = 54 + expected_offset
    action, state = _mock_buy_company(company_id, price=90, min_price=10, max_price=200)
    result = mapper.price_head_slot_for_action(action, state)
    assert result is not None
    action_type, slot, observed, lo, hi = result
    assert action_type == "BuyCompany"
    assert slot == expected_slot
    assert observed == 90
    assert lo == 10
    # ``max_price`` is min(company.max_price, buying_power); here buying_power=1000.
    assert hi == 200


def test_buy_company_unknown_company_returns_none(mapper):
    action, state = _mock_buy_company("XXX", price=10)
    assert mapper.price_head_slot_for_action(action, state) is None


def test_buy_company_max_clamped_to_buying_power(mapper):
    """``max_price`` should be ``min(company.max_price, buying_power)``."""
    action, state = _mock_buy_company("SV", price=20, min_price=5, max_price=500, buying_power=100)
    result = mapper.price_head_slot_for_action(action, state)
    assert result is not None
    _, _, _, lo, hi = result
    assert lo == 5
    assert hi == 100, "buying_power should clamp max_price"


# ----- Slot disjointness across action types ----------------------------------


def test_all_slot_ranges_are_disjoint(mapper):
    """The three slot ranges (Bid, BuyTrain, BuyCompany) must not overlap.

    This guards against an off-by-one in the slot-layout constants in
    ``ActionMapper`` getting silently absorbed by an unrelated code change.
    """
    bid_slots = set()
    for c in ("SV", "CS", "DH", "MH", "CA", "BO"):
        action, state = _mock_bid(c, price=10)
        _, slot, *_ = mapper.price_head_slot_for_action(action, state)
        bid_slots.add(slot)

    buy_train_slots = set()
    for corp in _CORPS:
        for t in _TRAINS:
            action, state = _mock_buy_train(corp, t, price=100)
            _, slot, *_ = mapper.price_head_slot_for_action(action, state)
            buy_train_slots.add(slot)

    buy_company_slots = set()
    for c in ("SV", "CS", "DH", "MH", "CA", "BO"):
        action, state = _mock_buy_company(c, price=20)
        _, slot, *_ = mapper.price_head_slot_for_action(action, state)
        buy_company_slots.add(slot)

    # Counts match the design constants.
    assert len(bid_slots) == 6
    assert len(buy_train_slots) == 48
    assert len(buy_company_slots) == 6

    # No overlap across the three groups.
    assert bid_slots.isdisjoint(buy_train_slots)
    assert bid_slots.isdisjoint(buy_company_slots)
    assert buy_train_slots.isdisjoint(buy_company_slots)

    # Full slot universe is contiguous 0..59 (no gaps).
    all_slots = bid_slots | buy_train_slots | buy_company_slots
    assert all_slots == set(range(60))
