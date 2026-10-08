"""Move descriptions (rl18xx.agent.advisor.describe), one per action type, and on real positions."""

import pytest

from rl18xx.agent.advisor.advisor import describe_index, has_price_choice
from rl18xx.agent.advisor.describe import DescribeContext, describe_action, describe_actor, map_positions
from rl18xx.agent.advisor.live_game import LiveGame

from .conftest import truncated


@pytest.fixture
def ctx():
    return DescribeContext(
        player_names={1: "alice", 2: "bob"},
        company_names={
            "SV": "Schuylkill Valley",
            "CS": "Champlain & St.Lawrence",
            "DH": "Delaware & Hudson",
            "MH": "Mohawk & Hudson",
        },
        company_owners={"MH": "alice", "CS": "PRR", "DH": "NYC"},
        corporation_presidents={"PRR": "alice", "NYC": "bob"},
        hex_names={"H12": "Altoona", "F16": "Scranton"},
        city_counts={"E5": 2, "H12": 1},
        round_type="Operating",
        step="LayTile",
    )


@pytest.mark.parametrize(
    "action, step, expected",
    [
        ({"type": "pass", "entity": "PRR"}, "LayTile", "Pass: lay no tile"),
        ({"type": "pass", "entity": "PRR"}, "PlaceToken", "Pass: place no token"),
        ({"type": "pass", "entity": "PRR"}, "BuyTrain", "Pass: buy no train"),
        ({"type": "pass", "entity": "PRR"}, "BuyCompany", "Pass: buy no private"),
        ({"type": "pass", "entity": "1"}, "WaterfallAuction", "Pass (no bid)"),
        ({"type": "pass", "entity": "1"}, "BuySellParShares", "Pass"),
        ({"type": "pass", "entity": "DH"}, "SpecialToken", "Pass with Delaware & Hudson (DH)"),
        (
            {"type": "bid", "entity": "1", "company": "CS", "price": "45"},
            None,
            "Bid $45 on Champlain & St.Lawrence (CS)",
        ),
        ({"type": "par", "entity": "2", "corporation": "NYC", "share_price": "67"}, None, "Par NYC at $67"),
        ({"type": "par", "entity": "2", "corporation": "NYC", "share_price": "100,0,6"}, None, "Par NYC at $100"),
        (
            {"type": "buy_shares", "entity": "1", "corporation": "PRR", "percent": "10", "source": "ipo"},
            None,
            "Buy a 10% PRR share from the IPO",
        ),
        (
            {"type": "buy_shares", "entity": "1", "corporation": "PRR", "percent": "10", "source": "market"},
            None,
            "Buy a 10% PRR share from the market",
        ),
        (
            {"type": "buy_shares", "entity": "MH", "corporation": "NYC", "percent": "10", "source": "ipo"},
            None,
            "Exchange Mohawk & Hudson (MH) for a 10% NYC share from the IPO",
        ),
        (
            {"type": "sell_shares", "entity": "1", "corporation": "PRR", "percent": "30"},
            None,
            "Sell 30% of PRR (3 shares)",
        ),
        (
            {"type": "sell_shares", "entity": "1", "corporation": "PRR", "percent": "10"},
            None,
            "Sell 10% of PRR (1 share)",
        ),
        (
            {"type": "lay_tile", "entity": "PRR", "hex": "H12", "tile": "57-0", "rotation": "2"},
            None,
            "Lay tile #57 on H12 (Altoona), rotation 2",
        ),
        (
            {"type": "lay_tile", "entity": "CS", "hex": "B20", "tile": "3-0", "rotation": "0"},
            None,
            "Use Champlain & St.Lawrence (CS) to lay tile #3 on B20, rotation 0",
        ),
        ({"type": "place_token", "entity": "NYC", "hex": "E5", "city_index": "1"}, None, "Place a token on E5, city 2"),
        (
            {"type": "place_token", "entity": "NYC", "hex": "H12", "city_index": "0"},
            None,
            "Place a token on H12 (Altoona)",
        ),
        (
            {"type": "place_token", "entity": "DH", "hex": "F16", "city_index": "0"},
            None,
            "Use Delaware & Hudson (DH) to place a token on F16 (Scranton)",
        ),
        ({"type": "run_routes", "entity": "PRR"}, None, "Run trains (the engine's best routes)"),
        ({"type": "dividend", "entity": "PRR", "kind": "payout"}, None, "Pay out"),
        ({"type": "dividend", "entity": "PRR", "kind": "withhold"}, None, "Withhold"),
        (
            {"type": "buy_train", "entity": "PRR", "train": "3-1", "variant": "3", "from": "depot", "price": "180"},
            None,
            "Buy a 3-train from the bank for $180",
        ),
        (
            {"type": "buy_train", "entity": "PRR", "train": "2-4", "variant": "2", "from": "NYC", "price": "1"},
            None,
            "Buy a 2-train from NYC (bob) for $1",
        ),
        (
            {"type": "buy_train", "entity": "PRR", "variant": "D", "from": "depot", "price": "800", "trade_in": True},
            None,
            "Buy a D-train for $800, trading in a 4, 5 or 6-train",
        ),
        (
            {"type": "buy_train", "entity": "PRR", "train": "D-0", "price": 800, "exchange": "5-1"},
            None,
            "Buy a D-train for $800, trading in a 5-train",
        ),
        ({"type": "discard_train", "entity": "PRR", "train": "4-1"}, None, "Discard a 4-train"),
        (
            {"type": "buy_company", "entity": "NYC", "company": "MH", "price": "55"},
            None,
            "Buy Mohawk & Hudson (MH) from alice for $55",
        ),
        ({"type": "bankrupt", "entity": "PRR"}, None, "Declare bankruptcy"),
        ({"type": "choose", "entity": "PRR", "choice": "x"}, None, "choose (choice=x)"),
    ],
)
def test_each_action_type_reads_as_a_sentence(ctx, action, step, expected):
    if step:
        ctx.step = step
    assert describe_action(action, ctx) == expected


def test_the_waterfall_sells_its_cheapest_private_and_auctions_the_rest(ctx):
    ctx.first_unsold_private = "SV"
    sale = {"type": "bid", "entity": "1", "company": "SV", "price": "20", "fixed_price": True}
    assert describe_action(sale, ctx) == "Buy Schuylkill Valley (SV) for $20"
    assert describe_action({"type": "bid", "entity": "1", "company": "CS", "price": "45"}, ctx).startswith("Bid $45")


def test_actors(ctx):
    assert describe_actor({"entity": "2"}, ctx) == "bob"
    assert describe_actor({"entity": "PRR"}, ctx) == "PRR (alice)"
    assert describe_actor({"entity": "MH"}, ctx) == "MH (alice)"
    assert describe_actor({"entity": "B&O"}, ctx) == "B&O"


def test_the_context_of_a_new_game(game):
    live = LiveGame(truncated(game, 0))
    ctx = DescribeContext.from_game(live.rust, live.names)
    assert ctx.round_type == "Auction" and ctx.step == "WaterfallAuction"
    assert ctx.first_unsold_private == "SV"
    assert ctx.company_names["MH"] == "Mohawk & Hudson"
    assert ctx.hex_names["H12"] == "Altoona"


def test_every_legal_move_along_the_test_game_has_a_description(game):
    live = LiveGame(truncated(game, 0))
    types = set()
    for count in range(0, len(game["actions"]), 7):
        live.update(truncated(game, count))
        state = live.rust
        rs = state._game
        ctx = DescribeContext.from_game(state, live.names)
        for index in rs.factored_legal_indices():
            price_range = rs.price_range_for_index(index)
            price = int(price_range[0]) if price_range is not None else None
            fixed = price_range is not None and not has_price_choice(rs, index)
            entry = describe_index(state, index, price, ctx, fixed)
            assert "=" not in entry["description"], entry
            types.add(entry["type"])
            if entry["type"] == "lay_tile":
                assert entry["map"] == map_positions()["hexes"][entry["hex"]]
                assert entry["tile"].isalnum() and 0 <= entry["rotation"] < 6
    assert {"bid", "pass", "par", "buy_shares", "sell_shares", "lay_tile", "dividend", "buy_train"} <= types


def test_map_positions_follow_the_sites_hex_layout():
    positions = map_positions()
    assert positions["layout"] == "pointy"
    hexes = positions["hexes"]
    assert len(hexes) > 90 and len({tuple(xy) for xy in hexes.values()}) == len(hexes)
    xs = sorted({x for x, _ in hexes.values()})
    ys = sorted({y for _, y in hexes.values()})
    assert xs[0] == 100 and ys[0] == 100
    # Pointy hexes: columns half a hex (100 * sqrt(3) / 2) apart, rows 150 apart.
    assert xs[1] - xs[0] == pytest.approx(86.6, abs=0.02)
    assert ys[1] - ys[0] == pytest.approx(150)
