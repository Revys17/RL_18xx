"""The 1822 family's RL layouts (``1822-v1``, ``1822PNW-v1``, ``1822MX-v1``, ``1822CA-v1``; kind
"generator") through the Python surface: the layouts, the state's legal indices over the family's
legal-move generator, decode / apply by index (against ``process_action`` of the decoded dict),
price ranges, the logged action dicts, and the 1830 index path refusing the titles. The exhaustive
checks are the in-crate fuzz (``rl::g1822::tests``) and ``scripts/g1822_completeness.py --index`` on
the corpora.
"""

import json
import random

import pytest

import engine_rs
from rl18xx.agent.alphazero.action_descriptors import action_descriptors
from rl18xx.agent.alphazero.game_records import make_game_record
from rl18xx.agent.alphazero.titles import layout_metadata, new_game

# Pinned in engine-rs/src/rl/tests.rs too: (version, fingerprint, total).
LAYOUTS = {
    "1822": ("1822-v1", "e14224b87f0bceda", 747618),
    "1822PNW": ("1822PNW-v1", "9e28492a5e6f1838", 197711),
    "1822MX": ("1822MX-v1", "3e7e94f94384898b", 256264),
    "1822CA": ("1822CA-v1", "c65c38fd4001b44f", 390157),
}

# The families each 18xx.games action type may index to.
FAMILIES = {
    "pass": {"Pass"},
    "bid": {"Bid.company", "Bid.token"},  # (block Bid.minor: family Bid.company)
    "par": {"Par"},
    "buy_shares": {"BuyShares", "RedeemShares"},
    "sell_shares": {"SellShares", "IssueShares"},
    "place_token": {"PlaceToken"},
    "remove_token": {"RemoveToken"},
    "hex_token": {"HexToken"},
    "lay_tile": {"LayTile"},
    "buy_train": {"BuyTrain.depot", "BuyTrain.pool", "BuyTrain.exchange", "BuyTrain.corp"},
    "discard_train": {"DiscardTrain"},
    "dividend": {"Dividend"},
    "run_routes": {"RunRoutes"},
    "merge": {"Merge"},
    "choose": {"Choose", "Dividend"},
    "choose_ability": {"ChooseAbility"},
    "acquire_company": {"AcquireCompany"},
    "assign": {"Assign"},
    "payoff_player_debt": {"PayoffPlayerDebt"},
    "payoff_player_debt_partial": {"PayoffPlayerDebtPartial"},
}
PRICED = {
    "Bid.company": "bid",
    "Bid.minor": "bid_range",
    "Bid.token": "bid",
    "BuyTrain.corp": "train_transfer",
    "PayoffPlayerDebtPartial": "amount",
}
PLAYERS = {"1822": (3, 6), "1822PNW": (3, 5), "1822MX": (3, 5), "1822CA": (3, 6)}


def _snap(g) -> tuple:
    """The rules state (``g1822_snapshot`` plus holdings, companies, tiles and tokens, depot)."""
    return (
        g.g1822_snapshot(),
        tuple(p.cash for p in g.players),
        tuple(
            (
                c.sym,
                c.cash,
                c.share_price.price if c.share_price else None,
                tuple(sorted(t.name for t in c.trains)),
                tuple(sorted((str(s.owner), s.percent, s.president) for s in c.shares)),
            )
            for c in g.corporations
        ),
        tuple((co.sym, str(co.owner), co.closed) for co in g.companies),
        tuple(
            (
                h.id,
                h.tile.name,
                h.tile.rotation,
                tuple(tuple(t.corporation_id if t else None for t in c.tokens) for c in h.tile.cities),
            )
            for h in g.hexes
        ),
        tuple(sorted(t.name for t in g.depot.trains)),
        tuple(sorted(t.name for t in g.depot.discarded)),
        g.bank.cash,
        bool(g.finished),
    )


@pytest.mark.parametrize("title", list(LAYOUTS))
def test_layout(title):
    d = action_descriptors(title)
    version, fingerprint, total = LAYOUTS[title]
    assert (d.version, d.fingerprint, d.total) == (version, fingerprint, total)
    assert (d.kind, d.price_units) == ("generator", "step")
    assert engine_rs.rl_title_status(title) == {
        "title": title,
        "layout_ready": True,
        "layout_kind": "generator",
        "encoder_ready": False,
    }
    names = [b["name"] for b in d.blocks]
    assert names[0] == "Pass" and "LayTile" in names
    assert {b["name"]: b["price_kind"] for b in d.blocks if b["price_kind"]} == {
        k: v for k, v in PRICED.items() if k in names
    }
    end = 0
    for b in d.blocks:
        assert b["offset"] == end
        end += b["size"]
    assert end == d.total
    assert layout_metadata(title) == {
        "title": title,
        "layout_kind": "generator",
        "layout_version": version,
        "layout_fingerprint": fingerprint,
    }


def test_combo_lays_and_state_list_choices_are_described():
    d = action_descriptors("1822")
    combo = next(b for b in d.blocks if b["name"] == "LayTile:P12+P10")
    row = d.describe(combo["offset"])
    assert (row["family"], row["via"], row["ent0"]) == ("LayTile", "P12", ("company", "P10"))
    pullman = next(b for b in d.blocks if b["name"] == "Choose.pullman")
    row = d.describe(pullman["offset"] + 7)  # train axis entry 1 ("3"), rank 1
    assert (row["choice"], row["ent0"], row["param"]) == ("route:pullman", ("train", "3"), 1)
    mx = action_descriptors("1822MX")
    bid = next(b for b in mx.blocks if b["name"] == "Bid.token")
    row = mx.describe(bid["offset"])
    assert row["family"] == "Bid.token" and "hex" in row and row["city"] == 0 and row["price_kind"] == "bid"
    pres = next(b for b in mx.blocks if b["name"] == "BuyShares.president")
    assert mx.describe(pres["offset"])["flags"] == ["president_cert"]
    assert {"BC", "P1", "P2"} <= set(mx.vocab["tile"]), "the pseudo-tiles are tiles of the manifest"


@pytest.mark.parametrize("title", list(LAYOUTS))
def test_the_1830_path_refuses_the_family(title):
    game = new_game(title, 4, seed=1)
    for call in (
        lambda: game.factored_legal_indices(),
        lambda: game.apply_action_index(0, None),
        lambda: game.price_range_for_index(0),
        lambda: engine_rs.RustMCTSPlayer(game),
    ):
        with pytest.raises(RuntimeError, match="generator"):
            call()


@pytest.mark.parametrize("title", list(LAYOUTS))
def test_walk_by_index(title):
    """Indices are distinct and ascending, each descriptor's family fits its move, decode gives the
    move back at any price on its range, applying by index leaves the state ``process_action`` of
    the decoded dict leaves, and the game advances by index with the move's dict logged."""
    d = action_descriptors(title)
    block_family = {b["name"]: b["family"] for b in d.blocks}
    lo, hi = PLAYERS[title]
    players = lo if title != "1822" else hi
    rng = random.Random(len(title))
    game = new_game(title, players, seed=3)
    seen = set()
    for _ in range(300):
        if game.finished:
            break
        moves = json.loads(game.rl_legal_moves_json())
        if not moves:
            break  # a Ruby dead end (the engine mirrors Ruby raising)
        idx = [m["index"] for m in moves]
        assert idx == sorted(set(idx)) and idx == game.rl_legal_indices()
        assert set(idx) <= set(game.rl_legal_indices("rules"))
        for m in moves:
            fam = d.families[d.table[m["index"], d.col("family")]]
            assert fam == block_family[m["block"]], m
            assert fam in FAMILIES[m["wire"]["type"]], m
            assert (m["range"] is not None) == (m["block"] in PRICED), m
            seen.add(m["block"].split(":")[0])
        non_pass = [m for m in moves if m["wire"]["type"] != "pass"]
        m = rng.choice(non_pass) if non_pass and rng.random() < 0.8 else rng.choice(moves)
        price = None
        if m["range"]:
            lo_p, hi_p, step = m["range"]
            assert tuple(m["range"]) == game.rl_price_range(m["index"])
            price = lo_p + step * rng.randint(0, min(10, (hi_p - lo_p) // step))
            with pytest.raises(RuntimeError, match="outside the move's range"):
                game.rl_decode_index(m["index"], hi_p + step)
        wire = json.loads(game.rl_decode_index(m["index"], price))
        key = {"bid": "price", "buy_train": "price", "payoff_player_debt_partial": "amount"}.get(m["wire"]["type"])
        expected = dict(m["wire"], **({key: price} if price is not None else {}))
        assert wire == expected
        by_dict = game.pickle_clone()
        by_dict.process_action(dict(wire))
        game.rl_apply_index(m["index"], price)
        assert game.raw_actions[-1] == expected
        assert _snap(game) == _snap(by_dict), m
    assert {"Pass", "Bid.company", "LayTile"} <= seen, seen
    with pytest.raises(RuntimeError, match="out of bounds"):
        game.rl_decode_index(d.total)

    # The logged dicts replay in a fresh engine, and a record carries the layout.
    replay = new_game(title, players, seed=3)
    for a in game.raw_actions:
        replay.process_action(dict(a))
    assert _snap(replay) == _snap(game)
    assert replay.rl_legal_indices() == game.rl_legal_indices()
    record = make_game_record(game.raw_actions, players, title=title, seed=3)
    assert record["rl18xx"]["layout"] == layout_metadata(title)
