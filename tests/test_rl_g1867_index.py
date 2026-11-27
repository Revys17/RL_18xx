"""1867's RL layout ``1867-v1`` (kind "generator") through the Python surface: the layout, the
state's legal indices over the legal-move generator, decode / apply by index, price ranges, the
logged action dicts, and the 1830 index path refusing the title. The exhaustive checks are the
in-crate fuzz (``rl::g1867::tests``) and ``scripts/g1867_completeness.py --index`` on the corpus.
"""

import json
import random

import pytest

import engine_rs
from rl18xx.agent.alphazero.action_descriptors import action_descriptors
from rl18xx.agent.alphazero.game_records import make_game_record
from rl18xx.agent.alphazero.titles import layout_metadata, new_game

FINGERPRINT_1867 = "4987cd19b7e912e2"  # pinned in engine-rs/src/rl/tests.rs too
TOTAL_1867 = 38875

# The families each 18xx.games action type may index to.
FAMILIES = {
    "pass": {"Pass"},
    "bid": {"Bid.company", "Bid.corporation"},
    "par": {"Par"},
    "buy_shares": {"BuyShares", "RedeemShares"},
    "sell_shares": {"SellShares"},
    "place_token": {"PlaceToken"},
    "remove_token": {"RemoveToken"},
    "lay_tile": {"LayTile"},
    "buy_train": {"BuyTrain.depot", "BuyTrain.pool", "BuyTrain.exchange", "BuyTrain.corp"},
    "discard_train": {"DiscardTrain"},
    "dividend": {"Dividend"},
    "buy_company": {"BuyCompany"},
    "run_routes": {"RunRoutes"},
    "merge": {"Merge"},
    "convert": {"Convert"},
    "choose": {"Choose"},
}
PRICED = {"Bid.company", "Bid.corporation", "BuyTrain.corp", "BuyCompany"}


@pytest.fixture(scope="module")
def d67():
    return action_descriptors("1867")


def test_layout(d67):
    assert (d67.version, d67.fingerprint, d67.total, d67.kind) == ("1867-v1", FINGERPRINT_1867, TOTAL_1867, "generator")
    assert engine_rs.rl_title_status("1867") == {
        "title": "1867",
        "layout_ready": True,
        "layout_kind": "generator",
        "encoder_ready": False,
    }
    names = [b["name"] for b in d67.blocks]
    assert names[0] == "Pass" and names[-1] == "LayTile"
    assert {b["name"]: b["price_kind"] for b in d67.blocks if b["price_kind"]} == {
        "Bid.company": "bid",
        "Bid.corporation": "bid_range",  # range-relative cells (the price census)
        "BuyTrain.corp": "train_transfer",
        "BuyCompany": "company_purchase",
    }
    kinds = dict(zip(d67.price_kinds, d67.price_cell_types))
    assert kinds["bid"] == "Bid" and kinds["bid_range"] == "BuyTrain"
    end = 0
    for b in d67.blocks:
        assert b["offset"] == end
        end += b["size"]
    assert end == d67.total
    assert len({tuple(r) for r in d67.table.tolist()}) == d67.total, "every index means something else"
    ex = d67.block_of(d67.total - 1)
    assert ex["name"] == "LayTile"
    exchange = next(b for b in d67.blocks if b["name"] == "BuyTrain.exchange")
    assert d67.describe(exchange["offset"] + 1) == {
        "index": exchange["offset"] + 1,
        "family": "BuyTrain.exchange",
        "ent0": ("train", "8"),
        "ent1": ("train", "5"),
        "choice": "train_source:pool",
        "flags": [],
    }


def test_the_1830_path_refuses_1867():
    game = new_game("1867", 4, seed=1)
    for call in (
        lambda: game.factored_legal_indices(),
        lambda: game.apply_action_index(0, None),
        lambda: game.price_range_for_index(0),
    ):
        with pytest.raises(RuntimeError, match="generator"):
            call()
    # The search tree takes 1867's generator moves (mcts.rs TreeMoves::Generator, 2026-11-27).
    assert engine_rs.RustMCTSPlayer(game).legal_action_indices_at_root() == game.rl_legal_indices()
    with pytest.raises(RuntimeError, match="no generator layout"):
        new_game("1830", 4)._game.rl_legal_indices()


@pytest.mark.parametrize("players,seed", [(3, 11), (6, 12)])
def test_walk_by_index(d67, players, seed):
    """Indices are distinct and ascending, each descriptor's family fits its move, decode gives the
    move back at any price on its range, and the game advances by index with the move's dict logged."""
    rng = random.Random(seed)
    game = new_game("1867", players, seed=seed)
    seen = set()
    for _ in range(400):
        if game.finished:
            break
        moves = json.loads(game.rl_legal_moves_json())
        idx = [m["index"] for m in moves]
        assert idx == sorted(set(idx)) and idx == game.rl_legal_indices()
        assert set(idx) <= set(game.rl_legal_indices("rules"))
        for m in moves:
            fam = d67.families[d67.table[m["index"], d67.col("family")]]
            assert fam == m["block"] and fam in FAMILIES[m["wire"]["type"]], m
            assert (m["range"] is not None) == (fam in PRICED), m
            seen.add(fam)
        m = rng.choice(moves)
        price = None
        if m["range"]:
            lo, hi, step = m["range"]
            assert tuple(m["range"]) == game.rl_price_range(m["index"])
            price = lo + step * rng.randint(0, (hi - lo) // step)
            with pytest.raises(RuntimeError, match="outside the move's range"):
                game.rl_decode_index(m["index"], hi + step)
        wire = json.loads(game.rl_decode_index(m["index"], price))
        expected = dict(m["wire"], **({"price": price} if price is not None else {}))
        assert wire == expected
        game.rl_apply_index(m["index"], price)
        assert game.raw_actions[-1] == expected
    assert {"Pass", "Bid.company", "Bid.corporation", "BuyShares", "LayTile", "RunRoutes"} <= seen, seen
    with pytest.raises(RuntimeError, match="not legal"):
        game.rl_apply_index(TOTAL_1867 - 1)
    with pytest.raises(RuntimeError, match="out of bounds"):
        game.rl_decode_index(TOTAL_1867)

    # The logged dicts replay in a fresh engine, and a record carries the layout.
    replay = new_game("1867", players, seed=seed)
    for a in game.raw_actions:
        replay.process_action(dict(a))
    assert replay.rl_legal_indices() == game.rl_legal_indices()
    assert [p.cash for p in replay.players] == [p.cash for p in game.players]
    record = make_game_record(game.raw_actions, players, title="1867", seed=seed)
    assert record["rl18xx"]["layout"] == layout_metadata("1867")
    assert record["rl18xx"]["layout"]["layout_fingerprint"] == FINGERPRINT_1867


def test_price_range_carries_the_step_on_every_layout():
    """``rl_price_range`` on 1830 is ``price_range_for_index`` plus the step (the $5 bid ladder, 1
    otherwise), and its cells are the legacy ones; on 1867 the step is the move's grid and the cells,
    in step units, hold grid prices only."""
    from rl18xx.agent.alphazero import price_pmf
    from rl18xx.agent.alphazero.action_descriptors import range_price_cells

    d30 = action_descriptors("1830")
    assert (d30.price_units, d30.price_grid) == ("price", False)
    rng = random.Random(5)
    game = new_game("1830", 4)._game
    priced = 0
    for _ in range(300):
        if game.finished:
            break
        legal = game.factored_legal_indices()
        for i in legal:
            r = game.price_range_for_index(i)
            got = game.rl_price_range(i)
            if r is None:
                assert got is None
                continue
            family = d30.families[d30.table[i, d30.col("family")]]
            assert got == (r[0], r[1], 5 if family == "Bid.company" else 1), (i, family)
            kind = d30.table[i, d30.col("price_kind")]
            if kind >= 0 and r[1] > r[0]:
                ct = d30.price_cell_types[kind]
                legacy = price_pmf.price_cells(ct, r[0], r[1], got[2])
                cells = range_price_cells(ct, *got, grid=False)
                assert (cells.counts == legacy.counts).all()
                assert all(cells.cell_of(p) == legacy.cell_of(p) for p in range(r[0], r[1] + 1))
                priced += 1
        i = rng.choice(legal)
        r = game.price_range_for_index(i)
        game.apply_action_index(i, rng.randint(*r) if r else None)
    assert priced > 20
    with pytest.raises(RuntimeError, match="not legal"):
        game.rl_price_range(26536 if 26536 not in game.factored_legal_indices() else 0)

    d67 = action_descriptors("1867")
    assert (d67.price_units, d67.price_grid) == ("step", True)
    cells = range_price_cells("BuyTrain", 100, 300, 5, grid=True)  # bid_range's cells of a minor bid
    assert int(cells.counts.sum()) == 41
    prices = sorted({cells.member(c, u) for c in range(len(cells.counts)) for u in range(int(cells.counts[c]))})
    assert prices == list(range(100, 301, 5))
    assert all(cells.member(cells.cell_of(p), 0) <= p for p in prices)
    with pytest.raises(ValueError, match="grid"):
        cells.cell_of(102)
