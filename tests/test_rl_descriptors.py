"""The RL action descriptors (``engine_rs.rl_layout_info``, ``action_descriptors``) against the 1830
``ActionMapper``: every index of the frozen layout against ``ActionMapper.actions`` (the static
table its decode reads), and every legal index of sampled states against the action
``ActionMapper.map_index_to_action`` decodes there. Also the "not bridge-ready" seam: titles without
a layout refuse every 1830 index path.
"""

import logging
import random

import numpy as np
import pytest

import engine_rs
from rl18xx.agent.alphazero.action_descriptors import NONE, action_descriptors, descriptor_vocab
from rl18xx.agent.alphazero.action_mapper import ActionMapper
from rl18xx.rust_adapter import RustGameAdapter

logging.disable(logging.CRITICAL)

FINGERPRINT_1830 = "e721aa6520116e2a"  # pinned in engine-rs/src/rl/tests.rs too
NOT_READY = ("1822", "1822PNW", "1822MX", "1822CA")  # 1867: tests/test_rl_g1867_index.py


@pytest.fixture(scope="module")
def d30():
    return action_descriptors("1830")


def test_layout_info_shape_and_pins(d30):
    assert d30.total == engine_rs.policy_size_py() == ActionMapper().action_encoding_size == 26537
    assert d30.table.shape == (26537, len(d30.fields)) and d30.table.dtype == np.int32
    assert not d30.table.flags.writeable
    assert d30.fields == (
        "family",
        "ent0_kind",
        "ent0",
        "ent1_kind",
        "ent1",
        "via",
        "hex",
        "city",
        "tile",
        "rot",
        "param",
        "choice",
        "price_kind",
        "flags",
    )
    assert (d30.version, d30.fingerprint) == ("1830-v1", FINGERPRINT_1830)
    # Blocks tile the index space.
    end = 0
    for b in d30.blocks:
        assert b["offset"] == end
        end += b["size"]
    assert end == d30.total
    # The global vocabularies are the same object of the module and of every layout.
    vocab = descriptor_vocab()
    assert tuple(vocab["families"]) == d30.families and tuple(vocab["choices"]) == d30.choices
    assert vocab["none"] == NONE
    # Title vocabularies are title data: the map order is the engine's hex order.
    game = engine_rs.BaseGame({1: "a", 2: "b", 3: "c", 4: "d"})
    assert d30.vocab["hex"] == tuple(h.id for h in game.hexes)
    assert d30.vocab["corporation"] == tuple(c.sym for c in game.corporations)


def _expected(mapper, index):
    """The descriptor ActionMapper.actions[index] implies, as {field: name or number}."""
    cls, args = mapper.actions[index]
    name = cls.__name__
    company_block = index >= mapper.action_offsets["CompanyBuyShares"]
    if name == "Pass":
        return {"family": "Pass"}
    if name == "Bid":
        return {"family": "Bid.company", "ent0": ("company", args[0]), "price_kind": "bid"}
    if name == "Par":
        return {"family": "Par", "ent0": ("corporation", args[0]), "param": args[1]}
    if name == "BuyShares":
        if company_block:
            return {
                "family": "ExchangeShares",
                "via": args[0],
                "ent0": ("corporation", args[1]),
                "choice": f"source:{args[2]}",
            }
        return {"family": "BuyShares", "ent0": ("corporation", args[0]), "choice": f"source:{args[1]}"}
    if name == "SellShares":
        return {"family": "SellShares", "ent0": ("corporation", args[0]), "param": 10 * args[1]}
    if name == "PlaceToken":
        if company_block:
            return {"family": "PlaceToken", "via": args[0], "hex": args[1], "city": args[2]}
        return {"family": "PlaceToken", "hex": args[0], "city": args[1]}
    if name == "LayTile":
        if company_block:
            co, tile, hex_id, rot = args
            return {"family": "LayTile", "via": co, "hex": hex_id, "tile": tile, "rot": rot}
        tile, hex_id, rot = args
        return {"family": "LayTile", "hex": hex_id, "tile": tile, "rot": rot}
    if name == "BuyTrain":
        if args == ["depot"]:
            return {"family": "BuyTrain.depot", "flags": ["train_from_state"]}
        if len(args) == 3 and args[:2] == ["depot", "D"]:
            if args[2] == "full":
                return {"family": "BuyTrain.depot", "ent0": ("train", "D")}
            return {"family": "BuyTrain.exchange", "ent0": ("train", "D"), "flags": ["train_from_state"]}
        if len(args) == 2:
            return {"family": "BuyTrain.pool", "ent0": ("train", args[0])}
        corp, train, price = args
        out = {
            "family": "BuyTrain.corp",
            "ent0": ("corporation", corp),
            "ent1": ("train", train),
            "price_kind": "train_transfer",
        }
        if price != "1":
            out["flags"] = ["dead"]
        return out
    if name == "DiscardTrain":
        return {"family": "DiscardTrain", "ent0": ("train", args[0])}
    if name == "Dividend":
        return {"family": "Dividend", "choice": f"dividend:{args[0]}"}
    if name == "BuyCompany":
        out = {"family": "BuyCompany", "ent0": ("company", args[0]), "price_kind": "company_purchase"}
        if args[1] != "min":
            out["flags"] = ["dead"]
        return out
    if name in ("Bankrupt", "RunRoutes"):
        return {"family": name}
    raise AssertionError(f"unexpected ActionMapper entry {index}: {name} {args}")


def test_every_1830_index_matches_the_action_mapper_table(d30):
    mapper = ActionMapper()
    assert len(mapper.actions) == d30.total
    bad = []
    for index in range(d30.total):
        got = d30.describe(index)
        want = _expected(mapper, index)
        want.setdefault("flags", [])
        got_cmp = {k: v for k, v in got.items() if k != "index"}
        if got_cmp != want:
            bad.append((index, got_cmp, want))
    assert not bad, f"{len(bad)} mismatches, first: {bad[:3]}"


def _token_hex(game, city_ref):
    """The hex of a decoded token's city ("<tile instance>-<city index>"), None if not unique."""
    tile = city_ref.rsplit("-", 1)[0]
    hexes = [h.id for h in game.hexes if tile in (getattr(h.tile, "id", None), getattr(h.tile, "name", None))]
    return hexes[0] if len(hexes) == 1 else None


def _decoded_fields(action, mapper, index, game):
    """What a decoded 1830 action says, in descriptor terms (for the fields the index pins)."""
    d = action.to_dict()
    t = d["type"]
    out = {}
    if t == "bid":
        out["ent0"] = ("company", d["company"])
    elif t == "par":
        out["ent0"] = ("corporation", d["corporation"])
        out["param"] = int(str(d["share_price"]).split(",")[0])
    elif t in ("buy_shares", "sell_shares"):
        corp = d["shares"][0].rsplit("_", 1)[0]
        out["ent0"] = ("corporation", corp)
        if t == "sell_shares":
            out["param"] = int(d["percent"])
        if d["entity_type"] == "company":
            out["via"] = d["entity"]
    elif t == "place_token":
        out["hex"] = _token_hex(game, d["city"])
        out["city"] = int(d["city"].rsplit("-", 1)[1])
        if d["entity_type"] == "company":
            out["via"] = d["entity"]
    elif t == "lay_tile":
        out["hex"] = d["hex"]
        out["tile"] = d["tile"].rsplit("-", 1)[0]
        out["rot"] = int(d["rotation"])
        if d["entity_type"] == "company":
            out["via"] = d["entity"]
    elif t == "buy_train":
        out["train"] = d["train"].rsplit("-", 1)[0]
        out["exchange"] = d.get("exchange")
    elif t == "discard_train":
        out["ent0"] = ("train", d["train"].rsplit("-", 1)[0])
    elif t == "dividend":
        out["choice"] = f"dividend:{d['kind']}"
    elif t == "buy_company":
        out["ent0"] = ("company", d["company"])
    return t, out


def _check_decode(desc, t, decoded):
    """Whether the decoded action agrees with the descriptor on every field both name."""
    fam = desc["family"]
    family_types = {
        "Pass": "pass",
        "Bid.company": "bid",
        "Par": "par",
        "BuyShares": "buy_shares",
        "ExchangeShares": "buy_shares",
        "SellShares": "sell_shares",
        "PlaceToken": "place_token",
        "LayTile": "lay_tile",
        "BuyTrain.depot": "buy_train",
        "BuyTrain.pool": "buy_train",
        "BuyTrain.exchange": "buy_train",
        "BuyTrain.corp": "buy_train",
        "DiscardTrain": "discard_train",
        "Dividend": "dividend",
        "BuyCompany": "buy_company",
        "Bankrupt": "bankrupt",
        "RunRoutes": "run_routes",
    }
    if family_types[fam] != t:
        return f"type {t} for {fam}"
    for key in ("ent0", "param", "via", "tile", "rot", "choice", "city"):
        if key in decoded and key in desc and decoded[key] != desc[key]:
            return f"{key}: decoded {decoded[key]} vs descriptor {desc[key]}"
    if fam == "LayTile" and "via" in decoded and "via" in desc and decoded["via"] != desc["via"]:
        return "via"
    if fam == "LayTile" and decoded.get("hex") != desc["hex"]:
        return f"hex {decoded.get('hex')} vs {desc['hex']}"
    if fam == "PlaceToken" and decoded.get("hex") not in (None, desc["hex"]):
        return f"hex {decoded.get('hex')} vs {desc['hex']}"
    if fam.startswith("BuyTrain"):
        if "train_from_state" not in desc["flags"]:
            ent = desc["ent1"] if fam == "BuyTrain.corp" else desc["ent0"]
            if ent != ("train", decoded["train"]):
                return f"train {decoded['train']} vs {ent}"
        if fam == "BuyTrain.exchange" and not decoded["exchange"]:
            return "no trade-in"
        if fam == "BuyTrain.depot" and decoded["exchange"]:
            return "a trade-in on a depot slot"
    return None


@pytest.mark.parametrize("seed", [3, 17])
def test_legal_indices_decode_to_what_their_descriptors_say(d30, seed):
    """Python's ActionMapper decode (on the adapter) at every state of a random native walk agrees
    with the descriptor of every legal index."""
    rng = random.Random(seed)
    mapper = ActionMapper()
    n = rng.choice([3, 4, 5])
    game = engine_rs.BaseGame({i: f"P{i}" for i in range(1, n + 1)})
    checked = 0
    families = set()
    for step in range(600):
        if game.finished:
            break
        adapter = RustGameAdapter(game)
        indices, price_ranges, _ = mapper.get_legal_actions_factored(adapter)
        assert sorted(indices) == game.factored_legal_indices()
        rows = d30.gather(indices)
        assert not (rows[:, d30.col("flags")] & d30.flags["dead"]).any(), "a legal index is dead"
        for index in indices:
            desc = d30.describe(index)
            pr = price_ranges.get(index)
            # A real price range only on priced indices.
            if pr is not None and pr[0] != pr[1]:
                assert "price_kind" in desc, (index, desc, pr)
            if desc["family"] == "RunRoutes":
                families.add("RunRoutes")
                continue  # the auto-router decode is slow; nothing to compare
            action = (
                mapper.map_index_to_action_with_price(index, adapter, int(pr[0]))
                if pr is not None
                else mapper.map_index_to_action(index, adapter)
            )
            t, decoded = _decoded_fields(action, mapper, index, game)
            problem = _check_decode(desc, t, decoded)
            assert problem is None, f"seed {seed} step {step} index {index} {desc}: {problem} ({action.to_dict()})"
            checked += 1
            families.add(desc["family"])
        index = rng.choice(indices)
        pr = price_ranges.get(index)
        game.apply_action_index(index, rng.randint(*pr) if pr else None)
    assert checked > 1000
    assert {"Bid.company", "Par", "BuyShares", "SellShares", "LayTile", "Dividend"} <= families, families


@pytest.mark.parametrize("title", NOT_READY)
def test_titles_without_a_layout_are_not_bridge_ready(title):
    status = engine_rs.rl_title_status(title)
    assert status == {"title": title, "layout_ready": False, "layout_kind": "none", "encoder_ready": False}
    with pytest.raises(RuntimeError, match="not bridge-ready"):
        engine_rs.rl_layout_info(title)
    with pytest.raises(RuntimeError, match="not bridge-ready"):
        action_descriptors(title)
    with pytest.raises(RuntimeError, match="not bridge-ready"):
        engine_rs.policy_size_for_py(title)
    first = 1001 if title.startswith("1822") else 1
    game = engine_rs.BaseGame.new_titled(title, {first + i: f"Player {i + 1}" for i in range(4)}, None, 1)
    for call in (
        lambda: game.factored_legal_indices(),
        lambda: game.apply_action_index(0, None),
        lambda: game.price_range_for_index(0),
        lambda: game.price_head_slot_for_index(0),
        lambda: engine_rs.RustMCTSPlayer(game),
    ):
        with pytest.raises(RuntimeError, match="not bridge-ready"):
            call()


def test_rl_title_status_rejects_unknown_titles():
    with pytest.raises(ValueError):
        engine_rs.rl_title_status("1899")
    with pytest.raises(RuntimeError, match="unknown game title"):
        engine_rs.rl_layout_info("1899")
