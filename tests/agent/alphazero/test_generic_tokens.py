"""The title-generic token encoder (``engine_rs.rl_encode_batch``, RL bridge
milestone B1) on random walks of 1830 (2-6 players, the 1830 legal-index API)
and 1867 (3-6 players, ``g1867_legal_moves_json``): shapes, masks, pointer
validity, determinism, value ranges, map de-duplication, and a forward /
backward pass of the small ``TokenTrunkStub`` on a mixed batch.

The in-crate tests (``cargo test rl_encode``) walk the 1822 family too and pin
the vocabularies; ``-m benchmark`` times the batch call against 1830's
``encode_games``.
"""

import json
import random
import time

import numpy as np
import pytest
import torch

engine_rs = pytest.importorskip("engine_rs")

from rl18xx.agent.alphazero.action_descriptors import action_descriptors  # noqa: E402
from rl18xx.agent.alphazero.generic_tokens import (  # noqa: E402
    TokenStubConfig,
    TokenTrunkStub,
    encode_batch,
    raw_game,
    schema,
    spec_info,
    to_torch,
)
from rl18xx.agent.alphazero.titles import new_game  # noqa: E402
from scripts.g1867_fuzz import pick as pick_1867  # noqa: E402

SCHEMA = schema()
ENT_TYPES = {name: i for i, name in enumerate(SCHEMA["entity_types"])}
CORP_TYPES = {ENT_TYPES[k] for k in ("major", "minor", "national")}


def walk_1830(num_players: int, seed: int, steps: int = 600, every: int = 3) -> list:
    rng = random.Random(seed)
    # auction_unlock: uniformly random play would otherwise lock the waterfall auction
    game = raw_game(new_game("1830", num_players, auction_unlock=True))
    positions = []
    for step in range(steps):
        if game.finished:
            break
        legal = [int(i) for i in game.factored_legal_indices()]
        if not legal:
            break
        if step % every == 0:
            positions.append(game.pickle_clone())
        index = rng.choice(legal)
        price_range = game.price_range_for_index(index)
        price = rng.randint(*price_range) if price_range else None
        game.apply_action_index(index, price)
    return positions


def walk_1867(num_players: int, seed: int, steps: int = 600, every: int = 3) -> list:
    rng = random.Random(seed * 1_000_003 + num_players)
    game = new_game("1867", num_players, seed=seed)
    positions = []
    for step in range(steps):
        if game.finished:
            break
        moves = json.loads(game.g1867_legal_moves_json(None))
        if not moves:
            break
        if step % every == 0:
            positions.append(game.pickle_clone())
        game.process_action(pick_1867(moves, rng))
    return positions


@pytest.fixture(scope="module")
def positions() -> dict:
    out = {}
    for n in (2, 3, 4, 5, 6):
        out[("1830", n)] = walk_1830(n, seed=n)
    for n in (3, 4, 5, 6):
        out[("1867", n)] = walk_1867(n, seed=n)
    return out


def check_batch(arrays: dict, games: list) -> None:
    b = len(games)
    f_hex, f_ent, s_max = SCHEMA["f_hex"], SCHEMA["f_ent"], SCHEMA["s_max"]
    hex_feat, ent_feat = arrays["hex_feat"], arrays["ent_feat"]
    m, h = hex_feat.shape[:2]
    e = ent_feat.shape[1]
    assert hex_feat.shape == (m, h, f_hex) and hex_feat.dtype == np.float32
    assert arrays["hex_ptr"].shape == arrays["hex_res"].shape == arrays["hex_slot_city"].shape == (m, h, s_max)
    assert arrays["hex_ids"].shape == (m, h, 2)
    assert arrays["hex_mask"].shape == (m, h) and arrays["hex_mask"].dtype == np.bool_
    assert ent_feat.shape == (b, e, f_ent) and ent_feat.dtype == np.float32
    assert arrays["ent_type"].shape == arrays["ent_mask"].shape == (b, e)
    assert arrays["ent_ptr"].shape == (b, e, SCHEMA["p_ent"])
    assert arrays["ent_hex"].shape == (b, e, SCHEMA["p_hex"])
    r = arrays["rel_val"].shape[1]
    assert arrays["rel"].shape == (b, r, 3) and arrays["rel_mask"].shape == (b, r)
    assert arrays["meta"].shape == (b, SCHEMA["meta_len"])
    map_index = arrays["map_index"]
    if map_index is None:
        assert m == b
        map_index = np.arange(b)

    clip = SCHEMA["feature_clip"]
    for key in ("hex_feat", "ent_feat", "rel_val"):
        assert np.isfinite(arrays[key]).all(), key
        assert np.abs(arrays[key]).max(initial=0) <= clip, key

    for i, game in enumerate(games):
        info = spec_info(game.title)
        nh, ne = info["num_hexes"], info["num_entities"]
        k = map_index[i]
        hm = arrays["hex_mask"][k]
        assert hm[:nh].all() and not hm[nh:].any()
        types = arrays["ent_type"][i]
        mask = arrays["ent_mask"][i]
        assert (mask == (types != 0)).all()
        static = info["entity_types"]
        num_players = len(game.players)
        for j in range(ne):
            expect = 0 if 1 <= j <= SCHEMA["max_seats"] and j - 1 >= num_players else static[j]
            assert types[j] == expect, (game.title, j)
        assert not mask[ne:].any()

        def is_corp(x):
            return x >= 0 and types[x] in CORP_TYPES

        for p in arrays["hex_ptr"][k][:nh].ravel():
            assert p in (-1, -2) or is_corp(p)
        for p in arrays["hex_res"][k][:nh].ravel():
            assert p == -1 or is_corp(p)
        for p in arrays["ent_ptr"][i].ravel():
            assert p == -1 or (0 <= p < ne and mask[p])
        for p in arrays["ent_hex"][i].ravel():
            assert p == -1 or 0 <= p < nh
        rm = arrays["rel_mask"][i]
        rel = arrays["rel"][i][rm]
        assert ((rel[:, 2] > 0) & (rel[:, 2] < len(SCHEMA["relation_kinds"]))).all()
        assert mask[rel[:, 0]].all() and mask[rel[:, 1]].all()
        assert (arrays["rel"][i][~rm] == 0).all()
        meta = arrays["meta"][i]
        assert SCHEMA["titles"][meta[0] - 1] == game.title
        assert meta[4] == num_players and 0 <= meta[5] < num_players
        if not game.finished:
            assert meta[1] > 0 and meta[2] > 0


def test_schema_widths():
    s = SCHEMA
    assert s["f_hex"] == len(s["hex_features"])
    assert s["f_ent"] == max(len(v) for v in s["entity_features"].values())
    assert s["max_seats"] == 6
    assert len(s["meta_columns"]) == s["meta_len"]
    for title in ("1830", "1867", "1822", "1822PNW", "1822MX", "1822CA"):
        info = spec_info(title)
        assert len(info["hex_coords"]) == info["num_hexes"] == info["hex_axial"].shape[0]
        assert len(info["entity_names"]) == info["num_entities"] == len(info["entity_types"])
        assert info["tile_feat"].shape == (info["num_tiles"], s["f_tile"])
        assert info["hex_neighbors"].shape == (info["num_hexes"], 6)
    with pytest.raises(ValueError):
        spec_info("9999")


@pytest.mark.parametrize("title,players", [("1830", n) for n in (2, 3, 4, 5, 6)] + [("1867", n) for n in (3, 4, 5, 6)])
def test_walk_positions_encode(positions, title, players):
    games = positions[(title, players)]
    assert len(games) > 50
    for dedupe in (False, True):
        arrays = encode_batch(games, dedupe_maps=dedupe)
        check_batch(arrays, games)
    # deterministic, and de-duplication only re-indexes the hex blocks
    plain = encode_batch(games)
    again = encode_batch(games)
    deduped = encode_batch(games, dedupe_maps=True)
    for key, value in plain.items():
        if value is not None:
            assert np.array_equal(value, again[key]), key
    for key in ("hex_feat", "hex_ptr", "hex_res", "hex_slot_city", "hex_ids", "hex_mask"):
        assert np.array_equal(deduped[key][deduped["map_index"]], plain[key]), key
    for key in ("ent_feat", "ent_type", "ent_ptr", "rel", "rel_val", "meta"):
        assert np.array_equal(deduped[key], plain[key]), key
    assert deduped["hex_feat"].shape[0] < len(games), "positions of one map share a block"
    # one position at a time gives the same rows
    single = encode_batch(games[len(games) // 2 : len(games) // 2 + 1])
    j = len(games) // 2
    assert np.array_equal(single["ent_feat"][0], plain["ent_feat"][j])
    assert np.array_equal(single["hex_feat"][0], plain["hex_feat"][j])


def test_mixed_batch_pads_to_the_largest(positions):
    games = positions[("1830", 4)][-3:] + positions[("1867", 6)][-3:] + positions[("1830", 2)][:2]
    arrays = encode_batch(games, dedupe_maps=True)
    check_batch(arrays, games)
    assert arrays["hex_feat"].shape[1] == max(spec_info(t)["num_hexes"] for t in ("1830", "1867"))
    assert arrays["ent_feat"].shape[1] == spec_info("1867")["num_entities"]


# Encoder version 1's widths: version 2's appended columns are the 1822 family's state (docs/rl_encoder.md).
V1_WIDTHS = {"hex": 106, "global": 29, "player": 16, "major": 53, "minor": 53, "national": 53, "private": 28,
             "concession": 28, "minor_proxy": 28, "train_type": 17, "phase": 22}  # fmt: skip


def test_v2_columns_are_zero_for_1830_and_1867(positions):
    """Version 2 appends the 1822 family's state; for 1830 and 1867 every appended column is 0 (their
    version-1 columns are pinned bit for bit by the cargo test ``v1_columns_of_1830_and_1867_unchanged``)."""
    assert SCHEMA["version"] == 2
    games = [g for key in sorted(positions) for g in positions[key][::7]]
    arrays = encode_batch(games)
    assert not arrays["hex_feat"][..., V1_WIDTHS["hex"] :].any()
    for t, name in enumerate(SCHEMA["entity_types"]):
        if t == 0:
            continue
        sel = arrays["ent_type"] == t
        assert not arrays["ent_feat"][sel][:, V1_WIDTHS[name] :].any(), name


def walk_1822_family(title: str, num_players: int, seed: int, steps: int = 300, every: int = 10) -> list:
    from scripts.g1822_fuzz import pick as pick_1822

    rng = random.Random(seed * 1_000_003 + num_players)
    ids = {1000 + i + 1: f"Player {i + 1}" for i in range(num_players)}
    game = engine_rs.BaseGame.new_titled(title, ids, None, seed)
    out = []
    for step in range(steps):
        if game.finished:
            break
        moves = json.loads(game.g1822_legal_moves_json(None))
        if not moves:
            break
        if step % every == 0:
            out.append(game.pickle_clone())
        game.process_action(dict(pick_1822(moves, rng)))
    return out


def test_g1822_family_encodes(positions):
    """The 1822 family's positions (version 2's family state) batch with 1830's: shapes, masks,
    pointers, relations, determinism; the family's bid boxes and bidding tokens are on the tokens."""
    family = [g for k, t in enumerate(("1822", "1822PNW", "1822MX", "1822CA")) for g in walk_1822_family(t, 4, k + 1)]
    games = family + positions[("1830", 4)][:3]
    for dedupe in (False, True):
        check_batch(encode_batch(games, dedupe_maps=dedupe), games)
    arrays = encode_batch(family)
    again = encode_batch(family)
    for key, value in arrays.items():
        if value is not None:
            assert np.array_equal(value, again[key]), key
    comp = SCHEMA["entity_features"]["minor_proxy"]
    proxy = arrays["ent_type"] == SCHEMA["entity_types"].index("minor_proxy")
    assert (arrays["ent_feat"][proxy][:, comp.index("bidbox_slot")] > 0).any()
    seat = arrays["ent_type"] == SCHEMA["entity_types"].index("player")
    left = arrays["ent_feat"][seat][:, SCHEMA["entity_features"]["player"].index("bid_tokens_left")]
    assert ((left >= 0) & (left <= 1)).all() and (left > 0).any()


def test_descriptors_address_tokens(positions):
    """A legal 1830 index's descriptor (milestone A1) names the hex / entity tokens
    of the same position: its hex is a real hex token, a PlaceToken's city exists on
    that hex, and its entities are tokens of the descriptor's kind."""
    d = action_descriptors("1830")
    info = spec_info("1830")
    base = info["descriptor_entity_base"]
    kinds = list(d.ent_kinds)
    names = info["entity_names"]
    assert info["hex_coords"] == list(d.vocab["hex"])
    assert info["tile_ids"] == list(d.vocab["tile"])
    for kind, prefix in (("corporation", "corp:"), ("company", "company:"), ("train", "train:")):
        for i, sym in enumerate(d.vocab[kind]):
            assert names[base[kind] + i] == prefix + sym
    checked = 0
    for game in positions[("1830", 4)] + positions[("1830", 6)]:
        legal = [int(i) for i in game.factored_legal_indices()]
        if not legal:
            continue
        arrays = encode_batch([game])
        rows = d.gather(legal)
        for row in rows:
            hex_ = row[d.col("hex")]
            if hex_ >= 0:
                assert arrays["hex_mask"][0, hex_]
                city = row[d.col("city")]
                if city >= 0:
                    assert city in set(arrays["hex_slot_city"][0, hex_].tolist())
            for kind_col, ent_col in (("ent0_kind", "ent0"), ("ent1_kind", "ent1")):
                kind, ent = row[d.col(kind_col)], row[d.col(ent_col)]
                if ent < 0 or kinds[kind] == "player":
                    continue
                token = base[kinds[kind]] + ent
                prefix = {"corporation": "corp:", "company": "company:", "train": "train:"}[kinds[kind]]
                assert names[token].startswith(prefix)
                assert arrays["ent_mask"][0, token]
            checked += 1
    assert checked > 200


def test_seat_zero_is_the_player_to_move(positions):
    for game in positions[("1830", 4)][10:40] + positions[("1867", 5)][10:40]:
        arrays = encode_batch([game])
        rotation = arrays["meta"][0, 5]
        ids = [p.id for p in game.players]  # seating order
        cash = {p.id: p.cash for p in game.players}
        player_features = SCHEMA["entity_features"]["player"]
        col = player_features.index("cash")
        sc = game.starting_cash if hasattr(game, "starting_cash") else None
        for seat in range(len(ids)):
            pid = ids[(seat + rotation) % len(ids)]
            value = arrays["ent_feat"][0, 1 + seat, col]
            if sc:
                assert value == pytest.approx(cash[pid] / sc)
            else:
                # relative check: seat cash ratios follow the rotated players
                ref = arrays["ent_feat"][0, 1, col]
                if ref:
                    assert value / ref == pytest.approx(cash[pid] / cash[ids[rotation]])


def test_stub_forward_backward(positions):
    torch.manual_seed(0)
    games = positions[("1830", 4)][::40] + positions[("1867", 5)][::40] + positions[("1830", 6)][-2:]
    inputs = to_torch(encode_batch(games, dedupe_maps=True))
    model = TokenTrunkStub(TokenStubConfig(d_model=32, num_heads=4))
    out = model(inputs)
    b = len(games)
    assert out["trunk"].shape == (b, 32)
    assert out["value_logits"].shape == (b, SCHEMA["max_seats"])
    assert torch.isfinite(out["trunk"]).all() and torch.isfinite(out["entities"]).all()
    present = inputs["ent_mask"][:, 1 : 1 + SCHEMA["max_seats"]]
    assert (out["value_logits"][~present] == -1e4).all()
    # a winner-CE loss trains through every part of the stub
    target = torch.zeros(b, dtype=torch.long)
    loss = torch.nn.functional.cross_entropy(out["value_logits"], target)
    loss.backward()
    for name, p in model.named_parameters():
        if name.startswith(("rel_bias", "dist_bias", "token_bias", "film")):
            continue  # zero-initialized biases: gradients only after a step
        assert p.grad is not None, name
    assert torch.isfinite(torch.stack([p.grad.abs().sum() for p in model.parameters() if p.grad is not None])).all()
    # padding is inert: the same position alone or in the mixed batch gives the same outputs
    model.eval()
    with torch.no_grad():
        alone = model(to_torch(encode_batch(games[:1])))
        batched = model(to_torch(encode_batch(games)))
    assert torch.allclose(alone["value_logits"][0], batched["value_logits"][0], atol=1e-4)


@pytest.mark.benchmark
def test_throughput_vs_1830_encoder(positions):
    games = [g for n in (2, 3, 4, 5, 6) for g in positions[("1830", n)]]
    by_count: dict = {}
    for g in games:
        by_count.setdefault(len(g.players), []).append(g)

    def best(fn, reps=5):
        times = []
        for _ in range(reps):
            t = time.perf_counter()
            fn()
            times.append(time.perf_counter() - t)
        return min(times)

    token = best(lambda: [encode_batch(gs) for gs in by_count.values()])
    token_dedupe = best(lambda: [encode_batch(gs, dedupe_maps=True) for gs in by_count.values()])
    legacy = best(lambda: [engine_rs.encode_games(gs) for gs in by_count.values()])
    legacy_dedupe = best(lambda: [engine_rs.encode_games(gs, dedupe_maps=True) for gs in by_count.values()])
    n = len(games)
    print(
        f"\n1830, {n} positions, batch call incl. numpy: token encoder {n / token:,.0f}/s "
        f"({n / token_dedupe:,.0f}/s maps deduped), 1830 encode_games {n / legacy:,.0f}/s "
        f"({n / legacy_dedupe:,.0f}/s deduped)"
    )
    games_67 = [g for n in (3, 4, 5, 6) for g in positions[("1867", n)]]
    t67 = best(lambda: encode_batch(games_67))
    print(f"1867, {len(games_67)} positions: token encoder {len(games_67) / t67:,.0f}/s")
