"""1867 on the title-generic pipeline (RL bridge milestone A3): the human-game importer
(``generic_import_1867``), the dataset and loader on 1867 records, legal prices (every sampled price on the
move's step grid, by every sampling path), and 1867 games played by the PG worker and the eval.

Importer, on the four Ruby fixtures (``tests/fixtures/1867``, 4 players): every kept label is the one index the
human action matches among the rules' moves (A2's index completeness, ``scripts/g1867_completeness.py
--index``, counts equal game by game), the legal list is the title policy's (``model_generic.legal_actions``),
decode gives the human move back, price targets lie on the step grid, auto-actions are not decisions,
outcomes are in seating order, forced positions are kept, out-of-scope games are dropped and counted, a
replay rejection keeps the game up to the rejection, and a match that is not exactly one index is skipped.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch

engine_rs = pytest.importorskip("engine_rs")

from rl18xx.agent.alphazero import generic_data as gd  # noqa: E402
from rl18xx.agent.alphazero import generic_import_1867 as gi67  # noqa: E402
from rl18xx.agent.alphazero import model_generic as mg  # noqa: E402
from rl18xx.agent.alphazero.titles import new_game  # noqa: E402

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "1867"
# 4 players each; nationalization_cash carries recorded auto-actions, the hotseat 19792 is short.
GAMES = ("21268.json", "nationalization_cash.json", "hs_ahjzadkh_19792.json")
SMALL = dict(
    d_model=32, num_heads=4, hex_layers=1, entity_layers=1, fusion_layers=1, d_trunk=64, trunk_blocks=1,
    d_head=32, d_edge=8, d_cond=16, d_id=8,
)  # fmt: skip


def load_fixture(name: str) -> dict:
    game = json.loads((FIXTURES / name).read_text())
    game["id"] = Path(name).stem
    return game


@pytest.fixture(scope="module")
def importer():
    return gd.importer_for("1867")


@pytest.fixture(scope="module")
def records(importer):
    out = {}
    for name in GAMES:
        record, stats = gd.convert_game(load_fixture(name), importer)
        assert record is not None, stats
        out[name] = (record, stats)
    return out


# ---------------------------------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------------------------------


def test_registered():
    assert isinstance(gd.importer_for("1867"), gi67.Importer1867)
    with pytest.raises(gd.UnsupportedTitle):
        gd.importer_for("1822")


@pytest.mark.parametrize("name", GAMES)
def test_counts_equal_index_completeness(name, records):
    """Kept labels and skips are A2's index-level completeness, decision for decision."""
    from scripts.g1867_completeness import audit_game_index

    record, stats = records[name]
    audit = audit_game_index(str(FIXTURES / name), apply_check=False)
    assert audit["error"] is None
    kinds = {}
    for (_, kind), count in audit["counts"].items():
        kinds[kind] = kinds.get(kind, 0) + count
    assert gd.record_decisions(record) == stats["positions"] == kinds.get("exact", 0)
    assert sum(stats.get("policy", {}).values()) == kinds.get("policy", 0)
    assert sum(stats.get("ambiguous", {}).values()) == kinds.get("ambiguous", 0) == 0
    missing = sum(stats.get("unmatched", {}).values())
    replay_only = sum(sum(v.values()) for k, v in stats.items() if k.startswith("replay_only:"))
    assert missing == kinds.get("missing", 0)
    assert replay_only == sum(v for k, v in kinds.items() if k not in ("exact", "policy", "missing", "ambiguous"))
    skipped = missing + replay_only + sum(stats.get("policy", {}).values())
    assert stats["positions"] + skipped + sum(stats.get("not_decision", {}).values()) == stats["actions"]


@pytest.mark.parametrize("name", GAMES)
def test_labels_are_the_one_matching_index(name, records):
    """At every kept decision: the stored legal list is the title policy's, the human action matches exactly
    one of the rules' indexed moves, that index is the label, and decoding it gives the human move back."""
    record, _ = records[name]
    actions = json.loads(record["actions"])
    offsets = record["legal_offsets"]
    picks = list(range(0, gd.record_decisions(record), 3))
    for k, game in gd.replay(record, picks):
        lo, hi = offsets[k], offsets[k + 1]
        la = mg.legal_actions(game)
        assert la.indices.tolist() == record["legal_indices"][lo:hi].tolist()
        assert np.array_equal(la.price_range, record["legal_price_range"][lo:hi])
        human = actions[int(record["action_pos"][k])]
        hits = gi67.matching_indexed(game, human, json.loads(game.rl_legal_moves_json("rules")))
        assert [h["index"] for h in hits] == [int(record["chosen_index"][k])]
        assert la.indices[int(record["target"][k])] == record["chosen_index"][k]
        price = int(record["price"][k]) if record["price"][k] >= 0 else None
        if human["type"] not in gi67.TYPE_ONLY:
            wire = json.loads(game.rl_decode_index(int(record["chosen_index"][k]), price, "rules"))
            assert gi67.action_key(game, wire) == gi67.action_key(game, human)


def test_price_targets_on_the_step_grid(records):
    seen = 0
    for record, stats in records.values():
        offsets = record["legal_offsets"]
        for k in range(gd.record_decisions(record)):
            row = offsets[k] + record["target"][k]
            lo, hi, step = (int(x) for x in record["legal_price_range"][row])
            price = int(record["price"][k])
            if price >= 0:
                assert lo <= price <= hi and (price - lo) % step == 0
                seen += hi > lo
        assert stats["price_outside_range"] == 0
    assert seen == sum(s["price_targets"] for _, s in records.values()) > 50


def test_auto_actions_are_not_decisions(records):
    from tests.title_replay_harness import ruby_filtered_actions

    _, stats = records["nationalization_cash.json"]
    game = load_fixture("nationalization_cash.json")
    # the auto-actions Ruby processes (of actions not undone; messages and programs dropped)
    autos = sum(bool(a.get("_auto")) for a in ruby_filtered_actions(game["actions"], mark_auto=True))
    assert stats["not_decision"]["auto"] == autos > 0


def test_outcome_targets_in_seating_order(records):
    from rl18xx.agent.alphazero.policy_gradient import default_placement_points, placement_points

    for record, _ in records.values():
        players = {pid: n for pid, n in record["players"]}
        assert list(players) == [1, 2, 3, 4] and players[1] == "Player 1"
        game = gd.new_engine_game("1867", players, (), record["seed"])
        for action in json.loads(record["actions"]):
            game.process_action(action)
        result = game.result()
        worth = [result[pid] for pid in players]
        assert record["worth"].tolist() == pytest.approx(worth)
        assert record["win_share"].sum() == pytest.approx(1.0)
        assert record["win_share"][int(np.argmax(worth))] > 0
        assert record["placement"].tolist() == pytest.approx(placement_points(worth).tolist())
        assert max(record["placement"]) == default_placement_points(4)[0]


def test_forced_positions_kept(records):
    record, stats = records["21268.json"]
    lengths = np.diff(record["legal_offsets"])
    assert stats["forced"] == int((lengths == 1).sum()) > 0
    fams = mg.vocab()["families"]
    assert "RunRoutes" in {fams[f] for f in record["family"].tolist()}


def test_out_of_scope_games_are_dropped_and_counted(importer):
    base = load_fixture("hs_ahjzadkh_19792.json")
    cases = {
        "players_2": dict(base, players=base["players"][:2]),
        "grid_market": dict(base, settings=dict(base.get("settings") or {}, optional_rules=["grid_market"])),
        "optional_rules": dict(base, settings=dict(base.get("settings") or {}, optional_rules=["other"])),
        "status_active": dict(base, status="active"),
        "title": dict(base, title="1830"),
    }
    for reason, game in cases.items():
        record, stats = gd.convert_game(game, importer)
        assert record is None and stats["dropped"] == {reason: 1}, reason


def test_a_rejected_action_keeps_the_game_up_to_it(importer, records):
    full, _ = records["hs_ahjzadkh_19792.json"]
    game = load_fixture("hs_ahjzadkh_19792.json")
    cut = 150
    bad = {"type": "dividend", "entity": "NOPE", "entity_type": "corporation", "kind": "payout", "id": 10**6}
    game["actions"] = game["actions"][:cut] + [bad] + game["actions"][cut:]
    record, stats = gd.convert_game(game, importer)
    assert record is not None and record["stopped"].startswith("dividend") and not record["finished"]
    assert stats["stopped"] == 1 and record["num_actions"] < full["num_actions"]
    n = gd.record_decisions(record)
    assert n > 0
    assert record["chosen_index"].tolist() == full["chosen_index"][:n].tolist()
    assert record["action_pos"].tolist() == full["action_pos"][:n].tolist()


class _Flaky(gi67.Importer1867):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def match(self, state, action, legal_indices):
        self.calls += 1
        if self.calls % 5 == 0:
            return None, "ambiguous"
        return super().match(state, action, legal_indices)


def test_not_exactly_one_match_is_skipped(records):
    record, _ = records["hs_ahjzadkh_19792.json"]
    flaky, stats = gd.convert_game(load_fixture("hs_ahjzadkh_19792.json"), _Flaky())
    ambiguous = sum(stats["ambiguous"].values())
    assert ambiguous > 0
    # the flaky "ambiguous" lands on every 5th match call, kept or skipped in the full record
    dropped = gd.record_decisions(record) - gd.record_decisions(flaky)
    assert 0 < dropped <= ambiguous
    full = dict(zip(record["action_pos"].tolist(), record["chosen_index"].tolist()))
    assert all(full[p] == i for p, i in zip(flaky["action_pos"].tolist(), flaky["chosen_index"].tolist()))
    assert flaky["worth"].tolist() == record["worth"].tolist()


# ---------------------------------------------------------------------------------------------------
# Dataset, loader, collate
# ---------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dataset_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("generic_1867")
    games = root / "games"
    games.mkdir()
    for name in GAMES[1:]:
        (games / name).write_text(json.dumps(load_fixture(name)))
    out = root / "ds"
    # all games: hs_ahjzadkh_19792's record ends before the game does (the finished-only default would drop it)
    gd.convert_dataset(
        [str(games)], str(out), title="1867", workers=1, validation_percentage=0.5, finished_only=False
    )
    return out


def test_finished_only_drops_and_lists_unfinished_games(tmp_path, importer):
    """1867's default (``TitleSpec.supervised_finished_only``) keeps only games whose log reaches the game's end:
    a record that ends early and one the replay rejects part-way are dropped, counted by reason and player count
    and listed in ``dropped.jsonl``; ``finished_only=False`` (1830's default) keeps them."""
    from rl18xx.agent.alphazero.titles import title_spec

    assert title_spec("1867").supervised_finished_only and not title_spec("1830").supervised_finished_only
    games = tmp_path / "games"
    games.mkdir()
    for name in ("nationalization_cash.json", "hs_ahjzadkh_19792.json"):  # finished; record ends early
        (games / name).write_text(json.dumps(load_fixture(name)))
    rejected = load_fixture("21268.json")
    rejected["id"] = "rejected"
    bad = {"type": "dividend", "entity": "NOPE", "entity_type": "corporation", "kind": "payout", "id": 10**6}
    rejected["actions"] = rejected["actions"][:300] + [bad] + rejected["actions"][300:]
    (games / "rejected.json").write_text(json.dumps(rejected))
    meta = gd.convert_dataset([str(games)], str(tmp_path / "ds"), title="1867", workers=1)
    assert meta["finished_only"] is True
    ds = gd.GenericDataset(tmp_path / "ds")
    assert ds.game_ids() == ["nationalization_cash"]
    assert all(ds.record(g)["finished"] for g in ds.game_ids())
    assert meta["stats"]["dropped"] == {"unfinished_record_ends": 1, "unfinished_replay_rejected": 1}
    assert meta["counts"]["dropped_by_reason_and_players"] == {
        "unfinished_record_ends": {"4": 1}, "unfinished_replay_rejected": {"4": 1}
    }
    lines = [json.loads(x) for x in (tmp_path / "ds" / "dropped.jsonl").read_text().splitlines()]
    assert {(e["id"], e["reason"], e["players"]) for e in lines} == {
        ("hs_ahjzadkh_19792", "unfinished_record_ends", 4), ("rejected", "unfinished_replay_rejected", 4)
    }
    keep = gd.convert_dataset([str(games)], str(tmp_path / "all"), title="1867", workers=1, finished_only=False)
    assert keep["finished_only"] is False and sum(keep["counts"]["games"].values()) == 3
    # the importer-independent check at the end (an importer that cannot tell after load)
    record, stats = gd.convert_game(load_fixture("hs_ahjzadkh_19792.json"), _Unknown(), finished_only=True)
    assert record is None and stats["dropped"] == {"unfinished_record_ends": 1}


class _Unknown(gi67.Importer1867):
    def unfinished_reason(self, game):
        return "unknown"


def test_dataset_round_trip_and_loader(dataset_dir, records):
    ds = gd.GenericDataset(dataset_dir)
    assert ds.meta["titles"] == ["1867"]
    assert ds.meta["layouts"]["1867"]["fingerprint"] == mg.action_descriptors("1867").fingerprint
    for name in GAMES[1:]:
        rec = ds.record(Path(name).stem)
        ref, _ = records[name]
        for key in ("chosen_index", "target", "legal_indices", "legal_price_range", "price", "meta", "worth"):
            assert np.array_equal(rec[key], ref[key]), key
    seen = 0
    for split in gd.SPLITS:
        if not ds.game_ids(split):
            continue
        for batch in gd.PositionBatches(dataset_dir, split, 64, shuffle=True, buffer_positions=300, seed=2,
                                        drop_last=False):
            seen += len(batch["target_index"])
            assert batch["inputs"]["price_grid"].all()  # 1867 rows: cells in step units
            assert set(batch["num_players"].tolist()) == {4}
    assert seen == ds.num_positions()


def test_collate_equals_build_inputs_on_1867(records):
    record, _ = records["21268.json"]
    picks = sorted(random.Random(4).sample(range(gd.record_decisions(record)), 64))
    items, games = [], []
    for k, game in gd.replay(record, picks):
        items.append(gd.position_item(record, k, game))
        games.append(game.pickle_clone())
    batch = gd.collate(items)
    ref = mg.build_inputs(games, [mg.legal_actions(g) for g in games])
    for key, value in ref.items():
        if value is None:
            assert batch["inputs"][key] is None
        else:
            assert np.array_equal(value, batch["inputs"][key]), key
    assert ref["price_grid"].all() and len(ref["price_grid"]) > 0
    # price targets: valid exactly where the chosen row has an open price, in the grid's cells
    for j, row in enumerate(batch["inputs"]["price_rows"]):
        if batch["price_valid"][j]:
            b = int(batch["inputs"]["legal_pos"][row])
            assert row == batch["inputs"]["legal_offsets"][b] + batch["target_index"][b]


# ---------------------------------------------------------------------------------------------------
# Legal prices: every sampling path stays on the move's step grid
# ---------------------------------------------------------------------------------------------------


def walk_1867(num_players: int, seed: int, steps: int = 2500, per_block: int = 15) -> dict:
    """Random-walk positions with an open-price legal row, up to ``per_block`` per priced block (the index
    API; random prices on the grid)."""
    rng = random.Random(seed)
    game = new_game("1867", num_players, seed=seed)
    out: dict = {}
    for _ in range(steps):
        if game.finished:
            break
        moves = json.loads(game.rl_legal_moves_json())
        blocks = {m["block"] for m in moves if m["range"] and m["range"][1] > m["range"][0]}
        fresh = [b for b in blocks if len(out.setdefault(b, [])) < per_block]
        if fresh:
            out[fresh[0]].append(game.pickle_clone())
        m = rng.choice(moves)
        price = None
        if m["range"]:
            lo, hi, step = m["range"]
            price = lo + step * rng.randrange((min(hi, 2 * lo + step) - lo) // step + 1)
        game.rl_apply_index(int(m["index"]), price)
    return out


@pytest.fixture(scope="module")
def priced_1867():
    """Positions with open prices of every priced block (Bid.company, Bid.corporation, BuyTrain.corp,
    BuyCompany), from random walks at 3-6 players."""
    by_block: dict = {}
    for n, seed in ((3, 1), (4, 2), (5, 3), (6, 4), (4, 5), (3, 6)):
        for block, games in walk_1867(n, seed).items():
            by_block.setdefault(block, []).extend(games)
        if len(by_block) == 4 and min(len(v) for v in by_block.values()) >= 10:
            break
    assert set(by_block) == {"Bid.company", "Bid.corporation", "BuyTrain.corp", "BuyCompany"}, sorted(by_block)
    return [g for games in by_block.values() for g in games[:30]]


@pytest.fixture(scope="module")
def random_model():
    torch.manual_seed(1)
    m = mg.GenericPolicyValueModel(mg.GenericModelConfig(**SMALL)).eval()
    with torch.no_grad():  # zero-initialized outputs give uniform cells: make the heads non-trivial
        for p in m.parameters():
            if p.abs().sum() == 0:
                p.normal_(0, 0.5)
    return m


def _is_legal_price(game, index: int, price: int) -> bool:
    lo, hi, step = game.rl_price_range(index)
    if not (lo <= price <= hi and (price - lo) % step == 0):
        return False
    game.pickle_clone().rl_apply_index(index, price)  # raises on an illegal price
    return True


def test_sampled_1867_prices_are_legal_and_apply(priced_1867, random_model, tmp_path):
    from rl18xx.agent.alphazero import generic_inference as gi
    from rl18xx.agent.alphazero import generic_policy as gpol

    path = tmp_path / "m.pth"
    mg.save_checkpoint(random_model, path, titles=["1867"])
    policy = gpol.GenericPolicy(str(path))
    server = gi.LocalGenericServer(random_model, seed=5)
    rng = np.random.default_rng(0)
    kinds = set()
    checked = 0
    games = priced_1867
    outputs = policy.evaluate(games)
    legal = [o.legal for o in outputs]
    arrays = mg.build_inputs(games, legal)
    for draw in range(4):
        reply = server.run(arrays, price_eps=0.3 if draw % 2 else 0.0)
        offsets = arrays["legal_offsets"]
        for j, row in enumerate(arrays["price_rows"]):
            b = int(arrays["legal_pos"][row])
            k = int(row - offsets[b])
            la = legal[b]
            if not la.price_range[k, 1] > la.price_range[k, 0]:
                continue
            index = int(la.indices[k])
            kinds.add(int(la.rows[k, mg.C_PRICE_KIND]))
            # the PG worker's / eval's path: the server's cell, a price inside it on the layout's grid
            price = int(la.price_cells(k).sample_price(int(reply.price_cell[j]), rng))
            assert _is_legal_price(games[b], index, price), (index, price, la.price_range[k])
            # the policy-only player's path (eval_policy_only.py generic players)
            price = gpol.sample_price(games[b], outputs[b], k, rng, eps=0.2)
            assert _is_legal_price(games[b], index, price), (index, price, la.price_range[k])
            checked += 2
    assert checked > 400
    v = mg.vocab()["price_kinds"]
    assert {v[k] for k in kinds} == {"bid", "bid_range", "company_purchase", "train_transfer"}


def test_cells_without_the_grid_would_offer_illegal_1867_prices(priced_1867):
    """The bug the grid fixes: the 1830 cells of a $5-step range hold every integer."""
    off_grid = 0
    for game in priced_1867:
        la = mg.legal_actions(game)
        for k in np.flatnonzero(la.price_range[:, 1] > la.price_range[:, 0]):
            lo, hi, step = (int(x) for x in la.price_range[k])
            if step == 1:
                continue
            old = mg.price_cells_for(int(la.rows[k, mg.C_PRICE_KIND]), lo, hi, step)  # grid=False
            members = [old.member(c, u) for c in np.flatnonzero(old.nonempty) for u in range(int(old.counts[c]))]
            off_grid += sum((p - lo) % step != 0 for p in members)
            grid = la.price_cells(k)
            members = [grid.member(c, u) for c in np.flatnonzero(grid.nonempty) for u in range(int(grid.counts[c]))]
            assert sorted(members) == list(range(lo, hi + 1, step))
    assert off_grid > 0


def test_1830_price_cells_are_unchanged():
    """1830's legal lists keep the legacy cells (price space, the $5 "Bid" ladder): same cells, same draws."""
    from tests.agent.alphazero.test_generic_pg import walk_1830

    compared = 0
    for game in walk_1830(4, 3, steps=300, every=3):
        la = mg.legal_actions(game)
        assert not la.price_grid
        for k in np.flatnonzero(la.rows[:, mg.C_PRICE_KIND] >= 0):
            lo, hi, step = (int(x) for x in la.price_range[k])
            if hi <= lo:
                continue
            old = mg.price_cells_for(int(la.rows[k, mg.C_PRICE_KIND]), lo, hi, step)
            new = la.price_cells(k)
            assert np.array_equal(old.counts, new.counts)
            for cell in np.flatnonzero(old.nonempty):
                a, b = np.random.default_rng(int(cell)), np.random.default_rng(int(cell))
                assert old.sample_price(int(cell), a) == new.sample_price(int(cell), b)
            compared += 1
    assert compared > 5


# ---------------------------------------------------------------------------------------------------
# 1867 games: PG worker and eval
# ---------------------------------------------------------------------------------------------------


@pytest.fixture
def clients(random_model, monkeypatch):
    from rl18xx.agent.alphazero import generic_inference as gi
    from rl18xx.agent.alphazero import generic_pg as gpg

    server = gi.LocalGenericServer(random_model, seed=3)
    local = {name: gi.LocalGenericClient(server) for name in ("learner", "sl", "pool")}
    monkeypatch.setattr(gpg, "_CLIENTS", local)
    return local


def test_1867_pg_games_and_the_trainers_probabilities(clients, random_model, tmp_path):
    """1867 games from the opening and from random openings through the PG worker; the trainer recomputes
    every kept move's sampling probability (price cells included: the rows carry the grid) exactly."""
    from rl18xx.agent.alphazero import generic_pg as gpg

    empty = tmp_path / "starts.jsonl"
    empty.write_text("")
    settings = {
        "title": "1867", "num_players": 4, "positions_per_seat": 40, "learner_seats": 2,
        "sl_opponent_fraction": 0.5, "pool_ready": False, "pool_label": None, "learner_temperature": 0.5,
        "opponent_temperature": 1.0, "opponent_price_eps": 0.05, "max_decisions": 150, "start_positions": str(empty),
        "random_start_fraction": 1.0, "auction_fraction": 0.5, "reward": "placement", "league": None,
    }  # fmt: skip
    result = gpg.play_generic_pg_games(3, settings)
    rows = [r for g in result["games"] for r in g["rows"]]
    assert rows and all(r.price_grid for r in rows)
    assert sum(r.price_cell >= 0 for r in rows) > 3
    batch = gpg.batch_to_device(gpg.collate_rows(rows), "cpu")
    with torch.no_grad():
        out = random_model(batch["inputs"])
        losses = gpg.generic_pg_losses(out, out, batch, clip=0.2, kl_coef=0.02, entropy_coef=0.0, temperature=0.5)
    assert float(losses["log_ratio_abs_mean"]) < 1e-4


def test_1867_title_play_applies_like_rl_apply_index():
    """``TitlePlay.apply`` on 1867 processes the row's action dict (``LegalActions.wires``) at the price: the
    same state and logged action as ``rl_apply_index``, without re-enumerating; an illegal price raises."""
    from rl18xx.agent.alphazero.generic_play import title_play

    play = title_play("1867")
    rng = random.Random(0)
    checked = 0
    for n, seed in ((3, 4), (5, 6)):
        raw = new_game("1867", n, seed=seed)
        for _ in range(600):
            if raw.finished:
                break
            legal = play.legal(raw)
            assert legal.wires is not None and len(legal.wires) == len(legal)
            k = rng.randrange(len(legal))
            price = play.fixed_price(raw, legal, k)
            if play.open_price(legal, k):
                lo, hi, step = (int(x) for x in legal.price_range[k])
                price = lo + step * rng.randrange((hi - lo) // step + 1)
                if step > 1 and hi - lo > step:
                    with pytest.raises(ValueError):
                        play.apply(raw.pickle_clone(), legal, k, lo + 1)
            by_index = raw.pickle_clone()
            by_index.rl_apply_index(int(legal.indices[k]), price)
            play.apply(raw, legal, k, price)
            assert raw.raw_actions[-1] == by_index.raw_actions[-1]
            assert raw.__getstate__() == by_index.__getstate__()
            checked += 1
    assert checked > 500


def test_start_files_skip_games_the_replay_rejects():
    """A start file takes starts only from games that replay to the end of their record: a record the engine
    rejects part-way gives none, not even its Stock Rounds before the rejection."""
    from rl18xx.agent.alphazero import generic_play as gp

    game = load_fixture("21268.json")
    replayed, reason = gp.replay_human_game("1867", game)
    assert reason is None
    start, midgame, why = gp.start_records("1867", replayed)
    assert start is not None and midgame is not None and why is None
    bad = {"type": "dividend", "entity": "NOPE", "entity_type": "corporation", "kind": "payout", "id": 10**6}
    game["actions"] = game["actions"][:600] + [bad] + game["actions"][600:]
    replayed, reason = gp.replay_human_game("1867", game)
    assert replayed is None and reason.startswith("replay_failed")


def test_1867_fixed_price_from_the_legal_list_is_rl_price_range():
    """``TitlePlay.fixed_price`` on 1867 reads the legal list's ranges (no second generator run): for every row
    of every position of random walks at 3-6 players it equals ``rl_price_range``'s low end (None for a move
    without a price), and every move the generator gives a range has a price-kind row (so the list holds it)."""
    from rl18xx.agent.alphazero.generic_play import title_play
    from rl18xx.agent.alphazero.model_generic import C_PRICE_KIND

    play = title_play("1867")
    rng = random.Random(1)
    checked = priced = 0
    for n, seed in ((3, 11), (4, 12), (5, 13), (6, 14)):
        raw = new_game("1867", n, seed=seed)
        for _ in range(250):
            if raw.finished:
                break
            legal = play.legal(raw)
            ranged = {m["index"] for m in json.loads(raw.rl_legal_moves_json()) if m["range"] is not None}
            for k in range(len(legal)):
                index = int(legal.indices[k])
                r = raw.rl_price_range(index)
                assert play.fixed_price(raw, legal, k) == (None if r is None else int(r[0])), index
                assert (index in ranged) == (legal.rows[k, C_PRICE_KIND] >= 0), index
                checked += 1
                priced += r is not None
            k = rng.randrange(len(legal))
            price = play.fixed_price(raw, legal, k)
            if play.open_price(legal, k):
                lo, hi, step = (int(x) for x in legal.price_range[k])
                price = lo + step * rng.randrange((hi - lo) // step + 1)
            play.apply(raw, legal, k, price)
    assert checked > 2000 and priced > 50
