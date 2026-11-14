"""The 1822 family (1822, 1822PNW, 1822MX, 1822CA) on the title-generic pipeline: the human-game importer
(``generic_import_1822``), records, dataset and loader, start files (the opening is the bid-box Stock Round:
human starts are where it ends), legal prices of every price kind the family uses (``bid``, ``bid_range``,
``train_transfer``, ``amount``, 1822MX's ``Bid.token``) on every sampling path, and family games played by the
PG worker.

Importer, on Ruby fixtures (``tests/fixtures/1822*``, base rules): every kept label is the one index the human
action matches among the rules' moves (A4's index completeness, ``scripts/g1822_completeness.py --index``,
counts equal game by game), the legal list is the title policy's, matching on the policy's list equals matching
on the rules' list, decode gives the human move back, price targets lie on the step grid, auto-actions are not
decisions, outcomes are in seating order (seats 1001+), forced positions are kept, out-of-scope games (player
counts, optional rules, status, title) and unfinished ones are dropped and counted.
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
from rl18xx.agent.alphazero import generic_import_1822 as gi22  # noqa: E402
from rl18xx.agent.alphazero import generic_play as gp  # noqa: E402
from rl18xx.agent.alphazero import model_generic as mg  # noqa: E402

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"
# One base-rules fixture per title at the pipeline's player counts (1822CA's 1.json holds debt payoffs).
GAMES = {
    "1822": "discard_trains_after_sr_export.json",  # 3 players (its first LIMIT actions: test time)
    "1822PNW": "586.json",  # 3 players, short
    "1822MX": "86.json",  # 3 players, short
    "1822CA": "1.json",  # 4 players
}
LIMIT = {"1822": 450}
SMALL = dict(
    d_model=32, num_heads=4, hex_layers=1, entity_layers=1, fusion_layers=1, d_trunk=64, trunk_blocks=1,
    d_head=32, d_edge=8, d_cond=16, d_id=8,
)  # fmt: skip


def load_fixture(title: str, name: str) -> dict:
    game = json.loads((FIXTURES / title / name).read_text())
    game["id"] = Path(name).stem
    return game


def fixture_game(title: str) -> dict:
    """The title's test game (``GAMES``, cut to ``LIMIT`` file actions)."""
    game = load_fixture(title, GAMES[title])
    if title in LIMIT:
        game["actions"] = game["actions"][: LIMIT[title]]
    return game


class _Checking(gi22.Importer1822Family):
    """Also matches every decision on the rules' list and counts disagreements with the importer."""

    def __init__(self, title):
        super().__init__(title)
        self.same = self.differ = 0

    def match(self, state, action, legal_indices):
        out = super().match(state, action, legal_indices)
        g = self.raw(state)
        hits = gi22.matching_indexed(g, action, json.loads(g.rl_legal_moves_json("rules")))
        ref = int(hits[0]["index"]) if len(hits) == 1 else None
        got = out[0] if out[1] == "ok" else (ref if out[1] == "policy" else None)
        if got == ref:
            self.same += 1
        else:
            self.differ += 1
        return out


@pytest.fixture(scope="module")
def records():
    """Each title's test game converted by a checking importer (all games: some end before the game does)."""
    out = {}
    for title in GAMES:
        importer = _Checking(title)
        record, stats = gd.convert_game(fixture_game(title), importer)
        assert record is not None, stats
        out[title] = (record, stats, importer)
    return out


@pytest.fixture(scope="module")
def game_files(tmp_path_factory):
    """The test games as files (``scripts/g1822_completeness.py`` reads files)."""
    root = tmp_path_factory.mktemp("g1822_games")
    paths = {}
    for title in GAMES:
        paths[title] = root / f"{title}_{GAMES[title]}"
        paths[title].write_text(json.dumps(fixture_game(title)))
    return paths


# ---------------------------------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------------------------------


def test_registered():
    from rl18xx.agent.alphazero.titles import title_spec

    for title in gi22.FAMILY:
        importer = gd.importer_for(title)
        assert isinstance(importer, gi22.Importer1822Family) and importer.title == title
        assert title_spec(title).supervised_finished_only
        assert gd.importer_for(title, allow_optional_rules=True).allow_optional_rules


def test_policy_list_matching_equals_rules_matching(records):
    """The importer matches non-policy action types on the title policy's list (one generator run per decision):
    the same label as matching every decision on the rules' list."""
    for title, (record, stats, importer) in records.items():
        assert importer.differ == 0 and importer.same == gd.record_decisions(record) + sum(
            sum(v.values()) for k, v in stats.items() if k in ("policy", "unmatched") or k.startswith("replay_only:")
        ), title


@pytest.mark.parametrize("title", list(GAMES))
def test_counts_equal_index_completeness(title, records, game_files, monkeypatch):
    """Kept labels and skips are A4's index-level completeness, decision for decision."""
    import scripts.g1822_completeness as gc

    monkeypatch.setattr(gc, "TITLE", title)
    record, stats, _ = records[title]
    audit = gc.audit_game_index(str(game_files[title]), apply_check=False)
    kinds: dict = {}
    for (_, kind), count in audit["counts"].items():
        kinds[kind] = kinds.get(kind, 0) + count
    assert gd.record_decisions(record) == stats["positions"] == kinds.get("exact", 0)
    assert sum(stats.get("policy", {}).values()) == kinds.get("policy", 0)
    assert sum(stats.get("ambiguous", {}).values()) == kinds.get("ambiguous", 0) == 0
    missing = sum(stats.get("unmatched", {}).values()) + sum(
        sum(v.values()) for k, v in stats.items() if k.startswith("replay_only:")
    )
    assert missing == kinds.get("missing", 0)
    skipped = missing + sum(stats.get("policy", {}).values())
    assert stats["positions"] + skipped + sum(stats.get("not_decision", {}).values()) == stats["actions"]


@pytest.mark.parametrize("title", list(GAMES))
def test_labels_are_the_one_matching_index(title, records):
    """At kept decisions: the stored legal list is the title policy's, the human action matches exactly one of
    the rules' indexed moves, that index is the label, and decoding it gives the human move back."""
    record, _, _ = records[title]
    actions = json.loads(record["actions"])
    offsets = record["legal_offsets"]
    for k, game in gd.replay(record, range(0, gd.record_decisions(record), 4)):
        lo, hi = offsets[k], offsets[k + 1]
        la = mg.legal_actions(game)
        assert la.indices.tolist() == record["legal_indices"][lo:hi].tolist()
        assert np.array_equal(la.price_range, record["legal_price_range"][lo:hi])
        human = actions[int(record["action_pos"][k])]
        hits = gi22.matching_indexed(game, human, json.loads(game.rl_legal_moves_json("rules")))
        assert [h["index"] for h in hits] == [int(record["chosen_index"][k])]
        assert la.indices[int(record["target"][k])] == record["chosen_index"][k]
        price = int(record["price"][k]) if record["price"][k] >= 0 else None
        if human["type"] not in gi22.TYPE_ONLY:
            wire = json.loads(game.rl_decode_index(int(record["chosen_index"][k]), price, "rules"))
            assert gi22.action_key(game, wire) == gi22.action_key(game, human)


def test_price_targets_on_the_step_grid(records):
    seen, blocks = 0, set()
    fams = mg.vocab()["families"]
    for record, stats, _ in records.values():
        offsets = record["legal_offsets"]
        for k in range(gd.record_decisions(record)):
            row = offsets[k] + record["target"][k]
            lo, hi, step = (int(x) for x in record["legal_price_range"][row])
            price = int(record["price"][k])
            if price >= 0:
                assert lo <= price <= hi and (price - lo) % step == 0
                seen += hi > lo
                blocks.add(fams[int(record["family"][k])])
        assert stats["price_outside_range"] == 0
    assert seen == sum(s["price_targets"] for _, s, _ in records.values()) > 50
    assert {"Bid.company", "BuyTrain.corp"} <= blocks


def test_auto_actions_are_not_decisions(records):
    from tests.title_replay_harness import ruby_filtered_actions

    for title, (record, stats, _) in records.items():
        game = fixture_game(title)
        autos = sum(bool(a.get("_auto")) for a in ruby_filtered_actions(game["actions"], mark_auto=True))
        assert stats.get("not_decision", {}).get("auto", 0) == autos, title


def test_outcome_targets_in_seating_order(records):
    from rl18xx.agent.alphazero.policy_gradient import default_placement_points, placement_points

    for title, (record, _, _) in records.items():
        players = {pid: n for pid, n in record["players"]}
        n = len(players)
        assert list(players) == [1001 + i for i in range(n)] and players[1001] == "Player 1"
        game = gd.new_engine_game(title, players, (), record["seed"])
        for action in json.loads(record["actions"]):
            game.process_action(action)
        result = game.result()
        worth = [result[pid] for pid in players]
        assert record["worth"].tolist() == pytest.approx(worth)
        assert record["win_share"].sum() == pytest.approx(1.0)
        assert record["win_share"][int(np.argmax(worth))] > 0
        assert record["placement"].tolist() == pytest.approx(placement_points(worth).tolist())
        assert max(record["placement"]) == default_placement_points(n)[0]


def test_forced_positions_kept(records):
    record, stats, _ = records["1822"]
    lengths = np.diff(record["legal_offsets"])
    assert stats["forced"] == int((lengths == 1).sum()) > 0
    fams = mg.vocab()["families"]
    assert "RunRoutes" in {fams[f] for f in record["family"].tolist()}


def test_out_of_scope_games_are_dropped_and_counted():
    base = load_fixture("1822MX", "86.json")
    importer = gd.importer_for("1822MX")
    six = [dict(p, id=1000 + i) for i, p in enumerate(base["players"] * 2)]
    cases = {
        "players_2": dict(base, players=base["players"][:2]),
        "players_6": dict(base, players=six),  # Ruby's 1822MX ends at 5
        "optional_rules": dict(base, settings=dict(base.get("settings") or {}, optional_rules=["higher_cert_limit"])),
        "status_active": dict(base, status="active"),
        "title": dict(base, title="1822"),
    }
    for reason, game in cases.items():
        record, stats = gd.convert_game(game, importer)
        assert record is None and stats["dropped"] == {reason: 1}, reason
    record, stats = gd.convert_game(load_fixture("1822", "hs_lxemeslq_30797.json"), gd.importer_for("1822"))
    assert record is None and stats["dropped"] == {"players_7": 1}


def test_optional_rules_only_on_request():
    """Base rules only by default; ``allow_optional_rules`` (a measurement) replays a game under its own rules."""
    game = load_fixture("1822PNW", "merger_high_value_minors.json")  # remove_three_ls
    record, stats = gd.convert_game(game, gd.importer_for("1822PNW"))
    assert record is None and stats["dropped"] == {"optional_rules": 1}
    record, stats = gd.convert_game(game, gd.importer_for("1822PNW", allow_optional_rules=True))
    assert record is not None and record["optional_rules"] == ["remove_three_ls"] and stats["positions"] > 0
    for k, g in gd.replay(record, [gd.record_decisions(record) - 1]):
        assert mg.legal_actions(g).indices.tolist()  # replays under its rules


def test_finished_only_drops_unfinished_games(records):
    """A record that ends before the game does and one the replay rejects part-way are dropped (the family's
    ``supervised_finished_only``), counted by reason; without the option the rejected one is kept up to the
    rejection."""
    importer = gd.importer_for("1822MX")
    game = load_fixture("1822MX", "86.json")  # ends early
    record, stats = gd.convert_game(game, importer, finished_only=True)
    assert record is None and stats["dropped"] == {"unfinished_record_ends": 1}
    bad = {"type": "dividend", "entity": "NOPE", "entity_type": "corporation", "kind": "payout", "id": 10**6}
    rejected = dict(game, actions=game["actions"][:40] + [bad] + game["actions"][40:])
    record, stats = gd.convert_game(rejected, importer, finished_only=True)
    assert record is None and stats["dropped"] == {"unfinished_replay_rejected": 1}
    record, stats = gd.convert_game(rejected, importer, finished_only=False)
    assert record is not None and record["stopped"].startswith("dividend") and not record["finished"]
    full, _, _ = records["1822MX"]
    n = gd.record_decisions(record)
    assert 0 < n < gd.record_decisions(full)
    assert record["chosen_index"].tolist() == full["chosen_index"][:n].tolist()


# ---------------------------------------------------------------------------------------------------
# Dataset, loader, collate
# ---------------------------------------------------------------------------------------------------


def test_dataset_round_trip_and_loader(tmp_path):
    """A conversion with the family's defaults keeps the finished game only (``dropped.jsonl`` lists the other);
    the loader serves every position of it."""
    games = tmp_path / "games"
    games.mkdir()
    for name in ("1822MX_game_end_stock_market.json", "86.json"):  # finished; ends early
        (games / name).write_text(json.dumps(load_fixture("1822MX", name)))
    meta = gd.convert_dataset([str(games)], str(tmp_path / "ds"), title="1822MX", workers=1, validation_percentage=0.5)
    assert meta["finished_only"] is True
    assert meta["stats"]["dropped"] == {"unfinished_record_ends": 1}
    lines = [json.loads(x) for x in (tmp_path / "ds" / "dropped.jsonl").read_text().splitlines()]
    assert [(e["id"], e["reason"], e["players"]) for e in lines] == [("86", "unfinished_record_ends", 3)]
    ds = gd.GenericDataset(tmp_path / "ds")
    assert ds.game_ids() == ["1822MX_game_end_stock_market"]
    assert ds.meta["layouts"]["1822MX"]["fingerprint"] == mg.action_descriptors("1822MX").fingerprint
    rec = ds.record("1822MX_game_end_stock_market")
    assert rec["finished"] and [p for p, _ in rec["players"]] == [1001, 1002, 1003]
    seen = 0
    for split in gd.SPLITS:
        if not ds.game_ids(split):
            continue
        for batch in gd.PositionBatches(tmp_path / "ds", split, 64, shuffle=True, buffer_positions=200, seed=1,
                                        drop_last=False):
            seen += len(batch["target_index"])
            assert batch["inputs"]["price_grid"].all()
            assert set(batch["num_players"].tolist()) == {3}
    assert seen == ds.num_positions() > 500


def test_value_targets_follow_the_reordered_players(records):
    """The family reorders ``BaseGame.players`` every Stock Round and the encoder's seats follow the current order
    (seat 0 = the mover): an item's targets name each seat's player by id, not by a rotation of the seating order."""
    record, _, _ = records["1822"]
    seating = [pid for pid, _ in record["players"]]
    reordered = 0
    for k, game in gd.replay(record, range(0, gd.record_decisions(record), 3)):
        item = gd.position_item(record, k, game)
        current = [p.id for p in game.players]
        rot = int(item["enc"]["meta"][0][5])
        n = len(seating)
        for seat in range(n):
            pid = current[(seat + rot) % n]
            assert item["win"][seat] == record["win_share"][seating.index(pid)]
            assert item["score"][seat] == record["placement"][seating.index(pid)]
        mover = game.g1822_acting_player()
        if mover is not None:
            assert current[rot] == mover
        reordered += current != seating
    assert reordered > 20  # the replay passes Stock Rounds that reorder the players


def test_collate_equals_build_inputs(records):
    record, _, _ = records["1822CA"]
    picks = sorted(random.Random(4).sample(range(gd.record_decisions(record)), 48))
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


# ---------------------------------------------------------------------------------------------------
# Legal prices: every price kind of the family, every sampling path, on the move's step grid
# ---------------------------------------------------------------------------------------------------

#: Fixtures (title, file) whose human moves reach every priced block: bids (Bid.company, Bid.minor), trains
#: bought from another corporation, debt payoffs (CA 1.json), the NdeM's token auction (Bid.token).
PRICE_SOURCES = (
    ("1822CA", "1.json"),
    ("1822PNW", "rockport_token.json"),
    ("1822MX", "1822MX_game_end_stock_market.json"),  # the NdeM's token auctions late in the game
    ("1822", "discard_trains_after_sr_export.json"),
)
PRICED_TYPES = {"bid", "buy_train", "payoff_player_debt", "payoff_player_debt_partial"}


@pytest.fixture(scope="module")
def priced_family():
    """Positions with an open-price legal row of each priced block, harvested from fixture replays where the
    human's next action is a priced type (at most 12 per (title, block))."""
    from tests.title_replay_harness import replay_game

    out: dict = {}
    for title, name in PRICE_SOURCES:
        game = load_fixture(title, name)

        def on_action(rust, i, action, title=title):
            if action["type"] not in PRICED_TYPES or rust.finished:
                return
            moves = json.loads(rust.rl_legal_moves_json())
            for block in {m["block"] for m in moves if m["range"] and m["range"][1] > m["range"][0]}:
                kept = out.setdefault((title, block), [])
                if len(kept) < 12:
                    kept.append(rust.pickle_clone())

        replay_game(game, on_action=on_action)
    blocks = {b for _, b in out}
    assert {"Bid.company", "Bid.minor", "Bid.token", "BuyTrain.corp", "PayoffPlayerDebtPartial"} <= blocks, blocks
    return [g for games in out.values() for g in games]


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


def test_sampled_family_prices_are_legal_and_apply(priced_family, random_model, tmp_path):
    """The PG worker's / eval's path (the server's cell, a price inside it on the grid), the policy-only player's
    (``generic_policy.sample_price``) and ``TitlePlay.apply``: every sampled price of every priced block lies on
    the move's grid in its range and applies."""
    from rl18xx.agent.alphazero import generic_inference as gi
    from rl18xx.agent.alphazero import generic_policy as gpol

    path = tmp_path / "m.pth"
    mg.save_checkpoint(random_model, path, titles=list(gi22.FAMILY))
    policy = gpol.GenericPolicy(str(path))
    server = gi.LocalGenericServer(random_model, seed=5)
    rng = np.random.default_rng(0)
    kinds, steps = set(), set()
    checked = 0
    games = priced_family
    outputs = policy.evaluate(games)
    legal = [o.legal for o in outputs]
    arrays = mg.build_inputs(games, legal)
    for draw in range(3):
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
            steps.add(int(la.price_range[k, 2]))
            price = int(la.price_cells(k).sample_price(int(reply.price_cell[j]), rng))
            assert _is_legal_price(games[b], index, price), (index, price, la.price_range[k])
            price = gpol.sample_price(games[b], outputs[b], k, rng, eps=0.2)
            assert _is_legal_price(games[b], index, price), (index, price, la.price_range[k])
            if draw == 0:  # the PG worker applies the row's wire at the price
                by_play, by_index = games[b].pickle_clone(), games[b].pickle_clone()
                gp.title_play(games[b].title).apply(by_play, la, k, price)
                by_index.rl_apply_index(index, price)
                assert by_play.raw_actions[-1] == by_index.raw_actions[-1]
            checked += 2
    assert checked > 300
    v = mg.vocab()["price_kinds"]
    assert {v[k] for k in kinds} == {"bid", "bid_range", "train_transfer", "amount"}
    assert {1, 5, 10} <= steps  # $1 trains and payoffs, $5 bid boxes, $10 NdeM token bids


# ---------------------------------------------------------------------------------------------------
# Starts: the opening is the bid-box Stock Round
# ---------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("title", list(GAMES))
def test_start_records_cut_at_the_end_of_the_first_stock_round(title):
    game = fixture_game(title)
    replayed, reason = gp.replay_human_game(title, game)
    assert reason is None
    start, midgame, why = gp.start_records(title, replayed)
    assert start is not None and why is None and start["start"] == "sr1_end"
    n = replayed.num_players
    raw = gp.apply_actions(gp.new_title_game(title, n, seed=start["seed"]), start["actions"])
    assert raw.round.round_type != "Stock" and not raw.finished
    cut = len(start["actions"])
    assert cut == gp.opening_end_cut(title, n, replayed.actions, replayed.seed) > 0
    previous = gp.apply_actions(gp.new_title_game(title, n, seed=start["seed"]), start["actions"][:-1])
    assert previous.round.round_type == "Stock"  # the last action of SR1
    if midgame is not None:
        cuts = gp.stock_round_cuts(title, n, replayed.actions, replayed.seed)
        assert cuts[1] == 0 and all(int(k) >= 2 and cuts[int(k)] == v for k, v in midgame["cuts"].items())


@pytest.mark.parametrize("title", list(GAMES))
def test_random_openings_play_the_first_stock_round(title):
    n = 3
    actions = gp.random_opening_actions(title, n, random.Random(2), seed=9)
    assert actions and any(a["type"] == "bid" for a in actions)
    raw = gp.apply_actions(gp.new_title_game(title, n, seed=9), actions)
    assert raw.round.round_type != "Stock" and not raw.finished
    previous = gp.apply_actions(gp.new_title_game(title, n, seed=9), actions[:-1])
    assert previous.round.round_type == "Stock"


def test_family_starts_sample_and_replay(tmp_path):
    title = "1822MX"
    replayed, _ = gp.replay_human_game(title, load_fixture(title, "ndem_dont_double_up_tokens.json"))
    start, midgame, _ = gp.start_records(title, replayed)
    sfile, mfile = tmp_path / "s.jsonl", tmp_path / "m.jsonl"
    sfile.write_text(json.dumps(start) + "\n")
    mfile.write_text(json.dumps(midgame) + "\n" if midgame else "")
    rng = random.Random(0)
    labels, seeds = set(), set()
    for _ in range(30):
        s, raw, skipped = gp.replayed_start(title, 3, str(sfile), 0.3, rng=rng, midgame_path=str(mfile),
                                            midgame_fraction=0.3, auction_fraction=0.3)
        assert skipped == 0
        label = s.label.split(":")[0]
        labels.add(label)
        if label == "auction":  # the fresh game, its company stacks shuffled by a drawn seed
            assert not s.actions and raw.round.round_type == "Stock" and s.seed is not None
            seeds.add(s.seed)
        elif label in ("human", "random"):
            assert raw.round.round_type != "Stock"
        else:
            assert raw.round.round_type == "Stock" and s.label.startswith("midgame:ndem_dont_double_up_tokens:sr")
    assert labels == {"auction", "human", "midgame", "random"} and len(seeds) > 1


# ---------------------------------------------------------------------------------------------------
# Family games through the PG worker
# ---------------------------------------------------------------------------------------------------


@pytest.fixture
def clients(random_model, monkeypatch):
    from rl18xx.agent.alphazero import generic_inference as gi
    from rl18xx.agent.alphazero import generic_pg as gpg

    server = gi.LocalGenericServer(random_model, seed=3)
    local = {name: gi.LocalGenericClient(server) for name in ("learner", "sl", "pool")}
    monkeypatch.setattr(gpg, "_CLIENTS", local)
    return local


@pytest.mark.parametrize("title", ["1822PNW", "1822MX"])
def test_family_pg_games_and_the_trainers_probabilities(title, clients, random_model, tmp_path):
    """Games from the opening (the bid-box Stock Round) and from random openings through the PG worker; the
    trainer recomputes every kept move's sampling probability (price cells on the grid) exactly."""
    from rl18xx.agent.alphazero import generic_pg as gpg

    empty = tmp_path / "starts.jsonl"
    empty.write_text("")
    settings = {
        "title": title, "num_players": 3, "positions_per_seat": 40, "learner_seats": 1,
        "sl_opponent_fraction": 0.5, "pool_ready": False, "pool_label": None, "learner_temperature": 0.5,
        "opponent_temperature": 1.0, "opponent_price_eps": 0.05, "max_decisions": 120, "start_positions": str(empty),
        "random_start_fraction": 1.0, "auction_fraction": 0.5, "reward": "placement", "league": None,
    }  # fmt: skip
    result = gpg.play_generic_pg_games(2, settings)
    rows = [r for g in result["games"] for r in g["rows"]]
    assert rows and all(r.price_grid for r in rows)
    assert sum(r.price_cell >= 0 for r in rows) > 3
    batch = gpg.batch_to_device(gpg.collate_rows(rows), "cpu")
    with torch.no_grad():
        out = random_model(batch["inputs"])
        losses = gpg.generic_pg_losses(out, out, batch, clip=0.2, kl_coef=0.02, entropy_coef=0.0, temperature=0.5)
    assert float(losses["log_ratio_abs_mean"]) < 1e-4
    # seats: the reservoir's seat is the mover's seating seat, encoder seat 0 is the mover, and the targets name
    # players by id (the family reorders its players every Stock Round: the seat frame need not be a rotation)
    for g in result["games"]:
        for r in g["rows"]:
            assert r.order[0] == r.seat and sorted(r.order.tolist()) == list(range(3))
            assert r.win[0] == g["win_share"][r.seat] and r.advantage == g["reward"][r.seat]
            assert r.win[:3].tolist() == [g["win_share"][i] for i in r.order]
