"""1822's optional rules on the title-generic pipeline (1822+, its No Removals / Single Stack modifiers, the
multi-share tax haven; ``docs/rl_model.md`` "1822+: the optional rules"): the rules helpers of ``titles``
(canonical rules, combination names, the rules mix), the importer's scope and records under each combination,
the encoder's rule flags on the replayed positions, start lines per combination and their sampling, and the PG
worker drawing each game's combination from a mix.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch

engine_rs = pytest.importorskip("engine_rs")

from rl18xx.agent.alphazero import generic_data as gd  # noqa: E402
from rl18xx.agent.alphazero import generic_play as gp  # noqa: E402
from rl18xx.agent.alphazero import model_generic as mg  # noqa: E402
from rl18xx.agent.alphazero.generic_tokens import encode_batch, schema  # noqa: E402
from rl18xx.agent.alphazero.titles import draw_rules, rules_name, title_spec  # noqa: E402

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "1822"
PLUS = ("plus_expansion",)
ALL = ("plus_expansion", "plus_expansion_no_removals", "plus_expansion_single_stack", "tax_haven_multiple")
FLAGS = ("rule_plus_expansion", "rule_plus_no_removals", "rule_plus_single_stack", "rule_tax_haven_multiple")
SMALL = dict(
    d_model=32, num_heads=4, hex_layers=1, entity_layers=1, fusion_layers=1, d_trunk=64, trunk_blocks=1,
    d_head=32, d_edge=8, d_cond=16, d_id=8,
)  # fmt: skip


def load(name: str, limit: int = 0, rules=None) -> dict:
    game = json.loads((FIXTURES / name).read_text())
    game["id"] = Path(name).stem
    if limit:
        game["actions"] = game["actions"][:limit]
    if rules is not None:
        game["settings"] = dict(game.get("settings") or {}, optional_rules=list(rules))
    return game


def flags(game) -> tuple:
    cols = [schema()["entity_features"]["global"].index(f) for f in FLAGS]
    return tuple(int(x) for x in encode_batch([game])["ent_feat"][0, 0, cols])


# ---------------------------------------------------------------------------------------------------
# titles
# ---------------------------------------------------------------------------------------------------


def test_rules_helpers():
    spec = title_spec("1822")
    assert spec.optional_rules == ALL
    # canonical: the spec's order, modifiers without 1822+ dropped (Ruby ignores them)
    assert spec.canonical_rules(["tax_haven_multiple", "plus_expansion"]) == ("plus_expansion", "tax_haven_multiple")
    assert spec.canonical_rules(["plus_expansion_no_removals", "plus_expansion_single_stack"]) == ()
    with pytest.raises(ValueError):
        spec.canonical_rules(["higher_cert_limit"])
    assert spec.canonical_rules(["higher_cert_limit"], strict=False) == ("higher_cert_limit",)
    # names round-trip
    for name in ("base", "plus", "plus+nr", "plus+ss+th", "plus+nr+ss+th", "th"):
        assert spec.rules_name(spec.rules_from_name(name)) == name
    assert spec.rules_name(["plus_expansion_single_stack", "tax_haven_multiple"]) == "th"
    for bad in ("nr", "ss+th", "plus+higher_cert_limit"):
        with pytest.raises(ValueError):
            spec.rules_from_name(bad)
    # the mix: the default is the human games' combinations, base included; strings and dicts parse
    mix = spec.default_rules_mix()
    assert "base" in mix and "plus+nr+ss+th" in mix and all(w > 0 for w in mix.values())
    assert spec.parse_rules_mix(None) == mix == spec.parse_rules_mix("default")
    assert spec.parse_rules_mix("plus") == {"plus": 1.0}
    assert spec.parse_rules_mix({"plus_expansion+tax_haven_multiple": 2, "base": 1}) == {"plus+th": 2.0, "base": 1.0}
    with pytest.raises(ValueError):
        spec.parse_rules_mix("plus:0")
    rng = random.Random(3)
    drawn = Counter(rules_name("1822", draw_rules({"plus": 3, "base": 1}, rng, "1822")) for _ in range(4000))
    assert set(drawn) == {"plus", "base"} and 0.70 < drawn["plus"] / 4000 < 0.80
    assert draw_rules({"plus+nr": 1}, rng, "1822") == ("plus_expansion", "plus_expansion_no_removals")
    # every other title: base rules only, unchanged
    for title in ("1830", "1867", "1822PNW", "1822MX", "1822CA"):
        assert title_spec(title).default_rules_mix() == {"base": 1.0}
    assert title_spec("1822PNW").optional_rules == title_spec("1822CA").optional_rules == ()


# ---------------------------------------------------------------------------------------------------
# Importer and records
# ---------------------------------------------------------------------------------------------------


def test_importer_scope_and_records_under_each_combination():
    """1822 games under any combination of its four rules are in scope, replayed and recorded under their canonical
    rules (combination name and the export's own list kept); other rules stay out; the replayed positions carry the
    rule flags."""
    importer = gd.importer_for("1822")
    cases = {
        "plus": load("33845.json", 300),  # 1822+
        "plus+nr+ss": load("1822_game_end_stock_market.json", 300),
        "base": load("discard_trains_after_sr_export.json", 200, rules=["plus_expansion_single_stack"]),  # inert
        "th": load("discard_trains_after_sr_export.json", 200, rules=["tax_haven_multiple"]),
    }
    for name, game in cases.items():
        record, stats = gd.convert_game(game, importer)
        assert record is not None, (name, stats)
        assert record["rules"] == name and gd.record_rules_name(record) == name
        assert tuple(record["optional_rules"]) == title_spec("1822").rules_from_name(name)
        assert record["settings_optional_rules"] == list(game["settings"].get("optional_rules") or [])
        assert stats["positions"] > 50 and not stats.get("unmatched") and not stats.get("ambiguous")
        want = tuple(int(r in record["optional_rules"]) for r in ALL)
        last = gd.record_decisions(record) - 1
        for _, g in gd.replay(record, [0, last]):
            assert sorted(g.optional_rules) == sorted(record["optional_rules"])
            assert flags(g) == want, name
    record, stats = gd.convert_game(load("33845.json", 50, rules=["plus_expansion", "foo"]), importer)
    assert record is None and stats["dropped"] == {"optional_rules": 1}


def test_dataset_counts_by_rules(tmp_path):
    games = tmp_path / "games"
    games.mkdir()
    for gid, name, rules in (("1", "33845.json", None), ("2", "discard_trains_after_sr_export.json", None)):
        game = load(name, 150, rules)
        (games / f"{gid}.json").write_text(json.dumps(game))
    meta = gd.convert_dataset([str(games)], str(tmp_path / "data"), title="1822", workers=1, finished_only=False)
    by_rules = meta["counts"]["by_rules"]
    assert set(by_rules) == {"plus", "base"}
    assert sum(by_rules["plus"]["games"].values()) == 1 and by_rules["plus"]["games_by_players"] == {"4": 1}
    index = [json.loads(line) for line in (tmp_path / "data" / "games.jsonl").read_text().splitlines()]
    assert {e["id"]: (e["rules"], e["optional_rules"]) for e in index} == {
        "1": ("plus", ["plus_expansion"]),
        "2": ("base", []),
    }


# ---------------------------------------------------------------------------------------------------
# Starts
# ---------------------------------------------------------------------------------------------------


def test_start_lines_carry_their_rules_and_sample_by_combination(tmp_path):
    title = "1822"
    lines = {}
    for name in ("33845.json", "1822_game_end_stock_market.json"):
        replayed, reason = gp.replay_human_game(title, load(name))
        assert reason is None
        start, midgame, why = gp.start_records(title, replayed)
        assert start is not None and why is None and midgame is not None
        assert start["optional_rules"] == midgame["optional_rules"] == list(title_spec(title).canonical_rules(
            replayed.optional_rules))  # fmt: skip
        lines[rules_name(title, start["optional_rules"])] = (start, midgame)
    assert set(lines) == {"plus", "plus+nr+ss"}
    sfile, mfile = tmp_path / "s.jsonl", tmp_path / "m.jsonl"
    sfile.write_text("".join(json.dumps(s) + "\n" for s, _ in lines.values()))
    mfile.write_text("".join(json.dumps(m) + "\n" for _, m in lines.values()))
    rng = random.Random(0)
    for want in ("plus", "plus+nr+ss"):
        rules = title_spec(title).rules_from_name(want)
        kinds = Counter()
        for _ in range(12):
            s, raw, skipped = gp.replayed_start(title, 4, str(sfile), 0.0, rng=rng, midgame_path=str(mfile),
                                                midgame_fraction=0.5, optional_rules=rules)  # fmt: skip
            assert skipped == 0 and s.optional_rules == rules
            assert tuple(raw.optional_rules) == rules and flags(raw)[0] == 1
            kinds[s.label.split(":")[0]] += 1
            if s.label.startswith(("human:", "midgame:")):
                game_id = s.label.split(":")[1]
                assert rules_name(title, lines[want][0]["optional_rules"]) == want
                assert game_id == lines[want][0]["id"]
        assert set(kinds) == {"human", "midgame"}
    # a combination without lines of this count: a random opening under its rules; base lines don't count
    s, raw, _ = gp.replayed_start(title, 4, str(sfile), 0.0, rng=rng, optional_rules=("tax_haven_multiple",))
    assert s.label == "random" and tuple(raw.optional_rules) == ("tax_haven_multiple",)
    assert raw.round.round_type != "Stock" and flags(raw) == (0, 0, 0, 1)
    s, raw, _ = gp.replayed_start(title, 4, str(sfile), 0.0, rng=rng, auction_fraction=1.0, optional_rules=ALL)
    assert s.label == "auction" and tuple(raw.optional_rules) == ALL and flags(raw) == (1, 1, 1, 1)
    # old lines without the field are base rules
    old = tmp_path / "old.jsonl"
    old.write_text(json.dumps({k: v for k, v in lines["plus"][0].items() if k != "optional_rules"}) + "\n")
    gp._LOADED.clear()
    assert gp.load_start_file(str(old), title)[4][0].optional_rules == ()


# ---------------------------------------------------------------------------------------------------
# PG worker
# ---------------------------------------------------------------------------------------------------


@pytest.fixture
def clients(monkeypatch):
    from rl18xx.agent.alphazero import generic_inference as gi
    from rl18xx.agent.alphazero import generic_pg as gpg

    torch.manual_seed(1)
    model = mg.GenericPolicyValueModel(mg.GenericModelConfig(**SMALL)).eval()
    server = gi.LocalGenericServer(model, seed=3)
    local = {name: gi.LocalGenericClient(server) for name in ("learner", "sl", "pool")}
    monkeypatch.setattr(gpg, "_CLIENTS", local)
    return local


def test_pg_games_draw_their_rules_from_the_mix(clients, tmp_path):
    from rl18xx.agent.alphazero import generic_pg as gpg

    empty = tmp_path / "starts.jsonl"
    empty.write_text("")
    settings = {
        "title": "1822", "num_players": 3, "positions_per_seat": 8, "learner_seats": 1,
        "sl_opponent_fraction": 0.5, "pool_ready": False, "pool_label": None, "learner_temperature": 0.5,
        "opponent_temperature": 1.0, "opponent_price_eps": 0.05, "max_decisions": 25, "start_positions": str(empty),
        "random_start_fraction": 1.0, "auction_fraction": 1.0, "reward": "placement", "league": None,
        "save_game_every": 1, "rules_mix": {"plus+nr": 1.0, "th": 1.0},
        "seed": [1822, 0],  # unseeded, 6 draws of a 50/50 mix were all one combination 1 time in 32
    }  # fmt: skip
    result = gpg.play_generic_pg_games(6, settings)
    seen = set()
    for g in result["games"]:
        assert g["rules"] in ("plus+nr", "th")
        seen.add(g["rules"])
        saved = g["saved_game"]
        assert tuple(saved["optional_rules"]) == title_spec("1822").rules_from_name(g["rules"])
    assert seen == {"plus+nr", "th"}
    # the config resolves its mix (None: the title's default; a name: that combination alone)
    cfg = gpg.GenericPGConfig(policy_checkpoint="x", title="1822", rules_mix=None)
    assert title_spec("1822").parse_rules_mix(cfg.rules_mix) == title_spec("1822").default_rules_mix()


def test_eval_summary_by_rules():
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
    from eval_generic import by_rules

    records = [
        {"rules": "plus", "decisions": 10, "termination": "finished", "win_share": [1, 0, 0], "seats": ["A", "B", "B"]},
        {"rules": "plus", "decisions": 30, "termination": "finished", "win_share": [0, 1, 0], "seats": ["A", "B", "B"]},
        {"rules": "base", "decisions": 20, "termination": "dead_end", "win_share": [0, 0, 1], "seats": ["B", "B", "A"]},
    ]
    out = by_rules(records, 1 / 3)
    assert out["plus"]["games"] == 2 and out["plus"]["decisions_max"] == 30 and out["plus"]["A_score"] == 0.5
    assert out["base"]["terminations"] == {"dead_end": 1} and out["base"]["A_score"] == 1.0
    assert np.isclose(out["base"]["A_score_equal"], 1 / 3)


# ---------------------------------------------------------------------------------------------------
# All-pass openings: redrawn in self-play and evals (the rule itself unchanged)
# ---------------------------------------------------------------------------------------------------


def all_pass_game(title: str, n: int = 3, seed: int = 5):
    """A family game whose opening Stock Round everybody passed: Ruby ends it ("dnf"), every seat tied."""
    raw = gp.raw_game(gp.new_title_game(title, n, seed=seed))
    while not raw.finished:
        moves = json.loads(raw.g1822_legal_moves_json(None))
        raw.process_action(dict(next(m for m in moves if m["wire"]["type"] == "pass")["wire"]))
    return raw


def test_all_pass_opening_is_recognized():
    for title in ("1822", "1822PNW", "1822MX", "1822CA"):
        raw = all_pass_game(title)
        assert raw.finished and len(raw.raw_actions) == 3 and gp.all_pass_opening(title, raw), title
        fresh = gp.raw_game(gp.new_title_game(title, 3, seed=5))
        assert not gp.all_pass_opening(title, fresh)
    # a finished human game ends otherwise; 1830 / 1867 never open with a Stock Round
    from tests.title_replay_harness import replay_game

    captured = {}
    game = load("discard_trains_after_sr_export.json")
    replay_game(game, on_action=lambda rust, i, a: captured.setdefault("g", rust))
    assert captured["g"].finished and not gp.all_pass_opening("1822", captured["g"])
    assert not gp.all_pass_opening("1867", gp.raw_game(gp.new_title_game("1867", 3, seed=1)))


def _redraw_first_start(monkeypatch, title: str, n: int):
    """``generic_play.replayed_start`` whose first start is an all-pass opening (later ones: the real starts)."""
    real = gp.replayed_start
    calls = {"n": 0}

    def fake(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return gp.GenericStart(gp.AUCTION_START, (), 5), all_pass_game(title, n), 0
        return real(*args, **kwargs)

    monkeypatch.setattr(gp, "replayed_start", fake)
    return calls


def test_pg_redraws_an_all_pass_opening(clients, tmp_path, monkeypatch):
    """A game whose opening Stock Round ends with nothing bought is dropped (no record, no rows) and its slot
    plays a fresh draw, counted in ``all_pass_redrawn``; with ``redraw_all_pass`` off it is kept; normal games are
    unaffected."""
    from rl18xx.agent.alphazero import generic_pg as gpg

    empty = tmp_path / "starts.jsonl"
    empty.write_text("")
    settings = {
        "title": "1822MX", "num_players": 3, "positions_per_seat": 8, "learner_seats": 1,
        "sl_opponent_fraction": 0.5, "pool_ready": False, "pool_label": None, "learner_temperature": 0.5,
        "opponent_temperature": 1.0, "opponent_price_eps": 0.05, "max_decisions": 20, "start_positions": str(empty),
        "random_start_fraction": 1.0, "auction_fraction": 1.0, "reward": "placement", "league": None,
    }  # fmt: skip
    calls = _redraw_first_start(monkeypatch, "1822MX", 3)
    result = gpg.play_generic_pg_games(2, settings)
    assert result["all_pass_redrawn"] == 1 and calls["n"] == 3 and result["num_players"] == 3
    assert len(result["games"]) == 2
    for g in result["games"]:
        assert g["termination"] == "max_length" and g["decisions"] == 20 and g["rows"]
    # kept when switched off: a 0-decision game with every seat tied
    _redraw_first_start(monkeypatch, "1822MX", 3)
    result = gpg.play_generic_pg_games(1, dict(settings, redraw_all_pass=False))
    assert result["all_pass_redrawn"] == 0 and result["games"][0]["termination"] == "finished"
    assert result["games"][0]["decisions"] == 0 and len(set(result["games"][0]["win_share"])) == 1
    # normal games: nothing redrawn
    monkeypatch.setattr(gp, "replayed_start", _REAL_REPLAYED_START)
    result = gpg.play_generic_pg_games(2, settings)
    assert result["all_pass_redrawn"] == 0 and [g["termination"] for g in result["games"]] == ["max_length"] * 2


_REAL_REPLAYED_START = gp.replayed_start


def test_saved_pg_games_from_later_starts_replay(clients, tmp_path):
    """A saved game of a mid-game start replays from the fresh game under its rules and seed: its players are in
    seating order (the family reorders ``raw.players`` every Stock Round; before, the order after the start's
    actions was saved and such games did not replay)."""
    from rl18xx.agent.alphazero import generic_pg as gpg

    replayed, _ = gp.replay_human_game("1822", load("33845.json"))
    _, midgame, _ = gp.start_records("1822", replayed)
    mfile, empty = tmp_path / "m.jsonl", tmp_path / "s.jsonl"
    mfile.write_text(json.dumps(midgame) + "\n")
    empty.write_text("")
    settings = {
        "title": "1822", "num_players": 4, "positions_per_seat": 4, "learner_seats": 2,
        "sl_opponent_fraction": 0.5, "pool_ready": False, "pool_label": None, "learner_temperature": 1.0,
        "opponent_temperature": 1.0, "opponent_price_eps": 0.0, "max_decisions": 12, "start_positions": str(empty),
        "random_start_fraction": 0.0, "midgame_positions": str(mfile), "midgame_fraction": 1.0, "reward": "placement",
        "league": None, "save_game_every": 1, "rules_mix": {"plus": 1.0},
    }  # fmt: skip
    result = gpg.play_generic_pg_games(2, settings)
    for g in result["games"]:
        saved = g["saved_game"]
        assert saved["start"].startswith("midgame:33845:sr") and saved["optional_rules"] == ["plus_expansion"]
        assert [pid for pid, _ in saved["players"]] == [1001, 1002, 1003, 1004]
        game = engine_rs.BaseGame.new_titled("1822", dict(saved["players"]), saved["optional_rules"], saved["seed"])
        for action in saved["raw_actions"]:
            game.process_action(dict(action))
        assert len(game.raw_actions) == len(saved["raw_actions"])


def test_eval_redraws_an_all_pass_opening(tmp_path, monkeypatch):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
    from eval_generic import play_games
    from eval_policy_only import arrangements

    torch.manual_seed(2)
    path = tmp_path / "m.pth"
    mg.save_checkpoint(mg.GenericPolicyValueModel(mg.GenericModelConfig(**SMALL)), path, titles=["1822PNW"])
    settings = {
        "title": "1822PNW", "paths": {"A": str(path), "B": str(path)}, "temperatures": {"A": 1.0, "B": 1.0},
        "server": False, "seed": 0, "start_positions": None, "random_start_fraction": 1.0, "auction_start": True,
        "max_decisions": 15, "auction_stall_decisions": 0, "price_eps": 0.0, "save_games": 0,
        "arrangements": arrangements(1, 3),
    }  # fmt: skip
    _redraw_first_start(monkeypatch, "1822PNW", 3)
    records = play_games([0, 1, 2], settings)
    assert sorted(r["idx"] for r in records) == [0, 1, 2]
    assert sorted(r["all_pass_redrawn"] for r in records) == [0, 0, 1]
    assert all(r["termination"] == "max_length" and r["decisions"] == 15 for r in records)
