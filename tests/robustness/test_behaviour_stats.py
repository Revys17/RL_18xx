"""M2 behaviour statistics (scripts/robustness/behaviour_stats.py): the event extraction and the counts, on Caleb's
game 21 against the bots (whose counts the 2026-11-14 analysis' own code reproduces) and small synthetic games."""

import json
import sys
from pathlib import Path

import pytest

ROBUST = Path(__file__).resolve().parents[2] / "scripts" / "robustness"
sys.path.insert(0, str(ROBUST))

import behaviour_stats as bs  # noqa: E402
import robust_games as rg  # noqa: E402

GAME_21 = ROBUST / "data" / "caleb_games" / "21.json.gz"


@pytest.fixture(scope="module")
def game21():
    item = rg.load_item(str(GAME_21))
    return item, bs.analyse_item(item, keep_events=True)


def test_game21_groups_and_replay(game21):
    item, rec = game21
    assert item.num_players == 4
    assert item.groups == {1: "human", 2: "bot", 3: "bot", 4: "bot"}
    assert item.site_ids is not None and len(item.site_ids) == len(item.actions)
    assert rec["error"] is None and rec["finished"] and rec["start_k"] == 0


def test_game21_counts_match_the_analysis(game21):
    """Game 21 as docs/bot_weaknesses_2026_11.md describes it: Bot 3 bought C&O before Caleb looted and dumped it
    ([343]), Bot 2 then sold NYC's $900 to Caleb ([408]), three bots bought into the trainless C&O, NYNH bankrupt."""
    _, rec = game21
    c = rec["counts"]
    expected = {
        "broke_dump": 3, "broke_dump_p4": 3, "looted_broke_dump": 2, "broke_dump_victim_bankrupt": 1,
        "pres_sells_lost": 5, "handover_cash300": 2, "takeover_bought": 2, "dump": 7,
        "buy_into_trainless_broke": 3, "bought_then_dumped_same_sr": 3, "bankrupt": 1, "contrib": 1, "contrib$": 630,
        "shuffle_$1": 4, "overpay_own_train": 4, "pres_sells": 16,
    }
    for name, n in expected.items():
        assert c.get(name, 0) == n, name
    # attribution: the bots were the victims of the broke dumps, sold the treasuries away and bought the doomed shares
    assert c["broke_dump|bot"] == 3 and c["handover_cash300|bot"] == 2 and c["buy_into_trainless_broke|bot"] == 3
    assert c["pres_sells_lost|human"] == 1 and c["pres_sells_lost|bot"] == 4
    assert c.get("trainless_turn", 0) == 0 and c.get("suitcase2", 0) == 0


def test_game21_auction(game21):
    _, rec = game21
    rows = {r["co"]: r for r in rec["auction"]}
    assert rows["CA"] == {"co": "CA", "price": 165, "over": 5, "contested": False, "group": "human", "won": 1.0,
                          "place": 0}
    assert rows["MH"]["price"] == 115 and not rows["MH"]["contested"] and rows["MH"]["group"] == "human"
    assert rows["DH"]["price"] == 215 and rows["DH"]["contested"] and rows["DH"]["group"] == "bot"


def test_summary_rates_and_report(game21):
    _, rec = game21
    s = bs.summarize([rec, dict(rec)], label="two copies")
    assert s["games"] == 2 and s["seat_games"] == {"human": 2, "bot": 6}
    assert s["rates"]["broke_dump"]["all"] == pytest.approx(3.0)
    assert s["rates"]["broke_dump"]["bot"] == pytest.approx(1.0)  # 6 dumps over 6 bot seat-games
    assert s["auction"]["CA"]["contested"] == 0.0 and s["auction"]["CA"]["by_group"]["human"]["n"] == 2
    baselines = bs.load_baselines()
    text = bs.format_report([s], baselines)
    assert "broke_dump" in text and "two copies" in text
    for name in ("human", "gen6"):
        assert name in baselines and baselines[name]["games"] > 1000


def _ev(**kw):
    ev = {"auction": {}, "pres_changes": [], "contributions": [], "or_sales": [], "bankruptcies": [], "own_sales": [],
          "buys": [], "train_sales": [], "corp_turns": [], "game_end": {"finished": True, "players": {}}}
    ev.update(kw)
    return ev


def test_loot_then_dump_attribution():
    corp_after = {"cash": 10, "trains": [], "floated": True, "price": 50, "pres": 2, "ipo": 0, "market": 50}
    ev = _ev(
        pres_changes=[{"site_id": 100, "corp": "NYC", "from": 1, "to": 2, "cause": "sold", "phase": "5",
                       "corp_after": corp_after, "depot_next": "5", "round": "Stock", "sr": "6"}],
        train_sales=[{"site_id": 80, "buyer": "PRR", "seller": "NYC", "train": "5", "price": 1, "buyer_pres": 1,
                      "seller_pres": 1}],
        bankruptcies=[{"site_id": 120, "player": 2}],
        buys=[{"site_id": 90, "player": 2, "corp": "NYC", "sr": "6", "round": "Stock", "pres": 1,
               "corp_before": {"cash": 10, "trains": [], "floated": True}, "depot_next": "5"}],
    )
    groups = {1: "A", 2: "B"}
    m = bs.game_metrics(ev, 0, groups.get)
    assert m["broke_dump"] == 1 and m["broke_dump|B"] == 1 and m["broke_dump_p4|B"] == 1
    assert m["looted_broke_dump|B"] == 1 and m["broke_dump_victim_bankrupt|B"] == 1
    assert m["buy_into_trainless_broke|B"] == 1 and m["bought_then_dumped_same_sr|B"] == 1
    assert m["shuffle_$1|A"] == 1 and m["bankrupt|B"] == 1
    # the loot outside the window, or by another president, doesn't count
    ev["train_sales"][0]["site_id"] = 100 - bs.LOOT_WINDOW - 1
    assert bs.game_metrics(ev, 0, groups.get)["looted_broke_dump"] == 0
    # events before the recorded game's own start are not counted
    assert bs.game_metrics(ev, 101, groups.get)["broke_dump"] == 0


def test_suitcase_streaks():
    def turn(sid, trains, pres=1, corp="CPR"):
        return {"site_id": sid, "corp": corp, "pres": pres, "trains_start": [], "trains_end": trains, "ran": False}

    ev = _ev(corp_turns=[turn(1, []), turn(2, []), turn(3, []), turn(4, ["2"]), turn(5, [], corp="ERIE"),
                         turn(6, [])])
    m = bs.game_metrics(ev, 0, {1: "human"}.get)
    assert m["trainless_turn"] == 5 and m["suitcase2|human"] == 1 and m["suitcase3|human"] == 1


def test_collections_and_formats(tmp_path):
    """An eval_policy_only collection (make_game_record files, sides A/B) and an eval_head_to_head one (bare lists)."""
    from rl18xx.agent.alphazero.game_records import make_game_record, save_game_record

    actions = rg.read_json(GAME_21)["actions"][:40]
    actions = [{k: v for k, v in a.items() if k not in ("created_at", "user")} for a in actions]
    po = tmp_path / "po"
    rel = save_game_record(po, "m0_0", make_game_record(actions, 4, meta={"sides": ["A", "B", "B", "B"]}))
    (po / "games.jsonl").write_text(json.dumps({"game_file": rel, "seats": ["A", "B", "B", "B"]}) + "\n")
    h2h = tmp_path / "h2h"
    (h2h / "games").mkdir(parents=True)
    (h2h / "games" / "7.json").write_text(json.dumps(actions))
    (h2h / "games.jsonl").write_text(json.dumps({"game_idx": 7, "seats": ["A", "B", "A", "B"]}) + "\n")
    (h2h / "settings.json").write_text(json.dumps({"start_positions": None}))
    sources = list(rg.iter_sources([str(po), str(h2h)]))
    assert len(sources) == 2
    items = [rg.load_item(*s) for s in sources]
    assert items[0].groups == {1: "A", 2: "B", 3: "B", 4: "B"} and items[0].auction_unlock is False
    assert items[1].groups == {1: "A", 2: "B", 3: "A", 4: "B"} and items[1].auction_unlock is True
    recs = [bs.analyse_item(i) for i in items]
    assert all(r["error"] is None for r in recs)
    assert {r["co"] for r in recs[0]["auction"]} >= {"MH", "CS", "SV"}
