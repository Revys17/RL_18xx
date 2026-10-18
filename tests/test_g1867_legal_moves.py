"""1867 legal-move generator: the RL-readiness table and short random walks
through the wire path (generator -> 18xx.games dict -> process_action).

The long walks run in-crate (``cargo test g1867::fuzz``) and through
``scripts/g1867_fuzz.py``, whose recorded games
``scripts/g1867_fuzz_ruby_check.py`` replays in Ruby HEAD; corpus
completeness is ``scripts/g1867_completeness.py``.
"""

import json
import random
from collections import Counter

import pytest

import engine_rs

from scripts.g1867_fuzz import pick, walk
from tests.title_replay_harness import supported_titles

pytestmark = pytest.mark.skipif("1867" not in supported_titles(), reason="engine lacks 1867")

# Every (step kind, action type) a 1867 step can offer (steps.rs step_actions
# arms over 1867's auction / stock / operating / merger step lists).
OFFERED = {
    "SingleItemAuction": ["bid", "pass"],
    "MajorTrainless": ["pass", "choose"],
    "DiscardTrain": ["discard_train"],
    "HomeToken": ["place_token"],
    "BuySellParShares": ["bid", "par", "buy_shares", "sell_shares", "pass"],
    "BuyCompany": ["buy_company", "pass"],
    "BuyCompanyPreloan": ["buy_company", "pass"],
    "RedeemShares": ["buy_shares", "pass"],
    "Track": ["lay_tile", "pass"],
    "Token": ["place_token", "pass"],
    "Route": ["run_routes"],
    "Dividend": ["dividend"],
    "BuyTrain": ["buy_train", "pass"],
    "ReduceTokens": ["remove_token"],
    "PostMergerShares": ["buy_shares", "pass"],
    "Merge": ["merge", "convert", "pass"],
}
REPLAY_ONLY = {("Bankrupt", "bankrupt"), ("BuyTrain", "sell_shares")}


def test_every_offered_step_action_type_generates():
    table = json.loads(engine_rs.BaseGame.g1867_generator_status_json())
    for step, types in OFFERED.items():
        for t in types:
            assert table[step][t] == "generated", (step, t, table[step][t])
    for step, t in REPLAY_ONLY:
        assert table[step][t].startswith("replay-only: "), (step, t)
        assert len(table[step][t]) > len("replay-only: ") + 10, "replay-only entries carry a reason"


@pytest.mark.parametrize("players,seed", [(3, 11), (4, 12), (5, 13), (6, 14)])
def test_short_wire_walk(players, seed):
    rep = walk(players, seed, 300)
    assert rep["error"] is None, rep["error"]
    assert rep["moves"] == 300 or rep["finished"]


def test_construction_player_counts():
    for n in (3, 6):
        engine_rs.BaseGame.new_titled("1867", {i + 1: f"P{i + 1}" for i in range(n)}, None, 1)
    with pytest.raises(ValueError):
        engine_rs.BaseGame.new_titled("1867", {i + 1: f"P{i + 1}" for i in range(7)}, None, 1)


def test_cross_player_policy():
    """The title's policy (nothing across players) only removes train /
    private purchases and merges with another player's minor from the rules'
    move set, and a walk under the rules' set meets all three (seed 23: the
    first excluded merge at move ~225)."""
    with pytest.raises(ValueError):
        engine_rs.BaseGame.new_titled("1867", {i + 1: f"P{i + 1}" for i in range(3)}, None, 5).g1867_legal_moves_json(
            "bogus"
        )
    g = engine_rs.BaseGame.new_titled("1867", {i + 1: f"Player {i + 1}" for i in range(4)}, None, 23)
    rng = random.Random(23)
    excluded = Counter()
    key = lambda m: json.dumps(m, sort_keys=True)  # noqa: E731
    for n in range(900):
        if g.finished:
            break
        rules = json.loads(g.g1867_legal_moves_json("rules"))
        title = json.loads(g.g1867_legal_moves_json())
        rules_keys = {key(m) for m in rules}
        assert all(key(m) in rules_keys for m in title)
        dropped = [m for m in rules if key(m) not in {key(t) for t in title}]
        assert all(m["wire"]["type"] in ("buy_train", "buy_company", "merge") for m in dropped), dropped
        owner = {c.sym: c.owner_id_str for c in g.corporations}
        for m in dropped:
            if m["wire"]["type"] == "merge":  # a merge add of a minor the initiator's owner doesn't own
                w = m["wire"]
                assert owner[w["corporation"]] != owner[w["entity"]], w
        excluded.update(m["wire"]["type"] for m in dropped)
        g.process_action({**pick(rules, rng), "id": n + 1})
    assert set(excluded) == {"buy_train", "buy_company", "merge"}, excluded
