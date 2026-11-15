"""M1 regression set (scripts/robustness/regression_set.py): the fixture replays to every position, its moves are legal
and labelled consistently, and the scorer's plumbing (legacy search, generic policy/value) works on stand-in nets."""

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch

ROBUST = Path(__file__).resolve().parents[2] / "scripts" / "robustness"
sys.path.insert(0, str(ROBUST))

import regression_set as rs  # noqa: E402


@pytest.fixture(scope="module")
def positions():
    return rs.load_fixture()


@pytest.fixture(scope="module")
def records(positions):
    return {gid: rs.load_game_record(gid) for gid in sorted({p["game"] for p in positions})}


def test_fixture_shape(positions):
    assert 30 <= len(positions) <= 60
    assert len({p["id"] for p in positions}) == len(positions)
    by = Counter(p["weakness"] for p in positions)
    assert set(by) == set(rs.WEAKNESSES) and min(by.values()) >= 8
    for p in positions:
        assert (rs.DATA / p["file"]).exists()
        assert p["criterion"] in ("avoid_bad", "choose_preferred")
        assert p["bad_moves"] and p["preferred_moves"]
        assert p["mover"]["name"].startswith("Bot")
    assert [(p["game"], p["site_action_id"]) for p in positions] == [(g, s) for g, s, *_ in rs.CURATED]


def test_every_position_replays_with_consistent_moves(positions, records):
    """At each position the bot is to move, its played move is legal (one index), every bad and preferred move is
    legal, and no move is both. (Repeats are refused by ``build``.)"""
    for p in positions:
        game = rs.rust_game_at(records[p["game"]], p["action_index"])
        assert rs.acting_player(game) == p["mover"]["player_id"], p["id"]
        moves = rs.legal_moves(game)
        played = rs.match_indices(game, p["played"], moves)
        assert len(played) == 1, p["id"]
        bad = {i for a in p["bad_moves"] for i in rs.match_indices(game, a, moves)}
        pref = {i for a in p["preferred_moves"] for i in rs.match_indices(game, a, moves)}
        assert len(bad) >= 1 and len(pref) >= 1 and not bad & pref, p["id"]
        if p["criterion"] == "avoid_bad":
            assert played[0] in bad, p["id"]
        else:
            assert played[0] not in pref, p["id"]


def test_sellout_bad_moves_hand_the_presidency_over(positions, records):
    for p in [p for p in positions if p["kind"] == "sellout"][:3]:
        game = rs.rust_game_at(records[p["game"]], p["action_index"])
        corp = p["context"]["corp"]["name"]
        assert p["context"]["corp"]["president"] == p["mover"]["player_id"]
        for action in p["bad_moves"]:
            after = rs.apply_action(game, action)
            assert int(after.corporation_by_id(corp).owner.id) != p["mover"]["player_id"]


def test_match_ignores_share_ids_and_open_prices(positions, records):
    p = next(p for p in positions if p["id"] == "g21-a6")  # Caleb's CA $165 stands: a raise at any price matches
    game = rs.rust_game_at(records[21], p["action_index"])
    ca = [a for a in p["preferred_moves"] if a.get("company") == "CA"][0]
    assert rs.match_indices(game, dict(ca, price=ca["price"] + 25)) == rs.match_indices(game, ca)
    q = next(p for p in positions if p["id"] == "g21-a235")  # a market C&O share, whichever certificate
    game = rs.rust_game_at(records[21], q["action_index"])
    shares = q["played"]["shares"][0].split("_")
    other = dict(q["played"], shares=[f"{shares[0]}_{int(shares[1]) + 1}"])
    try:
        rs.apply_action(game, other)
    except Exception:
        pytest.skip("the neighbouring certificate isn't in the market here")
    assert rs.match_indices(game, other) == rs.match_indices(game, q["played"])


def test_score_with_a_stand_in_legacy_network(positions, records):
    from tests.agent.alphazero.search_test_utils import HashNet

    overrides = dict(num_readouts=16, min_readouts=16, mcts_mean_q=True, fpu_reduction=0.0, c_puct_init=0.05,
                     c_puct_by_round={}, dedupe_leaves=True, parallel_readouts=8, prior_temperature=0.5)
    system = rs.LegacySystem.from_network(HashNet(), overrides, label="hashnet")
    chosen = [p for p in positions if p["id"] in ("g21-a33", "g21-a408", "g10-a2")]
    results = [rs.score_position(system, p, records[p["game"]]) for p in chosen]
    for r in results:
        assert 0.0 <= r["policy"]["p_bad"] <= 1.0 and 0.0 <= r["policy"]["p_preferred"] <= 1.0
        assert r["policy"]["p_bad"] + r["policy"]["p_preferred"] <= 1.0 + 1e-6
        assert r["search"]["readouts"] >= 16 and 0.0 <= r["search"]["share_bad"] <= 1.0
        assert isinstance(r["pass"], bool)
        assert set(r["value"]["before"]) == {"search"}  # a stand-in has no win / score heads
    s = rs.summarize(results)
    assert s["all"]["n"] == 3 and "search_chose_bad" in s["all"]
    assert "auction" in rs.format_summary(s)
    # no search: the policy's argmax decides
    r = rs.score_position(system, chosen[0], records[chosen[0]["game"]], do_search=False)
    assert "search" not in r and r["pass"] == (not r["policy"]["argmax_bad"])


def test_generic_policy_and_value(positions, records, tmp_path):
    """The generic hook: a tiny random generic checkpoint gives probabilities over the legal moves and values."""
    from rl18xx.agent.alphazero import model_generic as mg

    torch.manual_seed(0)
    small = dict(d_model=32, num_heads=4, hex_layers=1, entity_layers=1, fusion_layers=1, d_trunk=64, trunk_blocks=1,
                 d_head=32, d_edge=8, d_cond=16, d_id=8)
    model = mg.GenericPolicyValueModel(mg.GenericModelConfig(**small)).eval()
    path = tmp_path / "generic.pth"
    mg.save_checkpoint(model, path, titles=["1830"])
    system = rs.load_system(f"generic:{path}@0.5")
    assert system.kind == "generic" and system.prior_temperature == 0.5
    p = next(p for p in positions if p["id"] == "g21-a408")
    r = rs.score_position(system, p, records[21], do_search=False)
    assert "search" not in r
    game = rs.rust_game_at(records[21], p["action_index"])
    legal = [i for i, _ in rs.legal_moves(game)]
    probs = system.policy(game, legal)
    assert probs.shape == (len(legal),) and np.isclose(probs.sum(), 1.0)
    v = system.values(game, p["mover"]["player_id"])
    assert 0.0 <= v["win"] <= 1.0 and set(v) == {"win", "score", "search"}
    with pytest.raises(NotImplementedError):
        system.search(game)
