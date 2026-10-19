"""Hindsight price branching (``price_hindsight``): decisions, events, links,
alternatives, targets, and the branch runner's common random numbers."""

import random

import numpy as np
import pytest

from rl18xx.agent.alphazero import price_hindsight as ph
from rl18xx.agent.alphazero import price_pmf
from rl18xx.agent.alphazero.lockstep import ModelEvaluator
from rl18xx.agent.alphazero.policy_gradient import choose_price
from rl18xx.agent.alphazero.start_positions import apply_actions, new_game, random_start_actions

from .search_test_utils import HashNet


def _random_game(seed: int, max_actions: int = 2000) -> list:
    """A uniformly random game from a random first-Stock-Round start (its action log)."""
    rng = np.random.default_rng(seed)
    g = new_game(4)
    apply_actions(g, random_start_actions(4, random.Random(seed)))
    rs = g._game
    for _ in range(max_actions):
        if g.finished:
            break
        legal = [int(x) for x in rs.factored_legal_indices()]
        move = int(legal[int(rng.integers(len(legal)))])
        price, _, _ = choose_price(g, move, None, rng, 0.0)
        rs.apply_action_index(move, price)
    return list(rs.raw_actions)


@pytest.fixture(scope="module")
def game_actions():
    return _random_game(0)


def test_counterfactual_cash_on_a_clone():
    g = new_game(4)
    rs = g._game
    probe = rs.pickle_clone()
    probe.counterfactual_transfer_cash("player:2", 30)
    probe.counterfactual_transfer_cash("corp:PRR", 7)
    assert [p.cash for p in probe.players] == [600, 630, 600, 600]
    assert [p.cash for p in rs.players] == [600, 600, 600, 600]
    assert probe.corporation_by_id("PRR").cash == 7
    assert probe.bank.cash == rs.bank.cash - 37
    with pytest.raises(ValueError):
        probe.counterfactual_transfer_cash("player:9", 1)


def test_mover_seat_is_the_encoders_rotation(game_actions):
    g = new_game(4)
    rs = g._game
    for i, action in enumerate(game_actions):
        if i % 5 == 0 and not g.finished:
            assert ph.mover_seat(rs) == int(rs.encode_canonical_py()[3])
        rs.process_action(dict(action))


def test_scan_finds_price_decisions_that_replay_as_played(game_actions):
    scan = ph.scan_game("r0", game_actions, None, probe=False)
    assert scan.error is None
    prices = [d for d in scan.decisions if d["kind"] == "price"]
    assert prices, "a random game buys privates and trains between corporations"
    for d in prices:
        assert d["lo"] < d["hi"] and d["lo"] <= d["price"] <= d["hi"]
        assert d["payer"].startswith("corp:") and d["payee"] is not None
        before = ph.replay(game_actions, d["i"])._game
        by_index = before.pickle_clone()
        by_index.apply_action_index(d["index"], d["price"])
        by_log = before.pickle_clone()
        by_log.process_action(dict(game_actions[d["i"]]))
        a, b = by_index.encode_canonical_py(), by_log.encode_canonical_py()
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])
    money = [d for d in scan.decisions if d["kind"] in ph.MONEY_TYPES]
    assert all(d["index"] not in d["alternatives"] and d["alternatives"] for d in money)


def test_unaffordable_probe_and_shortfall():
    # A player with $0 in the first stock round: with the probe's cash every share
    # purchase becomes legal, and the shortfall of one is its price.
    g = new_game(4)
    apply_actions(g, random_start_actions(4, random.Random(3)))
    rs = g._game
    entity = rs.current_entity_id
    cash = ph.entity_cash(rs, entity)
    poor = rs.pickle_clone()
    poor.counterfactual_transfer_cash(entity, -cash)
    legal = set(int(x) for x in poor.factored_legal_indices())
    probe = poor.pickle_clone()
    probe.counterfactual_transfer_cash(entity, 100)
    new = sorted(set(int(x) for x in probe.factored_legal_indices()) - legal)
    assert new
    move = new[0]
    need = ph.shortfall_for(probe, entity, move, 100, offset=100)
    assert need is not None and 1 <= need <= 100
    assert ph.shortfall_for(poor, entity, move, 100) == need
    # One dollar less is not enough.
    under = poor.pickle_clone()
    under.counterfactual_transfer_cash(entity, need - 1)
    assert move not in set(int(x) for x in under.factored_legal_indices())


def test_link_events_directions():
    d_paid = {"kind": "price", "game_id": "g", "i": 10, "decision_number": 5, "payer": "corp:PRR", "payee": "player:1"}
    d_far = {**d_paid, "i": 1, "decision_number": 0}
    d_other = {**d_paid, "i": 11, "decision_number": 6, "payer": "corp:NYC", "payee": "player:2"}
    events = [
        {"game_id": "g", "i": 20, "decision_number": 9, "kind": "unaffordable", "constrained": ["player:1"]},
        {"game_id": "g", "i": 21, "decision_number": 10, "kind": "president_contribution", "constrained": ["corp:PRR", "player:1"]},
    ]
    links = ph.link_events([d_paid, d_far, d_other], events, window=6)
    got = sorted((e["kind"], d["i"], helps) for e, d, helps in links)
    # The player was paid (a higher price helps); before the emergency the corporation
    # paid its own president: the pool both pay from is unchanged (0).
    assert got == [("president_contribution", 10, 0), ("unaffordable", 10, 1)]
    corp_only = [{**events[1], "constrained": ["corp:PRR"]}]
    assert [(d["i"], h) for _, d, h in ph.link_events([d_paid], corp_only, 6)] == [(10, -1)]
    # A lost bid: the loser's bid on that company (higher helps) and, as cash, its bid elsewhere.
    bid = {"kind": "price", "type": "bid", "game_id": "g", "decision_number": 1, "payer": "player:3", "payee": None}
    same, other = {**bid, "i": 2, "company": "CA"}, {**bid, "i": 3, "company": "MH"}
    lost = [{"game_id": "g", "i": 5, "decision_number": 4, "kind": "bid_lost", "constrained": ["player:3"],
             "company": "CA"}]
    assert sorted((d["company"], h) for _, d, h in ph.link_events([same, other], lost, 6)) == [("CA", 1), ("MH", -1)]


def test_price_alternatives():
    assert ph.price_alternatives(40, 10, 40) == [40, 10, 20, 30, 35, 39]
    assert ph.price_alternatives(1, 1, 300, deltas=(-5, 5)) == [1, 6, 300]
    assert ph.price_alternatives(100, 50, 200, deltas=(-1, 1), extremes=False) == [100, 99, 101]


def test_cell_targets_follow_advantages():
    cells = price_pmf.price_cells("BuyCompany", 20, 80)
    probs = cells.cell_probs(np.zeros(price_pmf.NUM_CELLS))
    hi_cell, lo_cell = cells.cell_of(80), cells.cell_of(20)
    adv = {hi_cell: (0.0, 0.0, [80]), lo_cell: (0.05, 0.01, [20])}
    t = ph.price_cell_target_from_advantages(probs, adv, beta=0.05)
    assert t[lo_cell] > probs[lo_cell] and t[hi_cell] < probs[hi_cell]
    assert abs(t.sum() - 1) < 1e-9 and np.all(t[~cells.nonempty] == 0)
    # Below z standard errors: no change.
    weak = {hi_cell: (0.0, 0.0, [80]), lo_cell: (0.005, 0.01, [20])}
    np.testing.assert_allclose(ph.price_cell_target_from_advantages(probs, weak, 0.05, z=2.0), probs)
    assert ph.expected_under(probs, {hi_cell: 0.0, lo_cell: 0.1}) == pytest.approx(0.05)


def test_split_half_gain_and_paired_advantages():
    def alt(price, ys):
        return {"price": price, "delta": price - 40, "rollouts": [{"r": r, "win": [y, 0, 0, 0]} for r, y in enumerate(ys)]}

    result = {"seat": 0, "alts": [alt(40, [0, 1, 0, 1]), alt(35, [1, 1, 1, 1]), alt(30, [0, 0, 0, 0])]}
    adv = {a["price"]: (m, se) for a, m, se, _ in ph.paired_advantages(result)}
    assert adv[35][0] == pytest.approx(0.5) and adv[30][0] == pytest.approx(-0.5)
    assert ph.split_half_gain(result) == (-5, pytest.approx(0.0))  # odd rollouts: 1 - 1
    cells = ph.cell_advantages({**result, "alts": [{**a, "cell": k} for k, a in enumerate(result["alts"])]})
    assert cells[0][0] == 0.0 and cells[1][0] == pytest.approx(0.5)


def _script():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[3] / "scripts" / "price_hindsight.py"
    spec = importlib.util.spec_from_file_location("price_hindsight_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_finetune_rows_hindsight_and_played_targets():
    script = _script()
    cells = price_pmf.price_cells("BuyCompany", 20, 80)
    probs = cells.cell_probs(np.zeros(price_pmf.NUM_CELLS))
    hi, lo = cells.cell_of(80), cells.cell_of(20)

    def alt(price, cell, ys):
        return {"price": price, "delta": price - 80, "cell": cell, "rollouts": [{"r": r, "win": [y, 0, 0, 0]} for r, y in enumerate(ys)]}

    job = {"job_id": 0, "decision": {"kind": "price", "slot": 55, "lo": 20, "hi": 80, "price": 80}, "heldout": False}
    result = {"job": 0, "seat": 0, "cell_probs": probs.tolist(), "alts": [alt(80, hi, [0, 0, 1, 0]), alt(20, lo, [1, 1, 1, 0])]}
    awr = script.finetune_rows({0: job}, [result], "win", beta=0.1, z=0.0, kind="awr")[0]
    played = script.finetune_rows({0: job}, [result], "win", beta=0.1, z=0.0, kind="played")[0]
    assert awr["target"][lo] > probs[lo] and awr["cell_adv"][lo][0] == pytest.approx(0.5)
    assert played["target"][hi] == 1.0 and played["target"].sum() == 1.0


def _job(actions, decision, seed=7):
    return {"job_id": 0, "seed": seed, "actions": actions, "decision": decision, "event": None}


def test_branch_runner_is_deterministic_with_common_random_numbers(game_actions):
    scan = ph.scan_game("r0", game_actions, None, probe=False)
    d = next(x for x in scan.decisions if x["kind"] == "price")
    net = HashNet()
    settings = {"rollouts": 2, "temperature": 0.5, "max_decisions": 12, "horizons": [0, 5],
                "deltas": [-5, 5], "extremes": False}
    runs = []
    for _ in range(2):
        runner = ph.BranchRunner({"policy": ModelEvaluator(net), "value": ModelEvaluator(net)}, settings)
        results, decisions = runner.run([_job(game_actions, d)])
        runs.append(results[0])
        assert decisions > 0
    a, b = runs
    assert a["error"] is None and len(a["alts"]) >= 2
    assert a["alts"][0]["price"] == d["price"]
    for x, y in zip(a["alts"], b["alts"]):
        assert [r["win"] for r in x["rollouts"]] == [r["win"] for r in y["rollouts"]]
        assert [r["values"] for r in x["rollouts"]] == [r["values"] for r in y["rollouts"]]
        assert len(x["rollouts"]) == 2 and all("0" in r["values"] for r in x["rollouts"])
    assert "cell_probs" in a and abs(sum(a["cell_probs"]) - 1) < 1e-4
