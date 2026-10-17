"""1822 legal-move generator: the RL-readiness table and a short random
walk through the wire path (generator -> 18xx.games dict -> process_action).

The long walks run in-crate (``cargo test g1822::fuzz``) and through
``scripts/g1822_fuzz.py``, whose recorded games ``scripts/ruby_diff.py``
replays in Ruby HEAD.
"""

import json
import random

import pytest

import engine_rs

from scripts.g1822_fuzz import pick, walk
from tests.title_replay_harness import supported_titles

pytestmark = pytest.mark.skipif("1822" not in supported_titles(), reason="engine lacks 1822")


def test_every_step_action_type_generates():
    table = engine_rs.BaseGame.g1822_generator_table()
    assert table, "empty generator table"
    missing = [(step, t, status) for step, t, status in table if status != "generated"]
    assert not missing, missing


@pytest.mark.parametrize("players,seed,rules", [(3, 11, []), (4, 12, ["plus_expansion"]), (6, 13, [])])
def test_short_wire_walk(players, seed, rules):
    rep = walk(players, seed, rules, 250)
    assert rep["error"] is None, rep["error"]
    assert rep["moves"] == 250 or rep["finished"]


def test_cross_player_policy():
    """The title's policy (1822: no purchases across players) only removes
    train / private / minor purchases from the rules' move set, and a walk
    under the rules' set meets some (privates other players own)."""
    with pytest.raises(ValueError):
        engine_rs.BaseGame.new_titled(
            "1822", {1001 + i: f"Player {i + 1}" for i in range(3)}, None, 5
        ).g1822_legal_moves_json("bogus")
    g = engine_rs.BaseGame.new_titled("1822", {1001 + i: f"Player {i + 1}" for i in range(4)}, None, 21)
    rng = random.Random(21)
    excluded = 0
    for n in range(700):
        if g.finished:
            break
        rules = json.loads(g.g1822_legal_moves_json("rules"))
        title = json.loads(g.g1822_legal_moves_json())
        key = lambda m: json.dumps(m, sort_keys=True)  # noqa: E731
        rules_keys = {key(m) for m in rules}
        assert all(key(m) in rules_keys for m in title)
        dropped = [m for m in rules if key(m) not in {key(t) for t in title}]
        assert all(m["wire"]["type"] in ("buy_train", "acquire_company", "merge") for m in dropped), dropped
        excluded += len(dropped)
        g.process_action({**pick(rules, rng), "id": n + 1})
    assert excluded > 0, "the walk met no cross-player purchase"
