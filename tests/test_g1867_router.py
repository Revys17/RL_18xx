"""1867 auto-router (``engine-rs/src/g1867/router.rs``) on the vendored Ruby
fixtures: at every recorded ``run_routes`` the engine's optimum (asked for
BEFORE the record is applied) is at least the recorded revenue, every
emitted route re-prices identically through the replay pricer from its own
``connections``, and the result carries 18xx.games' run_routes route dicts.

The corpus-scale check (and the Ruby HEAD validation of every run the engine
finds above the record) is ``scripts/g1867_router_check.py``.
"""

import json
from pathlib import Path

import pytest

import engine_rs

from scripts.g1867_router_check import check_game
from tests.title_replay_harness import replay_game, supported_titles

pytestmark = pytest.mark.skipif("1867" not in supported_titles(), reason="engine lacks 1867")

FIXTURES = sorted((Path(__file__).parent / "fixtures" / "1867").glob("*.json"))


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.stem)
def test_router_matches_or_beats_every_recorded_run(path):
    r = check_game(str(path))
    assert r["error"] is None, r["error"]
    assert r["counts"]["below"] == 0, r["below"]
    assert r["counts"]["equal"] > 0
    assert not r["conn_mismatch"], r["conn_mismatch"]
    assert r["inexact"] == 0


def test_route_dict_shape():
    g = engine_rs.BaseGame.new_titled("1867", {1: "a", 2: "b", 3: "c", 4: "d"}, None, 1)
    # no trains yet: nothing runs
    res = json.loads(g.g1867_auto_routes_json("CPR"))
    assert res["routes"] == [] and res["revenue"] == 0 and res["exact"]
    # the first recorded run of a fixture, from the engine
    game = json.load(open(next(p for p in FIXTURES if p.stem == "21268")))
    seen = []

    def on_action(rust, _i, action):
        if action["type"] == "run_routes" and not seen:
            seen.append((action, json.loads(rust.g1867_auto_routes_json(str(action["entity"])))))

    replay_game(game, on_action=on_action)
    action, res = seen[0]
    assert res["revenue"] == sum(r["revenue"] for r in res["routes"]) >= sum(r["revenue"] for r in action["routes"])
    for r in res["routes"]:
        assert set(r) == {"train", "hexes", "connections", "nodes", "revenue"}
        assert len(r["nodes"]) == len(r["hexes"]) == len(r["connections"]) + 1
        assert all(c[0] == h for c, h in zip(r["connections"], r["hexes"]))
