"""Auto-routing a corporation that runs two D-trains on a dense late-game map.

Cleaning 18xx.games game 268548 spun at 100% CPU for hours on the router that
predates 0c6889c. At PRR's ``run_routes`` (two D-trains) the importer asks the
action helper for every legal action, including the auto-routed ``RunRoutes``.
That router enumerated ~75k candidate routes per D-train (each walked from
both ends) and tried every pairing: ~5.6e9 combinations. The current router
(engine-rs/src/router.rs) keeps one walk per route and combines them by branch
and bound. The fixture is that game, anonymized and truncated right after
PRR's ``run_routes`` (see its ``description``).

The guard is a subprocess timeout, not ``signal.alarm``: the hang is inside a
single Rust call that never returns to the interpreter, so a Python signal
handler would never get to run.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).parent.parent
FIXTURE = REPO / "tests" / "fixtures" / "1830" / "router_two_d_trains.json"
# Seconds after the fix (most of it imports); hours before it.
TIMEOUT_S = 120

_CLEAN_AND_ROUTE = r"""
import json, logging, sys
logging.disable(logging.CRITICAL)
from rl18xx.agent.alphazero.pretraining import (
    _get_game_object_for_game_with_reason,
    _load_cleaned_game_via_rust,
)

with open(sys.argv[1]) as f:
    game = json.load(f)
cleaned, reason = _get_game_object_for_game_with_reason(game)
result = {"reason": reason}
if cleaned is not None:
    record = cleaned.to_dict()
    result["n_actions"] = len(record["actions"])
    # Back up to just before PRR's run_routes and auto-route it directly.
    before = _load_cleaned_game_via_rust({**record, "actions": record["actions"][:-1]})
    routes, revenue = before.auto_routes_for(before.current_entity)
    result.update(entity=before.current_entity.id, revenue=revenue, routes=routes)
print(json.dumps(result))
"""


@pytest.fixture(scope="module")
def outcome():
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(REPO), env.get("PYTHONPATH")]))
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _CLEAN_AND_ROUTE, str(FIXTURE)],
            cwd=REPO,
            env=env,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(f"cleaning + auto-routing the two-D-train fixture took over {TIMEOUT_S}s")
    assert proc.returncode == 0, proc.stderr[-4000:]
    return json.loads(proc.stdout.splitlines()[-1])


def _human_routes():
    with open(FIXTURE) as f:
        run_routes = json.load(f)["actions"][-1]
    assert run_routes["type"] == "run_routes" and run_routes["entity"] == "PRR"
    return run_routes["routes"]


def _hex_pairs(route: dict) -> set:
    """Adjacent hex pairs (hexsides) a Rust route dict's chains traverse."""
    pairs = set()
    for chain in route["connections"].split("|"):
        hexes = chain.split(",")
        pairs.update(tuple(sorted(pair)) for pair in zip(hexes, hexes[1:]))
    return pairs


def test_cleaner_finishes_the_game(outcome):
    assert outcome["reason"] is None
    assert outcome["n_actions"] == 688


def test_router_runs_both_d_trains(outcome):
    assert outcome["entity"] == "PRR"
    routes = outcome["routes"]
    assert len(routes) == 2
    assert sum(int(r["revenue"]) for r in routes) == outcome["revenue"]
    assert not _hex_pairs(routes[0]) & _hex_pairs(routes[1])
    # 590 + 290, which Python's Route.revenue() accepts. The players ran
    # 600 + 260 (G17 on the first D-train instead of the second).
    assert outcome["revenue"] == 880
    assert sum(r["revenue"] for r in _human_routes()) == 860
