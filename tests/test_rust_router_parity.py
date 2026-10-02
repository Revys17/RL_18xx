"""The native router finds the same best revenue as Python's AutoRouter.

Self-play's native RunRoutes decode (``decode.rs`` → ``BaseGame::optimal_routes``
→ ``router::calculate_corp_routes``) picks each corporation's routes itself, and
self-play trains on the revenue it pays — so it must be the revenue-maximising
set of *legal* routes. The oracle is the Python engine's ``AutoRouter`` (a port
of Ruby's auto_router.rb): it walks every route from every reachable node,
validates each with ``Route.revenue()`` (route.rb) and searches the
combinations.

Random native games are replayed in the Python engine, and before every
logged run_routes the AutoRouter's optimum must equal the total the native
router paid. A recorded human game (anonymized; bigger networks and D trains
than random play reaches) is replayed in both engines, and at every run_routes
the native router must match the AutoRouter's optimum recorded in the fixture
and never pay less than the human's own (legal) routes.
"""

import copy
import json
import logging
import random
from pathlib import Path

import pytest

import engine_rs
from rl18xx.game.engine.actions import BaseAction
from rl18xx.game.engine.autorouter import AutoRouter
from rl18xx.game.gamemap import GameMap

logging.disable(logging.CRITICAL)

MAX_MOVES = 3000
HUMAN_FIXTURE = Path(__file__).parent / "fixtures" / "1830" / "router_d_trains.json"

# (seed, players, auction_unlock): cheap-to-check games that reach phase D.
CASES = [
    (0, 4, False),
    (1, 3, True),
    (3, 2, True),
    (5, 4, True),
    (7, 5, True),
    (10, 4, False),
    (14, 6, False),
]


def _players(n):
    return {i: f"Player {i}" for i in range(1, n + 1)}


def _python_optimum(py_game, corp_sym):
    """The AutoRouter's best total revenue for ``corp_sym`` right now."""
    timeouts = []
    best = AutoRouter(py_game, flash=timeouts.append).compute(py_game.corporation_by_id(corp_sym))
    assert not timeouts, f"AutoRouter timed out ({timeouts}); its answer may not be optimal"
    return sum(route.revenue() for route in best if route is not None)


@pytest.mark.parametrize("seed,num_players,auction_unlock", CASES)
def test_native_router_matches_autorouter(seed, num_players, auction_unlock):
    players = _players(num_players)
    rng = random.Random(seed)
    game = engine_rs.BaseGame(players)
    game.set_auction_unlock(auction_unlock)
    for _ in range(MAX_MOVES):
        if game.finished:
            break
        idx = int(rng.choice(game.factored_legal_indices()))
        price_range = game.price_range_for_index(idx)
        price = None
        if price_range is not None and price_range[0] != price_range[1]:
            price = rng.randint(price_range[0], price_range[1])
        game.apply_action_index(idx, price)

    py_game = GameMap().game_by_title("1830")(players)
    py_game.auction_unlock = auction_unlock
    checked = 0
    for move, action in enumerate(game.raw_actions, start=1):
        if action["type"] == "run_routes":
            paid = sum(route["revenue"] for route in action["routes"])
            best = _python_optimum(py_game, action["entity"])
            assert paid == best, f"move {move}: native router paid {paid}, AutoRouter's best is {best}: {action}"
            checked += 1
        py_game.process_action(BaseAction.action_from_dict(copy.deepcopy(action), py_game))
    assert checked > 50, "too few run_routes to mean anything"


def _native_run_routes(rust_game):
    """The run_routes the native decode would log here (applied to a clone)."""
    clone = rust_game.pickle_clone()
    idx = next(
        i for i in clone.factored_legal_indices() if clone.decode_index_to_map(int(i), None)["type"] == "run_routes"
    )
    clone.apply_action_index(int(idx), None)
    return clone.raw_actions[-1]


def _recomputed_revenues(run_routes, py_game):
    """Each route of a logged run_routes rebuilt in Python with its revenue
    stripped; Python's validation raises a GameError for an illegal route."""
    stripped = copy.deepcopy(run_routes)
    stripped["routes"] = [{k: v for k, v in r.items() if k != "revenue"} for r in run_routes["routes"]]
    return [route.revenue() for route in BaseAction.action_from_dict(stripped, py_game).routes]


def test_native_router_on_a_human_game():
    fixture = json.loads(HUMAN_FIXTURE.read_text())
    players = {p["id"]: p["name"] for p in fixture["players"]}
    rules = fixture["settings"]["optional_rules"]
    rust_game = engine_rs.BaseGame(players, optional_rules=rules)
    py_game = GameMap().game_by_title("1830")(players, optional_rules=rules)
    expected = iter(fixture["autorouter_revenue"])
    checked = 0
    for move, action in enumerate(fixture["actions"], start=1):
        if action["type"] == "run_routes":
            native = _native_run_routes(rust_game)
            paid = [route["revenue"] for route in native["routes"]]
            best = next(expected)
            human = sum(route["revenue"] for route in action["routes"])
            assert sum(paid) == best, f"move {move}: native router paid {sum(paid)}, AutoRouter's best is {best}"
            assert sum(paid) >= human, f"move {move}: native router paid {sum(paid)}, the human ran {human}"
            assert _recomputed_revenues(native, py_game) == paid, f"move {move}: {native}"
            checked += 1
        rust_game.process_action(copy.deepcopy(action))
        py_game.process_action(BaseAction.action_from_dict(copy.deepcopy(action), py_game))
    assert next(expected, None) is None, "fixture has more run_routes optima than run_routes"
    assert checked > 50
