"""The model playing hotseat seats (rl18xx.agent.advisor.autoplay, Advisor.start_move,
POST /api/move), with untrained CPU models."""

import copy
import json
import time
from pathlib import Path

import pytest

from rl18xx.agent.advisor.advisor import Advisor
from rl18xx.agent.advisor.app import create_app
from rl18xx.agent.advisor.autoplay import (
    AutoplayError,
    SiteView,
    bot_position,
    match_train_ids,
    plan_move,
    swap_city_refs,
    swap_city_refs_in_log,
    to_site_action,
)
from rl18xx.agent.advisor.live_game import LiveGame

from .conftest import truncated

CHROME = {"Origin": "chrome-extension://abcdefghijklmnop"}


@pytest.fixture(scope="module")
def advisor(advisor_models):
    return Advisor(advisor_models)


def site_view(live, **overrides) -> SiteView:
    """The site's view when it agrees with the engine on who moves."""
    acting = live.acting()
    if acting["kind"] == "player":
        entity = {"id": live.user_id(acting["id"]), "type": "player"}
    else:
        entity = {"id": acting["id"], "type": acting["kind"]}
    entity["player"] = live.user_id(acting["player_id"])
    body = {"last_id": 0, "entity": entity, "step": "Track", "actions": ["lay_tile", "pass"]}
    body.update(overrides)
    return SiteView.parse(body)


def wait(job, seconds=60):
    deadline = time.time() + seconds
    while job.status == "running" and time.time() < deadline:
        time.sleep(0.05)
    return job


# ------------------------------------------------------------------ the plan
def test_the_site_view_is_checked():
    with pytest.raises(AutoplayError):
        SiteView.parse(None)
    with pytest.raises(AutoplayError):
        SiteView.parse({"entity": {"id": 1, "type": "bank"}})
    view = SiteView.parse(
        {"last_id": "7", "entity": {"id": "PRR", "type": "corporation", "player": 2}, "actions": ["pass"]}
    )
    assert view.last_id == 7 and view.entity_id == "PRR" and view.player == 2 and view.actions == ["pass"]
    assert SiteView.parse({"finished": True}).finished


def test_the_model_searches_where_both_engines_agree(game):
    live = LiveGame(game)  # a corporation of Player 3's lays track
    plan = plan_move(live, site_view(live), seats=[2, 3, 4])
    assert plan.kind == "search" and plan.seat == 3


def test_a_human_seat_or_a_finished_game_is_not_the_models(game):
    live = LiveGame(game)
    with pytest.raises(AutoplayError, match="doesn't play"):
        plan_move(live, site_view(live), seats=[1, 2])
    with pytest.raises(AutoplayError, match="over"):
        plan_move(live, SiteView.parse({"finished": True}), seats=[1, 2, 3, 4])


def test_ruby_waiting_for_a_pass_the_engine_took_is_an_implicit_pass(game):
    """Our 1830 forbids buying another player's private; Ruby still offers it (and
    pass) in a blocking BuyCompany step after the engine has moved on."""
    live = LiveGame(game)
    owner = live.user_id(live.acting()["player_id"])
    site = SiteView.parse(
        {
            "last_id": 267,
            "entity": {"id": "PRR", "type": "corporation", "player": owner},
            "step": "BuyCompany",
            "actions": ["buy_company", "pass"],
        }
    )
    plan = plan_move(live, site, seats=[3])
    assert plan.kind == "pass" and plan.action == {"type": "pass", "entity": "PRR", "entity_type": "corporation"}
    no_pass = SiteView.parse(
        {**vars(site), "entity": {"id": "PRR", "type": "corporation", "player": owner}, "actions": ["buy_company"]}
    )
    with pytest.raises(AutoplayError, match="disagree"):
        plan_move(live, no_pass, seats=[3])


def test_an_action_the_engine_rejected_stops_the_model(game):
    live = LiveGame(game)
    live.error = {"action": 9, "type": "buy_train", "message": "GameError: Not a buyable train"}
    with pytest.raises(AutoplayError, match="undo it"):
        plan_move(live, site_view(live), seats=[3])


# ------------------------------------------------------------- the format
def test_g19_city_references_swap_until_the_hex_gets_a_tile():
    token = {"type": "place_token", "entity": "B&O", "entity_type": "corporation", "city": "G19-0-0", "slot": 0}
    route = {"type": "run_routes", "routes": [{"train": "2-0", "nodes": ["G19-1", "F20-0"]}]}
    assert swap_city_refs(token, ["G19"])["city"] == "G19-0-1"
    assert swap_city_refs(swap_city_refs(token, ["G19"]), ["G19"]) == token
    assert swap_city_refs(route, ["G19"])["routes"][0]["nodes"] == ["G19-0", "F20-0"]
    assert swap_city_refs({**token, "city": "F20-0-0"}, ["G19"])["city"] == "F20-0-0"
    upgraded = {"type": "lay_tile", "entity": "NYNH", "entity_type": "corporation", "hex": "G19", "tile": "54-0"}
    log = swap_city_refs_in_log([token, upgraded, {**token, "city": "54-0-0"}, route])
    assert log[0]["city"] == "G19-0-1" and log[2]["city"] == "54-0-0" and log[3] == route


def test_engine_actions_go_to_the_site_with_site_player_ids(game):
    live = LiveGame(game)
    bid = {"type": "bid", "entity": 2, "entity_type": "player", "company": "CA", "price": 165, "id": 4}
    assert to_site_action(bid, live, []) == {
        "type": "bid",
        "entity": 102,
        "entity_type": "player",
        "company": "CA",
        "price": 165,
    }


# ------------------------------------------------------------ start_move
def test_the_models_move_is_searched_and_given_in_the_sites_format(advisor, game):
    live = LiveGame(game)
    job = wait(advisor.start_move(live, site_view(live, last_id=267), seats=[3], readouts=16))
    assert job.status == "done" and job.kind == "move", job.error
    result = job.result
    assert result["kind"] == "search" and result["system"] == "main" and result["visits"] >= 16
    action = result["action"]
    assert action["entity"] == live.acting()["id"] and action["entity_type"] == "corporation"
    assert "id" not in action and result["seat"] == 3 and result["last_id"] == 267
    assert result["description"] and result["actor"]
    # The same position again: the same job.
    assert advisor.start_move(live, site_view(live, last_id=267), seats=[3], readouts=16) is job


def test_the_auction_is_searched_by_the_auction_system_and_bids_by_site_id(advisor, game):
    live = LiveGame(truncated(game, 0))
    job = wait(advisor.start_move(live, site_view(live, actions=["bid", "pass"]), seats=[1, 2, 3, 4], readouts=8))
    assert job.status == "done", job.error
    result = job.result
    assert result["system"] == "auction"
    assert result["action"]["entity"] == 101 and result["action"]["type"] in ("bid", "pass")


def test_an_implicit_pass_needs_no_search(advisor, game):
    live = LiveGame(game)
    owner = live.user_id(live.acting()["player_id"])
    site = SiteView.parse(
        {"last_id": 267, "entity": {"id": "PRR", "type": "corporation", "player": owner}, "actions": ["pass"]}
    )
    job = advisor.start_move(live, site, seats=[3], readouts=200)
    assert job.status == "done" and job.result["kind"] == "implicit_pass"
    assert job.result["action"] == {"type": "pass", "entity": "PRR", "entity_type": "corporation"}


# ---------------------------------------------------------------- the route
def hotseat(game: dict) -> dict:
    out = copy.deepcopy(game)
    out["id"] = "hs_abcdefgh_1761868800"
    return out


def test_the_move_route(advisor, game):
    client = create_app(advisor, port=5002).test_client()
    live = LiveGame(game)
    site = vars(site_view(live, last_id=267))
    body = {
        "game": game,
        "seats": [3],
        "readouts": 8,
        "site": {**site, "entity": {"id": site["entity_id"], "type": site["entity_type"], "player": site["player"]}},
    }
    assert client.post("/api/move", json=body).status_code == 403  # not from the extension
    body["game"] = hotseat(game)
    response = client.post("/api/move", json=body, headers=CHROME)
    assert response.status_code == 200, response.get_json()
    job = response.get_json()["job"]
    for _ in range(600):
        status = client.post(f"/api/move/{job}", headers=CHROME).get_json()
        if status["status"] != "running":
            break
        time.sleep(0.05)
    assert status["status"] == "done" and status["result"]["action"]["entity_type"] == "corporation"
    body["seats"] = [1]
    response = client.post("/api/move", json=body, headers=CHROME)
    assert response.status_code == 409 and response.get_json()["autoplay"]


# ------------------------------------------------------------ train ids
TRAIN_IDS = Path(__file__).resolve().parents[2] / "fixtures" / "advisor" / "train_ids_game.json"


@pytest.fixture(scope="module")
def train_ids_live():
    """An engine-generated game where NYC buys B&M's 4-2 while B&M also holds 4-1 (the legal list
    keeps one 4 per seller), followed as a site game."""
    return LiveGame(json.loads(TRAIN_IDS.read_text()))


def test_following_a_game_keeps_the_sites_train_ids(train_ids_live):
    rust = train_ids_live.rust
    assert train_ids_live.error is None
    assert rust.train_by_id("4-2").owner.id == "NYC" and rust.train_by_id("4-1").owner.id == "B&M"


def _site(**trains):
    return SiteView.parse(
        {"last_id": 1, "entity": {"id": "B&M", "type": "corporation", "player": 1},
         "trains": {corp: [{"id": t, "name": t.split("-")[0]} for t in ids] for corp, ids in trains.items()},
         "depot": [{"id": "5-0", "name": "5"}]}  # fmt: skip
    )


def test_the_sites_train_ids_go_into_the_models_moves(train_ids_live):
    rust = train_ids_live.rust  # B&M holds 4-1, NYC 4-2
    route = {"type": "run_routes", "entity": "B&M", "entity_type": "corporation",
             "routes": [{"train": "4-1", "hexes": ["F20"], "revenue": 50}]}  # fmt: skip
    # The engines agree: nothing changes.
    assert match_train_ids(route, rust, _site(**{"B&M": ["4-1"], "NYC": ["4-2"]})) == route
    # The site has them the other way round: the route goes out on the site's own 4.
    swapped = match_train_ids(route, rust, _site(**{"B&M": ["4-2"], "NYC": ["4-1"]}))
    assert swapped["routes"][0]["train"] == "4-2" and route["routes"][0]["train"] == "4-1"
    buy = {"type": "buy_train", "entity": "PRR", "entity_type": "corporation", "train": "4-2", "price": 1}
    assert match_train_ids(buy, rust, _site(**{"B&M": ["4-2"], "NYC": ["4-1"]}))["train"] == "4-1"  # NYC's 4
    discard = {"type": "discard_train", "entity": "B&M", "entity_type": "corporation", "train": "4-1"}
    assert match_train_ids(discard, rust, _site(**{"B&M": ["4-2"]}))["train"] == "4-2"
    # No such train on the site: say which, rather than post a move it refuses.
    with pytest.raises(AutoplayError, match="B&M has no 4-train"):
        match_train_ids(route, rust, _site(**{"B&M": [], "NYC": ["4-1", "4-2"]}))
    # An older page that doesn't report trains: the move as it is.
    assert match_train_ids(route, rust, SiteView.parse({"entity": {"id": "B&M", "type": "corporation"}})) == route


# ------------------------------------------------------------ train shuffling
SHUFFLE = Path(__file__).resolve().parents[2] / "fixtures" / "1830" / "train_shuffle_game.json"


def test_the_bots_obey_the_rule_against_train_shuffling_and_the_humans_dont(advisor):
    """At a purchase of a sister corporation's last train in a third OR in a row (BaseGame.shuffle_runs): the
    followed game and the advice offer it at the price played (the real rules); the model's seat isn't
    offered it and doesn't make it; the followed game never gets the rule."""
    from tests.test_train_shuffle_rule import _expected, _first_refused, _offers

    data = json.loads(SHUFFLE.read_text())
    i = _first_refused(data)
    buyer, seller, _ = _expected(data)[i]
    live = LiveGame({"id": "hs_shuffle", "title": "1830", "players": data["players"], "actions": data["actions"][:i]})
    assert live.error is None and live.acting()["id"] == buyer
    assert not live.rust._game.no_buyback_rule
    assert _offers(live.rust.get_factored_choices(), seller, data["actions"][i]["price"])
    assert not _offers(bot_position(live).get_factored_choices(), seller)
    assert not live.rust._game.no_buyback_rule

    site = site_view(live, last_id=i, step="BuyTrain", actions=["buy_train", "pass"])
    job = wait(advisor.start_move(live, site, seats=[live.acting()["player_id"]], readouts=8))
    assert job.status == "done", job.error
    action = job.result["action"]
    if action["type"] == "buy_train":
        assert live.rust.train_by_id(action["train"]).owner.id != seller, action
    assert not live.rust._game.no_buyback_rule
    assert _offers(live.rust.get_factored_choices(), seller, data["actions"][i]["price"])


# ------------------------------------------------------------- kept trees
def test_the_models_next_move_goes_on_with_its_kept_tree(advisor_models, game):
    """The model plays every seat, each move posted to the "site" and followed back as the live bot's
    games are: a move from the position the model's last move led to goes on with that tree (the root
    topped up to the budget), and a position the tree doesn't lead to (another player moved) starts
    a fresh one."""
    advisor = Advisor(advisor_models)
    live = LiveGame(game)
    readouts = 12
    searched = []
    while len(searched) < 6:
        view = site_view(live, last_id=len(game["actions"]))
        job = wait(advisor.start_move(live, view, seats=[1, 2, 3, 4], readouts=readouts))
        assert job.status == "done", job.error
        result = job.result
        if result["kind"] == "search":
            searched.append(result)
            assert result["visits"] >= readouts and 0 <= result["inherited"] <= result["visits"]
        game["actions"].append({**result["action"], "id": len(game["actions"]) + 1, "user": 101})
        assert live.update(game) and live.error is None
    assert searched[0]["inherited"] == 0
    assert any(r["inherited"] > 0 for r in searched[1:]), [r["inherited"] for r in searched]

    # Another player's move in between: the position isn't the kept tree's root.
    kept = advisor._trees[live.game_id]
    assert kept[0] == live.position_key
    other = Advisor(advisor_models)
    other._trees[live.game_id] = ("not this position", kept[1], kept[2])
    view = site_view(live, last_id=len(game["actions"]))
    job = wait(other.start_move(live, view, seats=[1, 2, 3, 4], readouts=readouts))
    assert job.status == "done", job.error
    assert job.result["inherited"] == 0
