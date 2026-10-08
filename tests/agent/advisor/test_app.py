"""The advisor backend's routes (rl18xx.agent.advisor.app), with untrained CPU models."""

import copy
import time

import pytest

from rl18xx.agent.advisor.advisor import Advisor
from rl18xx.agent.advisor.app import create_app

from .conftest import truncated

CHROME = {"Origin": "chrome-extension://abcdefghijklmnop"}
FIREFOX = {"Origin": "moz-extension://0f1e2d3c-4b5a-6978-8796-a5b4c3d2e1f0"}
PORT = 5002


@pytest.fixture(scope="module")
def advisor(advisor_models):
    return Advisor(advisor_models)


@pytest.fixture
def client(advisor):
    return create_app(advisor, port=PORT).test_client()


def advise(client, game, headers=CHROME):
    return client.post("/api/advise", json=game, headers=headers)


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"Origin": "https://18xx.games"},
        {"Origin": "https://evil.example"},
        {"Origin": "null"},
        {"Origin": f"http://127.0.0.1:{PORT}"},  # the backend's own page, only with --debug-page
    ],
)
def test_only_extensions_may_call_the_api(client, game, headers):
    for response in (
        advise(client, game, headers),
        client.post("/api/think", json={"game_id": 123}, headers=headers),
        client.get("/api/think/abc", headers=headers),
        client.get("/api/health", headers=headers),
    ):
        assert response.status_code == 403
        assert "extension" in response.get_json()["error"]


def test_requests_for_another_host_name_are_refused(client):
    response = client.get("/api/health", headers={**CHROME, "Host": "attacker.example:5002"})
    assert response.status_code == 403


@pytest.mark.parametrize("headers", [CHROME, FIREFOX])
def test_extensions_get_answers_with_cors_headers(client, headers):
    response = client.get("/api/health", headers=headers)
    assert response.status_code == 200 and response.get_json()["ok"]
    assert response.headers["Access-Control-Allow-Origin"] == headers["Origin"]
    preflight = client.options(
        "/api/advise",
        headers={**headers, "Access-Control-Request-Method": "POST", "Access-Control-Request-Private-Network": "true"},
    )
    assert preflight.status_code == 200
    assert "POST" in preflight.headers["Access-Control-Allow-Methods"]
    assert preflight.headers["Access-Control-Allow-Private-Network"] == "true"


def test_advise_on_a_game_in_progress(client, game):
    response = advise(client, game)
    assert response.status_code == 200
    advice = response.get_json()
    assert advice["supported"] and advice["started"] and advice["game_id"] == 123
    assert [p["name"] for p in advice["players"]] == ["Player 1", "Player 2", "Player 3", "Player 4"]
    assert advice["acting"]["player"] == "Player 3"
    assert sum(p["probability"] for p in advice["win"]["players"]) == pytest.approx(1.0)
    assert 1 <= len(advice["moves"]) <= 5
    tile = next(m for m in advice["moves"] if m["type"] == "lay_tile")
    assert tile["hex"] and tile["tile"] and isinstance(tile["rotation"], int) and len(tile["map"]) == 2
    assert advice["recent"] and all("probability" in r for r in advice["recent"])
    assert response.headers["Cache-Control"] == "no-store"


def test_unsupported_and_unstarted_games(client, game):
    other = advise(client, {**game, "title": "1867"}).get_json()
    assert other["supported"] is False and "1867" in other["reason"]
    assert [p["name"] for p in other["players"]] == ["Player 1", "Player 2", "Player 3", "Player 4"]
    rules = advise(client, {**game, "settings": {"optional_rules": ["two_player_variant"]}}).get_json()
    assert rules["supported"] is False and "two_player_variant" in rules["reason"]
    new = advise(client, {**game, "status": "new", "actions": []}).get_json()
    assert new["supported"] and new["started"] is False
    assert advise(client, {"id": 1}).status_code == 400
    assert client.post("/api/advise", data="not json", headers=CHROME).status_code == 400


def test_the_game_is_followed_across_requests_and_rebuilt_after_an_undo(client, game):
    count = len(game["actions"]) - 3
    first = advise(client, truncated(game, count)).get_json()
    moved = advise(client, truncated(game, count + 1)).get_json()
    assert moved["position"] != first["position"]
    undone = truncated(game, count + 1)
    undone["actions"].append({"type": "undo", "entity": 101, "entity_type": "player", "id": count + 2})
    back = advise(client, undone).get_json()
    assert back["position"] == first["position"]
    assert [m["index"] for m in back["moves"]] == [m["index"] for m in first["moves"]]


def test_think_starts_a_search_and_reports_it(client, game):
    advice = advise(client, game).get_json()
    assert client.post("/api/think", json={"game_id": 999}, headers=CHROME).status_code == 404
    stale = client.post("/api/think", json={"game_id": 123, "position": "old", "readouts": 8}, headers=CHROME)
    assert stale.status_code == 409
    started = client.post(
        "/api/think", json={"game_id": 123, "position": advice["position"], "readouts": 16}, headers=FIREFOX
    )
    assert started.status_code == 200
    job = started.get_json()["job"]
    busy = client.post("/api/think", json={"game_id": 123, "readouts": 16}, headers=CHROME)
    if busy.status_code == 409:  # unless the first search is already over
        assert busy.get_json()["job"]["job"] == job
    deadline = time.time() + 120
    while True:
        status = client.post(f"/api/think/{job}", json={}, headers=CHROME).get_json()
        if status["status"] != "running" or time.time() > deadline:
            break
        time.sleep(0.05)
    assert status["status"] == "done", status
    assert status["progress"] == 1.0 and status["result"]["moves"]
    assert client.get("/api/think/nope", headers=CHROME).status_code == 404


def test_the_debug_page_is_off_unless_asked_for(advisor, client, game):
    assert client.get("/debug").status_code == 404
    debug = create_app(advisor, port=PORT, debug_page=True).test_client()
    page = debug.get("/debug")
    assert page.status_code == 200 and b"/debug/src/panel.js" in page.data
    assert debug.get("/debug/src/panel.js").status_code == 200
    own = {"Origin": f"http://localhost:{PORT}"}
    assert advise(debug, copy.deepcopy(game), own).status_code == 200
    assert advise(debug, game, {"Origin": "http://localhost:5003"}).status_code == 403
