"""The bots: the hub (rl18xx.agent.advisor.bot_hub), the local site's connector (site_bots), the
18xx.games connector (public_bots) and the admin page (admin), against fakes of the sites, with
untrained CPU models."""

import base64
import copy
import json

import pytest

from rl18xx.agent.advisor import admin
from rl18xx.agent.advisor.advisor import Advisor, SearchJob
from rl18xx.agent.advisor.app import create_app
from rl18xx.agent.advisor.bot_hub import LOCAL, PUBLIC, BotHub, Connector
from rl18xx.agent.advisor.live_game import LiveGame
from rl18xx.agent.advisor.public_bots import NoAnswer, Pacer, PublicSite, Refused, take_join_requests
from rl18xx.agent.advisor.site_bots import LocalSite, SiteError

from .conftest import truncated


@pytest.fixture(scope="module")
def advisor(advisor_models):
    return Advisor(advisor_models)


class Idle(Connector):
    def tick(self):
        return 1.0


@pytest.fixture
def hub(advisor, tmp_path):
    hub = BotHub(advisor, tmp_path / "bots.json", connectors={LOCAL: Idle, PUBLIC: Idle})
    yield hub
    hub.stop()


def engine_view(game: dict, **overrides) -> dict:
    """The site engine's view (routes/bot.rb ``bot_view``) when it agrees with ours on who moves."""
    live = LiveGame(game)
    acting = live.acting()
    if acting["kind"] == "player":
        entity = {"id": live.user_id(acting["id"]), "type": "player"}
    else:
        entity = {"id": acting["id"], "type": acting["kind"]}
    entity["player"] = live.user_id(acting["player_id"])
    view = {
        "last_id": game["actions"][-1]["id"] if game["actions"] else 0,
        "finished": False,
        "error": None,
        "entity": entity,
        "step": "Track",
        "actions": ["lay_tile", "pass"],
        "acting": [entity["player"]],
    }
    view.update(overrides)
    return view


class StubAdvisor:
    """An advisor whose every move is ``action``."""

    def __init__(self, action: dict):
        self.action = action

    def start_move(self, live, site, seats, readouts):
        result = {"action": self.action, "actor": "B&M (Player 3)", "description": "Buy a train"}
        return SearchJob("stub", live.game_id, None, readouts, status="done", result=result, kind="move")

    def running_search(self):
        return None


# ------------------------------------------------------------- the local site
class FakeLocal:
    """The local site's bot API over one game: ``refuse`` answers each post with the next (status, message)."""

    url = "http://site.test"

    def __init__(self, game: dict, bots, view: dict):
        self.game_data, self.bots, self.view = game, list(bots), view
        self.posts = []
        self.refuse = []
        self.listed = True

    def games(self):
        if not self.listed:
            return []
        listing = {k: v for k, v in self.game_data.items() if k != "actions"}
        return [{**listing, "status": "active", "acting": self.view["acting"], "bots": self.bots}]

    def game(self, game_id):
        assert game_id == self.game_data["id"]
        return {"game": copy.deepcopy(self.game_data), "bots": self.bots, "view": copy.deepcopy(self.view)}

    def post(self, game_id, user_id, last_id, action):
        self.posts.append({"user_id": user_id, "last_id": last_id, "action": action})
        if self.refuse:
            raise SiteError(*self.refuse.pop(0))
        return {}


def local(hub, site) -> LocalSite:
    hub.settings[LOCAL].update(url=site.url, readouts=8)
    return LocalSite(hub, client=site)


def test_a_bots_turn_gets_the_models_move_posted_as_that_bot(hub, game):
    view = engine_view(game)  # a corporation of Player 3 (site id 103) lays track
    site = FakeLocal(game, bots=[102, 103, 104], view=view)
    assert local(hub, site).tick() == 0.0
    [post] = site.posts
    assert post["user_id"] == 103 and post["last_id"] == 267
    assert post["action"]["entity"] == view["entity"]["id"] and post["action"]["entity_type"] == "corporation"
    state = hub.games[(LOCAL, game["id"])]
    assert state.status == "active" and state.bot_turn and state.last_move["actor"]
    assert [p["bot"] for p in state.players] == [False, True, True, True]
    assert hub.events[-1]["level"] == "move"


def test_a_humans_turn_is_left_alone(hub, game):
    site = FakeLocal(game, bots=[101, 102], view=engine_view(game))
    assert local(hub, site).tick() > 0 and site.posts == []
    state = hub.games[(LOCAL, game["id"])]
    assert not state.bot_turn and state.waiting_on == "Player 3"


def test_the_auction_bids_as_the_bot_whose_turn_it_is(hub, game):
    start = truncated(game, 0)
    view = engine_view(start, step="WaterfallAuction", actions=["bid", "pass"])
    site = FakeLocal(start, bots=[101, 102, 103, 104], view=view)
    assert local(hub, site).tick() == 0.0
    [post] = site.posts
    assert post["user_id"] == 101 and post["last_id"] == 0 and post["action"]["entity"] == 101


def test_a_refused_move_leaves_the_game_until_it_changes_or_a_retry(hub, game):
    site = FakeLocal(game, bots=[103], view=engine_view(game))
    site.refuse = [(500, "Illegal action: no"), (500, "Illegal action: no")]
    bots = local(hub, site)
    bots.tick()
    state = hub.games[(LOCAL, game["id"])]
    assert len(site.posts) == 1 and state.stuck_at == 267 and "Illegal action" in state.stuck
    bots.tick()
    assert len(site.posts) == 1  # the same position: not tried again
    hub.retry(LOCAL, game["id"])  # the admin page
    bots.tick()
    assert len(site.posts) == 2 and state.stuck_at == 267
    site.view["last_id"] = 268  # a human's undo, say
    bots.tick()
    assert len(site.posts) == 3 and state.stuck is None


def test_a_game_that_moved_on_is_looked_at_again(hub, game):
    site = FakeLocal(game, bots=[103], view=engine_view(game))
    site.refuse = [(409, "The game has moved on")]
    local(hub, site).tick()
    assert hub.games[(LOCAL, game["id"])].stuck is None


def test_ruby_offering_a_purchase_our_1830_skips_gets_a_pass(hub, game):
    view = engine_view(game, step="BuyTrain", actions=["buy_train", "pass"])
    site = FakeLocal(game, bots=[103], view=view)
    site.refuse = [(500, "Illegal action: wrong step")]
    buy = {"type": "buy_train", "entity": view["entity"]["id"], "entity_type": "corporation", "train": "3-0"}
    hub.advisor = StubAdvisor(buy)
    local(hub, site).tick()
    first, second = site.posts
    entity = view["entity"]
    assert first["action"] == buy
    assert second["action"] == {"type": "pass", "entity": entity["id"], "entity_type": entity["type"]}


def test_a_game_the_model_cant_play_is_left_alone(hub, game):
    other = {**copy.deepcopy(game), "title": "1889"}
    site = FakeLocal(other, bots=[103], view=engine_view(game))
    local(hub, site).tick()
    assert site.posts == [] and "1889" in hub.games[(LOCAL, game["id"])].stuck


def test_a_game_that_leaves_the_list_is_finished(hub, game):
    site = FakeLocal(game, bots=[101], view=engine_view(game))
    bots = local(hub, site)
    bots.tick()
    site.listed = False
    bots.tick()
    assert hub.games[(LOCAL, game["id"])].status == "finished"


# ---------------------------------------------------------------- 18xx.games
class FakePublic:
    """18xx.games' API over one game, and the local engine's answers about it."""

    url = "https://18xx.test"

    def __init__(self, game: dict, view: dict, status="active"):
        self.game_data, self.view, self.status = game, view, status
        self.pacer = Pacer(lambda: 0.0)
        self.requests = []
        self.posts = []
        self.refuse = []  # exceptions for the next posts
        self.engine_error = None
        self.auto_actions = None

    # the site
    def games(self, token):
        self.requests.append(("games", token))
        listing = {k: v for k, v in self.game_data.items() if k != "actions"}
        return [{**listing, "status": self.status, "acting": self.view["acting"]}]

    def game(self, game_id):
        self.requests.append(("game", game_id))
        return {**copy.deepcopy(self.game_data), "status": self.status}

    def join(self, game_id, token):
        self.requests.append(("join", token))
        name = token[len("token-") :].replace("-", " ")  # as public() writes the token files
        self.game_data["players"].append({"id": 900 + len(self.game_data["players"]), "name": name})
        return self.game(game_id)

    def post(self, game_id, token, move):
        self.posts.append((token, move))
        if self.refuse:
            raise self.refuse.pop(0)
        return {"status": "active", "acting": [101]}

    # the local engine
    def engine(self, game, action=None):
        out = {"view": copy.deepcopy(self.view)}
        if action is not None:
            if self.engine_error:
                out["error"] = self.engine_error
            else:
                out["action"] = {**action, "id": self.view["last_id"] + 1}
                if self.auto_actions:
                    out["action"]["auto_actions"] = self.auto_actions
        return out


def public(hub, site, tmp_path, accounts=("Player 3", "Player 4")) -> PublicSite:
    directory = tmp_path / "accounts"
    directory.mkdir(exist_ok=True)
    for name in accounts:
        (directory / f"{name}.token").write_text(f"token-{name.replace(' ', '-')}\n")
    hub.settings[PUBLIC].update(url=site.url, accounts_dir=str(directory), join_file=str(tmp_path / "join"), readouts=8)
    return PublicSite(hub, client=site, engine=site)


def test_a_bots_turn_on_18xx_games_is_posted_as_the_sites_page_posts_it(hub, game, tmp_path):
    view = engine_view(game)
    site = FakePublic(game, view)
    site.auto_actions = [{"type": "pass", "entity": 104, "entity_type": "player"}]
    bots = public(hub, site, tmp_path)
    assert bots.tick() == 0.0
    [(token, move)] = site.posts
    assert token == "token-Player-3"  # the account of the corporation's president
    assert move["id"] == 268 and move["entity"] == view["entity"]["id"] and "user" not in move
    assert move["auto_actions"] == site.auto_actions
    state = hub.games[(PUBLIC, game["id"])]
    assert [p["bot"] for p in state.players] == [False, False, True, True] and state.last_move
    assert [r[0] for r in site.requests] == ["games", "games", "game"]  # a list per account, then the game
    assert not bots.followed[game["id"]]["acting"] == view["acting"]  # the post's answer: a human's turn now


def test_a_move_that_got_no_answer_is_not_resent_and_the_site_is_left_alone(hub, game, tmp_path, monkeypatch):
    monkeypatch.setattr("rl18xx.agent.advisor.public_bots.LOST_POST_WAIT", 0)
    site = FakePublic(game, engine_view(game))
    site.refuse = [NoAnswer("timed out")]
    bots = public(hub, site, tmp_path)
    assert bots.tick() == 5.0
    assert len(site.posts) == 1 and site.pacer.counts()["paused_until"]
    assert any("got no answer" in e["message"] for e in hub.events)
    with pytest.raises(NoAnswer):  # every request waits out the pause
        site.pacer.turn()


def test_a_move_the_engine_refuses_in_a_purchase_step_our_1830_skips_is_a_pass(hub, game, tmp_path):
    view = engine_view(game, step="BuyTrain", actions=["buy_train", "pass"])
    site = FakePublic(game, view)
    buy = {"type": "buy_train", "entity": view["entity"]["id"], "entity_type": "corporation", "train": "3-0"}
    hub.advisor = StubAdvisor(buy)
    site.refuse = [Refused(500, "Illegal action: no")]
    public(hub, site, tmp_path).tick()
    (_, first), (_, second) = site.posts
    assert first["type"] == "buy_train" and second["type"] == "pass" and second["id"] == 268


def test_a_move_the_engine_refuses_elsewhere_leaves_the_game(hub, game, tmp_path):
    site = FakePublic(game, engine_view(game))
    site.engine_error = "GameError: no"
    public(hub, site, tmp_path).tick()
    assert site.posts == [] and "engine refuses" in hub.games[(PUBLIC, game["id"])].stuck


def test_the_bots_join_a_new_game(hub, game, tmp_path):
    new = {**copy.deepcopy(game), "actions": [], "players": [{"id": 1, "name": "Human"}]}
    site = FakePublic(new, engine_view(game), status="new")
    bots = public(hub, site, tmp_path, accounts=("Bot A", "Bot B", "Bot C"))
    bots.join(new["id"], 2)
    (tmp_path / "join").write_text(f"{new['id']}\n")
    bots.tick()
    joins = [r for r in site.requests if r[0] == "join"]
    assert joins == [("join", "token-Bot-A"), ("join", "token-Bot-B"), ("join", "token-Bot-C")]
    assert hub.games[(PUBLIC, new["id"])].status == "new" and site.posts == []


def test_the_pacer_spaces_requests_and_pauses_longer_each_time_the_site_stays_silent():
    now = [1000.0]
    slept = []

    def sleep(s):
        slept.append(s)
        now[0] += s

    pacer = Pacer(lambda: 2.0, clock=lambda: now[0], sleep=sleep)
    pacer.turn()
    pacer.turn()
    assert slept == [2.0] and pacer.counts()["last_minute"] == 2
    assert pacer.failed() == 120.0
    with pytest.raises(NoAnswer):
        pacer.turn()
    now[0] += 121
    pacer.turn()
    assert pacer.failed() == 240.0  # still silent: twice as long
    now[0] += 241
    pacer.turn()
    pacer.answered()
    assert pacer.failed() == 120.0


def test_the_join_file_is_read_once_and_emptied(tmp_path):
    path = tmp_path / "join"
    path.write_text("274001\n 274002 junk\n")
    assert take_join_requests(path) == ["274001", "274002"]
    assert take_join_requests(path) == [] and take_join_requests(tmp_path / "missing") == []


# ----------------------------------------------------------- the admin page
def test_the_admin_page_needs_the_password_and_switches_sites(hub, tmp_path):
    password = admin.read_password(tmp_path / "pw")
    assert oct((tmp_path / "pw").stat().st_mode & 0o777) == "0o600"
    app = create_app(hub.advisor, port=5004)
    admin.register(app, hub, password)
    client = app.test_client()
    auth = {"Authorization": "Basic " + base64.b64encode(f"admin:{password}".encode()).decode()}
    assert client.get("/admin").status_code == 401
    assert (
        client.get("/admin", headers={"Authorization": "Basic " + base64.b64encode(b"admin:x").decode()}).status_code
        == 401
    )
    assert client.get("/admin", headers=auth).status_code == 200
    state = client.get("/admin/api/state", headers=auth).get_json()
    assert set(state["sites"]) == {LOCAL, PUBLIC} and state["sites"][PUBLIC]["settings"]["enabled"] is False
    # A change needs the X-Admin header (no other site's page can send it here).
    assert client.post(f"/admin/api/site/{PUBLIC}", json={"enabled": True}, headers=auth).status_code == 403
    changed = client.post(f"/admin/api/site/{PUBLIC}", json={"enabled": True}, headers={**auth, "X-Admin": "1"})
    assert changed.status_code == 200 and hub.connectors[PUBLIC].running
    assert json.loads(hub.settings_path.read_text())[PUBLIC]["enabled"] is True  # kept for the next start
    bad = client.post(f"/admin/api/site/{PUBLIC}", json={"min_interval": 0.2}, headers={**auth, "X-Admin": "1"})
    assert bad.status_code == 400 and hub.settings[PUBLIC]["min_interval"] == 2.0
    assert (
        client.post(f"/admin/api/site/{PUBLIC}", json={"url": "x"}, headers={**auth, "X-Admin": "1"}).status_code == 400
    )
    client.post(f"/admin/api/site/{PUBLIC}", json={"enabled": False}, headers={**auth, "X-Admin": "1"})
    assert not hub.connectors[PUBLIC].running
    # The extension's API keeps its own rule: only the extension's pages.
    assert client.post("/api/health").status_code == 403
