"""1867 bot seats (rl18xx.agent.advisor.titled): following a site game in the Rust engine, the bot's move as the
site stores it, and the local site's connector playing it -- on a Ruby fixture game (tests/fixtures/1867, already
the replay harness's) with a tiny untrained generic 1867 checkpoint on CPU. That Ruby accepts the moves is checked
live (docs/rl_model.md "1867 bots on the local site"); here our own engine, as the site's game feeds it back, must."""

import copy
import json
from pathlib import Path

import pytest
import torch

from rl18xx.agent.advisor.app import parse_title_policy
from rl18xx.agent.advisor.autoplay import AutoplayError, SiteView
from rl18xx.agent.advisor.bot_hub import LOCAL, PUBLIC, BotHub, CannotMove
from rl18xx.agent.advisor.live_game import support_problem
from rl18xx.agent.advisor.site_bots import LocalSite
from rl18xx.agent.advisor.titled import TitledGame, TitledPolicy, titled_problem

from .conftest import GENERIC_SMALL, truncated
from .test_site_bots import FakeLocal, Idle

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "1867" / "hs_ahjzadkh_19792.json"
HOME_TOKEN_AT = 63  # the fixture's actions before a won minor's home token (Ruby's view names its owner)


@pytest.fixture(scope="module")
def site_game() -> dict:
    """A finished 4-player 1867 game as the site serves it (Ruby's fixture: site user ids, auto actions)."""
    return json.loads(FIXTURE.read_text())


@pytest.fixture(scope="module")
def policy(tmp_path_factory) -> TitledPolicy:
    """A tiny untrained generic 1867 checkpoint laid out as a policy-gradient run (learner_temperature 0.5)."""
    from rl18xx.agent.alphazero import model_generic as mg

    run = tmp_path_factory.mktemp("generic_1867_run")
    (run / "config.json").write_text(json.dumps({"learner_temperature": 0.5}))
    torch.manual_seed(3)
    model = mg.GenericPolicyValueModel(mg.GenericModelConfig(**GENERIC_SMALL)).eval()
    with torch.no_grad():
        for p in model.parameters():
            if p.abs().sum() == 0:
                p.normal_(0, 0.1)
    path = run / "learner" / "1.pth"
    mg.save_checkpoint(model, path, titles=["1867"])
    return TitledPolicy("1867", path, seed=0)


def active(game: dict, count: int) -> dict:
    return {**truncated(game, count), "status": "active"}


def ruby_view(live: TitledGame, game: dict, **overrides) -> dict:
    """The site engine's view (routes/bot.rb ``bot_view``) where it agrees with ours on who moves."""
    kind, entity = live.acting()
    if kind == "player":
        player = ident = live.user_id(entity)
    else:
        ident, owner = entity, live.raw.corporation_by_id(entity).owner_id_str
        player = live.user_id(int(owner.split(":")[1]))
    view = {
        "last_id": game["actions"][-1]["id"] if game["actions"] else 0,
        "finished": False,
        "error": None,
        "entity": {"id": ident, "type": kind, "player": player},
        "step": live.step(),
        "actions": ["pass"],
        "acting": [player],
    }
    view.update(overrides)
    return view


def test_following_a_game_replays_it_as_the_importer_does(site_game):
    from rl18xx.agent.alphazero.generic_import_1867 import replay_export

    live = TitledGame(site_game)
    expected, _, stopped, raw = replay_export(site_game)
    assert stopped is None and live.error is None
    assert live.finished == raw.finished  # (this record ends with the site's end_game: not over by the rules)
    assert [dict(a) for a in live.raw.raw_actions] == expected
    assert live.names == {i + 1: p["name"] for i, p in enumerate(site_game["players"])}
    assert titled_problem(site_game) is None
    assert "1867" in support_problem(site_game)  # the advice stays 1830-only


def test_updates_replay_only_new_actions_and_an_undo_replays_the_game(site_game):
    live = TitledGame(active(site_game, 100))
    raw = live.raw
    assert live.update(active(site_game, 150))
    assert live.raw is raw  # extended in place
    fresh = TitledGame(active(site_game, 150))
    assert live.raw.raw_actions == fresh.raw.raw_actions and live.position_key == fresh.position_key
    assert not live.update(active(site_game, 150))

    undone = active(site_game, 150)
    last = undone["actions"][-1]
    undone["actions"].append({"type": "undo", "entity": last["entity"], "entity_type": last["entity_type"],
                              "id": last["id"] + 1})  # fmt: skip
    assert live.update(undone)
    assert live.raw is not raw
    assert live.raw.raw_actions == TitledGame(undone).raw.raw_actions
    assert len(live.raw.raw_actions) < len(fresh.raw.raw_actions)


def test_an_action_the_engine_refuses_stops_the_replay_before_it(site_game, policy):
    game = active(site_game, 120)
    index = next(i for i, a in enumerate(game["actions"]) if a["type"] == "lay_tile")
    game["actions"][index] = {**game["actions"][index], "hex": "A1"}
    live = TitledGame(game)
    assert live.error is not None and live.error["type"] == "lay_tile"
    assert len(live.raw.raw_actions) == live.error["action"] - 1
    with pytest.raises(AutoplayError, match="could not follow"):
        policy.move(live, SiteView.parse(ruby_view(live, game)), range(1, 5))


@pytest.mark.parametrize("count", [0, 20, HOME_TOKEN_AT, 77, 100, 150, 200])
def test_the_bots_move_is_legal_and_the_site_game_feeds_it_back(site_game, policy, count):
    game = active(site_game, count)
    live = TitledGame(game)
    if count == HOME_TOKEN_AT:  # Ruby's HomeToken view: the Stock Round's player (no actions of its own)
        assert live.acting()[0] == "corporation" and live.step() == "HomeToken"
        bidder = game["actions"][-1]["entity"]
        entity = {"id": bidder, "type": "player", "player": bidder}
        view = ruby_view(live, game, entity=entity, actions=[], acting=[bidder])
    else:
        view = ruby_view(live, game)
    users = [p["id"] for p in game["players"]]
    result = policy.move(live, SiteView.parse(view), range(1, 5))
    action = result["action"]
    assert not {"id", "user", "created_at"} & set(action)
    if action["entity_type"] == "player":
        assert action["entity"] in users
    assert result["temperature"] == 0.5 and result["actor"] and result["description"]
    # The site stores it (numbered, as the bot's user) and serves it back in the game's actions.
    stored = {
        **action,
        "id": (game["actions"][-1]["id"] if game["actions"] else 0) + 1,
        "user": view["entity"]["player"],
    }
    after = copy.deepcopy(game)
    after["actions"].append(stored)
    assert live.update(after) and live.error is None
    logged = dict(live.raw.raw_actions[-1])
    assert {k: v for k, v in logged.items() if k not in ("id", "user")} == {
        **action,
        **({"entity": live.player_mapping[action["entity"]]} if action["entity_type"] == "player" else {}),
    }


def test_the_bot_stops_where_the_engines_disagree_or_its_seat_isnt_a_bot(site_game, policy):
    game = active(site_game, 100)
    live = TitledGame(game)
    view = ruby_view(live, game)
    kind, entity = live.acting()
    seat = live.player_mapping[view["entity"]["player"]]
    with pytest.raises(AutoplayError, match="doesn't play|don't play"):
        policy.move(live, SiteView.parse(view), [s for s in range(1, 5) if s != seat])
    other = next(c for c in ("CNR", "GT", "CPR") if c != entity)
    wrong = dict(view, entity={"id": other, "type": "corporation", "player": view["entity"]["player"]})
    with pytest.raises(AutoplayError, match="engines disagree"):
        policy.move(live, SiteView.parse(wrong), range(1, 5))


@pytest.fixture
def hub(tmp_path, policy):
    hub = BotHub(None, tmp_path / "bots.json", connectors={LOCAL: Idle, PUBLIC: Idle}, titled={"1867": policy})
    yield hub
    hub.stop()


def test_the_local_site_gets_the_1867_bots_move_and_a_refusal_is_never_a_pass(hub, site_game):
    game = active(site_game, 100)
    live = TitledGame(game)
    view = ruby_view(live, game, step="BuyTrain", actions=["buy_train", "pass"])
    bot = view["entity"]["player"]
    site = FakeLocal(game, bots=[bot], view=view)
    hub.settings[LOCAL].update(url=site.url)
    bots = LocalSite(hub, client=site)
    assert bots.tick() == 0.0
    [post] = site.posts
    assert post["user_id"] == bot and post["last_id"] == view["last_id"]
    assert hub.games[(LOCAL, game["id"])].last_move["actor"]
    assert hub.events[-1]["level"] == "move"

    # Ruby refuses a purchase in a step that offers a pass: the bots stop and say so (an 1830 game would pass).
    buy = {
        "type": "buy_train",
        "entity": view["entity"]["id"],
        "entity_type": "corporation",
        "train": "3-0",
        "price": 1,
    }
    hub.titled["1867"] = StubPolicy(buy)
    site.refuse = [(500, "Illegal action")]
    bots.tick()
    assert [p["action"] for p in site.posts[1:]] == [buy]
    state = hub.games[(LOCAL, game["id"])]
    assert state.stuck_at == view["last_id"] and "Illegal action" in state.stuck


class StubPolicy:
    """A titled policy whose every move is ``action``."""

    def __init__(self, action: dict):
        self.action = action

    def move(self, live, site, seats):
        return {"action": self.action, "actor": "CA (Player 2)", "description": "buy train"}


def test_1867_bots_play_on_the_local_site_only(hub, site_game):
    game = active(site_game, 100)
    live = TitledGame(game)
    with pytest.raises(CannotMove, match="local site only"):
        hub.search_move(PUBLIC, game, ruby_view(live, game), {p["id"] for p in game["players"]}, 8)


def test_title_policy_specs():
    assert parse_title_policy("1867=generic:/x/learner/600.pth@0.5") == ("1867", "/x/learner/600.pth", 0.5)
    assert parse_title_policy("1867=/x/learner/600.pth") == ("1867", "/x/learner/600.pth", None)
    with pytest.raises(ValueError):
        parse_title_policy("/x/learner/600.pth")
    with pytest.raises(ValueError, match="1889"):
        TitledPolicy("1889", "/nonexistent.pth")
