"""The rule against train shuffling (``BaseGame.no_buyback``, the project's, not Ruby's).

A corporation whose last train another corporation bought may not, on its next turn, buy a train from
any corporation: it buys from the depot or goes bankrupt. A self-play rule: off unless set
(``set_no_buyback_rule``; ``start_positions.new_game`` sets it), so human games import and the advisor
follows them under the real rules. Without it two corporations of one president pass
one train back and forth at $1 and neither ever buys a train or goes bankrupt: 76% of lr1's generator
games did (40,480 buybacks in 4,096 games). Blocking only the buyback, or only another corporation's last
train, still lets three corporations rotate trains forever (85% of 13,056 lr1 games had purchases only
the strict rule refuses). The fixtures are lr1 games: 174 buybacks, and three corporations rotating one
train 73 times. The rule is on the legal-move lists only, so the games still replay as played.
"""

import json
import logging
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
logging.disable(logging.CRITICAL)

from engine_rs import BaseGame as RustGame  # noqa: E402
from rl18xx.game.action_helper import ActionHelper  # noqa: E402
from rl18xx.game.factored_action_helper import FactoredActionHelper  # noqa: E402
from rl18xx.game.gamemap import GameMap  # noqa: E402
from rl18xx.rust_adapter import RustGameAdapter  # noqa: E402

from tests.test_factored_action_helper_rust_parity import _canonical_key  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures" / "1830"


@pytest.fixture(scope="module")
def shuffle_game():
    return json.loads((FIXTURES / "train_shuffle_game.json").read_text())


@pytest.fixture(scope="module")
def rotation_game():
    return json.loads((FIXTURES / "train_rotation_game.json").read_text())


@pytest.fixture(scope="module", params=["shuffle_game", "rotation_game"])
def any_game(request):
    return request.getfixturevalue(request.param)


def _players(data):
    return {p["id"]: p["name"] for p in data["players"]}


def _rust(data, rule=True):
    game = RustGame(_players(data))
    game.set_auction_unlock(data["auction_unlock"])
    game.set_no_buyback_rule(rule)
    return RustGameAdapter(game)


def _python(data, rule=True):
    game = GameMap().game_by_title("1830")(_players(data))
    game.auction_unlock = data["auction_unlock"]
    game.no_buyback_rule = rule
    return game


def _seller(game, action):
    """The corporation that owns the train ``action`` buys, or None (the depot)."""
    for corp in game.corporations:
        if any(t.id == action["train"] for t in corp.trains):
            return corp.id
    return None


def _cross_corp_buys(data):
    """Index -> seller of every purchase from another corporation, replayed in the Rust engine."""
    game = _rust(data)
    out = {}
    for i, action in enumerate(data["actions"]):
        if action["type"] == "buy_train" and not action.get("exchange"):
            seller = _seller(game, action)
            if seller is not None and seller != action["entity"]:
                out[i] = seller
        game.process_action(dict(action))
    return out


def _offers(choices, seller, price=None):
    return any(
        c.type == "BuyTrain"
        and c.entity.get("source") == seller
        and (price is None or c.price_range[0] <= price <= c.price_range[1])
        for c in choices
    )


def test_the_game_replays_as_played(any_game):
    game = _rust(any_game)
    for action in any_game["actions"]:
        game.process_action(dict(action))
    assert game.finished


def _refused_and_offered(data):
    """Replays ``data`` with the rule on and checks every purchase from another corporation: refused, with
    no purchase from any corporation offered, when another corporation bought the buyer's last train before
    this turn; otherwise offered as played. Returns the refused indices and the number offered."""
    buys = _cross_corp_buys(data)
    game = _rust(data)
    corporations = {c.id for c in game.corporations}
    refused, offered = set(), 0
    for i, action in enumerate(data["actions"]):
        if i in buys:
            choices = game.get_factored_choices()
            if any(blocked == action["entity"] for blocked, _, _ in game.no_buyback):
                assert not any(
                    c.type == "BuyTrain" and c.entity.get("source") in corporations for c in choices
                ), (i, action)
                assert game.train_buyback_blocked(action["entity"], buys[i])
                refused.add(i)
            else:
                assert _offers(choices, buys[i], action["price"]), (i, action, game.no_buyback)
                offered += 1
        game.process_action(dict(action))
    assert refused and offered
    return refused, offered


def test_every_buyback_is_refused(shuffle_game):
    refused, _ = _refused_and_offered(shuffle_game)
    assert set(shuffle_game["buyback_indices"]) == refused and len(refused) == 174


def test_a_rotation_through_a_third_corporation_is_refused(rotation_game):
    """No corporation in the rotation buys back from the corporation that bought its train, so a buyback-only
    rule never fires; each takes a third corporation's last train, which the rule refuses too."""
    refused, _ = _refused_and_offered(rotation_game)
    assert set(rotation_game["rotation_indices"]) <= refused and len(rotation_game["rotation_indices"]) == 73


def test_both_engines_agree_through_the_game(any_game):
    """The Python engine keeps the same blocks (bookkeeping on each sale, arming at each turn start) and
    both legal-move lists agree at every position of the game -- with every block in play."""
    rust = _rust(any_game)
    python = _python(any_game)
    helper = FactoredActionHelper()
    for i, action in enumerate(any_game["actions"]):
        assert sorted(rust.no_buyback) == sorted(python.no_buyback), i
        r = {_canonical_key(c) for c in rust.get_factored_choices()}
        p = {_canonical_key(c) for c in helper.get_choices(python)}
        assert r == p, (i, action, r ^ p)
        rust.process_action(dict(action))
        python.process_action(dict(action))


def test_action_helper_refuses_a_buyback(shuffle_game):
    """The non-factored ActionHelper (Python search, the importer's matching) applies the rule too."""
    i = shuffle_game["buyback_indices"][0]
    python = _python(shuffle_game)
    for action in shuffle_game["actions"][:i]:
        python.process_action(dict(action))
    buyback = shuffle_game["actions"][i]
    seller = _seller(python, buyback)
    entity = python.current_entity
    assert any(getattr(t.owner, "id", None) == seller for t in python.round.active_step().buyable_trains(entity))
    actions = ActionHelper().get_buy_train_actions(python)
    assert not any(type(a).__name__ == "BuyTrain" and getattr(a.train.owner, "id", None) == seller for a in actions)
    assert python.train_buyback_blocked(buyback["entity"], seller)


def test_off_by_default_and_on_in_generator_games(shuffle_game):
    """Human-game import and the advisor build engines directly: the real rules, buybacks offered."""
    from rl18xx.agent.alphazero.start_positions import new_game

    assert not RustGame(_players(shuffle_game)).no_buyback_rule
    assert not GameMap().game_by_title("1830")(_players(shuffle_game)).no_buyback_rule
    assert new_game(4)._game.no_buyback_rule and not new_game(4, no_buyback_rule=False)._game.no_buyback_rule
    i = shuffle_game["buyback_indices"][0]
    for game in (_rust(shuffle_game, rule=False), _python(shuffle_game, rule=False)):
        for action in shuffle_game["actions"][:i]:
            game.process_action(dict(action))
        seller = _seller(game, shuffle_game["actions"][i])
        assert not game.train_buyback_blocked(shuffle_game["actions"][i]["entity"], seller)
    rust = _rust(shuffle_game, rule=False)
    for action in shuffle_game["actions"][:i]:
        rust.process_action(dict(action))
    assert _offers(rust.get_factored_choices(), _seller(rust, shuffle_game["actions"][i]), 1)
