"""The rule against train shuffling (``BaseGame.no_buyback``, the project's, not Ruby's).

A corporation whose last train another corporation bought may not, on its next turn, buy a train back
from that buyer if the buyer would be left trainless. A self-play rule: off unless set
(``set_no_buyback_rule``; ``start_positions.new_game`` sets it), so human games import and the advisor
follows them under the real rules. Without it two corporations of one president pass
one train back and forth at $1 and neither ever buys a train or goes bankrupt: 76% of lr1's generator
games did (40,480 buybacks in 4,096 games). The fixture is the worst of them, 174 buybacks; the rule is
on the legal-move lists only, so the game still replays as played.
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

FIXTURE = Path(__file__).parent / "fixtures" / "1830" / "train_shuffle_game.json"


@pytest.fixture(scope="module")
def shuffle_game():
    return json.loads(FIXTURE.read_text())


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


def test_the_game_replays_as_played(shuffle_game):
    game = _rust(shuffle_game)
    for action in shuffle_game["actions"]:
        game.process_action(dict(action))
    assert game.finished


def test_every_buyback_is_refused_and_every_other_purchase_still_offered(shuffle_game):
    buys = _cross_corp_buys(shuffle_game)
    buybacks = set(shuffle_game["buyback_indices"])
    assert buybacks <= set(buys) and len(buybacks) == 174
    game = _rust(shuffle_game)
    refused, offered = 0, 0
    for i, action in enumerate(shuffle_game["actions"]):
        if i in buys:
            choices = game.get_factored_choices()
            if i in buybacks:
                assert not _offers(choices, buys[i]), (i, action)
                assert game.train_buyback_blocked(action["entity"], buys[i])
                refused += 1
            else:
                assert _offers(choices, buys[i], action["price"]), (i, action, game.no_buyback)
                offered += 1
        game.process_action(dict(action))
    assert refused == 174 and offered == len(buys) - 174


def test_both_engines_agree_through_the_game(shuffle_game):
    """The Python engine keeps the same blocks (bookkeeping on each sale, arming at each turn start) and
    both legal-move lists agree at every position of the game -- with every block in play."""
    rust = _rust(shuffle_game)
    python = _python(shuffle_game)
    helper = FactoredActionHelper()
    for i, action in enumerate(shuffle_game["actions"]):
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
