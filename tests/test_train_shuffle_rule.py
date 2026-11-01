"""The rule against train shuffling (``BaseGame.shuffle_runs``, the project's, not Ruby's).

One president's corporations may buy each other's last train in at most two ORs in a row
(``SHUFFLE_MAX_ORS``); in the next OR a corporation that needs a train buys it from the depot or goes
bankrupt. Purchases within an OR and of spare trains are never limited. A self-play rule: off unless set
(``set_no_buyback_rule``; ``start_positions.new_game`` sets it), so human games import and the advisor
follows them under the real rules. Without it corporations of one president passed trains around at $1 and
kept lost positions alive: every bot system ran such purchases 3+ ORs in a row in 65-77% of its games, humans
in 6.9% (lr1's ladder games vs 3,612 human games). The fixtures are lr1 games: two corporations passing one
train back and forth (174 buybacks) and three corporations rotating one (73 rotations). The rule is on the
legal-move lists only, so the games still replay as played.
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
MAX_ORS = 2


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


def _president(corp):
    owner = (corp.owner_id_str or "").split(":")[-1]
    return int(owner) if owner.isdigit() else None


def _expected(data):
    """Index -> (buyer, seller, refused) for every purchase of a sister corporation's last train, from bookkeeping
    of the rule's own (ORs counted by their (set, number), presidents read from the share register)."""
    game = _rust(data, rule=False)
    raw = game._game
    ors, runs, out = [], {}, {}
    for i, action in enumerate(data["actions"]):
        if raw.round.round_type == "Operating" and (not ors or ors[-1] != (raw.turn, raw.round.round_num)):
            ors.append((raw.turn, raw.round.round_num))
        if action["type"] == "buy_train" and not action.get("exchange"):
            corps = {c.sym: c for c in raw.corporations}
            seller = next((c for c in corps.values() if any(t.id == action["train"] for t in c.trains)), None)
            buyer = corps[action["entity"]]
            pres = _president(buyer)
            if seller is not None and seller.sym != buyer.sym and len(seller.trains) == 1 and pres == _president(seller):
                serial = len(ors)
                last, run = runs.get(pres, (0, 0))
                out[i] = (buyer.sym, seller.sym, last + 1 == serial and run >= MAX_ORS)
                if last != serial:
                    runs[pres] = (serial, run + 1 if last + 1 == serial else 1)
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


def test_a_third_or_in_a_row_of_last_train_purchases_is_refused(any_game):
    """Every purchase of a sister's last train in a third OR in a row is refused, with no train of that seller
    offered; every other one is offered at the price played."""
    expected = _expected(any_game)
    game = _rust(any_game)
    refused = offered = 0
    for i, action in enumerate(any_game["actions"]):
        if i in expected:
            buyer, seller, blocked = expected[i]
            choices = game.get_factored_choices()
            assert game.train_buyback_blocked(buyer, seller) == blocked, (i, action, game.shuffle_runs)
            if blocked:
                assert not _offers(choices, seller), (i, action)
                refused += 1
            else:
                assert _offers(choices, seller, action["price"]), (i, action, game.shuffle_runs)
                offered += 1
        game.process_action(dict(action))
    # The recorded games go on after a refused purchase, so a later one in the same OR counts as that OR's
    # second (never limited): about half of a continuous shuffle's purchases show as refused here; in play the
    # first refusal ends it. The lr1 games' buybacks / rotations are all such purchases.
    marked = set(any_game.get("buyback_indices") or any_game.get("rotation_indices"))
    assert marked <= set(expected)
    assert (refused, offered) == {174: (88, 90), 73: (25, 63)}[len(marked)]


def test_both_engines_agree_through_the_game(any_game):
    """The Python engine keeps the same bookkeeping (each sister last-train purchase, the OR count) and both
    legal-move lists agree at every position of the game -- with the rule biting."""
    rust = _rust(any_game)
    python = _python(any_game)
    helper = FactoredActionHelper()
    for i, action in enumerate(any_game["actions"]):
        assert rust.shuffle_runs == python.shuffle_runs and rust.or_serial == python.or_serial, i
        r = {_canonical_key(c) for c in rust.get_factored_choices()}
        p = {_canonical_key(c) for c in helper.get_choices(python)}
        assert r == p, (i, action, r ^ p)
        rust.process_action(dict(action))
        python.process_action(dict(action))


def _first_refused(data):
    return next(i for i, (_, _, blocked) in sorted(_expected(data).items()) if blocked)


def test_action_helper_refuses_it(shuffle_game):
    """The non-factored ActionHelper (Python search, the importer's matching) applies the rule too."""
    i = _first_refused(shuffle_game)
    python = _python(shuffle_game)
    for action in shuffle_game["actions"][:i]:
        python.process_action(dict(action))
    buyer, seller, _ = _expected(shuffle_game)[i]
    entity = python.current_entity
    assert any(getattr(t.owner, "id", None) == seller for t in python.round.active_step().buyable_trains(entity))
    actions = ActionHelper().get_buy_train_actions(python)
    assert not any(type(a).__name__ == "BuyTrain" and getattr(a.train.owner, "id", None) == seller for a in actions)
    assert python.train_buyback_blocked(buyer, seller)


def test_off_by_default_and_on_in_generator_games(shuffle_game):
    """Human-game import and the advisor build engines directly: the real rules, the purchase offered."""
    from rl18xx.agent.alphazero.start_positions import new_game

    assert not RustGame(_players(shuffle_game)).no_buyback_rule
    assert not GameMap().game_by_title("1830")(_players(shuffle_game)).no_buyback_rule
    assert new_game(4)._game.no_buyback_rule and not new_game(4, no_buyback_rule=False)._game.no_buyback_rule
    i = _first_refused(shuffle_game)
    buyer, seller, _ = _expected(shuffle_game)[i]
    for game in (_rust(shuffle_game, rule=False), _python(shuffle_game, rule=False)):
        for action in shuffle_game["actions"][:i]:
            game.process_action(dict(action))
        assert not game.train_buyback_blocked(buyer, seller)
    rust = _rust(shuffle_game, rule=False)
    for action in shuffle_game["actions"][:i]:
        rust.process_action(dict(action))
    assert _offers(rust.get_factored_choices(), seller, shuffle_game["actions"][i]["price"])
