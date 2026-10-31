"""A corporation at its train limit buys a train only by trading one in.

Ruby's BuyTrain step opens whenever ``can_buy_train?`` -- which includes 1830's D trade-in, needing no
room -- and its ``process_buy_train`` never checks room, so it accepts any purchase there; the site's
buy view offers depot and other corporations' trains only with room (assets/app/view/game/
buy_trains.rb). Our legal moves offered them, and the bots bought trains at the limit and discarded
(77 times in 5,527 lr1 ladder games). The fixture is an engine-generated game cut at such a purchase:
B&M at its 2-train limit in phase D, which bought ERIE's 5-train.
"""

import json
import logging
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
logging.disable(logging.CRITICAL)

from engine_rs import BaseGame as RustGame  # noqa: E402
from rl18xx.game.factored_action_helper import FactoredActionHelper  # noqa: E402
from rl18xx.game.gamemap import GameMap  # noqa: E402
from rl18xx.rust_adapter import RustGameAdapter  # noqa: E402

from tests.test_factored_action_helper_rust_parity import _canonical_key  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "1830" / "buy_train_at_limit.json"


@pytest.fixture(scope="module")
def position():
    return json.loads(FIXTURE.read_text())


def _players(data):
    return {p["id"]: p["name"] for p in data["players"]}


@pytest.fixture(scope="module")
def rust_game(position):
    game = RustGameAdapter(RustGame(_players(position)))
    for action in position["actions"]:
        game.process_action(dict(action))
    return game


@pytest.fixture(scope="module")
def python_game(position):
    game = GameMap().game_by_title("1830")(_players(position))
    for action in position["actions"]:
        game.process_action(dict(action))
    return game


def _buy_trains(choices):
    return [c for c in choices if c.type == "BuyTrain"]


def test_the_position_is_a_corporation_at_its_limit(rust_game, position):
    corp = rust_game.corporation_by_id(position["bad_action"]["entity"])
    assert str(rust_game.phase.name) == "D" and len(corp.trains) == 2  # the limit in phase D


def test_at_the_limit_only_a_trade_in_is_offered(rust_game, position):
    buys = _buy_trains(rust_game.get_factored_choices())
    sources = {str(b.entity.get("source")) for b in buys}
    assert "ERIE" not in sources  # another corporation's train: no room for it
    for buy in buys:  # what is left trades a train in for the discounted D
        assert buy.entity.get("source") == "depot" and str(buy.entity.get("train")).startswith("D")
        assert buy.entity.get("exchange") or buy.price_range[1] < 1100


def test_both_engines_agree_at_the_limit(rust_game, python_game):
    rust = {_canonical_key(c) for c in rust_game.get_factored_choices()}
    python = {_canonical_key(c) for c in FactoredActionHelper().get_choices(python_game)}
    assert rust == python
