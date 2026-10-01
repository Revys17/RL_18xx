"""1830 ``optional_6_train`` (Ruby G1830 ``num_trains``: a 3rd 6-train) in both
engines and through the human-game import.

18xx.games games with the rule buy a ``6-2`` that the base-rules depot never
has. The cleaner used to build both engines without the game's optional rules
and crashed dereferencing ``train_by_id("6-2")`` (``None``). The fixture is an
anonymized 2-player game truncated right after that buy (see its
``description``).
"""

import copy
import json
import logging
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
logging.disable(logging.CRITICAL)

from engine_rs import BaseGame as RustGame  # noqa: E402
from rl18xx.game.gamemap import GameMap  # noqa: E402
from rl18xx.agent.alphazero.pretraining import (  # noqa: E402
    _get_game_object_for_game_with_reason,
    _load_cleaned_game_via_rust,
)
from tests import cleaning_diff  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "1830" / "optional_6_train.json"
PLAYERS = {1: "Player 1", 2: "Player 2"}


def _load_fixture():
    with open(FIXTURE) as f:
        return json.load(f)


def _six_ids(game):
    return [t.id for t in game.depot.trains if t.name == "6"]


def _train_owner(game, train_id):
    return game.train_by_id(train_id).owner.id


@pytest.mark.parametrize(
    "rules, sixes",
    [([], ["6-0", "6-1"]), (["optional_6_train"], ["6-0", "6-1", "6-2"])],
)
def test_both_engines_build_the_optional_third_six(rules, sixes):
    py = GameMap().game_by_title("1830")(PLAYERS, optional_rules=rules)
    rust = RustGame(PLAYERS, optional_rules=rules)
    assert _six_ids(py) == sixes
    assert _six_ids(rust) == sixes
    assert [t.id for t in py.depot.trains] == [t.id for t in rust.depot.trains]
    assert list(rust.optional_rules) == rules


def test_rust_rejects_unimplemented_optional_rules():
    # The Python engine models multiple_brown_from_ipo; the Rust engine does
    # not, so accepting it would silently break parity.
    with pytest.raises(ValueError, match="multiple_brown_from_ipo"):
        RustGame(PLAYERS, optional_rules=["multiple_brown_from_ipo"])


@pytest.fixture(scope="module")
def cleaned():
    """The fixture cleaned on each engine: {use_rust: (game, reason)}."""
    return {
        use_rust: _get_game_object_for_game_with_reason(_load_fixture(), use_rust=use_rust)
        for use_rust in (True, False)
    }


@pytest.mark.parametrize("use_rust", [True, False], ids=["rust", "python"])
def test_cleaner_replays_the_optional_6_train_game(cleaned, use_rust):
    game, reason = cleaned[use_rust]
    assert reason is None
    assert len(game.raw_actions) > 300
    assert _train_owner(game, "6-2") == "C&O"


def test_engines_agree_after_cleaning(cleaned):
    (rust, _), (py, _) = cleaned[True], cleaned[False]
    assert {p.id: p.cash for p in rust.players} == {p.id: p.cash for p in py.players}
    for corp in py.corporations:
        rcorp = rust.corporation_by_id(corp.id)
        assert rcorp.cash == corp.cash, corp.id
        assert sorted(t.id for t in rcorp.trains) == sorted(t.id for t in corp.trains), corp.id
    assert [t.id for t in rust.depot.upcoming] == [t.id for t in py.depot.upcoming]


def test_strict_lockstep_parity():
    assert cleaning_diff.diagnose_game(_load_fixture(), strict=True)["status"] == "parity"


@pytest.mark.parametrize("use_rust", [True, False], ids=["rust", "python"])
def test_cleaned_game_keeps_its_optional_rules(cleaned, use_rust):
    """``to_dict`` -> ``_load_cleaned_game_via_rust`` is how cleaned games
    reach training-data conversion; the rule must survive that hop."""
    game_dict = cleaned[use_rust][0].to_dict()
    assert game_dict["settings"]["optional_rules"] == ["optional_6_train"]
    reloaded = _load_cleaned_game_via_rust(game_dict)
    assert reloaded is not None
    assert list(reloaded.optional_rules) == ["optional_6_train"]
    assert _train_owner(reloaded, "6-2") == "C&O"


@pytest.mark.parametrize("use_rust", [True, False], ids=["rust", "python"])
def test_unknown_train_is_dropped_with_a_reason(use_rust):
    """Recorded under train rules the engine isn't running: an explicit drop,
    not an AttributeError."""
    game = copy.deepcopy(_load_fixture())
    game["settings"]["optional_rules"] = []
    assert _get_game_object_for_game_with_reason(game, use_rust=use_rust) == (None, "unknown_train")
