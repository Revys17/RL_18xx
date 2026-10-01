"""18xx.games ``log`` actions (the consent audit line) through the human-game
import.

When a player confirms another player's consent (Ruby client
``Actionable#check_consent``), 18xx.games records
``{"type": "log", "message": "• confirmed receiving consent from ..."}`` before
the real action. Ruby's ``Action::Log < Message`` is state-neutral
(``Step::Message#process_log`` only appends to the game log), but both our
engines reject it, so ``filter_actions`` strips it — AFTER the undo/redo pass,
because Ruby's ``Game::Base.filtered_actions`` special-cases only ``message``:
a bare ``undo`` right after a ``log`` undoes the log, not the preceding game
action.

The fixture is an anonymized 3-player game truncated at action 64 (see its
``description``): two consent logs (59-60) are each followed by a rejected
click, then two bare undos (61-62) that consume the logs, leaving passes 57-58
in place for the buy at 63.
"""

import copy
import json
import logging
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
logging.disable(logging.CRITICAL)

from rl18xx.agent.alphazero.pretraining import (  # noqa: E402
    _get_game_object_for_game_with_reason,
    filter_actions,
)
from tests import cleaning_diff  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "1830" / "consent_log_undo.json"
LOG_IDS = (59, 60)
UNDO_IDS = (61, 62)


def _load_fixture():
    with open(FIXTURE) as f:
        return json.load(f)


def _without_undos(game):
    """The same game with the two bare undos removed, so both logs survive the
    undo/redo pass and must be stripped (this raised before the fix). Since a
    log is state-neutral, the resulting game state is identical."""
    game = copy.deepcopy(game)
    game["actions"] = [a for a in game["actions"] if a["id"] not in UNDO_IDS]
    return game


STREAMS = {"undone_logs": _load_fixture, "surviving_logs": lambda: _without_undos(_load_fixture())}


def test_fixture_shape():
    by_id = {a["id"]: a for a in _load_fixture()["actions"]}
    assert all(by_id[i]["type"] == "log" for i in LOG_IDS)
    assert all(by_id[i]["type"] == "undo" and "action_id" not in by_id[i] for i in UNDO_IDS)


@pytest.mark.parametrize("stream", STREAMS)
def test_filter_actions_strips_logs_and_keeps_the_passes(stream):
    filtered = filter_actions(STREAMS[stream]()["actions"])
    assert not any(a["type"] == "log" for a in filtered)
    # The bare undos consumed the logs, not passes 57-58 (Player 3, Player 1).
    buy = next(i for i, a in enumerate(filtered) if a.get("shares") == ["C&O_1"])
    assert [(a["type"], a["entity"]) for a in filtered[buy - 2 : buy + 1]] == [
        ("pass", 3),
        ("pass", 1),
        ("buy_shares", 2),
    ]


def _a(type_, id_, **kw):
    return {"type": type_, "id": id_, **kw}


@pytest.mark.parametrize(
    "actions, kept",
    [
        # A bare undo after a log undoes the log itself (Ruby filtered_actions:
        # only ``message`` is skipped when finding the undo target).
        ([_a("pass", 1), _a("log", 2), _a("undo", 3)], [1]),
        ([_a("pass", 1), _a("log", 2), _a("undo", 3), _a("undo", 4)], []),
        ([_a("pass", 1), _a("log", 2), _a("undo", 3), _a("redo", 4)], [1]),
        # An undo targeting a log keeps everything up to and including it.
        ([_a("pass", 1), _a("log", 2), _a("pass", 3), _a("undo", 4, action_id=2)], [1]),
        # Auto actions recorded on a log are real game actions and survive.
        ([_a("log", 1, auto_actions=[_a("pass", None)])], [None]),
    ],
)
def test_filter_actions_log_undo_semantics(actions, kept):
    filtered = filter_actions(actions)
    assert [a["type"] for a in filtered] == ["pass"] * len(kept)
    assert [a["id"] for a in filtered] == list(range(1, len(kept) + 1))


@pytest.fixture(scope="module")
def cleaned():
    """Each stream cleaned on each engine: {(stream, use_rust): (game, reason)}."""
    return {
        (stream, use_rust): _get_game_object_for_game_with_reason(load(), use_rust=use_rust)
        for stream, load in STREAMS.items()
        for use_rust in (True, False)
    }


@pytest.mark.parametrize("use_rust", [True, False], ids=["rust", "python"])
@pytest.mark.parametrize("stream", STREAMS)
def test_cleaner_replays_the_consent_log_game(cleaned, stream, use_rust):
    game, reason = cleaned[(stream, use_rust)]
    assert reason is None
    assert game.share_by_id("C&O_1").owner.id == 2
    assert game.current_entity.id == 3
    assert {p.id: p.cash for p in game.players} == {1: 88, 2: 206, 3: 120}


@pytest.mark.parametrize("stream", STREAMS)
def test_strict_lockstep_parity(stream):
    assert cleaning_diff.diagnose_game(STREAMS[stream](), strict=True)["status"] == "parity"
