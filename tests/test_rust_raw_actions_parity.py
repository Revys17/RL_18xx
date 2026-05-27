"""Parity tests for the Rust engine's full action log.

The Rust ``BaseGame.raw_actions`` getter used to return only a sliding window
of recent ``{entity, type}`` summaries. After the action-log refactor, it
returns the full history of action dicts processed (mirroring Python's
``BaseGame.raw_actions``).

These tests ensure:

* every action processed shows up in ``raw_actions``;
* the ``type`` and ``entity`` fields match the Python engine step for step;
* clones (used by MCTS) preserve the log.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.disable(logging.CRITICAL)

from engine_rs import BaseGame as RustGame
from rl18xx.game.action_helper import ActionHelper
from rl18xx.game.gamemap import GameMap
from rl18xx.rust_adapter import RustGameAdapter


PLAYERS = {1: "P1", 2: "P2", 3: "P3", 4: "P4"}


def _make_pair():
    py_game = GameMap().game_by_title("1830")(PLAYERS)
    rust_game = RustGameAdapter(RustGame(PLAYERS))
    return py_game, rust_game


def _normalize_entity(value):
    """Python serializes player entity as int, Rust as ``str(int)``. Normalize
    to ``str`` for comparison."""
    if value is None:
        return None
    return str(value)


def test_rust_raw_actions_starts_empty():
    rust_game = RustGameAdapter(RustGame(PLAYERS))
    assert rust_game.raw_actions == []


def test_rust_raw_actions_full_history():
    """Apply the first ~50 actions of a real auction/stock sequence to both
    engines and verify the Rust action log matches Python step for step."""
    py_game, rust_game = _make_pair()
    helper = ActionHelper()

    steps = 0
    max_steps = 50
    while steps < max_steps:
        choices = helper.get_all_choices(py_game)
        if not choices:
            break
        action = choices[-1] if steps == 0 else choices[0]
        action_dict = action.to_dict()

        py_game.process_action(action)
        rust_dict = {k: v for k, v in action_dict.items() if k != "created_at"}
        rust_game.process_action(rust_dict)
        steps += 1

    assert steps > 0, "expected some actions to apply"
    assert len(rust_game.raw_actions) == len(py_game.raw_actions), (
        f"Rust log has {len(rust_game.raw_actions)} entries, "
        f"Python has {len(py_game.raw_actions)}"
    )

    for i, (ra, pa) in enumerate(zip(rust_game.raw_actions, py_game.raw_actions)):
        assert ra.get("type") == pa.get("type"), (
            f"step {i}: type mismatch — rust={ra.get('type')!r}, py={pa.get('type')!r}"
        )
        assert _normalize_entity(ra.get("entity")) == _normalize_entity(pa.get("entity")), (
            f"step {i}: entity mismatch — rust={ra.get('entity')!r}, py={pa.get('entity')!r}"
        )


def test_rust_raw_actions_preserved_across_clone():
    """``pickle_clone`` should copy the full log so MCTS children inherit it."""
    py_game, rust_game = _make_pair()
    helper = ActionHelper()

    for _ in range(10):
        choices = helper.get_all_choices(py_game)
        if not choices:
            break
        action = choices[0]
        py_game.process_action(action)
        d = {k: v for k, v in action.to_dict().items() if k != "created_at"}
        rust_game.process_action(d)

    before = list(rust_game.raw_actions)
    clone = rust_game.pickle_clone()
    after = list(clone.raw_actions)

    assert len(before) == len(after) > 0
    for a, b in zip(before, after):
        assert a.get("type") == b.get("type")
        assert a.get("entity") == b.get("entity")


def test_rust_raw_actions_includes_action_args():
    """Each entry should carry the action's own arguments, not just metadata."""
    rust_game = RustGameAdapter(RustGame(PLAYERS))

    # First auction action: pick a known company so we can assert on the payload.
    rust_game.process_action({"type": "bid", "entity": 1, "company": "SV", "price": 25})

    log = rust_game.raw_actions
    assert len(log) == 1
    entry = log[0]
    assert entry["type"] == "bid"
    # entity may come back as int or string depending on the conversion path
    assert _normalize_entity(entry["entity"]) == "1"
    assert entry["company"] == "SV"
    assert int(entry["price"]) == 25


if __name__ == "__main__":
    test_rust_raw_actions_starts_empty()
    test_rust_raw_actions_full_history()
    test_rust_raw_actions_preserved_across_clone()
    test_rust_raw_actions_includes_action_args()
    print("OK")
