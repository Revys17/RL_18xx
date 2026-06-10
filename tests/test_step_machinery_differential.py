"""Differential test for the table-driven step machinery (engine-rs/src/steps.rs).

The Rust engine grew a port of Ruby/Python's round architecture: per-title
ordered step lists + ONE shared ``actions_for`` accumulation loop
(``BaseGame.step_action_types()``). It must reproduce the historical
hand-derived ``legal_action_types()`` enumeration EXACTLY (as a set — the
legacy Vec order is hand-arranged and every consumer is set/index-based).

Coverage here: HUMAN-GAME import trajectories (the states real training data
walks through, incl. teleport / home-token / crowded-corp interleavings).
Random-walk differential coverage lives in the Rust crate
(``steps::tests::differential_step_machinery_*``).

The applied-action stream is produced by the production cleaning loop
(``tests/cleaning_diff.trace_clean`` on the Rust engine) and replayed on a
fresh Rust game with the old-vs-new comparison before every applied action.
"""

import json
import logging
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
logging.disable(logging.CRITICAL)

from engine_rs import BaseGame as RustGame  # noqa: E402
from rl18xx.rust_adapter import RustGameAdapter  # noqa: E402
from tests.cleaning_diff import trace_clean  # noqa: E402
from rl18xx.agent.alphazero.pretraining import _process_pass_leniently  # noqa: E402

# A fixed, deterministic sample across the corpus (every ~400th game of the
# sorted listing) so the differential sees a spread of eras/player counts.
_CORPUS = sorted((Path(__file__).parent.parent / "human_games" / "1830").glob("*.json"))
_SAMPLE = [str(p) for p in _CORPUS[::400]] if _CORPUS else []


def _types_old(game):
    return sorted(set(game.legal_action_types()))


def _types_new(game):
    return sorted(set(game.step_action_types()))


@pytest.mark.skipif(not _SAMPLE, reason="human_games corpus not present")
@pytest.mark.parametrize("path", _SAMPLE, ids=[Path(p).stem for p in _SAMPLE])
def test_step_machinery_matches_legacy_on_human_games(path):
    game_json = json.load(open(path))
    num_players = len(game_json["players"])
    if not (2 <= num_players <= 6):
        pytest.skip("malformed player count — cleaning drops it before any state is walked")

    # 1) Production cleaning on the Rust engine records the applied stream.
    #    (Outcome may be a drop — the states walked up to the drop still count.)
    trace = trace_clean(game_json, use_rust=True)
    applied = trace["applied"]
    if not applied:
        pytest.skip(f"no applied actions ({trace['outcome'].get('status')})")

    # 2) Replay the identical stream on a fresh game, differencing the two
    #    enumerations at every state.
    players = {i + 1: f"Player {i + 1}" for i in range(num_players)}
    adapter = RustGameAdapter(RustGame(players))
    rust = adapter._game

    checked = 0
    for entry in applied:
        old = _types_old(rust)
        new = _types_new(rust)
        assert old == new, (
            f"{Path(path).stem} before applied action #{entry['index']} ({entry['label']}): "
            f"legacy {old} != step machinery {new} "
            f"[round={entry.get('round')}, op_step={entry.get('op_step')}, entity={entry.get('entity')}]"
        )
        checked += 1
        action = entry["action"]
        if action.get("type") == "pass":
            # Mirror the cleaning's lenient pass handling (a rejected pass is
            # dropped silently there; the recorded stream keeps only applied
            # ones, but entity/state drift makes strictness unhelpful here).
            _process_pass_leniently(adapter, action, use_rust=True)
        else:
            adapter.process_action(action)

    # Final state too.
    assert _types_old(rust) == _types_new(rust)
    assert checked == len(applied)
