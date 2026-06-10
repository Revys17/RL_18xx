"""Replay test for the table-driven step machinery (engine-rs/src/steps.rs).

The Rust engine's round architecture is a port of Ruby/Python's: per-title
ordered step lists + ONE shared ``actions_for`` accumulation loop. Since the
Stage B switch, ``legal_action_types()`` (and the whole factored enumeration /
decode / MCTS stack behind it) IS the step machinery; the historical
hand-derived dispatch survives only as a test-only oracle inside the crate,
where the random-walk differential lives
(``steps::tests::differential_step_machinery_*`` vs
``legacy_legal_action_types_oracle``).

What this test still pins, on HUMAN-GAME import trajectories (the states real
training data walks through, incl. teleport / home-token / crowded-corp
interleavings):
  * the machinery enumerates without error at every walked state, and the
    production cleaning stream replays cleanly through it;
  * the two PyO3 surfaces (``legal_action_types`` / ``step_action_types``)
    stay wired to the same loop.
Cross-engine (vs the Python reference) coverage for these same trajectories
is the index-level parity harness (``tests/parity_runner.py --human-games``).

The applied-action stream is produced by the production cleaning loop
(``tests/cleaning_diff.trace_clean`` on the Rust engine) and replayed on a
fresh Rust game with the comparison before every applied action.
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


# NOTE: post-Stage-B these two PyO3 surfaces delegate to the SAME shared
# accumulation loop, so comparing them is NOT a differential against the
# legacy dispatch — that differential lives in-crate
# (steps::tests::differential_* vs legacy_legal_action_types_oracle, which is
# a byte-faithful frozen copy of the pre-refactor dispatch). This test's
# value is the no-crash replay over human-game trajectories plus pinning
# both surfaces to one loop.
def _types_via_legal_action_types(game):
    return sorted(set(game.legal_action_types()))


def _types_via_step_action_types(game):
    return sorted(set(game.step_action_types()))


@pytest.mark.skipif(not _SAMPLE, reason="human_games corpus not present")
@pytest.mark.parametrize("path", _SAMPLE, ids=[Path(p).stem for p in _SAMPLE])
def test_step_machinery_replays_human_games(path):
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
        old = _types_via_legal_action_types(rust)
        new = _types_via_step_action_types(rust)
        assert old == new, (
            f"{Path(path).stem} before applied action #{entry['index']} ({entry['label']}): "
            f"legal_action_types {old} != step_action_types {new} "
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
    assert _types_via_legal_action_types(rust) == _types_via_step_action_types(rust)
    assert checked == len(applied)
