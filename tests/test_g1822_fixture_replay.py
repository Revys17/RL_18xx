"""1822 fixture-replay gate (validation strategy leg 1).

tobymao/18xx's own 1822 fixtures (public/fixtures/1822, vendored
anonymized by scripts/vendor_fixture.py — players "Player N", no chat, no
timestamps) must replay END TO END through the Rust engine with the exact
recorded final scores.

  * discard_trains_after_sr_export — 3 players, base rules, a full game
    ending on the bank (exercises minors, majors, concessions, acquisitions,
    destination tokens, P16 tax haven, P18, trains discarded after an SR
    export).
  * hs_lxemeslq_30797 — 7 players, ended manually after the first OR.
  * 33845 — 4 players, 1822+ (plus_expansion).
  * 1822_game_end_stock_market — 4 players, 1822+ with No Removals and
    Single Stack, ends on the stock market's end cell.

Per-action agreement with Ruby (state after every file action) is checked
offline with scripts/ruby_diff.py --title 1822 (docker), not here.
"""

import json
from pathlib import Path

import pytest

from tests.title_replay_harness import ReplayError, check_fixture, replay_game, supported_titles

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "1822"
FIXTURES = sorted(FIXTURE_DIR.glob("*.json"))

# fixture stem -> filtered actions the engine must accept (the full counts:
# any rejection anywhere is a regression).
PREFIX_WATERMARK = {
    "discard_trains_after_sr_export": 918,
    "hs_lxemeslq_30797": 21,
    "33845": 1196,
    "1822_game_end_stock_market": 1210,
}

pytestmark = pytest.mark.skipif("1822" not in supported_titles(), reason="engine lacks 1822")


def test_fixtures_are_vendored():
    assert {p.stem for p in FIXTURES} == set(PREFIX_WATERMARK)


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.stem)
def test_fixture_prefix_frontier(path):
    watermark = PREFIX_WATERMARK[path.stem]
    if watermark is None:
        pytest.skip("not yet replayable")
    game = json.loads(path.read_text())
    try:
        report = replay_game(game)
        applied, first_rejection = report.actions_applied, None
    except ReplayError as exc:
        applied, first_rejection = exc.step, exc
    assert applied >= watermark, (
        f"replay frontier regressed: {applied} < watermark {watermark}; first rejection: {first_rejection}"
    )


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.stem)
def test_fixture_replays_to_recorded_result(path):
    if PREFIX_WATERMARK[path.stem] is None:
        pytest.skip("not yet replayable")
    report = check_fixture(path)
    assert report.actions_applied == report.actions_total
