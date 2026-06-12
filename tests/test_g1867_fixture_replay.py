"""1867 fixture-replay gate (Phase 1 validation strategy, leg 1).

Two layers while Phase 1 lands mechanic by mechanic:

1. ``test_fixture_prefix_frontier`` — the RATCHET. Each fixture must replay
   at least as far as the pinned watermark (the action index where the
   engine currently stops, at the boundary of the next unimplemented
   mechanic). Landing a mechanic moves the frontier forward: bump the
   watermark in the same commit. A regression below the watermark is red.
2. ``test_fixture_replays_to_recorded_result`` — the FINISH LINE: every
   action accepted and exact final scores. xfail (non-strict) until the
   full ruleset lands; drop the marker when it first passes.
"""

import json
from pathlib import Path

import pytest

from tests.title_replay_harness import (
    ReplayError,
    TitleUnsupported,
    check_fixture,
    replay_game,
    supported_titles,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "1867"
FIXTURES = sorted(FIXTURE_DIR.glob("*.json"))

# fixture stem -> minimum number of actions the engine must accept.
# Current frontier: the COMPLETE opening single-item auction; every fixture
# stops at its first stock-round action (a minor-founding bid that carries
# `corporation` — the incremental-capitalization/SR-bid mechanic, next).
PREFIX_WATERMARK = {
    "21268": 35,
    "hs_ahjzadkh_19792": 38,
    "hs_wuveadew_21268": 35,
    "nationalization_cash": 34,
}


def test_fixtures_are_vendored():
    assert len(FIXTURES) >= 4, "1867 fixtures missing from tests/fixtures/1867"


@pytest.mark.skipif(
    "1867" not in supported_titles(),
    reason="engine does not register the 1867 title yet (Phase 1 in progress)",
)
@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.stem)
def test_fixture_prefix_frontier(path):
    game = json.loads(path.read_text())
    try:
        report = replay_game(game)
        applied, first_rejection = report.actions_applied, None
    except ReplayError as exc:
        applied, first_rejection = exc.step, exc
    watermark = PREFIX_WATERMARK[path.stem]
    assert applied >= watermark, (
        f"replay frontier regressed: {applied} < watermark {watermark}; "
        f"first rejection: {first_rejection}"
    )


@pytest.mark.xfail(
    reason="1867 Phase 1 in progress — mechanics landing seam by seam",
    strict=False,
)
@pytest.mark.skipif(
    "1867" not in supported_titles(),
    reason="engine does not register the 1867 title yet (Phase 1 in progress)",
)
@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.stem)
def test_fixture_replays_to_recorded_result(path):
    try:
        report = check_fixture(path)
    except TitleUnsupported as exc:  # pragma: no cover - guarded by skipif
        pytest.skip(str(exc))
    assert report.actions_applied == report.actions_total
