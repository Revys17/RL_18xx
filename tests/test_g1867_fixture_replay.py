"""1867 fixture-replay gate (Phase 1 validation strategy, leg 1).

Replays the four vendored tobymao/18xx regression fixtures through the Rust
engine, asserting per-action acceptance and exact final scores. Skips while
the engine does not yet register the 1867 title — the tests turn red the
moment a wrong-scoring 1867 lands, which is the point.
"""

from pathlib import Path

import pytest

from tests.title_replay_harness import (
    TitleUnsupported,
    check_fixture,
    supported_titles,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "1867"
FIXTURES = sorted(FIXTURE_DIR.glob("*.json"))


def test_fixtures_are_vendored():
    assert len(FIXTURES) >= 4, "1867 fixtures missing from tests/fixtures/1867"


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
