"""1822PNW fixture-replay gate.

tobymao/18xx's own 1822PNW fixtures (public/fixtures/1822PNW, vendored
anonymized by scripts/vendor_ruby_fixture.py — players "Player N", no chat,
no timestamps) must replay END TO END through the Rust engine with the
exact recorded final scores. They cover the Merger round (incl. a minor
merged before its turn and high-value minors), P19's coal mine (merged,
on a hex with a cost, the Rockport token), P20 replacing a closed minor,
the paper mill / ski haus / coal mine together, a major out of tokens,
and the bank / stock-market game ends (incl. both in the same turn).

Per-action agreement with Ruby (state after every file action) is checked
offline with scripts/ruby_diff.py --title 1822PNW (docker), not here.
"""

from pathlib import Path

import pytest

from tests.title_replay_harness import check_fixture, supported_titles

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "1822PNW"
FIXTURES = sorted(FIXTURE_DIR.glob("*.json"))

pytestmark = pytest.mark.skipif("1822PNW" not in supported_titles(), reason="engine lacks 1822PNW")


def test_fixtures_are_vendored():
    assert len(FIXTURES) == 17


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.stem)
def test_fixture_replays_to_recorded_result(path):
    report = check_fixture(path)
    assert report.actions_applied == report.actions_total
