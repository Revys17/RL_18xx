"""1822MX fixture-replay gate.

tobymao/18xx's own 1822MX fixtures (public/fixtures/1822MX, vendored
anonymized by scripts/vendor_ruby_fixture.py — players "Player N", no chat,
no log or program texts, no timestamps) must replay END TO END through the
Rust engine with the exact recorded final scores. They cover the NdeM (its
first home among M14/M15/M17, minors it replaces without doubling up its
tokens, its shareholders laying its track, its privatization: the cash-out
and the token auction, incl. a token bought in Mexico City), C1's minor 18,
FCM unable to lay through Mexico City (M22's connection rule), the
higher_cert_limit option and the stock-market game end.

Per-action agreement with Ruby (state after every file action) is checked
offline with scripts/ruby_diff.py --title 1822MX (docker), not here.
"""

from pathlib import Path

import pytest

from tests.title_replay_harness import check_fixture, supported_titles

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "1822MX"
FIXTURES = sorted(FIXTURE_DIR.glob("*.json"))

pytestmark = pytest.mark.skipif("1822MX" not in supported_titles(), reason="engine lacks 1822MX")


def test_fixtures_are_vendored():
    assert len(FIXTURES) == 6


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.stem)
def test_fixture_replays_to_recorded_result(path):
    report = check_fixture(path)
    assert report.actions_applied == report.actions_total
