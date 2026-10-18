"""1822CA fixture-replay gate.

tobymao/18xx's own 1822CA fixtures (public/fixtures/1822CA, vendored
anonymized by scripts/vendor_ruby_fixture.py — players "Player N", no chat,
no log or program texts, no timestamps) must replay END TO END through the
Rust engine with the exact recorded final scores. They cover the big cities
(Toronto, Ottawa, Montreal, Quebec, Winnipeg) and their P14-P18 free
upgrades, QMOO's Quebec home following the city tiles, ICR's and the
exit-specific destinations, destination tokens re-placed across lays, P10's
Winnipeg token, the sawmill, grain trains and pullmans attached in the Route
step ("None" skips), the minor-acquisition bonus lay, the Detroit-Duluth
opening, the tax haven holding several shares, the bank break's extra OR,
the stock-market end, player loans, and a 7-player game.

Per-action agreement with Ruby (state after every file action, incl. city
token slots) is checked offline with scripts/ruby_diff.py --title 1822CA
--tokens (docker), not here.
"""

from pathlib import Path

import pytest

from tests.title_replay_harness import check_fixture, supported_titles

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "1822CA"
FIXTURES = sorted(FIXTURE_DIR.glob("*.json"))

pytestmark = pytest.mark.skipif("1822CA" not in supported_titles(), reason="engine lacks 1822CA")


def test_fixtures_are_vendored():
    assert len(FIXTURES) == 7


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.stem)
def test_fixture_replays_to_recorded_result(path):
    report = check_fixture(path)
    assert report.actions_applied == report.actions_total
