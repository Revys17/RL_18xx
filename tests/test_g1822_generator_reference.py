"""The 1822 family generator's fast paths against the reference generator (the code before the 2026-11-14
speed-ups, `g1822::legal::reference`) on the vendored Ruby fixtures: at every human decision of one fixture per
title, the sorted move lists, the RL index rows and the state left behind agree under the title's cross-player
policy and the rules' (`BaseGame.g1822_reference_diff`), and the fast graph walk and neighbour map agree with
the reference ones for every tokened corporation (`g1822_graph_check`).

The full checks: `scripts/g1822_generator_diff.py` (every decision of each title's scrape corpus, both policies)
and the in-crate walks (`cargo test --release --no-default-features generator_matches_reference`,
`g1822_diff_long -- --ignored`).
"""

import json
from pathlib import Path

import pytest

import engine_rs

from scripts.g1822_generator_profile import human_actions, new_game
from tests.title_replay_harness import supported_titles

FIXTURES = Path(__file__).parent / "fixtures"
# One fixture per title (the shortest that reaches the late game), every k-th decision.
CASES = [("1822", 3), ("1822PNW", 3), ("1822MX", 3), ("1822CA", 3)]


def _fixture(title: str) -> Path:
    paths = sorted((FIXTURES / title).glob("*.json"), key=lambda p: p.stat().st_size)
    return paths[len(paths) // 2]


@pytest.mark.parametrize("title,every", CASES, ids=[c[0] for c in CASES])
def test_generator_matches_reference_on_a_fixture(title, every):
    if title not in supported_titles():
        pytest.skip(f"engine lacks {title}")
    game = json.load(open(_fixture(title)))
    g, seat = new_game(title, game)
    decisions = compared = 0
    for _fid, auto, a in human_actions(game, seat):
        if g.finished:
            break
        if not auto:
            decisions += 1
            if decisions % every == 0:
                assert g.g1822_graph_check() is None
                for policy in (None, "rules"):
                    moves, diff = g.g1822_reference_diff(policy)
                    assert diff is None, f"decision {decisions} policy {policy}: {diff}"
                    compared += moves
        g.process_action(a)
    assert compared > 0


def test_reference_flag_is_restored():
    prev = engine_rs.BaseGame.g1822_set_reference_generator(True)
    assert engine_rs.BaseGame.g1822_set_reference_generator(prev) is True
    assert engine_rs.BaseGame.g1822_set_reference_generator(False) is False
