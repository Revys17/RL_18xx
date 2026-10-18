"""1822CA legal-move generator: short random walks through the wire path
(generator -> 18xx.games dict -> process_action) for every supported
player count, and the engine's player range.

The long walks run in-crate (``cargo test g1822::fuzz::ca``) and through
``scripts/g1822_fuzz.py --title 1822CA``, whose recorded games
``scripts/ruby_diff.py --title 1822CA --tokens`` replays in Ruby HEAD.
"""

import json
import random

import pytest

import engine_rs

from scripts.g1822_fuzz import pick, walk
from tests.title_replay_harness import supported_titles

pytestmark = pytest.mark.skipif("1822CA" not in supported_titles(), reason="engine lacks 1822CA")


@pytest.mark.parametrize("players,seed", [(3, 21), (4, 22), (5, 23), (6, 24)])
def test_short_wire_walk(players, seed):
    rep = walk(players, seed, [], 300, None, "1822CA")
    assert rep["error"] is None, rep["error"]
    assert rep["moves"] == 300 or rep["finished"] or rep["dead_end"]


def test_player_counts():
    # Ruby's 2-7 replay (the corpus and a 7-player fixture); the generator is
    # validated for 3-6
    for n in range(2, 8):
        engine_rs.BaseGame.new_titled("1822CA", {1001 + i: f"Player {i + 1}" for i in range(n)}, None, 7)
    for n in (1, 8):
        with pytest.raises(ValueError):
            engine_rs.BaseGame.new_titled("1822CA", {1001 + i: f"Player {i + 1}" for i in range(n)}, None, 7)


def test_every_decision_has_an_acting_player():
    g = engine_rs.BaseGame.new_titled("1822CA", {1001 + i: f"Player {i + 1}" for i in range(4)}, None, 41)
    rng = random.Random(41)
    for n in range(800):
        if g.finished:
            break
        moves = json.loads(g.g1822_legal_moves_json())
        if not moves:
            break
        assert g.g1822_acting_player() in (1001, 1002, 1003, 1004), (n, moves[0])
        g.process_action({**pick(moves, rng), "id": n + 1})
