"""1822MX legal-move generator: short random walks through the wire path
(generator -> 18xx.games dict -> process_action) and the acting player of
the presidentless NdeM's decisions.

The long walks run in-crate (``cargo test g1822::fuzz::mx``) and through
``scripts/g1822_fuzz.py --title 1822MX``, whose recorded games
``scripts/ruby_diff.py --title 1822MX`` replays in Ruby HEAD.
"""

import json
import random

import pytest

import engine_rs

from scripts.g1822_fuzz import pick, walk
from tests.title_replay_harness import supported_titles

pytestmark = pytest.mark.skipif("1822MX" not in supported_titles(), reason="engine lacks 1822MX")


@pytest.mark.parametrize("players,seed,rules", [(3, 11, []), (4, 12, ["higher_cert_limit"]), (5, 13, [])])
def test_short_wire_walk(players, seed, rules):
    rep = walk(players, seed, rules, 300, None, "1822MX")
    assert rep["error"] is None, rep["error"]
    assert rep["moves"] == 300 or rep["finished"] or rep["dead_end"]


def test_player_counts():
    for n in (3, 4, 5):
        engine_rs.BaseGame.new_titled("1822MX", {1001 + i: f"Player {i + 1}" for i in range(n)}, None, 7)
    with pytest.raises(ValueError):
        engine_rs.BaseGame.new_titled("1822MX", {1001 + i: f"Player {i + 1}" for i in range(6)}, None, 7)


def test_ndem_decisions_have_an_acting_player():
    """Every decision the NdeM's steps offer belongs to a seated player."""
    g = engine_rs.BaseGame.new_titled("1822MX", {1001 + i: f"Player {i + 1}" for i in range(3)}, None, 31)
    rng = random.Random(31)
    seen = 0
    for n in range(1500):
        if g.finished:
            break
        moves = json.loads(g.g1822_legal_moves_json())
        if not moves:
            break
        actor = g.g1822_acting_player()
        assert actor in (1001, 1002, 1003), (n, actor, moves[0])
        if moves[0]["wire"].get("entity") == "NDEM":
            seen += 1
        g.process_action({**pick(moves, rng), "id": n + 1})
    assert seen > 0, "the walk never reached an NdeM decision"
