"""1822CA legal-move generator: short random walks through the wire path
(generator -> 18xx.games dict -> process_action) for every supported
player count, and the engine's player range.

The long walks run in-crate (``cargo test g1822::fuzz::ca``) and through
``scripts/g1822_fuzz.py --title 1822CA``, whose recorded games
``scripts/ruby_diff.py --title 1822CA --tokens`` replays in Ruby HEAD.
"""

import json
import random
from pathlib import Path

import pytest

import engine_rs

from scripts.g1822_fuzz import pick, walk
from tests.title_replay_harness import supported_titles

pytestmark = pytest.mark.skipif("1822CA" not in supported_titles(), reason="engine lacks 1822CA")

FIXTURES = Path(__file__).parent / "fixtures" / "g1822_generator"


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


@pytest.mark.parametrize(
    "hex_id,corp,tile,error",
    [
        ("AF12", "GT", "M6-0", "M3-0 is not legally rotated for M6"),
        ("A7", "GTP", "512-0", "216-1 is not legally rotated for 512"),
    ],
)
def test_destination_hex_lay_is_applied_before_it_is_offered(hex_id, corp, tile, error):
    """A corporation's own lay skips the generator's apply filter, but on a destination hex the process first picks
    up the destination tokens (G1822CA#preprocess_action) and checks the rotation on the graph without them. In
    these policy self-play games (tests/fixtures/g1822_generator) the corporation's destination token is its only
    connection to its destination: the generator offered lays there that both engines refuse (Ruby HEAD in the rig:
    "M3 is not legally rotated for M6", "216 is not legally rotated for 512" -- the error a 2026-11-14
    policy-gradient smoke game died on)."""
    data = json.loads((FIXTURES / f"1822CA_destination_lay_{hex_id}.json").read_text())
    players = {p["id"]: p["name"] for p in data["players"]}
    g = engine_rs.BaseGame.new_titled("1822CA", players, None, data["settings"]["seed"])
    for a in data["actions"]:
        g.process_action(dict(a))
    moves = json.loads(g.g1822_legal_moves_json())
    assert any(m["wire"]["entity"] == corp and m["wire"]["type"] == "lay_tile" for m in moves)
    assert [m["wire"] for m in moves if m["wire"].get("hex") == hex_id] == []
    assert g.g1822_refused_generated_moves("lay_tile") == []
    lay = {"type": "lay_tile", "entity": corp, "entity_type": "corporation", "hex": hex_id, "tile": tile}
    with pytest.raises(Exception, match=error):
        g.pickle_clone().process_action({**lay, "rotation": 0})
