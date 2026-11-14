"""1822MX legal-move generator: short random walks through the wire path
(generator -> 18xx.games dict -> process_action) and the acting player of
the presidentless NdeM's decisions.

The long walks run in-crate (``cargo test g1822::fuzz::mx``) and through
``scripts/g1822_fuzz.py --title 1822MX``, whose recorded games
``scripts/ruby_diff.py --title 1822MX`` replays in Ruby HEAD.
"""

import json
import random
from pathlib import Path

import numpy as np
import pytest

import engine_rs

from rl18xx.agent.alphazero.generic_tokens import encode_batch
from scripts.g1822_fuzz import pick, walk
from tests.title_replay_harness import supported_titles

pytestmark = pytest.mark.skipif("1822MX" not in supported_titles(), reason="engine lacks 1822MX")

FIXTURES = Path(__file__).parent / "fixtures" / "g1822_generator"


def _ndem_stations(g) -> int:
    return sum(1 for h in g.hexes for c in h.tile.cities for t in c.tokens if t is not None and t.corporation_id == "NDEM")


def _ndem_tokens(g) -> list:
    return next(c for c in g.corporations if c.sym == "NDEM").tokens


def _position_is_sound(g):
    """The RL index API lists the moves (distinct indices), every generated move applies under both policies, and
    the encoder gives finite features."""
    idx = [m["index"] for m in json.loads(g.rl_legal_moves_json())]
    assert idx and len(idx) == len(set(idx))
    for policy in (None, "rules"):
        assert g.g1822_refused_generated_moves(None, policy) == []
    for key, value in encode_batch([g]).items():
        if isinstance(value, np.ndarray) and value.dtype.kind == "f":
            assert np.isfinite(value).all(), key


def test_ndem_takes_an_eleventh_station():
    """Ruby gives the NdeM ten tokens and crashes when the eleventh unbid first-box minor is to become an NdeM
    station (`find_token_by_type` nil: a Ruby dead end that ended 18 of 39 T 0.5 smoke games). The printed rules
    (5.12.6) turn the minor's own token into the station, with no cap, and the user chose them (2026-11-14): the NdeM
    takes a new token. The fixture is such a smoke game (Ruby HEAD agrees through id 773 and raises at 774),
    continued by random generator moves to a 12th station, the privatization auction of the stations and the bank
    break."""
    data = json.loads((FIXTURES / "1822MX_ndem_twelve_stations.json").read_text())
    players = {p["id"]: p["name"] for p in data["players"]}
    g = engine_rs.BaseGame.new_titled("1822MX", players, None, data["settings"]["seed"])
    actions = data["actions"]
    for a in actions[:773]:
        g.process_action(dict(a))
    assert _ndem_stations(g) == 10 and len(_ndem_tokens(g)) == 10 and all(t.used for t in _ndem_tokens(g))
    assert [m["wire"]["type"] for m in json.loads(g.rl_legal_moves_json())] == ["pass"]
    _position_is_sound(g)
    g.process_action(dict(actions[773]))  # the Stock Round's last pass: M23's home becomes the 11th station
    assert _ndem_stations(g) == 11 and len(_ndem_tokens(g)) == 11 and all(t.used for t in _ndem_tokens(g))
    _position_is_sound(g)

    peak = 11
    for a in actions[774:]:
        before = _ndem_stations(g)
        g.process_action(dict(a))
        if _ndem_stations(g) > before:
            peak = _ndem_stations(g)
            _position_is_sound(g)
    assert peak == 12 and len(_ndem_tokens(g)) == 12
    assert g.finished and json.loads(g.g1822_snapshot())["game_end_reason"] == "bank"

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
