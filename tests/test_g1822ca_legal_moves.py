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

from rl18xx.agent.alphazero.action_descriptors import action_descriptors
from scripts.g1822_fuzz import pick, walk
from tests.title_replay_harness import supported_titles

pytestmark = pytest.mark.skipif("1822CA" not in supported_titles(), reason="engine lacks 1822CA")

FIXTURES = Path(__file__).parent / "fixtures" / "g1822_generator"
# 1822CA EXTRA_TRAINS: the companies' trains, `buyable: false` in Ruby (their card ids are "<type>-<n>")
COMPANY_TRAINS = ("2P", "P", "LP", "G")


def _replay_fixture(name: str):
    data = json.loads((FIXTURES / name).read_text())
    players = {p["id"]: p["name"] for p in data["players"]}
    return data["actions"], engine_rs.BaseGame.new_titled("1822CA", players, None, data["settings"]["seed"])


def _icr_trains(g) -> list:
    return sorted(json.loads(g.g1822_snapshot())["corps"]["ICR"]["tr"])


def test_grain_name_left_on_a_train():
    """Attaching the grain train to a corporation's last local train when it has no route on the graph takes away
    its run (`can_run_route?` asks for an L / LP by name): its Route step is skipped, run_routes never detaches the
    grain train and Route#setup drops it with the name left on ("LP-G" for good). Ruby HEAD does exactly this (the
    fixture replays there with zero divergence). The engine mirrors it, its memoized crowded_corps (the LP still an
    extra train until trains next move), the later grain / pullman offers on the "LP-G" and its runs; the RL
    descriptors name the train's type, "LP" (a 2026-11-14 policy game died on 'Choose.grain ["route:grain",
    "LP-G", "0"] is not in the layout'), and no corporation is offered another's company train."""
    actions, g = _replay_fixture("1822CA_grain_name_left_ICR.json")
    d = action_descriptors("1822CA")
    for a in actions[:593]:
        g.process_action(dict(a))
    assert _icr_trains(g) == ["3", "3", "G", "LP", "P"]
    grain = {m["wire"]["choice"]: m for m in json.loads(g.rl_legal_moves_json()) if m["block"] == "Choose.grain"}
    row = d.describe(grain["2"]["index"])
    assert (row["choice"], row["ent0"], row["param"]) == ("route:grain", ("train", "LP"), 0)
    g.process_action(dict(actions[593]))
    snap = json.loads(g.g1822_snapshot())
    assert snap["step"] != "Route" and "LP-G" in _icr_trains(g), "the attachment took ICR's run"
    for a in actions[594:596]:
        g.process_action(dict(a))
    snap = json.loads(g.g1822_snapshot())
    # not made to discard (Ruby's crowded_corps memo predates the rename): the next corporation operates
    assert (snap["ent"], snap["step"]) == ("CPR", "AcquireCompany") and "LP-G" in _icr_trains(g)

    offered = {"Choose.grain": 0, "Choose.pullman": 0}
    corp_buys = 0
    for a in actions[596:]:
        for policy in (None, "rules"):
            for m in json.loads(g.rl_legal_moves_json(policy)):
                wire = m["wire"]
                if m["block"] == "BuyTrain.corp":
                    corp_buys += 1
                    assert wire["train"].split("-")[0] not in COMPANY_TRAINS, wire
                if wire["type"] == "discard_train":
                    assert wire["train"] != "LP-0", "no variant left for it in the pool: Ruby breaks"
                if m["block"] in offered and wire["entity"] == "ICR" and policy is None:
                    row = d.describe(m["index"])
                    assert row["ent0"][1] != "LP-G" and not row["ent0"][1].endswith("G"), row
                    if row["ent0"] == ("train", "LP"):
                        offered[m["block"]] += 1
        g.process_action(dict(a))
    assert offered["Choose.grain"] > 0 and offered["Choose.pullman"] > 0, offered
    assert corp_buys > 0
    assert "LP-G" in _icr_trains(g)


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
