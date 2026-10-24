"""Search learning for the long run (docs/long_run_recipe.md): league seats (members searched
with their own networks, PFSP, per-member scores, the main system's snapshots), a followed
separate value (``value_follow``: a value-train run's promoted value) and generation-only runs
(``freeze_policy`` with ``value_follow``)."""

import json
import os
import random

import numpy as np
import pytest

engine_rs = pytest.importorskip("engine_rs")

from rl18xx.agent.alphazero import search_learning as sl  # noqa: E402
from rl18xx.agent.alphazero.lockstep import ModelEvaluator  # noqa: E402

from .search_test_utils import HashNet  # noqa: E402
from .test_search_learning import _save_checkpoints, _settings, _tiny_model, start_file  # noqa: E402,F401


class _Counter:
    """A stand-in for a shared multiprocessing Value."""

    def __init__(self, value=0):
        self.value = value


def _league(members, share=1.0, seat_weights=None, weights=None, server_versions=None):
    return {
        "share": share,
        "seat_weights": seat_weights or {1: 0.5, 2: 0.5},
        "members": members,
        "weights": weights or {name: 1.0 for name in members},
        "server_versions": server_versions or {},
    }


def _member(server, mode="search", readouts=4, temperature=1.0, kind="fixed"):
    return {"server": server, "kind": kind, "mode": mode, "readouts": readouts, "prior_temperature": 1.0,
            "temperature": temperature, "value_head": "win"}  # fmt: skip


def _population(tmp_path, members):
    path = tmp_path / "population.json"
    path.write_text(json.dumps({"members": members}))
    return str(path)


# ----------------------------------------------------------------------------- dealing and scoring


def test_league_seats_go_to_one_member_drawn_by_weight():
    league = _league(
        {"a": _member("lg0"), "b": _member("lg1")}, share=1.0, seat_weights={1: 0.5, 2: 0.5}, weights={"a": 1.0, "b": 0}
    )
    rng = random.Random(0)
    counts = []
    for _ in range(200):
        servers, members = sl.deal_seats({"league": league}, rng)
        others = [m for m in members if m != sl.MAIN]
        assert set(others) == {"a"}  # weight 0: never drawn
        assert all((s == "lg0") == (m == "a") for s, m in zip(servers, members))
        assert all((s == "learner") == (m == sl.MAIN) for s, m in zip(servers, members))
        counts.append(len(others))
    assert set(counts) == {1, 2}
    servers, members = sl.deal_seats({"league": dict(league, share=0.0)}, rng)
    assert servers == ["learner"] * 4 and members == [sl.MAIN] * 4


def test_league_result_scores_the_main_systems_seats_against_the_member():
    record = {
        "seat_members": [sl.MAIN, "gen4", sl.MAIN, "gen4"],
        "win_share": [1.0, 0.0, 0.0, 0.0],
        "fractions": [0.4, 0.3, 0.1, 0.2],
    }
    r = sl.league_result(record)
    assert r["member"] == "gen4" and r["main_seats"] == 2
    assert r["win"] == pytest.approx(1.0) and r["worth"] == pytest.approx(0.5)
    assert r["pairwise"] == pytest.approx(0.5)  # seat 0 ahead of both, seat 2 behind both
    assert sl.league_result(dict(record, league_mixed=True)) is None
    assert sl.league_result(dict(record, seat_members=[sl.MAIN] * 4)) is None
    assert sl.league_result(dict(record, seat_members=[sl.MAIN, "a", "b", sl.MAIN])) is None

    scores = {}
    sl.add_league_score(scores, r, "fixed")
    sl.add_league_score(scores, dict(r, win=0.0, worth=0.3), "fixed")
    s = scores["gen4"]
    assert s["games"] == 2 and s["main_seats"] == 4 and s["fair"] == pytest.approx(1.0)
    assert s["win"] == pytest.approx(1.0) and s["worth"] == pytest.approx(0.8)
    assert s["win_excess_sq"] == pytest.approx(0.5**2 + 0.5**2)
    assert s["worth_excess_sq"] == pytest.approx(0.0**2 + 0.2**2)


def test_league_seats_keep_snapshots_in_slots_and_restore_from_state(tmp_path):
    population = _population(
        tmp_path,
        [
            {"name": "gen4", "spec": "p+v"},
            {"name": "rusher", "spec": "q", "mode": "policy", "temperature": 0.5, "min_share": 0.1},
        ],
    )
    cfg = sl.SearchLearningConfig(
        policy_checkpoint="p", value_checkpoint="v", population=population, league_snapshot_slots=2,
        league_readouts=48, prior_temperature=1.0,
    )
    league = sl.LeagueSeats(cfg, None, now=0.0)
    assert set(league.members) == {"gen4", "rusher"}
    assert league.members["rusher"]["mode"] == "policy" and league.members["gen4"]["readouts"] == 48
    specs = league.server_specs("idle")
    assert specs == {"lg0": "p+v", "lg1": "q", "snap0": "idle", "snap1": "idle"}
    assert not league.snapshot_due(1.0) and league.snapshot_due(cfg.league_snapshot_hours * 3600)
    servers = [league.add_snapshot(f"self_u{u}", f"spec{u}", now=float(u)) for u in (10, 20, 30)]
    assert servers == ["snap0", "snap1", "snap0"]  # the third replaces the oldest
    assert set(league.members) == {"gen4", "rusher", "self_u20", "self_u30"}
    assert league.server_specs("idle")["snap0"] == "spec30"
    settings = league.settings()
    assert sum(settings["weights"].values()) == pytest.approx(1.0)
    assert settings["weights"]["rusher"] >= 0.1 - 1e-9  # pinned
    assert settings["members"]["self_u30"]["kind"] == "snapshot"

    record = {"seat_members": [sl.MAIN, "gen4", sl.MAIN, sl.MAIN], "win_share": [0, 1, 0, 0],
              "fractions": [0.2, 0.4, 0.2, 0.2]}  # fmt: skip
    assert league.record(record)
    assert league.take_scores()["gen4"]["games"] == 1 and league.take_scores() == {}
    assert league.league.win_rate("gen4") < 0.5

    again = sl.LeagueSeats(cfg, json.loads(json.dumps(league.state())), now=99.0)
    assert set(again.members) == set(league.members) and again.snapshot_time == 30.0
    assert again.league.win_rate("gen4") == pytest.approx(league.league.win_rate("gen4"))
    league.write(tmp_path / "league.json")
    written = json.loads((tmp_path / "league.json").read_text())["members"]
    assert written["self_u30"]["spec"] == "spec30" and "weight" in written["gen4"]


def test_population_names_are_checked(tmp_path):
    with pytest.raises(SystemExit):
        sl.load_league_members(_population(tmp_path, [{"name": "self_x", "spec": "p"}]), 48)
    with pytest.raises(SystemExit):
        sl.load_league_members(_population(tmp_path, [{"name": "a", "spec": "p", "mode": "fast"}]), 48)
    with pytest.raises(SystemExit):
        sl.load_league_members(_population(tmp_path, []), 48)


def test_member_seats_search_with_their_own_server_and_give_no_rows(start_file):
    settings = _settings(start_file, p_full=1.0, max_decisions=30)
    settings["league"] = _league(
        {"searcher": _member("lg0", readouts=4), "sampler": _member("lg1", mode="policy", temperature=0.5)},
        share=1.0,
    )
    seen = []

    class Recording(ModelEvaluator):
        def __init__(self, name):
            super().__init__(HashNet())
            self.name = name

        def send(self, packed):
            seen.append(self.name)
            return super().send(packed)

    gen = sl.Generator(settings, {name: Recording(name) for name in ("learner", "lg0", "lg1")})
    out = gen.run(8.0)
    assert out["games"] and out["stats"]["league_searches"] > 0
    assert {"learner", "lg0", "lg1"} <= set(seen)
    for g in out["games"]:
        members = g["seat_members"]
        assert len({m for m in members if m != sl.MAIN}) == 1 and members.count(sl.MAIN) in (2, 3)
        main = {seat for seat, m in enumerate(members) if m == sl.MAIN}
        assert {r.rotation for r in g["rows"]} <= main  # rows only from the main system's seats
        assert not g["league_mixed"]
        assert sl.league_result(g) is not None


def test_a_game_whose_member_server_changed_hands_is_not_scored(start_file, monkeypatch):
    counter = _Counter(3)
    monkeypatch.setattr(sl, "_LEAGUE_VERSIONS", {"snap0": counter})
    settings = _settings(start_file)
    settings["league"] = _league({"self_u1": _member("snap0", kind="snapshot")}, server_versions={"snap0": 3})
    gen = sl.Generator(settings, {"learner": ModelEvaluator(HashNet()), "snap0": ModelEvaluator(HashNet())})
    g = gen._new_game()
    assert g.league_version == 3 and not g.league_mixed
    counter.value = 4  # the slot took another snapshot mid-game
    record = gen._finish(g)
    assert record["league_mixed"] and sl.league_result(record) is None
    g = gen._new_game()  # dealt from settings older than the server
    assert g.league_mixed


# ----------------------------------------------------------------------------- value_follow


def _value_run(tmp_path, versions):
    from rl18xx.agent.alphazero.policy_gradient import _save

    run = tmp_path / "value_run"
    (run / "promoted").mkdir(parents=True, exist_ok=True)
    for v in versions:
        if not (run / "promoted" / f"{v}.pth").exists():
            _save(_tiny_model(10 + v), run / "promoted" / f"{v}.pth")
    entries = [{"version": v, "path": f"promoted/{v}.pth", "cycle": v} for v in versions]
    (run / "promoted.json").write_text(json.dumps({"current": versions[-1], "versions": entries}))
    return run


def test_value_follower_takes_the_newest_promoted_value(tmp_path):
    run = _value_run(tmp_path, [0])
    f = sl.ValueFollower(run)
    assert f.poll() and f.version == 0 and f.path == run / "promoted" / "0.pth"
    assert not f.poll()  # unchanged
    (run / "promoted.json").write_text('{"current": 1, "vers')  # half-written: keep the last one
    assert not f.poll() and f.version == 0
    _value_run(tmp_path, [0, 1])
    assert f.poll() and f.version == 1


def test_value_follow_validation():
    base = dict(policy_checkpoint="p", value_checkpoint="")
    with pytest.raises(SystemExit):
        sl._validate(sl.SearchLearningConfig(**base, value_follow="r", shared=True))
    with pytest.raises(SystemExit):
        sl._validate(sl.SearchLearningConfig(**base, freeze_policy=True))  # neither shared nor value_follow
    sl._validate(sl.SearchLearningConfig(**base, freeze_policy=True, value_follow="r"))
    with pytest.raises(SystemExit):
        sl._validate(sl.SearchLearningConfig(**base, population="x", league_share=1.5))


def test_a_frozen_run_with_a_followed_value_and_league_seats(tmp_path, start_file, monkeypatch):
    """Generation only: the frozen prior's checkpoints are links, the followed value's promotion
    reaches the server, league members (a searcher, a sampler, the main system's snapshot) play
    their seats, every game is saved with its seat members, and the history scores each member."""
    from rl18xx.agent.alphazero import search_data

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    policy, critic = _save_checkpoints(tmp_path)
    value_run = _value_run(tmp_path, [0])
    population = _population(
        tmp_path,
        [
            {"name": "searcher", "spec": f"{policy}+{critic}", "readouts": 3},
            {"name": "sampler", "spec": str(policy), "mode": "policy", "temperature": 0.5},
        ],
    )
    polls = []
    original = sl.ValueFollower.poll

    def poll(self):
        polls.append(1)
        if len(polls) == 2:  # the first update's check: the value run promotes version 1 just before
            _value_run(tmp_path, [0, 1])
        return original(self)

    monkeypatch.setattr(sl.ValueFollower, "poll", poll)
    cfg = sl.SearchLearningConfig(
        policy_checkpoint=str(policy), value_checkpoint="", value_follow=str(value_run), freeze_policy=True,
        out_dir=str(tmp_path / "runs"), run_name="frozen", p_full=0.5, n_full=4, parallel_readouts=2, workers=2,
        games_per_worker=3, task_seconds=2.0, max_decisions=16, start_positions=start_file,
        random_start_fraction=1.0, inference_batch_size=64, rows_per_update=8, max_updates=3, snapshot_every=1,
        anchor_every=0, save_data=str(tmp_path / "saved"), save_data_shard_games=2, population=population,
        league_share=1.0, league_seat_weights={1: 1.0}, league_snapshot_hours=1e-9, league_snapshot_slots=1,
        league_readouts=3,
    )
    run_dir = sl.run(cfg)
    history = [json.loads(line) for line in (run_dir / "history.jsonl").read_text().splitlines()]
    assert [h["update"] for h in history] == [1, 2, 3]
    assert history[0]["value_version"] == 1 and history[0]["value_reloaded"]
    assert all(h["rows_trained"] == 0 and h["league_share"] == 1.0 for h in history)
    assert not history[-1]["value_reloaded"]
    scored = {}
    for h in history:
        for name, s in h["league"].items():
            scored[name] = scored.get(name, 0) + s["games"]
            assert s["kind"] == ("snapshot" if name.startswith(sl.SNAPSHOT_PREFIX) else "fixed")
            assert s["fair"] == pytest.approx(s["games"] * 3 / 4)  # one member seat per game
    assert {"searcher", "sampler"} <= set(scored)
    inodes = {os.stat(run_dir / "learner" / f"{u}.pth").st_ino for u in range(4)}
    assert len(inodes) == 1  # the frozen prior: one file, linked per update
    assert not (run_dir / "critic").exists()
    league = json.loads((run_dir / "league.json").read_text())["members"]
    snapshots = [n for n in league if n.startswith(sl.SNAPSHOT_PREFIX)]
    assert len(snapshots) == 1 and str(value_run / "promoted") in league[snapshots[0]]["spec"]
    state = json.loads((run_dir / "state.json").read_text())
    assert state["league"]["snapshots"][0]["name"] == snapshots[0]

    games = list(search_data.iter_games(tmp_path / "saved"))
    assert games and all(g.meta["value_version"] in (0, 1) for g in games)
    assert all(sum(m != sl.MAIN for m in g.meta["seat_members"]) == 1 for g in games)
