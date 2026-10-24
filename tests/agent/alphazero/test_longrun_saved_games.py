"""The long run's daily jobs on saved games: scripts/longrun/style_stats.py (style statistics and the drift guard)
and scripts/longrun/integrity_check.py (rows rebuilt twice bit for bit, shard checksums) -- on games a tiny CPU
generator (HashNet) plays and saves, plus the statistics' aggregation on synthetic game summaries."""

import argparse
import dataclasses
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import pytest

engine_rs = pytest.importorskip("engine_rs")

from rl18xx.agent.alphazero import search_data as sd  # noqa: E402
from rl18xx.agent.alphazero import search_learning as sl  # noqa: E402
from rl18xx.agent.alphazero.lockstep import ModelEvaluator  # noqa: E402

from .search_test_utils import HashNet  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
LONGRUN = REPO_ROOT / "scripts" / "longrun"


def _load(name):
    sys.path.insert(0, str(LONGRUN))
    spec = importlib.util.spec_from_file_location(name, LONGRUN / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ST = _load("style_stats")
IC = _load("integrity_check")
LEAGUE = ["main", "hunter", "main", "hunter"]


@pytest.fixture(scope="module")
def saved(tmp_path_factory):
    """Shards of generator games (two of them with league seats) and the packed games."""
    tmp = tmp_path_factory.mktemp("saved")
    starts = tmp / "starts.jsonl"
    starts.write_text("")
    cfg = sl.SearchLearningConfig(
        policy_checkpoint="p", value_checkpoint="v", learner_temperature=0.5, prior_temperature=0.5,
        start_positions=str(starts), random_start_fraction=1.0, n_full=4, p_full=0.3, games_per_worker=4,
        max_decisions=300, parallel_readouts=4, save_data="unused",
    )
    gen = sl.Generator(sl.worker_settings(cfg, version=3), {"learner": ModelEvaluator(HashNet())})
    games, started = [], time.time()
    while len(games) < 8 and time.time() - started < 60:
        games += gen.run(3.0)["games"]
    packed = [g["saved"] for g in games]
    for p in packed[:2]:
        p["meta"]["seat_members"] = list(LEAGUE)
    data = tmp / "data"
    with sd.SearchDataWriter(data, dataclasses.asdict(cfg), 3) as writer:
        for p in packed:
            writer.add(p)
    return data, packed, cfg


def _args(data, out, **kw):
    base = dict(data=data, out=out, hours=24.0, since=None, until=None, max_games=0, seed=0, baseline=None,
                min_pars=1, min_dividends=1)
    return argparse.Namespace(**{**base, **kw})


def test_style_stats_on_saved_games(saved, tmp_path):
    data, packed, _ = saved
    record = ST.run(_args(data, tmp_path / "style.jsonl"))
    assert record["games"] == record["games_in_window"] == len(packed) and record["league_games"] == 2
    assert not record["errors"]
    # Pars after the start, counted straight from the logs: every seat, and the main system's seats.
    pars_all = pars_main = 0
    for p in packed:
        actions = json.loads(p["actions"])
        members = p["meta"].get("seat_members") or ["main"] * 4
        for a in actions[int(p["meta"]["start_actions"]):]:
            if a["type"] == "par":
                pars_all += 1
                pars_main += members[int(a["entity"]) - 1] == "main"
    assert record["all"]["pars"] == pars_all and record["main"]["pars"] == pars_main
    assert record["main"]["seats"] == 4 * len(packed) - 4 and record["main"]["games"] == len(packed) - 2
    assert abs(sum(record["all"]["par_share"].values()) - 1) < 1e-9
    assert 0 <= record["all"]["withhold_rate"] <= 1 and record["all"]["dividend_decisions"] > 0
    # Train timing: each type first bought no earlier than the type before it.
    summaries = [ST.game_summary(g) for f in sd.shard_paths(data) for g in sd.read_shard(f)]
    for s in summaries:
        ors = [s["first_train"][t][0] for t in ST.TRAIN_TYPES if t in s["first_train"]]
        assert ors == sorted(ors) and all(o >= 1 for o in ors)
        assert s["first_train"]["2"][0] >= 1
    assert record["all"]["length_cap_share"] == sum(p["meta"]["termination"] == "max_length" for p in packed) / len(
        packed)
    # A second run against the first as its baseline: deltas and the drift guard's withhold flag.
    again = ST.run(_args(data, tmp_path / "style.jsonl", baseline="first"))
    assert again["baseline"]["time"] == record["time"]
    assert again["compare"]["withhold_rate"]["change"] == pytest.approx(0.0)
    assert again["drift"]["withhold"]["flag"] is False and again["drift"]["par_share"]["flag"] is False
    assert len(ST.read_jsonl(tmp_path / "style.jsonl")) == 2
    # The window: shards written before it are left out.
    old = time.time() - 3 * 86400
    for f in sd.shard_paths(data)[:1]:
        os.utime(f, (old, old))
    windowed = ST.run(_args(data, None))
    assert windowed["games_in_window"] == len(packed) - len(sd.read_shard(sd.shard_paths(data)[0]))
    for f in sd.shard_paths(data)[:1]:
        f.touch()


def _summary(members, pars=(), dividends=(), bankrupt=(), first=None, termination="finished", decisions=300,
             start_kind="random"):
    return {"uid": "g", "start_kind": start_kind, "num_players": 4, "members": list(members),
            "termination": termination, "decisions": decisions, "actions": decisions + 30, "value_version": 1,
            "pars": [list(p) for p in pars], "dividends": [list(d) for d in dividends], "bankrupt": list(bankrupt),
            "first_train": first or {}, "operating_rounds": 10}


def test_aggregate_scopes_and_the_drift_guard():
    main4 = ["main"] * 4
    games = [
        _summary(main4, pars=[(0, "B&O", 100), (1, "PRR", 82)], dividends=[(0, "payout"), (1, "withhold")],
                 first={"3": [4, 200, 0], "D": [12, 700, 1]}),
        _summary(main4, pars=[(2, "PRR", 90)], dividends=[(2, "payout")], bankrupt=[3],
                 first={"3": [6, 260, 2]}, termination="max_length"),
        _summary(LEAGUE, pars=[(0, "PRR", 67), (1, "NYC", 100)], dividends=[(1, "withhold"), (0, "withhold")],
                 bankrupt=[1], first={"3": [2, 100, 1]}),
        _summary(main4, first={"3": [1, 50, 0]}, start_kind="midgame"),  # its trains came from a human prefix
    ]
    main, every = ST.aggregate(games, "main"), ST.aggregate(games, "all")
    assert main["pars"] == 4 and main["par_share"] == {"PRR": 0.75, "B&O": 0.25}
    assert main["par_share_max"] == {"corporation": "PRR", "share": 0.75}
    assert main["withhold_rate"] == pytest.approx(2 / 4) and every["withhold_rate"] == pytest.approx(3 / 5)
    assert main["seats"] == 14 and main["bankruptcies"] == 1 and main["bankruptcies_per_game"] == pytest.approx(1 / 3.5)
    assert every["bankruptcies"] == 2 and every["bankruptcies_per_game"] == pytest.approx(2 / 4)
    assert main["games"] == 3 and main["whole_games"] == 2  # the league game and the mid-game start are out
    assert main["train_first"]["3"]["or"]["mean"] == pytest.approx(5.0)
    assert main["train_first"]["3"]["bought_share"] == 1
    assert main["train_first"]["D"]["bought_share"] == pytest.approx(0.5)
    assert every["train_first"]["3"]["or"]["n"] == 3
    assert main["length_cap_share"] == pytest.approx(1 / 3)
    flags = ST.drift(main, None, min_pars=4)
    assert flags["par_share"]["flag"] is True and flags["withhold"]["flag"] is None  # no baseline: n/a
    assert ST.drift(main, None, min_pars=5)["par_share"]["flag"] is None  # too few pars to judge
    base = {**main, "withhold_rate": 0.2, "dividend_decisions": 100}
    now = {**main, "withhold_rate": 0.45, "dividend_decisions": 100}
    assert ST.drift(now, base)["withhold"]["flag"] is True
    assert ST.drift({**now, "withhold_rate": 0.39}, base)["withhold"]["flag"] is False


def test_load_baseline(tmp_path):
    out = tmp_path / "style.jsonl"
    rows = [{"time": "a", "window": {"until": "2026-11-01T06:00:00"}}, {"time": "b", "window": {"until":
            "2026-11-02T06:00:00"}}, {"time": "c", "window": {"until": "2026-11-02T18:00:00"}}]
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert ST.load_baseline("first", out)["time"] == "a"
    assert ST.load_baseline("2026-11-02", out)["time"] == "c"
    assert ST.load_baseline("2026-11-05", out) is None
    single = tmp_path / "base.json"
    single.write_text(json.dumps(rows[1]))
    assert ST.load_baseline(str(single), out)["time"] == "b"
    assert ST.load_baseline(str(out), tmp_path / "other.jsonl")["time"] == "c"
    assert ST.load_baseline(None, out) is None


def _integrity(data, tmp_path, *extra):
    argv = ["integrity_check.py", "--data", str(data), "--games", "4", "--log", str(tmp_path / "integrity.jsonl"),
            "--checksums", str(tmp_path / "sums.txt"), "--seed", "1", *extra]
    old = sys.argv
    sys.argv = argv
    try:
        return IC.main()
    finally:
        sys.argv = old


def test_integrity_check(saved, tmp_path, monkeypatch):
    data, packed, cfg = saved
    monkeypatch.setenv("WORKDIR", str(tmp_path))
    monkeypatch.setenv("STATE_DIR", "state")
    monkeypatch.setenv("LONGRUN_INSTANCE", "lr1-daily")
    monkeypatch.setenv("ALERT_COMMAND", "")
    assert _integrity(data, tmp_path) == 0
    first = json.loads((tmp_path / "integrity.jsonl").read_text().splitlines()[-1])
    assert first["ok"] and first["games"] == 4 and first["rows"] > 0
    assert first["checksums_listed"] == len(sd.shard_paths(data))
    assert _integrity(data, tmp_path) == 0  # every shard listed now: re-hashed, unchanged
    assert json.loads((tmp_path / "integrity.jsonl").read_text().splitlines()[-1])["checksums_verified"] > 0

    # A game whose saved row doesn't match its replay (the mover's rotation changed): a mismatch, exit 1, an alert.
    bad_dir = tmp_path / "bad"
    game = sd.read_shard(sd.shard_paths(data)[0])[0]
    bad = game.packed()
    bad["columns"] = dict(bad["columns"])
    rotations = bad["columns"]["row_rotation"].copy()
    assert len(rotations) > 0
    rotations[0] = (int(rotations[0]) + 1) % 4
    bad["columns"]["row_rotation"] = rotations
    bad_dir.mkdir()
    sd.write_shard(bad_dir / "x-00000.sld", [bad], dataclasses.asdict(cfg))
    assert _integrity(bad_dir, tmp_path, "--games", "1") == 1
    last = json.loads((tmp_path / "integrity.jsonl").read_text().splitlines()[-1])
    assert not last["ok"] and "doesn't replay" in last["mismatches"][0]["error"]
    assert "CRIT [lr1-daily] integrity" in (tmp_path / "state" / "alerts.log").read_text()

    # A listed shard that changed or vanished.
    sums = tmp_path / "sums2.txt"
    copy = tmp_path / "copy"
    copy.mkdir()
    for f in sd.shard_paths(data):
        (copy / f.name).write_bytes(f.read_bytes())
    assert IC.checksums(copy, sums, 0, __import__("random").Random(0))["problems"] == []
    victim = sd.shard_paths(copy)[0]
    victim.write_bytes(victim.read_bytes()[:-1] + b"\0")
    result = IC.checksums(copy, sums, 100, __import__("random").Random(0))
    assert any("changed" in p for p in result["problems"])
    victim.unlink()
    assert any("gone" in p for p in IC.checksums(copy, sums, 100, __import__("random").Random(0))["problems"])
    assert IC.checksums(copy, sums, 100, __import__("random").Random(0), allow_missing=True)["missing"]


def test_rebuild_comparison_is_bit_for_bit():
    import numpy as np

    a = np.array([1.0, 2.0], dtype=np.float32)
    assert IC.same_value(a, a.copy())
    assert not IC.same_value(a, a.astype(np.float64))
    assert not IC.same_value(np.array([0.0], dtype=np.float32), np.array([-0.0], dtype=np.float32))  # bits differ
    assert IC.same_value((1, 2, a), (1, 2, a.copy())) and not IC.same_value((1, a), (2, a))
