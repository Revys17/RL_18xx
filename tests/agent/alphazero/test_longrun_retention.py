"""scripts/longrun/retention.py on temporary workdirs: snapshot thinning by age, the kept references (ladder panel /
ledger / state, league files, the generator's resume snapshot and anchor), unpromoted value candidates, optimizer
state, log rotation, dry-run dates -- and the hard guards: a config asking to delete saved games, promoted values,
eval collections or the ladder ledger is refused and nothing at all is deleted."""

import importlib.util
import json
import os
import subprocess
import sys
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
LONGRUN = REPO_ROOT / "scripts" / "longrun"
spec = importlib.util.spec_from_file_location("retention", LONGRUN / "retention.py")
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)

NOW = 1_800_000_000.0
HOUR = 3600.0


def touch(path, age_hours=0.0, data=b"x" * 100, now=NOW):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    os.utime(path, (now - age_hours * HOUR, now - age_hours * HOUR))
    return path


def snapshot_config(**extra):
    return {
        "snapshots": [{"dir": "gen", "files": ["learner/{n}.pth", "critic/{n}.pth"],
                       "keep": [{"hours": 48, "every": 10}, {"hours": 336, "every": 100}, {"every": 1000}]}],
        **extra,
    }


def plan_paths(config, workdir):
    plan = R.Retention(config, workdir, NOW).plan()
    return {str(p.relative_to(Path(workdir).resolve())) for _, p in plan["delete"]}, plan


def test_snapshot_thinning_by_age(tmp_path):
    # update n written (3000 - n) / 10 hours ago: 2990 is 1 h old, 2500 50 h, 1000 200 h, 0 300 h.
    for n in range(0, 3000, 10):
        touch(tmp_path / "gen" / "learner" / f"{n}.pth", (3000 - n) / 10)
        touch(tmp_path / "gen" / "critic" / f"{n}.pth", (3000 - n) / 10)
    touch(tmp_path / "gen" / "learner" / "2995.pth", 0.1)  # the newest (not a multiple of 10)
    deleted, plan = plan_paths(snapshot_config(), tmp_path)
    kept = {n for n in list(range(0, 3000, 10)) + [2995] if f"gen/learner/{n}.pth" not in deleted}
    expected = set(range(2520, 3000, 10)) | {2995}  # every 10th for 48 h (ages up to 48 h: n >= 2520)
    expected |= {n for n in range(0, 3000, 100) if (3000 - n) / 10 <= 336}  # every 100th for two weeks: all here
    expected |= {0, 1000, 2000}
    assert kept == expected
    assert "gen/critic/2510.pth" in deleted and "gen/critic/2500.pth" not in deleted


def test_references_are_kept(tmp_path):
    for n in (110, 120, 130, 140, 150):
        touch(tmp_path / "gen" / "learner" / f"{n}.pth", 400)
        touch(tmp_path / "gen" / "critic" / f"{n}.pth", 400)
    touch(tmp_path / "gen" / "learner" / "160.pth", 1)
    (tmp_path / "gen" / "state.json").write_text(json.dumps({"snapshot_update": 150, "anchor_update": 130,
                                                             "config": {"policy_checkpoint": "start/0.pth"}}))
    # A ladder whose ledger and panel name 110; a league file naming 120's learner + critic.
    panel = {"name": "lad", "run": "gen", "checkpoint": "learner/{update}.pth", "spec": "{checkpoint}@200",
             "panel": [{"name": "x", "spec": "gen/learner/140.pth@200"}]}
    (tmp_path / "panel.json").write_text(json.dumps(panel))
    touch(tmp_path / "logs" / "eval" / "lad" / "ledger.jsonl", data=json.dumps(
        {"a_spec": "gen/learner/110.pth@200", "b_spec": "x@200"}).encode() + b"\n")
    (tmp_path / "gen" / "league.json").write_text(json.dumps({"members": {"self_u120": {
        "spec": "gen/learner/120.pth+gen/critic/120.pth#score", "kind": "snapshot"}}}))
    config = snapshot_config(protect={"panels": ["panel.json"], "league_files": ["gen/league.json"],
                                      "generator_runs": ["gen"]})
    deleted, plan = plan_paths(config, tmp_path)
    assert deleted == {"gen/critic/110.pth", "gen/critic/130.pth", "gen/critic/140.pth"}  # the anchor: learner only
    assert not plan["refused"]


def test_value_candidates_league_links_optimizer_and_logs(tmp_path):
    for cycle, age in ((1, 200), (2, 200), (3, 200), (4, 10)):
        touch(tmp_path / "vrun" / "candidates" / f"{cycle}.pth", age)
    with open(tmp_path / "vrun" / "cycles.jsonl", "w") as f:
        for cycle, promoted in ((1, False), (2, True), (4, False)):  # 3: not recorded -> kept
            f.write(json.dumps({"cycle": cycle, "promoted": promoted}) + "\n")
    touch(tmp_path / "gen" / "league" / "self_u100_v3" / "learner" / "100.pth", 400)
    touch(tmp_path / "gen" / "league" / "self_u900_v9" / "learner" / "900.pth", 400)
    touch(tmp_path / "gen" / "league" / "self_u950_v9" / "learner" / "950.pth", 24)  # left, but young
    (tmp_path / "gen" / "league.json").write_text(json.dumps({"members": {"self_u900_v9": {
        "spec": "gen/league/self_u900_v9/learner/900.pth+vrun/promoted/9.pth"}}}))
    for name, age in (("optimizer.a.pth", 24 * 20), ("optimizer.b.pth", 24 * 19), ("optimizer.c.pth", 24 * 2),
                      ("optimizer.pth", 1)):
        touch(tmp_path / "gen" / name, age)
    touch(tmp_path / "logs" / "big.log", data=b"y" * 2000)
    touch(tmp_path / "logs" / "big.log.1")
    touch(tmp_path / "logs" / "big.log.4")
    touch(tmp_path / "logs" / "big.log.5")
    touch(tmp_path / "logs" / "small.log", data=b"z" * 10)
    config = {
        "protect": {"league_files": ["gen/league.json"]},
        "value_candidates": [{"dir": "vrun/candidates", "cycles": "vrun/cycles.jsonl", "days": 7}],
        "unreferenced": [{"glob": "gen/league/**/*.pth", "min_age_days": 7}],
        "optimizer_state": [{"glob": "gen/optimizer*.pth"}],
        "logs": [{"glob": "logs/*.log", "max_bytes": 1000, "rotations": 4}],
    }
    deleted, plan = plan_paths(config, tmp_path)
    ages = {"optimizer.a.pth": 24 * 20, "optimizer.b.pth": 24 * 19, "optimizer.c.pth": 24 * 2, "optimizer.pth": 1}
    newest_of_week = {}
    for name, age in sorted(ages.items(), key=lambda kv: -kv[1]):  # oldest first: the newest of a week wins
        newest_of_week[date.fromtimestamp(NOW - age * HOUR).isocalendar()[:2]] = name
    expect_opt = {f"gen/{n}" for n in ages if n not in newest_of_week.values() and n != "optimizer.pth"}
    assert expect_opt  # (a, b, c and the newest can't all fall in different weeks)
    expected = {"vrun/candidates/1.pth", "gen/league/self_u100_v3/learner/100.pth", "logs/big.log.5"}
    assert deleted == expected | expect_opt
    assert [(p.name, size) for p, size, _ in plan["rotate"]] == [("big.log", 2000)]
    summary = R.run(config, tmp_path, dry_run=False, now=NOW, echo=lambda *_: None)
    assert not (tmp_path / "vrun" / "candidates" / "1.pth").exists()
    assert all((tmp_path / "vrun" / "candidates" / f"{c}.pth").exists() for c in (2, 3, 4))
    assert (tmp_path / "logs" / "big.log").stat().st_size == 0
    assert (tmp_path / "logs" / "big.log.1").read_bytes() == b"y" * 2000
    assert (tmp_path / "logs" / "big.log.2").exists() and not (tmp_path / "logs" / "big.log.5").exists()
    assert summary["files"] == len(deleted) and summary["rotated"][0]["bytes"] == 2000


def test_dry_run_dates(tmp_path):
    assert R.is_dry_run({}, today=date(2026, 12, 1))  # no date: always a dry run
    assert R.is_dry_run({"dry_run_until": "2026-11-08"}, today=date(2026, 11, 8))
    assert not R.is_dry_run({"dry_run_until": "2026-11-08"}, today=date(2026, 11, 9))
    assert R.is_dry_run({"dry_run_until": "2026-11-08"}, today=date(2026, 11, 9), force_dry=True)
    assert not R.is_dry_run({}, execute=True)
    for n in (1, 2, 3):
        touch(tmp_path / "gen" / "learner" / f"{n}.pth", 400)
    summary = R.run(snapshot_config(), tmp_path, dry_run=True, now=NOW, echo=lambda *_: None)
    assert summary["dry_run"] and summary["files"] == 2 and summary["bytes"] == 200
    assert all((tmp_path / "gen" / "learner" / f"{n}.pth").exists() for n in (1, 2, 3))
    assert "would delete 2 files" in R.summary_line(summary)


def _guarded_workdir(tmp_path, now=NOW):
    touch(tmp_path / "saved_games" / "lr1" / "20261101-1-00000.sld", 900, now=now)
    touch(tmp_path / "saved_games" / "lr1" / "20261101-1-00001.sld", 900, now=now)
    touch(tmp_path / "vrun" / "promoted" / "3.pth", 900, now=now)
    touch(tmp_path / "vrun" / "promoted" / "4.pth", 900, now=now)
    touch(tmp_path / "logs" / "eval" / "ladder_lr1" / "ledger.jsonl", 900, now=now)
    touch(tmp_path / "logs" / "eval" / "ladder_lr1" / "ladder.log", 900, data=b"l" * 5000, now=now)
    for n in (1, 2, 3):
        touch(tmp_path / "gen" / "learner" / f"{n}.pth", 400, now=now)  # a valid rule's deletions


def test_guards_refuse_saved_games_promoted_values_evals_and_the_ledger(tmp_path):
    _guarded_workdir(tmp_path)
    bad_configs = [
        {"snapshots": [{"dir": "saved_games", "files": ["lr1/{n}.sld"], "keep": []}]},
        {"unreferenced": [{"glob": "saved_games/**/*.sld", "min_age_days": 10000}]},  # refused though too young
        {"unreferenced": [{"glob": "saved_games/**/*", "min_age_days": 1}]},
        {"snapshots": [{"dir": "vrun/promoted", "files": ["{n}.pth"], "keep": []}]},
        {"value_candidates": [{"dir": "vrun/promoted", "cycles": "vrun/cycles.jsonl", "days": 0}]},
        {"unreferenced": [{"glob": "logs/eval/**/*", "min_age_days": 1}]},
        {"logs": [{"glob": "logs/eval/*/*.log", "max_bytes": 10, "rotations": 1}]},
        {"optimizer_state": [{"glob": "logs/eval/ladder_lr1/ledger.jsonl"}]},
    ]
    (tmp_path / "vrun" / "cycles.jsonl").write_text(json.dumps({"cycle": 3, "promoted": False}) + "\n")
    before = sorted(str(p) for p in tmp_path.rglob("*") if p.is_file())
    for bad in bad_configs:
        config = {**snapshot_config(), **{k: v for k, v in bad.items()}} if "snapshots" not in bad else {
            "snapshots": snapshot_config()["snapshots"] + bad["snapshots"]}
        summary = R.run(config, tmp_path, dry_run=False, now=NOW, echo=lambda *_: None)
        assert summary["refused"], bad
        assert summary["dry_run"]  # a refusal turns the whole run into a listing
        assert sorted(str(p) for p in tmp_path.rglob("*") if p.is_file()) == before, bad
    # The same workdir with only the valid rule deletes for real.
    summary = R.run(snapshot_config(), tmp_path, dry_run=False, now=NOW, echo=lambda *_: None)
    assert not summary["refused"] and summary["files"] == 2


def test_guards_symlinks_and_outside_paths(tmp_path):
    outside = tmp_path / "outside"
    touch(outside / "learner" / "1.pth", 400)
    touch(outside / "learner" / "2.pth", 400)
    work = tmp_path / "work"
    work.mkdir()
    (work / "linked").symlink_to(outside)
    r = R.Retention({}, work, NOW)
    assert "symlink" in r.guard(work / "linked" / "learner" / "1.pth")
    assert r.guard(outside / "learner" / "1.pth") == "outside the workdir"
    summary = R.run({"snapshots": [{"dir": "linked", "files": ["learner/{n}.pth"], "keep": []}]}, work,
                    dry_run=False, now=NOW, echo=lambda *_: None)
    assert summary["refused"] and (outside / "learner" / "1.pth").exists()
    assert r.guard(work / "gen" / "history.jsonl") == "a run's metadata"
    touch(work / "evals" / "m1" / "games.jsonl")
    assert r.guard(work / "evals" / "m1" / "games" / "0.json") == "an eval collection"


def test_cli_refusal_exits_3_and_alerts(tmp_path):
    import time

    _guarded_workdir(tmp_path, now=time.time())
    config = tmp_path / "retention.json"
    config.write_text(json.dumps({"dry_run_until": "2000-01-01",
                                  "unreferenced": [{"glob": "saved_games/**/*.sld", "min_age_days": 0}]}))
    env = {**os.environ, "STATE_DIR": "longrun/daily", "LONGRUN_INSTANCE": "lr1-daily", "ALERT_COMMAND": ""}
    out = subprocess.run([sys.executable, str(LONGRUN / "retention.py"), "--config", str(config), "--workdir",
                          str(tmp_path), "--log", "longrun/daily/retention.jsonl"], capture_output=True, text=True,
                         env=env)
    assert out.returncode == 3, out.stdout + out.stderr
    assert "REFUSED" in out.stdout and (tmp_path / "saved_games" / "lr1" / "20261101-1-00000.sld").exists()
    assert "WARN [lr1-daily] retention" in (tmp_path / "longrun" / "daily" / "alerts.log").read_text()
    line = json.loads((tmp_path / "longrun" / "daily" / "retention.jsonl").read_text().splitlines()[-1])
    assert line["refused"][0]["why"] == "saved games"


def test_the_runs_retention_config_is_valid():
    config = json.loads((REPO_ROOT / "scripts" / "launch" / "2026-11" / "retention.json").read_text())
    for key in ("snapshots", "value_candidates", "unreferenced", "optimizer_state", "logs", "protect"):
        assert key in config
    assert config["snapshots"][0]["keep"] == [{"hours": 48, "every": 10}, {"hours": 336, "every": 100},
                                              {"every": 1000}]
    date.fromisoformat(config["dry_run_until"])
