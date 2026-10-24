"""The long-run harness's shell side without systemd: install.sh / install_lr1.sh writing the four instances' config
(SYSTEMCTL=true), launch.sh's command line for each KIND (a stub deployment whose python records its arguments), and
daily.sh's job order and step record."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
LONGRUN = REPO_ROOT / "scripts" / "longrun"
LAUNCH = REPO_ROOT / "scripts" / "launch" / "2026-11"

STUB_PYTHON = """#!/bin/sh
case "$*" in *snapshot_check.py*) echo 10; exit 0 ;; esac
echo "$@" >> "$STUB_LOG"
[ -n "$STUB_OUT" ] && echo "$STUB_OUT"
case "$*" in *"$STUB_FAIL"*) [ -n "$STUB_FAIL" ] && exit 3 ;; esac
exit ${STUB_RC:-0}
"""


def env(**extra):
    base = {"PATH": os.environ["PATH"], "HOME": os.environ.get("HOME", "/tmp"), "LANG": "C.UTF-8"}
    return {**base, **{k: str(v) for k, v in extra.items()}}


@pytest.fixture
def deployment(tmp_path):
    """A stub deployment: DEPLOYMENT.json, a .venv python that records its arguments, the real launch.sh / daily.sh,
    a deploy.sh whose verify passes, the run's launch files."""
    d = tmp_path / "deploy"
    (d / ".venv" / "bin").mkdir(parents=True)
    (d / "scripts" / "longrun").mkdir(parents=True)
    (d / "DEPLOYMENT.json").write_text(json.dumps({"commit": "abc123"}))
    (d / ".longrun.lock").touch()
    py = d / ".venv" / "bin" / "python"
    py.write_text(STUB_PYTHON)
    py.chmod(0o755)
    for name in ("launch.sh", "daily.sh", "install.sh", "rl18xx-longrun@.service", "rl18xx-longrun-watchdog@.service",
                 "rl18xx-longrun-daily@.service", "rl18xx-longrun-daily@.timer", "watchdog.py"):
        shutil.copy(LONGRUN / name, d / "scripts" / "longrun" / name)
    (d / "scripts" / "longrun" / "deploy.sh").write_text("#!/bin/bash\nexit 0\n")
    shutil.copytree(LAUNCH, d / "scripts" / "launch" / "2026-11")
    work = tmp_path / "work"
    work.mkdir()
    return d, work


def test_install_lr1_writes_all_four_instances(deployment, tmp_path):
    d, work = deployment
    config = tmp_path / "config"
    out = subprocess.run(["bash", str(d / "scripts" / "launch" / "2026-11" / "install_lr1.sh"), str(d), str(work)],
                         env=env(XDG_CONFIG_HOME=config, SYSTEMCTL="true"), capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    envs = {}
    for name in ("lr1-gen", "lr1-value", "lr1-ladder", "lr1-daily"):
        values = {}
        for line in (config / "rl18xx-longrun" / f"{name}.env").read_text().splitlines():
            key, sep, value = line.partition("=")
            if sep and not key.startswith("#"):
                values[key] = value[1:-1]
        envs[name] = values
    assert envs["lr1-gen"]["KIND"] == "generator" and "--freeze-policy" in envs["lr1-gen"]["START_ARGS"]
    assert "--value-follow model_checkpoints_value/lr1" in envs["lr1-gen"]["START_ARGS"]
    assert "--save-data saved_games/lr1" in envs["lr1-gen"]["TRAIN_ARGS"]
    assert f"--population {d}/scripts/launch/2026-11/league_longrun.json" in envs["lr1-gen"]["TRAIN_ARGS"]
    assert envs["lr1-gen"]["HOST_CHECKS"] == "1" and envs["lr1-value"]["HOST_CHECKS"] == "0"
    assert envs["lr1-value"]["KIND"] == "value" and envs["lr1-value"]["RUN_DIR"] == "model_checkpoints_value/lr1"
    assert envs["lr1-ladder"]["PANEL"].endswith("eval_ladder_longrun.json")
    assert envs["lr1-ladder"]["RUN_DIR"] == "logs/eval/ladder_lr1"  # the panel's out root (name ladder_lr1)
    panel = json.loads((LAUNCH / "eval_ladder_longrun.json").read_text())
    assert envs["lr1-ladder"]["RUN_DIR"] == f"logs/eval/{panel['name']}"
    assert envs["lr1-daily"]["KIND"] == "daily" and envs["lr1-daily"]["DAILY"].endswith("scripts/longrun/daily.sh")
    assert envs["lr1-daily"]["WATCH_DIRS"] == "longrun/lr1-gen:longrun/lr1-value:longrun/lr1-ladder"
    assert "LAUNCHER" not in envs["lr1-daily"] and "RUN_DIR" not in envs["lr1-daily"]
    units = config / "systemd" / "user"
    for template in ("rl18xx-longrun@.service", "rl18xx-longrun-watchdog@.service", "rl18xx-longrun-daily@.service",
                     "rl18xx-longrun-daily@.timer"):
        assert (units / template).exists()
    limits = (units / "rl18xx-longrun@lr1-gen.service.d" / "10-limits.conf").read_text()
    assert "MemoryMax=135G" in limits and "RestartSec=120" in limits
    daily = (units / "rl18xx-longrun-daily@lr1-daily.service.d" / "10-limits.conf").read_text()
    assert "MemoryMax=16G" in daily and "RestartSec" not in daily
    # The run's memory caps stay under ~230 GB together (docs/longrun_runbook.md).
    caps = []
    for n in ("lr1-gen", "lr1-value", "lr1-ladder"):
        lines = (units / f"rl18xx-longrun@{n}.service.d" / "10-limits.conf").read_text().splitlines()
        caps.append(int(next(line for line in lines if line.startswith("MemoryMax="))[10:-1]))
    assert sum(caps) <= 230
    # The example env files are what the script writes (paths and timestamps aside).
    for name, values in envs.items():
        example = {}
        for line in (LAUNCH / "env-examples" / f"{name}.env").read_text().splitlines():
            key, sep, value = line.partition("=")
            if sep and not key.startswith("#"):
                example[key] = value[1:-1]
        norm = lambda v: v.replace(str(d), "D").replace(str(work), "W")  # noqa: E731
        ex = lambda v: v.replace("/home/revys/longrun/deploy/longrun-2026-11", "D").replace(  # noqa: E731
            "/home/revys/longrun/work/lr1", "W")
        assert {k: norm(v) for k, v in values.items()} == {k: ex(v) for k, v in example.items()}, name
    # Uninstall: config gone, templates removed with the last instance (a stub systemctl: nothing is active).
    stub = tmp_path / "systemctl"
    stub.write_text('#!/bin/sh\ncase "$*" in *is-active*) exit 3 ;; esac\nexit 0\n')
    stub.chmod(0o755)
    for name in ("lr1-gen", "lr1-value", "lr1-ladder", "lr1-daily"):
        out = subprocess.run(["bash", str(d / "scripts" / "longrun" / "install.sh"), "--uninstall", name],
                             env=env(XDG_CONFIG_HOME=config, SYSTEMCTL=stub), capture_output=True, text=True)
        assert out.returncode == 0, out.stderr
    assert not list((config / "rl18xx-longrun").glob("*.env")) and not (units / "rl18xx-longrun@.service").exists()


def _launch(d, work, tmp_path, **settings):
    log = tmp_path / "stub.log"
    out = subprocess.run(["bash", str(d / "scripts" / "longrun" / "launch.sh"), "start"], capture_output=True,
                         text=True, env=env(DEPLOY=d, WORKDIR=work, LONGRUN_INSTANCE="t", STUB_LOG=log, **settings))
    calls = log.read_text().splitlines() if log.exists() else []
    log.unlink(missing_ok=True)
    return out.returncode, calls, out


def _records(work, name):
    path = work / "longrun" / "t" / name
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def test_launch_command_for_each_kind(deployment, tmp_path):
    d, work = deployment
    rc, calls, out = _launch(d, work, tmp_path, KIND="value", RUN_DIR="model_checkpoints_value/lr1",
                             TRAIN_ARGS="--data saved_games/lr1 --workers 8")
    assert rc == 0, out.stdout + out.stderr
    assert calls == [f"-s {d}/main.py value-train --run-dir model_checkpoints_value/lr1 --data saved_games/lr1 "
                     f"--workers 8"]
    assert _records(work, "starts.jsonl")[-1]["mode"] == "run" and _records(work, "starts.jsonl")[-1]["kind"] == "value"
    assert _records(work, "exits.jsonl")[-1]["rc"] == 0

    (work / "panel.json").write_text("{}")
    (work / "logs" / "eval" / "ladder_lr1").mkdir(parents=True)
    (work / "logs" / "eval" / "ladder_lr1" / "STOP").touch()
    rc, calls, out = _launch(d, work, tmp_path, KIND="ladder", RUN_DIR="logs/eval/ladder_lr1", PANEL="panel.json",
                             TRAIN_ARGS="--workers 3")
    assert rc == 0, out.stdout + out.stderr
    assert calls == [f"-s {d}/scripts/eval_ladder.py run --panel panel.json --workers 3"]
    assert not (work / "logs" / "eval" / "ladder_lr1" / "STOP").exists()  # a stale STOP is removed at start
    assert _launch(d, work, tmp_path, KIND="ladder", RUN_DIR="logs/eval/ladder_lr1")[0] == 76  # no PANEL
    assert _launch(d, work, tmp_path, KIND="bogus", RUN_DIR="x")[0] == 76

    # The generator, unchanged: fresh (START_ARGS + TRAIN_ARGS), then --resume (TRAIN_ARGS) once state.json exists.
    rc, calls, out = _launch(d, work, tmp_path, RUN_DIR="model_checkpoints_search/lr1", START_ARGS="--policy p.pth",
                             TRAIN_ARGS="--workers 2")
    assert rc == 0, out.stdout + out.stderr
    assert calls == [f"-s {d}/main.py search-learning --out-dir model_checkpoints_search --run-name lr1 "
                     f"--policy p.pth --workers 2"]
    assert _records(work, "starts.jsonl")[-1]["mode"] == "fresh"
    (work / "model_checkpoints_search" / "lr1").mkdir(parents=True)
    (work / "model_checkpoints_search" / "lr1" / "state.json").write_text("{}")
    rc, calls, out = _launch(d, work, tmp_path, RUN_DIR="model_checkpoints_search/lr1", START_ARGS="--policy p.pth",
                             TRAIN_ARGS="--workers 2")
    assert calls == [f"-s {d}/main.py search-learning --resume model_checkpoints_search/lr1 --workers 2"]
    last = (work / "longrun" / "t" / "starts.jsonl").read_text().splitlines()[-1]
    assert '"mode": "resume", "snapshot_update": 10,' in last  # what dryrun.sh's check 7b greps for

    # A non-finite stop is exit 75 (not restarted), for any kind.
    rc, _, _ = _launch(d, work, tmp_path, KIND="value", RUN_DIR="v", STUB_RC=1,
                       STUB_OUT="RuntimeError: value trainer: non-finite loss at step 7")
    assert rc == 75 and _records(work, "exits.jsonl")[-1]["reason"] == "nonfinite"


def test_daily_runs_the_jobs_in_order(deployment, tmp_path):
    d, work = deployment
    log = tmp_path / "stub.log"
    settings = dict(DEPLOY=d, WORKDIR=work, LONGRUN_INSTANCE="lr1-daily", STUB_LOG=log, STATE_DIR="longrun/lr1-daily",
                    RETENTION_CONFIG="r.json", SAVE_DATA="saved_games/lr1")
    out = subprocess.run(["bash", str(d / "scripts" / "longrun" / "daily.sh")], env=env(**settings),
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    calls = [line.split()[1].rsplit("/", 1)[1] for line in log.read_text().splitlines()]
    assert calls == ["retention.py", "style_stats.py", "integrity_check.py", "status_report.py"]
    steps = json.loads((work / "longrun" / "lr1-daily" / "daily_steps.json").read_text())
    assert steps["steps"] == {"retention": 0, "style_stats": 0, "integrity": 0}
    text = log.read_text()
    assert "--baseline first" in text and "--checksums longrun/lr1-daily/shard_checksums.txt" in text
    # A failing job: recorded, the others still run, the unit fails.
    log.unlink()
    out = subprocess.run(["bash", str(d / "scripts" / "longrun" / "daily.sh")],
                         env=env(**settings, STUB_FAIL="integrity_check.py"), capture_output=True, text=True)
    assert out.returncode == 1
    steps = json.loads((work / "longrun" / "lr1-daily" / "daily_steps.json").read_text())
    assert steps["steps"]["integrity"] == 3 and len(log.read_text().splitlines()) == 4
    # Without SAVE_DATA / RETENTION_CONFIG only the report runs.
    log.unlink()
    out = subprocess.run(["bash", str(d / "scripts" / "longrun" / "daily.sh")],
                         env=env(DEPLOY=d, WORKDIR=work, STUB_LOG=log), capture_output=True, text=True)
    assert out.returncode == 0 and len(log.read_text().splitlines()) == 1


def test_shell_scripts_parse():
    for script in ("launch.sh", "install.sh", "daily.sh", "dryrun.sh", "deploy.sh", "host_hold.sh"):
        assert subprocess.run(["bash", "-n", str(LONGRUN / script)]).returncode == 0, script
    assert subprocess.run(["bash", "-n", str(LAUNCH / "install_lr1.sh")]).returncode == 0


def test_the_runs_panel_and_league_files():
    import importlib.util
    import sys

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("eval_ladder", REPO_ROOT / "scripts" / "eval_ladder.py")
    ladder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ladder)
    cfg = ladder.load_config(LAUNCH / "eval_ladder_longrun.json")
    assert ladder.scheduled(cfg) and cfg["baseline_update"] == 0 and cfg["every"] == 1
    assert [m.name for m in ladder.panel_members(cfg)] == ["u0", "gen4critic", "az8_u170", "azl_u2100",
                                                           "sl10_gen4critic"]
    # u0 = checkpoint 10 + the value trainer's version 0, searched as every candidate is.
    assert ladder.run_spec(cfg, 0) == ladder.panel_members(cfg)[0].spec
    ckpt10 = "model_checkpoints/AlphaZeroTransformer/20261004_134558_804974562/10.pth"
    assert ladder.run_spec(cfg, 7).startswith(f"{ckpt10}+model_checkpoints_value/lr1/promoted/7.pth@200")
    # Daily: 2 members x 2 directions x 120 games; weekly: 480 games per pairing in both modes.
    start = "2026-11-01T00:00:00"
    state = {"candidates": [0], "current": None, "schedules": {"0": {"kind": "full", "started": start}},
             "last_full": start}
    rot = ladder.choose_schedule(cfg, state, 3, now=ladder._parse_time("2026-11-02T00:00:00"))
    plan = ladder.plan_matches(cfg, 3, range(4), rot)
    assert rot["kind"] == "rotation" and len(plan) == 4 and sum(m.games for m in plan) == 4 * 120
    full = ladder.choose_schedule(cfg, state, 20, now=ladder._parse_time("2026-11-08T00:00:01"))
    assert full["kind"] == "full" and full["value_check"] == 0
    plan = ladder.plan_matches(cfg, 20, range(21), full)
    assert {m.games for m in plan} == {240} and {m.mode for m in plan} == {"2v2", "1v3"}
    league = json.loads((LAUNCH / "league_longrun.json").read_text())["members"]
    assert {m["name"]: m["mode"] for m in league} == {
        "gen4critic": "search", "az8_u170": "search", "azl_u2100": "search", "sl10_v0": "search",
        "bankrupt_hunter": "policy", "rust_exploiter": "policy"}
    assert all(m.get("min_share") == 0.05 and m["temperature"] == 0.5 for m in league if m["mode"] == "policy")
