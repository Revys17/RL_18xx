"""The long-run harness's maintenance metrics (scripts/longrun/maintenance.py, M1-M9), the watchdog's additions
(per-kind progress files, maintenance samples, Xid scan, the shared alert path) and snapshot_check.py on a
value-follow run -- on temporary directories and synthetic histories; no systemd, no GPU."""

import importlib.util
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
LONGRUN = REPO_ROOT / "scripts" / "longrun"
DAY = 86400.0


def _load(name):
    sys.path.insert(0, str(LONGRUN))
    spec = importlib.util.spec_from_file_location(name, LONGRUN / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mt = _load("maintenance")
wd = _load("watchdog")
NOW = 1_800_000_000.0


def iso(t):
    return datetime.fromtimestamp(t).astimezone().isoformat(timespec="seconds")


def settings(tmp_path, **env):
    base = {"WORKDIR": str(tmp_path), "M7_HOLD_CHECK": "0", "REBOOT_REQUIRED_FILE": str(tmp_path / "no-reboot"),
            "M5_XID_CHECK": "0"}
    return mt.Settings({**base, **{k: str(v) for k, v in env.items()}})


def by_id(metrics):
    return {m["id"]: m for m in metrics}


def usage(days, free, total=2000.0, sizes=None, every_hours=6, **extra):
    """Samples over the last ``days`` days: free(t_days_ago) and sizes(t_days_ago) give the values."""
    out = []
    n = int(days * 24 / every_hours)
    for i in range(n, -1, -1):
        ago = i * every_hours / 24
        r = {"time": iso(NOW - ago * DAY), "free_gb": free(ago), "total_gb": total, **extra}
        if sizes:
            r["sizes_gb"] = {k: f(ago) for k, f in sizes.items()}
        out.append(r)
    return out


def test_m1_days_to_full(tmp_path):
    s = settings(tmp_path)
    # 600 GB free, growing 10 GB/day: 60 days -- green; 25 GB/day: 24 days -- yellow; 50 GB/day: 12 -- red.
    for rate, status in ((10, "green"), (25, "yellow"), (50, "red")):
        m = mt.m1_disk(s, usage(7, lambda ago, r=rate: 600 + r * ago), NOW)
        assert m["status"] == status, (rate, m)
        assert m["rate_gb_per_day"] == pytest.approx(rate, rel=1e-6)
    # Under 150 GB free is red whatever the rate; no history is n/a; one sample has no rate.
    assert mt.m1_disk(s, usage(7, lambda ago: 140.0), NOW)["status"] == "red"
    assert mt.m1_disk(s, [], NOW)["status"] == "n/a"
    one = mt.m1_disk(s, usage(0, lambda ago: 600.0), NOW)
    assert one["status"] == "green" and one["value"] is None


def test_m2_growth_against_expected_rates(tmp_path):
    s = settings(tmp_path, M2_EXPECTED_GB_PER_DAY="saved=1.4,evals=0.06")
    hist = usage(7, lambda ago: 900.0, sizes={"saved": lambda ago: 100 - 1.4 * ago, "evals": lambda ago: 5 - 0.2 * ago})
    m = mt.m2_growth(s, hist, NOW)
    assert m["status"] == "yellow" and "evals" in m["detail"] and "OVER" in m["detail"]
    assert m["value"]["saved"] == pytest.approx(1.4, rel=1e-6)
    ok = usage(7, lambda ago: 900.0, sizes={"saved": lambda ago: 100 - 2.0 * ago})
    assert mt.m2_growth(s, ok, NOW)["status"] == "green"  # under 2x
    assert mt.m2_growth(s, usage(7, lambda ago: 900.0), NOW)["status"] == "n/a"


def test_m3_m6_m8_m9(tmp_path):
    s = settings(tmp_path, WORKTREES_DIR=tmp_path / "wt", INTEGRITY_LOG="integrity.jsonl", VALUE_RUN_DIR="vrun",
                 BACKUP_DIR="backup")
    for size, status in ((150, "green"), (250, "yellow"), (350, "red")):
        assert mt.m3_worktrees(s, [{"time": iso(NOW), "worktrees_gb": size}])["status"] == status
    # M6: the integrity log's last line.
    assert mt.m6_integrity(s, NOW)["status"] == "n/a"
    mt.append_jsonl(tmp_path / "integrity.jsonl", {"time": iso(NOW - 3600), "games": 50, "rows": 4000,
                                                   "mismatches": []})
    assert mt.m6_integrity(s, NOW)["status"] == "green"
    assert mt.m6_integrity(s, NOW + 2 * DAY)["status"] == "yellow"  # didn't run for two days
    mt.append_jsonl(tmp_path / "integrity.jsonl", {"time": iso(NOW), "mismatches": [{"uid": "g1", "error": "x"}]})
    assert mt.m6_integrity(s, NOW)["status"] == "red"
    # M8: a cycle's duration against its target (x the cycle's own sample scale).
    (tmp_path / "vrun").mkdir()
    mt.append_jsonl(tmp_path / "vrun" / "cycles.jsonl", {"cycle": 1, "seconds": 1.5 * 3600, "sample_scale_used": 1})
    assert mt.m8_cycle(s)["status"] == "green"
    mt.append_jsonl(tmp_path / "vrun" / "cycles.jsonl", {"cycle": 2, "seconds": 2.5 * 3600, "sample_scale_used": 1})
    assert mt.m8_cycle(s)["status"] == "yellow"
    mt.append_jsonl(tmp_path / "vrun" / "cycles.jsonl", {"cycle": 3, "seconds": 2.5 * 3600, "sample_scale_used": 2,
                                                         "sample_scale": 4})
    assert mt.m8_cycle(s)["status"] == "green"
    # M9: the newest file in the backup.
    (tmp_path / "backup").mkdir()
    f = tmp_path / "backup" / "week.tar"
    f.write_bytes(b"x")
    os.utime(f, (NOW - 9 * DAY, NOW - 9 * DAY))
    assert mt.m9_backup(s, NOW)["status"] == "yellow"
    os.utime(f, (NOW - DAY, NOW - DAY))
    assert mt.m9_backup(s, NOW)["status"] == "green"
    assert mt.m9_backup(settings(tmp_path), NOW)["status"] == "n/a"


def test_m4_leaks_swap_and_memory(tmp_path):
    s = settings(tmp_path, M4_SWAP_MIN_GB=0.5)
    gb = 1e9

    def rss(rate_per_day, days=3, every_hours=1):
        out = []
        for i in range(int(days * 24 / every_hours), -1, -1):
            ago = i * every_hours / 24
            out.append({"time": iso(NOW - ago * DAY), "cgroup_bytes": int((80 + rate_per_day * (days - ago)) * gb),
                        "procs": [[101, 5, int((10 + rate_per_day * (days - ago)) * gb), "python main.py"],
                                  [102, 6, int(3 * gb), "python server"]]})
        return out

    hist = usage(1, lambda ago: 900.0, mem_available_min_gb=120.0, swap_used_gb=0.0)
    leak = mt.m4_memory(s, hist, {"longrun/gen": rss(3.0)}, NOW)
    assert leak["status"] == "yellow" and "pid 101" in leak["detail"] and "cgroup" in leak["detail"]
    assert "pid 102" not in leak["detail"]
    assert mt.m4_memory(s, hist, {"longrun/gen": rss(1.0)}, NOW)["status"] == "green"
    # A process younger than two days can't be called a leak yet.
    assert mt.m4_memory(s, hist, {"longrun/gen": rss(5.0, days=1.5)}, NOW)["status"] == "green"
    swap = usage(1, lambda ago: 900.0, mem_available_min_gb=120.0, swap_used_gb=2.0)
    assert mt.m4_memory(s, swap, {}, NOW)["status"] == "yellow"
    low = usage(1, lambda ago: 900.0, mem_available_min_gb=20.0, swap_used_gb=0.0)
    assert mt.m4_memory(s, low, {}, NOW)["status"] == "red"


def test_m5_xid_and_throttling(tmp_path):
    s = settings(tmp_path)
    calm = [{"time": iso(NOW - 3600), "throttle_hours": [{"hour": "x", "fraction": 0.05}]}]
    hot = [{"time": iso(NOW - 3600), "throttle_hours": [{"hour": "x", "fraction": 0.2}]}]
    assert mt.m5_gpu(s, calm, NOW, ([], None))["status"] == "green"
    assert mt.m5_gpu(s, hot, NOW, ([], None))["status"] == "yellow"
    red = mt.m5_gpu(s, calm, NOW, (["... NVRM: Xid (PCI:0000:01:00): 79, GPU has fallen off the bus"], None))
    assert red["status"] == "red" and "Xid" in red["detail"]
    assert mt.m5_gpu(s, [], NOW, (None, "permission denied"))["status"] == "n/a"


def test_m7_reboot_and_holds(tmp_path):
    s = settings(tmp_path, M7_HOLD_CHECK=1)
    assert mt.m7_host(s, hold=(True, [], None))["status"] == "green"
    assert mt.m7_host(s, hold=(False, ["linux-generic"], None))["status"] == "red"
    (tmp_path / "no-reboot").touch()
    assert mt.m7_host(s, hold=(True, [], None))["status"] == "yellow"


def test_evaluate_runs_on_nothing(tmp_path):
    metrics = mt.evaluate(settings(tmp_path), now=NOW, history=[], rss_by_dir={}, xid=([], None))
    assert [m["id"] for m in metrics] == [f"M{i}" for i in range(1, 10)]


# ----------------------------------------------------------------------------- the watchdog


@pytest.fixture
def dog_env(tmp_path, monkeypatch):
    for key in list(os.environ):
        if key in ("KIND", "PROGRESS_FILE", "HOST_CHECKS", "USAGE_LOG", "USAGE_PATHS", "ALERT_COMMAND", "RUN_DIR",
                   "STATE_DIR", "WORKDIR", "STALL_MINUTES"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("WORKDIR", str(tmp_path))
    monkeypatch.setenv("GPU_CHECK", "0")
    monkeypatch.setattr(wd, "nvidia_mismatch", lambda: None)
    monkeypatch.setattr(wd, "unit_state", lambda unit: {"ActiveState": "active", "SubState": "running",
                                                        "NRestarts": 0, "ExecMainStatus": 0, "MainPID": 1,
                                                        "ActiveEnterTimestampMonotonic": 0, "ControlGroup": "/x",
                                                        "InvocationID": "i", "LoadState": "loaded"})
    monkeypatch.setattr(wd, "cgroup_memory", lambda cg: None)
    monkeypatch.setattr(wd.maintenance, "xid_lines", lambda since, until=None: ([], None))
    return tmp_path


@pytest.mark.parametrize("kind,progress", [("generator", "history.jsonl"), ("value", "status.json"),
                                           ("ladder", "heartbeat")])
def test_watchdog_progress_file_by_kind(dog_env, monkeypatch, kind, progress):
    tmp_path = dog_env
    monkeypatch.setenv("KIND", kind)
    monkeypatch.setenv("RUN_DIR", "run")
    monkeypatch.setenv("STALL_MINUTES", "30")
    (tmp_path / "run").mkdir()
    s = wd.Settings("t")
    assert s.progress == tmp_path / "run" / progress
    path = s.progress
    path.write_text('{"update": 3, "time": "2026-11-01T00:00:00"}\n' if kind == "generator" else "{}\n")
    old = time.time() - 3600
    os.utime(path, (old, old))
    conditions, _ = wd.Watchdog(s, quiet=True).check()
    assert "stall" in conditions and path.name in conditions["stall"][1]
    path.touch()
    conditions, _ = wd.Watchdog(s, quiet=True).check()
    assert "stall" not in conditions


def test_watchdog_samples_rss_usage_and_xid(dog_env, monkeypatch):
    tmp_path = dog_env
    monkeypatch.setenv("RUN_DIR", "run")
    monkeypatch.setenv("HOST_CHECKS", "1")
    monkeypatch.setenv("USAGE_LOG", "longrun/usage.jsonl")
    (tmp_path / "saved").mkdir()
    (tmp_path / "saved" / "a.sld").write_bytes(b"x" * 10000)
    monkeypatch.setenv("USAGE_PATHS", "saved=saved")
    monkeypatch.setattr(wd.maintenance, "rss_sample", lambda cg: {"time": wd.maintenance.now_iso(),
                                                                  "cgroup_bytes": 5, "procs": [[1, 2, 3, "x"]]})
    monkeypatch.setattr(wd.maintenance, "xid_lines",
                        lambda since, until=None: (["NVRM: Xid (PCI:0000:01:00): 13, Graphics Exception"], None))
    dog = wd.Watchdog(wd.Settings("t"))
    dog.state["throttle_hours"] = [{"hour": "2026-11-01T00", "fraction": 0.25}]
    dog.round()
    alerts = (tmp_path / "longrun" / "t" / "alerts.log").read_text()
    assert "CRIT [t] xid" in alerts
    rss = [json.loads(line) for line in (tmp_path / "longrun" / "t" / "rss.jsonl").read_text().splitlines()]
    assert rss[0]["procs"] == [[1, 2, 3, "x"]]
    sample = json.loads((tmp_path / "longrun" / "usage.jsonl").read_text().splitlines()[0])
    assert sample["sizes_gb"]["saved"] > 0 and sample["throttle_hours"][0]["fraction"] == 0.25
    assert sample["mem_available_min_gb"] is not None
    # The next round within the intervals samples nothing new and doesn't scan again.
    dog.round()
    assert len((tmp_path / "longrun" / "usage.jsonl").read_text().splitlines()) == 1
    assert (tmp_path / "longrun" / "t" / "alerts.log").read_text().count("xid") == 1


def test_throttle_fraction_per_hour(dog_env):
    dog = wd.Watchdog(wd.Settings("t"), quiet=True)
    dog.state["throttle_hour"] = {"hour": "2000-01-01T00", "rounds": 10, "throttled": 3}
    dog.count_throttle(True)
    assert dog.state["throttle_hours"] == [{"hour": "2000-01-01T00", "fraction": 0.3}]
    assert dog.state["throttle_hour"]["rounds"] == 1 and dog.state["throttle_hour"]["throttled"] == 1


def test_send_alert_logs_and_runs_the_command(tmp_path):
    out = tmp_path / "cmd.out"
    wd.send_alert(tmp_path / "state", "lr1-daily", "WARN", "status", "daily report YELLOW",
                  f'echo "$ALERT_LEVEL $ALERT_KEY $ALERT_INSTANCE" >> {out}')
    wd.send_alert(tmp_path / "state", "lr1-daily", "INFO", "note", "only logged", f"echo x >> {out}")
    log = (tmp_path / "state" / "alerts.log").read_text().splitlines()
    assert len(log) == 2 and "WARN [lr1-daily] status: daily report YELLOW" in log[0]
    assert out.read_text().split() == ["WARN", "status", "lr1-daily"]


# ----------------------------------------------------------------------------- snapshot_check on a value-follow run


def test_snapshot_check_value_follow_run(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    sc = _load("snapshot_check")
    work = tmp_path / "work"
    run = work / "model_checkpoints_search" / "lr1"
    (run / "learner").mkdir(parents=True)
    policy = work / "model_checkpoints" / "ckpt" / "10.pth"
    policy.parent.mkdir(parents=True)
    policy.write_bytes(b"policy")
    config = {"policy_checkpoint": "model_checkpoints/ckpt/10.pth", "value_checkpoint": "", "value_follow":
              "model_checkpoints_value/lr1", "freeze_policy": True, "shared": False, "out_dir":
              "model_checkpoints_search", "run_name": "lr1", "snapshot_every": 10}
    (run / "learner" / "0.pth").write_bytes(b"frozen")
    for u in (10, 20):
        os.link(run / "learner" / "0.pth", run / "learner" / f"{u}.pth")  # a frozen run: hard links
    torch.save({"learner": {"state": {}, "param_groups": []}, "critic": None}, run / "optimizer.pth")
    (run / "config.json").write_text(json.dumps(config))
    state = {"config": config, "snapshot_update": 20, "kl_coef": 0.02, "anchor_update": 0, "totals": {}}
    (run / "state.json").write_text(json.dumps(state))
    with open(run / "history.jsonl", "w") as f:
        for u in range(1, 21):
            f.write(json.dumps({"update": u, "anchor_update": 0, "games_total": u * 10}) + "\n")  # no kl_coef
    assert not (run / "critic").exists()
    assert sc.check(run) == 20
    # An own critic is still required without value_follow / shared.
    (run / "state.json").write_text(json.dumps({**state, "config": {**config, "value_follow": None}}))
    with pytest.raises(sc.NotResumable, match="critic"):
        sc.check(run)
    # A torn state.json is rebuilt at the newest snapshot (a history line without kl_coef takes the config's).
    (run / "state.json").write_text('{"config": {"pol')
    monkeypatch.chdir(tmp_path)
    assert sc.repair(run) == 20
    rebuilt = json.loads((run / "state.json").read_text())
    assert rebuilt["snapshot_update"] == 20 and rebuilt["anchor_update"] == 0 and rebuilt["kl_coef"] == 0.02
