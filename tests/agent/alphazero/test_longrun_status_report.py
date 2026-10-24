"""scripts/longrun/status_report.py on synthetic runs: the generator's league scores (P1, P2), the value trainer's
cycles and promotions (P3, P4, the value rules), the ladder's ratings (P5, P8), style statistics (P6), generation
health (P7), watchdog alerts, baselines and milestones -- series that should come out green, a yellow daily alarm,
a stall, a decline, the value rules' red, and a workdir with nothing in it."""

import importlib.util
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
LONGRUN = REPO_ROOT / "scripts" / "longrun"
sys.path.insert(0, str(LONGRUN))
spec = importlib.util.spec_from_file_location("status_report", LONGRUN / "status_report.py")
SR = importlib.util.module_from_spec(spec)
spec.loader.exec_module(SR)

NOW_ISO = "2026-12-01T06:00:00"
NOW = datetime.fromisoformat(NOW_ISO).timestamp()
DAY, HOUR = 86400.0, 3600.0


def iso(t):
    return datetime.fromtimestamp(t).isoformat(timespec="seconds")


def sums(kind, games, excess, sd=0.1, seats=2):
    fair = games * seats / 4
    return {"kind": kind, "games": games, "main_seats": seats * games, "win": fair + excess * games, "fair": fair,
            "worth": fair + excess * games / 2, "win_excess_sq": games * (excess**2 + sd**2),
            "worth_excess_sq": games * ((excess / 2) ** 2 + sd**2)}


class Run:
    """A synthetic long-run workdir."""

    def __init__(self, root: Path, days=10):
        self.root, self.days = root, days
        (root / "gen").mkdir(parents=True)
        (root / "vrun").mkdir()
        (root / "ladder").mkdir()
        (root / "state").mkdir()
        (root / "longrun" / "gen").mkdir(parents=True)
        (root / "longrun" / "gen" / "watchdog_state.json").write_text(json.dumps({"active": {}}))
        (root / "panel.json").write_text(json.dumps({"name": "lad", "out_root": "ladder", "run": "vrun",
                                                     "checkpoint": "promoted/{update}.pth", "spec": "x+{checkpoint}",
                                                     "panel": [{"name": "u0", "update": 0}]}))
        self.ledger = []

    def history(self, fixed, snapshot=None, games=20000, gph=lambda k: 3000.0, capped=lambda k: 0, **extra):
        """``fixed(k)`` / ``snapshot(k)``: the true win excess on day k (0 = the last 24 h); 4 lines a day."""
        update = 0
        with open(self.root / "gen" / "history.jsonl", "w") as f:
            for k in range(self.days - 1, -1, -1):
                for j in range(4):
                    update += 1
                    t = NOW - k * DAY - (j + 0.5) * 6 * HOUR
                    n = games // 4
                    league = {"gen4critic": sums("fixed", n // 2, fixed(k)), "az8": sums("fixed", n // 2, fixed(k))}
                    if snapshot:
                        league["self_u1"] = sums("snapshot", n // 2, snapshot(k))
                    term = {"finished": 100 - capped(k), "max_length": capped(k)}
                    line = {"update": update, "time": iso(t), "games": 100, "games_per_hour": gph(k),
                            "decisions_per_second": 500.0, "evaluations_per_second": 9000.0, "terminations": term,
                            "skipped_starts": 0, "inherited_visit_share": 0.8, "league_share": 0.3, "league": league,
                            "value_version": 3, "saved_games": update * 100, "save_errors": 0,
                            "saver_errors": 0, **extra}
                    f.write(json.dumps(line) + "\n")

    def value(self, ces, gains=None, last=None):
        """Promoted versions 0..len(ces)-1, one every 12 h ending now, data doubling each time."""
        versions, cycles = [], []
        for v, ce in enumerate(ces):
            t = NOW - (len(ces) - 1 - v) * 12 * HOUR - 60
            versions.append({"version": v, "path": f"promoted/{v}.pth", "cycle": v, "time": iso(t),
                             "games_total": 37000 * 2**v, "fixed_ce": ce, "rolling_ce": ce - 0.1})
            if v:
                diff = (gains[v - 1] if gains else ce - ces[v - 1])
                cycles.append({"cycle": v, "time": iso(t), "seconds": 3000, "promoted": True, "version": v,
                               "fixed_ce": ce, "rolling_ce": ce - 0.1, "fixed_diff": diff, "fixed_diff_se": 0.001,
                               "rolling_diff": -0.004, "rolling_diff_se": 0.001, "diverged": False,
                               "consecutive_failures": 0, "cycle_hours": 12, "sample_scale": 1, "candidate": v})
        if last:
            cycles.append({**cycles[-1], **last, "cycle": len(ces)})
        (self.root / "vrun" / "promoted.json").write_text(json.dumps({"current": len(ces) - 1, "versions": versions}))
        (self.root / "vrun" / "cycles.jsonl").write_text("".join(json.dumps(c) + "\n" for c in cycles))
        self.versions = versions

    def ladder(self, elo, se=8.0, value_checks=()):
        """Candidate v (each promoted version but 0) rated ``elo(days ago)`` at 2v2."""
        players = {"u0": {"update": 0, "elo": 0.0, "se": 0.0}}
        for v in self.versions[1:]:
            ago = (NOW - datetime.fromisoformat(v["time"]).timestamp()) / DAY
            players[f"u{v['version']}"] = {"update": v["version"], "elo": elo(ago), "se": se}
            self.ledger.append({"candidate": f"u{v['version']}", "candidate_update": v["version"],
                                "opponent": "gen4critic", "mode": "2v2", "candidate_side": "A",
                                "schedule": "rotation", "game_records": []})
        for u, score in value_checks:
            # 2v2: side A (the candidate) holds two seats; a win share that averages ``score``.
            games = [["human:%d" % i, "AABB", [score, 0.0, 1 - score, 0.0], [0.25] * 4] for i in range(60)]
            self.ledger.append({"candidate": f"u{u}", "candidate_update": u, "opponent": "u0", "mode": "2v2",
                                "candidate_side": "A", "purpose": "value_check", "schedule": "full",
                                "game_records": games})
        (self.root / "ladder" / "ratings.json").write_text(json.dumps({"fits": {"2v2": {"players": players}}}))
        (self.root / "ladder" / "ledger.jsonl").write_text("".join(json.dumps(r) + "\n" for r in self.ledger))

    def style(self, flags=(), withhold=0.1):
        record = {"time": iso(NOW - HOUR), "games": 5000, "flags": list(flags),
                  "main": {"pars": 400, "par_share_max": {"corporation": "PRR", "share": 0.3},
                           "withhold_rate": withhold, "dividend_decisions": 900, "train_first": {}}}
        with open(self.root / "state" / "style_stats.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")

    def report(self, *extra, alert=False, now=NOW_ISO):
        argv = ["--workdir", str(self.root), "--gen-run-dir", "gen", "--value-run-dir", "vrun", "--panel",
                str(self.root / "panel.json"), "--state-dir", "state", "--watch-dirs", "longrun/gen", "--log",
                "log.md", "--now", now, *extra]
        if not alert:
            argv.append("--no-alert")
        assert SR.main(argv) == 0
        return json.loads((self.root / "state" / "status.jsonl").read_text().splitlines()[-1])


@pytest.fixture(autouse=True)
def quiet_host(monkeypatch, tmp_path):
    """The host's own state (a pending reboot, swap in use, apt holds, the kernel log) stays out of the rules."""
    monkeypatch.setenv("M7_HOLD_CHECK", "0")
    monkeypatch.setenv("REBOOT_REQUIRED_FILE", str(tmp_path / "no-reboot"))
    monkeypatch.setenv("M5_XID_CHECK", "0")
    monkeypatch.setenv("M4_SWAP_MIN_GB", "100000")
    monkeypatch.setenv("M4_MEM_RED_GB", "0")
    monkeypatch.setenv("M1_RED_FREE_GB", "0")
    for key in ("GEN_RUN_DIR", "VALUE_RUN_DIR", "LADDER_PANEL", "WATCH_DIRS", "STATE_DIR", "REPORT_LOG",
                "RUN_START", "ALERT_COMMAND", "USAGE_PATHS", "WORKTREES_DIR", "BACKUP_DIR", "INTEGRITY_LOG"):
        monkeypatch.delenv(key, raising=False)


def rules(record):
    return {r["rule"]: r["level"] for r in record["rules"]}


def healthy(run):
    run.history(lambda k: 0.03 - 0.002 * k, snapshot=lambda k: 0.02 - 0.001 * k)
    run.value([0.920, 0.917, 0.915, 0.913, 0.911, 0.909])
    run.ladder(lambda ago: 60 - 10 * ago)
    run.style()


def test_green_run(tmp_path):
    run = Run(tmp_path)
    healthy(run)
    record = run.report()
    assert record["status"] == "green", record["rules"]
    metrics = {m["id"]: m for m in record["metrics"]}
    assert metrics["P1"]["value"] == pytest.approx(0.03, abs=0.003) and metrics["P1"]["se"] < 0.002
    assert metrics["P1"]["change3"][0] == pytest.approx(0.006, abs=0.003)
    assert metrics["P2"]["value"] == pytest.approx(0.02, abs=0.003)
    assert metrics["P3"]["value"] == pytest.approx(0.909) and metrics["P3"]["baseline"][0] == pytest.approx(0.920)
    assert metrics["P5"]["value"] == pytest.approx(60, abs=6)
    assert metrics["P7"]["value"] == pytest.approx(3000)
    text = (tmp_path / "log.md").read_text()
    assert text.startswith("# Long-run log") and "-- GREEN" in text and "| P1 |" in text and "| M1 |" in text
    assert "gen4critic" in text  # the per-member table
    base = json.loads((tmp_path / "state" / "baseline.json").read_text())
    assert set(base["metrics"]) >= {"P1", "P2", "P3", "P5", "P6", "P7"}


def test_yellow_daily_alarm_and_its_alert(tmp_path):
    run = Run(tmp_path)
    run.history(lambda k: 0.02 if k == 0 else 0.03, snapshot=lambda k: 0.02)
    run.value([0.920, 0.917, 0.915, 0.913])
    run.ladder(lambda ago: 60 - 10 * ago)
    run.style()
    out = tmp_path / "alert.out"
    import os

    os.environ["ALERT_COMMAND"] = f'echo "$ALERT_LEVEL $ALERT_KEY" >> {out}'
    try:
        record = run.report(alert=True)
    finally:
        del os.environ["ALERT_COMMAND"]
    assert record["status"] == "yellow", record["rules"]
    assert rules(record)["P1 alarm"] == "yellow"
    assert "WARN [daily] status: daily report YELLOW" in (tmp_path / "state" / "alerts.log").read_text()
    assert out.read_text().split() == ["WARN", "status"]


def test_stall_is_red_only_when_all_three_agree(tmp_path):
    run = Run(tmp_path)
    run.history(lambda k: 0.02, snapshot=lambda k: 0.02)
    ces = [0.920 - 0.001 * v for v in range(11)] + [0.910, 0.910, 0.9102]  # 14 versions over 6.5 days
    run.value(ces, gains=[-0.001] * 10 + [-0.0005, 0.0001, 0.0002])
    run.ladder(lambda ago: 50.0)
    run.style()
    record = run.report()
    assert rules(record).get("stall") == "red", record["rules"]
    assert "decline" not in rules(record)
    # P5 still rising: no stall.
    run.ladder(lambda ago: 80 - 15 * ago)
    assert "stall" not in rules(run.report())
    # P1 still rising (the last 5 days above the 5 before): no stall either.
    run.ladder(lambda ago: 50.0)
    run.history(lambda k: 0.06 - 0.005 * k, snapshot=lambda k: 0.02)
    assert "stall" not in rules(run.report())


def test_decline_needs_p1_and_p5(tmp_path):
    run = Run(tmp_path)
    run.history(lambda k: 0.0 if k < 5 else 0.05, snapshot=lambda k: 0.02)
    run.value([0.920, 0.917, 0.915, 0.913, 0.911, 0.909, 0.907, 0.905, 0.903, 0.901, 0.899, 0.897, 0.895, 0.893,
               0.891, 0.889])
    run.ladder(lambda ago: 0.0 if ago < 3 else 100.0, se=5.0)
    run.style()
    record = run.report()
    assert rules(record).get("decline") == "red", record["rules"]
    assert rules(record).get("P5 alarm") == "yellow"
    # The ladder holding up: P1's drop alone is no decline (a yellow alarm, not a rollback).
    run.ladder(lambda ago: 100.0, se=5.0)
    again = rules(run.report())
    assert "decline" not in again


def test_value_rules(tmp_path):
    run = Run(tmp_path)
    healthy(run)
    run.value([0.920, 0.917, 0.915], last={"promoted": False, "diverged": True, "rolling_diff": 0.02,
                                            "reason": "diverged", "consecutive_failures": 3})
    run.ladder(lambda ago: 60 - 10 * ago, value_checks=[(1, 0.45), (2, 0.46)])
    record = run.report()
    r = rules(record)
    assert r["hard stop: value trainer diverging"] == "red"
    assert r["value: promotion failing"] == "yellow"
    assert r["value: weekly play check"] == "red"
    assert record["status"] == "red"
    run.value([0.920, 0.917, 0.915], last={"promoted": False, "consecutive_failures": 6, "rolling_diff": 0.001})
    run.ladder(lambda ago: 60 - 10 * ago, value_checks=[(2, 0.55)])
    r = rules(run.report())
    assert r["value: promotion failing"] == "red" and "value: weekly play check" not in r
    assert "hard stop: value trainer diverging" not in r


def test_p3_alarm_and_p4_gap(tmp_path):
    run = Run(tmp_path)
    healthy(run)
    run.value([0.920, 0.917, 0.915, 0.9151, 0.9152], gains=[-0.003, -0.002, 0.0001, 0.0001])
    assert rules(run.report()).get("P3 alarm") == "yellow"


def test_style_generation_and_watchdog_rules(tmp_path):
    run = Run(tmp_path)
    healthy(run)
    run.style(flags=["par_share"])
    run.history(lambda k: 0.03 - 0.002 * k, gph=lambda k: 1000.0 if k == 0 else 3000.0, capped=lambda k: 5,
                save_errors=1)
    (tmp_path / "longrun" / "gen" / "watchdog_state.json").write_text(json.dumps(
        {"active": {"crash_loop": {"level": "CRIT"}, "stall": {"level": "WARN"},
                    "reboot_required": {"level": "INFO"}}}))
    record = run.report()
    r = rules(record)
    assert r["P6 drift guard"] == "yellow"
    assert r["P7 throughput"] == "yellow" and r["P7 length cap"] == "yellow"
    assert r["hard stop: saved-data write failure"] == "red"
    assert r["hard stop: gen crash_loop"] == "red" and r["watchdog: gen stall"] == "yellow"
    assert not any("reboot_required" in k for k in r)


def test_nothing_yet(tmp_path):
    (tmp_path / "w").mkdir()
    assert SR.main(["--workdir", str(tmp_path / "w"), "--now", NOW_ISO, "--no-alert"]) == 0
    record = json.loads((tmp_path / "w" / "longrun" / "daily" / "status.jsonl").read_text())
    assert record["status"] == "green", record["rules"]
    assert all(m["status"] == "n/a" for m in record["metrics"])
    assert "n/a" in (tmp_path / "w" / "docs" / "long_run_log.md").read_text()
    # A half-written or garbage input is n/a too, never a crash.
    run = Run(tmp_path / "r")
    (run.root / "gen" / "history.jsonl").write_text('{"update": 1, "time": "garbage"}\n{"upd')
    (run.root / "vrun" / "promoted.json").write_text("{")
    (run.root / "ladder" / "ratings.json").write_text("[]")
    assert run.report()["status"] in ("green", "yellow")


def test_baselines_and_milestones(tmp_path):
    run = Run(tmp_path)
    run.history(lambda k: 0.01, snapshot=lambda k: 0.0)
    run.value([0.920])
    run.ladder(lambda ago: 0.0)
    run.style()
    first = run.report(now=iso(NOW - 4 * DAY))  # day 0: the baselines (P1 0.01)
    base = json.loads((tmp_path / "state" / "baseline.json").read_text())
    assert base["metrics"]["P1"]["value"] == pytest.approx(0.01, abs=0.003)
    assert first["day"] == pytest.approx(0, abs=0.01)
    # Day 4: P1 up by far more than 2 SE, three promotions with fixed-set gains -> the day-3 milestone is met.
    run.history(lambda k: 0.03, snapshot=lambda k: 0.0)
    run.value([0.920, 0.917, 0.915, 0.913])
    run.ladder(lambda ago: 30 - 5 * ago)
    record = run.report()
    assert json.loads((tmp_path / "state" / "baseline.json").read_text())["metrics"]["P1"]["value"] == pytest.approx(
        0.01, abs=0.003)  # kept
    assert "milestone day 3" not in rules(record)
    assert "milestone day 3: met" in (tmp_path / "log.md").read_text()
    # A milestone missed on its date is a yellow.
    run.history(lambda k: 0.01, snapshot=lambda k: 0.0)
    assert rules(run.report()).get("milestone day 3") == "yellow"
    # --set-baseline re-records now.
    run.report("--set-baseline", "P1")
    assert json.loads((tmp_path / "state" / "baseline.json").read_text())["metrics"]["P1"]["value"] == pytest.approx(
        0.01, abs=0.003)
    run.history(lambda k: 0.04, snapshot=lambda k: 0.0)
    run.report("--set-baseline", "P1")
    assert json.loads((tmp_path / "state" / "baseline.json").read_text())["metrics"]["P1"]["value"] == pytest.approx(
        0.04, abs=0.003)


def test_league_excess_and_slopes():
    s = sums("fixed", 10000, 0.02, sd=0.1)
    mean, se = SR.excess(s)
    assert mean == pytest.approx(0.02) and se == pytest.approx(0.1 / math.sqrt(10000))
    assert SR.excess({**s, "games": 0}) is None
    slope, slope_se = SR.wls_slope([(-k, 0.01 * (5 - k), 0.001) for k in range(5)])
    assert slope == pytest.approx(0.01) and slope_se > 0
    assert SR.wls_slope([(0, 1, 1), (1, 1, 1)]) is None
