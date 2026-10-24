#!/usr/bin/env python3
"""The long run's daily status report (docs/long_run_recipe.md, "Progress metrics, cadence and stop rules").

    status_report.py [--log docs/long_run_log.md] [--set-baseline [P1,P6,...|all]] [--no-alert] [--print-only]

Every metric of the recipe -- P1-P8 (training-game scores, value CE, ladder, style, generation health, the weekly
panel) and M1-M9 (maintenance, scripts/longrun/maintenance.py) -- with its day-0 baseline, its 3- and 7-day change
(with SE where the metric has one) and a green / yellow / red status from the recipe's rules: the daily alarms, stall
(P1 + P3 + P5 together), decline (P1 + P5), the value rules (divergence, promotion failures, the weekly value play
check), the hard stops (crash loops, non-finite stops, saved-data write failures, integrity mismatches, disk red)
and the milestones. Appends a dated markdown section to ``--log``, one JSON line to ``$STATE_DIR/status.jsonl``,
and raises a WARN (yellow) or CRIT (red) ``status`` alert through the watchdog's alert path. Missing inputs (day 0,
no ladder yet, no league games) show as n/a; the report never fails on them.

"Days" are 24-hour windows ending at the report's time (window 0 = the last 24 h). The baseline
(``$STATE_DIR/baseline.json``) records each metric's first available value (P1 / P2 / P6 / P7: the first day with
data; P3: the start value's fixed-set CE; P5: u0 = 0 Elo); ``--set-baseline`` re-records the named metrics now.

Inputs (environment, as the daily job's .env sets them; each also a flag): WORKDIR, GEN_RUN_DIR (history.jsonl),
VALUE_RUN_DIR (cycles.jsonl, promoted.json), LADDER_PANEL (the panel file: its ladder's ratings.json, ledger,
state), STATE_DIR (default longrun/daily: style_stats.jsonl, integrity.jsonl, retention.jsonl, daily_steps.json,
baseline.json, status.jsonl, alerts.log), WATCH_DIRS (':'-separated state dirs of the three instances: their
watchdogs' active alerts and alerts.log), REPORT_LOG, RUN_START (ISO; default: the baseline's creation), and the
maintenance settings (maintenance.py). Runs with the deployment's venv or system python3 (standard library only).
"""

import argparse
import json
import math
import os
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import maintenance as mt  # noqa: E402

GREEN, YELLOW, RED, NA = mt.GREEN, mt.YELLOW, mt.RED, mt.NA
DAY = 86400.0
LENGTH_CAP = ("max_length", "max_decisions")
HARD_STOP_KEYS = ("crash_loop", "failed", "nonfinite", "refused", "oom_kill", "xid", "integrity")
MILESTONES = ((3, "day 3"), (14, "week 2"), (28, "week 4"), (56, "week 8"), (112, "week 16"))


# ----------------------------------------------------------------------------- small helpers


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError, TypeError):
        return default


def t_of(record, key="time"):
    return mt.record_time(record, key)


def fmt(value, se=None, digits=3, signed=False):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    sign = "+" if signed else ""
    text = f"{value:{sign}.{digits}f}"
    return text + (f" ± {se:.{digits}f}" if se is not None and not math.isnan(se) else "")


def diff(a, b):
    """(a - b, SE) of two (value, se) pairs; None if either is missing."""
    if a is None or b is None or a[0] is None or b[0] is None:
        return None
    se = math.sqrt((a[1] or 0.0) ** 2 + (b[1] or 0.0) ** 2) if a[1] is not None and b[1] is not None else None
    return a[0] - b[0], se


def wls_slope(points):
    """Weighted least-squares slope of [(x, y, se)] and its SE (None with fewer than 3 points)."""
    pts = [(x, y, se) for x, y, se in points if y is not None and se]
    if len(pts) < 3:
        return None
    w = [1 / se**2 for _, _, se in pts]
    sw = sum(w)
    mx = sum(wi * x for wi, (x, _, _) in zip(w, pts)) / sw
    my = sum(wi * y for wi, (_, y, _) in zip(w, pts)) / sw
    sxx = sum(wi * (x - mx) ** 2 for wi, (x, _, _) in zip(w, pts))
    if sxx <= 0:
        return None
    slope = sum(wi * (x - mx) * (y - my) for wi, (x, y, _) in zip(w, pts)) / sxx
    return slope, math.sqrt(1 / sxx)


def ols_slope(points):
    pts = [(x, y) for x, y in points if x is not None and y is not None]
    if len(pts) < 2:
        return None
    mx, my = sum(x for x, _ in pts) / len(pts), sum(y for _, y in pts) / len(pts)
    sxx = sum((x - mx) ** 2 for x, _ in pts)
    return None if sxx <= 0 else sum((x - mx) * (y - my) for x, y in pts) / sxx


def weighted_mean(items):
    """Inverse-variance mean of [(value, se)] and its SE."""
    items = [(v, se) for v, se in items if v is not None and se]
    if not items:
        return None
    w = [1 / se**2 for _, se in items]
    return sum(wi * v for wi, (v, _) in zip(w, items)) / sum(w), math.sqrt(1 / sum(w))


def row(mid, name, now=None, se=None, baseline=None, change3=None, change7=None, status=NA, detail="", digits=3,
        text=None):
    return {
        "id": mid,
        "name": name,
        "value": now,
        "se": se,
        "text": text if text is not None else fmt(now, se, digits),
        "baseline": baseline,
        "change3": change3,
        "change7": change7,
        "status": status,
        "detail": detail,
        "digits": digits,
    }


# ----------------------------------------------------------------------------- settings


class Settings:
    def __init__(self, args):
        env = os.environ
        pick = lambda flag, key, default="": flag if flag is not None else env.get(key, default).strip()  # noqa: E731
        self.workdir = Path(pick(args.workdir, "WORKDIR", ".") or ".")
        self.gen_run = self._p(pick(args.gen_run_dir, "GEN_RUN_DIR"))
        self.value_run = self._p(pick(args.value_run_dir, "VALUE_RUN_DIR"))
        self.panel = self._p(pick(args.panel, "LADDER_PANEL"))
        self.state_dir = self._p(pick(args.state_dir, "STATE_DIR", "longrun/daily") or "longrun/daily")
        self.watch_dirs = [self._p(d) for d in pick(args.watch_dirs, "WATCH_DIRS").split(":") if d]
        self.log = self._p(pick(args.log, "REPORT_LOG", "docs/long_run_log.md") or "docs/long_run_log.md")
        self.style = self._p(pick(args.style_stats, "STYLE_STATS")) or self.state_dir / "style_stats.jsonl"
        self.retention = self._p(pick(args.retention_log, "RETENTION_LOG")) or self.state_dir / "retention.jsonl"
        self.baseline = self.state_dir / "baseline.json"
        self.run_start = pick(args.run_start, "RUN_START")
        self.instance = env.get("LONGRUN_INSTANCE", "daily")
        self.alert_command = env.get("ALERT_COMMAND", "").strip()
        env_m = dict(env)
        env_m.setdefault("WORKDIR", str(self.workdir))
        env_m.setdefault("INTEGRITY_LOG", str(self.state_dir / "integrity.jsonl"))
        if self.value_run:
            env_m.setdefault("VALUE_RUN_DIR", str(self.value_run))
        env_m.setdefault("RSS_DIRS", ":".join(str(d) for d in self.watch_dirs))
        self.maint = mt.Settings(env_m)

    def _p(self, value):
        if not value:
            return None
        p = Path(value)
        return p if p.is_absolute() else self.workdir / p


# ----------------------------------------------------------------------------- generator (P1, P2, P7)

FIELDS = ("games", "main_seats", "win", "fair", "worth", "win_excess_sq", "worth_excess_sq")


def window_index(t, now):
    """0 for the last 24 h, 1 for the 24 h before, ...; None in the future."""
    if t > now + 60:
        return None
    return int((now - t) // DAY)


def load_history(path, now, keep_days=None):
    """Per-window aggregates of the generator's history: league sums per member (kind), games, terminations,
    per-update rates, save errors."""
    windows = {}
    try:
        f = open(path)
    except (OSError, TypeError):
        return windows
    with f:
        for line in f:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            k = window_index(t_of(r), now)
            if k is None or (keep_days is not None and k >= keep_days):
                continue
            w = windows.setdefault(k, {"league": {}, "games": 0, "terminations": {}, "gph": [], "dps": [], "eps": [],
                                       "reuse": [], "skipped": 0, "save_errors": 0, "saver_errors": [],
                                       "league_share": [], "value_versions": set(), "updates": 0})
            w["updates"] += 1
            w["games"] += int(r.get("games") or 0)
            for key, value in (r.get("terminations") or {}).items():
                w["terminations"][key] = w["terminations"].get(key, 0) + int(value)
            for key, field in (("gph", "games_per_hour"), ("dps", "decisions_per_second"),
                               ("eps", "evaluations_per_second"), ("reuse", "inherited_visit_share"),
                               ("league_share", "league_share")):
                if isinstance(r.get(field), (int, float)):
                    w[key].append(float(r[field]))
            w["skipped"] += int(r.get("skipped_starts") or 0)
            w["save_errors"] += int(r.get("save_errors") or 0)
            if r.get("saver_errors") is not None:
                w["saver_errors"].append(int(r["saver_errors"]))
            if r.get("value_version") is not None:
                w["value_versions"].add(int(r["value_version"]))
            for member, sums in (r.get("league") or {}).items():
                m = w["league"].setdefault(member, {"kind": sums.get("kind", "fixed"), **{f: 0.0 for f in FIELDS}})
                for f in FIELDS:
                    m[f] += float(sums.get(f) or 0.0)
    return windows


def pooled(windows, ks, kind=None, member=None):
    """League sums pooled over windows ``ks`` and the members of ``kind`` (or one ``member``)."""
    total = {f: 0.0 for f in FIELDS}
    for k in ks:
        for name, sums in windows.get(k, {}).get("league", {}).items():
            if (member is not None and name != member) or (kind is not None and sums["kind"] != kind):
                continue
            for f in FIELDS:
                total[f] += sums[f]
    return total


def excess(sums, outcome="win"):
    """(mean excess over the fair share, SE) per game, or None without games."""
    n = sums["games"]
    if n <= 0:
        return None
    mean = (sums[outcome] - sums["fair"]) / n
    var = max(sums[f"{outcome}_excess_sq"] / n - mean**2, 0.0)
    return mean, math.sqrt(var / n) if n > 1 else None


def league_metrics(windows, base, kind, mid, name):
    """P1 / P2: (row, extra) with the series the rules read."""
    series = {k: excess(pooled(windows, [k], kind)) for k in range(0, max(windows, default=-1) + 1)}
    series = {k: v for k, v in series.items() if v is not None}
    now = series.get(0)
    worth = excess(pooled(windows, [0], kind), "worth")
    b = base.get(mid)
    r = row(mid, name, *(now or (None, None)), baseline=b,
            change3=diff(now, series.get(3)), change7=diff(now, series.get(7)),
            status=GREEN if now else NA, digits=4,
            detail=(f"net worth {fmt(*worth, digits=4, signed=True)}; {int(pooled(windows, [0], kind)['games'])} games"
                    if worth else "no league games of this kind in the last 24 h"))
    return r, series


def block(windows, start, length, kind):
    return excess(pooled(windows, range(start, start + length), kind))


# ----------------------------------------------------------------------------- value (P3, P4, value rules)


def value_data(value_run):
    if not value_run:
        return [], {}
    cycles = mt.read_jsonl(value_run / "cycles.jsonl")
    promoted = read_json(value_run / "promoted.json", {}) or {}
    return cycles, promoted


def promoted_versions(promoted):
    out = []
    for v in promoted.get("versions", []):
        try:
            out.append({**v, "version": int(v["version"]), "t": t_of(v)})
        except (KeyError, TypeError, ValueError):
            continue
    return sorted(out, key=lambda v: v["version"])


def version_at(versions, t):
    """The promoted version current at time ``t``."""
    before = [v for v in versions if v["t"] <= t]
    return before[-1] if before else None


def p3_slope(versions):
    pts = [(math.log2(v["games_total"]), v["fixed_ce"]) for v in versions
           if v.get("games_total") and v.get("fixed_ce") is not None and v["games_total"] > 0]
    return ols_slope(pts)


# ----------------------------------------------------------------------------- ladder (P5, P8)


def load_ladder(settings):
    """(panel, ratings.json, ledger records, ladder state) -- each empty when missing."""
    panel = read_json(settings.panel, {}) if settings.panel else {}
    if not panel:
        return {}, {}, [], {}
    root = Path(panel.get("out_root") or Path("logs") / "eval" / panel.get("name", ""))
    root = root if root.is_absolute() else settings.workdir / root
    ledger = Path(panel["ledger"]) if panel.get("ledger") else root / "ledger.jsonl"
    ledger = ledger if ledger.is_absolute() else settings.workdir / ledger
    return panel, read_json(root / "ratings.json", {}) or {}, mt.read_jsonl(ledger), read_json(root / "state.json", {})


def candidate_ratings(ratings, fit="2v2"):
    """name -> {update, elo, se} of the run's candidates in one fit (u0 included: 0 by anchoring)."""
    players = ((ratings.get("fits") or {}).get(fit) or {}).get("players") or {}
    return {n: p for n, p in players.items() if p.get("update") is not None}


def candidate_times(ratings, records, versions, state):
    """name -> unix time the candidate's system existed from: its promotion (promoted.json), else its schedule's
    start, else its first ledger record."""
    by_version = {v["version"]: v["t"] for v in versions}
    times = {}
    schedules = (ratings.get("schedules") or {}) | ((state or {}).get("schedules") or {})
    for r in records:
        u = r.get("candidate_update")
        if u is None:
            continue
        name = r["candidate"]
        t = by_version.get(int(u)) or t_of(schedules.get(str(u), {}), "started") or t_of(r)
        if t and (name not in times or t < times[name]):
            times[name] = t
    return times


def candidate_schedules(ratings, records):
    """update -> "full" / "rotation" (the ladder's schedules, else the ledger's record fields)."""
    out = {}
    for r in records:
        if r.get("candidate_update") is not None:
            out.setdefault(int(r["candidate_update"]), r.get("schedule", "full"))
    for u, s in (ratings.get("schedules") or {}).items():
        out[int(u)] = s.get("kind", out.get(int(u), "full"))
    return out


def p5_block(cands, times, now, start_day, days=3):
    """Inverse-variance mean 2v2 Elo of the candidates whose systems date from [now - (start+days) d, now - start d)."""
    lo, hi = now - (start_day + days) * DAY, now - start_day * DAY
    return weighted_mean([(p["elo"], p["se"]) for n, p in cands.items()
                          if n in times and lo <= times[n] < hi and p.get("update") != 0])


def excess_vs(records, candidate, opponent, mode):
    """The candidate's mean win share above its seats / 4 against one opponent in one mode, both directions pooled,
    with a start-clustered SE: (mean, se, games)."""
    values, clusters = [], []
    for r in records:
        if r.get("candidate") != candidate or r.get("opponent") != opponent or r.get("mode") != mode:
            continue
        side = r["candidate_side"]
        for g in r.get("game_records") or []:
            values.append(sum(v for v, s in zip(g[2], g[1]) if s == side) - g[1].count(side) / 4)
            clusters.append(g[0])
    if not values:
        return None
    mean = sum(values) / len(values)
    groups = {}
    for v, c in zip(values, clusters):
        groups[c] = groups.get(c, 0.0) + (v - mean)
    g = len(groups)
    se = math.sqrt(g / (g - 1) * sum(s * s for s in groups.values())) / len(values) if g > 1 else None
    return mean, se, len(values)


def value_checks(records):
    """[(candidate update, score, se, games)] of the weekly value play checks (2v2: the candidate's two seats' win
    share), oldest first."""
    out = {}
    for r in records:
        if r.get("purpose") != "value_check":
            continue
        u = r.get("candidate_update")
        side = r["candidate_side"]
        for g in r.get("game_records") or []:
            out.setdefault(u, ([], []))
            out[u][0].append(sum(v for v, s in zip(g[2], g[1]) if s == side))
            out[u][1].append(g[0])
    result = []
    for u in sorted(out, key=lambda x: (x is None, x)):
        vals, cl = out[u]
        mean = sum(vals) / len(vals)
        groups = {}
        for v, c in zip(vals, cl):
            groups[c] = groups.get(c, 0.0) + (v - mean)
        k = len(groups)
        se = math.sqrt(k / (k - 1) * sum(s * s for s in groups.values())) / len(vals) if k > 1 else None
        result.append((u, mean, se, len(vals)))
    return result


# ----------------------------------------------------------------------------- watchdogs, daily jobs


def watchdog_findings(dirs, now):
    """[(instance dir name, active non-INFO alerts {key: level}, WARN count 24 h, CRIT count 24 h)]."""
    out = []
    for d in dirs:
        state = read_json(Path(d) / "watchdog_state.json", {}) or {}
        active = {k: v.get("level") for k, v in (state.get("active") or {}).items() if v.get("level") != "INFO"}
        warn = crit = 0
        try:
            for line in (Path(d) / "alerts.log").read_text().splitlines():
                stamp = line.split(" ", 1)[0]
                try:
                    if now - datetime.fromisoformat(stamp).timestamp() > DAY:
                        continue
                except ValueError:
                    continue
                warn += " WARN " in line
                crit += " CRIT " in line
        except OSError:
            pass
        heartbeat = Path(d) / "watchdog.heartbeat"
        stale = None
        try:
            stale = (now - heartbeat.stat().st_mtime) / 60
        except OSError:
            pass
        out.append((Path(d).name, active, warn, crit, stale))
    return out


# ----------------------------------------------------------------------------- the report


class Report:
    def __init__(self, settings, now=None, write=True):
        self.s = settings
        self.write = write  # False: append nothing (not even the usage sample)
        self.now = time.time() if now is None else now
        self.rows, self.member_rows, self.rules, self.notes = [], [], [], []
        self.base_file = read_json(settings.baseline, None) or {"created": None, "metrics": {}}
        self.base = self.base_file.setdefault("metrics", {})

    def stamp(self):
        return datetime.fromtimestamp(self.now).astimezone().isoformat(timespec="seconds")

    def rule(self, level, name, message):
        self.rules.append({"level": level, "rule": name, "message": message})

    def baseline_value(self, mid):
        b = self.base.get(mid)
        return (b.get("value"), b.get("se")) if b else None

    def record_baseline(self, mid, value, se=None, force=False, **extra):
        if value is None or (mid in self.base and not force):
            return
        self.base[mid] = {"value": value, "se": se, "date": self.stamp(), **extra}

    # -- sections

    def generator(self, force):
        windows = load_history(self.s.gen_run / "history.jsonl", self.now) if self.s.gen_run else {}
        self.windows = windows
        for mid, kind, name in (("P1", "fixed", "score vs fixed league members (pooled win excess)"),
                                ("P2", "snapshot", "score vs past main-line snapshots (pooled win excess)")):
            r, series = league_metrics(windows, self.base, kind, mid, name)
            if series.get(0):
                self.record_baseline(mid, *series[0], force=mid in force)
            r["baseline"] = self.baseline_value(mid)
            self.rows.append(r)
            setattr(self, mid.lower(), series)
        # Per fixed member, last 24 h.
        names = sorted({n for n, s in windows.get(0, {}).get("league", {}).items()})
        for name in names:
            kind = windows[0]["league"][name]["kind"]
            win, worth = excess(pooled(windows, [0], member=name)), excess(pooled(windows, [0], member=name), "worth")
            week = excess(pooled(windows, range(7), member=name))
            self.member_rows.append((name, kind, int(windows[0]["league"][name]["games"]), win, worth, week))
        p1 = self.p1
        # Daily alarm: P1 below its 5-day maximum by >= 2 SE.
        if p1.get(0):
            best_k = max((k for k in range(5) if p1.get(k)), key=lambda k: p1[k][0])
            d = diff(p1[0], p1[best_k])
            if best_k != 0 and d and d[1] and d[0] <= -2 * d[1]:
                self.rule(YELLOW, "P1 alarm", f"P1 {fmt(*p1[0], digits=4)} is below its 5-day maximum "
                                              f"{fmt(*p1[best_k], digits=4)} ({best_k} days ago) by >= 2 SE")
        # Drift: P2 rising by >= 2 SE over 3 days while P1 is flat.
        d2, d1 = diff(self.p2.get(0), self.p2.get(3)), diff(p1.get(0), p1.get(3))
        if d2 and d2[1] and d2[0] >= 2 * d2[1] and d1 and d1[1] and abs(d1[0]) < 2 * d1[1]:
            self.rule(YELLOW, "drift (P2 up, P1 flat)", f"P2 rose {fmt(*d2, digits=4, signed=True)} in 3 days while "
                                                        f"P1 moved {fmt(*d1, digits=4, signed=True)}: raise the "
                                                        f"league share to 50% now (--league-share 0.5)")
        self.stall_p1 = None
        fit = wls_slope([(-k, p1[k][0], p1[k][1]) for k in range(5) if p1.get(k)])
        if fit:
            upper = fit[0] + 1.96 * fit[1]
            self.stall_p1 = upper < 0.001
            self.notes.append(f"P1 5-day slope {fit[0]:+.4f} ± {fit[1]:.4f} per day (upper 95% {upper:+.4f})")
        self.decline_p1 = None
        current = block(windows, 0, 5, "fixed")
        blocks = [b for b in (block(windows, k, 5, "fixed") for k in range(0, max(windows, default=0) + 1)) if b]
        if current and blocks:
            best = max(blocks, key=lambda b: b[0])
            d = diff(current, best)
            self.decline_p1 = bool(d and d[1] and d[0] <= -3 * d[1])
        self.p7(windows, force)

    def p7(self, windows, force):
        w0 = windows.get(0)
        if not w0 or not w0["gph"]:
            self.rows.append(row("P7", "generation health (games/h, median of updates)", status=NA,
                                 detail="no generator history in the last 24 h"))
            self.saved_errors(windows)
            return
        gph = statistics.median(w0["gph"])
        week = [x for k in range(1, 8) for x in windows.get(k, {}).get("gph", [])]
        median7 = statistics.median(week) if week else None
        capped = sum(w0["terminations"].get(k, 0) for k in LENGTH_CAP)
        games = sum(w0["terminations"].values()) or w0["games"]
        cap_share = capped / games if games else None
        status = GREEN
        if median7 and gph < 0.8 * median7:
            status = YELLOW
            self.rule(YELLOW, "P7 throughput", f"{gph:.0f} games/h, below 80% of the 7-day median {median7:.0f}")
        if cap_share is not None and cap_share > 0.03:
            status = YELLOW
            self.rule(YELLOW, "P7 length cap", f"{cap_share:.1%} of the last day's games hit the length cap (> 3%)")
        self.record_baseline("P7", gph, force="P7" in force)
        then3 = windows.get(3, {}).get("gph")
        then7 = windows.get(7, {}).get("gph")
        med = lambda xs: statistics.median(xs) if xs else None  # noqa: E731
        detail = (
            f"decisions/s {fmt(med(w0['dps']), digits=0)}, evaluations/s {fmt(med(w0['eps']), digits=0)}, "
            f"length cap {fmt(cap_share, digits=3)}, skipped starts {w0['skipped']}, reuse "
            f"{fmt(med(w0['reuse']), digits=2)}, league share {fmt(med(w0['league_share']), digits=2)}, "
            f"{w0['games']} games in {w0['updates']} updates"
            + (f", value v{min(w0['value_versions'])}-v{max(w0['value_versions'])}" if w0["value_versions"] else "")
        )
        self.rows.append(row("P7", "generation health (games/h, median of updates)", gph,
                             baseline=self.baseline_value("P7"),
                             change3=(gph - med(then3), None) if then3 else None,
                             change7=(gph - med(then7), None) if then7 else None,
                             status=status, detail=detail, digits=0))
        self.saved_errors(windows)

    def saved_errors(self, windows):
        w0 = windows.get(0) or {}
        saver = w0.get("saver_errors") or []
        grew = len(saver) > 1 and saver[-1] > saver[0]
        if w0.get("save_errors") or grew:
            self.rule(RED, "hard stop: saved-data write failure",
                      f"{w0.get('save_errors', 0)} games not packed and saver errors {saver[:1]} -> {saver[-1:]} "
                      f"in the last 24 h (docs/long_run_recipe.md, 'Rules')")

    def value(self, force):
        cycles, promoted = value_data(self.s.value_run)
        versions = promoted_versions(promoted)
        self.versions, self.cycles = versions, cycles
        fallback = versions[-1] if versions else None
        current = next((v for v in versions if v["version"] == promoted.get("current")), fallback)
        if versions:
            first = versions[0]
            self.record_baseline("P3", first.get("fixed_ce"), force="P3" in force, version=first["version"])
        slope_all = p3_slope(versions)
        slope3 = p3_slope(versions[-3:]) if len(versions) >= 3 else None
        self.slope_all = slope_all
        self.stall_p3 = None if slope3 is None else slope3 >= 0
        status = GREEN if current else NA
        promoted_cycles = [c for c in cycles if c.get("promoted")]
        if len(promoted_cycles) >= 2:
            gains = []
            for c in promoted_cycles[-2:]:
                if c.get("fixed_diff") is not None:
                    gains.append(c["fixed_diff"] < 0)
            if len(gains) == 2 and not any(gains):
                status = YELLOW
                self.rule(YELLOW, "P3 alarm", "the last two promoted values gained nothing on the fixed held-out set")
        ce = current.get("fixed_ce") if current else None
        then = lambda days: version_at(versions, self.now - days * DAY)  # noqa: E731
        def ch(v):
            return (ce - v["fixed_ce"], None) if v and ce is not None and v.get("fixed_ce") is not None else None

        self.rows.append(row(
            "P3", "fixed-set value CE (current promoted value)", ce, baseline=self.baseline_value("P3"),
            change3=ch(then(3)), change7=ch(then(7)), status=status, digits=4,
            detail=(f"v{current['version']}" if current else "no promoted value yet")
            + (f"; slope per doubling of games {slope_all:+.4f} (gate: -0.006)" if slope_all is not None else "")
            + (f", last 3 promotions {slope3:+.4f}" if slope3 is not None else ""),
        ))
        # P4: the latest candidate's rolling-set gain and the rolling/fixed gap.
        last = cycles[-1] if cycles else None
        status, detail = NA, "no value cycle yet"
        value = se = None
        if last:
            value, se = last.get("rolling_diff"), last.get("rolling_diff_se")
            status = GREEN
            gaps = [c["fixed_ce"] - c["rolling_ce"] for c in cycles
                    if c.get("fixed_ce") is not None and c.get("rolling_ce") is not None]
            detail = (f"cycle {last.get('cycle')}: candidate {last.get('candidate')}, promoted {last.get('promoted')}, "
                      f"fixed diff {fmt(last.get('fixed_diff'), last.get('fixed_diff_se'), 4, True)}, "
                      f"fixed-minus-rolling gap {fmt(gaps[-1], digits=4) if gaps else 'n/a'}, reason "
                      f"{last.get('reason') or '-'}")
            if len(gaps) >= 4 and all(gaps[-i] > gaps[-i - 1] for i in range(1, 4)):
                status = YELLOW
                self.rule(YELLOW, "P4 alarm", "the fixed-minus-rolling CE gap widened 3 cycles in a row: the value may "
                                              "be fitting the newest self-play only")
            if last.get("diverged") or (value is not None and value > 0.01):
                status = RED
                self.rule(RED, "hard stop: value trainer diverging",
                          f"cycle {last.get('cycle')}: rolling CE worse than its start by "
                          f"{fmt(value, digits=4)} ({last.get('reason') or 'diverged'})")
            failures = int(last.get("consecutive_failures") or 0)
            if failures >= 6 or last.get("review"):  # (the trainer's own "review the data mix" flag)
                self.rule(RED, "value: promotion failing", f"{failures} cycles in a row without a promotion: "
                                                           f"review the data mix")
            elif failures >= 3:
                self.rule(YELLOW, "value: promotion failing", f"{failures} cycles in a row without a promotion: "
                                                              f"24-h cycles, twice the sample "
                                                              f"(cycle_hours {last.get('cycle_hours')}, "
                                                              f"sample x{last.get('sample_scale')})")
        self.rows.append(row("P4", "rolling-set CE gain of the latest candidate", value, se, status=status,
                             detail=detail, digits=4))

    def ladder(self, force):
        panel, ratings, records, state = load_ladder(self.s)
        self.records = records
        cands = candidate_ratings(ratings, "2v2")
        times = candidate_times(ratings, records, getattr(self, "versions", []), state)
        schedules = candidate_schedules(ratings, records)
        self.cands, self.times, self.schedules = cands, times, schedules
        dated = sorted((times[n], n) for n in cands if n in times and cands[n].get("update") != 0)
        self.record_baseline("P5", 0.0, 0.0, force="P5" in force, note="u0 (the anchor)")
        cur, prev = p5_block(cands, times, self.now, 0), p5_block(cands, times, self.now, 3)
        self.flat_p5 = self.decline_p5 = None
        status = GREEN if dated else NA
        latest = cands[dated[-1][1]] if dated else None
        if cur and prev:
            d = diff(cur, prev)
            self.flat_p5 = not (d[1] and d[0] >= 2 * d[1])
            if d[1] and d[0] <= -2 * d[1]:
                status = YELLOW
                self.rule(YELLOW, "P5 alarm", f"the ladder's 3-day mean {fmt(*cur, digits=0)} Elo is below the "
                                              f"previous 3 days' {fmt(*prev, digits=0)} by >= 2 SE")
        if cur:
            blocks = [b for b in (p5_block(cands, times, self.now, k) for k in range(0, 120)) if b]
            best = max(blocks, key=lambda b: b[0]) if blocks else None
            d = diff(cur, best) if best else None
            self.decline_p5 = bool(d and d[1] and d[0] <= -2 * d[1])

        def as_of(days):
            older = [(t, n) for t, n in dated if t <= self.now - days * DAY]
            return (cands[older[-1][1]]["elo"], cands[older[-1][1]]["se"]) if older else None

        now_pair = (latest["elo"], latest["se"]) if latest else None
        self.rows.append(row(
            "P5", "daily ladder: newest candidate's 2v2 Elo vs u0", *(now_pair or (None, None)),
            baseline=self.baseline_value("P5"), change3=diff(now_pair, as_of(3)), change7=diff(now_pair, as_of(7)),
            status=status, digits=0,
            detail=(f"{dated[-1][1]}; 3-day mean {fmt(*cur, digits=0) if cur else 'n/a'}, previous 3 days "
                    f"{fmt(*prev, digits=0) if prev else 'n/a'}; {len(dated)} candidates rated"
                    if dated else ("no ladder ratings yet" if panel else "no ladder panel (LADDER_PANEL)")),
        ))
        self.p8(ratings, records)

    def p8(self, ratings, records):
        full = sorted(u for u, k in self.schedules.items() if k == "full")
        status, detail, value, se = NA, "no full-panel candidate yet", None, None
        fits = ratings.get("fits") or {}
        if full:
            name = f"u{full[-1]}"
            two = ((fits.get("2v2") or {}).get("players") or {}).get(name)
            one = ((fits.get("1v3") or {}).get("players") or {}).get(name)
            value, se = (two["elo"], two["se"]) if two else (None, None)
            status = GREEN
            detail = (f"{name}: 2v2 {fmt(value, se, 0, True)}, 1v3 "
                      f"{fmt(one['elo'], one['se'], 0, True) if one else 'n/a'} Elo vs u0")
            decision = {d["fit"]: d for d in ratings.get("decision") or [] if "error" not in d}
            if "1v3" in decision and "2v2" in decision:
                d1, d2 = decision["1v3"], decision["2v2"]
                detail += (f"; vs u{d2['earlier']}: 2v2 {d2['elo']:+.0f} ± {d2['elo_se']:.0f} ({d2['verdict']}), 1v3 "
                           f"{d1['elo']:+.0f} ± {d1['elo_se']:.0f} ({d1['verdict']})")
                if d1["z"] <= -2 and d2["diff"] > 0:
                    status = YELLOW
                    self.rule(YELLOW, "1v3 guard", "the 1v3 rating fell by >= 2 SE while 2v2 rose: the gain is table "
                                                   "composition -- league share 50%, re-check next week")
        checks = value_checks(records)
        if checks:
            last = checks[-1]
            detail += f"; weekly value check u{last[0]}: {fmt(last[1], last[2], 3)} ({last[3]} games)"
            low = [c for c in checks[-2:] if c[1] < 0.47]
            if len(checks) >= 2 and len(low) == 2:
                status = RED
                self.rule(RED, "value: weekly play check", "the weekly value play check scored < 0.47 twice in a row: "
                                                          "stop promoting, revert to the best value by ladder rating, "
                                                          "review")
            elif last[1] < 0.47:
                status = YELLOW if status != RED else status
                self.rule(YELLOW, "value: weekly play check", f"the newest promoted value scored {last[1]:.3f} < 0.47 "
                                                             f"against the value of a week earlier (once)")
        self.rows.append(row("P8", "weekly full panel (latest full candidate, 2v2 Elo vs u0)", value, se,
                             status=status, detail=detail, digits=0))

    def style(self, force):
        records = mt.read_jsonl(self.s.style)
        if not records:
            self.rows.append(row("P6", "style (drift guard)", status=NA, detail="no style statistics yet"))
            return
        last = records[-1]
        m = last.get("main") or {}
        top = m.get("par_share_max") or {}
        withhold = m.get("withhold_rate")
        if "P6" not in self.base or "P6" in force:
            self.base["P6"] = {"value": withhold, "se": None, "date": last.get("time"), "par_share_max": top,
                               "withhold_rate": withhold, "first_3_or": _get(m, "train_first", "3", "or", "mean"),
                               "first_D_or": _get(m, "train_first", "D", "or", "mean")}
        base = self.base["P6"]
        flags = list(last.get("flags") or [])
        # The withhold flag against the report's own baseline when the style run had none.
        if (last.get("drift") or {}).get("withhold", {}).get("flag") is None and base.get("withhold_rate") is not None \
                and withhold is not None and m.get("dividend_decisions", 0) >= 50:
            if withhold >= 2 * max(base["withhold_rate"], 0.01) and base.get("date") != last.get("time"):
                flags.append("withhold")
        status = YELLOW if flags else GREEN
        if flags:
            self.rule(YELLOW, "P6 drift guard",
                      f"{', '.join(flags)} (par top {top.get('corporation')} {fmt(top.get('share'), digits=2)}, "
                      f"withhold {fmt(withhold, digits=3)} vs baseline {fmt(base.get('withhold_rate'), digits=3)}): "
                      f"league share 50% for the next week")
        age = (self.now - t_of(last)) / 3600
        if age > 36:
            status = mt.worst(status, YELLOW)
            self.notes.append(f"style statistics are {age:.0f} h old")
        def withhold_then(days):
            r = next((r for r in reversed(records) if t_of(r) <= self.now - days * DAY), None)
            rate = (r or {}).get("main", {}).get("withhold_rate")
            return None if rate is None else (rate, None)

        d3, d7 = withhold_then(3), withhold_then(7)
        self.rows.append(row(
            "P6", "style: withhold rate (main seats)", withhold, baseline=(base.get("withhold_rate"), None),
            change3=(withhold - d3[0], None) if d3 and withhold is not None else None,
            change7=(withhold - d7[0], None) if d7 and withhold is not None else None, status=status,
            detail=(f"par top {top.get('corporation')} {fmt(top.get('share'), digits=2)} of {m.get('pars', 0)}; mean "
                    f"par {fmt(m.get('par_price_mean'), digits=1)}; first 3 / D in OR "
                    f"{fmt(_get(m, 'train_first', '3', 'or', 'mean'), digits=1)} / "
                    f"{fmt(_get(m, 'train_first', 'D', 'or', 'mean'), digits=1)}; bankruptcies/game "
                    f"{fmt(m.get('bankruptcies_per_game'), digits=3)}; decisions "
                    f"{fmt(_get(m, 'decisions', 'mean'), digits=0)}; "
                    f"length cap {fmt(m.get('length_cap_share'), digits=3)}; {last.get('games')} games"),
        ))

    def maintenance(self):
        history = mt.read_jsonl(self.s.maint.usage_log)
        try:
            sample = mt.disk_sample(self.s.maint, {"instance": self.s.instance, "source": "status_report"})
            history.append(sample)
            if self.write:
                mt.append_jsonl(self.s.maint.usage_log, sample)
        except Exception as e:  # noqa: BLE001
            self.notes.append(f"usage sample failed: {e!r}")
        self.maint = mt.evaluate(self.s.maint, now=self.now, history=history)
        for m in self.maint:
            if m["status"] == RED:
                self.rule(RED, f"{m['id']} {m['name']}", m["detail"])
            elif m["status"] == YELLOW:
                self.rule(YELLOW, f"{m['id']} {m['name']}", m["detail"])
        retention = mt.read_jsonl(self.s.retention)
        self.retention_line = None
        if retention:
            sys.path.insert(0, str(HERE))
            from retention import summary_line

            self.retention_line = f"{summary_line(retention[-1])} ({retention[-1].get('time')})"
            if retention[-1].get("refused"):
                self.rule(YELLOW, "retention refused", self.retention_line)

    def watchdogs(self):
        self.dogs = watchdog_findings(self.s.watch_dirs, self.now)
        for name, active, warn, crit, stale in self.dogs:
            for key, level in active.items():
                if level == "CRIT":
                    hard = any(key.startswith(k) for k in HARD_STOP_KEYS) or key.startswith("disk:")
                    self.rule(RED, f"{'hard stop' if hard else 'watchdog'}: {name} {key}",
                              f"the watchdog of {name} has an active CRIT alert {key}")
                elif level == "WARN":
                    self.rule(YELLOW, f"watchdog: {name} {key}",
                              f"the watchdog of {name} has an active WARN alert {key}")
            if stale is not None and stale > 30:
                self.rule(YELLOW, f"watchdog: {name} silent", f"its heartbeat is {stale:.0f} min old")
        steps = read_json(self.s.state_dir / "daily_steps.json", {}) or {}
        self.steps = steps
        failed = [k for k, v in (steps.get("steps") or {}).items() if v not in (0, "0")]
        if failed and self.now - t_of(steps) < DAY:
            self.rule(YELLOW, "daily jobs", f"failed: {', '.join(failed)} (see the daily job's journal)")

    def stall_and_decline(self):
        parts = {"P1": self.stall_p1, "P3": getattr(self, "stall_p3", None), "P5": self.flat_p5}
        self.stall = all(v is True for v in parts.values())
        text = ", ".join(f"{k} {'yes' if v else 'n/a' if v is None else 'no'}" for k, v in parts.items())
        self.notes.append(f"stall conditions: {text}")
        if self.stall:
            self.rule(RED, "stall", "P1 shows no gain over 5 days, P3's slope over the last 3 promotions is >= 0 and "
                                    "P5 is flat over 6 days: review within a day (stop the main line?)")
        dec = {"P1": self.decline_p1, "P5": self.decline_p5}
        self.notes.append("decline conditions: " + ", ".join(
            f"{k} {'yes' if v else 'n/a' if v is None else 'no'}" for k, v in dec.items()))
        if all(v is True for v in dec.values()):
            self.rule(RED, "decline", "P1 is below its best 5-day mean by >= 3 SE and P5's 3-day mean below its best "
                                      "by >= 2 SE: roll back to the best value by ladder rating, then review")

    def milestones(self):
        start = self.s.run_start or self.base_file.get("created")
        self.day = None
        self.milestone_rows = []
        if not start:
            return
        try:
            self.day = (self.now - datetime.fromisoformat(start).timestamp()) / DAY
        except ValueError:
            return
        for day, label in MILESTONES:
            if self.day < day:
                continue
            ok, why = self.milestone(day)
            recent = self.day - day < 2
            self.milestone_rows.append((label, ok, why))
            if ok is False and recent:
                self.rule(YELLOW, f"milestone {label}", why)

    def milestone(self, day):
        """(met: True / False / None for n/a, explanation)."""
        cands, times = getattr(self, "cands", {}), getattr(self, "times", {})

        def at(d):  # the newest rated candidate whose system dates from before run day d
            start = self.now - self.day * DAY
            older = sorted((times[n], n) for n in cands if n in times and times[n] <= start + d * DAY)
            return older[-1][1] if older else None

        latest = at(self.day)
        if day == 3:
            b, now = self.baseline_value("P1"), self.p1.get(0)
            d = diff(now, b)
            gains = sum(1 for c in getattr(self, "cycles", []) if c.get("promoted") and (c.get("fixed_diff") or 0) < 0)
            if d is None or not d[1]:
                return None, f"P1 vs its baseline n/a; {gains} promotions with fixed-set gains"
            ok = d[0] >= 2 * d[1] and gains >= 2
            return ok, f"P1 vs baseline {fmt(*d, digits=4, signed=True)} (needs >= 2 SE); {gains} promotions with " \
                       f"fixed-set gains (needs 2)"
        if day == 14:
            ce = next((v.get("fixed_ce") for v in reversed(getattr(self, "versions", []))), None)
            start_ce = (self.baseline_value("P3") or (0.920, None))[0] or 0.920
            if not latest:
                return None, "no rated candidate"
            p = cands[latest]
            ok = p["se"] > 0 and p["elo"] >= 2 * p["se"] and ce is not None and ce < start_ce
            return ok, f"{latest} {p['elo']:+.0f} ± {p['se']:.0f} Elo vs u0 (needs >= 2 SE); fixed CE " \
                       f"{fmt(ce, digits=4)} (needs < {start_ce:.3f})"
        if day in (28, 56):
            earlier = at(day / 2)
            if not latest or not earlier or latest == earlier:
                return None, "no candidates to compare"
            d = diff((cands[latest]["elo"], cands[latest]["se"]), (cands[earlier]["elo"], cands[earlier]["se"]))
            ok = d[1] is not None and d[0] >= d[1]
            why = f"{latest} vs {earlier}: {d[0]:+.0f} ± {d[1]:.0f} Elo (needs >= 1 SE)"
            if day == 28:
                now_x = excess_vs(self.records, latest, "gen4critic", "1v3")
                base_x = excess_vs(self.records, "u0", "gen4critic", "1v3")
                if now_x and base_x:
                    ok = ok and now_x[0] > base_x[0]
                    why += f"; 1v3 vs gen4critic {now_x[0]:+.3f} vs day 0 {base_x[0]:+.3f}"
            else:
                slope = getattr(self, "slope_all", None)
                ok = ok and slope is not None and slope <= -0.003
                why += f"; P3 slope per doubling {fmt(slope, digits=4)} (needs <= -0.003)"
            return ok, why
        if day == 112:
            pts = [((times[n] - (self.now - self.day * DAY)) / DAY, cands[n]["elo"], cands[n]["se"]) for n in cands
                   if n in times and 56 <= (times[n] - (self.now - self.day * DAY)) / DAY <= 84 and cands[n]["se"]]
            fit = wls_slope(pts)
            if not fit:
                return None, "not enough week 8-12 candidates for a slope"
            return True, (f"week 8-12 slope {fit[0]:+.1f} ± {fit[1]:.1f} Elo/day: "
                          + ("extend" if fit[0] > fit[1] else "end the run"))
        return None, ""

    # -- output

    def build(self, force=()):
        force = set(force)
        if self.base_file.get("created") is None:
            self.base_file["created"] = self.stamp()
        for section in (lambda: self.generator(force), lambda: self.value(force), lambda: self.ladder(force),
                        lambda: self.style(force), self.maintenance, self.watchdogs):
            try:
                section()
            except Exception as e:  # noqa: BLE001 -- a broken input must not stop the report
                import traceback

                traceback.print_exc()
                self.notes.append(f"a section failed: {e!r}")
        for name in ("p1", "p2"):
            if not hasattr(self, name):
                setattr(self, name, {})
        for name in ("stall_p1", "decline_p1", "flat_p5", "decline_p5"):
            if not hasattr(self, name):
                setattr(self, name, None)
        self.stall_and_decline()
        self.milestones()
        self.status = mt.worst(GREEN, *(r["level"] for r in self.rules))
        return self

    def markdown(self):
        stamp = datetime.fromtimestamp(self.now).astimezone()
        day = f"day {self.day:.0f}" if self.day is not None else "day n/a"
        lines = [f"## {stamp:%Y-%m-%d %H:%M} ({day}) -- {self.status.upper()}", ""]
        if self.rules:
            for r in sorted(self.rules, key=lambda r: -mt.RANK[r["level"]]):
                lines.append(f"- **{r['level'].upper()}** {r['rule']}: {r['message']}")
        else:
            lines.append("- no rule fired")
        lines += ["", "| # | metric | now | day-0 baseline | 3-day change | 7-day change | status | detail |",
                  "|---|---|---|---|---|---|---|---|"]
        for r in sorted(self.rows, key=lambda r: r["id"]):
            d = r["digits"]
            base = fmt(*r["baseline"], digits=d) if r["baseline"] else "n/a"
            ch3 = fmt(*r["change3"], digits=d, signed=True) if r["change3"] else "n/a"
            ch7 = fmt(*r["change7"], digits=d, signed=True) if r["change7"] else "n/a"
            lines.append(f"| {r['id']} | {r['name']} | {r['text']} | {base} | {ch3} | {ch7} | {r['status']} | "
                         f"{r['detail'].replace('|', '/')} |")
        if self.member_rows:
            lines += ["", "League members, last 24 h (main seats' win / net-worth excess over their fair share):", "",
                      "| member | kind | games | win excess | net-worth excess | win excess, 7 days |",
                      "|---|---|---|---|---|---|"]
            for name, kind, games, win, worth, week in self.member_rows:
                lines.append(f"| {name} | {kind} | {games} | {fmt(*win, digits=4, signed=True) if win else 'n/a'} | "
                             f"{fmt(*worth, digits=4, signed=True) if worth else 'n/a'} | "
                             f"{fmt(*week, digits=4, signed=True) if week else 'n/a'} |")
        lines += ["", "| # | maintenance | status | detail |", "|---|---|---|---|"]
        for m in getattr(self, "maint", []):
            lines.append(f"| {m['id']} | {m['name']} | {m['status']} | {m['detail'].replace('|', '/')} |")
        lines.append("")
        lines.append(f"- retention: {self.retention_line or 'n/a (no retention run yet)'}")
        for name, active, warn, crit, stale in getattr(self, "dogs", []):
            act = ", ".join(f"{k} ({v})" for k, v in active.items()) or "none"
            lines.append(f"- watchdog {name}: active {act}; last 24 h {warn} WARN, {crit} CRIT")
        steps = (getattr(self, "steps", {}) or {}).get("steps")
        if steps:
            lines.append("- daily jobs: " + ", ".join(f"{k} {'ok' if v in (0, '0') else f'exit {v}'}"
                                                      for k, v in steps.items()))
        for label, ok, why in getattr(self, "milestone_rows", []):
            lines.append(f"- milestone {label}: {'met' if ok else 'n/a' if ok is None else 'NOT met'} ({why})")
        for note in self.notes:
            lines.append(f"- {note}")
        return "\n".join(lines) + "\n\n"

    def summary(self):
        fired = [r for r in self.rules if r["level"] == self.status]
        return "; ".join(f"{r['rule']}: {r['message']}" for r in fired)[:900] or "all green"

    def record(self):
        return {"time": self.stamp(), "status": self.status, "day": self.day, "rules": self.rules,
                "metrics": [{k: v for k, v in r.items() if k != "digits"} for r in self.rows],
                "maintenance": getattr(self, "maint", []), "notes": self.notes}


def _get(d, *keys):
    for k in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for flag in ("workdir", "gen-run-dir", "value-run-dir", "panel", "state-dir", "watch-dirs", "log", "style-stats",
                 "retention-log", "run-start"):
        parser.add_argument(f"--{flag}", default=None)
    parser.add_argument("--set-baseline", nargs="?", const="all", default=None,
                        help="re-record these metrics' baselines now (comma-separated, or all)")
    parser.add_argument("--now", default=None, help="report as of this ISO time (tests)")
    parser.add_argument("--no-alert", action="store_true")
    parser.add_argument("--print-only", action="store_true", help="print the section; write nothing")
    args = parser.parse_args(argv)
    settings = Settings(args)
    now = datetime.fromisoformat(args.now).timestamp() if args.now else None
    force = ()
    if args.set_baseline:
        force = ("P1", "P2", "P3", "P5", "P6", "P7") if args.set_baseline == "all" else args.set_baseline.split(",")
    report = Report(settings, now, write=not args.print_only).build(force)
    text = report.markdown()
    print(text, end="")
    if args.print_only:
        return 0
    settings.state_dir.mkdir(parents=True, exist_ok=True)
    tmp = settings.baseline.with_suffix(".tmp")
    tmp.write_text(json.dumps(report.base_file, indent=1))
    tmp.replace(settings.baseline)
    mt.append_jsonl(settings.state_dir / "status.jsonl", report.record())
    settings.log.parent.mkdir(parents=True, exist_ok=True)
    new = not settings.log.exists()
    with settings.log.open("a") as f:
        if new:
            f.write("# Long-run log\n\nDaily status reports (scripts/longrun/status_report.py) and weekly reviews.\n\n")
        f.write(text)
    if report.status in (YELLOW, RED) and not args.no_alert:
        import watchdog

        watchdog.send_alert(settings.state_dir, settings.instance, "CRIT" if report.status == RED else "WARN",
                            "status", f"daily report {report.status.upper()}: {report.summary()} ({settings.log})",
                            settings.alert_command)
    return 0


if __name__ == "__main__":
    sys.exit(main())
