#!/usr/bin/env python3
"""The long run's maintenance metrics M1-M9 (docs/long_run_recipe.md, "Maintenance"). Standard library only, like
the watchdog that imports it: the watchdog appends samples here (``USAGE_LOG``: disk, sizes by category, the lowest
MemAvailable, swap, GPU throttling per hour; ``$STATE_DIR/rss.jsonl``: the RSS of every process of its unit) and
scans the kernel log for Xid errors; the daily status report evaluates the histories with :func:`evaluate`.

    maintenance.py sample [--append]     one usage sample now (printed; appended to USAGE_LOG with --append)
    maintenance.py evaluate              M1-M9 from the histories, as the daily report computes them

Settings come from the environment (the daily job's .env; thresholds default to the recipe's):

    WORKDIR                  relative paths resolve here (default .)
    USAGE_LOG                the usage history (default longrun/usage.jsonl)
    USAGE_PATHS              categories for M2: "saved=saved_games/lr1;checkpoints=model_checkpoints_search/lr1:
                             model_checkpoints_value/lr1;evals=logs/eval;logs=logs/*.log:longrun" (':' between paths,
                             globs allowed)
    M2_EXPECTED_GB_PER_DAY   "saved=1.4,checkpoints=0.5,evals=0.06,logs=0.05" (yellow above M2_FACTOR x, default 2)
    WORKTREES_DIR            M3: the session worktrees (.claude/worktrees of the source repository); empty = n/a
    RSS_DIRS                 M4: the state dirs whose rss.jsonl to read (':' between them)
    INTEGRITY_LOG            M6: integrity_check.py's log
    HOST_HOLD                M7: path of host_hold.sh (default: beside this file); M7_HOLD_CHECK=0 to skip
    VALUE_RUN_DIR            M8: the value trainer's run (cycles.jsonl)
    BACKUP_DIR               M9: the off-disk backup (newest file's age); empty = n/a
    M1_YELLOW_DAYS 30, M1_RED_DAYS 14, M1_RED_FREE_GB 150, M3_YELLOW_GB 200, M3_RED_GB 300, M4_LEAK_GB_PER_DAY 2,
    M4_SWAP_MIN_GB 0.5, M4_MEM_RED_GB 30, M5_THROTTLE_FRACTION 0.10, M8_TARGET_HOURS 1, M8_FACTOR 2, M9_MAX_DAYS 8,
    DU_TIMEOUT_SEC 600, M5_XID_CHECK 1 (0: don't scan the kernel log), REBOOT_REQUIRED_FILE /var/run/reboot-required
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

GREEN, YELLOW, RED, NA = "green", "yellow", "red", "n/a"
RANK = {NA: -1, GREEN: 0, YELLOW: 1, RED: 2}
DAY = 86400.0


def env_float(name, default):
    value = os.environ.get(name, "").strip()
    return float(value) if value else float(default)


def parse_map(text, value=str):
    """'a=x:y;b=z' -> {'a': value('x:y'), ...} (';' or ',' between entries)."""
    out = {}
    for part in text.replace(",", ";").split(";"):
        key, sep, val = part.partition("=")
        if sep and key.strip():
            out[key.strip()] = value(val.strip())
    return out


def worst(*statuses):
    return max(statuses, key=lambda s: RANK.get(s, -1)) if statuses else NA


class Settings:
    def __init__(self, env=None):
        env = os.environ if env is None else env
        get = lambda k, d="": env.get(k, d).strip()  # noqa: E731
        num = lambda k, d: float(get(k) or d)  # noqa: E731
        self.workdir = Path(get("WORKDIR", ".") or ".")
        self.usage_log = self.path(get("USAGE_LOG") or "longrun/usage.jsonl")
        self.usage_paths = parse_map(get("USAGE_PATHS"))
        self.expected = parse_map(get("M2_EXPECTED_GB_PER_DAY", "saved=1.4,checkpoints=0.5,evals=0.06,logs=0.05"),
                                  float)
        self.m2_factor = num("M2_FACTOR", 2)
        self.worktrees = self.path(get("WORKTREES_DIR")) if get("WORKTREES_DIR") else None
        self.rss_dirs = [self.path(p) for p in get("RSS_DIRS").split(":") if p]
        self.integrity_log = self.path(get("INTEGRITY_LOG")) if get("INTEGRITY_LOG") else None
        self.host_hold = Path(get("HOST_HOLD") or Path(__file__).resolve().parent / "host_hold.sh")
        self.hold_check = get("M7_HOLD_CHECK", "1") != "0"
        self.value_run = self.path(get("VALUE_RUN_DIR")) if get("VALUE_RUN_DIR") else None
        self.backup = self.path(get("BACKUP_DIR")) if get("BACKUP_DIR") else None
        self.m1_yellow_days = num("M1_YELLOW_DAYS", 30)
        self.m1_red_days = num("M1_RED_DAYS", 14)
        self.m1_red_free_gb = num("M1_RED_FREE_GB", 150)
        self.m3_yellow_gb = num("M3_YELLOW_GB", 200)
        self.m3_red_gb = num("M3_RED_GB", 300)
        self.m4_leak = num("M4_LEAK_GB_PER_DAY", 2)
        self.m4_swap = num("M4_SWAP_MIN_GB", 0.5)
        self.m4_mem_red = num("M4_MEM_RED_GB", 30)
        self.m5_throttle = num("M5_THROTTLE_FRACTION", 0.10)
        self.m8_target = num("M8_TARGET_HOURS", 1)
        self.m8_factor = num("M8_FACTOR", 2)
        self.m9_days = num("M9_MAX_DAYS", 8)
        self.du_timeout = num("DU_TIMEOUT_SEC", 600)
        self.reboot_file = Path(get("REBOOT_REQUIRED_FILE") or "/var/run/reboot-required")
        self.xid_check = get("M5_XID_CHECK", "1") != "0"

    def path(self, p):
        p = Path(p)
        return p if p.is_absolute() else self.workdir / p


# ----------------------------------------------------------------------------- files


def append_jsonl(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record, separators=(",", ":")) + "\n")


def read_jsonl(path, since=None):
    """The JSON lines of ``path`` (missing file: []); ``since`` (unix time): only records whose ``time`` is later."""
    records = []
    try:
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if since is not None and record_time(r) < since:
                    continue
                records.append(r)
    except OSError:
        pass
    return records


def record_time(record, key="time"):
    try:
        return datetime.fromisoformat(str(record[key])).timestamp()
    except (KeyError, TypeError, ValueError):
        return 0.0


def now_iso():
    return datetime.now().astimezone().isoformat(timespec="seconds")


# ----------------------------------------------------------------------------- probes


def run(cmd, timeout=20):
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return out.returncode, out.stdout, out.stderr.strip()
    except (OSError, subprocess.TimeoutExpired) as e:
        return 127, "", str(e)


def expand(patterns, workdir):
    paths = []
    for pattern in patterns:
        full = pattern if os.path.isabs(pattern) else str(Path(workdir) / pattern)
        found = sorted(glob.glob(full))
        paths += found if found else []
    return paths


def du_bytes(paths, timeout=600):
    """Disk usage of ``paths`` together (hard links once), by ``du``; None if du failed or timed out."""
    paths = [p for p in paths if os.path.lexists(p)]
    if not paths:
        return 0
    code, out, _ = run(["nice", "-n", "19", "du", "-s", "-c", "-B1", *paths], timeout=timeout)
    if code not in (0, 1) or not out.strip():
        return None
    try:
        return int(out.strip().splitlines()[-1].split()[0])
    except (ValueError, IndexError):
        return None


def meminfo():
    out = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, _, rest = line.partition(":")
            out[key] = int(rest.split()[0]) / 2**20  # GiB
    except (OSError, ValueError, IndexError):
        pass
    return out


def disk_sample(settings, extra=None):
    """A usage sample now: free space on the workdir's filesystem, the categories' sizes, the worktrees' size, swap
    in use. ``extra``: fields the caller adds (the watchdog: its lowest MemAvailable, GPU throttling per hour)."""
    usage = shutil.disk_usage(settings.workdir)
    sample = {
        "time": now_iso(),
        "free_gb": usage.free / 1e9,
        "total_gb": usage.total / 1e9,
        "sizes_gb": {},
    }
    for name, spec in settings.usage_paths.items():
        size = du_bytes(expand(spec.split(":"), settings.workdir), settings.du_timeout)
        sample["sizes_gb"][name] = None if size is None else size / 1e9
    if settings.worktrees:
        size = du_bytes([str(settings.worktrees)], settings.du_timeout)
        sample["worktrees_gb"] = None if size is None else size / 1e9
    mem = meminfo()
    if "SwapTotal" in mem:
        sample["swap_used_gb"] = (mem["SwapTotal"] - mem.get("SwapFree", 0.0)) * 2**30 / 1e9
    if "MemAvailable" in mem:
        sample["mem_available_gb"] = mem["MemAvailable"]
    sample.update(extra or {})
    return sample


def proc_stat(pid):
    """(start time in clock ticks, RSS in bytes, command line) of a process, or None."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        start = int(stat.rsplit(")", 1)[1].split()[19])
        rss = 0
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                rss = int(line.split()[1]) * 1024
        cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
        return start, rss, cmd[:160]
    except (OSError, ValueError, IndexError):
        return None


def rss_sample(control_group):
    """The RSS of every process in a unit's cgroup and the cgroup's memory.current; None without a cgroup."""
    if not control_group:
        return None
    base = Path("/sys/fs/cgroup") / control_group.lstrip("/")
    try:
        pids = [int(p) for p in (base / "cgroup.procs").read_text().split()]
    except (OSError, ValueError):
        return None
    try:
        current = int((base / "memory.current").read_text())
    except (OSError, ValueError):
        current = None
    procs = []
    for pid in pids:
        info = proc_stat(pid)
        if info:
            procs.append([pid, info[0], info[1], info[2]])
    return {"time": now_iso(), "cgroup_bytes": current, "procs": procs}


def xid_lines(since_epoch, until_epoch=None):
    """Kernel-log lines with an NVIDIA Xid error since ``since_epoch``: (lines, None), or (None, why) when the
    kernel log can't be read (journalctl -k needs the adm / systemd-journal group; dmesg is often restricted)."""
    cmd = ["journalctl", "-k", "--no-pager", "-q", "-o", "short-iso", f"--since=@{int(since_epoch)}"]
    if until_epoch:
        cmd.append(f"--until=@{int(until_epoch)}")
    code, out, err = run(cmd, timeout=60)
    if code == 0:
        return [line for line in out.splitlines() if "NVRM: Xid" in line], None
    code2, out2, err2 = run(["dmesg", "--time-format", "iso"], timeout=30)
    if code2 == 0:
        return [line for line in out2.splitlines() if "NVRM: Xid" in line], None
    return None, f"journalctl: {(err or out).strip()[:120] or code}; dmesg: {(err2 or out2).strip()[:120] or code2}"


def hold_check(host_hold):
    """(all held, packages not held, error) from ``host_hold.sh check``."""
    if not Path(host_hold).is_file():
        return None, [], f"{host_hold} not found"
    code, out, err = run(["bash", str(host_hold), "check"], timeout=60)
    missing = [line.strip() for line in out.splitlines() if line.strip()]
    if code == 0:
        return True, [], None
    if code == 1:
        return False, missing, None
    return None, [], (err or out).strip()[:200] or f"exit {code}"


# ----------------------------------------------------------------------------- evaluation


def metric(mid, name, value, status, detail="", **extra):
    return {"id": mid, "name": name, "value": value, "status": status, "detail": detail, **extra}


def slope_per_day(points):
    """Least-squares slope (per day) of [(unix time, value)]; None with fewer than two distinct times."""
    pts = [(t, v) for t, v in points if v is not None]
    if len(pts) < 2:
        return None
    mt = sum(t for t, _ in pts) / len(pts)
    mv = sum(v for _, v in pts) / len(pts)
    sxx = sum((t - mt) ** 2 for t, _ in pts)
    if sxx <= 0:
        return None
    return sum((t - mt) * (v - mv) for t, v in pts) / sxx * DAY


def window(history, now, days):
    return [r for r in history if now - record_time(r) <= days * DAY]


def m1_disk(s, history, now):
    if not history:
        return metric("M1", "disk: days to full", None, NA, "no usage samples yet")
    last = history[-1]
    free = last.get("free_gb")
    recent = window(history, now, 7)
    span = (record_time(recent[-1]) - record_time(recent[0])) / DAY if len(recent) > 1 else 0.0
    rate = slope_per_day([(record_time(r), r.get("total_gb", 0) - r.get("free_gb", 0)) for r in recent]) \
        if span >= 0.5 else None
    days = None if rate is None else (float("inf") if rate <= 0 else free / rate)
    status = GREEN
    if free is not None and free < s.m1_red_free_gb:
        status = RED
    elif days is not None and days < s.m1_red_days:
        status = RED
    elif days is not None and days < s.m1_yellow_days:
        status = YELLOW
    if days is None:
        detail = f"{free:.0f} GB free; growth rate n/a (history spans {span:.1f} days)"
    else:
        detail = f"{free:.0f} GB free, growing {rate:+.1f} GB/day over {span:.1f} days"
    return metric("M1", "disk: days to full", days, status, detail, free_gb=free, rate_gb_per_day=rate)


def m2_growth(s, history, now):
    recent = window(history, now, 7)
    names = sorted({k for r in recent for k in (r.get("sizes_gb") or {})})
    if not names:
        return metric("M2", "growth by category", None, NA, "no category sizes (USAGE_PATHS)")
    rates, status, parts = {}, GREEN, []
    span = (record_time(recent[-1]) - record_time(recent[0])) / DAY if len(recent) > 1 else 0.0
    for name in names:
        pts = [(record_time(r), (r.get("sizes_gb") or {}).get(name)) for r in recent]
        rate = slope_per_day(pts) if span >= 0.5 else None
        rates[name] = rate
        expected = s.expected.get(name)
        flag = rate is not None and expected is not None and rate > s.m2_factor * expected
        if flag:
            status = YELLOW
        parts.append(f"{name} {'n/a' if rate is None else f'{rate:+.2f}'} GB/day"
                     + (f" (expected {expected:g}{', OVER ' + format(s.m2_factor, 'g') + 'x' if flag else ''})"
                        if expected is not None else ""))
    if all(r is None for r in rates.values()):
        status = NA
    return metric("M2", "growth by category", rates, status, "; ".join(parts))


def m3_worktrees(s, history):
    if not s.worktrees:
        return metric("M3", "session worktrees", None, NA, "WORKTREES_DIR not set")
    sizes = [r.get("worktrees_gb") for r in history if r.get("worktrees_gb") is not None]
    if not sizes:
        return metric("M3", "session worktrees", None, NA, f"no size of {s.worktrees} yet")
    size = sizes[-1]
    status = RED if size > s.m3_red_gb else YELLOW if size > s.m3_yellow_gb else GREEN
    return metric("M3", "session worktrees", size, status, f"{size:.0f} GB in {s.worktrees}")


def rss_leaks(records, now, leak_gb_per_day, days=2):
    """Processes (by pid and start time) whose RSS grew more than ``leak_gb_per_day`` in each of the last ``days``
    days, and the unit's cgroup if it did: [(label, [growth per day in GB])]."""
    series = {}
    for r in records:
        t = record_time(r)
        for pid, start, rss, cmd in r.get("procs") or []:
            series.setdefault((pid, start), {"cmd": cmd, "points": []})["points"].append((t, rss / 1e9))
        if r.get("cgroup_bytes") is not None:
            series.setdefault("cgroup", {"cmd": "the unit's cgroup", "points": []})["points"].append(
                (t, r["cgroup_bytes"] / 1e9))
    leaks = []
    for key, s in series.items():
        pts = sorted(s["points"])
        if not pts or pts[0][0] > now - days * DAY + 3600:  # not watched for the whole period
            continue
        growth = []
        for d in range(days, 0, -1):
            lo, hi = now - d * DAY, now - (d - 1) * DAY
            inside = [v for t, v in pts if lo - 1800 <= t <= hi + 1800]
            if len(inside) < 2:
                break
            before = min(pts, key=lambda p: abs(p[0] - lo))[1]
            after = min(pts, key=lambda p: abs(p[0] - hi))[1]
            growth.append(after - before)
        if len(growth) == days and all(g > leak_gb_per_day for g in growth):
            label = s["cmd"] if key == "cgroup" else f"pid {key[0]} ({s['cmd'][:60]})"
            leaks.append((label, growth))
    return leaks


def m4_memory(s, history, rss_by_dir, now):
    day = window(history, now, 1)
    lows = [r.get("mem_available_min_gb", r.get("mem_available_gb")) for r in day]
    lows = [v for v in lows if v is not None]
    low = min(lows) if lows else None
    swap = next((r.get("swap_used_gb") for r in reversed(day) if r.get("swap_used_gb") is not None), None)
    leaks = []
    for directory, records in rss_by_dir.items():
        for label, growth in rss_leaks(records, now, s.m4_leak):
            leaks.append(f"{Path(directory).name}: {label} +{', +'.join(f'{g:.1f}' for g in growth)} GB/day")
    status = GREEN if (low is not None or rss_by_dir) else NA
    notes = []
    if low is not None:
        notes.append(f"lowest MemAvailable {low:.0f} GiB")
    if swap is not None:
        notes.append(f"swap in use {swap:.1f} GB")
    if leaks:
        status = YELLOW
        notes.append("RSS growing: " + "; ".join(leaks))
    if swap is not None and swap > s.m4_swap:
        status = worst(status, YELLOW)
    if low is not None and low < s.m4_mem_red:
        status = RED
    return metric("M4", "RAM", low, status, ", ".join(notes) or "no samples", swap_gb=swap, leaks=leaks)


def m5_gpu(s, history, now, xid):
    """``xid``: (lines, error) from :func:`xid_lines` over the last day."""
    lines, error = xid
    hours = [h for r in window(history, now, 1) for h in (r.get("throttle_hours") or [])]
    worst_hour = max((h.get("fraction", 0.0) for h in hours), default=None)
    status, notes = GREEN, []
    if worst_hour is not None:
        notes.append(f"worst hour throttled {worst_hour:.0%}")
        if worst_hour > s.m5_throttle:
            status = YELLOW
    if lines is None:
        notes.append(f"kernel log unreadable ({error})")
        if worst_hour is None:
            status = NA
    elif lines:
        status = RED
        notes.append(f"{len(lines)} Xid error(s): {lines[-1][:160]}")
    else:
        notes.append("no Xid errors")
    return metric("M5", "GPU", len(lines) if lines is not None else None, status, ", ".join(notes))


def m6_integrity(s, now):
    if not s.integrity_log:
        return metric("M6", "saved-data integrity", None, NA, "INTEGRITY_LOG not set")
    records = read_jsonl(s.integrity_log)
    if not records:
        return metric("M6", "saved-data integrity", None, NA, "no integrity check yet")
    last = records[-1]
    age = (now - record_time(last)) / 3600
    bad = last.get("mismatches") or []
    if bad:
        return metric("M6", "saved-data integrity", len(bad), RED,
                      f"{len(bad)} mismatch(es) at {last.get('time')}: {str(bad[0])[:160]}")
    status = YELLOW if age > 36 else GREEN
    detail = (f"{last.get('games', 0)} games, {last.get('rows', 0)} rows rebuilt bit for bit, "
              f"{last.get('checksums_verified', 0)} shard checksums verified ({age:.0f} h ago)")
    if last.get("error"):
        status, detail = YELLOW, f"the check failed to run: {last['error']}"
    return metric("M6", "saved-data integrity", 0, status, detail)


def m7_host(s, hold=None):
    """``hold``: (all held, missing, error) from :func:`hold_check` (None: run it now if M7_HOLD_CHECK)."""
    status, notes = GREEN, []
    if s.reboot_file.exists():
        status = YELLOW
        notes.append(f"a reboot is pending ({s.reboot_file})")
    if s.hold_check:
        held, missing, error = hold if hold is not None else hold_check(s.host_hold)
        if held is False:
            status = RED
            notes.append(f"not held: {' '.join(missing[:8])}{' ...' if len(missing) > 8 else ''} "
                         f"(sudo bash scripts/longrun/host_hold.sh hold)")
        elif held is None:
            notes.append(f"hold check failed: {error}")
        else:
            notes.append("driver / kernel held")
    return metric("M7", "host", None, status, "; ".join(notes) or "no reboot pending")


def m8_cycle(s):
    if not s.value_run:
        return metric("M8", "value-trainer cycle time", None, NA, "VALUE_RUN_DIR not set")
    cycles = read_jsonl(s.value_run / "cycles.jsonl")
    timed = [c for c in cycles if c.get("seconds") is not None]
    if not timed:
        return metric("M8", "value-trainer cycle time", None, NA, "no value cycle yet")
    last = timed[-1]
    hours = float(last["seconds"]) / 3600
    scale = float(last.get("sample_scale_used", last.get("sample_scale")) or 1)  # the cycle's own sample scale
    target = s.m8_target * scale
    status = YELLOW if hours > s.m8_factor * target else GREEN
    return metric("M8", "value-trainer cycle time", hours, status,
                  f"cycle {last.get('cycle')} took {hours:.2f} h (target {target:g} h"
                  f"{', sample x' + format(scale, 'g') if scale != 1 else ''})")


def m9_backup(s, now):
    if not s.backup:
        return metric("M9", "backup age", None, NA, "BACKUP_DIR not set")
    newest = 0.0
    for root, _, files in os.walk(s.backup):
        for name in files:
            try:
                newest = max(newest, os.stat(os.path.join(root, name)).st_mtime)
            except OSError:
                pass
    if not newest:
        return metric("M9", "backup age", None, YELLOW, f"nothing in {s.backup}")
    days = (now - newest) / DAY
    return metric("M9", "backup age", days, YELLOW if days > s.m9_days else GREEN, f"newest copy {days:.1f} days old")


def evaluate(s, now=None, history=None, rss_by_dir=None, xid=None, hold=None):
    """M1-M9 (M6 from integrity_check.py's log). The histories default to the files the settings name; ``xid`` /
    ``hold`` default to scanning the kernel log for the last day / running ``host_hold.sh check``."""
    now = time.time() if now is None else now
    history = read_jsonl(s.usage_log) if history is None else history
    if rss_by_dir is None:
        rss_by_dir = {str(d): read_jsonl(Path(d) / "rss.jsonl", since=now - 3 * DAY) for d in s.rss_dirs}
    if xid is None:
        xid = xid_lines(now - DAY, now) if s.xid_check else (None, "M5_XID_CHECK=0")
    return [
        m1_disk(s, history, now),
        m2_growth(s, history, now),
        m3_worktrees(s, history),
        m4_memory(s, history, rss_by_dir, now),
        m5_gpu(s, history, now, xid),
        m6_integrity(s, now),
        m7_host(s, hold),
        m8_cycle(s),
        m9_backup(s, now),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("sample", "evaluate"))
    parser.add_argument("--append", action="store_true", help="sample: append to USAGE_LOG")
    args = parser.parse_args()
    s = Settings()
    if args.mode == "sample":
        sample = disk_sample(s)
        if args.append:
            append_jsonl(s.usage_log, sample)
        print(json.dumps(sample, indent=1))
        return 0
    for m in evaluate(s):
        print(f"{m['id']:<3} {m['status']:<7} {m['name']}: {m['detail']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
