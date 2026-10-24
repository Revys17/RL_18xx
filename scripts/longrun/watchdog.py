#!/usr/bin/env python3
"""The long-run watchdog (rl18xx-longrun-watchdog@.service; docs/longrun_runbook.md). Standard library only,
run with the system python3, so it keeps working whatever happens to the run's environment.

    watchdog.py [--instance NAME]           check every WATCH_INTERVAL_SEC, alert on changes
    watchdog.py [--instance NAME] --once    one round of checks with alerts (tests)
    watchdog.py [--instance NAME] status    print the run's state and every active alert, change nothing

Alerts on: a crash or an automatic restart, a crash loop (CRASH_LOOP_RESTARTS automatic restarts within
CRASH_LOOP_WINDOW_MIN, or systemd's start limit), a non-finite-loss stop, a refused start (deployment
modified, snapshot unreadable), the unit failing for any other reason (an OOM kill among them), no new
history.jsonl line for STALL_MINUTES while the run is up, free disk under DISK_MIN_GB, system MemAvailable
under MEM_AVAILABLE_MIN_GB, the run's cgroup at its MemoryHigh, GPU temperature (GPU_TEMP_MAX_C, or less than
GPU_TLIMIT_MIN_C below its slowdown limit), thermal / hardware throttling, nvidia-smi failing, and NVIDIA
libraries upgraded under the loaded kernel module (the next CUDA process fails until a reboot), and (HOST_CHECKS=1,
on one instance) NVIDIA Xid errors in the kernel log.

The progress file depends on the instance's KIND (launch.sh): generator (search-learning) RUN_DIR/history.jsonl,
value (main.py value-train) RUN_DIR/status.json (its heartbeat: every poll and evaluation), ladder
(eval_ladder.py run) RUN_DIR/heartbeat; PROGRESS_FILE
overrides it.

Maintenance samples for the daily report (maintenance.py): every RSS_EVERY_MIN (60) the RSS of every process of the
unit to $STATE_DIR/rss.jsonl; with HOST_CHECKS=1, every USAGE_EVERY_MIN (360) a usage sample to USAGE_LOG (free
disk, sizes by category, worktrees, the lowest MemAvailable since the last sample, swap, GPU throttling per hour)
and every XID_CHECK_MIN (5) a scan of the kernel log for Xid errors (a CRIT alert).

Each alert is a line in $STATE_DIR/alerts.log and on stdout (the journal); warnings and worse also go to a
desktop notification when a notification server is running (notify-send, else D-Bus) and to ALERT_COMMAND
if set (run by sh with ALERT_LEVEL, ALERT_KEY, ALERT_MESSAGE in its environment). A condition alerts when it
starts, again every ALERT_REPEAT_MIN while it lasts, and logs when it clears.

Settings come from the environment (the instance's ~/.config/rl18xx-longrun/<instance>.env): WORKDIR,
RUN_DIR, LOG and STATE_DIR as launch.sh reads them, and the thresholds above.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import maintenance  # noqa: E402  (standard library only, beside this file)

INFO, WARN, CRIT = "INFO", "WARN", "CRIT"
EXIT_MEANINGS = {
    75: "the training stopped on a non-finite loss or gradient (not restarted: needs a decision)",
    76: "the frozen deployment changed since it was built, or the instance config is incomplete",
    77: "the snapshot to resume from is missing or unreadable",
}
# The progress file of each kind of instance: a line (or a touch) means the run is moving.
PROGRESS_FILES = {"generator": "history.jsonl", "value": "status.json", "ladder": "heartbeat"}
THROTTLE_FIELDS = (
    "clocks_event_reasons.sw_thermal_slowdown",
    "clocks_event_reasons.hw_thermal_slowdown",
    "clocks_event_reasons.hw_slowdown",
    "clocks_event_reasons.hw_power_brake_slowdown",
)


def load_env_file(instance):
    """The instance's ~/.config/rl18xx-longrun/<instance>.env when run by hand (systemd sets these itself)."""
    path = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "rl18xx-longrun" / f"{instance}.env"
    if "WORKDIR" in os.environ or not path.is_file():
        return
    for line in path.read_text().splitlines():
        key, sep, value = line.partition("=")
        if sep and not key.lstrip().startswith("#"):
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            os.environ.setdefault(key.strip(), value)


def env_float(name, default):
    value = os.environ.get(name, "").strip()
    return float(value) if value else float(default)


class Settings:
    def __init__(self, instance):
        self.instance = instance
        self.workdir = Path(os.environ.get("WORKDIR", "."))
        resolve = lambda p: p if p.is_absolute() else self.workdir / p  # noqa: E731
        self.run_dir = resolve(Path(os.environ.get("RUN_DIR", "")))
        self.log = resolve(Path(os.environ.get("LOG", f"logs/longrun_{instance}.log")))
        self.state_dir = resolve(Path(os.environ.get("STATE_DIR", f"longrun/{instance}")))
        self.unit = os.environ.get("UNIT", f"rl18xx-longrun@{instance}.service")
        self.interval = env_float("WATCH_INTERVAL_SEC", 60)
        self.stall_minutes = env_float("STALL_MINUTES", 60)
        self.disk_min_gb = env_float("DISK_MIN_GB", 100)
        self.mem_available_min_gb = env_float("MEM_AVAILABLE_MIN_GB", 16)
        self.gpu_temp_max = env_float("GPU_TEMP_MAX_C", 87)
        self.gpu_tlimit_min = env_float("GPU_TLIMIT_MIN_C", 5)
        self.gpu_check = os.environ.get("GPU_CHECK", "1") != "0"
        self.crash_loop_restarts = int(env_float("CRASH_LOOP_RESTARTS", 3))
        self.crash_loop_window = env_float("CRASH_LOOP_WINDOW_MIN", 60) * 60
        self.repeat = env_float("ALERT_REPEAT_MIN", 240) * 60
        self.alert_command = os.environ.get("ALERT_COMMAND", "").strip()
        self.kind = os.environ.get("KIND", "generator").strip() or "generator"
        progress = os.environ.get("PROGRESS_FILE", "").strip() or PROGRESS_FILES.get(self.kind, "history.jsonl")
        self.progress = Path(progress) if Path(progress).is_absolute() else self.run_dir / progress
        self.host_checks = os.environ.get("HOST_CHECKS", "0") == "1"
        self.usage_every = env_float("USAGE_EVERY_MIN", 360) * 60
        self.rss_every = env_float("RSS_EVERY_MIN", 60) * 60
        self.xid_every = env_float("XID_CHECK_MIN", 5) * 60


# ----------------------------------------------------------------------------- probes


def run(cmd, timeout=20):
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return out.returncode, out.stdout.strip(), out.stderr.strip()
    except (OSError, subprocess.TimeoutExpired) as e:
        return 127, "", str(e)


def unit_state(unit):
    props = (
        "ActiveState SubState Result NRestarts ExecMainStatus ExecMainCode MainPID InvocationID "
        "ActiveEnterTimestampMonotonic ControlGroup LoadState"
    ).split()
    code, out, err = run(["systemctl", "--user", "show", unit] + [f"-p{p}" for p in props])
    if code != 0:
        return {"error": err or out}
    state = dict(line.split("=", 1) for line in out.splitlines() if "=" in line)
    for key in ("NRestarts", "ExecMainStatus", "MainPID", "ActiveEnterTimestampMonotonic"):
        state[key] = int(state.get(key) or 0)
    return state


def last_json_line(path: Path):
    try:
        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 65536))
            lines = f.read().splitlines()
        for line in reversed(lines):
            try:
                return json.loads(line)
            except ValueError:
                continue
    except OSError:
        pass
    return None


def read_new_lines(path: Path, offset: int):
    """JSON records appended to ``path`` since byte ``offset``; (records, new offset)."""
    try:
        size = path.stat().st_size
    except OSError:
        return [], 0
    if size < offset:  # replaced or truncated
        offset = 0
    with path.open("rb") as f:
        f.seek(offset)
        data = f.read()
    end = data.rfind(b"\n") + 1
    records = []
    for line in data[:end].splitlines():
        try:
            records.append(json.loads(line))
        except ValueError:
            pass
    return records, offset + end


def progress_label(record, path=None):
    """What the last progress record says: an update (generator), a cycle (value trainer), or the file's age."""
    if record and "update" in record:
        return f"update {record['update']} at {record.get('time')}"
    if record and "cycle" in record:
        return f"cycle {record['cycle']} at {record.get('time')}"
    if path is not None:
        try:
            return f"{path.name} touched {datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec='seconds')}"
        except OSError:
            pass
    return "nothing yet"


def gpu_query():
    fields = ("index", "temperature.gpu", "temperature.gpu.tlimit") + THROTTLE_FIELDS
    code, out, err = run(["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"])
    if code != 0:
        return None, (err or out or f"exit {code}").splitlines()[0]
    gpus = []
    for line in out.splitlines():
        values = [v.strip() for v in line.split(",")]
        gpus.append(dict(zip(fields, values)))
    return gpus, None


def nvidia_mismatch():
    """(loaded kernel module version, installed libnvidia-compute version) when they differ."""
    try:
        loaded = Path("/proc/driver/nvidia/version").read_text().split("Kernel Module")[1].split()[2]
    except (OSError, IndexError):
        return None
    code, out, _ = run(["dpkg-query", "-W", "-f", "${Version} ${db:Status-Abbrev}\\n", "libnvidia-compute-*"])
    installed = [line.split()[0] for line in out.splitlines() if line.split()[1:2] == ["ii"]] if code == 0 else []
    for version in installed:
        upstream = version.split(":")[-1].split("-")[0]
        if upstream != loaded:
            return loaded, upstream
    return None


def mem_available_gb():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 2**20
    return None


def cgroup_memory(control_group):
    if not control_group:
        return None
    base = Path("/sys/fs/cgroup") / control_group.lstrip("/")
    try:
        current = int((base / "memory.current").read_text())
        high = (base / "memory.high").read_text().strip()
        events = dict(line.split() for line in (base / "memory.events").read_text().splitlines())
    except (OSError, ValueError):
        return None
    return {
        "current": current,
        "high": None if high == "max" else int(high),
        "events": {k: int(v) for k, v in events.items()},
    }


def notify(title, body, critical=False):
    """A desktop notification if a notification server is running; silently nothing otherwise."""
    if shutil.which("notify-send"):
        run(["notify-send", "-u", "critical" if critical else "normal", "-a", "rl18xx", title, body], timeout=5)
        return
    if shutil.which("gdbus"):
        hints = "{'urgency': <byte 2>}" if critical else "{}"
        run(["gdbus", "call", "--session", "--dest", "org.freedesktop.Notifications",
             "--object-path", "/org/freedesktop/Notifications",
             "--method", "org.freedesktop.Notifications.Notify",
             "rl18xx", "0", "", title, body, "[]", hints, "0"], timeout=5)


def send_alert(state_dir, instance, level, key, message, alert_command="", quiet=False):
    """One alert: a line on stdout (the journal) and in <state_dir>/alerts.log; WARN and CRIT also a desktop
    notification and ALERT_COMMAND (sh, with ALERT_LEVEL / ALERT_KEY / ALERT_MESSAGE / ALERT_INSTANCE set). The
    daily jobs (status_report.py, integrity_check.py, retention.py) alert through this too."""
    line = f"{datetime.now().astimezone().isoformat(timespec='seconds')} {level} [{instance}] {key}: {message}"
    print(line, flush=True)
    if quiet:
        return line
    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    with (state_dir / "alerts.log").open("a") as f:
        f.write(line + "\n")
    if level == INFO:
        return line
    notify(f"rl18xx {instance}: {level} {key}", message, critical=level == CRIT)
    if alert_command:
        env = {**os.environ, "ALERT_LEVEL": level, "ALERT_KEY": key, "ALERT_MESSAGE": message,
               "ALERT_INSTANCE": instance}
        try:
            subprocess.run(["sh", "-c", alert_command], env=env, timeout=60,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except (OSError, subprocess.TimeoutExpired):
            pass
    return line


# ----------------------------------------------------------------------------- the watchdog


class Watchdog:
    def __init__(self, settings: Settings, quiet=False):
        self.s = settings
        self.quiet = quiet
        self.state_file = settings.state_dir / "watchdog_state.json"
        try:
            self.state = json.loads(self.state_file.read_text())
        except (OSError, ValueError):
            self.state = {}
        self.state.setdefault("active", {})  # condition key -> {"since", "last_sent", "message"}
        self.state.setdefault("offsets", {})
        self.state.setdefault("restart_times", [])  # unix times of systemd's automatic restarts
        self.state.setdefault("throttle_seen", 0)
        self.started = time.time()

    # -- output

    def emit(self, level, key, message):
        send_alert(self.s.state_dir, self.s.instance, level, key, message, self.s.alert_command, self.quiet)

    notify = staticmethod(notify)

    # -- one round

    def check(self):
        """Every condition true now (key -> (level, message)), and one-shot events [(level, key, message)]."""
        s, now = self.s, time.time()
        conditions, events = {}, []
        unit = unit_state(s.unit)
        self.unit = unit

        # Launcher records: starts, exits (crash / non-finite / refused / stop).
        for name in ("starts.jsonl", "exits.jsonl"):
            first = name not in self.state["offsets"]  # no saved state: skip what happened before we started
            records, offset = read_new_lines(s.state_dir / name, self.state["offsets"].get(name, 0))
            self.state["offsets"][name] = offset
            for r in records:
                try:
                    when = datetime.fromisoformat(r["time"]).timestamp()
                except (KeyError, TypeError, ValueError):
                    when = now
                if first and when < self.started - 300:
                    continue
                events.append(self.describe_record(name, r))
        events = [e for e in events if e]

        # systemd's view: automatic restarts, failure, start limit.
        if "error" in unit:
            conditions["systemd"] = (WARN, f"systemctl --user show failed: {unit['error']}")
        else:
            previous = self.state.get("nrestarts")  # (a manual start resets systemd's counter)
            if previous is not None and unit["NRestarts"] > previous:
                events.append((WARN, "restart", f"systemd restarted the run automatically (restart "
                               f"#{unit['NRestarts']} since the last manual start)"))
                # The crash-loop count: systemd's own restarts (an OOM stop can cut the launcher's record).
                self.state["restart_times"] += [now] * (unit["NRestarts"] - previous)
            self.state["nrestarts"] = unit["NRestarts"]
            active = unit.get("ActiveState")
            if active == "failed":
                result, status, code = unit.get("Result"), unit.get("ExecMainStatus"), unit.get("ExecMainCode")
                if code == "1" and status in EXIT_MEANINGS:  # (CLD_EXITED) a status systemd doesn't restart
                    why = EXIT_MEANINGS[status]
                else:
                    # Under Restart=on-failure any other failure is restarted, so a unit left failed is
                    # one systemd gave up on: its start limit. Result keeps the last failure's kind.
                    last = {"oom-kill": "an OOM kill (MemoryMax)", "exit-code": f"exit status {status}",
                            "signal": f"signal {status}", "start-limit-hit": "a refused start"}.get(result, result)
                    why = f"crash loop: systemd's start limit stopped the restarts (last failure: {last})"
                conditions["failed"] = (CRIT, f"the run is down (result {result}): {why}")
            elif active == "inactive" and unit.get("LoadState") == "loaded":
                conditions["inactive"] = (INFO, "the run is not running (stopped, or never started)")

        recent = [t for t in self.state["restart_times"] if now - t < s.crash_loop_window]
        self.state["restart_times"] = recent
        if len(recent) >= s.crash_loop_restarts:
            conditions["crash_loop"] = (CRIT, f"{len(recent)} automatic restarts in the last "
                                              f"{s.crash_loop_window / 60:.0f} min")

        # Progress: a new progress-file line (history.jsonl) or rewrite / touch (status.json, heartbeat)
        # every STALL_MINUTES while the run is up.
        history = s.progress
        self.last = last_json_line(history)
        if unit.get("ActiveState") == "active":
            started = time.time() - (time.monotonic() - unit["ActiveEnterTimestampMonotonic"] / 1e6)
            try:
                last_line = history.stat().st_mtime
            except OSError:
                last_line = 0.0
            ref = max(last_line, started)
            if now - ref > s.stall_minutes * 60:
                conditions["stall"] = (WARN, f"no progress in {history.name} for {(now - ref) / 60:.0f} min "
                                             f"(last: {progress_label(self.last)}; threshold "
                                             f"{s.stall_minutes:.0f} min)")
            memory = cgroup_memory(unit.get("ControlGroup"))
            if memory:
                key = unit.get("InvocationID")
                kills = memory["events"].get("oom_kill", 0)
                if self.state.get("oom", {}).get("invocation") == key and kills > self.state["oom"]["kills"]:
                    events.append((CRIT, "oom_kill", f"{kills} OOM kill(s) inside the run's cgroup"))
                self.state["oom"] = {"invocation": key, "kills": kills}
                if memory["high"] and memory["current"] > 0.95 * memory["high"]:
                    conditions["memory_high"] = (WARN, f"the run uses {memory['current'] / 2**30:.1f} GiB, at its "
                                                       f"MemoryHigh {memory['high'] / 2**30:.0f} GiB (throttled)")

        # Disk, memory, GPU, host.
        seen_devices = set()
        for path in (s.run_dir if s.run_dir.exists() else s.workdir, s.workdir):
            try:
                device = path.stat().st_dev
            except OSError:
                continue
            if device in seen_devices:
                continue
            seen_devices.add(device)
            free = shutil.disk_usage(path).free / 1e9
            if free < s.disk_min_gb:
                conditions[f"disk:{path}"] = (CRIT if free < s.disk_min_gb / 4 else WARN,
                                             f"{free:.0f} GB free on the filesystem of {path} "
                                             f"(threshold {s.disk_min_gb:.0f} GB)")
        available = mem_available_gb()
        if available is not None:
            low = self.state.get("mem_min_gb")
            self.state["mem_min_gb"] = available if low is None else min(low, available)
        if available is not None and available < s.mem_available_min_gb:
            conditions["mem_available"] = (WARN, f"MemAvailable {available:.1f} GiB "
                                                 f"(threshold {s.mem_available_min_gb:.0f} GiB)")
        if s.gpu_check and shutil.which("nvidia-smi"):
            gpus, error = gpu_query()
            if error:
                conditions["gpu"] = (CRIT, f"nvidia-smi failed: {error}")
            throttled = False
            for gpu in gpus or []:
                index = gpu["index"]
                try:
                    temp = float(gpu["temperature.gpu"])
                    if temp >= s.gpu_temp_max:
                        conditions[f"gpu{index}_temp"] = (WARN, f"GPU {index} at {temp:.0f} C "
                                                                f"(threshold {s.gpu_temp_max:.0f} C)")
                except ValueError:
                    pass
                try:
                    headroom = float(gpu["temperature.gpu.tlimit"])
                    if headroom <= s.gpu_tlimit_min:
                        conditions[f"gpu{index}_tlimit"] = (WARN, f"GPU {index} is {headroom:.0f} C from its "
                                                                  f"slowdown temperature")
                except ValueError:
                    pass
                active_reasons = [f.split(".")[1] for f in THROTTLE_FIELDS if gpu.get(f) == "Active"]
                if active_reasons:
                    throttled = True
                    if self.state["throttle_seen"] >= 1:  # two rounds in a row: not a blip
                        conditions[f"gpu{index}_throttle"] = (WARN, f"GPU {index} throttling: "
                                                                    f"{', '.join(active_reasons)}")
            self.state["throttle_seen"] = self.state["throttle_seen"] + 1 if throttled else 0
            if gpus:
                self.count_throttle(throttled)
        mismatch = nvidia_mismatch()
        if mismatch:
            conditions["nvidia_mismatch"] = (CRIT, f"NVIDIA libraries {mismatch[1]} are installed but kernel module "
                                                   f"{mismatch[0]} is loaded: a restarted run can't use the GPU "
                                                   f"until a reboot (docs/longrun_runbook.md, 'Host stability')")
        if Path("/var/run/reboot-required").exists():
            conditions["reboot_required"] = (INFO, "the host wants a reboot (/var/run/reboot-required)")
        if s.host_checks:
            events += self.check_xid(now)
            if self.state.get("xid_error"):
                conditions["xid_unreadable"] = (INFO, f"can't scan the kernel log for Xid errors: "
                                                      f"{self.state['xid_error']}")
        return conditions, events

    def count_throttle(self, throttled):
        """GPU throttling per clock hour (the daily report's "throttling > 10% of the hour", M5)."""
        hour = datetime.now().strftime("%Y-%m-%dT%H")
        bucket = self.state.get("throttle_hour")
        if bucket and bucket["hour"] != hour:
            done = self.state.setdefault("throttle_hours", [])
            done.append({"hour": bucket["hour"], "fraction": bucket["throttled"] / max(bucket["rounds"], 1)})
            del done[:-48]
            bucket = None
        bucket = bucket or {"hour": hour, "rounds": 0, "throttled": 0}
        bucket["rounds"] += 1
        bucket["throttled"] += int(bool(throttled))
        self.state["throttle_hour"] = bucket

    def check_xid(self, now):
        """NVIDIA Xid errors logged since the last scan (every XID_CHECK_MIN): one CRIT event each scan."""
        last = self.state.get("xid_checked")
        if last is not None and now - last < self.s.xid_every:
            return []
        since = last if last is not None else now - 600
        lines, error = maintenance.xid_lines(since, now)
        self.state["xid_checked"] = now
        self.state["xid_error"] = error
        if not lines:
            return []
        self.state["xid_total"] = self.state.get("xid_total", 0) + len(lines)
        return [(CRIT, "xid", f"{len(lines)} NVIDIA Xid error(s) in the kernel log: {lines[-1][:200]} -- stop the "
                              f"run and inspect the hardware / driver (docs/long_run_recipe.md M5)")]

    def samples(self, now):
        """The maintenance histories (module docstring): RSS per process of the unit, and the host's usage."""
        unit = getattr(self, "unit", {}) or {}
        if unit.get("ActiveState") == "active" and now - self.state.get("rss_sampled", 0) >= self.s.rss_every:
            sample = maintenance.rss_sample(unit.get("ControlGroup"))
            if sample:
                maintenance.append_jsonl(self.s.state_dir / "rss.jsonl", sample)
            self.state["rss_sampled"] = now
        if self.s.host_checks and now - self.state.get("usage_sampled", 0) >= self.s.usage_every:
            sampled = self.state.get("throttle_sampled", "")
            hours = [h for h in self.state.get("throttle_hours", []) if h["hour"] > sampled]
            extra = {"mem_available_min_gb": self.state.get("mem_min_gb"), "throttle_hours": hours,
                     "instance": self.s.instance}
            ms = maintenance.Settings()
            maintenance.append_jsonl(ms.usage_log, maintenance.disk_sample(ms, extra))
            self.state["usage_sampled"] = now
            self.state["mem_min_gb"] = None
            if hours:
                self.state["throttle_sampled"] = hours[-1]["hour"]

    def describe_record(self, name, r):
        detail = f": {r['detail']}" if r.get("detail") else ""
        if name == "starts.jsonl":
            how = {"resume": f"resumed from snapshot {r.get('snapshot_update')}",
                   "run": f"started ({r.get('kind')}: it resumes its own state)"}.get(r.get("mode"), "started fresh")
            return INFO, "start", f"{how} (pid {r.get('pid')}, deployment {str(r.get('commit', ''))[:10]})"
        event, rc = r.get("event"), r.get("rc")
        if event == "stop":
            return INFO, "stop", "stopped through systemd"
        if event == "refused":
            return CRIT, "refused", f"start refused (status {rc}): {EXIT_MEANINGS.get(rc, r.get('reason'))}{detail}"
        if event == "exit" and r.get("reason") == "nonfinite":
            return CRIT, "nonfinite", f"non-finite-loss stop, not restarted{detail}"
        if event == "exit" and rc == 0:
            return INFO, "exit", "main.py exited cleanly (max updates or a STOP file)"
        if event == "exit":
            return WARN, "crash", f"main.py exited with status {rc}{detail}"
        return None

    def round(self):
        conditions, events = self.check()
        now = time.time()
        try:
            self.samples(now)
        except Exception as e:  # noqa: BLE001 -- a sample must never stop the checks
            events.append((WARN, "watchdog", f"maintenance sample failed: {e!r}"))
        for level, key, message in events:
            self.emit(level, key, message)
        active = self.state["active"]
        for key, (level, message) in conditions.items():
            entry = active.get(key)
            if entry is None:
                self.emit(level, key, message)
                active[key] = {"since": now, "last_sent": now, "level": level}
            elif level != INFO and now - entry["last_sent"] >= self.s.repeat:
                self.emit(level, key, f"still: {message} (since "
                                      f"{datetime.fromtimestamp(entry['since']).isoformat(timespec='minutes')})")
                entry["last_sent"] = now
        for key in [k for k in active if k not in conditions]:
            if active[key].get("level") != INFO:
                self.emit(INFO, key, "cleared")
            del active[key]
        self.s.state_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.state_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=1))
        tmp.replace(self.state_file)
        (self.s.state_dir / "watchdog.heartbeat").write_text(datetime.now().astimezone().isoformat() + "\n")

    def status(self):
        conditions, _ = self.check()
        u, s = self.unit, self.s
        print(f"instance   {s.instance} ({s.unit})")
        if "error" not in u:
            print(f"unit       {u.get('ActiveState')}/{u.get('SubState')} result={u.get('Result')} "
                  f"main pid {u.get('MainPID')} restarts {u.get('NRestarts')} status {u.get('ExecMainStatus')}")
        print(f"run dir    {s.run_dir}")
        try:
            snap = json.loads((s.run_dir / "state.json").read_text())["snapshot_update"]
        except (OSError, ValueError, KeyError):
            snap = None
        print(f"kind       {s.kind} (progress: {s.progress})")
        if self.last or s.progress.exists():
            rate = f", {self.last['games_per_hour']:.0f} games/h" if (self.last or {}).get("games_per_hour") else ""
            print(f"progress   {progress_label(self.last, s.progress)}, snapshot {snap}{rate}")
        else:
            print(f"progress   nothing yet, snapshot {snap}")
        print(f"log        {s.log}")
        print(f"alerts     {s.state_dir / 'alerts.log'}")
        for key, (level, message) in sorted(conditions.items()):
            print(f"  {level} {key}: {message}")
        if not conditions:
            print("  no active alerts")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", nargs="?", choices=("watch", "status"), default="watch")
    parser.add_argument("--instance", default=os.environ.get("LONGRUN_INSTANCE", "manual"))
    parser.add_argument("--once", action="store_true", help="One round of checks, then exit")
    args = parser.parse_args()
    load_env_file(args.instance)
    settings = Settings(args.instance)
    if args.mode == "status":
        Watchdog(settings, quiet=True).status()
        return
    dog = Watchdog(settings)
    if not args.once:
        dog.emit(INFO, "watchdog", f"watching {settings.unit} every {settings.interval:.0f} s")
    while True:
        try:
            dog.round()
        except Exception as e:  # noqa: BLE001 -- the watchdog must outlive its own bugs
            dog.emit(WARN, "watchdog", f"check failed: {e!r}")
        if args.once:
            return
        time.sleep(settings.interval)


if __name__ == "__main__":
    sys.exit(main())
