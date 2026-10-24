#!/usr/bin/env python3
"""The long run's retention job (docs/long_run_recipe.md, "Retention policy"). Standard library only.

    retention.py --config scripts/launch/2026-11/retention.json [--workdir DIR] [--log longrun/retention.jsonl]
                 [--dry-run | --execute] [--quiet]

Dry run unless the config's ``dry_run_until`` date has passed (no date: always a dry run, unless ``--execute``): it
lists what it would delete and the bytes. Then for real. The rules (paths relative to the workdir, ``${VAR}``
expanded from the environment, e.g. ``${DEPLOY}``):

- ``snapshots``: generator / policy snapshots ``<dir>/<file pattern with {n}>`` thinned by age: kept while some tier
  ``{"hours": h, "every": k}`` has the file younger than ``h`` hours (no ``hours``: forever) and ``n`` a multiple of
  ``k`` -- the recipe's every 10th for 48 h, every 100th for 2 weeks, every 1000th forever. The newest file of each
  pattern, the generator's resume snapshot and KL anchor (its ``state.json``) and every referenced checkpoint
  (below) are always kept;
- ``value_candidates``: ``<dir>/<cycle>.pth`` older than ``days`` whose cycle the value trainer's ``cycles.jsonl``
  records as not promoted (a cycle it doesn't record is kept);
- ``unreferenced``: files matching ``glob`` older than ``min_age_days`` that nothing references (a league snapshot
  that left the league);
- ``optimizer_state``: files matching ``glob``: the newest, plus the newest of each ISO week, are kept;
- ``logs``: files matching ``glob`` above ``max_bytes`` are rotated (copied to ``.1``, older rotations shifted up to
  ``rotations``, the file truncated in place -- its writers append, so they carry on), rotations beyond that deleted.

Referenced checkpoints (``protect``): every path in the specs of the ladder panels (members, run checkpoints of
members by update, the baseline), of the ladders' ledgers and state (candidates), of league files
(``<generator run>/league.json``) and population files, and the generator runs' snapshot / anchor / starting
checkpoints.

**Hard guards**, whatever the config says: nothing is deleted or rotated that is a saved game (``*.sld``,
``*.journal``, anything under a ``protect.save_data`` directory), a promoted value (a ``promoted`` directory or a
path a ``promoted.json`` lists), an eval collection (anything under ``logs/eval`` or a directory holding
``games.jsonl``), a ladder ledger (``ledger.jsonl``) or a run's metadata (``history.jsonl``, ``state.json``,
``config.json``, ``cycles.jsonl``, ``promoted.json``, ``league.json``); nor anything outside the workdir or reached
through a symlinked directory. A rule that would touch one is refused: then nothing at all is deleted (exit 3, a WARN
``retention`` alert) -- fix the config.

Appends one JSON summary line per run to ``--log`` (the daily status report shows it).
"""

import argparse
import glob
import json
import os
import re
import shutil
import sys
import time
from datetime import date, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN_METADATA = {"history.jsonl", "state.json", "config.json", "cycles.jsonl", "promoted.json", "league.json"}
SAVED_SUFFIXES = (".sld", ".journal", ".sld.tmp")


# ----------------------------------------------------------------------------- references


def spec_paths(spec):
    """The checkpoint files of a player / server spec (``<policy>[+<value>][#head]@...:...``)."""
    checkpoint = str(spec).partition(":")[0].partition("@")[0].partition("#")[0]
    return [p for p in checkpoint.split("+") if p]


def _json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def _jsonl(path):
    out = []
    try:
        for line in Path(path).read_text().splitlines():
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    except OSError:
        pass
    return out


class Retention:
    def __init__(self, config, workdir, now=None):
        self.cfg = config
        self.workdir = Path(workdir).resolve()
        self.now = time.time() if now is None else now
        protect = config.get("protect") or {}
        self.save_data = [self.abs(p) for p in protect.get("save_data", [])]
        self.references, self.ladder_roots, self.promoted_listed = set(), [], set()
        self.notes = []
        self._collect(protect)

    def path(self, p):
        p = Path(os.path.expandvars(str(p)))
        return p if p.is_absolute() else self.workdir / p

    def abs(self, p):
        return Path(os.path.abspath(self.path(p)))

    def _ref(self, p):
        self.references.add(str(Path(os.path.realpath(self.path(p)))))

    def _collect(self, protect):
        for panel_file in protect.get("panels", []):
            panel = _json(self.path(panel_file))
            if panel is None:
                self.notes.append(f"panel {panel_file} unreadable: its checkpoints can't be protected")
                continue
            run, pattern = panel.get("run", ""), panel.get("checkpoint", "learner/{update}.pth")
            run_ckpt = lambda u: f"{run.rstrip('/')}/{pattern.format(update=u)}"  # noqa: E731
            for m in panel.get("panel", []):
                for p in spec_paths(m["spec"]) if "spec" in m else [run_ckpt(m["update"])]:
                    self._ref(p)
            if panel.get("baseline_update") is not None:
                self._ref(run_ckpt(panel["baseline_update"]))
            root = self.path(panel.get("out_root") or Path("logs") / "eval" / panel.get("name", ""))
            self.ladder_roots.append(Path(os.path.abspath(root)))
            ledger = self.path(panel["ledger"]) if panel.get("ledger") else root / "ledger.jsonl"
            for r in _jsonl(ledger):
                for side in ("a_spec", "b_spec"):
                    for p in spec_paths(r.get(side, "")):
                        self._ref(p)
            state = _json(root / "state.json") or {}
            current = [state["current"]] if state.get("current") is not None else []
            for u in list(state.get("candidates", [])) + current:
                self._ref(run_ckpt(u))
        for league_file in protect.get("league_files", []) + protect.get("populations", []):
            data = _json(self.path(league_file))
            if data is None:
                continue
            members = data.get("members", data) if isinstance(data, dict) else data
            items = members.values() if isinstance(members, dict) else members
            for m in items:
                spec = m.get("spec") or m.get("path") if isinstance(m, dict) else m
                for p in spec_paths(spec or ""):
                    self._ref(p)
        for run in protect.get("generator_runs", []):
            run_dir = self.path(run)
            state = _json(run_dir / "state.json")
            if state is None:
                continue
            snap, anchor = state.get("snapshot_update"), int(state.get("anchor_update") or 0)
            config = state.get("config") or {}
            if snap is not None:
                self._ref(run_dir / "learner" / f"{snap}.pth")
                self._ref(run_dir / "critic" / f"{snap}.pth")
            if anchor:
                self._ref(run_dir / "learner" / f"{anchor}.pth")
            for key in ("policy_checkpoint", "value_checkpoint"):
                if config.get(key):
                    self._ref(config[key])
        for promoted in protect.get("promoted_json", []):
            data = _json(self.path(promoted)) or {}
            base = self.path(promoted).parent
            for v in data.get("versions", []):
                if v.get("path"):
                    p = Path(v["path"])
                    self.promoted_listed.add(str(Path(os.path.realpath(p if p.is_absolute() else base / p))))

    # -- guards

    def guard(self, path):
        """Why ``path`` must never be deleted or rotated (a hard guard), or None."""
        path = Path(os.path.abspath(path))
        try:
            rel = path.relative_to(self.workdir)
        except ValueError:
            return "outside the workdir"
        probe = self.workdir
        for part in rel.parts[:-1]:
            probe = probe / part
            if probe.is_symlink():
                return f"reached through the symlinked directory {probe}"
        name = path.name
        if name.endswith(SAVED_SUFFIXES) or any(path == d or d in path.parents for d in self.save_data):
            return "saved games"
        if self._shard_dir(path.parent):
            return "saved games (a directory of shards)"
        if "promoted" in rel.parts or str(Path(os.path.realpath(path))) in self.promoted_listed:
            return "a promoted value"
        if name == "ledger.jsonl" or any(path == r or r in path.parents for r in self.ladder_roots):
            return "the ladder (ledger, matches)"
        parts = rel.parts
        if any(parts[i : i + 2] == ("logs", "eval") for i in range(len(parts))):
            return "an eval collection"
        for parent in path.parents:
            if parent == self.workdir or self.workdir not in parent.parents:
                break
            if (parent / "games.jsonl").exists():
                return "an eval collection"
        if name in RUN_METADATA:
            return "a run's metadata"
        return None

    def _shard_dir(self, directory):
        cache = self.__dict__.setdefault("_shard_dirs", {})
        if directory not in cache:
            cache[directory] = directory.is_dir() and any(p.name.endswith(SAVED_SUFFIXES) for p in directory.iterdir())
        return cache[directory]

    def referenced(self, path):
        return str(Path(os.path.realpath(path))) in self.references

    # -- rules

    def files(self, pattern):
        return sorted(Path(p) for p in glob.glob(str(self.path(pattern)), recursive=True) if os.path.isfile(p))

    def age_hours(self, path):
        return (self.now - os.stat(path).st_mtime) / 3600

    def snapshot_rule(self, rule):
        """(delete, keep-count) of one ``snapshots`` rule."""
        base = self.path(rule["dir"])
        delete, kept = [], 0
        state = _json(base / "state.json") or {}
        for pattern in rule.get("files", ["learner/{n}.pth", "critic/{n}.pth"]):
            # The resume snapshot (every pattern) and the KL anchor (the learner's) stay.
            pinned = {state.get("snapshot_update")} | ({state.get("anchor_update")} if "learner" in pattern else set())
            regex = re.compile(re.escape(pattern).replace(re.escape("{n}"), r"(\d+)") + "$")
            found = []
            for p in self.files(base / pattern.replace("{n}", "*")):
                m = regex.search(p.relative_to(base).as_posix())
                if m:
                    found.append((int(m.group(1)), p))
            newest = max((n for n, _ in found), default=None)
            for n, p in found:
                age = self.age_hours(p)
                keep = n == newest or n in pinned or self.referenced(p) or any(
                    n % int(t["every"]) == 0 and ("hours" not in t or age <= float(t["hours"]))
                    for t in rule.get("keep", [])
                )
                if keep:
                    kept += 1
                else:
                    delete.append(p)
        return delete, kept

    def value_candidate_rule(self, rule):
        base = self.path(rule["dir"])
        cycles = {}
        for c in _jsonl(self.path(rule["cycles"])) if rule.get("cycles") else []:
            if c.get("cycle") is not None:
                cycles[int(c["cycle"])] = bool(c.get("promoted"))
        delete, kept = [], 0
        for p in self.files(base / "*.pth"):
            try:
                cycle = int(p.stem)
            except ValueError:
                kept += 1
                continue
            if cycles.get(cycle) is False and self.age_hours(p) > 24 * float(rule.get("days", 7)) \
                    and not self.referenced(p):
                delete.append(p)
            else:
                kept += 1
        return delete, kept

    def unreferenced_rule(self, rule):
        delete, kept = [], 0
        for p in self.files(rule["glob"]):
            if not self.referenced(p) and self.age_hours(p) > 24 * float(rule.get("min_age_days", 7)):
                delete.append(p)
            else:
                kept += 1
        return delete, kept

    def optimizer_rule(self, rule):
        found = [(os.stat(p).st_mtime, p) for p in self.files(rule["glob"])]
        if not found:
            return [], 0
        keep = {max(found)[1]}
        weeks = {}
        for mtime, p in found:
            week = date.fromtimestamp(mtime).isocalendar()[:2]
            if week not in weeks or mtime > weeks[week][0]:
                weeks[week] = (mtime, p)
        keep |= {p for _, p in weeks.values()}
        delete = [p for _, p in found if p not in keep and not self.referenced(p)]
        return delete, len(found) - len(delete)

    def log_rule(self, rule):
        """(rotations to do [(path, size)], old rotations to delete)."""
        rotations, extra = [], []
        n = int(rule.get("rotations", 4))
        for p in self.files(rule["glob"]):
            if re.search(r"\.\d+$", p.name):
                continue  # a rotation itself
            if p.stat().st_size > float(rule.get("max_bytes", 1e9)):
                rotations.append((p, p.stat().st_size))
            for q in glob.glob(glob.escape(str(p)) + ".*"):
                m = re.search(r"\.(\d+)$", q)
                if m and int(m.group(1)) > n:
                    extra.append(Path(q))
        return rotations, extra

    def plan(self):
        """Everything the config asks for: {"delete": [(rule, path)], "rotate": [(path, size, rotations)],
        "refused": [(rule, path, why)], "kept": {rule: count}}."""
        out = {"delete": [], "rotate": [], "refused": [], "kept": {}}

        def take(label, paths, kept):
            out["kept"][label] = out["kept"].get(label, 0) + kept
            for p in paths:
                why = self.guard(p)
                if why:
                    out["refused"].append((label, str(p), why))
                else:
                    out["delete"].append((label, p))

        for i, rule in enumerate(self.cfg.get("snapshots", [])):
            take(f"snapshots[{i}] {rule['dir']}", *self.snapshot_rule(rule))
        for i, rule in enumerate(self.cfg.get("value_candidates", [])):
            take(f"value_candidates[{i}] {rule['dir']}", *self.value_candidate_rule(rule))
        for i, rule in enumerate(self.cfg.get("unreferenced", [])):
            take(f"unreferenced[{i}] {rule['glob']}", *self.unreferenced_rule(rule))
        for i, rule in enumerate(self.cfg.get("optimizer_state", [])):
            take(f"optimizer_state[{i}] {rule['glob']}", *self.optimizer_rule(rule))
        for i, rule in enumerate(self.cfg.get("logs", [])):
            label = f"logs[{i}] {rule['glob']}"
            rotations, extra = self.log_rule(rule)
            take(label, extra, 0)
            for p, size in rotations:
                why = self.guard(p)
                if why:
                    out["refused"].append((label, str(p), why))
                else:
                    out["rotate"].append((p, size, int(rule.get("rotations", 4))))
        # Rules themselves must not point into guarded places (a config asking to thin saved games, say), whether
        # or not anything there is old enough to go yet.
        for key in ("snapshots", "value_candidates"):
            for i, rule in enumerate(self.cfg.get(key, [])):
                directory = self.path(rule["dir"])
                why = self.guard(directory / "probe.pth")
                if not why and directory.is_dir() and any(
                    f.endswith(SAVED_SUFFIXES) for _, _, files in os.walk(directory) for f in files
                ):
                    why = "saved games (a directory of shards)"
                if why:
                    out["refused"].append((f"{key}[{i}] {rule['dir']}", str(directory), why))
        for key in ("unreferenced", "optimizer_state", "logs"):
            for i, rule in enumerate(self.cfg.get(key, [])):
                for p in self.files(rule["glob"]):
                    why = self.guard(p)
                    if why:
                        out["refused"].append((f"{key}[{i}] {rule['glob']}", str(p), why))
                        break
        return out


def rotate(path, rotations):
    """Copy-truncate rotation: .{k} -> .{k+1} (the oldest dropped), the file -> .1, the file truncated in place."""
    path = Path(path)
    oldest = path.with_name(f"{path.name}.{rotations}")
    if oldest.exists():
        oldest.unlink()
    for k in range(rotations - 1, 0, -1):
        src = path.with_name(f"{path.name}.{k}")
        if src.exists():
            src.rename(path.with_name(f"{path.name}.{k + 1}"))
    shutil.copy2(path, path.with_name(f"{path.name}.1"))
    with open(path, "r+b") as f:
        f.truncate(0)


def is_dry_run(config, today=None, force_dry=False, execute=False):
    if force_dry:
        return True
    if execute:
        return False
    until = config.get("dry_run_until")
    if not until:
        return True
    return (today or date.today()) <= date.fromisoformat(str(until))


def run(config, workdir, dry_run, now=None, log=None, echo=print):
    r = Retention(config, workdir, now)
    plan = r.plan()
    summary = {
        "time": datetime.now().astimezone().isoformat(timespec="seconds"),
        "dry_run": dry_run,
        "refused": [{"rule": a, "path": b, "why": c} for a, b, c in plan["refused"]],
        "notes": r.notes,
        "kept": plan["kept"],
        "files": 0,
        "bytes": 0,
        "hard_linked": 0,
        "by_rule": {},
        "rotated": [],
        "errors": [],
    }
    if plan["refused"]:
        dry_run = summary["dry_run"] = True  # a refused rule: nothing at all is deleted
        for a, b, c in plan["refused"][:20]:
            echo(f"REFUSED {a}: {b} ({c})")
    for label, path in plan["delete"]:
        try:
            st = os.stat(path)
        except OSError:
            continue
        linked = st.st_nlink > 1
        summary["files"] += 1
        summary["hard_linked"] += int(linked)
        summary["bytes"] += 0 if linked else st.st_size
        entry = summary["by_rule"].setdefault(label, {"files": 0, "bytes": 0})
        entry["files"] += 1
        entry["bytes"] += 0 if linked else st.st_size
        note = ", hard-linked" if linked else ""
        echo(f"{'would delete' if dry_run else 'delete'} {path} ({st.st_size / 1e6:.1f} MB{note})")
        if not dry_run:
            try:
                os.unlink(path)
            except OSError as e:
                summary["errors"].append(f"{path}: {e!r}")
    for path, size, rotations in plan["rotate"]:
        summary["rotated"].append({"path": str(path), "bytes": size})
        echo(f"{'would rotate' if dry_run else 'rotate'} {path} ({size / 1e9:.2f} GB, keeping {rotations})")
        if not dry_run:
            try:
                rotate(path, rotations)
            except OSError as e:
                summary["errors"].append(f"{path}: {e!r}")
    if log:
        Path(log).parent.mkdir(parents=True, exist_ok=True)
        with open(log, "a") as f:
            f.write(json.dumps(summary, separators=(",", ":")) + "\n")
    return summary


def summary_line(s):
    verb = "would delete" if s["dry_run"] else "deleted"
    mode = "dry run" if s["dry_run"] else "for real"
    line = f"retention ({mode}): {verb} {s['files']} files, {s['bytes'] / 1e9:.2f} GB"
    if s["hard_linked"]:
        line += f" ({s['hard_linked']} hard-linked, freeing nothing alone)"
    if s["rotated"]:
        line += f"; {'would rotate' if s['dry_run'] else 'rotated'} {len(s['rotated'])} log(s)"
    if s["refused"]:
        line += f"; REFUSED {len(s['refused'])} (nothing deleted): {s['refused'][0]['path']} ({s['refused'][0]['why']})"
    if s["errors"]:
        line += f"; {len(s['errors'])} error(s)"
    return line


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, default=Path(os.environ.get("WORKDIR") or "."))
    parser.add_argument("--log", type=Path, default=None, help="append the JSON summary here")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--dry-run", action="store_true", help="only list (whatever dry_run_until says)")
    group.add_argument("--execute", action="store_true", help="delete even before dry_run_until")
    parser.add_argument("--quiet", action="store_true", help="only the summary line")
    parser.add_argument("--no-alert", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    dry = is_dry_run(config, force_dry=args.dry_run, execute=args.execute)
    log = args.log if args.log is None or args.log.is_absolute() else args.workdir / args.log
    summary = run(config, args.workdir, dry, log=log, echo=(lambda *_: None) if args.quiet else print)
    print(summary_line(summary))
    if summary["refused"] or summary["errors"]:
        if not args.no_alert:
            sys.path.insert(0, str(HERE))
            import watchdog

            state_dir = Path(os.environ.get("STATE_DIR") or "longrun/daily")
            watchdog.send_alert(state_dir if state_dir.is_absolute() else args.workdir / state_dir,
                                os.environ.get("LONGRUN_INSTANCE", "daily"), "WARN", "retention",
                                summary_line(summary), os.environ.get("ALERT_COMMAND", "").strip())
        return 3 if summary["refused"] else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
