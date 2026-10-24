"""Long-run evaluation ladder: a run's checkpoints rated against a fixed reference panel.

Why: azl's old health signal, uM vs u(M-300), won every step (0.52-0.57) while u1800 vs u0 scored 0.500 where the
chained steps implied ~0.71 -- four-player self-play drifts non-transitively (docs/search_learning_design.md "Rating
a long run against a fixed panel"). This ladder judges every checkpoint against the same fixed opponents instead.

``run`` (the scheduler): every ``every`` updates it takes the run's newest checkpoint whose update is a multiple of
``every`` (skipping ahead when it fell behind) and plays it against each panel member, one
``eval_head_to_head.py`` match at a time on ``workers`` workers (the GPU-share cap):

- 2v2 and 1v3 (``--a-seats 1``; the candidate alone against three panel copies, and the reverse);
- each pairing both ways: the candidate as side A, then the panel member;
- every match from the same starts: seed 0 and the v2 start file, start g // 6 in 2v2 and g // 4 in 1v3, so every
  match and both modes share starts.

Each finished match is appended to a JSONL ledger: both specs, the shared settings, the scores (a start-clustered
SE too), seat win rates, net-worth share, and every game's start, seats, win shares and net-worth shares. A crash
loses at most the match in progress (state.json resumes the candidate; a half-written match directory is set aside).
The panel file is re-read before every match, so ``workers``, ``games`` and the members can change mid-run.

``report``: ratings for every checkpoint and panel member, fit separately on 2v2 win shares, 1v3 win shares and
net-worth shares (both modes). The model is a four-seat Luce (Bradley-Terry) model: seat i wins with probability
proportional to exp(r[its player] + seat effect + side-A effect), so 2v2 between equals scores 0.5 and 1v3 0.25 on
one rating scale, and a share outcome (ties split, net-worth share) is fit as a fractional multinomial logit
(quasi-likelihood). Standard errors are sandwich SEs clustered by start position across all matches (a human start
is one cluster whatever start index drew it). Ratings print in Elo (400 / ln 10 per unit of log-strength, so a
+100 Elo gap is 0.64 in 2v2) against an anchor (default u0), by update, with a plot if asked.

``decide``: whether the latest candidate is significantly above the one ``--back`` candidates earlier, fit on the
fixed panel's matches only (no run-vs-run games; older run checkpoints played as sparse "self" members are left
out) and on the panel members both faced, with each candidate's per-member excess shares beside it.

``import``: add existing single-match eval directories (e.g. the old ladder's ``logs/eval/azl_u*_vs_*``) to a
ledger when their settings match the panel's; players are named by matching their specs to the panel file.

``check``: every checkpoint a panel file names (members' specs, run checkpoints of members given by update), and
whether it exists.

**Schedules** (optional panel keys; a panel without them plays every candidate against the whole panel, as above):

- ``"rotate": k`` -- a candidate plays only ``k`` panel members, in ``rotate_modes`` (default ``["2v2"]``) at
  ``rotate_games`` games per match (default ``games``), the members taken in turn from the panel's order (those with
  a rotate mode), so consecutive candidates cycle through the panel;
- ``"full_every_hours": h`` -- with ``rotate``, a candidate gets the whole panel (every member and mode at its
  ``games``, plus the ``self`` members) when no full schedule started in the last ``h`` hours;
- ``"baseline_update": u`` -- before any other candidate, update ``u`` (the run's start) plays the whole panel: the
  day-0 baseline (a panel member that is the same checkpoint is skipped, as for any candidate);
- ``"value_check": {"games": n, "modes": ["2v2"]}`` -- each full schedule after the first also plays the previous full
  candidate (a week older with ``full_every_hours`` 168; ``purpose`` "value_check" in the ledger; when that candidate
  is a panel member, its own matches are tagged instead).

A candidate's schedule is chosen when it is taken and kept in ``state.json`` (a resume plays the same plan); each
ledger record says its ``schedule`` ("full" or "rotation") and finishing ``time``. ``decide --full`` (and the
reports of a rotating panel) compares only candidates that played full schedules. With ``"every": 1`` over a run
whose checkpoints are numbered versions (a value run's ``promoted/{update}.pth``), the ladder takes the newest one
whenever it is free.

A ``STOP`` file in the ladder's output directory ends ``run`` before its next match (exit 0; the current candidate
resumes on the next start). ``heartbeat`` there is touched while the ladder makes progress (each poll, each finished
game of a match): the long-run watchdog's stall check reads it.

    uv run python scripts/eval_ladder.py run --panel scripts/launch/2026-10/eval_ladder_azl.json --results-log logs/az1.log
    uv run python scripts/eval_ladder.py run --panel <panel.json> --dry-run      # the next candidate's commands
    uv run python scripts/eval_ladder.py report --panel <panel.json> --plot logs/eval/ladder_azl/ratings.png
    uv run python scripts/eval_ladder.py decide --panel <panel.json> --back 2
    uv run python scripts/eval_ladder.py import --panel <panel.json> logs/eval/azl_u*_vs_*
    uv run python scripts/eval_ladder.py check --panel <panel.json>               # every checkpoint it names exists?

Panel file (JSON; spec paths are relative to the directory the ladder runs in):

    {
      "name": "ladder_azl",                          # output under logs/eval/<name>/ unless "out_root" is given
      "run": "model_checkpoints_search/azl_20261019",
      "checkpoint": "learner/{update}.pth",         # a run checkpoint, relative to "run"
      "spec": "{checkpoint}@200/0.05:mean,dedupe,par=8,ptemp=1,budget=total",  # how run checkpoints play
      "every": 600, "games": 120, "modes": ["2v2", "1v3"], "workers": 12,
      "panel": [
        {"name": "u0", "update": 0},                # a run checkpoint (played with "spec")
        {"name": "gen4critic", "spec": "<policy>+<value>@200/..."},  # anything eval_head_to_head.py takes
        {"name": "...", "spec": "...", "modes": ["1v3"], "games": 240}  # per-member overrides
      ],
      "self": {"every": 1800, "count": 1, "modes": ["2v2"]}  # optional: older run checkpoints, sparsely
    }

``games`` is per match (per mode and direction), rounded up to the arrangements (6 in 2v2, 4 in 1v3). Other keys
and their defaults are in ``DEFAULTS`` (``eval_args``, ``env``, ``command_prefix`` e.g. ``"nice -n 10"``, ``seed``,
``start_positions``, ``softpick``, ``anchor``, ``back``, ...).
"""

import argparse
import json
import math
import os
import re
import shlex
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
import eval_head_to_head as h2h  # noqa: E402  (numpy only at import; its heavy imports are lazy)

MODES = {"2v2": 2, "1v3": 1}  # mode -> seats side A holds
ELO_PER_UNIT = 400 / math.log(10)
# (label, modes, outcome) of the three ratings.
FITS = (("2v2", ("2v2",), "win"), ("1v3", ("1v3",), "win"), ("worth", ("2v2", "1v3"), "worth"))
DEFAULTS = {
    "checkpoint": "learner/{update}.pth",
    "every": 600,
    "games": 120,
    "modes": ["2v2", "1v3"],
    "workers": 12,
    "seed": 0,
    "start_positions": "human_games/start_positions_1830_4p_v2.jsonl",
    "random_start_fraction": 0.2,
    "softpick": 0,
    "max_length": 1000,
    "stall": 400,
    "eval_args": ["--games-per-task", "8", "--dedupe-maps", "--server-replicas", "2"],
    "env": {"OMP_NUM_THREADS": "2", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    "command_prefix": "",
    "python": None,  # default: this interpreter
    "poll_seconds": 300,
    "settle_seconds": 60,  # a checkpoint younger than this may still be being written
    "max_attempts": 2,
    "retry_seconds": 60,  # pause after a failed match (e.g. the GPU out of memory at startup)
    "min_games_fraction": 0.9,  # a match that lost more games to crashes is a failure
    "anchor": "u0",
    "back": 1,
    # Schedules (module docstring): off unless set.
    "rotate": 0,
    "rotate_modes": ["2v2"],
    "rotate_games": None,
    "full_every_hours": None,
    "baseline_update": None,
    "value_check": None,
}
HEARTBEAT_SECONDS = 30  # how often a running match is checked for finished games
# Settings every match of a ledger must share for their ratings to be comparable.
SHARED_SETTINGS = ("seed", "start_positions", "random_start_fraction", "softpick", "max_length", "stall")


# ---------------------------------------------------------------------------
# Panel file


def load_config(path) -> dict:
    cfg = {**DEFAULTS, **json.loads(Path(path).read_text())}
    for key in ("name", "run", "spec", "panel"):
        if key not in cfg:
            raise SystemExit(f"{path}: missing {key!r}")
    if "{checkpoint}" not in cfg["spec"]:
        raise SystemExit(f"{path}: spec needs a {{checkpoint}} placeholder")
    if "{update}" not in cfg["checkpoint"]:
        raise SystemExit(f"{path}: checkpoint needs an {{update}} placeholder")
    names = set()
    for member in cfg["panel"]:
        if ("spec" in member) == ("update" in member):
            raise SystemExit(f"{path}: panel member {member} needs exactly one of spec / update")
        if member["name"] in names:
            raise SystemExit(f"{path}: two panel members named {member['name']!r}")
        names.add(member["name"])
        for mode in member.get("modes", cfg["modes"]):
            if mode not in MODES:
                raise SystemExit(f"{path}: unknown mode {mode!r} (one of {', '.join(MODES)})")
    for mode in cfg["rotate_modes"] + list((cfg.get("value_check") or {}).get("modes", [])):
        if mode not in MODES:
            raise SystemExit(f"{path}: unknown mode {mode!r} (one of {', '.join(MODES)})")
    if cfg["rotate"] and not any(set(m.get("modes", cfg["modes"])) & set(cfg["rotate_modes"]) for m in cfg["panel"]):
        raise SystemExit(f"{path}: no panel member plays a rotate mode ({cfg['rotate_modes']})")
    return cfg


def scheduled(cfg) -> bool:
    """The panel uses schedules (rotation, a periodic full panel, a baseline first): state.json keeps them."""
    return bool(cfg.get("rotate")) or cfg.get("full_every_hours") is not None or cfg.get("baseline_update") is not None


def out_root(cfg) -> Path:
    return Path(cfg.get("out_root") or Path("logs") / "eval" / cfg["name"])


def ledger_path(cfg) -> Path:
    return Path(cfg.get("ledger") or out_root(cfg) / "ledger.jsonl")


def run_name(update) -> str:
    return f"u{update}"


def run_spec(cfg, update) -> str:
    checkpoint = f"{cfg['run'].rstrip('/')}/{cfg['checkpoint'].format(update=update)}"
    return cfg["spec"].format(checkpoint=checkpoint)


@dataclass(frozen=True)
class Member:
    name: str
    spec: str
    role: str  # "candidate", "panel" (a fixed reference) or "self" (an older checkpoint of the run)
    update: Optional[int] = None
    modes: tuple = ()
    games: int = 0


def panel_members(cfg) -> list:
    members = []
    for m in cfg["panel"]:
        update = int(m["update"]) if "update" in m else None
        spec = m["spec"] if "spec" in m else run_spec(cfg, update)
        members.append(
            Member(
                m["name"], spec, "panel", update, tuple(m.get("modes", cfg["modes"])), int(m.get("games", cfg["games"]))
            )
        )
    return members


def self_members(cfg, candidate: int, available) -> list:
    """The ``count`` newest run checkpoints at multiples of ``self.every`` at least that far behind ``candidate``."""
    opts = cfg.get("self")
    if not opts:
        return []
    every = int(opts["every"])
    fixed = {m.update for m in panel_members(cfg) if m.update is not None}
    older = [u for u in sorted(available) if u % every == 0 and 0 < u <= candidate - every and u not in fixed]
    return [
        Member(
            run_name(u),
            run_spec(cfg, u),
            "self",
            u,
            tuple(opts.get("modes", cfg["modes"])),
            int(opts.get("games", cfg["games"])),
        )
        for u in older[-int(opts.get("count", 1)) :]
    ]


# ---------------------------------------------------------------------------
# Scheduling


@dataclass(frozen=True)
class Match:
    key: str
    name: str
    mode: str
    candidate: Member
    opponent: Member
    candidate_side: str  # "A": the candidate is side A (listed first)
    games: int
    schedule: str = "full"  # "full" or "rotation" (the candidate's schedule)
    purpose: Optional[str] = None  # "value_check": the weekly check against the previous full candidate

    @property
    def a(self) -> Member:
        return self.candidate if self.candidate_side == "A" else self.opponent

    @property
    def b(self) -> Member:
        return self.opponent if self.candidate_side == "A" else self.candidate

    def fields(self) -> dict:
        """The ledger's description of this match (its results are added by ``read_match``)."""
        return {
            "key": self.key,
            "name": self.name,
            "mode": self.mode,
            "candidate": self.candidate.name,
            "candidate_update": self.candidate.update,
            "candidate_side": self.candidate_side,
            "opponent": self.opponent.name,
            "opponent_update": self.opponent.update,
            "opponent_role": self.opponent.role,
            "a": self.a.name,
            "b": self.b.name,
            "schedule": self.schedule,
            **({"purpose": self.purpose} if self.purpose else {}),
        }


def match_key(candidate: str, opponent: str, mode: str, a: str) -> str:
    return f"{candidate}|{opponent}|{mode}|A={a}"


def games_for(mode: str, games: int) -> int:
    """``games`` rounded up to whole sets of the mode's seat arrangements, as eval_head_to_head.py plays them."""
    n = len(h2h.arrangements(MODES[mode]))
    return math.ceil(games / n) * n


def plan_matches(cfg, candidate: int, available=(), schedule: Optional[dict] = None) -> list:
    """Every match of ``candidate`` in play order: each panel member (then self member) in turn, each of its modes,
    the candidate as side A and then as side B. ``schedule`` (``choose_schedule``): a rotation plays only its
    members in the rotate modes; a full schedule adds its value check."""
    cand = Member(run_name(candidate), run_spec(cfg, candidate), "candidate", candidate)
    kind = (schedule or {}).get("kind", "full")
    check, check_modes, tagged = (schedule or {}).get("value_check"), (), False
    if kind == "rotation":
        names = set(schedule["members"])
        games = int(cfg.get("rotate_games") or cfg["games"])
        opponents = [
            Member(m.name, m.spec, m.role, m.update, tuple(x for x in m.modes if x in cfg["rotate_modes"]), games)
            for m in panel_members(cfg)
            if m.name in names
        ]
    else:
        opponents = panel_members(cfg) + self_members(cfg, candidate, available)
        if check is not None:
            vc = cfg.get("value_check") or {}
            check_modes = tuple(vc.get("modes", ["2v2"]))
            tagged = any(m.update == check for m in opponents)
            if not tagged:
                opponents = [m for m in opponents if m.name != run_name(check)] + [
                    Member(
                        run_name(check), run_spec(cfg, check), "self", check, check_modes,
                        int(vc.get("games", cfg["games"])),
                    )
                ]
    matches = []
    for opp in opponents:
        if opp.name == cand.name or opp.spec == cand.spec:
            continue
        for mode in opp.modes:
            purpose = "value_check" if check is not None and opp.update == check and mode in check_modes else None
            for side in ("A", "B"):
                a = cand.name if side == "A" else opp.name
                name = f"{cand.name}_vs_{opp.name}_{mode}_{'candA' if side == 'A' else 'oppA'}"
                matches.append(
                    Match(
                        match_key(cand.name, opp.name, mode, a),
                        name,
                        mode,
                        cand,
                        opp,
                        side,
                        games_for(mode, opp.games),
                        kind,
                        purpose,
                    )
                )
    return matches


def _parse_time(text) -> Optional[float]:
    try:
        return datetime.fromisoformat(str(text)).timestamp()
    except (TypeError, ValueError):
        return None


def _now_iso(now: Optional[float] = None) -> str:
    return datetime.fromtimestamp(time.time() if now is None else now).astimezone().isoformat(timespec="seconds")


def choose_schedule(cfg, state: dict, candidate: int, now: Optional[float] = None) -> dict:
    """The schedule a newly taken candidate plays (module docstring, "Schedules"): ``{"kind": "full" | "rotation",
    "started", "members" (rotation), "rotation_index" (rotation), "value_check" (full: the previous full candidate)}``.
    Pure: :func:`commit_schedule` records it in ``state``."""
    now = time.time() if now is None else now
    if not cfg.get("rotate") or candidate == cfg.get("baseline_update"):
        kind = "full"
    else:
        every, last = cfg.get("full_every_hours"), _parse_time(state.get("last_full"))
        kind = "full" if every is not None and (last is None or now - last >= float(every) * 3600) else "rotation"
    schedule = {"kind": kind, "started": _now_iso(now)}
    if kind == "rotation":
        cand = Member(run_name(candidate), run_spec(cfg, candidate), "candidate", candidate)
        eligible = [
            m.name
            for m in panel_members(cfg)
            if set(m.modes) & set(cfg["rotate_modes"]) and m.name != cand.name and m.spec != cand.spec
        ]
        start = int(state.get("rotation_next", 0)) % len(eligible)
        k = min(int(cfg["rotate"]), len(eligible))
        schedule.update(members=[eligible[(start + i) % len(eligible)] for i in range(k)], rotation_index=start)
    elif cfg.get("value_check"):
        earlier = [
            int(u) for u, s in (state.get("schedules") or {}).items() if s.get("kind") == "full" and int(u) < candidate
        ]
        if earlier:
            schedule["value_check"] = max(earlier)
    return schedule


def commit_schedule(state: dict, candidate: int, schedule: dict) -> None:
    state.setdefault("schedules", {})[str(candidate)] = schedule
    if schedule["kind"] == "rotation":
        state["rotation_next"] = schedule["rotation_index"] + len(schedule["members"])
    else:
        state["last_full"] = schedule["started"]


def candidate_schedule(cfg, state: dict, candidate: int, dry_run: bool = False) -> Optional[dict]:
    """The candidate's schedule: the one state.json kept, else a new one (recorded unless ``dry_run``); None for a
    panel without schedules."""
    if not scheduled(cfg):
        return None
    schedule = (state.get("schedules") or {}).get(str(candidate))
    if schedule is None:
        schedule = choose_schedule(cfg, state, candidate)
        if not dry_run:
            commit_schedule(state, candidate, schedule)
    return schedule


def baseline_due(cfg, state: dict, available) -> Optional[int]:
    """``baseline_update`` when it has not been evaluated yet and its checkpoint exists."""
    base = cfg.get("baseline_update")
    if base is None or int(base) in state["candidates"] or int(base) not in set(available):
        return None
    return int(base)


def available_updates(cfg, settle_seconds: float = 0.0) -> list:
    """Updates with a checkpoint file in the run directory, older than ``settle_seconds``."""
    run = Path(cfg["run"])
    before, after = cfg["checkpoint"].split("{update}", 1)
    pattern = re.compile(re.escape(before) + r"(\d+)" + re.escape(after))
    now, found = time.time(), []
    for path in run.glob(cfg["checkpoint"].replace("{update}", "*")):
        m = pattern.fullmatch(path.relative_to(run).as_posix())
        if m and now - path.stat().st_mtime >= settle_seconds:
            found.append(int(m.group(1)))
    return sorted(found)


def next_candidate(cfg, state: dict, available, start_update: int = 0) -> Optional[int]:
    """The newest checkpoint at a multiple of ``every`` past the last candidate (and ``start_update``)."""
    last = max(state["candidates"], default=0)
    due = [u for u in available if u > 0 and u % cfg["every"] == 0 and u > last and u >= start_update]
    return max(due) if due else None


def load_state(root: Path) -> dict:
    path = root / "state.json"
    return json.loads(path.read_text()) if path.exists() else {"candidates": [], "current": None}


def save_state(root: Path, state: dict) -> None:
    tmp = root / "state.json.tmp"
    tmp.write_text(json.dumps(state, indent=1))
    os.replace(tmp, root / "state.json")


def eval_command(cfg, match: Match, out_dir: Path, workers: int) -> list:
    return [
        *shlex.split(cfg.get("command_prefix") or ""),
        cfg.get("python") or sys.executable,
        str(SCRIPTS_DIR / "eval_head_to_head.py"),
        "--match",
        match.a.spec,
        match.b.spec,
        "--games",
        str(match.games),
        "--a-seats",
        str(MODES[match.mode]),
        "--seed",
        str(cfg["seed"]),
        "--start-positions",
        cfg["start_positions"],
        "--random-start-fraction",
        str(cfg["random_start_fraction"]),
        "--softpick",
        str(cfg["softpick"]),
        "--max-length",
        str(cfg["max_length"]),
        "--stall",
        str(cfg["stall"]),
        "--workers",
        str(workers),
        *cfg["eval_args"],
        "--out",
        str(out_dir),
    ]


class Log:
    """Timestamped lines to the ladder's own log, stdout, and (results only) an optional shared results log."""

    def __init__(self, path: Path, results_log: Optional[Path] = None):
        self.path, self.results_log = path, results_log

    def __call__(self, message: str, result: bool = False) -> None:
        line = f"{datetime.now().astimezone().isoformat(timespec='seconds')} {message}"
        print(line, flush=True)
        for path in [self.path] + ([self.results_log] if result and self.results_log else []):
            with open(path, "a") as f:
                f.write(line + "\n")


def heartbeat(root: Path) -> None:
    """Touch the ladder's progress file (the long-run watchdog's stall check)."""
    try:
        (root / "heartbeat").touch()
    except OSError:
        pass


def run_match(cfg, match: Match, root: Path, workers: int, log) -> Optional[dict]:
    """Play one match; its ledger record, or None if it failed."""
    out_dir = root / match.name
    if out_dir.exists():  # a crashed attempt: eval_head_to_head.py appends to games.jsonl, so start clean
        stamp = f"{out_dir.name}.failed_{datetime.now():%Y%m%d_%H%M%S}"
        aside, n = out_dir.with_name(stamp), 1
        while aside.exists():  # two failures within a second
            aside, n = out_dir.with_name(f"{stamp}_{n}"), n + 1
        out_dir.rename(aside)
        log(f"set aside the incomplete {out_dir} as {aside.name}")
    cmd = eval_command(cfg, match, out_dir, workers)
    log(f"start {match.name} ({match.games} games, {workers} workers)")
    env = {**os.environ, **{k: str(v) for k, v in cfg["env"].items()}}
    heartbeat(root)
    with open(root / f"{match.name}.out", "w") as out, open(root / f"{match.name}.err", "w") as err:
        out.write(shlex.join(cmd) + "\n")
        out.flush()
        proc = subprocess.Popen(cmd, stdout=out, stderr=err, env=env)
        size = None
        while True:
            try:
                code = proc.wait(timeout=HEARTBEAT_SECONDS)
                break
            except subprocess.TimeoutExpired:
                try:
                    grown = (out_dir / "games.jsonl").stat().st_size
                except OSError:
                    grown = None
                if grown is not None and grown != size:  # a game finished: progress
                    size = grown
                    heartbeat(root)
    heartbeat(root)
    if code != 0:
        log(f"{match.name} failed (exit {code}); see {root / match.name}.err")
        return None
    try:
        record = read_match(out_dir)
    except (OSError, ValueError, KeyError) as e:
        log(f"{match.name}: unreadable results ({e!r})")
        return None
    if (record["a_spec"], record["b_spec"]) != (match.a.spec, match.b.spec):
        log(f"{match.name}: the results are for other specs")
        return None
    if record["games"] < cfg["min_games_fraction"] * match.games:
        log(f"{match.name}: only {record['games']} of {match.games} games finished")
        return None
    return {"ladder": cfg["name"], "source": "ladder", "time": _now_iso(), **match.fields(), **record}


def run_ladder(args) -> None:
    cfg = load_config(args.panel)
    root = out_root(cfg)
    if not args.dry_run:
        root.mkdir(parents=True, exist_ok=True)
    log = Log(root / "ladder.log", args.results_log)
    state = load_state(root)
    attempts = defaultdict(int)
    if not args.dry_run:
        log(f"ladder {cfg['name']} started: run {cfg['run']}, every {cfg['every']} updates, ledger {ledger_path(cfg)}")
    stop = root / "STOP"
    while True:
        if stop.exists() and not args.dry_run:
            log(f"{stop} exists: stopping (the current candidate resumes on the next start)")
            return
        if not args.dry_run:
            heartbeat(root)
        cfg = load_config(args.panel)
        available = available_updates(cfg, cfg["settle_seconds"])
        candidate = state.get("current")
        if candidate is None:
            candidate = baseline_due(cfg, state, available)
        if candidate is None:
            candidate = next_candidate(cfg, state, available, args.start_update)
        if candidate is None:
            if args.once or args.dry_run:
                print("no candidate due", flush=True)
                return
            time.sleep(cfg["poll_seconds"])
            continue
        schedule = candidate_schedule(cfg, state, candidate, dry_run=args.dry_run)
        if args.dry_run:
            done = {r["key"] for r in read_ledger(ledger_path(cfg))}
            if schedule:
                print(f"# {run_name(candidate)}: {schedule['kind']} schedule"
                      + (f" ({', '.join(schedule['members'])})" if schedule.get("members") else ""), flush=True)
            for m in plan_matches(cfg, candidate, available, schedule):
                if m.key not in done:
                    print(shlex.join(eval_command(cfg, m, root / m.name, args.workers or cfg["workers"])))
            return
        state["current"] = candidate
        save_state(root, state)
        if schedule:
            log(f"{run_name(candidate)}: {schedule['kind']} schedule"
                + (f" against {', '.join(schedule['members'])}" if schedule.get("members") else "")
                + (f", value check against {run_name(schedule['value_check'])}" if "value_check" in schedule else ""))
        while True:
            cfg = load_config(args.panel)  # re-read before every match
            if stop.exists():
                log(f"{stop} exists: stopping before the next match ({run_name(candidate)} resumes on the next start)")
                return
            done = {r["key"] for r in read_ledger(ledger_path(cfg))}
            todo = [
                m
                for m in plan_matches(cfg, candidate, available, schedule)
                if m.key not in done and attempts[m.key] < cfg["max_attempts"]
            ]
            if not todo:
                break
            match = todo[0]
            attempts[match.key] += 1
            record = run_match(cfg, match, root, args.workers or cfg["workers"], log)
            if record is None:
                time.sleep(cfg["retry_seconds"])
                continue
            append_ledger(ledger_path(cfg), record)
            log(
                f"{cfg['name']}/{match.name} {record['a_score']:.3f} +- {record['a_score_se_start']:.3f} "
                f"(equal {record['a_score_equal']}) worth {record['a_worth_share']:.3f} games {record['games']}",
                result=True,
            )
        done = {r["key"] for r in read_ledger(ledger_path(cfg))}
        given_up = [m.name for m in plan_matches(cfg, candidate, available, schedule) if m.key not in done]
        if given_up:
            log(f"gave up after {cfg['max_attempts']} attempts: {', '.join(given_up)}", result=True)
        state["candidates"].append(candidate)
        state["current"] = None
        save_state(root, state)
        try:
            write_reports(cfg, root, log, state)
        except Exception as e:  # a report must never stop a months-long ladder
            log(f"report failed: {e!r}")
        if args.once:
            return


# ---------------------------------------------------------------------------
# Ledger


def start_cluster(label: Optional[str], seed: int, start: int) -> str:
    """Games from one start position are correlated. A human start is one position whichever start index drew it;
    any other start (random, or none) is its own (seed, index)."""
    if label and label.startswith("human:"):
        return label
    return f"{label or 'none'}:{seed}:{start}"


def clustered_mean(values, clusters, weights=None) -> tuple:
    """The (weighted) mean of ``values`` and its SE with observations correlated within a cluster (CR1)."""
    values = np.asarray(values, dtype=float)
    weights = np.ones_like(values) if weights is None else np.asarray(weights, dtype=float)
    if not len(values):
        return float("nan"), float("nan")
    mean = float(np.sum(weights * values) / np.sum(weights))
    _, inv = np.unique(np.asarray(clusters), return_inverse=True)
    n_clusters = int(inv.max()) + 1
    if n_clusters < 2:
        return mean, float("nan")
    sums = np.bincount(inv.reshape(-1), weights=weights * (values - mean))
    return mean, float(math.sqrt(n_clusters / (n_clusters - 1) * np.sum(sums**2)) / np.sum(weights))


def clustered_mean_se(values, clusters) -> float:
    return clustered_mean(values, clusters)[1]


def read_match(out_dir: Path) -> dict:
    """The results of one single-match eval_head_to_head.py directory, in ledger form."""
    settings = json.loads((Path(out_dir) / "settings.json").read_text())
    if len(settings["matches"]) != 1:
        raise ValueError(f"{out_dir} holds {len(settings['matches'])} matches; the ledger takes one per directory")
    a_spec, b_spec = settings["matches"][0]
    seatings = [tuple(s) for s in settings.get("arrangements") or h2h.ARRANGEMENTS]
    a_seats = seatings[0].count("A")
    mode = {m: k for k, m in MODES.items()}.get(a_seats)
    if mode is None:
        raise ValueError(f"{out_dir}: side A holds {a_seats} seats (the ladder takes 2v2 and 1v3)")
    games = []
    with open(Path(out_dir) / "games.jsonl") as f:
        for line in f:
            try:
                games.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # a line cut short by a crash
    if not games:
        raise ValueError(f"{out_dir}: no games")
    summary = h2h.summarize(f"{a_spec} vs {b_spec}", {"A": a_spec, "B": b_spec}, games)
    records, a_scores = [], []
    for g in games:
        start = (g["game_idx"] % h2h.MATCH_STRIDE) // len(seatings)
        seats = "".join(g["seats"])
        records.append(
            [
                start_cluster(g.get("start"), settings["seed"], start),
                seats,
                [round(float(v), 6) for v in g["win_share"]],
                [round(float(v), 5) for v in g["worth_share"]],
            ]
        )
        a_scores.append(sum(w for w, s in zip(g["win_share"], seats) if s == "A"))
    return {
        "mode": mode,
        "a_spec": a_spec,
        "b_spec": b_spec,
        "out": str(out_dir),
        "settings": {k: settings.get(k) for k in SHARED_SETTINGS},
        "games": summary["games"],
        "a_score": summary["A_score"],
        "a_score_se": summary["A_score_se"],
        "a_score_se_start": clustered_mean_se(a_scores, [r[0] for r in records]),
        "a_score_equal": summary["A_score_equal"],
        "a_worth_share": summary["A_worth_share"],
        "seat_win_rate": summary["seat_win_rate"],
        "by_arrangement": summary["by_arrangement"],
        "terminations": summary["terminations"],
        "game_records": records,  # [start cluster, seats, win shares, net-worth shares] per game
    }


def read_ledger(path) -> list:
    path = Path(path)
    if not path.exists():
        return []
    records = []
    with open(path) as f:
        for number, line in enumerate(f, 1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"warning: {path}:{number} is not JSON (cut short?); skipped", file=sys.stderr)
    return records


def spec_conflicts(records) -> list:
    """Names the ledger gives two different specs (a renamed or edited panel member would mix two players)."""
    specs, conflicts = {}, []
    for r in records:
        for side in ("a", "b"):
            if specs.setdefault(r[side], r[f"{side}_spec"]) != r[f"{side}_spec"]:
                conflicts.append(r[side])
    return sorted(set(conflicts))


def append_ledger(path, record: dict) -> None:
    path = Path(path)
    conflicts = spec_conflicts(read_ledger(path) + [record])
    if conflicts:
        raise ValueError(f"{path}: {', '.join(conflicts)} would have two specs; rename the panel member")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record, separators=(",", ":")) + "\n")
        f.flush()
        os.fsync(f.fileno())


def settings_mismatch(cfg, settings: dict) -> list:
    """Shared settings of a match that differ from the panel's (the start file is compared by name)."""
    bad = []
    for key in SHARED_SETTINGS:
        ours, theirs = cfg[key], settings.get(key)
        if key == "start_positions":
            ours, theirs = Path(ours).name if ours else None, Path(theirs).name if theirs else None
        if ours != theirs:
            bad.append(f"{key} {theirs!r} (panel: {ours!r})")
    return bad


def identify(cfg, spec: str) -> Optional[Member]:
    """The player a spec is: a panel member with that exact spec, or a run checkpoint played with the run's spec."""
    for m in panel_members(cfg):
        if m.spec == spec:
            return m
    parts = run_spec(cfg, "\0").split("\0")
    m = re.fullmatch(r"(\d+)".join(re.escape(p) for p in parts), spec)
    if m:
        update = int(m.group(1))
        return Member(run_name(update), spec, "self", update)
    return None


def import_dirs(cfg, ledger, dirs, force: bool = False) -> int:
    """Add finished single-match eval directories to the ledger; returns how many were added."""
    done = {r["key"] for r in read_ledger(ledger)}
    added = 0
    for d in dirs:
        try:
            record = read_match(Path(d))
        except (OSError, ValueError, KeyError) as e:
            print(f"skip {d}: {e}")
            continue
        bad = settings_mismatch(cfg, record["settings"])
        if bad and not force:
            print(f"skip {d}: {'; '.join(bad)}")
            continue
        a, b = identify(cfg, record["a_spec"]), identify(cfg, record["b_spec"])
        if a is None or b is None:
            print(f"skip {d}: unknown player {record['a_spec'] if a is None else record['b_spec']}")
            continue
        if a.role == "panel" and b.role == "panel":
            print(f"skip {d}: two panel members")
            continue
        if b.role == "panel" or (a.role != "panel" and a.update > b.update):
            cand, opp, side = a, b, "A"
        else:
            cand, opp, side = b, a, "B"
        cand = Member(cand.name, cand.spec, "candidate", cand.update)
        match = Match(
            match_key(cand.name, opp.name, record["mode"], a.name),
            Path(d).name,
            record["mode"],
            cand,
            opp,
            side,
            record["games"],
        )
        if match.key in done:
            print(f"skip {d}: {match.key} is in the ledger")
            continue
        append_ledger(ledger, {"ladder": cfg["name"], "source": "import", **match.fields(), **record})
        done.add(match.key)
        added += 1
        print(f"added {d}: {match.key}, {record['games']} games, A {record['a_score']:.3f}")
    return added


# ---------------------------------------------------------------------------
# Ratings


@dataclass
class GameTable:
    names: list  # player names (ids index this)
    players: np.ndarray  # (games, 4) player id at each seat
    side_a: np.ndarray  # (games, 4) the seat is side A's
    y: np.ndarray  # (games, 4) outcome shares (win shares or net-worth shares)
    cluster: np.ndarray  # (games,) start cluster id
    n_clusters: int


def game_table(records, modes, outcome: str) -> GameTable:
    names, ids, clusters = [], {}, {}
    players, side_a, ys, cl = [], [], [], []
    k = 2 if outcome == "win" else 3
    for r in records:
        if r["mode"] not in modes:
            continue
        ia, ib = (ids.setdefault(r[s], len(ids)) for s in ("a", "b"))
        for game in r["game_records"]:
            sides = [s == "A" for s in game[1]]
            players.append([ia if s else ib for s in sides])
            side_a.append(sides)
            ys.append(game[k])
            cl.append(clusters.setdefault(game[0], len(clusters)))
    names = sorted(ids, key=ids.get)
    return GameTable(
        names,
        np.array(players, dtype=np.int64).reshape(-1, 4),
        np.array(side_a, dtype=bool).reshape(-1, 4),
        np.array(ys, dtype=float).reshape(-1, 4),
        np.array(cl, dtype=np.int64),
        len(clusters),
    )


@dataclass
class RatingFit:
    label: str
    anchor: str
    rating: dict  # name -> log-strength, the anchor 0
    columns: dict  # name -> parameter column (the anchor: None)
    cov: np.ndarray  # sandwich clustered by start: the reported SEs
    cov_iid: np.ndarray  # sandwich with every game independent
    games: dict  # name -> games played
    n_games: int
    n_clusters: int
    side_a: Optional[tuple]  # (effect, se) of being listed as side A, or None when not fit
    seat: list  # (effect, se) of seats 1-3 against seat 0
    dropped: list  # players not connected to the anchor

    def _vector(self, plus: str, minus: Optional[str] = None) -> np.ndarray:
        c = np.zeros(len(self.cov))
        for name, sign in ((plus, 1.0), (minus, -1.0)):
            if name is not None and self.columns[name] is not None:
                c[self.columns[name]] += sign
        return c

    def contrast(self, a: str, b: str, iid: bool = False) -> tuple:
        """``rating[a] - rating[b]`` and its SE (log-strength units)."""
        c = self._vector(a, b)
        cov = self.cov_iid if iid else self.cov
        return self.rating[a] - self.rating[b], float(math.sqrt(max(c @ cov @ c, 0.0)))

    def se(self, name: str, iid: bool = False) -> float:
        return self.contrast(name, self.anchor, iid)[1]


def _probs(theta: np.ndarray, col: np.ndarray) -> np.ndarray:
    logits = theta[col].sum(-1)
    logits -= logits.max(-1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(-1, keepdims=True)


def _scatter_vector(col: np.ndarray, w: np.ndarray, n_params: int) -> np.ndarray:
    """sum over rows and seats of w[row, seat] * x[row, seat] (x: the seat's design row, its columns ``col``)."""
    k = col.shape[-1]
    return np.bincount(col.ravel(), weights=np.repeat(w.ravel(), k), minlength=n_params + 1)[:n_params]


def _scatter_matrix(col: np.ndarray, w: np.ndarray, n_params: int) -> np.ndarray:
    """sum over rows and seat pairs of w[row, i, j] * x[row, i] x[row, j]^T."""
    rows = col[:, :, None, :, None] * (n_params + 1)
    flat = rows + col[:, None, :, None, :]
    weights = np.broadcast_to(w[:, :, :, None, None], flat.shape)
    m = np.bincount(flat.ravel(), weights=weights.ravel(), minlength=(n_params + 1) ** 2)
    return m.reshape(n_params + 1, n_params + 1)[:n_params, :n_params]


def _information(col: np.ndarray, p: np.ndarray, totals: np.ndarray, n_params: int) -> np.ndarray:
    w = totals[:, None, None] * (p[:, :, None] * np.eye(4) - p[:, :, None] * p[:, None, :])
    return _scatter_matrix(col, w, n_params)


def fit_ratings(
    table: GameTable,
    anchor: Optional[str] = None,
    side_effect: bool = False,
    seat_effects: bool = True,
    ridge: float = 0.01,
    label: str = "",
    max_iter: int = 100,
) -> RatingFit:
    """Four-seat Luce model fit by Newton's method on the (quasi-)likelihood sum y log p, with a small ridge that
    only keeps an undefeated player finite. ``seat_effects``: seats 1-3 against seat 0. ``side_effect``: a
    log-strength bonus for side A's seats (dropped when the design can't separate it from the ratings). Off by
    default: the ladder plays every pairing both ways, so a listing bias cancels; on one-way data (the old ladder
    always listed the newer checkpoint first) the term also soaks up non-transitive step wins."""
    if len(table.players) == 0:
        raise ValueError("no games")
    names = table.names
    if anchor not in names:
        counts = np.bincount(table.players.ravel(), minlength=len(names))
        anchor = names[int(np.argmax(counts))]
    # Ratings are identified only within the anchor's connected component.
    parent = list(range(len(names)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in np.unique(np.stack([table.players.min(1), table.players.max(1)], 1), axis=0):
        parent[find(a)] = find(b)
    root = find(names.index(anchor))
    connected = np.array([find(i) == root for i in range(len(names))])
    keep = connected[table.players[:, 0]]
    players, side_a, y, cluster = table.players[keep], table.side_a[keep], table.y[keep], table.cluster[keep]
    rated = [n for i, n in enumerate(names) if connected[i]]
    dropped = [n for i, n in enumerate(names) if not connected[i]]
    others = [n for n in rated if n != anchor]
    for use_side in ([True, False] if side_effect else [False]):
        n_params = len(others) + (3 if seat_effects else 0) + (1 if use_side else 0)
        dummy = n_params
        player_col = np.full(len(names), dummy)
        for i, n in enumerate(others):
            player_col[names.index(n)] = i
        col = np.full(players.shape + (3,), dummy, dtype=np.int64)
        col[:, :, 0] = player_col[players]
        if seat_effects:
            col[:, 1:, 1] = len(others) + np.arange(3)
        if use_side:
            col[:, :, 2] = np.where(side_a, n_params - 1, dummy)
        uniq, inv, _ = np.unique(col.reshape(len(col), -1), axis=0, return_inverse=True, return_counts=True)
        inv = inv.reshape(-1)
        ucol = uniq.reshape(len(uniq), 4, 3)
        sums = np.stack([np.bincount(inv, weights=y[:, i], minlength=len(uniq)) for i in range(4)], 1)
        totals = sums.sum(1)
        info0 = _information(ucol, np.full((len(uniq), 4), 0.25), totals, n_params)
        if np.linalg.matrix_rank(info0) == n_params:
            break
        if not use_side:
            raise ValueError(f"{label}: ratings not identified")
    theta = np.zeros(n_params + 1)

    def objective(t):
        return float(
            np.sum(sums * np.log(np.clip(_probs(t, ucol), 1e-300, None))) - 0.5 * ridge * t[:n_params] @ t[:n_params]
        )

    current = objective(theta)
    for _ in range(max_iter):
        p = _probs(theta, ucol)
        grad = _scatter_vector(ucol, sums - totals[:, None] * p, n_params) - ridge * theta[:n_params]
        info = _information(ucol, p, totals, n_params) + ridge * np.eye(n_params)
        step, scale = np.linalg.solve(info, grad), 1.0
        while True:
            trial = theta.copy()
            trial[:n_params] += scale * step
            value = objective(trial)
            if value >= current - 1e-12 * abs(current) - 1e-9 or scale < 1e-6:
                break
            scale /= 2
        theta, current = trial, value
        if np.max(np.abs(scale * step)) < 1e-9:
            break
    p = _probs(theta, ucol)
    bread = np.linalg.inv(_information(ucol, p, totals, n_params) + ridge * np.eye(n_params))
    resid = y - y.sum(1, keepdims=True) * p[inv]  # each game's score, seat by seat
    _, cluster = np.unique(cluster, return_inverse=True)  # the clusters left after dropping games
    cluster = cluster.reshape(-1)
    n_clusters = int(cluster.max()) + 1
    flat = cluster[:, None, None] * (n_params + 1) + col
    per_cluster = np.bincount(flat.ravel(), weights=np.repeat(resid.ravel(), 3), minlength=n_clusters * (n_params + 1))
    scores = per_cluster.reshape(n_clusters, n_params + 1)[:, :n_params]
    meat = scores.T @ scores * n_clusters / max(n_clusters - 1, 1)
    outer = np.stack(
        [
            np.stack([np.bincount(inv, weights=resid[:, i] * resid[:, j], minlength=len(uniq)) for j in range(4)], 1)
            for i in range(4)
        ],
        1,
    )
    meat_iid = _scatter_matrix(ucol, outer, n_params) * len(y) / max(len(y) - 1, 1)
    cov, cov_iid = bread @ meat @ bread, bread @ meat_iid @ bread
    sd = np.sqrt(np.maximum(np.diag(cov), 0.0))
    first, second = players.min(1), players.max(1)  # a game's two players
    played = np.bincount(first, minlength=len(names)) + np.bincount(second[second != first], minlength=len(names))
    seat_base = len(others)
    return RatingFit(
        label=label,
        anchor=anchor,
        rating={anchor: 0.0, **{n: float(theta[i]) for i, n in enumerate(others)}},
        columns={anchor: None, **{n: i for i, n in enumerate(others)}},
        cov=cov,
        cov_iid=cov_iid,
        games={n: int(played[names.index(n)]) for n in rated},
        n_games=len(y),
        n_clusters=n_clusters,
        side_a=(float(theta[n_params - 1]), float(sd[n_params - 1])) if use_side else None,
        seat=[(float(theta[seat_base + i]), float(sd[seat_base + i])) for i in range(3)] if seat_effects else [],
        dropped=dropped,
    )


def fit_all(records, anchor: Optional[str] = None, fits=FITS, side_effect: bool = False) -> dict:
    """label -> RatingFit (or the reason there is none)."""
    out = {}
    for label, modes, outcome in fits:
        table = game_table(records, modes, outcome)
        try:
            out[label] = fit_ratings(table, anchor=anchor, label=label, side_effect=side_effect)
        except ValueError as e:
            out[label] = str(e)
    return out


def directions(records) -> dict:
    """(candidate, opponent, mode) -> {candidate side: games}."""
    out = defaultdict(lambda: defaultdict(int))
    for r in records:
        out[(r["candidate"], r["opponent"], r["mode"])][r["candidate_side"]] += r["games"]
    return out


def listing_effect(records) -> tuple:
    """Side A's 2v2 win-share bonus for being listed first, from the 2v2 pairings played both ways: side A's score
    minus 0.5, each direction weighted equally within its pairing (a pairing's two directions seat the same
    arrangements of the same starts, so only the listing differs). (effect, start-clustered SE, pairings)."""
    both = {k: v for k, v in directions(records).items() if k[2] == "2v2" and len(v) == 2}
    values, clusters, weights = [], [], []
    for r in records:
        sides = both.get((r["candidate"], r["opponent"], r["mode"]))
        if sides is None:
            continue
        weight = sum(sides.values()) / 2 / sides[r["candidate_side"]]
        for game in r["game_records"]:
            values.append(sum(v for v, s in zip(game[2], game[1]) if s == "A") - 0.5)
            clusters.append(game[0])
            weights.append(weight)
    mean, se = clustered_mean(values, clusters, weights)
    return mean, se, len(both)


def player_updates(records) -> dict:
    """name -> run update (None for panel members that are not run checkpoints)."""
    updates = {}
    for r in records:
        updates[r["candidate"]] = r.get("candidate_update")
        updates.setdefault(r["opponent"], r.get("opponent_update"))
    return updates


def fixed_panel_names(records) -> list:
    return sorted({r["opponent"] for r in records if r.get("opponent_role") == "panel"})


def _ordered_players(records, names) -> list:
    """Run checkpoints by update, then the other players by name."""
    updates = player_updates(records)
    run = sorted((n for n in names if updates.get(n) is not None), key=lambda n: updates[n])
    return run + sorted(n for n in names if updates.get(n) is None)


def format_report(records, fits: dict) -> str:
    updates = player_updates(records)
    names = set()
    for fit in fits.values():
        if isinstance(fit, RatingFit):
            names |= set(fit.rating)
    fixed = set(fixed_panel_names(records))
    lines = [
        f"{len(records)} matches, {sum(r['games'] for r in records)} games; fixed panel: {', '.join(sorted(fixed))}"
    ]
    for problem in spec_conflicts(records):
        lines.append(f"WARNING: {problem} has two specs in the ledger")
    if len({json.dumps(r["settings"], sort_keys=True) for r in records}) > 1:
        lines.append("WARNING: the ledger mixes match settings (seed / starts / softpick / limits)")
    header = f"{'player':<16}{'update':>7}"
    for label in fits:
        header += f"  {label + ' Elo':>12} {'games':>6}"
    lines += ["", "Elo ± SE clustered by start (400/ln10 per unit of log-strength); * = fixed panel", header]
    for name in _ordered_players(records, names):
        update = updates.get(name)
        row = f"{name + ('*' if name in fixed else ''):<16}{'' if update is None else update:>7}"
        for fit in fits.values():
            if isinstance(fit, RatingFit) and name in fit.rating:
                cell = (
                    "0 (anchor)"
                    if name == fit.anchor
                    else f"{fit.rating[name] * ELO_PER_UNIT:+.0f} ± {fit.se(name) * ELO_PER_UNIT:.0f}"
                )
                row += f"  {cell:>12} {fit.games.get(name, 0):>6}"
            else:
                row += f"  {'':>12} {'':>6}"
        lines.append(row)
    lines.append("")
    effect, se, pairings = listing_effect(records)
    lines.append(
        f"listing (side A) effect, 2v2 pairings played both ways: {effect:+.3f} ± {se:.3f} in score "
        f"({pairings} pairings)"
        if pairings
        else "listing (side A) effect: no 2v2 pairing played both ways"
    )
    one_way = [f"{c} vs {o} {m}" for (c, o, m), v in directions(records).items() if len(v) == 1]
    if one_way:
        lines.append(
            f"played one way only ({len(one_way)}): {', '.join(sorted(one_way)[:8])}"
            + (" ..." if len(one_way) > 8 else "")
        )
    for label, fit in fits.items():
        if not isinstance(fit, RatingFit):
            lines.append(f"{label}: no fit ({fit})")
            continue
        ratio = [fit.se(n) / fit.se(n, iid=True) for n in fit.rating if n != fit.anchor and fit.se(n, iid=True) > 0]
        side = (
            f", side A term {fit.side_a[0] * ELO_PER_UNIT:+.1f} ± {fit.side_a[1] * ELO_PER_UNIT:.1f} Elo"
            if fit.side_a
            else ""
        )
        seats = ", ".join(f"seat{i + 1} {e * ELO_PER_UNIT:+.0f}" for i, (e, _) in enumerate(fit.seat))
        lines.append(
            f"{label}: anchor {fit.anchor}, {fit.n_games} games, {fit.n_clusters} start clusters{side}, "
            f"{seats} (vs seat0); clustered/iid SE median {np.median(ratio) if ratio else float('nan'):.2f}"
            + (f"; not connected: {', '.join(fit.dropped)}" if fit.dropped else "")
        )
    return "\n".join(lines)


def fits_to_json(records, fits: dict) -> dict:
    updates = player_updates(records)
    out = {}
    for label, fit in fits.items():
        if not isinstance(fit, RatingFit):
            out[label] = {"error": fit}
            continue
        out[label] = {
            "anchor": fit.anchor,
            "games": fit.n_games,
            "clusters": fit.n_clusters,
            "side_a_elo": [v * ELO_PER_UNIT for v in fit.side_a] if fit.side_a else None,
            "seat_elo": [[e * ELO_PER_UNIT, s * ELO_PER_UNIT] for e, s in fit.seat],
            "players": {
                n: {
                    "update": updates.get(n),
                    "elo": fit.rating[n] * ELO_PER_UNIT,
                    "se": fit.se(n) * ELO_PER_UNIT,
                    "se_iid": fit.se(n, iid=True) * ELO_PER_UNIT,
                    "games": fit.games.get(n, 0),
                }
                for n in fit.rating
            },
            "dropped": fit.dropped,
        }
    return out


def plot_ratings(records, fits: dict, path) -> None:
    """One panel per rating: run checkpoints by update (95% intervals), fixed references as labeled lines."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    updates = player_updates(records)
    shown = {k: v for k, v in fits.items() if isinstance(v, RatingFit)}
    if not shown:
        return
    fig, axes = plt.subplots(len(shown), 1, figsize=(9, 3.0 * len(shown)), sharex=True, squeeze=False)
    fig.patch.set_facecolor("#fcfcfb")
    for ax, (label, fit) in zip(axes[:, 0], shown.items()):
        ax.set_facecolor("#fcfcfb")
        run = sorted((updates[n], n) for n in fit.rating if updates.get(n) is not None)
        xs = [u for u, _ in run]
        ys = [fit.rating[n] * ELO_PER_UNIT for _, n in run]
        err = [1.96 * fit.se(n) * ELO_PER_UNIT for _, n in run]
        ax.errorbar(xs, ys, yerr=err, color="#2a78d6", lw=2, marker="o", ms=4, capsize=0, elinewidth=1)
        for name in fit.rating:
            if updates.get(name) is None:
                v = fit.rating[name] * ELO_PER_UNIT
                ax.axhline(v, color="#8a8984", lw=1, ls="--")
                ax.annotate(
                    name,
                    (1.0, v),
                    xycoords=("axes fraction", "data"),
                    xytext=(4, 0),
                    textcoords="offset points",
                    va="center",
                    fontsize=8,
                    color="#52514e",
                )
        ax.set_ylabel(f"{label} Elo\n(anchor {fit.anchor})", color="#0b0b0b")
        ax.grid(axis="y", color="#e6e5e0", lw=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors="#52514e")
    axes[-1, 0].set_xlabel("update", color="#0b0b0b")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Decision


def excess_share(records, candidate: str, opponent: str, modes, outcome: str) -> tuple:
    """The candidate's mean share above its equal share (seats / 4) against one opponent, both directions pooled:
    (mean, start-clustered SE, games)."""
    values, clusters = [], []
    k = 2 if outcome == "win" else 3
    for r in records:
        if r["candidate"] != candidate or r["opponent"] != opponent or r["mode"] not in modes:
            continue
        side = r["candidate_side"]
        for game in r["game_records"]:
            values.append(sum(v for v, s in zip(game[k], game[1]) if s == side) - game[1].count(side) / 4)
            clusters.append(game[0])
    if not values:
        return float("nan"), float("nan"), 0
    return float(np.mean(values)), clustered_mean_se(values, clusters), len(values)


def decide(
    records,
    back: int = 1,
    latest: Optional[int] = None,
    earlier: Optional[int] = None,
    anchor: Optional[str] = None,
    z_crit: float = 1.645,
    fits=FITS,
    candidates=None,
) -> list:
    """Per rating: is the latest candidate significantly above the one ``back`` candidates earlier? Fit on the
    matches against the fixed panel only, restricted to the panel members both candidates played. ``candidates``:
    only these candidate updates count (e.g. :func:`full_candidates`, the ones that played the whole panel)."""
    fixed = [r for r in records if r.get("opponent_role") == "panel"]
    if candidates is not None:
        keep = set(candidates)
        fixed = [r for r in fixed if r["candidate_update"] in keep]
    results = []
    for label, modes, outcome in fits:
        recs = [r for r in fixed if r["mode"] in modes]
        names = {r["candidate_update"]: r["candidate"] for r in recs}
        cands = sorted(names)
        result = {"fit": label}
        results.append(result)
        late = latest if latest is not None else (cands[-1] if cands else None)
        if late not in names:
            result["error"] = f"no fixed-panel matches for {'any candidate' if late is None else run_name(late)}"
            continue
        if earlier is not None:
            early = earlier
        else:
            i = cands.index(late) - back
            early = cands[i] if i >= 0 else None
        if early not in names:
            result["error"] = (
                f"no candidate {back} back from {names[late]}"
                if earlier is None
                else f"no fixed-panel matches for {run_name(earlier)}"
            )
            continue

        def faced(u):
            return {r["opponent"] for r in recs if r["candidate_update"] == u}

        common = faced(late) & faced(early)
        dirs = directions(recs)
        result.update(
            latest=late,
            earlier=early,
            opponents=sorted(common),
            missing=sorted(faced(early) - faced(late)),
            one_way=sorted(
                f"{c} vs {o} {m}"
                for (c, o, m), v in dirs.items()
                if len(v) == 1 and c in (names[late], names[early]) and o in common
            ),
        )
        if not common:
            result["error"] = "no panel member played by both"
            continue
        table = game_table([r for r in recs if r["opponent"] in common], modes, outcome)
        try:
            fit = fit_ratings(table, anchor=anchor if anchor in common else sorted(common)[0], label=label)
        except ValueError as e:
            result["error"] = f"no fit ({e})"
            continue
        diff, se = fit.contrast(names[late], names[early])
        z = diff / se if se > 0 else float("nan")
        result.update(
            diff=diff,
            se=se,
            elo=diff * ELO_PER_UNIT,
            elo_se=se * ELO_PER_UNIT,
            z=z,
            p_one_sided=0.5 * math.erfc(z / math.sqrt(2)),
            verdict="above" if z > z_crit else "below" if z < -z_crit else "no significant change",
            implied_2v2=1 / (1 + math.exp(-diff)),  # latest's expected 2v2 share (win or worth) against earlier
            per_opponent={
                opp: {
                    "latest": excess_share(recs, names[late], opp, modes, outcome),
                    "earlier": excess_share(recs, names[early], opp, modes, outcome),
                }
                for opp in sorted(common)
            },
        )
    return results


def full_candidates(records) -> set:
    """Candidate updates that played a full schedule (records written before schedules existed count as full)."""
    return {r["candidate_update"] for r in records if r.get("schedule", "full") == "full"}


def format_decision(results) -> str:
    lines = ["Decision (fixed panel only): latest vs earlier, one-sided"]
    for r in results:
        if "error" in r:
            lines.append(f"  {r['fit']:<6} {r['error']}")
            continue
        lines.append(
            f"  {r['fit']:<6} {run_name(r['latest'])} vs {run_name(r['earlier'])}: {r['elo']:+.0f} ± {r['elo_se']:.0f} Elo, "
            f"z {r['z']:+.2f}, p {r['p_one_sided']:.3f} -> {r['verdict'].upper()} "
            f"(latest's implied 2v2 share against earlier {r['implied_2v2']:.3f}; panel {', '.join(r['opponents'])}"
            + (f"; latest has not played {', '.join(r['missing'])} yet" if r["missing"] else "")
            + ")"
        )
        if r["one_way"]:
            lines.append(f"         played one way only (a listing bias does not cancel): {', '.join(r['one_way'])}")
        for opp, v in r["per_opponent"].items():
            (lm, ls, ln), (em, es, en) = v["latest"], v["earlier"]
            lines.append(
                f"         vs {opp:<14} excess share {lm:+.3f} ± {ls:.3f} ({ln} games)  "
                f"earlier {em:+.3f} ± {es:.3f} ({en})"
            )
    return "\n".join(lines)


def decision_line(cfg, results) -> str:
    parts = []
    for r in results:
        if "error" not in r:
            parts.append(f"{r['fit']} {r['elo']:+.0f} ± {r['elo_se']:.0f} Elo (z {r['z']:+.1f}, {r['verdict']})")
    first = next((r for r in results if "error" not in r), None)
    head = f"{run_name(first['latest'])} vs {run_name(first['earlier'])}" if first else "no comparison yet"
    return f"{cfg['name']} {head} vs fixed panel: " + ("; ".join(parts) or "-")


def write_reports(cfg, root: Path, log, state: Optional[dict] = None) -> None:
    records = read_ledger(ledger_path(cfg))
    fits = fit_all(records, anchor=cfg["anchor"])
    full = full_candidates(records) if scheduled(cfg) else None
    results = decide(records, back=cfg["back"], anchor=cfg["anchor"], candidates=full)
    (root / "report.txt").write_text(format_report(records, fits) + "\n\n" + format_decision(results) + "\n")
    out = {"fits": fits_to_json(records, fits), "decision": results}
    if state is not None and state.get("schedules"):
        out["schedules"] = state["schedules"]
    (root / "ratings.json").write_text(json.dumps(out, indent=1))
    try:
        plot_ratings(records, fits, root / "ratings.png")
    except Exception as e:  # the plot is a convenience; the ledger and report are what matter
        log(f"plot failed: {e!r}")
    log(decision_line(cfg, results), result=True)


def spec_paths(spec: str) -> list:
    """The checkpoint files of a player spec (``<policy>[+<value>][#head]@<readouts>/<c_puct>:<options>``)."""
    checkpoint = spec.partition(":")[0].partition("@")[0].partition("#")[0]
    return [p for p in checkpoint.split("+") if p]


def check_panel(cfg) -> list:
    """``(player, path, role, exists)`` for every checkpoint the panel file names: the members' specs, and the run
    checkpoints of members given by update and of ``baseline_update``."""
    rows = []
    for m in panel_members(cfg):
        role = "run" if m.update is not None else "panel"
        rows += [(m.name, path, role, Path(path).exists()) for path in spec_paths(m.spec)]
    base = cfg.get("baseline_update")
    if base is not None and all(m.update != int(base) for m in panel_members(cfg)):
        rows += [(run_name(base), path, "run", Path(path).exists()) for path in spec_paths(run_spec(cfg, base))]
    return rows


# ---------------------------------------------------------------------------


def _ledger_arg(args) -> Path:
    if args.ledger:
        return args.ledger
    if not args.panel:
        raise SystemExit("give --ledger or --panel")
    return ledger_path(load_config(args.panel))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("run", help="schedule and play the panel matches of each new checkpoint")
    p.add_argument("--panel", type=Path, required=True)
    p.add_argument("--workers", type=int, default=None, help="override the panel file's workers (the GPU-share cap)")
    p.add_argument("--start-update", type=int, default=0, help="first candidate update considered")
    p.add_argument("--once", action="store_true", help="evaluate at most one candidate, then exit")
    p.add_argument("--dry-run", action="store_true", help="print the next candidate's remaining commands and exit")
    p.add_argument("--results-log", type=Path, default=None, help="also append result lines here (e.g. logs/az1.log)")

    for name, help_ in (("report", "ratings of every player"), ("decide", "latest vs N candidates earlier")):
        p = sub.add_parser(name, help=help_)
        p.add_argument("--panel", type=Path, default=None)
        p.add_argument("--ledger", type=Path, default=None)
        p.add_argument("--anchor", default=None, help="rating 0 (default: the panel file's anchor, u0)")
        p.add_argument("--json", type=Path, default=None, help="also write the results as JSON here")
        if name == "report":
            p.add_argument("--plot", type=Path, default=None, help="PNG of rating against update")
            p.add_argument("--fixed-only", action="store_true", help="only the matches against the fixed panel")
            p.add_argument(
                "--side-effect",
                action="store_true",
                help="fit a side-A (listing) term too; on one-way data it also absorbs non-transitivity",
            )
        else:
            p.add_argument("--back", type=int, default=None, help="compare with the candidate this many back (1)")
            p.add_argument("--latest", type=int, default=None, help="latest candidate update (default: newest)")
            p.add_argument("--earlier", type=int, default=None, help="earlier candidate update (instead of --back)")
            p.add_argument("--z", type=float, default=1.645, help="one-sided critical z (1.645: 5%%)")
            p.add_argument("--full", action="store_true", help="only candidates that played the whole panel")

    p = sub.add_parser("check", help="does every checkpoint the panel file names exist?")
    p.add_argument("--panel", type=Path, required=True)
    p.add_argument("--strict", action="store_true", help="also fail on run checkpoints not written yet")

    p = sub.add_parser("import", help="add existing single-match eval directories to the ledger")
    p.add_argument("--panel", type=Path, required=True)
    p.add_argument("--ledger", type=Path, default=None)
    p.add_argument("--force", action="store_true", help="import even when the shared settings differ")
    p.add_argument("dirs", nargs="+", type=Path)

    args = parser.parse_args()
    if args.command == "run":
        run_ladder(args)
        return
    if args.command == "check":
        cfg = load_config(args.panel)
        missing = 0
        for name, path, role, exists in check_panel(cfg):
            print(f"{'ok     ' if exists else 'MISSING'} {name:<16} {role:<6} {path}")
            missing += not exists and (role == "panel" or args.strict)
        if missing:
            raise SystemExit(f"{missing} checkpoint(s) missing")
        return
    if args.command == "import":
        cfg = load_config(args.panel)
        n = import_dirs(cfg, args.ledger or ledger_path(cfg), args.dirs, force=args.force)
        print(f"{n} matches added")
        return
    cfg = load_config(args.panel) if args.panel else DEFAULTS
    anchor = args.anchor or cfg["anchor"]
    records = read_ledger(_ledger_arg(args))
    if not records:
        raise SystemExit("the ledger is empty")
    if args.command == "report":
        if args.fixed_only:
            records = [r for r in records if r.get("opponent_role") == "panel"]
        fits = fit_all(records, anchor=anchor, side_effect=args.side_effect)
        print(format_report(records, fits))
        if args.plot:
            plot_ratings(records, fits, args.plot)
            print(f"plot: {args.plot}")
        if args.json:
            args.json.write_text(json.dumps(fits_to_json(records, fits), indent=1))
    else:
        back = args.back if args.back is not None else cfg["back"]
        results = decide(
            records,
            back=back,
            latest=args.latest,
            earlier=args.earlier,
            anchor=anchor,
            z_crit=args.z,
            candidates=full_candidates(records) if args.full else None,
        )
        print(format_decision(results))
        if args.json:
            args.json.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
