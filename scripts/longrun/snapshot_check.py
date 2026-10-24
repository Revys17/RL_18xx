"""Is a search-learning run directory's snapshot resumable? (launch.sh, before every --resume.)

    snapshot_check.py <run dir>            print the snapshot's update and exit 0, or say what is wrong, exit 1
    snapshot_check.py <run dir> --repair   make it resumable again (docs/longrun_runbook.md, "Recovering a
                                           snapshot"), keeping every file it replaces as <name>.bad-<time>

``search_learning.run`` writes the checkpoints atomically but ``optimizer.pth`` and ``state.json`` in place,
so a kill in the middle of a snapshot (an OOM kill, a power cut, the 5-s user-manager timeout at shutdown)
can leave either unreadable; this catches that before a resume crashes on it, again and again. The repair
moves an unreadable ``optimizer.pth`` aside (the resume starts AdamW afresh) and rebuilds an unreadable
``state.json`` at the newest snapshot whose checkpoints exist, from ``config.json`` (the run's merged
config, written at every start) and that update's ``history.jsonl`` line (``kl_coef``, ``anchor_update``).
"""

import json
import sys
import time
from pathlib import Path


class NotResumable(Exception):
    pass


def _state(run_dir: Path):
    path = run_dir / "state.json"
    try:
        state = json.loads(path.read_text())
        return state, int(state["snapshot_update"]), state["config"]
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise NotResumable(f"{path} is unreadable: {e!r}")


def _workdir(run_dir: Path, config: dict) -> Path:
    """Where the run's relative paths (its starting policy) resolve: <workdir>/<out_dir>/<run_name> is the
    run directory, whatever directory this is called from."""
    out = Path(config.get("out_dir") or ".")
    tail = out / config.get("run_name", run_dir.name)
    resolved = run_dir.resolve()
    if not out.is_absolute() and resolved.parts[-len(tail.parts):] == tail.parts:
        return Path(*resolved.parts[: -len(tail.parts)])
    return Path.cwd()


def _needed(run_dir: Path, update: int, config: dict, anchor: int):
    needed = [run_dir / "learner" / f"{update}.pth"]
    if not config.get("shared"):
        needed.append(run_dir / "critic" / f"{update}.pth")
    # The KL anchor: the learner at anchor_update, or the run's starting policy.
    needed.append(run_dir / "learner" / f"{anchor}.pth" if anchor else _workdir(run_dir, config) / config["policy_checkpoint"])
    return needed


def _optimizer_problem(run_dir: Path):
    optimizer = run_dir / "optimizer.pth"
    if not optimizer.exists():  # a run's first snapshot writes one; resume tolerates its absence
        return None
    import torch

    try:
        opt = torch.load(optimizer, map_location="cpu", weights_only=False)
        if "learner" not in opt:
            raise KeyError("learner")
    except Exception as e:  # noqa: BLE001 -- any failure to read it is the finding
        return f"{optimizer} is unreadable: {e!r}"
    return None


def check(run_dir: Path) -> int:
    state, update, config = _state(run_dir)
    missing = [str(p) for p in _needed(run_dir, update, config, int(state.get("anchor_update", 0))) if not p.is_file()]
    if missing:
        raise NotResumable(f"snapshot {update} is missing {', '.join(missing)}")
    problem = _optimizer_problem(run_dir)
    if problem:
        raise NotResumable(problem)
    return update


def _aside(path: Path) -> Path:
    target = path.with_name(f"{path.name}.bad-{time.strftime('%Y%m%dT%H%M%S')}")
    path.rename(target)
    return target


def repair(run_dir: Path) -> int:
    try:
        _state(run_dir)
    except NotResumable as e:
        print(e)
        config = json.loads((run_dir / "config.json").read_text())
        every = int(config.get("snapshot_every", 10))
        history = {}
        for line in (run_dir / "history.jsonl").read_text().splitlines():
            try:
                record = json.loads(line)
                history[int(record["update"])] = record  # the last line for an update wins (resumes repeat them)
            except (ValueError, KeyError, TypeError):
                continue
        candidates = sorted((u for u in history if u % every == 0), reverse=True)
        for update in candidates:
            record = history[update]
            anchor = int(record.get("anchor_update", 0))
            if all(p.is_file() for p in _needed(run_dir, update, config, anchor)):
                state = {
                    "config": config,
                    "snapshot_update": update,
                    "kl_coef": float(record["kl_coef"]),
                    "anchor_update": anchor,
                    "totals": {"games": int(record.get("games_total", 0))},
                }
                if (run_dir / "state.json").exists():
                    print(f"kept the unreadable state.json as {_aside(run_dir / 'state.json').name}")
                (run_dir / "state.json").write_text(json.dumps(state, indent=2))
                print(f"rebuilt state.json at snapshot {update} from config.json and history.jsonl")
                break
        else:
            raise NotResumable("no snapshot with its checkpoints and a history line to rebuild state.json from")
    problem = _optimizer_problem(run_dir)
    if problem:
        print(problem)
        print(f"moved it to {_aside(run_dir / 'optimizer.pth').name}: the resume starts the optimizer afresh")
    return check(run_dir)


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) not in (1, 2) or (len(args) == 2 and args[1] != "--repair"):
        raise SystemExit(__doc__)
    try:
        print((repair if len(args) == 2 else check)(Path(args[0])))
    except NotResumable as e:
        raise SystemExit(str(e))
