"""Is a search-learning run directory's snapshot resumable? (launch.sh, before every --resume.)

    snapshot_check.py <run dir>

Prints the snapshot's update and exits 0, or prints what is wrong and exits 1. ``search_learning.run``
writes the checkpoints atomically but ``optimizer.pth`` and ``state.json`` in place, so a kill in the
middle of a snapshot (an OOM kill, a power cut, the 5-s user-manager timeout at shutdown) can leave either
unreadable; this catches that before a resume crashes on it, again and again.
"""

import json
import sys
from pathlib import Path


def check(run_dir: Path) -> int:
    state_path = run_dir / "state.json"
    try:
        state = json.loads(state_path.read_text())
        update = int(state["snapshot_update"])
        config = state["config"]
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise SystemExit(f"{state_path} is unreadable: {e!r}")
    needed = [run_dir / "learner" / f"{update}.pth"]
    if not config.get("shared"):
        needed.append(run_dir / "critic" / f"{update}.pth")
    anchor = int(state.get("anchor_update", 0))
    # The KL anchor: the learner at anchor_update, or the run's starting policy.
    needed.append(run_dir / "learner" / f"{anchor}.pth" if anchor else Path(config["policy_checkpoint"]))
    missing = [str(p) for p in needed if not p.is_file()]
    if missing:
        raise SystemExit(f"snapshot {update} is missing {', '.join(missing)}")
    optimizer = run_dir / "optimizer.pth"
    if optimizer.exists():  # a run's first snapshot always writes one; resume tolerates its absence
        import torch

        try:
            opt = torch.load(optimizer, map_location="cpu", weights_only=False)
            if "learner" not in opt:
                raise KeyError("learner")
        except Exception as e:  # noqa: BLE001 -- any failure to read it is the finding
            raise SystemExit(f"{optimizer} is unreadable: {e!r}")
    return update


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    print(check(Path(sys.argv[1])))
