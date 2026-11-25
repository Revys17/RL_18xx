"""Set up the run that gives A4 u1528 a critic trained on every seat's decisions (2026-11-25, the user: "Yes, go
ahead"; docs/robustness_measurement.md "Why search stops the bots contesting").

A copy of A4's resume state with u1528's learner and critic, resumed with the policy frozen (learner lr 0 in the
config and in the copied optimizer state: a resume restores the optimizer's own lr) and ``critic_seats`` "all". The
league, the opponent pool, the A2 start priorities and every game setting are A4's at u1520 (its last snapshot);
the league and pool stop growing (``league_snapshot_every`` / ``pool_refresh_every`` out of reach), since a frozen
learner's snapshots would only add copies of u1528. Only the critic changes.

Run from the main checkout (data paths are relative to it): ``python prepare.py`` (idempotent: refuses an existing
run directory).
"""

import json
import shutil
import sys
from pathlib import Path

import torch

SRC = Path("model_checkpoints_generic_pg/robust_A4_20261118")
RUN = "u1528_critic_all_20261125"
DST = Path("model_checkpoints_generic_pg") / RUN
START = 1528
FAR = 10**9

if DST.exists():
    sys.exit(f"{DST} exists")
state = json.loads((SRC / "state.json").read_text())
cfg = state["config"]
cfg.update(
    run_name=RUN,
    lr=0.0,
    critic_seats="all",
    league_snapshot_every=FAR,
    pool_refresh_every=FAR,
    max_updates=START + 200,
)
state["snapshot_update"] = START
(DST / "learner").mkdir(parents=True)
(DST / "critic").mkdir()
for part in ("learner", "critic"):
    shutil.copy2(SRC / part / f"{START}.pth", DST / part / f"{START}.pth")
anchor = int(state.get("anchor_update", 0))
if anchor:
    shutil.copy2(SRC / "learner" / f"{anchor}.pth", DST / "learner" / f"{anchor}.pth")
if state.get("start_priorities"):
    shutil.copy2(SRC / state["start_priorities"], DST / state["start_priorities"])
opt = torch.load(SRC / "optimizer.pth", map_location="cpu")
for group in opt["learner"]["param_groups"]:
    group["lr"] = 0.0
torch.save(opt, DST / "optimizer.pth")
(DST / "state.json").write_text(json.dumps(state, indent=2))
(DST / "config.json").write_text(json.dumps(cfg, indent=2))
print(f"{DST}: from u{START}, learner frozen, critic on every seat, to u{cfg['max_updates']}")
