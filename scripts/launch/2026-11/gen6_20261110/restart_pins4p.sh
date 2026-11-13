#!/bin/bash
# Limits gen6's exploiter pins to 4-player games (the user, 2026-11-12, after the 2p eval vs pg2p sat at ~.24 from
# u1300 to u1700). Waits for gen6's next snapshot, stops unit rl18xx-gen6, adds "min_share_players": [4] to the
# pinned members of league_v8_gen5.json (backup kept), checks the per-count pins with the new code, and relaunches
# rl18xx-gen6 (run.sh, which resumes) on deployment gen6-96ba807 (min_share_players support). Unit rl18xx-gen6-restart2.
set -uo pipefail
X=/home/revys/longrun/experiments/gen6_20261110
W=/home/revys/longrun/work/lr1
R=$W/model_checkpoints_pg/gen6_20261110
POP=$W/model_checkpoints_pg/populations/league_v8_gen5.json
OLDD=/home/revys/longrun/deploy/gen6-76e51c4
NEWD=/home/revys/longrun/deploy/gen6-96ba807
MPS=(-E CUDA_MPS_PIPE_DIRECTORY=/home/revys/longrun/ops/mps/pipe -E CUDA_MPS_LOG_DIRECTORY=/home/revys/longrun/ops/mps/log)
log() { echo "$(date '+%F %T') $*" >> $X/results.txt; }
snap() { python3 -c "import json; print(json.load(open('$R/state.json'))['snapshot_update'])"; }

bash $NEWD/scripts/longrun/deploy.sh verify $NEWD > /dev/null 2>&1 || { log "restart2: $NEWD does not verify; nothing changed"; exit 1; }
start=$(snap)
log "restart2: waiting for the snapshot after u$start to limit the exploiter pins to 4p ($NEWD)"
while [ "$(snap)" = "$start" ]; do
  systemctl --user is-active -q rl18xx-gen6 || { log "restart2: rl18xx-gen6 not active; nothing changed"; exit 1; }
  sleep 5
done
sleep 2
systemctl --user stop rl18xx-gen6
systemctl --user reset-failed rl18xx-gen6 2>/dev/null
BACKUP=${POP%.json}.pre-pins4p-$(date +%Y%m%d%H%M%S).json
cp -p $POP $BACKUP
D=$NEWD
if ! python3 - "$POP" <<'PY'
import json, sys
p = sys.argv[1]; d = json.load(open(p))
for m in d["members"]:
    if isinstance(m, dict) and m.get("min_share"):
        m["min_share_players"] = [4]
open(p + ".tmp", "w").write(json.dumps(d, indent=2) + "\n")
import os; os.replace(p + ".tmp", p)
PY
then log "restart2: editing $POP failed; relaunching unchanged"; cp -p $BACKUP $POP; D=$OLDD
elif ! (cd $W && PYTHONPATH=$NEWD $NEWD/.venv/bin/python - >> $X/restart_pins4p.log 2>&1 <<'PY'
import json
from rl18xx.agent.alphazero import policy_gradient as pg
s = json.load(open("model_checkpoints_pg/gen6_20261110/state.json"))
cfg = pg.PGConfig(**s["config"])
lg = pg.league_for_run(cfg, s["league"])
for n in lg.counts:
    l = lg.league(n); w = dict(zip(l.members, l.weights()))
    ex = sum(v for m, v in w.items() if "/gx" in m); pg2p = sum(v for m, v in w.items() if "pg2p" in m)
    print(f"{n}p pins {len(l.min_shares)} exploiters {ex:.3f} pg2p {pg2p:.3f}")
    assert (len(l.min_shares) == 5) == (n == 4), n
PY
); then log "restart2: pin check failed (restart_pins4p.log); population restored, relaunching unchanged"; cp -p $BACKUP $POP; D=$OLDD
fi
systemd-run --user --unit rl18xx-gen6 -p MemoryHigh=110G -p MemoryMax=130G -p Restart=on-failure -p RestartSec=120 \
  "${MPS[@]}" -E D=$D -E G5=2800 /usr/bin/bash $X/run.sh
log "restart2: rl18xx-gen6 relaunched from u$(snap) on $D (population backup $BACKUP)"
