#!/bin/bash
# Moves gen6 onto per-count PFSP (the user, 2026-11-12: "pick opponents based on their strength at each level").
# Waits for gen6's next snapshot (state.json's snapshot_update advances), stops unit rl18xx-gen6, seeds the
# per-count league rates from the saved training games (seed_count_league.py --write; the pooled state.json is
# kept as state.pooled-<time>.json), and relaunches rl18xx-gen6 (run.sh, which resumes) on deployment
# gen6-76e51c4 with the same unit properties. Run as unit rl18xx-gen6-restart.
set -uo pipefail
X=/home/revys/longrun/experiments/gen6_20261110
W=/home/revys/longrun/work/lr1
R=$W/model_checkpoints_pg/gen6_20261110
NEWD=/home/revys/longrun/deploy/gen6-76e51c4
MPS=(-E CUDA_MPS_PIPE_DIRECTORY=/home/revys/longrun/ops/mps/pipe -E CUDA_MPS_LOG_DIRECTORY=/home/revys/longrun/ops/mps/log)
log() { echo "$(date '+%F %T') $*" >> $X/results.txt; }
snap() { python3 -c "import json; print(json.load(open('$R/state.json'))['snapshot_update'])"; }

bash $NEWD/scripts/longrun/deploy.sh verify $NEWD > /dev/null 2>&1 || { log "restart: $NEWD does not verify; nothing changed"; exit 1; }
start=$(snap)
log "restart: waiting for the snapshot after u$start to move gen6 onto per-count PFSP ($NEWD)"
while [ "$(snap)" = "$start" ]; do
  systemctl --user is-active -q rl18xx-gen6 || { log "restart: rl18xx-gen6 not active; nothing changed"; exit 1; }
  sleep 5
done
sleep 2
systemctl --user stop rl18xx-gen6
systemctl --user reset-failed rl18xx-gen6 2>/dev/null
cd $W
if ! PYTHONPATH=$NEWD $NEWD/.venv/bin/python $X/seed_count_league.py model_checkpoints_pg/gen6_20261110 --since-update 270 --write >> $X/seed_count_league.log 2>&1; then
  log "restart: seeding failed (seed_count_league.log); relaunching on the old deployment unchanged"
  NEWD=/home/revys/longrun/deploy/gen6-5d0ac8e
fi
systemd-run --user --unit rl18xx-gen6 -p MemoryHigh=110G -p MemoryMax=130G -p Restart=on-failure -p RestartSec=120 \
  "${MPS[@]}" -E D=$NEWD -E G5=2800 /usr/bin/bash $X/run.sh
log "restart: rl18xx-gen6 relaunched from u$(snap) on $NEWD"
