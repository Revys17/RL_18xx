#!/bin/bash
# Step 2 of the exploiter plan agreed with the user (2026-11-12): no pins at any count (exploiters become plain PFSP
# members) and the 8 round-2 president:<corp> exploiters retired. Waits for gen6's next snapshot, stops rl18xx-gen6,
# removes min_share / min_share_players from league_v8_gen5.json and the 8 members from the file AND from state.json's
# per-count league stats (a resume keeps members the file dropped), checks with the deployment's code, and relaunches
# on the same deployment (gen6-96ba807). Backups of both files kept. Unit rl18xx-gen6-restart3.
set -uo pipefail
X=/home/revys/longrun/experiments/gen6_20261110
W=/home/revys/longrun/work/lr1
R=$W/model_checkpoints_pg/gen6_20261110
POP=$W/model_checkpoints_pg/populations/league_v8_gen5.json
D=/home/revys/longrun/deploy/gen6-96ba807
MPS=(-E CUDA_MPS_PIPE_DIRECTORY=/home/revys/longrun/ops/mps/pipe -E CUDA_MPS_LOG_DIRECTORY=/home/revys/longrun/ops/mps/log)
log() { echo "$(date '+%F %T') $*" >> $X/results.txt; }
snap() { python3 -c "import json; print(json.load(open('$R/state.json'))['snapshot_update'])"; }

start=$(snap)
log "restart3: waiting for the snapshot after u$start to unpin the exploiters and retire the president exploiters"
while [ "$(snap)" = "$start" ]; do
  systemctl --user is-active -q rl18xx-gen6 || { log "restart3: rl18xx-gen6 not active; nothing changed"; exit 1; }
  sleep 5
done
sleep 2
systemctl --user stop rl18xx-gen6
systemctl --user reset-failed rl18xx-gen6 2>/dev/null
T=$(date +%Y%m%d%H%M%S)
cp -p $POP ${POP%.json}.pre-unpin-$T.json
cp -p $R/state.json $R/state.pre-unpin-$T.json
if ! python3 $X/unpin_edit.py $POP $R/state.json >> $X/restart_unpin.log 2>&1 \
   || ! (cd $W && PYTHONPATH=$D $D/.venv/bin/python $X/unpin_check.py >> $X/restart_unpin.log 2>&1); then
  log "restart3: edit or check failed (restart_unpin.log); both files restored, relaunching unchanged"
  cp -p ${POP%.json}.pre-unpin-$T.json $POP; cp -p $R/state.pre-unpin-$T.json $R/state.json
fi
systemd-run --user --unit rl18xx-gen6 -p MemoryHigh=110G -p MemoryMax=130G -p Restart=on-failure -p RestartSec=120 \
  "${MPS[@]}" -E D=$D -E G5=2800 /usr/bin/bash $X/run.sh
log "restart3: rl18xx-gen6 relaunched from u$(snap) on $D (backups *.pre-unpin-$T.json)"
