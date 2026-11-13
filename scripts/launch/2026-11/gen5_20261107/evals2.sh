#!/bin/bash
# gen5 measurements, from 2026-11-10 ~15:00 (replaces evals.sh; the user: keep going until it plateaus). Every 100
# updates: vs gen4 2v2 + 1v3 (400 games, as before) and vs the checkpoint 200 updates older, 2v2 + 1v3 (800 games),
# backfilled from u300. plateau.py's rule touches the run's STOP file. When the run stops at its old 1000 cap, this
# starts it again (run.sh's resume runs to 4000), once. Ends when the run has ended for good. Shares the GPU eval lock.
set -uo pipefail
G=/home/revys/longrun/experiments/gen5_20261107
D=/home/revys/longrun/deploy/az0-2p-7e21baa; PY=$D/.venv/bin/python
W=/home/revys/longrun/work/lr1; RUN=gen5_20261107; R=$W/model_checkpoints_pg/$RUN
GEN4=$W/model_checkpoints_pg/rab_place_20261014/learner/1800.pth
cd $W
log() { echo "$(date '+%F %T') $*" >> $G/results.txt; }
po() {  # name a_seats A B games
  [ -f $G/policy_only/$1/summary.json ] && return 0
  ( flock 9
    OMP_NUM_THREADS=2 $PY $D/scripts/eval_policy_only.py --num-players 4 --a-seats $2 --match "$3" "$4" \
      --games $5 --workers 16 --max-decisions 1000 --auction-start --auction-stall-decisions 150 \
      --seed 0 --out $G/policy_only/$1 > $G/policy_only/$1.out 2>&1 ) 9>/home/revys/longrun/ops/gpu_eval.lock
  if [ -f $G/policy_only/$1/summary.json ]; then
    python3 /home/revys/longrun/experiments/pg2p_20261109/po_line.py $G/policy_only/$1 "policy-only $1" >> $G/results.txt
  else log "policy-only $1: failed (policy_only/$1.out)"; fi
}
hundreds() { ls $R/learner/ 2>/dev/null | sed 's/\.pth$//' | grep -E '^[0-9]+$' | sort -n | awk '$1 > 0 && $1 % 100 == 0'; }
mkdir -p $G/policy_only
log "evals2: vs gen4 + vs u-200 every 100 updates; plateau rule (plateau.py); restart at 1000 once"
while :; do
  for u in $(hundreds | sort -rn); do   # newest first, then the backfill
    L=$R/learner/$u.pth
    po u${u}_2v2_vs_gen4 2 "$L@0.5" "$GEN4@0.5" 400
    po u${u}_1v3_vs_gen4 1 "$L@0.5" "$GEN4@0.5" 400
    p=$((u - 200))
    if [ $p -ge 100 ] && [ -f $R/learner/$p.pth ]; then
      po u${u}_2v2_vs_u$p 2 "$L@0.5" "$R/learner/$p.pth@0.5" 800
      po u${u}_1v3_vs_u$p 1 "$L@0.5" "$R/learner/$p.pth@0.5" 800
    fi
    break   # one hundred per pass, so the newest is never behind the backfill
  done
  for u in $(hundreds); do  # backfill, oldest first
    p=$((u - 200)); [ $p -ge 100 ] && [ -f $R/learner/$p.pth ] || continue
    if [ ! -f $G/policy_only/u${u}_1v3_vs_u$p/summary.json ]; then
      po u${u}_2v2_vs_u$p 2 "$R/learner/$u.pth@0.5" "$R/learner/$p.pth@0.5" 800
      po u${u}_1v3_vs_u$p 1 "$R/learner/$u.pth@0.5" "$R/learner/$p.pth@0.5" 800
      continue 2   # back to the top: a newer hundred may have arrived
    fi
  done
  v=$(python3 $G/plateau.py $G/results.txt | tail -1)
  if [[ $v == STOP* ]] && [ ! -f $R/STOP ]; then
    touch $R/STOP; log "plateau rule: two flat hundreds in a row (${v#STOP }) -> STOP file; the run ends after its current update"
  fi
  state=$(systemctl --user show -p ActiveState --value rl18xx-gen5)
  snap=$(python3 -c "import json; print(json.load(open('$R/state.json'))['snapshot_update'])")
  if [ "$state" = inactive ] && [ "$snap" = 1000 ] && [ ! -f $R/STOP ] && [ ! -f $G/restarted_at_1000 ]; then
    touch $G/restarted_at_1000; log "gen5: reached its 1000 cap; started again (resume to 4000 / the plateau rule)"
    systemctl --user start rl18xx-gen5; sleep 60; continue
  fi
  if [ "$state" = failed ]; then log "evals2: rl18xx-gen5 failed; stopping"; break; fi
  if [ "$state" = inactive ] && [ "$(hundreds | tail -1)" = "$(ls $G/policy_only | grep -oE '^u[0-9]+_1v3_vs_gen4$' | tr -d u | cut -d_ -f1 | sort -n | tail -1)" ]; then
    log "evals2: the run has ended (snapshot u$snap)"; break
  fi
  sleep 120
done
