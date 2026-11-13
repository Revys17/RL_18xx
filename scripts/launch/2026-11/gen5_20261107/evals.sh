#!/bin/bash
# gen5 measurements (added 2026-11-10 at launch): every 100 updates, policy-only vs gen4 (both T 0.5), 4 players,
# 2v2 and 1v3 (the run's seat in 1v3 = one seat), 400 games each, every game from the auction under the real rules
# (stall 150, as the run). One results.txt line each. Ends with the run. Shares the GPU eval lock.
set -uo pipefail
G=/home/revys/longrun/experiments/gen5_20261107
D=/home/revys/longrun/deploy/az0-2p-7e21baa; PY=$D/.venv/bin/python
W=/home/revys/longrun/work/lr1; RUN=gen5_20261107
GEN4=$W/model_checkpoints_pg/rab_place_20261014/learner/1800.pth
cd $W
po() {  # name a_seats A B
  [ -f $G/policy_only/$1/summary.json ] && return 0
  ( flock 9
    OMP_NUM_THREADS=2 $PY $D/scripts/eval_policy_only.py --num-players 4 --a-seats $2 --match "$3" "$4" \
      --games 400 --workers 16 --max-decisions 1000 --auction-start --auction-stall-decisions 150 \
      --seed 0 --out $G/policy_only/$1 > $G/policy_only/$1.out 2>&1 ) 9>/home/revys/longrun/ops/gpu_eval.lock
  if [ -f $G/policy_only/$1/summary.json ]; then
    python3 /home/revys/longrun/experiments/pg2p_20261109/po_line.py $G/policy_only/$1 "policy-only $1" >> $G/results.txt
  else echo "$(date '+%F %T') policy-only $1: failed (policy_only/$1.out)" >> $G/results.txt; fi
}
mkdir -p $G/policy_only
po u0_2v2_vs_gen4 2 "$GEN4@0.5" "$GEN4@0.5"
while :; do
  u=$(ls $W/model_checkpoints_pg/$RUN/learner/ 2>/dev/null | sed 's/\.pth$//' | grep -E '^[0-9]+$' | sort -n | awk '$1 > 0 && $1 % 100 == 0' | tail -1)
  if [ -n "$u" ] && [ ! -f $G/policy_only/u${u}_1v3_vs_gen4/summary.json ]; then
    L=$W/model_checkpoints_pg/$RUN/learner/$u.pth
    po u${u}_2v2_vs_gen4 2 "$L@0.5" "$GEN4@0.5"
    po u${u}_1v3_vs_gen4 1 "$L@0.5" "$GEN4@0.5"
  else
    systemctl --user is-active -q rl18xx-gen5 || break
    sleep 300
  fi
done
echo "$(date '+%F %T') evals done" >> $G/results.txt
