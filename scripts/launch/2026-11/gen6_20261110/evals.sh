#!/bin/bash
# gen6 measurements (DESIGN.md): every 100 updates, policy-only at T 0.5 vs gen5's final checkpoint at each player
# count (2p 1v1 + vs pg2p u2000, 3p 1v2, 4p 2v2 + 1v3, 5p 2v3, 6p 3v3), 400 games each, every game from the
# auction under the real rules. Needs D and G5 as run.sh. Shares the GPU eval lock. Ends with the run.
set -uo pipefail
X=/home/revys/longrun/experiments/gen6_20261110
D=${D:?set D}; G5=${G5:?set G5}; PY=$D/.venv/bin/python
W=/home/revys/longrun/work/lr1; RUN=gen6_20261110; R=$W/model_checkpoints_pg/$RUN
GEN5=$W/model_checkpoints_pg/gen5_20261107/learner/$G5.pth
PG2P=$W/model_checkpoints_pg/pg2p_20261109/learner/2000.pth
cd $W
po() {  # name players a_seats A B
  [ -f $X/policy_only/$1/summary.json ] && return 0
  ( flock 9
    OMP_NUM_THREADS=2 $PY $D/scripts/eval_policy_only.py --num-players $2 --a-seats $3 --match "$4" "$5" \
      --games 400 --workers 16 --max-decisions 1500 --auction-start --auction-stall-decisions 150 \
      --seed 0 --out $X/policy_only/$1 > $X/policy_only/$1.out 2>&1 ) 9>/home/revys/longrun/ops/gpu_eval.lock
  if [ -f $X/policy_only/$1/summary.json ]; then
    python3 /home/revys/longrun/experiments/pg2p_20261109/po_line.py $X/policy_only/$1 "policy-only $1" >> $X/results.txt
  else echo "$(date '+%F %T') policy-only $1: failed (policy_only/$1.out)" >> $X/results.txt; fi
}
battery() {  # name-prefix checkpoint
  po $1_2p_1v1_vs_gen5 2 1 "$2@0.5" "$GEN5@0.5"
  po $1_2p_1v1_vs_pg2p 2 1 "$2@0.5" "$PG2P@0.5"
  po $1_3p_1v2_vs_gen5 3 1 "$2@0.5" "$GEN5@0.5"
  po $1_4p_2v2_vs_gen5 4 2 "$2@0.5" "$GEN5@0.5"
  po $1_4p_1v3_vs_gen5 4 1 "$2@0.5" "$GEN5@0.5"
  po $1_5p_2v3_vs_gen5 5 2 "$2@0.5" "$GEN5@0.5"
  po $1_6p_3v3_vs_gen5 6 3 "$2@0.5" "$GEN5@0.5"
}
mkdir -p $X/policy_only
battery u0 "$GEN5"   # the baseline: gen5 against itself
while :; do
  u=$(ls $R/learner/ 2>/dev/null | sed 's/\.pth$//' | grep -E '^[0-9]+$' | sort -n | awk '$1 > 0 && $1 % 100 == 0' | tail -1)
  if [ -n "$u" ] && [ ! -f $X/policy_only/u${u}_6p_3v3_vs_gen5/summary.json ]; then
    battery u$u $R/learner/$u.pth
  else
    systemctl --user is-active -q rl18xx-gen6 || break
    sleep 300
  fi
done
echo "$(date '+%F %T') evals done" >> $X/results.txt
