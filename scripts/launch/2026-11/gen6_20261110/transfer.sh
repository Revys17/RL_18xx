#!/bin/bash
# gen6 prep (2026-11-10): how gen5 (4-player trained) already plays at 2, 3, 5 and 6 players, vs gen4, policy-only at
# T 0.5, every game from the auction under the real rules; the stall limit (300) and decision cap (2000) are raised to
# measure the auction's and the game's length at each count (gen6's per-count limits). 400 games each, 40 kept.
set -uo pipefail
X=/home/revys/longrun/experiments/gen6_20261110
D=/home/revys/longrun/deploy/az0-2p-7e21baa; PY=$D/.venv/bin/python
W=/home/revys/longrun/work/lr1
GEN4=$W/model_checkpoints_pg/rab_place_20261014/learner/1800.pth
G5=$W/model_checkpoints_pg/gen5_20261107/learner/$1.pth
cd $W
po() {  # name players a_seats A B
  [ -f $X/transfer/$1/summary.json ] && return 0
  ( flock 9
    OMP_NUM_THREADS=2 $PY $D/scripts/eval_policy_only.py --num-players $2 --a-seats $3 --match "$4" "$5" \
      --games 400 --workers 16 --max-decisions 2000 --auction-start --auction-stall-decisions 300 \
      --seed 0 --out $X/transfer/$1 --save-games 400 > $X/transfer/$1.out 2>&1 ) 9>/home/revys/longrun/ops/gpu_eval.lock
  if [ -f $X/transfer/$1/summary.json ]; then
    python3 /home/revys/longrun/experiments/pg2p_20261109/po_line.py $X/transfer/$1 "policy-only $1" >> $X/results.txt
  else echo "$(date '+%F %T') policy-only $1: failed (transfer/$1.out)" >> $X/results.txt; fi
}
echo "$(date '+%F %T') transfer check: gen5 u$1 vs gen4 at 2/3/5/6 players" >> $X/results.txt
po g5u$1_2p_1v1 2 1 "$G5@0.5" "$GEN4@0.5"
po g5u$1_3p_1v2 3 1 "$G5@0.5" "$GEN4@0.5"
po g5u$1_5p_2v3 5 2 "$G5@0.5" "$GEN4@0.5"
po g5u$1_6p_3v3 6 3 "$G5@0.5" "$GEN4@0.5"
po g5u$1_4p_2v2 4 2 "$G5@0.5" "$GEN4@0.5"
echo "$(date '+%F %T') transfer check done" >> $X/results.txt
