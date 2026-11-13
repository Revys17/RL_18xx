#!/bin/bash
# One-off transitivity check (2026-11-11): does u-200 chaining overstate gen5's progress? The latest checkpoint against
# fixed older ones (u1000, u1600), 800 games each, the same settings as evals2.sh. Shares the GPU eval lock.
set -uo pipefail
G=/home/revys/longrun/experiments/gen5_20261107
D=/home/revys/longrun/deploy/az0-2p-7e21baa; PY=$D/.venv/bin/python
W=/home/revys/longrun/work/lr1; R=$W/model_checkpoints_pg/gen5_20261107
cd $W; mkdir -p $G/transitivity
GEN4=$W/model_checkpoints_pg/rab_place_20261014/learner/1800.pth
SEED=${SEED:-0}
po() {  # name a_seats A B games
  [ -f $G/transitivity/$1/summary.json ] && return 0
  ( flock 9
    OMP_NUM_THREADS=2 $PY $D/scripts/eval_policy_only.py --num-players 4 --a-seats $2 --match "$3" "$4" \
      --games $5 --workers 16 --max-decisions 1000 --auction-start --auction-stall-decisions 150 \
      --seed $SEED --out $G/transitivity/$1 > $G/transitivity/$1.out 2>&1 ) 9>/home/revys/longrun/ops/gpu_eval.lock
  python3 /home/revys/longrun/experiments/pg2p_20261109/po_line.py $G/transitivity/$1 "policy-only $1" >> $G/transitivity.txt
}
U=${1:?usage: transitivity.sh <U> <P>...}; shift
for p in "$@"; do
  if [ $p = gen4 ]; then  # a rerun of evals2's gen4 match with another seed (SEED=1 ...)
    po u${U}_2v2_vs_gen4_s$SEED 2 "$R/learner/$U.pth@0.5" "$GEN4@0.5" 400
    po u${U}_1v3_vs_gen4_s$SEED 1 "$R/learner/$U.pth@0.5" "$GEN4@0.5" 400
    continue
  fi
  po u${U}_2v2_vs_u$p 2 "$R/learner/$U.pth@0.5" "$R/learner/$p.pth@0.5" 800
  po u${U}_1v3_vs_u$p 1 "$R/learner/$U.pth@0.5" "$R/learner/$p.pth@0.5" 800
done
