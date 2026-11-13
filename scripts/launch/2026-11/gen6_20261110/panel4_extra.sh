#!/bin/bash
# Extra 4p 1v3 out-of-family panel for one gen6 checkpoint (2026-11-12): same settings, lock and output as
# outfamily.sh's panel4. Used to place the u400 -> u600 drop (.619 -> .593) before or after the per-count PFSP
# restart at u490.   panel4_extra.sh <update>
set -uo pipefail
X=/home/revys/longrun/experiments/gen6_20261110
D=/home/revys/longrun/deploy/gen6-5d0ac8e; PY=$D/.venv/bin/python
W=/home/revys/longrun/work/lr1; R=$W/model_checkpoints_pg/gen6_20261110; U=$1
declare -A OPP=(
  [gen4]=model_checkpoints_pg/rab_place_20261014/learner/1800.pth
  [sl10]=model_checkpoints/AlphaZeroTransformer/20261004_134558_804974562/10.pth
  [gen2]=model_checkpoints_pg/gen1_20261009/learner/1500.pth
  [gx2rust]=model_checkpoints_pg/gx2_rust_20261013/learner/300.pth
  [gx2bankrupt]=model_checkpoints_pg/gx2_bankrupt_20261013/learner/300.pth
)
cd $W
for o in gen4 sl10 gen2 gx2rust gx2bankrupt; do
  n=u${U}_4p_1v3_vs_$o
  [ -f $X/outfamily/$n/summary.json ] && continue
  ( flock 9
    OMP_NUM_THREADS=2 $PY $D/scripts/eval_policy_only.py --num-players 4 --a-seats 1 --match "$R/learner/$U.pth@0.5" "$W/${OPP[$o]}@0.5" \
      --games 800 --workers 16 --max-decisions 1000 --auction-start --auction-stall-decisions 150 \
      --seed 0 --out $X/outfamily/$n > $X/outfamily/$n.out 2>&1 ) 9>/home/revys/longrun/ops/gpu_eval.lock
  if [ -f $X/outfamily/$n/summary.json ]; then
    python3 /home/revys/longrun/experiments/pg2p_20261109/po_line.py $X/outfamily/$n "policy-only $n" >> $X/outfamily.txt
  else echo "$(date '+%F %T') policy-only $n: failed (outfamily/$n.out)" >> $X/outfamily.txt; fi
done
