#!/bin/bash
# Out-of-family panel (2026-11-11): gen5 beats its own older checkpoints by more every hundred, while its scores
# against gen4 drift down (u2100 .877 -> u2700 .786 at 2v2). Is gen5 getting worse against styles outside its own
# family? Each checkpoint ALONE against three copies of an outside opponent (the bot's real situation: one seat
# among others), 800 games, the same settings as evals2.sh. Shares the GPU eval lock.
#   outfamily.sh <U>...      (opponents fixed below)
set -uo pipefail
G=/home/revys/longrun/experiments/gen5_20261107
D=/home/revys/longrun/deploy/az0-2p-7e21baa; PY=$D/.venv/bin/python
W=/home/revys/longrun/work/lr1; R=$W/model_checkpoints_pg/gen5_20261107
cd $W; mkdir -p $G/outfamily
declare -A OPP=(
  [gen4]=model_checkpoints_pg/rab_place_20261014/learner/1800.pth
  [sl10]=model_checkpoints/AlphaZeroTransformer/20261004_134558_804974562/10.pth
  [gen2]=model_checkpoints_pg/gen1_20261009/learner/1500.pth
  [gx2rust]=model_checkpoints_pg/gx2_rust_20261013/learner/300.pth
  [gx2bankrupt]=model_checkpoints_pg/gx2_bankrupt_20261013/learner/300.pth
)
po() {  # name A B
  [ -f $G/outfamily/$1/summary.json ] && return 0
  ( flock 9
    OMP_NUM_THREADS=2 $PY $D/scripts/eval_policy_only.py --num-players 4 --a-seats 1 --match "$2" "$3" \
      --games 800 --workers 16 --max-decisions 1000 --auction-start --auction-stall-decisions 150 \
      --seed 0 --out $G/outfamily/$1 > $G/outfamily/$1.out 2>&1 ) 9>/home/revys/longrun/ops/gpu_eval.lock
  python3 /home/revys/longrun/experiments/pg2p_20261109/po_line.py $G/outfamily/$1 "policy-only $1" >> $G/outfamily.txt
}
for o in gen4 sl10 gen2 gx2rust gx2bankrupt; do
  for u in "$@"; do
    po u${u}_1v3_vs_$o "$R/learner/$u.pth@0.5" "${OPP[$o]}@0.5"
  done
done
