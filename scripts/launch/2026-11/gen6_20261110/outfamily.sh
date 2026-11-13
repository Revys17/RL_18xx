#!/bin/bash
# gen6's out-of-family panel (2026-11-11; gen5 lesson: matches inside the family kept "gaining" while one seat against
# outside styles was flat for 1,200 updates). Every 200 updates, the checkpoint ALONE against copies of outside
# opponents, policy-only at T 0.5 from the auction:
#   4p 1v3 vs gen4 / SL ck10 / gen2 / gx2_rust / gx2_bankrupt, 800 games, max 1000 decisions (= gen5's panel, so
#      gen5 u2800's mean .593 is the baseline: gen5_20261107/outfamily.txt)
#   2p 1v1, 3p 1v2, 5p 1v4, 6p 1v5 vs gen4, 800 games, max 1500 decisions (baseline: u0 = gen5 u2800, run first)
# Lines go to outfamily.txt. Shares the GPU eval lock. Ends when rl18xx-gen6 has ended and its last 200 is done.
set -uo pipefail
X=/home/revys/longrun/experiments/gen6_20261110
D=/home/revys/longrun/deploy/gen6-5d0ac8e; PY=$D/.venv/bin/python
W=/home/revys/longrun/work/lr1; R=$W/model_checkpoints_pg/gen6_20261110
GEN5=$W/model_checkpoints_pg/gen5_20261107/learner/2800.pth
declare -A OPP=(
  [gen4]=model_checkpoints_pg/rab_place_20261014/learner/1800.pth
  [sl10]=model_checkpoints/AlphaZeroTransformer/20261004_134558_804974562/10.pth
  [gen2]=model_checkpoints_pg/gen1_20261009/learner/1500.pth
  [gx2rust]=model_checkpoints_pg/gx2_rust_20261013/learner/300.pth
  [gx2bankrupt]=model_checkpoints_pg/gx2_bankrupt_20261013/learner/300.pth
)
cd $W; mkdir -p $X/outfamily
po() {  # name players max_decisions A B
  [ -f $X/outfamily/$1/summary.json ] && return 0
  ( flock 9
    OMP_NUM_THREADS=2 $PY $D/scripts/eval_policy_only.py --num-players $2 --a-seats 1 --match "$4" "$5" \
      --games 800 --workers 16 --max-decisions $3 --auction-start --auction-stall-decisions 150 \
      --seed 0 --out $X/outfamily/$1 > $X/outfamily/$1.out 2>&1 ) 9>/home/revys/longrun/ops/gpu_eval.lock
  if [ -f $X/outfamily/$1/summary.json ]; then
    python3 /home/revys/longrun/experiments/pg2p_20261109/po_line.py $X/outfamily/$1 "policy-only $1" >> $X/outfamily.txt
  else echo "$(date '+%F %T') policy-only $1: failed (outfamily/$1.out)" >> $X/outfamily.txt; fi
}
counts() {  # prefix checkpoint
  for n in 2 3 5 6; do po $1_${n}p_1v$((n - 1))_vs_gen4 $n 1500 "$2@0.5" "$W/${OPP[gen4]}@0.5"; done
}
panel4() {  # prefix checkpoint
  for o in gen4 sl10 gen2 gx2rust gx2bankrupt; do po $1_4p_1v3_vs_$o 4 1000 "$2@0.5" "$W/${OPP[$o]}@0.5"; done
}
counts u0 "$GEN5"
while :; do
  u=$(ls $R/learner/ 2>/dev/null | sed 's/\.pth$//' | grep -E '^[0-9]+$' | sort -n | awk '$1 > 0 && $1 % 200 == 0' | tail -1)
  if [ -n "$u" ] && [ ! -f $X/outfamily/u${u}_6p_1v5_vs_gen4/summary.json ]; then
    panel4 u$u $R/learner/$u.pth
    counts u$u $R/learner/$u.pth
  else
    systemctl --user is-active -q rl18xx-gen6 || break
    sleep 300
  fi
done
echo "$(date '+%F %T') outfamily done" >> $X/outfamily.txt
