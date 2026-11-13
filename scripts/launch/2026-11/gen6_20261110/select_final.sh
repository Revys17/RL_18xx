#!/bin/bash
# Picks gen6's final checkpoint (2026-11-12; gen6 stopped at u2431 by the plateau rule): u2000 / u2200 / u2400 on a
# FRESH set of games (seed 1) of the outside panel, pooled later with the seed-0 runs: 4p 1v3 vs gen4 / SL ck10 /
# gen2 / gx2_rust / gx2_bankrupt at 1600 games, 2p/3p/5p/6p alone vs gen4 at 800, 2p vs pg2p at 800. Policy-only at
# T 0.5 from the auction, as outfamily.sh. Matches interleaved across candidates. Lines -> select.txt.
set -uo pipefail
X=/home/revys/longrun/experiments/gen6_20261110
D=/home/revys/longrun/deploy/gen6-96ba807; PY=$D/.venv/bin/python
W=/home/revys/longrun/work/lr1; R=$W/model_checkpoints_pg/gen6_20261110
declare -A OPP=(
  [gen4]=model_checkpoints_pg/rab_place_20261014/learner/1800.pth
  [sl10]=model_checkpoints/AlphaZeroTransformer/20261004_134558_804974562/10.pth
  [gen2]=model_checkpoints_pg/gen1_20261009/learner/1500.pth
  [gx2rust]=model_checkpoints_pg/gx2_rust_20261013/learner/300.pth
  [gx2bankrupt]=model_checkpoints_pg/gx2_bankrupt_20261013/learner/300.pth
  [pg2p]=model_checkpoints_pg/pg2p_20261109/learner/2000.pth
)
cd $W; mkdir -p $X/select
po() {  # name players games max_decisions A B
  [ -f $X/select/$1/summary.json ] && return 0
  ( flock 9
    OMP_NUM_THREADS=2 $PY $D/scripts/eval_policy_only.py --num-players $2 --a-seats 1 --match "$5" "$6" \
      --games $3 --workers 32 --max-decisions $4 --auction-start --auction-stall-decisions 150 \
      --seed 1 --out $X/select/$1 > $X/select/$1.out 2>&1 ) 9>/home/revys/longrun/ops/gpu_eval.lock
  if [ -f $X/select/$1/summary.json ]; then
    python3 /home/revys/longrun/experiments/pg2p_20261109/po_line.py $X/select/$1 "policy-only $1" >> $X/select.txt
  else echo "$(date '+%F %T') policy-only $1: failed (select/$1.out)" >> $X/select.txt; fi
}
for o in gen4 sl10 gen2 gx2rust gx2bankrupt; do
  for u in 2000 2200 2400; do po u${u}_4p_1v3_vs_$o 4 1600 1000 "$R/learner/$u.pth@0.5" "$W/${OPP[$o]}@0.5"; done
done
for n in 2 3 5 6; do
  for u in 2000 2200 2400; do po u${u}_${n}p_1v$((n - 1))_vs_gen4 $n 800 1500 "$R/learner/$u.pth@0.5" "$W/${OPP[gen4]}@0.5"; done
done
for u in 2000 2200 2400; do po u${u}_2p_1v1_vs_pg2p 2 800 1500 "$R/learner/$u.pth@0.5" "$W/${OPP[pg2p]}@0.5"; done
echo "$(date '+%F %T') select done" >> $X/select.txt
