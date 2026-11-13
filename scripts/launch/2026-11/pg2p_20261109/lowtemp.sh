#!/bin/bash
# Is the ~0.9 ceiling vs older selves sampling noise in close games? (2026-11-09 20:00) The same matches at lower
# temperature for both sides (policy-only, auction starts, as evals.sh). Results in lowtemp.txt, not results.txt
# (keeps them out of the Elo fit and the stop rules).
set -uo pipefail
source /home/revys/longrun/experiments/pg2p_20261109/common.sh
cd $W
m() {  # name A B
  n=$1; [ -f $X/policy_only/$n/summary.json ] || ( flock 9
    OMP_NUM_THREADS=2 $PY $D/scripts/eval_policy_only.py --num-players $NP --a-seats 1 --match "$2" "$3" \
      --games 400 --workers 16 --max-decisions $MAXDEC --auction-start --auction-stall-decisions $STALL \
      --seed 0 --out $X/policy_only/$n > $X/policy_only/$n.out 2>&1 ) 9>/home/revys/longrun/ops/gpu_eval.lock
  [ -f $X/policy_only/$n/summary.json ] && python3 $X/po_line.py $X/policy_only/$n "policy-only $n" >> $X/lowtemp.txt
}
for t in 0.2 0.05; do
  for b in 700 600 1200; do m lowT${t}_u1300_vs_u$b "$(ckpt 1300)@$t" "$(ckpt $b)@$t"; done
done
