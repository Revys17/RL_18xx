#!/bin/bash
# Measurements (DESIGN.md), policy-only, 2 players, 1v1 in both seatings, 400 games each, the run's own start mix
# is NOT used: every game from the auction under the real rules (as az0's, so the numbers compare).
# u0 = checkpoint 10 at the learner's temperature 0.5. Every 50 updates (newest first if behind): u vs checkpoint 10
# (T 0.7, as az0's bar), vs gen4 (its own 0.5), vs u-100. One results.txt line each.
set -uo pipefail
source /home/revys/longrun/experiments/pg2p_20261109/common.sh
cd $W
po() {  # name A B
  [ -f $X/policy_only/$1/summary.json ] && return 0
  ( flock 9
    OMP_NUM_THREADS=2 $PY $D/scripts/eval_policy_only.py --num-players $NP --a-seats 1 --match "$2" "$3" \
      --games 400 --workers 16 --max-decisions $MAXDEC --auction-start --auction-stall-decisions $STALL \
      --seed 0 --out $X/policy_only/$1 > $X/policy_only/$1.out 2>&1 ) 9>/home/revys/longrun/ops/gpu_eval.lock
  if [ -f $X/policy_only/$1/summary.json ]; then python3 $X/po_line.py $X/policy_only/$1 "policy-only $1" >> $X/results.txt
  else log "policy-only $1: failed (policy_only/$1.out)"; fi
}
po u0_vs_ck10 "$CK10@0.5" "$CK10@0.7"
po u0_vs_gen4 "$CK10@0.5" "$GEN4@0.5"
while :; do
  u=$(ls $W/model_checkpoints_pg/$RUN/learner/ 2>/dev/null | sed 's/\.pth$//' | grep -E '^[0-9]+$' | sort -n | awk '$1 > 0 && $1 % 50 == 0' | tail -1)
  if [ -n "$u" ] && [ ! -f $X/policy_only/u${u}_vs_ck10/summary.json ]; then
    back=$((u - 100)); [ -f "$(ckpt $back)" ] || back=0
    po u${u}_vs_ck10 "$(ckpt $u)@0.5" "$CK10@0.7"
    po u${u}_vs_gen4 "$(ckpt $u)@0.5" "$GEN4@0.5"
    po u${u}_vs_u${back} "$(ckpt $u)@0.5" "$(ckpt $back)@0.5"
  else
    systemctl --user is-active -q rl18xx-pg2p || break
    sleep 300
  fi
done
log "evals done"
