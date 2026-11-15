#!/bin/bash
# Step 0 of docs/robustness_test_plan.md (docs/robustness_measurement.md): which search settings should the bots use,
# with no retraining? gen6 u2400's policy + its critic 2400 at 400 readouts (the live bots' PUCT: mean Q, c_puct 0.05,
# 8 leaves a batch, each distinct leaf once, kept trees with a total budget), varying
#   the leaf value: the critic's win head (#win), its placement-points head (#score: the live bots since 2026-11-14
#                   18:32) or (1 - w) win + w score (#blend=0.5);
#   the prior temperature: 0.5 (gen6's learner temperature, today) or 1.0 (lets search overrule the sharp policy).
# Variants (eval_head_to_head.py player specs, the syntax of scripts/launch/2026-11/bridge/search_value_gen6.sh):
#   win05 / score05 / blend05: #win / #score / #blend=0.5 at ptemp 0.5;  win10 / score10: at ptemp 1.0
# Two measurements:
#   m1  -- the regression set (scripts/robustness/regression_set.py score; ~3 s a position on 4 CPU threads, 59
#          positions: ~3 min a variant on CPU, no GPU needed)
#   h2h -- 4-player 2v2 head-to-heads against the live setting (score05), 480 games each (+-0.023), all six seat
#          arrangements, from the private auction under the real rules (STARTS=sr1: human first-Stock-Round starts +
#          20% random auction endings, as search_value_gen6.sh), argmax play (--softpick 0):
#            win_vs_score      win05 vs score05     the head the bots searched until 11-14
#            blend_vs_score    blend05 vs score05
#            ptemp1_vs_ptemp05 score10 vs score05   prior temperature with the live head
#            win_ptemp1        win10 vs win05       prior temperature with the win head (optional: STEPS)
# ~1.5-2 GPU h per 480-game match at 400 readouts with both sides searching (scaled from the 2026-11-02 @200 2v2
# matches, ~55 min; budget=total keeps trees, so evaluations grow less than 2x): ~6-8 GPU h for the first three.
#
#   bash scripts/robustness/search_settings.sh                    # m1 for every variant, then the three main matches
#   bash scripts/robustness/search_settings.sh m1                 # the regression set only (CPU is fine)
#   bash scripts/robustness/search_settings.sh win_vs_score       # chosen matches
#   DRY=1 DRY_ROOT=<scratch> bash scripts/robustness/search_settings.sh   # CPU: m1 on 3 positions at 16 readouts,
#                                                                         # 6-game matches at 8 readouts
# Results: $OUT/m1_<variant>.jsonl (+ .log), $OUT/<match>/ and $OUT/results.txt.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1  # the repository root
ROOT=$PWD
LR1=${LR1:-/home/revys/longrun/work/lr1}
GEN6L=${GEN6L:-$LR1/model_checkpoints_pg/gen6_20261110/learner/2400.pth}
GEN6C=${GEN6C:-$LR1/model_checkpoints_pg/gen6_20261110/critic/2400.pth}
HG=${HG:-human_games}
CPUS=${CPUS:-24-31,56-63}
DRY=${DRY:-}
if [ -n "$DRY" ]; then
  export CUDA_VISIBLE_DEVICES=
  OUT=${OUT:-${DRY_ROOT:?DRY=1 needs DRY_ROOT}/search_settings}
  READOUTS=${READOUTS:-8}
  M1_READOUTS=${M1_READOUTS:-16}
  M1_LIMIT=(--limit "${M1_POSITIONS:-3}")
  GAMES=${GAMES:-6}
  WORKERS=${WORKERS:-3}
  EXTRA=(--games-per-task 2 --dedupe-maps --server-replicas 1 --max-length 300)
else
  OUT=${OUT:-logs/robustness/step0_search_settings}
  READOUTS=${READOUTS:-400}
  M1_READOUTS=${M1_READOUTS:-$READOUTS}
  M1_LIMIT=()
  GAMES=${GAMES:-480}
  WORKERS=${WORKERS:-12}
  EXTRA=(--games-per-task 8 --dedupe-maps --server-replicas 2)
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
fi
PUCT="/0.05:mean,dedupe,par=8,budget=total"
declare -A VARIANT=(
  [win05]="$GEN6L+$GEN6C@$READOUTS$PUCT,ptemp=0.5"
  [score05]="$GEN6L+$GEN6C#score@$READOUTS$PUCT,ptemp=0.5"
  [blend05]="$GEN6L+$GEN6C#blend=0.5@$READOUTS$PUCT,ptemp=0.5"
  [win10]="$GEN6L+$GEN6C@$READOUTS$PUCT,ptemp=1.0"
  [score10]="$GEN6L+$GEN6C#score@$READOUTS$PUCT,ptemp=1.0"
)
if [ "${STARTS:-auction}" = sr1 ]; then
  STARTS_ARGS=(--start-positions "$HG/start_positions_1830_4p_v2.jsonl" --random-start-fraction 0.2)
else
  STARTS_ARGS=(--no-auction-unlock)
fi
PY=(taskset -c "$CPUS" nice -n 19 env OMP_NUM_THREADS=2 "$ROOT/.venv/bin/python" -P)
for f in "$GEN6L" "$GEN6C"; do [ -f "$f" ] || { echo "missing checkpoint $f"; exit 1; }; done

m1() {  # the regression set for every variant (the variant's spec minus its readouts: the scorer's --search)
  for v in win05 score05 blend05 win10 score10; do
    local spec=${VARIANT[$v]}
    local model=${spec%@*} search="@${spec##*@}"
    search=${search/,budget=total/}  # a fresh tree per position: nothing to keep
    if [ -s "$OUT/m1_$v.jsonl" ]; then echo "m1 $v: done, kept"; continue; fi
    "${PY[@]}" scripts/robustness/regression_set.py score --model "$model" --search "$search" \
      --readouts "$M1_READOUTS" --threads 4 --label "$v" "${M1_LIMIT[@]}" --out "$OUT/m1_$v.jsonl" \
      > "$OUT/m1_$v.log" 2>&1 || { echo "m1 $v failed: $OUT/m1_$v.log"; continue; }
  done
  "$ROOT/.venv/bin/python" -P scripts/robustness/regression_set.py summary "$OUT"/m1_*.jsonl | tee -a "$OUT/results.txt"
}

STEPS=("$@")
[ ${#STEPS[@]} -eq 0 ] && STEPS=(m1 win_vs_score blend_vs_score ptemp1_vs_ptemp05)
mkdir -p "$OUT"
for step in "${STEPS[@]}"; do
  case "$step" in
    m1) m1; continue ;;
    win_vs_score) A=${VARIANT[win05]}; B=${VARIANT[score05]} ;;
    blend_vs_score) A=${VARIANT[blend05]}; B=${VARIANT[score05]} ;;
    ptemp1_vs_ptemp05) A=${VARIANT[score10]}; B=${VARIANT[score05]} ;;
    win_ptemp1) A=${VARIANT[win10]}; B=${VARIANT[win05]} ;;
    *) echo "unknown step $step (m1 win_vs_score blend_vs_score ptemp1_vs_ptemp05 win_ptemp1)"; exit 2 ;;
  esac
  if [ -e "$OUT/$step/summary.json" ]; then echo "$step: done, kept"; continue; fi
  "${PY[@]}" scripts/eval_head_to_head.py --match "$A" "$B" --games "$GAMES" --num-players 4 --a-seats 2 \
    --softpick 0 "${STARTS_ARGS[@]}" --workers "$WORKERS" "${EXTRA[@]}" --out "$OUT/$step" > "$OUT/$step.log" 2>&1
  status=$?
  line=$("$ROOT/.venv/bin/python" -P - "$OUT/$step/summary.json" 2>/dev/null <<'EOF'
import json, sys
for s in json.load(open(sys.argv[1])):
    print("A %.3f +- %.3f over %s games (fair 0.50; worth share %.3f; terminations %s; seat win rates %s)" % (
        s["A_score"], s["A_score_se"], s["games"], s["A_worth_share"], json.dumps(s.get("terminations")),
        json.dumps(s.get("seat_win_rate"))))
EOF
)
  echo "$step (exit $status): ${line:-no summary, see $OUT/$step.log}" | tee -a "$OUT/results.txt"
done
