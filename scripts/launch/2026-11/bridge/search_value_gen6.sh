#!/bin/bash
# The pending gen6 search-value check (legacy 1830 engine and models; scripts/eval_head_to_head.py): does search on top
# of gen6 u2400 beat the bare policy, and with which value?
#   S_critic = gen6 u2400's policy + its own PG critic (critic/2400) searched at 200 readouts
#   S_v4     = gen6 u2400's policy + lr1's promoted value v4 (what the bots search with today: swap_bots.sh) at 200
#   P        = gen6 u2400 policy-only at T 0.5 (how gen6 was trained and judged)
# Matches, 4 players 2v2 (all six seat arrangements), 480 games each (+-0.023):
#   critic_vs_policy  S_critic vs P      search gain with the critic
#   v4_vs_policy      S_v4 vs P          search gain with v4
#   critic_vs_v4      S_critic vs S_v4   which value to search gen6 with (the bots' choice)
#   control           P@T0.05 vs P@T0.5  greediness alone (pg2p's control: ~.49 there), so a search gain is search
# Search settings = the eval ladder's PG-policy setting (scripts/launch/2026-11/eval_ladder_longrun.json, gen4critic):
# @200/0.05:mean,dedupe,par=8,ptemp=0.5,budget=total (ptemp 0.5 = gen6's learner temperature), --softpick 0, games
# from human SR1 starts (human_games/start_positions_1830_4p_v2.jsonl, + 20% random auction endings) -- v4 was
# trained past the auction and the bots play the auction with gen6 + critic, so the check is about the game after it;
# AUCTION=1 starts every game at the private auction (real rules) instead. The train-shuffle rule is on (as always).
# NOT LAUNCHED. ~1 GPU h per searched match (the 2026-11-02 @200 2v2 480-game matches took ~55 min with both sides
# searching; one side here is policy-only), the control ~10 min: ~3-3.5 GPU h in all.
#
#   bash scripts/launch/2026-11/bridge/search_value_gen6.sh                       # all four matches
#   bash scripts/launch/2026-11/bridge/search_value_gen6.sh critic_vs_v4 control  # chosen ones
#   DRY=1 DRY_ROOT=<scratch> GEN6L=<tiny.pth> GEN6C=<tiny.pth> V4=<tiny.pth> bash .../search_value_gen6.sh  # CPU dry
#   run (tiny stand-in checkpoints, 6 games per match at 8 readouts)
#
# Run from a checkout with human_games/ (or set HG). Results: $OUT/<match>/ and $OUT/results.txt.
set -uo pipefail
cd "$(dirname "$0")/../../../.." || exit 1  # the repository root
ROOT=$PWD
LR1=/home/revys/longrun/work/lr1
GEN6L=${GEN6L:-$LR1/model_checkpoints_pg/gen6_20261110/learner/2400.pth}
GEN6C=${GEN6C:-$LR1/model_checkpoints_pg/gen6_20261110/critic/2400.pth}
V4=${V4:-$LR1/model_checkpoints_value/lr1/promoted/4.pth}
HG=${HG:-human_games}
CPUS=${CPUS:-24-31,56-63}
DRY=${DRY:-}
if [ -n "$DRY" ]; then
  export CUDA_VISIBLE_DEVICES=
  OUT=${OUT:-${DRY_ROOT:?DRY=1 needs DRY_ROOT}/search_value_gen6}
  READOUTS=${READOUTS:-8}
  GAMES=${GAMES:-6}
  WORKERS=${WORKERS:-3}
  EXTRA=(--games-per-task 2 --dedupe-maps --server-replicas 1 --max-length 300)
else
  OUT=${OUT:-logs/eval/search_value_gen6_u2400}
  READOUTS=${READOUTS:-200}
  GAMES=${GAMES:-480}
  WORKERS=${WORKERS:-12}
  EXTRA=(--games-per-task 8 --dedupe-maps --server-replicas 2)
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
fi
SEARCH="@$READOUTS/0.05:mean,dedupe,par=8,ptemp=0.5,budget=total"
S_CRITIC="$GEN6L+$GEN6C$SEARCH"
S_V4="$GEN6L+$V4$SEARCH"
P="$GEN6L:sample=0.5"
P_GREEDY="$GEN6L:sample=0.05"
if [ -n "${AUCTION:-}" ]; then
  STARTS=(--no-auction-unlock)
else
  STARTS=(--start-positions "$HG/start_positions_1830_4p_v2.jsonl" --random-start-fraction 0.2)
fi
PY=(taskset -c "$CPUS" nice -n 19 env OMP_NUM_THREADS=2 "$ROOT/.venv/bin/python")
for f in "$GEN6L" "$GEN6C" "$V4"; do [ -f "$f" ] || { echo "missing checkpoint $f"; exit 1; }; done

STEPS=("$@")
[ ${#STEPS[@]} -eq 0 ] && STEPS=(critic_vs_policy v4_vs_policy critic_vs_v4 control)
mkdir -p "$OUT"
for step in "${STEPS[@]}"; do
  case "$step" in
    critic_vs_policy) A=$S_CRITIC; B=$P ;;
    v4_vs_policy) A=$S_V4; B=$P ;;
    critic_vs_v4) A=$S_CRITIC; B=$S_V4 ;;
    control) A=$P_GREEDY; B=$P ;;
    *) echo "unknown match $step (critic_vs_policy v4_vs_policy critic_vs_v4 control)"; exit 2 ;;
  esac
  if [ -e "$OUT/$step/summary.json" ]; then echo "$step: done, kept"; continue; fi
  "${PY[@]}" scripts/eval_head_to_head.py --match "$A" "$B" --games "$GAMES" --num-players 4 --a-seats 2 \
    --softpick 0 "${STARTS[@]}" --workers "$WORKERS" "${EXTRA[@]}" --out "$OUT/$step" > "$OUT/$step.log" 2>&1
  status=$?
  line=$("$ROOT/.venv/bin/python" - "$OUT/$step/summary.json" 2>/dev/null <<'EOF'
import json, sys
for s in json.load(open(sys.argv[1])):
    print("A %.3f +- %.3f over %s games (fair 0.50; worth share %.3f; terminations %s; seat win rates %s)" % (
        s["A_score"], s["A_score_se"], s["games"], s["A_worth_share"], json.dumps(s.get("terminations")),
        json.dumps(s.get("seat_win_rate"))))
EOF
)
  echo "$step (exit $status): ${line:-no summary, see $OUT/$step.log}" | tee -a "$OUT/results.txt"
done
