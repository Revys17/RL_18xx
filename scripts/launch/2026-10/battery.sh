#!/bin/bash
# Validation battery before the pause (user, 2026-10-15): starts when gen4 has finished (and the
# value_v3 job is done), then frees the GPU. Results in logs/battery.log; saved games for the
# style profiles and the bankruptcy count under logs/eval/battery_*.
#  1. generational ladder (policy-only @0.5, 2v2 + 1v3): each main model vs the previous one,
#     plus gen4/1800 vs checkpoint 10 and vs pg4/750
#  2. gen4/1800 vs every exploiter / population member (2v2)
#  3. search: gen4@64 with its own (never PG-trained) value head vs with its critic vs with
#     value_v3; value_v3 @200 vs @64
#  4. the round-3 hunter's bankruptcy count vs gen4/1800 (gen3: 12.0%, gen2: 17.5%)
#  5. 120 self-play games per main model since gen1 for the style profiles (analysed on CPU)
cd "$(dirname "$0")/../../.."  # the repository root
LOG=logs/battery.log
say() { echo "$(date -Iseconds) $*" >> $LOG; }
EV=(env OMP_NUM_THREADS=1 .venv/bin/python scripts/eval_policy_only.py --games-per-task 90)
H2H=(env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2 .venv/bin/python scripts/eval_head_to_head.py)
CK10=model_checkpoints/AlphaZeroTransformer/20261004_134558_804974562/10.pth
PG3=model_checkpoints/AlphaZeroTransformer/20261007_094500_804974562/1.pth
PG4=model_checkpoints_pg/pg4_20261007/learner/750.pth
G=model_checkpoints_pg/gen1_20261009
GEN1=$G/learner/1000.pth; GEN2=$G/learner/1500.pth; GEN3=$G/learner/2000.pth
R=model_checkpoints_pg/rab_place_20261014
GEN4=$R/learner/1800.pth; GEN4C=$R/critic/1800.pth

# (2026-10-16 rerun: gen4 and value_v3 are done; the first run waited on a stale "gen4 exited" line)
[ -f $GEN4 ] || { say "no $GEN4"; exit 1; }
V3=$(python3 -c "import json; print(json.load(open('model_checkpoints_value/AlphaZeroTransformer/current_best.json'))['checkpoint_num'])")
V3=model_checkpoints_value/AlphaZeroTransformer/20261004_134558_804974562/$V3.pth
say "battery: gen4 = $GEN4, value_v3 = $V3"

run() {  # run NAME PREFIX cmd...: log the " vs " lines
  local name=$1 prefix=$2
  shift 2
  if "$@" > logs/eval_$name.out 2> logs/eval_$name.err; then
    grep " vs " logs/eval_$name.out | sed "s#^#$(date -Iseconds) $prefix#" >> $LOG
  else
    say "$name: failed: $(grep -E 'Error|died' logs/eval_$name.err | tail -1 | cut -c1-200)"
  fi
}

# 1. ladder
run battery_ladder "ladder 2v2 " "${EV[@]}" --match $PG3@0.5 $CK10@0.5 --match $PG4@0.5 $PG3@0.5 --match $GEN1@0.5 $PG4@0.5 \
  --match $GEN2@0.5 $GEN1@0.5 --match $GEN3@0.5 $GEN2@0.5 --match $GEN4@0.5 $GEN3@0.5 --match $GEN4@0.5 $CK10@0.5 \
  --match $GEN4@0.5 $PG4@0.5 --games 1200 --workers 32 --out logs/eval/battery_ladder
run battery_ladder_1v3 "ladder 1v3 " "${EV[@]}" --match $PG3@0.5 $CK10@0.5 --match $PG4@0.5 $PG3@0.5 \
  --match $GEN1@0.5 $PG4@0.5 --match $GEN2@0.5 $GEN1@0.5 --match $GEN3@0.5 $GEN2@0.5 --match $GEN4@0.5 $GEN3@0.5 \
  --a-seats 1 --games 1200 --workers 32 --out logs/eval/battery_ladder_1v3

# 2. every exploiter / member of the population
MATCHES=()
for m in $(python3 -c "import json; print(' '.join(m['path'] for m in json.load(open('model_checkpoints_pg/populations/league_v7.json'))['members']))"); do
  case $m in *rab_place*) continue ;; esac
  MATCHES+=(--match $GEN4@0.5 $m@0.5)
done
run battery_members "members 2v2 " "${EV[@]}" "${MATCHES[@]}" --games 1200 --workers 32 --out logs/eval/battery_members

# 3. search
run battery_search_critic "search " "${H2H[@]}" --match $GEN4+$GEN4C@64 $GEN4@64 --games 240 \
  --start-positions human_games/start_positions_1830_4p.jsonl --workers 24 --out logs/eval/battery_search_critic
run battery_search_v3 "search " "${H2H[@]}" --match $GEN4+$V3@64 $GEN4+$GEN4C@64 --games 240 \
  --start-positions human_games/start_positions_1830_4p.jsonl --workers 24 --out logs/eval/battery_search_v3
run battery_search_depth "search " "${H2H[@]}" --match $GEN4+$V3@200 $GEN4+$V3@64 --games 240 \
  --start-positions human_games/start_positions_1830_4p.jsonl --workers 24 --out logs/eval/battery_search_depth
for d in battery_search_critic battery_search_v3 battery_search_depth; do
  python3 -c "import json; [print('$d', s['A'][-60:], 'vs', s['B'][-60:], round(s['A_score'],3), '+-', round(s['A_score_se'],3)) for s in json.load(open('logs/eval/$d/summary.json'))]" 2>&1 | sed "s#^#$(date -Iseconds) #" >> $LOG
done

# 4. bankruptcies
H2=model_checkpoints_pg/gx2_bankrupt_20261013/learner/300.pth
"${EV[@]}" --match $H2@0.5 $GEN4@0.5 --games 400 --save-games 400 --workers 32 --out logs/eval/battery_bankrupt > /dev/null 2>&1
CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/count_bankruptcies.py logs/eval/battery_bankrupt 2>&1 | grep -v Warn | sed "s#^#$(date -Iseconds) bankruptcies #" >> $LOG

# 5. self-play games for the style profiles
for spec in gen2_1500=$GEN2 gen3_2000=$GEN3 gen4_final=$GEN4; do
  name=${spec%%=*}; path=${spec#*=}
  "${EV[@]}" --match $path@0.5 $path@0.5 --games 120 --save-games 120 --workers 32 --out logs/eval/member_profiles/$name > /dev/null 2>&1
done
say "battery done: the GPU is free"
