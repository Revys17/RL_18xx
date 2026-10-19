#!/bin/bash
# The Q-target pilot (2026-10-19, docs/search_learning_design.md "Q-based policy targets"): two
# arms from az4 u100 with az4 / az5b's settings and --target cq, 12 workers each, side by side,
# each stopped after 90 minutes (touch <run>/STOP): A with KataGo surprise weighting 0.5 (as
# az4/az5), B without. Then the offline checks: each snapshot's own 200-readout search of the 600
# target-quality positions (search_target_quality.py search / steps) and near-greedy policy-only
# checks (T 0.1 both sides, 1200 games vs gen4).
cd "$(dirname "$0")/../../.."
P=model_checkpoints_search/az4_20261018/learner/100.pth
arm() {  # arm RUN SURPRISE
  env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2 .venv/bin/python main.py search-learning \
    --policy $P --shared --value-stop-grad --value-head win --run-name $1 \
    --target cq --c-puct 0.05 --n-full 200 --n-fast 48 --p-full 0.25 \
    --reuse-tree total --dedupe-maps --server-replicas 2 --workers 12 --games-per-worker 32 \
    --start-positions human_games/start_positions_1830_4p_v2.jsonl --random-start-fraction 0.2 \
    --midgame-positions human_games/midgame_positions_1830_4p_v2.jsonl --midgame-fraction 0.3 --auction-fraction 0.15 \
    --full-move-temperature 1 --explore-decisions 120 \
    --critic-positions-per-game 0 --value-lambda 0.97 --value-bootstrap-weight 0.5 \
    --value-loss-weight 1 --score-loss-weight 1 --aux-loss-weight 0.1 --surprise-weight $2 --surprise-cap 8 \
    --lr 5e-6 --critic-lr 3e-5 --kl-coef 0 --kl-target 0 --anchor-every 0 --target-temperature 1 \
    --window-rows 131072 --rows-per-update 8192 --min-window-rows 16384 --steps-per-update 8 --minibatch 2048 \
    --microbatch 1024 --max-updates 60 --snapshot-every 10 > logs/search_learning_$1.log 2>&1 &
  (sleep 5400; touch model_checkpoints_search/$1/STOP) &
}
arm qtarget_pilot_20261019 0
sleep 220
arm qtarget_pilot_sw05_20261019 0.5
wait
COLS="--collection logs/eval/az4_u200_vs_az1_u320 --collection logs/eval/az3_u200_vs_az1_u320 \
  --collection logs/eval/az1_u300_sys200 --games 40 --per-game 5"
for spec in qtarget_pilot_20261019/learner/29 qtarget_pilot_sw05_20261019/learner/28; do
  out=logs/target_quality/steps/$(echo $spec | tr / _)
  .venv/bin/python scripts/search_target_quality.py search --model model_checkpoints_search/$spec.pth --out $out \
    $COLS --config 0.05@200
  .venv/bin/python scripts/eval_policy_only.py --games-per-task 90 --workers 12 \
    --start-positions human_games/start_positions_1830_4p_v2.jsonl \
    --match model_checkpoints_search/$spec.pth@0.1 model_checkpoints_pg/rab_place_20261014/learner/1800.pth@0.1 \
    --games 1200 --out logs/eval/greedy_$(echo $spec | tr / _)
done
