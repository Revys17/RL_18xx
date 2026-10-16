#!/bin/bash
# After the reward A/B (logs/reward_ab.log): pick the branch with the best round-robin win
# score (ties within 0.02 of the win branch go to the win branch, the simpler objective) and
# continue it as the main run ("gen4") from update 300 to 800 against the same population.
# Checks every 100 (clean @0.5): 2v2 vs gen3/2000, the round-3 placement exploiter and the
# round-3 hunter; 1v3 vs 3x gen3/2000. Keeps the GPU busy until the user decides otherwise.
cd "$(dirname "$0")/../../.."  # the repository root
LOG=logs/gen4.log
GEN3=model_checkpoints_pg/gen1_20261009/learner/2000.pth
P2=model_checkpoints_pg/gx2_placement_20261013/learner/300.pth
H2=model_checkpoints_pg/gx2_bankrupt_20261013/learner/300.pth
say() { echo "$(date -Iseconds) $*" >> $LOG; }
EV=(env OMP_NUM_THREADS=1 .venv/bin/python scripts/eval_policy_only.py)

until grep -q "reward A/B done" logs/reward_ab.log; do sleep 120; done
ARM=$(python3 - <<'PY'
import json
s = {m["A"].split("/")[-3].split("_")[1] + ">" + m["B"].split("/")[-3].split("_")[1]: m["A_score"]
     for m in json.load(open("logs/eval/rab_roundrobin/summary.json"))}
p_w, b_w, p_b = s["place>win"], s["blend>win"], s["place>blend"]
score = {"win": ((1 - p_w) + (1 - b_w)) / 2, "place": (p_w + p_b) / 2, "blend": (b_w + 1 - p_b) / 2}
best = max(score, key=score.get)
if best != "win" and score[best] - score["win"] < 0.02:
    best = "win"
print(best, json.dumps({k: round(v, 3) for k, v in score.items()}))
PY
)
say "round-robin scores ${ARM#* }: continuing ${ARM%% *} as gen4"
ARM=${ARM%% *}
RUN=model_checkpoints_pg/rab_${ARM}_20261014
env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2 .venv/bin/python main.py policy-gradient \
  --resume $RUN --max-updates 800 --midgame-fraction 0.3 --auction-fraction 0.15 --microbatch 1024 --save-game-every 20 \
  > logs/policy_gradient_gen4.log 2>&1 &
PID=$!
say "gen4 started (pid $PID): $RUN resumed from 300 to 800"

check() {  # check NAME PREFIX cmd...: log the " vs " lines; retry twice on failure
  local name=$1 prefix=$2
  shift 2
  for attempt in 1 2 3; do
    if "$@" > logs/eval_$name.out 2> logs/eval_$name.err; then
      grep " vs " logs/eval_$name.out | sed "s#^#$(date -Iseconds) $prefix#" >> $LOG
      return 0
    fi
    say "$name: failed (attempt $attempt): $(grep -E 'Error|died' logs/eval_$name.err | tail -1 | cut -c1-200)"
    sleep 120
  done
}
L=$RUN/learner
for M in $(seq 400 100 800); do
  until [ -f $L/$M.pth ]; do
    if ! kill -0 $PID 2>/dev/null; then say "gen4 ended before $M ($(tail -1 logs/policy_gradient_gen4.log | cut -c1-160))"; exit 0; fi
    sleep 30
  done
  sleep 5
  check gen4_2v2_$M "2v2 " "${EV[@]}" --match $L/$M.pth@0.5 $GEN3@0.5 --match $L/$M.pth@0.5 $P2@0.5 \
    --match $L/$M.pth@0.5 $H2@0.5 --games 1200 --workers 16 --out logs/eval/gen4_2v2_$M
  check gen4_1v3_$M "1v3 " "${EV[@]}" --match $L/$M.pth@0.5 $GEN3@0.5 --a-seats 1 --games 1200 --workers 16 \
    --out logs/eval/gen4_1v3_$M
done
wait $PID
say "gen4 exited $?"
