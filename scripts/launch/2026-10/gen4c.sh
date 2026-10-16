#!/bin/bash
# gen4 resumed 1040 -> 1800 with 96 games per worker task (bigger inference batches), population v7
# placement exploiter pinned at 10% since gen3 and gen4 kept losing ground to it), alongside
# policy-only value-net data from gen4/800. Checks every 100 (clean @0.5): 2v2 vs gen4/800,
# gen3/2000 and the placement exploiter; 1v3 vs 3x gen4/800.
cd "$(dirname "$0")/../../.."  # the repository root
LOG=logs/gen4.log
RUN=model_checkpoints_pg/rab_place_20261014
L=$RUN/learner
G4=$L/800.pth
GEN3=model_checkpoints_pg/gen1_20261009/learner/2000.pth
P2=model_checkpoints_pg/gx2_placement_20261013/learner/300.pth
say() { echo "$(date -Iseconds) $*" >> $LOG; }
EV=(env OMP_NUM_THREADS=1 .venv/bin/python scripts/eval_policy_only.py)
env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2 .venv/bin/python main.py policy-gradient \
  --resume $RUN --max-updates 1800 --games-per-task 96 --population model_checkpoints_pg/populations/league_v7.json --midgame-fraction 0.3 \
  --auction-fraction 0.15 --microbatch 1024 --save-game-every 20 >> logs/policy_gradient_gen4.log 2>&1 &
PID=$!
say "gen4 resumed (pid $PID): 1040 -> 1800 with 96 games per task against league_v7 (placement exploiter pinned at 10%)"
check() {
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
for M in $(seq 1100 100 1800); do
  until [ -f $L/$M.pth ]; do
    if ! kill -0 $PID 2>/dev/null; then say "gen4 ended before $M ($(tail -1 logs/policy_gradient_gen4.log | cut -c1-160))"; exit 0; fi
    sleep 30
  done
  sleep 5
  check gen4_2v2_$M "2v2 " "${EV[@]}" --match $L/$M.pth@0.5 $G4@0.5 --match $L/$M.pth@0.5 $GEN3@0.5 \
    --match $L/$M.pth@0.5 $P2@0.5 --games 1200 --workers 16 --games-per-task 90 --out logs/eval/gen4_2v2_$M
  check gen4_1v3_$M "1v3 " "${EV[@]}" --match $L/$M.pth@0.5 $G4@0.5 --a-seats 1 --games 1200 --workers 16 \
    --out logs/eval/gen4_1v3_$M
done
wait $PID
say "gen4 exited $? at 1800"
