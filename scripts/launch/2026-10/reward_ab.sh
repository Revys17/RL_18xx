#!/bin/bash
# Reward A/B on the main run (2026-10-14): once gen3 (and its final checks) are done, three
# branches start from gen3/2000, identical except for the reward -- A win (1/0/0/0), B placement
# (1/0.5/0.2/0), C blend (1/0.3/0.1/0) -- 300 updates each against population v6 (v5 +
# gen3/2000), side by side. Judged only by WIN RATE: at 150 and 300 each branch plays 2v2 and
# 1v3 against gen3/2000; at the end a round robin of the three (2v2) and each alone vs 3x gen3.
cd "$(dirname "$0")/../../.."  # the repository root
LOG=logs/reward_ab.log
RUN=model_checkpoints_pg/gen1_20261009
GEN3=$RUN/learner/2000.pth
POP=model_checkpoints_pg/populations/league_v6.json
say() { echo "$(date -Iseconds) $*" >> $LOG; }
PG=(env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2 .venv/bin/python main.py policy-gradient)
EV=(env OMP_NUM_THREADS=1 .venv/bin/python scripts/eval_policy_only.py)
COMMON=(--policy $GEN3 --value $RUN/critic/2000.pth --population $POP --anchor-every 50
        --learner-temperature 0.5 --opponent-temperature 0.5 --minibatch 2048 --microbatch 1024 --lr 2e-5
        --kl-target 0.08 --midgame-fraction 0.3 --auction-fraction 0.15 --save-game-every 20 --max-updates 300
        --workers 32)

until grep -qE "gen3 exited|gen3 ended before" logs/gen3.log; do sleep 120; done
if [ ! -f $GEN3 ]; then say "no $GEN3: stopping"; exit 1; fi
.venv/bin/python - <<'PY' >> $LOG 2>&1
import json
from pathlib import Path
pops = Path("model_checkpoints_pg/populations")
v = json.loads((pops / "league_v5.json").read_text())
v["members"].append({"path": "model_checkpoints_pg/gen1_20261009/learner/2000.pth", "note": "gen3/2000 (start of the reward A/B)"})
(pops / "league_v6.json").write_text(json.dumps(v, indent=2) + "\n")
print(f"population v6: {len(v['members'])} members")
PY

declare -A SPEC=([win]=win [place]=placement [blend]=placement:1,0.3,0.1,0)
PIDS=()
for arm in win place blend; do
  "${PG[@]}" "${COMMON[@]}" --reward "${SPEC[$arm]}" --run-name rab_${arm}_20261014 > logs/policy_gradient_rab_$arm.log 2>&1 &
  PIDS+=($!)
  say "rab_$arm (${SPEC[$arm]}) started (pid $!)"
  sleep 120
done

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
for M in 150 300; do
  for arm in win place blend; do
    L=model_checkpoints_pg/rab_${arm}_20261014/learner
    until [ -f $L/$M.pth ]; do
      if ! ps -eo args | grep -qE "[r]un-name rab_${arm}_20261014"; then say "rab_$arm ended before $M"; break; fi
      sleep 60
    done
    [ -f $L/$M.pth ] || continue
    check rab_${arm}_$M "$arm@$M 2v2 " "${EV[@]}" --match $L/$M.pth@0.5 $GEN3@0.5 --games 1200 --workers 12 \
      --out logs/eval/rab_${arm}_${M}_2v2
    check rab_${arm}_${M}_1v3 "$arm@$M 1v3 " "${EV[@]}" --match $L/$M.pth@0.5 $GEN3@0.5 --a-seats 1 --games 1200 \
      --workers 12 --out logs/eval/rab_${arm}_${M}_1v3
  done
done
for pid in "${PIDS[@]}"; do wait $pid; done
say "all three branches done"
W=model_checkpoints_pg/rab_win_20261014/learner/300.pth
P=model_checkpoints_pg/rab_place_20261014/learner/300.pth
B=model_checkpoints_pg/rab_blend_20261014/learner/300.pth
check rab_roundrobin "round robin " "${EV[@]}" --match $P@0.5 $W@0.5 --match $B@0.5 $W@0.5 --match $P@0.5 $B@0.5 \
  --games 2400 --workers 32 --out logs/eval/rab_roundrobin
say "reward A/B done"
