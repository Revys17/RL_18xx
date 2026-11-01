#!/bin/bash
# The auction-learning job end to end, unattended:
#   1. wait for lr1 to pause: its gen, value and ladder units inactive and the GPU idle (< 30% two
#      readings a minute apart); give up at DEADLINE (default noon tomorrow);
#   2. train (auction_train.sh) for TRAIN_HOURS (default 9; then a STOP file) or its max updates;
#   3. evaluate the last checkpoint, auction-only, scored by v3 (scripts/eval_auction.py): one and two
#      seats against gen4 + its critic (the auction system the bots play now), one seat against the
#      supervised policy sampled as people bid;
#   4. write logs/auction_<run>_done.json: the GPU is free (lr1 can resume).
#   setsid nohup scripts/launch/2026-10/auction_pipeline.sh auc1 >> logs/auction_pipeline.log 2>&1 < /dev/null &
cd "$(dirname "$0")/../../.."  # the repository root
# Dozens of workers, each with pipes to several inference servers: the default 1024 open files is too
# few (auc1's first start died of it). The evals inherit this; the training unit sets its own.
ulimit -n "$(ulimit -Hn)"
RUN=${1:-auc1}
TRAIN_HOURS=${TRAIN_HOURS:-9}
DEADLINE=$(date -d "${DEADLINE:-tomorrow 12:00}" +%s)
LR1=${LR1:-/home/revys/longrun/work/lr1}
CK=${CK:-/home/revys/workspace/RL_18xx/.claude/worktrees/gpu-experiments-training-93d287}
SL10=$LR1/model_checkpoints/AlphaZeroTransformer/20261004_134558_804974562/10.pth
GEN4CRITIC=$CK/model_checkpoints_pg/rab_place_20261014/learner/1800.pth+$CK/model_checkpoints_pg/rab_place_20261014/critic/1800.pth
RUN_DIR=$PWD/model_checkpoints_search/$RUN
DONE=logs/auction_${RUN}_done.json
UNIT=rl18xx-auction-train-$RUN
say() { echo "$(date -Iseconds) $*"; }
busy() {
  for unit in lr1-gen lr1-value lr1-ladder; do
    systemctl --user is-active -q "rl18xx-longrun@$unit" && return 0
  done
  local util
  util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)
  [ "${util:-100}" -ge 30 ]
}

say "waiting for lr1 to pause (run $RUN, deadline $(date -d @"$DEADLINE" -Iseconds))"
while busy || { sleep 60; busy; }; do
  if [ "$(date +%s)" -ge "$DEADLINE" ]; then say "deadline passed: not starting $RUN"; exit 1; fi
  sleep 60
done
say "lr1 paused and the GPU idle: starting $RUN for at most $TRAIN_HOURS h"
scripts/launch/2026-10/auction_train.sh "$RUN"
started=$(date +%s)
sleep 60
while systemctl --user is-active -q "$UNIT"; do
  if [ $(( $(date +%s) - started )) -ge $(( TRAIN_HOURS * 3600 )) ] && [ ! -e "$RUN_DIR/STOP" ]; then
    say "time budget reached: STOP"
    touch "$RUN_DIR/STOP"
  fi
  sleep 60
done
LAST=$(ls "$RUN_DIR"/learner/*.pth 2>/dev/null | sed 's#.*/##; s#\.pth$##' | sort -n | tail -1)
say "training ended after $(( ($(date +%s) - started) / 60 )) min; last update ${LAST:-none}"
if [ -z "$LAST" ] || [ "$LAST" -eq 0 ]; then
  say "training FAILED (no update): see logs/auction_train_$RUN.log"
fi
OUTS=()
if [ -n "$LAST" ] && [ "$LAST" -gt 0 ]; then
  CAND="$RUN_DIR/learner/$LAST.pth+$RUN_DIR/critic/$LAST.pth"
  mkdir -p logs/eval_auction
  evaluate() {  # name, then eval_auction.py's arguments
    local out="logs/eval_auction/${RUN}_u${LAST}_$1.json"
    shift
    say "eval -> $out"
    systemd-run --user --scope -q -p MemoryMax=80G .venv/bin/python scripts/eval_auction.py --candidate "$CAND" \
      --games 1200 --workers 24 --out "$out" "$@" >> "logs/eval_auction/${RUN}_u${LAST}.log" 2>&1 && OUTS+=("$out")
  }
  evaluate vs_gen4critic_1seat --baseline "$GEN4CRITIC" --baseline-prior-temperature 0.5 --seats 1
  evaluate vs_gen4critic_2seats --baseline "$GEN4CRITIC" --baseline-prior-temperature 0.5 --seats 2
  evaluate vs_sl10sampled_1seat --baseline "$SL10" --baseline-mode policy --seats 1
fi
python3 - "$DONE" "$RUN" "${LAST:-0}" "${OUTS[@]}" <<'PY'
import json, sys, datetime
done, run, last, outs = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4:]
evals = {}
for path in outs:
    try:
        evals[path] = json.load(open(path))
    except Exception as e:
        evals[path] = {"error": repr(e)}
json.dump({"run": run, "last_update": last, "trained": last > 0,
           "finished": datetime.datetime.now().isoformat(timespec="seconds"), "gpu_free": True, "evals": evals},
          open(done, "w"), indent=2)
PY
say "done: $DONE (the GPU is free)"
