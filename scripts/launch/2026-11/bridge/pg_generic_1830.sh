#!/bin/bash
# Generic 1830 policy gradient (RL bridge milestone C2; docs/rl_model.md "Policy gradient for the generic
# model"): the 1830 GPU check's supervised generic checkpoint (gpu_check_1830.sh, $OUT/final.pth) refined by
# policy-only policy gradient with gen6's recipe -- mixed player counts 2-6 with per-count PFSP, placement
# reward, learner / opponent temperature 0.5, KL target 0.08 with the anchor moving every 50 updates, half the
# games from the real-rules auction, 30% from mid-game cuts, 20% random auction endings, per-count decision
# caps -- on generic networks served by generic inference servers.
#
# NOT LAUNCHED. Prerequisites: the GPU check passed (its final.pth within noise of SL ckpt 10), the GPU free.
#
#   bash scripts/launch/2026-11/bridge/pg_generic_1830.sh            # new run, or resume from state.json
#   bash scripts/launch/2026-11/bridge/pg_generic_1830.sh evals      # one round of policy-only evals now
#
# Departures from gen6's run.sh (scripts/launch/2026-11/gen6_20261110/run.sh), each named:
# - Start: the generic SL checkpoint (gen6 started from gen5); the critic starts from VALUE, the same recipe trained
#   1 epoch with its value heads refit (gen6: gen5's critic) -- the 10-epoch SL's value heads memorized its games.
# - League: the generic SL checkpoint alone at first, then the learner's own snapshots (every 50 updates, the
#   latest 4) -- legacy checkpoints (gen4 .. gen6, the exploiters) cannot sit in a generic league (they need
#   the legacy encoder and server); they are the eval panel instead. So the first updates are pg1-like
#   (vs SL + own pool), not gen6-like (vs a 33-member league).
# - Learner server: 2 replicas (gen6: 1); each opponent slot 1.
# Everything else as gen6: rows 65536, minibatch 2048 / microbatch 1024, lr 2e-5, critic lr 3e-5, kl 0.02
# -> target 0.08 (floor 0.005), anchor every 50, 4000 updates backstop, snapshot every 10, save every 20th game.
#
# Throughput (estimate, docs/rl_model.md): the generic default model costs ~1.5x the 1830 v2 model per position
# (CPU-measured), the workers' per-decision CPU is lower (~0.1 ms vs ~0.26 ms legacy encode); if gen6
# (~125k games/h) was GPU-bound, expect ~80k games/h (~1.5x longer per update). Read games_per_hour_run in
# history.jsonl; RL18XX_SERVER_PROFILE=60 prints each server's split (idle / collect / forward / replies).
set -uo pipefail
cd "$(dirname "$0")/../../../.."  # the repository root (run from a checkout with human_games/)

RUN=${RUN:-gpg1830_20261113}
SL=${SL:-model_checkpoints_generic/bridge_1830_check_20261113/final.pth}  # gpu_check_1830.sh's $OUT/final.pth
# The critic's start (main.py policy-gradient-generic --value): the same recipe trained 1 epoch, value heads refit
# (win CE .952 vs the 10-epoch SL's memorized 1.193; docs/training_runs_2026_11.md). VALUE=$SL: the SL's own heads.
VALUE=${VALUE:-model_checkpoints_generic/bridge_1830_critic1ep_20261118/final.pth}
OUT_DIR=${OUT_DIR:-model_checkpoints_generic_pg}
POP=${POP:-$OUT_DIR/populations/league_generic_1830_v1.json}
EVAL_OUT=${EVAL_OUT:-logs/eval/$RUN}
CPUS=${CPUS:-24-31,56-63}
SL10=${SL10:-/home/revys/longrun/work/lr1/model_checkpoints/AlphaZeroTransformer/20261004_134558_804974562/10.pth}
GEN6=${GEN6:-/home/revys/longrun/work/lr1/model_checkpoints_pg/gen6_20261110/learner/2400.pth}
export OMP_NUM_THREADS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ulimit -n "$(ulimit -Hn)"
PY=(taskset -c "$CPUS" nice -19 "$PWD/.venv/bin/python")

if [ "${1:-}" = "evals" ]; then
  # Policy-only evals of the newest snapshot, T 0.5, real-rules auction starts (gen6's evals.sh at 400 games):
  # vs the generic SL start (within-family progress) and vs legacy SL ckpt 10 / gen6 u2400 (the outside panel).
  U=$(ls "$OUT_DIR/$RUN/learner" | sed 's/\.pth$//' | sort -n | tail -1)
  L="generic:$OUT_DIR/$RUN/learner/$U.pth@0.5"
  mkdir -p "$EVAL_OUT"
  COMMON=(--games 400 --auction-start --workers 16 --generic-server --save-games 6)
  for spec in "4 2 generic:$SL@0.5 sl" "4 1 generic:$SL@0.5 sl" "2 1 generic:$SL@0.5 sl" \
              "4 2 $SL10@0.5 ckpt10" "4 1 $SL10@0.5 ckpt10" "4 1 $GEN6@0.5 gen6" "3 1 $GEN6@0.5 gen6" \
              "5 2 $GEN6@0.5 gen6" "6 3 $GEN6@0.5 gen6" "2 1 $GEN6@0.5 gen6"; do
    set -- $spec
    name="u${U}_${1}p_${2}v$(( $1 - $2 ))_$4"
    OMP_NUM_THREADS=1 "${PY[@]}" scripts/eval_policy_only.py --match "$L" "$3" --num-players "$1" --a-seats "$2" \
      "${COMMON[@]}" --out "$EVAL_OUT/$name" 2>&1 | tail -1 | tee -a "$EVAL_OUT/results.txt"
  done
  exit 0
fi

COMMON=(--run-name "$RUN" --out-dir "$OUT_DIR" --title 1830)
if [ -f "$OUT_DIR/$RUN/state.json" ]; then
  exec "${PY[@]}" main.py policy-gradient-generic --resume "$OUT_DIR/$RUN" "${COMMON[@]}"
fi
[ -f "$SL" ] || { echo "no SL checkpoint at $SL (run gpu_check_1830.sh first)"; exit 1; }
mkdir -p "$(dirname "$POP")"
[ -f "$POP" ] || printf '{"members": ["%s"]}\n' "$SL" > "$POP"
exec "${PY[@]}" main.py policy-gradient-generic "${COMMON[@]}" --policy "$SL" ${VALUE:+--value "$VALUE"} \
  --player-mix 2:15,3:20,4:35,5:20,6:10 \
  --max-decisions 1200 --max-decisions-by-players 2:1000,3:1100,5:1300,6:1500 \
  --workers 32 --games-per-task 96 --positions-per-seat 32 \
  --population "$POP" --opponent-slots 3 --pfsp-power 2.0 \
  --slot-refresh-every 2 --league-snapshot-every 50 --league-max-snapshots 4 --reward placement --anchor-every 50 \
  --rows-per-update 65536 --minibatch 2048 --microbatch 1024 --ppo-epochs 1 --clip 0.2 --lr 2e-5 --critic-lr 3e-5 \
  --kl-coef 0.02 --kl-target 0.08 --kl-coef-min 0.005 --entropy-coef 0.0 --learner-temperature 0.5 \
  --opponent-temperature 0.5 --opponent-price-eps 0.0 --max-updates 4000 --snapshot-every 10 \
  --pool-refresh-every 5 --random-start-fraction 0.2 --midgame-fraction 0.3 \
  --auction-fraction 0.5 --no-auction-unlock --auction-stall-decisions 150 --save-game-every 20 \
  --start-positions 'human_games/start_positions_1830_{n}p_v2.jsonl' \
  --midgame-positions 'human_games/midgame_positions_1830_{n}p_v2.jsonl' \
  --inference-batch-size 1024 --server-replicas 2 --threads 8
# --server-pipeline (a collector thread merging the next batch) was faster on the CPU transport benchmark, but the
# legacy server's collector thread was slower on the GPU (GIL): try it only with RL18XX_SERVER_PROFILE on.
