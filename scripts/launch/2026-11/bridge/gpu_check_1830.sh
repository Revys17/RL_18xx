#!/bin/bash
# The 1830 GPU check of the title-generic model (design docs/rl_bridge_design.md §4.4; recipe and expected
# numbers in docs/rl_model.md "The 1830 GPU check"): the default-size generic model pretrained on the human
# corpus exactly as SL checkpoint 10 was (two stages: joint run early-stopped on validation loss, then the
# value heads refit on the frozen best), then compared with SL checkpoint 10
#   - on the same validation positions: top-1 per descriptor family (Par / BuyTrain.* / LayTile / ...),
#     policy CE, value CE against the equal-odds baseline, winner accuracy;
#   - in policy-only head-to-heads at T 0.5 from the private auction (real rules, the train-shuffle rule on):
#     4 players 2v2, 1v3 both ways, 2 players 1v1; 800 games each.
# Pass bar (design): "within noise" of checkpoint 10.
#
#   bash scripts/launch/2026-11/bridge/gpu_check_1830.sh                 # every step in order
#   bash scripts/launch/2026-11/bridge/gpu_check_1830.sh pretrain eval   # chosen steps: convert pretrain compare eval
#
# Estimated (2026-11-13, CPU-measured; no GPU was available to measure): convert ~2 min (CPU); pretrain ~5-6 GPU h
# (10 epochs of 2.38M positions at ~1.4x the 1830 v2 model's cost per position -- SL ckpt 10's own two-stage run
# took ~3.5 h; read the real rate off the first log lines' "elapsed"); compare ~10-20 min; eval ~10-20 min per
# match (the legacy side on the GPU inference server, the generic side in the 16 pool workers on CPU).
#
# Run it from a checkout that has human_games/ (the main checkout; worktrees don't), or set DATA_SRC / DATA to
# absolute paths. Steps are idempotent only for convert (an existing dataset is kept); pretrain appends to $OUT.
set -euo pipefail
cd "$(dirname "$0")/../../../.."  # the repository root

RUN=${RUN:-bridge_1830_check_20261113}
DATA_SRC=${DATA_SRC:-human_games/1830_clean_all}  # lmdb_v6k's source = SL ckpt 10's games (60 no longer replay)
DATA=${DATA:-human_games/generic_1830_clean_all_v1}
OUT=${OUT:-model_checkpoints_generic/$RUN}
EVAL_OUT=${EVAL_OUT:-logs/eval/$RUN}
SL10=${SL10:-/home/revys/longrun/work/lr1/model_checkpoints/AlphaZeroTransformer/20261004_134558_804974562/10.pth}
DEVICE=${DEVICE:-cuda}
GENERIC_DEVICE=${GENERIC_DEVICE:-cpu}  # the generic player in eval_policy_only's pool workers
# GENERIC_SERVER=1: the generic player on a generic inference server (the GPU; milestone C2) instead
GENERIC_SERVER=${GENERIC_SERVER:-}
CPUS=${CPUS:-24-31,56-63}
GAMES=${GAMES:-800}
WORKERS=${WORKERS:-16}
PY=(taskset -c "$CPUS" nice -19 env OMP_NUM_THREADS=4 "$PWD/.venv/bin/python")
PY1=(taskset -c "$CPUS" nice -19 env OMP_NUM_THREADS=1 "$PWD/.venv/bin/python")  # one thread per pool worker
STEPS=("$@")
[ ${#STEPS[@]} -eq 0 ] && STEPS=(convert pretrain compare eval)
want() { local s; for s in "${STEPS[@]}"; do [ "$s" = "$1" ] && return 0; done; return 1; }

if want convert; then
  if [ -e "$DATA/meta.json" ]; then
    echo "convert: $DATA exists, kept"
  else
    "${PY[@]}" main.py convert-generic --data-dir "$DATA_SRC" --output "$DATA" --workers 8
  fi
fi

if want pretrain; then
  mkdir -p "$OUT"
  # The legacy defaults (main.py pretrain): batch 256, lr 1e-3 cosine, 10 epochs, patience 4, value / score /
  # price weights 0.1, label smoothing 0.03, entropy 0.01, value refit every 250 steps (patience 4).
  "${PY[@]}" main.py pretrain-generic --data "$DATA" --out "$OUT" --model-preset default \
    --device "$DEVICE" --workers 6 --threads 4 --epochs 10 --batch-size 256 --lr 1e-3 \
    2>&1 | tee -a "$OUT/train.log"
fi

if want compare; then
  mkdir -p "$EVAL_OUT"
  "${PY[@]}" main.py compare-generic --generic "$OUT/final.pth" --legacy "$SL10" --data "$DATA" \
    --split validation --device "$DEVICE" --batch-size 256 --out "$EVAL_OUT/compare_validation.json" \
    2>&1 | tee "$EVAL_OUT/compare_validation.log"
fi

if want eval; then
  mkdir -p "$EVAL_OUT"
  G="generic:$OUT/final.pth@0.5"
  L="$SL10@0.5"
  COMMON=(--games "$GAMES" --auction-start --workers "$WORKERS" --generic-device "$GENERIC_DEVICE" --save-games 12)
  [ -n "$GENERIC_SERVER" ] && COMMON+=(--generic-server)
  "${PY1[@]}" scripts/eval_policy_only.py --match "$G" "$L" --num-players 4 --a-seats 2 "${COMMON[@]}" \
    --out "$EVAL_OUT/h2h_4p_2v2" 2>&1 | tee "$EVAL_OUT/h2h_4p_2v2.log"
  "${PY1[@]}" scripts/eval_policy_only.py --match "$G" "$L" --num-players 4 --a-seats 1 "${COMMON[@]}" \
    --out "$EVAL_OUT/h2h_4p_1v3_generic_alone" 2>&1 | tee "$EVAL_OUT/h2h_4p_1v3_generic_alone.log"
  "${PY1[@]}" scripts/eval_policy_only.py --match "$L" "$G" --num-players 4 --a-seats 1 "${COMMON[@]}" \
    --out "$EVAL_OUT/h2h_4p_1v3_sl10_alone" 2>&1 | tee "$EVAL_OUT/h2h_4p_1v3_sl10_alone.log"
  "${PY1[@]}" scripts/eval_policy_only.py --match "$G" "$L" --num-players 2 --a-seats 1 "${COMMON[@]}" \
    --out "$EVAL_OUT/h2h_2p_1v1" 2>&1 | tee "$EVAL_OUT/h2h_2p_1v1.log"
fi
