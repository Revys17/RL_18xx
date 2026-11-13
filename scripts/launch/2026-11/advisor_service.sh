#!/bin/bash
# The advisor for advice and hotseat autoplay, as the systemd user service rl18xx-advisor (port 5002,
# reachable on the LAN), on the system measured strongest (2026-11-07, @200 under the train-shuffle rule:
# gen4 + lr1 value v4 vs gen4 + its critic 0.582 in 2v2):
# - the model's seats: gen4 searched with lr1's value v4; in the private auction gen4 + its critic;
# - advice: gen4's top moves, win estimates from value_v2 (the best measured on human games), and
#   lr1 v4 for "Think harder" (what the bots play);
# - the local 18xx.games copy's engine panel may ask for advice (--web-site-port 9292).
#   scripts/launch/2026-11/advisor_service.sh       # (re)start it; journalctl --user -u rl18xx-advisor
set -euo pipefail
cd "$(dirname "$0")/../../.."  # the repository root
UNIT=${UNIT:-rl18xx-advisor}
PORT=${PORT:-5002}
LR1=/home/revys/longrun/work/lr1
GEN4=$LR1/model_checkpoints_pg/rab_place_20261014/learner/1800.pth
CRITIC=$LR1/model_checkpoints_pg/rab_place_20261014/critic/1800.pth
V4=$LR1/model_checkpoints_value/lr1/promoted/4.pth
SL10=$LR1/model_checkpoints/AlphaZeroTransformer/20261004_134558_804974562/10.pth
# value_v2 exists only in this worktree's checkout.
VALUE_V2=/home/revys/workspace/RL_18xx/.claude/worktrees/gpu-experiments-training-93d287/model_checkpoints_value/AlphaZeroTransformer/20261004_134558_804974562/26.pth
systemctl --user stop "$UNIT" 2>/dev/null || true
systemctl --user reset-failed "$UNIT" 2>/dev/null || true
systemd-run --user --unit="$UNIT" -p MemoryMax=12G -p WorkingDirectory="$PWD" \
  -E CUDA_VISIBLE_DEVICES= -E OMP_NUM_THREADS=4 -E PYTHONDONTWRITEBYTECODE=1 \
  "$PWD/.venv/bin/python" main.py advisor --device cpu --cpu-threads 4 --host 0.0.0.0 --port "$PORT" \
  --policy "$GEN4" --auction-policy "$SL10" --value "$VALUE_V2" --search-value "$V4" \
  --play-policy "$GEN4" --play-value "$V4" --play-auction-policy "$GEN4" --play-auction-value "$CRITIC" \
  --web-site-port 9292
echo "$UNIT on 0.0.0.0:$PORT (journalctl --user -u $UNIT)"
