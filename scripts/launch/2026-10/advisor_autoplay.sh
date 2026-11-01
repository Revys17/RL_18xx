#!/bin/bash
# The advisor backend with hotseat autoplay against the strongest system measured so far
# (2026-10-30, the long run lr1's eval ladder): the supervised policy (checkpoint 10, priors at
# temperature 1) searched with lr1's promoted value v3 -- ~55 Elo above gen4 + its critic in 2v2
# and 1v3 (u0..u3 tie within noise) -- and, for searches from the private auction (lr1 never plays
# it), gen4 + its critic (its run started 15% of games there). The advice networks stay what the
# advisor served before (gen4, value_v2, gen4's critic for "Think harder"). CPU only, LAN-reachable.
#   scripts/launch/2026-10/advisor_autoplay.sh [port]     # default 5003
# HOST=127.0.0.1 keeps it on this machine (the live bot's own backend); AUCTION_POLICY / AUCTION_VALUE
# replace the auction system (e.g. an auction-learning run's learner/<n>.pth and critic/<n>.pth).
cd "$(dirname "$0")/../../.."  # the repository root
PORT=${1:-5003}
HOST=${HOST:-0.0.0.0}
LR1=${LR1:-/home/revys/longrun/work/lr1}
CK=${CK:-/home/revys/workspace/RL_18xx/.claude/worktrees/gpu-experiments-training-93d287}  # gen4's checkout
SL10=$LR1/model_checkpoints/AlphaZeroTransformer/20261004_134558_804974562/10.pth
V3=$LR1/model_checkpoints_value/lr1/promoted/3.pth
GEN4=$CK/model_checkpoints_pg/rab_place_20261014/learner/1800.pth
CRITIC=$CK/model_checkpoints_pg/rab_place_20261014/critic/1800.pth
VALUE_V2=$CK/model_checkpoints_value/AlphaZeroTransformer/20261004_134558_804974562/26.pth
AUCTION_POLICY=${AUCTION_POLICY:-$GEN4}
AUCTION_VALUE=${AUCTION_VALUE:-$CRITIC}
mkdir -p logs
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 $(command -v setsid) nohup .venv/bin/python main.py advisor --device cpu --cpu-threads 4 \
  --host "$HOST" --port "$PORT" \
  --policy "$GEN4" --auction-policy "$SL10" --value "$VALUE_V2" --search-value "$CRITIC" \
  --play-policy "$SL10" --play-value "$V3" --play-auction-policy "$AUCTION_POLICY" --play-auction-value "$AUCTION_VALUE" \
  >> "logs/advisor_autoplay_$PORT.log" 2>&1 &
echo "advisor (autoplay) pid $! on $HOST:$PORT; log logs/advisor_autoplay_$PORT.log"
