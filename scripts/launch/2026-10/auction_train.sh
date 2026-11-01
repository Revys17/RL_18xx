#!/bin/bash
# Auction learning (search_learning --auction-only; docs/search_learning_design.md "Auction learning"):
# episodes are the waterfall auction alone under the real rules, every decision searched (200, value-
# driven PUCT as lr1 plays), scored by lr1's value v3 (pinned) at the first Stock Round. The policy
# starts from the supervised checkpoint 10 (KL-anchored to it), the critic from v3 itself. Half the
# games seat 1-2 league opponents (auction_league.json). Needs the GPU to itself: lr1 paused first.
# Runs as a transient user unit under a memory cap; STOP file: model_checkpoints_search/auc1/STOP.
#   scripts/launch/2026-10/auction_train.sh [run name]        # journalctl --user -u rl18xx-auction-train
cd "$(dirname "$0")/../../.."  # the repository root
RUN=${1:-auc1}
LR1=${LR1:-/home/revys/longrun/work/lr1}
SL10=$LR1/model_checkpoints/AlphaZeroTransformer/20261004_134558_804974562/10.pth
V3=$LR1/model_checkpoints_value/lr1/promoted/3.pth
mkdir -p logs
systemd-run --user --unit="rl18xx-auction-train-$RUN" --same-dir -p MemoryMax=120G -p LimitNOFILE=1048576 \
  -p StandardOutput=append:"$PWD/logs/auction_train_$RUN.log" -p StandardError=append:"$PWD/logs/auction_train_$RUN.log" \
  --setenv=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True --setenv=OMP_NUM_THREADS=2 \
  .venv/bin/python main.py search-learning --run-name "$RUN" --out-dir model_checkpoints_search \
  --auction-only --terminal-value "$V3" --policy "$SL10" --value "$V3" --auction-prefix-decisions 4 \
  --target cq --c-puct 0.05 --p-full 1.0 --n-full 200 --n-fast 0 --prior-temperature 1 \
  --reuse-tree total --dedupe-maps --server-replicas 2 --full-move-temperature 1 --explore-decisions 0 \
  --workers 40 --games-per-worker 32 --microbatch 1024 --minibatch 2048 --steps-per-update 16 \
  --rows-per-update 16384 --min-window-rows 16384 --window-rows 131072 --score-loss-weight 0 \
  --kl-target 0.08 --anchor-every 0 --snapshot-every 10 --max-updates 600 \
  --population scripts/launch/2026-10/auction_league.json --league-share 0.5
echo "auction learning $RUN: unit rl18xx-auction-train-$RUN, log logs/auction_train_$RUN.log, run model_checkpoints_search/$RUN"
