#!/bin/bash
# Value net v3 (item 8 of the agreed plan): once the policy-only data from gen4/800 is written
# (training_examples/value_v3, 400k games, value_v2's recipe), fine-tune value_v2 (value
# checkpoint 26, current_best of model_checkpoints_value) on it: value loss only, 10 epochs,
# lr 1e-4. Then score value_v2 and the new net with scripts/eval_value_head.py on held-out human
# games (by game stage) and on the new data's validation split.
cd "$(dirname "$0")/../../.."  # the repository root
LOG=logs/value_v3.log
say() { echo "$(date -Iseconds) $*" >> $LOG; }
until ! ps -eo args | grep -qE "[p]olicy-selfplay .*value_v3"; do sleep 120; done
say "value data done: $(tail -1 logs/policy_selfplay_value_v3.log | cut -c1-160)"
env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python main.py pretrain --model-dir model_checkpoints_value \
  --data-dir training_examples/value_v3 --epochs 10 --lr 1e-4 --policy-loss-weight 0 --price-loss-weight 0 \
  --no-value-refit > logs/value_net_v3.log 2>&1
say "value_v3 training exited $? ($(grep -E 'Saved pretraining checkpoint|value head:' logs/value_net_v3.log | tail -2 | cut -c40-260 | tr '\n' ' '))"
NEW=$(python3 -c "import json; print(json.load(open('model_checkpoints_value/AlphaZeroTransformer/current_best.json'))['checkpoint_num'])")
for ckpt in 26 $NEW; do
  .venv/bin/python scripts/eval_value_head.py --model-dir model_checkpoints_value --checkpoint 20261004_134558_804974562/$ckpt \
    > logs/eval_value_head_value_ckpt$ckpt.log 2>&1
  say "value checkpoint $ckpt on human games: $(grep -iE 'overall|all positions|winner' logs/eval_value_head_value_ckpt$ckpt.log | tail -3 | tr '\n' ' ' | cut -c1-300)"
  .venv/bin/python scripts/eval_value_head.py --model-dir model_checkpoints_value --checkpoint 20261004_134558_804974562/$ckpt \
    --lmdb training_examples/value_v3/validation > logs/eval_value_head_v3val_ckpt$ckpt.log 2>&1
  say "value checkpoint $ckpt on value_v3 validation: $(grep -iE 'overall|all positions|winner' logs/eval_value_head_v3val_ckpt$ckpt.log | tail -3 | tr '\n' ' ' | cut -c1-300)"
done
say "value_v3 done"
