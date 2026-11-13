#!/bin/bash
# The run (DESIGN.md) as unit rl18xx-pg2p (Restart=on-failure resumes from state.json).
set -uo pipefail
source /home/revys/longrun/experiments/pg2p_20261109/common.sh
cd $W
export OMP_NUM_THREADS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ulimit -n "$(ulimit -Hn)"
START="--policy model_checkpoints/AlphaZeroTransformer/20261004_134558_804974562/10.pth --value $CRITIC"
if [ -f $W/model_checkpoints_pg/$RUN/state.json ]; then
  exec $PY $D/main.py policy-gradient --resume model_checkpoints_pg/$RUN --run-name $RUN --out-dir model_checkpoints_pg $START
fi
[ -d $W/model_checkpoints_pg/$RUN ] && mv $W/model_checkpoints_pg/$RUN $W/model_checkpoints_pg/$RUN.partial-$(date +%Y%m%d%H%M%S)
exec $PY $D/main.py policy-gradient --run-name $RUN --out-dir model_checkpoints_pg $START --num-players $NP \
  ${TRAIN_ARGS:---workers 32 --games-per-task 96 --rows-per-update 65536 --max-updates $UPDATES} --positions-per-seat 32 \
  --population model_checkpoints_pg/populations/league_v7_gen4.json --opponent-slots 3 --pfsp-power 2.0 \
  --slot-refresh-every 2 --league-snapshot-every 50 --league-max-snapshots 4 --reward win --anchor-every 50 \
  --minibatch 2048 --microbatch 1024 --ppo-epochs 1 --clip 0.2 --lr 2e-5 --critic-lr 3e-5 \
  --kl-coef 0.02 --kl-target 0.08 --kl-coef-min 0.005 --entropy-coef 0.0 --learner-temperature 0.5 \
  --opponent-temperature 0.5 --opponent-price-eps 0.0 --gae-lambda 1.0 --snapshot-every 10 \
  --pool-refresh-every 5 --max-decisions $MAXDEC \
  --start-positions human_games/start_positions_1830_2p_v2.jsonl --random-start-fraction 0.2 \
  --midgame-positions human_games/midgame_positions_1830_2p_v2.jsonl --midgame-fraction 0.3 \
  --auction-fraction 0.5 --no-auction-unlock --auction-stall-decisions $STALL --save-game-every 20
