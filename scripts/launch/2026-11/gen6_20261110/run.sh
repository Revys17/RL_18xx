#!/bin/bash
# gen6 (DESIGN.md): policy gradient from gen5's final checkpoint at every player count, gen5's recipe otherwise.
# NOT LAUNCHED. At launch: set D (the deployment of branch gen6-mixed-count), G5 (gen5's final update), create the
# league file (DESIGN.md), then run as unit rl18xx-gen6 (Restart=on-failure resumes from state.json).
set -uo pipefail
D=${D:?set D to the gen6 deployment (scripts/longrun/deploy.sh create <tag> /home/revys/longrun/deploy/gen6-<sha>)}
G5=${G5:?set G5 to gen5s final update (e.g. 1400)}
W=/home/revys/longrun/work/lr1
RUN=gen6_20261110
POP=model_checkpoints_pg/populations/league_v8_gen5.json
cd $W
export OMP_NUM_THREADS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ulimit -n "$(ulimit -Hn)"
START="--policy model_checkpoints_pg/gen5_20261107/learner/$G5.pth --value model_checkpoints_pg/gen5_20261107/critic/$G5.pth"
if [ -f $W/model_checkpoints_pg/$RUN/state.json ]; then
  exec $D/.venv/bin/python $D/main.py policy-gradient --resume model_checkpoints_pg/$RUN --run-name $RUN \
    --out-dir model_checkpoints_pg $START
fi
[ -d $W/model_checkpoints_pg/$RUN ] && mv $W/model_checkpoints_pg/$RUN $W/model_checkpoints_pg/$RUN.partial-$(date +%Y%m%d%H%M%S)
exec $D/.venv/bin/python $D/main.py policy-gradient --run-name $RUN --out-dir model_checkpoints_pg $START \
  --player-mix 2:15,3:20,4:35,5:20,6:10 \
  --max-decisions 1200 --max-decisions-by-players 2:1000,3:1100,5:1300,6:1500 \
  --workers 32 --games-per-task 96 --positions-per-seat 32 \
  --population $POP --opponent-slots 3 --pfsp-power 2.0 \
  --slot-refresh-every 2 --league-snapshot-every 50 --league-max-snapshots 4 --reward placement --anchor-every 50 \
  --rows-per-update 65536 --minibatch 2048 --microbatch 1024 --ppo-epochs 1 --clip 0.2 --lr 2e-5 --critic-lr 3e-5 \
  --kl-coef 0.02 --kl-target 0.08 --kl-coef-min 0.005 --entropy-coef 0.0 --learner-temperature 0.5 \
  --opponent-temperature 0.5 --opponent-price-eps 0.0 --gae-lambda 1.0 --max-updates 4000 --snapshot-every 10 \
  --pool-refresh-every 5 --random-start-fraction 0.2 --midgame-fraction 0.3 \
  --auction-fraction 0.5 --no-auction-unlock --auction-stall-decisions 150 --save-game-every 20
