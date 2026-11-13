#!/bin/bash
# gen5 (the user's plan, 2026-11-07): policy gradient from gen4 (rab_place_20261014 1800, learner + critic) under the
# current rules (the 2-OR train rule), gen4's recipe and league plus gen4 itself, with REAL auction training: half of
# the games start at the auction (gen4: 15%), played without the auction_unlock variant, and a game still in the
# auction after 150 decisions ends scored on net worth (a stall costs the seats that didn't buy). 1000 updates, then a
# check (vs gen4 2v2 + 1v3, auction prices vs humans). Deployment gen5-ae3438e. A restart resumes the run.
set -uo pipefail
D=/home/revys/longrun/deploy/az0-2p-7e21baa   # 2026-11-09: branch az0-2p-20261108 (7e21baa) = gen5-ae3438e + the audit fixes (tempered samplers, league crediting at game time); Ruby scoring of bankrupt players (the user: a feature)
W=/home/revys/longrun/work/lr1
RUN=gen5_20261107
cd $W
export OMP_NUM_THREADS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ulimit -n "$(ulimit -Hn)"
# A restart resumes from the last snapshot (state.json, every 10 updates). Audit 2026-11-08: the old branch passed
# --resume without its run directory (argparse exit 2) and no --policy/--value (unresolvable in this workdir).
# 2026-11-10 the user: keep going until it plateaus -> a resume runs to 4000 (a backstop); evals2.sh restarts the run
# when it stops at 1000 and touches STOP when the plateau rule fires (DESIGN.md).
if [ -f $W/model_checkpoints_pg/$RUN/state.json ]; then
  exec $D/.venv/bin/python $D/main.py policy-gradient --resume model_checkpoints_pg/$RUN --run-name $RUN \
    --out-dir model_checkpoints_pg --policy model_checkpoints_pg/rab_place_20261014/learner/1800.pth \
    --value model_checkpoints_pg/rab_place_20261014/critic/1800.pth --max-updates 4000
fi
# Crashed before its first snapshot: start over (the partial directory is kept aside).
[ -d $W/model_checkpoints_pg/$RUN ] && mv $W/model_checkpoints_pg/$RUN $W/model_checkpoints_pg/$RUN.partial-$(date +%Y%m%d%H%M%S)
exec $D/.venv/bin/python $D/main.py policy-gradient --run-name $RUN --out-dir model_checkpoints_pg \
  --policy model_checkpoints_pg/rab_place_20261014/learner/1800.pth \
  --value model_checkpoints_pg/rab_place_20261014/critic/1800.pth \
  --workers 32 --games-per-task 96 --learner-seats 2 --positions-per-seat 32 \
  --population model_checkpoints_pg/populations/league_v7_gen4.json --opponent-slots 3 --pfsp-power 2.0 \
  --slot-refresh-every 2 --league-snapshot-every 50 --league-max-snapshots 4 --reward placement --anchor-every 50 \
  --rows-per-update 65536 --minibatch 2048 --microbatch 1024 --ppo-epochs 1 --clip 0.2 --lr 2e-5 --critic-lr 3e-5 \
  --kl-coef 0.02 --kl-target 0.08 --kl-coef-min 0.005 --entropy-coef 0.0 --learner-temperature 0.5 \
  --opponent-temperature 0.5 --opponent-price-eps 0.0 --gae-lambda 1.0 --max-updates 1000 --snapshot-every 10 \
  --pool-refresh-every 5 --max-decisions 1000 \
  --start-positions human_games/start_positions_1830_4p_v2.jsonl --random-start-fraction 0.2 \
  --midgame-positions human_games/midgame_positions_1830_4p_v2.jsonl --midgame-fraction 0.3 \
  --auction-fraction 0.5 --no-auction-unlock --auction-stall-decisions 150 --save-game-every 20
