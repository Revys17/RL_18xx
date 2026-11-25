#!/bin/bash
# A4 u1528's critic retrained on every seat's decisions, the policy frozen (prepare.py says how). Runs beside 1867's
# PG on all CPUs at nice 19, from the frozen deployment of master at the critic fix (43a04f8+; the main checkout's
# code is older and 1867's PG imports from it). Unit rl18xx-critic-all; `touch $W/$RUN_DIR/STOP` ends it after the
# current update. Restart resumes from state.json.
set -uo pipefail
D=${D:-/home/revys/longrun/deploy/critic-all-d8d95ac}
W=/home/revys/workspace/RL_18xx
RUN_DIR=model_checkpoints_generic_pg/u1528_critic_all_20261125
CPUS=${CPUS:-0-63}
cd $W || exit 1
export OMP_NUM_THREADS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ulimit -n "$(ulimit -Hn)"
[ -f $RUN_DIR/state.json ] || { echo "no $RUN_DIR/state.json: run prepare.py first"; exit 1; }
exec taskset -c "$CPUS" nice -n 19 $D/.venv/bin/python $D/main.py policy-gradient-generic --resume $RUN_DIR \
  --critic-seats all
