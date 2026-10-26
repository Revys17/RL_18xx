#!/bin/bash
# Install the long run lr1 (docs/long_run_recipe.md, docs/longrun_runbook.md "The long run (2026-11)"): the three
# long-lived instances and the daily jobs, under the user's systemd. Writes config only (install.sh); starts nothing.
#
#   bash <deployment>/scripts/launch/2026-11/install_lr1.sh <deployment> <workdir>
#
#   lr1-gen     the generator: main.py search-learning, phase 1 (checkpoint 10 frozen, the value trainer's
#               promoted value followed, league seats, every game saved)
#   lr1-value   the value trainer: main.py value-train (cycles over the saved games, the promotion gate)
#   lr1-ladder  the eval ladder: scripts/eval_ladder.py run --panel eval_ladder_longrun.json
#   lr1-daily   the daily jobs at 06:00 (retention, style statistics, integrity check, status report)
#
# Paths below are relative to the workdir, which needs (runbook): human_games -> the repository's; real copies of
# the panel / league checkpoints at the same relative paths (model_checkpoints/..., model_checkpoints_pg/...,
# model_checkpoints_search/az8_20261019, azl_20261019); logs/value_gate/{data,runs/base_36989} (the fixed held-out
# set and the start value). Names (lr1, saved_games/lr1, model_checkpoints_value/lr1, logs/eval/ladder_lr1) must
# match eval_ladder_longrun.json and retention.json.
set -euo pipefail

D=$(realpath "${1:?deployment}")
W=$(realpath "${2:?workdir}")
INSTALL=$D/scripts/longrun/install.sh
L=$D/scripts/launch/2026-11
CKPT10=model_checkpoints/AlphaZeroTransformer/20261004_134558_804974562/10.pth
VALUE_RUN=model_checkpoints_value/lr1
GEN_RUN=model_checkpoints_search/lr1
SAVE=saved_games/lr1
LADDER_ROOT=logs/eval/ladder_lr1
ALERTS=()  # e.g. ALERTS=(--set 'ALERT_COMMAND=curl -s -d "$ALERT_MESSAGE" https://ntfy.sh/<topic>')

# The generator. START_ARGS: the first start only (what a resume keeps from state.json); TRAIN_ARGS: every start
# (the settings a resume may change, and the rest so a resume can't silently drop one).
bash "$INSTALL" lr1-gen --kind generator --deploy "$D" --workdir "$W" --run-dir "$GEN_RUN" \
    --start-args "--policy $CKPT10 --value-follow $VALUE_RUN --freeze-policy --league-snapshot-slots 6" \
    --train-args "--max-updates 10000000 --workers 40 --games-per-worker 32 --microbatch 1024 \
--target cq --c-puct 0.05 --p-full 0.25 --n-full 200 --n-fast 48 --prior-temperature 1 \
--reuse-tree total --dedupe-maps --server-replicas 2 --full-move-temperature 1 --explore-decisions 120 --explore-from-game-start \
--midgame-positions human_games/midgame_positions_1830_4p_v2.jsonl --midgame-fraction 0.3 --auction-fraction 0 \
--population $L/league_longrun.json --league-share 0.3 --league-readouts 48 --league-snapshot-hours 168 \
--save-data $SAVE" \
    --memory-high 120G --memory-max 135G --restart-sec 120 \
    --set STALL_MINUTES=90 --set DISK_MIN_GB=100 --set HOST_CHECKS=1 --set USAGE_LOG=longrun/usage.jsonl \
    --set "USAGE_PATHS=saved=$SAVE;checkpoints=$GEN_RUN:$VALUE_RUN;evals=logs/eval;logs=logs/*.log:longrun" \
    --set WORKTREES_DIR=/home/revys/workspace/RL_18xx/.claude/worktrees "${ALERTS[@]}"

# The value trainer: the same arguments on every start (it resumes its run directory; v0 = --init). Since
# 2026-10-26 half of every batch comes from the value gate's training games and the gain is required on the eval
# ladder's games of u0 (logs/value_gate/ladder_lr1_u0: scripts/value_gate.py build --root logs/eval/ladder_lr1
# --since "2026-10-23 00:00" --test "*"), not on the generator's held-out games (docs/long_run_recipe.md).
bash "$INSTALL" lr1-value --kind value --deploy "$D" --workdir "$W" --run-dir "$VALUE_RUN" \
    --train-args "--data $SAVE --init logs/value_gate/runs/base_36989/learner/0.pth --fixed-data logs/value_gate/data \
--anchor-data logs/value_gate/data --anchor-share 0.5 --gate-data logs/value_gate/ladder_lr1_u0 --workers 8" \
    --memory-high 36G --memory-max 44G --restart-sec 120 \
    --set STALL_MINUTES=120 --set GPU_CHECK=0 --set HOST_CHECKS=0 "${ALERTS[@]}"

# The eval ladder (~15% of the GPU: 12 workers). It resumes its candidate from its state.json.
bash "$INSTALL" lr1-ladder --kind ladder --deploy "$D" --workdir "$W" --run-dir "$LADDER_ROOT" \
    --panel "$L/eval_ladder_longrun.json" --train-args "" \
    --memory-high 26G --memory-max 32G --restart-sec 120 \
    --set STALL_MINUTES=90 --set GPU_CHECK=0 --set HOST_CHECKS=0 "${ALERTS[@]}"

# The daily jobs (CPU only). M7_HOLD_CHECK=0: the driver and kernel are not held (the user's decision, 2026-10-26:
# updates stay on; a driver upgrade shows as the watchdog's nvidia_mismatch and is fixed by a reboot).
bash "$INSTALL" lr1-daily --kind daily --deploy "$D" --workdir "$W" --memory-high 12G --memory-max 16G --nice 10 \
    --set STATE_DIR=longrun/lr1-daily --set GEN_RUN_DIR="$GEN_RUN" --set VALUE_RUN_DIR="$VALUE_RUN" \
    --set LADDER_PANEL="$L/eval_ladder_longrun.json" --set SAVE_DATA="$SAVE" \
    --set WATCH_DIRS=longrun/lr1-gen:longrun/lr1-value:longrun/lr1-ladder \
    --set RETENTION_CONFIG="$L/retention.json" --set REPORT_LOG=long_run_log.md \
    --set USAGE_LOG=longrun/usage.jsonl \
    --set "USAGE_PATHS=saved=$SAVE;checkpoints=$GEN_RUN:$VALUE_RUN;evals=logs/eval;logs=logs/*.log:longrun" \
    --set WORKTREES_DIR=/home/revys/workspace/RL_18xx/.claude/worktrees --set M7_HOLD_CHECK=0 "${ALERTS[@]}"

cat <<'NEXT'

Next (docs/longrun_runbook.md, "The long run (2026-11)"):
  systemctl --user start rl18xx-longrun@lr1-value      # first: it writes promoted/0.pth (v0) for the others
  systemctl --user start rl18xx-longrun@lr1-gen
  systemctl --user start rl18xx-longrun@lr1-ladder     # day 0: u0 against the whole panel (the baseline)
  systemctl --user enable --now rl18xx-longrun-daily@lr1-daily.timer
  systemctl --user enable rl18xx-longrun@lr1-gen rl18xx-longrun@lr1-value rl18xx-longrun@lr1-ladder  # back after a reboot
NEXT
