#!/bin/bash
# 1822PNW on the GPU: supervised pretraining of the default-size generic model on the finished human games (base
# rules), then policy gradient from it with gen6's recipe adapted, and the eval rounds. Steps and environment:
# title_recipe.sh; the plan, the estimates and the queue: docs/rl_model.md "GPU run queue". NOT LAUNCHED.
#
#   bash scripts/launch/2026-11/bridge/title_1822PNW.sh pretrain      # ~2.5 GPU h; <sl>/family_report.txt
#   nohup bash scripts/launch/2026-11/bridge/title_1822PNW.sh pg > logs/gpg1822PNW.log 2>&1 &   # resumes when re-run
#   bash scripts/launch/2026-11/bridge/title_1822PNW.sh evals | evalloop
#   DRY=1 DRY_ROOT=<scratch> bash .../title_1822PNW.sh pretrain pg    # the CPU dry run (tiny everything)
#
# Data: human_games/generic_1822PNW_v1 (417 finished games, 413,741 / 23,293 positions; docs/rl_model.md "1822
# family"). Starts: human_games/{start,midgame}_positions_1822PNW_{n}p.jsonl (SR1-end starts, mid-game cuts SR2-SR6).
#
# gen6's recipe (scripts/launch/2026-11/gen6_20261110/run.sh) with these departures, each named:
# - Start: the generic 1822PNW SL checkpoint; the critic from its value heads. League: the SL alone, then own
#   snapshots (every 50, the latest 4), PFSP per player count -- no other PNW agents exist.
# - Players 3-5 (Ruby's range) by the human games' shares: 150 / 204 / 63 of 417 -> 3:36,4:49,5:15. Base rules (PNW's
#   options stay out: the user's decision of 2026-11-14).
# - Decision caps 1,800 at every count (the family proposal: ~1.5x the longest self-play game, 1,209).
# - Starts as title_1822.sh (auction 0.5 = the fresh game, mid-game 0.3, random 0.2; all-pass openings redrawn).
# - Microbatch 512 (~29 MB of fp32 activations per position), server batches 512, games per task 32.
TITLE=1822PNW
DATA_NAME=generic_1822PNW_v1
PLAYER_MIX=3:36,4:49,5:15
CAPS=3:1800,4:1800,5:1800
MAX_DECISIONS=1800
MICROBATCH=512
INFER_BATCH=512
GAMES_PER_TASK=32
TITLE_MAX_UPDATES=1000  # backstop; the panel's plateau rule decides (docs/rl_model.md "GPU run queue")
EVAL_EVERY=${EVAL_EVERY:-100}            # as title_1822.sh: eval rounds every 100 updates,
PANEL_UPDATES=${PANEL_UPDATES:-100 250}  # the panel's frozen snapshots u100 / u250
source "$(dirname "${BASH_SOURCE[0]}")/title_recipe.sh"
title_main "$@"
