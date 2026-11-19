#!/bin/bash
# 1822CA on the GPU: supervised pretraining of the default-size generic model on the finished human games (base
# rules), then policy gradient from it with gen6's recipe adapted, and the eval rounds. Steps and environment:
# title_recipe.sh; the plan, the estimates and the queue: docs/rl_model.md "GPU run queue". NOT LAUNCHED.
#
#   bash scripts/launch/2026-11/bridge/title_1822CA.sh pretrain       # ~1.7 GPU h; <sl>/family_report.txt
#   nohup bash scripts/launch/2026-11/bridge/title_1822CA.sh pg > logs/gpg1822CA.log 2>&1 &   # resumes when re-run
#   bash scripts/launch/2026-11/bridge/title_1822CA.sh evals | evalloop
#   DRY=1 DRY_ROOT=<scratch> bash .../title_1822CA.sh pretrain critic pg     # the CPU dry run (tiny everything)
#
# Data: human_games/generic_1822CA_v1 (124 finished games, 145,928 / 12,896 positions; converted after the
# destination-hex and P / G train fixes). Starts: human_games/{start,midgame}_positions_1822CA_{n}p.jsonl (SR1-end
# starts, mid-game cuts SR2-SR6; every line replays on the current engine, checked 2026-11-14).
#
# gen6's recipe (scripts/launch/2026-11/gen6_20261110/run.sh) with these departures, each named:
# - Start: the generic 1822CA SL checkpoint; the critic from its value heads. League: the SL alone, then own
#   snapshots (every 50, the latest 4), PFSP per player count -- no other CA agents exist.
# - Player mix by the human games' shares at 3-6 players: 26 / 68 / 23 / 7 of 124 -> 3:21,4:55,5:19,6:5 (6 players:
#   10 human starts; most 6-player games start fresh or random), each count raised to >= 10% (title_recipe.sh
#   MIX_FLOOR, gen6 style) -> PG mix 3:19.9,4:52.1,5:18,6:10.
# - Decision caps 3,000 at every count (the family proposal: ~1.5x the longest self-play game, 2,017).
# - Starts as title_1822.sh (auction 0.5 = the fresh game, mid-game 0.3, random 0.2; all-pass openings redrawn).
# - Microbatch 256 (the largest title: ~53 MB of fp32 activations per position, ~69 MB at 1,000 legal tile lays),
#   server batches 512, games per task 24 (the longest games: ~1,300 decisions).
TITLE=1822CA
DATA_NAME=generic_1822CA_v1
PLAYER_MIX=${PLAYER_MIX:-3:21,4:55,5:19,6:5}
CAPS=3:3000,4:3000,5:3000,6:3000
MAX_DECISIONS=3000
MICROBATCH=256
INFER_BATCH=512
GAMES_PER_TASK=24
TITLE_MAX_UPDATES=1000  # backstop; the panel's plateau rule decides (docs/rl_model.md "GPU run queue")
EVAL_EVERY=${EVAL_EVERY:-100}            # as title_1822.sh: eval rounds every 100 updates,
PANEL_UPDATES=${PANEL_UPDATES:-100 250}  # the panel's frozen snapshots u100 / u250
source "$(dirname "${BASH_SOURCE[0]}")/title_recipe.sh"
title_main "$@"
