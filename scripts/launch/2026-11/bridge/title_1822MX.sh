#!/bin/bash
# 1822MX on the GPU: supervised pretraining of the default-size generic model on the finished human games (base
# rules), then policy gradient from it with gen6's recipe adapted, and the eval rounds. Steps and environment:
# title_recipe.sh; the plan, the estimates and the queue: docs/rl_model.md "GPU run queue". NOT LAUNCHED.
#
#   bash scripts/launch/2026-11/bridge/title_1822MX.sh pretrain       # ~1 GPU h; <sl>/family_report.txt
#   nohup bash scripts/launch/2026-11/bridge/title_1822MX.sh pg > logs/gpg1822MX.log 2>&1 &   # resumes when re-run
#   bash scripts/launch/2026-11/bridge/title_1822MX.sh evals | evalloop
#   DRY=1 DRY_ROOT=<scratch> bash .../title_1822MX.sh pretrain critic pg     # the CPU dry run (tiny everything)
#
# Data: human_games/generic_1822MX_v1 (229 finished games, 214,582 / 10,937 positions; docs/rl_model.md "1822
# family"). Starts: human_games/{start,midgame}_positions_1822MX_{n}p.jsonl (SR1-end starts, mid-game cuts SR2-SR6).
#
# gen6's recipe (scripts/launch/2026-11/gen6_20261110/run.sh) with these departures, each named:
# - Start: the generic 1822MX SL checkpoint; the critic from its value heads. League: the SL alone, then own
#   snapshots (every 50, the latest 4), PFSP per player count -- no other MX agents exist.
# - Players 3-5 (Ruby's range) by the human games' shares: 78 / 114 / 37 of 229 -> 3:34,4:50,5:16 (every count
#   above title_recipe.sh's 10% floor: the PG mix as is). Base rules (MX's
#   higher_cert_limit stays out: the user's decision of 2026-11-14).
# - Decision caps 1,650 at every count, RE-MEASURED 2026-11-14 after the NdeM fix (06afad3: the NdeM takes a new
#   token for an 11th+ station, so games that used to dead-end in Ruby's crash run to their end): 138 self-play games
#   of the family smoke's SL model at T 0.5 from the fresh game, 3-5 players, all finished, none dead-ended, longest
#   1,010 / 964 / 1,088 at 3 / 4 / 5 players -> ~1.5 x 1,088 (the old cap, 1,600, was 1.5x the pre-fix 990); T 1.0
#   (46 games) up to 1,419. docs/rl_model.md "GPU run queue".
# - Starts as title_1822.sh (auction 0.5 = the fresh game, mid-game 0.3, random 0.2; all-pass openings redrawn).
# - Microbatch 512 (~25 MB of fp32 activations per position), server batches 512, games per task 32.
TITLE=1822MX
DATA_NAME=generic_1822MX_v1
PLAYER_MIX=${PLAYER_MIX:-3:34,4:50,5:16}
CAPS=3:1650,4:1650,5:1650
MAX_DECISIONS=1650
MICROBATCH=512
INFER_BATCH=512
GAMES_PER_TASK=32
TITLE_MAX_UPDATES=1000  # backstop; the panel's plateau rule decides (docs/rl_model.md "GPU run queue")
EVAL_EVERY=${EVAL_EVERY:-100}            # as title_1822.sh: eval rounds every 100 updates,
PANEL_UPDATES=${PANEL_UPDATES:-100 250}  # the panel's frozen snapshots u100 / u250
source "$(dirname "${BASH_SOURCE[0]}")/title_recipe.sh"
title_main "$@"
