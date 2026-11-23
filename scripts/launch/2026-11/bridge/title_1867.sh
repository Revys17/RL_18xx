#!/bin/bash
# 1867 on the GPU: supervised pretraining of the default-size generic model on the finished human games, then policy
# gradient from it with gen6's recipe adapted, and the eval rounds. Steps and environment: title_recipe.sh; the plan,
# the estimates and the queue: docs/rl_model.md "GPU run queue". SL + critic done 2026-11-18/19; PG queued 2026-11-22
# behind 1830's plateau run with A4's settings (title_recipe.sh), run gpg1867_a4_20261122
# (/home/revys/longrun/experiments/gpg1867_20261122/queue.sh).
#
#   bash scripts/launch/2026-11/bridge/title_1867.sh pretrain         # ~1.5 GPU h; <sl>/family_report.txt
#   nohup bash scripts/launch/2026-11/bridge/title_1867.sh pg > logs/gpg1867.log 2>&1 &   # resumes when re-run
#   bash scripts/launch/2026-11/bridge/title_1867.sh evals            # one eval round of the newest u%200 snapshot
#   bash scripts/launch/2026-11/bridge/title_1867.sh evalloop         # every 200 updates until the run stops
#   DRY=1 DRY_ROOT=<scratch> bash .../title_1867.sh pretrain critic pg       # the CPU dry run (tiny everything)
#
# Run from a checkout with human_games/ (the main checkout), or set HG to its absolute path.
#
# Data: human_games/generic_1867_finished_v1 (840 finished games, 508,051 / 29,311 positions; A3). Starts:
# human_games/{start,midgame}_positions_1867_{n}p.jsonl (SR1 after the human auction; mid-game cuts SR2-SR6).
#
# The recipe = gen6's (scripts/launch/2026-11/gen6_20261110/run.sh) with these departures, each named:
# - Start: the generic 1867 SL checkpoint (gen6: gen5); the critic starts from the `critic` step's 1-epoch run (its value
#   heads before they memorize; no 1867 critic exists).
# - League: the SL checkpoint at T 1.0 and 0.5 (each pinned at 12.5%: 1830's arm A1, title_recipe.sh SL_SEATS) at
#   first, then the learner's own snapshots (every 50, the latest 4), PFSP per player count (CountLeague) -- no other
#   1867 agents exist (gen6: a 33-member league). Starts by prioritized level replay (1830's arm A2, START_PRIORITY).
# - Player mix by the human games' shares at 3-6 players: 3p 205, 4p 509, 5p 109, 6p 17 of 840 finished games ->
#   3:24,4:61,5:13,6:2 (gen6: a flattened 2:15,3:20,4:35,5:20,6:10), each count raised to >= 10% of the games
#   (title_recipe.sh MIX_FLOOR, 2026-11-15, gen6 style) -> PG mix 3:22,4:56,5:11.9,6:10 (MIX_FLOOR=0: the human shares).
# - Decision caps 3:1200,4:1200,5:1400,6:1500 (A3's proposal: ~1.5x the longest T 1.0 game, ~1.7-2.4x the humans').
# - Auction stalls off (--auction-stall-decisions 0): 1867's single-item auction cannot lock (A3); auction_unlock
#   is 1830's.
# - Starts: auction (the opening) 0.5, mid-game 0.3, random opening 0.2 as gen6 -> ~50% openings, ~15% mid-game,
#   ~7% random, ~28% human SR1 starts (A3's outline had auction 0.2).
# - Learner server 2 replicas; games per task 64 (gen6: 96; 1867 games are ~1.2x 1830's).
TITLE=1867
DATA_NAME=generic_1867_finished_v1
PLAYER_MIX=${PLAYER_MIX:-3:24,4:61,5:13,6:2}
CAPS=3:1200,4:1200,5:1400,6:1500
MAX_DECISIONS=1500
MICROBATCH=1024   # ~15 MB of fp32 activations per position (1830: 12.4): ~15 GB a microbatch
INFER_BATCH=1024
GAMES_PER_TASK=64
TITLE_MAX_UPDATES=3000  # backstop; the panel's plateau rule decides (docs/rl_model.md "GPU run queue")
source "$(dirname "${BASH_SOURCE[0]}")/title_recipe.sh"
title_main "$@"
