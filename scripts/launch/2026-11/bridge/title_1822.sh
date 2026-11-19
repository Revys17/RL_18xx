#!/bin/bash
# 1822 (the 1822+ rules mix) on the GPU: supervised pretraining of the default-size generic model on the finished human
# games of every rules combination, then policy gradient from it with gen6's recipe adapted, and the eval rounds.
# Steps and environment: title_recipe.sh; the plan, the estimates and the queue: docs/rl_model.md "GPU run queue".
# NOT LAUNCHED.
#
#   bash scripts/launch/2026-11/bridge/title_1822.sh pretrain         # ~4-4.5 GPU h; <sl>/family_report.txt
#   nohup bash scripts/launch/2026-11/bridge/title_1822.sh pg > logs/gpg1822.log 2>&1 &   # resumes when re-run
#   bash scripts/launch/2026-11/bridge/title_1822.sh evals | evalloop
#   DRY=1 DRY_ROOT=<scratch> bash .../title_1822.sh pretrain critic pg       # the CPU dry run (tiny everything)
#
# Run from a checkout with human_games/ (the main checkout), or set HG to its absolute path.
#
# Data: human_games/generic_1822_v2 (525 finished games of 10 rules combinations, 534,873 / 31,883 positions;
# docs/rl_model.md "1822+: the optional rules"). Starts: human_games/{start,midgame}_positions_1822_{n}p_v2.jsonl (the
# end of the opening bid-box Stock Round, mid-game cuts SR2-SR6; every combination, each line with its rules).
#
# gen6's recipe (scripts/launch/2026-11/gen6_20261110/run.sh) with these departures, each named:
# - Start: the generic 1822 SL checkpoint; the critic from its value heads. League: the SL alone, then own snapshots
#   (every 50, the latest 4), PFSP per player count -- no other 1822 agents exist.
# - Rules: every game draws its combination from the title's default mix (titles.TitleSpec.rules_mix: proportional
#   to the finished human games, 1822+ 77%, base 18%, tax haven alone 5%); a drawn combination without human starts
#   of the count falls through to a random opening under its rules.
# - Player mix by the human games' shares at 3-6 players: 149 / 254 / 98 / 24 of 525 -> 3:28,4:48,5:19,6:5, each
#   count raised to >= 10% (title_recipe.sh MIX_FLOOR, gen6 style) -> PG mix 3:26.5,4:45.5,5:18,6:10.
# - Decision caps 2,200 at every count (the 1822+ proposal: ~1.5x the longest self-play game, 1,493).
# - Starts: auction 0.5 = the fresh game (the opening Stock Round's bid boxes played by the policy, a seeded company
#   shuffle), mid-game 0.3, random opening 0.2 (SR1 played by uniform random moves) -> ~50% fresh games, ~15% mid-game,
#   ~7% random, ~28% human SR1-end starts. All-pass openings (the whole table passes in SR1: Ruby ends the game, every
#   seat tied) are redrawn (the default, the user's decision); auction stalls cannot happen (no Auction round).
# - Microbatch 256 (1822 positions hold ~40 MB of fp32 activations in training, 3.2x 1830's), server batches 512,
#   games per task 32 (games are ~2.3x 1830's in decisions and ~70x in generator CPU per decision).
TITLE=1822
DATA_NAME=generic_1822_v2
START_SUFFIX=_v2
RULES_MIX=default
PLAYER_MIX=${PLAYER_MIX:-3:28,4:48,5:19,6:5}
CAPS=3:2200,4:2200,5:2200,6:2200
MAX_DECISIONS=2200
MICROBATCH=256
INFER_BATCH=512
GAMES_PER_TASK=32
TITLE_MAX_UPDATES=1000  # backstop; the panel's plateau rule decides (docs/rl_model.md "GPU run queue")
# Eval rounds every 100 updates; the panel freezes the snapshots u100 and u250 (complete from u300, so the plateau rule
# can first call at u500). 1867 (cheaper updates): every 200, u200 / u500.
EVAL_EVERY=${EVAL_EVERY:-100}
PANEL_UPDATES=${PANEL_UPDATES:-100 250}
source "$(dirname "${BASH_SOURCE[0]}")/title_recipe.sh"
title_main "$@"
