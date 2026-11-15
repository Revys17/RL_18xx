# GPU runbook (prepared 2026-11-15, for when the GPU is free)

Everything below is built, tested and dry-run on CPU; nothing has run on the GPU, so every time is an estimate.
Run from the main checkout (`/home/revys/workspace/RL_18xx`, it has `human_games/`), after `uv sync`. Long steps under
`nohup` / `systemd-run`. CPU pinning: the scripts default to cores 24-31,56-63 (`CPUS`, `WORKERS` override) — ask the
user whether a run may use the whole machine (the 1822 family's PG is CPU-bound: ~2x faster with all CPUs).

Background: `docs/rl_model.md` (the generic model and pipeline), `docs/bot_weaknesses_2026_11.md` (why step 3),
`docs/robustness_test_plan.md` + `docs/robustness_measurement.md` (step 3's design and instruments).

## The queue

| # | Step | Command | GPU h (est.) | Gate / output |
|---|---|---|---|---|
| 1 | **1830 check of the generic model**: SL on SL ckpt 10's games, compare, head-to-heads vs ckpt 10 | `bash scripts/launch/2026-11/bridge/gpu_check_1830.sh pretrain compare eval` | 6-7 | Within noise of ckpt 10 on per-family top-1 and the 4 head-to-heads. Its `final.pth` is the start of every 1830 run below. Watch the first log lines' `elapsed` for the real rate. |
| 1b | **Smoke the generic search/serve path on the GPU** (never run on CUDA) | a 24-game `eval_head_to_head.py` match: `generic:<1830 SL>+generic:<1830 SL>#score@100/...` vs gen6 u2400+critic (syntax: `docs/rl_model.md` "Head-to-head evals of generic searches"), with `RL18XX_SERVER_PROFILE=60` | 0.2 | Games finish; server throughput vs the estimate. |
| 2 | **Step 0: search settings on gen6** (win vs score vs blend; prior T 0.5 vs 1.0) + the **gen6 search-value check** (critic vs lr1 v4 vs policy-only) | `bash scripts/robustness/search_settings.sh` (m1 on CPU + h2h) and `bash scripts/launch/2026-11/bridge/search_value_gen6.sh` | 6-8 + 3-3.5 (`GAMES=240` halves) | Which head / temperature the live bots keep (today: score head, T 0.5); whether search beats policy-only for gen6. M1 baselines are already measured on CPU (win 0.19, score 0.47, SL ckpt 10 policy 0.71). |
| 3 | **Robustness arms on generic 1830** (first pass, 300 updates each): A0 baseline, A1 human-imitation seats (+ `REF_KL=0.02` variant), A2 self-prioritized starts, A3 main exploiters (refresh 50/50), A4 combined, A5p practice positions | per arm: `nohup bash scripts/launch/2026-11/bridge/robust_arm.sh <arm> pg > logs/robust_<arm>.log 2>&1 &` plus `... <arm> evalloop` (and `... A3 exploiters` for A3); second pass: `RESUME_MAX_UPDATES=600 bash ... <arm> pg` | ~45-55 (first pass incl. eval rounds) + ~15 (second pass of two) | Pre-set rule (test plan): adopt an arm that improves M1 (regression set) and M3 (exploiter panel) beyond noise without losing M4 (out-of-family panel); M2 (behaviour stats) toward the human corpus. Status per arm: `rounds.txt`, `robust_status.txt`. |
| 4 | **Carry the winning 1830 recipe** to plateau; the user plays it on the site | the adopted arm to plateau; then `scripts/launch/2026-11/bridge/swap_bots_generic.sh <deploy> <learner> <critic>` (dry-run by default; `--apply` only with the user's OK) | depends | M5: the user's games, re-analysed with `scripts/robustness/behaviour_stats.py`. |
| 5 | **1867**: SL, then PG with the winning recipe | `bash scripts/launch/2026-11/bridge/title_1867.sh pretrain`, then `nohup ... title_1867.sh pg ... &` and `... evalloop` (fold step 3's arm flags into `title_recipe.sh` first) | ~1.5 + ~34 (1,000 updates) | `family_report.txt`; the panel's plateau rule (`panel_status.py`). The SL step doesn't depend on step 3 and can run any time the GPU is idle. |
| 6 | **1822 family**: SL for 1822 (1822+ rules mix), PNW, MX, CA; then PG, a first pass to 300 updates each | `title_1822.sh` / `title_1822PNW.sh` / `title_1822MX.sh` / `title_1822CA.sh` `pretrain`, then `pg` + `evalloop` | ~9 + ~150 (all CPUs) / ~350 (pinned cores) | As step 5. CA may need `checkpoint_layers` if memory is tight (`docs/rl_encoder.md` / `rl_model.md` model fit). |

Total: roughly 270-300 GPU h with all CPUs (11-12 days), much more on the pinned cores.

## Settings decided (change only with the user)

- Bots: gen6 u2400 + critic 2400, placement-score head (`--play-value-head score`), deployment advisor-20261114.
- Player-count floor 10% in every title's PG mix (`MIX_FLOOR`).
- Unfinished games dropped from new titles' supervised data (1830 keeps ckpt 10's corpus for the check).
- 1822 trains on the 1822+ rules mix (proportional to finished human games); all-pass openings redrawn; the NdeM
  takes extra tokens (printed rules 5.12.6).
- A1: SL at T 1.0 and 0.5, 12.5% each, at every count; A3: exploiters refreshed every 50 main updates.

## Known gaps / watch items

- The generic inference server, generic search and generic PG have never run on CUDA: step 1b first.
- Start files are read lazily (one line per draw; a mixed-count worker holds ~10 MB of start data instead of ~1.35 GB
  with the SR 2-10 cuts and the practice library; same draws for the same seed).
- M1 has 8-21 positions per weakness: one position moves a pass rate 5-12 points; M3/M4 carry the statistics.
- The site only plays 1830; 1867 / 1822 bots need the site and the advisor's live game to support those titles.
