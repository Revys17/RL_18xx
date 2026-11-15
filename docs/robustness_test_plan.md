# Test plan: making the bots robust to human tactics (2026-11-14)

Written while the GPU is on another project; to start once it is free. The problem and its evidence:
`docs/bot_weaknesses_2026_11.md` (Caleb won 7 of 11 finished games against the bots: uncontested cheap
privates, loot-and-dump, sell-outs that hand over treasuries, doomed buy-ins by losing bots, suitcasing).
Priorities from the user: weaknesses 1 (auction) and 2 (loot-and-dump) first.

Already done: the live bots search the critic's placement-points head (`--play-value-head score`,
2026-11-14 18:32, deployment advisor-20261114) — aimed at weakness 4.

## Questions this plan answers

1. Which training changes make the bots stop falling for the five tactics, without losing strength?
2. Can the bots learn the defences "naturally" (human-like opponents, self-chosen hard positions,
   exploiters with the plain reward), or do some tactics need targeted training?
3. Do search settings (value head, prior temperature) fix part of it with no retraining?

## Measurement first (build before any arm runs)

Every arm is judged on the same four instruments, all policy-only or with the live bots' search settings.
Built 2026-11-15 (M1-M3, Step 0; usage, baselines, costs): `docs/robustness_measurement.md`, `scripts/robustness/`.

| | Instrument | What it measures | Built from |
|---|---|---|---|
| M1 | **Regression set** | Per weakness, positions from Caleb's games just before the bots' mistakes (e.g. game 15 [402] buying NYC, game 21 [2] the CA/MH opening, game 21 [408] selling NYC with $900, game 19 [297] buying into a soon-looted company); for each: the policy's probability of the bad move, the search's choice (400 readouts, the bots' settings), and the value's estimate of the danger (does it see the dump coming). Pass = the bad move is no longer chosen and its probability falls. | `scratchpad/botgames/` of session 48dd27c6 (timelines with site action ids; `probe.py`) -> a script in `scripts/` + a fixture file |
| M2 | **Behaviour stats** | In each arm's eval games and in games against human-like seats: contested vs uncontested private prices (CA, MH, DH, CS), loot-then-dump rate and victim bankruptcies, sell-outs with >= $300 treasury, buys into trainless broke companies, suitcase rate (companies ending a turn trainless for 2+ ORs). Compared with the human corpus and gen6. | `events.py` / `corpus.py` / `compare.py` in the same scratch folder -> `scripts/behaviour_stats.py` |
| M3 | **Exploiter panel** | The bot alone (1v3) against three copies of each: the human-imitation SL model (T 1.0 and 0.5), gx2_bankrupt, the new main exploiters (below). Robustness = the bot's share against each. | existing exploiters + those trained in arm A3 |
| M4 | **Strength guard** | The 4-player out-of-family panel (gen4, SL ck10, gen2, gx2_rust, gx2_bankrupt; 800 games each) and the other counts vs gen4, as for gen6. An arm may not lose more than noise here. | `outfamily.sh` (gen6) |
| M5 | **Live check** | Caleb plays the winning arm on the site; the analysis repeated on those games. | the site + the analysis scripts |

Pre-set decision rule: an arm is adopted if it improves M1 and M3 beyond noise (2x the seed / game-sample spread)
without losing M4 beyond noise; M2 should move toward the human corpus. Arms that tie are broken by M2.

## Step 0: search settings, no training (~3-4 GPU h, overlaps the gen6 search-value check)

On gen6 u2400 + critic 2400 at 400 readouts: win head vs placement head vs `blend=0.5`; prior temperature 0.5
(today) vs 1.0 (lets search overrule the sharp policy more often). Measured on M1 (cheap) and a 4p head-to-head
(480 games each). Tells us how much of weaknesses 3-4 is search, and gives the search settings for M1/M3 below.

## Step 1: the arms (1830 on the generic model)

All arms start from the generic 1830 SL checkpoint of the GPU check (`gpu_check_1830.sh`) and use gen6's recipe
(`pg_generic_1830.sh`: placement reward, mixed 2-6 counts, per-count PFSP, KL to a moving anchor, learner T 0.5),
the same seed set, and the same update budget: a first pass of 300 updates each, then the promising ones to 600.

| Arm | Change | Mechanism ("natural" vs targeted) |
|---|---|---|
| A0 | Baseline: gen6's recipe on the generic model | — |
| A1 | **Human-imitation seats**: the SL checkpoint is a league member with a pinned share (e.g. 25% of opponent seats, every count), played at T 1.0 (its human-like spread) as well as 0.5. Optionally a KL pull toward the SL policy on top of the anchor (piKL-style). | natural: human tactics come from human data (looting ~0.16/game, suitcasing ~1.1/game) |
| A2 | **Self-prioritized starts** (prioritized level replay over the human mid-game library): each start position is scored by the learner's regret there (e.g. mean absolute advantage, or the critic's error, over the games started from it) and replayed in proportion; a share of starts stays uniform. | natural: the run picks the positions it handles worst |
| A3 | **Main exploiters**: from the current main, against a frozen copy of it, the plain placement reward, higher temperature / entropy and no KL pull; refreshed every ~100 main updates; each joins the league (PFSP decides how often it is drawn). Their games are mined (M2) to see which tactics they discover. | natural: they find whatever beats the main |
| A4 | A1 + A2 (+ A3 if it helped alone) | combined |
| A5 (only if 1 or 2 remain) | Targeted: for the auction, human-auction seats (SL bidding only during the auction) or a value trained on search-played auctions (lr1's v4, trained on search games, was the one value that rated CA/MH bids correctly); for loot-and-dump, a looter exploiter rewarded for other players' forced train contributions and bankruptcies | targeted fallback |

Practice positions from the human corpus (~300 loot-and-dump situations, plus Caleb's) are not an arm on their
own: A2's library includes them, and A2 decides how often they matter.

## Step 2: confirm and carry over

- The adopted recipe runs to plateau on 1830 (judged by M4 + M3), then Caleb plays it (M5).
- The same recipe is used for 1867 and the 1822 family (their GPU queue, `docs/rl_model.md` "GPU run queue"):
  M3/M4 per title use each title's SL checkpoint and frozen snapshots; M1 exists only for 1830 until humans play
  those titles against the bots.

## Compute (estimates, not measured)

- Step 0: ~3-4 GPU h.
- Step 1: ~80-90 s per update for generic 1830 PG (C2's estimate) -> ~7 h per 300-update arm; A0-A4 first pass
  ~35-40 h, exploiters in A3 add ~5 h per refresh; second pass of two arms to 600 updates ~15 h.
  Total ~60-70 GPU h after the 1830 GPU check (~6-7 h).
- Per arm, with the scripts as built (`docs/rl_model.md` "Robustness mechanisms", "GPU estimates per arm"): A0 / A2
  ~7.3 h, A1 ~7.3 h (~7.9 h with the reference KL), A3 ~12 h (the main + two exploiters sharing the GPU), A4 ~7.9 h
  (~12.5 h with A3): first pass ~42-47 GPU h.

## Work before the GPU (CPU)

1. M1: the regression-set builder and scorer (positions + labels from Caleb's games; scored for any checkpoint and
   search setting).
2. M2: the behaviour-stats script in the repo (from the analysis scripts), runnable on eval / PG saved games.
3. A1: pinned share + per-member temperature for league members in `policy-gradient-generic` (check what CountLeague
   pins already give); optional KL toward the SL policy.
   **Built 2026-11-15** (details: `docs/rl_model.md` "Robustness mechanisms"): CountLeague's `min_share` /
   `min_share_players` pins already gave a pinned share per count; added a population entry's `"temperature"` (member
   key `<path>@<T>`: the SL can be a member at T 1.0 and at 0.5) and `"min_share_by_players"`; `--ref-kl-coef` /
   `--ref-checkpoint` / `--ref-temperature`: a fixed piKL-style KL toward the SL on top of the anchor (default off).
4. A2: a prioritized start sampler in the generic start-position code (regret estimates from the run's own rows).
   **Built**: `--start-priority`, prioritized level replay over the human SR1 starts and mid-game cuts per count
   (score = mean |reward - V(s)| of the learner's rows from that start, i.e. the critic's error; rank-based with
   staleness 0.3, 30% uniform; the start-kind mix unchanged); scores persist in `start_priorities.json`, the top starts
   are in the history.
5. A3: main-exploiter mode in `policy-gradient-generic` (fixed frozen opponent, higher T, no KL) + a refresh loop.
   **Built**: `--exploiter-of <ckpt>` (the frozen main in every other seat), `--population-reload-every` (a running
   main picks up members added to its league file), `scripts/launch/2026-11/bridge/exploiter_loop.py` (every 100 main
   updates an exploiter from that snapshot, 100 updates, joins pinned at 10%, the newest 2 keep their pins).
6. Dry-run every arm on CPU at tiny size, as the title recipes were.
   **Done**: `scripts/launch/2026-11/bridge/robust_arm.sh <A0..A4>` (`--seed 1830`, 300 updates, `pg_generic_1830.sh`'s
   evals + 4p 1v3 vs SL@1.0 and vs each exploiter), each arm dry-run on the CPU (`DRY=1`; results in
   `docs/rl_model.md` "Robustness mechanisms"). Also: the title recipes' player mixes now give every count >= 10%
   (`MIX_FLOOR`).

## Open questions

- Pinned share and temperature for the SL seats (A1): 25% and T 1.0 are guesses.
- The regret score for A2: advantage magnitude, critic error, or loss change.
- Whether 2-player and 5-6-player games need their own M1 positions (Caleb's games are 4p and one 6p).
- Whether the live bots should keep the placement head or a blend (Step 0 decides).
