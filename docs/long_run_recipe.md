# Long-run recipe (draft, 2026-10-23)

A ~4-month 1830 training run on the workstation (Threadripper PRO 9975WX, 64 threads, 251 GB, one RTX PRO
6000 Blackwell 96 GB). It is built on what the last week measured, not on what worked in Go:

| finding | source | consequence for the run |
|---|---|---|
| Search systems sit at a fixed point set by the value; the trained policy made a worse search prior | 2026-10-21 diagnosis (azl factorial: u0 prior + u1200 value 0.554 vs u0, u1200 prior + u1200 value 0.474) | the value is the lever; the prior must stay broad and independent |
| A separate value network keeps improving with saved games (~0.006 test CE per doubling, 4.6k-37k, no flattening) and, trained offline over 8 passes, beats azl's in-run value in play (0.542 +- 0.023, net worth 0.507) | `docs/value_gate.md` | save every game; train the value offline, many passes, as its own network |
| Dense auxiliary targets moved neither outcome CE nor play | `docs/value_gate.md` | leave them out |
| Four-player self-play drifts non-transitively (chained step wins implied 0.71, measured 0.500; side-A effect +0.011 +- 0.013) | diagnosis + controls | opponents from a league; judge only against a fixed panel, 2v2 and 1v3 |
| ~20 short runs in 5 days, judged on +-0.03 evals with proxies that didn't track strength | diagnosis | pre-registered gates and stop rules, weekly decisions, forks instead of restarts |

## The run in one picture

```
            +--------------------- frozen deployment (tag longrun-2026-11) ----------------------+
            |                                                                                      |
 prior  --> |  GENERATOR  (search-learning, lockstep self-play; 70% learner-only tables,          | --> saved games
 (ckpt 10,  |              30% with 1-2 league seats; PUCT 200 full / 48 fast, reuse total)      |     (--save-data, xz,
  frozen in |       ^ reloads the promoted value every cycle                                     |      ~20 KB / game)
  phase 1)  |       |                                                                            |         |
            |  VALUE TRAINER  (offline, many passes over all saved games, warm start)  <---------+---------+
            |       |  promotes a new value only through the promotion gate                      |
            |       v                                                                            |
            |  EVAL LADDER  (fixed panel, 2v2 + 1v3, side-swapped, start-clustered Elo)          |
            +--------------------------------------------------------------------------------------+
               harness: systemd units + watchdog + alerts (scripts/longrun), driver/kernel held
```

Three long-lived processes, each a harness unit: the generator, the value trainer, the eval ladder.
Nothing else runs on the GPU during the run except forks under the fork policy below.

## Components

### 1. Prior (policy)

- **Start: checkpoint 10** (supervised on 4k human games; `model_checkpoints/AlphaZeroTransformer/20261004_134558_804974562/10.pth`),
  searched at prior temperature 1 -- azl's u0 policy, the broad prior the factorial found to be the better
  search prior.
- **Phase 1 (weeks 1-3): frozen.** The run tests one hypothesis at a time: does the system improve as the
  value trains on a growing pool of its own games? Search targets are still saved with every game, so a
  policy can be trained later from the same data.
- **Phase 2 (from week 3, as a fork first): anchored policy learning.** Search learning's policy loss with
  completed-Q targets, no surprise weighting, track lays distrusted, and a **fixed KL anchor to checkpoint
  10** (target 0.03, never moved; az1 kept its strength with one). It becomes the main line only through
  the prior gate (below).

### 2. Value

- **A separate network** (the full `ModelTransformer`: its own entity and map encoders, trunk, win and score
  heads); outcome losses only (win CE + placement-points MSE). **Start:** the value gate's 36,989-game value
  (`logs/value_gate/runs/base_36989/learner/0.pth`, test CE 0.920; beat azl u2100's value 0.542).
- **Trained offline by the value trainer, in cycles** (every ~12 h, or every ~40k new games):
  - data: positions re-encoded from the saved games (`search_data.rebuild_rows`), 16-32 decision positions per
    game; every game of the last 200k, plus an equal-sized uniform sample of all older games (older games
    come from weaker systems -- keep them, at a lower share);
  - **anchor** (since 2026-10-26): half of every batch from the value gate's training games (the 37k eval
    games of ~40 systems the start value learned from, `--anchor-data`), the other half from the saved games;
  - warm start from the current promoted value; AdamW lr 3e-5, batch 256, 1-2 epochs over the cycle's sample;
    no early stopping (the candidate is the trained state);
  - **held-out sets:** a fixed one (the value gate's test collections, never trained on: comparable for the
    whole run), a **gate set** of eval-style games (the eval ladder's games of the start system against the
    panel, `--gate-data`, never trained on) and a rolling one (the newest ~3% of saved games by hash,
    refreshed each cycle; reported, not gated on).
- **Promotion gate** (each cycle): promote only if paired CE on the gate set improves by >= 2 paired SE and
  the fixed set is not worse by > 2 paired SE. Otherwise keep the current value and extend the next cycle's
  training. (Until 2026-10-26 the gain was required on the rolling set: see "Changed during the G0 pilot".) **Weekly play check:** the newest promoted value vs the value promoted a week earlier,
  same prior, 480 games (`<prior>+<value>` specs): see the stop rules.
- The generator reloads the promoted value between updates; kept trees are dropped on a version change
  (version = value promotion number).

### 3. Opponents (league)

- **70% of games: all four seats the current system** (prior + promoted value). **30%: 1-2 seats from the
  league**, sampled by PFSP on the main system's recent score against each member (as policy-gradient's
  league does), each searched with its own networks at the fast budget (48); search targets are recorded
  only for main-system seats, but every game trains the value -- which then learns positions against
  styles other than its own.
- **League members:** the main line's own systems every 7 days (at most 6 kept); fixed styles from the
  existing work -- gen4 + critic, az8 u170, azl u2100, checkpoint 10 + the start value (human-like), and
  two policy-gradient exploiters (the bankruptcy hunter and the rust exploiter, policy-only at T 0.5).
- **Drift guard:** the run logs style statistics (par choices per corporation, first 3-train / D timing,
  withhold rate, bankruptcies, game length) daily; if one corporation's par share or the withhold rate moves
  past the guard thresholds below, the league share goes up to 50% for the next week.

### 4. Data

- `--save-data` on every generator game (all seats, all searches): ~20 KB per game. At ~3k games/hour
  (a dedicated GPU, a composite costs ~1.2-1.3 forwards per evaluation) that is ~9M games and ~180 GB over
  4 months. The disk has ~990 GB free (+ ~215 GB of checkpoint cleanup awaiting approval).
- Retention: every saved game, always (Maintenance has the policy for everything else). Shards are
  append-only xz; a crash loses at most the open journal's torn record.
- Nightly copy of the shard list's checksums; a weekly off-disk backup of the newest week's shards if
  another disk is available.

### 5. Evaluation (eval ladder)

- **Panel** (fixed for the whole run; `scripts/eval_ladder.py`, panel file under `scripts/launch/2026-11/`):
  the run's own start system (u0 = checkpoint 10 + start value), gen4 + critic @200, az8 u170, azl u2100,
  and checkpoint 10 + gen4's critic (a human-like prior with a different value). Sparse self members: the
  main line every 4 weeks.
- **Schedule (~15% of the GPU):** daily, 2v2 at 240 games against two panel members in rotation (each
  pairing both ways); weekly, the full panel at 480 games, 2v2 and 1v3, plus the weekly value check.
- **Ratings:** start-clustered four-seat Luce/Elo on 2v2 wins, 1v3 wins and net-worth share, separately
  (`eval_ladder.py report`); `decide --back 1` at every weekly review.

### 6. Harness

- Frozen deployment (`scripts/longrun/deploy.sh create`) at the tag, outside `.claude/worktrees`; three
  instances of the systemd unit (generator, value trainer, ladder) with memory limits; the watchdog with
  alerts on crash loops, non-finite stops, stalls (no history line for 30 min), disk < 100 GB, GPU
  throttling; driver and kernel held (`host_hold.sh`, needs sudo).
- Changes during the run happen only in a new deployment, started from the main line's snapshot and
  saved data (a fork) -- never by editing the running one.

## Phase 0: before launch (~1 week)

**Merge** (in this order, each followed by `pytest tests/` and `cargo test --no-default-features`):

| branch | what | note |
|---|---|---|
| `claude/search-path-fixes` | resume keeps unpassed settings (+ e96d852), live network version, score-head terminal values, eval forced-price pricing | |
| `claude/keen-wescoff-0bb6a3` | price search honours mean Q | conflict: `RustMCTSPlayer::from_game` initializer -- keep theirs, add `terminal_score_*` |
| `claude/agitated-kare-1d2742` | `--save-data`, encoder determinism fix | merges cleanly with search-path-fixes |
| `claude/sleepy-sammet-8fd0f9` | eval ladder | |
| `claude/recursing-kalam-c6eba1` | long-run harness | |
| `claude/value-gate` | value gate, this recipe | |

**Build** (each with tests):

1. **Value trainer** (`main.py value-train`, from `scripts/value_gate.py train`): reads `--save-data` shards,
   samples per the cycle rule, trains, applies the promotion gate, writes `<value run>/value/<n>.pth` and a
   `promoted.json`; resumable.
2. **Search learning with a followed separate value:** `--value-follow <value run>` (load the latest promoted
   value between updates; no in-loop critic training), and `--freeze-policy` for a separate-value learner
   (generation only: no policy updates, targets still saved).
3. **League seats in search learning:** `--population <json>`, `--league-share`, PFSP over members, members'
   own inference servers, fast-budget search for their seats, rows only from the main system's seats.
4. **Style-statistics logger** (daily, from saved games) and the drift-guard thresholds.
5. **The run's panel file** and the ladder as a harness instance.
6. **Per-member league scores in the generator's history** (P1, P2: win and net-worth excess per member and
   per kind -- fixed vs past snapshot), part of item 3.
7. **Daily status report** (`scripts/longrun/status_report.py`): every P and M metric, baselines, 3- and 7-day
   changes with SE, green / yellow / red, appended to `docs/long_run_log.md`, alerts on yellow and red.
8. **Retention job** (`scripts/longrun/retention.py`): the retention table as a config file, dry-run mode,
   daily; and the watchdog additions (days to full, RSS growth, Xid scan, nightly integrity check).

**Status (2026-10-23): merged and built on `claude/longrun-phase0`, not launched.** The six branches merged in
order (conflicts: `from_game`'s initializer as planned, two doc sections kept side by side; the saved searches
now record the server's live version); `pytest tests/` 1354 passed on the merge and 1425 on the build (one flaky test fixed), `cargo test` 222. Built, each
with tests:

| item | where | as built (where it differs from the text above) |
|---|---|---|
| 1 value trainer | `main.py value-train`, `value_trainer.py` | rolling held-out set: the held-out games among the newest 200k, 8 positions each; **half of them choose the early-stopped state, the other half gate it** (else the gate judges the candidate on the positions that chose it); a cycle with no gain never promotes; divergence also when training ends above its start; escalation doubles the games trigger too (40k -> 80k), else 24-h cycles never happen |
| 2 followed separate value, frozen prior | `search-learning --value-follow <value run> --freeze-policy` | the frozen prior's per-update checkpoints are hard links; the server reloads only on a promotion |
| 3, 6 league seats, per-member scores | `search-learning --population --league-share ...` | one member per league game, PFSP on pairwise net-worth results; history `league` per member (sums and squares for the SE); snapshots in 6 server slots, games straddling a slot change unscored |
| 4 style statistics | `scripts/longrun/style_stats.py` | replays the day's games (the log alone doesn't name presidents or OR numbers) |
| 5 panel, ladder instance | `scripts/launch/2026-11/eval_ladder_longrun.json`, `eval_ladder.py` rotation keys | u0 plays the whole panel first (the day-0 baseline, ~3.8k games); then two members per candidate in rotation, a full panel + value check weekly |
| 7 daily report | `scripts/longrun/status_report.py` (daily timer) | "days" are 24-h windows ending at the report |
| 8 retention, watchdog additions | `scripts/longrun/retention.py` + `retention.json`, `maintenance.py`, `integrity_check.py` | dry run until `dry_run_until` (set to launch + 7 days when approved) |

Instances: `scripts/launch/2026-11/install_lr1.sh <deployment> <workdir>` (generator, value trainer, ladder,
daily jobs; memory caps 120 / 36 / 26 / 12 GB -- azl alone measured ~80 GB and the league adds up to 12
inference servers; the pilot measures the real peaks). Runbook: `docs/longrun_runbook.md`, "The long run
(2026-11)". The generator's settings are azl's (cq targets, c 0.05, 200 / 48, reuse total, exploration in the
first 120 decisions, 30% mid-game starts) without auction starts.

**Decided before the pilot (2026-10-23):**
- **Stall rule:** P1's condition is now "the last 5 days' pooled score is not above the 5 days before by >= 1
  SE" (the draft's "5-day slope's upper 95% bound < +0.001/day" never fired on a flat series at real noise:
  ~0.0013/day SE puts a flat series' bound near +0.0025/day). The other two conditions are unchanged.
- **Promotion gate:** as written; the daily report tracks the promotion rate. If promotions dry up as the pool
  grows (a 40k-game cycle on > 1M games adds a few % of data, an expected gain below 2 SE), revisit then
  (a non-inferiority gate or longer cycles).
- **Exploration window:** counted from the game's first Stock Round (`--explore-from-game-start`): a mid-game
  start's recorded decisions use up the 120-decision window, so mid-game starts explore only what is left of
  it, if anything; SR1 starts are unchanged.
- **Recipe questions:** 16 positions per game for the value (the value gate's); the league's search members all
  at the fast budget (48); human evaluation deferred to weeks 4, 8 and 16; the auction and 2-6 players stay out.
- **GPU:** azl stopped for the pilot; azlc keeps running beside it, so the pilot's throughput and memory peaks
  are measured on a shared GPU and CPU (G0's throughput line is judged with that in mind).
- **Host:** the driver/kernel holds (`host_hold.sh`, sudo) and the pending reboot are the user's; the reboot waits
  for azlc. Until then the report shows them as yellow / red. *2026-10-26:* the user chose not to hold: updates
  stay on, a driver upgrade is handled by a reboot (the watchdog's `nvidia_mismatch` alert); M7's hold check is
  off and the units start at boot (docs/longrun_runbook.md "Host stability").
- Workdir: the panel and league checkpoints, `logs/value_gate/data` and the start value are copied from the
  session worktrees into the workdir (the runbook's setup).
- Not built: the phase-2 prior factorial (P8), the off-disk backup itself (M9 only checks its age).

**Changed during the G0 pilot (2026-10-26):** the value trainer's data and gate (deployment `longrun-2026-11b`,
the value unit and the daily jobs; the generator and ladder kept `longrun-2026-11`).
- *Why:* cycles 1-3 never promoted: each candidate was better on the generator's held-out games (rolling
  -0.022) and worse on the fixed set (+0.011, +0.017, +0.021). Played under the same prior (480 games 2v2),
  cycle 2's candidate lost to the start value 0.440 +- 0.023. A value trained from the start value on the eval
  games plus the generator's games mixed tied it, 0.506 +- 0.023, with the fixed set unchanged (+0.0007).
- *Calibration* (paired CE vs the start value; gen4's critic, the start value's own predecessor, scores 0.335
  against it in play): gen4's critic +0.098 fixed / +0.073 ladder games / +0.023 rolling; cycle 2 +0.017 /
  +0.001 / -0.024; the mixed value +0.001 / -0.003 / -0.020. CE on eval-style games (fixed, ladder) tracks play
  at ~2 points of score per nat; CE on the generator's own games does not (it rates cycle 2 as far above the
  start value as the start value is above gen4's critic). A 480-game play test (SE 0.023) can't resolve a
  per-cycle gain of ~0.003 nats (~0.007 score); CE on ~60k eval positions (SE 0.001) can.
- *So:* `--anchor-data logs/value_gate/data --anchor-share 0.5`, `--gate-data logs/value_gate/ladder_lr1_u0`
  (the ladder's 3,802 games of u0 vs the panel, 60,832 positions), no early stopping; a restart with changed
  gate settings forgets the old gate's failures. Play stays the ladder's job (rotation and the weekly check).
- *Open:* the gate set is static (u0's ladder games); refreshing it with each promoted value's ladder games
  is not built. The anchor is re-used every cycle (~1 pass per cycle): the fixed guard is what would catch
  the value memorizing it.

**Baselines (day 0):** the start system against the full panel (2v2 + 1v3, 480 games), the start value's
CE on both held-out sets, style statistics of the start system's games.

**Gate G0 -- 48-hour pilot at full configuration.** Launch only if all hold:
- no crash, no non-finite loss, no watchdog alert left unexplained;
- saved games rebuild into rows bit for bit (spot check 100 games);
- at least two value cycles completed, at least one promotion, and the generator reloaded it;
- the ladder produced ratings for two candidates;
- throughput >= 2.5k games/hour with the ladder running;
- the pilot's system is not below the start system on the panel by > 2 SE.

## Progress metrics, cadence and stop rules (pre-registered)

The run is judged at three cadences. The fast metrics raise alarms and can call a review on any day; a stop
or a rollback needs two independent signals to agree (training-game scores and the eval ladder), so one
noisy series never ends the run -- the lesson of the 100-300-update restarts.

### What is tracked

| # | metric | source | cadence | noise (SE) | what it tells |
|---|---|---|---|---|---|
| P1 | **score vs each fixed league member**: the main system's seats' win share minus their fair share, and net-worth share, in league games | training games (logged per update, aggregated daily) | continuous | ~2-4k games per member per day (30% of ~3k games/h, PFSP over ~11 members): ~0.01 per member per day, ~0.004 pooled over the 5 fixed members | the fastest strength signal: the same fixed opponents every day, thousands of games (policy-gradient's `score_vs_sl` played this role). Training conditions (exploration in the first 120 decisions, members at the fast budget), so read it as a trend against its own past, not against eval numbers |
| P2 | **score vs past main-line snapshots** in league games | training games | continuous | as P1 | with P1, the drift detector: rising against past selves while flat against fixed members is azl's non-transitive drift |
| P3 | **fixed-set value CE** (the value gate's test collections, never trained on) for every promoted value | value trainer | every cycle (~12 h) | paired ~0.001 | the run's premise, plotted against log2(saved games): the gate measured -0.006 per doubling; the slope must stay negative |
| P4 | **gate-set CE gain** of each candidate value (eval ladder games; before 2026-10-26 the rolling set: the newest ~3% of games), the rolling-set gain, and the gap between rolling and fixed CE | value trainer | every cycle | paired ~0.001 | promotion; a widening gap (rolling improves, fixed doesn't) = the value fitting the newest self-play only |
| P5 | **daily ladder**: the current system vs two panel members in rotation, 2v2, 240 games, both directions | eval ladder | daily | ~0.032 per pairing, ~0.023 per day (two pairings), ~0.013 over 3 days (~20 Elo) | the eval-conditions check of P1, start-clustered |
| P6 | **style statistics**: par share per corporation, first 3-train / D timing, withhold rate, bankruptcies, game length | saved games | daily | -- | drift guard |
| P7 | **generation health**: games/h, decisions/s, evaluations/s, max-length share, skipped starts, reuse share | generator history | continuous | -- | throughput and stuck-game detection |
| P8 | **full panel** (2v2 + 1v3, 480 games), the **weekly value play check** (new promoted value vs a week earlier, same prior), phase 2's **prior factorial** | eval ladder | weekly | ~0.023 per match | the formal judgment; 1v3 is the closest to one bot among humans |

Expected movement, for calibrating the alarms: the data doubles from the start's 37k games in ~half a day,
then ~1M games by week 2, ~2M by week 4, ~4M by week 8 -- the value's gains per doubling (P3) come fast early
and then once per ~2x elapsed time. Day 0 records every metric's baseline.

### Daily report

`scripts/longrun/status_report.py` (phase-0 build) runs every morning: every metric above with its baseline,
its 3-day and 7-day change and SE, and a green / yellow / red status against the rules below; it appends to
`docs/long_run_log.md` and sends an alert for anything yellow or red. Reading it takes five minutes; a
yellow means "look today", a red means "review today".

### Rules

**Hard stops** (any time; the watchdog alerts): crash loop, non-finite loss, saved-data write failure or an
integrity-check mismatch (Maintenance), disk red, the value trainer diverging (rolling CE worse than its
start by > 0.01). Fix, then resume from the last snapshot -- the saved data makes a resume lossless.

**Daily alarms** (yellow, each triggers a same-day look, not a stop):
- P1 pooled score below its 5-day maximum by >= 2 SE;
- P3: two promoted values in a row with no fixed-set CE gain;
- P4: rolling-minus-fixed CE gap widening for 3 consecutive cycles;
- P2 rising by >= 2 SE over 3 days while P1 is flat (drift): raise the league share to 50% at once;
- P5 3-day rolling below the previous 3 days by >= 2 SE;
- P6 drift guard: any corporation's par share > 0.5 of the main system's pars, or the withhold rate doubled;
- P7 throughput below 80% of its 7-day median, or max-length games > 3%.

**Stall** (review within a day; decide there whether to stop the main line): P1 pooled over the last 5 days
is not above the 5 days before by >= 1 SE **and** P3's slope over the last 3 promotions
is >= 0 **and** P5 is flat over 6 days. All three, not one: they are independent measurements (training
games, held-out prediction, eval matches).

**Decline** (roll back to the best value by ladder rating, then review): P1 pooled below its best 5-day
mean by >= 3 SE **and** P5's 3-day rolling below its best by >= 2 SE.

**Value rules**
- Promotion (every cycle): paired gate-set CE (eval ladder games) better by >= 2 paired SE and fixed-set CE
  not worse by > 2 paired SE; else keep the current value. (Before 2026-10-26: the rolling set.)
- Promotion fails 3 cycles in a row: 24-h cycles, twice the sample; 3 more failures: review the data mix.
- Weekly value play check < 0.47 twice in a row: stop promoting, revert to the best value by ladder rating,
  review (a value that predicts better but plays worse).

**Prior rule (phase 2)**: a trained policy replaces checkpoint 10 as the main line's prior only if, under the
current promoted value, it beats checkpoint 10's prior in a 480-game factorial at > 0.5 + 1 SE and its own
P1 and ladder rating are not lower. Otherwise the fork is dropped and retried later with the KL target halved.

**Weekly review** (Mondays): the week's daily reports, P8, the ladder's `decide --back 1`; numbers into
`docs/long_run_log.md` before deciding. **1v3 guard:** 1v3 rating down by >= 2 SE while 2v2 rises means the
gain is table composition -- league share 50%, re-check next week.

**Milestones** (reviewed at the dates; between them the daily rules above decide):

| when | expectation |
|---|---|
| day 3 | P1 pooled above its day-0 baseline by >= 2 SE; at least two promotions with fixed-set CE gains |
| week 2 | the main system above the start system on the panel by >= 2 SE (2v2); fixed-set CE below the start's 0.920 |
| week 4 | above week 2 by >= 1 SE; 1v3 against gen4 + critic above the day-0 baseline |
| week 8 | above week 4 by >= 1 SE; P3's slope per doubling still at >= half the gate's (~0.003) |
| week 16 | end, or extend if the week-8 to week-12 slope is still positive at 1 SE |

## Maintenance

Measured on 2026-10-23: 973 GB free on `/`; a generator snapshot is 35 MB (azl kept every 10th update:
469 files, 16 GB in under 3 days, ~6 GB/day -- ~700 GB over 4 months if nothing thinned it); saved games
~20 KB each (~1.4 GB/day at 3k games/h); eval games ~108 KB each (~0.4 GB/week at the ladder's schedule);
`history.jsonl` ~3.4 KB per update (~1 GB over the run); session worktrees ~5 GB each (165 GB today).

### Tracked (the watchdog and the daily report)

| # | metric | yellow | red | action |
|---|---|---|---|---|
| M1 | **disk: days to full** = free space / 7-day growth rate | < 30 days | < 14 days, or < 150 GB free | yellow: check the retention job ran and plan the move of old shards; red: move the oldest saved shards off the disk (manual), clear finished worktrees |
| M2 | **growth by category** (saved games, checkpoints, evals, logs) | any category > 2x its expected rate | -- | usually the retention job failed or a log is flooding |
| M3 | **session worktrees** (`.claude/worktrees`) | > 200 GB | > 300 GB | clean up worktrees of finished sessions (manual) |
| M4 | **RAM**: lowest MemAvailable of the day; per-process RSS growth | RSS growing > 2 GB/day for 2 days (a leak), or swap in use | MemAvailable < 30 GB | yellow: a controlled restart (resume) at the next snapshot; red: stop the ladder first, then investigate |
| M5 | **GPU**: throttle reasons, temperature, Xid errors in the kernel log | throttling > 10% of the hour | any Xid error | red: stop the run and inspect the hardware/driver |
| M6 | **saved-data integrity**: every night, 50 random games of the last day rebuilt into rows (`search_data.rebuild_rows`) and compared bit for bit | -- | any mismatch | stop promoting values from the affected shards; investigate before the next cycle |
| M7 | **host**: driver/kernel hold still in place; `/var/run/reboot-required` | a reboot pending | the hold removed | plan a reboot at a snapshot (the run resumes); restore the hold |
| M8 | **value-trainer cycle time** | > 2x its target | -- | the pool outgrew the sampling: shrink the old-games share |
| M9 | **backup age** (if a second disk exists) | newest week not copied for > 8 days | -- | run the copy |

### Retention policy (automated by a daily job; the policy is approved once, before launch)

| item | kept | deleted |
|---|---|---|
| saved games | **all, always** (the value's data) | never; moved off-disk only by hand when M1 is red |
| promoted values | all (~35 MB, at most ~250 over the run) | -- |
| unpromoted value candidates | 7 days | after 7 days |
| generator / policy snapshots (phase 2 forks; phase 1's prior is frozen, so none are written) | every 10th update for 48 h, every 100th for 2 weeks, every 1,000th forever, plus every ladder candidate and league snapshot | the rest, on that schedule |
| optimizer state | the latest + one per weekly review | older |
| eval collections and the ladder ledger | all | -- |
| logs | rotated at 1 GB, 4 rotations | older rotations |

The job runs in dry-run mode for the first week (it lists what it would delete in the daily report), then
for real. Anything outside the table is deleted only by hand.

## Fork policy

- One fork at a time on the GPU, from a main-line snapshot and its saved data, in its own deployment, with a
  written hypothesis, horizon and success criterion; it is judged on the same panel.
- Planned forks: (1) phase 2's anchored policy (week 3); (2) a larger value network (96 GB allows it; same
  data, judged by the weekly value check); (3) placement-utility search (`#blend`, now that terminal values
  follow the head) if the value plateaus.

## Budget

| resource | generator | value trainer | ladder | total |
|---|---|---|---|---|
| GPU | ~75% | ~10% (cycles of ~1 h every 12 h) | ~15% | dedicated |
| CPU threads | ~40 workers | ~8 (re-encoding) | ~12 workers | 64 |
| RAM | ~60 GB | ~30 GB (a cycle's sample) | ~20 GB | ~110 GB of 251 |
| disk | ~1.4 GB/day saved games | promoted values ~35 MB each (~9 GB over the run) | eval collections ~0.4 GB/week | ~200 GB over the run with the retention job (~900 GB without it) |

## Open questions to settle in phase 0

1. Is the GPU fully dedicated for the 4 months (the other project, the advisor's inference)?
2. 16 or 32 positions per game for the value (the value gate used 16)?
3. The league members' search budget: fast (48) for all, or full (200) for the fixed panel-style members?
4. Human evaluation: a few advisor-assisted games against people at weeks 4, 8 and 16, as a check the panel
   can't give.
5. The auction (15% of starts) and the 2-6 player roadmap stay out of this run; extend later from its data.
