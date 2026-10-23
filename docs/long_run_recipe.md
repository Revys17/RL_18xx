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
  - warm start from the current promoted value; AdamW lr 3e-5, batch 256, 1-2 epochs over the cycle's sample,
    early stopping on the rolling held-out set;
  - **held-out sets:** a fixed one (the value gate's test collections, never trained on: comparable for the
    whole run) and a rolling one (the newest ~3% of games by hash, refreshed each cycle).
- **Promotion gate** (each cycle): promote only if paired test CE on the rolling set improves by >= 2 paired
  SE and the fixed set is not worse by > 2 paired SE. Otherwise keep the current value and extend the next
  cycle's training. **Weekly play check:** the newest promoted value vs the value promoted a week earlier,
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
- Retention: everything. Shards are append-only xz; a crash loses at most the open journal's torn record.
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

**Baselines (day 0):** the start system against the full panel (2v2 + 1v3, 480 games), the start value's
CE on both held-out sets, style statistics of the start system's games.

**Gate G0 -- 48-hour pilot at full configuration.** Launch only if all hold:
- no crash, no non-finite loss, no watchdog alert left unexplained;
- saved games rebuild into rows bit for bit (spot check 100 games);
- at least two value cycles completed, at least one promotion, and the generator reloaded it;
- the ladder produced ratings for two candidates;
- throughput >= 2.5k games/hour with the ladder running;
- the pilot's system is not below the start system on the panel by > 2 SE.

## Weekly review: gates and stop rules (pre-registered)

Every Monday: `eval_ladder.py report` + `decide --back 1`, the weekly value check, style statistics, value
CE on both held-out sets, throughput, disk. Write the numbers into `docs/long_run_log.md` before deciding.

**Hard stops** (any time; the watchdog alerts): crash loop, non-finite loss, saved-data write failure, disk
< 100 GB, the value trainer diverging (rolling CE worse than its start by > 0.01). Fix, then resume from
the last snapshot -- the saved data makes a resume lossless.

**Value rules**
- Promotion gate fails 3 cycles in a row: extend cycles to 24 h and double the cycle's sample; fails 3 more:
  review the data mix (share of old games, positions per game) before continuing.
- Weekly value check (new promoted vs a week earlier, same prior): < 0.47 twice in a row -> stop promoting,
  revert to the best value by ladder rating, review. (A value that predicts better but plays worse is the
  non-stationarity / self-play-overfit failure; the league share goes up in the next attempt.)

**System rules (the ladder, panel matches only)**
- **Progress:** continue while the 2v2 panel rating is above its value two weeks earlier at >= 1 SE.
- **Stall:** no significant gain (1 SE) over 3 consecutive weeks -> stop the main line, keep the data, and
  decide the next recipe change from the evidence (a fork against the stalled line, not a restart).
- **Decline:** below the run's own best rating by >= 2 SE at two consecutive reviews -> roll back to the best
  checkpoint (value + prior) and review.
- **1v3 guard:** if the 1v3 rating falls by >= 2 SE while 2v2 rises, the gain is table-composition
  (two copies helping each other): raise the league share to 50% and re-check next week.
- **Drift guard:** any single corporation's par share > 0.5 of the main system's pars, or the withhold rate
  doubled from day 0 -> league share 50% for a week.

**Prior rule (phase 2)**: a trained policy replaces checkpoint 10 as the main line's prior only if, under the
current promoted value, it beats checkpoint 10's prior in a 480-game factorial (`<trained>+<value>` vs
`<ckpt10>+<value>`) at > 0.5 + 1 SE, and its own ladder rating is not lower. Otherwise the fork is
dropped and phase 2 is retried later with a tighter anchor (KL target halved).

**Milestones** (what "worth continuing" means; reviewed at the dates, not before):

| when | expectation (else: the stall rule decides) |
|---|---|
| week 2 | the main system above the start system on the panel by >= 2 SE (2v2), value CE on the fixed set below the start's 0.920 |
| week 4 | above week 2 by >= 1 SE; 1v3 against gen4 + critic above the day-0 baseline |
| week 8 | above week 4 by >= 1 SE; the value's fixed-set CE still falling per doubling of data at >= half the gate's rate (~0.003) |
| week 16 | end, or extend if the week-8 to week-12 slope is still positive at 1 SE |

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
| disk | ~1.5 GB/day saved games | checkpoints ~35 MB each, keep promoted | eval collections ~0.5 GB/week | ~200 GB over the run |

## Open questions to settle in phase 0

1. Is the GPU fully dedicated for the 4 months (the other project, the advisor's inference)?
2. 16 or 32 positions per game for the value (the value gate used 16)?
3. The league members' search budget: fast (48) for all, or full (200) for the fixed panel-style members?
4. Human evaluation: a few advisor-assisted games against people at weeks 4, 8 and 16, as a check the panel
   can't give.
5. The auction (15% of starts) and the 2-6 player roadmap stay out of this run; extend later from its data.
