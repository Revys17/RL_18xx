# Training runs, November 2026: pg2p, gen5, gen6, the generic 1830 check, the new titles' SL

Continues `docs/training_runs_2026_10.md` (pg1 -> gen4). All three runs are policy-only policy gradient
(`main.py policy-gradient`) with a league, the placement reward and no search; they ran from the frozen
deployments under `/home/revys/longrun/deploy/` in the lr1 work dir (`/home/revys/longrun/work/lr1`).
Launch scripts, eval scripts, design notes and raw results are archived in
`scripts/launch/2026-11/{pg2p_20261109,gen5_20261107,gen6_20261110}/` (each has a `DESIGN.md`); the league
file is `scripts/launch/2026-11/populations/league_v8_gen5.json` (as launched; `_final` = after gen6's changes).

Evals are policy-only at T 0.5 from the private auction under the real rules, seats rotated
(`scripts/eval_policy_only.py`). "1v3" = one copy of the checkpoint alone against three copies of the
opponent (fair 0.25); "panel" = the out-of-family panel below.

## Summary

| Run | Start | Players | Updates | Result |
|---|---|---|---|---|
| pg2p (`pg2p_20261109`) | SL ckpt 10 | 2 | 2000 (cap) | vs gen4 in 2p .995; 2p-only training carried ~half of gen4's 4p gain over ckpt 10 into 4p |
| gen5 (`gen5_20261107`) | gen4 | 4 | ~2870, final u2800 | vs gen4 2v2 .80-.88, 1v3 ~.56; outside panel .593 |
| **gen6** (`gen6_20261110`) | gen5 u2800 | 2-6 mixed | 2431 (plateau stop) | final **u2400** (the user's pick; bots since 2026-11-13); panel .623 |

## pg2p: two players from the supervised policy (2026-11-09/10)

- PG from SL ckpt 10, 2-player games only, win reward, learner temperature 0.5. Ran to its 2000-update cap
  and was still improving (chain Elo per 100 updates +70 to +270 to the end).
- Fixed-opponent bars saturate early (vs ckpt 10 / gen4 ~.95 by u400). An apparent ceiling vs old selves
  (~.9) was T 0.5 sampling noise in close games: near-greedy evals show the newer checkpoint winning ~all.
- Search on top (stage 2/3, 2p): the policy + its PG critic at 200 readouts beat the bare policy ~.82
  (control: the policy at T 0.05 is ~.49 vs T 0.5, so the gain is search, not greediness). A value refit
  offline to a better win CE searched worse than the critic (.429 head to head): CE does not measure a
  search value.
- Transfer: pg2p u2000 in 4p vs ckpt 10 2v2 .826 / 1v3 .417; vs gen4 .241 / .060 (ckpt 10 vs gen4 .077).
  gen4 in 2p is about pg2p's first 40 updates. Transfer 2p -> 4p is large, 4p -> 2p small: one mixed-count
  model is worth it.

## gen5: four players from gen4 (2026-11-10/11)

- PG from gen4, 4 players, placement reward, 50% of games from the real auction (no `auction_unlock`),
  league `league_v7_gen4` + own snapshots; ran from deployment `gen5-ae3438e`, then `az0-2p-7e21baa`
  (audit fixes, Ruby scoring for bankrupt players).
- vs gen4: 2v2 .530 (u100, = baseline) -> .629 (u300) -> .746 (u600) -> .801 (u1000) -> .877 (u2100)
  -> .786 (u2700); 1v3 .245 -> .564 (u1000).
- The u-200 plateau rule kept finding gains (within-family progress), while the out-of-family panel was
  flat: u1600 .590, u2200 .595, u2700 .606. **Lesson: judge a plateau for a human-facing bot by an
  out-of-family panel.** Stopped by hand at ~u2870; final u2800 (panel .593).
- Learner auction prices drifted near human prices except CS (high, falling) and CA (bought rarely).

## gen6: one model for 2-6 players (2026-11-11/12)

- PG from gen5 u2800 with `--player-mix 2:15,3:20,4:35,5:20,6:10` (worker tasks pick a count by a
  deficit schedule; per-count decision caps; start files per count), gen5's recipe otherwise; league
  `league_v8_gen5` (league_v7_gen4 + gen5 u2800 + pg2p u2000), 3 opponent slots, PFSP power 2.
- Measured every 100 updates against gen5 u2800 at each count, and every 200 on the outside panel.

Changes during the run (each a restart at a snapshot, resumable throughout):

| Update | Change | Why |
|---|---|---|
| 490 | PFSP per player count (`CountLeague`, 76e51c4) | Pooled over counts, pg2p's rate (.59) was dominated by 3-6p games it loses, so it sat in ~2.5% of 2p games while it won 98% of 2p evals |
| 1760 | Exploiter pins at 4p only (`min_share_players`, 96ba807) | gen6 vs pg2p flat at ~.24 for 400 updates; five pinned 4p exploiters held ~53% of draws at every count while gen6 beat them 70-89% |
| 2080 | No pins; the 8 round-2 president:<corp> exploiters retired | 4p panel held (.632 at u2000); exploiters become plain PFSP members (~13% of draws) |
| 2431 | Stopped | Two panels (u2200 .621, u2400 .618) did not beat .632 |

Results (selected):

| Match (fair) | gen5 u2800 | u600 | u1200 | u2000 | u2400 |
|---|---|---|---|---|---|
| 2p vs pg2p (.50) | .028 | .025 | .158 | .305 | .394 |
| 2p vs gen5 (.50) | .510 | .907 | .915 | .953 | .887 |
| 3p 1v2 vs gen5 (.33) | .308 | .674 | .697 | .706 | .627 |
| 4p 2v2 vs gen5 (.50) | .502 | .485 | .563 | .558 | .619 |
| 4p 1v3 vs gen5 (.25) | .253 | .268 | .273 | .318 | .333 |
| 5p 2v3 vs gen5 (.40) | .410 | .502 | .534 | .550 | .551 |
| 6p 3v3 vs gen5 (.50) | .510 | .568 | .562 | .710 | .752 |
| 4p panel mean (.25) | .593 | .593 | .618 | .632 | .618 |

- **Out-of-family panel** (`outfamily.sh`, 800 games each, alone vs copies): 4p 1v3 vs gen4, SL ckpt 10,
  gen2 (`gen1_20261009/learner/1500`), gx2_rust, gx2_bankrupt; 2p/3p/5p/6p alone vs gen4. The 4p mean
  sat at ~.61-.63 from u200 on (an apparent dip at u500-u600 was noise: extra panels at u300-u550).
- **Final pick** (`select_final.sh`: u2000 / u2200 / u2400 re-measured on fresh games, pooled with the
  regular panels): 4p panel .636 / .615 / .623; vs gen4 alone 2p .951 / .933 / .890, 3p .779 / .723 / .719,
  5p .531 / .511 / .555, 6p .532 / .548 / .574; 2p vs pg2p .308 / .383 / .411. u2000 is stronger at 2-4p
  and vs the bankruptcy hunter (.551 vs .513 at u2400, which fell after the pins came off); u2400 at 5-6p
  and vs pg2p. The user picked **u2400** for the bots.
- Within-family progress kept going to the end (vs gen5 and pg2p), while the outside panel plateaued: the
  same split as gen5.

## The generic 1830 check (step 1 of `docs/gpu_runbook_2026_11.md`, 2026-11-18)

The title-generic model (`docs/rl_model.md`, default preset, 8.6M parameters) trained by supervision on SL ckpt 10's
corpus (`human_games/generic_1830_clean_all_v1`: 2.38M training positions; validation = the same hash split), then
compared with ckpt 10 on the same 117,053 validation decisions and played against it policy-only at T 0.5 from the
private auction (real rules, 800 games a match). `scripts/launch/2026-11/bridge/gpu_check_1830.sh pretrain compare
eval`; checkpoint `model_checkpoints_generic/bridge_1830_check_20261113/final.pth` (main checkout), evals in
`logs/eval/bridge_1830_check_20261113/`, log `/home/revys/longrun/experiments/bridge_1830_check_20261118/run.log`.
10 epochs took 63 min on the GPU (~25 steps/s of 256 positions; the 6-h estimate was far off), the compare 2 min,
the four matches 28 min.

| | generic SL | ckpt 10 |
|---|---|---|
| top-1 / top-5 (all decisions) | **.777** / .957 | .718 / .938 |
| top-1 by count 2 / 3 / 4 / 5 / 6p | .756 / .765 / .781 / .771 / - | .677 / .710 / .723 / .718 / - |
| policy NLL | .695 | .899 |
| win CE (equal odds 1.373) / winner accuracy | 1.193 / .577 | **.929** / .578 |

| Match (generic first) | Score | Fair |
|---|---|---|
| 4p 2v2 | **.755** +- .015 (worth share .564) | .50 |
| 4p 1v3, generic alone | **.536** +- .018 | .25 |
| 4p 1v3, ckpt 10 alone (ckpt 10's score) | .091 +- .010 | .25 |
| 2p 1v1 | **.698** +- .016 | .50 |

- **Far past the gate** ("within noise of ckpt 10"): the generic SL policy is much stronger than ckpt 10's.
- **Per-family top-1 misleads where a family competes with Pass.** ckpt 10 is ahead on PlaceToken (.766 vs .255)
  and LayTile (.410 vs .337) when the human placed / laid, because its 1830 head splits "token / tile vs pass" by a
  fixed per-block ratio: it almost never passes there (1 of 616 human passes at token-legal positions). Humans pass
  at 65-67% of the positions where a token is legal. Over **all** positions where the family is legal (validation,
  first 25-40 games): token-legal top-1 generic **.691** vs ckpt 10 .281 (958 positions); lay-legal **.457** vs .416
  (1,693; given that the human laid: .353 vs .438, given a pass: .714 vs .230). The generic head's state-dependent
  group score (`rl_model.md` "Departures") is what wins here.
- **The value heads are weak**: the joint run's win CE was best at epoch 1 (.951) and memorized after (1.49 at epoch
  7); the refit on the frozen epoch-8 trunk reached only 1.193 (best at its first evaluation, train CE .63 vs
  validation 1.19: the trunk already carries the training games' outcomes). At 3 and 5 players it is no better than
  equal odds. Every generic PG run starts its critic from a checkpoint's value heads, so the critic gets its own start
  (next item).
- **The critic's start** (`model_checkpoints_generic/bridge_1830_critic1ep_20261118/final.pth`): the same recipe for 1
  epoch (`--epochs 1 --refit-eval-steps 50 --refit-patience 8`, 7 min + a 5-min refit): win CE **.952** (2-6p .409 /
  .839 / .915 / 1.174 / 1.438; ckpt 10 .395 / .735 / .898 / 1.195 / 1.385), train CE = validation (no memorization);
  its policy (top-1 ~.75) is unused. `robust_arm.sh` and `pg_generic_1830.sh` start the critic from it (`VALUE`,
  `--value`); the learner still starts from the 10-epoch SL. For 1867 / the 1822 family the same split is likely
  needed: `title_recipe.sh`'s `critic` step does it since 2026-11-18 (next section).

## The new titles' SL and critic starts (steps 5-6 of the GPU runbook, 2026-11-18/19)

`title_<t>.sh pretrain critic` for 1867, 1822 (the 1822+ rules mix), 1822PNW, 1822MX and 1822CA: the 1830 check's
recipe (10 epochs, the value refit on the frozen best) and the 1-epoch critic start. Checkpoints in
`model_checkpoints_generic/bridge_<t>_sl_20261114/` and `bridge_<t>_critic1ep_20261114/` (main checkout); each SL run's
`family_report.txt` has the per-family table.

**The first SL runs picked the wrong checkpoint.** The joint stage's best checkpoint (and its early stop) were chosen on
the whole validation loss, value terms included. The value terms memorize the training games' outcomes within about an
epoch and then climb, so on these small corpora they pulled the choice to epochs 2-6 while the policy kept improving to
the end. Fixed in 05163b1 (`pretrain-generic --select-metric`, default `auto` = the loss without its value terms when the
value refit is on; `docs/rl_model.md` stage 1). All five were rerun 2026-11-19 (`/home/revys/longrun/experiments/
sl_rerun_20261119/`); the first runs are kept as `bridge_<t>_sl_20261114_totalsel/` (their `summary.json` paths still
name the old directory).

| Title | Train / validation games | top-1, first run (step) | **top-1, rerun** (step of 10 epochs) | policy NLL first -> rerun | critic start: win CE (equal odds) |
|---|---|---|---|---|---|
| 1867 | 794 / 46 | .742 (11,890) | **.745** (19,817, the last) | .829 -> .819 | .847 (1.362) |
| 1822 (1822+ mix) | 496 / 29 | .698 (4,173) | **.715** (20,867, the last) | .964 -> .915 | .947 (1.382) |
| 1822PNW | 395 / 22 | .687 | **.722** (14,518, the 9th epoch) | .946 -> .851 | 1.022 (1.265) |
| 1822MX | 218 / 11 | .697 | **.710** (7,520, the 9th epoch) | .913 -> .879 | .843 (1.377) |
| 1822CA | 114 / 10 | .619 | **.680** (5,103, the 9th epoch) | 1.174 -> 1.030 | 1.260 (1.367) |

- Rerun times (all CPUs busy with step 2's evals and the auction check): 1867 62 min, 1822 83, PNW 41, MX 19, CA 32;
  each critic start ~4-7 min.
- 1822 and 1867 were best at the last evaluation: their policy was still improving after 10 epochs; more epochs untried.
- Top-1 by player count is flat (1822 .713 / .715 / .712 / .745 at 3-6p; 1867 .732 / .753 / .746 / .729).
- **The SL runs' own value heads are useless** (refit win CE = equal odds for the four 1822 titles, 1867 1.169 vs 1.362):
  after nine epochs the trunk carries the training games' outcomes. Irrelevant for PG, whose critic starts from the
  critic start (`VALUE`); the critic starts are informative except 1822CA's (1.260 vs 1.367 on 10 validation games; a
  critic trained on the 1822 family's mix may serve it better, untried).
- Families under chance (>= 20 positions): `SellShares` in every title, `BuyTrain.corp` in 1867 / 1822 / PNW / CA,
  `PlaceToken` in the 1822 family (.05-.09 vs ~.11), `AcquireCompany` / `RedeemShares` in MX and CA. These are the
  families that compete with a pass; as in the 1830 check, top-1 over the positions where the human made the move
  undercounts a model that passes there most of the time. Rare families: 1867 `Par` .016 (63 positions, chance .028),
  the 1822 titles' `Par` .12-.13 over chance .035-.05.

## Open threads

- gen6 can resume from its u2430 snapshot (`state.json`), e.g. with exploiters trained from gen6 against a
  frozen gen6 (3p first), if the user wants more 1830 strength.
- pg2p remains stronger at 2 players (gen6 u2400 .394, worth share .500).
- Watch the bankruptcy hunter: unpinned, gen6's 4p 1v3 against it fell from .55 to .49-.53.
