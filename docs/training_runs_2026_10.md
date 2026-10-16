# Training runs, October 2026 (the AlphaGo-style line)

A record of what was trained, from what, against whom, and how it scored. Run
directories are under `model_checkpoints_pg/<run>/` (`config.json` holds the
exact `PGConfig`, `history.jsonl` the per-update metrics, `state.json` the
league/pool state); logs under `logs/` are not in git. The launch scripts from
round 3 on are kept in `scripts/launch/2026-10/` (earlier ones were lost with a
scratch directory).

## How runs were judged

- **Policy-only evals** (`scripts/eval_policy_only.py`): every seat plays its
  policy at temperature 0.5 (`<ckpt>@0.5`), 1200 games, human SR1 start
  positions (+20% random auction endings). **2v2**: each side holds two seats
  over all six arrangements, 0.5 = equal; SE ~0.014. **1v3**: one seat against
  three, equal = 0.25 -- the closest to one bot at a table of humans.
- **Search evals** (`scripts/eval_head_to_head.py`): **every result before
  2026-10-16 is invalid** -- the Rust tree searched every operating-round node
  for seat 0 (fixed 3217c28; see CLAUDE.md, MCTS). The seat-0 edge it produced
  (seat 0 won 0.27-0.50 of search games, 0.25 in policy-only games) was first
  noticed 2026-10-07 and not chased.
- Styles (openings, first 3-train / D, payout rate, B&O in OR1, stock rounds per
  game) from 60-120 saved self-play games per checkpoint (`logs/eval/member_profiles/`).

## The runs

| Run (directory) | From / anchor | Opponents | Reward | Updates | Outcome |
|---|---|---|---|---|---|
| SL ckpt 10 (`model_checkpoints/AlphaZeroTransformer/20261004_134558_804974562/10.pth`) | pretraining on lmdb_v6k (4,136 human games, forced positions kept) | -- | -- | -- | the base of everything below; human top-1 0.712 |
| `pg1_20261006` | ckpt 10 | ckpt 10 + own snapshots, temperature 1 | win | 98 | stopped: mostly learned to be sharper (ckpt 10 @0.5 beats itself @1 0.87) |
| `pg2_20261006` | ckpt 10, learner temperature 0.5 | same, T 0.5 | win | 95 | peaked at 50 (0.575 vs ckpt 10), then degraded: dropout was on in training only; opponents' 5% random prices inflated the score |
| `pg3_20261006` | ckpt 10, all fixes (eval mode, minibatch 2048, lr 2e-5, KL target 0.08) | ckpt 10 + snapshots | win | 500 | best 350: 0.805 vs ckpt 10 (2v2); promoted to `model_checkpoints/AlphaZeroTransformer/20261007_094500_804974562/1.pth` |
| `pg4_20261007` | pg3/350, moving KL anchor every 100 (R-NaD style), 30% mid-game starts from 200 | ckpt 10 25% + pool | win | 750 | 0.796 vs pg3/350; narrowed to one opening (C&O+NYNH 100%), first 3-train in SR8, never ran the B&O early |
| `exploit1_20261007`, `probe_pg4_300/600` | ckpt 10 / copies of pg4 | fixed pg4/150; three frozen copies | win | 200 / 100 | no single-seat hole found (probes 0.27-0.29 at 1v3); exploit1 needed two seats to compete |
| `x2_20261008`, `x3_20261008` | ckpt 10 | fixed pg4/750 (30% / 60% mid-game) | win | 300 | human-like fast-train styles; x2 even with pg4 at 2v2 |
| `lx1_20261008` | ckpt 10 | population v2 (league) | win | 300 | weak: PFSP scored pairs by the learner's best seat and its own snapshots flooded the league (fixed e1a08eb) |
| `gen1_20261009` (updates 0-1000) | ckpt 10, fresh | population v3 (11 members, PFSP) | win | 1000 | ~= pg4/750 (0.487); re-converged on C&O+NYNH |
| `gx_*_20261010`, `gx_rust_20261012` (round 2) | ckpt 10, 2 seats | fixed gen1/1000 | placement / bankrupt / rusher / president:<corp> x8 / rust | 200-300 | placement strongest (0.567 vs gen1); the rust exploiter really rushes (3-train SR3, D SR7) |
| gen2 = `gen1_20261009` 1000-1500 | resume | league_v4 (24 members), hunter + rust pinned 10% | win | 500 | 0.628 vs gen1/1000; bankrupt vs the hunter 8.7% (gen1 12.5%) |
| `gx2_*_20261013` (round 3) | ckpt 10, 2 seats | fixed gen2/1500 | bankrupt / rust / placement | 300 | gen2 hard to exploit: 0.383 / 0.393 / 0.471 |
| gen3 = `gen1_20261009` 1500-2000 | resume | league_v5, 15% of games start at the auction | win | 500 | 0.582 vs gen2; ~84k games/h after the speed-ups |
| `rab_win/place/blend_20261014` (reward A/B) | gen3/2000 | league_v6 | win / placement 1,.5,.2,0 / 1,.3,.1,0 | 300 each | placement best: vs gen3 0.589 (1v3 0.295); round robin place-win 0.543 -> adopted |
| **gen4** = `rab_place_20261014` 300-1800 | resume (games_per_task 96 from 1040) | league_v6, then v7 (+gen4/800, the placement exploiter pinned 10%) | placement | 1500 | **0.737 vs gen3, 1v3 0.478; beats all 31 population members** |

Populations: `model_checkpoints_pg/populations/league_v1_base.json` ... `league_v7.json`
(`min_share` pins a member's minimum sampling share).

## Final ladder (battery, 2026-10-16, policy-only @0.5, 1200 games each)

| Match | 2v2 | 1v3 (equal 0.25) |
|---|---|---|
| pg3/350 vs ckpt 10 | 0.805 | -- |
| pg4/750 vs pg3/350 | 0.796 | 0.590 |
| gen1/1000 vs pg4/750 | 0.487 | 0.245 |
| gen2/1500 vs gen1/1000 | 0.628 | 0.318 |
| gen3/2000 vs gen2/1500 | 0.582 | 0.294 |
| gen4/1800 vs gen3/2000 | 0.737 | 0.478 |
| gen4/1800 vs ckpt 10 | 0.905 | -- |
| gen4/1800 vs pg4/750 | 0.689 | -- |

gen4's weakest 2v2 matchups: the round-3 placement exploiter 0.613, the round-2
one 0.641, pg4/750 0.666. Games against the round-3 bankruptcy hunter that end
with gen4 bankrupt: 8.3% (gen3 12.0%, gen2 17.5%).

## Value networks (`model_checkpoints_value/AlphaZeroTransformer/20261004_134558_804974562/`)

| Net | Data | Notes |
|---|---|---|
| 17 = value_v1 | 60k policy games of ckpt 10, 4 positions each | first value net; no memorization |
| 26 = value_v2 | 400k policy games of ckpt 10 (40 opening decisions at T 1, then 0.5) | best on human games (lmdb_v6k validation CE 0.932, winner 0.586); the advisor's value net |
| 35 = value_v3 | value_v2 fine-tuned on 400k policy games of gen4/800 | better on agent games (CE 0.868 vs 0.979), worse on human games (1.006, winner 0.556) |

## Speed-ups along the way

Batched inference-server replies, the rotation head without its (B, H, T, 800)
tensor, skipping the tile grid where no tile lay is legal, no log-softmax when
legal moves are given, baselines computed at update time, microbatch 1024 and
`--games-per-task 96` (league opponents' servers saw ~35-state batches at 32):
39.5k -> ~84k policy-gradient games/hour for one run. bf16 trunks were only 1.03-1.06x
faster than TF32 and fused attention was slower; neither is used.
