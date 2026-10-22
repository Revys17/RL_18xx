# Value gate: dense targets and the data learning curve (2026-10-22)

The 2026-10-21 review concluded that search learning sits at a fixed point set by the value
network. Before a months-long run, two questions:

1. **Does the value keep improving with more games?** If value quality is still rising at the
   largest dataset we can assemble, more data is the lever and a long run pays. If it is flat, a
   long run buys little until the target or the architecture changes.
2. **Do dense auxiliary targets help?** KataGo trains its trunk on hundreds of low-noise labels
   per position (final ownership of every point, the score) besides the one win/loss bit. 1830's
   analogues are below.

## Dense targets (`rl18xx/agent/alphazero/dense_targets.py`)

All are read from a finished game (its replay and its action log). Seat-indexed targets are in
the position's canonical frame (the mover first), so they line up with the player tokens.

| group | shape | target | loss |
|---|---|---|---|
| player | (4, 4) | final net-worth share, cash share, bankrupt, presidencies / 8 | MSE (bankrupt: BCE) |
| holdings | (4, 8) | final percent of each corporation held / 100 | MSE |
| short | (4,) | net-worth share 100 engine actions later (or at the end) | MSE |
| corp | (8, 4) | floated, final share price / 350, trains held / 4, cash / 1000 | BCE / MSE |
| president | (8,) | final president's canonical seat (4 = none) | CE |
| next run | (8,) | the corporation's next run's revenue / 400 (masked if it never runs again) | MSE |
| hex colour | (93,) | final tile colour (empty / yellow / green / brown / gray / other) | CE |
| hex token | (93, 8) | each corporation has a token in the hex at the end | BCE |
| hex route | (93, 8) | the hex is on each corporation's last run | BCE |

`DenseHeads` (an MLP per group) read the value network's own player and corporation tokens
(entity encoder), per-hex embeddings (map encoder) and trunk, captured by forward hooks, so the
network class is unchanged and a checkpoint trained with them loads as a plain one. Gradients
flow into every encoder: the hex targets train the map encoder on value-relevant features, which
the shared policy+value network could not (its track-lay sibling ranking was stuck at ~0.70-0.75
behind a policy-trained map encoder; az1's separate critic ranks track lays 0.94).

## Data and protocol (`scripts/value_gate.py`)

- **Games**: every eval_head_to_head collection of search players saved after the 2026-10-16
  search fix (139 collections; siblings_v2's source collections excluded), replayed in the Rust
  engine: 42,038 games, 449 skipped (pre-rule-change logs). 16 decision positions per game,
  uniform over the game past the auction: 672,608 positions (8.1 GB, nodes float16 -- exact up to
  2/3's rounding).
- **Splits**: test = the newest collections (`azl_u2100_*`, `azlc_u2070_*`, `ctrl_*`: 3,100 games,
  azl-family play); val = 5% of the rest by game-id hash (1,949 games, early stopping); train =
  36,989 games in a fixed random order, so smaller training sets are nested prefixes.
- **Training**: a separate value network (the whole network trains on value losses: entity and
  map encoders, trunk, heads), initialized from gen4's critic, AdamW lr 3e-5 (heads 3e-4), batch
  256, bf16 autocast, early-stopped on val win CE (patience 4 evaluations, at most 8 epochs).
  Outcome losses: win CE + placement-points MSE. Dense arm: + 0.3 x the dense loss.
- **Metrics**: test win CE (overall and by game third), winner hit, score MSE; sibling ranking
  (siblings_v2 vs az1's critic 280) for selected checkpoints.

## Reference values on the test games

| value | test win CE | early / mid / late | winner hit |
|---|---|---|---|
| gen4's critic (the init) | 1.019 | 1.360 / 1.150 / 0.508 | 0.542 |
| az1's critic 280 | 0.972 | 1.297 / 1.070 / 0.514 | 0.554 |
| az8 u170 (= azl u0's value side) | 0.956 | 1.296 / 1.065 / 0.471 | 0.555 |
| azl u1200 | 0.927 | 1.287 / 1.011 / 0.447 | 0.572 |
| azl u2100 | 0.923 | 1.284 / 1.006 / 0.445 | 0.574 |

(SE ~0.003 per row; paired differences between values on the same positions are much tighter.)

## Results

**Learning curve** (test win CE on the 3,100 held-out games, 49,600 positions; each run early-stopped
on val, all near convergence; the 36,989-game outcome-only run was still improving at the 8-epoch cap):

| training games | outcome losses | + dense (0.3) | outcome losses: early / mid / late | winner hit |
|---|---|---|---|---|
| 0 (gen4's critic) | 1.019 | -- | 1.360 / 1.150 / 0.508 | 0.542 |
| 1,150 | 0.9451 | 0.9456 | 1.289 / 1.049 / 0.462 | 0.563 |
| 2,300 | 0.9417 | 0.9421 | 1.289 / 1.041 / 0.460 | 0.565 |
| 4,600 | 0.9390 | 0.9386 | 1.285 / 1.035 / 0.462 | 0.567 |
| 9,200 | 0.9318 | 0.9315 | 1.281 / 1.025 / 0.454 | 0.572 |
| 18,500 | 0.9258 | 0.9257 | 1.278 / 1.017 / 0.448 | 0.574 |
| 36,989 | **0.9203** | 0.9208 | 1.274 / 1.010 / 0.441 | 0.576 |

1. **More games keep helping, at a steady rate.** From 4,600 to 36,989 games the test CE falls ~0.006
   per doubling (log-linear fit -0.0062), with no sign of flattening. The first ~1k games buy the most
   (1.019 -> 0.945: the critic adapting to these systems' play), then every doubling buys about the
   same. Naive extrapolation: ~0.911 at 100k games, ~0.901 at 300k, ~0.89 at 1M (a floor exists --
   early positions barely move: 1.289 -> 1.274 -- so treat these as upper bounds on the gain).
2. **Offline multi-pass training beats the in-run value on the same kind of data.** A separate value
   network trained on 36,989 saved games (8 epochs) reaches 0.920 on azl-family test games, better
   than azl u2100's own value (0.923), which saw ~111k self-play games but each position once at lr
   1e-5. Saving games and training the value over many passes extracts more per game.
3. **Dense targets don't move outcome CE at any size** (differences <= 0.0005, below noise). Their heads
   did learn (36,989 games: hex colour CE 1.6 -> 0.60, token BCE 0.69 -> 0.04, route BCE 0.69 -> 0.13,
   holdings MSE 0.02), so the features are there; win CE just doesn't need them.

**Sibling ranking** (siblings_v2, Spearman against az1's critic 280 -- agreement with that critic, not
ground truth: it is itself gen4's critic after 280 updates on az1 games):

| value | test CE | all | stock | track | train |
|---|---|---|---|---|---|
| gen4's critic | 1.019 | 0.838 | 0.857 | 0.805 | 0.814 |
| outcome losses, 1,150 games | 0.945 | 0.876 | 0.888 | 0.853 | 0.870 |
| outcome losses, 18,500 games | 0.926 | 0.860 | 0.894 | 0.778 | 0.793 |
| + dense, 18,500 games | 0.926 | 0.829 | 0.883 | 0.664 | 0.790 |
| outcome losses, 36,989 games | 0.920 | 0.808 | 0.857 | 0.685 | -- |
| + dense, 36,989 games | 0.921 | 0.799 | 0.864 | 0.619 | -- |
| azl u2100 (shared, in-run) | 0.923 | 0.752 | 0.762 | 0.696 | 0.765 |
| az1's critic 280 (the reference) | 0.972 | 0.993 | | | |

Agreement falls as training data grows (0.876 -> 0.860 -> 0.808) while test CE improves: the metric
tracks distance from the gen4-critic lineage, not ranking quality. Dense targets reshape track-lay
sibling order substantially (0.778 -> 0.664 at 18,500 games) with no change in outcome CE. Neither
metric can say whether that ranking is better; only play can.

## What this decides, and what it doesn't

- **A long run is worth it for the value, if the games are saved and the value is trained over many
  passes** (as a separate network or with its own encoders). The curve shows no saturation at 37k
  games; a months-long run produces millions.
- **Dense targets: not proven.** No outcome-CE gain; a large, unexplained change in track-lay ranking.
  The deciding test is a search match.
- **Next: two play tests** (480 games each, @200, the azl ladder's settings), both with azl u2100's
  policy as the prior so only the value differs:
  1. azl u2100 + the 36,989-game outcome-only value (a composite) vs azl u2100's own system: does the
     offline value beat the in-run value in play?
  2. azl u2100 + the dense 36,989-game value vs azl u2100 + the outcome-only one: do dense targets help
     search?

Data: `logs/value_gate/data` (gitignored); runs `logs/value_gate/runs/<arm>_<games>/learner/0.pth`;
results `logs/value_gate/curve.jsonl`, `siblings.jsonl`, `reference.jsonl`; queue `logs/value_gate/curve.sh`.
