# Why azf got worse (diagnosis, 2026-11-01)

azf (search learning, one shared network, from `azf_start3` = checkpoint 10's policy + a value side distilled from
lr1's v4) ran 24 h under the 2-OR shuffle cap. Day-1 check: u1844 vs its start, 2v2 at 200 readouts, 480 games,
**0.433 +- 0.023**; it stopped itself at u1910. Five reviews ran in parallel: the code azf ran (deployment
longrun-2026-11g, a3aca14), its training metrics against azl's, the value side, the policy, and the games. Their
scripts and raw outputs: `scratchpad/diag/{algo,metrics,value,policy,behavior}` of session df10a4aa.

## The value side's own training did it

| system vs the start, 2v2 @200, 480 games | score |
|---|---|
| u1844 (its policy + its value) | 0.433 +- 0.023 |
| u1844's policy + the start's value side | 0.519 +- 0.023 |
| the start's policy + u1844's value side | 0.475 +- 0.023 |
| u600's network + the start's value side (one network) | 0.500 +- 0.023 |

In play the swaps point the same way but each is only ~1 SE from even, and the whole system (0.433) lost more
than the two swaps add up to (0.494; the gap is ~1.5 SE). Read with the probes below: the value side measurably got
worse; a policy trained on that value's targets plays worse with it than either part does alone.

- **Hybrids:** u1844's shared parts (map encoder, entity encoder, fusion, trunk, policy and price heads) under
  u0's value side (value entity encoder + win/score heads) reproduce u0's value exactly (sibling ranking 0.838 vs
  0.841, held-out CE within 0.0006); u0 with u1844's value side reproduces u1844. Trunk drift (4.2%) is harmless;
  the value side's updates did all the damage.
- **Sibling ranking vs v4** (how the value orders a position's children, 1,080 positions): 0.841 -> 0.632 at u1844
  (stock 0.907 -> 0.653, early game 0.877 -> 0.549); -0.07 already at u50. azl, at a third of the value lr, lost
  0.04 by u100 against its own teacher.
- **Held-out win CE** (paired, same positions, games no network trained on): +0.054 +- 0.005 on the value gate's
  test games, +0.030 +- 0.005 on lr1's ladder games, worse early-game CE on all four sets; winner hit
  0.569 -> 0.535. Its only gain (-0.01) is on its own generator's late positions.
- **A new bias toward the player to move:** p(mover) +0.010 to +0.019 on every held-out set (azl: +0.002 to
  +0.005). Value rows come only from the main system's seats, and in league games (20% of rows) those seats win
  0.345 per row against 0.273 in self-play; the network can't see which seats are league members. A search on that
  value over-rates moves that keep the turn.

### Why online value training hurts here
- ~65 games per update, ~140 rows per game sharing one outcome, every row trained, critic_lr 3e-5 (azl 1e-5),
  the raw weights served after every update. Stock-round siblings differ by ~0.006 in value; outcome CE can't see
  sibling order, and the noise of a few thousand games overwrote what v4 learned offline over 40k+ games with an
  EMA and a gate (lr1's value trainer needed the EMA because raw-weight jitter hid real gains).
- Half of each value target is bootstrapped (`value_bootstrap_weight 0.5`, lambda 0.97 per search, fast ones
  included): the value chases its own search values.
- The league's row selection (above).

## The KL anchor froze the policy for 90% of the run
`kl_coef` sat at its floor (0.005) until u107; KL to the start then crossed 1.5 x 0.03 and, because measured KL
lags ~20 updates behind the coefficient (the replay window), the x1.5-per-update rule wound it up to 0.97 by u120.
It stepped down once to 0.649 at u154 -- 30x the configured 0.02 -- and never moved again: KL stayed inside the
dead band (0.02-0.045). KL to the start 0.025 -> 0.040 over 1,700 updates (azl, no anchor: 0.14 -> 0.40).

## The policy alone got better, then fit worse targets
- Policy-only, near-greedy, 2v2 all seatings vs u0: u100 0.591, u300 0.612, **u600 0.614**, u1000 0.614,
  u1400 0.572, u1844 0.562 (+-0.016); u1844 vs u600 directly 0.440 +- 0.012 (1,800 games). 1v3: u600 0.339,
  u1844 0.311 (fair 0.25). azl u2100, 15x more drift, 0.573.
- The gain is in stock-round decisions. Human agreement unchanged (top-1 0.687 -> 0.682).
- After u1000 the policy kept moving toward the search targets (KL(target || pi) 0.345 -> 0.323) while getting
  weaker: the targets come from the degrading value (cq = logits + sigma(completed Q)).

## What the trained system does differently (day-1 games, paired, per player-game)
- More stock churn: +1.14 +- 0.15 share sales per player-game, more buys, +0.19 presidency dumps (the mover bias's
  prediction: selling keeps the turn).
- More of the shuffling the cap still allows: last-train buys between its own corporations 2.23 vs 1.72 per
  player-game (humans 0.12). All legal: the rule was on in every game, no purchase broke the cap, and every run of
  such purchases lasted 1 OR (1,063 runs) or 2 ORs (926), never 3 (`scratchpad/cap_check.py`). Before the rule,
  bots made 2.2-4.1 buybacks per player-game in runs of any length. Using the 2-OR allowance more, u1844 runs into
  the cap 48% more often (0.453 vs 0.307 forced purchases with the cap in force) and goes bankrupt in phase 6
  three times as often (29 vs 9). Once caught, both networks go bankrupt at the same ~20% rate.
- Weaker corporations: revenue -$340 +- 76 (-7.6%), fewer SR1 floats (0.50 vs 0.62), fewer 6/D trains.
- The score is lost in the 320 games without a bankruptcy (u1844 0.406 +- 0.027 there); bankruptcy games come out
  even. In azf's own games shuffling rose ~10% by u800 and stayed; bankruptcy stayed at ~31%.

## The network can't see the cap
Neither encoder (`encoder.py`, `engine-rs/src/encoder.rs`) has any input for the rule against train shuffling:
no `shuffle_runs`, no OR count. Whether a president's corporations have used 0, 1 or 2 of their ORs of last-train
purchases is invisible to the policy and the value; only the legal-move list changes, and only once the blocked
purchase is the move at hand. A 200-readout search rarely reaches the next OR's train step, so nothing in the
system can plan around the cap -- and 148 of the 160 bankruptcies in the day-1 games came with the cap in force.

## Context: azl didn't improve either
azl (no anchor, no league, value lr 1e-5) scored 0.562 at u300, then every checkpoint u300-u3300 sat within 1-2
SE of u0 on a fixed panel and below it on net-worth share (search_learning_design.md "Rating a long run").
Fixing azf's two faults brings the loop back to azl's plateau, not to progress. The one thing that has made
the system stronger is the offline value trainer (lr1: every saved game, many passes, EMA, a paired gate on fixed
data): ckpt 10 + v4 is the strongest system so far.

## Changes
1. **Don't train the value online in the generator.** Train it offline over every saved game (lr1's
   `value-train`: EMA, paired gate on fixed eval data, positions from every seat -- no league selection bias) and
   have the generator follow the promoted value. azf left ~107k games under the 2-OR rule in `saved_games/azf`,
   the data the value needs to learn what the cap does to shuffling.
2. **Give the value its own tower inside the one network.** The shared design caps a distilled value at 0.84
   sibling ranking vs v4 (its map/trunk features can't be copied in). A checkpoint holding a policy tower and a
   value tower is still one model that plays the whole game; the value tower can then be copied exactly from the
   offline trainer's promoted value.
3. **KL anchor:** off (azl) or a fixed small coefficient; if adaptive, small steps both ways (x1.05), a cap, and
   no dead band that can hold a wound-up coefficient.
4. **Monitors that would have caught azf at u50:** sibling ranking vs the teacher and held-out CE / p(mover) on a
   fixed set every 50 updates; near-greedy policy-only vs the start every 100 updates (900 games in ~65 s); the
   fixed-panel ladder for strength.
5. **League:** if the value ever trains on generator rows again, take value rows from every seat or from
   all-main games only.

## Follow-up: three value experiments (2026-11-02)
All play tests 2v2 @200, both seatings, 480 games (+-0.023); CE paired vs v4 on 30.6k positions of the 1,913 new-rule
eval games. Outputs and notes: `/home/revys/longrun/experiments/{capfeat,rollout_targets,one_trunk}_20261101/`.

| experiment | offline | play |
|---|---|---|
| cap inputs (flag `encode_shuffle_cap`, both encoders bit-identical) | CE -0.0012 all, -0.005 where a cap run is live | start policy + cap value vs start 0.492; vs same value without the inputs 0.505; 26% fewer sister last-train buys, bankruptcies unchanged |
| rollout targets (u600 at T=0.7, K=8; T=1 makes the value worse than v4) | C -0.0053; all-azf outcomes -0.0048; both stacked -0.0075 (rollouts help early, outcomes late; one rollout gets ~85%) | start policy + stacked value vs start 0.514 |
| one trunk: u600 policy + v4 value distilled into one network, value gradients into the trunk, dropout off | S0 (warm from u600): policy KL 0.0002, sibling vs v4 0.851 (start 0.841), CE = v4 | S0 vs u600 + v4 (its teachers) **0.541 +- 0.016** (960 games); vs start 0.527; S2 (x1.5 widths from scratch) vs start 0.540, vs S1 (1x) 0.512 |

- Value accuracy gains of 0.001-0.0075 nats are worth <= ~0.015 score: none of them shows in play.
- One network can hold both jobs at today's width when warm-started; the trunk gradient adds only track-lay ranking
  (+0.024 over a stop-grad control). Width helps only from scratch.
- The entity encoders carry dropout 0.1 (`nn.TransformerEncoderLayer`'s default; train-mode vs inference value KL
  ~0.1), but only supervised pretraining (checkpoint 10) and the one-trunk experiment's own trainer run in train
  mode: the value trainer, value gate, search learning and policy gradient all train in eval mode on purpose, so v4
  and azf never trained with dropout.
