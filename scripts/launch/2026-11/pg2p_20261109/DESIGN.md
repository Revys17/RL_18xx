# 2-player policy gradient + league from checkpoint 10 (2026-11-09 ~01:35; started by Claude without the user's go)

Step 3 of the user's order (S loop 2p -> AlphaZero 2p -> our strategies in 2p -> gen5); see
/home/revys/longrun/experiments/strategies2p_DRAFT/DESIGN.md for the whole step. Started at night because az0 was
stopped (auction-stall collapse, az0_2p_20261108/results.txt) and the user's standing instruction is to keep the GPU
busy with the next sensible experiment. Stage 1 only: the policy. Stop it any time (`systemctl --user stop
rl18xx-pg2p rl18xx-pg2p-evals`); nothing else depends on it.

## The run (run.sh): gen1's recipe (gen4's line from checkpoint 10), in 2 players
- From checkpoint 10 (gen1 began there too); critic: the 2p loop's V0 (a value trained on checkpoint 10's own 2p
  policy-only games, model_checkpoints_value/l2p_v0/promoted/0.pth).
- League: league_v7_gen4.json (31 members: checkpoint 10, pg/exploiter checkpoints, gen4), PFSP power 2, 3 opponent
  slots, own snapshots every 50 updates (4 kept); in 2 players the learner holds one seat, a member the other.
- gen1's optimizer settings: lr 2e-5, critic 3e-5, KL to an anchor moved every 50 updates (coef 0.02, target 0.08),
  learner and opponents at temperature 0.5, PPO clip 0.2, 1 epoch, 65,536 rows per update, GAE lambda 1.
- Reward `win` (= placement in 2 players).
- Starts: half the games at the real-rules waterfall auction (`--no-auction-unlock`, stall after 150 auction
  decisions scored on net worth), the rest the 2p human start / mid-game files (random endings 0.2, mid-game 0.3).
- Deployment az0-2p-213ec33: all 2026-11-08 audit fixes (tempered samplers, league crediting at game time, bankrupt
  players scored last). Train-shuffle rule on (every generator). Up to 2000 updates; no stop rule.

## Departures from gen4's own run (named)
- 2 players; the 2p critic; reward win (identical to placement in 2p); max 1500 decisions (2p games are longer per
  player); auction starts 0.5 with the real rules (gen5's choice, the user's: gen4 used 0.15 with auction_unlock).
- Bankrupt players scored last (audit fix; gen4 used Ruby's scoring).

## Measurements (evals.sh, unit rl18xx-pg2p-evals): policy-only, 1v1 both seatings, 400 games, all from the
real-rules auction (as az0's evals): u0 baseline (checkpoint 10 at 0.5 vs checkpoint 10 at 0.7 / gen4 at 0.5), then
every 50 updates u vs checkpoint 10 @0.7, vs gen4 @0.5, vs u-100. Watch `auction_stalls` in history.jsonl (az0's
collapse was an auction stalemate).
