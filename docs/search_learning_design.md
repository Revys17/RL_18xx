# Learning from search (1830): design draft

*2026-10-16. Status: draft, waiting on the verification matches below before
any implementation. Applies to 1830 only (the multi-title RL bridge is on hold).*

## Why now

Until 2026-10-16 the Rust search could not improve on the policy: every
operating-round node searched for seat 0 (fixed in 3217c28), duplicate leaves
were counted as visits (about 17 distinct positions per 64-readout search), and
minigo's exploration constant and `W / (1 + N)` value were sized for -1..1
values while ours are 0-1 win shares near 0.25, so visits followed the prior.
That is also why the 2026-10 AlphaZero loop never learned: its policy targets
were the prior plus noise. With the fixes and a value-driven setting (mean Q,
`c_puct_init` 0.15 in every round, priors at temperature 0.5, gen4's critic as
the value):

- where the search changes gen4's move, the search's move wins +0.09 to +0.17
  more for the mover in policy-only rollouts (`scripts/search_override_rollouts.py`);
- @200 vs gen4's argmax (no search), 2v2, 240 games: **0.942 +- 0.015**; the
  same search with the default exploration @64: 0.554 +- 0.032.

## Verification before building (queued on the GPU, `logs/orchestration/gpu_search_tests_round3.sh`)

1. gen4 argmax vs gen4 sampled at temperature 0.5 (how strong the 0.94 baseline is).
2. Value-driven search @200 vs gen4 at temperature 0.5, 2v2 and 1v3.
3. Value-driven search @64 vs argmax (does the gain survive a small budget?).
4. @800 vs @200 (scaling).

Go ahead if (2) clearly beats 0.5 in 2v2 and 0.25 in 1v3.

## The loop

The AlphaZero / expert-iteration idea on one GPU, using the tricks that let
KataGo reach AlphaZero-level Go with ~50x less compute, and Gumbel
AlphaZero's targets, which improve the policy with very few simulations:

1. **Self-play with playout-cap randomization.** Each decision gets the full
   search with probability `p_full` (start 0.25) and becomes a training
   example; the other decisions are played by the policy (temperature 0.5) and
   are not trained on. Start positions as today (human auction endings,
   random endings, mid-game cuts, auction starts).
2. **Gumbel root.** At searched decisions: draw Gumbel noise g, keep the top
   k moves by `g + logits` (k ~ 8-16), spend the budget n by sequential
   halving, play `argmax(g + logits + sigma(q))`. The policy target is
   `softmax(logits + sigma(completed q))`, with unvisited moves' q completed
   by the node's value. `sigma(q) = (c_visit + max_b N(b)) * c_scale * q`.
   This target is an improvement even at n = 16-32, where visit counts are
   noise. Below the root, the value-driven PUCT above.
3. **Training.** Policy: cross-entropy to the Gumbel targets on searched
   positions, plus the existing KL to a moving anchor (as in policy
   gradient) for stability. Price head: the search's chosen price cell as a
   target where a price was searched. Value (the critic that the search uses):
   win-share / placement outcomes of the self-play games, few positions per
   game with the stop-grad and lower value LR that fixed memorization before.
4. **Opponents.** All seats the current model at first (search makes it
   stronger than its own policy); later the league and exploiters, which the
   policy-gradient line showed prevent narrowing.
5. **Judging.** Each round: the policy alone vs gen4 at temperature 0.5 (2v2
   and 1v3, `eval_policy_only.py`), since the point is to distil search into
   the policy; and policy + search vs gen4 + search.

## Cost on one GPU

Policy-gradient generation runs ~84k games/hour (~15k network evaluations/s).
The search harness reaches ~5.8k/s today: batches of 8 leaves per request and
per-leaf position encoding in Python. Engineering first:

- many trees per worker in lockstep, so a request carries leaves from several
  games (as `policy_selfplay` batches positions);
- native leaf encoding (no `RustGameAdapter` per leaf).

Then, with ~750 decisions a game: `p_full` 0.25 at n = 32 is ~9x the evaluations
of policy-only play (~9k games/hour at the 15k/s ceiling); at n = 200 ~50x
(~1.6k games/hour). Each searched position carries a much better target than a
policy-gradient sample, so this is not the cost comparison that matters; test
how few games a round needs.

## Steps (each testable)

1. Harness throughput: lockstep multi-tree workers + native leaf encoding;
   target >= 12k evaluations/s on the eval and self-play paths.
2. Gumbel root selection and completed-Q targets in the Rust tree (an option;
   unit tests on rigged values: the target shifts toward the better move at
   n = 4; sequential halving spends exactly n).
3. A self-play generator with playout-cap randomization writing (state,
   legal, Gumbel target, price target, outcome) rows.
4. A training stage reusing the policy-gradient run directory layout
   (anchor, critic, snapshots, history.jsonl, league), with the target
   cross-entropy in place of the PPO loss.
5. A first run from gen4/1800 + its critic; judge every ~50 updates.

## Open questions (decided by the verification results)

- The search budget n and `p_full`; how low `c_puct_init` should go (0.15
  was a single guess) and whether mean-Q's first-play value needs a reduction.
- Whether the critic needs retraining before the loop starts (search follows
  its mistakes more readily at low exploration).
- Placement vs win reward for the value targets (the policy-gradient A/B
  favored placement).
