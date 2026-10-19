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

## Status of steps 1-2 (2026-10-17, measured on CPU only)

**Step 1 (built).** `rl18xx/agent/alphazero/lockstep.py` +
`engine-rs/src/mcts_lockstep.rs`: `RustMCTSPlayer.start_search()` gives a
`TreeSearch`; `LockstepSearcher.round()` has every tree select its batch as
`tree_search` does, encodes the leaves in Rust into one stacked request per
network (`mcts._native_encode`, bit-for-bit `_rust_encode` on 146k human
positions) and scatters the sparse legal priors / values / price rows back in
Rust. A lockstep tree ends identical to the same tree searched alone.
`eval_head_to_head.py --games-per-task N` plays N games per worker from a
shared queue in lockstep. Also: tree nodes no longer clone the whole JSON
action log (they log only the actions since the root), which was most of an
expansion's cost late in a game.

Measured with `scripts/bench_search_harness.py` (48 human positions from SR1 to
SR6, @64, the value-driven settings, interleaved runs of master and the branch):

| | before (master) | after, one tree | after, lockstep |
|---|---|---|---|
| in-process, stub network: worker time outside the network per evaluation | 814-902 us | 704-764 us | 208-217 us (x32 trees) |
| in-process, gen4 + critic on 8 CPU threads: share of time outside the forward | 19.5-20.0% | 17.9-18.8% | 4.6-4.7% |
| inference server + 8 workers, stub network: evaluations/s | 4,108 | 4,760 | 15,912 (x16), 15,731 (x32) |

With the stub network the old path is bound by the server process, which spent
65% of its time receiving 17-leaf requests; lockstep requests carry ~180 (x16)
to ~330 (x32) leaves and the server sat idle 22-28%, so 8 workers on 16 CPU
threads were the limit (~2k evaluations/s each). What a lockstep worker still
spends per evaluation (`scripts/bench_leaf_costs.py`): ~140 us selecting and
expanding the leaf in Rust (cloning the parent's `BaseGame`, applying the move
and its JSON log entry, two legal-move enumerations of ~20 us), ~60 us freeing
finished trees (one game per node), ~14 us encoding, ~10 us of Python per leaf.
The next CPU win is a cheaper `BaseGame` clone (Arc-shared immutable parts,
no JSON logging off the played path) -- `game.rs` work.

**Step 2 (built, off by default).** `engine-rs/src/gumbel.rs`,
`SelfPlayConfig.gumbel_k` / `gumbel_c_visit` (50) / `gumbel_c_scale` (1) /
`gumbel_noise_scale` (1), eval `:gumbel=<k>,gscale=,cvisit=,cscale=`. Root:
top-k by g + log-prior, mctx's sequential-halving schedule over exactly the
budget (in-flight simulations count toward a move's visits, `dedupe_leaves`
duplicates don't, batches stop at the budget), the move
argmax(g + logits + sigma(q)) over the most-visited kept moves, the target
softmax(logits + sigma(completed q)) in `searches_pi` (unvisited moves get the
root's v_mix; sigma(q) = (c_visit + max N) * c_scale * q, q the mover's mean
0-1 share). Below the root: unchanged PUCT. Price-bearing root moves are
single candidates whose q aggregates their price grandchildren (the policy
head's view); the played price is the most-visited grandchild and the price
target their visit distribution, as before. A reused subtree's visits count
for q but not for the schedule. Logits are the tree's priors (after
`prior_temperature`), so with ptemp 0.5 the target improves on
softmax(logits / 0.5), not on the raw policy -- decide which in step 4.

**For step 3.** Drive self-play games through `LockstepSearcher` (16-32 games
per worker); a playout-cap "fast" move is a `PolicyRequest` in the same round
(or a `TreeSearch` with a small budget); write rows at decision time with
`_native_encode` instead of `extract_data`'s Python replay; the Gumbel target
is `player.searches_pi[-1]` and the price target `player.price_targets[-1]`
after `play_move`.

## Status of steps 3-4 (2026-10-17, CPU only; GPU pilot pending)

Built: `rl18xx/agent/alphazero/search_learning.py`, `main.py search-learning`.
Decisions taken (user, 2026-10-17): the policy is softmax(logits / T) with T
the starting run's `learner_temperature` (0.5 for gen4) and the loss is the
cross-entropy of log_softmax(logits / T) to the search target over the legal
moves; targets `visits` (the validated value-driven PUCT: mean Q, c_puct 0.15
everywhere, dedupe, par 8, priors at T) or `gumbel`; the price head's target
is the searched grandchildren's visits by price cell; playout-cap
randomization with p_full 0.25; all seats the learner (league hook:
`deal_seats` / `opponent_servers`); KL to a moving anchor at T; the critic on
outcomes with `critic_losses` (win CE + placement points); a replay window, no
importance ratios.

**Generation.** `Generator`: each worker keeps `games_per_worker` games going
across pool tasks (every `task_seconds` it returns the games that ended), all
of them in lockstep rounds; a searched decision gets a bare Rust tree
(`_tree`), its move goes straight onto the game (the most-visited price for a
price slot), and its row is written then (canonical encoding, legal moves,
target, price-cell target) and completed at the game's end (net-worth shares;
placement points for the score head; 32 rows flagged for the critic). Starts
as policy-gradient runs; a start that no longer replays is skipped.

**What changed in the design: pruned visit targets.** Measured on 40 human
positions with gen4 at T = 0.5 (`scripts/search_target_stats.py`):

| target | entropy | KL(target ‖ policy) | target mass where the policy < 1% |
|---|---|---|---|
| gen4's policy | 0.13 | 0 | 0.1% |
| raw visits @64 / @200 | 0.77 / 0.56 | 2.39 / 1.58 | 18.6% / 12.3% |
| pruned visits @64 / @200 | 0.16 / 0.17 | 0.18 / 0.35 | 1.7% / 2.2% |
| Gumbel @64 / @200 | 0.13 / 0.18 | 0.06 / 0.27 | 1.4% / 3.1% |

With first-play urgency at the parent's value, most root moves get a visit or
two whatever their prior, so the raw distribution would teach the sharp policy
those exploratory moves. `visit_target` "pruned" (default) applies KataGo's
policy-target pruning (`RustMCTSPlayer.pruned_child_n_at_root`: every move but
the most visited keeps only the visits its PUCT score needed against the most
visited one; a move left with <= 1 visit gets 0). In all 40 positions the
search's most visited move was the policy's argmax (@64 and @200): the
signal is the shape and the few disagreements, so watch `top1_agreement`
and `search_kl` in the history.

**Training.** `_update`: `steps_per_update` steps of `minibatch` rows drawn
from the window (microbatched), the critic on the sampled rows still flagged
(each position once). History per update: `ce`, `search_kl` (KL(target ‖
policy)), `target_entropy`, `top1_agreement`, `price_ce`, `kl_anchor`,
`critic_*`, `reuse_update` / `reuse_run`, `row_age_mean`, games/h, searched
rows/h, evaluations/s.

**CPU rates per worker** (`scripts/bench_search_selfplay.py`, one core, in
process, human / random / mid-game / auction starts as gen4's run, p_full 0.25):

| network | n_full | decisions/s | searched rows/h | evaluations/s | evaluations per decision | games/h (608-decision games) |
|---|---|---|---|---|---|---|
| free (stub) | 64 | 180 | 162k | 3,170 | 17.6 | ~1,070 |
| free (stub) | 200 | 77 | 70k | 4,021 | 52.1 | ~460 |
| gen4 + critic, 8 CPU threads | 64 | 13.6 | 11.7k | 236 | 17.4 | (forward-bound) |

The worker spends ~240-300 us per evaluation outside the network, so on the
GPU the network sets the rate: at ~11-15k evaluations/s (a composite costs
~1.33 forwards) about 630-850 decisions/s at n_full 64 (~150-210 rows/s) and
210-290 at n_full 200 (~55-70 rows/s); 16 workers already supply more.

## Speed (2026-10-18, GPU shared with az1 -- relative numbers, back to back)

az1_20261017 ran at ~770 games/h, ~13.1k network evaluations/s, its server
spending 74% in the forward, 17% receiving requests. Four changes, each an
option or proven to give the same results:

**Where the forward goes** (`logs/perf/staged*.py`-style stage timing, CUDA
events, gen4 + its critic in TF32 as served, requests recorded from the
generator; GPU ms per batch):

| stage | B = 64 | B = 256 | B = 600 | B = 668, maps deduped (262) |
|---|---|---|---|---|
| hex map transformer, 4 attention layers (policy / critic) | 0.7 / 0.7 | 5.4 / 5.3 | 21.0 / 20.8 | 6.5 / 5.6 |
| hex input, structural bias, out-LN + pool (each net) | 0.1 | 0.3 | 1.0-3.5 | 0.4 |
| economic-entity transformer (each) | 0.2-0.7 | 0.2 | 0.4 | 0.4 (+ launch lag) |
| cross-modal fusion (each) | 0.1 | 0.2 | 0.4 | 0.4 |
| FiLM blocks x6 (each) | 0.1 | 0.2 | 0.2 | 0.2 |
| policy head: hex + tile / rotation grid / place token / rest | 0.2 | 0.5 | 0.5 / 0.5 / 0.4 / 0.2 | same |
| price head, value + score heads, softmax | < 0.1 each | | 0.1 each | |
| reply gather + device-to-host | 12 (sync on launches) | 6.5 | 4.4 | 4.0 |
| total GPU | 15 (launch-bound) | 21 | 55 | 24 |

The hex map transformer is ~77% of a 600-state forward. Inside it (kernel
table): the attention core over 93 hexes with a per-state additive bias --
fp32 batched matmuls (magma), the bias add, the scale, the softmax, layout
copies -- is ~14 of 23 ms, memory-bound; the TF32 GEMMs ~4 ms. Small batches
are launch-bound (~1,600 kernels per composite forward).

1. **Packed server fast path** (default; `InferenceServer.fast_path`): lockstep
   requests go to the model as stacked arrays (`packed_arrays` /
   `packed_inputs` / `run_packed`, one set of device inputs for both networks)
   instead of 600 per-state tuples stacked twice; numpy, not torch, on the CPU
   (torch's thread pool cost 100x on the loaded machine: a 12 MB `torch.cat`
   190 ms with 24 threads). Bit for bit the old replies (27k positions: max
   |diff| 0). Server-only replay (`scripts/bench_inference_server.py`): 1.1-1.5x.
2. **Each distinct hex map once** (`--dedupe-maps`): the map encoder's only input
   is the map, and a request's leaves mostly share their trees' maps (6% distinct
   in SR1 / auction requests, 39% in SR2-6 + ORs, ~21-35% in generation).
   Rust dedupes while encoding (`map_index`), the server runs the map encoder on
   the distinct maps and gathers. GPU per batch 81 -> 27 ms (mid-game), 62 -> 6
   (early); outputs within 3e-4 of the plain batch's (TF32's own noise vs fp32:
   1e-2), every argmax equal; bit for bit on the mid-game capture.
3. **Server replicas** (`--server-replicas N`): a collector thread that receives
   the next batch during the forward measured *slower* (GIL contention: 20.4k vs
   24.2k states/s), so the overlap is between processes: N servers, each with its
   workers' requests. fast + dedupe: 24.2k -> 39.4k states/s with 2, 3 a further
   1.26x (server-only replay, 8 clients).
4. **Tree reuse** (`--reuse-tree total`): each game keeps its tree, advanced along
   every move played (`advance_root_to` with the played price, `root_log_matches`
   to confirm the position), and a search stops at 1 + n root visits counting the
   inherited ones (KataGo's maxVisits; `additional` = eval_head_to_head's count,
   n more on top). gen4, az1 settings: 98.9% of searches start on a kept subtree,
   86.7% of their final root visits inherited, evaluations per decision 90.5 ->
   21.2; every decision still sees >= 1 + n root visits (148 on average vs 89).
   A Gumbel root's sequential halving still spends its n (inherited visits count
   for q). Worker RSS 1.35 -> 1.8 GB (a tree node is ~124 KB, mostly its game
   clone); `additional` trees grow ~4.5x (capped by `--reuse-max-root-visits`).

**Generation end to end** (`scripts/bench_search_generation.py`, 10 workers x 32
games, 150 s each, the same hour):

| config | decisions/s | evaluations/s | evaluations / decision | searched rows/h |
|---|---|---|---|---|
| before (unpacked server) | 68.4 | 6,255 | 91.4 | 63k |
| packed fast path | 71.9 | 6,519 | 90.6 | 66k |
| + dedupe + 2 replicas | 112.7 | 10,054 | 89.2 | 100k |
| + reuse total (1 replica, no dedupe) | 254.9 | 5,026 | 19.7 | 228k |
| all: dedupe + reuse total + 2 replicas | 492.5 | 11,027 | 22.4 | 443k |

Not done: `torch.compile` of the trunk (dynamic shapes) cut its GPU time ~17%
and launches ~30%, with TF32-sized differences (6.9e-3 on trunk features) and a
recompile on every reload (a new model object); CUDA graphs need bucketed batch,
map and tile-lay-row counts. SDPA / `baddbmm` attention variants: within +-10%
at the deduped sizes. Next exact step: cache map encodings across rounds (a
tree's root map recurs in every round of its search; per-worker map ids, a
server-side cache dropped on reload). Changes to the network (retraining,
strength to re-check): one map encoder for policy and critic (~-45% forward), a
lighter critic, a smaller or local-attention map encoder, bf16 for the trunk
only (the policy head stays fp32; prove no argmax moves first).

## Open questions (decided by the verification results)

- The search budget n and `p_full`; how low `c_puct_init` should go (0.15
  was a single guess) and whether mean-Q's first-play value needs a reduction.
- Whether the critic needs retraining before the loop starts (search follows
  its mistakes more readily at low exploration).
- Placement vs win reward for the value targets (the policy-gradient A/B
  favored placement).

## What the pilots and az1 taught us, and az2 (agreed with the user 2026-10-18)

**Pilots (2026-10-17, 100 updates each, the critic frozen or trained on mostly-policy games):**
every one made the policy worse when it plays alone (pruned visits: 0.481 / 0.475 / 0.278 at
updates 20 / 50 / 100, policy-only 2v2 vs gen4@0.5; one-hot targets 0.395 at 20; gen4 at T=1
0.407; checkpoint 10 flat). At the positions where a pilot's top move changed, the new move was
no better than the old one in policy-only rollouts (-0.019 / -0.009 / +0.001 +- 0.016), while
the search's own changes win +0.09..+0.17: the policy did not absorb the search's improvements,
it got flatter. Missing: the critic learning from search-played games (75% of the pilots' moves
were raw-policy moves), and gen4 is sharper than the search's visits (86-95% top-move agreement).

**az1 (the full loop: every move searched, the critic trained on the outcomes):** stable where
the pilots weren't (KL to gen4 at its 0.02 limit, entropy ~0.15, top-move agreement 0.87 ->
0.92), even with gen4 at updates 25 / 50 (system @64 0.487 / 0.504, policy-only 0.497), its
critic slowly better on held-out search games (win CE 0.909 -> 0.892 by update 50, mid and late
game). Limited by throughput (~600-770 games/h) and by one outcome per 700-decision game.

**az2, in order (each an option, judged against az1 by system strength per GPU-hour):**
1. One shared network: policy + value heads on one trunk (AlphaZero / KataGo / MuZero), about
   1.33x throughput (no separate critic forward). gen4's own value head was never trained during
   policy gradient: fit it to the critic's predictions and to outcomes first, then switch.
2. Better value targets: n-step bootstrapped / lambda-return targets from the search's root
   values later in the game (MuZero; KataGo's short-term value heads) instead of only the final
   outcome; the score / net-worth head blended into the search's utility (KataGo; tested at play
   time first: `#blend=<w>`); more auxiliary targets (each player's final net worth, cash,
   floated corporations); every searched position per game for the value (not 32: rows are used
   once, so AlphaGo's one-position-per-game anti-memorization doesn't apply).
3. Policy surprise weighting (KataGo): oversample rows where the search target disagrees with
   the policy -- the 8-14% that carry the signal.
4. Gumbel root + completed-Q targets at small budgets (16-64 readouts) if learning per GPU-hour
   holds up -- the route to thousands of games per hour.
5. Reanalyze (MuZero): re-search replay-buffer positions with the latest network for fresh
   targets without new games, if game generation still dominates the cost.
Not taken from MuZero: the learned dynamics model (we have the exact engine). Pure performance
work first (tree reuse in the lockstep generator, compiled forward, server pipelining).

## az2's training side: built (2026-10-18; items 1-3, all options, off by default)

| option (`main.py search-learning`) | default | what |
|---|---|---|
| `--shared` | off | one network: the server loads the learner alone and the search reads its own value heads (no critic forward); the value loss joins the policy loss on the same forward; the value heads train at `--critic-lr` in their own optimizer group, each group clipped on its own |
| `--value-loss-weight` / `--score-loss-weight` | 1 / 1 | `--shared`: the win CE's and score MSE's weight in the learner's loss (the trunk's share of the value signal) |
| `--value-stop-grad` | off | `--shared`: value heads on detached trunk features (the trunk learns only the policy) |
| `--value-head win\|score\|blend=<w>` | win | the search's value; `blend=<w>` = (1-w) win + w score, on the shared network too (`composite_model.SharedValueModel`, one forward, the server's packed fast path) |
| `--value-nstep n` | 0 (outcome) | win target (1-w) outcome + w * the root value n searches later (MuZero n-step, gamma 1) |
| `--value-lambda l` | off | or the TD(lambda) return over the game's later root values (G_i = (1-l) v_{i+1} + l G_{i+1}, the outcome after the last) |
| `--value-bootstrap-weight w` | 0.5 | the bootstrap's share of the win target (with n-step or lambda) |
| `--aux-loss-weight` | 0 | training-only heads (`ModelTransformerConfig.aux_seat_targets` / `aux_global_targets`, never run by `forward`): final net-worth share + cash share per seat (MSE), corporations floated (BCE) |
| `--critic-positions-per-game 0` | 32 | every searched row trains the value, once (verified by a test) |
| `--surprise-weight a` / `--surprise-cap` | 0 / 8 | KataGo's policy surprise weights on the target CE: (1-a) + a s / mean(s), s = KL(target \|\| policy) at T (the microbatch), capped, mean 1 |
| `--target gumbel` | **now the default** | every search (full and fast) uses the Gumbel root |

Every search records its root value (mean backed-up value per seat, absolute order; read before
a kept tree advances). History adds `critic_outcome_ce` (the value on fresh rows vs the outcome,
comparable to az1's `critic_ce` whatever the target), `search_value_ce` (root values vs outcomes),
`aux_*`, `surprise_*`, `value_grad_norm`.

**Why n-step / lambda on the main head, not KataGo's short-term value heads.** The point is a
lower-variance target for the value the search reads (one outcome per ~750 decisions); KataGo's
short-term heads are auxiliary -- the head the search reads still learns the outcome -- and would
need new served heads. Bootstrapping biases toward the current value net, hence the mix with the
outcome (`--value-bootstrap-weight`) and lambda close to 1 (0.97: a horizon of ~33 searches, ~8
decisions per seat).

**Value-head refit (`scripts/refit_value_heads.py`).** gen4's own heads on held-out search games
(az1_u50_sys64, 7,670 positions: every 25th action after the 20th) and the refits, fit on 3,840
other gen4-search games (48 positions each, 10% held out by game for the choices) toward 0.75 *
critic + 0.25 * outcome:

| value | win CE | winner | CE early / mid / late |
|---|---|---|---|
| gen4's critic | 0.909 | 0.597 | 1.290 / 0.981 / 0.402 |
| gen4's own heads (stale) | 1.198 | 0.504 | 1.337 / 1.347 / 0.881 |
| heads only (trunk frozen: the policy unchanged) | 0.951 | 0.564 | 1.288 / 1.004 / 0.514 |
| + fine-tune (4 epochs, trunk lr 5e-5, KL anchor 1) | **0.936** | 0.574 | 1.280 / 0.981 / 0.501 |

The fine-tune keeps the policy: KL(gen4 \|\| tuned) 0.0026 at T 0.5 on the held-out positions
(p99 0.057), top move unchanged at 99.3%. The late game is the gap left (the critic's own trunk
sees it better); the shared training continues from there. (A first fit kept the first k of 2k
sampled positions in game order -- mostly openings -- and scored 0.992 / late 0.63.)

**Real rows, GPU smoke test** (refit learner, `--shared`, Gumbel 64/16, reuse total, 12 workers x
8 games, 6 updates of 1,024 rows): surprise weights' effective sample size 0.31 of the rows, the top
10% of rows carry 47% of the weight, the 8.5% whose target's top move is not the policy's carry
37% (4.4x their share), 2% capped; target entropy 0.15-0.16, top-1 agreement 0.91-0.92,
search_kl 0.39-0.51; fresh-row value CE vs outcome 0.84-1.24 (whole games, 6-11 games per update).

**Generation, back to back** (`scripts/bench_search_generation.py`, 10 workers x 32 games, 150 s,
dedupe + reuse total + 2 replicas, GPU shared with az1):

| network, search | decisions/s | evaluations / decision |
|---|---|---|
| gen4 + critic, Gumbel 200/48 | 106.8 | 86.6 |
| shared, Gumbel 200/48 | 125.9 (1.18x) | 86.5 |
| shared, PUCT 200/48 | 390.5 | 22.7 |
| shared, Gumbel 64/16 | 267.0 | 28.0 |

A Gumbel root spends its whole budget every search (inherited visits inform q, not the
schedule), so `--reuse-tree total` saves nothing with Gumbel: at az1's budget it is ~3x the
evaluations of PUCT with reuse. A Gumbel budget that counts inherited visits would be a tree
change; small budgets (Gumbel's strength) are the cheap route.

## Why the shared network searched badly (2026-10-18)

The refit shared network (az1 update 280's policy, value heads fit to az1's critic, trunk tuned 4
epochs at 5e-5) scored 0.308 at 200 readouts against gen4 + critic, where az1 u250 (policy +
critic) scored 0.613, though its held-out win CE was 0.926 against the critic's 0.881.

**Not a serving bug** (`scripts/diag_value_paths.py`): on 1,742 held-out positions the values
through the refit's evaluation, the full forward, the packed fast path, a real inference server
(2 replicas, lockstep, `dedupe_maps`) and a search tree's root (rotated back from absolute seats)
agree to 3e-6, for the shared network and the composite alike; the round type the model reads
back from the state equals the encoder's in every position.

**The value can't rank a position's children** (`scripts/diag_sibling_values.py`): 1,080 held-out
positions (az1_u250/u300_sys200, az2_u0_sys200), every child built as the search builds it
(forced moves collapsed, the price head's price), each child's value for the parent's mover
against az1's critic 280 (Spearman across siblings / the critic's best child / the critic's loss
choosing the source's best child):

| value | Spearman | top-1 | regret | stock | track | train buy |
|---|---|---|---|---|---|---|
| az1's critic 250 | 0.942 | 0.831 | 0.001 | 0.945 | 0.944 | 0.927 |
| gen4's critic | 0.838 | 0.689 | 0.003 | 0.857 | 0.805 | 0.814 |
| shared, stale heads | 0.248 | 0.312 | 0.025 | 0.269 | 0.095 | 0.280 |
| shared, heads refit | 0.422 | 0.357 | 0.017 | 0.516 | 0.088 | 0.305 |
| shared, refit + tune (played 0.308) | 0.475 | 0.413 | 0.014 | 0.570 | 0.122 | 0.373 |
| az2b update 52 (52 updates of shared training) | 0.478 | 0.417 | 0.014 | 0.569 | 0.147 | 0.384 |
| the policy's prior | 0.376 | 0.434 | 0.010 | 0.438 | 0.109 | 0.426 |

The refit ranks siblings barely better than the policy prior; value spreads are not compressed
(std across siblings 0.028 vs the critic's 0.025: the differences are there, in the wrong order).
Win CE can't see this: the mover's value varies 27x more between positions than between siblings
(variance 0.038 vs 0.0014), and the critic's own values shuffled at random among each position's
siblings lose only 0.0099 nats -- a fifth of the 0.045 CE gap.

**The policy network lacks the features, at every depth** (`diag_value_heads_on_children.py`,
`diag_value_probe_depth.py`): heads fit to the critic on 9.5k positions *and their 130k
children* ranked 0.48-0.50 on the shared trunk, 0.94 on the critic's trunk and 0.92 on gen4's
critic's; fit on the policy network's player tokens, its fusion output or any of its six blocks
(or with a 1024-wide head), 0.50-0.54, while the critic's player tokens alone -- its entity
encoder, which never sees the map -- give 0.873. gen4's value heads were never trained during
policy gradient, so nothing in the policy's encoders was ever shaped by value; the critic
lineage's were, over hundreds of thousands of games.

**Training the shared trunk on the critic is slow** (`distill_value_on_children.py`, positions +
children, KL(policy || tuned) at T 0.5 anchoring the policy): 0.475 -> 0.578 (470k states,
2 epochs) -> 0.693 (2M states, 1.5 epochs, trunk lr 3e-4, value weight 3) -> 0.733 (+1 epoch on
3.4M states), the policy drifting (KL 0.005, policy-only 0.483 vs 0.504) -- at 0.578 it played
0.415 +- 0.042 (142 of 240 games). The reverse (a critic network distilled toward az1's policy)
kept the ranking (0.876) but its policy (KL 0.16, top-1 0.92) played 0.136 policy-only.

**Fix: a value-only entity encoder** (`ModelTransformerConfig.value_entity_encoder`,
`scripts/build_value_encoder_network.py`): the value heads read their player tokens from a
second entity encoder (4.2% of the parameters; a 512-state forward 102 vs 104 ms on the shared
GPU); map encoder, fusion and trunk stay shared, and the policy path is bit for bit the policy
network's. Copied from az1's critic 280 and untrained it
ranks 0.747; with the encoder and the heads fit to the critic on 3.4M positions + children
(policy frozen, ~20 min) **0.920** (top-1 0.812, regret 0.001; stock 0.966, track 0.747, train
0.902), held-out win CE 0.882 (the critic 0.881; by stage 1.235 / 0.935 / 0.416). **It played
0.667 +- 0.030** (240 games, 200 readouts, `/0.05:mean,dedupe,par=8,ptemp=0.5`, vs gen4 + critic;
seat win rates 0.27-0.38; the refit 0.308, az1 u250 / u300 with their critics 0.613 / 0.658), its
search targets as close to its policy as policy + critic's (KL(pruned visits || policy) 0.197 vs
0.162 at 200 readouts on 60 human positions). Checkpoint:
`model_checkpoints_pg/shared_diag_venc1_20261018/learner/280.pth`. Training the shared trunk too
afterwards (trunk lr 5e-5, KL weight 3, 0.5 epoch) moved the ranking only to 0.921. `search-learning --shared`
trains the encoder with the value heads at `--critic-lr`; `value_stop_grad` stops the value loss
at the shared layers but still trains it.

**For a shared AlphaZero run:** judge a shared network's value by sibling ranking, not win CE;
the value loss on played positions at the policy's trunk lr (az2: 5e-6, a sixth of the critic's
3e-5) moved the ranking nowhere in 52-90 updates (0.475 -> 0.478); a value that misranks
children also makes the search's targets noisy: at 200 readouts on 60 human positions,
KL(pruned visits || policy) 0.47 for the refit vs 0.16 for policy + critic, 4.1% vs 1.4% of the
target on moves the policy gives under 1% -- what az2b's policy then learned (entropy 0.29 ->
0.55; it also ran without a KL anchor).

## Q-based policy targets (2026-10-18/19)

Every run without an anchor flattened (az2b, az3, az4; az3: target entropy 0.24 -> 0.58, policy
0.19 -> 0.94 over ~300 updates, near-greedy policy-only 0.514 -> 0.188 vs gen4). The idea tested here: a target
in which a move gains probability because its Q is better, not because the prior sent visits there.

**Built (all options, off by default; the default tree is bit for bit the old build's).**

| option (`main.py search-learning`) | what |
|---|---|
| `--target cq` | Gumbel MuZero's improved policy softmax(logits + sigma(completed q)) on the PUCT root (`search_targets.completed_q_target`): logits = the tree's priors (after `prior_temperature`), unvisited moves at the root's v_mix, sigma = (`--gumbel-c-visit` 50 + max N) * `--gumbel-c-scale` 1 * q on the mover's 0-1 share; `--cq-rescale` min-max rescales the completed q first (mctx's default, there with c_scale 0.1). Equal to `gumbel_policy_target` on a Gumbel root (tested). The move played is unchanged (most visited / explore temperature). |
| `--target rpo` | Grill et al. 2020's pi_bar = lambda_N pi / (alpha - q), alpha by bisection in [max(q + lambda pi), max q + lambda], lambda_N = c sqrt(sum N) / (\|A\| + sum N); `--rpo-c` 0 = the tree's own effective c_puct at the root (2 ln((1 + N + c_base) / c_base) + 2 c_puct_init, ~0.11 here); `--rpo-unvisited vmix` (default, as cq) / `parent` / `min`; `--rpo-qnorm` min-max q over the root's value and visited children first. |
| `--fast-row-weight w` | fast searches (`--n-fast`) also become rows (same kind of target; `visits` = pruned visits) weighted w in the policy and price loss; history keeps the plain keys over full rows (comparable to earlier runs) and adds `*_fast`, `fast_rows_trained`, `new_fast_rows`, `fast_rows_per_hour`; `rows_per_update` counts full rows only (the update cadence per game is unchanged), the window holds both. `--no-fast-row-value`: fast rows train the policy only (default: the value trains on every row once, as on full rows -- they are positions with outcomes like any other). |
| `--q-normalize` (`SelfPlayConfig.mcts_q_normalize`, eval `:qnorm`, `RustMCTSPlayer.set_q_normalize`) | MuZero-style min-max Q in PUCT selection, per node over the node's own mean value and its visited children's (mctx `qtransform_by_parent_and_siblings`, the default of its `muzero_policy`), not tree-wide: the mover changes from node to node and a tree-wide range would mix four seats' values and the whole tree's spread into one scale. A virtual loss still counts as a zero-value visit; the pruned target reads the normalized terms. Needs mean Q; c_puct is then sized for a 0-1 range (MuZero's 1.25 is `/0.625`). |

Rust: `root_target_stats()` (legal, N, the mover's mean q, priors, completed q, v_mix, raw value,
root value, effective c_puct, open-price slots; a price slot's N and q aggregate its price
grandchildren, the policy head's view), `set_q_normalize`. Bit-for-bit check: seeded HashNet
searches (minigo Q, mean Q, mean Q + FPU, Gumbel; alone with `suggest_move`, a second search on the
kept subtree, and lockstep), digests of every root statistic from the old and the new build, three
processes each: 184 of 192 searches deterministic in both builds (8 hit HashMap-order ties between
price grandchildren -- run-to-run noise of both builds), all 184 identical across builds. Tests:
`test_search_targets.py` (14), `tests/agent/alphazero` 905 passed, `cargo test` 216 passed.

**Fast-row cost.** No new encoding: `_next_need` already encodes every decision
(`encode_canonical_py`, ~77 us) to find the mover, and a fast search kept it unused. A target is
29 us (cq) / 310 us (rpo, numpy bisection) / 0.8 ms (pruned visits: the dense 26.5k-slot vector);
a row is 20.9 KB pickled. With n_fast on 75% of decisions a game gives ~4x the rows: at az4's
~400k full rows/h, ~1.2M fast rows/h (~25 GB/h through the result queue); the training cost per
update is unchanged (steps x minibatch), each row is reused ~4x less.

### Offline: which target carries signal at 48 and 200 readouts (`scripts/search_target_quality.py`)

600 decision positions (past the auction; 40 games x 5 from each of az4_u200_vs_az1_u320,
az3_u200_vs_az1_u320, az1_u300_sys200; stock 308, track 101, train 95, dividend 80, other 16),
the generator's search (mean Q, c_puct 0.05, FPU 0, dedupe, 8 leaves, priors at T 0.5) on fresh
trees by az4 u100 (the shared value-encoder network the az5 arms start from; its policy's entropy
here 0.598). Every target is computed from the same trees. Two references, because they fail
differently:

- **self**: the same network @800 (more search; 800 vs 1600 agree on the move 94.5%). It shares
  the judged search's value errors: a 200-readout tree is nearly the first 200 readouts of the 800
  one, so a target that trusts the 200 tree's q is rewarded for agreeing with the same noise.
- **az1**: az1 u320 + critic 320 @800 -- an independent value of about the same strength as a
  search system (against gen4 + critic @200: az1 u300 0.658, the az4 lineage's start 0.667; az3
  u200 0.538 against it). Its own errors make it a pessimistic judge; the truth is between the two.

Q_ref(a) = the reference tree's mean value of root move a for the mover (unvisited: its v_mix).
regret = Q_ref(reference move) - E_target[Q_ref]; spread = target mass on moves that are neither
the policy's argmax nor the reference's move (the prior's own: 0.169); recall = among the 18-19% of
positions where the reference's move is not the policy's argmax, the share where the target's
argmax is the reference's move. Standard errors: top-1 ~0.016, regret ~0.0006 (paired differences
between targets on the same trees ~0.0001-0.0004).

| budget | target | H | KL(t‖π) | low (π < 1%) | spread (az1) | top-1 self / az1 | regret self / az1 (x1e-3) | recall self / az1 |
|---|---|---|---|---|---|---|---|---|
| -- | policy prior | 0.598 | 0 | 0.011 | 0.169 | 0.817 / 0.810 | 10.94 / 9.93 | 0 / 0 |
| 48 | most visited (one-hot) | 0 | 0.360 | 0.008 | 0.035 | 0.833 / 0.812 | 4.59 / 4.95 | 0.19 / 0.13 |
| 48 | raw visits | 1.165 | 1.026 | 0.189 | 0.388 | 0.822 / 0.800 | 16.07 / 14.94 | 0.16 / 0.11 |
| 48 | pruned visits | 0.341 | 0.269 | 0.023 | 0.127 | 0.830 / 0.808 | 4.92 / 5.55 | 0.18 / 0.13 |
| 48 | pruned visits² | 0.216 | 0.293 | 0.019 | 0.084 | 0.830 / 0.808 | 4.66 / 5.16 | 0.18 / 0.13 |
| 48 | **cq** (c_scale 1) | 0.418 | 0.207 | 0.016 | 0.147 | 0.837 / 0.808 | 4.17 / 5.13 | 0.29 / 0.21 |
| 48 | cq c_scale 0.3 | 0.493 | 0.048 | 0.008 | 0.139 | 0.832 / 0.817 | 5.99 / 6.42 | 0.14 / 0.11 |
| 48 | cq c_scale 3 | 0.320 | 0.571 | 0.036 | 0.199 | 0.817 / 0.752 | 2.81 / 4.78 | 0.55 / 0.31 |
| 48 | cq rescaled, c_scale 0.1 (mctx) | 0.340 | 0.587 | 0.040 | 0.214 | 0.760 / 0.740 | 3.69 / 5.54 | 0.41 / 0.31 |
| 48 | rpo (tree's c ~0.11) | 0.481 | 0.489 | 0.057 | 0.197 | 0.810 / 0.763 | 4.24 / 5.73 | 0.45 / 0.29 |
| 48 | rpo c 0.5 | 0.538 | 0.075 | 0.016 | 0.155 | 0.830 / 0.813 | 7.03 / 7.15 | 0.16 / 0.12 |
| 48 | rpo, q min-max | 0.148 | 1.941 | 0.179 | 0.362 | 0.640 / 0.595 | 2.34 / 4.97 | 0.65 / 0.45 |
| 200 | most visited (one-hot) | 0 | 0.454 | 0.013 | 0.058 | 0.880 / 0.817 | 2.55 / 3.67 | 0.40 / 0.21 |
| 200 | raw visits | 0.887 | 0.642 | 0.116 | 0.276 | 0.878 / 0.817 | 7.05 / 8.11 | 0.39 / 0.21 |
| 200 | pruned visits (az4) | 0.411 | 0.306 | 0.025 | 0.145 | 0.880 / 0.817 | 2.72 / 4.32 | 0.40 / 0.21 |
| 200 | pruned visits² (az5b) | 0.213 | 0.353 | 0.019 | 0.093 | 0.880 / 0.817 | 2.25 / 3.82 | 0.40 / 0.21 |
| 200 | **cq** (c_scale 1) | 0.358 | 0.320 | 0.017 | 0.144 | 0.893 / 0.805 | 2.01 / 3.87 | 0.54 / 0.28 |
| 200 | cq c_scale 0.3 | 0.420 | 0.103 | 0.007 | 0.121 | 0.857 / 0.825 | 3.92 / 4.87 | 0.25 / 0.17 |
| 200 | cq c_scale 3 | 0.280 | 0.873 | 0.062 | 0.221 | 0.862 / 0.745 | 0.28 / 3.42 | 0.83 / 0.46 |
| 200 | cq rescaled, c_scale 0.1 (mctx) | 0.267 | 0.988 | 0.084 | 0.245 | 0.783 / 0.703 | 1.47 / 4.13 | 0.63 / 0.35 |
| 200 | rpo (tree's c) | 0.457 | 0.585 | 0.062 | 0.190 | 0.900 / 0.777 | 2.03 / 4.55 | 0.75 / 0.38 |
| 200 | rpo c 0.5 | 0.511 | 0.119 | 0.016 | 0.145 | 0.852 / 0.820 | 5.30 / 5.93 | 0.25 / 0.16 |
| 200 | rpo, q min-max | 0.128 | 2.003 | 0.186 | 0.365 | 0.685 / 0.582 | -0.05 / 3.94 | 0.81 / 0.43 |

(`search_target_quality.py search` then `score`, with `--reference-data` for az1.) Paired regret
differences (`paired`) against pruned visits at the same budget (self / az1, x1e-3): cq -0.75 /
-0.42 (+-0.25 / 0.33) at 48 and -0.71 / -0.45 (+-0.17 / 0.15) at 200; pruned² -0.26 / -0.40 and
-0.47 / -0.50 (+-0.1); one-hot -0.33 / -0.60 and -0.17 / -0.65; cq c_scale 3 -2.11 / -0.77 and
-2.44 / -0.90 with top-1 vs az1 -0.057 / -0.072 (+-0.016); rpo -0.68 / +0.18 and -0.69 / +0.23. cq at
48 vs pruned at 200: +1.45 / +0.81 (+-0.3).

What this says:

1. **Little of the value's sibling ranking survives at these budgets.** The more a target trusts
   the 48/200-readout q (cq c_scale 3, rpo with the tree's c, any min-max q), the better it agrees
   with the same network's deeper search (regret down to ~0 at 200) and the worse with az1's
   (top-1 -0.04 to -0.23, spread above the prior's own 0.169): the gain is the shared value noise,
   amplified. Normalizing q to its min-max range is the worst case -- it scales 0.01 noise to the
   whole range.
2. **The mild completed-Q target (cq, c_scale 1) is a small, consistent improvement over pruned
   visits** on both references (regret -0.4 to -0.75e-3, recall of the reference's move 0.13 ->
   0.21 at 48 and 0.21 -> 0.28 at 200 vs az1, spread equal), with top-1 vs az1 within noise
   (-0.012 +- 0.010 at 200). It is not sharper than pruned visits by entropy; it moves a little
   more mass to the moves deeper search prefers. Sharpening the visits (pruned², one-hot) buys a
   similar regret drop by concentrating on the search's own move.
3. **48-readout targets carry real signal**: about half the prior's regret (vs az1 9.9 -> 5.1-5.6e-3)
   against 3.9-4.3e-3 at 200; fast rows are worth training on at a reduced weight if data, not
   compute, limits.
4. The pruned, sharpened and cq targets are all *sharper* than this policy (0.21-0.42 vs 0.60), so
   "targets echo a flat prior" can't be the whole loop at u100 -- see the next part.

**q-normalize (the searches' own moves):**

| search | top-1 self / az1 | regret self / az1 (x1e-3) | leaves the policy's argmax | of those, az1's move |
|---|---|---|---|---|
| 0.05 @48 (generator) | 0.833 / 0.812 | 4.59 / 4.95 | 0.060 | 0.42 |
| 0.15 @48 | 0.822 / 0.808 | 5.79 / 5.32 | 0.035 | 0.33 |
| qnorm 0.25 @48 | 0.802 / 0.758 | 3.90 / 4.81 | 0.128 | 0.25 |
| qnorm 0.625 @48 | 0.810 / 0.787 | 4.74 / 4.87 | 0.075 | 0.29 |
| 0.05 @200 (generator) | 0.880 / 0.817 | 2.55 / 3.67 | 0.098 | 0.41 |
| 0.15 @200 | 0.850 / 0.825 | 3.94 / 4.27 | 0.050 | 0.53 |
| qnorm 0.25 @200 | 0.785 / 0.702 | 2.84 / 4.25 | 0.257 | 0.20 |
| qnorm 0.625 @200 | 0.820 / 0.757 | 2.88 / 4.17 | 0.172 | 0.26 |
| 0.05 @800 (self reference) | 1 / 0.807 | 0 / 3.24 | 0.183 | 0.44 |
| qnorm 0.625 @800 | 0.812 / 0.690 | 2.26 / 4.04 | 0.300 | 0.23 |

Normalized Q makes the search leave the policy's move 2-3x as often, mostly onto moves neither
reference prefers (20-29% of those departures are az1's move vs 41-44% for the generator's
search); top-1 drops 2-12 points against both references and regret does not improve (paired vs
the generator's search, az1: +0.5 +- 0.5e-3 at 200, -0.1 +- 0.6 at 48). Rejected: `--q-normalize`
stays off.

### What broadens the policy

**Training pushes the policy past its targets, mostly in stock rounds.** The same 600 positions
searched @200 by checkpoints along az4 (policy = the tree's priors at T 0.5; `search_target_quality.py
steps`):

| checkpoint | policy H, all / stock / track / train | pruned-visit target H, all / stock / track / train |
|---|---|---|
| az4 u0 | 0.178 / 0.184 / 0.310 / 0.134 | 0.224 / 0.249 / 0.390 / 0.122 |
| az4 u50 | 0.375 / 0.503 / 0.426 / 0.198 | 0.285 / 0.327 / 0.452 / 0.171 |
| az4 u100 | 0.598 / 0.834 / 0.616 / 0.306 | 0.411 / 0.503 / 0.607 / 0.210 |
| az4 u150 | 0.749 / 1.009 / 0.776 / 0.452 | 0.485 / 0.582 / 0.669 / 0.310 |
| az4 u200 | 0.869 / 1.151 / 0.901 / 0.566 | 0.566 / 0.672 / 0.782 / 0.379 |
| az5a u30 (anchor) | 0.672 / 0.919 / 0.706 / 0.375 | 0.445 / 0.537 / 0.673 / 0.246 |
| az5b u30 (visits²) | 0.628 / 0.858 / 0.639 / 0.364 | 0.426 / 0.517 / 0.611 / 0.236 (its own target, visits², 0.222) |

At the start the targets were the flatter ones (0.224 vs 0.178: the echo + exploration). From u50
the policy is broader than its targets and the gap grows (stock: 1.15 vs 0.67 at u200), and az5b
broadens toward targets of entropy 0.22. Stock rounds (half the decisions) carry most of it; track
lays follow their targets; dividends and routes stay sharp.

**Suspect: KataGo's surprise weighting** (`--surprise-weight 0.5` in every run since az2; az1, which
stayed sharp, had none -- and an anchor). One CE step moves the policy's mass off its argmax by the
row-weighted mean of (target off-argmax mass - policy off-argmax mass). On the 600 positions (u100
policy, its own targets; `search_target_quality.py pressure`):

| target | push off the argmax, unweighted @48 / @200 | with surprise weights (a 0.5, cap 8) @48 / @200 |
|---|---|---|
| pruned visits (az4) | -0.029 / +0.004 | **+0.074 / +0.134** |
| pruned visits² (az5b) | -0.078 / -0.050 | **+0.011 / +0.078** |
| one-hot | -0.136 / -0.097 | -0.094 / +0.040 |
| cq | -0.001 / +0.010 | **+0.118 / +0.155** |
| cq c_scale 3 | +0.069 / +0.111 | +0.287 / +0.350 |

Unweighted, the pruned and cq targets are neutral and the sharpened ones sharpen; surprise weights
put the step on the rows where target and policy disagree most -- largely noise (only ~40% of the
targets' departures from the policy's argmax are az1's move) -- and turn every target into a
broadening one, in the order of the runs' observed broadening (az4 > az5b). The prediction for the
pilot below: cq with surprise weights broadens like az4; without them it stays flat.

### Pilot: cq targets with and without surprise weighting (2026-10-19, ~90 min each)

Both arms from az4 u100 with az4 / az5b's settings (`scripts/launch/2026-10/qtarget_pilot.sh`: `--shared
--value-stop-grad`, PUCT 200 / 48, p_full 0.25, reuse total, explore temperature 1 for 120
decisions, lambda 0.97 values, aux 0.1, lr 5e-6, no anchor, window 131,072 rows, 8,192 rows per
update, 8 steps of 2,048), `--target cq`, 12 workers each, run side by side (GPU shared with az5a,
az5b and evals: ~1,200 games/h, ~220k rows/h each): **A** `--surprise-weight 0.5` (as az4/az5),
**B** `--surprise-weight 0` (`model_checkpoints_search/qtarget_pilot_sw05_20261019`,
`qtarget_pilot_20261019`). History per update (full rows; updates after the start, az4 counted from
its u100):

| run | policy H +1 / +10 / +20 / +28 | target H +1 / +10 / +20 / +28 | KL(t‖π) +1 / +28 | top-1 +1 / +28 |
|---|---|---|---|---|
| az4 (pruned, surprise 0.5) | 0.576 / 0.598 / 0.632 / 0.649 | 0.355 / 0.365 / 0.390 / 0.395 | 0.286 / 0.271 | 0.904 / 0.892 |
| az5a (pruned, surprise 0.5, moving anchor) | 0.581 / 0.596 / 0.638 / 0.649 | 0.371 / 0.375 / 0.396 / 0.405 | 0.298 / 0.281 | 0.894 / 0.893 |
| az5b (pruned², surprise 0.5) | 0.583 / 0.558 / 0.586 / 0.596 | 0.192 / 0.178 / 0.187 / 0.190 | 0.362 / 0.315 | 0.885 / 0.900 |
| **A**: cq, surprise 0.5 | 0.573 / 0.625 / 0.648 / **0.680** | 0.315 / 0.339 / 0.355 / 0.376 | 0.362 / 0.341 | 0.855 / 0.857 |
| **B**: cq, surprise 0 | 0.576 / 0.516 / 0.523 / **0.530** (0.523 at +29) | 0.328 / 0.328 / 0.331 / 0.337 | 0.356 / 0.318 | 0.857 / 0.871 |

The prediction held: with surprise weights the cq arm broadened faster than az4 (as its weighted
push, +0.155, exceeds pruned visits' +0.134) and its targets followed (0.315 -> 0.376: the loop);
without them the policy *sharpened* in the first five updates (0.576 -> 0.526) and then stayed
flat for 24 updates, its targets flat at 0.33, KL(target || policy) falling and top-1 agreement
rising. Both arms had the same row staleness (mean row age 11.9 updates) and the same fit
(steps, lr), so neither stale rows nor plain forward-KL mass covering drives the rise at this
rate; surprise weighting does.

Per step type, on the 600 fixed positions (each checkpoint's own 200-readout search; policy H /
cq target H):

| checkpoint | all | stock (308) | track (101) | train (95) | dividend (80) |
|---|---|---|---|---|---|
| start (az4 u100) | 0.598 / 0.358 | 0.834 / 0.428 | 0.616 / 0.600 | 0.306 / 0.154 | 0.074 / 0.035 |
| A u10 / u20 / u28 | 0.655 / 0.686 / 0.706 | 0.924 / 0.963 / 0.984 | 0.641 / 0.667 / 0.693 | 0.338 / 0.364 / 0.389 | 0.084 / 0.092 / 0.104 |
| B u10 / u20 / u29 | 0.551 / 0.557 / 0.565 | 0.731 / 0.731 / 0.734 | 0.642 / 0.665 / 0.685 | 0.310 / 0.319 / 0.329 | 0.079 / 0.083 / 0.088 |
| B u29's own targets | 0.371 | 0.415 | 0.683 | 0.172 | 0.052 |

The broadening is in the stock round (half the decisions): surprise weighting adds ~0.15 there in
28 updates; without it the stock policy dropped 0.10 and held. What remains in B is a slow rise in
track lays (+0.07 in 29 updates), where the policy equals its targets and the targets themselves
broaden (0.600 -> 0.683) -- the echo loop proper, in the step type with the most near-equal options
and the value's weakest sibling ranking (the value-encoder network at the lineage's start: track
0.747 against az1's critic's 0.944; diagnosis above).

**But the policy-only strength still fell.** Near-greedy policy-only checks (T 0.1 both sides,
1,200 games vs gen4, `logs/orchestration/greedy_checks.sh`'s protocol): start (az4 u100) 0.434 +-
0.014, **B u29 0.382**, A u28 0.350 (+- 0.014; az4 u150 0.314, az5a / az5b u50 0.362 / 0.332). Not
flattening is not enough: B's policy moves its argmax toward the targets' (top-1 0.857 -> 0.878),
and per the offline table only ~40% of the targets' departures from the policy's move are an
independent strong search's move. With this lineage's value (sibling ranking 0.920 overall, 0.747
in track at its start) the targets' disagreements with the policy are largely noise, and distilling
them costs the raw policy strength even when the policy stays sharp. System strength (policy +
search) was not measured for the pilot.

### Recommendation

1. **Turn surprise weighting off** (`--surprise-weight 0`) in every search-learning run: it is the
   main driver of the flattening loop (A vs B: one flag apart, broadening vs flat; predicted
   offline by the weighted push).
2. **Target: `--target cq`** (c_scale 1, no rescale) or pruned visits -- at these budgets the
   completed-Q target's extra signal is small (regret -0.4 to -0.7e-3 against both references,
   more mass on deeper search's move) and its unweighted pressure on the policy is neutral; do not
   trust q harder (c_scale >= 3, `rpo` with the tree's c, min-max q): that amplifies the value's
   noise. `rpo` at c >= 0.5 is ~ the prior. `--q-normalize`: off.
3. **Keep az1's fixed anchor** (KL to the starting policy, target 0.02, never moved -- az1, the one
   run that stayed sharp *and* kept its policy-only strength, 0.524 near-greedy at u320, had it and
   no surprise weights; az5a's anchor moved every 50 updates and broadened anyway): without surprise
   weights the policy no longer needs it against flattening, but the targets' noisy departures
   still erode policy-only strength. Next arm, from az4 u100 (comparable with az5a / az5b) with
   az5b's settings and `--target cq --target-temperature 1 --surprise-weight 0 --kl-coef 0.02
   --kl-target 0.02 --anchor-every 0`, judged by system strength vs az4 u100 every 50 updates *and*
   the near-greedy check; as its pair the same without the anchor (`--kl-coef 0 --kl-target 0`),
   to see what the anchor costs in distilled search gains. (Starting instead from the shared
   network's start, `model_checkpoints_pg/shared_diag_venc1_20261018/learner/280.pth` -- az1 u280's
   policy, sharper and stronger alone -- is the alternative if the az4 lineage's flattened u100 is
   the wrong base.)
4. **Fast rows**: their targets carry real signal (48-readout cq halves the prior's regret against
   both references), but they would quadruple the rows of a target whose departures are mostly noise.
   Try `--fast-row-weight 0.3` only after (1)-(3) hold policy-only strength.
5. The ceiling is the value's sibling ranking (track lays in particular): a better-ranking value
   (more critic distillation on children, `distill_value_on_children.py`) is what would make any
   Q-based target pay.

## Confidence-filtered targets and reanalyzed sibling values (2026-10-19)

Two levers aimed at the problem above: where the search's target departs from the policy's top move,
only ~40% of departures are az1's move, and the value ranks track-lay siblings worst. Both are options,
off by default, built and measured offline; neither has been trained in the loop yet.

### What was built

| option / tool | what |
|---|---|
| Rust `root_target_stats()["child_q_var"]` | the sample variance of each root move's backed-up values for the mover (a new per-child sum of squares, `RustMCTSNode.child_w2`, real backups only). Selection never reads it: 288 of 288 seeded HashNet search digests (minigo Q, mean Q, mean Q + FPU, Gumbel; alone, on a kept subtree, lockstep) identical between the old and the new build; `cargo test` 216 passed |
| `search_targets.departure_confidence` | z = (q_t - q_p) / sqrt(var_t / N_t + var_p / N_p), t = the target's argmax, p = the policy's (the tree's priors), each variance shrunk toward the root's pooled variance with 2 pseudo-visits (KataGo's LCB move selection uses the same standard error); `departure_share` = N_t / (N_t + N_p); `confident_target` keeps or shrinks a departure |
| `--target-confidence <x>` | departures whose signal is below x shrink toward the policy (`--target-confidence-mode prior`: replaced by the tree's priors; `soft`: keep signal / x of the departure); `--target-confidence-signal z` (default) or `share`; `--target-confidence-prior-weight` 2 |
| `--target-distrust-steps <types>` | comma-separated step types (`search_learning.decision_step`: stock, track, token, route, dividend, train, other -- `diag_sibling_values`'s classes) whose departures always shrink to the policy. Rows that agree with the policy are never touched. History: `target_departure_share`, `target_departure_shrunk_share`, `target_departure_distrusted_share` |
| `search_target_quality.py confidence` | per signal / rule (`z=2`, `share=0.5`, `nontrack=1`, combinations): departures kept, precision / recall against a reference's move, the share the reference values above the policy's move, the filtered target's regret, AUCs |
| `search_target_quality.py pairwise` | each departure's two children (the target's move, the policy's), built as the judged search built them, searched as roots by an independent network (az1 u320 + critic, 400 readouts): a prior-free comparison of the two moves |
| `scripts/reanalyze_siblings.py` | `build`: positions of saved games, up to 16 children each (top 8 by prior + 8 random), every child searched as a root by the network (the generator's settings) -- targets: the root's mean value (`mean`) or its most visited move's (`best`), in the child's canonical frame; `check`: do those targets rank siblings closer to a reference than the raw values; `train`: fine-tune the value side only (value entity encoder + heads, trunk frozen, as `--value-stop-grad` trains it) on cached trunk features -- win CE to the targets, a ListNet ranking loss over each position's children (`--rank-weight`), outcomes, or a teacher's values (`--teacher`, the distillation baseline) -- evaluated on held-out sets against az1's critic and against az1's searched values |

Tests: `test_target_confidence.py` (12), `test_reanalyze_siblings.py` (5); `tests/agent/alphazero` 922 passed, 1 skipped; `cargo test` 216 passed.

### Confidence filter, offline

Two position sets, every target computed from the same trees: the study's 600 positions (searched again
by az4 u100 and by az6 u200 at 48 / 200 / 800 with the new build; az1 @800 reference reused), and 1,800
new positions of the same three collections (games 40-159, disjoint), searched by az6 u200 at 48 / 200 /
800 and by az1 u320 + critic 320 @800. On top of "is it az1's move" (precision), the pairwise
comparison: az1 searches the target's move and the policy's move each as a root.

**The departures are mostly improvements.** az6 u200, cq target, 1,800 positions:

| | @48 | @200 |
|---|---|---|
| departures (target argmax != policy argmax) | 177 (9.8%) | 251 (13.9%) |
| precision: the departure is az1@800's move | 0.367 | 0.371 |
| az1@800's root values the departure's move above the policy's | 0.695 | 0.729 |
| pairwise: az1 values the departure's child above the policy's | 0.706 | 0.765 |
| pairwise mean gain q(target move) - q(policy move), win share | +0.021 | +0.021 |

The ~40% is the strict measure (az1's single top move among several near-equal ones); by az1's own
comparison of the two moves 70-77% of departures improve on the policy's move. The exception is track
lays (per step type, @200; @48 alike):

| step | departures (share of the step's positions) | precision | pairwise better | pairwise gain (x1e-3) |
|---|---|---|---|---|
| stock | 151 (16.6%) | 0.384 | 0.848 | +22.9 +- 2.8 |
| track | 55 (15.9%) | 0.200 | 0.491 | +3.1 +- 2.0 |
| train | 32 (13.5%) | 0.594 | 0.844 | +47.3 +- 14.0 |
| dividend / other | 13 | 0.38 | 0.77 | +18 |

Track-lay departures are coin flips by az1's judgment -- where the value's sibling ranking is weakest.

**Rules** (az6 u200, cq, 1,800 positions; "gain kept": the share of the total pairwise gain the kept
departures carry; d_regret: the filtered target's regret against az1@800 minus the unfiltered one's, x1e-3,
< 0 better):

| budget | rule | departures kept | precision | recall | az1 root better | pairwise better | gain kept | d_regret |
|---|---|---|---|---|---|---|---|---|
| 48 | none | 1.000 | 0.367 | 0.147 | 0.695 | 0.706 | 1.000 | 0 |
| 48 | z >= 1 | 0.881 | 0.365 | 0.129 | 0.705 | 0.718 | 0.941 | +0.06 |
| 48 | z >= 2 | 0.605 | 0.364 | 0.088 | 0.701 | 0.738 | 0.733 | +0.30 |
| 48 | z >= 3 | 0.384 | 0.456 | 0.070 | 0.794 | 0.809 | 0.602 | +0.43 |
| 48 | share >= 0.5 | 0.384 | 0.515 | 0.079 | 0.809 | 0.809 | 0.587 | +0.20 |
| 48 | **no track** | 0.712 | 0.452 | 0.129 | 0.770 | 0.770 | 0.955 | **-0.15** |
| 200 | none | 1.000 | 0.371 | 0.211 | 0.729 | 0.765 | 1.000 | 0 |
| 200 | z >= 1 | 0.924 | 0.371 | 0.195 | 0.746 | 0.780 | 0.999 | -0.11 |
| 200 | z >= 2 | 0.801 | 0.368 | 0.168 | 0.741 | 0.796 | 0.918 | +0.04 |
| 200 | z >= 3 | 0.661 | 0.410 | 0.154 | 0.771 | 0.849 | 0.807 | +0.13 |
| 200 | share >= 0.5 | 0.602 | 0.450 | 0.154 | 0.775 | 0.808 | 0.764 | +0.34 |
| 200 | **no track** | 0.781 | 0.418 | 0.186 | 0.786 | 0.842 | 0.968 | **-0.18** |

(Combinations add little: z >= 1 with no track keeps 0.761 at 200 with precision 0.414.) Standard errors:
precision ~0.03-0.04, pairwise better ~0.03.

How well each signal separates the departures that are az1's move (AUC; 0.5 = no information):

| | z | q gap | share | same network @800 (self) |
|---|---|---|---|---|
| cq @48 / @200, az1's move | 0.53 / 0.56 | 0.40 / 0.50 | 0.69 / 0.68 | z 0.66 / 0.74 |
| cq @48 / @200, pairwise better | 0.57 / 0.64 | 0.49 / 0.51 | 0.62 / 0.63 | |
| pruned visits @48 / @200, az1's move | 0.67 / 0.58 | 0.64 / 0.58 | 0.51 / 0.60 | z 0.82 / 0.81 |

The z-score measures the search's sampling noise -- it predicts whether the same network's deeper
search keeps the departure (AUC 0.66-0.82) -- but barely whether an independent search agrees (0.53-0.58
for cq): the remaining disagreement is the value's systematic error, which no statistic of one tree can
see, and backed-up values within a subtree are correlated, so the standard error is optimistic. For
pruned-visit targets at 48 readouts, where departures include low-visit noise, z does work (precision
0.395 -> 0.525 at z >= 1 keeping 52%, regret -0.61e-3). On the study's 600 positions the picture is the
same (az4 u100 @200: 81 departures, precision 0.395, pairwise better 0.741; z >= 2 keeps 77%, precision
0.403, pairwise better 0.806, the dropped ones' mean gain -2.3e-3; on 1,800 positions the dropped ones
still gain +8.8e-3 on average).

**Verdict.** Raising precision clearly above 40% costs real signal: the rules that do it (share >= 0.5,
z >= 3) drop 34-62% of the departures and 19-41% of the pairwise gain, and make the target's regret
against az1 worse. The one rule that is better on every measure is to stop departing in track lays
(`--target-distrust-steps track`): precision 0.37 -> 0.42-0.45, pairwise better 0.71/0.77 -> 0.77/0.84, 96-97%
of the gain kept, regret better. Caveat on every az1 number: az1 is not wholly independent of az6 -- az6's
value encoder was distilled from az1's critic 280 (az6 u200 still ranks siblings 0.843 like it) -- so
agreement with az1 is partly shared lineage; the disagreements are the informative part. It touches ~3% of rows (track lays are ~19% of decisions, 15-16% of those
depart).

### Reanalyzed sibling values, offline

**Where the value stands.** On the held-out sibling set (`siblings_v2`, 1,080 positions, vs az1's critic
280): the value-encoder network at the lineage's start 0.920 overall (stock 0.966, track 0.747), az4 u100
0.869 (0.901, 0.744), **az6 u200 0.843 (0.871, 0.741)**: the loop's value training (outcomes / lambda
returns on played positions) has been eroding the ranking it started with; track never moved.

**The targets** (300 positions of az6 / az5 search games, 2,856 children; every child searched as a root
by az6 u200; Spearman across siblings against az1's critic 280 (raw values) and against az1 u320 +
critic searching every child at 256 readouts (`pre256_az1`)):

| values | vs critic 280: all / stock / track / dividend | vs az1's search @256: all / stock / track / dividend |
|---|---|---|
| az1's critic 280 | 1 | 0.703 / 0.797 / 0.490 / 0.150 |
| az6 u200, raw | 0.830 / 0.864 / 0.817 / 1.000 | 0.681 / 0.777 / 0.463 / 0.150 |
| az6 reanalyzed @64, root mean | 0.731 / 0.816 / 0.585 / 0.550 | 0.708 / 0.821 / 0.415 / 0.600 |
| az6 reanalyzed @64, best move's value | 0.702 / 0.797 / 0.564 / -0.250 | 0.689 / 0.805 / 0.410 / 0.450 |
| az6 reanalyzed @256, root mean | 0.697 / 0.787 / 0.552 / 0.150 | 0.721 / 0.830 / 0.427 / 1.000 |

The references disagree: against the critic's raw values the reanalyzed values rank worse than raw; against
az1's search they rank better (stock, dividend: there both searches reverse the raw values' order).
Searched values carry the mover's own improved play (a child where the parent's mover moves next gains,
one where an opponent moves next loses: in dividend decisions the mean gap between the two kinds goes from
-0.054 raw to +0.031 @256) and Monte Carlo noise larger than track siblings' spread (0.006).

**Fine-tunes** of az6 u200's value side (value encoder + heads; the trunk and the policy bit for bit
unchanged) on 2,400 positions / 22k children of az4-az6 search games (lr 3e-5, 400 steps of 256 children;
the az1-search test set's games kept out):

| training | vs critic 280: all / stock / track | vs az1 search: all / stock / track | win CE: siblings_v2 / held-out games |
|---|---|---|---|
| start (az6 u200) | 0.845 / 0.871 / 0.750 | 0.682 / 0.779 / 0.461 | 1.250 / 0.989 |
| outcomes only, 1.8k positions (800 steps, lr 5e-5) | 0.754 / 0.782 / 0.677 | 0.600 / 0.711 / 0.417 | 1.241 / 0.972 |
| reanalyzed (mean @64), win CE | 0.840 / 0.870 / 0.736 | 0.681 / 0.777 / 0.465 | 1.227 / 0.990 |
| reanalyzed (best @64), win CE | 0.844 / 0.877 / 0.733 | 0.678 / 0.773 / 0.467 | 1.227 / 0.985 |
| reanalyzed (mean @64), ranking loss | 0.840 / 0.867 / 0.739 | 0.687 / 0.784 / 0.472 | 1.252 / 0.999 |
| teacher: critic 280's values, win CE | 0.892 / 0.937 / 0.741 | 0.692 / 0.777 / 0.470 | 1.186 / 1.008 |
| teacher: critic 280's values, ranking loss | 0.890 / 0.936 / 0.745 | 0.683 / 0.775 / 0.468 | 1.195 / 1.003 |
| teacher, win CE, lr 1e-4 | 0.910 / 0.956 / 0.753 | 0.691 / 0.778 / 0.463 | 1.185 / 1.012 |
| teacher, win CE, lr 1e-4, + map read-out (probe) | 0.914 / 0.955 / 0.768 | 0.690 / 0.778 / 0.471 | 1.185 / 1.013 |
| reanalyzed (mean @64), ranking loss, lr 1e-4, + map read-out | 0.835 / 0.865 / 0.722 | 0.692 / 0.791 / 0.446 | 1.255 / 1.002 |

(Spearman standard errors ~0.01 on siblings_v2, ~0.02 on the 300-position az1-search set.)

1. **Self-reanalysis does not move the ranking** against either reference (within +-0.006); its targets
   are close to the raw values (held-out CE to them 1.065 -> 1.062), so there is little to learn.
2. **The value side can learn stock ranking** (the teacher: 0.871 -> 0.937 in 400 steps) **but not track
   ranking, from any target.** Track-lay siblings differ in the map: 38% of them encode identical game
   states (stock siblings: none), and with `--value-stop-grad` the value reads the map only through the
   frozen policy trunk. No value-side training can rank them better than that trunk allows. A value-only
   attention read-out over the shared map encoder's per-hex embeddings (`--map-readout`, a probe: each
   seat's value token attends over the hexes, zero-initialized into the trunk features the heads read)
   moved track only 0.750 -> 0.768 with the teacher (stock to 0.955, train 0.833 -> 0.890): the policy-trained
   map encoder does not carry it either -- az1's critic, with its own value-trained map encoder, ranks track
   0.94.
3. **Outcome training on few positions erodes the ranking** (the control: 0.845 -> 0.754), the same
   direction as the lineage's drift (0.920 -> 0.843).

**Cost.** @64 readouts a reanalyzed child costs 67 evaluations (10.5-10.8 ms of forward time in-process on
the shared GPU, batches of ~96 trees); a position (9.4 children on average, at most 16) ~630 evaluations,
0.27 s wall in one process. In az6's generation (~24.6k evaluations/s, ~170 full rows/s) reanalyzing every
full row's children would cost ~3.5x the generation's evaluations; 10% of them ~35%.

**Not wired into the loop** (the offline metric did not move). The design, if a variant earns it: the
generator, at a full search with probability `reanalyze_prob`, builds k children on a separate tree (one
expansion, so the played tree's progressive widening is untouched) and adds a `TreeSearch` per child at
`reanalyze_readouts` to its lockstep rounds without blocking the game; finished searches become
`SiblingRow`s (canonical encoding, the root's value vector, version) returned beside the games; the learner
keeps them in their own window and adds `reanalyze_weight` x (win CE or the ranking loss over each
position's children) to the value loss. The GPU share is `reanalyze_prob * k * 67 / 40` of generation at
k children @64.

### Recommendation

1. **Next arm: az6's flags + `--target-distrust-steps track`**, from az4 u100 like az6 (comparable update for
   update), judged as az6 (system strength vs its start every 50 updates, near-greedy policy-only checks).
   It drops the one class of departure the independent search calls a coin flip and keeps 96-97% of the
   departures' measured gain. Expect a small effect: it changes ~3% of rows.
2. Do not add `--target-confidence` (z) to cq targets: at these budgets it filters the search's sampling
   noise, not the value's error, and costs gain. (For pruned-visit targets at 48 readouts, z >= 1 does help.)
3. The departures that remain are ~80% improvements by az1's own comparison (+0.021 win share each) -- yet
   az6's policy-only strength falls. That points less at the targets' noise than at what the policy does
   with them (search-dependent moves played without search; a sharper policy target is not a better
   greedy player). Worth checking directly: play the policy's old vs new top move at the positions where
   training changed it, policy-only, as `search_override_rollouts.py` does.
4. For the value: (a) stop the erosion -- a lower `--critic-lr` or a fixed critic-distillation term on
   children (the teacher run shows the value side takes it: stock 0.871 -> 0.955); (b) track ranking needs a
   map encoder trained on value -- a small value-only map encoder, or the value gradient into the map
   encoder (not through `--value-stop-grad`'s detach) -- since a read-out over the policy's map encoder
   barely moved it; test offline with `reanalyze_siblings.py train --teacher critic280` first, then with
   reanalyzed targets at >= 256 readouts (closer to az1's search than @64: 0.721 vs 0.708). Self-reanalysis
   as built is not worth its cost (~0.27 s and ~630 evaluations per position) until (b) exists.

## Hindsight price learning: counterfactual price branching (2026-10-19)

**The question (the user's framing).** A human learns prices ten moves later: "had I paid myself $5 more
for that private, I could buy the share I want now". The search can't: the consequence is past its horizon,
the value is too blunt to see $5, and training only sees the price that was played. The engine is
deterministic, so rewind: find the moments a player or corporation was short of money, go back to the price
decisions that moved its money, replay them at nearby prices, and see whether a nearby price would have
played out better.

**Built** (`rl18xx/agent/alphazero/price_hindsight.py`, `scripts/price_hindsight.py`; tests
`test_price_hindsight.py`, 10):

| piece | what |
|---|---|
| Rust `BaseGame.counterfactual_transfer_cash(entity, amount)` | moves cash between the bank and a player / corporation on a clone (the rules never call it); a Rust unit test checks totals and errors |
| constraint events (`scan_game`, `unaffordable_events`) | **unaffordable**: at a stock- or operating-round decision the acting entity gets $100 more on a clone; if the policy (gen4) at that richer state puts its top move, with p >= 0.25, on a move that was not legal before, it wanted something it couldn't afford -- the shortfall is the least extra cash making it legal (bisection on the legal-move enumeration: only cash changed, so this is affordability, not other illegality). **emr_sale** (a president selling in an operating round to fund a forced train; shortfall = what the sales raised), **president_contribution** (a train bought for more than the corporation's cash), **bankrupt**, **bid_lost** (in agent-played auctions, a company won for <= $10 above this player's best bid) |
| back-tracing (`link_events`) | price-head decisions (auction bids, a corporation buying a private, a train bought from another corporation, with an open range) within 100 decisions before the event whose money went to or came from the constrained entity (its president too for the operating-round emergencies); the direction that would have helped (+1 paid to it, -1 paid by it, 0 an internal transfer between the two); a lost bid links to the loser's own bids on that company |
| branching (`BranchRunner`) | from the state before the decision: the played price, played +- $1 / $5 / $10 / $20 and the range's ends (clipped, de-duplicated); **open loop**: the game's own later actions replayed to the event where still legal (cash at the event, is the wanted move legal now, the value there); **closed loop**: 8 policy-only continuations per alternative to the game's end (every seat gen4 @0.5, prices from its head) with **common random numbers** (continuation r of every alternative draws from one seed, so branches part only where the price changed something); final win share, net-worth share and placement points of every seat, and the value (az7b u220) of the deciding player at 0 / 10 / 30 / 60 / 100 decisions after the branch. Par prices and dividends can be branched the same way (the other moves of their kind) |
| `scripts/price_hindsight.py` | `detect` (inference server, workers) -> `jobs` (stratified per group / price type / event kind, controls = unlinked price decisions, money jobs) -> `branch` (resumable) -> `report` -> `finetune` (the price head alone on cached trunk features, toward advantage-weighted cell targets `p_old(c) exp(A(c) / beta)`, or toward the played price as the control; held-out games' branch outcomes as the test) |

**Data.** 4,303 games: 3,220 search-eval games of the az1-az7 collections (az3 onward, 140 per collection;
search systems on both sides) and 1,083 of gen4's policy-gradient games (`rab_place`, 170 of them started at
the auction, the only agent-played bids). 48 PG games no longer replay (pre-10-16 rules). Detection took ~10
min (12 workers, gen4 serving the probes).

| events (per game) | count | shortfall |
|---|---|---|
| unaffordable | 326,259 (76.7) | median ~$50; 13k <= $5, 14k $6-10, 30k $11-20 |
| -- wanted move: share purchase (stock round) / tile lay (terrain) / train / private / par | 266,973 / 45,799 / 11,520 / 1,628 / 337 | |
| president contribution | 29,205 (6.9) | 70% > $50 |
| emergency share sales | 6,373 (1.5) | 88% > $100 |
| bankruptcy | 636 (0.15) | |
| lost bid (<= $10) | 374 (2.3 per auction-start game) | 95% <= $5 |

Price decisions: 93,396 cross-corporation trains (~22 a game: half at $1, 17% at the maximum, mostly the
same president moving a train and cash between his corporations), 18,586 privates sold to corporations
(99.5% at the maximum, twice face value), 4,577 auction bids (91% at the minimum bid); plus 28,621 pars and
426,826 dividends. With a 100-decision window 79,423 of the 116,559 price decisions link to at least one event
(241,493 links; the median distance to the nearest event is ~74 decisions).

**Branched**: 1,160 jobs -- 828 linked decisions (110 per decision type x primary event kind, covering 2,518
events: 1,578 unaffordable, 597 president contributions, 200 emergency sales, 115 lost bids, 28 bankruptcies),
212 unlinked controls, 60 pars, 60 dividends -- 58,869 continuations, 26.1M decisions in 80 min (12 workers,
shared GPU, ~5.4k decisions/s; ~22.5k decisions per job; 57 continuations hit the 1,000-decision cap).

**Results** (paired against the played price, continuation r vs r; `logs/price_hindsight/main/report.txt`):

| decision (linked) | jobs | continuations identical to the played price's (win / worth) | split-half gain: win / placement / net-worth share | significantly better / worse (worth) |
|---|---|---|---|---|
| auction bid | 220 | 0.79 / 0.04 | +0.000 +- 0.012 / +0.006 +- 0.012 / +0.0000 +- 0.0014 | 0.15 / 0.77 |
| private sold to a corporation | 278 | 0.89 / 0.29 | +0.000 +- 0.010 / +0.014 +- 0.009 / +0.0008 +- 0.0018 | 0.11 / 0.26 |
| train between corporations | 330 | 0.98 / 0.35 | +0.002 +- 0.003 / +0.004 +- 0.003 / +0.0029 +- 0.0009 | 0.22 / 0.39 |
| (controls, unlinked: bid / private / train) | 22 / 80 / 110 | | -0.023 / +0.003 / +0.009 (win) | |
| par (alternative par prices) | 60 | 0.69 / 0.03 | +0.042 +- 0.030 / +0.041 +- 0.030 / +0.006 +- 0.004 | 0.27 / 0.38 |
| dividend (payout <-> withhold) | 60 | 0.86 / 0.00 | +0.013 +- 0.018 / +0.015 +- 0.018 / +0.003 +- 0.003 | 0.05 / 0.35 |

(Split-half gain: the alternative the even continuations prefer, if any beats the played price, scored on the
odd ones -- an unbiased estimate of what switching to the nearby price that looks best gains. "Significantly":
some alternative beyond 2 SE, of ~5-6 tried.)

Mean advantage by price move (win share, over jobs; placement and worth mostly agree):

| move | bid | private | train |
|---|---|---|---|
| +-$1 | +0.003 +- 0.005 (+1) | -0.003 +- 0.004 (-1) | +0.001 +- 0.001 / +0.001 +- 0.002 |
| +-$5 | +0.002 +- 0.010 (+5) | -0.005 +- 0.005 | -0.000 / -0.001 |
| +-$10 | -0.012 +- 0.011 (+10) | -0.004 +- 0.006 | +0.000 / -0.001 |
| +-$20 | -0.037 +- 0.012 (+20) | -0.006 +- 0.007 | -0.004 / -0.009 +- 0.004 |
| range end | max: -0.188 +- 0.013 | min (face / 2): -0.042 +- 0.010 | max: -0.016 +- 0.005; $1: -0.004 +- 0.008 |

1. **A nearby price rarely plays out better, and never by much.** The prices played -- privates at twice
   face, bids at the minimum, trains at $1 or the maximum -- sit at or near a local optimum: moves of $1-10
   are within +-0.005 win share of them, $20 and the range's far end are clearly worse, and in every class
   "significantly worse" alternatives outnumber "significantly better" ones. How often a nearby price wins:
   where the even continuations pick a different price, the odd ones confirm it about as often as not
   (net-worth share, all jobs of the type: trains 172 of 440 decisions confirmed vs 121 reversed, the only
   clear edge; privates 120 vs 131 of 358; bids 101 vs 102 of 242; on the win share most picks are ties).
   The one positive class is trains between
   corporations, +0.003 net-worth share (3 SE, ~0.3% of a player's worth) and, moving the cash toward the
   constrained corporation rather than away, +0.010 +- 0.004 win share (118 jobs). Par prices are the one
   larger hint (+0.04 +- 0.03 win share on 60 jobs) -- a categorical choice, not a price-head slot.
2. **The hindsight hypothesis itself fails here.** At the 330 unaffordable events, the alternatives whose
   open-loop replay made the wanted move affordable at the event did *worse* than the played price: win
   -0.047 +- 0.018, placement -0.078 +- 0.018, worth -0.025 +- 0.005 (the other alternatives -0.022 / -0.028 /
   -0.007). Shortfalls are mostly large (median ~$50; emergencies > $100) and the cash that would cover them
   comes from a price far from the played one, which costs more than the wanted move gains. In only 81 of the
   330 did any tested price make the wanted move legal at the event (open-loop replay).
3. **The effect is small against the continuations' noise.** Per alternative (8 paired continuations) the
   noise SE is 0.109 win / 0.090 placement / 0.0135 worth; the implied true-effect SD is 0.067 / 0.089 / 0.023
   -- almost all of it the far-off prices above. Split-half reliability of an alternative's advantage: 0.17
   win, 0.32 placement, 0.58 worth (of its full 8-continuation mean: 0.29 / 0.49 / 0.73): per-decision targets on the win
   are mostly noise even with common random numbers, which kept 79-98% of win outcomes identical.
4. **The value is less blunt than feared.** az7b's value at the branch point (one forward per alternative)
   correlates 0.27 / 0.37 / 0.45 (win / placement / worth) with the alternatives' final advantages -- about 0.5
   with their noise-free part -- and 30 decisions later 0.39 / 0.56 / 0.69 (~0.75-0.8 noise-free). It sees the
   direction of a price effect early; it is biased slightly against alternatives (mean -0.01).

**Price-head fine-tune (gen4, head only, trunk frozen; 756 training / 284 held-out decisions by game; 300
steps, lr 1e-4, KL to the old head 0.1).** Held-out gain = the change in the expected branch outcome under
the head, over the tested cells, on held-out games' own continuations:

| target | train gain | held-out gain | KL(old \|\| new) held-out |
|---|---|---|---|
| hindsight, net-worth share (beta 0.005) | +0.00027 +- 0.00009 worth | -0.00002 +- 0.00010 worth | 0.037 |
| hindsight, win share (beta 0.05) | +0.0012 +- 0.0005 win | -0.0005 +- 0.0006 win | 0.027 |
| control: imitate the played price | +0.00019 +- 0.00007 worth | -0.00010 +- 0.00011 worth | 0.042 |

The fit is real in-sample and nil on held-out games -- what the targets carry beyond the played price is noise
at this scale. Policy-only, the hindsight-tuned head (worth, `model_checkpoints_ph/ph_worth_awr`) vs gen4, both
@0.5, 1200 games each: 2v2 0.506 +- 0.014; 1v3 0.227 +- 0.012 and, with another start seed, 0.268 +- 0.013
(together 0.248 +- 0.009, equal 0.25): no regression, no gain.

**Recommendation: don't wire hindsight branching into the loop.** It works (2.5k events, paired branches,
exact rewinds), but the prices the agents play are already where nearby prices can't beat them by more than
the noise, and the "wanted but unaffordable" moves are mostly not worth the price change that would fund
them. Cost, had it paid: a job (~6 alternatives x 8 continuations) is ~22.5k decisions, ~38 policy-gradient
games' worth; at a 10% share of PG's ~53M decisions/hour (~90k games of ~590 decisions) that is ~230 price
rows an hour against the ~170k price draws PG already trains on by REINFORCE each hour (~1,870 per update,
~92 updates/hour). Scoring by the value 30 decisions after the branch
(r ~0.8 with the noise-free effect) would cut it ~15x, by the value at the branch point (~0.5) ~4,000x --
which is what the search's price grandchildren already do at searched decisions. What's left worth a look:
par prices (the only sizable hint, a policy-head choice: more jobs), and the toward-the-constrained-corporation
direction for trains between one president's corporations (a 1% win-share effect).

## AlphaGo-style split: policy gradient for the policy, search games for the value (2026-10-19)

**Why.** Every search-learning run gains early and then plateaus (best system: az6 u200 @200, 0.692 vs gen4 +
critic). Two limits, both measured above: distilling the search's targets costs the policy its strength alone
(near-greedy checks fall in every arm, even without flattening), and the value's sibling ranking erodes under
its own outcome training (0.920 -> 0.843 along az4-az6; az7b, value off the trunk at critic lr 1e-5, 0.874 at
u50 -> 0.860 at u200). The policy-gradient line (gen1 -> gen4) is the one recipe that kept improving a policy
(league, placement reward, KL anchor). AlphaGo's split: the policy from RL self-play, the value from games,
search only at play time. Here: the policy path trained by PPO (as gen4), the value trained on *search* games
(as az7b: lambda-returns, every position once, trunk detached), search @200-800 at play time.

**One network or two.** One shared value-encoder network (the user's preference; ~1.04 forwards per search
evaluation against ~1.2-1.33 for a policy + critic composite): the policy path (entity encoder, map encoder,
fusion, trunk, policy and price heads) and the value side (value entity encoder + win / score heads, 4% of the
parameters) reading the trunk. The value never trains the trunk (`value_stop_grad`): az7a (value gradient
into the trunk) lost to az7b 0.45 +- 0.03 at u100 and ranks siblings worse (0.848 vs 0.860 at u200), and in
PPO a value gradient in the trunk would move the very policy the ratios are taken against.

**How PG interacts with the value.** The value side reads trunk features PPO keeps moving, so a value side
left alone goes stale as the policy improves -- the price of sharing. Two runs train the two parts of one
network and follow each other's latest checkpoint every update (`shared_split.Follower`):

| run | trains | follows |
|---|---|---|
| `policy-gradient --shared --value-follow <value run>` | the policy path (PPO, league, placement reward, KL anchor); baseline = the network's own value (the score head for placement), at update time | before each update: the value run's latest value side |
| `search-learning --shared --value-stop-grad --freeze-policy --policy-follow <PG run>` | the value side only (lambda 0.97 / bootstrap 0.5, aux heads, every row once, critic lr 1e-5); no policy, price or anchor loss | before each update: the PG run's latest policy path -- games are searched with PG's current policy as the prior |

Each PG checkpoint is then a whole network -- PG's policy with the newest value side -- and is what the search
loads at play time. The value lags PPO by one value update (~5 min at az7b's cadence).

**Measured: trunk drift costs the value side nothing at these rates; its own outcome training is what erodes
it** (2026-10-19, `scripts/check_shared_split.py --mix` + `diag_sibling_values.py score` on siblings_v2 vs
critic280; trunk change = relative L2 change of the trunk features on 3,000 of its states):

| network | trunk change vs the value side's own | Spearman all / stock / track / train |
|---|---|---|
| az7b u100 (matched) | -- | 0.869 / 0.904 / 0.741 / 0.865 |
| az7b u250 (matched: 150 more updates of value training) | -- | 0.858 / 0.896 / 0.737 / 0.825 |
| u200 trunk + u100 value side | 15.8% (100 updates) | 0.870 / 0.905 / 0.750 / 0.854 |
| u250 trunk + u100 value side | (150 updates) | 0.870 / 0.905 / 0.750 / 0.853 |
| u250 trunk + u200 value side | 7.4% per 50 | 0.860 |

A value side read through a trunk 150 search-learning updates newer ranks as well as when it was trained
(0.870 vs 0.869), while 150 updates of its own training on outcomes took the matched network from 0.869 to
0.858 (train purchases 0.865 -> 0.825): the erosion is the value targets, not the moving trunk. PPO moves
gen4's trunk less than search learning moves az7b's: 7.0% per 100 updates (rab_place 1700 -> 1800), 9.1%
over 200 (the moving anchor bounds it), against az7b's 15.8% per 100. So a value side can follow PG with a
lag of many updates -- or not train at all -- without losing its ranking; what needs care is the value run's
own training (lower `--critic-lr`, or a teacher term as `reanalyze_siblings.py train --teacher` showed keeps
stock ranking). If drift ever mattered: more value-run workers, a lower PG lr, or a value-only map encoder
(which track-lay ranking wants anyway, "Reanalyzed sibling values" above).

**Which policy to start from.** gen4: the strongest policy alone and the line PG has been improving; the az
policies are broader (entropy ~0.66 vs gen4's ~0.15 at T 0.5) and weaker alone. AlphaGo found its broader SL
policy a better search prior than its sharp RL policy, so the zero-training checks below compare the two
priors under the same value first; a second arm from az7b's network is the fallback if gen4's prior searches
worse.

**Built (2026-10-19; all options, off by default -- runs without them are unchanged).**

| piece | what |
|---|---|
| `shared_split.py` | `is_value_key` / `value_parameters` / `policy_parameters` (value side = `value_econ_transformer`, `win_loss_head`, `score_head`, aux heads), `load_part(model, state, "value" \| "policy")` (in place, the other part untouched), `latest_checkpoint`, `Follower` (loads one part of another run's newest checkpoint when newer; a checkpoint deleted mid-read is retried) |
| `policy_gradient.py` | `--shared` (no critic: `subtract_baselines` on the learner's own value heads; optimizer groups: policy path at `--lr`, value side at `--critic-lr`, each clipped on its own); `--value-follow <run>`; `--value-pg-weight w` (also train the value side on PG rows, trunk detached; default 0). History: `value_follow_update` / `value_follow_loaded` |
| `search_learning.py` | `--freeze-policy` (needs `--shared`: `_update_value_only` -- value-only forwards on the rows still flagged for the value, the optimizer holding the value side alone; the policy statistics are NaN); `--policy-follow <run>`. History: `policy_follow_update` / `policy_follow_loaded` |
| `scripts/build_value_encoder_network.py` | a critic that is itself a shared value-encoder network (az7b) now gives its *value* encoder |
| `scripts/check_shared_split.py` | for each checkpoint of either run, which checkpoint of the other run each part equals (bit for bit), plus the follow columns; `--mix POLICY VALUE OUT` builds the drift probe |

Tests: `test_shared_split.py` (9: the partition, part loads bit for bit, a plain network refused, the
follower, PG shared updates leave the value side untouched and `--value-pg-weight` trains only it, value-only
updates leave the policy path untouched and train each row once, option validation).

**Smoke test** (`logs/ag_smoke.sh`, 3 + 3 workers, ~15 min, start = gen4's policy path + az7b u220's value
side copied over unfitted): the value run made 14 updates (critic CE 0.70-1.18 on 4-10 fresh mid-game starts
each), PG 3 (kl_anchor 0.0003 -> 0.0013, entropy 0.11, 0.47 +- 0.03 vs gen4 over 320 games -- gen4's own
policy, so ~0.5 is expected; the critic-row Brier 0.078-0.084 with the unfitted value side), each following
the other: PG 1 / 2 / 3 carry value-run 6 / 7 / 14's value side, value-run 6-7 PG 1's policy path and 8-14 PG
2's, all bit for bit (`check_shared_split.py`: OK).

### First experiment (ag1)

**Step 0 -- the start network ag0: gen4's policy path + a value side fit to gen4's trunk** (~1 h of GPU):

    G4=model_checkpoints_pg/rab_place_20261014/learner/1800.pth
    A7=model_checkpoints_search/az7b_20261019/learner/220.pth   # or its latest snapshot
    uv run python scripts/build_value_encoder_network.py --policy $G4 --critic $A7 \
        --out model_checkpoints_ag/ag0_build/learner/0.pth
    # positions + children of recent search games (or reuse the train_children sets of the 10-18 distillation)
    uv run python scripts/diag_sibling_values.py build --no-played --collection logs/eval/az7a_u100_vs_az7b_u100 \
        --collection logs/eval/az7b_u100_vs_az6_u100 --collection logs/eval/az6_u200_vs_az5b_u200 \
        --games 200 --per-game 12 --out logs/ag/train_children
    uv run python scripts/distill_value_on_children.py --data logs/ag/train_children --test logs/shared_diag/siblings_v2 \
        --init model_checkpoints_ag/ag0_build/learner/0.pth --teacher az7b=$A7 --train-only value \
        --epochs 2 --out model_checkpoints_ag/ag0/learner/0.pth
    uv run python scripts/diag_sibling_values.py score --data logs/shared_diag/siblings_v2 \
        --source ag0=model_checkpoints_ag/ag0/learner/0.pth --source az7b=$A7 --reference critic280

Gate: ag0 ranks siblings >= ~0.85 (az7b u200: 0.860; the 10-18 fit of the same kind reached 0.920 from az1's
critic) with its policy bit for bit gen4's (`--train-only value`). Measured before any fit (2026-10-19): the
build alone -- gen4's trunk with az7b u220's value side copied over -- already ranks 0.844 (az7b u220 itself
0.863, az6 u200 0.843; stock 0.881 vs 0.900, track 0.694 vs 0.745, train purchases 0.851 vs 0.821), and its win
probabilities are within KL 0.001 of az7b's on siblings_v2's states (the value entity encoder carries most of
the value; cf. the drift result): the distillation is a short polish, not a rebuild.

**Step 1 -- zero-training checks** (240 games each, `eval_head_to_head.py` as the evals_v9 queue:
`@200/0.05:mean,dedupe,par=8,ptemp=0.5`, `--softpick 0 --start-positions human_games/start_positions_1830_4p_v2.jsonl
--dedupe-maps --server-replicas 2`): ag0 vs gen4 + critic (az6 u200 scored 0.692), ag0 vs az7b (the same value
behind gen4's prior vs az7b's), ag0 vs az6 u200. These are what the run has to beat; ag0 vs az7b decides
whether the second arm (from az7b's network) runs too.

**Step 2 -- the two runs, side by side** (GPU shared: e.g. 24 PG workers, 16 value workers; start the value
run first; both with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2`):

    # the value learner: az7b's generation and value settings, no policy training
    uv run python main.py search-learning --policy model_checkpoints_ag/ag0/learner/0.pth --shared --value-stop-grad \
      --freeze-policy --policy-follow model_checkpoints_ag/ag1_pg --out-dir model_checkpoints_ag --run-name ag1_val \
      --value-head win --target cq --c-puct 0.05 --n-full 200 --n-fast 48 --p-full 0.25 \
      --reuse-tree total --dedupe-maps --server-replicas 2 --workers 16 --games-per-worker 32 \
      --start-positions human_games/start_positions_1830_4p_v2.jsonl --random-start-fraction 0.2 \
      --midgame-positions human_games/midgame_positions_1830_4p_v2.jsonl --midgame-fraction 0.3 --auction-fraction 0.15 \
      --full-move-temperature 1 --explore-decisions 120 \
      --critic-positions-per-game 0 --value-lambda 0.97 --value-bootstrap-weight 0.5 --aux-loss-weight 0.1 \
      --critic-lr 1e-5 --window-rows 131072 --rows-per-update 8192 --min-window-rows 16384 --steps-per-update 8 \
      --minibatch 2048 --microbatch 1024 --max-updates 2000 --snapshot-every 10
    # the policy learner: gen4's run settings, the baseline the network's own score head
    uv run python main.py policy-gradient --policy model_checkpoints_ag/ag0/learner/0.pth --shared \
      --value-follow model_checkpoints_ag/ag1_val --out-dir model_checkpoints_ag --run-name ag1_pg \
      --population model_checkpoints_pg/populations/league_v7.json --reward placement \
      --learner-temperature 0.5 --opponent-temperature 0.5 --lr 2e-5 --kl-coef 0.02 --kl-target 0.08 --anchor-every 50 \
      --midgame-positions human_games/midgame_positions_1830_4p.jsonl --midgame-fraction 0.3 --auction-fraction 0.15 \
      --workers 24 --games-per-task 96 --rows-per-update 65536 --minibatch 2048 --microbatch 1024 \
      --save-game-every 20 --max-updates 600 --snapshot-every 10

**Step 3 -- judging, every 50 PG updates** (a PG checkpoint is the whole system):
- system: `ag1_pg/learner/<u>.pth@200/...` vs az6 u200 and vs gen4 + critic (240 games);
- the policy alone: `eval_policy_only.py` vs gen4 at T 0.5, 2v2 and 1v3 (1200 games, the PG line's measure),
  and the near-greedy check (T 0.1, as evals_v9);
- the value: sibling ranking on `siblings_v2` vs critic280 (the PG checkpoint: the followed value side on the
  current trunk), the value run's `critic_outcome_ce` / `search_value_ce`;
- frozen vs co-trained value, on the same policy: `check_shared_split.py --mix ag1_pg/learner/<u>.pth
  model_checkpoints_ag/ag0/learner/0.pth <out>` (update u's policy path, ag0's value side, never retrained)
  judged @200 like the checkpoint itself -- whether the value run earns its GPU share (by the drift result
  above, a frozen value side stays valid on PG's trunk; the question is whether training on the new policy's
  search games improves it or only erodes it); and the drift probe on the run's own checkpoints
  (`--mix <u+50> <u>`).

Success: the system beats az6 u200 (> 0.55) within ~300 PG updates while the policy alone keeps gaining on
gen4 (as gen1 -> gen4 did) and the ranking stays >= ag0's. Watch for: the co-trained value's ranking falling
below the frozen one's (its outcome training eroding it: lower `--critic-lr`, or stop the value run and keep
ag0's value side); the system flat while the policy gains (the sharp prior: `ptemp=1` at play time, or the
az7b-start arm). If the value run is not needed, the cheapest form of the split is PG alone with `--shared`
and no `--value-follow`.

**Control arm (optional, two networks): ag1t** -- the same PG run from gen4 with its own critic as the
baseline (gen4's run continued as a new run), judged as `<PG checkpoint>+<az7b checkpoint>` composites. It
asks whether co-training the value side on the moving trunk beats a fixed, separately trained value network
at equal PG updates, for the cost of a second PG run.
