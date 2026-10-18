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
