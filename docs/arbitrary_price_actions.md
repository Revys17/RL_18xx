# Arbitrary price actions: auction bids and cross-corp train purchases

Feasibility study, 2026-10-01. Reproduce with `scripts/price_census.py` → `scripts/price_head_compare.py`.

## Summary

- **The architecture already supports arbitrary prices structurally.** A Bid / cross-corp BuyTrain / BuyCompany is one categorical policy slot, and its concrete price is chosen in a second tree level: progressive-widening (PW) "price grandchildren" sampled from `ContinuousPriceHead`. This exists in both the Python (`mcts.py`) and Rust (`mcts.rs`) MCTS, with parity tests, price targets in self-play and pretraining, and a price NLL loss. In the legacy 14-bucket cross-corp price grid in the 26,537-wide policy vector, bucket 0 is reused as the canonical slot index and the other 13 buckets are dead: nothing marks them legal.
- **Keeping the price out of the action list is the right call; the Gaussian is the problem.** The pre-May fixed grid could represent only 61% of human cross-corp train prices, 91% of bids, and 95% of company purchases without rewriting the action, and for jump bids also the encoded state. A continuous head uses 100% verbatim. But the shipped Gaussian head plus sampler finds the human's bid only **6%** of the time after 4 expansions, and the human's cross-corp price **58%** of the time.
- **An exact probability mass function (pmf) over every legal integer price gets both properties.** It is built from point masses (atoms) plus range-relative histogram bins, with mass spread uniformly inside each bin. It still trains on raw human prices with no snapping, and still lets search play any legal price. Searched with Gumbel-top-k sampling, it finds the human's price **97% (bids) / 70% (cross-corp trains)** of the time at 4 expansions and **99% / 96%** at 14: about the same as a fixed menu.
- **The bid failure is a latent bug, not just a modelling limitation.** It hasn't shown up because no checkpoint has ever had a trained price head: the PyG collate bug that kept it untrained was only fixed on 2026-09-30, and the only checkpoint is from Aug 2. **The first self-play run on a properly pretrained checkpoint should be expected to pick its auction bids from a handful drawn uniformly between the min bid and the player's cash.** This follows from the loss/sampler pair, not from the proxy features used here; see problem 1 below.
- **Recommendation:** keep the current continuous design, its two-level tree and its sampled PW. Replace the Gaussian with the exact-pmf head and replace the sampler with exact Gumbel-top-k draws from it. Fix the grandchild priors and add an exploration floor. Changes are confined to the price head, the loss, and the price-sampling code of both MCTS implementations. The policy layout, enumeration, LMDB schema, and pretraining data are unchanged.

## What exists today

| Piece | Where |
|---|---|
| Legal `[lo, hi]` per price-bearing slot | `factored.rs::factored_bid` (`min_bid..max_bid`), `factored_buy_train` (`spend_minmax`: 1..corp cash), `factored_buy_company` |
| Price collapsed out of the policy index | `ActionMapper.index_for_factored` / `canonical_index_for_action` (one index per company; per (seller corp, train type)) |
| `ContinuousPriceHead`: 60 slots × (μ, log σ) | `model_transformer.py:952` (μ = $150 + $100·raw, σ = exp(raw + 3)) |
| PW price grandchildren, k = ⌈pw_c·N^α⌉ | `mcts.py::_select_or_expand_price_child`, `mcts.rs::_select_or_expand_price_child` |
| Sampler | `mcts.py::sample_price_for_pw`, `mcts.rs::sample_price_for_pw` |
| Targets `(slot, price, weight, lo, hi)` | self-play: visits over grandchildren; pretraining: human price (`_factored_price_target`) |
| Loss | truncated-Normal NLL, `train.py::_compute_price_nll_loss` |
| Committed price at the root | most-visited grandchild |

## What human prices look like

From 2,607 cleaned games (112k priced actions), each replayed through the Rust engine to get the exact legal range at every decision:

**Bids** (34,296 with a free price; median legal range is $380 wide):

- 92.5% are exactly the min bid.
- 99.1% are on the min + $5·j ladder.
- None are more than $95 over the min.
- 0.01% are all-in.

**Cross-corp BuyTrain** (13,809; 99% have a free price):

- 100% are between two corps with the same president. This is a rule, not a habit: `ALLOW_TRAIN_BUY_FROM_OTHER_PLAYERS = False` in 1830. The price is just how a player splits cash between two of their own companies.
- Mode shares: **$1 (min) 31%**, **max 11%**, **max − 1 12%**, interior 46%.
- The interior is roughly uniform over the range, and 46% of interior prices are round $10 amounts. Semantic anchors barely appear: "seller can now exactly afford the next depot train" is about 3%, and face value 0.4%.
- Normalized-position histogram (10 bins): `[4942, 657, 656, 751, 640, 447, 405, 469, 607, 4235]`.

**BuyCompany** (9,522): 94% at max.

So bids are a short discrete ladder anchored at a state-dependent minimum. Train prices are two point masses plus a diffuse interior. A single Gaussian in absolute dollars cannot represent either well.

## Why the current head and sampler fail

`scripts/price_head_compare.py` fits the shipped head (same parametrization and loss as training) and an exact-pmf head on the same features. Train and held-out sets are split by game. It then measures coverage: after k expansions under one slot, does the candidate set contain the price the human played? It also reports the exact log-likelihood of the human's integer price.

| Bid coverage | k=1 | k=4 | k=14 |
|---|---|---|---|
| Gaussian, untrained (every checkpoint today) | 0.25 | 0.43 | 0.58 |
| Gaussian, trained, shipped sampler | **0.01** | **0.06** | **0.18** |
| Gaussian, trained, correct truncated-Normal sampler | 0.81 | 0.93 | 0.94 |
| Exact pmf, Gumbel-top-k samples (any legal price) | **0.87** | **0.97** | **0.99** |
| *(fixed menu, top-k by prior; reference)* | 0.93 | 0.97 | 0.99 |

| Cross-corp BuyTrain coverage | k=1 | k=4 | k=14 |
|---|---|---|---|
| Gaussian, untrained | 0.07 | 0.13 | 0.20 |
| Gaussian, trained, shipped sampler | 0.30 | 0.58 | 0.65 |
| Gaussian, trained, correct truncated-Normal sampler | 0.05 | 0.18 | 0.40 |
| Gaussian, trained, shipped sampler + {min, max} anchors | 0.62 | 0.63 | 0.65 |
| Exact pmf, i.i.d. samples | 0.32 | 0.65 | 0.86 |
| Exact pmf, Gumbel-top-k samples (any legal price) | **0.32** | **0.70** | **0.96** |
| *(fixed menu, top-k by prior; reference)* | 0.49 | 0.76 | 0.96 |

Exact log-likelihood of the human's integer price, mean bits per decision (lower is better):

| | Uniform over legal range | Trained Gaussian | Exact pmf |
|---|---|---|---|
| Bid | 8.54 | 1.95 | **0.61** |
| Cross-corp BuyTrain | 8.39 | 8.39 (no better than uniform) | **5.17** |

Four distinct problems:

1. **Train/sample mismatch (a bug).** The loss is a truncated-Normal NLL. For data piled at `lo`, its maximum-likelihood fit puts μ *below* `lo` with small σ: an edge-hugging tail. The trained head does this on 100% of held-out bids. The sampler is not a truncated-Normal sampler, though: it rejects draws outside `[lo − σ, hi + σ]` eight times and then falls back to uniform over the whole range. 99.9% of bids hit that fallback. A correct inverse-CDF sampler on the same parameters reaches 93%.
2. **One mode, absolute dollars.** Train prices have three modes; the trained head gives up with σ ≈ $4,900. The shipped sampler's clamping happens to turn that into "$1 or max", which is why the shipped row looks better than the correct sampler. Interior prices are only found 18% of the time at k=4, and more samples don't help: coverage plateaus at 65%. The mean is in absolute dollars, so the head must reconstruct `min_bid` from the encoded state to the dollar. A range-relative output gets that for free.
3. **No exploration floor.** Every candidate price comes from the head. If the head is wrong (untrained, collapsed, or drifted), search cannot discover the right price, and the visit-weighted price targets then reinforce the error.
4. **The head is ignored once prices exist.** At the PW cap, each grandchild gets `slot_prior / k`. The head's density only proposes prices; it never ranks them. Root Dirichlet noise doesn't reach the price level either.

This matters more at low search budgets. `loop_config.json` currently runs `num_readouts: 8`, where a slot sees one or two price expansions. At k=1 the head's most likely cell beats a random draw (0.49 vs 0.32 for trains), so the first widening of a slot should take the most likely cell and later widenings should follow the Gumbel order.

## Implemented design: exact-pmf price head on the existing continuous design

**Why this is not the old design.** Three things are separate, and only the first changes:

1. The network's price output.
2. The set of prices MCTS searches.
3. The training target.

The pre-May design made all three one fixed grid, so off-grid human prices had to be snapped (and jump bids also needed a falsified bid ladder in the encoded state). The current design separates them correctly; it just used a Gaussian for (1). The exact-pmf head assigns a probability to **every** legal integer price:

- `P(price) = P(cell) / |cell|`, where the cell is an atom or a histogram bin.
- Any human price is a valid target as-is.
- Search samples a cell and then a uniform integer inside it, so it can play any legal price.

The bins are the density's resolution, not an action list.

**Cells** (range-relative; the function `cell(type, price, lo, hi)` must match between Python and Rust):

| Type | Cells |
|---|---|
| Bid | ladder atoms `lo + step·j`, j = 0..19 (step = `bid_price_step()`, $5 in 1830), plus one residual cell holding every other legal price |
| Cross-corp BuyTrain | atoms `lo`, `hi`, `hi−1`, plus 20 interior bins over `(lo, hi−1)` |
| BuyCompany | atoms `lo`, `hi`, plus 8 interior bins |

Cells with no integers in a narrow range are masked and the rest renormalized.

**If $1 resolution matters** inside a bin (e.g. "seller can exactly afford the next depot train"), factor the price as `P(bin) · P(offset | bin)` with a second small categorical. This is the same autoregressive pattern `HierarchicalPolicyHead` uses for LayTile. It is exact at every dollar and is a head-only change; start without it.

**Head.** `ContinuousPriceHead` → `PricePmfHead` (`model_transformer.py`): trunk → logits `(num_slots, NUM_CELLS)`, NUM_CELLS = 24 (cells a type doesn't use are always empty). A shared MLP with a slot embedding generalizes across companies and corps better than 60 independent output blocks. It replaces `price_mean` / `price_log_std` in `last_price_components`, `PriceComponents` (Rust), the inference server reply, and `_coerce_price_components_for_rust`.

**Loss.** Exact NLL, which is cross-entropy over cells minus a constant `log |cell|`. Targets keep the schema `(slot, price, weight, lo, hi)`, and `price → cell` is computed at loss time. Pretraining uses raw human prices, self-play uses visited prices. Existing LMDB rows stay valid.

**MCTS** (keep the two-level tree, PW schedule, and `price_children` bookkeeping keyed by concrete price):

- **Proposals:** draw Gumbel noise per price slot once at node creation, giving a without-replacement cell order. The j-th widening takes the j-th cell and a uniform price inside it (deterministic given the seed, which keeps the PW parity tests tractable). This replaces `sample_price_for_pw` / `_snap_price` and removes the train/sample mismatch by construction.
- **Exploration floor:** proposals use `(1 − ε)·head + ε·uniform` over non-empty cells (`SelfPlayConfig.price_explore_eps`, default 0.05), so every cell is eventually reached even where the head is confidently wrong. With the Gumbel order this is the price level's only noise; root Dirichlet noise stays categorical.
- **Grandchild priors:** use the head's cell probability, renormalized over the expanded grandchildren, as the price level's own PUCT prior (it is no longer scaled by the slot prior; the old `slot_prior / k` made price-level exploration vanish under low-prior slots).
- **No grandchild at commit time** (the slot was never descended): commit the slot's first proposal, a price in the head's most likely cell, in both engines. Rust used to commit the range minimum, which is the wrong default for BuyCompany (94% at max).
- The committed price is still the most-visited grandchild, and multi-agent dict broadcasting is unchanged.

**Unchanged.** The 26,537 flat policy layout and `action_index.rs`, the factored enumeration and `price_range` plumbing, `decode.rs` (`map_index_to_action_with_price`), `price_head_slot_for_index`, the LMDB format, and pretraining conversion (no rewriting comes back).

**Checkpoints.** The new head has new parameters. Load older checkpoints with the price head skipped; nothing is lost, because no checkpoint's price head was ever trained.

### Status (2026-10-01)

Implemented on `claude/arbitrary-bid-amounts-638fdc`:

- **Cells:** `price_pmf.py` and `engine-rs/src/price_pmf.rs`, parity-tested over every price of 311 ranges.
- **Head and loss:** `PricePmfHead` and the exact-pmf NLL.
- **MCTS proposals:** in both engines.
- **Plumbing:** self-play, inference server, and the Rust FFI.
- **Old checkpoints:** load with a warning; the Gaussian head's tensors are dropped.
- **Refit:** `main.py refit-price-head` fits a fresh head on a checkpoint's frozen trunk from `human_games/lmdb_v3` and saves a new checkpoint without moving `current_best` (`--promote` moves it).

### Results on the training branch's current_best (2026-10-01)

**Refit:** fitted a fresh head on the frozen trunk of `20260930_173105` checkpoint 9 with `main.py refit-price-head`. It used 53,413 training and 3,249 validation rows with price targets from `lmdb_v3`, and converged in about 1,000 of 3,000 steps (about 3 minutes on the GPU). Held-out loss went from 4.72 bits/price (fresh, uniform head) to **1.67**, with the head's top cell matching the human's in 80% of cases.

On held-out human decisions:

| Type | Bits/price | Head's top cell = human's | Human's cell in first 4 PW proposals |
|---|---|---|---|
| Bid | 0.61 | 0.93 | 0.95 |
| BuyCompany | 0.55 | 0.95 | 0.98 |
| Cross-corp BuyTrain | 5.36 | 0.34 | 0.51 |

The cross-corp train interior is the genuinely uncertain part. A joint pretraining run, where the trunk also gets price gradients from this head, should help it.

**Opening self-play:** Rust MCTS at 64 readouts with the refit head, three 4-player games, 40 moves each. Each chosen bid slot searched 2–6 price grandchildren, and all 98 committed bids were the minimum bid, the human $5 ladder. Before the fix, self-play opened with bids up to the bidder's entire cash.

### Work plan (as planned)

1. `cell()` in Python (`action_mapper.py`) and Rust (new `price_cells.rs`, step from `title::bid_price_step`). Add a parity test sweeping every `(type, price, lo, hi)` in the census.
2. `PmfPriceHead` plus the exact-NLL loss (`model_transformer.py`, `train.py`), and the `last_price_components` shape through `self_play._slice_price_components`, `inference_server`, and `rust_mcts_player._coerce_price_components_for_rust`.
3. Proposals and priors in `mcts.py` (`_sample_price_for_slot`, `_select_or_expand_price_child`, `inject_noise`) and `mcts.rs` (same functions plus `PriceComponents`). Update `test_rust_mcts_parity_pw.py`, the price-target tests in `self_play_test.py`, and `test_rust_mcts_player_e2e.py`.
4. Pretrain fresh on `lmdb_v3`. Report price-head exact NLL (bits) and coverage@k per type alongside the existing metrics.
5. Gate check: the self-play price distribution should look like the census (bids at or near min, trains at $1 / max / interior), not uniform.

Rough size: under 1k lines across both engines, smaller than the menu variant because the PW tree stays. Most of the risk is in step 3's parity.

## Alternatives considered

- **Keep the Gaussian; fix the sampler and add anchors** (inverse-CDF truncated-Normal sampling, always materialize `lo` / `hi`). About a day of work and a reasonable stopgap if self-play must run before the redesign. It still finds interior train prices only ~18% of the time (k=4), its likelihood for train prices is no better than uniform, and it keeps the absolute-dollar parametrization.
- **Fixed price menu as the search's action set** (range-relative cells as concrete actions, PUCT over them). Similar coverage, but it re-merges the head and the action set, so off-menu human prices go back to being snapped. Rejected for the same reason the pre-May grid was abandoned.
- **Mixture of discretized logistics** (PixelCNN++-style) on the normalized range, plus atoms. Also an exact pmf with smoother within-bin shape. Equivalent in principle; more fiddly than bins, with no clear win on this data.
- **Fold prices into the flat policy** (joint "Bid CA @ +$10" actions). Same snapping problem, plus a `POLICY_SIZE` change.
- **Gumbel MuZero at the price level** (sequential halving over the Gumbel-top-k proposals). It has the best policy-improvement guarantees at tiny budgets and builds directly on the proposal scheme above; worth adding if `num_readouts` stays at 8.

## Open questions

- **Exact interior amounts.** Do they matter beyond 5% of the range? The data shows almost no semantic anchors. An engine-computed "seller can afford the next depot train" cell is cheap to add if self-play shows it matters.
- **1867** (`ALLOW_TRAIN_BUY_FROM_OTHER_PLAYERS = true`, and SR bid-founding with a $5 step): the cell layout is title-parametrized, so this extends without layout changes. Cross-player train prices there are adversarial rather than a cash split, so the interior may matter more.
