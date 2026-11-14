# Title-generic policy / value model (RL bridge milestone B2)

*2026-11-13. Code: `rl18xx/agent/alphazero/model_generic.py`; tests:
`tests/agent/alphazero/test_model_generic.py`. Inputs: B1's typed tokens (`docs/rl_encoder.md`) and A1's
action descriptors (`docs/rl_descriptors.md`). Design: `docs/rl_bridge_design.md` §2.6, §2.7, §3.3, §4 and
the user's decisions of 2026-11-13 at its top (policy-gradient path first, `MAX_SEATS` 6).*

One architecture for every registered title. Per position it reads the encoder's token arrays and the
descriptor rows of the position's **legal** actions, and returns

* **policy** log-probabilities over the legal actions only (a flat list of legal rows for the whole
  batch, normalized per position: no padding to any layout's size);
* a **price** distribution for each price-bearing legal action over `price_pmf`'s cells of the action's
  legal range (never a fixed price menu);
* per-seat **win** logits (softmax over the present seats) and a **score** head (placement points), for
  6 seats with padding masked. `value_only=True` skips the policy and price heads; a critic is this
  architecture used value-only.

Milestone C1 (2026-11-13, below: "Supervised data and pretraining", "The 1830 GPU check") gives it a supervised
dataset, a two-stage training loop, a comparison with legacy checkpoints and a policy-only player for
`eval_policy_only.py`. Milestone C2 (below: "Policy gradient for the generic model") trains it by policy-only
policy gradient against a league, on generic inference servers. Milestone A3 ("1867: data, starts, caps and the CPU
smoke") gives 1867 its human importer, a supervised dataset, start files, legal prices on every sampling path and an
end-to-end CPU smoke; C3 ("Mixed-title training and the sharing probe") trains on several datasets at once. The last
section ("1822 family: data, starts, caps and the CPU smoke", 2026-11-14) does A3's work for 1822, 1822PNW, 1822MX
and 1822CA -- and fixes the seat bookkeeping for titles that reorder their players. The frozen 1830 model
(`model_transformer.py`) is untouched.

## API

```python
from rl18xx.agent.alphazero import model_generic as mg

model = mg.GenericPolicyValueModel()                     # default ~8.6M parameters
legal = [mg.legal_actions(g) for g in games]             # titles with a layout (1830): indices, rows, price ranges
arrays = mg.build_inputs(games, legal, dedupe_maps=False, config=model.config)   # numpy (CPU)
inputs = mg.to_device(arrays, device)
out = model(inputs)                                      # out["logp"] (N,), out["price_logp"] (P, 24), ...
price = mg.to_device(mg.price_targets(arrays, prices))   # prices (P,) aligned with arrays["price_rows"], -1 = none
loss = (mg.policy_loss(out, inputs, target_index=idx) + mg.price_loss(out, **price)
        + mg.value_losses(out, win_target, score_target)["ce"])
mg.save_checkpoint(model, path, titles=["1830"]); model, meta = mg.load_checkpoint(path)
```

`LegalActions(rows, price_range, indices, price_grid)` is one position's legal list: descriptor rows
`(n, 14)`, each priced row's `(lo, hi, step)` (-1 elsewhere), the layout indices, and whether legal prices
lie on the `lo + step * k` grid. `legal_actions(game)` builds it for a title with an RL layout: 1830
(`factored_legal_indices`, `action_descriptors(title).gather`, `rl_price_range`: `price_range_for_index`'s
range with the $5 bid ladder as step; `price_grid` False) and 1867 (`rl_legal_moves_json`: the generator
layout's indices and each move's range and step; `price_grid` True, cells in step units --
`docs/rl_descriptors.md` §6). `legal_actions(game, indices)` takes legal indices the caller already
has (the policy-only player); it is the one place a range's step is decided (the supervised records store its
ranges). `add_legal_inputs(arrays, legal)` adds the flat legal list to token arrays that are already built
(`build_inputs` = `encode_batch` + `add_legal_inputs`; the supervised collate stacks per-position encodings
and calls it).

### Inputs (`build_inputs`)

The token arrays of `encode_batch` (`docs/rl_encoder.md` "Batch arrays"), plus with legal lists:

| key | shape | content |
|---|---|---|
| `legal_rows` | (N, 14) | descriptor rows of every position's legal actions, positions in order |
| `legal_pos` | (N,) | the position of each row |
| `legal_offsets` | (B + 1,) | position b's rows are `offsets[b]:offsets[b+1]`; a target "index into the legal list" is relative to `offsets[b]` |
| `map_rows` | (Nm,) | rows of map families (`PlaceToken`, `RemoveToken`, `LayTile`, `HexToken`) |
| `map_k1`, `map_k2` | (Nm,) | each map row's (group, hex) and (group, hex, tile \| city) key |
| `k1_rep`, `k1_group` | (K1,) | per (group, hex) key: a representative row and its group id |
| `k2_rep`, `k2_k1` | (K2,) | per (group, hex, tile \| city) key: a representative row and its (group, hex) key |
| `price_rows` | (P,) | rows with a `price_kind` |
| `price_range` | (P, 3) | their legal (lo, hi, step) |
| `price_cell_mask` | (P, 24) | the non-empty cells of each range (`price_pmf.PriceCells.nonempty`) |

A **group** is a map row's descriptor without its map fields (hex, city, tile, rot) plus the position: e.g.
"the operating corporation lays a tile" and "the CS lays a tile (via CS)" are two groups. Keys are
computed from content (`np.unique`), so nothing depends on the order of the legal rows. Without legal
lists the inputs serve `value_only` forwards. `check_inputs(config, arrays)` refuses ids beyond the model's
vocabularies (a checkpoint older than an appended family).

## Architecture

```
 hex_feat, label, laid tile, slot pointers ──► map encoder (per distinct map) ──► hex tokens (M, H, d)
                                                         │ map_index
 ent_feat by type, ids, pointers ─► entity layer(s) ──► fusion rounds: entities ⇄ hexes ──► trunk ──► FiLM blocks
                                    (relation bias)        (token / home / reservation bias)   │
 legal descriptor rows ─────────────────────────────────────────────────────────────────────────┴─► policy, price, value heads
```

1. **Map encoder**, once per distinct map (so `dedupe_maps` saves its work, as in 1830's search): hex
   features + label embedding + title embedding + the laid tile (tile-feature projection + a per-(title,
   tile) id embedding, zero-initialized); each slot's token and reservation pointer resolved to the
   corporation's *static* embedding (per-(title, entity) id + its static entity type), neutral tokens and
   empty slots as learned vectors. `hex_layers` (4) pre-norm attention layers with a per-head bias from the
   clipped hex distance, the direction (0-5 across a shared edge, else 6), track connectivity across the
   shared edge (both sides have track on it, read from the hex features' edge bits) and "the two hexes hold
   tokens of the same corporation". The map encoder sees no per-position state: everything that varies
   between positions on one map enters in the fusion.
2. **Entities**, per position: a per-type linear projection of the features (one weight per entity type)
   + type embedding + per-(title, entity) id embedding (low-rank, `d_id` 64) + a seat embedding for seats 1-6
   (seat 0 = the mover, title-independent); one hop of the four entity pointers (owner, link, phase_a,
   phase_b) and the two hex pointers (home, destination); the acting entity (`meta[3]`) marked; the
   conditioning embedding added. `entity_layers` (1) self-attention layers with B1's relation bias (a learned
   per-head constant and value scale per relation kind and direction).
3. **Fusion**, per position: the map's hex tokens plus, per hex, whether the acting entity has a token,
   a reservation, its home or its destination there, plus the conditioning embedding. Then
   `fusion_layers` (2) rounds of entities <- hexes and hexes <- entities (cross-attention biased per head by
   "entity holds a token / reserves / calls this hex home / destination") and entity self-attention with
   the relation bias.
4. **Trunk**: [global token, acting entity token (or a learned "none"), attention-pooled entities,
   attention-pooled hexes] -> `d_trunk` (256) -> `trunk_blocks` (2) FiLM residual blocks (1830's
   `FiLMResBlock`, identity at init). Conditioning (FiLM and the token streams) = embedding(title) +
   embedding(round kind) + embedding(step kind) + embedding(seat count), from the encoder's `meta`.
5. **Heads**, in fp32 even under autocast (as 1830's policy and price heads):
   * **Descriptor scorer** (every legal row): `h = W_ctx trunk[pos] + E_family + W_0 tok(ent0) + W_1
     tok(ent1) + W_via tok(via) + W_hex tok(hex) (non-map rows) + E_choice + E_price_kind + W_param
     [param / scale, (param / scale)^2, has_param] + W_flags bits`, each missing field a learned "none";
     score = MLP(h). Entity fields resolve to the encoder's entity tokens: corporation / company / train ->
     `descriptor_entity_base[kind] + index` of the position's title, a player p -> seat `1 + (p - rotation)
     mod n`, `via` -> its company token. `param` is scaled per family (Par: $ / the title's highest market
     price; share percents / 100; a `Choose` position / 10).
   * **Map families** keep the autoregressive factorization over the legal values only. A map row's logit
     is its **group score** (the scorer above on the row without its map fields, so equal over the group)
     plus `log P(hex | group) + log P(tile | city | hex, group) + log P(rot | hex, tile, group)`:
     - `P(hex)`: MLP(h_group + W tok(hex)), softmax over the group's legal hexes;
     - `P(tile | hex)` (tile lays): MLP(h_group + W tok(hex) + W tile_emb), softmax over the hex's legal
       tiles, `tile_emb` = tile-feature projection (color, track, stops, revenue, label, supply: from
       `rl_spec_info(title)["tile_feat"]`, so tiles generalize across titles) + per-(title, tile) id;
       `P(city | hex)` (tokens): the same MLP with the mean over that city's slots of their occupant
       (the occupying corporation's fused entity token, neutral / empty vectors) and reservation, + a city
       index embedding;
     - `P(rot | hex, tile)` (tile lays): MLP(h_group + W tok(hex) + W tile_emb + E_rot + edge term),
       softmax over the (hex, tile)'s legal rotations. The **edge term** sums, over the edges the tile's
       track reaches at that rotation (the catalog tile's edges rotated by r: edge e -> e + r, checked
       against laid tiles), the neighbor across the edge: projected neighbor hex token + raw edge features
       (neighbor exists, has track / a city, town or offboard on the facing edge, this hex's wall and border
       cost on the edge) + a direction vector. A rotation is then scored by what it connects to, which
       transfers across hexes and titles; 1830's rotation head (6 logits from the hex embedding) cannot.
       The neighbor projection runs per hex before the lay rows gather it, for lay rows only (since
       2026-11-13, "Model fit across titles": the same values as projecting each row's gathered tokens).
     Other map families (`PlaceToken`, `RemoveToken`: hex · city; `HexToken`: hex) have one row per key at
     the last level (log 1 = 0).
   * One **softmax over the position's legal rows** (`segment_log_softmax`) then balances map groups against
     each other and against non-map rows by the state-dependent group score.
   * **Price head** (every priced legal row): `MLP(W h_row + W_range range_features)` -> 24 cell logits,
     empty cells of the row's range at -1e4 before the log-softmax. The cell structure comes from the
     descriptor: `price_kind` -> `price_cell_types` ("Bid" ladder at the range's step, "BuyTrain" atoms
     lo / hi / hi-1 + 20 bins -- also `bid_range`'s and, since the 1822 family's layouts, `amount`'s --,
     "BuyCompany" lo / hi + 8 bins) and the range
     from the state; `P(price) = softmax(nonempty cells)[cell] / |cell|`, exactly 1830's `price_pmf`.
     The range features are lo, hi, hi - lo and the step in units of the starting cash per player (the
     encoder's money unit), log1p(hi - lo), a degenerate-range flag and log1p(lo). Zero-initialized output
     (uniform over cells at init, as 1830).
   * **Value heads**: per seat k, MLP([seat token k, trunk]) (one MLP for every seat, as 1830's), win
     logits with absent seats at -1e4, score with absent seats at 0. Zero-initialized outputs (equal odds).
     `model.value_stop_grad = True` trains them on detached features (1830's runtime switch).

## Losses

| function | what | 1830 counterpart |
|---|---|---|
| `policy_loss(out, inputs, target_index=)` | CE to an index into each position's legal list (-1: none), optional per-position weights | supervised pretraining's CE to the human move |
| `policy_loss(out, inputs, target_probs=)` | CE to a distribution over each position's legal rows (sparse pi) | search targets |
| `chosen_log_prob`, `policy_entropy` | (B,) log-prob of a chosen row; entropy over the legal rows | PPO ratio / entropy |
| `price_targets(arrays, prices)` -> `price_loss(...)` | exact NLL `-(log P(cell) - log |cell|)` of observed prices, ranges with one price skipped | `train._compute_price_nll_loss` |
| `value_losses(out, win_target, score_target)` | win CE over the present seats, score MSE over the present seats (placement points), `winner_hit` | `policy_gradient.critic_losses` |

A price-bearing move's full log-probability is `chosen_log_prob + price_logp[cell] - log |cell|`.

## Numerics

bf16 autocast for training, fp32 for inference, as 1830: every attention mask and bias is cast to the
query's dtype, masked logits are -1e4 (finite in bf16), the heads run in fp32 under autocast, segment
softmaxes run in fp32 (`segment_logsumexp` subtracts a detached per-segment max). Padded entities and hexes
are zeroed after the last layer; the global token and hex 0 always exist, so no attention row is fully
masked. Tested: forward + backward under `torch.autocast("cpu", torch.bfloat16)` with finite losses,
gradients and outputs (padded rows included) and per-position probabilities summing to 1.

## Checkpoints

`save_checkpoint(model, path, titles, extra=None)` writes `{architecture: "GenericTokenModel", state_dict,
config, generic_meta}` (atomic rename); `generic_meta` = `{format, encoder_version (ENCODER_VERSION),
titles, layouts: {title: {version, fingerprint, total} or None}, vocab: every global vocabulary, max_seats,
num_cells, extra}`. `load_checkpoint(path)` (`weights_only=True`) refuses with `CheckpointMismatch`:
another architecture (e.g. an 1830 `AlphaZeroTransformer` file), another `ENCODER_VERSION`, `MAX_SEATS` or
cell count, a saved vocabulary that is not a prefix of the running one (ids moved; an appended entry is
fine), and a title whose saved layout fingerprint differs from the engine's. The static per-title tables
(hex geometry, tile table, descriptor bases, normalizers) are non-persistent buffers rebuilt from the engine,
guarded by the encoder version and fingerprints. A generic checkpoint never goes through
`checkpointer.py`'s legacy loader (which pads or drops head tensors, design §5.7).

## Sizes

Default `GenericModelConfig` (`d_model` 256, 8 heads, ff x2, 4 map layers, 1 entity layer, 2 fusion rounds,
`d_trunk` 256 x 2 FiLM blocks, `d_head` 192): **8,607,950 parameters** (1830 v2 default: 7,967,331).

| part | parameters |
|---|---|
| map encoder (4 layers, hex / label / laid-tile / slot inputs, biases) | 2,361,008 |
| entities (per-type projections, pointers, 1 self-attention layer) | 1,066,208 |
| fusion (2 rounds x 3 attention blocks, conditioning) | 3,188,416 |
| per-(title, entity) and per-(title, tile) id embeddings (rank 64) | 161,792 |
| pools + trunk (2 FiLM blocks) | 860,672 |
| policy scorer + map factors + price head | 804,220 |
| win + score heads | 165,634 |

The 1830 v2 model spends 3.4M on its trunk MLP and 2.4M on its map; this one puts its budget in token
processing (map, entities, fusion: 6.6M), because its heads read tokens, not only the pooled trunk. The
tests use a small config (`d_model` 32, one layer of each kind; 151,646 parameters).

## Throughput

CPU, fp32, no grad, batch 64, one thread (`pytest -m benchmark -s tests/agent/alphazero/test_model_generic.py`,
2026-11-13 on the shared machine at load ~30; multi-thread timings there swung 3-10x between repetitions and
are not reported). Default configs, the same 1830 positions for both models:

| batch | generic forward | generic value-only | 1830 v2 forward | 1830 v2 value-only |
|---|---|---|---|---|
| 64 positions spread over 3-5 player walks (240 legal rows, 2 tile-lay positions) | 98/s | 100/s | 149/s | 173/s |
| 64 tile-lay positions (905 legal rows) | 96/s | 100/s | 126/s | 172/s |

Inputs (encode + legal lists + `build_inputs`, single thread): 27k positions/s spread, 11.5k/s tile-lay,
against 47k / 16k for 1830's `encode_games` + legal indices.

So the generic forward costs ~1.5x the 1830 model's (~1.3x on tile-lay positions, where 1830's dense
(hex, tile, rot) grid is expensive and the legal-only head is not). The difference is the fusion: each round's
hexes <- entities block is about one more hex-sized layer (93 tokens through attention + feed-forward), and the
entities <- hexes block projects every hex again, so the generic model runs ~6.5 hex-layer equivalents per
position against 1830's 4 (its trunk is one vector, cheap however wide). Value-only saves little here, because
the legal-only heads are cheap (1830's value-only skips a 26,537-wide policy assembly). If generation speed
matters more than the fusion depth, `fusion_layers=1` or `hex_layers=3` are the knobs (not measured for
quality).

## Tests

```bash
CUDA_VISIBLE_DEVICES= taskset -c 24-31,56-63 nice -19 uv run pytest tests/agent/alphazero/test_model_generic.py
CUDA_VISIBLE_DEVICES= taskset -c 24-31,56-63 nice -19 uv run pytest -m benchmark -s tests/agent/alphazero/test_model_generic.py
```

Positions are random walks of 1830 at 2-6 players (legal indices and descriptors from the 1830 layout) and
1867 at 3-6 (legal indices and descriptors from the 1867 layout `1867-v1`, `docs/rl_descriptors.md` §8). A
mixed batch takes five positions per (title, player count), covering tile lays, tokens and each price kind.

* `test_default_size`; `test_tile_exits_match_laid_tiles` (the rotation convention of the edge term, and
  opposite edges of `hex_neighbors`);
* `test_mixed_batch_shapes_and_normalization`: 1830 and 1867, nine player counts in one batch; shapes;
  probabilities sum to 1 per position; entropy within [0, log n]; price cells sum to 1 with empty cells ~0;
  win shares over the present seats only, absent seats' score 0;
* `test_map_factors_are_distributions`: P(hex) per group, P(tile | city) per hex, P(rot) per (hex, tile) and
  the whole factorization per group each sum to 1;
* `test_legal_order_invariance`: shuffled legal lists give the same (permuted) policy and price outputs;
* `test_padding_is_inert` (plain and de-duplicated maps): a position alone vs in the mixed batch;
* `test_forward_backward` (fp32 and bf16 autocast): every loss finite and positive, finite gradients, only
  the input projections of entity types absent from 1830 / 1867 (1822's concession, minor proxy) without one;
* `test_value_only_matches_the_full_forward` (also inputs without legal rows);
* `test_price_head_covers_every_legal_price`: `P(cell) / |cell|` sums to 1 over every integer of each range
  (bid, train transfer and company purchase ranges) and the price loss is `-log P(price)`;
* `test_build_inputs_checks_vocab`; `test_checkpoint_round_trip_and_refusals`;
* `test_overfits_a_fixed_action_and_price`: the small model learns a fixed legal action (> 0.9) and a fixed
  price cell (> 0.9) on eight 1830 positions (priced and tile-lay ones) in 300 Adam steps;
* (2026-11-13) `test_edge_term_matches_the_reference` (the per-hex edge term = B2's original, values, policy and
  gradients), `test_checkpointed_layers_match` (gradient checkpointing: the same outputs and gradients),
  `test_g1822_family_positions` (1822, 1822PNW, 1822MX and 1822CA positions with stub tile-lay lists in one batch with
  1830 / 1867 ones: finite outputs and gradients, every entity type's projection trained, padding inert).

## Departures from the design docs

* **Map groups get a state-dependent score**, not a per-block normalization (design §2.6 point 2) or a
  constant "learned family bias" (`rl_descriptors.md` §5.4). With 1830's per-block log-softmax, how much
  mass "lay a tile" gets against "pass" depends only on which subsets are legal; here a group's mass is
  `exp(group score)` against the other legal rows', so the model can say "lay now" or "pass now" from the
  state. The 1830 head is the special case of a constant group score.
* **The price head reads the legal range** (lo, hi, step in starting-cash units), not only the descriptor
  and trunk (design §2.7, `rl_descriptors.md` §6). Bins are range-relative, so without the range the head
  cannot tell a $1-$40 bin from a $1-$700 one. Config `price_range_features` (on).
* **The per-(title, entity) id embedding** is initialized at std 0.02 (low rank), not zero (design §3.3):
  the map encoder resolves token pointers to static embeddings and needs to tell corporations apart from
  the start. The tile id embedding is zero-initialized as designed.
* **The map encoder is map-only** (as 1830's), with per-position state entering in the fusion; the design
  did not say. This keeps `dedupe_maps` useful (B1's map de-duplication).

## Open questions

1. **Price steps** (settled by A2): `BaseGame.rl_price_range(idx)` gives `(lo, hi, step)` for every layout
   (1830: `price_range_for_index`'s values with the $5 bid ladder; 1867: the move's own step, prices on its
   grid, cells in step units: `price_cells_for(..., grid=True)`).
2. **1867 rows** (settled by A2): the tests use the real 1867 layout. The model needs nothing
   title-specific beyond `descriptor_entity_base` and the tile table.
3. ~~**`amount` price kind** (1822 partial debt payoff) has no cell spec~~: settled by A4 -- "BuyTrain"
   cells (atoms at the lowest payment, the highest -- 78% of human payments -- and one below it, 20 bins),
   from the family's price census (`docs/rl_descriptors.md` §9); `test_amount_prices_have_cells`. A
   checkpoint written while `amount` had no cells still loads (`load_checkpoint`'s vocabulary check).
4. **Cost of very large legal sets** (settled 2026-11-13, "Model fit across titles"): the edge term in fact
   materialized `(lay rows, 6, d_model)` -- the neighbor tokens were gathered per row, then projected by a linear
   layer that saved them for its backward -- not `(lay rows, 6, d_edge)`. It now projects per hex first and
   computes lay rows only: the same values; what a lay row saves for the edge term is its exits (6 floats). The
   remaining per-row cost (~16 KB a lay row in training) is the scorer's own projections and MLPs.
5. **Fusion bias memory** `(B, heads, E, H)` (settled 2026-11-13): measured, it is not what makes 1822CA dear. The
   bias itself is transient (not saved for the backward); the E x H attention maps the fusion saves are ~8% of
   1822CA's activations. The cost is the hex streams (map layers + fusion at 279 hexes) and the map's H x H
   attention; `model.checkpoint_layers` (gradient checkpointing) cuts 1822CA's activations by 72%. No low-rank form.
6. **Sharing across titles** (design §4.2): the architecture shares everything except the id embeddings and
   the per-title tables; whether to train one model over several titles is still E0/E1's question.
7. The per-type entity projections of the 1822 family's concession / minor-proxy types train only on 1822
   data; an 1830 / 1867 model leaves them at init (harmless, but a DDP run must set
   `find_unused_parameters`).

## Model fit across titles (2026-11-13: the 1822 family, encoder v2)

`scripts/bench_generic_model_fit.py` runs the **default-size** model (8.6M parameters) on random-walk positions of
each title, each (title, mode) in its own process: a training step (forward + backward of the win / score /
policy losses) and a no-grad forward, per position, with the bytes autograd saves for the backward (distinct
storages, parameters excluded: the activation memory a training step holds) and the peak-RSS growth. CPU, fp32,
2 threads, batch 32, mid / late-game positions (move 60 on), 4 players (1822CA 5), 2026-11-13 on the shared machine
(timings +-20%). The 1822 family has no RL layout yet, so it runs the value path (`value_only`) and stub tile-lay
lists (`--modes lays`: well-formed LayTile rows on random hexes, tiles and rotations of the title's tables).

**Value path** (map encoder, entities, fusion, trunk, value heads):

| title | hexes H | entities E | train ms / pos | infer ms / pos | saved MB / pos | peak RSS MB / pos |
|---|---|---|---|---|---|---|
| 1830 | 93 | 33 | 17.4 | 5.5 | 12.4 | 13.6 |
| 1867 | 94 | 54 | 21.5 | 6.4 | 14.7 | 16.2 |
| 1822 | 211 | 127 | 61.6 | 16.5 | 39.8 | 46.3 |
| 1822PNW | 163 | 102 | 42.8 | 12.9 | 29.2 | 33.1 |
| 1822MX | 130 | 108 | 34.1 | 12.4 | 25.2 | 28.0 |
| 1822CA | 279 | 138 | 84.7 | 21.7 | 53.1 | 55.8 |

With 1830's / 1867's real legal lists (109 / 84 rows a position) the full forward costs the same as the value path
(1830 18.0 ms and 12.5 MB, 1867 21.4 ms and 14.7 MB): the legal-only heads are cheap at those sizes.

So an 1822CA position costs ~4-5x an 1830 one in time and ~4.3x in activations -- the token counts (3x the hexes,
4.2x the entities), not one component. Where 1822CA's activations go (8 positions, value path, by tensor shape): the
map layers' H x H attention ~17%, the hex streams (B x H tokens through four map layers and two fusion rounds: linear
inputs, feed-forward, q / k / v) ~40%, the fusion's E x H attention maps ~8%, the entity streams ~14%, E x E ~3%. The fusion bias
`(B, heads, E, H)` that B2's open question 5 worried about is transient (not saved); a low-rank form would save little.
On a GPU, a training batch of 64 1822CA positions holds ~3.4 GB of fp32 activations (1830: ~0.8 GB) before bf16
autocast -- fine on the training GPUs, and the lever below if not.

**Lay-heavy lists** (1,000 stub lays a position):

| title | edge term | train ms / pos | infer ms / pos | saved MB / pos |
|---|---|---|---|---|
| 1830 | B2's (project each row's 6 gathered neighbor tokens) | 51.8 | 14.8 | 33.8 |
| 1830 | per hex, lay rows only (now) | 42.5 | 12.1 | 28.0 |
| 1867 | now | 47.6 | 13.2 | 30.2 |
| 1822CA | B2's | 110.3 | 32.0 | 74.4 |
| 1822CA | now | 112.1 | 32.1 | 68.7 |

B2's edge term ran each lay row's six gathered neighbor tokens (d_model each) through `edge_tok`, which saved them
for its backward: `rows x 6 x 256` floats, ~6 MB a position at 1,000 lays. `_edge_term` now projects every hex to
`d_edge` once and gathers for lay rows only (non-lay map rows get 0, as before): the same values and gradients
(`_edge_term_reference` keeps B2's form for the tests and `--edge-term reference`). The remaining ~16 KB a lay row
(~4,000 floats) is the scorer's projections of its gathered entity / hex / tile tokens (ent0, ent1, via, hex ref,
f1-f3 hex and tile: d_model each, saved by their linear layers) and the four MLPs. Projecting those per token before
the gather would halve it for lay-heavy batches but costs more than it saves on the common short lists (a stock
round's 20-100 rows against 279 hexes x 4 projections); not done.

**Gradient checkpointing** (`model.checkpoint_layers = True`, a runtime switch like `value_stop_grad`, not saved
in checkpoints): the map, entity and fusion attention blocks are recomputed in the backward instead of storing their
activations. The same outputs and gradients (`test_checkpointed_layers_match`):

| title | mode | train ms / pos | saved MB / pos | peak RSS MB / pos |
|---|---|---|---|---|
| 1830 | value | 20.5 (17.4 without) | 2.9 (12.4) | 4.8 |
| 1822 | value | 73.5 (61.6) | 10.8 (39.8) | 16.7 |
| 1822CA | value | 106.7 (84.7) | 14.7 (53.1) | 17.7 |
| 1830 | 1,000 lays | 46.5 (42.5) | 18.5 (28.0) | 21.4 |
| 1822CA | 1,000 lays | 137.0 (112.1) | 30.3 (68.7) | 37.1 |

~1.2-1.25x the training time for 3.5-4x less activation memory. Off by default; turn it on for 1822CA-sized
training batches if GPU memory binds.

Tests: `test_edge_term_matches_the_reference`, `test_checkpointed_layers_match`, `test_g1822_family_positions`
(above, "Tests").

## Supervised data and pretraining (milestone C1, 2026-11-13)

Code: `generic_data.py` (records, replay, collate, loader), `generic_pretrain.py` (two-stage training, metrics,
comparison with a legacy checkpoint), `generic_policy.py` (policy-only player); `main.py convert-generic`,
`pretrain-generic`, `compare-generic`; `scripts/eval_policy_only.py` players `generic:<checkpoint>[@T]`. Tests:
`tests/agent/alphazero/test_generic_data.py`, `test_generic_pretrain.py`.

### Dataset: compact game records, replayed and encoded in the loader

Storage was decided by measuring 1830's corpus (2.5M positions): stored token arrays are ~58 KB per position
raw, ~4.8 KB LZ4-compressed per position (**~12 GB**, a little under the ~20 GB threshold; ~0.34 KB per position
only when a whole game is compressed as one block), while a game record with every label is ~60 bytes per
position (**144 MB** for the corpus). Replay (~15 us per action) + encode (~25 us) runs at ~12k positions/s per
loader worker unshuffled, ~6k/s through the shuffle buffer -- several times what the GPU consumes. So a dataset
holds records and the loader re-encodes: it is independent of the encoder version (B1's encoder is version 1 and
its open points foresee additions; stored arrays would need a re-conversion per bump), small, and the same format
serves 1867 and mixed-title runs.

A dataset directory: LMDB `g:<game id>` -> lz4(pickle(record)), `meta.json` (titles, layout version and
fingerprint, the encoder version at conversion, sources, split rule, `finished_only`, counts -- by split, player
count and source, dropped games by reason and player count --, match / skip statistics), `games.jsonl` (id,
source, split, players, decisions, actions, finished, skipped labels) and `dropped.jsonl` (id, source, players,
reason of every dropped game; since A3). A **record** (`generic_data.convert_game`):
the action log as the engine replays it (JSON), players in seating order, optional rules, rule variants (off for
human games), the layout version and fingerprint, and per decision:

| field | content |
|---|---|
| `action_pos` | the decision's action in the log (the position is the state before it) |
| `legal_offsets`, `legal_indices` | the legal layout indices (sparse; `model_generic.legal_actions`) |
| `legal_price_range` | (lo, hi, step) of every legal row with a price kind, -1 elsewhere |
| `chosen_index`, `target` | the human move's layout index and its row in the legal list |
| `price` | the human price when the chosen row has a price kind and it lies in the range, else -1 |
| `family` | the chosen row's descriptor family |
| `meta` | the encoder's meta row at conversion (title, round / step kind, acting entity, seats, rotation, ...); the loader checks the replayed rotation against it |

and per game the outcome in **seating order**: net worth (`result()`), win share (ties split) and placement
points (`policy_gradient.default_placement_points`). The collate rotates them into the encoder's seats (seat k =
player `(k + rotation) mod n`, as the legacy `canonicalize_value_target`). The split is
`pretraining.in_validation_split(game id, 0.05)`: a game is held out exactly when it was held out of `lmdb_v6k`
/ SL checkpoint 10.

**Labels.** Each decision's human action must match exactly one legal index, else the position is skipped and
counted by action type (`unmatched`, `ambiguous`, `not_legal`) and the action still applied. 1830's importer is
the legacy converter's own matching (`pretraining._action_dict_to_factored_index` on the replayed
`RustGameAdapter`), so labels, legal lists and price targets equal the legacy converter's index for index
(`test_matches_legacy_converter`). Forced positions (one legal index) are kept. A title without an importer
raises `UnsupportedTitle` (`generic_data.register_importer`; 1867's came with milestone A3, below: "1867: data,
starts, caps and the CPU smoke"; there is no engine-side `match_human_action` -- the matching is
`generic_import_1867.matching_indexed` over `rl_legal_moves_json`).

**Conversions (2026-11-13, current engine, 7 processes, ~70-110 s each):**

| source | games converted | dropped | positions train / validation | legacy LMDB |
|---|---|---|---|---|
| `1830_clean_all` (lmdb_v6k's source = SL ckpt 10's games) | 4,076 | 266 error records, 60 no longer replay (rules fixed 2026-10-16: 56 "Run Routes cannot process pass", 2 ERIE E11, 2 SR buys) | 2,376,021 / 117,053 | lmdb_v6k: 4,136 games, 2,416,316 / 118,944 (converted on the old engine) |
| `1830_clean_20261017` (the re-clean on the current rules) | 4,216 | 187 error records | 2,474,663 / 122,211 | lmdb_v7: **identical** counts (2,474,663 / 122,211) |

Skipped labels: `unmatched` bankrupt 945 / 963 and discard_train 2 / 2, `ambiguous` 0, `not_legal` 0. Forced
positions kept: 311,771 / 325,314 (12.5%). Price targets (open ranges): 101,285 / 105,094. Matched positions by
family (`1830_clean_all`): Pass 987,430; BuyShares 315,395; RunRoutes 284,904; Dividend 284,633; LayTile 272,865;
SellShares 82,196; BuyTrain.depot 81,424; Bid.company 66,504; PlaceToken 38,703; Par 29,828; BuyTrain.corp
27,166; BuyCompany 16,181; BuyTrain.exchange 2,734; DiscardTrain 1,594; ExchangeShares 746; BuyTrain.pool 602;
Bankrupt 169. Legal lists: mean 11.5 rows, median 6, p99 77, max 185 over the 2.49M positions (design §2.6 and
open question 4 above expected tile-lay states of "a few thousand"; 1830's human games never come near). 856 of
the 4,076 games (254,558 positions) end before the game's end (the log stops; the site
scored them on `result()`): their value targets are net worth at that point, as in the legacy data.

**Loader.** `PositionBatches` (a `torch` `IterableDataset`; `DataLoader(..., batch_size=None, num_workers=W)`,
spawn): each worker takes every W-th game of the epoch's order (permuted per seed and epoch), replays it, encodes
each decision (`position_item`: `encode_batch([game])`, the legal descriptor rows gathered from the stored
indices, targets in the seat frame) and passes it through a shuffle buffer of `buffer_positions` LZ4-compressed
items (~5 KB each; 32k by default, ~150 MB per worker), so a batch of 256 mixes ~50 games (legacy: a full
shuffle). `collate` pads and stacks the single-position encodings exactly as `encode_batch` pads a batch, then
calls `add_legal_inputs` (bit-identical to `build_inputs` on the same positions, tested, 1830 + 1867 padding
included) and adds `target_index`, `price_cell` / `price_log_count` / `price_valid` (the chosen row's observed
price), `win` / `score` (B, 6), `family`, `num_players`.

### Training (`main.py pretrain-generic`)

The legacy recipe (`pretraining.pretrain_model` + `_refit_value_heads` with `main.py pretrain`'s defaults):

* **Stage 1**: `(1 - 0.03) * CE(target) + 0.03 * mean over the legal rows of -log p` (label smoothing over the
  legal set; the legacy pi gave the target 1 - eps and every legal index eps / n) - 0.01 entropy + 0.1 price NLL
  + 0.1 win CE + 0.1 score MSE (placement points); AdamW lr 1e-3 (decay 1e-4 on matrices only, as
  `train.build_optimizer`), cosine to 0 over the planned steps, clip 1.0, batch 256, 10 epochs; bf16 autocast on
  CUDA (`--bf16 on` forces it on CPU), fp32 evaluation; validation per epoch (or `--eval-every` steps), the best
  total validation loss saved as `best_joint.pth`, early stop after 4 evaluations without improvement.
* **Stage 2**: `best_joint.pth`'s value heads re-initialized (zero outputs: equal odds) and trained alone on the
  frozen network (`value_stop_grad`, value-only forwards), validated every 250 steps on the win CE (patience 4,
  at most 20k steps), the best heads saved as `refit.pth`; `final.pth` = the refit.
* Logged to `<out>/tb` and `<out>/history.jsonl`: losses, top-1 / top-5 over the legal moves, top-1 per
  descriptor family and per player count, the chance top-1 of a uniform policy (mean 1 / legal count), price NLL,
  win CE against the equal-odds baseline ln(n), winner accuracy (the argmax seat is a winner). Checkpoints via
  `save_checkpoint` with `extra` = stage, step, metrics, the training config and the dataset's meta.

Departures from the legacy loop: batches mix player counts (the model masks absent seats; the legacy sampler made
each batch one count, which only reorders an epoch -- every bucket is exhausted, so the totals are the same);
positions are mixed through a shuffle buffer, not a full shuffle; the score target is placement points (the
legacy one was net-worth fractions), as the policy-gradient critic's.

`main.py compare-generic --generic <final.pth> --legacy <SL ckpt 10> --data <dataset>` scores both models on the
same decisions of a split (the legacy model through its encoder and dense policy read at the same legal indices,
renormalized; its value over the present seats): top-1 / top-5 / CE overall, per family and per player count, win
CE and winner accuracy. It also counts positions where the two encoders disagree on the mover (0 on the smoke
games).

### Policy-only play (`generic_policy.py`)

`GenericPolicy(path).evaluate(games, legal_indices)` builds the inputs of a batch of positions from their legal
indices (`legal_actions(game, indices)` + `build_inputs`) and runs one fp32 forward; `choose` samples a move from
`tempered(probs, T)` (as every legacy policy-only player) and `sample_price` a price: the engine's price for a
fixed one (lo == hi), else a cell of the price head over the move's legal range (`--price-eps` mixes uniform cells
in, as `policy_gradient.choose_price`) and a price inside it. In `eval_policy_only.py` a player
`generic:<checkpoint>[@T]` needs no inference server: each pool worker loads the checkpoint once
(`--generic-device`, default cpu) and scores its games' pending generic positions in one forward while the legacy
sides' requests are on their servers. Everything else (starts, `--auction-start`, the rule against train
shuffling via `start_positions.new_game`, seat arrangements, scoring) is the legacy eval's. In-process inference
is enough: the default model scores ~100 positions/s per CPU thread, i.e. 16 workers play an 800-game 4-player
match's ~225k generic decisions in a few minutes; a GPU inference-server path can come with policy gradient.

### CPU smoke (2026-11-13, `CUDA_VISIBLE_DEVICES=`, cores 24-31,56-63)

* Conversions: the two above (whole corpora; a `--limit 100` conversion takes ~5 s).
* Training: small config (151,646 parameters), the first 100 training games (56,211 positions), batch 64, 400
  steps, 2 loader workers: 64 s for stage 1 (~6 steps/s). Training loss (window means) 1.825 -> 1.256, policy CE
  1.370 -> 0.959, top-1 0.61 -> 0.71. Validation (first 20 validation games, 10,346 positions) at steps
  100 / 200 / 300 / 400: top-1 0.675 / 0.673 / 0.677 / 0.678 (chance 0.298), top-5 0.92, policy CE 1.033 -> 0.973,
  win CE 1.483 -> 1.359 (equal odds 1.508); stage 2 (200 steps) win CE 1.352, winner accuracy 0.367.
* `compare-generic` on 6 validation games (3,068 positions): SL ckpt 10 top-1 0.705 (its documented human
  top-1 is 0.712), the smoke model 0.677; mover disagreements 0.
* Policy-only games against SL ckpt 10 (both T 0.5, from the auction, real rules + train-shuffle rule): 4 players
  2v2, 6 games, 25 s; 2 players 1v1, 4 games, 22 s; all played to the end. The 150k-parameter smoke model lost
  every game (worth share 0.41 / 0.39), as expected of 400 steps on 100 games; its auction bids are plausible
  (e.g. DH $75 / $85, B&O $220).

## The 1830 GPU check (design §4.4)

`scripts/launch/2026-11/bridge/gpu_check_1830.sh [convert] [pretrain] [compare] [eval]` (all by default; paths and
devices from environment variables at its top):

1. **convert**: `main.py convert-generic --data-dir human_games/1830_clean_all --output
   human_games/generic_1830_clean_all_v1` (CPU, ~2 min). `DATA_SRC=human_games/1830_clean_20261017` trains on the
   re-cleaned corpus instead (lmdb_v7's games, 140 more). Both split by the same game-id hash, so no validation
   game of either was in ckpt 10's training: either comparison is fair.
2. **pretrain**: `main.py pretrain-generic --model-preset default` with the legacy defaults (10 epochs, batch
   256, lr 1e-3 cosine, patience 4, value refit), `--device cuda`, 6 loader workers.
3. **compare**: `main.py compare-generic` of `final.pth` and SL ckpt 10 on every validation decision (117k):
   top-1 per family (Par / BuyShares / LayTile / BuyTrain.* separately), win CE vs equal odds, winner accuracy.
4. **eval**: `eval_policy_only.py`, generic vs SL ckpt 10 both at T 0.5, `--auction-start`, 800 games each: 4
   players 2v2, 1v3 with the generic model alone, 1v3 with ckpt 10 alone, 2 players 1v1 (equal strength = 0.5 /
   0.25 / 0.75 / 0.5), 12 games of each saved for the viewer.

Pass bar (design): within noise of ckpt 10 -- per-family top-1 within ~0.01 (the per-family standard errors on
117k positions are 0.002-0.01; Par has ~1.4k validation positions, se ~0.013), win CE no worse, and the
head-to-heads within two standard errors of equal (800 games: se ~0.015 in 2v2).

**GPU time (estimate; no GPU was available to measure).** SL ckpt 10's own two-stage run took ~3.5 h
(`20261004_134558`: session start 13:45, checkpoint 10 at 17:20). The generic model costs ~1.3-1.5x the 1830
model per position (forward, CPU-measured above), so stage 1 ~4.5-5.5 h for 10 epochs (less if it stops early),
stage 2 ~0.3 h, compare ~0.2 h, the four evals ~0.5-1 h (mostly CPU: the generic side in the pool workers, the
legacy side on the GPU server): **~6-7 GPU hours** in all, about half a day of wall clock. Read the real rate off
the first `step ... ` lines' `elapsed` (`<out>/history.jsonl`); the loader is not the limit (~6k positions/s per
worker through the shuffle buffer, 6 workers).

## Open questions after C1

1. **Which corpus**: `1830_clean_all` (ckpt 10's games, 60 no longer replay) or `1830_clean_20261017` (the
   re-clean on the current rules, = lmdb_v7). The recipe uses the former; the latter is the corpus every
   1830 start file now comes from.
2. **Unfinished human games** (10% of positions): their value targets are net worth when the log stops, as in the
   legacy data. A `finished` flag is in each record if the value loss should skip them.
3. **Mixed titles** (E0 / E1, 1830 + 1867): `PositionBatches` reads one dataset; a mixed run needs a loader over
   several (each game already carries its title, and batches can mix titles). Built in C3 (below: "Mixed-title
   training and the sharing probe").
4. **1867 import**: done (A3, `generic_import_1867`; below).
5. **No resume** of a pretraining run (the legacy one has none either).

## Policy gradient for the generic model (milestone C2, 2026-11-13)

Code: `generic_inference.py` (sparse-legal inference server), `generic_play.py` (how a title is played, where
games start), `generic_pg.py` (the trainer); `main.py policy-gradient-generic`;
`scripts/build_generic_start_positions.py`, `scripts/eval_generic.py`, `scripts/bench_generic_server.py`;
`scripts/eval_policy_only.py --generic-server`; recipe `scripts/launch/2026-11/bridge/pg_generic_1830.sh`.
Tests: `tests/agent/alphazero/test_generic_pg.py` (14, ~20 s on CPU, one spawns a server, one runs `generic_pg.run`
end to end with a resume). The legacy stage (`policy_gradient.py`, `inference_server.py`) is unchanged: the
generic stage imports its model-agnostic pieces.

### Inference server (`generic_inference.py`)

`GenericInferenceServer` subclasses the legacy `InferenceServer` (control ops, batching up to `batch_size`
positions within `batch_timeout_ms`, reload, health, the optional collector thread, the process handles, replicas,
`RL18XX_SERVER_PROFILE`) and replaces the forward:

* **Request** (`GenericBatch`, one per worker per round): the positions as `model_generic.build_inputs` arrays --
  the encoder's token arrays plus the flat legal descriptor rows, map keys and price ranges, all built in the
  worker -- as one LZ4-compressed pickle (1830: ~58 KB per position raw, ~1.6 KB compressed, ~12 us per position to
  compress; `--no-compress-requests` sends them raw), the request's `price_eps`, and whether to return the full
  price-cell log-probabilities.
* **Merge** (`merge_inputs`, on the collector thread with `--server-pipeline`): token arrays padded to the batch
  maxima with the encoder's own padding (any titles and player counts), `map_index` offset, legal rows / positions
  / map keys (groups, (group, hex), (group, hex, tile | city)) / price rows offset. The merged forward equals each
  request's own forward (tested, < 2e-5).
* **Reply** (`GenericReply`): per legal row its log-probability at temperature 1 (the worker tempers); per priced
  legal row a price **cell** sampled on the device from the head over the row's legal range mixed with the
  request's `price_eps` uniform over the range's non-empty cells (`policy_gradient.choose_price`'s distribution,
  tested empirically) and that draw's log-probability -- the worker draws the price inside the cell
  (`PriceCells.sample_price`) for the row it plays; per position win shares and score-head outputs over the 6 seats
  (encoder frame). All of a batch's reply arrays go to the host in **one** float32 buffer, one non-blocking copy to
  pinned memory queued behind the forward (the legacy `_StagedReplies` idea): one GPU wait per batch.
* `start_generic_inference_server(..., replicas=, pipeline=, threads=)` returns the legacy `ServerHandle` /
  `ReplicatedServerHandle`; `GenericInferenceClient` (a legacy `InferenceClient` with `send_inputs` /
  `receive_outputs`) is a worker's connection. `LocalGenericServer` / `LocalGenericClient` run the same forward
  in-process (tests, `eval_generic.py` without `--server`, benchmarks). Model loading: fp32 with TF32 matmuls on the
  GPU (`load_generic_server_model`; the trainer uses TF32 too, so the PPO ratio sees one precision, as the legacy).

### Games (`generic_pg.play_generic_pg_games`, pool workers)

As the legacy worker, with the generic pieces: a task plays `games_per_task` games of one player count (the
legacy `CountSchedule` hands out counts); each starts from `generic_play.replayed_start` (below) and is dealt seats
by the legacy `_deal_seats` (league slots by PFSP, or the fixed opponent / pool). Every round: forced moves are
applied without the network (one legal row and no open price; `TitlePlay.is_decision`); the pending positions are
encoded in **one** native call (`encode_batch`), whose `meta[:, 5]` (the mover's seat) routes each position to its
seat's server; each server's share is sliced (`slice_positions`) and gets its legal lists (`add_legal_inputs`), all
requests are sent before any reply is awaited. The move is sampled from `softmax(logp / T)` computed in log space
(`tempered_logp`: a sharp policy's tiny probabilities are never clipped), the price as above. The learner keeps a
reservoir of `positions_per_seat` decisions per seat (`PGRow`): the position's token arrays as the server saw them
(LZ4, ~2 KB), its legal rows and ranges, the row played, the probability it was sampled with (tempered row +
price cell), the cell, its seat and the player count. At the game's end each row gets its seat's reward and the
critic's targets (win share and placement points in the encoder's seat frame, `generic_data.seat_frame`).
Outcomes come from `result()` in seating order; the decision cap scores a game on net worth as it stands, as the
legacy (the Rust engine has no `end_game`; the adapter's was a no-op).

Measured per decision on the CPU (`bench_generic_server.py worker`, 1830 positions at 3-6 players): legal rows
31 us, encode 42 us, slice + legal inputs 9 us, compression 13 us: **~95 us**, against **~256 us** for the legacy
worker's `_rust_encode` alone.

### Update (`generic_pg._update`)

* **Baselines** first, in one value-only pass of the critic over the update's rows (`subtract_baselines`):
  `A = reward - V(s)`, the critic's win share of the mover (reward `win`) or its score head (reward
  `placement`). `gae_lambda` 1 only (the legacy default; the lambda-return path needs values at generation time).
* **PPO**: per microbatch, the rows are collated back into model inputs (`collate_rows`: decompress, pad as
  `encode_batch` pads, `add_legal_inputs`; ~65-110 us per row on the CPU, built one microbatch ahead on a helper
  thread so it overlaps the device work; `collate_seconds` in the history is the wait for it) and run through the
  learner and the anchor (no grad). The learner's policy is `segment_log_softmax(logits / T)` over each position's legal rows
  (`T` = `learner_temperature`; the map families' factorization is inside the logits, so tempering it tempers the
  joint); a move with an open price adds its cell's log-probability. Loss: clipped surrogate +
  `kl_coef` * KL(learner || anchor) over the legal rows and the played row's price cells (both at `T`) -
  `entropy_coef` * entropy (`generic_pg_losses`). Clip 1.0 on the gradient norm, finite checks, adaptive
  `kl_coef` toward `kl_target`, the anchor moving every `anchor_every` updates -- the legacy stage's.
* **Critic**: the same architecture used value-only (`value_only=True`), initialized from `--value` (default: the
  policy checkpoint's own value heads), trained on every row once (epoch 0) with `model_generic.value_losses`:
  win CE over the present seats + `score_loss_weight` * MSE of the score head to placement points (with a
  `placement` reward = the reward; C1's pretraining trains the same targets).
* No dropout in the generic model, so train / eval mode does not change the ratio (the legacy trains in eval mode
  for that reason; the generic stage does too).

### What is reused from the legacy stage, and what is not

Imported from `policy_gradient.py` unchanged: `CountLeague` / `League` / `league_for_run` / `league_slots` /
`load_population` / `load_min_shares` (PFSP per player count, `min_share` and `min_share_players` pins, own
snapshots joining every `league_snapshot_every` updates, the latest `league_max_snapshots` kept), `_deal_seats`,
`CountSchedule`, `player_counts`, `count_path`, `count_limits` (per-count decision and auction-stall caps),
`learner_seats_for`, `default_learner_seat_weights`, `check_players` (seat and start-file checks), `parse_reward`,
`placement_points` / `default_placement_points` (per count), `_Scores`, `resume_overrides`. The run directory
(`learner/`, `critic/`, `state.json`, `optimizer.pth`, `history.jsonl`, `games/`, `STOP`) and the history keys are
the legacy run's, plus `log_ratio_abs_mean`, `collate_seconds`, `baseline_seconds`, `worker_seconds` and
`learner_price_decisions`.

Not ported: the 1830 exploiter rewards (`president:<CORP>`, `rusher`, `rust`, `bankrupt`: rejected), `--shared` /
`--value-follow` / `--value-pg-weight` (the AlphaGo-style split), `gae_lambda` < 1, values at generation time.
**Legacy checkpoints in a generic league: not supported** (refused by `check_config`). Not cheap: a legacy member
needs the legacy encoder in the worker and a legacy server, and a league slot reloads members of either kind, so
slots would have to be split by kind and PFSP restricted per kind. For a generic 1830 run the legacy bots (gen4 ..
gen6, the exploiters) are the eval panel instead (`eval_policy_only.py` seats both kinds).

### Starts (`generic_play.py`, `scripts/build_generic_start_positions.py`)

`replayed_start(title, n, ...)` draws the legacy mix -- the opening (`auction_fraction`), a mid-game cut
(`midgame_fraction`), a random opening (`random_fraction`, and always without human starts of the count), else a
human first-Stock-Round start -- and replays it on a new game, skipping and counting starts that no longer replay.
**1830 delegates to `start_positions.replayed_start`**, so a generic 1830 run starts exactly as the legacy runs
(random auction endings that know the SV and the B&O par, real-rules or `auction_unlock` auctions). Other titles:
start files with a `title` (and `seed`) field; random openings play the opening rounds through the title's own
legal-move generator (`g1867_legal_moves_json` / `g1822_legal_moves_json`: uniform over moves, prices uniform
multiples of the step in `[lo, 2 lo]` -- intentionally un-human, as 1830's) or its layout. (The 1822 family opens
with a Stock Round: this said its random opening was the empty game; since the family's data milestone its opening
*is* that Stock Round -- human and random starts are where it ends, the auction start is the fresh game; last
section.) `TitlePlay` (legal rows, decisions, fixed prices,
applying a row) reads the title's layout (`mg.legal_actions`; 1830 through `apply_action_index` /
`price_range_for_index`, a generator layout -- 1867 -- through `rl_apply_index` / `rl_price_range` since A3: before
that 1867 could not be played, `apply_action_index` refuses a generator layout); `register_title_play` is the hook
for a title that needs something else.

The builder replays each usable human game (1830: cleaned exports as the generic importer replays them, the raw
exports consulted for optional rules as the legacy builders do; 1867 / the 1822 family: raw exports through
`tests/title_replay_harness.replay_game`, as the corpus audits), cuts the engine's own log where Stock Rounds begin
(`stock_round_cuts`), checks every prefix replays strictly to an unfinished Stock Round, and writes
`start_positions_<title>_<N>p<suffix>.jsonl` / `midgame_positions_<title>_<N>p<suffix>.jsonl`. Checked: on the
first 300 games of `1830_clean_20261017` the 4-player output equals the legacy `_v2` files exactly (153 SR1 starts,
the same post-auction positions; 151 mid-game games, the same cuts); on the first 200 files of `human_games/1867`:
139 SR1 starts / 136 mid-game games over 3-6 players (50 files without actions, 7 without a Stock Round, 3 that no
longer replay), and starts of every kind sampled from them replay (human / mid-game / auction / random).

### Evals

* `eval_policy_only.py` (1830): `generic:<ckpt>[@T]` players against legacy or generic ones (C1); new
  `--generic-server` hosts the generic players on generic inference servers (the GPU) instead of in the pool workers.
* `scripts/eval_generic.py` (any title with a layout): generic vs generic only, starts from `generic_play`, moves and
  prices drawn exactly as the PG workers draw them, each side in-process (CPU) or on a server (`--server`); seat
  arrangements and scoring are `eval_policy_only.py`'s.

### CPU smoke (2026-11-13, `CUDA_VISIBLE_DEVICES=`, cores 24-31,56-63)

C1's smoke checkpoint (small config, 151,646 parameters) as policy, critic and league member, with its stage-1 best
(`best_joint.pth`, pinned `min_share` 0.2 at 4 players) as the second member; gen6's recipe scaled down: player mix
2:1,3:1,4:2,5:1,6:1 (one opponent slot per count), placement reward, T 0.5, KL target 0.08, anchor every 2,
auction 0.3 (real rules, stall 150), mid-game 0.3, random 0.2, per-count caps, 8 workers x 8 games, 3,000 rows per
update, minibatch 512 / microbatch 256, lr 2e-5, servers on the CPU (1 thread each), own snapshots joining the
league every 2 updates. The command (paths shortened):

```bash
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 taskset -c 24-31,56-63 nice -19 .venv/bin/python main.py policy-gradient-generic \
  --policy <c1 smoke>/final.pth --title 1830 --out-dir <out> --run-name smoke1 --player-mix 2:1,3:1,4:2,5:1,6:1 \
  --max-decisions 1200 --max-decisions-by-players 2:1000,3:1100,5:1300,6:1500 --workers 8 --games-per-task 8 \
  --positions-per-seat 16 --population <league.json> --opponent-slots 1 --slot-refresh-every 1 \
  --league-snapshot-every 2 --league-max-snapshots 2 --reward placement --anchor-every 2 --rows-per-update 3000 \
  --minibatch 512 --microbatch 256 --lr 2e-5 --critic-lr 3e-5 --kl-coef 0.02 --kl-target 0.08 \
  --learner-temperature 0.5 --opponent-temperature 0.5 --max-updates 4 --snapshot-every 2 \
  --random-start-fraction 0.2 --midgame-fraction 0.3 --auction-fraction 0.3 --no-auction-unlock \
  --auction-stall-decisions 150 --save-game-every 10 --start-positions <human_games>/start_positions_1830_{n}p_v2.jsonl \
  --midgame-positions <human_games>/midgame_positions_1830_{n}p_v2.jsonl --server-cpu --server-threads 1 --threads 4 \
  --inference-batch-size 256
# then: ... policy-gradient-generic --resume <out>/smoke1 --max-updates 6
```

* 4 updates, then a **resume** from `state.json` for updates 5-6 (optimizer, league state and anchor restored):
  ~33 s per update (~20 s generation at ~1,500 decisions/s, ~13 s update), **8.5-11.7k games/h** on CPU servers.
  688 games, all finished (mean 446 decisions; per count 400 / 391 / 439 / 502 / 468 at 2-6 players); starts human
  252, auction 219, mid-game 159, random 58; 62 saved games.
* Losses finite throughout; at each update's first minibatch `approx_kl` ~1e-8 to 8e-4 (the trainer recomputes the
  probabilities the servers sampled with), `log_ratio_abs_mean` <= 0.01; KL to the anchor 1e-5 to 5e-4 at lr 2e-5
  (`kl_coef` adapted down to its floor); critic CE 1.18 -> 1.07-1.14 (equal odds over 2-6 seats ~1.3), the
  critic's weights moved (L1 54 over 6 updates); snapshots 2 / 4 / 6 kept (learner and critic), the league holding
  the two members plus the latest two own snapshots per count.
* Worker time: ~5% encode + send, ~5% playing moves, ~90% waiting for the CPU servers (the network is the limit).
* Policy-only evals of update 6 against its start (T 0.5, real-rules auction starts): `eval_policy_only.py` 4p 2v2
  0.625 +- 0.071 (48 games, in-process, 16 s); 4p 1v3 0.396 +- 0.071 (fair 0.25; `--generic-server`, 30 s);
  `eval_generic.py` 3p 1v2 0.458 +- 0.073 (fair 0.33; human SR1 starts, 11 s), 2p 0.500 +- 0.104 (`--server`).
  Six updates at KL ~1e-4 cannot have changed much: these show the paths work, not a gain.
* A second run (smoke2: the same setup at lr / critic lr 1e-4, anchor and own snapshots every 10 updates,
  30 updates, 3,400 games in ~16 min, ~12.4k games/h; one auction stall, the rest finished): KL to the moving
  anchor up to 0.02 within an anchor period, critic CE 1.21 -> 0.82-1.0 (winner hit 0.46 -> 0.60-0.68, the
  critic's weights L1 310 from its start), the learner's per-update score against the league 0.89 -> 1.2-1.8 (fair
  1.0). Update 30 against its start (policy-only, T 0.5, real-rules auction starts): **4p 2v2 0.933 +- 0.016**
  (240 games, `eval_policy_only.py`), **4p 1v3 0.821 +- 0.025** (fair 0.25; `--generic-server`), **2p 0.883 +-
  0.029** and **6p 3v3 0.783 +- 0.038** (120 games each, `eval_generic.py`). Gains are cheap from a 400-step
  supervised model; they show the loop learns at every player count, not what it does from a strong start.

### Throughput

CPU (`bench_generic_server.py`, shared machine at load ~30, 1830 positions at 3-6 players):

| measurement | result |
|---|---|
| worker CPU per decision (legal + encode + split + compress) | ~95 us (legacy `_rust_encode` alone: ~256 us) |
| request size per position | 58 KB raw, 1.6 KB LZ4 |
| server merge + decompress (`_prepare`) | 12-19 us per position |
| small model forward, in-process, 2 threads | 1.5-1.9k positions/s (batch 64-1024) |
| default model forward, in-process, 2 threads | ~160 positions/s |
| spawned server, small model, 8 workers x 32 positions, 2 threads | 1.6k/s; compressed 1.7k/s; + collector thread 2.1k/s; 2 replicas 3.6k/s |
| spawned server, no network (transport ceiling), 12 workers | 36k/s raw, 35k/s compressed, 52k/s with the collector thread or 2 replicas |

So on the CPU the network is the limit, and one server process moves ~35k positions/s (50k with the collector
thread) before its forward -- above what a GPU forward of the default model is expected to give (below).

**GPU estimate (not measured: no GPU).** The legacy servers ran ~14-25k states/s per GPU (2 replicas, staged
replies; `docs/search_learning_design.md` "Reply path") and gen6 made ~125k games/h. The generic default model costs
~1.5x the 1830 v2 model per position (CPU forward, "Throughput" above; ~6.5 against 4 hex-layer equivalents), the
generic worker ~0.4x the legacy worker's CPU per decision, and the server's transport ceiling is above the GPU's
rate. If gen6 was bounded by the GPU forward, a generic 1830 run makes **~80k games/h** (~125k / 1.5); if it was
bounded by the workers' CPU, closer to 125k. Updates add ~10 s of CPU collate per 65,536 rows (two passes at
~70 us per row; the legacy update has none), overlapped with the device work by the prefetch thread as far as the
GIL lets it. `fusion_layers=1` or `hex_layers=3` would cut the forward if speed
matters more than depth (not measured for quality). Read the real numbers off `games_per_hour_run` and
`RL18XX_SERVER_PROFILE=60`.

### GPU recipe (1830)

`scripts/launch/2026-11/bridge/pg_generic_1830.sh` (NOT LAUNCHED): from the GPU check's
`model_checkpoints_generic/bridge_1830_check_20261113/final.pth` (policy, critic and the first league member),
gen6's settings -- mix 2:15,3:20,4:35,5:20,6:10, caps 1000/1100/1200/1300/1500, 32 workers x 96 games, 32 rows
per seat, 3 opponent slots per count, PFSP power 2, slot refresh 2, own snapshots every 50 (4 kept), placement,
anchor every 50, 65,536 rows, minibatch 2,048 / microbatch 1,024, lr 2e-5 / critic 3e-5, KL 0.02 -> target 0.08
(floor 0.005), T 0.5 / 0.5, auction 0.5 real rules (stall 150), mid-game 0.3, random 0.2, 4,000 updates,
snapshot every 10, every 20th game saved -- plus a learner server with 2 replicas (the collector thread,
`--server-pipeline`, is left off: the legacy server's was slower on the GPU; measure it with the profile). Departures
named in the script: the start (generic SL, not gen5), the league (the SL alone + own snapshots: no legacy members
possible), the learner's replicas. `pg_generic_1830.sh evals` runs one round of policy-only evals of the newest
snapshot (T 0.5, real-rules auctions, 400 games, generic players on the GPU): vs the generic SL (4p 2v2, 1v3, 2p)
and the outside panel (SL ckpt 10 4p 2v2 / 1v3; gen6 u2400 at 2-6 players). Expected wall clock at ~80k games/h:
~1,200 games per update (65,536 rows / ~54 rows per game: 32 per learner seat, ~1.7 learner seats per game over
the mix) -> ~55 s of generation per update, plus the update itself (the legacy's, + ~10 s of collate); gen6 ran
~60 s per update, so expect ~80-90 s: ~1,000 updates a day.

### 1867 outline (status after A3)

1. **Layout and import**: done. A2's layout `1867-v1` and index API (`rl_legal_moves_json`, `rl_apply_index`,
   `rl_price_range`); `TitlePlay` plays a generator layout through them (A3: before, `apply_action_index` /
   `price_range_for_index` refused 1867, so no 1867 game could be played); the importer
   `generic_data.register_importer("1867", ...)` = `generic_import_1867.Importer1867` (A3, below). Nothing else in
   `generic_pg` / `generic_inference` is title-specific.
2. **Start files**: built (A3, below) with `scripts/build_generic_start_positions.py --title 1867 --source
   human_games/1867 --source human_games/scrape_2026_10/1867 --out-dir human_games --jobs 8` ->
   `start_positions_1867_{3..6}p.jsonl` / `midgame_positions_1867_{3..6}p.jsonl` (the `TitleSpec` templates point
   there).
3. **Supervised**: `convert-generic --title 1867 --data-dir human_games/1867 human_games/scrape_2026_10/1867`
   (finished games only: 840 games, 537,362 positions), `pretrain-generic` (default preset on the GPU), the value
   refit; no legacy model to compare with, so judge by per-family top-1 against chance and the win CE against equal
   odds (A3's CPU smoke: below).
4. **Policy gradient**: `policy-gradient-generic --title 1867 --policy <1867 final.pth>` with the 1830 recipe at
   3-6 players (`--player-mix 3:25,4:40,5:25,6:10`, no 2-player game), `--auction-fraction 0.2 --midgame-fraction
   0.3 --random-start-fraction 0.2` (design §5.6: ~80% from human SR1 starts), placement reward, the decision caps
   and auction-stall setting proposed below ("Decision caps and auction stalls"). 1867's auction cannot lock, so no
   `auction_unlock` (an 1830 variant, ignored for 1867).
5. **Evals**: `scripts/eval_generic.py --title 1867` (generic vs generic: the SL start, older snapshots); an
   out-of-family panel has to be built from 1867 checkpoints of other runs (no legacy 1867 bots exist).

### Open questions after C2

1. **Placement points at 3, 5 and 6 players** are 1830's gen6 tables (`PLACEMENT_POINTS_BY_PLAYERS`: 3p
   1/.4/0, 5p 1/.5/.25/.1/0, 6p 1/.5/.3/.15/.05/0); the design (§5.4) suggested `(N-1-rank)/(N-1)`. Keep the
   tables for 1867?
2. **League for a generic 1830 run**: generic SL + own snapshots only (legacy bots can't be members). Worth
   building kind-split slots so gen4 .. gen6 and the exploiters can be opponents (see above), or is the outside
   panel enough?
3. **Model size for PG**: the default generic model (~8.6M) costs ~1.5x the legacy per position; a
   `fusion_layers=1` model would be faster but needs its own supervised run.
4. **`--server-pipeline` on the GPU**: faster on the CPU transport benchmark (52k vs 36k/s), but the legacy
   collector thread was slower on the GPU (GIL); measure with the profile on the first GPU run.
5. **1867 decision caps and stall policy**: measured and proposed in A3 (below, "Decision caps and auction
   stalls"); the user decides.

### Notes on the design documents (found during C2)

* `docs/rl_bridge_design.md` §0, §1a and §4.4 still say `MAX_SEATS = 8` (superseded by the 2026-11-13 decision:
  6), and §1a's table cites §4.4 for the seat width, which is §4.5.
* §5.4 suggests placement points `(N-1-rank)/(N-1)`; 1830's runs use per-count tables (open question 1 above).
* §5.3 lists the acting seat among the native getters the loop needs; the generic workers read it from the
  encoder's `meta[:, 5]` (one native call encodes every pending position and routes it).
* `docs/rl_bridge_work_packages.md` says to start each package from branch `gen6-mixed-count`; master has it all.

## 1867: data, starts, caps and the CPU smoke (milestone A3, 2026-11-13)

Code: `generic_import_1867.py` (the importer and the matching, shared with `scripts/g1867_completeness.py`),
`generic_data.py` (importer hooks), `generic_play.py` (`TitlePlay` on a generator layout), the price-grid fix in
`model_generic.LegalActions.price_cells` and its callers. Tests: `tests/agent/alphazero/test_generic_1867.py` (21,
~75 s on the CPU).

### The importer (`generic_import_1867.Importer1867`)

* **Input**: raw 18xx.games exports (`human_games/1867`, `human_games/scrape_2026_10/1867`), replayed as the corpus
  audits replay them (`replay_export`: Ruby's filtered actions, each recorded auto-action processed right after
  its parent, seats renumbered 1..N as "Player k" -- no names kept -- and the engine seed of `settings.seed` / the
  game id). The record's log is the engine's own (`raw_actions`, one entry per processed action).
* **Scope** (A2's completeness scope): finished or archived games, 3-6 players, base rules. Dropped and counted:
  `players_<n>` (2-player games), `grid_market`, `optional_rules` (any other rule), `status_<s>` (abandoned
  `active` / `new` games), `no_actions`.
* **Decisions**: every logged action except recorded auto-actions (programmed passes, the BuyCompanyPreloan
  auto-pass: `not_decision.auto`) and actions after the game's end.
* **Labels**: the human action is matched against the **rules'** indexed moves (`rl_legal_moves_json("rules")`)
  with `matching_indexed` -- type, entity and the outcome-relevant parameters, ranged prices by range *and step*,
  `run_routes` by type (the RunRoutes index is the router's optimum; the record replays the human's own routes).
  Exactly one hit is the label; none is `unmatched` (or `replay_only:<class>` for the documented replay-only
  classes: `pre_9655_merger_order`, `trainless_out_of_order`, `dealing_out_of_turn`), several `ambiguous`. The
  legal list stored with the position is the **title's cross-player policy**'s (`model_generic.legal_actions`, what
  self-play and the evals are offered), so a human move the policy excludes is skipped as `policy`.
* **Prices**: the chosen row's human price with its `(lo, hi, step)` (step units: `docs/rl_descriptors.md` §6),
  checked on the grid.
* **Outcomes**: `result()` in seating order -> win share (ties split) and placement points
  (`policy_gradient.default_placement_points` of the player count), as 1830's records.
* **Unfinished games are dropped** (the user's decision, "just for correctness": `convert-generic
  --finished-only`, 1867's default through `titles.TitleSpec.supervised_finished_only`; `--keep-unfinished`
  turns it off, 1830's default is off so its conversions and the 1830 GPU check are unchanged). A game is kept only
  when its log reaches the game's end in our engine; dropped and counted by reason: `unfinished_replay_rejected`
  (an old record the current rules refuse part-way) and `unfinished_record_ends` (the record stops before the end:
  a manual end on the site, an archived or abandoned game). `Importer1867.unfinished_reason` decides right after
  the replay, before any matching; for other importers `convert_game` checks `finished` at the end. Without the
  option a rejected game is kept up to the rejection (`finished` False, `stopped` = the action and the engine's
  message) with net worth there as its value target, as 1830's unfinished games.

### The dataset (both corpora, one conversion)

```bash
CUDA_VISIBLE_DEVICES= taskset -c 24-31,56-63 nice -19 uv run python main.py convert-generic --title 1867 \
    --data-dir human_games/1867 human_games/scrape_2026_10/1867 --output <dir> --workers 8      # ~12-30 min
# finished games only by default for 1867; --keep-unfinished converts every in-scope game (below: "All games")
```

**The 1867 dataset (finished games only, the default):** 1,394 files -> **840 games, 537,362 positions** (train
794 games / 508,051 positions, validation 46 / 29,311; 38 MB): `human_games/1867` 565 games / 362,065 positions,
the 2026-10 scrape 275 / 175,297. Dropped: `unfinished_record_ends` 101 (3p 33, 4p 51, 5p 16, 6p 1),
`unfinished_replay_rejected` 24 (3p 4, 4p 15, 5p 4, 6p 1) -- 125 games, 34,351 positions of the all-games
conversion --, plus the scope drops below (165 `grid_market`, 103 `players_2`, 79 `status_active`, 4 `status_new`,
78 `no_actions`); every dropped game is in `dropped.jsonl`. The kept games' records equal the all-games
conversion's.

| players | games (train / validation) | positions (train / validation) |
|---|---|---|
| 3 | 205 (191 / 14) | 125,540 (117,052 / 8,488) |
| 4 | 509 (487 / 22) | 326,325 (312,587 / 13,738) |
| 5 | 109 (100 / 9) | 73,209 (66,889 / 6,320) |
| 6 | 17 (16 / 1) | 12,288 (11,523 / 765) |

Skipped labels: `policy` 36, `replay_only` 798 (pre-#9655 merger order 623, trainless out of order 167, dealing
out of turn 8), `unmatched` 13 (place_token 10, buy_shares 3: the 3 unexplained lay_tile misses were in rejected
games); forced positions 108,035; price targets 40,968.

**All games** (`--keep-unfinished`, the first A3 conversion; the numbers the A2 comparison below uses): 1,394 files
-> **965 games, 571,713 positions** (train 915 games / 540,775 positions, validation 50 / 30,938;
`in_validation_split(game id, 0.05)`; 40 MB): `human_games/1867` 664 games / 389,776 positions, the 2026-10 scrape
301 / 181,937. Dropped: 165 `grid_market`, 103 `players_2`, 79 `status_active`, 4 `status_new`, 78 `no_actions`
(in A2's scope but without a game action: 50 files with none, 28 with only `end_game` / messages / programs). 840
games end by the rules, 101 end early on the site (a manual end: `result()` as it stands), 24 old records stop at
a rejected action. `meta.json` holds the counts by source and player count; `games.jsonl` each game's source and
skipped labels.

| players | games | positions (train / validation) | human decisions per game (median, p99, max) |
|---|---|---|---|
| 3 | 242 | 123,822 / 8,659 | 488, 624, 656 |
| 4 | 575 | 333,572 / 14,245 | 503, 628, 707 |
| 5 | 129 | 71,281 / 7,269 | 512, 622, 658 |
| 6 | 19 | 12,100 / 765 | 546, 627, 631 |

("Decisions" as the PG workers count them: several legal rows, or one with an open price; positions include the
forced ones, 20.1%.) Legal lists: mean 19.4 rows, median 3, p90 52, p99 206, max 336 (LayTile positions mean 72,
max 336; 1830's corpus: mean 11.5, max 185). Price targets (open ranges): 45,051. Matched positions by family:
Pass 231,767; LayTile 86,844; RunRoutes 72,535; Dividend 36,248; BuyShares 35,808; BuyTrain.depot 27,005;
PlaceToken 19,306; Bid.corporation 18,148; Bid.company 16,871; Merge 7,915; BuyTrain.corp 6,792; BuyCompany
4,074; SellShares 2,358; Convert 1,860; BuyTrain.exchange 1,456; Par 1,055; RedeemShares 724; RemoveToken 379;
BuyTrain.pool 273; Choose 218; DiscardTrain 77.

Skipped labels (never guessed): `policy` 38 (merge 30, buy_train 5, buy_company 3), `replay_only` 865
(pre-#9655 merger order 686: buy_shares 428 + pass 258; trainless out of order 170: pass 107 + choose 63;
dealing out of turn 9), `unmatched` 17 (place_token 11, lay_tile 3, buy_shares 3), `ambiguous` 0, `not_legal` 0;
30,098 recorded auto-actions applied as non-decisions.

**Against A2's index completeness** (`docs/rl_descriptors.md` §8), decision for decision: A2's exactly-one-index
decisions 181,944 (scrape) + 389,807 (old) = 571,751, minus the 7 + 31 the title's policy excludes = **571,713 =
the dataset's positions**; replay-only 3 + 862 = 865; A2's 40 old-corpus misses = these 17 + the 23 actions the
replay stops at (not decisions here: the log ends before them). The test
`test_counts_equal_index_completeness` pins the equality per game on the Ruby fixtures.

### Start files

```bash
CUDA_VISIBLE_DEVICES= taskset -c 24-31,56-63 nice -19 uv run python scripts/build_generic_start_positions.py \
    --title 1867 --source human_games/1867 --source human_games/scrape_2026_10/1867 --out-dir human_games --jobs 8
```

(~6 min.) Starts use no outcome, so the builder's scope is wider than the importer's: abandoned (`active`) games
and records that end early give starts too. A game the engine's replay rejects anywhere gives none (C2's rule; a
first A3 build had taken the Stock Rounds before the rejection and was rebuilt after the user's "for
correctness" call: 24 games, 24 SR1 starts and 24 mid-game games fewer). Excluded: 172 optional rules, 106 player
count, 80 without actions, 46 without a Stock Round, 24 replay failed, 4 `new`.

| players | SR1 starts | mid-game games | mid-game cuts SR2 / SR3 / SR4 / SR5 / SR6 |
|---|---|---|---|
| 3 | 242 | 233 | 233 / 227 / 220 / 217 / 208 |
| 4 | 575 | 567 | 567 / 564 / 554 / 537 / 514 |
| 5 | 124 | 121 | 121 / 118 / 116 / 114 / 112 |
| 6 | 21 | 19 | 19 / 19 / 19 / 19 / 19 |

Every line was replayed again: each SR1 start and each cut ends at an unfinished Stock Round; starts of every
kind (human, mid-game, auction, random) sample and replay through `generic_play.replayed_start`. 6 players is
thin (21 games): the 1830 recipe's random openings (`--random-start-fraction`) and auction starts carry most 6p
games. (The CPU smoke below ran on the first build, 24 starts more.)

### Ruby HEAD check of A2's 17 unexplained old-corpus misses

Ruby HEAD 849c522 in docker (`scripts/g1867_ruby_probe.rb`): each game processed to the file action before the
miss, then the recorded action processed **as a live player** (`@loading` off, so Ruby's player-time checks run).
None is a generator bug:

* **11 are illegal at Ruby HEAD in the same state** (same round, step and entity): 9 home tokens a minor or major
  may no longer place there -- Toronto F16 (28418@641, 39631@631, 42964@581, 48185@477, 67332@710), Montreal L12
  (21494@776, 29292@510, 81411@536) and N12 (31586@669): "Cannot place token on <hex> as the hex is not available"
  (Ruby's `home_token_locations` changed since these games); 2 tile lays on M11 with rotations Ruby HEAD refuses
  (24437@899 "M11 is not legally rotated for 8", 45921@503 "9 is not legally rotated for 26").
* **6 lie beyond Ruby HEAD's replay horizon**: Ruby's replay of the same game stops earlier -- on the pre-#9655
  merger order in 32894, 42154, 41062 and 72923 (place_token 2, buy_shares 1, lay_tile 1), or on the refused home
  token above in 39631 (its buy_shares @693 / @702 come after 39631@631).

Our record replay accepts these old records (the SR home-token handler only checks for a free slot; tile rotations
are a player-time check), so the games replay on; the generator, like a live player at Ruby HEAD, does not offer
the moves.

### Legal prices: the layout's grid on every path

A2 put 1867's legal prices on the move's step grid and its cells in step units (`price_grid`), but four places
rebuilt cells without the grid -- in price space, where a $5-step range's bins hold every integer, so a sampled
price could be off the grid (`rl_apply_index` refuses it: "price 27 is outside the move's range [25, 315] step 5")
and the trainer's cell probabilities differed from the server's:

| where | before | now |
|---|---|---|
| `generic_policy.sample_price` (`eval_policy_only.py` generic players) | `price_range_for_index` (raises on 1867), cells without the grid | `legal.price_cells(k)`; no-price moves through `TitlePlay.fixed_price` |
| `generic_pg.play_generic_pg_games` (PG worker) | `price_cells_for(kind, lo, hi, step)` | `legal.price_cells(k)` |
| `scripts/eval_generic.py` | same | same fix |
| `generic_pg.collate_rows` (trainer) | `LegalActions` without the grid, so 1867 cell masks and log-probabilities in price space | `PGRow.price_grid` carried from the worker |
| `generic_data.collate` (supervised) | same: 1867 price targets in price-space cells | items carry `price_grid` (`ActionDescriptors.price_grid`) |

`LegalActions.price_cells(k)` (`model_generic`) is now the one place a legal row's cells are built for sampling
or scoring. 1830 is unchanged (`price_grid` False: the legacy cells, same draws; tested). Also fixed: `TitlePlay`
applied and priced moves through `apply_action_index` / `price_range_for_index`, which refuse a generator layout,
so no 1867 game could be played by the generic pipeline (PG workers, `eval_generic.py`) before A3; a generator
layout now goes through `rl_apply_index` / `rl_price_range`. Tests (`test_generic_1867.py`): prices sampled from a
random small model on up to 30 positions per priced block (all four, random walks at 3-6 players), through the PG worker's path
(server cell + `price_cells(k).sample_price`) and the policy player's (`sample_price`, eps 0.2), every one on the
grid inside its range and applied by `rl_apply_index`; without the grid the same ranges' cells hold off-grid
prices; the trainer recomputes the sampling log-probabilities of 1867 PG rows with price cells exactly
(`log_ratio_abs_mean` < 1e-4).

### CPU smoke (2026-11-13, `CUDA_VISIBLE_DEVICES=`, cores 24-31,56-63 shared with other jobs at load ~30)

**Supervised.** `pretrain-generic --model-preset small` (151,742 parameters), batch 64, 3 loader workers, shuffle
buffer 8,192, all 915 training games of the all-games conversion, validation on the first 20 validation games
(12,285 positions; chance top-1 0.411):

| run | steps (positions) | stage-1 time | train loss (first -> last 50) | val top-1 / top-5 | val policy CE | val price NLL | val win CE (equal odds 1.388) / winner |
|---|---|---|---|---|---|---|---|
| 600 steps | 600 (38k) | 2.5 min | 1.668 -> 1.301 | 0.691 / 0.917 | 1.008 | 1.458 | 0.932 / 0.587 |
| 4k steps | 4,000 (256k, ~0.5 epoch) | 15 min (cores shared) | 1.668 -> 1.154 | 0.729 / 0.928 | 0.892 | 1.421 | 0.807 / 0.614 |

Validation top-1 by player count (4k steps; chance in brackets): 3p 0.711 (0.380), 4p 0.744 (0.398), 5p 0.736
(0.453), 6p 0.710 (0.434).

Per family on validation (top-1 vs chance, mean 1 / legal count):

| family | n | chance | 600 steps | 4k steps |
|---|---|---|---|---|
| Pass | 5,122 | 0.459 | 0.905 | 0.942 |
| LayTile | 1,824 | 0.050 | 0.070 | 0.225 |
| RunRoutes | 1,552 | 0.927 | 1.000 | 1.000 |
| Dividend | 798 | 0.326 | 0.885 | 0.885 |
| BuyShares | 680 | 0.173 | 0.591 | 0.616 |
| BuyTrain.depot | 552 | 0.451 | 0.993 | 0.989 |
| Bid.company | 412 | 0.500 | 0.869 | 0.643 |
| PlaceToken | 405 | 0.153 | 0.131 | 0.200 |
| Bid.corporation | 366 | 0.289 | 0.191 | 0.270 |
| Merge | 177 | 0.285 | 0.130 | 0.153 |
| BuyTrain.corp | 116 | 0.346 | 0 | 0 |
| BuyCompany | 83 | 0.384 | 0 | 0 |
| SellShares | 54 | 0.134 | 0 | 0 |
| BuyTrain.exchange | 45 | 0.184 | 0 | 0.289 |
| Convert | 29 | 0.452 | 0 | 0.172 |
| Par | 23 | 0.033 | 0 | 0 |
| RedeemShares | 21 | 0.452 | 0 | 0 |
| RemoveToken / Choose / BuyTrain.pool / DiscardTrain | 12 / 7 / 5 / 2 | | | |

The supervised path works on 1867: losses fall, the value head learns (win CE 1.388 -> 0.81, the refit 0.844),
and every family with at least 400 validation positions (eight of them) plus BuyTrain.exchange is above chance
after 4k steps. A 150k-parameter model after half an epoch does not yet beat chance on Bid.corporation (0.270 vs
0.289), Merge, Convert and the rare families: there it picks the state's common alternative (mostly Pass; Par's
chance is 0.033 with ~43 legal pars). That is the GPU run's job (default preset, 10 epochs); judge it by this
table.

Stage 2 (value refit, 400 steps): val win CE 0.942 (600 steps) / 0.844 (4k), winner accuracy 0.586 / 0.605.

**Policy-only games** (`scripts/eval_generic.py --title 1867 --auction-start --max-decisions 8000
--auction-stall-decisions 0`, the model against itself, every game from the opening auction, in-process CPU
inference). Every game finished (none hit a cap), and self-play scores sit at the fair share within noise:

| model, T | players (seats) | games | wall | decisions per game: median / p90 / p99 / max | auction decisions: median / max | score (fair) |
|---|---|---|---|---|---|---|
| 600 steps, 0.5 | 3 (1v2) | 24 | 51 s | 570 / 636 / 665 / 670 | - | 0.375 +- 0.101 (0.33) |
| 600 steps, 0.5 | 4 (2v2) | 24 | 63 s | 566 / 608 / 654 / 661 | - | 0.458 +- 0.104 (0.50) |
| 600 steps, 0.5 | 5 (2v3) | 30 | 102 s | 562 / 613 / 699 / 723 | - | 0.300 +- 0.085 (0.40) |
| 600 steps, 0.5 | 6 (3v3) | 40 | 82 s | 570 / 665 / 710 / 724 | - | 0.550 +- 0.080 (0.50) |
| 600 steps, 1.0 | 3 | 24 | 201 s | 600 / 707 / 775 / 783 | 34 / 49 | 0.250 +- 0.090 |
| 600 steps, 1.0 | 4 | 24 | 199 s | 630 / 731 / 783 / 794 | 38 / 52 | 0.542 +- 0.104 |
| 600 steps, 1.0 | 5 | 30 | 335 s | 698 / 794 / 873 / 892 | 46 / 58 | 0.300 +- 0.085 |
| 600 steps, 1.0 | 6 | 40 | 819 s | 746 / 866 / 920 / 925 | 57 / 69 | 0.500 +- 0.080 |
| 4k steps, 0.5 | 3 (1v2) | 48 | 348 s | 615 / 683 / 744 / 780 | 27 / 37 | 0.354 +- 0.070 (0.33) |
| 4k steps, 0.5 | 4 (2v2) | 48 | 322 s | 568 / 696 / 745 / 751 | 30 / 40 | 0.479 +- 0.073 (0.50) |
| 4k steps, 0.5 | 5 (2v3) | 50 | 396 s | 632 / 709 / 760 / 765 | 37 / 43 | 0.340 +- 0.068 (0.40) |
| 4k steps, 0.5 | 6 (3v3) | 60 | 337 s | 618 / 678 / 696 / 698 | 44 / 50 | 0.533 +- 0.065 (0.50) |
(Wall times are on cores shared with other jobs; the T 1.0 batch ran beside a pretraining and the PG smoke.)

**Policy gradient** (`main.py policy-gradient-generic --title 1867`): the 4k-step model as policy, critic and
league member, the 600-step model as the second member (`min_share` 0.2 at 4 players); mix 3:1,4:2,5:1,6:1 (one
opponent slot per count), placement reward, T 0.5 / 0.5, KL 0.02 -> target 0.08, anchor and own snapshots every 2
updates (2 kept), starts auction 0.3 / mid-game 0.3 / random 0.2 / else human SR1 (the files above),
`--auction-stall-decisions 0`, `--max-decisions 4000` (to measure, not to cap), 8 workers x 4 games, 16 rows per
seat, 2,000 rows per update, minibatch 512 / microbatch 256, lr 2e-5 / critic 3e-5, CPU servers (1 thread each):

```bash
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 taskset -c 24-31,56-63 nice -19 .venv/bin/python main.py policy-gradient-generic \
  --policy <4k>/final.pth --title 1867 --out-dir <out> --run-name smoke1 --player-mix 3:1,4:2,5:1,6:1 \
  --max-decisions 4000 --workers 8 --games-per-task 4 --positions-per-seat 16 --population <league.json> \
  --opponent-slots 1 --slot-refresh-every 1 --league-snapshot-every 2 --league-max-snapshots 2 --reward placement \
  --anchor-every 2 --rows-per-update 2000 --minibatch 512 --microbatch 256 --lr 2e-5 --critic-lr 3e-5 --kl-coef 0.02 \
  --kl-target 0.08 --learner-temperature 0.5 --opponent-temperature 0.5 --max-updates 3 --snapshot-every 2 \
  --random-start-fraction 0.2 --midgame-fraction 0.3 --auction-fraction 0.3 --auction-stall-decisions 0 \
  --save-game-every 10 --start-positions <starts>/start_positions_1867_{n}p.jsonl \
  --midgame-positions <starts>/midgame_positions_1867_{n}p.jsonl --server-cpu --server-threads 1 --threads 4 \
  --inference-batch-size 256
# then: ... policy-gradient-generic --resume <out>/smoke1 --max-updates 5
```

* 3 updates, then a **resume** to 5. The resume continues from the last snapshot (`--snapshot-every 2`: update 2),
  so update 3 ran twice (the legacy semantics; both are in `history.jsonl`). 416 games, **all finished** (no
  decision cap, no stall, no skipped start); starts human 180, auction 113, mid-game 85, random 38; 52 saved games.
  Mean 507-546 decisions per game, the longest 801 (6 players); at most 57 auction decisions.
* Losses finite; at each update's first minibatch `approx_kl` within 8e-4 and `log_ratio_abs_mean` 0.001-0.004 (the
  trainer recomputes the servers' sampling probabilities, price cells included: the rows carry the grid); KL to
  the anchor <= 6e-5 at lr 2e-5 (`kl_coef` adapted to its floor); critic CE 0.83-1.09 over 3-6 seats; the league
  grew to 4 members (own snapshots per count).
* **Throughput** (cores shared with other jobs): 620-830 games/h (90-120 decisions/s) for updates 1-3, then
  860-1,240 games/h (124-186 decisions/s) after applying moves from their action dicts (`LegalActions.wires`:
  workers' play time per update 1,060-1,250 s -> 590-680 s). 1830's C2 smoke made 8.5-11.7k games/h on the same
  setup. The difference is the rules side: on human positions the 1867 generator takes ~8 ms per decision (median
  2.4 ms, p99 93 ms; tile-lay positions 12 ms, the router behind RunRoutes ~6 ms) against ~0.03 ms for 1830's
  enumeration, and `rl_apply_index` re-enumerated the moves (another ~7 ms) until the wire fix. On the GPU, 1867
  policy gradient will be bound by the workers' CPU: at ~9 ms per decision, 32 workers give ~3.5k decisions/s,
  **~20-25k games/h** (gen6 made ~125k 1830 games/h) unless the generator gets faster
  (`docs/g1867_remaining_work.md` "Generator speed": cache the graph walk per state).
* **Update 2026-11-13 (generator speed-up, `docs/g1867_remaining_work.md` "Generator speed")**: the same moves
  (differential tests: 0 differences on every human decision and 80 fuzz walks) for ~5x less CPU — human
  decisions 6.5 -> 1.05 ms mean (p50 1.8 -> 0.5, p99 58 -> 7.9; thread CPU, same positions). `TitlePlay.fixed_price`
  also ran the generator a second time for every position the PG worker played (`rl_price_range` re-enumerates);
  it now reads the legal list's ranges. Worker calls per position on whole games: 8.5 -> 1.2 ms. Expected with
  GPU servers: ~90-100k games/h on 32 workers (the server becomes the limit); the router (`run_routes`) is now the
  largest rules cost.

### Decision caps and auction stalls (proposal, for the user to decide)

How long 1867 games are, in decisions (several legal rows or an open price), every game played to its end:

| source | 3p max | 4p max | 5p max | 6p max | medians |
|---|---|---|---|---|---|
| human games (finished, 965) | 656 | 707 | 658 | 631 | 488-546 (p99 622-628) |
| smoke models T 0.5, from the auction (600 steps; 4k steps) | 670; 780 | 661; 751 | 723; 765 | 724; 698 | 562-570; 568-632 |
| smoke model T 1.0, from the auction | 783 | 794 | 892 | 925 | 600-746 |
| PG smoke (416 games, mixed starts, T 0.5) | 738 | 762 | 749 | 801 | means 512-531 |
| 1830 gen6 caps, for scale | 1100 | 1200 | 1300 | 1500 | |

So 1867 games are not longer than 1830's (the C2 outline expected longer games from mergers and loans; humans
average ~500 decisions, the smoke models 560-630): their length grows with randomness and player count, not with the
rules' extras. **Proposal: `--max-decisions-by-players 3:1200,4:1200,5:1400,6:1500`** (~1.5x the longest T 1.0
game, ~1.7-2.4x the human maximum), i.e. 1830's gen6 caps with 5 players one notch up; watch `p<n>_max_length` in
the PG history -- a cap that fires on more than a few games per update means a degenerate policy (1830's bots
learned train shuffling), not a short cap.

**Auction stalls cannot happen in 1867.** The opening auction is Ruby's `SingleItemAuction`: one lot at a time
(cheapest first), ascending bids (each at least $5 above the last, up to the bidder's cash), and when every player
declines, a dutch phase that drops the lot $5 per all-pass round (`company.discount += 5`) down to $0, where the
first player is made to buy it (the engine's documented fix of Ruby's crash there). Every lot therefore ends after at
most ~(opening bid / 5 + 1) all-pass rounds plus the bids the players' cash allows (opening bids $20-$60, cash
$210-$420), and the hidden "3" company is never auctioned. 1830's lock (everyone passing forever on a non-SV
private) has no counterpart; `auction_unlock` is 1830's. Measured over 442 eval and 416 PG games: at most 69
auction decisions (600 steps at T 1.0, 6 players), medians 27-57. The SR minor auctions (`Bid.corporation`) are Stock-Round bids bounded by cash the same way.
**Proposal: `--auction-stall-decisions 0` (off) for 1867**; the decision cap is the backstop (or 300 if the user
wants a stall line in the history anyway: four times the measured maximum).

### Open questions after A3 (the user's decisions)

1. **Decision caps and the stall setting** for 1867 runs: the proposal above (`3:1200,4:1200,5:1400,6:1500`,
   `--auction-stall-decisions 0`).
2. **Legal lists of the supervised data** are the title's cross-player policy's (what self-play is offered); the
   38 human moves it excludes are skipped. The alternative is the rules' lists (then the model also scores moves
   it will never be offered).
3. **Unfinished human games**: decided (the user, 2026-11-13, "just for correctness") -- 1867's supervised data
   keeps finished games only (`--finished-only`, 1867's default; above). 1830's conversions keep theirs (C1's open
   question 2 stands for 1830).
4. **6 players is thin**: 17 finished games / 12,288 positions, 21 SR1 starts (3p 205 games, 4p 509, 5p 109). A
   6-player share in the PG mix leans on random openings and auction starts.
5. **1867 generation speed**: the generator, not the network, will bound 1867 policy gradient on the GPU (~20-25k
   games/h estimated vs gen6's ~125k for 1830); worth the graph-walk caching of `g1867_remaining_work.md` item 2
   before a long run?
6. **Placement points at 3, 5 and 6 players** (C2's open question 1) apply to 1867 as they are.
7. The start files and the dataset live in the A3 scratch area until copied to `human_games/` (the `TitleSpec`
   templates `human_games/{start,midgame}_positions_1867_{n}p.jsonl`).

### Notes on the design documents (found during A3)

* C1's text said 1867's importer would come from an engine-side `match_human_action`; there is none (the matching
  is Python's, over `rl_legal_moves_json`, as A2's completeness).
* C2's text said `TitlePlay` reads a title's layout through `apply_action_index` / `price_range_for_index`; both
  refuse a generator layout, so 1867 games could not be played (fixed: `rl_apply_index` / `rl_price_range`, then the
  action dicts).
* `docs/rl_descriptors.md` §6 said code that samples prices must pass the grid; so must code that rebuilds cells to
  *score* them (`generic_data.collate`, `generic_pg.collate_rows`), or 1867's supervised price targets and the
  trainer's PPO ratio for price decisions are computed in the wrong cells (fixed).
* §8 counted 25 old-corpus games that end in a replay rejection; there are 24 (`replay_export` over the corpus).
* The C2 outline expected 1867 games to be longer than 1830's (mergers, loans): they are not (~500-570 decisions
  per game; 1830's PG games averaged ~450).

## Mixed-title training and the sharing probe (milestone C3, 2026-11-13)

Code: `generic_data.py` (`DataSource`, `MixedPositionBatches`, split overrides, `nested_subset`), `generic_pretrain.py`
(several datasets, per-dataset metrics and selection, `load_init`), `main.py pretrain-generic`; the probe harness
`scripts/sharing_probe.py`. Tests: `tests/agent/alphazero/test_generic_mixed.py` (9, ~25 s on CPU; 1830 fixture games
and synthetic 1867 games -- random walks over 1867's layout labelled by a test importer).

### Several datasets in one run (`pretrain-generic --data A --data B`)

* **Datasets and labels.** `--data PATH` or `--data LABEL=PATH`, repeatable; the label (default: the dataset's
  title) names the dataset in every per-dataset option and metric. One dataset without `--mix` runs C1's loop
  unchanged (`--loader epochs`: a pass per epoch).
* **Mix** (`--mix 1830:0.5,1867:0.5`): each item of a batch comes from dataset k with probability share_k (a mix by
  positions; default: the datasets' training positions, i.e. concatenated). Share 0 = evaluated, never trained on.
  Batches mix titles (`collate` pads every title's encoding to the batch maxima: tested with 1830 + 1867 items in
  one batch, forward / backward, probabilities summing to 1 per position).
* **The stream loader** (`MixedPositionBatches`, `--loader stream`, automatic with several datasets or a mix):
  endless and step-based. Per dataset, each loader worker reads every W-th game of that dataset's order (permuted per
  seed, label and pass) into its own shuffle buffer (capped at `--buffer-positions` and at the worker's share of the
  dataset's positions, so a small dataset's buffer is its whole shard); a dataset that runs out starts its next pass
  in a new order, so a 100-game target cycles while 1830 is still in its first pass. Batches carry `source`.
  An "epoch" is nominal here: the training positions of the datasets with a share, `/ batch`; `--max-steps` sets the
  run's length and the cosine schedule.
* **Split overrides** (`--splits LABEL:FILE`, JSON `{split: [game ids]}`): replace a dataset's train / validation
  assignment; any split name (e.g. `dev`, carved out of the training games); a game in no split is in none; ids
  must exist and be in one split only. `--train-max-games` / `--val-max-games` / `--report-max-games` take `N` or
  `LABEL:N,...`.
* **Evaluation and selection.** `--eval-titles` (default all) are evaluated on `--select-split` (default
  `validation`) every `--eval-every` steps, each logged separately (history `{"kind": "eval", "source": label}`,
  TensorBoard `val_<label>/*`: policy CE, top-1 / top-5, top-1 per family and per player count, price NLL, win CE vs
  the equal-odds baseline, winner accuracy). `--select-title` picks the dataset whose total loss selects
  `best_joint.pth` and drives early stopping and the value refit's early stopping (default: the one dataset, else
  the mix-weighted mean over the evaluated ones). Training windows log policy CE / top-1 / share per dataset.
  The value refit (stage 2) trains on the run's mixed stream. `--final-eval` scores `final.pth` on every dataset's
  `--report-split` (`summary.json["final_eval"]`).
* **`--init <checkpoint>`** (fine-tuning: the checkpoint's weights, a fresh optimizer and schedule) goes through
  `load_init`: B2's `load_checkpoint` refusals (architecture, encoder version, seats, price cells, vocabularies
  append-only, layout fingerprints vs the engine), plus a requested model config that differs from the checkpoint's
  (with `--init` the run keeps the checkpoint's architecture; a preset is optional and must match), a dataset title
  beyond the checkpoint's title rows, and a dataset layout that differs from the checkpoint's for the same title.
  The run's checkpoints name the init's titles plus the datasets' (`extra["init"]` records where it started).

Two model changes came with this (both keep every output; tests in `test_model_generic.py`): the hex distance /
direction attention bias is computed once per title in the batch and gathered (`_static_hex_bias`: bit-identical
values; the embeddings' dense backward over `M x H x H` indices was ~30% of a CPU training step, so the small model
trains ~1.4x faster on the CPU), and a model with fewer title rows than the engine knows (a checkpoint trained before
a title was registered) can be built again (the per-title parameter scale table indexed past its rows).

### The sharing probe (`scripts/sharing_probe.py`)

The question (design §4.2.6, scaled to what runs on the CPU while the GPU is elsewhere): at small model scale, does
training on 1830 as well help an 1867 model, against 1867 alone? Arms, at each 1867 training-set size and seed:

* **a**: 1867 alone;
* **b**: joint, every batch 50% 1867 / 50% 1830 positions;
* **c**: 1830 alone first (`c_pre`, one run per seed, shared by the sizes), then 1867 alone from its best checkpoint.

Data: 1867 = A3's `generic_1867` (converted at A3's 2a758f2; 965 games of 3-6 players, 915 training / 50
validation, 540,775 / 30,938 positions; the same games as its earlier `generic_1867_v1`); 30 of its training games are a fixed
**dev** split (18,081 positions: early stopping and the value refit's selection); the other 885 form the pool from
which the sizes are nested subsets (`nested_subset`, seed 0): **100, 300 and all 885 games** (55,954 / 175,887 /
522,694 positions). Results are on 1867's **validation** split (all 50 games, 30,938 positions, never used for any
choice). 1830 = `generic_1830_clean_all_v1` (3,878 training games, 2.38M positions, 2-6 players; 30 validation games
for the interference check).

Recipe (every arm the same; `sharing_probe.py plan` defaults): the probe model `PROBE_MODEL` (d_model 64, 4 heads,
2 map layers, 1 entity layer, 1 fusion round, d_trunk 128, d_head 64: **538,550 parameters**, 1/16 of the default),
fp32 on the CPU, batch 128, C1's losses and AdamW at lr 1e-3 with cosine decay over the step cap. **Equal compute
per arm**: every arm's 1867 phase is capped at **5,000 steps** (640k positions; arm b's are half 1830, so it sees
at most 320k 1867 positions); `c_pre` runs 2,500 steps (320k 1830 positions = arm b's 1830 share at the cap),
selected on 1830 validation, no value refit. Evaluation on dev every min(max(the size's training positions, 32k),
100k) 1867 positions (arm b: twice as many steps apart), early stop after 3 evaluations without improvement of the
dev total loss; then the value refit (at most 800 steps, evaluated every 400, patience 2, on dev win CE), then the
final evaluation on validation. Seeds 0, 1, 2 at 100 and 885 games, 0 and 1 at 300 (budget); they vary the
initialization, the game orders and the shuffle draws; the dev / validation games and the subsets are fixed, so
arms are compared on the same positions. 27 cells.

CPU throughput behind these numbers (2026-11-13, cores 24-31,56-63 shared with other jobs, 1830 positions, batch
128, stage 1): alone, the probe model trains at 300 positions/s on 1 thread + 1 loader worker (434 on 2 threads), the
150k smoke config at 444 (719 on 2 threads); bf16 autocast was 1-11% faster in paired runs and is not used (training
and evaluation stay in one precision). A cell is ~1.1 GB (trainer) + ~0.6 GB (loader worker). With 7 cells at a time
(1 thread each) next to the other agents' jobs on the same cores, each cell trained at **~107 positions/s** (1867,
measured on the first launch): a first plan (cap 7,812 steps, `c_pre` 3,906, 50 dev games, refit 1,600, three seeds
everywhere) would have taken ~10 hours, so it was stopped after ~8 minutes (before any 1867 evaluation) and cut to
the above (~5-6 hours).

**Price-grid fix (A3) and what was rerun.** The second launch ran on master's code, whose supervised collate built
1867's price cells in price space instead of the layout's step grid (A3 fixed it: `LegalActions.price_cells`), so
1867's price targets and price loss were wrong there -- and the price loss (weight 0.1) reaches the shared trunk.
It was stopped after ~2 hours, A3's branch merged, and every 1867 cell rerun on the merged code with A3's
`generic_1867` (`probe_1867_fixed`). Collated batches before and after the merge (`MixedPositionBatches`, same
seed): 1830's bit-identical; 1867's differ exactly in `price_cell_mask`, `price_grid`, `price_cell` and
`price_log_count` (in one batch of 128: 1 of 92 price cells, 3 log counts, 80 of 2,208 mask entries). So the three
1830-only `c_pre` cells were reused (`runs/C_PRE_REUSED.txt`). Four pre-fix full-size cells had finished (a s0,
b s0-s2); they are kept (`probe_1867_prefix`) to show how much the wrong price targets moved policy CE, top-1 and win
CE ("Results" below).

#### Decision rule (fixed 2026-11-13, before any 1867 cell ran)

Metric: 1867 validation **policy CE** (mean NLL of the human move over the legal moves) of each run's final
checkpoint. Per (arm, size), the **seed spread** is the range (max - min) over its seeds (3; 2 at 300 games). For X in {b, c} at size
n: gain = mean CE(a) - mean CE(X); noise = the larger of the two spreads. X **passes** at n iff gain > 2 x noise
**and** X's win CE (after the value refit) is not worse than a's by more than 2 x the larger win-CE spread.

* **Adopt sharing for 1867 only if some X passes at the full size (885 games).** These are 0.5M-parameter CPU
  models: a signal, not proof for the full-size models. So "adopt" means: run the full-size comparison (E1: the
  default model, 1867-only against X, plus a policy-only 1867 head-to-head) on the GPU next, and start 1867's
  supervised stage from X only if it holds there. If both pass, prefer c (per-title weights, releases decoupled)
  unless b's gain exceeds c's by more than 2 x noise.
* If X passes only at 100 or 300 games: keep 1867 per-title; record it as evidence for warm-starting data-poor
  titles (1822CA, 1822MX, 1828) from another title's model (family pooling stays the first sharing step).
* If nothing passes: keep per-title models (the default). A null result at 0.5M parameters does not rule out
  transfer at 8M (published multi-game transfer grows with model size, design §4.2.3); it means no GPU time for
  cross-title sharing before 1867's per-title pipeline.
* An X worse than a by more than 2 x noise at the full size is recorded as interference.
* Reported, not part of the decision: top-1 overall and per family, top-5, winner accuracy; the 1830 validation of
  the joint model and of c after fine-tuning against `c_pre` (interference / retention).
* Budget caveat: with equal step caps, arm b sees half of a's 1867 positions (conservative for b), and c's extra is
  its 1830 phase. If a's full-size runs are still improving at the cap (best = last evaluation, flagged in the
  report), a full-size pass is provisional on that budget -- E1 decides anyway.

The harness writes the per-cell runs under `<out>/runs/`, `report.md` / `report.json` (the table per arm x size with
mean and half the seed range, per-family top-1, the 1830 interference rows, each pass / fail and the verdict).

#### Results (2026-11-13, `probe_1867_fixed`, code 882dd81 = C3 + A3's price-grid fix)

All 27 cells finished (24 run in 4.8 hours wall at 7 at a time, + the three reused `c_pre` cells, 47 minutes each).
1867 validation, 50 games / 30,938 positions (chance top-1 of a uniform policy over the legal moves 0.396; A3's
own small model reached top-1 0.729 after 4,000 steps); mean +- half the seed range:

| arm | 1867 games | seeds | policy CE | top-1 | top-5 | win CE (equal odds 1.368) | winner acc | best / last step |
|---|---|---|---|---|---|---|---|---|
| a: 1867 only | 100 | 3 | 0.958 +- 0.043 | 0.710 +- 0.018 | 0.923 | 1.260 +- 0.162 | 0.375 | 728 / ~2,000 (stopped) |
| b: joint 50/50 | 100 | 3 | 0.976 +- 0.046 | 0.703 +- 0.019 | 0.922 | 1.231 +- 0.162 | 0.461 | 1,165 / ~3,800 (stopped) |
| c: 1830 then 1867 | 100 | 3 | 0.968 +- 0.041 | 0.705 +- 0.015 | 0.923 | 1.368 +- 0 | 0.284 | 583 / ~1,900 (stopped) |
| a | 300 | 2 | 0.890 +- 0.007 | 0.730 | 0.932 | 1.094 +- 0.036 | 0.522 | 2,343 / 4,686 (stopped) |
| b | 300 | 2 | 0.878 +- 0.011 | 0.733 | 0.933 | 1.037 +- 0.071 | 0.565 | 5,000 (cap) |
| c | 300 | 2 | 0.876 +- 0.011 | 0.733 | 0.935 | 1.077 +- 0.001 | 0.534 | 2,343 / 4,686 (stopped) |
| a | 885 | 3 | **0.856 +- 0.002** | 0.735 | 0.937 | 0.900 +- 0.039 | 0.600 | 5,000 (cap) |
| b | 885 | 3 | 0.877 +- 0.008 | 0.732 | 0.934 | 0.961 +- 0.054 | 0.576 | 5,000 (cap) |
| c | 885 | 3 | **0.856 +- 0.013** | 0.737 | 0.935 | 0.927 +- 0.064 | 0.597 | 5,000 (cap) |

The rule (gain > 2 x the larger seed range): size 885 -- b -0.021 vs range 0.016, c -0.000 vs 0.026; size 300 -- b
+0.013 vs 0.023, c +0.014 vs 0.022; size 100 -- b -0.018 vs 0.091, c -0.010 vs 0.086; no win-CE loss beyond noise
anywhere. **Verdict: no sharing signal at any size -- keep per-title models (the default).** At the full size the
1830-pretrained start ends where 1867 alone does (c = a within 0.001), and the joint run, which sees half the 1867
positions at the same step cap, is 0.021 worse (not beyond its noise); at 300 games both shared arms are ~0.013
better, under the noise of two seeds. Top-1 per family moves by a few points either way with no consistent arm
(e.g. at 885: Bid.company a 0.703 / b 0.675 / c 0.721, Merge 0.217 / 0.173 / 0.232, LayTile 0.252 / 0.244 / 0.253).
Every full-size cell ended at the cap with its best checkpoint the last evaluation (cosine schedule): the
comparison is at 5,000 steps.

1830 validation (30 games): `c_pre` (1830 only, 2,500 steps) 0.834 policy CE; the joint model 0.828 at 300 / 885
games (no interference: 1867 in the batch did not hurt 1830 at this size), 0.902 at 100 (stopped early); c after
fine-tuning on 1867 1.00-1.02 (it forgets 1830).

Caveats, recorded after the results (none changes the pre-set verdict):

* **At 100 games the selection, not the data, decided.** Early stopping uses C1's dev total loss, which includes
  0.1 x the win CE; on 100 games the value side memorizes within one or two evaluations, so every 100-game cell kept
  a checkpoint from 437-1,748 steps and stopped soon after while its dev policy CE was still falling (e.g. a s0: 1.014, 0.928, 0.915, 0.904,
  0.893 at 437 .. 2,185 steps; best by total at 874). Seeds that kept the 437-step checkpoint have validation policy
  CE ~1.0, those that kept 874 ~0.92: hence the +-0.04 ranges. Exploratory (dev, not validation): the lowest dev
  policy CE each 100-game run reached was a 0.904 (range 0.029), b 0.888 (0.031), c 0.893 (0.025) -- the shared arms
  ~0.01-0.015 ahead, inside the noise. The value refit on 100 games never beat equal odds on dev in 7 of 9 cells
  (fresh heads kept: win CE = 1.368). A small-data rerun should select on dev policy CE (or train the value heads
  detached); not done here.
* Small model (0.54M), one budget (5,000 steps), one joint share (0.5), fine-tuning at the from-scratch learning
  rate: a signal, not proof, for the default-size models (design §4.2.3: transfer grows with model size).
* **The wrong 1867 price targets (pre-fix code) moved the policy little.** The four full-size cells that finished
  on master's code before the fix (`probe_1867_prefix`), against the same seeds after it: policy CE a s0 0.8597 ->
  0.8551, b s0 / s1 / s2 0.8891 / 0.8713 / 0.8916 -> 0.8763 / 0.8686 / 0.8849 (0.003-0.013 lower after the fix),
  top-1 within 0.001, price NLL 1.70-1.75 -> 1.41-1.46; win CE 0.03-0.10 higher after the fix in all four, inside
  its seed range at that size (0.08-0.13: the refit selects on 30 dev games). None of the pre-fix cells is used above.

Artifacts: `<scratchpad>/c3/probe_1867_fixed/` (`plan.json`, `splits/`, `runs/<cell>/` with checkpoints, history and
summary, `report.md`, `report.json`); the pre-fix cells in `probe_1867_prefix/`.

#### Harness validation on 1830 alone (2026-11-13)

Before the 1867 data existed the harness ran end to end with 1830 as both target and source (`plan --target
1830=<d> --source 1830=<d> --pseudo-target-games 40`: disjoint games -- a 40-game pseudo-target + 4 dev games and
half the validation games against the other 3,834 training games), sizes 10 and 40 games, 2 seeds, tiny budgets
(arm a and c's fine-tuning 100 steps, arm b 200 -- that plumbing run predates the equal step caps --, `c_pre` 100;
6 validation games): 14 cells in 9.3 minutes at 4 at a time; every cell wrote its summary, the report built its
tables and applied the rule. Target validation policy CE (mean +- half the seed range): a 1.005 +- 0.010 / 1.012 +-
0.004 (10 / 40 games), b 0.965 +- 0.009 / 0.965 +- 0.005, c 0.971 +- 0.002 / 0.971 +- 0.002; b and c "pass" at 40
games. Every cell was budget-bound (best = last evaluation) and b had twice the steps, so these numbers check the
plumbing only, not transfer.

## 1822 family: data, starts, caps and the CPU smoke (2026-11-14)

Code: `generic_import_1822.py` (the importer and the matching, shared with `scripts/g1822_completeness.py`),
`generic_data.py` (the four titles registered; `importer_for(title, **options)`, `convert-generic
--allow-optional-rules`; `seat_order`), `generic_play.py` (the opening is the first Stock Round:
`OPENS_WITH_STOCK_ROUND`, `opening_done`, `opening_end_cut`; seeded auction starts; `price_field`), `generic_pg.py`
(seats by player id: `mover_seat`, `PGRow.order`, `game_outcome(raw, seating)`; Ruby dead ends and refused moves
end a game: `_advance`, `apply_move`), `scripts/eval_generic.py` (the same), `model_generic.legal_actions(...,
moves=)`, `titles.py` (`supervised_finished_only` for the four titles), `scripts/build_generic_start_positions.py`
(SR1-end starts). Tests: `tests/agent/alphazero/test_generic_1822.py` (20 tests, 34 cases, ~3 min on an idle CPU;
the 1830 / 1867 suites pass unchanged). Scratch area of this milestone: `<g1822data>` =
`/tmp/claude-1000/-home-revys-workspace-RL-18xx--claude-worktrees-1830-rl-analysis-ea948b/48dd27c6-ebff-40c4-8d11-59fd0cf94716/scratchpad/g1822data`
(datasets, start files, runs, evals, PG runs, logs, the driver scripts in `scripts/`).

### The importer (`generic_import_1822.Importer1822Family`)

* **Input**: raw 18xx.games exports (`human_games/1822` -- the 2020-22 corpus -- and
  `human_games/scrape_2026_10/<title>`), replayed as the corpus audits replay them (`replay_export`: Ruby's
  filtered actions, recorded auto-actions processed right after their parent, seats renumbered 1001..1000+N as
  "Player k" -- the family's minors have numeric syms --, the engine seed of `settings.seed` / the game id).
* **Scope**: finished or archived games of the title at the pipeline's player counts (1822 / 1822CA 3-6,
  1822PNW / 1822MX 3-5), base rules only (`TitleSpec.optional_rules` is empty for the family). Dropped and
  counted: `players_<n>`, `optional_rules`, `status_<s>`, `title`, `no_actions`; then (finished games only, the
  family's `supervised_finished_only`) `unfinished_replay_rejected` and `unfinished_record_ends`. Other variants
  (1822CA ERS / WRS, 1822MRS / NRS, the PNW short scenario) are other titles on the site and not in these
  directories.
* **Labels**: every decision (recorded auto-actions and actions after the game's end are not decisions) is
  matched with `scripts/g1822_completeness.py --index`'s matching (moved into the importer module; the script
  imports it): type, entity and the outcome-relevant parameters, ranged prices by range *and step*, `run_routes`
  to the mover's RunRoutes index. Exactly one hit among the **rules'** moves is the label; the stored legal list
  is the **title's cross-player policy**'s (`RESTRICTED`), so a human move it excludes is skipped as `policy`.
  The policy only removes `buy_train` / `acquire_company` / `merge` moves (`g1822_cross_player_excluded`, and
  falls back to the rules' set when it would leave nothing), so other types are matched on the policy's own list
  -- one generator run per decision instead of two -- and the rules' list is asked for only for those three types
  (tested: the same label at every decision of the test fixtures). Misses: `replay_only:out_of_turn` (the rules
  offer the human's entity nothing: another crowded corporation's discard, the tax-haven owner out of turn, a run
  by a corporation that is not operating -- `g1822_remaining_work.md` "Replay-only"), `replay_only:reacquire_owned`,
  else `unmatched`; several hits `ambiguous`.
* **Prices**: the chosen row's human price (`price` for bids and trains, `amount` for a partial debt payoff) with
  its `(lo, hi, step)` in step units (the generator's grid: $5 bid boxes, $10 1822MX token bids, $1 trains and
  payoffs).
* **Outcomes**: `result()` in seating order -> win share and placement points
  (`policy_gradient.default_placement_points` of the count), as 1867's.

### Where a game starts: the opening is the first Stock Round

The family has no separate auction: its first Stock Round *is* the opening auction -- the players bid on the bid
boxes' minors, concessions and privates (and, in 1822MX, may buy NdeM shares), and the bids are resolved when the
round ends; the first Operating Round runs the minors won there. So the four start kinds are (`generic_play`,
`OPENS_WITH_STOCK_ROUND`):

| start | 1830 / 1867 | the 1822 family |
|---|---|---|
| `auction` (`--auction-fraction`, `eval_generic.py --auction-start`) | the fresh game (the private / minor auction) | the fresh game: SR1's bidding is played by the policy. The engine's company-stack shuffle depends on the seed, so the start draws one (`seed=None` meant seed 0: every opening would have dealt the same bid boxes) |
| `human:<id>` (start file) | the start of SR1, after the human auction | **the end of SR1** -- the first position of another round (OR1, or the Choices round before it), with the humans' winning bids (`opening_end_cut`; lines carry `"start": "sr1_end"`) |
| `random` | random auction endings (1830) / a random opening auction played to SR1 (1867) | SR1 played to its end by uniformly random legal moves (`opening_done`), bids uniform multiples of $5 in `[lo, 2 lo]` and, for a $0 minimum (privates), `[0, 20 steps]` -- intentionally un-human, as 1830's random endings; 22-59 actions at 3-5 players |
| `midgame:<id>:sr<k>` | the start of Stock Round k = 2..6 | the same (SR2..SR6 are again bid-box rounds) |

Why the end of SR1 and not its start or SR2: SR1's start is the fresh game (= the auction start, no human
information); SR2 would also hand OR1 to the humans. The end of SR1 is exactly what an 1830 SR1 start is: the
humans' opening allocation at the humans' prices, every later decision the policy's.

### Legal prices: every price kind of the family

The family prices five blocks: `Bid.company` (`bid`, the "Bid" ladder, $5 steps; privates open at $0), `Bid.minor`
(`bid_range`), 1822MX's `Bid.token` (`bid`, $10 steps), `BuyTrain.corp` (`train_transfer`, $1) and
`PayoffPlayerDebtPartial` (`amount`, "BuyTrain" cells, $1). Every sampling path goes through
`LegalActions.price_cells(k)` on the step grid (A3), so the cells hold legal prices only. One path was wrong for
the family: `TitlePlay.apply` wrote the sampled price into the action dict's `price` field, but a partial payoff
carries it in `amount` -- the PG workers and `eval_generic.py` paid the range's low end ($1) whatever was sampled,
and the trainer scored the sampled amount (a wrong PPO ratio for those moves). Fixed (`generic_play.price_field`,
also used by the random openings). Test (`test_sampled_family_prices_are_legal_and_apply`): positions with an open
price of every priced block, harvested from fixture replays (1822CA `1.json`'s debt payoffs, 1822PNW
`rockport_token`, 1822MX `1822MX_game_end_stock_market`'s NdeM token auctions, 1822's bids and train sales), a
random small model's server cells and the policy-only player's draws (eps 0.2): every price on its move's grid in
range and applied by `rl_apply_index`; the kinds seen are exactly {bid, bid_range, train_transfer, amount} and the
steps {1, 5, 10}; `TitlePlay.apply` at the sampled price logs the same action as `rl_apply_index`.

### The datasets (base rules, finished games only)

```bash
CUDA_VISIBLE_DEVICES= taskset -c 24-31,56-63 nice -19 uv run python main.py convert-generic --title 1822 \
    --data-dir human_games/1822 human_games/scrape_2026_10/1822 --output <dir> --workers 7
# 1822PNW / 1822MX / 1822CA: --data-dir human_games/scrape_2026_10/<title>
```

Dataset directories (scratch, to be copied to `human_games/` -- see "Open questions"):
`<g1822data>/datasets/{1822,1822PNW,1822MX,1822CA}_v1`.

| title | files | games (train / val) | positions (train / val) | games (positions) at 3 / 4 / 5 / 6 players | conversion |
|---|---|---|---|---|---|
| 1822 | 3,976 (2020-22: 3,233; scrape 743) | 84 (81 / 3) | 92,633 (89,408 / 3,225) | 19 (20,108) / 45 (48,881) / 18 (21,400) / 2 (2,244) | 19 min, 7 workers |
| 1822PNW | 597 | 417 (395 / 22) | 437,034 (413,741 / 23,293) | 150 (150,176) / 204 (215,459) / 63 (71,399) / - | 127 min, 7 workers |
| 1822MX | 363 | 229 (218 / 11) | 225,519 (214,582 / 10,937) | 78 (75,473) / 114 (113,099) / 37 (36,947) / - | 37 min, 4 workers |
| 1822CA | 187 | 124 (114 / 10) | 158,824 (145,928 / 12,896) | 26 (32,341) / 68 (87,014) / 23 (29,733) / 7 (9,736) | ~45 min, 4 workers |

(Conversion times on the shared cores, beside the generator agent's profiling and the other jobs: 2-3x an idle
machine.) Validation games per count: 1822 1 / 2 / 0 / 0, PNW 12 / 7 / 3, MX 2 / 7 / 2, CA 2 / 6 / 2 / 0.

**Dropped games** (every one in `dropped.jsonl`):

| title | optional rules | unfinished: record ends / replay rejected | players out of range | other |
|---|---|---|---|---|
| 1822 | 2,837 (2020-22 2,251, scrape 586: `plus_expansion` and its variants, `tax_haven_multiple`) | 52 (2020-22 39, scrape 13) / 568 (all 2020-22) | 2p 39, 7p 51 | `status_active` 236, `status_new` 6, `no_actions` 103 |
| 1822PNW | 65 (`starting_packet` 34, `remove_three_ls` 21, `remove_two_ls` 10) | 94 / 6 | - | `no_actions` 15 |
| 1822MX | 51 (`higher_cert_limit`) | 50 / 0 | 2p 28 | `no_actions` 5 |
| 1822CA | 0 | 16 / 8 | 2p 32, 7p 4 | `no_actions` 3 |

**The 2020-22 1822 corpus gives one game**: of its 710 base-rules finished / archived games at 3-6 players,
102 have no actions, 568 are records the current rules reject part-way (A4: older rules -- bid boxes, the minors'
first turn, routes), 39 end early, and the one that finishes (70186) is four passes: every player passed in SR1,
and a Stock Round in which nothing is bought ends the game (Ruby `nothing_sold_in_sr`, end reason `dnf`). All
other 1822 data is the 2026-10 scrape's 83 base-rules games.

**Labels.** Skipped (never guessed): `policy` 1822 4 / PNW 15 / MX 9 / CA 10 (private acquisitions, train buys and
minor acquisitions across players); `replay_only:out_of_turn` 10 / 13 / 9 / 12 (discards by a crowded corporation
other than the first, one PNW `choose`); `replay_only:reacquire_owned` 0 / 2 / 2 / 1; `unmatched` 0 / 7 / 0 / 0
(PNW: 3 raises of one's own bid, 3 UI-stale choices, 1 partial payoff above the player's cash -- A4's documented
misses); `ambiguous` 0 everywhere. Recorded auto-actions applied as non-decisions: 7,068 / 30,476 / 17,866 /
13,360. Forced positions kept (one legal row): 22% / 22% / 22.7% / 19.9%. Price targets (open ranges): 12,577 /
70,855 / 31,585 / 24,964, none outside its range or off its grid.

**Legal lists** (rows per position; the family's are longer than 1867's, mean 19.4, max 336, and 1830's, mean
11.5, max 185):

| title | mean | median | p90 | p99 | max | LayTile positions: mean / p99 / max |
|---|---|---|---|---|---|---|
| 1822 | 24.6 | 4 | 33 | 384 | 921 | 133.5 / 486 / 921 |
| 1822PNW | 17.5 | 4 | 23 | 273 | 839 | 94.2 / 390 / 839 |
| 1822MX | 15.7 | 3 | 25 | 217 | 654 | 73.0 / 294 / 654 |
| 1822CA | 28.0 | 5 | 43 | 418 | 829 | 140.7 / 525 / 829 |

**Human decisions per game** (several legal rows or an open price; median / p99 / max): 1822 3p 840 / 954 / 962,
4p 871 / 1,109 / 1,138, 5p 876 / 1,128 / 1,139, 6p 852 / 913 / 914; PNW 787 / 1,042 / 1,099, 812 / 1,128 / 1,155,
867 / 1,026 / 1,027; MX 767 / 945 / 947, 770 / 937 / 960, 734 / 936 / 939; CA 1,009 / 1,356 / 1,371, 1,016 / 1,275 /
1,287, 1,016 / 1,187 / 1,196, 1,073 / 1,088 / 1,088. Human games end by the bank (1822 82 of 84, MX 211 of 229, CA
124 of 124, PNW 214 of 417) or the stock market, never in a Ruby dead end.

Matched positions by family (1822 / PNW / MX / CA): Pass 35,690 / 165,648 / 84,853 / 57,806; Bid.company (the
`Bid.company` and `Bid.minor` blocks) 11,943 / 69,789 / 29,468 / 24,213; LayTile 11,989 / 56,740 / 31,721 / 22,702;
RunRoutes 10,469 / 48,539 / 25,626 / 16,566; Dividend 5,815 / 23,190 / 12,295 / 9,785; Choose 2,953 / 17,867 /
8,794 / 6,291; BuyShares 4,124 / 15,635 / 9,134 / 6,013; BuyTrain.depot 2,939 / 12,531 / 5,971 / 4,424;
AcquireCompany 1,368 / 8,671 / 4,017 / 3,080; Merge 899 / 4,053 / 2,880 / 1,616; PlaceToken 1,066 / 3,534 / 1,509 /
1,502; BuyTrain.corp 799 / 1,985 / 1,535 / 1,060; Par 736 / 739 / 1,355 / 1,091; BuyTrain.exchange 483 / 3,760 /
1,397 / 916; SellShares 487 / 1,702 / 1,613 / 608; Bid.token (MX) 1,325; RemoveToken (MX) 628; Assign 724 (PNW) /
107 (CA); DiscardTrain, RedeemShares, IssueShares, ChooseAbility, PayoffPlayerDebt(Partial), BuyTrain.pool: the rest.

### Seats: the family reorders its players

The family reorders `BaseGame.players` every Stock Round (Ruby's `reorder_players`: by cash), 1830 and 1867 never
do. The encoder's seats follow the **current** order (seat k = player `(k + rotation) mod n` of `BaseGame.players`
as they stand, seat 0 the mover: checked on the 1822 fixture, 132 of 132 sampled positions; against the seating
order only 61 of 132), but three places read `meta[:, 5]` as an index into the seating order:

| where | effect for the family (before) | now |
|---|---|---|
| `generic_data.position_item` (supervised) | win / score targets of a reordered position on the wrong seats (most positions after SR1) | `seat_order(game, seating, rotation)`: each encoder seat's seating index by player id; `seat_frame(..., order)` |
| `generic_pg.play_generic_pg_games` | moves routed to the wrong seat's network, the learner's rows kept for the wrong seat, rewards and critic targets of other players, `game_outcome` in the final order, saved games' players in the final order (they did not replay) | `_Game.seating` (ids at the start), `mover_seat`, `PGRow.order`, `game_outcome(raw, seating)`, `seating_players` |
| `scripts/eval_generic.py` | routing and scoring as above, saved games did not replay | the same helpers |

1830 and 1867 are unchanged (there `order` is the rotation; their tests pass unchanged). The effect on the 1822
supervised smoke: validation win CE 1.30 (equal odds 1.30, winner 0.24) before, 0.69 (winner 0.68) after 500 steps.
Tests: `test_value_targets_follow_the_reordered_players` (every seat's target names the player by id on the 1822
fixture, seat 0 the mover), the PG test's rows (seat, order, targets).

### Start files

```bash
CUDA_VISIBLE_DEVICES= taskset -c 24-31,56-63 nice -19 uv run python scripts/build_generic_start_positions.py \
    --title 1822 --source human_games/1822 --source human_games/scrape_2026_10/1822 --out-dir human_games --jobs 3
# 1822PNW / 1822MX / 1822CA: --source human_games/scrape_2026_10/<title>   (2-10 min each)
```

Files: `<g1822data>/starts/{start,midgame}_positions_<title>_<N>p.jsonl` (the `TitleSpec` templates' names). As for
1867, starts use no outcome, so the builder's scope is wider than the importer's (abandoned `active` games and
records that end early give starts too); a game the replay rejects anywhere gives none; games with optional rules
are left out.

| title | players | SR1-end starts | mid-game games | cuts SR2 / SR3 / SR4 / SR5 / SR6 |
|---|---|---|---|---|
| 1822 | 3 | 23 (one from the 2020-22 corpus) | 22 | 22 / 22 / 21 / 21 / 21 |
| 1822 | 4 | 48 | 48 | 48 / 48 / 47 / 47 / 47 |
| 1822 | 5 | 21 | 20 | 20 / 20 / 19 / 19 / 19 |
| 1822 | 6 | **3** | **3** | 3 / 3 / 3 / 3 / 3 |
| 1822PNW | 3 | 168 | 166 | 166 / 165 / 165 / 164 / 162 |
| 1822PNW | 4 | 230 | 229 | 229 / 229 / 228 / 226 / 225 |
| 1822PNW | 5 | 70 | 70 | 70 / 69 / 69 / 69 / 69 |
| 1822MX | 3 | 89 | 89 | 89 / 89 / 88 / 86 / 86 |
| 1822MX | 4 | 133 | 133 | 133 / 131 / 131 / 128 / 127 |
| 1822MX | 5 | 42 | 42 | 42 / 41 / 40 / 40 / 38 |
| 1822CA | 3 | 28 | 28 | 28 / 28 / 28 / 28 / 27 |
| 1822CA | 4 | 74 | 74 | 74 / 74 / 72 / 72 / 71 |
| 1822CA | 5 | 26 | 26 | 26 / 26 / 26 / 26 / 25 |
| 1822CA | 6 | **10** | **10** | 10 / 10 / 10 / 10 / 9 |

Excluded: 1822 optional rules 2,956, replay failed 667 (the 2020-22 records), no actions 104, player count 95,
never past SR1 53, `new` 6; PNW optional rules 65, never past SR1 43, no actions 15, replay failed 6; MX optional
rules 51, player count 28, never past SR1 15, no actions 5; CA player count 36, replay failed 8, no actions 3, never
past SR1 2. ("Never past SR1": records that stop in the opening Stock Round -- abandoned games the site shows as
finished; the converter drops them as `unfinished_record_ends`.) SR1 lasts a median 42-118 actions of the logs
(more players, longer).

**Thin coverage**: 1822 at 6 players (3 games) and every 1822 count but 4 (the base-rules scrape has 83 finished
games); 1822CA at 6 players (10) and 3 / 5 players (26-28). A PG mix leans on auction starts (the fresh game,
seeded) and random openings there, as 1867's 6-player games do.

### CPU smoke (2026-11-14, `CUDA_VISIBLE_DEVICES=`, cores 24-31,56-63)

The cores were shared with the generator agent's profiling at machine load 50-80: wall times below are 2-4x an idle
machine's; the per-decision generator CPU is thread time.

**Supervised** (`pretrain-generic --model-preset small`, 154,494 parameters, batch 64, 1,500 steps = 96k positions,
eval every 500, 3 loader workers, shuffle buffer 8,192, validation on the first 20 validation games, value refit 400
steps; `<g1822data>/runs/<title>_sl_s1500/`, driver `<g1822data>/scripts/smoke_pretrain.sh`):

| title | train positions (epochs) | stage 1 | train loss first -> last 50 | val positions (games) | val top-1 / top-5 (chance) | val policy CE | val price NLL | val win CE (equal odds) / winner | refit win CE / winner |
|---|---|---|---|---|---|---|---|---|---|
| 1822 | 89,408 (1.07) | 25 min | 1.799 -> 1.223 | 3,225 (3) | 0.666 / 0.904 (0.384) | 1.089 | 0.778 | 0.628 (1.302) / 0.699 | 0.607 / 0.719 |
| 1822PNW | 413,741 (0.23) | 64 min | 1.712 -> 1.205 | 21,163 (20) | 0.685 / 0.933 (0.396) | 0.955 | 1.128 | 1.020 (1.267) / 0.546 | 1.009 / 0.546 |
| 1822MX | 214,582 (0.45) | 50 min | 1.758 -> 1.192 | 10,937 (11) | 0.678 / 0.924 (0.399) | 0.991 | 0.769 | 1.073 (1.377) / 0.500 | 1.025 / 0.511 |
| 1822CA | 145,928 (0.66) | 81 min | 1.884 -> 1.367 | 12,896 (10) | 0.625 / 0.896 (0.337) | 1.161 | 0.797 | 1.175 (1.367) / 0.482 | 1.151 / 0.462 |

Validation top-1 by player count: 1822 3p 0.691 / 4p 0.656; PNW 0.684 / 0.684 / 0.693; MX 0.685 / 0.674 / 0.685; CA
0.648 / 0.624 / 0.604 (no 5-6p validation game for 1822, no 6p one for CA). 1822's validation is 3 games: its win CE
(0.63, winner 0.70) is a 3-game number.

Per family on validation, top-1 against chance (mean 1 / legal count), families with at least 100 validation
positions in some title (n in brackets):

| family | 1822 | 1822PNW | 1822MX | 1822CA |
|---|---|---|---|---|
| Pass | 0.882 vs 0.461 (1,220) | 0.911 vs 0.494 (8,118) | 0.901 vs 0.512 (4,150) | 0.890 vs 0.427 (4,526) |
| Bid.company (incl. Bid.minor) | 0.291 vs 0.132 (364) | 0.352 vs 0.165 (3,205) | 0.311 vs 0.143 (1,608) | 0.189 vs 0.100 (2,274) |
| LayTile | 0.222 vs 0.045 (424) | 0.221 vs 0.051 (2,782) | 0.286 vs 0.060 (1,470) | 0.227 vs 0.036 (1,782) |
| RunRoutes | 1.000 vs 1.000 (383) | 0.996 vs 0.978 (2,419) | 0.989 vs 0.980 (1,189) | 1.000 vs 0.991 (1,270) |
| Dividend | 0.716 vs 0.312 (208) | 0.739 vs 0.260 (1,113) | 0.667 vs 0.266 (601) | 0.784 vs 0.287 (776) |
| Choose | 0.696 vs 0.541 (92) | 0.819 vs 0.434 (856) | 0.874 vs 0.502 (373) | 0.878 vs 0.463 (517) |
| BuyShares | 0.599 vs 0.048 (182) | 0.614 vs 0.063 (751) | 0.573 vs 0.059 (482) | 0.537 vs 0.044 (488) |
| BuyTrain.depot | 0.922 vs 0.372 (116) | 0.880 vs 0.398 (615) | 0.891 vs 0.434 (275) | 0.903 vs 0.365 (359) |
| AcquireCompany | 0.079 vs 0.379 (38) | **0.225 vs 0.361** (417) | **0.280 vs 0.369** (193) | **0.187 vs 0.384** (246) |
| Merge | 0.690 vs 0.422 (29) | 0.461 vs 0.432 (204) | 0.809 vs 0.527 (115) | 0.701 vs 0.402 (134) |
| BuyTrain.exchange | 1.000 vs 0.318 (16) | 1.000 vs 0.334 (182) | 1.000 vs 0.341 (65) | 1.000 vs 0.341 (76) |
| PlaceToken | 0.025 vs 0.117 (40) | **0.000 vs 0.109** (173) | 0.000 vs 0.177 (68) | **0.065 vs 0.139** (124) |
| BuyTrain.corp | 0.107 vs 0.302 (28) | **0.150 vs 0.249** (113) | 0.569 vs 0.288 (58) | 0.221 vs 0.234 (86) |

Every family with at least 100 validation positions is above chance except AcquireCompany (all four titles),
PlaceToken (PNW, CA) and BuyTrain.corp (PNW): the smoke model passes there (A3's 1867 smoke showed the same pattern
for its rare families after 600-4k steps). Par, SellShares, RedeemShares, RemoveToken (MX) and ChooseAbility stay
below chance with 20-90 positions. The value head learns in every title (win CE below equal odds -- after the seat
fix: before it, 1822's sat at equal odds). Improving these is the GPU run's job; judge it by this table.

**Policy-only self-play** (`scripts/eval_generic.py --title <t> --match M@0.5 M@0.5 --a-seats 1 --auction-start
--max-decisions 8000 --auction-stall-decisions 0`: the model against itself, every game from the fresh game -- the
bid-box SR1 played by the model --, in-process CPU inference, 3-4 workers; `<g1822data>/evals/<title>_t05_<N>p/`,
every game saved):

| title | players | games | wall | decisions: median / p90 / p99 / max | terminations | engine end reasons (saved games) | A's score (fair) |
|---|---|---|---|---|---|---|---|
| 1822 | 3 | 12 | 7 min | 880 / 1,183 / 1,278 / 1,289 | finished 12 | bank 12 | 0.250 +- 0.131 (0.33) |
| 1822 | 4 | 12 | 4 min | 786 / 987 / 1,092 / 1,104 | finished 12 | bank 4, stock market 5, dnf 3 | 0.313 +- 0.124 (0.25) |
| 1822 | 5 | 15 | 7 min | 769 / 942 / 999 / 1,004 | finished 15 | bank 7, stock market 4, dnf 1 | 0.080 +- 0.067 (0.20) |
| 1822 | 6 | 12 | 10 min | 980 / 1,191 / 1,225 / 1,229 | finished 12 | bank 6, stock market 6 | 0.250 +- 0.131 (0.17) |
| 1822PNW | 3 | 12 | 15 min | 858 / 1,027 / 1,107 / 1,116 | finished 12 | bank 10, stock market 2 | 0.333 +- 0.142 (0.33) |
| 1822PNW | 4 | 12 | 15 min | 1,050 / 1,120 / 1,182 / 1,190 | finished 12 | bank 11, stock market 1 | 0.250 +- 0.131 (0.25) |
| 1822PNW | 5 | 15 | 12 min | 952 / 1,091 / 1,193 / 1,209 | finished 15 | bank 11, stock market 1 | 0.133 +- 0.091 (0.20) |
| 1822MX | 3 | 12 | 3 min | 554 / 672 / 729 / 736 | **dead end 10**, finished 2 | dnf 1, bank 1 | 0.278 +- 0.129 (0.33) |
| 1822MX | 4 | 12 | 6 min | 813 / 906 / 912 / 912 | dead end 3, finished 9 | bank 8, dnf 1 | 0.271 +- 0.129 (0.25) |
| 1822MX | 5 | 15 | 9 min | 803 / 852 / 865 / 867 | dead end 5, finished 10 | bank 8 | 0.133 +- 0.091 (0.20) |
| 1822CA | 3 | 12 | 13 min | 1,300 / 1,794 / 1,996 / 2,017 | finished 12 | stock market 5, bank 5, dnf 2 | 0.389 +- 0.135 (0.33) |
| 1822CA | 4 | 12 | 15 min | 1,303 / 1,699 / 1,734 / 1,737 | finished 12 | bank 7, stock market 3, dnf 2 | 0.042 +- 0.028 (0.25) |
| 1822CA | 5 | 15 | 17 min | 1,464 / 1,689 / 1,828 / 1,848 | finished 15 | bank 10, dnf 2 | 0.240 +- 0.104 (0.20) |
| 1822CA | 6 | 12 | 14 min | 1,380 / 1,577 / 1,897 / 1,936 | finished 12 | bank 11, stock market 1 | 0.333 +- 0.142 (0.17) |

(5-player runs: 15 games, the first 12 saved.) At T 1.0 (4 players, 8 games each): 1822 median 948, max 1,389, all
finished (bank 8); 1822MX median 769, max 1,238, all finished, no dead end. Scores sit at the fair share within noise
except 1822CA 4p (0.04 +- 0.03 over 12 games, two of them all-pass ties; the arrangements rotate A over every seat).
Two endings besides the game's end (below): 1822MX's Ruby dead end, and the all-pass opening Stock Round (`dnf`
after 3-5 decisions). An earlier 1822MX run with the model trained before the seat fix (same policy recipe) gave
16 dead ends in 39 games.

**Policy gradient** (`main.py policy-gradient-generic --title <t>`: the title's smoke model as policy, critic (its
value heads) and SL opponent, no league; mix 1822 / CA 3:1,4:2,5:1,6:1, PNW / MX 3:1,4:2,5:1; placement reward,
T 0.5 / 0.5, KL 0.02 -> target 0.08, anchor every 2, starts auction 0.3 (the fresh game, seeded) / mid-game 0.3 /
random 0.2 / else human SR1-end starts (the files above), `--auction-stall-decisions 0`, `--max-decisions 8000` (to
measure, not to cap), 3-4 workers x 2 games, 16 rows per seat, 400-600 rows per update, minibatch 256 / microbatch
128, lr 2e-5 / critic 3e-5, CPU servers with 1 thread each, every 4th game saved, snapshot every update, then a
**resume** (`--resume <run> --max-updates N`); `<g1822data>/pg/<title>_smoke/`, driver
`<g1822data>/scripts/smoke_pg.sh`):

| title | updates (resume) | games | terminations | starts: auction / human / mid-game / random | decisions: mean per count, max | approx_kl, \|log ratio\| | KL to anchor | critic CE | games/h (decisions/s) |
|---|---|---|---|---|---|---|---|---|---|
| 1822 | 5 (3, then a resume from snapshot 3 to 5) | 100 | finished 100 | 29 / 40 / 28 / 3 | 790-1,070; 1,313 | <= 4e-4, <= 0.0033 | <= 8e-5 | 1.16-1.39 | 65-94 (17-32) |
| 1822PNW | 3 (2 + 1) | 48 | finished 48 | 10 / 21 / 11 / 6 | 720-905; 1,096 | <= 4e-4, <= 0.0024 | <= 1.5e-4 | 0.93-1.31 | 33-68 (8-16) |
| 1822MX | 3 (2 + 1) | 48 | finished 39, **dead end 9** | 13 / 17 / 11 / 7 | 627-826; 990 | <= 5e-4, <= 0.0023 | <= 5e-5 | 0.83-0.94 | 92-102 (17-22) |
| 1822CA | 3 (1; update 2 died on a refused move -- below --; a resume from snapshot 1 to 3) | 42 | finished 42 | 11 / 19 / 9 / 3 | 1,078-1,486; 1,873 | <= 6e-4, <= 0.0020 | <= 1.2e-4 | 0.90-1.25 | 27-65 (10-28) |

Every resume continued from its last snapshot (optimizer, anchor, pool restored), losses stayed finite, the trainer
recomputed the servers' sampling probabilities (price cells on the grid) within `|log ratio|` 0.0033, and no start
failed to replay (`skipped_starts` 0). The workers wait on the 1-thread CPU servers for ~70-75% of their time (the
network is the CPU limit here; on the GPU the generator will be). 1822 / CA self-play games run longer than human
games (CA up to 1,873 decisions against the humans' 1,371).

**Generator CPU on self-play positions** (`rl_legal_moves_json()` under the title's policy, thread CPU, replaying the
4-player (PNW 3-player) eval games above at machine load ~75, so high):

| title | decisions | mean ms | p50 | p90 | p99 | max | where |
|---|---|---|---|---|---|---|---|
| 1822 | 7,656 | 18.3 | 5.4 | 47.6 | 168 | 537 | stock round 53% (15.9 ms each), Track 37% (46.4 ms) |
| 1822PNW | 10,756 | 42.5 | 4.7 | 69.1 | 785 | 2,414 | Track 84% (178 ms; every PNW lay applied to a clone), stock round 9% |
| 1822MX | 8,812 | 8.6 | 2.7 | 19.6 | 94 | 638 | Track 47% (24 ms), stock round 45% (6.6 ms) |
| 1822CA | 14,352 | 25.8 | 9.4 | 65.9 | 230 | 694 | stock round 72% (28 ms), Track 11%, Token 6% (70 ms), Route 2% (37 ms) |

Self-play spends more of its time in stock rounds than human play (A4's table had Track first for 1822, PNW and MX):
the bots' stock rounds are long (SR1 alone ran up to 351 player actions at 1822 5p; humans' at most 209). Against
1830's ~0.03 ms and 1867's ~1.2 ms per decision the family is 7-35x 1867: a GPU PG run is bound by the workers'
generator CPU (32 workers at ~25 ms: ~1.3k decisions/s, ~4-5k games/h of ~1,000-1,400 decisions) until the
generator speed-up lands.

### Decision caps and stall policies (proposal, for the user to decide)

Decisions per game (several legal rows or an open price), every game played to its end or its dead end:

| title | human (finished) max, 3/4/5/6p | self-play T 0.5 max, evals (3/4/5/6p); PG | T 1.0 4p max | proposal `--max-decisions-by-players` |
|---|---|---|---|---|
| 1822 | 962 / 1,138 / 1,139 / 914 | 1,289 / 1,104 / 1,004 / 1,229; PG 1,313 | 1,389 | 3:2000,4:2000,5:2000,6:2200 |
| 1822PNW | 1,099 / 1,155 / 1,027 | 1,116 / 1,190 / 1,209; PG 1,096 | - | 3:1800,4:1800,5:1800 |
| 1822MX | 947 / 960 / 939 | 736 / 912 / 867; PG 990 | 1,238 | 3:1600,4:1600,5:1600 |
| 1822CA | 1,371 / 1,287 / 1,196 / 1,088 | 2,017 / 1,737 / 1,848 / 1,936; PG 1,873 | - | 3:3000,4:3000,5:3000,6:3000 |

(~1.5x the longest self-play game and at least ~1.6x the human maximum, A3's rule for 1867. A cap that fires on more
than a few games per update means a degenerate policy, not a short cap.)

**Stalls.** The family has no Auction round, so `--auction-stall-decisions` never fires (it checks for an `Auction`
round; leave it 0). Its opening auction is SR1, where every bid locks cash and must beat the last, so SR1 ends
(self-play's longest: 351 player actions). Three other early endings, each a decision for the user:

1. **The all-pass opening** (all titles): a Stock Round in which nothing is bought ends the game (Ruby
   `nothing_sold_in_sr`, end reason `dnf`; the one 2020-22 game the 1822 dataset keeps is such a game). From the
   fresh game the T 0.5 smoke models pass all around in SR1 in ~10% of games (1822 4 of 51, MX 2 of 39, CA 7 of 51,
   PNW 0 of 39): a 3-6-decision game with every seat tied. With the placement reward a tie pays every seat the mean
   points, so these games carry no gradient toward bidding: wasted starts. Proposal: the generator redraws a start
   whose game ends in its opening Stock Round (a generator policy, not a rules change; counted like
   `skipped_starts`), or the family starts from SR1-end and mid-game positions only (`--auction-fraction 0`).
   No later Stock Round ended this way in these games.
2. **1822MX's Ruby dead end**: "the NdeM has no token left for the minor's home (Ruby raises)" in a late Stock Round
   (turn 10-11): Ruby crashes there too, and the engine mirrors it. Never in the 229 human games; 18 of 39 T 0.5
   self-play games (10 of 12 at 3p), 9 of 48 PG games, none of 8 at T 1.0. `generic_pg._advance` now ends such a
   game as it stands (termination `dead_end`) instead of crashing the worker. Options: keep that (biased: the game
   is cut late and scored on net worth), drop those games' rows, or a self-play rule deviation (e.g. the NdeM skips a
   home token it does not have), as the user chose for 1830's train shuffling. *Follow-up (2026-11-14,
   `g1822_remaining_work.md` "The NdeM's eleventh home")*: the eleventh unbid first-box minor before the first
   7-train; humans let that minor go as often (79% of Stock Rounds vs the bots' 90%) but reach phase 7 two Stock
   Rounds sooner (median 8 vs 10). The printed rules have no cap (each removed minor's own token turns into an NdeM
   station); Ruby's ten tokens are an implementation limit. **Decided by the user and implemented the same day**:
   the NdeM gets a new token per removed minor (rules-faithful; differs from Ruby only where Ruby crashes), so this
   ending is gone.
3. **A generated move the engine refuses** (a generator bug): one 1822CA PG game hit "216-1 is not legally rotated
   for 512" -- the generator offered the upgrade of tile 216 to 512 at a rotation `process_action` (the strict
   tile-lay rules) refuses -- and the pool task, i.e. the whole run, died. `generic_pg.apply_move` (and
   `eval_generic.py`) now end such a game as it stands (`rejected_move`; the move, price and error in the record,
   the game always saved). The failing game itself was not saved (the crash predates the handler), and the 42 CA
   PG games after it and the 51 CA eval games did not hit it again; a scan of every generated lay of the saved CA
   eval games through `process_action` was too slow to finish on the loaded machine
   (`<g1822data>/scripts/scan_refused.py`). For the generator agent: a game that hits it now lands in the run's
   `games/` with the refused move.

### Against A4's index-level completeness

A4's table (`docs/rl_descriptors.md` §9) counts every 2026-10 scrape game at the pipeline's player counts **with
any optional rules**, and every game to where its replay stops. The base-rules, finished-only datasets are subsets
of it (1822 83 of its 683 games, PNW 417 of 597, MX 229 of 335, CA 124 of 151); per game the importer's counts are
the completeness script's (`test_counts_equal_index_completeness` on one fixture per title). For an exact corpus
check, 1822CA -- whose 151 in-scope games all use the base rules -- was converted once more with
`--keep-unfinished --allow-optional-rules` (`<g1822data>/datasets/1822CA_allgames`, 125 min with 3 workers):

| | A4 (`--index`, the rules' moves) | the importer (all 151 in-scope games) |
|---|---|---|
| games with decisions | 151 in scope | 148 (3 have no action) |
| decisions with exactly one index | 171,343 (12 of them policy-excluded) | **171,331 labels** + 12 `policy` skips (5 private, 4 train, 3 minor acquisitions) = 171,343 |
| misses | 22: 13 out-of-order discards, 1 re-acquisition, 8 at records Ruby HEAD rejects | 13 `replay_only:out_of_turn` (discards) + 1 `replay_only:reacquire_owned`; the 8 are where the replay stops (7 passes, 1 lay: `stopped`), not decisions |
| total | 171,365 | 171,331 + 12 + 14 + 8 = 171,365 |

So the importer labels exactly A4's exactly-one-index decisions minus the policy-excluded ones, game for game, with
no ambiguous match. For the other titles the base-rules subsets show the same composition: PNW 15 `policy`, 13
`out_of_turn` (12 discards, 1 choose), 2 re-acquisitions, 7 `unmatched` (3 raises of one's own bid, 3 UI-stale
choices, 1 payoff above cash -- A4's documented PNW misses); MX 9 / 9 / 2 / 0; 1822 4 / 10 / 0 / 0.

### The 1822 family outline (status after this milestone)

1. **Layouts and import**: done (A4's layouts; this importer). Datasets in `<g1822data>/datasets/`, to be copied to
   `human_games/` with the start files.
2. **Starts**: built (SR1-end starts, mid-game cuts at SR2-SR6); thin for 1822 (every count) and 1822CA at 6 players.
3. **Supervised**: `convert-generic --title <t>` (finished games, base rules), `pretrain-generic` (default preset on
   the GPU, per title or mixed as in C3), the value refit; judge by the per-family table above.
4. **Policy gradient**: `policy-gradient-generic --title <t>` with the caps above and the user's choices on the
   all-pass opening and the MX dead end; generation is bound by the generator's CPU until its speed-up lands.
5. **Evals**: `scripts/eval_generic.py --title <t>` (generic vs generic).

### Open questions (the user's decisions)

1. **Optional rules for 1822**: the pipeline's scope is base rules (`TitleSpec.optional_rules` is empty), which leaves
   1822 with 83 finished games (92,633 positions) while 586 of the scrape's 683 in-scope 1822 games use 1822+
   (`plus_expansion` and its variants, `tax_haven_multiple`); the engine and A4's layout support all four rules (A4's
   fuzz and completeness ran them). Admit 1822+ -- and 1822MX's `higher_cert_limit` (51 games), PNW's
   `starting_packet` / `remove_three_ls` / `remove_two_ls` (65)? `convert-generic --allow-optional-rules` converts
   them (a measurement); self-play would need them in `TitleSpec.optional_rules`.
2. **The start**: the end of SR1 as the family's human start (above).
3. **Caps and the three early endings** (above).
4. **Random openings** play SR1 uniformly (bids `[lo, 2 lo]`, privates `[0, 20 steps]`): keep?
5. **Thin data**: 1822 (83 games; 6p 2 games, 3 starts), 1822CA 6p (7 games, 10 starts).
6. **Copy** the datasets and start files from the scratch area into `human_games/` (the `TitleSpec` templates' names).
7. **Placement points** at 3, 5 and 6 players: 1830's gen6 tables apply (C2's open question 1).

### Notes on the design documents (found here)

* `docs/rl_encoder.md` "Seats" called the encoder's player order the seating order; the 1822 family reorders
  `BaseGame.players` every Stock Round, and three consumers read the rotation as a seating index (fixed; the doc now
  says so).
* C2's starts text said the family's random opening is the empty game, and the start builder that the family's SR1
  starts are empty ("only its mid-game file is useful"): superseded by the SR1-end starts.
* The auction start carried no seed: the family's engine then used seed 0, so every opening dealt the same bid boxes
  (fixed for the family; 1830 / 1867 unchanged).
* `TitlePlay.apply` put a partial debt payoff's price in `price` while the action's field is `amount`
  (`docs/rl_descriptors.md` §6 had it right): every sampled payoff paid $1 (fixed).
* `docs/rl_descriptors.md` §9's completeness table counts games with optional rules in "Games in scope" (683 for
  1822); the pipeline's base-rules scope is 97 of those (83 finished).
* `scripts/g1822_completeness.py`'s `_in_range` raised on a missing price (it now shares the importer's matching).
* `rl_title_status(t)["encoder_ready"]` is False for the family: the legacy flat encoder's flag (encoder v2 encodes the
  family).
