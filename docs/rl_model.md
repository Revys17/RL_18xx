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
policy gradient against a league, on generic inference servers. The frozen 1830 model (`model_transformer.py`)
is untouched.

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
     Other map families (`PlaceToken`, `RemoveToken`: hex · city; `HexToken`: hex) have one row per key at
     the last level (log 1 = 0).
   * One **softmax over the position's legal rows** (`segment_log_softmax`) then balances map groups against
     each other and against non-map rows by the state-dependent group score.
   * **Price head** (every priced legal row): `MLP(W h_row + W_range range_features)` -> 24 cell logits,
     empty cells of the row's range at -1e4 before the log-softmax. The cell structure comes from the
     descriptor: `price_kind` -> `price_cell_types` ("Bid" ladder at the range's step, "BuyTrain" atoms
     lo / hi / hi-1 + 20 bins, "BuyCompany" lo / hi + 8 bins; `amount` raises: no cells yet) and the range
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
  price cell (> 0.9) on eight 1830 positions (priced and tile-lay ones) in 300 Adam steps.

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
3. **`amount` price kind** (1822 partial debt payoff) has no cell spec; `price_cells_for` raises. 1867 never
   uses it (its loans are automatic); fix the cells from a price census when 1822's family lands.
4. **Cost of very large legal sets**: the edge term materializes `(lay rows, 6, d_edge)`; at ~1,000 lays
   per position and batch 64 that is ~12M floats. Fine for 1830 / 1867 (the random walks' legal lists stay
   under ~150 rows; real tile-lay states reach a few thousand, design §2.6); for 1822CA-sized maps, compute it
   per (hex, tile) for all six rotations at once.
5. **Fusion bias memory** `(B, heads, E, H)`: ~20M floats per batch of 64 1822CA positions (E 138, H 279).
   Fine on the GPU; a per-head low-rank form if it is not.
6. **Sharing across titles** (design §4.2): the architecture shares everything except the id embeddings and
   the per-title tables; whether to train one model over several titles is still E0/E1's question.
7. The per-type entity projections of the 1822 family's concession / minor-proxy types train only on 1822
   data; an 1830 / 1867 model leaves them at init (harmless, but a DDP run must set
   `find_unused_parameters`).

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
fingerprint, the encoder version at conversion, sources, split rule, counts, match / skip statistics) and
`games.jsonl` (id, split, players, decisions, actions, finished). A **record** (`generic_data.convert_game`):
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
raises `UnsupportedTitle` (`generic_data.register_importer`: 1867's comes with milestone A2's
`match_human_action`, design §5.2).

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
4. **1867 import**: waits for A2's layout and `match_human_action` (`generic_data.register_importer`).
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
multiples of the step in `[lo, 2 lo]` -- intentionally un-human, as 1830's) or its layout; the 1822 family opens
with a Stock Round, so its random opening is the empty game. `TitlePlay` (legal rows, decisions, fixed prices,
applying a row) reads the title's layout (`mg.legal_actions`, `apply_action_index`, `price_range_for_index`);
`register_title_play` is the hook if 1867's layout serves its legal rows another way (A2).

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

### 1867 outline (after A2 / A3)

1. **Layout and import** (package A2): `rl_layout_info("1867")`, `factored_legal_indices` / `apply_action_index` /
   `price_range_for_index` serving 1867 (else a `register_title_play("1867", ...)` adapter over its legal path) and
   `generic_data.register_importer("1867", ...)` on `match_human_action`. Nothing else in `generic_pg` /
   `generic_inference` is title-specific.
2. **Start files**: `scripts/build_generic_start_positions.py --title 1867 --source human_games/1867 --source
   human_games/scrape_2026_10/1867 --out-dir human_games --jobs 8` -> `start_positions_1867_{3..6}p.jsonl` and
   `midgame_positions_1867_{3..6}p.jsonl` (the `TitleSpec` templates point there; works today, see above).
3. **Supervised**: `convert-generic --title 1867`, `pretrain-generic` (default preset), the value refit; no legacy
   model to compare with, so judge by per-family top-1 against chance and the win CE against equal odds.
4. **Policy gradient**: `policy-gradient-generic --title 1867 --policy <1867 final.pth>` with the 1830 recipe at
   3-6 players (`--player-mix 3:25,4:40,5:25,6:10`, no 2-player game), `--auction-fraction 0.2 --midgame-fraction
   0.3 --random-start-fraction 0.2` (design §5.6: ~80% from human SR1 starts; 1867's auction cannot lock, so no
   `auction_unlock` and the stall cap is only a backstop), decision caps measured from 1867 games first (longer than
   1830's: mergers, loans), placement reward.
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
5. **1867 decision caps and stall policy**: unknown until 1867 games are played by a policy (random-walk games are
   no guide).

### Notes on the design documents (found during C2)

* `docs/rl_bridge_design.md` §0, §1a and §4.4 still say `MAX_SEATS = 8` (superseded by the 2026-11-13 decision:
  6), and §1a's table cites §4.4 for the seat width, which is §4.5.
* §5.4 suggests placement points `(N-1-rank)/(N-1)`; 1830's runs use per-count tables (open question 1 above).
* §5.3 lists the acting seat among the native getters the loop needs; the generic workers read it from the
  encoder's `meta[:, 5]` (one native call encodes every pending position and routes it).
* `docs/rl_bridge_work_packages.md` says to start each package from branch `gen6-mixed-count`; master has it all.

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

Data: 1867 = A3's `generic_1867_v1` (965 games of 3-6 players, 915 training / 50 validation, 540,775 / 30,938
positions; every record replays and encodes on this branch: 571,713 positions); 30 of its training games are a fixed
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
