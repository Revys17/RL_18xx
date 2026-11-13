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
policy gradient against a league, on generic inference servers. Milestone A3 (the last section: "1867: data,
starts, caps and the CPU smoke") gives 1867 its human importer, a supervised dataset, start files, legal prices on
every sampling path and an end-to-end CPU smoke. The frozen 1830 model (`model_transformer.py`) is untouched.

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
   several (each game already carries its title, and batches can mix titles).
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
multiples of the step in `[lo, 2 lo]` -- intentionally un-human, as 1830's) or its layout; the 1822 family opens
with a Stock Round, so its random opening is the empty game. `TitlePlay` (legal rows, decisions, fixed prices,
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
