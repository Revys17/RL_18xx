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
`eval_policy_only.py`; nothing trains it by policy gradient yet. The frozen 1830 model (`model_transformer.py`)
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

`LegalActions(rows, price_range, indices)` is one position's legal list: descriptor rows `(n, 14)`, each
priced row's `(lo, hi, step)` (-1 elsewhere) and the layout indices. `legal_actions(game)` builds it for a
title with an RL layout (today 1830: `factored_legal_indices`, `action_descriptors(title).gather`,
`price_range_for_index`). A title whose layout is not on master yet (1867) passes its own rows (the tests
map 1867 generator moves with a stub). `legal_actions(game, indices)` takes legal indices the caller already
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
1867 at 3-6 (`g1867_legal_moves_json` mapped to descriptor rows by a stub in the test, following
`docs/rl_descriptors.md` §8; **TODO(A2)**: switch to the 1867 layout's `legal_actions` when it lands). A
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

1. **1830 bid step**: `price_range_for_index` returns (lo, hi) without a step, so `legal_actions` uses
   1830's $5 ladder for every bid. New titles' generator ranges carry their step; the layout path should
   too (A2 / C), as `rl_descriptors.md` §9.3 asks.
2. **1867 rows**: the tests use a stub mapping; the real 1867 layout (A2) decides share sources, trade-ins
   and the city axis. The model needs nothing title-specific beyond `descriptor_entity_base` and the tile
   table.
3. **`amount` price kind** (1822 partial debt payoff) has no cell spec; `price_cells_for` raises. Fix the
   cells from a price census when the family is first used.
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
Bankrupt 169. 856 of the 4,076 games (254,558 positions) end before the game's end (the log stops; the site
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
4. **1867 import**: waits for A2's layout and `match_human_action` (`generic_data.register_importer`).
5. **No resume** of a pretraining run (the legacy one has none either).
