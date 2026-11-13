# Title-generic token encoder (RL bridge milestone B1)

*2026-11-13. Design: `docs/rl_bridge_design.md` §3 (and the user's decisions of 2026-11-13 at its top);
package B of `docs/rl_bridge_work_packages.md`. Encoder version 1 (`ENCODER_VERSION`).*

The 1830 encoder (`engine-rs/src/encoder.rs`, `encoder.py`) is a flat vector whose layout is tied to 1830's
corporations, privates and map; it stays frozen for the 1830 models. This encoder turns a position of **any**
registered title (1830, 1867, 1822, 1822PNW, 1822MX, 1822CA) into a bundle of typed tokens for a transformer:

* **hex tokens**, one per map hex, with title-generic tile, track, stop, revenue, terrain/cost and border features.
  A token in a city slot is a **pointer** to the owning corporation's entity token, not a per-corporation one-hot;
* **entity tokens**: one global token, six seat tokens (seat 0 = the player to move), the title's corporations
  (major / minor / national), companies (private / concession / minor proxy), train types and phases, each type with a
  fixed, title-independent feature schema;
* a sparse **relation list** (shares, presidencies, companies owned, trains held, standing bids, sales this round)
  for an attention bias;
* **meta** ints (title, round kind, step kind, acting entity, seats, rotation, ...) for FiLM conditioning.

What is title-specific -- entity lists, hex order and geometry, tile table, normalizers -- is derived from the title
data into an `RlSpec` per title. The vocabularies a model embeds (titles, round / step kinds, labels, terrains, phase
statuses, market zones, entity types, relation kinds) are global, append-only lists, so ids never move when a title
is added.

## Code and API

| | |
|---|---|
| `engine-rs/src/rl/encode/mod.rs` | constants, entity types, relation kinds, global vocabularies |
| `engine-rs/src/rl/encode/schema.rs` | feature names per token type (the encoder writes exactly these, in order) |
| `engine-rs/src/rl/encode/spec.rs` | `RlSpec` per title (`rl_spec_for(title)`), track features, tile table |
| `engine-rs/src/rl/encode/encode.rs` | `BaseGame::encode_tokens() -> TokenEncoding` |
| `engine-rs/src/rl/encode/py.rs` | PyO3: `rl_encode_batch`, `rl_encoder_schema`, `rl_spec_info` |
| `engine-rs/src/rl/encode/tests.rs` | cargo tests (walks of every title, vocab coverage, geometry, batching, bench) |
| `rl18xx/agent/alphazero/generic_tokens.py` | `encode_batch`, `to_torch`, `TitleGeometry`, the `TokenTrunkStub` module |
| `tests/agent/alphazero/test_generic_tokens.py` | pytest: 1830 / 1867 walks, batching, stub forward / backward, benchmark |

The encoder is a submodule of milestone A1's RL bridge module (`engine-rs/src/rl/`, `docs/rl_descriptors.md`):
registered by one line in `rl/mod.rs`, its Python functions added by `rl::py::register`. Nothing else in the crate
depends on it. Its entity, hex and tile orders are A1's descriptor vocabularies (`rl::Vocab`), so a descriptor's
`ent0` / `ent1` / `via` / `hex` / `tile` index addresses a token directly (see "Descriptors" below).

```python
import engine_rs
arrays = engine_rs.rl_encode_batch(games, dedupe_maps=False)   # any titles / player counts in one call
schema = engine_rs.rl_encoder_schema()                         # widths, feature names, vocabularies
info = engine_rs.rl_spec_info("1867")                          # hex order + geometry, entity names, tile table, normalizers
```

### Batch arrays

H, E, R are the batch maxima (hexes, entities, relations); padding is masked.

| key | shape | dtype | content (padding) |
|---|---|---|---|
| `hex_feat` | (M, H, 106) | f32 | hex features (0) |
| `hex_ptr` | (M, H, 8) | i32 | entity index of the token in each slot; -1 empty slot / no slot, -2 a token of no corporation (1867's neutral placeholders) |
| `hex_res` | (M, H, 8) | i32 | entity index of the corporation a slot is reserved for (home reservations, 1867's CN reserve, 1822's city / tile reservations); -1 none |
| `hex_slot_city` | (M, H, 8) | i32 | the city (0..) each slot belongs to; -1 = no such slot |
| `hex_ids` | (M, H, 2) | i32 | (label id, tile id) -- see Vocabularies |
| `hex_mask` | (M, H) | bool | real hexes |
| `map_index` | (B,) or None | i64 | with `dedupe_maps`: the hex block (row of the M) of each position |
| `ent_type` | (B, E) | i32 | entity type id (0 = padding, incl. absent seats) |
| `ent_feat` | (B, E, 53) | f32 | per-type features packed from column 0, the rest 0 |
| `ent_ptr` | (B, E, 4) | i32 | entity pointers (-1 none) |
| `ent_hex` | (B, E, 2) | i32 | hex pointers (-1 none) |
| `ent_mask` | (B, E) | bool | `ent_type != 0` |
| `rel` | (B, R, 3) | i32 | (src entity, dst entity, kind); padding (0, 0, 0) |
| `rel_val` | (B, R) | f32 | relation value |
| `rel_mask` | (B, R) | bool | real relations |
| `meta` | (B, 8) | i32 | see Meta |

Without `dedupe_maps` M = B. With it, positions whose hex arrays are bit-identical share one block (as 1830's
`encode_games(dedupe_maps=True)`): pointers name corporations, whose entity indices do not depend on the seat rotation,
so the positions of one map share a block whoever is to move. A search's leaves mostly share their tree's map.

Every float is clipped to [-16, 16] and NaN / inf become 0 (`FEATURE_CLIP`); the normalizers keep almost everything
in [-1, 2], cash and net worth of a rich late-game player in 1830 reach ~8.

## Entities

### Layout (per title, fixed)

Entity index: `0` global, `1..=6` seats, then the title's corporations, companies, train types and phases, each in
title-data order (`rl_spec_info(title)["entity_names"]`, `["entity_offsets"]`).

| title | layout | hexes H | entities E | corporations | companies | train types | phases | tiles |
|---|---|---|---|---|---|---|---|---|
| 1830 | pointy | 93 | 33 | 8 | 6 | 6 | 6 | 46 |
| 1867 | flat | 94 | 54 | 25 (8 major, 16 minor, CN national) | 6 (incl. the hidden "3") | 9 | 7 | 68 |
| 1822 | flat | 211 | 127 | 40 (30 minors, 10 majors) | 61 (21 P, 10 C, 30 M) | 12 | 7 | 62 |
| 1822PNW | pointy | 163 | 102 | 31 | 45 | 12 | 7 | 57 |
| 1822MX | pointy | 130 | 108 | 32 | 49 | 13 | 7 | 53 |
| 1822CA | flat | 279 | 138 | 41 | 70 | 13 | 7 | 97 |

Entity types (`ENTITY_TYPES`): 0 pad, 1 global, 2 player, 3 major, 4 minor, 5 national, 6 private,
7 concession, 8 minor_proxy, 9 train_type, 10 phase. Corporation kinds come from `CorpType`; company kinds are
"private" except in the 1822 family, where `C<n>` is a concession (linked to the major it founds) and `M<x>` is the
bid-box proxy of minor `<x>` (linked to it).

Closed corporations and companies keep their tokens (with `closed` = 1): indices never move during a game.

### Seats

Seats are the players in seating order (`BaseGame.players`, the descriptor vocabulary's players), rotated so that
seat 0 is the player deciding the current move -- the mover rule of 1830's canonical encoding
(`acting_player_id_pub`: the active player, or the active corporation's president), and for the 1822 family Ruby's
`acting_for_entity` (`g1822_acting_player`'s logic: a company's owner, the NdeM's chooser, ...). With nobody to move
(a finished game) the rotation is 0. `meta[5]` is the rotation: seat k is player `(k + rotation) mod n` of the seating
order. (1830's legacy encoder rotates the players sorted by id; every game `titles.new_game` builds numbers its
players 1..n in seating order, so the two agree there.) Seats beyond the player count are padding (type 0, masked).
Every per-seat column (`pct_seat_k`, `bid_seat_k`) and seat pointer uses the rotated seats. `MAX_SEATS` stays 6.

### Feature schemas

Money is divided by **sc** = the game's starting cash per player; share-market prices by the title's highest market
price; stop revenues by its highest stop revenue (catalog tiles, preprinted hexes, offboard tiers); lay and border
costs by its highest such cost; train prices by its dearest train; company revenue by its highest company revenue.
The values are in `rl_spec_info(title)["normalizers"]`.

**global** (29)

| feature | value |
|---|---|
| bank_cash | bank / initial bank |
| bank_broken, game_end_triggered, final_turn_set, finished | flags |
| turn | turn / 20 |
| round_num, total_ors | OR k of the set / ORs in the set; ORs in the set / 3 |
| phase_index | phase / (phases - 1) |
| num_seats, cert_limit, starting_cash | n / 6; limit / 30; sc / initial bank |
| loans_remaining, loan_value, loan_interest | / title loans; / sc; / loan value (0 without loans) |
| market_pool_limit | percent / 100 |
| tiles_laid_this_turn, tokens_placed_this_turn, route_revenue | / 2; count; this turn's revenue / sc |
| auction_discount, auction_active | / sc; a lot (waterfall auction, single-item lot, 1867 stock-round minor auction) is running |
| sold_this_turn, bought_this_turn, parred_this_turn, consecutive_passes | stock-round turn state; passes / seats |
| events_fired, trainless_majors, pending_tokens, crowded_corps | / 8, / 4, / 2, / 2 |

**player** (16): present, is_mover, cash (/ sc), certs (/ cert limit, counted as the engine's
`num_certs_internal`), cert_room, priority_deal, order_pos (stock / auction order from the priority deal / (n - 1)),
committed_bids (/ sc), net_worth (cash + shares at market + company face values, / sc / 4), share_value (/ sc / 4),
company_value (/ sc), bankrupt, passed, presidencies (/ 4), sold_corps (corporations sold this stock round / 4), debt
(1822 player debt / sc).

**corporation** (major / minor / national, 53)

| feature | value |
|---|---|
| started, floated, operated, closed | flags |
| cash | / sc |
| share_price, par_price | / max market price |
| market_row, market_col | / rows, / columns |
| zone_par .. zone_endgame (9) | the current cell's types over `MARKET_ZONES` (a 1867 cell can carry several) |
| ipo_units, market_units, other_units | share units in the treasury / IPO, the market pool, other holders (another corporation, 1822's tax haven), / total units |
| total_units, president_units | units / 10; president cert units / total |
| share_unit_pct, float_pct, incremental_cap | / 100; / 100; flag |
| tokens_total, tokens_placed, next_token_price | / 8; placed / total; / max token price |
| loans, max_loans | / the class's max loans; max loans / 10 |
| trains, train_room, train_limit | / 4 (the phase's limit for its class) |
| or_order_pos, in_or_order, operated_this_or, is_acting | position in the live OR's order / (n - 1) |
| has_president, max_ownership, market_price_ipo, startable | startable = `corporation_startable` in this phase |
| home_reserved | its home reservation is on the map |
| destination_pending, exchange_tokens | 1822: destination token not placed; exchange tokens / 4 |
| pct_seat_0 .. pct_seat_5, pct_market, pct_ipo | percent / 100 |

**company** (private / concession / minor proxy, 28): value (/ sc), revenue (/ max; 1822's current revenue), closed,
owner_seat, owner_corp, unowned, auctionable, up_for_auction (remaining in the waterfall, the single-item lot, or in
an 1822 bid box), min_bid (/ sc: the waterfall's `min_bid_for`; else the high bid + the title's bid step, or the
company's `min_bid`), high_bid, num_bids (/ seats), discount, no_buy, ability_used, ability flags (blocks_hexes,
tile_lay, teleport, exchange, no_buy, shares, hex_bonus, close; 1822 family: tile lay / teleport from the family
ability table), bid_seat_0..5 (/ sc).

**train type** (17): distance (/ 8, capped), unlimited_distance (>= 99: 1822's E, P+), multiplier (/ 2), price
(/ max), depot_count (/ base count; unlimited supply: 1 while any), depot_log (log1p / log1p(20)), pool_count (/ 4),
unlimited_supply, base_count (/ 10; unlimited 1), held_count (/ 8), rusted, obsolete (its phase reached), available
(next up, or `available_on` reached with cards in the depot), next_up, max_discount (best trade-in / price), events
(/ 3), variant (a type without cards of its own: 1822's '2', 'E').

**phase** (22): current, past, index (/ (n - 1)), train_limit, minor_train_limit (/ 4), operating_rounds (/ 3), tiles
yellow / green / brown / gray, a flag per `PHASE_STATUSES` entry, st_other.

### Entity pointers (`ent_ptr`, 4 columns: owner, link, phase_a, phase_b)

| type | owner | link | phase_a | phase_b |
|---|---|---|---|---|
| global | acting entity | priority-deal seat | current phase | current lot (company or corporation) |
| corporation | president seat | 1822PNW: associated major (of a minor); 1822: the founding concession (of a major) | | |
| company | owning seat or corporation | 1822: the minor a proxy stands for / the major a concession founds | | |
| train type | | | rusts-on phase | obsolete-on phase, else available-on phase |
| phase | | | train type whose purchase starts it | |

`ent_hex` (home, destination): corporation home (title data; 1822PNW's P20 override; else the hex of its first placed
token, which covers 1867's chosen homes) and 1822 destination; company: the first hex its abilities name.

## Hexes

Hex order = map order (`GameTitle::hex_definitions()` = `BaseGame.hexes` = the descriptor vocabulary's `hex`), so a
descriptor's `hex` addresses the hex token directly. (1830's frozen action layout keeps its historical hex order for
its indices; its descriptors already map into map order.)
`rl_spec_info(title)` carries `hex_coords`, `hex_axial` (H, 2) (axial (q, r); hex distance = (|dq| + |dr| + |dq + dr|) / 2,
pointy maps from doubled-width, flat from doubled-height coordinates), `hex_neighbors` (H, 6) (spec hex index across
each edge, -1 off-map; the engine's adjacency) and `hex_pos` (H, 2) (normalized x across, y down).

**Hex features** (106)

| columns | features |
|---|---|
| 0-1 | pos_x, pos_y |
| 2-7 | tile color one-hot: white, yellow, green, brown, gray, red |
| 8-10 | tile_laid, preprinted (the map's own tile: city / town / offboard / path / preprinted DSL), has_label |
| 11-16 | rotation one-hot (laid tiles only; track bits are already rotated) |
| 17-21 | cities / 6, slots / 6, free_slots / 6, towns / 2, offboard |
| 22-29 | max city revenue, max town revenue, offboard revenue in the current phase, its yellow / green / brown / gray tiers (undefined tiers fall through as `phase_revenue` does), offboard_grouped -- revenues / max stop revenue |
| 30 | upgrade_cost / max lay cost |
| 31-37 | terrain flags over `TERRAINS` (mountain, water, hill, swamp, forest, river), terrain_other |
| 38-43 | impassable wall per edge |
| 44-49 | unpaid border crossing cost per edge / max lay cost (1867's waters, 1822's live borders) |
| 50-98 | track (49): 15 direct edge-edge pair bits, 15 edge-edge bits joined through a stop or junction, per edge whether it reaches a city / town / offboard (6 x 3), junction flag |
| 99-105 | blocked (a live private blocks lays here), hex_bonus (open companies' route bonus here / max stop revenue), tokens / 6, neutral_tokens / 6, extra_tokens / 2 (1822CA's tokens outside slots), reserved_slots / 6, laid_this_turn |

The track block replaces 1830's per-city edge bits: "edges i and j share a stop" identifies an OO city, a three-city
Montreal or 1822's six-city London without a per-city slot dimension. Edge numbers are relative to the title's layout
(pointy vs flat); the title embedding tells them apart.

**Slots.** Each city's slots, then its extra tokens, in city order, up to `S_MAX` = 8 per hex (the most on any
registered map is 6; the tests assert no tile in any catalog exceeds 7 and no walk truncates). `hex_slot_city` gives
each slot's city.

**Tile ids** (`hex_ids[..., 1]`): 0 no laid tile (blank, preprinted), `k + 1` = `tile_ids[k]`
(`rl_spec_info(title)["tile_ids"]`, the title's tile manifest order `tile_counts()` = the descriptor vocabulary's
`tile`), `len + 1` a tile not in the list.
`rl_spec_info(title)["tile_feat"]` (T, 64) gives each catalog tile at rotation 0: color, label flag, stops, revenues,
upgrade cost, the same 49 track features and its supply -- for tile embeddings that generalize across titles (the
LayTile head's P(tile | hex) in B2).

## Relations (`rel`)

| kind | src -> dst | value |
|---|---|---|
| 1 shares | seat or corporation -> corporation | percent held / 100 |
| 2 president | seat -> corporation | 1 |
| 3 owns_company | seat or corporation -> company | 1 |
| 4 trains | corporation -> train type | trains of that type / 4 |
| 5 bid | seat -> company (waterfall, single-item, 1822 bid boxes) or corporation (1867's stock-round minor auction) | price / sc (may be 0) |
| 6 sold | seat -> corporation sold this stock round | 1 this turn, 0.5 earlier |

Treasury, IPO and market-pool holdings are corporation features, not relations. Relations come in a fixed order
(corporations in title order, then companies, bids, sales), so an encoding is deterministic. The longest list seen in
the walks: 80 (1830), 69 (1867), 52 (1822CA).

## Meta

`meta[b]` = (title id, round kind, step kind, acting entity index or -1, number of seats, rotation, finished, OR k of
the set or 0).

* **Round kind**: the live round (1822 family: its own round, `Choices` and PNW's `Merger` included).
* **Step kind**: the step the current decision belongs to. 1822 family: Ruby's `active_step`. 1830 / 1867: read off the
  round state without the step machinery's `&mut` evaluation: MajorTrainless, pending home tokens, discards and a
  pending DH teleport first, then the auction / stock / merger round's step or the operating turn's program counter;
  1867's in-stock-round minor auction is its own id (`BuySellParSharesViaBid`). On 9,318 positions of 1867 walks it
  equals the step of the generator's legal moves every time (cargo test `random_walks_1867`).

## Vocabularies

| list | ids |
|---|---|
| `TITLES` | 1830, 1867, 1822, 1822PNW, 1822MX, 1822CA -> 1..6 (0 unknown) |
| `ROUND_KINDS` | none, Auction, Stock, Operating, Merger, Choices |
| `STEP_KINDS` | none + the shared `StepKind` names + the 1822 `g1822::Step` names (a shared name is one id) + `BuySellParSharesViaBid` (38) |
| `LABELS` | 0 none, 1 unknown, then B, NY, OO, M, O, T, Y, BM, C, EC, L, S, Por, Sea, MC, A, CP, Q, W, YP |
| `TERRAINS` | mountain, water, hill, swamp, forest, river (+ other) |
| `PHASE_STATUSES` | can_buy_companies, export_train, can_buy_trains, can_acquire_minor_bidbox, can_par, full_capitalisation, minors_green_upgrade, minor_float_phase1/2/3on, can_convert_concessions (+ other) |
| `MARKET_ZONES` | par, par_1, par_2, convert_range, max_price, no_cert_limit, unlimited, multiple_buy, endgame |

`vocab_covers_every_title` (cargo) fails when a title's tiles, preprinted hexes, phases or market use a label,
terrain, status or zone missing from these lists: a new title appends to them (never reorders) and bumps nothing else.

## Throughput (single thread, CPU, release build)

Positions from random walks, 4 players, encode only (cargo `bench_encoders`, ignored by default):

| title | token encoder | 1830 encoder (`encode_canonical`) |
|---|---|---|
| 1830 | 24-27 us (~38k/s) | 18-19 us (~55k/s) |
| 1867 | 36-41 us (~26k/s) | -- |
| 1822 | 78-110 us (~10k/s) | -- |
| 1822CA (5p) | ~100 us (~10k/s) | -- |

So on 1830 the token encoder costs ~1.4x the frozen encoder (it writes ~10k hex floats against ~3.6k). Stacking a
batch costs ~3 us per position (1830 / 1867, plain) and ~8-10 us with map de-duplication (hashing the hex block);
1822 maps are 2-3x larger. Through Python (`rl_encode_batch` incl. numpy, `pytest -m benchmark`): 1830 25-28k
positions/s against `encode_games`' 42-50k/s; 1867 20-22k/s. The 1822 family is slower because its step kind and mover
come from Ruby's `active_step` (which evaluates step actions) and its maps and entity lists are 2-3x larger; it is
not on a hot path yet (its bridge comes after 1867's). Timings vary +-15% on the shared machine.

## Tests

```bash
# cargo (the PyO3 test binary needs the venv's libpython; docs/verification_rituals.md)
PY="$(pwd)/.venv/bin/python"; LIBDIR="$("$PY" -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')"
(cd engine-rs && PYO3_PYTHON="$PY" LD_LIBRARY_PATH="$LIBDIR" cargo test --release --no-default-features rl::encode)
(cd engine-rs && PYO3_PYTHON="$PY" LD_LIBRARY_PATH="$LIBDIR" cargo test --release --no-default-features rl::encode::tests::bench -- --ignored --nocapture)
CUDA_VISIBLE_DEVICES= uv run pytest tests/agent/alphazero/test_generic_tokens.py
CUDA_VISIBLE_DEVICES= uv run pytest -m benchmark -s tests/agent/alphazero/test_generic_tokens.py
```

Cargo: random walks of 1830 at 2-6 players, 1867 at 3-6 and each 1822-family title at both ends of its 3-6 range,
encoding every position and checking shapes, finite and clipped values, entity types and padding, pointer validity
(token / reservation pointers name corporations, owners name seats or corporations, train / phase pointers name
phases / train types), every token on the map appears exactly once among the pointers, share holdings add up to each
corporation's certificates, cert counts equal the engine's, seat 0 is the mover, relations valid and unique, the
1867 step kind agrees with the generator, determinism (twice and on a clone); vocabulary coverage of every title;
geometry (axial coordinates vs the engine's adjacency, both ways); the 1822 company kinds; mixed-title batch padding
and map de-duplication; seat rotation. pytest: the same on 1830 / 1867 walks through the Python API, plus
`TokenTrunkStub` forward / backward on a mixed 1830 + 1867 batch and padding invariance (a position alone or in a
padded mixed batch gives the same value logits).

## The PyTorch stub

`generic_tokens.TokenTrunkStub` exists to prove the arrays are consumable; it is not the model (B2 is): a per-type input
projection of `ent_feat` (+ type embedding + the owner pointer's embedding), hex features + label embedding with each
slot's token and reservation pointer resolved to the corporation's embedding, entity self-attention with a relation
bias (a learned per-head constant and value scale per relation kind and direction), hex self-attention with a per-head
bias per clipped hex distance (from `hex_axial`), entity -> hex cross-attention biased where the corporation holds a
token, FiLM on embedding(title) + embedding(round kind) + embedding(step kind), and per-seat value logits (six, absent
seats -1e4). No policy or price head.

## Descriptors -> tokens

A1's descriptor fields (`docs/rl_descriptors.md` §2, §4) index the tokens without a lookup table:

| descriptor field | token |
|---|---|
| `hex` | hex token `hex` (same map order) |
| `city` | the slots of that hex whose `hex_slot_city` is `city` |
| `tile` | tile table row `tile` (`tile_feat`; `hex_ids` tile id = `tile + 1`) |
| `ent0` / `ent1` of kind corporation / company / train | entity `base[kind] + index`, `base` = `rl_spec_info(title)["descriptor_entity_base"]` (`e_corp0`, `e_company0`, `e_train0`) |
| `via` (a company) | entity `base["company"] + via` |
| a player (seating-order index p) | seat token `1 + (p - rotation) mod n` (`RlSpec::descriptor_entity`) |

Pinned by `token_orders_are_the_descriptor_vocabularies` (every title's orders equal `rl::Vocab::for_title`, and
`BaseGame.hexes` is that hex order), `descriptors_index_the_tokens` (every one of 1830-v1's 26,537 descriptors names
the token of its vocabulary entry) and pytest `test_descriptors_address_tokens` (legal 1830 indices of walk positions:
their hexes, cities and entities are real tokens of the same encoding).

## Open points for B2 and later

* The per-(title, entity) learned id embedding of design §3.3 is left to the model: an entity's index is stable within a
  title, so `(title, index)` keys it.
* Not encoded yet: 1822 player debt interest, the tax haven's holdings as a holder token (its shares count as
  `other_units`), 1822PNW's builder cubes / hex assignments, 1822MX's NdeM state, 1822CA's sawmill / grain extras, a
  history of recent actions. The design's per-type "extra" slice for such title extras is not allocated; adding one
  is a version bump.
* The 1830 / 1867 step kind is read off the round state, not from the step machinery; it agreed with the 1867
  generator on every walk position, and 1830's legacy encoder uses only the round type. If a mismatch appears,
  `step_offers()` (which needs `&mut self`) is the oracle.
