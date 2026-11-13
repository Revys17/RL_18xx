# RL action descriptors: the schema

*Milestones A1 (the schema, the frozen 1830 layout) and A2 (the 1867 layout and its index API) of
the title-generic RL bridge (2026-11-13; design `docs/rl_bridge_design.md` §2, work packages
`docs/rl_bridge_work_packages.md` package A). Code: `engine-rs/src/rl/`,
`rl18xx/agent/alphazero/action_descriptors.py`, `rl18xx/agent/alphazero/titles.py`.*

Every title gets **one index space** (its *layout*): a policy index per distinct legal move, the
price of a price-bearing move excepted (prices go to price heads). Every index of every layout also
gets a **descriptor**: one row of small integers that says what the index means in
title-independent terms. A generic policy head never keeps one output per index; it scores the
legal indices of a state from their descriptors, so the same weights score "buy a share of X" or
"lay tile T on hex H" in any title, and a layout can be any size (1822CA's tile lays alone are
162k indices).

Two layouts exist: **1830 v1**, the frozen flat layout of `action_index.rs::build_layout` (26,537
indices) with a descriptor for every index, and **1867 v1** (38,875 indices), a registry layout
built from title data whose index function maps the moves of 1867's legal-move generator
(`g1867/legal.rs`) to indices (§8). The 1822 family has no layout yet ("not bridge-ready"); §9 lists
what the schema cannot express for it.

## 1. Where it lives, how to read it

| What | Where |
|---|---|
| Schema, global vocabularies | `engine-rs/src/rl/descriptor.rs` |
| Block registry, descriptors per index, fingerprint | `engine-rs/src/rl/layout.rs` |
| 1830's frozen layout as a registry | `engine-rs/src/rl/factored_layout.rs` |
| 1867's layout and its move -> index function | `engine-rs/src/rl/g1867.rs` (tests: `rl/g1867/tests.rs`) |
| Index API of generator titles (legal indices, decode, apply, price range) | `engine-rs/src/rl/generator.rs`: Rust `BaseGame::rl_indexed_moves` / `rl_move_for_index` / `rl_apply_index`; Python `BaseGame.rl_legal_indices`, `rl_legal_moves_json`, `rl_decode_index`, `rl_apply_index`, `rl_price_range` |
| Python surface | `engine_rs.rl_layout_info(title)`, `engine_rs.rl_descriptor_vocab()`, `engine_rs.rl_title_status(title)` (`engine-rs/src/rl/py.rs`) |
| Python wrapper (cached, numpy) | `rl18xx.agent.alphazero.action_descriptors.action_descriptors(title)` |
| Readiness per title | `GameTitle::rl_layout_kind` / `rl_layout_ready` / `rl_encoder_ready` (`engine-rs/src/title/mod.rs`) |

```python
from rl18xx.agent.alphazero.action_descriptors import action_descriptors

d = action_descriptors("1830")            # RuntimeError "not bridge-ready" for 1822*
legal = game.factored_legal_indices()     # the 1830 path's legal indices (sorted);
                                          # 1867: game.rl_legal_indices()
rows = d.gather(legal)                    # int32 [n_legal, 14], columns d.fields
fam = rows[:, d.col("family")]            # ids into d.families
d.describe(139 + 46 * 6 + 2 * 6 + 5)      # {'family': 'LayTile', 'hex': 'I1', 'tile': '16', 'rot': 5, ...}
```

`rl_layout_info(title)` returns: `title`, `version`, `kind`, `total`, `fingerprint` (16 hex
digits), `fields`, `descriptors` (int32 numpy `[total, len(fields)]`), `blocks` (name, family,
family_id, offset, size, price_kind, flags, axes with their entries), `vocab` (the title's
corporation / company / train / hex / tile lists), and the global vocabularies of
`rl_descriptor_vocab()` (`families`, `map_families`, `ent_kinds`, `price_kinds`,
`price_cell_types`, `choices`, `flags`, `none`).

## 2. The descriptor: fields and types

All fields are int32; **-1 means "none"** in every field (`NONE`). A family uses only some fields;
the others stay -1.

| # | Field | Meaning | Values |
|---|---|---|---|
| 0 | `family` | The kind of decision | id in the global family vocabulary (§3) |
| 1 | `ent0_kind` | Kind of the first entity the move refers to | id in `ent_kinds`: player 0, corporation 1, company 2, train 3 |
| 2 | `ent0` | That entity | index in the title vocabulary of its kind (§4) |
| 3 | `ent1_kind` | Kind of the second entity | as `ent0_kind` |
| 4 | `ent1` | The second entity | as `ent0` |
| 5 | `via` | The company whose **ability** performs the move (1830: DH's teleport, CS's lay, MH's exchange) | company vocabulary index; -1 when the acting player or corporation acts itself |
| 6 | `hex` | Hex | hex vocabulary index (map order) |
| 7 | `city` | City on that hex's tile | 0-based city index of the tile on the hex |
| 8 | `tile` | Tile laid | tile vocabulary index |
| 9 | `rot` | Rotation | 0-5 |
| 10 | `param` | The family's number | Par: par price in $; BuyShares / SellShares / RedeemShares / IssueShares: percent moved; Choose: the option's position when options are a state list (§9) |
| 11 | `choice` | A categorical option | id in the global choice vocabulary (§3): share source, dividend kind, a named choice |
| 12 | `price_kind` | How the move's price is expressed (§6) | id in `price_kinds`, -1 = no price parameter |
| 13 | `flags` | Bits | `dead` 1, `train_from_state` 2 (§3) |

Two conventions:

- **The acting entity is never in the descriptor.** It is the state's current entity (the
  operating corporation, the player to act), which the encoder already gives the model. `via` is
  the one exception: a company acting through its ability, because the legal set can hold the
  same move by the corporation and by its company (1830's CS lay on B20 vs the corporation's own
  lay there).
- **Entities are typed slots, not per-family columns.** `ent0`/`ent1` say *which* entity token the
  head reads (`tok[a]`, `tok[b]` in design §2.6): the encoder's entity sequence is
  `[players..., corporations..., companies..., train types..., ...]`, and token =
  `base[kind] + index`. Train types are entities (design §3.1).

## 3. Global vocabularies (title-independent, append-only)

A trained model embeds these ids. **Never renumber or reuse an id; append new entries at the end**
(`descriptor.rs`, pinned by `global_vocab_ids_are_pinned`). Appending does not change an existing
layout's fingerprint.

**Families** (`family`; the map families are the ones a head factors over the map, §5):

| id | family | refers to | used by |
|---|---|---|---|
| 0 | `Pass` | - | all |
| 1 | `Bid.company` | ent0 company; price `bid` | 1830 waterfall, 1867 single-item auction, 1822 bid boxes |
| 2 | `Bid.corporation` | ent0 corporation; price `bid` | 1867 SR minor bids |
| 3 | `Par` | ent0 corporation, param par $ | all |
| 4 | `BuyShares` | ent0 corporation, choice source, param percent where bundles exist | all |
| 5 | `SellShares` | ent0 corporation, param percent | all |
| 6 | `RedeemShares` | param percent (the acting corporation's own shares) | 1867, 1822 |
| 7 | `IssueShares` | param percent | 1822 |
| 8 | `ExchangeShares` | via company, ent0 corporation, choice source | 1830 MH -> NYC |
| 9 | `PlaceToken` (map) | hex, city; via for company tokens | all (also 1867 SR home tokens) |
| 10 | `RemoveToken` (map) | hex, city | 1867 ReduceTokens, 1822MX NdeM |
| 11 | `LayTile` (map) | hex, tile, rot; via for company lays | all |
| 12 | `BuyTrain.depot` | ent0 train type (or `train_from_state`) | all |
| 13 | `BuyTrain.pool` | ent0 train type (a discarded train at face) | all |
| 14 | `BuyTrain.exchange` | ent0 new train type, ent1 traded train type | 1830 D, 1867 8 / 2+2 / 5+5E |
| 15 | `BuyTrain.corp` | ent0 selling corporation, ent1 train type; price `train_transfer` | all |
| 16 | `DiscardTrain` | ent0 train type | all |
| 17 | `Dividend` | choice dividend kind | all |
| 18 | `BuyCompany` | ent0 company; price `company_purchase` | 1830, 1867 |
| 19 | `RunRoutes` | - (the router's optimum) | all |
| 20 | `Bankrupt` | - | 1830, 1822 |
| 21 | `Merge` | ent0 corporation | 1867, 1822 (minor acquisition, PNW mergers, MX NdeM) |
| 22 | `Convert` | - | 1867 |
| 23 | `Choose` | choice (or param, §9) | 1867 nationalize, 1822 family |
| 24 | `AcquireCompany` | ent0 company | 1822 |
| 25 | `ChooseAbility` | via company, choice | 1822 |
| 26 | `HexToken` (map) | hex, choice token type | 1822 destination tokens |
| 27 | `Assign` | via company, hex or ent0 corporation | 1822PNW, 1822CA |
| 28 | `PayoffPlayerDebt` | - | 1822 |
| 29 | `PayoffPlayerDebtPartial` | price `amount` | 1822 |

**Entity kinds** (`ent0_kind`, `ent1_kind`): 0 `player`, 1 `corporation`, 2 `company`, 3 `train`.

**Price kinds** (`price_kind`, §6): 0 `bid` (cells "Bid"), 1 `train_transfer` ("BuyTrain"),
2 `company_purchase` ("BuyCompany"), 3 `amount` (cells not fixed yet), 4 `bid_range` ("BuyTrain"
cells for a bid: range-relative, with an atom at the all-in top; since `1867-v1`, §8 price census).

**Choices** (`choice`): 0 `source:ipo`, 1 `source:market`, 2 `dividend:payout`,
3 `dividend:withhold`, 4 `dividend:half`, 5 `choose:nationalize`, 6 `train_source:depot`,
7 `train_source:pool` (6-7 since `1867-v1`: where a trade-in's new train comes from). A title's
named choices are appended when its layout lands (1822: `choose:double`, ...).

**Flags** (`flags`, bits):

- `dead` (1): the index is never legal. Only in frozen layouts: slots the index function never
  produces (1830: the 13 extra price slots of every cross-corporation train, the "max" slot of
  every BuyCompany; 630 indices). A head can mask them once.
- `train_from_state` (2): the train is decided by the state, not by the index (1830: the single
  depot slot = the depot's next train; the D trade-in = which owned train is traded).

## 4. Title vocabularies

Entity, hex and tile fields index lists that come from the title data, in its order
(`layout::Vocab`; `rl_layout_info(title)["vocab"]`):

| Vocabulary | Order | 1830 | 1867 |
|---|---|---|---|
| `corporation` | `GameTitle::corporations()` (every kind: majors, minors, nationals) | PRR NYC CPR B&O C&O ERIE NYNH B&M | 8 majors, 16 minors, the CN (25) |
| `company` | `GameTitle::companies()` (auctionable or not; the 1822 family's minors M1.. and concessions C1.. are companies too) | SV CS DH MH CA BO | C&SL NFB MB QB SCT and the hidden "3" |
| `train` | `GameTitle::trains()` names, variants included (1822's "2" is the L card's variant) | 2 3 4 5 6 D | 2 3 4 5 6 7 8 2+2 5+5E |
| `hex` | `GameTitle::hex_definitions()` = `BaseGame.hexes` order | 93 hexes, map order | 94 hexes |
| `tile` | `GameTitle::tile_counts()` (the title's tile manifest) | 46 tiles | 68 tiles |
| players | the game's seating order (seat index; no 1830 / 1867 family refers to a player) | - | - |

These are **not** the frozen layout's orders: 1830's action layout keeps its historical hex and
tile orders (`action_hex_order`, `action_tile_order`), and its descriptors map them into the map /
manifest order. A descriptor's `hex` is therefore the hex token's position in the encoder's
natural hex order, whatever order the layout's indices run in. The generic encoder (package B)
must order its hex and entity tokens by these vocabularies (or map through them); the vocab lists
are part of the layout fingerprint.

## 5. How a policy head consumes descriptors

1. **Only legal indices are scored.** A state's legal index list (from the 1830 path,
   `factored_legal_indices`; for new titles, the generator's moves mapped to indices) is gathered
   into descriptor rows: `rows = d.gather(legal)`. Nothing is computed per layout index; cost
   scales with the legal-move count (usually under 200, tile-lay states up to a few thousand).
2. **Non-map families** score each row:
   `logit = MLP(trunk ⊕ E_family[family] ⊕ tok(ent0) ⊕ tok(ent1) ⊕ E_company[via] ⊕ E_choice[choice] ⊕ f(param))`,
   with a learned "none" embedding wherever a field is -1, and `param` fed as a scaled number
   (Par: $ / the title's max market price; percents / 100).
3. **Map families** keep the autoregressive factorization over the map:
   - `LayTile`: **P(hex) · P(tile | hex) · P(rot | hex, tile)**, each factor a softmax over the
     values that occur among the legal lays only (the legal hexes; for a hex, its legal tiles; for
     a (hex, tile), its legal rotations). `hex` indexes the hex tokens; `tile` indexes a tile
     table whose embeddings come from **tile features** (color, exits, stops, revenue, label), so
     tiles generalize across titles; `rot` 0-5 is relative to the hex's layout (pointy / flat).
     A company lay (`via` set) is the same factorization conditioned on the company.
   - `PlaceToken` / `RemoveToken`: P(hex) · P(city | hex). `HexToken`: P(hex) (· the token type).
4. **Between blocks**: each family's logits are combined into one softmax over the whole legal set
   (a map family contributes log P(hex) + log P(tile|hex) + log P(rot|hex,tile) + a learned
   family bias as its index's logit), so the training target stays a distribution over the legal
   indices (sparse `pi`, design §5.1) and the 1830 per-block normalization is a special case.
5. **Masks**: a legal index is never `dead` (tested); `flags` can still be fed as features
   (`train_from_state` tells the head the train comes from the state's depot).

The 1830 index space feeds the generic model through these descriptors without re-indexing: the
model's output for index i is computed from row i, so 1830 data (`pi` over 1830 indices) trains
it directly (design §4.4's 1830 check).

## 6. Prices: price heads, never a fixed price menu

A price-bearing move is **one** index. Its descriptor's `price_kind` says how the price is
expressed; the legal range comes from the state:

- **Legal range**: `[lo, hi]` and its step are read from the state whenever the index is legal —
  1830: `price_range_for_index` (the factored enumeration), new titles: the generator's
  `PriceRange { lo, hi, step }`. They are never part of the layout.
- **Price head**: for each legal priced index, a head conditioned on the same descriptor inputs
  (`trunk ⊕ E_family ⊕ tok(ent0) ⊕ tok(ent1)`, design §2.7) emits `NUM_CELLS` (24) logits; the
  cells are `price_pmf`'s for the kind's cell type (`price_cell_types`):
  - `bid` -> "Bid": a ladder of 20 atoms `lo + step·j` plus one bin (1830: step = the title's
    `bid_price_step`, $5);
  - `train_transfer` -> "BuyTrain": atoms at lo, hi, hi-1 plus 20 bins;
  - `company_purchase` -> "BuyCompany": atoms at lo, hi plus 8 bins;
  - `amount` (1822 partial debt payoff): cells to be fixed from a human price census when the
    family is first used;
  - `bid_range` -> "BuyTrain" cells for a bid: atoms at lo, hi, hi-1 plus 20 bins (1867 minor
    bids: a quarter go all-in, §8 price census).
  `P(price) = softmax(nonempty cell logits)[cell] / |cell|` (`price_pmf.rs`), exactly as 1830's
  head; MCTS progressive widening over prices is unchanged.
- **One call for every layout: `BaseGame.rl_price_range(idx)` -> `(lo, hi, step)`** (None for an
  index without a price; RuntimeError for an illegal index). How to read it is the layout's
  `price_units` (`rl_layout_info(title)["price_units"]`, `ActionDescriptors.price_units` /
  `.price_grid`):
  - **"step" (generator layouts, 1867)**: the move's own `PriceRange`; legal prices are exactly
    `lo + step·k <= hi` (1867: $5 for both auctions, Ruby `MUST_BID_INCREMENT_MULTIPLE`; $1 for
    trains and companies; never a title constant). The cells are taken in step units, `k = (price
    - lo) / step` over `PriceCells(cell_type, 0, (hi - lo) / step, 1)`, so every cell holds legal
    prices only (`bid`'s 20 atoms are the same ladder as `PriceCells("Bid", lo, hi, step)`).
  - **"price" (1830)**: `(lo, hi)` exactly as `price_range_for_index` (a depot train's degenerate
    `lo == hi` included), `step` = the title's `bid_price_step` ($5) for a bid and 1 otherwise;
    every integer in `[lo, hi]` is legal and the cells are the legacy head's,
    `PriceCells(cell_type, lo, hi, step)`.

  `action_descriptors.range_price_cells(cell_type, lo, hi, step, grid)` (`RangePriceCells`) gives
  either partition addressed in prices (`cell_of(price)`, `member(cell, u)` -> price,
  `counts`, `nonempty`); Rust: `rl::generator::price_cells` / `price_at` / `price_step`.
  `model_generic.legal_actions(game)` reads the ranges this way for both titles
  (`LegalActions.price_grid`, `inputs["price_grid"]`, `price_cells_for(..., grid=)`); its 1830
  values are unchanged. Code that builds cells or samples prices itself (the generic policy player,
  policy gradient, evals) must pass `grid=` the position's `price_grid` before it plays 1867.
- **No price kind** means the price is not a decision: par cost, a depot or pool train at face, a
  token's price, a trade-in's fixed discount. (1830's enumeration carries a degenerate range
  `lo == hi` for depot trains; that is not a price parameter.)
- The frozen 1830 layout keeps its legacy price slots (14 per cross-corporation train, min/max
  per company) as indices, but only the first of each is live; the rest are `dead`. The legacy
  60-slot price head (`price_head_slot_for_index`, `mcts.rs::price_head_entity_key`) stays for
  legacy checkpoints; a generic head keys prices by descriptor instead.

## 7. Layouts: blocks, versions, fingerprints

A layout is an ordered list of **blocks**: one family over a tuple of **axes** whose entries come
from the title data (`layout::Axis`: entity lists, hexes, (hex, city) slots, tiles, rotations,
numbers, choices, legacy price slots). `index = block.offset + mixed_radix(positions)`, the first
axis slowest; an axis with one entry pins a field for the whole block (the company of a company
block). Axes come from title data through `Axis::companies(title, auctionable_only)`,
`Axis::corporations(title, kind)`, `Axis::trains`, `Axis::map_hexes`, `Axis::tiles`,
`Axis::par_prices(title, kind)` (the market's cells of the title's par types for a corporation
kind); `RlLayout::build` refuses an axis entry the title vocabulary lacks, an empty block or a
repeated block name.

A layout is identified by `(title, version)` and **frozen once a checkpoint trained on it
exists**. Its **fingerprint** (FNV-1a 64 over the version, the field list, the title
vocabularies, every block with its axes, and every descriptor row) is pinned per title by
`rl::tests::layout_fingerprints_are_pinned_per_title`; every title with a layout must be pinned
there. Any change of what an index means changes the fingerprint and needs a version bump (and a
re-converted dataset).

| Title | Version | Total | Fingerprint |
|---|---|---|---|
| 1830 | `1830-v1` | 26,537 | `e721aa6520116e2a` |
| 1867 | `1867-v1` | 38,875 | `4987cd19b7e912e2` |
| TEST-5SHARE (test builds only) | `TEST-5SHARE-v1` | 26,513 | `7196801649141472` |

### The frozen 1830 layout (`1830-v1`)

`factored_layout.rs` walks the `SlotLayout` that `build_layout` builds (unchanged, still pinned by
`layout_matches_frozen_1830_constants`) and refuses to build unless every block lands at the
`SlotLayout`'s own offset. Nothing computes an index from the registry: the index function stays
`action_index::legal_action_to_index_in`.

| Block | Family | Offset | Size | Axes |
|---|---|---|---|---|
| Pass | Pass | 0 | 1 | (also every company's pass) |
| Bid | Bid.company | 1 | 6 | company; price `bid` |
| Par | Par | 7 | 48 | corporation x par $ (67, 71, 76, 82, 90, 100) |
| BuyShares | BuyShares | 55 | 16 | corporation x source (ipo, market) |
| SellShares | SellShares | 71 | 40 | corporation x percent (10..50) |
| PlaceToken | PlaceToken | 111 | 28 | (hex, city) of every city on the initial map |
| LayTile | LayTile | 139 | 25,668 | hex (frozen order) x tile (frozen order) x rot |
| BuyTrain | BuyTrain.depot | 25,807 | 1 | `train_from_state`: the depot's next train |
| BuyTrain.pool | BuyTrain.pool | 25,808 | 6 | train type |
| BuyTrain.corp | BuyTrain.corp | 25,814 | 672 | corporation x train type x 14 legacy price slots (first live); price `train_transfer` |
| DiscardTrain | DiscardTrain | 26,486 | 6 | train type |
| Dividend | Dividend | 26,492 | 2 | payout, withhold |
| BuyCompany | BuyCompany | 26,494 | 12 | company x (min live, max dead); price `company_purchase` |
| Bankrupt | Bankrupt | 26,506 | 1 | |
| RunRoutes | RunRoutes | 26,507 | 1 | |
| CompanyBuyShares | ExchangeShares | 26,508 | 2 | via MH, NYC x source |
| CompanyLayTile:DH | LayTile | 26,510 | 6 | via DH, hex F16, tile 57 x rot |
| CompanyLayTile:CS | LayTile | 26,516 | 18 | via CS, hex B20, tiles 3/4/58 x rot |
| CompanyPlaceToken:DH | PlaceToken | 26,534 | 1 | via DH, F16 city 0 |
| BuyTrainDFull | BuyTrain.depot | 26,535 | 1 | train D |
| BuyTrainDTradeIn | BuyTrain.exchange | 26,536 | 1 | new D, traded train `train_from_state` |

Legacy collapses the descriptors cannot undo (they are in the index function, which is frozen):

- `BuyTrain.pool` slots also take a trade-in of a train whose name is in the pool (Python's
  check order); the descriptor says "pool train of type T".
- A teleport company's lay with a tile outside its block falls through to the regular `LayTile`
  slots (Python parity), so a slot without `via` may carry a company lay.
- `Pass` is shared by every passing entity (the state says who).

Verified (all green): every index against `ActionMapper.actions` (the static table the Python
decode reads, `tests/test_rl_descriptors.py`); every legal index of random-walk states against
the Python `ActionMapper` decode (same file) and against the factored enumeration plus the native
Rust decode (`rl::tests::descriptors_agree_with_the_index_function_and_decode_on_1830_walks`:
9,394 states at 2-6 players, every family offered).

### Readiness: the "not bridge-ready" seam

`GameTitle::rl_layout_kind()` (`None` / `Factored`) splits the old `alphazero_bridge_ready`
(which stays the per-title override and default of both): `rl_layout_ready()` = a layout exists,
`rl_encoder_ready()` = the legacy flat encoder can be derived. Every 1830-path index computation
now takes its layout from the game's own title (`BaseGame::factored_layout()`):
`factored_legal_indices`, `decode_index` / `apply_action_index`, `price_range_for_index`,
`price_head_slot_for_index`, the Rust MCTS tree (its layout is fixed from the root game), and the
Gumbel / dense policy vectors. For a title whose kind is `None` each of these raises
`RuntimeError: title 1867 is not bridge-ready: ...` instead of silently indexing the game with
1830's layout (which `factored_legal_indices` used to do). 1830 resolves to the very same
`SlotLayout` as before (pointer-equal), so its behaviour is unchanged.

A title whose moves come from its own generator has the third kind, **`Generator`** (1867 today;
the 1822 family next): a registry layout built from title data plus an index function from the
generator's moves to indices (§8). It never goes through `build_layout`. The 1830 index path
refuses it with its own message ("indexed by its generator layout ... not by the 1830 factored
index path"), and the generator API (`rl_legal_indices`, ...) refuses 1830 ("index it through
factored_legal_indices"). `rl_title_status` / `rl_layout_info` report the kind (`"none"`,
`"factored"`, `"generator"`); `rl_encoder_ready` stays false for 1867 (the generic encoder is
package B's).

## 8. The 1867 layout (`1867-v1`)

A registry layout built from title data (`rl/g1867.rs::block_specs`, kind `Generator`): 38,875
indices, fingerprint `4987cd19b7e912e2`. Every move of the legal-move generator
(`g1867/legal.rs`) maps to exactly one index (`rl/g1867.rs::move_key` -> `RlLayout::index_of`); a
state's moves map to distinct indices; a price-bearing move is one index whose range comes from the
move. 98.7% of the indices are tile lays.

| Block | Family | Offset | Size | Axes (title data) | Generator moves (step: action) |
|---|---|---|---|---|---|
| Pass | Pass | 0 | 1 | - | every step's pass (whose pass, and what it means, is state) |
| Bid.company | Bid.company | 1 | 5 | auctionable companies C&SL NFB MB QB SCT (the hidden "3" is not) | SingleItemAuction: bid on the lot up for auction; price `bid` ($5 grid; dutch mode: the asking price only, `lo == hi`) |
| Bid.corporation | Bid.corporation | 6 | 16 | minors | BuySellParShares: a minor's opening bid ($100+) or a raise in its live auction (never both in one state); price `bid_range` |
| Par | Par | 22 | 88 | majors x the market's major par cells (par_2 + par: $70 80 90 100 110 120 135 150 165 180 200) | BuySellParShares: par |
| BuyShares | BuyShares | 110 | 16 | majors x {`source:ipo` (treasury), `source:market`} | BuySellParShares: buy one certificate; PostMergerShares: buy a treasury share of the converted major (`source:ipo`) |
| SellShares | SellShares | 126 | 40 | majors x percent 10..50 (one unit up to the pool limit) | BuySellParShares: sell a bundle (one bundle per percent) |
| RedeemShares | RedeemShares | 166 | 5 | percent 10..50 | RedeemShares: the acting major buys back its own pool shares (cumulative bundles) |
| PlaceToken | PlaceToken | 171 | 19 | city slots `(hex, city)` | HomeToken (SR, and an OR re-placement): a home city; Token: a connected city (the distance price is state) |
| RemoveToken | RemoveToken | 190 | 19 | city slots | ReduceTokens: one of the merged major's tokens (duplicates in a city are removed by the merge, so `(hex, city)` names the token) |
| BuyTrain.depot | BuyTrain.depot | 209 | 9 | trains 2 3 4 5 6 7 8 2+2 5+5E | BuyTrain: a depot train at face (the head, or a phase-available later type) |
| BuyTrain.pool | BuyTrain.pool | 218 | 9 | trains | BuyTrain: a discarded train at face |
| BuyTrain.exchange | BuyTrain.exchange | 227 | 36 | new (8, 2+2, 5+5E: the trains with a discount table) x traded (5 6 7 8 2+2 5+5E: the types those tables discount) x {`train_source:depot`, `train_source:pool`} | BuyTrain: a phase-8 trade-in; the source axis tells a depot train from a pool train of the same type (the generator offers both; 186 such states in the fuzz below) |
| BuyTrain.corp | BuyTrain.corp | 263 | 216 | majors + minors (the CN only holds tokens) x trains | BuyTrain: another corporation's train; price `train_transfer` |
| DiscardTrain | DiscardTrain | 479 | 9 | trains | DiscardTrain |
| Dividend | Dividend | 488 | 3 | `dividend:payout`, `dividend:half`, `dividend:withhold` | Dividend |
| BuyCompany | BuyCompany | 491 | 5 | auctionable companies | BuyCompany / BuyCompanyPreloan: a player's private for $1..face; price `company_purchase` |
| RunRoutes | RunRoutes | 496 | 1 | - | Route: the auto-router's optimum (`g1867_auto_routes`), one move |
| Merge | Merge | 497 | 24 | majors + minors | Merge: add a minor to the merge set, or name the major (unfloated) that takes the set or a conversion; the two never share a legal set |
| Convert | Convert | 521 | 1 | - | Merge: convert |
| Choose | Choose | 522 | 1 | `choose:nationalize` | MajorTrainless: nationalize (the decline is the pass) |
| LayTile | LayTile | 523 | 38,352 | 94 hexes x 68 tiles x 6 rotations | Track: either of the turn's two lays (the slot rules, the $20 second lay, terrain and border costs are state) |

Choices of the layout:

- **Not in the layout**: `bankrupt` (replay-only: Ruby 1867 has no Bankrupt step), a BuyTrain
  step's `sell_shares` (1867's EMR never sells), and the replay-only out-of-order decisions
  (`g1867_remaining_work.md` "Replay-only"). A future rules change that generates one of them adds
  a block and bumps the version.
- **City slots** = for each hex the most cities over its upgrade closure (its initial tile and
  every manifest tile reachable by `g1867_upgrades_to`, phases and supply ignored): 19 slots on 16
  hexes. The closure adds none today: L12 (Montreal) already starts as the triple city, and 1867's
  upgrades never add a city to a hex. A `city` is the city's position on the hex's current tile.
- **Pass is one index** for every passing entity and step; the fuzz and both corpora never show
  two passes in one legal set (the API would raise on it).
- **Indices mean the same move under either cross-player policy**: the title's policy only removes
  moves (`policy=None` vs `"rules"`).

### The index API (`rl/generator.rs`)

| Rust (`BaseGame`) | Python (`engine_rs.BaseGame`) | |
|---|---|---|
| `rl_indexed_moves(policy)` | `rl_legal_indices(policy=None)`, `rl_legal_moves_json(policy=None)` | the state's moves with their indices, ascending: `[{"index", "block", "wire", "range": [lo, hi, step] or null, "step"}]`; raises if a move has no index or two share one |
| `rl_move_for_index(idx, price, policy)` | `rl_decode_index(idx, price=None, policy=None)` | index -> the move (18xx.games action dict in Python) with `price` checked against its range and step (None: `lo`) |
| `rl_apply_index(idx, price, policy)` | `rl_apply_index(idx, price=None, policy=None)` | apply it and log its action dict (so `raw_actions` replays the game) |
| - | `rl_price_range(idx, policy=None)` | `(lo, hi, step)` or None |
| `rl::generator::price_cells(kind, range)`, `price_at`, `price_step` | `price_pmf.price_cells(cell_type, 0, K, 1)` | the price head's cells in step units (§6) |

`policy`: None = the title's cross-player policy (`RESTRICTED`), `"rules"` = everything Ruby
accepts. `rl_layout_info("1867")` / `action_descriptors("1867")` give the descriptors (kind
`"generator"`). Game records of the 1867 path carry `rl18xx.layout` = `{title, layout_kind,
layout_version, layout_fingerprint}` (`titles.layout_metadata`, `game_records.make_game_record(...,
title="1867")`, `scripts/g1867_fuzz.py`; written, not enforced). 1830 records are unchanged.

### Verification (2026-11-13)

- **Layout**: blocks contiguous, every descriptor row distinct, no `dead` index, price kinds on
  exactly the four priced blocks, fingerprint pinned (`rl::g1867::tests::layout_blocks`,
  `rl::tests::layout_fingerprints_are_pinned_per_title`, `tests/test_rl_g1867_index.py`).
- **Fuzz through the index API** (`rl::g1867::tests::index_fuzz_many`, `G1867_INDEX_SEEDS=5`):
  40 walks, 3-6 players x 5 seeds x {title's policy, rules' policy}, every game to its end: 48,535
  states, 339,600 legal moves. At every state: every move indexed, indices distinct, each
  index's descriptor agrees with its move (entity, hex/city, tile/rotation, percent, par $,
  choice, train type and source), decode(index) through the API gives back the chosen move at its
  chosen price, both ends of every price range apply (96,486 price checks), and apply-by-index
  leaves the same state (serialized) and logs the same action dict as apply-by-move; the walk
  advances by index. Every block was offered and played; 186 states offered one trade-in pair
  from both the depot and the pool. The default test (`index_fuzz_3_to_6_players`, 6 walks) runs
  in ~75 s.
- **Index-level corpus completeness** (`scripts/g1867_completeness.py --index`, rules' moves,
  every human decision including `run_routes`, which matches the RunRoutes index by type): for an
  exact match also `rl_decode_index` == the human action (key and price) and `rl_apply_index`
  leaves the state the human action leaves (players' cash, corporations' cash / price / trains /
  certificates, companies, tiles and tokens, depot, bank, round and step).

  | Corpus | Games in scope (3/4/5/6 p) | Decisions | Exactly one index | Ambiguous | Decode / apply mismatch | Policy-excluded | Replay-only classes | Missing |
  |---|---|---|---|---|---|---|---|---|
  | 2026-10 scrape (`human_games/scrape_2026_10/1867`) | 316 (46/219/43/8) | 181,947 | 181,944 (99.998%) | 0 | 0 / 0 | 7 (2 companies, 5 merger adds) | 3 (out-of-turn post-merger passes) | 0 |
  | 2020-22 corpus (`human_games/1867`, all 1,027 files) | 727 (222/397/96/12) | 390,709 | 389,807 (99.769%) | 0 | 0 / 0 | 31 (1 company, 5 trains, 25 merger adds) | 862 (429 pre-#9655 merger-order buys, 370 passes and 63 choices out of order) | 40 |

  "Exactly one index" counts the policy-excluded decisions (legal under the rules, one index).
  The 40 misses of the old corpus (place_token 22, buy_train 6, buy_shares 3, lay_tile 3,
  discard_train 2, pass 2, merge 1, run_routes 1) are the generator's own misses: the index layer
  adds none (every generated move has exactly one index). 23 are the action at which the replay
  itself then stops (24 old games end in a replay rejection, recounted in A3: e.g. 23812@607, 32537@630,
  24437@929, 87092@379). The other 17 are old records the generator does not offer: 11
  place_token (21494@776 is the documented home token Ruby HEAD also refuses; 28418@641,
  29292@510, 32894@633, 39631@631, 42154@657, 42964@581, 48185@477, 67332@710, 81411@536 and
  31586@669 lie beyond the 300-file sample `g1867_remaining_work.md` probed), 3 buy_shares
  (39631@693/@702, 41062@688) and 3 lay_tile in games the replay rejects a few actions later
  (24437@899, 45921@503, 72923@336). Probed against Ruby HEAD 849c522 in A3 (`docs/rl_model.md` "Ruby HEAD
  check of A2's 17 unexplained misses"): none is a generator bug -- 11 are refused by a live player at Ruby HEAD
  in the same state (9 home tokens in Toronto / Montreal / N12, 2 tile rotations on M11), 6 lie past Ruby HEAD's
  replay horizon (pre-#9655 merger order, or after a home token Ruby refuses). The 2026-10 scrape, which Ruby HEAD
  replays, has none.

### Price census and price specs

Human prices of the price-bearing indices, both corpora (exact matches only;
`--census-rows`), against `price_pmf` cell layouts taken in step units (§6). "Bits" = E[log2
|cell of the human price|], the resolution a cell layout leaves (0 = every human price is a cell
of its own):

| Block (n) | Where humans price | "Bid" cells | "BuyTrain" cells | "BuyCompany" cells | Spec |
|---|---|---|---|---|---|
| Bid.company (16,871) | 72-83% at the minimum bid, 99% within 5 steps; never beyond 19 steps | 100% on atoms, 0.00 bits | 0.23 | 0.51 | `bid` ("Bid" ladder, step $5 from the move) |
| Bid.corporation (18,148) | openings (11,481): 29-33% at $100, 27-29% all-in (hi = the bidder's limit), 18-20% beyond 19 steps; raises (6,667): 51-59% at the minimum raise, 13-18% all-in | 0.45 (openings 0.69: the all-in bids and everything past the ladder fall in one bin of ~150 prices) | **0.20** (openings 0.26, raises 0.09) | 0.60 | **`bid_range`** ("BuyTrain" cells: atoms lo, hi, hi-1 + 20 bins) |
| BuyTrain.corp (6,797) | $1: 18-20%; all the buyer's cash (hi): 43-55%; hi-1: 3-8%; the train's face value 4-5% (2-trains are 66% of these purchases) | 4.54 | **0.78** (69-77% on atoms) | 1.37 | `train_transfer` ("BuyTrain") |
| BuyCompany (4,077) | at hi 96-97% (face, when the buyer can pay it: 87-92% at face), $1: 0.5% | 4.85 | 0.05 | **0.09** (97% on atoms) | `company_purchase` ("BuyCompany") |

The only spec that differs from 1830's is the minor bid: price kind `bid_range` (appended to the
price-kind vocabulary, id 4) with "BuyTrain" cells. A face-value atom for train transfers would
catch 4-5% more; not worth a new cell layout. Pars are an index axis, not a price: 1,055 human pars
used all 11 cells ($70: 49, $80: 41, $90: 40, $100: 67, $110: 59, $120: 65, $135: 55, $150: 84,
$165: 67, $180: 108, $200: 420).

**`amount` stays without cells**: 1867 has no price-bearing decision of that kind. Its loans are
automatic (Ruby `AutomaticLoan` in the Track and BuyTrain steps takes them while a cost exceeds
cash; `LoanOperations` pays the interest and repays them; there is no `take_loan` action), and
redemptions are indices (`RedeemShares[percent]`), not prices. The kind's first user is 1822's partial player-debt payoff, whose census decides its
cells (the generic model's price head rightly raises on it until then).

## 9. What the schema cannot express yet

1867: nothing (§8). The 1822 family's generator moves (`g1822_legal_moves_json`) need the
generator index function (`rl/generator.rs`, today typed on 1867's `Move`) generalized to their
move type.

The 1822 family (to settle with its layout):

1. **Multi-company tile lays** (`lay_tile` `combo_entities`, Ruby SpecialTrack combos): `via`
   holds one company. Needs a second `via` column or a per-title combo vocabulary in `choice`.
2. **Choices that index a state list** (the Pullman choice, 1822CA's city choice, the minor
   acquisition's pay / token choices, 1822PNW's conversion share counts): `Choose` with
   `param` = the option's position is expressible, but the descriptor then says nothing about the
   option. Where an option is an entity, hex or token, the layout should give it those fields
   instead (a token choice as hex/city).
3. **Bids on something the state names**: 1822MX's NdeM token auction bids on "the token up for
   auction" (`Bid` with no company); it needs a target-less bid family or hex/city fields on a
   bid. Its bids also move in $10 steps while MX company bids move in $5: the price ladder step
   must come from the move's `PriceRange.step` (§6), not from the title's `bid_price_step` -- done
   for generator layouts since `1867-v1` (`rl_price_range` returns the move's step, cells in step
   units).
4. **Pseudo-tiles**: 1822MX's builder cubes ("BC") and port tiles are laid as tiles; the tile
   vocabulary (the title's manifest) must include them, with tile features.
5. **Many named choices**: P16 tax haven, P17 / P18 / P5 ability choices, choices-round "double",
   dividend-step choose: expressible as appended global choices plus `via`, but the vocabulary
   grows by title; the strings need to be enumerated from the generator first.

Sizes from title data (lay slots alone): 1822 78,492 (211 hexes x 62 tiles x 6), 1822PNW 55,746,
1822MX 41,340, 1822CA 162,378 (279 x 97 x 6) -- larger than design §2.6's "about 100k", which the
legal-only head does not mind.
