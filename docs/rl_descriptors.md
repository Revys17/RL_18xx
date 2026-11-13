# RL action descriptors: the schema

*Milestone A1 of the title-generic RL bridge (2026-11-13; design `docs/rl_bridge_design.md` §2,
work packages `docs/rl_bridge_work_packages.md` package A). Code: `engine-rs/src/rl/`,
`rl18xx/agent/alphazero/action_descriptors.py`, `rl18xx/agent/alphazero/titles.py`.*

Every title gets **one index space** (its *layout*): a policy index per distinct legal move, the
price of a price-bearing move excepted (prices go to price heads). Every index of every layout also
gets a **descriptor**: one row of small integers that says what the index means in
title-independent terms. A generic policy head never keeps one output per index; it scores the
legal indices of a state from their descriptors, so the same weights score "buy a share of X" or
"lay tile T on hex H" in any title, and a layout can be any size (1822CA's tile lays alone are
162k indices).

Today exactly one real layout exists: **1830 v1**, the frozen flat layout of
`action_index.rs::build_layout` (26,537 indices), now with a descriptor for every index. 1867 and
the 1822 family have no layout yet ("not bridge-ready"); §7 sketches how 1867 maps onto the schema
and §8 lists what the schema cannot express.

## 1. Where it lives, how to read it

| What | Where |
|---|---|
| Schema, global vocabularies | `engine-rs/src/rl/descriptor.rs` |
| Block registry, descriptors per index, fingerprint | `engine-rs/src/rl/layout.rs` |
| 1830's frozen layout as a registry | `engine-rs/src/rl/factored_layout.rs` |
| Python surface | `engine_rs.rl_layout_info(title)`, `engine_rs.rl_descriptor_vocab()`, `engine_rs.rl_title_status(title)` (`engine-rs/src/rl/py.rs`) |
| Python wrapper (cached, numpy) | `rl18xx.agent.alphazero.action_descriptors.action_descriptors(title)` |
| Readiness per title | `GameTitle::rl_layout_kind` / `rl_layout_ready` / `rl_encoder_ready` (`engine-rs/src/title/mod.rs`) |

```python
from rl18xx.agent.alphazero.action_descriptors import action_descriptors

d = action_descriptors("1830")            # RuntimeError "not bridge-ready" for 1867 / 1822*
legal = game.factored_legal_indices()     # the 1830 path's legal indices (sorted)
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
| 10 | `param` | The family's number | Par: par price in $; BuyShares / SellShares / RedeemShares / IssueShares: percent moved; Choose: the option's position when options are a state list (§8) |
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
| 23 | `Choose` | choice (or param, §8) | 1867 nationalize, 1822 family |
| 24 | `AcquireCompany` | ent0 company | 1822 |
| 25 | `ChooseAbility` | via company, choice | 1822 |
| 26 | `HexToken` (map) | hex, choice token type | 1822 destination tokens |
| 27 | `Assign` | via company, hex or ent0 corporation | 1822PNW, 1822CA |
| 28 | `PayoffPlayerDebt` | - | 1822 |
| 29 | `PayoffPlayerDebtPartial` | price `amount` | 1822 |

**Entity kinds** (`ent0_kind`, `ent1_kind`): 0 `player`, 1 `corporation`, 2 `company`, 3 `train`.

**Price kinds** (`price_kind`, §6): 0 `bid` (cells "Bid"), 1 `train_transfer` ("BuyTrain"),
2 `company_purchase` ("BuyCompany"), 3 `amount` (cells not fixed yet).

**Choices** (`choice`): 0 `source:ipo`, 1 `source:market`, 2 `dividend:payout`,
3 `dividend:withhold`, 4 `dividend:half`, 5 `choose:nationalize`. A title's named choices are
appended when its layout lands (1822: `choose:double`, ...).

**Flags** (`flags`, bits):

- `dead` (1): the index is never legal. Only in frozen layouts: slots the index function never
  produces (1830: the 13 extra price slots of every cross-corporation train, the "max" slot of
  every BuyCompany; 630 indices). A head can mask them once.
- `train_from_state` (2): the train is decided by the state, not by the index (1830: the single
  depot slot = the depot's next train; the D trade-in = which owned train is traded).

## 4. Title vocabularies

Entity, hex and tile fields index lists that come from the title data, in its order
(`layout::Vocab`; `rl_layout_info(title)["vocab"]`):

| Vocabulary | Order | 1830 |
|---|---|---|
| `corporation` | `GameTitle::corporations()` (every kind: majors, minors, nationals) | PRR NYC CPR B&O C&O ERIE NYNH B&M |
| `company` | `GameTitle::companies()` (auctionable or not; the 1822 family's minors M1.. and concessions C1.. are companies too) | SV CS DH MH CA BO |
| `train` | `GameTitle::trains()` names, variants included (1822's "2" is the L card's variant) | 2 3 4 5 6 D |
| `hex` | `GameTitle::hex_definitions()` = `BaseGame.hexes` order | 93 hexes, map order |
| `tile` | `GameTitle::tile_counts()` (the title's tile manifest) | 46 tiles |
| players | the game's seating order (seat index; no 1830 / 1867 family refers to a player) | - |

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
    family is first used.
  `P(price) = softmax(nonempty cell logits)[cell] / |cell|` (`price_pmf.rs`), exactly as 1830's
  head; MCTS progressive widening over prices is unchanged.
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

A title whose moves come from its own generator (1867, the 1822 family) will get a third kind
(a registry layout built from title data, mapping generator moves to indices); it must never go
through `build_layout`, which cannot express it.

## 8. Sketch: 1867 on the schema (not implemented)

Built from title data (`rl::tests::print_title_axis_sizes`): 94 hexes, 68 tiles, 6 companies (5
auctionable; the hidden "3" is not), 8 majors + 16 minors + the CN, 11 major par cells (par_2 +
par), trains 2 3 4 5 6 7 8 2+2 5+5E, 19 city slots on the initial map. Every generator move of
`g1867/legal.rs` maps to one block:

| Generator (`g1867_gen`) | Block | Size | Notes |
|---|---|---|---|
| any step: pass | `Pass` | 1 | |
| SingleItemAuction bid (range $5 grid, dutch: the asking price only) | `Bid.company[auctionable companies]` | 5 | price `bid`, step from the move |
| BuySellParShares: minor opening bid ($100+) and raise in a live minor auction | `Bid.corporation[minors]` | 16 | price `bid`; opening vs raise is the step state |
| BuySellParShares: par | `Par[majors x par $]` | 88 | |
| BuySellParShares: buy one certificate | `BuyShares[majors x {ipo, market}]` | 16 | minors are one 100% certificate, never bought as shares |
| BuySellParShares: sell a bundle | `SellShares[majors x percent 10..50]` | 40 | |
| PostMergerShares: buy a treasury share of the converted major | `BuyShares[majors x ipo]` (same block) | - | |
| HomeToken (SR), Token (OR): place_token | `PlaceToken[(hex, city)]` | ~19-21 | city slots = max cities over each hex's upgrade closure (L12's triple city); the distance price is state |
| ReduceTokens: remove_token | `RemoveToken[(hex, city)]` | ~19-21 | the corporation has one token per city, so (hex, city) names it |
| Track: lay_tile (two lays per turn) | `LayTile[hex x tile x rot]` | 38,352 | the slot rules (second lay $20, no upgrade after an upgrade) are state |
| Route: run_routes | `RunRoutes` | 1 | |
| Dividend: payout / half / withhold | `Dividend[choice]` | 3 | |
| BuyTrain: a depot train at face | `BuyTrain.depot[train]` | 9 | |
| BuyTrain: a discarded train at face | `BuyTrain.pool[train]` | 9 | |
| BuyTrain: trade-in (phase 8) | `BuyTrain.exchange[new x traded]` | ~18 | the (new, traded) pairs of the trains' discount tables |
| BuyTrain: another corporation's train ($1..cash) | `BuyTrain.corp[corps x train]` | ~216 | price `train_transfer` |
| DiscardTrain | `DiscardTrain[train]` | 9 | |
| BuyCompany / BuyCompanyPreloan ($1..face) | `BuyCompany[companies]` | 5 | price `company_purchase` |
| RedeemShares: buy back own pool shares (cumulative bundles) | `RedeemShares[percent 10..50]` | 5 | |
| Merge: merge (add a minor, or name the major) | `Merge[corporations]` | 24 | the two meanings never share a legal set (the step state says which) |
| Merge: convert | `Convert` | 1 | |
| MajorTrainless: choose nationalize | `Choose[choose:nationalize]` | 1 | |

About 38,900 indices, 99% of them tile lays (design §2.3's estimate holds). Open points for the
1867 layout (A2): a trade-in whose new train exists both in the depot and in the pool (the
generator offers both, with different train ids) needs a source axis or a rule; the city-slot
axis needs the upgrade closure (the 1830 rule "cities of the initial tile" misses L12's third
city); the price census for 1867 bids and company purchases decides the cells.

## 9. What the schema cannot express yet

1867: nothing beyond the open points above.

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
   must come from the move's `PriceRange.step` (§6), not from the title's `bid_price_step`.
4. **Pseudo-tiles**: 1822MX's builder cubes ("BC") and port tiles are laid as tiles; the tile
   vocabulary (the title's manifest) must include them, with tile features.
5. **Many named choices**: P16 tax haven, P17 / P18 / P5 ability choices, choices-round "double",
   dividend-step choose: expressible as appended global choices plus `via`, but the vocabulary
   grows by title; the strings need to be enumerated from the generator first.

Sizes from title data (lay slots alone): 1822 78,492 (211 hexes x 62 tiles x 6), 1822PNW 55,746,
1822MX 41,340, 1822CA 162,378 (279 x 97 x 6) -- larger than design §2.6's "about 100k", which the
legal-only head does not mind.
