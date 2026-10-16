# Title-generic RL bridge: design and gap analysis (1867 first)

*Phase 1 deliverable, 2026-10-16. A design for review: nothing here is implemented
yet. Branch base: `master` @ 05184fa. Line numbers below are for that commit.*

The rules engine can play 1867 (it replays the 18xx.games corpus as Ruby HEAD does,
`docs/g1867_remaining_work.md`), but everything between the engine and the
network (action indices, encoder, model, MCTS, data pipeline, evals) assumes
1830. This document says what has to change so that 1867, and then about fifteen
more titles, can be trained, and in what order.

## 0. Summary

**Recommended design**

- **Action space: one index space per title and layout version, frozen once a
  checkpoint exists.** 1830 keeps its current layout byte for byte
  (`action_index.rs::build_layout`, pinned by `layout_matches_frozen_1830_constants`).
  New titles get their layout from a generic *block registry*: a list of action
  families such as `Bid.company[company]`, `LayTile[hex x tile x rot]` and
  `BuyTrain.exchange[new x traded]`, whose axes come from the title data.
  Every index in every layout, the frozen 1830 one included, also gets a
  **descriptor**: its family, the entities, hex and tile it refers to, and a
  parameter. A layout fingerprint test pins each title's layout.
- **Legal actions: a step-driven enumerator**, keyed on the engine's `StepKind`
  (the step machinery already says which action *types* each step offers),
  added beside `factored.rs`. `factored.rs` stays the 1830 path, because its
  parity with Python is load-bearing. It also becomes an oracle: on 1830 the new
  enumerator must reproduce `factored_legal_indices` exactly, over random walks
  and the corpus, before it is trusted on 1867.
- **Decoding stays "enumerate, then find the index"**, as `decode.rs` already does.
  It is title-generic once every index lookup goes through `layout_for(title)`.
- **Verification per title**, in place of the Python oracle 1830 had:
  - enumerate-and-apply fuzzing: every legal action applies, indices are
    injective, decoding round-trips, nothing panics;
  - corpus completeness: every human action matches exactly one enumerated action;
  - **fuzz games replayed in Ruby HEAD** with the existing docker rig, so that any
    rejection is an enumeration bug.
- **Encoder: typed tokens instead of a flat vector.**
  - Hex tokens carry title-generic tile, track, stop and border features. A token
    occupying a city slot is a *pointer* to a corporation token, not a
    per-corporation one-hot.
  - Entity tokens cover players, corporations of every kind (major, minor,
    national), companies, train types, phases and one global token. Each token
    type has a fixed, title-independent feature schema.
  - Entity-to-entity relations (holdings, presidency, ownership, trains held)
    become an attention bias.
  - Variable counts are handled by padding and masks.
  - What stays title-specific is vocabularies and normalizers, declared in a
    per-title `RlSpec`.
- **Model: one architecture for every title.** Whether titles also share
  *weights* is a separate question.
  - Hex and entity transformers, cross-modal fusion, and a FiLM trunk
    conditioned on title + round kind + step kind.
  - A **descriptor-conditioned policy head that scores only the legal actions**.
    It keeps the P(hex)·P(tile|hex)·P(rot|hex,tile) factorization for tile lays.
  - A pointer-keyed price head.
  - The existing per-seat value head, which already shares one MLP across
    seats, widened to `MAX_SEATS = 8`.
- **Training: per-title models by default.** Section 4.2 assesses sharing
  candidly:
  - **Family pooling** (1822 + CA/MX/PNW, 1867 + 1861, 1817 + 18USA) is the
    first sharing step. It is the best-supported form and covers the small
    titles.
  - **One model across all titles is not recommended as a commitment.** It is
    tested cheaply, by jointly pretraining 1830 + 1867 against 1867 alone
    (experiments E0/E1, with a pre-set decision rule), and adopted only if that
    wins.
  - RL fine-tuning is per title in every case.
  - The 1830 production model keeps its legacy path until the generic model
    matches it on 1830.

**1867, the first user.** The largest gaps are rules gaps that only self-play
reaches, not RL plumbing:

- The native **auto-router is 1830-only.** It sees a train as `(distance, is_D)`
  (`game.rs:4203-4205`): no free towns, no 5+5E, no x2 trains, no bonuses. So
  self-play `RunRoutes` would pay 1830 revenue in 1867 games.
- **Impassable borders**: the walls are hardcoded to 1830's hex ids
  (`factored.rs:1138-1147`) and so are applied to 1867's C11, C13, D12, F8, C17,
  B16 and E7, while 1867's real walls are ignored.
- **The factored enumerator has no generator** for:
  - SR minor bids and the home-token choice;
  - merge, convert, choose and remove_token;
  - the three discounted trains with their trade-ins.

  The single-item opening auction reaches `factored_bid` only because it shares
  `AuctionState`, which applies the waterfall's `min_bid_for` / `may_purchase`
  rules instead of the ascending-plus-dutch ones.

**Estimates (agent-days, focused work, ranges are honest uncertainty):**

| Track | Effort |
|---|---|
| Generic layer (S1-S8) | 38-56 days |
| 1867 track (T1-T7d) | 30-46 days, plus 8-11 for the optional T8 items, plus 2-3 weeks of training wall-clock |

With two or three agents working in parallel (section 7):
- first supervised 1867 model about 6-8 weeks after start;
- first policy-gradient 1867 model about 10-12 weeks after start.

**Decisions for the user** (section 8):
1. Model sharing. Recommended: per-title models plus family pooling, with
   cross-family sharing only if E0/E1 pass.
2. `MAX_SEATS` of 8 or 12.
3. Whether 1830 eventually moves to the generic path.
4. 1867 scope for the first run (base rules, 3-6 players).
5. GPU time for the 1830 validation of the generic model.
6. Whether 1867 self-play includes the opening auction.
7. Whether to take 1861 as the cheap second title.

---

## 1. Inventory of 1830-specific assumptions in the RL layer

Each row gives where the assumption lives, what it assumes, and what it must
become. **F** means the code must stay frozen for 1830, so the change is
additive: a new path beside the frozen one. The list is exhaustive for the RL
layer. Rules-engine 1830-isms are listed only where self-play reaches them.

### 1a. Rust bridge (`engine-rs/src`)

| Where | Assumption | Becomes |
|---|---|---|
| `action_index.rs:25` | `POLICY_SIZE = 26537` | Stays as the 1830 constant (F). All other code reads `layout_for(title).total`. |
| `action_index.rs:103-105` `layout()` | Global 1830 layout | Callers that have a game use `layout_for(&game.title)`. `layout()` stays only for the 1830 Python shims. |
| `action_index.rs:107-313` `build_layout` | 1830-shaped blocks. 14 train-price slots per corp and train type, of which only the first is ever used (`first_price = 0`, `:548`). `share_location_offsets = ["ipo","market"]`. Dividend is only payout/withhold. | Frozen as the "1830 v1" builder (F). New titles use the block registry (§2). |
| `action_index.rs:167-170` | At most one discounted train (1867 has three: 8, 2+2, 5+5E) | `BuyTrain.exchange[new x traded]` block (§2.3) |
| `action_index.rs:564-585` | `action_offsets_py` and `policy_size_py` are 1830-only. `policy_size_for_py` panics for 1867. | `layout_info_py(title)`: offsets, descriptors, fingerprint |
| `factored.rs:166-229` | Only the 12 1830 action types are dispatched; `_ => {}` drops merge, convert, choose and remove_token | Step-driven enumerator for new titles (§2.4). `factored.rs` stays as the 1830 path (F). |
| `factored.rs:247-294` | `Bid` exists only in `Round::Auction`, with waterfall rules: 1867's single-item auction shares `AuctionState` (`single_item: Some`), but its ascending/dutch rules (`rounds/single_auction.rs`) are not what this computes; SR corporation bids are absent | Per-StepKind generators: SingleItemAuction, SR CorporationBid, later 1822 bid boxes |
| `factored.rs:296-364` (`:351`) | Par affordability is `2 * price` (20% president cert); no corp-type or phase gate | Uses the title's par types, startable phases, cert size and capitalization |
| `factored.rs:625-723` | Pending tokens exist only in the OR; no stock-round home token | HomeToken generator in the SR (1867 majors and minors choose a home) |
| `factored.rs:1138-1147` | **1830 walls hardcoded by hex id**, applied to every title | Read `HexDef.borders` (impassable) per title. Same rule in the graph/connectivity. |
| `factored.rs:1798-1811` `factored_legal_indices` | Uses the 1830 layout. **On a 1867 game it silently mis-indexes.** | `layout_for(&self.title)`, plus a debug assert that the title is bridge-ready |
| `decode.rs:105`, `:121`, `:854` | Bounds use `POLICY_SIZE`; offsets and price slots use `layout()` | Title layout; price slot taken from the descriptor (§2.7) |
| `encoder.rs:28` `MAX_PLAYERS = 6` | Value width, MCTS `VALUE_SIZE` | `MAX_SEATS = 8` engine-wide (§4.4). The legacy 1830 model keeps 6, padded at the boundary. |
| `encoder.rs:31-36` | Scales: share price 350, private revenue 30, hex revenue 80, lay cost 120, round type 2 | Per-title normalizers derived from title data in `RlSpec` |
| `encoder.rs:70-71`, `:114`, `:121` | Cert limits and cash only for 2..6 players, with fallbacks 16 and 600 | Seats 2..`MAX_SEATS`; the title supplies its tables |
| `encoder.rs:126`, `:551-560` | Node features have a per-corp one-hot for 2 cities (`ci >= 2 break`) | Pointer features per city slot (§3). 1867 L12 has three cities. |
| `encoder.rs:219-227` | Round type is 0/1/2, with **Merger mapped to 1.0 as a placeholder** | Round-kind and step-kind vocabularies (§3.3) |
| `encoder.rs:318`, `:388`, `:396` | Normalized as `1/train_counts`; 1867's unlimited trains have count `u32::MAX`, so the feature is about 0 | Count features with an "unlimited" flag (§3.2) |
| `encoder.rs:372-379` | Market zones are the 4 1830 zones | Flags over the full `MarketZone` set: par_1, par_2, convert_range, max_price, ... |
| `encoder.rs:463`, `:469` | Scaled by `OR count / 3` and `train_limit / 4` | Title-normalized |
| `encoder.rs` (whole) | No minors/majors/national distinction, no loans, interest, minor train limits or home-choice state | Generic token schema (§3) |
| `mcts.rs:31` `VALUE_SIZE` | 6 | `MAX_SEATS` (8) |
| `mcts.rs:121-190` `price_head_entity_key` | Re-derives the 1830 price slots from `layout()` offsets | Slot key read from the index descriptor |
| `mcts.rs:195-197` `bid_step()` | Takes the title from the global 1830 layout | `self.title`'s `bid_price_step` |
| `mcts.rs:465`, `:1001-1012`, `:1192` | Dense vectors of 1830 `layout().total`; priors must match that length | Per-tree layout total; add `incorporate_results_sparse` (legal priors only) |
| `mcts.rs:1490-1505` | c_puct keyed by Auction/Stock/Operating; Merger falls through to the default | Keyed by round kind from the vocabulary |
| `price_pmf.rs:18-25` | Cell layouts only for Bid, BuyTrain and BuyCompany | Cell spec per price-bearing family, with title parameters (§2.7) |
| `game.rs:2579-2585`, `:2863-2879` | Public `BaseGame(...)` builds "1830" | Pipeline always calls `new_titled` (exists at `:2884`) through one Python factory |
| `game.rs:4156-4220` `optimal_routes` | `trains: Vec<(u32, bool)>` = (distance, is D) (`:4203-4205`); `router.rs:751` counts every stop | Per-title route rules (§6, T2) |
| `title/mod.rs:335` `alphazero_bridge_ready` | One flag covering layout and encoder together | Split into `rl_layout_ready()` and `rl_encoder_ready()`, plus a registry of `RlSpec`s |
| `title/mod.rs:503-508` `parse_coord`; `model_transformer.py:39-52` | Single-letter column coordinates | Check each new title's coordinates (multi-letter maps exist in the 18xx catalog) |

### 1b. Python model, encoder and config (`rl18xx/agent/alphazero`)

| Where | Assumption | Becomes |
|---|---|---|
| `encoder.py:4`, `:17-56` | Imports the Python g1830 engine for every constant; `ROUND_TYPE_MAP` has 3 kinds; hex order taken from `Map_1830` | Frozen for the legacy model (F). The generic encoder is native Rust (§3). |
| `encoder.py:345-369` `canonicalize_perspective` | Rotation plan over 1830 sections | Generic: seat-indexed fields rotated in Rust inside `encode_tokens` |
| `mcts.py:70-114` `_rust_encode` | `edge_index` from the Python 1830 encoder, cached globally per process | Dropped: the transformer path never uses it |
| `model_transformer.py:71-189` | Gather indices from `Encoder_1830Graph.compute_section_layout`; `NUM_CORPORATIONS=8`, `NUM_PRIVATES=6`; `PLAYER/CORP/PRIVATE/GLOBAL_FEAT_SIZE` | `model_generic.py`: typed token inputs, no gather (§4) |
| `model_transformer.py:214-308` `_pad_state_to_max_players` | Pads 1830 sections | Seat padding done in the encoder, with masks |
| `model_transformer.py:1083-1135`, `:1596-1600` | Policy-head buffers built from the Python 1830 `ActionMapper` | Descriptor tables from `layout_info_py(title)` |
| `model_transformer.py:1155` `other_head` | `Linear(d_trunk, num_other)`: weights tied to absolute 1830 slots | Descriptor scorer (§2.6) |
| `model_transformer.py:1436-1529` | Axial coordinates assume pointy layout; structural matrices from a throwaway **4-player 1830** game (`:1528`) | Per-title geometry from Rust (`hex_adjacency`, layout-aware axial), cached per title |
| `model_transformer.py:1592` | `phase_embedding(num_round_types=3)` | FiLM input = embedding(title) + embedding(round kind) + embedding(step kind) |
| `model_transformer.py:1821`, `:1883` | `_infer_num_players_from_state_size` uses MAX_PLAYERS = 6 | Gone: `num_seats` is an explicit input |
| `config.py:22-28` | `policy_size` from `policy_size_py()` (1830) | Per title, from the layout |
| `config.py:138-153`, `:190`, `:224-230` | Sizes 408, 50, 93 hexes, 46 tiles, 3 round types; asserts `max_players <= 6` | `GenericModelConfig`: d-model sizes only, plus the set of titles the model covers |
| `price_pmf.py:31-53` | `SLOTS` = 6 + 48 + 6 1830 slots; `BID_STEP_1830` | Pointer-keyed slots (§2.7). The 1830 list stays for the legacy head (F). |
| `action_mapper.py` (whole) | 1830 tables (`:36-409`), price-head tuples (`:670-672`), MH/DH/CS/D special cases (`:834-1010`) | Frozen for 1830 (F). No Python mirror for new titles: index work is native Rust. |

### 1c. Pipeline (self-play, policy-gradient, data, evals)

| Where | Assumption | Becomes |
|---|---|---|
| `start_positions.py:65-71` `new_game`; `self_play.py:281-288`; `loop.py:496-513`; `arena.py:29-32`; `advisor/live_game.py:146-152`; `pretraining.py:1216`, `:1547`, `:1677`; `eval_head_to_head.py:208-210` | Engine built with no title; players always named "Player N" | One `titles.new_game(title, n, optional_rules, seed)` factory |
| `mcts.py:60-65` | `POLICY_SIZE` from the 1830 `ActionMapper`; `VALUE_SIZE = 6`; `BID_STEP_1830` | Per title / `MAX_SEATS` |
| `rust_mcts_player.py:27-33`, `:297-332`, `:502-514`, `:544-548` | Dense 1830 policy; value width 6 | Sparse legal priors; `MAX_SEATS` |
| `inference_server.py:126-144`, `:745-750`, `:809-811` | Sparse replies exist, but the client densifies them to the 1830 `POLICY_SIZE`; `encoder_type` is "Transformer" | Stay sparse end to end; each request carries its title |
| `dataset.py:105-122`, `:150`, `:192-218`, `:286-294` | `pi` is dense (26,537 floats per row); `ActionMapper` masks; self-play stores `encoded_state[:6]`, dropping rotation and seat count | Schema v3: sparse legal and pi, title and layout id, token bundle (§5.1) |
| `checkpointer.py:126-200`, `:214-224`, `:241-265` | **Hides layout mismatches**: forces `policy_size` to the current mapper, pads or drops head tensors, `strict=False` | Store title(s), layout fingerprints and encoder version; refuse to load on mismatch |
| `checkpointer.py:302-305`, `:407-476` | `current_best` is per architecture | Per (title or base, architecture) |
| `config.py:391-401`, `:584-588`; `self_play.py:290-310`; `rust_mcts_player.py:173-185`; `loop.py:516-530`, `:347`, `:797` | Player counts 2..6 with hidden 4-player defaults. **The loop runs 4-player only unless `loop_config.json` sets a distribution**, contrary to the comment at `loop.py:114-118`. | Distribution per title in `TitleSpec`, seats 2..`MAX_SEATS` |
| `loop.py:180`, `:533`, `:547-559`, `:1329` | 4-seat gate, 4-player win-rate dicts | Per N |
| `policy_selfplay.py:164-166` | `new_game(4)`, `sample_start_position(4, ...)` | Per title and N |
| `policy_gradient.py:96`, `:109`, `:223-224`, `:244-246`, `:277`, `:327-329`, `:385`, `:536-540`, `:674-711`, `:719-725`, `:913-941` | `NUM_PLAYERS = 4`; 1830 `CORPORATIONS` for the `president:` reward; bankrupt bonus "ends the game in 1830"; saved games labeled 1830; lay-tile range from `ActionMapper` | Per title and N (§5.4) |
| `config.py:356-362` | `c_puct_by_round` keyed by Auction/WaterfallAuction/Stock/Operating | Keyed by round-kind vocabulary |
| `self_play.py:1209`, `:1310-1341`; `loop.py:1112-1119`; `eval_head_to_head.py:243-257` | Phase metrics and stall detection keyed by "Auction" (the `auction_unlock` training wheels) | Per-title stall policy. 1867's dutch auction cannot lock. |
| `start_positions.py:53-54`, `:118-206`, `:269-418`; `build_start_positions.py`; `build_midgame_positions.py` | Human prefix is the 1830 waterfall (bid/pass/par only); random starts know SV/B&O; files named `*_1830_4p` with no title field | Generic human cut at "first decision of round R" plus a title predicate; random generator per title is optional (§5.6) |
| `game_records.py:59-69`, `:109-112` | Writes `"title": "1830"` and `optional_rules: []`; infers 4 players | Title from the engine |
| `pretraining.py:1172-1180` `ENGINE_OPTIONAL_RULES` | One global tuple (`optional_6_train`) | Per title, taken from `GameTitle::optional_rules()` |
| `pretraining.py:906-1141` (`_matching_choices`, `_factored_choice_matches`, `_factored_price_target`) | Matches only 1830 types; Bid only against companies; CompanyBuyShares only for MH; SellShares count equals certificate count; price slots from the 1830 tuples | Native Rust matcher: human wire dict to the set of indices (§5.2) |
| `pretraining.py:1286-1517` `replay_human_action` | Drop rules keyed on MH, CS, DH and the D train; Python step classes for isinstance checks | Per-title drop and leniency rules in Rust (most can disappear: new titles have no Python oracle to disagree with) |
| `pretraining.py:1835`, `:2896-2909` | Title filter only checks that a title key is present; fallback raw-JSON directories are 1830's | Filter on the title value |
| `rust_adapter.py:41-127`, `:546-614`, `:674-680`, `:1205-1307`, `:1745-1910`, `:2705-2713` | Python 1830 data, 10% shares, waterfall step proxies, no Merger round | Frozen for 1830 cleaning and Python MCTS (F). **New titles bypass the adapter**: the hot path needs only native calls (§5.3). |
| `eval_head_to_head.py:80-90`, `:225`, `:275`; `eval_policy_only.py:48-64`, `:98-155`, `:263`, `:350`; `arena.py:21-27`; `main.py:289`, `:558` | 4 seats, `combinations(range(4), 2)`, 1830 train limits and scripted B&O rules | Seat arrangements per N; scripted baselines per title, or none |
| `advisor/live_game.py:34-36`, `advisor/describe.py:19-61`, `advisor/advisor.py:57`, `:138`, `:282-285` | 1830 only, 4 players, 1830 step names | Later; out of scope for training |

### 1d. Already title-generic (keep)

- `pretraining.filter_actions` (Ruby `filtered_actions` semantics) and `in_validation_split`.
- `BaseGame.new_titled` and `supported_titles_py`.
- `encode_for_gnn` dispatches through `spec_for(&self.title)`.
- The decomposed policy loss reads its layout from the model's components (`train.py:235-329`).
- The price NLL takes its slot count from the logits.
- PFSP `League`, `composite_model`, and per-seat value padding (`train.py:517-521`).
- The inference server has no fixed buffers.
- `filter_actions` + `new_titled` is already how the 1867 harness drives the engine (`tests/title_replay_harness.py`).
- The step machinery (`steps.rs`): every title's legal action *types* come from its step lists through one shared loop.

---

## 2. Action space per title

### 2.1 Principles

1. **An index space is identified by `(title, layout_version)` and frozen once a
   checkpoint trained on it exists.**
   - The 1830 v1 layout is `build_layout` today and never changes; the frozen test
     stays.
   - Each new title gets `layout_fingerprint(title)`: a hash of the ordered
     (family, axes, entries) list, pinned by a cargo test. An intentional change
     means a version bump plus a re-converted LMDB.
2. **Layouts are generated from title data, not written by hand.**
   - Axes come from `companies()` (auctionable only), `corporations()` grouped by
     `corp_type`, `trains()`, `action_hex_order()`, `action_tile_order()`, market
     par cells, the dividend kinds the title's Dividend step offers, and a
     title-declared choice vocabulary.
   - Title code declares *which* families exist (`RlSpec::families()`); the
     builder does the rest.
3. **Every index carries a descriptor** (`IndexDescriptor { family, a, b, hex,
   city, tile, rot, param }`), and the frozen 1830 layout gets one too, derived
   by walking `build_layout`'s blocks. The generic policy and price heads read
   descriptors, never offsets. The 1830 index space can therefore feed the
   generic model without re-indexing.
4. **One legal action maps to one index, and each index means at most one legal
   action in a given state.** The only intentional collapse is price, which lives
   in `price_range` plus the price head. Collapses like 1830's
   `BuyTrainDTradeIn` (which train is traded in is left to decode) are not
   allowed in new layouts: if the choice changes the outcome, it gets an axis.

### 2.2 The block registry (families)

A family is a named block with an ordered tuple of axes. `index = block_offset +
mixed_radix(axis positions)`. Axis entry lists are snapshotted from title data
when the layout is built.

| Family | Axes (from title data) | Price | Titles (from the corpus action census, appendix A) |
|---|---|---|---|
| `Pass` | (none) | | all |
| `Bid.company` | auctionable company | yes | 1830 (waterfall), 1867/1861 (single-item), 1846 (draft), 1822 family (bid boxes on companies, minors, concessions), 1871, 1828, 18 India |
| `Bid.corporation` | corp (by type) | yes | 1867/1861 SR minor founding; 1817/18USA IPO auctions |
| `Par` | corp x par cell | (cell) | most |
| `BuyShares` | corp x source{ipo/treasury, market} (x count where multiple buys are allowed) | | all |
| `SellShares` | corp x count (1..pool_limit / unit) | | all |
| `CorpShares` | target corp x op{redeem, issue} x count | | 1867/1861 redeem; 1817/18USA issue/redeem; 18 India corporate buy/sell |
| `PlaceToken` | city slot (hex, city): the **max** city count over the hex's upgrade closure | | all (also SR home tokens) |
| `RemoveToken` | city slot | | 1867/1861, 1862, 1817, 1822MX |
| `LayTile` | hex x tile x rotation(6) | | all |
| `BuyTrain.depot` | train type | | all |
| `BuyTrain.pool` | train type (discard pool at face) | | all |
| `BuyTrain.exchange` | new type x traded type | | 1830 (D: in the frozen v1 shape), 1867 (8, 2+2, 5+5E x {5,6,7,8,2+2,5+5E}) |
| `BuyTrain.corp` | seller corp x train type | yes | all with cross-corp sales |
| `DiscardTrain` / `ScrapTrain` / `SellTrain` | train type | | all / 18USA / 18 India |
| `Dividend` | kind (title list: payout, half, withhold, ...) | | all |
| `BuyCompany` / `SellCompany` | company | yes | most / 18 India |
| `RunRoutes`, `Bankrupt` | (none) | | all |
| `Merge` | corp (a minor to add, or the major that ends or receives a merge: the state disambiguates) | | 1867/1861, 1822 family, 1862, 1817/18USA, 1828 |
| `Convert` | (none) or corp | | 1867/1861, 1862, 1817/18USA |
| `Choose` | choice id (per-title vocabulary, e.g. 1867 `nationalize`) | | 1867, 1822 family (choices round), 1862 (parliament/refusal), 1871, 1828, 1849 |
| `TakeLoan` / `PayoffLoan` | (none) or count | | 1817/18USA (6-7% of actions), 1849 |
| `Short` | corp | | 1817/18USA |
| `Assign` | target entity (corp or hex) | | 1846, 1822 family, 1817/18USA, 1849 |
| `AcquireCompany` | company | yes? | 1822 family |
| `CompanyAbility.*` | per ability, axes from `AbilityDef` (the CS/DH sub-blocks generalized) | | most |

*Not every listed family exists yet. Each new title adds the families it needs
and writes its generator(s) (§2.4). The registry is the contract.*

### 2.3 The 1867 layout concretely

| Block | Size | Notes |
|---|---|---|
| Pass | 1 | |
| Bid.company | 5 | C&SL, NFB, MB, QB, SCT. Hidden "3" excluded (`auctionable: false`). |
| Bid.corporation | 16 | Minors. Price: min $100, Ruby `MIN_BID`. |
| Par | 8 x \|major par cells\| | Cells tagged par_2/par, from $70 to $200 |
| BuyShares | 8 x 2 | Majors only; a minor is one 100% certificate |
| SellShares | 8 x 5 | 50% pool / 10% unit |
| CorpShares (redeem) | 8 | The acting major buys back its own market share |
| PlaceToken | ~20 | Max cities over each hex's upgrade closure. L12 = 3 once the triple city lands (§6). |
| RemoveToken | ~20 | Same city slots |
| LayTile | 94 x 68 x 6 = 38,352 | |
| BuyTrain.depot / pool | 9 + 9 | 2, 3, 4, 5, 6, 7, 8, 2+2, 5+5E |
| BuyTrain.exchange | 3 x 6 = 18 | Phase-8 trade-ins: the discount depends on the traded type (`g1867.rs:712-719`) |
| BuyTrain.corp | 24 x 9 | Price-bearing |
| DiscardTrain | 9 | |
| Dividend | 3 | payout, half, withhold |
| BuyCompany | 5 | Price $1..face (`company_buy_price_range`) |
| RunRoutes, Bankrupt | 2 | |
| Merge | 24 | Minors and majors |
| Convert | 1 | |
| Choose | 1 | `nationalize` |

The total is about 38,900, 97% of it LayTile, against 1830's 26,537. The
policy head never materializes it (§2.6).

### 2.4 Legal-action enumeration: step-driven, beside `factored.rs`

- The engine already computes the action *types* through each title's step list
  (`steps.rs::step_actions`, one shared accumulation loop).
- The new enumerator (`engine-rs/src/rl/enumerate.rs`) maps each
  **(StepKind, action type)** pair to a generator that yields `RlAction { family,
  args, price_range }`.
- Generators for shared step kinds (Track, Token, Route, Dividend, BuyTrain,
  DiscardTrain, BuySellParShares) are written once with `GameTitle` hooks, reusing
  the title-generic helpers that already exist:
  - `lay_tile_candidate_hexes`, `factored_layable_tiles_for` (after the wall fix);
  - `tokenable_cities_for`;
  - `buyable_shares`, `sellable_bundles`, `buyable_trains_for`.
- Title-only step kinds get generators in `rl/titles/gXXXX.rs`. For 1867 these
  are SingleItemAuction, the SR CorporationBid and home token, Merge,
  PostMergerShares, ReduceTokens, MajorTrainless, RedeemShares and
  BuyCompanyPreloan.
- **A StepKind or type with no generator is a hard error in debug and fuzz
  builds.** The fuzzer then finds it immediately; it never becomes a silent
  "no legal moves".

Why not extend `factored.rs`:
- It mirrors Python's `FactoredActionHelper`, and the 1830 training data depends
  on that parity (`docs/rust_engine_*_audit.*`).
- It dispatches by action type, so it cannot tell one step's `buy_shares` (an SR
  buy, a PostMergerShares buy, a redemption) from another's.
- Another agent is editing it right now.

How the new enumerator earns trust: on 1830 it must produce exactly the index
set `factored_legal_indices` produces. Run this on in-crate random walks and on
the 3,243-game corpus. It proves the shared generators against a Python-verified
oracle before 1867 depends on them. Eventually 1830 could move to it.

Enumeration must be **sound** (every enumerated action applies) and **complete**
(every legal action is enumerated). Replay never tested either, because the
replay handler trusts the record. §6 lists the 1867 places where enumeration
needs rules the replay path never checked.

### 2.5 Index to decode

- `decode.rs` decodes by re-enumerating and picking the action whose index
  matches (`legal_action_for_index`, `decode.rs:91-98`), then building the
  `Action` (`build_action`, `:396-812`). That is title-generic as soon as:
  - the lookup uses the game's layout;
  - every family has a builder. Merge, Convert, RemoveToken (with the slot read
    back from the city), Choose and CorporationBid are new.
- `process_action_native` already logs a faithful 18xx.games wire dict
  (`decode.rs:140-150`). That is what makes the Ruby replay of fuzz games
  possible (§2.8).

### 2.6 Generalizing the policy head

The legacy `HierarchicalPolicyHead` (`model_transformer.py:999-1350`) has two
parts:
- autoregressive LayTile (P(hex) from per-hex embeddings, P(tile|hex),
  P(rot|hex,tile)) and PlaceToken (P(hex), P(slot|hex)) heads. These are
  already geometry-generic.
- a flat `other_head = Linear(d_trunk, num_other)`, whose rows are tied to
  absolute 1830 slots.

The generic head (`DescriptorPolicyHead`) changes three things:

1. **It scores only the legal indices.**
   - The inputs are the batch's legal index lists (MCTS and the data already
     carry them) and their descriptors, gathered from a per-title descriptor
     table.
   - `logit_i = MLP(trunk ⊕ E_family ⊕ tok[a] ⊕ tok[b] ⊕ E_param)` for non-map
     families.
   - Map families keep the autoregressive factorization: LayTile, PlaceToken and
     RemoveToken read hex token embeddings, and tile embeddings computed from
     **tile features** (color, exits, stops, revenue, label), so tiles generalize
     across titles.
   - `tok[a]` is the referenced entity's output token (corporation, company,
     train type, seat), so the same weights score "buy shares of X" for any title.
2. Each family is normalized inside its block, as today (each block is a
   distribution). The softmax over legal actions in MCTS and training rebalances
   between blocks. Training targets stay a distribution over legal indices.
3. Cost scales with the number of legal actions (usually under 200; tile-lay
   states up to a few thousand), not with `policy_size`. LayTile at 38k slots
   (1867) or about 100k (1822) is no problem.

The legacy head stays for legacy 1830 checkpoints. The generic model on 1830
uses descriptors for the frozen 1830 layout.

### 2.7 Generalizing the price head

The cells (atoms plus bins, `price_pmf.rs` / `price_pmf.py`) already generalize.
What is 1830-specific is the slot list (60 fixed `(type, entity)` slots) and the
cell specs keyed by three type names. The generic version:

- **Slot = the price-bearing index's descriptor.**
  - Head input is trunk ⊕ E_family ⊕ tok[a] ⊕ tok[b], the same as the policy
    scorer, then `NUM_CELLS` logits.
  - `price_head_entity_key` (`mcts.rs:121-190`) and `price_head_slot_for_index`
    (`decode.rs:851-893`) become a descriptor lookup.
- **Cell spec per family and title**: `(n_atoms, n_bins, atom rule)`.
  - Bid families use a ladder `lo + step*j`, with the step from title data
    (`bid_price_step`; 1822 bid boxes may need their own).
  - BuyTrain.corp uses `{lo, hi, hi-1}`; BuyCompany uses `{lo, hi}`.
  - Specs are declared in `RlSpec::price_specs()`. Before fixing atom positions
    per title, run a per-title human-price census (`scripts/price_census.py`,
    generalized). The 1830 atoms came from that census.
- MCTS progressive widening over prices is unchanged.

### 2.8 Verification per title (the gate before any training data is trusted)

All of these run as cargo tests (fast, in-crate) and as a Python CLI over corpora.

1. **Layout tests.**
   - Blocks are contiguous; `total` is the sum of block sizes; descriptors are
     unique; axis entries resolve to live entities.
   - The fingerprint is pinned per title; for 1830 the frozen test stays.
2. **Enumerate-and-apply fuzzing** (`rl/verify.rs`, extending the
   `test_title.rs:259` pattern to every bridge-ready title). Random-walk N games,
   for every player count, at every decision:
   - every legal action has an index, and no two legal actions share one;
   - `decode(index)` applies, on a clone, to the same state hash as applying the
     enumerated action directly;
   - every price in `price_range` applies (spot-check lo, hi and a random price);
   - money is conserved;
   - nothing panics. The remaining `unimplemented!` calls (`merger.rs:1097`,
     `game.rs:1306`) must be unreachable from enumeration or turned into
     `GameError`;
   - the game ends within the title's move bound.
   Uniform random walks rarely reach late-game states, so also walk from human
   mid-game prefixes, and from a 1-ply greedy policy that prefers
   train-buying/progress actions.
3. **Corpus completeness.** Replay every usable human game. At each human
   decision the recorded action must match **exactly one** enumerated index (the
   generalized `_matching_choices` rule). Report the match rate by action family;
   the target is 100% minus documented record quirks. This doubles as the
   pretraining label check.
4. **Ruby replay of fuzz games (soundness against the real oracle).**
   - Export fuzz and early self-play games (wire dicts from
     `process_action_native`).
   - Replay them in Ruby HEAD with the docker rig (`scripts/g1867_ruby_dump.rb` /
     `g1867_ruby_diff.py`, generalized to any title).
   - Any Ruby rejection, or a per-action state divergence, is a Rust enumeration
     or rules bug, or a documented Ruby-HEAD rule change.
   - This is the only check that catches *illegal-but-applies* moves: tile lays
     through walls, unaffordable tokens, wrong merge candidates. Replay-based
     audits cannot see these.
5. **Encoder tests.** Shapes and masks per title; every pointer resolves; no
   NaN or infinity; seat rotation is involutive. For 1830 only, generic-encoder
   features are cross-checked against the legacy encoder where both define them.

---

## 3. Encoder per title

### 3.1 What is title-generic: a typed-token bundle

`BaseGame::encode_tokens()` (Rust, `rl/encode.rs`) returns:

| Tensor | Shape | Content |
|---|---|---|
| `hex_feat` | `[H, F_hex]` | See below |
| `hex_ptr` | `[H, S_max]` | Entity index in each token slot on the hex, ordered by (city, slot); -1 = empty, special ids for neutral and reserved. S_max = 6. |
| `ent_type` | `[E]` | player, major, minor, national, company, train type, phase, global, plus later types (concession, bond, ...) |
| `ent_feat` | `[E, F_ent]` | Per-type schema packed into one width; unused columns are zero |
| `ent_ptr` | `[E, P]` | Pointer fields: owner, president, home hex, merge target, ... |
| `rel` | sparse `(i, j, k, value)` | Relations: percent of j held by i, president-of, owns company, trains of type j held, loan count |
| `meta` | | Title id, round kind, step kind, acting entity, `num_seats`, seat rotation, layout id |

Feature schemas (all title-generic semantics):

- **Hex:**
  - layout-aware axial coordinates;
  - tile color one-hot;
  - label id (a global label vocabulary: OO, Y, O, B, NY, Chi, T, ...);
  - cities, slots, towns, offboard, port;
  - current-phase stop revenue and all four tiers;
  - terrain/upgrade cost; per-edge wall and water-cost (6 x 2);
  - track: 15 edge-pair bits plus 6 x 3 edge-to-stop bits;
  - rotation; blocked-by-ability flag; preprinted flag.
- **Player:** cash / starting cash, certificates / limit, priority, turn-order
  position, committed bid cash, net worth / starting cash, bankrupt flag.
- **Corporation:**
  - kind (major, minor, national, ...); floated, operated, closed;
  - cash; share price; par; market row and column; zone flags over the full
    `MarketZone` set;
  - share units in IPO/treasury and in the market; total units (2, 5 or 10);
    capitalization mode;
  - tokens remaining / total; loans and the loan limit;
  - train limit (by kind); operating-order position; per-seat percent held
    (`MAX_SEATS` wide, seat-rotated).
- **Company:** face, revenue, closed, auctionable, current bid per seat, min bid,
  up-for-auction flag, ability flags (blocks, teleport, tile lay, exchange,
  hex bonus, shares).
- **Train type:**
  - distance; per-stop-kind buckets (pay/visit, so 1867's "towns free" and the
    5+5E are features, not code); multiplier; price;
  - count in depot, count in pool, unlimited flag;
  - rusts-on / obsolete-on phase pointers; available-on; discount count.
- **Phase:** current flag, train limits, OR count, tile colors, status flags.
- **Global:** bank / initial bank, OR k of n, game-end triggers latched, turn,
  loans available.

How the model uses them:
- Pointers are resolved in-model: after layer 0, add a projection of the pointed
  token's embedding.
- Relations become a per-head attention bias in the entity transformer, the same
  mechanism as the hex map's `StructuralAttentionBias`, and a bias in the
  entity-to-hex cross-attention (corporation j tokened on hex h).

### 3.2 What is title-specific (`RlSpec`)

- Vocabularies the title introduces: new entity kinds, choice ids, label ids,
  dividend kinds, round and step kinds.
- Normalizers derived from data:
  - max market price; max stop revenue; starting cash per player count; bank;
  - cert limits; max trains per type (unlimited becomes the flag, plus
    log1p(count));
  - max hex distance.
- Map geometry: hex list, `HexLayout`, adjacency (already per title), axial
  conversion per layout. The model's distance and direction matrices are computed
  by Rust per title and cached per title.
- Title extras that do not fit a schema (1822 concessions and destinations,
  1817 short interest, 1862 parliament state) go into a small per-type "extra"
  slice of `ent_feat`, declared by the title. The extra slice is zero for every
  other title.

### 3.3 Variable counts in the transformer

- Hex and entity sequences are padded to the batch maximum, with key-padding
  masks (the mechanism already used for player tokens,
  `model_transformer.py:767-782`). H is 93 for 1830 and 94 for 1867; E is about
  56 for 1867 and about 100 for 1822.
- Token identity is the type embedding, plus the title embedding, plus an
  optional per-(title, entity) learned id embedding, which is zero-initialized
  and weight-decayed so features dominate. If titles ever share weights
  (§4.2), the per-title id embedding is the only unshared part.
- Seats are canonicalized: seat 0 is the acting seat, the same rule as
  `acting_player_id_pub`, `game.rs:353-367`. Padded seats are masked. The value
  head emits `MAX_SEATS` logits, with padded seats at -1e4 (as now,
  `model_transformer.py:1985-1989`).
- Round and step: FiLM input = E_title + E_round_kind + E_step_kind, so the
  merger round, 1822's choices round and 1862's parliament each get their own
  conditioning, not "Stock".

---

## 4. Model: one per title, or a shared trunk

### 4.1 The data

Recent finished human games per title, from the 2026-10 scrape (appendix A):

| Title | Recent games |
|---|---|
| 1846 | 4,180 |
| 1889 | 2,031 |
| 1830 | 1,796 (about 4,100 with older; about 2,900 used in pretraining) |
| 1862 | 1,230 |
| 18USA | 958 |
| 1871 | 917 |
| 1822 | 743 |
| 18 India | 708 |
| 1849 | 676 |
| 1861 | 649 |
| 1822PNW | 597 |
| 18Chesapeake | 531 |
| 1817 | 489 |
| 1867 | 367 (about 1,000 replayable once the 2020-22 corpus is counted, §6) |
| 1822MX | 363 |
| 1822CA | 187 |
| 1828 | 72 |

### 4.2 One shared model across titles: candid feasibility

Two questions tend to get merged here; they need separate answers.

- **(A) One architecture and codebase for every title.** Recommended
  regardless. Per-title models need it too: the alternative is a hand-written
  flat encoder, action mapper and model wiring per title, like 1830's
  (`encoder.py`, `action_mapper.py` and the gather tables in
  `model_transformer.py` add up to about 3,000 lines), times sixteen.
- **(B) One set of weights trained on several titles.** A hypothesis. The
  evidence is mixed, and it is cheap to test before committing (§4.2.6).

#### 4.2.1 What would be shared, and what stays per title

The action spaces being different is *not* the blocker. Indices stay per title,
and the descriptors (§2.1) map them into one scoring function. The real
question is whether shared *representations* help.

| Component | Shared? | Notes |
|---|---|---|
| Token input projections (player, corporation, company, train type, phase, hex) | Shared | Feature schemas are title-generic by construction (§3.1). Title-only "extra" slices get a zero-initialized per-title projection. |
| Hex transformer: axial positions, distance/direction/track bias | Shared | Track physics is the same in every title. Pointy and flat layouts need a layout embedding, because direction ids mean different geometric directions. |
| Entity transformer, relation bias, entity-to-hex cross-attention | Shared | |
| FiLM trunk | Shared | Conditioned on title + round kind + step kind |
| Value and score heads (per seat) | Shared | |
| Policy scorer and price head for **common families**: Pass, Par, Buy/SellShares, LayTile, PlaceToken, BuyTrain.*, Dividend, BuyCompany, DiscardTrain, RunRoutes | Shared | The family embedding means the same thing in every title ("buy a share of X") |
| Families unique to one title family: 1822 bid boxes and choices round, 1817/18USA shorts, loans and acquisitions, 1862 parliament, 1867 merge/convert | Effectively per family | Only those titles send gradient through them, so "sharing" them is moot |
| Entity id embeddings, title embedding, descriptor tables, layouts, normalizers | Per title | |

So a shared model is a shared *trunk and common-family heads* plus per-family
heads, all conditioned on the title.

#### 4.2.2 What transfer could buy, and where titles conflict

**Plausible transfer** (same mechanics, same meaning):
- **Reading the map.** Track connectivity, which hexes carry revenue for a
  corporation's trains, blocking. Tile geometry is shared across titles: the
  tobymao tile ids overlap heavily between 1830, 1889, 1846, 18Chesapeake and
  1867.
- **Route value of a tile lay**, the hex transformer's core skill.
- **Stock-market basics.** A payout raises the price and a withhold lowers it;
  president dumps; cert limits; selling pressure.
- **The value head's bookkeeping**: who is ahead from cash, shares and
  holdings.
- **Train-rush and rust timing as a concept.** Rust and obsolete phases are
  features (pointers), not code.

**Conflicts** (same features, different meaning or different best play):
- **Openings.** Waterfall (1830), single-item with dutch fallback (1867), bid
  boxes (1822), private draft (1846), corporation IPO auctions (1817). Openings
  are where policies differ most.
- **Capitalization.** Full (1830) against incremental (1867, 1817): buying an
  IPO share means something different for the corporation's cash.
- **Financing.** Loans and shorts (1817, 18USA) change what the stock market is
  for: a falling price can be good for the shorting player. 1830 has neither.
- **Minors and mergers.** 1867's merger rounds, 1822's minor acquisitions,
  1862's parliament.
- **Game end.** Bank break against final-phase OR sets changes what late-game
  moves are worth.

Conflicts are where interference lives. Title conditioning (FiLM, title
embedding) is the mechanism that separates them, and capacity decides whether
it can.

#### 4.2.3 Evidence from published multi-game work

From memory; verify the specifics before quoting any of them.

| Work | Setting | Finding |
|---|---|---|
| AlphaZero (Silver et al., 2018) | Chess, shogi, Go | One algorithm, **separate network per game** |
| KataGo (Wu, 2019) | Go | **One network across board sizes 9-19 and several rulesets**, with the rules as input features. Works well. Variants of one game share well when the rules are inputs. |
| Polygames (Cazenave et al., 2020); Soemers et al. (Ludii, transfer of fully convolutional policy-value nets, ~2021-23) | Many board games | Transfer between **variants of one game** often helps: partly zero-shot, more after fine-tuning. Between **different games** results are mixed and sometimes negative. |
| Policy distillation (Rusu et al., 2016); Actor-Mimic (Parisotto et al., 2016) | Atari | Distilling several single-game experts into one network gets close to expert level. Transfer to new games speeds some up and slows others. |
| IMPALA (Espeholt et al., 2018); PopArt (Hessel et al., 2019) | DMLab-30, Atari-57 | One multi-task RL agent showed positive transfer on DMLab-30. On Atari-57 it needed reward normalization (PopArt) to reach median human level, and **stayed below single-game specialists**. |
| Multi-Game Decision Transformer (Lee et al., 2022) | ~40 Atari games, offline | One transformer trained on mixed offline data. Transfer and fine-tuning efficiency **grew with model size**; small models gained much less. |
| Gato (Reed et al., 2022) | Hundreds of tasks | Very large generalist; competent on many tasks, generally below specialists |
| Player of Games (Schmid et al., 2021) | Perfect and imperfect information games | General algorithm, separate networks per game |

How this applies here:
1. **Three regimes, three levels of evidence.**
   - Variants inside a family (1822/CA/MX/PNW, 1867/1861, 1817/18USA) look like
     KataGo's rulesets and Ludii's variants. The evidence **favors sharing**.
   - Joint *supervised* pretraining on mixed human data looks like the Decision
     Transformer and distillation results: **positive, but scale-dependent**.
   - One RL-trained network for many distinct titles looks like IMPALA/PopArt on
     Atari: the **weakest** evidence, and specialists win there.
2. **Scale.** The Decision Transformer's transfer showed up in large models.
   Ours must stay small (about 5-30M parameters), because every MCTS leaf is a
   forward pass and self-play throughput is the budget. At small scale,
   interference is more likely than at Gato scale.
3. **No direct precedent.** Nothing published covers multi-title 18xx. The titles
   are more alike than distinct Atari games (same tiles, same stock-market
   logic) and less alike than Go rulesets.

**Conclusion.** The evidence supports family-level sharing and joint supervised
pretraining followed by per-title RL. It does **not** support committing to one
RL-trained network for all titles.

#### 4.2.4 Data sizes and the small titles

Decisions per title are roughly games x median actions (appendix A):

| Title | Games | Decisions |
|---|---|---|
| 1830 | ~2,900 | ~2.2M |
| 1867 | ~1,000 replayable | ~0.7M |
| 1822CA | 187 | ~250k |
| 1828 | 72 | ~76k |

- **Per-title supervised training on 70-400 games will be weak whatever we do.**
  - On 1830's ~2,900 games the value head memorizes in under an epoch.
  - Rare action types (Par) already degrade under long pretraining on 1830.
  - 1828's unique decisions (its mergers, its private set) have a few hundred
    examples in total.
- **Family pooling changes the small titles' situation the most, where the
  evidence is strongest.** Pooled human games:

  | Family | Pooled games |
  |---|---|
  | 1822 + CA + MX + PNW | about 1,890 (1822CA alone: 187) |
  | 1867 + 1861 | about 1,650 |
  | 1817 + 18USA | about 1,450 |

- **1828 has no family.** It stands alone in Ruby. Sharing could help it only
  through the generic skills (map, stock market, value bookkeeping). Its unique
  parts stay data-poor either way. Honest options: start RL from a related
  base, or deprioritize 1828.
- **Data-rich titles gain little.** 1846 (4,180), 1889 (2,031) and 1830 (~2,900)
  are unlikely to gain much at the supervised stage and carry the interference
  risk; train them per title.
- **Human data matters mainly for the starting policy.** The AlphaGo-style
  pipeline uses human data to get a reasonable starting policy; RL then has
  unlimited self-play data. So the value of sharing is concentrated where human
  data is thin and a bad start is costly. That is the small titles, which the
  family pools already cover.

#### 4.2.5 Costs and risks of sharing

- **Engineering.** The generic architecture (S4-S8) is needed for per-title
  models anyway. Weight sharing on top adds about 5-8 agent-days:
  - title embedding;
  - mixed-title LMDB and a title-balanced sampler;
  - inference-server routing by title;
  - a per-title eval matrix;
  - multi-title checkpoint metadata.
- **Capacity and throughput.** A shared model across four families probably
  needs 2-4 times today's 7.3M parameters. That means slower leaves and fewer
  self-play games per hour. Mitigation: share only in pretraining; distill the
  fine-tuned specialists into small per-title nets if inference cost bites.
- **Interference and imbalance.**
  - A naive mix is dominated by 1846's 4,180 games. It needs title-balanced
    sampling (e.g. p ∝ n^0.5) and per-title loss tracking.
  - Gradients conflict on the conflicting concepts (§4.2.2).
- **Coupled releases.** Every change to a shared base touches every title, so
  each release needs per-title evals and iteration slows. Per-title fine-tunes
  decouple releases.
- **Checkpoint compatibility.**
  - The frozen 1830 *layout* is fine: descriptors make the v1 indices usable, so
    1830 data is reusable after re-encoding the states.
  - The **legacy 1830 weights cannot load** into a generic or shared model
    (different encoder and heads), so a shared model starts a new lineage.
  - Bringing 1830's RL strength in costs a policy distillation from the current
    best 1830 checkpoint on 1830 states, which the distillation literature says
    works.
- **Debuggability.** A regression in one title can come from another title's
  data.

#### 4.2.6 A cheap experiment, and the result that would justify sharing

**E0, representation probe (early).**
- Prerequisites: only the generic encoder (the S7 encoder half) and 1867 replay,
  which already works. No 1867 enumerator or layout is needed.
- Set-up:
  - Train the trunk with three heads:
    - a value head (final result);
    - a head predicting the human action's **family** (legal types come from the
      step machinery, which already works for 1867);
    - a LayTile *hex* head (target = the hex the human laid on).
  - Arms: (a) 1867 alone and (b) 1830 + 1867, with title-balanced sampling.
  - Repeat each arm at 75, 150, 300 and about 800 training 1867 games, against a
    fixed held-out set (`in_validation_split`).
- Metrics:
  - held-out 1867 value CE against equal odds;
  - action-family NLL;
  - hex top-5;
  - for arm (b), the 1830 held-out metrics against 1830-only (the interference
    check).
- Cost: about 1 GPU-day (8 small runs) and 2-3 agent-days. It can run around
  week 3-4.

**E1, full supervised comparison (once 1867 labels exist, T7d).**
- Set-up: full policy + value + price pretraining, 1867-only against joint
  1830 + 1867, same model size, at full 1867 data and at a 150-game subsample.
- Metrics:
  - held-out human move prediction: per-family top-1 and NLL, overall NLL;
  - value CE;
  - policy-only 1867 head-to-head, joint against separate (about 480 games, 4
    players, 2v2 seat rotations, with a confidence interval);
  - 1830 held-out metrics for the joint model.

**Decision rule (fixed before running).**
- **Pursue cross-family sharing only if all three hold:**
  - at the 150-game subsample, joint beats separate on 1867 held-out NLL by at
    least 5% relative (or by at least 2 points of top-1 on most families), with
    clearly lower value CE;
  - at full data, joint is no worse than separate: NLL within 1%, and the
    head-to-head confidence interval includes or exceeds 0.5;
  - 1830 held-out NLL degrades by at most 1%.
- **If joint helps only at the subsample:** use shared pretraining as a warm
  start for the small titles (1822CA, 1822MX, 1828), and per-title models for
  the data-rich ones.
- **If joint never helps:** drop cross-family sharing. Test family sharing on
  its own when two members exist (1822 against 1822CA; a separate agent has now
  started the 1822 rules engine), or go fully per title.

E0 and E1 test **cross-family** transfer. 1830 and 1867 differ in minors,
mergers and capitalization, so this is the hard case. A positive result is
strong evidence; a negative one does not rule out sharing within a family.

### 4.3 Recommendation

1. **Build the generic architecture (A) regardless.** It is what makes 1867 and
   the next fourteen titles affordable at all.
2. **Default to per-title models**, trained from scratch on their own human data
   and then RL. For 1867, an 1867-only model is the baseline; the joint model is
   the comparison arm of E1.
3. **Treat family pooling as the first sharing step**: 1822 with CA, MX and PNW;
   1867 with 1861; 1817 with 18USA. Do it once two members of a family exist. It
   is the best-supported and lowest-risk form of sharing, and it is where the
   small titles live.
4. **Do not commit to one model across all titles.** Run E0 early. Adopt
   cross-family shared pretraining only if E1's decision rule passes. RL
   fine-tuning is per title in every case.

### 4.4 Validating the generic model on 1830 first

Before 1867 relies on the generic model, train it on 1830 (re-converted
`lmdb_v6k`: the same games, the new encoder and descriptors, the same 1830 v1
indices). Compare it with the legacy pretrained model on:
- per-type top-1, and Par/BuyTrain/LayTile separately;
- value CE against the equal-odds baseline;
- policy-only head-to-head (`eval_policy_only.py`).

The pass bar is "within noise". 1867 has no baseline, so this is the only
architecture-level check. It needs GPU time (decision 5).

### 4.5 Player-count width

- Corpus maxima: 1822 7 players (21 games), 1822CA 7 (4), 18USA 7 (3),
  1862 8 (23), 1817 8 (1), 9 (1) and 12 (5).
- **`MAX_SEATS = 8`** covers every game except seven 1817 games with 9 to 12
  players. The value head is already one shared MLP applied per seat
  (`model_transformer.py:1608-1621`, `:1976-1981`), so widening it costs no
  weights.
- The cost is MCTS arrays (`[f32; 8]` per child, +33%) and padded seat tokens.
- The legacy 1830 checkpoint keeps width 6; its values are padded to 8 at the
  `rust_mcts_player` boundary.

---

## 5. Pipeline: what changes per title

### 5.1 Data schema (LMDB v3)

- A DB-level `__meta__` record: title(s), layout ids, encoder version, value
  frame.
- Per row: `(title_id, token_bundle, legal_idx int32[], pi float16[len(legal)],
  value[num_seats], num_seats, seat_rotation, price_targets with descriptor
  slots)`.
- Self-play no longer drops `rotation` or `num_players` (today it stores
  `encoded_state[:6]`, `dataset.py:292`).
- Sparse pi removes about 106 KB of dense floats per row (26,537 x 4 bytes). It
  matters more for 1822-sized layouts.
- The v2 reader stays for legacy data.

### 5.2 Human import and pretraining

- `filter_actions` is reused unchanged.
- Engines are built with `new_titled(title, players, title_optional_rules)`.
- **Label matching moves to Rust:** `game.match_human_action(wire_dict) ->
  Vec<index>`. Enumerate at the state, keep the legal actions whose wire form
  matches, and resolve the same things `_resolve_human_action` resolves (city
  through the board, train source by seller, exchange) generically.
  - More than one match means the action is ambiguous: skip it and count it,
    the existing policy.
  - Zero matches means an enumeration-completeness bug (§2.8.3).
- Drop and leniency rules become per-title data. For new titles they should be
  almost empty: the Python-parity drop rules (`mh_out_of_turn`,
  `ruby_depot_phase_skip`, ...) exist only because 1830 had two engines.
- Value target is our engine's `result()`, which equals Ruby HEAD's (for 1867,
  `recorded_drift` games are usable training data).
- The validation split stays `in_validation_split(game_id)`; title is part of the
  DB.
- The two-stage pretraining (joint run, then value-head refit) is unchanged.
  Keep forced positions (as `lmdb_v6k` does).

### 5.3 Self-play and MCTS (Rust tree)

- The adapter is not needed on the new-title hot path. What the loop uses is:
  - `finished`, `result`, `encode_tokens`, `factored_legal_indices`,
    `price_range_for_index` and `price_head_slot_for_index` (title-aware), and
    `advance_root`;
  - acting seat and round kind (native getters), and `raw_actions` / `to_dict`.
  Each of these gets a native method so `rust_mcts_player.py` and
  `policy_selfplay.py` work on the raw engine object for every title.
  `RustGameAdapter` remains for 1830 cleaning and Python MCTS.
- Add `incorporate_results_sparse` to `mcts.rs`. The inference server already
  answers sparse (`legal_lengths`); stop densifying (`inference_server.py:809`).
- Value width `MAX_SEATS`; c_puct keyed by round kind; forced-move collapsing
  and decision counting are unchanged (already generic).
- `auction_unlock` and `auction_stall_moves` are 1830 waterfall training wheels;
  they become a per-title "stall policy". 1867's dutch auction cannot lock.

### 5.4 Policy-only self-play and policy gradient

- `NUM_PLAYERS` becomes a per-run player-count distribution from `TitleSpec`.
- League (PFSP) and advantage code are already N-generic.
- Rewards:
  - `win`/`placement`: placement points per N. Suggested `(N-1-rank)/(N-1)`, or
    keep the 4-player table and interpolate.
  - `president:<CORP>`: corp list from the title.
  - `rusher` and `rust`: phase/rust events (generic).
  - `bankrupt`: only for titles where bankruptcy ends the game (a title flag).
- KL-to-SL anchor is unchanged (per-title SL checkpoint, or the shared base).
- Saved game records carry the engine's title (`game_records.py:64`).

### 5.5 Evals

- `eval_head_to_head` and `eval_policy_only` take `--title` and `--players N`.
- Arrangements are generated per N (A gets k of N seats, for all rotations or a
  balanced subset); equal strength is k/N.
- Scripted rule baselines (`eval_policy_only.py:98-155`, 1830 B&O and train
  scripts) are per title or dropped.
- `PeriodicEvaluator` and `current_best` are keyed per (title, architecture).

### 5.6 Start positions

- One generic builder, `build_start_positions.py --title T --cut <predicate>`,
  with title-defined cut predicates: "first decision of SR1", "start of SR k".
  Output `start_positions_<title>_<N>p.jsonl` with a `title` field; loaders key
  on (title, N).
- No Python-engine parity check for new titles. Instead the prefix must replay in
  Rust, and it was already validated by the corpus audit.
- Random starts are optional per title: 1830's generator knows SV and B&O.
- For 1867, recommended (decision 6):
  - about 80% of games start at SR1 from human post-auction positions (the
    auction is 5 companies plus dutch fallback; short);
  - about 20% are full games including the auction;
  - mid-game cuts from human games, as for 1830.

### 5.7 Checkpoints

- Config stores titles, layout fingerprints, encoder version and `MAX_SEATS`.
- Loading refuses on any mismatch. Silently padding or dropping head tensors
  (`checkpointer.py:126-265`) must not apply to new layouts.
- Directory layout `model_checkpoints/<arch>/<title|base>/...`.

---

## 6. 1867 specifically: everything needed to make it trainable

These are in addition to the generic layer (§7, S1-S8). Effort is in agent-days.
"Ruby-verified" means the fuzz-game Ruby replay (§2.8.4) passes for that
feature.

### Rules gaps only self-play reaches (the replay path never needed them)

| # | Gap | Evidence | Work | Effort |
|---|---|---|---|---|
| T1 | **Impassable borders.** 1830 walls are hardcoded by hex id, so 1867's C11, C13, D12, F8, C17, B16 and E7 get 1830's walls, and 1867's own walls (D18, C9, D10, E11, C11, D12, C13, H14, A17, A19, E19, H16, and D2's DSL borders) are ignored. | `factored.rs:1138-1147`; data in `g1867.rs:1330-1460` | `passable_exits_for` and `legal_rotations_for` read `HexDef.borders` and DSL `border=edge:N` (no type = wall). For 1830 the data reproduces today's list exactly (test). Graph and frontier expansion (`lay_tile_candidate_hexes`) respect walls. | 1-2 |
| T2 | **Auto-router is 1830-only.** Trains are `(distance, is_D)`; every stop counts. 1867 needs: free towns (pay/visit buckets); the 5+5E paying offboards only, x2, with a tokened counted stop; x2 for 2+2; NFB/MB/QB/SCT +$10 hex bonuses; Timmins-capital +$40 x multiplier; phase-tier offboard revenue; grouped Detroit (exists). | `game.rs:4203-4205`, `router.rs:751-760`; the pricing exists in `revenue.rs::price_bucketed_stops` for recorded routes | Router walks with bucket-aware distance and prices candidate route sets with `revenue.rs` (shared with replay). Oracle: for every human `run_routes` in the corpus, the engine's optimum ≥ the human revenue, and equal in the large majority; inspect and Ruby-check any case above the human value. | 4-6 |
| T3 | **Tile-lay legality in enumeration.** The handler enforces the lay-slot rules (`rounds/operating.rs:467-500`) but not the tile's own legality (connectivity, upgrade path, exits, walls); the enumerator does the reverse and has no slot awareness. Needs: the two-lay `TILE_LAYS` in enumeration (second lay is not an upgrade after an upgrade, costs $20, cannot reuse the hex); $80 water borders and the $20 water hexes in affordability (enumeration checks terrain only, `factored.rs:898-899`); stubs G15/L10/K13 (`StubsAreRestricted`, TODO(1867-map)); the unowned M13 blocker (TODO(1867-blocker)); Y/O labels and J12's future label; X-tile upgrade rules; L12 as a **triple** city (today DoubleCity). | `g1867.rs:1310-1320`, `:620-628`, `:1443-1446`; `factored.rs:814-922` | Hooks per `TileLayDef` slot plus a stub check plus the triple city in the tile/`HexType` model. Ruby-verified. | 5-8 |
| T4 | **Tokens.** $40 x distance pricing in affordability; CN reservation and neutral tokens; minor/major home-token *choice* in the SR (HomeToken step in `stock_steps`, unenumerated); token removal choices in ReduceTokens. | `steps.rs:740`, `:784`, `:1108` (HomeToken arms); `stock_bid.rs:255` `process_stock_home_token` | Generators plus a Ruby check | 2-3 |
| T5 | **Trade-ins.** The three discounted trains need `BuyTrain.exchange[new x traded]`. Trade-ins are legal only after `train_trade_allowed` fires (the first 8-train must not be offered with a trade-in); the event is a no-op today. | `game.rs:1256`, `g1867.rs:712-719`, `action_index.rs:167-170` | Event flag, generator gate, handler gate | 1-2 |
| T6 | **States humans never reach.** The failed-merge state is `unimplemented!` ("undo-only"), `merger.rs:1097`, and so is an unknown train event, `game.rs:1306`. Self-play must not reach them. | as cited | Enumerator applies Ruby's president-affordability guard (`finish_merge_to_major`); turn the remaining panics into `GameError` (a panic kills a self-play worker) | 1 |

### 1867 RL pieces (once the generic layer exists)

| # | Item | Effort |
|---|---|---|
| T7a | 1867 generators: SingleItemAuction bid/pass (ascending + dutch, min = value - discount, max = cash); SR CorporationBid (≥ $100, cert limit, closed minors not re-biddable); major Par (par_2/par cells, startable from phase 4, incremental capitalization affordability); SR buy (treasury at market price) and sell (SELL_AFTER operate, `down_block_pres`); RedeemShares; BuyCompanyPreloan / BuyCompany (phase-gated, $1..face); BuyTrain (depot per type, pool, cross-corp with price, exchanges; AutomaticLoan buying power); Dividend (payout/half/withhold; minors auto); merger round (Merge/Convert/Pass with candidate rules ≤10 / ≤6 per owner, the merge-target major, PostMergerShares buy/pass in phases C/D, ReduceTokens); MajorTrainless choose/pass | 8-12 |
| T7b | 1867 `RlSpec`: families, layout, fingerprint pin, price specs from a 1867 price census, normalizers, choice vocabulary (`nationalize`) | 2-3 |
| T7c | Fuzz to zero violations at 2-6 players (2p is only valid once the variant lands); corpus completeness ≥ 99.5% on full-replay games; Ruby replay of ≥ 200 fuzz games with zero unexplained rejections | 4-6 (mostly fixing what they find) |
| T7d | Import: about **1,000 full-replay games**: 598 match + 108 mismatch/drift from the 2020-22 corpus, plus 317 from the 2026-10 scrape. 3-6 players, no grid_market. Convert to LMDB v3; per-family match report. | 2-3 |
| T7e | Train: supervised (1867-only and joint 1830+1867, §4.2 experiment) → value net from policy self-play → PG with league; evals at 4 players (230 of 367 recent games are 4-player) plus a 3/5/6 mix | 2-3 weeks wall-clock |

### Optional for the first run (decision 4)

| # | Item | Effort |
|---|---|---|
| T8a | 2-player variant: remove two random privates, `TRAINS_REMOVE_2_PLAYER`, 70% cap. Self-play needs only our own RNG. Replaying 2p records needs Ruby `Random.new(seed)` semantics. 2p is 23 of 367 recent games and 79+ rejections overall. | 2 (self-play only) + 2-3 (Ruby RNG for corpus) |
| T8b | `grid_market` optional rule: 2-D grid plus 1867 ceiling rule. 31 of 367 recent, 154 of 1,027 old games. | 3-5 |
| T8c | `CERT_LIMIT_CHANGE_ON_BANKRUPTCY` (TODO(1867-cert-limit)) | 1 |

**1867 total: T1-T7d is about 30-46 agent-days, plus T8 at 8-11 if taken, plus
training wall-clock.**

---

## 7. Plan, effort and parallelism

Every step can be tested on its own and is 1830-no-op unless stated.
1830-touching steps run the full ritual (`docs/verification_rituals.md`: frozen
layout and encoder tests, parity gates, pytest), plus a pre/post 1830
head-to-head smoke run.

| Step | What | Test | Effort | Depends on |
|---|---|---|---|---|
| **S1** | Python `titles.py`: `TitleSpec` (player range, optional rules, data paths, eval defaults); one `new_game(title, ...)` factory replacing every constructor site (§1c); title/layout/encoder metadata in game records, LMDB meta and checkpoints (written, not yet enforced) | pytest; a 1867 game constructs and steps via the factory | 2-3 | none |
| **S2** | Rust: `layout_for(&self.title)` in `factored_legal_indices`, `decode_index`, `price_head_slot_for_index` and MCTS; split `alphazero_bridge_ready`; `layout_info_py(title)` | Frozen tests plus parity gates; a 1867 call raises "not bridge-ready" instead of mis-indexing | 2 | none (touches `factored.rs:1798-1811`: coordinate with the stock-round agent) |
| **S3** | `MAX_SEATS = 8`: `mcts.rs` arrays, Python `VALUE_SIZE`, config cap; legacy 6-wide model padded | MCTS unit tests with 7-8 seat values; 1830 smoke run | 1-2 | S2 |
| **S4** | `rl/layout.rs` block registry plus descriptors, including a descriptor table for the frozen 1830 v1 layout; fingerprint test per title | Layout unit tests; descriptor(1830 index) agrees with the `ActionMapper` decode | 3-4 | S2 |
| **S5** | `rl/enumerate.rs` step-driven enumerator plus shared-step generators | **Differential: equals `factored_legal_indices` on 1830** random walks and corpus | 6-9 | S4 |
| **S6** | `rl/verify.rs` plus the Python CLI: fuzz invariants, decode round trip, corpus completeness, Ruby fuzz-game replay (generalize the `g1867_ruby_*` scripts) | Run on 1830 (must be clean) | 4-6 | S5 |
| **S7** | Generic token encoder (Rust) plus `model_generic.py` (descriptor policy head, pointer price head, relation bias, title FiLM) | Unit tests; **1830 validation against the legacy model** (§4.4, needs GPU) | 12-18 | S4 (encoder can start in parallel with S5) |
| **E0** | Sharing probe (§4.2.6): value, action-family and hex heads; 1867-only against 1830+1867, at four 1867 data sizes | Pre-set decision rule | 2-3 (+ about 1 GPU-day) | The S7 encoder half; 1867 replay (exists) |
| **S8** | Pipeline: LMDB v3, dataset/train, sparse inference server and `incorporate_results_sparse`, native hot-path getters, policy_selfplay/PG per title and N, evals per N, checkpointer enforcement, start-position builder per title | Pipeline smoke run on 1830 with the generic model | 8-12 | S3, S7 |
| **T1-T6** | 1867 rules gaps (§6) | Fuzz, corpus and Ruby replays | 14-22 | T1-T2 can start now; T3-T6 after S6 exists for checking |
| **T7a-d** | 1867 generators, `RlSpec`, verification, import | §2.8 all green | 16-24 | S5, S6 (T7a can start against S5's framework) |
| **T7e** | 1867 training: 1867-only baseline, plus the E1 joint arm (§4.2.6) | Evals; E1 decision rule | wall-clock | S8, T7 |

Totals:
- Generic layer S1-S8: about 38-56 agent-days.
- 1867 T1-T7d: about 30-46 agent-days, plus T8a-c at 8-11 if in scope.

Critical path: S2 → S4 → S5 → S6 → T7 → T7e. S7 and S8 run beside it.

With three agents:
- **A:** S1-S6, then T7a-d.
- **B:** T1-T6 rules gaps; they need only the existing engine plus Ruby probes.
- **C:** S7-S8 model and pipeline.

That gives a supervised 1867 model about 6-8 weeks out and a first PG model
about 10-12 weeks out.

### File ownership while other agents write new title engines

| Area | Files | Owner |
|---|---|---|
| New RL bridge (new files only) | `engine-rs/src/rl/{mod,layout,enumerate,encode,verify}.rs`, `engine-rs/src/rl/titles/g1867.rs` (later `g1822.rs`, ...), `rl18xx/agent/alphazero/{titles,model_generic,encoder_generic}.py`, `scripts/build_start_positions.py` (generalized), `scripts/ruby_diff.py` (generalized) | RL-bridge agents |
| Shared, small edits | `lib.rs` (registration), `mcts.rs`, `decode.rs`, `price_pmf.rs`, `factored.rs:1798-1811` (one line), `title/mod.rs` (`RlSpec` hook / bridge-ready split) | RL-bridge agents; rebase often; serialize with whoever owns `factored.rs` |
| Rules engine per title | `title/gXXXX.rs`, `rounds/*`, `steps.rs`, `game.rs`, `tiles.rs`, `map.rs`, `router.rs`, `revenue.rs` | Title-engine agents. **An 1822 engine agent is active now** (title files under `engine-rs/src/title/`, plus new steps and rounds). T1-T6 are rules work in these same files, so assign them to an 1867 engine owner and serialize their `steps.rs` / `game.rs` / `router.rs` edits with the 1822 agent. |
| 1830 frozen | `action_index.rs::build_layout`, `encoder.rs`, `factored.rs` (enumeration body), `action_mapper.py`, `encoder.py`, `model_transformer.py` | Untouched by the bridge (beyond §1a's one-line `layout_for` changes) |

**RL-readiness contract for each new title**, to be delivered by its engine agent:
1. Every StepKind and action type the title can offer has a generator, or is
   listed as "replay-only".
2. `RlSpec`: families, choice vocabulary, price specs, normalizers, extras.
3. Fuzz clean at every supported player count.
4. Corpus completeness report.
5. Ruby fuzz replay clean.
6. Layout fingerprint pinned.

This is also the definition of `rl_layout_ready()`.

The 1822 engine is being written now, so 1822's new step kinds should be
written with this contract in mind. The cheapest form is item 1: each step kind
records whether it is "replay-only" or has a generator.

1822 needs families that 1867 does not:
- bid-box bids on companies, minors and concessions;
- choices-round `Choose`;
- `AcquireCompany`;
- `choose_ability`.

Those get descriptors under §2.2. Generators can come later, because the
fuzzer refuses any title whose step kinds lack them.

### Risks

1. **1830 regression through shared files** (`mcts.rs`, `decode.rs`,
   `factored.rs` index call). Mitigation: frozen tests, parity gates, an 1830
   head-to-head smoke run per merge, and keeping 1830 production on the legacy
   model.
2. **Enumeration correctness with no second engine.** Mitigation: §2.8, and
   especially the Ruby replay of fuzz games. Budget for T7c finding real bugs.
3. **The generic model is weaker than the legacy one on 1830.** Mitigation:
   §4.4 gate; the legacy path remains.
4. **Shared-model interference.** Mitigation: per-title by default; sharing only
   after E0/E1 (§4.2.6); per-title RL fine-tunes either way.
5. **Self-play reaches states the corpus never shows** (lock-ins, odd mergers,
   bank breaks). Mitigation: fuzz from mid-game cuts, panics become errors,
   per-title move bounds.
6. **Ruby HEAD versus old records.** Mitigation: label data with our engine's
   result (equal to Ruby HEAD's); use the drift bucket as documented.
7. **Scale.** 16 titles of engine work dominate the total. The bridge is a
   one-time cost plus about 1-2 weeks of generators and spec per title once it
   exists. Families share the most: 1861 on 1867, the 1822 variants, 18USA on
   1817.

---

## 8. Decisions that need the user

1. **Model sharing** (§4.2-4.3).
   - Recommended: per-title models on one shared architecture, with family
     pooling (1822 + CA/MX/PNW, 1867 + 1861, 1817 + 18USA) as the first sharing
     step.
   - Cross-family shared pretraining only if E0/E1 pass their pre-set decision
     rule. RL is per title in every case.
   - Not recommended: committing now to one network for all titles.
   - Also decide whether E0 is worth its roughly 1 GPU-day plus 2-3 agent-days
     at week 3-4. If not, E1 alone settles the question at week 7-8.
2. **`MAX_SEATS`.** Recommended 8, which excludes seven 1817 games with 9-12
   players. The alternative is 12.
3. **1830 on the generic path.** Keep 1830 production on the legacy
   model/encoder until the generic model matches it (recommended), or plan to
   retire the legacy path once it does.
4. **First 1867 run scope.** Recommended: base rules, 3-6 players; skip the 2p
   variant and grid_market (T8) for now, about 14% of recent games.
5. **GPU time** for the 1830 validation of the generic model (§4.4), E0/E1, and
   1867 training. The GPU is shared with the user's jobs.
6. **1867 self-play starts.** Recommended about 80% human post-auction SR1, 20%
   full games with the auction, plus mid-game cuts.
7. **Second title.** 1861 is the cheapest next title: Ruby's 1861 extends 1867,
   and the corpus action vocabulary is identical. 1889 and 18Chesapeake are
   nearly 1830 (no new action types). The 1822 family is the expensive one but
   has four titles. Its rules engine is now being written by another agent, so
   the RL-readiness contract (§7) should be agreed with that agent early.

---

## Appendix A. Corpus census (aggregates only)

Source: `human_games/scrape_2026_10/<title>`, finished games only. Action
shares are of all non-client actions (`message`, `undo`, `redo`, `end_game`,
`log` and `program_*` excluded).

| Title | Games | Players: games | Optional rules (games) | Median actions | Non-1830 action types (share of actions) |
|---|---:|---|---|---:|---|
| 1830 | 1,796 | 2:300 3:265 4:900 5:216 6:115 | optional_6_train 303, multiple_brown_from_ipo 270 | 765 | none |
| 1867 | 367 | 2:23 3:56 4:230 5:47 6:11 | grid_market 31 | 687 | merge 1.4, convert 0.3, remove_token 0.1, choose <0.1 |
| 1861 | 649 | 2:32 3:59 4:461 5:78 6:19 | column_market 12 | 728 | the same as 1867 |
| 1822 | 743 | 2:39 3:180 4:364 5:111 6:28 7:21 | plus_expansion 578, ... | 1,153 | choose 4.5, acquire_company 1.9, merge 1.1, choose_ability 0.3, payoff_player_debt |
| 1822CA | 187 | 2..7 (7:4) | none | 1,345 | as 1822, plus assign |
| 1822MX | 363 | 2..5 | higher_cert_limit 51 | 1,044 | as 1822, plus remove_token |
| 1822PNW | 597 | 3..5 | starting_packet 34, ... | 1,134 | as 1822, plus assign |
| 1846 | 4,180 | 3:558 4:2725 5:897 | second_ed_co 136, first_ed 81 | 456 | assign 0.5 |
| 1889 | 2,031 | 2..6 | beginner_game 124 | 519 | none |
| 18Chesapeake | 531 | 2..6 | none | 523 | none |
| 1862 | 1,230 | 2..8 (7:7, 8:23) | long_length 259, eight_tokens 122, short_length 65 | 592 | choose 3.4, merge 1.1, remove_token 0.2, convert 0.1 |
| 18USA | 958 | 2..7 (7:3) | seventeen_trains 17 | 1,280 | take_loan 6.3, payoff_loan 4.4, choose 2.2, assign 1.7, short 0.9, merge 0.6, convert 0.6, scrap_train 0.2 |
| 1817 | 489 | 3..12 (8:1, 9:1, 12:5) | volatility 178, modern_trains 124, short_squeeze 65 | 1,485 | take_loan 7.0, payoff_loan 4.9, assign 1.3, short 1.3, merge 0.7, convert 0.7, choose 0.4 |
| 1849 | 676 | 3..5 | delay_ift 253, bonds 45, ... | 528 | choose 0.6, assign 0.2, take_loan, payoff_debt, payoff_loan, buy_token |
| 18 India | 708 | 2..5 | none | 598 | sell_train 3.4, corporate_buy_company 1.5, remove_border 0.9, corporate_buy/sell_shares, select_multiple_companies, sell_company, ... |
| 1871 | 917 | 3:543 4:374 | none | 678 | choose 2.0, offer 1.9, split 0.4, purchase_train 0.1 |
| 1828 | 72 | 3..5 | ensure_good_privates 26 | 1,057 | choose 1.7, merge 0.5, swap_train 0.3, special_buy 0.2, failed_merge |

Bids are 13-15% of all actions in the 1822 family (bid boxes), against 2.6% in
1830, so price-head quality matters much more there.
