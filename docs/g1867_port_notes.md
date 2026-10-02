# 1867 Port Notes — Engine-Gap Analysis

Companion to the data draft at `engine-rs/src/title/g1867_draft.rs` (not compiled,
not registered). Ruby reference: `docs/ruby_reference/g_1867/` (vendored from
tobymao/18xx `lib/engine/game/g_1867/`). Line numbers below refer to those vendored
files and to the current `engine-rs/src/` tree.

The big picture: 1867 is **not** "1830 with different numbers". It adds four
structural things the engine has no model for at all — **minor corporations**,
a **merger/conversion round**, **loans + interest**, and a **non-operating
national corporation (CN)** — plus a long tail of smaller rule deltas. The
existing title seams (`GameTitle` trait, `StepDesc` lists, `RoundTransition`,
`Capitalization::Incremental`, `MarketMovement::OneDimensional`) cover maybe a
third of what 1867 needs; the rest is new engine surface.

---

## 1. Where the existing definition structs could NOT represent the Ruby data

These are the `TODO(...)` markers in `g1867_draft.rs`, with the struct that needs
extending.

| # | Gap | Ruby source | Engine struct to extend |
|---|-----|------------|------------------------|
| 1 | **No corporation `type`** (`major` / `minor` / `national`). Drives train limits, operating order, dividend rules, par rules, market movement, sold-out bumps, loan limits, merger eligibility. | entities.rb:109/199/388 (`type:` on every corp) | `title/mod.rs:277` `CorporationDef` and `entities.rs:253` `Corporation` (no `type` field) |
| 2 | **No MinorDef** — 16 minors are corporations with `shares: [100]`, `float_percent: 100`, `tokens: [0]`, `max_ownership_percent: 100`; treated as size-2 corps (game.rb:358 `CORPORATION_SIZES`, game.rb:368 comment) | entities.rb:189-380 | net-new struct (drafted as `MinorDef` in g1867_draft.rs) |
| 3 | **No home hex**: 1867 corporations choose their home city when parred/won (`HOME_TOKEN_TIMING = :par`); `CorporationDef.home_hex` is a mandatory `&str` | game.rb:306, game.rb:448-482 `home_token_locations`/`unconnected_hexes` | `title/mod.rs:283` `CorporationDef.home_hex` → needs `Option` + a home-choice step |
| 4 | **No `always_market_price`** — treasury (IPO) shares sell at *market* price, not par | entities.rb:108 etc. | `CorporationDef` / share-buy pricing in `rounds/stock.rs` |
| 5 | **No `max_ownership_percent`** (100 for minors; majors default 60, bumped to 70 in 2P — game.rb:643-649, 936-941) | entities.rb:198 etc. | `CorporationDef`, `entities.rs::Corporation` |
| 6 | **AbilityDef has no `HexBonus`** (`{type: 'hex_bonus', owner_type: 'corporation', hexes: [...], amount: 10}` on NFB/MB/QB/SCT — +$10 route revenue per listed hex) | entities.rb:34-96; consumed in game.rb:667-671 `revenue_for` | `title/mod.rs:416` `AbilityDef` + `abilities.rs` + `router.rs` revenue hook |
| 7 | **`BlocksHexes` requires an `owner_type`**; the hidden company `'3'` blocks M13 with *no* owner gate, until closed by the first 3-train | entities.rb:8-16; game.rb:960-961, 980-991 | `title/mod.rs:421` `AbilityDef::BlocksHexes` |
| 8 | **CompanyDef has no `discount`** (auction price floor: dutch auction lowers `min_bid = value - discount` by $5/round) | entities.rb:22/31/49/... `discount:`; step/single_item_auction.rb `enter_dutch` (`company.discount += 5`) | `title/mod.rs:295` `CompanyDef`; company entity needs a *mutable* discount |
| 9 | **TrainDef.distance is scalar u32**; every 1867 train is bucketed `[{nodes: [city offboard], pay: N, visit: N}, {nodes: [town], pay: 0, visit: 99}]` (towns free), and the 5+5E *inverts* the buckets (pays offboards only) | game.rb:200-304 | `title/mod.rs:306` `TrainDef.distance`; `router.rs` route legality + revenue |
| 10 | **TrainDef has no `multiplier`** (2+2 and 5+5E double revenue; also scales the capitals+Timmins bonus, game.rb:678) | game.rb:274/292 | `TrainDef` |
| 11 | **TrainDef.count can't say `'unlimited'`** (8, 2+2, 5+5E) | game.rb:257/276/293 | `TrainDef.count` (draft uses `u32::MAX`) |
| 12 | **Train `discount` exists but is unconditioned** — 1867's trade-in map is only usable in phase 8 (`discountable_trains_allowed?` = phase 8, enabled by `train_trade_allowed` event) | game.rb:261-268; step/buy_train.rb `discountable_trains_allowed?` | `TrainDef` + BuyTrain step logic |
| 13 | **PhaseDef.train_limit is a single u8**; Ruby is per-type `{minor: 2, major: 4}` etc. | game.rb:144-198 | `title/mod.rs:326` `PhaseDef` (needs per-corp-type limits) |
| 14 | **Phase `tiles` includes `gray`** (phases 7-8) — fine as data, but the track step must honor a 4th color | game.rb:186/194 | tile-color gating in `rounds/operating.rs` |
| 15 | **MarketZone enum can't represent 1867 cell types**: `x`=par_1 (minor par), `z`=par_2 (major par), `p`=par (both), `C`=convert_range, `m`=max_price — and they are *combinable flags* (`100pC`, `165zCm`, `135pmC`) | game.rb:37-66 COLUMN_MARKET, 68-142 GRID_MARKET, 352-357 MARKET_TEXT | `title/mod.rs:344` `MarketZone` → needs a flag set on `MarketCell`; par-price queries by corp type (`share_prices_with_types`) in `core.rs::StockMarket` |
| 16 | **HexDef has no border model**: impassable edges (D18, C9, D10, E11, C11, D12, C13, H14, blue-lake edges) and per-edge water costs of $80 (N8, N10, M11, O9, M9, L10). 1830's impassable borders are *hardcoded per hex* in `factored.rs:1136`. | map.rb:165-188 | `title/mod.rs:353` `HexDef` + adjacency in `title/mod.rs::compute_adjacency` / `factored.rs` / `graph.rs` |
| 17 | **No HexType for**: triple city (L12 Montreal, 3 separate 1-slot cities), blue lake ports (E19/H16, flat revenue 10), 4-tier offboard revenue (`yellow_20\|green_30\|brown_40\|gray_40`), grouped offboards (A17+A19 `groups:Detroit, hide:1` count as one — since 2026-10-02 the DSL's `groups:` is parsed onto `Offboard.groups` and the router stops at one per group) | map.rb:198-217 | `title/mod.rs:359` `HexType`, `router.rs` phase-revenue lookup |
| 18 | **No stub support** (`stub=edge:N` on G15/L10/K13; new tiles must connect to stubs — `StubsAreRestricted`, game.rb:366) | map.rb:182/185/188 | tile-lay legality in `rounds/operating.rs` / `factored.rs` |
| 19 | **No `future_label`** (J12 Ottawa: `label=Y` now, gray upgrades to `label:O` tile X8) and no label-aware upgrade paths for Y/M/T/O | map.rb:184/199/201 | `tiles.rs` catalog + upgrade legality |
| 20 | **Tile catalog has no 1867 tiles** — `tile_catalog_1830()` only; X1-X8 are custom DSL tiles, plus ~25 standard ids 1830 never uses (17, 21, 22, 30, 31, 87, 88, 120, 122, 124, 201, 202, 204, 207, 208, 621-626, 637, 639, 801, 911) | map.rb:7-126 | `tiles.rs` (add a 1867 catalog; `GameTitle::tile_catalog` seam already exists at `title/mod.rs:50`) |
| 21 | **Cert limit is not a pure fn of player count**: `CERT_LIMIT_CHANGE_ON_BANKRUPTCY = true` recomputes on bankruptcy | game.rb:27, 321 | `GameTitle::cert_limit` (`title/mod.rs:37`) + game state |
| 22 | **TILE_LAYS: two lays per OR** (second must not be an upgrade if first was, costs $20, can't reuse the hex) — engine assumes 1830's single lay | game.rb:330-333 | `rounds/operating.rs` track step; needs a per-title lay descriptor |
| 23 | **`TILE_UPGRADES_MUST_USE_MAX_EXITS = [cities]`** — city upgrades must keep max exits | game.rb:35 | upgrade legality |
| 24 | **2-player variant**: remove trains per `TRAINS_REMOVE_2_PLAYER`, 70% ownership cap, remove 2 random privates | game.rb:362, 932-954 | game setup; (probably out of scope for RL, but it's data the structs can't hold) |

---

## 2. Mechanics inventory

For each mechanic: the Ruby implementation, the engine seam it plugs into, and
whether it is configuration, extension, or net-new.

### 2.1 Minors (operating, dividends, train limit) — NET-NEW entity class

- Ruby: entities.rb:189-380 (16 corps `type: 'minor'`); game.rb:401-407
  (`minor?`/`major?`), 532-535 (`operating_order`: **floated minors operate
  before majors**), 444-446 (loans max 2), step/dividend.rb (minors *skip the
  dividend choice*: auto payout if revenue > 0; payout = `revenue/2` to treasury
  **and** `revenue/2` to the owner; majors get a third option `half` via
  `Engine::Step::HalfPay` — `DIVIDEND_TYPES = [payout, half, withhold]`).
- Dividend price movement (step/dividend.rb:246-256): minors use base movement;
  majors move **left 1 on zero revenue**, **right 1 only if revenue >= share
  price**, otherwise **no movement**. The engine's 1830 dividend movement is
  hardcoded in `rounds/operating.rs:869` (right on payout) / `:882` (left on
  withhold) — needs a per-title dividend policy.
- Train limit is per type and per phase (game.rb:144-198): `{minor: 2}` →
  `{minor: 1, major: 3}` → ... Engine has a single `PhaseDef.train_limit`.
- Minor founding is by **bid** (see 2.6), not par; minors cannot start after the
  first 5-train (`minors_cannot_start`, game.rb:1001-1013) and are all
  nationalized on the first 8 (`minors_nationalized`, game.rb:1015-1023).
- Green minors (BBG LPS QLS SLA TGB THB) only become available when the first
  3-train fires `green_minors_available` (game.rb:361, 974-991 —
  `@future_corporations` split in `setup`).
- Seam: `entities.rs::Corporation` gets a `kind` enum; `operating_order` /
  train-limit / dividend code branches on it. Genuinely new: the half-treasury
  dividend, the per-type phase limit, the availability windows.

### 2.2 Minor → major conversion and minor mergers — NET-NEW round + steps

- Ruby: step/merge.rb (the whole file), round/merger.rb.
- **Convert** (`process_convert`/`finish_convert`): a minor whose share price is
  in the `convert_range` (`C` cells, $100-165) converts into an unfloated major.
  New par = greatest `par/par_2` price ≤ the minor's price; owner receives the
  20% president cert free; tokens/cash/companies/loans/trains transfer; minor
  closes.
- **Merge** (`process_merge`/`finish_merge_to_major`): 2-10 *connected* floated
  minors (token-graph connectivity, `connected_cities`) merge into an unfloated
  major; one owner may contribute at most 6. New par = `min(200, max(100,
  min_price + max_price))` snapped down to a `par/par_2` cell; each contributed
  minor pays out one 10% share (president handling special-cased); duplicate
  tokens removed (`Engine::Step::TokenMerger`), then **ReduceTokens** (step E,
  step/reduce_tokens.rb: max 2 surviving tokens, all in different hexes, $40
  third token re-added via game.rb:794-798 `fix_token_count!`), then
  **PostMergerShares** (steps C/D, step/post_merger_shares.rb: involved players,
  in order, may buy treasury shares — multiple for involved players up to 60%,
  then one each for everyone; merge *fails* (undo) if nobody can hold 20%).
- Round flow: a **Merger Round runs after every OR in phases 3-7**
  (game.rb:876-897). `G1867::Round::Merger` iterates floated minors
  (`merge_corporations`, game.rb:417-419).
- Seams touched: `steps.rs:184-212` `FinishedRound`/`RoundStart` need a `Merger`
  variant; `game.rs:814-829` `start_round` needs a new arm; `StepKind`
  (steps.rs:55-74) needs `Merge`, `PostMergerShares`, `ReduceTokens`,
  `MajorTrainless`; new action types (`convert`, `merge`, `choose`) must be
  added to `actions.rs` **and to the AlphaZero action layout**
  (`action_index.rs` / `action_mapper.py` — POLICY_SIZE changes!). Genuinely
  net-new: everything; nothing similar exists.

### 2.3 Loans and interest — NET-NEW financial model

- Ruby: `include InterestOnLoans` (game.rb:364) + game.rb:376-378 (rate $5),
  421-446 (interest computed **at OR start**, on loans held *before* this OR's
  takes — `calculate_interest` is called in `operating_round`, game.rb:840),
  499-530 (`take_loan` nets $45 of a $50 loan; `repay_loan` costs $50;
  `buying_power(full:)` = cash + remaining loan capacity × $45), 899-907
  (72 loans of $50; majors 5 max, minors 2).
- **AutomaticLoan** (step/track.rb, step/buy_train.rb include it): loans are
  taken implicitly when spending beyond cash during track/token/train steps.
- **LoanOperations step** (step/loan_operations.rb): runs with *no player
  actions* after dividends — pays interest (nationalize the corp if it can't
  pay!), then auto-repays loans while cash ≥ $50, then re-snapshots interest.
- **BuyCompanyPreloan** (step/buy_company_preloan.rb): a blocking BuyCompany
  window *before* LoanOperations, auto-passed when the corp has no loans (so a
  corp can dodge repayment by buying a private first).
- End-game: each outstanding loan moves the share price left one step
  (game.rb:765-785 `end_game!`; mirrored in `player_value`, game.rb:745-763).
- Seams: `entities.rs::Corporation` needs a `loans: Vec<Loan>` (net-new Loan
  model — Ruby `lib/engine/loan.rb` is id+amount); new StepKinds; BuyTrain's
  `must_buy_train` definition changes (see 2.10). Genuinely net-new.

### 2.4 Share redemption — new step, small

- Ruby: step/redeem_shares.rb (subclass of `Engine::Step::IssueShares` with
  only the redeem half); game.rb:484-489 `redeemable_shares` = pool shares the
  corp can afford. Runs early in the OR (before Track).
- Note: 1867 has redeem only, **no issue** (the step exposes `buy_shares` only).
- Seam: new `StepKind::RedeemShares` + `steps.rs::step_actions` arm + a
  corp-buys-from-pool transfer in `rounds/stock.rs`-style share plumbing.
  Moderate, mostly new but localized.

### 2.5 Opening auction: SingleItemAuction — NET-NEW auction format

- Ruby: step/single_item_auction.rb; game.rb:806-810. Companies are auctioned
  **one at a time, cheapest first** (sorted by value, the hidden `'3'`
  excluded). Normal ascending auction (min increment $5,
  `MUST_BID_INCREMENT_MULTIPLE = true`); if *everyone* declines, it flips to a
  **dutch auction**: price drops $5 (`company.discount += 5`) per full pass
  until someone buys (a $0 price force-buys for the first bidder). Winner's
  *right-hand neighbor* opens bidding on the next company
  (`@round.goto_entity!` + `next_entity_index!`). All-passed leftovers are
  removed from the game.
- Seam: `StepKind` + the auction round already exists as a kind
  (`rounds/auction.rs` is waterfall-only). The dutch fallback and the mutable
  company discount are net-new; the PassableAuction skeleton has no Rust
  equivalent.

### 2.6 Stock round: minors by bid, majors by par-from-phase-4 — extension of BuySellParShares

- Ruby: step/buy_sell_par_shares.rb (`< BuySellParSharesViaBid`): starting a
  **minor** is a live auction inside the stock round (MIN_BID $100, only your
  buy action for the turn); `win_bid` pars the minor at
  `min(bid/2, 135)` snapped down to a `par_1/par` cell, and the **entire bid**
  goes into the minor's treasury (the par×2 dance in the code is bookkeeping).
  Turn order resumes from the winner. Starting a **major** is a normal par
  (`par_2/par` cells) but only from phase 4 (`majors_can_ipo`), and only if a
  home token location exists.
- Home token: chosen, placed via `G1867::Step::HomeToken` in the **stock
  round** (game.rb:813-818; step/home_token.rb).
- Stock round config deltas vs 1830 (game.rb:306-316): `SELL_AFTER = :operate`,
  `SELL_BUY_ORDER = :sell_buy`, `MUST_SELL_IN_BLOCKS = false`,
  `POOL_SHARE_DROP = :none` (pool sales don't drop price),
  `SELL_MOVEMENT = :down_block_pres` (see 2.8), sold-out bump **majors only**
  (round/stock.rb `sold_out?`).
- Seam: `rounds/stock.rs` BuySellParShares is the closest existing machinery;
  the in-SR auction sub-state (`@auctioning`, bids, winner resume) is net-new;
  `RoundStart::Stock` itself is fine.

### 2.7 Incremental capitalization — existing seam, unimplemented

- Ruby: game.rb:31 `CAPITALIZATION = :incremental`; `ipo_name` = 'Treasury'
  (game.rb:372-374). Share buys from the treasury pay the **corporation** (at
  market price — `always_market_price: true`), not the bank; float (20%) brings
  no lump sum.
- Engine seam: `title/mod.rs:265-275` `Capitalization::Incremental` exists;
  `rounds/stock.rs:777-799 check_float` has the explicit
  `unimplemented!("incremental capitalization lands with 1867")` arm. The buy-
  side payment routing (buyer → corp treasury, at market price) and float-
  without-cash are the work.

### 2.8 1867 sell movement — engine hardcodes 1830's

- Ruby: `SELL_MOVEMENT = :down_block_pres` (game.rb:310): selling moves the
  price down **one step per block sold** (not per share), and **president-held
  shares can't be dumped through the pool below presidency**; plus
  `POOL_SHARE_DROP = :none`.
- Engine: `rounds/stock.rs:531-541` hardcodes 1830's `down_share` (one
  `move_down` per share unit) — needs a per-title sell-movement policy. The 1-D
  market movement itself is already done: `core.rs:96-240` `StockMarket` +
  `MarketMovement::OneDimensional` (`title/mod.rs:86-94`) implement
  left/right-only movement. Still missing from `core.rs`: the G1867 subclass
  quirks (stock_market.rb): minor on a `max_price` cell doesn't move right;
  2-D grid-market ceiling rule for majors (up → right+down-one).

### 2.9 CN national formation & nationalization — NET-NEW

- Ruby: entities.rb:381-390 (CN: 8 free tokens, never operates); game.rb
  setup:956-978 (CN pulled out of `@corporations`, shares cleared), 913-930
  `add_neutral_tokens` (neutral green tokens on D2 + L12 third city, real CN
  token on F16 Toronto; greens removed at phase 3), 360 `NATIONAL_RESERVATIONS`
  (L12 slot reserved; claimed when gray tile 639 is laid — game.rb:651-660
  `place_639_token` hooked from step/track.rb).
- `nationalize!` (game.rb:568-641): repay loans while cash ≥ $50 → move price
  left once + **twice per remaining loan** (game.rb:557-562) → treasury to bank
  → bank pays every shareholder price×shares → replace its tokens with CN
  tokens (highest city revenue first, skip hexes CN already tokens, respect the
  L12 reservation) → minors close, majors **reset** (back to startable; 2P 70%
  fix game.rb:643-649).
- Triggers: `trainless_nationalization` event (first 4/6/8 trains) → flag →
  after the train buy/export (`post_train_buy`, game.rb:741-743, 1030-1045):
  trainless operated minors nationalize immediately, trainless operated majors
  queue for the **MajorTrainless choose step** (step/major_trainless.rb — first
  step of SR, MR *and* OR); unpaid loan interest (2.3); passing BuyTrain with
  no train (step/buy_train.rb `pass!`); `minors_nationalized` on the first 8.
- Seam: nothing exists. Needs a special entity (not in the operating order, can
  hold tokens/companies), token-type `neutral` (passable by all), reservation
  machinery, and a `choose` action.

### 2.10 Train export + 1867 BuyTrain/EMR

- Export: phases 4-7 carry `export_train` status; at the end of **each OR** the
  next depot train is exported to the CN, *triggering phase change as if
  purchased* (game.rb:323-327, 858-864 `or_round_finished`; Ruby
  `depot.export!`). Engine has no depot-export and no phase-change-on-export;
  plugs in where the OR-set transition is computed (`game.rs` round transition +
  phase code).
- BuyTrain (step/buy_train.rb): `MUST_BUY_TRAIN = :always`, but "must" only if
  affordable **with max loans** (`buying_power(full:)`); EMR is loans-only —
  **no share selling** (`can_sell?` false), **no president contribution**
  (`president_may_contribute?` false); trade-in discounts only in phase 8;
  passing with no train ⇒ nationalization. Engine seam: BuyTrain step logic in
  `rounds/operating.rs` + the new loan model; the engine's 1830 EMR
  (president-sells/contributes, `operating.rs:1798` forced sales) must be
  disabled per title.
- Bankruptcy: `can_go_bankrupt?` = emergency liquidity < 0 (game.rb:683-689);
  cert limit recomputes (`CERT_LIMIT_CHANGE_ON_BANKRUPTCY`).

### 2.11 Routes & revenue — router extensions

- Bucketed distances + free towns + best-combination stop selection with the
  "at least one counted stop tokened" constraint: game.rb:706-739
  `compute_stops` (replaces base auto behavior; the 5+5E pays offboards only).
  Engine seam: `router.rs` (1830 counts all stops against a scalar distance).
- Revenue add-ons (game.rb:662-681 `revenue_for`): `hex_bonus` company
  abilities (+$10/route per listed hex); **capitals bonus**: route touching
  Timmins (D2) *and* one of Toronto/Montreal/Quebec (F16/L12/O7) gets +$40 ×
  train multiplier; explicit "route visits same hex twice" error; multiplier
  trains double everything. Blue ports pay flat 10; 4-tier offboard revenue
  needs green/gray phase tiers.
- Token step (step/token.rb): token cost = base price × **hex distance from the
  corp's nearest existing token** (`adjust_token_price_ability!`) — completely
  different from 1830's fixed price list; affordability check ranges over
  tokenable cities. Engine seam: token pricing in `rounds/operating.rs`.

### 2.12 Companies in 1867

- `CompanyPriceUpToFace` (game.rb:365): corporations may buy privates for
  $1..face (the engine's 1830 BuyCompany allows 0.5×-2×; per-title price band
  needed).
- `nationalize_companies` event (first 6-train): privates close, owner paid
  face value by the bank (game.rb:1047-1062) — different from 1830's
  uncompensated `close_companies`.
- The hidden `'3'` company (phase blocker for M13) is never auctioned/owned.
- 2P setup removes two random privates (game.rb:950-953) — randomness in setup
  is also new for the engine.

### 2.13 Misc

- `interest`/round bookkeeping (`clear_interest_paid` at every round
  transition, game.rb:877).
- Final OR set after game-end trigger is **3 ORs** (game.rb:787-790), not the
  phase's 2 — `RoundTransition`/`total_ors` already carries a count, so this
  plugs in, but the trigger plumbing (`one_more_full_or_set` priority over
  `current_or`, game.rb:315-316) is new.
- `reorder_players` happens after the auction **and** after each SR
  (game.rb:879-895) — same as 1830, no gap.
- Encoder/action-space: every new action (bid-in-SR, convert, merge, choose,
  redeem, home-token-choice, loan implicits) and every new entity (16 minors,
  CN) changes `action_index.rs`/`encoder.rs`/`action_mapper.py`. The roadmap's
  per-title action-layout work is a hard prerequisite for *training*, though
  not for a playable rules engine.

---

## 3. Top 10 engine gaps, ordered by how much they block a minimal playable 1867

1. **Corporation `type` (minor/major/national) + MinorDef** — touches operating
   order, train limits, dividends, par rules, market movement, loans, merger
   eligibility; nothing else can be built without it.
   (`entities.rs:253`, `title/mod.rs:277`.)
2. **Merger/Conversion round** — new `RoundStart`/`FinishedRound` variant, 4-5
   new StepKinds, new actions (`convert`/`merge`/`choose`); runs after *every*
   OR in phases 3-7, so the game literally cannot proceed past phase 3 without
   it. (`steps.rs:55-74,184-212`, `game.rs:814`.)
3. **Loans + interest** — net-new `Loan` model on `Corporation`, AutomaticLoan
   spending, LoanOperations auto-step, interest snapshots, loan-aware
   buying power; 1867's EMR *is* the loan system. (game.rb:364-530, 899-907.)
4. **CN national + `nationalize!`** — net-new non-operating entity, neutral
   tokens, reservations, shareholder payout, token replacement, major reset;
   it is the failure path for trains, interest, and the 8-train events.
   (game.rb:568-660, 913-930.)
5. **Incremental capitalization + always_market_price + bid-founded minors** —
   `rounds/stock.rs:788` is literally `unimplemented!`; treasury-held shares,
   market-price buys, and the SR bid auction are the heart of the 1867 economy.
6. **Stock-market cell types + sell movement** — par_1/par_2/par/convert_range/
   max_price as combinable flags (`title/mod.rs:344` can't), par-price queries
   by corp type, `down_block_pres` selling (engine hardcodes per-share
   `move_down`, `rounds/stock.rs:531-541`), majors-only sold-out bump. The 1-D
   movement itself already exists (`core.rs`, `MarketMovement::OneDimensional`).
7. **SingleItemAuction** — the game can't even *start* without the opening
   auction format (dutch fallback, mutable company discount, winner-neighbor
   ordering). Localized but mandatory.
8. **Map/graph features** — impassable borders + per-edge water costs as data
   (today hardcoded for 1830 in `factored.rs:1136`), stubs, triple-city
   Montreal, blue ports, 4-tier offboards, grouped Detroit, future_label, the
   1867 tile catalog incl. X1-X8, two tile lays ($20 second), home-token
   *choice*. Without these the map is unbuildable, though many items are
   individually small.
9. **Router/dividend deltas** — bucketed distances with free towns, 5+5E
   offboard-only + token-in-counted-stops, multiplier trains, hex_bonus +
   capitals/Timmins bonuses; 1867 dividend triple-choice (half pay) with
   revenue-vs-price movement and minor auto-half-pay. A game is "playable but
   wrong everywhere" without them.
10. **Train export + event machinery** — per-OR depot export with
    phase-change-as-if-purchased, plus the seven 1867 events
    (green_minors_available, majors_can_ipo, minors_cannot_start,
    nationalize_companies, minors_nationalized, trainless_nationalization →
    MajorTrainless choose step, train_trade_allowed). The engine's event
    surface today is essentially 1830's single `close_companies`.

Honorable mentions that don't block "playable" but block "correct": token cost
× hex distance (2.11), CompanyPriceUpToFace band, redeem-shares step,
BuyCompanyPreloan ordering trick, cert-limit-on-bankruptcy, final-3-OR set,
end-game loan price adjustment, 2-player variant.
