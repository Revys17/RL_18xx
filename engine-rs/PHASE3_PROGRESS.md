# Phase 3 Progress: Action Processing + Round Logic

## Status: In Progress

**Validation across 3 test games: 231+60+113 = 404 matched actions, 0 Rust errors**

| Game | Matched | Mismatches | Errors | Total |
|------|---------|------------|--------|-------|
| manual_game | 231 | 30 | 0 | 693 |
| manual_game_bankrupcy | 60 | 28 | 0 | 88 |
| manual_game_discard_train | 113 | 13 | 0 | 126 |

All remaining mismatches trace back to the `can_run_route()` graph connectivity check returning false negatives for certain board positions. This causes incorrect auto-withhold share price movements during the operating round's skip_steps. The graph/map infrastructure fix is a Phase 4 dependency.

---

## What Was Implemented

### New Files
- **`src/actions.rs`** (~400 lines) — `Action` enum with 13 variants, `GameError`, `RouteData`, `DividendKind`. Parses actions from Python dicts including share_price as `"price,row,col"` strings, share IDs like `"PRR_2"`, and auto-detection of IPO vs market share source.
- **`src/rounds/mod.rs`** (~300 lines) — `Round` enum (Auction/Stock/Operating), `AuctionState` with waterfall auction model (bids per company, pending_par, active auction), `StockState` with consecutive-pass round ending, `OperatingState` with step progression.
- **`src/rounds/auction.rs`** (~300 lines) — Waterfall auction: placement bids (instant buy of cheapest), bid accumulation on non-cheapest, active auction (lowest bidder next), price reduction on all-pass, bid cascade resolution. Company abilities: CA grants free PRR share, BO triggers B&O pending par (free president share).
- **`src/rounds/stock.rs`** (~600 lines) — Par, BuyShares (IPO at par price, market at market price), SellShares (price moves DOWN per share in 1830), Pass. Auto-advance between players (handles implicit implicit passes passes). Sold-out price increase (move UP in 1830). Certificate limit enforcement. 60% ownership limit. President change detection.
- **`src/rounds/operating.rs`** (~650 lines) — LayTile (tile ID variant handling `57-0`→`57`), PlaceToken, RunRoutes (accepts routes from action), Dividend (payout=right, withhold=left), BuyTrain (train ID variant `2-0`→`2`, depot and inter-corp). Auto-advance through skippable steps. OR boundary detection and corp repositioning.

### Updated Files
- **`src/core.rs`** — Added `StockMarket` with full 1830 market grid, movement methods (left/right/up/down), par price lookup.
- **`src/entities.rs`** — Added `EntityId` helpers (is_player, player_id, corp_sym, ipo). Corporation share accounting (percent_owned_by, check_floated, president_id). Player cert counting (includes private companies) and value calculation (includes company face values).
- **`src/game.rs`** — `process_action()` PyO3 entry point, internal action dispatch, round transitions (Auction→Stock→OR→Stock cycles), phase advancement, train rusting, company closing, game end detection (bankruptcy=immediate, bank=full_or), final scoring. Auto-transitions between rounds when game data contains implicit passes. Market cell occupancy tracking for operating order sort. `legal_action_types()` for validation. Company revenue payouts at OR start.
- **`src/title/g1830.rs`** — Added company abilities (`grants_share`, `triggers_par`) to CompanyDef.
- **`src/lib.rs`** — Registered `actions` and `rounds` modules.

### Validation Infrastructure
- **`tests/validate_rust_engine.py`** — Replays human games through both Python and Rust engines. Resolves undo/redo in action sequences. Compares bank cash, player cash, corporation cash, float status at every action. Handles round-boundary timing differences.

### Key Fixes Applied During Validation
1. **Bank starting cash**: Deduct player starting cash from bank (12000 - 4×600 = 9600 for 4p).
2. **Company abilities**: CA grants free PRR 10% share. BO triggers CompanyPendingPar for B&O (free president share, no payment).
3. **IPO vs market pricing**: IPO shares sold at par price, market shares at current price. Auto-detect source from share ownership when not specified in action dict.
4. **Sell movement**: Fixed from `move_left` to `move_down` per share (1830's `SELL_MOVEMENT = "down_share"`).
5. **Sold-out increase**: Fixed from `move_right` to `move_up` (1830's `sold_out_stock_movement`).
6. **Stock round turn ending**: Changed from `player_passed` array to consecutive-pass counter. Players who acted then passed are NOT marked as permanently passed.
7. **Auto-advance between players**: When game data has implicit passes (from Python implicit passes), auto-advance to the target player at the start of each stock round action.
8. **Auto-advance through OR steps**: When an OR action targets a step beyond the current one, advance through intermediate steps (PlaceToken→RunRoutes→Dividend→BuyTrain).
9. **Auto-advance across corps**: When an OR action targets a different corp, advance through remaining corps' turns. Handles OR boundary crossing (different corp ordering).
10. **Operating order sort**: Matches Python's `sort_order_key`: `[-price, -column, row, cell_position, name]`. Tracks market cell occupancy for tiebreaks.
11. **Dividend revenue preservation**: Don't reset revenue when repositioning to a corp that's already mid-operation.
12. **Priority deal after auction**: Uses `entity_index` from auction end for `after_last_to_act` player reordering.
13. **Tile/train ID variants**: Strip `-N` suffix from IDs (e.g., `57-0`→`57`, `2-4`→`2`).
14. **President change from market**: When president share is sold to market, `check_president_change` now handles the market→player president swap with normal share return.
15. **Partial bundle sell**: When selling shares including the president cert, excess shares are returned from market to the seller (handles 50% sell from 60% ownership).
16. **MH exchange ability**: Company entity buy_shares actions bypass round dispatch, transfer a share to the company owner, and close the company.
17. **Company ability actions**: CS lay_tile and DH lay_tile/place_token handled via `try_process_company_ability()`, with terrain costs charged to the owning corporation.
18. **place_token city key parsing**: Actions with `city: "tile-instance-city_index"` format resolved to the correct hex at processing time.
19. **Tile catalog for city structure**: `tile_cities()` function provides slot counts for all city tiles, used when upgrading tiles to create proper city structures.
20. **Payout revenue distribution**: Fixed to match 1830's full capitalization rules — IPO shares generate no revenue on payout; only player-held shares pay out.

---

## Known Remaining Issues

### Fixed (this session)

1. ~~**buy_company "Corporation has no president"** (action #86)~~ **FIXED**
   - Root cause: `check_president_change` didn't handle president share sold to market.
   - Fix: added market-to-player president swap path.

2. ~~**MH (Mohawk & Hudson) exchange ability**~~ **FIXED**
   - Implemented as `try_process_company_exchange()` — MH's owner gets a NYC share, MH closes.

3. ~~**place_token parsing from city key**~~ **FIXED**
   - PlaceToken actions use `city: "57-0-0"` format instead of `hex` key.
   - Parser now handles city key: parses tile instance ID and resolves hex at processing time.

4. ~~**Tile catalog missing for lay_tile**~~ **FIXED**
   - New tiles placed on blank hexes had no city structure.
   - Added `tile_cities()` catalog in g1830.rs with slot counts for all city tiles.

5. ~~**Revenue payout to IPO shares**~~ **FIXED**
   - Rust was paying IPO share revenue to the corporation. In 1830 (full capitalization), payout sends 0 to the corp — only player-owned shares get revenue.

6. ~~**Company ability actions (CS, DH)**~~ **FIXED**
   - Added `try_process_company_ability()` for CS lay_tile and DH lay_tile/place_token.
   - Terrain costs charged to the owning corporation.

7. ~~**Partial bundle sell with president share**~~ **FIXED**
   - Selling 50% from 60% (including president) was transferring all certs to market instead of returning excess.
   - Added partial bundle handling: excess shares returned from market to seller after president change.

### Fixed (this session — session 2)

8. ~~**Stock→OR timing (company revenue payout)**~~ **FIXED**
   - Root cause: Rust's stock round didn't end at the same boundary as Python. Python auto-passes players who can't act, ending the stock round during the last stock action.
   - Fix: Added `stock_auto_pass_cascade()` with `can_player_act_in_stock()` check. After each stock action, cascade through remaining players.

9. ~~**SELL_AFTER="first" rule missing**~~ **FIXED**
   - In 1830, selling is blocked in the first stock round. Added `turn` counter that increments on last OR→Stock transition. Sell check: `self.turn > 1`.

10. ~~**President change on ties**~~ **FIXED**
    - `check_president_change` was using strict `>` which gave presidency to the first player in iteration order when shares were equal. Fixed to give tie-breaking priority to the current president.

11. ~~**OR auto-advance between corps**~~ **FIXED**
    - Added `or_advance_to_corp()` in `process_operating_action` to advance past corps whose turns are complete.

12. ~~**OR step auto-advance**~~ **FIXED**
    - Added `or_advance_to_step()` to advance past skipped steps when the action targets a later step (e.g., buy_train without run_routes).

13. ~~**Inter-corp train purchase detection**~~ **FIXED**
    - `or_process_buy_train` was buying from depot even when the price didn't match. Now uses price comparison to distinguish depot vs inter-corp purchases.

14. ~~**Auto-advance counting acted player's pass as consecutive**~~ **FIXED**
    - `stock_advance_to_player` was always incrementing `consecutive_passes`. Now resets to 0 when the auto-passed player had `acted_this_turn = true`.

15. ~~**Silent pass error handling**~~ **FIXED**
    - Pass actions that fail (because the player was auto-passed by the cascade) are now silently skipped, matching Python's error handling.

### Remaining Issues

1. **Graph connectivity false negatives** (all games)
   - `can_run_route()` returns false for some valid board positions, causing incorrect auto-withhold price movements. This is the sole remaining source of cash mismatches.
   - Root cause: the graph/map connectivity computation doesn't accurately trace tile connections. This is a Phase 4 dependency (route calculation).

2. **Emergency sell during forced train buy** (not tested yet)
   - When a corp must buy a train but can't afford it, the president sells shares. Partially implemented but no test games exercise this path.

### Phase 4 Dependencies

9. **Route calculation**
   - Currently accepts routes from the action dict. For legal move generation and MCTS, the Rust engine needs to calculate optimal routes independently. This is Phase 4.

10. **Tile upgrade validation**
    - LayTile currently accepts any tile on any hex. Proper validation (legal upgrades, rotation, connectivity, reachability) requires the graph/route system from Phase 4.

---

## Test Coverage

- **22 Rust unit tests**: construction, auction (buy, pass, price reduction, full auction→stock), stock round (par, float, sell price movement, sold restriction), company revenue payout, phase advance/train rusting, company closing, clone, market movements.
- **Validation script**: Replays human game (748 effective actions), comparing state at every step.
- **Build quality**: `cargo clippy -- -D warnings` clean, `cargo fmt` applied.

---

## File Statistics

| File | Lines (approx) | Purpose |
|------|----------------|---------|
| `src/actions.rs` | 400 | Action enum, parsing |
| `src/rounds/mod.rs` | 300 | Round/state types |
| `src/rounds/auction.rs` | 300 | Waterfall auction |
| `src/rounds/stock.rs` | 600 | Stock round |
| `src/rounds/operating.rs` | 650 | Operating round |
| `src/game.rs` (additions) | 500 | Transitions, auto-advance |
| `src/core.rs` (additions) | 150 | Stock market |
| `src/entities.rs` (additions) | 100 | Share helpers |
| `src/title/g1830.rs` (additions) | 30 | Company abilities |
| `tests/validate_rust_engine.py` | 120 | Validation script |
| **Total new/modified** | **~3,150** | |

---

## Next Steps (Priority Order)

1. Add sell_shares support during OR BuyTrain step (emergency sell) — first blocker at action #281.
2. Implement forced train buy / bankruptcy flow (president contributes cash).
3. Run validation on more human games (250 available in `human_games/1830/`).
4. Target: full game replay without any state mismatches.
