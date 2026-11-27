# The site on engine-rs, step 1: 1830 in the browser and in the server's shadow

Status: **design, 2026-11-25** (branch `site/engine-rs-1830`, RL_18xx worktree `.claude/worktrees/site-engine`).
Approved by Caleb: his site (18xx.revys.net, `rl18xx-next`) drops the Ruby/Opal engine for engine-rs, keeping only the
titles engine-rs implements. **Step 1** (this doc): 1830 runs on engine-rs (WebAssembly) in the browser behind a switch,
with Opal as the fallback, and engine-rs runs in **shadow** beside the server's Ruby engine on every 1830 action, on the
**dev site only** (:9301 and scratch stacks). Ruby stays authoritative. Production (:9292) is not touched.

Read with: `docs/WASM_ENGINE_SPIKE.md` (the spike's plan and API, branch `spike/wasm-engine`) and
`docs/WASM_ENGINE_SPIKE_RESULTS.md` (its evaluation: X1-X3, M4-M10). Section numbers here are referenced from code
(`engine-rs/src/neutrality.rs` cites §8.4).

---

## 0. In one page

| Question | Decision |
|---|---|
| The spike | **Brought forward, not rewritten.** Merged onto master a615a40 it has 2 trivial conflicts and 85 compile errors, all of them pyo3 gating in files master added since (rl/*, g1822/*, mcts_lockstep); with those gated it reproduces the spike's numbers exactly (§2.1). Redone: the engine edits outside `src/site` (small separate commits, last, after X2/X3), the records hooks (behind `site_rules`), the client's words/rawmenu split (master's `moves.ts` moved on) and the goldens and title pack (from today's Opal). New: the shadow key, the `site_engine` CLI, `check`, `programInfo`, `version`. |
| WASM API | The spike's `WasmGame` unchanged where it exists; API 2 adds `version`, `check`, `programInfo`, `shadowKey` (§3). The served glue is `--target no-modules`, loaded with `importScripts` like the Opal bundles, so the site always serves a matching glue + wasm pair. RawMenu: the spike's facts (§3.3). |
| site_rules | A runtime flag on `BaseGame`, never a cargo feature, never set by Python (no setter is exported). Five branches, each naming its Ruby method (§4.2), plus four record hooks also behind it (§4.3). All edits outside `src/site` are listed function by function (§4); they come **last**, after rebasing onto the training session's X2/X3 commits. |
| Server integration | **A native Rust CLI, `site_engine`, run by the queue process** (a sidecar process per check, JSON on stdin/stdout), built in Docker (a `rust:bookworm` stage) for the dev stack and in `Dockerfile.next-prod`. Not magnus (in-process: an abort takes a unicorn worker and its move with it; libclang + Rust in both images) and not wasmtime (a second, non-wasm-bindgen ABI plus a native gem, for bytes the node parity suite already checks). §5.1 |
| Shadow | In the request, after Ruby processed the move: Ruby's **state key** (§5.3) is captured (rescued, ~1 ms) and a row is queued. The queue's worker runs `site_engine state` on the game up to that action and compares Rust's key with Ruby's: acceptance, acting players, round/turn, step/entity/actions, finished/result, the state key, the stored run's revenues. Disagreements: table `engine_shadow`, the queue log, the admin page and `GET /api/admin/engine_shadow`. Never blocks or fails a move. |
| Run revenue | The server stops trusting posted revenue **for every title**: `record_action` strips each route's `revenue`, `revenue_str`, `subsidy` and the action's `extra_revenue`, `subsidy`, recomputes the last two with the engine's own `extra_revenue` / `routes_subsidy`, and stores what Ruby computes. A posted revenue that differs is logged (`revenue_mismatch`) and the response says `repriced: true` (the page resyncs). §6 |
| Auto actions | **Ruby-only in step 1.** A Rust page posts with `X-Engine: rust`; the server adds the auto actions (`add_auto_actions: true`, as for bots) and returns the stored action, which the page applies. Rust replays recorded auto actions and the shadow checks the state after them. The Programmer in Rust is step 2 work. §7.1 |
| Client switch | Site flag (`RUST_ENGINE_TITLES=1830`, `RUST_ENGINE_DEFAULT` = `opal` or `rust`) in the boot block; a per-browser setting (Automatic / Site's engine / Rust); `?engine=` for one page. Any Rust failure re-opens the game on Opal with a note and reports it to the server (same table). Removing 1830 from the flag is the kill switch. §7.2 |
| Verification | Corpus: every 1830 game in the :9301 DB, production 1830 games (read-only bot-API export), Ruby's and RL_18xx's site-format 1830 fixtures. Per position: Rust vs Opal goldens (snapshot, action, tracks, log, every seated player's menu, tile options, routes, details) and Rust vs Ruby (state key). Every offered option acted in Rust and judged by Ruby, both directions. The training-neutrality test (pinned before any change: commit 359463c). §8 |
| Packages | **A** engine (RL_18xx `site/engine-rs-1830`), **B** server (rl18xx-next `next/rust-shadow`), **C** client (living-board `rust-1830`), disjoint files, contracts in §9.2; A publishes a skeleton wasm package and `site_engine` on day 1. |

---

## 1. Step 1 at a glance

```
 browser (living board)                          server (rl18xx-next, Roda)                   queue process
 ───────────────────────                         ──────────────────────────                    ─────────────
 Session ── worker.js (Opal)  ◄── fallback ──┐   POST /api/game/<id>/action
        └─ worker-rs.js ── importScripts      │     record_action: Ruby (authoritative)
            engine_rs.js (no-modules glue)    │       - revenue recomputed (all titles)  §6
            engine_rs_bg.wasm (1830)          │       - X-Engine: rust → add_auto_actions  §7.1
            title-1830.json (pack, client)    │       - EngineShadow.capture(engine) → row  §5.2 ───►  ShadowWorker.tick
        menus, tile options, exact router,    │     response: game + stored action                    site_engine state
        log, details, snapshots               │                                                         compare keys
        fallback report ──────────────────────┴──► POST /api/game/<id>/engine_report                   engine_shadow table
```

What does **not** change in step 1: Ruby decides every move on the server; `POST /api/bot/engine` and the bot API keep
their contracts (bots' runs are priced by the server like everyone's, §6); other titles run on Opal; hand-drawn runs and games with `auto_routing: false` run on Opal; local
(hotseat) games may run on Rust; production is not deployed.

What step 2 needs (not here): the Programmer (auto actions) in Rust, hand-drawn runs (M11), Rust authoritative on the
server for 1830 (`site_engine serve` or in-process), then 1867 and the 1822 family, then retiring Opal and Ruby.

---

## 2. The spike: what comes forward, what is redone

### 2.1 Evidence: the spike on today's master

`git merge-tree master spike/wasm-engine` (master a615a40, 27 engine-rs commits since the spike's base b59b625): two
conflicts, both trivial (`action_index.rs` a doc comment beside a `cfg`; `game.rs` the clone, where master split
`clone_for_search_log(log)` and the spike `clone_inner(map)`). Extracted into a scratch copy, `cargo check
--no-default-features --features site` gave 85 errors, **all pyo3 gating in files master added**: `rl/py.rs`,
`rl/encode/py.rs`, `rl/generator.rs` (its `#[pymethods]` block), `g1822/py.rs`, `g1867/py.rs` (module declarations),
`mcts_lockstep.rs` (new `#[pyfunction]`s). Gated, the site layer compiled unchanged, and the spike's harness on the
spike's corpus and goldens gave **exactly the spike's numbers**:

| `site_replay` on the 939 cached :9301 games, goldens of living-board b8a07d0 | Merged onto master a615a40 |
|---|---|
| Strict replay | 939 / 939 (914 + 25 with programs) |
| Finished games' result vs the site's | 135 / 135 (47 bank, 53 bankruptcy, 35 manual end) |
| Snapshots / step actions / tracks | 78,809 / 78,809; 78,506 / 78,506; 8,966 / 8,966 |
| Menus | 78,572 / 78,745 (99.78%: 142 pass labels, 16 M&H exchange (X2), 12 stock-turn history (M4), 2 bankruptcy (X3), 1+1 singletons) |
| Tile options / auto routes | 9,773 / 9,773; 4,584 equal revenue, never below |
| Log lines | 163,717 / 166,119 (98.55%) |

So the engine changes since Nov 9 (1822/1867 generators, the RL bridge, `tokenable_cities` order, encoder v2/v3) did
not move 1830's site behaviour. Bringing the spike forward is a port, not a rewrite.

### 2.2 Forward as is (rebased, restructured into reviewable commits)

- **pyo3 optional** (`pyo3-noop/`, `src/pyerr.rs`, `cfg(feature = "python")`), extended to master's new files; one
  commit with the pre-push hook change (§4.1, §8.6).
- **`src/site/**`**: `json.rs` (site JSON ↔ `Action`, G19 and tile-id translation, Ruby's tile supply order),
  `game.rs` (`SiteGame`: Ruby's `filtered_actions`, site player ids, recorded auto actions, `message` / `program_*` /
  `end_game`, turn starts, Ruby's player/share-holder/company orders), `view.rs`, `track.rs`, `menu.rs`,
  `details.rs`, `log.rs`, `records.rs`, `compare.rs`, `wasm.rs`; `src/bin/site_replay.rs`; `site/build-wasm.sh`.
- The `only-1830` registry feature (the trimmed wasm).

### 2.3 Redone or new

| What | Why / how |
|---|---|
| Engine edits outside `src/site` | Re-done as small separate commits, each `if self.site_rules` naming its Ruby method, **after** rebasing onto X2/X3 (§4). The spike mixed them into its feature commits. |
| Records hooks (operating history, player values per round) | Behind `if self.site_rules` (the spike called them unconditionally and returned early on `site.is_none()`). Storage stays in `src/site/records.rs`, compiled in every build. |
| `clone_without_map` | Not added to `game.rs`: `src/site/log.rs` takes master's `clone_for_search_without_log()` and clears `hexes` / `tile_counts_remaining` itself. |
| Menu's exchange and bankruptcy offers | Must call the same predicates the fixed engine uses after X2 (`steps::company_exchange_available`, already corrected on the training session's branch, 96689b7) and X3; until the rebase they are known gaps. |
| Evaluation fixes | M4 (stock-turn history across turns), M5 (B&O `comesWith`, market cell types), M6 (log: the beats' shapes to 100%; Ruby's new run strings with a stop visited twice, rl18xx-next c57803b), M8 (tile drawings' location/borders), M10 (`user` on appended steps: client). |
| New engine API | `version`, `check`, `programInfo`, `shadowKey` (§3.2); the shadow key (§5.3); the `site_engine` CLI (§5.4); `--target no-modules` glue. |
| Client words/rawmenu split | living-board master moved 49 commits past the spike (1867, own-only, the auction lock UI, par/station fixes): the spike branch conflicts in `moves.ts` and `styles.css`. C re-applies the approach on today's master instead of merging the spike branch. |
| Goldens, title pack | Regenerated from today's living-board master and :9301's Opal bundles (the site's Ruby changed: auction lock 1b5c534, run strings c57803b). |
| `selfplay_checksum.py` | Replaced by the cargo test `neutrality.rs` (§8.4), which needs no wheel and runs under every feature set. |

### 2.4 The evaluation's open classes in step 1

- **X1** (bids on another private while one is up for auction): on *this* site it no longer disagrees: the site's
  Ruby refuses such bids since rl18xx-next 1b5c534 (2026-11-12, Caleb's auction lock), as Rust does; only the refusal's
  words differ, and neither menu offers the bid. The decision on X1 itself stays Caleb's; step 1 changes nothing.
- **X2** (M&H exchange offered where Ruby refuses) and **X3** (bankruptcy not offered at 2 positions): fixed by the
  training session in engine-rs on master; A rebases onto them and re-runs everything (§9.4). A does not touch them.
  (X2 landed, 96689b7. X3 turned out to be the site's own rule, not the engine's: Caleb kept voluntary bankruptcy for
  the site's players, so the site menu offers it and the bots' enumeration keeps the printed rule: §11.2.)
- **M4-M10**: A and C fix them (§2.3).

---

## 3. The WASM API and RawMenu

### 3.1 Principles (kept from the spike)

Every structured value crosses as a JSON string; errors are thrown with a code prefix the worker maps: `unsupported:`
(title, rule, feature out of scope: the session falls back to Opal), `refused:` (the engine refuses the action: its
words), `resync:` (undo/redo or an id gap: re-open), `internal:` (a bug: fall back and report). Players keep their site
user ids; seating is the JSON's `players` order. Undo/redo resolution is Ruby's `filtered_actions` over all raw actions.
Action ids may skip (chat hidden from a viewer). Every snapshot/menu/step shape is living-board's `types.ts`
(`Snapshot`, `StaticInfo`, `Step`, `ActionLite`, `TileSpec`) and the spike's `RawMenu`.

**Output determinism** (new rule): no output may depend on hash-map iteration order (BTreeMap, or sort), so the native
build (server, harness) and the wasm build (browser) give byte-identical JSON; A checks it on the corpus (§8.2).

### 3.2 `WasmGame`, API 2

```ts
// the served package: engine_rs.js (wasm-bindgen --target no-modules: defines `wasm_bindgen`), engine_rs_bg.wasm
// worker-rs: importScripts(glue); await wasm_bindgen(wasmUrl)  (node tests: vm + wasm_bindgen.initSync(bytes))
export class WasmGame {
  /** NEW: {engine: "engine-rs", commit, api: 2, key: 1, titles: ["1830"], features: [...]}; the worker refuses
   *  (falls back) unless api === 2 */
  static version(): string;
  static load(gameJson: string): WasmGame;          // throws unsupported: (title, optional rule)
  staticInfo(): string;                             // engine part of StaticInfo (+ the pack)
  startLog(): string;                               // step 0's log lines
  replay(maxSteps: number, untilActionId?: number): string;   // {steps, loaded, total, done, redo, stopped?}
  view(): string;                                   // Snapshot now
  currentActionId(): number;
  append(actionJson: string): string;              // a stored action (live): its Step or "null" (chat); resync:
  act(actionJson: string): string;                  // a move: processes it, {posted, step}; posted = site action, no auto_actions
  /** NEW: as act, on a copy: {posted} or throws refused:; nothing changes (the worker posts it, then appends the
   *  stored action: §7.1) */
  check(actionJson: string): string;
  menu(playerId: number): string;                   // RawMenu (§3.3) or "null"
  tileOptions(hex: string, via?: string): string;
  autoRoute(): string;                              // {action: run_routes site JSON, revenue, routes (+track)} | "null"
  details(partsJson: string): string;               // info / tiles / sheet (engine parts)
  restartInfo(): string;                            // ports:restart
  lastRun(sym?: string): string;                    // ports:last_run
  meta(): string;                                   // ports:meta
  /** NEW: the Auto tab's facts (living-board ProgramInfo): available program types (Ruby
   *  available_programmed_actions: program_buy_shares, program_share_pass), multiple, forceUnconditional, stock,
   *  stockName, mine (from recorded program_* actions), buyable (corp, floated, toFloat, ipo, market, owned) */
  programInfo(playerId: number): string;
  /** NEW: the shadow key v1 of the position now (§5.3): the same JSON site_engine prints */
  shadowKey(): string;
  legalActions(): string;                           // the menu's actions flattened (titles:actions)
  at(actionId: number): WasmGame;                   // a copy at a past position (checkpoints every 100 actions)
  free(): void;
}
```

`act` keeps the spike's rules: `run_routes` is accepted only as the auto-router's run with the engine's own revenue;
`act` no longer refuses when a player has a program (`unsupported:programs` is gone): in such games the worker uses
`check` and the server adds the auto actions (§7.1).

**Boundary translations** (`src/site/json.rs`, both directions, each tested): G19's two preprinted cities swapped while
G19 is preprinted (`place_token.city`, `routes[].nodes`, tokens, reservations, station options, the shadow key's
tokens) — never renumbered in the engine (policy layout); tile instance ids in Ruby's supply order (an upgraded-over
tile goes back at the end); share ids `<sym>_<i>`, train ids `<name>-<n>`, city ids `<tile id>-<city index>`.

### 3.3 RawMenu

The spike's §3.3 shape, unchanged: facts and complete site actions, the words stay in TypeScript (`words.ts`), so both
engines say the same thing by construction. Changes for step 1: `exchange` and `bankrupt` from the engine's fixed
predicates (after X2/X3); `turn` reset at each new Ruby turn (M4); the auction offers only the company up for auction
while one is (as Ruby's lock and today's hand); `programs` from `programInfo`; own companies only for trains and
privates (living-board CLIENT.md §14), while the engine still *processes* cross-player purchases (`site_rules`).

---

## 4. site_rules and every engine edit outside `src/site`

**For the training session** (the parent relays this list). Bodies of the functions they named — the exchange
enumeration and processing in `rounds/stock.rs`, the buy-train enumeration in `rounds/operating.rs`, `factored.rs`,
the ability/exchange helpers (`steps::company_exchange_available`, `game::exchange_ability_available`) and their tests —
are **not edited** while X2/X3 are pending; groups V, S and H below are applied **after** A rebases onto their commits,
as small separate commits. Group P touches only `cfg` attributes and imports. Nothing changes `action_index.rs`'s
layout, the encoder, `decode`, `POLICY_SIZE` 26537 or the `src/rl` layouts (cfg attributes only), and nothing renumbers
G19.

### 4.1 Group P: pyo3 optional (one commit, with `scripts/pre-push.hook`)

| File | Edit |
|---|---|
| `Cargo.toml` | `crate-type = ["cdylib", "rlib"]`; `pyo3`, `numpy` optional; features `python = ["dep:pyo3", "dep:numpy"]`, `extension-module = ["python", "pyo3/extension-module"]` (default, so the wheel is unchanged), `site = []`, `wasm = ["site", "dep:wasm-bindgen"]`, `only-1830 = []`; `wasm-bindgen = "=0.2.123"` optional (must equal the CLI); `getrandom` `js` for wasm32; `[[bin]] site_replay`, `site_engine` (`required-features = ["site"]`); profile `wasm-release` |
| new `pyo3-noop/`, `src/pyerr.rs` | the spike's no-op attribute macros; `PyResult`/`PyErr`/`exceptions` facade |
| `lib.rs` | `cfg(python)` on `use pyo3`, `supported_titles_py`, the `#[pymodule]`; `pub mod pyerr; pub mod site;` |
| imports + `cfg(python)` on Python-only items, no body change | `action_index.rs` (`action_offsets_py`, `policy_size_py`, `policy_size_for_py`, `legal_action_to_index_py`, `py_to_json_value`, `_force_unused`), `actions.rs` (`Action::from_py_dict`, the `extract_*` helpers; `From<GameError> for PyErr` via `pyerr`), `core.rs`, `entities.rs`, `graph.rs`, `decode.rs` (`apply_action_index`, `decode_index_to_map`: `crate::pyerr::PyErr`), `encoder.rs`, `factored.rs` (`LegalAction::to_py_dict`, `json_to_py`, the `#[pymethods]` block: `get_factored_choices`, `factored_legal_indices`), `game.rs` (`py_to_json`, `json_to_py_obj`, the `#[pymethods]` constructors and `process_action(dict)`, `raw_actions`; `pyo3::exceptions` → `crate::pyerr::exceptions`), `gumbel.rs`, `mcts.rs`, `mcts_lockstep.rs` (its `#[pyfunction]`s), `price_pmf.rs`, `steps.rs`, `g1822/mod.rs` + `g1822/py.rs`, `g1867/mod.rs` + `g1867/py.rs`, `g1867/router.rs`, `rl/mod.rs` + `rl/py.rs`, `rl/encode/mod.rs` + `rl/encode/py.rs`, `rl/generator.rs` (its `#[pymethods]` block and `policy`) |
| `title/mod.rs` `all_titles` | a `cfg(feature = "only-1830")` arm (the registry of the 1830-only wasm) |

### 4.2 Group S: the site_rules branches (each its own commit)

`site_rules` is a `BaseGame` field like `no_buyback_rule`: runtime, default false, set only by `src/site`
(`SiteGame::load`), never by Python (no setter is exported), never in `start_positions`, the generators, the importer
or eval. Only the site's house-rule differences go behind it: player-to-player private purchases, buying trains from
other players' corporations, and Ruby's Buy Trains step staying open while they hold trains. The legal-move
enumeration keeps the RL rule in every mode (the client's menu is own-only too).

| # | Function (file) | Branch | Ruby method it mirrors |
|---|---|---|---|
| S0 | `BaseGame` struct, `build_seeded`, `clone_for_search_log` (`game.rs`) | the field `site_rules: bool` (and `site: Option<Box<SiteRecords>>`, §4.3): declared, initialised false / None, copied | — |
| S1 | `or_process_buy_train` (`rounds/operating.rs`), the "other_trains" membership | `cross_president = title rule \|\| self.site_rules` | `Engine::Depot#other_trains(corporation)` with `ALLOW_TRAIN_BUY_FROM_OTHER_PLAYERS = true` (base; 1830 keeps it), accepted by `Engine::Step::BuyTrain#process_buy_train` |
| S2 | `operating_step_auto_skips` (`rounds/operating.rs`), the BuyTrain arm's `can_buy_inter_corp` | `cross = title rule \|\| self.site_rules`: the step stays open while any other corporation holds a train | `Engine::Step::BuyTrain#actions` → `Engine::Step::Train#can_buy_train?` → `buyable_trains` (incl. `@depot.other_trains(entity)`) |
| S3 | `player_has_actions` (`rounds/stock.rs`) | `if self.site_rules && self.site_interplayer_purchase_open(..) { return true }`: the turn waits for the player | `Engine::Game::G1830::Step::BuySellParShares#purchasable_companies` (not bought, cash > 0, phase status `can_buy_companies_from_other_players`, not turn 1) → `Engine::Step::BuySellParShares#actions` |
| S4 | `step_actions` (`steps.rs`), the Stock arm | push `"buy_company"` under the same condition | `Engine::Step::BuySellParShares#actions` (`actions << 'buy_company' if !purchasable_companies(entity).empty?`) |
| S5 | `stock_process_buy_company` (`rounds/stock.rs`) | `if self.site_rules`: `bought_this_turn = true`, and the priority deal is not moved | `Engine::Step::BuySellParShares#process_buy_company` (`@round.current_actions << action`, so `bought?` via `PURCHASE_ACTIONS`; no `track_action`, so `@round.last_to_act` is unchanged) |
| S6 | `stock_process_buy_company` (`rounds/stock.rs`), before the purchase moves anything | `if self.site_rules { self.site_after_buy_company(company_sym, player_id)? }`: the private's `shares` ability again (C&A's PRR_1): refused while a player holds the share ("Cannot buy share from player"), else the buyer takes it free and the presidency follows | `Engine::Step::BuySellParShares#process_buy_company` → `Engine::Game::Base#after_buy_company` → `SharePool#buy_shares(player, share, exchange: :free)` (verification finding, 2026-11-26) |

The helpers the branches call (`site_interplayer_purchase_open`, `cross_player_train_buys`, `site_after_buy_company`) live in
`src/site/rules.rs`, compiled in every build (they are the engine half of site mode; the rest of `src/site` needs the
`site` feature). S5 is Ruby-faithful processing of a move the RL rules never make; it is behind `site_rules` because
the Python importer path keeps today's behaviour (Python is frozen) and the training session may make it unconditional
later. For 1867 and the 1822 family the title rule already equals Ruby's, so `||` changes nothing there; each later
title re-checks this per title.

### 4.3 Group H: record hooks (behind `site_rules`, one commit)

| # | Function (file) | Hook | Ruby method it mirrors |
|---|---|---|---|
| H1 | `or_process_dividend` (`rounds/operating.rs`) | `if self.site_rules { self.site_note_operated(corp, revenue, kind, false) }` | `Engine::Step::Dividend#process_dividend` (`operating_history[[turn, round_num]] = OperatingInfo`) |
| H2 | `operating_step_auto_skips` (`rounds/operating.rs`), the Dividend auto-skip at revenue 0 | `... site_note_operated(corp, 0, "withhold", true)` | `Engine::Step::Dividend#skip!` (a `withhold` dividend with no routes) |
| H3 | `transition_to_next_round` (`game.rs`) | `if self.site_rules { self.site_note_round_finished() }` | `Engine::Game::Base#transition_to_next_round!` → `store_player_info` |
| H4 | `end_game` (`game.rs`) | same, before `finished = true` | `Engine::Game::Base#end_game!` → `store_player_info` |
| H5 | `process_action` (`game.rs`), the D&H teleport branch's inline auto-withhold (a company with trains and no route after D&H's token) | `if self.site_rules { self.site_note_operated(corp_sym, 0, "withhold", true) }` | `Engine::Step::Dividend#skip!` (a `withhold` with no routes; that branch never reaches H2's `operating_step_auto_skips`) (verification finding, 2026-11-26) |

S2 and H2 touch the same function, `operating_step_auto_skips`; if X3 edits it, both wait for X3 and are re-applied on
top. H5's branch (in `process_action`) also left `ever_operated` unset where the Dividend arm on master sets it; at
the training session's review that is now set in every mode (U1, §4.6).

### 4.4 Group V: visibility only (`fn` → `pub(crate) fn`, bodies unchanged)

`game.rs`: `end_game`, `active_players`, `priority_deal_player_py`, `num_certs`, `cert_limit_py`,
`get_tokenable_cities`, `legal_action_types`, `current_entity_id`, `upgradeable_tiles_for`, `legal_tile_rotations`,
`buyable_trains_for`, `president_may_contribute`, `dividend_options`, `active_step_type`, `teleport_pending`,
`auctioning_company`, `auction_companies`, `auction_bids`, `auction_min_bid`, `auction_max_bid`,
`auction_pending_par`, `operating_step`, `stock_sold_corps`, `stock_bought_this_turn`, `stock_bought_from_ipo`,
`stock_can_buy_more`, `stock_bought_corp_this_turn`, `buyable_shares`, `sellable_bundles`, `par_prices_with_coords`,
`connected_hexes`, `must_buy_train` (those still private on master; A re-checks after the rebase).
`steps.rs` `company_exchange_available`: only after X2 lands (it is X2's function), and only if X2 left it private.

### 4.5 Training neutrality

With `site_rules` off the RL engine must be byte-identical to master. The test `engine-rs/src/neutrality.rs` (§8.4)
was pinned on master a615a40 **before** any change (commit 359463c, the first of this branch) and must pass at every
commit of A, under `--features python` and `--features site`. When X2/X3 land on master and A rebases: if the
digests move, the move is master's (the training session's commit message for X2 says its enumeration hashes are
unchanged): re-pin from master itself in a separate commit that names the master commit.


### 4.6 Group U: unconditional (asked for at the training session's review, 2026-11-26)

| # | Function (file) | Edit | Ruby method it mirrors |
|---|---|---|---|
| U1 | `process_action_internal` (`game.rs`), the D&H teleport branch's inline auto-withhold (H5's place) | `self.corporations[ci].ever_operated = true`, in every mode (not only `site_rules`), as the Dividend auto-skip arm of `operating_step_auto_skips` does | `Engine::Step::Dividend#skip!` → `process_dividend` (`operating_history` written: the corporation has operated) |

In 1830 the flag feeds only the encoder's "operated" feature (`rl/encode/encode.rs`; `SELL_AFTER = :operate` is
1867's), and a company with trains at D&H's teleport has always operated before (it gets trains only in its own Buy
Trains step, after its Dividend step), so no position changes: the neutrality digests are unchanged. Test:
`site::tests::d_and_h_teleport_without_a_route_operates_the_company_in_every_mode` (17@246 with the flag cleared
first, `site_rules` on and off; it fails without U1).
### 4.7 Step 2 (1867): the edits outside `src/site`

Listed function by function in docs/SITE_ENGINE_RS_1867.md §5.3 (branch `site/engine-rs-1867`): **P2** (`Cargo.toml`,
`title/mod.rs` `all_titles`: the cfg arm `site-titles` registering 1830 and 1867, replacing `only-1830`), **V2**
(`steps.rs` `blocking_step_entity`, `step_actions`: `fn` → `pub(crate) fn`), **P3** (`src/clock.rs`; `g1867/legal.rs`,
`g1867/router.rs`: their timing counters read `crate::clock::Instant`, which is `std::time::Instant` on native and a
zero-reading stand-in on wasm32, where std's panics). No `site_rules` branch; `site_interplayer_purchase_open` reads
Ruby's phase status from a table in `src/site/rules.rs` (the review note; §5.2 there).

---

## 5. The server: shadow mode

### 5.1 The integration: a native CLI run by the queue (`site_engine`)

| Option | For | Against | Verdict |
|---|---|---|---|
| **magnus / rb-sys extension** in the Ruby process | in-process, µs calls; natural for step 2 | an abort, stack overflow or panic across FFI takes the unicorn worker and the move with it; rb-sys needs libclang and a Rust toolchain in both images and rebuilds on every Ruby upgrade; the site's gems are installed in the `gems` stage, engine-rs lives in another repo | no |
| **wasmtime** running the same wasm | sandboxed (a trap is a Ruby exception); arch-identical to the browser | the browser's module is wasm-bindgen's and needs its JS glue, so the server needs a second C ABI and a WASI or getrandom shim; a native gem; compiling a 1.4 MB module per process (or a cache); and the browser bytes are already checked position by position by C's node suite | no |
| **Rust sidecar service** (HTTP) | isolated; a natural step-2 engine service | a long-running service, a port, health checks, a compose service in dev and prod, all for a check that runs a few times a minute | not yet |
| **Rust sidecar process: `site_engine` CLI** | isolated per check (a crash is one failed check), stateless (a full replay from the stored actions: ~10 µs/action, a 1,000-action game in ~15 ms with the JSON), no port and no service, trivially testable (`site_engine state < game.json`), the same binary is the corpus tool, and it grows a `serve` mode for step 2 | process spawn (~ms), negligible at the dev site's rate | **chosen** |

**Build** (it builds in the site's Docker images and runs on the side-work cores):
- The engine source is vendored into rl18xx-next as `engine-rs/` by `scripts/engine-rs-pull.sh <RL_18xx commit>`
  (`git archive` of A's branch; `engine-rs/PINNED` holds the commit and branch). It changes only by that script, as
  `client/` changes only by subtree pull. (Not under `vendor/`: the dev stack mounts an anonymous volume there.)
- `engine-rs/site/build-site.sh <out>` (A owns it; it travels with the source) builds `<out>/bin/site_engine`
  (`cargo build --release --no-default-features --features site --bin site_engine`) and `<out>/pkg/`
  (`--features wasm,only-1830`, wasm-bindgen `--target no-modules`, wasm-opt `-Oz`, `BUILD.json` with commit, api,
  key version, sizes).
- Production: `Dockerfile.next-prod` gains a stage `FROM rust:<pinned>-bookworm AS engine` (wasm32 target,
  `wasm-bindgen-cli` 0.2.123 `--locked`, binaryen from a pinned release tarball checked by sha256, like esbuild) that
  runs `build-site.sh /out`; the runtime stage copies `/out/bin/site_engine` to `/usr/local/bin/` and `/out/pkg` to
  `public/engine-rs/<sha12>/`. bookworm matches `ruby:3.4`'s glibc.
- Development: `scripts/engine-build.sh` runs the same stage (`Dockerfile.engine`, image `18xx_engine:dev`, under
  `taskset -c 24-31,56-63 nice -n 19`) and copies its outputs into the mounted tree: `engine-rs/dist/bin/site_engine`
  and `public/engine-rs/<sha12>/` (both gitignored). The shared dev image `18xx_rack:dev` is not rebuilt; the queue and
  rack containers see the binary through the mount and run it with their cpuset (24-31,56-63).
- `ENGINE_RS_BIN` names the binary (dev `/18xx/engine-rs/dist/bin/site_engine`, prod `/usr/local/bin/site_engine`).

### 5.2 The flow

1. **In the request** (`record_action`, `start_game!`; bot moves go through `record_action`): after Ruby processed the
   action (auto actions included), if `EngineShadow.enabled?(game)` (`ENGINE_SHADOW=1` and the title in
   `ENGINE_SHADOW_TITLES`, default `1830`; never `settings.pin` games), `EngineShadow.capture(engine, users, action)`
   builds Ruby's state (§5.3) — rescued (any exception: no shadow, one log line), about a millisecond — inside the lock;
   after the lock, `EngineShadow.enqueue(game, action_id, state)` inserts a `pending` row (rescued). Chat messages are
   not shadowed. Nothing in the shadow can refuse, delay or fail the move.
2. **The queue process** (`queue.rb`: `scheduler.every '3s', overlap: false`): `EngineShadow::Worker.tick` takes up to 50
   pending rows, oldest first. For each, it builds the game JSON from the database (players, settings, title, **all**
   raw actions with `action_id <= row.action_id`, chat included: not `Game#to_h`, which hides chat from viewers), runs
   `site_engine state` with a 10 s hard timeout (`Process.kill` on expiry), compares, and writes the row. Engine
   missing or failing: the row stays `pending` with a back-off; pending for over a day: `skipped`.
3. **The comparison** (§5.3, §5.5) writes `status`: `equal`, `differs`, `refused` (Rust stopped on a recorded action:
   which action and Rust's words), `error` (the CLI failed or timed out), `skipped`. A refusal at the same action as an
   earlier row of the same game is `known` (no new log line), so one divergence is reported once until an undo
   removes it.

### 5.3 The shadow key, v1 (implemented twice: Ruby in B, Rust in A, compared as JSON)

Hash keys sorted, arrays in the order given, integers as integers. `players` in Ruby's `engine.players` order (it
rotates); corporations in Ruby's `engine.corporations` order.

| Field | Ruby (B, `lib/engine_shadow.rb`) | Rust (A, `src/site/shadow.rs`) |
|---|---|---|
| `v` | 1 | 1 |
| `round`, `turn`, `round_num` | `engine.round.name`, `engine.turn`, `engine.round.round_num` | the site layer's Ruby round names |
| `phase`, `bank` | `engine.phase.name`, `engine.bank.cash` | |
| `acting` | `engine.active_players_id` (what `set_game_state` stores in `games.acting`) | site ids |
| `entity`, `step`, `actions` | as `bot_view`: the active step's first active entity (or `current_entity`) `id.to_s`; `step.class.name.split('::').last`; `step.actions(entity).map(&:to_s)` | RawMenu's `step` / `types` facts |
| `priority` | `engine.priority_deal_player&.id` | |
| `players[]` | `{id, cash, shares: {corp.id => p.percent_of(corp)} (non-zero), companies: ids sorted, bankrupt}` | |
| `corps{sym}` | `{cash, price: share_price&.price, cell: share_price&.coordinates, par: par_price&.price, floated: floated?, closed: closed?, ipo: ipo_shares percent, market: share_pool.percent_of(c), president: owner&.id, trains: trains.map(&:id), tokens: used tokens as "<hex>-<city index on its tile>" in token order, companies: ids sorted, last_run: [turn, round, revenue, kind] of the latest operating_history entry or null}` | G19 in Ruby's numbering |
| `companies{sym}` | `"closed"`, `"p<id>"`, `"c<sym>"` or `"bank"` | |
| `depot` | `{upcoming: depot.upcoming.map(&:id), discarded: depot.discarded.map(&:id)}` | |
| `hexes{id}` | for hexes whose tile isn't the original: `"<tile.name>@<rotation>"` | |
| `finished`, `result`, `manually_ended`, `game_end_reason` | `engine.finished`, `engine.result` (keys as strings), `engine.manually_ended`, `engine.game_end_reason&.to_s` | |
| `last_action` | for a `run_routes`: `{revenues: routes' revenue, extra_revenue, subsidy}`; else `{type}` | Rust's own pricing of the stored routes (`recorded_route_revenue` on the position before) |

The key is deliberately smaller than the client's Snapshot (which C compares against Opal at every position): it is
what decides a game (money, holdings, track, tokens, trains, who acts and what they may do) and cheap to read in MRI.
A version change (v2) is a commit to this doc first; rows whose versions differ are `skipped`.

**v1 as built** (A, `src/site/shadow.rs`, 2026-11-26; clarifications within v1, B's Ruby must do the same):
- Finished: `acting` is `[]`, `entity` and `step` are `null`, `actions` is `[]` (bot_view returns before reading the
  step; nobody acts in a finished game). `manually_ended` is a boolean (Ruby's `nil` before the end: `false`);
  `game_end_reason` is `null` until finished; `result` is always present (the players' current values, keys as
  strings).
- Closed corporations are left out of `corps` and of every player's `shares` (Ruby's `close_corporation` deletes
  them from `engine.corporations` and the holders' `shares_by_corporation`).
- `corps.ipo`: the percent the IPO (the corporation) holds, 100 before par; `market` the pool's percent.
- `tokens`: the corporation's used tokens in `corporation.tokens` order, `"#{t.hex.id}-#{t.city.index}"`.
- `companies`: `"bank"` also for an unowned private (the auction).
- `depot.upcoming`: Ruby's `depot.upcoming` (one D at a time: it is unlimited, `clone_unlimited`).
- `hexes`: only hexes whose tile is not the original (`"#{tile.name}@#{tile.rotation}"`).
- `last_action`: `state` mode: the raw action at `upto` (any type, e.g. `{"type": "undo"}`); `keys` mode: the kept
  action the position follows, `null` at position 0. A `run_routes`: `{"revenues": [...], "extra_revenue": n,
  "subsidy": n}` (no `type`; integers, Ruby's `nil` as 0). Rust prices each route itself on the position before it
  (the stops its connection chains end at, each at its revenue in the phase).
- `entity`: `id.to_s` (a player's id as a string, a corporation's or company's sym).

### 5.4 `site_engine` CLI (A owns it; B calls it)

One JSON object on stdin, one JSON line on stdout; exit 0 whenever a verdict was reached (a Rust refusal included),
2 bad input, 3 unsupported (title / optional rule), 101 a panic (the worker records `error`).

```
site_engine version                    -> {"engine":"engine-rs","commit":"<sha>","api":2,"key":1,"titles":["1830"]}
site_engine state  {"game": <site game JSON>, "upto": <action id | null>}
    -> {"ok": true, "state": <key v1>, "stopped": null | {"actionId", "index", "message"}, "ms": 12}
site_engine keys   {"game": ...}       -> {"ok": true, "positions": [{"actionId": 0, "state": ...}, ...], "stopped": ...}
       (every kept position in one replay: the corpus mode, §8.2)
site_engine act    {"game": ..., "action": ..., "add_auto_actions": false}
    -> {"action": <the site action as Rust would store it> | null, "error": null | "<words>", "state": ...}
       (the act-level suite, §8.3; Rust has no Programmer, so add_auto_actions: true answers unsupported)
```

`serve` (an HTTP mode over the same handlers) is reserved for step 2.

### 5.5 What is compared per action, and where disagreements go

Per shadowed action: (1) Rust accepts every raw action up to it (else `refused`); (2) `acting`; (3) `round`, `turn`,
`round_num`, `phase`; (4) `entity`, `step`, `actions`; (5) `finished`, `result`, `manually_ended`, `game_end_reason`; (6) the state key's
remaining fields; (7) `last_action` (a stored run's revenues as Ruby computed them, §6, against Rust's pricing).
`diffs` lists up to 20 field paths with both values (`corps.PRR.cash: ruby 120, rust 110`).

**Where:** table `engine_shadow` (migration by B: `id`, `game_id` (cascade), `action_id`, `title`, `source`
(`action`, `start`, `bot`, `client`, `revenue_mismatch`), `status`, `ruby` jsonb, `rust` jsonb, `diffs` jsonb,
`message`, `engine_commit`, `ms`, `created_at`, `checked_at`; indexes `(status, id)`, `(game_id, action_id)`;
`equal` rows older than 14 days are deleted by the daily job); a queue log line per non-equal row
(`[engine-shadow] game=G action=A status=differs fields=...`); the admin page's "Engine shadow" panel (counts by
status for 24 h / 7 d, the latest 50 non-equal rows linking to `/game/<id>?action=<a>`); and
`GET /api/admin/engine_shadow?status=&since=&game=` (admins) for the parity tooling.

---

## 6. Run revenue on the server (every title)

Today `Action::RunRoutes.h_to_args` passes the posted `revenue` into `Route.new` and `Route#revenue` is
`@revenue ||= begin <every check> end`, so a posted revenue skips every route check and is stored as is. In
`record_action` (B, `lib/run_revenue.rb`; site code only, `lib/engine` untouched):

1. For `params['type'] == 'run_routes'`: remember the posted numbers; strip `revenue`, `revenue_str` and `subsidy` from
   every route and `extra_revenue`, `subsidy` from the action.
2. Build the routes from the stripped hash on the engine at the current position
   (`Engine::Action::RunRoutes.h_to_args(stripped, engine)`) and set `extra_revenue = engine.extra_revenue(entity,
   routes)`, `subsidy = engine.routes_subsidy(routes)` — what every client computes (living-board `manual.ts`; the
   classic `route_selector.rb`). Process as today: `Route#revenue` now runs every check, and the stored action
   (`raw_actions.last.to_h`) carries Ruby's own revenue and revenue string.
3. If a posted number differs from the stored one: store Ruby's, write an `engine_shadow` row (`source:
   revenue_mismatch`, both values, any title) and answer `repriced: true` with the stored action, so the page resyncs.
   Not a refusal: revenue is a function of the routes the player chose, and refusing would leave a page with a wrong
   pricing stuck. An invalid route still raises in Ruby and is refused as today.
4. Applies to bots' moves too (`/api/bot/game/<id>/action` goes through `record_action`). `POST /api/bot/engine`
   (18xx.games games) is not changed (its contract).

Before switching it on, B runs it over every stored `run_routes` in the :9301 database and the production export
(all titles): stored == recomputed, or the title's difference is explained (a title that needs a client-only input
gets an allowlist entry with a reason until fixed).

---

## 7. Auto actions; the client switch and fallback

### 7.1 Auto actions (programs): Ruby-only in step 1

- A Rust page **posts with `X-Engine: rust`**. For such a request the server processes with `add_auto_actions: true,
  validate_auto_actions: false` (as for bots) and ignores posted `auto_actions`; Opal pages keep today's
  `validate_auto_actions: true`. The response of `POST /api/game/<id>/action` gains `action` (the stored action) for
  every client (`repriced` when §6 changed it).
- The Rust worker: in a game where no player has an enabled program (the snapshot's `programs`), moves are applied
  locally at once (`act`) and posted, as today (with no programs Ruby makes no auto actions in 1830). Where any player
  has one, the move is checked (`check`), posted, and the **stored** action from the response is applied with
  `append` (the MessageBus echo of the same id is then ignored). If the stored action differs from what was applied
  locally in any other way (`repriced`), the session resyncs.
- Rust replays recorded `auto_actions` after their action and `program_*` actions as state (the spike's L4/G1); the
  shadow checks the position after them. The Auto tab works in Rust mode (`programInfo`; creating or cancelling a
  program posts a `program_*` action, which Ruby validates).
- Step 2 ports Ruby's Programmer (`programmed_auto_actions` per step, `program_disable` reasons) so Rust can generate
  them; the shadow will then compare generated auto actions too.

### 7.2 The switch and the fallback

- **Site flag** (B): env `RUST_ENGINE_TITLES` (dev `1830`, prod unset) and `RUST_ENGINE_DEFAULT` (`opal` until the
  acceptance in §8.6, then `rust` on dev). The boot block gains `engine.rust = {titles, default, glue, wasm, commit,
  api}` (absolute, content-addressed, immutable URLs under `public/engine-rs/<sha12>/`), and `GET
  /api/titles/engine?title=` returns the same `rust` object for those titles. CSP: `script-src` gains exactly
  `'wasm-unsafe-eval'` (the worker compiles the wasm; nothing else is widened). `.wasm` is served as
  `application/wasm` with immutable caching.
- **Browser setting** (C): Settings row "Rules engine: Automatic / The site's engine / Rust (1830)" in `localStorage`
  `lb.engine` (`auto` | `opal` | `rust`, default `auto` = the site's default); `?engine=rust|opal` overrides it for
  that page only. A header badge says when a game runs on Rust.
- **Choice:** Rust only when the title is in `engine.rust.titles`, `settings.auto_routing !== false`, the choice is
  `rust` (or `auto` with default `rust`), and `WasmGame.version().api === 2`.
- **Fallback:** any `unsupported:` / `internal:` from the Rust worker, a replay that stops on a recorded action, a wasm
  that fails to load or compile: the session re-opens the game on Opal (lazy-loading the Opal bundles), shows one
  sentence saying why, and posts `POST /api/game/<id>/engine_report {kind: "fallback" | "refused", action_id, message,
  commit}` (B: logged-in players of the game, at most one per game and user per minute, stored as an `engine_shadow`
  row with `source: client`). A move the Rust menu offered but the server refused also reports (`refused`).
- **Kill switch:** remove the title from `RUST_ENGINE_TITLES` (restart rack): every page opens it on Opal. A bad
  engine build is rolled back by re-pinning `engine-rs/PINNED` and `scripts/engine-build.sh`.
- In Rust mode the controls for runs drawn by hand are hidden (the exact auto-router's run is offered); `titles:actions`
  answers from `legalActions`.

---

## 8. Verification

### 8.1 The corpus (shared at `$RUST1/corpus`, §9.1)

| Set | How | Owner |
|---|---|---|
| Every 1830 game on :9301 (the spike's 939, ids to 1115, plus every newer one: ids are past 1,200) | `GET /api/game/<id>` read-only (chat hidden is fine for engines); deduplicated by players, rules and action stream; refreshed before each acceptance run | A |
| Production 1830 games | the bot API's read-only export: `curl -s -H @<(printf 'Authorization: Bearer %s\n' "$(cat ~/.config/18xx/local_site_bot_token)") http://127.0.0.1:9292/api/bot/game/<id>` (the `game` key), files mode 600 in `$RUST1/corpus/prod/`, deleted when step 1's acceptance is done (real users' names); never printed, never committed | A (B and C read while it exists) |
| Ruby's 1830 fixtures | rl18xx-next `spec/fixtures/1830/` (`1830_game_end_bank`, `26855`, `29133`) | A |
| RL_18xx site-format fixtures | `tests/fixtures/1830/*` except the engine-format `train_rotation_game.json`, `train_shuffle_game.json`; `tests/test_games/*`; `tests/fixtures/advisor/*` (RL streams: they must stop where Opal stops, with the same reason) | A |
| Generated games | Rust self-play walks (the neutrality walks' generator with `site_rules` on, own-only enumeration) converted to site JSON and replayed by Ruby (`scripts/ruby_replay_check.py`'s rig): every Ruby refusal is a Rust rule bug | A |

Games Opal or Ruby can't replay are listed with the reason and excluded; nothing else is excluded.

### 8.2 Per position

- **Rust vs Opal (A's `site_replay --goldens`, C's `test/rust/parity.cjs`)**: goldens regenerated by C from today's
  living-board master with :9301's Opal bundles. At every kept position: snapshot (every field), ActionLite, run
  tracks, log lines; the menu of **every seated player** (null menus included); tile options at every lay position
  for the hexes the menu lists; the auto-router at every run position; `details`, `restartInfo`, `lastRun`, `meta` and
  `programInfo` every 10 positions; StaticInfo merged with the pack.
- **Rust vs Ruby (B's `scripts/engine_rs/ruby_keys.rb` + A's `site_engine keys`)**: Ruby's shadow key at every kept
  position of every corpus game (MRI, a throwaway `18xx_rack:dev` container with no network and the source
  read-only), against Rust's. Same corpus, so the two key implementations are debugged before the live shadow.
- **Native vs wasm:** the digest of every step's JSON (and every key) from `site_replay` and from the wasm in node must
  be equal for the whole corpus (§3.1 determinism).

### 8.3 Acting on every offered option, judged by Ruby

Menu equality is not enough (X1 hid behind equal menus). A's `site_replay --acts` writes, at every decision position
of the distinct corpus, every RawMenu option as a case (the min and max of priced options; every tile and rotation for
the hexes of sampled lay positions, all hexes for 50 games), acted in Rust (`act`: accepted with the posted JSON and
the key after, or refused with the words). B's `scripts/engine_rs/ruby_verdicts.rb` runs each case through Ruby
exactly as `record_action` does (both `validate_auto_actions: true` with Opal's auto actions and `add_auto_actions:
true` as for `X-Engine: rust`), recording the verdict and the key after. C runs the same cases through Opal's `act`
(the posted JSON must be identical, `created_at` aside) and, the other way, acts **Opal's** menu options in Rust.
Required: no option accepted by one engine and refused by the other (in either direction), equal keys after every
accepted option, Rust's posted JSON equal to Opal's, and every Rust auto route accepted by Ruby with its revenue
stripped at Rust's revenue.

### 8.4 Training neutrality (pinned before any change)

`engine-rs/src/neutrality.rs`, commit **359463c** (the first commit of `site/engine-rs-1830`, on master a615a40 with
nothing else changed): 60 fixed-seed 1830 walks (2-6 players; the plain rules, the RL rule variants
`no_buyback_rule` + `auction_unlock`, and those with both optional rules; uniform and 80%-busy picks) driven through
the RL index API exactly as self-play drives it (factored enumeration → the frozen 1830 layout → price range →
`decode_index` → `process_action_native`). Each walk hashes every position's sorted legal index set (with pass
flags), the index and price chosen, the action log and the result (FNV-1a; no hash-map order: identical digests in
separate processes). 74,106 moves, 58 of 60 walks finished (44 in phase D), ~5 s in release; the full
`cargo test --release --no-default-features` on 359463c: 267 passed, 0 failed, 11 ignored. It runs in every cargo gate
(`--features python`, `--features site`), so every commit of A shows the RL engine unchanged with `site_rules` off. It
replaces the spike's `selfplay_checksum.py`.

### 8.5 Live

- The shadow on B's scratch stack, then on :9301: test-user games (`lbfb_p1`..`p4` only) covering every move kind of
  the spike's §4.1 (auction, bid war, B&O par, buys/sales/pars, C&StL and D&H, stations, runs, dividends, depot / own
  company / trade-in trains, emergency money, discards, privates into companies, the player-to-player purchase, M&H,
  bankruptcy, manual end), posted from Rust pages and from Opal pages, with programs on in some. Bots' games on :9301
  are shadowed as they come. Games 47, 1218, 1219 and real users' games are never acted in.
- Browser (C, Playwright, C's server.mjs and B's stack): the 20 largest and 20 random games open on Rust with every
  step equal to Opal's; live following, undo / redo re-open; every fallback path (non-1830, `auto_routing: false`,
  a stopped replay, a wasm that fails, an API mismatch, the kill switch); a hotseat game from the auction to the end on
  Rust, every position equal to Opal's replay of its JSON.

### 8.6 Gates

- **A, every commit:** `cargo test --release --no-default-features --features python` (neutrality included),
  `--no-default-features --features site`, `cargo check --target wasm32-unknown-unknown --no-default-features
  --features wasm`, all under `taskset -c 24-31,56-63 nice -n 19`.
- **A, before proposing the branch** (rule 6): the full pre-push gate in A's own worktree venv (`uv sync` there,
  `CUDA_VISIBLE_DEVICES=`): `pytest tests/` (~45 min) and the cargo gates. A pytest test comparing Python with Rust
  that fails is reported with whether it fails on master too; Python is never changed for it (the training session
  retires such tests).
- **Step 1 accepted on dev** (then `RUST_ENGINE_DEFAULT=rust` on :9301): the corpus targets of §9.3 met; §8.3 with no
  disagreement; X2/X3 landed and rebased; 7 days of shadow on :9301 with every non-equal row explained; neutrality and
  all gates green. Production stays untouched; its Docker build is only checked locally (§9.3 B5).

---

## 9. Packages, files and contracts

### 9.1 Shared ground

- `RUST1=/tmp/claude-1000/-home-revys-workspace/252285c2-5545-52b0-bcda-165a3e4e9ceb/scratchpad/rust1`:
  `pkg/<sha>/` (A: `engine_rs.js` (no-modules glue), `engine_rs_bg.wasm`, `engine_rs.d.ts`, `site_engine` (host
  build), `BUILD.json`) and `pkg/current` (a symlink A moves with `ln -sfn` after the directory is complete);
  `corpus/` (§8.1); `goldens/`, `titlepack/` (C); `acts/` (§8.3 cases and verdicts); `reports/`; each package's
  scratch in `A/`, `B/`, `C/`.
- Rules for all three: production (:9292, project `18xx-next-prod`) is off limits except the read-only export; never
  touch `~/workspace/18xx-local`, the RL_18xx main checkout (no `uv sync`, build or checkout there) or
  `~/longrun/deploy/*`; never push; never stop or reconfigure :5002, :5004, :5012, :5014, :5022, rl18xx-gen*,
  watchdogs, mps; no GPU; every build, test and sweep under `taskset -c 24-31,56-63 nice -n 19`; on :9301 act only as
  `lbfb_p1`..`p4` (credentials in the session scratchpad's `creds/`, never printed); leave dev games 47, 1218, 1219
  unchanged; commits end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Ports: B's scratch stack **:9372** (`scripts/next-stack.sh up 9372`; `docker compose ... down -v` at the end); C's
  `server.mjs` **:9473**; A's browser smoke, if any, **:9474**.

### 9.2 Contracts (a change is a commit to this doc first; the parent relays it)

| Contract | From → to | Content |
|---|---|---|
| **K1 WasmGame API 2** | A → C | §3.2 signatures; JSON shapes = living-board `types.ts` and the spike's RawMenu (§3.3); errors §3.1; `version()` gates it |
| **K2 package files** | A → B, C | `$RUST1/pkg/current/{engine_rs.js, engine_rs_bg.wasm, engine_rs.d.ts, BUILD.json, site_engine}`; glue and wasm are a pair; `--target no-modules` |
| **K3 `site_engine` CLI** | A → B | §5.4 commands, exit codes, JSON |
| **K4 shadow key v1** | A ↔ B | §5.3, field for field; both sides on the same corpus first (§8.2) |
| **K5 the build** | A → B | `engine-rs/site/build-site.sh <out>` → `<out>/bin/site_engine`, `<out>/pkg/*` (as K2); needs Rust ≥ the host's, the wasm32 target, wasm-bindgen-cli 0.2.123, wasm-opt (optional); B's Docker stages only call it |
| **K6 vendoring** | A → B | rl18xx-next `engine-rs/` = `git archive <commit> engine-rs` of `site/engine-rs-1830`; `engine-rs/PINNED`; only `scripts/engine-rs-pull.sh` changes it |
| **K7 boot / titles API** | B → C | `boot.engine.rust = {titles, default, glue, wasm, commit, api}`; the same object in `GET /api/titles/engine` for those titles (C adds the optional `rust` field to `TitleEngine`) |
| **K8 posting** | B → C | header `X-Engine: rust` → server-side auto actions; response `{...game, action, repriced?}`; refusals as today (`error`) |
| **K9 engine_report** | B → C | `POST /api/game/<id>/engine_report {kind, action_id, message, commit}` → 204; rate-limited |
| **K10 the title pack** | C | `title-1830.<hash>.json` built into client/dist by C (generated from the site's Opal bundles by `scripts/rust/titlepack.cjs`); the worker gets its URL at build time |
| **K11 act cases** | A → B, C | `$RUST1/acts/<run>/cases.jsonl` `{game_id, upto, action, rust: {verdict, posted?, error?, key?}}`; B writes `ruby.jsonl`, C `opal.jsonl`, same order |

### 9.3 The packages

**A — engine** (RL_18xx worktree `~/workspace/RL_18xx/.claude/worktrees/site-engine`, branch `site/engine-rs-1830`).
Files: `engine-rs/**` (within §4's list outside `src/site`), `scripts/pre-push.hook`, `scripts/site_engine/**` (probes,
the corpus fetch), `docs/SITE_ENGINE_RS.md`. Never: Python (`rl18xx/`), anything in the site or client repos.

| Step | Work |
|---|---|
| A0 (day 1) | Group P + the hook (§4.1, §8.6); the spike's `src/site` ported onto master (§2.2), compiled under `site` and `wasm`, with the spike's engine edits kept as ONE temporary top commit `TEMP site_rules (redone in A5)`; `build-site.sh`; `site_engine version/state` with key v1's first fields (`acting`, `round`, `turn`, `finished`, `result`); **publish `$RUST1/pkg/<sha>` (skeleton: `load`, `replay`, `view`, `version`, `shadowKey`)** and tell the parent |
| A1 | Key v1 complete; `site_engine keys`; corpus fetch (§8.1); strict replay and results on the whole corpus |
| A2 | API 2 (`check`, `programInfo`), RawMenu fixes (M4, the auction lock, own-only), M5, M8 |
| A3 | Log lines: the beats' shapes to 100%, Ruby's run strings (c57803b), ≥ 99.5% of lines |
| A4 | `site_replay --goldens` against C's new goldens, `--acts` (K11), native-vs-wasm digest |
| A5 | After X2/X3 are on master: rebase; drop the TEMP commit; groups V, S (S0-S5), H as small separate commits (§4); menu predicates aligned with X2/X3; re-pin neutrality from master only if master moved it |
| A6 | Full gate (§8.6), the report in this doc (§10), the branch proposed to the training session |

Acceptance: strict replay of every corpus game Opal replays, every finished game's result equal; snapshots, step
actions and tracks 100% (any exception listed with its reason); menus ≥ 99.9% of acting positions with every rest
classified; tile options 100%; auto routes never below Opal and equal revenue at 100%; log: beats' shapes 100%, lines
≥ 99.5%; Rust key = Ruby key at 100% of corpus positions; native = wasm; §8.3 with no disagreement; neutrality and all
gates green.

**B — server** (rl18xx-next worktree `~/workspace/18xx-ws/rust-shadow`, branch `next/rust-shadow` from `rl18xx-next`).
Files: `routes/game.rb` (`record_action`, `start_game!`, the response), `routes/admin.rb`, a new `routes/engine_rs.rb`
(`engine_report`, `GET /api/admin/engine_shadow`), `api.rb` (CSP, the boot block), `routes/titles.rb` (the `rust`
object), `lib/engine_shadow.rb`, `lib/run_revenue.rb`, `lib/engine_rs.rb` (binary, pkg manifest, URLs), `queue.rb`,
`migrate/*_engine_shadow.rb`, `models/engine_shadow.rb`, `Dockerfile.engine`, `Dockerfile.next-prod` (engine stage),
`docker-compose.next-dev.yml` / `docker-compose.next-prod.yml` (env only), `scripts/engine-rs-pull.sh`,
`scripts/engine-build.sh`, `scripts/engine_rs/*` (`ruby_keys.rb`, `ruby_verdicts.rb`, `revenue_audit.rb`), `.gitignore`,
`engine-rs/` (vendored, only by the pull script), `spec/**` for the above, `docs/rl18xx-next/ENGINE_RS.md` (B's notes),
a section in `DEPLOY.md`. Never: `lib/engine/**`, `client/**` (subtree pulls only, at integration).

| Step | Work |
|---|---|
| B0 | Vendoring and builds (K5, K6) against A's skeleton; the CLI runs inside the dev containers with their cpuset |
| B1 | §6 run revenue for every title + `revenue_audit.rb` over the :9301 DB and the production export |
| B2 | §5: capture, enqueue, migration, worker, comparison, logging, admin panel and API; `ruby_keys.rb` (§8.2) |
| B3 | K7, K8, K9; CSP `'wasm-unsafe-eval'`; `application/wasm` and immutable caching for `public/engine-rs/` |
| B4 | `ruby_verdicts.rb` (§8.3) |
| B5 | `Dockerfile.next-prod` builds locally (a scratch tag; nothing deployed; production untouched) |

Acceptance: the existing specs (`rspec spec/lib` and the routes' specs in a throwaway container); the new specs;
`POST /api/bot/engine` byte-identical responses before/after on a set of games; a shadow row for every 1830 action on
the scratch stack with the request overhead < 5 ms p95; a missing or crashing `site_engine` never affects a move;
the revenue audit clean or explained per title; the production image built locally.

**C — client** (living-board worktree `~/workspace/18xx-concepts/lb-rust-1830`, branch `rust-1830` from master).
Files: `src/engine-rs/**` (worker, api, menu, pack, steps, pkg loader), `src/engine/words.ts` and `rawmenu.ts` (the
word templates moved out of today's `moves.ts`, which imports them), `src/engine/session.ts` (choice, fallback,
`X-Engine`, applying the stored action, `engine_report`), `src/engine/requests.ts` (`TitleEngine.rust?`),
`src/titles/engine.ts` (passing `rust` through), `src/ui/settings.ts`, `src/account/SettingsPage.tsx`, the header
badge, the hand hiding hand-drawn runs in Rust mode, `build.mjs` (`worker-rs.js`, the pack as a hashed asset),
`server.mjs` (`.wasm`, serving `$RUST1/pkg/current` and a boot `engine.rust` in dev), `scripts/rust/titlepack.cjs`,
`test/rust/**` (goldens, parity incl. acts, play, bench), `docs/RUST_ENGINE.md`. Never: RL_18xx, rl18xx-next.

| Step | Work |
|---|---|
| C0 | Goldens and the title pack from today's master into `$RUST1` (A tests against them from A4) |
| C1 | `worker-rs` over the skeleton package (K1, K2), the switch and fallback (§7.2) |
| C2 | `words.ts` / `rawmenu.ts` on today's `moves.ts` (Opal path unchanged, its suites green), RawMenu → MoveMenu |
| C3 | Posting (K8: `check` + stored action in games with programs), `engine_report` (K9), the Auto tab on `programInfo` |
| C4 | `parity.cjs` (positions, acts both ways: §8.3), `play.cjs` (browser, B's stack and :9301 test users), `bench.cjs` |
| C5 | Merge living-board master into `rust-1830` (round-info, routes-fix may have landed: resolve carefully), suites green |

Acceptance: without Rust nothing changes (`npm run typecheck`, the existing node suites); with Rust the corpus
games open with every step equal to Opal's; live following, undo / redo, every fallback path and the kill switch
work; a test-user game with every move kind posted from Rust and accepted, one with programs, one with a forced
`repriced`; both engines' words identical; the bench's numbers in `docs/RUST_ENGINE.md`.

### 9.4 Order and integration

1. Day 1: A0 publishes the skeleton; B0 and C0/C1 start against it. The training session's X2 (already on its branch,
   96689b7) and X3 land on master whenever they do; A watches `git -C ~/workspace/RL_18xx log master` for them.
2. A1-A4, B1-B4, C2-C4 in parallel; the corpus-level key comparison (A1 + B2) early, so both key implementations are
   right before the live shadow.
3. A5 after X2/X3 (rebase, the small engine commits), A6; the training session reviews and merges A's branch with
   Caleb's OK (we never merge into RL master or push). The site pins A's branch commit (K6).
4. C5, then C's branch merged into living-board master; B: `git subtree pull --prefix=client` of that master,
   `scripts/client-build.sh`, `scripts/engine-rs-pull.sh <A's commit>`, `scripts/engine-build.sh`, B's branch merged
   into `rl18xx-next`, :9301 picks it up (rerun; `ENGINE_SHADOW=1`, `RUST_ENGINE_TITLES=1830`,
   `RUST_ENGINE_DEFAULT=opal`); after §8.6's acceptance, `RUST_ENGINE_DEFAULT=rust` on dev.

---

## 10. Risks and open points

- **Two key implementations** (Ruby and Rust) can disagree by their own bugs: the corpus comparison (§8.2) runs before
  the live shadow, and every field of §5.3 is read from one plain engine attribute on the Ruby side.
- **X2/X3 timing**: until they land, the Rust menu offers an M&H exchange Ruby refuses (the server refuses, the page
  resyncs) and lacks bankruptcy at 2 known positions; the dev default stays Opal until then. (Both closed: X2 on
  master, X3 in the site menu, §11.2.)
- **X1**: no disagreement on this site since the auction lock; Caleb's decision pending, nothing in step 1 depends on it.
- **Download size**: the 1830-only wasm is ~+28% over Opal (412 vs 321 KB brotli, the evaluation). Trimming needs the
  1822/1867 code paths behind features (~40 call sites in shared functions): deferred, not step 1.
- **Run revenue for other titles**: §6 changes behaviour for every title; the audit (B1) gates it.
- **Version skew**: glue and wasm are one pair served by the site (K2); the client checks the API version; the shadow
  records the engine commit per row.
- **Upstream Ruby changes** in `lib/engine` (merged from tobymao or the site's own, like 1b5c534 and c57803b) can
  break parity silently: the corpus run (§8) is repeated before each re-pin of `engine-rs/PINNED` and after each
  `lib/engine` change.

---

## 11. Package A as built (2026-11-26)

Branch `site/engine-rs-1830` on origin/master **96689b7** (X2); after the training session's review (U1, §4.6) rebased
onto **cd97116** (its docs and two `test_value_trainer` fixture fixes: no engine change, no conflicts). The neutrality
test, unchanged, matched at that base before anything was added; it passes at every later commit with `site_rules`
off. X3 is Caleb's open choice: nothing
was added for it, and S/H did not wait for it (§4.3: S2 and H2 touch `operating_step_auto_skips`, which an X3 fix may
touch too).

**Commits** (oldest first): neutrality test; this doc; **P** (pyo3 optional + `scripts/pre-push.hook`); **V**
(visibility); **S0** (the two fields); the spike's site layer ported; **S1-S5** and **H1-H4** (one commit each, H in
one, each `title rule || self.site_rules` / `if self.site_rules`, naming its Ruby method; made right after the port,
on 96689b7, so every later number includes them); then site-only commits (API 2, the shadow key, the CLI, the build,
the log, the menu, tests, tools). Outside `src/site` nothing else changed.

**Speed** (`scripts/bench_leaf_costs.py --positions 32`, 5 interleaved runs each, one core, `nice`; µs per leaf,
median): select+expand 114.4 before (P) / 106.7 after (S0-S5, H); clone a game 54.4 / 50.4; legal-move enumeration
20.1 / 18.4; free a game 24.5 / 22.4. The "after" runs were faster across the board, the Python-only parts too
(adapter encode 269 / 248): machine load, not the change; no slowdown is measurable.

**Artifacts** (K2): `$RUST1/pkg/<commit>/` and `pkg/current` (`engine_rs.js` no-modules glue, `engine_rs_bg.wasm`
1.44 MB / 537 KB gzip, `engine_rs.d.ts`, `BUILD.json`, `site_engine` host build), published by
`scripts/site_engine/publish.sh`; `engine-rs/site/build-site.sh <out>` (K5) builds the same from a `git archive` of
the branch with `PINNED` and no git.

**Corpus** (§8.1): :9301 1,041 games (315 distinct), the production export 19 games (read-only bot API, files 600,
deleted after the checks), the fixtures 14 (Ruby's spec games, RL_18xx's site-format fixtures and test games, the
advisor's RL streams).

| Check | Result |
|---|---|
| Strict replay | :9301 1,041 / 1,041; production 19 / 19; fixtures: the 4 that stop, stop where the site's Ruby stops, with its words |
| Finished games' result vs the site's | 144 / 144 (:9301), 14 / 14 (production) |
| Shadow key v1 vs the reference Ruby key (`scripts/site_engine/ruby/keys_reference.rb`, the site's Ruby in a throwaway container) | 147,037 / 147,037 positions (:9301), 11,767 / 11,767 (production), 4,759 / 4,759 (fixtures) |
| Stored runs vs the engine's own pricing (`last_action`) | 11,643 / 11,643 (:9301), 1,151 / 1,151 (production) |
| Native vs wasm (`replay_digest`: every step's JSON and key, final snapshot and menus) | 1,057 / 1,057 games equal; two native runs equal |
| Spike goldens (living-board b8a07d0, 303 games): snapshots, actions, tracks, tile options, auto routes | 78,809 / 78,809; 78,506 / 78,506; 8,966 / 8,966; 9,773 / 9,773; 4,584 equal, never below |
| StaticInfo's engine part | 303 / 303 (M5 fixed) |
| Menus | 78,741 / 78,745 before the auction-lock facts (left: 2 bankruptcy offers = X3; 2 singletons: D22 listed by Opal at 10@305, D&H's token at 725@391); the auction lock then differs from these pre-lock goldens at 2,930 positions by design: C's new goldens judge it |
| Log lines | 166,087 / 166,119 (99.98%; was 98.55%); left: bankrupt sales, D&H teleport skips, a home-token choice's words, one auction position |
| Act cases (§8.3, K11) | `$RUST1/acts/r1/cases.jsonl`: 315 distinct games, every 10th decision position, tiles every 5th: 76,181 cases, 75,412 accepted, 769 refused (all depot trains the menu lists that the company can't afford) |

**Not done here** (open for A, or for B/C): Rust vs C's new goldens (none in `$RUST1/goldens` yet; `site_replay
--goldens` takes them as they come); generated games (Rust self-play with `site_rules` replayed by Ruby, §8.1 last
row); M8 (tile drawings' location / borders: the pack's, C); the two menu singletons; S2's review note: the site's
menu lists only the player's own companies' trains (Caleb's own-only rule, rl18xx-next CLIENT.md §14), while the
engine processes cross-player purchases (`site_rules`), so the menu needs no cross-player list.

### 11.1 The verification's findings, fixed (2026-11-26)

The step-1 verification (a scratch stack played by the test users; Rust against Opal at every position; the site's
Ruby judging cross-player moves) found these in package A. Each fix names the Ruby method it follows.

| Finding | Fix | Where |
|---|---|---|
| D&H's teleport, a company with trains and no route: the automatic withhold was never recorded (Rust's last run and Spreadsheet missed the OR; the shadow `differs` at every move after) | **H5** (§4.3): the D&H branch of `process_action` records it (`Dividend#skip!`) | `game.rs`, own commit |
| A "this stock round" pass program kept after a stock round ended through an operating round no company operates in (Ruby drops it there) | the site layer reads the rounds a move started from the records (`store_player_info`'s entries) and runs each program's `disable?` per round (`check_programmed_actions` after every `transition_to_next_round!`); one program per player (`ALLOW_MULTIPLE_PROGRAMS` false) | `src/site/game.rs` |
| C&A bought from another player: Rust accepted every case and moved no share | **S6** (§4.2): `after_buy_company` -- refused while a player holds PRR_1, else the buyer takes it free | `rounds/stock.rs` (own commit), `src/site/rules.rs` |
| The log left out an empty operating round (priority deal, heading, the privates' income) | the same records: Ruby's `next_round!` lines for every round a move went through | `src/site/log.rs` |
| restartInfo: a turn opening with M&H's exchange started at the exchange | a private's move (the Exchange step's) starts no turn; the pass's words read BuySellParShares's own actions | `src/site/game.rs`, `log.rs` |

Found on the way, also in `log.rs` (the log steps the verification counted): the opening lines' order (the phase before
the optional rules), no "moves down from $X to $X" when a sale leaves the price in its cell (the spike's fallback;
`log_share_price` writes nothing then), the sold-out rises and the corporations' private income in `Corporation#<=>`
order, "-- The bank has broken --" where `Bank#spend` writes it, B&O's close before the phase, M&H's close on its
exchange unlogged, D&H's lay-track skip once its token is placed or declined, no Buy Trains skip once that step has
acted, a pending discard holding the skips, the home token's "must choose city for home token" after "operates", the
auction's all-pass (minimum bid, or the privates' income), a bankruptcy's sales, end_game's master-mode line.

| Check (after) | Before | After |
|---|---|---|
| Live scratch games (7, 3,908 positions; `site_replay --goldens`, the goldens of living-board's golden.cjs): snapshots | 3,813 | 3,907 (left: `snap.auctioning` at 18@6, which the client derives itself, `steps.ts`) |
| same: log steps / lines | 3,524 / 6,115 | 3,908 / 6,167 (all) |
| :9301 corpus against living-board's new Opal goldens (315 games, 82,083 positions): log steps / lines | 80,426 / 171,542 | 82,082 / 171,590 (left: one `log` action's consent line, which Opal's golden leaves out) |
| Spike goldens (303 games, 78,809 positions): log steps / lines | 77,365 / 166,087 | 78,808 / 166,119 |
| C&A bought between players, judged by the site's Ruby (the verification's `xplayer.rb`, C&A only, 4 per game, 331 distinct games) | 126 of the verification's cases disagreed | 842 cases, all agree: 785 refused by both, 57 accepted by both with equal keys |
| Shadow key v1 against the reference Ruby key | 147,037 / 147,037 (:9301); 4,759 / 4,759 (fixtures) | unchanged |

Left as they were: X3 (bankruptcy not offered at 3 corpus positions and one live one: the training session's; Rust
should not be anyone's default until it lands), X1 (Caleb's), and the menu singletons of §11. S2, H2 (in
`operating_step_auto_skips`, next to X3's buy-train area) and now S6 and H5 were made before X3 landed: re-checked,
and re-applied if needed, when A rebases onto it. (X3 is now closed in the site layer, §11.2: no engine fix is coming,
so nothing waits to be rebased onto.)

### 11.2 The 1830 polish (2026-11-26): X3 in the site menu, the snapshot's `ending` and `auctioning`

Three commits on f6a9fe5, **all in `src/site`** (and two site fixtures): no engine edit outside `src/site`, so §4's
list is unchanged (nothing new for the training session to review there), and the neutrality digests are unchanged
(the gate below).

**X3, voluntary bankruptcy: kept for the site's players** (Caleb's decision, 2026-11-26). What the three positions
were (:9301 8@474 and 12@701: CPR with $1 in Buy Trains; the verification's live game 15@597: NYC with $160): a
company with **no train and no route** (so not obliged to buy one: Game#must_buy_train? reads Ruby's route_info,
whose `route_train_purchase` is missing there), whose president's cash and sellable shares can't reach the cheapest
depot train. Ruby (`Step::Bankrupt`, first in 1830's operating round) lists `bankrupt` for the operating company at
every step and accepts it whenever `can_go_bankrupt?` holds (total_emr_buying_power below `depot.min_depot_price`);
the site's hand on Opal (living-board `manual.ts` `bankruptRaw`) shows it in Buy Trains when the active step's
`must_buy_train?` isn't `false` -- and Ruby answers `nil` without a route, which JavaScript takes as true. So the site
offers bankruptcy to a company with no train facing a depot with trains whenever its president can't reach the
cheapest depot train, route or not. engine-rs's processing already accepted it (its Buy Trains guard is the same
`can_go_bankrupt_emr`); only the site menu lacked it.

| What | Where |
|---|---|
| `bankrupt_offer` (`src/site/menu.rs`): Buy Trains, no crowded company, the company has no train, Ruby's `depot_trains` not empty, `can_go_bankrupt_emr`. The route term (`ruby_must_buy`) is gone from it. The words stay Opal's ("CPR must buy a train, ...": `words.ts`, the same for both engines) | site menu only |
| The bots keep the printed rule: `legal_action_types` and the factored enumeration (shared code) are not changed; at 15@597 they offer no `bankrupt` (the test asserts it) | unchanged |
| Ruby accepts each of the three (the site's Ruby in a throwaway container, `bankrupt` processed at 8@474, 12@701, 15@597: no exception, the game ends `bankrupt`) | checked |

**The snapshot's `ending` and `auctioning`** (`src/site/view.rs`), as living-board's Opal adapter sends them:
- `ending`: Ruby `Game#game_ending_description` read as adapter.ts `endingOf` reads it, `{when, why, text}` ("after OR
  15.3", "bank broken", "Bank Broken : Game Ends at conclusion of OR 15.3"), null until an end is triggered and once
  the game is over; always present (as on Opal). The latched triggers resolved as `BaseGame::should_end_now` resolves
  them; 1830's broken bank is `:full_or` (`OR <turn>.<operating_rounds>` in an operating round, the phase's count
  elsewhere). The `:current_or` and final-phase texts follow base.rb and are unchecked against 1867's Ruby until the
  site layer runs 1867.
- `auctioning`: Ruby `Auctioner#auctioning` (WaterfallAuction#active_auction: the company up for auction while it
  has more than one bid; adapter.ts `auctionedNow`), absent otherwise as on Opal. The client's Rust worker derived it
  itself until now (`steps.ts withAuctioning`, removed in living-board `rust-1830-polish`).

**Commits** (on f6a9fe5): `3446997` site menu: X3; `ce08085` site view: `ending` and `auctioning`; then this doc
(after the rebase onto 995ccef, §11.3: ffea71c, 10aaf36, d5deac4).
Tests: `site::tests::voluntary_bankruptcy_is_offered_where_the_sites_hand_offers_it` (fixture `bankrupt_voluntary`:
the verification's game 15, players renamed: offered at 597 with Opal's facts, `min` 630, `power` 246; absent from
`legal_action_types`; accepted, the game ends; not offered at 598 once NYC holds a train; B&M's must-buy bankruptcy
at 613 offered by both), `the_snapshot_says_when_the_game_ends_as_ruby_does` (fixture `bank_broken`: :9301's bot game
44, renamed; the bank breaks at 794 in OR 15.1 of 3; Opal's `ending` from 794 to 880, null at 793 and at the end,
881), `the_snapshot_names_the_company_up_for_auction` (18@5 none, 6 C&A, 7 none).

| Check (f6a9fe5 -> ce08085) | Before | After |
|---|---|---|
| `site_replay --goldens` (living-board's Opal goldens, `$RUST1/goldens`): the :9301 corpus, 315 games, 82,083 positions: snapshots | 76,821 (auctioning 3,036, ending 2,226 positions) | **82,083** |
| same: menus | 322,143 / 322,150 (2 bankruptcy = X3) | **322,145** / 322,150 (left: 4 M&H exchange IPO share NYC_3/NYC_2, M9; 1 D&H teleport station Opal's hand lacks, O2) |
| The verification's live games (7, 3,908 positions, `$RUST1/V/live/goldens`): snapshots / menus | 3,907 / 15,607 | **3,908 / 15,608** |
| living-board `test/rust/parity.cjs --all` on the new package (329 games incl. the 14 fixtures): `snap` (strict, `ending` included) | 0 / 86,842 (snapKnown 86,130) | **86,130** / 86,842 (the :9301 games 82,083 / 82,083; the rest are the RL fixtures' streams, as before) |
| same, the live games: snapshots / menus / hand | 0 / 3,908 strict; 15,607 / 15,608 | **3,908 / 3,908; 15,608 / 15,608** |
| Strict replay, results, log, tile options, auto routes | unchanged | unchanged |

**Gate** (ce08085, the agreement's item 5; `taskset -c 24-31,56-63 nice -n 19`, `CUDA_VISIBLE_DEVICES=`, the worktree's
own venv): `cargo test --release --no-default-features --features python` 268 passed, 0 failed, 11 ignored
(neutrality included); `--features site` 287 passed, 0 failed, 10 ignored; `cargo check --target
wasm32-unknown-unknown --no-default-features --features wasm` and the default `cargo check` clean; `uv sync`, then
`pytest tests/`: 1,823 passed, 6 skipped, 2 deselected, 0 failed (38 min, on cores shared with other runs).

**Package** (K2): `build-site.sh` at ce08085: `engine_rs_bg.wasm` 1,486,257 bytes (552,818 gzip), `engine_rs.js`
22,806, `site_engine` 5,153,968; in the polish's own scratch (`rust2/polish/pkg/ce0808514096`; `$RUST1/pkg/current`
was left as it was).

### 11.3 Rebased onto origin/master 995ccef (2026-11-27)

For the training session's review and merge, the polish was rebased from f6a9fe5 (step 1, merged) onto RL_18xx
origin/master **995ccef**: f6a9fe5 plus the training session's docs commit (CLAUDE.md, docs/rl_model.md: the 1867
bots play gpg1867_a4 u1200 on both sites); no engine change, no conflicts (the polish touches neither file). Every
commit kept, in order, each patch unchanged (`git patch-id`): §11.2's `3446997` → **ffea71c** (X3), `ce08085` →
**10aaf36** (`ending`, `auctioning`), its doc `7bcc696` → **d5deac4**; then this note. `site/engine-rs-1867` was
rebased onto it in turn (SITE_ENGINE_RS_1867.md §13.3).

**Gate** at d5deac4 (`taskset -c 24-31,56-63 nice -n 19`, `CUDA_VISIBLE_DEVICES=`, the worktree's own venv): `cargo
test --release --no-default-features --features python` 268 passed, 0 failed, 11 ignored (neutrality included);
`--features site` 287 passed, 0 failed, 10 ignored; `cargo check --target wasm32-unknown-unknown --no-default-features
--features wasm` and `--features wasm,only-1830` and the default `cargo check`: no errors (only the shared code's
existing dead-code warnings). The same numbers as at ce08085. The full `pytest tests/` ran on the 1867 head, which
contains this branch (SITE_ENGINE_RS_1867.md §13.3).

