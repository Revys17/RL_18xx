# The site on engine-rs, step 2: 1867 in the browser and in the server's shadow

Status: **design, 2026-11-26** (branch `site/engine-rs-1867`, RL_18xx worktree `.claude/worktrees/site-engine-1867`,
on `site/engine-rs-1830` f6a9fe5; it is rebased onto the polished step-1 head before the build). Approved by Caleb:
his site (18xx.revys.net, `rl18xx-next`) moves from the Ruby/Opal engine to engine-rs, keeping the titles engine-rs
implements; on 2026-11-26 he answered the step-1 questions "1 yes 2 keep for humans 3 yes": the Rust comparison runs
on 18xx.revys.net too (a deploy step of the parent's, not of this design), voluntary bankruptcy stays for human players
(1830's X3; 1867 has no bankruptcy at all), and step 2 starts. **Step 2** (this doc): 1867 runs on engine-rs (wasm) in
the browser behind the same switch as 1830, with Opal as the fallback, and engine-rs runs in shadow beside the
server's Ruby on every 1867 action, on the **dev site only** (:9301 and scratch stacks). Ruby stays authoritative.
Production (:9292) is not touched.

Read with: `docs/SITE_ENGINE_RS.md` (step 1: the site layer `src/site`, WasmGame API 2, RawMenu, `site_rules` S0-S6,
H1-H5, U1, the shadow key v1, `site_engine`, contracts K1-K11, §11 the record), `docs/g1867_remaining_work.md` (engine-rs's
1867: rules fidelity, the legal-move generator, the router, the Ruby-HEAD checks), rl18xx-next
`docs/rl18xx-next/ENGINE_RS.md` and living-board `docs/RUST_ENGINE.md` (step 1 as built), living-board `docs/TITLES.md`
§10.1-§10.3 (1867's hands on Opal). Section numbers here are cited from code (`engine-rs/src/neutrality.rs` cites §7).

---

## 0. In one page

| Question | Decision |
|---|---|
| Where 1867 stands | engine-rs's 1867 already replays every 1867 game the site has: **24 / 24** (:9301's 10, production's 10, the 4 fixtures) to the last action, every finished game's result equal (13 / 13); 9,872 of 9,881 recorded decisions match a move of the legal-move generator's Ruby set exactly and the other 9 by train type and holder; the exact router's run is never below a stored run (1,273 runs: 1,208 equal, 65 higher, 0 lower); Ruby's step auto actions are predicted exactly at every position of every site game, programs aside (§2.2). The work is the site layer, not the rules. |
| The 1867 site layer | Built on what the training session already verified: the **legal-move generator decides** what a live move may be (Ruby's player-time checks: `@game.loading` false), the replay trusts the record as Ruby's `load` does, menus are facts read off the generator's moves, the router is `g1867_auto_routes`. New: `src/site/g1867/*` (act, autos, menu, view, key, log, tiles, details), dispatched by title; 1830's code paths unchanged (its goldens must stay byte-identical). §3 |
| Auto actions | 1867 makes **step** auto actions without any program (G1867 Token's, Merge's and BuyCompanyPreloan's passes; ~6% of moves carry one). The site layer predicts them (three Ruby predicates, §3.3) so a Rust page still acts at once and posts; the server's stored action must equal it (else resync + report). Programs' auto actions stay the server's, as in step 1. |
| site_rules | **No new branch for 1867.** S1-S6 and H1-H5 are title-generic and inert or right for 1867; the minors' automatic payout and the nationalized majors' fresh start are recorded in `src/site` (§5.1). The review note (`site_interplayer_purchase_open` hardcodes 1830, phases 2-4) is fixed in `src/site/rules.rs` with Ruby's phase status table, **not** in the title's `PhaseDef.status` (it feeds the encoder's `PHASE_STATUSES` features: §5.2). Outside `src/site`: only the registry's cfg arm `site-titles` (1830 + 1867, replacing `only-1830`) and any visibility-only edits, each listed. |
| Shadow key | **v2** = v1 unchanged in meaning + fields present only where the title has them: per corporation `loans`, `interest`; `loans_left`, `national`, `trainless`, `merger`, and for every title `auction` (the bids in progress, which v1 never compared). Ruby side in `lib/engine_shadow.rb`, Rust in `src/site/shadow.rs` + `src/site/g1867/key.rs`. §6 |
| Neutrality | Pinned **first**: commit **9e06cad** (24 fixed-seed 1867 walks through the `1867-v1` index API, the router, the encoder every 5th position; 32,050 moves, all 24 games finished; digests equal under `--features site` and `--features python`, in separate processes, and on RL_18xx origin/master cd97116). §7 |
| Client | The same hands as Opal (`MergerHand.tsx`, the minor auction strip, the reduce-tokens cards, nationalize, redeem, the loans' words), now from RawMenu **facts** for 1867's parts too (today they come from Opal already worded): the facts/words split of step 1 extended; the Opal path's hand unchanged word for word. Pack `title-1867.<hash>.json`. §4, §8 |
| Verification | The site corpus is small (24 games, ~12.7k actions), so it is joined by **generated games** (engine-rs walks under the own-only policy, converted to site JSON, judged by the site's Ruby) and, if the parent grants a read-only copy, RL_18xx's 993 human 1867 games. Per position: Rust vs Opal (snapshots, menus of every seat, tiles, routes, log, details) and Rust vs Ruby (key v2); every offered option acted in Rust and judged by Ruby and Opal. §10 |
| Packages | **A** engine (this worktree, `site/engine-rs-1867`), **B** server (new rl18xx-next worktree, `next/rust-1867`), **C** client (new living-board worktree, `rust-1867`); contracts K1-K11 carried over with the deltas of §11.2. |

---

## 1. Step 2 at a glance

```
 browser (living board)                            server (rl18xx-next)                         queue process
 ───────────────────────                           ────────────────────                          ─────────────
 Session ── worker.js (Opal) ◄── fallback ──┐      POST /api/game/<id>/action (Ruby decides)
        └─ worker-rs.js                      │        X-Engine: rust → add_auto_actions: true
            engine_rs_bg.wasm (1830 + 1867)  │        stored action back (with Ruby's auto actions)
            title-1830.json / title-1867.json│        EngineShadow.capture → key v2 row ──────────►  site_engine state
        menus from the generator's moves,    │      ENGINE_SHADOW_TITLES=1830,1867 (dev)              compare v2 keys
        act = generator-checked + step autos │      RUST_ENGINE_TITLES=1830,1867 (dev)                engine_shadow table
        exact router, log, details           │
        fallback report ─────────────────────┴──► POST /api/game/<id>/engine_report
```

What does **not** change: Ruby decides every move; `POST /api/bot/engine` and the bot API keep their contracts; the bots'
1867 backend (RL_18xx's Python importer following site games) is untouched; 1830's behaviour on both engines is
byte-identical (its goldens, keys and act cases are re-run); runs drawn by hand and `auto_routing: false` stay on Opal;
1867 for 2 players and the `grid_market` rule stay on Opal (engine-rs has neither: `unsupported:`); production is not
deployed.

After step 2 (not here): the Programmer (program auto actions) in Rust, Rust authoritative on the server, the 1822
family, retiring Opal and Ruby.

---

## 2. Where 1867 stands in engine-rs

### 2.1 What the training session built and verified (docs/g1867_remaining_work.md)

- **Rules**: 1867 replays the human corpus as Ruby HEAD does: on the 300-game sample per-action state (corporation
  price/cash, player cash, bank, round) shows no Rust-vs-Ruby divergence on any action both engines reach; 203 games
  Ruby replays fully end with equal scores in all 203. The 2026-10 scrape (367 recent games): 317 match, the 19
  rejections are all 2-player games (out of scope).
- **The legal-move generator** (`g1867/legal.rs`, `g1867_legal_moves_with(policy)`): every step × action type is
  generated or listed as replay-only; Ruby's player-time checks are applied (tile legality incl. `StubsAreRestricted`,
  `TILE_UPGRADES_MUST_USE_MAX_EXITS`, `check_track_restrictions!`, two-lay `TILE_LAYS`, costs within full buying power;
  Ruby's graph walk for lay targets, token connectivity and merge candidates; `home_token_locations`; trade-ins in phase
  8 only). Completeness on the 2026-10 scrape: **158,897 of 158,900** human decisions match exactly one move (the 3
  others are out-of-turn post-merger passes, replay-only). Fuzz: 140 walks replayed by Ruby HEAD, 139 clean (1 meets
  Ruby's orphan-train defect, §5.4). The fast generator equals the reference generator move for move on 1,145,266
  corpus comparisons. Ranged moves carry `PriceRange {lo, hi, step}`.
- **Policies**: `CrossPlayerPolicy::RESTRICTED` (the title's: no train from another player's company, no private from
  another player, merger adds only of the initiator's owner's minors; the user's ruling of 2026-10-18) and `RULES`
  (everything Ruby accepts). If the restricted set is empty the generator falls back to the rules' set.
- **Not offered although Ruby accepts them** (T6, Ruby dead ends a player escapes only by undo): a minor bid with no
  home location; a merge add / done / convert from which no major can be named; the pass or buy that leaves the merged
  major's presidency in the treasury; a 5+5E traded for a 2+2.
- **The router** (`g1867/router.rs`, `g1867_auto_routes`): exact (bucketed distances, x2 trains, hex bonuses, the
  capitals + Timmins bonus, Detroit's group) within a 20M-step safety budget the corpus never reaches; Ruby HEAD validates
  and prices every engine run that beats a record (1,676 on the scrape) at the engine's prices.

### 2.2 The site's 1867 games through engine-rs today (design probe, 2026-11-26)

A scratch probe (not committed; `$RUST2/design/probe`) replayed every 1867 game the site has, as the site layer would
(Ruby's undo filter, site user ids as player ids, recorded auto actions after their action, `process_action_internal`),
and at every recorded decision asked the generator and the router. Corpus: :9301 (`GET /api/game/<id>`, every 1867 game
not `new`: 1218-1227), production (the bot API's read-only export: 24-33, files 600, deleted with the scratch), rl18xx-next
`public/fixtures/1867/*` (19792, 21268 twice, nationalization_cash).

| | :9301 (10 games) | production (10) | fixtures (4) |
|---|---|---|---|
| Raw actions / engine actions processed | 2,667 / 2,828 | 7,394 / 6,575 | 2,674 / 2,412 |
| Strict replay to the last action | 10 / 10 | 10 / 10 | 4 / 4 |
| Finished games' result = the site's | 1 / 1 | 8 / 8 | 4 / 4 |
| Recorded decisions matching a move of Ruby's set (`RULES`) | 2,379 / 2,379 | 5,417 / 5,419 | 2,076 / 2,083 |
| ... of the own-only set (`RESTRICTED`) | 2,375 (4: merges of another player's minors, fixture replays) | 5,417 | 2,074 |
| Stored runs vs the router: equal / router higher / router lower | 288 / 0 / 0 | 728 / 1 / 0 | 192 / 64 / 0 |
| Ruby's step auto actions after each move = the prediction of §3.3 | 2,667 / 2,667 | 6,144 / 6,148 (4: programs' passes) | 2,229 / 2,339 (old records; programs) |
| Generator per decision: p50 / p99 / max (native) | 0.36 / 3.5 / 6.1 ms | 0.40 / 6.7 / 20.6 ms | 0.35 / 3.9 / 16.6 ms |
| Router per run: max (native) | 1.7 ms | 17.6 ms | 21.5 ms |

The 9 decisions that matched no move are the probe matcher's, not the generator's: the generator lists one train (and
one discard) per type and holder, the record names another instance of the same type and holder (the importer matches
by type and holder for the same reason). Plain replay of a whole 1867 game: 25-50 ms native.

### 2.3 What step 1's site layer assumes about 1830

`src/site` reads 1830 through hand-written gates: `menu.rs ruby_types` (the 1830 `OperatingStep` lists),
`RUBY_HEX_ORDER`, the G19 city swap applied to every game, the numeric tile order, the step class names inferred from
step labels (`shadow.rs ruby_step_class`), the waterfall auction, the registry trimmed to 1830 (`only-1830`). 1867 runs
through the shared step machine (`steps.rs step_offers`: the title's step list, the blocking step, each step's types)
and the generator, so its site layer reads those instead of re-deriving Ruby's gates.

---

## 3. The 1867 site layer (package A, `src/site/**`)

Layout: `src/site/g1867/{mod, act, autos, menu, view, key, log, tiles, details}.rs` (all `cfg(feature = "site")`),
dispatched by title at the entry points of `SiteGame` / `WasmGame` / `site_engine`. 1830's paths are not edited beyond
the dispatch, and every 1830 suite of step 1 (goldens, keys, acts, native = wasm) is re-run unchanged.

### 3.1 Load and replay

- **`SiteGame::load`** accepts `1867` with 3-6 players (2 players: `unsupported: 1867 for 2 players`, engine-rs has no
  2-player setup) and no optional rule (`grid_market`: `unsupported: optional rule grid_market`). The game is
  `BaseGame::build_titled("1867", site ids, names)` (1867 takes no seed), `site_rules = true`, site records on.
- **Actions** (`json.rs`): `merge` `{corporation}` (add a minor, name the major of a conversion or of a merger: one wire
  type, three meanings, as Ruby), `convert`, `remove_token` `{city, slot}`, `choose` `{choice: "nationalize"}` join
  the step-1 parser (the importer's dicts are the site's JSON); `program_merger_pass` is a program (step 1's
  `note_program`). City references: 1867's preprinted tiles are `"<hex>-<index>"` in Ruby and engine-rs alike (the
  generator's wire form, `title::g1867::preprinted_tile_id`), so **no city swap** (G19's stays 1830's: the swap list
  becomes per title).
- **Tile supply** (`TileSupply`): Ruby's `TILES` order of `g_1867/map.rb` (not a numeric sort: `3, 4, 5, 6, 7, 8, 9,
  14, 15, ... 57, 58, 63, 70, 87, 88, 120, 122, 124, 201, 202, 204, 207, 208, 611, 619, 621-626, 637, 639, 801, ...`),
  as a table in `src/site/g1867/tiles.rs` checked against the pack (C). **Unlimited tiles** (7, 8, 9): Ruby keeps one
  instance in the supply and `Game::Base#add_extra_tile` appends `<name>-<highest index + 1>` when one is laid
  (`update_tile_lists`); the recorded ids confirm it (9-0..9-17, 8-0..8-11 in the corpus). An upgraded-over tile goes
  back at the end, as 1830's.
- **Replay** trusts the record exactly as Ruby's `load` (`@game.loading`): `process_action_internal` per kept action,
  then its recorded `auto_actions`. A chat **message's** auto actions are processed too (Ruby's `process_action` runs
  them after any action, a message included; step 1's `process_next` returns before them: fixed for every title in
  `src/site`; none occurs in the 1867 corpus).
- **Records** (`src/site/records.rs`, what the shadow key's `last_run`, `ports:last_run`, the Spreadsheet read):
  - majors' dividends and the revenue-0 withholds: H1 / H2, as 1830;
  - **a minor's automatic payout** (`G1867::Step::Dividend#skip!`: `payout` when the run paid, else `withhold`) is noted
    by the site layer right after the minor's `run_routes` (the Dividend step runs inside that action), from the run's
    revenue; no engine edit;
  - **a nationalized major** (`G1867::Game#nationalize!` → `reset_corporation`: a fresh corporation in Ruby, its
    `operating_history` empty) has its records cleared when the site layer sees it reset (its par gone while it stays
    in Ruby's corporations); a nationalized minor closes (its records go with it, as a closed 1830 company's);
  - round ends and the game's end: H3 / H4 (title-generic).
- **Ruby's corporation list** (`src/site/g1867/view.rs ruby_corporations`): Ruby's `@corporations` membership *and
  order*, which the snapshot's `corps` array and the key follow: the minors that are not green first (title order); at
  `green_minors_available` (the first 3) `@future_corporations` appended (the majors, then the green minors, title
  order); at `minors_cannot_start` (the first 5) the unowned minors dropped; closed minors dropped; at
  `minors_nationalized` (the first 8) every minor dropped; a nationalized major keeps its place. The national (CN) is
  never in it (Ruby deletes it in `setup`).
- **Turn starts** (`ports:restart`, Ruby's `next_turn!`): the round's entity changes, as step 1 (`turn_key`), with the
  merger round's entities (minors, the major replacing its minor, the players while shares are dealt) and the trainless
  majors' freeze (the round stays on its entity while `trainless_major` is not empty).

### 3.2 Act and check: the generator decides

Ruby checks a live move with `@game.loading` false (tile legality, rotations, home locations, merge candidates,
trade-in timing, ...); engine-rs's replay handler skips those checks like Ruby's `load`, and the generator applies them
(§2.1). So in 1867:

- **`act(action)` / `check(action)`**: the move must match a move of `g1867_legal_moves_with(RULES)` (Ruby's whole set:
  the engine does not enforce the site's own-only rule, the menus do), on the importer's outcome key (type, entity;
  company / corporation; a share bundle's corporation, holder, percent and president's certificate; hex, tile name and
  rotation; city by hex and index; train by type and holder; kind; choice) and, for a ranged move, a price on the range
  **and its step** (bids on the $5 grid, `MUST_BID_INCREMENT_MULTIPLE`; a private for $1..face; another company's train
  for $1..cash). Then it is processed (`process_action_internal`) with its step auto actions (§3.3), and `posted` is the
  site action as Ruby would store it: the tile instance Ruby's supply names, the share ids, the auto actions.
  No match: `refused:` with words naming the rule ("Not a legal lay: ...", "Bid must be a multiple of $5", ...).
- `run_routes` keeps step 1's rule: `act` accepts the router's run only (§3.7); `append` takes any stored run (Ruby
  priced it).
- **Ruby accepts, Rust refuses**: the T6 dead ends (§2.1) and nothing else; they are never offered by either hand in
  the corpus (checked in §10.3, listed if met).
- Cost: one generator call per act (p50 0.4 ms, max ~21 ms native on the corpus).

### 3.3 Auto actions: 1867's step auto actions in Rust (`src/site/g1867/autos.rs`)

Ruby, after every move, runs `round.auto_actions` (the active step's `auto_actions(current_entity)`) until it returns
none; Opal pages post them, the server validates them (`validate_auto_actions`), and a Rust page's server adds them
(`add_auto_actions`, K8). 1867 has three **step** auto actions that need no program:

| Step | Ruby | Rust predicate (on the position after the move, repeated while it holds) |
|---|---|---|
| `G1867::Step::Token#auto_actions` | `pass` unless `can_really_place_token?` (a token left, `graph.can_token?`, some tokenable city at the distance price within the buying power) | the blocking step is Token and the generator offers no `place_token` |
| `G1867::Step::Merge#auto_actions` | `pass` when not converting / naming a major / merging, the minor can't convert and `mergeable_candidates` is empty (else the programs') | the blocking step is Merge, no conversion or merge under way, `!merger_can_convert` and `merger_mergeable_candidates` empty |
| `G1867::Step::BuyCompanyPreloan#auto_actions` | `pass` when the company has no loans | the blocking step is BuyCompanyPreloan and the company's `loans == 0` |

(The blocking step is the last of `step_offers`' offers to the round's entity: the non-blocking steps open beside it
come first.) The probe's prediction equals the recorded step auto actions after **every** move of every :9301 and
production game (8,811 positions; the 4 differences are programs' passes by players), and the fixtures' differences
are old records made before the client added these passes, and programs.

- **`act`** processes the move, then the predicted auto actions, and returns them in `posted.auto_actions`. A Rust page
  in a game where no player has a program applies the move at once and posts it (as 1830); the stored action in the
  answer must equal what it applied, auto actions included (`created_at`, `user`, `id` aside), else the page reads the
  game again and reports (`engine_report refused`).
- **Programs** (`program_merger_pass`, `program_buy_shares`, `program_share_pass`): as step 1, a game where any player
  has one uses `check` → post → append the stored action (the server's Programmer adds the programs' moves).
- **Hotseat** 1867 games run on Rust without programs; with programs they go back to Opal (as 1830).

### 3.4 View: Snapshot and StaticInfo for 1867

Shapes are living-board's `types.ts`; values equal Opal's adapter at every position (§10.2). 1867's parts:

| Field | Source (Ruby, via Opal's adapter) | Rust |
|---|---|---|
| `corps[]` (order, membership) | `game.corporations` (`operators`) | `ruby_corporations` (§3.1) |
| `corps[].loans`, `loanMax`, `interest` | `corp.loans.size`, `maximum_loans` (2 minor, 5 major), `interest_owed` ($5 × the OR snapshot) | `corporations[i].loans`, the title's max, the OR's `interest_snapshot` |
| `corps[].trainLimit`, `trainLimits` | `train_limit(corp)`, the phase's per-kind hash | `corp_train_limit`, `PhaseDef.minor_train_limit` |
| a minor | one 100% certificate, `certs[sym] = 100`, `presidentPercent` 100, `shares` 1, `kind: "minor"` | from the corp type |
| `round` | "Auction" / "Stock" / "Operating" / "Merger Round" (short `MR`, `round_num`, `total_rounds`) | `Round::*` |
| `stepLabel` | the step's `description` ("Bid on Companies", "Choose if Major is nationalized", "Convert or Merge Minor Corporation", "Merge Minor Corporations", "Choose Major Corporation", "Buy Shares Post Merge", "Choose tokens to remove", "Redeem Shares", ...) | per `StepKind` and the merger state |
| `auction`, `bids`, `auctioning` | the opening `SingleItemAuction` (one company up, its `min_bid` with the dutch discount), the stock round's minor auction (`auctioning_corporation`, its bids) | `Round::Auction` single-item state, `StockState`'s corporation auction |
| `ending` | `game_ending_description` (bank `:current_or`; final phase `:one_more_full_or_set`: "Game Ends at conclusion of OR x.3") | from `game_end_triggered`, `final_turn`, `bank_broken`; the polish's `ending` for 1830 is reused if it lands first |
| `opOrder` / `waiting` | the operating round's entities; otherwise `operating_order` (floated minors, then majors, each by `Corporation#<=>`) | the same |
| `depot` | upcoming / discarded; 8, 2+2 and 5+5E unlimited (one at a time); the export at each OR's end in phases 4-7 | `depot` |
| `hexes[].tok` | the national's tokens (CN: F16's from the start, the 639's on Montreal, nationalized companies' stations), the neutral tokens on D2 and L12 until phase 3, Montreal's reservation (L12) | tokens with owner `CN`, neutral slots |
| `companies` | the hidden "3" (bank, then closed at phase 3), privates nationalized to CN at the first 6 | owners |

StaticInfo's engine part (game id, status, players, description) as 1830; the rest is the pack (§4).

### 3.5 RawMenu facts, decision by decision

RawMenu stays facts plus complete site actions (step 1 §3.3); `types` is Ruby's `step.actions(entity)` in Ruby's order
(read per step class: e.g. `BuySellParSharesViaBid#actions`: `%w[bid pass]` while a minor is auctioned, else the base
list then `bid`, then `pass`; `Merge`: `merge` (+ `convert`) + `pass`, or `merge` alone while naming a major;
`MajorTrainless`: `pass choose`; `PostMergerShares`: `buy_shares pass`). The options are the generator's `RULES` moves
filtered by the **own-only rule** (the menus never offer another player's train, private or minor; the minors left out
of a merger are named). The parts that today come from Opal already worded become facts (living-board's
`MoveMenu` types keep their shapes; C's `words.ts` words them):

| Decision (Ruby step) | Facts (RawMenu) | From |
|---|---|---|
| Opening auction (`SingleItemAuction`: lowest-value private up; all decline → dutch, $5 off; a dutch bid buys) | `auction` [{sym, name, value, minBid, maxBid, buy: dutch}], `bidStep` {5, multiple}, `auctioning` {sym, bidders}, `pass` | generator `bid` ranges (dutch: the asking price only) |
| Start a minor (`BuySellParShares` via bid: `can_bid_any?`) | `auctionStart: true`, `auction` lots = the minors it may start, each with `par` {prices, cap 135} (the par a winning bid gives: the highest minor par ≤ min(bid/2, 135)); min bid $100 | generator `CorporationBid` moves |
| A minor's auction under way | `auction` [the minor], `auctioning`, `pass` (leaves it) | the stock state's auction |
| Start a major (phase ≥ 4: `par`) | `par` [{corp, prices (`par_2`/`par` cells with price × 2 ≤ cash), floatPercent, presPercent 20, sharePercent 10}] | generator `Par` moves |
| Home station (`HomeToken`, after a par or a won minor) | `tokens` [{hex, city, cities, price 0, action}] (`home_token_locations`: a minor any city with no token; a major any hex with no normal token, only the unconnected ones while there are any) | generator `place_token` (HomeToken) |
| Stock round (`buy_shares`, `sell_shares`, `pass`) | `buy`, `sell`, `turn`, `blocks` as 1830 (treasury shares at market price, `SELL_AFTER :operate`, `:down_block_pres`, `SELL_BUY_ORDER :sell_buy`) | step 1's stock facts over the generator's bundles |
| Keep or nationalize a trainless major (`MajorTrainless`, any round) | `nationalize` {corp, action `choose nationalize`}; `pass` with `passKind: "trainless_keep"` | generator |
| Buy a private (`BuyCompany`, phase 3-5; `BuyCompanyPreloan`) | `companyBuy` (the president's own privates, $1..face) | generator, own-only |
| Redeem (`RedeemShares`) | `redeem` [{percent, price, action}] (market shares the treasury can afford) | generator `buy_shares` by the corporation |
| Track (`G1867::Step::Track`, `AutomaticLoan`) | `tileHexes` (hexes with a lay the generator offers), `tileDear` [{hex, cost, terrain}] (hexes with a legal lay the company can't pay even with its loans: the generator on a copy whose treasury is topped up, minus the real set), `tilePower` (cash + loans it may still take), `autoLoans` {corp, loans, max, per 45, interest} | generator lays (two lays, the second $20 and not on the first's hex, no second upgrade), stubs, max exits, the restriction check, reachability |
| Station (`G1867::Step::Token`, priced by distance) | `tokens` [{hex, city, cities, price = the token's price ($40) × the hex distance from the nearest own station (`adjust_token_price_ability!`)}], cash only | generator `place_token` |
| Run (`Route`) | `routes {pending}`; the run itself from `autoRoute` (§3.7) | router |
| Dividend (majors; minors have none) | `dividend` [payout, half, withhold] with previews | generator + step 1's preview |
| Trains (`G1867::Step::BuyTrain`) | `trains` (depot; `loans` {n, per} when it must buy and takes loans), `trainBuy` {mustBuy (Ruby's: trainless, a depot train affordable with every loan), buyingPower (full when it must), others (own only), exchanges (phase 8 trade-ins: 8, 2+2, 5+5E at the discount table)} | generator `BuyTrain` moves (own-only), `try_take_loan` rules |
| End the turn without a train | `pass` with `passNationalizes: <corp>` (G1867 `BuyTrain#pass!` nationalizes a trainless company: the hand asks first) | trainless and floated |
| Discard (`DiscardTrain`) | `discard` | generator |
| Merger round, a minor's turn (`Merge`) | `merger` {minor, phase start, convert? {action, par}, partners [{sym, price, action}] (own minors only), othersLeftOut [{sym, owner}], merging []}; `pass` with `passKind: "merger_stay"` | generator `Convert` / `Merge` (RULES and RESTRICTED: the difference is `othersLeftOut`) |
| ... adding minors | `merger` {phase adding, merging, partners, othersLeftOut}; `pass` with `passKind: "merger_done"` ("Done Adding") | the merger state |
| ... picking the major | `merger` {phase major, mode convert / merge, majors [{sym, action}], par (`merged_par_price` or the convert price)} | generator `Merge` naming a major |
| Stations over the limit (`ReduceTokens`) | `reduceTokens` {corp, issues: [`limit`] / [`sameHex`] (Ruby's `help`), options [{hex, city, slot, owner, action}]} | generator `RemoveToken` |
| Shares after a merger (`PostMergerShares`) | `buy` (the new major's 10% treasury shares at par, the dealing order), `pass` | generator |
| Loans | none to decide: taken automatically for track (`Track` includes `AutomaticLoan`) and for a depot train the company must buy (`BuyTrain#try_take_loan`); stations are paid in cash; interest and forced repayment at `LoanOperations` (no action); facts `autoLoans`, `tilePower`, `trains[].loans` | engine |

`unsupported` is empty for 1867 (any type left over is a bug, reported as in step 1). `programs` lists Ruby's
`available_programmed_actions` (`program_merger_pass`, `program_buy_shares`, `program_share_pass`).

**1867's facts as built** (A, `src/site/g1867/menu.rs`; the shared parts keep step 1's shapes and Opal's values;
these replace Opal's already-worded parts, which C words in `words.ts`):

| Key | Shape |
|---|---|
| `merger` | `{minor, phase: "start"\|"adding"\|"major", merging: [sym], partners: [{sym, price, action}], othersLeftOut: [{sym, owner, ownerName}], majors: [{sym, action}] (phase major), mode?: "convert"\|"merge" (phase major), par? (phase major: the conversion's or Merge#merged_par_price snapped to a par / par_2 cell), convert?: {par, action} (phase start, when `convert` is in types)}` |
| `reduceTokens` | `{corp, issues: ["limit"] or ["limit", "sameHex"], options: [{hex, city (Ruby's city id), slot, owner (the token's company), action}]}` in Ruby's hex order |
| `redeem` | `[{corp, percent, price, action}]` (RedeemShares) |
| `nationalize` | `{corp, verb: "nationalize", action}` (MajorTrainless) |
| `passKind` | `"merger_stay"` (Merge, no merger under way), `"merger_done"` (Done Adding), `"trainless_keep"` (MajorTrainless) |
| `passNationalizes` | the company's sym when ending BuyTrain without a train nationalizes it (trainless and floated) |
| `autoLoans` | `{corp, loans, max, per, interest: 5}` for every company actor; `per` 45 where the step counts the loans (Track; BuyTrain when it must buy), else 50 |
| `tilePower` | cash + 45 × the loans still to take (Track) |
| `trains[].loans` | `{n, per: 45}` for a depot train a must-buy company pays with loans |
| `auctionStart` / `auction[].par` | `true` when a stock-round turn may put minors up; each minor lot `{sym, name, value: 0, target: "corporation", minBid, maxBid, buy: false, par: {prices (the minors' par cells, high to low), cap: 135}}` |

### 3.6 Tile options

`tileOptions(hex, via?)` for 1867: the generator's lays on that hex (every tile, every rotation it takes), each
`{tile (Ruby's next instance), name, color, rotation, spec, cost, terrain, fee?, action}` with `cost` and `terrain` as
Opal computes them (the hex's terrain upgrade cost + the lay's own fee: the second lay's $20) so both engines show the
same numbers. Ruby also charges a water border ($80, `remove_border_calculate_cost!`) when a lay completes a crossing:
the generator already refuses lays the company can't pay with it; Opal's hand does not count it, so where it matters
Rust offers fewer options (classified "Opal offers, Ruby refuses", as step 1's O1). Order: Ruby's tile supply order,
then rotation.

### 3.7 The exact router

`autoRoute()` = `g1867_auto_routes(corp)`: the routes as site JSON (train ids, Ruby's `hexes`, `connections`, `nodes`),
each route's revenue and the total, and the run tracks for the board. `act` accepts exactly that run; the server strips
the revenues and Ruby re-prices it (step 1 §6; 1867 is in `RunRevenue::VERIFIED`). On the site corpus the router is never
below a stored run (1,273 runs, §2.2); time native ≤ 21 ms (the RL corpus' worst 5+5E walks ~0.6 s; wasm ~2-3×,
inside the worker).

### 3.8 Log text (`src/site/g1867/log.rs`)

The lines Ruby writes, in Ruby's words and order, beside step 1's shared ones (shares, tiles, tokens, runs, dividends,
trains, phases, rounds). 1867's own, by the Ruby method that writes them: the auction (`SingleItemAuction`: "X is up for
auction, minimum bid is $N", "Nobody bid on X, reducing price to $N", "Nobody bought X, ...", bids, "wins the auction for
X with a bid of $N"); the minors' auctions (`BuySellParSharesViaBid`: "auctions X for $N", "bids $N for X";
`G1867::Step::BuySellParShares#win_bid`: "wins bid on X for $N", the par lines); loans (`take_loan`: "X takes a loan and
receives $45", `repay_loan`: "X pays off a loan for $50", `InterestOnLoans#pay_interest!`: "X pays $N interest for K
loans", `interest_unpaid!`: "owes ... but only has ..."); nationalization (`nationalize!`: "X is nationalized", the price
lines, "X settles with shareholders $N = $P (...)", the CN's tokens; `MajorTrainless`: "X declines to nationalize");
the merger round ("-- Merger Round T.R (of N) --", "Adding X to merge of ...", "Finish merge of ...", "X converts into Y
receiving ...", "X merges into Y receiving ...", "Y has tokens (...)", "Y will be above token limit and must decide which
tokens to remove", "Y removes token from ...", the post-merger shares); events (`green_minors_available`,
`majors_can_ipo`, `minors_cannot_start`, `minors_nationalized`, `nationalize_companies`: "-- Event: Private companies are
nationalized --", "X nationalized from P for $N", `trainless_nationalization`), the exports (`Depot#export!`: "-- Event:
A N train exports --"), the 639's CN token (`place_639_token`), the skips Ruby logs (`log_skip` for majors only in
Token and RedeemShares), and the end ("-- Loans are "paid off" by moving share price left one step per loan --").
Target as step 1: every step's beats' shapes, ≥ 99.5% of lines against Opal's goldens, each rest listed.

### 3.9 Details, ports and the rest

`details` (Info, Tiles, Spreadsheet: phase, tiles left, player values after every round incl. loans' price drop at the
end, `player_value` with loans), `restartInfo`, `lastRun`, `meta`, `programInfo` (the three program types; `mine`;
`buyable`), `shadowKey` (v2, §6), `legalActions` (the generator's moves, own-only), `at` (checkpoints). The players'
liquidity (Spreadsheet) follows the 1830 polish if it lands there; else both titles get it here.

### 3.10 The wasm package

One package for both titles: the registry's cfg arm `site-titles` (1830 + 1867) replaces `only-1830` (build-site.sh,
BUILD.json `titles`, `features`); `version()` = `{api: 2, key: 2, titles: ["1830", "1867"]}`. The 1867 code is in the
1830-only wasm already (shared code, title-gated); registering 1867 adds its data: **+27 KB raw, +10 KB gzip**
(1,482,792 → 1,510,281 bytes after wasm-opt; 551,116 → 561,633 gzip -9; measured on f6a9fe5). The site layer's 1867
code adds to that (to be measured, §13).

---

## 4. The 1867 title pack (package C, contract K10)

`scripts/rust/titlepack.cjs --title 1867` (the step-1 script with the title as a parameter) → `src/engine-rs/packs/
title-1867.json`, built as `title-1867.<sha12>.json` (+ .br, .gz). From the site's Opal bundles (`g_1867.js`), what
engine-rs doesn't carry:

- StaticInfo without the game's own fields, by player count (3-6: certificate limit 21/16/13/11, starting cash
  420/315/252/210): the corporations (8 majors, 16 minors with `kind: "minor"`, `presidentPercent` 100, `shares` 1; CN
  `kind: "national"`), their colours and logos; the companies (the hidden "3" `hidden: true`, values, revenues, the
  `hex_bonus` descriptions); the one-row market (`marketKind: "1d"`, the cells' types: `p` par, `x` minor par, `z` major
  par, `C` convert range, `m` minor's cap) and its zones with `MARKET_TEXT`; the flat layout's hexes and locations;
  train types (2-8, 2+2 ×2, 5+5E ×2; 8 and the doubles unlimited, `available_on` 8; distances as Opal words them);
  phases with per-kind limits and status words (`export_train`'s `STATUS_TEXT`).
- Every tile drawing (preprinted hexes with stubs G15, L10, K13, the water and impassable borders, the Y/O labels and
  J12's future O, Montreal's 3-city and Toronto's 2-city tiles, 1867's custom tiles), the Tiles drawer's list in Ruby's
  supply order with counts ("unlimited").
- The Info drawer's words (`EVENTS_TEXT`: green minors, majors can IPO, minors can't start, minors nationalized,
  privates nationalized, trainless nationalization, train trade-in), the trains' events per type.
- No optional-rule variants (none supported). Deterministic; re-made whenever the site's Opal changes 1867's data
  (parity's `static` layer says so).

A's test compares the pack's facts the engine also has (tile counts and order, phases, trains, market prices and
types, corporations and companies) with engine-rs's title data and the site layer's tables.

---

## 5. site_rules and every engine edit outside `src/site`

**For the training session** (the parent relays this section). Nothing here changes `action_index.rs`'s layout, the
encoder, `decode`, `POLICY_SIZE` 26537, `src/rl` layouts or g1867's policy layout; the 1867 neutrality walks (§7) and
1830's must pass at every commit with `site_rules` off.

### 5.1 No new `site_rules` branch for 1867

| Candidate | Decision |
|---|---|
| S1/S2 (train purchases from other players' companies, the Buy Trains step staying open) | Title-generic `title rule \|\| site_rules`: 1867's title rule already equals Ruby's (`Depot#other_trains` with `ALLOW_TRAIN_BUY_FROM_OTHER_PLAYERS`); no change. |
| S3/S4 (player-to-player privates) | 1867's phases have no `can_buy_companies_from_other_players`: never open (after §5.2's fix, by Ruby's data rather than by title name). |
| S5/S6 (`buy_company` in a stock round) | 1867 has none. |
| H1/H2 (operating history) | Majors' dividends (`or_process_dividend`) and the revenue-0 withholds (`operating_step_auto_skips`) are the shared paths 1867 takes: right as they are. |
| The minors' automatic payout (`G1867::Step::Dividend#skip!` with revenue > 0: `operating_step_auto_skips`' minor arm) | Recorded in `src/site` after the minor's `run_routes` (§3.1), **no hook**. (If the site layer can't see a case, the fallback is one hook `H6` there, `if self.site_rules { self.site_note_operated(corp, revenue, "payout", true) }`, its own commit, listed here first.) |
| A nationalized major's records (`reset_corporation`) | Cleared in `src/site` (§3.1), no hook. |
| Merger round, auctions, loans, nationalization, redemption | Processed as Ruby by the shared engine; nothing differs between the site and self-play. |
| Mergers with another player's minors | Ruby allows them; the engine processes them (an Opal page or a bot could post one); the site's menus never offer them (own-only). No branch: the engine's `RULES` set already equals Ruby's. |
| Ruby's dead ends and defects (§5.4) | Not emulated. |

### 5.2 `site_interplayer_purchase_open` reads Ruby's phase status (the review note)

Today (`src/site/rules.rs`): `if self.title != "1830" || bought || self.turn == 1 { return false }` and phases
`"2" | "3" | "4"`. Ruby: `Engine::Step::BuySellParShares#purchasable_companies` requires the phase status
`can_buy_companies_from_other_players` (1830's `PHASES`: phase 2 `['can_buy_companies_from_other_players']`, 3 and 4
`%w[can_buy_companies can_buy_companies_from_other_players]`); `G1830::Step::BuySellParShares` adds `@game.turn == 1`.

engine-rs's `PhaseDef.status` is **not** the place for it: the token encoder reads every phase status into its phase
features (`rl/encode` `PHASE_STATUSES`; an unknown flag becomes `st_other`, and the vocabulary test lists it), so adding
Ruby's flag to 1830's title data would change every 1830 encoding and the neutrality digests. Instead, in
`src/site/rules.rs`:

```rust
/// Ruby's phase status flags the RL title data leaves out (only site mode reads them): (title, phase, flags), from
/// the site's Ruby PHASES (g_1830/game.rb:104-125). 1867 has none.
const RUBY_SITE_PHASE_STATUS: &[(&str, &str, &[&str])] = &[
    ("1830", "2", &["can_buy_companies_from_other_players"]),
    ("1830", "3", &["can_buy_companies_from_other_players"]),
    ("1830", "4", &["can_buy_companies_from_other_players"]),
];
pub(crate) fn ruby_phase_status(&self) -> impl Iterator<Item = &'static str>   // PhaseDef.status ∪ the table
pub fn site_interplayer_purchase_open(&self, player_id: u32, bought: bool) -> bool {
    // Engine::Step::BuySellParShares#purchasable_companies: not bought, cash > 0, the phase's status has
    // can_buy_companies_from_other_players, another player holds a private that may change hands;
    // G1830::Step::BuySellParShares's own `@game.turn == 1` refusal: a second small table of the titles whose step
    // adds it (["1830"]), beside the first
}
```

A test asserts the table plus `PhaseDef.status` equals the pack's phases (Ruby's `PHASES` through Opal) for 1830 and
1867. No change outside `src/site`; S3/S4 call sites unchanged. (The alternative, a `GameTitle::site_phase_status`
method, would touch `title/mod.rs` and `title/g1830.rs`; the training session may prefer it: then it is one commit,
data only, listed here.)

### 5.3 Edits outside `src/site` (each its own commit, listed function by function)

**As built** (2026-11-27; each its own commit on `site/engine-rs-1867`, none touches a body the RL path runs
differently on native):

| # | Commit | File, item | Edit |
|---|---|---|---|
| P2 | 1879856 | `Cargo.toml` (feature `site-titles` in place of `only-1830`), `title/mod.rs` `all_titles` (its cfg arm registers `[G1830, G1867]`) | cfg only: without the feature the registry is unchanged. (The same commit renames the feature in the site's own `site/build-site.sh` and `src/site/cli.rs`.) |
| V2 | 50678a6 | `steps.rs`: `BaseGame::blocking_step_entity`, `BaseGame::step_actions` | `fn` → `pub(crate) fn`, bodies unchanged (the site layer reads Ruby's active step, its entity and `step.actions(entity)` for 1867's key and menus) |
| P3 | 105f93f | new `src/clock.rs`; `lib.rs` (`pub mod clock;`); `g1867/legal.rs` (`stats::start`, `stats::stop`, `g1867_candidates`, `g1867_validate`, the lay search's timings) and `g1867/router.rs` (`g1867_route_search`'s walk / combine / finish timings): `std::time::Instant` → `crate::clock::Instant` | `std::time::Instant::now` panics on wasm32-unknown-unknown, and the generator and the router read it unconditionally for their diagnostics counters, so the wasm aborted at the first 1867 menu. On native `crate::clock::Instant` **is** `std::time::Instant` (a re-export: same type, same readings); on wasm32 a stand-in reading zero. The readings feed only `stats` and `AutoRoutes.micros`. |

V2's other candidates (`merger_tokens_above_limits`, `merger_token_hex_count`, `g1867_wire_tile_id`,
`token_slot_for`) were not needed. No `site_rules` branch was added. Both neutrality pins pass at every commit.

### 5.4 Known Ruby / Rust differences left as they are

From docs/g1867_remaining_work.md ("Ruby HEAD dead ends and defects the engine goes past"); none occurs in the site
corpus. If one occurs live, the shadow reports it (`differs` or `refused`, once per game) and the choice is Caleb's
(the site's Ruby rules are not changed here, and engine-rs is not made to copy a Ruby defect):

- **Orphan trains**: a major nationalized while it owns trains (unpaid interest) keeps them in Ruby's old object, and
  `Depot#other_trains` offers them to everyone; engine-rs drops them (the rulebook). The own-only hand never offers
  them (no president).
- **Stale operator** after a MajorTrainless nationalization with more majors queued, and **stale `@corporations`**
  after `event_minors_nationalized!` resets a major: Ruby can't match the next action; engine-rs ends the turn.
- **SingleItemAuction with nobody able to bid**: Ruby raises; engine-rs gives the lot to the first player at $0.
- The replay pricer counts a 5+5E stop on Toronto/Montreal as tokened if any city of the hex is (Ruby: per city): the
  shadow's `last_action` would differ there.
- Ruby's stale `@crowded_corps` memo letting a closed minor record a discard after the first 8 (one old 18xx.games
  game).

---

## 6. The shadow key v2 (contract K4)

v2 = every v1 field with its v1 meaning (1830's v1 part unchanged), plus the fields below, **present only where the
title has them** (absent, not null, otherwise; so a 1830 key gains only `auction`). Hash keys sorted, integers as
integers, ids as in v1. `v: 2`; rows whose versions differ are `skipped` (B and A deploy together: the pin).

| Field | Ruby (B, `lib/engine_shadow.rb`) | Rust (A, `src/site/shadow.rs` + `g1867/key.rs`) |
|---|---|---|
| `corps[sym].loans` | `corp.loans.size` where `engine.maximum_loans(corp) > 0` | `corporations[i].loans` (titles with loans) |
| `corps[sym].interest` | in an operating round: `engine.loans_due_interest(corp)` (`@interest`, the OR snapshot); else absent | `OperatingState.interest_snapshot` |
| `corps` membership (v1: `engine.corporations`) | unchanged: 1867's minors and majors as Ruby holds them (future ones absent, nationalized majors reset: `price` null) | `ruby_corporations` (§3.1) |
| `loans_left` | `engine.loans.size` (titles with loans) | `loans_remaining` |
| `national` | `{tokens: national.tokens.select(&:used).map {"hex-city"}, companies: national.companies ids sorted, reservations: @national_reservations}` (titles with a national) | the CN's tokens, its companies, `national_reservations` |
| `trainless` | `engine.trainless_major.map(&:id)` (titles that have it) | `trainless_major` |
| `merger` (a merger round only) | `{converted: round.converted&.id, merge_type: round.merge_type&.to_s, converts: round.converts ids, dealing: round.share_dealing_players ids, multiple: round.share_dealing_multiple ids, merging: Merge step's @merging ids \|\| [], converting: @converting ? true : false, merge_major: @merge_major ? true : false, removing: round.corporations_removing_tokens&.first&.id}` | `MergerState` (+ `converts` kept by the site layer) |
| `auction` (every title; absent when no bid is open) | `{lot: the company / corporation being auctioned or null, bids: [[sym, player id, price], ...] in Ruby's order, dutch: true where the step has @dutch_mode}` from the active step (`bids`, `auctioning`, 1867's `auctioning_corporation`) | the auction states; 1830's waterfall bids |

`step` for 1867 is the Ruby class name per `StepKind` (`SingleItemAuction`, `BuySellParShares`, `HomeToken`,
`MajorTrainless`, `DiscardTrain`, `BuyCompany`, `RedeemShares`, `Track`, `Token`, `Route`, `Dividend`,
`BuyCompanyPreloan`, `BuyTrain`, `Merge`, `PostMergerShares`, `ReduceTokens`), `actions` Ruby's `step.actions(entity)`
(§3.5), `round` Ruby's `round.name` ("Merger Round" for 1867's). `last_action` for a run: as v1 (1867's runs have no
`extra_revenue` or `subsidy`: 0).

Order of work: both sides on the corpus first (§10.2: B's `ruby_keys.rb` + A's `site_engine keys`, 1830 and 1867), then
the live shadow (`ENGINE_SHADOW_TITLES=1830,1867` on :9301).

**v2 as built** (A, 2026-11-27; the reference Ruby twin is `scripts/site_engine/ruby/keys_reference.rb`, B's
`lib/engine_shadow.rb` must do the same):
- `v` is 2 for every title. `loans` (and `loans_left`) where any corporation has `maximum_loans > 0`; base
  `Game#maximum_loans` is **private** (`send`). `interest` only in an operating round, `loans_due_interest(corp)`
  (a corporation reset by nationalization is a new object: 0).
- `par` is Ruby's `par_price`, which is the **share price** for a company `always_market_price` (all of 1867's).
- `national.reservations` is a **copy** of `@national_reservations` (a live array: serialising it later shows its
  final state at every position).
- `merger.dealing` / `multiple`: the round's lists **while `converted` is set**, else `[]` (Ruby keeps the last ones
  after the dealing; v2 does not compare them then). `converts` and `merge_type` as Ruby's round state (the site layer
  records them from the `merge` naming a major).
- `auction.bids` sorted by (item, price, player); `lot` is the step's `@auctioning` (1830: the private with more than
  one bid; 1867: the lot up, or the minor being auctioned); `dutch` only for 1867's `SingleItemAuction`; `auction` is
  absent unless the active step is an auction with a lot or a bid (1867's HomeToken pending: absent).
- 1867's `step` / `entity` / `actions`: G1867::Step::HomeToken's `active_entities` is the round's entity (in a stock
  round the player who started the company), with `actions` `[]`; Track lists `lay_tile, pass` and BuyCompany /
  BuyCompanyPreloan `buy_company, pass` whenever they are the active step; BuySellParShares lists the base actions,
  `pass`, then `bid` (`bid, pass` when the base list is empty or during an auction).
- `game_end_reason` for 1867: the latched trigger first in GAME_END_TIMING_PRIORITY (`final_phase` over `bank`).

---

## 7. Training neutrality for 1867 (pinned first: commit 9e06cad)

`engine-rs/src/neutrality.rs`, 1867 part (`training_neutrality_1867_walks`, `training_neutrality_1867_walk_is_deterministic`),
the first commit of `site/engine-rs-1867`, on `site/engine-rs-1830` f6a9fe5 with nothing of 1867 changed:

- 24 fixed-seed walks: 3-6 players; the title's cross-player policy (self-play's) twice for each walk under Ruby's set
  (`RULES`, the importer's matching set); busy (80% non-pass) and uniform picks; up to 3,000 moves.
- Driven as the PG workers drive 1867: `rl_indexed_moves(policy)` (the generator, the `1867-v1` index, ascending) → a
  price on the chosen move's range and step → `rl_apply_index(idx, price, policy)` (the wire dict logged, the router's run
  for Route, `process_action_internal`).
- Each walk hashes (FNV-1a, no hash-map order): every position's legal moves (index, price range, pass), the index and
  price chosen, the generic token encoder's output (`encode_tokens`: every buffer, f32s by their bits) at every 5th
  position, the action log (serde_json, sorted keys) and the result.
- Pinned: **32,050 moves, all 24 walks finished in phase 8**, ~40 s in release. The digests are equal under
  `--no-default-features --features site` and `--features python` (libpython 3.11), in two separate processes, and on
  **RL_18xx origin/master cd97116** (a `git archive` of its `engine-rs` with the same walks): step 1's branch left 1867
  self-play byte-identical to master.
- It runs in every cargo gate; with `site_rules` off every commit of A must leave the digests unchanged. Re-pin only from
  RL_18xx master itself, for an intended 1867 change there, and say so in the commit.

---

## 8. The client's 1867 Rust path (package C, living-board `rust-1867`)

- **Facts, then words** (step 1's split, extended): `rawmenu.ts` gains the 1867 facts of §3.5 (`merger`,
  `reduceTokens`, `redeem`, `nationalize`, `autoLoans`, `tilePower`, `trains[].loans`, `passKind`, `passNationalizes`,
  `auctionStart` + lots' `par`) in place of today's "(Opal only) already worded" parts; `words.ts` makes today's
  `MoveMenu` pieces from them (the same labels, subs, questions: "Pass: CV stays a minor", "Done adding: pick the
  major", "Keep X", "No train: GWR is nationalized", "TN takes 1 loan ($45 each).", the reduce-tokens help "The new public
  company can only keep two tokens and cannot have two tokens in the same hex. Choose which tokens to remove.", ...).
  `moves.ts` (Opal) produces the same facts; its hand must stay word for word what it is today
  (`test/rust/words_titles.cjs`: the four 1867 fixtures identical; `words_same.cjs` for 1830 unchanged).
- **The hands** (`MergerHand.tsx`, `PlayPanel.tsx`'s minor auction strip and par, the reduce-tokens cards, nationalize,
  redeem, the train's loans, the track's power, the pass warnings): unchanged; they read `MoveMenu`. Every move still asked
  first.
- **The worker** (`src/engine-rs/worker.ts`): plays the titles `version().titles` and the site's flag both name; loads
  the game's pack by title (`title-<title>.<hash>.json`, both named in index.html); `withAuctioning` (1830's stopgap) only
  where the view sends no `auctioning`.
- **Posting** (§3.3): no program on: `act` (move + Rust's step auto actions) applied at once, posted with
  `X-Engine: rust`; the stored action (with Ruby's auto actions) must equal it, else re-read and `engine_report`. Programs
  on, or a `program_*` action: `check` → post → append the stored action (step 1). Hotseat: Rust unless programs.
- **The switch**: Settings → Rules engine "Rust (1830, 1867)" (the titles from the site's flag); `?engine=`; the badge;
  every fallback path of step 1 (2-player 1867, `grid_market`, `auto_routing: false`, a stopped replay, a wasm that fails)
  re-opens on Opal and reports.
- **Details** (`details.ts`): Info, Tiles, Spreadsheet for 1867 from the pack and engine-rs's part (loans in the values).
- **Admin → Engine**: lists 1867 rows like 1830's (title column).

---

## 9. The server (package B, rl18xx-next `next/rust-1867`)

- **Key v2** in `lib/engine_shadow.rb` (`KEY_VERSION = 2`, §6), each new field behind `respond_to?` / the title's data
  (1830's key changes only by `auction`); `spec/lib/engine_shadow_spec.rb` over the 1867 fixtures; `scripts/engine_rs/
  ruby_keys.rb` and `shadow_replay.rb` take 1867 corpora (files and the database).
- **Switches** (no code change needed): `ENGINE_SHADOW_TITLES=1830,1867` and `RUST_ENGINE_TITLES=1830,1867` on :9301
  after the corpus checks; `RUST_ENGINE_DEFAULT` stays `opal` until §10.5's acceptance. The boot block's `rust.titles`
  follows the setting; `GET /api/titles/engine?title=1867` gets `rust`.
- **Vendoring**: `scripts/engine-rs-pull.sh <A's commit>` (K6), `scripts/engine-build.sh` (the `site-titles` feature is
  build-site.sh's: K5's interface is unchanged).
- **Act cases**: `ruby_verdicts.rb` judges 1867 cases both ways (`validate_auto_actions: true` with Rust's posted auto
  actions: Rust's step autos must be Ruby's; and `add_auto_actions: true` as for `X-Engine: rust`).
- **Unchanged**: `record_action`'s flow, `X-Engine`, `engine_report`, the run pricing (1867 already `VERIFIED`),
  `POST /api/bot/engine`.
- Docs: ENGINE_RS.md "step 2", DEPLOY.md "engine-rs, step 2" (the production settings for later: Caleb's yes to the
  comparison on 18xx.revys.net is carried out by the parent's deploy, one setting at a time, not by this package).

---

## 10. Verification

### 10.1 The corpus (`$RUST2/corpus`, §11.1)

| Set | How | Size |
|---|---|---|
| :9301 | every 1867 game not `new` (`GET /api/game/<id>`), refreshed before each acceptance run (1218, 1219, 1224, 1225 and the test users' 1220-1223, 1226, 1227 today) | 10 games, 2,667 actions |
| Production | the bot API's read-only export (`curl -s -H @<(printf 'Authorization: Bearer %s\n' "$(cat ~/.config/18xx/local_site_bot_token)") http://127.0.0.1:9292/api/bot/game/<id>`, `game` key), files 600 in `$RUST2/corpus/prod/`, never printed or committed, deleted when step 2's checks are done | 10 games (8 finished), 7,394 actions |
| Fixtures | rl18xx-next `public/fixtures/1867/*` (= its `spec/fixtures/1867/*`) and RL_18xx `tests/fixtures/1867/*` (the same four files, but its `nationalization_cash.json` differs from the site's: both are used) | 4 (+1) files, 3 games (21268 is recorded twice: `21268.json` and `hs_wuveadew_21268.json`) |
| Generated | engine-rs walks (§10.4) as site JSON | ≥ 300 games (3-6 players), ~400k actions |
| RL_18xx human 1867 games | `human_games/1867` + `scrape_2026_10/1867` (993 in scope): they live only in the RL_18xx main checkout, which this work never touches; A asks the parent once for a read-only copy into `$RUST2` (real 18xx.games users' names: 600, never printed, deleted after) | if granted |

Test users' new games (played on B's scratch stack and on :9301 as `lbfb_p1`..`p4`, §10.5) join as they are made. Games
47, 1218 and 1219 and real users' games are only read.

### 10.2 Per position

- **Rust vs Opal** (C's `golden.cjs --title 1867`, then A's `site_replay --goldens` and C's `parity.cjs`): snapshot,
  ActionLite, run tracks, log lines, every seated player's RawMenu and the hand's menu after `words.ts`, tile options at
  every lay position for every ringed hex, the router at every run position (never below Opal's; Opal's 1867 router stops
  at 10 s, so equal-or-above, every Rust run accepted by Ruby at Rust's revenue), details, restartInfo, lastRun, meta,
  programInfo every 10th position, StaticInfo with the pack.
- **Rust vs Ruby**: key v2 at every kept position of every corpus game (B's `ruby_keys.rb` in a throwaway
  `18xx_rack:dev`, A's `site_engine keys`), 1830's corpus included (v2's `auction` is new there).
- **Native vs wasm**: digests of every step's JSON and key equal (step 1's `pkg_smoke.cjs --digest`).
- **Auto actions**: at every position after a move, Rust's predicted step auto actions = the recorded ones (games
  without programs; §2.2's numbers are the baseline).

### 10.3 Every offered option, judged by Ruby (K11)

A's `site_replay --acts` at every decision position (every position of the site corpus; every 5th of generated games):
every RawMenu option (bids at both ends of their range, minor bids, pars, home stations, buys, sales, every tile and
rotation of every ringed hex, stations, the router's run, dividends, trains incl. own-company trains at $1 and at full
price, trade-ins, discards, privates, preloan buys, redemptions, merges, conversions, done, the majors, removed stations,
post-merger shares, nationalize / keep, passes incl. the nationalizing one), acted in Rust (`check`: accepted with the
posted JSON and the key after, or refused with the words). B's `ruby_verdicts.rb` runs each as `record_action` does;
C runs them through Opal's `act` and acts Opal's options in Rust. Required: no option accepted by one engine and refused
by the other except the T6 dead ends (Rust refuses), listed; equal keys after every accepted option; Rust's posted JSON
(tile instance, share ids, auto actions) equal to Ruby's stored and to Opal's.

### 10.4 Generated games

`site_replay --generate` (A): engine-rs walks under `RESTRICTED` (most) and `RULES` (some, so mergers across owners and
cross-player purchases are exercised), every move from the generator, converted to site JSON as a Rust page would post it
(Ruby's tile instances, share ids, the predicted auto actions, ids and users). Ruby replays each in a throwaway container
(no network, source read-only): **every Ruby refusal is a Rust or conversion bug** (or a §5.4 case, then listed); Ruby's
key at every position against Rust's; Opal's goldens over them for §10.2.

### 10.5 Live (dev only)

- B's scratch stack (`scripts/next-stack.sh up 9387` from B's worktree; `docker compose ... down -v` at the end) with
  A's package and C's client: test users `lbfb_p1`..`p4` play 1867 from Rust pages and Opal pages (4 seats: the opening
  auction incl. a dutch purchase, minor auctions won and passed, home stations, two lays with the $20 fee and terrain on
  loans, stations by distance, runs, majors' dividends, the minors' payouts, trains with automatic loans, a trainless pass
  that nationalizes, a major's start in phase 4, conversions and multi-minor merges with done/major/reduce-tokens/post-merger
  shares, redemptions, private purchases incl. preloan, nationalize and keep, phase 5 and 8 events, trade-ins, the end), one
  game with a merger-pass program on (programs path), undo/redo, a run posted mispriced (`repriced`).
- Then :9301 with the shadow on (`ENGINE_SHADOW_TITLES=1830,1867`), Rust opt-in for 1867.
- Browser (C, Playwright, server.mjs :9487 and B's stack): every corpus game opens on Rust with every step equal to Opal's;
  live following; every fallback path (2-player 1867, `grid_market`, hand-drawn runs, a stopped replay, the kill switch);
  a hotseat 1867 from the auction to the end on Rust, every position equal to Opal's replay of its JSON.

### 10.6 Gates and acceptance

- **A, every commit**: `cargo test --release --no-default-features --features python` (both neutrality tests), `--features
  site`, `cargo check --target wasm32-unknown-unknown --no-default-features --features wasm,site-titles`, the default
  `cargo check`; under `taskset -c 24-31,56-63 nice -n 19`, `CUDA_VISIBLE_DEVICES=`.
- **A, before proposing the branch**: `uv sync` in A's worktree, the full `pytest tests/` there (~25-45 min); a failing
  Python-vs-Rust parity test is reported with whether master fails it too, never patched in Python.
- **Step 2 accepted on dev** (then `RUST_ENGINE_DEFAULT=rust` for 1867 on :9301, Caleb's call): strict replay of every
  corpus game Opal replays, every finished result equal; snapshots, actions and tracks 100%; menus ≥ 99.9% of acting
  positions (rest classified); tile options 100% (rest classified as §3.6's border cases); auto routes never below Opal's;
  log lines ≥ 99.5%, beats 100%; key v2 = Ruby's at 100% of positions (1830 too); predicted auto actions = Ruby's at
  100% of positions without programs; §10.3 with no disagreement but the T6 list; generated games all replayed by Ruby;
  native = wasm; 7 days of shadow on :9301 with every non-equal row explained; both neutrality pins and all gates green.

---

## 11. Packages, files and contracts

### 11.1 Shared ground

- `RUST2=/tmp/claude-1000/-home-revys-workspace/252285c2-5545-52b0-bcda-165a3e4e9ceb/scratchpad/rust2`: `pkg/<sha>/` and
  `pkg/current` (A: K2's files, now 1830 + 1867); `corpus/` (§10.1); `goldens/`, `titlepack/` (C); `acts/` (K11);
  `reports/`; `A/`, `B/`, `C/` each package's scratch; `design/` this design's probe (its production export and the
  outputs made from it were deleted when the design was written).
- Rules for all three (step 1 §9.1, unchanged): production off limits but the read-only export; never touch
  `~/workspace/18xx-local`, the RL_18xx main checkout or `~/longrun/deploy/*`; never push; never stop or reconfigure
  :5002, :5004, :5012, :5014, :5022, rl18xx-gen*, watchdogs, mps; no GPU; every build, test and sweep under `taskset -c
  24-31,56-63 nice -n 19`; on :9301 only `lbfb_p1`..`p4` (credentials in the session scratchpad's `creds/`, never
  printed); games 47, 1218, 1219 unchanged; commits end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Ports (free at design time; check before use): B's scratch stack **:9387**, C's `server.mjs` **:9487**, A's browser
  smoke **:9488**.

### 11.2 Contracts (step 1's K1-K11; a change is a commit to this doc first, the parent relays it)

| Contract | Step 2 |
|---|---|
| K1 WasmGame API 2 | Unchanged signatures; `version()` → `key: 2`, `titles: ["1830", "1867"]`; RawMenu gains §3.5's facts (additive); `act`'s `posted` carries the predicted step auto actions (1867; 1830 has none without programs) |
| K2 package files | Unchanged names; the wasm plays both titles |
| K3 `site_engine` | Unchanged commands; `version` titles and key 2; `state` / `keys` / `act` for 1867 (exit 3 for 2-player 1867 and `grid_market`) |
| K4 shadow key | **v2** (§6) |
| K5 the build | `build-site.sh <out>` unchanged for B; inside, `--features wasm,site-titles` |
| K6 vendoring | `engine-rs/` = `git archive <commit> engine-rs` of `site/engine-rs-1867`; `PINNED` names the branch |
| K7 boot / titles API | `rust.titles` lists 1867 when `RUST_ENGINE_TITLES` does (no code change) |
| K8 posting | Unchanged; the client compares the stored auto actions with its own (§3.3) |
| K9 engine_report | Unchanged |
| K10 title pack | `title-1867.<sha12>.json` beside 1830's; the worker loads the game's |
| K11 act cases | Unchanged shape; 1867 kinds added (§10.3) |

### 11.3 The packages

**A — engine** (worktree `~/workspace/RL_18xx/.claude/worktrees/site-engine-1867`, branch `site/engine-rs-1867`; rebased
onto the polished `site/engine-rs-1830` before A1). Files: `engine-rs/src/site/**`, `engine-rs/src/bin/*`,
`engine-rs/site/*`, §5.3's list, `engine-rs/src/neutrality.rs` (done), `scripts/site_engine/**`, this doc. Never Python,
the site's or the client's repos.

| Step | Work |
|---|---|
| A0 | (done) the 1867 neutrality pin (9e06cad); this doc |
| A1 | Rebase onto the polished step-1 head (re-check the pins); `site-titles` (P2); load / replay / records for 1867 (§3.1); `site_engine state/keys` with key v2 (1830 + 1867); corpus fetch with `--title`; strict replay and results; **publish a package** (`$RUST2/pkg`) for B and C |
| A2 | Act / check (§3.2), step auto actions (§3.3), the generated-games tool (§10.4) |
| A3 | View, StaticInfo's engine part, RawMenu facts (§3.4-§3.5), tile options, router, details and ports |
| A4 | Log lines (§3.8) |
| A5 | §10.2 against C's goldens, `--acts` (K11), native = wasm; §5.2's table and test |
| A6 | Full gate (§10.6), the record (§13), the branch proposed to the training session with §5's list |

**B — server** (new rl18xx-next worktree, branch `next/rust-1867` from `rl18xx-next`). Files: `lib/engine_shadow.rb`,
`scripts/engine_rs/*`, `spec/lib/engine_shadow_spec.rb` (+ fixtures), `engine-rs/` (pull script only), docs
(ENGINE_RS.md, DEPLOY.md). Never `lib/engine/**`, `client/**` (subtree pulls only, at integration).

| Step | Work |
|---|---|
| B0 | Vendor A1's package; the CLI and wasm build in the dev containers |
| B1 | Key v2 (§6) and its specs; `ruby_keys.rb` on the 1830 and 1867 corpora against A's keys |
| B2 | `ruby_verdicts.rb` for 1867 act cases (both auto-action modes) and generated games' Ruby replays |
| B3 | Scratch stack :9387 with A's package and C's client: §10.5's live games, shadow rows all equal |

**C — client** (new living-board worktree, branch `rust-1867` from master). Files: `src/engine/rawmenu.ts`,
`src/engine/words.ts`, `src/engine/moves.ts` (facts for 1867's parts, words moved out), `src/engine-rs/**`,
`scripts/rust/titlepack.cjs` (title parameter), `src/engine-rs/packs/title-1867.json`, `build.mjs` / `src/index.html`
(the second pack), `src/ui/settings.ts` / `src/engine-rs/EngineCard.tsx` (the titles' label), `test/rust/**` (1867 in
golden, parity, acts, play, fallback, words_titles), `docs/RUST_ENGINE.md`.

| Step | Work |
|---|---|
| C0 | Title pack 1867; goldens of the 1867 corpus into `$RUST2/goldens` (A tests against them from A3) |
| C1 | Facts/words split for 1867's parts on the Opal path, its hand word for word unchanged |
| C2 | The worker for 1867 (pack by title, posting with Rust's auto actions), the switch's titles |
| C3 | `parity.cjs`, `acts.cjs` (both ways), `play.cjs` (hotseat, site, live), `fallback.cjs` for 1867 |
| C4 | Merge living-board master into `rust-1867`; all suites green |

### 11.4 Order and integration

1. A0 (done). The 1830 polish finishes on `site/engine-rs-1830`; A rebases (A1) and re-runs both neutrality tests.
2. A1 publishes; B0/B1 and C0/C1 start against it. Keys first (A1 + B1) so both v2 implementations agree before the live
   shadow.
3. A2-A5, B2-B3, C2-C4 in parallel; A6. The training session reviews A's branch (§5) and merges with Caleb's OK; the site
   pins A's branch commit until then.
4. C's branch into living-board master; B: `git subtree pull --prefix=client` of that master (subtree commit message
   amended), `scripts/client-build.sh`, `scripts/engine-rs-pull.sh <A's commit>`, `scripts/engine-build.sh`, B's branch
   into `rl18xx-next`; :9301: `ENGINE_SHADOW_TITLES=1830,1867`, `RUST_ENGINE_TITLES=1830,1867`, `RUST_ENGINE_DEFAULT=opal`;
   after §10.6, Rust as 1867's dev default is Caleb's call.

---

## 12. Risks and open points

- **A small real corpus** (24 games): generated games carry most of the coverage; the RL_18xx human corpus would add
  ~600k positions if the parent grants a read-only copy.
- **Ruby defects** (§5.4) and the T6 dead ends: rare; the shadow reports them; decisions are Caleb's.
- **Opal's own 1867 gaps**: water-border costs not shown (§3.6), the router's 10 s cut-off (its runs below the best on a
  few late positions, TITLES.md §10.2): parity counts those as Opal's, listed.
- **The generator's speed in wasm**: native p99 ~7 ms, max ~21 ms per position on the site corpus; the browser 2-3×; a
  late 5+5E router call up to ~1-2 s in wasm (in the worker, with the hand's "Reading the rules…" as today).
- **Key v2 touches 1830** (`auction`): both sides re-checked on the 1830 corpus before the switch.
- **Programs** stay server-made (check → post → append): a Rust page in a game with programs waits for the server for
  each move, as step 1.
- **Upstream Ruby changes** to 1867 (`lib/engine/game/g_1867`, merged from tobymao) can break parity silently: the corpus
  run is repeated before each re-pin and after each `lib/engine` change.

---

## 13. The record

- **2026-11-26, A0**: worktree and branch created from `site/engine-rs-1830` f6a9fe5; commit **9e06cad** (42ac72c after
  the rebase below) pins the 1867
  neutrality walks (§7: digests equal under `site` and `python`, across processes, and on RL_18xx origin/master
  cd97116). The design probe (§2.2) and the wasm size measurement (§3.10) were made on a scratch copy of this branch;
  nothing else of 1867 is changed yet.

### 13.1 Package A as built (2026-11-27)

**Rebase.** `site/engine-rs-1867` rebased onto the polished `site/engine-rs-1830` **7bcc696** (X3's menu, the
snapshot's `ending` / `auctioning`, §11.2 there): the pin and this doc became 42ac72c and 22922a2; both neutrality
tests ran unchanged before any 1867 change and passed (1830: 60 walks, 74,106 moves; 1867: 24 walks, 32,050 moves,
all finished; under `--features site` and `--features python`).

**Commits** (oldest first, after the pin and the design): 1879856 **P2**; 50678a6 **V2**; d23c40c the corpus fetch
by title and the reference Ruby key v2; 431ffeb **A1** (1867 loads and replays; key v2 for both titles); 843460b the
build's titles; 20b1a4c **A2** (act / check through the generator, the step auto actions; the 1867 snapshot and
StaticInfo); a4a08e4 **A3** (RawMenu facts, tile options, the exact router); 86556e3 turn starts, lastRun,
programInfo, meta; 4caff61 §5.2 (Ruby's phase status table) and the 1867 site tests; d077343 the generated games and
the Ruby live judge (§10.4); 907569f **A4** (log lines); d30120a act cases for 1867; 105f93f **P3** (the wasm clock);
1d9d2e7 the merger's converts, each station's city, the docs. Outside `src/site`: P2, V2, P3 only (§5.3), each its own
commit. No `site_rules` branch.

**Package** (K2): `$RUST2/pkg/1d9d2e7b2fb2` (`pkg/current`): `version()` `{api: 2, key: 2, titles: ["1830",
"1867"]}`; the wasm (`wasm,site-titles`) 1,879,310 bytes, 685,212 gzip (the 1830-only one: 1,482,966 / ~551k: the 1867
generator, router and site layer are now linked in, +134 KB gzip); skeletons published on the way: 843460b04692 (A1),
d30120a3311b, 105f93f625b6. `site_engine state|keys|act` take 1867 (exit 3: 2 players, `grid_market`).

**API additions** (additive to K1/K3): RawMenu's 1867 facts (§3.5 "as built"); `act`'s `posted.auto_actions` (the
predicted step passes, 1867); shadow key v2 (§6 "as built"); `site_replay --title`, `--dump <id>`, `--act-check`,
`--generate N [--seed] [--moves] [--out] [--ruby-dead-ends]`; `scripts/site_engine/ruby/live_replay.rb`;
`fetch_corpus.py --title`; `run_keys.sh` reads `CORPUS`.

**Corpus** (§10.1): :9301 10 games (1218-1227), production 11 (24-34, the bot API's read-only export, files 600,
deleted after the checks), fixtures 5 (the site's four + RL_18xx's `nationalization_cash`); 320 generated games.
Opal goldens of the corpus were made in A's scratch with living-board's own `golden.cjs` machinery for `g_1867`
(read-only use; C's official ones replace them).

| Check | Result |
|---|---|
| Strict replay | 26 / 26 corpus games; 320 / 320 generated |
| Key v2 = the reference Ruby key | :9301 2,677 / 2,677; production 6,441 / 6,442; fixtures 3,012 / 3,033 (21: old records whose runs carry no revenue, Ruby's key `null`); generated (as the server stored them) 361,305 / 361,813; 1830 147,037 / 147,037 (:9301), 4,759 / 4,759 (fixtures) |
| Recorded moves accepted by act's check (the generator, Ruby's set) | 12,080 / 12,080 |
| Predicted step auto actions = recorded | 11,829 / 11,933 (the 104: old fixture records made before the client posted these passes; :9301 and production all equal) |
| Generated games judged by the site's Ruby move by move (`validate_auto_actions: true`, revenues stripped, as record_action) | 314 / 320 to the end; 0 runs repriced; the 6 stops are Ruby's: 3 × the opening auction with nobody able to bid (§5.4), 2 × MajorTrainless' phantom after an in-loop reset (§5.4), 1 × Ruby's stale graph cache (below). A first batch without the dead-end avoidance: 248 / 320, 71 of the 72 stops Ruby's stale operator after a MajorTrainless nationalization with majors queued (§5.4) |
| Native = wasm (every step's JSON and key, `replay_digest`) | 346 / 346 (corpus + generated) |
| Against the site's Opal (26 games, 12,149 positions) | snapshots 12,144; actions 12,104 / 12,123; tracks 1,726 / 1,726; menus 47,302 / 48,265; tile options 34,041 / 34,067; restartInfo 11,866; auto routes 1,383 compared: 1,352 equal, 31 above Opal's (its 10 s cut-off), 0 below; log lines 29,341 / 29,925 (98.05%), steps 10,770 |
| 1830 unchanged | spike goldens: snapshots 82,083 / 82,083, menus 322,145 / 322,150, tile options 42,082, routes 7,562 equal, log lines 171,590 / 171,590; live set 3,908 / 3,908 everywhere |
| Act cases (K11) | `$RUST2/acts/r1/cases.jsonl` (:9301 + fixtures, every 5th decision position): 7,248 cases, 6,995 accepted, 253 refused (247 depot trains listed with `short` the company can't pay, as 1830's); `acts/r1_prod` (production, 600, deleted with the export): 10,115 cases, 9,895 accepted |

**Gate** at 1d9d2e7 (§10.6; under `taskset -c 24-31,56-63 nice -n 19`, `CUDA_VISIBLE_DEVICES=`): `cargo test --release
--no-default-features --features python` 270 passed, 0 failed, 11 ignored (both neutrality pins: 1830 60 walks /
74,106 moves, 1867 24 walks / 32,050 moves, unchanged); `--features site` 295 passed, 0 failed, 10 ignored; `cargo
check --target wasm32-unknown-unknown --features wasm,site-titles` and `--features wasm` clean; default `cargo check`
clean; `uv sync` in this worktree and the full `pytest tests/`: 1,822 passed, 7 skipped, 0 failed (21 min). No
Python-vs-Rust parity test failed.

**Ruby defects met** (§5.4 and one more; the site's Ruby is not changed here, the choice is Caleb's): the stale
operator after a MajorTrainless nationalization with other majors queued (no later action matches: the game would
be stuck on the live site); MajorTrainless' phantom after `event_minors_nationalized!`; the opening auction with
nobody able to bid (`undefined method 'cash' for nil`); and **Ruby's stale graph cache**: Merge#mergeable_candidates
(and Token's can_really_place_token?) read `@game.graph` without clearing it after stations moved, so Ruby auto-passes
a minor that has merge candidates (`g.graph.clear` and Ruby lists them, as Rust does); on the live site a Rust page then
sees a stored action with an extra pass and re-reads the game (the shadow reports it).

**Known gaps** (Rust's, listed for B and C):
- Key: a company with two stations in one multi-city hex (Montreal, Toronto) sometimes lists them in the other order
  (`corps.*.tokens`: 1 production position, 505 generated positions in 7 games); a crowded company first in
  DiscardTrain (1 generated position).
- Menus vs Opal: 1867's parts are facts (C words them); Opal's par offers for minors in phase 4+ (Ruby starts minors
  by bid: the generator offers none); an over-the-limit train from the president's other company (34 positions); one
  hex where the generator refuses lays Opal's hand lists (26 options, prod 26@186: to be judged by Ruby in B's
  verdicts).
- Log: multi-minor merger results (shares dealt, closes, "merges into ... receiving", the token count), ReduceTokens'
  "removes token from", an export on a phase change, nationalized minors' closes in some orders; the old fixtures'
  runs without revenue.
- restartInfo: one hotseat fixture whose action ids skip; 1830 game 22 (27 positions, a new comparison, unchanged code).
- Programs' auto actions stay the server's (as step 1).
