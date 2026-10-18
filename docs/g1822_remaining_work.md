# 1822 — status checkpoint & remaining work

*Resume-point doc for the 1822 (The Railways of Great Britain) port to the
Rust engine. Architecture rationale: `docs/multi_title_roadmap.md`
(Phase 2); RL-bridge contract: `docs/rl_bridge_design.md`. Seam commits
carry dense descriptions, including how each was confirmed against Ruby
(`git log --oneline --grep=1822`). Every title targets 3-6 players; the
engine also replays 2 and 7 (the 1822 rules for them cost nothing extra)
but no effort goes into 2p/7p-only rules.*

## Where things stand

| leg | status |
|---|---|
| Title data (map 211 hexes, 62 tiles, 61 companies, 40 corps, trains + variants, phases, 2-D market) | done, verbatim from `lib/engine/game/g_1822/` (HEAD 46a592e) |
| Registration behind `GameTitle` (`title/g1822.rs`), players 2-7 construct | done |
| Optional rules: `plus_expansion`, `plus_expansion_no_removals`, `plus_expansion_single_stack`, `tax_haven_multiple` | done (all four declared; anything else is rejected at construction) |
| Seeded company shuffle (Ruby's LCG `rand`, `sort_by!`) | done — stack order matches Ruby exactly |
| Stock round, choices round, all 16 OR steps (replay) | done |
| Fixture replay (`tests/test_g1822_fixture_replay.py`) | all 4 tobymao fixtures replay end to end with exact scores; zero per-action divergence from Ruby HEAD |
| Corpus audit, 2026-10 scrape (743 games, 2-7p, all rule mixes) | **743/743 outcome match**, with the strict tile-lay rules on |
| Corpus audit, older `human_games/1822` (2,991 finished, 2020-22) | 313 outcome match, 139 `recorded_drift` (our scores == Ruby HEAD's full replay), 2,538 `ruby_rejects_same` (Ruby HEAD rejects the record at the same file action, no earlier state divergence — old-rules records), 1 rejected (33845: its `settings.seed` was back-filled after play; Ruby HEAD accepts bids outside the bid boxes and diverges at action 10). Ruby outcomes: `tests/g1822_ruby_head_results.json` |
| Ruby per-action diff (`scripts/ruby_diff.py --title 1822`) | working (generic over titles) |
| Tile-lay legality (`g1822/tile_rules.rs`) | done, ONE implementation for replay and generator: upgrades_to?, legal rotations (consent rights as `preprocess_action` leaves them, stubs, D35 S, P2, P5), potential tiles, max-exits, check_track_restrictions! |
| Legal-move generator (`g1822/legal.rs`) | done: every step × action type generates (table below); cross-player purchase policy behind a title switch (see "Cross-player purchases") |
| Auto-router (`g1822/router.rs`) + router check | done (numbers below) |
| Fuzz: in-crate random walks (`cargo test g1822::fuzz`), Python wire walks + Ruby HEAD replay | clean (numbers below) |
| Corpus completeness (`scripts/g1822_completeness.py`) | see numbers below |
| **1822PNW** (3-5p): title, replay, router, generator | done — see "1822PNW" below (17/17 Ruby fixtures, 589/597 corpus + 8 Ruby-rejects-same, zero per-action divergence, router 0 shortfalls, fuzz + Ruby replay clean) |
| 1822MX, 1822CA | not started (MX: 363 scraped games, 6 Ruby fixtures, 12 overridden steps incl. the NdeM auction / cash-out; CA: 187 games) |

### Validation numbers (2026-10-16)

* **Router check** (`scripts/g1822_router_check.py --ruby`, every recorded
  `run_routes` of the 743-game scrape, 72,554 runs; router v2,
  2026-10-17): the engine's optimum equals the human's in 66,149 (91.2%)
  and beats it in 6,402 (8.8%; Ruby HEAD validates and prices every one
  of those engine runs at the engine's total). 3 are below, none a router
  shortfall: 2 records are stale (Ruby HEAD prices the human run under the
  engine's optimum) and 1 human run Ruby HEAD rejects. 1.3 ms per call on
  average, the slowest call 0.59 s (2,600 s of router time before → 96 s).
  Against the legacy router at every one of those 72,554 positions v2 is
  never lower and higher in 46. Before v2 (legacy walk): 66,167 equal /
  6,376 above / 11 below incl. 7 real shortfalls (dense late-game maps
  where its 3M-step budget ran out), 25-33 ms per call, calls up to 400 s
  on a loaded machine (the combination search, not the walk).
* **Fuzz**: in-crate 16 walks (3-6p × base/plus × both purchase
  policies, up to 3,000 moves) — no panic, no rejected generated move,
  no stuck state. The Python wire driver's 48 walks (3-6p × base/plus ×
  seeds 1-3 under the title's policy and 4-6 under the rules', 52,192
  moves, all games finished) replay in Ruby HEAD with zero state
  divergence and zero rejections. Earlier runs' Ruby rejections and
  divergences were each a generator/engine bug, fixed: London pending
  token in an unconnected city, CHPR `available` with only a
  destination token, an E-tour through a terminal path, a chain through
  a junction twice, a tied destination bonus on the wrong route.
* **Completeness** (743-game scrape, all 605,384 human decisions except
  `run_routes`, against the rules' move set): 605,301 match exactly one
  generated move (99.986%), none matches two; the 83 misses are the
  replay-only cases below (77 out-of-order discards, 2 re-acquisitions of
  an owned company, 2 stale minor-acquisition choices, 1 raise of one's
  own highest bid, 1 out-of-turn tax-haven buy). 20 decisions (0.003%:
  7 private acquisitions, 6 train buys, 7 minor acquisitions) are legal
  but excluded by the cross-player policy.

## Architecture (read first)

* `engine-rs/src/title/g1822.rs` — data + `GameTitle` impl. 1822's step
  lists are EMPTY: the title sets `is_g1822_family()`, and
  `BaseGame::g1822` (`Some` exactly for these titles) routes every action
  to `crate::g1822` before the shared 1830/1867 dispatch.
* `engine-rs/src/g1822/` — a Ruby-faithful step machine
  (`Round::Base#process_action`, `skip_steps`, per-step `passed` flags,
  `after_process`, the `process_single_action` round loop):
  `mod.rs` (state, setup, dispatch, game end, player value), `stock.rs`,
  `operating.rs`, `track.rs` (tile lays, company tile abilities),
  `tile_rules.rs` (lay legality: one implementation, used by replay's
  strict checks and by the generator), `tokens.rs` (reservations,
  home/destination/pending/special tokens), `trains.rs` (depot, buying,
  rusting, phases, events), `acquisition.rs` (AcquireCompany, company
  choices, MinorAcquisition), `shares.rs` (SharePool transfers incl.
  presidency), `graph.rs` (a direct port of Ruby `Graph#compute` /
  `Node#walk` / `Path#walk`), `legal.rs` (generator), `router.rs`
  (auto-router), `fuzz.rs` (in-crate random walks), `py.rs`.
* Replay follows Ruby's `@game.loading` semantics: checks Ruby skips while
  loading a saved game (token and merge connectivity, blocking abilities)
  are not applied to records; the generator applies them. Tile-lay
  legality (rotation, track restrictions) IS applied to records (Ruby
  validates lays when a game is processed action by action, as the diff
  rig does; the whole scrape passes).
* Ruby quirks the engine reproduces on purpose (each confirmed by a Ruby
  probe; see the seam commits): the crowded-corps memo, the
  `route_info` latch, consent blockers stripped while iterating (every
  second blocker survives), messages carrying auto actions, payoff as a
  current action, `tile_cost_with_discount`, the cheapest-train rule,
  discarded trains losing variants, minor-14 London group, local runs of
  a single city, the destination bonus from `tokens.first`'s city.
* Shared-file hooks (all no-ops unless `g1822.is_some()`):
  `process_action_internal`, `step_action_types_impl`,
  `calculate_results`, the constructor (`build_seeded`, `new_titled(...,
  seed=)`), the PyO3 `process_action` (`combo_entities`, `share_price`).
  `factored.rs` is untouched: 1822 never reaches it. Impassable borders
  come from `BaseGame.walls` (`title::title_walls`, shared by every
  title); `g1822_neighbors` reads them (the family's earlier own reading of
  the HexDef `impassable` flags was identical for 1822 and 1822PNW — test
  `g1822::graph::walls_tests`). Preprinted blue hexes build WHITE for the
  family (its code reads blue from the hex DSL; 1867's build red).
* Seats: 1822 minors are "1".."30", so the replay harness seats players at
  1001+ (`SEAT_ID_BASE`); the constructor rejects colliding ids.

## RL-readiness contract (bridge design §7)

`legal::g1822_generator_status(step, type)` is the table; pytest checks
that every entry is `generated` (`tests/test_g1822_legal_moves.py`).
Python: `BaseGame.g1822_legal_moves_json(policy=None)` returns
`{"wire": <18xx.games action dict>, "range": [lo, hi, step] | null}` per
move; a ranged move's dict holds `lo` (bids, a train bought from another
corporation, partial debt payoff). `run_routes` is generated by the
auto-router (one move: the optimum). Ruby UI rules the generator follows
beyond the server's: bidding tokens per player count, max-exit tiles on
city hexes, purchasable-only acquisitions, connected-only minor
acquisitions and London pending tokens, forced destination token / pass.

New action families for the RL layer (not in 1830's layout): bid-box
`bid` (company + price range), `choose` (P9 double cash, P8 EGR,
pullman attach, minor-acquisition pay/token choices, dividend choose),
`acquire_company`, `choose_ability` (P16 tax haven in the SR; P5 LCDR,
P17 LUR, P18 CHPR in the OR), `merge` (minor acquisition), `hex_token`
(destination), `payoff_player_debt(_partial)`, company `lay_tile` with
combos.

Replay-only (accepted from records, never generated — documented, not
gaps):

* P16 tax haven `choose_ability` by an owner who is not the current
  player (Ruby's `choices_ability` lets it act out of turn).
* `discard_train` for a crowded corporation other than the first: Ruby's
  view lets any crowded corp discard first; the generator discards in
  crowded-corps order (the same end states).
* UI-stale submissions Ruby's process accepts: a second minor-acquisition
  `choose` with a pay choice in the token state, raising one's own
  highest bid (`can_bid_company?` excludes it), re-acquiring a company
  the corporation already owns (`process_acquire_company` does not check).

## Router (v2, 2026-10-17)

`g1822/router.rs`, `g1822_best_routes` → `g1822_best_routes_v2` (the
legacy walk stays as `g1822_best_routes_legacy` / the pymethod
`g1822_router_best_legacy`, for comparison only):

* **Per-kind walks.** Each distinct train kind walks the chain graph on
  its own cost cap (numeric: every stop but France costs 1; pullman:
  towns and France are free, other stops cost 1; local: city + town; E:
  unbounded) instead of one walk bounded by the widest train.
* **Branch and bound.** A partial route is cut when its value plus an
  upper bound on what the rest can add cannot reach the kind's threshold.
  The bound is a walk relaxation of the chain graph (`ub[h][u]`: the best
  revenue of any WALK of h more costly stops from u — nodes may repeat),
  pullman towns folded into their town components (a component's whole
  revenue, then any exit), capped by the h best costly stops plus every
  free stop. This is what "prices stop sets instead of walking every town
  permutation" became: a route into a town cluster is bounded by the
  cluster, not walked through every order of its towns.
* **Thresholds from the combination.** A scouting pass keeps each kind's
  64 best routes; their best combination is a lower bound LB. A route for
  train t can only be in a combination ≥ LB if its value ≥ LB − Σ(best
  single route of the other trains); the full pass keeps every route above
  that (TOP_K = 4,000, best first). The combination search is exact
  unless a kind overflows TOP_K above its threshold or a walk exhausts its
  4M-step budget (counters: `BaseGame.g1822_router_stats()`).
* **Combination search** on dense track / city ids with a bonus-aware
  bound (Σ revenue + the single biggest destination bonus, not Σ (revenue
  + bonus)): the legacy search spent minutes double-counting the bonus
  (264603 @1315: 404 s → 0.56 s).
* **E-trains**: a 300k-step walk plus the token-to-token tours, which may
  now end Channel city → France (`english_channel_etrain?`; 264603 @1341).
* Revenue-first children, token-city starts first; routes deduplicated by
  (oriented stops, track) and recorded from either end. Numeric and
  pullman routes are priced incrementally on the walk; test builds assert
  every such price equals `g1822_price_route` (the fuzz runs it).
* `G1822_ROUTER_DEBUG=1` prints per-call walk/combination timings.

## Cross-player purchases (confirmed by the user, 2026-10-17)

The project disallows, in self-play, buying trains from corporations run
by OTHER players and buying companies from other players (1830's
engines deviate from Ruby there on purpose). For 1822 replay stays
Ruby-faithful (the corpus has such purchases: 20 of 605k decisions in
the 743-game scrape), and the generator applies a title-level switch:
`GameTitle::generator_cross_player() -> CrossPlayerPolicy {trains,
companies, minors}`, default `RESTRICTED` (all false), which 1822 uses.
`g1822_legal_moves_with(CrossPlayerPolicy::RULES)` / the Python
`policy="rules"` give the Ruby-legal set (completeness and fuzz check
both). "Other player" = the seller is not the buying corporation's
president. Cases, and what the switch does with each:

1. Train from a major another player presides — excluded (`trains`).
2. Train from a minor another player owns (a minor's owner is its
   president) — excluded (`trains`).
3. Private company another player owns, acquired by a major or minor
   (1822 acquisitions are free: the seller receives nothing) — excluded
   (`companies`).
4. Minor another player owns, acquired by a major (MinorAcquisition
   `merge`; the minor's owner is paid in cash and/or the major's shares,
   chosen by the acquirer; the minor's trains, cash and companies move
   with it) — excluded (`minors`). **Confirmed (user, 2026-10-17): a major
   may acquire only its own president's minors.** The real rules allow
   another player's minor when both players agree; agreement between
   players is deliberately not modelled (it would need a proposal /
   accept step by another seat, which self-play bots could learn to
   collude through).
5. Bid-box minors bought from the bank (phase `can_acquire_minor_bidbox`)
   — never excluded (the bank sells).
6. Companies owned by a corporation — 1822 has no corporation-to-
   corporation company sales; not applicable.
7. A buyer or seller without a president — counted as another player
   (excluded); no such case arises in 1822 play.
8. Share purchases (IPO, pool, tax haven) and emergency sales are not
   purchases from a player — unaffected.
9. If the policy would leave no legal move, the generator returns the
   rules' set rather than a dead end (counter `stats::CROSS_FALLBACK`;
   never hit in the fuzz).

## Remaining work (ordered)

1. ~~Router speed/strength~~ done 2026-10-17 (router v2, see "Router"
   below and the numbers above).
2. RL bridge: action layout and encoder for 1822 (the action families
   above), behind `alphazero_bridge_ready()` (false today). See
   `docs/rl_bridge_design.md`.
3. Variants: 1822PNW done (see "1822PNW"); 1822MX and 1822CA next, the
   same way — `scripts/g1822_title_data.{rb,py}` generates the data module
   from Ruby, a `Family` row + `Variant` arm, the variant's steps in
   `g1822/<variant>.rs`, its generator in `g1822/<variant>_legal.rs`, and
   the same validation (fixtures, audit, Ruby diff, router check, fuzz,
   completeness — every script takes `--title`).

## The family table (per-title constants, 2026-10-17)

Ruby's G1822 code reads its class constants as `self.class::X`
throughout (the one exception: `london_hex` reads `LONDON_HEX` lexically,
only for minor 14), so a subclass's override applies inside inherited
code. `crate::g1822::family::Family` is that lookup: one static table
per title (`title::g1822::FAMILY`, `title::g1822pnw::FAMILY`, ...),
reached through `BaseGame::fam()` (via `GameTitle::g1822_family()`), with
every field a variant does not override copied from 1822's. A constant a
variant sets to `nil` is `""` (no id is empty, so `Family::is(c, x)` and
lookups fail exactly as Ruby's `== nil` / `hex_by_id(nil)`). Behaviour a
variant overrides (a Ruby METHOD, not a constant) branches on
`BaseGame::variant()`; the predicate methods G1822 defines on constants
(`can_upgrade_one_phase_ahead?`, `can_only_lay_plain_or_towns?`,
`must_be_on_terrain?`, `must_be_on_estuary?`, `must_remove_town?`,
`company_ability_extra_track?`) are methods in `g1822/mod.rs` so a
variant's override has one place to go.

| field(s) | Ruby constant(s) | used by |
|---|---|---|
| `player_range` | meta.rb PLAYER_RANGE | constructor (player count) |
| `bidding_box_{minor,concession,private}_count` | BIDDING_BOX_*_COUNT | bid boxes (mod.rs) |
| `bidding_box_start_{minor,concession,private}` | BIDDING_BOX_START_* | setup_companies |
| `bidding_tokens`, `bidding_tokens_per_action`, `min_bid_increment` | BIDDING_TOKENS(_PER_ACTION), MIN_BID_INCREMENT | stock round, generator |
| `minor_start_par_price`, `minor_bidbox_price`, `token_price`, `upgrade_cost_l_to_2`, `player_loan_interest_rate` | same names | float, minor acquisition, tokens, L→2, loans |
| `starting_companies(_plus)`, `starting_corporations(_plus)`, `plus_expansion_bidbox` | STARTING_*, PLUS_EXPANSION_BIDBOX_1..3 | setup, 1822+ bid boxes |
| `two_home_corporation`, `minor_14_id`, `minor_14_home_hex` | TWO_HOME_CORPORATION, MINOR_14_ID, MINOR_14_HOME_HEX | LNWR destination, minor 14 London |
| `company_{mtonr,lcdr,egr,double_cash,gswr,ber,lsr,10x_revenue,osth,lur,chpr,5x_revenue,hsbc}` (+ `_double_cash_revenue`, `_gswr_discount`, `_hsbc_tiles`) | COMPANY_* | company abilities / choices everywhere |
| `private_{mail_contracts,remove_revenue,phase_revenue,trains}`, `private_acquisition` | PRIVATE_*, PRIVATE_COMPANIES_ACQUISITION | company_bought, mail, acquisition |
| `company_trains`, `concessions`, `abilities`, `corp_token_abilities`, `consent_blockers` | setup_companies' trains; entities.rb abilities (exchange-from-par, tile_lay / teleport / discount, :token, blocks_hexes_consent) | setup, track, tokens, tile_rules |
| `extra_trains`, `extra_train_pullman`, `extra_train_permanents`, `local_trains`, `e_train`, `local_train_can_carry_mail` | EXTRA_TRAINS, EXTRA_TRAIN_PULLMAN, EXTRA_TRAIN_PERMANENTS, LOCAL_TRAINS, E_TRAIN, LOCAL_TRAIN_CAN_CARRY_MAIL | trains, router, mail |
| `trains`, `train_variant(s)`, `destination`, `exchange_tokens` | TRAINS, entities.rb destinations, EXCHANGE_TOKENS | depot, rusting, tokens |
| `london_hex`, `english_channel_hex`, `france_hex`, `cardiff_hex`, `merthyr_tydfil_pontypool_hex` | LONDON_HEX, ENGLISH_CHANNEL_HEX, FRANCE_HEX, CARDIFF_HEX, MERTHYR_TYDFIL_PONTYPOOL_HEX | tokens, tile rules, router |
| `upgradeable_s_hex_name`, `upgradeable_s_yellow_city_tile`, `upgradeable_s_yellow_rotations`, `track_plain`, `track_town`, `minor_green_upgrade` | UPGRADEABLE_S_*, TRACK_PLAIN, TRACK_TOWN, MINOR_GREEN_UPGRADE | tile rules |
| `double_hex` | DOUBLE_HEX | router (`revenue_for`) |
| `hex_dsl`, `hex_dsl_all`, `tile_dsl` | map.rb HEXES / TILES | graph, tile rules, router, borders |
| `game_end_lock_first_trigger` | GAME_END_LOCK_FIRST_TRIGGER | game end |

1822's behaviour is byte-identical through the move (a per-action digest
of state snapshot + both legal-move sets + the router's best run over the
whole 743-game scrape, before vs after: identical — `logs/digest.py`).

## Known issues / divergences

* Lanes are not modelled in 1822's graph (`lanes:2` parses as one path):
  reachability is unaffected on this map; the router counts lane
  capacity per hexside instead. (1822PNW models them: see "1822PNW".)
* Mail-contract subsidies price the route's first/last stop hexes by their
  best node (exact except a city+town hex like D35 when a non-L train stops
  at its town).
* Player debt saturates at i32::MAX (Ruby is unbounded): only degenerate
  fuzz games compound interest that far.
* Ruby dead ends the fuzz meets (Ruby raises on the only way on, e.g. the
  tax-haven presidency-distance crash, a broke major owing minor 14's
  London token): the engine raises "(Ruby raises)" there too and the
  walk stops (`fuzz::ruby_dead_end`).
* The old corpus mostly predates current Ruby rules; Ruby HEAD rejects
  those records itself (bucket `ruby_rejects_same`).

## 1822PNW (2026-10-18)

Pacific Northwest, 3-5 players: a `GameTitle` of the 1822 family
(`title/g1822pnw.rs`, data `title/g1822pnw_data.rs` generated verbatim
from `lib/engine/game/g_1822_pnw/` by `scripts/g1822_title_data.{rb,py}`),
the family table's `Variant::Pnw` row, and its own behaviour in
`g1822/pnw.rs` (replay) and `g1822/pnw_legal.rs` (generator). Ruby's
`G1822PNW` subclasses, and what the engine does for each:

* Steps: PNW `AcquireCompany` (P20 backroom negotiations → choose the
  associated minor to replace), `Assign` (P15 paper mill, P17 ski haus;
  parallel company actions, never blocking), `SpecialTrack` (P10 cubes,
  P12/P13 port tiles, P16 portage, P18 boom town, P19 coal mine + coal
  token, extra-track cubes), `Track` (builder cubes, token cities,
  forests, H19's city+town tiles), `Dividend` (3× jumps), `MinorAcquisition`
  (regionals after two payouts, the coal token moves), `IssueShares` (no
  minors), `BuySellParShares` (associated-minor conversion: par → choose
  the exchange), the Merger round (`Merge`: merge / par / shares / token,
  `TokenMerger`), the regional railways (A, B, C: bid for in the SR, never
  operate), starting packets, the tokencity tile sequences (Portland O8,
  Seattle H11), mail subsidy on laid port tiles, `MUST_SELL_IN_BLOCKS`.
* Map: lanes are modelled in the graph walk for PNW (Ruby
  `Path.make_lanes`: a `lanes:2` path is two paths, crossings
  lane-matched, the walk's edge counters keyed by lane — Portland's
  `lanes:2` and P9's `b_lane` reach hexes a merged path cannot); `track:thin`
  ferry paths are never walked; builder cubes are icons on the TILE (a tile
  upgraded away keeps its cubes in the supply and brings them back, with
  their £40 discounts, when it is laid again — 243949).
* Ruby quirks reproduced (each probe-confirmed): Ruby's Graph cache is
  never cleared after a home token, a regional's float token or a
  closure, so a graph the Merger round computed (`Merge#mergeable`)
  answers stale through the stock round into the next OR until the first
  tile lay / token (`G1822State::ruby_graphs`, PNW only — 238430, 239750);
  the route latch also fills from those computes; the coal token is not a
  graph origin (`city_tokened_by?` takes :normal and :destination only)
  and a route's token stop may not hold it (`check_other`); a minor's
  `tokens.first` is its home token, never the coal token; any
  corporation may `choose` in the Route step, and run_routes detaches the
  pullman from whoever holds it (265923); a pullman choice is `to_i`'d
  ("withhold" → train 0, 253182); a portage lay uses up the turn's lay for
  every other company (`SpecialTrack#get_tile_lay`); a blue hex only
  upgrades to blue.

### Validation numbers (2026-10-18)

* **Ruby fixtures**: all 17 of `public/fixtures/1822PNW` (vendored
  anonymized by `scripts/vendor_ruby_fixture.py`) replay end to end with
  the recorded scores (`tests/test_g1822pnw_fixture_replay.py`) and zero
  per-action divergence from Ruby HEAD.
* **Corpus audit** (597-game 2026-10 scrape, 3-5p): 589 outcome match, 8
  rejected — all 8 `ruby_rejects_same` (Ruby HEAD rejects the record at
  the same action: 4 tokens into a full I12, 3 old-rules rotations, a ski
  haus on the coal mine). Ruby per-action diff of all 597: zero state
  divergence.
* **Router check** (59,301 recorded runs): engine = human in 55,336
  (93.3%), above in 3,944 (6.7%; Ruby HEAD validates and prices every
  one of those runs at the engine's total), below in 21 — none a router
  shortfall: Ruby HEAD rejects 17 of those human runs (old servers let two
  trains share track) and prices 4 under the engine's optimum (stale
  records). 1.6 ms per call, the slowest 0.44 s. The router knows PNW's
  revenue extras (forest +10 per timber hex crossed, the ski haus +30, the
  portage −10 per PNW1/PNW2 tile unless P16, the biggest mill and lumber
  baron bonuses per run — each a per-run maximum like the destination
  bonus, with E-train rules) and counts lane capacity per path END (P9's
  `b_lane` lets two routes, or one route twice, through its hexside).
* **Fuzz**: in-crate `g1822::fuzz::pnw_random_walks_3_to_5_players` (12
  walks, both purchase policies, 9,488 moves, all finished). Python wire
  walks: 30 under the title's policy (seeds 1-10) and 30 under the
  rules' (seeds 11-20), 3-5p, 43,406 moves, all finished; Ruby HEAD replays all 60
  with zero divergence and zero rejections. Bugs the walks found, fixed:
  Assign's wire fields, conversions at a par no exchange works at (or that
  leaves the player short of their open bids / the major short of the
  difference — Ruby raises), parring a major whose associated minor
  another player owns (Ruby crashes in `after_par`), regionals the major
  cannot pay for, laying on a port tile, the portage lay rule.
* **Completeness** (`scripts/g1822_completeness.py --title 1822PNW`, all
  474,694 human decisions of the 597 games except `run_routes`, against
  the rules' move set): 474,659 match exactly one generated move
  (99.993%), none matches two; 16 (0.003%: 10 private acquisitions, 4
  merges, 2 train buys) are legal but excluded by the cross-player
  policy. The 35 misses are replay-only: 8 decisions Ruby HEAD itself
  rejects (the `ruby_rejects_same` records), 14 out-of-order discards, 3
  re-acquisitions of an owned company, 3 raises of one's own highest
  bid, 5 UI-stale chooses (3 share counts sent at the merge token prompt
  — Ruby takes them as `exchange` —, a "withhold" taken as pullman train
  0, another corporation's pullman choice in the Route step), 1 cube on
  a hex that holds none (the UI never offers it; process does not check),
  1 partial payoff above the player's cash. Generator calls leave the
  state untouched (the latch and the Ruby graph cache are restored):
  replay with the generator interleaved rejects exactly the 8 records
  Ruby rejects.

### Cross-player cases (PNW)

The title uses the family default (`RESTRICTED`). PNW adds:

1. Regionals (A/B/C) are owned by the player who won their bid and are
   acquired by majors through MinorAcquisition — counted as "another
   player's minor" (excluded unless the major's president owns it). The
   ambiguity: regionals never operate, so they are closer to a private
   than a minor; the policy treats them like minors (the user's rule for
   minors).
2. Merger-round merges need both minors to have the same owner (a rule,
   not the policy).
3. Conversion (SR) and P20 are the owner's own minor / the bank's — not
   purchases.
4. P20's replaced associated minor is a bank minor — unaffected.

### PNW known issues

* The generator applies every PNW lay to a clone (no pre-validated fast
  path as in 1822): ~11 ms per state on average, Track 64 ms (1822: 47).
* The Ruby graph-cache emulation is entered only where Rust mirrors a full
  Ruby compute (the Merger round's connection checks); a graph Ruby
  caches elsewhere and later reads stale would still be fresh here (none
  in the corpus or the fuzz).
* 1822 itself keeps one path per `lanes:2` path and per-hexside lane
  caps (a_lane / b_lane widths are read for PNW only); its corpus never
  needs more.

## Rigs and gates

```bash
# Ruby probe rig (once): clone + gem dir
git clone --depth 1 https://github.com/tobymao/18xx.git logs/ruby/18xx
docker run --rm --cpuset-cpus 24-31,56-63 -v $PWD/logs/ruby/gems:/usr/local/bundle ruby:3.2-slim gem install require_all
# per-action Rust-vs-Ruby diff (first divergence; --verbose shows rows)
CUDA_VISIBLE_DEVICES="" uv run python scripts/ruby_diff.py --title 1822 tests/fixtures/1822/<fixture>.json
CUDA_VISIBLE_DEVICES="" uv run python scripts/ruby_diff.py --title 1822 --corpus-dir human_games/scrape_2026_10/1822 <id>...
# Ruby HEAD outcomes (scores / rejecting action) merged into the results file
... scripts/ruby_diff.py --title 1822 --corpus-dir human_games/1822 --audit /tmp/a.json --results-json tests/g1822_ruby_head_results.json
# corpus audit (8 workers)
CUDA_VISIBLE_DEVICES="" taskset -c 24-31,56-63 uv run python tests/corpus_audit.py --title 1822 --glob 'human_games/scrape_2026_10/1822/*.json' --jobs 8 --json /tmp/a.json
# generator: wire fuzz (records games) + Ruby replay of them
CUDA_VISIBLE_DEVICES="" uv run python scripts/g1822_fuzz.py --players 3 4 5 6 --seeds 1-3 --moves 2000 --out /tmp/fz [--policy rules]
CUDA_VISIBLE_DEVICES="" uv run python scripts/ruby_diff.py --title 1822 --refresh --jobs 8 /tmp/fz/*.json
# completeness and router check (memory: cap each worker, e.g. ulimit -v 12000000)
CUDA_VISIBLE_DEVICES="" uv run python scripts/g1822_completeness.py --jobs 8 --json /tmp/c.json
CUDA_VISIBLE_DEVICES="" uv run python scripts/g1822_router_check.py --jobs 6 --ruby --json /tmp/r.json
# 1822PNW: the same scripts with --title 1822PNW (ruby_diff takes directories too)
CUDA_VISIBLE_DEVICES="" uv run python scripts/ruby_diff.py --title 1822PNW tests/fixtures/1822PNW
CUDA_VISIBLE_DEVICES="" uv run python scripts/g1822_fuzz.py --title 1822PNW --players 3 4 5 --seeds 1-10 --out /tmp/fzp
CUDA_VISIBLE_DEVICES="" uv run python scripts/ruby_diff.py --title 1822PNW --refresh --work /tmp/rd /tmp/fzp
# per-commit gates: see docs/g1867_remaining_work.md (cargo test, pytest,
# parity_runner --random 0:100, tests/g1867_corpus_audit.py --limit 300).
```
