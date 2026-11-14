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
| **1822MX** (2-5p): title, replay, router, generator | done — see "1822MX" below (6/6 Ruby fixtures, 363/363 corpus, zero per-action divergence incl. city token slots, router 0 shortfalls, completeness 99.995%, fuzz + Ruby replay clean) |
| **1822CA** (2-7p, 3-6 for training): title, replay, router, generator | done — see "1822CA" below (7/7 Ruby fixtures, 178/187 corpus + 9 Ruby-rejects-same, zero per-action divergence incl. city token slots, router 1 shortfall, completeness 99.985%, fuzz + Ruby replay clean) |

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

## Generator speed (2026-11-14): same moves, ~7-15x less CPU

The generator was the RL bridge's self-play bottleneck (2026-11-13: 14-25 ms per human
decision on average, p99 up to 0.6 s, a CA position up to 7 s; 1867's generator: 1.2 ms). It
is now several times faster with **exactly the same move lists**: the code it replaced stays
callable as the **reference generator** (`g1822::legal::reference`, a thread-local flag:
`g1822_legal_moves_reference`, Python `BaseGame.g1822_set_reference_generator(on)`), and the
differential tests compare the two state by state.

### Where the time went (before; `scripts/g1822_generator_profile.py --mode profile`)

A sample of each title's 2026-10 scrape (every 6th / 5th / 4th / 2nd game for 1822 / PNW / MX /
CA: 117k / 107k / 74k / 102k positions), `rl_legal_moves_json()` per human decision, thread CPU
on the shared cores (load ~45 on the box, so ~2x the layout agent's numbers):

| Title | decisions | p50 | p90 | p99 | max | mean (ms) | where |
|---|---|---|---|---|---|---|---|
| 1822 | 93,191 | 4.6 | 71 | 413 | 4,790 | 31.4 | Track 63% (111 ms a decision), stock round 20%, Token 8% |
| 1822PNW | 83,253 | 3.8 | 59 | 1,129 | 3,472 | 54.0 | Track 88% (256 ms) |
| 1822MX | 57,328 | 3.6 | 36 | 189 | 1,838 | 16.1 | Track 55% (42 ms), stock round 27% |
| 1822CA | 81,456 | 7.2 | 69 | 332 | 91,311 | 33.8 | stock round 34%, Track 29%, Token 18% (190 ms; one position 91 s), Route (all positions: 20% of the time, the router up to 24-34 s a call) |

By section: the validation clones (`clone_for_search` copied the JSON action log, most of a
late clone: ~1.2 ms each, 0.8-2M per sample) were 40-80% of the time; the graph walk
(`g1822_graph_fresh`, 0.35-3.6M computes per sample, uncached — every gate, lay, token and every
validation clone computed it again; on CA's dense late maps up to ~0.8 s a walk); the neighbour
map (`g1822_neighbors`, recomputed per call through linear `hex_dsl` scans: 0.67 ms, 1.2M calls
on the 1822 sample); the restriction walk of a lay (clone + apply with a second rollback copy +
graph: ~10 ms, 125k calls on the 1822 sample); PNW applied every lay twice (restriction check,
then validation); the candidate dedup formatted every pair (quadratic); minor acquisition's
dead-end lookahead ran the whole generator on each choice's next state; CA's router.

### What changed (`g1822/`, plus `game.rs`' log-free clone and the titles' DSL lookups)

Every item keeps its old code on the reference path (`reference::on()`), except two pure
changes shared by both: the `hex_dsl` / `tile_dsl` lookups are hash indexes (the same first
row as the linear scans) and the action's rollback copy omits the log (below).

* **Clones without the JSON action log** (`BaseGame::clone_for_search_without_log`,
  `g1822_validation_clone`): the family's processing never reads the log (1822 4p after 944
  actions: a clone took 5.2 ms with it, 0.28 ms without). The restriction walk applies on its
  clone without a rollback copy, and `g1822_process_action`'s own rollback copy (every applied
  move, replay included) leaves the log out — a refused action puts the live log back.
* **Graph memo** (`graph::graph_memo`, installed for a whole `g1822_legal_moves_with` call):
  `g1822_graph_fresh` is a pure function of the corporation, the hexes (tiles, paths, tokens)
  and PNW / MX's opened neighbour links; a result is handed out only for inputs EQUAL to the
  ones it was computed from (hexes compared in full). Two maps are kept: the first one seen
  (the state's) and the latest other one (a lay's clone computes its new map's graph two or
  three times).
* **The graph walk on tables** (`fastgraph.rs`): the same walk as `graph.rs` (`node_walk` /
  `path_walk`: same calls, same order, same visited and counter rules) reading a path copy's
  exits, lanes, nodes and flags, a hex's paths per edge, junction paths, converging exits and
  the paths through each node from per-hex tables cached per (title, hex, tile) instead of
  re-deriving and allocating them at every step; the `ignore:1` / `track:thin` flags and the
  DSL's lane specs are read once per title. CA's 91 s Token position: 0.3 s.
* **Neighbour map** (`g1822_neighbors`, now an `Arc`): a per-title table of the static part
  and a thread-local memo keyed by the dynamic part (whether each impassable-coloured
  neighbour's current tile has track towards the hex; PNW / MX: the opened links).
* **Candidate dedup by a set** of `same_move`'s key.
* **1822PNW lays**: one apply decides both the restrictions and the validity (the reference
  applies every lay twice), and plain lays are decided without applying them
  (`g1822_pnw_lay_prevalidate`: the Track step's portage gate, the slot rules (an upgrade = a
  non-white colour), forest / `upgrades_to?` / supply, Hex#lay's token move, the cost with the
  builder cubes the laid instance carries; None — apply it — for token cities, an old blue /
  gray / red tile, H11 / H13, the PNW tiles, H19's city+town tiles, multi-lane track around the
  hex, a corporation Ruby's graph cache answers stale), then checked like 1822's own lays
  (restriction shortcuts, else apply + walk). Test builds assert every verdict against the
  apply on the walks.
* **Stock round**: pars of a corporation `can_par?` refuses are not generated (`process_par`
  rejects every price); the bid-box bids form one validation class (`g1822_bid_own_checks`:
  a member applies iff its own price checks pass and the class's shared conditions hold — the
  proof is in its doc — so one member is applied on a clone).
* **Minor acquisition's dead-end lookahead** (`g1822_has_legal_move`): a choice's next state
  has a legal move if a pass the generator would offer there applies (`g1822_pass_probe`;
  BuyTrain's pass excluded: its validity reads the route latch); else the whole generator.
* **Router**: a route that cannot reach the walk's threshold even with every extra at its cap
  is not priced (the price would be dropped anyway: the same routes kept, the same steps
  counted), and the combination search scans each kind's routes in list order with their track
  as bitsets (`fits`: some track already at capacity = the bitset meets the `full` set) — the
  same search, the same step counts. CA's worst calls (259628 @1954 / @2019): ~24 s → 5-9 s;
  `G1822_ROUTER_DEBUG=1` now prints each walk's time and the combination's step count.

### Differential tests (zero differences required)

* `BaseGame.g1822_reference_diff(policy)` / Rust `g1822_reference_diff_with`: the fast and the
  reference generator's moves at one state, sorted (action, combo, share price, price range,
  certificate ids), the RL index rows of both lists (index, wire dict, range; every move an
  index, no index twice), and the state (the family's state incl. the route latch and Ruby's
  graph cache, the map, corporations, players) unchanged by either call.
  `g1822_graph_check()`: the fast walk against the reference walk for every tokened
  corporation, and the neighbour map's table + memo against the walk over every hex.
* Corpus: `scripts/g1822_generator_diff.py --title <t>` — every human decision of the title's
  2026-10 scrape, under the title's policy AND the rules' (`--both-once`: the reference runs once,
  under the rules', and the title's list is derived from it as the generator derives it -- the
  moves the policy excludes dropped unless nothing else is left). The final code, 2026-11-14:

  | Title | games (stop at a record the replay rejects) | positions | moves compared | differences |
  |---|---|---|---|---|
  | 1822 | 743 (0) | 677,938 | 30,743,140 | 0 |
  | 1822PNW | 597 (8) | 533,995 | 18,533,640 | 0 |
  | 1822MX | 363 (0) | 303,868 | 9,344,779 | 0 |
  | 1822CA | 187 (9) | 196,831 | 10,711,158 | 0 |

  (MX, CA and 1822 also passed on earlier versions of the change with two reference runs.)
* Fuzz: `cargo test --release --no-default-features generator_matches_reference` (three walks
  per title) and the long `g1822_diff_long -- --ignored` (`G1822_DIFF_TITLES`, `_SEEDS`,
  `_FIRST`, `_PLAYERS`, `_POLICY`, `_MOVES`; a move the rules let Ruby raise on, e.g. a ranged
  payoff Ruby can't take, ends the walk as a dead end). Long walks at 3 to the title's maximum
  players, each state compared under both policies, 0 differences: 1822 16 walks (15,632
  states, 394,126 moves), PNW 18 (14,137 / 262,333), MX 12 (7,576 / 99,561), CA 16 (18,919 /
  453,516).
* `tests/test_g1822_generator_reference.py`: one Ruby fixture per title, every third decision.

### Speed (after; the same positions)

The same samples and script after the change (thread CPU per human decision, ms; the machine
under more load than for "before", ~60, so these are conservative):

| Title | before p50 / p90 / p99 / max / mean | after p50 / p90 / p99 / max / mean | total | all positions, mean |
|---|---|---|---|---|
| 1822 | 4.6 / 71 / 413 / 4,790 / 31.4 | 1.3 / 8.9 / 51 / 372 / 4.4 | 7.1x | 26.0 → 3.8 |
| 1822PNW | 3.8 / 59 / 1,129 / 3,472 / 54.0 | 0.9 / 7.7 / 66 / 194 / 4.5 | 12.0x | 43.6 → 3.8 |
| 1822MX | 3.6 / 36 / 189 / 1,838 / 16.1 | 0.9 / 4.9 / 24 / 142 / 2.4 | 6.7x | 12.9 → 2.0 |
| 1822CA | 7.2 / 69 / 332 / 91,311 / 33.8 | 1.7 / 10 / 40 / 5,532 / 5.3 | 6.4x | 32.2 → 6.5 |

(A third of the sample profiled again before and after at the same time: 1822 10.3x on the
decisions.) What is left, per decision: 1822 Track 14.6 ms (58%: the restriction walks' clone +
apply, company lays), stock round 2.5 ms (23%); PNW Track 18.9 ms (78%: lays still applied —
token cities, impassable old colours, multi-lane hexes — and company lays); MX Track 6.5 ms (57%);
CA stock round 3.3 ms, Track 9.3 ms, Token 15 ms, Route 53 ms on its decisions — and the router
is most of CA's time on all positions (`run_routes`-only positions included): ~30 ms a call on
average, 5-9 s at the densest late positions (results unchanged; a faster router would have to
change which routes the step budgets let it find).

Self-play worker per position (`scripts/bench_generic_server.py worker --title <t> --reference`,
128 random-walk positions, the TitlePlay path: legal list + encode + split + compress + fixed
price + apply; ms, fast vs reference generator, under load):

| Title | legal | apply | total |
|---|---|---|---|
| 1822 | 5.6 vs 22.8 | 1.6 vs 2.5 | 8.6 vs 26.4 |
| 1822PNW | 7.5 vs 42.0 | 1.4 vs 1.3 | 9.9 vs 44.0 |
| 1822MX | 3.7 vs 7.6 | 1.1 vs 1.2 | 5.5 vs 9.4 |
| 1822CA | 4.1 vs 10.5 | 1.7 vs 2.4 | 6.9 vs 14.0 |

(the reference column takes the TitlePlay path's fixed price from the legal list, as the fast
one; the script's own reference total also re-runs the generator for it.)

## Tile instances (2026-11-14): the instance Ruby's UI offers

Ruby's UI offers, for each tile name, the FIRST instance of that name in the supply
(`Tracker#potential_tiles` keeps `@game.tiles.uniq(&:name)`; G1822 / PNW / MX override the
colours, not the order), and a lay sends that instance's id. The supply (`@tiles`) starts as
`<name>-0..` per TILES entry (an unlimited tile: one instance) and `update_tile_lists` changes it
at every lay: an unlimited tile adds an instance one past its highest (`add_extra_tile`), the
laid instance leaves, the old tile (unless preprinted) goes LAST. The generator laid the
lowest-numbered instance off the map instead — a tile upgraded away, or for an unlimited tile
(the 7 / 8 / 9 of 1822, PNW and CA) the instance laid first, came back before the fresh ones.
In 1822PNW that is a different move: an instance upgraded off the map keeps its builder cubes
and brings their £40 discounts back when laid again.

The engine now keeps Ruby's supply order (`G1822State::tile_supply_tail`: the instances
appended since setup, in order — maintained by `g1822_lay_tile` for every lay, replay
included) and the generator offers the first instance of the name (`tile_instance_for`: the
initial instances still unlaid in index order, then the tail). The layouts do not change (an
index names the tile, not the instance).

Checks: `scripts/g1822_tile_instance_check.py --title <t>` (every human lay of the corpus: the
instance the human's UI sent against the generator's; `--ruby` asks Ruby HEAD for its first
instance, `scripts/g1822_tile_instance_ruby.rb`) and the in-crate walk test
`offered_tile_instance_is_rubys` (every offered lay at every state of 3-6p walks of each title
against an independently kept model of Ruby's supply list). 2026-11-14, every human lay of the
2026-10 scrapes (all player counts):

| Title | human lays | new instance == human | old rule == human | decisions with lays where the offered instance changes |
|---|---|---|---|---|
| 1822 | 86,266 | 86,265 | 68,378 | 63,255 of 94,458 (2,670,913 lay moves) |
| 1822PNW | 60,721 | 60,721 | 52,543 | 37,916 of 74,912 (1,097,700) |
| 1822MX | 34,853 | 34,853 | 26,465 | 29,429 of 45,718 (1,039,453) |
| 1822CA | 28,109 | 28,109 | 22,499 | 20,480 of 30,785 (1,089,958) |

The one miss (230019 @985, 1822) is the human's, not the generator's: Ruby HEAD's first instance
there is 8-3 (the generator's), the human sent 8-13 (and laid 8-3 next). Ruby HEAD
(`--ruby`) agrees with the generator at every human lay of 20 affected PNW games (2,293 lays; the
old rule: 349 differ) and of 230019 (133). Six PNW wire fuzz games from the new generator
(`scripts/g1822_fuzz.py`, 3-5p) replay in Ruby HEAD with zero divergence, Ruby's first instance =
the generator's at all 850 lays. The walk test checks 104,818 offered lays (38,335 where the old
rule differs). The PNW index-level completeness rerun (`scripts/g1822_completeness.py --index
--title 1822PNW`): 533,995 decisions, 533,960 with exactly one index (16 of them policy-excluded),
the same 35 replay-only misses, 0 ambiguous, 0 decode and **0 apply mismatches** (were 10).

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
2. RL bridge: ~~action layout~~ done 2026-11-13 (milestone A4: `1822-v1`,
   `1822PNW-v1`, `1822MX-v1`, `1822CA-v1`, registry layouts over this
   generator, `docs/rl_descriptors.md` §9; `rl_layout_kind` = Generator,
   `rl_legal_indices` / `rl_decode_index` / `rl_apply_index` /
   `rl_price_range`); the encoder is milestone B's (`rl_encoder_ready`
   stays false). See `docs/rl_bridge_design.md`.
3. Variants: 1822PNW, 1822MX and 1822CA done (see their sections); the
   1822CA scenarios (ERS / WRS) and 1822Africa / 1822MRS / 1822NRS would go
   the same way — `scripts/g1822_title_data.{rb,py}` generates the data module
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

* ~~**Tile instance**~~ fixed 2026-11-14 (see "Tile instances" below):
  the generator laid the lowest-numbered free instance of a tile, Ruby's
  UI the first of the name in its supply; in PNW an instance upgraded off
  the map keeps its builder cubes and brings their discount back when laid
  again (232649 @1168: the human laid 9-13, the generator offered 9-2,
  $10 cheaper). The generator now offers Ruby's instance.
* PNW lays: since 2026-11-14 plain lays are pre-validated
  (`g1822_pnw_lay_prevalidate`, "Generator speed" above); token cities,
  impassable old colours, H11 / H13, the PNW tiles and multi-lane hexes are
  still applied to a clone (once).
* The Ruby graph-cache emulation is entered only where Rust mirrors a full
  Ruby compute (the Merger round's connection checks); a graph Ruby
  caches elsewhere and later reads stale would still be fresh here (none
  in the corpus or the fuzz).
* 1822 itself keeps one path per `lanes:2` path and per-hexside lane
  caps (a_lane / b_lane widths are read for PNW only); its corpus never
  needs more.

## 1822MX (2026-10-18)

The Railways of Mexico, 2-5 players (3-5 for training): a `GameTitle` of
the 1822 family (`title/g1822mx.rs`, data `title/g1822mx_data.rs`
generated from `lib/engine/game/g_1822_mx/`), the family table's
`Variant::Mx` row, its rules in `g1822/mx.rs` (replay) and
`g1822/mx_legal.rs` (generator). Ruby's `G1822MX` overrides, and what the
engine does for each:

* **The NdeM** (Ferrocarriles Nacionales de Mexico, `NDEM`): a major with
  no president (owned by the pool), floated at setup at par 100 with its
  shares in the pool, outside the cert limit, never sold out, its sales
  moving nothing (`setup_ndem`, MX SharePool / BSPS / `sell_shares_and_
  change_price`). It takes over the home of the earliest of M14/M15/M17 in
  the shuffled stack and of every first-bid-box minor nobody bids on while
  it is open (`replace_minor_with_ndem`: the company is deleted, the
  reservation goes, a token unless it already has one in that city, a new
  pool share), receives the unbid minors' L trains (`send_train_to_ndem`:
  an L flips to a 2 in phase 2, its oldest trains rust over the limit),
  operates last, its shareholders lay its track in turn (MX Track:
  `@ndem_tile_layers`), never tokens, pays out automatically (its own
  price rule: right once per full share price earned). The first 7-train
  privatizes it (`close_ndem`): it moves to operate next, runs once more,
  `CashOutNdem` sells every player's NdeM shares to the pool and
  `AuctionNdemTokens` auctions its tokens — the chooser (priority order)
  picks one (`remove_token`) or passes, players bid in 10s up to their
  richest exchange-token major's cash (`bid` with no company), the winner
  names the buying major (`merge`); then `close!` (its tokens stay on the
  map). `BaseGame.g1822_acting_player()` names the deciding player for
  the NdeM's moves (tile layer / chooser / bidder / winner, else its first
  shareholder in priority).
* C1 (FCM's concession, a £200 minimum: `discount -100`, now emitted by
  the data generator) brings minor 18, floated on purchase (par 50, $100,
  its company stays the player's without being a certificate:
  `G1822State::owner_only`); M18 sits last in the minor stack.
* Companies: P7 double cash (G1822's double-cash code — MX overrides every
  reader of `COMPANY_DOUBLE_CASH` with the same code for P7), P8 advanced
  tile lay, P9 extra tile lay (or a cube), P10/P11 three builder cubes
  (when: track), P12/P13 remove town, P14/P15 mail, P16 stock drop (−$10 a
  round, `close_p16` through SpecialChoose — it never blocks — or when
  unpaid, moving its owner left), P17/P18 port tiles (blue spikes; blue
  upgrades to blue only; turned to the spike's first path), train
  privates P1-P6 incl. P3's 3/2P (half revenue after its bonuses, rounded
  to 10).
* Map: builder cubes are block icons on the hex's current tile (mountain
  2, hill 1, river 1; 80/40/20 off) and every lay strips them from the
  new tile; Mexico City (N23) holds several homes (`multiple_tokens_
  allowed_on_home_hex?`) and is one route group; M22 must connect to
  Mexico City while nobody holds C2 or presides MC; preprinted green
  hexes (N25) are green; `track:thin` ferries are never walked; the
  neighbour map is Ruby's setup-time `connect_hexes`.
* Ruby quirks reproduced: minor 14's $20 home token is placed free
  (City#place_token spends nothing); a `["local", hex]` run is rebuilt on
  the hex's FIRST tokened city (two local runs in two cities of one hex
  are illegal); `remove_extra_tokens!` returns the token latest in the
  corporation's token list (map tokens now carry their corporation
  token's identity, `Token::serial`, MX only) and Hex#lay does not sort
  cheaters (Ruby's `transform_values` sort is a no-op); the tokens up for
  auction are those at the NdeM's turn start, following their city through
  upgrades; Ruby crashes (nil.place) when the NdeM must take an 11th home
  with all ten tokens out, and on `close!` when the NdeM holds cash (a
  train bought from it) — the engine raises "(Ruby raises)" there too.
* Preprinted tile ids are "<coord>-<index in its HEXES group>" (Ruby
  init_hexes); the wire city ids use them (`Family::hex_index`, generated;
  1822 / PNW keep index 0 as validated).

### Validation numbers (2026-10-18)

* **Ruby fixtures**: all 6 of `public/fixtures/1822MX` (vendored
  anonymized) replay with the recorded scores
  (`tests/test_g1822mx_fixture_replay.py`) and zero per-action divergence
  from Ruby HEAD, city token slots included (`ruby_diff.py --tokens`).
* **Corpus** (363-game 2026-10 scrape, 2-5p, 51 with higher_cert_limit):
  363/363 outcome match; Ruby per-action diff of all 363: zero state
  divergence, city token slots included.
* **Router check** (34,769 recorded runs): engine = human in 32,292
  (92.9%), above in 2,462 (7.1%; Ruby HEAD validates and prices every one
  at the engine's total), below in 15 — none a shortfall: Ruby HEAD
  rejects all 15 human runs (13 run two local trains in two cities of one
  hex, see above). 0.8 ms per call.
* **Completeness** (all 269,099 human decisions except `run_routes`,
  against the rules' move set): 269,085 match exactly one generated move
  (99.9948%), none matches two; 14 (0.005%: 2 private acquisitions, 6 train
  buys, 6 minor acquisitions) are legal but excluded by the cross-player
  policy. The 14 misses are replay-only: 12 out-of-order discards and 2
  re-acquisitions of an owned company. NdeM auction decisions (778
  remove_token, its bids, merges) all match.
* **Fuzz**: in-crate `g1822::fuzz::mx_random_walks_3_to_5_players` (12
  walks, both policies, higher_cert_limit on even seeds); wire walks: 30
  under the title's policy (seeds 1-5) and 30 under the rules' (seeds
  6-10), 3-5p × base / higher_cert_limit, 42,579 moves — 26 finished, the
  rest at the Ruby dead ends above (the NdeM's 11th home, its cash at
  close!, a ranged payoff that ends the round into the 11th home); 25
  reach the NdeM auction. Ruby HEAD replays all 60 with zero divergence
  (token slots included) and zero rejections. Bugs the walks found, fixed:
  wire city ids of preprinted hexes, a port tile upgraded to a city tile,
  the 3/2P priced in full, two local runs in one hex, remove_extra_tokens'
  choice of token.

### Cross-player cases (MX)

The title uses the family default (`RESTRICTED`). MX adds:

1. The NdeM has no president: a train bought from it is "from another
   player" (excluded). Under the rules' set such a purchase gives the NdeM
   cash, and Ruby's `close!` then raises at its privatization.
2. The NdeM token auction sells the state's tokens to a major the winner
   presides — not a purchase from a player.
3. Minor 18 comes with C1 (bank) — unaffected.

### MX known issues

* The NdeM running out of tokens and the NdeM holding cash at close are
  Ruby dead ends the engine mirrors (see above); real games come close
  (2 of 363 used all ten tokens).
* `G1822::Step::Choose#auto_actions` (an inconsequential double-cash choice
  is passed automatically) is not mirrored by the generator, which offers
  `choose` / `pass` (both legal).

## 1822CA (2026-10-18)

The Railways of Canada, 2-7 players as Ruby (3-6 for training; the 2p and
7p games replay, nothing 2p/7p-specific was validated beyond that): a
`GameTitle` of the 1822 family (`title/g1822ca.rs`, data
`title/g1822ca_data.rs` generated from `lib/engine/game/g_1822_ca/`, a flat
map with "AA".."AP" and lowercase "a6"/"a8" coordinates — `parse_coord`
now reads Ruby's LETTERS / NEGATIVE_LETTERS), the family table's
`Variant::Ca` row, its rules in `g1822/ca.rs` (constants, tiles, sawmill,
acquisition lay, DD), `g1822/ca_tokens.rs` (QMOO, destinations, P10),
`g1822/ca_route.rs` (pullman / grain attachments, small mail) and
`g1822/ca_legal.rs` (generator). The CA scenarios (1822CA ERS / WRS: 41 /
25 scraped games) are separate titles and not done. Ruby's `G1822CA`
overrides, and what the engine does for each:

* **Map and tiles.** The five big cities (Toronto AC21, Ottawa AE15,
  Montreal AF12, Quebec AH8, Winnipeg N16 — several cities each, one route
  group each) with CA's rotation table (`legal_tile_rotation?`) and
  BIG_CITY_ILLEGAL_TILE_UPGRADES; mixed-slot cities keep their slots
  (Tracker `city_sizes_maintained`); the white AG13 (a city and a town)
  takes 5 / 6 / 57 (its town goes); the mountain passes E11 / F16 take
  7 / 8 / 9 despite their labels; TRACK_RESTRICTION :permissive (only "New
  track must override old one"); max exits for unlabeled cities only;
  MAJOR_TILE_LAYS with cannot_reuse_same_hex on both lays; lanes: the A /
  B tiles, AC23 and Y29 have `lanes:2` paths (Ruby `Path.make_lanes` as for
  PNW, plus `lane_match_different_sizes?`: a 2-lane path meets 1-lane track
  on one lane only).
* **Detroit-Duluth.** A blocker corporation `DD` fills Q19 and Y27 (their
  cities also have `visit_cost:99`) until the first 3-train
  (`open_detroit_duluth`: tokens destroyed, the DD tiles laid without the
  supply).
* **QMOO's home** (`update_home`): its reservation follows Quebec's cities
  (a city when exactly one has room, else the hex → a pending token), a
  yellow lay over its placed home token takes it back up; the extra
  PendingToken step after Track places it. ICR's destination (any Quebec
  city it connects to: `place_token`) and the exit-specific destinations
  (GNWR / NTR at Winnipeg, GT at Montreal: `destination_exits`) pick the
  city; `@destinated` keeps DestinationToken quiet afterwards. Destination
  tokens are cheaters; a lay on a destination hex picks them up first and
  puts them back after (`preprocess_action` / `action_processed`: map
  tokens now carry Ruby's `Token#cheater`, read by CA only).
* **Companies.** P14-P18 free big-city upgrades (extra track, `count_per_or`
  1 — reset at each OR — and gone once the city is gray), P19 / P20 pass
  tiles and P29 / P30 remove-town (consume the whole lay; never on the
  sawmill hex), P12 advanced tile lay (+ P29 / P30 combos), P21 three free
  tiles in any order (closes on the Track pass once used), CA Tracker: no
  consume-tile-lay private after a lay; P10 Winnipeg token (an exchange
  token into an extra slot, `City#extra_tokens`, any time in its owner's
  turn — its ability also opens the Token step); P13 sawmill
  (AssignSawmill: assign, then open +$20 for all / closed +$10 for the
  owner; best single run, ×2 for an E-train or a destination run through
  it); P22 / P23 mail, P24 / P25 small mail (phase subsidy, interleaved
  with the large ones as `mail_contract_bonus`); P26 / P27 grain trains;
  P11 tax haven holding several shares (TAX_HAVEN_MULTIPLE).
* **Routes.** The Route step's choices: attach the pullman (CA: "<name>+<2d>",
  d cities / offboards, towns up to 2d plus the cities' unused share, every
  stop pays) and then the grain train ("<name>G": up to 12 grain-elevator
  stops besides the train's own; no single-city local run), "None" skipping
  either; `grain_and_port_bonus` (majors; $10 per elevator + $10 per port
  side, or $20 per port side for the grain-attached train; never the
  pullman one nor E); a second grain train is discarded first; grain trains
  never run.
* **MinorAcquisition → AcquisitionTrack**: the acquiring major's bonus lay
  (slot 0's lay/upgrade whatever it laid before). Ruby keeps the major's
  graph from `entity_connects?` when the token choice is 'remove' (no
  clear): the bonus lay sees it stale — emulated (the PNW graph memo, CA
  `process_merge` only); Ruby HEAD rejects one corpus game there (263806),
  as Rust does.
* Stock: the one-row market's sell movement (president: left per 10%,
  others one; 2 players: left per share for the president only), sold out
  → right, issuing → left per share; the bank break adds one OR to the set
  (`game_end_set_final_turn!`).

### Validation numbers (2026-10-18)

* **Ruby fixtures**: all 7 of `public/fixtures/1822CA` (vendored
  anonymized; one 7-player) replay with the recorded scores
  (`tests/test_g1822ca_fixture_replay.py`) and zero per-action divergence
  from Ruby HEAD, city token slots included.
* **Corpus** (187-game 2026-10 scrape, 2-7p): 178 outcome match, 9
  rejected at a file action Ruby HEAD also rejects (8 minors passing a
  Route step they must run — old rules — and 263806's stale-graph lay
  above); Ruby per-action diff of all 187: zero state divergence (token
  slots included) up to those points.
* **Router check** (20,945 recorded runs): engine = human in 18,629
  (88.9%), above in 2,313 (11.0%; Ruby HEAD validates and prices every one
  at the engine's total), below in 3: Ruby HEAD rejects 2 of those human
  runs (a plain 7 and a plain 5 run with 18 / 15 stops after an undone
  pullman choice), 1 real shortfall (257158 @1556: a grain-attached 7, a
  pullman 5 and an LP — 1,020 vs 1,030; the compatible pullman route is
  not among the 16,000 best kept). 30 ms per call on average, up to ~20 s
  in the densest late positions (see known issues). CA router settings:
  scouting depth 512, TOP_K 16,000, repeated full passes while the
  combination improves (the better of the passes kept), E-trains searched
  last in the combination (they share no track with the rest), a
  150M-step combination budget, E walks 3M steps and branch-and-bound E
  tours (the other titles are unchanged).
* **Completeness** (all 175,886 human decisions except `run_routes`,
  rules' move set): 175,860 match exactly one generated move (99.985%),
  none matches two; 13 (0.007%: 6 private acquisitions, 4 train buys, 3
  minor acquisitions) are excluded by the cross-player policy. The 26
  misses are replay-only or rejected: 16 out-of-order discards, 1
  re-acquisition of an owned company, 9 at the Ruby HEAD rejections above.
* **Fuzz**: in-crate `g1822::fuzz::ca_random_walks_3_to_6_players` (16
  walks, both policies: 16,746 moves, all finished); wire walks: 20 under
  the title's policy (seeds 1-5) and 20 under the rules' (seeds 6-10),
  3-6p, 47,323 moves — 39 finished, 1 at a Ruby dead end (an emergency
  sale whose presidency change is ranked against the tax haven: Ruby
  raises). They cover P10 (32 tokens), the sawmill (35), P12-P21 / P29-P30
  lays, 284 "None" route choices and the attachments, 58 destination
  tokens (ICR's city choice in 3 games), minor acquisitions with every
  token choice. Ruby HEAD replays all 40 with zero divergence (token slots
  included) and zero rejections; one (4p s10) Ruby could not finish: from
  file action 1509 its DestinationToken pass check (`nodes_connected?`, a
  path walk) ran over 15 minutes on that dense random map — compared up
  to there. The first Ruby replay found the one generator bug: a
  destination `place_token` wire without `token_type` (Ruby's PlaceToken
  then takes a normal token).
  `scripts/ruby_shims.rb`: CA's `check_distance(route, visits, train =
  nil)` ends in a bare `super` into G1822's two-argument method, which MRI
  rejects (ArgumentError) on every route Ruby validates; the site runs Opal,
  which ignores the extra argument — the rig mirrors Opal (human records
  carry their revenue, so Ruby never validates them).

### Cross-player cases (CA)

The title uses the family default (`RESTRICTED`); CA adds no acquisition
of its own (P10, P13 and the big-city privates are the owner's abilities;
the acquisition bonus lay follows a minor acquisition the policy already
filtered).

### CA known issues

* Router: one shortfall (above) and slow late positions — a CA pullman's
  bound lets every town count (towns are only bounded by 3d stops), so
  pullman / grain walks on the densest maps run long (up to ~20 s for one
  call). Not bridge-ready anyway.
* Kept for Ruby parity: Ruby's `remove_extra_tokens!` / Hex#lay quirks as
  MX (token identity on), and the stale graph after a 'remove' acquisition.
* The scenarios 1822CA ERS / WRS are not implemented.
* A destination `place_token` always places the destination token (Ruby
  takes `action.token`, by `token_type` — a record without it would put a
  normal token there; the site and the generator always name it).
* `lane_match_different_sizes?` is on for CA only; 1822PNW (validated as
  it is) still treats a 2-lane / 1-lane crossing as unconnected — kept for
  byte-identity, worth a probe if PNW ever shows a lane divergence.

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
# --tokens: also compare every city's tokens slot by slot (fresh --work dir)
CUDA_VISIBLE_DEVICES="" uv run python scripts/ruby_diff.py --title 1822MX --tokens --work /tmp/rd tests/fixtures/1822MX
# 1822PNW / 1822MX: the same scripts with --title (ruby_diff takes directories too)
CUDA_VISIBLE_DEVICES="" uv run python scripts/ruby_diff.py --title 1822PNW tests/fixtures/1822PNW
CUDA_VISIBLE_DEVICES="" uv run python scripts/g1822_fuzz.py --title 1822PNW --players 3 4 5 --seeds 1-10 --out /tmp/fzp
CUDA_VISIBLE_DEVICES="" uv run python scripts/ruby_diff.py --title 1822PNW --refresh --work /tmp/rd /tmp/fzp
# 1822CA: likewise (3-6 players; the rig loads scripts/ruby_shims.rb — MRI vs Opal on check_distance)
CUDA_VISIBLE_DEVICES="" uv run python scripts/g1822_fuzz.py --title 1822CA --players 3 4 5 6 --seeds 1-5 --out /tmp/fzc [--policy rules]
CUDA_VISIBLE_DEVICES="" uv run python scripts/ruby_diff.py --title 1822CA --tokens --refresh --work /tmp/rdc /tmp/fzc
# generator speed (2026-11-14): per-section profile, the corpus differential test (fast vs the
# reference generator, both policies; --both-once runs the reference once per position), the
# fuzz differential walks, the tile instance against the human records / Ruby HEAD
CUDA_VISIBLE_DEVICES="" taskset -c 24-31,56-63 nice -19 uv run python -P scripts/g1822_generator_profile.py --title 1822PNW --every 5 --jobs 8 --out /tmp/p.jsonl.gz
uv run python -P scripts/g1822_generator_profile.py --summary /tmp/p.jsonl.gz --decisions
CUDA_VISIBLE_DEVICES="" taskset -c 24-31,56-63 nice -19 uv run python -P scripts/g1822_generator_diff.py --title 1822PNW --jobs 8 --both-once
(cd engine-rs && cargo test --release --no-default-features generator_matches_reference)
(cd engine-rs && G1822_DIFF_TITLES=1822PNW G1822_DIFF_SEEDS=3 cargo test --release --no-default-features g1822_diff_long -- --ignored --nocapture)
CUDA_VISIBLE_DEVICES="" uv run python -P scripts/g1822_tile_instance_check.py --title 1822PNW --jobs 8 [--ruby]
# per-commit gates: see docs/g1867_remaining_work.md (cargo test, pytest,
# parity_runner --random 0:100, tests/g1867_corpus_audit.py --limit 300).
```
