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

### Validation numbers (2026-10-16)

* **Router check** (`scripts/g1822_router_check.py --ruby`, every recorded
  `run_routes` of the 743-game scrape, 72,554 runs): the engine's optimum
  equals the human's in 66,167 (91.2%) and beats it in 6,376 (8.8%; Ruby
  HEAD validates and prices every one of those engine runs at the
  engine's total). 11 are below: 2 records are stale (Ruby HEAD prices
  the human run under the engine's optimum), 2 human runs Ruby HEAD
  rejects, 7 real shortfalls (0.010%: dense late-game maps with pullman
  7-trains where the 3M-step walk budget runs out; 10-240 short). 33 ms
  per call on average; the budget-bound calls take 2-8 s (one 18 s).
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
  `factored.rs` is untouched: 1822 never reaches it (its own generator
  reads impassable borders from title data via `g1822_neighbors`).
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

1. Router speed/strength on dense late-game maps: the walk is budget-
   bound there (2-8 s a call, 7 recorded runs still above the engine's
   optimum). A search that prices stop sets instead of walking every
   town permutation for pullman (towns free) trains, plus revenue-first
   chain ordering, would make it exact and fast. ~1-2 days.
2. RL bridge: action layout and encoder for 1822 (the action families
   above), behind `alphazero_bridge_ready()` (false today). See
   `docs/rl_bridge_design.md`.
3. Variants (1822CA/MX/PNW) subclass this: see "What variants need".

## What variants need (1822CA / MX / PNW)

Each is a Ruby `G1822::Game` subclass; port each as a new `GameTitle`
with `is_g1822_family() = true` and its own data module, then branch in
`crate::g1822` where Ruby overrides a method. From the census (bridge
design appendix A): CA adds `assign` and mixed 1-/2-slot cities (the
`city_sizes_maintained` check is already ported), MX adds `remove_token`
and `higher_cert_limit`, PNW adds `assign`, `starting_packet` and its own
regional rules (and its own `check_route_token`). The step machine,
graph, tile rules, generator, router and share pool are title-neutral
apart from constants: the ones in `g1822/mod.rs` (company ids, bid-box
sizes, private lists), `tile_rules.rs` (`CONSENT_BLOCKERS`, stubs) and
`router.rs` (`DOUBLE_HEX`, Merthyr/London/France hexes) should move
behind a per-title table first.

## Known issues / divergences

* Lanes are not modelled in the graph (`lanes:2` parses as one path):
  reachability is unaffected on this map; the router counts lane
  capacity per hexside instead.
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
# per-commit gates: see docs/g1867_remaining_work.md (cargo test, pytest,
# parity_runner --random 0:100, tests/g1867_corpus_audit.py --limit 300).
```
