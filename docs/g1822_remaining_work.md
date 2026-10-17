# 1822 — status checkpoint & remaining work

*Resume-point doc for the 1822 (The Railways of Great Britain) port to the
Rust engine. Architecture rationale: `docs/multi_title_roadmap.md`
(Phase 2). Seam commits carry dense descriptions, including how each was
confirmed against Ruby (`git log --oneline --grep=1822`). Every title
targets 3-6 players; the engine also replays 2 and 7 (the 1822 rules for
them cost nothing extra) but no effort goes into 2p/7p-only rules.*

## Where things stand

| leg | status |
|---|---|
| Title data (map 211 hexes, 62 tiles, 61 companies, 40 corps, trains + variants, phases, 2-D market) | done, verbatim from `lib/engine/game/g_1822/` (HEAD 46a592e) |
| Registration behind `GameTitle` (`title/g1822.rs`), players 2-7 construct | done |
| Optional rules: `plus_expansion`, `plus_expansion_no_removals`, `plus_expansion_single_stack`, `tax_haven_multiple` | done (all four declared; anything else is rejected at construction) |
| Seeded company shuffle (Ruby's LCG `rand`, `sort_by!`) | done — stack order matches Ruby exactly |
| Stock round, choices round, all 16 OR steps (replay) | done |
| Fixture replay (`tests/test_g1822_fixture_replay.py`) | all 4 tobymao fixtures replay end to end with exact scores; zero per-action divergence from Ruby HEAD |
| Corpus audit, 2026-10 scrape (743 games, 2-7p, all rule mixes) | **743/743 outcome match** |
| Corpus audit, older `human_games/1822` (2,991 finished, 2020-22) | 313 outcome match, 139 `recorded_drift` (our scores == Ruby HEAD's full replay), 2,538 `ruby_rejects_same` (Ruby HEAD rejects the record at the same file action, no earlier state divergence — old-rules records: first-turn housekeeping, L-train runs), 1 rejected (33845: its `settings.seed` was back-filled after play; Ruby HEAD accepts bids outside the bid boxes and diverges at action 10). Ruby outcomes: `tests/g1822_ruby_head_results.json` |
| Ruby per-action diff (`scripts/ruby_diff.py --title 1822`) | working (generic over titles) |
| Tile-lay legality (`g1822/tile_rules.rs`) | written: upgrades_to?, legal rotations (consent/stubs, D35 S, P2, P5), potential tiles, max-exits, check_track_restrictions! — validating via the generator + Ruby replay of fuzz games |
| Legal-move generator (`g1822/legal.rs`) + in-crate random-walk fuzz | in progress |
| Auto-router + router check | not started (the generator runs no trains meanwhile — legal in Ruby) |

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
  `tile_rules.rs` (lay legality Ruby checks only for live players),
  `tokens.rs` (reservations, home/destination/pending/special tokens),
  `trains.rs` (depot, buying, rusting, phases, events),
  `acquisition.rs` (AcquireCompany, company choices, MinorAcquisition),
  `shares.rs` (SharePool transfers incl. presidency), `graph.rs` (a direct
  port of Ruby `Graph#compute` / `Node#walk` / `Path#walk`), `legal.rs`
  (generator + fuzz), `py.rs`.
* Replay follows Ruby's `@game.loading` semantics: checks Ruby skips while
  loading a saved game (rotation legality, track restrictions, token and
  merge connectivity, blocking abilities) are not applied to records; the
  generator applies them.
* Shared-file hooks (all no-ops unless `g1822.is_some()`):
  `process_action_internal`, `step_action_types_impl`,
  `calculate_results`, the constructor (`build_seeded`, `new_titled(...,
  seed=)`), the PyO3 `process_action` (`combo_entities`, `share_price`).
  `factored.rs` is untouched: 1822 never reaches it (its own generator
  reads borders from title data via `g1822_neighbors`).
* Seats: 1822 minors are "1".."30", so the replay harness seats players at
  1001+ (`SEAT_ID_BASE`); the constructor rejects colliding ids.

## RL-readiness contract (bridge design §7)

`legal::g1822_generator_status(step, type)` is the table. Every step and
action type generates except `run_routes`, which runs no trains until the
auto-router exists (replay-only for real routes). Ruby's UI rules the
generator follows beyond the server's: bidding tokens per player count
(the browser's lookup works; the server's never matches), max-exit tiles
on city hexes, purchasable-only acquisitions, connected-only minor
acquisitions, forced destination token / pass.

## Remaining work (ordered)

1. Generator + fuzz clean at 3-6 players (`cargo test --release
   --no-default-features g1822::legal`); Python fuzz driver that records
   wire dicts, then Ruby replay of fuzz games (`scripts/ruby_diff.py` on
   the recorded JSON) — the soundness check for illegal-but-applies moves.
2. Corpus completeness: every human move in the 2026-10 scrape matches
   exactly one generated move (report by action family).
3. Auto-router (L/LP local, E, P+/pullman, 2P, destination bonuses,
   English Channel/France, London) + router check: engine optimum ≥
   every recorded `run_routes` revenue, equal in the large majority
   (share `revenue.rs::price_bucketed_stops` where it fits).
4. Variants (1822CA/MX/PNW) subclass this: see "What variants need".

## What variants need (1822CA / MX / PNW)

Each is a Ruby `G1822::Game` subclass; port each as a new `GameTitle`
with `is_g1822_family() = true` and its own data module, then branch in
`crate::g1822` where Ruby overrides a method. From the census (bridge
design appendix A): CA adds `assign` and mixed 1-/2-slot cities (the
`city_sizes_maintained` check is already ported), MX adds `remove_token`
and `higher_cert_limit`, PNW adds `assign`, `starting_packet` and its own
regional rules. The step machine, graph, tile rules and share pool are
title-neutral; the constants in `g1822/mod.rs` (company ids, bid-box
sizes, private lists) should move behind a per-title table first.

## Known issues / divergences

* Lanes are not modelled in the graph (`lanes:2` parses as one path):
  reachability is unaffected on this map; route-overlap legality is.
* Mail-contract subsidies price the route's first/last stop hexes by their
  best node (exact except a city+town hex like D35 when a non-L train stops
  at its town).
* The old corpus mostly predates current Ruby rules; Ruby HEAD rejects
  those records itself (bucket `ruby_rejects_same` in the audit once
  `tests/g1822_ruby_head_results.json` exists).

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
# per-commit gates: see docs/g1867_remaining_work.md (cargo test, pytest,
# parity_runner --random 0:100, tests/g1867_corpus_audit.py --limit 300).
```
