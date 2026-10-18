# 1867 — status checkpoint & remaining work

*Checkpoint updated 2026-10-18 (self-play readiness: the rules side of
the RL-readiness contract — impassable borders from title data, the
auto-router, the legal-move generator, fuzz, corpus completeness and Ruby
replay of fuzz games). Earlier checkpoints: 2026-10-15 (rules fidelity,
the replay seams below), 2026-08-02 (merged to `master` the same day; the
local history was rewritten then — commit hashes in older docs/memory
predate it; seam commits are identifiable by subject,
`git log --oneline --grep=1867`). This is the resume-point doc for the
1867 port; architecture rationale lives in `docs/multi_title_roadmap.md`
and `docs/rl_bridge_design.md`, and every seam commit carries a dense
description, including how it was confirmed against Ruby.*

## Where things stand

**Rules engine: 1867 replays the human corpus as Ruby HEAD does.** All
four Ruby fixtures (`tests/fixtures/1867/`) replay completely with exact
recorded scores (`tests/test_g1867_fixture_replay.py`). The 300-game
sample (`tests/g1867_corpus_audit.py --limit 300`):

| bucket | 2026-08-02 | **2026-10-15** | meaning |
|---|---|---|---|
| outcome_match | 114 | **206** | full replay, exact recorded scores |
| recorded_drift | — | **60** | full replay, scores ≠ record but **= Ruby HEAD's full replay** (see below) |
| result_mismatch | 64 | **3** | full replay, scores ≠ record; Ruby HEAD can't replay these to compare |
| rejected | 95 | **4** | replay stops on a rejected action (all explained below) |
| not_finished | 27 | 27 | corpus file status `active`/`new` — **never replayed** (abandoned games; not an engine bucket) |

(`--ruby-results ''` gives the raw recorded-only view: 206 / 63 mismatch
/ 4 rejected / 27.) Beyond the sample: the 2026-10 scrape (367 RECENT
games) is 317 match / 19 rejected / 0 mismatch — all 19 rejections are
2-player games (the unimplemented 2p variant, item 4 below); the full
1,027-file 2020-22 corpus is 598 match / 108 mismatch (60 of them in the
sample are Ruby-verified drift) / 84 rejected (60 of them 2-player) /
154 grid_market skipped / 83 unfinished.

**Ruby-HEAD verification of the whole sample** (`scripts/g1867_ruby_diff.py
--all-replayable 300`, Ruby clone 46a592e, 2026-10-07): per-file-action
state (corp price/cash, player cash, bank, round type) of all 273
replayable games shows **no Rust-vs-Ruby divergence on any file action
both engines reach**. 203 games replay fully in Ruby HEAD: our final
scores equal Ruby's in **all 203** (143 match the record too; the other
60 are recorded-result drift: old server scoring, e.g. before the
loan-drop change). Game end lands on the same action in all 176 games
Ruby ends by rule or not at all; the other 27 are `end_game` manual ends
(the shared `filter_actions` strips that action), whose scores still
match. The 70 games Ruby HEAD can't finish stop on the pre-#9655 merger
order or on rules the server changed since (Montreal token slots, tile
rotations, token-merger pricing — see the known issues).

What exists mechanically: single-item opening auction; SR bid-founding +
home-token choice + class-typed par ladder (majors z+p, minors x+p);
minors/majors with incremental cap (treasury shares at market price);
flat-top map + two-lay rule + distance-priced tokens + $80 water
borders + Y/O hex labels; loans/interest/LoanOperations/
BuyCompanyPreloan with AutomaticLoan in Track and BuyTrain; engine-
computed route revenue for bare routes (a recorded revenue is
authoritative, as in Ruby); merger round (convert/merge/
PostMergerShares/ReduceTokens, presidency changes while dealing);
CN nationalization; per-OR train export + per-title GAME_END_TIMING
priority; RedeemShares; Ruby `city_map` token transfer; loan-adjusted
valuation + end-game loan settlement; 1867 share-price movement (block
sells by presidents only, minors capped at $165, in-place moves keep cell
order).

Since 2026-10-18, for self-play (base rules, 3-6 players): impassable
borders from title data, the engine's own router and a legal-move
generator, fuzzed and checked against Ruby HEAD — next section.

## Self-play readiness (2026-10-18): legal-move generator, router, verification

The rules side of the RL-readiness contract (`docs/rl_bridge_design.md` §6
T1-T6, §7) is in place for the base rules at 3-6 players. Self-play can
drive 1867 through `BaseGame::g1867_legal_moves()` (Python:
`g1867_legal_moves_json(policy=None)`) and the engine's own router.

| contract item | status (numbers of 2026-10-18) |
|---|---|
| 1. every step x action type generated or replay-only | `g1867::legal::g1867_generator_status` (Python `g1867_generator_status_json`; `tests/test_g1867_legal_moves.py`); replay-only list below |
| T1 impassable borders from title data | `title::title_walls` for ALL titles (`BaseGame.walls`); 1830 byte-identical (test), 1867's ten walled hexes pinned |
| T2 router | `g1867/router.rs` (`g1867_auto_routes`); router check below |
| T3 tile lays | `g1867/tile_rules.rs` + the Ruby graph walk `g1867/graph.rs` |
| T4 tokens | distance price, Ruby `home_token_locations`, ReduceTokens choices |
| T5 trade-ins | phase 8 only: generator AND handler (`GameTitle::trade_in_allowed`) |
| T6 states humans never reach | failed merge = GameError; merges/conversions that cannot finish not offered; no panics reachable (fuzz) |
| 3. fuzz | in-crate (`cargo test g1867::fuzz`) + wire (`scripts/g1867_fuzz.py`) |
| 4. corpus completeness | `scripts/g1867_completeness.py` |
| 5. Ruby replay of fuzz games | `scripts/g1867_fuzz_ruby_check.py` |
| RlSpec / layout fingerprint (contract items 2, 6) | NOT done — the RL bridge is on hold |

### Verification numbers

* **Router check** (`scripts/g1867_router_check.py --ruby`, Ruby HEAD
  a600e13): every recorded human `run_routes`, engine optimum vs the record.
  2026-10 scrape (316 in-scope games, 23,047 runs): 21,368 equal (92.7%),
  1,676 above — Ruby HEAD validates and prices ALL 1,676 engine runs at the
  engine's total and per-route prices — 3 below: 2 human runs Ruby HEAD
  itself rejects (NoToken: 255086@324, 259821@517 — the record runs over
  track laid later), 1 stale record (258152@367: Ruby prices the human run
  at the engine's 70). Old corpus, first 300 files (273 checked, 16,535
  runs): 11,651 equal, 4,884 above (3,526 Ruby-confirmed; 1,358 sit in 65
  old games Ruby HEAD cannot replay that far), 0 below. Time per call:
  mean 1.9 ms, p99 30 ms, max 0.85 s (dense late maps with a 5+5E).
* **Completeness** (`scripts/g1867_completeness.py`, the rules' move set,
  every human decision except `run_routes`): 2026-10 scrape (316 in-scope
  games) — 158,897 of 158,900 decisions match exactly one generated move
  (**99.998%**), none matches two, none is missing; the 3 others are
  out-of-turn post-merger passes (replay-only, below). Old corpus, first
  300 files: 116,472 of 116,748
  exact (99.76%); 269 misses in documented replay-only classes (212
  pre-#9655 merger order, 54 MajorTrainless decisions out of queue order,
  3 out-of-turn dealing); the other 7 are old records Ruby HEAD rejects at
  the same decision or that lie beyond its replay horizon (4 home tokens in
  Montreal/Toronto — Ruby HEAD's `home_token_locations` there equals ours,
  e.g. 21494@776 Ruby offers M9/E13/D2 only; 23812@607, 25757@651; 24437@899
  and @929). Cross-player policy: the title's policy would exclude 2 human
  decisions in the scrape (company purchases from another player) and 2 in
  the old sample (trains from another player's corporation).
* **Fuzz, in-crate**: 3-6 players x restricted + rules policies, every
  generated move applies, no panic, no stuck state (`cargo test
  g1867::fuzz`; `perf_walk` is the profiling target). ~2 ms per decision
  with the router.
* **Fuzz, wire + Ruby HEAD** (`scripts/g1867_fuzz.py` then
  `scripts/g1867_fuzz_ruby_check.py`, final round): 140 walks — 3-6
  players x seeds 1-25 under the title's policy and x seeds 26-35 under the
  rules' policy, ~160,000 moves, every game played to its end, no
  generator error. Ruby HEAD replays all 140: 139 with no state divergence
  and no rejection, 1 meets Ruby's orphan-train defect (below: Ruby offered
  a train of a nationalized major at $1; the engine had dropped it).
  Earlier rounds found every bug listed under "Engine fixes"; with the
  driver taking the dead-end choices (`--allow-ruby-dead-ends`) 21 of 140
  games run into Ruby's stale-operator dead end.
* **Determinism**: the move list of a state is sorted (Debug order of step
  and action), so a seeded walk is reproducible across processes (the
  sort changed the walks: 12 fresh ones, 3-6 players x seeds 1-3, all
  replay clean in Ruby HEAD).

### Generator design

* `steps.rs::step_offers` (read-only, title-neutral) hands the shared step
  machine's offers — the title's step list, the blocking step and each
  active step's action types, every type attributed to the step the
  dispatch gate gives it to — to per-(step kind, type) generators. Every
  candidate is applied to a clone unless the generator already checked what
  the process checks (tile lays). Moves carry 18xx.games wire dicts
  (`g1867_move_wire`; Ruby's preprinted tile ids "<hex>-<index in its
  map.rb HEAD list>", `title::g1867::PREPRINTED_GROUPS`) and, for prices,
  a `PriceRange` (auction bids $5 grid up to cash — dutch mode only the
  asking price; minor bids $100+ on the $5 grid; a company for $1..face
  within cash; another corporation's train for $1..cash).
* Ruby player-time checks the generator applies (replay trusts records for
  them, like Ruby's `@game.loading`): tile legality (upgrades_to?, rotations
  with StubsAreRestricted and `hex.neighbors`, two-lay TILE_LAYS, costs
  within full buying power, the M13 blocker, max-exit tiles on city hexes
  — a UI rule, check_track_restrictions! on the tile as laid); Ruby's graph
  walk for lay targets, token connectivity and merge candidates (Ruby
  visits a path once per token walk: track reachable only by re-crossing a
  walked hexside is NOT connected — the replay graph map.rs is undirected);
  `home_token_locations` (minors: a city with no tokens; majors: no normal
  token on the hex, preferring hexes whose track reaches no node — Ruby's
  `Path#walk` with edge counters); trade-ins in phase 8 only.
* Not offered although Ruby accepts them, because Ruby then cannot go on
  (dead ends a player escapes only with undo): a minor bid with no home
  location (HomeToken waits forever); a merge add / done-adding pass / convert
  from which no major can be named (`finish_merge_to_major` raises "no player
  can become president"; every major started); the pass or buy that leaves
  the merged major's presidency in the treasury (`@broken_merge`, now a
  GameError); a 5+5E traded for a 2+2 ($600 - $750: Spender#spend raises).

### Replay-only (accepted from records, never generated)

* `bankrupt` — Ruby 1867 has no Bankrupt step (a corporation that must buy
  can always afford a depot train with loans; otherwise it passes into
  nationalization).
* BuyTrain `sell_shares` — 1867's EMR never sells shares.
* MajorTrainless pass/choose by a queued major other than the first (Ruby's
  `actions` accept any queued major; the generator decides in queue order).
* PostMergerShares buy/pass by a player other than the active one (Ruby's
  `actions` ignore the dealing order).
* The pre-#9655 merger order (dealing before ReduceTokens; the dispatch
  accommodation below) — enumeration stays canonical.

### Cross-player purchases (`GameTitle::generator_cross_player`)

Default `RESTRICTED` (no title override): the generator drops (1) a train
bought from a corporation (major or minor) another player presides, and
(2) a private bought from another player (1867 companies are bought for
$1..face; the seller is paid). "Other player" = the seller is not the
buying corporation's president; a corporation without a president counts
as another player. `policy="rules"` gives Ruby's set (completeness, fuzz).
If the policy would leave no move, the rules' set is returned (counter
`legal::stats::CROSS_FALLBACK`). Not affected: depot trains, treasury and
pool shares, the post-merger dealing (treasury shares). Open questions
for the user (left as Ruby has them):

* **Mergers of other players' minors** (Merge step): a minor may merge with
  any connected minor, whatever its owner; each owner gets a 10% share of
  the new major (no cash changes hands). Treated as NOT a purchase — offered.
  Should self-play only merge minors of one player?
* **Conversion targets**: the converting minor's owner picks any unstarted
  major — no other player involved.
* **Nationalization payouts** (MajorTrainless / interest default) move cash
  from the bank, not between players.

### Engine fixes found by the fuzz (each probed against Ruby HEAD)

Replay-neutral: the 300-game sample (206/60/3/4/27), the 2026-10 scrape
(317/19) and the full corpus (598/60/48/84/154/83) audits are unchanged.

1. A live SR minor auction's acting player is the next bidder (enumeration
   had no blocking step there; dispatch by the action's entity was fine).
2. While the MajorTrainless queue is non-empty the OR stays on the current
   operator (`or_advance_to_next_corp` parks it at Done) — Ruby
   `after_process` returns while a step is active. We used to end the round
   or start the next turn before the choices.
3. A MajorTrainless nationalization of the current operator ends its turn
   (`nationalize!` → `force_next_entity!`).
4. The OR-end export of the first 8: when the crowded minor Ruby's
   DiscardTrain names as current entity closes, force_next_entity! runs
   G1867 BuyTrain#pass!, nationalizing the round's last operator if the
   rust left it trainless.
5. The first 8 bought while a crowded minor closes: the purchase's trailing
   `pass!` (only when the buyer can no longer buy) lands on the NEXT
   operator, nationalized if trainless before `post_train_buy`.
6. Trade-ins only in phase 8 (handler gate).
7. 1867 construction with 7+ players raises ValueError (was a panic).

### Ruby HEAD dead ends and defects the engine goes past (documented, not emulated)

* **Orphan trains** (a decision for the user): a major nationalized while
  it still owns trains (interest it cannot pay, LoanOperations) is reset by
  `reset_corporation`, which leaves the OLD object owning its trains — and
  `Depot#other_trains` keeps offering them to every other corporation (the
  price goes to the phantom, out of play). The engine drops a reset major's
  trains. The fuzzer flags the point (`rl18xx.ruby_orphan_trains_at`).
  Emulating it would mean trains for sale by nobody.

* **Stale operator after a MajorTrainless nationalization** with other
  majors still queued: Ruby does not force the next entity, and
  `reset_corporation` swaps a new object into `@corporations` while the
  round's `@entities` keep the old one as operator — no action by id ever
  matches it again ("Blocking step Lay/Upgrade Track cannot process action
  ..."). The engine ends that turn (item 3). The wire fuzzer passes instead
  of taking such a choice (`g1867_ruby_dead_end_choice`, `--allow-ruby-dead-ends`
  keeps them) so Ruby can replay whole games.
* **Stale @corporations after an in-loop reset** (item 4): Ruby's
  `event_minors_nationalized!` ends with `@corporations = corporations`
  (its partition taken before the loop), restoring the pre-reset object of
  the operator it just nationalized; that phantom then sits in the
  trainless queue and never matches an action. We don't emulate the phantom.
* **SingleItemAuction with nobody able to bid**: when a lot comes up that
  no player can bid on until its price reaches $0, the recursive dutch
  fallback fakes a bid by `@active_bidders.first` == nil and crashes
  ("undefined method `cash' for nil"). The engine gives the lot to the
  first player at $0. The wire fuzzer keeps $5 back once every other player
  is below $5.

## Replay seams of 2026-10-15 (one commit each, all Ruby-probed)

1. **Water borders** — `border=...type:water,cost:80` charged by the lay
   that completes the crossing (both tiles target each other), once.
2. **Share price** — `:down_block_pres` (one step left per sale, only a
   president seller); minors don't move right from the `m` cell.
3. **Closed minors** can't be re-bid; minor bids respect the cert limit
   (SR-end round flow).
4. **Track affordability** uses full buying power (AutomaticLoan).
5. **Recorded route revenue is authoritative** (Ruby `@revenue ||=`).
6. **Treasury shares** are offered at the market price
   (`always_market_price`).
7. **White-hex labels** Y (E17/D16/O7/J12) + J12's future label O.
8. **Minorless merger round** waits for a crowded corp's discard.
9. **In-place market move** keeps the cell order (operating-order ties).
10. **"par" offered from `min_price`** ($50) — Ruby's can_ipo_any? quirk.
11. **ReduceTokens frees the removed city's charter token** (two tokens of
    one survivor on Montreal) — matches Ruby HEAD's post-2025-05-18 token
    merger (tobymao/18xx ea4e57e, #11750); see known issues for the 8 old
    full-corpus games this rejects, exactly as Ruby HEAD does.
12. **GAME_END_TIMING_PRIORITY per title** — 1867's final OR set outranks
    a bank break.
13. **First 8-train force-ends the buyer's turn** when a crowded minor
    closes (Ruby `close_corporation` → `force_next_entity!`).
14. **Post-merger dealing moves the presidency** to a larger holder.

## Remaining work (ordered by expected yield)

1. **The RL bridge — the actual goal** (on hold by the user's decision).
   1867 is still `alphazero_bridge_ready() == false`. Needed: per-title
   action layout (the generator's families: single-item and corporation
   bids with price ranges, par cells, buy/sell bundles, two-lay tile lays,
   distance-priced tokens, home tokens, three discounted trains + trade-ins,
   merge / convert / remove_token / MajorTrainless `choose`), encoder spec,
   native decode, MCTS wiring, VALUE_SIZE/POLICY_SIZE plumbing, RlSpec and
   the layout fingerprint (contract items 2 and 6). Constraint: the 1830
   layout is FROZEN (checkpoint-compatible). Before that: decide the open
   cross-player questions above.
2. **Generator speed for self-play**: ~2 ms per decision in the fuzz
   (tile lays dominate: one Ruby graph walk per upgrade candidate; offers
   recompute the Track gate). Fine for fuzzing; an MCTS hot loop would want
   the walk cached per state and the clone-validation of share moves
   replaced by direct checks.
3. **Router**: exact on the corpus with a 20M-step safety budget per walk
   (the corpus peaks at 1.6M normal / 5.8M 5+5E); denser self-play maps may
   hit it with a 5+5E (the result stays legal, `exact` false — counter
   `g1867_router_counters()`). The REPLAY pricer (`recorded_route_revenue`
   / `revenue::classify_stop`) treats a stop as tokened if any city on its
   hex is; Ruby (and the router) check per city — it only matters for a
   5+5E on Toronto/Montreal and never showed on the corpus.
4. **2-player variant** (`TODO(1867-2p)`): 60 of the full corpus's 84
   rejections and all 19 of the 2026-10 scrape's are 2-player games
   rejected at the opening auction ("MB is not up for auction") — Ruby
   removes two random privates (seeded by `settings.seed`; needs Ruby's
   RNG semantics), trains per `TRAINS_REMOVE_2_PLAYER`, 70% ownership cap.
   Out of the user's scope (3-6 players).
5. **`grid_market`** optional rule (154 full-corpus games skipped; market
   data present, unplumbed; 1867 declares no `GameTitle::optional_rules`).
6. **Small open replay quirks** (documented with probes, not emulated):
   - Ruby's stale `@crowded_corps` memo after the first 8 lets a CLOSED
     minor (Ruby keeps its unbuyable trains, CLOSED_CORP_TRAINS_REMOVED)
     record a later `discard_train`, reclaiming the train to the discard
     pile — 38753 file id 516 ("TN discards 5"); we drop a closed
     minor's trains at once. 1 full-corpus game (48699 likely the same,
     past Ruby's horizon).
   - The 4 sample rejections: 23812 (file id 607) and 25757 (651) are
     rejected by **Ruby HEAD at the same file action**; 24437 (Ruby HEAD
     stops at 899, we at 929) and 29948 (Ruby HEAD stops at 300 on the
     pre-#9655 order, we at 309: an old-order PostMergerShares/
     ReduceTokens interleaving where our dealing state still expects a
     player) are past Ruby HEAD's replay horizon — unverifiable.
   - The 3 unverifiable sample mismatches (21388, 23223, 28941) all stop
     in Ruby HEAD on the pre-#9655 merger order.
   - The SR home-token handler (`process_stock_home_token`) still only
     checks for a free slot (old records put homes where Ruby HEAD no
     longer allows them); the generator applies `home_token_locations`.
7. Then the other titles (`docs/multi_title_roadmap.md`).

## Known issues / failure points (read before resuming)

- **Old records vs current rules.** Ruby HEAD rejects some 2020-22
  records on its own (pre-#9655 merger order; Montreal token slots; tile
  rotations; the pre-2025-05-18 token-merger bug that handed merged
  majors an extra $0 token). Our engine follows CURRENT rules: e.g. the
  ReduceTokens fix (seam 11) makes 8 old full-corpus games reject a
  later buy/token for want of cash — Ruby HEAD rejects 3 of them at the
  very same file action (69817@428, 32537@630, 69520@610), the other 5
  are past its horizon. Before "fixing" a corpus rejection, run
  `scripts/g1867_ruby_diff.py` on it: if Ruby HEAD rejects the same
  action, it is an old-record incompatibility, not our bug.
- **Pre-#9655 merger-order accommodation**: current rules order the
  merger round ReduceTokens → PostMergerShares (GTG errata,
  tobymao/18xx#9655); the 2020-21 corpus recorded the OLD order. Our
  dispatch gate admits both orders *while both phases pend*;
  enumeration/self-play stays canonical (ReduceTokens first).
- **Recorded results drift**: the audit's `recorded_drift` bucket uses
  the committed `tests/g1867_ruby_head_results.json` (game id → Ruby
  HEAD's final scores by seat, no player data; regenerate with
  `scripts/g1867_ruby_diff.py --all-replayable 300 --results-json ...`
  when the Ruby clone moves). Result keys in corpus files can also be
  stale player names — `align_recorded_keys` handles that.
- **Action-id mapping when probing**: corpus `id`s are renumbered by
  `pretraining.filter_actions`. `scripts/g1867_ruby_diff.py` tags every
  filtered action with its originating FILE id and keys both engines'
  snapshots by it (Ruby via `Game#process_to_action`). Never compare by
  the filtered index.
- **Walls** come from title data (`title::title_walls`, `BaseGame.walls`):
  `passable_exits_for`, the lay frontier and the 1867 graph/router read
  them. Only `type:impassable` borders are walls (a no-type border is a
  drawn line). 1867's walls are mirrored on both sides, so the shared
  geometric adjacency (map.rs, router) can never cross one with legal track.
- **Two graphs**: replay uses the Python-port `map.rs` walk (undirected
  reachability); the generator uses `g1867/graph.rs`, Ruby's Graph#compute
  (each path once per token walk, edge counters, converging exits). They
  differ exactly where Ruby's player-time checks are stricter than a
  record replay needs.
- **Shared-code change reachable by 1830** (seam 9): `update_market_cell`
  no longer re-appends a corp "moving" in place on the top-left cell. In
  1830 that only happens on a sold-out bump of a corp on the $60 cell;
  the new behaviour is exactly the Python oracle's. Verified with
  `parity_runner --human` on an 81-game 1830 sample (0 failures) on top
  of the standard gates.
- **Trainless-major freeze**: while `trainless_major` is non-empty every
  round is frozen on its current entity — since 2026-10-18 that includes
  the OR's current operator, whose turn may be over (parked at Done) but
  the next operator only starts once the queue drains (Ruby
  Operating#after_process). **Cheater token slots** (Ruby `city_map` can
  append tokens past `normal_slots`). **1830 safety**: all 1867 behaviour
  sits behind `GameTitle` hooks with 1830-inert defaults or 1867-only
  state (trainless queue, phase-8 trade-ins); before long 1830 training
  runs on a new engine build, run the full-corpus ritual in
  `docs/verification_rituals.md`.

## Ruby probe rig

- `git clone --depth 1 https://github.com/tobymao/18xx.git logs/ruby/18xx`
  (inside the worktree's ignored `logs/`; read-only use) and the
  `ruby:3.2-slim` image (`docker pull ruby:3.2-slim` if missing).
- `scripts/g1867_ruby_dump.rb` dumps a game per file action in Ruby;
  `scripts/g1867_ruby_diff.py` runs it in docker (persistent gem dir
  `logs/ruby/gems` for `require_all`, Ruby dumps cached in
  `logs/ruby/dumps`), replays the same game in Rust, and prints the first
  divergence, either engine's rejection, final-result and game-end
  agreement. Typical: `--audit <audit.json>` (its failing games),
  explicit ids, or `--all-replayable 300`.
- `scripts/g1867_ruby_probe.rb <game.json> <file id> <snippet.rb>`:
  load to a file id in docker and eval a snippet against the live game
  (`g`): states, logs, `caller` traces of monkeypatched methods (how the
  force_next_entity quirks of this session were found). Never print
  player names from human games.
- For a targeted look, load at a FILE id and inspect:
  `Engine::Game.load(data, at_action: id)` then `g.process_to_action(id2)`
  and `g.log` — monkeypatching e.g. `Round::Operating#next_entity!` with
  `caller` found the force_next_entity quirk (seam 13).

## Gates to keep green per commit (the working rhythm)

```bash
# inline env — shell state does not persist:
cd engine-rs && PYO3_PYTHON=$PWD/../.venv/bin/python \
  LD_LIBRARY_PATH=$(../.venv/bin/python -c 'import sysconfig;print(sysconfig.get_config_var("LIBDIR"))') \
  RUSTFLAGS="-L $(../.venv/bin/python -c 'import sysconfig;print(sysconfig.get_config_var("LIBDIR"))')" \
  cargo test --release --no-default-features          # 209 passed / 2 ignored on 2026-10-18
CUDA_VISIBLE_DEVICES="" uv run pytest tests/          # 1161 passed / 1 skipped on 2026-10-18
CUDA_VISIBLE_DEVICES="" uv run python tests/parity_runner.py --random 0:100
CUDA_VISIBLE_DEVICES="" uv run python tests/g1867_corpus_audit.py --limit 300 --json /tmp/audit.json
#   → diff game-by-game vs the previous audit JSON; rejections may only
#     move DEEPER or convert; full replays may not start mismatching
#     (a new mismatch must be shown Ruby-HEAD-equal with g1867_ruby_diff.py).
#     Also worth a look: the 2026-10 scrape (recent games, Ruby HEAD
#     replays them) and the full corpus.
# Shared-code changes reachable by 1830: tests/parity_runner.py --human on a
# few hundred 1830 games (read its --out JSON summary).
```

Self-play verification (after generator / router / rules changes):

```bash
# generator: wire fuzz (records games) + Ruby HEAD replay of them
CUDA_VISIBLE_DEVICES="" uv run python scripts/g1867_fuzz.py --players 3 4 5 6 --seeds 1-25 --out /tmp/fz [--policy rules] [--timing]
RUBY_DOCKER_CPUSET=40-63 CUDA_VISIBLE_DEVICES="" uv run python scripts/g1867_fuzz_ruby_check.py /tmp/fz --jobs 4
# completeness (cap each worker's memory, e.g. ulimit -v 16000000)
CUDA_VISIBLE_DEVICES="" uv run python scripts/g1867_completeness.py --glob 'human_games/scrape_2026_10/1867/*.json' --jobs 12
CUDA_VISIBLE_DEVICES="" uv run python scripts/g1867_completeness.py --glob 'human_games/1867/*.json' --limit 300 --jobs 12
# router check (+ --ruby: Ruby HEAD validates every engine run above the record)
CUDA_VISIBLE_DEVICES="" uv run python scripts/g1867_router_check.py --ruby --jobs 6
```
