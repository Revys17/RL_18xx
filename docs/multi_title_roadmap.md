# Multi-Title Support Roadmap (1867 & 1822/1822CA)

Plan for extending the engine beyond 1830 to the other 18xx titles we want to train agents
for: **1867: The Railways of Canada** and **1822 / 1822CA: The Railways of Great Britain
(+ Canada)**.

## STATUS (2026-06-12) — start here

- **Branches:** 1867 work happens on `multi-title`, 1830 training fixes on `master`,
  rebase periodically. Phase 0 is merged to master; Phase 0.5 lives on `multi-title`
  (commits `ddeed22..c13dec6`).
- **Phase 0 is DONE and merged to master** (ability system + step/round machinery, both
  verified 1830-behavior-preserving — see "Phase 0 outcomes & carry-forwards" at the
  bottom). The engine is also now *validating* (rejects malformed/ill-timed actions like
  Python does) and the 1830 verification rituals are documented and gated
  (`docs/verification_rituals.md`; pre-push hook runs pytest + cargo incl. the
  frozen-1830-action-layout test).
- **Phase 0.5 is DONE on `multi-title`** (2026-06-12): all seven seams landed as
  separate 1830-no-op commits, each green on the fast gate (cargo 91 + pytest 326) plus
  strict random-walk parity slices; full-corpus 4-axis lockstep re-run at the end. See
  the checked-off list below for what each seam produced.
- **Next up: Phase 1 (1867)** — validation harness first (fixture replays + 18xx.games
  corpus), then data translation, then mechanics. The `GameTitle` trait + round-flow
  hook give g1867.rs its plug-in points.
- **Validation strategy for new titles is fixture/corpus-based, NOT dual-engine** — the
  Python engine stays 1830-only (see "Per-title validation strategy" below). This
  supersedes the older "every title needs Python-vs-Rust parity" language.

Reference rules: the Ruby source at
[tobymao/18xx](https://github.com/tobymao/18xx/tree/master/lib/engine/game) —
[`g_1867`](https://github.com/tobymao/18xx/tree/master/lib/engine/game/g_1867),
[`g_1822`](https://github.com/tobymao/18xx/tree/master/lib/engine/game/g_1822),
[`g_1822_ca`](https://github.com/tobymao/18xx/tree/master/lib/engine/game/g_1822_ca).

## Branching strategy

Do this work on a **dedicated long-lived branch** (e.g. `multi-title`), kept separate from
the primary MCTS/1830 training work on `master`.

- The 1830 self-play/training runs must stay safe: the title abstraction (Phase 0) touches
  hot-path code (`BaseGame::new`, the encoder, the action layout), and a regression there
  silently corrupts training data.
- `master` stays the source of competitive 1830 training. Merge `multi-title` → `master`
  only once the refactor is proven 1830-behavior-preserving (parity tests green) — ideally
  land Phase 0 first as a self-contained, no-op-for-1830 merge, then develop each title on
  top.
- Rebase the branch on `master` periodically so engine bug-fixes from active 1830 work flow
  in. Avoid the reverse (don't let in-progress title code leak onto `master`).

## Guiding principle: one engine, not a fork

Keep a **single engine** with a title abstraction + per-title data/step modules — do **not**
fork the repo per title. This mirrors upstream tobymao/18xx, which runs 100+ titles in one
engine via a shared base + per-title `step/` and `round/` modules. Forking would duplicate
the hardest, most valuable code (router, tile catalog, hex geometry, graph) into N diverging
copies.

There are two layers with opposite reuse profiles, and the plan treats them differently:

| Layer | Files | Reuse across titles |
|-------|-------|---------------------|
| **Rules engine** | `router.rs`, `tiles.rs`, `map.rs`, `graph.rs`, `core.rs`, `rounds/*`, `mcts.rs` (+ Python equivalents) | High — keep unified behind a `GameTitle` abstraction |
| **AlphaZero bridge** | `encoder.rs`/`encoder.py`, `action_index.rs`/`action_mapper.py`, the models | None — inherently per-title (different action space, observation vector, and a separately-trained network; no weight transfer) |

So "unify vs fork" only really applies to the rules engine. The RL bridge is per-title
regardless — implement it as per-title submodules in the same repo.

## Current state (1830-only) — what blocks a second title

The 1830 *data* is already cleanly data-driven (`engine-rs/src/title/g1830.rs`,
`rl18xx/.../game/title/g1830.py`): corporations, companies, trains, phases, market grid, hex
map, tile counts. Translating new data from the Ruby files is mechanical.

~~The original blocker list (no step abstraction, sym if-chain abilities, hardcoded round
sequence) is RESOLVED by Phase 0.~~ What still blocks a second title, as of 2026-06-11:

- **Title dispatch funnel not yet a trait.** Per-title data/step-lists/round-cycle are
  single-sourced but hardcode g1830 at exactly three points
  (`steps.rs::{round_step_descs, operating_step_descs, round_cycle}`) plus
  `BaseGame::new()`'s direct `g1830::*` calls and the `"1830"` title string. The
  `GameTitle` trait goes there (Phase 0.5).
- **Round-cycle veneer.** The `RoundKind` cycle is a static list that cannot express
  1867's merger rounds interleaved *within* the OR set, and OR-set repetition bypasses
  the cycle (early return in `transition_to_next_round`). Needs the per-title round-flow
  hook (Phase 0.5 — see carry-forwards).
- **10-share assumption.** Share creation hardcodes 1 president @20% + 8 @10%; dividend
  math uses `total_shares = 10` (`rounds/operating.rs`); SellShares slot blocks and
  percent reconstruction assume 10% units (`action_index.rs`/`decode.rs`). 1867 mixes
  5- and 10-share corps.
- **60% float + full capitalization** hardcoded (`entities.rs` check_floated; the
  dividend/capitalization split in `rounds/operating.rs`). 1867 floats differently and is
  incrementally capitalized.
- **Stock market movement policy.** `StockMarket::new_1830()` + 2-D movement baked into
  `core.rs` `move_left/right`; 1867 uses a 1-D market.
- **Phase/train data partially unwired.** `TrainDef.rusts_on` exists in title data but
  rusting/close/phase-trigger maps are re-hardcoded in `game.rs`; `PhaseDef` lacks
  `status`/`events` (can_buy_companies, close_companies). Train model lacks
  obsolescence/export and bucketed distances (1822's L-trains; 1867 trains are scalar so
  this can wait).
- **Fixed action space + encoder.** `action_index.rs` pins `POLICY_SIZE = 26537`. The
  layout builder is well-factored (offsets computed from table lengths in one place,
  pinned by the frozen-layout cargo test) so per-title re-derivation is cheap — but
  `config.py` carries the literal 26537 twice, `encoder.rs` keeps its own const copies of
  title data (corp/private ids, train counts, cert limit, starting cash, tile counts),
  and `mcts.rs` hardcodes `VALUE_SIZE = 6` and re-lists layout tables in
  `price_head_entity_key` / `price_grid_step` ("Bid" → 5 is 1830's increment).
- **Absent mechanics** (needed by the new titles): minors, mergers/conversions, loans &
  interest, share issue/redeem, destination tokens, concessions, bidbox auction. No
  `Minor` entity exists. These are Phase 1/2 net-new code — the step machinery gives them
  a place to plug in, it does not write them.

## Relative complexity (Ruby `game.rb` / `entities.rb` size as a proxy)

| Title | game.rb | entities.rb | Custom steps / rounds beyond the generic base |
|-------|--------:|------------:|------------------------------------------------|
| 1830 | 6 KB | 7.5 KB | none (pure base steps) — why it ported first |
| **1867** | 37 KB | 11 KB | `merge`, `loan_operations`, `redeem_shares`, `post_merger_shares`, `single_item_auction`, `major_trainless`, `reduce_tokens`, `buy_company_preloan` + a **merger round**; minors→majors, mixed 5/10-share corps, CN formation |
| **1822** | 85 KB | 58 KB | `bidbox_auction`, `minor_acquisition`, `destination_token`, `issue_shares`, `acquire_company`, `special_track/token/choose`, `choose` + a **choices round**; ~30 minors, concessions, bonds/loans, L + permanent trains, phase-gated privates |
| **1822CA** | 32 KB | 69 KB | *extends `G1822::Game`* + `acquisition_track`, `assign_sawmill`, scenario data |

Takeaways: **1867** is the right second title (forces the abstraction + the
minor/major/merge machinery at moderate scale). **1822** is the deep end of commonly-played
18xx. **1822CA extends 1822**, so the 1822 base is a hard prerequisite for it.

---

## Phase 0 — Composable Step/Round layer + ability system ✅ DONE (2026-06-10, merged to master)

Landed as two 1830-no-op refactors verified at exact baseline on every gate; details and
commit ranges in "Phase 0 outcomes & carry-forwards" at the bottom of this doc. What
exists now: per-company `AbilityDef` data interpreted by generic code (no sym if-chains);
per-title ordered step lists + ONE shared `actions_for` accumulation loop +
table-driven `skip_steps`; `RoundKind` cycle description driving round transitions; the
legacy dispatch frozen as an in-crate differential oracle; a frozen-1830-action-layout
cargo test wired into the pre-push hook.

## Phase 0.5 — Pre-1867 seams ✅ DONE (2026-06-12, branch `multi-title`, `ddeed22..c13dec6`)

All seams landed as separate 1830-no-op commits, each verified on the fast gate
(+ strict random-walk parity slices for the money-math seams; full corpus 4-axis
lockstep at the end). What each produced:

- [x] **Per-title round-flow hook** (`ddeed22`): `RoundKind`/`round_cycle()` deleted;
  `FinishedRound`/`RoundStart`/`RoundTransition` + a per-title `next_round` FUNCTION
  (Ruby `next_round!` shape). OR-set repetition flows through the hook too (no more
  early-return). 1867 adds a `RoundStart::Merger` variant + its own flow fn.
- [x] **`GameTitle` trait** (`f2a6ef4`): the full title-dispatch funnel in
  `title/mod.rs` — machinery descriptions (step lists, round flow) + static data
  (corps/companies/trains/phases/hexes/preprinted DSL/tile counts/catalog/cash/cert) +
  `all_titles()` registry + `resolve()`; `BaseGame::title_def()` is the access point.
  Def structs + hex geometry moved to `title/mod.rs`. Ability lookups are per-title
  keyed (syms may collide across titles).
- [x] **Share-structure / float / capitalization** (`69b21cf`): `CorporationDef.shares`
  / `float_percent` / `Capitalization` (Full | Incremental); `Corporation` carries
  unit/float/cap with helpers; ALL unit math (dividends, buy/sell pricing, price
  drops, president dump/partial, par cost, bundles, decode count<->percent, factored
  percent emissions) uses the corp's unit. Incremental cap is `unimplemented!()` until
  1867 adds the buy-side treasury hook.
- [x] **Stock-market movement policy** (`22e3f0e`): `MarketMovement`
  (TwoDimensional | OneDimensional) + `market_grid()` on the trait;
  `StockMarket::new(grid, movement)`; 1-D markets pin at row ends, no vertical moves.
- [x] **Phase/train data wired** (`1f4332e`): `PhaseDef.on`/`status`,
  `TrainDef.events` (mirrors g1830.py exactly); check_phase_advance, rusting, the
  close_companies event, can_buy_companies gates and the discounted-train exchange all
  data-driven.
- [x] **Action layout + encoder per title** (`da28620`): `build_layout(title)` with
  per-title memoized layouts; city tables DERIVED by constructing initial tiles; par
  prices from the market grid; `SlotLayout.total` replaces derived POLICY_SIZE uses;
  config.py's two 26537s now `engine_rs.policy_size_py()`; `EncoderSpec` replaces all
  encoder const copies; `VALUE_SIZE` single-sourced from `encoder::MAX_PLAYERS`; Bid
  price-grid step from title data. NOTE: 1830's layout hex/tile orders are FROZEN
  Python-ActionMapper artifacts — `g1830::action_{hex,tile}_order()` override the
  trait's clean default derivations; new titles get the clean defaults.
- [x] **`next_operating_pc` duplicate-pc hazard** (`c13dec6`): the
  overlay+blocking-override (SpecialToken) pattern is the documented, ENFORCED shape —
  a registry-wide cargo test rejects any title listing two steps with one pc.
- [x] Green 1830 ritual after each seam (`docs/verification_rituals.md`).

Still intentionally 1830-pinned after Phase 0.5 (by design, for the 1867 RL layer to
parameterize when it exists): `mcts.rs` runs against the global 1830 `layout()`;
`action_index.rs`'s hardcoded company list inside the layout builder is gone but the
Python `action_mapper.py`/encoder remain 1830-only (per-title Python mirroring is a
Phase 1 RL-layer task); the PUBLIC constructors build only "1830" until a constructor
title argument lands with g1867 (in-crate code uses `BaseGame::build_titled`).

**Abstraction smoke-tested pre-1867 (`ee78ea9`):** a synthetic `TEST-5SHARE` title
(test-build registry only — 1830's map/data with 5-share corps in the 1867-major
shape `[40,20,20,20]` and a ≥2-OR flow hook, clean default action orders) runs
enumerate-apply random walks with cash-conservation asserted per action, pins the
5-share par/float math, and proves per-title isolation (every frozen-1830 test now
runs with two titles registered). g1867.rs gets working plug-in points + an executable
harness pattern from day one.

## Per-title validation strategy (supersedes "every title needs Python↔Rust parity")

The 1830 methodology — dual-engine lockstep against the Python oracle — does NOT scale:
there is no Python 1867 engine, and porting one (~the size of the Rust port itself) would
build a redundant engine solely to verify the first. **The Python engine stays
1830-only.** New titles are validated by:

1. **Ruby fixture replays (gold standard, few):** tobymao/18xx ships completed games with
   expected results per title under `public/fixtures/<title>/` (1867: 4 fixtures incl.
   nationalization edge cases; 1822: 4). Build a harness that replays a fixture's action
   stream through the Rust engine and asserts final scores + per-action acceptance. This
   is exactly how the Ruby repo regression-tests titles.
2. **18xx.games human corpora (volume):** download completed 1867 games via the same API
   used for the 1830 corpus; replay through the cleaning/import pipeline and assert
   outcome (final `result()`) against the recorded scores. This is the per-title analogue
   of the import-outcome audit — thousands of games, outcome-level assertions.
3. **In-crate tests + the frozen-oracle pattern:** every new step/mechanic gets unit
   tests; any refactor of shared machinery keeps the proven pattern of freezing the old
   implementation as a `#[cfg(test)]` differential oracle.
4. **Ruby source as line-by-line porting reference** (port the architecture's step
   boundaries; diff behavior against fixtures when in doubt).

Caveat to track: fixture + corpus replays assert recorded-game trajectories and final
scores, not full state at every step (weaker than 1830's compare_state lockstep). The
random-walk fuzzing (in-crate walks + enumerate-apply self-consistency: every enumerated
action must apply cleanly, mirroring the slot=None lesson) is the compensating control.

## Phase 1 — 1867: The Railways of Canada

**Validation harness FIRST:**
- [x] Fixture-replay harness (Ruby `public/fixtures/1867/*.json` → replay → assert final
  scores + per-action acceptance); download an 1867 human corpus from 18xx.games and
  stand up the outcome-replay audit. Build these before/alongside the first mechanics so
  every new step lands against an executable oracle. *(Done: tests/title_replay_harness.py;
  fixture gate runs as a prefix-watermark RATCHET — tests/test_g1867_fixture_replay.py —
  with the full-score assertion xfailed until the ruleset completes.)*

**Data translation (mechanical):** map, hexes, tiles, trains (incl. permanent), phases,
market, minors + majors, privates → `title/g1867.rs` (Rust only — no `g1867.py`; the
Python engine stays 1830-only per the validation strategy above). *(Done: title registers,
rules-incomplete — `alphazero_bridge_ready() = false` keeps the action-layout/encoder maps
1830+TEST-5SHARE only; tile catalog in tiles::tile_catalog_1867.)*

**New mechanics (the hard part):**
- [ ] `Minor` entity + minor operating/ownership; minors→majors conversion. NOTE: 1867
  minors have numeric-string ids ("1".."16") — the decode/entity-classification layer
  already resolves ids against game collections (fixed 2026-06-09), but watch for any
  remaining parses-as-int assumptions.
- [ ] **Merger round** + steps: `merge`, `post_merger_shares`, `reduce_tokens`,
  `major_trainless` — new `RoundKind` + `StepKind`s listed in g1867's descriptions,
  injected via the Phase 0.5 round-flow hook. **← THE frontier blocker
  (2026-06-12): all four fixtures now stop exactly at the first phase-3 merger
  round (watermarks 196/207/196/184; hs_ahjzadkh on a literal `merge` action —
  CV merging minors into the CNR major with `remove_token` follow-ups).**
- [ ] Variable share structures (5-share and 10-share corps) — depends on Phase 0.5
  parameterization.
- [x] Loans & interest *(done 2026-06-12: rounds/loans.rs take/auto-take/repay +
  `StepKind::{BuyCompanyPreloan, LoanOperations}` with their own pcs; interest is
  $5 × the OR-START loan snapshot (`OperatingState.interest_snapshot`, Ruby
  `calculate_interest`) paid at the LoanOperations position, taking loans to cover
  a shortfall, then FORCED repayment while cash ≥ $50, then re-snapshot; unpayable
  interest = loud `unimplemented!` until nationalization lands)*. Share
  `redeem`/issue still open (major-only — lands with majors/merger work).
- [x] **Engine-computed route revenue** *(done 2026-06-12: revenue.rs + the
  `GameTitle::recorded_route_revenue` hook — recorded `connections` chains →
  phase-priced stops → Ruby `compute_stops` bucketed allocation (towns fill spare
  pay, 5+5E offboards-only + tokened-stop) → multiplier / hex_bonus / Timmins-
  capital bonuses. Oracle: recorded revenues cross-check exactly — 28/28 fixture
  routes reached + 3668/3668 over 300 corpus games; pytest
  test_route_revenue_cross_check is the regression gate. 4-tier offboard revenue
  (green/gray) landed with it.)*
- [x] Company prices $1-to-face (`CompanyPriceUpToFace`) via
  `GameTitle::company_buy_price_range` *(2026-06-12)*.
- [x] Track-step affordability gate *(Ruby `can_lay_tile?`: blocks only while the
  next lay-allowance slot is usable and its cost ≤ FULL buying power — cash +
  takeable loans × $45; probed both directions on fixture 21268)* and
  preprinted-multi-city upgrade token mapping *(L12 Montreal → X3: exit-based
  city matching now also derives the PREPRINTED tile's per-city exits from the
  live tile — positional fallback misplaced home tokens, killing routes)*.
- [x] Embedded `auto_actions` replay *(harness: each recorded action's
  server-generated auto actions — programmed passes, BuyCompanyPreloan
  auto-passes for loan-free corps — are processed right after their parent,
  exactly like Ruby's `process_action`)*.
- [x] `single_item_auction` setup *(done: rounds/single_auction.rs — ascending + dutch
  fallback + $0 force-buy; whole opening auction replays on all 4 fixtures and 300/300
  sampled corpus games)*; — [ ] CN (national) formation / end-game.

**RL layer (per-title):**
- [ ] New encoder feature set + a per-title action layout (Phase 0.5's parameterized
  `build_layout()` fed with 1867 data → its own `POLICY_SIZE`); the Python
  `action_mapper`'s role for 1867 is index-layout mirroring only (training-target side) —
  scope it to what pretraining/self-play actually consume.
- [ ] `RustGameAdapter` audit for residual 1830-isms (it was de-1830'd with the ability
  work, but the encoder/cleaning surfaces it synthesizes were only ever exercised by
  1830).
- [ ] New model instance + a **from-scratch training run** (no warm-start from 1830).

## Phase 2 — 1822: The Railways of Great Britain (base mechanics)

The largest build. Data translation is big (entities ~58 KB), and the mechanics are novel:
- [ ] Concessions + ~30 minors; `bidbox_auction` (rolling bid boxes) and the **choices
  round**.
- [ ] `minor_acquisition` / `acquire_company`; minors → majors via concessions.
- [ ] Bonds / loans; `issue_shares`; phase-gated private abilities (`special_*`, `choose`).
- [ ] `destination_token` runs (per-major destination bonuses).
- [ ] L trains + permanent trains; train export/obsolescence rules.
- [ ] RL layer: new encoder/action space/model + training run (as above).

> **Player-count constraint:** 1822 supports up to **7 players**, which exceeds the current
> `VALUE_SIZE = MAX_PLAYERS = 6` baked into the value head. Either widen the value head (and
> encoder player slots) before the 1822 RL layer, or cap 1822 at 6 players initially.
> (`VALUE_SIZE = 6` is currently hardcoded in `mcts.rs`, the encoder, and the model —
> single-sourcing it is a Phase 0.5 item.)

## Phase 3 — 1822CA (incremental on 1822)

In Ruby this `extends G1822::Game`, so it reuses Phase 2's machinery.
- [ ] Translate the large CA entities/map/scenario data → `title/g1822_ca.rs` (Rust only).
- [ ] CA-specific steps: `acquisition_track`, `assign_sawmill`; scenario/`trains.rb` variants.
- [ ] RL layer + training run.

---

## Cross-cutting / risks

- **Validation discipline.** Every title needs an executable oracle before its self-play
  data is trusted — for 1830 that was Python↔Rust parity; for new titles it is the
  fixture/corpus strategy above (see "Per-title validation strategy"). Stand the per-title
  audit up FIRST, not after the mechanics.
- **Training cost, not just code.** Each title is a separate from-scratch AlphaZero run —
  budget compute accordingly. A shared GNN/transformer trunk across titles is an interesting
  research angle but is *not* assumed here.
- **Sequencing.** Phase 0 ✅ → Phase 0.5 seams → 1867 → 1822 → 1822CA. Don't attempt
  1822CA before 1822 base.
- **Effort (in units of the original 1830 engine work):** Phase 0 ✅ (landed); Phase 0.5
  ≈ 0.15–0.25×; 1867 ≈ 0.7–1.0×; 1822 ≈ ≥1× (plausibly the largest single chunk);
  1822CA ≈ 0.3–0.5× on top of 1822. Plus one training run per title.

## Related docs

- `CLAUDE.md` — current architecture (dual Python/Rust engine, v1/v2 models).
- `docs/rust_engine_*_audit.*`, `docs/cleaning_engine_parity.*`, `docs/rust_engine_bugs.md` —
  the parity tooling to replicate per title.
- Root `roadmap.md` — lists multi-title support as a stretch goal among other engine/model work.

## Phase 0 outcomes & carry-forwards (2026-06-10, branch `multi-title`)

Phase 0 landed in two refactors, both verified 1830-behavior-preserving (full
corpus 0/3243 import-outcome parity, strict 4-axis subset, random walks,
pytest 326 + cargo 70, independent diff reviews):

- **Ability system** (`3aec050..297f925`): per-company `AbilityDef` data on
  `CompanyDef`; CS/DH/MH/BO/CA/SV sym if-chains across five layers replaced by
  data queries; blocked-hex map single-sourced; 1830 action layout pinned by a
  frozen-layout cargo test (wired into the pre-push hook).
- **Step/round machinery** (`de74e09..338c5fc`): `steps.rs` —
  `StepKind`/`StepDesc` per-title ordered step lists + ONE shared `actions_for`
  accumulation loop; `skip_steps` table-driven; `OperatingStep::next()`
  deleted; legacy dispatch frozen as a `#[cfg(test)]` oracle for the in-crate
  random-walk differential; `RoundKind` cycle description drives
  `transition_to_next_round`.

Carry-forwards from the sign-off review (do these WITH the 1867 work):

1. **Stage C is a thin veneer.** 1867's merger rounds interleave *within* the
   OR set (SR → MR → OR → MR → OR) and OR counts are phase-dependent —
   a static `&[RoundKind]` cycle can't express that, and the OR-set repetition
   early-returns inside `transition_to_next_round`, bypassing the cycle. The
   1867 work should replace the static list with a per-title round-flow hook
   (or give `RoundKind::OperatingSet` interleave entries) rather than extend
   the current shape.
2. **`next_operating_pc` is first-match by pc** (`steps.rs`): a title listing
   two steps with the same `operating_pc` (e.g. 1822's minor-first-OR BuyTrain
   in front of the normal BuyTrain) would mis-sequence. Use a pc-less overlay
   step + `step_blocking_override` (the SpecialToken pattern) for such steps,
   or generalize the pc walker to positional indices.
3. **Title dispatch funnel**: `steps.rs::{round_step_descs,
   operating_step_descs, round_cycle}` hardcode g1830 — the `GameTitle` trait
   goes exactly there.
4. Minor: the blocking step's `step_actions` is computed twice per enumeration
   (blocking scan + accumulation) — memoize if MCTS profiles show it; add
   debug_asserts for the "crowding forces pc=DiscardTrain" / "pending tokens
   force pc=PlaceToken" invariants the new state-gated `step_active` relies on.
5. **Known both-engine wart** (pre-existing, parity-faithful): pending
   home-token enumeration on an OO hex emits full cities (`slot=None`) that
   `process_action` then rejects — a legal-masked policy index can be
   unapplyable, which kills a self-play playout if sampled. Fixing requires a
   Python-side decision first (the enumeration is reference behavior).
