# Multi-Title Support Roadmap (1867 & 1822/1822CA)

Plan for extending the engine beyond 1830 to the other 18xx titles we want to train agents
for: **1867: The Railways of Canada** and **1822 / 1822CA: The Railways of Great Britain
(+ Canada)**.

## STATUS (2026-06-11) — start here

- **Branches:** `master` and `multi-title` are identical at the same commit; 1867 work
  happens on `multi-title`, 1830 training fixes on `master`, rebase periodically.
- **Phase 0 is DONE and merged to master** (ability system + step/round machinery, both
  verified 1830-behavior-preserving — see "Phase 0 outcomes & carry-forwards" at the
  bottom). The engine is also now *validating* (rejects malformed/ill-timed actions like
  Python does) and the 1830 verification rituals are documented and gated
  (`docs/verification_rituals.md`; pre-push hook runs pytest + cargo incl. the
  frozen-1830-action-layout test).
- **Next up: Phase 0.5 (pre-1867 seams), then Phase 1 (1867).** Phase 0.5 is the
  remaining title-parameterization work — do it FIRST, while the green 1830 gates are a
  free regression oracle for every seam cut.
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

## Phase 0.5 — Pre-1867 seams (do FIRST, each one 1830-behavior-preserving)

Every item here has the green 1830 gate suite as a free regression oracle; after 1867
exists, every refactor must preserve two titles at once. Recommended order:

- [ ] **Per-title round-flow hook** replacing the static `RoundKind` cycle (carry-forward
  #1): `transition_to_next_round` walks a title-supplied flow that can express 1867's
  SR → MR → OR → MR → OR interleaving and phase-dependent OR counts. Validate as a 1830
  no-op (the 1830 flow re-expressed in the new shape).
- [ ] **`GameTitle` trait** at the three `steps.rs` funnel points + `BaseGame::new()`
  (+ title string, `title/mod.rs` dispatch). Pure plumbing; 1830 the only impl at first.
- [ ] **Share-structure / float / capitalization parameterization**: `CorporationDef`
  gains a shares array (Ruby's `shares: [40,20,...]` shape), `float_percent`, and a
  capitalization mode (full vs incremental — hooks in `distribute_revenue` and the
  float payout); kill the `percent/10` unit assumption in `entities.rs` /
  `action_index.rs` / `decode.rs`.
- [ ] **Stock-market movement policy** as title data/enum (2-D vs 1-D), with
  `StockMarket::new_1830()` becoming the 1830 instance of a generic constructor.
- [ ] **Wire phase/train data**: use the existing `TrainDef.rusts_on` (delete game.rs's
  hardcoded rust/close/phase-trigger maps); add `PhaseDef.status`/`events`
  (can_buy_companies, close_companies) and consume them where game.rs hardcodes phase
  names.
- [ ] **Action layout + encoder per title**: `build_layout()` takes title data;
  `POLICY_SIZE` derived from the layout (replace `config.py`'s two literal 26537s with
  the mapper-derived value); encoder consumes title data instead of its own const
  copies; `mcts.rs` derives `price_head_entity_key`/`price_grid_step` from the layout
  and single-sources `VALUE_SIZE`. Keep 1830's numbers byte-identical (the frozen-layout
  test is the guard) so existing checkpoints still load.
- [ ] **`next_operating_pc` duplicate-pc hazard** (carry-forward #2): either generalize
  the pc walker to positional indices now, or document the overlay+blocking-override
  pattern as the required shape for duplicate-pc steps (1822's minor-first-OR BuyTrain).
- [ ] Green 1830 ritual after each seam (`docs/verification_rituals.md`).

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
- [ ] Fixture-replay harness (Ruby `public/fixtures/1867/*.json` → replay → assert final
  scores + per-action acceptance); download an 1867 human corpus from 18xx.games and
  stand up the outcome-replay audit. Build these before/alongside the first mechanics so
  every new step lands against an executable oracle.

**Data translation (mechanical):** map, hexes, tiles, trains (incl. permanent), phases,
market, minors + majors, privates → `title/g1867.rs` (Rust only — no `g1867.py`; the
Python engine stays 1830-only per the validation strategy above).

**New mechanics (the hard part):**
- [ ] `Minor` entity + minor operating/ownership; minors→majors conversion. NOTE: 1867
  minors have numeric-string ids ("1".."16") — the decode/entity-classification layer
  already resolves ids against game collections (fixed 2026-06-09), but watch for any
  remaining parses-as-int assumptions.
- [ ] **Merger round** + steps: `merge`, `post_merger_shares`, `reduce_tokens`,
  `major_trainless` — new `RoundKind` + `StepKind`s listed in g1867's descriptions,
  injected via the Phase 0.5 round-flow hook.
- [ ] Variable share structures (5-share and 10-share corps) — depends on Phase 0.5
  parameterization.
- [ ] Loans & interest (`loan_operations`, `buy_company_preloan`), share `redeem`/issue.
  NOTE: loan interest depends on route revenue detail — check whether the native
  RunRoutes decode's single-synthetic-route collapse (decode.rs) suffices or needs
  per-train routes for 1867.
- [ ] `single_item_auction` setup; CN (national) formation / end-game.

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
