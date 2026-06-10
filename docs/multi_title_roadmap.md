# Multi-Title Support Roadmap (1867 & 1822/1822CA)

Plan for extending the engine beyond 1830 to the other 18xx titles we want to train agents
for: **1867: The Railways of Canada** and **1822 / 1822CA: The Railways of Great Britain
(+ Canada)**. This is a long-horizon effort to begin *after* the 1830 agent is competent.

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

The blockers are 1830 *mechanics and constants* baked into the engine and RL layer:

- **No title dispatch.** `engine-rs/src/title/mod.rs` is one line (`pub mod g1830;`);
  `BaseGame::new()` (`game.rs`, ~line 1456) hardwires `g1830::*` and the title string
  `"1830"`. `StockMarket::new_1830()` / 1830-specific 2D market movement live in `core.rs`.
- **10-share assumption.** Share creation hardcodes 1 president @20% + 8 @10%
  (`game.rs` ~line 1504); dividend math uses `total_shares = 10`
  (`rounds/operating.rs:869`). 1867 mixes 5- and 10-share corps; 1822 majors are 10-share.
- **60% float threshold** hardcoded (`entities.rs:535`). 1867/1822 float differently.
- **Fixed action space.** `action_index.rs:23` pins `POLICY_SIZE = 26537`, computed from
  1830's exact counts (8 corps / 6 privates / 6 par prices / ~96 hexes / 46 tiles / 6 train
  types). `encoder.rs` sizes the observation vector the same way. Python mirrors live in
  `action_mapper.py` / `encoder.py`.
- **Absent mechanics** (needed by the new titles): minors, mergers/conversions, loans &
  interest, share issue/redeem, destination tokens, concessions, bidbox auction. No `Minor`
  entity exists.

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

## Phase 0 — Title abstraction refactor (prerequisite, 1830 stays a no-op)

Goal: make "1830" one title among several, with the 1830 parity tests as the regression
oracle. Should land as a behavior-preserving change for 1830.

- [ ] Define a `GameTitle` trait/config in Rust (and the Python analogue) supplying title
  data + mechanic hooks: corporations, companies, trains, phases, market grid+movement, hex
  map, tiles, starting cash / cert limits / bank, **share structure**, **float threshold**,
  **dividend rule**, valid player counts.
- [ ] Route `BaseGame::new()` through the trait instead of `g1830::*` directly; replace the
  hardcoded title string. Add `title/mod.rs` dispatch.
- [ ] Parameterize the baked-in constants: share structure (drop the 20%/10%×8 assumption),
  `total_shares` in dividend math (`rounds/operating.rs:869`), float threshold
  (`entities.rs:535`), market-movement rule (`core.rs`).
- [ ] Make the encoder + action layout **title-parameterized** rather than `const`: counts of
  corps/privates/hexes/tiles/trains/par-prices come from the title. Keep 1830's numbers
  identical so existing checkpoints still load.
- [ ] Mirror all of the above in the Python engine + `RustGameAdapter`, and keep
  `action_mapper.py` ↔ `action_index.rs` in lockstep.
- [ ] Green parity tests/audits for 1830 (`docs/rust_engine_*_audit.*`,
  `docs/cleaning_engine_parity.*`).

## Phase 1 — 1867: The Railways of Canada

**Data translation (mechanical):** map, hexes, tiles, trains (incl. permanent), phases,
market, minors + majors, privates → `title/g1867.rs` + `g1867.py`.

**New mechanics (the hard part):**
- [ ] `Minor` entity + minor operating/ownership; minors→majors conversion.
- [ ] **Merger round** + steps: `merge`, `post_merger_shares`, `reduce_tokens`,
  `major_trainless`.
- [ ] Variable share structures (5-share and 10-share corps) — depends on Phase 0
  parameterization.
- [ ] Loans & interest (`loan_operations`, `buy_company_preloan`), share `redeem`/issue.
- [ ] `single_item_auction` setup; CN (national) formation / end-game.

**RL layer (per-title):**
- [ ] New encoder feature set + `action_index`/`action_mapper` layout sized for 1867
  (different corp/hex/tile/train counts → different `POLICY_SIZE`).
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
> encoder player slots) in Phase 0, or cap 1822 at 6 players initially.

## Phase 3 — 1822CA (incremental on 1822)

In Ruby this `extends G1822::Game`, so it reuses Phase 2's machinery.
- [ ] Translate the large CA entities/map/scenario data → `title/g1822_ca.rs` + `g1822_ca.py`.
- [ ] CA-specific steps: `acquisition_track`, `assign_sawmill`; scenario/`trains.rb` variants.
- [ ] RL layer + training run.

---

## Cross-cutting / risks

- **Parity discipline.** Every title needs the same Python-vs-Rust parity guarantee 1830 has.
  Stand up a per-title corpus audit (mirror `scripts/audit_rust_engine_corpus.py`) before
  trusting self-play data. Human-game corpora from 18xx.games can seed these audits.
- **Training cost, not just code.** Each title is a separate from-scratch AlphaZero run —
  budget compute accordingly. A shared GNN/transformer trunk across titles is an interesting
  research angle but is *not* assumed here.
- **Sequencing.** Phase 0 → 1867 → 1822 → 1822CA. Don't attempt 1822CA before 1822 base.
- **Effort (in units of the original 1830 engine work):** Phase 0 ≈ 0.3–0.5×; 1867 ≈
  0.7–1.0×; 1822 ≈ ≥1× (plausibly the largest single chunk); 1822CA ≈ 0.3–0.5× on top of
  1822. Plus one training run per title.

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
