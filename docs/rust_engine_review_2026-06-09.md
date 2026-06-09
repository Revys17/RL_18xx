# Rust engine readiness review — `rust-parity` vs `master` (2026-06-09)

Five-lens review of the full branch diff (34 commits, ~4,600 insertions, 41 files):
Rust core engine, Rust action-space/search stack, Python-side changes (oracle
integrity), test-harness integrity, and multi-title extensibility. Purpose:
decide whether the Rust engine can serve as the source of truth for game state
and action processing, and find design choices that would restrict the upcoming
1867 / 1822 extension.

## Verdict

**Yes for the production loop, with a precise boundary.** The Rust engine is
trustworthy as the source of truth **for action streams produced by its own
(parity-verified) enumerator and decoder** — self-play, MCTS, pretraining
encoding. The ported rules logic is faithful (suspicious hunks were checked
against the Python/Ruby oracle and held up), the enumerate→index→decode core is
structurally sound, and the parity claim itself is honest: no oracle-bending was
found in any Python-engine change, and every modified test was a strengthening.

What the engine is **not** is a *validating* engine: `process_action` trusts
callers far more than Python does. For arbitrary external action streams it will
accept (and silently apply) several things Python rejects. That class is
enumerator-protected today but bounds the "source of truth" claim.

## Bugs found (ranked) — Rust core

1. **[major] Exchange-BuyTrain mutates before validating** —
   `rounds/operating.rs:1133-1183`: the trade-in train is removed and pushed to
   `depot.discarded` *before* the affordability check; an `Err` leaves
   half-committed state. Python additionally raises "Cannot contribute funds
   when exchanging"; Rust silently lets the president fund an exchange
   shortfall. Unreachable from the enumerator (it gates on
   `corp_cash >= exchange_price`), but it breaks the engine-wide
   clone-then-commit invariant that a failed action leaves state untouched.
2. **[major] Player-owned CS/DH LayTile guard hole** — `game.rs:1006-1019`: the
   company-ability operator guard only fires when the company is
   corporation-owned; a player-owned CS/DH `lay_tile` falls through to the
   generic LayTile arm, lays on whatever hex the action names (no B20/F16
   check), charges no one, and marks `ability_used`. Python rejects it
   (`owner_type: "corporation"`). Unreachable from the enumerator; silent
   state corruption if ever reached.
3. **[major, systemic] Validation gaps vs Python** (each enumerator-protected;
   listed as a class):
   - SellShares accepted at any OR step (`operating.rs:329-354`, an
     import-stream accommodation baked into the engine);
   - no `can_sell` validation on either sell path (ownership of named certs,
     50% pool cap, president-dump legality);
   - Bankrupt processed without `can_go_bankrupt` (`game.rs:408-447`);
   - BuyCompany not phase-gated (`operating.rs:1593-1618`; Python requires
     `can_buy_companies` in phase status);
   - inter-corp BuyTrain missing `check_spend`, depot buys take `action.price`
     verbatim, president contribution not gated on `must_buy_train`.
   Either add the checks or document loudly that `process_action` is defined
   only for enumerator/decoder-produced actions and that `Err` ⇒ discard the
   game object.
4. **[minor] Stock-round sell fallback uses wrong president face value** —
   `stock.rs:449-453` subtracts 10 where the OR path subtracts the president
   cert's 20. Aggregates end correct via partial-president compensation, but
   cert identity / `market_order` churn differently — and contrary to the
   comments, the fallback IS the hot path: native decode emits empty
   `share_indices` for every SellShares (`decode.rs:341-346`).
5. **[minor] MH exchange suppressed during out-of-turn DiscardTrain**
   (`game.rs:2305-2318`) — possibly a missing legal action (restrictive, not
   corrupting); Python's Exchange step keys on the company, not the operator.
   Low-medium confidence.
6. **[minor] `sell_bankrupt_shares` swallows sale errors** (`game.rs:1610-1623`,
   `break` instead of propagating).
7. Nits: `unwrap()` reachable in grandchild PUCT iff `min_price_children=0`
   (`mcts.rs:665`); layout total check is `debug_assert` (compiled out of the
   release wheel, `action_index.rs:178-181`); Bankrupt entity fallback parses
   id as player and skips liquidation silently (`game.rs:418-447`).

## Action-space / decode stack — clean

Block offsets hand-recomputed to exactly `POLICY_SIZE = 26537`; no overlaps.
Decode-by-re-enumeration makes index/decode disagreement structurally impossible
at a given state. The known state-dependent hazards (depot-discarded BuyTrain
slots, DH `CompanyPlaceToken`, OR-emergency SellShares president resolution) are
fixed symmetrically in enumerator and decoder. Only constructible decode-vs-
Python gap: discarded-D + unaffordable plain buy + affordable trade-in (Rust's
behavior is strictly more sensible; vanishingly rare; never exercised by the
corpus). Same slot also unions price ranges into a phantom PW slot (parity-
faithful waste, trains the price head on noise for that slot).

## MCTS search layer ≠ Python search (all pre-existing, promoted to production by `use_rust_mcts=True`)

These don't affect rules parity, but they make Rust self-play **a different
data-generating process** than the Python reference. Since training hasn't
started, decide deliberately now whether Rust's behavior is the spec:

1. **No softpick**: `RustMCTSPlayer.pick_move` is pure argmax;
   Python samples ∝ visit counts while `move_number < softpick_move_cutoff`
   (500). Game diversity rests entirely on Dirichlet noise.
2. **Temperature cutoff keyed to decision count** (`len(searches_pi)`), Python
   keys to engine `move_number` (includes forced-chain actions) — π targets go
   soft much deeper into the game on the Rust path.
3. **Forced-chain price-bearing actions apply at range MINIMUM**
   (`mcts.rs:777`); Python samples from the price head / midpoint. A forced
   emergency cross-corp BuyTrain executes at $1 deterministically.
4. **PW bookkeeping reads aggregate slot N for price grandchildren**
   (`mcts.rs:1287-1293`, `1117-1119`) where Python uses per-price counts —
   inflates the exploration term under PW slots ~√(slot_N/price_N) and mixes
   sibling values into the post-advance root stats.
5. Rust omits Python's categorical top-k progressive widening at wide nodes
   (>20 actions) and per-round c_puct (fixed 1.25).
6. **Encoder swap** (`encoder.py:392-398` delegates to Rust `encode_for_gnn`):
   verified only by a manual script (no pytest collection) with 0.02 absolute
   per-feature tolerance, comparing Python-encoder-on-Rust-state vs
   Rust-encoder-on-Rust-state (the real Python game is advanced but never
   compared). Old Python-encoded examples and new Rust-encoded ones may differ
   within tolerance — don't mix datasets across the transition.

## Harness integrity — trust the 0/3243, know its edges

No weakened assertions anywhere in the diff; harness history is consistent
hardening. Fixed during this review (commit `2c9cd97`): `pass_acceptance_divergence`
and `decision_divergence` added to the corpus-gate BAD set, `cleaning_diff` CLI
bad-list now covers all divergence statuses, `parity_runner` exits non-zero on
failures. Remaining known edges:

- **No CI**; the pytest-gated subset is far weaker than the manual corpus gates
  (`test_rust_adapter_compat.py` is assert-free; the factored-parity pytest
  swallows engine crashes via `terminated_early`; several `test_rust_*.py` files
  contain no collectable test functions). The 0/3243 claim decays with every
  engine commit until someone re-runs `tests/index_parity_corpus.py`.
- **compare_state blind spots** (aggregate share sums documented; also): tile
  rotation, token city/slot index, depot discard pile, company ownership,
  auction state, SR current player, operating order, priority deal, turn
  counter, asymmetric graph connectivity. Mostly backstopped indirectly by the
  enum/index/decode axes — probabilistically, not guaranteed.
- Production **Rust-side cleaning decisions** (the game-26861 class) are outside
  the strict gate (which is Python-oracle-driven); covered only by
  `diagnose_decisions` / `parity_runner` human mode.
- Random mode never compares the terminal state (breaks before the post-final
  check); corpus decode axis tests min price only (lo/hi/mid only in
  `decode_parity_check.py`'s 5 synthetic games).
- One standing corpus BAD: game 87895 `python_error` (Python-oracle
  WaterfallAuction limitation, not a Rust divergence) — a "clean" full strict
  run exits 1 with exactly this one.

## Multi-title extensibility — the roadmap list is right but incomplete

Census: **~110 literal 1830 leaks** outside `title/g1830.rs` (+28 direct
`g1830::` calls). Blocked-hex map exists in **three copies** (game.rs,
factored.rs, rust_adapter.py); encoder.rs keeps a third copy of title data
(CERT_LIMIT, STARTING_CASH, TRAIN_COUNTS, TILE_INITIAL_COUNTS); `config.py` has
the literal `26537` twice; `TrainDef.rusts_on` exists in g1830.rs but is unused
(rusting re-hardcoded in game.rs).

**Two top blockers MISSING from the roadmap:**

1. **No ability system** — private-company powers are sym if-chains
   (`co.sym == "CS" || co.sym == "DH"`, MH/NYC, BO no_buy) duplicated across
   **five layers**: game.rs processing+choices, factored.rs enumeration,
   action_index.rs slot routing, decode.rs entity resolution, rust_adapter.py.
   Ruby models these as per-company `abilities:` data interpreted by generic
   steps. 1822's ~30 minors/concessions are infeasible as if-chains.
2. **Step machinery is hand-flattened** — `OperatingStep::next()` is a
   hardcoded sequence and every blocking/non-blocking interleaving is a
   hand-derived match arm. The ~10 strict-enum parity commits on this branch
   are the cost signature of that design; 1867's merger round and 1822's
   choices round would reproduce the whole campaign. Port Ruby's ordered
   step-list + `actions_for` accumulation loop once.

Recommended Phase-0 order (each item has today's green 1830 corpus as its
regression oracle — all are cheaper BEFORE 1867 exists): ability system →
step-list refactor → title trait + data params (incl. `PhaseDef.status/events`,
share-structure array + float % + capitalization mode on `CorporationDef`,
stock-market movement policy enum) → per-title `build_layout()` with derived
`POLICY_SIZE` → entity-id hygiene.

**Branch-fresh code to reshape now (cheapest while fresh):**

- `decode.rs:112-120` `entity_json`: id-parses-as-u32 ⇒ `"player"`. **1867/1822
  minors have numeric-string corp ids ("1".."30") — guaranteed collision.**
  Resolve against actual game collections instead. ~20 lines now, painful after
  action logs exist.
- `decode.rs:454` trade-in donor list hardcodes `["4","5","6"]` — derive from
  `Train.discount` keys (data already exists).
- `decode.rs:536-558` RunRoutes decode collapses to one synthetic route with
  total revenue — 1867 loan interest and 1822 destination runs need per-train
  detail, and the degenerate route dicts go into the replayable action log.
- `mcts.rs:117-119` + `decode.rs:605-607` re-list COMPANIES/CORPS/TRAINS —
  derive from `layout()`; `mcts.rs:184-189` `"Bid" → 5` is 1830's increment;
  `VALUE_SIZE = 6` independently hardcoded at `mcts.rs:29`.
- `factored.rs` EMR blocks inline 1830's `EBUY_*` flags as prose — make them
  named booleans on title config now.

## Recommended actions (in order)

1. Fix core findings 1 and 2 (mutate-before-validate exchange; player-owned
   CS/DH hole) — small, real, and they protect the source-of-truth contract.
2. Decide the MCTS search-semantics questions (softpick, temperature key,
   forced-chain price, PW bookkeeping) BEFORE generating training data; if
   Rust's behavior is the spec, document it as such and retire the Python
   player as "reference".
3. Add a CI step (or at least a pre-training ritual) that runs pytest + 
   `index_parity_corpus.py` + `parity_runner --random`, now that all three
   gate correctly on exit code.
4. Do the entity-id hygiene fix in decode.rs now, before multi-title work
   starts and before any more action logs accumulate.
5. Treat the ability system + step-list refactor as Phase 0 of multi-title,
   ahead of the roadmap's current Phase-0 list.
