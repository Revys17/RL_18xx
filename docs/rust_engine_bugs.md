# Rust engine divergences from Python reference

## Full raw-corpus audit — all player counts (docs/rust_engine_full_corpus_audit.md)

After running `audit_full_raw_corpus.py` on the full 1991 completed games
in `human_games/1830/` (all player counts 2–6):

- **Perfect: 1661 / 1991 (83.43%)**
- Dropped by cleaning filters: 316 (15.87%)
- Divergence: **0** (all 4 prior divergences fixed)
- Cleaning error: 14 (0.70%, Python-side cleaning rejections)
- **Replay-survivors-only: 1661 / 1675 (99.16%)**

### Divergence fixes that landed (3 new fixes, post initial audit)

| Fix | Reason | File |
|---|---|---|
| `stock_process_buy_shares` honors `share_indices` from action | Rust was picking the first matching share, not the specific index named by the action (e.g., `NYC_8`). Same class as the `discard_train` bug. | engine-rs/src/rounds/stock.rs |
| `try_process_company_exchange` calls `check_president_change` | MH exchange could push the receiving player over the threshold for new president, but Rust didn't trigger the swap. | engine-rs/src/game.rs |
| MH exchange + post-OR transition | After an MH exchange ends the last corp's turn and the OR finishes, Rust needs to transition_to_next_round (in addition to skip_steps/advance_to_next_corp). | engine-rs/src/game.rs |

Per-player-count:

| Players | Total | Perfect | Dropped | Diverged | Other |
|---|---|---|---|---|---|
| 2 | 35 | 23 (66%) | 11 | 1 | 0 |
| 3 | 238 | 187 (79%) | 47 | 1 | 3 |
| 4 | 1261 | 1115 (88%) | 146 | 0 | 0 |
| 5 | 334 | 243 (73%) | 82 | 2 | 7 |
| 6 | 123 | 89 (72%) | 30 | 0 | 4 |

Variable player count (2–6) is functionally working. 4p has the
highest perfect-rate because it's the most-played count and the
cleaning heuristics were originally tuned for it.

### Remaining latent bugs (18 games, 1.07% of corpus)

**4 divergences:**
- Games 48181 (5p), 48554 (5p): NYC shares — Rust shows 10% in market
  where Python shows 10% on a player. Triggered by `buy_shares` during
  Stock Round. Possibly a 5-player share-pool transfer edge case.
- Game 54058 (2p): At `buy_shares`#172, Rust still thinks the round is
  Operating but Python has transitioned to Stock. 2p-specific OR→SR
  transition timing.
- Game 77513 (3p): NYC president diverges (Rust=player:3, Python=player:2).
  Possibly the clockwise-from-prev tiebreaker has a 3-player edge case.

**14 Python-engine cleaning errors:**
- 4× `Blocking step ... cannot process action` (LayTile during SBS, etc.)
- 5× `Not a buyable train`
- 3× `NYC/PRR cannot lay token - remaining token slots are reserved on E11`
- (these are Python engine refusals, not engine divergences — Python
  refuses the action before Rust sees it. They're either real engine
  bugs Python catches or game-specific issues with action sequencing.)

The pretraining filter (`get_game_object_for_game`) silently drops all
18 of these — they show up as `cleaning_error` or `divergence` in the
audit but won't be in training data.

---

## Cleaned-subset audit (4p only) — docs/rust_engine_corpus_audit.md

For reference, the previous round of audits on the pre-existing
`human_games/1830_clean/` subset (4p only, prior import run):
1115/1261 perfect (88.42%); 99.91% of replay-eligible games.

## Final state — corpus audit: 1113/1261 perfect (88.26%); 99.73% of replay-eligible games

### FIXED — 8 distinct engine bug fixes

| # | Bug | File:line | Impact |
|---|---|---|---|
| 1 | `operating_order` was static from OR start; Python re-sorts unoperated tail after sell-driven share-price moves | `engine-rs/src/game.rs:1231` (`recalculate_operating_order`), called from `rounds/operating.rs:or_emergency_sell` | +190 games (47.74% → 62.81%) |
| 2 | End-of-SR sold-out iteration order was insertion-order, not operating-order — `market_cell_corps` arrival order then diverged | `engine-rs/src/rounds/stock.rs:check_sold_out_price_increases` | (rolled into #1 audit) |
| 3 | `SellShares` `num_shares` count derived from share-object iteration counter instead of `percent / share_percent` — president-share bundles reported count off-by-one | `engine-rs/src/game.rs:sellable_bundles` | enumeration parity |
| 4 | `or_process_discard_train` matched by base name (e.g. "3") not full ID (e.g. "3-4") — wrong instance discarded | `engine-rs/src/rounds/operating.rs:or_process_discard_train` | +34 games (cascaded into many bank_cash + buy_train) |
| 5 | `or_emergency_sell` never handled president-dump — only transferred non-president shares | `engine-rs/src/rounds/operating.rs:or_emergency_sell` | +40 games |
| 6 | `DiscardTrain` step required strict `crowded_corps[0]` — Python accepts any crowded corp | `engine-rs/src/rounds/operating.rs:30-86` | +11 games |
| 7 | `or_emergency_sell` didn't pass seller as `previous_president` — clockwise-from-prev tiebreaker fell back to candidates[0] | `engine-rs/src/rounds/operating.rs:1160` | +2 games |
| 8 | `BuyTrain` source-detection picked inter-corp when action.price ≠ depot.price; should check train ID in depot first; should use action.price as actual_price | `engine-rs/src/rounds/operating.rs:880` | +1 game |
| 9 | `or_emergency_sell` missing partial-bundle return (sell percent < pres face value leaves half-president with seller) | `engine-rs/src/rounds/operating.rs:1160` | +2 games |

Plus the parity-test cleanup: `_is_ignored_key` and `_price_range_close` tolerance
removed from `tests/test_factored_action_helper_rust_parity.py` so divergences
surface as test failures.

### REMAINING — 3 games (0.27% of replay-eligible)

- **MH any-time exchange** (games 68955, 70674): MH private company
  exchanges for an NYC share between corp turns in an OR. Python advances
  to the next operator after the exchange; Rust stays on the previous
  corp. The plan's "Any-time actions" support is partially started but
  the engines don't isolate these as separate decision nodes. Deferred —
  the user plans to handle MH support as a separate piece of work that
  also touches the cleaning-side pass insertion.
- **Game 54156 (Rust rejects mis-attributed dividend)**: Ruby's action
  log records `dividend entity=C&O` at action 462 with `current_operator
  = PRR`. The cause is likely a master-mode action (player undoes too far
  back, then re-issues another corp's final action under their own corp's
  name). Both engines now reject this as a strict-dispatch violation
  (Rust rejects with `rust_rejected_dividend`; Python's
  `process_dividend` would error similarly via the BaseStep dispatcher).
  Honest failure rather than silent wrong-corp attribution.

### Semantic alignment work

After the engine fixes above, several action processors in Rust still
silently ignored `action.entity` and applied the action to whichever corp
was the current operator. This masked mis-attributed actions (like game
54156) instead of surfacing them as errors. Per user request, Rust now
validates `entity_id == current_corp_sym` strictly, matching Python's
dispatch:

| Processor | Validation added | File:line |
|---|---|---|
| `or_process_lay_tile` | entity_id == current_corp | operating.rs:386 |
| `or_process_place_token` | entity_id == current_corp | operating.rs:647 |
| `or_process_run_routes` | entity_id == current_corp | operating.rs:758 |
| `or_process_dividend` | entity_id == current_corp | operating.rs:767 |
| `or_process_buy_train` | entity_id == current_corp | operating.rs:891 |
| `or_process_buy_company` | entity_id == current_corp | operating.rs:1334 |
| `or_process_pass` | entity_id == current_corp | operating.rs:1713 |

Skipped:
- `or_process_discard_train` — accepts any corp in `crowded_corps` list
  (already validated by the dispatcher at operating.rs:38).
- `or_emergency_sell` — entity is a player ID (the seller), not a corp.

Python's `process_dividend` also changed: it now uses
`self.current_entity` instead of `action.entity`, so the engines align
on master-mode dividends (game 54156 — pays PRR per current operator,
not C&O per action.entity).

### Lessons learned

- The `current_entity` cluster I initially diagnosed as a single
  "tiebreaker" bug turned out to be **two distinct bugs** (operating_order
  recalc + sold-out iteration order). Each subagent investigation needed
  empirical verification before fix.
- Several "downstream" clusters (bank_cash, share_ownership, president)
  were cascading from a single upstream `discard_train` ID-matching bug
  (#4 above) — fixing the upstream resolved 34 games across all three
  clusters.
- The `crowded_corps` "fix" I applied based on a subagent's wrong
  diagnosis was reverted (it broke 14 games). Always verify a fix's audit
  delta vs the prior baseline.

The factored-action parity test
(`tests/test_factored_action_helper_rust_parity.py`) used to pass by sweeping a
collection of known engine divergences under two rugs:

1. `_is_ignored_key()` silently dropped any `CompanyBuyShares` or `SellShares`
   key from the diff.
2. `_price_range_close()` allowed BuyTrain price ranges to differ by up to $5.
3. The `random_game` test tolerated up to 5 mismatching steps out of 120; the
   `long_game` test tolerated a 15% mismatch rate.

All four escape hatches have been removed. The strict parity tests now run
with `set()` equality on the categorical keys and exact equality on price
ranges, and they surface **8 distinct divergence categories** across the
existing parametrized seeds (42–46, max 500 steps each):

| Category                          | Direction      | Seeds with hits      | Max count |
|-----------------------------------|----------------|----------------------|-----------|
| `SellShares` enumeration          | py_only        | 42, 43, 44, 45, 46   | 11        |
| `LayTile` reachability (py side)  | py_only        | 42, 43, 44, 45, 46   | 52        |
| `LayTile` reachability (rust side)| rust_only      | 42, 43, 44, 45, 46   | 93        |
| `Pass` availability               | rust_only      | 42, 43, 44, 45, 46   | 16        |
| `PlaceToken` enumeration          | py_only        | 42, 43, 44, 46       | 3         |
| `BuyTrain` source (depot/discard) | py_only / rust_only | 42              | 23        |
| `CompanyBuyShares` (MH → NYC)     | py_only        | 46                   | 2         |
| `BuyCompany` (DH)                 | py_only        | 43                   | 1         |

The test files now point to this document via comments and assertion messages.

Cross-reference:
- `tests/test_rust_action_parity.py` covers encoder state-vector parity; the
  bugs catalogued here are at the *action enumeration* layer and are
  orthogonal to that test.
- `tests/test_rust_raw_actions_parity.py` covers action serialization, not
  enumeration; it is unaffected by these bugs.

## Bug categories

### 1. `SellShares` enumeration (count off-by-one or missing)

The Python helper emits `SellShares` options that the Rust port does not.
Concrete examples (`py_only`, `rust_only=set()`):

- seed=43, step=70: py emits `('SellShares', 'CPR', 2)`; rust emits nothing.
- seed=43, step=75: py emits `('SellShares', 'CPR', 3)`; rust emits nothing.
- seed=44, step=98: py emits `('SellShares', 'C&O', 2)`; rust emits nothing.
- seed=46, step=35: py emits `('SellShares', 'B&M', 2)`; rust emits nothing.

These are typically partial-percent president-dump bundle counts. Python
surfaces the larger bundle sizes that emerge after a president-share dump;
Rust currently caps the enumeration. Tracked previously by `_is_ignored_key`
on the `SellShares` branch (now removed). Likely root cause: the Rust
president-share split bundle logic doesn't iterate over all legal share
counts when the seller is the president.

### 2. `LayTile` reachability divergences (both directions)

Far and away the largest bucket. Both engines emit tile-lay options the
other does not, indicating the corp's connected-hex set diverges. Examples:

- seed=42, step=187 (py_only): `('LayTile', 'I17', '7', 1)`.
- seed=42, step=268 (rust_only): `('LayTile', 'C23', '7', 5)`.
- seed=44, step=129 (py_only): `('LayTile', 'E19', '57', 4)`.
- seed=44, step=306 (rust_only): `('LayTile', 'F8', '9', 3)`.

Symptom: the graph/route reachability calculations in Python vs. Rust drift
once tiles are laid. Suspected sources: (a) edge-port enumeration differs
when a tile rotation creates a new connection, (b) one engine treats a
blocked hex as reachable while the other does not, (c) station-token graph
edges are pruned differently. This category dominates the long-game
mismatch counts (35–93 mismatches per 500-step seed).

### 3. `Pass` availability in emergency BuyTrain

Rust emits `Pass` in operating-round contexts where Python does not. All
five seeds hit this category (rust_only). Examples:

- seed=42, step=193: rust emits `('Pass',)` not present in py.
- seed=43, step=184: same pattern.
- seed=46, step=182: same pattern.

This typically surfaces in emergency-buy-train scenarios: the Rust port
allows `Pass` as a legal action when the active corporation has no
affordable train and the president lacks the cash to force a buy, whereas
Python escalates to `Bankrupt` (or chains a forced sell). The two engines
disagree on the "can pass / must declare bankruptcy" boundary.

### 4. `PlaceToken` enumeration

Python emits `PlaceToken` options the Rust port omits:

- seed=42, step=293: py emits `('PlaceToken', 'F16', 0)`.
- seed=43, step=266: py emits `('PlaceToken', 'E11', 0)`.
- seed=44, step=315: py emits `('PlaceToken', 'F16', 0)`.
- seed=46, step=323: py emits `('PlaceToken', 'F16', 0)`.

F16 is a recurring offender, suggesting a specific city-slot or
home-station rule that Rust evaluates differently (most likely the
"reserved home" slot accounting, or whether a corp can place its second
token while the first is still pending).

### 5. `BuyTrain` source (depot vs. discard) and price range

seed=42 is the only seed in the existing set that reaches D-train rust
phase deep enough for this to fire. Both directions hit:

- seed=42, step=318 (py_only): `('BuyTrain', 'depot', '3', None)`.
- seed=42, step=318 (rust_only): `('BuyTrain', 'discard', '3', None)`.

The two engines disagree on whether a discarded 3-train sits in the
depot's available pool or in the discard bank — i.e., the discard-pile vs.
depot-active-trains accounting drifts after a forced discard. The
previously-tolerated `<= $5` price_range delta likely also originated here
(`spend_minmax` differences between Python's `train.price - already_spent`
and Rust's face-value cap).

### 6. `CompanyBuyShares` (MH → NYC exchange timing)

Python's helper surfaces the MH → NYC reserved-share exchange whenever the
MH owner has a claimable share, *including before NYC is parred*. The Rust
port requires NYC to be parred first.

- seed=46, step=19: py emits `('CompanyBuyShares', 'MH', 'NYC')`; rust empty.
- seed=46, step=23: same.

Tracked as task #12 (Investigate any-time MH exchange engine support). The
1830 rulebook is ambiguous; the Python helper follows the
`tobymao/18xx` Ruby source which permits the exchange any time after MH is
owned by a player. Rust restricts it to "after NYC pars," matching the
common play interpretation. Resolution requires a rules call before code.

### 7. `BuyCompany` enumeration (DH)

Single observation:

- seed=43, step=273 (py_only): `('BuyCompany', 'DH')`.

Python offered the corp a chance to buy the Delaware & Hudson private at
that step; Rust did not. The likely cause is the "is this private
available to be bought by this corporation right now?" predicate
(turn-order, phase gating, or owner-is-still-the-player). Low frequency;
needs targeted reproduction.

## How to reproduce / debug

```bash
uv run pytest tests/test_factored_action_helper_rust_parity.py -v --tb=short
```

Each failing assertion prints the first three mismatching steps and their
`py_only` / `rust_only` key sets. For a per-seed category breakdown, use
the `__main__` block at the bottom of the test file:

```bash
uv run python tests/test_factored_action_helper_rust_parity.py
```

## Status

- Parity tests now run **without tolerance** and surface every divergence.
- 6 of 10 tests in the file currently fail (4 random_game + 2 initial / price
  range smoke tests pass; long_game[42–44] and random_game[43, 44, 46]
  fail). This is the intended state until the engines are fixed.
- No further test-side changes should be made to suppress these
  divergences. Each category above corresponds to a real engine bug.
