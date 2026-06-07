# Rust ↔ Python 1830 engine parity — status (2026-06-06; RESOLVED 2026-06-07)

Consolidation of a parity push. All work is on branch **`rust-parity`**, not merged.
`master` additionally has `67a58ce address dependabot`.

## ✅ RESOLVED 2026-06-07 — human-import parity is now EXACT

Full human corpus (**all 3243** `human_games/1830/*.json`) = **0 failures**; random
0:200 = **0 failures** (verified with the honest harness on a clean rebuild of the
committed source). The ~25% gap below was **two engine-rs root causes**, both fixed:

1. **`b45c6e4` — BuyTrain inserted-pass OR-desync** (the bulk, ~556 games). On an
   inserted `pass` at the BuyTrain step Rust prematurely advanced `current_entity`
   to the next corp and reset the OR step to LayTile; Python keeps the current corp
   on BuyTrain (mandatory/emergency train purchase → bankruptcy path). Fixed in
   `rounds/operating.rs` `or_process_pass`.
2. **`532f509` — per-cert share identity** (the `rust_import_error` /
   "All shares must be owned by the same owner" class). (a) sells move the EXACT
   certs named in the action (`share_indices`), mirroring Python
   `SharePool.transfer_shares`; (b) every president swap picks the new president's
   certs in acquisition (`acquired_seq`) order = Python's
   `shares_for_presidency_swap(president.shares_of(corp))` — the MH→NYC exchange
   had skipped the snapshot and used Vec-index order, drifting per-cert identity.

Diagnosed with **`tests/cleaning_diff.py`** (`e4067c6`), a dual-engine cleaning
diagnostic (lockstep + decision-trace) that pinpoints the first divergence per game.
**Note:** `compare_state` is AGGREGATE-ONLY (per-owner share sums), so per-cert
index→owner drift is invisible to it — it only surfaces when an action names a
specific cert id. Prefer `result()` + the cleaning-import metric for share
correctness.

The rest of this document is the pre-resolution snapshot.

## Two parity axes (they are different tests)

1. **Random-walk parity** — walk seeds in lockstep on both engines; at each step compare
   the *factored legal-action enumeration* (categorical + exact price) **and** the game
   *state* (`compare_state`).
2. **Human-game IMPORT parity** — run each human game through the **cleaning pipeline**
   (`pretraining._get_game_object_for_game_with_reason`) on BOTH engines and compare the
   imported game (drop-reason + final `result()` + `compare_state`). This is the
   training-data metric. It is NOT a raw `filter_actions` replay (the cleaning does
   per-action substitution + pass/skip insertion).

## Current status

| Axis | Status |
|---|---|
| Random-walk (enumeration + state) | **CLEAN** — 150 seeds, 0 failures (verified with the honest harness) |
| Factored enumeration (seeds 42-46 + broad) | clean |
| **Human-game import** | **~75% parity.** Per 200 games: 33 `result_mismatch` (share price/ownership) + 18 `rust_import_error` (same class) + 8 both-drop |

The human-import gap is **pre-existing**, not caused by this session (the pre-session
baseline `67a58ce` gives the identical 33/18 over 200 games). The old
`docs/rust_engine_*_audit` "Divergence: 0" is overstated.

## Harness fixes (required for honest measurement)

The original parity tests had **three stacked blind spots**, each masking real divergences:
1. `python_side` loophole — a game counted as "Python can't replay" without checking Rust
   also rejects (Rust could be more permissive). Fixed: runner now verifies Rust-also-rejects.
2. plain-replay vs cleaning — the runner replayed raw `filter_actions` output, hitting
   blocking-step rejections the real import resolves. Fixed: human mode runs the real
   cleaning import.
3. `compare_state` returned early for **finished** games, skipping ALL end-state checks
   (bank/cash/shares/**prices**). Fixed: end state is now compared; only stale round/step
   structure is skipped. (This is why `compare_state`=0 while `result()` differed by hundreds.)

Also: runner now catches pyo3 `PanicException` (a `BaseException`) so one bad game can't
crash a sweep, and classifies bad-data games (invalid player counts) correctly.

**Takeaway: never trust a "clean" parity result unless it was measured with the honest harness.**

## Engine fixes landed this session (branch `rust-parity`)

- Player seating in input order (was sorting ids → wrong turn order on human games).
- Operating order, entity handling, D-train depot slots (`TrainDef.available_on`/`discount`).
- Blocking-step rejection + "not a buyable train" (Rust rejects what Python rejects).
- MH→NYC pre-par exchange, emergency president share-sale, CS/DH special-track + DH teleport
  token, BuyTrain `spend_minmax` exactness, LayTile graph-reachability unification.

## OPEN WORK: human-import share-state divergences (~25%)

**Root class:** Rust's share-transaction processing (sells / market price movement /
president-share handling) diverges from Python during import, leaving different share
ownership + prices. The 18 `rust_import_error` are the same divergence surfacing as a
Python `ShareBundle` "All shares must be owned by the same owner" when reconstructing a
SellShares from the already-divergent Rust state. Likely a *handful* of root causes.

**To diagnose (the simple approaches all fail):**
- raw `filter_actions` replay breaks at blocking steps before reaching the divergence;
- `game.raw_actions` does not round-trip to a fresh game;
- the cleaning does per-engine action substitution + pass-insertion.
- → **Instrument `_get_game_object_for_game_with_reason` to run both engines in lockstep**
  and `compare_state` after each *applied* action; the first divergence is the bug.

## Tooling / how to run

```bash
# strict parity runner (reads --out JSON; ALWAYS read the file, not stdout — Rust panic
# messages go to stderr and corrupt a piped JSON parse)
uv run python tests/parity_runner.py --random 0:150 \
  --human-games "$(ls human_games/1830/*.json | head -400 | paste -sd,)" --out /tmp/p.json
```
- `tests/parity_runner.py` — strict runner (random + human/cleaning modes).
- `tests/validate_rust_engine.py::compare_state` — the (now complete) state comparison.
- `tests/_parity_diag.py` — per-category random-walk divergence diagnostic.
