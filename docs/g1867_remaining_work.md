# 1867 — status checkpoint & remaining work

*Checkpoint written 2026-08-02 at `multi-title` = `86a011b` (about to be
fast-forwarded into `master`). This is the resume-point doc for the 1867
port; the full history and architecture rationale live in
`docs/multi_title_roadmap.md` and the per-commit bodies of `git log`
(every seam commit carries a dense description).*

## Where things stand

**Rules engine: 1867 plays end to end.** All four Ruby fixtures
(`tests/fixtures/1867/`) replay **completely with exact recorded final
scores** (`tests/test_g1867_fixture_replay.py` — full-length watermarks,
finish-line assertions, per-route revenue cross-check; no xfails).
Against the 300-game human-corpus sample (`tests/g1867_corpus_audit.py`):

| bucket | count | meaning |
|---|---|---|
| outcome_match | **114** | full replay, exact final scores |
| rejected | 95 | replay stops on a rejected action (classes below) |
| result_mismatch | 64 | full replay, scores differ (see triage below) |
| not_finished | 27 | full replay, engine doesn't flag game end |

Every landed rule was probed against the real Ruby engine
(ruby:3.2-slim docker + tobymao/18xx clone — rig and gotchas below), and
every commit passed: cargo (150), pytest (339), `parity_runner --random
0:100` (1830 lockstep), and a **game-by-game corpus-audit diff proving
zero regressions** (no rejection ever moved earlier).

What exists mechanically: single-item opening auction; SR bid-founding +
home-token choice + class-typed par ladder (majors z+p, minors x+p);
minors/majors with incremental cap; flat-top map + two-lay rule +
distance-priced tokens; loans/interest/LoanOperations/BuyCompanyPreloan;
engine-computed route revenue (28/28 + 3668/3668 cross-checked); merger
round (convert/merge/PostMergerShares/ReduceTokens); CN nationalization
(trainless events, MajorTrainless choose, neutral tokens, Montreal
reservation, 639); per-OR train export + GAME_END_CHECK timing;
RedeemShares; Ruby `city_map` upgrade token transfer; loan-adjusted
player valuation + end-game loan settlement.

## Remaining work (ordered by expected yield)

1. **Share-price movement divergence** (~58 of the 95 rejections: ~22
   "Lay Track cannot process lay_tile", ~27 SR Sell/Buy/Sell, 9
   "Redeem Shares cannot process lay_tile"). One root cause suspected:
   some 1-D market movement rule still differs (a probed example had our
   NYC at $180 vs Ruby ≤$165, which flips operating order and cascades).
   The sold-out bump (majors-only, `up == right`) and sell-movement are
   already ported — probe 2-3 games from the audit JSON to find what's
   left (candidates: sell-price left-per-block semantics, dividend edge,
   price-protection). Highest-yield seam remaining.
2. **Round-flow sequencing stragglers** (8 games): probed 21494@544 —
   Ruby is in a *merger round* where we sit at BuyTrain. Something
   upstream (likely tied to 1) shifts which OR/MR boundary a game is on.
3. **Game-end detection** (27 not_finished + the `finished=False` drift
   on games Ruby marks finished, e.g. 20505/23143): bank `:current_or` /
   `final_phase :one_more_full_or_set` edges. Scores already match via
   the virtual loan-walk, so this is bookkeeping, not valuation.
4. **result_mismatch triage** (64): at least 3 are **recorded-result
   drift** in old server games (proven: full Ruby replays at HEAD score
   them exactly as we do — the stored results predate the server's
   loan-drop scoring). Expect a mix of drift, class-1 fallout, and small
   real rules gaps; triage with the audit JSON + Ruby probes.
5. Small classes: merge-candidate validation (5), misc singletons.
6. **Not yet implemented** (fail loudly if reached): phase-8 train
   trade-in enforcement (`train_trade_allowed` discounts are data-wired
   but unenforced), the `grid_market` optional rule (data unplumbed; the
   only optional rule appearing in the 1867 corpus), the 2-player
   variant (70% cap / train+company removal, `TODO(1867-2p)`).
7. **The RL bridge — the actual goal.** Everything above is the rules
   engine; 1867 is still `alphazero_bridge_ready() == false`. Needed:
   per-title action layout (slot layout can't yet express 3 discounted
   trains, merger-round actions, `choose`), encoder spec, native decode,
   MCTS wiring, VALUE_SIZE/POLICY_SIZE plumbing (already single-sourced
   behind `engine_rs.policy_size_py()`), self-play integration. Design
   constraint: the 1830 layout is FROZEN (checkpoint-compatible); 1867
   gets clean derivations.
8. Then **1822** (Phase 2 of `docs/multi_title_roadmap.md`).

## Known issues / failure points (read before resuming)

- **Pre-#9655 merger-order accommodation**: current rules order the
  merger round ReduceTokens → PostMergerShares (GTG errata,
  tobymao/18xx#9655); the 2020-21 corpus recorded the OLD order, and
  current Ruby rejects its own records. Our dispatch gate admits both
  orders *while both phases pend*; **enumeration/self-play stays
  canonical** (ReduceTokens first). Don't "fix" the enumeration to match
  old records.
- **Corpus audit hygiene**: recorded `result` keys can be stale player
  names (renames) — `align_recorded_keys` in the audit handles it; and
  recorded results themselves can predate server scoring changes (the
  loan-drop drift above). A mismatch is not automatically our bug:
  confirm with a full Ruby-docker replay before chasing it.
- **Action-id mapping when probing**: fixture/corpus `id` fields are
  renumbered AND `pretraining.filter_actions` renumbers again. Map
  filtered index → file id via `original_id` where present; most corpus
  actions have none — match by `(created_at, type, entity)`. Probing the
  wrong id fabricates phantom divergences (burned twice).
- **Ruby probe rig**: `ruby:3.2-slim` image + shallow clone at
  `/tmp/18xx` (re-clone if missing); `gem install require_all`, mount
  the game dir, `Engine::Game.load(data, at_action: <FILE id>)`.
  Probe templates from past sessions: `/tmp/probe_natl*.rb` (may need
  re-creation after reboot).
- **1830 safety**: all 1867 behavior is behind `GameTitle` hooks with
  1830-inert defaults; shared code touched includes the dispatch gate,
  skip/advance machinery, market movement, `min_depot_price`, and
  game-end timing — each change was parity-gated per commit
  (`parity_runner --random 0:100` + full pytest + the frozen-1830-layout
  cargo pin). Before long 1830 training runs on a new engine build, run
  the full-corpus ritual (`docs/verification_rituals.md`):
  `tests/index_parity_corpus.py` (expect exactly the standing 87895
  python_error) — hours, background it.
- **Trainless-major freeze semantics**: while `trainless_major` is
  non-empty the whole round freezes (Ruby blocking-step behavior) —
  skip_steps/transition guards + the choose interceptor in
  `game.rs::process_action_internal` own resume. If a future seam sees a
  "stuck" round, check the queue first.
- **Cheater token slots**: the Ruby `city_map` transfer can append
  tokens past `normal_slots` (merged cities). Encoder work (item 7) must
  decide how to represent over-capacity cities.

## Gates to keep green per commit (the working rhythm)

```bash
# inline env — shell state does not persist:
cd engine-rs && PYO3_PYTHON=$PWD/../.venv/bin/python \
  LD_LIBRARY_PATH=$(../.venv/bin/python -c 'import sysconfig;print(sysconfig.get_config_var("LIBDIR"))') \
  RUSTFLAGS="-L $(../.venv/bin/python -c 'import sysconfig;print(sysconfig.get_config_var("LIBDIR"))')" \
  cargo test --release --no-default-features
uv run pytest tests/
uv run python tests/parity_runner.py --random 0:100
uv run python tests/g1867_corpus_audit.py --limit 300 --json /tmp/audit.json
#   → diff game-by-game vs the previous audit JSON; rejections may only
#     move DEEPER or convert. Keep fixture watermarks green (full-length).
```
