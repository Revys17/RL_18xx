# 1867 — status checkpoint & remaining work

*Checkpoint updated 2026-10-15 (rules-fidelity session: items 1-5 of the
2026-08-02 list). Earlier checkpoint: 2026-08-02 (merged to `master` the
same day; the local history was rewritten then — commit hashes in older
docs/memory predate it; seam commits are identifiable by subject,
`git log --oneline --grep=1867`). This is the resume-point doc for the
1867 port; architecture rationale lives in `docs/multi_title_roadmap.md`
and the per-commit bodies (every seam commit carries a dense
description, including how it was confirmed against Ruby).*

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
2-player games (the unimplemented 2p variant, item 2 below); the full
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

## This session's seams (2026-10-15, one commit each, all Ruby-probed)

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

1. **The RL bridge — the actual goal.** 1867 is still
   `alphazero_bridge_ready() == false`. Needed: per-title action layout
   (slot layout can't yet express 3 discounted trains, merger-round
   actions, `choose`), encoder spec, native decode, MCTS wiring,
   VALUE_SIZE/POLICY_SIZE plumbing (single-sourced behind
   `engine_rs.policy_size_py()`), self-play integration. Constraint: the
   1830 layout is FROZEN (checkpoint-compatible); 1867 gets clean
   derivations. Rules-side, enumeration (as opposed to replay) is far less
   exercised than dispatch: expect gaps when self-play first drives it
   (e.g. tile-lay legality is not re-checked by the replay handler;
   impassable borders are not enforced in the graph — see known issues).
2. **2-player variant** (`TODO(1867-2p)`): 60 of the full corpus's 84
   rejections and all 19 of the 2026-10 scrape's are 2-player games
   rejected at the opening auction ("MB is not up for auction") — Ruby
   removes two random privates (seeded by `settings.seed`; needs Ruby's
   RNG semantics), trains per `TRAINS_REMOVE_2_PLAYER`, 70% ownership cap.
3. **Not yet implemented** (fail loudly if reached): phase-8 train
   trade-in enforcement (`train_trade_allowed` discounts are data-wired
   but unenforced), the `grid_market` optional rule (154 full-corpus
   games skipped; market data present, unplumbed; 1867 declares no
   `GameTitle::optional_rules`).
4. **Small open replay quirks** (documented with probes, not emulated):
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
5. Then **1822** (Phase 2 of `docs/multi_title_roadmap.md`).

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
- **Impassable borders are data only** (`HexDef.borders` walls): the
  graph/adjacency is geometric. Replay-neutral (recorded routes are
  legal), but self-play enumeration could route/lay through a wall.
- **Shared-code change reachable by 1830** (seam 9): `update_market_cell`
  no longer re-appends a corp "moving" in place on the top-left cell. In
  1830 that only happens on a sold-out bump of a corp on the $60 cell;
  the new behaviour is exactly the Python oracle's. Verified with
  `parity_runner --human` on an 81-game 1830 sample (0 failures) on top
  of the standard gates.
- **Trainless-major freeze**, **cheater token slots** (Ruby `city_map`
  can append tokens past `normal_slots`), **1830 safety** (all 1867
  behaviour behind `GameTitle` hooks with 1830-inert defaults; before
  long 1830 training runs on a new engine build, run the full-corpus
  ritual in `docs/verification_rituals.md`) — unchanged from 2026-08-02.

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
  cargo test --release --no-default-features          # 172 on 2026-10-15
CUDA_VISIBLE_DEVICES="" uv run pytest tests/          # 1029 passed / 3 skipped
CUDA_VISIBLE_DEVICES="" uv run python tests/parity_runner.py --random 0:100
CUDA_VISIBLE_DEVICES="" uv run python tests/g1867_corpus_audit.py --limit 300 --json /tmp/audit.json
#   → diff game-by-game vs the previous audit JSON; rejections may only
#     move DEEPER or convert; full replays may not start mismatching
#     (a new mismatch must be shown Ruby-HEAD-equal with g1867_ruby_diff.py).
#     Also worth a look: the 2026-10 scrape (recent games, Ruby HEAD
#     replays them) and the full corpus.
```
