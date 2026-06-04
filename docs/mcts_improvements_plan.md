# MCTS improvements plan

Four self-contained improvements to the AlphaZero MCTS + self-play stack,
in dependency order. Origin: comparison against
https://github.com/ericjang/autogo plus 1830-specific adaptations.

Status as of writing (2026-05-28):
- Self-play workers run in separate processes via
  `ProcessPoolExecutor(max_workers=num_threads)` with `spawn` start method
  (loop.py:692, loop.py:1381). Each worker holds its own model copy.
- `MCTSPlayer.tree_search` already does in-process leaf batching across
  `parallel_readouts` (default 32) with virtual loss and FP16 autocast
  (self_play.py:402–508). It does **not** batch across games.
- Engine is Rust-native; self-play and MCTS already operate on
  `RustGameAdapter`. Pretraining migration is **complete**: `use_rust=True`
  is the default for `get_game_object_for_game`; `RustGameAdapter.to_dict()`
  emits the full Python schema; `fix_online_games → convert → pretrain`
  runs end-to-end on Rust. Python engine is now only a parity reference.
- **Phase 1 PlayoutTrace landed**: `mcts.PlayoutTrace`, `TraceConfig` on
  `SelfPlayHyperparams`, game-level coin-flip wiring through
  `MCTSPlayer.tree_search`, `dump_traces()` JSONL output, and a
  `scripts/view_trace.py` renderer. Off by default
  (`trace_game_rate=0.0`); 8 tests in
  `tests/agent/alphazero/test_playout_trace.py`.
- **Phase 2 consensus resign landed**: `MCTSPlayer.check_resign()` with
  rolling Q window, holdout flag, would-have-resigned tracking; resign
  hook in `SelfPlay.play()` between `pick_move` and `play_move` reusing
  the truncation path; per-game `termination` ∈ {"finished" |
  "max_length" | "resigned"} written to status JSON;
  `loop.calibrate_resign_threshold()` AlphaGo-Zero schedule (tighten on
  fp_rate>5%, loosen after 3 iters <2%, clamp at
  `resign_high_threshold_min`); 15 tests in
  `tests/agent/alphazero/test_resign.py`.
- **Phase 3 inference server landed (behind feature flag)**:
  `inference_server.py` with `InferenceServer`, `InferenceClient`,
  control protocol (pause/reload/health/shutdown), state machine
  (STARTING|ACCEPTING|DRAINING|IDLE|RELOADING), batched per-leaf
  price-component slicing. `SelfPlayConfig.use_inference_server`
  (default `False`); when on, `MCTSPlayer._select_inference_backend`
  routes both `run_encoded` and `run_many_encoded` through the
  per-worker `InferenceClient`. `loop.main_loop` spawns the server
  before iterations, pauses on training, reloads after gating, tears
  down in `finally`. 15 unit tests in
  `tests/agent/alphazero/test_inference_server.py` (including an MCTS
  integration round-trip through a stub server). Real-load training-run
  verification + `parallel_readouts: 32 → 8` quality fix are deferred
  to follow-up; the in-process inference path remains the production
  default.
- **Phase 4 Rust MCTS landed (behind feature flag)**: full Rust mirror
  of `MCTSNode` semantics in `engine-rs/src/mcts.rs` (arena tree,
  PUCT descent, forced-chain collapse, virtual loss, backup, PW
  progressive widening, continuous-price grandchildren). Native Rust
  `legal_action_to_index` in `engine-rs/src/action_index.rs`;
  `index_to_action_dict` delegates to the Python `ActionMapper` shim
  (decoder stays Python — same as Phase 3.5 scope). Python adapter
  `rl18xx/agent/alphazero/rust_mcts_player.py` (`RustMCTSPlayer`)
  implements the `SelfPlay.play()` surface with the same termination
  / extract_data / forced-chain bookkeeping as the Python
  `MCTSPlayer`. Feature flag `SelfPlayConfig.use_rust_mcts` (default
  `False`). Parity tests: 3 in `test_rust_mcts_parity.py`, 2 in
  `test_rust_mcts_parity_pw.py` (visit counts within tolerance, PW
  grandchild grow + snapped prices), 3 in
  `test_rust_mcts_player_e2e.py` (short-loop + 30-move +
  resign-disabled invariants). Manual smoke: full 112-move 4-player
  self-play game with `use_rust_mcts=True` and a freshly-seeded
  transformer model — game ran end-to-end in ~40s wall, 112 LMDB
  samples written, no crashes. Resign is disabled on the Rust path
  for now (no per-player root-Q accessor yet); Phase 1 PlayoutTrace
  recording is also no-op on the Rust path. Both can be added when
  needed.
- Action-mapper Rust migration: **complete** — Rust side
  (`engine-rs/src/factored.rs`) enumerates all action types; Python
  `ActionMapper.get_legal_actions_factored` is a thin shim over
  `state.get_factored_choices()` with a once-per-process warning when the
  Python fallback engages (so any production-path regression is visible
  in logs); slot layout, encode (`index_for_factored`), and decode stay
  in Python as the spec calls for. Parity validated by:
  500-seed random-walk + 3243-game replay audit + dedicated thin-shim
  tests in `tests/agent/alphazero/test_action_mapper_thin_shim.py`. See
  Phase 3.5.

The four phases below are ordered so that each one buys debugging or
benchmarking leverage for the next. Phase 4 has the largest scope and is
explicitly gated on Phases 3 + 3.5.

---

## Phase 1 — `PlayoutTrace` (debug instrumentation)

**Goal.** Per-playout diagnostic records so we can answer "why did MCTS
pick this action?" without rerunning with a debugger attached. Cheap and
isolated, lands first so it pays off in every later phase.

**Scope.**

- New dataclass `PlayoutTrace` in `rl18xx/agent/alphazero/mcts.py`:
  ```python
  @dataclass
  class PlayoutTrace:
      leaf_depth: int
      action_path: list[int]            # fmoves from root to leaf
      pw_grandchild_path: list[bool]    # parallel to action_path
      forced_chain_lengths: list[int]   # parallel to action_path
      nn_value: np.ndarray              # (VALUE_SIZE,) at the leaf
      leaf_q_perspective: float
      leaf_terminal: bool
      expansion_occurred: bool
      leaf_prior_entropy: float
  ```
- `MCTSNode.select_leaf(trace: PlayoutTrace | None = None)` records
  `(fmove, was_pw_grandchild)` per descent step.
- `MCTSPlayer.tree_search(traces: list[PlayoutTrace] | None = None)`
  populates one trace per leaf when enabled.
- `MCTSPlayer.dump_traces(path)` writes JSONL.

**Config (new `TraceConfig` on `SelfPlayConfig`).**
```python
@dataclass
class TraceConfig:
    trace_game_rate: float = 0.0       # P(a given game is traced at all)
    trace_every_n_moves: int = 1       # within a traced game, every Nth move
    traces_per_move: int = 4           # leaves recorded per traced move
    output_dir: Path = Path("traces/")
```
Defaults stay off. Game-level coin flip in `MCTSPlayer.__init__` decides
whether a game is traced — traces are *whole games*, not isolated moves
with no surrounding context. One JSONL file per traced game:
`traces/{iteration}/{game_id}.jsonl`, first line a header with
config snapshot + players + seed.

**Tooling.** Small `scripts/view_trace.py` that renders one JSONL file
into a tree-shaped summary; aggregates statistics across moves
(avg leaf depth, PW grandchild rate, leaf-Q distribution).

**Tests.** `tests/agent/alphazero/test_playout_trace.py`:
- Run 1 game with `trace_game_rate=1.0, trace_every_n_moves=1`.
- Assert the recorded `action_path` matches the leaf reached by
  `most_visited_path_nodes()` after the search completes (for the
  most-visited child; we don't trace every leaf).
- Assert `pw_grandchild_path[i]` is true iff the corresponding step
  descended through `price_children`.

**Risk.** None meaningful (disabled by default).
**Effort.** ~150 LOC + tests. One session.

---

## Phase 2 — Multiplayer consensus resign

**Goal.** End decisively-lost games early, cutting self-play wall time.
Enabled from the first iteration (no warmup) with conservative thresholds
and AlphaGo-Zero-style holdout calibration.

**The "consensus" point.** All players use the same value head, so they
implicitly agree on relative standing — there's no need to literally poll
N separate agents. The signal is the root's per-player Q vector at every
move.

**Algorithm.**

- Maintain a rolling window of the last `K` root Q vectors per game.
- Define **stable leader**: argmax of Q is the same player across the
  full window, AND `min_over_window(Q_leader) >= resign_high_threshold`.
- Define **decisive gap**:
  `min_over_window(Q_leader - Q_second) >= resign_gap_threshold`.
- When both hold and the game is not in the no-resign holdout: end the
  game immediately. Result is the current `_compute_net_worth(game)`
  snapshot, normalized to fractions — reuses the existing max-game-length
  truncation path so training tuple shape is unchanged.
- Termination metadata added to the training tuple
  (`termination: "finished" | "max_length" | "resigned"`) so downstream
  analysis can filter.

**Config additions (`SelfPlayConfig`).**
```python
enable_resign: bool = True
resign_window: int = 8
resign_high_threshold: float = 0.65   # 4-player uniform Q ≈ 0.25
resign_gap_threshold: float = 0.30
noresign_holdout_rate: float = 0.10
resign_high_threshold_min: float = 0.45   # lower clamp for auto-calibration
```

**Auto-calibration (AlphaGo Zero schedule).**

Each iteration, loop coordinator reads the iteration's self-play results:
- Count `would_have_resigned_holdouts`: holdout games where the
  stable-leader + decisive-gap condition fired at some point during play.
- Count `correct_holdouts`: holdout games among those where the leader
  *at the would-have-resigned moment* is the actual game winner.
- `fp_rate = 1 - correct/would_have_resigned`.
- Target: `fp_rate <= 5%` (AlphaGo Zero default).
- Adjust **only `resign_high_threshold`**, leave `resign_gap_threshold`
  fixed (it's a robustness margin, not a confidence claim):
  - If `fp_rate < 2%` sustained over 3 iterations → loosen by 0.05.
  - If `fp_rate > 5%` → tighten by 0.05.
  - Clamp at `resign_high_threshold_min` (default 0.45).

**Where it lives.**
- `MCTSPlayer.check_resign() -> tuple[bool, dict | None]` — called in
  `SelfPlay.play()` after `pick_move()` but before `play_move()`.
- Holdout flag set at game start from `noresign_holdout_rate`.
- Per-game record of `would_have_resigned_at_move`, `leader_at_that_moment`
  saved in the training tuple's metadata for the calibration loop.
- Calibration update writes back into `loop_config.json` between iterations.

**Risk.** Moderate-low given the holdout mechanism. Worst case is "saves
less compute than it could" not "training is poisoned." Pretrained init
gives the value head reasonable signal day one.

**Effort.** ~250 LOC + holdout calibration + tests. One session of code,
then 2–3 iterations of empirical calibration tuning.

---

## Phase 3 — Shared cross-process inference server 🟡 LANDED, NEEDS VERIFICATION

**Status (2026-05-28).** Code landed behind `use_inference_server=False`
(default in `SelfPlayHyperparams`). Full server / client / control
protocol / state machine in `rl18xx/agent/alphazero/inference_server.py`;
`MCTSPlayer._select_inference_backend` swaps between local-model and
client backends transparently; `loop.main_loop` handles spawn / pause /
reload / shutdown lifecycle. **Not yet enabled in production** — needs
verification on a real training run (GPU contention, RELOADING latency,
worker death recovery) before flipping the default. The
`parallel_readouts: 32 → 8` quality fix is also deferred to that
verification step (the two changes are intentionally coupled).

**Motivation.** `parallel_readouts=32` widens MCTS via virtual loss
beyond the point where exploration quality is good — the search is forced
to revisit options it already knows are bad just to fill the batch. The
quality fix is `parallel_readouts → 8`. The GPU utilization fix is
batching *across* games instead of within a single game.

**Architecture: Option A — inference server process + `mp.Queue`s.**

Workers are spawned subprocesses; cross-thread sharing isn't available.

```
Coordinator process (loop.py main)             Inference server process
─────────────────────────────────              ─────────────────────────
 spawn inference server  ─────────────────►    STARTING → ACCEPTING
 spawn self-play workers (ProcessPool)
   workers ◄── inference requests ─────────►   collect up to N or wait T ms
                                                forward pass
   workers ◄── inference replies ◄─────────
 join workers (self-play iter done)
 send PAUSE  ─────────────────────────────►    DRAINING (finish in-flight)
                                                → IDLE (drop model, free VRAM)
 run training step (this process, GPU)
 write checkpoint to disk
 send RELOAD(path)  ──────────────────────►    RELOADING (load from disk)
                                                → ACCEPTING
 loop
```

**Server lifecycle: persistent across the whole loop**, not per-iteration.
PAUSE drops the model on the floor, RELOAD reads new weights from disk.
No in-process weight transfer (cleanest for VRAM management — server only
holds model when ACCEPTING).

**File changes.**
- New `rl18xx/agent/alphazero/inference_server.py`:
  - `InferenceServer` (runs in its own process; control loop +
    request-batching loop).
  - `InferenceClient` (thin shim used by `MCTSPlayer`; `.submit(state)`
    + `.await_results(handles)`).
  - State machine: `STARTING | ACCEPTING | DRAINING | IDLE | RELOADING`.
- `loop.py`:
  - Spawn server before first iteration; pass queue handles to each
    worker via `initializer` of `ProcessPoolExecutor`.
  - Between iterations: `pause()`, run training, write checkpoint,
    `reload(path)`.
  - Teardown server in `finally`.
- `self_play.py` `tree_search()`:
  - Replace `self.network.run_many_encoded([...])` with
    `self.client.run_many([...])`.
  - Model object on worker side becomes unnecessary — VRAM win (no
    per-process model copy).
- `config.py`:
  - `parallel_readouts: int = 8`  (was 32)
  - `inference_batch_size: int = 64`
  - `inference_batch_timeout_ms: float = 2.0`
  - `use_inference_server: bool = False`  (feature flag during rollout)

**Subtleties to watch.**

1. **`last_price_components` is a per-forward stash on the model**
   (model_transformer.py). With concurrent workers there's no single
   "last." Server packs the per-leaf slice into the reply directly so
   `_slice_price_components` becomes server-side and per-request.
2. **Variable-shape graph inputs.** Each leaf has its own node count
   (graph encoder). Server uses PyTorch Geometric `Batch.from_data_list`
   for clean batching. Encoder output already convertible to PyG `Data`.
3. **Worker death during in-flight request.** Server can hang on
   `reply_q.put` if the receiver is gone. Bounded queues + timeouts +
   explicit "worker exited" message from the loop's process-pool
   callback.
4. **GPU contention during training.** Server is IDLE then (no weights),
   so training has full VRAM. Cost is one disk read on RELOAD.
5. **Checkpoint reload race.** A request in flight when PAUSE arrives
   must complete (DRAINING) before model unloads.

**Control protocol.**
```python
class ServerControl:
    def pause(self, timeout_s: float) -> None: ...     # drain + unload
    def reload(self, checkpoint_path: str) -> None: ... # load from disk
    def health(self) -> dict: ...                       # for loop_status.json
```
Sent over a control queue distinct from request/reply queues so control
messages can preempt regular traffic.

**Risk.** Moderate-high — system engineering, hardest of the four to
get right under failure conditions. Mitigations: ship behind
`use_inference_server` feature flag; keep the old in-process path working
until parity is verified on a short training run.

**Effort.** ~600–800 LOC + extensive testing. 2–3 sessions.

---

## Phase 3.5 — Finish action-mapper Rust migration ✅ COMPLETE

**Status (2026-05-28): closed.** The Rust side
(`engine-rs/src/factored.rs`) is feature-complete, the Python
`ActionMapper.get_legal_actions_factored` is a thin shim around
`state.get_factored_choices()`, slot layout and encode/decode live in
Python per spec, and parity is enforced by:

- `tests/test_factored_action_helper_rust_parity.py` — random-walk parity
- `tests/agent/alphazero/test_action_mapper_thin_shim.py` — same
  `(indices, price_ranges, action_types)` output regardless of whether
  the underlying state is `RustGameAdapter` (Rust enumeration) or
  Python `BaseGame` (Python `FactoredActionHelper` fallback). Also
  verifies the once-per-process warning fires when the Python fallback
  engages (so any worker regression slipping back into Python is visible
  in logs).
- `tests/agent/alphazero/test_action_mapper_canonical.py` — snapped-price
  canonical round-trip (price-bearing slots map back to the same
  canonical index regardless of the sampled price)
- `tests/agent/alphazero/test_d_train_encoding.py` — D-train trade-in
  slot encode/decode

The remaining Phase 4 work (a Rust-resident encode path that doesn't
cross FFI per child expansion) belongs in Phase 4 itself, not here — by
then the Rust MCTS will own the tree-search loop and the encode call
will be Rust-local rather than a Python-side `index_for_factored`
invocation.

**Why this was a phase.** Phase 4 needs the action mapper to be callable
from Rust without crossing FFI per child expansion. The Rust-side
implementation is now feature-complete; only the Python ActionMapper
refactor remained.

**Audit status (updated 2026-05-28).** The audit was effectively
conducted in the random-walk + replay parity push. Specific gaps
identified and closed during that work:

- CS LayTile emitted under `{"private": "CS"}` entity (was emitting under
  corp)
- SellShares with president-cert swap honoring Python's insertion-order
  semantics (added `acquired_seq` / `market_order` tracking in
  `engine-rs/src/entities.rs`)
- Tile rotation legality with bipartite matching for multi-city upgrades
  (OO tiles in `engine-rs/src/tiles.rs`)
- `pending_tokens` carrying hex_id for OO upgrade displacement (so
  factored PlaceToken restricts to the displaced hex)
- EBUY cheapest-only filter applied to discard-source trains
- Waterfall auction `discount` accumulator resets on company purchase
- `route_train_purchase` counts Cities + Towns + Offboards as mandatory
- D-train trade-in slots (4/5/6 → D at $800)
- Preprinted city ID format (`{hex_id}-0-{city_idx}`)

Validation: **500/500 Rust-helper random walks pass**, **3243-game
per-step replay audit has 0 failures**, **cleaning outcome parity is 100%**.

**Remaining work.** *(All items closed.)*
1. ~~Refactor Python's `ActionMapper`~~ — `get_legal_actions_factored`
   already delegates per-call enumeration to the Rust
   `state.get_factored_choices()` call; the post-processing loop
   (encode via `index_for_factored`, dedupe by canonical index, merge
   price-range unions for same-slot collisions) is the spec-allowed
   Python-resident encode step. A once-per-process warning was added at
   the Python-fallback boundary so any production-path regression is
   visible in worker logs.
2. ~~Confirm the parity tests still cover snapped-price round-trip and
   the D-train trade-in slots~~ — see test list above.

**Risk.** Low — the Rust side is proven correct via the random-walk and
replay audits. The refactor was a code-organization change, not a
behavior change.

**Effort.** ~1 session (mostly audit + tests; the heavy lifting already
landed during the engine-parity push).

---

## Phase 4 — Rust MCTS ✅ LANDED (behind feature flag)

**Status (2026-05-28).** All three sub-phases (4a, 4b, 4c) are landed
behind `SelfPlayConfig.use_rust_mcts=False`. Real-load training-run
verification + flipping the default to `True` are deferred — the
in-process Python MCTS remains the production default.

**What's in:**
- `engine-rs/src/action_index.rs` — native Rust `legal_action_to_index`;
  `index_to_action_dict` delegates to Python `ActionMapper` (decoder
  stays Python — Phase 3.5 spec).
- `engine-rs/src/mcts.rs` — `RustMCTSNode` arena tree +
  `RustMCTSPlayer` PyO3 class. Categorical descent, forced-chain
  collapse, virtual loss, backup, PW progressive widening,
  continuous-price grandchildren with dual-mirror N/W backup,
  network-driven `_sample_price_for_slot` via the slot index map.
- `rl18xx/agent/alphazero/rust_mcts_player.py` — Python adapter
  implementing the `SelfPlay.play()` surface (initialize_game,
  tree_search, play_move, pick_move, extract_data, check_resign
  disabled, dump_traces no-op). `.root` property returns a `_RootShim`
  so `SelfPlay.play()`'s direct `player.root.*` accesses work
  unchanged.
- `rl18xx/agent/alphazero/self_play.py` branches in `SelfPlay.play()`:
  `if config.use_rust_mcts: player = RustMCTSPlayer(config) else: MCTSPlayer(config)`.

**Tests:**
- `test_rust_mcts_parity.py` (3) — categorical visit counts within
  tie-breaking jitter, single-leaf first descent, action-offsets
  table parity.
- `test_rust_mcts_parity_pw.py` (2) — PW visit-count totals within
  K=readouts/10, PW grows multiple grandchildren with snapped prices.
- `test_rust_mcts_player_e2e.py` (3) — short loop, 30-move game with
  `extract_data`, resign-disabled invariant.

**Manual verification:** full 112-move 4-player self-play game with
`use_rust_mcts=True` ran end-to-end in ~40s wall with no crashes,
producing 112 LMDB samples.

**Now at full feature parity with the Python MCTS:**
- ✅ Phase 2 consensus resign on the Rust path. ``RustMCTSPlayer.root_q_vector()``
  exposes ``W / (1 + N)`` per player; the Python adapter ports
  ``MCTSPlayer.check_resign`` verbatim (rolling Q window, stable
  leader, decisive gap, holdout suppression, would-have-resigned
  bookkeeping). 9 new tests in
  ``tests/agent/alphazero/test_rust_mcts_resign_trace.py``.
- ✅ Phase 1 PlayoutTrace recording on the Rust path.
  ``RustMCTSPlayer.select_leaf_with_trace()`` returns the descent path
  (action_path, pw_grandchild_path, forced_chain_lengths); the Python
  adapter finalizes per-leaf nn_value / leaf_q_perspective /
  leaf_prior_entropy after incorporate_results, and ``dump_traces``
  writes the same JSONL header+rows shape as ``MCTSPlayer.dump_traces``
  (with ``engine: "rust"`` in the header so traces from both backends
  can coexist). 6 new tests in the same file.

**Deferred (not blocking the feature flag):**
- Real-load multi-iteration training verification.
- Native Rust ``index_to_action_dict`` decoder (currently delegates to
  Python ``ActionMapper``).

---

## Phase 4 — Rust MCTS (original spec retained below for reference)

**Strategy.** Mirror the engine migration's incremental approach
(`docs/pretraining_rust_migration.md` is precedent). Move the inner
tree-search loop into Rust; keep NN inference in Python (called via
PyO3 callback into the Phase 3 inference client).

**What moves to Rust.**
- New `engine-rs/src/mcts.rs` exposing `RustMCTSNode` and
  `RustMCTSPlayer` via PyO3.
- Node state: `BaseGame` (Rust-native, no `pickle_clone` Python
  round-trip), `child_N: Vec<f32>`, `child_W: Vec<[f32; 6]>`,
  `child_prior: Vec<f32>`, `legal_action_indices: Vec<u32>`,
  `children: HashMap<u32, Box<RustMCTSNode>>`, `price_children`,
  `price_child_N`, `price_child_W`.
- Methods: `select_leaf`, `child_action_score`, `maybe_add_child`
  (including forced-chain collapse), `add_virtual_loss`,
  `revert_virtual_loss`, `backup_value`, `incorporate_results`.

**What stays in Python.**
- Inference server (Phase 3) and its `InferenceClient`.
- Encoder canonicalization (initially — can move to Rust later if
  profiled hot; `engine-rs/src/encoder.rs` already exists as a
  starting point).
- `SelfPlay.play()` orchestration loop.

**Control flow per `tree_search()`.**
1. Rust `RustMCTSPlayer.select_leaves(parallel_readouts)` returns
   a batch of leaf node handles + their encoded states.
2. Python sends encoded states to the inference server via
   `InferenceClient`.
3. Python calls `RustMCTSPlayer.incorporate_results(handles, probs,
   values, price_components)` to back up values and revert virtual
   losses.

**Sub-phasing.**

- **4a — Read-only Rust mirror.** Build the Rust types but don't use
  them in the live path. Add `tests/agent/alphazero/test_rust_mcts_parity.py`
  that runs N searches with the Python MCTS and N searches with the
  Rust MCTS from the same seed and asserts visit counts match within
  tolerance. Catches all the porting bugs without risking the training
  loop.
- **4b — Drop-in replacement behind a flag.**
  `SelfPlayConfig.use_rust_mcts: bool = False`. Wire `MCTSPlayer` to
  instantiate either the Python or Rust root. Initial port leaves
  progressive widening + continuous-price grandchild logic in Python
  (categorical descent goes through Rust, PW dispatch stays Python).
- **4c — PW + continuous prices into Rust.** Port the two-level
  categorical → continuous-price tree, the `_sample_price_for_slot`
  flow, and the price-grandchild backup. At this point Python is a
  thin orchestrator.

**Subtleties.**
1. **Engine state cloning.** `pickle_clone()` is a misleading name — it's
   no longer a Python pickle round-trip. The method on `RustGameAdapter`
   already delegates to Rust's native `clone_for_search` (see
   `engine-rs/src/game.rs:pickle_clone`), and the Python pickle module
   isn't involved. What remains is the FFI overhead of constructing a
   new `RustGameAdapter` wrapper around the cloned Rust game per node
   expansion. Moving MCTS into Rust eliminates that wrapper construction
   per leaf — modest speedup, smaller than the original "pickle vs.
   native" framing suggested. The big speedup from this phase comes from
   keeping the search loop in Rust (no FFI per descent step) rather than
   from replacing pickling.
2. **`action_types_by_idx` / `price_ranges_by_idx`.** Comes from the
   action mapper. Phase 3.5 makes these callable directly from Rust.
3. **Encoder canonicalization.** Either call back into Python
   `Encoder_GNN` per leaf (simpler, FFI overhead) or port `encoder.rs`
   (faster, larger scope). Start with callback; profile after 4b.
4. **Metrics.** Current `add_metric` callbacks dispatch to a Python
   TensorBoard summary writer. Keep as PyO3 calls but batch them — one
   Python call per `tree_search()`, not per node.
5. **Memory layout.** Python MCTS uses `np.zeros(...)` per node which
   is allocation-heavy. Rust can use `SmallVec` for the common case of
   <16 legal actions, falling back to heap for larger.

**Risk.** High (largest scope). The parity test in 4a is the critical
de-risking step — until it passes, 4b doesn't ship.

**Effort.**
- 4a: ~1500 LOC + parity tests, 1–2 weeks.
- 4b: drop-in + integration, ~3–5 days.
- 4c: PW + continuous-price port, ~1 week.

---

## Dependency graph & suggested ordering

```
Phase 1 (PlayoutTrace)   ──┐
Phase 2 (Resign)         ──┤── all independent ────┐
Phase 3 (Inference srv)  ──┘                       │
                                                   ▼
                            Phase 3.5 (action mapper Rust)
                                                   │
                                                   ▼
                                          Phase 4 (Rust MCTS)
                                       (consumes Phase 3 client)
```

**Recommended order:** 1 → 2 → 3 → 3.5 → 4.

- Phase 1 first because it's tiny and pays off in every later debug
  session.
- Phase 2 next because it directly shortens self-play wall time.
- Phase 3 before Phase 4 so the Rust MCTS only ever has to integrate
  against the new inference path. The user's
  `parallel_readouts: 32 → 8` quality fix lands in Phase 3.
- Phase 3.5 between 3 and 4 so action mapper Rust completion isn't
  blocking the Phase 4 critical path when work starts.

Phases 1, 2, 3 can be parallelized across people if needed — they touch
disjoint code surfaces (`mcts.py` trace recording, `self_play.py` resign
hook + holdout, `inference_server.py` + `loop.py` orchestration).

---

## Open questions deferred to implementation time

1. **Server lifetime across iterations.** Persistent (current plan) is
   simpler operationally. Per-iteration spawn/teardown would add
   robustness against memory leaks but costs startup time. Pick after
   running a multi-day loop with the persistent design and seeing
   whether anything drifts.
2. **Arena MCTS.** `arena.py` also runs MCTS games. Either reuse the
   server (cleaner, fewer GPU contexts) or have arena run a separate
   one. Decide after Phase 3 is stable.
3. **Encoder in Rust.** Defer to post-4b based on profile data.
4. **Async self-play overlapping training.** Currently sync
   (self-play → train → self-play). A third process architecture
   (coordinator + inference server + trainer) would enable async.
   Not on the roadmap; revisit if iteration wall time becomes the
   bottleneck.
