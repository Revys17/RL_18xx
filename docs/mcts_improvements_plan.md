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
- Action-mapper Rust migration: Rust side (`engine-rs/src/factored.rs`) is
  **feature-complete** — all action types enumerate correctly, the
  500-seed random-walk test using Rust enumeration passes 100%, and the
  parity tests (`tests/test_factored_action_helper_rust_parity.py`) cover
  the surface. The remaining work is refactoring Python's `ActionMapper`
  into a thin shim around the Rust output (slot layout, encode/decode,
  and price-head logic still live in Python). See Phase 3.5.

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

## Phase 3 — Shared cross-process inference server

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

## Phase 3.5 — Finish action-mapper Rust migration

**Why this is a phase.** Phase 4 needs the action mapper to be callable
from Rust without crossing FFI per child expansion. The Rust-side
implementation is now feature-complete; only the Python ActionMapper
refactor remains.

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

**Remaining work.**
1. Refactor Python's `ActionMapper` (`rl18xx/agent/alphazero/action_mapper.py`)
   into a thin shim around `state.get_factored_choices()`. The current
   class still owns the slot layout (`self.actions`), encode (`get_index_for_action`,
   `canonical_index_for_action`, `index_for_factored`), and decode
   (`map_index_to_action`, `map_index_to_action_with_price`). After the
   refactor, slot layout stays in Python (the model's policy head depends
   on it being stable), but the per-call enumeration and price-range
   metadata come from Rust verbatim.
2. Confirm the parity tests still cover snapped-price round-trip and the
   D-train trade-in slots.

**Risk.** Low — the Rust side is proven correct via the random-walk and
replay audits. The refactor is a code-organization change, not a
behavior change.

**Effort.** ~1–2 days.

---

## Phase 4 — Rust MCTS

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
