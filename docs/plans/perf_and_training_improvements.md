# Performance & Training Improvements Plan

Items 2-8 from the improvement list. Item 1 (pretraining on human games) is being done separately.

## Status Key
- [ ] Not started
- [x] Done

---

## 2. Run With New Schedules

**Status:** [x] Done (implemented this session)

Game length schedule (150→1000) and readout schedule (64→200) are wired up and controlled via CLI:
```bash
uv run python main.py train \
  --game-length-schedule 150 1000 150 \
  --readout-schedule 64 200 150
```
Both ramp linearly over 150 checkpoints. Schedule values are logged per-iteration in `metrics_history.jsonl`.

---

## 3. Shared Inference Server

**Status:** [ ] Not started
**Impact:** High — biggest single perf unlock after schedules
**Effort:** Medium-high (2-3 sessions)

### Problem
Each self-play process loads its own model copy and runs inference independently. With 8 threads, GPU utilization is poor — each process does batch-32 inference (from `parallel_readouts`) sequentially, and the GPU idles between processes.

### Design

Replace per-process model loading with a centralized inference server:

```
┌─────────────┐     ┌─────────────┐
│ Self-play 1  │────>│             │
│ Self-play 2  │────>│  Inference  │──> GPU (single model)
│ Self-play 3  │────>│   Server    │
│ ...          │────>│             │
└─────────────┘     └─────────────┘
```

**Architecture:**
1. **Server process** runs in a separate process, owns the GPU model
2. **Request queue** (`multiprocessing.Queue`): self-play processes push `(request_id, encoded_states)` tuples
3. **Batch collector**: server drains the queue, accumulates until batch is full or timeout (~5ms), runs single forward pass
4. **Response queues** (one per worker, `multiprocessing.Queue`): server pushes `(request_id, probs, values)` back

**Key files to modify:**
- `loop.py:275` (`run_self_play`): remove `get_latest_model()`, replace with queue-based inference client
- `self_play.py:218-228`: replace `self.network.run_many_encoded(...)` with `inference_client.request(encoded_states)`
- New file: `rl18xx/agent/alphazero/inference_server.py`

**Gating consideration:** When gating is re-enabled, the server needs to handle two models simultaneously (candidate + current best). Options:
- Run two server instances (simplest, uses 2x model memory = ~220MB — fits easily)
- Swap model weights in-place between gating games (gating is sequential, not parallel with self-play)

**Implementation steps:**
1. Create `InferenceServer` class with start/stop lifecycle
2. Create `InferenceClient` class with `request(encoded_states) -> (probs, values)` blocking call
3. Modify `run_self_play` to accept an `InferenceClient` instead of loading model
4. Modify `MCTSPlayer` / `SelfPlay` to use client instead of `self.network`
5. Start server in `main()` before the loop, pass client references to workers
6. Add server metrics (batch size distribution, queue depth, latency)

---

## 4. Continuous Self-Play

**Status:** [ ] Not started
**Impact:** High — eliminates idle time between phases
**Effort:** Medium (1-2 sessions, easier after inference server)
**Depends on:** Item 3 (inference server) — much cleaner with shared inference

### Problem
The loop is synchronous: generate N experiences → train → repeat. GPU idles during self-play, CPU idles during training.

### Design

Decouple self-play and training into concurrent processes:

```
┌──────────────────┐        ┌──────────────┐
│  Self-play pool   │──LMDB──│   Training   │
│  (continuous)     │        │   (periodic) │
└──────────────────┘        └──────────────┘
         │                          │
         └──── Inference Server ────┘
```

**Self-play workers** run continuously, writing to LMDB. When a new checkpoint is saved, workers pick it up on their next game start.

**Training process** wakes up on a trigger (every N new experiences, or on a timer), reads the latest `max_training_window` examples from LMDB, trains for `num_epochs`, saves a new checkpoint.

**Key changes:**
- `loop.py`: split the main loop into two threads/processes — one managing self-play workers, one managing training
- LMDB writes need to be safe for concurrent read/write (LMDB supports this natively with `MDB_NOLOCK` for readers)
- Model hot-swap: self-play workers call `get_latest_model()` at the start of each new game, so they naturally pick up new checkpoints
- Training trigger: poll LMDB entry count, train when `new_entries >= target_experiences`

**Implementation steps:**
1. Extract training phase into a standalone function that can run independently
2. Create a `SelfPlayManager` that runs workers continuously (no experience target — just keep going)
3. Create a `TrainingManager` that watches LMDB and triggers training
4. Wire both into `main()` with `threading` or `multiprocessing`
5. Handle checkpoint promotion / gating in the training manager

---

## 5. Separate Learning Rate for Value Head

**Status:** [ ] Not started
**Impact:** Medium — may help value head escape mean-collapse faster
**Effort:** Low (30 min)

### Problem
Value head gradients are large (`grad_norm_value: 0.43`) but the trunk barely responds (`grad_norm_trunk: 0.04`). The value head is collapsing to the mean. A higher LR for the value head could help it pull the trunk toward value-relevant features.

### Implementation

In `train.py:152-154`, replace the single optimizer with parameter groups:

```python
optimizer = optim.Adam([
    {"params": value_head_params, "lr": config.lr * config.value_lr_multiplier},
    {"params": other_params, "lr": config.lr},
], weight_decay=config.weight_decay, betas=(0.9, 0.999), eps=1e-8)
```

**Key files:**
- `train.py:152`: optimizer creation
- `config.py:164`: add `value_lr_multiplier: float = 3.0` to `TrainingConfig`

**Partition logic:** iterate `model.named_parameters()`, put anything matching `"value_head"` into one group, everything else into another.

**Risk:** Low — if the multiplier is too aggressive, value loss will oscillate. Start with 3x, monitor `grad_norm_value` and `value_loss` stability. The grad clip (`max_norm=1.0`) provides a safety net.

---

## 6. Log Per-Game Move Counts by Phase

**Status:** [ ] Not started
**Impact:** Medium — diagnostic, tells you if curriculum is working
**Effort:** Low (30 min)

### Problem
No visibility into how game time is distributed across phases (auction, stock, operating). Can't tell if the model is learning to progress through phases faster.

### Implementation

In `SelfPlay.play()` (`self_play.py:470+`), track phase transitions during the play loop:

```python
phase_move_counts = {"Auction": 0, "WaterfallAuction": 0, "Stock": 0, "Operating": 0}
```

On each move, increment `phase_move_counts[game.round.__class__.__name__]`.

After the game ends, include `phase_move_counts` in the game status JSON (already written to `self_play_games_status/`). Aggregate in `loop.py` and log to `metrics_history.jsonl` and TensorBoard.

**Key files:**
- `self_play.py:470-560`: play loop where phase tracking goes
- `self_play.py:536`: status update call — add phase counts
- `loop.py:940-980`: metrics aggregation — average phase counts across games

---

## 7. FP16 Training

**Status:** [ ] Not started
**Impact:** Medium — halves activation memory, allows larger batch or frees VRAM for inference server
**Effort:** Low (1 hour)

### Problem
Training runs in FP32. Self-play inference already uses FP16 via `torch.amp.autocast`. Training could do the same.

### Implementation

In `train.py`, wrap the forward + loss computation in `torch.amp.autocast("cuda")` and use `GradScaler`:

```python
scaler = torch.amp.GradScaler("cuda")

with torch.amp.autocast("cuda"):
    policy_logits, value_pred, aux_pred = model(game_state_data, batch_data)
    # ... compute losses ...

scaler.scale(total_loss / accum_steps).backward()
if (batch_idx + 1) % accum_steps == 0:
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
```

**Key files:**
- `train.py:216-274`: forward + backward block to wrap
- `train.py:152`: add GradScaler creation
- `config.py`: add `use_fp16_training: bool = True` to `TrainingConfig`

**Considerations:**
- Only enable on CUDA (not MPS, not CPU)
- The `GradScaler` handles loss scaling to prevent FP16 underflow in gradients
- The KL divergence value loss and log_softmax are numerically sensitive — `autocast` keeps these in FP32 automatically
- Optimizer state remains FP32 (Adam master weights) — this is handled by `GradScaler`

---

## 8. Adaptive MCTS Readouts (Short-Circuit Low-Action Positions)

**Status:** [ ] Partially done
**Impact:** Medium — ~60-70% of positions have <=2 legal moves
**Effort:** Low (30 min)

### Current State
Forced moves (exactly 1 legal action) are already short-circuited in three places:
- `self_play.py:494` — skips MCTS entirely in the play loop
- `self_play.py:264` — `suggest_move()` returns directly
- `self_play.py:147` — `pick_move()` returns the only action

### Problem
Positions with 2-3 legal actions still run full MCTS (64-200 readouts). With `mean_legal_actions: 2.5`, most positions don't benefit from deep search.

### Implementation

Add a `min_readouts_threshold` to `SelfPlayConfig` (e.g., 5 legal actions). When `num_legal_actions <= threshold`, use `config.min_readouts` (default 50) instead of `config.num_readouts`.

In `SelfPlay.play()` at `self_play.py:490-498`, after the `num_legal_actions == 1` check:

```python
if player.root.num_legal_actions == 1:
    # existing short-circuit
elif player.root.num_legal_actions <= self.config.adaptive_readout_threshold:
    effective_readouts = self.config.min_readouts
else:
    effective_readouts = self.config.num_readouts
```

Then pass `effective_readouts` to the MCTS search loop instead of always using `config.num_readouts`.

**Key files:**
- `config.py:204`: add `adaptive_readout_threshold: int = 5` to `SelfPlayConfig`
- `self_play.py:490-498`: add the threshold check
- `self_play.py:250-260`: `suggest_move()` uses `self.config.num_readouts` — make it accept an override

**Expected impact:** With mean_legal_actions of 2.5, roughly 70% of positions have <=5 legal actions and would use `min_readouts` (50) instead of `num_readouts` (64-200). At the high end of the readout schedule (200), this saves ~75% of inference for those positions.

---

## Priority Order

| Priority | Item | Impact | Effort | Dependencies |
|----------|------|--------|--------|-------------|
| 1 | 5. Value LR multiplier | Medium | 30 min | None |
| 2 | 8. Adaptive readouts | Medium | 30 min | None |
| 3 | 6. Phase move logging | Medium | 30 min | None |
| 4 | 7. FP16 training | Medium | 1 hour | None |
| 5 | 3. Inference server | High | 2-3 sessions | None |
| 6 | 4. Continuous self-play | High | 1-2 sessions | 3 |

Items 5, 6, 8 can be done in a single session. Item 7 in one more. Items 3-4 are the big lifts.
