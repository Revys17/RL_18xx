# Autoresearch for RL 18xx — Roadmap

## Overview

Adapt [karpathy/autoresearch](https://github.com/karpathy/autoresearch) for this repo: an autonomous hill-climbing loop where an LLM agent edits model/training/encoder code, evaluates against a fixed corpus of human 1830 games, and keeps or reverts each change.

### Key design decisions

- **Training-only autoresearch (Option A):** The experiment loop does NOT run self-play. Each experiment trains on a fixed human game dataset and evaluates on a frozen evaluation corpus. ~5 min per experiment when encoder unchanged, ~15-20 min when encoder changes. Target: 80-120 experiments per overnight run.
- **Metric:** Policy cross-entropy on held-out human game positions (+ top-1/top-5 accuracy, value loss). Deterministic, fast, monotonically meaningful up to ~1600 Elo equivalent.
- **Lives inside this repo** at `autoresearch/`.
- **Encoder is mutable:** The agent can modify `encoder.py`. The evaluation harness re-encodes the human game corpus when encoder changes are detected (via file hash).
- **Human game corpus:** 250 usable games from `human_games/1830_clean/`. Split 200 training / 50 evaluation. Evaluation games are frozen and locked.

### What autoresearch targets

The agent optimizes how efficiently the network learns from human game data. Improvements transfer to self-play training because better architectures, encoders, and training dynamics are universal.

### What autoresearch does NOT target (initially)

MCTS parameters, self-play configuration, tree search efficiency. These are a smaller search space addressed in Phase 7 or manually.

---

## Phase 1: Self-play performance fixes

**Goal:** 2-5x speedup in MCTS tree search. Benefits all downstream work including the actual training runs after autoresearch.

**Why first:** Self-play speed is the bottleneck for model gating (Phase 6), replay buffer fill rate, and eventual self-play autoresearch (Phase 7). The fixes are independent of everything else.

### 1a. Cache `fmove` index in MCTS nodes

`mcts.py` — `MCTSNode.N` and `MCTSNode.W` properties call `self.parent.legal_action_indices.index(self.fmove)` on every access. This is a linear scan repeated thousands of times per game.

**Fix:** Store `self._parent_index` at node creation time in `__init__` or `maybe_add_child`.

**Difficulty:** Easy (< 1 hour)

### 1b. Pre-allocate policy array in `select_leaf`

`mcts.py` — `child_action_score` property allocates a full 26,535-element numpy array via `_expand_to_full_policy_size()` on every call during `select_leaf()`.

**Fix:** Allocate once per node and reuse, or rework `select_leaf` to operate in compressed (legal-actions-only) space.

**Difficulty:** Easy (< 1 hour)

### 1c. Profile and benchmark

Enable the existing per-move timing metrics in `self_play.py` (currently disabled via `metrics=None` at `loop.py:199`). Run before/after benchmarks on a fixed game to measure improvement.

**Difficulty:** Easy

### 1d. Investigate `pickle_clone` cost

`mcts.py:224` — `game_object.pickle_clone()` is called once per leaf expansion per readout. With 64 readouts × ~200 moves per game, this is ~12,800 clone operations per game.

**Options:** Copy-on-write game state, incremental undo/redo, cheaper serialization format.

**Difficulty:** Medium-Hard. Scope carefully — this may be a deep rabbit hole into the game engine. Profile first to confirm it's actually the dominant cost before investing.

---

## Phase 2: Remove holdout dataset, clean up training loop

**Goal:** Simplify the codebase by removing dead code paths. Can run in parallel with Phase 1.

**Context:** The holdout set currently does nothing. It feeds a `ReduceLROnPlateau` scheduler that can never trigger (`num_epochs=1`, patience=5), and the validation losses it produces are logged but never acted on. The 5% of games routed to holdout are wasted training data.

### 2a. Remove holdout split from self-play

- Remove the `random.random() < self.config.holdout_pct` coin flip in `self_play.py:544-548`
- Remove `holdout_pct` and `holdout_dir` from `SelfPlayConfig` in `config.py`
- All self-play games go to the training set

### 2b. Remove or fix LR scheduler

- `ReduceLROnPlateau` with patience=5 cannot trigger when `num_epochs=1`
- Either remove the scheduler entirely, or implement a real LR strategy (e.g., cosine decay within an epoch, or increase `num_epochs`)
- Decision: remove for now; autoresearch can explore LR strategies later

### 2c. Remove `val_dir` plumbing

- Remove `training_config.val_dir` assignment in `loop.py:367`
- Remove validation dataset loading and validation loop in `train.py` (or gate it on `val_dir` being explicitly provided)
- Remove validation loss fields from `LoopMetrics` and TensorBoard logging
- Keep `TrainingConfig.val_dir` as an optional field for pretraining use

**Difficulty:** Easy (half day total for all of Phase 2)

---

## Phase 3: Build evaluation harness

**Goal:** A locked evaluation pipeline that measures model quality on human game positions.

**Depends on:** Phase 2 (minor — just needs holdout removal so concepts don't conflict)

### 3a. Build the evaluation corpus

Script that produces the frozen evaluation dataset from human games.

**Input:** 250 clean games from `human_games/1830_clean/`

**Process:**
1. Load each `BaseGame` from its JSON serialization
2. Replay actions one by one through the game engine
3. At each action step, record:
   - The `BaseGame` state (or enough info to reconstruct it — e.g., the game init params + action history up to this point)
   - The human player's chosen action
   - The list of legal actions at this state
   - The game outcome (for value targets)
4. Split at the game level: 200 games → training, 50 games → evaluation
5. Write to `autoresearch/eval_corpus/` in a compact format

**Key design choice:** Store enough to reconstruct `BaseGame` states, NOT pre-encoded tensors. This allows re-encoding when the encoder changes. The most compact approach is to store each game as `(game_init_params, action_list, outcome)` and replay on demand. Slower but maximally flexible.

**Alternative:** Store serialized `BaseGame` snapshots at each action step. Faster evaluation but much larger on disk (~50K snapshots × game state size).

**Recommended approach:** Store `(game_id, action_index, action, legal_action_indices, outcome)` per position, plus the game file reference. The evaluation harness replays games from the clean files, which already exist. This avoids duplicating game state storage entirely.

**Difficulty:** Medium (1-2 days). Most of the replay logic exists in `pretraining.py:606` (`convert_game_to_training_data`).

### 3b. Build `autoresearch/evaluate.py`

The locked evaluation script. The autoresearch agent MUST NOT modify this file.

**What it does:**
1. Loads the 50 frozen evaluation games (by game ID list, hardcoded or in a config the agent can't touch)
2. Checks if encoder has changed since last run (hash of `encoder.py`)
3. Replays each game through the engine, encoding each position with the **current** encoder
4. Runs model forward pass on all encoded positions (batched)
5. Computes metrics:
   - `policy_loss: X.XXXXXX` — cross-entropy vs human moves (smoothed targets: 0.97 on played move, 0.03 spread over legal alternatives)
   - `top1_accuracy: X.XXXX` — fraction where model's argmax matches human move
   - `top5_accuracy: X.XXXX` — fraction where human move is in model's top 5
   - `value_loss: X.XXXXXX` — MSE vs game outcomes
6. Prints metrics on stdout in grep-friendly format (one `key: value` per line)

**Caching:** When encoder hasn't changed, cache the encoded tensors to skip re-encoding on subsequent runs. Store cache alongside a hash of encoder.py. Invalidate when hash changes.

**Difficulty:** Medium (1 day)

### 3c. Build the training data pipeline for autoresearch experiments

The 200 training games need to be available as training data in the same format the model expects.

**Process:**
1. Replay the 200 training games, encode with current encoder
2. Write to LMDB in the same format as self-play data (same `SelfPlayDataset` / `HumanPlayDataset` interface)
3. Re-encode when encoder changes (same hash-based invalidation as evaluate.py)

**This is essentially a streamlined version of the existing pretraining pipeline** but with:
- Fixed game set (200 games, not all 250)
- Hash-based re-encoding
- Output compatible with the standard training loop

**Difficulty:** Medium (1 day, mostly adapting `pretraining.py` logic)

---

## Phase 4: Autoresearch scaffolding

**Goal:** `program.md`, experiment runner, and results tracking.

**Depends on:** Phase 3

### 4a. Write `autoresearch/program.md`

The instruction file that tells the LLM agent how to operate. Defines:

**Mutable files** (agent may edit):
- `rl18xx/agent/alphazero/model.py` — architecture changes
- `rl18xx/agent/alphazero/train.py` — optimizer, LR, loss function, training dynamics
- `rl18xx/agent/alphazero/encoder.py` — feature engineering, encoding scheme
- `rl18xx/agent/alphazero/config.py` — hyperparameters, model dimensions
- `rl18xx/agent/alphazero/dataset.py` — data loading/augmentation

**Locked files** (agent must NOT edit):
- `autoresearch/evaluate.py` — evaluation harness
- `autoresearch/run_experiment.py` — experiment runner
- `autoresearch/eval_corpus/` — evaluation data
- `rl18xx/game/engine/` — game engine (entire directory)
- `rl18xx/agent/alphazero/action_mapper.py` — action space definition
- `rl18xx/agent/alphazero/pretraining.py` — human game processing

**Experiment loop:**
```
1. Read program.md, results.tsv, and current mutable files
2. Form hypothesis and edit mutable files
3. git commit -m "experiment: <description>"
4. uv run python autoresearch/run_experiment.py > run.log 2>&1
5. grep "^policy_loss:\|^top1_accuracy:\|^top5_accuracy:\|^value_loss:" run.log
6. Append to results.tsv: commit | policy_loss | top1_acc | top5_acc | value_loss | status | description
7. If policy_loss improved: status=keep. Else: git reset --hard HEAD~1, status=discard
8. Go to step 1. Do NOT pause to ask the human.
```

**Research directions to suggest:**
- Encoder: add missing game state features (private abilities, OR count, turn order, game history)
- Encoder: fix inconsistent normalization (tile rotation is raw 0-5)
- Architecture: experiment with GNN depth, heads, hidden dimensions
- Architecture: try different graph pooling (attention pooling, set transformer)
- Architecture: experiment with residual block count and width
- Training: try different optimizers (SGD+momentum like original AlphaZero, AdamW, Muon)
- Training: explore LR schedules (cosine, warmup+decay, cyclical)
- Training: experiment with loss weighting between policy and value
- Training: try label smoothing, mixup, or other regularization
- Training: experiment with batch size
- Encoder: add edge features beyond direction (tile connectivity, revenue flow)

**Constraints:**
- All changes must pass `uv run pytest tests/agent/alphazero/model_test.py` (basic smoke test)
- Policy size must remain 26,535 (action space is fixed)
- Value size must remain 4 (player count is fixed)
- Model must accept the encoder's output format (but both can change together)
- Simpler is better — a small improvement that removes code beats one that adds complexity

**Difficulty:** Easy-Medium (1 day)

### 4b. Build `autoresearch/run_experiment.py`

Wraps the full experiment cycle. This is the equivalent of `uv run train.py` in Karpathy's setup.

**Steps:**
1. Hash `encoder.py` and compare to cached hash
2. If encoder changed:
   - Re-encode 200 training games → write to LMDB
   - Re-encode 50 evaluation games → write to cache
   - Update cached hash
3. Load model (fresh init or from a baseline checkpoint — TBD)
4. Train 1 epoch on the 200-game training LMDB
5. Run evaluation on the 50-game frozen corpus
6. Print all metrics to stdout
7. Exit with code 0 on success, non-zero on crash/NaN

**Open question:** Should each experiment start from a fixed baseline checkpoint, or from the accumulated state of previous kept experiments? Autoresearch uses the latter (hill-climbing on a single train.py). For us, if we're modifying architecture/encoder, we likely need to train from scratch each time since weights aren't compatible across architecture changes. But for hyperparameter-only changes, continuing from a checkpoint is valid.

**Recommended approach:** Always train from scratch (random init) for 1 epoch. This isolates the effect of each change and avoids weight-compatibility issues. The downside is we can't measure improvements that only manifest after many epochs, but for autoresearch's hill-climbing loop this is acceptable.

**Difficulty:** Medium (1-2 days)

### 4c. Results tracking

`autoresearch/results.tsv` — tab-separated, append-only:

```
commit	policy_loss	top1_acc	top5_acc	value_loss	status	description
```

Status values: `keep`, `discard`, `crash`

Crashed runs get `0.000000` for all metrics.

---

## Phase 5: First autoresearch run

**Goal:** Validate the loop and produce real improvements.

**Depends on:** Phase 4

### 5a. Establish baseline

- Run `evaluate.py` on the current model architecture with random init + 1 epoch training
- Record baseline metrics
- Commit as the starting point on a dedicated branch (e.g., `autoresearch/run1`)

### 5b. Overnight run

- Launch Claude Code pointed at the repo
- Agent reads `program.md` and begins the experiment loop
- Target: 80-120 experiments in 8 hours
- Monitor `results.tsv` growth to confirm the loop is running

### 5c. Analyze results

- Review `results.tsv` for kept vs discarded experiments
- Review git log for the progression of changes
- Identify which research directions produced the most improvement
- Decide which changes to merge back to master

---

## Phase 6: Implement missing AlphaZero features

**Goal:** Model gating and replay buffer. Now informed by autoresearch improvements and benefiting from Phase 1 speed fixes.

**Depends on:** Phase 1 (self-play speed), Phase 5 (autoresearch improvements to the model)

### 6a. Replay buffer

Replace per-model LMDB directories with a single rolling store.

- Configurable window size (number of games or number of positions)
- New self-play data appends to the store
- Old data beyond the window is pruned
- Training samples uniformly from the entire window
- This matches AlphaZero's 500K-game sliding window design

### 6b. Model gating

After each training phase, evaluate the new model against the current best:

- Run N games (e.g., 20-40) of new model vs current best using MCTS with reduced readouts
- Both sides play all 4 seats across games (rotate positions for fairness)
- Promote new model only if win rate exceeds threshold (e.g., 55%)
- If rejected, discard the trained weights and continue self-play with the current best

**Design consideration:** With 4 players, "win rate" needs careful definition. Options:
- 1v1v1v1 where new model plays one seat and current best plays the other three — win rate = fraction of games where new model wins
- Head-to-head across multiple seat assignments

### 6c. Win rate and Elo tracking

- Populate the currently-empty `win_rates_by_player` in `LoopMetrics`
- Log win rate to TensorBoard per iteration
- Optionally implement Elo tracking across model versions (Bayesian Elo or similar)

---

## Phase 7 (Optional): Self-play autoresearch

**Goal:** Let autoresearch optimize MCTS and self-play configuration.

**Depends on:** Phases 5 + 6

**Approach:**
- Expand mutable files to include `mcts.py`, `self_play.py`
- Metric: win rate vs a fixed baseline checkpoint over N games
- Slower loop (~30-60 min per experiment) but targets a different optimization surface
- Research directions: c_puct tuning, readout count, parallel readouts, noise parameters, temperature schedule, tree reuse strategies

---

## Execution notes

### Parallelism

- Phases 1 and 2 can run in parallel (no dependencies)
- Within Phase 3, 3a and 3b can be developed concurrently once the corpus format is agreed
- Phase 4a (program.md) can be drafted while Phase 3 is in progress

### Testing

- All phases should preserve passing tests: `uv run pytest tests/`
- Phase 1 changes should be benchmarked before/after on a fixed self-play game
- Phase 3 evaluation harness should be validated against known pretraining loss values

### Risk areas

- **Phase 1d (pickle_clone):** May require deep changes to the game engine. Profile first, scope carefully, consider deferring if the other Phase 1 fixes give sufficient speedup.
- **Phase 3a (corpus format):** The choice between storing game references vs snapshots affects evaluation speed and flexibility. Game references are recommended but slower.
- **Phase 4b (train from scratch vs checkpoint):** Architecture changes force training from scratch. This limits what the agent can discover about long-training-run dynamics, but keeps experiments fast and isolated.

### File structure after completion

```
autoresearch/
  program.md              # Agent instruction file (locked)
  evaluate.py             # Evaluation harness (locked)
  run_experiment.py       # Experiment runner (locked)
  results.tsv             # Experiment log (append-only)
  eval_corpus/
    eval_game_ids.json    # List of 50 frozen evaluation game IDs
    train_game_ids.json   # List of 200 training game IDs
    cache/                # Cached encoded tensors (invalidated on encoder change)
  analysis.ipynb          # Post-hoc analysis notebook
```
