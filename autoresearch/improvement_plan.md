# RL 18xx Model Improvement Plan

## Context

We've completed ~40 autoresearch experiments, improving combined_loss from 3.653 to 3.380 (7.5%) through architectural and training changes. We also fixed a critical bug where the value head was stuck at MSE=1.0 by switching to cross-entropy loss. This plan covers 10 work items to further improve the model, ordered by implementation sequence.

---

## 1. Retest Gated Fusion After Value Loss Change

**Why:** Gated fusion learns to route between MLP (game state) and GNN (map) branches. With the value head now producing real gradients through the shared trunk, the gate dynamics will change — it may need to rebalance.

**Plan:**
- Run the current model (which includes gated fusion) through 3-epoch eval to establish a new baseline with the CE value loss
- If the baseline is reasonable, gated fusion stays. If combined_loss regresses significantly, revert to simple concat+linear fusion and retest

**Files:** No changes needed — just run the experiment
**Effort:** ~30 minutes (one experiment run)

---

## 2. Update Input Encoder

**Why:** The encoder is missing several high-impact game state features that are publicly observable but not encoded.

**Plan — add these features to `encode_game_state()` in encoder.py:**

| Feature | Size | Source |
|---------|------|--------|
| Operating rounds remaining in current set | 1 | `game.phase.operating_rounds - game.round.round_num` (normalized) |
| Current OR number within set | 1 | `game.round.round_num / game.phase.operating_rounds` |
| Corporation operating order | 8 | One-hot or fractional position for each corp |
| Player turn order (stock round) | 4 | Fractional position for each player relative to priority deal |
| Train limit for current phase | 1 | `game.phase.train_limit / 4` (max is 4) |
| Private company closed status | 6 | 1.0 if closed, 0.0 if open (currently only ownership is encoded) |

**Total new features:** ~21, bringing game_state_size from 377 to ~398.

**Files to modify:**
- `rl18xx/agent/alphazero/encoder.py` — add features to `encode_game_state()`, update `GAME_STATE_ENCODING_STRUCTURE`, update `_calculate_encoding_size()`
- `rl18xx/agent/alphazero/config.py` — update `game_state_size` default
- `tests/agent/alphazero/encoder_gnn_test.py` — update expected tensor shapes and add tests for new features
- `autoresearch/prepare_training_data.py` — will need to re-run to re-encode training data with new features

**Effort:** ~2-3 hours coding + ~30 min re-encoding data
**Note:** This triggers full re-encoding of training data (~30 min). Bundle all encoder changes together.

---

## 3. Factored Action Head

**Why:** The current 25,707-index flat policy head is 96.6% LayTile (24,840 indices = 93 hexes x 46 tiles x 6 rotations). The legal mask is >99.6% zeros. The final policy layer has 6.6M parameters mostly wasted on illegal actions.

**Plan — two-level factored head:**

**Level 1: Action Type Head** — predict which action type (16 types). This is always a small, dense prediction since only 1-3 action types are legal simultaneously.

**Level 2: Action Parameter Heads** — one small head per action type:
- `LayTile`: 3 sub-heads → P(hex|93) × P(tile|hex, 46) × P(rotation|hex,tile, 6) = 145 outputs vs 24,840
- `BuyTrain`: P(source|9) × P(type|6) × P(price|14) = 29 outputs vs 679  
- `Par`: P(corp|8) × P(price|6) = 14 outputs vs 48
- `SellShares`: P(corp|8) × P(quantity|5) = 13 outputs vs 40
- Other types: keep flat (all have <30 indices)

**Architecture:**
```
trunk_features (512-dim)
    ├── action_type_head → softmax over 16 types
    ├── lay_tile_hex_head → softmax over 93 hexes
    ├── lay_tile_tile_head → softmax over 46 tiles (conditioned on hex via concat)
    ├── lay_tile_rotation_head → softmax over 6 rotations
    ├── buy_train_head → flat softmax over ~29 options
    ├── par_head → flat softmax over 14 options
    └── ... (other small heads)
```

**Final policy:** P(action) = P(type) × P(params|type). Legal masking applied per-head.

**Training loss:** Cross-entropy on the joint probability. The target pi from MCTS gives probability for each concrete action — decompose into type probability and conditional parameter probability.

**Key challenge:** The action_mapper's `get_index_for_action()` and `map_index_to_action()` need to work with the factored representation. Two approaches:
- **Option A (simpler):** Keep the flat 25,707 index space for MCTS/self-play compatibility. The factored head computes per-action probabilities internally but outputs a flat 25,707 vector. This means no changes to MCTS, self-play, or action_mapper.
- **Option B (cleaner):** Full factored representation end-to-end. Requires changing MCTS, self-play, and action_mapper.

**Recommendation:** Option A — factored computation but flat output interface. This is a model-only change that doesn't touch locked files.

**Files to modify:**
- `rl18xx/agent/alphazero/model.py` — new `FactoredPolicyHead` class replacing the single linear policy head
- `rl18xx/agent/alphazero/config.py` — add factored head config params
- `rl18xx/agent/alphazero/action_mapper.py` — add methods to decompose flat indices into (type, params) tuples for the factored head's internal masking. Add methods to provide per-type legal masks.
- `tests/agent/alphazero/model_test.py` — update expected output shapes, add factored head tests

**Effort:** ~4-6 hours. This is the most complex change.

---

## 4. Model Gating + Arena Evaluation

**Why:** Currently the training loop (`loop.py:383-388`) always saves the latest model with no quality check. A bad training run propagates to self-play, creating a death spiral. AlphaZero only promotes models that beat the previous best 55%+.

**Plan:**
1. After training, run arena evaluation: new model vs current best (both with MCTS, 0 noise, 0 temperature)
2. Play N games (e.g., 10-20). New model gets 2 seats, old model gets 2 seats (alternating positions).
3. If new model's win rate >= 55%, promote it. Otherwise, keep the old model for next self-play iteration.
4. Save both models (promoted and rejected) for analysis, but only the promoted model is used for self-play.

**Implementation:**
- Add `evaluate_model(new_model, old_model, num_games)` function to `loop.py`
- Use existing `Arena` class from `arena.py` — it already supports 4-player matches
- Create `MCTSAgent` instances with `softpick_move_cutoff=0, dirichlet_noise_weight=0` (deterministic play, already done in `arena.py:test_mcts_agent_against_random_agent`)
- Add `--gate-games` CLI arg to loop.py (default 10)
- Add `--gate-threshold` CLI arg (default 0.55)

**Files to modify:**
- `rl18xx/agent/alphazero/loop.py` — add gating logic after training phase
- `rl18xx/agent/agent.py` — verify MCTSAgent interface supports arena use

**Effort:** ~2-3 hours

---

## 5. Training Data Windowing

**Why:** The LMDB grows monotonically. Old data from weak early models pollutes training as the model improves. AlphaZero uses a sliding window of the most recent ~500K games.

**Plan:**
- Add `max_training_examples` config parameter (default: keep all, for backward compat)
- Before training, if total examples exceed the window, create a new LMDB with only the most recent N examples
- Alternatively: use index-based windowing in the DataLoader (skip first K entries). This is simpler and doesn't require LMDB manipulation.

**Simpler approach:** Modify `SelfPlayDataset.__init__` to accept an optional `start_index` parameter. The DataLoader only sees examples from `start_index` to `length`. The loop calculates `start_index = max(0, total - window_size)` before each training phase.

**Files to modify:**
- `rl18xx/agent/alphazero/dataset.py` — add `start_index` support to `SelfPlayDataset`
- `rl18xx/agent/alphazero/config.py` — add `max_training_window` to `TrainingConfig`
- `rl18xx/agent/alphazero/loop.py` — pass window parameter when creating dataset

**Effort:** ~1 hour

---

## 6. Fix Child W Seeding

**Why:** `mcts.py:327` seeds every child's W with the parent's network value before any visits: `self.child_W_compressed = np.tile(value, (self.num_legal_actions, 1))`. This biases Q estimates toward the parent's value and can prevent proper exploration of moves that change the evaluation.

**Plan:**
- Change child_W initialization to zeros: `self.child_W_compressed = np.zeros((self.num_legal_actions, VALUE_SIZE))`
- The child_N is already initialized to 0, so Q = W/(1+N) = 0 for unvisited children
- Unvisited children will be explored via the U (exploration) term in UCB, which is driven by the prior policy

**Risk:** The original comment says this prevents collapse when one player is clearly winning. Need to test empirically. If it causes issues, a softer approach is to seed with a fraction: `0.5 * value` instead of full `value`.

**Files to modify:**
- `rl18xx/agent/alphazero/mcts.py` — change line 327
- `tests/agent/alphazero/mcts_test.py` — update `test_backup_incorporate_results` if it checks child_W values

**Effort:** ~30 minutes + testing

---

## 7. Update Dirichlet Noise Factor

**Why:** Current `alpha=0.03` is copied from Go (branching factor ~250). AlphaZero formula: `alpha = 10/avg_branching_factor`. For 1830, typical branching is 6-36 moves, so alpha should be ~0.3-1.7.

**Plan:**
- Change from fixed alpha to adaptive: `alpha = 10.0 / num_legal_actions`
- This automatically scales with the action space at each position
- During auction (few actions): high alpha = strong exploration
- During LayTile (many actions): lower alpha = focused exploration

**Implementation:**
```python
# In MCTSNode.inject_noise():
alpha_value = 10.0 / self.num_legal_actions
alpha = np.full(self.num_legal_actions, alpha_value)
```

**Files to modify:**
- `rl18xx/agent/alphazero/mcts.py` — update `inject_noise()` method
- `rl18xx/agent/alphazero/config.py` — remove `dirichlet_noise_alpha` or keep as fallback
- `tests/agent/alphazero/mcts_test.py` — update noise tests

**Effort:** ~30 minutes

---

## 8. Update Training Epoch Count

**Why:** Currently 1 epoch per iteration. The model barely fits each batch of new data before generating more. Our autoresearch showed 3 epochs is significantly better.

**Plan:**
- Change `TrainingConfig.num_epochs` default from 1 to 3
- Consider making it configurable via loop.py CLI: `--num-epochs`

**Files to modify:**
- `rl18xx/agent/alphazero/config.py` — change default
- `rl18xx/agent/alphazero/loop.py` — add CLI arg

**Effort:** ~15 minutes

---

## 9. Update Parallel Readouts

**Why:** Default `num_readouts=64` (from loop.py CLI) is very low. AlphaZero used 800. The config default is 200 which is more reasonable but still low for 1830's complexity.

**Plan:**
- Change loop.py default from 64 to 200 (matching SelfPlayConfig default)
- Consider 400 for better quality self-play (at the cost of 2x slower game generation)
- Add a note that this should be tuned based on available compute

**Files to modify:**
- `rl18xx/agent/alphazero/loop.py` — change CLI default

**Effort:** ~5 minutes

---

## 10. Migrate Engine to Rust

**Why:** `pickle_clone()` is called thousands of times per MCTS search and is the primary bottleneck. Rust would give 50-100x speedup on cloning and game state operations.

**Assessment from exploration:**
- Engine is ~18,400 lines across 13 files
- No blocking Python patterns (no eval/exec, no deep monkey patching)
- Main challenge: `pickle_clone()` relies on Python's automatic serialization
- Estimated effort: **19-28 weeks** for full migration

**Recommended phased approach:**

**Phase 1 (2-3 weeks):** Rust core with PyO3 bindings for hot paths only
- Implement `BaseGame` state representation in Rust
- Implement `pickle_clone()` equivalent (serde-based deep copy, ~100x faster)
- Implement `process_action()` for the most common actions
- Python wrapper exposes same API — zero changes to encoder/MCTS/self-play

**Phase 2 (4-6 weeks):** Migrate round logic and graph
- Port `round.py` (5,763 lines) — turn sequencing, action validation
- Port `graph.py` (3,587 lines) — route calculation, hex topology
- Port `entities.py` (1,552 lines) — Player, Corporation, etc.

**Phase 3 (3-4 weeks):** Full engine in Rust
- Port remaining files (actions.py, core.py, abilities.py, autorouter.py)
- Port g1830.py game definition
- Remove Python engine entirely

**Phase 4 (2-3 weeks):** Testing and validation
- Port or regenerate all engine tests
- Validate game output matches Python version on corpus of human games
- Performance benchmarking

**Files to create:**
- `engine/Cargo.toml` — Rust project config
- `engine/src/lib.rs` — PyO3 module entry point
- `engine/src/game.rs`, `engine/src/entities.rs`, etc.
- `pyproject.toml` — add maturin build dependency

**Total effort:** 11-16 weeks phased, or 19-28 weeks all-at-once

---

## Implementation Order

| Priority | Item | Effort | Impact | Dependencies |
|----------|------|--------|--------|-------------|
| 1 | Retest gated fusion | 30 min | Validation | None |
| 2 | Dirichlet noise fix | 30 min | High (exploration quality) | None |
| 3 | Training epochs (1→3) | 15 min | High (proven in autoresearch) | None |
| 4 | Parallel readouts (64→200) | 5 min | Medium (search quality) | None |
| 5 | Child W seeding fix | 30 min | Medium (search correctness) | None |
| 6 | Training data windowing | 1 hr | Medium (data quality) | None |
| 7 | Input encoder update | 2-3 hrs | High (feature coverage) | Triggers re-encoding |
| 8 | Factored action head | 4-6 hrs | High (parameter efficiency) | None |
| 9 | Model gating + arena | 2-3 hrs | High (training stability) | None |
| 10 | Rust engine migration | 11-16 wks | Very High (10-100x speed) | All above complete first |

## Verification

After each change:
- Run `uv run pytest tests/agent/alphazero/ -x` for unit tests
- Run `uv run python -m autoresearch.run_experiment --num-epochs 3` for eval metrics
- Compare combined_loss, policy_loss, value_loss, value_accuracy against baseline
