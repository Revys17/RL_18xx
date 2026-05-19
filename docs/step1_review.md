# Step 1 Review — Training-Pipeline Codebase Audit

This document is a structured review of the components called out in **Step 1** of `roadmap.md`:

- (a) Model initialization / saving / loading
- (b) Pretraining pipeline
- (c) Self-play
- (d) Model design (encoder, network, policy head, value head)
- (e) Job monitoring

Each section is grouped by severity (**HIGH**, **MED**, **LOW**) with file:line references and concrete suggested fixes. The "Top fixes" section below pulls out the items that need to land before pretraining → self-play can be trusted end-to-end.

## Resolved during walkthrough

The following items from the original findings have already been addressed in code:

- **SSME deletion** — `Encoder_SSME` (encoder.py), `AlphaZeroSSMEModel` (model.py), and `make_encoded_ssme_game_state_model_friendly` (pretraining.py) have been deleted. The dispatcher branch in `Encoder_1830.get_encoder_for_model` is removed. The "(d) MED Encoder_SSME is broken" finding below is resolved.
- **Rename `V1 → GNN`, `V2 → Transformer`** — `AlphaZeroV2Model` → `AlphaZeroTransformerModel`, `ModelConfig` → `ModelGNNConfig`, `ModelV2Config` → `ModelTransformerConfig`, `Encoder_V2` → `Encoder_Transformer`, `model_v2.py` → `model_transformer.py`. CLI flag `--model-type {v1,v2}` → `{gnn,transformer}`. Architecture name string updated from `AlphaZeroV2` to `AlphaZeroTransformer`. References to the old names below are historical (they reflect the code at the time the finding was written); the bugs themselves still apply under the new names unless noted otherwise.
- **Pretraining V2-encoder mis-dispatch (Top fix #5)** — The pretraining `isinstance(encoder, Encoder_GNN)` issue is mooted by the action-space redesign: `make_action_model_friendly` and `make_encoded_game_state_model_friendly` are slated for deletion when continuous-price PW lands (see "Design decisions" below). Until then, the dispatch still has the latent bug for the Transformer encoder; the proper short-term fix is `type(encoder) is Encoder_GNN`.

---

## Top fixes — priority ordering

If we only land a handful of these before retraining, do these:

1. **Pretraining never persists its output** (a-HIGH, b-HIGH). `do_pretraining` returns the in-memory model and exits — the warm-started weights are discarded. Add `save_model(model, model_dir)` at the end of `do_pretraining`.
2. **Train / inference distribution mismatch via canonicalization** (d-HIGH). Training canonicalizes the active player to slot 0 (incl. value target); inference uses the raw active player slot. The model sees `one_hot(0)` for the player indicator at training, `one_hot(0..3)` at inference. Either canonicalize at inference too, or remove canonicalization.
3. **Value head is a classifier in training, a regressor in MCTS** (d-HIGH). `train.py` does KL-div on softmax(value); `mcts.py` accumulates raw logits as Q. Pick one convention. Cleanest: apply softmax in `run_many_encoded` so Q is in `[0, 1]`.
4. **Gating evaluation is deterministic** (c-HIGH). `gate_games=10` gives at best 2 statistically-distinct trajectories (one per seat permutation). Promotion noise is enormous. Add small Dirichlet noise or `softpick_move_cutoff > 0` during gating.
5. **Pretraining mis-dispatches the V2 encoder** (b-HIGH). `Encoder_V2(Encoder_GNN)` so `isinstance(encoder, Encoder_GNN)` matches V2 and routes through V1's magic offsets (335, 359) — silently corrupts the V2 state vector. Make the dispatch `type(encoder) is Encoder_GNN`, add a V2 branch, or move this onto the encoder class as a method.
6. **`get_model_from_path` only walks one level** (a-HIGH). New layout is `<dir>/<arch>/<session>/...` (two levels). The fallback picks the alphabetically-greatest arch and dies on missing config. Delete `get_model_from_path` and migrate the one remaining caller (`arena.py:85`) to `get_latest_model`.
7. **CLI args fall through to the wrong parameters** (c-HIGH). `loop.py:796-807` calls `main()` positionally with a parameter list that doesn't match `main()`'s signature — `--num_threads` actually becomes `cleanup`, etc. Switch to keyword args.
8. **V2 structural matrices never initialized** (d-HIGH). `_compute_structural_matrices` is defined but unreachable; `distance_matrix=zeros`, `direction_matrix=6` (sentinel "not adjacent"). The Hex Transformer's structural bias is a uniform zero. Invoke this on first `encode()`.
9. **V2 `HexMapTransformer` uses sample-0 track connectivity for the whole batch** (d-HIGH). Comment claims "all leaves from same game tree" — false for training batches. Propagate per-sample connectivity through `StructuralAttentionBias`.

Other systemic themes (more details inline):

- **Atomic file writes.** Status JSONs and `config.json` are written with raw `open("w")` while readers (dashboard, gunicorn workers, self-play workers) poll them concurrently. Pattern: write to `.tmp`, then `os.replace`.
- **Dead code.** `Encoder_SSME` + `AlphaZeroSSMEModel` are partially-stubbed copies of an old V1; `profiler.py` is unwired and bit-rotted; `aux_phase_head` is declared but never called; `make_encoded_ssme_game_state_model_friendly` has wrong signature.
- **Validation is unimplemented.** `TrainingConfig.val_dir` exists; nothing reads it. `do_pretraining` concatenates `train + val` before training. No val loss in TensorBoard.
- **Coverage gaps in monitoring.** No GPU utilization, no training-data-freshness, no per-iteration wall-clock, no gating win-rate history. `Metrics.add_scalar` ignores the `loop`/`game` arguments it accepts.

---

## (a) Model init / save / load

### [HIGH] `do_pretraining` never persists trained weights
`rl18xx/agent/alphazero/pretraining.py:764-777`. `do_pretraining` loads the latest model with `get_latest_model(model_dir)`, calls `train_model(model, ...)`, and returns the metrics — but never calls `save_model` (or `save_weights`). The fine-tuned weights live only in the returned Python object and are discarded when the CLI returns. Note that `train_model` itself only saves the **optimizer** state (`train.py:433`), not the model state dict. Pretraining is effectively a no-op against disk.

Fix: at the end of `do_pretraining`, call `save_model(model, model_dir)` so the next `train` run picks up the pretrained checkpoint via `get_latest_model("model_checkpoints")`. This also dovetails with the gating bypass at `loop.py:618` (first iteration auto-promotes), so a pretrained seed flows naturally into self-play.

### [HIGH] `autoresearch/run_experiment.py` calls `save_model` with a removed `new=` kwarg
`autoresearch/run_experiment.py:130` does `save_model(model, tmpdir, new=False)`, but the current signature (`checkpointer.py:136`) is `save_model(model, model_checkpoint_dir) -> int`. This will raise `TypeError: save_model() got an unexpected keyword argument 'new'` the moment it runs. Either delete `new=False` or — if `autoresearch/` is unused — drop the dead code. (Quick `grep` shows `run_experiment.py` is the only stale caller; `evaluate.py` already uses `get_latest_model`.)

### [HIGH] `get_model_from_path`'s "old format nested dir" fallback walks one level, but pretraining used to write two
`checkpointer.py:96-133`. The fallback at lines 115-121 handles "old flat format with one nested subdir". But before the new layout, pretraining wrote under `model_checkpoints/<arch>/<session>/<checkpoint>.pth` (two levels), and `get_model_from_path("model_checkpoints")` will (a) pick the alphabetically-greatest `<arch>` directory (`max(subdirs, key=lambda x: x.name)`) and (b) then try to read `<arch>/config.json` — which doesn't exist, because configs live in the *session* directory.

Worse, when both `v1` and `v2` arch dirs coexist, `max(...)` picks `v2` lexicographically, silently hiding `v1`. And `checkpoint_path` is computed against the *outer* `p`, not the nested dir, so it falls through to "no .pth files" and raises a confusing error.

Fix: remove `get_model_from_path` entirely or rewrite it to delegate to `_find_latest_session` / `_find_latest_checkpoint`. The only caller that needs path-specific loading is `rl18xx/agent/arena.py:85` (`test_mcts_versions`); migrate that to accept a session dir and call a small new helper that loads from a specific session path. Everywhere else (`arena.py:62`, `main.py:59`, etc.) already uses `get_latest_model`.

### [MED] `--model-dir` UX forces users to know the architecture/session subpath
`main.py:129,141` document `--model-dir` as just "Model checkpoint directory", and the implementation (`get_latest_model` at `arena.py:62` / `main.py:59`) does walk `<dir>/<arch>/<session>` correctly. Good. But `cmd_pretrain` (`main.py:129`) passes `args.model_dir` directly to `do_pretraining` -> `get_latest_model(model_dir)`. There's no `ensure_seed_model` call in `cmd_pretrain`, so a user running `python main.py pretrain` on a clean checkout gets `FileNotFoundError: No session directories found in model_checkpoints`. Fresh-start friction.

Fix: in `cmd_pretrain`, call `ensure_seed_model(args.model_type)` (add a `--model-type` arg to pretrain, defaulting to `v2`) before `do_pretraining`. Alternatively, have `do_pretraining` itself create a fresh model if none exists.

### [MED] `get_latest_model` should take an explicit `arch` and default to the configured model type
`checkpointer.py:36-54` finds the "latest" session by sorting `<session>` names lexicographically across all `<arch>` dirs. If a user runs v1, then v2, then v1 again, the "latest" returned depends on whichever timestamp string sorts highest globally — not the model the user most recently trained.

Fix: have `get_latest_model(dir, arch=None)` accept an explicit arch and walk only that arch dir, returning a clear error if no sessions exist for it. CLI flags like `--model-type v2` should be honored at load time, not only at create time — today `ensure_seed_model("v1")` is silently ignored when v2 exists on disk. With the arch filter, the user would get a clean "no v1 sessions found, create one with `--fresh`" instead. Within a single arch, lexicographic order on `{timestamp}_{seed}` is unambiguous (timestamps monotonically increase), so no `latest.json` pointer is needed.

### [MED] Bundle config into the `.pth` instead of a sidecar `config.json`
`checkpointer.py:149-151`. Each `save_model` call dumps `config.json` via plain `open("w")` + `json.dump`. Self-play workers (`loop.py:193`) call `get_latest_model("model_checkpoints")` concurrently with the main loop's `save_model` (`loop.py:623,640`); a worker can hit a half-written `config.json` (rare but possible — `json.dump` doesn't write atomically). Also, rewriting an identical config every iteration is wasted I/O.

Fix: switch to the standard PyTorch pattern of `torch.save({"state_dict": ..., "config": config.to_json()}, path)` and read it back via `torch.load`. This eliminates the race entirely (one atomic write per checkpoint), makes the artifact self-describing, and removes the possibility of the sidecar desyncing from the latest `.pth`. Optionally still write `config.json` once at session creation as a human-readable sidecar, but treat the `.pth`-embedded copy as the source of truth.

(Retracted in discussion: an earlier "optimizer state session footgun" finding. The session-keying is correct by design — optimizer state belongs to a specific (session, checkpoint) pair, and the canonical load path preserves timestamp+seed from `config.json` via `from_json`. Worth a comment at the top of `checkpointer.py` documenting that invariant so future entry points don't construct a fresh `ModelConfig()` and try to load weights into it.)

### [LOW] Defensive `getattr(model.config, "seed", None)` masks real bugs
`checkpointer.py:29-31` and `model.py:188-190` both do `getattr(self.config, "seed", None)` with a fallback to `"unknown"`. Both `ModelConfig` and `ModelV2Config` always set `seed` in `__post_init__`, so this fallback is unreachable in normal use — but if it ever fires, sessions named `..._unknown` will collide across runs and cause `save_model` to write into the same directory as a previous run, mixing weights. Fix: drop the defensive `getattr`; let `AttributeError` propagate.

### [LOW] `ensure_seed_model` doesn't tell the user which `model_type` was already on disk
`loop.py:364-399`. If the user passes `--model-type v1` but `model_checkpoints/v2/...` already exists, `has_checkpoints` is `True` so we return silently and self-play loads the v2 model. The user's explicit flag was ignored. Log a warning when the discovered model's arch doesn't match the requested `model_type`. (This also surfaces issue MED-3 above.)

### [LOW] Unused / inconsistent imports in checkpointer
`checkpointer.py:4` imports `datetime` but never uses it. `checkpointer.py:7` imports `torch` only for the optimizer helpers (fine, but it pulls torch into a module that loaders elsewhere may want for a quick path check). Minor cleanup; drop `datetime`.

### [LOW] `_load_model_from_config` heuristic is brittle
`checkpointer.py:14`. It picks V2 if `"d_entity" in config_data or "hex_transformer_layers" in config_data`. If V2's config ever drops/renames either field (or V1 gains a `d_entity` for some reason), loads silently route to the wrong class. Store an explicit `"architecture": "v1"|"v2"` field in `to_json` and dispatch on that; fall back to the heuristic for legacy configs.

---

## (b) Pretraining pipeline

### [HIGH] Pretraining should not share `train_model` with the RL loop

The pretraining objective is fundamentally different from the RL self-play objective:

- **Loss**: Pretraining is a classification problem — the human's chosen action is a hard target. Cross-entropy (with optional label smoothing) is the natural fit. Self-play has soft MCTS visit-count targets where the KL-flavored loss is correct. The current code uses the same KL formulation for both, which works mathematically but obscures the intent.
- **Validation**: Required for SL; the current `do_pretraining` (`pretraining.py:764-777`) merges train+val into one dataset (`HumanPlayDataset(training_data + validation_data)`), never runs `model.eval()`, and reports no top-1 / top-5 accuracy or val loss. `TrainingConfig.val_dir` is wired but unused.
- **LR schedule**: A finite-step cosine decay is standard for SL; the current linear warmup → constant LR is tuned for indefinite RL training.
- **Optimizer state**: `train_model` unconditionally calls `load_optimizer_state` (`train.py:165`), which will pull in optimizer/scheduler state from any prior RL session — silently overriding the SL learning rate and momenta. Pretraining must start from a fresh optimizer.
- **Checkpointing policy**: SL wants "best val loss" + "latest"; RL wants "every iteration".

Fix: add a separate `pretrain_model(model, train_dataset, val_dataset, config)` wrapper. Share low-level helpers (forward pass, gradient step) with `train_model` but own its own loss (`nn.CrossEntropyLoss(label_smoothing=0.03)` on the chosen-action index), cosine LR, validation pass, and checkpoint policy. `do_pretraining` then calls `pretrain_model` instead of `train_model`. Have the wrapper save the best-val-loss checkpoint to `model_dir` (see "HIGH — checkpoint saving belongs to the trainer" below).

### [HIGH] Checkpoint saving belongs to the trainer, not its caller

`train_model` currently saves only `optimizer.pth` (`train.py:433`) and returns the in-memory model; the responsibility for persisting weights is left to whoever called it. `do_pretraining` doesn't do it, hence the "pretraining is a no-op against disk" bug. The same trap exists for any future entry point that calls `train_model` directly.

Fix: have `train_model` (or its pretraining wrapper) always emit at least a "last" checkpoint via `save_model(model, model_dir)` before returning. The pretraining wrapper should additionally track the best-val-loss checkpoint and write that as the canonical one (e.g. save to the session as the next numbered `.pth`, plus track a `best.json` recording which checkpoint number was best). Callers should never need to remember to save.

### [HIGH] Validation split is per-game and uses 5% of ~250 games

`convert_game_to_training_data` (`pretraining.py:682-693`) decides validation membership once per game via `random.random() < 0.05`. With ~250 games that's an expected ~12 validation games (~6000 positions) — variance is large and some runs will produce zero or one val games. Worse, splitting by *game* is correct (prevents within-game leakage) but the per-game decision is made inside the function that already returns two lists, which is confusing and also yields nondeterministic data depending on `random` state from other code.

Fix: lift the split to `convert_games_to_training_dataset` / `do_pretraining`, seed it (`random.Random(seed).shuffle(games)`), expose `validation_percentage` and `seed` as `TrainingConfig` fields (replace the magic `0.05`), and require a minimum number of val games (e.g. `max(10, int(0.1 * len(games)))`).

### [HIGH] ~~`make_encoded_game_state_model_friendly` mis-dispatches for `Encoder_V2`~~ **SUPERSEDED**

*Both `make_action_model_friendly` and `make_encoded_game_state_model_friendly` will be deleted entirely when continuous prices via PW lands (see "Pretraining redesign" below). The dispatch bug becomes moot. Until that change is implemented, the short-term fix is `type(encoder) is Encoder_GNN`.*

Original finding:

`Encoder_V2(Encoder_GNN)` (`encoder.py:1752`) is a subclass of `Encoder_GNN`, so `isinstance(encoder, Encoder_GNN)` (`pretraining.py:585`) matches V2 first and the V2 encoder is routed into `make_encoded_gnn_game_state_model_friendly`. That function then writes into magic offsets `335` (bids) and `359` (min_bid) of `encoded_game_data` (`pretraining.py:616-617`) which are specific to the V1 GNN encoding layout — they will silently corrupt the V2 state vector.

Also, `make_encoded_ssme_game_state_model_friendly` (`pretraining.py:674-679`) is a stub (`pass`), and even if it returned anything, the dispatcher calls it with the wrong signature: line 588 passes `(encoder, encoded_game_state, action, game_state)` but the function only takes `(encoded_game_state, action, game_state)` — a TypeError if SSME is ever used.

Fix: check `type(encoder) is Encoder_GNN` explicitly, add a separate `Encoder_V2` branch (returning `encoded_game_state` unchanged until a real implementation exists), and fix the SSME signature. Better still, push the "model-friendly" transformation onto the encoder class as a method (`encoder.make_action_friendly(state, action, game)`) so the dispatch is polymorphic and the magic offsets live next to the layout that defined them.

### [HIGH] ~~Action-space projection should live in the data-cleaning step, not the encoding step~~ **SUPERSEDED**

*Now that the model gets a continuous price head (Bid / BuyCompany / BuyTrain prices are direct outputs of the network), action-space projection isn't needed at all. The model can be trained directly on raw human action prices via NLL on the continuous price head. Both `make_action_model_friendly` and `make_encoded_game_state_model_friendly` are deleted entirely. See "Pretraining redesign" below.*

Original finding:

Two functions exist solely to project the raw human action onto the model's restricted action space:

- `make_action_model_friendly` (`pretraining.py:465`) rewrites: `Bid` → minimum legal multiple of 5 (only for first bid on a company), `BuyTrain` → nearest legal predefined price, `BuyCompany` → nearest of `{min, max, owner_cash}`.
- `make_encoded_game_state_model_friendly` (`pretraining.py:576`) patches the *encoded state vector* so that historical bids appear as a consecutive multiple-of-5 ladder. This exists because the cleaned game state's bidding history is still in raw human prices, and the model's "minimum bid" interpretation requires the prior bids to look canonical too.

Both are deterministic and model-independent. They belong in the cleaning pass: replay each game with canonicalized actions at write time, so `human_games/*.json` contains a game-state history that is already in model-friendly form. Then:

- `make_action_model_friendly` disappears (canonical actions are already stored on disk).
- `make_encoded_game_state_model_friendly` disappears (the encoded state derived from canonical history is naturally canonical).
- The encoding step becomes pure: `encoder.encode(state)` + `action_mapper.get_index_for_action(action, state)`.
- The brittle magic offsets `335`/`359` (`pretraining.py:616-617`) — which already mis-target the V2 layout — go away.

The one risk to validate: canonicalizing each bid to the rounded-up-to-multiple-of-5 minimum legal bid must remain a legal bid at every step. For 1830's auction this should hold (multiple-of-5 min_bid ≥ engine's min_bid, and turn order in the priority deal isn't affected by bid magnitude), but it should be asserted while regenerating `human_games/`.

### [HIGH] ~~`pi` target is not a valid probability distribution~~ **SUPERSEDED**

*The flat 26,535-dim `pi` target goes away entirely once the autoregressive policy head + structured targets land. Pretraining instead emits a structured target (type index + per-type parameter indices) and the loss is three independent cross-entropies per autoregressive level (see "Pretraining redesign" below). Until then, the math fix is still needed for any v0 training with the flat head.*

Original finding:

`pretraining.py:721-724`:
```python
pi[legal_action_indices] += epsilon / len(legal_action_indices)
pi[action_index] = 1.0 - epsilon
```
The smear adds `epsilon/n_legal` to every legal index *including* `action_index`, and then the chosen index is overwritten to `1-epsilon` (so the chosen-index smear is lost). Total mass is `1 - epsilon + epsilon * (n_legal - 1)/n_legal = 1 - epsilon/n_legal`, i.e. slightly under 1. The cross-entropy in `train.py:222` (`-sum(pi * log_probs)`) is still well-defined, but the target is inconsistent and target_entropy diagnostics will be biased.

Fix:
```python
pi = torch.zeros(action_mapper.action_encoding_size)
pi[legal_action_indices] = epsilon / len(legal_action_indices)
pi[action_index] = (1.0 - epsilon) + epsilon / len(legal_action_indices)
```
Also assert `pi.sum() == 1` and that `action_index in legal_action_indices`. Move `epsilon` to `TrainingConfig.pretrain_label_smoothing` (currently a magic `0.03` at line 721).

### [HIGH] `train_model` writes no TensorBoard / dashboard output

`train.py` collects an extensive `TrainingMetrics` dataclass (`train.py:25-77`) but never writes to a `SummaryWriter`, and `do_pretraining` discards the returned metrics. The dashboard / TensorBoard mentioned in `CLAUDE.md` will show nothing for pretraining runs.

Fix: thread a `SummaryWriter` (path under `runs/pretrain_<timestamp>/`) into `train_model` and log `losses/total`, `losses/policy`, `losses/value`, `policy/top1_acc`, `policy/top5_acc`, `value/explained_variance`, `grads/total`, `lr` each epoch (and validation losses if added). Persist `metrics.to_json()` next to the checkpoint.

### [MED] `filter_actions` undo/redo loses history on redo, mishandles redo-after-action

`pretraining.py:82-118`:
- On `undo`, the *pre-undo* snapshot is pushed onto `filtered_actions_history`. On `redo`, the snapshot is restored and popped. But if the user undoes, takes a new action, then redoes, the redo will restore a snapshot that *predates* the new action — silently dropping the new action. The Ruby reference implementation stacks redos differently (`redo` only works when the next action is exactly the popped one).
- `undo` to an `action_id` uses `x["id"] <= action_id`, but `action_id` semantics in the source data are post-filter; for raw JSON it may not match. There's no check that an undo with `action_id` actually shrinks the list.
- `auto_actions` are mutated in place (`del action["auto_actions"]`), which then alters the caller's dict (and re-running `filter_actions` on the same data yields a different result).

Fix: deepcopy actions at the top of `filter_actions`; validate `action_id` exists in current `filtered_actions` before truncation; for redo, require that the snapshot's "next" action matches and otherwise clear the redo stack on any non-undo/redo action (matching standard editor semantics).

### [MED] `should_skip_action` "condition 1" disabled, leaves brittle 3-action lookahead

`pretraining.py:217-238`: the commented-out condition 1 says "ignore for now", and condition 2 reads three actions ahead, indexing `filtered_actions[i+2]` only after checking `i + 2 < len(filtered_actions)`. If a real game ends with this exact pass pattern as its last 3 actions, the check returns False correctly. But the BuyTrain pass skip should really be driven by the legal-move set ("the upcoming move is illegal because we're past BuyTrain"), not by pattern-matching the next two raw actions. This is the kind of heuristic that silently miscompiles a small fraction of games.

Fix: drive skip detection from `ActionHelper.get_all_choices(game_state)` — if the pass is not legal at this step, skip it; if it is legal but the next non-pass action by the same entity is also legal at this state, skip the pass. Same approach for the BuyCompany/TrackStep skips.

### [MED] `HumanPlayDataset` is fully in-memory; eager accumulation in `do_pretraining`

`HumanPlayDataset.__init__` (`dataset.py:53-56`) stores `examples: List[Any]` directly — no LMDB, no disk backing. `do_pretraining` (`pretraining.py:770-775`) extends two lists before constructing the dataset; each example is a tuple of tensors so for 250 games × ~500 positions, expect roughly 1-3 GB of resident tensors (node_features + edge_index alone are several KB each).

A disk path already exists: `convert_games_to_training_dataset` writes via `TrainingExampleProcessor.write_samples` to LMDB (`pretraining.py:731-761`), and `SelfPlayDataset` (`dataset.py:23`) is the LMDB-backed reader. But `do_pretraining` re-encodes from JSON every call and bypasses LMDB entirely.

Fix: have `do_pretraining` call `convert_games_to_training_dataset(...)` to produce `<save_path>/training` and `<save_path>/validation` LMDB envs, then construct two `SelfPlayDataset`s and pass them to `train_model`. Delete the in-memory `HumanPlayDataset` once that's done (or keep it only for tiny test fixtures).

### [MED] `convert_games_to_training_dataset` will crash on first call

`pretraining.py:742-743` opens `save_path / "progress.json"` for reading inside the loop, but the file is never created — `save_path.mkdir(parents=True, exist_ok=True)` only makes the directory. The very first game iteration will raise `FileNotFoundError`. Also, `TrainingExampleProcessor.write_samples` is invoked with already-encoded examples (a list of `(encoded_state, legal_actions, pi, value)` tuples) rather than the `(game_state, ...)` tuples its sibling `write_lmdb`/`make_dataset_from_selfplay` expect — different shape, no canonicalization, so SelfPlay and HumanPlay data will be inconsistent.

Fix: initialize `progress.json` to `{}` if missing; either make pretraining encode/canonicalize the same way as `make_dataset_from_selfplay`, or refactor `write_samples` to be format-agnostic and add a `make_dataset_from_human_play` path.

### [LOW] Value target is correct but value head expects logits

`value_loss` in `train.py:228-238`: for `pi`-target pretraining, `actual_value` (`pretraining.py:700-706`) is `1` for the winner, `0` for ties, `-1` for losers. The branch logic at line 229 detects `is_score_values = (value >= 0).all()` — this is True only when the row contains no losers. With `actual_value` being `[1, -1, -1, -1]` for a 4-player game (typical case), it goes into the winners-mask branch, which converts to a one-hot probability target and then KL-divs the value head's softmax. That is consistent, but it means the value head is learning a 4-way classifier of "who wins", not a per-player ±1 regression — fine for AlphaZero, just worth documenting.

Fix: add a docstring on `value_loss` describing the two branches, and assert in pretraining that ties produce `value = [0, 0, ...]` (your code at line 704 sets winners to 0 on ties — but losers remain `-1`, which is wrong: a 4-way tie is `[0, 0, 0, 0]` but a 2-way tie is `[0, 0, -1, -1]`). The value head will train to predict `[0.5, 0.5, 0, 0]` for that 2-way-tie case, which is actually reasonable, but worth making explicit.

### [LOW] Code duplication and magic numbers

`convert_game_to_training_data` and `convert_games_to_training_dataset` both iterate games, both call `convert_game_to_training_data`, and both build datasets — the only real difference is in-memory vs. LMDB sink. Refactor into one inner generator (`def iter_training_examples(games, encoder)`) that yields per-position tuples, and have two thin consumers. Magic numbers: `epsilon=0.03` (`pretraining.py:721`), `validation_percentage=0.05` (line 686), bid/min_bid offsets `335`/`359` (lines 616-617), `min_diff_amount=2000` (lines 519, 562). Move them to constants or config fields.

### [LOW] `check_action_in_all_actions` returns inconsistently and over-mutates

`pretraining.py:121-176` mutates the input dicts (`action_args["variant"] = None` etc.) and returns either `None` or `a.to_dict()`. The "found" case returns `None` (signaling no update needed); the "not found, but matched by tile" case returns the helper's action. This conflates two meanings of `None` (matched, or not matched and not recoverable). A boolean + optional dict tuple would be clearer.

### [LOW] `filter_to_completed_4_player_games` drops games with any `end_game` action

`pretraining.py:69-75` skips a game if *any* action has type `end_game`. That's correct for games that aborted via end-game, but a normal completed game does not have this action — so the check is harmless in practice. Add a comment explaining the intent so future readers don't assume normal games trigger it.

### [LOW] No deduplication between game IDs in `load_games_from_json`

`pretraining.py:46-57` enumerates files by glob but does not check for duplicate IDs (`game["id"] = game_file.stem`). If `fix_online_games` writes an output file with the same stem twice (e.g. via a re-run with `overwrite=True`), the dataset accumulates duplicates. Add a `seen = set()` guard.

---

## (c) Self-play

### [HIGH] Argument order mismatch when calling `main()` from CLI
`loop.py:796-807` invokes `main()` positionally: `main(num_loop_iterations, num_games_per_iteration, num_threads, cleanup, args.num_readouts, ...)`. But `main()` (loop.py:418-431) has the signature `main(num_loop_iterations, num_threads, cleanup, num_readouts, ...)` — there is no `num_games_per_iteration` parameter. The CLI's `num_games_per_iteration` value silently binds to `num_threads`, `num_threads` binds to `cleanup`, the boolean `cleanup` binds to `num_readouts`, and `args.num_readouts` binds to `num_epochs`. So running `python loop.py --num_threads 8 --num_readouts 200` actually runs with whatever value `--num_games_per_iteration` was given as the thread count, and `num_readouts` becomes `True/False`. The `target_experiences` argparse flag is also missing entirely. Fix: name all args at the call site (`main(num_loop_iterations=..., num_threads=..., cleanup=..., num_readouts=..., target_experiences=args.target_experiences, ...)`) and add the missing argparse arguments for `target_experiences`, `model_type`, `fresh`.

### [HIGH] Gating evaluation is effectively deterministic — `gate_games` does not give independent samples
`loop.py:251-361` runs `gate_games` games but sets `dirichlet_noise_weight=0`, `softpick_move_cutoff=0` (always argmax), and a single fixed initial game state from `_create_fresh_game()`. Combined with deterministic network priors and argmax leaf selection, every even-indexed game produces an identical trajectory, every odd-indexed game produces an identical trajectory. `num_games=10` is at best 2 statistically-distinct outcomes (one per seat permutation). A 1-0 win for the candidate becomes 5-0 in the win-rate; promotion noise is enormous. Fix options: (a) inject small Dirichlet noise during gating (e.g., `dirichlet_noise_weight=0.1`) and/or use a positive `softpick_move_cutoff`, (b) seat-rotate across more permutations (4 perms × N each), (c) vary `random`/`numpy.random` seeds and rely on any stochastic tiebreaks. Also caps gating readouts at `min(num_readouts, 50)` (line 633) which is fine but should be documented.

### [HIGH] Duplicate-leaf path drops backups in `tree_search`
`mcts.py:365-378` returns early from `incorporate_results` when `self.is_expanded` is True, so the call to `self.backup_value(...)` at the end (line 400) is skipped. In `self_play.py:201-222`, the same leaf can appear twice in `leaves[]` if virtual loss didn't deflect a second descent (e.g., when only one strong child exists). The first incorporate_results expands+backs up; the second reverts the second virtual loss but provides no backup. Net effect: that visit's network value is lost. Fix: in `incorporate_results`, if already expanded, still call `backup_value(value, up_to=up_to)` (or equivalently, dedupe `leaves` and adjust virtual-loss counts before backing up). Minor in practice but a real correctness bug.

### [MED] `_play_gate_game` initializes all 4 agents with the same `BaseGame` object
`loop.py:290-292`: every agent's root MCTSNode is constructed with the same `game_state` reference. Each agent's `play_move` triggers its own `pickle_clone`, so subsequent state diverges harmlessly, but as long as nothing in MCTS mutates the root game (currently true), this works. The risk is silent if anything in the engine mutates state from a read path — clone-on-demand is implicit. Fix: pass a fresh `pickle_clone()` to each agent's `initialize_game` to be explicit and decouple them.

### [MED] `np.random` is not seeded per worker; only `random.seed(os.getpid())` is set
`loop.py:181` reseeds Python's `random` but not `numpy.random`. Dirichlet noise (`mcts.py:470`) and any future numpy random use will share whatever global state spawn-inherited workers happen to get. With `spawn` start method, each worker typically gets fresh OS entropy for numpy, so this is empirically OK — but it's an implicit dependency. Fix: explicitly `np.random.seed(os.getpid() ^ int(time.time()))` (or a more principled seed) in `run_self_play` to make per-worker variability explicit.

### [MED] `cleanup_files()` crashes when state directories don't exist
`loop.py:160-176`: iterates `TENSORBOARD_LOG_DIR_BASE.iterdir()` and `SELF_PLAY_GAMES_STATUS_PATH.iterdir()` without first `mkdir(parents=True, exist_ok=True)` or `if exists()`. `SELF_PLAY_GAMES_STATUS_PATH` is created at import of `self_play.py` (line 28), so that's lucky. `TENSORBOARD_LOG_DIR_BASE = Path("runs/alphazero_runs")` is not pre-created; first cleanup run on a clean checkout will raise `FileNotFoundError`. Fix: guard each `iterdir` with `if path.exists()` or `mkdir` first.

### [MED] `inject_noise` runs on every move even when there is only one legal action
`self_play.py:463` calls `player.root.inject_noise()` unconditionally, before the `num_legal_actions == 1` branch. For forced moves this is wasted work (still calls `np.random.dirichlet`, modifies a 1-element prior pointlessly). Fix: move `inject_noise` inside the `else` branch where MCTS will actually consume it.

### [MED] Tree reuse across moves is partial — the new root is kept but ancestor stats are discarded
`prune_mcts_tree_retain_parent` (`self_play.py:327-360`) keeps the new root and its subtree but detaches `parent_of_new_root` from its parent (sets it to a `DummyNode`). That means the new root's `parent` is `parent_of_new_root` (the old root, which now has DummyNode as its parent). Subsequent reads of `self.root.N` go through `self.parent.child_N_compressed[self._parent_index]` which still points into the OLD root's compressed arrays. This is correct for visit counts (they're preserved), but the chain `new_root.parent.parent = DummyNode` means walking ancestors stops early. `add_virtual_loss(up_to=self.root)` and `backup_value(up_to=...)` won't try to walk above `parent_of_new_root` anyway. So this works — but it's load-bearing and easy to break. A cleaner design is to have the new root own its own `child_N`/`child_W` arrays and reset `parent` to a real DummyNode whose defaultdict holds the carried-over stats. Worth a comment at minimum.

### [MED] `MCTSPlayer.is_done()` compares numpy array to Python list
`self_play.py:262`: `return self.result != [0.0, 0.0, 0.0, 0.0] or self.root.is_done()`. `self.result` is `np.zeros(len(players))` — a numpy array. `numpy_array != list` returns an element-wise boolean array, and `__bool__` on it would raise `ValueError: The truth value of an array with more than one element is ambiguous`. The fact this hasn't crashed suggests this code path is never executed (or Python implicit conversion takes a fast path). Either way, fix to `not np.array_equal(self.result, np.zeros_like(self.result)) or self.root.is_done()`.

### [MED] `is_done` end-of-game branch uses `==` instead of `>=` against max_game_length
`self_play.py:526`: `if player.root.game_object.move_number == self.config.max_game_length:`. The `MCTSNode.is_done` (mcts.py:416) uses `>=`. If `move_number` ever overshoots `max_game_length` (e.g., from a multi-action `auto_actions` step in the engine), the `end_game()` branch is skipped, the game is recorded as "Completed" but `game_ended_by_max_length` metric stays 0 and the engine may not produce a valid `result()`. Use `>=` for parity.

### [MED] `inject_noise` modifies priors in-place without resetting from `original_prior_compressed`
`mcts.py:464-474`: noise is multiplied into `child_prior_compressed`. The code maintains a separate `original_prior_compressed` saved by `incorporate_results` (line 392), but `inject_noise` never resets from it. This is fine in current flow (each new root has clean priors from a single prior `incorporate_results`), but any future change that calls `inject_noise` twice on the same node will silently compound noise. Fix: `self.child_prior_compressed = self.original_prior_compressed * (1 - w) + dirichlet * w` to make this idempotent.

### [MED] `selfplay_dir is not None` branches are dead code
`self_play.py:592, 608`: after `SelfPlayConfig.__post_init__` always sets `self.selfplay_dir = Path("training_examples") / self.selfplay_dir`, the value is never None. The `if self.config.selfplay_dir is not None` guards are dead. Either honor an optional disable path (allow `None` to skip persistence) or drop the guard. As written, training data is always written, which is correct for the loop but the dead guard misleads readers.

### [LOW] Repeated `time.time()` instrumentation duplicates `profiler.py` concerns and is mostly unused
`mcts.py` and `self_play.py` have dozens of inline `time.time()` / `time.perf_counter()` blocks pushed into `add_metric`. `profiler.py` exists for benchmarking but isn't used here. In production runs `Metrics` is disabled (`metrics=None` at loop.py:198), so the bulk of these calls produce no signal but still pay the syscall cost. Either gate timing entirely on `self.config.metrics is not None`, or extract a context-manager helper to keep call sites tidy. The `select_leaf` timer (mcts.py:279, 300) for example runs even when metrics are off.

### [LOW] `played_actions` seed assumes `raw_actions` is a list of dicts
`self_play.py:101`: `list(game_state.raw_actions)`. For Python engine, `raw_actions` is a list of dicts. For RustGameAdapter, `raw_actions` is also a list of dicts (rust_adapter.py:1536). OK today, but `extract_data` later passes these dicts to `process_action`, which only works because both engines accept dicts. Worth a single comment.

### [LOW] `gc.collect()` in `run_game` and `main` loop is sprinkled without measurement
`self_play.py:617, 652` and `loop.py:726` explicitly call `gc.collect()`. With the elaborate `_recursive_clear_references` already breaking reference cycles, these full collections may be costly and unnecessary. Profile before keeping.

### [LOW] `pretrain_dir` / arena worker has no kill propagation on `ProcessPoolExecutor` shutdown
`loop.py:586`: `executor.shutdown(wait=True)` blocks on outstanding futures but the signal handler at line 217 calls `cleanup_and_exit` which terminates child processes directly via psutil. There's a race: SIGINT during `f.result()` will call `cleanup_and_exit`, but the `finally: executor.shutdown(wait=True)` then waits indefinitely for the now-killed workers. Use `executor.shutdown(wait=False, cancel_futures=True)` in the finally, or have the signal handler set a flag instead of killing.

### [LOW] LMDB lifecycle has no per-game flush or fsync
`dataset.py:97-115`: `TrainingExampleProcessor.write_samples` opens the LMDB env, writes inside a transaction, but the `env` is never closed before the function returns. LMDB writers across workers (each worker calls `write_lmdb` independently into the same LMDB path) rely on LMDB's internal locking — that's safe per LMDB's design, but `lmdb.open` is called for each game inside each worker, repeatedly opening the same env. Cache the env in the processor, and call `env.close()` (or `env.sync()` explicitly) before returning to ensure durability against worker crashes.

### [LOW] No mid-game recovery / crash resumption
On a crash mid-iteration (`loop.py:728-732`), `LOOP_STATUS_PATH` gets an error message but partial self-play games' LMDB writes are NOT rolled back. Training next iteration includes whatever made it to disk before the crash — could include unfinished or corrupt-result games if a worker died mid-write. `run_game` already returns early if `player.result is None or all zero` (lines 599-604), good. But if `play()` crashes after some `played_actions` but before `set_result`, the result is unset and the function returns without writing data — correct. Crash-during-LMDB-write would corrupt the env. Document the recovery story or add an LMDB consistency check at startup.

### [LOW] Hardcoded magic numbers
`mcts.py:16` `POLICY_SIZE = 26535` (also `ActionMapper.action_encoding_size` — duplicated, hardcoded). `mcts.py:17` `VALUE_SIZE = 4`. `mcts.py:468` `alpha_value = 10.0 / num_legal_actions` (the "10" is a hyperparameter not exposed in config). `loop.py:298` `if game_state.move_number >= 1000:` — hardcoded gating-eval move cap, should mirror `max_game_length`. `loop.py:415 ESTIMATED_MOVES_PER_GAME = 1000`. Hoist into `SelfPlayConfig` / a constants module.

### [LOW] `inject_noise` alpha derivation diverges from `dirichlet_noise_alpha` config field
`SelfPlayConfig.dirichlet_noise_alpha=0.03` (config.py:200) is declared "kept for backward compatibility" but `inject_noise` actually uses `10.0 / num_legal_actions`. Two sources of truth confuse future readers. Either remove the unused field or have `inject_noise` use a `max(config.dirichlet_noise_alpha, 10/n)` style fallback.

### [LOW] `loop.py` blocking-wait loop polls with `f.done()` then falls through to `concurrent.futures.wait`
`loop.py:528-538`: first scans `f.done()` synchronously (cheap but a busy poll), then if none done, blocks via `wait(... FIRST_COMPLETED)`. The first loop is redundant — `wait` handles the "any-done" case fine. Removing the manual polling simplifies the code.

### Known issues (already tracked)
- Temperature in `pick_move`: **addressed** — `pick_move` now uses temperature-1 sampling on compressed visit counts and argmax post-cutoff (self_play.py:139-158). Roadmap can be marked done for this item.
- `sim_count_this_move` UnboundLocalError: **fixed** — initialized to 0 at self_play.py:465.
- `tree_search_duration_this_move` timing bug: **fixed** — recorded around the MCTS loop at self_play.py:485-489.
- Lazy encoding for MCTS nodes: **implemented** — `ensure_encoded()` defers encoding (mcts.py:154-165), called only before NN evaluation.
- ActionMapper caching: **implemented** — `_cached_action_mapper` module singleton at mcts.py:27-31.

---

## (d) Model design

### [HIGH] v2 structural matrices never initialized — Hex Transformer bias is degenerate
`AlphaZeroV2Model._compute_structural_matrices` (`model_v2.py:685-735`) is defined but **never invoked anywhere in the file or the rest of the codebase** (verified via grep). The buffers `distance_matrix` and `direction_matrix` are registered with default values of all-zeros and all-6 (`model_v2.py:682-683`). Since `direction_matrix=6` is the sentinel for "not adjacent" used in `TrackConnectivityComputer` (line 617: `adj_mask = (direction_matrix != 6)`) and in `StructuralAttentionBias.direction_bias`, the entire Hex Transformer attention bias is uniform across the graph: every distance bucket is 0, every direction is "non-adjacent" (slot 6), and track connectivity is identically zero. The Transformer degenerates to position-encoded global attention with no inductive bias. **Fix**: invoke `self._compute_structural_matrices(game)` lazily from the first `encode` (similar to `Encoder_GNN.initialize`), or precompute it at model construction by instantiating a throw-away game.

### [HIGH] v2 `_extract_active_player` is wrong for operating rounds
`AlphaZeroV2Model._extract_active_player` (`model_v2.py:820-822`) does `game_state_data[:, :NUM_PLAYERS].argmax(dim=1)`. The `active_entity` slot is one-hot over 12 indices (4 players + 8 corps); when a corporation is active during an OR, all four player slots are zero and `argmax` silently returns 0. The correct active-player signal during a corp's turn is the president (`active_president` one-hot at offset 12). Today this is dormant because `run_many_encoded` always passes the encoder-supplied `gs[5]` (line 851), but a direct `forward(...)` call without explicit indices (e.g. when re-loading from training data without round/player annotations) hits this code path. **Fix**: argmax over `[12:16]` (president) for OR phases (detect via `round_type` slot), else argmax over `[0:4]`.

### [HIGH] Canonicalization inverts the active-player signal at training but not inference
`Encoder_GNN.canonicalize_perspective` (`encoder.py:311-335`) is applied in `dataset.py:85-93` so that the active player is rotated to slot 0 and `active_player_idx` is stored as 0. At inference (`model.py:369`, `model_v2.py:851`) the encoder's raw `gs[5]` (true active player index, 0–3) is used uncanonicalized. The value head receives `F.one_hot(active_player_idx, num_classes=4)` (`model.py:471`, `model_v2.py:933`). Training therefore always sees `one_hot(0)` for canonical states, while inference sees `one_hot(0..3)` for raw states — and per-player value slots are also rotated only during training. **The model is trained on a different distribution than it sees at inference, and on the player indicator never observes anything other than position 0.** This is silent and severe. **Fix**: canonicalize at encode-time (apply rotation in `encode`, set `gs[5]=0` always), or remove canonicalization. The one-hot indicator becomes dead input either way.

### [HIGH] v1 value head trained as classifier, used as regressor in MCTS
`train.py:230-238` treats `value_pred` as logits of a softmax distribution over players (KL-divergence loss to the player-share target). At MCTS time (`mcts.py:219`: `child_Q_compressed[:, self.active_player_index]`) and in `run_many_encoded` (`model.py:390`) the **raw logits** are stored as Q, accumulated by `backup_value`, and used directly in PUCT scoring. There is no softmax/normalization on the MCTS path. A logit pair like `(5, -5)` represents probability `(0.999, 0.001)` to training but is treated as `(5, -5)` to MCTS — completely inconsistent units. Backed-up Q values grow with depth (no bounded range), and the c_puct constants in `SelfPlayConfig` (≈1.0–1.5) are calibrated to AlphaZero-style values in `[-1, 1]`. **Fix**: apply softmax inside `run_many_encoded` before returning so Q is in `[0,1]`, OR change the training loss to MSE/cross-entropy on a regression target.

### [HIGH] v2 HexMapTransformer uses only sample-0 track connectivity for the whole batch
`HexMapTransformer.forward` (`model_v2.py:292-294`):
```python
track_conn = track_connectivity[0] if track_connectivity.dim() == 3 else track_connectivity
bias = self.structural_bias(distance_matrix, direction_matrix, track_conn)
```
Track connectivity is computed per-sample by `TrackConnectivityComputer` (returns `(B, N, N)`), but the Transformer drops all but batch index 0 with the comment "all leaves from same game tree during self-play". This is false: training batches mix games and positions, and even within a single MCTS batch, parallel readouts evaluate different leaves with different track states. The track bias is identical across the batch, equal to sample 0's connectivity. **Fix**: keep the batch dim — re-derive `StructuralAttentionBias.forward` to accept `(B, N, N)` track input and broadcast appropriately, returning `(B, heads, N, N)`; in `HexTransformerLayer.forward`, drop the `unsqueeze(0)`.

### [HIGH] v1 FiLM trunk: random init drives signal toward zero, no skip around modulation
In `AlphaZeroGNNModel` (`model.py:286-289`) the FiLM layers are vanilla `nn.Linear(film_embed_dim, trunk_dim*2)` initialized by the generic Kaiming pass in `initialize_weights`. Gamma is initialized as Kaiming-normal centered at 0, beta is 0. Each block does `current_features = gamma * res_block(...) + beta` outside the residual (`model.py:460-464`), so the residual identity path is destroyed by every block. With 7 blocks and `gamma ~ N(0, fan-based)`, the trunk output is approximately zero after a few layers. v2's `FiLMResBlock` does init gamma=1, beta=0 and keeps FiLM inside the residual; v1 should do the same. **Fix**: add identity init for v1's `film_layers` (set bias `[:trunk_dim] = 1`, weight = 0) and move `gamma * x + beta` inside each ResBlock before the second linear, preserving the skip.

### [MED] Player one-hot indicator is dead input after canonicalization
The value head (`model.py:471-473`, `model_v2.py:933-935`) concatenates `F.one_hot(active_player_idx, 4)` to the trunk. Combined with the canonicalization issue above, training-time inputs are always `one_hot(0)`, so the value head learns nothing from this signal and inference-time variation is out-of-distribution. Once canonicalization is fixed end-to-end, this indicator becomes structurally redundant (the canonical state vector already places the active player at slot 0). **Fix**: remove the one-hot input entirely once canonicalization is unified, or use it only if you intentionally break canonicalization.

### [MED] `nn.Bilinear` output dimension is double-counted in v1 policy head
`FactoredPolicyHead.bilinear = nn.Bilinear(num_hexes=93, num_tiles=46, out=93*46=4278)` (`model.py:98`). Parameter count: `4278 * 93 * 46 = 18,302,964` weights plus biases ≈ 18.3M. The user is aware. What's also worth noting: the bilinear inputs are themselves linear projections from `(B, num_hexes)` (raw scores) and `(B, num_tiles)` (raw logits), so the bilinear is effectively a third-order tensor learned from low-rank inputs — most of those 18M params see vanishing gradients. Even before moving to v2, simply replacing the bilinear with an outer-product (`hex_logits[:,:,None] * tile_logits[:,None,:]`) plus a learned `(H, T)` bias matrix would cut params by >99% with marginal capacity loss. **Fix**: as in v2 (`V2PolicyHead`).

### [MED] BatchNorm in v1 trunk is RL-hostile and depends on batch_size > 1
Every layer in v1 uses `BatchNorm1d`: MLP (`model.py:231,234`), GNN BN layers (`model.py:268`), fusion (`model.py:280`), and inside `ResBlock` (`model.py:25,28`). Self-play feeds **single-state batches** through MCTS (`run` → `run_many` of length 1). At `model.eval()` this uses running stats, but those running stats are trained on highly correlated mini-batches of MCTS-explored positions (lots of duplicate or near-duplicate states from the same game). Running mean/var drift wildly between training iterations, producing distribution shift even for the same logical state. v2 already moved to `LayerNorm`; the same swap in v1 (or in any retained v1 paths) is recommended. **Fix**: replace all `BatchNorm1d` with `LayerNorm`.

### [MED] ~~`Encoder_SSME` is a copy-paste of an old `Encoder_GNN` and is broken~~ **RESOLVED**
`encoder.py:959-1750` defines `Encoder_SSME`. It (1) does not include the four newer sections present in `Encoder_GNN`: `or_structure`, `train_limit`, `private_closed`, `player_turn_order` (cf. `Encoder_GNN.GAME_STATE_ENCODING_STRUCTURE` at `encoder.py:127-131`) — so its encoding size and feature semantics drift from v1; (2) `_calculate_map_node_features` and `_precompute_adjacency` are empty `pass`; (3) `get_sequence_features` references attributes (`self.total_features_dim`, `self.hex_sequence_order`, `self.hex_positions`, `self.base_features_dim`, `self.idx_to_hex_coord`, `self.MAX_HEX_REVENUE`, etc.) that are never assigned in `__init__`, so any call would `AttributeError`. Likewise `AlphaZeroSSMEModel` (`model.py:481-550`) has `init_model = pass` and `run_many_encoded` references undefined `batched_game_state_tensor`. Pure dead/broken code that complicates maintenance. **Fix**: delete `Encoder_SSME` and `AlphaZeroSSMEModel` (and the dispatcher branch in `Encoder_1830.get_encoder_for_model`).

### [MED] v2 `aux_phase_head` is dead, `hex_head` fallback is unreachable
- `AlphaZeroV2Model.aux_phase_head` (`model_v2.py:794`) and `predict_phase` (`model_v2.py:942-944`) are declared but never invoked — `forward` returns only `aux_action_count_pred`, and grep confirms no caller. The accompanying `phase_aux_loss_weight=0.01` in `ModelV2Config` is also unused.
- `FactoredPolicyHead.hex_head` (`model.py:78`) / `V2PolicyHead.hex_head` (`model_v2.py:526`): only used when `node_repr is None`; every call site (`model.py:468`, `model_v2.py:930`) always passes node embeddings. Either remove the fallback or assert it's wired so a training/inference divergence doesn't accidentally reach it.

### [MED] `aux_action_count` regression target uses log-scale but predictions are clamped, biasing the loss
`train.py:241-244`:
```python
legal_action_count = legal_action_mask.sum(dim=1)
aux_target = torch.log(legal_action_count.float().clamp(min=1))
aux_pred_clamped = aux_action_count_pred.squeeze(1).clamp(-10, 10)
aux_loss = F.mse_loss(aux_pred_clamped, aux_target)
```
Legal-action counts in 1830 range up to a few hundred, so `aux_target` spans roughly `[0, 6]`. Clamping predictions to `[-10, 10]` is safe but the `aux_action_count_head` is a single `Linear(trunk, 1)` with no nonlinearity and no scale prior. At init its outputs are zero-mean and small; gradient flow from this auxiliary head into the trunk is fine, but the head should output `softplus` or be initialized with a positive bias to avoid early dead clamping behavior. Minor.

### [MED] Inconsistent rotation feature scale vs other node features
In `Encoder_GNN.get_node_features` (`encoder.py:893`):
```python
node_features[hex_idx, feature_offset] = tile.rotation
```
Stored raw in `[0, 5]`, while all other features are in `[0, 1]`. In v1 this fed through a `Linear` so it's recoverable, but in v2 the `port_feature_mask @ node_features` matrix multiply (`model_v2.py:613`) reads only the connectivity columns — rotation is not included — so this doesn't cascade. Still, normalizing `tile.rotation / 5.0` (or one-hot encoding it) would make magnitudes consistent and is essentially free.

### [LOW] `Encoder_1830` is a non-Singleton dispatcher, but its `Encoder_*` subclasses are Singletons
`Encoder_GNN`, `Encoder_SSME`, `Encoder_V2` all use `metaclass=Singleton`. This means `Encoder_V2` and `Encoder_GNN` share a single instance per class. `Encoder_V2(Encoder_GNN)` (`encoder.py:1752`) inherits via metaclass, so creating an `Encoder_V2` after an `Encoder_GNN` returns separate singletons, but each subclass instance carries its own `initialized` flag and player map. This is fine in practice, but the `player_id_to_idx` mapping is mutated on first call to `initialize(game)` and never rebuilt — if the same encoder instance is reused across games with different player rosters (different player names), the mapping silently stays stale. **Fix**: invalidate the mapping when player IDs change, or rebuild every call.

### [LOW] Encoder rotation feature missing for v1 GNN with `gnn_layers=4` — receptive field hard-capped
Known to the user; flagging for completeness that `gnn_layers: int = 4` (`config.py:33`) gives a 4-hop receptive field but the 1830 map diameter is much larger (~12 hops max distance, encoded by `max_hex_distance: int = 12` in v2 config). v1's policy head can compensate via the global pooled `map_embed`, but per-hex `node_repr_proj` used in the attention `hex_scorer` only sees its 4-hop neighborhood, which is the very signal the policy head relies on to score hex moves. This is what motivates v2 — but if v1 is to be trained further, bumping `gnn_layers` to 6–8 or adding a virtual global node would help.

### [LOW] `np.unique(...).T` on the edges list silently relies on lexicographic stability
`encoder.py:189`:
```python
edge_np = np.unique(np.array(edges), axis=0).T
self.base_edge_index = edge_index[0:2, :].long()
self.base_edge_attributes = edge_index[2, :].long()
```
This relies on each (src, dst) pair having a unique direction across the entire graph. If the 1830 map ever introduces an asymmetric adjacency (or `all_neighbors` returns the same neighbor under two different direction codes due to tile geometry), `np.unique` would keep both rows and the resulting edge_attr would associate one src→dst with two different directions — sortable but semantically wrong. Defensive fix: deduplicate on `(src, dst)` only and assert single direction.

### [LOW] Hardcoded `num_classes=self.config.value_size` for player one-hot couples value head to player count
`model.py:471` and `model_v2.py:933` use `value_size` (number of players) both as the value head output dim AND as the number of classes for the active-player one-hot. These happen to be the same in 4-player 1830 but the code reads them as independent quantities; a 3-player or 6-player variant would silently break (player index out of range). Make this explicit with a `num_players` config field or assert equality.

### [LOW] Hex coordinate axial conversion in v2 is a custom regex-free parse vulnerable to multi-letter row IDs
`hex_coord_to_axial` (`model_v2.py:34-47`) splits letters/digits by char and uses `ord(letter) - ord('A')`. 1830 has rows A–K (single letter), but if any title adds rows like `AA1` (some 18xx games do), this silently produces `ord('AA'[0].upper()) - ord('A') = 0`, swallowing the second character. Defensive only — 1830 is fine.

### [LOW] `ActionMapper.get_index_for_action` round-trip is one-way
The action mapper is constructed as forward (action → index) and is also used in reverse (`map_index_to_action`, `action_mapper.py:671`), but the two paths take divergent routes: forward uses arithmetic offsets over carefully ordered fields, reverse looks up `self.actions[index]` and reconstructs from the stored args list. There's no test that round-trips `decode(encode(a)) == a` for the full legal-action enumeration. Given the subtlety (especially BuyTrain price coding "all" vs "all-but-one" at `action_mapper.py:506-509` and the company-action branch on `index >= self.action_offsets["CompanyBuyShares"]`), a round-trip test over a generated game would surface latent bugs cheaply. No bug found here, but it's untested.

### [LOW] Legal action index list returned `sorted` for performance — but `get_legal_action_indices` re-runs `ActionHelper` every call
`action_mapper.py:669` returns `sorted(indices)`. `mcts.py:387` then does `move_probabilities[self.legal_action_indices]`. This is fine, but every MCTS expansion calls `ActionHelper(state).get_all_choices_limited(state)` (`action_mapper.py:655-656`), which traverses the game graph; this is the dominant non-NN cost in MCTS for many positions. Not a model-design bug, but worth flagging since the legal_action_mask is also rematerialized in `convert_indices_to_mask` (line 644-646) instead of reused.

---

## (e) Job monitoring

### [HIGH] `LOOP_STATUS_PATH` / `LOOP_CONFIG_FILE_PATH` are resolved relative to CWD, not the dashboard package
`rl18xx/agent/dashboard/dashboard.py:19-21` uses `Path("../../../loop_status.json")` etc. — these are *relative* paths interpreted from the process CWD. `startup.sh:23` runs gunicorn with `--chdir ./rl18xx/agent/dashboard`, so it works today, but only by coincidence; any one of (a) running `python dashboard.py` directly, (b) starting from a different CWD, (c) Flask reloader respawn from a different CWD breaks every status read silently — the dashboard just shows "Status file not yet created." Also breaks the cleanup logic in `loop.py:160-176`, which writes the same files via `Path("loop_status.json")` (CWD = repo root). Fix: anchor at the project root, e.g. `REPO_ROOT = Path(__file__).resolve().parents[3]; LOOP_STATUS_PATH = REPO_ROOT / "loop_status.json"`.

### [HIGH] Status / games-status reads are not atomic — partial-write JSONDecodeError races
`loop.py:129-135` and `self_play.py:406-410` both do `open(path, "w") + json.dump(...)`. A concurrent dashboard read (running every 1–2s) hits a half-written file and `json.load` raises. `dashboard.py:117-120` catches it and returns a 500 — the UI then shows "Could not load self-play games" mid-training. The dashboard polls `/api/games_status` every 1s for *all* `self_play_games_status/*.json` (line 91), and each worker rewrites its file every move, so collisions are continuous at scale. Fix: write to `path.tmp` then `os.replace(tmp, path)` for atomicity; in the reader, on JSONDecodeError, retry once after 50ms before surfacing the error.

### [HIGH] `get_games_in_progress` scans the entire history every second
`dashboard.py:85-121` globs `self_play_games_status/*.json` and parses every file on every `/api/games_status` poll (every 1s). After a few iterations this is hundreds–thousands of small files, all re-parsed continuously on 4 gunicorn workers. The "current loop only" client-side filter (`index.html:732-742`) does the filtering *after* the server has done the full scan. Fix: pass `?loop=N` to the API, filter server-side by filename prefix (`L{N}_G*.json`), and/or keep a tiny in-memory cache keyed by mtime. Also, the front-end filter on line 736 uses `g.loop_number + 1 === currentLoop || g.loop_number === String(currentLoop)` — the +1 confuses 0- vs 1-based loop numbering and looks like a band-aid for a bug elsewhere.

### [HIGH] `cleanup_files()` crashes when `runs/` or `self_play_games_status/` don't exist
`loop.py:167-176` calls `TENSORBOARD_LOG_DIR_BASE.iterdir()` and `SELF_PLAY_GAMES_STATUS_PATH.iterdir()` without an `.exists()` guard. On a fresh checkout (as the project is right now — neither `runs/` nor `training_examples/` exists), `--keep-old-files` defaults to false and this raises `FileNotFoundError` before anything starts. Fix: `if path.exists(): for item in path.iterdir(): ...`.

### [MED] `Metrics.add_scalar` ignores `loop`/`game` and uses a private rolling step
`metrics.py:17-21` takes `loop, game=None` but writes `self.metric_step[name]`, an in-process auto-increment counter. Consequence: (a) when the loop calls `metrics.add_scalar("Training/Total_Loss", x, loop)`, the x-axis on TensorBoard is *not* `loop` — it's "Nth call to this exact metric name ever in this process," which only coincidentally equals loop when called once per iteration. (b) Per-loop metrics like `f"Training/Epoch_Loss/Loop{loop}"` (`loop.py:660`) intentionally make new metric names per loop, so each gets its own counter starting at 0 — fine, but it explodes scalar tag count. (c) `add_metric` from `self_play.py` would, if enabled, write thousands of `MCTS/Run_Network_Time` points per game with step=N-th-call, completely unrelated to game/move number — making the curves uninterpretable. Fix: actually use the passed `loop`/`game` argument as the global step; offer separate `add_scalar_loop` / `add_scalar_per_move` helpers with explicit step semantics.

### [MED] Per-game TensorBoard metrics are disabled and `Metrics` is not multiprocess-safe
`loop.py:198` sets `metrics=None` for every self-play worker with the comment "disabled… too much space on disk." That kills *all* per-move MCTS metrics in `self_play.py` (`MCTS/Select_Leaf_Time`, `MCTS/Run_Network_Time`, leaf-depth histograms, memory usage, etc., lines 190-239, 518-587). Even the `Metrics.lock` (`metrics.py:11`) is a `threading.Lock`, not multiprocess-safe — workers are spawned subprocesses and would each open their own `SummaryWriter` to the same dir, mangling event files. Fix: keep workers' per-game metrics, but aggregate by writing one row per game (already done in `play()` at lines 566-587) into a per-iteration JSONL or LMDB, then have the parent process flush aggregated histograms/scalars into TB once per loop. That gives you MCTS time histograms, throughput, etc., without the disk explosion.

### [MED] Coverage gaps in what's logged
Despite the rich `train.py` metrics, the following are missing:
- **Self-play throughput**: games/min, moves/sec, queue-wait time per worker. The data exists (start/end timestamps in the status JSONs) but is never aggregated.
- **Game-length distribution**: only the mean (`loop.py:593`); not a histogram. `loop_metrics.game_lengths` is collected but only written to a sidecar JSON.
- **Gating win-rate history**: written as a scalar but only when a gate runs; no rolling avg or confidence interval; the rejected-vs-promoted ratio over time is not surfaced.
- **GPU utilization & VRAM**: `/api/system_metrics` only does CPU+RAM via psutil (`dashboard.py:174-189`). For a CUDA training loop this is the wrong half — add `torch.cuda.utilization()` / `torch.cuda.memory_allocated()` / `nvidia-smi --query-gpu=...`.
- **Training data freshness**: how stale is the oldest example in the training window? (Critical for diagnosing on-policy vs off-policy drift with `max_training_window`.)
- **Validation loss**: there's none. `TrainingConfig.val_dir` exists but `train.py` never uses it. The split mentioned in CLAUDE.md ("selfplay + holdout split") is unimplemented in `train.py`.
- **Model graph / weight histograms**: never call `SummaryWriter.add_graph` or `add_histogram` of parameters. Gradient *norms* are logged but the actual weight/grad *distributions* aren't.
- **Sample games / text**: no `add_text` of action sequences; no way to inspect a representative game from TensorBoard.
- **Per-loop wall-clock**: time-per-iteration is not recorded as a scalar; the only timestamps are the log filenames.

### [MED] Dashboard does not expose checkpoints or per-iteration drill-down
`dashboard.py` shows only the *current* loop's status. It never reads `model_checkpoints/` (which exists per CLAUDE.md and on disk), never lists per-iteration training metrics from `logs/loop/loop_metrics_*.json` (which `loop.py:138-145` writes), never links a per-game `logs/self_play/*.log` file to the games-list rows. The Self-Play Games table has no link/expandable row to inspect the per-game log or final scores. Fix: add a `/checkpoints` route listing checkpoint dirs with model name, mtime, optional gating win-rate badge; an `/iteration/<N>` view that renders the relevant `loop_metrics` slice + the games for that loop with log links.

### [MED] `endAfterCurrentLoop()` and form-only POST clobber `training_config`
`index.html:681-698` and `dashboard.py:191-250`: the POST handler rebuilds `loop_config_to_save` from `request.form`. Anything not in the form (e.g. `target_experiences`, present in `loop_config.json` since `loop.py:469-472`) gets *dropped*. After a single click of "End After Current Loop" or "Update Configuration", the `target_experiences` field disappears from disk and `loop.py:468` falls back to `target_experiences=0`. Fix: load existing config, merge form deltas, write back; or add all editable fields to the form.

### [LOW] Float→int coercion in form parser is footgun
`dashboard.py:221-224`: `if val.is_integer(): val = int(val)`. `weight_decay=0.0001` survives, but a user typing `1.0` for `lr` silently becomes `1` (int), and downstream PyTorch happily uses int LR. Fix: keep the type declared in `TrainingConfig` field annotations; only coerce when the dataclass field is `int`.

### [LOW] `profiler.py` is dead code in production
`rl18xx/agent/alphazero/profiler.py` is only used by its own `__main__` (line 178). Nothing in `loop.py`, `train.py`, `self_play.py` imports it. Note: the call to `manual_big_clone()` on line 125 doesn't match the current `pickle_clone()` API documented in CLAUDE.md, so the file is bit-rotted. Either delete it or wire it into a `make benchmark` target so the comparison is run regularly and its output is dumped somewhere TensorBoard-visible (e.g. `add_text`).

### [LOW] Hardcoded TensorBoard URL `/tensorboard/` is broken without a reverse proxy
`dashboard.py:22` sets `TENSORBOARD_URL_PATH = "/tensorboard/"`, surfaced via the "Open TensorBoard" link in `index.html:467`. But `startup.sh` binds TensorBoard on `:6006` directly with no proxy — the link 404s. Either link to `http://<host>:6006/` (using `request.host`), or document/add an nginx config that proxies `/tensorboard/` to `localhost:6006`.

### [LOW] Dashboard file structure / template style
`dashboard.py` is a single 275-line module; `index.html` is one 935-line file with all CSS + JS inline (no `static/` directory exists). Polling intervals (`setInterval(..., 1000)`/`2000`) are hard-coded; no exponential backoff on errors; `fetch` chains lack request-cancellation, so slow gunicorn workers can queue up multiple in-flight requests per tab. The script makes 3 independent calls every 1–2s — fold into a single `/api/dashboard_state` endpoint to halve worker load and remove cross-polling drift. Consider splitting into Flask blueprints (`status`, `config`, `games`, `system`) and moving CSS/JS to `static/`.

### [LOW] `loop_config.json` "num_games_per_iteration" knob is now mostly ignored
The schema in the file shows `num_games_per_iteration: 50` but `loop.py:468` computes the count from `target_experiences // ESTIMATED_MOVES_PER_GAME` regardless. The dashboard form still asks for "Games per Iteration" as a required field (`index.html:514-517`) — misleading and confusing. Either remove the form field or honor it.

---

## Cross-cutting themes

Several findings recur across sections; calling them out so a single round of cleanup can resolve many issues at once:

1. **Atomic file writes.** `loop_status.json`, `self_play_games_status/*.json`, `model_checkpoints/.../config.json`, and `loop_config.json` are all written non-atomically while being polled by readers. Pattern: `tmp = path.with_suffix(".tmp"); tmp.write_text(...); os.replace(tmp, path)`.
2. **Dead / broken code.** `Encoder_SSME`, `AlphaZeroSSMEModel`, `aux_phase_head`, `make_encoded_ssme_game_state_model_friendly` stub with wrong signature, `profiler.py`, `autoresearch/run_experiment.py` calls obsolete API. All can be deleted or wired correctly without behavior change.
3. **Train/inference distribution mismatch.** Canonicalization is applied to training data but not to inference inputs. Pick one — apply both places or neither.
4. **Magic numbers scattered across files.** `epsilon=0.03`, `validation_percentage=0.05`, bid offsets `335`/`359`, `ESTIMATED_MOVES_PER_GAME=1000`, `10.0` in Dirichlet alpha derivation. Hoist to config / a constants module.
5. **No validation anywhere.** `TrainingConfig.val_dir` is wired but unused; `do_pretraining` merges val into train; `train_model` lacks a no-grad eval pass; TensorBoard has no val curves.
6. **Frictionful fresh-start.** `cleanup_files`, `cleanup_model_and_data`, dashboard pathing, and `cmd_pretrain` all break on a clean checkout in different ways. A `make fresh` target (or `python main.py setup`) that creates the seed checkpoint + required directories would solve all of them.
7. **Statistical noise underestimated.** Gating eval (deterministic), value head's mismatched units (drives MCTS toward wrong actions silently), and tiny validation split (≤12 games) all introduce noise that masquerades as signal.

---

## Notes for next steps

- The roadmap doc lists items the user is already aware of (temperature in `pick_move`, lazy encoding, etc.). Several of those are now **already fixed in code** — flagged in the "Known issues" subsections. The roadmap doc itself should be updated.
- After landing the Top-9 fixes, a sane sequence to retest the pipeline end-to-end:
  1. `uv run python main.py pretrain --data-dir human_games --epochs 10` produces a saved checkpoint and val curves
  2. `uv run python main.py arena --agents mcts mcts mcts mcts --readouts 50` sanity-checks the pretrained model
  3. `uv run python main.py train --iterations 1 --gate-games 4` runs one full self-play → train → gate cycle with sensible defaults
- Many MED/LOW items can be batched into a single "cleanup" commit (atomic writes, dead-code removal, magic-number hoisting) without behavior changes.

---

## `mcts_nn_roadmap.md` progress audit

Walking through the older roadmap against current code:

| Item | Status | Note |
|---|---|---|
| 1.1 Temperature in `pick_move` | ✅ done | self_play.py:139-158 |
| 1.2 `sim_count_this_move` init | ✅ done | self_play.py:465 |
| 1.3 `tree_search_duration` timing | ✅ done | self_play.py:485-489 |
| 2.1 Masked-logits policy loss | ✅ done | train.py:219 (`masked_fill(~legal_mask, -inf)`) |
| 3.1 GNN residual connections | ✅ done | model.py:255, 435-441 |
| 3.2 Kaiming `leaky_relu` init | ✅ done | model.py:328 |
| **3.3 Autoregressive `FactoredPolicyHead`** | **❌ not done** | New design below — apply to Transformer only |
| 4.1 Lazy MCTS-node encoding | ✅ done | mcts.py:154-165 |
| 4.2 `parallel_readouts = 32` | ✅ done | config.py:205 |
| 4.3 Adaptive readouts by complexity | ✅ done | self_play.py:247-251, 475-479 |
| 4.4 Cached ActionMapper | ✅ done | mcts.py:27-31 |
| 5.1 FiLM phase conditioning | 🟡 partial | Transformer correct (inside ResBlock, identity init). GNN still applied *after* the block (model.py:460-464) — see (d) findings |
| 5.2 Per-player value head + active indicator | 🟡 done-but-broken | Indicator is wired but coupled with the canonicalization mismatch (d-HIGH). Drop the one-hot once canonicalization is unified |
| 5.3 Deeper value head | ✅ done | `value_head_layers = 3` |
| 5.4 Aux action-count loss | ✅ done | train.py:241-244 |
| 5.5 LayerNorm after gated fusion | 🟡 partial | Transformer uses LN throughout. GNN still has BN+LN stacked (model.py:280) — see (d-MED BatchNorm) |
| 5.6 Attention-based hex scoring | ✅ done | `hex_scorer` (model.py:72) and Transformer policy head |
| 6.1 Depth-discounted backup | ✅ done | `backup_discount = 0.995` (config.py:206), applied in mcts.py:413 |
| 6.2 Per-round-type `c_puct` | ✅ done | `c_puct_by_round` dict (config.py:199-225), applied in mcts.py:244 |
| 6.3 Progressive widening | 🟡 partial | `pw_c=1.0, pw_alpha=0.5` and the PW select logic exist (mcts.py:286), but PW currently only constrains expansion over the existing **discrete** action set. Not yet wired to a continuous price head — see "Continuous prices via PW" below |
| 6.4 Score-value targets | ✅ done | `use_score_values = True` (config.py:209), branch in train.py:229-237 |
| 6.5 FP16 inference | ✅ done | `use_fp16_inference = True` (config.py:210) |
| 6.6 Policy entropy bonus | 🟡 partial | `entropy_weight = 0.01` (config.py:164) declared; usage in train.py not verified — confirm and wire if missing |
| 6.7 Gradient accumulation | ✅ done | `gradient_accumulation_steps` config + train.py:181 |

**Remaining roadmap work**: 3.3 (autoregressive policy head), 6.6 verification (entropy bonus actually consumed), and confirming 6.3 extends to continuous price actions (see below).

---

## Design decisions from the model-design walkthrough

These were settled in conversation while reviewing (d) and are recorded here as the agreed direction for the Transformer model. They are *new requirements*, not bugs.

### Canonicalization unified at encode-time

`Encoder_GNN.canonicalize_perspective` rotates player slots and the value target so the active player sits at slot 0. Today this only happens at training time (`dataset.py:85-93`); inference uses the raw encoder output. The model is trained on one distribution and evaluated on another (HIGH finding in (d)).

**Decision**: apply canonicalization inside `encoder.encode()`. Always emit `active_player_idx = 0`. Drop the `one_hot(active_player_idx)` indicator from the value head — with canonicalization, the active player's position is implicit (slot 0). Training and inference paths converge.

`encoder.encode()` should also return the rotation amount used (or equivalently, the absolute → canonical player index mapping). MCTS uses this to **unrotate value vectors** before backup: each node's value vector flows up the tree in absolute player order, and each ancestor's PUCT computation reads `Q[absolute_active_player_idx]` directly.

### Token layout: entity ordering and identity

The encoder emits a variable-length sequence of entity tokens. The principles for each entity type:

**Player tokens** — exploit player permutation symmetry:
- Slot 0 = **active player** (whoever is making the current decision).
- Slots 1..N−1 = remaining players in **SR-priority order, starting from the next-to-have-priority**.
- Per-token features: cash, certs, share count per corp, privates owned, cash spent in current auction round, etc.
- Per-token flags:
  - `has_priority`: 1 for whoever holds SR priority (= slot 0 during SR, typically slot >0 during OR).
  - `is_president_of_active_corp`: 1 for slot 0 during OR (redundant with positional info, but explicit signal speeds learning).
- **No player-identity feature**. Player_1 vs Player_3 isn't strategically meaningful — it's a label. Excluding identity gives the model free player-permutation symmetry, which is a real 4× (or up to 6×) data augmentation effect during training. For human game data this also means individual player styles get washed out, which is the right tradeoff (those styles are noise, not strategy).

**Corp tokens** — preserve corp identity, capture order via features:
- Stable order, sorted by corp short name (B&O, B&M, C&O, CPR, ERIE, NYC, NYNH, PRR). Identity is permanent; positions never shuffle across moves.
- Per-token features: cash, trains, share price, tokens placed, percent floated, home hex, plus a `corp_id` one-hot.
- Per-token operating-round features:
  - `has_operated_this_or`: 0/1
  - `is_currently_operating`: 0/1
  - `will_operate_this_or`: 0/1 (= floated AND not operated AND not currently)
  - `operating_order_idx`: normalized rank (0 = next, 1 = after, ...). Updates when share prices change.
  - `share_price_rank`: smooth signal for the order.
- Why stable identity over dynamic operating order: corps in 1830 have persistent strategic identities (B&O's Baltimore home, NYC's adjacency to the rich east), and the model wants to build per-corp representations across many states. Operating order is a single dynamic signal that's cheap to encode as a feature; corp identity isn't.
- Emergency share dumps that shuffle another corp's price → `operating_order_idx` and `share_price_rank` features update; token positions stay put. The Transformer sees a feature-update, not a structural permutation.

**Private tokens** — fixed identity order (length 6 for 1830, variable across titles).

**Global token** — phase, OR-set number, bank cash, train limit, etc.

### Cross-modal: making "player X owns corp Y" trivially attendable

Redundant cross-reference features on both sides:
- Player tokens already include `shares_of_corp_K` (one feature per corp K).
- Corp tokens add `president = player_at_slot_K` (one-hot over player slots) and `presidents_share_owned` (0/1).
- Player tokens add `is_president_of_corp_K` (one-hot over corps).

The Transformer can derive ownership from shares alone via attention, but the explicit cross-references make it a one-attention-step lookup. Cheap to compute, speeds early-training convergence.

### Hierarchical policy head: type → params (with autoregressive sub-types)

The policy is structured hierarchically as `P(type) × P(params | type)`, where the parameter space varies per type and can itself be autoregressively factored.

**Top level — type categorical** (~12 entries):
```
Pass, Bid, Par, BuyShares, SellShares, LayTile, PlaceToken,
BuyTrain, Dividend, CompanyBuyShares, Bankrupt, [any-time exchanges]
```

**Per-type parameter sub-heads**:

| Type | Param structure | Sub-head shape |
|---|---|---|
| Pass | (none) | — |
| Bid | per-private (6 entries) + continuous price | categorical(6) + Normal(mean, log_std) per private |
| Par | corp × par-price (8 × 6 = 48) | parallel-factored: Linear(d_trunk, 8) + Linear(d_trunk, 6) |
| BuyShares | corp × source (8 × 2 = 16) | parallel-factored |
| SellShares | corp × count (8 × 5 = 40) | parallel-factored |
| **LayTile** | hex × tile × rotation | **autoregressive: P(h) × P(t \| h) × P(r \| h, t)** |
| **PlaceToken** | hex × slot | **autoregressive: P(h) × P(slot \| h)** |
| BuyTrain | source × type (~20) + continuous price | categorical(~20) + Normal(mean, log_std) per source+type |
| Dividend | binary {payout, withhold} | Linear(d_trunk, 2) |
| CompanyBuyShares | corp × target × side (~30) | parallel-factored |
| Bankrupt | (none) | — |
| Any-time | ~5 fixed exchanges | per-exchange binary {exchange, skip} |

**Why autoregressive for LayTile/PlaceToken**: the (hex, tile, rotation) triple has strong internal correlation — a good tile depends on the chosen hex's geography. The current independent factoring misses this; the autoregressive factoring captures it explicitly. PlaceToken's (hex, slot) has the same structure on a smaller scale.

**Why parallel-factored elsewhere**: Pass/Par/Bid/SellShares/Dividend have no meaningful intra-type correlation to exploit. Adding autoregressive machinery would be cost without benefit.

### Autoregressive conditioning mechanism (concrete)

Conditioning happens via **concatenated inputs to each sub-head**, not via sequential decoding. The network computes all conditional distributions in parallel; downstream code picks the slice it cares about.

For LayTile:
```python
# Inputs
trunk:       (B, d_trunk)
node_embs:   (B, 93, d_map)         # per-hex embeddings from map encoder

# P(hex | state)
hex_logits = hex_scorer(node_embs).squeeze(-1)                        # (B, 93)

# P(tile | state, hex=h) — emitted for ALL hexes in parallel
trunk_b = trunk.unsqueeze(1).expand(-1, 93, -1)
tile_input = torch.cat([trunk_b, node_embs], dim=-1)                  # (B, 93, d_trunk + d_map)
tile_logits = tile_head(tile_input)                                    # (B, 93, 46)
# tile_logits[h] is "given the model wants to lay a tile at hex h, what tile?"

# P(rotation | state, hex=h, tile=t) — emitted for ALL (hex, tile) pairs
tile_embs = tile_embedding.weight                                      # (46, d_tile)
rot_input = concat_with_broadcast(trunk, node_embs, tile_embs)        # (B, 93, 46, d_trunk+d_map+d_tile)
rot_logits = rotation_head(rot_input)                                  # (B, 93, 46, 6)
```

Each sub-head emits **logits** (matching the existing canonical-output pattern — logits are kept for numerical stability of `log_softmax`, the masking pattern `masked_fill(-inf)`, and for compatibility with `F.cross_entropy` / `F.kl_div`).

### MCTS prior assembly and training loss decomposition

**MCTS inference path**: `run_many_encoded` still returns `(probabilities, log_probs, values)` with the prior indexed by the legal-action set. The prior is computed *only for legal actions*, not as a flat 26,535-dim tensor:

```python
type_logits, sub_heads, value_logits, aux = model.forward(state)
log_P_type = F.log_softmax(type_logits, dim=-1)

for i, action in enumerate(legal_actions):
    if action.type == LayTile and action == (h, t, r):
        log_P[i] = log_P_type[LayTile_idx]
                 + log_P_hex[h]
                 + log_P_tile_given_hex[h, t]
                 + log_P_rot_given_hex_tile[h, t, r]
    elif action.type == Bid and action.private == p:
        log_P[i] = log_P_type[Bid_idx] + log_P_private[p]
        # price is sampled separately from Normal(price_mean[p], exp(price_log_std[p]))
    ...
prior = exp(log_P)  # only over legal actions
```

The flat 26,535-dim policy tensor disappears entirely. The MCTS interface stays the same — it consumes a prior over the legal action set.

**Self-play training loss**: the MCTS visit-count target `pi` (a distribution over the flat legal-action set) is **decomposed** into per-level marginals:

```python
# For the LayTile portion of pi:
pi_lay = pi[lay_tile_indices].view(B, 93, 46, 6)
pi_hex_marginal = pi_lay.sum(dim=(2, 3))                  # (B, 93)
pi_tile_given_hex = pi_lay.sum(dim=3) / pi_hex_marginal.unsqueeze(-1).clamp(min=eps)
                                                           # (B, 93, 46)
pi_joint = pi_lay                                          # (B, 93, 46, 6)

loss_hex  = -(pi_hex_marginal * log_P_hex).sum(-1).mean()
loss_tile = -(pi_hex_marginal.unsqueeze(-1) * pi_tile_given_hex * log_P_tile).sum((-1, -2)).mean()
loss_rot  = -(pi_joint * log_P_rot).sum((-1, -2, -3)).mean()
```

Three weighted cross-entropy losses, one per autoregressive level. Each sub-head gets gradient signal weighted by how much visit mass passed through that decision. Parallel-factored types stay as standard CE on their flat sub-target.

**Supervised (pretraining) loss**: target is one-hot on `(h*, t*, r*)` (plus epsilon label smoothing). The decomposed loss reduces to:
```python
loss = -log_P_hex[h*] - log_P_tile[h*, t*] - log_P_rot[h*, t*, r*]
```
Three standard cross-entropies. Same shape as `nn.CrossEntropyLoss` with hard targets.

**Continuous price head loss**:
- Self-play: target is the visit-weighted mean price per `(type, entity)`; NLL against `Normal(predicted_mean, exp(predicted_log_std))`.
- Supervised: target is the human's chosen price; same NLL.

### Value head: per-player, variable-length, dual

**Inputs**: both the per-player entity tokens AND the trunk output. The trunk is doing phase-conditioned global reasoning (the FiLM ResBlocks add information the entity Transformer alone doesn't have); concat them and feed to the per-player MLP:

```python
class DualValueHead(nn.Module):
    def forward(self, trunk_out, per_player_embeddings):
        # trunk_out:              (B, d_trunk)
        # per_player_embeddings:  (B, num_players, d_entity) — from post-fusion entity tokens
        B, P, _ = per_player_embeddings.shape
        trunk_b = trunk_out.unsqueeze(1).expand(-1, P, -1)
        value_input = torch.cat([per_player_embeddings, trunk_b], dim=-1)  # (B, P, d_e + d_trunk)
        win_loss = self.win_loss_mlp(value_input).squeeze(-1)              # (B, P)
        score = self.score_mlp(value_input).squeeze(-1)                    # (B, P)
        return win_loss, score
```

**Variable-length** because the MLP weights are *shared across player slots* — same MLP applied independently to each player's `(entity_emb + trunk)` concat. This is what gives variable-player support and player permutation symmetry.

**Dual heads (KataGo-style)**:
- **Win-loss head**: predicts {+1 / 0 / −1} player-share-of-winners. Used by MCTS for backup. Softmax over players gives the win-probability distribution that MCTS reads.
- **Score head**: predicts normalized net-worth fraction at game end. Trained with MSE. **Not consumed by MCTS** — purely auxiliary, providing dense gradient signal to the trunk during training. Outcome is sparse (3 values per player); score is continuous and unique per game, which gives the trunk richer supervised signal.

Both heads run in parallel from the same per-player + trunk input.

### Value vector ordering for MCTS

- Network emits values in **active-player-relative order**: `value[0]` = active player, `value[1]` = next in SR priority, etc. Matches the canonicalized token layout.
- **MCTS backs up in absolute player order**: each node's value vector is unrotated to absolute order using the rotation amount recorded by the encoder. This means a value backed up from depth-3 to depth-1 stays consistent regardless of which player is active at each depth.
- At any node, PUCT reads `Q[absolute_active_player_idx]` — a single lookup.
- Multi-corp ownership during OR doesn't complicate this: value vectors are *player-level*, not turn-level. A player who's president of two corps just has the same `value[player_idx]` regardless of which of their corps is currently operating.

Prior work: Sturtevant 2003 (per-player value vectors in multi-player search), Cazenave 2017 (multi-player MCTS), DeepMind Hanabi 2020 (per-player value heads).

### Continuous-price action space via progressive widening

Full continuous prices for Bid / BuyCompany / BuyTrain, handled in MCTS via progressive widening.

**Output structure**: see the "Hierarchical policy head" table above. The categorical level identifies `(type, entity)`; the continuous price level is parameterized by `Normal(mean, exp(log_std))` per legal (type, entity), truncated to that action's legal range.

**MCTS structure** — two-level tree:
```
BuyTrain decision node
├── BuyTrain(source=depot, type=3T)            ← deterministic price, single child
├── BuyTrain(source=depot, type=4T)            ← single child
├── BuyTrain(source=corp_A, train_X)
│   ├── price=$300  (sampled at categorical visit 1)
│   ├── price=$250  (sampled at categorical visit 4)
│   └── ...                                     ← PW: k = pw_c × N^pw_alpha
└── BuyTrain(source=corp_B, train_Y)
    └── ...
```

Categorical level fully expanded; continuous level uses PW with `c=1.0, α=0.5` (existing config). Each sampled price snaps to the legal grid ($5 increments for bids); duplicate snapped prices increment the existing child's visit count instead of creating duplicates.

**Benefits**:
- Eliminates `make_action_model_friendly` and `make_encoded_game_state_model_friendly` entirely.
- Human game data trains directly on raw human prices, no fudging.
- Model plays realistic games against humans.

Prior literature: Couëtoux et al. 2011 "Continuous UCT", Couëtoux & Helmstetter 2014 "Progressive Widening".

### Any-time actions at fixed trigger points

Engine and action_mapper already support the relevant exchanges (e.g., MH→NYC). Only the decision-point placement needs to be added. Confirmed trigger points:

(a) **Between any player's turn in the stock round** — including before the first and after the last (`num_players + 1` triggers per SR).
(b) **Immediately after private revenue is paid** at the start of each OR set, before any corp operates.
(c) **Any turn the player is active** in stock and operating rounds — handled automatically by adding the action to the legal-action set.

Implementation requires engine work: a `Game.pending_anytime_decisions()` method exposing `[(player_id, eligible_action)]` at trigger points, plus Round/Step changes to pause and yield those decision-points. MCTS treats each pending decision as a separate decision node owned by the polled player; action space is `{exchange, skip}`. Tree depth grows modestly (~`num_eligible × 2` extra nodes per trigger).

**Open question (needs engine investigation)**: when does the existing engine allow non-active player actions today? If `BuySellParShares` or `Auction` steps already have any concept of "pause and ask another player," extend those; otherwise add new pause/yield mechanics.

### Variable player count (2-6)

The action space itself is already player-count-invariant — every action is keyed to corps or hexes, not to specific player slots. What needs to change:

- **Encoder**: emit `num_players` player tokens instead of always 4. The Transformer is set-length-agnostic, so the same model code handles 2-6 players given the right input.
- **Value head**: shared per-token MLP, output dim = num_players. Already designed for variable length above.
- **Per-player-count game constants** (starting cash, cert limits, train limits): the engine already handles these; the encoder just needs to expose them as features (e.g., in the global token).

Variable player count is **Transformer-only**. The GNN's fixed-slot flat-vector encoder would require essentially rewriting it as a Transformer to support variable counts.

### Swappable map encoder: three variants

Three map-encoder architectures, all supporting the same downstream pipeline. Architectural shape:

```python
class MapEncoder(nn.Module):
    """Returns per-hex embeddings + pooled summary."""
    def forward(self, node_features, structural_info) -> tuple[Tensor, Tensor]:
        # per_hex_embeddings: (B, num_hexes, d_map)
        # map_pool:           (B, d_map_pool)
        ...

class HexTransformerMapEncoder(MapEncoder):       # pure attention
    # Pre-norm Transformer, structural attention bias (distance/direction/track-connectivity)
    # Full receptive field per layer, parameter-heavier than ResNet

class HexResNetMapEncoder(MapEncoder):            # pure convolution
    # 93 hexes mapped to 11×17 offset rectangular grid; standard Conv2d ResNet
    # 10+ layers at ~128 channels → ~21×21 receptive field, covers full 1830 map
    # Parameter-efficient for local spatial patterns

class HexHybridMapEncoder(MapEncoder):            # ResNet + attention
    # 4-6 Conv2d ResNet blocks for local mixing
    # 1-2 self-attention layers at the top for global mixing
    # Best of both: efficient local patterns + global route reasoning
```

The rest of the model (entity Transformer, cross-modal fusion, FiLM trunk, policy + value heads) is shared. Configured via `ModelConfig.map_encoder = "transformer" | "resnet" | "hybrid"`.

**HexTransformerMapEncoder details**:
- Pre-norm Transformer over 93 hex tokens.
- Adjacency captured via `StructuralAttentionBias`: per-head learnable biases added to attention logits before softmax, for distance bucket, direction (6 hex sides), and dynamic track connectivity.
- Without the structural bias the model has to learn adjacency from data; the bias is the inductive prior that "neighbors matter more" while still allowing global attention.
- The current code has this; it's the HIGH bug (`_compute_structural_matrices` never invoked) that breaks the bias today.

**HexResNetMapEncoder details**:
- Offset-grid mapping: 93 hexes onto an 11×17 rectangular grid (axial coordinates with mask for non-hex cells).
- Standard `Conv2d` ResNet, target ~10 layers at ~128 channels.
  - Receptive field after 10 layers: ~21×21, enough to cover any 1830 route end-to-end.
  - Param count: ~1.5M, vs ~2.4M for the Hex Transformer.
- Convolution mixes channels locally per pixel: encodes both static hex adjacency (via offset-grid topology) and dynamic track-path adjacency (via per-pixel "track exits in direction K" channels).
- A conv can directly learn "track-east on this hex + track-west on the east neighbor = connection" in one layer; long-range route propagation requires depth. 10 layers covers 1830's longest routes (~15 hexes).
- Optionally extend with 1-2 global attention layers at the top for explicit long-range relational reasoning (the "hybrid" variant). Start without, add if route reasoning struggles empirically.
- Native hexagonal convolutions (HexaConv, Hoogeboom et al. 2018) are an alternative but require custom kernels; offset-grid with `Conv2d` works with standard PyTorch.

**Why both**: side-by-side experimentation. ResNets are parameter-efficient and well-optimized in cuDNN; Transformers are more flexible for long-range relational reasoning. The pretraining + self-play infrastructure should treat them as interchangeable.

### Architecture renaming consideration

Once the map encoder is swappable, "AlphaZeroTransformer" is misleading (only the map encoder differs). A future refactor could rename to `AlphaZeroModel` parameterized by `map_encoder` config, with checkpoint dirs like `AlphaZero-hex_transformer/...` and `AlphaZero-hex_resnet/...`. Defer until both encoders are implemented and benchmarked.

### Cross-title strategy

Cross-title **weight** transfer not pursued — encoders, action spaces, and output heads are too title-specific.

Cross-title **architecture/code** reuse is the goal: the same model code works for any 18XX with appropriate title-config inputs. Per-title encoder subclasses define the action space, tile catalog, and any title-specific entity feature templates; the trunk, entity Transformer, value head, and policy head structure stay shared. Each title gets its own trained model. Hyperparameter intuitions (trunk dim, LR, num_res_blocks) transfer as starting points.

### Pooling consistency: use attention pool everywhere

Today's code uses three different pooling mechanisms in three places (`[CLS]` token in the entity Transformer, attention pool in the hex Transformer, mean/CLS in cross-modal fusion). Standardize on **attention pool** — a learnable query vector that attends to encoder output tokens — and use it in all three places.

```python
class AttentionPool(nn.Module):
    def __init__(self, d, num_heads=4):
        self.query = nn.Parameter(torch.randn(1, 1, d))
        self.attn = nn.MultiheadAttention(d, num_heads, batch_first=True)
        self.ln = nn.LayerNorm(d)
    def forward(self, tokens):                  # (B, N, d)
        B = tokens.size(0)
        q = self.query.expand(B, -1, -1)        # (B, 1, d)
        out, _ = self.attn(q, tokens, tokens)
        return self.ln(out.squeeze(1))          # (B, d)
```

Use cases:
- `HexTransformerMapEncoder` → `map_pool` (already using attention pool today).
- `EntityTransformer` → `entity_summary` (replaces today's `[CLS]` token output).
- Post-fusion summarization → `fused_summary` (replaces today's mean-pool).
- For `HexResNetMapEncoder` / `HexHybridMapEncoder`, flatten the spatial grid to a sequence and apply the same `AttentionPool` — works for any token sequence.

Why standardize: the three map-encoder variants (Transformer, ResNet, Hybrid) all need a consistent way to produce `map_pool`. `[CLS]` doesn't have a natural place in a CNN. Attention pool works everywhere.

Mathematically, `[CLS]` and attention pool are nearly equivalent — the difference is whether the summary token participates in the encoder's full self-attention (`[CLS]`) or sits in a separate attention step after the encoder (pool). For our use, the cleaner separation is preferable.

### History features (deferred, but worth tracking)

The model today sees a single state snapshot. Some strategically-relevant signals are *momentum-based* — e.g., is player X aggressively accumulating Y's shares? Is Z's share price trending up? Are corps liquidating to buy trains? Single-state encoding cannot see these.

**MCTS provides forward lookahead, not backward memory** — they're orthogonal. MCTS lets the model simulate "if I do X, then Y happens"; history features let it see "Y has been happening for the last 3 turns." Both useful, neither replaces the other.

For 18XX the cheap version is **delta features** added to global / per-entity tokens:
- Δshare-price per corp over last 1, 5, 10 actions
- Δcash per player over last 1, 5 actions
- Δshares-owned per (player, corp) pair over last 1, 5 actions
- Last-action-type one-hot (immediate context)
- Actions since last stock round / OR set boundary

~50-100 additional scalar features total, computed from the action history. The encoder maintains a small rolling buffer of recent actions and computes deltas at encode time.

Stacked state history (AlphaGo-style: last N full states as input channels) is too expensive for 18XX — game states are ~1KB encoded and 90% of features don't change between adjacent moves.

GRU/LSTM hidden state across MCTS visits complicates parallelism (hidden state has to propagate through tree expansion) and offers little over delta features for this domain.

**Recommendation**: defer until a baseline is working. Add delta features once the variable-player Transformer is trained and we have a reference point to compare against.

## Design decisions from the self-play walkthrough

These were settled in conversation while reviewing (c). They are *new requirements*, not bugs in the existing findings.

### Truncated games use net-worth-derived win/loss + score targets

Today, games hitting `max_game_length=1000` are recorded as draws (all zeros). This throws away signal from a large fraction of self-play games, especially in early training when random play rarely terminates naturally.

**Decision**: at truncation, use net worth at the truncation step to compute both targets:
- **Win-loss target**: {+1, 0, -1} share-of-winners derived from "whose net worth is highest." Same formula as natural game-end with bank-broken / train-exhausted.
- **Score target**: normalized net-worth fraction. Identical formula for natural and truncated games.

Track `truncation_rate` as a training metric. If the model learns to stall at a winning position to game truncation, add a small `-0.1` truncation penalty to all players' values. Otherwise rely on the engine's natural economic incentives (dividends, train obsolescence) to push toward completion.

### Always save trained checkpoints; gate decides promotion separately

Today, `train_model` returns the model in-memory and `save_model` is only called after gating succeeds. This conflates "save weights to disk" with "promote as the new best," and means a crash between training and gating loses the candidate.

**Decision**: separate the responsibilities.
- `train_model` (or its wrapper) **always** saves the trained candidate as a numbered checkpoint immediately after training.
- A separate `current_best` pointer (`model_checkpoints/<arch>/current_best.json` or symlink) records which checkpoint is the current best.
- Gating updates the pointer if the candidate passes; otherwise the pointer stays unchanged.
- Failed candidates remain on disk for inspection/replay; optional retention policy later.

Benefits: crash safety, ability to inspect rejected candidates, cleaner separation of train and gate responsibilities.

### Variable player count throughout self-play

Self-play infrastructure needs to support 2-6 player games:
- `_create_fresh_game(num_players)` parameterized.
- `SelfPlay.run_game` creates `num_players` `MCTSPlayer` instances.
- `evaluate_candidate` / `_play_gate_game` parameterized by player count.
- Loop config gains `player_count_distribution` (e.g., `{4: 0.6, 3: 0.2, 2: 0.05, 5: 0.1, 6: 0.05}`) — trains the model on the full distribution rather than just 4-player games.
- LMDB schema unchanged — the encoded state's variable-length token sequence implicitly carries the player count.

Enabled by the Transformer model's variable-length entity sequences.

### Forced-move chaining inside MCTS

Today, the outer `MCTSPlayer.play()` skips MCTS for `num_legal_actions == 1`, but inside the tree, forced-move nodes are expanded and evaluated like any other. Long forced sequences (BuyTrain pass cascades, certain end-of-OR sequences) waste tree depth and visit budget.

**Decision**: add forced-move chaining inside `MCTSNode.maybe_add_child`. When the resulting node has `num_legal_actions == 1`, auto-advance through the forced action and store the *post-cascade* state as the actual child. The cascade is invisible to MCTS — it sees only nodes with real decisions. Visit counts and values are recorded only on the chain-terminating node.

Expected impact: 10-30% tree-depth savings on game positions with frequent forced passes.

### Self-play league composition (deferred)

Pure self-play against the current best can converge to degenerate Nash equilibria. OpenAI Five's league approach mixed in past versions and scripted baselines to prevent strategy collapse.

**Decision (deferred)**: once baseline self-play works, add `LeagueConfig.opponent_distribution`:
- 80% of games: current_best in all opponent seats.
- 15% of games: random sample from last K promoted checkpoints.
- 5% of games: scripted heuristic baseline (par at min, lay tiles for revenue, etc.).

Past-checkpoint games still produce useful training data — MCTS visit counts at each position reflect "what should be done from this state," regardless of who the opponent was. Don't add this until baseline self-play is producing improving models; the complexity isn't worth it for v0.

### Gating is optional; performance-based is the right tool *for development*

Standard AlphaZero (2017) and AlphaGo Zero (2017) **don't gate** — they always promote the latest trained model and rely on the system to self-correct if a bad update slips through. KataGo uses gating in early training but disables it once stable. The historical AlphaGo (2016) gated at 55% win rate, but that was specifically called out as removed in AlphaGo Zero because gating noise was costing more than it saved.

**Decision for now**: keep gating on during development. The safety net catches regressions from training bugs (NaN losses, optimizer issues, data corruption). Once the system is stable (several successful iterations in a row, no regressions), disable via `--no-gate` and rely on always-promote.

**Improved gating mechanics** (when on):
- Run **16-24 games** per gate evaluation, with seat rotation.
- **Seat-rotation tournament**: candidate plays seat 0 for K games, seat 1 for K games, etc. Current_best fills the other three seats. Controls for priority-deal seat advantages.
- Fix the deterministic-gating bug (add small Dirichlet noise during gating; today `dirichlet_noise_weight=0` makes every gate game identical given the same seat assignment).
- Optional diagnostic alongside: a small **self-play comparison** (K games where all 4 seats are candidate, K games where all 4 are current_best) — compare game length, truncation rate, and score distributions. Useful for detecting candidate degeneration (e.g., produces longer / more-truncated games).

**Symmetric-pair tournaments** (paired games starting from the same state with mirrored seat assignments) are tighter statistically but add engineering complexity for limited 4-player benefit since games diverge quickly anyway. Not worth pursuing unless we move to 2-player titles.

**Later (deferred)**: Elo-based continuous tracking across all promoted checkpoints. KataGo-style. Useful once the system is producing many checkpoints and we want continuous skill measurement; overkill for v0.

### Rolling training window, not "current iteration only" or "forever"

Today `max_training_window=0` means "use all data." After many iterations this becomes huge and includes stale data from very early models.

**Decision**: rolling window of **3-5 iterations** worth of examples (configurable). Provides enough volume for training while preventing very old data from dominating.

- AlphaGo Zero used last 500K games (very large window).
- KataGo uses similar rolling windows.
- Pure "current iteration only" starves training of volume — 50-100 games × ~500 moves = 25-50K examples per iteration is too thin for a 7M-param model.

Set `max_training_window ≈ 5 × games_per_iteration × avg_moves` as a starting default.

### No self-play holdout (pretraining has one; self-play doesn't)

**Decision**: no train/val split for self-play training.

- Self-play generates fresh data every iteration; there's nothing fixed to overfit to.
- Training is low-epoch (typically 1-3 per iteration), further reducing overfitting risk.
- Pretraining (supervised learning on ~250 human games) **does** need a holdout — that's a real overfitting regime. Wire `TrainingConfig.val_dir` to be used by `pretrain_model` only.
- Self-play training uses all generated data within the rolling window.

### Adaptive `parallel_readouts` (defer)

Today `parallel_readouts=32` is constant. High-branching positions (LayTile with 200 legal actions) benefit from larger batches; low-branching positions (forced sequences, end-of-OR cleanup) waste compute via virtual-loss duplication.

**Decision (deferred)**: scale `parallel_readouts` adaptively with `num_legal_actions`, similar to how `num_readouts` is already scaled. Defer until baseline benchmarks show GPU underutilization or virtual-loss thrashing.

### MCTS uses Rust engine exclusively

Today the MCTS code has dual paths: Rust adapter when the game state happens to be a `RustGameAdapter`, Python `BaseGame` otherwise. `_create_fresh_game()` in `loop.py:243` currently returns a Python `GameMap` game; `self_play.py:281-282` has a Python fallback. So MCTS effectively uses the Python engine by default.

**Decision**: MCTS uses `RustGameAdapter` exclusively. Changes:
- `loop.py:_create_fresh_game()` returns `RustGameAdapter(RustGame(players))`.
- Remove the Python `GameMap` fallback in `self_play.py:86-87, 281-282`.
- The `isinstance(self.root.game_object, RustGameAdapter)` check in `mcts.py:154` becomes unconditional.
- `Encoder_1830.encode(game)` keeps both paths (used elsewhere — pretraining, action-helper testing), but MCTS only takes the Rust path.

Python engine stays in the repo for replay tooling, action-helper enumeration (where used), and as a reference implementation. The training loop never touches it.

## Pretraining redesign (consolidated from the model + self-play changes)

The model architecture, action space, and value head changes downstream require the pretraining pipeline to be substantially restructured. Captured here in one place since they intersect.

### Training example shape changes

Today's training example: `(encoded_state, legal_action_indices, pi, value)` where `pi` is a 26,535-dim near-one-hot distribution.

New shape, given autoregressive policy + continuous prices + dual value head:

```python
TrainingExample = (
    encoded_state,           # variable-length entity tokens + node features (from new encoder)
    legal_action_mask,       # for in-network masking, structured per-sub-head
    target_action,           # structured: { type_idx, params }
    target_prices,           # dict: {(action_type, entity_idx): raw_price} for price actions
    win_loss_target,         # (num_players,) — share-of-winners in canonical order
    score_target,            # (num_players,) — normalized net-worth fraction in canonical order
    num_players,             # for variable-length batching
)
```

The structured `target_action` looks like:
```python
{
    "type_idx": 5,                           # LayTile
    "hex_idx": 42,                           # for LayTile/PlaceToken
    "tile_idx": 12,                          # for LayTile
    "rotation_idx": 3,                       # for LayTile
    # OR for Bid:
    # "type_idx": 1, "private_idx": 4
    # ... continuous price stored separately in target_prices
}
```

The flat `pi` tensor disappears. The encoder no longer emits a 26,535-dim view of the policy target.

### Loss decomposition for pretraining

The total loss is a sum of structured cross-entropies + price NLL + value losses + aux:

```python
# Type-level CE
type_loss = -log_P_type[target_type_idx]

# Per-type parameter CE (only for the chosen type's branch)
if target_type == LayTile:
    type_param_loss = (
        -log_P_hex[target_hex_idx]
        - log_P_tile_given_hex[target_hex_idx, target_tile_idx]
        - log_P_rot_given_hex_tile[target_hex_idx, target_tile_idx, target_rot_idx]
    )
elif target_type == Bid:
    type_param_loss = -log_P_private[target_private_idx]
    # Continuous price NLL on top:
    price_loss = -Normal(price_mean[target_private_idx],
                         exp(price_log_std[target_private_idx])).log_prob(target_price)
elif target_type == Par:
    type_param_loss = -log_P_corp[target_corp_idx] - log_P_par_price[target_par_idx]
# ... etc

# Value losses (dual head)
win_loss_loss = F.kl_div(log_softmax(win_loss_logits), win_loss_target)
score_loss = F.mse_loss(score_pred, score_target)

# Aux
aux_loss = F.mse_loss(aux_action_count_pred, log(legal_count))

total = (type_loss + type_param_loss
       + price_loss_weight * price_loss
       + value_loss_weight * win_loss_loss
       + score_loss_weight * score_loss
       + aux_loss_weight * aux_loss)
```

The "label smoothing" approach from the old `pi` target becomes natural CE on the structured target. The `epsilon=0.03` smear is replaced by PyTorch's `CrossEntropyLoss(label_smoothing=...)` on each sub-head independently.

### Action-space projection deleted

`make_action_model_friendly` (`pretraining.py:465`) and `make_encoded_game_state_model_friendly` (`pretraining.py:576`) are **deleted entirely**. Pretraining trains on the human's raw action prices:
- For Bid: target price = raw bid amount from the JSON. NLL on the continuous price head.
- For BuyTrain: target price = raw train price from the JSON.
- For BuyCompany: target price = raw company price from the JSON.

The brittle magic offsets (335, 359) for bid encoding go away. `convert_game_to_training_data` becomes a clean encoder → action-index → structured-target pipeline with no model-friendly fudging.

Cleaned games (`human_games/*.json`) don't need to be regenerated — the engine replay still works, but the encoding step no longer projects actions onto a restricted space.

### Canonicalization respected at encode time

`convert_game_to_training_data` (`pretraining.py:682`) currently builds `actual_value` with `player_mapping = {p.id: i for i, p in enumerate(sorted(game.players, key=lambda x: x.id))}` — so the value is indexed by stable sorted player ID.

With encoder canonicalization at encode time, the canonical order is "active player at slot 0, others in SR-priority order from active." The value target needs to match: rotate `actual_value` per training example so it aligns with the encoder's emitted token order at that position.

For pretraining specifically: at each historical position, the active player varies. The target needs to be rotated to match the encoder's canonical view at that position. Same logic as self-play; just made explicit since pretraining was previously emitting `actual_value` in an absolute order.

### Dual value targets per training example

The win-loss target (`actual_value`) already exists in `convert_game_to_training_data:700-706`. The score target needs to be added:

```python
final_net_worth = compute_final_net_worth(game)  # from game.result() or similar at game end
total = sum(final_net_worth.values())
score_target = {p_id: nw / total for p_id, nw in final_net_worth.items()}  # normalized fraction
```

Both targets are stored on every training example from the same game (the targets are game-level, not position-level). Both are rotated to canonical player order to match the encoder.

### Pretraining stays 4-player only

Human game data is all 4-player. Pretraining doesn't need variable-player support — only self-play does (where games at varying player counts are generated synthetically).

The Transformer model can handle 4-player inputs through the variable-length entity sequence (it's just a special case where N_players=4 always during pretraining). No code branching needed.

### `pretrain_model` wrapper structure

Combining the earlier decisions (separate from `train_model`, CE loss, cosine LR, validation, save best) with the new loss structure:

```python
def pretrain_model(model, train_dataset, val_dataset, config):
    """SL training for the Transformer model on human game data."""
    optimizer = Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)
    # NO load_optimizer_state — pretraining always starts fresh
    summary_writer = SummaryWriter(f"runs/pretrain_{timestamp}")
    
    best_val_loss = float('inf')
    for epoch in range(config.num_epochs):
        # Train pass
        for batch in train_loader:
            losses = compute_structured_losses(model, batch, config)  # the decomposition above
            total_loss = sum(losses.values())
            total_loss.backward()
            grad_clip(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            log_train_metrics(summary_writer, losses, lr, ...)
        
        # Val pass (eval mode, no_grad)
        val_losses = evaluate(model, val_loader, config)
        log_val_metrics(summary_writer, val_losses, ...)
        
        # Save best
        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            save_model(model, config.model_dir)  # writes best checkpoint to disk
        
    return TrainingMetrics(...)
```

Outputs to TensorBoard: `loss/{type, hex, tile, rot, price, win_loss, score, aux}/{train, val}`, plus `policy/top1_acc`, `policy/top5_acc`, `value/explained_variance` per epoch.

### Pretraining supports variable player count (2-6)

The user will re-add the original uncleaned games covering 2-6 player counts. Pretraining must handle them.

Changes:
- `convert_game_to_training_data` (`pretraining.py:682`) accepts variable `num_players`. The hardcoded `players = {1: "Player 1", ..., 4: "Player 4"}` at lines 332-337 and 697 becomes derived from the game's actual player list.
- Value targets become length `num_players` per example (replaces `torch.full((4,), -1.0)` at line 700). Both win-loss and score targets are sized to the game's player count.
- Player mapping for canonicalization rotates over the actual player count (not always mod 4).
- **Batching strategy**: bucket-by-player-count. Training examples split into 5 buckets (one per player count present in the dataset). DataLoader samples across buckets, batches within bucket — no padding overhead, clean variable-length tensors. Slightly more complex sampler logic; standard pattern for variable-length training.
- **Sampler balance**: weight buckets to give roughly equal training time per player count, not proportional to data volume. Otherwise 4-player games (the bulk of the human dataset) dominate and the model under-fits the rarer counts.

### `FactoredActionHelper`: new helper class alongside the existing one

The existing `ActionHelper` (Python) has significant test coverage and is used by the game engine validation path. Modifying its semantics would either break those tests or require maintaining backwards-compat layers that fight the new factored design.

**Decision**: a new class, `FactoredActionHelper`, ships alongside the existing one.

- **`ActionHelper`** (existing, Python): unchanged. Continues to serve game engine tests and any code that wants the "every discrete legal action including price expansion" view.
- **`FactoredActionHelper`** (new, Python + Rust): factored categorical enumeration with price-range metadata. Used by the AlphaZero pipeline (pretraining replay, MCTS, action_mapper).

The action_mapper rewrite targets `FactoredActionHelper` output specifically — translates between its structured legal-action list and the network's structured policy outputs / training target indices.

### `FactoredActionHelper` output schema

The enumeration returns structured `LegalAction` objects with categorical descriptors plus price-range metadata for price-bearing types:

```python
@dataclass
class LegalAction:
    type: str                                  # "Pass", "LayTile", "Bid", "BuyTrain", ...
    entity: dict                               # e.g., {"corp": "PRR"}, {"private": "MH"}
    params: dict = field(default_factory=dict) # categorical params, e.g., {"hex": ..., "tile": ..., "rotation": ...}
    price_range: Optional[Tuple[int, int]] = None  # (min, max) inclusive, in $1 units, for price-bearing types
```

Examples:

```python
# Categorical (no price)
LegalAction(type="LayTile", params={"hex": "E14", "tile": "59", "rotation": 3})
LegalAction(type="Par", entity={"corp": "PRR"}, params={"par_price": 90})
LegalAction(type="SellShares", entity={"corp": "B&O"}, params={"count": 2})

# Price-bearing: continuous price within [min, max]
LegalAction(type="Bid", entity={"private": "MH"},
            price_range=(min_legal_bid_for_MH, bidder_cash))
LegalAction(type="BuyCompany", entity={"company": "C&A"},
            price_range=(company.min_price, min(company.max_price, buyer.cash)))
LegalAction(type="BuyTrain", entity={"source": "PRR", "train": "train_3"},
            price_range=(1, buyer.cash))

# Depot trains: fixed price (engine determines), so min == max
LegalAction(type="BuyTrain", entity={"source": "depot", "train_type": "3T"},
            price_range=(180, 180))
```

The price range comes from the engine — `min_bid(company)`, `company.min_price`/`max_price`, etc. The FactoredActionHelper just exposes them uniformly per entry.

The network's continuous price head consumes the range:
- **Network**: clip / truncate the predicted `Normal(μ, σ)` to `[price_min, price_max]` when computing probabilities.
- **MCTS PW**: sample from the truncated distribution for child price selection — sampled prices are guaranteed legal without rejection sampling.

### Pretraining and MCTS migrate to `FactoredActionHelper` (Rust)

Implementation phases:

1. **Python `FactoredActionHelper` reference**: clean-room implementation in Python, validated against the existing `ActionHelper` for categorical equivalence (modulo price dimension collapsing). Lives in `rl18xx/game/factored_action_helper.py`.
2. **Rust `FactoredActionHelper`**: each `step` in `engine-rs/src/rounds/*.rs` implements `get_factored_choices(&Game) -> Vec<LegalAction>`. Engine validates the categorical equivalence via parity tests against the Python reference (similar to existing `tests/test_rust_action_parity.py`).
3. **`RustGameAdapter.get_factored_choices(game)`** in `rust_adapter.py` exposes the Rust implementation through a Python-friendly interface.
4. **Migrate consumers**: pretraining (`convert_game_to_training_data`), MCTS (`MCTSNode.legal_action_indices`), action_mapper to consume `FactoredActionHelper` output. Drop the existing `get_all_choices_limited` usage.

Effort estimate: ~3-5 days. Python reference is a few hundred lines; Rust port is similar; parity tests are mechanical.

Pretraining replay still runs through `BaseGame.process_action` or `RustGameAdapter.process_action` — engines unchanged for the game-state side. Only the choice-enumeration side is replaced.

**Once landed, both pretraining and self-play run entirely on the Rust engine.** The Python engine + Python `ActionHelper` remain in the repo for replay tooling, game engine tests, and as a reference, but the AlphaZero training paths never touch them.

### Cleaning-side heuristics stay in Python

The fuzzy action matching (`check_action_in_action_helper`) and pass insertion/skip heuristics (`should_add_pass`, `should_skip_action`) live in `pretraining.py` and stay there. They're data-cleaning logic specific to human game replay (small action-format drift, redundant passes from Ruby's logging) — not engine logic. They consume the `FactoredActionHelper` output and use Python's flexibility for the fuzzy matching.

### Consistent player IDs in pretraining game creation

Currently `pretraining.py:332-337,697` always uses `{1: "Player 1", ..., 4: "Player 4"}` regardless of source JSON player identities. The `player_mapping` translates from JSON player IDs to these consistent slots.

For variable player count, the pattern stays the same — always `{1: "Player 1", ..., N: "Player N"}` for an N-player game:
- 2-player → `{1: "Player 1", 2: "Player 2"}`
- 3-player → `{1: "Player 1", 2: "Player 2", 3: "Player 3"}`
- 4-player → existing
- 5-player → `{1..5}`
- 6-player → `{1..6}`

This keeps the encoder's `player_id_to_idx` deterministic and prevents JSON-name leakage. `player_mapping` is constructed per game from the source player ordering to these consistent 1..N slots.

---

## GNN as legacy

All forward design work targets the Transformer model only. The GNN model stays in the repo for ablation comparison but receives no feature work. The known v1-specific bugs (FiLM applied after blocks, BatchNorm in the trunk) are not fixed unless the user explicitly wants to keep the GNN viable for direct comparison runs.
