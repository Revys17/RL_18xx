# Autoresearch Program — RL 18xx

You are an autonomous ML research agent. Your job is to improve a neural network that plays the board game 1830 by modifying its architecture, encoder, and training configuration, then measuring whether each change improves performance on a fixed evaluation corpus of human game positions.

## How this works

You run experiments in a loop. Each experiment is one atomic change to the codebase. You commit the change, run the experiment, check whether the metric improved, and keep or revert. You do this forever until the human interrupts you.

## Setup

1. Create a new git branch: `autoresearch/<date>` (e.g., `autoresearch/mar30`)
2. Read this entire file
3. Read `autoresearch/results.tsv` to see what has already been tried
4. Read the current state of the mutable files listed below
5. Start the experiment loop

## The experiment loop

```
while true:
    1. Form a hypothesis about what change might improve policy_loss
    2. Edit one or more mutable files to implement the change
    3. git add <changed files> && git commit -m "experiment: <short description>"
    4. uv run python -m autoresearch.run_experiment > run.log 2>&1
    5. Extract metrics: grep "^combined_loss:\|^policy_loss:\|^top1_accuracy:\|^top5_accuracy:\|^value_loss:" run.log
    6. If run crashed or produced NaN, log status=crash in results.tsv
    7. If combined_loss decreased (improved): status=keep
    8. If combined_loss stayed same or increased: git reset --hard HEAD~1, status=discard
    9. Append to autoresearch/results.tsv: commit | combined_loss | policy_loss | top1_acc | top5_acc | value_loss | status | description
    10. Go to step 1
```

**Do NOT pause to ask the human if you should continue. The loop runs until the human interrupts you, period.**

## Mutable files (you may edit these)

- `rl18xx/agent/alphazero/model.py` — network architecture (GNN, residual blocks, heads)
- `rl18xx/agent/alphazero/train.py` — optimizer, LR, loss function, training dynamics
- `rl18xx/agent/alphazero/encoder.py` — game state feature engineering, node features, encoding scheme
- `rl18xx/agent/alphazero/config.py` — hyperparameters, model dimensions
- `rl18xx/agent/alphazero/dataset.py` — data loading, augmentation, preprocessing

## Locked files (you must NOT edit these)

- `autoresearch/evaluate.py` — the evaluation harness
- `autoresearch/run_experiment.py` — the experiment runner
- `autoresearch/build_eval_corpus.py` — corpus generation
- `autoresearch/eval_corpus/` — evaluation and training game ID lists
- `autoresearch/program.md` — this file
- `rl18xx/game/engine/` — the entire game engine directory
- `rl18xx/agent/alphazero/action_mapper.py` — action space definition
- `rl18xx/agent/alphazero/mcts.py` — MCTS implementation
- `rl18xx/agent/alphazero/self_play.py` — self-play pipeline
- `rl18xx/agent/alphazero/pretraining.py` — human game processing

## Primary metric

**`combined_loss`** = `policy_loss` + `value_loss`. Lower is better.

- `policy_loss` — cross-entropy between the model's predicted move probabilities and human move targets (smoothed one-hot: 0.97 on played move, 0.03 spread over legal alternatives)
- `value_loss` — MSE between predicted and actual game outcomes (per-player, 4-element vector)

Secondary metrics (logged but not used for keep/discard):
- `top1_accuracy` — does the model's top prediction match the human move?
- `top5_accuracy` — is the human move in the model's top 5?

## Constraints

1. **Policy size must remain 26,535.** The action space is fixed.
2. **Value size must remain 4.** There are always 4 players.
3. **The model must accept encoder output.** If you change the encoder, update `config.py` dimensions to match (e.g., `game_state_size`, `map_node_features`).
4. **Changes must not break tests.** Run `uv run pytest tests/agent/alphazero/model_test.py -x` as a quick sanity check if your change is structural.
5. **Simplicity criterion.** All else being equal, simpler is better. A small improvement that deletes code beats one that adds complexity. Do not add unnecessary abstractions.
6. **One hypothesis per experiment.** Don't bundle multiple unrelated changes. If an experiment fails, you want to know what caused it.

## Research directions to explore

### Encoder improvements (high potential)
- Add missing game state features: private company special abilities (DH teleport, CS free tile, MH exchange), number of operating rounds remaining, player turn order, game history
- Fix inconsistent normalization: tile rotation at node feature index 6 is raw 0-5 while everything else is normalized to [0,1]
- Add richer edge features beyond direction (tile connectivity strength, revenue flow)
- Experiment with different normalization schemes for cash, shares, prices

### Architecture changes
- Experiment with GNN depth (currently 3 layers), width, and attention heads
- Try different graph pooling strategies (attention pooling, set transformer, multi-head pooling)
- Adjust residual block count (currently 5) and width (currently 512)
- Experiment with the fusion strategy (currently concat + linear)
- Try different activation functions (GELU, SiLU)
- Consider adding layer normalization alongside or instead of batch normalization
- Experiment with skip connections between GNN and output heads

### Training dynamics
- Try different optimizers: SGD+momentum (what AlphaZero uses), AdamW, LAMB
- Explore learning rate schedules: cosine decay, warmup + decay, cyclical
- Experiment with loss weighting between policy and value heads
- Try label smoothing, temperature scaling on the policy targets
- Experiment with batch size effects
- Try gradient accumulation for effectively larger batches
- Experiment with dropout rate (currently 0.1 everywhere)

### Regularization and generalization
- Try different dropout rates per component (GNN vs trunk vs heads)
- Experiment with weight decay values
- Try mixup or cutmix on the input features
- Consider spectral normalization on the output heads

## What you know about the codebase

- The model is a dual-branch GNN: a flat MLP processes 377 game-state features, a GATv2Conv GNN processes hex map nodes (50 features each), they fuse into a shared trunk of 5 residual blocks (512-dim), and split into policy (26,535 softmax) and value (4 tanh) heads.
- The encoder extracts features from a live BaseGame object. It produces: game_state_tensor (1, 377), node_features_tensor (NUM_HEXES, 50), edge_index (2, E), edge_attributes (E,).
- Training data comes from 200 human 1830 games (~128K positions). Evaluation uses 50 separate held-out games (~31K positions).
- The game is 1830 — a 4-player economic railroad board game with stock trading, route building, and train operations.

## Results format

`autoresearch/results.tsv` is tab-separated:

```
commit	policy_loss	top1_acc	top5_acc	value_loss	status	description
```

Status is one of: `keep`, `discard`, `crash`.
Crashed runs get `0.000000` for all numeric metrics.
