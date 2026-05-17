# 18XX Reinforcement Learning Agent

An AlphaZero-style agent for the board game **1830: Railways & Robber Barons**. The game engine is a Python port of the Ruby implementation from [tobymao/18xx](https://github.com/tobymao/18xx), with an optional Rust-accelerated engine via PyO3. Training combines self-play, MCTS, and a dual-head (policy + value) neural network.

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/). A Rust toolchain is needed to build the optional Rust engine.

```bash
uv sync          # Install dependencies and build the Rust engine
```

## CLI

All commands run through `main.py`:

```bash
uv run python main.py <command> [options]
```

Pass `--help` to any command for the full flag list.

### `train`

Iterative self-play + network training, with optional gating against the current best.

```bash
uv run python main.py train [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--iterations` | `0` | Loop iterations (`0` = indefinite) |
| `--target-experiences` | `10000` | Target examples per iteration |
| `--threads` | `2` | Parallel self-play workers |
| `--readouts` | `64` | MCTS simulations per move |
| `--max-training-window` | `100000` | Max examples per train (`0` = all) |
| `--gate-games` | `10` | Arena games for gating |
| `--gate-threshold` | `0.55` | Win rate to promote |
| `--no-gate` | off | Skip gating |
| `--model-type` | `v2` | Initial architecture (`v1`/`v2`) |
| `--fresh` | off | Wipe checkpoints and data |
| `--keep-old-files` | off | Preserve previous-run data |

Loop flow per iteration: self-play → train → gate → promote (if win rate ≥ threshold).

Settings are also read from `loop_config.json` (CLI overrides). The file is hot-reloaded between iterations.

### `pretrain`

Bootstraps the model from human game data before self-play.

```bash
uv run python main.py pretrain [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--data-dir` | `human_games` | Game JSON files or LMDB path |
| `--model-dir` | `model_checkpoints` | Output checkpoint directory |
| `--epochs` | `10` | Training epochs |
| `--batch-size` | `256` | Batch size |

Game files are JSON exports from [18xx.games](https://18xx.games); only finished 4-player games are used.

### `arena`

4-player match between any mix of agents.

```bash
uv run python main.py arena [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--agents` | `mcts mcts mcts mcts` | 4 agent types: `mcts` or `random` |
| `--model-dir` | none | Checkpoint directory to load |
| `--readouts` | `200` | MCTS simulations per move |
| `--browser` | off | Sync to 18xx.games for visualization |

### `dashboard`

Web UI for monitoring training (http://localhost:5001).

```bash
uv run python main.py dashboard [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--host` | `0.0.0.0` | Server host |
| `--port` | `5001` | Server port |
| `--debug` | off | Flask debug mode |

### `replay`

```bash
uv run python main.py replay path/to/game.log
```

Replays a game log in the browser via 18xx.games.

## Dashboard

Shows current loop/phase, system metrics, loss curves (Chart.js), model lineage, and active self-play games. Training parameters (batch size, LR, epochs, weight decay, loss weights, gradient accumulation, training window) are editable inline and hot-reloaded.

API endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/loop_config` | GET/POST | Read or update loop configuration |
| `/api/current_status` | GET | Current training loop status |
| `/api/games_status` | GET | Active self-play games |
| `/api/metrics_history` | GET | Training metrics (JSONL). `?last=N` for recent entries |
| `/api/model_history` | GET | Model gating decisions |
| `/api/system_metrics` | GET | CPU and memory usage |

## Services

`startup.sh` runs both monitoring services in the background:

| Service | Port | Source |
|---------|------|--------|
| TensorBoard | 6006 | `runs/alphazero_runs/` |
| Dashboard | 5001 | Flask + Gunicorn (4 workers) |

```bash
./startup.sh      # Start both
```

## Configuration

`loop_config.json` is read on every iteration (hot-reload). CLI flags override these values.

```json
{
    "num_loop_iterations": 0,
    "num_games_per_iteration": 50,
    "num_threads": 8,
    "num_readouts": 200,
    "target_experiences": 50000,
    "training_config": {
        "batch_size": 256,
        "lr": 0.001,
        "num_epochs": 3,
        "weight_decay": 0.0001,
        "shuffle_examples": true,
        "value_loss_weight": 1.0,
        "entropy_weight": 0.01,
        "gradient_accumulation_steps": 1,
        "max_training_window": 0
    }
}
```

## Model Architecture

Two model versions are available; **v2 is the default**.

**v2 (Transformer, ~500K params):**
- 4-layer hex-map attention over 93 hex positions with positional encoding
- 2-layer attention over economic entities (players, corporations, privates, global state)
- Cross-modal fusion: economic entities attend to map nodes
- FiLM phase conditioning in the trunk
- Factored policy head over the 26,535-action space
- Auxiliary phase and action-count prediction losses

**v1 (GNN, ~25.6M params):** GATv2Conv over the hex map graph. Superseded by v2.

## Training Metrics

| Source | Path | Contents |
|--------|------|----------|
| Per-iteration | `logs/loop/metrics_history.jsonl` | Loss components, policy/value diagnostics, gradient norms |
| Gating | `logs/loop/model_history.jsonl` | Win rates, promotion decisions |
| Per-game | `self_play_games_status/L{loop}_G{game}.json` | Move count, timing, state |
| TensorBoard | `runs/alphazero_runs/` | Full training curves |

## Game Engine

```python
from rl18xx.game.gamemap import GameMap
from rl18xx.game.action_helper import ActionHelper

game = GameMap().game_by_title("1830")({1: "P1", 2: "P2", 3: "P3", 4: "P4"})
action_helper = ActionHelper()

actions = action_helper.get_all_choices(game)
game.process_action(actions[0])
```

See [game_usage.md](game_usage.md) for engine documentation.

### Rust Engine (optional)

A PyO3 Rust implementation of the engine is built by `uv sync` (requires a Rust toolchain) and used for route calculation and state encoding. The Python agent bridges to it via `RustGameAdapter`.

## Tests

```bash
uv run pytest tests/
uv run pytest tests/agent/alphazero/model_test.py
uv run pytest tests/agent/alphazero/model_test.py::test_run_single
```

## Project Structure

```
rl18xx/
  game/engine/          Game engine (rules, entities, graph, actions, rounds)
  agent/alphazero/      AlphaZero agent
    loop.py             Training loop orchestration
    model.py            v1 GNN model
    model_v2.py         v2 Transformer model
    mcts.py             Monte Carlo Tree Search
    self_play.py        Self-play game generation
    train.py            Network training
    encoder.py          Game state -> tensor encoding
    action_mapper.py    Action index <-> Action object mapping
    checkpointer.py     Checkpoint management
    config.py           Configuration dataclasses
    pretraining.py      Human game data pretraining
  agent/arena.py        Agent vs agent evaluation
  agent/dashboard/      Flask training dashboard
  client/               18xx.games integration
  rust_adapter.py       Rust engine bridge (PyO3)
engine-rs/              Rust game engine
tests/
main.py
loop_config.json
startup.sh
docs/
```

## Tech Stack

Python 3.11+ · Rust (optional engine) · PyTorch 2.6 · PyTorch Geometric 2.6 · Flask + Gunicorn · Chart.js · TensorBoard · LMDB + LZ4 · uv · Maturin
