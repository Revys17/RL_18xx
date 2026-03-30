# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Reinforcement learning agent for the board game **1830**, using AlphaZero-style training. The game engine is a Python port of the Ruby implementation from [tobymao/18xx](https://github.com/tobymao/18xx). Currently only the 1830 title is implemented.

## Commands

```bash
# Package management (uses uv, not pip)
uv sync                          # Install dependencies
uv add <package>                 # Add a dependency

# Run tests
uv run pytest tests/             # All tests
uv run pytest tests/agent/alphazero/model_test.py  # Single file
uv run pytest tests/agent/alphazero/model_test.py::test_run_single  # Single test

# Formatting (line length 120)
uv run black --line-length 120 <file>

# Training loop
uv run python -m rl18xx.agent.alphazero.loop   # Main training loop (reads loop_config.json)

# Services (started via startup.sh)
# TensorBoard on :6006 (logs in runs/alphazero_runs/)
# Dashboard on :5001 (Flask app at rl18xx/agent/dashboard/dashboard.py)
```

## Architecture

### Game Engine (`rl18xx/game/engine/`)

The engine models 1830's full rules. Key concepts:

- **BaseGame** (`game/base.py`): Central game class. Holds all state, processes actions via `game.process_action(action)`. Clone with `game.clone(game.raw_actions)`.
- **Rounds & Steps** (`round.py`): Turn structure — Auction, Stock, and Operating rounds. Each step defines which actions are legal and how to process them. Access via `game.round` / `game.active_step()`.
- **Entities** (`entities.py`): Players, Corporations, Companies, Bank, SharePool, StockMarket, Train depot.
- **Graph** (`graph.py`): Hex map with tiles, nodes, edges, paths. Used for route calculation.
- **Actions** (`actions.py`): All action types (Bid, Par, BuyShares, SellShares, LayTile, RunRoutes, BuyTrain, etc.).
- **ActionHelper** (`game/action_helper.py`): Enumerates all legal actions at any game state via `get_all_choices(game)`.
- **Game title data** lives in `game/engine/game/title/g1830.py`.

Many engine files are very large (100K+ lines) — these are faithful ports from Ruby, not generated code.

### AlphaZero Agent (`rl18xx/agent/alphazero/`)

- **Model** (`model.py`): GNN-based architecture using GATv2Conv (PyTorch Geometric). Dual-head: policy (26,535 actions) + value (per-player). Shared trunk with residual blocks.
- **Encoder** (`encoder.py`): Converts `BaseGame` state into graph tensors (node features, edge index, game state vector).
- **Action Mapper** (`action_mapper.py`): Bidirectional mapping between action indices and game `Action` objects.
- **MCTS** (`mcts.py`): Monte Carlo Tree Search with configurable c_puct, Dirichlet noise, parallel readouts.
- **Self-Play** (`self_play.py`): Generates training games. Writes examples to `training_examples/` (selfplay + holdout split).
- **Training** (`train.py`): Network training from LMDB-stored examples.
- **Loop** (`loop.py`): Orchestrates self-play → train iterations. Configured via `loop_config.json`, status in `loop_status.json`.
- **Config** (`config.py`): `ModelConfig`, `TrainingConfig`, `SelfPlayConfig` dataclasses.
- **Checkpointer** (`checkpointer.py`): Model save/load with versioned directories under `model_checkpoints/`.
- **Pretraining** (`pretraining.py`): Pre-training from human game data.

### Client (`rl18xx/client/`)

Integration with the online 18xx.games platform:
- `ruby_backend_api_client.py`: API client for the Ruby backend.
- `game_sync.py`: Synchronizes local game state with an online game.
- `replay_game_from_log_file.py`: Replays games from JSON action logs in the browser.

### Agent Interface (`rl18xx/agent/agent.py`)

Abstract base class with: `initialize_game`, `get_game_state`, `suggest_move`, `play_move`.

## Key Patterns

- Game state is always accessed through `BaseGame` — never construct engine objects directly.
- Legal moves come from `ActionHelper.get_all_choices(game)`, applied via `game.process_action(action)`.
- The encoder/action_mapper bridge between the game engine's object model and the neural network's tensor representation.
- Training data is stored in LMDB databases compressed with LZ4.
- CUDA is used when available, with automatic CPU fallback.

## Tech Stack

- Python 3.11, managed with `uv`
- PyTorch 2.6 + PyTorch Geometric 2.6
- Flask + Gunicorn for dashboard
- TensorBoard for training metrics
- LMDB + LZ4 for training data storage
