# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Reinforcement learning agent for the board game **1830**, using AlphaZero-style training. The rules are a Python port of the Ruby implementation from [tobymao/18xx](https://github.com/tobymao/18xx), with a **Rust reimplementation** (`engine-rs/`) used as a performance-accelerated drop-in for the training hot loop. Only the 1830 title is implemented; see `docs/multi_title_roadmap.md` for the plan to add 1867 and 1822/1822CA.

There are **two engines that must stay at parity**:
- **Python engine** (`rl18xx/game/engine/`) — the reference/oracle. Readable, faithful to the Ruby source, used for correctness checks.
- **Rust engine** (`engine-rs/`, imported as the `engine_rs` module) — the fast path. Built by maturin during `uv sync`, bridged to the Python-facing API by `RustGameAdapter` (`rl18xx/rust_adapter.py`). Used by self-play, pretraining cleaning, encoding, and an optional Rust MCTS.

When changing rules logic, change it in **both** engines (or update the adapter) and re-run the parity tests/audits in `docs/` — divergence silently corrupts training data.

## Commands

```bash
# Package management (uses uv, not pip). The build backend is maturin:
# `uv sync` compiles engine-rs and installs the `engine_rs` extension module.
uv sync                          # Install deps AND (re)build the Rust engine
uv add <package>                 # Add a dependency

# Rebuild the Rust engine after editing engine-rs/src/*.rs
uv sync                          # uv invalidates its cache on .rs changes (see [tool.uv] cache-keys)
# or, from engine-rs/:
cargo build --release            # type-check / iterate without reinstalling the wheel

# Run tests
uv run pytest tests/             # All tests
uv run pytest tests/agent/alphazero/model_test.py             # Single file
uv run pytest tests/agent/alphazero/model_test.py::test_run_single  # Single test
uv run pytest -m benchmark tests/agent/alphazero/bench_mcts_game.py  # Benchmarks (skipped by default)

# Formatting (line length 120)
uv run black --line-length 120 <file>

# Entry points (all via main.py)
uv run python main.py train              # AlphaZero training loop (self-play + training)
uv run python main.py pretrain           # Pre-train from human game data
uv run python main.py arena              # Run agent vs agent matches
uv run python main.py dashboard          # Start training dashboard (port 5001)
uv run python main.py replay <log_file>  # Replay a game in the browser

# Services (via startup.sh)
./startup.sh                     # Starts TensorBoard (:6006) + Dashboard (:5001)
```

## Architecture

### Game Engine — Python reference (`rl18xx/game/engine/`)

Models 1830's full rules. Key concepts:

- **BaseGame** (`game/base.py`): Central game class. Holds all state, processes actions via `game.process_action(action)`. Clone efficiently with `game.pickle_clone()`.
- **Rounds & Steps** (`round.py`): Turn structure — Auction, Stock, and Operating rounds. Each step defines which actions are legal and how to process them. Access via `game.round` / `game.active_step()`.
- **Entities** (`entities.py`): Players, Corporations, Companies, Bank, SharePool, StockMarket, Train depot.
- **Graph** (`graph.py`): Hex map with tiles, nodes, edges, paths. Used for route calculation.
- **Actions** (`actions.py`): All action types (Bid, Par, BuyShares, SellShares, LayTile, RunRoutes, BuyTrain, etc.).
- **ActionHelper** (`game/action_helper.py`): Enumerates all legal actions at any game state via `get_all_choices(game)`. `factored_action_helper.py` is the factored variant.
- **Game title data** lives in `game/engine/game/title/g1830.py`.

Several engine files are very large (100K+ bytes) — these are faithful ports from Ruby, not generated code.

### Game Engine — Rust accelerator (`engine-rs/`, module `engine_rs`)

A PyO3 crate that re-implements the engine for speed (it recently reached full 1830 parity with the Python engine). Mirrors the Python layout: `game.rs` (`BaseGame`), `rounds/{auction,stock,operating}.rs`, `entities.rs`, `core.rs`, `graph.rs`, `tiles.rs`, `map.rs`, `router.rs`, `actions.rs`. Title data lives in `src/title/g1830.rs` (parallel to the Python `g1830.py`). It also exposes `RustMCTSPlayer` (`mcts.rs`) and the action-index layout (`action_index.rs`, `POLICY_SIZE = 26537`).

- `RustGameAdapter` (`rl18xx/rust_adapter.py`) wraps a Rust `BaseGame` so the encoder, `ActionHelper`, and MCTS can use it wherever the Python `BaseGame` is expected — it bridges naming differences and synthesizes proxy objects that pass the Python engine's `isinstance` checks.
- Currently 1830-specific: there is no title dispatch (`title/mod.rs` is one line), and the action space / encoder constants are hardcoded to 1830.

### AlphaZero Agent (`rl18xx/agent/alphazero/`)

- **Models**: two architectures, **v2 is the default**.
  - `model_transformer.py` (**v2**): hex-map attention + entity attention + cross-modal fusion + FiLM phase conditioning; factored policy head over the ~26,537-action space (26,535 base actions + 2 D-train depot slots) plus per-player value, with auxiliary losses.
  - `model.py` (**v1**): GNN using GATv2Conv (PyTorch Geometric). Superseded by v2.
- **Encoder** (`encoder.py`): Converts game state into tensors (node features, edge index, game state vector).
- **Action Mapper** (`action_mapper.py`): Bidirectional mapping between action indices and game `Action` objects. Must agree with the Rust `action_index.rs` layout.
- **MCTS** (`mcts.py`): Python MCTS with configurable c_puct, Dirichlet noise, parallel readouts. An optional Rust tree (`rust_mcts_player.py` → `engine_rs.RustMCTSPlayer`) is used when `config.use_rust_mcts` is set.
- **Self-Play** (`self_play.py`): Generates training games (defaults to the Rust engine via `RustGameAdapter`). Writes examples to `training_examples/` (selfplay + holdout split).
- **Training** (`train.py`) / **Dataset** (`dataset.py`): Network training from LMDB-stored examples.
- **Loop** (`loop.py`): Orchestrates self-play → train → gate iterations. Configured via `loop_config.json` (hot-reloaded), status in `loop_status.json`.
- **Config** (`config.py`): `ModelConfig`, `TrainingConfig`, `SelfPlayConfig` dataclasses.
- **Checkpointer** (`checkpointer.py`): Model save/load with versioned directories under `model_checkpoints/`.
- **Pretraining** (`pretraining.py`): Supervised pre-training from human game data (JSON exports from 18xx.games; cleaning uses the Rust engine).
- **Inference server** (`inference_server.py`), **Metrics** (`metrics.py`).

### Client (`rl18xx/client/`)

Integration with the online 18xx.games platform:
- `ruby_backend_api_client.py`: API client for the Ruby backend.
- `game_sync.py`: Synchronizes local game state with an online game.
- `replay_game_from_log_file.py`: Replays games from JSON action logs in the browser.

### Agent Interface (`rl18xx/agent/agent.py`)

Abstract base class with: `initialize_game`, `get_game_state`, `suggest_move`, `play_move`.

### Arena (`rl18xx/agent/arena.py`)

Runs matches between agents (MCTS or random). Can optionally sync to 18xx.games for browser visualization.

## Key Patterns

- Game state is always accessed through `BaseGame` (Python) or `RustGameAdapter` (Rust) — never construct engine objects directly.
- Legal moves come from `ActionHelper.get_all_choices(game)`, applied via `game.process_action(action)`.
- The encoder/action_mapper bridge between the engine's object model and the neural network's tensor representation. The action layout is duplicated in Python (`action_mapper.py`) and Rust (`action_index.rs`) and must stay in sync.
- **Parity is load-bearing**: the Rust engine and the `RustGameAdapter` must reproduce Python's behavior exactly (down to which games the cleaning pipeline drops). Parity audits and bug logs live in `docs/` (`rust_engine_*_audit.*`, `cleaning_engine_parity.*`, `rust_engine_bugs.md`).
- `pickle_clone()` (Python) is used for fast state cloning in MCTS; it strips log/action history, graph caches, hex neighbor links, and the tile catalog before pickling, then restores shared/rebuilt data on the clone.
- Training data is stored in LMDB databases compressed with LZ4.
- CUDA is used when available, with automatic CPU fallback.

## Tech Stack

- Python 3.11, managed with `uv`
- Rust (PyO3 + maturin) for the accelerated engine — build backend is maturin (`[tool.maturin]` in `pyproject.toml`, manifest `engine-rs/Cargo.toml`)
- PyTorch 2.6 + PyTorch Geometric 2.6
- Flask + Gunicorn for dashboard
- TensorBoard for training metrics
- LMDB + LZ4 for training data storage
