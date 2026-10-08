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
uv run python main.py convert --data-dir human_games/1830_clean_all --output human_games/lmdb_v6k  # Cleaned games -> LMDB
uv run python main.py pretrain --fresh --data-dir human_games/lmdb_v6k  # Fresh model on the pre-converted LMDB
uv run python scripts/build_start_positions.py   # Rebuild self-play's first-Stock-Round start file
uv run python scripts/build_midgame_positions.py  # Rebuild the mid-game (Stock Round 2-6) start file
uv run python scripts/eval_head_to_head.py --match 107 7 --games 240  # Strength: checkpoint vs checkpoint
uv run python scripts/eval_value_head.py # Value-head quality by game stage (current_best by default)
uv run python main.py policy-selfplay --output training_examples/value_v2 --games 400000  # Value-net data
uv run python main.py policy-gradient --policy <session>/<num> --value <session>/<num>  # RL-policy stage
uv run python main.py arena              # Run agent vs agent matches
uv run python main.py dashboard          # Start training dashboard (port 5001); http://localhost:5001/games browses saved games
uv run python main.py replay <game_file> # Check a saved game in the Python engine, print its viewer URL (--log: the game log)
uv run python main.py advisor            # Model advisor backend (127.0.0.1:5002) for the browser extension in extension/
uv run python scripts/eval_policy_only.py --match A B --out logs/eval/<name> --save-games 60  # + keep 60 games/match to view
uv run python main.py policy-gradient ... --save-game-every 1000  # + keep ~1 in 1000 training games to view

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
- **Loop** (`loop.py`): Orchestrates self-play → train → gate iterations. Configured via `loop_config.json` (hot-reloaded), status in `loop_status.json`. `selfplay_overrides` (a dict of `SelfPlayHyperparams` fields, e.g. search settings) applies to every self-play game; `eval_every` runs a background head-to-head of the current best against the run's starting checkpoint (`scripts/eval_head_to_head.py`; `Eval/Score_vs_Start`, `logs/loop/eval_history.jsonl`; 0.5 = no stronger). Gating is usually off (`--no-gate`), so this eval is how a run's strength is tracked.
- **Config** (`config.py`): `ModelConfig`, `TrainingConfig`, `SelfPlayConfig` dataclasses.
- **Checkpointer** (`checkpointer.py`): Model save/load with versioned directories under `model_checkpoints/`.
- **Pretraining** (`pretraining.py`): Supervised pre-training from human game data (JSON exports from 18xx.games; cleaning uses the Rust engine). Two-stage by default: a joint run with a small value-loss weight, early-stopped on validation loss, then the value heads are re-fit on the frozen best checkpoint (`_refit_value_heads`) because on ~2,900 games the value head memorizes after under an epoch while the policy keeps improving. Logs value winner accuracy vs. the equal-odds baseline each epoch. `main.py convert` keeps forced positions (exactly one legal action index) by default — dropping them (`--skip-forced`, as self-play does) made pretraining clearly worse (lmdb_v5/lmdb_v6 skip them; lmdb_v6k keeps them). Each human action must match exactly one policy index (`_matching_choices`: token cities resolved through the board, train purchases by the seller, CS/DH abilities as the private); ambiguous or unmatched actions are skipped and counted, never guessed. lmdb_v5 and earlier took the first match, so ~2% of their labels (most `place_token`, some `buy_train`) were wrong. The train/validation split is a hash of the game id (`in_validation_split`), stable across conversions from lmdb_v6 on.
- **AlphaGo-style stages** (the current direction): the supervised policy, then -- with no search -- (a) `main.py policy-selfplay` (`policy_selfplay.py`) plays fast policy-only games (~60k games/hour on 48 workers) and keeps a few positions per game as value-network data (value nets live in `model_checkpoints_value/`, trained with `main.py pretrain --model-dir model_checkpoints_value --policy-loss-weight 0 ...`); (b) `main.py policy-gradient` (`policy_gradient.py`) refines the policy by PPO-clipped policy gradient against the supervised policy and a pool of its own snapshots, with a KL penalty to the supervised policy and a separate critic (a value net, trained on as it goes); checkpoints in `model_checkpoints_pg/<run>/`, progress = `score_vs_sl` in `history.jsonl` / TensorBoard `PG/*`. `composite_model.PolicyValueComposite` searches or serves with one network's policy and another's value (`eval_head_to_head.py` player `<policy>+<value>`); `run_values_encoded` / `forward(value_only=True)` skip the policy head (~2/3 of a forward). A much better value net (2026-10-06: v2, 400k policy games) did not make 64-readout search stronger, which is why the policy-gradient stage comes next.
- **Inference server** (`inference_server.py`), **Metrics** (`metrics.py`).
- **Saved games / game viewer** (`game_records.py`, `rl18xx/agent/dashboard/game_viewer.py`): a *collection* is a directory with `games/<name>.json` (one game's action log) and a `games.jsonl` index. `eval_head_to_head.py` saves every game (bare action lists; seats from `settings.json`); `eval_policy_only.py --save-games N` (first N games per match) and `policy-gradient --save-game-every N` (~1 in N, into `model_checkpoints_pg/<run>/games/`) write `make_game_record` files: an 18xx.games-style export with the seats' labels, update, opponent, start and result under `"rl18xx"`. All off by default. The dashboard's `/games` page lists collections under `logs/eval`, `model_checkpoints_pg` and `logs/games` and steps through a game (map, players, corporations, market, log; arrow keys) by replaying it in the Python engine (~0.2 s per full game, cached per worker). `main.py replay <file>` copies files from elsewhere (e.g. a self-play log's `Game actions:` line) into `logs/games/replays/`.

### Client (`rl18xx/client/`)

Integration with a self-hosted 18xx.games server (the Ruby backend, `http://localhost:9292` with accounts a/b/c/z -- never the public site):
- `ruby_backend_api_client.py`: API client for the Ruby backend.
- `game_sync.py`: Mirrors a local game onto that server (`arena --browser`, `main.py replay --local-server`).
- `replay_game_from_log_file.py`: `main.py replay`: replays a saved game locally and points at the dashboard's game viewer (no server needed).

### Advisor (`rl18xx/agent/advisor/`, `extension/`)

Model help while playing real games on 18xx.games: a browser extension (`extension/`, Manifest V3, Chrome + Firefox; install steps in `extension/README.md`) floats a panel over the site's game page, reads the game from the site (`/api/game/<id>`, same origin, at most every 10 s) and asks the local backend `main.py advisor` (Flask on 127.0.0.1, answers extension origins only, never contacts 18xx.games). The backend follows each game incrementally in the Rust engine (`live_game.py`: `pretraining.filter_actions` + `replay_human_action`, the human importer with its training-data drop rules off; an undo rebuilds) and answers win estimates (value net), the policy's top-5 moves with prices (the supervised policy during the auction), the model's probability for the last moves, and an optional Rust-MCTS search (`advisor.py`, `describe.py`). Read-only: the user makes the moves on the site. Test game: `tests/fixtures/advisor/game_1830_4p.json` (`scripts/make_advisor_fixture.py`, no human data).

### Agent Interface (`rl18xx/agent/agent.py`)

Abstract base class with: `initialize_game`, `get_game_state`, `suggest_move`, `play_move`.

### Arena (`rl18xx/agent/arena.py`)

Runs matches between agents (MCTS or random). Can optionally sync to 18xx.games for browser visualization.

## Key Patterns

- Game state is always accessed through `BaseGame` (Python) or `RustGameAdapter` (Rust) — never construct engine objects directly.
- Legal moves come from `ActionHelper.get_all_choices(game)`, applied via `game.process_action(action)`.
- The encoder/action_mapper bridge between the engine's object model and the neural network's tensor representation. The action layout is duplicated in Python (`action_mapper.py`) and Rust (`action_index.rs`) and must stay in sync.
- **Optional rules**: engines are built with a recorded game's `settings.optional_rules` filtered to the ones BOTH implement (`pretraining.ENGINE_OPTIONAL_RULES`; today 1830's `optional_6_train`). The Rust constructor raises ValueError for any rule its title doesn't declare (`GameTitle::optional_rules`). Games with other optional rules replay under the base rules and are dropped if the engine rejects an action.
- **Self-play starts at the first Stock Round** (`start_positions.py`, `SelfPlayHyperparams.start_positions_path`): 80% from the real post-auction position of a 4-player human game (`human_games/start_positions_1830_4p.jsonl`, built by `scripts/build_start_positions.py`), 20% (`random_start_fraction`) from a random auction ending reached through legal actions (each private to a random player at a multiple of $5 in [face, 2x face]; the SV, which can't be bid on, goes at face or -- in a quarter of them -- $5-$20 below after all-pass rounds, free at $0; random B&O par). Training examples cover only the self-play decisions. The waterfall auction itself is to be trained separately. Policy-gradient runs can also start a share of games mid-game (`--midgame-fraction`): a human game cut at the start of Stock Round 2-6 (`human_games/midgame_positions_1830_4p.jsonl`, built by `scripts/build_midgame_positions.py`), because self-play from SR1 settled into slow games that rarely reach trainless companies, lethal dumps or bankruptcies (2026-10-07: agents' dumps hit a trainless company 10% of the time vs humans' 38%).
- **Rule variant `auction_unlock`** (both engines, off by default; `set_auction_unlock` in Rust, `BaseGame.auction_unlock` in Python): an all-pass waterfall-auction round discounts the next private by $5 like the SV when nobody has bid on it. Real 1830 only discounts the SV, so an auction where everyone keeps passing never ends — self-play kept hitting that; humans pass with a non-SV private next in ~5% of games. Gate games and self-play games that start at the auction turn it on (`SelfPlayHyperparams.auction_unlock`); start-position games, human import and arena don't. It is training wheels: the loop logs how often it fires (`SelfPlay/Auction_Unlock_Game_Rate`) so it can be turned off once self-play stops locking the auction.
- **Game limits count decisions**: `max_game_length`, `resign_min_move`, `softpick_move_cutoff` and `auction_stall_moves` count moves where the player had more than one legal action (`MCTSPlayer` / `RustMCTSPlayer.decisions`). Forced actions are applied automatically (collapsed into the preceding search edge) and are never training examples. The search trees only bound engine actions as a safety net (`max_engine_actions`, both engines).
- **Value-head memorization**: every position of a game shares one outcome, so the value head memorizes games quickly (train-window CE far below new-game CE). Self-play training uses `TrainingConfig.value_stop_grad` (value heads on detached trunk features) with `value_lr_multiplier` 0.3; the loop logs `Value/CE_New_Games` vs `Value/CE_Trained_Window` each iteration. `scripts/value_overfit_replay.py` replays a finished run's training under other settings to compare them offline. A resigned game's value target names the player the search was confident in (`self_play.resigned_result`), not the net-worth leader at the resign point.
- **MCTS selection Q**: both engines default to minigo's `W / (1 + N)` with unvisited children at 0. Values here are 0-1 win shares (0 = certain loss), so that reads unexplored moves as losses and, with c_puct ~2-3 against sibling value gaps of ~0.02, the search mostly follows the prior. `SelfPlayHyperparams.mcts_mean_q` (Rust only) switches to mean Q with first-play urgency at the parent's value (`fpu_reduction`). Note `c_puct_by_round` overrides `c_puct_init` in every 1830 round. Search tests on the pretrained model (2026-10-04): with a net-worth leaf heuristic (`leaf_value_heuristic`), 200 readouts beat 8 (0.60), so the tree works (but until 2026-10-06 evals gave every position with <= 5 legal moves `min_readouts` = 50 whatever the spec, so that was ~119 vs ~31 readouts a decision; specs now apply everywhere); with the network's value, a more value-driven search plays worse (0.33) -- the value head can't yet rank sibling moves. Self-play keeps the default search until it can; re-test (`scripts/eval_head_to_head.py`, `:mean`, `par=`, `value=networth` player options) on later checkpoints.
- **Numerics**: training is bf16, inference fp32, and the first non-finite loss or gradient stops training (and pretraining) without saving; `Numerics/*` logs per-block activation peaks and weight norms.
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
