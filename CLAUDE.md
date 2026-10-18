# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Reinforcement learning agent for the board game **1830**, using AlphaZero-style training. The rules are a Python port of the Ruby implementation from [tobymao/18xx](https://github.com/tobymao/18xx), with a **Rust reimplementation** (`engine-rs/`) used as a performance-accelerated drop-in for the training hot loop. Only 1830 is trained. The Rust engine also implements **1867** and **1822** (Rust only, validated against Ruby: `docs/g1867_remaining_work.md`, `docs/g1822_remaining_work.md`); their RL bridge is designed but on hold (`docs/rl_bridge_design.md`, `docs/multi_title_roadmap.md`). Every title targets 3-6 players.

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
uv run python scripts/eval_head_to_head.py --match A B --games-per-task 8 ...  # + several games per worker, searched in lockstep
uv run python scripts/bench_search_harness.py --model stub --mode server --concurrency 1 16  # Search-harness evaluations/s (CPU)
uv run python scripts/eval_value_head.py # Value-head quality by game stage (current_best by default)
uv run python main.py policy-selfplay --output training_examples/value_v2 --games 400000  # Value-net data
uv run python main.py policy-gradient --policy <session>/<num> --value <session>/<num>  # RL-policy stage
uv run python main.py search-learning --policy <run>/learner/<n>.pth --value <run>/critic/<n>.pth  # Policy toward search targets
    # + --reuse-tree total --dedupe-maps --server-replicas 2: ~7x the decisions/s (docs/search_learning_design.md "Speed")
uv run python scripts/bench_search_generation.py --configs base dedupe+total+rep2  # generation alone, configs back to back
uv run python scripts/bench_inference_server.py serve --requests <captured.npz>    # server throughput on recorded requests
uv run python main.py policy-gradient --resume model_checkpoints_pg/rab_place_20261014 --max-updates 2000 \
    --population model_checkpoints_pg/populations/league_v7.json  # continue gen4 against the league
uv run python scripts/search_decision_diag.py --player <policy.pth>+<value.pth> --variant b64=@64 \
    --variant fix800=@800:dedupe,par=8,ptemp=0.5 --out logs/search_diag/run1.jsonl  # how often search leaves the policy's move
uv run python scripts/search_override_rollouts.py --diag logs/search_diag/run1.jsonl --policy <policy.pth> \
    --variant fix800 --out logs/search_diag/run1_rollouts.jsonl  # ...and whether those moves are better
uv run python scripts/count_bankruptcies.py logs/eval/<out>  # bankruptcies in an eval's saved games
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
- **Lockstep search** (`lockstep.py`, `engine-rs/src/mcts_lockstep.rs`): one worker searches many Rust trees at once. `RustMCTSPlayer.start_search()` gives a `TreeSearch` (or None at a single legal move without an open price); each `LockstepSearcher.round()` has every tree select its next batch exactly as `tree_search` does, encodes all leaves natively into one stacked request per network (`BaseGame.encode_canonical_py` / `mcts._native_encode`, bit-for-bit `_rust_encode`: tests + `scripts/check_native_encode.py` on 146k human positions), and scatters the server's sparse legal priors, values and price rows back in Rust; `PolicyRequest`s (a policy player's move) ride in the same requests. A lockstep tree ends identical to the same tree under `suggest_move` (test_lockstep_search.py); the one difference is that an unexpanded root gets a one-leaf expansion round. `eval_head_to_head.py --games-per-task N` uses it; a future self-play generator should too. 2026-10-17, CPU, stub network: ~210 vs ~850 us of worker time per evaluation, and with an inference server and 8 workers 15.9k vs 4.1k evaluations/s (the old path's server spent 65% receiving 17-leaf requests). Search-tree nodes log only the actions since the root (`root_log`; games handed to Python get the whole log): cloning and freeing each node's full JSON action log was most of an expansion late in a game. What is left per evaluation is mostly the engine: cloning and freeing each node's `BaseGame` and its legal-move enumeration (`scripts/bench_leaf_costs.py`).
- **Self-Play** (`self_play.py`): Generates training games (defaults to the Rust engine via `RustGameAdapter`). Writes examples to `training_examples/` (selfplay + holdout split).
- **Training** (`train.py`) / **Dataset** (`dataset.py`): Network training from LMDB-stored examples.
- **Loop** (`loop.py`): Orchestrates self-play → train → gate iterations. Configured via `loop_config.json` (hot-reloaded), status in `loop_status.json`. `selfplay_overrides` (a dict of `SelfPlayHyperparams` fields, e.g. search settings) applies to every self-play game; `eval_every` runs a background head-to-head of the current best against the run's starting checkpoint (`scripts/eval_head_to_head.py`; `Eval/Score_vs_Start`, `logs/loop/eval_history.jsonl`; 0.5 = no stronger). Gating is usually off (`--no-gate`), so this eval is how a run's strength is tracked.
- **Config** (`config.py`): `ModelConfig`, `TrainingConfig`, `SelfPlayConfig` dataclasses.
- **Checkpointer** (`checkpointer.py`): Model save/load with versioned directories under `model_checkpoints/`.
- **Pretraining** (`pretraining.py`): Supervised pre-training from human game data (JSON exports from 18xx.games; cleaning uses the Rust engine). Two-stage by default: a joint run with a small value-loss weight, early-stopped on validation loss, then the value heads are re-fit on the frozen best checkpoint (`_refit_value_heads`) because on ~2,900 games the value head memorizes after under an epoch while the policy keeps improving. Logs value winner accuracy vs. the equal-odds baseline each epoch. `main.py convert` keeps forced positions (exactly one legal action index) by default — dropping them (`--skip-forced`, as self-play does) made pretraining clearly worse (lmdb_v5/lmdb_v6 skip them; lmdb_v6k keeps them). Each human action must match exactly one policy index (`_matching_choices`: token cities resolved through the board, train purchases by the seller, CS/DH abilities as the private); ambiguous or unmatched actions are skipped and counted, never guessed. lmdb_v5 and earlier took the first match, so ~2% of their labels (most `place_token`, some `buy_train`) were wrong. The train/validation split is a hash of the game id (`in_validation_split`), stable across conversions from lmdb_v6 on.
- **AlphaGo-style stages** (the current direction): the supervised policy, then -- with no search -- (a) `main.py policy-selfplay` (`policy_selfplay.py`) plays fast policy-only games (~60k games/hour on 48 workers) and keeps a few positions per game as value-network data (value nets live in `model_checkpoints_value/`, trained with `main.py pretrain --model-dir model_checkpoints_value --policy-loss-weight 0 ...`); (b) `main.py policy-gradient` (`policy_gradient.py`) refines the policy by PPO-clipped policy gradient against the supervised policy and a pool of its own snapshots, with a KL penalty to the supervised policy and a separate critic (a value net, trained on as it goes); checkpoints in `model_checkpoints_pg/<run>/`, progress = `score_vs_sl` in `history.jsonl` / TensorBoard `PG/*`. `composite_model.PolicyValueComposite` searches or serves with one network's policy and another's value (`eval_head_to_head.py` player `<policy>+<value>`); `run_values_encoded` / `forward(value_only=True)` skip the policy head (~2/3 of a forward). A much better value net (2026-10-06: v2, 400k policy games) did not make 64-readout search stronger, which is why the policy-gradient stage came next (that search result is void: see MCTS selection Q below).
- **Search learning** (`search_learning.py`, `main.py search-learning`; docs/search_learning_design.md steps 3-4, 2026-10-17, not yet run on the GPU): expert iteration. Each worker plays `--games-per-worker` games in lockstep (`Generator`, games carried across 30-s pool tasks); a decision with several legal moves is searched with probability `--p-full` (0.25) at `--n-full` (200) readouts with the validated value-driven settings and becomes a row (`SearchRow`: native encoding, legal moves, target, price-cell target, outcome filled in at the game's end); the other decisions are policy moves at T in the same lockstep rounds. The policy is softmax(logits / T), T = the starting run's `learner_temperature` (0.5 for gen4); loss = CE of log_softmax(logits / T) to the target over the legal moves + the price-cell CE + KL to a moving anchor at T (`--kl-target`, `--anchor-every`); the critic (the search's value) trains on 32 rows per game, each once, with `policy_gradient.critic_losses` (win CE + score head on placement points). Rows go into a replay window (`--window-rows`); each update waits for `--rows-per-update` new rows and takes `--steps-per-update` steps (reuse = steps * minibatch / rows per update; no importance ratios -- the targets don't depend on the behaviour policy). `--target visits` (default) uses KataGo's policy-target pruning (`pruned_child_n_at_root`, `--visit-target raw` for the whole distribution): on gen4 the raw visits put 12-19% of the target on moves its policy gives < 1% (first-play exploration), KL(target || policy) 1.6-2.4 nats against 0.2-0.35 pruned (`scripts/search_target_stats.py`); `--target gumbel` uses the completed-Q target. Run dir like policy-gradient's under `model_checkpoints_search/`, history `SL/*`. Generation per worker on the CPU with a free network (`scripts/bench_search_selfplay.py`): ~180 decisions/s at n_full 64, ~77 at 200 (~17.6 / ~52 evaluations per decision), so on the GPU the network is the limit.
- **Search speed-ups** (2026-10-18; docs/search_learning_design.md "Speed"): (1) the inference server's packed fast path (default, bit for bit the old replies): `PackedStates` go to the model as stacked numpy arrays (`model_transformer.packed_arrays`/`packed_inputs`/`run_packed`, one input set for policy + critic; legal indices pickled flat as `LegalIndices`). Keep torch CPU ops off hot paths in servers: torch's intra-op pool on the loaded machine made a 12 MB `torch.cat` take 190 ms (24 threads) vs 0.5 ms (2); run with `OMP_NUM_THREADS=2`. (2) `--dedupe-maps` (`LockstepSearcher(dedupe_maps=)`, `mcts_collect_leaves(..., dedupe_maps=True)` -> `map_index`): each distinct hex map of a request once; the map encoder (77% of a 600-state forward, its only input the map) runs per map: 2.3-10x less GPU work, outputs within float rounding. (3) `--server-replicas N` (`start_inference_server(replicas=)`, `ReplicatedServerHandle`; worker w -> replica w % N): one replica's forward overlaps another's receive/reply; the collector-thread pipeline (`InferenceServer(pipeline=True)`) is slower (GIL). (4) `--reuse-tree total` (KataGo maxVisits: a kept subtree's root visits count toward 1 + n; `additional` = suggest_move's / eval's count): trees advance along every move (`advance_root_to(move, price)`, `root_log_matches`), dropped on a network version change or where the tree priced an open-price forced move itself; ~87% of root visits inherited, evaluations per decision 90 -> 21. All four together: 68 -> 492 decisions/s with 10 workers (`scripts/bench_search_generation.py`); defaults stay off for (2)-(4).
- **League and goal exploiters** (policy-gradient, 2026-10): `--population <json>` (members under `model_checkpoints_pg/populations/`, `min_share` pins a member's minimum share) samples opponents by PFSP ((1-p)^2 on the learner's mean score against them), deals the learner 1-3 seats per game (`--learner-seat-weights`), and adds the learner's own snapshot every `--league-snapshot-every` 50 updates (at most 4 kept); a resume picks up new population members. `--reward` sets a goal: `win`, `placement` (1/0.5/0.2/0 by finishing place; `placement:1,0.3,0.1,0`), `bankrupt` (+0.5 when another player's bankruptcy ends the game), `president:<CORP>`, `rusher`, `rust` (+0.1 per opponent train its purchase rusts); with a shaped reward the critic's score head learns it and is the baseline (`<policy>+<critic>#score`). Goal exploiters start from ckpt 10 with 2 seats against a fixed main checkpoint and join the population. `--auction-fraction` starts that share of games at the private auction. The main line is **gen4 = `model_checkpoints_pg/rab_place_20261014/learner/1800.pth`** (placement reward, which won a reward A/B against win and a blend); the run history and the generational ladder are in `docs/training_runs_2026_10.md`, launch scripts in `scripts/launch/2026-10/`. Judge checkpoints by policy-only 2v2 **and** 1v3 evals (`eval_policy_only.py --a-seats 1`): table composition changes 4-player results (an exploiter at parity 2v2 can be at 0.16 alone).
- **Inference server** (`inference_server.py`), **Metrics** (`metrics.py`).
- **Saved games / game viewer** (`game_records.py`, `rl18xx/agent/dashboard/game_viewer.py`): a *collection* is a directory with `games/<name>.json` (one game's action log) and a `games.jsonl` index. `eval_head_to_head.py` saves every game (bare action lists; seats from `settings.json`); `eval_policy_only.py --save-games N` (first N games per match) and `policy-gradient --save-game-every N` (~1 in N, into `model_checkpoints_pg/<run>/games/`) write `make_game_record` files: an 18xx.games-style export with the seats' labels, update, opponent, start and result under `"rl18xx"`. All off by default. The dashboard's `/games` page lists collections under `logs/eval`, `model_checkpoints_pg` and `logs/games` and steps through a game (map, players, corporations, market, log; arrow keys) by replaying it in the Python engine (~0.2 s per full game, cached per worker). `main.py replay <file>` copies files from elsewhere (e.g. a self-play log's `Game actions:` line) into `logs/games/replays/`.

### Client (`rl18xx/client/`)

Integration with a self-hosted 18xx.games server (the Ruby backend, `http://localhost:9292` with accounts a/b/c/z -- never the public site):
- `ruby_backend_api_client.py`: API client for the Ruby backend.
- `game_sync.py`: Mirrors a local game onto that server (`arena --browser`, `main.py replay --local-server`).
- `replay_game_from_log_file.py`: `main.py replay`: replays a saved game locally and points at the dashboard's game viewer (no server needed).

### Advisor (`rl18xx/agent/advisor/`, `extension/`)

Model help while playing real games on 18xx.games: a browser extension (`extension/`, Manifest V3, Chrome + Firefox; install steps in `extension/README.md`) floats a panel over the site's game page, reads the game from the site (`/api/game/<id>`, same origin, at most every 10 s) and asks the local backend `main.py advisor` (Flask on 127.0.0.1, answers extension origins only, never contacts 18xx.games). The backend follows each game incrementally in the Rust engine (`live_game.py`: `pretraining.filter_actions` + `replay_human_action`, the human importer with its training-data drop rules off; an undo rebuilds) and answers win estimates (value net), the policy's top-5 moves with prices (the supervised policy during the auction), the model's probability for the last moves, and an optional Rust-MCTS search (`advisor.py`, `describe.py`). Read-only: the user makes the moves on the site. It also follows hotseat games (the site keeps those in the browser's localStorage) and the site's history view (`?action=N`: advice for the game truncated to that action). `--host 0.0.0.0` serves other machines on the LAN (the extension then needs the backend URL and the optional host permission). Served today: policy gen4/1800, auction policy ckpt 10, value_v2 (value checkpoint 26; value_v3 is worse on human games). Test game: `tests/fixtures/advisor/game_1830_4p.json` (`scripts/make_advisor_fixture.py`, no human data).

### Agent Interface (`rl18xx/agent/agent.py`)

Abstract base class with: `initialize_game`, `get_game_state`, `suggest_move`, `play_move`.

### Arena (`rl18xx/agent/arena.py`)

Runs matches between agents (MCTS or random). Can optionally sync to 18xx.games for browser visualization.

## Key Patterns

- Game state is always accessed through `BaseGame` (Python) or `RustGameAdapter` (Rust) — never construct engine objects directly.
- Legal moves come from `ActionHelper.get_all_choices(game)`, applied via `game.process_action(action)`.
- The encoder/action_mapper bridge between the engine's object model and the neural network's tensor representation. The action layout is duplicated in Python (`action_mapper.py`) and Rust (`action_index.rs`) and must stay in sync.
- **Optional rules**: engines are built with a recorded game's `settings.optional_rules` filtered to the ones BOTH implement (`pretraining.ENGINE_OPTIONAL_RULES`; today 1830's `optional_6_train` and `multiple_brown_from_ipo`). The Rust constructor raises ValueError for any rule its title doesn't declare (`GameTitle::optional_rules`). Games with other optional rules replay under the base rules and are dropped if the engine rejects an action.
- **Self-play starts at the first Stock Round** (`start_positions.py`, `SelfPlayHyperparams.start_positions_path`): 80% from the real post-auction position of a 4-player human game (`human_games/start_positions_1830_4p.jsonl`, built by `scripts/build_start_positions.py`), 20% (`random_start_fraction`) from a random auction ending reached through legal actions (each private to a random player at a multiple of $5 in [face, 2x face]; the SV, which can't be bid on, goes at face or -- in a quarter of them -- $5-$20 below after all-pass rounds, free at $0; random B&O par). Training examples cover only the self-play decisions. The waterfall auction itself is to be trained separately. Policy-gradient runs can also start a share of games mid-game (`--midgame-fraction`): a human game cut at the start of Stock Round 2-6 (`human_games/midgame_positions_1830_4p.jsonl`, built by `scripts/build_midgame_positions.py`), because self-play from SR1 settled into slow games that rarely reach trainless companies, lethal dumps or bankruptcies (2026-10-07: agents' dumps hit a trainless company 10% of the time vs humans' 38%).
- **Rule variant `auction_unlock`** (both engines, off by default; `set_auction_unlock` in Rust, `BaseGame.auction_unlock` in Python): an all-pass waterfall-auction round discounts the next private by $5 like the SV when nobody has bid on it. Real 1830 only discounts the SV, so an auction where everyone keeps passing never ends — self-play kept hitting that; humans pass with a non-SV private next in ~5% of games. Gate games and self-play games that start at the auction turn it on (`SelfPlayHyperparams.auction_unlock`); start-position games, human import and arena don't. It is training wheels: the loop logs how often it fires (`SelfPlay/Auction_Unlock_Game_Rate`) so it can be turned off once self-play stops locking the auction.
- **Game limits count decisions**: `max_game_length`, `resign_min_move`, `softpick_move_cutoff` and `auction_stall_moves` count moves where the player had more than one legal action (`MCTSPlayer` / `RustMCTSPlayer.decisions`). Forced actions are applied automatically (collapsed into the preceding search edge) and are never training examples. The search trees only bound engine actions as a safety net (`max_engine_actions`, both engines).
- **Value-head memorization**: every position of a game shares one outcome, so the value head memorizes games quickly (train-window CE far below new-game CE). Self-play training uses `TrainingConfig.value_stop_grad` (value heads on detached trunk features) with `value_lr_multiplier` 0.3; the loop logs `Value/CE_New_Games` vs `Value/CE_Trained_Window` each iteration. `scripts/value_overfit_replay.py` replays a finished run's training under other settings to compare them offline. A resigned game's value target names the player the search was confident in (`self_play.resigned_result`), not the net-worth leader at the resign point.
- **MCTS selection Q**: both engines default to minigo's `W / (1 + N)` with unvisited children at 0. Values here are 0-1 win shares (0 = certain loss), so that reads unexplored moves as losses and, with c_puct ~2-3 against sibling value gaps of ~0.02, the search mostly follows the prior. `SelfPlayHyperparams.mcts_mean_q` (Rust only) switches to mean Q with first-play urgency at the parent's value (`fpu_reduction`). Note `c_puct_by_round` overrides `c_puct_init` in every 1830 round. **Every Rust-MCTS result before 2026-10-16 is void**: the tree took a node's player to move from the active *player*, empty on a corporation's turn, so every operating-round node searched for seat 0 whoever was moving (fixed 3217c28: `BaseGame::acting_player_id_pub` is the active corporation's president, as in `RustGameAdapter.active_players`). It showed as a seat-0 edge growing with search (seat 0 won up to 0.50 of search-eval games; policy-only evals 0.25 per seat) -- check `seat_win_rate` in eval summaries. It voids the AlphaZero-loop self-play targets, the "value-driven search plays worse" and "a better value net doesn't help search" results, and the advisor's search before that date; the policy-gradient line never searched. Two more search issues (options, off by default until games pick settings): a virtual loss lowers W but not N, so a batch re-selects unexpanded children and every copy was backed up as a visit (`@64` with `parallel_readouts` 32 evaluated ~17 distinct positions; at > 20 legal moves the top-k mask sends the whole first batch to the top-prior move) -- `dedupe_leaves` (eval `:dedupe`) backs each distinct leaf up once, as minigo does; and a policy-gradient learner *is* softmax(logits / learner_temperature) while the tree used temperature 1 -- `prior_temperature` (`ptemp=`). gen4's policy is so sharp that even @800 the search leaves its top move at only ~3-8% of positions (`scripts/search_decision_diag.py`); `scripts/search_override_rollouts.py` plays both moves out to check whether those departures are better. In head-to-heads, `--softpick` samples the first decisions from visit counts, so a lower budget's flatter visits also mean a more random opening. **Gumbel root** (option, off: `SelfPlayConfig.gumbel_k` > 0, eval `:gumbel=<k>`, `gscale=` / `cvisit=` / `cscale=`; `engine-rs/src/gumbel.rs`, Danihelka et al. 2022 / mctx): top-k root moves by Gumbel noise + log-prior, sequential halving over exactly the budget, plays argmax(g + logits + sigma(q)) (no softpick, no Dirichlet noise) and records softmax(logits + sigma(completed q)) in `searches_pi`; sigma(q) = (c_visit 50 + max N) * c_scale 1 * q on the mover's 0-1 share, unvisited moves completed by the root's v_mix. Below the root PUCT is unchanged; a price-bearing root move is one candidate whose q aggregates its price grandchildren, and plays its most-visited price. Logits are the tree's priors, i.e. after `prior_temperature`.
- **Numerics**: training is bf16, inference fp32, and the first non-finite loss or gradient stops training (and pretraining) without saving; `Numerics/*` logs per-block activation peaks and weight norms.
- **Parity is load-bearing**: the Rust engine and the `RustGameAdapter` must reproduce Python's behavior exactly (down to which games the cleaning pipeline drops). Parity audits and bug logs live in `docs/` (`rust_engine_*_audit.*`, `cleaning_engine_parity.*`, `rust_engine_bugs.md`).
- **Ruby is the source of truth for rules**: Python-Rust parity proves our two engines agree, not that they match Ruby, and replaying human games can't catch moves humans can't make on the site. `scripts/ruby_replay_check.py` replays games OUR engine generated (self-play exports, random walks) through Ruby HEAD in docker (`scripts/ruby_diff.py`'s rig); every Ruby rejection or divergence is a rule we get wrong (2026-10-16: the IPO lockout on multiple buys, ERIE's E11 reservation on a laid tile, the bank breaking at $0 instead of below zero, must-sell). Fix both engines and keep parity. **Intentional exceptions** (the user's, for simplicity; self-play generators only, replay stays Ruby-faithful): players may not buy trains from corporations run by other players, nor companies from other players (1830 `GameTitle::train_buy_from_other_players()` false; the importer drops `cross_player_*` games); 1822-family majors acquire only minors and (PNW) regionals their president owns; 1867 merges only minors all owned by one player (99.4% of human mergers are). 1822/1867 put these behind `GameTitle::generator_cross_player` (`RESTRICTED` default, `policy="rules"` = Ruby's set): keep them switchable, the user may reopen cross-player deals for a strong agent. Known leftovers: Ruby's route-info cache quirk (Python removed it on purpose) and G19's two cities numbered the other way round from Ruby (labels only; changing them would move policy slots).
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
