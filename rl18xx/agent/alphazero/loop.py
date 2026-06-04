from concurrent.futures import as_completed, ProcessPoolExecutor
import os
import json
import logging
import gc
import shutil
import time
from datetime import datetime
from dataclasses import dataclass, asdict
from rl18xx.agent.alphazero.checkpointer import (
    get_latest_model,
    save_model,
    save_optimizer_state,
    session_name_for,
    set_current_best,
)
from rl18xx.agent.alphazero.config import SelfPlayConfig, TrainingConfig
from rl18xx.agent.alphazero.metrics import Metrics
from rl18xx.agent.alphazero.self_play import MCTSPlayer, SelfPlay, SELF_PLAY_GAMES_STATUS_PATH
from rl18xx.agent.alphazero.train import train
from rl18xx.shared.atomic_io import atomic_write_json
from pathlib import Path
from typing import Optional
import signal
import sys
import psutil
import atexit
import random
import numpy as np
import torch

MODEL_CHECKPOINT_DIR = "model_checkpoints"

LOGGER = logging.getLogger(__name__)

LOOP_LOCK_FILE = Path("loop.lock")
LOOP_CONFIG_PATH = Path("loop_config.json")
LOOP_STATUS_PATH = Path("loop_status.json")
TENSORBOARD_LOG_DIR_BASE = Path("runs/alphazero_runs")
SELF_PLAY_LOGS_PATH = Path("logs/self_play")
METRICS_HISTORY_PATH = Path("logs/loop/metrics_history.jsonl")
MODEL_HISTORY_PATH = Path("logs/loop/model_history.jsonl")


def _safe_read_json(filepath: Path) -> dict | None:
    """Read a JSON file safely, tolerating partial/concurrent writes."""
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def _get_checkpoint_count() -> int:
    from rl18xx.agent.alphazero.checkpointer import _find_latest_session, _find_latest_checkpoint

    try:
        session_dir = _find_latest_session(MODEL_CHECKPOINT_DIR)
        checkpoint_path = _find_latest_checkpoint(session_dir)
        return int(checkpoint_path.stem)
    except FileNotFoundError:
        return 0


def get_scheduled_value(checkpoint: int, schedule: tuple[int, int, int]) -> int:
    """Linearly interpolate from schedule[0] to schedule[1] over schedule[2] checkpoints."""
    start_val, end_val, ramp = schedule
    if checkpoint >= ramp:
        return end_val
    t = checkpoint / max(ramp, 1)
    return int(start_val + t * (end_val - start_val))


def setup_logging(level: int, log_file: str, console: bool = False) -> logging.Logger:
    # Set up logging to both console and file
    log_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)

    # File handler
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(log_formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    if console:
        root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)


@dataclass
class LoopConfig:
    num_loop_iterations: int
    num_games_per_iteration: int
    num_threads: int
    training_config: TrainingConfig
    num_readouts: int
    target_experiences: int = 0  # 0 = use num_games_per_iteration instead
    # Variable player count support: each self-play game samples its player
    # count from this distribution. ``None`` (the default) falls back to
    # SelfPlayHyperparams' default — 4-player-heavy mix that still exposes
    # the model to 2-, 3-, 5-, and 6-player games. Override per run via the
    # loop config file.
    player_count_distribution: Optional[dict] = None
    # Phase 2 consensus resign settings. The high threshold is the only one
    # auto-calibrated between iterations; the rest are stable hyperparams.
    enable_resign: bool = True
    resign_window: int = 8
    resign_high_threshold: float = 0.65
    resign_gap_threshold: float = 0.30
    noresign_holdout_rate: float = 0.10
    resign_high_threshold_min: float = 0.45
    # Phase 3 cross-process inference server. Default off — flip to True
    # to route self-play workers' inference through a single GPU-bound
    # server process (cross-game batching). Verify on a short training
    # run before promoting.
    use_inference_server: bool = False
    inference_batch_size: int = 64
    inference_batch_timeout_ms: float = 2.0


@dataclass
class LoopMetrics:
    """Tracks metrics across the entire training loop"""

    loop_iteration: int = 0
    games_played_total: int = 0
    training_examples_total: int = 0
    training_losses: list = None
    policy_losses: list = None
    value_losses: list = None
    game_lengths: list = None
    win_rates_by_player: dict = None
    # Phase 2 resign auto-calibration history (most recent first ~3
    # iterations). Populated by ``calibrate_resign_threshold``.
    recent_resign_fp_rates: list = None

    def __post_init__(self):
        if self.training_losses is None:
            self.training_losses = []
        if self.policy_losses is None:
            self.policy_losses = []
        if self.value_losses is None:
            self.value_losses = []
        if self.game_lengths is None:
            self.game_lengths = []
        if self.win_rates_by_player is None:
            self.win_rates_by_player = {0: [], 1: [], 2: [], 3: []}
        if self.recent_resign_fp_rates is None:
            self.recent_resign_fp_rates = []


def load_loop_config(
    num_loop_iterations: int,
    num_games_per_iteration: int,
    num_threads: int,
    default_training_config: TrainingConfig,
    num_readouts: int,
    target_experiences: int = 0,
) -> LoopConfig:
    """Load loop config. CLI args always take precedence. File values are used as defaults
    for training_config fields only (batch_size, lr, etc.) and can be hot-reloaded mid-run."""
    training_config = default_training_config
    player_count_distribution: Optional[dict] = None
    resign_overrides: dict = {}

    # If config file exists, merge training hyperparams from it (hot-reload support)
    if LOOP_CONFIG_PATH.exists():
        try:
            with open(LOOP_CONFIG_PATH, "r") as f:
                file_config = json.load(f)
            if "training_config" in file_config:
                training_config = TrainingConfig.from_json(file_config["training_config"])
            if "player_count_distribution" in file_config and file_config["player_count_distribution"]:
                # JSON serializes int keys as strings — convert back.
                raw = file_config["player_count_distribution"]
                player_count_distribution = {int(k): float(v) for k, v in raw.items()}
            # Phase 2 resign params: pick up auto-calibrated threshold + any
            # manually edited bounds from the file. Missing keys keep defaults.
            for key in (
                "enable_resign",
                "resign_window",
                "resign_high_threshold",
                "resign_gap_threshold",
                "noresign_holdout_rate",
                "resign_high_threshold_min",
            ):
                if key in file_config and file_config[key] is not None:
                    resign_overrides[key] = file_config[key]
        except Exception as e:
            LOGGER.warning(f"Error reading loop config file: {e}. Using CLI defaults.")

    loop_config = LoopConfig(
        num_loop_iterations=num_loop_iterations,
        num_games_per_iteration=num_games_per_iteration,
        num_threads=num_threads,
        training_config=training_config,
        num_readouts=num_readouts,
        target_experiences=target_experiences,
        player_count_distribution=player_count_distribution,
        **resign_overrides,
    )

    # Write current config to file for visibility / hot-reload editing
    serializable = asdict(loop_config)
    serializable["training_config"] = training_config.to_json()
    atomic_write_json(LOOP_CONFIG_PATH, serializable, indent=4)

    return loop_config


def update_loop_status(status_data: dict):
    """Writes the current loop status to a JSON file."""
    try:
        atomic_write_json(LOOP_STATUS_PATH, status_data, indent=4, default=str)
    except Exception as e:
        LOGGER.error(f"Error writing to {LOOP_STATUS_PATH}: {e}")


def save_loop_metrics(metrics: LoopMetrics, filepath: Path):
    """Saves the loop metrics to a JSON file."""
    try:
        metrics_dict = asdict(metrics)
        atomic_write_json(filepath, metrics_dict, indent=4)
    except Exception as e:
        LOGGER.error(f"Error saving loop metrics: {e}")


def load_loop_metrics(filepath: Path) -> LoopMetrics:
    """Loads loop metrics from a JSON file."""
    try:
        if filepath.exists():
            with open(filepath, "r") as f:
                metrics_dict = json.load(f)
            return LoopMetrics(**metrics_dict)
    except Exception as e:
        LOGGER.error(f"Error loading loop metrics: {e}")
    return LoopMetrics()


def append_metrics_history(record: dict):
    """Append a single metrics record to the JSONL history file."""
    try:
        METRICS_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(METRICS_HISTORY_PATH, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except Exception as e:
        LOGGER.error(f"Error appending to metrics history: {e}")


def append_model_history(record: dict):
    """Append a single model gating record to the JSONL history file."""
    try:
        MODEL_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(MODEL_HISTORY_PATH, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except Exception as e:
        LOGGER.error(f"Error appending to model history: {e}")


def cleanup_files():
    if LOOP_CONFIG_PATH.exists():
        LOOP_CONFIG_PATH.unlink()
    if LOOP_STATUS_PATH.exists():
        LOOP_STATUS_PATH.unlink()
    if LOOP_LOCK_FILE.exists():
        LOOP_LOCK_FILE.unlink()
    if TENSORBOARD_LOG_DIR_BASE.exists():
        for item in TENSORBOARD_LOG_DIR_BASE.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
    if SELF_PLAY_GAMES_STATUS_PATH.exists():
        for item in SELF_PLAY_GAMES_STATUS_PATH.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()


def run_self_play(
    game_idx_in_iteration: int,
    tb_log_dir: str,
    timestamp: str,
    loop: int,
    num_readouts: int = 32,
    max_game_length: int = 1000,
    num_players: int = 4,
):
    process_root_logger = logging.getLogger()
    random.seed(os.getpid())
    np.random.seed(os.getpid() ^ int(time.time()))
    for handler in process_root_logger.handlers[:]:
        process_root_logger.removeHandler(handler)
        handler.close()
    process_root_logger.setLevel(logging.INFO)
    game_log_file = SELF_PLAY_LOGS_PATH / f"{timestamp}_game_L{loop}_G{game_idx_in_iteration}.log"
    file_handler = logging.FileHandler(game_log_file)
    log_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(log_formatter)
    process_root_logger.addHandler(file_handler)
    logging.info(
        f"Self-play process started for L{loop}/G{game_idx_in_iteration} "
        f"({num_players}-player). Logging to {game_log_file}"
    )

    # Phase 3: when the inference server is enabled, this worker's
    # ``InferenceClient`` was stashed by ``worker_init_inference`` at
    # process boot. Fetch it (or None) and route inference through it
    # instead of holding a local model copy.
    from rl18xx.agent.alphazero.inference_server import get_worker_client
    inference_client = get_worker_client()
    model = None if inference_client is not None else get_latest_model("model_checkpoints")

    # Phase 2: pull resign params from the loop config file (auto-calibrated
    # between iterations). Missing keys fall back to SelfPlayHyperparams
    # defaults (frozen). Reading on each game pickups any mid-run tuning.
    resign_kwargs: dict = {}
    file_cfg = _safe_read_json(LOOP_CONFIG_PATH) if LOOP_CONFIG_PATH.exists() else None
    if file_cfg:
        for key in (
            "enable_resign",
            "resign_window",
            "resign_high_threshold",
            "resign_gap_threshold",
            "noresign_holdout_rate",
            "resign_high_threshold_min",
        ):
            if key in file_cfg and file_cfg[key] is not None:
                resign_kwargs[key] = file_cfg[key]
    # Phase 3 inference-server flag mirrors what the parent set up. When
    # we have a client, ``MCTSPlayer`` routes inference through it.
    server_kwargs: dict = {}
    if inference_client is not None:
        server_kwargs["use_inference_server"] = True
        if file_cfg:
            for key in ("inference_batch_size", "inference_batch_timeout_ms"):
                if key in file_cfg and file_cfg[key] is not None:
                    server_kwargs[key] = file_cfg[key]

    try:
        # Tensorboard logging is disabled for now because it's using up too much space on disk
        self_play_config = SelfPlayConfig(
            network=model,
            metrics=None,  # Metrics(os.path.join(tb_log_dir, f"game_L{loop}_G{game_idx_in_iteration}")),
            global_step=loop,
            game_idx_in_iteration=game_idx_in_iteration,
            game_id=f"L{loop}_G{game_idx_in_iteration}",
            num_readouts=num_readouts,
            max_game_length=max_game_length,
            # Lock the worker to the sampled count by collapsing the distribution
            # to a singleton — the child SelfPlay only ever opens one game and
            # we want it to use the seat count chosen by the parent process.
            player_count_distribution={num_players: 1.0},
            inference_client=inference_client,
            **resign_kwargs,
            **server_kwargs,
        )
        selfplay = SelfPlay(self_play_config)
        selfplay.run_game()
        logging.info(f"Self-play game L{loop}/G{game_idx_in_iteration} completed successfully.")
    except Exception as e_proc:
        logging.error(f"Error during self-play game L{loop}/G{game_idx_in_iteration}: {e_proc}", exc_info=True)
        raise  # Re-raise to be caught by the main process's future.result()
    finally:
        # Clean up handlers for this process to ensure files are flushed and closed.
        for handler in process_root_logger.handlers[:]:
            handler.close()
            process_root_logger.removeHandler(handler)


def cleanup_and_exit(signum=None, frame=None):
    LOGGER.info("--- Received interrupt signal, cleaning up ---")
    parent = psutil.Process(os.getpid())
    children = parent.children(recursive=True)

    for child in children:
        try:
            LOGGER.info(f"Terminating child process {child.pid}")
            child.terminate()
        except psutil.NoSuchProcess:
            pass

    # Give them a moment to terminate gracefully
    gone, alive = psutil.wait_procs(children, timeout=3)

    # Force kill any remaining processes
    for child in alive:
        try:
            LOGGER.warning(f"Force killing child process {child.pid}")
            child.kill()
        except psutil.NoSuchProcess:
            pass
    if LOOP_LOCK_FILE.exists():
        LOOP_LOCK_FILE.unlink()


def _create_fresh_game(num_players: int = 4):
    """Create a new ``num_players``-player 1830 game instance.

    Players are always named ``"Player 1"``..``"Player N"`` so the encoder's
    canonical ``player_id_to_idx`` mapping stays deterministic regardless of
    how this game arose (self-play sampler, gate game, or pretraining replay).
    Player count must be in 2..6 (1830 rules).

    The training loop runs exclusively on the Rust engine; the Python engine is
    not used here. If the Rust engine isn't installed, this raises ImportError.
    """
    assert 2 <= num_players <= 6, f"num_players must be 2..6, got {num_players}"
    from engine_rs import BaseGame as RustGame
    from rl18xx.rust_adapter import RustGameAdapter
    players = {i + 1: f"Player {i + 1}" for i in range(num_players)}
    return RustGameAdapter(RustGame(players))


def _sample_player_count(distribution: dict[int, float] | None) -> int:
    """Sample a player count from ``distribution``.

    ``distribution`` maps ``num_players -> weight``. Weights need not sum to 1.
    If ``distribution`` is ``None`` or empty, falls back to a 4-player game
    (the legacy default).
    """
    if not distribution:
        return 4
    counts = list(distribution.keys())
    weights = [distribution[c] for c in counts]
    total = sum(weights)
    if total <= 0:
        return 4
    return random.choices(counts, weights=weights, k=1)[0]


# Small Dirichlet noise weight injected during gating so each game produces a
# distinct trajectory. With deterministic priors and argmax move selection, gate
# games would otherwise be identical given the same seat assignment, collapsing
# `gate_games=N` to at most `num_seats` distinct outcomes. See
# docs/step1_review.md "Improved gating mechanics".
GATING_DIRICHLET_NOISE_WEIGHT = 0.1


def _gate_seat_assignment(game_index: int, num_seats: int = 4) -> list[bool]:
    """Return ``is_candidate[seat]`` for each seat in the given gate game.

    Candidate cycles through every seat as ``game_index`` advances:
    game 0 -> seat 0, game 1 -> seat 1, ..., game ``num_seats-1`` -> seat
    ``num_seats-1``, then wraps. ``current_best`` fills the other seats. With
    ``gate_games`` a multiple of ``num_seats``, each seat configuration is
    sampled the same number of times, controlling for priority-deal advantages.
    """
    candidate_seat = game_index % num_seats
    return [seat == candidate_seat for seat in range(num_seats)]


def _play_gate_game(
    candidate_model,
    current_best_model,
    game_index: int,
    num_readouts: int,
    num_players: int = 4,
) -> dict:
    """Play a single gating arena game and return result info.

    Seat rotation: candidate occupies a single seat
    (``game_index % num_players``); ``current_best`` fills the remaining
    seats. With ``gate_games`` a multiple of ``num_players``, every seat is
    sampled equally.

    A small Dirichlet noise weight (``GATING_DIRICHLET_NOISE_WEIGHT``) is
    injected on the root prior so each game produces a distinct trajectory.
    ``softpick_move_cutoff=0`` (argmax move selection) is kept — trajectory
    diversity comes from the noised root prior, not random move selection.

    Returns a dict with 'candidate_seats', 'winner_seat', and 'scores'.
    """
    eval_config_candidate = SelfPlayConfig(
        softpick_move_cutoff=0,
        dirichlet_noise_weight=GATING_DIRICHLET_NOISE_WEIGHT,
        num_readouts=num_readouts,
        network=candidate_model,
    )
    eval_config_best = SelfPlayConfig(
        softpick_move_cutoff=0,
        dirichlet_noise_weight=GATING_DIRICHLET_NOISE_WEIGHT,
        num_readouts=num_readouts,
        network=current_best_model,
    )

    is_candidate_by_seat = _gate_seat_assignment(game_index, num_seats=num_players)
    agents = [
        MCTSPlayer(eval_config_candidate if is_cand else eval_config_best)
        for is_cand in is_candidate_by_seat
    ]
    candidate_seats = {seat for seat, is_cand in enumerate(is_candidate_by_seat) if is_cand}

    game_state = _create_fresh_game(num_players=num_players)
    for agent in agents:
        agent.initialize_game(game_state)

    agent_by_player_id = {player.id: agent for player, agent in zip(game_state.players, agents)}
    seat_by_player_id = {player.id: seat for seat, player in enumerate(game_state.players)}

    while not game_state.finished:
        if game_state.move_number >= 1000:
            game_state.end_game()
            break

        current_player = game_state.active_players()[0]
        move = agent_by_player_id[current_player.id].suggest_move()
        for agent in agents:
            agent.play_move(move)

    result = game_state.result()
    best_score = max(result.values())
    winner_player_id = next(pid for pid, score in result.items() if score == best_score)
    winner_seat = seat_by_player_id[winner_player_id]

    return {
        "candidate_seats": candidate_seats,
        "winner_seat": winner_seat,
        "scores": {seat_by_player_id[pid]: score for pid, score in result.items()},
    }


def evaluate_candidate(
    candidate_model,
    current_best_model,
    num_games: int = 10,
    num_readouts: int = 50,
) -> float:
    """Play candidate vs current_best in arena games, return candidate win rate.

    Each game has 4 players: 1 with the candidate model, 3 with the current
    best. The candidate's seat rotates through all four seats across games
    (see ``_gate_seat_assignment``), controlling for priority-deal advantages.
    A small Dirichlet noise weight is injected on the root prior to ensure
    each game produces a distinct trajectory (without it, deterministic priors
    + argmax move selection collapse all gate games at a given seat assignment
    onto an identical trajectory).

    To disable gating entirely once training is stable, pass ``--no-gate``
    to the loop; that's the AlphaZero / AlphaGo Zero approach (always promote
    the latest trained model).

    A win is counted when the overall game winner occupies the candidate seat.
    Games that crash are skipped and not counted toward the total.
    """
    candidate_wins = 0
    games_completed = 0

    for game_index in range(num_games):
        LOGGER.info(f"Gating game {game_index + 1}/{num_games} starting...")
        try:
            result = _play_gate_game(candidate_model, current_best_model, game_index, num_readouts)
            games_completed += 1
            if result["winner_seat"] in result["candidate_seats"]:
                candidate_wins += 1
                LOGGER.info(
                    f"Gating game {game_index + 1}: candidate WON (seat {result['winner_seat']}). "
                    f"Scores: {result['scores']}"
                )
            else:
                LOGGER.info(
                    f"Gating game {game_index + 1}: candidate LOST (winner seat {result['winner_seat']}). "
                    f"Scores: {result['scores']}"
                )
        except Exception as e:
            LOGGER.error(f"Gating game {game_index + 1} crashed, skipping: {e}", exc_info=True)

    if games_completed == 0:
        LOGGER.warning("All gating games crashed. Returning 0.0 win rate.")
        return 0.0

    win_rate = candidate_wins / games_completed
    LOGGER.info(f"Gating evaluation complete: {candidate_wins}/{games_completed} wins ({win_rate:.1%})")
    return win_rate


def ensure_seed_model(model_type: str = "transformer"):
    """Create an initial model checkpoint if none exists."""
    p = Path(MODEL_CHECKPOINT_DIR)
    # Check for any session directories (new format: <arch>/<session>/)
    has_checkpoints = False
    if p.exists():
        for arch_dir in p.iterdir():
            if arch_dir.is_dir() and any(arch_dir.iterdir()):
                has_checkpoints = True
                break

    if has_checkpoints:
        return

    LOGGER.info(f"No model checkpoints found. Creating fresh {model_type} model...")
    if model_type == "transformer":
        from rl18xx.agent.alphazero.config import ModelTransformerConfig
        from rl18xx.agent.alphazero.model_transformer import AlphaZeroTransformerModel
        import torch

        config = ModelTransformerConfig()
        torch.manual_seed(config.seed)
        LOGGER.info(f"Seeding model initialization with seed={config.seed}")
        model = AlphaZeroTransformerModel(config)
    else:
        import warnings
        warnings.warn(
            "model_type='v1' (AlphaZeroGNNModel) is deprecated. The transformer "
            "model (v2) is the default and recommended architecture.",
            DeprecationWarning,
            stacklevel=2,
        )
        from rl18xx.agent.alphazero.model import AlphaZeroGNNModel
        from rl18xx.agent.alphazero.config import ModelGNNConfig
        import torch

        config = ModelGNNConfig()
        torch.manual_seed(config.seed)
        LOGGER.info(f"Seeding model initialization with seed={config.seed}")
        model = AlphaZeroGNNModel(config)

    checkpoint_num = save_model(model, MODEL_CHECKPOINT_DIR)
    # Bootstrap the current_best pointer so self-play workers can find the seed
    # via the same code path as the steady-state (pointer-based) load.
    set_current_best(
        MODEL_CHECKPOINT_DIR,
        model.architecture_name(),
        session_name_for(model),
        checkpoint_num,
    )
    LOGGER.info(f"Saved initial {model_type} model: {model.get_name()} (checkpoint {checkpoint_num})")


def cleanup_model_and_data():
    """Remove all model checkpoints and training data for a fresh start."""
    checkpoint_dir = Path(MODEL_CHECKPOINT_DIR)
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
        LOGGER.info(f"Removed {checkpoint_dir}")

    training_dir = Path("training_examples")
    if training_dir.exists():
        shutil.rmtree(training_dir)
        LOGGER.info(f"Removed {training_dir}")


ESTIMATED_MOVES_PER_GAME = 1000


@dataclass
class SelfPlayIterationResult:
    games_completed: int
    game_lengths: list
    experiences: int
    wall_time: float


@dataclass
class SelfPlayAggregateStats:
    avg_game_length: float
    min_game_length: int
    max_game_length: int
    games_per_hour: float
    moves_per_second: float
    phase_move_counts: dict
    timing_avgs: dict


@dataclass
class TrainingIterationResult:
    model: object  # AlphaZeroModel
    train_metrics: object  # TrainingMetrics or None
    wall_time: float


@dataclass
class GatingIterationResult:
    win_rate: object  # Optional[float]
    promoted: bool
    checkpoint_num: object  # Optional[int]
    wall_time: float


_SELFPLAY_TIMING_KEYS = [
    "total_game_s", "tree_search_s", "inference_s", "encoding_s",
    "leaf_selection_s", "backup_s", "pick_move_s", "play_move_s",
    "data_extraction_s",
]


def _server_model_factory(checkpoint_path: Optional[str] = None):
    """Inference-server-side model loader.

    Used by the Phase 3 ``InferenceServer`` to (re)load the current-best
    model at STARTING and RELOADING transitions. The ``checkpoint_path``
    argument is informational only — we always defer to
    ``get_latest_model`` so the gated current-best pointer is honored.
    Must be a module-level top-function so it pickles across the spawn
    boundary into the server child process.
    """
    return get_latest_model(MODEL_CHECKPOINT_DIR)


def _submit_selfplay_game(
    executor, game_idx, tb_log_dir, timestamp, loop, num_readouts, max_game_length, num_players=4
):
    """Submit a single self-play game to the executor."""
    return executor.submit(
        run_self_play,
        game_idx,
        tb_log_dir,
        timestamp,
        loop,
        num_readouts,
        max_game_length,
        num_players,
    )


def _ingest_completed_selfplay_game(
    future, game_idx, loop, games_completed_count, experiences_this_iteration,
    game_lengths_this_iteration, loop_metrics, target_experiences,
):
    """Block on a finished self-play future, harvest its game-stats JSON, mutate counters.

    Returns (games_completed_count, experiences_this_iteration). Logs on failure
    so a single crashed game doesn't take down the iteration.
    """
    try:
        future.result()
        games_completed_count += 1
        game_file = SELF_PLAY_GAMES_STATUS_PATH / f"L{loop}_G{game_idx}.json"
        gdata = _safe_read_json(game_file) if game_file.exists() else None
        if gdata:
            moves = gdata.get("moves_played", 0)
            experiences_this_iteration += moves
            game_lengths_this_iteration.append(moves)
            loop_metrics.game_lengths.append(moves)
        LOGGER.info(
            f"Loop {loop+1}: Game {games_completed_count} completed. "
            f"Experiences: {experiences_this_iteration}/{target_experiences}"
        )
    except Exception as e:
        LOGGER.error(f"Error in self-play game {game_idx}: {e}", exc_info=True)
    return games_completed_count, experiences_this_iteration


def _run_selfplay_iteration(
    loop: int,
    loop_config: "LoopConfig",
    scheduled_game_length: int,
    tb_log_dir: str,
    timestamp: str,
    loop_metrics: "LoopMetrics",
    server_handle: Optional["ServerHandle"] = None,
) -> SelfPlayIterationResult:
    """Drive the process-pool self-play phase until target experiences are reached.

    When ``server_handle`` is set (Phase 3 inference server enabled), workers
    boot with ``worker_init_inference`` so they pull a unique slot ticket
    and stash an ``InferenceClient`` on the module globals; ``run_self_play``
    then attaches that client to the worker's ``SelfPlayConfig``.
    """
    import concurrent.futures

    games_completed_count = 0
    game_lengths_this_iteration: list[int] = []
    experiences_this_iteration = 0
    game_idx = 0
    selfplay_start_time = time.time()

    LOGGER.info(
        f"Loop {loop+1}: Running self-play until {loop_config.target_experiences} experiences "
        f"using {loop_config.num_threads} processes..."
    )

    # Sample a player count per game. ``None`` falls back to legacy 4-player.
    player_count_distribution = loop_config.player_count_distribution

    executor_kwargs: dict = {"max_workers": loop_config.num_threads}
    if server_handle is not None:
        from rl18xx.agent.alphazero.inference_server import worker_init_inference
        executor_kwargs["initializer"] = worker_init_inference
        executor_kwargs["initargs"] = (
            server_handle.request_q,
            server_handle.reply_qs,
            server_handle.ticket_q,
        )
    executor = ProcessPoolExecutor(**executor_kwargs)
    try:
        pending_futures: dict = {}
        for _ in range(loop_config.num_threads):
            num_players = _sample_player_count(player_count_distribution)
            f = _submit_selfplay_game(
                executor, game_idx, tb_log_dir, timestamp, loop,
                loop_config.num_readouts, scheduled_game_length, num_players,
            )
            pending_futures[f] = game_idx
            game_idx += 1

        while experiences_this_iteration < loop_config.target_experiences and pending_futures:
            done_futures = [f for f in pending_futures if f.done()]
            if not done_futures:
                completed, _ = concurrent.futures.wait(
                    pending_futures.keys(), return_when=concurrent.futures.FIRST_COMPLETED
                )
                done_futures = list(completed)

            for f in done_futures:
                gidx = pending_futures.pop(f)
                games_completed_count, experiences_this_iteration = _ingest_completed_selfplay_game(
                    f, gidx, loop, games_completed_count, experiences_this_iteration,
                    game_lengths_this_iteration, loop_metrics, loop_config.target_experiences,
                )
                if experiences_this_iteration < loop_config.target_experiences:
                    num_players = _sample_player_count(player_count_distribution)
                    f_new = _submit_selfplay_game(
                        executor, game_idx, tb_log_dir, timestamp, loop,
                        loop_config.num_readouts, scheduled_game_length, num_players,
                    )
                    pending_futures[f_new] = game_idx
                    game_idx += 1

        # Drain any in-flight games left over after we hit the target.
        for f in list(pending_futures.keys()):
            gidx = pending_futures[f]
            games_completed_count, experiences_this_iteration = _ingest_completed_selfplay_game(
                f, gidx, loop, games_completed_count, experiences_this_iteration,
                game_lengths_this_iteration, loop_metrics, loop_config.target_experiences,
            )
    finally:
        executor.shutdown(wait=True)

    return SelfPlayIterationResult(
        games_completed=games_completed_count,
        game_lengths=game_lengths_this_iteration,
        experiences=experiences_this_iteration,
        wall_time=time.time() - selfplay_start_time,
    )


def calibrate_resign_threshold(
    loop: int,
    loop_config: "LoopConfig",
    loop_metrics: "LoopMetrics",
    metrics: Optional[Metrics] = None,
) -> dict:
    """Auto-calibrate ``resign_high_threshold`` from the iteration's holdout games.

    Implements the AlphaGo-Zero schedule from
    docs/mcts_improvements_plan.md Phase 2:

    - Scan ``L{loop}_G*.json`` for **completed holdout** games (those with
      ``noresign_holdout=True``). These are games that ignored the resign
      signal and played to completion, so we have ground truth for whether
      the resign signal would have been correct.
    - For each holdout that recorded a ``would_have_resigned`` moment, the
      decision is a **false positive** when the leader at that moment is
      not the actual winner of the game.
    - ``fp_rate = 1 - correct / would_have_resigned`` (denominator is
      holdouts where the conditions ever held; absent denominator yields
      ``None``).
    - Tightening: if ``fp_rate > 5%`` → bump threshold by +0.05.
    - Loosening: if the last 3 iterations all had ``fp_rate < 2%`` →
      lower threshold by -0.05 and clear the running history (so we don't
      loosen again immediately).
    - Clamped at ``loop_config.resign_high_threshold_min``.

    Mutates ``loop_config.resign_high_threshold`` and persists the entire
    loop config back to ``LOOP_CONFIG_PATH`` so the next iteration's
    ``load_loop_config`` (and the worker's resign_kwargs lookup) picks it
    up. Updates ``loop_metrics.recent_resign_fp_rates`` for the rolling
    loosen window.

    Returns the calibration info dict for logging.
    """
    holdouts_total = 0
    holdouts_with_signal = 0
    correct = 0
    resigned_total = 0
    for game_file in SELF_PLAY_GAMES_STATUS_PATH.glob(f"L{loop}_G*.json"):
        gdata = _safe_read_json(game_file)
        if not gdata or gdata.get("status") != "Completed":
            continue
        if gdata.get("termination") == "resigned":
            resigned_total += 1
        if not gdata.get("noresign_holdout"):
            continue
        holdouts_total += 1
        whr = gdata.get("would_have_resigned")
        if not whr:
            continue
        holdouts_with_signal += 1
        leader_idx = whr.get("leader")
        result = gdata.get("result_per_player") or []
        if leader_idx is None or not result:
            continue
        winner_idx = int(np.argmax(result))
        if winner_idx == int(leader_idx):
            correct += 1

    fp_rate: Optional[float] = None
    if holdouts_with_signal > 0:
        fp_rate = 1.0 - correct / holdouts_with_signal

    # Maintain the 3-iteration history for the loosen condition.
    if fp_rate is not None:
        loop_metrics.recent_resign_fp_rates.append(float(fp_rate))
        if len(loop_metrics.recent_resign_fp_rates) > 3:
            loop_metrics.recent_resign_fp_rates = loop_metrics.recent_resign_fp_rates[-3:]

    old_threshold = float(loop_config.resign_high_threshold)
    min_threshold = float(loop_config.resign_high_threshold_min)
    adjustment = 0.0
    if fp_rate is not None:
        if fp_rate > 0.05:
            adjustment = +0.05
        elif (
            len(loop_metrics.recent_resign_fp_rates) >= 3
            and all(rate < 0.02 for rate in loop_metrics.recent_resign_fp_rates[-3:])
        ):
            adjustment = -0.05
            # Clear history so we don't loosen again on the very next iteration
            # before observing 3 fresh low-fp readings at the new threshold.
            loop_metrics.recent_resign_fp_rates = []
    new_threshold = max(min_threshold, min(0.99, old_threshold + adjustment))

    if abs(new_threshold - old_threshold) > 1e-9:
        loop_config.resign_high_threshold = new_threshold
        # Persist back so subsequent workers + next iteration's
        # load_loop_config pick up the updated threshold.
        try:
            serializable = asdict(loop_config)
            serializable["training_config"] = loop_config.training_config.to_json()
            atomic_write_json(LOOP_CONFIG_PATH, serializable, indent=4)
        except Exception as e:
            LOGGER.warning(f"Failed to persist resign threshold update: {e}")

    info = {
        "iteration": loop,
        "holdouts_total": holdouts_total,
        "holdouts_with_signal": holdouts_with_signal,
        "correct_holdouts": correct,
        "fp_rate": fp_rate,
        "old_threshold": old_threshold,
        "new_threshold": float(loop_config.resign_high_threshold),
        "adjustment": adjustment,
        "resigned_games": resigned_total,
    }
    LOGGER.info(
        f"Loop {loop+1}: resign calibration — holdouts={holdouts_total} "
        f"(signal={holdouts_with_signal}, correct={correct}, fp_rate={fp_rate}); "
        f"threshold: {old_threshold:.3f} -> {info['new_threshold']:.3f} "
        f"(resigned={resigned_total})"
    )
    if metrics is not None:
        metrics.add_scalar("Resign/Holdouts_Total", holdouts_total, loop)
        metrics.add_scalar("Resign/Holdouts_With_Signal", holdouts_with_signal, loop)
        if fp_rate is not None:
            metrics.add_scalar("Resign/Holdout_FP_Rate", fp_rate, loop)
        metrics.add_scalar("Resign/High_Threshold", info["new_threshold"], loop)
        metrics.add_scalar("Resign/Resigned_Games", resigned_total, loop)
    return info


def _aggregate_selfplay_stats(
    loop: int,
    sp: SelfPlayIterationResult,
    metrics: Metrics,
    loop_metrics: "LoopMetrics",
) -> SelfPlayAggregateStats:
    """Compute aggregate stats from per-game JSONs and emit basic self-play TensorBoard scalars."""
    metrics.add_scalar("SelfPlay/Completed_Games_Total_for_Iteration", sp.games_completed, loop)
    loop_metrics.games_played_total += sp.games_completed

    avg_game_length = 0.0
    if sp.game_lengths:
        avg_game_length = sum(sp.game_lengths) / len(sp.game_lengths)
        metrics.add_scalar("SelfPlay/Avg_Game_Length", avg_game_length, loop)
        LOGGER.info(f"Loop {loop+1}: Average game length: {avg_game_length:.1f} moves")
    metrics.add_scalar("SelfPlay/Experiences_This_Iteration", sp.experiences, loop)
    LOGGER.info(f"Loop {loop+1}: Total experiences this iteration: {sp.experiences}")

    phase_counts = {"Auction": 0, "WaterfallAuction": 0, "Stock": 0, "Operating": 0, "Other": 0}
    for status_file in SELF_PLAY_GAMES_STATUS_PATH.glob(f"L{loop}_G*.json"):
        status_json = _safe_read_json(status_file)
        if status_json:
            for phase, count in status_json.get("phase_move_counts", {}).items():
                phase_counts[phase] = phase_counts.get(phase, 0) + count
    for phase, count in phase_counts.items():
        metrics.add_scalar(f"SelfPlay/Phase_Moves/{phase}", count, loop)
    LOGGER.info(f"Loop {loop+1}: Phase move counts: {phase_counts}")

    timing_sums = {k: 0.0 for k in _SELFPLAY_TIMING_KEYS}
    timing_count = 0
    total_sims = 0
    for game_file in SELF_PLAY_GAMES_STATUS_PATH.glob("*.json"):
        gdata = _safe_read_json(game_file)
        if not gdata:
            continue
        timing = gdata.get("timing")
        if timing and gdata.get("status") == "Completed":
            for k in _SELFPLAY_TIMING_KEYS:
                timing_sums[k] += timing.get(k, 0.0)
            total_sims += timing.get("total_sims", 0)
            timing_count += 1

    timing_avgs: dict = {}
    if timing_count > 0:
        for k in _SELFPLAY_TIMING_KEYS:
            timing_avgs[f"avg_{k}"] = round(timing_sums[k] / timing_count, 3)
        timing_avgs["avg_sims_per_game"] = round(total_sims / timing_count, 1)

    moves_per_second = sp.experiences / sp.wall_time if sp.wall_time > 0 else 0
    games_per_hour = sp.games_completed / (sp.wall_time / 3600) if sp.wall_time > 0 else 0

    return SelfPlayAggregateStats(
        avg_game_length=avg_game_length,
        min_game_length=min(sp.game_lengths) if sp.game_lengths else 0,
        max_game_length=max(sp.game_lengths) if sp.game_lengths else 0,
        games_per_hour=games_per_hour,
        moves_per_second=moves_per_second,
        phase_move_counts=phase_counts,
        timing_avgs=timing_avgs,
    )


def _run_training_iteration(
    loop: int,
    loop_config: "LoopConfig",
    metrics: Metrics,
) -> TrainingIterationResult:
    """Train the current-best model on the current self-play data.

    `train_model` now always saves a numbered checkpoint at the end of the run
    (regardless of gating outcome). The candidate's `checkpoint_num` is on
    `train_metrics.checkpoint_num` and gets passed to gating below; gating is
    responsible for moving the `current_best` pointer if the candidate wins.
    """
    training_config = loop_config.training_config
    if not isinstance(training_config, TrainingConfig):
        LOGGER.error(f"FIX THIS: Training config is not a TrainingConfig: {training_config}")
        training_config = TrainingConfig.from_json(training_config)

    training_config.metrics = metrics
    training_config.global_step = loop

    model = get_latest_model(MODEL_CHECKPOINT_DIR)
    training_config.train_dir = Path(f"training_examples/selfplay/{model.get_name()}")

    training_start_time = time.time()
    _, train_metrics = train(training_config, model, model_checkpoint_dir=MODEL_CHECKPOINT_DIR)
    return TrainingIterationResult(
        model=model,
        train_metrics=train_metrics,
        wall_time=time.time() - training_start_time,
    )


def _run_gating_iteration(
    loop: int,
    model,
    candidate_checkpoint_num,
    no_gate: bool,
    gate_games: int,
    gate_threshold: float,
    num_readouts: int,
    status: dict,
    metrics: Metrics,
) -> GatingIterationResult:
    """Evaluate the freshly-trained candidate and (maybe) promote it.

    The candidate checkpoint has already been written to disk by `train_model`
    (`candidate_checkpoint_num`). Gating's only side effect on disk is updating
    the per-arch `current_best.json` pointer — never `save_model`. A rejected
    candidate still lives on disk for later analysis but is not loaded by
    self-play workers because the pointer doesn't move.

    On the first iteration there is no incumbent to compare against, so we
    unconditionally promote to bootstrap the pointer. `--no-gate` does the
    same on every iteration.
    """
    gating_start_time = time.time()
    gate_win_rate = None
    arch = model.architecture_name()
    # The on-disk session directory is `<arch>/<timestamp>_<seed>/`; the pointer
    # records that name (not `model.get_name()`, which is the arch-prefixed form).
    session = session_name_for(model)

    if no_gate or loop == 0:
        if loop == 0:
            LOGGER.info("First iteration; skipping gating, promoting candidate to current_best.")
            reason = "first_iteration"
        else:
            LOGGER.info("Gating disabled (--no-gate); promoting candidate to current_best.")
            reason = "gating_disabled"
        promoted = True
        if candidate_checkpoint_num is not None:
            set_current_best(MODEL_CHECKPOINT_DIR, arch, session, candidate_checkpoint_num)
        else:
            LOGGER.warning("No candidate checkpoint to promote (train_model did not save).")
        append_model_history({
            "loop": loop + 1,
            "timestamp": datetime.now().isoformat(),
            "checkpoint_num": candidate_checkpoint_num,
            "architecture": arch,
            "session": session,
            "promoted": True,
            "win_rate": None,
            "gate_games": 0,
            "reason": reason,
        })
    else:
        status["status_message"] = f"Running gating evaluation ({gate_games} games)..."
        update_loop_status(status)

        # The current-best pointer points to the prior winner; the freshly
        # trained candidate sits at `candidate_checkpoint_num` in the same
        # session, but it is NOT pointed to yet.
        current_best_model = get_latest_model(MODEL_CHECKPOINT_DIR)
        gate_win_rate = evaluate_candidate(
            candidate_model=model,
            current_best_model=current_best_model,
            num_games=gate_games,
            num_readouts=min(num_readouts, 50),
        )
        metrics.add_scalar("Gating/Win_Rate", gate_win_rate, loop)
        metrics.add_scalar("Gating/Promoted", 1.0 if gate_win_rate >= gate_threshold else 0.0, loop)

        if gate_win_rate >= gate_threshold:
            LOGGER.info(f"Model promoted! Win rate: {gate_win_rate:.1%} >= {gate_threshold:.1%}")
            if candidate_checkpoint_num is not None:
                set_current_best(MODEL_CHECKPOINT_DIR, arch, session, candidate_checkpoint_num)
            else:
                LOGGER.warning("Gating passed but no candidate checkpoint to promote.")
            promoted = True
        else:
            LOGGER.info(
                f"Model rejected. Win rate: {gate_win_rate:.1%} < {gate_threshold:.1%}. "
                f"Candidate checkpoint {candidate_checkpoint_num} remains on disk; "
                f"current_best pointer unchanged."
            )
            promoted = False

        append_model_history({
            "loop": loop + 1,
            "timestamp": datetime.now().isoformat(),
            "checkpoint_num": candidate_checkpoint_num,
            "architecture": arch,
            "session": session,
            "promoted": promoted,
            "win_rate": gate_win_rate,
            "gate_games": gate_games,
            "reason": "gating",
        })

    return GatingIterationResult(
        win_rate=gate_win_rate,
        promoted=promoted,
        checkpoint_num=candidate_checkpoint_num,
        wall_time=time.time() - gating_start_time,
    )


def _last_or(lst, default=0.0):
    return lst[-1] if lst else default


def _oldest_training_example_age_minutes() -> float:
    """Wall-clock age (minutes) of the oldest LMDB ``data.mdb`` in the rolling
    self-play training window. Returns 0.0 if no LMDB files are found.

    Uses file mtime as a cheap proxy for "age of the oldest example". The LMDB
    is rewritten as new self-play games land in it, so its mtime tracks the
    most-recent write — the *minimum* mtime across all session LMDBs is the
    closest cheap proxy for the oldest data currently in the training window.
    """
    train_root = Path("training_examples/selfplay")
    if not train_root.exists():
        return 0.0
    mdb_files = list(train_root.rglob("data.mdb"))
    if not mdb_files:
        return 0.0
    now = time.time()
    oldest_mtime = min(f.stat().st_mtime for f in mdb_files)
    return max(0.0, (now - oldest_mtime) / 60.0)


def _emit_tensorboard_metrics(
    loop: int,
    sp_stats: SelfPlayAggregateStats,
    sp: SelfPlayIterationResult,
    training_wall_time: float,
    gating_wall_time: float,
    train_metrics,
    loop_metrics: "LoopMetrics",
    metrics: Metrics,
):
    """Write timing + training TensorBoard scalars for the iteration."""
    iteration_wall_time = sp.wall_time + training_wall_time + gating_wall_time

    LOGGER.info(
        f"Loop {loop+1} timing breakdown:\n"
        f"  Self-play:  {sp.wall_time:.1f}s ({sp.wall_time/iteration_wall_time*100:.0f}%)\n"
        f"  Training:   {training_wall_time:.1f}s ({training_wall_time/iteration_wall_time*100:.0f}%)\n"
        f"  Gating:     {gating_wall_time:.1f}s ({gating_wall_time/iteration_wall_time*100:.0f}%)\n"
        f"  Total:      {iteration_wall_time:.1f}s"
    )
    if sp_stats.timing_avgs:
        ta = sp_stats.timing_avgs
        LOGGER.info(
            f"Loop {loop+1} self-play timing (avg per game):\n"
            f"  Tree search:     {ta.get('avg_tree_search_s', 0):.1f}s\n"
            f"    Inference:     {ta.get('avg_inference_s', 0):.1f}s\n"
            f"    Encoding:      {ta.get('avg_encoding_s', 0):.1f}s\n"
            f"    Leaf select:   {ta.get('avg_leaf_selection_s', 0):.1f}s\n"
            f"    Backup:        {ta.get('avg_backup_s', 0):.1f}s\n"
            f"  Pick move:       {ta.get('avg_pick_move_s', 0):.1f}s\n"
            f"  Play move:       {ta.get('avg_play_move_s', 0):.1f}s\n"
            f"  Data extraction: {ta.get('avg_data_extraction_s', 0):.1f}s\n"
            f"  Total game:      {ta.get('avg_total_game_s', 0):.1f}s\n"
            f"  Avg sims/game:   {ta.get('avg_sims_per_game', 0):.0f}\n"
            f"  Moves/sec:       {sp_stats.moves_per_second:.1f}"
        )

    metrics.add_scalar("Timing/Selfplay_Wall_Time_s", sp.wall_time, loop)
    metrics.add_scalar("Timing/Training_Wall_Time_s", training_wall_time, loop)
    metrics.add_scalar("Timing/Gating_Wall_Time_s", gating_wall_time, loop)
    metrics.add_scalar("Timing/Total_Iteration_s", iteration_wall_time, loop)
    metrics.add_scalar("Timing/Moves_Per_Second", sp_stats.moves_per_second, loop)
    for k, v in sp_stats.timing_avgs.items():
        metrics.add_scalar(f"Timing/SelfPlay_{k}", v, loop)

    # System / GPU + training-data freshness
    gpu_memory_allocated_mb = (
        torch.cuda.memory_allocated() / 1e6 if torch.cuda.is_available() else 0
    )
    gpu_max_memory_mb = (
        torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0
    )
    metrics.add_scalar("system/gpu_memory_allocated_mb", gpu_memory_allocated_mb, loop)
    metrics.add_scalar("system/gpu_max_memory_mb", gpu_max_memory_mb, loop)
    metrics.add_scalar(
        "training/oldest_example_age_minutes",
        _oldest_training_example_age_minutes(),
        loop,
    )

    if not (train_metrics and train_metrics.epochs_trained > 0):
        return

    loop_metrics.training_losses.append(train_metrics.avg_total_loss)
    loop_metrics.policy_losses.append(train_metrics.avg_policy_loss)
    loop_metrics.value_losses.append(train_metrics.avg_value_loss)
    loop_metrics.training_examples_total += train_metrics.training_examples

    metrics.add_scalar("Training/Total_Loss", train_metrics.avg_total_loss, loop)
    metrics.add_scalar("Training/Policy_Loss", train_metrics.avg_policy_loss, loop)
    metrics.add_scalar("Training/Value_Loss", train_metrics.avg_value_loss, loop)
    metrics.add_scalar("Training/Examples_This_Iteration", train_metrics.training_examples, loop)
    metrics.add_scalar("Training/Examples_Total", loop_metrics.training_examples_total, loop)

    for epoch_idx, epoch_loss in enumerate(train_metrics.epoch_losses):
        metrics.add_scalar(f"Training/Epoch_Loss/Loop{loop}", epoch_loss, epoch_idx)

    metrics.add_scalar("Training/Entropy", _last_or(train_metrics.epoch_entropy), loop)
    metrics.add_scalar("Training/Aux_Loss", _last_or(train_metrics.epoch_aux_losses), loop)

    metrics.add_scalar("Policy/Top1_Accuracy", _last_or(train_metrics.epoch_top1_accuracy), loop)
    metrics.add_scalar("Policy/Top5_Accuracy", _last_or(train_metrics.epoch_top5_accuracy), loop)
    metrics.add_scalar("Policy/KL_Divergence", _last_or(train_metrics.epoch_policy_kl), loop)
    metrics.add_scalar("Policy/Network_Entropy", _last_or(train_metrics.epoch_policy_entropy), loop)
    metrics.add_scalar("Policy/MCTS_Target_Entropy", _last_or(train_metrics.epoch_target_entropy), loop)
    metrics.add_scalar("Policy/Max_Prob_Concentration", _last_or(train_metrics.epoch_legal_move_concentration), loop)
    metrics.add_scalar("Policy/Mean_Legal_Actions", _last_or(train_metrics.epoch_mean_legal_actions), loop)

    metrics.add_scalar("Value/Explained_Variance", _last_or(train_metrics.epoch_value_explained_variance), loop)
    metrics.add_scalar("Value/Correlation", _last_or(train_metrics.epoch_value_correlation), loop)
    metrics.add_scalar("Value/MAE", _last_or(train_metrics.epoch_value_mae), loop)
    metrics.add_scalar("Value/MSE", _last_or(train_metrics.epoch_value_mse), loop)
    metrics.add_scalar("Value/Pred_Mean", _last_or(train_metrics.epoch_value_pred_mean), loop)
    metrics.add_scalar("Value/Pred_Std", _last_or(train_metrics.epoch_value_pred_std), loop)
    metrics.add_scalar("Value/Pred_Min", _last_or(train_metrics.epoch_value_pred_min), loop)
    metrics.add_scalar("Value/Pred_Max", _last_or(train_metrics.epoch_value_pred_max), loop)
    metrics.add_scalar("Value/Target_Mean", _last_or(train_metrics.epoch_value_target_mean), loop)
    metrics.add_scalar("Value/Target_Std", _last_or(train_metrics.epoch_value_target_std), loop)
    metrics.add_scalar("Value/Target_Min", _last_or(train_metrics.epoch_value_target_min), loop)
    metrics.add_scalar("Value/Target_Max", _last_or(train_metrics.epoch_value_target_max), loop)

    metrics.add_scalar("Gradients/Total_Norm", _last_or(train_metrics.epoch_grad_norm_total), loop)
    metrics.add_scalar("Gradients/Policy_Head_Norm", _last_or(train_metrics.epoch_grad_norm_policy_head), loop)
    metrics.add_scalar("Gradients/Value_Head_Norm", _last_or(train_metrics.epoch_grad_norm_value_head), loop)
    metrics.add_scalar("Gradients/Trunk_Norm", _last_or(train_metrics.epoch_grad_norm_trunk), loop)
    metrics.add_scalar("Gradients/CV", _last_or(train_metrics.epoch_grad_norm_cv), loop)

    metrics.add_scalar("Training/Learning_Rate", _last_or(train_metrics.epoch_lr), loop)
    metrics.add_scalar("Aux/Pred_Mean", _last_or(train_metrics.epoch_aux_pred_mean), loop)
    metrics.add_scalar("Aux/Target_Mean", _last_or(train_metrics.epoch_aux_target_mean), loop)
    metrics.add_scalar("Aux/Correlation", _last_or(train_metrics.epoch_aux_correlation), loop)

    LOGGER.info(
        f"Loop {loop+1} training metrics - Total Loss: {train_metrics.avg_total_loss:.4f}, "
        f"Policy Loss: {train_metrics.avg_policy_loss:.4f}, Value Loss: {train_metrics.avg_value_loss:.4f}, "
        f"Examples: {train_metrics.training_examples}"
    )


def _build_history_record(
    loop: int,
    model,
    train_metrics,
    sp: SelfPlayIterationResult,
    sp_stats: SelfPlayAggregateStats,
    gating: GatingIterationResult,
    loop_config: "LoopConfig",
    scheduled_game_length: int,
    scheduled_readouts: int,
    checkpoint_count: int,
    gate_games: int,
    loop_metrics: "LoopMetrics",
    training_wall_time: float,
) -> dict:
    """Build the dashboard history JSONL record for an iteration."""
    iteration_wall_time = sp.wall_time + training_wall_time + gating.wall_time
    has_train = bool(train_metrics and train_metrics.epochs_trained > 0)
    return {
        "loop": loop + 1,
        "timestamp": datetime.now().isoformat(),
        "model_session": model.get_name(),
        "model_architecture": model.architecture_name(),
        "promoted": gating.promoted,
        # Losses
        "total_loss": train_metrics.avg_total_loss if has_train else None,
        "policy_loss": train_metrics.avg_policy_loss if has_train else None,
        "value_loss": train_metrics.avg_value_loss if has_train else None,
        "entropy": _last_or(train_metrics.epoch_entropy) if train_metrics else 0.0,
        "aux_loss": _last_or(train_metrics.epoch_aux_losses) if train_metrics else 0.0,
        # Per-epoch granularity
        "epoch_losses": list(train_metrics.epoch_losses) if train_metrics else [],
        "epoch_policy_losses": list(train_metrics.epoch_policy_losses) if train_metrics else [],
        "epoch_value_losses": list(train_metrics.epoch_value_losses) if train_metrics else [],
        # Policy health
        "top1_accuracy": _last_or(train_metrics.epoch_top1_accuracy) if train_metrics else 0.0,
        "top5_accuracy": _last_or(train_metrics.epoch_top5_accuracy) if train_metrics else 0.0,
        "policy_entropy": _last_or(train_metrics.epoch_policy_entropy) if train_metrics else 0.0,
        "mcts_target_entropy": _last_or(train_metrics.epoch_target_entropy) if train_metrics else 0.0,
        "policy_kl": _last_or(train_metrics.epoch_policy_kl) if train_metrics else 0.0,
        "max_prob_concentration": _last_or(train_metrics.epoch_legal_move_concentration) if train_metrics else 0.0,
        "mean_legal_actions": _last_or(train_metrics.epoch_mean_legal_actions) if train_metrics else 0.0,
        # Value health
        "value_explained_variance": _last_or(train_metrics.epoch_value_explained_variance) if train_metrics else 0.0,
        "value_correlation": _last_or(train_metrics.epoch_value_correlation) if train_metrics else 0.0,
        "value_mae": _last_or(train_metrics.epoch_value_mae) if train_metrics else 0.0,
        "value_mse": _last_or(train_metrics.epoch_value_mse) if train_metrics else 0.0,
        "value_pred_mean": _last_or(train_metrics.epoch_value_pred_mean) if train_metrics else 0.0,
        "value_pred_std": _last_or(train_metrics.epoch_value_pred_std) if train_metrics else 0.0,
        "value_pred_min": _last_or(train_metrics.epoch_value_pred_min) if train_metrics else 0.0,
        "value_pred_max": _last_or(train_metrics.epoch_value_pred_max) if train_metrics else 0.0,
        "value_target_mean": _last_or(train_metrics.epoch_value_target_mean) if train_metrics else 0.0,
        "value_target_std": _last_or(train_metrics.epoch_value_target_std) if train_metrics else 0.0,
        "value_target_min": _last_or(train_metrics.epoch_value_target_min) if train_metrics else 0.0,
        "value_target_max": _last_or(train_metrics.epoch_value_target_max) if train_metrics else 0.0,
        # Gradients
        "grad_norm_total": _last_or(train_metrics.epoch_grad_norm_total) if train_metrics else 0.0,
        "grad_norm_policy": _last_or(train_metrics.epoch_grad_norm_policy_head) if train_metrics else 0.0,
        "grad_norm_value": _last_or(train_metrics.epoch_grad_norm_value_head) if train_metrics else 0.0,
        "grad_norm_trunk": _last_or(train_metrics.epoch_grad_norm_trunk) if train_metrics else 0.0,
        "grad_norm_cv": _last_or(train_metrics.epoch_grad_norm_cv) if train_metrics else 0.0,
        # Self-play
        "games_played": sp.games_completed,
        "experiences": sp.experiences,
        "avg_game_length": sp_stats.avg_game_length,
        "min_game_length": sp_stats.min_game_length,
        "max_game_length": sp_stats.max_game_length,
        "total_experiences_cumulative": loop_metrics.training_examples_total,
        "selfplay_wall_time_seconds": round(sp.wall_time, 1),
        "selfplay_games_per_hour": round(sp_stats.games_per_hour, 1),
        "phase_move_counts": sp_stats.phase_move_counts,
        # Gating
        "gate_win_rate": gating.win_rate,
        "gate_games_played": gate_games if gating.win_rate is not None else 0,
        # Training config snapshot
        "learning_rate": _last_or(train_metrics.epoch_lr) if train_metrics else loop_config.training_config.lr,
        "batch_size": loop_config.training_config.batch_size,
        "num_epochs": loop_config.training_config.num_epochs,
        "num_readouts": scheduled_readouts,
        "scheduled_max_game_length": scheduled_game_length,
        "checkpoint_count": checkpoint_count,
        "training_examples": train_metrics.training_examples if train_metrics else 0,
        # Aux diagnostics
        "aux_pred_mean": _last_or(train_metrics.epoch_aux_pred_mean) if train_metrics else 0.0,
        "aux_target_mean": _last_or(train_metrics.epoch_aux_target_mean) if train_metrics else 0.0,
        "aux_correlation": _last_or(train_metrics.epoch_aux_correlation) if train_metrics else 0.0,
        # Performance timing
        "training_wall_time_seconds": round(training_wall_time, 1),
        "gating_wall_time_seconds": round(gating.wall_time, 1),
        "total_iteration_time_seconds": round(iteration_wall_time, 1),
        "selfplay_moves_per_second": round(sp_stats.moves_per_second, 1),
        # Self-play timing breakdown (avg per game)
        **{f"selfplay_{k}": v for k, v in sp_stats.timing_avgs.items()},
    }


def main(
    num_loop_iterations: int,
    num_threads: int,
    cleanup: bool,
    num_readouts: int,
    num_epochs: int = 3,
    max_training_window: int = 100000,
    gate_games: int = 10,
    gate_threshold: float = 0.55,
    no_gate: bool = False,
    model_type: str = "transformer",
    fresh: bool = False,
    target_experiences: int = 10000,
    batch_size: int = 256,
    game_length_schedule: tuple[int, int, int] = (150, 1000, 150),
    readout_schedule: tuple[int, int, int] = (64, 200, 150),
):
    if cleanup:
        cleanup_files()

    if LOOP_LOCK_FILE.exists():
        LOGGER.error("Loop already running. Exiting.")
        exit(1)
    LOOP_LOCK_FILE.touch()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    setup_logging(logging.INFO, f"logs/loop/loop_{timestamp}.log", console=True)

    if fresh:
        LOGGER.info("Fresh start requested. Clearing model checkpoints and training data.")
        cleanup_model_and_data()

    ensure_seed_model(model_type)

    tb_log_dir = os.path.join(TENSORBOARD_LOG_DIR_BASE, f"experiment_{timestamp}")
    metrics = Metrics(tb_log_dir)
    self_play_logs_path = Path("logs/self_play")
    self_play_logs_path.mkdir(parents=True, exist_ok=True)
    default_training_config = TrainingConfig(batch_size=batch_size, num_epochs=num_epochs, max_training_window=max_training_window)

    # Initialize loop metrics
    loop_metrics_path = Path(f"logs/loop/loop_metrics_{timestamp}.json")
    loop_metrics = LoopMetrics()

    # Register signal handlers
    signal.signal(signal.SIGINT, cleanup_and_exit)
    signal.signal(signal.SIGTERM, cleanup_and_exit)
    atexit.register(cleanup_and_exit)

    # Phase 3: optionally spawn the cross-process inference server before
    # the main loop. Persistent across iterations; paused while training
    # runs in this process (so the GPU isn't contended), then reloaded
    # from the newly-saved checkpoint.
    inference_server_handle = None
    initial_loop_config = load_loop_config(
        num_loop_iterations,
        max(1, target_experiences // game_length_schedule[0]),
        num_threads,
        default_training_config,
        readout_schedule[0],
        target_experiences,
    )
    if initial_loop_config.use_inference_server:
        from rl18xx.agent.alphazero.inference_server import start_inference_server
        LOGGER.info(
            f"Spawning inference server (batch_size={initial_loop_config.inference_batch_size}, "
            f"batch_timeout_ms={initial_loop_config.inference_batch_timeout_ms})"
        )
        # ``num_workers`` is the maximum concurrent self-play workers, used
        # for the per-worker reply-queue allocation.
        inference_server_handle = start_inference_server(
            num_workers=num_threads,
            model_factory=_server_model_factory,
            checkpoint_path=None,
            batch_size=initial_loop_config.inference_batch_size,
            batch_timeout_ms=initial_loop_config.inference_batch_timeout_ms,
            autocast_device="cuda" if torch.cuda.is_available() else None,
        )

    loop = 0
    try:
        while True:
            LOGGER.info(f"--- Starting loop {loop+1}/{num_loop_iterations} ---")

            # Compute scheduled values based on persistent checkpoint count
            checkpoint_count = _get_checkpoint_count()
            scheduled_game_length = get_scheduled_value(checkpoint_count, game_length_schedule)
            scheduled_readouts = get_scheduled_value(checkpoint_count, readout_schedule)
            LOGGER.info(
                f"Loop {loop+1}: Schedule (checkpoint {checkpoint_count}): "
                f"max_game_length={scheduled_game_length}, num_readouts={scheduled_readouts}"
            )

            num_games_estimate = max(1, target_experiences // scheduled_game_length)
            loop_config = load_loop_config(
                num_loop_iterations,
                num_games_estimate,
                num_threads,
                default_training_config,
                scheduled_readouts,
                target_experiences,
            )

            if loop_config.num_loop_iterations > 0 and loop >= loop_config.num_loop_iterations:
                break

            LOGGER.info(
                f"Loop {loop+1}: Targeting {loop_config.target_experiences} experiences "
                f"(~{num_games_estimate} games)."
            )
            LOGGER.info(f"Loop {loop+1}: Using training params: {loop_config.training_config}")
            LOGGER.info(f"Loop {loop+1}: Using {loop_config.num_threads} processes for self-play.")
            loop_metrics.loop_iteration = loop

            status = {
                "current_loop": loop + 1,
                "target_loop_count": loop_config.num_loop_iterations,
                "games_completed_this_iteration": 0,
                "total_games_this_iteration": loop_config.num_games_per_iteration,
                "status_message": "Running self-play games",
                "tensorboard_log_dir": tb_log_dir,
                "loop_metrics": {
                    "total_games_played": loop_metrics.games_played_total,
                    "total_training_examples": loop_metrics.training_examples_total,
                    "avg_training_loss": sum(loop_metrics.training_losses[-10:])
                    / len(loop_metrics.training_losses[-10:])
                    if loop_metrics.training_losses
                    else 0,
                    "avg_policy_loss": sum(loop_metrics.policy_losses[-10:]) / len(loop_metrics.policy_losses[-10:])
                    if loop_metrics.policy_losses
                    else 0,
                    "avg_value_loss": sum(loop_metrics.value_losses[-10:]) / len(loop_metrics.value_losses[-10:])
                    if loop_metrics.value_losses
                    else 0,
                },
            }
            update_loop_status(status)

            sp = _run_selfplay_iteration(
                loop, loop_config, scheduled_game_length, tb_log_dir, timestamp, loop_metrics,
                server_handle=inference_server_handle,
            )
            sp_stats = _aggregate_selfplay_stats(loop, sp, metrics, loop_metrics)
            # Phase 2: auto-calibrate the resign high threshold from this
            # iteration's holdout games (no-op until enough holdouts have
            # accumulated). Mutates loop_config + persists to disk so the
            # next iteration's workers pick up the new threshold.
            if loop_config.enable_resign:
                calibrate_resign_threshold(loop, loop_config, loop_metrics, metrics)

            status["status_message"] = "Self-play phase completed. Starting training."
            update_loop_status(status)
            LOGGER.info(f"--- Starting training on self-play data ({sp.games_completed} games) ---")

            # Phase 3: pause the inference server so it drops its model
            # and frees VRAM before training takes over the GPU.
            if inference_server_handle is not None:
                try:
                    inference_server_handle.pause()
                    LOGGER.info("Inference server paused before training")
                except Exception as e:
                    LOGGER.warning(f"Inference server pause failed (continuing): {e}")

            train_result = _run_training_iteration(loop, loop_config, metrics)
            model = train_result.model
            train_metrics = train_result.train_metrics
            training_wall_time = train_result.wall_time
            candidate_checkpoint_num = (
                train_metrics.checkpoint_num if train_metrics is not None else None
            )

            gating = _run_gating_iteration(
                loop, model, candidate_checkpoint_num, no_gate, gate_games, gate_threshold,
                num_readouts, status, metrics,
            )

            # Phase 3: reload the inference server from the (possibly
            # gated-and-promoted) checkpoint so the next iteration's
            # workers infer against the latest weights.
            if inference_server_handle is not None:
                try:
                    inference_server_handle.reload(None)
                    LOGGER.info("Inference server reloaded with latest checkpoint")
                except Exception as e:
                    LOGGER.warning(f"Inference server reload failed (continuing): {e}")

            _emit_tensorboard_metrics(
                loop, sp_stats, sp, training_wall_time, gating.wall_time,
                train_metrics, loop_metrics, metrics,
            )

            LOGGER.info(f"--- Finished training on self-play data ---")

            history_record = _build_history_record(
                loop, model, train_metrics, sp, sp_stats, gating,
                loop_config, scheduled_game_length, scheduled_readouts,
                checkpoint_count, gate_games, loop_metrics, training_wall_time,
            )
            append_metrics_history(history_record)

            status["status_message"] = f"Loop {loop+1} training complete. Preparing for next loop."
            status["loop_metrics"].update(
                {
                    "total_games_played": loop_metrics.games_played_total,
                    "total_training_examples": loop_metrics.training_examples_total,
                    "latest_training_loss": loop_metrics.training_losses[-1] if loop_metrics.training_losses else None,
                    "latest_policy_loss": loop_metrics.policy_losses[-1] if loop_metrics.policy_losses else None,
                    "latest_value_loss": loop_metrics.value_losses[-1] if loop_metrics.value_losses else None,
                }
            )
            update_loop_status(status)
            if num_loop_iterations > 0:
                metrics.add_scalar("Loop/Progress", (loop + 1) / num_loop_iterations * 100, loop)

            # Save loop metrics after each iteration
            save_loop_metrics(loop_metrics, loop_metrics_path)

            gc.collect()
            loop += 1
    except Exception as e:
        LOGGER.error(f"Error in loop: {e}", exc_info=True)
        status["status_message"] = f"Error in loop: {e}"
        update_loop_status(status)
        raise
    finally:
        if inference_server_handle is not None:
            try:
                inference_server_handle.shutdown()
                LOGGER.info("Inference server shut down")
            except Exception as e:
                LOGGER.warning(f"Inference server shutdown error (ignoring): {e}")
        if LOOP_LOCK_FILE.exists():
            LOOP_LOCK_FILE.unlink()

    LOGGER.info("--- Finished all loops ---")

    # Final save of loop metrics
    save_loop_metrics(loop_metrics, loop_metrics_path)
    LOGGER.info(f"Loop metrics saved to {loop_metrics_path}")

    # Log summary statistics
    if loop_metrics.training_losses:
        LOGGER.info(
            f"Final average training loss: {sum(loop_metrics.training_losses) / len(loop_metrics.training_losses):.4f}"
        )
    if loop_metrics.game_lengths:
        LOGGER.info(
            f"Average game length across all loops: {sum(loop_metrics.game_lengths) / len(loop_metrics.game_lengths):.1f} moves"
        )
    LOGGER.info(f"Total games played: {loop_metrics.games_played_total}")
    LOGGER.info(f"Total training examples: {loop_metrics.training_examples_total}")

    metrics.close()
    exit(0)


if __name__ == "__main__":
    import argparse
    import multiprocessing

    multiprocessing.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser(description="Run AlphaZero training loop")
    parser.add_argument(
        "--num_loop_iterations", type=int, default=5, help="Number of iterations of the full training loop"
    )
    parser.add_argument(
        "--target_experiences", type=int, default=10000,
        help="Target experience count per iteration (default: 10000)"
    )
    parser.add_argument("--num_threads", type=int, default=2, help="Number of threads to use for self-play")
    parser.add_argument(
        "--keep-old-files",
        action="store_true",
        help="Keep old files from previous runs. By default, old files are deleted.",
    )
    parser.add_argument("--num_readouts", type=int, default=200, help="Number of readouts to use for self-play")
    parser.add_argument("--batch_size", type=int, default=256, help="Training batch size (default: 256)")
    parser.add_argument("--num-epochs", type=int, default=3, help="Training epochs per iteration")
    parser.add_argument(
        "--max_training_window", type=int, default=100000, help="Max training examples to use (0 = all data, default: 100000)"
    )
    parser.add_argument(
        "--gate-games", type=int, default=10, help="Number of arena games for model gating (default: 10)"
    )
    parser.add_argument(
        "--gate-threshold", type=float, default=0.55, help="Minimum win rate to promote model (default: 0.55)"
    )
    parser.add_argument("--no-gate", action="store_true", help="Disable model gating (always promote)")
    parser.add_argument(
        "--model-type", type=str, default="transformer", choices=["gnn", "transformer"],
        help="Model architecture for initial checkpoint (default: transformer)"
    )
    parser.add_argument(
        "--fresh", action="store_true",
        help="Clear all model checkpoints and training data to start from scratch"
    )
    args = parser.parse_args()

    main(
        num_loop_iterations=args.num_loop_iterations,
        num_threads=args.num_threads,
        cleanup=not args.keep_old_files,
        num_readouts=args.num_readouts,
        num_epochs=args.num_epochs,
        max_training_window=args.max_training_window,
        gate_games=args.gate_games,
        gate_threshold=args.gate_threshold,
        no_gate=args.no_gate,
        model_type=args.model_type,
        fresh=args.fresh,
        target_experiences=args.target_experiences,
        batch_size=args.batch_size,
    )
