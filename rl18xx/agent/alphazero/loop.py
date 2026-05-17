from concurrent.futures import as_completed, ProcessPoolExecutor
import os
import json
import logging
import gc
import shutil
import time
from datetime import datetime
from dataclasses import dataclass, asdict
from rl18xx.agent.alphazero.checkpointer import get_latest_model, save_model, save_optimizer_state
from rl18xx.agent.alphazero.config import SelfPlayConfig, TrainingConfig
from rl18xx.agent.alphazero.metrics import Metrics
from rl18xx.agent.alphazero.self_play import MCTSPlayer, SelfPlay, SELF_PLAY_GAMES_STATUS_PATH
from rl18xx.agent.alphazero.train import train
from rl18xx.game.gamemap import GameMap
from pathlib import Path
import signal
import sys
import psutil
import atexit
import random

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

    # If config file exists, merge training hyperparams from it (hot-reload support)
    if LOOP_CONFIG_PATH.exists():
        try:
            with open(LOOP_CONFIG_PATH, "r") as f:
                file_config = json.load(f)
            if "training_config" in file_config:
                training_config = TrainingConfig.from_json(file_config["training_config"])
        except Exception as e:
            LOGGER.warning(f"Error reading loop config file: {e}. Using CLI defaults.")

    loop_config = LoopConfig(
        num_loop_iterations=num_loop_iterations,
        num_games_per_iteration=num_games_per_iteration,
        num_threads=num_threads,
        training_config=training_config,
        num_readouts=num_readouts,
        target_experiences=target_experiences,
    )

    # Write current config to file for visibility / hot-reload editing
    serializable = asdict(loop_config)
    serializable["training_config"] = training_config.to_json()
    with open(LOOP_CONFIG_PATH, "w") as f:
        json.dump(serializable, f, indent=4)

    return loop_config


def update_loop_status(status_data: dict):
    """Writes the current loop status to a JSON file."""
    try:
        with open(LOOP_STATUS_PATH, "w") as f:
            json.dump(status_data, f, indent=4, default=str)
    except Exception as e:
        LOGGER.error(f"Error writing to {LOOP_STATUS_PATH}: {e}")


def save_loop_metrics(metrics: LoopMetrics, filepath: Path):
    """Saves the loop metrics to a JSON file."""
    try:
        metrics_dict = asdict(metrics)
        with open(filepath, "w") as f:
            json.dump(metrics_dict, f, indent=4)
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
    for item in TENSORBOARD_LOG_DIR_BASE.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()
    for item in SELF_PLAY_GAMES_STATUS_PATH.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()


def run_self_play(game_idx_in_iteration: int, tb_log_dir: str, timestamp: str, loop: int, num_readouts: int = 32, max_game_length: int = 1000):
    process_root_logger = logging.getLogger()
    random.seed(os.getpid())
    for handler in process_root_logger.handlers[:]:
        process_root_logger.removeHandler(handler)
        handler.close()
    process_root_logger.setLevel(logging.INFO)
    game_log_file = SELF_PLAY_LOGS_PATH / f"{timestamp}_game_L{loop}_G{game_idx_in_iteration}.log"
    file_handler = logging.FileHandler(game_log_file)
    log_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(log_formatter)
    process_root_logger.addHandler(file_handler)
    logging.info(f"Self-play process started for L{loop}/G{game_idx_in_iteration}. Logging to {game_log_file}")

    model = get_latest_model("model_checkpoints")
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


def _create_fresh_game():
    """Create a new 4-player 1830 game instance."""
    game_map = GameMap()
    game_class = game_map.game_by_title("1830")
    players = {1: "Player 1", 2: "Player 2", 3: "Player 3", 4: "Player 4"}
    return game_class(players)


def _play_gate_game(candidate_model, current_best_model, game_index: int, num_readouts: int) -> dict:
    """Play a single gating arena game and return result info.

    Even-indexed games: candidate gets seats 0,1; best gets seats 2,3.
    Odd-indexed games: best gets seats 0,1; candidate gets seats 2,3.

    Returns a dict with 'candidate_seats', 'winner_seat', and 'scores',
    or None if the game crashed.
    """
    eval_config_candidate = SelfPlayConfig(
        softpick_move_cutoff=0,
        dirichlet_noise_weight=0,
        num_readouts=num_readouts,
        network=candidate_model,
    )
    eval_config_best = SelfPlayConfig(
        softpick_move_cutoff=0,
        dirichlet_noise_weight=0,
        num_readouts=num_readouts,
        network=current_best_model,
    )

    if game_index % 2 == 0:
        agents = [
            MCTSPlayer(eval_config_candidate),
            MCTSPlayer(eval_config_candidate),
            MCTSPlayer(eval_config_best),
            MCTSPlayer(eval_config_best),
        ]
        candidate_seats = {0, 1}
    else:
        agents = [
            MCTSPlayer(eval_config_best),
            MCTSPlayer(eval_config_best),
            MCTSPlayer(eval_config_candidate),
            MCTSPlayer(eval_config_candidate),
        ]
        candidate_seats = {2, 3}

    game_state = _create_fresh_game()
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

    Each game has 4 players: 2 with the candidate model, 2 with the current best.
    Positions alternate between games. A win is counted when the overall game
    winner occupies a candidate seat.

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


def ensure_seed_model(model_type: str = "v2"):
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
    if model_type == "v2":
        from rl18xx.agent.alphazero.config import ModelV2Config
        from rl18xx.agent.alphazero.model_v2 import AlphaZeroV2Model
        import torch

        config = ModelV2Config()
        torch.manual_seed(config.seed)
        LOGGER.info(f"Seeding model initialization with seed={config.seed}")
        model = AlphaZeroV2Model(config)
    else:
        from rl18xx.agent.alphazero.model import AlphaZeroGNNModel
        from rl18xx.agent.alphazero.config import ModelConfig
        import torch

        config = ModelConfig()
        torch.manual_seed(config.seed)
        LOGGER.info(f"Seeding model initialization with seed={config.seed}")
        model = AlphaZeroGNNModel(config)

    checkpoint_num = save_model(model, MODEL_CHECKPOINT_DIR)
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
    model_type: str = "v2",
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

            games_completed_count = 0
            game_lengths_this_iteration = []
            experiences_this_iteration = 0
            game_idx = 0
            selfplay_start_time = time.time()

            LOGGER.info(
                f"Loop {loop+1}: Running self-play until {loop_config.target_experiences} experiences "
                f"using {loop_config.num_threads} processes..."
            )
            executor = ProcessPoolExecutor(max_workers=loop_config.num_threads)
            try:
                pending_futures = {}
                # Submit initial batch
                for _ in range(loop_config.num_threads):
                    f = executor.submit(run_self_play, game_idx, tb_log_dir, timestamp, loop, loop_config.num_readouts, scheduled_game_length)
                    pending_futures[f] = game_idx
                    game_idx += 1

                while experiences_this_iteration < loop_config.target_experiences and pending_futures:
                    done_futures = []
                    for f in list(pending_futures.keys()):
                        if f.done():
                            done_futures.append(f)

                    if not done_futures:
                        import concurrent.futures

                        completed, _ = concurrent.futures.wait(
                            pending_futures.keys(), return_when=concurrent.futures.FIRST_COMPLETED
                        )
                        done_futures = list(completed)

                    for f in done_futures:
                        gidx = pending_futures.pop(f)
                        try:
                            f.result()
                            games_completed_count += 1
                            game_file = SELF_PLAY_GAMES_STATUS_PATH / f"L{loop}_G{gidx}.json"
                            gdata = _safe_read_json(game_file) if game_file.exists() else None
                            if gdata:
                                moves = gdata.get("moves_played", 0)
                                experiences_this_iteration += moves
                                game_lengths_this_iteration.append(moves)
                                loop_metrics.game_lengths.append(moves)

                            LOGGER.info(
                                f"Loop {loop+1}: Game {games_completed_count} completed. "
                                f"Experiences: {experiences_this_iteration}/{loop_config.target_experiences}"
                            )
                        except Exception as e:
                            LOGGER.error(f"Error in self-play game {gidx}: {e}", exc_info=True)

                        # Submit another game if we haven't reached target
                        if experiences_this_iteration < loop_config.target_experiences:
                            f_new = executor.submit(
                                run_self_play, game_idx, tb_log_dir, timestamp, loop, loop_config.num_readouts, scheduled_game_length
                            )
                            pending_futures[f_new] = game_idx
                            game_idx += 1

                # Wait for any remaining in-flight games
                for f in list(pending_futures.keys()):
                    try:
                        f.result()
                        gidx = pending_futures[f]
                        games_completed_count += 1
                        game_file = SELF_PLAY_GAMES_STATUS_PATH / f"L{loop}_G{gidx}.json"
                        gdata = _safe_read_json(game_file) if game_file.exists() else None
                        if gdata:
                            moves = gdata.get("moves_played", 0)
                            experiences_this_iteration += moves
                            game_lengths_this_iteration.append(moves)
                            loop_metrics.game_lengths.append(moves)
                    except Exception as e:
                        LOGGER.error(f"Error in trailing self-play game: {e}", exc_info=True)
            finally:
                executor.shutdown(wait=True)

            selfplay_wall_time = time.time() - selfplay_start_time

            metrics.add_scalar("SelfPlay/Completed_Games_Total_for_Iteration", games_completed_count, loop)
            loop_metrics.games_played_total += games_completed_count

            avg_game_length = 0.0
            if game_lengths_this_iteration:
                avg_game_length = sum(game_lengths_this_iteration) / len(game_lengths_this_iteration)
                metrics.add_scalar("SelfPlay/Avg_Game_Length", avg_game_length, loop)
                LOGGER.info(f"Loop {loop+1}: Average game length: {avg_game_length:.1f} moves")
            metrics.add_scalar("SelfPlay/Experiences_This_Iteration", experiences_this_iteration, loop)
            LOGGER.info(f"Loop {loop+1}: Total experiences this iteration: {experiences_this_iteration}")

            # Aggregate phase move counts across games (Item 6)
            aggregated_phase_counts = {"Auction": 0, "WaterfallAuction": 0, "Stock": 0, "Operating": 0, "Other": 0}
            for status_file in SELF_PLAY_GAMES_STATUS_PATH.glob(f"L{loop}_G*.json"):
                status_json = _safe_read_json(status_file)
                if status_json:
                    for phase, count in status_json.get("phase_move_counts", {}).items():
                        aggregated_phase_counts[phase] = aggregated_phase_counts.get(phase, 0) + count
            for phase, count in aggregated_phase_counts.items():
                metrics.add_scalar(f"SelfPlay/Phase_Moves/{phase}", count, loop)
            LOGGER.info(f"Loop {loop+1}: Phase move counts: {aggregated_phase_counts}")

            # Collect aggregate self-play stats from game JSONs
            selfplay_min_game_length = min(game_lengths_this_iteration) if game_lengths_this_iteration else 0
            selfplay_max_game_length = max(game_lengths_this_iteration) if game_lengths_this_iteration else 0
            selfplay_games_per_hour = (
                games_completed_count / (selfplay_wall_time / 3600) if selfplay_wall_time > 0 else 0
            )

            # Aggregate per-game timing data from status JSONs
            selfplay_timing_keys = [
                "total_game_s", "tree_search_s", "inference_s", "encoding_s",
                "leaf_selection_s", "backup_s", "pick_move_s", "play_move_s",
                "data_extraction_s",
            ]
            selfplay_timing_sums = {k: 0.0 for k in selfplay_timing_keys}
            selfplay_timing_count = 0
            selfplay_total_sims = 0
            for game_file in SELF_PLAY_GAMES_STATUS_PATH.glob("*.json"):
                gdata = _safe_read_json(game_file)
                if not gdata:
                    continue
                timing = gdata.get("timing")
                if timing and gdata.get("status") == "Completed":
                    for k in selfplay_timing_keys:
                        selfplay_timing_sums[k] += timing.get(k, 0.0)
                    selfplay_total_sims += timing.get("total_sims", 0)
                    selfplay_timing_count += 1

            selfplay_timing_avgs = {}
            if selfplay_timing_count > 0:
                for k in selfplay_timing_keys:
                    selfplay_timing_avgs[f"avg_{k}"] = round(selfplay_timing_sums[k] / selfplay_timing_count, 3)
                selfplay_timing_avgs["avg_sims_per_game"] = round(selfplay_total_sims / selfplay_timing_count, 1)
            selfplay_moves_per_second = (
                experiences_this_iteration / selfplay_wall_time if selfplay_wall_time > 0 else 0
            )

            status["status_message"] = "Self-play phase completed. Starting training."
            update_loop_status(status)
            LOGGER.info(f"--- Starting training on self-play data ({games_completed_count} games) ---")
            training_config = loop_config.training_config

            if not isinstance(training_config, TrainingConfig):
                LOGGER.error(f"FIX THIS: Training config is not a TrainingConfig: {training_config}")
                training_config = TrainingConfig.from_json(training_config)

            training_config.metrics = metrics
            training_config.global_step = loop

            # Get the model and train it, capturing metrics
            model = get_latest_model("model_checkpoints")
            training_config.train_dir = Path(f"training_examples/selfplay/{model.get_name()}")

            # Train the model and capture metrics
            training_start_time = time.time()
            _, train_metrics = train(training_config, model, model_checkpoint_dir=MODEL_CHECKPOINT_DIR)
            training_wall_time = time.time() - training_start_time

            # Model gating: evaluate candidate before promoting
            gate_win_rate = None
            gate_promoted = None
            gating_start_time = time.time()
            if no_gate or loop == 0:
                if loop == 0:
                    LOGGER.info("First iteration; skipping gating, saving model directly.")
                else:
                    LOGGER.info("Gating disabled (--no-gate); saving model directly.")
                checkpoint_num = save_model(model, MODEL_CHECKPOINT_DIR)
                gate_promoted = True
                append_model_history(
                    {
                        "loop": loop + 1,
                        "timestamp": datetime.now().isoformat(),
                        "checkpoint_num": checkpoint_num,
                        "architecture": model.architecture_name(),
                        "session": model.get_name(),
                        "promoted": True,
                        "win_rate": None,
                        "gate_games": 0,
                        "reason": "first_iteration" if loop == 0 else "gating_disabled",
                    }
                )
            else:
                status["status_message"] = f"Running gating evaluation ({gate_games} games)..."
                update_loop_status(status)

                current_best_model = get_latest_model("model_checkpoints")
                win_rate = evaluate_candidate(
                    candidate_model=model,
                    current_best_model=current_best_model,
                    num_games=gate_games,
                    num_readouts=min(num_readouts, 50),
                )
                gate_win_rate = win_rate
                metrics.add_scalar("Gating/Win_Rate", win_rate, loop)
                metrics.add_scalar("Gating/Promoted", 1.0 if win_rate >= gate_threshold else 0.0, loop)

                if win_rate >= gate_threshold:
                    LOGGER.info(f"Model promoted! Win rate: {win_rate:.1%} >= {gate_threshold:.1%}")
                    checkpoint_num = save_model(model, MODEL_CHECKPOINT_DIR)
                    gate_promoted = True
                else:
                    LOGGER.info(f"Model rejected. Win rate: {win_rate:.1%} < {gate_threshold:.1%}")
                    checkpoint_num = None
                    gate_promoted = False

                append_model_history(
                    {
                        "loop": loop + 1,
                        "timestamp": datetime.now().isoformat(),
                        "checkpoint_num": checkpoint_num,
                        "architecture": model.architecture_name(),
                        "session": model.get_name(),
                        "promoted": gate_promoted,
                        "win_rate": win_rate,
                        "gate_games": gate_games,
                        "reason": "gating",
                    }
                )

            gating_wall_time = time.time() - gating_start_time
            iteration_wall_time = selfplay_wall_time + training_wall_time + gating_wall_time

            # Log performance timing summary
            LOGGER.info(
                f"Loop {loop+1} timing breakdown:\n"
                f"  Self-play:  {selfplay_wall_time:.1f}s ({selfplay_wall_time/iteration_wall_time*100:.0f}%)\n"
                f"  Training:   {training_wall_time:.1f}s ({training_wall_time/iteration_wall_time*100:.0f}%)\n"
                f"  Gating:     {gating_wall_time:.1f}s ({gating_wall_time/iteration_wall_time*100:.0f}%)\n"
                f"  Total:      {iteration_wall_time:.1f}s"
            )
            if selfplay_timing_avgs:
                LOGGER.info(
                    f"Loop {loop+1} self-play timing (avg per game):\n"
                    f"  Tree search:     {selfplay_timing_avgs.get('avg_tree_search_s', 0):.1f}s\n"
                    f"    Inference:     {selfplay_timing_avgs.get('avg_inference_s', 0):.1f}s\n"
                    f"    Encoding:      {selfplay_timing_avgs.get('avg_encoding_s', 0):.1f}s\n"
                    f"    Leaf select:   {selfplay_timing_avgs.get('avg_leaf_selection_s', 0):.1f}s\n"
                    f"    Backup:        {selfplay_timing_avgs.get('avg_backup_s', 0):.1f}s\n"
                    f"  Pick move:       {selfplay_timing_avgs.get('avg_pick_move_s', 0):.1f}s\n"
                    f"  Play move:       {selfplay_timing_avgs.get('avg_play_move_s', 0):.1f}s\n"
                    f"  Data extraction: {selfplay_timing_avgs.get('avg_data_extraction_s', 0):.1f}s\n"
                    f"  Total game:      {selfplay_timing_avgs.get('avg_total_game_s', 0):.1f}s\n"
                    f"  Avg sims/game:   {selfplay_timing_avgs.get('avg_sims_per_game', 0):.0f}\n"
                    f"  Moves/sec:       {selfplay_moves_per_second:.1f}"
                )

            # Write timing metrics to TensorBoard
            metrics.add_scalar("Timing/Selfplay_Wall_Time_s", selfplay_wall_time, loop)
            metrics.add_scalar("Timing/Training_Wall_Time_s", training_wall_time, loop)
            metrics.add_scalar("Timing/Gating_Wall_Time_s", gating_wall_time, loop)
            metrics.add_scalar("Timing/Total_Iteration_s", iteration_wall_time, loop)
            metrics.add_scalar("Timing/Moves_Per_Second", selfplay_moves_per_second, loop)
            if selfplay_timing_avgs:
                for k, v in selfplay_timing_avgs.items():
                    metrics.add_scalar(f"Timing/SelfPlay_{k}", v, loop)

            # Update loop metrics with training results
            if train_metrics and train_metrics.epochs_trained > 0:
                loop_metrics.training_losses.append(train_metrics.avg_total_loss)
                loop_metrics.policy_losses.append(train_metrics.avg_policy_loss)
                loop_metrics.value_losses.append(train_metrics.avg_value_loss)
                loop_metrics.training_examples_total += train_metrics.training_examples

                # Core losses
                metrics.add_scalar("Training/Total_Loss", train_metrics.avg_total_loss, loop)
                metrics.add_scalar("Training/Policy_Loss", train_metrics.avg_policy_loss, loop)
                metrics.add_scalar("Training/Value_Loss", train_metrics.avg_value_loss, loop)
                metrics.add_scalar("Training/Examples_This_Iteration", train_metrics.training_examples, loop)
                metrics.add_scalar("Training/Examples_Total", loop_metrics.training_examples_total, loop)

                # Per-epoch losses
                for epoch_idx, epoch_loss in enumerate(train_metrics.epoch_losses):
                    metrics.add_scalar(f"Training/Epoch_Loss/Loop{loop}", epoch_loss, epoch_idx)

                # Last-epoch values for per-loop TensorBoard scalars
                def _last(lst, default=0.0):
                    return lst[-1] if lst else default

                # Loss components
                metrics.add_scalar("Training/Entropy", _last(train_metrics.epoch_entropy), loop)
                metrics.add_scalar("Training/Aux_Loss", _last(train_metrics.epoch_aux_losses), loop)

                # Policy diagnostics
                metrics.add_scalar("Policy/Top1_Accuracy", _last(train_metrics.epoch_top1_accuracy), loop)
                metrics.add_scalar("Policy/Top5_Accuracy", _last(train_metrics.epoch_top5_accuracy), loop)
                metrics.add_scalar("Policy/KL_Divergence", _last(train_metrics.epoch_policy_kl), loop)
                metrics.add_scalar("Policy/Network_Entropy", _last(train_metrics.epoch_policy_entropy), loop)
                metrics.add_scalar("Policy/MCTS_Target_Entropy", _last(train_metrics.epoch_target_entropy), loop)
                metrics.add_scalar(
                    "Policy/Max_Prob_Concentration", _last(train_metrics.epoch_legal_move_concentration), loop
                )
                metrics.add_scalar("Policy/Mean_Legal_Actions", _last(train_metrics.epoch_mean_legal_actions), loop)

                # Value diagnostics
                metrics.add_scalar(
                    "Value/Explained_Variance", _last(train_metrics.epoch_value_explained_variance), loop
                )
                metrics.add_scalar("Value/Correlation", _last(train_metrics.epoch_value_correlation), loop)
                metrics.add_scalar("Value/MAE", _last(train_metrics.epoch_value_mae), loop)
                metrics.add_scalar("Value/MSE", _last(train_metrics.epoch_value_mse), loop)
                metrics.add_scalar("Value/Pred_Mean", _last(train_metrics.epoch_value_pred_mean), loop)
                metrics.add_scalar("Value/Pred_Std", _last(train_metrics.epoch_value_pred_std), loop)
                metrics.add_scalar("Value/Pred_Min", _last(train_metrics.epoch_value_pred_min), loop)
                metrics.add_scalar("Value/Pred_Max", _last(train_metrics.epoch_value_pred_max), loop)
                metrics.add_scalar("Value/Target_Mean", _last(train_metrics.epoch_value_target_mean), loop)
                metrics.add_scalar("Value/Target_Std", _last(train_metrics.epoch_value_target_std), loop)
                metrics.add_scalar("Value/Target_Min", _last(train_metrics.epoch_value_target_min), loop)
                metrics.add_scalar("Value/Target_Max", _last(train_metrics.epoch_value_target_max), loop)

                # Gradient norms
                metrics.add_scalar("Gradients/Total_Norm", _last(train_metrics.epoch_grad_norm_total), loop)
                metrics.add_scalar("Gradients/Policy_Head_Norm", _last(train_metrics.epoch_grad_norm_policy_head), loop)
                metrics.add_scalar("Gradients/Value_Head_Norm", _last(train_metrics.epoch_grad_norm_value_head), loop)
                metrics.add_scalar("Gradients/Trunk_Norm", _last(train_metrics.epoch_grad_norm_trunk), loop)
                metrics.add_scalar("Gradients/CV", _last(train_metrics.epoch_grad_norm_cv), loop)

                # Learning rate
                metrics.add_scalar("Training/Learning_Rate", _last(train_metrics.epoch_lr), loop)

                # Aux diagnostics
                metrics.add_scalar("Aux/Pred_Mean", _last(train_metrics.epoch_aux_pred_mean), loop)
                metrics.add_scalar("Aux/Target_Mean", _last(train_metrics.epoch_aux_target_mean), loop)
                metrics.add_scalar("Aux/Correlation", _last(train_metrics.epoch_aux_correlation), loop)

                LOGGER.info(
                    f"Loop {loop+1} training metrics - Total Loss: {train_metrics.avg_total_loss:.4f}, "
                    f"Policy Loss: {train_metrics.avg_policy_loss:.4f}, Value Loss: {train_metrics.avg_value_loss:.4f}, "
                    f"Examples: {train_metrics.training_examples}"
                )

            LOGGER.info(f"--- Finished training on self-play data ---")

            # Append metrics history record for the dashboard
            def _last_or(lst, default=0.0):
                return lst[-1] if lst else default

            history_record = {
                "loop": loop + 1,
                "timestamp": datetime.now().isoformat(),
                "model_session": model.get_name(),
                "model_architecture": model.architecture_name(),
                "promoted": gate_promoted,
                # Losses
                "total_loss": train_metrics.avg_total_loss
                if train_metrics and train_metrics.epochs_trained > 0
                else None,
                "policy_loss": train_metrics.avg_policy_loss
                if train_metrics and train_metrics.epochs_trained > 0
                else None,
                "value_loss": train_metrics.avg_value_loss
                if train_metrics and train_metrics.epochs_trained > 0
                else None,
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
                "max_prob_concentration": _last_or(train_metrics.epoch_legal_move_concentration)
                if train_metrics
                else 0.0,
                "mean_legal_actions": _last_or(train_metrics.epoch_mean_legal_actions) if train_metrics else 0.0,
                # Value health
                "value_explained_variance": _last_or(train_metrics.epoch_value_explained_variance)
                if train_metrics
                else 0.0,
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
                "games_played": games_completed_count,
                "experiences": experiences_this_iteration,
                "avg_game_length": avg_game_length,
                "min_game_length": selfplay_min_game_length,
                "max_game_length": selfplay_max_game_length,
                "total_experiences_cumulative": loop_metrics.training_examples_total,
                "selfplay_wall_time_seconds": round(selfplay_wall_time, 1),
                "selfplay_games_per_hour": round(selfplay_games_per_hour, 1),
                "phase_move_counts": aggregated_phase_counts,
                # Gating
                "gate_win_rate": gate_win_rate,
                "gate_games_played": gate_games if gate_win_rate is not None else 0,
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
                "gating_wall_time_seconds": round(gating_wall_time, 1),
                "total_iteration_time_seconds": round(iteration_wall_time, 1),
                "selfplay_moves_per_second": round(selfplay_moves_per_second, 1),
                # Self-play timing breakdown (avg per game)
                **{f"selfplay_{k}": v for k, v in selfplay_timing_avgs.items()},
            }
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
        "--num_games_per_iteration", type=int, default=25, help="Number of self-play games to run per iteration"
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
    args = parser.parse_args()

    num_loop_iterations = args.num_loop_iterations
    num_games_per_iteration = args.num_games_per_iteration
    num_threads = args.num_threads
    cleanup = not args.keep_old_files
    main(
        num_loop_iterations,
        num_games_per_iteration,
        num_threads,
        cleanup,
        args.num_readouts,
        num_epochs=args.num_epochs,
        max_training_window=args.max_training_window,
        gate_games=args.gate_games,
        gate_threshold=args.gate_threshold,
        no_gate=args.no_gate,
        batch_size=args.batch_size,
    )
