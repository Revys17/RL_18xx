from concurrent.futures import as_completed, ProcessPoolExecutor
import os
import json
import logging
import gc
import shutil
from datetime import datetime
from dataclasses import dataclass, asdict
from rl18xx.agent.alphazero.checkpointer import get_latest_model
from rl18xx.agent.alphazero.config import SelfPlayConfig, TrainingConfig
from rl18xx.agent.alphazero.metrics import Metrics
from rl18xx.agent.alphazero.self_play import SelfPlay, SELF_PLAY_GAMES_STATUS_PATH
from rl18xx.agent.alphazero.train import train_latest_model, train, TrainingMetrics
from pathlib import Path
import signal
import sys
import psutil
import atexit

LOGGER = logging.getLogger(__name__)
LOOP_LOCK_FILE = Path("loop.lock")
LOOP_CONFIG_PATH = Path("loop_config.json")
LOOP_STATUS_PATH = Path("loop_status.json")
TENSORBOARD_LOG_DIR_BASE = Path("runs/alphazero_runs")
SELF_PLAY_LOGS_PATH = Path("logs/self_play")


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

@dataclass
class LoopMetrics:
    """Tracks metrics across the entire training loop"""
    loop_iteration: int = 0
    games_played_total: int = 0
    training_examples_total: int = 0
    training_losses: list = None
    policy_losses: list = None
    value_losses: list = None
    validation_losses: list = None
    validation_policy_losses: list = None
    validation_value_losses: list = None
    game_lengths: list = None
    win_rates_by_player: dict = None
    
    def __post_init__(self):
        if self.training_losses is None:
            self.training_losses = []
        if self.policy_losses is None:
            self.policy_losses = []
        if self.value_losses is None:
            self.value_losses = []
        if self.validation_losses is None:
            self.validation_losses = []
        if self.validation_policy_losses is None:
            self.validation_policy_losses = []
        if self.validation_value_losses is None:
            self.validation_value_losses = []
        if self.game_lengths is None:
            self.game_lengths = []
        if self.win_rates_by_player is None:
            self.win_rates_by_player = {0: [], 1: [], 2: [], 3: []}

def load_loop_config(
    num_loop_iterations: int,
    num_games_per_iteration: int,
    num_threads: int,
    default_training_config: TrainingConfig,
    num_readouts: int
) -> LoopConfig:
    try:
        if not LOOP_CONFIG_PATH.exists():
            loop_config = LoopConfig(
                num_loop_iterations=num_loop_iterations,
                num_games_per_iteration=num_games_per_iteration,
                num_threads=num_threads,
                training_config=default_training_config.to_json(),
                num_readouts=num_readouts
            )

            with open(LOOP_CONFIG_PATH, 'w') as f:
                json.dump(asdict(loop_config), f, indent=4)

            loop_config.training_config = default_training_config
            return loop_config

        with open(LOOP_CONFIG_PATH, 'r') as f:
            loop_config_json = json.load(f)
        training_config = TrainingConfig.from_json(loop_config_json["training_config"])
        loop_config = LoopConfig(
            num_loop_iterations=loop_config_json["num_loop_iterations"],
            num_games_per_iteration=loop_config_json["num_games_per_iteration"],
            num_threads=loop_config_json["num_threads"],
            training_config=training_config,
            num_readouts=loop_config_json["num_readouts"]
        )
        return loop_config
    except Exception as e:
        LOGGER.error(f"Error loading loop config: {e}. Using default config.")
        return LoopConfig(
            num_loop_iterations=num_loop_iterations,
            num_games_per_iteration=num_games_per_iteration,
            num_threads=num_threads,
            training_config=default_training_config,
            num_readouts=num_readouts
        )

def update_loop_status(status_data: dict):
    """Writes the current loop status to a JSON file."""
    try:
        with open(LOOP_STATUS_PATH, 'w') as f:
            json.dump(status_data, f, indent=4, default=str)
    except Exception as e:
        LOGGER.error(f"Error writing to {LOOP_STATUS_PATH}: {e}")

def save_loop_metrics(metrics: LoopMetrics, filepath: Path):
    """Saves the loop metrics to a JSON file."""
    try:
        metrics_dict = asdict(metrics)
        with open(filepath, 'w') as f:
            json.dump(metrics_dict, f, indent=4)
    except Exception as e:
        LOGGER.error(f"Error saving loop metrics: {e}")

def load_loop_metrics(filepath: Path) -> LoopMetrics:
    """Loads loop metrics from a JSON file."""
    try:
        if filepath.exists():
            with open(filepath, 'r') as f:
                metrics_dict = json.load(f)
            return LoopMetrics(**metrics_dict)
    except Exception as e:
        LOGGER.error(f"Error loading loop metrics: {e}")
    return LoopMetrics()

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

def run_self_play(game_idx_in_iteration: int, tb_log_dir: str, timestamp: str, loop: int, num_readouts: int = 32):
    process_root_logger = logging.getLogger()
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
        self_play_config = SelfPlayConfig(
            network=model, 
            metrics=Metrics(os.path.join(tb_log_dir, f"game_L{loop}_G{game_idx_in_iteration}")),
            global_step=loop, 
            game_idx_in_iteration=game_idx_in_iteration,
            game_id=f"L{loop}_G{game_idx_in_iteration}",
            num_readouts=num_readouts
        )
        selfplay = SelfPlay(self_play_config)
        selfplay.run_game()
        logging.info(f"Self-play game L{loop}/G{game_idx_in_iteration} completed successfully.")
    except Exception as e_proc:
        logging.error(f"Error during self-play game L{loop}/G{game_idx_in_iteration}: {e_proc}", exc_info=True)
        raise # Re-raise to be caught by the main process's future.result()
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

def main(num_loop_iterations: int, num_games_per_iteration: int, num_threads: int, cleanup: bool):
    if cleanup:
        cleanup_files()

    if LOOP_LOCK_FILE.exists():
        LOGGER.error("Loop already running. Exiting.")
        exit(1)
    LOOP_LOCK_FILE.touch()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    setup_logging(logging.INFO, f"logs/loop/loop_{timestamp}.log")

    tb_log_dir = os.path.join(TENSORBOARD_LOG_DIR_BASE, f"experiment_{timestamp}")
    metrics = Metrics(tb_log_dir)
    self_play_logs_path = Path("logs/self_play")
    self_play_logs_path.mkdir(parents=True, exist_ok=True)
    default_training_config = TrainingConfig()
    num_readouts = SelfPlayConfig().num_readouts
    
    # Initialize loop metrics
    loop_metrics_path = Path(f"logs/loop/loop_metrics_{timestamp}.json")
    loop_metrics = LoopMetrics()

    # Register signal handlers
    signal.signal(signal.SIGINT, cleanup_and_exit)
    signal.signal(signal.SIGTERM, cleanup_and_exit)
    atexit.register(cleanup_and_exit)

    try:
        for loop in range(num_loop_iterations):
            LOGGER.info(f"--- Starting loop {loop+1}/{num_loop_iterations} ---")
            loop_config = load_loop_config(
                num_loop_iterations,
                num_games_per_iteration,
                num_threads,
                default_training_config,
                num_readouts
            )

            # See if the run config has been updated and we should end early
            if loop >= loop_config.num_loop_iterations:
                LOGGER.info(f"Desired number of loops was updated - Exiting early.")
                break

            LOGGER.info(f"Loop {loop+1}: Using {loop_config.num_games_per_iteration} games per iteration.")
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
                    "avg_training_loss": sum(loop_metrics.training_losses[-10:]) / len(loop_metrics.training_losses[-10:]) if loop_metrics.training_losses else 0,
                    "avg_policy_loss": sum(loop_metrics.policy_losses[-10:]) / len(loop_metrics.policy_losses[-10:]) if loop_metrics.policy_losses else 0,
                    "avg_value_loss": sum(loop_metrics.value_losses[-10:]) / len(loop_metrics.value_losses[-10:]) if loop_metrics.value_losses else 0
                }
            }      
            update_loop_status(status)
            
            # Create the directory for self-play logs once per run, if not already handled by cleanup

            games_completed_count = 0
            game_results = []  # Track results for win rate calculation
            game_lengths_this_iteration = []
            executor = ProcessPoolExecutor(max_workers=loop_config.num_threads)
            
            try:
                futures = [executor.submit(run_self_play, i, tb_log_dir, timestamp, loop, loop_config.num_readouts) for i in range(loop_config.num_games_per_iteration)]
                for i, future in enumerate(as_completed(futures)):
                    try:
                        future.result() # Wait for game to complete and raise exceptions if any
                        games_completed_count += 1
                        status["games_completed_this_iteration"] = games_completed_count
                        status["status_message"] = f"Self-play: {games_completed_count}/{loop_config.num_games_per_iteration} games completed."
                        update_loop_status(status)
                        metrics.add_scalar("SelfPlay/Progress_in_Iteration", (games_completed_count / loop_config.num_games_per_iteration) * 100, loop)
                    except Exception as e:
                        LOGGER.error(f"Error in self-play game {i+1}: {e}", exc_info=True)
                        status["status_message"] = f"Error in self-play game {i+1}. Check logs."
                        update_loop_status(status)
            finally:
                executor.shutdown(wait=True)

            metrics.add_scalar("SelfPlay/Completed_Games_Total_for_Iteration", games_completed_count, loop)
            loop_metrics.games_played_total += games_completed_count
            
            # Collect game statistics from status files
            try:
                for game_file in SELF_PLAY_GAMES_STATUS_PATH.glob(f"L{loop}_G*.json"):
                    with open(game_file, 'r') as f:
                        game_data = json.load(f)
                    if game_data.get("status") == "Completed":
                        game_lengths_this_iteration.append(game_data.get("moves_played", 0))
                        loop_metrics.game_lengths.append(game_data.get("moves_played", 0))
                
                # Calculate average game length for this iteration
                if game_lengths_this_iteration:
                    avg_game_length = sum(game_lengths_this_iteration) / len(game_lengths_this_iteration)
                    metrics.add_scalar("SelfPlay/Avg_Game_Length", avg_game_length, loop)
                    LOGGER.info(f"Loop {loop+1}: Average game length: {avg_game_length:.1f} moves")
            except Exception as e:
                LOGGER.error(f"Error collecting game statistics: {e}")

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
            training_config.train_dir = training_config.root_dir / f"selfplay/{model.get_name()}"
            training_config.val_dir = training_config.root_dir / f"holdout/{model.get_name()}"
            
            # Train the model and capture metrics
            train_metrics = train(training_config, model)
            
            # Update loop metrics with training results
            if train_metrics and train_metrics.epochs_trained > 0:
                loop_metrics.training_losses.append(train_metrics.avg_total_loss)
                metrics.add_scalar("Training/Total_Loss", train_metrics.avg_total_loss, loop)
                
                loop_metrics.policy_losses.append(train_metrics.avg_policy_loss)
                metrics.add_scalar("Training/Policy_Loss", train_metrics.avg_policy_loss, loop)
                
                loop_metrics.value_losses.append(train_metrics.avg_value_loss)
                metrics.add_scalar("Training/Value_Loss", train_metrics.avg_value_loss, loop)
                
                # Log per-epoch metrics
                for epoch_idx, epoch_loss in enumerate(train_metrics.epoch_losses):
                    metrics.add_scalar(f"Training/Epoch_Loss/Loop{loop}", epoch_loss, epoch_idx)
                    
                if train_metrics.val_total_loss > 0:
                    loop_metrics.validation_losses.append(train_metrics.val_total_loss)
                    metrics.add_scalar("Training/Val_Total_Loss", train_metrics.val_total_loss, loop)
                    
                    loop_metrics.validation_policy_losses.append(train_metrics.val_policy_loss)
                    metrics.add_scalar("Training/Val_Policy_Loss", train_metrics.val_policy_loss, loop)
                    
                    loop_metrics.validation_value_losses.append(train_metrics.val_value_loss)
                    metrics.add_scalar("Training/Val_Value_Loss", train_metrics.val_value_loss, loop)
                    
                loop_metrics.training_examples_total += train_metrics.training_examples
                metrics.add_scalar("Training/Examples_This_Iteration", train_metrics.training_examples, loop)
                metrics.add_scalar("Training/Examples_Total", loop_metrics.training_examples_total, loop)
                    
                LOGGER.info(f"Loop {loop+1} training metrics - Total Loss: {train_metrics.avg_total_loss:.4f}, "
                           f"Policy Loss: {train_metrics.avg_policy_loss:.4f}, Value Loss: {train_metrics.avg_value_loss:.4f}, "
                           f"Examples: {train_metrics.training_examples}")
            
            LOGGER.info(f"--- Finished training on self-play data ---")

            status["status_message"] = f"Loop {loop+1} training complete. Preparing for next loop."
            status["loop_metrics"].update({
                "total_games_played": loop_metrics.games_played_total,
                "total_training_examples": loop_metrics.training_examples_total,
                "latest_training_loss": loop_metrics.training_losses[-1] if loop_metrics.training_losses else None,
                "latest_policy_loss": loop_metrics.policy_losses[-1] if loop_metrics.policy_losses else None,
                "latest_value_loss": loop_metrics.value_losses[-1] if loop_metrics.value_losses else None
            })
            update_loop_status(status)
            metrics.add_scalar("Loop/Progress", (loop + 1) / num_loop_iterations * 100, loop)
            
            # Save loop metrics after each iteration
            save_loop_metrics(loop_metrics, loop_metrics_path)
            
            gc.collect()
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
        LOGGER.info(f"Final average training loss: {sum(loop_metrics.training_losses) / len(loop_metrics.training_losses):.4f}")
    if loop_metrics.game_lengths:
        LOGGER.info(f"Average game length across all loops: {sum(loop_metrics.game_lengths) / len(loop_metrics.game_lengths):.1f} moves")
    LOGGER.info(f"Total games played: {loop_metrics.games_played_total}")
    LOGGER.info(f"Total training examples: {loop_metrics.training_examples_total}")
    
    metrics.close()
    exit(0)


if __name__ == "__main__":
    import argparse
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    parser = argparse.ArgumentParser(description="Run AlphaZero training loop")
    parser.add_argument(
        "--num_loop_iterations", type=int, default=5, help="Number of iterations of the full training loop"
    )
    parser.add_argument(
        "--num_games_per_iteration", type=int, default=25, help="Number of self-play games to run per iteration"
    )
    parser.add_argument(
        "--num_threads", type=int, default=2, help="Number of threads to use for self-play"
    )
    parser.add_argument(
        "--keep-old-files", action="store_true", help="Keep old files from previous runs. By default, old files are deleted."
    )
    args = parser.parse_args()

    num_loop_iterations = args.num_loop_iterations
    num_games_per_iteration = args.num_games_per_iteration
    num_threads = args.num_threads
    cleanup = not args.keep_old_files
    main(num_loop_iterations, num_games_per_iteration, num_threads, cleanup)
