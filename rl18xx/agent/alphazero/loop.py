from concurrent.futures import as_completed, ProcessPoolExecutor
import os
import json
import logging
import gc
import shutil # Added for cleanup
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from dataclasses import fields
from rl18xx.agent.alphazero.checkpointer import get_latest_model
from rl18xx.agent.alphazero.config import SelfPlayConfig, TrainingConfig
from rl18xx.agent.alphazero.metrics import Metrics
from rl18xx.agent.alphazero.self_play import SelfPlay, SELF_PLAY_GAMES_STATUS_PATH
from rl18xx.agent.alphazero.train import train_latest_model
from pathlib import Path
from rl18xx.agent.alphazero.model import AlphaZeroModel

LOGGER = logging.getLogger(__name__)
LOOP_LOCK_FILE = Path("loop.lock")
RUNTIME_CONFIG_PATH = Path("runtime_config.json")
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
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(log_formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    if console:
        root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)


def load_runtime_config(
    cli_default_num_games: int,
    program_default_training_params: dict,
    cli_default_num_threads: int
):
    """Loads runtime configuration, starting with program defaults and overriding from JSON file."""
    effective_num_games = cli_default_num_games
    effective_training_params = {}
    effective_num_threads = cli_default_num_threads

    for param_name, param_val in program_default_training_params.items():
        if isinstance(param_val, Path):
            effective_training_params[param_name] = str(param_val)
        else:
            effective_training_params[param_name] = param_val

    if not RUNTIME_CONFIG_PATH.exists():
        default_save_config = {
            "num_games_per_iteration": effective_num_games,
            "training_params": effective_training_params,
            "num_threads": effective_num_threads
        }
        with open(RUNTIME_CONFIG_PATH, 'w') as f:
            json.dump(default_save_config, f, indent=4)
        LOGGER.info(f"Created default {RUNTIME_CONFIG_PATH}")
        return effective_num_games, effective_training_params, effective_num_threads
    
    try:
        with open(RUNTIME_CONFIG_PATH, 'r') as f:
            overrides = json.load(f)
        
        effective_num_games = overrides.get("num_games_per_iteration", effective_num_games)
        effective_num_threads = overrides.get("num_threads", effective_num_threads)

        if "training_params" in overrides:
            for key, value in overrides["training_params"].items():
                if key in effective_training_params: # Only override known params
                    effective_training_params[key] = value
                else:
                    LOGGER.warning(f"Unknown training parameter '{key}' in runtime_config.json will be ignored.")
        LOGGER.info(f"Loaded runtime configuration from {RUNTIME_CONFIG_PATH}")
    except json.JSONDecodeError as e:
        LOGGER.error(f"Error decoding {RUNTIME_CONFIG_PATH}: {e}. Using defaults.")
    except Exception as e:
        LOGGER.error(f"Error loading {RUNTIME_CONFIG_PATH}: {e}. Using defaults.")
    return effective_num_games, effective_training_params, effective_num_threads


def update_loop_status(status_data: dict):
    """Writes the current loop status to a JSON file."""
    try:
        with open(LOOP_STATUS_PATH, 'w') as f:
            json.dump(status_data, f, indent=4)
    except Exception as e:
        LOGGER.error(f"Error writing to {LOOP_STATUS_PATH}: {e}")

def cleanup_files():
    if RUNTIME_CONFIG_PATH.exists():
        RUNTIME_CONFIG_PATH.unlink()
    if LOOP_STATUS_PATH.exists():
        LOOP_STATUS_PATH.unlink()
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
    if LOOP_LOCK_FILE.exists():
        LOOP_LOCK_FILE.unlink()

def run_self_play(game_idx_in_iteration: int, tb_log_dir: str, timestamp: str, loop: int):
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
            game_idx_in_iteration=game_idx_in_iteration
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
    _base_tc_defaults = TrainingConfig() # Instance to get defaults
    PROGRAM_DEFAULT_TRAINING_PARAMS = {
        field.name: getattr(_base_tc_defaults, field.name)
        for field in fields(TrainingConfig)
    }
    del _base_tc_defaults

    try:
        for loop in range(num_loop_iterations):
            LOGGER.info(f"--- Starting loop {loop+1}/{num_loop_iterations} ---")
            (
                current_num_games_per_iteration,
                current_training_params_dict,
                current_num_threads
            ) = load_runtime_config(
                num_games_per_iteration,
                PROGRAM_DEFAULT_TRAINING_PARAMS,
                num_threads
            )
            LOGGER.info(f"Loop {loop+1}: Using {current_num_games_per_iteration} games per iteration.")
            LOGGER.info(f"Loop {loop+1}: Using training params: {current_training_params_dict}")
            LOGGER.info(f"Loop {loop+1}: Using {current_num_threads} processes for self-play.")

            metrics.add_scalar("Config/Num_Games_per_Iteration", current_num_games_per_iteration, loop)
            for param_name, param_val in current_training_params_dict.items():
                if isinstance(param_val, (int, float)): # Tensorboard only scalars
                    metrics.add_scalar(f"Config/Training_{param_name}", param_val, loop)
            metrics.add_scalar("Config/Num_Threads", current_num_threads, loop)
            status = {
                "current_loop": loop + 1,
                "total_loops": num_loop_iterations,
                "games_completed_this_iteration": 0,
                "total_games_this_iteration": current_num_games_per_iteration,
                "status_message": "Running self-play games",
                "tensorboard_log_dir": tb_log_dir
            }
            update_loop_status(status)
            
            # Create the directory for self-play logs once per run, if not already handled by cleanup
            self_play_logs_path = Path("logs/self_play")
            self_play_logs_path.mkdir(parents=True, exist_ok=True)

            games_completed_count = 0
            with ProcessPoolExecutor(max_workers=current_num_threads) as executor:
                futures = [executor.submit(run_self_play, i, tb_log_dir, timestamp, loop) for i in range(current_num_games_per_iteration)]
                for i, future in enumerate(as_completed(futures)):
                    try:
                        future.result() # Wait for game to complete and raise exceptions if any
                        games_completed_count += 1
                        status["games_completed_this_iteration"] = games_completed_count
                        status["status_message"] = f"Self-play: {games_completed_count}/{current_num_games_per_iteration} games completed."
                        update_loop_status(status)
                        # Log to TensorBoard - global step for games can be loop * total_games + current_game
                        # Or simply games completed per iteration at loop step
                        metrics.add_scalar("SelfPlay/Progress_in_Iteration", (games_completed_count / current_num_games_per_iteration) * 100, loop)

                    except Exception as e:
                        LOGGER.error(f"Error in self-play game {i+1}: {e}", exc_info=True)
                        status["status_message"] = f"Error in self-play game {i+1}. Check logs."
                        update_loop_status(status)

            metrics.add_scalar("SelfPlay/Completed_Games_Total_for_Iteration", games_completed_count, loop)

            status["status_message"] = "Self-play phase completed. Starting training."
            update_loop_status(status)
            LOGGER.info(f"--- Starting training on self-play data ({games_completed_count} games) ---")
            try:
                # Filter params for TrainingConfig constructor to avoid unexpected keyword arguments
                valid_tc_keys = {f.name for f in fields(TrainingConfig)}
                filtered_training_params_for_constructor = {
                    k: v for k, v in current_training_params_dict.items() if k in valid_tc_keys
                }
                training_config_instance = TrainingConfig(**filtered_training_params_for_constructor)
            except TypeError as e:
                LOGGER.error(f"Error creating TrainingConfig with params {current_training_params_dict}: {e}. Using default TrainingConfig.")
                training_config_instance = TrainingConfig()

            training_config_instance.metrics = metrics
            training_config_instance.global_step = loop
            train_latest_model(training_config_instance)
            LOGGER.info(f"--- Finished training on self-play data ---")

            status["status_message"] = f"Loop {loop+1} training complete. Preparing for next loop."
            update_loop_status(status)
            metrics.add_scalar("Loop/Progress", (loop + 1) / num_loop_iterations * 100, loop)
            
            gc.collect() # Explicit garbage collection at end of loop
    except Exception as e:
        LOGGER.error(f"Error in loop {loop+1}: {e}", exc_info=True)
    except KeyboardInterrupt as e:
        LOGGER.info("--- Keyboard interrupt received ---")
    finally:
        if LOOP_LOCK_FILE.exists():
            LOOP_LOCK_FILE.unlink()

    LOGGER.info("--- Finished all loops ---")
    metrics.close()
    exit(0)


if __name__ == "__main__":
    import argparse
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    parser = argparse.ArgumentParser(description="Run AlphaZero training loop")
    parser.add_argument(
        "--num_loop_iterations", type=int, default=100, help="Number of iterations of the full training loop"
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
