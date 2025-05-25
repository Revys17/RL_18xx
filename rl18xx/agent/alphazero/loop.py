# The main training loop for the model - play to generate data, then train on it

import logging
import os
import torch
from datetime import datetime
import glob
import pickle
from typing import List, Tuple, Dict, Any
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from rl18xx.game.gamemap import GameMap
from rl18xx.agent.alphazero.model import Model
from rl18xx.agent.alphazero.encoder import Encoder_1830
from rl18xx.agent.alphazero.action_mapper import ActionMapper
from rl18xx.agent.alphazero.self_play import run_self_play_game, TrainingExample
from rl18xx.agent.alphazero.train import (
    train_model,
    load_training_data,
    SelfPlayDataset,
)

LOGGER = logging.getLogger(__name__)

# --- Configuration Constants (will be moved to argparse later) ---
GAME_TITLE = "1830"
PLAYER_OPTIONS = {"1": "P1", "2": "P2", "3": "P3", "4": "P4"}

# Loop iterations
NUM_LOOP_ITERATIONS = 10  # Total number of self-play -> train cycles

# Self-Play Configuration
NUM_SELF_PLAY_GAMES_PER_ITERATION = 5  # Number of games to play to generate data for one training iteration
MCTS_SIMULATIONS = 10  # Number of MCTS simulations per move (keep low for faster iterations initially)
MCTS_INITIAL_TEMPERATURE = 1.0
MCTS_TEMP_DECAY_STEPS = 200  # Game steps after which MCTS action selection becomes greedy
MCTS_C_PUCT = 5.0
MCTS_DIRICHLET_ALPHA = 0.03
MCTS_DIRICHLET_NOISE_FACTOR = 0.25

# Training Configuration
TRAINING_EPOCHS_PER_ITERATION = 5  # Number of epochs to train the model on new data
TRAINING_BATCH_SIZE = 32  # Batch size for training
TRAINING_LEARNING_RATE = 0.0005  # Learning rate
POLICY_LOSS_WEIGHT = 1.0
VALUE_LOSS_WEIGHT = 1.0

# Paths
BASE_CHECKPOINT_DIR = "model_checkpoints"
BASE_TRAINING_DATA_DIR = "training_data"
LOG_DIR = "logs"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_latest_iteration_checkpoint(checkpoint_dir: str) -> Tuple[str | None, int]:
    """
    Finds the latest model checkpoint based on iteration number.
    Checkpoints are expected to be named like 'model_iteration_X.pth'.
    Returns the path to the latest checkpoint and its iteration number.
    If no checkpoint is found, returns (None, -1).
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "model_iteration_*.pth"))
    if not checkpoint_files:
        return None, -1

    latest_file = None
    latest_iteration = -1
    for f_path in checkpoint_files:
        try:
            filename = os.path.basename(f_path)
            iteration_num_str = filename.split("model_iteration_")[1].split(".pth")[0]
            iteration_num = int(iteration_num_str)
            if iteration_num > latest_iteration:
                latest_iteration = iteration_num
                latest_file = f_path
        except Exception as e:
            LOGGER.warning(f"Could not parse iteration number from checkpoint file {f_path}: {e}")
            continue

    if latest_file:
        LOGGER.info(f"Found latest checkpoint: {latest_file} for iteration {latest_iteration}")
    else:
        LOGGER.info(f"No valid iteration checkpoints found in {checkpoint_dir}")
    return latest_file, latest_iteration


def initialize_components_and_model_params(
    game_title: str, player_options: Dict[str, str]
) -> Tuple[type, Encoder_1830, ActionMapper, Dict[str, Any]]:
    """
    Initializes game class, encoder, action mapper, and derives model parameters.
    """
    game_map = GameMap()
    game_class = game_map.game_by_title(game_title)
    if not game_class:
        LOGGER.error(f"Game class for '{game_title}' not found.")
        raise ValueError(f"Game class for '{game_title}' not found.")

    # Create a temporary game instance to configure encoder and action mapper
    # This ensures they are configured for the specific game being played.
    temp_game_for_config = game_class(player_options)
    num_players_for_config = len(temp_game_for_config.players)

    encoder = Encoder_1830()  # Assuming Encoder_1830, make configurable if other games are used
    action_mapper = ActionMapper()

    # Get model input/output sizes from encoder and action_mapper
    dummy_game_state_encoded, dummy_map_nodes_encoded, dummy_edge_index = encoder.encode(temp_game_for_config)

    model_params = {
        "game_state_size": dummy_game_state_encoded.shape[1],
        "map_node_features": dummy_map_nodes_encoded.shape[1],
        "policy_size": action_mapper.action_encoding_size,
        "value_size": num_players_for_config,
        # Add other model architectural params if they are not defaults or need to be configured
        # e.g., "mlp_hidden_dim": 256, "gnn_layers": 4, etc.
    }
    LOGGER.info(f"Initialized components for {game_title}. Model params: {model_params}")
    return game_class, encoder, action_mapper, model_params


def train_loop(
    game_title: str,
    player_options: Dict[str, str],
    num_loop_iterations: int,
    num_self_play_games_per_iteration: int,
    mcts_simulations: int,
    mcts_initial_temperature: float,
    mcts_temp_decay_steps: int,
    mcts_c_puct: float,
    mcts_dirichlet_alpha: float,
    mcts_dirichlet_noise_factor: float,
    training_epochs_per_iteration: int,
    training_batch_size: int,
    training_learning_rate: float,
    policy_loss_weight: float,
    value_loss_weight: float,
    base_checkpoint_dir: str,
    base_training_data_dir: str,
    base_log_dir: str,
    device: torch.device,
):
    LOGGER.info("===== Initializing AlphaZero Training Loop =====")
    
    # --- Loop-level TensorBoard Writer ---
    # Logs to a general directory for the overall loop progress
    loop_writer_log_dir = os.path.join(base_log_dir, "loop_runs")
    os.makedirs(loop_writer_log_dir, exist_ok=True)
    loop_writer = SummaryWriter(log_dir=loop_writer_log_dir)
    LOGGER.info(f"Loop-level TensorBoard logs will be saved to: {loop_writer_log_dir}")

    # --- 1. Initialize game components and get model parameters ---
    game_class, encoder, action_mapper, model_constructor_params = initialize_components_and_model_params(
        game_title, player_options
    )

    # --- 2. Select/Initialize Model ---
    model = Model(**model_constructor_params)
    model.to(device)

    latest_checkpoint_path, start_iteration = get_latest_iteration_checkpoint(base_checkpoint_dir)

    if latest_checkpoint_path:
        LOGGER.info(f"Loading model from checkpoint: {latest_checkpoint_path}")
        try:
            model.load_state_dict(torch.load(latest_checkpoint_path, map_location=device))
            LOGGER.info(f"Successfully loaded model weights for iteration {start_iteration}.")
            current_iteration = start_iteration + 1  # Start next iteration
        except Exception as e:
            LOGGER.error(f"Failed to load model from {latest_checkpoint_path}: {e}. Starting from scratch.")
            current_iteration = 0
    else:
        LOGGER.info("No existing checkpoint found. Starting training from scratch (iteration 0).")
        current_iteration = 0

    # --- 3. Main Self-Play and Training Loop ---
    for i in range(current_iteration, num_loop_iterations):
        iteration_start_time_dt = datetime.now()
        LOGGER.info(f"\n===== Starting Loop Iteration {i}/{num_loop_iterations -1} =====")

        # --- 3.a. Self-Play Phase ---
        self_play_start_time = datetime.now()
        LOGGER.info(f"--- Iteration {i}: Self-Play Phase ---")
        iteration_data_dir = os.path.join(base_training_data_dir, f"iteration_{i}")
        os.makedirs(iteration_data_dir, exist_ok=True)

        all_new_training_examples: List[TrainingExample] = []
        for game_num in range(num_self_play_games_per_iteration):
            LOGGER.info(
                f"Starting self-play game {game_num + 1}/{num_self_play_games_per_iteration} for iteration {i}..."
            )
            game_examples = run_self_play_game(
                game_class=game_class,
                game_options=player_options,
                model=model,  # Use the current model
                encoder=encoder,
                action_mapper=action_mapper,
                mcts_simulations=mcts_simulations,
                initial_temperature=mcts_initial_temperature,
                temperature_decay_steps=mcts_temp_decay_steps,
                c_puct=mcts_c_puct,
                dirichlet_alpha=mcts_dirichlet_alpha,
                dirichlet_noise_factor=mcts_dirichlet_noise_factor,
                device=device,
            )
            if game_examples:
                all_new_training_examples.extend(game_examples)
                # Save examples from this game immediately
                game_data_filename = (
                    f"self_play_iter_{i}_game_{game_num}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
                )
                game_data_filepath = os.path.join(iteration_data_dir, game_data_filename)
                try:
                    with open(game_data_filepath, "wb") as f:
                        # run_self_play_game returns a list of examples, which is what load_training_data expects per file
                        pickle.dump(game_examples, f)
                    LOGGER.info(f"Saved {len(game_examples)} examples from game {game_num+1} to {game_data_filepath}")
                except Exception as e:
                    LOGGER.error(f"Failed to save game data for game {game_num+1}: {e}")
            else:
                LOGGER.warning(f"Self-play game {game_num + 1} for iteration {i} produced no examples.")

        if not all_new_training_examples:
            LOGGER.warning(
                f"Iteration {i}: No training examples generated from self-play. Skipping training for this iteration."
            )
            continue  # Skip to next iteration if no data

        LOGGER.info(
            f"Iteration {i}: Generated a total of {len(all_new_training_examples)} new training examples from self-play."
        )
        self_play_duration_seconds = (datetime.now() - self_play_start_time).total_seconds()
        loop_writer.add_scalar('Iteration/NumSelfPlayExamples', len(all_new_training_examples), i)
        loop_writer.add_scalar('Iteration/SelfPlayDuration_seconds', self_play_duration_seconds, i)

        # --- 3.b. Training Phase ---
        training_phase_start_time = datetime.now()
        LOGGER.info(f"--- Iteration {i}: Training Phase ---")
        # Load all data generated in this iteration (or a window of recent iterations if desired)
        # For now, just use data from the current iteration's self-play
        training_data_for_this_iteration = load_training_data(iteration_data_dir)

        if not training_data_for_this_iteration:
            LOGGER.error(
                f"Iteration {i}: Failed to load any training data from {iteration_data_dir}. Skipping training."
            )
            continue

        LOGGER.info(f"Iteration {i}: Loaded {len(training_data_for_this_iteration)} examples for training.")

        train_dataset = SelfPlayDataset(training_data_for_this_iteration)

        # Checkpoint directory for the train_model function (for its internal epoch checkpoints)
        # These are specific to this training session.
        train_session_checkpoint_dir = os.path.join(base_checkpoint_dir, f"iteration_{i}_training_checkpoints")
        os.makedirs(train_session_checkpoint_dir, exist_ok=True)

        train_model(
            model=model,  # Train the same model instance
            dataset=train_dataset,
            epochs=training_epochs_per_iteration,
            batch_size=training_batch_size,
            learning_rate=training_learning_rate,
            device=device,
            policy_loss_weight=policy_loss_weight,
            value_loss_weight=value_loss_weight,
            checkpoint_dir=train_session_checkpoint_dir,  # For intermediate epoch saves during this training run
            save_every_n_epochs=max(1, training_epochs_per_iteration // 2),  # Example: save midway and at end
            loop_iteration=i # Pass the current main loop iteration
        )
        training_phase_duration_seconds = (datetime.now() - training_phase_start_time).total_seconds()
        loop_writer.add_scalar('Iteration/TrainingPhaseDuration_seconds', training_phase_duration_seconds, i)

        # --- Save main iteration checkpoint ---
        main_iteration_checkpoint_path = os.path.join(base_checkpoint_dir, f"model_iteration_{i}.pth")
        try:
            torch.save(model.state_dict(), main_iteration_checkpoint_path)
            LOGGER.info(f"Saved main checkpoint for iteration {i} to {main_iteration_checkpoint_path}")
        except Exception as e:
            LOGGER.error(f"Failed to save main checkpoint for iteration {i}: {e}")

        iteration_duration_seconds = (datetime.now() - iteration_start_time_dt).total_seconds()
        loop_writer.add_scalar('Iteration/TotalDuration_seconds', iteration_duration_seconds, i)
        LOGGER.info(f"===== Finished Loop Iteration {i}. Duration: {iteration_duration_seconds:.2f}s =====")

    LOGGER.info("===== AlphaZero Training Loop Completed =====")
    loop_writer.close()


def setup_logging(level: int, log_file: str) -> logging.Logger:
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
    # root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)


if __name__ == "__main__":
    os.makedirs(LOG_DIR, exist_ok=True)
    log_file_path = os.path.join(LOG_DIR, f"AlphaZeroLoop_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    setup_logging(level=logging.INFO, log_file=log_file_path)
    LOGGER.info(f"Log file for this run: {log_file_path}")
    LOGGER.info(f"Using device: {DEVICE}")

    try:
        train_loop(
            game_title=GAME_TITLE,
            player_options=PLAYER_OPTIONS,
            num_loop_iterations=NUM_LOOP_ITERATIONS,
            num_self_play_games_per_iteration=NUM_SELF_PLAY_GAMES_PER_ITERATION,
            mcts_simulations=MCTS_SIMULATIONS,
            mcts_initial_temperature=MCTS_INITIAL_TEMPERATURE,
            mcts_temp_decay_steps=MCTS_TEMP_DECAY_STEPS,
            mcts_c_puct=MCTS_C_PUCT,
            mcts_dirichlet_alpha=MCTS_DIRICHLET_ALPHA,
            mcts_dirichlet_noise_factor=MCTS_DIRICHLET_NOISE_FACTOR,
            training_epochs_per_iteration=TRAINING_EPOCHS_PER_ITERATION,
            training_batch_size=TRAINING_BATCH_SIZE,
            training_learning_rate=TRAINING_LEARNING_RATE,
            policy_loss_weight=POLICY_LOSS_WEIGHT,
            value_loss_weight=VALUE_LOSS_WEIGHT,
            base_checkpoint_dir=BASE_CHECKPOINT_DIR,
            base_training_data_dir=BASE_TRAINING_DATA_DIR,
            base_log_dir=LOG_DIR,
            device=DEVICE,
        )
    except Exception as e:
        LOGGER.exception("An unhandled exception occurred during the training loop.")
    finally:
        LOGGER.info("AlphaZero Training Loop script finished.")
