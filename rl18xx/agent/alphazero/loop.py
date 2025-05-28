# The main training loop for the model - play to generate data, then train on it

from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
import logging
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from rl18xx.agent.alphazero.checkpointer import get_latest_model
from rl18xx.agent.alphazero.config import SelfPlayConfig, TrainingConfig
from rl18xx.agent.alphazero.self_play import SelfPlay, log_memory_usage
from rl18xx.agent.alphazero.train import train_latest_model

LOGGER = logging.getLogger(__name__)


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


def main(num_loop_iterations: int, num_games_per_iteration: int):
    # Starting with 2 threads
    num_threads = 2

    for loop in range(num_loop_iterations):
        LOGGER.info(f"--- Starting loop {loop+1}/{num_loop_iterations} ---")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        setup_logging(logging.INFO, f"logs/loop/loop_{timestamp}.log")

        model = get_latest_model("model_checkpoints")

        def run_self_play():
            self_play_config = SelfPlayConfig(network=model)
            selfplay = SelfPlay(self_play_config)
            selfplay.run_game()

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(run_self_play) for _ in range(num_games_per_iteration)]
            for i, future in enumerate(as_completed(futures)):
                future.result()

        LOGGER.info(f"--- Starting training on self-play data ---")
        training_config = TrainingConfig()
        train_latest_model(training_config)
        LOGGER.info(f"--- Finished training on self-play data ---")

    LOGGER.info("--- Finished all loops ---")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run AlphaZero training loop")
    parser.add_argument(
        "--num_loop_iterations", type=int, default=100, help="Number of iterations of the full training loop"
    )
    parser.add_argument(
        "--num_games_per_iteration", type=int, default=25, help="Number of self-play games to run per iteration"
    )
    args = parser.parse_args()

    num_loop_iterations = args.num_loop_iterations
    num_games_per_iteration = args.num_games_per_iteration
    main(num_loop_iterations, num_games_per_iteration)
