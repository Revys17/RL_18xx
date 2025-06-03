import time
import copy
import logging
from typing import Dict, Any, List
from rl18xx.game.gamemap import GameMap
from rl18xx.game.engine.game.base import BaseGame
from rl18xx.game.action_helper import ActionHelper
import cProfile
import pstats

LOGGER = logging.getLogger(__name__)

class BaseGameLoadProfiler:
    """
    Profiles the performance of BaseGame.load(game_dict).
    """
    def __init__(self, game_dict: Dict[str, Any]):
        """
        Initializes the profiler with a game dictionary.

        Args:
            game_dict: The dictionary representing a game state, typically from game.to_dict().
        """
        self.game_dict = game_dict
        self._load_times: List[float] = []

    def profile_load(self, num_trials: int = 1, game_class: type = None) -> float:
        """
        Profiles BaseGame.load() for the stored game_dict.

        Args:
            num_trials: The number of times to run BaseGame.load().
            game_class: The actual BaseGame class (or a subclass) to call .load() on.
                        This is needed because BaseGame.load is a classmethod.

        Returns:
            The average time taken for BaseGame.load() in seconds.
            
        Raises:
            ValueError: If num_trials is not positive.
            TypeError: If game_class is not provided or not a type.
        """
        if not num_trials > 0:
            raise ValueError("Number of trials must be positive.")

        current_trial_times: List[float] = []
        LOGGER.info(f"Starting profiling of {game_class.__name__}.load() for {num_trials} trial(s)...")
        for i in range(num_trials):
            start_time = time.perf_counter()
            _ = game_class.load(self.game_dict) # type: ignore
            end_time = time.perf_counter()
            current_trial_times.append(end_time - start_time)
            if (i + 1) % (max(1, num_trials // 10)) == 0:
                 LOGGER.debug(f"Profile trial {i+1}/{num_trials} completed.")
        
        avg_time = sum(current_trial_times) / num_trials
        self._load_times.extend(current_trial_times)
        LOGGER.info(f"{game_class.__name__}.load() profiled. Average time: {avg_time:.6f} seconds.")
        return avg_time

    def get_all_load_times(self) -> List[float]:
        """Returns a list of all recorded load times from profile_load()."""
        return self._load_times

class BaseGameCloneBenchmarker:
    """
    Benchmarks and compares BaseGame.load() vs. game_instance.clone().
    """
    def __init__(self, game_instance: BaseGame, num_trials: int = 10):
        """
        Initializes the benchmarker.

        Args:
            game_instance: An instance of BaseGame (or its subclass) to benchmark.
            num_trials: The number of trials for each method.
        
        Raises:
            TypeError: If game_instance is not of the expected type.
            AttributeError: If game_instance does not have a 'clone' or 'to_dict' method.
            ValueError: If num_trials is not positive.
        """

        if not isinstance(game_instance, BaseGame):
            raise TypeError(f"game_instance must be an instance of BaseGame. Got {type(game_instance)}")
        if not hasattr(game_instance, 'clone'):
            raise AttributeError("game_instance must have a 'clone' method for benchmarking.")
        if not hasattr(game_instance, 'to_dict'):
            raise AttributeError("game_instance must have a 'to_dict' method for benchmarking.")
        if not num_trials > 0:
            raise ValueError("Number of trials must be positive.")
            
        self.game_instance = game_instance
        self.num_trials = num_trials
        # Assuming to_dict() is relatively fast. If to_dict() is very slow,
        # this benchmark might slightly favor clone() more than it should in the MCTS context
        # where to_dict() is called once per node initialization.
        self.game_dict = self.game_instance.to_dict()
        self.game_class = game_instance.__class__ # Get the actual class of the instance

    def benchmark(self) -> Dict[str, Any]:
        """
        Benchmarks BaseGame.load() vs game_instance.clone().

        Returns:
            A dictionary with average times and all trial times for 'load' and 'clone'.
        """
        load_times: List[float] = []
        LOGGER.info(f"Starting benchmark for {self.game_class.__name__}.load() over {self.num_trials} trials...")
        for i in range(self.num_trials):
            start_time = time.perf_counter()
            _ = self.game_class.load(self.game_dict)
            end_time = time.perf_counter()
            load_times.append(end_time - start_time)
            if (i + 1) % (max(1, self.num_trials // 10)) == 0:
                LOGGER.debug(f"{self.game_class.__name__}.load() trial {i+1}/{self.num_trials} completed.")
        avg_load_time = sum(load_times) / self.num_trials if self.num_trials > 0 else 0
        LOGGER.info(f"Finished benchmark for {self.game_class.__name__}.load(). Average time: {avg_load_time:.6f} seconds.")

        clone_times: List[float] = []
        LOGGER.info(f"Starting benchmark for game.clone() over {self.num_trials} trials...")
        for i in range(self.num_trials):
            profiler = cProfile.Profile()
            start_time = time.perf_counter()
            profiler.enable()
            cloned_game = self.game_instance.manual_big_clone()
            profiler.disable()
            end_time = time.perf_counter()
            if i == 0:
                stats = pstats.Stats(profiler).sort_stats('cumulative')
                stats.print_stats(30) # Print top 30 cumulative time consumers
            clone_times.append(end_time - start_time)
            # Basic check to ensure clone is different (optional, adds minor overhead)
            assert cloned_game is not self.game_instance, "Clone method returned the same instance!"
            # You might add more assertions here, e.g., checking a few key attributes are deeply copied.
            if (i + 1) % (max(1, self.num_trials // 10)) == 0:
                LOGGER.debug(f"game.clone() trial {i+1}/{self.num_trials} completed.")
        avg_clone_time = sum(clone_times) / self.num_trials if self.num_trials > 0 else 0
        LOGGER.info(f"Finished benchmark for game.clone(). Average time: {avg_clone_time:.6f} seconds.")
        
        return {
            "load_avg_time": avg_load_time,
            "clone_avg_time": avg_clone_time,
            "load_times_all": load_times,
            "clone_times_all": clone_times,
        }

    def print_comparison(self, results: Dict[str, Any]):
        """Prints a formatted comparison of the benchmark results."""
        avg_load_time = results["load_avg_time"]
        avg_clone_time = results["clone_avg_time"]

        print("\n--- Benchmark Comparison ---")
        print(f"Method                     | Average Time (s) | Trials")
        print(f"---------------------------|------------------|--------")
        print(f"{self.game_class.__name__}.load()          | {avg_load_time:<16.6f} | {self.num_trials}")
        print(f"game_instance.clone()      | {avg_clone_time:<16.6f} | {self.num_trials}")

        if avg_clone_time > 0 and avg_load_time > 0 :
            if avg_load_time > avg_clone_time:
                # Ensure avg_clone_time is not zero to avoid DivisionByZeroError
                speedup = avg_load_time / avg_clone_time if avg_clone_time else float('inf')
                print(f"\nCloning is approximately {speedup:.2f} times faster than loading.")
            elif avg_clone_time > avg_load_time:
                 # Ensure avg_load_time is not zero
                slowdown = avg_clone_time / avg_load_time if avg_load_time else float('inf')
                print(f"\nLoading is approximately {slowdown:.2f} times faster than cloning.")
            else:
                print(f"\nCloning and loading have similar performance.")

        elif avg_clone_time == 0 and avg_load_time > 0:
             print(f"\nCloning is significantly faster (near instantaneous compared to loading).")
        elif avg_load_time == 0 and avg_clone_time > 0:
             print(f"\nLoading is significantly faster (near instantaneous compared to cloning) - this would be unexpected.")
        else: # Both are zero or one is zero and the other is not positive
            print(f"\nCould not determine speedup due to zero or non-positive average times.")
        print("--------------------------\n")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    
    game_class = GameMap().game_by_title("1830")
    players = {1: "Player 1", 2: "Player 2", 3: "Player 3", 4: "Player 4"}
    initial_game = game_class(players)
    action_helper = ActionHelper()
    for _ in range(100):
        initial_game.process_action(action_helper.get_all_choices(initial_game)[0])

    # 2. Benchmark load vs clone
    print("\nBenchmarking BaseGame.load() vs game.clone()...")
    benchmarker = BaseGameCloneBenchmarker(initial_game, num_trials=100)
    results = benchmarker.benchmark()
    benchmarker.print_comparison(results) 
