#!/usr/bin/env python3
"""Profiling interface for 1830 game instances to identify performance bottlenecks."""

import time
import cProfile
import pstats
import io
from functools import wraps
import numpy as np
import json
from typing import List, Dict, Any, Callable
import gc

from rl18xx.game.gamemap import GameMap
from rl18xx.game.engine.game.base import BaseGame
from rl18xx.game import ActionHelper


def timing_decorator(func: Callable) -> Callable:
    """Decorator to measure execution time of functions."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start
        wrapper.total_time += duration
        wrapper.call_count += 1
        return result
    wrapper.total_time = 0
    wrapper.call_count = 0
    return wrapper


class GameProfiler:
    """Profile game operations to identify performance bottlenecks."""
    
    def __init__(self):
        self.timing_results = {}
        
    def profile_game_initialization(self, num_iterations: int = 10) -> Dict[str, float]:
        """Profile creating new game instances."""
        print(f"\n=== Profiling Game Initialization ({num_iterations} iterations) ===")
        
        times = []
        for i in range(num_iterations):
            gc.collect()  # Clean slate
            start = time.time()
            
            game_map = GameMap()
            game_class = game_map.game_by_title("1830")
            players = {"1": "Player 1", "2": "Player 2", "3": "Player 3", "4": "Player 4"}
            game = game_class(players)
            
            duration = time.time() - start
            times.append(duration)
            print(f"  Iteration {i+1}: {duration:.3f}s")
            
        avg_time = np.mean(times)
        std_time = np.std(times)
        print(f"\nAverage initialization time: {avg_time:.3f}s ± {std_time:.3f}s")
        
        self.timing_results['game_init'] = {
            'mean': avg_time,
            'std': std_time,
            'min': np.min(times),
            'max': np.max(times)
        }
        return self.timing_results['game_init']
    
    def profile_game_loading(self, num_actions: List[int] = [0, 10, 50, 100, 200]) -> Dict[str, Any]:
        """Profile loading games from different numbers of actions."""
        print(f"\n=== Profiling Game Loading from Actions ===")
        
        # First create a game and play some moves to get actions
        game_map = GameMap()
        game_class = game_map.game_by_title("1830")
        players = {"1": "Player 1", "2": "Player 2", "3": "Player 3", "4": "Player 4"}
        base_game = game_class(players)
        action_helper = ActionHelper()
        
        # Play moves to accumulate actions
        actions_list = []
        for i in range(max(num_actions)):
            all_actions = action_helper.get_all_choices(base_game)
            if not all_actions:
                print(f"No more actions available after {i} moves")
                break
            # Pick a random action
            action = all_actions[0]
            base_game.process_action(action)
            # Store the raw action dict instead of the action object
            actions_list.append(action.to_dict())
        
        results = {}
        
        for n in num_actions:
            if n > len(actions_list):
                print(f"  Skipping {n} actions (only {len(actions_list)} available)")
                continue
                
            print(f"\n  Testing with {n} actions:")
            
            # Create game dict with n actions
            game_dict = {
                "id": "profile_test",
                "players": [{"name": f"Player {i+1}", "id": str(i+1)} for i in range(4)],
                "title": "1830",
                "description": "",
                "min_players": "4",
                "max_players": "4",
                "settings": {
                    "optional_rules": [],
                    "seed": ""
                },
                "actions": actions_list[:n]
            }
            
            times = []
            for i in range(5):  # Fewer iterations for longer operations
                gc.collect()
                start = time.time()
                loaded_game = BaseGame.load(game_dict)
                duration = time.time() - start
                times.append(duration)
                print(f"    Iteration {i+1}: {duration:.3f}s")
            
            avg_time = np.mean(times)
            results[n] = {
                'mean': avg_time,
                'std': np.std(times),
                'min': np.min(times),
                'max': np.max(times)
            }
            print(f"  Average: {avg_time:.3f}s")
        
        self.timing_results['game_loading'] = results
        return results
    
    def profile_action_processing(self, num_moves: int = 100) -> Dict[str, Any]:
        """Profile processing individual actions."""
        print(f"\n=== Profiling Action Processing ({num_moves} moves) ===")
        
        # Initialize game
        game_map = GameMap()
        game_class = game_map.game_by_title("1830")
        players = {"1": "Player 1", "2": "Player 2", "3": "Player 3", "4": "Player 4"}
        game = game_class(players)
        action_helper = ActionHelper()
        
        action_times = []
        get_choices_times = []
        
        for i in range(num_moves):
            # Time getting all choices
            gc.collect()
            start = time.time()
            all_actions = action_helper.get_all_choices(game)
            get_choices_duration = time.time() - start
            get_choices_times.append(get_choices_duration)
            
            if not all_actions:
                print(f"  No more actions after {i} moves")
                break
            
            # Time processing action
            action = all_actions[0]
            start = time.time()
            game.process_action(action)
            action_duration = time.time() - start
            action_times.append(action_duration)
            
            # Refresh helper
            if i % 20 == 0:
                print(f"  Processed {i} actions...")
        
        results = {
            'get_choices': {
                'mean': np.mean(get_choices_times),
                'std': np.std(get_choices_times),
                'min': np.min(get_choices_times),
                'max': np.max(get_choices_times),
                'total': np.sum(get_choices_times)
            },
            'process_action': {
                'mean': np.mean(action_times),
                'std': np.std(action_times),
                'min': np.min(action_times),
                'max': np.max(action_times),
                'total': np.sum(action_times)
            }
        }
        
        print(f"\nGet choices average: {results['get_choices']['mean']:.3f}s")
        print(f"Process action average: {results['process_action']['mean']:.3f}s")
        print(f"Total get choices time: {results['get_choices']['total']:.1f}s")
        print(f"Total process action time: {results['process_action']['total']:.1f}s")
        
        self.timing_results['action_processing'] = results
        return results
    
    def profile_with_cprofile(self, func: Callable, *args, **kwargs):
        """Run a function with cProfile to get detailed profiling."""
        print(f"\n=== Detailed cProfile for {func.__name__} ===")
        
        profiler = cProfile.Profile()
        profiler.enable()
        
        result = func(*args, **kwargs)
        
        profiler.disable()
        
        # Print stats
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        ps.print_stats(30)  # Top 30 functions
        print(s.getvalue())
        
        return result
    
    def profile_game_dict_operations(self) -> Dict[str, Any]:
        """Profile to_dict() and from dict operations."""
        print(f"\n=== Profiling Game Dict Operations ===")
        
        # Initialize game and play some moves
        game_map = GameMap()
        game_class = game_map.game_by_title("1830")
        players = {"1": "Player 1", "2": "Player 2", "3": "Player 3", "4": "Player 4"}
        game = game_class(players)
        action_helper = ActionHelper()
        
        # Play 50 moves
        for i in range(50):
            all_actions = action_helper.get_all_choices(game)
            if not all_actions:
                break
            game.process_action(all_actions[0])
        
        # Profile to_dict
        to_dict_times = []
        for i in range(10):
            gc.collect()
            start = time.time()
            game_dict = game.to_dict()
            duration = time.time() - start
            to_dict_times.append(duration)
            print(f"  to_dict iteration {i+1}: {duration:.3f}s")
        
        # Profile BaseGame.load
        load_times = []
        for i in range(10):
            gc.collect()
            start = time.time()
            loaded_game = BaseGame.load(game_dict)
            duration = time.time() - start
            load_times.append(duration)
            print(f"  load iteration {i+1}: {duration:.3f}s")
        
        results = {
            'to_dict': {
                'mean': np.mean(to_dict_times),
                'std': np.std(to_dict_times),
                'min': np.min(to_dict_times),
                'max': np.max(to_dict_times)
            },
            'load': {
                'mean': np.mean(load_times),
                'std': np.std(load_times),
                'min': np.min(load_times),
                'max': np.max(load_times)
            }
        }
        
        print(f"\nto_dict average: {results['to_dict']['mean']:.3f}s")
        print(f"load average: {results['load']['mean']:.3f}s")
        
        self.timing_results['dict_operations'] = results
        return results
    
    def save_results(self, filename: str = "profiling_results.json"):
        """Save profiling results to a JSON file."""
        with open(filename, 'w') as f:
            json.dump(self.timing_results, f, indent=2)
        print(f"\nResults saved to {filename}")


def main():
    """Run all profiling tests."""
    profiler = GameProfiler()
    
    # Run basic profiling
    profiler.profile_game_initialization(num_iterations=5)
    profiler.profile_game_loading(num_actions=[0, 10, 25, 50, 100])
    profiler.profile_action_processing(num_moves=50)
    profiler.profile_game_dict_operations()
    
    # Run detailed profiling on game loading
    print("\n" + "="*60)
    print("Running detailed cProfile on game loading...")
    
    # Create a test game dict with 50 actions
    game_map = GameMap()
    game_class = game_map.game_by_title("1830")
    players = {"1": "Player 1", "2": "Player 2", "3": "Player 3", "4": "Player 4"}
    game = game_class(players)
    action_helper = ActionHelper()
    
    actions = []
    for i in range(50):
        all_actions = action_helper.get_all_choices(game)
        if not all_actions:
            break
        action = all_actions[0]
        game.process_action(action)
        actions.append(action.to_dict())
    
    game_dict = {
        "id": "profile_test",
        "players": [{"name": f"Player {i+1}", "id": str(i+1)} for i in range(4)],
        "title": "1830",
        "description": "",
        "min_players": "4",
        "max_players": "4",
        "settings": {
            "optional_rules": [],
            "seed": ""
        },
        "actions": actions
    }
    
    # Profile BaseGame.load with cProfile
    profiler.profile_with_cprofile(BaseGame.load, game_dict)
    
    # Save results
    profiler.save_results()
    
    print("\n" + "="*60)
    print("Profiling complete!")


if __name__ == "__main__":
    main()