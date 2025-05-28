#!/usr/bin/env python3
"""Profile MCTS operations to identify the specific bottleneck in node expansion."""

import time
import cProfile
import pstats
import io
import gc
import numpy as np
from typing import Dict, Any

from rl18xx.game.gamemap import GameMap
from rl18xx.game.engine.game.base import BaseGame
from rl18xx.game import ActionHelper
from rl18xx.agent.alphazero.mcts import MCTSNode
from rl18xx.agent.alphazero.config import SelfPlayConfig
from rl18xx.agent.alphazero.action_mapper import ActionMapper


class MCTSProfiler:
    """Profile MCTS operations to identify bottlenecks."""
    
    def __init__(self):
        self.timing_results = {}
        
    def profile_node_expansion(self, num_actions: int = 50) -> Dict[str, Any]:
        """Profile the critical MCTS node expansion path."""
        print(f"\n=== Profiling MCTS Node Expansion with {num_actions} actions ===")
        
        # Create a game with some actions
        game_map = GameMap()
        game_class = game_map.game_by_title("1830")
        players = {"1": "Player 1", "2": "Player 2", "3": "Player 3", "4": "Player 4"}
        game = game_class(players)
        action_helper = ActionHelper()
        
        # Play some moves
        for i in range(num_actions):
            all_actions = action_helper.get_all_choices(game)
            if not all_actions:
                break
            game.process_action(all_actions[0])
        
        # Create config
        config = SelfPlayConfig()
        
        # Profile MCTSNode creation
        print("\n1. Profiling MCTSNode creation from game state:")
        node_creation_times = []
        for i in range(5):
            gc.collect()
            start = time.time()
            node = MCTSNode(game, config=config)
            duration = time.time() - start
            node_creation_times.append(duration)
            print(f"   Iteration {i+1}: {duration:.3f}s")
        
        avg_node_creation = np.mean(node_creation_times)
        print(f"   Average: {avg_node_creation:.3f}s")
        
        # Profile maybe_add_child with BaseGame.load
        print("\n2. Profiling maybe_add_child with BaseGame.load:")
        
        # Get game dict for loading
        game_dict = game.to_dict()
        action_mapper = ActionMapper()
        legal_actions = action_mapper.get_legal_action_indices(game)
        
        load_times = []
        load_process_times = []
        
        for i in range(min(10, len(legal_actions))):
            gc.collect()
            
            # Time BaseGame.load
            start_load = time.time()
            new_game = BaseGame.load(game_dict)
            load_duration = time.time() - start_load
            load_times.append(load_duration)
            
            # Time action processing
            start_process = time.time()
            action_to_take = action_mapper.map_index_to_action(legal_actions[i], new_game)
            new_game.process_action(action_to_take)
            process_duration = time.time() - start_process
            load_process_times.append(process_duration)
            
            print(f"   Action {i+1}: Load={load_duration:.3f}s, Process={process_duration:.3f}s")
        
        # Profile maybe_add_child with game.clone()
        print("\n3. Profiling maybe_add_child with game.clone():")
        
        clone_times = []
        clone_process_times = []
        
        for i in range(min(10, len(legal_actions))):
            gc.collect()
            
            # Time game.clone()
            start_clone = time.time()
            new_game = game.clone(game.raw_actions)
            clone_duration = time.time() - start_clone
            clone_times.append(clone_duration)
            
            # Time action processing
            start_process = time.time()
            action_to_take = action_mapper.map_index_to_action(legal_actions[i], new_game)
            new_game.process_action(action_to_take)
            process_duration = time.time() - start_process
            clone_process_times.append(process_duration)
            
            print(f"   Action {i+1}: Clone={clone_duration:.3f}s, Process={process_duration:.3f}s")
        
        results = {
            'node_creation': {
                'mean': avg_node_creation,
                'std': np.std(node_creation_times)
            },
            'baseload': {
                'load_mean': np.mean(load_times),
                'process_mean': np.mean(load_process_times),
                'total_mean': np.mean(load_times) + np.mean(load_process_times)
            },
            'clone': {
                'clone_mean': np.mean(clone_times),
                'process_mean': np.mean(clone_process_times),
                'total_mean': np.mean(clone_times) + np.mean(clone_process_times)
            }
        }
        
        print(f"\nSummary:")
        print(f"  Node creation average: {results['node_creation']['mean']:.3f}s")
        print(f"\n  BaseGame.load approach:")
        print(f"    - Load time: {results['baseload']['load_mean']:.3f}s")
        print(f"    - Process time: {results['baseload']['process_mean']:.3f}s") 
        print(f"    - Total: {results['baseload']['total_mean']:.3f}s")
        print(f"\n  game.clone() approach:")
        print(f"    - Clone time: {results['clone']['clone_mean']:.3f}s")
        print(f"    - Process time: {results['clone']['process_mean']:.3f}s")
        print(f"    - Total: {results['clone']['total_mean']:.3f}s")
        print(f"\n  Speedup: {results['baseload']['total_mean'] / results['clone']['total_mean']:.1f}x")
        
        return results
    
    def profile_detailed_load(self, num_actions: int = 50):
        """Profile BaseGame.load in detail with cProfile."""
        print(f"\n=== Detailed profiling of BaseGame.load with {num_actions} actions ===")
        
        # Create a game with actions
        game_map = GameMap()
        game_class = game_map.game_by_title("1830")
        players = {"1": "Player 1", "2": "Player 2", "3": "Player 3", "4": "Player 4"}
        game = game_class(players)
        action_helper = ActionHelper()
        
        for i in range(num_actions):
            all_actions = action_helper.get_all_choices(game)
            if not all_actions:
                break
            game.process_action(all_actions[0])
        
        game_dict = game.to_dict()
        
        # Profile with cProfile
        profiler = cProfile.Profile()
        profiler.enable()
        
        # Run BaseGame.load multiple times
        for _ in range(5):
            BaseGame.load(game_dict)
        
        profiler.disable()
        
        # Print stats
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        ps.print_stats(50)  # Top 50 functions
        print(s.getvalue())
        
        # Also profile just the process_action part
        print("\n=== Detailed profiling of process_action during load ===")
        
        # Create a fresh game instance
        fresh_game = BaseGame.load(game_dict)
        fresh_game.raw_actions = []  # Clear actions so we can profile processing
        
        profiler2 = cProfile.Profile()
        profiler2.enable()
        
        # Process first 10 actions
        for action in game_dict["actions"][:10]:
            fresh_game.process_action(action)
        
        profiler2.disable()
        
        s2 = io.StringIO()
        ps2 = pstats.Stats(profiler2, stream=s2).sort_stats('cumulative')
        ps2.print_stats(30)
        print(s2.getvalue())


def main():
    """Run MCTS profiling."""
    profiler = MCTSProfiler()
    
    # Test with different numbers of actions
    for num_actions in [25, 50, 100]:
        profiler.profile_node_expansion(num_actions)
    
    # Run detailed profiling only if requested
    # profiler.profile_detailed_load(50)
    
    print("\n" + "="*60)
    print("MCTS Profiling complete!")


if __name__ == "__main__":
    main()