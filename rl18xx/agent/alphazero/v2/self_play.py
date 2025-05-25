from rl18xx.game.gamemap import GameMap
from rl18xx.game.engine.game import BaseGame
import random
import os
import socket
import time
import rl18xx.agent.alphazero.v2.mcts as mcts
from rl18xx.agent.alphazero.v2.model import AlphaZeroModel
from rl18xx.agent.alphazero.v2.config import MegaConfig, ModelConfig
import logging
from rl18xx.agent.alphazero.v2.dataset import TrainingExampleProcessor
import numpy as np
from typing import Any, Optional
import torch
import gc
import psutil

LOGGER = logging.getLogger(__name__)

# Helper function to log memory usage
def log_memory_usage(stage_name=""):
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    LOGGER.info(f"Memory usage {stage_name}: RSS={mem_info.rss / 1024**2:.2f} MB, VMS={mem_info.vms / 1024**2:.2f} MB")


class MCTSPlayer:
    def __init__(self, config: MegaConfig):
        self.config = config
        self.network = config.network
        self.root = None
        self.result = np.zeros(getattr(config, 'num_distinct_players', 4))
        super().__init__()
        self.initialize_game()

    def get_game_state(self):
        return BaseGame.load(self.root.game_dict) if self.root else None

    def get_root(self):
        return self.root

    def get_result_string(self):
        return self.result_string

    def get_new_game_state(self):
        game_map = GameMap()
        game_class = game_map.game_by_title("1830")
        players = {"1": "Player 1", "2": "Player 2", "3": "Player 3", "4": "Player 4"}
        return game_class(players)

    def initialize_game(self, game_state=None):
        log_memory_usage(stage_name="Before MCTSPlayer.initialize_game")
        if game_state is None:
            game_state = self.get_new_game_state()

        self.root = mcts.MCTSNode(game_state, config=self.config)
        self.result = np.zeros(len(game_state.players))
        self.result_string = None
        self.searches_pi = []
        LOGGER.info(f"Initialized game. Root node N: {self.root.N}")
        log_memory_usage(stage_name="After MCTSPlayer.initialize_game")

    def suggest_move(self, game_state):
        start_time = time.time()
        log_memory_usage(stage_name="Start MCTSPlayer.suggest_move")

        current_readouts = self.root.N
        while self.root.N < current_readouts + self.config.num_readouts:
            self.tree_search()
        LOGGER.info(f"{game_state['move_number']}: Searched {self.config.num_readouts} times in {time.time() - start_time} seconds. Root N: {self.root.N}, Children: {len(self.root.children if self.root.children else [])}")

        # print some stats on moves considered.
        LOGGER.debug(f"Root description: {self.root.describe()}")
        LOGGER.debug(f"Game State: {self.root.game_dict}")

        log_memory_usage(stage_name="End MCTSPlayer.suggest_move")
        return self.pick_move()

    def play_move(self, action_index):
        """Notable side effects:
          - finalizes the probability distribution according to
          this roots visit counts into the class' running tally, `searches_pi`
          - Makes the node associated with this move the root, for future
            `inject_noise` calls.
        """
        log_memory_usage(stage_name="Before MCTSPlayer.play_move")
        self.searches_pi.append(self.root.children_as_pi(self.root.game_dict["move_number"] < self.config.softpick_move_cutoff))
        
        self.root = self.root.maybe_add_child(action_index)
        # Prune the tree
        self.prune_mcts_tree_retain_parent(self.root)

        LOGGER.debug(f"Played move. New root N: {self.root.N}, Searches_pi length: {len(self.searches_pi)}")
        log_memory_usage(stage_name="After MCTSPlayer.play_move")

        self.game_state = self.root.game_dict  # for showboard
        return True

    def pick_move(self):
        """Picks a move to play, based on MCTS readout statistics.

        Highest N is most robust indicator. In the early stage of the game, pick
        a move weighted by visit count; later on, pick the absolute max."""
        if self.root.game_dict["move_number"] >= self.config.softpick_move_cutoff:
            return self.root.best_child()
        
        cdf = self.root.children_as_pi(squash=True).cumsum()
        selection = random.random()
        action_index = cdf.searchsorted(selection)
        if self.root.child_N[action_index] == 0:
            LOGGER.error(f"Action index {action_index} has no visits. Root children_as_pi: {self.root.children_as_pi(squash=True)}")
            raise ValueError(f"Action index {action_index} has no visits. Root children_as_pi: {self.root.children_as_pi(squash=True)}")
        return action_index


    def tree_search(self, parallel_readouts=None):
        if parallel_readouts is None:
            parallel_readouts = min(self.config.parallel_readouts, self.config.num_readouts)
        leaves = []
        failsafe = 0
        # Add a loop counter for debugging the selection of leaves
        select_leaf_attempts = 0
        max_select_leaf_attempts = parallel_readouts * 2 # This is the existing failsafe limit

        select_leaves_start = time.time()
        while len(leaves) < parallel_readouts and failsafe < max_select_leaf_attempts:
            select_leaf_attempts += 1
            failsafe += 1
            leaf = self.root.select_leaf()
            LOGGER.info(f"tree_search: Selected leaf for move {leaf.game_dict['move_number']}, action {leaf.fmove}")
            if leaf.is_done():
                LOGGER.info(f"tree_search: Found finished game for leaf {leaf.game_dict}. Result: {leaf.game_result_string()}")
                value = leaf.game_result()
                leaf.backup_value(value, up_to=self.root)
                continue

            leaf.add_virtual_loss(up_to=self.root)
            leaves.append(leaf)
        select_leaves_end = time.time()

        LOGGER.info(f"tree_search: Found {len(leaves)} leaves after {select_leaf_attempts} attempts.")

        if select_leaf_attempts >= max_select_leaf_attempts and len(leaves) < parallel_readouts:
            LOGGER.warning(f"tree_search: Failsafe triggered while selecting leaves. Found {len(leaves)}/{parallel_readouts} leaves after {select_leaf_attempts}/{max_select_leaf_attempts} attempts.")

        run_network_start = 0
        run_network_end = 0
        revert_and_incorporate_start = 0
        revert_and_incorporate_end = 0
        if leaves:
            run_network_start = time.time()
            with torch.no_grad():
                move_probs, _, values = self.network.run_many_encoded([leaf.encoded_game_state for leaf in leaves])
            run_network_end = time.time()
            
            revert_and_incorporate_start = time.time()
            for i, (leaf, move_prob, value) in enumerate(zip(leaves, move_probs, values)):
                # LOGGER.debug(f"tree_search: Processing leaf {i+1}/{len(leaves)}")
                leaf.revert_virtual_loss(up_to=self.root)
                leaf.incorporate_results(move_prob, value, up_to=self.root)
            revert_and_incorporate_end = time.time()
        
        LOGGER.info(f"Tree search timing: Select leaves: {select_leaves_end - select_leaves_start} seconds," +
                    f"Run network: {run_network_end - run_network_start} seconds, Revert and incorporate: " +
                    f"{revert_and_incorporate_end - revert_and_incorporate_start} seconds")
        return leaves

    def show_path_to_root(self, node: mcts.MCTSNode):
        pos = node.game_dict
        diff = node.game_dict["move_number"] - self.root.game_dict["move_number"]
        if len(pos["actions"]) == 0:
            return

        path = " ".join(str(move) for move in pos["actions"][-diff:])
        if node.game_dict["move_number"] >= self.config.max_game_length:
            path += f" (depth cutoff reached) {node.game_result_string()}"
        elif node.game_dict["finished"]:
            path += f" (game over) {node.game_result_string()}"
        return path

    def is_done(self):
        return self.result != [0.0, 0.0, 0.0, 0.0] or self.root.is_done()

    def set_result(self, result):
        self.result = np.array(result)
        string = self.root.game_result_string()
        self.result_string = string

    def extract_data(self):
        log_memory_usage(stage_name="Start MCTSPlayer.extract_data")
        assert len(self.searches_pi) == self.root.game_dict["move_number"], f"searches_pi length {len(self.searches_pi)} != move_number {self.root.game_dict['move_number']}"
        assert not np.array_equal(self.result, [0.0, 0.0, 0.0, 0.0]), f"result {self.result} is 0"

        # Create a list of tuples shaped like this:
        # (game_state, searches_pi, result)
        game_state = self.get_new_game_state()
        for i, action in enumerate(self.root.game_dict["actions"]):
            # Consider returning the encoded game state instead of the game state object
            yield game_state, self.searches_pi[i], self.result
            game_state = game_state.clone(game_state.raw_actions)
            game_state.process_action(action)

    def get_num_readouts(self):
        return self.num_readouts

    def set_num_readouts(self, readouts):
        self.num_readouts = readouts
    
    def _recursive_clear_references(self, node: mcts.MCTSNode, stats: Optional[dict] = None):
        """
        Recursively clears parent and children references to help with garbage collection.
        Sets node.parent to None and clears node.children.
        Also explicitly clears large data attributes of the node.
        This is called on subtrees that are being pruned.
        """
        if stats is not None:
            stats['cleared_nodes'] = stats.get('cleared_nodes', 0) + 1

        node.parent = None # Break link to its parent
        
        # Explicitly clear large attributes to help GC
        node.game_dict = None
        node.encoded_game_state = None
        # If there are other large numpy arrays specific to the node that are safe to clear,
        # they could be added here too. For now, game_dict and encoded_game_state are primary.
                                    
        children_to_visit = list(node.children.values())
        node.children.clear() # Clear this node's children dict

        for child in children_to_visit:
            self._recursive_clear_references(child, stats) # Recurse


    def prune_mcts_tree_retain_parent(self, new_search_root: mcts.MCTSNode):
        parent_of_new_root = new_search_root.parent

        # Case 1: new_search_root is the first real node of the game (its parent is DummyNode).
        if isinstance(parent_of_new_root, mcts.DummyNode) or new_search_root.fmove is None:
            LOGGER.info(f"Noop during pruning: new root is the initial game root or has a DummyNode parent.")
            return

        LOGGER.info(f"Pruning tree. New root: {new_search_root}, its parent (old root): {parent_of_new_root}")
        # Identify and collect siblings of new_search_root for pruning.
        siblings_to_prune_roots = []
        for action_index, child_node in parent_of_new_root.children.items():
            if action_index != new_search_root.fmove:
                siblings_to_prune_roots.append(child_node)

        # Modify parent_of_new_root.children to only contain new_search_root.
        parent_of_new_root.children.clear()
        parent_of_new_root.children[new_search_root.fmove] = new_search_root
        LOGGER.debug(f"Pruning: {parent_of_new_root} now only has child {new_search_root}.")

        # Recursively clear references in the pruned sibling subtrees.
        pruning_stats = {'cleared_nodes': 0, 'sibling_subtrees_pruned': 0}
        for sibling_root in siblings_to_prune_roots:
            LOGGER.debug(f"Pruning: Clearing references for subtree rooted at {sibling_root}.")
            pruning_stats['sibling_subtrees_pruned'] += 1
            self._recursive_clear_references(sibling_root, stats=pruning_stats)
        
        LOGGER.info(f"Pruning stats: Sibling subtrees pruned: {pruning_stats['sibling_subtrees_pruned']}, Total nodes cleared in subtrees: {pruning_stats['cleared_nodes']}")

        # Detach parent_of_new_root from its original parent (pruning ancestors).
        # Its new parent becomes a DummyNode.
        LOGGER.debug(f"Pruning: Setting parent of {parent_of_new_root} to DummyNode.")
        parent_of_new_root.parent = mcts.DummyNode()


class SelfPlay:
    def __init__(self, config: MegaConfig, model_config: Optional[ModelConfig] = None):
        self.config = config
        assert config.network is not None or model_config is not None, "Network must be provided"
        if model_config is not None:
            self.config.network = AlphaZeroModel(model_config)
        self.config.network.eval()
        log_memory_usage(stage_name="SelfPlay initialized")

    def play(self):
        """Plays out a self-play match, returning a MCTSPlayer object containing:
            - the final position
            - the n x 26535 tensor of floats representing the mcts search probabilities
            - the n x 4 tensor of floats representing the original value-net estimate
            where n is the number of moves in the game
        """
        player = MCTSPlayer(self.config)
        log_memory_usage(stage_name="After MCTSPlayer created in SelfPlay.play")

        # Must run this once at the start to expand the root node.
        first_node = player.root.select_leaf()
        with torch.no_grad():
            probs, _, val = self.config.network.run_encoded(first_node.encoded_game_state)
        first_node.incorporate_results(probs, val, first_node)
        del first_node
        log_memory_usage(stage_name="After first node expansion in SelfPlay.play")

        move_counter = 0
        try:
            while True:
                log_memory_usage(stage_name=f"SelfPlay.play loop start, move {move_counter}")
                start_time = time.time()
                player.root.inject_noise()
                current_readouts = player.root.N
                target_readouts_for_move = current_readouts + self.config.num_readouts
                
                # MCTS simulation loop
                sim_count_this_move = 0
                max_sims_display_interval = max(1, self.config.num_readouts // 10) # Log progress roughly 10 times per move

                tree_search_start_time = time.time()
                while player.root.N < target_readouts_for_move:
                    if sim_count_this_move > 0 and sim_count_this_move % max_sims_display_interval == 0:
                        LOGGER.debug(f"Move {move_counter}: MCTS sim progress: {player.root.N}/{target_readouts_for_move} (sim_batch {sim_count_this_move})")
                        log_memory_usage(stage_name=f"SelfPlay.play move {move_counter}, MCTS sim {player.root.N}")
                    
                    player.tree_search() # This internally calls network.run_many for parallel_readouts
                    sim_count_this_move += self.config.parallel_readouts
                tree_search_end_time = time.time()

                LOGGER.info(f"Move {move_counter}: MCTS simulations complete. Root N: {player.root.N}, Target: {target_readouts_for_move}")
                log_memory_usage(stage_name=f"SelfPlay.play after MCTS simulations for move {move_counter}")

                pick_move_start_time = time.time()
                move = player.pick_move()
                pick_move_end_time = time.time()
                LOGGER.info(f"Selected move: {move}")
                play_move_start_time = time.time()
                player.play_move(move)
                play_move_end_time = time.time()
                move_counter += 1

                end_time = time.time()
                LOGGER.info(f"Timing Info: Total: {end_time - start_time}, Tree Search: {tree_search_end_time - tree_search_start_time}, Pick Move: {pick_move_end_time - pick_move_start_time}, Play Move: {play_move_end_time - play_move_start_time}")
                if move_counter % 20 == 0:
                    log_memory_usage(stage_name=f"SelfPlay.play during game, after move {move_counter}")
                    LOGGER.info(f"GC counts: {gc.get_count()}")

                if player.root.is_done():
                    player.set_result(player.root.game_result())
                    LOGGER.info(f"Game finished after {move_counter} moves. Result: {player.get_result_string()}")
                    break


        except Exception as e:
            LOGGER.error(f"Error in self-play after {move_counter} moves: {e}", exc_info=True)
            LOGGER.error(f"Game state: {player.root.game_dict}")
            # It might be useful to still try and get data from the player if an error occurs mid-game
            # For now, just re-raise or handle as per existing logic.
        
        log_memory_usage(stage_name="SelfPlay.play finished")
        return player


    def run_game(self):
        """Takes a played game and record results and game data."""
        log_memory_usage(stage_name="Start SelfPlay.run_game")
        if self.config.selfplay_dir is not None:
            os.makedirs(self.config.selfplay_dir, exist_ok=True)
            os.makedirs(self.config.holdout_dir, exist_ok=True)

        player = self.play()
        log_memory_usage(stage_name="After SelfPlay.play in run_game")

        output_name = '{}-{}'.format(int(time.time()), socket.gethostname())
        
        if player.result is None or np.all(np.array(player.result) == 0.0):
             LOGGER.warning(f"Game {output_name} finished with no conclusive result or result not set. Skipping data extraction.")
             return

        game_data = player.extract_data()
        log_memory_usage(stage_name="After player.extract_data in run_game")


        processor = TrainingExampleProcessor()
        game_examples = processor.make_dataset_from_selfplay(game_data)
        log_memory_usage(stage_name="After make_dataset_from_selfplay in run_game")

        if self.config.selfplay_dir is not None:
            # Hold out 5% of games for validation.
            if random.random() < self.config.holdout_pct:
                fname = self.config.holdout_dir
            else:
                fname = self.config.selfplay_dir

            processor.write_lmdb(game_examples, fname)
            log_memory_usage(stage_name="After write_lmdb in run_game")

        # Explicitly delete large objects and collect garbage
        del player
        del game_data
        del game_examples
        gc.collect()
        LOGGER.info("Explicitly deleted player, game_data, game_examples and ran gc.collect()")
        log_memory_usage(stage_name="End SelfPlay.run_game after GC")


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

def main():
    setup_logging(logging.INFO, "self_play.log")
    log_memory_usage(stage_name="Start of main")
    model_config = ModelConfig()
    config = MegaConfig()
    selfplay = SelfPlay(config, model_config=model_config)

    num_games_to_run = getattr(config, 'num_games_to_run', 1)
    for i in range(num_games_to_run):
        LOGGER.info(f"--- Starting game {i+1}/{num_games_to_run} ---")
        selfplay.run_game()
        log_memory_usage(stage_name=f"After game {i+1}/{num_games_to_run}")
        LOGGER.info(f"GC counts after game {i+1}: {gc.get_count()}")


if __name__ == '__main__':
    main()

