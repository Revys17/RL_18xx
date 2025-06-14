from rl18xx.game.gamemap import GameMap
from rl18xx.game.engine.game import BaseGame
import random
import os
import socket
import time
from torch.utils.tensorboard import SummaryWriter
import rl18xx.agent.alphazero.mcts as mcts
from rl18xx.agent.alphazero.model import AlphaZeroModel
from rl18xx.agent.alphazero.config import SelfPlayConfig, ModelConfig
from rl18xx.agent.alphazero.checkpointer import get_latest_model
from rl18xx.agent.alphazero.dataset import TrainingExampleProcessor
from rl18xx.agent.alphazero.action_mapper import ActionMapper
from rl18xx.agent.agent import Agent
import numpy as np
from typing import Optional, List, Tuple, Generator
import torch
import gc
import psutil
import json
from datetime import datetime
import logging
from pathlib import Path

LOGGER = logging.getLogger(__name__)
SELF_PLAY_GAMES_STATUS_PATH = Path("self_play_games_status")
SELF_PLAY_GAMES_STATUS_PATH.mkdir(parents=True, exist_ok=True)

class MCTSPlayer(Agent):
    def __init__(self, config: SelfPlayConfig):
        self.config = config
        self.network = config.network
        self.initialize_game()

    def __str__(self):
        return f"MCTSPlayer"
    
    def __repr__(self):
        return self.__str__()

    def add_metric(self, name, value):
        if self.config.metrics is None:
            return
        self.config.metrics.add_scalar(name, value, self.config.global_step, self.config.game_idx_in_iteration)

    def log_memory_usage(self, stage_name: str):
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        self.add_metric(f"{stage_name}/Memory/RSS", mem_info.rss / 1024**2)
        self.add_metric(f"{stage_name}/Memory/VMS", mem_info.vms / 1024**2)

    def add_histogram(self, name, values):
        if self.config.metrics is None:
            return
        self.config.metrics.add_histogram(name, values, self.config.global_step, self.config.game_idx_in_iteration)

    def get_game_state(self):
        return self.root.game_object

    def get_root(self):
        return self.root

    def get_result_string(self):
        return self.result_string

    def get_new_game_state(self):
        game_map = GameMap()
        game_class = game_map.game_by_title("1830")
        players = {1: "Player 1", 2: "Player 2", 3: "Player 3", 4: "Player 4"}
        return game_class(players)

    def initialize_game(self, game_state: Optional[BaseGame] = None):
        if game_state is None:
            game_state = self.get_new_game_state()

        self.root = mcts.MCTSNode(game_state, config=self.config)
        self.result = np.zeros(len(game_state.players))
        self.result_string = None
        self.searches_pi = []
        LOGGER.info(f"Initialized game. Root node N: {self.root.N}")

    def play_move(self, action_index):
        """Notable side effects:
        - finalizes the probability distribution according to
        this roots visit counts into the class' running tally, `searches_pi`
        - Makes the node associated with this move the root, for future
          `inject_noise` calls.
        """
        self.log_memory_usage(stage_name="MCTSPlayer.play_move")
        self.searches_pi.append(
            self.root.children_as_pi(self.root.game_object.move_number < self.config.softpick_move_cutoff)
        )

        self.root = self.root.maybe_add_child(action_index)
        # Prune the tree
        self.prune_mcts_tree_retain_parent(self.root)

        LOGGER.debug(f"Played move. New root N: {self.root.N}, Searches_pi length: {len(self.searches_pi)}")
        self.log_memory_usage(stage_name="MCTSPlayer.play_move")

        self.game_state = self.root.game_object.to_dict()  # for showboard
        return True

    def pick_move(self):
        """Picks a move to play, based on MCTS readout statistics.

        Highest N is most robust indicator. In the early stage of the game, pick
        a move weighted by visit count; later on, pick the absolute max."""
        if self.root.game_object.move_number >= self.config.softpick_move_cutoff:
            return self.root.best_child()
        
        if self.root.num_legal_actions == 1:
            return self.root.legal_action_indices[0]

        cdf = self.root.children_as_pi(squash=True).cumsum()
        selection = random.random()
        action_index = cdf.searchsorted(selection)
        if action_index >= len(self.root.children_as_pi(squash=True)):
            LOGGER.error(
                f"Action index {action_index} is out of bounds. Root children_as_pi: {self.root.children_as_pi(squash=True)}"
            )
            raise ValueError(
                f"Action index {action_index} is out of bounds. Root children_as_pi: {self.root.children_as_pi(squash=True)}"
            )

        if self.root.child_N[action_index] == 0:
            LOGGER.error(
                f"Action index {action_index} has no visits. Root children_as_pi: {self.root.children_as_pi(squash=True)}"
            )
            raise ValueError(
                f"Action index {action_index} has no visits. Root children_as_pi: {self.root.children_as_pi(squash=True)}"
            )
        return action_index

    def tree_search(self, parallel_readouts=None):
        if parallel_readouts is None:
            parallel_readouts = min(self.config.parallel_readouts, self.config.num_readouts)

        # metrics
        leaf_depths_collected = []
        leaf_initial_qs_collected = []
        leaf_prior_entropies_collected = []

        leaves = []
        failsafe = 0
        select_leaf_attempts = 0
        max_select_leaf_attempts = parallel_readouts * 2

        select_leaves_start = time.time()
        while len(leaves) < parallel_readouts and failsafe < max_select_leaf_attempts:
            select_leaf_attempts += 1
            failsafe += 1
            leaf = self.root.select_leaf()
            if leaf.is_done():
                LOGGER.info(f"tree_search: Found finished game for leaf. Result: {leaf.game_result_string()}")
                self.add_metric("MCTS/Finished_Games", 1)
                value = leaf.game_result()
                leaf.backup_value(value, up_to=self.root)
                continue

            leaf.add_virtual_loss(up_to=self.root)
            leaves.append(leaf)
        select_leaves_end = time.time()

        self.add_metric("MCTS/Select_Leaf_Time", select_leaves_end - select_leaves_start)
        self.add_metric("MCTS/Select_Leaf_Attempts", select_leaf_attempts)
        self.add_metric("MCTS/Max_Select_Leaf_Attempts", max_select_leaf_attempts)
        self.add_metric("MCTS/Leaves_Found", len(leaves))

        if select_leaf_attempts >= max_select_leaf_attempts and len(leaves) < parallel_readouts:
            LOGGER.warning(
                f"tree_search: Failsafe triggered while selecting leaves. Found {len(leaves)}/{parallel_readouts} leaves after {select_leaf_attempts}/{max_select_leaf_attempts} attempts."
            )
            self.add_metric("MCTS/Failsafe_Triggered", 1)

        if leaves:
            run_network_start = time.time()
            with torch.no_grad():
                move_probs, _, values = self.network.run_many_encoded([leaf.encoded_game_state for leaf in leaves])
            run_network_duration = time.time() - run_network_start

            revert_and_incorporate_start = time.time()
            for i, (leaf, move_prob, value) in enumerate(zip(leaves, move_probs, values)):
                leaf.revert_virtual_loss(up_to=self.root)
                leaf.incorporate_results(move_prob, value, up_to=self.root)

                # metrics
                leaf_depths_collected.append(leaf.depth)
                leaf_initial_qs_collected.append(value[leaf.active_player_index].item())
                if np.sum(leaf.child_prior_compressed) > 1e-6: # Ensure it's not all zeros
                    normalized_prior_compressed = leaf.child_prior_compressed / np.sum(leaf.child_prior_compressed)
                    leaf_prior_entropies_collected.append(mcts.calculate_entropy(normalized_prior_compressed))
            revert_and_incorporate_duration = time.time() - revert_and_incorporate_start
            self.add_histogram("MCTS_Player/TreeSearch_Leaf_Depths", np.array(leaf_depths_collected))
            self.add_histogram("MCTS_Player/TreeSearch_Leaf_Initial_Network_Q", np.array(leaf_initial_qs_collected))
            self.add_histogram("MCTS_Player/TreeSearch_Leaf_Prior_Entropies", np.array(leaf_prior_entropies_collected))
        else:
            run_network_duration = 0
            revert_and_incorporate_duration = 0

        self.add_metric("MCTS/Run_Network_Time", run_network_duration)
        self.add_metric("MCTS/Revert_And_Incorporate_Time", revert_and_incorporate_duration)
        return leaves

    def suggest_move(self):
        if self.root.num_legal_actions == 1:
            return self.pick_move()

        current_readouts = self.root.N
        target_readouts_for_move = current_readouts + self.config.num_readouts

        # MCTS simulation loop
        while self.root.N < target_readouts_for_move:
            self.tree_search()

        move = self.pick_move()
        return move

    def is_done(self):
        return self.result != [0.0, 0.0, 0.0, 0.0] or self.root.is_done()

    def set_result(self, result):
        self.result = np.array(result)
        string = self.root.game_result_string()
        self.result_string = string

    def extract_data(self) -> Generator[Tuple[BaseGame, torch.Tensor, torch.Tensor], None, None]:
        assert (
            len(self.searches_pi) == self.root.game_object.move_number
        ), f"searches_pi length {len(self.searches_pi)} != move_number {self.root.game_object.move_number}"
        assert not np.array_equal(self.result, [0.0, 0.0, 0.0, 0.0]), f"result {self.result} is 0"

        result = torch.tensor(self.result)
        game_state = self.get_new_game_state()
        action_mapper = ActionMapper()
        for i, action in enumerate(self.root.game_object.raw_actions):
            yield (
                game_state,
                torch.tensor(action_mapper.get_legal_action_indices(game_state)),
                torch.tensor(self.searches_pi[i])
                if isinstance(self.searches_pi[i], np.ndarray)
                else self.searches_pi[i],
                result,
            )
            game_state = game_state.deep_copy_clone()
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
            stats["cleared_nodes"] = stats.get("cleared_nodes", 0) + 1

        node.parent = None  # Break link to its parent

        # Explicitly clear large attributes to help GC
        node.game_object = None
        node.encoded_game_state = None
        # If there are other large numpy arrays specific to the node that are safe to clear,
        # they could be added here too. For now, game_object and encoded_game_state are primary.

        children_to_visit = list(node.children.values())
        node.children.clear()  # Clear this node's children dict

        for child in children_to_visit:
            self._recursive_clear_references(child, stats)  # Recurse

    def prune_mcts_tree_retain_parent(self, new_search_root: mcts.MCTSNode):
        parent_of_new_root = new_search_root.parent

        # Case 1: new_search_root is the first real node of the game (its parent is DummyNode).
        if isinstance(parent_of_new_root, mcts.DummyNode) or new_search_root.fmove is None:
            LOGGER.info(f"Noop during pruning: new root is the initial game root or has a DummyNode parent.")
            return

        LOGGER.debug(f"Pruning tree. New root: {new_search_root}, its parent (old root): {parent_of_new_root}")
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
        pruning_stats = {"cleared_nodes": 0, "sibling_subtrees_pruned": 0}
        for sibling_root in siblings_to_prune_roots:
            LOGGER.debug(f"Pruning: Clearing references for subtree rooted at {sibling_root}.")
            pruning_stats["sibling_subtrees_pruned"] += 1
            self._recursive_clear_references(sibling_root, stats=pruning_stats)

        self.add_metric("MCTS/Sibling_Subtrees_Pruned", pruning_stats["sibling_subtrees_pruned"])
        self.add_metric("MCTS/Total_Nodes_Cleared_In_Subtrees", pruning_stats["cleared_nodes"])

        # Detach parent_of_new_root from its original parent (pruning ancestors).
        # Its new parent becomes a DummyNode.
        LOGGER.debug(f"Pruning: Setting parent of {parent_of_new_root} to DummyNode.")
        parent_of_new_root.parent = mcts.DummyNode()


class SelfPlay:
    def __init__(self, config: SelfPlayConfig, model_config: Optional[ModelConfig] = None):
        self.config = config
        assert config.network is not None or model_config is not None, "Network must be provided"
        if model_config is not None:
            self.config.network = AlphaZeroModel(model_config)
        self.config.network.eval()

    def add_metric(self, name, value):
        if self.config.metrics is None:
            return
        self.config.metrics.add_scalar(name, value, self.config.global_step, self.config.game_idx_in_iteration)

    def log_memory_usage(self, stage_name: str):
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        self.add_metric(f"{stage_name}/Memory/RSS", mem_info.rss / 1024**2)
        self.add_metric(f"{stage_name}/Memory/VMS", mem_info.vms / 1024**2)

    def update_self_play_game_progress(
        self,
        game_id: str,
        loop_number: int,
        game_number: int,
        moves_played: int,
        max_moves: int,
        current_round: str,
        last_action: str,
        game_start_time_unix: float,
        status: str,
    ):
        file = SELF_PLAY_GAMES_STATUS_PATH / f"{game_id}.json"
        status_data = {
            "loop_number": loop_number,
            "game_number": game_number,
            "status": status,
            "moves_played": moves_played,
            "max_moves": max_moves,
            "current_round": current_round,
            "last_action": last_action,
            "start_time_unix": game_start_time_unix,
            "last_update_unix": time.time()
        }
        try:
            with open(file, 'w') as f:
                json.dump(status_data, f, indent=4)
        except IOError as e:
            LOGGER.error(f"Error writing to {SELF_PLAY_GAMES_STATUS_PATH}: {e}")
        except Exception as e: # Catch any other unexpected error during file write
            LOGGER.error(f"Unexpected error writing {SELF_PLAY_GAMES_STATUS_PATH}: {e}", exc_info=True)

    def play(self):
        """Plays out a self-play match, returning a MCTSPlayer object containing:
        - the final position
        - the n x 26535 tensor of floats representing the mcts search probabilities
        - the n x 4 tensor of floats representing the original value-net estimate
        where n is the number of moves in the game
        """
        player = MCTSPlayer(self.config)

        game_start_time = time.time()
        self.update_self_play_game_progress(
            game_id=self.config.game_id,
            loop_number=self.config.global_step,
            game_number=self.config.game_idx_in_iteration,
            moves_played=0,
            max_moves=self.config.max_game_length,
            current_round="N/A",
            last_action="N/A",
            game_start_time_unix=game_start_time,
            status="Starting Up"
        )

        total_tree_search_time_for_game = 0
        total_pick_move_time_for_game = 0
        total_play_move_time_for_game = 0
        total_move_time_for_game = 0
        num_forced_moves_in_game = 0
        num_mcts_moves_in_game = 0
        total_sims_for_mcts_moves = 0

        # Must run this once at the start to expand the root node.
        first_node = player.root.select_leaf()
        with torch.no_grad():
            probs, _, val = self.config.network.run_encoded(first_node.encoded_game_state)
        first_node.incorporate_results(probs, val, first_node)
        del first_node
        move_counter = 0
        game_ended_by_max_length = 0
        try:
            while True:
                LOGGER.info(f"SelfPlay.play loop start, move {move_counter}")
                self.log_memory_usage(stage_name=f"SelfPlay.play")
                start_time_for_move_processing = time.time()
                player.root.inject_noise()

                tree_search_duration_this_move = 0
                if player.root.num_legal_actions == 1:
                    LOGGER.info(f"Move {move_counter}: Only one legal action. Skipping MCTS.")
                    num_forced_moves_in_game += 1
                else:
                    num_mcts_moves_in_game += 1
                    current_readouts = player.root.N
                    target_readouts_for_move = current_readouts + self.config.num_readouts

                    # MCTS simulation loop
                    sim_count_this_move = 0
                    while player.root.N < target_readouts_for_move:
                        player.tree_search()
                        sim_count_this_move += self.config.parallel_readouts
                    total_sims_for_mcts_moves += (player.root.N - current_readouts)

                tree_search_end_time_this_move = time.time() - tree_search_duration_this_move
                total_tree_search_time_for_game += tree_search_end_time_this_move

                pick_move_start_time = time.time()
                move = player.pick_move()
                pick_move_duration_this_move = time.time() - pick_move_start_time
                total_pick_move_time_for_game += pick_move_duration_this_move
                LOGGER.info(f"Selected move: {move}")
                play_move_start_time = time.time()
                player.play_move(move)
                play_move_duration_this_move = time.time() - play_move_start_time
                total_play_move_time_for_game += play_move_duration_this_move
                move_counter += 1
                self.update_self_play_game_progress(
                    game_id=self.config.game_id,
                    loop_number=self.config.global_step,
                    game_number=self.config.game_idx_in_iteration,
                    moves_played=move_counter,
                    max_moves=self.config.max_game_length,
                    current_round=player.root.game_object.round.round_description(),
                    last_action=player.root.game_object.actions[-1].description(),
                    game_start_time_unix=game_start_time,
                    status="In Progress"
                )

                move_time_this_move = time.time() - start_time_for_move_processing
                total_move_time_for_game += move_time_this_move

                self.add_metric("SelfPlay/Tree_Search_Time_ms", tree_search_duration_this_move * 1000)
                self.add_metric("SelfPlay/Pick_Move_Time_ms", pick_move_duration_this_move * 1000)
                self.add_metric("SelfPlay/Play_Move_Time_ms", play_move_duration_this_move * 1000)
                self.add_metric("SelfPlay/Move_Time_ms", move_time_this_move * 1000)
                self.add_metric("SelfPlay/Num_MCTS_Moves", sim_count_this_move)
                self.add_metric("SelfPlay/Total_Sims_For_MCTS_Moves", total_sims_for_mcts_moves)

                if player.root.is_done():
                    if player.root.game_object.move_number == self.config.max_game_length:
                        LOGGER.info(f"Game ended by max length. Ending game.")
                        player.root.game_object.end_game()
                        game_ended_by_max_length = 1

                    player.set_result(player.root.game_result())
                    LOGGER.info(f"Game finished after {move_counter} moves. Result: {player.root.game_object.result()}, mapped to: {player.result} via {player.root.player_mapping}")
                    
                    self.update_self_play_game_progress(
                        game_id=self.config.game_id,
                        loop_number=self.config.global_step,
                        game_number=self.config.game_idx_in_iteration,
                        moves_played=move_counter,
                        max_moves=self.config.max_game_length,
                        current_round="Finished",
                        last_action=player.root.game_object.actions[-1].description(),
                        game_start_time_unix=game_start_time,
                        status="Completed"
                    )
                    break

        except Exception as e:
            LOGGER.error(f"Error in self-play after {move_counter} moves: {e}", exc_info=True)
            LOGGER.error(f"Game actions: {player.root.game_object.raw_actions}")
            # It might be useful to still try and get data from the player if an error occurs mid-game
            # For now, just re-raise or handle as per existing logic.
            self.update_self_play_game_progress(
                game_id=self.config.game_id,
                loop_number=self.config.global_step,
                game_number=self.config.game_idx_in_iteration,
                moves_played=move_counter,
                max_moves=self.config.max_game_length,
                current_round=player.root.game_object.round.round_description(),
                last_action=player.root.game_object.actions[-1].description(),
                game_start_time_unix=game_start_time,
                status="Error"
            )

        self.add_metric("SelfPlay/Game_Length_Moves", move_counter)
        self.add_metric("SelfPlay/Game_Total_Time_Seconds", total_move_time_for_game)
        self.add_metric("SelfPlay/Game_Num_Forced_Moves", num_forced_moves_in_game)
        self.add_metric("SelfPlay/Game_Ended_By_Max_Length", game_ended_by_max_length)

        if player.result is not None and len(player.result) > 0:
            self.add_metric("SelfPlay/Game_Result_Player0", player.result[0])
            self.add_metric("SelfPlay/Game_Result_Player1", player.result[1])
            self.add_metric("SelfPlay/Game_Result_Player2", player.result[2])
            self.add_metric("SelfPlay/Game_Result_Player3", player.result[3])

        if num_mcts_moves_in_game > 0:
            avg_sims_per_mcts_move = total_sims_for_mcts_moves / num_mcts_moves_in_game
            avg_tree_search_time_per_mcts_move_ms = (total_tree_search_time_for_game / num_mcts_moves_in_game) * 1000
            self.add_metric("SelfPlay/Avg_Sims_Per_MCTS_Move", avg_sims_per_mcts_move)
            self.add_metric("SelfPlay/Avg_Tree_Search_Time_Per_MCTS_Move_ms", avg_tree_search_time_per_mcts_move_ms)
        
        avg_pick_move_time_ms = (total_pick_move_time_for_game / move_counter if move_counter > 0 else 0) * 1000
        avg_play_move_time_ms = (total_play_move_time_for_game / move_counter if move_counter > 0 else 0) * 1000
        self.add_metric("SelfPlay/Avg_Pick_Move_Time_ms", avg_pick_move_time_ms)
        self.add_metric("SelfPlay/Avg_Play_Move_Time_ms", avg_play_move_time_ms)

        return player

    def run_game(self):
        """Takes a played game and record results and game data."""
        if self.config.selfplay_dir is not None:
            os.makedirs(self.config.selfplay_dir, exist_ok=True)
            os.makedirs(self.config.holdout_dir, exist_ok=True)

        player = self.play()

        LOGGER.info(f"Player result: {player.result}")
        LOGGER.info(f"Game actions: {player.root.game_object.raw_actions}")
        if player.result is None or np.all(np.array(player.result) == 0.0):
            LOGGER.warning(
                f"Game {self.config.game_id} finished with no conclusive result or result not set. Skipping data extraction."
            )
            self.add_metric("SelfPlay/Games_Skipped_No_Result", 1)
            return

        game_data = player.extract_data()

        if self.config.selfplay_dir is not None:
            # Hold out 5% of games for validation.
            if random.random() < self.config.holdout_pct:
                save_path = self.config.holdout_dir / self.config.network.get_name()
            else:
                save_path = self.config.selfplay_dir / self.config.network.get_name()

            processor = TrainingExampleProcessor()
            processor.write_lmdb(game_data, save_path)

        # Explicitly delete large objects and collect garbage
        del player
        del game_data
        gc.collect()
        LOGGER.info("Explicitly deleted player, game_data, game_examples and ran gc.collect()")


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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    setup_logging(logging.INFO, f"logs/self_play/self_play_{timestamp}.log")

    model = get_latest_model("model_checkpoints")
    config = SelfPlayConfig(network=model)
    selfplay = SelfPlay(config)

    num_games_to_run = getattr(config, "num_games_to_run", 1)
    for i in range(num_games_to_run):
        LOGGER.info(f"--- Starting game {i+1}/{num_games_to_run} ---")
        selfplay.run_game()
        LOGGER.info(f"GC counts after game {i+1}: {gc.get_count()}")


if __name__ == "__main__":
    main()
