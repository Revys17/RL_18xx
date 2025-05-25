# rl18xx/agent/alphazero/self_play.py

import logging
import time
import numpy as np
import torch
from datetime import datetime
from typing import List, Tuple, Dict
import pickle
from rl18xx.game.gamemap import GameMap
from rl18xx.agent.alphazero.model import Model
from rl18xx.agent.alphazero.encoder import Encoder_1830
from rl18xx.agent.alphazero.action_mapper import ActionMapper
from rl18xx.agent.alphazero.mcts import MCTS

LOGGER = logging.getLogger(__name__)

# Define the structure for a training example
# ( (game_state_tensor, map_nodes_tensor, raw_edge_input_tensor), mcts_policy_vector, final_value_vector )
TrainingExample = Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], np.ndarray, np.ndarray]


def run_self_play_game(
    game_class: type,  # e.g., the specific game class like Game_1830
    game_options: Dict,  # Options to initialize the game (e.g., player names)
    model: Model,
    encoder: Encoder_1830,
    action_mapper: ActionMapper,
    mcts_simulations: int,
    initial_temperature: float = 1.0,
    temperature_decay_steps: int = 10,  # Steps after which temperature becomes 0
    c_puct: float = 1.0,
    dirichlet_alpha: float = 0.03,
    dirichlet_noise_factor: float = 0.25,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),  # Device for model inference
) -> List[TrainingExample]:
    """
    Runs a single game of self-play using MCTS and the provided model.

    Args:
        game_class: The class of the game to play (e.g., from game_map.game_by_title).
        game_options: Dictionary of player IDs to names for game initialization.
        model: The neural network model.
        encoder: The game state encoder.
        action_mapper: The action mapper.
        mcts_simulations: The number of simulations to run per MCTS search.
        initial_temperature: Initial temperature for action selection sampling.
        temperature_decay_steps: Number of game steps before temperature drops to 0 (greedy selection).
        c_puct: PUCT constant for MCTS.
        dirichlet_alpha: Alpha parameter for Dirichlet noise.
        dirichlet_noise_factor: Weight of Dirichlet noise in root priors.
        device: The torch device ('cpu' or 'cuda') for model calculations.

    Returns:
        A list of training examples (state_encoding, mcts_policy, final_value)
        generated during the game. Returns empty list if game fails prematurely.
    """
    start_time = time.perf_counter()
    LOGGER.info(f"Starting new self-play game with {mcts_simulations} simulations per move.")
    model.to(device)
    model.eval()

    try:
        game = game_class(game_options)
        num_players = len(game.players)
        LOGGER.info(f"Game initialized: {game.id}, Players: {[p.id for p in game.players]}")
    except Exception as e:
        LOGGER.exception("Failed to initialize game.")
        return []

    game_history: List[
        Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], np.ndarray]
    ] = []  # Stores (encoded_state_tuple, mcts_policy)
    game_steps = 0

    while not game.finished:
        step_start_time = time.perf_counter()
        current_player_id = game.active_players()[0]  # Assumes MCTS runs for the primary active player
        LOGGER.debug(f"--- Game Step {game_steps}, Player {current_player_id}'s turn ---")
        if game_steps % 100 == 0:
            LOGGER.debug(f"Game actions: {game.raw_actions}")

        # --- Run MCTS ---
        try:
            # Create a fresh MCTS instance for the current state
            mcts = MCTS(
                root_state=game,  # Pass the current game state
                model=model,
                encoder=encoder,
                action_mapper=action_mapper,
                num_simulations=mcts_simulations,
                c_puct=c_puct,
                alpha=dirichlet_alpha,
                noise_factor=dirichlet_noise_factor,
                device=device,
            )

            # Run the search
            mcts.search()

            # Determine temperature for action selection
            temperature_for_action_selection = initial_temperature if game_steps < temperature_decay_steps else 0.0
            LOGGER.debug(f"Using temperature {temperature_for_action_selection} for action selection.")

            # Get the MCTS policy (target for the policy head) - ALWAYS use temp=1 for this
            _, policy_vector_target = mcts.get_policy(temperature=1.0)
            # Get the policy dict for action selection using the potentially decayed temperature
            policy_dict_for_action, _ = mcts.get_policy(temperature=temperature_for_action_selection)

            # Store state and policy *before* making the move
            # The encoder returns a tuple of tensors.
            current_game_state_tensor, current_map_nodes_tensor, current_raw_edge_tensor = encoder.encode(game)

            # Store on CPU
            encoded_state_tuple_cpu = (
                current_game_state_tensor.cpu(),
                current_map_nodes_tensor.cpu(),
                current_raw_edge_tensor.cpu(),
            )
            game_history.append((encoded_state_tuple_cpu, policy_vector_target))
            LOGGER.debug(
                f"Stored state encoding shapes: game_state={encoded_state_tuple_cpu[0].shape}, map_nodes={encoded_state_tuple_cpu[1].shape}, raw_edge={encoded_state_tuple_cpu[2].shape}"
            )
            # LOGGER.debug(f"Policy vector target shape: {policy_vector_target.shape}")

            # --- Select and Play Action ---
            if temperature_for_action_selection == 0:
                # Greedy selection: choose action with max visits (break ties randomly)
                # Use policy_dict_for_action which was derived with temp=0
                action_index = max(policy_dict_for_action, key=policy_dict_for_action.get)
                LOGGER.debug(
                    f"Greedy action selection: Chose index {action_index} (Prob: {policy_dict_for_action[action_index]:.4f})"
                )
            else:
                # Sample action based on policy probabilities from policy_dict_for_action
                action_indices = list(policy_dict_for_action.keys())
                probabilities = list(policy_dict_for_action.values())
                # Ensure probabilities sum to 1 (might be slightly off due to floats)
                probabilities = np.array(probabilities, dtype=np.float64)
                probabilities /= np.sum(probabilities)
                action_index = np.random.choice(action_indices, p=probabilities)
                LOGGER.debug(
                    f"Sampled action selection: Chose index {action_index} (Prob: {policy_dict_for_action[action_index]:.4f})"
                )

            # Map index to game action object
            action = action_mapper.map_index_to_action(action_index, game)
            LOGGER.info(f"Step {game_steps}: Player {current_player_id} plays action index {action_index} -> {action}")

            # Apply action to the game state
            game.process_action(action)
            game_steps += 1

        except Exception as e:
            LOGGER.exception(f"Error during game step {game_steps}. Aborting self-play game.")
            # Potentially save partial game data or debug info here
            return []  # Return empty list indicating failure
        except KeyboardInterrupt as e:
            LOGGER.info("Self-play interrupted by user.")
            LOGGER.info(f"Game log: {game.log}")
            LOGGER.info(f"Game actions: {game.raw_actions}")
            exit()

        step_end_time = time.perf_counter()
        LOGGER.debug(f"Game step {game_steps-1} finished in {(step_end_time - step_start_time)*1000:.1f} ms.")

    # --- Game Finished ---
    game_end_time = time.perf_counter()
    LOGGER.info(f"Game finished after {game_steps} steps in {(game_end_time - start_time):.2f} seconds.")
    LOGGER.info(f"Full game action sequence: {game.raw_actions}")

    # Determine final scores and outcome vector
    try:
        sorted_players = sorted(game.players, key=lambda p: p.id)
        player_id_to_idx = {p.id: i for i, p in enumerate(sorted_players)}
        final_scores = game.result()
        if not final_scores:  # Handle empty scores if game ends abnormally
            raise ValueError("Game ended with no scores available. Cannot calculate value vector.")
        LOGGER.info(f"Final scores: {final_scores}")

        max_score = max(final_scores.values()) if final_scores else -float("inf")
        outcome_vector = np.zeros(num_players, dtype=np.float32)
        for player_id, score in final_scores.items():
            player_idx = player_id_to_idx.get(player_id)
            if player_idx is None:
                raise ValueError(f"Player ID {player_id} from final scores not in player map.")
            outcome_vector[player_idx] = 1.0 if score == max_score else -1.0

        LOGGER.info(f"Calculated final value vector: {outcome_vector}")

    except Exception as e:
        LOGGER.exception("Error calculating final scores or value vector.")
        return []

    # --- Assemble Training Data ---
    training_data: List[TrainingExample] = []
    for state_encoding, mcts_policy in game_history:
        training_data.append((state_encoding, mcts_policy, outcome_vector))

    LOGGER.info(f"Generated {len(training_data)} training examples from the game.")
    return training_data


if __name__ == "__main__":
    # --- Setup ---

    # --- Game Setup ---
    game_title = "1830"  # Example
    game_map = GameMap()
    game_class_from_map = game_map.game_by_title(game_title)
    if not game_class_from_map:
        raise ValueError(f"Game class for '{game_title}' not found.")

    player_options = {"1": "Alice", "2": "Bob", "3": "Charlie", "4": "Dave"}
    temp_game_for_config = game_class_from_map(player_options)
    num_players_for_config = len(temp_game_for_config.players)

    # --- Encoder and ActionMapper ---
    encoder = Encoder_1830()
    action_mapper = ActionMapper()

    # --- Model Setup ---
    dummy_game_state_encoded, dummy_map_nodes_encoded, dummy_raw_edges_encoded = encoder.encode(temp_game_for_config)

    # game_state_tensor has shape (batch, features)
    game_state_size = dummy_game_state_encoded.shape[1]

    # map_nodes_tensor has shape (num_nodes, features)
    map_node_features = dummy_map_nodes_encoded.shape[1]

    policy_output_size = action_mapper.action_encoding_size
    value_output_size = num_players_for_config

    model = Model(
        game_state_size=game_state_size,
        map_node_features=map_node_features,
        policy_size=policy_output_size,
        value_size=value_output_size,
        # Add other necessary model parameters here if they are not defaults:
        # mlp_hidden_dim=...,
        # gnn_node_proj_dim=...,
        # ... etc.
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # --- Run ---
    training_data = run_self_play_game(
        game_class=game_class_from_map,
        game_options=player_options,
        model=model,
        encoder=encoder,
        action_mapper=action_mapper,
        mcts_simulations=10,
        initial_temperature=1.0,
        temperature_decay_steps=5,
        device=device,
    )

    if not training_data:
        raise ValueError("No training examples generated.")

    LOGGER.info(f"\n--- Example Training Data ---")
    state_ex, policy_ex, value_ex = training_data[0]
    LOGGER.info(f"State Encoding Shape: {state_ex[0].shape}, {state_ex[1].shape}, {state_ex[2].shape}")
    LOGGER.info(f"Policy Vector Shape: {policy_ex.shape}, Sum: {np.sum(policy_ex):.4f}")
    LOGGER.info(f"Value Vector: {value_ex}")
    # Save to file
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"training_data/{game_title}_training_data_{date_str}.pkl", "wb") as f:
        pickle.dump(training_data, f)
