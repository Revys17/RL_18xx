import math
import numpy as np
from typing import Dict, Optional, Tuple
from rl18xx.game.engine.game import BaseGame
from rl18xx.agent.alphazero.model import Model
from rl18xx.agent.alphazero.encoder import Encoder_1830
from rl18xx.agent.alphazero.action_mapper import ActionMapper
import torch
import logging
import time

LOGGER = logging.getLogger(__name__)


class Node:
    """
    Represents a node in the Monte Carlo Tree Search.

    Stores statistics needed for the PUCT algorithm and tree traversal.
    """

    def __init__(
        self,
        parent: Optional["Node"],
        prior_prob: float,
        state: BaseGame,
        num_players: int,
    ):
        """
        Initializes a Node.

        Args:
            parent: The parent node. None for the root node.
            prior_prob: The prior probability P(s, a) of reaching this node
                        (representing state 's' via the action 'a' from the parent).
                        For the root node, this is often set to 1.0 or 0.0 initially.
            state: The game state represented by this node. Kept for potential
                   debugging or state-specific logic, though not strictly
                   required if transitions are handled externally.
            num_players: The number of players in the game.
        """
        self.parent = parent
        self.state = state  # Store the state for expansion/evaluation
        self.children: Dict[int, "Node"] = {}  # Maps action index to child Node
        self.num_players = num_players  # Store num_players

        # --- MCTS Statistics (for the edge leading to this node) ---
        # N(s_parent, a): How many times action 'a' was taken from parent state s_parent
        self.visit_count: int = 0
        self.total_action_value: np.ndarray = np.zeros(num_players, dtype=np.float32)
        self.average_action_value: np.ndarray = np.zeros(num_players, dtype=np.float32)
        self.prior_probability: float = prior_prob
        # Simple unique ID for logging if needed
        self.id = id(self)
        LOGGER.debug(f"Node {self.id} created. Parent: {id(parent) if parent else 'None'}, Prior: {prior_prob:.4f}")

    def is_leaf(self) -> bool:
        """Checks if the node is a leaf node (has no children)."""
        return len(self.children) == 0

    def get_total_parent_visits(self) -> int:
        """Calculates N(s) for the parent state s by summing visits of sibling edges."""
        # Note: This is N(s_parent), not N(s) used in PUCT numerator for child selection
        if self.parent:
            # Sum visit_count of all children of the parent
            return sum(child.visit_count for child in self.parent.children.values())
        return self.visit_count  # Root node's parent visits is just its own visits

    def update_stats(self, value_vector: np.ndarray):
        """Updates stats using the outcome vector for all players."""
        if value_vector.shape[0] != self.num_players:
            LOGGER.error(
                "Value vector length %d != num_players %d during update.",
                value_vector.shape[0],
                self.num_players,
            )
            raise ValueError(f"Value vector length {value_vector.shape[0]} != num_players {self.num_players}")
        self.visit_count += 1
        self.total_action_value += value_vector
        # Q(s_parent, a) = W(s_parent, a) / N(s_parent, a)
        self.average_action_value = self.total_action_value / self.visit_count
        # LOGGER.debug(f"Node {self.id} updated. Visits: {self.visit_count}, Avg Value: {self.average_action_value}")

    def __repr__(self):
        # Represent average value for the current player at this node's state
        # Need a way to get player index here if we want state-specific Q display
        q_repr = np.array2string(self.average_action_value, precision=3, floatmode="fixed")
        parent_id = id(self.parent) if self.parent else "None"
        return (
            f"Node(id={id(self)}, parent={parent_id}, N={self.visit_count}, Q={q_repr}, "
            f"P={self.prior_probability:.3f}, children={len(self.children)})"
        )


class MCTS:
    def __init__(
        self,
        root_state: BaseGame,
        model: Model,
        encoder: Encoder_1830,
        action_mapper: ActionMapper,
        num_simulations: int,
        c_puct: float = 1.0,
        alpha: float = 0.03,
        noise_factor: float = 0.25,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        LOGGER.info("Initializing MCTS.")
        LOGGER.info(f"  Num Simulations: {num_simulations}")
        LOGGER.info(f"  c_puct: {c_puct}")
        LOGGER.info(f"  Dirichlet Alpha: {alpha}, Noise Factor: {noise_factor}")
        LOGGER.info(f"  Device: {device}")

        self.players = root_state.players
        self.num_players = len(root_state.players)
        # Sort players by ID for consistent indexing
        sorted_players = sorted(self.players, key=lambda p: p.id)
        self.player_id_to_idx = {p.id: i for i, p in enumerate(sorted_players)}
        self.player_idx_to_id = {i: p.id for i, p in enumerate(sorted_players)}
        LOGGER.info(f"  Num Players: {self.num_players}, Player Map: {self.player_id_to_idx}")

        self.root = Node(parent=None, prior_prob=0.0, state=root_state, num_players=self.num_players)
        LOGGER.info(f"  Root node created: {self.root}")

        self.model = model
        self.encoder = encoder
        self.action_mapper = action_mapper
        self.num_simulations = num_simulations
        self.device = device

        self.c_puct = c_puct
        self.alpha = alpha
        self.noise_factor = noise_factor
        self.model.eval()  # Ensure model is in evaluation mode
        LOGGER.debug("MCTS Initialization complete.")

    def get_player_index(self, state: BaseGame) -> int:
        """Gets the integer index [0..N-1] for the current player in the state."""
        player_id = state.active_players()[0].id
        LOGGER.debug(f"Looking up index for player ID: {player_id}")
        if player_id not in self.player_id_to_idx:
            LOGGER.error(f"Player ID '{player_id}' not found in player map {self.player_id_to_idx}")
            # This can happen if current_entity is not a player (e.g., a Corporation)
            # MCTS should typically only run when it's a player's turn.
            # Or we need a convention for non-player entities.
            # For now, raise error if not found.
            raise ValueError(f"Player ID '{player_id}' not found in player map {self.player_id_to_idx}")
        idx = self.player_id_to_idx[player_id]
        LOGGER.debug(f"Found index: {idx}")
        return idx

    def search(self) -> Node:
        """Performs the MCTS search for a configured number of simulations."""
        search_start_time = time.perf_counter()
        LOGGER.info(f"Starting MCTS search for {self.num_simulations} simulations.")
        for i in range(self.num_simulations):
            sim_start_time = time.perf_counter()
            LOGGER.info(f"--- Simulation {i+1}/{self.num_simulations} ---")

            leaf = self.select_leaf()
            LOGGER.debug(f"Selected leaf: {leaf}")

            value = self.expand_and_evaluate(leaf)
            LOGGER.debug(f"Evaluated value vector: {value}")

            self.backpropagate(leaf, value)
            LOGGER.debug(f"Backpropagation complete from leaf {id(leaf)}")

            sim_end_time = time.perf_counter()
            LOGGER.debug(f"--- Simulation {i+1} finished in {(sim_end_time - sim_start_time)*1000:.3f} ms ---")

        search_end_time = time.perf_counter()
        total_duration_s = search_end_time - search_start_time
        sims_per_sec = self.num_simulations / total_duration_s if total_duration_s > 0 else float("inf")
        LOGGER.info(f"MCTS search finished. Total time: {total_duration_s:.3f} s ({sims_per_sec:.2f} sims/sec)")
        LOGGER.info(f"Root node after search: {self.root}")
        return self.root

    def select_leaf(self) -> Node:
        """Selects a leaf node using the PUCT criteria."""
        LOGGER.debug("Starting leaf selection.")
        current_node = self.root
        path = [id(current_node)]
        while not current_node.is_leaf():
            current_node = self.select_child(current_node)
            path.append(id(current_node))
        LOGGER.debug(f"Leaf selection path (node IDs): {path}")
        LOGGER.debug(f"Selected leaf node: {id(current_node)}")
        return current_node

    def select_child(self, node: Node) -> Node:
        """Selects the child with the highest PUCT score."""
        select_start_time = time.perf_counter()
        LOGGER.debug(f"Selecting child from node: {id(node)}")
        best_score = float("-inf")
        best_child = None

        # Apply Dirichlet noise to root priors if applicable
        apply_noise = node == self.root and self.alpha > 0 and self.noise_factor > 0
        if apply_noise:
            LOGGER.debug("Applying Dirichlet noise to root node priors.")
            num_children = len(node.children)
            if num_children == 0:
                LOGGER.warning("Root node has no children, cannot apply Dirichlet noise.")
                noise = None
            else:
                noise = np.random.dirichlet([self.alpha] * num_children)
                LOGGER.debug(f"Generated noise vector (first 5): {noise[:5]}")
        else:
            noise = None  # Ensure noise is not used if not applicable

        # Get children items once to ensure consistent order for noise indexing
        child_items = list(node.children.items())
        # action_indices = [idx for idx, _ in child_items] # Keep track if needed

        parent_player_idx = self.get_player_index(node.state)
        LOGGER.debug(f"Parent node ({id(node)}) player index: {parent_player_idx}")

        parent_total_visits = node.visit_count  # N(s) for the current node
        sqrt_parent_visits = math.sqrt(parent_total_visits) if parent_total_visits > 0 else 1.0  # Avoid sqrt(0)
        LOGGER.debug(f"Parent node visit count N(s): {parent_total_visits}, Sqrt(N(s)): {sqrt_parent_visits:.3f}")

        for i, (action_idx, child) in enumerate(child_items):  # Iterate with index
            # Q value (exploitation term)
            # Get the value from the perspective of the player whose turn it is AT THE PARENT node
            q_value = child.average_action_value[parent_player_idx]

            # U value (exploration bonus)
            prior = child.prior_probability
            original_prior = prior  # For logging

            if apply_noise and noise is not None:
                # Use the loop index 'i' which corresponds to the noise vector index
                noisy_prior = (1 - self.noise_factor) * prior + self.noise_factor * noise[i]
                LOGGER.debug(
                    f"  Action {action_idx}: Noise applied. Original P={prior:.3f}, Noise={noise[i]:.3f}, New P={noisy_prior:.3f}"
                )
                prior = noisy_prior

            # Ensure visit_count is at least 1 for the denominator
            u_value = self.c_puct * prior * (sqrt_parent_visits / (1 + child.visit_count))
            puct_score = q_value + u_value

            LOGGER.debug(
                f"  Child Action {action_idx} (Node {id(child)}): "
                f"N={child.visit_count}, Q={q_value:.3f}, "
                f"P={original_prior:.3f}{' (Noisy P='+str(round(prior,3))+')' if apply_noise and noise is not None else ''}, "
                f"U={u_value:.3f} => PUCT={puct_score:.3f}"
            )

            if puct_score > best_score:
                best_score = puct_score
                best_child = child
                LOGGER.debug(f"    New best child: Action {action_idx} (Node {id(child)}), Score: {best_score:.3f}")

        if best_child is None:
            # This could happen if node.children is empty, but select_leaf should prevent that.
            # Or if all scores are -inf (e.g., Q values are NaN or -inf).
            LOGGER.error(
                f"select_child failed to find a best child for node {id(node)} with {len(child_items)} children."
            )
            raise RuntimeError(f"select_child failed to find a best child for node with {len(child_items)} children.")

        select_end_time = time.perf_counter()
        LOGGER.debug(
            f"Selected child: {id(best_child)} (Action Index: {best_child.parent.children.get(id(best_child), 'N/A')}) "  # Find action idx back - slightly inefficient
            f"with score {best_score:.3f}. Selection took {(select_end_time - select_start_time)*1000:.3f} ms."
        )

        return best_child

    def expand_and_evaluate(self, leaf: Node) -> np.ndarray:  # Returns value vector
        """Expands leaf node, evaluates using network, returns value vector."""
        expand_start_time = time.perf_counter()
        LOGGER.debug(f"Expanding and evaluating leaf node: {id(leaf)}")

        # --- 1. Check for Terminal State ---
        if leaf.state.finished:
            LOGGER.info(f"Leaf node {id(leaf)} is a terminal state.")
            final_scores = leaf.state.result()
            LOGGER.debug(f"Final scores: {final_scores}")
            max_score = max(final_scores.values()) if final_scores else -float("inf")
            outcome_vector = np.full(self.num_players, 0.0, dtype=np.float32)
            winners = 0
            losers = 0
            for player_id, score in final_scores.items():
                player_idx = self.player_id_to_idx.get(player_id)
                if player_idx is not None:
                    if score == max_score:
                        outcome_vector[player_idx] = 1.0
                        winners += 1
                    else:
                        outcome_vector[player_idx] = -1.0
                        losers += 1
                else:
                    LOGGER.warning(f"Player ID {player_id} from final scores not in player map.")
            # Optional: Normalize scores (e.g., 1/W for winners, -1/L for losers)
            # if winners > 0 and losers > 0:
            #     outcome_vector[outcome_vector == 1.0] = 1.0 / winners
            #     outcome_vector[outcome_vector == -1.0] = -1.0 / losers
            # elif winners > 0: # All winners (draw)
            #     outcome_vector.fill(1.0 / winners)

            LOGGER.info(f"Terminal outcome vector: {outcome_vector}")
            expand_end_time = time.perf_counter()
            LOGGER.debug(f"Terminal evaluation took {(expand_end_time - expand_start_time)*1000:.3f} ms.")
            return outcome_vector

        # --- 2. Network Evaluation ---
        LOGGER.debug("Encoding state for network evaluation.")
        try:
            # Assuming encoder returns tensor with batch dim [1, ...] on CPU
            game_state, (map_nodes, raw_edge_input_tensor) = self.encoder.encode(leaf.state)
            # Move the encoded state to the correct device
            game_state = game_state.to(self.device)
            map_nodes = map_nodes.to(self.device)
            raw_edge_input_tensor = raw_edge_input_tensor.to(self.device)
            LOGGER.debug(f"Moved encoded state to device: {game_state.device}")
        except Exception as e:
            LOGGER.exception(f"Error during state encoding or moving for node {id(leaf)}.")
            raise e

        LOGGER.debug(f"Encoded state shape: {game_state.shape}")
        network_start_time = time.perf_counter()
        with torch.no_grad():
            # Model returns (policy_logits, value_vector) on self.device
            try:
                edge_index = raw_edge_input_tensor[0:2, :]
                edge_attr_categorical = raw_edge_input_tensor[2, :].long()
                policy_logits, value_vector_tensor = self.model(game_state, map_nodes, edge_index, edge_attr_categorical)

                # Move results back to CPU for numpy conversion and masking
                value_vector = value_vector_tensor.squeeze(0).cpu().numpy()
                policy_vector_tensor = torch.softmax(policy_logits, dim=1).squeeze(0).cpu()
            except Exception as e:
                LOGGER.exception(f"Error during model forward pass for node {id(leaf)}.")
                raise e
        network_end_time = time.perf_counter()
        LOGGER.debug(f"Network evaluation took {(network_end_time - network_start_time)*1000:.3f} ms.")
        LOGGER.debug(f"Raw value vector output (CPU): {value_vector}")
        LOGGER.debug(f"Raw policy vector shape (CPU): {policy_vector_tensor.shape}")

        # --- 3. Expand Children ---
        LOGGER.debug("Expanding children based on policy.")
        try:
            # Mask is created on CPU
            legal_mask_tensor = self.action_mapper.get_legal_action_mask(leaf.state)
        except Exception as e:
            LOGGER.info(f"State actions: {leaf.state.raw_actions}")
            LOGGER.exception(f"Error getting legal action mask for node {id(leaf)}.")
            raise e

        # Both policy_vector_tensor and legal_mask_tensor are now on CPU
        masked_policy = policy_vector_tensor * legal_mask_tensor
        policy_sum = torch.sum(masked_policy)
        LOGGER.debug(f"Policy sum after masking (CPU): {policy_sum.item():.4f}")

        if policy_sum > 1e-6:
            legal_policy = masked_policy / policy_sum
        else:
            # No legal moves found by mask, but state not finished? Problem!
            LOGGER.error(f"No legal moves found by mask for non-terminal node {id(leaf)}. Policy sum ~0.")
            # Log relevant state info if possible
            # LOGGER.error(f"State details: {leaf.state}") # Be careful, state repr might be huge
            raise ValueError("No legal moves found by mask for non-terminal state")

        # Expand children for all legal actions
        legal_policy_np = legal_policy.numpy()
        action_indices = np.where(legal_policy_np > 1e-8)[0]  # Use small epsilon
        LOGGER.debug(f"Found {len(action_indices)} legal actions with non-zero policy to expand.")

        num_expanded = 0
        for action_index in action_indices:
            prior_p = legal_policy_np[action_index]
            try:
                LOGGER.debug(f"Mapping index {action_index} to action")
                next_state = leaf.state.clone(leaf.state.raw_actions)
                action = self.action_mapper.map_index_to_action(action_index, next_state)
                LOGGER.debug(f"Mapping index {action_index} to action {action}")
                next_state.process_action(action)
                new_child = Node(
                    parent=leaf,
                    prior_prob=prior_p,
                    state=next_state,
                    num_players=self.num_players,
                )
                leaf.children[action_index] = new_child
                num_expanded += 1
                LOGGER.debug(
                    f"  Expanded child for action index {action_index} (Action: {action}), Prior: {prior_p:.4f}, New Node: {id(new_child)}"
                )
            except Exception as e:
                LOGGER.exception(f"Error expanding child for action index {action_index} from node {id(leaf)}.")
                LOGGER.info(f"Raw actions: {leaf.state.raw_actions}, attempted action: {action_index}")
                raise e

        LOGGER.debug(f"Successfully expanded {num_expanded} children for node {id(leaf)}.")
        expand_end_time = time.perf_counter()
        LOGGER.debug(f"Expansion and evaluation finished in {(expand_end_time - expand_start_time)*1000:.3f} ms.")
        return value_vector

    def backpropagate(self, leaf: Node, value_vector: np.ndarray):  # Takes vector
        """Backpropagates the value vector up the tree."""
        LOGGER.debug(f"Starting backpropagation from leaf {id(leaf)} with value {value_vector}")
        current_node = leaf
        path_len = 0
        while current_node:
            LOGGER.debug(f"  Updating node {id(current_node)}")
            current_node.update_stats(value_vector)
            current_node = current_node.parent
            path_len += 1
        LOGGER.debug(f"Backpropagation complete. Updated {path_len} nodes.")

    def get_policy(self, temperature: float = 1.0) -> Tuple[Dict[int, float], np.ndarray]:
        """Calculates the final policy distribution based on root visit counts."""
        policy_start_time = time.perf_counter()
        LOGGER.debug(f"Calculating final policy from root node {id(self.root)} with temperature {temperature}.")
        if not self.root.children:
            LOGGER.error("Root node has no children after search, cannot calculate policy.")
            raise ValueError("Root node has no children after search")

        # Get visit counts and corresponding action indices
        child_items = list(self.root.children.items())
        action_indices = [idx for idx, child in child_items]
        visit_counts = np.array([child.visit_count for idx, child in child_items], dtype=np.float32)
        LOGGER.debug(f"Root children visit counts: {dict(zip(action_indices, visit_counts))}")

        if np.sum(visit_counts) == 0:
            LOGGER.warning("All root children have zero visits. Falling back to prior probabilities.")
            # Fallback: Uniform? Or based on priors? Let's use priors.
            priors = np.array(
                [child.prior_probability for idx, child in child_items],
                dtype=np.float32,
            )
            LOGGER.debug(f"Root children priors: {dict(zip(action_indices, priors))}")
            if np.sum(priors) > 1e-6:
                probs = priors / np.sum(priors)
                LOGGER.debug("Using normalized priors for policy.")
            else:  # If priors also sum to zero, uniform
                LOGGER.warning("Root children priors also sum to zero. Falling back to uniform distribution.")
                probs = np.ones_like(visit_counts) / len(visit_counts) if len(visit_counts) > 0 else np.array([])
                LOGGER.debug("Using uniform distribution for policy.")
        elif temperature == 0:  # Choose greedily based on visits
            LOGGER.debug("Temperature is 0, choosing greedily based on max visit count.")
            probs = np.zeros_like(visit_counts)
            max_visit_indices = np.where(visit_counts == np.max(visit_counts))[0]
            LOGGER.debug(f"Max visit count indices: {max_visit_indices}")
            # Break ties randomly if multiple max visits
            chosen_index_in_list = np.random.choice(max_visit_indices)
            probs[chosen_index_in_list] = 1.0
            LOGGER.debug(
                f"Chosen index (in list of children): {chosen_index_in_list}, Action Index: {action_indices[chosen_index_in_list]}"
            )
        else:
            # Apply temperature
            LOGGER.debug(f"Applying temperature {temperature} to visit counts.")
            temp_visits = visit_counts ** (1.0 / temperature)
            probs = temp_visits / np.sum(temp_visits)
            LOGGER.debug(f"Probabilities after temperature: {dict(zip(action_indices, probs))}")

        # Create policy dictionary and full vector
        policy_dict = {idx: prob for idx, prob in zip(action_indices, probs)}
        full_policy_vector = np.zeros(self.action_mapper.action_encoding_size, dtype=np.float32)
        for idx, prob in policy_dict.items():
            full_policy_vector[idx] = prob

        policy_end_time = time.perf_counter()
        LOGGER.debug(f"Policy calculation finished in {(policy_end_time - policy_start_time)*1000:.3f} ms.")
        LOGGER.debug(f"Final policy dict: {policy_dict}")
        LOGGER.debug(f"Final policy vector shape: {full_policy_vector.shape}, Sum: {np.sum(full_policy_vector):.4f}")

        return policy_dict, full_policy_vector
