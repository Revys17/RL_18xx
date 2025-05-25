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

    Stores statistics for its children's edges and its own state visits.
    """

    def __init__(
        self,
        parent: Optional["Node"],
        state: BaseGame,
        num_players: int,
        action_encoding_size: int,
        action_index: Optional[int] = None,  # Action that led to this node
    ):
        """
        Initializes a Node.

        Args:
            parent: The parent node. None for the root node.
            state: The game state represented by this node.
            num_players: The number of players in the game.
            action_encoding_size: The total number of possible actions in the game.
            action_index: The index of the action taken from the parent to reach this node.
        """
        self.parent = parent
        self.state = state
        self.num_players = num_players
        self.action_encoding_size = action_encoding_size
        self.action_index: Optional[int] = action_index  # Action that led to this node

        self.children: Dict[int, "Node"] = {}

        # --- Statistics for the state 's' represented by this node ---
        # N(s): How many times this state 's' has been visited during simulations
        self.node_total_visits: int = 0

        # --- Statistics for edges (s,a) leading to children of this node ---
        # P(s,a): Prior probabilities for actions 'a' from this state 's'
        self.child_prior_probabilities: np.ndarray = np.zeros(action_encoding_size, dtype=np.float32)
        # N(s,a): Visit counts for actions 'a' from this state 's'
        self.child_visit_counts: np.ndarray = np.zeros(action_encoding_size, dtype=np.int32)
        # W(s,a): Total action values for actions 'a' from this state 's'
        self.child_total_action_values: np.ndarray = np.zeros((action_encoding_size, num_players), dtype=np.float32)
        # Q(s,a): Average action values for actions 'a' from this state 's'
        self.child_average_action_values: np.ndarray = np.zeros((action_encoding_size, num_players), dtype=np.float32)

        self.id = id(self)

    def is_leaf(self) -> bool:
        """Checks if the node is a leaf node (has no children explored yet)."""
        return len(self.children) == 0

    def expand_child(self, action_idx: int, child_state: BaseGame) -> "Node":
        """
        Creates a new child Node for the given action and state,
        adds it to this node's children, and returns the new child.
        Does NOT assign priors; that's handled by MCTS.expand_and_evaluate.
        """
        if action_idx in self.children:
            LOGGER.warning(f"Node {self.id} trying to expand child for action {action_idx} which already exists.")
            return self.children[action_idx]

        child_node = Node(
            parent=self,
            state=child_state,
            num_players=self.num_players,
            action_encoding_size=self.action_encoding_size,
            action_index=action_idx,
        )
        self.children[action_idx] = child_node
        # LOGGER.debug(f"Node {self.id} expanded child for action {action_idx}, new child node {child_node.id}")
        return child_node

    def __repr__(self):
        parent_id_repr = id(self.parent) if self.parent else "None"
        q_edge_repr = "N/A"
        p_edge_repr = "N/A"
        n_edge_repr = "N/A"

        if self.parent and self.action_index is not None:
            if 0 <= self.action_index < self.parent.action_encoding_size:
                q_val = self.parent.child_average_action_values[self.action_index]
                q_edge_repr = np.array2string(q_val, precision=3, floatmode="fixed", suppress_small=True)
                p_val = self.parent.child_prior_probabilities[self.action_index]
                p_edge_repr = f"{p_val:.3f}"
                n_val = self.parent.child_visit_counts[self.action_index]
                n_edge_repr = str(n_val)
            else:
                q_edge_repr = "InvalidActionIdx"
                p_edge_repr = "InvalidActionIdx"
                n_edge_repr = "InvalidActionIdx"


        return (
            f"Node(id={self.id}, parent={parent_id_repr}, act_idx={self.action_index}, "
            f"N_edge={n_edge_repr}, Q_edge={q_edge_repr}, P_edge={p_edge_repr}, "
            f"N_state={self.node_total_visits}, children={len(self.children)})"
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
        apply_virtual_loss: bool = False, # Default to False, enable explicitly
        virtual_loss_c: float = 1.0,     # Value of the virtual loss
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):

        self.players = root_state.players
        self.num_players = len(root_state.players)
        # Sort players by ID for consistent indexing
        sorted_players = sorted(self.players, key=lambda p: p.id)
        self.player_id_to_idx = {p.id: i for i, p in enumerate(sorted_players)}
        self.player_idx_to_id = {i: p.id for i, p in enumerate(sorted_players)}

        self.model = model
        self.encoder = encoder
        self.action_mapper = action_mapper
        self.num_simulations = num_simulations
        self.device = device
        self.action_encoding_size = self.action_mapper.action_encoding_size


        self.root = Node(
            parent=None,
            state=root_state,
            num_players=self.num_players,
            action_encoding_size=self.action_encoding_size,
            action_index=None, # Root has no action leading to it
        )


        self.c_puct = c_puct
        self.alpha = alpha
        self.noise_factor = noise_factor
        self.apply_virtual_loss = apply_virtual_loss
        self.virtual_loss_c = virtual_loss_c
        self.model.eval()

    def get_player_index(self, state: BaseGame) -> int:
        """Gets the integer index [0..N-1] for the current player in the state."""
        player_id = state.active_players()[0].id
        if player_id not in self.player_id_to_idx:
            raise ValueError(f"Player ID '{player_id}' not found in player map {self.player_id_to_idx}")
        idx = self.player_id_to_idx[player_id]
        return idx

    def search(self) -> Node:
        """Performs the MCTS search for a configured number of simulations."""
        search_start_time = time.perf_counter()
        LOGGER.debug(f"Starting MCTS search with {self.num_simulations} simulations.")

        if self.root.is_leaf() and not self.root.state.finished:
            LOGGER.debug(f"Root node {id(self.root)} is a leaf, expanding it first (priors only).")
            # Expand root: gets priors from NN, applies noise if any.
            # Does not backpropagate value, N(s) and N(s,a) for root remain 0.
            _ = self.expand_and_evaluate(self.root)

        for i in range(self.num_simulations):
            sim_start_time = time.perf_counter()
            LOGGER.info(f"--- Simulation {i+1}/{self.num_simulations} ---")

            current_node = self.root
            path = [current_node]

            # 1. Selection
            LOGGER.debug("Starting selection phase...")
            while not current_node.is_leaf():
                if not current_node.children: # Should be caught by is_leaf if children dict is empty
                    LOGGER.warning(f"Node {id(current_node)} has no children dict but not is_leaf(). This is unexpected. Breaking selection.")
                    break 

                best_action_idx, next_node_candidate = self.select_child(current_node)

                if next_node_candidate is None: 
                    LOGGER.error(f"select_child returned None for non-leaf node {id(current_node)}. This is an error.")
                    break 
                
                # Increment visit counts N(s,a) and N(s) for the current edge/node (parent)
                # This is the single increment for N for this simulation pass.
                current_node.child_visit_counts[best_action_idx] += 1
                current_node.node_total_visits += 1 
                LOGGER.debug(f"  Selected action {best_action_idx} from node {id(current_node)}. N_sa now {current_node.child_visit_counts[best_action_idx]}, N_s now {current_node.node_total_visits}")


                if self.apply_virtual_loss:
                    player_idx_at_current = self.get_player_index(current_node.state)
                    LOGGER.debug(f"  Applying virtual loss for action {best_action_idx} from node {id(current_node)} for player {player_idx_at_current}")
                    current_node.child_total_action_values[best_action_idx, player_idx_at_current] -= self.virtual_loss_c
                    
                    # Update average Q(s,a) for this edge, reflecting the virtual loss and new N(s,a)
                    # This Q is used if PUCT is recalculated for this node before backpropagation (e.g. in parallel MCTS)
                    current_node.child_average_action_values[best_action_idx, :] = \
                        current_node.child_total_action_values[best_action_idx, :] / \
                        current_node.child_visit_counts[best_action_idx] # N(s,a) is now initial_N+1
                    LOGGER.debug(f"    Stats for ({id(current_node)}, {best_action_idx}) after VL: N_sa={current_node.child_visit_counts[best_action_idx]}, Q_sa={current_node.child_average_action_values[best_action_idx,:]}")

                current_node = next_node_candidate
                path.append(current_node)
            
            leaf_node = current_node
            LOGGER.debug(f"Selection finished. Leaf node: {id(leaf_node)}")

            # 2. Expansion & Evaluation
            # If leaf is terminal, value is from game. If not, from model.
            # expand_and_evaluate will expand if not terminal and not yet expanded by NN.
            value_vector = self.expand_and_evaluate(leaf_node)
            LOGGER.debug(f"Leaf {id(leaf_node)} evaluated. Value: {value_vector}")

            # 3. Backpropagation
            self.backpropagate(leaf_node, value_vector)
            
            sim_end_time = time.perf_counter()
            LOGGER.debug(f"--- Simulation {i+1} finished in {(sim_end_time - sim_start_time)*1000:.3f} ms ---")

        search_end_time = time.perf_counter()
        total_duration_s = search_end_time - search_start_time
        sims_per_sec = self.num_simulations / total_duration_s if total_duration_s > 0 else float("inf")
        LOGGER.info(f"MCTS search finished. Total time: {total_duration_s:.3f} s ({sims_per_sec:.2f} sims/sec)")
        LOGGER.info(f"Root node after search: {self.root}")
        return self.root

    def select_child(self, node: Node) -> Tuple[int, Node]:
        """Selects the child with the highest PUCT score using vectorized operations."""
        select_start_time = time.perf_counter()
        LOGGER.debug(f"Selecting child from node: {id(node)} with {len(node.children)} existing children.")

        child_action_indices = np.array(list(node.children.keys()), dtype=np.int32)
        if len(child_action_indices) == 0:
            # This should not happen if select_leaf ensures we only call select_child on non-leaf nodes.
            # However, if a node was expanded but all its children had zero policy (e.g. due to masking all out),
            # it might be a leaf in terms of MCTS structure but non-leaf by game termination.
            # The calling code (select_leaf) should handle this. If we reach here, it's an issue.
            raise RuntimeError(f"select_child called on node {id(node)} which has no children in its children dictionary.")

        parent_player_idx = self.get_player_index(node.state)
        LOGGER.debug(f"Parent node ({id(node)}) player index: {parent_player_idx}, N_state(s): {node.node_total_visits}")

        # N(s) for the current node (parent of the children being considered)
        N_s = node.node_total_visits
        sqrt_N_s = math.sqrt(N_s) if N_s > 0 else 1.0 # Avoid sqrt(0), if N_s is 0, U will be 0 unless N(s,a) is also 0.

        # Extract relevant slices from parent's (current node's) arrays for its children
        # These are stats for edges (node, action_idx)
        child_total_w_sa = node.child_total_action_values[child_action_indices, parent_player_idx]
        child_visits_N_sa = node.child_visit_counts[child_action_indices].astype(np.float32) # Ensure float for division
        child_priors_P_sa = node.child_prior_probabilities[child_action_indices]

        # Q values: Q(s,a) = W(s,a) / N(s,a)
        # For unvisited children (N(s,a)=0), Q(s,a) is 0.
        q_values = np.zeros_like(child_total_w_sa, dtype=np.float32)
        # Create a mask for visited children to avoid division by zero
        # child_visits_N_sa is already float, so direct division is fine if we handle zeros.
        # N(s,a) can be 0. If N(s,a) is 0, Q(s,a) is 0.
        # If N(s,a) > 0, then Q(s,a) = W(s,a)/N(s,a)
        # The average_action_value array already stores Q(s,a) correctly.
        q_values = node.child_average_action_values[child_action_indices, parent_player_idx]


        # Apply Dirichlet noise to root's children's priors for exploration
        priors_for_puct = child_priors_P_sa.copy() # Make a copy to potentially modify with noise
        if node == self.root and self.alpha > 0 and self.noise_factor > 0 and len(child_action_indices) > 0:
            num_active_children = len(child_action_indices)
            noise = np.random.dirichlet([self.alpha] * num_active_children).astype(np.float32)
            priors_for_puct = (1 - self.noise_factor) * priors_for_puct + self.noise_factor * noise
            LOGGER.debug(f"Applied Dirichlet noise to root child priors. Original: {child_priors_P_sa}, Noisy: {priors_for_puct}")


        # U values: c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
        # Add 1 to N(s,a) in the denominator to encourage exploration of unvisited/less visited actions.
        u_values = self.c_puct * priors_for_puct * (sqrt_N_s / (1.0 + child_visits_N_sa))
        
        puct_scores = q_values + u_values
        LOGGER.debug(f"Node {id(node)} children details:")
        for i, action_idx in enumerate(child_action_indices):
            LOGGER.debug(
                f"  Child action {action_idx}: Q={q_values[i]:.3f}, P_orig={child_priors_P_sa[i]:.3f}, "
                f"P_puct={priors_for_puct[i]:.3f}, N(s,a)={child_visits_N_sa[i]:.0f}, U={u_values[i]:.3f}, PUCT={puct_scores[i]:.3f}"
            )


        # Select the child action that maximizes the PUCT score
        # np.argmax returns the index within the `child_action_indices` array
        best_idx_in_slice = np.argmax(puct_scores)
        best_action_idx = child_action_indices[best_idx_in_slice]
        best_child_node = node.children[best_action_idx]

        select_end_time = time.perf_counter()
        LOGGER.debug(
            f"Selected child: Node {id(best_child_node)} (Action Index: {best_action_idx}) "
            f"with PUCT score {puct_scores[best_idx_in_slice]:.3f}. Selection took {(select_end_time - select_start_time)*1000:.3f} ms."
        )
        return best_action_idx, best_child_node

    def expand_and_evaluate(self, leaf: Node) -> np.ndarray:
        """Expands leaf node, evaluates using network, returns value vector."""
        expand_start_time = time.perf_counter()
        LOGGER.debug(f"Expanding and evaluating leaf node: {id(leaf)}")

        # --- 1. Check for Terminal State ---
        if leaf.state.finished:
            LOGGER.info(f"Leaf node {id(leaf)} is a terminal state.")
            final_scores = leaf.state.result()
            LOGGER.debug(f"Final scores: {final_scores}")
            max_score = max(final_scores.values()) if final_scores else -float("inf")
            outcome_vector = np.zeros(self.num_players, dtype=np.float32)
            for player_id, score in final_scores.items():
                player_idx = self.player_id_to_idx.get(player_id)
                if player_idx is None:
                    raise ValueError(f"Player ID {player_id} from final scores not in player map.")
                outcome_vector[player_idx] = 1.0 if score == max_score else -1.0
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
        game_state_tensor, map_nodes_tensor, raw_edge_tensor = self.encoder.encode(leaf.state)
        game_state_tensor_b = game_state_tensor.to(self.device)  # (1, gs_features)
        x_map_nodes_batched = map_nodes_tensor.float().to(self.device)  # (N, node_features)
        edge_index_batched = raw_edge_tensor[0:2, :].long().to(self.device)  # (2, E)
        edge_attr_categorical_batched = raw_edge_tensor[2, :].long().to(self.device)  # (E,)
        num_nodes_in_graph = x_map_nodes_batched.shape[0]
        node_batch_idx = torch.zeros(num_nodes_in_graph, dtype=torch.long, device=self.device)  # (N,)

        network_start_time = time.perf_counter()
        with torch.no_grad():
            policy_logits_tensor, value_vector = self.model(
                game_state_tensor_b,  # (1, gs_features)
                x_map_nodes_batched,  # (N, node_features)
                edge_index_batched,  # (2, E)
                node_batch_idx,  # (N,)
                edge_attr_categorical_batched,  # (E,)
            )

        policy_probs_tensor = torch.softmax(policy_logits_tensor, dim=1)
        policy_probs_np = policy_probs_tensor.squeeze(0).cpu().numpy()
        value_np = value_vector.squeeze(0).cpu().numpy()
        network_end_time = time.perf_counter()
        LOGGER.debug(f"Network evaluation took {(network_end_time - network_start_time)*1000:.3f} ms.")
        LOGGER.debug(f"Raw value vector output (CPU): {value_vector}")

        # --- 3. Expand Children ---
        LOGGER.debug("Expanding children based on policy.")
        try:
            legal_mask_np = self.action_mapper.get_legal_action_mask(leaf.state)
        except Exception as e:
            LOGGER.info(f"State actions: {leaf.state.raw_actions}")
            LOGGER.exception(f"Error getting legal action mask for node {id(leaf)}.")
            raise e

        if np.sum(legal_mask_np) == 0:
            from rl18xx.game.action_helper import ActionHelper
            LOGGER.error(leaf.state.raw_actions)
            LOGGER.error(ActionHelper(leaf.state).get_all_choices_limited())
            LOGGER.error(np.nonzero(legal_mask_np).squeeze().tolist())
            LOGGER.error(f"No legal moves found by mask for non-terminal node {id(leaf)}. Policy sum ~0.")
            raise ValueError("No legal moves found by mask for non-terminal state")

        masked_policy = policy_probs_np * legal_mask_np
        policy_sum = np.sum(masked_policy)

        if policy_sum < 1e-8: # If all legal moves have ~zero policy
            if np.sum(legal_mask_np) > 0: # Check if there were any legal moves at all
                LOGGER.warning(
                    f"Node {id(leaf)}: Policy sum is ~0 ({policy_sum:.2e}) but legal moves exist. "
                    f"Using uniform distribution over legal moves for expansion priors."
                )
                # Fallback: uniform probability for legal actions if policy is degenerate
                num_legal_moves = np.sum(legal_mask_np)
                legal_policy_np = legal_mask_np / num_legal_moves
            else:
                # This case should ideally be caught by terminal state check or if no legal_mask_np earlier
                LOGGER.error(f"No legal moves and policy sum is zero for non-terminal node {id(leaf)}. This is problematic.")
                # This implies the game might be over or in a dead-end state not caught by `leaf.state.finished`
                # For safety, return a neutral value; backpropagation will handle it.
                # The node remains a leaf in MCTS terms if no children are expanded.
                return np.zeros(self.num_players, dtype=np.float32) # Or some other default like -1 for current player
        else:
            legal_policy_np = masked_policy / policy_sum
        
        leaf.child_prior_probabilities = legal_policy_np.copy()

        action_indices_to_expand = np.where(legal_policy_np > 1e-8)[0] # Use a small threshold
        LOGGER.debug(f"Found {len(action_indices_to_expand)} legal actions with non-zero policy to expand.")

        num_expanded = 0
        for action_index in action_indices_to_expand:
            prior_p = legal_policy_np[action_index]
            try:
                next_state = leaf.state.clone(leaf.state.raw_actions)
                action = self.action_mapper.map_index_to_action(action_index, next_state)
                LOGGER.debug(f"Mapping index {action_index} to action {action} for node {id(leaf)}")
                next_state.process_action(action)

                # Expand child: creates node, sets prior in parent (leaf), adds to leaf.children
                new_child = leaf.expand_child(action_index, next_state)
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
        return value_np

    def backpropagate(self, leaf: Node, value_vector: np.ndarray):
        """Backpropagates the value vector up the tree."""
        LOGGER.debug(f"Starting backpropagation from leaf {id(leaf)} with value {value_vector}")
        current_node: Optional[Node] = leaf
        path_len = 0
        while current_node is not None: 
            if path_len == 0: 
                current_node.node_total_visits +=1 
                LOGGER.debug(f"  Updating leaf node {id(current_node)}: N_state={current_node.node_total_visits}")

            if current_node.parent is not None and current_node.action_index is not None:
                parent_node = current_node.parent
                action_idx_from_parent = current_node.action_index
                
                player_idx_at_parent = self.get_player_index(parent_node.state)

                # --- Inlined logic from former Node.update_child_stats_after_simulation ---
                # Update W(s,a) for parent_node and action_idx_from_parent
                parent_node.child_total_action_values[action_idx_from_parent, :] += value_vector

                # Revert virtual loss if it was applied
                if self.apply_virtual_loss:
                    parent_node.child_total_action_values[
                        action_idx_from_parent, player_idx_at_parent
                    ] += self.virtual_loss_c

                # Update average action value Q(s,a) using the visit count from selection
                if parent_node.child_visit_counts[action_idx_from_parent] > 0:
                    parent_node.child_average_action_values[action_idx_from_parent, :] = (
                        parent_node.child_total_action_values[action_idx_from_parent, :] / 
                        parent_node.child_visit_counts[action_idx_from_parent]
                    )
                else:
                    # This case should ideally not be hit if N(s,a) was incremented in selection
                    # and this action was on the path.
                    LOGGER.warning(
                        f"Node {parent_node.id}, child action {action_idx_from_parent}: "
                        f"child_visit_count is 0 during Q value update. "
                        f"W_sa={parent_node.child_total_action_values[action_idx_from_parent, :]}. "
                        f"Q_sa will not be updated meaningfully."
                    )
                # --- End of inlined logic ---
                
                LOGGER.debug(
                    f"    Updated parent {id(parent_node)} stats for child action {action_idx_from_parent}: "
                    f"N_sa={parent_node.child_visit_counts[action_idx_from_parent]}, "
                    f"Q_sa={parent_node.child_average_action_values[action_idx_from_parent]}"
                )

            current_node = current_node.parent
            path_len += 1
        LOGGER.debug(f"Backpropagation complete. Updated {path_len} nodes.")

    def get_policy(self, temperature: float = 1.0) -> Tuple[Dict[int, float], np.ndarray]:
        """Calculates the final policy distribution based on root children visit counts."""
        policy_start_time = time.perf_counter()
        if not self.root.children: # Check if any children were ever expanded
            LOGGER.error("Root node has no children (no actions explored/expanded), cannot calculate policy.")
            return {}, np.zeros(self.action_encoding_size, dtype=np.float32)


        # Get action indices of actually expanded children from the root
        child_action_indices = np.array(list(self.root.children.keys()), dtype=np.int32)
        if len(child_action_indices) == 0:
            LOGGER.warning("Root node has children dict but it's empty. This is unusual if not terminal.")
            return {}, np.zeros(self.action_encoding_size, dtype=np.float32)


        # Get visit counts for these specific children actions from the root's child_visit_counts array
        visit_counts_N_sa = self.root.child_visit_counts[child_action_indices].astype(np.float32)
        LOGGER.debug(f"Root children visit counts N(s_root, a): {dict(zip(child_action_indices, visit_counts_N_sa))}")

        probs: np.ndarray # This array will store probabilities corresponding to child_action_indices

        if np.sum(visit_counts_N_sa) == 0:
            LOGGER.warning(
                "All root's explored children have zero visits. This might indicate an issue or very few simulations. "
                "Falling back to priors or uniform if priors are also zero."
            )
            # Fallback to priors of these explored children
            child_priors = self.root.child_prior_probabilities[child_action_indices]
            if np.sum(child_priors) > 1e-6:
                probs = child_priors / np.sum(child_priors)
                LOGGER.debug("Using normalized priors of explored children for policy.")
            else: # If priors also sum to zero (or no children), uniform over explored children
                LOGGER.warning("Root's explored children priors also sum to zero. Falling back to uniform distribution over them.")
                num_explored_children = len(child_action_indices)
                if num_explored_children > 0:
                    probs = np.ones_like(visit_counts_N_sa) / num_explored_children
                else: # Should have been caught by earlier checks
                    probs = np.array([], dtype=np.float32) 
                LOGGER.debug("Using uniform distribution for policy over explored children.")
        elif temperature == 0:  # Choose greedily based on visits
            LOGGER.debug("Temperature is 0, choosing greedily based on max visit count.")
            probs = np.zeros_like(visit_counts_N_sa) # Initialize probs for explored children
            if len(visit_counts_N_sa) > 0: # Ensure there are counts to process
                max_visit_val = np.max(visit_counts_N_sa)
                # Find all indices (relative to child_action_indices) that have this max visit count
                best_indices_in_slice = np.where(visit_counts_N_sa == max_visit_val)[0]
                
                # Break ties randomly if multiple max visits
                chosen_relative_index = np.random.choice(best_indices_in_slice)
                probs[chosen_relative_index] = 1.0
                LOGGER.debug(
                    f"Chosen index (relative to explored children): {chosen_relative_index}, "
                    f"Action Index: {child_action_indices[chosen_relative_index]}"
                )
            else: # Should ideally not be reached if child_action_indices is not empty
                probs = np.array([], dtype=np.float32)
        else:
            # Apply temperature
            LOGGER.debug(f"Applying temperature {temperature} to visit counts {visit_counts_N_sa}.")
            temp_visits = visit_counts_N_sa ** (1.0 / temperature)
            LOGGER.debug(f"Temperature-adjusted visits: {temp_visits}")
            sum_temp_visits = np.sum(temp_visits)
            if sum_temp_visits > 1e-8: # Avoid division by zero
                probs = temp_visits / sum_temp_visits
            else: # Fallback if sum is zero (e.g., all visits zero, or extreme temperature)
                LOGGER.warning(f"Sum of temperature-adjusted visits is ~0. Falling back to uniform over explored children.")
                num_explored_children = len(child_action_indices)
                if num_explored_children > 0:
                    probs = np.ones_like(visit_counts_N_sa) / num_explored_children
                else:
                    probs = np.array([], dtype=np.float32)
            LOGGER.debug(f"Probabilities after temperature: {dict(zip(child_action_indices, probs)) if len(probs) == len(child_action_indices) else 'probs length mismatch'}")

        # Create policy dictionary (sparse: only non-zero probabilities) and full policy vector
        policy_dict: Dict[int, float] = {}
        full_policy_vector = np.zeros(self.action_encoding_size, dtype=np.float32)

        if len(child_action_indices) == len(probs): # Ensure alignment before populating
            for i, action_idx in enumerate(child_action_indices):
                prob_val = probs[i]
                full_policy_vector[action_idx] = prob_val # Populate full vector with the calculated probability
                if prob_val > 1e-8: # Only add to dictionary if probability is meaningfully non-zero
                    policy_dict[action_idx] = prob_val
        else:
            # This case indicates an issue earlier, policy_dict will be empty and full_policy_vector zero.
            LOGGER.error(
                f"Mismatch in lengths between child_action_indices ({len(child_action_indices)}) and "
                f"calculated probs ({len(probs)}). Policy will be empty/zero."
            )
            # policy_dict remains {}, full_policy_vector remains all zeros

        policy_end_time = time.perf_counter()
        LOGGER.debug(f"Policy calculation finished in {(policy_end_time - policy_start_time)*1000:.3f} ms.")
        LOGGER.debug(f"Final policy dict: {policy_dict}")
        LOGGER.debug(f"Final policy vector shape: {full_policy_vector.shape}, Sum: {np.sum(full_policy_vector):.4f}")

        return policy_dict, full_policy_vector
