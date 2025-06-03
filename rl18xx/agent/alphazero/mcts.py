from __future__ import annotations
import collections
import math
import numpy as np
from typing import Optional
import scipy
import torch
import logging
from rl18xx.game.engine.game.base import BaseGame
from rl18xx.agent.alphazero.encoder import Encoder_1830
from rl18xx.agent.alphazero.action_mapper import ActionMapper
from rl18xx.agent.alphazero.config import SelfPlayConfig
import time

POLICY_SIZE = 26535  # This is calculated dynamically but shouldn't be.
VALUE_SIZE = 4  # We always want to play a 4-player game.

LOGGER = logging.getLogger(__name__)

# TODO: See if we really need this.
def calculate_entropy(probabilities: np.ndarray) -> float:
    """Calculates the entropy of a probability distribution."""
    probabilities = probabilities[probabilities > 0]  # Avoid log(0)
    if len(probabilities) == 0 or not np.isclose(np.sum(probabilities), 1.0, atol=1e-5):
        # If not a valid distribution (e.g. all zeros, or doesn't sum to 1)
        # Or if sum is not 1, scipy.stats.entropy might give misleading results or errors
        # For non-normalized positive arrays, it calculates sum(p_i * log(p_i)), which isn't Shannon entropy.
        # We expect normalized probabilities here.
        if not np.isclose(np.sum(probabilities), 1.0, atol=1e-5) and len(probabilities) > 0:
            LOGGER.debug(f"Probabilities do not sum to 1 for entropy calculation: sum={np.sum(probabilities)}")
        return 0.0
    return scipy.stats.entropy(probabilities)

class DummyNode:
    """A fake node of a MCTS search tree.

    This node is intended to be a placeholder for the root node, which would
    otherwise have no parent node. If all nodes have parents, code becomes
    simpler."""

    def __init__(self):
        self.parent = None
        self.game_object = None
        self.legal_action_indices = [None]
        self.child_N_compressed = collections.defaultdict(float)
        self.child_W_compressed = collections.defaultdict(lambda: np.zeros([VALUE_SIZE], dtype=np.float32))


class MCTSNode:
    def __init__(
        self,
        game_state: BaseGame,
        fmove: Optional[int] = None,
        parent: Optional[MCTSNode | DummyNode] = None,
        config: Optional[SelfPlayConfig] = None,
    ):
        if parent is None:
            parent = DummyNode()
        self.parent = parent
        self.fmove = fmove  # move index that led to this position
        self.is_expanded = False
        self.losses_applied = 0  # number of virtual losses on this node

        self.depth = 0
        if isinstance(parent, MCTSNode):
            self.depth = parent.depth + 1

        self.game_object: BaseGame = game_state
        self.encoded_game_state = Encoder_1830().encode(self.game_object) # Use game_object
        self.action_mapper = ActionMapper()

        self.player_mapping = {p.id: i for i, p in enumerate(sorted(self.game_object.players, key=lambda x: x.id))}
        self.active_player_index = self.player_mapping[self.game_object.active_players()[0].id]

        self.legal_action_indices = self.action_mapper.get_legal_action_indices(self.game_object) # Use game_object
        self.num_legal_actions = len(self.legal_action_indices)
        self.child_N_compressed = np.zeros(self.num_legal_actions, dtype=np.float32)
        self.child_W_compressed = np.zeros([self.num_legal_actions, VALUE_SIZE], dtype=np.float32)
        self.original_prior_compressed = np.zeros(self.num_legal_actions, dtype=np.float32)
        self.child_prior_compressed = np.zeros(self.num_legal_actions, dtype=np.float32)

        self.children = {}
        self.config = config or SelfPlayConfig()
        self.add_metric("MCTS/Depth", self.depth)

        if self.is_done():
            self.game_object.end_game()

    def add_metric(self, name, value):
        if self.config.metrics is None:
            return
        self.config.metrics.add_scalar(name, value, self.config.global_step, self.config.game_idx_in_iteration)

    def __repr__(self):
        return f"<MCTSNode move_number={self.game_object.move_number}, move=[{self.fmove}], N={self.N if self.parent and self.fmove is not None else 'N/A'}, to_play={self.active_player_index}>"

    def _expand_to_full_policy_size(self, compressed_array: np.ndarray, default_value: float = 0.0):
        """Expand compressed array back to full policy size for vectorized operations"""
        if compressed_array.ndim == 1:
            full_array = np.full(POLICY_SIZE, default_value, dtype=np.float32)
            full_array[self.legal_action_indices] = compressed_array
        else:  # 2D array
            full_array = np.full([POLICY_SIZE, compressed_array.shape[1]], default_value, dtype=np.float32)
            full_array[self.legal_action_indices] = compressed_array
        return full_array

    @property
    def legal_action_mask(self):
        mask = np.zeros(POLICY_SIZE, dtype=np.float32)
        mask[self.legal_action_indices] = 1.0
        return mask

    @property
    def child_N(self):
        return self._expand_to_full_policy_size(self.child_N_compressed)

    @property
    def child_W(self):
        return self._expand_to_full_policy_size(self.child_W_compressed)

    @property
    def original_prior(self):
        return self._expand_to_full_policy_size(self.original_prior_compressed)

    @original_prior.setter
    def original_prior(self, probs):
        self.original_prior_compressed = probs[self.legal_action_indices]

    @property
    def child_prior(self):
        return self._expand_to_full_policy_size(self.child_prior_compressed)

    @child_prior.setter
    def child_prior(self, probs):
        self.child_prior_compressed = probs[self.legal_action_indices]

    @property
    def child_action_score(self):
        expanded_child_action_score = self._expand_to_full_policy_size(
            self.child_action_score_compressed, default_value=-1000.0
        )
        return expanded_child_action_score

    @property
    def child_action_score_compressed(self):
        q_values_for_current_player = self.child_Q_compressed[:, self.active_player_index]
        return q_values_for_current_player + self.child_U_compressed

    @property
    def child_Q(self):
        LOGGER.warning("only use child_Q for tests")
        return self._expand_to_full_policy_size(self.child_Q_compressed)

    @property
    def child_Q_compressed(self):
        # child_W_compressed is [num_legal_actions, num_players]
        # child_N_compressed is [num_legal_actions]
        # Make this an explicit broadcast
        return self.child_W_compressed / (1 + self.child_N_compressed[:, np.newaxis])

    @property
    def child_U(self):
        LOGGER.warning("only use child_U for tests")
        return self._expand_to_full_policy_size(self.child_U_compressed)

    @property
    def child_U_compressed(self):
        # U for all children (legal moves only)
        # U(s, a) = c_puct * P(s, a) * sqrt(N(s)) / (1 + N(s, a)))
        # c_puct = 2.0 * (log(1.0 + N(s, parent)) + c_puct_init) * sqrt(1 / (1 + N(s, a)))
        # P(s, a) = child_prior
        # N(s) = self.N
        # N(s, a) = child_N
        c_puct = 2.0 * (
            math.log((1.0 + self.N + self.config.c_puct_base) / self.config.c_puct_base) + self.config.c_puct_init
        )
        p_s_a = self.child_prior_compressed
        n_s = max(1, self.N - 1)
        n_s_a = self.child_N_compressed
        return c_puct * p_s_a * math.sqrt(n_s) / (1 + n_s_a)

    @property
    def Q(self):
        return self.W / (1.0 + self.N)

    @property
    def N(self):
        compressed_index = self.parent.legal_action_indices.index(self.fmove)
        return self.parent.child_N_compressed[compressed_index]

    @N.setter
    def N(self, value):
        compressed_index = self.parent.legal_action_indices.index(self.fmove)
        self.parent.child_N_compressed[compressed_index] = value

    @property
    def W(self):
        compressed_index = self.parent.legal_action_indices.index(self.fmove)
        return self.parent.child_W_compressed[compressed_index]

    @W.setter
    def W(self, value):
        compressed_index = self.parent.legal_action_indices.index(self.fmove)
        self.parent.child_W_compressed[compressed_index] = value

    @property
    def Q_perspective(self):
        """Return value of position, from perspective of player to play."""
        return self.Q[self.active_player_index]

    def select_leaf(self):
        start_time = time.time()
        current = self
        num_added = 0
        while current.is_expanded:
            best_move = np.argmax(current.child_action_score)
            current = current.maybe_add_child(best_move)
            num_added += 1
        end_time = time.time()
        self.add_metric("MCTS/Select_Leaf_Time", end_time - start_time)
        self.add_metric("MCTS/Select_Leaf_Path_Length", num_added)
        return current

    def maybe_add_child(self, action_index: int):
        """Adds child node for action_index if it doesn't already exist, and returns it."""
        start_time = time.time()

        if action_index not in self.children:
            clone_start_time = time.time()
            try:
                new_position = self.game_object.deep_copy_clone()
            except Exception as e:
                LOGGER.error(f"Error cloning game_object in MCTSNode (fmove={self.fmove}): {e}", exc_info=True)
                raise e
            clone_duration = time.time() - clone_start_time

            process_action_start_time = time.time()
            try:
                action_to_take = self.action_mapper.map_index_to_action(action_index, new_position)
                new_position.process_action(action_to_take)
            except Exception as e:
                LOGGER.error(
                    f"Error processing action in maybe_add_child. Parent fmove: {self.fmove}, "
                    f"Action index: {action_index}, Action to take: {action_to_take}",
                    exc_info=True
                )
                LOGGER.error(f"Parent game actions: {self.game_object.raw_actions}")
                raise e
            
            self.children[action_index] = MCTSNode(
                new_position,
                fmove=action_index,
                parent=self,
                config=self.config
            )
            process_action_duration = time.time() - process_action_start_time
            
            self.add_metric("MCTS/Maybe_Add_Child_Clone_Duration", clone_duration)
            self.add_metric("MCTS/Maybe_Add_Child_Process_Action_Duration", process_action_duration)
        
        overall_call_duration = time.time() - start_time
        self.add_metric("MCTS/Maybe_Add_Child_Overall_Duration", overall_call_duration)
        return self.children[action_index]

    def add_virtual_loss(self, up_to):
        """Propagate a virtual loss up to the root node.

        Args:
            up_to: The node to propagate until. (Keep track of this! You'll
                need it to reverse the virtual loss later.)
        """
        # We need to apply a loss to the current node's value for the player
        # that took the action to get us here (i.e. the parent node's player).
        # We should ignore the root node.
        if self.parent is None or self is up_to:
            return
        self.losses_applied += 1
        prev_player = self.parent.active_player_index
        self.W[prev_player] -= 1
        self.parent.add_virtual_loss(up_to)

    def revert_virtual_loss(self, up_to):
        if self.parent is None or self is up_to:
            return
        self.losses_applied -= 1
        prev_player = self.parent.active_player_index
        self.W[prev_player] += 1
        self.parent.revert_virtual_loss(up_to)

    def incorporate_results(self, move_probabilities, value, up_to):
        assert move_probabilities.shape == (
            POLICY_SIZE,
        ), f"move_probabilities.shape: {move_probabilities.shape} must be ({POLICY_SIZE},)"
        assert value.shape == (VALUE_SIZE,), f"value.shape: {value.shape} must be ({VALUE_SIZE},)"
        # A finished game should not be going through this code path - should
        # directly call backup_value() on the result of the game.
        assert not self.game_object.finished

        # If a node was picked multiple times (despite vlosses), we shouldn't
        # expand it more than once.
        if self.is_expanded:
            return
        self.is_expanded = True

        # Move to numpy
        if isinstance(move_probabilities, torch.Tensor):
            move_probabilities = move_probabilities.cpu().numpy()
        if isinstance(value, torch.Tensor):
            value = value.cpu().numpy()

        # Zero out illegal moves.
        move_probs = move_probabilities * self.legal_action_mask
        scale = sum(move_probs)
        if scale > 0:
            # Re-normalize move_probabilities.
            move_probs /= scale

        self.original_prior = self.child_prior = move_probs
        # initialize child Q as current node's value, to prevent dynamics where
        # if B is winning, then B will only ever explore 1 move, because the Q
        # estimation will be so much larger than the 0 of the other moves.
        #
        # Conversely, if W is winning, then B will explore all 362 moves before
        # continuing to explore the most favorable move. This is a waste of search.
        #
        # The value seeded here acts as a prior, and gets averaged into Q calculations.
        self.child_W_compressed = np.tile(value, (self.num_legal_actions, 1))
        self.backup_value(value, up_to=up_to)

    def backup_value(self, value, up_to):
        """Propagates a value estimation up to the root node.

        Args:
            value: the value to be propagated
            up_to: the node to propagate until.
        """
        self.N += 1
        self.W += value
        if self.parent is None or self is up_to:
            return
        self.parent.backup_value(value, up_to)

    def is_done(self):
        return self.game_object.finished or self.game_object.move_number >= self.config.max_game_length

    def game_result(self) -> Optional[np.ndarray]:
        if not self.is_done():
            LOGGER.warning(f"Getting game result for unfinished game (node fmove: {self.fmove}).")

        result = self.game_object.result()
        winning_score = max(result.values())

        value = np.full(VALUE_SIZE, -1.0, dtype=np.float32)
        winners = [self.player_mapping[pid] for pid, score in result.items() if score == winning_score]
        if len(winners) > 1:
            value[winners] = 0.0
        else:
            value[winners] = 1.0

        return value

    def game_result_string(self) -> Optional[str]:
        if not self.is_done():
            LOGGER.warning(f"game_result_string: Game is not finished (node fmove: {self.fmove}). Returning empty string.")
            return ""

        result = self.game_object.result()
        winning_score = max(result.values())

        winners = []
        for player_id, score in result.items():
            if score == winning_score:
                winners.append(str(player_id))
        result_string = f"{', '.join(winners)} ({winning_score})"

        return result_string

    def inject_noise(self):
        if self.num_legal_actions == 0:
            return

        alpha = np.full(self.num_legal_actions, self.config.dirichlet_noise_alpha)
        dirichlet = np.random.dirichlet(alpha)
        self.child_prior_compressed = (
            self.child_prior_compressed * (1 - self.config.dirichlet_noise_weight)
            + dirichlet * self.config.dirichlet_noise_weight
        )

    def children_as_pi(self, squash=False) -> np.ndarray:
        probs = self.child_N
        if squash:
            probs = probs**0.98
        sum_probs = np.sum(probs)
        if sum_probs == 0:
            return probs
        return probs / np.sum(probs)

    def best_child(self) -> int:
        # Sort by child_N tie break with action score.
        return np.argmax(self.child_N + self.child_action_score / 10000)

    def most_visited_path_nodes(self) -> list[MCTSNode]:
        node = self
        output = []
        while node.children:
            node = node.children.get(node.best_child())
            assert node is not None
            output.append(node)
        return output

    def most_visited_path(self) -> str:
        output = []
        current_node: MCTSNode | None = self
        path_nodes = self.most_visited_path_nodes()

        for i, node_on_path in enumerate(path_nodes):
            parent_move_number = node_on_path.parent.game_object.move_number if isinstance(node_on_path.parent, MCTSNode) else 'root'
            output.append(f"{parent_move_number}: {node_on_path.fmove} (N={node_on_path.N:.0f}) ==> ")

        if path_nodes:
            final_node_on_path = path_nodes[-1]
            output.append(f"Q: {final_node_on_path.Q_perspective:.3f} (Player {final_node_on_path.active_player_index}) N={final_node_on_path.N:.0f}\n")
        else:
            output.append(f"Q: {self.Q_perspective:.3f} (Player {self.active_player_index}) N={self.N:.0f} (Root)\n")
        return "".join(output)

    def rank_children(self):
        ranked_children = list(range(self.num_legal_actions))
        ranked_children.sort(
            key=lambda i: (self.child_N_compressed[i], self.child_action_score_compressed[i]), reverse=True
        )
        return ranked_children

    def describe(self):
        ranked_children_indices = self.rank_children()
        soft_n = self.child_N_compressed / max(1, sum(self.child_N_compressed))
        prior = self.child_prior_compressed
        safe_prior = np.where(prior == 0, 1e-9, prior)
        p_delta = soft_n - prior
        p_rel = p_delta / safe_prior
        
        output = []
        output.append("Q (player {}): {:.4f}, N: {:.0f}\n".format(self.active_player_index, self.Q_perspective, self.N))
        output.append(self.most_visited_path())
        output.append("idx  | Action (from parent) | Q (curr) |    U    |   P(a|s) | P_orig  |    N   | soft-N | p-delta | p-rel")
        
        for compressed_idx in ranked_children_indices[:15]:
            global_action_index = self.legal_action_indices[compressed_idx]

            if self.child_N_compressed[compressed_idx] == 0:
                break
            
            action_display = f"{global_action_index}"

            output.append(
                "\n{:4} | {:<18} | {: .3f}  | {: .3f} | {: .5f} | {:.5f} | {:6.0f} | {:.4f} | {: .5f} | {: .2f}".format(
                    compressed_idx,
                    action_display,
                    self.child_Q_compressed[compressed_idx][self.active_player_index],
                    self.child_U_compressed[compressed_idx],
                    self.child_prior_compressed[compressed_idx],
                    self.original_prior_compressed[compressed_idx],
                    self.child_N_compressed[compressed_idx],
                    soft_n[compressed_idx],
                    p_delta[compressed_idx],
                    p_rel[compressed_idx],
                )
            )
        return "".join(output)
