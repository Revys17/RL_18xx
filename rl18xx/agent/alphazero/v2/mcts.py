from __future__ import annotations
import collections
import math
import numpy as np
from typing import Optional
import torch
import logging
from rl18xx.game.engine.game.base import BaseGame
from rl18xx.agent.alphazero.encoder import Encoder_1830
from rl18xx.agent.alphazero.action_mapper import ActionMapper
from rl18xx.agent.alphazero.v2.config import MegaConfig

POLICY_SIZE = 26535 # This is calculated dynamically but shouldn't be.
VALUE_SIZE = 4 # We always want to play a 4-player game.

LOGGER = logging.getLogger(__name__)

class DummyNode:
    """A fake node of a MCTS search tree.

    This node is intended to be a placeholder for the root node, which would
    otherwise have no parent node. If all nodes have parents, code becomes
    simpler."""

    def __init__(self):
        self.parent = None
        self.game_dict = None
        self.legal_action_indices = [None]
        self.child_N_compressed = collections.defaultdict(float)
        self.child_W_compressed = collections.defaultdict(lambda: np.zeros([VALUE_SIZE], dtype=np.float32))


class MCTSNode:
    def __init__(self, game_state: BaseGame, fmove: Optional[int]=None, parent: Optional[MCTSNode]=None, config: Optional[MegaConfig]=None):
        if parent is None:
            parent = DummyNode()
        self.parent = parent
        self.fmove = fmove  # move index that led to this position
        self.is_expanded = False
        self.losses_applied = 0  # number of virtual losses on this node

        self.game_dict = game_state.to_dict()
        self.encoded_game_state = Encoder_1830().encode(game_state)
        self.action_mapper = ActionMapper()
        self.player_mapping = {p.id: i for i, p in enumerate(sorted(game_state.players, key=lambda x: x.id))}
        self.active_player_index = self.player_mapping[game_state.active_players()[0].id]

        # using child_() allows vectorized computation of action score.
        self.legal_action_indices = self.action_mapper.get_legal_action_indices(game_state)
        self.num_legal_actions = len(self.legal_action_indices)
        self.child_N_compressed = np.zeros(self.num_legal_actions, dtype=np.float32)
        self.child_W_compressed = np.zeros([self.num_legal_actions, VALUE_SIZE], dtype=np.float32)
        self.original_prior_compressed = np.zeros(self.num_legal_actions, dtype=np.float32)
        self.child_prior_compressed = np.zeros(self.num_legal_actions, dtype=np.float32)

        # self.illegal_moves = 1 - self.action_mapper.get_legal_action_mask(game_state)
        # self.child_N = np.zeros([POLICY_SIZE], dtype=np.float32)
        # self.child_W = np.zeros([POLICY_SIZE, VALUE_SIZE], dtype=np.float32)
        # # save a copy of the original prior before it gets mutated by d-noise.
        # self.original_prior = np.zeros([POLICY_SIZE], dtype=np.float32)
        # self.child_prior = np.zeros([POLICY_SIZE], dtype=np.float32)

        self.children = {}  # map of flattened moves to resulting MCTSNode
        self.config = config or MegaConfig()

    def __repr__(self):
        return f"<MCTSNode move_number={len(self.game_dict['actions'])}, move=[{self.fmove}], N={self.N if self.parent else 'N/A'}, to_play={self.active_player_index}>"

    def _expand_to_full_policy_size(self, compressed_array: np.ndarray, default_value: float=0.0):
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
        LOGGER.info(f"setting child_prior for node {self.game_dict['move_number']}, action {self.fmove}: {probs}")
        self.child_prior_compressed = probs[self.legal_action_indices]

    @property
    def child_action_score(self):
        LOGGER.info(f"expanded child_action_score: {self._expand_to_full_policy_size(self.child_action_score_compressed)}")
        return self._expand_to_full_policy_size(self.child_action_score_compressed)

    @property
    def child_action_score_compressed(self):
        child_Q_compressed = self.child_Q_compressed
        child_U_compressed = self.child_U_compressed
        q_values_for_current_player = child_Q_compressed[:, self.active_player_index]
        LOGGER.info(f"current player: {self.active_player_index}")
        LOGGER.info(f"q_values_for_current_player: {q_values_for_current_player}")
        LOGGER.info(f"child_U_compressed: {child_U_compressed}")
        LOGGER.info(f"child_action_score_compressed: {q_values_for_current_player + child_U_compressed}")
        return (q_values_for_current_player + child_U_compressed)

    @property
    def child_Q_compressed(self):
        # child_W_compressed is [num_legal_actions, num_players]
        # child_N_compressed is [num_legal_actions]
        # Make this an explicit broadcast
        LOGGER.info(f"child_W_compressed: {self.child_W_compressed}")
        LOGGER.info(f"child_N_compressed: {self.child_N_compressed}")
        LOGGER.info(f"child_Q_compressed: {self.child_W_compressed / (1 + self.child_N_compressed[:, np.newaxis])}")
        return self.child_W_compressed / (1 + self.child_N_compressed[:, np.newaxis])

    @property
    def child_U_compressed(self):
        # U for all children (legal moves only)
        # U(s, a) = c_puct * P(s, a) * sqrt(N(s)) / (1 + N(s, a)))
        # c_puct = 2.0 * (log(1.0 + N(s, parent)) + c_puct_init) * sqrt(1 / (1 + N(s, a)))
        # P(s, a) = child_prior
        # N(s) = self.N
        # N(s, a) = child_N
        c_puct = 2.0 * (math.log((1.0 + self.N + self.config.c_puct_base) / self.config.c_puct_base) + self.config.c_puct_init)
        LOGGER.info(f"c_puct: {c_puct}")
        p_s_a = self.child_prior_compressed
        LOGGER.info(f"p_s_a: {p_s_a}")
        n_s = max(1, self.N - 1)
        LOGGER.info(f"n_s: {n_s}")
        n_s_a = self.child_N_compressed
        LOGGER.info(f"n_s_a: {n_s_a}")
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
        current = self
        while True:
            # if a node has never been evaluated, we have no basis to select a child.
            if not current.is_expanded:
                break

            best_move = np.argmax(current.child_action_score)
            current = current.maybe_add_child(best_move)
        return current

    def maybe_add_child(self, action_index: int):
        """Adds child node for action_index if it doesn't already exist, and returns it."""
        if action_index not in self.children:
            #LOGGER.info(f"Adding child {action_index} to {self.game_dict['move_number']}")
            #LOGGER.info(f"Game dict: {self.game_dict}")
            new_position = BaseGame.load(self.game_dict)
            new_position.process_action(self.action_mapper.map_index_to_action(action_index, new_position))
            self.children[action_index] = MCTSNode(new_position, fmove=action_index, parent=self, config=self.config)
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
        LOGGER.info(f"adding virtual loss to node {self.game_dict['move_number']}, action {self.fmove}")
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
        self.W[self.active_player_index] += 1
        self.parent.revert_virtual_loss(up_to)

    def incorporate_results(self, move_probabilities, value, up_to):
        assert move_probabilities.shape == (POLICY_SIZE,), f"move_probabilities.shape: {move_probabilities.shape} must be ({POLICY_SIZE},)"
        assert value.shape == (VALUE_SIZE,), f"value.shape: {value.shape} must be ({VALUE_SIZE},)"
        # A finished game should not be going through this code path - should
        # directly call backup_value() on the result of the game.
        assert not self.game_dict['finished']

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
        LOGGER.info(f"move_probs: {move_probabilities}")
        LOGGER.info(f"legal_action_mask: {self.legal_action_mask}")
        move_probs = move_probabilities * self.legal_action_mask
        LOGGER.info(f"move_probs after legal_action_mask: {move_probs}")
        scale = sum(move_probs)
        LOGGER.info(f"scale: {scale}")
        if scale > 0:
            # Re-normalize move_probabilities.
            move_probs *= 1 / scale

        LOGGER.info(f"scaled move_probs: {move_probs}")
        self.original_prior = self.child_prior = move_probs
        # initialize child Q as current node's value, to prevent dynamics where
        # if B is winning, then B will only ever explore 1 move, because the Q
        # estimation will be so much larger than the 0 of the other moves.
        #
        # Conversely, if W is winning, then B will explore all 362 moves before
        # continuing to explore the most favorable move. This is a waste of search.
        #
        # The value seeded here acts as a prior, and gets averaged into Q calculations.
        LOGGER.info(f"setting child_W_compressed for node {self.game_dict['move_number']}, action {self.fmove}: {value}")
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
        return self.game_dict['finished'] or self.game_dict['move_number'] >= self.config.max_game_length

    def game_result(self) -> Optional[np.ndarray]:
        if not self.game_dict['finished']:
            LOGGER.warning(f"Getting game result for unfinished game.")

        result = self.game_dict['result']
        winning_score = max(result.values())

        value = np.full(VALUE_SIZE, -1.0, dtype=np.float32)
        winners = [self.player_mapping[pid] for pid, score in result.items() if score == winning_score]
        value[winners] = 1.0

        return value
    
    def game_result_string(self) -> Optional[str]:
        if not self.game_dict['finished']:
            LOGGER.warning(f"game_result_string: Game is not finished. Returning empty string.")
            return ""

        result = self.game_dict['result']
        winning_score = max(result.values())

        winners = []
        for player, score in result.items():
            if score == winning_score:
                winners.append(player)
        result_string = f"{', '.join(winners)} ({winning_score})"

        return result_string

    def inject_noise(self):
        if self.num_legal_actions == 0:
            return

        alpha = np.full(self.num_legal_actions, self.config.dirichlet_noise_alpha)
        dirichlet = np.random.dirichlet(alpha)
        self.child_prior_compressed = (
            self.child_prior_compressed * (1 - self.config.dirichlet_noise_weight) +
            dirichlet * self.config.dirichlet_noise_weight
        )

    def children_as_pi(self, squash=False):
        probs = self.child_N
        if squash:
            probs = probs ** .98
        sum_probs = np.sum(probs)
        if sum_probs == 0:
            return probs
        return probs / np.sum(probs)

    def best_child(self):
        # Sort by child_N tie break with action score.
        return np.argmax(self.child_N + self.child_action_score / 10000)

    def most_visited_path_nodes(self):
        node = self
        output = []
        while node.children:
            node = node.children.get(node.best_child())
            assert node is not None
            output.append(node)
        return output

    def most_visited_path(self):
        output = []
        node = self
        for node in self.most_visited_path_nodes():
            output.append(f"{self.game_dict['move_number']}: {node.fmove} ({node.N}) ==> ")

        output.append(f"Q: {node.Q}\n")
        return ''.join(output)

    def rank_children(self):
        ranked_children = list(range(self.num_legal_actions))
        ranked_children.sort(key=lambda i: (
            self.child_N_compressed[i], self.child_action_score_compressed[i]), reverse=True)
        return ranked_children

    def describe(self):
        ranked_children = self.rank_children()
        soft_n = self.child_N_compressed / max(1, sum(self.child_N_compressed))
        prior = self.child_prior_compressed
        p_delta = soft_n - prior
        p_rel = np.divide(p_delta, prior, out=np.zeros_like(
            p_delta), where=prior != 0)
        # Dump out some statistics
        output = []
        output.append("Q (player {}): {:.4f}\n".format(self.active_player_index, self.Q[self.active_player_index]))
        output.append(self.most_visited_path())
        output.append(
            "move : action    Q (curr)     U     P   P-Dir    N  soft-N  p-delta  p-rel")
        for i in ranked_children[:15]:
            if self.child_N_compressed[i] == 0:
                break
            output.append("\n{!s:4} : {: .3f} {: .3f} {:.3f} {:.3f} {:.3f} {:5d} {:.4f} {: .5f} {: .2f}".format(
                f"{self.game_dict['move_number']}: {self.fmove}",
                self.child_action_score_compressed[i],
                self.child_Q_compressed[i][self.active_player_index],
                self.child_U_compressed[i],
                self.child_prior_compressed[i],
                self.original_prior[i],
                int(self.child_N_compressed[i]),
                soft_n[i],
                p_delta[i],
                p_rel[i]))
        return ''.join(output)