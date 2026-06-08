from __future__ import annotations
import collections
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Union
import scipy
import torch
import logging
from rl18xx.game.engine.game.base import BaseGame
from rl18xx.agent.alphazero.action_mapper import ActionMapper
from rl18xx.agent.alphazero.config import SelfPlayConfig
from rl18xx.rust_adapter import RustGameAdapter
import time


@dataclass
class PlayoutTrace:
    """Per-playout diagnostic record (Phase 1 of MCTS improvements plan).

    Populated by ``MCTSNode.select_leaf`` (descent path + PW/forced-chain
    markers) and finalized by ``MCTSPlayer.tree_search`` after the leaf is
    evaluated (NN value, leaf Q, expansion flag, prior entropy). Lives only
    when ``SelfPlayConfig.trace`` is enabled.

    ``move_idx`` is the parent ``play_move`` index this trace was recorded
    under; ``action_path`` / ``pw_grandchild_path`` / ``forced_chain_lengths``
    are parallel arrays indexed by descent step from the root.
    """

    move_idx: int
    leaf_depth: int = 0
    action_path: list[int] = field(default_factory=list)
    pw_grandchild_path: list[bool] = field(default_factory=list)
    forced_chain_lengths: list[int] = field(default_factory=list)
    nn_value: Optional[np.ndarray] = None
    leaf_q_perspective: float = 0.0
    leaf_terminal: bool = False
    expansion_occurred: bool = False
    leaf_prior_entropy: float = 0.0

    def to_jsonable(self) -> dict:
        return {
            "move_idx": self.move_idx,
            "leaf_depth": self.leaf_depth,
            "action_path": list(self.action_path),
            "pw_grandchild_path": list(self.pw_grandchild_path),
            "forced_chain_lengths": list(self.forced_chain_lengths),
            "nn_value": None if self.nn_value is None else [float(v) for v in self.nn_value],
            "leaf_q_perspective": float(self.leaf_q_perspective),
            "leaf_terminal": bool(self.leaf_terminal),
            "expansion_occurred": bool(self.expansion_occurred),
            "leaf_prior_entropy": float(self.leaf_prior_entropy),
        }

# Flat policy size: computed once from the ActionMapper so it stays in sync
# with the action encoding (no hardcoded magic). ActionMapper is a Singleton,
# so this is the same instance MCTS reuses at runtime.
POLICY_SIZE = ActionMapper().action_encoding_size
VALUE_SIZE = 6  # Max-N: model emits 6 per-player values; entries [num_players:] are zero-padded.

# Continuous-price progressive-widening constants.
# Price grid snapping rules: Bid uses $5 ticks (matches the human ladder for
# auctions), every other price-bearing type uses $1 ticks. ``PRICE_GRID`` maps
# action-type names to snap step in raw dollars; missing types are treated as
# fixed-price (depot trains, market trains) and never reach the PW sampler.
PRICE_GRID = {
    "Bid": 5,
    "BuyTrain": 1,
    "BuyCompany": 1,
}

LOGGER = logging.getLogger(__name__)

# Module-level singletons (stateless, computed once)
_cached_edge_index = None
_cached_edge_attrs = None
_cached_action_mapper = None


def _get_action_mapper() -> ActionMapper:
    global _cached_action_mapper
    if _cached_action_mapper is None:
        _cached_action_mapper = ActionMapper()
    return _cached_action_mapper


def _rust_encode(game: RustGameAdapter) -> tuple:
    """Encode a RustGameAdapter using the Rust-native encoder.

    Returns (game_state, node_features, edge_index, edge_attrs, round_type_idx,
    active_player_idx, rotation). After canonicalization ``active_player_idx`` is
    always 0; ``rotation`` is the original absolute active player index.
    """
    global _cached_edge_index, _cached_edge_attrs

    from rl18xx.agent.alphazero.encoder import ROUND_TYPE_MAP, Encoder_1830Graph

    encoder = Encoder_1830Graph()
    encoder.initialize(game)

    # Compute static edge index once (hex adjacency never changes)
    if _cached_edge_index is None:
        result = encoder.encode(game)
        _cached_edge_index, _cached_edge_attrs = result[2], result[3]

    round_name = game.round.__class__.__name__
    round_type_idx = ROUND_TYPE_MAP.get(round_name, 0)
    player_ids = sorted([p.id for p in game.players])
    player_id_to_idx = {pid: i for i, pid in enumerate(player_ids)}
    rotation = player_id_to_idx.get(game.active_players()[0].id, 0)

    gs_flat, nf_flat, enc_size, num_hexes, num_nf = game._game.encode_for_gnn()
    gs_np = np.asarray(gs_flat, dtype=np.float32)
    if rotation != 0:
        gs_np = encoder.canonicalize_perspective(gs_np, rotation)
    gs_tensor = torch.from_numpy(gs_np).unsqueeze(0)
    nf_tensor = torch.tensor(nf_flat, dtype=torch.float32).reshape(num_hexes, num_nf)
    return (gs_tensor, nf_tensor, _cached_edge_index, _cached_edge_attrs, round_type_idx, 0, rotation)


def _snap_price(price: float, action_type: str, price_min: int, price_max: int) -> int:
    """Snap a continuous price sample to the legal grid for ``action_type``.

    Returns the snapped integer price clamped to ``[price_min, price_max]``.
    Bids snap to $5 multiples (matching the auction ladder humans bid on);
    other price-bearing actions snap to $1 (the engine's smallest legal
    increment). Unknown types snap to $1 by default.
    """
    step = PRICE_GRID.get(action_type, 1)
    snapped = int(round(price / step) * step)
    if snapped < price_min:
        # Round up to first legal multiple >= price_min.
        rem = price_min % step
        snapped = price_min if rem == 0 else price_min + (step - rem)
    if snapped > price_max:
        # Round down to last legal multiple <= price_max.
        snapped = price_max - (price_max % step) if step > 1 else price_max
    # Final defensive clamp (handles edge cases where snapping ran past bounds).
    return int(max(min(snapped, price_max), price_min))


def sample_price_for_pw(
    price_mean: float,
    price_log_std: float,
    action_type: str,
    price_range: tuple,
    rng: Optional[np.random.Generator] = None,
) -> int:
    """Sample a snapped legal price from the head's truncated Normal.

    Building block for the MCTS continuous-price progressive-widening flow.
    Once the action mapper migration to ``FactoredActionHelper`` lands, the
    expansion code path can call this for each (action_type, entity) child
    whose ``price_range`` has min != max, snapping the sampled price to the
    legal grid via ``_snap_price``.

    Currently unused by the in-tree expansion code (the existing flat action
    mapper still enumerates discrete (action, price) tuples), but exposed so
    the FactoredActionHelper migration can drop it in without rebuilding
    MCTS from scratch. See docs/step1_review.md "Continuous-price action
    space via progressive widening".

    Args:
        price_mean:    Network's predicted ``μ`` for this (type, entity) slot.
        price_log_std: Network's predicted ``log σ`` for the same slot.
        action_type:   ``"Bid"`` / ``"BuyTrain"`` / ``"BuyCompany"``.
        price_range:   ``(min, max)`` inclusive legal range from
                       FactoredActionHelper.
        rng:           Optional numpy Generator; defaults to ``np.random``.

    Returns:
        Snapped integer price guaranteed to lie in ``[price_range[0],
        price_range[1]]`` on the appropriate grid step.
    """
    if rng is None:
        rng = np.random.default_rng()

    p_min, p_max = price_range
    if p_min == p_max:
        return int(p_min)

    sigma = float(np.exp(np.clip(price_log_std, -1.0, 8.5)))
    mu = float(price_mean)

    # Rejection-sample once or twice; if both fall outside, fall back to a
    # uniform sample on the legal range so PW always returns a usable child.
    # In practice the truncation tails are mild and rejection terminates fast.
    for _ in range(8):
        sample = rng.normal(mu, sigma)
        if p_min - sigma <= sample <= p_max + sigma:
            return _snap_price(sample, action_type, int(p_min), int(p_max))
    # Last-ditch: uniform sample on the snap grid inside the legal range.
    step = PRICE_GRID.get(action_type, 1)
    n_choices = max(1, (int(p_max) - int(p_min)) // step + 1)
    return int(p_min) + int(rng.integers(0, n_choices)) * step


def pw_target_children(visits: int, pw_c: float, pw_alpha: float, min_children: int = 1) -> int:
    """Number of price children a categorical PW node should have at ``visits`` visits.

    Standard progressive-widening schedule: ``k = ceil(pw_c * N^pw_alpha)``,
    floored at ``min_children`` so the first visit always has a sampled
    price to descend through. The MCTS expansion code compares the current
    child count against this target and samples a fresh price whenever
    ``len(price_children) < pw_target_children(...)``.
    """
    target = pw_c * (max(0, visits) ** pw_alpha)
    return max(int(min_children), int(math.ceil(target)))


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
        game_state: Union[BaseGame, RustGameAdapter],
        fmove: Optional[int] = None,
        parent: Optional[MCTSNode | DummyNode] = None,
        config: Optional[SelfPlayConfig] = None,
    ):
        if parent is None:
            parent = DummyNode()
        self.parent = parent
        self.fmove = fmove  # move index that led to this position
        # Cache the index into parent's compressed arrays to avoid repeated linear scans
        if fmove is not None and isinstance(parent, MCTSNode):
            self._parent_index = parent.legal_action_indices.index(fmove)
        else:
            self._parent_index = 0  # DummyNode case
        self.is_expanded = False
        self.losses_applied = 0  # number of virtual losses on this node

        self.depth = 0
        if isinstance(parent, MCTSNode):
            self.depth = parent.depth + 1

        self.config = config or SelfPlayConfig()
        self.game_object = game_state
        self.encoded_game_state = None  # lazy — computed on demand via ensure_encoded()

        # Action indices auto-applied between the parent's chosen action and this node's
        # state. Empty for ordinary nodes; populated when maybe_add_child collapses a
        # forced-move chain (sequence of states with exactly one legal action).
        self.forced_action_chain: list[int] = []

        self.action_mapper = _get_action_mapper()

        self.player_mapping = {p.id: i for i, p in enumerate(sorted(self.game_object.players, key=lambda x: x.id))}
        self.active_player_index = self.player_mapping[self.game_object.active_players()[0].id]

        t0 = time.perf_counter()
        # Factored enumeration: legal categorical slots + per-slot price ranges.
        # Price-bearing slots (Bid/BuyTrain/BuyCompany) carry ``(min, max)``
        # metadata that MCTS uses to sample a concrete price via the
        # network's continuous-price head + ``sample_price_for_pw``. Slots
        # with ``price_range[0] == price_range[1]`` are fixed-price (depot
        # trains, exchange trains) and skip the sampler entirely.
        (
            self.legal_action_indices,
            self.price_ranges_by_idx,
            self.action_types_by_idx,
        ) = self.action_mapper.get_legal_actions_factored(self.game_object)
        # Price chosen at expansion time for this node (if the parent's
        # picked child was a price-bearing slot). When non-``None`` this node
        # is a *price grandchild* of its categorical parent: its N/W live in
        # the parent's ``price_child_N`` / ``price_child_W`` dicts instead of
        # the categorical compressed arrays. ``maybe_add_child`` sets this
        # when materializing a price-bearing slot via the PW sampler.
        self.sampled_price: Optional[int] = None
        t1 = time.perf_counter()
        self.add_metric("MCTS/Node_ActionEnum_Duration", t1 - t0)
        self.num_legal_actions = len(self.legal_action_indices)
        self.child_N_compressed = np.zeros(self.num_legal_actions, dtype=np.float32)
        self.child_W_compressed = np.zeros([self.num_legal_actions, VALUE_SIZE], dtype=np.float32)
        self.original_prior_compressed = np.zeros(self.num_legal_actions, dtype=np.float32)
        self.child_prior_compressed = np.zeros(self.num_legal_actions, dtype=np.float32)

        # ``children`` only holds non-price-bearing categorical slots — one
        # ``MCTSNode`` per action index. Price-bearing slots (Bid / BuyTrain /
        # BuyCompany with a non-degenerate ``price_range``) are stored in
        # ``price_children`` as a two-level dict: action_index → snapped_price
        # → grandchild. This gives MCTS a categorical → continuous-price tree
        # structure so progressive widening can materialize multiple price
        # alternatives under a single categorical slot (Task #28 design).
        #
        # The categorical slot's aggregate visit / win-loss stats in
        # ``child_N_compressed`` / ``child_W_compressed`` are still updated
        # during backup (they sum across all grandchildren), so the existing
        # PUCT formula at the categorical level is unchanged.
        self.children: dict[int, MCTSNode] = {}
        self.price_children: dict[int, dict[int, MCTSNode]] = {}
        # Per-grandchild N/W. Indexed first by parent action_index, then by
        # snapped price. Grandchildren read their own N/W from these arrays
        # (see ``MCTSNode.N`` / ``MCTSNode.W`` properties) rather than from
        # the parent's compressed arrays.
        self.price_child_N: dict[int, dict[int, float]] = {}
        self.price_child_W: dict[int, dict[int, np.ndarray]] = {}
        # Cached rng for PW price sampling — reusing one generator is
        # negligibly faster than ``np.random.default_rng()`` per call and
        # keeps any future deterministic-seed test wiring simple.
        self._pw_rng = np.random.default_rng()
        self.add_metric("MCTS/Depth", self.depth)

        if self.is_done():
            self.game_object.end_game()

    def add_metric(self, name, value):
        if self.config.metrics is None:
            return
        self.config.metrics.add_scalar(name, value, self.config.global_step, self.config.game_idx_in_iteration)

    def ensure_encoded(self):
        """Lazily encode the game state on first access. Most MCTS nodes are never
        selected as leaves, so deferring encoding avoids wasted computation."""
        if self.encoded_game_state is not None:
            return
        t0 = time.perf_counter()
        self.encoded_game_state = _rust_encode(self.game_object)
        t1 = time.perf_counter()
        self.add_metric("MCTS/Node_Encode_Duration", t1 - t0)

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
        # U(s, a) = c_puct * P(s, a) * sqrt(N(s)) / (1 + N(s, a))
        # Per-round-type c_puct_init (Phase 6.2)
        round_name = self.game_object.round.__class__.__name__
        c_puct_init = self.config.c_puct_by_round.get(round_name, self.config.c_puct_init)
        c_puct = 2.0 * (
            math.log((1.0 + self.N + self.config.c_puct_base) / self.config.c_puct_base) + c_puct_init
        )
        p_s_a = self.child_prior_compressed
        n_s = max(1, self.N - 1)
        n_s_a = self.child_N_compressed
        return c_puct * p_s_a * math.sqrt(n_s) / (1 + n_s_a)

    @property
    def Q(self):
        return self.W / (1.0 + self.N)

    @property
    def _is_price_grandchild(self) -> bool:
        """True iff this node was expanded as a PW price-grandchild of its
        categorical parent (i.e., ``sampled_price`` is set AND the parent
        registers a price-children dict for ``fmove``).

        Price grandchildren live in ``parent.price_children[fmove][price]``
        and their N/W are stored in the parent's per-price dicts rather than
        the categorical compressed arrays — letting multiple grandchildren
        coexist under the same categorical slot with independent statistics.
        """
        if self.sampled_price is None or not isinstance(self.parent, MCTSNode):
            return False
        if self.fmove is None:
            return False
        # Only treat as a grandchild if the parent actually has a price-range
        # entry for this slot. Fixed-price slots (depot trains) set
        # ``sampled_price`` for bookkeeping but live in the regular categorical
        # ``children`` dict, not ``price_children``.
        return self.fmove in self.parent.price_children

    @property
    def N(self):
        if self._is_price_grandchild:
            return self.parent.price_child_N[self.fmove].get(self.sampled_price, 0.0)
        return self.parent.child_N_compressed[self._parent_index]

    @N.setter
    def N(self, value):
        if self._is_price_grandchild:
            self.parent.price_child_N[self.fmove][self.sampled_price] = float(value)
            return
        self.parent.child_N_compressed[self._parent_index] = value

    @property
    def W(self):
        if self._is_price_grandchild:
            return self.parent.price_child_W[self.fmove][self.sampled_price]
        return self.parent.child_W_compressed[self._parent_index]

    @W.setter
    def W(self, value):
        if self._is_price_grandchild:
            self.parent.price_child_W[self.fmove][self.sampled_price] = value
            return
        self.parent.child_W_compressed[self._parent_index] = value

    @property
    def Q_perspective(self):
        """Return value of position, from perspective of player to play."""
        return self.Q[self.active_player_index]

    def select_leaf(self, trace: Optional[PlayoutTrace] = None):
        start_time = time.time()
        current = self
        num_added = 0
        while current.is_expanded:
            scores = current.child_action_score_compressed
            # Progressive widening (Phase 6.3): only apply when many legal actions
            if current.num_legal_actions > 20:
                k = max(1, int(current.config.pw_c * (current.N ** current.config.pw_alpha)))
                if k < current.num_legal_actions:
                    top_k_indices = np.argpartition(current.child_prior_compressed, -k)[-k:]
                    masked_scores = np.full_like(scores, -np.inf)
                    masked_scores[top_k_indices] = scores[top_k_indices]
                    best_compressed_idx = np.argmax(masked_scores)
                else:
                    best_compressed_idx = np.argmax(scores)
            else:
                best_compressed_idx = np.argmax(scores)
            best_move = current.legal_action_indices[best_compressed_idx]
            # If ``best_move`` is a price-bearing slot, the categorical "child"
            # is the price-children dict — descend through the PW selector
            # which either grows a new grandchild or PUCT-selects an existing
            # one. Otherwise the standard categorical descent applies.
            price_range = current.price_ranges_by_idx.get(best_move)
            is_pw_descent = price_range is not None and price_range[0] != price_range[1]
            if is_pw_descent:
                current = current._select_or_expand_price_child(best_move, price_range)
            else:
                current = current.maybe_add_child(best_move)
            if trace is not None:
                trace.action_path.append(int(best_move))
                trace.pw_grandchild_path.append(bool(is_pw_descent))
                trace.forced_chain_lengths.append(len(current.forced_action_chain))
            num_added += 1
        if trace is not None:
            trace.leaf_depth = num_added
        end_time = time.time()
        self.add_metric("MCTS/Select_Leaf_Time", end_time - start_time)
        self.add_metric("MCTS/Select_Leaf_Path_Length", num_added)
        return current

    def _select_or_expand_price_child(self, action_index: int, price_range: tuple) -> "MCTSNode":
        """PW descent for a price-bearing categorical slot.

        Implements the two-level (categorical → continuous-price) MCTS tree:
        once the outer PUCT selects ``action_index``, this either grows a new
        grandchild via ``sample_price_for_pw`` (when the slot's grandchild
        count is below the PW target ``k = ceil(pw_c * N^pw_alpha)``) or
        PUCT-selects among existing grandchildren using their independent
        per-price stats. The chosen grandchild is the returned node — the
        outer ``select_leaf`` loop continues its descent from there.

        ``action_index``: the categorical flat-policy index.
        ``price_range``:  ``(min, max)`` from ``price_ranges_by_idx`` (already
                          known to be non-degenerate by the caller).
        """
        action_type = self.action_types_by_idx.get(action_index, "")
        slot_visits = int(self.child_N_compressed[self._index_for(action_index)])
        existing = self.price_children.get(action_index, {})
        target = pw_target_children(
            slot_visits,
            self.config.pw_c,
            self.config.pw_alpha,
            min_children=self.config.min_price_children,
        )

        if len(existing) < target:
            # Sample a fresh price and either expand a new grandchild or — if
            # the snapped price collides with an existing grandchild — return
            # that one (it will receive a fresh visit on the next backup).
            sampled_price = self._sample_price_for_slot(action_index, action_type, price_range)
            if sampled_price in existing:
                return existing[sampled_price]
            return self.maybe_add_child(action_index, price=sampled_price)

        # PW cap reached: PUCT among existing grandchildren using per-price
        # stats. The scoring formula mirrors ``child_U`` but with grandchild N
        # values, sharing the categorical prior P (the head doesn't yet
        # produce per-price priors).
        round_name = self.game_object.round.__class__.__name__
        c_puct_init = self.config.c_puct_by_round.get(round_name, self.config.c_puct_init)
        c_puct = 2.0 * (
            math.log((1.0 + slot_visits + self.config.c_puct_base) / self.config.c_puct_base) + c_puct_init
        )
        n_s = max(1, slot_visits - 1)
        # Categorical-slot prior, divided among grandchildren so PUCT's
        # exploration mass scales with prior, not just grandchild count.
        idx = self._index_for(action_index)
        slot_prior = float(self.child_prior_compressed[idx])
        per_grandchild_prior = slot_prior / max(1, len(existing))

        best_score = -np.inf
        best_grandchild = None
        for grandchild in existing.values():
            n_sa = grandchild.N
            q_sa = (grandchild.W[self.active_player_index] / (1 + n_sa)) if n_sa > 0 else 0.0
            u_sa = c_puct * per_grandchild_prior * math.sqrt(n_s) / (1 + n_sa)
            score = q_sa + u_sa
            if score > best_score:
                best_score = score
                best_grandchild = grandchild
        return best_grandchild

    def _index_for(self, action_index: int) -> int:
        """Translate a flat action index into its compressed-array slot.

        Mirror of the ``parent.legal_action_indices.index(...)`` lookup used
        by child construction, but used from the parent's perspective when
        we already know the action_index belongs to ``self``.
        """
        return self.legal_action_indices.index(action_index)

    def maybe_add_child(self, action_index: int, price: Optional[int] = None):
        """Adds child node for action_index if it doesn't already exist, and returns it.

        For non-price-bearing slots (and fixed-price slots like depot trains) the
        child is a single ``MCTSNode`` stored under ``self.children[action_index]``.

        For price-bearing slots with a non-degenerate ``price_range``, the child
        is materialized as a *price grandchild* under
        ``self.price_children[action_index][snapped_price]``. When ``price`` is
        ``None``, a price is sampled via ``_sample_price_for_slot`` (the
        network's ``ContinuousPriceHead`` posterior); when ``price`` is given
        (e.g., by ``select_leaf`` after PW deciding to expand), it's used
        directly. If a grandchild already exists at the snapped price, it is
        returned without creating a duplicate.

        Forced-move chaining: after applying ``action_index``, if the resulting state has
        exactly one legal action, that action is automatically applied; this repeats until
        the game has either >1 legal actions or is terminal. Visit counts and Q values
        accumulate only on the chain-terminating (real-decision) node. The intermediate
        action indices are recorded on ``child.forced_action_chain`` for debugging and
        traceability. ``MCTSPlayer.played_actions`` is unaffected — it still records only
        the active player's chosen action at each non-forced node, since the cascade
        actions are deterministic functions of the engine state and don't generate
        training examples of their own.
        """
        start_time = time.time()

        price_range = self.price_ranges_by_idx.get(action_index)
        is_pw_slot = price_range is not None and price_range[0] != price_range[1]

        # Fast path: existing child / grandchild.
        if is_pw_slot:
            action_type = self.action_types_by_idx.get(action_index, "")
            if price is None:
                price = self._sample_price_for_slot(action_index, action_type, price_range)
            else:
                price = _snap_price(int(price), action_type, int(price_range[0]), int(price_range[1]))
            slot_grandchildren = self.price_children.get(action_index)
            if slot_grandchildren is not None and price in slot_grandchildren:
                self.add_metric("MCTS/Maybe_Add_Child_Overall_Duration", time.time() - start_time)
                return slot_grandchildren[price]
        else:
            if action_index in self.children:
                self.add_metric("MCTS/Maybe_Add_Child_Overall_Duration", time.time() - start_time)
                return self.children[action_index]

        clone_start_time = time.time()
        try:
            new_position = self.game_object.pickle_clone()
        except Exception as e:
            LOGGER.error(f"Error cloning game_object in MCTSNode (fmove={self.fmove}): {e}", exc_info=True)
            raise e
        clone_duration = time.time() - clone_start_time

        process_action_start_time = time.time()
        # Price-bearing slot: use the (now-resolved) ``price``; fixed-price
        # slots use the engine-determined price; categorical slots use the
        # default-price mapper.
        sampled_price: Optional[int] = None
        try:
            if is_pw_slot:
                sampled_price = int(price)
                action_to_take = self.action_mapper.map_index_to_action_with_price(
                    action_index, new_position, sampled_price
                )
            elif price_range is not None:
                # Fixed-price slot (depot trains, exchange trains): use
                # the engine-determined price directly.
                sampled_price = int(price_range[0])
                action_to_take = self.action_mapper.map_index_to_action_with_price(
                    action_index, new_position, sampled_price
                )
            else:
                action_to_take = self.action_mapper.map_index_to_action(action_index, new_position)
            new_position.process_action(action_to_take)
        except Exception as e:
            LOGGER.error(
                f"Error processing action in maybe_add_child. Parent fmove: {self.fmove}, "
                f"Action index: {action_index}, sampled_price: {sampled_price}",
                exc_info=True,
            )
            LOGGER.error(f"Parent game actions: {self.game_object.raw_actions}")
            raise e

        # Auto-advance through forced moves: collapse any chain of states that have
        # exactly one legal action, so MCTS doesn't waste tree depth / visits on them.
        forced_chain: list[int] = []
        forced_chain_start = time.time()
        while not (new_position.finished or new_position.move_number >= self.config.max_game_length):
            legal_indices, forced_price_ranges, forced_action_types = (
                self.action_mapper.get_legal_actions_factored(new_position)
            )
            if len(legal_indices) != 1:
                break
            forced_action_index = legal_indices[0]
            forced_price_range = forced_price_ranges.get(forced_action_index)
            try:
                if forced_price_range is not None and forced_price_range[0] != forced_price_range[1]:
                    forced_action_type = forced_action_types.get(forced_action_index, "")
                    forced_price = self._sample_price_for_slot(
                        forced_action_index, forced_action_type, forced_price_range
                    )
                    forced_action = self.action_mapper.map_index_to_action_with_price(
                        forced_action_index, new_position, forced_price
                    )
                elif forced_price_range is not None:
                    forced_action = self.action_mapper.map_index_to_action_with_price(
                        forced_action_index, new_position, int(forced_price_range[0])
                    )
                else:
                    forced_action = self.action_mapper.map_index_to_action(forced_action_index, new_position)
                new_position.process_action(forced_action)
            except Exception as e:
                LOGGER.error(
                    f"Error processing forced action during chain collapse. Parent fmove: {self.fmove}, "
                    f"Originating action_index: {action_index}, Forced action index: {forced_action_index}",
                    exc_info=True,
                )
                raise e
            forced_chain.append(forced_action_index)
        forced_chain_duration = time.time() - forced_chain_start

        # Register grandchild stats *before* constructing the MCTSNode so the
        # child's N/W setters (invoked transitively by add_metric/init) read
        # the new per-price slots rather than the categorical compressed
        # arrays. ``sampled_price`` is attached *after* construction; the
        # ``_is_price_grandchild`` check guards against partial init.
        if is_pw_slot:
            self.price_children.setdefault(action_index, {})
            self.price_child_N.setdefault(action_index, {})
            self.price_child_W.setdefault(action_index, {})
            self.price_child_N[action_index][sampled_price] = 0.0
            self.price_child_W[action_index][sampled_price] = np.zeros(VALUE_SIZE, dtype=np.float32)

        child = MCTSNode(new_position, fmove=action_index, parent=self, config=self.config)
        child.forced_action_chain = forced_chain
        child.sampled_price = sampled_price
        if is_pw_slot:
            self.price_children[action_index][sampled_price] = child
        else:
            self.children[action_index] = child
        process_action_duration = time.time() - process_action_start_time

        self.add_metric("MCTS/Maybe_Add_Child_Clone_Duration", clone_duration)
        self.add_metric("MCTS/Maybe_Add_Child_Process_Action_Duration", process_action_duration)
        self.add_metric("MCTS/Maybe_Add_Child_Forced_Chain_Duration", forced_chain_duration)
        self.add_metric("MCTS/Forced_Chain_Length", len(forced_chain))

        overall_call_duration = time.time() - start_time
        self.add_metric("MCTS/Maybe_Add_Child_Overall_Duration", overall_call_duration)
        return child

    def _sample_price_for_slot(
        self,
        action_index: int,
        action_type: str,
        price_range: tuple,
    ) -> int:
        """Sample a snapped legal price for a price-bearing categorical slot.

        Pulls ``(μ, log σ)`` from the network's ``ContinuousPriceHead`` if
        the parent leaf has them attached (set by ``incorporate_results``),
        falling back to a wide-Normal prior centered at the midpoint of the
        legal range so MCTS still produces usable children before the
        network is fully wired (or for the GNN model which has no price
        head). The sampled price is snapped to the action type's price grid
        (``PRICE_GRID``) and guaranteed to lie in ``[price_range[0],
        price_range[1]]``.
        """
        p_min, p_max = price_range
        if p_min == p_max:
            return int(p_min)

        # Default prior: uniform-ish Normal centered at the midpoint, scaled
        # to cover the legal range comfortably. Used when no price head
        # output is available (e.g., GNN model or initial expansion before
        # incorporate_results runs).
        midpoint = (p_min + p_max) / 2.0
        spread = max((p_max - p_min) / 4.0, 1.0)
        price_mean = float(midpoint)
        price_log_std = float(np.log(spread))

        price_components = getattr(self, "price_components", None)
        if price_components is not None:
            slot_index_map = price_components.get("slot_index")
            entity_key = self._price_head_entity_key(action_index)
            if slot_index_map is not None and entity_key is not None:
                slot = slot_index_map.get((action_type, entity_key))
                if slot is not None:
                    means = price_components.get("price_mean")
                    log_stds = price_components.get("price_log_std")
                    if means is not None and log_stds is not None:
                        # means/log_stds are torch tensors (1D, num_slots) sliced
                        # to this leaf during incorporate_results.
                        price_mean = float(means[slot])
                        price_log_std = float(log_stds[slot])

        return sample_price_for_pw(price_mean, price_log_std, action_type, price_range)

    def _price_head_entity_key(self, action_index: int):
        """Resolve the (action_type, entity_key) the ContinuousPriceHead uses
        as its slot key for ``action_index``.

        The price head slot layout (see ``model_transformer.ContinuousPriceHead``):
          - ``Bid``        → ``(company_sym,)``
          - ``BuyTrain``   → ``(corp_sym, train_type)`` (corp-to-corp only)
          - ``BuyCompany`` → ``(company_sym,)``

        Returns ``None`` if the flat ``action_index`` doesn't fall in a
        head-modelled slot (e.g., depot trains, market discarded trains —
        these are fixed-price and never sampled).
        """
        offsets = self.action_mapper.action_offsets
        bid_start = offsets["Bid"]
        bid_end = bid_start + len(self.action_mapper.company_offsets)
        if bid_start <= action_index < bid_end:
            companies = list(self.action_mapper.company_offsets.keys())
            return (companies[action_index - bid_start],)

        buy_train_start = offsets["BuyTrain"]
        # First slot is depot; next 6 are market discarded — both fixed-price.
        cross_corp_start = buy_train_start + 1 + len(self.action_mapper.train_type_offsets)
        cross_corp_end = cross_corp_start + (
            len(self.action_mapper.corporation_offsets)
            * len(self.action_mapper.train_type_offsets)
            * len(self.action_mapper.train_price_offsets)
        )
        if cross_corp_start <= action_index < cross_corp_end:
            rel = action_index - cross_corp_start
            per_corp = len(self.action_mapper.train_type_offsets) * len(self.action_mapper.train_price_offsets)
            corp_idx = rel // per_corp
            train_idx = (rel % per_corp) // len(self.action_mapper.train_price_offsets)
            corp_sym = list(self.action_mapper.corporation_offsets.keys())[corp_idx]
            train_type = list(self.action_mapper.train_type_offsets.keys())[train_idx]
            return (corp_sym, train_type)

        buy_company_start = offsets["BuyCompany"]
        buy_company_end = buy_company_start + (
            len(self.action_mapper.company_offsets) * len(self.action_mapper.buy_company_price_offsets)
        )
        if buy_company_start <= action_index < buy_company_end:
            rel = action_index - buy_company_start
            company_idx = rel // len(self.action_mapper.buy_company_price_offsets)
            companies = list(self.action_mapper.company_offsets.keys())
            return (companies[company_idx],)

        return None

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
        # Grandchildren also need to subtract from the categorical slot's W
        # so PUCT at the categorical level sees the virtual loss too.
        if self._is_price_grandchild:
            self.parent.child_W_compressed[self._parent_index, prev_player] -= 1
        self.parent.add_virtual_loss(up_to)

    def revert_virtual_loss(self, up_to):
        if self.parent is None or self is up_to:
            return
        self.losses_applied -= 1
        prev_player = self.parent.active_player_index
        self.W[prev_player] += 1
        if self._is_price_grandchild:
            self.parent.child_W_compressed[self._parent_index, prev_player] += 1
        self.parent.revert_virtual_loss(up_to)

    def _unrotate_value(self, value: np.ndarray) -> np.ndarray:
        """Translate a network-produced value vector from canonical to absolute order.

        The encoder rotates player-indexed sections so the active player sits at
        slot 0; the network learns and emits values in that canonical frame. MCTS
        backs values up into ``W`` arrays indexed by **absolute** player position
        (different ancestors in the tree have different active players, so the W
        array can't be in any single canonical frame). Inverting the encoder's
        rotation here gives us absolute-order values.

        Rotation math:
            encoder did:  canonical[i] = absolute[(i + rotation) mod N]
            invert with:  absolute[i]  = canonical[(i - rotation) mod N]
                        = np.roll(canonical, shift=+rotation)
        """
        if self.encoded_game_state is None or len(self.encoded_game_state) < 7:
            # No rotation recorded — assume the value is already in absolute order
            # (covers test stubs and any legacy callers that bypass the encoder).
            return value
        rotation = self.encoded_game_state[6]
        if rotation == 0:
            return value
        return np.roll(value, shift=int(rotation))

    def incorporate_results(self, move_probabilities, value, up_to, price_components=None):
        """Incorporate the network's prediction at this leaf.

        ``price_components`` (optional) carries the per-leaf slice of the
        ``ContinuousPriceHead`` outputs:

            {
                "price_mean":    1D tensor (num_slots,)
                "price_log_std": 1D tensor (num_slots,)
                "slot_index":    {(action_type, entity_key) -> int}
            }

        When present, ``maybe_add_child`` reads from it to draw legal prices
        for price-bearing categorical slots via ``sample_price_for_pw``.
        ``None`` (e.g., the GNN model with no price head) falls back to a
        uniform-Normal default prior on the legal range.
        """
        if price_components is not None:
            self.price_components = price_components
        assert move_probabilities.shape == (
            POLICY_SIZE,
        ), f"move_probabilities.shape: {move_probabilities.shape} must be ({POLICY_SIZE},)"
        assert value.shape == (VALUE_SIZE,), f"value.shape: {value.shape} must be ({VALUE_SIZE},)"
        # A finished game should not be going through this code path - should
        # directly call backup_value() on the result of the game.
        assert not self.game_object.finished

        # Move to numpy (needed for backup either way)
        if isinstance(value, torch.Tensor):
            value = value.cpu().numpy()

        # Network values come back in **canonical** order (slot 0 = active player at
        # this leaf) because the encoder canonicalizes the game state. MCTS stores
        # W in **absolute** player order so different ancestors (with different
        # active players) can index it consistently. Unrotate here.
        value = self._unrotate_value(value)

        # If a node was picked multiple times (despite vlosses), we shouldn't
        # re-expand it — but we still need to back up the network's value so the
        # second visit isn't lost.
        if self.is_expanded:
            self.backup_value(value, up_to=up_to)
            return
        self.is_expanded = True

        if isinstance(move_probabilities, torch.Tensor):
            move_probabilities = move_probabilities.cpu().numpy()

        # Extract only legal action probabilities (avoids allocating full 26,535-elem arrays)
        legal_probs = move_probabilities[self.legal_action_indices]
        scale = legal_probs.sum()
        if scale > 0:
            legal_probs /= scale

        self.original_prior_compressed = legal_probs.copy()
        self.child_prior_compressed = legal_probs
        # Standard AlphaZero: initialize child W to zeros rather than seeding
        # with the parent's value. Seeding biases Q estimates toward the parent's
        # evaluation and can prevent proper exploration of moves that significantly
        # change the position. The exploration term (child_U) already handles the
        # explore-vs-exploit tradeoff via the prior policy and visit counts.
        self.child_W_compressed = np.zeros((self.num_legal_actions, VALUE_SIZE))
        self.backup_value(value, up_to=up_to)

    def backup_value(self, value, up_to):
        """Propagates a value estimation up to the root node with depth discounting.

        Args:
            value: the value to be propagated (discounted at each level)
            up_to: the node to propagate until.

        When this node is a PW *price grandchild*, the per-price stats
        (``parent.price_child_N`` / ``parent.price_child_W``) hold the
        N/W; we additionally accumulate visits into the categorical slot's
        ``child_N_compressed`` / ``child_W_compressed`` so PUCT at the
        categorical-selection level still sees the aggregate over all
        grandchildren.
        """
        self.N += 1
        self.W += value
        if self._is_price_grandchild:
            # Mirror visit / value into the categorical slot so categorical-
            # level PUCT integrates over all price children. This avoids
            # treating each grandchild as a separate top-level action while
            # still letting PW grow distinct grandchildren under one slot.
            self.parent.child_N_compressed[self._parent_index] += 1
            self.parent.child_W_compressed[self._parent_index] += value
        if self.parent is None or self is up_to:
            return
        self.parent.backup_value(value * self.config.backup_discount, up_to)

    def is_done(self):
        return self.game_object.finished or self.game_object.move_number >= self.config.max_game_length

    def game_result(self) -> Optional[np.ndarray]:
        """Per-player value vector emitted at game end.

        Returns a length-``VALUE_SIZE`` array that serves two roles:

        1. **MCTS backup**: backed up through the tree via ``backup_value``
           and accumulated into ``child_W`` for the win-loss head's PUCT
           signal. With ``use_score_values=True`` (the default) this is the
           normalized net-worth fractions; with ``use_score_values=False``
           it falls back to the legacy {-1, 0, +1} win/loss vector.

        2. **Training data**: also written to the training tuple's ``value``
           slot in ``self_play.extract_data``. The KataGo-style dual value
           head reads this stored value at training time and derives both
           the share-of-winners target (win-loss head) and the normalized
           net-worth target (score head) from it via
           ``train._derive_dual_value_targets``. The score head therefore
           gets a real continuous signal when ``use_score_values=True`` and
           a degenerate (== win/loss) signal when it's ``False``.
        """
        if not self.is_done():
            LOGGER.warning(f"Getting game result for unfinished game (node fmove: {self.fmove}).")

        result = self.game_object.result()
        num_players = len(result)

        if self.config.use_score_values:
            # Phase 6.4: normalized score fractions for richer gradient signal.
            # Slot order matches ``player_mapping`` (sorted player ids) so it
            # lines up with ``active_player_index`` and the encoder's player
            # feature layout. Phantom slots [num_players:] stay 0.0.
            value = np.zeros(VALUE_SIZE, dtype=np.float32)
            scores = np.array(
                [result[pid] for pid in sorted(result.keys())],
                dtype=np.float32,
            )
            total = scores.sum()
            if total > 0:
                value[:num_players] = scores / total
            else:
                # Equal share among real players; phantom slots stay 0.
                value[:num_players] = 1.0 / num_players
            return value

        # Legacy win/loss encoding. Phantom slots stay at -1.0 (not 0.0) so
        # ``train._derive_dual_value_targets``'s ``value > -0.5`` winner check
        # excludes them — at 0.0 they would be misread as tied winners and
        # bleed mass into the win/loss target.
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
            LOGGER.warning(
                f"game_result_string: Game is not finished (node fmove: {self.fmove}). Returning empty string."
            )
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
        if self.original_prior_compressed is None:
            return
        if self.num_legal_actions == 0:
            return
        w = self.config.dirichlet_noise_weight
        if w <= 0:
            self.child_prior_compressed = self.original_prior_compressed.copy()
            return
        concentration = getattr(self.config, "dirichlet_noise_concentration", 10.0)
        alpha_value = concentration / max(1, self.num_legal_actions)
        noise = np.random.dirichlet([alpha_value] * self.num_legal_actions)
        self.child_prior_compressed = self.original_prior_compressed * (1 - w) + noise * w

    def children_as_pi(self, temperature: float = 1.0) -> np.ndarray:
        """Return visit-count policy vector, optionally with temperature scaling.

        When temperature=1.0, returns visit counts normalized to a probability distribution.
        When temperature<1.0, sharpens toward the most-visited action (temperature→0 = argmax).
        When temperature>1.0, flattens (more exploratory).

        Returns a full-size (POLICY_SIZE,) array for training data compatibility.
        """
        if temperature < 1e-8:
            # Temperature → 0: deterministic argmax
            probs = np.zeros_like(self.child_N, dtype=np.float64)
            probs[np.argmax(self.child_N)] = 1.0
            return probs
        probs = self.child_N.astype(np.float64)
        if temperature != 1.0:
            probs = probs ** (1.0 / temperature)
        sum_probs = np.sum(probs)
        if sum_probs == 0:
            # No visits — uniform over legal actions to avoid NaN in training
            probs = np.zeros(POLICY_SIZE, dtype=np.float64)
            probs[self.legal_action_indices] = 1.0 / self.num_legal_actions
            return probs
        return probs / sum_probs

    def best_child(self) -> int:
        # Pick by visit count, tie-break with action score. Works on compressed arrays.
        scores = self.child_N_compressed + self.child_action_score_compressed / 10000
        return self.legal_action_indices[np.argmax(scores)]

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
            parent_move_number = (
                node_on_path.parent.game_object.move_number if isinstance(node_on_path.parent, MCTSNode) else "root"
            )
            output.append(f"{parent_move_number}: {node_on_path.fmove} (N={node_on_path.N:.0f}) ==> ")

        if path_nodes:
            final_node_on_path = path_nodes[-1]
            output.append(
                f"Q: {final_node_on_path.Q_perspective:.3f} (Player {final_node_on_path.active_player_index}) N={final_node_on_path.N:.0f}\n"
            )
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
        output.append(
            "idx  | Action (from parent) | Q (curr) |    U    |   P(a|s) | P_orig  |    N   | soft-N | p-delta | p-rel"
        )

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
