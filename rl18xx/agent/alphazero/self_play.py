from rl18xx.game.engine.game import BaseGame
from rl18xx.rust_adapter import RustGameAdapter
import random
import os
import socket
import time
from torch.utils.tensorboard import SummaryWriter
import rl18xx.agent.alphazero.mcts as mcts
from rl18xx.agent.alphazero.model_transformer import AlphaZeroTransformerModel
from rl18xx.agent.alphazero.config import SelfPlayConfig, ModelTransformerConfig
from rl18xx.agent.alphazero.checkpointer import get_latest_model
from rl18xx.agent.alphazero.dataset import TrainingExampleProcessor
from rl18xx.agent.alphazero.action_mapper import ActionMapper
from rl18xx.agent.agent import Agent
from rl18xx.shared.atomic_io import atomic_write_json
import numpy as np
from typing import Optional, List, Tuple, Generator, Union
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


def _slice_price_components(batched: Optional[dict], leaf_index: int) -> Optional[dict]:
    """Slice the model's batched ``last_price_components`` dict for a single leaf.

    The transformer model emits ``price_mean`` / ``price_log_std`` as ``(B,
    num_slots)`` tensors during ``forward()``. MCTS needs them per-leaf (1D,
    ``num_slots``) so it can read a slot's ``(μ, log σ)`` for PW price
    sampling. Returns ``None`` if the model doesn't emit price components
    (e.g., the GNN model).
    """
    if batched is None:
        return None
    means = batched.get("price_mean")
    log_stds = batched.get("price_log_std")
    if means is None or log_stds is None:
        return None
    # Detach + move to CPU once at the slice boundary; downstream MCTS reads
    # scalar values via ``float(tensor[slot])`` which is safe on CPU only.
    return {
        "price_mean": means[leaf_index].detach().cpu(),
        "price_log_std": log_stds[leaf_index].detach().cpu(),
        "slot_index": batched.get("slot_index"),
        "num_slots": batched.get("num_slots"),
    }


def _get_autocast_device() -> str | None:
    """Return the device string for torch.amp.autocast, or None if unavailable.

    Note: MPS autocast (FP16) is currently slower than FP32 for small models
    due to type-casting overhead, so we only enable it for CUDA.
    """
    if torch.cuda.is_available():
        return "cuda"
    return None


def _compute_net_worth(game) -> dict:
    """Return {player_id: net_worth} for each player.

    net_worth = cash + sum(share_price * shares_owned) + face value of owned private companies.
    Privates' purchase cost is sunk; their ongoing revenue is captured in cash.

    Prefers the engine's canonical scoring (``game.result()``), which both the Python
    and Rust engines compute identically. Falls back to per-player accessors if needed.
    The same formula is used for natural game-end (bank-broken / train-exhausted) and
    for truncated games, so targets are comparable.
    """
    # Fast path: engine exposes a result() that already sums cash + shares + company values.
    if hasattr(game, "result"):
        try:
            result = game.result()
            if result:
                return dict(result)
        except Exception as e:
            LOGGER.warning(f"_compute_net_worth: game.result() failed, falling back: {e}")

    # Fallback: compute from primitives. Mirrors Player.value semantics in both engines.
    net_worth = {}
    for player in game.players:
        nw = getattr(player, "cash", 0)
        # Shares: prefer player.value if available; otherwise iterate corporations.
        try:
            player_value = player.value
            if callable(player_value):
                player_value = player_value(game.corporations)
            nw = int(player_value)
        except Exception:
            for corp in game.corporations:
                if not getattr(corp, "ipoed", False):
                    continue
                sp = getattr(corp, "share_price", None)
                price = getattr(sp, "price", 0) if sp is not None else 0
                # Prefer the engine's own per-player percentage if exposed.
                if hasattr(player, "percent_of"):
                    try:
                        percent = player.percent_of(corp)
                    except Exception:
                        percent = 0
                    nw += (percent * price) // 10
                else:
                    # Last-resort: count shares directly.
                    shares = []
                    if hasattr(player, "shares_by_corporation"):
                        sbc = player.shares_by_corporation
                        sbc = sbc() if callable(sbc) else sbc
                        shares = sbc.get(corp, []) if hasattr(sbc, "get") else []
                    nw += price * len(shares)
        # Add face value of owned private companies (cash equivalent at game-end).
        try:
            companies = getattr(player, "companies", []) or []
            nw += sum(int(getattr(c, "value", 0) or 0) for c in companies)
        except Exception:
            pass
        net_worth[player.id] = int(nw)
    return net_worth


class MCTSPlayer(Agent):
    def __init__(self, config: SelfPlayConfig):
        self.config = config
        self.network = config.network
        # Cumulative timing accumulators (seconds); populated during tree_search.
        self.cumulative_inference_time = 0.0
        self.cumulative_encoding_time = 0.0
        self.cumulative_leaf_selection_time = 0.0
        self.cumulative_backup_time = 0.0
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
        """Build a fresh game using the player-count distribution on the config.

        Variable player count is supported via
        ``SelfPlayHyperparams.player_count_distribution``. Each new game samples
        a player count (2..6); a single self-play session always uses one
        player count throughout — the count is sampled once when the root game
        is first constructed and ``initialize_game`` reuses that count for the
        replay-game built by ``extract_data``.
        """
        from engine_rs import BaseGame as RustGame
        num_players = self._sample_num_players()
        # Cache so extract_data's replay game uses the same player count.
        self._num_players = num_players
        players = {i + 1: f"Player {i + 1}" for i in range(num_players)}
        return RustGameAdapter(RustGame(players))

    def _sample_num_players(self) -> int:
        """Sample a player count from the config's distribution.

        Falls back to 4 if the distribution is missing/empty (so legacy configs
        constructed without explicitly setting ``player_count_distribution``
        keep their old behaviour).
        """
        # Reuse a previously sampled count if one is stashed (e.g., by
        # ``extract_data`` rebuilding the game state for replay).
        cached = getattr(self, "_num_players", None)
        if cached is not None:
            return cached
        dist = getattr(self.config, "player_count_distribution", None)
        if not dist:
            return 4
        counts = list(dist.keys())
        weights = [dist[c] for c in counts]
        total = sum(weights)
        if total <= 0:
            return 4
        return random.choices(counts, weights=weights, k=1)[0]

    def _compute_net_worth(self, game=None) -> dict:
        """Return {player_id: net_worth} for the given game (defaults to root state).

        Thin instance wrapper around the module-level ``_compute_net_worth``. The same
        formula is used for natural game-end and for truncated games so that targets
        are comparable across both branches.
        """
        if game is None:
            game = self.root.game_object
        return _compute_net_worth(game)

    def initialize_game(self, game_state: Optional[Union[BaseGame, RustGameAdapter]] = None):
        if game_state is None:
            game_state = self.get_new_game_state()
        else:
            # Sync the cached player count to whatever the external game has so
            # ``extract_data``'s replay game matches (otherwise a freshly
            # constructed MCTSPlayer would default to 4 players).
            self._num_players = len(game_state.players)

        self.root = mcts.MCTSNode(game_state, config=self.config)
        self.result = np.zeros(len(game_state.players))
        self.result_string = None
        self.searches_pi = []
        # Per-move ContinuousPriceHead targets aggregated from MCTS visits.
        # Each entry is a list of ``(slot_idx, price, weight, price_min,
        # price_max)`` tuples (empty for categorical-only moves). Captured in
        # ``play_move`` before pruning discards the price grandchildren.
        self.price_targets: list[list[tuple]] = []
        # Track action dicts for replay in extract_data.
        # Seed with any actions already in the game state (e.g., from a terminal_game_state test fixture).
        self.played_actions = list(game_state.raw_actions) if hasattr(game_state, "raw_actions") else []
        LOGGER.info(f"Initialized game. Root node N: {self.root.N}")

    def play_move(self, action_index):
        """Notable side effects:
        - finalizes the probability distribution according to
        this roots visit counts into the class' running tally, `searches_pi`
        - Makes the node associated with this move the root, for future
          `inject_noise` calls.
        """
        self.log_memory_usage(stage_name="MCTSPlayer.play_move")
        temperature = 1.0 if self.root.game_object.move_number < self.config.softpick_move_cutoff else 0.0
        self.searches_pi.append(self.root.children_as_pi(temperature=temperature))

        # If the picked categorical slot is price-bearing, descend to the
        # most-visited price grandchild and commit to its concrete price.
        # Otherwise materialize the categorical child directly.
        price_range = self.root.price_ranges_by_idx.get(action_index)
        if price_range is not None and price_range[0] != price_range[1]:
            new_root = self._select_most_visited_price_grandchild(action_index)
            action_obj = self.root.action_mapper.map_index_to_action_with_price(
                action_index, self.root.game_object, new_root.sampled_price
            )
        else:
            # Capture the action dict before advancing (pickle_clone strips action history)
            action_obj = self.root.action_mapper.map_index_to_action(action_index, self.root.game_object)
            new_root = self.root.maybe_add_child(action_index)
        self.played_actions.append(action_obj.to_dict() if hasattr(action_obj, "to_dict") else action_obj)

        # Capture ContinuousPriceHead targets from MCTS price grandchildren
        # BEFORE pruning destroys them. For price-bearing slots, every
        # explored grandchild contributes one (slot_idx, price, weight,
        # price_min, price_max) tuple weighted by its share of visits within
        # the slot. Categorical-only moves contribute an empty list.
        self.price_targets.append(
            self._extract_price_targets(action_obj, action_index, price_range)
        )

        # Record PW grandchild fan-out so the user has empirical data to
        # tune ``pw_c`` / ``pw_alpha`` against. Aggregates over every
        # *visited* PW slot at this state (not just the chosen one): if a
        # slot saw 0 expansions it doesn't contribute.
        for _slot, grandchildren in self.root.price_children.items():
            visited = [gc for gc in grandchildren.values() if gc.N > 0]
            if visited:
                self.add_metric(
                    "MCTS/Price/Grandchildren_Per_PW_Slot", len(visited)
                )

        self.root = new_root
        # Prune the tree
        self.prune_mcts_tree_retain_parent(self.root)

        LOGGER.debug(f"Played move. New root N: {self.root.N}, Searches_pi length: {len(self.searches_pi)}")
        self.log_memory_usage(stage_name="MCTSPlayer.play_move")

        self.game_state = self.root.game_object.to_dict()  # for showboard
        return True

    def _extract_price_targets(
        self,
        action_obj,
        action_index: int,
        price_range: Optional[tuple],
    ) -> list[tuple]:
        """Aggregate ContinuousPriceHead targets across visited price grandchildren.

        For a price-bearing categorical slot, MCTS may have sampled several
        prices via progressive widening; each grandchild has its own visit
        count. We turn those into a list of
        ``(slot_idx, price, weight, price_min, price_max)`` tuples — the
        same shape pretraining writes — so the price head's NLL loss sees a
        visit-weighted mixture of observed-good prices for this state.

        Returns an empty list for categorical-only moves, for fixed-price
        slots, and when the action doesn't map to a ContinuousPriceHead slot
        (e.g. depot trains, exchange trains).
        """
        if price_range is None or price_range[0] == price_range[1]:
            return []
        # Defensive: a slot-resolver bug must NOT kill the whole game. Self-play
        # spends minutes per game; losing one is much worse than missing a
        # price-target for one move. Log and continue with an empty list.
        try:
            slot_info = self.root.action_mapper.price_head_slot_for_action(
                action_obj, self.root.game_object
            )
        except Exception as e:
            LOGGER.warning(
                "price_head_slot_for_action raised for action_index=%s: %s",
                action_index, e,
            )
            return []
        if slot_info is None:
            return []
        _action_type, slot_index, _observed_price, price_min, price_max = slot_info
        slot_grandchildren = self.root.price_children.get(action_index, {})
        if not slot_grandchildren:
            return []
        total_visits = float(sum(gc.N for gc in slot_grandchildren.values()))
        if total_visits <= 0:
            return []
        return [
            (
                int(slot_index),
                float(gc.sampled_price),
                float(gc.N) / total_visits,
                float(price_min),
                float(price_max),
            )
            for gc in slot_grandchildren.values()
            if gc.sampled_price is not None and gc.N > 0
        ]

    def _select_most_visited_price_grandchild(self, action_index: int) -> "mcts.MCTSNode":
        """Pick the most-visited price grandchild under ``action_index``.

        Called by ``play_move`` when the committed categorical slot is
        price-bearing. Falls back to creating one via the PW sampler if no
        grandchildren exist yet (rare: implies the slot was selected before
        any descent reached it during ``tree_search``).
        """
        slot_grandchildren = self.root.price_children.get(action_index, {})
        if not slot_grandchildren:
            return self.root.maybe_add_child(action_index)
        # Tie-break stably by snapped price so behaviour is deterministic across runs.
        return max(
            slot_grandchildren.values(),
            key=lambda gc: (gc.N, gc.sampled_price if gc.sampled_price is not None else 0),
        )

    def pick_move(self):
        """Picks a move to play, based on MCTS readout statistics.

        Highest N is most robust indicator. In the early stage of the game, pick
        a move weighted by visit count (temperature=1.0); later on, pick the
        absolute max (equivalent to temperature→0).

        Works on compressed arrays (only legal actions) to avoid allocating
        full 26,535-element arrays on every move.
        """
        if self.root.game_object.move_number >= self.config.softpick_move_cutoff:
            return self.root.best_child()

        if self.root.num_legal_actions == 1:
            return self.root.legal_action_indices[0]

        # Use compressed visit counts directly
        visit_counts = self.root.child_N_compressed.astype(np.float64)
        total = visit_counts.sum()
        if total == 0:
            # No visits; fall back to uniform over legal actions.
            compressed_idx = random.randrange(self.root.num_legal_actions)
            return self.root.legal_action_indices[compressed_idx]

        probs = visit_counts / total
        cdf = probs.cumsum()
        selection = random.random()
        compressed_idx = cdf.searchsorted(selection)
        compressed_idx = min(compressed_idx, self.root.num_legal_actions - 1)
        return self.root.legal_action_indices[compressed_idx]

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

        leaf_selection_duration = select_leaves_end - select_leaves_start
        self.cumulative_leaf_selection_time += leaf_selection_duration
        self.add_metric("MCTS/Select_Leaf_Time", leaf_selection_duration)
        self.add_metric("MCTS/Select_Leaf_Attempts", select_leaf_attempts)
        self.add_metric("MCTS/Max_Select_Leaf_Attempts", max_select_leaf_attempts)
        self.add_metric("MCTS/Leaves_Found", len(leaves))

        if select_leaf_attempts >= max_select_leaf_attempts and len(leaves) < parallel_readouts:
            LOGGER.warning(
                f"tree_search: Failsafe triggered while selecting leaves. Found {len(leaves)}/{parallel_readouts} leaves after {select_leaf_attempts}/{max_select_leaf_attempts} attempts."
            )
            self.add_metric("MCTS/Failsafe_Triggered", 1)

        if leaves:
            # Encoding (separate from inference for profiling)
            encode_start = time.time()
            for leaf in leaves:
                leaf.ensure_encoded()
            encode_duration = time.time() - encode_start
            self.cumulative_encoding_time += encode_duration

            # Phase 6.5: FP16 inference for ~2x GPU throughput (CUDA or MPS)
            inference_start = time.time()
            autocast_device = _get_autocast_device() if self.config.use_fp16_inference else None
            if autocast_device:
                with torch.no_grad(), torch.amp.autocast(autocast_device):
                    move_probs, _, values = self.network.run_many_encoded(
                        [leaf.encoded_game_state for leaf in leaves]
                    )
            else:
                with torch.no_grad():
                    move_probs, _, values = self.network.run_many_encoded(
                        [leaf.encoded_game_state for leaf in leaves]
                    )
            inference_duration = time.time() - inference_start
            self.cumulative_inference_time += inference_duration

            # Continuous price head outputs (Task #28): the transformer model
            # stashes ``last_price_components`` (batched tensors + slot index
            # lookup) after each forward pass. Slice per-leaf so MCTS PW can
            # sample prices from the head's Normal. None for the GNN model
            # (no price head) — MCTS falls back to a wide-Normal default.
            batched_price_components = getattr(self.network, "last_price_components", None)

            # Combined for backward-compat metric
            run_network_duration = encode_duration + inference_duration

            revert_and_incorporate_start = time.time()
            for i, (leaf, move_prob, value) in enumerate(zip(leaves, move_probs, values)):
                leaf.revert_virtual_loss(up_to=self.root)
                leaf_price_components = _slice_price_components(batched_price_components, i)
                leaf.incorporate_results(
                    move_prob, value, up_to=self.root, price_components=leaf_price_components
                )

                # metrics
                leaf_depths_collected.append(leaf.depth)
                leaf_initial_qs_collected.append(value[leaf.active_player_index].item())
                if np.sum(leaf.child_prior_compressed) > 1e-6:  # Ensure it's not all zeros
                    normalized_prior_compressed = leaf.child_prior_compressed / np.sum(leaf.child_prior_compressed)
                    leaf_prior_entropies_collected.append(mcts.calculate_entropy(normalized_prior_compressed))
            revert_and_incorporate_duration = time.time() - revert_and_incorporate_start
            self.cumulative_backup_time += revert_and_incorporate_duration
            self.add_histogram("MCTS_Player/TreeSearch_Leaf_Depths", np.array(leaf_depths_collected))
            self.add_histogram("MCTS_Player/TreeSearch_Leaf_Initial_Network_Q", np.array(leaf_initial_qs_collected))
            self.add_histogram("MCTS_Player/TreeSearch_Leaf_Prior_Entropies", np.array(leaf_prior_entropies_collected))
        else:
            run_network_duration = 0
            encode_duration = 0
            inference_duration = 0
            revert_and_incorporate_duration = 0

        self.add_metric("MCTS/Run_Network_Time", run_network_duration)
        self.add_metric("MCTS/Encode_Time", encode_duration)
        self.add_metric("MCTS/Inference_Time", inference_duration)
        self.add_metric("MCTS/Revert_And_Incorporate_Time", revert_and_incorporate_duration)
        return leaves

    def adaptive_readouts(self) -> int:
        """Scale readouts by position complexity. Low-branching nodes use min_readouts."""
        if self.root.num_legal_actions <= self.config.adaptive_readout_threshold:
            return self.config.min_readouts
        return self.config.num_readouts

    def suggest_move(self, override_readouts: int = None):
        if self.root.num_legal_actions == 1:
            return self.pick_move()

        readouts = override_readouts if override_readouts is not None else self.adaptive_readouts()
        target_readouts_for_move = self.root.N + readouts

        while self.root.N < target_readouts_for_move:
            self.tree_search()

        return self.pick_move()

    def is_done(self):
        return (not np.array_equal(self.result, np.zeros_like(self.result))) or self.root.is_done()

    def set_result(self, result):
        self.result = np.array(result)
        string = self.root.game_result_string()
        self.result_string = string

    def extract_data(self) -> Generator[Tuple[Union[BaseGame, RustGameAdapter], torch.Tensor, torch.Tensor, torch.Tensor, list], None, None]:
        assert (
            len(self.searches_pi) == len(self.played_actions)
        ), f"searches_pi length {len(self.searches_pi)} != played_actions length {len(self.played_actions)}"
        # ``self.result`` is sized to the number of players in the game (set by
        # ``initialize_game``), so use the array's own shape rather than a
        # hard-coded length-4 vector.
        assert not np.array_equal(self.result, np.zeros_like(self.result)), f"result {self.result} is 0"

        # ``price_targets`` is built in lockstep with ``played_actions`` /
        # ``searches_pi`` (one entry per ``play_move`` call). Pad missing
        # entries with an empty list to keep the schema uniform if a legacy
        # caller bypassed ``play_move``.
        if len(self.price_targets) < len(self.played_actions):
            pad = [[]] * (len(self.played_actions) - len(self.price_targets))
            self.price_targets.extend(pad)

        result = torch.tensor(self.result)
        # Training loop runs exclusively on the Rust engine, so the replay game
        # is always a fresh RustGameAdapter.
        game_state = self.get_new_game_state()
        action_mapper = ActionMapper()
        for i, action in enumerate(self.played_actions):
            yield (
                game_state,
                torch.tensor(action_mapper.get_legal_action_indices(game_state)),
                torch.tensor(self.searches_pi[i])
                if isinstance(self.searches_pi[i], np.ndarray)
                else self.searches_pi[i],
                result,
                self.price_targets[i],
            )
            game_state = game_state.pickle_clone()
            game_state.process_action(action)

    def _recursive_clear_references(self, node: mcts.MCTSNode, stats: Optional[dict] = None):
        """
        Recursively clears parent and children references to help with garbage collection.
        Sets node.parent to None and clears node.children + node.price_children.
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
        # Price grandchildren live under price_children[action_index][price].
        for slot_grandchildren in getattr(node, "price_children", {}).values():
            children_to_visit.extend(slot_grandchildren.values())
        node.children.clear()  # Clear this node's children dict
        if hasattr(node, "price_children"):
            node.price_children.clear()
        if hasattr(node, "price_child_N"):
            node.price_child_N.clear()
        if hasattr(node, "price_child_W"):
            node.price_child_W.clear()

        for child in children_to_visit:
            self._recursive_clear_references(child, stats)  # Recurse

    def prune_mcts_tree_retain_parent(self, new_search_root: mcts.MCTSNode):
        parent_of_new_root = new_search_root.parent

        # Case 1: new_search_root is the first real node of the game (its parent is DummyNode).
        if isinstance(parent_of_new_root, mcts.DummyNode) or new_search_root.fmove is None:
            LOGGER.info(f"Noop during pruning: new root is the initial game root or has a DummyNode parent.")
            return

        LOGGER.debug(f"Pruning tree. New root: {new_search_root}, its parent (old root): {parent_of_new_root}")
        # Identify and collect siblings of new_search_root for pruning. Siblings live
        # in two places now: the categorical ``children`` dict, and the price-children
        # nested dict (one per categorical slot). If the new root is a price
        # grandchild, all *other* grandchildren of its own slot are siblings to prune;
        # if it's a categorical child, all entries in both dicts other than its own
        # ``fmove`` are siblings.
        siblings_to_prune_roots = []
        new_root_is_grandchild = new_search_root.sampled_price is not None and (
            new_search_root.fmove in getattr(parent_of_new_root, "price_children", {})
        )
        for action_index, child_node in parent_of_new_root.children.items():
            if action_index != new_search_root.fmove or new_root_is_grandchild:
                siblings_to_prune_roots.append(child_node)
        for action_index, slot_grandchildren in getattr(parent_of_new_root, "price_children", {}).items():
            for price, grandchild in slot_grandchildren.items():
                if grandchild is new_search_root:
                    continue
                siblings_to_prune_roots.append(grandchild)

        # Modify parent_of_new_root to only contain new_search_root. Whether the
        # new root is a categorical child or a price grandchild governs which dict
        # it goes into. Capture grandchild N/W *before* clearing the parent's
        # price-children dicts (N/W are read from those dicts via properties).
        saved_grandchild_N = new_search_root.N if new_root_is_grandchild else None
        saved_grandchild_W = (
            new_search_root.W.copy()
            if new_root_is_grandchild and hasattr(new_search_root.W, "copy")
            else (new_search_root.W if new_root_is_grandchild else None)
        )
        parent_of_new_root.children.clear()
        if hasattr(parent_of_new_root, "price_children"):
            parent_of_new_root.price_children.clear()
        if hasattr(parent_of_new_root, "price_child_N"):
            parent_of_new_root.price_child_N.clear()
        if hasattr(parent_of_new_root, "price_child_W"):
            parent_of_new_root.price_child_W.clear()
        if new_root_is_grandchild:
            parent_of_new_root.price_children[new_search_root.fmove] = {
                new_search_root.sampled_price: new_search_root
            }
            parent_of_new_root.price_child_N[new_search_root.fmove] = {
                new_search_root.sampled_price: saved_grandchild_N
            }
            parent_of_new_root.price_child_W[new_search_root.fmove] = {
                new_search_root.sampled_price: saved_grandchild_W
            }
        else:
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
    def __init__(self, config: SelfPlayConfig, model_config: Optional[ModelTransformerConfig] = None):
        self.config = config
        assert config.network is not None or model_config is not None, "Network must be provided"
        if model_config is not None:
            self.config.network = AlphaZeroTransformerModel(model_config)
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
        timing: Optional[dict] = None,
        phase_move_counts: Optional[dict] = None,
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
            "last_update_unix": time.time(),
        }
        if timing is not None:
            status_data["timing"] = timing
        if phase_move_counts is not None:
            status_data["phase_move_counts"] = phase_move_counts
        try:
            atomic_write_json(file, status_data, indent=4)
        except IOError as e:
            LOGGER.error(f"Error writing to {SELF_PLAY_GAMES_STATUS_PATH}: {e}")
        except Exception as e:  # Catch any other unexpected error during file write
            LOGGER.error(f"Unexpected error writing {SELF_PLAY_GAMES_STATUS_PATH}: {e}", exc_info=True)

    def play(self):
        """Plays out a self-play match, returning a MCTSPlayer object containing:
        - the final position
        - the n x 26537 tensor of floats representing the mcts search probabilities
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
            status="Starting Up",
        )

        total_tree_search_time_for_game = 0
        total_pick_move_time_for_game = 0
        total_play_move_time_for_game = 0
        total_move_time_for_game = 0
        num_forced_moves_in_game = 0
        num_mcts_moves_in_game = 0
        total_sims_for_mcts_moves = 0

        # Phase move counts (Item 6)
        phase_move_counts = {"Auction": 0, "WaterfallAuction": 0, "Stock": 0, "Operating": 0, "Other": 0}

        # Must run this once at the start to expand the root node.
        first_node = player.root.select_leaf()
        first_node.ensure_encoded()
        autocast_device = _get_autocast_device() if self.config.use_fp16_inference else None
        if autocast_device:
            with torch.no_grad(), torch.amp.autocast(autocast_device):
                probs, _, val = self.config.network.run_encoded(first_node.encoded_game_state)
        else:
            with torch.no_grad():
                probs, _, val = self.config.network.run_encoded(first_node.encoded_game_state)
        # Slice the (single-leaf) price head outputs for the root expansion so
        # MCTS PW has price priors available on the first descent. ``None`` for
        # models without a price head (GNN).
        first_price_components = _slice_price_components(
            getattr(self.config.network, "last_price_components", None), 0
        )
        first_node.incorporate_results(
            probs, val, first_node, price_components=first_price_components
        )
        del first_node
        move_counter = 0
        game_ended_by_max_length = 0
        try:
            while True:
                LOGGER.info(f"SelfPlay.play loop start, move {move_counter}")
                self.log_memory_usage(stage_name=f"SelfPlay.play")
                start_time_for_move_processing = time.time()

                sim_count_this_move = 0
                tree_search_duration_this_move = 0
                if player.root.num_legal_actions == 1:
                    LOGGER.info(f"Move {move_counter}: Only one legal action. Skipping MCTS.")
                    num_forced_moves_in_game += 1
                else:
                    player.root.inject_noise()
                    num_mcts_moves_in_game += 1
                    current_readouts = player.root.N
                    target_readouts_for_move = current_readouts + player.adaptive_readouts()

                    tree_search_start = time.time()
                    while player.root.N < target_readouts_for_move:
                        player.tree_search()
                        sim_count_this_move += self.config.parallel_readouts
                    tree_search_duration_this_move = time.time() - tree_search_start
                    total_sims_for_mcts_moves += player.root.N - current_readouts
                total_tree_search_time_for_game += tree_search_duration_this_move

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

                # Track phase move counts (Item 6)
                round_class_name = player.root.game_object.round.__class__.__name__
                if round_class_name in phase_move_counts:
                    phase_move_counts[round_class_name] += 1
                else:
                    phase_move_counts["Other"] += 1

                self.update_self_play_game_progress(
                    game_id=self.config.game_id,
                    loop_number=self.config.global_step,
                    game_number=self.config.game_idx_in_iteration,
                    moves_played=move_counter,
                    max_moves=self.config.max_game_length,
                    current_round=player.root.game_object.round.round_description(),
                    last_action=player.root.game_object.actions[-1].description(),
                    game_start_time_unix=game_start_time,
                    status="In Progress",
                    phase_move_counts=phase_move_counts,
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
                    if player.root.game_object.move_number >= self.config.max_game_length:
                        # Truncated game: derive win/loss + score targets from net worth
                        # at the truncation step. end_game() flips the engine's `finished`
                        # flag but does not mutate player cash / share holdings; the
                        # subsequent game_result() therefore uses the same net-worth
                        # formula as a natural ending (game.result() == per-player value).
                        net_worth = _compute_net_worth(player.root.game_object)
                        LOGGER.info(
                            f"Game ended by max length ({move_counter} moves). "
                            f"Net worth at truncation: {net_worth}"
                        )
                        player.root.game_object.end_game()
                        game_ended_by_max_length = 1

                    player.set_result(player.root.game_result())
                    LOGGER.info(
                        f"Game finished after {move_counter} moves. Result: {player.root.game_object.result()}, mapped to: {player.result} via {player.root.player_mapping}"
                    )

                    game_timing = {
                        "total_game_s": round(total_move_time_for_game, 3),
                        "tree_search_s": round(total_tree_search_time_for_game, 3),
                        "inference_s": round(player.cumulative_inference_time, 3),
                        "encoding_s": round(player.cumulative_encoding_time, 3),
                        "leaf_selection_s": round(player.cumulative_leaf_selection_time, 3),
                        "backup_s": round(player.cumulative_backup_time, 3),
                        "pick_move_s": round(total_pick_move_time_for_game, 3),
                        "play_move_s": round(total_play_move_time_for_game, 3),
                        "num_mcts_moves": num_mcts_moves_in_game,
                        "num_forced_moves": num_forced_moves_in_game,
                        "total_sims": total_sims_for_mcts_moves,
                    }

                    self.update_self_play_game_progress(
                        game_id=self.config.game_id,
                        loop_number=self.config.global_step,
                        game_number=self.config.game_idx_in_iteration,
                        moves_played=move_counter,
                        max_moves=self.config.max_game_length,
                        current_round="Finished",
                        last_action=player.root.game_object.actions[-1].description(),
                        game_start_time_unix=game_start_time,
                        status="Completed",
                        timing=game_timing,
                        phase_move_counts=phase_move_counts,
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
                status="Error",
            )

        self.add_metric("SelfPlay/Game_Length_Moves", move_counter)
        self.add_metric("SelfPlay/Game_Total_Time_Seconds", total_move_time_for_game)
        self.add_metric("SelfPlay/Game_Num_Forced_Moves", num_forced_moves_in_game)
        self.add_metric("SelfPlay/Game_Ended_By_Max_Length", game_ended_by_max_length)
        # Per-game binary truncation indicator; averaging across games yields the
        # truncation rate. Logged unconditionally (0 for natural end, 1 for truncation).
        self.add_metric("self_play/truncated", float(game_ended_by_max_length))

        if player.result is not None and len(player.result) > 0:
            for i, score in enumerate(player.result):
                self.add_metric(f"SelfPlay/Game_Result_Player{i}", float(score))

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
        os.makedirs(self.config.selfplay_dir, exist_ok=True)

        player = self.play()

        LOGGER.info(f"Player result: {player.result}")
        LOGGER.info(f"Game actions: {player.root.game_object.raw_actions}")
        if player.result is None or np.all(np.array(player.result) == 0.0):
            LOGGER.warning(
                f"Game {self.config.game_id} finished with no conclusive result or result not set. Skipping data extraction."
            )
            self.add_metric("SelfPlay/Games_Skipped_No_Result", 1)
            return

        extraction_start = time.time()
        game_data = player.extract_data()

        save_path = self.config.selfplay_dir / self.config.network.get_name()

        processor = TrainingExampleProcessor(self.config.network.encoder)
        processor.write_lmdb(game_data, save_path)
        extraction_duration = time.time() - extraction_start

        # Update game status with extraction timing (atomic write)
        game_file = SELF_PLAY_GAMES_STATUS_PATH / f"{self.config.game_id}.json"
        if game_file.exists():
            try:
                with open(game_file, "r") as f:
                    status_data = json.load(f)
                if "timing" in status_data:
                    status_data["timing"]["data_extraction_s"] = round(extraction_duration, 3)
                    atomic_write_json(game_file, status_data, indent=4)
            except Exception as e:
                LOGGER.warning(f"Failed to update timing with extraction data: {e}")

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
