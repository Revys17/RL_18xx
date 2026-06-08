"""Phase 4b Rust-MCTS-backed Python adapter.

Wraps ``engine_rs.RustMCTSPlayer`` (the arena-based Rust MCTS tree from Phase
4a) with a Python class that mirrors the surface of
``rl18xx.agent.alphazero.self_play.MCTSPlayer`` needed by ``SelfPlay.play()``.

Scope (4b): **categorical descent only**. Bid / BuyTrain / BuyCompany slots
are treated as fixed-price at ``price_range[0]``. PW + continuous prices land
in Phase 4c.

The adapter does not subclass ``MCTSPlayer`` — its internal node bookkeeping
lives in Rust and ``MCTSPlayer``'s tree-walking code does not apply. The
methods below are the minimal set that ``SelfPlay.play()`` calls.
"""
from __future__ import annotations

import collections
import logging
import random
from typing import Optional, Generator, Tuple, Union

import numpy as np
import torch

import engine_rs
from rl18xx.agent.alphazero.config import SelfPlayConfig
from rl18xx.agent.alphazero.mcts import (
    POLICY_SIZE,
    VALUE_SIZE,
    _rust_encode,
)
from rl18xx.agent.alphazero.self_play import _compute_net_worth, _slice_price_components
from rl18xx.rust_adapter import RustGameAdapter

LOGGER = logging.getLogger(__name__)
_RESIGN_DISABLED_LOGGED = False


def _coerce_price_components_for_rust(leaf_pc: Optional[dict]) -> Optional[dict]:
    """Coerce a per-leaf ``price_components`` dict into a Rust-friendly shape.

    The Rust ``RustMCTSPlayer.incorporate_results`` decoder expects:
      - ``price_mean`` / ``price_log_std``: numpy float32 1D arrays.
      - ``slot_index``: dict of ``(action_type, (entity_key_parts,))`` -> int.
      - ``num_slots``: int.

    The transformer model emits torch tensors for the means/log-stds; this
    helper detaches them to numpy float32 once at the FFI boundary. ``None``
    propagates through (the GNN model has no price head, so PW falls back to
    a wide-Normal default on the Rust side).
    """
    if leaf_pc is None:
        return None
    means = leaf_pc.get("price_mean")
    log_stds = leaf_pc.get("price_log_std")
    slot_index = leaf_pc.get("slot_index")
    num_slots = leaf_pc.get("num_slots")
    if means is None or log_stds is None or slot_index is None:
        return None
    if isinstance(means, torch.Tensor):
        means_np = means.detach().cpu().numpy().astype(np.float32)
    else:
        means_np = np.asarray(means, dtype=np.float32)
    if isinstance(log_stds, torch.Tensor):
        log_stds_np = log_stds.detach().cpu().numpy().astype(np.float32)
    else:
        log_stds_np = np.asarray(log_stds, dtype=np.float32)
    return {
        "price_mean": means_np,
        "price_log_std": log_stds_np,
        "slot_index": slot_index,
        "num_slots": int(num_slots) if num_slots is not None else int(means_np.shape[0]),
    }


def _active_player_index(adapter) -> int:
    """Return the active-player index in the canonical (sorted-by-id) ordering.

    Mirrors ``MCTSNode.active_player_index`` so Phase 1 PlayoutTrace's
    ``leaf_q_perspective`` reads the value vector at the right slot.
    """
    sorted_ids = sorted(p.id for p in adapter.players)
    actives = adapter.active_players()
    if not actives:
        return 0
    active_id = actives[0].id
    try:
        return sorted_ids.index(active_id)
    except ValueError:
        return 0


def _select_backend(config: SelfPlayConfig):
    """Return the inference backend (mirrors ``MCTSPlayer._select_inference_backend``).

    Used so the Rust adapter shares Phase 3's inference-server / in-process
    backend selection logic without duplicating it.
    """
    if not bool(getattr(config, "use_inference_server", False)):
        return config.network
    client = getattr(config, "inference_client", None)
    if client is None:
        LOGGER.warning(
            "RustMCTSPlayer: use_inference_server=True but no inference_client "
            "on the config; falling back to local network."
        )
        return config.network
    return client


class RustMCTSPlayer:
    """Python-side adapter around ``engine_rs.RustMCTSPlayer``.

    Lives in this module so the Python ``MCTSPlayer`` (in ``self_play.py``)
    stays untouched. Implements only the surface ``SelfPlay.play()`` needs.
    """

    def __init__(self, config: SelfPlayConfig):
        self.config = config
        self.network = config.network
        self._backend = _select_backend(config)

        # Cumulative timing accumulators (mirroring MCTSPlayer's surface so
        # SelfPlay.play()'s game_timing block works unchanged).
        self.cumulative_inference_time = 0.0
        self.cumulative_encoding_time = 0.0
        self.cumulative_leaf_selection_time = 0.0
        self.cumulative_backup_time = 0.0

        # Phase 2 resign bookkeeping (Rust path full parity with Python).
        holdout_rate = float(getattr(self.config, "noresign_holdout_rate", 0.0))
        self._noresign_holdout: bool = random.random() < holdout_rate
        self._q_window: collections.deque = collections.deque(
            maxlen=max(1, int(getattr(self.config, "resign_window", 1)))
        )
        self._would_have_resigned_info: Optional[dict] = None
        self.termination: Optional[str] = None

        # Phase 1 PlayoutTrace (full parity with Python). Game-level coin flip
        # decides whether this whole game is traced; ``self.traces`` collects
        # one ``PlayoutTrace`` per recorded leaf across all moves.
        from rl18xx.agent.alphazero.mcts import PlayoutTrace as _PlayoutTrace
        self._PlayoutTrace = _PlayoutTrace
        trace_cfg = getattr(self.config, "trace", None)
        rate = float(getattr(trace_cfg, "trace_game_rate", 0.0)) if trace_cfg is not None else 0.0
        self._tracing_enabled = rate > 0.0 and random.random() < rate
        self.traces: list = []
        self._traces_collected_this_move = 0

        self.initialize_game()

    # ----------------------------------------------------------------- utils
    def __str__(self) -> str:
        return "RustMCTSPlayer"

    def __repr__(self) -> str:
        return self.__str__()

    def add_metric(self, name, value):
        if self.config.metrics is None:
            return
        self.config.metrics.add_scalar(
            name, value, self.config.global_step, self.config.game_idx_in_iteration
        )

    def _sample_num_players(self) -> int:
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

    def get_new_game_state(self) -> RustGameAdapter:
        from engine_rs import BaseGame as RustBaseGame

        num_players = self._sample_num_players()
        self._num_players = num_players
        players = {i + 1: f"Player {i + 1}" for i in range(num_players)}
        return RustGameAdapter(RustBaseGame(players))

    # ------------------------------------------------------- initialization
    def initialize_game(self, game_state: Optional[RustGameAdapter] = None):
        if game_state is None:
            game_state = self.get_new_game_state()
        else:
            self._num_players = len(game_state.players)

        self._game_state = game_state  # Python-side handle, kept for replays.
        # Pass PW knobs into the Rust player so its progressive-widening
        # schedule matches Python's ``SelfPlayConfig`` (Phase 4c).
        self._rust_player = engine_rs.RustMCTSPlayer(
            game_state,
            float(getattr(self.config, "pw_c", 1.0)),
            float(getattr(self.config, "pw_alpha", 0.5)),
            int(getattr(self.config, "min_price_children", 1)),
        )
        self.result = np.zeros(len(game_state.players))
        self.result_string: Optional[str] = None
        self.searches_pi: list = []
        self.price_targets: list[list[tuple]] = []
        # Seed with any actions already in the game (test fixtures), as
        # MCTSPlayer.initialize_game does.
        self.played_actions = (
            list(game_state.raw_actions) if hasattr(game_state, "raw_actions") else []
        )
        self.forced_action_dicts: list[list[dict]] = []
        self._initial_action_count = len(self.played_actions)
        LOGGER.info(
            "Initialized RustMCTSPlayer. Root legal action count: %d",
            len(self._rust_player.legal_action_indices_at_root()),
        )

    # ------------------------------------------------------------- getters
    def get_game_state(self):
        """Return a fresh adapter around the current root's BaseGame."""
        rust_game = self._rust_player.root_game_object()
        return RustGameAdapter(rust_game)

    def get_root(self):
        """Best-effort root surrogate (``SelfPlay.play()`` reads ``.root``)."""
        return _RootShim(self)

    @property
    def root(self):
        """``SelfPlay.play()`` reads ``player.root.<attr>`` directly all over
        the place; expose the shim under the canonical attribute name so the
        existing orchestration loop works unchanged."""
        return _RootShim(self)

    def get_result_string(self) -> Optional[str]:
        return self.result_string

    # -------------------------------------------------------- core MCTS API
    def _is_root_terminal(self) -> bool:
        return bool(self._rust_player.is_terminal(0))

    def _compute_terminal_value(self, leaf_idx: int) -> np.ndarray:
        """Compute the per-player terminal value vector at a leaf.

        Mirrors ``MCTSNode.game_result()`` with ``use_score_values=True``:
        normalized net-worth fractions in the first ``num_players`` slots,
        zeros for the padded slots up to ``VALUE_SIZE``.
        """
        rust_game = self._rust_player.get_game_for_idx(leaf_idx)
        adapter = RustGameAdapter(rust_game)
        net_worth = _compute_net_worth(adapter)
        value = np.zeros(VALUE_SIZE, dtype=np.float32)
        if not net_worth:
            return value
        ids = sorted(net_worth.keys())
        scores = np.array([float(net_worth[pid]) for pid in ids], dtype=np.float32)
        total = float(scores.sum())
        if total > 0:
            value[: len(scores)] = scores / total
        elif len(scores) > 0:
            value[: len(scores)] = 1.0 / len(scores)
        if not self.config.use_score_values:
            # Mirror the legacy {-1, 0, +1} win-loss vector for
            # use_score_values=False so callers that disable score values
            # still see consistent backups. ``_compute_net_worth`` returns
            # the score-style payload; convert to win/loss here.
            winning = float(max(net_worth.values()))
            wl = np.full(VALUE_SIZE, -1.0, dtype=np.float32)
            winners = [
                i for i, pid in enumerate(ids) if float(net_worth[pid]) == winning
            ]
            if len(winners) > 1:
                wl[winners] = 0.0
            else:
                wl[winners] = 1.0
            return wl
        return value

    def tree_search(self, parallel_readouts: Optional[int] = None):
        import time

        if parallel_readouts is None:
            parallel_readouts = min(
                self.config.parallel_readouts, self.config.num_readouts
            )

        # Phase 1 PlayoutTrace decision for this move (mirrors
        # MCTSPlayer.tree_search). Budgeted across all tree_search() calls
        # until play_move resets ``_traces_collected_this_move``.
        trace_cfg = getattr(self.config, "trace", None)
        trace_this_move = False
        traces_budget = 0
        if self._tracing_enabled and trace_cfg is not None:
            move_idx = len(self.searches_pi)
            every_n = max(1, int(trace_cfg.trace_every_n_moves))
            if move_idx % every_n == 0:
                trace_this_move = True
                traces_budget = max(0, int(trace_cfg.traces_per_move))

        leaves: list[int] = []  # arena indices
        encoded_states: list = []
        leaf_active_player_idx: list[int] = []  # mirrors leaves; for trace q_perspective
        leaf_traces: list = []  # parallel to leaves; None if untraced

        select_start = time.time()
        failsafe = 0
        max_attempts = parallel_readouts * 2
        while len(leaves) < parallel_readouts and failsafe < max_attempts:
            failsafe += 1
            # Trace this leaf? Only if tracing is on for this move AND we
            # still have budget AND we have a fresh, non-empty descent path
            # to record. The first cycle on a freshly-rooted tree often has
            # an empty action_path (root not yet expanded) — still useful
            # for recording leaf metadata.
            do_trace = (
                trace_this_move
                and self._traces_collected_this_move < traces_budget
            )
            if do_trace:
                idx, action_path, pw_path, forced_lens = (
                    self._rust_player.select_leaf_with_trace()
                )
                trace = self._PlayoutTrace(
                    move_idx=len(self.searches_pi),
                    leaf_depth=len(action_path),
                    action_path=[int(a) for a in action_path],
                    pw_grandchild_path=[bool(p) for p in pw_path],
                    forced_chain_lengths=[int(f) for f in forced_lens],
                )
            else:
                idx = self._rust_player.select_leaf()
                trace = None

            if self._rust_player.is_terminal(idx):
                value = self._compute_terminal_value(idx)
                self._rust_player.backup_value(idx, value.astype(np.float32).tolist())
                if trace is not None:
                    trace.leaf_terminal = True
                    trace.nn_value = np.asarray(value, dtype=np.float32)
                    # Active player at a terminal leaf: pull from the Rust game.
                    rust_game = self._rust_player.get_game_for_idx(idx)
                    adapter = RustGameAdapter(rust_game)
                    active_idx = _active_player_index(adapter)
                    trace.leaf_q_perspective = float(value[active_idx])
                    self.traces.append(trace)
                    self._traces_collected_this_move += 1
                continue
            self._rust_player.add_virtual_loss(idx)
            leaves.append(idx)
            # Encode each leaf via the Rust-side BaseGame -> adapter shim.
            rust_game = self._rust_player.get_game_for_idx(idx)
            adapter = RustGameAdapter(rust_game)
            encoded_states.append(_rust_encode(adapter))
            leaf_active_player_idx.append(_active_player_index(adapter))
            leaf_traces.append(trace)
            if trace is not None:
                self._traces_collected_this_move += 1
        select_duration = time.time() - select_start
        self.cumulative_leaf_selection_time += select_duration

        if not leaves:
            return

        # Inference (mirroring MCTSPlayer.tree_search backend selection).
        inference_start = time.time()
        backend = self._backend
        backend_is_client = backend is not self.network
        if backend_is_client:
            move_probs, _, values = backend.run_many_encoded(encoded_states)
        else:
            with torch.no_grad():
                move_probs, _, values = backend.run_many_encoded(encoded_states)
        inference_duration = time.time() - inference_start
        self.cumulative_inference_time += inference_duration

        # Phase 4c: pull batched continuous-price head outputs (transformer
        # model only) so the Rust MCTS can sample prices via PW.
        batched_price_components = getattr(backend, "last_price_components", None)

        backup_start = time.time()
        for i, (idx, probs, value) in enumerate(zip(leaves, move_probs, values)):
            self._rust_player.revert_virtual_loss(idx)
            probs_np = (
                probs.detach().cpu().numpy()
                if isinstance(probs, torch.Tensor)
                else np.asarray(probs)
            ).astype(np.float32)
            value_np = (
                value.detach().cpu().numpy()
                if isinstance(value, torch.Tensor)
                else np.asarray(value)
            ).astype(np.float32)
            leaf_pc = _slice_price_components(batched_price_components, i)
            pc_arg = _coerce_price_components_for_rust(leaf_pc)
            self._rust_player.incorporate_results(
                idx, probs_np, value_np, pc_arg
            )

            # Phase 1 PlayoutTrace finalization for the non-terminal leaf.
            trace = leaf_traces[i] if i < len(leaf_traces) else None
            if trace is not None:
                trace.nn_value = value_np.copy()
                active = leaf_active_player_idx[i]
                trace.leaf_q_perspective = float(value_np[active])
                trace.leaf_terminal = False
                trace.expansion_occurred = True  # new leaf reached on first incorp
                # Prior entropy on the legal slice. Pull from the network's
                # softmaxed policy restricted to this leaf's legal indices.
                legal_indices = list(
                    self._rust_player.legal_action_indices_for_idx(idx)
                )
                if legal_indices:
                    legal_probs = probs_np[legal_indices]
                    s = float(legal_probs.sum())
                    if s > 1e-9:
                        legal_probs = legal_probs / s
                        from rl18xx.agent.alphazero.mcts import calculate_entropy
                        trace.leaf_prior_entropy = float(calculate_entropy(legal_probs))
                self.traces.append(trace)
        self.cumulative_backup_time += time.time() - backup_start

    def adaptive_readouts(self) -> int:
        num_legal = len(self._rust_player.legal_action_indices_at_root())
        if num_legal <= self.config.adaptive_readout_threshold:
            return self.config.min_readouts
        return self.config.num_readouts

    def suggest_move(self, override_readouts: Optional[int] = None) -> int:
        legal = self._rust_player.legal_action_indices_at_root()
        if len(legal) == 1:
            return int(legal[0])
        readouts = (
            override_readouts if override_readouts is not None else self.adaptive_readouts()
        )
        target = self._rust_player.n_at_root() + readouts
        while self._rust_player.n_at_root() < target:
            self.tree_search()
        return self.pick_move()

    def pick_move(self) -> int:
        legal = self._rust_player.legal_action_indices_at_root()
        if len(legal) == 1:
            return int(legal[0])
        return int(self._rust_player.pick_best_action())

    def play_move(self, action_index: int) -> bool:
        rust_root_game = self._rust_player.root_game_object()
        move_idx = len(self.searches_pi)
        temperature = (
            1.0 if move_idx < self.config.softpick_move_cutoff else 0.0
        )

        # Snapshot the search-policy vector before advancing the root.
        pi = np.asarray(
            self._rust_player.pi_at_root(temperature), dtype=np.float32
        )
        self.searches_pi.append(pi)

        # Native price range for the chosen slot (replaces the Python
        # ActionMapper.get_legal_actions_factored price-range lookup).
        price_range = rust_root_game.price_range_for_index(int(action_index))

        # Capture price-head training targets BEFORE the root is rebased so the
        # per-(slot, price) visit counts under ``action_index`` are still
        # reachable. Mirrors ``MCTSPlayer._extract_price_targets``.
        price_targets_this_move = self._extract_price_targets(
            rust_root_game, action_index, price_range
        )

        # Snapshot the action log length BEFORE rebasing so we can recover the
        # chosen action + forced-chain dicts the Rust apply logged natively.
        raw_before = len(rust_root_game.raw_actions)

        # Rebase the Rust tree to the chosen child. ``advance_root`` selects the
        # most-visited PW price internally and applies the action natively,
        # logging a faithful, replayable dict to ``raw_actions`` — so the chosen
        # action dict and any forced-chain dicts are recovered straight from the
        # log, with no Python ActionMapper decode.
        self._rust_player.advance_root(action_index)

        new_root_game = self._rust_player.root_game_object()
        new_raw = list(new_root_game.raw_actions)
        action_dict = new_raw[raw_before] if len(new_raw) > raw_before else None
        forced_dicts = (
            new_raw[raw_before + 1:] if len(new_raw) > raw_before + 1 else []
        )
        self.played_actions.append(action_dict)
        self.forced_action_dicts.append(forced_dicts)
        self.price_targets.append(price_targets_this_move)

        # Reset Phase 1 trace-budget counter for the next move's tree_search.
        self._traces_collected_this_move = 0

        LOGGER.debug(
            "RustMCTSPlayer.play_move: action_index=%d, forced_chain_len=%d, "
            "n_at_root_after=%.0f price_targets=%d",
            int(action_index),
            len(forced_dicts),
            float(self._rust_player.n_at_root()),
            len(price_targets_this_move),
        )
        return True

    def _extract_price_targets(
        self,
        rust_root_game,
        action_index: int,
        price_range,
    ) -> list:
        """Build per-grandchild price-head training targets for the picked slot.

        Mirrors ``MCTSPlayer._extract_price_targets`` (Python path) but reads
        grandchild visit counts from the Rust tree via
        ``price_grandchildren_at_root``. Returns an empty list for
        categorical-only / fixed-price slots and on lookup errors.

        The price-head slot + legal range come from the native Rust
        ``price_head_slot_for_index`` (replacing the Python
        ``ActionMapper.price_head_slot_for_action``).
        """
        if price_range is None or price_range[0] == price_range[1]:
            return []
        try:
            slot_info = rust_root_game.price_head_slot_for_index(int(action_index))
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning(
                "price_head_slot_for_index raised for action_index=%s: %s",
                action_index, exc,
            )
            return []
        if slot_info is None:
            return []
        slot_index, price_min, price_max = slot_info
        try:
            grand = self._rust_player.price_grandchildren_at_root()
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning(
                "price_grandchildren_at_root raised for action_index=%s: %s",
                action_index, exc,
            )
            return []
        slot_grandchildren = grand.get(int(action_index), {})
        if not slot_grandchildren:
            return []
        total_visits = float(sum(slot_grandchildren.values()))
        if total_visits <= 0:
            return []
        return [
            (
                int(slot_index),
                float(price),
                float(n) / total_visits,
                float(price_min),
                float(price_max),
            )
            for price, n in slot_grandchildren.items()
            if n > 0
        ]

    def inject_noise(self):
        """Inject Dirichlet noise on the root prior (mirrors MCTSNode.inject_noise)."""
        w = float(getattr(self.config, "dirichlet_noise_weight", 0.0) or 0.0)
        if w <= 0.0:
            return
        concentration = float(
            getattr(self.config, "dirichlet_noise_concentration", 10.0)
        )
        self._rust_player.inject_noise(w, concentration)

    # ------------------------------------------------------------- finalization
    def is_done(self) -> bool:
        if not np.array_equal(self.result, np.zeros_like(self.result)):
            return True
        if self._is_root_terminal():
            return True
        # Truncation: the Rust BaseGame's move counter relative to max_game_length.
        rust_root_game = self._rust_player.root_game_object()
        return int(rust_root_game.move_number) >= int(self.config.max_game_length)

    def set_result(self, result):
        self.result = np.array(result)
        # Build a result string mirroring MCTSNode.game_result_string for parity
        # with the Python MCTSPlayer surface.
        rust_root_game = self._rust_player.root_game_object()
        adapter = RustGameAdapter(rust_root_game)
        try:
            r = adapter.result()
            winning_score = max(r.values())
            winners = [str(pid) for pid, score in r.items() if score == winning_score]
            self.result_string = f"{', '.join(winners)} ({winning_score})"
        except Exception as e:
            LOGGER.warning("set_result: building result_string failed: %s", e)
            self.result_string = None

    def _root_q_vector_for_resign(self) -> list[float]:
        """Return the per-player root Q vector. Indirection so tests can
        monkeypatch a known Q vector without poking at the PyO3 Rust object
        (which doesn't allow attribute injection)."""
        return list(self._rust_player.root_q_vector())

    def check_resign(self) -> tuple[bool, Optional[dict]]:
        """Phase 2 multiplayer consensus resign on the Rust MCTS path.

        Mirrors ``MCTSPlayer.check_resign`` exactly — reads the per-player
        root Q vector from the Rust tree via ``root_q_vector()``, appends to
        the rolling window, and checks (stable leader + decisive gap) over
        the full window. In holdout games the conditions are still observed
        and stashed in ``self._would_have_resigned_info`` so the calibrator
        can compute fp_rate.
        """
        if not bool(getattr(self.config, "enable_resign", False)):
            return False, None

        num_players = len(self._rust_player.root_game_object().players)
        q_list = self._root_q_vector_for_resign()
        # root_q_vector is already truncated to num_players in Rust; defensive
        # truncate here too in case num_players grew between calls.
        q = np.asarray(q_list[:num_players], dtype=np.float32).copy()
        self._q_window.append(q)

        window_target = max(1, int(self.config.resign_window))
        if len(self._q_window) < window_target:
            return False, None

        leaders = {int(np.argmax(vec)) for vec in self._q_window}
        if len(leaders) != 1:
            return False, None
        leader = next(iter(leaders))

        q_leader_min = float(min(vec[leader] for vec in self._q_window))
        if q_leader_min < float(self.config.resign_high_threshold):
            return False, None

        def _gap(vec: np.ndarray, leader_idx: int) -> float:
            other = np.delete(vec, leader_idx)
            if other.size == 0:
                return float("inf")
            return float(vec[leader_idx] - float(other.max()))

        gap_min = min(_gap(vec, leader) for vec in self._q_window)
        if gap_min < float(self.config.resign_gap_threshold):
            return False, None

        info = {
            "leader": leader,
            "q_leader_min": q_leader_min,
            "gap_min": float(gap_min),
            "move_number": int(self._rust_player.root_game_object().move_number),
            "window_size": len(self._q_window),
        }
        if self._would_have_resigned_info is None:
            self._would_have_resigned_info = dict(info)
        if self._noresign_holdout:
            return False, info
        return True, info

    def dump_traces(self, output_dir=None):
        """Write accumulated Phase 1 PlayoutTraces to a JSONL file.

        Mirrors ``MCTSPlayer.dump_traces`` so traced games on the Rust MCTS
        path land in the same on-disk shape as the Python path
        (``{output_dir}/{iteration}/{game_id}.jsonl`` with a header line +
        one trace per JSONL row). Returns the file path, or ``None`` if
        tracing was off or no traces were collected.
        """
        import json as _json
        from pathlib import Path as _Path

        if not self._tracing_enabled or not self.traces:
            return None
        trace_cfg = getattr(self.config, "trace", None)
        if trace_cfg is None:
            return None
        base = _Path(output_dir) if output_dir is not None else _Path(trace_cfg.output_dir)
        iteration_dir = base / str(int(self.config.global_step))
        iteration_dir.mkdir(parents=True, exist_ok=True)
        path = iteration_dir / f"{self.config.game_id}.jsonl"

        root_game = self._rust_player.root_game_object()
        root_adapter = RustGameAdapter(root_game)
        players = []
        for p in sorted(root_adapter.players, key=lambda x: x.id):
            players.append({"id": p.id, "name": getattr(p, "name", str(p.id))})
        header = {
            "kind": "header",
            "iteration": int(self.config.global_step),
            "game_idx_in_iteration": int(self.config.game_idx_in_iteration),
            "game_id": str(self.config.game_id),
            "players": players,
            "trace_config": {
                "trace_game_rate": float(trace_cfg.trace_game_rate),
                "trace_every_n_moves": int(trace_cfg.trace_every_n_moves),
                "traces_per_move": int(trace_cfg.traces_per_move),
            },
            "num_traces": len(self.traces),
            "engine": "rust",  # distinguish from MCTSPlayer-produced traces
        }
        with path.open("w") as f:
            f.write(_json.dumps(header))
            f.write("\n")
            for trace in self.traces:
                f.write(_json.dumps(trace.to_jsonable()))
                f.write("\n")
        LOGGER.info(f"Wrote {len(self.traces)} PlayoutTraces to {path}")
        return path

    # --------------------------------------------------------- data extract
    def extract_data(
        self,
    ) -> Generator[
        Tuple[RustGameAdapter, torch.Tensor, torch.Tensor, torch.Tensor, list],
        None,
        None,
    ]:
        """Mirror ``MCTSPlayer.extract_data``.

        ``played_actions`` and ``forced_action_dicts`` are tracked the same
        way the Python player tracks them, so this method is structurally
        identical to ``MCTSPlayer.extract_data``.
        """
        n_chosen = len(self.searches_pi)
        chosen_actions = self.played_actions[-n_chosen:] if n_chosen > 0 else []
        assert not np.array_equal(self.result, np.zeros_like(self.result)), (
            f"result {self.result} is 0"
        )

        if len(self.price_targets) < n_chosen:
            pad = [[]] * (n_chosen - len(self.price_targets))
            self.price_targets.extend(pad)
        if len(self.forced_action_dicts) < n_chosen:
            pad = [[]] * (n_chosen - len(self.forced_action_dicts))
            self.forced_action_dicts.extend(pad)

        result = torch.tensor(self.result)
        game_state = self.get_new_game_state()
        # Replay any pre-MCTS seed actions on the fresh game state first.
        seed_actions = (
            self.played_actions[:-n_chosen] if n_chosen > 0 else list(self.played_actions)
        )
        for seed_action in seed_actions:
            game_state = game_state.pickle_clone()
            game_state.process_action(seed_action)
        for i, action in enumerate(chosen_actions):
            yield (
                game_state,
                torch.tensor(game_state._game.factored_legal_indices()),
                torch.tensor(self.searches_pi[i])
                if isinstance(self.searches_pi[i], np.ndarray)
                else self.searches_pi[i],
                result,
                self.price_targets[i],
            )
            game_state = game_state.pickle_clone()
            game_state.process_action(action)
            for forced_action in self.forced_action_dicts[i]:
                game_state.process_action(forced_action)


class _RootShim:
    """Minimal stand-in for ``MCTSNode`` consumed by ``SelfPlay.play()``.

    ``SelfPlay.play()`` reads ``player.root.{game_object, num_legal_actions,
    N, is_done, game_object.end_game, game_object.move_number, game_object.round, ...}``
    in several places. The Rust path doesn't expose a Python MCTSNode, but we
    can synthesize a thin object that delegates to the Rust player for the
    fields ``play()`` actually touches.
    """

    def __init__(self, player: "RustMCTSPlayer"):
        self._player = player

    @property
    def game_object(self):
        rust_game = self._player._rust_player.root_game_object()
        return RustGameAdapter(rust_game)

    @property
    def player_mapping(self) -> dict:
        # Mirror MCTSNode.player_mapping: {player_id: slot_index} sorted by id.
        players = sorted(self.game_object.players, key=lambda p: p.id)
        return {p.id: i for i, p in enumerate(players)}

    @property
    def num_legal_actions(self) -> int:
        return len(self._player._rust_player.legal_action_indices_at_root())

    @property
    def N(self) -> float:
        return float(self._player._rust_player.n_at_root())

    def is_done(self) -> bool:
        if self._player._is_root_terminal():
            return True
        return (
            int(self.game_object.move_number) >= int(self._player.config.max_game_length)
        )

    def inject_noise(self):
        self._player.inject_noise()

    def select_leaf(self, *args, **kwargs):
        # Returned by SelfPlay.play()'s "must run this once at the start to
        # expand the root node" preamble. The Rust path handles root expansion
        # via the first ``tree_search`` call, so this is a no-op surrogate
        # that callers downstream will then ``ensure_encoded`` / inference on.
        # See _LeafShim below.
        return _LeafShim(self._player)

    def game_result(self):
        """Return per-player value vector for the terminal root (mirrors
        ``MCTSNode.game_result``). Used by ``SelfPlay.play()`` after is_done()."""
        return self._player._compute_terminal_value(0)

    def game_result_string(self) -> Optional[str]:
        return self._player.result_string


class _LeafShim:
    """First-node leaf shim used by ``SelfPlay.play()``'s preamble.

    ``SelfPlay.play()`` does:
        first_node = player.root.select_leaf()
        first_node.ensure_encoded()
        ... run_encoded(first_node.encoded_game_state) ...
        first_node.incorporate_results(probs, val, first_node, ...)

    On the Rust path the root expansion happens inside ``tree_search`` via
    ``select_leaf`` + ``incorporate_results`` on the Rust side, so the leaf
    shim's methods drive that flow once.
    """

    def __init__(self, player: "RustMCTSPlayer"):
        self._player = player
        self.encoded_game_state = None

    def ensure_encoded(self):
        rust_game = self._player._rust_player.root_game_object()
        adapter = RustGameAdapter(rust_game)
        self.encoded_game_state = _rust_encode(adapter)

    def incorporate_results(self, probs, value, up_to=None, price_components=None):
        # Drive the Rust expansion at the arena root (idx 0).
        if isinstance(probs, torch.Tensor):
            probs_np = probs.detach().cpu().numpy().astype(np.float32)
        else:
            probs_np = np.asarray(probs, dtype=np.float32)
        if isinstance(value, torch.Tensor):
            value_np = value.detach().cpu().numpy().astype(np.float32)
        else:
            value_np = np.asarray(value, dtype=np.float32)
        pc_arg = _coerce_price_components_for_rust(price_components)
        # Re-select the leaf (the root) and incorporate.
        idx = self._player._rust_player.select_leaf()
        self._player._rust_player.incorporate_results(idx, probs_np, value_np, pc_arg)
