"""Many Rust search trees in lockstep in one worker.

``RustMCTSPlayer.suggest_move`` searches one tree at a time: each batch of
``parallel_readouts`` leaves is one network request, and every leaf is
wrapped in a ``RustGameAdapter`` and encoded in Python. A worker that runs
several games (an eval's games, a self-play generator's) instead hands all
their pending searches to :class:`LockstepSearcher`: each round, every tree
selects its next batch exactly as ``tree_search`` would, all leaves are
encoded natively into one stacked array per network
(``engine_rs.mcts_collect_leaves``), one request per network carries the
leaves of every tree on it (sent to all networks before waiting on any), and
the replies -- each leaf's legal priors, value and price-head row -- are
scattered back into the trees in Rust (``engine_rs.mcts_apply_results``).
Python does per-round, not per-leaf, work.

Usage::

    searcher = LockstepSearcher({"main": evaluator_for(client)})
    searches = [player.start_search(key="main") for player in players]
    searcher.run([s for s in searches if s is not None])  # or .round() repeatedly
    moves = [s.player.pick_move() for s in searches if s is not None]

A search is the same as ``suggest_move``'s on that tree: the same batches,
virtual losses, ``dedupe_leaves``, terminal values, ``prior_temperature``,
budget (``adaptive_readouts``) and stopping rule, and with ``gumbel_k`` the
Gumbel root's exact budget. One difference: a root that is not yet expanded
is expanded by a one-leaf round first (the eval's ``_expand_root``), where
``tree_search`` on an unexpanded root would batch copies of the root.
:class:`PolicyRequest` puts a position that needs only the network's policy
(a policy player's move) into the same round. Not supported here (use
``suggest_move``): ``leaf_value_heuristic`` and PlayoutTrace recording.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Hashable, Optional

import numpy as np
import torch

import engine_rs
from rl18xx.agent.alphazero.inference_server import InferenceClient, PackedStates, run_packed_states


# ----------------------------------------------------------------------------- evaluators


@dataclass
class EvalResult:
    """A network's outputs for one packed request, as numpy."""

    priors: np.ndarray  # float32: each state's legal priors back to back
    values: np.ndarray  # (B, VALUE_SIZE) float32, each state's canonical frame
    price_logits: Optional[np.ndarray] = None  # (B, slots, cells) float32
    slot_index: Optional[dict] = None


class ClientEvaluator:
    """An inference-server client: requests go out on :meth:`send` and are
    waited for on :meth:`receive`, so a worker can keep several servers busy."""

    def __init__(self, client: InferenceClient):
        self.client = client

    def send(self, packed: PackedStates):
        return self.client.send_packed(packed)

    def receive(self, pending) -> EvalResult:
        reply = self.client.receive_raw(pending)
        if reply.legal_lengths is None:
            raise RuntimeError("lockstep requests carry legal indices; the server sent dense priors")
        pc = reply.price_components
        return EvalResult(
            priors=np.ascontiguousarray(reply.probs, dtype=np.float32),
            values=np.ascontiguousarray(reply.value, dtype=np.float32),
            price_logits=None if pc is None else np.ascontiguousarray(pc["price_logits"], dtype=np.float32),
            slot_index=None if pc is None else pc.get("slot_index"),
        )


class ModelEvaluator:
    """An in-process network (tests, CPU benchmarks): evaluated on :meth:`send`.
    ``forward_seconds`` accumulates the time inside the network call."""

    def __init__(self, network):
        self.network = network
        self.forward_seconds = 0.0
        self.states = 0

    def send(self, packed: PackedStates) -> EvalResult:
        legal = packed.legal_indices
        start = time.perf_counter()
        with torch.no_grad():
            probs, _, values = run_packed_states(self.network, [packed])
            lengths = np.fromiter((len(ix) for ix in legal), dtype=np.int64, count=len(legal))
            rows = np.repeat(np.arange(len(legal)), lengths)
            cols = np.concatenate(legal).astype(np.int64)
            if isinstance(probs, torch.Tensor) and probs.device.type != "cpu":
                index = (torch.from_numpy(rows).to(probs.device), torch.from_numpy(cols).to(probs.device))
                priors = probs[index].float().cpu().numpy()
            elif isinstance(probs, torch.Tensor):
                priors = probs.numpy()[rows, cols]
            else:
                probs = np.asarray(probs, dtype=np.float32)
                priors = np.concatenate([probs[i, ix] for i, ix in enumerate(legal)]).astype(np.float32)
            values = values.detach().float().cpu().numpy() if isinstance(values, torch.Tensor) else np.asarray(values)
            price = getattr(self.network, "last_price_components", None)
            logits = None if price is None else price.get("price_logits")
            if isinstance(logits, torch.Tensor):
                logits = logits.detach().float().cpu().numpy()
        self.forward_seconds += time.perf_counter() - start
        self.states += len(packed)
        return EvalResult(
            priors=np.ascontiguousarray(priors, dtype=np.float32),
            values=np.ascontiguousarray(values, dtype=np.float32),
            price_logits=None if logits is None else np.ascontiguousarray(logits, dtype=np.float32),
            slot_index=None if price is None else price.get("slot_index"),
        )

    def receive(self, pending: EvalResult) -> EvalResult:
        return pending


def evaluator_for(backend):
    """The evaluator for an inference client or an in-process network."""
    if isinstance(backend, (ClientEvaluator, ModelEvaluator)):
        return backend
    if isinstance(backend, InferenceClient):
        return ClientEvaluator(backend)
    return ModelEvaluator(backend)


def pack(batch: dict, legal: list) -> PackedStates:
    """A stacked batch (``mcts_collect_leaves`` / ``encode_games`` layout) as the
    server's request format, with each state's legal indices."""
    num_players = int(batch["num_players"])
    extras = [(int(rt), 0, int(rot), num_players) for rt, rot in zip(batch["round_types"], batch["rotations"])]
    return PackedStates(
        game_states=batch["game_states"],
        node_features=batch["node_features"],
        extras=extras,
        legal_indices=legal,
        map_index=batch.get("map_index"),
    )


def _concat_batches(a: dict, b: dict) -> dict:
    """Two stacked batches (``mcts_collect_leaves`` / ``encode_games`` layout) as
    one, rows of ``a`` first. Map-indexed batches (``dedupe_maps``) keep their
    maps; ``b``'s map indices shift past ``a``'s maps."""
    out = {key: np.concatenate([a[key], b[key]]) for key in ("game_states", "node_features", "round_types", "rotations")}
    out["num_players"] = a["num_players"]
    if "map_index" in a or "map_index" in b:
        ia = a["map_index"] if "map_index" in a else np.arange(len(a["game_states"]))
        ib = b["map_index"] if "map_index" in b else np.arange(len(b["game_states"]))
        out["map_index"] = np.concatenate([ia, ib + len(a["node_features"])])
    return out


def split_legal(flat: np.ndarray, lengths: np.ndarray) -> list:
    return np.split(flat, np.cumsum(lengths)[:-1]) if len(lengths) else []


# ----------------------------------------------------------------------------- work items


BUDGETS = ("additional", "total")


class TreeSearch:
    """One search to run on a ``RustMCTSPlayer`` (the Python wrapper) -- what
    its ``suggest_move`` does, driven by a :class:`LockstepSearcher`. Make it
    with ``player.start_search``; when ``done``, ``player.pick_move()`` gives
    the move.

    ``budget`` -- how ``readouts`` counts on a tree that already has visits at
    its root (a subtree kept from the previous move): "additional" (as
    ``suggest_move``) searches ``readouts`` more, "total" until the root has
    the visits a fresh search would end with (its expansion + ``readouts``),
    the inherited ones included -- none at all if it already has them. On a
    fresh tree the two are the same search. A Gumbel root's sequential halving
    always spends ``readouts`` of its own (``gumbel.rs``: inherited visits
    count for q, not toward the schedule)."""

    def __init__(self, player, readouts: int, key: Hashable = None, budget: str = "additional"):
        config = player.config
        if config.leaf_value_heuristic is not None:
            raise NotImplementedError("lockstep search does not support leaf_value_heuristic (use suggest_move)")
        if budget not in BUDGETS:
            raise ValueError(f"budget must be one of {BUDGETS}, got {budget!r}")
        self.player = player
        self.tree = player._rust_player
        self.readouts = int(readouts)
        self.key = key
        self.budget = budget
        self.gumbel = bool(self.tree.gumbel_enabled)
        self.max_leaves = int(min(config.parallel_readouts, config.num_readouts))
        self.dedupe = bool(config.dedupe_leaves)
        self.use_score_values = bool(config.use_score_values)
        self.prior_temperature = float(config.prior_temperature)
        self.num_players = int(self.tree.num_players)
        self.target: Optional[float] = None
        # Root visits the search started from (a kept subtree's; 0 for a fresh root).
        self.inherited = float(self.tree.n_at_root()) if self.tree.is_expanded(self.tree.root_idx) else 0.0
        self.done = False
        self.update()

    def _begin_if_expanded(self) -> None:
        if self.target is not None or not self.tree.is_expanded(self.tree.root_idx):
            return
        if self.gumbel:
            self.tree.begin_gumbel_search(self.readouts)
            self.target = 0.0
        elif self.budget == "total":
            # A fresh root has 1 visit (its expansion) here: 1 + readouts, as "additional".
            self.target = max(float(self.tree.n_at_root()), 1.0 + self.readouts)
        else:
            self.target = float(self.tree.n_at_root()) + self.readouts

    def leaves_wanted(self) -> int:
        if self.target is None:
            return 1  # the root's expansion
        if self.gumbel:
            return min(self.max_leaves, int(self.tree.gumbel_remaining()))
        return self.max_leaves

    def update(self) -> None:
        """After a round: start the search once the root is expanded; done?"""
        self._begin_if_expanded()
        if self.target is None:
            return
        if self.tree.is_terminal(self.tree.root_idx):
            self.done = True
        elif self.gumbel:
            self.done = int(self.tree.gumbel_remaining()) == 0
        else:
            self.done = float(self.tree.n_at_root()) >= self.target


@dataclass
class PolicyRequest:
    """A position that needs one network evaluation, not a search: filled
    with its legal priors (in ``legal`` order), canonical value and price-head
    row by the round it takes part in."""

    game: Any  # engine_rs.BaseGame
    legal: list
    key: Hashable = None
    priors: Optional[np.ndarray] = None
    value: Optional[np.ndarray] = None
    price_row: Optional[np.ndarray] = None
    num_players: int = field(init=False)

    def __post_init__(self):
        self.num_players = len(self.game.players)

    @property
    def done(self) -> bool:
        return self.priors is not None


# ----------------------------------------------------------------------------- driver


@dataclass
class RoundStats:
    rounds: int = 0
    requests: int = 0
    leaves: int = 0
    policy_states: int = 0
    terminal_backups: int = 0
    maps: int = 0  # node-feature rows sent (= states sent, unless dedupe_maps)
    collect_seconds: float = 0.0
    wait_seconds: float = 0.0
    apply_seconds: float = 0.0


class LockstepSearcher:
    """Runs :class:`TreeSearch` es (and :class:`PolicyRequest` s) in lockstep over
    one or more networks. ``evaluators`` maps a work item's ``key`` to an
    evaluator (``evaluator_for(client_or_network)``)."""

    def __init__(self, evaluators: dict, dedupe_maps: bool = False):
        self.evaluators = {k: evaluator_for(v) for k, v in evaluators.items()}
        self.stats = RoundStats()
        # Send each distinct hex map of a request once (``PackedStates.map_index``):
        # a request's leaves mostly share their trees' maps. The network's map
        # encoder then runs once per map -- the same outputs within float
        # rounding (its matmuls see fewer rows), so off by default.
        self.dedupe_maps = bool(dedupe_maps)

    def run(self, searches: list, requests: list = ()) -> None:
        """Run every search to completion (and evaluate every request)."""
        searches = [s for s in searches if not s.done]
        requests = [r for r in requests if not r.done]
        while searches or requests:
            self.round(searches, requests)
            searches = [s for s in searches if not s.done]
            requests = []

    def round(self, searches: list, requests: list = ()) -> None:
        """One round: every unfinished search contributes one batch of leaves,
        every request its position; one network request per (network, player
        count); results applied; ``done`` updated."""
        stats = self.stats
        stats.rounds += 1
        groups: dict = defaultdict(lambda: ([], []))
        for s in searches:
            if not s.done:
                groups[(s.key, s.num_players)][0].append(s)
        for r in requests:
            if not r.done:
                groups[(r.key, r.num_players)][1].append(r)

        t0 = time.perf_counter()
        in_flight = []
        try:
            for (key, _), (group, reqs) in groups.items():
                trees = [s.tree for s in group]
                batch = engine_rs.mcts_collect_leaves(
                    trees,
                    [s.leaves_wanted() for s in group],
                    [s.dedupe for s in group],
                    [s.use_score_values for s in group],
                    self.dedupe_maps,
                )
                in_flight.append([key, group, trees, batch, reqs, None])
                stats.terminal_backups += int(batch["terminal"].sum())
                legal = split_legal(batch["legal"], batch["legal_lengths"])
                n_leaves = len(legal)
                stats.maps += len(batch["node_features"])
                if reqs:
                    extra = engine_rs.encode_games([r.game for r in reqs], self.dedupe_maps)
                    stats.maps += len(extra["node_features"])
                    batch = _concat_batches(batch, extra) if n_leaves else extra
                    legal = legal + [np.asarray(r.legal, dtype=np.int32) for r in reqs]
                if not legal:
                    continue  # only terminal leaves this round
                stats.requests += 1
                stats.leaves += n_leaves
                stats.policy_states += len(reqs)
                in_flight[-1][5] = (self.evaluators[key].send(pack(batch, legal)), n_leaves)
            t1 = time.perf_counter()
            stats.collect_seconds += t1 - t0
            for entry in in_flight:
                key, group, trees, batch, reqs, sent = entry
                if sent is None:
                    temperatures = [s.prior_temperature for s in group]
                    no_rows = np.zeros((0, 6), np.float32)
                    engine_rs.mcts_apply_results(trees, np.zeros(0, np.float32), no_rows, temperatures)
                    entry[1] = None  # applied
                    continue
                pending, n_leaves = sent
                tw = time.perf_counter()
                result = self.evaluators[key].receive(pending)
                ta = time.perf_counter()
                stats.wait_seconds += ta - tw
                n_priors = int(batch["legal_lengths"].sum())
                price = result.price_logits
                engine_rs.mcts_apply_results(
                    trees,
                    result.priors[:n_priors],
                    result.values[:n_leaves],
                    [s.prior_temperature for s in group],
                    None if price is None or not n_leaves else np.ascontiguousarray(price[:n_leaves]),
                    result.slot_index if price is not None and n_leaves else None,
                )
                entry[1] = None  # applied
                offset = n_priors
                for j, r in enumerate(reqs):
                    n = len(r.legal)
                    r.priors = result.priors[offset:offset + n]
                    offset += n
                    r.value = result.values[n_leaves + j]
                    r.price_row = None if price is None else price[n_leaves + j]
                stats.apply_seconds += time.perf_counter() - ta
        except BaseException:
            # Leave no tree with leaves pending (their virtual losses would stay).
            for _, group, trees, *_ in in_flight:
                if group is not None:
                    engine_rs.mcts_discard_pending(trees)
            raise
        for s in searches:
            if not s.done:
                s.update()
