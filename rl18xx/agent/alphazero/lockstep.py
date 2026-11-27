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

A network that reads positions rather than the legacy encoding -- a title-generic
system, :class:`GenericEvaluator` (``takes_games``) -- gets its trees' leaves as games
(``engine_rs.mcts_collect_leaf_games``: the same selection, log-free clones instead of
encodings) and its policy requests' games; its values come back in the generic
encoder's frame and are moved into each tree's (:func:`to_tree_frames`). Legacy
evaluators take the unchanged path.
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
    # A games evaluator's frames (``takes_games``): each state's values have the player at seat
    # ``rotations[b]`` of the seating order first; the searcher moves leaves' values into their tree's frame.
    # None: the tree's own canonical frame (the legacy encoder's).
    rotations: Optional[np.ndarray] = None
    # With ``rotations``: each state's seating order (player ids, ``BaseGame.players``) when it may differ from
    # the players sorted by id (the tree's order; generator titles -- the 1822 family reorders its players).
    seat_ids: Optional[list] = None
    # A generator title's price outputs (1867, the 1822 family): per priced legal row, its position in the flat
    # legal list (``priors``' order, ascending) and its NUM_CELLS cell log-probabilities, in step units over the
    # move's own range -- not the legacy price head's slots (``price_logits`` / ``slot_index`` stay None).
    price_rows: Optional[np.ndarray] = None
    price_row_logits: Optional[np.ndarray] = None


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


_PRICE_SLOT: dict = {}  # layout index -> the legacy price head's slot (None: no slot); static per index


def _price_slot(game, index: int) -> Optional[int]:
    if index not in _PRICE_SLOT:
        slot = game.price_head_slot_for_index(int(index))
        _PRICE_SLOT[index] = None if slot is None else int(slot[0])
    return _PRICE_SLOT[index]


class GenericEvaluator:
    """A title-generic system (``model_generic`` checkpoints) as a lockstep network: positions go to it as
    games (``takes_games``: the searcher collects leaves with ``engine_rs.mcts_collect_leaf_games`` instead of
    encoding them for the legacy model). :meth:`send_games` builds the generic inputs in the worker
    (``model_generic.build_inputs``, maps de-duplicated) and sends them to the policy's client and, for a
    separate critic, the same arrays to the value's client -- ``generic_inference.GenericInferenceClient`` s
    (a generic inference server each, on the GPU in a real eval) or ``LocalGenericClient`` s (in-process:
    tests, CPU). :meth:`receive` answers in the legacy search's terms: the legal priors at temperature 1
    (the tree applies ``prior_temperature``), each priced row's cell log-probabilities in the legacy price
    head's slot row (``price_pmf.SLOT_INDEX``; for 1830 a generic priced row's cells are its slot's), and the
    value head's output -- ``value_head`` win, score or blend=<w>, of the critic or else the policy network's
    own heads -- in the generic encoder's frame (``EvalResult.rotations``)."""

    takes_games = True

    def __init__(self, policy_client, value_client=None, value_head: str = "win", compress: bool = True):
        from rl18xx.agent.alphazero.composite_model import parse_value_head

        self.policy_client = policy_client
        self.value_client = value_client
        self.kind, self.blend = parse_value_head(value_head)
        self.compress = compress
        self.states = 0

    def send_games(self, games: list, legal: list, price_ranges: Optional[list] = None):
        """``price_ranges``: per position its legal moves' ``(lo, hi, step)`` rows when known (a generator
        title's search leaves: ``mcts_collect_leaf_games``'s ``price_ranges``), else None -- then
        ``model_generic.legal_actions`` reads them from the engine."""
        from rl18xx.agent.alphazero import model_generic as mg

        if price_ranges is None:
            price_ranges = [None] * len(games)
        legal_actions = [
            mg.legal_actions(g, ix) if r is None else mg.legal_actions_with_ranges(g, ix, r)
            for g, ix, r in zip(games, legal, price_ranges)
        ]
        arrays = mg.build_inputs(games, legal_actions, dedupe_maps=True)
        sent = self.policy_client.send_inputs(arrays, want_price_logp=True, compress=self.compress)
        critic = None if self.value_client is None else self.value_client.send_inputs(arrays, compress=self.compress)
        self.states += len(games)
        return games, legal_actions, arrays, sent, critic

    def receive(self, pending) -> EvalResult:
        from rl18xx.agent.alphazero import price_pmf

        games, legal_actions, arrays, sent, critic = pending
        reply = self.policy_client.receive_outputs(sent)
        values_reply = reply if critic is None else self.value_client.receive_outputs(critic)
        win, score = values_reply.win, values_reply.score
        if self.kind == "win":
            values = win
        elif self.kind == "score":
            values = score
        else:
            values = (1.0 - self.blend) * win + self.blend * score
        if games and getattr(legal_actions[0], "price_grid", False):
            # A generator title (1867, the 1822 family): price rows per legal move, no legacy slots.
            empty = np.zeros((0, price_pmf.NUM_CELLS), dtype=np.float32)
            return EvalResult(
                priors=np.exp(np.asarray(reply.logp, dtype=np.float64)).astype(np.float32),
                values=np.ascontiguousarray(values, dtype=np.float32),
                rotations=np.asarray(arrays["meta"][:, 5], dtype=np.int64),
                seat_ids=[[int(p.id) for p in getattr(g, "_game", g).players] for g in games],
                price_rows=np.asarray(arrays["price_rows"], dtype=np.int64),
                price_row_logits=(
                    np.ascontiguousarray(reply.price_logp, dtype=np.float32)
                    if reply.price_logp is not None and len(arrays["price_rows"])
                    else empty
                ),
            )
        prices = np.zeros((len(games), price_pmf.NUM_SLOTS, price_pmf.NUM_CELLS), dtype=np.float32)
        if reply.price_logp is not None and len(arrays["price_rows"]):
            offsets, positions = arrays["legal_offsets"], arrays["legal_pos"]
            for j, row in enumerate(arrays["price_rows"]):
                b = int(positions[row])
                slot = _price_slot(games[b], int(legal_actions[b].indices[int(row - offsets[b])]))
                if slot is not None:
                    prices[b, slot] = reply.price_logp[j]
        return EvalResult(
            priors=np.exp(np.asarray(reply.logp, dtype=np.float64)).astype(np.float32),
            values=np.ascontiguousarray(values, dtype=np.float32),
            price_logits=prices,
            slot_index=dict(price_pmf.SLOT_INDEX),
            rotations=np.asarray(arrays["meta"][:, 5], dtype=np.int64),
        )


def to_tree_frames(
    values: np.ndarray,
    rotations: np.ndarray,
    tree_rotations: np.ndarray,
    num_players: int,
    seat_ids: Optional[list] = None,
):
    """Values with seat ``rotations[b]`` of the seating order first, re-rotated so seat ``tree_rotations[b]``
    is first (the frame ``mcts_apply_results`` unrotates by); padded seats untouched. The tree numbers the
    players sorted by id; ``seat_ids[b]`` (optional) is the seating order the generic frame rotates over, when it
    may differ from that (the 1822 family reorders its players): such a row is mapped through the player ids."""
    out = np.array(values, dtype=np.float32, copy=True)
    n = num_players
    reordered = set()
    if seat_ids is not None:
        for b, ids in enumerate(seat_ids):
            ids = [int(i) for i in ids]
            order = sorted(ids)
            if ids == order:
                continue  # the tree's order: a rotation (below)
            reordered.add(b)
            absolute = np.zeros(n, dtype=np.float32)
            for k in range(n):
                absolute[order.index(ids[(k + int(rotations[b])) % n])] = values[b, k]
            out[b, :n] = absolute[(np.arange(n) + int(tree_rotations[b])) % n]
    for b in np.flatnonzero(np.asarray(rotations) != np.asarray(tree_rotations)):
        if b in reordered:
            continue
        shift = int(tree_rotations[b] - rotations[b]) % num_players
        out[b, :num_players] = np.roll(values[b, :num_players], -shift)
    return out


def evaluator_for(backend):
    """The evaluator for an inference client or an in-process network."""
    if isinstance(backend, (ClientEvaluator, ModelEvaluator)) or getattr(backend, "takes_games", False) is True:
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
    count for q, not toward the schedule).

    ``root_noise`` -- AlphaZero's exploration noise: once the root is expanded
    (at once for a kept subtree), its priors are blended with Dirichlet noise at
    the player's ``config.dirichlet_noise_weight`` (the Rust tree's ``inject_noise``;
    alpha = ``dirichlet_noise_concentration`` / legal moves) before the search's
    own visits. Off by default (an eval's search never), and never on a Gumbel
    root (its Gumbel noise replaces it)."""

    def __init__(
        self, player, readouts: int, key: Hashable = None, budget: str = "additional", root_noise: bool = False
    ):
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
        self.root_noise = bool(root_noise)
        self.noised = False  # the root's priors carry this search's noise
        self.target: Optional[float] = None
        # Root visits the search started from (a kept subtree's; 0 for a fresh root).
        self.inherited = float(self.tree.n_at_root()) if self.tree.is_expanded(self.tree.root_idx) else 0.0
        self.done = False
        self.update()

    def _begin_if_expanded(self) -> None:
        if self.target is not None or not self.tree.is_expanded(self.tree.root_idx):
            return
        if self.root_noise and not self.gumbel:
            config = self.player.config
            weight = float(getattr(config, "dirichlet_noise_weight", 0.0) or 0.0)
            if weight > 0:
                self.tree.inject_noise(weight, float(getattr(config, "dirichlet_noise_concentration", 10.0)))
                self.noised = True
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
                if getattr(self.evaluators[key], "takes_games", False) is True:
                    # A network over games (GenericEvaluator): the leaves as games, no legacy encoding.
                    batch = engine_rs.mcts_collect_leaf_games(
                        trees,
                        [s.leaves_wanted() for s in group],
                        [s.dedupe for s in group],
                        [s.use_score_values for s in group],
                    )
                    in_flight.append([key, group, trees, batch, reqs, None])
                    stats.terminal_backups += int(batch["terminal"].sum())
                    legal = split_legal(batch["legal"], batch["legal_lengths"])
                    n_leaves = len(legal)
                    games = list(batch["games"]) + [r.game for r in reqs]
                    legal = legal + [np.asarray(r.legal, dtype=np.int32) for r in reqs]
                    stats.maps += len(games)
                    if not legal:
                        continue  # only terminal leaves this round
                    stats.requests += 1
                    stats.leaves += n_leaves
                    stats.policy_states += len(reqs)
                    if "price_ranges" in batch:
                        # A generator title's leaves come with their moves' price ranges (no second generation).
                        ranges = split_legal(batch["price_ranges"], batch["legal_lengths"]) + [None] * len(reqs)
                        sent = self.evaluators[key].send_games(games, legal, ranges)
                    else:
                        sent = self.evaluators[key].send_games(games, legal)
                    in_flight[-1][5] = (sent, n_leaves)
                    continue
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
                leaf_values = result.values[:n_leaves]
                if result.rotations is not None and n_leaves:
                    leaf_values = to_tree_frames(
                        leaf_values,
                        result.rotations[:n_leaves],
                        batch["rotations"],
                        int(batch["num_players"]),
                        None if result.seat_ids is None else result.seat_ids[:n_leaves],
                    )
                if result.price_rows is not None:
                    # Per-move price rows (a generator title): the leaves' rows go to their trees.
                    leaf_rows = result.price_rows < n_priors
                    engine_rs.mcts_apply_results(
                        trees,
                        result.priors[:n_priors],
                        leaf_values,
                        [s.prior_temperature for s in group],
                        None,
                        None,
                        np.ascontiguousarray(result.price_rows[leaf_rows], dtype=np.int64),
                        np.ascontiguousarray(result.price_row_logits[leaf_rows], dtype=np.float32).reshape(
                            -1, result.price_row_logits.shape[-1]
                        ),
                    )
                else:
                    engine_rs.mcts_apply_results(
                        trees,
                        result.priors[:n_priors],
                        leaf_values,
                        [s.prior_temperature for s in group],
                        None if price is None or not n_leaves else np.ascontiguousarray(price[:n_leaves]),
                        result.slot_index if price is not None and n_leaves else None,
                    )
                entry[1] = None  # applied
                offset = n_priors
                for j, r in enumerate(reqs):
                    n = len(r.legal)
                    r.priors = result.priors[offset:offset + n]
                    r.value = result.values[n_leaves + j]
                    if result.price_rows is not None:
                        # {legal row: cell log-probabilities} of the request's priced moves
                        mine = np.flatnonzero((result.price_rows >= offset) & (result.price_rows < offset + n))
                        r.price_row = {int(result.price_rows[i]) - offset: result.price_row_logits[i] for i in mine}
                    else:
                        r.price_row = None if price is None else price[n_leaves + j]
                    offset += n
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
