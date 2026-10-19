"""Cross-process inference server (Phase 3 of MCTS improvements plan).

Architecture sketch::

    Coordinator process (loop.py main)            Inference server process
    ─────────────────────────────────             ─────────────────────────
     spawn inference server  ─────────────────►   STARTING → ACCEPTING
     spawn self-play workers (ProcessPool)
       workers ◄── inference requests ─────────►  collect up to N or wait T ms
                                                   forward pass
       workers ◄── inference replies ◄─────────
     join workers (self-play iter done)
     send PAUSE  ─────────────────────────────►   DRAINING (finish in-flight)
                                                   → IDLE (drop model, free VRAM)
     run training step (this process, GPU)
     write checkpoint to disk
     send RELOAD(path)  ──────────────────────►   RELOADING (load from disk)
                                                   → ACCEPTING

The ``InferenceServer`` runs in its own process and owns the model on the
GPU. ``InferenceClient`` is a thin per-worker shim with a
``run_encoded`` / ``run_many_encoded`` API that mirrors the model's
inference contract — drop-in compatible with the in-process backend used
in ``self_play.MCTSPlayer.tree_search``.

The whole pipeline is gated by ``SelfPlayConfig.use_inference_server``
(default ``False``) so the in-process inference path continues to run
self-play until parity is verified on a real training run.
"""
from __future__ import annotations

import logging
import multiprocessing as mp
import queue
import time
from concurrent.futures import FIRST_COMPLETED, wait
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class ServerState(str, Enum):
    """Server lifecycle states (see module docstring)."""

    STARTING = "STARTING"
    ACCEPTING = "ACCEPTING"
    DRAINING = "DRAINING"
    IDLE = "IDLE"
    RELOADING = "RELOADING"
    SHUTDOWN = "SHUTDOWN"


class LegalIndices:
    """Per-state legal action indices stored flat: ``flat`` (int32, every state's
    indices back to back) and ``lengths`` (int64, per state). A sequence of the
    per-state arrays (views), so it stands in for the list of arrays; code that
    knows it reads the flat arrays (one pickle each instead of one per state,
    no re-concatenation on the server)."""

    __slots__ = ("flat", "lengths", "_offsets")

    def __init__(self, flat: np.ndarray, lengths: np.ndarray):
        self.flat = np.ascontiguousarray(flat, dtype=np.int32)
        self.lengths = np.ascontiguousarray(lengths, dtype=np.int64)
        self._offsets = None

    @classmethod
    def of(cls, legal) -> "LegalIndices":
        """``legal`` as a :class:`LegalIndices` (a list of arrays is concatenated)."""
        if isinstance(legal, LegalIndices):
            return legal
        lengths = np.fromiter((len(ix) for ix in legal), dtype=np.int64, count=len(legal))
        flat = np.concatenate([np.asarray(ix, dtype=np.int32) for ix in legal]) if len(legal) else np.zeros(0, np.int32)
        return cls(flat, lengths)

    @classmethod
    def concat(cls, parts: list) -> "LegalIndices":
        parts = [cls.of(p) for p in parts]
        if len(parts) == 1:
            return parts[0]
        return cls(np.concatenate([p.flat for p in parts]), np.concatenate([p.lengths for p in parts]))

    def _offs(self) -> np.ndarray:
        if self._offsets is None:
            self._offsets = np.concatenate([[0], np.cumsum(self.lengths)])
        return self._offsets

    def __len__(self) -> int:
        return len(self.lengths)

    def __getitem__(self, i):
        if isinstance(i, slice):
            return [self[j] for j in range(*i.indices(len(self)))]
        offs = self._offs()
        return self.flat[offs[i]:offs[i + 1]]

    def __iter__(self):
        offs = self._offs()
        return (self.flat[offs[i]:offs[i + 1]] for i in range(len(self.lengths)))

    def row_ids(self) -> np.ndarray:
        """Each flat entry's state."""
        return np.repeat(np.arange(len(self.lengths), dtype=np.int64), self.lengths)

    def __getstate__(self):
        return (self.flat, self.lengths)

    def __setstate__(self, state):
        self.flat, self.lengths = state
        self._offsets = None


@dataclass
class PackedStates:
    """A batch of encoded states as two stacked numpy arrays + small extras.

    Torch tensors on an ``mp.Queue`` each travel as a shared-memory segment
    passed by file descriptor — ~4.6 ms just to send 32 states (4 tensors
    each) and more to receive, which made the single server process the
    self-play bottleneck. Plain numpy pickles as bytes (~0.05 ms). Only the
    fields the transformer reads are kept; the graph edges are dropped.
    """

    game_states: np.ndarray     # (n, D) flat state vectors (one player count per pack)
    node_features: np.ndarray   # (n, num_hexes, features)
    extras: list                # per state: encoded tuple from index 4 on (round type, active, rotation[, num_players])
    # Optional per-state legal action indices. When set, the reply carries only
    # those priors (MCTS renormalizes over the legal set anyway): a few dozen
    # floats per leaf instead of a 26,537-wide vector, which made reply
    # serialization the server's bottleneck (~0.5 GB/s at 9k leaves/s).
    legal_indices: Optional[list] = None
    # Optional (n,) int64: ``node_features`` then holds each distinct hex map
    # once (m rows) and state i's map is row ``map_index[i]`` -- a lockstep
    # request's leaves mostly share their trees' maps, and the node features
    # are ~92% of a state's bytes (``engine_rs.mcts_collect_leaves(...,
    # dedupe_maps=True)``). The server's map encoder then runs once per map.
    map_index: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return len(self.extras)

    def __getstate__(self):
        # Pickled compactly (the request queue's cost is per object): the legal
        # indices as two flat arrays, the extras as one int64 array (rows iterate
        # as they did, as numpy ints).
        state = dict(self.__dict__)
        try:
            if state.get("legal_indices") is not None:
                state["legal_indices"] = LegalIndices.of(state["legal_indices"])
            extras = state.get("extras")
            if isinstance(extras, list) and extras and all(len(e) == len(extras[0]) for e in extras):
                state["extras"] = np.asarray(extras, dtype=np.int64)
        except (TypeError, ValueError):  # not plain integers: pickled as they are
            state = dict(self.__dict__)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def dense_node_features(self) -> np.ndarray:
        """``node_features`` with one row per state."""
        if self.map_index is None:
            return self.node_features
        return self.node_features[self.map_index]

    @classmethod
    def pack(cls, encoded_states: list) -> Optional["PackedStates"]:
        """Pack encoded tuples, or None if they aren't encoded tuples or
        their state widths differ (the caller then sends them as a list)."""
        if not all(isinstance(s, tuple) and len(s) >= 5 for s in encoded_states):
            return None
        rows = [_to_numpy(s[0]).reshape(-1) for s in encoded_states]
        if len({r.shape[0] for r in rows}) != 1:
            return None
        return cls(
            game_states=np.stack(rows),
            node_features=np.stack([_to_numpy(s[1]) for s in encoded_states]),
            extras=[tuple(s[4:]) for s in encoded_states],
        )

    def unpack(self) -> list:
        """Encoded tuples ``(gs (1, D), nf, None, None, *extras)`` as tensor views."""
        game_states = torch.from_numpy(self.game_states)
        node_features = torch.from_numpy(self.dense_node_features())
        return [
            (game_states[i:i + 1], node_features[i], None, None, *extra)
            for i, extra in enumerate(self.extras)
        ]


@dataclass
class InferenceRequest:
    """One inference request from a worker."""

    request_id: int
    worker_id: int
    encoded_state: Any = None  # the encoded-state tuple emitted by Encoder_1830Graph / _rust_encode
    # A whole ``run_many_encoded`` call in one message (``InferenceClient``):
    # one queue round trip per MCTS batch instead of one per leaf, as a
    # ``PackedStates`` (or a plain list of encoded tuples). Answered by a
    # single ``batched`` reply.
    encoded_states: Optional[Any] = None


@dataclass
class InferenceReply:
    """One inference result destined for a worker."""

    request_id: int
    probs: np.ndarray              # (POLICY_SIZE,) softmax probabilities
    log_probs: np.ndarray          # (POLICY_SIZE,) log-softmax
    value: np.ndarray              # (VALUE_SIZE,) per-player softmaxed win/loss
    price_components: Optional[dict]  # per-leaf sliced (1D tensors or None)
    # Sparse replies (requests with ``legal_indices``): ``probs`` is the flat
    # float32 concatenation of each leaf's legal priors, split by
    # ``legal_lengths``.
    legal_lengths: Optional[np.ndarray] = None
    # Batched replies (to ``encoded_states`` requests) stack the leaves:
    # probs (n, POLICY_SIZE) float16, value (n, VALUE_SIZE), log_probs None
    # (MCTS uses only probs; dropping log_probs and halving the dtype cuts
    # the ~210 KB/leaf payload ~4x), price_components the model's batched
    # dict as numpy (slot_index sent once).
    batched: bool = False


@dataclass
class ControlMessage:
    """Control op sent from the coordinator to the server."""

    op: str  # "pause" | "reload" | "shutdown" | "health"
    payload: Optional[dict] = None
    # In-process (thread) servers may carry a reply queue on the message. A
    # spawned server can't — multiprocessing queues only pass to a child at
    # creation — so it replies on its ``control_reply_q`` tagged with ``seq``.
    reply_q: Optional[Any] = None
    seq: Optional[int] = None


@dataclass
class HealthReport:
    """Snapshot of server state. Returned via control reply queue on ``health``."""

    state: str
    batches_served: int
    requests_served: int
    avg_batch_size: float
    avg_forward_ms: float
    queue_depth_approx: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _slice_price_components_for_reply(batched: Optional[dict], leaf_index: int) -> Optional[dict]:
    """Cheap per-leaf slice of the model's batched ``last_price_components``.

    Mirrors ``self_play._slice_price_components`` so the server-side output
    is drop-in compatible with what the in-process backend produces. The
    returned tensors are detached + moved to CPU + numpy-ized so they
    serialize cleanly over an ``mp.Queue`` (the queue would otherwise
    drag the CUDA storage into the worker process — wasteful and racy).
    """
    if batched is None:
        return None
    logits = batched.get("price_logits")
    if logits is None:
        return None
    return {
        "price_logits": logits[leaf_index].detach().float().cpu().numpy(),
        "slot_index": batched.get("slot_index"),
        "num_slots": batched.get("num_slots"),
    }


def _rows_to_numpy(batch: Any, start: int, n: int, dtype: torch.dtype) -> np.ndarray:
    """Rows ``[start, start + n)`` of a ``(B, ...)`` tensor (or list of row
    tensors) as one numpy array. The dtype conversion happens on the source
    device before a single transfer — numpy's float32→float16 cast runs at
    ~60 ns/element, ~0.4 s for a 256-leaf policy batch."""
    rows = batch[start:start + n]
    if not isinstance(rows, torch.Tensor):
        rows = torch.stack([torch.as_tensor(r) for r in rows])
    return rows.detach().to(dtype).cpu().numpy()


def _gather_legal_priors(probs: Any, start: int, legal: list) -> np.ndarray:
    """Flat float32 priors of each leaf's legal actions, gathered on the
    model's device and transferred once."""
    rows = probs[start:start + len(legal)]
    if not isinstance(rows, torch.Tensor):
        rows = torch.stack([torch.as_tensor(r) for r in rows])
    lengths = torch.tensor([len(ix) for ix in legal], device=rows.device)
    row_ids = torch.repeat_interleave(torch.arange(len(legal), device=rows.device), lengths)
    col_ids = torch.from_numpy(np.concatenate([np.asarray(ix, dtype=np.int64) for ix in legal])).to(rows.device)
    return rows[row_ids, col_ids].float().cpu().numpy()


def _accepts_legal_indices(model) -> bool:
    """Whether ``model.run_many_encoded`` takes ``legal_indices`` (the repo's models
    do; test doubles and older wrappers may not)."""
    import inspect

    try:
        params = inspect.signature(model.run_many_encoded).parameters
    except (TypeError, ValueError):
        return False
    return "legal_indices" in params or any(p.kind == p.VAR_KEYWORD for p in params.values())


def _supports_packed(model) -> bool:
    """Whether ``model`` has the stacked-input fast path (``run_packed``; the
    transformer and policy+value composites of transformers do)."""
    return hasattr(model, "run_packed") and bool(getattr(model, "supports_packed", True))


def _packed_parts(packs: list) -> Optional[list]:
    """``packed_inputs`` parts for a batch of ``PackedStates``, or None if one
    lacks the per-state player count (legacy 7-tuple encodings)."""
    parts = []
    for p in packs:
        try:
            extras = np.asarray(p.extras, dtype=np.int64)
        except (TypeError, ValueError):  # ragged or not integers
            return None
        if extras.ndim != 2 or extras.shape[1] < 4:
            return None
        parts.append((p.game_states, p.node_features, extras, p.map_index))
    return parts


def run_packed_states(model, packs: list) -> tuple:
    """``model.run_many_encoded`` over the states of ``packs`` (``PackedStates``
    with legal indices) through the stacked-input fast path when the model has
    one (else unpacked): ``(probs, log_probs, values)``."""
    parts = _packed_parts(packs) if _supports_packed(model) else None
    if parts is not None:
        legal = LegalIndices.concat([p.legal_indices for p in packs])
        return model.run_packed(model.packed_inputs(parts), legal_indices=legal)
    legal = [ix for p in packs for ix in p.legal_indices]
    states = [s for p in packs for s in p.unpack()]
    if _accepts_legal_indices(model):
        return model.run_many_encoded(states, legal_indices=legal)
    return model.run_many_encoded(states)


def _batched_legal_replies(batch: list, spans: list, probs: Any, values: Any, price: Optional[dict]) -> list:
    """Replies for a batch whose requests all carry legal indices: the legal priors,
    values and price logits of the whole batch move to the CPU in one transfer each
    and are sliced per request there (what ``_gather_legal_priors``,
    ``_rows_to_numpy`` and ``_slice_price_rows_for_reply`` give per request)."""
    if not isinstance(probs, torch.Tensor):
        probs = torch.stack([torch.as_tensor(r) for r in probs])
    if not isinstance(values, torch.Tensor):
        values = torch.stack([torch.as_tensor(r) for r in values])
    legal = LegalIndices.concat([req.encoded_states.legal_indices for req in batch])
    lengths = legal.lengths
    row_ids = torch.from_numpy(legal.row_ids()).to(probs.device)
    col_ids = torch.from_numpy(legal.flat.astype(np.int64)).to(probs.device)
    priors = probs[row_ids, col_ids].float().cpu().numpy()
    values_np = values.detach().to(torch.float32).cpu().numpy()
    logits = price.get("price_logits") if price is not None else None
    logits_np = logits.detach().float().cpu().numpy() if logits is not None else None
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    replies = []
    for req, (start, n) in zip(batch, spans):
        price_rows = None
        if logits_np is not None:
            price_rows = {
                "price_logits": logits_np[start : start + n],
                "slot_index": price.get("slot_index"),
                "num_slots": price.get("num_slots"),
            }
        replies.append(
            InferenceReply(
                request_id=req.request_id,
                probs=priors[offsets[start] : offsets[start + n]],
                log_probs=None,
                value=values_np[start : start + n],
                price_components=price_rows,
                batched=True,
                legal_lengths=lengths[start : start + n],
            )
        )
    return replies


class _StagedReplies:
    """A batch's reply data on its way to the host (``InferenceServer.staged_replies``):
    the legal priors, values and price logits gathered on the model's device into
    one float32 buffer and copied into pinned host memory by one non-blocking
    transfer, queued on the stream right behind the forward pass. The reply's
    device work then rides in the forward's submission: no second round trip to
    a GPU that other processes time-slice (``_batched_legal_replies`` waits on it
    five times: two index uploads, the gather, three downloads)."""

    __slots__ = ("host", "event", "n_priors", "value_shape", "logits_shape")

    def __init__(self, host, event, n_priors: int, value_shape: tuple, logits_shape: Optional[tuple]):
        self.host, self.event = host, event
        self.n_priors, self.value_shape, self.logits_shape = n_priors, value_shape, logits_shape

    def wait(self) -> None:
        """Block until the transfer is done."""
        if self.event is not None:
            self.event.synchronize()

    def arrays(self) -> tuple:
        """``(priors, values, logits or None)`` as numpy views of the host buffer, once
        the transfer is done. The views keep the (pinned) buffer alive, so the
        allocator reuses it only after the replies made of them are sent."""
        self.wait()
        data = self.host.numpy()
        n_values = int(np.prod(self.value_shape))
        priors = data[: self.n_priors]
        values = data[self.n_priors : self.n_priors + n_values].reshape(self.value_shape)
        logits = None
        if self.logits_shape is not None:
            logits = data[self.n_priors + n_values :].reshape(self.logits_shape)
        return priors, values, logits


def _inputs_device(inputs: Any):
    """The device of ``packed_inputs``' tensors (None if it holds none)."""
    if isinstance(inputs, dict):
        for value in inputs.values():
            if isinstance(value, torch.Tensor):
                return value.device
    return None


def _gather_index(legal: LegalIndices, device) -> tuple:
    """The (row, column) gather of each state's legal priors, as int64 device tensors
    (``_batched_legal_replies``' indices). Uploaded before the forward pass, so the
    copy overlaps nothing that waits on it."""
    rows = torch.from_numpy(legal.row_ids())
    cols = torch.from_numpy(legal.flat.astype(np.int64))
    if device is None or torch.device(device).type == "cpu":
        return rows, cols
    if torch.cuda.is_available():
        rows, cols = rows.pin_memory(), cols.pin_memory()
    return rows.to(device, non_blocking=True), cols.to(device, non_blocking=True)


def _stage_replies(probs: Any, values: Any, price: Optional[dict], gather: tuple) -> Optional[_StagedReplies]:
    """Queue a batch's reply data for the host (see ``_StagedReplies``): the same
    gather and the same dtype conversions as ``_batched_legal_replies``, so the
    same bits. None when the outputs aren't tensors (the caller then takes
    ``_batched_legal_replies``)."""
    logits = price.get("price_logits") if price is not None else None
    if not (isinstance(probs, torch.Tensor) and isinstance(values, torch.Tensor)):
        return None
    if logits is not None and not isinstance(logits, torch.Tensor):
        return None
    if values.device != probs.device or (logits is not None and logits.device != probs.device):
        return None
    rows, cols = gather
    if rows.device != probs.device:
        rows, cols = rows.to(probs.device), cols.to(probs.device)
    priors = probs[rows, cols].float()
    values = values.detach().to(torch.float32)
    parts = [priors.reshape(-1), values.reshape(-1)]
    if logits is not None:
        logits = logits.detach().float()
        parts.append(logits.reshape(-1))
    flat = torch.cat(parts)
    event = None
    if flat.device.type == "cuda":
        host = torch.empty(flat.shape, dtype=torch.float32, pin_memory=True)
        host.copy_(flat, non_blocking=True)
        event = torch.cuda.Event()
        event.record()
    else:
        host = flat.cpu()
    return _StagedReplies(
        host, event, int(priors.numel()), tuple(values.shape), None if logits is None else tuple(logits.shape)
    )


def _staged_legal_replies(batch: list, spans: list, staged: _StagedReplies, legal: LegalIndices, price: Optional[dict]) -> list:
    """``_batched_legal_replies``' replies from a :class:`_StagedReplies` (the same
    arrays, sliced the same way)."""
    priors, values_np, logits_np = staged.arrays()
    lengths = legal.lengths
    offsets = legal._offs()
    slot_index = price.get("slot_index") if price is not None else None
    num_slots = price.get("num_slots") if price is not None else None
    replies = []
    for req, (start, n) in zip(batch, spans):
        price_rows = None
        if logits_np is not None:
            price_rows = {"price_logits": logits_np[start : start + n], "slot_index": slot_index, "num_slots": num_slots}
        replies.append(
            InferenceReply(
                request_id=req.request_id,
                probs=priors[offsets[start] : offsets[start + n]],
                log_probs=None,
                value=values_np[start : start + n],
                price_components=price_rows,
                batched=True,
                legal_lengths=lengths[start : start + n],
            )
        )
    return replies


def _slice_price_rows_for_reply(batched: Optional[dict], start: int, n: int) -> Optional[dict]:
    """Rows ``[start, start + n)`` of the model's batched price components, as
    numpy, for a batched reply (``slot_index`` / ``num_slots`` sent once)."""
    if batched is None:
        return None
    logits = batched.get("price_logits")
    if logits is None:
        return None
    return {
        "price_logits": logits[start:start + n].detach().float().cpu().numpy(),
        "slot_index": batched.get("slot_index"),
        "num_slots": batched.get("num_slots"),
    }


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------


@dataclass
class _ServerStats:
    batches_served: int = 0
    requests_served: int = 0
    total_forward_ms: float = 0.0
    total_batch_size: int = 0


class InferenceServer:
    """Inference server. Runs in its own process — instantiate via ``start()``.

    The server loop:

    1. Poll the control queue (non-blocking).
    2. While ``ACCEPTING``, drain up to ``batch_size`` requests from the
       request queue, waiting at most ``batch_timeout_ms`` for the batch
       to fill.
    3. Run a single forward pass on the collected batch.
    4. Slice the per-leaf outputs and post a ``InferenceReply`` to each
       requesting worker's reply queue.
    5. Repeat.
    """

    def __init__(
        self,
        request_q: Any,
        reply_qs: list,
        control_q: Any,
        model_factory: Any,
        checkpoint_path: Optional[str],
        batch_size: int = 64,
        batch_timeout_ms: float = 2.0,
        autocast_device: Optional[str] = None,
        idle_poll_ms: float = 1.0,
        control_reply_q: Optional[Any] = None,
        fast_path: bool = True,
        pipeline: bool = False,
        staged_replies: bool = True,
    ):
        self.request_q = request_q
        self.reply_qs = reply_qs
        self.control_q = control_q
        self.control_reply_q = control_reply_q
        self.model_factory = model_factory
        self.checkpoint_path = checkpoint_path
        self.batch_size = batch_size
        self.batch_timeout_ms = batch_timeout_ms
        self.autocast_device = autocast_device
        self.idle_poll_ms = idle_poll_ms
        # Stacked requests (lockstep searches, policy play) go to the model as
        # arrays (``_run_forward_packed``); False: through per-state tuples.
        self.fast_path = fast_path
        # A collector thread receives and stacks the next batch while the
        # current one runs (``_PipelinedCollector``).
        self.pipeline = pipeline
        # Replies to requests with legal indices: their device work (the legal
        # priors' gather, the values and price rows) is queued behind the forward
        # pass and reaches the host in one non-blocking copy (``_StagedReplies``);
        # False: ``_batched_legal_replies``' separate transfers, each a round trip
        # to the GPU. The same replies bit for bit (``test_inference_fast_path``).
        self.staged_replies = staged_replies

        self.state = ServerState.STARTING
        self._model = None
        self.stats = _ServerStats()

    # --- model lifecycle ---------------------------------------------------

    def _load_model(self, checkpoint_path: Optional[str] = None):
        """Build / reload the model. Calls ``model_factory(checkpoint_path)``.

        ``model_factory`` is supplied by the caller because the inference
        server lives in a child process and the model loader needs the
        coordinator's checkpoint directory layout. The factory is
        responsible for moving the model to the right device + ``eval()``.
        """
        path = checkpoint_path or self.checkpoint_path
        self._model = self.model_factory(path)

    def _unload_model(self):
        """Free GPU memory between iterations. The next ``reload`` rebuilds."""
        self._model = None
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    # --- state transitions -------------------------------------------------

    def _handle_control(self, msg: ControlMessage):
        op = msg.op
        if op == "shutdown":
            LOGGER.info("InferenceServer: SHUTDOWN received")
            self.state = ServerState.SHUTDOWN
            self._reply_control(msg, {"ok": True})
            return
        if op == "pause":
            # Drain in-flight (handled by the main loop) then unload.
            LOGGER.info("InferenceServer: PAUSE received; entering DRAINING")
            self.state = ServerState.DRAINING
            self._reply_control(msg, {"ok": True})
            return
        if op == "reload":
            path = (msg.payload or {}).get("checkpoint_path")
            LOGGER.info(f"InferenceServer: RELOAD({path}) received")
            self.state = ServerState.RELOADING
            try:
                self._load_model(path)
                self.state = ServerState.ACCEPTING
                self._reply_control(msg, {"ok": True})
            except Exception as e:
                LOGGER.error(f"InferenceServer: RELOAD failed: {e}", exc_info=True)
                self.state = ServerState.IDLE
                self._reply_control(msg, {"ok": False, "error": str(e)})
            return
        if op == "health":
            report = HealthReport(
                state=self.state.value,
                batches_served=self.stats.batches_served,
                requests_served=self.stats.requests_served,
                avg_batch_size=(
                    self.stats.total_batch_size / self.stats.batches_served
                    if self.stats.batches_served > 0 else 0.0
                ),
                avg_forward_ms=(
                    self.stats.total_forward_ms / self.stats.batches_served
                    if self.stats.batches_served > 0 else 0.0
                ),
                queue_depth_approx=self._queue_depth(),
            )
            self._reply_control(msg, report)
            return
        LOGGER.warning(f"InferenceServer: unknown control op {op!r}")

    def _reply_control(self, msg: ControlMessage, payload: Any):
        try:
            if msg.reply_q is not None:
                msg.reply_q.put(payload)
            elif self.control_reply_q is not None:
                self.control_reply_q.put((msg.seq, payload))
        except Exception as e:
            LOGGER.warning(f"InferenceServer: control reply put failed: {e}")

    def _queue_depth(self) -> int:
        try:
            return self.request_q.qsize()
        except (NotImplementedError, OSError):
            return -1  # macOS / Manager.Queue don't always support qsize

    # --- request batching --------------------------------------------------

    @staticmethod
    def _request_size(req: InferenceRequest) -> int:
        return len(req.encoded_states) if req.encoded_states is not None else 1

    def _collect_batch(self) -> list:
        """Gather requests until ``batch_size`` leaves, waiting at most ``batch_timeout_ms``.

        Requests already queued are always taken (the timeout only bounds
        waiting on an empty queue — deserializing a backlog must not cut the
        batch short). Returns the (possibly empty) batch. Falls through
        quickly when the queue is empty so the control loop stays responsive.
        """
        deadline = time.monotonic() + (self.batch_timeout_ms / 1000.0)
        batch: list[InferenceRequest] = []
        # First request: short blocking get so we don't busy-wait when idle.
        try:
            req = self.request_q.get(timeout=self.idle_poll_ms / 1000.0)
            batch.append(req)
        except queue.Empty:
            return batch

        size = self._request_size(req)
        while size < self.batch_size:
            try:
                req = self.request_q.get_nowait()
            except queue.Empty:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    req = self.request_q.get(timeout=remaining)
                except queue.Empty:
                    break
            batch.append(req)
            size += self._request_size(req)
        return batch

    def _prepare(self, batch: list) -> Optional[dict]:
        """The model-independent CPU work of the fast path for ``batch`` (its
        stacked numpy inputs and legal indices), or None if the batch takes the
        unpacked path. The pipeline's collector thread runs it."""
        if not self.fast_path:
            return None
        packs = [req.encoded_states for req in batch]
        if not all(isinstance(p, PackedStates) and p.legal_indices is not None for p in packs):
            return None
        parts = _packed_parts(packs)
        if parts is None:
            return None
        from rl18xx.agent.alphazero.model_transformer import MAX_PLAYERS, packed_arrays

        return {
            "parts": parts,
            "arrays": packed_arrays(parts, MAX_PLAYERS),
            "legal": LegalIndices.concat([p.legal_indices for p in packs]),
        }

    def _run_forward(self, batch: list[InferenceRequest], prepared: Optional[dict] = None) -> list[InferenceReply]:
        if self.fast_path and _supports_packed(self._model):
            prepared = prepared if prepared is not None else self._prepare(batch)
            if prepared is not None:
                return self._run_forward_packed(batch, prepared)
        tu = time.monotonic()
        encoded_states: list = []
        spans: list[tuple[int, int]] = []
        for req in batch:
            if isinstance(req.encoded_states, PackedStates):
                items = req.encoded_states.unpack()
            elif req.encoded_states is not None:
                items = req.encoded_states
            else:
                items = [req.encoded_state]
            spans.append((len(encoded_states), len(items)))
            encoded_states.extend(items)
        autocast_ctx = (
            torch.amp.autocast(self.autocast_device)
            if self.autocast_device
            else _NullCtx()
        )
        tf = time.monotonic()
        all_legal = all(
            isinstance(req.encoded_states, PackedStates) and req.encoded_states.legal_indices is not None for req in batch
        )
        kwargs = {}
        if all_legal and _accepts_legal_indices(self._model):
            # Only each state's legal priors go back, so the model may skip work for
            # action blocks with no legal move (e.g. the tile-lay grid).
            kwargs["legal_indices"] = [ix for req in batch for ix in req.encoded_states.legal_indices]
        staged = legal = None
        with torch.no_grad(), autocast_ctx:
            probs, log_probs, values = self._model.run_many_encoded(encoded_states, **kwargs)
            # Pull the model's batched price-components stash (might be None
            # for the GNN model) and slice per-leaf in numpy form so the reply
            # serializes cleanly.
            batched_price_components = getattr(self._model, "last_price_components", None)
            if all_legal and self.staged_replies:
                # Queued behind the forward (``_StagedReplies``).
                legal = LegalIndices.concat([req.encoded_states.legal_indices for req in batch])
                device = probs.device if isinstance(probs, torch.Tensor) else None
                staged = _stage_replies(probs, values, batched_price_components, _gather_index(legal, device))
        if _PROFILE:
            if staged is not None:
                staged.wait()
            elif torch.cuda.is_available():
                torch.cuda.synchronize()
            _PROFILE.add("unpack", tf - tu)
            _PROFILE.add("forward", time.monotonic() - tf)
        tr = time.monotonic()
        if all_legal:
            # The usual case (policy-only play, policy-gradient workers): gather every
            # request's legal priors, values and price rows in one transfer each, then
            # slice on the CPU -- per-request transfers each waited on the GPU and took
            # more of the server's time than the forward pass.
            if staged is not None:
                replies = _staged_legal_replies(batch, spans, staged, legal, batched_price_components)
            else:
                replies = _batched_legal_replies(batch, spans, probs, values, batched_price_components)
            if _PROFILE:
                _PROFILE.add("replies", time.monotonic() - tr)
            return replies
        replies: list[InferenceReply] = []
        for req, (start, n) in zip(batch, spans):
            if req.encoded_states is None:
                replies.append(
                    InferenceReply(
                        request_id=req.request_id,
                        probs=_to_numpy(probs[start]),
                        log_probs=_to_numpy(log_probs[start]),
                        value=_to_numpy(values[start]),
                        price_components=_slice_price_components_for_reply(batched_price_components, start),
                    )
                )
                continue
            legal = getattr(req.encoded_states, "legal_indices", None)
            if legal is not None:
                probs_np, legal_lengths = _gather_legal_priors(probs, start, legal), np.array(
                    [len(ix) for ix in legal], dtype=np.int64
                )
            else:
                probs_np, legal_lengths = _rows_to_numpy(probs, start, n, torch.float16), None
            replies.append(
                InferenceReply(
                    request_id=req.request_id,
                    probs=probs_np,
                    log_probs=None,
                    value=_rows_to_numpy(values, start, n, torch.float32),
                    price_components=_slice_price_rows_for_reply(batched_price_components, start, n),
                    batched=True,
                    legal_lengths=legal_lengths,
                )
            )
        if _PROFILE:
            _PROFILE.add("replies", time.monotonic() - tr)
        return replies

    def _autocast(self):
        return torch.amp.autocast(self.autocast_device) if self.autocast_device else _NullCtx()

    def _run_forward_packed(self, batch: list, prepared: dict) -> list:
        """``_run_forward`` for requests that are all ``PackedStates`` with legal
        indices, on a model with ``run_packed``: the requests' stacked arrays go to
        the device as they are (no per-state tuples, one set of inputs for both
        networks of a composite). Same inputs, same forward: the replies are the
        unpacked path's (bit for bit without map indices, ``test_inference_fast_path``)."""
        tu = time.monotonic()
        spans, start = [], 0
        for req in batch:
            spans.append((start, len(req.encoded_states)))
            start += len(req.encoded_states)
        legal = prepared["legal"]
        staged = None
        with torch.no_grad(), self._autocast():
            try:
                inputs = self._model.packed_inputs(prepared["arrays"])
            except ValueError:  # laid out for another player count than the model's
                inputs = self._model.packed_inputs(prepared["parts"])
            # The replies' gather indices go up with the inputs, before the forward.
            gather = _gather_index(legal, _inputs_device(inputs)) if self.staged_replies else None
            tf = time.monotonic()
            probs, _, values = self._model.run_packed(inputs, legal_indices=legal)
            price = getattr(self._model, "last_price_components", None)
            if gather is not None:
                # Queued behind the forward: the batch's one wait on the GPU is below.
                staged = _stage_replies(probs, values, price, gather)
        if _PROFILE:
            # With staged replies "forward" includes their device work (a gather and
            # one copy) and "replies" is host work only.
            if staged is not None:
                staged.wait()
            elif torch.cuda.is_available():
                torch.cuda.synchronize()
            _PROFILE.add("unpack", tf - tu)
            _PROFILE.add("forward", time.monotonic() - tf)
        tr = time.monotonic()
        if staged is not None:
            replies = _staged_legal_replies(batch, spans, staged, legal, price)
        else:
            replies = _batched_legal_replies(batch, spans, probs, values, price)
        if _PROFILE:
            _PROFILE.add("replies", time.monotonic() - tr)
        return replies

    def _dispatch_replies(self, batch: list[InferenceRequest], replies: list[InferenceReply]):
        for req, reply in zip(batch, replies):
            try:
                self.reply_qs[req.worker_id].put(reply)
            except IndexError:
                LOGGER.error(
                    f"InferenceServer: worker_id={req.worker_id} out of range "
                    f"(N reply_qs={len(self.reply_qs)}); dropping reply"
                )
            except Exception as e:
                LOGGER.warning(
                    f"InferenceServer: reply dispatch to worker {req.worker_id} failed: {e}"
                )

    # --- main loop ---------------------------------------------------------

    def run(self):
        """Main server loop. Blocks until SHUTDOWN."""
        LOGGER.info(f"InferenceServer.run: STARTING (batch_size={self.batch_size}, "
                    f"batch_timeout_ms={self.batch_timeout_ms})")
        try:
            self._load_model()
            self.state = ServerState.ACCEPTING
            LOGGER.info("InferenceServer: ACCEPTING")
        except Exception as e:
            LOGGER.error(f"InferenceServer: initial model load failed: {e}", exc_info=True)
            if getattr(self, "exit_on_load_failure", False):
                # A server process that can't load its model must die: alive and idle,
                # its workers wait forever and wait_checking_servers (which looks for
                # dead servers) can't tell (a deleted checkpoint hung an eval for hours).
                raise
            self.state = ServerState.IDLE

        collector = _BatchCollector(self) if self.pipeline else None
        try:
            while self.state != ServerState.SHUTDOWN:
                # 1. Drain control queue (non-blocking).
                self._poll_control()
                if self.state == ServerState.SHUTDOWN:
                    break

                # 2. If draining, finish whatever's in the request queue then unload.
                if self.state == ServerState.DRAINING:
                    if collector is not None:
                        collector.pause_and_flush(self._serve)  # what it took first, then the queue here
                    if not self._drain_in_flight():
                        self._unload_model()
                        self.state = ServerState.IDLE
                        LOGGER.info("InferenceServer: IDLE")
                    continue

                # 3. Idle / reloading — wait for control.
                if self.state in (ServerState.IDLE, ServerState.RELOADING, ServerState.STARTING):
                    time.sleep(self.idle_poll_ms / 1000.0)
                    continue

                # 4. ACCEPTING: drain a batch + forward + reply.
                tc = time.monotonic()
                if collector is not None:
                    # The collector thread received (and stacked) this batch while the
                    # previous one ran.
                    collector.resume()
                    item = collector.get(self.idle_poll_ms / 1000.0)
                    if item is None:
                        if _PROFILE:
                            _PROFILE.add("idle", time.monotonic() - tc)
                        continue
                    batch, prepared = item
                    self._serve(batch, prepared, tc)
                    continue
                batch = self._collect_batch()
                if not batch:
                    if _PROFILE:
                        _PROFILE.add("idle", time.monotonic() - tc)
                    continue
                self._serve(batch, None, tc)
        finally:
            if collector is not None:
                collector.stop()

        LOGGER.info("InferenceServer.run: exited cleanly")

    def _serve(self, batch: list, prepared: Optional[dict] = None, collected_at: Optional[float] = None):
        """Forward ``batch`` and send its replies."""
        t0 = time.monotonic()
        replies = self._run_forward(batch, prepared)
        elapsed_ms = (time.monotonic() - t0) * 1000.0
        td = time.monotonic()
        self._dispatch_replies(batch, replies)
        leaves = sum(self._request_size(req) for req in batch)
        if _PROFILE:
            if collected_at is not None:
                _PROFILE.add("collect", t0 - collected_at)
            _PROFILE.add("dispatch", time.monotonic() - td)
            _PROFILE.batch(leaves)
        self.stats.batches_served += 1
        self.stats.requests_served += leaves
        self.stats.total_batch_size += leaves
        self.stats.total_forward_ms += elapsed_ms

    def _poll_control(self):
        while True:
            try:
                msg = self.control_q.get_nowait()
            except queue.Empty:
                return
            self._handle_control(msg)
            if self.state == ServerState.SHUTDOWN:
                return

    def _drain_in_flight(self) -> bool:
        """While DRAINING, finish any requests already on the queue. Returns
        True while there's still work to flush (caller stays in DRAINING),
        False once the queue is empty (caller transitions to IDLE)."""
        batch = self._collect_batch()
        if not batch:
            return False
        replies = self._run_forward(batch)
        self._dispatch_replies(batch, replies)
        return True


class _BatchCollector:
    """``InferenceServer(pipeline=True)``: a thread that receives the next batch
    (``_collect_batch``: the queue reads and unpickling) and stacks its inputs
    (``_prepare``: numpy, no model) while the server thread runs the current
    batch's forward pass and replies. Batches are served in the order they were
    collected, by the same code, so the replies are those of the unpipelined
    server; only when the work happens changes. One batch is held ready at a time
    (``depth``), so requests wait in the queue -- not in the server -- and a
    batch never waits behind more than one other.

    Control stays on the server thread: a pause (DRAINING) stops the collector,
    serves what it took, and drains the queue the usual way; it resumes once the
    server accepts again. The collector never touches the model, so a reload
    can happen while it collects (the batch it holds is served by the new
    model, as a request queued during a reload is without the pipeline)."""

    def __init__(self, server: "InferenceServer", depth: int = 1):
        import threading

        self.server = server
        self.ready: queue.Queue = queue.Queue(maxsize=depth)
        self._pause = threading.Event()
        self._idle = threading.Event()
        self._stop = threading.Event()
        self._idle.set()
        self.error: Optional[BaseException] = None
        self.thread = threading.Thread(target=self._loop, daemon=True, name="InferenceCollector")
        self.thread.start()

    def _loop(self):
        poll = self.server.idle_poll_ms / 1000.0
        try:
            while not self._stop.is_set():
                if self._pause.is_set():
                    self._idle.set()
                    time.sleep(poll)
                    continue
                self._idle.clear()
                if self._pause.is_set():
                    continue
                t0 = time.monotonic()
                batch = self.server._collect_batch()
                if not batch:
                    continue
                t1 = time.monotonic()
                prepared = self.server._prepare(batch)
                if _PROFILE:
                    _PROFILE.add("bg_collect", t1 - t0)
                    _PROFILE.add("bg_prepare", time.monotonic() - t1)
                while not self._stop.is_set():
                    try:
                        self.ready.put((batch, prepared), timeout=0.05)
                        break
                    except queue.Full:
                        continue
        except BaseException as e:  # surfaced on the server thread by get()
            self.error = e
            self._idle.set()
            raise

    def _check(self):
        if self.error is not None:
            raise RuntimeError("inference server: the batch collector thread failed") from self.error

    def get(self, timeout: float):
        """The next collected ``(batch, prepared)``, or None after ``timeout``."""
        self._check()
        try:
            return self.ready.get(timeout=timeout)
        except queue.Empty:
            return None

    def resume(self):
        self._pause.clear()

    def pause_and_flush(self, serve):
        """Stop collecting and ``serve(batch, prepared)`` every batch already taken."""
        self._pause.set()
        while True:
            self._check()
            try:
                batch, prepared = self.ready.get_nowait()
            except queue.Empty:
                if self._idle.is_set() and self.ready.empty():
                    return
                time.sleep(0.0005)
                continue
            serve(batch, prepared)

    def stop(self):
        self._stop.set()
        self.thread.join(timeout=5.0)


class _StageProfile:
    """``RL18XX_SERVER_PROFILE=<seconds>``: every that many seconds a server logs how
    its time split between waiting for requests (idle), collecting a batch (incl.
    unpickling), unpacking states, the forward pass (synchronized), building
    replies and dispatching them. With ``staged_replies`` (the default) the
    replies' device work (gather, one copy to the host) is queued with the
    forward and counts in "forward"; "replies" is then host work only (before
    2026-10-19 it held ~1/3 of a busy server's time: five more waits on the GPU
    per batch). With ``pipeline``, "collect" is the server
    thread's wait for the collector's next batch and ``bg_collect`` /
    ``bg_prepare`` the collector thread's own time (overlapping the rest)."""

    STAGES = ("idle", "collect", "unpack", "forward", "replies", "dispatch")

    def __init__(self, every_s: float):
        import os

        self.every_s, self.pid = every_s, os.getpid()
        self.reset()

    def reset(self):
        self.t = {k: 0.0 for k in self.STAGES}
        self.batches = self.leaves = 0
        self.start = time.monotonic()

    def add(self, stage: str, seconds: float):
        self.t[stage] = self.t.get(stage, 0.0) + seconds

    def batch(self, leaves: int):
        self.batches += 1
        self.leaves += leaves
        wall = time.monotonic() - self.start
        if wall >= self.every_s:
            parts = "  ".join(f"{k} {100 * v / wall:4.1f}%" for k, v in list(self.t.items()))
            print(
                f"[server {self.pid}] {wall:.0f}s: {self.batches} batches, {self.leaves / max(1, self.batches):.1f} "
                f"states/batch, {self.leaves / wall:.0f} states/s | {parts}",
                flush=True,
            )
            self.reset()


def _profile_from_env():
    import os

    every = os.environ.get("RL18XX_SERVER_PROFILE")
    return _StageProfile(float(every)) if every else None


_PROFILE = _profile_from_env()


class _NullCtx:
    def __enter__(self):
        return self
    def __exit__(self, *args):
        return False


def _to_numpy(t: Any) -> np.ndarray:
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    return np.asarray(t)


# ---------------------------------------------------------------------------
# Client (used by each self-play worker)
# ---------------------------------------------------------------------------


@dataclass
class InferenceClientConfig:
    request_timeout_s: float = 60.0
    """How long to wait for a reply before giving up. Server-side faults
    should propagate as exceptions rather than infinite hangs."""


class InferenceClient:
    """Per-worker shim around the inference server.

    Mirrors the model's ``run_encoded`` / ``run_many_encoded`` API so the
    self-play tree-search loop can swap backends without further changes.
    After each ``run_many_encoded`` call the client exposes a fabricated
    ``last_price_components`` dict so the existing slicer in
    ``self_play._slice_price_components`` can index it like the in-process
    case (2D tensors stacked from the per-leaf 1D arrays the server
    returns).
    """

    def __init__(
        self,
        request_q: Any,
        reply_q: Any,
        worker_id: int,
        client_config: Optional[InferenceClientConfig] = None,
    ):
        if isinstance(request_q, (list, tuple)):
            # A replicated server (``start_inference_server(replicas=R)``): each
            # worker sends to one replica, all of which can answer every worker.
            request_q = request_q[worker_id % len(request_q)]
        self.request_q = request_q
        self.reply_q = reply_q
        self.worker_id = worker_id
        self.client_config = client_config or InferenceClientConfig()
        self._next_request_id = 0
        # last_price_components mirrors the model attribute MCTS reads.
        self.last_price_components: Optional[dict] = None

    def encoder_type(self) -> str:
        # MCTS sometimes queries this; the server hosts the real model
        # (which has its own encoder_type). For the client we return the
        # transformer family by default — adjust if you ever need the GNN
        # codepath via the server.
        return "Transformer"

    def run_encoded(self, encoded_game_state):
        """Single-leaf path (used by SelfPlay.play's first-node expansion)."""
        probs, _log_probs, values = self.run_many_encoded([encoded_game_state])
        return probs[0], None, values[0]

    def run_many_encoded(self, encoded_game_states: list, legal_indices: Optional[list] = None):
        """Submit all leaves as one request and unpack the batched reply.

        Returns ``(probs, None, values)`` like the model's API minus
        log-probs, which aren't shipped (no MCTS caller uses them).
        """
        return self.receive(self.send(encoded_game_states, legal_indices))

    def send(self, encoded_game_states: list, legal_indices: Optional[list] = None):
        """Submit a request without waiting; pass the result to :meth:`receive`.
        Lets a worker with clients on several servers keep them busy at once."""
        if not encoded_game_states:
            raise ValueError("Received no game states to run.")
        rid = self._next_request_id
        self._next_request_id += 1
        packed = PackedStates.pack(encoded_game_states)
        if packed is not None and legal_indices is not None:
            packed.legal_indices = [np.asarray(ix, dtype=np.int32) for ix in legal_indices]
        self.request_q.put(
            InferenceRequest(
                request_id=rid,
                worker_id=self.worker_id,
                encoded_states=packed if packed is not None else list(encoded_game_states),
            )
        )
        return rid, packed, len(encoded_game_states)

    def send_packed(self, packed: PackedStates):
        """Submit states already stacked into a :class:`PackedStates` (the
        lockstep search's natively encoded leaves); answer with
        :meth:`receive_raw`."""
        if len(packed) == 0:
            raise ValueError("Received no game states to run.")
        rid = self._next_request_id
        self._next_request_id += 1
        self.request_q.put(InferenceRequest(request_id=rid, worker_id=self.worker_id, encoded_states=packed))
        return rid, packed, len(packed)

    def receive_raw(self, pending) -> InferenceReply:
        """The server's reply to a send as it arrived: with legal indices,
        ``probs`` holds each state's legal priors back to back (no dense
        policy vectors are built)."""
        return self._wait_reply(pending)

    def _wait_reply(self, pending) -> InferenceReply:
        rid, packed, num_states = pending
        # The reply queue is per-worker; anything else on it is a stale reply
        # to an earlier timed-out request.
        deadline = time.monotonic() + self.client_config.request_timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"InferenceClient: timeout after {self.client_config.request_timeout_s}s "
                    f"waiting for a reply to {num_states} leaves"
                )
            try:
                reply = self.reply_q.get(timeout=min(remaining, 1.0))
            except queue.Empty:
                continue
            if reply.request_id == rid:
                break
            LOGGER.debug(
                f"InferenceClient(worker={self.worker_id}): discarding stale reply for request {reply.request_id}"
            )
        return reply

    def receive(self, pending):
        """Wait for the reply to a :meth:`send` and unpack it like ``run_many_encoded``."""
        reply = self._wait_reply(pending)
        packed = pending[1]
        if reply.legal_lengths is not None:
            # Scatter the legal priors back into dense policy vectors.
            from rl18xx.agent.alphazero.mcts import POLICY_SIZE

            dense = np.zeros((len(reply.legal_lengths), POLICY_SIZE), dtype=np.float32)
            offset = 0
            for row, (ix, length) in enumerate(zip(packed.legal_indices, reply.legal_lengths)):
                dense[row, ix] = reply.probs[offset:offset + length]
                offset += length
            probs = torch.from_numpy(dense)
        else:
            probs = torch.from_numpy(reply.probs.astype(np.float32))
        values = torch.from_numpy(np.asarray(reply.value, dtype=np.float32))
        pc = reply.price_components
        self.last_price_components = (
            None
            if pc is None
            else {
                "price_logits": torch.from_numpy(pc["price_logits"]),
                "slot_index": pc.get("slot_index"),
                "num_slots": pc.get("num_slots"),
            }
        )
        return list(probs), None, list(values)


def _stack_price_components(per_leaf: list[Optional[dict]]) -> Optional[dict]:
    """Re-batch a list of per-leaf sliced dicts into a single dict that
    ``self_play._slice_price_components`` can index like the in-process case.

    Returns ``None`` if any leaf is None (the model has no price head, or
    the server returned an empty slice). The slot index map / num_slots
    are propagated from the first non-None entry (they're constant for the
    model).
    """
    non_none = [d for d in per_leaf if d is not None]
    if not non_none:
        return None
    if len(non_none) != len(per_leaf):
        # Mixed Some/None — the model can't have produced this; mark None
        # so MCTS falls back to uniform price cells.
        return None
    return {
        "price_logits": torch.stack([torch.from_numpy(d["price_logits"]) for d in per_leaf]),
        "slot_index": per_leaf[0].get("slot_index"),
        "num_slots": per_leaf[0].get("num_slots"),
    }


# ---------------------------------------------------------------------------
# Parent-side control handle
# ---------------------------------------------------------------------------


@dataclass
class ServerHandle:
    """Coordinator-side handle returned by ``start_inference_server``.

    Holds the queues that workers need (``request_q`` + ``reply_qs``) plus
    the control channel and the spawned process. Use :meth:`pause`,
    :meth:`reload`, :meth:`health`, :meth:`shutdown` to drive the server.
    """

    process: Any
    request_q: Any
    reply_qs: list
    control_q: Any
    ticket_q: Any
    num_workers: int
    ctx: Any  # mp context
    control_reply_q: Any = None
    _seq: int = 0

    def _send(self, op: str, payload: Optional[dict] = None, timeout_s: float = 30.0) -> Any:
        # Replies come back on the server's one control reply queue, tagged
        # with the request's seq; a reply to an earlier timed-out op is skipped.
        self._seq += 1
        seq = self._seq
        self.control_q.put(ControlMessage(op=op, payload=payload, seq=seq))
        deadline = time.monotonic() + timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"ServerHandle.{op}: no reply in {timeout_s}s")
            try:
                reply_seq, reply = self.control_reply_q.get(timeout=remaining)
            except queue.Empty:
                raise TimeoutError(f"ServerHandle.{op}: no reply in {timeout_s}s")
            if reply_seq == seq:
                return reply

    def reset_worker_slots(self) -> None:
        """Make every worker slot available to a new process pool.

        A pool worker takes a slot ticket at boot and never hands it back, so
        the next iteration's pool would block forever in
        ``worker_init_inference`` once the first pool had taken them all. Call
        between pools, with no workers alive. Also drops any reply left in a
        slot's queue so a new worker can't read its predecessor's.
        """
        for q in (self.ticket_q, *self.reply_qs):
            while True:
                try:
                    q.get(timeout=0.05)
                except queue.Empty:
                    break
        for i in range(self.num_workers):
            self.ticket_q.put(i)

    def pause(self, timeout_s: float = 30.0):
        return self._send("pause", timeout_s=timeout_s)

    def reload(self, checkpoint_path: Optional[str], timeout_s: float = 60.0):
        return self._send("reload", payload={"checkpoint_path": checkpoint_path}, timeout_s=timeout_s)

    def health(self, timeout_s: float = 5.0) -> HealthReport:
        return self._send("health", timeout_s=timeout_s)

    def shutdown(self, timeout_s: float = 30.0):
        if self.process is not None and not self.process.is_alive():
            return  # died already: nothing would answer
        try:
            self._send("shutdown", timeout_s=timeout_s)
        except TimeoutError:
            LOGGER.warning("ServerHandle.shutdown: timeout; terminating process")
        finally:
            if self.process is not None and self.process.is_alive():
                self.process.join(timeout=5.0)
                if self.process.is_alive():
                    self.process.terminate()


class _ReplicaProcesses:
    """The replicas' processes as one: alive while all are; the exit code of the
    first that died (what ``wait_checking_servers`` and the evals read)."""

    def __init__(self, processes: list):
        self.processes = processes

    def is_alive(self) -> bool:
        return all(p.is_alive() for p in self.processes)

    @property
    def exitcode(self):
        return next((p.exitcode for p in self.processes if not p.is_alive()), None)

    def join(self, timeout: Optional[float] = None):
        for p in self.processes:
            p.join(timeout)

    def terminate(self):
        for p in self.processes:
            p.terminate()


class ReplicatedServerHandle:
    """``ServerHandle`` for ``start_inference_server(replicas=R)``: R server
    processes on one model, with one ticket queue and one reply queue per worker
    shared by all of them, and a request queue each -- ``request_q`` is their
    list, from which an ``InferenceClient`` takes its worker's (``worker_id %
    R``). Control ops go to every replica. Each replica batches its own workers'
    requests; while one runs its forward pass the others receive, unpickle and
    reply, which a single server process does in series."""

    def __init__(self, replicas: list):
        self.replicas = replicas
        self.request_q = [h.request_q for h in replicas]
        self.reply_qs = replicas[0].reply_qs
        self.ticket_q = replicas[0].ticket_q
        self.num_workers = replicas[0].num_workers
        self.ctx = replicas[0].ctx
        self.process = _ReplicaProcesses([h.process for h in replicas])

    def reset_worker_slots(self) -> None:
        self.replicas[0].reset_worker_slots()  # the ticket and reply queues are shared

    def pause(self, timeout_s: float = 30.0):
        return self._all("pause", timeout_s)

    def reload(self, checkpoint_path: Optional[str], timeout_s: float = 60.0):
        replies = [h.reload(checkpoint_path, timeout_s=timeout_s) for h in self.replicas]
        bad = [r for r in replies if not (isinstance(r, dict) and r.get("ok"))]
        return bad[0] if bad else {"ok": True}

    def _all(self, op: str, timeout_s: float):
        replies = [getattr(h, op)(timeout_s=timeout_s) for h in self.replicas]
        bad = [r for r in replies if not (isinstance(r, dict) and r.get("ok"))]
        return bad[0] if bad else {"ok": True}

    def health(self, timeout_s: float = 5.0) -> HealthReport:
        reports = [h.health(timeout_s=timeout_s) for h in self.replicas]
        batches = sum(r.batches_served for r in reports)
        states = {r.state for r in reports}
        return HealthReport(
            state=reports[0].state if len(states) == 1 else "MIXED:" + ",".join(sorted(states)),
            batches_served=batches,
            requests_served=sum(r.requests_served for r in reports),
            avg_batch_size=sum(r.avg_batch_size * r.batches_served for r in reports) / max(batches, 1),
            avg_forward_ms=sum(r.avg_forward_ms * r.batches_served for r in reports) / max(batches, 1),
            queue_depth_approx=sum(max(r.queue_depth_approx, 0) for r in reports),
        )

    def shutdown(self, timeout_s: float = 30.0):
        for h in self.replicas:
            h.shutdown(timeout_s=timeout_s)


def wait_checking_servers(futures, handles: dict, executor, timeout: float = 30.0):
    """``concurrent.futures.wait(futures, timeout, FIRST_COMPLETED)`` that fails if a server died.

    Workers block forever on the replies of a dead inference server, so a plain
    wait (and the executor's shutdown) would hang instead of failing: an
    evaluation sharing a full GPU lost a server to an out-of-memory error and
    waited for hours. If any of ``handles`` (name -> ServerHandle) has exited,
    the executor's workers are killed, so its shutdown returns, and this raises.
    """
    done, pending = wait(futures, timeout=timeout, return_when=FIRST_COMPLETED)
    dead = [
        f"{name} (exit code {handle.process.exitcode})"
        for name, handle in handles.items()
        if handle.process is not None and not handle.process.is_alive()
    ]
    if dead:
        for process in list((getattr(executor, "_processes", None) or {}).values()):
            process.terminate()
        executor.shutdown(wait=False, cancel_futures=True)
        raise RuntimeError(f"inference server died: {', '.join(dead)}")
    return done, pending


def _server_main(
    request_q: Any,
    reply_qs: list,
    control_q: Any,
    model_factory: Any,
    checkpoint_path: Optional[str],
    batch_size: int,
    batch_timeout_ms: float,
    autocast_device: Optional[str],
    control_reply_q: Any = None,
    fast_path: bool = True,
    pipeline: bool = False,
    staged_replies: bool = True,
):
    """Entry point for the inference server child process."""
    server = InferenceServer(
        request_q=request_q,
        reply_qs=reply_qs,
        control_q=control_q,
        control_reply_q=control_reply_q,
        model_factory=model_factory,
        checkpoint_path=checkpoint_path,
        batch_size=batch_size,
        batch_timeout_ms=batch_timeout_ms,
        autocast_device=autocast_device,
        fast_path=fast_path,
        pipeline=pipeline,
        staged_replies=staged_replies,
    )
    server.exit_on_load_failure = True  # the process exits, so the coordinator sees it die
    server.run()


def start_inference_server(
    *,
    num_workers: int,
    model_factory: Any,
    checkpoint_path: Optional[str],
    batch_size: int = 64,
    batch_timeout_ms: float = 2.0,
    autocast_device: Optional[str] = None,
    mp_context: str = "spawn",
    fast_path: bool = True,
    pipeline: bool = False,
    replicas: int = 1,
    staged_replies: bool = True,
):
    """Spawn the inference server process and return a coordinator-side handle.

    Workers receive the request queue + their reply queue via
    ``ProcessPoolExecutor(initializer=worker_init_inference, initargs=...)``
    using ``ticket_q`` to self-assign a worker slot. ``fast_path`` /
    ``pipeline`` / ``staged_replies``: see ``InferenceServer``. ``replicas`` > 1: that many server
    processes on the same model (``ReplicatedServerHandle``); worker ``w`` sends
    to replica ``w % replicas``, so one replica receives and replies while
    another's forward runs.
    """
    ctx = mp.get_context(mp_context)
    reply_qs = [ctx.Queue() for _ in range(num_workers)]
    ticket_q = ctx.Queue()
    for i in range(num_workers):
        ticket_q.put(i)
    handles = [
        _start_server_process(
            ctx, reply_qs, ticket_q, num_workers, model_factory, checkpoint_path, batch_size, batch_timeout_ms,
            autocast_device, fast_path, pipeline, name="InferenceServer" if replicas == 1 else f"InferenceServer{r}",
            staged_replies=staged_replies,
        )
        for r in range(max(1, int(replicas)))
    ]
    return handles[0] if len(handles) == 1 else ReplicatedServerHandle(handles)


def _start_server_process(
    ctx, reply_qs, ticket_q, num_workers, model_factory, checkpoint_path, batch_size, batch_timeout_ms,
    autocast_device, fast_path, pipeline, name="InferenceServer", staged_replies=True,
) -> ServerHandle:
    request_q = ctx.Queue()
    control_q = ctx.Queue()
    control_reply_q = ctx.Queue()
    proc = ctx.Process(
        target=_server_main,
        kwargs=dict(
            request_q=request_q,
            reply_qs=reply_qs,
            control_q=control_q,
            control_reply_q=control_reply_q,
            model_factory=model_factory,
            checkpoint_path=checkpoint_path,
            batch_size=batch_size,
            batch_timeout_ms=batch_timeout_ms,
            autocast_device=autocast_device,
            fast_path=fast_path,
            pipeline=pipeline,
            staged_replies=staged_replies,
        ),
        daemon=True,
        name=name,
    )
    proc.start()
    return ServerHandle(
        process=proc,
        request_q=request_q,
        reply_qs=reply_qs,
        control_q=control_q,
        ticket_q=ticket_q,
        num_workers=num_workers,
        ctx=ctx,
        control_reply_q=control_reply_q,
    )


# ---------------------------------------------------------------------------
# Worker-side initialization (used as ProcessPoolExecutor initializer)
# ---------------------------------------------------------------------------


# Module-level state populated by ``worker_init_inference``. Workers retrieve
# the configured client via ``get_worker_client()`` from this module.
_WORKER_CLIENT: Optional[InferenceClient] = None
_WORKER_ID: Optional[int] = None


def worker_init_inference(request_q, reply_qs, ticket_q, client_config: Optional[InferenceClientConfig] = None):
    """Per-worker initializer for ``ProcessPoolExecutor``.

    Each worker pulls a unique slot ticket from ``ticket_q`` and uses it to
    index into the pre-allocated ``reply_qs`` list. The resulting
    ``InferenceClient`` is stashed module-globally so the worker entry
    point can fetch it via :func:`get_worker_client`.
    """
    global _WORKER_CLIENT, _WORKER_ID
    worker_id = ticket_q.get()
    _WORKER_ID = worker_id
    _WORKER_CLIENT = InferenceClient(
        request_q=request_q,
        reply_q=reply_qs[worker_id],
        worker_id=worker_id,
        client_config=client_config or InferenceClientConfig(),
    )


def get_worker_client() -> Optional[InferenceClient]:
    """Return the worker-process inference client, or None if not initialized."""
    return _WORKER_CLIENT


def get_worker_id() -> Optional[int]:
    return _WORKER_ID
