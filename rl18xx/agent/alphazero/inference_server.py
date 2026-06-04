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


@dataclass
class InferenceRequest:
    """One inference request from a worker."""

    request_id: int
    worker_id: int
    encoded_state: Any  # the encoded-state tuple emitted by Encoder_GNN / _rust_encode


@dataclass
class InferenceReply:
    """One inference result destined for a worker."""

    request_id: int
    probs: np.ndarray              # (POLICY_SIZE,) softmax probabilities
    log_probs: np.ndarray          # (POLICY_SIZE,) log-softmax
    value: np.ndarray              # (VALUE_SIZE,) per-player softmaxed win/loss
    price_components: Optional[dict]  # per-leaf sliced (1D tensors or None)


@dataclass
class ControlMessage:
    """Control op sent from the coordinator to the server."""

    op: str  # "pause" | "reload" | "shutdown" | "health"
    payload: Optional[dict] = None
    reply_q: Optional[Any] = None  # mp.Queue (proxy / shared) for synchronous ops


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
    means = batched.get("price_mean")
    log_stds = batched.get("price_log_std")
    if means is None or log_stds is None:
        return None
    return {
        "price_mean": means[leaf_index].detach().cpu().numpy(),
        "price_log_std": log_stds[leaf_index].detach().cpu().numpy(),
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
    ):
        self.request_q = request_q
        self.reply_qs = reply_qs
        self.control_q = control_q
        self.model_factory = model_factory
        self.checkpoint_path = checkpoint_path
        self.batch_size = batch_size
        self.batch_timeout_ms = batch_timeout_ms
        self.autocast_device = autocast_device
        self.idle_poll_ms = idle_poll_ms

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
        if msg.reply_q is None:
            return
        try:
            msg.reply_q.put(payload)
        except Exception as e:
            LOGGER.warning(f"InferenceServer: control reply put failed: {e}")

    def _queue_depth(self) -> int:
        try:
            return self.request_q.qsize()
        except (NotImplementedError, OSError):
            return -1  # macOS / Manager.Queue don't always support qsize

    # --- request batching --------------------------------------------------

    def _collect_batch(self) -> list:
        """Drain up to ``batch_size`` requests, waiting at most ``batch_timeout_ms``.

        Returns the (possibly empty) batch. Falls through quickly when the
        queue is empty so the control loop stays responsive.
        """
        deadline = time.monotonic() + (self.batch_timeout_ms / 1000.0)
        batch: list[InferenceRequest] = []
        # First request: short blocking get so we don't busy-wait when idle.
        try:
            req = self.request_q.get(timeout=self.idle_poll_ms / 1000.0)
            batch.append(req)
        except queue.Empty:
            return batch

        # Subsequent requests: drain without further blocking until either
        # the batch fills or the timeout expires.
        while len(batch) < self.batch_size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                req = self.request_q.get(timeout=remaining)
                batch.append(req)
            except queue.Empty:
                break
        return batch

    def _run_forward(self, batch: list[InferenceRequest]) -> list[InferenceReply]:
        encoded_states = [req.encoded_state for req in batch]
        autocast_ctx = (
            torch.amp.autocast(self.autocast_device)
            if self.autocast_device
            else _NullCtx()
        )
        with torch.no_grad(), autocast_ctx:
            probs, log_probs, values = self._model.run_many_encoded(encoded_states)
        # Pull the model's batched price-components stash (might be None
        # for the GNN model) and slice per-leaf in numpy form so the reply
        # serializes cleanly.
        batched_price_components = getattr(self._model, "last_price_components", None)
        replies: list[InferenceReply] = []
        for i, req in enumerate(batch):
            replies.append(
                InferenceReply(
                    request_id=req.request_id,
                    probs=_to_numpy(probs[i]),
                    log_probs=_to_numpy(log_probs[i]),
                    value=_to_numpy(values[i]),
                    price_components=_slice_price_components_for_reply(batched_price_components, i),
                )
            )
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
            self.state = ServerState.IDLE

        while self.state != ServerState.SHUTDOWN:
            # 1. Drain control queue (non-blocking).
            self._poll_control()
            if self.state == ServerState.SHUTDOWN:
                break

            # 2. If draining, finish whatever's in the request queue then unload.
            if self.state == ServerState.DRAINING:
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
            batch = self._collect_batch()
            if not batch:
                continue
            t0 = time.monotonic()
            replies = self._run_forward(batch)
            elapsed_ms = (time.monotonic() - t0) * 1000.0
            self._dispatch_replies(batch, replies)
            self.stats.batches_served += 1
            self.stats.requests_served += len(batch)
            self.stats.total_batch_size += len(batch)
            self.stats.total_forward_ms += elapsed_ms

        LOGGER.info("InferenceServer.run: exited cleanly")

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
        probs, log_probs, values = self.run_many_encoded([encoded_game_state])
        return probs[0], log_probs[0], values[0]

    def run_many_encoded(self, encoded_game_states: list):
        """Submit one request per encoded state, collect replies in order."""
        if not encoded_game_states:
            raise ValueError("Received no game states to run.")
        # Submit all requests up-front so the server can batch them.
        handles: list[int] = []
        for state in encoded_game_states:
            rid = self._next_request_id
            self._next_request_id += 1
            self.request_q.put(InferenceRequest(rid, self.worker_id, state))
            handles.append(rid)

        # Collect replies. The reply queue is per-worker so every message
        # we pull is for us. Order is recovered by request_id.
        pending = set(handles)
        gathered: dict[int, InferenceReply] = {}
        deadline = time.monotonic() + self.client_config.request_timeout_s
        while pending:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"InferenceClient: timeout after {self.client_config.request_timeout_s}s "
                    f"waiting for {len(pending)}/{len(handles)} replies"
                )
            try:
                reply = self.reply_q.get(timeout=min(remaining, 1.0))
            except queue.Empty:
                continue
            if reply.request_id in pending:
                gathered[reply.request_id] = reply
                pending.discard(reply.request_id)
            else:
                # Stale reply (shouldn't happen with per-worker queues, but
                # be defensive against unanticipated paths).
                LOGGER.debug(
                    f"InferenceClient(worker={self.worker_id}): discarding stale "
                    f"reply for request {reply.request_id}"
                )

        ordered = [gathered[rid] for rid in handles]
        probs_list = [torch.from_numpy(r.probs) for r in ordered]
        log_probs_list = [torch.from_numpy(r.log_probs) for r in ordered]
        values_list = [torch.from_numpy(r.value) for r in ordered]
        self.last_price_components = _stack_price_components([r.price_components for r in ordered])
        return probs_list, log_probs_list, values_list


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
        # so the slicer falls back to the wide-Normal prior.
        return None
    means = torch.stack([torch.from_numpy(d["price_mean"]) for d in per_leaf])
    log_stds = torch.stack([torch.from_numpy(d["price_log_std"]) for d in per_leaf])
    return {
        "price_mean": means,
        "price_log_std": log_stds,
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

    def _send(self, op: str, payload: Optional[dict] = None, timeout_s: float = 30.0) -> Any:
        reply_q = self.ctx.Queue()
        msg = ControlMessage(op=op, payload=payload, reply_q=reply_q)
        self.control_q.put(msg)
        try:
            return reply_q.get(timeout=timeout_s)
        except queue.Empty:
            raise TimeoutError(f"ServerHandle.{op}: no reply in {timeout_s}s")

    def pause(self, timeout_s: float = 30.0):
        return self._send("pause", timeout_s=timeout_s)

    def reload(self, checkpoint_path: Optional[str], timeout_s: float = 60.0):
        return self._send("reload", payload={"checkpoint_path": checkpoint_path}, timeout_s=timeout_s)

    def health(self, timeout_s: float = 5.0) -> HealthReport:
        return self._send("health", timeout_s=timeout_s)

    def shutdown(self, timeout_s: float = 30.0):
        try:
            self._send("shutdown", timeout_s=timeout_s)
        except TimeoutError:
            LOGGER.warning("ServerHandle.shutdown: timeout; terminating process")
        finally:
            if self.process is not None and self.process.is_alive():
                self.process.join(timeout=5.0)
                if self.process.is_alive():
                    self.process.terminate()


def _server_main(
    request_q: Any,
    reply_qs: list,
    control_q: Any,
    model_factory: Any,
    checkpoint_path: Optional[str],
    batch_size: int,
    batch_timeout_ms: float,
    autocast_device: Optional[str],
):
    """Entry point for the inference server child process."""
    server = InferenceServer(
        request_q=request_q,
        reply_qs=reply_qs,
        control_q=control_q,
        model_factory=model_factory,
        checkpoint_path=checkpoint_path,
        batch_size=batch_size,
        batch_timeout_ms=batch_timeout_ms,
        autocast_device=autocast_device,
    )
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
) -> ServerHandle:
    """Spawn the inference server process and return a coordinator-side handle.

    Workers receive the request queue + their reply queue via
    ``ProcessPoolExecutor(initializer=worker_init_inference, initargs=...)``
    using ``ticket_q`` to self-assign a worker slot.
    """
    ctx = mp.get_context(mp_context)
    request_q = ctx.Queue()
    reply_qs = [ctx.Queue() for _ in range(num_workers)]
    control_q = ctx.Queue()
    ticket_q = ctx.Queue()
    for i in range(num_workers):
        ticket_q.put(i)

    proc = ctx.Process(
        target=_server_main,
        kwargs=dict(
            request_q=request_q,
            reply_qs=reply_qs,
            control_q=control_q,
            model_factory=model_factory,
            checkpoint_path=checkpoint_path,
            batch_size=batch_size,
            batch_timeout_ms=batch_timeout_ms,
            autocast_device=autocast_device,
        ),
        daemon=True,
        name="InferenceServer",
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
