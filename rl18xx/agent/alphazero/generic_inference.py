"""Sparse-legal inference server for title-generic checkpoints (RL bridge milestone C2; ``docs/rl_model.md``
"Policy gradient for the generic model").

Many CPU workers play games against networks hosted by one server process each (a GPU in a real run), as the
legacy policy-gradient stage does with ``inference_server.py``. A worker sends a batch of positions as the
generic model's own input arrays -- the encoder's token arrays plus the flat legal descriptor rows of
``model_generic.build_inputs`` / ``add_legal_inputs``, built in the worker -- and gets back, per position:

* the log-probability of every legal row (``logp``, flat in the request's legal order; normalized per
  position at temperature 1 -- the worker tempers);
* for every price-bearing legal row, a price **cell** sampled on the server from the price head over the
  row's legal range (mixed with the request's ``price_eps`` uniform over the range's non-empty cells, as
  ``policy_gradient.choose_price`` mixes) and the log-probability it was drawn with; the worker draws the
  price inside the cell (``price_pmf.PriceCells.sample_price``) only for the row it plays. The full
  ``(P, 24)`` cell log-probabilities come back too on request (``want_price_logp``);
* per-seat win shares and score-head outputs over the 6 seats (encoder frame: seat 0 = the mover).

Kept from the legacy server (``inference_server.InferenceServer``, subclassed: control ops, batching,
reload, replicas, profile and the process handles are the legacy code):

* **numpy in, never torch tensors on a queue**: a request is one LZ4-compressed pickle of the arrays (the
  generic encoding is ~58 KB per 1830 position raw, ~2 KB compressed at ~7 us per position in the worker:
  the queue moves ~30x fewer bytes; ``compress=False`` sends the arrays as they are);
* **one GPU wait per batch**: every reply array (row log-probs, sampled cells and their log-probs, values,
  optional cell log-probs) is gathered on the device into one float32 buffer and copied to pinned host memory
  by one non-blocking transfer queued behind the forward (``_StagedReplies``' idea);
* ``OMP_NUM_THREADS=2`` for the server processes, replicas (``replicas`` > 1: worker ``w`` -> replica
  ``w % R``), an optional collector thread (``pipeline``) that decompresses and merges the next batch while
  the current one runs.

Requests from different workers are merged on the server (:func:`merge_inputs`): token arrays padded to
the batch maxima (hexes, entities, relations), legal rows / map keys / price rows offset, so a batch may mix
titles and player counts. The merged forward equals the per-request forwards (tested).
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import pickle
import time
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

import lz4.frame
import numpy as np
import torch

from rl18xx.agent.alphazero import inference_server as legacy
from rl18xx.agent.alphazero.inference_server import (
    InferenceClient,
    InferenceRequest,
    InferenceServer,
    ReplicatedServerHandle,
    ServerHandle,
)

LOGGER = logging.getLogger(__name__)

HEX_KEYS = ("hex_feat", "hex_ptr", "hex_res", "hex_slot_city", "hex_ids", "hex_mask")
ENTITY_KEYS = ("ent_type", "ent_feat", "ent_ptr", "ent_hex", "ent_mask")
RELATION_KEYS = ("rel", "rel_val", "rel_mask")
TOKEN_KEYS = HEX_KEYS + ENTITY_KEYS + RELATION_KEYS + ("meta",)
#: The flat legal-list arrays ``model_generic.add_legal_inputs`` adds.
LEGAL_KEYS = (
    "legal_rows", "legal_pos", "legal_offsets", "map_rows", "map_k1", "map_k2", "k1_rep", "k1_group", "k2_rep",
    "k2_k1", "price_rows", "price_range", "price_cell_mask",
)  # fmt: skip
_PAD_MINUS_ONE = {"hex_ptr", "hex_res", "hex_slot_city", "ent_ptr", "ent_hex"}
MAX_SEATS = 6


# ---------------------------------------------------------------------------------------------------
# Requests and replies
# ---------------------------------------------------------------------------------------------------


@dataclass
class GenericBatch:
    """One worker request: positions as ``build_inputs`` arrays (with legal lists), compressed or not."""

    size: int
    blob: Optional[bytes] = None  # lz4(pickle(arrays))
    arrays: Optional[dict] = None  # the arrays themselves (compress=False)
    #: Uniform mixing into this request's price-cell draws (0 for a learner: its draws are its policy).
    price_eps: float = 0.0
    #: Also return the full (P, NUM_CELLS) cell log-probabilities of the priced rows.
    want_price_logp: bool = False

    def __len__(self) -> int:
        return self.size

    def load(self) -> dict:
        if self.arrays is not None:
            return self.arrays
        return pickle.loads(lz4.frame.decompress(self.blob))


def pack_batch(
    arrays: dict, price_eps: float = 0.0, want_price_logp: bool = False, compress: bool = True
) -> GenericBatch:
    """A request of ``arrays`` (``model_generic.build_inputs`` with legal lists, ``map_index`` None or set)."""
    if "legal_rows" not in arrays:
        raise ValueError("a generic inference request needs legal lists (model_generic.add_legal_inputs)")
    size = int(len(arrays["meta"]))
    if compress:
        blob = lz4.frame.compress(pickle.dumps(arrays, protocol=pickle.HIGHEST_PROTOCOL))
        return GenericBatch(size=size, blob=blob, price_eps=float(price_eps), want_price_logp=bool(want_price_logp))
    return GenericBatch(size=size, arrays=arrays, price_eps=float(price_eps), want_price_logp=bool(want_price_logp))


@dataclass
class GenericReply:
    """The server's answer to one :class:`GenericBatch` (arrays in the request's order)."""

    request_id: int
    #: (N,) float32: log-probability of every legal row at temperature 1, per position's legal list.
    logp: np.ndarray
    #: (P,) int64 / float32: per priced row (the request's ``price_rows`` order) a sampled cell and the
    #: log-probability of the (eps-mixed) cell distribution it was drawn from.
    price_cell: np.ndarray
    price_cell_logp: np.ndarray
    #: (B, 6) float32: win shares (softmax over the present seats) and score-head outputs, encoder frame.
    win: np.ndarray
    score: np.ndarray
    #: (P, NUM_CELLS) float32 when the request asked (``want_price_logp``), else None.
    price_logp: Optional[np.ndarray] = None


# ---------------------------------------------------------------------------------------------------
# Merging requests
# ---------------------------------------------------------------------------------------------------


def _pad_axis1(a: np.ndarray, width: int, key: str) -> np.ndarray:
    if a.shape[1] >= width:
        return a
    pad = np.full((a.shape[0], width - a.shape[1]) + a.shape[2:], -1 if key in _PAD_MINUS_ONE else 0, dtype=a.dtype)
    return np.concatenate([a, pad], axis=1)


def merge_inputs(parts: Sequence[dict]) -> Tuple[dict, List[tuple]]:
    """Concatenate several requests' ``build_inputs`` arrays (each with legal lists) into one batch.

    Token arrays are padded to the batch maxima with ``encode_batch``'s own padding (hexes, entities,
    relations), ``map_index`` offset (a part without one gets ``arange``), and the legal lists offset:
    positions, legal rows, map keys (groups, (group, hex), (group, hex, tile | city)) and price rows.
    Returns ``(arrays, spans)``: ``spans[i] = (b0, b1, n0, n1, p0, p1)``, part ``i``'s positions, legal
    rows and price rows in the merged batch."""
    if not parts:
        raise ValueError("nothing to merge")
    for p in parts:
        if "legal_rows" not in p:
            raise ValueError("every part needs legal lists")
    sizes = [(len(p["meta"]), len(p["legal_rows"]), len(p["price_rows"])) for p in parts]
    spans, b0, n0, p0 = [], 0, 0, 0
    for b, n, pr in sizes:
        spans.append((b0, b0 + b, n0, n0 + n, p0, p0 + pr))
        b0, n0, p0 = b0 + b, n0 + n, p0 + pr
    if len(parts) == 1:
        return parts[0], spans

    out: dict = {}
    for keys in (HEX_KEYS, ENTITY_KEYS, RELATION_KEYS):
        width = max(int(p[keys[0]].shape[1]) for p in parts)
        for key in keys:
            out[key] = np.concatenate([_pad_axis1(p[key], width, key) for p in parts], axis=0)
    out["meta"] = np.concatenate([p["meta"] for p in parts], axis=0)
    if any(p.get("map_index") is not None for p in parts):
        index, m0 = [], 0
        for p in parts:
            mi = p.get("map_index")
            index.append((np.arange(len(p["meta"]), dtype=np.int64) if mi is None else np.asarray(mi, np.int64)) + m0)
            m0 += int(p["hex_feat"].shape[0])
        out["map_index"] = np.concatenate(index)
    else:
        out["map_index"] = None

    legal_rows, legal_pos, offsets = [], [], []
    map_rows, map_k1, map_k2, k1_rep, k1_group, k2_rep, k2_k1 = ([] for _ in range(7))
    price_rows, price_range, cell_mask = [], [], []
    b_off = n_off = k1_off = k2_off = g_off = 0
    for p in parts:
        n = len(p["legal_rows"])
        legal_rows.append(p["legal_rows"])
        legal_pos.append(p["legal_pos"] + b_off)
        offsets.append(p["legal_offsets"][:-1] + n_off)
        map_rows.append(p["map_rows"] + n_off)
        map_k1.append(p["map_k1"] + k1_off)
        map_k2.append(p["map_k2"] + k2_off)
        k1_rep.append(p["k1_rep"] + n_off)
        k1_group.append(p["k1_group"] + g_off)
        k2_rep.append(p["k2_rep"] + n_off)
        k2_k1.append(p["k2_k1"] + k1_off)
        price_rows.append(p["price_rows"] + n_off)
        price_range.append(p["price_range"])
        cell_mask.append(p["price_cell_mask"])
        b_off += len(p["meta"])
        n_off += n
        k1_off += len(p["k1_rep"])
        k2_off += len(p["k2_rep"])
        g_off += int(p["k1_group"].max()) + 1 if len(p["k1_group"]) else 0
    out.update(
        legal_rows=np.concatenate(legal_rows),
        legal_pos=np.concatenate(legal_pos),
        legal_offsets=np.concatenate(offsets + [np.array([n_off], dtype=np.int64)]),
        map_rows=np.concatenate(map_rows),
        map_k1=np.concatenate(map_k1),
        map_k2=np.concatenate(map_k2),
        k1_rep=np.concatenate(k1_rep),
        k1_group=np.concatenate(k1_group),
        k2_rep=np.concatenate(k2_rep),
        k2_k1=np.concatenate(k2_k1),
        price_rows=np.concatenate(price_rows),
        price_range=np.concatenate(price_range).reshape(-1, 3),
        price_cell_mask=np.concatenate(cell_mask).reshape(-1, cell_mask[0].shape[-1] if cell_mask else 24),
    )
    return out, spans


def slice_positions(arrays: dict, positions: Sequence[int]) -> dict:
    """The token arrays (``encode_batch`` without ``dedupe_maps``) of some positions of a batch, relations
    trimmed to the subset's longest list -- e.g. a worker's pending positions split by the seat's server."""
    if arrays.get("map_index") is not None:
        raise ValueError("slice_positions needs an encoding without map de-duplication")
    idx = np.asarray(positions, dtype=np.int64)
    out = {key: arrays[key][idx] for key in TOKEN_KEYS}
    width = int(out["rel_mask"].sum(axis=1).max()) if len(idx) else 0
    for key in RELATION_KEYS:
        out[key] = out[key][:, :width]
    out["map_index"] = None
    return out


# ---------------------------------------------------------------------------------------------------
# Price-cell sampling (device)
# ---------------------------------------------------------------------------------------------------


def sample_price_cells(
    price_logp: torch.Tensor, cell_mask: torch.Tensor, eps: torch.Tensor, generator: Optional[torch.Generator] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(cell, log P(cell))`` per priced row: a draw from the price head over the row's non-empty cells
    mixed with ``eps`` (per row) uniform -- ``policy_gradient.choose_price``'s distribution."""
    mask = cell_mask.bool()
    probs = torch.where(mask, price_logp.float().exp(), torch.zeros_like(price_logp, dtype=torch.float32))
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-30)
    live = mask.float().sum(dim=-1, keepdim=True).clamp(min=1.0)
    e = eps.float()[:, None]
    mixed = torch.where(mask, (1.0 - e) * probs + e / live, torch.zeros_like(probs))
    mixed = mixed / mixed.sum(dim=-1, keepdim=True)
    cell = torch.multinomial(mixed, 1, generator=generator).squeeze(1)
    return cell, mixed.gather(1, cell[:, None]).squeeze(1).log()


# ---------------------------------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------------------------------


class _Staged:
    """A batch's reply arrays on their way to the host: one float32 buffer, one transfer (``_StagedReplies``)."""

    __slots__ = ("host", "event", "sizes")

    def __init__(self, host, event, sizes):
        self.host, self.event, self.sizes = host, event, sizes

    def arrays(self) -> dict:
        if self.event is not None:
            self.event.synchronize()
        data = self.host.numpy()
        out, start = {}, 0
        for name, shape in self.sizes:
            n = int(np.prod(shape))
            out[name] = data[start : start + n].reshape(shape)
            start += n
        return out


def _stage(parts: list) -> _Staged:
    """``parts``: ``[(name, tensor)]`` on one device -> one float32 buffer copied to the host (pinned,
    non-blocking on CUDA, one event)."""
    flat = torch.cat([t.detach().float().reshape(-1) for _, t in parts]) if parts else torch.zeros(0)
    sizes = [(name, tuple(t.shape)) for name, t in parts]
    if flat.device.type == "cuda":
        host = torch.empty(flat.shape, dtype=torch.float32, pin_memory=True)
        host.copy_(flat, non_blocking=True)
        event = torch.cuda.Event()
        event.record()
        return _Staged(host, event, sizes)
    return _Staged(flat.cpu(), None, sizes)


class GenericInferenceServer(InferenceServer):
    """``InferenceServer`` for :class:`GenericBatch` requests and a ``GenericPolicyValueModel``: the legacy
    server's lifecycle, control ops, batching and dispatch, with a generic forward (:meth:`_run_forward`)."""

    def __init__(self, *args, seed: Optional[int] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.seed = seed
        self._generator: Optional[torch.Generator] = None

    def _device(self) -> torch.device:
        return next(self._model.parameters()).device

    def _gen(self, device: torch.device) -> torch.Generator:
        if self._generator is None or self._generator.device != device:
            self._generator = torch.Generator(device=device)
            if self.seed is not None:
                self._generator.manual_seed(int(self.seed))
            else:
                self._generator.seed()
        return self._generator

    def _prepare(self, batch: list) -> dict:
        """Decompress and merge the batch's requests (the collector thread's work with ``pipeline``)."""
        reqs = [req.encoded_states for req in batch]
        for r in reqs:
            if not isinstance(r, GenericBatch):
                raise TypeError(f"generic inference server got a {type(r).__name__} request")
        arrays, spans = merge_inputs([r.load() for r in reqs])
        eps = np.concatenate(
            [np.full(p1 - p0, r.price_eps, dtype=np.float32) for r, (_, _, _, _, p0, p1) in zip(reqs, spans)]
        )
        return {"arrays": arrays, "spans": spans, "eps": eps, "want_price_logp": any(r.want_price_logp for r in reqs)}

    def _run_forward(self, batch: list, prepared: Optional[dict] = None) -> list:
        from rl18xx.agent.alphazero import model_generic as mg

        profile = legacy._PROFILE
        tu = time.monotonic()
        if prepared is None:
            prepared = self._prepare(batch)
        arrays = prepared["arrays"]
        device = self._device()
        with torch.no_grad():
            inputs = mg.to_device(arrays, device)
            tf = time.monotonic()
            out = self._model(inputs)
            parts = [
                ("logp", out["logp"]),
                ("win", torch.softmax(out["win_logits"].float(), dim=-1)),
                ("score", out["score"]),
            ]
            if out["price_logp"] is not None and out["price_logp"].shape[0]:
                eps = torch.from_numpy(prepared["eps"]).to(device)
                cell, cell_logp = sample_price_cells(
                    out["price_logp"], inputs["price_cell_mask"], eps, self._gen(device)
                )
                parts += [("price_cell", cell), ("price_cell_logp", cell_logp)]
                if prepared["want_price_logp"]:
                    parts.append(("price_logp", out["price_logp"]))
            staged = _stage(parts)
        if profile:
            if staged.event is not None:
                staged.event.synchronize()
            profile.add("unpack", tf - tu)
            profile.add("forward", time.monotonic() - tf)
        tr = time.monotonic()
        data = staged.arrays()
        empty = np.zeros(0, dtype=np.float32)
        cells, cell_logp = data.get("price_cell", empty), data.get("price_cell_logp", empty)
        price_logp = data.get("price_logp", np.zeros((0, mg.NUM_CELLS), dtype=np.float32))
        replies = []
        for req, (b0, b1, n0, n1, p0, p1) in zip(batch, prepared["spans"]):
            replies.append(
                GenericReply(
                    request_id=req.request_id,
                    logp=data["logp"][n0:n1],
                    price_cell=cells[p0:p1].astype(np.int64),
                    price_cell_logp=cell_logp[p0:p1],
                    win=data["win"][b0:b1],
                    score=data["score"][b0:b1],
                    price_logp=price_logp[p0:p1] if req.encoded_states.want_price_logp else None,
                )
            )
        if profile:
            profile.add("replies", time.monotonic() - tr)
        return replies


def use_tf32() -> None:
    """TF32 matmuls in generation and training alike (``policy_gradient.use_tf32``: one precision for the ratio)."""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def load_generic_server_model(checkpoint_path: str):
    """Inference-server loader (module-level so it pickles): a generic checkpoint on the GPU if there is one,
    fp32 / TF32, eval mode."""
    from rl18xx.agent.alphazero import model_generic as mg

    use_tf32()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = mg.load_checkpoint(str(checkpoint_path), device)
    model.eval()
    return model


def load_generic_server_model_cpu(checkpoint_path: str):
    """``load_generic_server_model`` pinned to the CPU (tests, CPU benchmarks)."""
    from rl18xx.agent.alphazero import model_generic as mg

    model, _ = mg.load_checkpoint(str(checkpoint_path), "cpu")
    model.eval()
    return model


def _generic_server_main(
    request_q: Any,
    reply_qs: list,
    control_q: Any,
    model_factory: Any,
    checkpoint_path: Optional[str],
    batch_size: int,
    batch_timeout_ms: float,
    control_reply_q: Any = None,
    pipeline: bool = False,
    seed: Optional[int] = None,
    threads: Optional[int] = None,
):
    """Entry point of a generic inference server process."""
    if threads:
        torch.set_num_threads(int(threads))
    server = GenericInferenceServer(
        request_q=request_q,
        reply_qs=reply_qs,
        control_q=control_q,
        control_reply_q=control_reply_q,
        model_factory=model_factory,
        checkpoint_path=checkpoint_path,
        batch_size=batch_size,
        batch_timeout_ms=batch_timeout_ms,
        autocast_device=None,
        pipeline=pipeline,
        seed=seed,
    )
    server.exit_on_load_failure = True  # the process exits, so the coordinator sees it die
    server.run()


def start_generic_inference_server(
    *,
    num_workers: int,
    checkpoint_path: Optional[str],
    model_factory: Any = load_generic_server_model,
    batch_size: int = 512,
    batch_timeout_ms: float = 2.0,
    mp_context: str = "spawn",
    pipeline: bool = False,
    replicas: int = 1,
    seed: Optional[int] = None,
    threads: Optional[int] = None,
):
    """Spawn a generic inference server (``replicas`` > 1: that many processes on one checkpoint) and return
    the legacy coordinator-side handle (``ServerHandle`` / ``ReplicatedServerHandle``: ``reload``, ``health``,
    ``shutdown``, the workers' queues and slot tickets). ``batch_size`` counts positions."""
    ctx = mp.get_context(mp_context)
    reply_qs = [ctx.Queue() for _ in range(num_workers)]
    ticket_q = ctx.Queue()
    for i in range(num_workers):
        ticket_q.put(i)
    handles = []
    for r in range(max(1, int(replicas))):
        request_q, control_q, control_reply_q = ctx.Queue(), ctx.Queue(), ctx.Queue()
        proc = ctx.Process(
            target=_generic_server_main,
            kwargs=dict(
                request_q=request_q,
                reply_qs=reply_qs,
                control_q=control_q,
                control_reply_q=control_reply_q,
                model_factory=model_factory,
                checkpoint_path=checkpoint_path,
                batch_size=batch_size,
                batch_timeout_ms=batch_timeout_ms,
                pipeline=pipeline,
                seed=None if seed is None else int(seed) + r,
                threads=threads,
            ),
            daemon=True,
            name="GenericInferenceServer" if replicas == 1 else f"GenericInferenceServer{r}",
        )
        proc.start()
        handles.append(
            ServerHandle(
                process=proc,
                request_q=request_q,
                reply_qs=reply_qs,
                control_q=control_q,
                ticket_q=ticket_q,
                num_workers=num_workers,
                ctx=ctx,
                control_reply_q=control_reply_q,
            )
        )
    return handles[0] if len(handles) == 1 else ReplicatedServerHandle(handles)


# ---------------------------------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------------------------------


class GenericInferenceClient(InferenceClient):
    """A worker's connection to a generic server (the legacy client's queues, slot and reply matching)."""

    def send_inputs(
        self, arrays: dict, price_eps: float = 0.0, want_price_logp: bool = False, compress: bool = True
    ) -> tuple:
        """Submit ``arrays`` (``build_inputs`` with legal lists) without waiting; :meth:`receive_outputs`
        answers it."""
        batch = pack_batch(arrays, price_eps=price_eps, want_price_logp=want_price_logp, compress=compress)
        if batch.size == 0:
            raise ValueError("Received no positions to run.")
        rid = self._next_request_id
        self._next_request_id += 1
        self.request_q.put(InferenceRequest(request_id=rid, worker_id=self.worker_id, encoded_states=batch))
        return rid, None, batch.size

    def receive_outputs(self, pending) -> GenericReply:
        return self._wait_reply(pending)


class LocalGenericServer:
    """The server's forward in-process (no queues): ``run(arrays, price_eps)`` -> :class:`GenericReply`.
    For tests, CPU benchmarks and single-process tools."""

    def __init__(self, model, seed: Optional[int] = None):
        self.server = GenericInferenceServer(
            request_q=None, reply_qs=[], control_q=None, model_factory=lambda _path: model, checkpoint_path=None,
            seed=seed,
        )
        self.server._model = model

    def run_many(
        self, requests: Sequence[Tuple[dict, float]], want_price_logp: bool = False, compress: bool = True
    ) -> list:
        batch = [
            InferenceRequest(
                request_id=i,
                worker_id=0,
                encoded_states=pack_batch(arrays, price_eps=eps, want_price_logp=want_price_logp, compress=compress),
            )
            for i, (arrays, eps) in enumerate(requests)
        ]
        return self.server._run_forward(batch)

    def run(self, arrays: dict, price_eps: float = 0.0, want_price_logp: bool = False) -> GenericReply:
        return self.run_many([(arrays, price_eps)], want_price_logp=want_price_logp)[0]


class LocalGenericClient:
    """A :class:`GenericInferenceClient` stand-in that runs each request in-process on a
    :class:`LocalGenericServer` (games played without server processes: tests, single-process tools)."""

    def __init__(self, server: LocalGenericServer):
        self.server = server
        self._pending: dict = {}
        self._next = 0

    def send_inputs(self, arrays: dict, price_eps: float = 0.0, want_price_logp: bool = False, compress: bool = True):
        rid = self._next
        self._next += 1
        self._pending[rid] = (pack_batch(arrays, price_eps, want_price_logp, compress), arrays)
        return rid, None, len(arrays["meta"])

    def receive_outputs(self, pending) -> GenericReply:
        batch, _ = self._pending.pop(pending[0])
        request = InferenceRequest(request_id=pending[0], worker_id=0, encoded_states=batch)
        return self.server.server._run_forward([request])[0]


__all__ = [
    "GenericBatch",
    "GenericInferenceClient",
    "GenericInferenceServer",
    "GenericReply",
    "LocalGenericClient",
    "LocalGenericServer",
    "load_generic_server_model",
    "merge_inputs",
    "pack_batch",
    "sample_price_cells",
    "slice_positions",
    "start_generic_inference_server",
]
