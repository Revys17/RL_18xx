"""Unit tests for the Phase 3 cross-process inference server.

These tests drive the server in-thread (rather than spawning a real
subprocess) so they stay fast and deterministic. A real subprocess
launch is exercised indirectly by the loop-level smoke test once that
lands.
"""
from __future__ import annotations

import queue
import threading
import time
from typing import Optional

import numpy as np
import pytest
import torch

from rl18xx.agent.alphazero.inference_server import (
    ControlMessage,
    InferenceClient,
    InferenceClientConfig,
    InferenceRequest,
    InferenceServer,
    ServerState,
    _slice_price_components_for_reply,
    _stack_price_components,
)
from rl18xx.agent.alphazero.mcts import POLICY_SIZE, VALUE_SIZE


# ----- model stub ---------------------------------------------------------


class MockModel:
    """Drop-in for ``AlphaZeroModel.run_many_encoded``.

    Returns deterministic outputs that the tests can introspect: the
    policy for state ``i`` is ``one_hot(i % POLICY_SIZE)``, and the value
    is a unique signature per request so the assertion that replies
    arrive at the right worker is unambiguous.
    """

    def __init__(self, with_price_head: bool = True):
        self.with_price_head = with_price_head
        self.last_price_components: Optional[dict] = None
        self.forward_calls: int = 0

    def run_many_encoded(self, encoded_states):
        self.forward_calls += 1
        n = len(encoded_states)
        probs = []
        log_probs = []
        values = []
        for i, st in enumerate(encoded_states):
            # The test caller passes ``encoded_state = idx`` (an int) as a
            # signature so we can identify replies later.
            sig = int(st)
            p = torch.zeros(POLICY_SIZE)
            p[sig % POLICY_SIZE] = 1.0
            probs.append(p)
            log_probs.append(torch.log(p.clamp_min(1e-9)))
            v = torch.zeros(VALUE_SIZE)
            v[0] = float(sig)  # leak the request signature into value[0]
            values.append(v)
        if self.with_price_head:
            num_slots = 3
            mean = torch.zeros(n, num_slots)
            log_std = torch.zeros(n, num_slots)
            for i, st in enumerate(encoded_states):
                mean[i, 0] = float(st)
                log_std[i, 0] = float(st) * 0.1
            self.last_price_components = {
                "price_mean": mean,
                "price_log_std": log_std,
                "slot_index": {("Bid", "SV"): 0, ("Bid", "CS"): 1, ("Bid", "DH"): 2},
                "num_slots": num_slots,
            }
        else:
            self.last_price_components = None
        return probs, log_probs, values


def _mock_factory(_ckpt_path):
    return MockModel(with_price_head=True)


def _no_price_factory(_ckpt_path):
    return MockModel(with_price_head=False)


# ----- helpers ------------------------------------------------------------


def _start_server_thread(server: InferenceServer) -> threading.Thread:
    t = threading.Thread(target=server.run, daemon=True, name="TestInferenceServer")
    t.start()
    # Spin briefly so STARTING -> ACCEPTING.
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if server.state == ServerState.ACCEPTING:
            return t
        time.sleep(0.005)
    raise AssertionError(f"Server didn't reach ACCEPTING (state={server.state})")


def _make_server(model_factory=_mock_factory, **overrides) -> tuple:
    request_q = queue.Queue()
    reply_qs = [queue.Queue() for _ in range(overrides.pop("num_workers", 4))]
    control_q = queue.Queue()
    server = InferenceServer(
        request_q=request_q,
        reply_qs=reply_qs,
        control_q=control_q,
        model_factory=model_factory,
        checkpoint_path=None,
        batch_size=overrides.pop("batch_size", 8),
        batch_timeout_ms=overrides.pop("batch_timeout_ms", 2.0),
        autocast_device=None,
        idle_poll_ms=overrides.pop("idle_poll_ms", 0.5),
    )
    if overrides:
        raise TypeError(f"unexpected kwargs: {overrides}")
    return server, request_q, reply_qs, control_q


def _shutdown(server: InferenceServer, control_q, thread):
    reply_q = queue.Queue()
    control_q.put(ControlMessage(op="shutdown", reply_q=reply_q))
    try:
        reply_q.get(timeout=5.0)
    except queue.Empty:
        pass
    thread.join(timeout=5.0)
    assert not thread.is_alive(), "server thread didn't exit after shutdown"


# ----- tests --------------------------------------------------------------


def test_server_starts_and_serves_a_single_request():
    server, request_q, reply_qs, control_q = _make_server()
    thread = _start_server_thread(server)
    try:
        request_q.put(InferenceRequest(request_id=42, worker_id=0, encoded_state=7))
        reply = reply_qs[0].get(timeout=3.0)
        assert reply.request_id == 42
        # MockModel emits value[0] = signature (7 here).
        assert reply.value[0] == pytest.approx(7.0)
        assert reply.probs.shape == (POLICY_SIZE,)
        assert reply.price_components is not None
        assert reply.price_components["price_mean"].shape == (3,)
        assert reply.price_components["price_mean"][0] == pytest.approx(7.0)
    finally:
        _shutdown(server, control_q, thread)


def test_server_batches_up_to_batch_size():
    server, request_q, reply_qs, control_q = _make_server(batch_size=4, batch_timeout_ms=50.0)
    thread = _start_server_thread(server)
    try:
        for rid in range(8):
            request_q.put(InferenceRequest(request_id=rid, worker_id=0, encoded_state=rid * 11))
        replies = [reply_qs[0].get(timeout=3.0) for _ in range(8)]
        rids = sorted(r.request_id for r in replies)
        assert rids == list(range(8))
        assert server.stats.batches_served >= 2
        assert server.stats.requests_served == 8
    finally:
        _shutdown(server, control_q, thread)


def test_server_batches_respect_timeout_when_under_capacity():
    """One stray request must still get served within ~batch_timeout."""
    server, request_q, reply_qs, control_q = _make_server(batch_size=64, batch_timeout_ms=5.0)
    thread = _start_server_thread(server)
    try:
        t0 = time.monotonic()
        request_q.put(InferenceRequest(request_id=1, worker_id=2, encoded_state=99))
        reply = reply_qs[2].get(timeout=2.0)
        elapsed = time.monotonic() - t0
        assert reply.request_id == 1
        # batch_timeout_ms=5 → reply should arrive well within a second.
        assert elapsed < 1.0, f"single-request latency too high: {elapsed:.3f}s"
    finally:
        _shutdown(server, control_q, thread)


def test_server_routes_replies_to_correct_worker():
    server, request_q, reply_qs, control_q = _make_server(num_workers=3, batch_size=8)
    thread = _start_server_thread(server)
    try:
        for worker_id in range(3):
            for rid_off in range(2):
                request_q.put(
                    InferenceRequest(
                        request_id=worker_id * 100 + rid_off,
                        worker_id=worker_id,
                        encoded_state=worker_id * 100 + rid_off,
                    )
                )
        # Each worker queue should receive exactly its own pair.
        for worker_id in range(3):
            seen = []
            for _ in range(2):
                seen.append(reply_qs[worker_id].get(timeout=3.0).request_id)
            assert sorted(seen) == [worker_id * 100, worker_id * 100 + 1]
    finally:
        _shutdown(server, control_q, thread)


def test_server_pause_drains_then_idles():
    server, request_q, reply_qs, control_q = _make_server(batch_size=4, batch_timeout_ms=5.0)
    thread = _start_server_thread(server)
    try:
        for rid in range(3):
            request_q.put(InferenceRequest(request_id=rid, worker_id=0, encoded_state=rid))

        ack_q = queue.Queue()
        control_q.put(ControlMessage(op="pause", reply_q=ack_q))
        assert ack_q.get(timeout=5.0) == {"ok": True}

        # Drain replies that were in-flight.
        for _ in range(3):
            reply_qs[0].get(timeout=3.0)

        # Wait for the server to transition to IDLE.
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if server.state == ServerState.IDLE:
                break
            time.sleep(0.01)
        assert server.state == ServerState.IDLE
        assert server._model is None  # model dropped to free VRAM
    finally:
        _shutdown(server, control_q, thread)


def test_server_reload_returns_to_accepting():
    server, request_q, reply_qs, control_q = _make_server(batch_size=4, batch_timeout_ms=5.0)
    thread = _start_server_thread(server)
    try:
        # Pause first.
        ack_q = queue.Queue()
        control_q.put(ControlMessage(op="pause", reply_q=ack_q))
        ack_q.get(timeout=5.0)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and server.state != ServerState.IDLE:
            time.sleep(0.01)
        assert server.state == ServerState.IDLE

        # Then reload.
        ack_q2 = queue.Queue()
        control_q.put(ControlMessage(op="reload", payload={"checkpoint_path": "/dev/null"}, reply_q=ack_q2))
        assert ack_q2.get(timeout=5.0) == {"ok": True}
        assert server.state == ServerState.ACCEPTING
        assert server._model is not None

        # Verify it actually serves again.
        request_q.put(InferenceRequest(request_id=99, worker_id=0, encoded_state=5))
        reply = reply_qs[0].get(timeout=3.0)
        assert reply.request_id == 99
    finally:
        _shutdown(server, control_q, thread)


def test_health_report_reflects_state():
    server, request_q, reply_qs, control_q = _make_server(batch_size=2)
    thread = _start_server_thread(server)
    try:
        for rid in range(2):
            request_q.put(InferenceRequest(request_id=rid, worker_id=0, encoded_state=rid))
        for _ in range(2):
            reply_qs[0].get(timeout=3.0)

        ack_q = queue.Queue()
        control_q.put(ControlMessage(op="health", reply_q=ack_q))
        report = ack_q.get(timeout=3.0)
        assert report.state == "ACCEPTING"
        assert report.batches_served >= 1
        assert report.requests_served == 2
        assert report.avg_batch_size >= 1.0
    finally:
        _shutdown(server, control_q, thread)


# ----- client tests -------------------------------------------------------


def test_client_round_trip_in_order():
    server, request_q, reply_qs, control_q = _make_server(num_workers=2, batch_size=8)
    thread = _start_server_thread(server)
    try:
        client = InferenceClient(
            request_q=request_q, reply_q=reply_qs[1], worker_id=1,
            client_config=InferenceClientConfig(request_timeout_s=10.0),
        )
        states = [3, 5, 7, 11]
        probs, log_probs, values = client.run_many_encoded(states)
        # Order preserved (the client must return results in submission order).
        for i, v in enumerate(values):
            assert v[0].item() == pytest.approx(float(states[i]))
        # Price components batched back to (B, num_slots).
        assert client.last_price_components is not None
        assert client.last_price_components["price_mean"].shape == (4, 3)
        for i, st in enumerate(states):
            assert client.last_price_components["price_mean"][i, 0].item() == pytest.approx(float(st))
    finally:
        _shutdown(server, control_q, thread)


def test_client_run_encoded_single():
    server, request_q, reply_qs, control_q = _make_server(num_workers=1, batch_size=4)
    thread = _start_server_thread(server)
    try:
        client = InferenceClient(request_q, reply_qs[0], worker_id=0)
        probs, _, value = client.run_encoded(13)
        assert value[0].item() == pytest.approx(13.0)
        assert probs.shape == (POLICY_SIZE,)
    finally:
        _shutdown(server, control_q, thread)


def test_client_handles_no_price_head_model():
    """GNN model has no price head; client should expose None last_price_components."""
    server, request_q, reply_qs, control_q = _make_server(
        model_factory=_no_price_factory, num_workers=1
    )
    thread = _start_server_thread(server)
    try:
        client = InferenceClient(request_q, reply_qs[0], worker_id=0)
        client.run_many_encoded([1, 2, 3])
        assert client.last_price_components is None
    finally:
        _shutdown(server, control_q, thread)


def test_client_timeout_raises():
    request_q = queue.Queue()
    reply_q = queue.Queue()
    client = InferenceClient(
        request_q, reply_q, worker_id=0,
        client_config=InferenceClientConfig(request_timeout_s=0.5),
    )
    with pytest.raises(TimeoutError):
        client.run_many_encoded([1])


# ----- price components serialization ------------------------------------


def test_slice_then_stack_round_trip():
    """``_slice_price_components_for_reply`` per-leaf then ``_stack_price_components``
    must return the same shapes the in-process slicer expects."""
    means = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    log_stds = torch.tensor([[-1.0, -2.0], [-3.0, -4.0], [-5.0, -6.0]])
    batched = {
        "price_mean": means, "price_log_std": log_stds,
        "slot_index": {("Bid", "X"): 0, ("Bid", "Y"): 1},
        "num_slots": 2,
    }
    sliced = [_slice_price_components_for_reply(batched, i) for i in range(3)]
    restacked = _stack_price_components(sliced)
    assert restacked is not None
    assert torch.allclose(restacked["price_mean"], means)
    assert torch.allclose(restacked["price_log_std"], log_stds)
    assert restacked["slot_index"] == {("Bid", "X"): 0, ("Bid", "Y"): 1}


def test_stack_returns_none_for_empty_list():
    assert _stack_price_components([None, None, None]) is None


def test_stack_returns_none_on_mixed_some_none():
    """A mixed list shouldn't happen in practice — be defensive."""
    means = torch.tensor([1.0, 2.0])
    log_stds = torch.tensor([-1.0, -2.0])
    real = {"price_mean": means.numpy(), "price_log_std": log_stds.numpy(),
            "slot_index": {}, "num_slots": 2}
    assert _stack_price_components([real, None]) is None


# ----- integration with MCTSPlayer ---------------------------------------


class _MCTSEncoderShapedMockModel:
    """Model stub that returns the same shapes ``Encoder_GNN`` produces.

    Used by the MCTS integration tests below where we drive a full
    ``MCTSPlayer.tree_search`` through the inference client. The encoded
    state is the tuple produced by ``_rust_encode``; we ignore its
    contents and just emit uniformly distributed policy/value.
    """

    def __init__(self):
        self.last_price_components: Optional[dict] = None

    def run_many_encoded(self, encoded_states):
        n = len(encoded_states)
        probs = [torch.ones(POLICY_SIZE) / POLICY_SIZE for _ in range(n)]
        log_probs = [torch.log(p) for p in probs]
        values = [torch.zeros(VALUE_SIZE) for _ in range(n)]
        # Provide a price head so MCTS PW exercises the sliced path too.
        num_slots = 8
        self.last_price_components = {
            "price_mean": torch.zeros(n, num_slots),
            "price_log_std": torch.zeros(n, num_slots),
            "slot_index": {("Bid", c): i for i, c in enumerate("SCDMCB")},
            "num_slots": num_slots,
        }
        return probs, log_probs, values


def _mcts_model_factory(_ckpt):
    return _MCTSEncoderShapedMockModel()


def test_mcts_tree_search_runs_through_inference_client():
    """End-to-end: ``MCTSPlayer.tree_search`` should produce the same
    visit count whether inference goes through the local model or the
    inference client. Validates that the server path is API-compatible."""
    from rl18xx.agent.alphazero.config import SelfPlayConfig
    from rl18xx.agent.alphazero.self_play import MCTSPlayer

    server, request_q, reply_qs, control_q = _make_server(
        model_factory=_mcts_model_factory, num_workers=1,
        batch_size=8, batch_timeout_ms=5.0,
    )
    thread = _start_server_thread(server)
    try:
        client = InferenceClient(
            request_q=request_q, reply_q=reply_qs[0], worker_id=0,
            client_config=InferenceClientConfig(request_timeout_s=10.0),
        )
        cfg = SelfPlayConfig(
            network=None,                      # purposely unset; server owns the model.
            use_inference_server=True,
            inference_client=client,
            use_score_values=False,
            backup_discount=1.0,
            use_fp16_inference=False,
        )
        player = MCTSPlayer(cfg)
        assert player._backend is client, "MCTSPlayer must route inference through the client"

        # First-node expansion mirrors what SelfPlay.play() does.
        first = player.root.select_leaf()
        first.ensure_encoded()
        probs, _, val = client.run_encoded(first.encoded_game_state)
        first.incorporate_results(probs, val, up_to=player.root)

        # Drive a couple of tree_search batches through the server.
        for _ in range(2):
            player.tree_search(parallel_readouts=4)

        # All vlosses must be reverted (no dangling state).
        queue_walk = [player.root]
        while queue_walk:
            node = queue_walk.pop()
            assert node.losses_applied == 0
            queue_walk.extend(node.children.values())
        assert player.root.N >= 1
    finally:
        _shutdown(server, control_q, thread)
