"""The inference server's fast paths against the reference paths they replace.

- ``pad_states_numpy`` lays a state out for the max player count as
  ``_pad_state_to_max_players`` does (every player count).
- ``run_packed`` (stacked arrays, as the server receives lockstep requests)
  gives ``run_many_encoded``'s outputs (per-state tuples) bit for bit, for a
  transformer and a policy + value composite, mixed player counts included;
  so does the server's ``_run_forward_packed`` against its unpacked path.
- Map dedupe (``mcts_collect_leaves`` / ``encode_games`` with ``dedupe_maps``):
  ``node_features[map_index]`` is the plain layout bit for bit, each distinct
  map once; the network on it gives the plain batch's outputs (within float
  rounding: the map encoder's matmuls see fewer rows), and lockstep trees
  searched with it end identical.
"""

import queue

import numpy as np
import pytest
import torch

engine_rs = pytest.importorskip("engine_rs")

from rl18xx.agent.alphazero.composite_model import PolicyValueComposite  # noqa: E402
from rl18xx.agent.alphazero.config import ModelTransformerConfig  # noqa: E402
from rl18xx.agent.alphazero.inference_server import (  # noqa: E402
    InferenceRequest,
    InferenceServer,
    PackedStates,
    run_packed_states,
)
from rl18xx.agent.alphazero.lockstep import LockstepSearcher, ModelEvaluator, pack  # noqa: E402
from rl18xx.agent.alphazero.model_transformer import (  # noqa: E402
    MAX_PLAYERS,
    AlphaZeroTransformerModel,
    _layout_for,
    _pad_state_to_max_players,
    pad_states_numpy,
)

from .search_test_utils import HashNet, make_player, random_positions, tree_summary  # noqa: E402


def _small_model(seed: int) -> AlphaZeroTransformerModel:
    torch.manual_seed(seed)
    config = ModelTransformerConfig(
        device=torch.device("cpu"),
        d_entity=32,
        econ_transformer_layers=1,
        econ_transformer_ff_dim=64,
        d_map=32,
        hex_transformer_layers=2,
        hex_transformer_heads=4,
        hex_transformer_ff_dim=64,
        d_trunk=64,
        num_res_blocks=2,
    )
    model = AlphaZeroTransformerModel(config)
    # Non-zero value / price outputs (the heads' last layers start at zero).
    for head in (model.win_loss_head[-1], model.score_head[-1], model.price_head.out[-1]):
        torch.nn.init.normal_(head.weight, std=0.1)
    return model.eval()


@pytest.fixture(scope="module")
def models():
    return _small_model(0), _small_model(1)


@pytest.fixture(scope="module")
def packs():
    """Requests as lockstep sends them: positions of 4- and 3-player games
    (stock, operating rounds, the auction), several sharing a map."""
    out = []
    for num_players, seed in ((4, 0), (4, 1), (3, 2)):
        games = random_positions(num_players, seed, steps=240, every=6)
        legal = [np.asarray(g.factored_legal_indices(), dtype=np.int32) for g in games]
        out.append(pack(engine_rs.encode_games(games), legal))
    return out


@pytest.mark.parametrize("num_players", range(2, MAX_PLAYERS + 1))
def test_numpy_padding_matches_the_reference(num_players):
    rng = np.random.default_rng(num_players)
    states = rng.normal(size=(5, _layout_for(num_players)[2])).astype(np.float32)
    expected = _pad_state_to_max_players(torch.from_numpy(states), num_players, MAX_PLAYERS).numpy()
    assert np.array_equal(pad_states_numpy(states, num_players), expected)


def _unpacked(model, packs_):
    states = [s for p in packs_ for s in p.unpack()]
    legal = [ix for p in packs_ for ix in p.legal_indices]
    with torch.no_grad():
        probs, log_probs, values = model.run_many_encoded(states, legal_indices=legal)
    price = model.last_price_components["price_logits"].clone()
    return probs, values, price, log_probs


def _packed(model, packs_):
    with torch.no_grad():
        probs, log_probs, values = run_packed_states(model, packs_)
    return probs, values, model.last_price_components["price_logits"].clone(), log_probs


def test_packed_inputs_give_the_unpacked_outputs_bit_for_bit(models, packs):
    policy, critic = models
    for model in (policy, PolicyValueComposite(policy, critic), PolicyValueComposite(policy, critic, "score")):
        for subset in (packs[:1], packs[2:], packs):  # 4 players, 3 players, both in one batch
            a, b = _unpacked(model, subset), _packed(model, subset)
            for x, y in zip(a[:3], b[:3]):
                assert torch.equal(x, y)
            assert a[3] is None and b[3] is None


def test_map_dedupe_layout(packs):
    for num_players, seed in ((4, 0), (3, 2)):
        games = random_positions(num_players, seed, steps=240, every=6)
        games = games + games[:3]  # repeated positions: their maps are stored once
        plain = engine_rs.encode_games(games)
        deduped = engine_rs.encode_games(games, True)
        assert "map_index" not in plain
        index = deduped["map_index"]
        assert np.array_equal(deduped["node_features"][index], plain["node_features"])
        for key in ("game_states", "round_types", "rotations"):
            assert np.array_equal(deduped[key], plain[key])
        rows = deduped["node_features"].reshape(len(deduped["node_features"]), -1)
        assert len({r.tobytes() for r in rows}) == len(rows) < len(games)
        first = [int(np.flatnonzero(index == k)[0]) for k in range(len(rows))]
        assert first == sorted(first)  # maps numbered in order of first appearance


def test_map_dedupe_gives_the_same_network_outputs(models, packs):
    policy, critic = models
    model = PolicyValueComposite(policy, critic)
    deduped = []
    for p in packs:
        rows, unique, index = {}, [], []
        for r in p.node_features:
            if r.tobytes() not in rows:
                rows[r.tobytes()] = len(unique)
                unique.append(r)
            index.append(rows[r.tobytes()])
        unique, index = np.stack(unique), np.asarray(index, dtype=np.int64)
        assert len(unique) < len(p)
        deduped.append(
            PackedStates(p.game_states, unique, p.extras, legal_indices=p.legal_indices, map_index=index)
        )
        assert np.array_equal(deduped[-1].dense_node_features(), p.node_features)
    a, b = _packed(model, packs), _packed(model, deduped)
    for x, y in zip(a[:3], b[:3]):
        torch.testing.assert_close(x, y, rtol=1e-5, atol=1e-6)


def test_lockstep_with_map_dedupe_ends_with_the_same_trees():
    net = HashNet()
    positions = random_positions(4, 3, steps=200, every=20) + random_positions(3, 4, steps=150, every=30)
    config = dict(parallel_readouts=8, num_readouts=24, min_readouts=24, mcts_mean_q=True, dedupe_leaves=True)
    trees = {}
    for dedupe_maps in (False, True):
        players = [make_player(net, game, seed=k, **config) for k, game in enumerate(positions)]
        searcher = LockstepSearcher({None: ModelEvaluator(net)}, dedupe_maps=dedupe_maps)
        searcher.run([p.start_search() for p in players])
        trees[dedupe_maps] = [tree_summary(p) for p in players]
        if dedupe_maps:
            assert searcher.stats.maps < searcher.stats.leaves
    for a, b in zip(trees[False], trees[True]):
        assert a["n"] == b["n"] and a["arena"] == b["arena"] and a["prices"] == b["prices"]
        assert np.array_equal(a["child_n"], b["child_n"]) and np.array_equal(a["q"], b["q"])


def _serve(model, packs_, fast_path: bool) -> list:
    server = InferenceServer(
        request_q=queue.Queue(), reply_qs=[queue.Queue()], control_q=queue.Queue(),
        model_factory=lambda _: model, checkpoint_path=None, fast_path=fast_path,
    )
    server._load_model()
    batch = [InferenceRequest(request_id=i, worker_id=0, encoded_states=p) for i, p in enumerate(packs_)]
    return server._run_forward(batch)


def test_server_fast_path_replies_are_the_unpacked_replies(models, packs):
    model = PolicyValueComposite(*models)
    for fast, slow in zip(_serve(model, packs, True), _serve(model, packs, False)):
        assert fast.request_id == slow.request_id
        assert np.array_equal(fast.probs, slow.probs)
        assert np.array_equal(fast.value, slow.value)
        assert np.array_equal(fast.legal_lengths, slow.legal_lengths)
        assert np.array_equal(fast.price_components["price_logits"], slow.price_components["price_logits"])


def test_packed_states_pickle_compactly_and_unpickle_to_the_same_states(packs):
    import pickle

    from rl18xx.agent.alphazero.inference_server import LegalIndices

    for p in packs:
        q = pickle.loads(pickle.dumps(p))
        assert isinstance(q.legal_indices, LegalIndices) and isinstance(q.extras, np.ndarray)
        assert [ix.tolist() for ix in q.legal_indices] == [list(ix) for ix in p.legal_indices]
        for a, b in zip(q.unpack(), p.unpack()):
            assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
            assert [int(x) for x in a[4:]] == [int(x) for x in b[4:]]


# ----------------------------------------------------------------------------- pipelining


def _threaded_server(model, pipeline: bool):
    import threading
    import time

    from rl18xx.agent.alphazero.inference_server import ServerState

    server = InferenceServer(
        request_q=queue.Queue(), reply_qs=[queue.Queue(), queue.Queue()], control_q=queue.Queue(),
        model_factory=lambda _: model, checkpoint_path=None, batch_size=40, batch_timeout_ms=1.0,
        idle_poll_ms=0.5, pipeline=pipeline,
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 30
    while server.state != ServerState.ACCEPTING:
        assert time.monotonic() < deadline
        time.sleep(0.01)
    return server, thread


def _control(server, op, payload=None):
    from rl18xx.agent.alphazero.inference_server import ControlMessage

    reply = queue.Queue()
    server.control_q.put(ControlMessage(op=op, payload=payload, reply_q=reply))
    return reply.get(timeout=30)


def test_the_pipelined_server_gives_the_same_replies(models, packs):
    """A collector thread receives and stacks the next batch while one runs: the
    replies (and which worker gets them) are the unpipelined server's, through a
    pause (drain), a reload and more requests."""
    from rl18xx.agent.alphazero.inference_server import InferenceClient

    model = PolicyValueComposite(*models)
    out = {}
    for pipeline in (False, True):
        server, thread = _threaded_server(model, pipeline)
        clients = [InferenceClient(server.request_q, server.reply_qs[w], worker_id=w) for w in (0, 1)]
        got = []
        try:
            for phase in range(2):
                pending = [(w, clients[w].send_packed(p)) for _ in range(3) for w, p in enumerate(packs[:2])]
                pending += [(1, clients[1].send_packed(packs[2]))]
                for w, ticket in pending:
                    reply = clients[w].receive_raw(ticket)
                    got.append((w, reply.probs.copy(), reply.value.copy(), reply.price_components["price_logits"].copy()))
                if phase == 0:
                    assert _control(server, "pause") == {"ok": True}
                    assert _control(server, "reload", {"checkpoint_path": None}) == {"ok": True}
        finally:
            _control(server, "shutdown")
            thread.join(timeout=10)
        assert not thread.is_alive()
        out[pipeline] = got
    assert len(out[True]) == len(out[False]) == 14
    for a, b in zip(out[False], out[True]):
        assert a[0] == b[0]
        for x, y in zip(a[1:], b[1:]):
            assert np.array_equal(x, y)


def _pool_round_trip(signature):
    from rl18xx.agent.alphazero.inference_server import get_worker_client, get_worker_id

    _, _, values = get_worker_client().run_many_encoded([signature])
    return get_worker_id(), float(values[0][0])


def test_a_replicated_server_answers_every_worker():
    """``replicas=2``: two server processes, worker w sending to replica w % 2;
    every worker gets its own answers, and control ops reach both replicas."""
    import multiprocessing as mp
    import time
    from concurrent.futures import ProcessPoolExecutor

    from rl18xx.agent.alphazero.inference_server import (
        ReplicatedServerHandle,
        start_inference_server,
        worker_init_inference,
    )

    from .test_inference_server import _mock_factory

    handle = start_inference_server(
        num_workers=4, model_factory=_mock_factory, checkpoint_path=None, batch_size=4, batch_timeout_ms=2.0,
        replicas=2,
    )
    assert isinstance(handle, ReplicatedServerHandle) and len(handle.request_q) == 2
    try:
        deadline = time.monotonic() + 120
        while handle.health(timeout_s=60).state != "ACCEPTING":
            assert time.monotonic() < deadline
            time.sleep(0.2)
        pool = ProcessPoolExecutor(
            4, mp_context=mp.get_context("spawn"), initializer=worker_init_inference,
            initargs=(handle.request_q, handle.reply_qs, handle.ticket_q),
        )
        try:
            results = list(pool.map(_pool_round_trip, range(3, 23), timeout=180))
        finally:
            for proc in list(pool._processes.values()):
                proc.kill()
            pool.shutdown(wait=False, cancel_futures=True)
        assert [v for _, v in results] == [float(s) for s in range(3, 23)]
        assert len({w for w, _ in results}) > 1
        assert handle.pause(timeout_s=30) == {"ok": True}
        assert handle.reload(None, timeout_s=30) == {"ok": True}
        report = handle.health(timeout_s=30)
        assert report.state == "ACCEPTING" and report.batches_served >= 1
    finally:
        handle.shutdown(timeout_s=30)
    assert not any(p.is_alive() for p in handle.process.processes)
