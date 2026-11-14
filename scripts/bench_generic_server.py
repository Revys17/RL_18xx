"""Throughput of the generic sparse-legal inference path (``generic_inference``; RL bridge milestone C2).

Positions are 1830 random-walk positions at 2-6 players (legal rows from the 1830 layout).

* ``worker``: a worker's per-decision CPU cost -- encode (``encode_batch``), legal rows
  (``generic_play.TitlePlay.legal``), the split by server (``slice_positions`` + ``add_legal_inputs``) and the
  request's LZ4 compression -- per position, against the legacy worker's ``_rust_encode``.
* ``forward``: the server's forward in-process (``LocalGenericServer``: merge + forward + reply staging) at
  several batch sizes, positions/s.
* ``server``: a spawned server (``start_generic_inference_server``) and ``--workers`` client processes each
  sending requests of ``--request`` positions back to back (``--in-flight`` outstanding per worker):
  end-to-end positions/s with compression on / off, with and without the collector thread, replicas.

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 uv run python scripts/bench_generic_server.py worker forward \\
        --model small --batch 64 256 1024
    CUDA_VISIBLE_DEVICES= uv run python scripts/bench_generic_server.py server --model small --workers 8 \\
        --request 32 --configs plain compress compress+pipeline compress+rep2
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

SMALL = dict(
    d_model=32, num_heads=4, hex_layers=1, entity_layers=1, fusion_layers=1, d_trunk=64, trunk_blocks=1,
    d_head=32, d_edge=8, d_cond=16, d_id=8,
)  # fmt: skip


def positions(count: int, seed: int = 0, title: str = "1830") -> list:
    from rl18xx.agent.alphazero.generic_tokens import raw_game
    from rl18xx.agent.alphazero.titles import new_game

    rng = random.Random(seed)
    out = []
    if title != "1830":
        # A generator layout (1867): random walks through the title's own play adapter, 3-6 players.
        from rl18xx.agent.alphazero.generic_play import title_play

        play = title_play(title)
        while len(out) < count:
            game = raw_game(new_game(title, rng.choice((3, 4, 4, 5, 6)), seed=rng.randrange(1 << 30)))
            for step in range(rng.randrange(100, 900)):
                if game.finished:
                    break
                legal = play.legal(game)
                if len(legal) == 0:
                    break  # (a Ruby dead end the engine mirrors, e.g. 1822MX's NdeM holding cash at its close)
                k = rng.randrange(len(legal))
                price = play.fixed_price(game, legal, k)
                if play.open_price(legal, k):
                    lo, hi, st = (int(x) for x in legal.price_range[k])
                    price = lo + st * rng.randrange((hi - lo) // st + 1)
                play.apply(game, legal, k, price)
                if step % 13 == 0 and play.is_decision(play.legal(game)):
                    out.append(game.pickle_clone())
        return out[:count]
    while len(out) < count:
        n = rng.choice((2, 3, 4, 4, 5, 6))
        game = raw_game(new_game("1830", n, auction_unlock=True))
        for step in range(rng.randrange(100, 900)):
            if game.finished:
                break
            legal = [int(i) for i in game.factored_legal_indices()]
            index = rng.choice(legal)
            price_range = game.price_range_for_index(index)
            game.apply_action_index(index, rng.randint(*price_range) if price_range else None)
            if step % 13 == 0 and len(legal) > 1:
                out.append(game.pickle_clone())
    return out[:count]


def make_model(name: str):
    from rl18xx.agent.alphazero import model_generic as mg

    torch.manual_seed(0)
    config = mg.GenericModelConfig(**(SMALL if name in ("small", "null") else {}))
    return mg.GenericPolicyValueModel(config).eval()


class NullModel(torch.nn.Module):
    """No network: uniform policy and price cells, equal odds -- the server's transport / merge / reply
    ceiling (what one server process handles whatever the GPU does)."""

    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, inputs: dict, value_only: bool = False) -> dict:
        from rl18xx.agent.alphazero.model_generic import MASKED, segment_log_softmax

        b = inputs["meta"].shape[0]
        logits = torch.zeros(inputs["legal_rows"].shape[0])
        price = torch.zeros(inputs["price_cell_mask"].shape).masked_fill(~inputs["price_cell_mask"], MASKED)
        present = inputs["ent_mask"][:, 1:7]
        return {
            "logits": logits,
            "logp": segment_log_softmax(logits, inputs["legal_pos"], b),
            "price_logp": torch.log_softmax(price, dim=-1),
            "win_logits": torch.zeros(b, 6).masked_fill(~present, MASKED),
            "score": torch.zeros(b, 6),
        }


def null_factory(_path):
    return NullModel().eval()


def bench_worker(games: list, reps: int = 5, title: str = "1830", reference: bool = False) -> dict:
    """Per position: ``legal`` (the legal list: for a generator layout the generator itself, ``reference``
    = its pre-2026-11-13 paths, ``engine_rs.BaseGame.g1867_set_reference_generator``), encode, split,
    compress; for a generator layout also ``fixed_price`` (the worker's price of a row without an open
    price, on every row) and ``apply`` (one random row applied, on a clone made outside the timing)."""
    from rl18xx.agent.alphazero import model_generic as mg
    from rl18xx.agent.alphazero.generic_inference import pack_batch, slice_positions
    from rl18xx.agent.alphazero.generic_play import title_play
    from rl18xx.agent.alphazero.generic_tokens import encode_batch

    import engine_rs

    play = title_play(title)
    n = len(games)
    t = {"legal": 0.0, "encode": 0.0, "split_legal_inputs": 0.0, "compress": 0.0}
    if title == "1830":
        t["legacy_encode"] = 0.0
    else:
        t["fixed_price"] = t["apply"] = 0.0
    sizes = []
    rng = random.Random(0)
    previous = engine_rs.BaseGame.g1867_set_reference_generator(reference)
    # (the 1822 family's generator has its own reference paths, 2026-11-14)
    previous_1822 = engine_rs.BaseGame.g1822_set_reference_generator(reference)
    try:
        for _ in range(reps):
            t0 = time.perf_counter()
            legal = [play.legal(g) for g in games]
            t1 = time.perf_counter()
            enc = encode_batch(games)
            t2 = time.perf_counter()
            half = list(range(0, n, 2)), list(range(1, n, 2))
            parts = [mg.add_legal_inputs(slice_positions(enc, idx), [legal[i] for i in idx]) for idx in half]
            t3 = time.perf_counter()
            blobs = [pack_batch(p) for p in parts]
            t4 = time.perf_counter()
            t["legal"] += t1 - t0
            t["encode"] += t2 - t1
            t["split_legal_inputs"] += t3 - t2
            t["compress"] += t4 - t3
            sizes = [len(b.blob) for b in blobs]
            if title == "1830":
                from rl18xx.agent.alphazero.mcts import _rust_encode
                from rl18xx.rust_adapter import RustGameAdapter

                t5 = time.perf_counter()
                adapters = [RustGameAdapter(g) for g in games]
                [_rust_encode(a) for a in adapters]
                t["legacy_encode"] += time.perf_counter() - t5
                continue
            for g, lg in zip(games, legal):
                ks = [rng.randrange(len(lg)) for _ in range(1)]
                t5 = time.perf_counter()
                if reference:
                    # before 2026-11-13 fixed_price asked the engine (another generator run)
                    rs = [g.rl_price_range(int(lg.indices[k])) for k in ks]
                    prices = [None if r is None else int(r[0]) for r in rs]
                else:
                    prices = [play.fixed_price(g, lg, k) for k in ks]
                t["fixed_price"] += time.perf_counter() - t5
                c = g.pickle_clone()
                t6 = time.perf_counter()
                play.apply(c, lg, ks[0], prices[0])
                t["apply"] += time.perf_counter() - t6
    finally:
        engine_rs.BaseGame.g1867_set_reference_generator(previous)
        engine_rs.BaseGame.g1822_set_reference_generator(previous_1822)
    raw = sum(v.nbytes for v in enc.values() if isinstance(v, np.ndarray))
    out = {f"{k}_us_per_position": round(v / reps / n * 1e6, 1) for k, v in t.items()}
    parts_ = ("legal", "encode", "split_legal_inputs", "compress") + (("fixed_price", "apply") if title != "1830" else ())
    out["generic_total_us_per_position"] = round(sum(out[f"{k}_us_per_position"] for k in parts_), 1)
    out["token_bytes_per_position"] = round(raw / n)
    out["request_bytes_per_position_lz4"] = round(sum(sizes) / n)
    return out


def bench_forward(model, games: list, batches: list, reps: int = 3) -> dict:
    from rl18xx.agent.alphazero import model_generic as mg
    from rl18xx.agent.alphazero.generic_inference import LocalGenericServer

    server = LocalGenericServer(model, seed=0)
    out = {}
    for b in batches:
        chunk = (games * (b // len(games) + 1))[:b]
        # requests of 32 positions merged into one batch, as the server sees several workers' requests
        requests = []
        for i in range(0, b, 32):
            part = chunk[i : i + 32]
            requests.append((mg.build_inputs(part, [mg.legal_actions(g) for g in part]), 0.0))
        server.run_many(requests[:1])
        t0 = time.perf_counter()
        for _ in range(reps):
            server.run_many(requests)
        out[b] = round(b * reps / (time.perf_counter() - t0), 1)
    return out


# --------------------------------------------------------------------------------------- server mode


def _client_main(queues, worker_id, arrays_path, request, in_flight, seconds, compress, result_q):
    import pickle

    from rl18xx.agent.alphazero.generic_inference import GenericInferenceClient

    torch.set_num_threads(1)
    request_q, reply_qs, ticket_q = queues
    wid = ticket_q.get()
    client = GenericInferenceClient(request_q=request_q, reply_q=reply_qs[wid], worker_id=wid)
    with open(arrays_path, "rb") as f:
        requests = pickle.load(f)
    rng = random.Random(worker_id)
    done, t_end = 0, None
    pending = []
    t0 = time.perf_counter()
    while True:
        while len(pending) < in_flight:
            arrays = requests[rng.randrange(len(requests))]
            pending.append(client.send_inputs(arrays, compress=compress))
        reply = client.receive_outputs(pending.pop(0))
        done += len(reply.win)
        if time.perf_counter() - t0 > seconds:
            t_end = time.perf_counter()
            break
    for p in pending:
        client.receive_outputs(p)
    result_q.put((done, t_end - t0))


def bench_server(checkpoint: str, games: list, args, config: str) -> dict:
    import pickle

    from rl18xx.agent.alphazero import model_generic as mg
    from rl18xx.agent.alphazero.generic_inference import load_generic_server_model_cpu, start_generic_inference_server

    opts = set(config.split("+"))
    replicas = 2 if "rep2" in opts else 3 if "rep3" in opts else 1
    requests = []
    for i in range(0, len(games) - args.request + 1, args.request):
        part = games[i : i + args.request]
        requests.append(mg.build_inputs(part, [mg.legal_actions(g) for g in part]))
    fd, path = tempfile.mkstemp(suffix=".pkl")
    with os.fdopen(fd, "wb") as f:
        pickle.dump(requests, f)
    ctx = mp.get_context("spawn")
    handle = start_generic_inference_server(
        num_workers=args.workers,
        checkpoint_path=checkpoint,
        model_factory=null_factory if args.model == "null" else load_generic_server_model_cpu,
        batch_size=args.server_batch,
        pipeline="pipeline" in opts,
        replicas=replicas,
        threads=args.server_threads,
    )
    try:
        deadline = time.monotonic() + 120
        while handle.health(timeout_s=60).state != "ACCEPTING":
            assert time.monotonic() < deadline
            time.sleep(0.1)
        result_q = ctx.Queue()
        queues = (handle.request_q, handle.reply_qs, handle.ticket_q)
        procs = [
            ctx.Process(
                target=_client_main,
                args=(queues, w, path, args.request, args.in_flight, args.seconds, "compress" in opts, result_q),
            )
            for w in range(args.workers)
        ]
        for p in procs:
            p.start()
        results = [result_q.get(timeout=args.seconds + 300) for _ in procs]
        for p in procs:
            p.join()
        health = handle.health(timeout_s=30)
    finally:
        handle.shutdown()
        os.unlink(path)
    total = sum(r[0] for r in results)
    wall = max(r[1] for r in results)
    return {
        "config": config,
        "positions_per_second": round(total / wall, 1),
        "avg_batch": round(health.avg_batch_size, 1),
        "avg_forward_ms": round(health.avg_forward_ms, 2),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("modes", nargs="+", choices=["worker", "forward", "server"])
    parser.add_argument(
        "--model", choices=["small", "default", "null"], default="small",
        help="null (server mode): no network, the server process's transport ceiling",
    )
    parser.add_argument("--positions", type=int, default=512)
    parser.add_argument("--batch", type=int, nargs="+", default=[64, 256, 1024])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--request", type=int, default=32, help="Positions per worker request")
    parser.add_argument("--in-flight", type=int, default=1)
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--server-batch", type=int, default=512)
    parser.add_argument("--server-threads", type=int, default=2)
    parser.add_argument("--configs", nargs="+", default=["plain", "compress", "compress+pipeline"])
    parser.add_argument("--threads", type=int, default=2, help="torch threads for the forward mode")
    parser.add_argument(
        "--title", default="1830", help="worker mode: 1830 (default) or a generator title (1867, 1822, 1822PNW, ...)"
    )
    parser.add_argument(
        "--reference", action="store_true",
        help="worker mode, generator titles: also time the generator's reference (pre-2026-11-13) paths",
    )
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    games = positions(args.positions, title=args.title)
    print(json.dumps({"positions": len(games), "players": sorted({len(g.players) for g in games})}))
    model = make_model(args.model)
    print(json.dumps({"model": args.model, "parameters": model.num_parameters()}))
    if "worker" in args.modes:
        print(json.dumps({"worker": bench_worker(games[:256], title=args.title)}))
        if args.reference and args.title != "1830":
            print(json.dumps({"worker_reference_generator": bench_worker(games[:256], title=args.title, reference=True)}))
    if "forward" in args.modes:
        print(json.dumps({"forward_positions_per_second": bench_forward(model, games, args.batch)}))
    if "server" in args.modes:
        from rl18xx.agent.alphazero import model_generic as mg

        fd, ckpt = tempfile.mkstemp(suffix=".pth")
        os.close(fd)
        mg.save_checkpoint(model, ckpt, titles=["1830"])
        try:
            for config in args.configs:
                print(json.dumps({"server": bench_server(ckpt, games, args, config)}), flush=True)
        finally:
            os.unlink(ckpt)


if __name__ == "__main__":
    main()
