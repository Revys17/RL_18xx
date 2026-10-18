"""Inference-server throughput on recorded lockstep requests, server process and
IPC included -- the search-learning layout without the trees.

``capture``: plays search learning's generator in-process (gen4 + its critic on
the GPU, the az1 settings) and records every lockstep request it sends (states,
legal moves, each request's player count): real leaf batches, ~200 states each
from 32 games' searches.

``serve``: for each variant, a real inference server process (``batch_size``
512 as search learning's) and ``--clients`` processes that replay the recorded
requests as fast as the server answers, ``--in-flight`` each; reports the states
the server answered per second and its stage split (``RL18XX_SERVER_PROFILE``).
Variants run back to back, ``--rounds`` times in alternating order, so they see
the same machine load. A variant is ``+``-joined flags:

- ``base``: the unpacked path (per-state tuples, ``fast_path=False``) -- before 2026-10-18
- ``fast``: stacked arrays straight to the model (``fast_path``, bit for bit the same replies)
- ``dedupe``: requests carry each distinct hex map once (``map_index``, as
  ``LockstepSearcher(dedupe_maps=True)`` sends them)
- ``pipeline``: a collector thread receives the next batch during the forward
- ``rep<N>``: N server processes (``start_inference_server(replicas=N)``), the clients split between them

    uv run python scripts/bench_inference_server.py capture --out logs/perf/requests_mid.npz \\
        --midgame-fraction 1 --auction-fraction 0 --skip-seconds 50 --seconds 170 --every 3
    uv run python scripts/bench_inference_server.py serve --requests logs/perf/requests_mid.npz \\
        --variants base fast fast+dedupe fast+dedupe+pipeline --seconds 30 --rounds 2
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

GEN4 = (
    "model_checkpoints_pg/rab_place_20261014/learner/1800.pth"
    "+model_checkpoints_pg/rab_place_20261014/critic/1800.pth"
)


# ----------------------------------------------------------------------------- capture


def capture(args):
    import torch

    from rl18xx.agent.alphazero import search_learning as sl
    from rl18xx.agent.alphazero.lockstep import ModelEvaluator
    from rl18xx.agent.alphazero.policy_gradient import load_server_model

    class Recorder(ModelEvaluator):
        def __init__(self, network):
            super().__init__(network)
            self.saved, self.n, self.t0 = [], 0, time.time()

        def send(self, packed):
            self.n += 1
            full = sum(len(p.extras) for p in self.saved) >= args.limit
            if time.time() - self.t0 >= args.skip_seconds and self.n % args.every == 0 and not full:
                self.saved.append(packed)
            return super().send(packed)

    torch.set_num_threads(2)
    net = load_server_model(args.model)
    cfg = sl.SearchLearningConfig(
        policy_checkpoint="", value_checkpoint="", learner_temperature=0.5, prior_temperature=0.5,
        target="visits", p_full=0.25, n_full=200, n_fast=48, c_puct_init=0.05, games_per_worker=args.games,
        start_positions="human_games/start_positions_1830_4p_v2.jsonl",
        midgame_positions="human_games/midgame_positions_1830_4p_v2.jsonl",
        midgame_fraction=args.midgame_fraction, auction_fraction=args.auction_fraction,
    )
    recorder = Recorder(net)
    gen = sl.Generator(sl.worker_settings(cfg, 0), {"learner": recorder})
    gen.run(args.seconds)
    packs = recorder.saved
    legal = [ix for p in packs for ix in p.legal_indices]
    np.savez(
        args.out,
        game_states=np.concatenate([p.game_states for p in packs]),
        node_features=np.concatenate([p.dense_node_features() for p in packs]),
        extras=np.concatenate([np.asarray(p.extras, dtype=np.int64) for p in packs]),
        legal=np.concatenate(legal).astype(np.int32),
        lengths=np.array([len(ix) for ix in legal]),
        batch_sizes=np.array([len(p.extras) for p in packs]),
    )
    print(f"saved {sum(len(p.extras) for p in packs)} states in {len(packs)} requests to {args.out}")


def load_requests(path, dedupe: bool) -> list:
    """The recorded requests as ``PackedStates`` (with each distinct map once if ``dedupe``)."""
    from rl18xx.agent.alphazero.inference_server import LegalIndices, PackedStates

    d = np.load(path)
    starts = np.concatenate([[0], np.cumsum(d["batch_sizes"])])
    legal_starts = np.concatenate([[0], np.cumsum(d["lengths"])])
    out = []
    for r in range(len(d["batch_sizes"])):
        s, e = starts[r], starts[r + 1]
        nodes, map_index = d["node_features"][s:e], None
        if dedupe:
            rows, unique, index = {}, [], []
            for row in nodes:
                key = row.tobytes()
                if key not in rows:
                    rows[key] = len(unique)
                    unique.append(row)
                index.append(rows[key])
            nodes, map_index = np.stack(unique), np.asarray(index, dtype=np.int64)
        legal = LegalIndices(d["legal"][legal_starts[s]:legal_starts[e]], d["lengths"][s:e])
        out.append(PackedStates(
            game_states=np.ascontiguousarray(d["game_states"][s:e]), node_features=np.ascontiguousarray(nodes),
            extras=np.ascontiguousarray(d["extras"][s:e]), legal_indices=legal, map_index=map_index,
        ))
    return out


# ----------------------------------------------------------------------------- serve


def _client(request_q, reply_qs, ticket_q, path, dedupe, seconds, in_flight, start_at, results):
    os.environ["OMP_NUM_THREADS"] = "1"
    from rl18xx.agent.alphazero.inference_server import InferenceClient

    worker_id = ticket_q.get()
    client = InferenceClient(request_q=request_q, reply_q=reply_qs[worker_id], worker_id=worker_id)
    packs = load_requests(path, dedupe)
    rng = np.random.default_rng(worker_id)
    order = rng.permutation(len(packs))
    while time.time() < start_at:
        time.sleep(0.01)
    pending, k, answered = [], 0, []
    end = start_at + seconds
    while time.time() < end or pending:
        while len(pending) < in_flight and time.time() < end:
            p = packs[order[k % len(packs)]]
            k += 1
            pending.append(client.send_packed(p))
        if not pending:
            break
        reply = client.receive_raw(pending.pop(0))
        answered.append((time.time(), len(reply.legal_lengths)))
    results.put(answered)


def serve_variant(args, variant: str) -> dict:
    from rl18xx.agent.alphazero.inference_server import start_inference_server
    from rl18xx.agent.alphazero.policy_gradient import load_server_model

    flags = set(variant.split("+"))
    replicas = next((int(f[3:]) for f in flags if f.startswith("rep")), 1)
    unknown = {f for f in flags if not f.startswith("rep")} - {"base", "fast", "dedupe", "pipeline"}
    if unknown:
        raise SystemExit(f"unknown variant flags {unknown}")
    os.environ["RL18XX_SERVER_PROFILE"] = str(args.profile_every)
    handle = start_inference_server(
        num_workers=args.clients, model_factory=load_server_model, checkpoint_path=args.model,
        batch_size=args.batch_size, fast_path="base" not in flags, pipeline="pipeline" in flags, replicas=replicas,
    )
    deadline = time.time() + 300
    while handle.health(timeout_s=60).state != "ACCEPTING":
        if time.time() > deadline:
            raise RuntimeError("the inference server did not start")
        time.sleep(0.5)
    ctx = mp.get_context("spawn")
    results = ctx.Queue()
    start_at = time.time() + args.warmup  # the clients' imports and request loading
    procs = [
        ctx.Process(
            target=_client,
            args=(handle.request_q, handle.reply_qs, handle.ticket_q, args.requests, "dedupe" in flags,
                  args.seconds, args.in_flight, start_at, results),
        )
        for _ in range(args.clients)
    ]
    for p in procs:
        p.start()
    answered = [a for _ in procs for a in results.get(timeout=args.seconds + args.warmup + 300)]
    for p in procs:
        p.join()
    handle.shutdown()
    # Steady state: the replies between the first one + --skip (CUDA warm-up) and the end.
    first = min(t for t, _ in answered)
    lo, hi = first + args.skip, start_at + args.seconds
    states = sum(n for t, n in answered if lo <= t <= hi)
    return {
        "variant": variant, "states_per_s": round(states / max(hi - lo, 1e-9)),
        "requests": sum(1 for t, _ in answered if lo <= t <= hi), "window_s": round(hi - lo, 1),
    }


def serve(args):
    rounds = []
    for r in range(args.rounds):
        variants = args.variants if r % 2 == 0 else args.variants[::-1]
        for v in variants:
            result = serve_variant(args, v)
            result["round"] = r
            print(json.dumps(result), flush=True)
            rounds.append(result)
    summary = {}
    for v in args.variants:
        rates = [x["states_per_s"] for x in rounds if x["variant"] == v]
        summary[v] = {"states_per_s": rates, "median": float(np.median(rates))}
    base = summary[args.variants[0]]["median"]
    for v, s in summary.items():
        print(f"{v:28s} {s['median']:8.0f} states/s  x{s['median'] / base:.2f}   {s['states_per_s']}")
    if args.out:
        Path(args.out).write_text(json.dumps({"rounds": rounds, "summary": summary}, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture")
    c.add_argument("--out", required=True)
    c.add_argument("--model", default=GEN4)
    c.add_argument("--seconds", type=float, default=120)
    c.add_argument("--skip-seconds", type=float, default=0)
    c.add_argument("--every", type=int, default=1)
    c.add_argument("--limit", type=int, default=15000)
    c.add_argument("--games", type=int, default=32)
    c.add_argument("--midgame-fraction", type=float, default=0.3)
    c.add_argument("--auction-fraction", type=float, default=0.15)
    s = sub.add_parser("serve")
    s.add_argument("--requests", required=True)
    s.add_argument("--model", default=GEN4)
    s.add_argument("--variants", nargs="+", default=["base", "fast", "fast+dedupe", "fast+dedupe+pipeline"])
    s.add_argument("--clients", type=int, default=8)
    s.add_argument("--in-flight", type=int, default=2)
    s.add_argument("--batch-size", type=int, default=512)
    s.add_argument("--seconds", type=float, default=30)
    s.add_argument("--warmup", type=float, default=10, help="Seconds for the clients to start (after the server is up)")
    s.add_argument("--rounds", type=int, default=2)
    s.add_argument("--skip", type=float, default=5, help="Seconds after the first reply not counted (warm-up)")
    s.add_argument("--profile-every", type=float, default=10)
    s.add_argument("--out", default=None)
    args = parser.parse_args()
    os.chdir(REPO_ROOT)
    capture(args) if args.cmd == "capture" else serve(args)


if __name__ == "__main__":
    main()
