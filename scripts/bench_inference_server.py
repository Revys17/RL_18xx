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
- ``legacyreply``: replies through ``_batched_legal_replies`` (a GPU round trip per transfer, before
  2026-10-19) instead of staged replies (the reply's device work queued behind the forward, one copy)
- ``noprice``: replies without their price rows (99% of their bytes; the network still computes
  them) -- an upper bound on what a smaller reply transport could save, not a valid server

    uv run python scripts/bench_inference_server.py capture --out logs/perf/requests_mid.npz \\
        --midgame-fraction 1 --auction-fraction 0 --skip-seconds 50 --seconds 170 --every 3
    uv run python scripts/bench_inference_server.py serve --requests logs/perf/requests_mid.npz \\
        --variants base fast fast+dedupe fast+dedupe+pipeline --seconds 30 --rounds 2

``replies``: the reply path alone in this process (no server, no IPC): each recorded batch
forwarded once per reply path, timed step by step, the replies checked bit for bit equal,
and their pickling timed. With a run's own settings and network (az8: shared network, n_full 800):

    uv run python scripts/bench_inference_server.py capture --out logs/perf/requests_az8.npz \\
        --model model_checkpoints_search/az8_20261019/learner/10.pth \\
        --run-config model_checkpoints_search/az8_20261019/config.json --skip-seconds 40 --every 2 --limit 12000
    uv run python scripts/bench_inference_server.py replies --requests logs/perf/requests_az8.npz \\
        --model model_checkpoints_search/az8_20261019/learner/10.pth
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


# ----------------------------------------------------------------------------- reply size bound


class _NoPriceRows:
    """The served model without price rows in its replies (the ``noprice`` variant):
    the network still computes them, the replies leave them out -- what a reply
    without its largest part (99% of its bytes) could save in transport. Not a
    valid server for a search."""

    def __init__(self, model):
        self.model = model
        self.last_price_components = None

    def __getattr__(self, name):
        return getattr(self.model, name)

    def run_packed(self, inputs, legal_indices=None):
        out = self.model.run_packed(inputs, legal_indices)
        self.last_price_components = None
        return out


def noprice_model_factory(spec):
    from rl18xx.agent.alphazero.policy_gradient import load_server_model

    return _NoPriceRows(load_server_model(spec))


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
    if args.run_config:
        # A run's own settings (its config.json), e.g. az8's: n_full 800 / n_fast 64 / p_full 0.1,
        # Gumbel, tree reuse and map dedupe; --games overrides its games per worker.
        cfg = run_config(args.run_config, games_per_worker=args.games)
    else:
        cfg = sl.SearchLearningConfig(
            policy_checkpoint="", value_checkpoint="", learner_temperature=0.5, prior_temperature=0.5,
            target="visits", p_full=0.25, n_full=200, n_fast=48, c_puct_init=0.05, games_per_worker=args.games or 32,
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


def run_config(path, **overrides):
    """A search-learning run's ``config.json`` as a ``SearchLearningConfig`` (keys the
    current config lacks dropped), with ``overrides`` that are not None."""
    import dataclasses

    from rl18xx.agent.alphazero import search_learning as sl

    raw = json.loads(Path(path).read_text())
    fields = {f.name for f in dataclasses.fields(sl.SearchLearningConfig)}
    cfg = {k: tuple(v) if isinstance(v, list) else v for k, v in raw.items() if k in fields}
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    return sl.SearchLearningConfig(**cfg)


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


# ----------------------------------------------------------------------------- replies


def replies(args):
    """The reply path alone, in this process: the recorded requests grouped into
    server batches of ~``--batch-states``, each forwarded once per reply path and
    timed (CUDA synchronized, as the profiled server is): ``legacy``
    (``_batched_legal_replies`` after the forward's sync, its steps also timed one by
    one with a sync after each) and ``staged`` (``_stage_replies`` queued behind the
    forward, one sync). Checks the two give the same replies bit for bit, and times
    pickling them (the reply queue's feeder thread) and unpickling (the worker)."""
    import pickle
    from multiprocessing.reduction import ForkingPickler

    import torch

    from rl18xx.agent.alphazero import inference_server as srv
    from rl18xx.agent.alphazero.policy_gradient import load_server_model

    torch.set_num_threads(2)
    model = load_server_model(args.model)
    packs = load_requests(args.requests, dedupe=not args.no_dedupe)
    batches, cur = [], []
    for k, p in enumerate(packs):
        cur.append(srv.InferenceRequest(request_id=k, worker_id=k % 8, encoded_states=p))
        if sum(len(r.encoded_states) for r in cur) >= args.batch_states:
            batches.append(cur)
            cur = []
    server = srv.InferenceServer(
        request_q=None, reply_qs=[], control_q=None, model_factory=lambda _: model, checkpoint_path=None
    )
    server._load_model()
    sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
    t = {}

    def add(key, seconds):
        t[key] = t.get(key, 0.0) + seconds

    def forward(prepared):
        with torch.no_grad():
            inputs = model.packed_inputs(prepared["arrays"])
            gather = srv._gather_index(prepared["legal"], inputs["game_states"].device)
            probs, _, values = model.run_packed(inputs, legal_indices=prepared["legal"])
        return probs, values, getattr(model, "last_price_components", None), gather

    def spans_of(batch):
        out, start = [], 0
        for r in batch:
            out.append((start, len(r.encoded_states)))
            start += len(r.encoded_states)
        return out

    def legacy(batch, spans, prepared, profiled):
        """The server before staged replies: with the profile's sync after the forward, or without."""
        t0 = time.perf_counter()
        probs, values, price, _ = forward(prepared)
        if profiled:
            sync()
        t1 = time.perf_counter()
        out = srv._batched_legal_replies(batch, spans, probs, values, price)
        t2 = time.perf_counter()
        tag = "legacy (profiled)" if profiled else "legacy (unprofiled)"
        add(f"{tag}: forward", t1 - t0)
        add(f"{tag}: replies", t2 - t1)
        add(f"{tag}: total", t2 - t0)
        return out

    def steps(batch, prepared):
        """The legacy reply's device steps one by one, a sync after each."""
        probs, values, price, _ = forward(prepared)
        sync()
        legal = srv.LegalIndices.concat([r.encoded_states.legal_indices for r in batch])
        s0 = time.perf_counter()
        rows = torch.from_numpy(legal.row_ids()).to(probs.device)
        cols = torch.from_numpy(legal.flat.astype(np.int64)).to(probs.device)
        sync()
        s1 = time.perf_counter()
        g = probs[rows, cols].float()
        sync()
        s2 = time.perf_counter()
        g.cpu().numpy()
        s3 = time.perf_counter()
        values.detach().to(torch.float32).cpu().numpy()
        s4 = time.perf_counter()
        if price is not None:
            price["price_logits"].detach().float().cpu().numpy()
        s5 = time.perf_counter()
        for key, a, b in (("index upload", s0, s1), ("gather", s1, s2), ("priors down", s2, s3),
                          ("values down", s3, s4), ("logits down", s4, s5)):
            add("legacy step: " + key, b - a)

    def staged(batch, spans, prepared):
        t3 = time.perf_counter()
        with torch.no_grad():
            inputs = model.packed_inputs(prepared["arrays"])
            gather = srv._gather_index(prepared["legal"], inputs["game_states"].device)
            ta = time.perf_counter()
            probs, _, values = model.run_packed(inputs, legal_indices=prepared["legal"])
            price = getattr(model, "last_price_components", None)
            tb = time.perf_counter()
            stage = srv._stage_replies(probs, values, price, gather)
        tc = time.perf_counter()
        stage.arrays()  # waits for the transfer (the one sync)
        t4 = time.perf_counter()
        for key, a, b in (("inputs upload", t3, ta), ("forward launch", ta, tb), ("reply launch", tb, tc),
                          ("wait for the GPU", tc, t4)):
            add("staged step: " + key, b - a)
        out = srv._staged_legal_replies(batch, spans, stage, prepared["legal"], price)
        t5 = time.perf_counter()
        add("staged: forward + transfer", t4 - t3)
        add("staged: replies", t5 - t4)
        add("staged: total", t5 - t3)
        return out

    mismatches, n_states, pickled, k = 0, 0, 0, 0
    for rep in range(args.repeats):
        for batch in batches:
            prepared = server._prepare(batch)
            spans = spans_of(batch)
            if rep == 0:
                n_states += sum(n for _, n in spans)
            # The paths in rotating order (the GPU's other users vary over time).
            runs = [
                ("old", lambda: legacy(batch, spans, prepared, True)),
                ("old_unprofiled", lambda: legacy(batch, spans, prepared, False)),
                ("steps", lambda: steps(batch, prepared)),
                ("new", lambda: staged(batch, spans, prepared)),
            ]
            k += 1
            runs = runs[k % len(runs):] + runs[:k % len(runs)]
            out = {name: fn() for name, fn in runs}
            old, new = out["old"], out["new"]
            for a, b in zip(old, new):
                same = (a.request_id == b.request_id and np.array_equal(a.probs, b.probs)
                        and np.array_equal(a.value, b.value) and np.array_equal(a.legal_lengths, b.legal_lengths)
                        and a.probs.dtype == b.probs.dtype and a.value.dtype == b.value.dtype
                        and np.array_equal(a.price_components["price_logits"], b.price_components["price_logits"])
                        and a.price_components["slot_index"] == b.price_components["slot_index"])
                mismatches += not same
            # the transport's serialization: pickled by the reply queue's feeder, unpickled by the worker
            t6 = time.perf_counter()
            blobs = [bytes(ForkingPickler.dumps(r)) for r in new]
            t7 = time.perf_counter()
            for blob in blobs:
                pickle.loads(blob)
            t8 = time.perf_counter()
            add("pickle (server feeder thread)", t7 - t6)
            add("unpickle (worker)", t8 - t7)
            pickled += sum(len(b) for b in blobs)
            del out, old, new
    nb = len(batches) * args.repeats
    print(f"{len(batches)} batches x {args.repeats}, {n_states / len(batches):.0f} states/batch, "
          f"{pickled / nb / 1e6:.2f} MB pickled per batch; mismatched replies: {mismatches}")
    for key, v in t.items():
        print(f"  {key:34s} {1000 * v / nb:8.2f} ms/batch")
    if mismatches:
        raise SystemExit("the staged replies differ from the legacy ones")


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
    unknown = {f for f in flags if not f.startswith("rep")} - {
        "base", "fast", "dedupe", "pipeline", "legacyreply", "noprice"
    }
    if unknown:
        raise SystemExit(f"unknown variant flags {unknown}")
    os.environ["RL18XX_SERVER_PROFILE"] = str(args.profile_every)
    handle = start_inference_server(
        num_workers=args.clients, checkpoint_path=args.model,
        model_factory=noprice_model_factory if "noprice" in flags else load_server_model,
        batch_size=args.batch_size, fast_path="base" not in flags, pipeline="pipeline" in flags, replicas=replicas,
        staged_replies="legacyreply" not in flags,
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
    c.add_argument("--games", type=int, default=None, help="Games per worker (default 32, or the run config's)")
    c.add_argument("--run-config", default=None, help="A search-learning run's config.json: generate with its settings")
    c.add_argument("--midgame-fraction", type=float, default=0.3)
    c.add_argument("--auction-fraction", type=float, default=0.15)
    r = sub.add_parser("replies")
    r.add_argument("--requests", required=True)
    r.add_argument("--model", default=GEN4)
    r.add_argument("--batch-states", type=int, default=365, help="States per server batch (az8: ~365)")
    r.add_argument("--repeats", type=int, default=2)
    r.add_argument("--no-dedupe", action="store_true")
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
    {"capture": capture, "replies": replies, "serve": serve}[args.cmd](args)


if __name__ == "__main__":
    main()
