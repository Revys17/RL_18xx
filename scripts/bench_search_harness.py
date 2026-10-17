"""Search-harness throughput: network evaluations per second, and the share of
time spent outside the network, for the one-tree-at-a-time search
(``RustMCTSPlayer.suggest_move``: Python leaf encoding, a request per tree
batch, dense priors per leaf) against lockstep search (``lockstep.py``: many
trees per request, native leaf encoding, sparse priors scattered in Rust).

Positions are human 1830 positions (``human_games/midgame_positions_1830_4p.jsonl``
cut at random points from the first stock round to stock round 6, so stock and
operating rounds; ``--positions`` of them) and every one gets one search of
``--readouts`` from a fresh tree, with the value-driven eval settings by default
(mean Q, c_puct 0.15 in every round, dedupe, 8 leaves per batch, prior
temperature 0.5; ``--default-search`` for SelfPlayConfig's).

``--concurrency``: 1 is the one-tree path (``suggest_move``), N > 1 lockstep
with N trees per worker. ``--mode inproc`` (one process): the network runs
in-process; for the one-tree path it sits behind a client double that answers
like the inference server does (legal priors only, densified by the client),
so the per-leaf work matches an eval worker's; reports evaluations/s and the
time outside the network call per evaluation. ``--mode server``: an inference
server process and ``--workers`` worker processes that each build their share
of the positions, then all search (cycling through them) for ``--duration``
seconds -- the eval's layout, IPC included; reports the evaluations the server
answered in that window and the share of it the server was busy. Set
``RL18XX_SERVER_PROFILE=<seconds>`` for the server's stage split.

``--model stub``: a stand-in network whose forward is nearly free (uniform
priors, equal values, zero price logits): what is left is the harness. Or a
checkpoint (``<policy.pth>[+<value.pth>]``, composite_model.load_policy_value).

The GPU stays hidden (CUDA_VISIBLE_DEVICES="") unless ``--gpu`` is given:

    uv run python scripts/bench_search_harness.py --model stub --mode inproc --concurrency 1 32
    uv run python scripts/bench_search_harness.py --model stub --mode server --workers 8 --concurrency 1 16
    uv run python scripts/bench_search_harness.py --model <learner.pth>+<critic.pth> --torch-threads 8
    uv run python scripts/bench_search_harness.py --gpu --mode server --model <learner.pth>+<critic.pth> \
        --workers 48 --positions 2000 --readouts 200 --concurrency 1 8 16 --duration 60
"""

import os
import sys

if "--gpu" not in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import random  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

MIDGAME = REPO_ROOT / "human_games" / "midgame_positions_1830_4p.jsonl"
VALUE_DRIVEN = dict(
    mcts_mean_q=True, fpu_reduction=0.0, c_puct_init=0.15, c_puct_by_round={}, dedupe_leaves=True,
    parallel_readouts=8, prior_temperature=0.5,
)


# ----------------------------------------------------------------------------- networks


class StubNet:
    """A network whose forward costs almost nothing: uniform priors, equal values,
    zero price logits (the price head's shape). Accepts ``legal_indices``."""

    def __init__(self):
        from rl18xx.agent.alphazero import price_pmf

        self._price = price_pmf
        self.last_price_components = None

    def eval(self):
        return self

    def encoder_type(self):
        return "Transformer"

    def run_many_encoded(self, encoded_states, legal_indices=None):
        import torch

        from rl18xx.agent.alphazero.mcts import POLICY_SIZE, VALUE_SIZE

        n = len(encoded_states)
        probs = torch.full((n, POLICY_SIZE), 1.0 / POLICY_SIZE)
        values = torch.zeros((n, VALUE_SIZE))
        for i, s in enumerate(encoded_states):
            values[i, : int(s[7])] = 1.0 / int(s[7])
        self.last_price_components = {
            "price_logits": torch.zeros((n, self._price.NUM_SLOTS, self._price.NUM_CELLS)),
            "slot_index": dict(self._price.SLOT_INDEX),
            "num_slots": self._price.NUM_SLOTS,
        }
        return probs, None, values

    def run_encoded(self, encoded):
        probs, _, values = self.run_many_encoded([encoded])
        return probs[0], None, values[0]


def load_network(spec: str, torch_threads: int):
    import torch

    torch.set_num_threads(torch_threads)
    if spec == "stub":
        return StubNet()
    from rl18xx.agent.alphazero.composite_model import load_policy_value

    return load_policy_value(spec)


def model_factory(spec):
    """Inference-server loader (module-level so it pickles)."""
    return load_network(spec, int(os.environ.get("BENCH_SERVER_THREADS", "8")))


class TimedNet:
    """Wraps a network to time its calls (the forward pass, CPU)."""

    def __init__(self, network):
        self.network = network
        self.seconds = 0.0
        self.states = 0
        self.calls = 0

    def __getattr__(self, name):
        return getattr(self.network, name)

    def run_many_encoded(self, encoded_states, legal_indices=None):
        import torch

        start = time.perf_counter()
        with torch.no_grad():
            if legal_indices is None:
                out = self.network.run_many_encoded(encoded_states)
            else:
                out = self.network.run_many_encoded(encoded_states, legal_indices=legal_indices)
        self.seconds += time.perf_counter() - start
        self.states += len(encoded_states)
        self.calls += 1
        return out

    @property
    def last_price_components(self):
        return self.network.last_price_components


class ServerDouble:
    """In-process stand-in for an ``InferenceClient``: the server's sparse legal
    priors, then the client's densifying (``InferenceClient.receive``), so the
    one-tree path does a real eval worker's per-leaf work."""

    def __init__(self, network: TimedNet):
        self.network = network
        self.last_price_components = None

    def run_many_encoded(self, encoded_states, legal_indices=None):
        import torch

        from rl18xx.agent.alphazero.inference_server import PackedStates
        from rl18xx.agent.alphazero.mcts import POLICY_SIZE

        packed = PackedStates.pack(encoded_states)  # the client's stacking + the server's unpack
        states = packed.unpack()
        probs, _, values = self.network.run_many_encoded(states, legal_indices=legal_indices)
        lengths = torch.tensor([len(ix) for ix in legal_indices])
        rows = torch.repeat_interleave(torch.arange(len(legal_indices)), lengths)
        cols = torch.from_numpy(np.concatenate([np.asarray(ix, dtype=np.int64) for ix in legal_indices]))
        priors = probs[rows.to(probs.device), cols.to(probs.device)].float().cpu().numpy()
        dense = np.zeros((len(legal_indices), POLICY_SIZE), dtype=np.float32)
        offset = 0
        for row, ix in enumerate(legal_indices):
            dense[row, ix] = priors[offset:offset + len(ix)]
            offset += len(ix)
        pc = self.network.last_price_components
        self.last_price_components = None if pc is None else {
            "price_logits": torch.as_tensor(pc["price_logits"]).detach().float().cpu(),
            "slot_index": pc.get("slot_index"),
            "num_slots": pc.get("num_slots"),
        }
        return list(torch.from_numpy(dense)), None, list(values)

    def run_encoded(self, encoded):
        # A root expansion: the client sends no legal indices, the server dense priors.
        probs, _, values = self.network.run_many_encoded([encoded])
        self.last_price_components = self.network.last_price_components
        return probs[0], None, values[0]


# ----------------------------------------------------------------------------- positions


def position_prefixes(count: int, seed: int) -> list:
    """Action prefixes of human positions: random cuts of the mid-game file's
    games between their first stock round and their last cut (that replay)."""
    from rl18xx.agent.alphazero.start_positions import load_midgame_positions

    games = load_midgame_positions(MIDGAME)[4]
    rng = random.Random(seed)
    prefixes = []
    while len(prefixes) < count:
        _, actions, cuts = rng.choice(games)
        prefix = actions[: rng.randint(min(cuts.values()) - 30, max(cuts.values()))]
        if build_position(prefix) is not None:
            prefixes.append(prefix)
    return prefixes


def build_position(prefix):
    """The position after ``prefix``, moved on past forced moves to a decision."""
    from rl18xx.agent.alphazero.start_positions import apply_actions, new_game

    try:
        game = apply_actions(new_game(4), prefix)
    except RuntimeError:
        return None  # the engine rejects one of its actions (the file predates a rules change)
    rust = game._game
    while not rust.finished and len(rust.factored_legal_indices()) == 1:
        index = int(rust.factored_legal_indices()[0])
        price_range = rust.price_range_for_index(index)
        rust.apply_action_index(index, None if price_range is None else int(price_range[0]))
    return game


# ----------------------------------------------------------------------------- runs


def make_config(backend, readouts: int, default_search: bool, client: bool):
    from rl18xx.agent.alphazero.config import SelfPlayConfig

    search = {} if default_search else VALUE_DRIVEN
    kwargs = dict(
        network=None if client else backend, use_inference_server=client, inference_client=backend if client else None
    )
    return SelfPlayConfig(
        **kwargs, **search, num_readouts=readouts, min_readouts=readouts, dirichlet_noise_weight=0.0,
        enable_resign=False, noresign_holdout_rate=0.0, player_count_distribution={4: 1.0},
    )


def expand_root(player):
    from rl18xx.agent.alphazero.self_play import _slice_price_components

    leaf = player.root.select_leaf()
    leaf.ensure_encoded()
    probs, _, value = player._backend.run_encoded(leaf.encoded_game_state)
    price = _slice_price_components(getattr(player._backend, "last_price_components", None), 0)
    leaf.incorporate_results(probs, value, leaf, price_components=price)


def run_single_tree(positions, backend, readouts, default_search, client=True, deadline=None) -> dict:
    """The one-tree path: suggest_move per position (cycling through them until
    ``deadline``, when given)."""
    from rl18xx.agent.alphazero.rust_mcts_player import RustMCTSPlayer

    config = make_config(backend, readouts, default_search, client)
    totals = dict(select_encode=0.0, inference=0.0, backup=0.0, searches=0)
    start = time.perf_counter()
    k = 0
    while (k < len(positions)) if deadline is None else (time.time() < deadline):
        game = positions[k % len(positions)]
        k += 1
        player = RustMCTSPlayer(config)
        player.initialize_game(game.pickle_clone())
        expand_root(player)
        player.suggest_move()
        totals["select_encode"] += player.cumulative_leaf_selection_time
        totals["inference"] += player.cumulative_inference_time
        totals["backup"] += player.cumulative_backup_time
        totals["searches"] += 1
    totals["wall"] = time.perf_counter() - start
    return totals


def run_lockstep(positions, evaluator, readouts, default_search, concurrency, deadline=None) -> dict:
    """Lockstep: ``concurrency`` trees searched at once, refilled as searches
    finish (cycling through the positions until ``deadline``, when given)."""
    from rl18xx.agent.alphazero.lockstep import LockstepSearcher
    from rl18xx.agent.alphazero.rust_mcts_player import RustMCTSPlayer

    config = make_config(None, readouts, default_search, client=False)
    searcher = LockstepSearcher({None: evaluator})
    k = 0
    active = []
    start = time.perf_counter()
    setup = 0.0
    while True:
        t = time.perf_counter()
        while len(active) < concurrency and ((k < len(positions)) if deadline is None else (time.time() < deadline)):
            player = RustMCTSPlayer(config)
            player.initialize_game(positions[k % len(positions)].pickle_clone())
            k += 1
            search = player.start_search()
            if search is not None:
                active.append(search)
        setup += time.perf_counter() - t
        if not active:
            break
        searcher.round(active)
        active = [s for s in active if not s.done]
    stats = searcher.stats
    return dict(
        wall=time.perf_counter() - start, setup=setup, collect_send=stats.collect_seconds,
        wait=stats.wait_seconds, apply=stats.apply_seconds, rounds=stats.rounds, requests=stats.requests,
        leaves=stats.leaves, searches=k,
    )


def report(label: str, evals: int, wall: float, forward: float, extra: dict) -> dict:
    out = dict(
        label=label, evals=evals, seconds=round(wall, 3), evals_per_s=round(evals / wall, 1),
        forward_seconds=round(forward, 3), outside_forward_share=round(1 - forward / wall, 3),
        outside_forward_us_per_eval=round(1e6 * (wall - forward) / max(evals, 1), 1),
        **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in extra.items()},
    )
    print(json.dumps(out), flush=True)
    return out


def bench_inproc(args) -> list:
    prefixes = position_prefixes(args.positions, args.seed)
    positions = [build_position(p) for p in prefixes]
    positions = [g for g in positions if not g._game.finished]
    results = []
    for repeat in range(args.repeats):
        net = TimedNet(load_network(args.model, args.torch_threads)) if repeat == 0 or args.model == "stub" else net
        net.seconds = net.states = net.calls = 0
        if 1 in args.concurrency:
            t = run_single_tree(positions, ServerDouble(net), args.readouts, args.default_search)
            results.append(report("single_tree", net.states, t.pop("wall"), net.seconds, dict(calls=net.calls, **t)))
        for concurrency in [c for c in args.concurrency if c > 1]:
            from rl18xx.agent.alphazero.lockstep import ModelEvaluator

            net.seconds = net.states = net.calls = 0
            evaluator = ModelEvaluator(net)
            t = run_lockstep(positions, evaluator, args.readouts, args.default_search, concurrency)
            results.append(
                report(f"lockstep_x{concurrency}", net.states, t.pop("wall"), net.seconds, dict(calls=net.calls, **t))
            )
    return results


# ----------------------------------------------------------------------------- server mode

_CLIENT = None


def _worker_init(request_q, reply_qs, ticket_q):
    global _CLIENT
    from rl18xx.agent.alphazero.inference_server import InferenceClient

    os.environ["OMP_NUM_THREADS"] = "1"
    import torch

    torch.set_num_threads(1)
    worker_id = ticket_q.get()
    _CLIENT = InferenceClient(request_q=request_q, reply_q=reply_qs[worker_id], worker_id=worker_id)
    logging.getLogger().setLevel(logging.WARNING)


def _worker_task(prefixes, readouts, default_search, concurrency, start_at, deadline) -> dict:
    """Build this worker's positions, wait for the common start, search them
    (cycling) until the deadline."""
    import rl18xx.agent.alphazero.rust_mcts_player  # noqa: F401  (imports outside the measured window)
    import rl18xx.agent.alphazero.self_play  # noqa: F401

    positions = [g for g in (build_position(p) for p in prefixes) if not g._game.finished]
    late = time.time() - start_at
    time.sleep(max(0.0, start_at - time.time()))
    if concurrency <= 1:
        t = run_single_tree(positions, _CLIENT, readouts, default_search, deadline=deadline)
    else:
        from rl18xx.agent.alphazero.lockstep import ClientEvaluator

        t = run_lockstep(positions, ClientEvaluator(_CLIENT), readouts, default_search, concurrency, deadline=deadline)
    t["late"] = max(0.0, late)
    return t


def bench_server(args) -> list:
    """Every worker builds its positions, then all search from one start time
    for ``--duration`` seconds (cycling through their positions); throughput =
    evaluations the server answered in that window / its length. Server busy
    time = its batch handling (unpack, forward, replies), the rest it waited."""
    import multiprocessing as mp

    from rl18xx.agent.alphazero.inference_server import start_inference_server

    os.environ["BENCH_SERVER_THREADS"] = str(args.torch_threads)
    prefixes = position_prefixes(args.positions, args.seed)
    results = []
    for concurrency in args.concurrency:
        server = start_inference_server(
            num_workers=args.workers, model_factory=model_factory, checkpoint_path=args.model,
            batch_size=args.batch_size, batch_timeout_ms=2.0, autocast_device=None,
        )
        try:
            server.health(timeout_s=600)  # waits for the model to load
            shares = [prefixes[i::args.workers] for i in range(args.workers)]
            ctx = mp.get_context("fork")
            with ProcessPoolExecutor(
                max_workers=args.workers, mp_context=ctx, initializer=_worker_init,
                initargs=(server.request_q, server.reply_qs, server.ticket_q),
            ) as pool:
                start_at = time.time() + args.start_delay
                deadline = start_at + args.duration
                futures = [
                    pool.submit(
                        _worker_task, share, args.readouts, args.default_search, concurrency, start_at, deadline
                    )
                    for share in shares
                ]
                time.sleep(max(0.0, start_at - time.time()))
                before = server.health(timeout_s=60)
                time.sleep(max(0.0, deadline - time.time()))
                after = server.health(timeout_s=60)
                outs = [f.result() for f in futures]
            evals = after.requests_served - before.requests_served
            busy_ms = after.avg_forward_ms * after.batches_served - before.avg_forward_ms * before.batches_served
            busy = busy_ms / 1000.0
            batches = after.batches_served - before.batches_served
            label = "single_tree" if concurrency <= 1 else f"lockstep_x{concurrency}"
            out = dict(
                label=f"server_{label}", workers=args.workers, evals=evals, seconds=args.duration,
                evals_per_s=round(evals / args.duration, 1), server_busy_share=round(busy / args.duration, 3),
                batches=batches, avg_batch=round(evals / max(batches, 1), 1),
                worker_late_max=round(max(o["late"] for o in outs), 2),
                searches=sum(int(o.get("searches", 0)) for o in outs),
            )
            print(json.dumps(out), flush=True)
            results.append(out)
        finally:
            server.shutdown()
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="stub")
    parser.add_argument("--mode", choices=("inproc", "server"), default="inproc")
    parser.add_argument("--positions", type=int, default=48)
    parser.add_argument("--readouts", type=int, default=64)
    parser.add_argument(
        "--concurrency", type=int, nargs="+", default=[1, 16],
        help="1: the one-tree path (suggest_move); N > 1: lockstep with N trees per worker",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--start-delay", type=float, default=45.0, help="server mode: seconds to build positions")
    parser.add_argument("--duration", type=float, default=30.0, help="server mode: seconds measured")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--default-search", action="store_true", help="SelfPlayConfig's search settings")
    parser.add_argument("--out", type=Path, default=None, help="Append the results here as JSON lines")
    parser.add_argument("--gpu", action="store_true", help="Let the network use the GPU (default: CPU only)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    results = bench_inproc(args) if args.mode == "inproc" else bench_server(args)
    if args.out:
        with open(args.out, "a") as f:
            for r in results:
                f.write(json.dumps({**vars(args), "out": None, **r}, default=str) + "\n")


if __name__ == "__main__":
    main()
