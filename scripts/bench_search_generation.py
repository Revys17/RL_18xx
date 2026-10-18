"""Search learning's generation alone (no training): inference server(s) on the GPU
and ``--workers`` worker processes running ``search_learning.generate`` tasks, as
``main.py search-learning`` runs them -- to compare generation settings back to
back on one machine. Reports decisions/s, network evaluations/s, evaluations per
decision, searched rows/h, games/h (finished games), the tree-reuse shares and
the workers' peak RSS.

A config is ``+``-joined options (``base`` or ``unpacked``: the server path before 2026-10-18;
``fast``: the packed fast path, the default now):
``unpacked`` (the server's per-state path), ``dedupe`` (``--dedupe-maps``),
``total`` / ``additional`` (``--reuse-tree``), ``rep<N>`` (``--server-replicas``).
Search settings as az1_20261017 (visits target, pruned, n_full 200, n_fast 48,
p_full 0.25, c_puct 0.05, 32 games per worker, mid-game 0.3, auction 0.15).

    uv run python scripts/bench_search_generation.py --configs base dedupe+total+rep2 --workers 8 --seconds 150
"""

import argparse
import json
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

GEN4 = (
    "model_checkpoints_pg/rab_place_20261014/learner/1800.pth"
    "+model_checkpoints_pg/rab_place_20261014/critic/1800.pth"
)


def _peak_rss_mb() -> float:
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def _generate(settings, seconds):
    from rl18xx.agent.alphazero import search_learning as sl

    out = sl.generate(settings, seconds)
    out["games"] = [{k: v for k, v in g.items() if k != "rows"} | {"rows": len(g["rows"])} for g in out["games"]]
    out["rss_mb"] = _peak_rss_mb()
    return out


def run_config(args, config: str) -> dict:
    from functools import partial

    from rl18xx.agent.alphazero import search_learning as sl
    from rl18xx.agent.alphazero.inference_server import start_inference_server
    from rl18xx.agent.alphazero.policy_gradient import load_server_model

    flags = set(config.split("+"))
    replicas = next((int(f[3:]) for f in flags if f.startswith("rep")), 1)
    reuse = "total" if "total" in flags else "additional" if "additional" in flags else "off"
    cfg = sl.SearchLearningConfig(
        policy_checkpoint="", value_checkpoint="", learner_temperature=0.5, prior_temperature=0.5,
        target="visits", visit_target="pruned", p_full=0.25, n_full=200, n_fast=48, c_puct_init=0.05,
        workers=args.workers, games_per_worker=args.games_per_worker, task_seconds=args.task_seconds,
        start_positions="human_games/start_positions_1830_4p_v2.jsonl",
        midgame_positions="human_games/midgame_positions_1830_4p_v2.jsonl",
        midgame_fraction=0.3, auction_fraction=0.15, reuse_tree=reuse, dedupe_maps="dedupe" in flags,
        server_replicas=replicas,
    )
    handle = start_inference_server(
        num_workers=args.workers, model_factory=load_server_model, checkpoint_path=args.model,
        batch_size=cfg.inference_batch_size, autocast_device=None, fast_path=not flags & {"unpacked", "base"},
        replicas=replicas,
    )
    queues = {"learner": (handle.request_q, handle.reply_qs, handle.ticket_q)}
    executor = ProcessPoolExecutor(
        max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"),
        initializer=sl.worker_init, initargs=(queues,),
    )
    settings = sl.worker_settings(cfg, 0)
    totals, games, rss = {}, [], 0.0
    try:
        # Warm-up tasks (games start, CUDA warms up), then the measured ones.
        list(executor.map(_generate, [settings] * args.workers, [args.warmup] * args.workers))
        started = time.time()
        futures = [executor.submit(_generate, settings, args.task_seconds) for _ in range(args.workers)]
        while futures:
            done = [f for f in futures if f.done()]
            for f in done:
                futures.remove(f)
                out = f.result()
                for k, v in out["stats"].items():
                    if isinstance(v, (int, float)) and k not in ("games_in_progress",):
                        totals[k] = totals.get(k, 0) + v
                games.extend(out["games"])
                rss = max(rss, out["rss_mb"])
                if time.time() - started < args.seconds:
                    futures.append(executor.submit(_generate, settings, args.task_seconds))
            time.sleep(0.2)
        wall = time.time() - started
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
        handle.shutdown()
    decisions = totals.get("searched_decisions", 0) + totals.get("fast_search_decisions", 0) + totals.get("fast_decisions", 0)
    evals = totals.get("leaves", 0) + totals.get("policy_states", 0)
    mean_len = sum(g["decisions"] for g in games) / max(len(games), 1)
    return {
        "config": config, "wall_s": round(wall), "decisions_per_s": round(decisions / wall, 1),
        "evaluations_per_s": round(evals / wall), "evaluations_per_decision": round(evals / max(decisions, 1), 2),
        "searched_rows_per_h": round(totals.get("searched_decisions", 0) / wall * 3600),
        "games_per_h_est": round(decisions / wall * 3600 / mean_len, 1) if games else None,
        "games_finished": len(games),
        "reused_search_share": round(totals.get("reused_searches", 0) / max(totals.get("searches", 0), 1), 3),
        "inherited_visit_share": round(totals.get("inherited_visits", 0) / max(totals.get("final_visits", 0), 1), 3),
        "map_rows_per_state": round(totals.get("maps", 0) / max(evals, 1), 3),
        "worker_peak_rss_mb": round(rss),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--configs", nargs="+", default=["base", "dedupe+total+rep2"])
    parser.add_argument("--model", default=GEN4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--games-per-worker", type=int, default=32)
    parser.add_argument("--seconds", type=float, default=150)
    parser.add_argument("--task-seconds", type=float, default=30)
    parser.add_argument("--warmup", type=float, default=30)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    os.chdir(REPO_ROOT)
    results = []
    for r in range(args.rounds):
        for config in args.configs if r % 2 == 0 else args.configs[::-1]:
            result = run_config(args, config)
            result["round"] = r
            print(json.dumps(result), flush=True)
            results.append(result)
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
