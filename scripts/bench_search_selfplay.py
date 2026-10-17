"""Generation rate of search learning's self-play (search_learning.Generator)
in one worker process: games/h, searched rows/h, decisions/s and network
evaluations/s, and the worker's time outside the network per evaluation.

The network runs in-process (``lockstep.ModelEvaluator``): ``--model stub``
(nearly free: uniform priors, equal values -- what is left is the worker's own
cost, the ceiling one worker can feed a GPU) or a checkpoint
(``<policy>+<critic>``; on the CPU its forward dominates). Starts as gen4's
run: human starts, 20% random, ``--midgame-fraction`` 0.3, ``--auction-fraction``
0.15. Games still running at the end count toward rows and decisions, not games;
``games_per_hour_est`` is decisions/s over the finished games' mean length.
Never touches the GPU.

    uv run python scripts/bench_search_selfplay.py --model stub --n-full 64 200 --seconds 120
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="stub")
    parser.add_argument("--n-full", type=int, nargs="+", default=[64, 200])
    parser.add_argument("--p-full", type=float, default=0.25)
    parser.add_argument("--target", default="visits", choices=("visits", "gumbel"))
    parser.add_argument("--games-per-worker", type=int, default=24)
    parser.add_argument("--seconds", type=float, default=120.0)
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--learner-temperature", type=float, default=0.5)
    parser.add_argument("--midgame-fraction", type=float, default=0.3)
    parser.add_argument("--auction-fraction", type=float, default=0.15)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    from bench_search_harness import TimedNet, load_network

    from rl18xx.agent.alphazero import search_learning as sl
    from rl18xx.agent.alphazero.lockstep import ModelEvaluator

    net = TimedNet(load_network(args.model, args.torch_threads))
    for n_full in args.n_full:
        cfg = sl.SearchLearningConfig(
            policy_checkpoint="", value_checkpoint="", learner_temperature=args.learner_temperature,
            prior_temperature=args.learner_temperature, target=args.target, p_full=args.p_full, n_full=n_full,
            games_per_worker=args.games_per_worker,
            start_positions=str(REPO_ROOT / "human_games" / "start_positions_1830_4p.jsonl"),
            midgame_positions=str(REPO_ROOT / "human_games" / "midgame_positions_1830_4p.jsonl"),
            midgame_fraction=args.midgame_fraction, auction_fraction=args.auction_fraction,
        )
        gen = sl.Generator(sl.worker_settings(cfg, 0), {"learner": ModelEvaluator(net)})
        net.seconds = net.states = net.calls = 0
        start = time.perf_counter()
        out = gen.run(args.seconds)
        wall = time.perf_counter() - start
        stats = out["stats"]
        rows = sum(len(g["rows"]) for g in out["games"]) + sum(len(g.rows) for g in gen.games.values())
        decisions = stats.get("searched_decisions", 0) + stats.get("fast_decisions", 0)
        lengths = [g["decisions"] for g in out["games"]]
        mean_length = sum(lengths) / len(lengths) if lengths else float("nan")
        evals = stats["leaves"] + stats["policy_states"]
        result = {
            "model": args.model, "n_full": n_full, "p_full": args.p_full, "target": args.target,
            "games_per_worker": args.games_per_worker, "seconds": round(wall, 1),
            "games_finished": len(lengths), "mean_game_decisions": round(mean_length, 1),
            "games_per_hour_est": round(decisions / wall * 3600 / mean_length, 2) if lengths else None,
            "searched_rows_per_hour": round(rows / wall * 3600),
            "decisions_per_second": round(decisions / wall, 2),
            "evaluations_per_second": round(evals / wall, 1),
            "evaluations_per_decision": round(evals / max(decisions, 1), 2),
            "forward_seconds": round(net.seconds, 2),
            "outside_forward_us_per_eval": round(1e6 * (wall - net.seconds) / max(evals, 1), 1),
            "searched_share": round(stats.get("searched_decisions", 0) / max(decisions, 1), 3),
            "skipped_starts": stats.get("skipped_starts", 0),
        }
        print(json.dumps(result), flush=True)
        if args.out:
            with open(args.out, "a") as f:
                f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
