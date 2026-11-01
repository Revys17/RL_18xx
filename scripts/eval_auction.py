"""Auction-only head-to-head: a candidate auction system against a baseline, scored by a frozen
value network's estimate of the first Stock Round position (search_learning ``auction_only``).

Every game is a waterfall auction under the real rules. It opens with 0..``--prefix`` decisions,
every seat's, sampled from the supervised policy (``--prefix-policy``), so that two deterministic
systems play distinct games. Then the candidate holds ``--seats`` random seats and the baseline
the rest, each searching every decision at ``--readouts`` (value-driven PUCT, c_puct 0.05, mean Q,
the most visited move) or, with ``--baseline-mode policy``, sampling its policy. When the auction
ends, ``--terminal-value`` (lr1's v3 by default) estimates each seat's win chance: the score is the
candidate seats' summed estimate against their fair share (seats / 4), with the private prices each
side paid and the human reference (1,945 4-player games: median D&H $80, M&H $135, C&A $210).

    uv run python scripts/eval_auction.py --candidate <policy>+<value> --baseline <policy>+<value> \\
        --baseline-prior-temperature 0.5 --seats 1 --games 1200 --out logs/eval_auction/<name>.json
"""

import argparse
import json
import logging
import math
import multiprocessing
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

LR1 = Path("/home/revys/longrun/work/lr1")
DEFAULT_TERMINAL = LR1 / "model_checkpoints_value/lr1/promoted/3.pth"
DEFAULT_PREFIX = LR1 / "model_checkpoints/AlphaZeroTransformer/20261004_134558_804974562/10.pth"
HUMAN_MEDIAN = {"CS": 45, "DH": 80, "MH": 135, "CA": 210, "BO": 220}


def settings_for(args) -> dict:
    from rl18xx.agent.alphazero import search_learning as sl

    cfg = sl.SearchLearningConfig(
        policy_checkpoint="", value_checkpoint="", target="visits", p_full=1.0, n_full=args.readouts, n_fast=0,
        c_puct_init=0.05, mean_q=True, dedupe_leaves=True, parallel_readouts=8,
        prior_temperature=args.candidate_prior_temperature, learner_temperature=1.0, full_move_temperature=0.0,
        games_per_worker=args.games_per_worker, max_decisions=200, reuse_tree="off", dedupe_maps=True,
        auction_only=True, terminal_value=str(args.terminal_value), auction_prefix_decisions=args.prefix,
        auction_prefix_server="prefix",
    )  # fmt: skip
    member = {
        "server": "baseline",
        "kind": "fixed",
        "mode": args.baseline_mode,
        "readouts": args.readouts,
        "prior_temperature": args.baseline_prior_temperature,
        "temperature": args.baseline_temperature,
        "value_head": "win",
    }
    league = {
        "share": 1.0,
        "seat_weights": {4 - args.seats: 1.0},
        "members": {"baseline": member},
        "weights": {"baseline": 1.0},
        "server_versions": {},
    }
    return sl.worker_settings(cfg, version=0, league=league)


def summarize(records: list, seats: int) -> dict:
    from rl18xx.agent.alphazero import search_learning as sl

    scored = [r for r in records if r.get("win_share") is not None]
    excess, bought = [], defaultdict(lambda: {"candidate": [], "baseline": []})
    for r in scored:
        main = [s for s, m in enumerate(r["seat_members"]) if m == sl.MAIN]
        excess.append(sum(r["win_share"][s] for s in main) - len(main) / 4)
        for company, (buyer, price) in (r.get("auction") or {}).items():
            side = "candidate" if (int(buyer) - 1) in main else "baseline"
            bought[company][side].append(price)
    x = np.asarray(excess, dtype=np.float64)
    prices = {}
    for company in sl.AUCTION_FACE:
        sides = bought.get(company, {"candidate": [], "baseline": []})
        n = len(sides["candidate"]) + len(sides["baseline"])
        prices[company] = {
            "candidate_mean": float(np.mean(sides["candidate"])) if sides["candidate"] else None,
            "baseline_mean": float(np.mean(sides["baseline"])) if sides["baseline"] else None,
            "candidate_share": len(sides["candidate"]) / n if n else None,
            "human_median": HUMAN_MEDIAN.get(company),
        }
    return {
        "games": len(scored),
        "stalled": sum(1 for r in records if r["termination"] == "auction_stall"),
        "candidate_seats": seats,
        "fair_share": seats / 4,
        "win_excess": float(x.mean()) if len(x) else math.nan,
        "win_excess_se": float(x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else math.nan,
        "prices": prices,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidate", required=True, help="<policy>+<value> checkpoint spec")
    parser.add_argument("--candidate-prior-temperature", type=float, default=1.0)
    parser.add_argument("--baseline", required=True, help="<policy>+<value> checkpoint spec")
    parser.add_argument("--baseline-mode", choices=("search", "policy"), default="search")
    parser.add_argument("--baseline-prior-temperature", type=float, default=1.0)
    parser.add_argument("--baseline-temperature", type=float, default=1.0, help="--baseline-mode policy")
    parser.add_argument("--seats", type=int, choices=(1, 2, 3), default=1, help="The candidate's seats per game")
    parser.add_argument("--games", type=int, default=1200)
    parser.add_argument("--readouts", type=int, default=200)
    parser.add_argument("--prefix", type=int, default=6, help="Opening decisions sampled: uniform 0..N")
    parser.add_argument("--prefix-policy", default=str(DEFAULT_PREFIX))
    parser.add_argument("--terminal-value", default=str(DEFAULT_TERMINAL))
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--games-per-worker", type=int, default=16)
    parser.add_argument("--task-seconds", type=float, default=30.0)
    parser.add_argument("--out", default=None, help="Write the summary (JSON) here")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    import torch
    from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

    from rl18xx.agent.alphazero import search_learning as sl
    from rl18xx.agent.alphazero.inference_server import start_inference_server
    from rl18xx.agent.alphazero.policy_gradient import _load, load_server_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    terminal = _load(Path(args.terminal_value))
    terminal.eval()
    settings = settings_for(args)
    specs = {"learner": args.candidate, "baseline": args.baseline, "prefix": args.prefix_policy}
    handles = {
        name: start_inference_server(
            num_workers=args.workers,
            model_factory=load_server_model,
            checkpoint_path=spec,
            batch_size=512,
            autocast_device=None,
        )  # fmt: skip
        for name, spec in specs.items()
    }
    queues = {name: (h.request_q, h.reply_qs, h.ticket_q) for name, h in handles.items()}
    context = multiprocessing.get_context("spawn")
    records, started = [], time.time()
    try:
        with ProcessPoolExecutor(
            max_workers=args.workers, mp_context=context, initializer=sl.worker_init, initargs=(queues, None, {})
        ) as executor:
            futures = {executor.submit(sl.generate, settings, args.task_seconds) for _ in range(args.workers)}
            while len([r for r in records if r["termination"] == "auction_end"]) < args.games:
                done, futures = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    games = future.result()["games"]
                    sl.auction_outcomes(games, terminal, device)
                    records.extend(games)
                    futures.add(executor.submit(sl.generate, settings, args.task_seconds))
                summary = summarize(records, args.seats)
                logging.info(
                    f"{summary['games']} auctions in {time.time() - started:.0f}s: candidate win excess "
                    f"{summary['win_excess']:+.4f} ± {summary['win_excess_se']:.4f}"
                )
            for future in futures:
                future.cancel()
    finally:
        for handle in handles.values():
            handle.shutdown()
    summary = summarize(records, args.seats)
    summary.update(
        candidate=args.candidate, baseline=args.baseline, baseline_mode=args.baseline_mode,
        readouts=args.readouts, prefix=args.prefix, terminal_value=args.terminal_value,
        seconds=time.time() - started,
    )  # fmt: skip
    print(json.dumps(summary, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
