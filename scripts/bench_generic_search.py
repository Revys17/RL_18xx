"""CPU cost of one searched 1867 move: a title-generic learner + critic searched in-process, as the bots search
(``generic_search``: one tree, ``LockstepSearcher`` with in-process clients, the bots' settings), at mid-game decisions
of a Ruby fixture game (the first decision of each kind after ``--from`` engine actions). ``docs/rl_model.md``
"Searching 1867".

    CUDA_VISIBLE_DEVICES= uv run python scripts/bench_generic_search.py \\
        --learner $R/learner/2000.pth --critic $R/critic/2000.pth --threads 4 --readouts 200 400
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

STEPS = ("BuySellParShares", "Track", "Token", "Route", "BuyTrain", "Merge")


def main() -> None:
    import torch

    from rl18xx.agent.alphazero import model_generic as mg
    from rl18xx.agent.alphazero.generic_data import new_engine_game
    from rl18xx.agent.alphazero.generic_import_1867 import replay_export
    from rl18xx.agent.alphazero.generic_search import BOT_SEARCH, GenericSearchPlayer, local_evaluator, search_config
    from rl18xx.agent.alphazero.lockstep import LockstepSearcher
    from tests.title_replay_harness import game_seed

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--learner", required=True)
    ap.add_argument("--critic", required=True)
    ap.add_argument("--value-head", default="score")
    ap.add_argument("--fixture", default=str(REPO_ROOT / "tests/fixtures/1867/hs_ahjzadkh_19792.json"))
    ap.add_argument("--from", dest="start", type=int, default=90, help="Decisions after this many engine actions")
    ap.add_argument("--threads", type=int, default=4, help="torch threads (the bots' --cpu-threads)")
    ap.add_argument("--readouts", type=int, nargs="+", default=[200, 400])
    ap.add_argument("--prior-temperature", type=float, default=0.5)
    ap.add_argument("--repeats", type=int, default=2)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    game = json.loads(Path(args.fixture).read_text())
    log, _, _, _ = replay_export(game)
    n = len(game["players"])

    def position(count: int):
        raw = new_engine_game("1867", {i + 1: f"Player {i + 1}" for i in range(n)}, (), game_seed(game))
        for action in log[:count]:
            raw.process_action(dict(action))
        return raw

    raw = position(0)
    picked: dict = {}
    for k in range(len(log)):
        moves = json.loads(raw.rl_legal_moves_json())
        if k >= args.start and len(moves) > 1:
            key = (raw.round.round_type, moves[0]["step"])
            if key not in picked and key[1] in STEPS:
                picked[key] = (k, len(moves))
        raw.process_action(dict(log[k]))

    policy, _ = mg.load_checkpoint(args.learner, "cpu")
    critic, _ = mg.load_checkpoint(args.critic, "cpu")
    print(f"{sum(p.numel() for p in policy.parameters()) / 1e6:.1f}M parameters per network", flush=True)
    evaluator = local_evaluator(policy, critic, args.value_head)
    for (round_type, step), (k, num_legal) in sorted(picked.items(), key=lambda x: x[1][0]):
        raw = position(k)
        for readouts in args.readouts:
            times = []
            for _ in range(args.repeats):
                config = search_config(
                    **BOT_SEARCH, num_readouts=readouts, min_readouts=readouts,
                    prior_temperature=args.prior_temperature,
                )  # fmt: skip
                player = GenericSearchPlayer(config)
                player.reset(raw)
                searcher = LockstepSearcher({"main": evaluator})
                start = time.perf_counter()
                searcher.run([player.start_search(key="main")])
                move = player.pick_move()
                times.append(time.perf_counter() - start)
            st = searcher.stats
            print(
                f"{round_type:9} {step:17} action {k:3} legal {num_legal:3} @{readouts}: best {min(times):.2f}s "
                f"(runs {', '.join(f'{t:.2f}' for t in times)}), {st.rounds} rounds, {st.leaves} leaves, move {move}; "
                f"last run: tree + inputs {st.collect_seconds + st.apply_seconds:.2f}s, forwards {st.wait_seconds:.2f}s",
                flush=True,
            )


if __name__ == "__main__":
    main()
