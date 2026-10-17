"""Native leaf encoding vs ``mcts._rust_encode`` on human games: replays the
4-player games of ``human_games/midgame_positions_1830_4p.jsonl`` (auction to
stock round 6) and compares the two encodings bit-for-bit at every position
(``--every`` thins them). Exits non-zero on the first mismatch.

    uv run python scripts/check_native_encode.py --games 200
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse  # noqa: E402
import logging  # noqa: E402
import sys  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", default=str(REPO_ROOT / "human_games" / "midgame_positions_1830_4p.jsonl"))
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--every", type=int, default=1)
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    from rl18xx.agent.alphazero.mcts import _native_encode, _rust_encode
    from rl18xx.agent.alphazero.start_positions import load_midgame_positions, new_game

    games = load_midgame_positions(args.path)[4][: args.games]
    checked, rounds, rotations, unreplayable = 0, Counter(), Counter(), []
    for game_id, actions, _ in games:
        game = new_game(4)
        for i, action in enumerate(actions):
            if i == len(actions) - 1 or game._game.finished:
                break
            if i % args.every == 0 and not game._game.finished:
                expected, got = _rust_encode(game), _native_encode(game._game)
                same = all(
                    np.array_equal(got[k].numpy().view(np.uint32), expected[k].numpy().view(np.uint32)) for k in (0, 1)
                ) and tuple(got[4:]) == tuple(expected[4:])
                if not same:
                    raise SystemExit(f"game {game_id}, position {i}: encodings differ")
                checked += 1
                rounds[expected[4]] += 1
                rotations[expected[6]] += 1
            try:
                game.process_action(dict(action))
            except Exception as e:  # the file predates an engine rules change: stop this game here
                unreplayable.append((game_id, i, str(e)[:120]))
                break
    print(f"{checked} positions of {len(games)} games bit-for-bit equal; round types {dict(rounds)}, "
          f"rotations {dict(rotations)}")
    if unreplayable:
        print(f"{len(unreplayable)} games stopped early (the engine rejected an action): {unreplayable[:5]}")


if __name__ == "__main__":
    main()
