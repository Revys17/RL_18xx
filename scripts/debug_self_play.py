"""Smoke-test one self-play game with a small readout budget to surface the
all-zero result bug. Exits non-zero if the result is all zeros."""

import logging
import sys
import traceback

import numpy as np
import torch

from rl18xx.agent.alphazero.checkpointer import get_latest_model
from rl18xx.agent.alphazero.config import SelfPlayConfig
from rl18xx.agent.alphazero.self_play import SelfPlay


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        stream=sys.stdout,
    )

    model = get_latest_model("model_checkpoints")

    cfg = SelfPlayConfig(
        network=model,
        # Tiny budget so the test finishes fast.
        num_readouts=8,
        parallel_readouts=2,
        # Use default max_game_length (1000) so we exercise the natural-end path.
        game_id="debug-smoke",
        game_idx_in_iteration=0,
        global_step=0,
    )

    sp = SelfPlay(cfg)
    sp._num_players = 4  # Force 4-player game for repro
    try:
        player = sp.play()
    except Exception:
        traceback.print_exc()
        return 2

    print(f"player.result type: {type(player.result)}, value: {player.result}")
    print(f"player.result_string: {player.result_string}")
    print(f"moves played: {len(player.played_actions)}")
    print(f"root is_done: {player.root.is_done()}")
    print(f"game finished: {player.root.game_object.finished}")
    print(f"engine result: {player.root.game_object.result()}")

    if player.result is None or np.all(np.array(player.result) == 0.0):
        print("BUG REPRODUCED: result is all zeros")
        return 1
    return 0


if __name__ == "__main__":
    torch.set_num_threads(1)
    raise SystemExit(main())
