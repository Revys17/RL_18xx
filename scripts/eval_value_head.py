"""Evaluate a checkpoint's win-loss head on held-out human games, by game progress.

A value head that learned something should sit near the equal-odds baseline at
the opening and approach certainty about the eventual winner by the end game.
This walks a pretraining LMDB (rows are written game by game; a game's stored
absolute value vector is constant across its rows, which marks the boundaries),
runs the model over each game's positions, and reports per progress decile:

- winner accuracy: argmax seat is one of the actual winners (chance ~= 1/N)
- p(winner): probability mass the head puts on the actual winners
- value CE vs. the ln(N) equal-odds baseline

Usage:
    uv run python scripts/eval_value_head.py [--checkpoint SESSION/NUM] [--lmdb human_games/lmdb_v3/validation]
"""

import argparse
import io
import math
import sys
from collections import defaultdict
from pathlib import Path

import lmdb
import lz4.frame
import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from rl18xx.agent.alphazero.checkpointer import _load_model_from_session_checkpoint, get_current_best
from rl18xx.agent.alphazero.dataset import canonicalize_value_target
from rl18xx.agent.alphazero.train import _derive_dual_value_targets


def iter_games(lmdb_path: str, max_games: int):
    """Yield lists of ``(state, canonical_value)`` rows, one list per game."""
    env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False)
    with env.begin() as txn:
        total = txn.stat()["entries"]
        game, prev_value, n_games = [], None, 0
        for i in range(total):
            raw = txn.get(f"{i:08}".encode("ascii"))
            state, _legal, _pi, value, *_ = torch.load(
                io.BytesIO(lz4.frame.decompress(raw)), map_location="cpu", weights_only=False
            )
            if prev_value is not None and not torch.equal(value, prev_value):
                yield game
                n_games += 1
                if n_games >= max_games:
                    return
                game = []
            game.append((state, canonicalize_value_target(state, value)))
            prev_value = value
        if game:
            yield game


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", default="model_checkpoints")
    parser.add_argument("--checkpoint", default=None, help="SESSION/NUM (default: current_best)")
    parser.add_argument("--lmdb", default="human_games/lmdb_v3/validation")
    parser.add_argument("--max-games", type=int, default=10_000)
    parser.add_argument("--deciles", type=int, default=10)
    args = parser.parse_args()

    if args.checkpoint:
        session, num = args.checkpoint.split("/")
    else:
        best = get_current_best(args.model_dir)
        session, num = best["session"], best["checkpoint_num"]
    ckpt = next(Path(args.model_dir).glob(f"*/{session}/{num}.pth"))
    model = _load_model_from_session_checkpoint(ckpt.parent, ckpt)
    model.eval()
    print(f"checkpoint: {ckpt}")

    stats = defaultdict(lambda: {"n": 0, "hits": 0, "p_win": 0.0, "ce": 0.0, "baseline": 0.0})
    n_games = 0
    for game in iter_games(args.lmdb, args.max_games):
        n_games += 1
        gs = torch.stack([s[0].reshape(-1) for s, _ in game]).float().to(model.device)
        nodes = torch.stack([s[1] for s, _ in game]).float().to(model.device)
        values = torch.stack([v for _, v in game]).float().to(model.device)
        with torch.no_grad():
            _, win_loss_logits, _, _ = model(gs, nodes)
        win_target, _ = _derive_dual_value_targets(values)
        n_players = win_target.shape[1]
        log_probs = F.log_softmax(win_loss_logits.float(), dim=1)[:, :n_players]
        probs = log_probs.exp()
        ce = -(win_target * log_probs).sum(dim=1)
        hits = win_target.gather(1, probs.argmax(dim=1, keepdim=True)).squeeze(1) > 0
        p_win = (probs * (win_target > 0)).sum(dim=1)
        length = len(game)
        for i in range(length):
            d = min(int(i / length * args.deciles), args.deciles - 1)
            for key in (d, "all"):
                s = stats[key]
                s["n"] += 1
                s["hits"] += int(hits[i])
                s["p_win"] += float(p_win[i])
                s["ce"] += float(ce[i])
                s["baseline"] += math.log(n_players)

    print(f"games: {n_games}")
    print(f"{'progress':>10} {'positions':>9} {'winner_acc':>10} {'p(winner)':>9} {'value_CE':>9} {'uniform':>8}")
    for key in list(range(args.deciles)) + ["all"]:
        s = stats[key]
        if not s["n"]:
            continue
        label = f"{key * 100 // args.deciles}-{(key + 1) * 100 // args.deciles}%" if key != "all" else "all"
        n = s["n"]
        print(
            f"{label:>10} {n:>9} {s['hits'] / n:>10.3f} {s['p_win'] / n:>9.3f} "
            f"{s['ce'] / n:>9.4f} {s['baseline'] / n:>8.4f}"
        )


if __name__ == "__main__":
    main()
