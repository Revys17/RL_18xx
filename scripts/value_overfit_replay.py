"""Replay a finished self-play run's training under different value-loss settings.

Self-play data doesn't depend on the training settings within one iteration,
so how a setting changes value-head generalization can be measured offline:
start from the checkpoint the run had after iteration ``--first-iter - 1``,
then for each later iteration train exactly as the loop did (sample
``train_samples_per_iteration`` from the last ``max_training_window`` examples
present at that point) and score the value head on the NEXT iteration's games,
which it never trained on, and on its own training window.

Iteration boundaries come from the run log's "Training window active: using W
of N examples" lines (N = examples present when that iteration trained).
Checkpoints and optimizer state go to ``--out`` (never model_checkpoints/).

    uv run python scripts/value_overfit_replay.py --log logs/run100b_2026-10-02.log \\
        --start-ckpt model_checkpoints/AlphaZeroTransformer/<session>/57.pth --first-iter 51 \\
        --iters 25 --name w025 --set value_loss_weight=0.25 --set score_loss_weight=0.025
"""

import argparse
import json
import logging
import math
import random
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

LOGGER = logging.getLogger("value_overfit_replay")


def iteration_totals(log_path: Path) -> dict:
    """``{iteration: examples present when it trained}`` from a loop log.

    The window line is only logged once the data exceeds the window, so the
    first iterations of a run may be missing; they're numbered back from the
    last iteration (the highest ``Loop N:`` in the log).
    """
    text = log_path.read_text(errors="replace")
    totals = [int(m) for m in re.findall(r"Training window active: using \d+ of (\d+) examples", text)]
    loops = [int(m) for m in re.findall(r"Loop (\d+): ", text)]
    last = max(loops) if loops else len(totals)
    first = last - len(totals) + 1
    return {first + i: n for i, n in enumerate(totals)}


def parse_overrides(pairs: list) -> dict:
    out = {}
    for pair in pairs or []:
        key, value = pair.split("=", 1)
        if value.lower() in ("true", "false"):
            out[key] = value.lower() == "true"
        else:
            try:
                out[key] = int(value)
            except ValueError:
                out[key] = float(value)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=None, help="Self-play LMDB (default: the start checkpoint's)")
    parser.add_argument("--start-ckpt", type=Path, required=True)
    parser.add_argument("--first-iter", type=int, required=True, help="First iteration to replay (1-based)")
    parser.add_argument("--iters", type=int, default=25)
    parser.add_argument("--name", required=True)
    parser.add_argument("--set", action="append", default=[], help="TrainingConfig override key=value")
    parser.add_argument("--value-stop-grad", action="store_true", help="Value heads train on detached trunk features")
    parser.add_argument("--eval-sample", type=int, default=5000)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "logs" / "value_replay")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import torch
    from torch.utils.data import Subset

    from rl18xx.agent.alphazero.checkpointer import _load_model_from_session_checkpoint
    from rl18xx.agent.alphazero.config import TrainingConfig
    from rl18xx.agent.alphazero.dataset import SelfPlayDataset
    from rl18xx.agent.alphazero.train import train_model, value_cross_entropy

    out_dir = args.out / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[logging.FileHandler(out_dir / "replay.log"), logging.StreamHandler()],
    )
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    totals = iteration_totals(args.log)
    model = _load_model_from_session_checkpoint(args.start_ckpt.parent, args.start_ckpt)
    model.value_stop_grad = bool(args.value_stop_grad)
    data_dir = args.data or REPO_ROOT / "training_examples" / "selfplay" / model.get_name()
    dataset = SelfPlayDataset(data_dir)

    loop_config = json.loads((REPO_ROOT / "loop_config.json").read_text())
    base = dict(loop_config.get("training_config") or {})
    base.update(parse_overrides(args.set))
    config = TrainingConfig.from_json(base)
    config.metrics = None
    window = config.max_training_window
    LOGGER.info(f"Replay {args.name}: {config.to_json()} value_stop_grad={model.value_stop_grad}")

    results_path = out_dir / "results.jsonl"
    results_path.unlink(missing_ok=True)
    rng = random.Random(args.seed)
    for it in range(args.first_iter, args.first_iter + args.iters):
        if it not in totals or it + 1 not in totals:
            LOGGER.warning(f"No window line for iteration {it} or {it + 1}; stopping")
            break
        end = totals[it]
        start = max(0, end - window)
        sample = rng.sample(range(start, end), min(config.train_samples_per_iteration or (end - start), end - start))
        t0 = time.time()
        train_model(model, Subset(dataset, sorted(sample)), config, model_checkpoint_dir=str(out_dir / "ckpt"))
        train_s = time.time() - t0
        # Keep only the newest checkpoint file; optimizer state lives beside it.
        ckpts = sorted((out_dir / "ckpt").glob("*/*/*.pth"), key=lambda p: int(p.stem) if p.stem.isdigit() else -1)
        for old in ckpts[:-1]:
            if old.stem.isdigit():
                old.unlink()

        new_games = list(range(end, totals[it + 1]))
        trained = sorted(sample)
        eval_rng = random.Random(it)
        record = {
            "iter": it,
            "ce_new_games": value_cross_entropy(
                model, dataset, sorted(eval_rng.sample(new_games, min(args.eval_sample, len(new_games))))
            ),
            "ce_trained_window": value_cross_entropy(
                model, dataset, sorted(eval_rng.sample(range(start, end), min(args.eval_sample, end - start)))
            ),
            "ce_trained_this_iter": value_cross_entropy(
                model, dataset, eval_rng.sample(trained, min(args.eval_sample, len(trained)))
            ),
            "train_seconds": round(train_s, 1),
        }
        LOGGER.info(
            f"[{args.name}] iter {it}: CE new {record['ce_new_games']:.3f}, window {record['ce_trained_window']:.3f}, "
            f"this-iter sample {record['ce_trained_this_iter']:.3f} (uniform {math.log(4):.3f})"
        )
        with open(results_path, "a") as f:
            f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
