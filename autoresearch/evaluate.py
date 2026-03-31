#!/usr/bin/env python3
"""
LOCKED evaluation script for autoresearch.

Loads pre-encoded evaluation positions from LMDB, runs the model forward pass,
and computes metrics against human move targets. Output is grep-friendly.

DO NOT MODIFY THIS FILE — it is the fixed evaluation harness.
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from rl18xx.agent.alphazero.checkpointer import get_latest_model
from rl18xx.agent.alphazero.dataset import SelfPlayDataset

LOGGER = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EVAL_LMDB = Path(__file__).resolve().parent / "training_data" / "eval.lmdb"
DEFAULT_MODEL_DIR = REPO_ROOT / "model_checkpoints"


def compute_metrics(model, eval_lmdb_path: Path, batch_size: int = 64):
    """Run model on pre-encoded eval positions and compute metrics."""
    dataset = SelfPlayDataset(eval_lmdb_path)
    if len(dataset) == 0:
        return {}

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_policy_losses = []
    all_value_losses = []
    top1_correct = 0
    top5_correct = 0
    value_winner_correct = 0
    total = 0

    device = model.device
    model.eval()

    with torch.no_grad():
        for batch in loader:
            game_state_data, batch_data, legal_action_mask, pi, value = batch

            game_state_data = game_state_data.to(device).squeeze(1)
            batch_data = batch_data.to(device)
            legal_action_mask = legal_action_mask.to(device)
            pi = pi.to(device)
            value = value.to(device)

            _, move_log_probs, value_pred = model(game_state_data, batch_data)

            # Policy loss: cross-entropy vs smoothed human targets
            masked_log_probs = move_log_probs * legal_action_mask + 1e-8
            policy_loss = -torch.sum(pi * masked_log_probs, dim=1)
            all_policy_losses.extend(policy_loss.cpu().tolist())

            # Value loss: cross-entropy (who wins?)
            # Convert {-1, 0, +1} targets to probability distribution
            winners_mask = (value > -0.5).float()  # +1 and 0 are winners
            num_winners = winners_mask.sum(dim=1, keepdim=True).clamp(min=1)
            value_target_probs = winners_mask / num_winners
            value_log_probs = F.log_softmax(value_pred, dim=1)
            value_loss = -(value_target_probs * value_log_probs).sum(dim=1)
            all_value_losses.extend(value_loss.cpu().tolist())

            # Value accuracy: does the model predict the correct winner?
            predicted_winner = value_pred.argmax(dim=1)
            actual_winner = value.argmax(dim=1)
            value_winner_correct += (predicted_winner == actual_winner).sum().item()

            # Top-k accuracy among legal actions only
            logits = move_log_probs.clone()
            logits[legal_action_mask == 0] = float("-inf")

            # Human action is the argmax of pi (the 0.97 entry)
            human_actions = pi.argmax(dim=1)

            # Top-1
            predicted_top1 = logits.argmax(dim=1)
            top1_correct += (predicted_top1 == human_actions).sum().item()

            # Top-5
            _, predicted_top5 = logits.topk(min(5, logits.size(1)), dim=1)
            for i in range(human_actions.size(0)):
                if human_actions[i] in predicted_top5[i]:
                    top5_correct += 1

            total += human_actions.size(0)

    policy_loss_mean = np.mean(all_policy_losses)
    value_loss_mean = np.mean(all_value_losses)

    return {
        "policy_loss": policy_loss_mean,
        "value_loss": value_loss_mean,
        "combined_loss": policy_loss_mean + value_loss_mean,
        "top1_accuracy": top1_correct / total if total > 0 else 0.0,
        "top5_accuracy": top5_correct / total if total > 0 else 0.0,
        "value_accuracy": value_winner_correct / total if total > 0 else 0.0,
        "positions_evaluated": total,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on pre-encoded human game positions")
    parser.add_argument("--model-dir", type=str, default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--eval-lmdb", type=str, default=str(DEFAULT_EVAL_LMDB))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    eval_lmdb = Path(args.eval_lmdb)
    if not eval_lmdb.exists():
        print("ERROR: Eval LMDB not found. Run prepare_training_data.py first.", file=sys.stderr)
        sys.exit(1)

    model = get_latest_model(args.model_dir)
    model.eval()

    metrics = compute_metrics(model, eval_lmdb, batch_size=args.batch_size)

    if not metrics:
        print("ERROR: No positions to evaluate", file=sys.stderr)
        sys.exit(1)

    print(f"policy_loss: {metrics['policy_loss']:.6f}")
    print(f"value_loss: {metrics['value_loss']:.6f}")
    print(f"combined_loss: {metrics['combined_loss']:.6f}")
    print(f"top1_accuracy: {metrics['top1_accuracy']:.4f}")
    print(f"top5_accuracy: {metrics['top5_accuracy']:.4f}")
    print(f"value_accuracy: {metrics['value_accuracy']:.4f}")
    print(f"positions_evaluated: {metrics['positions_evaluated']}")


if __name__ == "__main__":
    main()
