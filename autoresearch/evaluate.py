#!/usr/bin/env python3
"""
LOCKED evaluation script for autoresearch.

Loads 50 evaluation games, replays them through the engine, encodes each
position, runs the model forward pass, and computes metrics against human
move targets. Output is grep-friendly (one key: value per line).

DO NOT MODIFY THIS FILE — it is the fixed evaluation harness.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset

from rl18xx.agent.alphazero.action_mapper import ActionMapper
from rl18xx.agent.alphazero.checkpointer import get_latest_model
from rl18xx.agent.alphazero.encoder import Encoder_GNN
from rl18xx.agent.alphazero.pretraining import (
    make_action_model_friendly,
    make_encoded_game_state_model_friendly,
)
from rl18xx.game.engine.actions import BaseAction
from rl18xx.game.engine.game import BaseGame
from rl18xx.game.gamemap import GameMap

LOGGER = logging.getLogger(__name__)

# Paths relative to repo root
REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_IDS_PATH = Path(__file__).resolve().parent / "eval_corpus" / "eval_game_ids.json"
DEFAULT_GAME_DIR = REPO_ROOT / "human_games" / "1830_clean"
DEFAULT_MODEL_DIR = REPO_ROOT / "model_checkpoints"


class EvalDataset(Dataset):
    """Dataset of pre-collected evaluation positions."""

    def __init__(self, positions: List[dict]):
        self.positions = positions
        self.action_mapper = ActionMapper()

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, index):
        pos = self.positions[index]
        game_state_data, node_data, edge_index, edge_attr = pos["encoded_state"]
        legal_action_mask = torch.from_numpy(self.action_mapper.convert_indices_to_mask(pos["legal_action_indices"]))
        data = Data(x=node_data, edge_index=edge_index, edge_attr=edge_attr)
        return game_state_data, data, legal_action_mask, pos["pi"], pos["value"], pos["action_index"]


def replay_game_positions(game_dict: dict, encoder: Encoder_GNN, action_mapper: ActionMapper) -> List[dict]:
    """Replay a single game and collect evaluation positions.

    Follows the same logic as pretraining.py:convert_game_to_training_data().
    Returns a list of position dicts, or empty list on failure.
    """
    try:
        game_obj = BaseGame.load(game_dict)
    except Exception as e:
        LOGGER.warning(f"Failed to load game {game_dict.get('id', '?')}: {e}")
        return []

    game_map = GameMap()
    game_class = game_map.game_by_title("1830")
    players = {1: "Player 1", 2: "Player 2", 3: "Player 3", 4: "Player 4"}
    fresh_game_state = game_class(players)

    # Compute value targets from game outcome
    winning_score = max(game_obj.result().values())
    actual_value = torch.full((4,), -1.0, dtype=torch.float32)
    player_mapping = {p.id: i for i, p in enumerate(sorted(game_obj.players, key=lambda x: x.id))}
    winners = [player_mapping[pid] for pid, score in game_obj.result().items() if score == winning_score]
    if len(winners) > 1:
        actual_value[winners] = 0.0
    else:
        actual_value[winners] = 1.0

    positions = []
    epsilon = 0.03

    for action in game_obj.raw_actions:
        try:
            updated_action = make_action_model_friendly(fresh_game_state, action.copy())
            encoded_game_state = encoder.encode(fresh_game_state)
            updated_encoded_game_state = make_encoded_game_state_model_friendly(
                encoder, encoded_game_state, updated_action, fresh_game_state
            )
            action_index = action_mapper.get_index_for_action(
                BaseAction.action_from_dict(updated_action, fresh_game_state), fresh_game_state
            )
            legal_action_indices = action_mapper.get_legal_action_indices(fresh_game_state)

            # Smoothed one-hot policy target
            pi = torch.zeros(action_mapper.action_encoding_size)
            pi[legal_action_indices] += epsilon / len(legal_action_indices)
            pi[action_index] = 1.0 - epsilon

            positions.append(
                {
                    "encoded_state": updated_encoded_game_state,
                    "action_index": torch.tensor(action_index, dtype=torch.long),
                    "legal_action_indices": legal_action_indices,
                    "pi": pi,
                    "value": actual_value,
                }
            )

            fresh_game_state.process_action(action)
        except Exception as e:
            LOGGER.warning(f"Error at action in game {game_dict.get('id', '?')}: {e}")
            return positions  # Return what we have so far

    return positions


def compute_metrics(
    model,
    positions: List[dict],
    batch_size: int = 64,
):
    """Run model on all positions and compute evaluation metrics."""
    if not positions:
        return {}

    dataset = EvalDataset(positions)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_policy_losses = []
    all_value_losses = []
    top1_correct = 0
    top5_correct = 0
    total = 0

    device = model.device
    model.eval()

    with torch.no_grad():
        for batch in loader:
            game_state_data, batch_data, legal_action_mask, pi, value, action_indices = batch

            # DataLoader adds an extra dimension to game_state_data
            game_state_data = game_state_data.to(device).squeeze(1)
            batch_data = batch_data.to(device)
            legal_action_mask = legal_action_mask.to(device)
            pi = pi.to(device)
            value = value.to(device)
            action_indices = action_indices.to(device)

            # Forward pass
            _, move_log_probs, value_pred = model(game_state_data, batch_data)

            # Policy loss: cross-entropy vs smoothed human targets
            masked_log_probs = move_log_probs * legal_action_mask
            masked_log_probs = masked_log_probs + 1e-8
            policy_loss = -torch.sum(pi * masked_log_probs, dim=1)
            all_policy_losses.extend(policy_loss.cpu().tolist())

            # Value loss: MSE vs game outcomes
            value_loss = F.mse_loss(value_pred, value, reduction="none").mean(dim=1)
            all_value_losses.extend(value_loss.cpu().tolist())

            # Top-k accuracy: mask illegal actions to -inf, then check argmax/top5
            # Use raw log_probs for ranking (before masking multiplication)
            logits_for_ranking = move_log_probs.clone()
            logits_for_ranking[legal_action_mask == 0] = float("-inf")

            # Top-1: does argmax match human move?
            predicted_top1 = logits_for_ranking.argmax(dim=1)
            top1_correct += (predicted_top1 == action_indices).sum().item()

            # Top-5: is human move in top 5?
            _, predicted_top5 = logits_for_ranking.topk(min(5, logits_for_ranking.size(1)), dim=1)
            for i in range(action_indices.size(0)):
                if action_indices[i] in predicted_top5[i]:
                    top5_correct += 1

            total += action_indices.size(0)

    return {
        "policy_loss": np.mean(all_policy_losses),
        "top1_accuracy": top1_correct / total if total > 0 else 0.0,
        "top5_accuracy": top5_correct / total if total > 0 else 0.0,
        "value_loss": np.mean(all_value_losses),
        "positions_evaluated": total,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on human game positions")
    parser.add_argument("--model-dir", type=str, default=str(DEFAULT_MODEL_DIR), help="Model checkpoint directory")
    parser.add_argument("--game-dir", type=str, default=str(DEFAULT_GAME_DIR), help="Directory with clean game JSONs")
    parser.add_argument("--eval-ids", type=str, default=str(EVAL_IDS_PATH), help="Path to eval_game_ids.json")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for evaluation")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Load evaluation game IDs
    eval_ids_path = Path(args.eval_ids)
    if not eval_ids_path.exists():
        print(f"ERROR: Eval game IDs not found at {eval_ids_path}. Run build_eval_corpus.py first.", file=sys.stderr)
        sys.exit(1)

    with open(eval_ids_path, "r") as f:
        eval_game_ids = json.load(f)

    LOGGER.info(f"Loaded {len(eval_game_ids)} evaluation game IDs")

    # Load model
    model = get_latest_model(args.model_dir)
    model.eval()

    # Set up encoder and action mapper
    encoder = model.encoder
    action_mapper = ActionMapper()

    # Replay games and collect positions
    game_dir = Path(args.game_dir)
    all_positions = []
    games_loaded = 0
    games_failed = 0

    for game_filename in eval_game_ids:
        game_path = game_dir / game_filename
        if not game_path.exists():
            LOGGER.warning(f"Game file not found: {game_path}")
            games_failed += 1
            continue

        with open(game_path, "r") as f:
            game_dict = json.load(f)

        if game_dict.get("status") == "error":
            LOGGER.warning(f"Skipping game {game_filename} with status=error")
            games_failed += 1
            continue

        positions = replay_game_positions(game_dict, encoder, action_mapper)
        if positions:
            all_positions.extend(positions)
            games_loaded += 1
        else:
            LOGGER.warning(f"No positions extracted from game {game_filename}")
            games_failed += 1

    LOGGER.info(f"Games loaded: {games_loaded}, failed: {games_failed}, total positions: {len(all_positions)}")

    if not all_positions:
        print("ERROR: No positions to evaluate", file=sys.stderr)
        sys.exit(1)

    # Compute metrics
    metrics = compute_metrics(model, all_positions, batch_size=args.batch_size)

    # Print grep-friendly output
    print(f"policy_loss: {metrics['policy_loss']:.6f}")
    print(f"top1_accuracy: {metrics['top1_accuracy']:.4f}")
    print(f"top5_accuracy: {metrics['top5_accuracy']:.4f}")
    print(f"value_loss: {metrics['value_loss']:.6f}")
    print(f"positions_evaluated: {metrics['positions_evaluated']}")


if __name__ == "__main__":
    main()
