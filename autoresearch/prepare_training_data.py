#!/usr/bin/env python3
"""
Prepare the 200 training games as an LMDB dataset for autoresearch experiments.

Loads training game IDs from eval_corpus/train_game_ids.json, replays and encodes
each game's positions, and writes them to an LMDB database compatible with
SelfPlayDataset / the standard training loop.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from tqdm import tqdm

from rl18xx.agent.alphazero.action_mapper import ActionMapper
from rl18xx.agent.alphazero.encoder import Encoder_GNN
from rl18xx.agent.alphazero.dataset import TrainingExampleProcessor
from rl18xx.agent.alphazero.pretraining import (
    make_action_model_friendly,
    make_encoded_game_state_model_friendly,
)
from rl18xx.game.engine.actions import BaseAction
from rl18xx.game.engine.game import BaseGame
from rl18xx.game.gamemap import GameMap

LOGGER = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_IDS_PATH = Path(__file__).resolve().parent / "eval_corpus" / "train_game_ids.json"
DEFAULT_GAME_DIR = REPO_ROOT / "human_games" / "1830_clean"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "training_data"


def convert_game_to_samples(game_dict: dict, encoder: Encoder_GNN, action_mapper: ActionMapper) -> list:
    """Replay a single game and return encoded training samples.

    Each sample is (encoded_state, legal_action_indices, pi, value) matching
    the format expected by TrainingExampleProcessor.write_samples / SelfPlayDataset.

    Follows the same logic as pretraining.py:convert_game_to_training_data().
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

    samples = []
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

            samples.append((updated_encoded_game_state, legal_action_indices, pi, actual_value))

            fresh_game_state.process_action(action)
        except Exception as e:
            LOGGER.warning(f"Error at action in game {game_dict.get('id', '?')}: {e}")
            return samples  # Return what we have so far

    return samples


def main():
    parser = argparse.ArgumentParser(description="Prepare training data LMDB from human games")
    parser.add_argument("--game-dir", type=str, default=str(DEFAULT_GAME_DIR), help="Directory with clean game JSONs")
    parser.add_argument("--train-ids", type=str, default=str(TRAIN_IDS_PATH), help="Path to train_game_ids.json")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Output LMDB directory")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Load training game IDs
    train_ids_path = Path(args.train_ids)
    if not train_ids_path.exists():
        print(f"ERROR: Train game IDs not found at {train_ids_path}. Run build_eval_corpus.py first.", file=sys.stderr)
        sys.exit(1)

    with open(train_ids_path, "r") as f:
        train_game_ids = json.load(f)

    LOGGER.info(f"Loaded {len(train_game_ids)} training game IDs")

    # Set up encoder and action mapper
    encoder = Encoder_GNN()
    action_mapper = ActionMapper()
    processor = TrainingExampleProcessor(encoder)

    # Replay games and collect samples
    game_dir = Path(args.game_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_samples = []
    games_processed = 0
    games_failed = 0

    for game_filename in tqdm(train_game_ids, desc="Processing training games"):
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

        samples = convert_game_to_samples(game_dict, encoder, action_mapper)
        if samples:
            all_samples.extend(samples)
            games_processed += 1
        else:
            LOGGER.warning(f"No samples from game {game_filename}")
            games_failed += 1

    LOGGER.info(f"Games processed: {games_processed}, failed: {games_failed}")
    LOGGER.info(f"Total training samples: {len(all_samples)}")

    if not all_samples:
        print("ERROR: No training samples generated", file=sys.stderr)
        sys.exit(1)

    # Write to LMDB
    lmdb_path = output_dir / "train.lmdb"
    processor.write_samples(all_samples, lmdb_path)

    print(f"games_processed: {games_processed}")
    print(f"games_failed: {games_failed}")
    print(f"total_samples: {len(all_samples)}")
    print(f"lmdb_path: {lmdb_path}")


if __name__ == "__main__":
    main()
