#!/usr/bin/env python3
"""
Prepare training and evaluation LMDBs from human games for autoresearch experiments.

Loads game IDs from eval_corpus/, replays and encodes each game's positions,
and writes them to LMDB databases compatible with SelfPlayDataset.

Both train and eval LMDBs are rebuilt when the encoder changes (detected via file hash).
"""

import argparse
import hashlib
import json
import logging
import shutil
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
EVAL_CORPUS_DIR = Path(__file__).resolve().parent / "eval_corpus"
DEFAULT_GAME_DIR = REPO_ROOT / "human_games" / "1830_clean"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "training_data"


def convert_game_to_samples(game_dict: dict, encoder: Encoder_GNN, action_mapper: ActionMapper) -> list:
    """Replay a single game and return encoded training samples.

    Each sample is (encoded_state, legal_action_indices, pi, value) matching
    the format expected by TrainingExampleProcessor.write_samples / SelfPlayDataset.
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
            return samples

    return samples


def process_game_list(game_ids: list, game_dir: Path, encoder: Encoder_GNN, action_mapper: ActionMapper, desc: str):
    """Process a list of game IDs and return all samples."""
    all_samples = []
    games_processed = 0
    games_failed = 0
    total = len(game_ids)

    print(f"[{desc}] Starting: {total} games", flush=True)
    for i, game_filename in enumerate(game_ids):
        if (i + 1) % 25 == 0 or i == 0:
            print(f"[{desc}] {i + 1}/{total} games, {len(all_samples)} samples so far", flush=True)

        game_path = game_dir / game_filename
        if not game_path.exists():
            LOGGER.warning(f"Game file not found: {game_path}")
            games_failed += 1
            continue

        with open(game_path, "r") as f:
            game_dict = json.load(f)

        if game_dict.get("status") == "error":
            games_failed += 1
            continue

        samples = convert_game_to_samples(game_dict, encoder, action_mapper)
        if samples:
            all_samples.extend(samples)
            games_processed += 1
        else:
            games_failed += 1

    print(f"[{desc}] Done: {games_processed} games, {len(all_samples)} samples", flush=True)
    return all_samples, games_processed, games_failed


def main():
    parser = argparse.ArgumentParser(description="Prepare training and eval LMDB from human games")
    parser.add_argument("--game-dir", type=str, default=str(DEFAULT_GAME_DIR))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Load game ID splits
    train_ids_path = EVAL_CORPUS_DIR / "train_game_ids.json"
    eval_ids_path = EVAL_CORPUS_DIR / "eval_game_ids.json"
    if not train_ids_path.exists() or not eval_ids_path.exists():
        print("ERROR: Game ID files not found. Run build_eval_corpus.py first.", file=sys.stderr)
        sys.exit(1)

    with open(train_ids_path) as f:
        train_game_ids = json.load(f)
    with open(eval_ids_path) as f:
        eval_game_ids = json.load(f)

    encoder = Encoder_GNN()
    action_mapper = ActionMapper()
    processor = TrainingExampleProcessor(encoder)
    game_dir = Path(args.game_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build training LMDB (delete first to prevent append-on-rerun)
    train_lmdb = output_dir / "train.lmdb"
    if train_lmdb.exists():
        print(f"Deleting existing {train_lmdb}", flush=True)
        shutil.rmtree(train_lmdb)
    train_samples, train_ok, train_fail = process_game_list(train_game_ids, game_dir, encoder, action_mapper, "train")
    if not train_samples:
        print("ERROR: No training samples generated", file=sys.stderr)
        sys.exit(1)
    print(f"Writing {len(train_samples)} training samples to LMDB...", flush=True)
    processor.write_samples(train_samples, train_lmdb)
    print(f"train_games_processed: {train_ok}", flush=True)
    print(f"train_games_failed: {train_fail}", flush=True)
    print(f"train_samples: {len(train_samples)}", flush=True)

    # Build eval LMDB (delete first to prevent append-on-rerun)
    eval_lmdb = output_dir / "eval.lmdb"
    if eval_lmdb.exists():
        print(f"Deleting existing {eval_lmdb}", flush=True)
        shutil.rmtree(eval_lmdb)
    eval_samples, eval_ok, eval_fail = process_game_list(eval_game_ids, game_dir, encoder, action_mapper, "eval")
    if not eval_samples:
        print("ERROR: No eval samples generated", file=sys.stderr)
        sys.exit(1)
    print(f"Writing {len(eval_samples)} eval samples to LMDB...", flush=True)
    processor.write_samples(eval_samples, eval_lmdb)
    print(f"eval_games_processed: {eval_ok}", flush=True)
    print(f"eval_games_failed: {eval_fail}", flush=True)
    print(f"eval_samples: {len(eval_samples)}", flush=True)

    # Save encoder hash so run_experiment.py knows data is up to date
    encoder_path = REPO_ROOT / "rl18xx" / "agent" / "alphazero" / "encoder.py"
    h = hashlib.sha256()
    with open(encoder_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    hash_path = output_dir / ".encoder_hash"
    hash_path.write_text(h.hexdigest())


if __name__ == "__main__":
    main()
