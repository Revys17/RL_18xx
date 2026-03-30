#!/usr/bin/env python3
"""
LOCKED experiment runner for autoresearch.

Trains the model for 1 epoch on the human game training data, then evaluates
on the frozen evaluation corpus. Prints metrics in grep-friendly format.

DO NOT MODIFY THIS FILE — it is part of the fixed autoresearch infrastructure.
"""

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

from rl18xx.agent.alphazero.config import ModelConfig, TrainingConfig
from rl18xx.agent.alphazero.model import AlphaZeroGNNModel
from rl18xx.agent.alphazero.train import train_model
from rl18xx.agent.alphazero.dataset import SelfPlayDataset

LOGGER = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
AUTORESEARCH_DIR = Path(__file__).resolve().parent
TRAINING_DATA_DIR = AUTORESEARCH_DIR / "training_data"
TRAINING_LMDB_PATH = TRAINING_DATA_DIR / "train.lmdb"
ENCODER_PATH = REPO_ROOT / "rl18xx" / "agent" / "alphazero" / "encoder.py"
ENCODER_HASH_PATH = AUTORESEARCH_DIR / "training_data" / ".encoder_hash"


def file_hash(path: Path) -> str:
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def encoder_changed() -> bool:
    """Check if encoder.py has changed since last training data preparation."""
    if not ENCODER_HASH_PATH.exists():
        return True
    stored_hash = ENCODER_HASH_PATH.read_text().strip()
    current_hash = file_hash(ENCODER_PATH)
    return stored_hash != current_hash


def save_encoder_hash():
    """Save current encoder hash."""
    ENCODER_HASH_PATH.parent.mkdir(parents=True, exist_ok=True)
    ENCODER_HASH_PATH.write_text(file_hash(ENCODER_PATH))


def prepare_training_data():
    """Re-encode training data if encoder has changed or data doesn't exist."""
    LOGGER.info("Preparing training data (encoder changed or data missing)...")
    result = subprocess.run(
        ["uv", "run", "python", "-m", "autoresearch.prepare_training_data"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    if result.returncode != 0:
        print(f"ERROR: prepare_training_data.py failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)
    LOGGER.info(result.stdout)
    save_encoder_hash()


def main():
    parser = argparse.ArgumentParser(description="Run one autoresearch experiment")
    parser.add_argument("--num-epochs", type=int, default=1, help="Training epochs per experiment")
    parser.add_argument("--batch-size", type=int, default=256, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    start_time = time.time()

    # Step 1: Check if training data needs re-encoding
    if encoder_changed() or not TRAINING_LMDB_PATH.exists():
        prepare_training_data()
    else:
        LOGGER.info("Training data is up to date (encoder unchanged)")

    # Step 2: Initialize fresh model (random weights)
    LOGGER.info("Initializing fresh model...")
    config = ModelConfig()
    model = AlphaZeroGNNModel(config)

    # Step 3: Train for N epochs on human game data
    LOGGER.info(f"Training for {args.num_epochs} epoch(s)...")
    training_config = TrainingConfig(
        train_dir=TRAINING_LMDB_PATH,
        batch_size=args.batch_size,
        lr=args.lr,
        num_epochs=args.num_epochs,
    )

    train_dataset = SelfPlayDataset(TRAINING_LMDB_PATH)
    if len(train_dataset) == 0:
        print("ERROR: Training dataset is empty. Run prepare_training_data.py first.", file=sys.stderr)
        sys.exit(1)

    metrics = train_model(model, train_dataset, training_config)
    LOGGER.info(f"Training complete. Loss: {metrics.avg_total_loss:.4f}")

    # Step 4: Evaluate on frozen evaluation corpus
    # Save model to a temporary checkpoint so evaluate.py can load it
    LOGGER.info("Running evaluation...")
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        from rl18xx.agent.alphazero.checkpointer import save_model

        save_model(model, tmpdir, new=False)

        eval_result = subprocess.run(
            [
                "uv",
                "run",
                "python",
                "-m",
                "autoresearch.evaluate",
                "--model-dir",
                tmpdir,
            ],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )

    elapsed = time.time() - start_time

    if eval_result.returncode != 0:
        print(f"ERROR: evaluate.py failed:\n{eval_result.stderr}", file=sys.stderr)
        sys.exit(1)

    # Step 5: Print metrics (pass through from evaluate.py)
    print(eval_result.stdout.strip())
    print(f"training_loss: {metrics.avg_total_loss:.6f}")
    print(f"training_policy_loss: {metrics.avg_policy_loss:.6f}")
    print(f"training_value_loss: {metrics.avg_value_loss:.6f}")
    print(f"experiment_time_seconds: {elapsed:.1f}")


if __name__ == "__main__":
    main()
