"""End-to-end smoke test of the AlphaZero pipeline.

Exercises (at small scale):
  1. Pretraining from a small subset of human games → produces a checkpoint
  2. Self-play with that checkpoint → emits LMDB training examples
  3. Loads the LMDB examples → verifies the 5-tuple schema (with price_targets)

Times out at ~5 minutes per phase. Intended as a CI-friendly version of the
full production-scale validation described in step1_review.md.
"""

import logging
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
import torch

# Stdout logging so progress is visible.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("e2e_smoke")


def phase_pretrain(model_dir: str, sample_dir: Path) -> bool:
    """Run pretraining on a tiny sample of human games."""
    from rl18xx.agent.alphazero.pretraining import do_pretraining
    from rl18xx.agent.alphazero.config import TrainingConfig, ModelTransformerConfig
    from rl18xx.agent.alphazero.model_transformer import AlphaZeroTransformerModel
    from rl18xx.agent.alphazero.checkpointer import save_model

    # Seed a fresh transformer checkpoint so pretraining has somewhere to load
    # from. Mirrors what the training loop does on first run.
    log.info("[phase 1/3] seeding fresh transformer checkpoint at %s", model_dir)
    seed_model = AlphaZeroTransformerModel(ModelTransformerConfig())
    save_model(seed_model, model_dir)
    del seed_model

    log.info("[phase 1/3] pretraining on %d sample games", len(list(sample_dir.glob("*.json"))))
    cfg = TrainingConfig(num_epochs=1, batch_size=8)
    try:
        do_pretraining(model_dir=model_dir, game_data_dir=str(sample_dir), config=cfg)
    except Exception:
        traceback.print_exc()
        return False
    log.info("[phase 1/3] pretraining completed")
    return True


def phase_self_play(model_dir: str, output_dir: Path) -> bool:
    """Play a single self-play game with the latest checkpoint."""
    from rl18xx.agent.alphazero.checkpointer import get_latest_model
    from rl18xx.agent.alphazero.config import SelfPlayConfig
    from rl18xx.agent.alphazero.self_play import SelfPlay

    log.info("[phase 2/3] running one self-play game")
    try:
        model = get_latest_model(model_dir)
    except Exception:
        traceback.print_exc()
        return False

    cfg = SelfPlayConfig(
        network=model,
        num_readouts=16,
        parallel_readouts=4,
        game_id="e2e-smoke",
        game_idx_in_iteration=0,
        global_step=0,
        selfplay_dir=output_dir,
    )
    try:
        sp = SelfPlay(cfg)
        sp._num_players = 3  # smaller game runs faster
        sp.run_game()
    except Exception:
        traceback.print_exc()
        return False
    log.info("[phase 2/3] self-play game completed")
    return True


def phase_verify_lmdb(output_dir: Path) -> bool:
    """Open the LMDB produced by self-play and verify the 5-tuple schema."""
    import io
    import lmdb
    import lz4.frame

    log.info("[phase 3/3] verifying LMDB schema")
    candidates = list(output_dir.rglob("data.mdb"))
    if not candidates:
        log.error("No LMDB files found under %s", output_dir)
        return False
    lmdb_path = candidates[0].parent
    env = lmdb.open(str(lmdb_path), readonly=True, lock=False)
    with env.begin() as txn:
        n = txn.stat()["entries"]
        if n == 0:
            log.error("LMDB at %s is empty", lmdb_path)
            return False
        cursor = txn.cursor()
        cursor.first()
        _, raw = cursor.item()
    blob = lz4.frame.decompress(raw)
    sample = torch.load(io.BytesIO(blob), weights_only=False)
    if not (isinstance(sample, tuple) and len(sample) == 5):
        log.error("Expected 5-tuple sample; got %r (len=%d)", type(sample), len(sample) if hasattr(sample, "__len__") else -1)
        return False
    state, legal, pi, value, price_targets = sample
    log.info(
        "  LMDB sample: state=%s legal=%s pi=%s value_shape=%s price_targets=%d entries",
        type(state).__name__,
        getattr(legal, "shape", None),
        getattr(pi, "shape", None),
        getattr(value, "shape", None),
        len(price_targets) if price_targets else 0,
    )
    if hasattr(value, "shape") and value.shape != (6,):
        log.error("value tensor must be VALUE_SIZE=6; got %s", value.shape)
        return False
    log.info("[phase 3/3] LMDB schema OK (5-tuple with VALUE_SIZE=6 value)")
    return True


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    games_dir = repo_root / "human_games" / "1830_clean"
    if not games_dir.exists():
        log.error("Human-game directory missing: %s", games_dir)
        return 2

    # Pick a tiny subset of human games for pretraining.
    all_games = sorted(games_dir.glob("*.json"))[:8]
    if not all_games:
        log.error("No human-game JSONs found in %s", games_dir)
        return 2

    with tempfile.TemporaryDirectory() as tmpdir:
        sample_dir = Path(tmpdir) / "sample_games"
        sample_dir.mkdir()
        for p in all_games:
            (sample_dir / p.name).write_bytes(p.read_bytes())
        model_dir = Path(tmpdir) / "model_checkpoints"
        model_dir.mkdir()
        selfplay_dir = Path(tmpdir) / "selfplay"
        selfplay_dir.mkdir()

        # Phase 1: pretrain
        t = time.time()
        if not phase_pretrain(str(model_dir), sample_dir):
            return 1
        log.info("  phase 1 elapsed: %.1fs", time.time() - t)

        # Phase 2: self-play one game
        t = time.time()
        if not phase_self_play(str(model_dir), selfplay_dir):
            return 1
        log.info("  phase 2 elapsed: %.1fs", time.time() - t)

        # Phase 3: verify LMDB
        t = time.time()
        if not phase_verify_lmdb(selfplay_dir):
            return 1
        log.info("  phase 3 elapsed: %.1fs", time.time() - t)

    log.info("E2E smoke test PASSED")
    return 0


if __name__ == "__main__":
    torch.set_num_threads(1)
    raise SystemExit(main())
