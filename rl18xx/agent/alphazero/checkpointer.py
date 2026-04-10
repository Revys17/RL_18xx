from rl18xx.agent.alphazero.model import AlphaZeroGNNModel, AlphaZeroModel
from rl18xx.agent.alphazero.config import ModelConfig, ModelV2Config
from pathlib import Path
from datetime import datetime
import json
import logging
import torch

LOGGER = logging.getLogger(__name__)


def _load_model_from_config(config_data: dict, checkpoint_path: str) -> AlphaZeroModel:
    """Load the appropriate model type based on config contents."""
    if "d_entity" in config_data or "hex_transformer_layers" in config_data:
        from rl18xx.agent.alphazero.model_v2 import AlphaZeroV2Model

        config = ModelV2Config.from_json(config_data)
        config.model_checkpoint_file = checkpoint_path
        return AlphaZeroV2Model(config)
    else:
        config = ModelConfig.from_json(config_data)
        config.model_checkpoint_file = checkpoint_path
        return AlphaZeroGNNModel(config)


def _get_session_dir(model: AlphaZeroModel, checkpoint_dir: str) -> Path:
    """Get the session directory for a model: <checkpoint_dir>/<architecture>/<timestamp>_<seed>/"""
    arch = model.architecture_name()
    seed = getattr(model.config, "seed", None)
    if seed is None:
        seed = "unknown"
    session = f"{model.config.timestamp}_{seed}"
    return Path(checkpoint_dir) / arch / session


def _find_latest_session(checkpoint_dir: str) -> Path:
    """Find the latest session directory across all architectures."""
    p = Path(checkpoint_dir)
    if not p.exists() or not p.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    latest_session = None
    for arch_dir in p.iterdir():
        if not arch_dir.is_dir():
            continue
        for session_dir in arch_dir.iterdir():
            if not session_dir.is_dir():
                continue
            if latest_session is None or session_dir.name > latest_session.name:
                latest_session = session_dir

    if latest_session is None:
        raise FileNotFoundError(f"No session directories found in {checkpoint_dir}")
    return latest_session


def _find_latest_checkpoint(session_dir: Path) -> Path:
    """Find the highest-numbered .pth checkpoint in a session directory."""
    checkpoints = sorted(
        [p for p in session_dir.glob("*.pth") if p.stem.isdigit()],
        key=lambda x: int(x.stem),
    )
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint files found in {session_dir}")
    return checkpoints[-1]


def _next_checkpoint_num(session_dir: Path) -> int:
    """Get the next checkpoint number for a session."""
    existing = [p for p in session_dir.glob("*.pth") if p.stem.isdigit()]
    if not existing:
        return 1
    return max(int(p.stem) for p in existing) + 1


def get_latest_model(model_checkpoint_dir: str) -> AlphaZeroModel:
    """Load the latest checkpoint from the most recent session."""
    session_dir = _find_latest_session(model_checkpoint_dir)
    checkpoint_path = _find_latest_checkpoint(session_dir)
    config_path = session_dir / "config.json"

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config_data = json.load(f)

    model = _load_model_from_config(config_data, str(checkpoint_path))
    checkpoint_num = int(checkpoint_path.stem)
    LOGGER.info(
        f"Loaded model from: {session_dir.parent.name}/{session_dir.name}/checkpoint {checkpoint_num}"
    )
    return model


def get_model_from_path(model_checkpoint_path: str) -> AlphaZeroModel:
    """Load a model from a specific path. Supports both old flat format and new hierarchy."""
    p = Path(model_checkpoint_path)
    if not p.exists() or not p.is_dir():
        raise FileNotFoundError(f"Model checkpoint directory not found: {model_checkpoint_path}")

    config_path = p / "config.json"
    checkpoint_path = p / "checkpoint.pth"

    # New format: look for numbered .pth files
    if not checkpoint_path.exists():
        numbered = sorted(
            [f for f in p.glob("*.pth") if f.stem.isdigit()],
            key=lambda x: int(x.stem),
        )
        if numbered:
            checkpoint_path = numbered[-1]

    # Old format: nested directory (from pretraining)
    if not config_path.exists():
        subdirs = [d for d in p.iterdir() if d.is_dir()]
        if subdirs:
            nested = max(subdirs, key=lambda x: x.name)
            config_path = nested / "config.json"
            if not checkpoint_path.exists():
                checkpoint_path = nested / "checkpoint.pth"

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    with open(config_path, "r") as f:
        config_data = json.load(f)

    model = _load_model_from_config(config_data, str(checkpoint_path))
    LOGGER.info(f"Loaded model from path {model_checkpoint_path}: {model.get_name()}")
    return model


def save_model(model: AlphaZeroModel, model_checkpoint_dir: str) -> int:
    """Save model as the next numbered checkpoint in its session directory.

    Returns the checkpoint number that was saved.
    """
    session_dir = _get_session_dir(model, model_checkpoint_dir)
    session_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_num = _next_checkpoint_num(session_dir)
    checkpoint_path = session_dir / f"{checkpoint_num}.pth"

    model.save_weights(str(checkpoint_path))

    config_path = session_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(model.config.to_json(), f, indent=4)

    return checkpoint_num


def save_optimizer_state(
    optimizer: torch.optim.Optimizer,
    scheduler,
    model_checkpoint_dir: str,
    model: AlphaZeroModel,
) -> None:
    """Save optimizer and scheduler state to the model's session directory."""
    session_dir = _get_session_dir(model, model_checkpoint_dir)
    session_dir.mkdir(parents=True, exist_ok=True)
    path = session_dir / "optimizer.pth"
    torch.save(
        {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()},
        path,
    )
    LOGGER.info(f"Saved optimizer state to {path}")


def load_optimizer_state(
    optimizer: torch.optim.Optimizer,
    scheduler,
    model_checkpoint_dir: str,
    model: AlphaZeroModel,
) -> bool:
    """Load optimizer and scheduler state from the model's session directory.

    Returns True if state was loaded, False if no saved state exists.
    """
    session_dir = _get_session_dir(model, model_checkpoint_dir)
    path = session_dir / "optimizer.pth"
    if not path.exists():
        LOGGER.info("No saved optimizer state found. Starting fresh.")
        return False

    state = torch.load(path, map_location="cpu")
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    LOGGER.info(f"Loaded optimizer state from {path}")
    return True
