from rl18xx.agent.alphazero.model import AlphaZeroGNNModel, AlphaZeroModel
from rl18xx.agent.alphazero.config import ModelConfig, ModelV2Config
from pathlib import Path
from datetime import datetime
import json
import logging

LOGGER = logging.getLogger(__name__)


def _load_model_from_config(config_data: dict, checkpoint_path: str) -> AlphaZeroModel:
    """Load the appropriate model type based on config contents."""
    # Detect v2 config by presence of v2-specific fields
    if "d_entity" in config_data or "hex_transformer_layers" in config_data:
        from rl18xx.agent.alphazero.model_v2 import AlphaZeroV2Model

        config = ModelV2Config.from_json(config_data)
        config.model_checkpoint_file = checkpoint_path
        return AlphaZeroV2Model(config)
    else:
        config = ModelConfig.from_json(config_data)
        config.model_checkpoint_file = checkpoint_path
        return AlphaZeroGNNModel(config)


def get_latest_model(model_checkpoint_dir: str) -> AlphaZeroModel:
    # Get the directory within model_checkpoint_dir with the latest timestamp in its name
    p = Path(model_checkpoint_dir)
    if not p.exists() or not p.is_dir():
        raise FileNotFoundError(f"Model checkpoint directory not found: {model_checkpoint_dir}")

    all_checkpoint_directories = [d for d in p.iterdir() if d.is_dir()]
    if not all_checkpoint_directories:
        raise FileNotFoundError(f"No checkpoint directories found in {model_checkpoint_dir}")

    # Sort by directory name, which includes the timestamp
    latest_checkpoint_directory = max(all_checkpoint_directories, key=lambda x: x.name)

    LOGGER.info(f"Loading model from: {latest_checkpoint_directory}")
    config_path = latest_checkpoint_directory / "config.json"
    checkpoint_path = latest_checkpoint_directory / "checkpoint.pth"

    # Handle nested checkpoint directories (e.g. from pretraining)
    if not config_path.exists():
        subdirs = [d for d in latest_checkpoint_directory.iterdir() if d.is_dir()]
        if subdirs:
            nested = max(subdirs, key=lambda x: x.name)
            config_path = nested / "config.json"
            checkpoint_path = nested / "checkpoint.pth"

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found in latest checkpoint directory: {config_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found in latest checkpoint directory: {checkpoint_path}")

    with open(config_path, "r") as f:
        config_data = json.load(f)

    model = _load_model_from_config(config_data, str(checkpoint_path))
    LOGGER.info(f"Loaded most recent model: {model.get_name()}")
    return model


def get_model_from_path(model_checkpoint_path: str) -> AlphaZeroModel:
    p = Path(model_checkpoint_path)
    if not p.exists() or not p.is_dir():
        raise FileNotFoundError(f"Model checkpoint directory not found: {model_checkpoint_path}")

    config_path = p / "config.json"
    checkpoint_path = p / "checkpoint.pth"

    with open(config_path, "r") as f:
        config_data = json.load(f)

    model = _load_model_from_config(config_data, str(checkpoint_path))
    LOGGER.info(f"Loaded model from path {model_checkpoint_path}: {model.get_name()}")
    return model


def save_model(model: AlphaZeroModel, model_checkpoint_dir: str, new=True):
    if new:
        model.config.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create a directory name with a timestamp for easy sorting
    full_directory_path = Path(model_checkpoint_dir) / model.get_name()
    full_directory_path.mkdir(parents=True, exist_ok=True)

    checkpoint_filename = full_directory_path / "checkpoint.pth"
    config_filename = full_directory_path / "config.json"

    model.save_weights(str(checkpoint_filename))

    with open(config_filename, "w") as f:
        json.dump(model.config.to_json(), f, indent=4)
