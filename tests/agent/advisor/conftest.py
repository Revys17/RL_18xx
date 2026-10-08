"""Shared fixtures for the advisor tests: the generated test game and small CPU models."""

import copy
import json
from pathlib import Path

import pytest
import torch

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "advisor" / "game_1830_4p.json"


@pytest.fixture(scope="session")
def server_game() -> dict:
    """An 1830 game in progress as 18xx.games serves it (scripts/make_advisor_fixture.py:
    our engine and model, no human data), stopped at a corporation's tile lay."""
    return json.loads(FIXTURE.read_text())


@pytest.fixture
def game(server_game) -> dict:
    return copy.deepcopy(server_game)


def truncated(game: dict, count: int) -> dict:
    """The server's copy of ``game`` when it had its first ``count`` actions."""
    return {**copy.deepcopy(game), "actions": copy.deepcopy(game["actions"][:count])}


@pytest.fixture(scope="session")
def advisor_models(tmp_path_factory):
    """Untrained transformer checkpoints (policy, auction policy and value), loaded on CPU."""
    from rl18xx.agent.advisor.advisor import AdvisorModels
    from rl18xx.agent.alphazero import policy_gradient as pg
    from rl18xx.agent.alphazero.config import ModelTransformerConfig
    from rl18xx.agent.alphazero.model_transformer import AlphaZeroTransformerModel

    torch.manual_seed(0)
    directory = tmp_path_factory.mktemp("advisor_ckpts")
    paths = []
    for name in ("policy", "auction", "value"):
        model = AlphaZeroTransformerModel(ModelTransformerConfig(device=torch.device("cpu")))
        path = directory / name / "1.pth"
        pg._save(model, path)
        paths.append(path)
    return AdvisorModels.load(*paths, device="cpu")
