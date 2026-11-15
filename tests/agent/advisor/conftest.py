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
def advisor_checkpoints(tmp_path_factory) -> list:
    """Untrained transformer checkpoints: policy, auction policy and value (paths)."""
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
    return paths


@pytest.fixture(scope="session")
def advisor_models(advisor_checkpoints):
    """The untrained transformer checkpoints (policy, auction policy and value), loaded on CPU."""
    from rl18xx.agent.advisor.advisor import AdvisorModels

    return AdvisorModels.load(*advisor_checkpoints, device="cpu")


# A tiny title-generic model (``generic_pretrain.SMALL_CONFIG``'s sizes).
GENERIC_SMALL = dict(
    d_model=32, num_heads=4, hex_layers=1, entity_layers=1, fusion_layers=1, d_trunk=64, trunk_blocks=1,
    d_head=32, d_edge=8, d_cond=16, d_id=8,
)  # fmt: skip


@pytest.fixture(scope="session")
def generic_checkpoints(tmp_path_factory) -> dict:
    """Untrained tiny generic checkpoints laid out as a generic policy-gradient run (``<run>/learner/1.pth``,
    ``<run>/critic/1.pth``, ``config.json`` with ``learner_temperature`` 0.5): ``{"learner", "critic"}`` paths.
    Zero-initialized output layers get random weights, so the heads aren't uniform."""
    from rl18xx.agent.alphazero import model_generic as mg

    run = tmp_path_factory.mktemp("generic_run")
    (run / "config.json").write_text(json.dumps({"learner_temperature": 0.5, "reward": "placement"}))
    paths = {}
    for seed, name in ((1, "learner"), (2, "critic")):
        torch.manual_seed(seed)
        model = mg.GenericPolicyValueModel(mg.GenericModelConfig(**GENERIC_SMALL)).eval()
        with torch.no_grad():
            for p in model.parameters():
                if p.abs().sum() == 0:
                    p.normal_(0, 0.1)
        path = run / name / "1.pth"
        mg.save_checkpoint(model, path, titles=["1830"])
        paths[name] = path
    return paths
