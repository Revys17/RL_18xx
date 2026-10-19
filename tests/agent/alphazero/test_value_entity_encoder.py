"""``ModelTransformerConfig.value_entity_encoder``: the value heads read their player
tokens from a value-only entity encoder; the policy path is untouched, and
``scripts/build_value_encoder_network.py`` combines a policy with a critic's encoder."""

import importlib.util
from pathlib import Path

import pytest
import torch

engine_rs = pytest.importorskip("engine_rs")

from rl18xx.agent.alphazero.mcts import _rust_encode  # noqa: E402
from rl18xx.rust_adapter import RustGameAdapter  # noqa: E402

from .search_test_utils import random_positions  # noqa: E402
from .test_search_learning import _tiny_model  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]


def _on_cpu(model):
    """``model`` on the CPU, its ``device`` too (the forward moves its inputs to ``model.device``:
    a loaded checkpoint says cuda where a GPU is visible)."""
    model = model.to("cpu")
    model.device = torch.device("cpu")
    return model


def _random_value_heads(model, seed):
    g = torch.Generator().manual_seed(seed)
    for head in (model.win_loss_head, model.score_head):
        with torch.no_grad():
            head[-1].weight.copy_(torch.randn(head[-1].weight.shape, generator=g) * 0.5)
    return model


def _states(n=3):
    return [_rust_encode(RustGameAdapter(g)) for g in random_positions(4, 3, steps=150, every=31)[:n]]


def _build_module():
    path = REPO_ROOT / "scripts" / "build_value_encoder_network.py"
    spec = importlib.util.spec_from_file_location("build_value_encoder_network", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_plain_model_has_no_value_encoder():
    assert _tiny_model(0).value_econ_transformer is None


def test_value_stop_grad_keeps_the_shared_layers_out_of_the_value_loss_but_trains_the_value_encoder(tmp_path):
    from rl18xx.agent.alphazero.policy_gradient import _save

    _save(_tiny_model(0), tmp_path / "p" / "learner" / "1.pth")
    _save(_random_value_heads(_tiny_model(1), 5), tmp_path / "c" / "critic" / "1.pth")
    model = _build_module().build(str(tmp_path / "p" / "learner" / "1.pth"), str(tmp_path / "c" / "critic" / "1.pth"))
    model = _on_cpu(model)
    model.value_stop_grad = True
    _, win, _, _ = model._forward_encoded_batch(_states(), value_only=True)
    win.float().pow(2).sum().backward()
    assert all(p.grad is None or not p.grad.any() for p in model.econ_transformer.parameters())
    assert all(p.grad is None or not p.grad.any() for p in model.res_blocks.parameters())
    assert any(p.grad is not None and p.grad.any() for p in model.value_econ_transformer.parameters())


def test_shared_search_learning_trains_the_value_encoder_at_critic_lr(tmp_path):
    from rl18xx.agent.alphazero import search_learning as sl
    from rl18xx.agent.alphazero.policy_gradient import _save

    _save(_tiny_model(0), tmp_path / "p" / "learner" / "1.pth")
    _save(_tiny_model(1), tmp_path / "c" / "critic" / "1.pth")
    model = _build_module().build(str(tmp_path / "p" / "learner" / "1.pth"), str(tmp_path / "c" / "critic" / "1.pth"))
    cfg = sl.SearchLearningConfig(policy_checkpoint="p", value_checkpoint="", shared=True, lr=1e-5, critic_lr=3e-5)
    groups = sl.learner_param_groups(model, cfg)
    value = {id(p) for h in (model.win_loss_head, model.score_head, model.value_econ_transformer) for p in h.parameters()}
    assert {id(p) for p in groups[1]["params"]} == value and groups[1]["lr"] == 3e-5
    assert len(groups[0]["params"]) + len(groups[1]["params"]) == len(list(model.parameters()))


def test_the_value_encoder_feeds_the_value_heads_only_and_survives_a_checkpoint(tmp_path):
    from rl18xx.agent.alphazero.policy_gradient import _load, _save

    policy = _random_value_heads(_tiny_model(0), 3).eval()
    critic = _random_value_heads(_tiny_model(1), 4).eval()
    _save(policy, tmp_path / "p" / "learner" / "1.pth")
    _save(critic, tmp_path / "c" / "critic" / "1.pth")
    model = _build_module().build(str(tmp_path / "p" / "learner" / "1.pth"), str(tmp_path / "c" / "critic" / "1.pth"))
    model = _on_cpu(model).eval()
    assert model.config.value_entity_encoder and model.value_econ_transformer is not None
    states = _states()
    with torch.no_grad():
        p_logits, p_win, _, _ = policy._forward_encoded_batch(states)
        p_price = policy.last_price_components["price_logits"]
        m_logits, m_win, _, _ = model._forward_encoded_batch(states)
        m_price = model.last_price_components["price_logits"]
    # The policy path is the policy network's, bit for bit; the value is not its value.
    assert torch.equal(p_logits, m_logits) and torch.equal(p_price, m_price)
    assert not torch.allclose(p_win, m_win)

    # The value heads read the value encoder's player tokens (plus the shared trunk).
    store, hook = model.capture_value_input()
    with torch.no_grad():
        model._forward_encoded_batch(states, value_only=True)
    hook.remove()
    d = model.config.d_entity
    gs = store["seat_input"]
    with torch.no_grad():
        game_states = torch.stack([torch.as_tensor(s[0]).reshape(-1).float() for s in states])
        _, value_tokens, _ = model.value_econ_transformer(
            _pad(game_states), num_players=torch.full((len(states),), 4, dtype=torch.long)
        )
    econ = model.value_econ_transformer
    assert torch.allclose(gs[:, :, :d], value_tokens[:, econ.player_token_start : econ.player_token_end], atol=1e-6)

    # Changing the value encoder changes the value, never the policy.
    with torch.no_grad():
        for p in model.value_econ_transformer.parameters():
            p.add_(0.1 * torch.randn_like(p))
        logits2, win2, _, _ = model._forward_encoded_batch(states)
    assert torch.equal(logits2, m_logits) and not torch.allclose(win2, m_win)

    _save(model, tmp_path / "v" / "learner" / "1.pth")
    loaded = _on_cpu(_load(tmp_path / "v" / "learner" / "1.pth")).eval()
    assert loaded.config.value_entity_encoder
    with torch.no_grad():
        l_logits, l_win, _, _ = loaded._forward_encoded_batch(states)
    assert torch.equal(l_logits, logits2) and torch.allclose(l_win, win2)


def _pad(game_states):
    from rl18xx.agent.alphazero.model_transformer import MAX_PLAYERS, _pad_state_to_max_players

    return _pad_state_to_max_players(game_states, 4, MAX_PLAYERS)
