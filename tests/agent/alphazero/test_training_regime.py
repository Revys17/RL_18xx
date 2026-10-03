"""Self-play training regime: sampled windows, the value generalization check, and stopping."""
import math

import pytest
import torch

from rl18xx.agent.alphazero.config import ModelTransformerConfig, SelfPlayConfig, TrainingConfig
from rl18xx.agent.alphazero.dataset import SelfPlayDataset
from rl18xx.agent.alphazero.mcts import VALUE_SIZE
from rl18xx.agent.alphazero.model_transformer import AlphaZeroTransformerModel
from rl18xx.agent.alphazero.dataset import collate_examples
from rl18xx.agent.alphazero.train import compute_losses, move_batch_to_device, train, value_cross_entropy

from tests.agent.alphazero.test_train_divergence import _UniformNet


@pytest.fixture
def selfplay_lmdb(tmp_path, monkeypatch):
    """A small real self-play LMDB (one short 4-player game)."""
    from rl18xx.agent.alphazero import self_play

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(self_play, "SELF_PLAY_GAMES_STATUS_PATH", tmp_path / "status")
    (tmp_path / "status").mkdir()
    config = SelfPlayConfig(
        network=_UniformNet(), num_readouts=4, parallel_readouts=2, min_readouts=2, use_score_values=True,
        player_count_distribution={4: 1.0}, enable_resign=False, auction_stall_moves=40, game_id="regime",
    )
    self_play.SelfPlay(config).run_game()
    return tmp_path / "training_examples" / "selfplay" / "dummy"


def test_training_samples_a_subset_of_the_window(selfplay_lmdb, tmp_path):
    total = len(SelfPlayDataset(selfplay_lmdb))
    assert total > 20
    model = AlphaZeroTransformerModel(ModelTransformerConfig(device=torch.device("cpu")))
    config = TrainingConfig(
        train_dir=selfplay_lmdb, num_epochs=1, batch_size=8, max_training_window=0, train_samples_per_iteration=16,
    )
    _, metrics = train(config, model, model_checkpoint_dir=str(tmp_path / "checkpoints"))
    assert metrics.training_examples == 16


def test_value_stop_grad_keeps_value_gradients_out_of_the_trunk(selfplay_lmdb, tmp_path):
    """With ``value_stop_grad`` the value loss trains only the value heads, so
    the trunk can't learn features that identify games."""
    model = AlphaZeroTransformerModel(ModelTransformerConfig(device=torch.device("cpu")))
    config = TrainingConfig(
        train_dir=selfplay_lmdb, num_epochs=1, batch_size=8, max_training_window=0, train_samples_per_iteration=8,
        value_stop_grad=True,
    )
    train(config, model, model_checkpoint_dir=str(tmp_path / "checkpoints"))
    assert model.value_stop_grad

    dataset = SelfPlayDataset(selfplay_lmdb)
    game_state, data, legal, pi, value, price_targets = move_batch_to_device(
        collate_examples([dataset[i] for i in range(4)]), torch.device("cpu")
    )
    # A short fixture game can end level (a stalled auction), and a uniform
    # target against an untrained head gives no gradient at all.
    value = torch.zeros_like(value)
    value[:, 0] = 1.0
    model.zero_grad()
    compute_losses(model, game_state, data, legal, pi, value, config, price_targets)["value_loss"].backward()
    grads = {name: param.grad for name, param in model.named_parameters()}
    value_head = [g for name, g in grads.items() if "win_loss_head" in name]
    elsewhere = [g for name, g in grads.items() if "win_loss_head" not in name and "score_head" not in name]
    assert any(g is not None and g.abs().sum() > 0 for g in value_head)
    assert all(g is None or g.abs().sum() == 0 for g in elsewhere)


def test_value_cross_entropy_of_a_uniform_head_is_log_players(selfplay_lmdb):
    """The generalization check reports the training loop's value loss: a
    head that can't tell 4 players apart scores ln 4."""

    class UniformValueModel(torch.nn.Module):
        device = torch.device("cpu")

        def forward(self, game_state_data, batch_data):
            logits = torch.full((len(game_state_data), VALUE_SIZE), -1e4)
            logits[:, :4] = 0.0
            return None, logits, None, None

    dataset = SelfPlayDataset(selfplay_lmdb)
    assert value_cross_entropy(UniformValueModel(), dataset, list(range(10))) == pytest.approx(math.log(4), abs=1e-4)


def test_stop_signal_exits_after_cleaning_up(monkeypatch):
    """SIGTERM/SIGINT used to clean up the children (inference server,
    self-play pool) and then return, so the loop carried on without them."""
    from rl18xx.agent.alphazero import loop

    with pytest.raises(SystemExit) as exc:
        loop.cleanup_and_exit(signum=15)
    assert exc.value.code == 143
    loop.cleanup_and_exit()  # the atexit path just cleans up
