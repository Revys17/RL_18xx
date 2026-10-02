"""train_model must not save a model whose training went non-finite."""
import pytest
import torch

from rl18xx.agent.alphazero.config import ModelTransformerConfig, SelfPlayConfig, TrainingConfig
from rl18xx.agent.alphazero.dataset import SelfPlayDataset
from rl18xx.agent.alphazero.encoder import Encoder_Transformer
from rl18xx.agent.alphazero.mcts import POLICY_SIZE, VALUE_SIZE
from rl18xx.agent.alphazero.model_transformer import AlphaZeroTransformerModel
from rl18xx.agent.alphazero.train import train_model


class _UniformNet:
    """Uniform priors, zero value — enough to play a few self-play moves."""

    encoder = Encoder_Transformer()

    def eval(self):
        return self

    def get_name(self):
        return "dummy"

    def encoder_type(self):
        return "GNN"

    def run_encoded(self, encoded_game_state):
        priors = torch.ones(POLICY_SIZE) / POLICY_SIZE
        return priors, torch.log(priors), torch.zeros(VALUE_SIZE)

    def run_many_encoded(self, encoded_game_states):
        priors = torch.ones(POLICY_SIZE) / POLICY_SIZE
        n = len(encoded_game_states)
        return [priors] * n, [torch.log(priors)] * n, [torch.zeros(VALUE_SIZE)] * n


def test_training_that_goes_non_finite_raises_instead_of_saving(tmp_path, monkeypatch):
    """A forward that overflows makes every batch's loss NaN; each is skipped,
    so without a guard the untrained (fp16-unsafe) weights were saved and
    promoted as a trained checkpoint."""
    from rl18xx.agent.alphazero import self_play

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(self_play, "SELF_PLAY_GAMES_STATUS_PATH", tmp_path / "status")
    (tmp_path / "status").mkdir()
    config = SelfPlayConfig(
        network=_UniformNet(), num_readouts=4, parallel_readouts=2, min_readouts=2, use_score_values=False,
        player_count_distribution={4: 1.0}, enable_resign=False, auction_stall_moves=12, game_id="nan",
    )
    self_play.SelfPlay(config).run_game()
    dataset = SelfPlayDataset(tmp_path / "training_examples" / "selfplay" / "dummy")
    assert len(dataset) > 0

    model = AlphaZeroTransformerModel(ModelTransformerConfig(device=torch.device("cpu")))
    forward = model.forward

    def overflowing_forward(*args, **kwargs):
        policy_logits, *rest = forward(*args, **kwargs)
        return (policy_logits * float("nan"), *rest)

    monkeypatch.setattr(model, "forward", overflowing_forward)
    checkpoints = tmp_path / "checkpoints"
    with pytest.raises(RuntimeError, match="Training diverged"):
        train_model(model, dataset, TrainingConfig(num_epochs=1, batch_size=4), model_checkpoint_dir=str(checkpoints))
    assert not list(checkpoints.rglob("*.pth"))
