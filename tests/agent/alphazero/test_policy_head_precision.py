"""The policy and price heads stay fp32 under bf16 autocast.

Checkpoint 10's LayTile hex logits sit near -945 with ~+-22 between hexes;
bf16 spaces values 4 apart there, so a bf16 head collapsed P(hex) (the top
tile lay differed from fp32 in ~40% of positions with one).
"""
import torch

from rl18xx.agent.alphazero.config import ModelTransformerConfig
from rl18xx.agent.alphazero.mcts import _rust_encode
from rl18xx.agent.alphazero.model_transformer import AlphaZeroTransformerModel
from rl18xx.agent.alphazero.start_positions import new_game


def test_hex_distribution_survives_bf16_autocast():
    torch.manual_seed(0)
    model = AlphaZeroTransformerModel(ModelTransformerConfig(device=torch.device("cpu"))).eval()
    final = model.policy_head.hex_scorer[2]
    with torch.no_grad():
        final.weight.mul_(0.25)  # a few nats between hexes, as trained heads have
        final.bias.fill_(-945.0)  # the large offset softmax ignores
    encoded = [_rust_encode(new_game(4))] * 2

    def log_p_hex(autocast: bool) -> torch.Tensor:
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
            model._forward_encoded_batch(encoded)
        return model.last_policy_components["log_p_hex"].float()

    reference = log_p_hex(False)
    assert reference.std(dim=-1).min() > 1.0  # a real spread, not a flat distribution
    assert (log_p_hex(True) - reference).abs().max() < 0.5
    assert model.last_price_components["price_logits"].dtype == torch.float32
