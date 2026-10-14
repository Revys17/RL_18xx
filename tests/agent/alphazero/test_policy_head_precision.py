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


def test_rotation_logits_equal_the_head_on_the_concatenated_inputs():
    """The rotation head projects trunk, hex and tile embeddings separately and adds
    them -- the same function as its first layer on their concatenation, without the
    (B, H, T, d_trunk + d_map + d_tile) input."""
    torch.manual_seed(0)
    model = AlphaZeroTransformerModel(ModelTransformerConfig(device=torch.device("cpu"))).eval()
    head = model.policy_head
    first = head.rotation_head[0]
    d_tile = head.tile_embedding.weight.shape[1]
    d_map = 24
    d_trunk = first.in_features - d_map - d_tile
    B, H, T = 3, 7, head.num_tiles
    trunk, nodes = torch.randn(B, d_trunk, dtype=torch.float64), torch.randn(B, H, d_map, dtype=torch.float64)
    head = head.double()
    concatenated = torch.cat(
        [
            trunk.view(B, 1, 1, -1).expand(-1, H, T, -1),
            nodes.unsqueeze(2).expand(-1, -1, T, -1),
            head.tile_embedding.weight.view(1, 1, T, -1).expand(B, H, -1, -1),
        ],
        dim=-1,
    )
    with torch.no_grad():
        expected = head.rotation_head(concatenated)
        actual = head._rotation_logits(trunk, nodes)
    assert actual.shape == (B, H, T, head.num_rotations)
    assert torch.allclose(actual, expected, atol=1e-10)


def _fixture_positions(limit_each: int = 3):
    """Decisions of a game played by random legal moves (after a legal random auction),
    with and without a legal tile lay."""
    import random

    import numpy as np

    from rl18xx.agent.alphazero.policy_gradient import choose_price
    from rl18xx.agent.alphazero.start_positions import apply_actions, random_start_actions

    model = AlphaZeroTransformerModel(ModelTransformerConfig(device=torch.device("cpu")))
    lo = int(model.policy_head.lay_tile_offset)
    hi = lo + int(model.policy_head.num_lay_tile)
    game = apply_actions(new_game(4), random_start_actions(4, random.Random(0)))
    rng = np.random.default_rng(0)
    with_tile, without = [], []
    for _ in range(2000):
        if game.finished or (len(with_tile) >= limit_each and len(without) >= limit_each):
            break
        legal = [int(i) for i in game._game.factored_legal_indices()]
        if len(legal) > 1:
            bucket = with_tile if any(lo <= i < hi for i in legal) else without
            if len(bucket) < limit_each:
                bucket.append((_rust_encode(game), legal))
        choice = legal[int(rng.integers(len(legal)))]
        price, _, _ = choose_price(game, choice, None, rng, 0.0)
        game._game.apply_action_index(choice, price)
    return with_tile + without


def test_skipping_the_tile_grid_without_a_legal_tile_lay_changes_no_legal_probability_or_gradient():
    torch.manual_seed(0)
    model = AlphaZeroTransformerModel(ModelTransformerConfig(device=torch.device("cpu"))).eval()
    positions = _fixture_positions()
    encoded = [e for e, _ in positions]
    from rl18xx.agent.alphazero.mcts import POLICY_SIZE

    legal = torch.zeros(len(positions), POLICY_SIZE, dtype=torch.bool)
    for b, (_, ix) in enumerate(positions):
        legal[b, ix] = True
    rows = model.lay_tile_rows(legal)
    assert rows.any() and not rows.all()  # both kinds of position in the batch

    def legal_log_probs(skip: bool):
        model.zero_grad(set_to_none=True)
        logits, _, _, _ = model._forward_encoded_batch(encoded, lay_tile_rows=rows if skip else None)
        log_p = torch.log_softmax(logits.masked_fill(~legal[:, : logits.shape[1]], float("-inf")), dim=1)
        picked = torch.stack([log_p[b, ix].sum() for b, (_, ix) in enumerate(positions)])
        picked.sum().backward()
        grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
        return log_p.masked_fill(~legal[:, : logits.shape[1]], 0.0).detach(), grads

    full, full_grads = legal_log_probs(False)
    skipped, skipped_grads = legal_log_probs(True)
    assert torch.allclose(full, skipped, atol=1e-5)
    assert full_grads.keys() == skipped_grads.keys()
    # float32 summation-order noise only, relative to the gradients' overall scale (a
    # parameter whose gradient is ~0 differs by more than its own size from rounding).
    scale = max(float(g.abs().max()) for g in full_grads.values())
    worst = max(float((full_grads[n] - skipped_grads[n]).abs().max()) for n in full_grads)
    assert worst < 1e-5 * scale, (worst, scale)
    # the run_many_encoded entry point derives the same rows from per-state legal indices
    probs_full, _, _ = model.run_many_encoded(encoded)
    probs_skip, _, _ = model.run_many_encoded(encoded, legal_indices=[ix for _, ix in positions])
    for b, (_, ix) in enumerate(positions):
        a, s = probs_full[b, ix], probs_skip[b, ix]
        assert torch.allclose(a / a.sum(), s / s.sum(), atol=1e-6)
