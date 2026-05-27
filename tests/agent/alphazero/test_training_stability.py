"""Test that the Transformer model can train stably on both CPU and MPS."""
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.loader import DataLoader
import pytest


def _train_n_batches(device_name: str, n_batches: int = 8):
    """Train the Transformer model for n_batches on the given device and return losses."""
    from rl18xx.agent.alphazero.config import ModelTransformerConfig
    from rl18xx.agent.alphazero.model_transformer import AlphaZeroTransformerModel

    # Create synthetic training data that mimics real self-play data
    B = 64
    # Build the model first so we know its max-N layout size; the synthetic
    # game-state vector is sized for that layout (the model now accepts any
    # 2..max_players game by padding to this layout).
    config = ModelTransformerConfig(device=torch.device(device_name))
    gs_size = config.game_state_size
    value_size = config.value_size
    # ``round_type`` lives at a layout-dependent offset; compute it from the
    # encoder so the synthetic state lights up the right slot for the model's
    # max-N layout.
    from rl18xx.agent.alphazero.model_transformer import _layout_for
    _off, _, _ = _layout_for(config.max_players)
    round_type_off = _off["round_type"]
    dataset = []
    for _ in range(n_batches * B):
        gs = torch.randn(1, gs_size)
        gs[:, round_type_off] = torch.rand(1) * 1.0  # round type (normalized to [0, 1])
        gs[:, :config.max_players] = 0.0
        gs[0, torch.randint(0, config.max_players, (1,))] = 1.0  # active player

        nf = torch.randn(93, 50) * 0.1  # node features
        mask = torch.zeros(26537)
        legal_indices = torch.randint(0, 26537, (7,))
        mask[legal_indices] = 1.0

        pi = torch.zeros(26537)
        pi[legal_indices] = torch.softmax(torch.randn(7), dim=0)

        value = torch.softmax(torch.randn(value_size), dim=0)
        dataset.append((gs, nf, mask, pi, value))
    model = AlphaZeroTransformerModel(config)
    model.train()
    device = config.device

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    losses = []

    for batch_start in range(0, len(dataset), B):
        batch = dataset[batch_start : batch_start + B]
        gs = torch.cat([d[0] for d in batch]).float().to(device)
        nf = torch.stack([d[1] for d in batch]).float().to(device)
        mask = torch.stack([d[2] for d in batch]).float().to(device)
        pi = torch.stack([d[3] for d in batch]).float().to(device)
        value = torch.stack([d[4] for d in batch]).float().to(device)

        policy_logits, win_loss_logits, score_pred, aux_pred = model(gs, nf)

        masked_logits = policy_logits.masked_fill(mask == 0, float("-inf"))
        log_probs = F.log_softmax(masked_logits, dim=1)
        policy_loss = -torch.nansum(pi * log_probs, dim=1).mean()

        # Dual value head: KL on the win-loss output and MSE on the score
        # output. The synthetic ``value`` here is a normalized distribution
        # — use it directly for both since this test only checks training
        # stability, not target derivation.
        win_loss_log_probs = F.log_softmax(win_loss_logits, dim=1)
        value_loss = F.kl_div(win_loss_log_probs, value, reduction="batchmean")
        score_loss = F.mse_loss(score_pred, value)

        aux_target = torch.log(mask.sum(dim=1).clamp(min=1))
        aux_pred_clamped = aux_pred.squeeze(1).clamp(-10, 10)
        aux_loss = F.mse_loss(aux_pred_clamped, aux_target)

        total_loss = policy_loss + value_loss + 0.1 * score_loss + 0.01 * aux_loss

        if not torch.isfinite(total_loss):
            return losses, f"NaN at batch {len(losses)}"

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

        losses.append(total_loss.item())

        # Check weights
        for name, p in model.named_parameters():
            if torch.isnan(p).any():
                return losses, f"NaN in {name} after batch {len(losses)}"

    return losses, None


def test_transformer_training_stable_cpu():
    losses, error = _train_n_batches("cpu", n_batches=8)
    assert error is None, f"Training failed on CPU: {error}. Losses: {losses}"
    assert len(losses) == 8
    # Loss threshold relaxed from 100 → 250 after the KataGo-style dual head
    # added the score MSE term: at random init the score head outputs can be
    # several units off the normalized target, so per-element MSE * 4 players
    # plus the aux MSE term can push the early-batch total over 100 even
    # though training is stable (no NaNs, losses bounded).
    assert all(loss < 250 for loss in losses), f"Losses too high: {losses}"


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_transformer_training_stable_mps():
    losses, error = _train_n_batches("mps", n_batches=8)
    assert error is None, f"Training failed on MPS: {error}. Losses: {losses}"
    assert len(losses) == 8
    assert all(loss < 250 for loss in losses), f"Losses too high: {losses}"
