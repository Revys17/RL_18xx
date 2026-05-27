import math
import numpy as np
from rl18xx.agent.alphazero.model import AlphaZeroModel
from rl18xx.agent.alphazero.config import TrainingConfig
from rl18xx.agent.alphazero.dataset import SelfPlayDataset
from rl18xx.agent.alphazero.checkpointer import get_latest_model, save_model, save_optimizer_state, load_optimizer_state

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from tqdm.auto import tqdm
import logging
from dataclasses import dataclass, field
from typing import Tuple
from IPython.display import display, clear_output, HTML
import matplotlib.pyplot as plt
import warnings

plt.style.use("default")
warnings.filterwarnings("ignore", message=".*can only test a child process.*")

LOGGER = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """Metrics collected during training"""

    avg_total_loss: float = 0.0
    avg_policy_loss: float = 0.0
    avg_value_loss: float = 0.0
    training_examples: int = 0
    epochs_trained: int = 0
    # Checkpoint number that train_model wrote out (None if training was skipped,
    # e.g. empty dataset). Set by train_model after the final save_model call.
    checkpoint_num: int = None
    epoch_losses: list = field(default_factory=list)
    epoch_policy_losses: list = field(default_factory=list)
    epoch_value_losses: list = field(default_factory=list)
    batch_losses: list = field(default_factory=list)
    batch_policy_losses: list = field(default_factory=list)
    batch_value_losses: list = field(default_factory=list)
    batch_numbers: list = field(default_factory=list)

    # --- Comprehensive metrics (per-epoch averages) ---
    # Loss components
    epoch_entropy: list = field(default_factory=list)
    epoch_aux_losses: list = field(default_factory=list)

    # Policy diagnostics
    epoch_policy_kl: list = field(default_factory=list)  # KL(MCTS target || network output)
    epoch_top1_accuracy: list = field(default_factory=list)  # fraction where argmax(pi) == argmax(logits)
    epoch_top5_accuracy: list = field(default_factory=list)
    epoch_policy_entropy: list = field(default_factory=list)  # entropy of network policy
    epoch_target_entropy: list = field(default_factory=list)  # entropy of MCTS target policy
    epoch_legal_move_concentration: list = field(default_factory=list)  # max policy prob among legal moves
    epoch_mean_legal_actions: list = field(default_factory=list)  # avg number of legal actions

    # Value diagnostics
    epoch_value_explained_variance: list = field(default_factory=list)
    epoch_value_pred_mean: list = field(default_factory=list)
    epoch_value_pred_std: list = field(default_factory=list)
    epoch_value_target_mean: list = field(default_factory=list)
    epoch_value_target_std: list = field(default_factory=list)
    epoch_value_mae: list = field(default_factory=list)  # mean absolute error
    epoch_value_mse: list = field(default_factory=list)  # mean squared error
    epoch_value_correlation: list = field(default_factory=list)  # correlation between pred and target
    epoch_value_pred_min: list = field(default_factory=list)
    epoch_value_pred_max: list = field(default_factory=list)
    epoch_value_target_min: list = field(default_factory=list)
    epoch_value_target_max: list = field(default_factory=list)

    # Gradient diagnostics
    epoch_grad_norm_total: list = field(default_factory=list)
    epoch_grad_norm_policy_head: list = field(default_factory=list)
    epoch_grad_norm_value_head: list = field(default_factory=list)
    epoch_grad_norm_trunk: list = field(default_factory=list)
    epoch_grad_norm_cv: list = field(default_factory=list)  # coefficient of variation of total grad norm

    # Learning rate
    epoch_lr: list = field(default_factory=list)

    # Aux head diagnostics
    epoch_aux_pred_mean: list = field(default_factory=list)
    epoch_aux_target_mean: list = field(default_factory=list)
    epoch_aux_correlation: list = field(default_factory=list)


def _compute_grad_norms(model: AlphaZeroModel) -> dict:
    """Compute gradient norms for different model components.

    The dual value head splits the legacy ``value_head`` into ``win_loss_head``
    + ``score_head``; both contribute to the aggregate ``value_head`` bucket so
    metrics stay comparable with pre-dual-head runs.
    """
    norms = {"total": 0.0, "policy_head": 0.0, "value_head": 0.0, "trunk": 0.0}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.data.norm(2).item() ** 2
            norms["total"] += grad_norm
            if "policy" in name:
                norms["policy_head"] += grad_norm
            elif "win_loss_head" in name or "score_head" in name:
                norms["value_head"] += grad_norm
            else:
                norms["trunk"] += grad_norm
    norms = {k: v**0.5 for k, v in norms.items()}
    return norms


def _derive_dual_value_targets(value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Derive ``(win_loss_target, score_target)`` from the stored per-example value.

    The stored ``value`` tensor on each training example can be in one of two
    legacy formats depending on ``SelfPlayConfig.use_score_values``:

    1. **Score fractions** (``use_score_values=True``, the default): all entries
       are non-negative and sum to 1; each entry is that player's share of
       total end-of-game net worth. This is already the natural target for the
       score head, and the win-loss target is the share-of-winners
       distribution derived from it (one mass on the argmax player, split
       evenly across ties).

    2. **Win/loss vector** (``use_score_values=False``, legacy): entries are
       in {-1, 0, +1}. We synthesize both heads' targets from it: the win-loss
       head gets the standard share-of-winners distribution and the score head
       falls back to the same target (cheap backward-compat — no real signal,
       but keeps the loss finite without forcing a data regeneration).

    Detection mirrors the auto-detect already in ``compute_losses``: a value
    tensor that is everywhere ``>= 0`` is treated as score fractions.
    """
    is_score_values = (value >= 0).all()
    if is_score_values:
        score_target = value
        # Share-of-winners derived from the score fractions. We threshold by
        # the per-row max instead of comparing for equality so ties (which
        # land on identical floats here because the encoder normalizes by
        # sum) get even mass.
        row_max = value.max(dim=1, keepdim=True).values
        winners_mask = (value >= row_max - 1e-6).float()
        num_winners = winners_mask.sum(dim=1, keepdim=True).clamp(min=1)
        win_loss_target = winners_mask / num_winners
    else:
        winners_mask = (value > -0.5).float()
        num_winners = winners_mask.sum(dim=1, keepdim=True).clamp(min=1)
        win_loss_target = winners_mask / num_winners
        # No independent score signal in this legacy encoding; use the win/loss
        # distribution so the score head sees a finite, sensible target.
        score_target = win_loss_target
    return win_loss_target, score_target


def _safe_correlation(x: torch.Tensor, y: torch.Tensor) -> float:
    """Compute Pearson correlation, returning 0 if degenerate."""
    if x.numel() < 2:
        return 0.0
    x_flat = x.flatten().float()
    y_flat = y.flatten().float()
    vx = x_flat - x_flat.mean()
    vy = y_flat - y_flat.mean()
    denom = (vx.norm() * vy.norm())
    if denom < 1e-8:
        return 0.0
    return (vx @ vy / denom).item()


def _compute_decomposed_policy_loss(
    components: dict,
    pi: torch.Tensor,
    legal_action_mask: torch.Tensor,
) -> Tuple[torch.Tensor, dict]:
    """Decomposed cross-entropy loss for the hierarchical policy head.

    For each autoregressive level (LayTile: hex / tile|hex / rot|hex,tile;
    PlaceToken: hex / slot|hex) we project the flat MCTS visit-count target
    ``pi`` onto that level's marginal and compute a standard cross-entropy
    against the sub-head's ``log_softmax``. Each sub-head's gradient is
    automatically weighted by the visit mass that passed through its level
    of the joint (this falls out of the marginalization — no manual scaling
    needed).

    For the parallel-factored "other" block (everything outside LayTile +
    PlaceToken), we do a standard masked CE on the compact ``log_p_other``
    head, with the legal-action mask routed through the same
    ``other_indices`` gather.

    Returns ``(total_policy_loss, diagnostics)`` where ``diagnostics`` carries
    the per-sub-head loss tensors (useful for TensorBoard).
    """
    # ---- Layout ----
    H = components["num_hexes"]
    T = components["num_tiles"]
    R = components["num_rotations"]
    S = components["max_city_slots"]
    lt_start = components["lay_tile_offset"]
    lt_end = components["lay_tile_end"]
    pt_start = components["place_token_offset"]
    pt_end = components["place_token_end"]
    other_indices = components["other_indices"]
    pt_to_hex = components["place_token_to_mapper_hex"]
    pt_to_slot = components["place_token_to_slot"]

    B = pi.shape[0]
    eps = 1e-8

    # ---- LayTile decomposition ----
    pi_lay = pi[:, lt_start:lt_end].view(B, H, T, R)  # (B, H, T, R)
    pi_hex_marginal = pi_lay.sum(dim=(2, 3))  # (B, H)
    pi_tile_marginal = pi_lay.sum(dim=3)  # (B, H, T)  joint over (hex, tile)

    log_p_hex = F.log_softmax(components["hex_logits"], dim=-1)  # (B, H)
    log_p_tile = F.log_softmax(components["tile_logits"], dim=-1)  # (B, H, T)
    log_p_rot = F.log_softmax(components["rotation_logits"], dim=-1)  # (B, H, T, R)

    # Three CE losses, one per autoregressive level. Each level's loss is
    # naturally weighted by the visit mass that flows through it (the
    # marginal-target shape encodes the weighting).
    loss_hex = -(pi_hex_marginal * log_p_hex).sum(dim=-1).mean()
    loss_tile = -(pi_tile_marginal * log_p_tile).sum(dim=(-1, -2)).mean()
    loss_rot = -(pi_lay * log_p_rot).sum(dim=(-1, -2, -3)).mean()

    # ---- PlaceToken decomposition ----
    pi_pt_flat = pi[:, pt_start:pt_end]  # (B, place_token_block_size)
    # Scatter the flat PlaceToken target into the (H, S) grid using the
    # precomputed (hex, slot) lookup so the per-hex / per-slot losses index
    # the same physical layout the sub-heads emit.
    pi_pt_grid = pi.new_zeros(B, H, S)
    pi_pt_grid[:, pt_to_hex, pt_to_slot] = pi_pt_flat
    pi_pt_hex_marginal = pi_pt_grid.sum(dim=-1)  # (B, H)

    log_p_pt_hex = F.log_softmax(components["place_token_hex_logits"], dim=-1)  # (B, H)
    log_p_pt_slot = F.log_softmax(components["place_token_slot_logits"], dim=-1)  # (B, H, S)

    loss_pt_hex = -(pi_pt_hex_marginal * log_p_pt_hex).sum(dim=-1).mean()
    loss_pt_slot = -(pi_pt_grid * log_p_pt_slot).sum(dim=(-1, -2)).mean()

    # ---- Other-action CE ----
    pi_other = pi[:, other_indices]  # (B, num_other)
    other_logits = components["other_logits"]  # (B, num_other)
    legal_other_mask = legal_action_mask[:, other_indices]  # (B, num_other)
    masked_other_logits = other_logits.masked_fill(legal_other_mask == 0, float("-inf"))
    log_p_other = F.log_softmax(masked_other_logits, dim=-1)
    safe_log_p_other = log_p_other.masked_fill(legal_other_mask == 0, 0.0)
    # If a sample has no legal "other" slots at all, ``masked_other_logits`` is
    # entirely -inf and ``log_softmax`` returns NaNs. Such samples also have
    # ``pi_other == 0`` along that row, so we zero out the safe-log to avoid
    # 0 * NaN.
    no_legal_other = (legal_other_mask.sum(dim=-1) == 0).unsqueeze(-1)
    safe_log_p_other = torch.where(no_legal_other, torch.zeros_like(safe_log_p_other), safe_log_p_other)
    loss_other = -(pi_other * safe_log_p_other).sum(dim=-1).mean()

    total = loss_hex + loss_tile + loss_rot + loss_pt_hex + loss_pt_slot + loss_other
    diagnostics = {
        "loss_lay_tile_hex": loss_hex,
        "loss_lay_tile_tile": loss_tile,
        "loss_lay_tile_rot": loss_rot,
        "loss_place_token_hex": loss_pt_hex,
        "loss_place_token_slot": loss_pt_slot,
        "loss_other": loss_other,
    }
    return total, diagnostics


def _compute_price_nll_loss(
    price_components: dict,
    price_targets: list,
) -> Tuple[torch.Tensor, dict]:
    """Continuous-price NLL on the network's ``Normal(mean, exp(log_std))`` head.

    ``price_targets`` is a per-example list of ``[(slot_idx, price, weight,
    price_min, price_max), ...]`` tuples. Each tuple says "this example had a
    target observation of ``price`` for the (action_type, entity) corresponding
    to ``slot_idx``, contributing ``weight`` mass to the loss; the legal price
    range is ``[price_min, price_max]``." Weight is typically 1.0 for
    pretraining (one observed action per state) and visit-fraction for
    self-play (multiple sampled prices per categorical child node).

    The truncated-Normal correction (subtracting ``log(Φ((max-μ)/σ) -
    Φ((min-μ)/σ))``) is implemented so the head's reported NLL reflects the
    actual truncated distribution MCTS samples from. Skipping it would
    over-penalize ``μ`` predictions outside the legal range — the gradient
    direction is the same but the magnitudes drift.

    Returns ``(price_loss, diagnostics)``. ``price_loss`` is the weighted
    average NLL across all (example, target) pairs; if there are no price
    targets in the batch it is a zero tensor on the same device as the head.
    ``diagnostics`` carries the per-action-type breakdown for TensorBoard.
    """
    mean = price_components["price_mean"]  # (B, num_slots)
    log_std = price_components["price_log_std"]  # (B, num_slots)
    device = mean.device
    dtype = mean.dtype

    # Flatten the per-example target lists into batched index tensors so the
    # NLL computation is a single vectorized operation.
    batch_idx: list = []
    slot_idx: list = []
    prices: list = []
    weights: list = []
    p_min: list = []
    p_max: list = []
    for b, targets in enumerate(price_targets or []):
        if not targets:
            continue
        for entry in targets:
            slot, price, w, pmn, pmx = entry
            batch_idx.append(b)
            slot_idx.append(slot)
            prices.append(float(price))
            weights.append(float(w))
            p_min.append(float(pmn))
            p_max.append(float(pmx))

    if not slot_idx:
        zero = torch.tensor(0.0, device=device, dtype=dtype)
        return zero, {"price_count": 0}

    bi = torch.tensor(batch_idx, device=device, dtype=torch.long)
    si = torch.tensor(slot_idx, device=device, dtype=torch.long)
    px = torch.tensor(prices, device=device, dtype=dtype)
    wt = torch.tensor(weights, device=device, dtype=dtype)
    lo = torch.tensor(p_min, device=device, dtype=dtype)
    hi = torch.tensor(p_max, device=device, dtype=dtype)

    mu = mean[bi, si]
    log_sigma = log_std[bi, si]
    # Numerical-safety clamp on log_std so the gradient stays finite even if
    # the head emits an extreme prediction; matches the clamp MCTS uses when
    # sampling. The bounds are roughly ``[$0.5, $5000]`` in raw price space.
    log_sigma = log_sigma.clamp(min=-1.0, max=8.5)
    sigma = log_sigma.exp()

    # Untruncated Normal log-pdf:  -0.5*((x-μ)/σ)² - log(σ) - 0.5*log(2π)
    log2pi = math.log(2.0 * math.pi)
    z = (px - mu) / sigma
    untruncated_log_prob = -0.5 * z * z - log_sigma - 0.5 * log2pi

    # Truncation correction: subtract log(Φ((hi-μ)/σ) - Φ((lo-μ)/σ)).
    # When ``hi == lo`` (fixed-price actions like depot trains shouldn't reach
    # this head, but we defend against it) treat the correction as 0 so the
    # NLL collapses to the untruncated log-pdf.
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    norm_cdf_hi = 0.5 * (1.0 + torch.erf((hi - mu) * inv_sqrt2 / sigma))
    norm_cdf_lo = 0.5 * (1.0 + torch.erf((lo - mu) * inv_sqrt2 / sigma))
    truncation_mass = (norm_cdf_hi - norm_cdf_lo).clamp(min=1e-8)
    fixed_price = (hi - lo).abs() < 1e-6
    log_trunc_correction = torch.where(
        fixed_price,
        torch.zeros_like(truncation_mass),
        torch.log(truncation_mass),
    )

    log_prob = untruncated_log_prob - log_trunc_correction
    nll = -log_prob  # (num_targets,)

    weighted = nll * wt
    total_weight = wt.sum().clamp(min=1e-8)
    price_loss = weighted.sum() / total_weight

    diagnostics = {
        "price_count": int(si.numel()),
        "price_nll_mean": price_loss.detach(),
        "price_mu_mean": mu.detach().mean(),
        "price_log_std_mean": log_sigma.detach().mean(),
    }
    return price_loss, diagnostics


def compute_losses(
    model: AlphaZeroModel,
    game_state_data: torch.Tensor,
    batch_data,
    legal_action_mask: torch.Tensor,
    pi: torch.Tensor,
    value: torch.Tensor,
    config: TrainingConfig,
    price_targets: list = None,
) -> dict:
    """Forward pass + loss computation shared between RL training and SL pretraining.

    KataGo-style dual value head: the network emits two value outputs from
    parallel heads on the same trunk — ``win_loss_logits`` (share-of-winners,
    KL-div trained) and ``score_pred`` (normalized net-worth fractions,
    MSE-trained). MCTS reads only the win-loss head; the score head provides
    a dense gradient signal to the trunk. Both targets are derived from the
    stored per-example ``value`` via ``_derive_dual_value_targets`` (which
    transparently handles the legacy win/loss-only encoding).

    Policy loss is computed via the autoregressive decomposition when the
    model exposes ``last_policy_components`` (the new HierarchicalPolicyHead);
    otherwise it falls back to the legacy masked-CE on the flat policy
    logits. This lets the GNN model — which still uses the outer-sum
    FactoredPolicyHead — share this helper without rewiring its loss path.

    Returns a dict containing:
        - "total_loss", "policy_loss", "value_loss", "score_loss", "aux_loss", "entropy"
        - "policy_logits", "policy_probs", "log_probs", "safe_log_probs"
        - "value_pred" (== ``win_loss_logits``, kept for backward compat),
          "win_loss_logits", "score_pred"
        - "win_loss_target", "score_target"
        - "aux_pred", "aux_target"
        - "policy_loss_breakdown" (dict of per-sub-head losses, only present
          when the hierarchical head is in use)
    The caller is responsible for backward, gradient clipping, and optimizer
    steps. Computing-side concerns (KL formulation, masking, FiLM, aux head)
    are intentionally identical to keep RL/SL gradient flow comparable; the
    only difference between train_model and pretrain_model is in optimizer
    construction, scheduler, validation, and checkpointing.
    """
    policy_logits, win_loss_logits, score_pred, aux_action_count_pred = model(game_state_data, batch_data)

    # --- Policy loss ---
    # Prefer the decomposed loss when the model exposes the hierarchical
    # policy components (current Transformer architecture). Fall back to the
    # legacy masked-CE on the flat policy logits for the GNN model and any
    # checkpoints that pre-date the hierarchical head.
    policy_components = getattr(model, "last_policy_components", None)
    masked_logits = policy_logits.masked_fill(legal_action_mask == 0, float("-inf"))
    log_probs = F.log_softmax(masked_logits, dim=1)
    safe_log_probs = log_probs.masked_fill(legal_action_mask == 0, 0.0)
    if policy_components is not None:
        policy_loss, policy_loss_breakdown = _compute_decomposed_policy_loss(
            policy_components, pi, legal_action_mask
        )
    else:
        policy_loss = -torch.sum(pi * safe_log_probs, dim=1).mean()
        policy_loss_breakdown = None

    # --- Policy entropy (network-output entropy over legal actions; same
    # quantity regardless of which loss formulation we used).
    policy_probs = F.softmax(masked_logits, dim=1)
    entropy = -torch.sum(policy_probs * safe_log_probs, dim=1).mean()

    # --- Dual value targets derived from the stored value tensor ---
    win_loss_target, score_target = _derive_dual_value_targets(value)

    # Variable-N: the model always outputs (B, max_players=6); the stored target
    # has dim=num_players for the source game. Pad target with zeros on the
    # right so the loss can broadcast — padded slots correspond to non-existent
    # players (zero win-share, zero net-worth fraction), which is semantically
    # correct.
    max_n = win_loss_logits.shape[1]
    if win_loss_target.shape[1] < max_n:
        pad = max_n - win_loss_target.shape[1]
        win_loss_target = F.pad(win_loss_target, (0, pad), value=0.0)
        score_target = F.pad(score_target, (0, pad), value=0.0)

    # --- Win-loss loss (KL-div on softmax distribution) ---
    win_loss_log_probs = F.log_softmax(win_loss_logits, dim=1)
    win_loss_loss = -(win_loss_target * win_loss_log_probs).sum(dim=1).mean()

    # --- Score loss (MSE on raw per-player predictions) ---
    score_loss = F.mse_loss(score_pred, score_target)

    # --- Auxiliary loss ---
    legal_action_count = legal_action_mask.sum(dim=1)
    aux_target = torch.log(legal_action_count.float().clamp(min=1))
    aux_pred = aux_action_count_pred.squeeze(1)
    aux_loss = F.mse_loss(aux_pred, aux_target)

    # --- Continuous price NLL ---
    # The price head is consumed only when the model exposes
    # ``last_price_components`` (current Transformer architecture) AND the
    # caller has supplied per-example price targets. Both conditions are
    # currently absent for the legacy GNN training path and for self-play /
    # pretraining batches that haven't been migrated to carry price targets
    # yet — in those cases the price loss is exactly zero and contributes
    # neither gradient nor noise to the total loss.
    price_components = getattr(model, "last_price_components", None)
    price_loss_value = torch.tensor(0.0, device=policy_loss.device, dtype=policy_loss.dtype)
    price_diagnostics = {"price_count": 0}
    if price_components is not None and price_targets is not None:
        price_loss_value, price_diagnostics = _compute_price_nll_loss(
            price_components, price_targets
        )

    total_loss = (
        policy_loss
        + config.value_loss_weight * win_loss_loss
        + config.score_loss_weight * score_loss
        + config.price_loss_weight * price_loss_value
        + model.config.aux_loss_weight * aux_loss
        - config.entropy_weight * entropy
    )

    return {
        "total_loss": total_loss,
        "policy_loss": policy_loss,
        # "value_loss" retains its historical meaning (the loss consumed by
        # MCTS-relevant supervision) so the metrics/logging path stays stable.
        "value_loss": win_loss_loss,
        "win_loss_loss": win_loss_loss,
        "score_loss": score_loss,
        "aux_loss": aux_loss,
        "entropy": entropy,
        "policy_logits": policy_logits,
        "policy_probs": policy_probs,
        "log_probs": log_probs,
        "safe_log_probs": safe_log_probs,
        # Keep "value_pred" pointing at the win-loss logits for callers that
        # historically used it for diagnostics (e.g. value-MSE-vs-target).
        "value_pred": win_loss_logits,
        "win_loss_logits": win_loss_logits,
        "score_pred": score_pred,
        "win_loss_target": win_loss_target,
        "score_target": score_target,
        "aux_pred": aux_pred,
        "aux_target": aux_target,
        "legal_action_count": legal_action_count,
        "policy_loss_breakdown": policy_loss_breakdown,
        "price_loss": price_loss_value,
        "price_diagnostics": price_diagnostics,
    }


def move_batch_to_device(batch, device):
    """Shared batch unpacking + device transfer used by both train and pretrain loops.

    The dataset emits a 6-tuple ``(game_state_data, batch_data,
    legal_action_mask, pi, value, price_targets)``. ``price_targets`` is a
    per-example list (one entry per item in the batch); each entry is either
    ``None`` or a list of ``(slot_idx, price, weight, price_min, price_max)``
    tuples — kept on CPU as Python objects because ``_compute_price_nll_loss``
    rebuilds the dense index/value tensors inside the loss helper. Legacy
    5-tuple batches (no price-target column) are tolerated and surfaced as
    ``price_targets=None`` for the caller.
    """
    if len(batch) == 6:
        game_state_data, batch_data, legal_action_mask, pi, value, price_targets = batch
    else:
        game_state_data, batch_data, legal_action_mask, pi, value = batch
        price_targets = None
    game_state_data = game_state_data.squeeze(1).float().to(device, non_blocking=True)
    batch_data = batch_data.to(device, non_blocking=True)
    legal_action_mask = legal_action_mask.float().to(device, non_blocking=True)
    pi = pi.float().to(device, non_blocking=True)
    value = value.float().to(device, non_blocking=True)
    return game_state_data, batch_data, legal_action_mask, pi, value, price_targets


def train(
    config: TrainingConfig,
    model: AlphaZeroModel,
    graph: bool = False,
    model_checkpoint_dir: str = "model_checkpoints",
) -> TrainingMetrics:
    if not config.train_dir.exists():
        LOGGER.warning(f"Training directory {config.train_dir} does not exist. Aborting training.")
        return model, TrainingMetrics()

    start_index = 0
    if config.max_training_window > 0:
        full_dataset = SelfPlayDataset(config.train_dir)
        total_examples = len(full_dataset)
        full_dataset.env.close()
        if total_examples > config.max_training_window:
            start_index = total_examples - config.max_training_window
            LOGGER.info(
                f"Training window active: using {config.max_training_window} of {total_examples} examples "
                f"(start_index={start_index})"
            )

    train_dataset = SelfPlayDataset(config.train_dir, start_index=start_index)

    metrics = train_model(model, train_dataset, config, graph, model_checkpoint_dir)
    return model, metrics


def train_model(
    model: AlphaZeroModel,
    train_dataset: Dataset,
    config: TrainingConfig,
    graph: bool = False,
    model_checkpoint_dir: str = "model_checkpoints",
) -> TrainingMetrics:
    # Separate learning rate for value heads (Item 5). With the KataGo-style
    # dual head, both ``win_loss_head`` and ``score_head`` use the elevated LR
    # — they replace the legacy ``value_head`` and the rationale (small final
    # MLP, slower than policy/trunk to specialize) applies to both.
    value_head_params = []
    other_params = []
    for name, param in model.named_parameters():
        if "win_loss_head" in name or "score_head" in name:
            value_head_params.append(param)
        else:
            other_params.append(param)
    optimizer = optim.Adam(
        [
            {"params": other_params, "lr": config.lr},
            {"params": value_head_params, "lr": config.lr * config.value_lr_multiplier},
        ],
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    metrics = TrainingMetrics()

    if len(train_dataset) == 0:
        LOGGER.warning("Dataset is empty. Skipping training.")
        return metrics

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=config.shuffle_examples, num_workers=0, pin_memory=False
    )

    total_steps = len(train_loader) * config.num_epochs
    warmup_steps = min(total_steps // 20, 200)  # 5% of first iteration, capped at 200 steps
    # Linear warmup then constant LR (stable for indefinite training).
    scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)

    # Try to load optimizer state from a previous iteration. Param-group structure may have changed
    # (e.g. value-head LR split), so tolerate a mismatch and start fresh in that case.
    try:
        loaded = load_optimizer_state(optimizer, scheduler, model_checkpoint_dir, model)
        if loaded:
            LOGGER.info(f"Resumed optimizer state. Current LR: {optimizer.param_groups[0]['lr']:.2e}")
    except (ValueError, RuntimeError, KeyError) as e:
        LOGGER.warning(
            f"Could not load optimizer state (likely parameter group mismatch): {e}. "
            f"Continuing with fresh optimizer."
        )

    metrics.training_examples = len(train_dataset)
    device = model.device

    # FP16 mixed-precision training (Item 7)
    use_amp = config.use_fp16_training and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    if use_amp:
        LOGGER.info("FP16 mixed-precision training enabled (CUDA).")
    else:
        LOGGER.info(f"FP16 training disabled (device={device.type}, config={config.use_fp16_training}).")

    if graph:
        plt.ion()
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Training Progress (Live - Batch Level)", fontsize=16, fontweight="bold")
        plt.tight_layout()
        plot_output = display(HTML("<div id='training-plot'></div>"), display_id=True)
    else:
        fig, axes = None, None

    accum_steps = config.gradient_accumulation_steps
    global_batch_number = 0
    for epoch in range(config.num_epochs):
        model.train()
        train_losses = []
        train_policy_losses = []
        train_value_losses = []
        # Per-epoch metric accumulators.
        epoch_entropies = []
        epoch_aux_losses_batch = []
        epoch_top1_correct = 0
        epoch_top5_correct = 0
        epoch_total_samples = 0
        epoch_legal_action_counts = []
        epoch_policy_max_probs = []
        epoch_value_preds_all = []
        epoch_value_targets_all = []
        epoch_aux_preds_all = []
        epoch_aux_targets_all = []
        epoch_grad_norms = []
        epoch_policy_entropies = []
        epoch_target_entropies = []
        epoch_policy_kl_values = []

        optimizer.zero_grad()
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs} [Train]", leave=False)

        for batch_idx, batch in enumerate(train_pbar):
            global_batch_number += 1
            # 6-tuple batches carry per-example ``price_targets``; tolerate
            # legacy 5-tuple batches by defaulting that field to None (the
            # price head simply contributes zero loss in that case).
            if len(batch) == 6:
                game_state_data, batch_data, legal_action_mask, pi, value, price_targets = batch
            else:
                game_state_data, batch_data, legal_action_mask, pi, value = batch
                price_targets = None
            game_state_data = game_state_data.squeeze(1).float().to(device, non_blocking=True)
            batch_data = batch_data.to(device, non_blocking=True)
            legal_action_mask = legal_action_mask.float().to(device, non_blocking=True)
            pi = pi.float().to(device, non_blocking=True)
            value = value.float().to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                policy_logits, win_loss_logits, score_pred, aux_action_count_pred = model(
                    game_state_data, batch_data
                )

                # --- Policy loss ---
                # Decomposed CE per autoregressive level when the
                # hierarchical head is in use (Transformer model); legacy
                # masked-CE fallback for the GNN model.
                policy_components = getattr(model, "last_policy_components", None)
                masked_logits = policy_logits.masked_fill(legal_action_mask == 0, float("-inf"))
                log_probs = F.log_softmax(masked_logits, dim=1)
                safe_log_probs = log_probs.masked_fill(legal_action_mask == 0, 0.0)
                if policy_components is not None:
                    policy_loss, _ = _compute_decomposed_policy_loss(
                        policy_components, pi, legal_action_mask
                    )
                else:
                    policy_loss = -torch.sum(pi * safe_log_probs, dim=1).mean()

                # --- Policy entropy ---
                policy_probs = F.softmax(masked_logits, dim=1)
                entropy = -torch.sum(policy_probs * safe_log_probs, dim=1).mean()

                # --- Dual value targets derived from the stored value tensor ---
                win_loss_target, score_target = _derive_dual_value_targets(value)

                # --- Win-loss loss (KL-div on share-of-winners distribution) ---
                win_loss_log_probs = F.log_softmax(win_loss_logits, dim=1)
                value_loss = -(win_loss_target * win_loss_log_probs).sum(dim=1).mean()

                # --- Score loss (MSE on normalized net-worth fractions) ---
                score_loss = F.mse_loss(score_pred, score_target)

                # --- Auxiliary loss ---
                legal_action_count = legal_action_mask.sum(dim=1)
                aux_target = torch.log(legal_action_count.float().clamp(min=1))
                aux_pred = aux_action_count_pred.squeeze(1)
                aux_loss = F.mse_loss(aux_pred, aux_target)

                # --- Continuous price NLL ---
                # The price head only contributes when the model exposes
                # ``last_price_components`` (Transformer architecture) AND the
                # current batch has price targets. Self-play batches today
                # don't carry price targets so ``price_loss_value`` stays at
                # zero in that case; pretraining batches do carry them and
                # this is the path that wires them into the optimizer.
                price_components = getattr(model, "last_price_components", None)
                price_loss_value = torch.tensor(
                    0.0, device=policy_loss.device, dtype=policy_loss.dtype
                )
                if price_components is not None and price_targets is not None:
                    price_loss_value, _ = _compute_price_nll_loss(
                        price_components, price_targets
                    )

                total_loss = (
                    policy_loss
                    + config.value_loss_weight * value_loss
                    + config.score_loss_weight * score_loss
                    + config.price_loss_weight * price_loss_value
                    + model.config.aux_loss_weight * aux_loss
                    - config.entropy_weight * entropy
                )

            # Keep ``value_pred`` as an alias for ``win_loss_logits`` so the
            # diagnostics block below (top-1 softmax stats, explained variance,
            # …) doesn't need rewriting — it always meant "the head MCTS reads".
            value_pred = win_loss_logits

            if not torch.isfinite(total_loss):
                LOGGER.warning(
                    f"Non-finite loss at batch {batch_idx}: total={total_loss.item():.4f}, "
                    f"policy={policy_loss.item():.4f}, value={value_loss.item():.4f}, "
                    f"aux={aux_loss.item():.4f}, entropy={entropy.item():.4f}. Skipping batch."
                )
                optimizer.zero_grad()
                continue

            # --- Gradient step ---
            if scaler is not None:
                scaler.scale(total_loss / accum_steps).backward()
                if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                    scaler.unscale_(optimizer)
                    # Check for inf/nan gradients before computing norms (GradScaler may produce these)
                    found_inf = any(
                        torch.isinf(p.grad).any() or torch.isnan(p.grad).any()
                        for p in model.parameters() if p.grad is not None
                    )
                    if not found_inf:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        grad_norms = _compute_grad_norms(model)
                        epoch_grad_norms.append(grad_norms)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()
            else:
                (total_loss / accum_steps).backward()
                if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    grad_norms = _compute_grad_norms(model)
                    epoch_grad_norms.append(grad_norms)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

            # --- Accumulate batch-level metrics ---
            train_losses.append(total_loss.item())
            train_policy_losses.append(policy_loss.item())
            train_value_losses.append(value_loss.item())
            epoch_entropies.append(entropy.item())
            epoch_aux_losses_batch.append(aux_loss.item())

            metrics.batch_losses.append(total_loss.item())
            metrics.batch_policy_losses.append(policy_loss.item())
            metrics.batch_value_losses.append(value_loss.item())
            metrics.batch_numbers.append(global_batch_number)

            # --- Policy diagnostics (computed on detached tensors) ---
            with torch.no_grad():
                batch_size = pi.size(0)
                epoch_total_samples += batch_size

                # Top-1 and top-5 accuracy (does MCTS's top move match network's top move?)
                target_top1 = pi.argmax(dim=1)
                pred_top1 = policy_probs.argmax(dim=1)
                epoch_top1_correct += (target_top1 == pred_top1).sum().item()

                pred_top5 = policy_probs.topk(5, dim=1).indices
                target_top1_expanded = target_top1.unsqueeze(1).expand_as(pred_top5)
                epoch_top5_correct += (pred_top5 == target_top1_expanded).any(dim=1).sum().item()

                # Legal move concentration (max prob among legal moves)
                legal_probs = policy_probs * legal_action_mask
                epoch_policy_max_probs.extend(legal_probs.max(dim=1).values.cpu().tolist())

                # Legal action counts
                epoch_legal_action_counts.extend(legal_action_count.cpu().tolist())

                # Per-sample policy entropy (network output)
                per_sample_entropy = -torch.sum(policy_probs * safe_log_probs, dim=1)
                epoch_policy_entropies.extend(per_sample_entropy.cpu().tolist())

                # Per-sample target entropy (MCTS policy)
                safe_pi = pi.clamp(min=1e-8)
                target_ent = -torch.sum(pi * torch.log(safe_pi), dim=1)
                epoch_target_entropies.extend(target_ent.cpu().tolist())

                # KL divergence: KL(MCTS target || network output)
                # = sum(pi * (log(pi) - log(policy_probs)))
                kl_per_sample = torch.sum(pi * (torch.log(safe_pi) - safe_log_probs), dim=1)
                epoch_policy_kl_values.extend(kl_per_sample.cpu().tolist())

                # Value statistics — both pred and target are share-of-winners
                # distributions (softmax of the win-loss head vs. the derived
                # win-loss target), so MAE / MSE / correlation are meaningful.
                value_pred_probs = F.softmax(value_pred, dim=1)
                epoch_value_preds_all.append(value_pred_probs.cpu())
                epoch_value_targets_all.append(win_loss_target.cpu())

                # Aux statistics
                epoch_aux_preds_all.append(aux_pred.cpu())
                epoch_aux_targets_all.append(aux_target.cpu())

            train_pbar.set_postfix(
                {
                    "loss": f"{total_loss.item():.4f}",
                    "policy": f"{policy_loss.item():.4f}",
                    "value": f"{value_loss.item():.4f}",
                    "entropy": f"{entropy.item():.4f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                }
            )

            if graph and global_batch_number % 100 == 0:
                update_live_plot(metrics, axes, epoch + 1, config.num_epochs, global_batch_number, fig, plot_output)

        # --- Compute epoch-level metrics ---
        avg_epoch_loss = np.mean(train_losses)
        avg_policy_loss = np.mean(train_policy_losses)
        avg_value_loss = np.mean(train_value_losses)

        metrics.epoch_losses.append(avg_epoch_loss)
        metrics.epoch_policy_losses.append(avg_policy_loss)
        metrics.epoch_value_losses.append(avg_value_loss)
        metrics.epoch_entropy.append(np.mean(epoch_entropies) if epoch_entropies else 0.0)
        metrics.epoch_aux_losses.append(np.mean(epoch_aux_losses_batch) if epoch_aux_losses_batch else 0.0)

        # Policy diagnostics
        metrics.epoch_top1_accuracy.append(epoch_top1_correct / max(epoch_total_samples, 1))
        metrics.epoch_top5_accuracy.append(epoch_top5_correct / max(epoch_total_samples, 1))
        metrics.epoch_policy_entropy.append(np.mean(epoch_policy_entropies) if epoch_policy_entropies else 0.0)
        metrics.epoch_target_entropy.append(np.mean(epoch_target_entropies) if epoch_target_entropies else 0.0)
        metrics.epoch_legal_move_concentration.append(np.mean(epoch_policy_max_probs) if epoch_policy_max_probs else 0.0)
        metrics.epoch_mean_legal_actions.append(np.mean(epoch_legal_action_counts) if epoch_legal_action_counts else 0.0)
        metrics.epoch_policy_kl.append(np.mean(epoch_policy_kl_values) if epoch_policy_kl_values else 0.0)

        # Value diagnostics
        if epoch_value_preds_all:
            all_preds = torch.cat(epoch_value_preds_all, dim=0)
            all_targets = torch.cat(epoch_value_targets_all, dim=0)
            metrics.epoch_value_pred_mean.append(all_preds.mean().item())
            metrics.epoch_value_pred_std.append(all_preds.std().item())
            metrics.epoch_value_target_mean.append(all_targets.mean().item())
            metrics.epoch_value_target_std.append(all_targets.std().item())
            metrics.epoch_value_mae.append((all_preds - all_targets).abs().mean().item())
            metrics.epoch_value_mse.append(((all_preds - all_targets) ** 2).mean().item())
            # Per-player correlation and explained variance (averaged across players).
            # Computing over flattened (N, 4) tensors is misleading because both preds
            # and targets are probability distributions clustered around 0.25.
            per_player_corr = []
            per_player_ev = []
            for p in range(all_preds.shape[1]):
                per_player_corr.append(_safe_correlation(all_preds[:, p], all_targets[:, p]))
                target_var = all_targets[:, p].var().item()
                residual_var = (all_targets[:, p] - all_preds[:, p]).var().item()
                per_player_ev.append(1.0 - residual_var / max(target_var, 1e-8))
            metrics.epoch_value_correlation.append(np.mean(per_player_corr))
            metrics.epoch_value_explained_variance.append(np.mean(per_player_ev))

            metrics.epoch_value_pred_min.append(all_preds.min().item())
            metrics.epoch_value_pred_max.append(all_preds.max().item())
            metrics.epoch_value_target_min.append(all_targets.min().item())
            metrics.epoch_value_target_max.append(all_targets.max().item())
        else:
            for lst in [metrics.epoch_value_pred_mean, metrics.epoch_value_pred_std,
                        metrics.epoch_value_target_mean, metrics.epoch_value_target_std,
                        metrics.epoch_value_mae, metrics.epoch_value_mse, metrics.epoch_value_correlation,
                        metrics.epoch_value_explained_variance, metrics.epoch_value_pred_min,
                        metrics.epoch_value_pred_max, metrics.epoch_value_target_min,
                        metrics.epoch_value_target_max]:
                lst.append(0.0)

        # Gradient norms (average across steps in epoch)
        if epoch_grad_norms:
            total_norms = [g["total"] for g in epoch_grad_norms]
            metrics.epoch_grad_norm_total.append(np.mean(total_norms))
            metrics.epoch_grad_norm_policy_head.append(np.mean([g["policy_head"] for g in epoch_grad_norms]))
            metrics.epoch_grad_norm_value_head.append(np.mean([g["value_head"] for g in epoch_grad_norms]))
            metrics.epoch_grad_norm_trunk.append(np.mean([g["trunk"] for g in epoch_grad_norms]))
            # Coefficient of variation: std / mean (measures gradient stability)
            mean_norm = np.mean(total_norms)
            std_norm = np.std(total_norms)
            metrics.epoch_grad_norm_cv.append(std_norm / max(mean_norm, 1e-8))
        else:
            for lst in [metrics.epoch_grad_norm_total, metrics.epoch_grad_norm_policy_head,
                        metrics.epoch_grad_norm_value_head, metrics.epoch_grad_norm_trunk,
                        metrics.epoch_grad_norm_cv]:
                lst.append(0.0)

        # Learning rate
        metrics.epoch_lr.append(optimizer.param_groups[0]["lr"])

        # Aux diagnostics
        if epoch_aux_preds_all:
            all_aux_preds = torch.cat(epoch_aux_preds_all, dim=0)
            all_aux_targets = torch.cat(epoch_aux_targets_all, dim=0)
            metrics.epoch_aux_pred_mean.append(all_aux_preds.mean().item())
            metrics.epoch_aux_target_mean.append(all_aux_targets.mean().item())
            metrics.epoch_aux_correlation.append(_safe_correlation(all_aux_preds, all_aux_targets))
        else:
            metrics.epoch_aux_pred_mean.append(0.0)
            metrics.epoch_aux_target_mean.append(0.0)
            metrics.epoch_aux_correlation.append(0.0)

        # Log epoch summary
        LOGGER.info(
            f"Epoch {epoch+1}/{config.num_epochs} Summary:\n"
            f"  Losses  - Total: {avg_epoch_loss:.4f}, Policy: {avg_policy_loss:.4f}, "
            f"Value: {avg_value_loss:.4f}, Entropy: {metrics.epoch_entropy[-1]:.4f}, "
            f"Aux: {metrics.epoch_aux_losses[-1]:.4f}\n"
            f"  Policy  - Top1: {metrics.epoch_top1_accuracy[-1]:.3f}, "
            f"Top5: {metrics.epoch_top5_accuracy[-1]:.3f}, "
            f"KL: {metrics.epoch_policy_kl[-1]:.4f}, "
            f"NetEntropy: {metrics.epoch_policy_entropy[-1]:.3f}, "
            f"MCTSEntropy: {metrics.epoch_target_entropy[-1]:.3f}, "
            f"MaxProb: {metrics.epoch_legal_move_concentration[-1]:.3f}\n"
            f"  Value   - ExplVar: {metrics.epoch_value_explained_variance[-1]:.3f}, "
            f"Corr: {metrics.epoch_value_correlation[-1]:.3f}, "
            f"MAE: {metrics.epoch_value_mae[-1]:.4f}, "
            f"MSE: {metrics.epoch_value_mse[-1]:.4f}\n"
            f"  Grads   - Total: {metrics.epoch_grad_norm_total[-1]:.3f}, "
            f"Policy: {metrics.epoch_grad_norm_policy_head[-1]:.3f}, "
            f"Value: {metrics.epoch_grad_norm_value_head[-1]:.3f}, "
            f"Trunk: {metrics.epoch_grad_norm_trunk[-1]:.3f}, "
            f"CV: {metrics.epoch_grad_norm_cv[-1]:.3f}\n"
            f"  LR: {metrics.epoch_lr[-1]:.2e}, "
            f"Avg Legal Actions: {metrics.epoch_mean_legal_actions[-1]:.1f}"
        )

        if graph:
            update_live_plot(metrics, axes, epoch + 1, config.num_epochs, global_batch_number, fig, plot_output)

    # Save optimizer state for next iteration
    save_optimizer_state(optimizer, scheduler, model_checkpoint_dir, model)

    # Always emit a numbered checkpoint so the trained candidate survives a
    # crash between training and gating. Promotion (i.e. updating the
    # current_best pointer) is the caller's responsibility — see loop.py's
    # _run_gating_iteration. Pretraining wrappers should likewise update the
    # pointer themselves once they've decided the run was successful.
    checkpoint_num = save_model(model, model_checkpoint_dir)
    metrics.checkpoint_num = checkpoint_num
    LOGGER.info(f"Saved trained model as checkpoint {checkpoint_num} (not yet promoted).")

    # Calculate final average metrics across all epochs
    metrics.epochs_trained = config.num_epochs
    if metrics.epoch_losses:
        metrics.avg_total_loss = np.mean(metrics.epoch_losses)
        metrics.avg_policy_loss = np.mean(metrics.epoch_policy_losses)
        metrics.avg_value_loss = np.mean(metrics.epoch_value_losses)

    if graph:
        plt.ioff()
        plt.savefig("training_curves_final.png", dpi=300, bbox_inches="tight")
        LOGGER.info("Final training curves saved as 'training_curves_final.png'")

    return metrics


def update_live_plot(
    metrics: TrainingMetrics, axes, current_epoch: int, total_epochs: int, current_batch: int, fig, plot_output
):
    """Update the live training plot during training with batch-level data"""
    for ax in axes.flat:
        ax.clear()

    if metrics.batch_numbers:
        batch_numbers = metrics.batch_numbers

        window_size = min(20, len(metrics.batch_losses) // 4)
        if window_size > 1:
            smoothed_losses = np.convolve(metrics.batch_losses, np.ones(window_size) / window_size, mode="valid")
            smoothed_policy_losses = np.convolve(
                metrics.batch_policy_losses, np.ones(window_size) / window_size, mode="valid"
            )
            smoothed_value_losses = np.convolve(
                metrics.batch_value_losses, np.ones(window_size) / window_size, mode="valid"
            )
            smoothed_batches = batch_numbers[window_size - 1 :]
        else:
            smoothed_losses = metrics.batch_losses
            smoothed_policy_losses = metrics.batch_policy_losses
            smoothed_value_losses = metrics.batch_value_losses
            smoothed_batches = batch_numbers

        if metrics.epoch_losses:
            batches_per_epoch = len(metrics.batch_numbers) // len(metrics.epoch_losses)
            epoch_batches = [i * batches_per_epoch for i in range(1, len(metrics.epoch_losses) + 1)]
        else:
            epoch_batches = []

        axes[0, 0].plot(batch_numbers, metrics.batch_losses, "b-", label="Train Raw", linewidth=0.5, alpha=0.3)
        axes[0, 0].plot(smoothed_batches, smoothed_losses, "b-", label="Train Smoothed", linewidth=2, alpha=0.8)
        if epoch_batches and epoch_batches[0] <= len(metrics.batch_numbers):
            axes[0, 0].plot(
                epoch_batches, metrics.epoch_losses, "g-", label="Train Epoch Avg",
                linewidth=3, marker="o", markersize=6,
            )
        axes[0, 0].set_title("Total Loss (Batch Level)")
        axes[0, 0].set_xlabel("Batch Number")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(batch_numbers, metrics.batch_policy_losses, "g-", label="Train Raw", linewidth=0.5, alpha=0.3)
        axes[0, 1].plot(smoothed_batches, smoothed_policy_losses, "g-", label="Train Smoothed", linewidth=2, alpha=0.8)
        if metrics.epoch_policy_losses and epoch_batches:
            axes[0, 1].plot(
                epoch_batches, metrics.epoch_policy_losses, "g-", label="Train Epoch Avg",
                linewidth=3, marker="o", markersize=6,
            )
        axes[0, 1].set_title("Policy Loss (Batch Level)")
        axes[0, 1].set_xlabel("Batch Number")
        axes[0, 1].set_ylabel("Loss")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].plot(
            batch_numbers, metrics.batch_value_losses, "orange", label="Train Raw", linewidth=0.5, alpha=0.3
        )
        axes[1, 0].plot(
            smoothed_batches, smoothed_value_losses, "orange", label="Train Smoothed", linewidth=2, alpha=0.8
        )
        if metrics.epoch_value_losses and epoch_batches:
            axes[1, 0].plot(
                epoch_batches, metrics.epoch_value_losses, "orange", label="Train Epoch Avg",
                linewidth=3, marker="o", markersize=6,
            )
        axes[1, 0].set_title("Value Loss (Batch Level)")
        axes[1, 0].set_xlabel("Batch Number")
        axes[1, 0].set_ylabel("Loss")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].plot(smoothed_batches, smoothed_losses, "b-", label="Train Total", linewidth=2, alpha=0.8)
        axes[1, 1].plot(smoothed_batches, smoothed_policy_losses, "g-", label="Train Policy", linewidth=2, alpha=0.8)
        axes[1, 1].plot(smoothed_batches, smoothed_value_losses, "orange", label="Train Value", linewidth=2, alpha=0.8)
        axes[1, 1].set_title("All Losses (Smoothed)")
        axes[1, 1].set_xlabel("Batch Number")
        axes[1, 1].set_ylabel("Loss")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(
        f"Training Progress (Live) - Epoch {current_epoch}/{total_epochs}, Batch {current_batch}",
        fontsize=16, fontweight="bold",
    )
    plt.tight_layout()
    plot_output.update(fig)


def plot_training_curves(metrics: TrainingMetrics):
    """Plot training and validation loss curves using matplotlib (static version)"""
    epochs = range(1, len(metrics.epoch_losses) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Training Progress", fontsize=16, fontweight="bold")

    axes[0, 0].plot(epochs, metrics.epoch_losses, "b-", label="Train", linewidth=2, marker="o", markersize=4)
    axes[0, 0].set_title("Total Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(epochs, metrics.epoch_policy_losses, "b-", label="Train", linewidth=2, marker="o", markersize=4)
    axes[0, 1].set_title("Policy Loss")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(epochs, metrics.epoch_value_losses, "b-", label="Train", linewidth=2, marker="o", markersize=4)
    axes[1, 0].set_title("Value Loss")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Loss")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(epochs, metrics.epoch_losses, "b-", label="Total", linewidth=2, marker="o", markersize=4)
    axes[1, 1].plot(epochs, metrics.epoch_policy_losses, "b--", label="Policy", linewidth=1.5, marker="^", markersize=3)
    axes[1, 1].plot(epochs, metrics.epoch_value_losses, "b:", label="Value", linewidth=1.5, marker="v", markersize=3)
    axes[1, 1].set_title("All Losses")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Loss")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    plt.savefig("training_curves.png", dpi=300, bbox_inches="tight")
    LOGGER.info("Training curves saved as 'training_curves.png'")
