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
    epoch_policy_kl: list = field(default_factory=list)  # KL divergence between old and new policy
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
    epoch_value_correlation: list = field(default_factory=list)  # correlation between pred and target

    # Gradient diagnostics
    epoch_grad_norm_total: list = field(default_factory=list)
    epoch_grad_norm_policy_head: list = field(default_factory=list)
    epoch_grad_norm_value_head: list = field(default_factory=list)
    epoch_grad_norm_trunk: list = field(default_factory=list)

    # Learning rate
    epoch_lr: list = field(default_factory=list)

    # Aux head diagnostics
    epoch_aux_pred_mean: list = field(default_factory=list)
    epoch_aux_target_mean: list = field(default_factory=list)
    epoch_aux_correlation: list = field(default_factory=list)


def _compute_grad_norms(model: AlphaZeroModel) -> dict:
    """Compute gradient norms for different model components."""
    norms = {"total": 0.0, "policy_head": 0.0, "value_head": 0.0, "trunk": 0.0}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.data.norm(2).item() ** 2
            norms["total"] += grad_norm
            if "policy" in name:
                norms["policy_head"] += grad_norm
            elif "value_head" in name:
                norms["value_head"] += grad_norm
            else:
                norms["trunk"] += grad_norm
    norms = {k: v**0.5 for k, v in norms.items()}
    return norms


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
    optimizer = optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay, betas=(0.9, 0.999), eps=1e-8
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
    # Linear warmup then constant LR — simple and stable for indefinite training
    scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)

    # Try to load optimizer state from previous iteration (warmup already completed → LR stays at config.lr)
    loaded = load_optimizer_state(optimizer, scheduler, model_checkpoint_dir, model)
    if loaded:
        LOGGER.info(f"Resumed optimizer state. Current LR: {optimizer.param_groups[0]['lr']:.2e}")

    metrics.training_examples = len(train_dataset)
    device = model.device

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
        # Per-epoch accumulators for comprehensive metrics
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

        optimizer.zero_grad()
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs} [Train]", leave=False)

        for batch_idx, batch in enumerate(train_pbar):
            global_batch_number += 1
            game_state_data, batch_data, legal_action_mask, pi, value = batch
            game_state_data = game_state_data.squeeze(1).float().to(device, non_blocking=True)
            batch_data = batch_data.to(device, non_blocking=True)
            legal_action_mask = legal_action_mask.float().to(device, non_blocking=True)
            pi = pi.float().to(device, non_blocking=True)
            value = value.float().to(device, non_blocking=True)

            policy_logits, value_pred, aux_action_count_pred = model(game_state_data, batch_data)

            # --- Policy loss ---
            masked_logits = policy_logits.masked_fill(legal_action_mask == 0, float("-inf"))
            log_probs = F.log_softmax(masked_logits, dim=1)
            safe_log_probs = log_probs.masked_fill(legal_action_mask == 0, 0.0)
            policy_loss = -torch.sum(pi * safe_log_probs, dim=1).mean()

            # --- Policy entropy ---
            policy_probs = F.softmax(masked_logits, dim=1)
            entropy = -torch.sum(policy_probs * safe_log_probs, dim=1).mean()

            # --- Value loss ---
            is_score_values = (value >= 0).all()
            if is_score_values:
                value_log_probs = F.log_softmax(value_pred, dim=1)
                value_loss = F.kl_div(value_log_probs, value, reduction="batchmean")
            else:
                winners_mask = (value > -0.5).float()
                num_winners = winners_mask.sum(dim=1, keepdim=True).clamp(min=1)
                value_target_probs = winners_mask / num_winners
                value_log_probs = F.log_softmax(value_pred, dim=1)
                value_loss = -(value_target_probs * value_log_probs).sum(dim=1).mean()

            # --- Auxiliary loss ---
            legal_action_count = legal_action_mask.sum(dim=1)
            aux_target = torch.log(legal_action_count.float().clamp(min=1))
            aux_pred_clamped = aux_action_count_pred.squeeze(1).clamp(-10, 10)
            aux_loss = F.mse_loss(aux_pred_clamped, aux_target)

            total_loss = (
                policy_loss
                + config.value_loss_weight * value_loss
                + model.config.aux_loss_weight * aux_loss
                - config.entropy_weight * entropy
            )

            if not torch.isfinite(total_loss):
                LOGGER.warning(
                    f"Non-finite loss at batch {batch_idx}: total={total_loss.item():.4f}, "
                    f"policy={policy_loss.item():.4f}, value={value_loss.item():.4f}, "
                    f"aux={aux_loss.item():.4f}, entropy={entropy.item():.4f}. Skipping batch."
                )
                optimizer.zero_grad()
                continue

            # --- Gradient step ---
            (total_loss / accum_steps).backward()
            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # Compute gradient norms before optimizer step
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

                # Value statistics
                value_pred_probs = F.softmax(value_pred, dim=1)
                epoch_value_preds_all.append(value_pred_probs.cpu())
                epoch_value_targets_all.append(value.cpu())

                # Aux statistics
                epoch_aux_preds_all.append(aux_pred_clamped.cpu())
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

        # Value diagnostics
        if epoch_value_preds_all:
            all_preds = torch.cat(epoch_value_preds_all, dim=0)
            all_targets = torch.cat(epoch_value_targets_all, dim=0)
            metrics.epoch_value_pred_mean.append(all_preds.mean().item())
            metrics.epoch_value_pred_std.append(all_preds.std().item())
            metrics.epoch_value_target_mean.append(all_targets.mean().item())
            metrics.epoch_value_target_std.append(all_targets.std().item())
            metrics.epoch_value_mae.append((all_preds - all_targets).abs().mean().item())
            metrics.epoch_value_correlation.append(_safe_correlation(all_preds, all_targets))

            # Explained variance: 1 - Var(target - pred) / Var(target)
            residual_var = (all_targets - all_preds).var().item()
            target_var = all_targets.var().item()
            ev = 1.0 - residual_var / max(target_var, 1e-8)
            metrics.epoch_value_explained_variance.append(ev)
        else:
            for lst in [metrics.epoch_value_pred_mean, metrics.epoch_value_pred_std,
                        metrics.epoch_value_target_mean, metrics.epoch_value_target_std,
                        metrics.epoch_value_mae, metrics.epoch_value_correlation,
                        metrics.epoch_value_explained_variance]:
                lst.append(0.0)

        # Gradient norms (average across steps in epoch)
        if epoch_grad_norms:
            metrics.epoch_grad_norm_total.append(np.mean([g["total"] for g in epoch_grad_norms]))
            metrics.epoch_grad_norm_policy_head.append(np.mean([g["policy_head"] for g in epoch_grad_norms]))
            metrics.epoch_grad_norm_value_head.append(np.mean([g["value_head"] for g in epoch_grad_norms]))
            metrics.epoch_grad_norm_trunk.append(np.mean([g["trunk"] for g in epoch_grad_norms]))
        else:
            for lst in [metrics.epoch_grad_norm_total, metrics.epoch_grad_norm_policy_head,
                        metrics.epoch_grad_norm_value_head, metrics.epoch_grad_norm_trunk]:
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
            f"NetEntropy: {metrics.epoch_policy_entropy[-1]:.3f}, "
            f"MCTSEntropy: {metrics.epoch_target_entropy[-1]:.3f}, "
            f"MaxProb: {metrics.epoch_legal_move_concentration[-1]:.3f}\n"
            f"  Value   - ExplVar: {metrics.epoch_value_explained_variance[-1]:.3f}, "
            f"Corr: {metrics.epoch_value_correlation[-1]:.3f}, "
            f"MAE: {metrics.epoch_value_mae[-1]:.4f}\n"
            f"  Grads   - Total: {metrics.epoch_grad_norm_total[-1]:.3f}, "
            f"Policy: {metrics.epoch_grad_norm_policy_head[-1]:.3f}, "
            f"Value: {metrics.epoch_grad_norm_value_head[-1]:.3f}, "
            f"Trunk: {metrics.epoch_grad_norm_trunk[-1]:.3f}\n"
            f"  LR: {metrics.epoch_lr[-1]:.2e}, "
            f"Avg Legal Actions: {metrics.epoch_mean_legal_actions[-1]:.1f}"
        )

        if graph:
            update_live_plot(metrics, axes, epoch + 1, config.num_epochs, global_batch_number, fig, plot_output)

    # Save optimizer state for next iteration
    save_optimizer_state(optimizer, scheduler, model_checkpoint_dir, model)

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
