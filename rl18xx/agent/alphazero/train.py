import numpy as np
from rl18xx.agent.alphazero.model import AlphaZeroModel
from rl18xx.agent.alphazero.config import TrainingConfig
from rl18xx.agent.alphazero.dataset import SelfPlayDataset
from rl18xx.agent.alphazero.checkpointer import get_latest_model, save_model

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


def train(config: TrainingConfig, model: AlphaZeroModel, graph: bool = False) -> TrainingMetrics:
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

    metrics = train_model(model, train_dataset, config, graph)
    return model, metrics


def train_model(
    model: AlphaZeroModel,
    train_dataset: Dataset,
    config: TrainingConfig,
    graph: bool = False,
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
    warmup_steps = total_steps // 20  # 5% warmup
    warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=1e-5)
    scheduler = optim.lr_scheduler.SequentialLR(optimizer, [warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])

    metrics.training_examples = len(train_dataset)
    device = model.device

    if graph:
        # Create figure once and reuse it
        plt.ion()
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Training Progress (Live - Batch Level)", fontsize=16, fontweight="bold")
        plt.tight_layout()

        # Create a dedicated output area for the plot
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
        optimizer.zero_grad()
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs} [Train]", leave=False)

        for batch_idx, batch in enumerate(train_pbar):
            global_batch_number += 1
            game_state_data, batch_data, legal_action_mask, pi, value = batch
            # DataLoader adds an extra dimension
            game_state_data = game_state_data.squeeze(1).float().to(device, non_blocking=True)
            batch_data = batch_data.to(device, non_blocking=True)
            legal_action_mask = legal_action_mask.float().to(device, non_blocking=True)
            pi = pi.float().to(device, non_blocking=True)
            value = value.float().to(device, non_blocking=True)

            policy_logits, value_pred, aux_action_count_pred = model(game_state_data, batch_data)

            # Cross-entropy loss for policy with proper illegal-action masking
            masked_logits = policy_logits.masked_fill(legal_action_mask == 0, float("-inf"))
            log_probs = F.log_softmax(masked_logits, dim=1)
            # Replace -inf with 0 in log_probs to avoid 0 * -inf = NaN in backward pass
            safe_log_probs = log_probs.masked_fill(legal_action_mask == 0, 0.0)
            policy_loss = -torch.sum(pi * safe_log_probs, dim=1).mean()

            # Phase 6.6: Policy entropy bonus to prevent premature collapse
            policy_probs = F.softmax(masked_logits, dim=1)
            entropy = -torch.sum(policy_probs * safe_log_probs, dim=1).mean()

            # Value loss — supports both score fractions and legacy win/loss targets
            # Score fractions: values are in [0, 1] summing to 1 → use KL divergence
            # Legacy win/loss: values are in {-1, 0, +1} → convert to prob distribution
            is_score_values = (value >= 0).all()
            if is_score_values:
                # KL divergence for score fraction targets (Phase 6.4)
                value_log_probs = F.log_softmax(value_pred, dim=1)
                value_loss = F.kl_div(value_log_probs, value, reduction="batchmean")
            else:
                # Legacy: convert {-1, 0, +1} to probability distribution
                winners_mask = (value > -0.5).float()
                num_winners = winners_mask.sum(dim=1, keepdim=True).clamp(min=1)
                value_target_probs = winners_mask / num_winners
                value_log_probs = F.log_softmax(value_pred, dim=1)
                value_loss = -(value_target_probs * value_log_probs).sum(dim=1).mean()

            # Auxiliary loss: predict log(legal_action_count) (Phase 5.4)
            legal_action_count = legal_action_mask.sum(dim=1)
            aux_target = torch.log(legal_action_count.float().clamp(min=1))
            # Clamp prediction to prevent huge MSE from untrained aux head
            aux_pred_clamped = aux_action_count_pred.squeeze(1).clamp(-10, 10)
            aux_loss = F.mse_loss(aux_pred_clamped, aux_target)

            total_loss = (
                policy_loss
                + config.value_loss_weight * value_loss
                + model.config.aux_loss_weight * aux_loss
                - config.entropy_weight * entropy
            )

            # Skip batches with NaN/Inf loss to prevent poisoning model weights
            if not torch.isfinite(total_loss):
                LOGGER.warning(
                    f"Non-finite loss at batch {batch_idx}: total={total_loss.item():.4f}, "
                    f"policy={policy_loss.item():.4f}, value={value_loss.item():.4f}, "
                    f"aux={aux_loss.item():.4f}, entropy={entropy.item():.4f}. Skipping batch."
                )
                optimizer.zero_grad()
                continue

            # Phase 6.7: Gradient accumulation for larger effective batch size
            (total_loss / accum_steps).backward()
            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            train_losses.append(total_loss.item())
            train_policy_losses.append(policy_loss.item())
            train_value_losses.append(value_loss.item())

            metrics.batch_losses.append(total_loss.item())
            metrics.batch_policy_losses.append(policy_loss.item())
            metrics.batch_value_losses.append(value_loss.item())
            metrics.batch_numbers.append(global_batch_number)

            train_pbar.set_postfix(
                {
                    "loss": f"{total_loss.item():.4f}",
                    "policy_loss": f"{policy_loss.item():.4f}",
                    "value_loss": f"{value_loss.item():.4f}",
                    "batch": global_batch_number,
                }
            )

            if graph and global_batch_number % 100 == 0:  # Update every 100 batches
                update_live_plot(metrics, axes, epoch + 1, config.num_epochs, global_batch_number, fig, plot_output)

        # Calculate average training losses for this epoch
        avg_epoch_loss = np.mean(train_losses)
        avg_policy_loss = np.mean(train_policy_losses)
        avg_value_loss = np.mean(train_value_losses)

        # Store epoch metrics
        metrics.epoch_losses.append(avg_epoch_loss)
        metrics.epoch_policy_losses.append(avg_policy_loss)
        metrics.epoch_value_losses.append(avg_value_loss)

        # Log epoch summary
        LOGGER.info(
            f"Epoch {epoch+1}/{config.num_epochs} Summary:\n"
            f"  Train - Total: {avg_epoch_loss:.4f}, Policy: {avg_policy_loss:.4f}, Value: {avg_value_loss:.4f}\n"
            f"  LR: {optimizer.param_groups[0]['lr']:.2e}"
        )

        if graph:
            update_live_plot(metrics, axes, epoch + 1, config.num_epochs, global_batch_number, fig, plot_output)

    # Calculate final average metrics across all epochs
    metrics.epochs_trained = config.num_epochs
    if metrics.epoch_losses:
        metrics.avg_total_loss = np.mean(metrics.epoch_losses)
        metrics.avg_policy_loss = np.mean(metrics.epoch_policy_losses)
        metrics.avg_value_loss = np.mean(metrics.epoch_value_losses)

    # Final plot and save if requested
    if graph:
        plt.ioff()  # Turn off interactive mode
        plt.savefig("training_curves_final.png", dpi=300, bbox_inches="tight")
        LOGGER.info("Final training curves saved as 'training_curves_final.png'")
        # Don't call plt.show() in Jupyter - the figure is already displayed

    return metrics


def update_live_plot(
    metrics: TrainingMetrics, axes, current_epoch: int, total_epochs: int, current_batch: int, fig, plot_output
):
    """Update the live training plot during training with batch-level data"""
    # Clear all axes
    for ax in axes.flat:
        ax.clear()

    # Plot batch-level data
    if metrics.batch_numbers:
        batch_numbers = metrics.batch_numbers

        # Calculate moving average for smoothing
        window_size = min(20, len(metrics.batch_losses) // 4)  # Adaptive window size
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

        # Calculate epoch batch positions for validation loss plotting
        if metrics.epoch_losses:
            batches_per_epoch = len(metrics.batch_numbers) // len(metrics.epoch_losses)
            epoch_batches = [i * batches_per_epoch for i in range(1, len(metrics.epoch_losses) + 1)]
        else:
            epoch_batches = []

        # Total Loss (batch level)
        axes[0, 0].plot(batch_numbers, metrics.batch_losses, "b-", label="Train Raw", linewidth=0.5, alpha=0.3)
        axes[0, 0].plot(smoothed_batches, smoothed_losses, "b-", label="Train Smoothed", linewidth=2, alpha=0.8)
        # Add epoch averages
        if epoch_batches and epoch_batches[0] <= len(metrics.batch_numbers):
            axes[0, 0].plot(
                epoch_batches,
                metrics.epoch_losses,
                "g-",
                label="Train Epoch Avg",
                linewidth=3,
                marker="o",
                markersize=6,
            )
        axes[0, 0].set_title("Total Loss (Batch Level)")
        axes[0, 0].set_xlabel("Batch Number")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Policy Loss (batch level)
        axes[0, 1].plot(batch_numbers, metrics.batch_policy_losses, "g-", label="Train Raw", linewidth=0.5, alpha=0.3)
        axes[0, 1].plot(smoothed_batches, smoothed_policy_losses, "g-", label="Train Smoothed", linewidth=2, alpha=0.8)
        if metrics.epoch_policy_losses and epoch_batches:
            axes[0, 1].plot(
                epoch_batches,
                metrics.epoch_policy_losses,
                "g-",
                label="Train Epoch Avg",
                linewidth=3,
                marker="o",
                markersize=6,
            )
        axes[0, 1].set_title("Policy Loss (Batch Level)")
        axes[0, 1].set_xlabel("Batch Number")
        axes[0, 1].set_ylabel("Loss")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Value Loss (batch level)
        axes[1, 0].plot(
            batch_numbers, metrics.batch_value_losses, "orange", label="Train Raw", linewidth=0.5, alpha=0.3
        )
        axes[1, 0].plot(
            smoothed_batches, smoothed_value_losses, "orange", label="Train Smoothed", linewidth=2, alpha=0.8
        )
        if metrics.epoch_value_losses and epoch_batches:
            axes[1, 0].plot(
                epoch_batches,
                metrics.epoch_value_losses,
                "orange",
                label="Train Epoch Avg",
                linewidth=3,
                marker="o",
                markersize=6,
            )
        axes[1, 0].set_title("Value Loss (Batch Level)")
        axes[1, 0].set_xlabel("Batch Number")
        axes[1, 0].set_ylabel("Loss")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Combined view with all smoothed losses
        axes[1, 1].plot(smoothed_batches, smoothed_losses, "b-", label="Train Total", linewidth=2, alpha=0.8)
        axes[1, 1].plot(smoothed_batches, smoothed_policy_losses, "g-", label="Train Policy", linewidth=2, alpha=0.8)
        axes[1, 1].plot(smoothed_batches, smoothed_value_losses, "orange", label="Train Value", linewidth=2, alpha=0.8)
        axes[1, 1].set_title("All Losses (Smoothed)")
        axes[1, 1].set_xlabel("Batch Number")
        axes[1, 1].set_ylabel("Loss")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

    # Add progress indicator to the main title
    plt.suptitle(
        f"Training Progress (Live) - Epoch {current_epoch}/{total_epochs}, Batch {current_batch}",
        fontsize=16,
        fontweight="bold",
    )

    # Update the display
    plt.tight_layout()

    # Force a display update
    plot_output.update(fig)


def plot_training_curves(metrics: TrainingMetrics):
    """Plot training and validation loss curves using matplotlib (static version)"""
    epochs = range(1, len(metrics.epoch_losses) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Training Progress", fontsize=16, fontweight="bold")

    # Total Loss
    axes[0, 0].plot(epochs, metrics.epoch_losses, "b-", label="Train", linewidth=2, marker="o", markersize=4)
    axes[0, 0].set_title("Total Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Policy Loss
    axes[0, 1].plot(epochs, metrics.epoch_policy_losses, "b-", label="Train", linewidth=2, marker="o", markersize=4)
    axes[0, 1].set_title("Policy Loss")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Value Loss
    axes[1, 0].plot(epochs, metrics.epoch_value_losses, "b-", label="Train", linewidth=2, marker="o", markersize=4)
    axes[1, 0].set_title("Value Loss")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Loss")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Combined view
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

    # Save the plot
    plt.savefig("training_curves.png", dpi=300, bbox_inches="tight")
    LOGGER.info("Training curves saved as 'training_curves.png'")
