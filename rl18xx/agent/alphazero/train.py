import numpy as np
from rl18xx.agent.alphazero.model import AlphaZeroGNNModel
from rl18xx.agent.alphazero.config import TrainingConfig
from rl18xx.agent.alphazero.dataset import SelfPlayDataset
from rl18xx.agent.alphazero.checkpointer import get_latest_model, save_model

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from tqdm.auto import tqdm
from typing import Optional
import logging
from dataclasses import dataclass, field
from IPython.display import display, clear_output, HTML
import matplotlib.pyplot as plt
import warnings

plt.style.use('default')
warnings.filterwarnings("ignore", message=".*can only test a child process.*")

LOGGER = logging.getLogger(__name__)

@dataclass
class TrainingMetrics:
    """Metrics collected during training"""
    avg_total_loss: float = 0.0
    avg_policy_loss: float = 0.0
    avg_value_loss: float = 0.0
    val_total_loss: float = 0.0
    val_policy_loss: float = 0.0
    val_value_loss: float = 0.0
    training_examples: int = 0
    epochs_trained: int = 0
    epoch_losses: list = field(default_factory=list)
    epoch_policy_losses: list = field(default_factory=list)
    epoch_value_losses: list = field(default_factory=list)
    val_losses: list = field(default_factory=list)
    val_policy_losses: list = field(default_factory=list)
    val_value_losses: list = field(default_factory=list)
    batch_losses: list = field(default_factory=list)
    batch_policy_losses: list = field(default_factory=list)
    batch_value_losses: list = field(default_factory=list)
    batch_numbers: list = field(default_factory=list)


def train(config: TrainingConfig, model: AlphaZeroGNNModel, graph: bool=False) -> TrainingMetrics:
    if not config.train_dir.exists():
        LOGGER.warning(f"Training directory {config.train_dir} does not exist. Aborting training.")
        return model, TrainingMetrics()

    train_dataset = SelfPlayDataset(config.train_dir)

    if not config.val_dir.exists():
        LOGGER.warning(f"Validation directory {config.val_dir} does not exist. Continuing without validation.")
        val_dataset = None
    else:
        val_dataset = SelfPlayDataset(config.val_dir)

    metrics = train_model(model, train_dataset, val_dataset, config, graph)
    return model, metrics


def train_model(
    model: AlphaZeroGNNModel,
    train_dataset: Dataset,
    val_dataset: Optional[Dataset],
    config: TrainingConfig,
    graph: bool=False,
) -> TrainingMetrics:
    optimizer = optim.Adam(
        model.parameters(), 
        lr=config.lr, 
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    # Add learning rate scheduler for better convergence
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=5
    )
    metrics = TrainingMetrics()

    if len(train_dataset) == 0:
        LOGGER.warning("Dataset is empty. Skipping training.")
        return metrics

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle_examples,
        num_workers=2,  # Add parallel data loading
        pin_memory=False  # Speed up GPU transfer
    )
    val_loader = None
    if val_dataset is not None and len(val_dataset) > 0:
        val_loader = DataLoader(
            val_dataset, 
            batch_size=config.batch_size, 
            shuffle=False,  # No need to shuffle validation data
            num_workers=2,
            pin_memory=False
        )
    
    metrics.training_examples = len(train_dataset)
    device = model.device
    
    if graph:
        # Create figure once and reuse it
        plt.ion()
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Progress (Live - Batch Level)', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Create a dedicated output area for the plot
        plot_output = display(HTML("<div id='training-plot'></div>"), display_id=True)
    else:
        fig, axes = None, None

    global_batch_number = 0
    for epoch in range(config.num_epochs):
        model.train()
        train_losses = []
        train_policy_losses = []
        train_value_losses = []
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs} [Train]", leave=False)

        for batch in train_pbar:
            global_batch_number += 1
            game_state_data, batch_data, legal_action_mask, pi, value = batch
            # DataLoader adds an extra dimension
            game_state_data = game_state_data.to(device, non_blocking=True).squeeze(1)
            batch_data = batch_data.to(device, non_blocking=True)
            legal_action_mask = legal_action_mask.to(device, non_blocking=True)
            pi = pi.to(device, non_blocking=True)
            value = value.to(device, non_blocking=True)

            optimizer.zero_grad()
            _, move_log_probs, value_pred = model(game_state_data, batch_data)

            # Cross-entropy loss for policy
            # Mask illegal actions
            masked_log_probs = move_log_probs * legal_action_mask
            masked_log_probs = masked_log_probs + 1e-8
            policy_loss = -torch.sum(pi * masked_log_probs, dim=1).mean()
            # MSE loss for value
            value_loss = F.mse_loss(value_pred, value)
            total_loss = policy_loss + config.value_loss_weight * value_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(total_loss.item())
            train_policy_losses.append(policy_loss.item())
            train_value_losses.append(value_loss.item())
            
            # Add these lines to store batch-level metrics:
            metrics.batch_losses.append(total_loss.item())
            metrics.batch_policy_losses.append(policy_loss.item())
            metrics.batch_value_losses.append(value_loss.item())
            metrics.batch_numbers.append(global_batch_number)
            
            train_pbar.set_postfix({
                'loss': f'{total_loss.item():.4f}',
                'policy_loss': f'{policy_loss.item():.4f}',
                'value_loss': f'{value_loss.item():.4f}',
                'batch': global_batch_number,
            })

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
        
        # Validation phase
        val_total_loss = float('inf')
        val_policy_loss = float('inf')
        val_value_loss = float('inf')
        
        if val_loader is not None:
            model.eval()
            val_losses = []
            val_policy_losses = []
            val_value_losses = []
            
            with torch.no_grad():
                val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.num_epochs} [Val]")
                
                for batch in val_pbar:
                    game_state_data, batch_data, legal_action_mask, pi, value = batch
                    game_state_data = game_state_data.to(device, non_blocking=True).squeeze(1)
                    batch_data = batch_data.to(device, non_blocking=True)
                    legal_action_mask = legal_action_mask.to(device, non_blocking=True)
                    pi = pi.to(device, non_blocking=True)
                    value = value.to(device, non_blocking=True)

                    _, move_log_probs, value_pred = model(game_state_data, batch_data)

                    # Calculate validation losses
                    masked_log_probs = move_log_probs * legal_action_mask + 1e-8
                    policy_loss = -torch.sum(pi * masked_log_probs, dim=1).mean()
                    value_loss = F.mse_loss(value_pred, value)
                    total_loss = policy_loss + config.value_loss_weight * value_loss

                    val_losses.append(total_loss.item())
                    val_policy_losses.append(policy_loss.item())
                    val_value_losses.append(value_loss.item())
                    
                    val_pbar.set_postfix({
                        'val_loss': f'{total_loss.item():.4f}',
                        'val_policy_loss': f'{policy_loss.item():.4f}',
                        'val_value_loss': f'{value_loss.item():.4f}'
                    })

            val_total_loss = np.mean(val_losses)
            val_policy_loss = np.mean(val_policy_losses)
            val_value_loss = np.mean(val_value_losses)
            
            # Store validation metrics
            metrics.val_losses.append(val_total_loss)
            metrics.val_policy_losses.append(val_policy_loss)
            metrics.val_value_losses.append(val_value_loss)
            
            # Update learning rate scheduler based on validation loss
            scheduler.step(val_total_loss)
        
        # Log epoch summary
        LOGGER.info(
            f"Epoch {epoch+1}/{config.num_epochs} Summary:\n"
            f"  Train - Total: {avg_epoch_loss:.4f}, Policy: {avg_policy_loss:.4f}, Value: {avg_value_loss:.4f}\n"
            f"  Val   - Total: {val_total_loss:.4f}, Policy: {val_policy_loss:.4f}, Value: {val_value_loss:.4f}\n"
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
    
    if val_dataset is not None and metrics.val_losses:
        metrics.val_total_loss = metrics.val_losses[-1]  # Final validation loss
        metrics.val_policy_loss = metrics.val_policy_losses[-1]
        metrics.val_value_loss = metrics.val_value_losses[-1]
    
    # Final plot and save if requested
    if graph:
        plt.ioff()  # Turn off interactive mode
        plt.savefig('training_curves_final.png', dpi=300, bbox_inches='tight')
        LOGGER.info("Final training curves saved as 'training_curves_final.png'")
        # Don't call plt.show() in Jupyter - the figure is already displayed
    
    return metrics


def update_live_plot(metrics: TrainingMetrics, axes, current_epoch: int, total_epochs: int, current_batch: int, fig, plot_output):
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
            smoothed_losses = np.convolve(metrics.batch_losses, np.ones(window_size)/window_size, mode='valid')
            smoothed_policy_losses = np.convolve(metrics.batch_policy_losses, np.ones(window_size)/window_size, mode='valid')
            smoothed_value_losses = np.convolve(metrics.batch_value_losses, np.ones(window_size)/window_size, mode='valid')
            smoothed_batches = batch_numbers[window_size-1:]
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
        axes[0, 0].plot(batch_numbers, metrics.batch_losses, 'b-', label='Train Raw', linewidth=0.5, alpha=0.3)
        axes[0, 0].plot(smoothed_batches, smoothed_losses, 'b-', label='Train Smoothed', linewidth=2, alpha=0.8)
        # Add epoch averages
        if epoch_batches and epoch_batches[0] <= len(metrics.batch_numbers):
            axes[0, 0].plot(epoch_batches, metrics.epoch_losses, 'g-', label='Train Epoch Avg', linewidth=3, marker='o', markersize=6)
        # Add validation loss
        if metrics.val_losses and len(metrics.val_losses) <= len(epoch_batches):
            axes[0, 0].plot(epoch_batches[:len(metrics.val_losses)], metrics.val_losses, 'r-', label='Validation', linewidth=3, marker='s', markersize=6)
        axes[0, 0].set_title('Total Loss (Batch Level)')
        axes[0, 0].set_xlabel('Batch Number')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Policy Loss (batch level)
        axes[0, 1].plot(batch_numbers, metrics.batch_policy_losses, 'g-', label='Train Raw', linewidth=0.5, alpha=0.3)
        axes[0, 1].plot(smoothed_batches, smoothed_policy_losses, 'g-', label='Train Smoothed', linewidth=2, alpha=0.8)
        if metrics.epoch_policy_losses and epoch_batches:
            axes[0, 1].plot(epoch_batches, metrics.epoch_policy_losses, 'g-', label='Train Epoch Avg', linewidth=3, marker='o', markersize=6)
        if metrics.val_policy_losses and len(metrics.val_policy_losses) <= len(epoch_batches):
            axes[0, 1].plot(epoch_batches[:len(metrics.val_policy_losses)], metrics.val_policy_losses, 'r-', label='Validation', linewidth=3, marker='s', markersize=6)
        axes[0, 1].set_title('Policy Loss (Batch Level)')
        axes[0, 1].set_xlabel('Batch Number')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Value Loss (batch level)
        axes[1, 0].plot(batch_numbers, metrics.batch_value_losses, 'orange', label='Train Raw', linewidth=0.5, alpha=0.3)
        axes[1, 0].plot(smoothed_batches, smoothed_value_losses, 'orange', label='Train Smoothed', linewidth=2, alpha=0.8)
        if metrics.epoch_value_losses and epoch_batches:
            axes[1, 0].plot(epoch_batches, metrics.epoch_value_losses, 'orange', label='Train Epoch Avg', linewidth=3, marker='o', markersize=6)
        if metrics.val_value_losses and len(metrics.val_value_losses) <= len(epoch_batches):
            axes[1, 0].plot(epoch_batches[:len(metrics.val_value_losses)], metrics.val_value_losses, 'r-', label='Validation', linewidth=3, marker='s', markersize=6)
        axes[1, 0].set_title('Value Loss (Batch Level)')
        axes[1, 0].set_xlabel('Batch Number')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Combined view with all smoothed losses
        axes[1, 1].plot(smoothed_batches, smoothed_losses, 'b-', label='Train Total', linewidth=2, alpha=0.8)
        axes[1, 1].plot(smoothed_batches, smoothed_policy_losses, 'g-', label='Train Policy', linewidth=2, alpha=0.8)
        axes[1, 1].plot(smoothed_batches, smoothed_value_losses, 'orange', label='Train Value', linewidth=2, alpha=0.8)
        # Add validation losses to combined view
        if metrics.val_losses and len(metrics.val_losses) <= len(epoch_batches):
            axes[1, 1].plot(epoch_batches[:len(metrics.val_losses)], metrics.val_losses, 'r-', label='Val Total', linewidth=3, marker='s', markersize=6)
        if metrics.val_policy_losses and len(metrics.val_policy_losses) <= len(epoch_batches):
            axes[1, 1].plot(epoch_batches[:len(metrics.val_policy_losses)], metrics.val_policy_losses, 'darkred', label='Val Policy', linewidth=3, marker='^', markersize=6)
        if metrics.val_value_losses and len(metrics.val_value_losses) <= len(epoch_batches):
            axes[1, 1].plot(epoch_batches[:len(metrics.val_value_losses)], metrics.val_value_losses, 'crimson', label='Val Value', linewidth=3, marker='v', markersize=6)
        axes[1, 1].set_title('All Losses (Smoothed + Validation)')
        axes[1, 1].set_xlabel('Batch Number')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    # Add progress indicator to the main title
    plt.suptitle(f'Training Progress (Live) - Epoch {current_epoch}/{total_epochs}, Batch {current_batch}', 
                 fontsize=16, fontweight='bold')
    
    # Update the display
    plt.tight_layout()
    
    # Force a display update
    plot_output.update(fig)


def plot_training_curves(metrics: TrainingMetrics):
    """Plot training and validation loss curves using matplotlib (static version)"""
    epochs = range(1, len(metrics.epoch_losses) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Progress', fontsize=16, fontweight='bold')
    
    # Total Loss
    axes[0, 0].plot(epochs, metrics.epoch_losses, 'b-', label='Train', linewidth=2, marker='o', markersize=4)
    if metrics.val_losses:
        axes[0, 0].plot(epochs, metrics.val_losses, 'r-', label='Validation', linewidth=2, marker='s', markersize=4)
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Policy Loss
    axes[0, 1].plot(epochs, metrics.epoch_policy_losses, 'b-', label='Train', linewidth=2, marker='o', markersize=4)
    if metrics.val_policy_losses:
        axes[0, 1].plot(epochs, metrics.val_policy_losses, 'r-', label='Validation', linewidth=2, marker='s', markersize=4)
    axes[0, 1].set_title('Policy Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Value Loss
    axes[1, 0].plot(epochs, metrics.epoch_value_losses, 'b-', label='Train', linewidth=2, marker='o', markersize=4)
    if metrics.val_value_losses:
        axes[1, 0].plot(epochs, metrics.val_value_losses, 'r-', label='Validation', linewidth=2, marker='s', markersize=4)
    axes[1, 0].set_title('Value Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Combined view
    axes[1, 1].plot(epochs, metrics.epoch_losses, 'b-', label='Train Total', linewidth=2, marker='o', markersize=4)
    axes[1, 1].plot(epochs, metrics.epoch_policy_losses, 'b--', label='Train Policy', linewidth=1.5, marker='^', markersize=3)
    axes[1, 1].plot(epochs, metrics.epoch_value_losses, 'b:', label='Train Value', linewidth=1.5, marker='v', markersize=3)
    if metrics.val_losses:
        axes[1, 1].plot(epochs, metrics.val_losses, 'r-', label='Val Total', linewidth=2, marker='s', markersize=4)
        axes[1, 1].plot(epochs, metrics.val_policy_losses, 'r--', label='Val Policy', linewidth=1.5, marker='<', markersize=3)
        axes[1, 1].plot(epochs, metrics.val_value_losses, 'r:', label='Val Value', linewidth=1.5, marker='>', markersize=3)
    axes[1, 1].set_title('All Losses')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Save the plot
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    LOGGER.info("Training curves saved as 'training_curves.png'")
