from rl18xx.agent.alphazero.model import AlphaZeroModel
from rl18xx.agent.alphazero.config import TrainingConfig
from rl18xx.agent.alphazero.dataset import MCTSDataset
from rl18xx.agent.alphazero.checkpointer import get_latest_model, save_model

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from typing import Optional
import logging
from dataclasses import dataclass, field

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


def train_latest_model(config: TrainingConfig) -> TrainingMetrics:
    model = get_latest_model(config.model_checkpoint_dir)
    config.train_dir = config.root_dir / f"selfplay/{model.get_name()}"
    config.val_dir = config.root_dir / f"holdout/{model.get_name()}"
    return train(config, model)


def train(config: TrainingConfig, model: AlphaZeroModel) -> TrainingMetrics:
    if not config.train_dir.exists():
        LOGGER.warning(f"Training directory {config.train_dir} does not exist. Aborting training.")
        return TrainingMetrics()

    train_dataset = MCTSDataset(config.train_dir)

    if not config.val_dir.exists():
        LOGGER.warning(f"Validation directory {config.val_dir} does not exist. Continuing without validation.")
        val_dataset = None
    else:
        val_dataset = MCTSDataset(config.val_dir)

    metrics = train_model(model, train_dataset, val_dataset, config, model.device)
    save_model(model, config.model_checkpoint_dir)
    return metrics


def train_model(
    model: AlphaZeroModel,
    train_dataset: MCTSDataset,
    val_dataset: Optional[MCTSDataset],
    config: TrainingConfig,
    device: torch.device,
) -> TrainingMetrics:
    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    metrics = TrainingMetrics()

    if len(train_dataset) == 0:
        LOGGER.warning("Dataset is empty. Skipping training.")
        return metrics

    train_loader = DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=config.shuffle_examples)
    
    metrics.training_examples = len(train_dataset)
    model.train()
    
    for epoch in range(config.num_epochs):
        total_epoch_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        num_batches = 0

        for batch in train_loader:
            game_state_data, batch_data, legal_action_mask, pi, value = batch
            # DataLoader adds an extra dimension
            game_state_data = game_state_data.to(device).squeeze(1)
            batch_data = batch_data.to(device)
            legal_action_mask = legal_action_mask.to(device)
            pi = pi.to(device)
            value = value.to(device)

            optimizer.zero_grad()
            _, move_log_probs, value_pred = model(game_state_data, batch_data)

            # Cross-entropy loss for policy
            # Mask illegal actions
            move_log_probs = move_log_probs * legal_action_mask
            policy_loss = -torch.sum(pi * move_log_probs, dim=1).mean()
            # MSE loss for value
            value_loss = F.mse_loss(value_pred, value)

            total_loss = policy_loss + config.value_loss_weight * value_loss

            total_loss.backward()
            optimizer.step()

            total_epoch_loss += total_loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            num_batches += 1

        avg_epoch_loss = total_epoch_loss / num_batches if num_batches > 0 else 0
        avg_policy_loss = total_policy_loss / num_batches if num_batches > 0 else 0
        avg_value_loss = total_value_loss / num_batches if num_batches > 0 else 0
        
        # Store epoch metrics
        metrics.epoch_losses.append(avg_epoch_loss)
        metrics.epoch_policy_losses.append(avg_policy_loss)
        metrics.epoch_value_losses.append(avg_value_loss)
        
        LOGGER.info(
            f"Epoch {epoch+1} Summary: Avg Total Loss: {avg_epoch_loss:.4f}, Avg P Loss: {avg_policy_loss:.4f}, Avg V Loss: {avg_value_loss:.4f}"
        )

    # Calculate average metrics across all epochs
    metrics.epochs_trained = config.num_epochs
    if metrics.epoch_losses:
        metrics.avg_total_loss = sum(metrics.epoch_losses) / len(metrics.epoch_losses)
        metrics.avg_policy_loss = sum(metrics.epoch_policy_losses) / len(metrics.epoch_policy_losses)
        metrics.avg_value_loss = sum(metrics.epoch_value_losses) / len(metrics.epoch_value_losses)
    
    if val_dataset is None:
        LOGGER.warning("No validation dataset provided. Skipping evaluation.")
        return metrics

    val_loader = DataLoader(val_dataset, batch_size=config.train_batch_size, shuffle=config.shuffle_examples)
    model.eval()
    with torch.no_grad():
        eval_total_loss = 0
        eval_policy_loss = 0
        eval_value_loss = 0
        num_eval_batches = 0
        for batch in val_loader:
            game_state_data, batch_data, legal_action_mask, pi, value = batch
            # DataLoader adds an extra dimension
            game_state_data = game_state_data.to(device).squeeze(1)
            batch_data = batch_data.to(device)
            legal_action_mask = legal_action_mask.to(device)
            pi = pi.to(device)
            value = value.to(device)

            _, move_log_probs, value_pred = model(game_state_data, batch_data)

            # Cross-entropy loss for policy
            move_log_probs = move_log_probs * legal_action_mask
            policy_loss = -torch.sum(pi * move_log_probs, dim=1).mean()
            # MSE loss for value
            value_loss = F.mse_loss(value_pred, value)

            total_loss = policy_loss + config.value_loss_weight * value_loss

            eval_total_loss += total_loss.item()
            eval_policy_loss += policy_loss.item()
            eval_value_loss += value_loss.item()
            num_eval_batches += 1

        avg_eval_total_loss = eval_total_loss / num_eval_batches if num_eval_batches > 0 else 0
        avg_eval_policy_loss = eval_policy_loss / num_eval_batches if num_eval_batches > 0 else 0
        avg_eval_value_loss = eval_value_loss / num_eval_batches if num_eval_batches > 0 else 0
        
        # Store validation metrics
        metrics.val_total_loss = avg_eval_total_loss
        metrics.val_policy_loss = avg_eval_policy_loss
        metrics.val_value_loss = avg_eval_value_loss
        
        LOGGER.info(
            f"Evaluation on val_loader after {config.num_epochs} epochs: "
            f"Avg Total Loss: {avg_eval_total_loss:.4f}, "
            f"Avg P Loss: {avg_eval_policy_loss:.4f}, "
            f"Avg V Loss: {avg_eval_value_loss:.4f}"
        )
    
    return metrics
