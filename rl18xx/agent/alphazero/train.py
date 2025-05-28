from rl18xx.agent.alphazero.model import AlphaZeroModel
from rl18xx.agent.alphazero.config import TrainingConfig
from rl18xx.agent.alphazero.dataset import MCTSDataset
from rl18xx.agent.alphazero.checkpointer import get_latest_model, save_model

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from functools import partial
from pathlib import Path
import logging

LOGGER = logging.getLogger(__name__)

def train_latest_model(config: TrainingConfig):
    model = get_latest_model(config.model_checkpoint_dir)
    config.train_dir = config.root_dir / f"selfplay/{model.get_name()}"
    config.val_dir = config.root_dir / f"holdout/{model.get_name()}"
    train(config, model)


def train(config: TrainingConfig, model: AlphaZeroModel):
    train_dataset = MCTSDataset(config.train_dir)
    val_dataset = MCTSDataset(config.val_dir)
    train_model(model, train_dataset, val_dataset, config, model.device)
    save_model(model, config.model_checkpoint_dir)


def train_model(
    model: AlphaZeroModel,
    train_dataset: MCTSDataset,
    val_dataset: MCTSDataset,
    config: TrainingConfig,
    device: torch.device
):
    optimizer = optim.Adam(
        model.parameters(), 
        lr=config.lr, 
        weight_decay=config.weight_decay
    )

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        LOGGER.warning("Dataset is empty. Skipping training.")
        return

    train_loader = DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=config.shuffle_examples)
    val_loader = DataLoader(val_dataset, batch_size=config.train_batch_size, shuffle=config.shuffle_examples)
    
    model.train()
    for epoch in range(config.num_epochs):
        total_epoch_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
        for i, batch in enumerate(progress_bar):
            game_state_data, batch_data, pi, value = batch
            # DataLoader adds an extra dimension
            game_state_data = game_state_data.to(device).squeeze(1)
            batch_data = batch_data.to(device)
            pi = pi.to(device)
            value = value.to(device)

            optimizer.zero_grad()
            _, move_log_probs, value_pred = model(game_state_data, batch_data)

            # Cross-entropy loss for policy
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
            
            progress_bar.set_postfix({
                "Total Loss": f"{total_loss.item():.4f}",
                "P Loss": f"{policy_loss.item():.4f}",
                "V Loss": f"{value_loss.item():.4f}"
            })

        avg_epoch_loss = total_epoch_loss / num_batches if num_batches > 0 else 0
        avg_policy_loss = total_policy_loss / num_batches if num_batches > 0 else 0
        avg_value_loss = total_value_loss / num_batches if num_batches > 0 else 0
        LOGGER.info(f"Epoch {epoch+1} Summary: Avg Total Loss: {avg_epoch_loss:.4f}, Avg P Loss: {avg_policy_loss:.4f}, Avg V Loss: {avg_value_loss:.4f}")

    model.eval()
    with torch.no_grad():
        eval_total_loss = 0
        eval_policy_loss = 0
        eval_value_loss = 0
        num_eval_batches = 0
        for batch in val_loader:
            game_state_data, batch_data, pi, value = batch
            # DataLoader adds an extra dimension
            game_state_data = game_state_data.to(device).squeeze(1)
            batch_data = batch_data.to(device)
            pi = pi.to(device)
            value = value.to(device)
            
            _, move_log_probs, value_pred = model(game_state_data, batch_data)

            # Cross-entropy loss for policy
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
        LOGGER.info(f"Evaluation on val_loader after {config.num_epochs} epochs: "
                    f"Avg Total Loss: {avg_eval_total_loss:.4f}, "
                    f"Avg P Loss: {avg_eval_policy_loss:.4f}, "
                    f"Avg V Loss: {avg_eval_value_loss:.4f}")
