from rl18xx.agent.alphazero.v2.model import AlphaZeroModel
from rl18xx.agent.alphazero.v2.config import MegaConfig, ModelConfig
from rl18xx.agent.alphazero.v2.dataset import MCTSDataset
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from functools import partial
import logging

LOGGER = logging.getLogger(__name__)

def train(config: MegaConfig, model_config: ModelConfig):
    model = AlphaZeroModel(model_config)
    dataset = MCTSDataset(config.selfplay_dir)
    train_model(model, dataset, config, model_config.device)

def train_model(model: AlphaZeroModel, dataset: MCTSDataset, config: MegaConfig, device: torch.device):
    optimizer = optim.Adam(
        model.parameters(), 
        lr=config.lr, 
        weight_decay=config.weight_decay
    )

    if len(dataset) == 0:
        LOGGER.warning("Dataset is empty. Skipping training.")
        return

    # Ensure config.value_loss_weight is defined in your MegaConfig, e.g., config.value_loss_weight = 1.0
    value_loss_weight = getattr(config, 'value_loss_weight', 1.0)
    if not hasattr(config, 'value_loss_weight'):
        LOGGER.warning(f"config.value_loss_weight not found. Defaulting to {value_loss_weight}.")

    train_loader = DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)
    
    model.train()
    for epoch in range(config.num_epochs):
        total_epoch_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
        for i, batch in enumerate(progress_bar):
            game_state_data, data, pi, value = batch
            game_state_data = game_state_data.to(device)
            data = data.to(device)
            pi = pi.to(device)
            value = value.to(device)
            
            optimizer.zero_grad()
            move_probs, value_pred = model(game_state_data, data)
            
            # Policy loss (assuming move_probs are log-probabilities and pi are target indices or compatible)
            policy_loss = F.nll_loss(move_probs[data.train_mask], pi[data.train_mask])
            
            # Value loss (MSE)
            # Ensure value_pred is squeezed correctly, and mask if necessary
            # Assuming value_pred and value correspond to elements selected by data.train_mask
            value_pred_masked = value_pred[data.train_mask].squeeze(-1)
            value_masked = value[data.train_mask]
            value_loss = F.mse_loss(value_pred_masked, value_masked)
            
            total_loss = policy_loss + value_loss_weight * value_loss
            
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
        # Note: This evaluates on the training data loader.
        # For proper evaluation, a separate validation loader should be used.
        for batch in train_loader: # Consider using a validation_loader here
            game_state_data, data, pi, value = batch
            game_state_data = game_state_data.to(device)
            data = data.to(device)
            pi = pi.to(device)
            value = value.to(device)
            
            move_probs, value_pred = model(game_state_data, data)
            
            policy_loss = F.nll_loss(move_probs[data.train_mask], pi[data.train_mask])
            value_pred_masked = value_pred[data.train_mask].squeeze(-1)
            value_masked = value[data.train_mask]
            value_loss = F.mse_loss(value_pred_masked, value_masked)
            
            total_loss = policy_loss + value_loss_weight * value_loss

            eval_total_loss += total_loss.item()
            eval_policy_loss += policy_loss.item()
            eval_value_loss += value_loss.item()
            num_eval_batches += 1

        avg_eval_total_loss = eval_total_loss / num_eval_batches if num_eval_batches > 0 else 0
        avg_eval_policy_loss = eval_policy_loss / num_eval_batches if num_eval_batches > 0 else 0
        avg_eval_value_loss = eval_value_loss / num_eval_batches if num_eval_batches > 0 else 0
        LOGGER.info(f"Evaluation on train_loader after {config.num_epochs} epochs: "
                    f"Avg Total Loss: {avg_eval_total_loss:.4f}, "
                    f"Avg P Loss: {avg_eval_policy_loss:.4f}, "
                    f"Avg V Loss: {avg_eval_value_loss:.4f}")
