import torch
from torch.utils.data import Dataset
import pickle
import os
import numpy as np
from typing import List, Tuple, Any
import torch.optim as optim
import torch.nn as nn
import logging
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from rl18xx.agent.alphazero.model import Model
from rl18xx.agent.alphazero.self_play import TrainingExample

# PyTorch Geometric imports
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyG_DataLoader

from rl18xx.game.gamemap import GameMap
from rl18xx.agent.alphazero.encoder import Encoder_1830
from rl18xx.agent.alphazero.action_mapper import ActionMapper


LOGGER = logging.getLogger(__name__)
TrainingExample = Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], np.ndarray, np.ndarray]


class SelfPlayDataset(Dataset):
    """
    Dataset for self-play training examples.
    Each example is converted into a PyTorch Geometric Data object.
    """

    def __init__(self, examples: List[TrainingExample]):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int) -> Data:  # Returns a PyG Data object
        state_tuple, policy_target_np, value_target_np = self.examples[idx]

        game_state_tensor, map_nodes_tensor, raw_edge_tensor = state_tuple

        # Convert numpy targets to tensors
        policy_target_tensor = torch.tensor(policy_target_np, dtype=torch.float32)
        value_target_tensor = torch.tensor(value_target_np, dtype=torch.float32)

        # Ensure policy_target_tensor is (1, num_actions)
        if policy_target_tensor.ndim == 1:
            policy_target_tensor = policy_target_tensor.unsqueeze(0)

        # Ensure value_target_tensor is (1, value_size)
        if value_target_tensor.ndim == 1:  # e.g. shape [V] or [1]
            value_target_tensor = value_target_tensor.unsqueeze(0)  # make it [1, V] or [1,1]

        node_features = map_nodes_tensor.float()
        edge_index = raw_edge_tensor[0:2, :].long()
        edge_attributes = raw_edge_tensor[2, :].long()

        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attributes,
            game_state=game_state_tensor.float(),  # Shape: (1, game_state_size)
            y_policy=policy_target_tensor,  # Expected Shape: (1, policy_size)
            y_value=value_target_tensor,  # Expected Shape: (1, value_size)
        )
        return data


def load_training_data(data_folder: str) -> List[TrainingExample]:
    all_examples = []
    for filename in os.listdir(data_folder):
        if filename.endswith(".pkl"):
            filepath = os.path.join(data_folder, filename)
            try:
                with open(filepath, "rb") as f:
                    examples = pickle.load(f)
                    all_examples.extend(examples)
                LOGGER.info(f"Loaded {len(examples)} examples from {filepath}")
            except Exception as e:
                LOGGER.error(f"Error loading data from {filepath}: {e}")
    return all_examples


def train_model(
    model: Model,
    dataset: SelfPlayDataset,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
    policy_loss_weight: float = 1.0,
    value_loss_weight: float = 1.0,
    checkpoint_dir: str = "model_checkpoints",
    save_every_n_epochs: int = 5,
    loop_iteration: int = -1,
):
    # --- TensorBoard Setup ---
    # Log to a subdirectory of checkpoint_dir, specific to this training session/loop iteration
    # Example: model_checkpoints_main_loop/iteration_0_training_checkpoints/runs/
    # or if loop_iteration is not passed, just checkpoint_dir/runs
    log_suffix = f"_iter_{loop_iteration}" if loop_iteration >= 0 else ""
    writer_log_dir = os.path.join(checkpoint_dir, f"runs{log_suffix}")
    writer = SummaryWriter(log_dir=writer_log_dir)
    LOGGER.info(f"TensorBoard logs for this training run will be saved to: {writer_log_dir}")

    model.to(device)
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    dataloader = PyG_DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    policy_loss_fn = nn.CrossEntropyLoss()
    value_loss_fn = nn.MSELoss()

    LOGGER.info(f"Starting training for {epochs} epochs on device: {device}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    for epoch in range(epochs):
        total_loss_epoch = 0.0
        total_policy_loss_epoch = 0.0
        total_value_loss_epoch = 0.0
        
        global_step_offset = epoch * len(dataloader) # For unique global steps across epochs

        for batch_idx, pyg_batch in enumerate(dataloader):
            pyg_batch = pyg_batch.to(device)

            game_states_batch = pyg_batch.game_state
            map_nodes_batch = pyg_batch.x
            edge_indices_batch = pyg_batch.edge_index
            node_to_graph_idx_batch = pyg_batch.batch
            edge_attrs_batch = pyg_batch.edge_attr

            policy_targets_batch = pyg_batch.y_policy
            value_targets_batch = pyg_batch.y_value

            optimizer.zero_grad()

            policy_logits, value_preds = model(
                game_states_batch, map_nodes_batch, edge_indices_batch, node_to_graph_idx_batch, edge_attrs_batch
            )

            loss_policy = policy_loss_fn(policy_logits, policy_targets_batch)
            loss_value = value_loss_fn(value_preds, value_targets_batch)

            current_batch_loss = (policy_loss_weight * loss_policy) + (value_loss_weight * loss_value)
            
            # --- TensorBoard Logging per Batch ---
            global_step = global_step_offset + batch_idx
            writer.add_scalar('Loss/Batch/Total', current_batch_loss.item(), global_step)
            writer.add_scalar('Loss/Batch/Policy', loss_policy.item(), global_step)
            writer.add_scalar('Loss/Batch/Value', loss_value.item(), global_step)

            current_batch_loss.backward()
            optimizer.step()

            total_loss_epoch += current_batch_loss.item()
            total_policy_loss_epoch += loss_policy.item()
            total_value_loss_epoch += loss_value.item()

            if batch_idx % 50 == 0:
                LOGGER.info(
                    f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(dataloader)}], "
                    f"Loss: {current_batch_loss.item():.4f} (Policy: {loss_policy.item():.4f}, Value: {loss_value.item():.4f})"
                )

        avg_loss_epoch = total_loss_epoch / len(dataloader)
        avg_policy_loss_epoch = total_policy_loss_epoch / len(dataloader)
        avg_value_loss_epoch = total_value_loss_epoch / len(dataloader)
        LOGGER.info(
            f"Epoch {epoch+1} Summary: Avg Loss: {avg_loss_epoch:.4f} "
            f"(Policy: {avg_policy_loss_epoch:.4f}, Value: {avg_value_loss_epoch:.4f})"
        )
        
        # --- TensorBoard Logging per Epoch ---
        writer.add_scalar('Loss/Epoch/Total', avg_loss_epoch, epoch)
        writer.add_scalar('Loss/Epoch/Policy', avg_policy_loss_epoch, epoch)
        writer.add_scalar('Loss/Epoch/Value', avg_value_loss_epoch, epoch)
        writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], epoch)

        if (epoch + 1) % save_every_n_epochs == 0 or (epoch + 1) == epochs:
            checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            LOGGER.info(f"Saved model checkpoint to {checkpoint_path}")

    LOGGER.info("Training finished.")
    writer.close() # Close the TensorBoard writer


# Code to test out the training method
if __name__ == "__main__":
    # --- Configuration ---
    DATA_DIR = "training_data"  # Where your .pkl files are
    CHECKPOINT_DIR = "model_checkpoints"
    NUM_EPOCHS = 50
    BATCH_SIZE = 64  # Adjust based on your GPU memory and dataset size
    LEARNING_RATE = 0.001
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load Data ---
    LOGGER.info(f"Loading training data from {DATA_DIR}...")
    training_examples = load_training_data(DATA_DIR)
    if not training_examples:
        LOGGER.error("No training data found. Exiting.")
        exit()
    LOGGER.info(f"Loaded a total of {len(training_examples)} training examples.")

    # --- Determine feature sizes from loaded data ---
    if not training_examples:
        LOGGER.error("No training data to derive sizes from. Exiting.")
        exit()

    # A TrainingExample is: (state_tuple, policy_target, value_target)
    # state_tuple is: (game_state_tensor, map_nodes_tensor, raw_edge_tensor)
    sample_state_tuple, sample_policy_target_np, sample_value_target_np = training_examples[0]
    sample_game_state_tensor, sample_map_nodes_tensor, sample_raw_edge_tensor = sample_state_tuple

    # game_state_tensor from file is (1, features)
    game_state_size = sample_game_state_tensor.shape[1]
    LOGGER.info(f"Determined game_state_size from data: {game_state_size}")

    # map_nodes_tensor is (num_nodes, node_features)
    if sample_map_nodes_tensor.ndim == 2:
        map_node_features = sample_map_nodes_tensor.shape[1]
    else:  # Should not happen based on data description
        LOGGER.error(f"Unexpected ndim for sample_map_nodes_tensor: {sample_map_nodes_tensor.ndim}")
        raise ValueError("Map nodes tensor has unexpected dimensions.")
    LOGGER.info(f"Determined map_node_features from data: {map_node_features}")

    # policy_target is np.ndarray (policy_size,)
    policy_output_size = sample_policy_target_np.shape[0]
    LOGGER.info(f"Determined policy_output_size from data: {policy_output_size}")

    # value_target is np.ndarray (value_size,)
    value_output_size = sample_value_target_np.shape[0]
    LOGGER.info(f"Determined value_output_size from data: {value_output_size}")

    # GNN edge categories and embedding dim (ensure these match your model's defaults or pass them)
    # These are typically fixed by the encoder and model architecture.
    # For now, assume model defaults are used (e.g., gnn_edge_categories=6)

    # --- Dataset and DataLoader ---
    dataset = SelfPlayDataset(training_examples)
    train_loader = PyG_DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)  # Adjust as needed

    # --- Model Setup ---
    game_title = "1830"
    game_map = GameMap()
    game_class_from_map = game_map.game_by_title(game_title)
    if not game_class_from_map:
        raise ValueError(f"Game class for '{game_title}' not found.")

    player_options = {"1": "Alice", "2": "Bob", "3": "Charlie", "4": "Dave"}
    temp_game_for_config = game_class_from_map(player_options)
    num_players_for_config = len(temp_game_for_config.players)
    encoder = Encoder_1830()
    action_mapper = ActionMapper()
    dummy_game_state_encoded, dummy_map_nodes_encoded, dummy_raw_edges_encoded = encoder.encode(temp_game_for_config)

    # game_state_tensor has shape (batch, features)
    game_state_size = dummy_game_state_encoded.shape[1]

    # map_nodes_tensor has shape (num_nodes, features) - unbatched
    num_map_nodes = dummy_map_nodes_encoded.shape[0]
    map_node_features = dummy_map_nodes_encoded.shape[1]

    policy_output_size = action_mapper.action_encoding_size
    value_output_size = num_players_for_config

    model = Model(
        game_state_size=game_state_size,
        map_node_features=map_node_features,
        policy_size=policy_output_size,
        value_size=value_output_size,
        # Add other necessary model parameters here if they are not defaults:
        # mlp_hidden_dim=...,
        # gnn_node_proj_dim=...,
        # ... etc.
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model_path_to_load = None
    if model_path_to_load and os.path.exists(model_path_to_load):
        try:
            model.load_state_dict(torch.load(model_path_to_load, map_location=DEVICE))
            LOGGER.info(f"Loaded model weights from {model_path_to_load}")
        except Exception as e:
            LOGGER.error(f"Error loading model from {model_path_to_load}: {e}. Training from scratch.")
            # Fallback to training from scratch if loading fails
    else:
        LOGGER.info(f"No model path specified or model not found at '{model_path_to_load}'. Training from scratch.")

    # --- Train ---
    train_model(
        model=model,
        dataset=dataset,
        epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        device=DEVICE,
        checkpoint_dir=CHECKPOINT_DIR,
    )
