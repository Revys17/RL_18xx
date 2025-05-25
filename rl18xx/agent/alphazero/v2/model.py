import logging
from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATv2Conv, global_mean_pool
from typing import Optional, List, Tuple

from rl18xx.game.engine.game.base import BaseGame
from rl18xx.agent.alphazero.encoder import Encoder_1830
from rl18xx.agent.alphazero.v2.config import ModelConfig

LOGGER = logging.getLogger(__name__)

class ResBlock(nn.Module):
    """
    A simple residual block with two linear layers.
    """

    def __init__(self, channels: int, dropout_rate: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels)
        self.bn1 = nn.BatchNorm1d(channels)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(channels, channels)
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        out = F.relu(self.bn1(self.fc1(x)))
        out = self.dropout1(out)
        out = self.bn2(self.fc2(out))
        # Note: ReLU is applied after adding the residual, common in many ResNet variants
        out = F.relu(out + residual)
        out = self.dropout2(out)
        return out


class AlphaZeroModel(nn.Module):
    """
    Neural Network model for an 18xx AlphaZero agent, incorporating a GNN for map data.
    """
    def __init__(self, config: ModelConfig):
        super(AlphaZeroModel, self).__init__()
        self.config = config
        self.device = config.device
        self.encoder = Encoder_1830()
        self.init_model()
        self.to(self.device)

    def init_model(self):
        # --- 1. Game State MLP Branch ---
        self.fc_game_state1 = nn.Linear(self.config.game_state_size, self.config.mlp_hidden_dim)
        self.bn_game_state1 = nn.BatchNorm1d(self.config.mlp_hidden_dim)
        self.dropout_gs1 = nn.Dropout(self.config.dropout_rate)
        self.fc_game_state2 = nn.Linear(self.config.mlp_hidden_dim, self.config.mlp_hidden_dim)  # Output embedding for game state
        self.bn_game_state2 = nn.BatchNorm1d(self.config.mlp_hidden_dim)
        self.dropout_gs2 = nn.Dropout(self.config.dropout_rate)
        self.map_node_features = self.config.map_node_features  # Keep for clarity if needed, though also implicit
        self.gnn_edge_categories = self.config.gnn_edge_categories

        # --- 2. Map GNN Branch ---
        # Initial linear projection for node features
        self.node_feature_initial_proj = nn.Linear(self.config.map_node_features, self.config.gnn_node_proj_dim)
        self.bn_node_initial_proj = nn.BatchNorm1d(self.config.gnn_node_proj_dim)

        # Edge Feature Embedding Layer
        if self.config.gnn_edge_categories > 0 and self.config.gnn_edge_embedding_dim > 0:
            self.edge_embedding = nn.Embedding(self.config.gnn_edge_categories, self.config.gnn_edge_embedding_dim)
            self.gnn_edge_dim_for_gat = self.config.gnn_edge_embedding_dim
        else:
            self.edge_embedding = None
            self.gnn_edge_dim_for_gat = None  # GAT will not use edge features
        
        self.gnn_layers_modulelist = nn.ModuleList()
        current_gnn_input_dim = self.config.gnn_node_proj_dim
        for i in range(self.config.gnn_layers):
            # Output of GATv2Conv is (num_nodes, heads * gnn_hidden_dim_per_head)
            gat_layer = GATv2Conv(
                current_gnn_input_dim,
                self.config.gnn_hidden_dim_per_head,
                heads=self.config.gnn_heads,
                concat=True,
                dropout=self.config.dropout_rate,
                edge_dim=self.gnn_edge_dim_for_gat,  # Pass edge feature dimension
            )
            self.gnn_layers_modulelist.append(gat_layer)
            current_gnn_input_dim = self.config.gnn_heads * self.config.gnn_hidden_dim_per_head
            # Add BatchNorm and ReLU after each GAT layer
            self.gnn_layers_modulelist.append(nn.BatchNorm1d(current_gnn_input_dim))
            self.gnn_layers_modulelist.append(nn.ReLU())
            self.gnn_layers_modulelist.append(nn.Dropout(self.config.dropout_rate))

        # Projection from final GNN node embeddings to the desired gnn_output_embed_dim before pooling
        self.gnn_final_node_proj = nn.Linear(current_gnn_input_dim, self.config.gnn_output_embed_dim)
        self.bn_gnn_final_node_proj = nn.BatchNorm1d(self.config.gnn_output_embed_dim)

        # --- 3. Fusion Layer ---
        # Dimension of concatenated embeddings from game state MLP and map GNN
        fused_input_dim = self.config.mlp_hidden_dim + self.config.gnn_output_embed_dim
        self.fusion_fc = nn.Linear(fused_input_dim, self.config.shared_trunk_hidden_dim)
        self.bn_fusion = nn.BatchNorm1d(self.config.shared_trunk_hidden_dim)
        self.dropout_fusion = nn.Dropout(self.config.dropout_rate)

        # --- 4. Shared Trunk (Residual Blocks) ---
        self.res_blocks_modulelist = nn.ModuleList()
        for _ in range(self.config.num_res_blocks):
            self.res_blocks_modulelist.append(ResBlock(self.config.shared_trunk_hidden_dim, self.config.dropout_rate))

        # --- 5. Output Heads ---
        self.policy_head = nn.Linear(self.config.shared_trunk_hidden_dim, self.config.policy_size)
        self.value_head = nn.Linear(self.config.shared_trunk_hidden_dim, self.config.value_size)

        if self.config.model_checkpoint_file:
            self.load_weights(self.config.model_checkpoint_file)
        else:
            self.initialize_weights()

    def initialize_weights(self):
        """Initializes weights using Kaiming He initialization for ReLU-activated layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def load_weights(self, save_file: str):
        try:
            state_dict = torch.load(save_file, map_location=self.device)
            self.load_state_dict(state_dict)
        except FileNotFoundError:
            LOGGER.error(f"Error: Weight file not found at {save_file}. Model weights remain as initialized.")
        except Exception as e:
            LOGGER.error(f"Error loading weights from {save_file}: {e}")

    def save_weights(self, save_file: str):
        try:
            torch.save(self.state_dict(), save_file)
            LOGGER.info(f"Successfully saved weights to {save_file}")
        except Exception as e:
            LOGGER.error(f"Error saving weights to {save_file}: {e}")

    def run(self, game_state: BaseGame) -> Tuple[Tensor, Tensor, Tensor]:
        probs, log_probs, values = self.run_many([game_state])
        return probs[0], log_probs[0], values[0]

    def run_encoded(self, encoded_game_state: Tuple[Tensor, Tensor, Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        probs, log_probs, values = self.run_many_encoded([encoded_game_state])
        return probs[0], log_probs[0], values[0]

    def run_many(self, game_states: List[BaseGame]) -> Tuple[Tensor, Tensor, Tensor]:
        encoded_game_states = [self.encoder.encode(game_state) for game_state in game_states]
        return self.run_many_encoded(encoded_game_states)

    def run_many_encoded(self, game_states: List[Tuple[Tensor, Tensor, Tensor]]) -> Tuple[Tensor, Tensor, Tensor]:
        batch_size = len(game_states)

        if batch_size == 0:
            raise ValueError("Received no game states to run.")

        # Create `batch_size` Data objects
        edge_index_tensor = game_states[0][2]

        # These values never change between game states, so only calculate once
        base_edge_index = edge_index_tensor[0:2, :].long()
        base_edge_attributes = edge_index_tensor[2, :].long()

        # Accumulate game state tensors and graph data
        game_state_tensors = []
        graph_data_list = []
        for i in range(batch_size):
            game_state_tensor, node_data, _ = game_states[i]
            game_state_tensors.append(game_state_tensor.to(self.device))
            graph_data_list.append(
                Data(
                    x=node_data,
                    edge_index=base_edge_index,
                    edge_attr=base_edge_attributes
                ).to(self.device)
            )

        # Create a Batch object from the list of graph data
        batched_game_state_tensor = torch.cat(game_state_tensors, dim=0)
        graph_batch = Batch.from_data_list(graph_data_list)

        # Verify graph data
        graph_batch.validate(raise_on_error=True)

        # Run the model
        outputs = self.forward(
            batched_game_state_tensor,
            graph_batch
        )

        # Extract outputs
        probabilities, log_probs, value = outputs[0], outputs[1], outputs[2]
        return probabilities, log_probs, value

    def forward(
        self,
        game_state_data: Tensor,
        map_data: Batch
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Performs the forward pass of the network.
        Assumes graph inputs are already batched in PyTorch Geometric style.
        """

        # --- 1. Game State Embedding ---
        gs_embed = F.relu(self.bn_game_state1(self.fc_game_state1(game_state_data.float())))
        gs_embed = self.dropout_gs1(gs_embed)
        gs_embed = F.relu(self.bn_game_state2(self.fc_game_state2(gs_embed)))
        gs_embed = self.dropout_gs2(gs_embed)  # Shape: (batch_size, mlp_hidden_dim)

        # --- 2. Map/Graph Embedding ---
        # Graph inputs are already in PyG batched format.
        # Initial node feature projection
        node_info, edge_index = map_data.x, map_data.edge_index
        node_repr = F.relu(self.bn_node_initial_proj(self.node_feature_initial_proj(node_info.float())))

        processed_edge_attr = None
        if self.edge_embedding is not None and map_data.edge_attr is not None:
            if map_data.edge_attr.dtype != torch.long:
                map_data.edge_attr = map_data.edge_attr.long()
            processed_edge_attr = self.edge_embedding(map_data.edge_attr)

        # Pass through GNN layers
        for i in range(0, len(self.gnn_layers_modulelist), 4):  # GAT, BN, ReLU, Dropout
            gat_layer = self.gnn_layers_modulelist[i]
            bn_layer = self.gnn_layers_modulelist[i + 1]
            relu_layer = self.gnn_layers_modulelist[i + 2]
            dropout_layer = self.gnn_layers_modulelist[i + 3]

            node_repr = gat_layer(node_repr, edge_index, edge_attr=processed_edge_attr)
            node_repr = relu_layer(bn_layer(node_repr))
            node_repr = dropout_layer(node_repr)

        # Final projection for node embeddings before pooling
        node_repr_proj = F.relu(self.bn_gnn_final_node_proj(self.gnn_final_node_proj(node_repr)))

        # Global pooling to get a graph-level embedding
        map_embed = global_mean_pool(node_repr_proj, map_data.batch)

        # --- 3. Fusion ---
        combined_embed = torch.cat((gs_embed, map_embed), dim=1)
        fused_features = F.relu(self.bn_fusion(self.fusion_fc(combined_embed)))
        fused_features = self.dropout_fusion(fused_features)  # Shape: (batch_size, shared_trunk_hidden_dim)

        # --- 4. Shared Trunk ---
        current_features = fused_features
        for res_block in self.res_blocks_modulelist:
            current_features = res_block(current_features)

        # --- 5. Output Heads ---
        policy_logits = self.policy_head(current_features)
        probabilities = F.softmax(policy_logits, dim=1)
        policy_log_probs = F.log_softmax(policy_logits, dim=1)

        value_estimates = torch.tanh(self.value_head(current_features))  # tanh to keep values in [-1, 1]

        return probabilities, policy_log_probs, value_estimates
