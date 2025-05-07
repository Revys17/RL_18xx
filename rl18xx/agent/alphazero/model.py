from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool
# from torch_geometric.utils import add_self_loops # Optional: if self-loops are needed and not in edge_index

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

class Model(nn.Module):
    """
    Neural Network model for an 18xx AlphaZero agent, incorporating a GNN for map data.
    """
    def __init__(self,
                 game_state_size: int,
                 num_map_nodes: int,
                 map_node_features: int,
                 policy_size: int,
                 value_size: int,
                 # Architectural Hyperparameters
                 mlp_hidden_dim: int = 256,
                 gnn_node_proj_dim: int = 128,
                 gnn_hidden_dim_per_head: int = 64,
                 gnn_layers: int = 3,
                 gnn_heads: int = 4,
                 gnn_output_embed_dim: int = 256,
                 gnn_edge_categories: int = 6, # Number of unique edge types (e.g., 6 for hex edges 0-5)
                 gnn_edge_embedding_dim: int = 32, # Dimension of the edge feature embeddings
                 shared_trunk_hidden_dim: int = 512,
                 num_res_blocks: int = 5,
                 dropout_rate: float = 0.1):
        """
        Initializes the layers of the neural network.
        """
        super(Model, self).__init__()

        self.num_map_nodes = num_map_nodes
        self.map_node_features = map_node_features
        self.gnn_edge_categories = gnn_edge_categories # Store for forward pass check

        # --- 1. Game State MLP Branch ---
        self.fc_game_state1 = nn.Linear(game_state_size, mlp_hidden_dim)
        self.bn_game_state1 = nn.BatchNorm1d(mlp_hidden_dim)
        self.dropout_gs1 = nn.Dropout(dropout_rate)
        self.fc_game_state2 = nn.Linear(mlp_hidden_dim, mlp_hidden_dim) # Output embedding for game state
        self.bn_game_state2 = nn.BatchNorm1d(mlp_hidden_dim)
        self.dropout_gs2 = nn.Dropout(dropout_rate)

        # --- 2. Map GNN Branch ---
        # Initial linear projection for node features
        self.node_feature_initial_proj = nn.Linear(map_node_features, gnn_node_proj_dim)
        self.bn_node_initial_proj = nn.BatchNorm1d(gnn_node_proj_dim)
                
        # Edge Feature Embedding Layer
        if self.gnn_edge_categories > 0 and gnn_edge_embedding_dim > 0:
            self.edge_embedding = nn.Embedding(gnn_edge_categories, gnn_edge_embedding_dim)
            self.gnn_edge_dim_for_gat = gnn_edge_embedding_dim
        else:
            self.edge_embedding = None
            self.gnn_edge_dim_for_gat = None # GAT will not use edge features

        self.gnn_layers_modulelist = nn.ModuleList()
        current_gnn_input_dim = gnn_node_proj_dim
        for i in range(gnn_layers):
            # Output of GATv2Conv is (num_nodes, heads * gnn_hidden_dim_per_head)
            gat_layer = GATv2Conv(current_gnn_input_dim,
                                  gnn_hidden_dim_per_head,
                                  heads=gnn_heads,
                                  concat=True,
                                  dropout=dropout_rate,
                                  edge_dim=self.gnn_edge_dim_for_gat # Pass edge feature dimension
                                 )
            self.gnn_layers_modulelist.append(gat_layer)
            current_gnn_input_dim = gnn_heads * gnn_hidden_dim_per_head
            # Add BatchNorm and ReLU after each GAT layer
            self.gnn_layers_modulelist.append(nn.BatchNorm1d(current_gnn_input_dim))
            self.gnn_layers_modulelist.append(nn.ReLU())
            self.gnn_layers_modulelist.append(nn.Dropout(dropout_rate))


        # Projection from final GNN node embeddings to the desired gnn_output_embed_dim before pooling
        self.gnn_final_node_proj = nn.Linear(current_gnn_input_dim, gnn_output_embed_dim)
        self.bn_gnn_final_node_proj = nn.BatchNorm1d(gnn_output_embed_dim)


        # --- 3. Fusion Layer ---
        # Dimension of concatenated embeddings from game state MLP and map GNN
        fused_input_dim = mlp_hidden_dim + gnn_output_embed_dim
        self.fusion_fc = nn.Linear(fused_input_dim, shared_trunk_hidden_dim)
        self.bn_fusion = nn.BatchNorm1d(shared_trunk_hidden_dim)
        self.dropout_fusion = nn.Dropout(dropout_rate)

        # --- 4. Shared Trunk (Residual Blocks) ---
        self.res_blocks_modulelist = nn.ModuleList()
        for _ in range(num_res_blocks):
            self.res_blocks_modulelist.append(ResBlock(shared_trunk_hidden_dim, dropout_rate))

        # --- 5. Output Heads ---
        self.policy_head = nn.Linear(shared_trunk_hidden_dim, policy_size)
        self.value_head = nn.Linear(shared_trunk_hidden_dim, value_size)

        self._initialize_weights()

    def _initialize_weights(self):
        """Initializes weights using Kaiming He initialization for ReLU-activated layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            # GATv2Conv layers have their own initialization, often Xavier, which is usually fine.
            # You could add specific initialization for GATv2Conv.lin_l, .lin_r, .att if needed.

    def forward(self,
                x_game_state: Tensor,
                x_map_nodes: Tensor,
                edge_index: Tensor,
                edge_attr_categorical: Tensor = None # Categorical edge features (num_edges,)
               ) -> tuple[Tensor, Tensor]:
        """
        Performs the forward pass of the network.

        Args:
            x_game_state (torch.Tensor): Encoded flat game state (batch_size, game_state_size).
            x_map_nodes (torch.Tensor): Encoded map node features (batch_size, num_map_nodes, map_node_features).
            edge_index (torch.Tensor): Graph connectivity in COO format (2, num_edges).
                                       Assumed to be for a single graph, shared across the batch.
            edge_attr_categorical (torch.Tensor, optional): Categorical edge type indices (num_edges,).
                                                        Expected to be long/int type.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - policy_log_probs (torch.Tensor): Log probabilities for each action (batch_size, policy_size).
                - value_estimates (torch.Tensor): Estimated state value for each player (batch_size, value_size).
        """
        batch_size = x_game_state.shape[0]

        # --- 1. Game State MLP Branch ---
        gs_embed = F.relu(self.bn_game_state1(self.fc_game_state1(x_game_state.float())))
        gs_embed = self.dropout_gs1(gs_embed)
        gs_embed = F.relu(self.bn_game_state2(self.fc_game_state2(gs_embed)))
        gs_embed = self.dropout_gs2(gs_embed) # Shape: (batch_size, mlp_hidden_dim)

        # --- 2. Map GNN Branch ---
        # Reshape map_nodes for PyG: (batch_size * num_map_nodes, map_node_features)
        map_node_flat = x_map_nodes.float().reshape(-1, self.map_node_features)

        # Create batch vector for PyG global pooling (maps each node to its graph in the batch)
        batch_indices_for_pool = torch.arange(batch_size, device=x_map_nodes.device).repeat_interleave(self.num_map_nodes)

        # Initial node feature projection
        node_repr = F.relu(self.bn_node_initial_proj(self.node_feature_initial_proj(map_node_flat)))

        # Prepare edge features if provided
        processed_edge_attr = None
        if self.edge_embedding is not None and edge_attr_categorical is not None:
            if edge_attr_categorical.ndim == 2 and edge_attr_categorical.shape[1] == 1:
                edge_attr_categorical = edge_attr_categorical.squeeze(1) # Ensure (num_edges,)
            if edge_attr_categorical.dtype != torch.long:
                 # Ensure it's long for nn.Embedding, but be careful with device placement
                edge_attr_categorical = edge_attr_categorical.long()
            processed_edge_attr = self.edge_embedding(edge_attr_categorical)
        elif self.gnn_edge_dim_for_gat is not None and edge_attr_categorical is None:
            # This case means model expects edge features but none were provided.
            # Depending on strictness, could raise an error or proceed without them if GAT handles None edge_dim.
            # For GATv2Conv, if edge_dim is set, edge_attr must be provided.
            raise ValueError("Model configured to use edge features (gnn_edge_embedding_dim > 0), but edge_attr_categorical was not provided.")

        # Pass through GNN layers
        for i in range(0, len(self.gnn_layers_modulelist), 4): # GAT, BN, ReLU, Dropout
            gat_layer = self.gnn_layers_modulelist[i]
            bn_layer = self.gnn_layers_modulelist[i+1]
            relu_layer = self.gnn_layers_modulelist[i+2]
            dropout_layer = self.gnn_layers_modulelist[i+3]

            node_repr = gat_layer(node_repr, edge_index, edge_attr=processed_edge_attr)
            node_repr = relu_layer(bn_layer(node_repr))
            node_repr = dropout_layer(node_repr)
        
        # Final projection for node embeddings before pooling
        node_repr_proj = F.relu(self.bn_gnn_final_node_proj(self.gnn_final_node_proj(node_repr)))

        # Global pooling to get a graph-level embedding
        map_embed = global_mean_pool(node_repr_proj, batch_indices_for_pool) # Shape: (batch_size, gnn_output_embed_dim)

        # --- 3. Fusion ---
        combined_embed = torch.cat((gs_embed, map_embed), dim=1)
        fused_features = F.relu(self.bn_fusion(self.fusion_fc(combined_embed)))
        fused_features = self.dropout_fusion(fused_features) # Shape: (batch_size, shared_trunk_hidden_dim)

        # --- 4. Shared Trunk ---
        current_features = fused_features
        for res_block in self.res_blocks_modulelist:
            current_features = res_block(current_features)
        
        # --- 5. Output Heads ---
        policy_logits = self.policy_head(current_features)
        policy_log_probs = F.log_softmax(policy_logits, dim=1)

        value_estimates = torch.tanh(self.value_head(current_features)) # tanh to keep values in [-1, 1]

        return policy_log_probs, value_estimates
