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
from rl18xx.agent.alphazero.config import ModelConfig

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
        out = F.gelu(self.bn1(self.fc1(x)))
        out = self.dropout1(out)
        out = self.bn2(self.fc2(out))
        # Note: ReLU is applied after adding the residual, common in many ResNet variants
        out = F.gelu(out + residual)
        out = self.dropout2(out)
        return out


class FactoredPolicyHead(nn.Module):
    """Decomposes the LayTile action space into factored sub-heads with bilinear
    hex-tile interaction and attention-based hex scoring from per-node GNN embeddings.

    Sub-heads:
    - hex_scorer: scores each hex directly from its GNN node embedding (Phase 5.6)
    - hex_head: trunk_dim -> num_hexes (fallback when node embeddings unavailable)
    - tile_head: trunk_dim -> num_tiles (46)
    - rotation_head: trunk_dim -> num_rotations (6)
    - bilinear: captures hex-tile correlations via learned interaction matrix
    - other_head: trunk_dim -> num_other (867) for all non-LayTile actions

    The joint logit for LayTile(hex, tile, rotation) is:
        logit(h,t,r) = bilinear(hex_scores, tile_logits)[h,t] + rot_logits[r]

    The hex_scorer lets the policy reason "hex E14 is good because of what's on E14"
    rather than decoding hex preferences from a global pooled vector.
    """

    def __init__(self, trunk_dim: int, gnn_node_dim: int, policy_size: int, lay_tile_info: dict):
        super().__init__()
        self.lay_tile_offset = lay_tile_info["offset"]
        self.num_hexes = lay_tile_info["num_hexes"]
        self.num_tiles = lay_tile_info["num_tiles"]
        self.num_rotations = lay_tile_info["num_rotations"]
        self.num_lay_tile = lay_tile_info["num_lay_tile"]
        self.num_other = policy_size - self.num_lay_tile
        self.policy_size = policy_size

        # Phase 5.6: Attention-based hex scoring from per-node GNN embeddings
        self.hex_scorer = nn.Sequential(
            nn.Linear(gnn_node_dim, gnn_node_dim // 2),
            nn.GELU(),
            nn.Linear(gnn_node_dim // 2, 1),
        )
        # Fallback hex head from pooled trunk (used when node embeddings unavailable)
        self.hex_head = nn.Linear(trunk_dim, self.num_hexes)

        # Build permutation: action mapper hex index -> GNN node index
        # GNN nodes are in HEX_COORDS_ORDERED (sorted), action mapper has its own order
        from rl18xx.agent.alphazero.encoder import HEX_COORDS_ORDERED
        from rl18xx.agent.alphazero.action_mapper import ActionMapper
        action_mapper = ActionMapper()
        gnn_coord_to_idx = {coord: i for i, coord in enumerate(HEX_COORDS_ORDERED)}
        mapper_hex_coords = list(action_mapper.hex_offsets.keys())
        # mapper_to_gnn[i] = GNN node index for action mapper hex i
        self.register_buffer(
            "mapper_to_gnn",
            torch.tensor([gnn_coord_to_idx[coord] for coord in mapper_hex_coords], dtype=torch.long),
        )

        # LayTile factored sub-heads
        self.tile_head = nn.Linear(trunk_dim, self.num_tiles)
        self.rotation_head = nn.Linear(trunk_dim, self.num_rotations)

        # Bilinear interaction for hex-tile correlations
        self.bilinear = nn.Bilinear(self.num_hexes, self.num_tiles, self.num_hexes * self.num_tiles)

        # All non-LayTile actions (flat, small)
        self.other_head = nn.Sequential(
            nn.Linear(trunk_dim, trunk_dim // 2),
            nn.GELU(),
            nn.Linear(trunk_dim // 2, self.num_other),
        )

    def forward(
        self,
        x: Tensor,
        node_repr: Optional[Tensor] = None,
        batch_index: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            x: (B, trunk_dim) pooled trunk features
            node_repr: (total_nodes, gnn_node_dim) per-node GNN embeddings (optional)
            batch_index: (total_nodes,) PyG batch assignment per node (optional)
        """
        batch_size = x.shape[0]

        # Hex scoring: prefer per-node attention when available
        if node_repr is not None and batch_index is not None:
            # Score each node: (total_nodes, 1)
            node_scores = self.hex_scorer(node_repr).squeeze(-1)  # (total_nodes,)
            # Scatter into (B, num_hexes) using batch_index
            # Each graph has exactly num_hexes nodes in GNN order
            hex_scores_gnn_order = node_scores.view(batch_size, self.num_hexes)  # (B, num_hexes) in GNN order
            # Permute from GNN order to action mapper hex order
            hex_logits = hex_scores_gnn_order[:, self.mapper_to_gnn]  # (B, num_hexes) in mapper order
        else:
            hex_logits = self.hex_head(x)  # (B, H) fallback

        tile_logits = self.tile_head(x)  # (B, T)
        rot_logits = self.rotation_head(x)  # (B, R)

        # Bilinear interaction captures hex-tile correlations: (B, H*T)
        hex_tile_interaction = self.bilinear(hex_logits, tile_logits)
        hex_tile_interaction = hex_tile_interaction.view(batch_size, self.num_hexes, self.num_tiles)  # (B, H, T)

        # Combine: (B, H, T, 1) + (B, 1, 1, R) -> (B, H, T, R)
        lay_tile_logits = hex_tile_interaction.unsqueeze(3) + rot_logits.unsqueeze(1).unsqueeze(2)
        lay_tile_flat = lay_tile_logits.reshape(batch_size, -1)  # (B, num_lay_tile)

        # Non-LayTile actions
        other_logits = self.other_head(x)  # (B, num_other)

        # Assemble full policy vector with LayTile block at its correct offset
        other_before = self.lay_tile_offset
        full_logits = torch.cat(
            [
                other_logits[:, :other_before],
                lay_tile_flat,
                other_logits[:, other_before:],
            ],
            dim=1,
        )

        return full_logits


class AlphaZeroModel(nn.Module):
    def encoder_type(self):
        raise NotImplementedError("Subclasses must implement this method")

    def load_weights(self, save_file: str):
        try:
            with open(save_file, "rb") as f:
                state_dict = torch.load(f, map_location=self.device)
            self.load_state_dict(state_dict)
        except FileNotFoundError:
            LOGGER.error(f"Error: Weight file not found at {save_file}. Model weights remain as initialized.")
        except Exception as e:
            LOGGER.error(f"Error loading weights from {save_file}: {e}")

    def save_weights(self, save_file: str):
        try:
            with open(save_file, "wb") as f:
                torch.save(self.state_dict(), f)
            LOGGER.info(f"Successfully saved weights to {save_file}")
        except Exception as e:
            LOGGER.error(f"Error saving weights to {save_file}: {e}")

    def get_name(self):
        raise NotImplementedError("Subclasses must implement this method")

    def run(self, game_state: BaseGame) -> Tuple[Tensor, Tensor, Tensor]:
        raise NotImplementedError("Subclasses must implement this method")

    def run_encoded(self, encoded_game_state: Tuple[Tensor, Tensor, Tensor, Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        raise NotImplementedError("Subclasses must implement this method")

    def run_many(self, game_states: List[BaseGame]) -> Tuple[Tensor, Tensor, Tensor]:
        raise NotImplementedError("Subclasses must implement this method")

    def run_many_encoded(
        self, game_states: List[Tuple[Tensor, Tensor, Tensor, Tensor]]
    ) -> Tuple[Tensor, Tensor, Tensor]:
        raise NotImplementedError("Subclasses must implement this method")


class AlphaZeroGNNModel(AlphaZeroModel):
    """
    Neural Network model for an 18xx AlphaZero agent, incorporating a GNN for map data.
    """

    def __init__(self, config: ModelConfig):
        super(AlphaZeroGNNModel, self).__init__()
        self.config = config
        self.device = config.device
        self.encoder = Encoder_1830.get_encoder_for_model(self)
        self.init_model()
        self.to(self.device)

    def encoder_type(self):
        return "GNN"

    def init_model(self):
        # --- 1. Game State MLP Branch ---
        self.fc_game_state1 = nn.Linear(self.config.game_state_size, self.config.mlp_hidden_dim)
        self.bn_game_state1 = nn.BatchNorm1d(self.config.mlp_hidden_dim)
        self.dropout_gs1 = nn.Dropout(self.config.dropout_rate)
        self.fc_game_state2 = nn.Linear(self.config.mlp_hidden_dim, self.config.mlp_hidden_dim)
        self.bn_game_state2 = nn.BatchNorm1d(self.config.mlp_hidden_dim)
        self.dropout_gs2 = nn.Dropout(self.config.dropout_rate)
        self.map_node_features = self.config.map_node_features
        self.gnn_edge_categories = self.config.gnn_edge_categories

        # --- 2. Map GNN Branch (with residual connections — Phase 3.1) ---
        self.node_feature_initial_proj = nn.Linear(self.config.map_node_features, self.config.gnn_node_proj_dim)
        self.bn_node_initial_proj = nn.BatchNorm1d(self.config.gnn_node_proj_dim)

        if self.config.gnn_edge_categories > 0 and self.config.gnn_edge_embedding_dim > 0:
            self.edge_embedding = nn.Embedding(self.config.gnn_edge_categories, self.config.gnn_edge_embedding_dim)
            self.gnn_edge_dim_for_gat = self.config.gnn_edge_embedding_dim
        else:
            self.edge_embedding = None
            self.gnn_edge_dim_for_gat = None

        gat_out_dim = self.config.gnn_heads * self.config.gnn_hidden_dim_per_head
        self.gnn_layers_modulelist = nn.ModuleList()
        self.gnn_bn_layers = nn.ModuleList()
        self.gnn_dropout_layers = nn.ModuleList()
        # Residual projection for first layer (input dim != output dim)
        self.gnn_residual_proj = nn.Linear(self.config.gnn_node_proj_dim, gat_out_dim)

        current_gnn_input_dim = self.config.gnn_node_proj_dim
        for i in range(self.config.gnn_layers):
            gat_layer = GATv2Conv(
                current_gnn_input_dim,
                self.config.gnn_hidden_dim_per_head,
                heads=self.config.gnn_heads,
                concat=True,
                dropout=self.config.dropout_rate,
                edge_dim=self.gnn_edge_dim_for_gat,
            )
            self.gnn_layers_modulelist.append(gat_layer)
            self.gnn_bn_layers.append(nn.BatchNorm1d(gat_out_dim))
            self.gnn_dropout_layers.append(nn.Dropout(self.config.dropout_rate))
            current_gnn_input_dim = gat_out_dim

        self.gnn_final_node_proj = nn.Linear(current_gnn_input_dim, self.config.gnn_output_embed_dim)
        self.bn_gnn_final_node_proj = nn.BatchNorm1d(self.config.gnn_output_embed_dim)

        # --- 3. Gated Fusion Layer ---
        self.gs_proj = nn.Linear(self.config.mlp_hidden_dim, self.config.shared_trunk_hidden_dim)
        self.map_proj = nn.Linear(self.config.gnn_output_embed_dim, self.config.shared_trunk_hidden_dim)
        fused_input_dim = self.config.mlp_hidden_dim + self.config.gnn_output_embed_dim
        self.gate_fc = nn.Linear(fused_input_dim, self.config.shared_trunk_hidden_dim)
        self.bn_fusion = nn.BatchNorm1d(self.config.shared_trunk_hidden_dim)
        self.ln_fusion = nn.LayerNorm(self.config.shared_trunk_hidden_dim)  # Phase 5.5
        self.dropout_fusion = nn.Dropout(self.config.dropout_rate)

        # --- 4. Shared Trunk with FiLM phase conditioning (Phase 5.1) ---
        self.phase_embedding = nn.Embedding(self.config.num_round_types, self.config.film_embed_dim)
        self.film_layers = nn.ModuleList([
            nn.Linear(self.config.film_embed_dim, self.config.shared_trunk_hidden_dim * 2)
            for _ in range(self.config.num_res_blocks)
        ])
        self.res_blocks_modulelist = nn.ModuleList()
        for _ in range(self.config.num_res_blocks):
            self.res_blocks_modulelist.append(ResBlock(self.config.shared_trunk_hidden_dim, self.config.dropout_rate))

        # --- 5. Output Heads ---
        from rl18xx.agent.alphazero.action_mapper import ActionMapper

        action_mapper = ActionMapper()
        lay_tile_info = action_mapper.get_lay_tile_index_info()
        self.policy_head = FactoredPolicyHead(
            self.config.shared_trunk_hidden_dim,
            self.config.gnn_output_embed_dim,
            self.config.policy_size,
            lay_tile_info,
        )

        # Deeper value head with active-player signal (Phase 5.2 + 5.3)
        head_hidden = self.config.shared_trunk_hidden_dim // 2
        value_input_dim = self.config.shared_trunk_hidden_dim + self.config.value_size  # trunk + one-hot player
        value_layers = [nn.Linear(value_input_dim, head_hidden), nn.GELU()]
        for _ in range(self.config.value_head_layers - 2):
            value_layers.extend([nn.Linear(head_hidden, head_hidden), nn.GELU()])
        value_layers.append(nn.Linear(head_hidden, self.config.value_size))
        self.value_head = nn.Sequential(*value_layers)

        # Auxiliary head: predict log(legal_action_count) (Phase 5.4)
        self.aux_action_count_head = nn.Linear(self.config.shared_trunk_hidden_dim, 1)

        if self.config.model_checkpoint_file:
            self.load_weights(self.config.model_checkpoint_file)
        else:
            self.initialize_weights()

    def initialize_weights(self):
        """Initializes weights using Kaiming He initialization for GELU-activated layers.
        Uses leaky_relu as the closest supported approximation to GELU in PyTorch's kaiming init."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def get_name(self):
        return f"AlphaZeroModel_{self.config.timestamp}"

    def run(self, game_state: BaseGame) -> Tuple[Tensor, Tensor, Tensor]:
        probs, log_probs, values = self.run_many([game_state])
        return probs[0], log_probs[0], values[0]

    def run_encoded(self, encoded_game_state: Tuple[Tensor, Tensor, Tensor, Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        probs, log_probs, values = self.run_many_encoded([encoded_game_state])
        return probs[0], log_probs[0], values[0]

    def run_many(self, game_states: List[BaseGame]) -> Tuple[Tensor, Tensor, Tensor]:
        encoded_game_states = [self.encoder.encode(game_state) for game_state in game_states]
        return self.run_many_encoded(encoded_game_states)

    def run_many_encoded(
        self, game_states: List[tuple]
    ) -> Tuple[Tensor, Tensor, Tensor]:
        batch_size = len(game_states)

        if batch_size == 0:
            raise ValueError("Received no game states to run.")

        base_edge_index = game_states[0][2]
        base_edge_attributes = game_states[0][3]

        game_state_tensors = []
        graph_data_list = []
        round_type_indices = []
        active_player_indices = []
        for i in range(batch_size):
            gs = game_states[i]
            game_state_tensor, node_data = gs[0], gs[1]
            round_type_idx = gs[4] if len(gs) > 4 else 0
            active_player_idx = gs[5] if len(gs) > 5 else 0
            game_state_tensors.append(game_state_tensor.to(self.device))
            graph_data_list.append(
                Data(x=node_data, edge_index=base_edge_index, edge_attr=base_edge_attributes).to(self.device)
            )
            round_type_indices.append(round_type_idx)
            active_player_indices.append(active_player_idx)

        batched_game_state_tensor = torch.cat(game_state_tensors, dim=0)
        graph_batch = Batch.from_data_list(graph_data_list)
        graph_batch.validate(raise_on_error=True)
        round_type_tensor = torch.tensor(round_type_indices, dtype=torch.long, device=self.device)
        active_player_tensor = torch.tensor(active_player_indices, dtype=torch.long, device=self.device)

        policy_logits, value_logits, _ = self.forward(
            batched_game_state_tensor, graph_batch, round_type_tensor, active_player_tensor
        )

        # MCTS callers expect (probabilities, log_probs, value_logits)
        probabilities = F.softmax(policy_logits, dim=1)
        log_probs = F.log_softmax(policy_logits, dim=1)
        return probabilities, log_probs, value_logits

    def forward(
        self,
        game_state_data: Tensor,
        map_data: Batch,
        round_type_idx: Optional[Tensor] = None,
        active_player_idx: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Performs the forward pass of the network.
        Returns (policy_logits, value_logits, aux_action_count_pred).

        Args:
            game_state_data: (B, game_state_size) game state vectors
            map_data: PyG Batch of graph data
            round_type_idx: (B,) integer tensor of round type indices (0=Stock, 1=Operating, 2=Auction).
                If None, defaults to 0 (Stock) for all samples.
            active_player_idx: (B,) integer tensor of active player indices (0-3).
                If None, defaults to 0 for all samples.
        """
        batch_size = game_state_data.shape[0]

        if round_type_idx is None:
            round_type_idx = torch.zeros(batch_size, dtype=torch.long, device=game_state_data.device)
        if active_player_idx is None:
            active_player_idx = torch.zeros(batch_size, dtype=torch.long, device=game_state_data.device)

        # --- 1. Game State Embedding ---
        gs_embed = F.gelu(self.bn_game_state1(self.fc_game_state1(game_state_data.float())))
        gs_embed = self.dropout_gs1(gs_embed)
        gs_embed = F.gelu(self.bn_game_state2(self.fc_game_state2(gs_embed)))
        gs_embed = self.dropout_gs2(gs_embed)  # (B, mlp_hidden_dim)

        # --- 2. Map/Graph Embedding with residual connections (Phase 3.1) ---
        node_info, edge_index = map_data.x, map_data.edge_index
        node_repr = F.gelu(self.bn_node_initial_proj(self.node_feature_initial_proj(node_info.float())))

        processed_edge_attr = None
        if self.edge_embedding is not None and map_data.edge_attr is not None:
            if map_data.edge_attr.dtype != torch.long:
                map_data.edge_attr = map_data.edge_attr.long()
            processed_edge_attr = self.edge_embedding(map_data.edge_attr)

        for i, gat_layer in enumerate(self.gnn_layers_modulelist):
            residual = node_repr
            node_repr = gat_layer(node_repr, edge_index, edge_attr=processed_edge_attr)
            node_repr = self.gnn_bn_layers[i](node_repr)
            # Residual: first layer needs projection (dim mismatch), rest use identity
            if i == 0:
                residual = self.gnn_residual_proj(residual)
            node_repr = F.gelu(node_repr + residual)
            node_repr = self.gnn_dropout_layers[i](node_repr)

        node_repr_proj = F.gelu(self.bn_gnn_final_node_proj(self.gnn_final_node_proj(node_repr)))
        map_embed = global_mean_pool(node_repr_proj, map_data.batch)

        # --- 3. Gated Fusion with LayerNorm (Phase 5.5) ---
        gs_projected = F.gelu(self.gs_proj(gs_embed))
        map_projected = F.gelu(self.map_proj(map_embed))
        gate_input = torch.cat((gs_embed, map_embed), dim=1)
        gate = torch.sigmoid(self.gate_fc(gate_input))
        fused_features = gate * gs_projected + (1 - gate) * map_projected
        fused_features = self.bn_fusion(fused_features)
        fused_features = self.ln_fusion(fused_features)
        fused_features = self.dropout_fusion(fused_features)

        # --- 4. Shared Trunk with FiLM phase conditioning (Phase 5.1) ---
        phase_embed = self.phase_embedding(round_type_idx)  # (B, film_embed_dim)
        current_features = fused_features
        for i, res_block in enumerate(self.res_blocks_modulelist):
            current_features = res_block(current_features)
            film_params = self.film_layers[i](phase_embed)  # (B, trunk_dim * 2)
            gamma, beta = film_params.chunk(2, dim=1)
            current_features = gamma * current_features + beta

        # --- 5. Output Heads ---
        # Phase 5.6: pass per-node GNN embeddings for attention-based hex scoring
        policy_logits = self.policy_head(current_features, node_repr_proj, map_data.batch)

        # Value head with active-player indicator (Phase 5.2)
        player_indicator = F.one_hot(active_player_idx, num_classes=self.config.value_size).float()
        value_input = torch.cat([current_features, player_indicator], dim=1)
        value_logits = self.value_head(value_input)

        # Auxiliary head: predict log(legal_action_count) (Phase 5.4)
        aux_action_count_pred = self.aux_action_count_head(current_features)

        return policy_logits, value_logits, aux_action_count_pred


class AlphaZeroSSMEModel(AlphaZeroModel):
    """
    Neural Network model for an 18xx AlphaZero agent, incorporating a SSME for map data.
    """

    def __init__(self, config: ModelConfig):
        super(AlphaZeroSSMEModel, self).__init__()
        self.config = config
        self.device = config.device
        self.encoder = Encoder_1830.get_encoder_for_model(self)
        self.init_model()
        self.to(self.device)

    def encoder_type(self):
        return "SSME"

    def init_model(self):
        pass

        if self.config.model_checkpoint_file:
            self.load_weights(self.config.model_checkpoint_file)
        else:
            self.initialize_weights()

    def initialize_weights(self):
        """Initializes weights using Kaiming He initialization for GELU-activated layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def get_name(self):
        return f"AlphaZeroSSMEModel_{self.config.timestamp}"

    def run(self, game_state: BaseGame) -> Tuple[Tensor, Tensor, Tensor]:
        probs, log_probs, values = self.run_many([game_state])
        return probs[0], log_probs[0], values[0]

    def run_encoded(self, encoded_game_state: Tuple[Tensor, Tensor, Tensor, Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        probs, log_probs, values = self.run_many_encoded([encoded_game_state])
        return probs[0], log_probs[0], values[0]

    def run_many(self, game_states: List[BaseGame]) -> Tuple[Tensor, Tensor, Tensor]:
        encoded_game_states = [self.encoder.encode(game_state) for game_state in game_states]
        return self.run_many_encoded(encoded_game_states)

    def run_many_encoded(
        self, game_states: List[Tuple[Tensor, Tensor, Tensor, Tensor]]
    ) -> Tuple[Tensor, Tensor, Tensor]:
        batch_size = len(game_states)

        if batch_size == 0:
            raise ValueError("Received no game states to run.")

        pass

        # Run the model
        policy_logits, value_logits = self.forward(batched_game_state_tensor, graph_batch)

        # MCTS callers expect (probabilities, log_probs, value_logits)
        probabilities = F.softmax(policy_logits, dim=1)
        log_probs = F.log_softmax(policy_logits, dim=1)
        return probabilities, log_probs, value_logits

    def forward(self, game_state_data: Tensor, map_data: Batch) -> tuple[Tensor, Tensor]:
        raise NotImplementedError("SSME model forward pass not yet implemented")
