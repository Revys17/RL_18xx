"""
AlphaZero v2 Model Architecture for 1830.

Replaces the GNN-based v1 with:
- Hex Map Transformer (global attention over 93 hexes with structural bias)
- Economic State Transformer (entity-group attention over players/corps/privates)
- Cross-modal fusion (economic entities attend to map nodes)
- FiLM-inside residual trunk (phase conditioning inside blocks)
- Simplified policy head (outer-sum factoring, ~500K vs 18.8M params)

See docs/network_architecture_v2.md for full design rationale.
"""

import math
import logging
from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from rl18xx.agent.alphazero.config import ModelV2Config
from rl18xx.agent.alphazero.model import AlphaZeroModel

LOGGER = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Hex coordinate utilities
# ──────────────────────────────────────────────────────────────────────────────


def hex_coord_to_axial(coord_str: str) -> tuple[float, float]:
    """Convert 18xx hex coordinate string (e.g., 'E14') to axial (q, r)."""
    letter_part = ""
    num_part = ""
    for c in coord_str:
        if c.isalpha():
            letter_part += c
        else:
            num_part += c
    row = ord(letter_part.upper()) - ord("A")
    col = int(num_part)
    q = col
    r = row - (col // 2)
    return (float(q), float(r))


# ──────────────────────────────────────────────────────────────────────────────
# Entity group index definitions for 4-player 1830 (390-dim game state vector)
# ──────────────────────────────────────────────────────────────────────────────

NUM_PLAYERS = 4
NUM_CORPORATIONS = 8
NUM_PRIVATES = 6
NUM_TRAIN_TYPES = 6
NUM_TILE_IDS = 46

# Offsets in the 390-dim game state vector (for 4 players):
_OFF = {
    "active_entity": 0,  # size 12 (4 players + 8 corps)
    "active_president": 12,  # size 4
    "round_type": 16,  # size 1
    "game_phase": 17,  # size 1
    "priority_deal": 18,  # size 4
    "bank_cash": 22,  # size 1
    "player_certs": 23,  # size 4
    "player_cash": 27,  # size 4
    "player_shares": 31,  # size 32 (4*8)
    "private_ownership": 63,  # size 72 (6*12)
    "private_revenue": 135,  # size 6
    "corp_floated": 141,  # size 8
    "corp_cash": 149,  # size 8
    "corp_trains": 157,  # size 48 (8*6)
    "corp_tokens": 205,  # size 8
    "corp_share_price": 213,  # size 16 (8*2)
    "corp_shares": 229,  # size 16 (8*2)
    "corp_market_zone": 245,  # size 32 (8*4)
    "depot_trains": 277,  # size 6
    "market_pool_trains": 283,  # size 6
    "depot_tiles": 289,  # size 46
    "auction_bids": 335,  # size 24 (6*4)
    "auction_min_bid": 359,  # size 6
    "auction_available": 365,  # size 6
    "auction_face_value": 371,  # size 6
    "or_structure": 377,  # size 2
    "train_limit": 379,  # size 1
    "private_closed": 380,  # size 6
    "player_turn_order": 386,  # size 4
}


def _build_player_indices() -> list[list[int]]:
    """Build gather indices for each player entity group (14 features each)."""
    groups = []
    for i in range(NUM_PLAYERS):
        idx = [
            _OFF["active_entity"] + i,
            _OFF["active_president"] + i,
            _OFF["priority_deal"] + i,
            _OFF["player_certs"] + i,
            _OFF["player_cash"] + i,
        ]
        idx.extend(range(_OFF["player_shares"] + i * NUM_CORPORATIONS, _OFF["player_shares"] + (i + 1) * NUM_CORPORATIONS))
        idx.append(_OFF["player_turn_order"] + i)
        groups.append(idx)
    return groups


def _build_corp_indices() -> list[list[int]]:
    """Build gather indices for each corporation entity group (18 features each)."""
    groups = []
    for j in range(NUM_CORPORATIONS):
        idx = [
            _OFF["active_entity"] + NUM_PLAYERS + j,
            _OFF["corp_floated"] + j,
            _OFF["corp_cash"] + j,
        ]
        idx.extend(range(_OFF["corp_trains"] + j * NUM_TRAIN_TYPES, _OFF["corp_trains"] + (j + 1) * NUM_TRAIN_TYPES))
        idx.append(_OFF["corp_tokens"] + j)
        idx.extend(range(_OFF["corp_share_price"] + j * 2, _OFF["corp_share_price"] + (j + 1) * 2))
        idx.extend(range(_OFF["corp_shares"] + j * 2, _OFF["corp_shares"] + (j + 1) * 2))
        idx.extend(range(_OFF["corp_market_zone"] + j * 4, _OFF["corp_market_zone"] + (j + 1) * 4))
        groups.append(idx)
    return groups


def _build_private_indices() -> list[list[int]]:
    """Build gather indices for each private company entity group (21 features each)."""
    owner_size = NUM_PLAYERS + NUM_CORPORATIONS  # 12
    groups = []
    for k in range(NUM_PRIVATES):
        idx = list(range(_OFF["private_ownership"] + k * owner_size, _OFF["private_ownership"] + (k + 1) * owner_size))
        idx.append(_OFF["private_revenue"] + k)
        idx.append(_OFF["private_closed"] + k)
        idx.extend(range(_OFF["auction_bids"] + k * NUM_PLAYERS, _OFF["auction_bids"] + (k + 1) * NUM_PLAYERS))
        idx.append(_OFF["auction_min_bid"] + k)
        idx.append(_OFF["auction_available"] + k)
        idx.append(_OFF["auction_face_value"] + k)
        groups.append(idx)
    return groups


def _build_global_indices() -> list[int]:
    """Build gather indices for the global entity group (64 features)."""
    idx = [_OFF["round_type"], _OFF["game_phase"], _OFF["bank_cash"]]
    idx.extend(range(_OFF["or_structure"], _OFF["or_structure"] + 2))
    idx.append(_OFF["train_limit"])
    idx.extend(range(_OFF["depot_trains"], _OFF["depot_trains"] + NUM_TRAIN_TYPES))
    idx.extend(range(_OFF["market_pool_trains"], _OFF["market_pool_trains"] + NUM_TRAIN_TYPES))
    idx.extend(range(_OFF["depot_tiles"], _OFF["depot_tiles"] + NUM_TILE_IDS))
    return idx


# Feature sizes per entity type
PLAYER_FEAT_SIZE = 14
CORP_FEAT_SIZE = 18
PRIVATE_FEAT_SIZE = 21
GLOBAL_FEAT_SIZE = 64
NUM_ENTITY_GROUPS = NUM_PLAYERS + NUM_CORPORATIONS + NUM_PRIVATES + 1  # 19


def _build_port_feature_mask(num_node_features: int = 50) -> Tensor:
    """Build (num_node_features, 6) mask mapping node features to port track indicators.

    Port p has track if any connect or port-revenue feature involving port p is nonzero.
    """
    mask = torch.zeros(num_node_features, 6)
    # connects_i_j features start at index 23
    feat_idx = 23
    for i in range(6):
        for j in range(i):
            mask[feat_idx, i] = 1.0
            mask[feat_idx, j] = 1.0
            feat_idx += 1
    # port_i_connects_revenue_j features start at index 38
    for i in range(6):
        for k in range(2):
            mask[38 + i * 2 + k, i] = 1.0
    return mask


# ──────────────────────────────────────────────────────────────────────────────
# Hex Map Transformer components
# ──────────────────────────────────────────────────────────────────────────────


class HexPositionalEncoding(nn.Module):
    """Projects continuous axial (q, r) hex coordinates through an MLP."""

    def __init__(self, d_model: int):
        super().__init__()
        self.coord_proj = nn.Sequential(
            nn.Linear(2, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model),
        )

    def forward(self, axial_coords: Tensor) -> Tensor:
        return self.coord_proj(axial_coords)


class StructuralAttentionBias(nn.Module):
    """Attention bias from hex grid structure: distance, direction, track connectivity."""

    def __init__(self, num_heads: int, max_distance: int = 12):
        super().__init__()
        self.num_heads = num_heads
        self.max_distance = max_distance
        self.distance_bias = nn.Embedding(max_distance + 1, num_heads)
        self.direction_bias = nn.Embedding(7, num_heads)  # 6 directions + "not adjacent"
        self.track_bias = nn.Parameter(torch.zeros(num_heads))

    def forward(self, distance_matrix: Tensor, direction_matrix: Tensor, track_connectivity: Tensor) -> Tensor:
        """Returns (num_heads, N, N) bias to add to attention logits."""
        clamped_dist = distance_matrix.clamp(max=self.max_distance)
        dist_bias = self.distance_bias(clamped_dist)  # (N, N, heads)
        dir_bias = self.direction_bias(direction_matrix)  # (N, N, heads)
        track_bias = track_connectivity.unsqueeze(-1) * self.track_bias  # (N, N, heads)
        return (dist_bias + dir_bias + track_bias).permute(2, 0, 1)  # (heads, N, N)


class HexTransformerLayer(nn.Module):
    """Pre-norm Transformer layer with structural attention bias."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))

    def forward(self, x: Tensor, structural_bias: Optional[Tensor] = None) -> Tensor:
        B, N, D = x.shape
        normed = self.ln1(x)
        qkv = self.qkv(normed).reshape(B, N, 3, self.num_heads, self.d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # each (B, heads, N, d_head)

        attn_logits = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        if structural_bias is not None:
            attn_logits = attn_logits + structural_bias.unsqueeze(0)

        attn = F.softmax(attn_logits, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        x = x + self.out_proj(out)
        x = x + self.ff(self.ln2(x))
        return x


class HexMapTransformer(nn.Module):
    """Complete map encoder replacing the GNN.

    Produces:
    - node_embeds: (B, N, d_model) per-hex embeddings for policy head
    - map_pool: (B, d_model) pooled board embedding for trunk
    """

    def __init__(self, num_node_features: int, d_model: int, num_heads: int, num_layers: int, d_ff: int, max_distance: int):
        super().__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(num_node_features, d_model)
        self.input_ln = nn.LayerNorm(d_model)
        self.pos_enc = HexPositionalEncoding(d_model)
        self.structural_bias = StructuralAttentionBias(num_heads, max_distance)
        self.layers = nn.ModuleList([HexTransformerLayer(d_model, num_heads, d_ff) for _ in range(num_layers)])
        self.output_ln = nn.LayerNorm(d_model)

        # Attention pooling
        self.pool_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pool_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)

    def forward(
        self,
        node_features: Tensor,
        axial_coords: Tensor,
        distance_matrix: Tensor,
        direction_matrix: Tensor,
        track_connectivity: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        B = node_features.shape[0]

        x = self.input_proj(node_features)
        pos = self.pos_enc(axial_coords)
        x = self.input_ln(x + pos.unsqueeze(0))

        # Use first sample's track connectivity (all leaves from same game tree during self-play)
        track_conn = track_connectivity[0] if track_connectivity.dim() == 3 else track_connectivity
        bias = self.structural_bias(distance_matrix, direction_matrix, track_conn)

        for layer in self.layers:
            x = layer(x, structural_bias=bias)

        node_embeds = self.output_ln(x)

        query = self.pool_query.expand(B, -1, -1)
        map_pool, _ = self.pool_attn(query, node_embeds, node_embeds)
        map_pool = map_pool.squeeze(1)

        return node_embeds, map_pool


# ──────────────────────────────────────────────────────────────────────────────
# Economic State Transformer
# ──────────────────────────────────────────────────────────────────────────────


class EconomicStateTransformer(nn.Module):
    """Treats the 390-dim game state as 19 entity groups and runs a small Transformer.

    Entity groups: 4 players (14 feat), 8 corps (18 feat), 6 privates (21 feat), 1 global (64 feat).
    Each projected to d_entity, with type + id embeddings.
    """

    def __init__(self, d_entity: int, num_layers: int, num_heads: int, d_ff: int):
        super().__init__()
        self.d_entity = d_entity

        # Per-type input projections
        self.player_proj = nn.Linear(PLAYER_FEAT_SIZE, d_entity)
        self.corp_proj = nn.Linear(CORP_FEAT_SIZE, d_entity)
        self.private_proj = nn.Linear(PRIVATE_FEAT_SIZE, d_entity)
        self.global_proj = nn.Linear(GLOBAL_FEAT_SIZE, d_entity)

        # Entity type embedding (4 types: player, corp, private, global)
        self.type_embedding = nn.Embedding(4, d_entity)
        # Entity ID embedding (max 19 entities)
        self.id_embedding = nn.Embedding(NUM_ENTITY_GROUPS, d_entity)

        # CLS token for pooled output
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_entity) * 0.02)

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_entity,
            nhead=num_heads,
            dim_feedforward=d_ff,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_ln = nn.LayerNorm(d_entity)

        # Precompute entity type/id tensors
        type_ids = (
            [0] * NUM_PLAYERS + [1] * NUM_CORPORATIONS + [2] * NUM_PRIVATES + [3]
        )
        self.register_buffer("type_ids", torch.tensor(type_ids, dtype=torch.long))
        self.register_buffer("entity_ids", torch.arange(NUM_ENTITY_GROUPS, dtype=torch.long))

        # Precompute gather indices for entity groups
        player_idx = _build_player_indices()
        corp_idx = _build_corp_indices()
        private_idx = _build_private_indices()
        global_idx = _build_global_indices()

        self.register_buffer("player_gather", torch.tensor(player_idx, dtype=torch.long))  # (4, 14)
        self.register_buffer("corp_gather", torch.tensor(corp_idx, dtype=torch.long))  # (8, 18)
        self.register_buffer("private_gather", torch.tensor(private_idx, dtype=torch.long))  # (6, 21)
        self.register_buffer("global_gather", torch.tensor(global_idx, dtype=torch.long))  # (64,)

    def forward(self, game_state: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            game_state: (B, 390) flat game state vector

        Returns:
            cls_embed: (B, d_entity) pooled economic embedding
            entity_embeds: (B, 19, d_entity) per-entity embeddings for cross-attention
        """
        B = game_state.shape[0]

        # Gather entity features
        player_feats = game_state[:, self.player_gather]  # (B, 4, 14)
        corp_feats = game_state[:, self.corp_gather]  # (B, 8, 18)
        private_feats = game_state[:, self.private_gather]  # (B, 6, 21)
        global_feats = game_state[:, self.global_gather]  # (B, 64)

        # Project to d_entity
        player_embeds = self.player_proj(player_feats)  # (B, 4, d)
        corp_embeds = self.corp_proj(corp_feats)  # (B, 8, d)
        private_embeds = self.private_proj(private_feats)  # (B, 6, d)
        global_embed = self.global_proj(global_feats).unsqueeze(1)  # (B, 1, d)

        # Stack all entities
        entities = torch.cat([player_embeds, corp_embeds, private_embeds, global_embed], dim=1)  # (B, 19, d)

        # Add type and ID embeddings
        entities = entities + self.type_embedding(self.type_ids) + self.id_embedding(self.entity_ids)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, d)
        seq = torch.cat([cls, entities], dim=1)  # (B, 20, d)

        # Transformer
        seq = self.transformer(seq)
        seq = self.output_ln(seq)

        cls_embed = seq[:, 0]  # (B, d)
        entity_embeds = seq[:, 1:]  # (B, 19, d)

        return cls_embed, entity_embeds


# ──────────────────────────────────────────────────────────────────────────────
# Cross-Modal Fusion
# ──────────────────────────────────────────────────────────────────────────────


class CrossModalFusion(nn.Module):
    """Two-stage fusion: cross-attention (econ → map) then concat + project.

    Stage A: Economic entities attend to map node embeddings.
    Stage B: Concatenate fused economic embedding with pooled map embedding, project to trunk dim.
    """

    def __init__(self, d_entity: int, d_map: int, d_trunk: int, num_heads: int):
        super().__init__()
        # Cross-attention: econ entities (Q) attend to map nodes (KV)
        # Project econ and map to a common dimension for cross-attention
        self.d_attn = d_entity  # use d_entity as the common dim
        self.q_proj = nn.Linear(d_entity, self.d_attn)
        self.k_proj = nn.Linear(d_map, self.d_attn)
        self.v_proj = nn.Linear(d_map, self.d_attn)
        self.cross_attn = nn.MultiheadAttention(self.d_attn, num_heads, batch_first=True)
        self.cross_attn_ln = nn.LayerNorm(self.d_attn)

        # Stage B: concat + project
        self.fusion_proj = nn.Linear(self.d_attn + d_map, d_trunk)
        self.fusion_ln = nn.LayerNorm(d_trunk)

    def forward(self, entity_embeds: Tensor, node_embeds: Tensor, map_pool: Tensor) -> Tensor:
        """
        Args:
            entity_embeds: (B, 19, d_entity) from EconomicStateTransformer
            node_embeds: (B, N, d_map) from HexMapTransformer
            map_pool: (B, d_map) pooled map embedding

        Returns:
            trunk_input: (B, d_trunk)
        """
        # Stage A: cross-attention
        q = self.q_proj(entity_embeds)
        k = self.k_proj(node_embeds)
        v = self.v_proj(node_embeds)
        cross_out, _ = self.cross_attn(q, k, v)  # (B, 19, d_attn)
        cross_out = self.cross_attn_ln(cross_out)
        econ_fused = cross_out.mean(dim=1)  # (B, d_attn) — mean pool over entities

        # Stage B: concat and project
        fused = torch.cat([econ_fused, map_pool], dim=1)  # (B, d_attn + d_map)
        return self.fusion_ln(self.fusion_proj(fused))


# ──────────────────────────────────────────────────────────────────────────────
# FiLM-inside Residual Block
# ──────────────────────────────────────────────────────────────────────────────


class FiLMResBlock(nn.Module):
    """Residual block with FiLM conditioning INSIDE (before second linear).

    FiLM is initialized to identity (gamma=1, beta=0) so it's a no-op at init.
    Uses LayerNorm (not BatchNorm) for RL stability.
    """

    def __init__(self, d_trunk: int, d_film: int):
        super().__init__()
        self.fc1 = nn.Linear(d_trunk, d_trunk)
        self.ln1 = nn.LayerNorm(d_trunk)
        self.fc2 = nn.Linear(d_trunk, d_trunk)
        self.ln2 = nn.LayerNorm(d_trunk)
        self.film = nn.Linear(d_film, d_trunk * 2)
        # Initialize FiLM to identity
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        self.film.bias.data[:d_trunk] = 1.0  # gamma = 1

    def forward(self, x: Tensor, phase_embed: Tensor) -> Tensor:
        residual = x
        out = F.gelu(self.ln1(self.fc1(x)))
        gamma, beta = self.film(phase_embed).chunk(2, dim=1)
        out = gamma * out + beta
        out = self.ln2(self.fc2(out))
        return F.gelu(out + residual)


# ──────────────────────────────────────────────────────────────────────────────
# v2 Policy Head (outer-sum factoring + structured sub-heads)
# ──────────────────────────────────────────────────────────────────────────────


class V2PolicyHead(nn.Module):
    """Simplified policy head with outer-sum factoring for LayTile actions.

    LayTile logit(h,t,r) = hex_score(h) + tile_logit(t) + rot_logit(r)
    Hex scores come from per-node Hex Transformer embeddings.
    Other actions use per-type linear sub-heads.

    ~500K params vs v1's 18.8M.
    """

    def __init__(self, d_trunk: int, d_map: int, policy_size: int, lay_tile_info: dict):
        super().__init__()
        self.lay_tile_offset = lay_tile_info["offset"]
        self.num_hexes = lay_tile_info["num_hexes"]
        self.num_tiles = lay_tile_info["num_tiles"]
        self.num_rotations = lay_tile_info["num_rotations"]
        self.num_lay_tile = lay_tile_info["num_lay_tile"]
        self.num_other = policy_size - self.num_lay_tile
        self.policy_size = policy_size

        # Hex scoring from per-node Hex Transformer embeddings
        self.hex_scorer = nn.Sequential(
            nn.Linear(d_map, d_map // 2),
            nn.GELU(),
            nn.Linear(d_map // 2, 1),
        )
        # Fallback hex head from trunk (when node embeddings unavailable)
        self.hex_head = nn.Linear(d_trunk, self.num_hexes)

        # Permutation: action mapper hex order → Hex Transformer node order
        from rl18xx.agent.alphazero.encoder import HEX_COORDS_ORDERED
        from rl18xx.agent.alphazero.action_mapper import ActionMapper

        action_mapper = ActionMapper()
        gnn_coord_to_idx = {coord: i for i, coord in enumerate(HEX_COORDS_ORDERED)}
        mapper_hex_coords = list(action_mapper.hex_offsets.keys())
        self.register_buffer(
            "mapper_to_gnn",
            torch.tensor([gnn_coord_to_idx[coord] for coord in mapper_hex_coords], dtype=torch.long),
        )

        # Tile and rotation sub-heads
        self.tile_head = nn.Linear(d_trunk, self.num_tiles)
        self.rotation_head = nn.Linear(d_trunk, self.num_rotations)

        # Structured per-type sub-heads for non-LayTile actions
        self.other_head = nn.Linear(d_trunk, self.num_other)

    def forward(self, trunk: Tensor, node_embeds: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            trunk: (B, d_trunk) trunk features
            node_embeds: (B, N, d_map) per-hex Transformer embeddings (optional)
        """
        B = trunk.shape[0]

        # Hex scores
        if node_embeds is not None:
            node_scores = self.hex_scorer(node_embeds).squeeze(-1)  # (B, N)
            hex_logits = node_scores[:, self.mapper_to_gnn]  # (B, num_hexes) in mapper order
        else:
            hex_logits = self.hex_head(trunk)

        tile_logits = self.tile_head(trunk)  # (B, T)
        rot_logits = self.rotation_head(trunk)  # (B, R)

        # Outer sum: logit(h,t,r) = hex(h) + tile(t) + rot(r)
        lay_tile_logits = (
            hex_logits.unsqueeze(2).unsqueeze(3)  # (B, H, 1, 1)
            + tile_logits.unsqueeze(1).unsqueeze(3)  # (B, 1, T, 1)
            + rot_logits.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, R)
        )
        lay_tile_flat = lay_tile_logits.reshape(B, -1)  # (B, H*T*R)

        # Non-LayTile actions
        other_logits = self.other_head(trunk)  # (B, num_other)

        # Assemble full policy vector
        other_before = self.lay_tile_offset
        full_logits = torch.cat(
            [other_logits[:, :other_before], lay_tile_flat, other_logits[:, other_before:]],
            dim=1,
        )
        return full_logits


# ──────────────────────────────────────────────────────────────────────────────
# Track Connectivity Computation
# ──────────────────────────────────────────────────────────────────────────────


class TrackConnectivityComputer(nn.Module):
    """Computes dynamic track connectivity from node features and direction matrix.

    Track connectivity[i,j] = 1 if hex i has track on port d (direction to j)
    AND hex j has track on port (d+3)%6 (opposite direction).
    """

    def __init__(self, num_node_features: int = 50):
        super().__init__()
        self.register_buffer("port_feature_mask", _build_port_feature_mask(num_node_features))

    def forward(self, node_features: Tensor, direction_matrix: Tensor) -> Tensor:
        """
        Args:
            node_features: (B, N, F) per-hex features
            direction_matrix: (N, N) int — direction from hex i to j (0-5 adjacent, 6 not adjacent)

        Returns:
            track_connectivity: (B, N, N) float — 1.0 if track-connected, 0.0 otherwise
        """
        B, N, _ = node_features.shape

        # Determine which ports have track on each hex: (B, N, 6)
        port_activity = node_features @ self.port_feature_mask  # (B, N, 6)
        port_has_track = (port_activity > 0).float()

        # Adjacency mask
        adj_mask = (direction_matrix != 6)  # (N, N)

        # For adjacent hex pairs: check source port d and dest port (d+3)%6
        dir_mat = direction_matrix.long()  # (N, N)
        opp_dir = (dir_mat + 3) % 6  # (N, N)

        # Gather source port indicators: src_ports[b, i, j] = port_has_track[b, i, dir[i,j]]
        dir_expanded = dir_mat.unsqueeze(0).expand(B, -1, -1)  # (B, N, N)
        # Clamp to valid range for non-adjacent (dir=6 → clamp to 5, masked out anyway)
        dir_clamped = dir_expanded.clamp(max=5)
        opp_clamped = opp_dir.unsqueeze(0).expand(B, -1, -1).clamp(max=5)

        # port_has_track: (B, N, 6)
        # For each (i,j), need port_has_track[b, i, dir[i,j]]
        # Reshape for gather: (B, N, N) index into last dim of (B, N, 6)
        pht_expanded_src = port_has_track.unsqueeze(2).expand(-1, -1, N, -1)  # (B, N, N, 6)
        src_ports = torch.gather(pht_expanded_src, 3, dir_clamped.unsqueeze(-1)).squeeze(-1)  # (B, N, N)

        pht_expanded_dst = port_has_track.unsqueeze(1).expand(-1, N, -1, -1)  # (B, N, N, 6)
        dst_ports = torch.gather(pht_expanded_dst, 3, opp_clamped.unsqueeze(-1)).squeeze(-1)  # (B, N, N)

        return src_ports * dst_ports * adj_mask.float().unsqueeze(0)


# ──────────────────────────────────────────────────────────────────────────────
# Full v2 Model
# ──────────────────────────────────────────────────────────────────────────────


class AlphaZeroV2Model(AlphaZeroModel):
    """v2 AlphaZero model: Hex Transformer + Economic Transformer + FiLM trunk.

    ~7.3M parameters with the default config.
    """

    def __init__(self, config: ModelV2Config):
        super().__init__()
        self.config = config
        self.device = config.device

        self._init_structural_data()
        self._init_model()
        self.to(self.device)

    def encoder_type(self):
        return "V2"

    def _init_structural_data(self):
        """Precompute static hex grid structural matrices."""
        from rl18xx.agent.alphazero.encoder import HEX_COORDS_ORDERED, Encoder_1830

        self.encoder = Encoder_1830.get_encoder_for_model(self)

        # Compute axial coordinates
        axial = torch.tensor([hex_coord_to_axial(c) for c in HEX_COORDS_ORDERED], dtype=torch.float32)
        if axial.shape[0] > 1:
            axial_min = axial.min(dim=0).values
            axial_max = axial.max(dim=0).values
            axial_range = (axial_max - axial_min).clamp(min=1e-6)
            axial = 2.0 * (axial - axial_min) / axial_range - 1.0
        self.register_buffer("axial_coords", axial)

        # Distance and direction matrices will be computed on first encode() call
        # since we need the game's hex adjacency information
        self._structural_matrices_initialized = False
        self.register_buffer("distance_matrix", torch.zeros(len(HEX_COORDS_ORDERED), len(HEX_COORDS_ORDERED), dtype=torch.long))
        self.register_buffer("direction_matrix", torch.full((len(HEX_COORDS_ORDERED), len(HEX_COORDS_ORDERED)), 6, dtype=torch.long))

    def _compute_structural_matrices(self, game):
        """Compute distance and direction matrices from game's hex grid. Called once."""
        if self._structural_matrices_initialized:
            return

        from rl18xx.agent.alphazero.encoder import HEX_COORDS_ORDERED

        hex_coords = HEX_COORDS_ORDERED
        coord_to_idx = {c: i for i, c in enumerate(hex_coords)}
        n = len(hex_coords)

        # Build adjacency from game
        adjacency = {}
        for hex_obj in game.hexes:
            if hex_obj.id not in coord_to_idx:
                continue
            neighbors = []
            for direction, neighbor in hex_obj.all_neighbors.items():
                if neighbor.id in coord_to_idx:
                    neighbors.append((neighbor.id, direction))
            adjacency[hex_obj.id] = neighbors

        # Direction matrix
        direction = torch.full((n, n), 6, dtype=torch.long)
        for src, nbrs in adjacency.items():
            i = coord_to_idx[src]
            for dst, d in nbrs:
                j = coord_to_idx[dst]
                direction[i, j] = d

        # Distance matrix via BFS
        distance = torch.full((n, n), n, dtype=torch.long)
        for start in range(n):
            distance[start, start] = 0
            queue = [start]
            visited = {start}
            while queue:
                current = queue.pop(0)
                coord = hex_coords[current]
                if coord in adjacency:
                    for neighbor_coord, _ in adjacency[coord]:
                        j = coord_to_idx[neighbor_coord]
                        if j not in visited:
                            visited.add(j)
                            distance[start, j] = distance[start, current] + 1
                            queue.append(j)

        self.distance_matrix.copy_(distance)
        self.direction_matrix.copy_(direction)
        self._structural_matrices_initialized = True
        LOGGER.info("Computed structural matrices for Hex Transformer")

    def _init_model(self):
        c = self.config

        # 1. Economic State Transformer
        self.econ_transformer = EconomicStateTransformer(
            d_entity=c.d_entity,
            num_layers=c.econ_transformer_layers,
            num_heads=c.econ_transformer_heads,
            d_ff=c.econ_transformer_ff_dim,
        )

        # 2. Hex Map Transformer
        self.hex_transformer = HexMapTransformer(
            num_node_features=c.map_node_features,
            d_model=c.d_map,
            num_heads=c.hex_transformer_heads,
            num_layers=c.hex_transformer_layers,
            d_ff=c.hex_transformer_ff_dim,
            max_distance=c.max_hex_distance,
        )

        # Track connectivity computer
        self.track_conn_computer = TrackConnectivityComputer(c.map_node_features)

        # 3. Cross-Modal Fusion
        self.fusion = CrossModalFusion(
            d_entity=c.d_entity,
            d_map=c.d_map,
            d_trunk=c.d_trunk,
            num_heads=c.cross_attn_heads,
        )

        # 4. Phase-Conditioned Trunk
        self.phase_embedding = nn.Embedding(c.num_round_types, c.film_embed_dim)
        self.res_blocks = nn.ModuleList([FiLMResBlock(c.d_trunk, c.film_embed_dim) for _ in range(c.num_res_blocks)])

        # 5. Policy Head
        from rl18xx.agent.alphazero.action_mapper import ActionMapper

        action_mapper = ActionMapper()
        lay_tile_info = action_mapper.get_lay_tile_index_info()
        self.policy_head = V2PolicyHead(c.d_trunk, c.d_map, c.policy_size, lay_tile_info)

        # 6. Value Head (per-player with active-player signal, LayerNorm)
        head_hidden = c.d_trunk // 2
        value_input_dim = c.d_trunk + c.value_size  # trunk + one-hot player
        value_layers = []
        in_dim = value_input_dim
        for i in range(c.value_head_layers - 1):
            out_dim = head_hidden
            value_layers.extend([nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim), nn.GELU()])
            in_dim = out_dim
        value_layers.append(nn.Linear(in_dim, c.value_size))
        self.value_head = nn.Sequential(*value_layers)

        # 7. Auxiliary Heads
        self.aux_action_count_head = nn.Linear(c.d_trunk, 1)
        self.aux_phase_head = nn.Linear(c.d_trunk, c.num_game_phases)

        # Load weights or initialize
        if c.model_checkpoint_file:
            self.load_weights(c.model_checkpoint_file)
        else:
            self._initialize_weights()

    def _initialize_weights(self):
        """Kaiming init for GELU layers. FiLM layers keep their identity init."""
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear) and "film" not in name:
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def get_name(self):
        return f"AlphaZeroV2Model_{self.config.timestamp}"

    def _extract_round_type(self, game_state_data: Tensor) -> Tensor:
        """Extract round type index from game state vector (offset 16, normalized by MAX_ROUND_TYPE_IDX=2)."""
        return (game_state_data[:, _OFF["round_type"]] * 2).round().long()

    def _extract_active_player(self, game_state_data: Tensor) -> Tensor:
        """Extract active player index from game state vector (one-hot at offsets 0-3)."""
        return game_state_data[:, :NUM_PLAYERS].argmax(dim=1)

    # --- External interface (same as v1) ---

    def run(self, game_state) -> Tuple[Tensor, Tensor, Tensor]:
        probs, log_probs, values = self.run_many([game_state])
        return probs[0], log_probs[0], values[0]

    def run_encoded(self, encoded_game_state) -> Tuple[Tensor, Tensor, Tensor]:
        probs, log_probs, values = self.run_many_encoded([encoded_game_state])
        return probs[0], log_probs[0], values[0]

    def run_many(self, game_states) -> Tuple[Tensor, Tensor, Tensor]:
        encoded_game_states = [self.encoder.encode(gs) for gs in game_states]
        return self.run_many_encoded(encoded_game_states)

    def run_many_encoded(self, game_states: list) -> Tuple[Tensor, Tensor, Tensor]:
        batch_size = len(game_states)
        if batch_size == 0:
            raise ValueError("Received no game states to run.")

        game_state_tensors = []
        node_features_list = []
        round_type_indices = []
        active_player_indices = []

        for gs in game_states:
            game_state_tensor = gs[0].to(self.device)
            node_data = gs[1].to(self.device)
            round_type_idx = gs[4] if len(gs) > 4 else 0
            active_player_idx = gs[5] if len(gs) > 5 else 0

            game_state_tensors.append(game_state_tensor)
            node_features_list.append(node_data)
            round_type_indices.append(round_type_idx)
            active_player_indices.append(active_player_idx)

        batched_gs = torch.cat(game_state_tensors, dim=0)  # (B, 390)
        batched_nodes = torch.stack(node_features_list, dim=0)  # (B, N, F)
        round_type_tensor = torch.tensor(round_type_indices, dtype=torch.long, device=self.device)
        active_player_tensor = torch.tensor(active_player_indices, dtype=torch.long, device=self.device)

        policy_logits, value_logits, _ = self.forward(
            batched_gs, batched_nodes, round_type_tensor, active_player_tensor
        )

        probabilities = F.softmax(policy_logits, dim=1)
        log_probs = F.log_softmax(policy_logits, dim=1)
        return probabilities, log_probs, value_logits

    def forward(
        self,
        game_state_data: Tensor,
        node_features_or_batch,
        round_type_idx: Optional[Tensor] = None,
        active_player_idx: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass of the v2 model.

        Args:
            game_state_data: (B, 390) game state vectors
            node_features_or_batch: either (B, N, F) node features tensor,
                or a PyG Batch (for training compatibility — extracts .x and reshapes)
            round_type_idx: (B,) round type indices. If None, extracted from game state.
            active_player_idx: (B,) active player indices. If None, extracted from game state.

        Returns:
            (policy_logits, value_logits, aux_action_count_pred)
        """
        batch_size = game_state_data.shape[0]

        # Handle PyG Batch from DataLoader (training) vs raw tensor (inference)
        if hasattr(node_features_or_batch, "x"):
            # PyG Batch — reshape flat nodes to (B, N, F)
            node_features = node_features_or_batch.x.view(batch_size, self.config.num_hexes, -1)
        else:
            node_features = node_features_or_batch

        # Extract round type and active player from game state if not provided
        if round_type_idx is None:
            round_type_idx = self._extract_round_type(game_state_data)
        if active_player_idx is None:
            active_player_idx = self._extract_active_player(game_state_data)

        # 1. Economic State Transformer
        cls_embed, entity_embeds = self.econ_transformer(game_state_data)

        # 2. Hex Map Transformer
        track_conn = self.track_conn_computer(node_features, self.direction_matrix)
        node_embeds, map_pool = self.hex_transformer(
            node_features,
            self.axial_coords,
            self.distance_matrix,
            self.direction_matrix,
            track_conn,
        )

        # 3. Cross-Modal Fusion
        trunk_input = self.fusion(entity_embeds, node_embeds, map_pool)

        # 4. Phase-Conditioned Trunk
        phase_embed = self.phase_embedding(round_type_idx)
        x = trunk_input
        for block in self.res_blocks:
            x = block(x, phase_embed)

        # 5. Policy Head
        policy_logits = self.policy_head(x, node_embeds)

        # 6. Value Head
        player_indicator = F.one_hot(active_player_idx, num_classes=self.config.value_size).float()
        value_input = torch.cat([x, player_indicator], dim=1)
        value_logits = self.value_head(value_input)

        # 7. Auxiliary Heads
        aux_action_count_pred = self.aux_action_count_head(x)

        return policy_logits, value_logits, aux_action_count_pred

    def predict_phase(self, trunk_features: Tensor) -> Tensor:
        """Auxiliary phase prediction (for training loss)."""
        return self.aux_phase_head(trunk_features)
