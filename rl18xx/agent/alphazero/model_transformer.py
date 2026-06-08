"""
AlphaZero Transformer Model Architecture for 1830.

An alternative to the GNN-based variant, built around:
- Hex Map Transformer (global attention over 93 hexes with structural bias)
- Economic State Transformer (entity-group attention over players/corps/privates)
- Cross-modal fusion (economic entities attend to map nodes)
- FiLM-inside residual trunk (phase conditioning inside blocks)
- Hierarchical policy head (autoregressive `P(h)·P(t|h)·P(r|h,t)` for LayTile
  and `P(h)·P(slot|h)` for PlaceToken; flat softmax for everything else)

See docs/network_architecture_v2.md for full design rationale.
"""

import math
import logging
from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from rl18xx.agent.alphazero.config import ModelTransformerConfig
from rl18xx.agent.alphazero.encoder import Encoder_1830Graph
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
# Entity group index definitions for variable-player-count 1830
#
# The flat game-state vector layout (section name -> (offset, size)) is owned by
# ``Encoder_1830Graph.compute_section_layout``. A single Transformer checkpoint
# supports the full 2..MAX_PLAYERS range: the model's fixed buffers (gather
# indices, id embedding, value head output) are built for ``MAX_PLAYERS``, and
# shorter games are padded to that layout — player slots beyond the sample's
# actual count get zero features and are masked out in attention via a
# per-sample ``key_padding_mask``.
#
# The Python encoder still emits a variable-length state (sized for the game's
# actual player count); the model performs the padding inside
# ``_forward_encoded_batch`` so every batched tensor has a single fixed length.
# ──────────────────────────────────────────────────────────────────────────────

MAX_PLAYERS = 6  # Module-level maximum; the model uses ``config.max_players``.
NUM_CORPORATIONS = Encoder_1830Graph.NUM_CORPORATIONS
NUM_PRIVATES = Encoder_1830Graph.NUM_PRIVATES
NUM_TRAIN_TYPES = Encoder_1830Graph.NUM_TRAIN_TYPES
NUM_TILE_IDS = Encoder_1830Graph.NUM_TILE_IDS


def _layout_for(num_players: int) -> tuple[dict, dict, int]:
    """Return ``(off, size, total_size)`` for the encoder's flat game-state layout."""
    layout, total = Encoder_1830Graph.compute_section_layout(num_players)
    off = {name: offset for name, (offset, _size) in layout.items()}
    size = {name: sz for name, (_offset, sz) in layout.items()}
    return off, size, total


# Module-level layout used by the model's fixed-shape buffers: built for the
# max player count, so a single checkpoint covers 2..MAX_PLAYERS.
_LAYOUT, _GAME_STATE_SIZE = Encoder_1830Graph.compute_section_layout(MAX_PLAYERS)
_OFF = {name: offset for name, (offset, _size) in _LAYOUT.items()}
_SIZE = {name: size for name, (_offset, size) in _LAYOUT.items()}


def _build_player_indices(num_players: int = MAX_PLAYERS, off: Optional[dict] = None) -> list[list[int]]:
    """Build gather indices for each player entity group (14 features each).

    ``num_players`` is the **layout** player count (i.e. the model's
    ``max_players`` slot allocation). Use the same value for both the layout
    and the embedding-table size so the gather indices and ``id_embedding``
    stay consistent.
    """
    if off is None:
        off = _OFF
    groups = []
    for i in range(num_players):
        idx = [
            off["active_entity"] + i,
            off["active_president"] + i,
            off["priority_deal_player"] + i,
            off["player_certs_remaining"] + i,
            off["player_cash"] + i,
        ]
        idx.extend(range(off["player_shares"] + i * NUM_CORPORATIONS, off["player_shares"] + (i + 1) * NUM_CORPORATIONS))
        idx.append(off["player_turn_order"] + i)
        groups.append(idx)
    return groups


def _build_corp_indices(num_players: int = MAX_PLAYERS, off: Optional[dict] = None) -> list[list[int]]:
    """Build gather indices for each corporation entity group (18 features each)."""
    if off is None:
        off = _OFF
    groups = []
    for j in range(NUM_CORPORATIONS):
        idx = [
            off["active_entity"] + num_players + j,
            off["corp_floated"] + j,
            off["corp_cash"] + j,
        ]
        idx.extend(range(off["corp_trains"] + j * NUM_TRAIN_TYPES, off["corp_trains"] + (j + 1) * NUM_TRAIN_TYPES))
        idx.append(off["corp_tokens_remaining"] + j)
        idx.extend(range(off["corp_share_price"] + j * 2, off["corp_share_price"] + (j + 1) * 2))
        idx.extend(range(off["corp_shares"] + j * 2, off["corp_shares"] + (j + 1) * 2))
        idx.extend(range(off["corp_market_zone"] + j * 4, off["corp_market_zone"] + (j + 1) * 4))
        groups.append(idx)
    return groups


def _build_private_indices(num_players: int = MAX_PLAYERS, off: Optional[dict] = None) -> list[list[int]]:
    """Build gather indices for each private company entity group (variable size).

    The owner one-hot is sized ``num_players + NUM_CORPORATIONS`` (so it
    matches the encoder), and the auction-bids slice is sized
    ``num_players``. The total per-private feature size is therefore
    ``num_players + NUM_CORPORATIONS + 1 + 1 + num_players + 3 ==
    2*num_players + NUM_CORPORATIONS + 5``.
    """
    if off is None:
        off = _OFF
    owner_size = num_players + NUM_CORPORATIONS
    groups = []
    for k in range(NUM_PRIVATES):
        idx = list(range(off["private_ownership"] + k * owner_size, off["private_ownership"] + (k + 1) * owner_size))
        idx.append(off["private_revenue"] + k)
        idx.append(off["private_closed"] + k)
        idx.extend(range(off["auction_bids"] + k * num_players, off["auction_bids"] + (k + 1) * num_players))
        idx.append(off["auction_min_bid"] + k)
        idx.append(off["auction_available"] + k)
        idx.append(off["auction_face_value"] + k)
        groups.append(idx)
    return groups


def _build_global_indices(off: Optional[dict] = None) -> list[int]:
    """Build gather indices for the global entity group (64 features)."""
    if off is None:
        off = _OFF
    idx = [off["round_type"], off["game_phase"], off["bank_cash"]]
    idx.extend(range(off["or_structure"], off["or_structure"] + 2))
    idx.append(off["train_limit"])
    idx.extend(range(off["depot_trains"], off["depot_trains"] + NUM_TRAIN_TYPES))
    idx.extend(range(off["market_pool_trains"], off["market_pool_trains"] + NUM_TRAIN_TYPES))
    idx.extend(range(off["depot_tiles"], off["depot_tiles"] + NUM_TILE_IDS))
    return idx


# Feature sizes per entity type
PLAYER_FEAT_SIZE = 14
CORP_FEAT_SIZE = 18
# Private feature size depends on the layout player count (owner one-hot +
# auction-bids). The model is built for ``MAX_PLAYERS``, so private features
# include 6 padded player slots in their owner one-hot and auction-bid regions.
PRIVATE_FEAT_SIZE = 2 * MAX_PLAYERS + NUM_CORPORATIONS + 5  # == 25 for 6 players
GLOBAL_FEAT_SIZE = 64
NUM_ENTITY_GROUPS = MAX_PLAYERS + NUM_CORPORATIONS + NUM_PRIVATES + 1  # 21 for 6 players


def _private_feat_size(num_players: int) -> int:
    """Per-private entity feature length for a given layout player count."""
    return 2 * num_players + NUM_CORPORATIONS + 5


# Section names whose offsets / sizes depend on the player-count layout. Used
# by ``_pad_state_to_max_players`` to remap a variable-N state vector into the
# model's fixed max-N layout.
_VARIABLE_PLAYER_SECTIONS = (
    "active_entity",          # size = num_players + NUM_CORPORATIONS
    "active_president",       # size = num_players
    "priority_deal_player",   # size = num_players
    "player_certs_remaining", # size = num_players
    "player_cash",            # size = num_players
    "player_shares",          # size = num_players * NUM_CORPORATIONS
    "private_ownership",      # size = NUM_PRIVATES * (num_players + NUM_CORPORATIONS)
    "auction_bids",           # size = NUM_PRIVATES * num_players
    "player_turn_order",      # size = num_players
)


def _pad_state_to_max_players(state: Tensor, num_players: int, max_players: int = MAX_PLAYERS) -> Tensor:
    """Re-lay out a variable-N game-state vector to the model's max-N layout.

    The encoder emits a 1D state of size ``_layout_for(num_players)[2]`` where
    every player-indexed section is exactly ``num_players`` wide. The model's
    fixed buffers expect a state sized for ``max_players``, with each
    player-indexed section ``max_players`` wide and the last
    ``max_players - num_players`` slots zero in each such section. This helper
    copies each section into its max-N offset; player-indexed sections copy
    the first ``num_players`` slots and leave the remainder zero.

    Args:
        state: ``(D_n,)`` 1D tensor sized for ``num_players``. Or ``(1, D_n)``;
            the leading singleton dim is supported and preserved.
        num_players: actual game player count (in ``[2, max_players]``).
        max_players: model layout player count (default ``MAX_PLAYERS``).

    Returns:
        ``(D_m,)`` or ``(1, D_m)`` tensor laid out for ``max_players``.
    """
    if num_players == max_players:
        return state

    src_off, src_size, _src_total = _layout_for(num_players)
    dst_off, dst_size, dst_total = _layout_for(max_players)

    # Support both (D,) and (1, D) inputs — the encoder returns (1, D).
    squeezed = False
    if state.dim() == 1:
        state = state.unsqueeze(0)
        squeezed = True
    B = state.size(0)
    out = state.new_zeros(B, dst_total)

    for name in src_off:
        src_s = src_off[name]
        dst_s = dst_off[name]
        n_src = src_size[name]
        if name not in _VARIABLE_PLAYER_SECTIONS:
            # Fixed-size section: copy verbatim into its new offset.
            out[:, dst_s:dst_s + n_src] = state[:, src_s:src_s + n_src]
            continue

        # Player-indexed section: copy each per-player (or per-corp / per-private
        # sub-block) into its max-N counterpart, leaving the unused player
        # slots zero. The per-section logic mirrors ``GAME_STATE_ENCODING_STRUCTURE``.
        if name == "active_entity":
            # Layout: [num_players player slots, NUM_CORPORATIONS corp slots].
            out[:, dst_s:dst_s + num_players] = state[:, src_s:src_s + num_players]
            out[:, dst_s + max_players:dst_s + max_players + NUM_CORPORATIONS] = (
                state[:, src_s + num_players:src_s + num_players + NUM_CORPORATIONS]
            )
        elif name in (
            "active_president", "priority_deal_player", "player_certs_remaining",
            "player_cash", "player_turn_order",
        ):
            # Flat per-player vector: copy first ``num_players`` slots.
            out[:, dst_s:dst_s + num_players] = state[:, src_s:src_s + num_players]
        elif name == "player_shares":
            # Per-player block of NUM_CORPORATIONS entries — copy the first
            # ``num_players`` blocks into the first ``num_players`` block slots.
            block = NUM_CORPORATIONS
            for p in range(num_players):
                out[:, dst_s + p * block:dst_s + (p + 1) * block] = (
                    state[:, src_s + p * block:src_s + (p + 1) * block]
                )
        elif name == "private_ownership":
            # Per-private block of (num_players + NUM_CORPORATIONS) entries.
            # Map into per-private blocks of (max_players + NUM_CORPORATIONS).
            src_block = num_players + NUM_CORPORATIONS
            dst_block = max_players + NUM_CORPORATIONS
            for k in range(NUM_PRIVATES):
                base_src = src_s + k * src_block
                base_dst = dst_s + k * dst_block
                # Player owner slots (first ``num_players`` of each block).
                out[:, base_dst:base_dst + num_players] = state[:, base_src:base_src + num_players]
                # Corporation owner slots (always NUM_CORPORATIONS).
                out[:, base_dst + max_players:base_dst + max_players + NUM_CORPORATIONS] = (
                    state[:, base_src + num_players:base_src + num_players + NUM_CORPORATIONS]
                )
        elif name == "auction_bids":
            # Per-private block of ``num_players`` entries; map to ``max_players``.
            for k in range(NUM_PRIVATES):
                base_src = src_s + k * num_players
                base_dst = dst_s + k * max_players
                out[:, base_dst:base_dst + num_players] = state[:, base_src:base_src + num_players]
        else:
            # Defensive — every name in ``_VARIABLE_PLAYER_SECTIONS`` should be
            # handled above. If a new variable-N section is added to the
            # encoder layout, plumb it here too.
            raise ValueError(f"Unhandled variable-player section {name!r}; update _pad_state_to_max_players.")

    if squeezed:
        out = out.squeeze(0)
    return out


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
# Shared pooling utility
# ──────────────────────────────────────────────────────────────────────────────


class AttentionPool(nn.Module):
    """Pool a token sequence to a fixed-size summary via a learnable query.

    Used as the single standardized pooling mechanism across the model — hex
    map, economic entities, and post-fusion summarization all share this so the
    three places that reduce a token sequence to a vector behave consistently.
    """

    def __init__(self, d: int, num_heads: int = 4):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.attn = nn.MultiheadAttention(d, num_heads, batch_first=True)
        self.ln = nn.LayerNorm(d)

    def forward(self, tokens: Tensor, key_padding_mask: Optional[Tensor] = None) -> Tensor:
        # tokens: (B, N, d) → (B, d). ``key_padding_mask`` (B, N) bool: True
        # at positions to ignore (padded tokens).
        B = tokens.size(0)
        q = self.query.expand(B, -1, -1)
        out, _ = self.attn(q, tokens, tokens, key_padding_mask=key_padding_mask)
        return self.ln(out.squeeze(1))


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
        """Returns (B, num_heads, N, N) bias to add to attention logits.

        distance_matrix / direction_matrix are static per-title (N, N) tensors; they're
        broadcast across the batch dim. track_connectivity is per-sample (B, N, N) — each
        sample has its own track state (different MCTS leaves, different positions in a
        training batch), so we keep the batch dim through to the final bias.
        """
        clamped_dist = distance_matrix.clamp(max=self.max_distance)
        dist_bias = self.distance_bias(clamped_dist)  # (N, N, heads)
        dir_bias = self.direction_bias(direction_matrix)  # (N, N, heads)
        static_bias = (dist_bias + dir_bias).permute(2, 0, 1)  # (heads, N, N)

        if track_connectivity.dim() == 2:
            track_connectivity = track_connectivity.unsqueeze(0)  # (1, N, N)
        # track_connectivity: (B, N, N) → (B, heads, N, N) via outer product with track_bias
        track_bias = track_connectivity.unsqueeze(1) * self.track_bias.view(1, -1, 1, 1)

        return static_bias.unsqueeze(0) + track_bias  # (B, heads, N, N)


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
            # structural_bias is (B, heads, N, N) — already batched, so add directly.
            attn_logits = attn_logits + structural_bias

        attn = F.softmax(attn_logits, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        x = x + self.out_proj(out)
        x = x + self.ff(self.ln2(x))
        return x


class HexMapTransformer(nn.Module):
    """Complete attention-based map encoder.

    Produces:
    - node_embeds: (B, N, d_model) per-hex embeddings for policy head
    - map_pool: (B, d_model) pooled board embedding for trunk

    This is the "transformer" variant of the swappable map encoder. The
    "resnet" variant (``HexResNetMapEncoder``) below exposes the same
    ``(per_hex_embeddings, map_pool)`` outputs.
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

        # Attention pooling (shared module — same mechanism used everywhere we
        # collapse a token sequence to a vector)
        self.map_pool = AttentionPool(d_model, num_heads=num_heads)

    def forward(
        self,
        node_features: Tensor,
        axial_coords: Tensor,
        distance_matrix: Tensor,
        direction_matrix: Tensor,
        track_connectivity: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        x = self.input_proj(node_features)
        pos = self.pos_enc(axial_coords)
        x = self.input_ln(x + pos.unsqueeze(0))

        # Pass the full per-sample track connectivity through — different leaves/positions
        # have different track state, so we must not collapse the batch dim.
        bias = self.structural_bias(distance_matrix, direction_matrix, track_connectivity)

        for layer in self.layers:
            x = layer(x, structural_bias=bias)

        node_embeds = self.output_ln(x)
        map_pool = self.map_pool(node_embeds)

        return node_embeds, map_pool


# Alias mirroring the design-doc name. ``HexMapTransformer`` is the original
# implementation; ``HexTransformerMapEncoder`` is the doc-canonical name for
# the swappable map-encoder family. They are the same class.
HexTransformerMapEncoder = HexMapTransformer


# ──────────────────────────────────────────────────────────────────────────────
# Hex ResNet Map Encoder
# ──────────────────────────────────────────────────────────────────────────────


def _build_offset_grid_mapping(hex_coords: list) -> tuple[int, int, list[tuple[int, int]]]:
    """Build offset-grid coordinates for the 1830 hex map.

    Each hex coordinate string (e.g. 'E14') is parsed into a ``(row, col)`` on
    an offset rectangular grid. 1830 uses a "pointy-top" layout where each
    letter-row has columns of one parity (e.g. row A: cols {9,11,17,19}); we
    compress this by using ``col // 2`` as the grid column, so neighbouring
    hexes share rows and adjacent grid columns the way a standard ResNet
    expects.

    Returns:
        ``(grid_rows, grid_cols, positions)`` where ``positions[i]`` is the
        ``(row, col)`` of the i-th hex in ``hex_coords``. Caller is expected
        to verify that all positions are unique within the returned grid.
    """
    positions = []
    for c in hex_coords:
        letter_part = "".join(ch for ch in c if ch.isalpha())
        num_part = "".join(ch for ch in c if ch.isdigit())
        row = ord(letter_part.upper()) - ord("A")
        col = int(num_part) // 2
        positions.append((row, col))
    rows = [p[0] for p in positions]
    cols = [p[1] for p in positions]
    grid_rows = max(rows) + 1
    grid_cols = max(cols) + 1
    # Sanity: positions must be unique on the grid.
    assert len(set(positions)) == len(positions), (
        "Offset-grid mapping produced collisions — check hex coordinate parsing."
    )
    return grid_rows, grid_cols, positions


class _ResNetBlock(nn.Module):
    """Standard 2-conv residual block: Conv-BN-GELU-Conv-BN + residual + GELU."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.gelu(out + residual)


class HexResNetMapEncoder(nn.Module):
    """Convolutional map encoder over an offset-grid projection of the 93 hexes.

    The 93 hex coordinates are scattered onto an offset rectangular grid (for
    1830: 11 rows x 13 cols). Non-hex grid cells are zero-padded and masked
    out of the pooled summary. A 1x1 conv projects per-hex features to the
    ResNet channel count; ``resnet_layers`` standard 3x3 residual blocks then
    mix neighbours. With ``resnet_layers=10`` the receptive field is ~21x21,
    enough to cover any 1830 route end-to-end.

    Exposes the same ``(per_hex_embeddings, map_pool)`` interface as
    ``HexMapTransformer``:
        - ``per_hex_embeddings``: (B, num_hexes, d_map) — gathered back from
          the grid in the original hex order.
        - ``map_pool``: (B, d_map) — masked-mean-pool over the real hex
          positions on the grid.

    ``d_map`` is set by an optional output projection if ``d_map != channels``.
    """

    def __init__(
        self,
        num_node_features: int,
        d_map: int,
        num_layers: int,
        channels: int,
        hex_coords: list,
    ):
        super().__init__()
        self.d_map = d_map
        self.channels = channels
        self.num_hexes = len(hex_coords)

        grid_rows, grid_cols, positions = _build_offset_grid_mapping(hex_coords)
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols

        # Per-hex flat grid index (used for scatter on input and gather on
        # output). Stored as a 1D long tensor of length ``num_hexes`` so the
        # forward pass can do a single ``index_*`` call per direction.
        flat_idx = torch.tensor(
            [r * grid_cols + c for (r, c) in positions], dtype=torch.long
        )
        self.register_buffer("hex_flat_idx", flat_idx)

        # Boolean grid mask: True at real hex positions, False at padding.
        # Stored as float (1.0/0.0) for use in the masked-mean-pool.
        grid_mask = torch.zeros(grid_rows, grid_cols, dtype=torch.float32)
        for (r, c) in positions:
            grid_mask[r, c] = 1.0
        self.register_buffer("grid_mask", grid_mask)
        # Number of real hex cells — guaranteed constant per title (== num_hexes).
        self.register_buffer(
            "grid_mask_sum", torch.tensor(grid_mask.sum().item(), dtype=torch.float32)
        )

        # 1x1 input projection: per-hex features -> ResNet channel count.
        # Applied on the grid (after scatter) so non-hex cells stay zero; the
        # 1x1 conv has no bias for the same reason.
        self.input_proj = nn.Conv2d(num_node_features, channels, kernel_size=1, bias=False)
        self.input_bn = nn.BatchNorm2d(channels)

        # Residual stack.
        self.blocks = nn.ModuleList([_ResNetBlock(channels) for _ in range(num_layers)])

        # Optional projection to d_map (skip if already matching to save params).
        if d_map != channels:
            self.output_proj = nn.Linear(channels, d_map)
        else:
            self.output_proj = nn.Identity()
        self.output_ln = nn.LayerNorm(d_map)

    def forward(self, node_features: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            node_features: (B, num_hexes, F) per-hex feature tensor.

        Returns:
            per_hex_embeddings: (B, num_hexes, d_map)
            map_pool:           (B, d_map)
        """
        B, N, F_in = node_features.shape
        assert N == self.num_hexes, f"Expected {self.num_hexes} hexes, got {N}"

        device = node_features.device
        grid_cells = self.grid_rows * self.grid_cols

        # Scatter per-hex features onto the offset grid (non-hex cells stay 0).
        # Shape: (B, F, grid_rows * grid_cols) → reshape → (B, F, R, C).
        grid_flat = node_features.new_zeros(B, F_in, grid_cells)
        # node_features is (B, N, F); transpose so the feature dim is the channel
        # dim, then scatter along the spatial dim using hex_flat_idx.
        nf_chan_first = node_features.transpose(1, 2)  # (B, F, N)
        flat_idx = self.hex_flat_idx.view(1, 1, -1).expand(B, F_in, -1)
        grid_flat.scatter_(2, flat_idx, nf_chan_first)
        x = grid_flat.view(B, F_in, self.grid_rows, self.grid_cols)

        # Project to channels and run the residual stack.
        x = F.gelu(self.input_bn(self.input_proj(x)))
        for block in self.blocks:
            x = block(x)

        # Gather back per-hex embeddings in the original hex order.
        # x: (B, C, R, C') → flatten spatial → gather along spatial dim.
        x_flat_spatial = x.view(B, self.channels, grid_cells)  # (B, C, R*C')
        gather_idx = self.hex_flat_idx.view(1, 1, -1).expand(B, self.channels, -1)
        per_hex_chan = torch.gather(x_flat_spatial, 2, gather_idx)  # (B, C, N)
        per_hex = per_hex_chan.transpose(1, 2).contiguous()  # (B, N, C)

        # Masked mean pool over real hex cells (grid_mask is 1 at real hexes,
        # 0 at padding). Equivalent to mean over the N real cells since the
        # ResNet may have produced nonzero activations at padding too.
        mask = self.grid_mask.view(1, 1, self.grid_rows, self.grid_cols)
        masked_x = x * mask  # zero-out padding contributions
        pool_sum = masked_x.flatten(2).sum(dim=2)  # (B, C)
        pool = pool_sum / self.grid_mask_sum.clamp(min=1.0)

        # Optional projection to d_map + LayerNorm for stable downstream use.
        per_hex = self.output_ln(self.output_proj(per_hex))
        pool = self.output_ln(self.output_proj(pool))

        return per_hex, pool


# ──────────────────────────────────────────────────────────────────────────────
# Economic State Transformer
# ──────────────────────────────────────────────────────────────────────────────


class EconomicStateTransformer(nn.Module):
    """Treats the flat game-state vector as N entity groups and runs a Transformer.

    Entity groups (all sized for the model's ``max_players`` layout):
    ``max_players`` players (14 feat each), 8 corps (18 feat),
    6 privates (``2*max_players + NUM_CORPORATIONS + 5`` feat),
    1 global (64 feat). Each is projected to ``d_entity``, with type + id
    embeddings added.

    The module is built for ``max_players`` slots; shorter games pad the
    unused player slots with zeros (see ``_pad_state_to_max_players``) and
    pass a per-sample ``num_players`` to ``forward()`` so the attention layers
    mask out the padded player tokens via a ``key_padding_mask``. A single
    checkpoint therefore covers the full 2..max_players range.
    """

    def __init__(self, d_entity: int, num_layers: int, num_heads: int, d_ff: int, max_players: int = MAX_PLAYERS):
        super().__init__()
        self.d_entity = d_entity
        self.max_players = max_players

        # The model's gather buffers / id embedding are sized for ``max_players``.
        off, _size, _total = _layout_for(max_players)
        private_feat_size = _private_feat_size(max_players)
        num_entity_groups = max_players + NUM_CORPORATIONS + NUM_PRIVATES + 1

        # Per-type input projections
        self.player_proj = nn.Linear(PLAYER_FEAT_SIZE, d_entity)
        self.corp_proj = nn.Linear(CORP_FEAT_SIZE, d_entity)
        self.private_proj = nn.Linear(private_feat_size, d_entity)
        self.global_proj = nn.Linear(GLOBAL_FEAT_SIZE, d_entity)

        # Entity type embedding (4 types: player, corp, private, global)
        self.type_embedding = nn.Embedding(4, d_entity)
        # Entity ID embedding (one entry per max-N entity slot)
        self.id_embedding = nn.Embedding(num_entity_groups, d_entity)

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_entity,
            nhead=num_heads,
            dim_feedforward=d_ff,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)
        self.output_ln = nn.LayerNorm(d_entity)

        # Attention pool replaces the old [CLS] token — same role (collapse the
        # entity sequence to a single summary vector), but consistent with the
        # pooling used by HexMapTransformer and CrossModalFusion.
        self.entity_pool = AttentionPool(d_entity, num_heads=num_heads)

        # Per-entity type / id buffers (built for max-N).
        type_ids = (
            [0] * max_players + [1] * NUM_CORPORATIONS + [2] * NUM_PRIVATES + [3]
        )
        self.register_buffer("type_ids", torch.tensor(type_ids, dtype=torch.long))
        self.register_buffer("entity_ids", torch.arange(num_entity_groups, dtype=torch.long))

        # Per-entity gather indices into the (padded) max-N game-state vector.
        player_idx = _build_player_indices(max_players, off)
        corp_idx = _build_corp_indices(max_players, off)
        private_idx = _build_private_indices(max_players, off)
        global_idx = _build_global_indices(off)

        self.register_buffer("player_gather", torch.tensor(player_idx, dtype=torch.long))
        self.register_buffer("corp_gather", torch.tensor(corp_idx, dtype=torch.long))
        self.register_buffer("private_gather", torch.tensor(private_idx, dtype=torch.long))
        self.register_buffer("global_gather", torch.tensor(global_idx, dtype=torch.long))

        # Total token sequence length: ``max_players + NUM_CORPORATIONS + NUM_PRIVATES + 1``.
        # Used to construct per-sample key-padding masks.
        self.num_entity_groups = num_entity_groups
        self.num_corp_tokens = NUM_CORPORATIONS
        self.num_private_tokens = NUM_PRIVATES
        # Index ranges (within the entity sequence) of the per-type token groups.
        # The sequence is laid out [players ... corps ... privates ... global].
        self.player_token_start = 0
        self.player_token_end = max_players  # exclusive

    def _build_key_padding_mask(self, num_players: Tensor) -> Tensor:
        """Build (B, num_entity_groups) bool mask: True at padded token positions.

        ``num_players`` is a (B,) long tensor of per-sample player counts. The
        layout is ``[players ... corps ... privates ... global]``; only the
        leading ``max_players`` slots can be padded (corp / private / global
        always exist).
        """
        B = num_players.size(0)
        device = num_players.device
        mask = torch.zeros(B, self.num_entity_groups, dtype=torch.bool, device=device)
        # arange(0..max_players-1) compared per-sample against num_players.
        slot_idx = torch.arange(self.max_players, device=device).unsqueeze(0)  # (1, max_players)
        # Padded player slot when slot_idx >= num_players.
        mask[:, self.player_token_start:self.player_token_end] = slot_idx >= num_players.unsqueeze(1)
        return mask

    def forward(self, game_state: Tensor, num_players: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            game_state: (B, D) flat game state vector laid out for ``max_players``.
                Shorter games are padded ahead of this call (see
                ``_pad_state_to_max_players``) so unused player slots are zero.
            num_players: (B,) long tensor of per-sample actual player counts in
                ``[2, max_players]``. If ``None``, every sample is treated as
                ``max_players`` (no padding mask).

        Returns:
            entity_summary: (B, d_entity) attention-pooled economic embedding.
            entity_embeds:  (B, num_entity_groups, d_entity) per-entity embeddings
                for cross-attention.
            key_padding_mask: (B, num_entity_groups) bool mask — True at padded
                token positions. Returned for downstream consumers (e.g. the
                cross-modal fusion) that also need to mask the padded tokens.
        """
        B = game_state.size(0)
        if num_players is None:
            num_players = torch.full((B,), self.max_players, dtype=torch.long, device=game_state.device)

        # Gather entity features
        player_feats = game_state[:, self.player_gather]  # (B, max_players, 14)
        corp_feats = game_state[:, self.corp_gather]  # (B, 8, 18)
        private_feats = game_state[:, self.private_gather]  # (B, 6, 2*max_players + 8 + 5)
        global_feats = game_state[:, self.global_gather]  # (B, 64)

        # Project to d_entity
        player_embeds = self.player_proj(player_feats)  # (B, max_players, d)
        corp_embeds = self.corp_proj(corp_feats)  # (B, 8, d)
        private_embeds = self.private_proj(private_feats)  # (B, 6, d)
        global_embed = self.global_proj(global_feats).unsqueeze(1)  # (B, 1, d)

        # Stack all entities
        entities = torch.cat([player_embeds, corp_embeds, private_embeds, global_embed], dim=1)
        # entities: (B, num_entity_groups, d)

        # Add type and ID embeddings
        entities = entities + self.type_embedding(self.type_ids) + self.id_embedding(self.entity_ids)

        # Build per-sample key-padding mask for the padded player slots.
        key_padding_mask = self._build_key_padding_mask(num_players)

        # Transformer over per-entity tokens. ``src_key_padding_mask=True`` at
        # a position tells the attention to ignore that key — exactly what we
        # want for the padded player slots.
        seq = self.transformer(entities, src_key_padding_mask=key_padding_mask)
        entity_embeds = self.output_ln(seq)  # (B, num_entity_groups, d)

        # Zero out the padded entity rows so any downstream consumer that
        # doesn't honour the mask still sees a benign zero contribution.
        valid = (~key_padding_mask).unsqueeze(-1).float()
        entity_embeds = entity_embeds * valid

        # Attention-pool the per-entity outputs to a single summary vector,
        # masking out padded slots.
        entity_summary = self.entity_pool(entity_embeds, key_padding_mask=key_padding_mask)

        return entity_summary, entity_embeds, key_padding_mask


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

        # Attention pool over the cross-attended entity tokens — replaces the old
        # mean-pool. Consistent with how every other token sequence in the model
        # is reduced to a single summary vector.
        self.fused_pool = AttentionPool(self.d_attn, num_heads=num_heads)

        # Stage B: concat + project
        self.fusion_proj = nn.Linear(self.d_attn + d_map, d_trunk)
        self.fusion_ln = nn.LayerNorm(d_trunk)

    def forward(
        self,
        entity_embeds: Tensor,
        node_embeds: Tensor,
        map_pool: Tensor,
        entity_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            entity_embeds: (B, num_entity_groups, d_entity) from EconomicStateTransformer
            node_embeds: (B, N, d_map) from HexMapTransformer
            map_pool: (B, d_map) pooled map embedding
            entity_key_padding_mask: (B, num_entity_groups) bool — True at padded
                entity slots (e.g. unused player tokens in shorter games). Used
                to mask the cross-attention queries' padded positions AND to
                exclude padded entities from the post-attention pool.

        Returns:
            trunk_input: (B, d_trunk)
        """
        # Stage A: cross-attention. The padded entity slots are queries that
        # we'd rather not waste attention on; we don't pass a Q-side mask to
        # ``MultiheadAttention`` (PyTorch doesn't accept one), but we zero out
        # the rows of ``cross_out`` corresponding to padded entities before
        # the pool so they contribute nothing downstream.
        q = self.q_proj(entity_embeds)
        k = self.k_proj(node_embeds)
        v = self.v_proj(node_embeds)
        cross_out, _ = self.cross_attn(q, k, v)  # (B, num_entity_groups, d_attn)
        cross_out = self.cross_attn_ln(cross_out)
        if entity_key_padding_mask is not None:
            valid = (~entity_key_padding_mask).unsqueeze(-1).float()
            cross_out = cross_out * valid
        econ_fused = self.fused_pool(cross_out, key_padding_mask=entity_key_padding_mask)

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
# Policy Head (autoregressive LayTile + PlaceToken, parallel-factored elsewhere)
# ──────────────────────────────────────────────────────────────────────────────


class ContinuousPriceHead(nn.Module):
    """Continuous-price head for Bid / BuyTrain / BuyCompany.

    Emits ``(mean, log_std)`` per legal ``(action_type, entity)`` slot. Slots are
    arranged in a fixed canonical order so MCTS / training can index into the
    output by (type, entity) key without re-querying the action mapper.

    Slot layout (matches ``ActionMapper`` entity ordering):

      - ``Bid``        : 6 slots, one per company (SV, CS, DH, MH, CA, BO).
      - ``BuyTrain``   : 48 slots — 8 corporations × 6 train types
                        (only the corp-to-corp variant is price-bearing; depot
                        and market trains have fixed prices and are not
                        produced by this head).
      - ``BuyCompany`` : 6 slots, one per company.

    Total: 60 slots × 2 (mean, log_std) = 120 outputs per example.

    The MCTS consumer reads ``(mean, exp(log_std))`` for a (type, entity) pair
    and uses it to parameterize a Normal distribution truncated to that
    action's legal ``[price_min, price_max]`` range. The pretraining /
    self-play loss uses the head's NLL against the observed price.

    ``log_std`` is centered near 0 (i.e. σ≈1 in raw-dollar units, which is too
    tight). To make the head start with a useful prior, we apply a small
    constant bias to log_std at init so the initial spread is ~$25 (log_std
    ≈ 3.2). This is a hyperparameter; tune with empirical data.
    """

    # Canonical slot ordering, matching ActionMapper's company / corporation
    # tables. Kept in sync with action_mapper.py — if those tables change, so
    # does ``_BUILD_SLOTS`` below.
    _COMPANIES = ("SV", "CS", "DH", "MH", "CA", "BO")
    _CORPORATIONS = ("PRR", "NYC", "CPR", "B&O", "C&O", "ERIE", "NYNH", "B&M")
    _TRAIN_TYPES = ("2", "3", "4", "5", "6", "D")

    LOG_STD_INIT = 3.0  # exp(3) ≈ 20 — reasonable starting spread for $-prices

    def __init__(self, d_trunk: int):
        super().__init__()

        # Build the slot list and a lookup dict so callers can map a
        # (type, entity) tuple to a slot index without iterating.
        slots: List[Tuple[str, tuple]] = []
        # Bids: per-company.
        for c in self._COMPANIES:
            slots.append(("Bid", (c,)))
        # BuyTrain: per (corp, train_type).
        for corp in self._CORPORATIONS:
            for ttype in self._TRAIN_TYPES:
                slots.append(("BuyTrain", (corp, ttype)))
        # BuyCompany: per-company.
        for c in self._COMPANIES:
            slots.append(("BuyCompany", (c,)))

        self.slots: List[Tuple[str, tuple]] = slots
        self.num_slots = len(slots)
        self.slot_index = {key: i for i, key in enumerate(slots)}

        # Bookkeeping for sub-block ranges (handy for diagnostics).
        self.bid_offset = 0
        self.bid_count = len(self._COMPANIES)
        self.buy_train_offset = self.bid_count
        self.buy_train_count = len(self._CORPORATIONS) * len(self._TRAIN_TYPES)
        self.buy_company_offset = self.buy_train_offset + self.buy_train_count
        self.buy_company_count = len(self._COMPANIES)

        # MLP: trunk → 2 outputs per slot. Use a single linear from a hidden
        # state, kept small relative to the trunk to avoid parameter bloat.
        hidden = max(d_trunk // 4, 128)
        self.mlp = nn.Sequential(
            nn.Linear(d_trunk, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.num_slots * 2),
        )

    def forward(self, trunk: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            trunk: (B, d_trunk) trunk features.

        Returns:
            mean:     (B, num_slots) per-slot price means (raw $-units).
            log_std:  (B, num_slots) per-slot log-σ. Add ``LOG_STD_INIT`` bias.
        """
        B = trunk.shape[0]
        raw = self.mlp(trunk).view(B, self.num_slots, 2)
        mean = raw[..., 0]
        log_std = raw[..., 1] + self.LOG_STD_INIT
        return mean, log_std


class HierarchicalPolicyHead(nn.Module):
    """Autoregressive policy head for LayTile (and PlaceToken), parallel-factored
    sub-heads everywhere else.

    For LayTile the joint is factored as
        P(h, t, r) = P(h) * P(t | h) * P(r | h, t)
    with the conditioning passed in as concatenated inputs to the next sub-head's
    MLP — there's no sequential decoding at inference, all conditionals are
    computed in parallel and the assembly code picks the slice it needs.

    PlaceToken is similarly factored as P(h) * P(slot | h) over the 23 city-bearing
    hexes (slot is in {0, 1} since max city_count is 2).

    All other action types ("Pass / CompanyPass", Bid, Par, BuyShares, SellShares,
    BuyTrain, DiscardTrain, Dividend, BuyCompany, Bankrupt, RunRoutes,
    CompanyBuyShares, CompanyLayTile, CompanyPlaceToken) have no internal
    structure worth exploiting, so they share a single compact ``other_head``
    (Linear from trunk into ``num_other`` slots) and the result is scattered
    into the flat policy.

    The forward returns:
        flat_logits:    (B, policy_size) — every slot holds a log-probability:
                         LayTile / PlaceToken slots get the autoregressive
                         joint ``log P(h) + log P(t|h) + log P(r|h,t)`` (or
                         ``log P(h) + log P(s|h)`` for PlaceToken); the
                         remaining slots get the per-block ``log_softmax`` of
                         the compact ``other_head``. Each of the three blocks
                         is a valid distribution in probability space (sums
                         to 1), and the downstream ``softmax`` over the legal
                         action mask in MCTS re-balances mass between blocks
                         based on which actions are actually legal.
        components:     dict of per-sub-head logits + log-softmaxes, used by
                         the decomposed training loss (see
                         ``train._compute_decomposed_policy_loss``).
    """

    # Per the user's design: ``node_embs`` is always supplied at inference (the
    # Hex Transformer produces it on every forward pass), so the old
    # trunk-only ``hex_head`` fallback in TransformerPolicyHead is unreachable
    # and has been removed.

    def __init__(self, d_trunk: int, d_map: int, policy_size: int, lay_tile_info: dict):
        super().__init__()
        self.lay_tile_offset = lay_tile_info["offset"]
        self.num_hexes = lay_tile_info["num_hexes"]
        self.num_tiles = lay_tile_info["num_tiles"]
        self.num_rotations = lay_tile_info["num_rotations"]
        self.num_lay_tile = lay_tile_info["num_lay_tile"]
        self.policy_size = policy_size

        # ---- LayTile autoregressive sub-heads ----
        # P(hex) — per-hex score from its Transformer node embedding.
        self.hex_scorer = nn.Sequential(
            nn.Linear(d_map, d_map // 2),
            nn.GELU(),
            nn.Linear(d_map // 2, 1),
        )

        # P(tile | hex) — input is (trunk ⊕ node_emb_at_hex), emitted in parallel
        # for every hex. The trunk gives "global game state" context, the
        # per-hex embedding gives "what's already here" context.
        tile_hidden = max(d_trunk // 2, 128)
        self.tile_head = nn.Sequential(
            nn.Linear(d_trunk + d_map, tile_hidden),
            nn.GELU(),
            nn.Linear(tile_hidden, self.num_tiles),
        )

        # P(rotation | hex, tile) — input is (trunk ⊕ node_emb_at_hex ⊕ tile_emb).
        # Computed for every (hex, tile) pair.
        d_tile = 32
        self.tile_embedding = nn.Embedding(self.num_tiles, d_tile)
        rot_hidden = max(d_trunk // 4, 64)
        self.rotation_head = nn.Sequential(
            nn.Linear(d_trunk + d_map + d_tile, rot_hidden),
            nn.GELU(),
            nn.Linear(rot_hidden, self.num_rotations),
        )

        # ---- PlaceToken autoregressive sub-heads ----
        # Resolve the PlaceToken block layout once at construction. The
        # ActionMapper enumerates PlaceToken actions in hex_offsets order,
        # emitting one entry per (hex, slot) pair for hexes with a city; the
        # max city_count is 2 in 1830.
        from rl18xx.agent.alphazero.action_mapper import ActionMapper
        from rl18xx.agent.alphazero.encoder import HEX_COORDS_ORDERED

        action_mapper = ActionMapper()
        self.place_token_offset = action_mapper.action_offsets["PlaceToken"]
        self.place_token_block_size = (
            action_mapper.action_offsets["LayTile"] - self.place_token_offset
        )
        self.max_city_slots = max(action_mapper.city_count.values())

        # For each PlaceToken flat index, record (mapper_hex_idx, slot_idx) so
        # the assembly code can vectorize the lookup. Also build a
        # (num_hexes, max_city_slots) "exists" mask so the slot softmax only
        # normalizes over the slots that actually exist at each hex.
        pt_to_hex = []
        pt_to_slot = []
        hex_keys = list(action_mapper.hex_offsets.keys())
        for h in hex_keys:
            if h in action_mapper.city_count:
                for s in range(action_mapper.city_count[h]):
                    pt_to_hex.append(action_mapper.hex_offsets[h])
                    pt_to_slot.append(s)
        self.register_buffer("place_token_to_mapper_hex", torch.tensor(pt_to_hex, dtype=torch.long))
        self.register_buffer("place_token_to_slot", torch.tensor(pt_to_slot, dtype=torch.long))
        # Slot-existence mask over the full (num_hexes, max_city_slots) grid.
        slot_mask = torch.zeros(self.num_hexes, self.max_city_slots)
        for h in hex_keys:
            if h in action_mapper.city_count:
                hi = action_mapper.hex_offsets[h]
                for s in range(action_mapper.city_count[h]):
                    slot_mask[hi, s] = 1.0
        self.register_buffer("place_token_slot_exists", slot_mask)
        # Per-hex slot head: takes (trunk ⊕ node_emb_at_hex) and emits a logit
        # per slot. Slots that don't physically exist at that hex are masked
        # to -inf in forward(), so the softmax only sees real slots.
        slot_hidden = max(d_trunk // 4, 64)
        self.place_token_slot_head = nn.Sequential(
            nn.Linear(d_trunk + d_map, slot_hidden),
            nn.GELU(),
            nn.Linear(slot_hidden, self.max_city_slots),
        )

        # Permutation: action mapper hex order → Hex Transformer node order.
        # Hex Transformer emits node_embeds in HEX_COORDS_ORDERED (sorted),
        # but the action mapper enumerates hexes in its own ``hex_offsets``
        # order. ``mapper_to_gnn[i]`` is the GNN node index for the i-th
        # mapper hex.
        gnn_coord_to_idx = {coord: i for i, coord in enumerate(HEX_COORDS_ORDERED)}
        mapper_hex_coords = list(action_mapper.hex_offsets.keys())
        self.register_buffer(
            "mapper_to_gnn",
            torch.tensor([gnn_coord_to_idx[coord] for coord in mapper_hex_coords], dtype=torch.long),
        )

        # ---- Other-action sub-head ----
        # Linear over only the slots OUTSIDE the LayTile and PlaceToken
        # blocks. Materializing a full ``policy_size``-wide head would add
        # ~13M parameters of dead weight (we never read the LayTile /
        # PlaceToken cells of it), so we compute a compact ``num_other``-wide
        # head and scatter its outputs into the flat policy via a precomputed
        # index buffer.
        pt_start = self.place_token_offset
        pt_end = pt_start + self.place_token_block_size
        lt_start = self.lay_tile_offset
        lt_end = lt_start + self.num_lay_tile
        other_indices = (
            list(range(0, pt_start))
            + list(range(pt_end, lt_start))
            + list(range(lt_end, policy_size))
        )
        self.num_other = len(other_indices)
        self.register_buffer("other_indices", torch.tensor(other_indices, dtype=torch.long))
        self.other_head = nn.Linear(d_trunk, self.num_other)

    def forward(self, trunk: Tensor, node_embeds: Tensor) -> Tuple[Tensor, dict]:
        """
        Args:
            trunk: (B, d_trunk) trunk features
            node_embeds: (B, N, d_map) per-hex Transformer embeddings

        Returns:
            flat_logits: (B, policy_size) assembled flat policy. The LayTile
                and PlaceToken slots are filled with log-probabilities (so a
                downstream ``log_softmax`` over the legal-action mask recovers
                the autoregressive joint); other slots hold raw logits from
                ``other_head``.
            components: dict of per-sub-head logits for the decomposed loss.
        """
        B = trunk.shape[0]
        H = self.num_hexes
        T = self.num_tiles
        R = self.num_rotations
        S = self.max_city_slots

        # Permute Hex Transformer's node embeddings into action-mapper hex order
        # so all downstream indexing aligns with the flat-action layout.
        node_embs_mapper = node_embeds[:, self.mapper_to_gnn, :]  # (B, H, d_map)

        # ---------------- LayTile autoregressive heads ----------------
        # P(hex): hex_logits is (B, H) raw logits over hexes.
        hex_logits = self.hex_scorer(node_embs_mapper).squeeze(-1)  # (B, H)
        log_p_hex = F.log_softmax(hex_logits, dim=-1)  # (B, H)

        # P(tile | hex): emit a (B, H, T) tile-conditional logit grid in parallel.
        trunk_b_h = trunk.unsqueeze(1).expand(-1, H, -1)  # (B, H, d_trunk)
        tile_input = torch.cat([trunk_b_h, node_embs_mapper], dim=-1)  # (B, H, d_trunk + d_map)
        tile_logits = self.tile_head(tile_input)  # (B, H, T)
        log_p_tile_given_hex = F.log_softmax(tile_logits, dim=-1)  # (B, H, T)

        # P(rot | hex, tile): emit a (B, H, T, R) grid in parallel.
        # Build the 4D input via broadcasted concatenation.
        tile_embs_table = self.tile_embedding.weight  # (T, d_tile)
        trunk_4d = trunk.view(B, 1, 1, -1).expand(-1, H, T, -1)  # (B, H, T, d_trunk)
        node_4d = node_embs_mapper.unsqueeze(2).expand(-1, -1, T, -1)  # (B, H, T, d_map)
        tile_4d = tile_embs_table.view(1, 1, T, -1).expand(B, H, -1, -1)  # (B, H, T, d_tile)
        rot_input = torch.cat([trunk_4d, node_4d, tile_4d], dim=-1)  # (B, H, T, d_trunk + d_map + d_tile)
        rot_logits = self.rotation_head(rot_input)  # (B, H, T, R)
        log_p_rot_given_hex_tile = F.log_softmax(rot_logits, dim=-1)  # (B, H, T, R)

        # Joint log-probability for LayTile slots: log P(h) + log P(t|h) + log P(r|h,t).
        # Shape (B, H, T, R), flattened to (B, H*T*R) in action-mapper layout.
        lay_tile_log_joint = (
            log_p_hex.view(B, H, 1, 1)
            + log_p_tile_given_hex.view(B, H, T, 1)
            + log_p_rot_given_hex_tile
        )  # (B, H, T, R)
        lay_tile_flat = lay_tile_log_joint.reshape(B, -1)  # (B, H*T*R)

        # ---------------- PlaceToken autoregressive heads ----------------
        # P(hex): a separate hex head — the "which city to token at" decision
        # has different semantics than "which hex to lay track on", so we
        # don't share weights.
        # Reuse the existing per-hex node embedding via a fresh slot head:
        # P(hex) for PlaceToken is naturally a scoring over the 93 hexes,
        # masked at the action-mask stage to the ones with a city *and* legal.
        # We compute it from the same node_embs_mapper through a separate scorer
        # to keep the LayTile P(h) cleanly separated from PlaceToken P(h).
        # For parameter parsimony we reuse the slot head's first layer by
        # concatenating only trunk + node_emb and emitting a (slots,) logit
        # vector — the marginal P(hex) is the log-sum-exp over slots, weighted
        # by the slot-exists mask.
        slot_input = torch.cat([trunk_b_h, node_embs_mapper], dim=-1)  # (B, H, d_trunk + d_map)
        pt_slot_logits = self.place_token_slot_head(slot_input)  # (B, H, S)

        # Mask out non-existent slots. Use a very negative value that survives
        # mixed precision (-1e4 instead of -inf so AMP doesn't produce NaNs).
        slot_exists = self.place_token_slot_exists  # (H, S)
        pt_slot_logits_masked = pt_slot_logits.masked_fill(
            slot_exists.unsqueeze(0) == 0, -1e4
        )  # (B, H, S)

        # P(hex) for PlaceToken: derive it as the marginal over slots —
        # logsumexp gives the "total preference" mass each hex carries before
        # slot conditioning, which is the natural quantity to softmax across
        # hexes. This couples the two heads through their shared MLP, which
        # is fine — both decisions condition on the same hex.
        pt_hex_logits = torch.logsumexp(pt_slot_logits_masked, dim=-1)  # (B, H)
        log_p_pt_hex = F.log_softmax(pt_hex_logits, dim=-1)  # (B, H)

        # P(slot | hex) is a softmax over the slot dim of pt_slot_logits_masked.
        log_p_pt_slot_given_hex = F.log_softmax(pt_slot_logits_masked, dim=-1)  # (B, H, S)

        # Assemble PlaceToken slots: for each (h, s) action in the action-mapper
        # PlaceToken block, fill log P(h) + log P(s|h). Slots are emitted in
        # mapper enumeration order via the precomputed gather buffers.
        # Shape (B, place_token_block_size).
        hex_idx_per_pt = self.place_token_to_mapper_hex  # (place_token_block_size,)
        slot_idx_per_pt = self.place_token_to_slot
        pt_hex_part = log_p_pt_hex[:, hex_idx_per_pt]  # (B, place_token_block_size)
        pt_slot_part = log_p_pt_slot_given_hex[
            :, hex_idx_per_pt, slot_idx_per_pt
        ]  # (B, place_token_block_size)
        place_token_flat = pt_hex_part + pt_slot_part  # (B, place_token_block_size)

        # ---------------- Other actions ----------------
        other_logits = self.other_head(trunk)  # (B, num_other) raw logits over
        # the slots outside the LayTile / PlaceToken blocks. We `log_softmax`
        # these into log-probs so the assembled flat policy has consistent
        # log-probability semantics across ALL three blocks — each block sums
        # to 1 in probability space, and the final `softmax` MCTS does over
        # the legal-action mask re-weights mass between blocks based on which
        # actions are legal at this state.
        log_p_other = F.log_softmax(other_logits, dim=-1)  # (B, num_other)

        # ---------------- Assemble flat policy ----------------
        # All blocks contribute log-probabilities (autoregressive joint for
        # LayTile / PlaceToken; flat log-softmax for everything else). The
        # ``new_full(-1e4)`` background ensures any slot we don't explicitly
        # fill (there shouldn't be any with the layout above, but defensively)
        # sits at a deep negative so it doesn't accidentally get picked.
        flat_logits = trunk.new_full((B, self.policy_size), -1e4)
        flat_logits.index_copy_(1, self.other_indices, log_p_other)
        pt_start = self.place_token_offset
        pt_end = pt_start + self.place_token_block_size
        lt_start = self.lay_tile_offset
        lt_end = lt_start + self.num_lay_tile
        flat_logits[:, pt_start:pt_end] = place_token_flat
        flat_logits[:, lt_start:lt_end] = lay_tile_flat

        components = {
            # LayTile per-level raw logits (train loss applies its own
            # log_softmax — pre-applying it here would double-normalize).
            "hex_logits": hex_logits,
            "tile_logits": tile_logits,
            "rotation_logits": rot_logits,
            # LayTile per-level pre-computed log-softmaxes (reused by the
            # flat-policy assembly above; surfaced for diagnostics).
            "log_p_hex": log_p_hex,                          # (B, H)
            "log_p_tile_given_hex": log_p_tile_given_hex,    # (B, H, T)
            "log_p_rot_given_hex_tile": log_p_rot_given_hex_tile,  # (B, H, T, R)
            # PlaceToken per-level logits (already slot-masked to existence).
            "place_token_hex_logits": pt_hex_logits,
            "place_token_slot_logits": pt_slot_logits_masked,
            "place_token_slot_exists": slot_exists,
            "log_p_pt_hex": log_p_pt_hex,                              # (B, H)
            "log_p_pt_slot_given_hex": log_p_pt_slot_given_hex,        # (B, H, S)
            # Other-action raw logits and log-softmax (over ``other_indices``).
            # Layout buffers so loss code can route flat targets back to each
            # sub-head's grid without re-importing the action mapper.
            "other_logits": other_logits,
            "log_p_other": log_p_other,
            "other_indices": self.other_indices,
            "place_token_to_mapper_hex": self.place_token_to_mapper_hex,
            "place_token_to_slot": self.place_token_to_slot,
            # Layout (so loss code can find each block without re-importing).
            "lay_tile_offset": lt_start,
            "lay_tile_end": lt_end,
            "place_token_offset": pt_start,
            "place_token_end": pt_end,
            "num_hexes": H,
            "num_tiles": T,
            "num_rotations": R,
            "max_city_slots": S,
            # Joint log-prob tensors (handy for diagnostics / pretraining).
            "lay_tile_log_joint": lay_tile_log_joint,  # (B, H, T, R)
            "place_token_log_joint_flat": place_token_flat,  # (B, place_token_block_size)
        }
        return flat_logits, components


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
# Full Transformer Model
# ──────────────────────────────────────────────────────────────────────────────


class AlphaZeroTransformerModel(AlphaZeroModel):
    """Transformer AlphaZero model: Hex Transformer + Economic Transformer + FiLM trunk.

    ~7.3M parameters with the default config.
    """

    def __init__(self, config: ModelTransformerConfig):
        super().__init__()
        self.config = config
        self.device = config.device

        self._init_structural_data()
        self._init_model()
        # Hex grid geometry is static per-title, so precompute distance/direction matrices
        # once at construction. Uses a throwaway RustGameAdapter so we don't pay the cost of
        # constructing a full Python BaseGame.
        self._precompute_structural_matrices_from_rust()
        self.to(self.device)

    def encoder_type(self):
        return "Transformer"

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

    def _precompute_structural_matrices_from_rust(self):
        """Build the static distance/direction matrices using a throwaway RustGameAdapter.

        Hex grid geometry doesn't change across games of the same title, so we can compute
        these once at construction. Using the Rust engine avoids the cost of standing up a
        full Python BaseGame; if the Rust engine isn't available we silently fall back
        and the caller must invoke ``_compute_structural_matrices(game)`` once before the
        first forward pass.
        """
        try:
            from engine_rs import BaseGame as RustGame
            from rl18xx.rust_adapter import RustGameAdapter
        except ImportError:
            LOGGER.warning(
                "Rust engine not available — structural matrices will not be precomputed. "
                "Call _compute_structural_matrices(game) before the first forward pass."
            )
            return

        throwaway_game = RustGameAdapter(RustGame({1: "P1", 2: "P2", 3: "P3", 4: "P4"}))
        self._compute_structural_matrices(throwaway_game)

    def _init_model(self):
        c = self.config
        # Cache the max-player layout count so other helpers
        # (``_extract_active_player``) read the right slice without going
        # back through the config.
        self._max_players = c.max_players

        # 1. Economic State Transformer — built for the max player count.
        # Shorter games pad their player feature regions to ``max_players``
        # and pass their actual ``num_players`` to ``forward`` for masking.
        self.econ_transformer = EconomicStateTransformer(
            d_entity=c.d_entity,
            num_layers=c.econ_transformer_layers,
            num_heads=c.econ_transformer_heads,
            d_ff=c.econ_transformer_ff_dim,
            max_players=c.max_players,
        )

        # 2. Map encoder — swappable between the attention-based Hex
        # Transformer and the convolutional Hex ResNet. Both produce the
        # same ``(per_hex_embeddings, map_pool)`` outputs; the rest of the
        # model is unaware of which variant is active.
        from rl18xx.agent.alphazero.encoder import HEX_COORDS_ORDERED

        if c.map_encoder == "transformer":
            self.map_encoder_kind = "transformer"
            self.hex_transformer = HexMapTransformer(
                num_node_features=c.map_node_features,
                d_model=c.d_map,
                num_heads=c.hex_transformer_heads,
                num_layers=c.hex_transformer_layers,
                d_ff=c.hex_transformer_ff_dim,
                max_distance=c.max_hex_distance,
            )
            # Track connectivity is only needed by the Transformer encoder
            # (it consumes ``structural_bias`` over per-sample track state).
            self.track_conn_computer = TrackConnectivityComputer(c.map_node_features)
        elif c.map_encoder == "resnet":
            self.map_encoder_kind = "resnet"
            self.hex_transformer = HexResNetMapEncoder(
                num_node_features=c.map_node_features,
                d_map=c.d_map,
                num_layers=c.resnet_layers,
                channels=c.resnet_channels,
                hex_coords=HEX_COORDS_ORDERED,
            )
            self.track_conn_computer = None
        else:
            raise ValueError(
                f"Unknown map_encoder kind {c.map_encoder!r}; expected 'transformer' or 'resnet'."
            )

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

        # 5. Policy Head — hierarchical / autoregressive (Phase 3.3).
        from rl18xx.agent.alphazero.action_mapper import ActionMapper

        action_mapper = ActionMapper()
        lay_tile_info = action_mapper.get_lay_tile_index_info()
        self.policy_head = HierarchicalPolicyHead(c.d_trunk, c.d_map, c.policy_size, lay_tile_info)

        # Continuous-price head: emits (mean, log_std) per legal (action_type,
        # entity) slot for Bid / BuyTrain / BuyCompany. MCTS samples a price
        # via progressive widening against this Normal; training applies NLL
        # against the observed price. See ContinuousPriceHead docstring for
        # the slot layout.
        self.price_head = ContinuousPriceHead(c.d_trunk)

        # 6. Dual Value Head (KataGo-style; per-player, LayerNorm) — no
        # active-player one-hot. The encoder canonicalizes game state so the
        # active player always sits at slot 0, which makes the one-hot
        # indicator a constant input. Both heads share the same MLP topology
        # but are trained with different targets/losses:
        #   - win_loss_head: KL-div against share-of-winners. Backed up by MCTS.
        #   - score_head:    MSE against normalized net-worth fractions.
        #                    Auxiliary signal for the trunk; not used by MCTS.
        self.win_loss_head = self._build_value_head_mlp()
        self.score_head = self._build_value_head_mlp()

        # 7. Auxiliary Heads — only the log-legal-action-count head is wired.
        # The phase-prediction head (`aux_phase_head` / `predict_phase`) was
        # declared but never invoked in any training loop; removed along with
        # its config knob (`phase_aux_loss_weight`) for clarity.
        self.aux_action_count_head = nn.Linear(c.d_trunk, 1)

        # Load weights or initialize
        if c.model_checkpoint_file:
            self.load_weights(c.model_checkpoint_file)
        else:
            self._initialize_weights()

    def _build_value_head_mlp(self) -> nn.Sequential:
        """Build the shared MLP topology used by both win-loss and score heads."""
        c = self.config
        head_hidden = c.d_trunk // 2
        layers = []
        in_dim = c.d_trunk
        for _ in range(c.value_head_layers - 1):
            out_dim = head_hidden
            layers.extend([nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim), nn.GELU()])
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, c.num_players))
        return nn.Sequential(*layers)

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

    def architecture_name(self) -> str:
        return "AlphaZeroTransformer"

    def _extract_round_type(self, game_state_data: Tensor) -> Tensor:
        """Extract round type index from game state vector (normalized by MAX_ROUND_TYPE_IDX=2).

        The ``round_type`` offset depends on the encoder's player-count layout;
        we use the model's ``max_players`` layout (the incoming state is always
        padded to that layout before this is called).
        """
        off, _, _ = _layout_for(self._max_players)
        return (game_state_data[:, off["round_type"]] * 2).round().long()

    def _extract_active_player(self, game_state_data: Tensor) -> Tensor:
        """Extract active player index from game state vector (one-hot at offsets 0..max_players-1)."""
        return game_state_data[:, :self._max_players].argmax(dim=1)

    # --- Architecture-specific batch assembly (run/run_many live on AlphaZeroModel) ---

    def _forward_encoded_batch(self, encoded_game_states: list) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Assemble a batch from encoded tuples and run ``forward``.

        Each tuple is one of:
          - 7-tuple ``(gs, nf, ei, ea, rt, ap, rotation)`` — legacy / mcts path
            without an explicit ``num_players``. The player count is inferred
            from the state vector's flat length.
          - 8-tuple ``(gs, nf, ei, ea, rt, ap, rotation, num_players)`` — current
            encoder format; ``num_players`` is taken directly.

        States with ``num_players < max_players`` are padded to the model's
        ``max_players`` layout (zero-fill in unused player slots) so every
        batched tensor shares one fixed length.
        """
        max_players = self._max_players
        game_state_tensors = []
        node_features_list = []
        round_type_indices = []
        active_player_indices = []
        num_players_list = []

        for gs in encoded_game_states:
            game_state_tensor = gs[0].to(self.device)
            node_data = gs[1].to(self.device)
            round_type_idx = gs[4] if len(gs) > 4 else 0
            active_player_idx = gs[5] if len(gs) > 5 else 0
            if len(gs) > 7:
                num_players = int(gs[7])
            else:
                # Legacy 7-tuple path: infer ``num_players`` from the state
                # vector's flat length so callers (mcts.py's ``_rust_encode``)
                # keep working without touching that file.
                num_players = self._infer_num_players_from_state_size(game_state_tensor.shape[-1])

            if num_players != max_players:
                game_state_tensor = _pad_state_to_max_players(game_state_tensor, num_players, max_players)

            game_state_tensors.append(game_state_tensor)
            node_features_list.append(node_data)
            round_type_indices.append(round_type_idx)
            active_player_indices.append(active_player_idx)
            num_players_list.append(num_players)

        batched_gs = torch.cat(game_state_tensors, dim=0)  # (B, max-N layout size)
        batched_nodes = torch.stack(node_features_list, dim=0)  # (B, N, F)
        round_type_tensor = torch.tensor(round_type_indices, dtype=torch.long, device=self.device)
        active_player_tensor = torch.tensor(active_player_indices, dtype=torch.long, device=self.device)
        num_players_tensor = torch.tensor(num_players_list, dtype=torch.long, device=self.device)

        return self.forward(batched_gs, batched_nodes, round_type_tensor, active_player_tensor, num_players_tensor)

    @staticmethod
    def _infer_num_players_from_state_size(size: int) -> int:
        """Reverse the encoder layout to recover the player count from a flat state size.

        Used only for the legacy 7-tuple code path (mcts.py's ``_rust_encode``);
        the current encoder emits ``num_players`` directly as the tuple's 8th
        element.
        """
        for n in range(2, MAX_PLAYERS + 1):
            _, total = Encoder_1830Graph.compute_section_layout(n)
            if total == size:
                return n
        raise ValueError(
            f"Game state of length {size} does not match any layout for 2..{MAX_PLAYERS} players."
        )

    def forward(
        self,
        game_state_data: Tensor,
        node_features_or_batch,
        round_type_idx: Optional[Tensor] = None,
        active_player_idx: Optional[Tensor] = None,
        num_players: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Forward pass of the Transformer model.

        Args:
            game_state_data: (B, max-N layout size) game state vectors. The
                ``_forward_encoded_batch`` caller pads each sample's state from
                its actual ``num_players`` layout to the model's ``max_players``
                layout (zero-fill in unused player slots) before stacking.
            node_features_or_batch: either (B, N, F) node features tensor,
                or a PyG Batch (for training compatibility — extracts .x and reshapes)
            round_type_idx: (B,) round type indices. If None, extracted from game state.
            active_player_idx: (B,) active player indices (unused; canonical
                form always has the active player at slot 0).
            num_players: (B,) long tensor of per-sample actual player counts in
                ``[2, max_players]``. If ``None``, every sample is treated as
                ``max_players`` (i.e. no attention masking) — the training
                pipeline must supply real counts for shorter games to honour
                the padded slots.

        Returns:
            ``(policy_logits, win_loss_logits, score_pred, aux_action_count_pred)``.
            ``win_loss_logits`` is the KL-trained share-of-winners head (consumed
            by MCTS after softmax); ``score_pred`` is the MSE-trained
            normalized-net-worth head (auxiliary, not used by MCTS).
        """
        batch_size = game_state_data.shape[0]

        # Handle PyG Batch from DataLoader (training) vs raw tensor (inference)
        if hasattr(node_features_or_batch, "x"):
            # PyG Batch — reshape flat nodes to (B, N, F)
            node_features = node_features_or_batch.x.view(batch_size, self.config.num_hexes, -1)
        else:
            node_features = node_features_or_batch

        # Variable-N inputs: if the caller passed a state shorter than the
        # configured max-N layout (e.g., 4-player encoder output for a model
        # built with max_players=6), pad each row up to the max-N layout. This
        # mirrors what ``_forward_encoded_batch`` does on the inference path.
        if game_state_data.shape[1] != self.config.game_state_size:
            inferred = self._infer_num_players_from_state_size(game_state_data.shape[1])
            padded_rows = []
            inferred_t = torch.full(
                (batch_size,), inferred, dtype=torch.long, device=game_state_data.device
            )
            for b in range(batch_size):
                padded_rows.append(
                    _pad_state_to_max_players(game_state_data[b], inferred, MAX_PLAYERS)
                )
            game_state_data = torch.stack(padded_rows, dim=0)
            if num_players is None:
                num_players = inferred_t

        # Extract round type from game state if not provided. ``active_player_idx``
        # is no longer consumed by the value head (state is canonicalized so the
        # active player is always slot 0), so we don't need to derive it.
        if round_type_idx is None:
            round_type_idx = self._extract_round_type(game_state_data)
        del active_player_idx  # unused — kept for API compatibility

        # 1. Economic State Transformer. The padded player slots get masked
        # out of attention via ``num_players``; the returned key-padding mask
        # is passed through to the cross-modal fusion so it can mask the same
        # slots during the cross-attention pool.
        entity_summary, entity_embeds, entity_key_padding_mask = self.econ_transformer(
            game_state_data, num_players=num_players
        )
        del entity_summary

        # 2. Map encoder (transformer or resnet, depending on config).
        # Both variants expose the same (per_hex_embeddings, map_pool)
        # interface; only their call signatures differ — the Transformer
        # consumes axial coords + distance/direction/track matrices for its
        # structural attention bias, the ResNet only needs the per-hex feature
        # tensor (it bakes spatial structure into its offset-grid scatter).
        if self.map_encoder_kind == "transformer":
            track_conn = self.track_conn_computer(node_features, self.direction_matrix)
            node_embeds, map_pool = self.hex_transformer(
                node_features,
                self.axial_coords,
                self.distance_matrix,
                self.direction_matrix,
                track_conn,
            )
        else:
            node_embeds, map_pool = self.hex_transformer(node_features)

        # 3. Cross-Modal Fusion. Pass the entity padding mask so the
        # cross-attention pool ignores padded player slots when collapsing
        # the entity sequence to a single summary.
        trunk_input = self.fusion(
            entity_embeds, node_embeds, map_pool,
            entity_key_padding_mask=entity_key_padding_mask,
        )

        # 4. Phase-Conditioned Trunk
        phase_embed = self.phase_embedding(round_type_idx)
        x = trunk_input
        for block in self.res_blocks:
            x = block(x, phase_embed)

        # 5. Policy Head — hierarchical / autoregressive.
        # The flat ``policy_logits`` has log-probabilities baked into the
        # LayTile and PlaceToken blocks (so a downstream ``log_softmax`` over
        # the legal-action mask recovers the autoregressive joint); other
        # action types hold raw logits from ``other_head``. The per-sub-head
        # ``policy_components`` dict is stashed on the model as
        # ``last_policy_components`` so the decomposed training loss can fetch
        # it without changing the long-standing forward-return contract.
        policy_logits, policy_components = self.policy_head(x, node_embeds)
        self.last_policy_components = policy_components

        # 5b. Continuous price head — emits (mean, log_std) per legal
        # (action_type, entity) slot. Stashed on the model alongside
        # ``last_policy_components`` so the training loss + MCTS PW can
        # consume it without changing the long-standing 4-tuple forward
        # contract.
        price_mean, price_log_std = self.price_head(x)
        self.last_price_components = {
            "price_mean": price_mean,
            "price_log_std": price_log_std,
            "slot_index": self.price_head.slot_index,
            "num_slots": self.price_head.num_slots,
        }

        # 6. Dual Value Heads — both take the canonicalized trunk (active
        # player at slot 0), no explicit indicator. MCTS softmaxes
        # ``win_loss_logits`` and backs it up through the tree; ``score_pred``
        # is auxiliary and discarded at inference time.
        win_loss_logits = self.win_loss_head(x)
        score_pred = self.score_head(x)

        # 7. Auxiliary Heads
        aux_action_count_pred = self.aux_action_count_head(x)

        return policy_logits, win_loss_logits, score_pred, aux_action_count_pred
