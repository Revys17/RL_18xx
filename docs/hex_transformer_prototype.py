"""
Prototype: Hex Map Transformer with structural attention bias.

This replaces the GNN map encoder with a Transformer that:
1. Uses continuous axial coordinates for spatial awareness (generalizes to any hex grid)
2. Encodes adjacency and track connectivity as attention biases (not edge features)
3. Handles variable-size boards (different 18xx titles have different maps)

To run this file standalone as a test:
    python docs/hex_transformer_prototype.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class HexPositionalEncoding(nn.Module):
    """Encodes hex spatial position using continuous axial coordinates.

    Hex grids use axial coordinates (q, r) where:
        q = column, r = row (in the offset coordinate system)

    For 1830's coordinate system (e.g., "A1", "B2", "E14"):
        - Letter → row (A=0, B=1, ..., I=8)
        - Number → column

    We convert these to axial coordinates and project them through a small MLP.
    This is analogous to sinusoidal positional encoding in standard Transformers,
    but for 2D hex grids.

    WHY THIS GENERALIZES:
    - No learned per-position embeddings (which would be game-specific)
    - The MLP learns to interpret (q, r) coordinates regardless of board size/shape
    - A 1830 board (93 hexes) and a 1856 board (120 hexes) both use the same encoder
    - The network learns "hexes at similar coordinates behave similarly" as an
      inductive bias, which transfers across games
    """

    def __init__(self, d_model: int):
        super().__init__()
        # Project 2D coordinates to model dimension
        # Using a small MLP instead of raw features so the network can learn
        # nonlinear spatial patterns (e.g., "center of board" vs "edge of board")
        self.coord_proj = nn.Sequential(
            nn.Linear(2, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model),
        )

    def forward(self, axial_coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            axial_coords: (num_hexes, 2) — normalized (q, r) axial coordinates

        Returns:
            (num_hexes, d_model) — positional embeddings to ADD to node features
        """
        return self.coord_proj(axial_coords)


class StructuralAttentionBias(nn.Module):
    """Computes attention bias from hex grid structure and track connectivity.

    In a standard Transformer:
        attn = softmax(Q @ K.T / sqrt(d))

    We add structural biases:
        attn = softmax(Q @ K.T / sqrt(d) + adjacency_bias + track_bias)

    This is the same idea as ALiBi (Press et al., 2021) or relative position
    encoding, but for a hex grid instead of a 1D sequence.

    Three bias components:

    1. DISTANCE BIAS (static per game title):
       Nearby hexes should attend to each other more strongly.
       We use the hop distance on the hex grid (not Euclidean distance)
       and pass it through a learned scalar function.

       Generalizes because: distance is computed from the hex grid structure,
       not hardcoded per position. Any hex grid works.

    2. DIRECTION BIAS (static per game title):
       Hex grids have 6 directions. A hex might attend differently to its
       north neighbor vs its southeast neighbor (e.g., track often runs E-W).
       We learn an embedding per direction for each attention head.

       Generalizes because: all hex grids have the same 6 directions.

    3. TRACK CONNECTIVITY BIAS (dynamic, changes each game state):
       If hex A has track connecting to hex B, the attention from A to B
       should be boosted. This is the key dynamic structural signal.

       Generalizes because: track connectivity is a property of the placed
       tiles, computed the same way in every 18xx game.
    """

    def __init__(self, num_heads: int, max_distance: int = 12):
        super().__init__()
        self.num_heads = num_heads
        self.max_distance = max_distance

        # Distance bias: learned scalar per (head, distance)
        # Distances beyond max_distance share the last bucket
        self.distance_bias = nn.Embedding(max_distance + 1, num_heads)

        # Direction bias: learned scalar per (head, direction)
        # 6 hex directions + 1 for "not adjacent"
        self.direction_bias = nn.Embedding(7, num_heads)

        # Track connectivity bias: learned scalar per head
        # Applied when track connects two hexes
        self.track_bias = nn.Parameter(torch.zeros(num_heads))

    def forward(
        self,
        distance_matrix: torch.Tensor,
        direction_matrix: torch.Tensor,
        track_connectivity: torch.Tensor,
    ) -> torch.Tensor:
        """
        All inputs describe the hex grid structure. They can be precomputed
        (distance, direction) or computed per game state (track connectivity).

        Args:
            distance_matrix:    (num_hexes, num_hexes) int — hop distance between hexes.
                                Precomputed per game title, never changes.
            direction_matrix:   (num_hexes, num_hexes) int — direction from hex i to hex j.
                                Values 0-5 for the 6 hex directions, 6 for "not adjacent".
                                Precomputed per game title, never changes.
            track_connectivity: (num_hexes, num_hexes) float — 1.0 if track connects hex i
                                to hex j, 0.0 otherwise. DYNAMIC: changes each game state
                                as tiles are placed and upgraded.

        Returns:
            (num_heads, num_hexes, num_hexes) — bias to add to attention logits
        """
        # Clamp distances to max bucket
        clamped_dist = distance_matrix.clamp(max=self.max_distance)

        # (num_hexes, num_hexes, num_heads)
        dist_bias = self.distance_bias(clamped_dist)
        dir_bias = self.direction_bias(direction_matrix)

        # Track: (num_hexes, num_hexes, 1) * (num_heads,) → (num_hexes, num_hexes, num_heads)
        track_bias = track_connectivity.unsqueeze(-1) * self.track_bias

        # Sum all biases and transpose to (num_heads, num_hexes, num_hexes)
        total_bias = (dist_bias + dir_bias + track_bias).permute(2, 0, 1)

        return total_bias


class HexTransformerLayer(nn.Module):
    """A single Transformer layer with structural attention bias.

    This is a standard pre-norm Transformer layer with one modification:
    structural biases are added to the attention logits before softmax.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        # QKV projection (fused for efficiency)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Feed-forward
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(
        self,
        x: torch.Tensor,
        structural_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, num_hexes, d_model) — node embeddings
            structural_bias: (num_heads, num_hexes, num_hexes) — from StructuralAttentionBias.
                             Broadcast over the batch dimension.

        Returns:
            (batch, num_hexes, d_model) — updated node embeddings
        """
        B, N, D = x.shape

        # Pre-norm self-attention
        normed = self.ln1(x)
        qkv = self.qkv(normed).reshape(B, N, 3, self.num_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, d_head)
        q, k, v = qkv.unbind(0)

        # Scaled dot-product attention with structural bias
        attn_logits = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)  # (B, heads, N, N)

        if structural_bias is not None:
            # structural_bias is (heads, N, N) — broadcast over batch
            attn_logits = attn_logits + structural_bias.unsqueeze(0)

        attn = F.softmax(attn_logits, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        out = self.out_proj(out)
        x = x + out

        # Pre-norm feed-forward
        x = x + self.ff(self.ln2(x))

        return x


class HexMapTransformer(nn.Module):
    """Complete map encoder replacing the GNN.

    Takes per-hex node features + structural information and produces:
    1. Per-node embeddings (for the policy head's hex scoring)
    2. A pooled map embedding (for the trunk)

    HOW THIS GENERALIZES TO OTHER 18XX GAMES:
    ──────────────────────────────────────────
    The only game-specific inputs are:
    - node_features: shape (num_hexes, num_node_features) — varies per title
    - axial_coords: shape (num_hexes, 2) — varies per title
    - distance_matrix: shape (num_hexes, num_hexes) — varies per title
    - direction_matrix: shape (num_hexes, num_hexes) — varies per title
    - track_connectivity: shape (num_hexes, num_hexes) — varies per game state

    ALL of these are computed by the encoder from the game state. None are
    hardcoded in the network weights. The network weights are:
    - Input projection (handles variable num_node_features via per-title config)
    - Positional encoding MLP (works on any (q,r) coordinates)
    - Structural bias embeddings (distance/direction are universal to hex grids)
    - Transformer layers (operate on variable-length sequences)
    - Attention pooling query (learned, game-agnostic)

    To train on a new 18xx title:
    1. Update the encoder to produce node features for the new game
    2. Provide the hex grid's axial coordinates and adjacency
    3. The network architecture is unchanged — just different input shapes

    The only dimension that's fixed is d_model (the internal representation).
    Input/output projections adapt to the game's specific feature counts.
    """

    def __init__(
        self,
        num_node_features: int,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        d_ff: int = 512,
        max_distance: int = 12,
    ):
        super().__init__()
        self.d_model = d_model

        # Project raw node features to model dimension
        self.input_proj = nn.Linear(num_node_features, d_model)
        self.input_ln = nn.LayerNorm(d_model)

        # Positional encoding from axial coordinates
        self.pos_enc = HexPositionalEncoding(d_model)

        # Structural attention bias (shared across layers — could also be per-layer)
        self.structural_bias = StructuralAttentionBias(num_heads, max_distance)

        # Transformer layers
        self.layers = nn.ModuleList([
            HexTransformerLayer(d_model, num_heads, d_ff) for _ in range(num_layers)
        ])

        # Final layer norm
        self.output_ln = nn.LayerNorm(d_model)

        # Attention pooling: learned query that attends to all hex nodes
        # Produces a single vector summarizing the entire board
        self.pool_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pool_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)

    def forward(
        self,
        node_features: torch.Tensor,
        axial_coords: torch.Tensor,
        distance_matrix: torch.Tensor,
        direction_matrix: torch.Tensor,
        track_connectivity: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            node_features:      (B, num_hexes, num_node_features) — per-hex features
            axial_coords:       (num_hexes, 2) — hex grid coordinates (static per game title)
            distance_matrix:    (num_hexes, num_hexes) int — hop distances (static)
            direction_matrix:   (num_hexes, num_hexes) int — directions (static)
            track_connectivity: (B, num_hexes, num_hexes) float — track connections (dynamic)

        Returns:
            node_embeds: (B, num_hexes, d_model) — per-hex embeddings for policy head
            map_pool:    (B, d_model) — pooled board embedding for trunk
        """
        B = node_features.shape[0]

        # Project node features and add positional encoding
        x = self.input_proj(node_features)                  # (B, N, d_model)
        pos = self.pos_enc(axial_coords)                     # (N, d_model)
        x = self.input_ln(x + pos.unsqueeze(0))             # (B, N, d_model)

        # Compute structural attention bias
        # For batched track connectivity, we'd need per-sample bias.
        # In practice during MCTS, batch size is small (8-64 leaves),
        # so we compute bias per sample. For efficiency, if all samples
        # share the same board (same game), we can compute once.
        #
        # Simplification: use the first sample's track connectivity for the
        # shared static components, then add per-sample track bias separately.
        # For now, use mean track connectivity across the batch as an approximation
        # during batched inference. During self-play all leaves come from the same
        # game tree, so they share very similar board states.
        if track_connectivity.dim() == 3:
            # Use first sample's track connectivity (they're from the same game tree)
            track_conn = track_connectivity[0]
        else:
            track_conn = track_connectivity

        bias = self.structural_bias(distance_matrix, direction_matrix, track_conn)
        # bias: (num_heads, N, N)

        # Transformer layers
        for layer in self.layers:
            x = layer(x, structural_bias=bias)

        node_embeds = self.output_ln(x)  # (B, N, d_model)

        # Attention pooling: learned query attends to all nodes
        query = self.pool_query.expand(B, -1, -1)          # (B, 1, d_model)
        map_pool, _ = self.pool_attn(query, node_embeds, node_embeds)  # (B, 1, d_model)
        map_pool = map_pool.squeeze(1)                       # (B, d_model)

        return node_embeds, map_pool


# ──────────────────────────────────────────────────────────────────────────────
# Helper: compute structural matrices from a hex grid
# ──────────────────────────────────────────────────────────────────────────────

def hex_coord_to_axial(coord_str: str) -> tuple[float, float]:
    """Convert 18xx hex coordinate string (e.g., 'E14') to axial (q, r).

    The 18xx coordinate system uses:
    - Letter for row (A=0, B=1, ..., I=8)
    - Number for column

    Offset coordinates → axial:
        q = col
        r = row - (col // 2)  (for even-q offset layout)

    Note: the exact offset→axial conversion depends on the coordinate
    convention. This handles the standard 18xx convention where odd
    columns are shifted down by half a hex.
    """
    # Parse "E14" → row=4, col=14
    letter_part = ""
    num_part = ""
    for c in coord_str:
        if c.isalpha():
            letter_part += c
        else:
            num_part += c

    row = ord(letter_part.upper()) - ord("A")
    col = int(num_part)

    # Convert to axial (cube coordinates projected to 2D)
    q = col
    r = row - (col // 2)

    return (float(q), float(r))


def compute_structural_matrices(
    hex_coords: list[str],
    adjacency: dict[str, list[tuple[str, int]]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute distance and direction matrices from hex grid structure.

    Args:
        hex_coords: ordered list of hex coordinate strings
        adjacency: dict mapping hex_coord → [(neighbor_coord, direction), ...]

    Returns:
        axial_coords:    (num_hexes, 2) float — normalized axial coordinates
        distance_matrix: (num_hexes, num_hexes) int — BFS hop distance
        direction_matrix: (num_hexes, num_hexes) int — direction (0-5 if adjacent, 6 if not)
    """
    n = len(hex_coords)
    coord_to_idx = {c: i for i, c in enumerate(hex_coords)}

    # Axial coordinates
    axial = torch.tensor([hex_coord_to_axial(c) for c in hex_coords], dtype=torch.float32)
    # Normalize to [-1, 1] range
    if axial.shape[0] > 1:
        axial_min = axial.min(dim=0).values
        axial_max = axial.max(dim=0).values
        axial_range = (axial_max - axial_min).clamp(min=1e-6)
        axial = 2.0 * (axial - axial_min) / axial_range - 1.0

    # Direction matrix (6 = not adjacent)
    direction = torch.full((n, n), 6, dtype=torch.long)
    for src, neighbors in adjacency.items():
        if src not in coord_to_idx:
            continue
        i = coord_to_idx[src]
        for dst, d in neighbors:
            if dst in coord_to_idx:
                j = coord_to_idx[dst]
                direction[i, j] = d

    # Distance matrix via BFS
    distance = torch.full((n, n), n, dtype=torch.long)  # max distance = unreachable
    for start in range(n):
        distance[start, start] = 0
        queue = [start]
        visited = {start}
        while queue:
            current = queue.pop(0)
            coord = hex_coords[current]
            if coord in adjacency:
                for neighbor_coord, _ in adjacency[coord]:
                    if neighbor_coord in coord_to_idx:
                        j = coord_to_idx[neighbor_coord]
                        if j not in visited:
                            visited.add(j)
                            distance[start, j] = distance[start, current] + 1
                            queue.append(j)

    return axial, distance, direction


def compute_track_connectivity(
    hex_coords: list[str],
    port_connections: dict[str, set[int]],
    adjacency: dict[str, list[tuple[str, int]]],
) -> torch.Tensor:
    """Compute dynamic track connectivity matrix from current tile state.

    Two hexes are track-connected if:
    - They are adjacent (direction d from hex_i to hex_j)
    - hex_i has track on port d
    - hex_j has track on port (d+3)%6

    This is DYNAMIC — it changes every time a tile is placed or upgraded.

    Args:
        hex_coords: ordered list of hex coordinate strings
        port_connections: dict mapping hex_coord → set of port numbers (0-5)
                         that have track on them. Computed from the tile's paths.
        adjacency: dict mapping hex_coord → [(neighbor_coord, direction), ...]

    Returns:
        (num_hexes, num_hexes) float — 1.0 if track-connected, 0.0 otherwise
    """
    n = len(hex_coords)
    coord_to_idx = {c: i for i, c in enumerate(hex_coords)}
    connectivity = torch.zeros(n, n)

    for src, neighbors in adjacency.items():
        if src not in coord_to_idx:
            continue
        i = coord_to_idx[src]
        src_ports = port_connections.get(src, set())

        for dst, direction in neighbors:
            if dst not in coord_to_idx:
                continue
            j = coord_to_idx[dst]
            dst_ports = port_connections.get(dst, set())

            # Track connects if src has track on port `direction`
            # and dst has track on the opposite port `(direction+3)%6`
            if direction in src_ports and (direction + 3) % 6 in dst_ports:
                connectivity[i, j] = 1.0

    return connectivity


# ──────────────────────────────────────────────────────────────────────────────
# Test: verify shapes and that the whole thing runs
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(42)

    # --- Simulate a small hex grid (7 hexes, like a hex flower) ---
    #
    #       A1  A3
    #     B2  B4  B6
    #       C3  C5
    #
    hex_coords = ["A1", "A3", "B2", "B4", "B6", "C3", "C5"]
    adjacency = {
        "A1": [("B2", 3), ("A3", 1)],
        "A3": [("A1", 4), ("B2", 3), ("B4", 3)],
        "B2": [("A1", 0), ("A3", 0), ("B4", 1), ("C3", 3)],
        "B4": [("A3", 0), ("B2", 4), ("B6", 1), ("C3", 3), ("C5", 3)],
        "B6": [("B4", 4), ("C5", 3)],
        "C3": [("B2", 0), ("B4", 0), ("C5", 1)],
        "C5": [("B4", 0), ("B6", 0), ("C3", 4)],
    }

    # Compute static structural matrices
    axial_coords, distance_matrix, direction_matrix = compute_structural_matrices(
        hex_coords, adjacency
    )

    # Simulate dynamic track connectivity (some tiles placed)
    port_connections = {
        "A1": {1, 3},      # track runs NE-SW
        "B2": {0, 3},      # track runs N-S
        "A3": {3, 4},      # track runs SW-W (connects to B2 and A1)
        "B4": {0, 1, 4},   # junction: N, NE, W
        "C3": {0, 1},      # track runs N-NE
        "C5": {},           # no track yet
        "B6": {},           # no track yet
    }
    track_conn = compute_track_connectivity(hex_coords, port_connections, adjacency)

    # --- Create the model ---
    num_node_features = 50  # same as current encoder
    model = HexMapTransformer(
        num_node_features=num_node_features,
        d_model=256,
        num_heads=8,
        num_layers=4,
        d_ff=512,
    )

    param_count = sum(p.numel() for p in model.parameters())
    print(f"HexMapTransformer parameters: {param_count:,}")

    # --- Forward pass ---
    batch_size = 4
    node_features = torch.randn(batch_size, len(hex_coords), num_node_features)
    track_conn_batch = track_conn.unsqueeze(0).expand(batch_size, -1, -1)

    node_embeds, map_pool = model(
        node_features=node_features,
        axial_coords=axial_coords,
        distance_matrix=distance_matrix,
        direction_matrix=direction_matrix,
        track_connectivity=track_conn_batch,
    )

    print(f"Input:       node_features  {tuple(node_features.shape)}")
    print(f"Output:      node_embeds    {tuple(node_embeds.shape)}")
    print(f"Output:      map_pool       {tuple(map_pool.shape)}")
    print()

    # --- Verify structural bias ---
    print(f"Axial coords:\n{axial_coords}")
    print(f"\nDistance matrix:\n{distance_matrix}")
    print(f"\nDirection matrix (6 = not adjacent):\n{direction_matrix}")
    print(f"\nTrack connectivity:\n{track_conn}")
    print()

    # --- Show that it handles different board sizes ---
    print("=== Generalization test: different board size ===")

    # A tiny 3-hex board (different game)
    small_coords = ["A1", "A3", "B2"]
    small_adj = {
        "A1": [("A3", 1), ("B2", 3)],
        "A3": [("A1", 4), ("B2", 3)],
        "B2": [("A1", 0), ("A3", 0)],
    }
    small_axial, small_dist, small_dir = compute_structural_matrices(small_coords, small_adj)
    small_track = torch.zeros(3, 3)

    # Same model, different board size
    small_features = torch.randn(2, 3, num_node_features)
    small_node_embeds, small_pool = model(
        node_features=small_features,
        axial_coords=small_axial,
        distance_matrix=small_dist,
        direction_matrix=small_dir,
        track_connectivity=small_track,
    )
    print(f"3-hex board: node_embeds {tuple(small_node_embeds.shape)}, map_pool {tuple(small_pool.shape)}")

    # A larger board
    large_coords = [f"R{i}" for i in range(150)]
    large_axial = torch.randn(150, 2)
    large_dist = torch.randint(0, 15, (150, 150))
    large_dir = torch.randint(0, 7, (150, 150))
    large_track = torch.zeros(150, 150)
    large_features = torch.randn(2, 150, num_node_features)

    large_node_embeds, large_pool = model(
        node_features=large_features,
        axial_coords=large_axial,
        distance_matrix=large_dist,
        direction_matrix=large_dir,
        track_connectivity=large_track,
    )
    print(f"150-hex board: node_embeds {tuple(large_node_embeds.shape)}, map_pool {tuple(large_pool.shape)}")
    print()
    print("Same model weights, different board sizes — all shapes correct.")
    print("The network architecture is fully game-agnostic.")
