# Network Architecture v2 — Design Document

## Context

### What 1830 is, for the network

1830 is a 4-player, full-information, deterministic economic board game. Three interleaved decision types:

1. **Auction** — bid on private companies (6 companies, ~20 legal actions)
2. **Stock** — buy/sell corporation shares, set par prices (~100 legal actions)
3. **Operating** — lay tiles, place tokens, run routes, buy trains (~5-500 legal actions)

The *economic* state (who owns what, share prices, cash flows) drives strategy more than the *spatial* state (which tiles are where). But the spatial state determines route revenue, which drives the economy. They're coupled.

Key characteristics:
- **Long games**: 200-500 moves
- **Huge action space**: 26,535 total actions, with many legal at once (esp. tile placement)
- **Phase transitions**: game phases (2-train → 3-train → ... → D-train) fundamentally change the strategic landscape — early game is about positioning, late game is about extracting revenue
- **Multi-agent**: 4 players, non-zero-sum (total wealth grows), but only the richest player wins
- **Sparse rewards**: outcome only known at game end
- **Structure in actions**: LayTile = (hex, tile, rotation), BuyTrain = (source, type, price), etc.

### What was wrong with v1

The v1 architecture had:
- 25.6M parameters, 71.6% in a single `nn.Bilinear` layer in the policy head
- A GNN processing map state into a single 256-dim vector (aggressive compression)
- A shallow 2-layer MLP for the 390-dim economic state vector
- FiLM conditioning applied *after* residual blocks (can only scale outputs, not modulate processing)
- No hex positional encoding (GNN is spatially blind)
- No history features (single snapshot)
- BatchNorm + LayerNorm stacked at fusion
- Dropout disabled everywhere (0.0)
- No action masking inside the network

### Design goals for v2

1. **Right-size the parameter budget** — no single component should dominate
2. **Give each input modality appropriate capacity** — economic state is complex, not just 2 layers
3. **Phase-aware processing** — fundamentally different evaluation by game phase
4. **Structured policy head** — exploit action structure without massive parameter count
5. **Stable for RL training** — work with small batches, no train/eval distribution mismatch

---

## Architecture

### Overview

```
  ┌─────────────────────────────┐    ┌──────────────────────────┐
  │   Economic State Encoder    │    │     Map State Encoder     │
  │   (390-dim → Transformer)   │    │  (93 nodes × GNN + pos)  │
  └────────────┬────────────────┘    └────────────┬─────────────┘
               │ (B, econ_dim)                    │ (B, 93, map_dim) + (B, map_pool_dim)
               └──────────┬───────────────────────┘
                          │
                 Cross-Modal Fusion
                          │
                          ▼
              Phase-Conditioned Trunk
              (N × ResBlock with FiLM)
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
    Policy Head      Value Head      Aux Heads
   (structured)    (per-player)    (action count)
```

### 1. Economic State Encoder

**Problem with v1**: 390 features → 2 linear layers. This flattens all cross-entity relationships.

**v2 approach**: Treat the 390-dim vector as a set of *entity groups* and use a small Transformer to capture cross-entity interactions.

The game state vector has natural groupings:
- Per-player features (cash, certs, shares, turn order) — 4 groups
- Per-corporation features (floated, cash, trains, price, shares, tokens, market zone) — 8 groups
- Per-private features (ownership, revenue, closed) — 6 groups
- Global features (round type, phase, bank cash, OR structure, train limit) — 1 group

**Architecture**:
```
Entity groups (19 groups):
  - 4 player groups:  each ~15 features → project to d_entity
  - 8 corp groups:    each ~18 features → project to d_entity
  - 6 private groups: each ~12 features → project to d_entity
  - 1 global group:   ~10 features → project to d_entity

Each group → Linear(group_size, d_entity)          # per-type projection
Add learned entity-type embedding (player/corp/private/global)
Add learned entity-id embedding (which player/corp/private)

→ Transformer encoder (2-3 layers, 4 heads, d_entity=128)
→ [CLS] token output = econ_embed (128-dim)
   Per-entity outputs preserved for cross-attention with map
```

**Why**: The Transformer captures interactions like "player 2 owns 60% of PRR, PRR has $500 and no trains, the depot has 4-trains available." These cross-entity relationships are exactly what makes 1830 strategy work — and they're invisible to an MLP that processes the flat 390-dim vector.

**Parameter cost**: ~600K (vs v1's 166K, but much more expressive)

### 2. Map State Encoder (Hex Transformer — replaces GNN)

**Why replace the GNN**: The GATv2 GNN has a local receptive field — 4 layers sees 4 hops. But 1830 routes span 8+ hexes (e.g., NYC to Chicago). The GNN literally can't evaluate a full route in one forward pass. A Transformer sees all 93 hexes simultaneously. With the hex grid being only 93 nodes, there's zero efficiency concern with full self-attention.

**Architecture**: See `docs/hex_transformer_prototype.py` for the complete, runnable implementation.

Core components:
1. **HexPositionalEncoding**: Projects continuous axial (q, r) coordinates through an MLP. No learned per-position embeddings — works on any hex grid size/shape.

2. **StructuralAttentionBias**: Three bias terms added to attention logits before softmax:
   - *Distance bias*: learned per (head, hop_distance). Precomputed via BFS, static per title.
   - *Direction bias*: learned per (head, direction). The 6 hex directions are universal.
   - *Track connectivity bias*: learned per head, applied when track connects two hexes. **Dynamic** — changes every time a tile is placed.

3. **Transformer layers**: Standard pre-norm Transformer with structural bias. 4 layers, 8 heads.

4. **Attention pooling**: Learned query vector attends to all node embeddings → single board summary vector.

The map encoder produces *two* outputs:
1. `node_embeds`: (B, 93, d_map) — per-hex embeddings, used by the policy head for tile placement
2. `map_pool`: (B, d_map) — pooled board embedding for the trunk

**Generalization to other 18xx titles**: The network weights are fully game-agnostic. All game-specific information comes from the inputs:
- `node_features` shape varies per title → handled by input projection config
- `axial_coords` vary per title → continuous, computed from hex grid
- `distance_matrix` / `direction_matrix` → computed from hex adjacency
- `track_connectivity` → computed from tile state per game step

Same model weights work on 1830 (93 hexes), 1856 (120 hexes), or any other title. See the generalization test in the prototype file.

**Parameter cost**: ~2.4M (vs v1's 1.8M GNN — modest increase for global receptive field)

### 3. Cross-Modal Fusion

**Problem with v1**: Gated fusion compresses everything into a single vector. The gate input uses raw embeddings but applies to projected versions. BN + LN stacked.

**v2 approach**: Two-stage fusion.

**Stage A — Cross-attention** (economic state attends to map):
```
econ_entities (19 × d_entity) cross-attend to node_embeds (93 × d_map)
→ map-informed entity embeddings (19 × d_entity)
→ mean pool → econ_fused (d_entity)
```

This lets the economic encoder ask "what does PRR's map position look like?" by attending to the hexes where PRR has tokens. Much richer than gating two pooled vectors. The Hex Transformer's per-node embeddings (`node_embeds`) serve as keys/values here.

**Stage B — Concatenate and project**:
```
concat(econ_fused, map_pool) → Linear → LayerNorm → trunk_input (d_trunk)
```

Simple concat + project. No gating, no BN. LayerNorm only.

**Parameter cost**: ~200K

### 4. Phase-Conditioned Trunk

**v2 changes from v1**:
- FiLM applied *inside* residual blocks (before second linear), not after
- FiLM initialized to identity (gamma=1, beta=0)
- Use LayerNorm instead of BatchNorm in res blocks
- 6 res blocks (vs 7) — reclaim parameters from the bloated policy head

**ResBlock with FiLM**:
```python
class FiLMResBlock(nn.Module):
    def __init__(self, d_trunk, d_film):
        self.fc1 = nn.Linear(d_trunk, d_trunk)
        self.ln1 = nn.LayerNorm(d_trunk)
        self.fc2 = nn.Linear(d_trunk, d_trunk)
        self.ln2 = nn.LayerNorm(d_trunk)
        self.film = nn.Linear(d_film, d_trunk * 2)
        # Initialize FiLM to identity
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        self.film.bias.data[:d_trunk] = 1.0  # gamma = 1

    def forward(self, x, phase_embed):
        residual = x
        out = F.gelu(self.ln1(self.fc1(x)))
        # FiLM modulates INSIDE the block
        gamma, beta = self.film(phase_embed).chunk(2, dim=1)
        out = gamma * out + beta
        out = self.ln2(self.fc2(out))
        return F.gelu(out + residual)
```

**Why inside**: This lets the phase embedding control *how* the block transforms features, not just scale the output. During auction rounds, the network can learn to emphasize private-company features. During operating rounds, it can emphasize route-revenue features.

**Parameter cost**: ~3.2M (similar to v1)

### 5. Policy Head

The action space breaks down as:
| Action type | Count | Structure |
|-------------|-------|-----------|
| Pass | 1 | flat |
| Bid | 6 | per-company |
| Par | 48 | corp × price |
| BuyShares | 16 | corp × location |
| SellShares | 40 | corp × num_shares |
| PlaceToken | ~30 | hex × city_slot |
| **LayTile** | **25,668** | **hex × tile × rotation** |
| BuyTrain | ~720 | source × type × price |
| Dividend | 2 | payout/withhold |
| Other | ~4 | bankrupt, run routes, etc. |

LayTile dominates (96.7% of the action space). The rest is 867 actions.

**v2 approach**: Keep the factored structure but fix the parameter explosion.

**LayTile head** — Low-rank interaction + attention hex scoring:
```python
# Hex scores from per-node GNN embeddings (Phase 5.6 — keep this, it's good)
hex_logits = hex_scorer(node_embeds)           # (B, 93) from per-node MLP

# Tile logits from trunk
tile_logits = tile_head(trunk)                 # (B, 46)

# Low-rank interaction instead of full Bilinear
hex_proj = Linear(93, rank)(hex_logits)        # (B, rank)  — rank=32
tile_proj = Linear(46, rank)(tile_logits)      # (B, rank)
# Outer product in projected space, then project back
interaction = hex_proj.unsqueeze(2) * tile_proj.unsqueeze(1)  # (B, rank, rank)
hex_tile = Linear(rank*rank, 93*46)(interaction.flatten(1))   # (B, 93*46)
hex_tile = hex_tile.view(B, 93, 46)

# Add rotation
rot_logits = rotation_head(trunk)              # (B, 6)
lay_tile_logits = hex_tile.unsqueeze(3) + rot_logits.unsqueeze(1).unsqueeze(2)  # (B, 93, 46, 6)
```

**Parameter cost**: rank=32 → ~32K + 1K + ~94K projection = ~130K total for hex-tile interaction (vs 18.3M for the Bilinear).

Alternative (even simpler, may be sufficient):
```python
# Just outer sum — 0 extra parameters for interaction
lay_tile_logits = hex_logits.unsqueeze(2).unsqueeze(3) + \
                  tile_logits.unsqueeze(1).unsqueeze(3) + \
                  rot_logits.unsqueeze(1).unsqueeze(2)
```

This is the original independent factoring but with the attention-based hex scoring from 5.6. The legal action mask does most of the work eliminating impossible combos. Start with this and only add the low-rank interaction if policy learning on tile placement is clearly struggling.

**Other actions head** — Structured by type:
```python
# Instead of one big Linear(512, 867), separate sub-heads per action type:
bid_logits = Linear(d_trunk, 6)(trunk)
par_logits = Linear(d_trunk, 48)(trunk)
buy_shares_logits = Linear(d_trunk, 16)(trunk)
sell_shares_logits = Linear(d_trunk, 40)(trunk)
# ... etc
# Concatenate in action-index order
```

This costs the same total parameters but gives each action type its own weights. Structural prior that different action types need different features.

**Masking inside the head**: Apply the legal action mask before returning logits:
```python
masked_logits = full_logits.masked_fill(~legal_mask.bool(), float('-inf'))
return masked_logits
```

This means downstream code never sees raw logits for illegal actions. Softmax concentrates probability mass on legal actions *inside* the network, so the policy head's gradients only flow through legal actions.

**Total policy head parameter cost**: ~500K (vs v1's 18.8M)

### 6. Value Head

Keep v1's improvements (per-player indicator, 3 layers) but with LayerNorm instead of no normalization:
```python
value_input = concat(trunk, one_hot_player)  # (B, d_trunk + 4)
→ Linear(d_trunk+4, d_trunk//2) → LayerNorm → GELU
→ Linear(d_trunk//2, d_trunk//2) → LayerNorm → GELU
→ Linear(d_trunk//2, 4)  # per-player value logits
```

**Parameter cost**: ~200K

### 7. Auxiliary Heads

Keep the legal-action-count predictor from Phase 5.4. Add one more:

**Game phase predictor**: Predict the current phase index from trunk features. This is trivially available from the input, but forcing the trunk to reconstruct it ensures phase information survives the trunk's transformations. Useful as a diagnostic — if the trunk can't predict the phase, the FiLM conditioning isn't working.

```python
phase_pred = Linear(d_trunk, num_phases)(trunk)
phase_loss = CrossEntropy(phase_pred, true_phase)
```

**Parameter cost**: ~2K (negligible)

---

## On dropout in RL

Your instinct is correct — dropout causes a train/eval mismatch that's problematic in RL:

1. During self-play (eval mode), dropout is off → deterministic policy
2. During training (train mode), dropout is on → stochastic forward pass
3. The MCTS policy targets were generated by the eval-mode network
4. Training the network with dropout means it's learning to match targets that came from a *different* function

This is especially bad early in training when the network is changing rapidly.

**However**, with 0.0 dropout and ~7M parameters (after fixing the bilinear), you'll overfit to the limited self-play data. Options:

- **Weight decay** (already using 1e-4) — this is the primary regularizer and fine for RL
- **Spectral normalization** — constrains layer Lipschitz constants, no train/eval gap
- **Stochastic depth** (drop entire res blocks during training) — coarser than dropout, less distribution mismatch
- **Data augmentation** — player permutation symmetry: for any training position, permuting the player indices gives a valid training example with permuted value targets. This is free 4x data augmentation.

Recommendation: keep dropout at 0.0, rely on weight decay + player permutation augmentation. If overfitting is still a problem, try spectral normalization before dropout.

---

## On history features

The current state contains *what* — who has how much money, what tiles are laid. But it doesn't contain *momentum* — is player 2 accumulating shares of B&O? Has the share price been rising or falling? Did someone just dump shares (affecting future strategy)?

Two approaches:

**Option A — Stacked frames** (simple, like AlphaGo):
Encode the last N game states and concatenate/stack them. For the economic state, this means N × 390 features. The Transformer entity encoder handles this naturally — add a "time step" positional embedding to each entity group.

**Problem**: N full game states is expensive. 1830 moves include many low-information forced moves (pass, run routes). Most of the 390 features don't change between moves.

**Option B — Delta encoding** (more efficient):
Compute the *change* in key features over the last N moves:
- Δ cash per player (last 1, 5, 10 moves)
- Δ share price per corp (last 1, 5, 10 moves)
- Δ shares owned per player-corp pair (last 1, 5 moves)
- Last action type (one-hot)
- Actions since last stock round / operating round

This adds ~50-100 scalar features to the game state vector. Much cheaper than full frame stacking and captures the most important temporal signals.

**Recommendation**: Start with Option B. Add delta features to the encoder. If the entity Transformer is used, these naturally slot in as additional features per entity group.

---

## On track connectivity (#11)

The current GNN uses static edges (hex adjacency with direction). Adjacency never changes, but *track connectivity* does — hex A might have track pointing toward hex B, but hex B might not have track pointing back. This is partially captured in node features (`connects_i_j`), but encoded *within* each node, not *between* nodes.

**v2 approach (Hex Transformer)**: Track connectivity is encoded as a structural attention bias. The `StructuralAttentionBias` module computes a per-head scalar boost for hex pairs that are track-connected. This is computed dynamically per game state from the placed tiles. See `compute_track_connectivity()` in `docs/hex_transformer_prototype.py` for the exact computation.

This is cleaner than the GNN approach because:
- Connectivity is a *relationship* between hexes, not a property of a single hex
- The attention bias directly says "pay more attention to hexes you're connected to"
- No need to choose between static edges with dynamic attributes vs dynamic edge existence — the full attention matrix handles both

---

## Model sizing

### Why not 500-600MB?

A 500MB model (FP32) is ~130M parameters. This is catastrophically wrong for this problem for two independent reasons:

**1. Self-play throughput collapse**

AlphaZero generates its own training data through self-play. Every move requires ~200 MCTS readouts, each needing a network forward pass. A 300-move game = ~60,000 forward passes. Model size directly controls how many games you can generate per day:

| Model size | Params | Est. games/day | Training examples/day |
|-----------|--------|----------------|----------------------|
| 7M  (v2 baseline) | 7M | ~7,700 | ~2.3M |
| 15M (v2 scaled) | 15M | ~5,800 | ~1.7M |
| 23M (AlphaZero-scale) | 23M | ~4,200 | ~1.3M |
| 130M (500MB) | 130M | ~1,100 | ~0.3M |

A 130M model generates **7x fewer games** than a 7M model. Since early AlphaZero training is data-starved by design (the model improves → generates better data → improves), throughput is the primary bottleneck.

**2. Overfitting with limited self-play data**

A rough rule of thumb: you need 10-100x as many training examples as parameters to avoid overfitting. With self-play:

| Model | Params | Min. examples needed | Games needed | Days at model's throughput |
|-------|--------|---------------------|-------------|---------------------------|
| 7M | 7M | 70M-350M | 230K-1.2M | 30-150 days |
| 23M | 23M | 230M-1.2B | 770K-4M | 180-950 days |
| 130M | 130M | 1.3B-6.5B | 4.3M-22M | 3,900-20,000 days |

A 130M model needs ~60x more data than a 7M model, but generates 7x less per day. The scaling works against you in both directions.

**3. Comparison to known AlphaZero implementations**

- **AlphaGo Zero** (19×19 Go, state-of-the-art board game AI): **23M params** (~90MB)
- **AlphaZero** (chess + shogi + Go): **23M params**
- **KataGo** (best public Go AI): **20-60M params**
- **Leela Chess Zero** (largest variant): ~350M params — but trained on **billions** of games across thousands of GPUs over years

DeepMind used 23M params for Go, which has a 19×19 = 361 action space and is widely considered more strategically complex than 1830. Our action space is larger (26K), but the legal action mask means the *effective* branching factor per move is similar (5-200 legal actions).

Your 3D sports game likely uses a different training paradigm — probably policy gradient methods (PPO, SAC) with environment simulators that can generate millions of frames per hour. That pipeline supports large models because data is cheap. In AlphaZero, data is expensive.

### Recommended sizing

**Target: 7-15M parameters (~30-60MB FP32)**

This is the sweet spot where:
- Self-play throughput is high enough to generate meaningful data per iteration
- The model has enough capacity for 1830's strategic complexity
- Overfitting is manageable with weight decay + player permutation augmentation
- FP16 inference (Phase 6.5) brings this to 15-30MB, fast enough for real-time MCTS

Start at the v2 baseline (~7M) and scale up once you've validated that:
1. The model trains successfully and loss decreases
2. Self-play games show improving play quality
3. The model is *underfitting* (training loss plateaus but validation loss could go lower)

If underfitting is observed, scale up by:
- Increasing `d_trunk` (512 → 768 → 1024) — this scales the trunk + res blocks
- Adding more res blocks (6 → 8 → 10)
- Increasing `d_map` (256 → 384)

Each step roughly doubles the parameter count. Don't go past ~30M unless you have multi-GPU self-play generating 5K+ games per day.

---

## Summary: parameter budget

| Component | v1 Params | v2 Params | Notes |
|-----------|-----------|-----------|-------|
| Economic encoder | 166K | 600K | Transformer over entity groups |
| Map encoder | 1.8M (GNN) | 2.4M (Hex Transformer) | Global attention + structural bias |
| Cross-modal fusion | 530K | 200K | Cross-attention replaces gating |
| Trunk (res blocks) | 3.7M | 3.2M | 6 blocks, LayerNorm, FiLM inside |
| FiLM layers | 237K | 200K | Identity init, inside blocks |
| Policy head | **18.8M** | **500K** | Outer-sum factoring + attention hex scorer |
| Value head | 199K | 200K | Same design, LayerNorm added |
| Aux heads | 1K | 3K | + phase predictor |
| **Total** | **25.6M** | **~7.3M** | **3.5x smaller, better distributed** |

The parameter budget is now dominated by the trunk (44%) and Hex Transformer (33%), which is where it should be — those are the components doing the actual reasoning.

---

## Implementation plan

### Step 1: Encoder changes (can do independently)
- Add hex axial coordinates to encoder output (for positional encoding)
- Compute distance/direction matrices from hex grid (precompute once per title, cache)
- Compute dynamic track connectivity from tile state (per encoding call)
- Add delta features for temporal signal
- Restructure game state vector into entity groups (add group boundaries to encoder output)

### Step 2: Model architecture
- Hex Transformer map encoder (see `docs/hex_transformer_prototype.py`)
- Economic state Transformer encoder
- Cross-attention fusion
- FiLM-inside res blocks with identity init
- Simplified policy head (outer-sum + attention hex scorer)
- Structured other-actions sub-heads
- In-network legal action masking
- LayerNorm everywhere, no BatchNorm
- Player permutation augmentation in training

### Step 3: Training adjustments
- Remove policy masking logic from `train.py` (now inside network)
- Add phase prediction aux loss
- Add player permutation augmentation to data loader
- Verify gradient flow with the new architecture (log gradient norms per component)

### Step 4: Validation
- Sanity check: run 1 self-play game end-to-end
- Compare forward pass time vs v1 (should be faster — 3.5x fewer params)
- Train for 1 iteration, verify loss decreases
- Compare policy entropy early in training (should be higher with better factoring)

### Scaling strategy
- Start at v2 baseline (~7M params)
- Monitor training/validation loss gap
- If underfitting: scale `d_trunk` (512 → 768), add res blocks (6 → 8)
- If overfitting: increase weight decay, add player permutation augmentation
- Don't exceed ~30M params without multi-GPU self-play infrastructure
