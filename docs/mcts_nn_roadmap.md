# MCTS & Neural Network Improvement Roadmap

Based on the review in `docs/mcts_nn_review.md` and subsequent discussion.

## Phase 1: Critical bug fixes (do before next training run)

### 1.1 Fix temperature in `pick_move` (CRITICAL)
**File**: `self_play.py:123`, `mcts.py:431-438`

`children_as_pi(squash=True)` applies `probs**0.98`, which is effectively a no-op. The exponent should be `1/T` where T=1.0 early (exploration) and T→0 late (exploitation).

**Changes**:
- Replace `squash` boolean with a `temperature` parameter on `children_as_pi`
- `children_as_pi(temperature)` computes `probs^(1/T)` then renormalizes
- In `pick_move`, use `temperature=1.0` when `move_number < softpick_move_cutoff`, and select `argmax` otherwise (equivalent to T→0)
- Add `temperature` to `SelfPlayConfig` if we want it configurable
- Update `play_move` to pass temperature to `children_as_pi` for the `searches_pi` recording

### 1.2 Fix `sim_count_this_move` UnboundLocalError (MEDIUM)
**File**: `self_play.py:434, 472`

`sim_count_this_move` is only defined inside the `else` branch (when MCTS runs), but referenced unconditionally at line 472.

**Fix**: Initialize `sim_count_this_move = 0` before the `if/else` block at ~line 425.

### 1.3 Fix `tree_search_duration_this_move` timing (MEDIUM)
**File**: `self_play.py:401, 440, 468`

`tree_search_duration_this_move` is initialized to 0 and never updated. Line 440 computes `time.time() - 0`.

**Fix**: Record `tree_search_start = time.time()` before the MCTS loop, compute `tree_search_duration_this_move = time.time() - tree_search_start` after it. For forced moves (1 legal action), duration stays 0.

---

## Phase 2: Training correctness

### 2.1 Fix policy loss masking (HIGH)
**File**: `train.py:125-131`

Current approach multiplies `log_softmax` output by a 0/1 mask, then adds `1e-8`. This works accidentally but is fragile.

**Change to**:
```python
# Get raw logits from model (need to return logits instead of / in addition to log_probs)
masked_logits = policy_logits.masked_fill(legal_action_mask == 0, float('-inf'))
log_probs = F.log_softmax(masked_logits, dim=1)
policy_loss = -torch.sum(pi * log_probs, dim=1).mean()
```

This requires the model's `forward` to return raw policy logits. Currently `forward` returns `(probabilities, log_probs, value_logits)`. Options:
- **Option A**: Return `(policy_logits, value_logits)` from forward, compute softmax/log_softmax in callers as needed. Cleaner separation.
- **Option B**: Return logits as a 4th element. More backward compatible.

Option A is cleaner. The callers that need probabilities (MCTS inference) can apply softmax themselves. Update:
- `model.py:forward` — return `(policy_logits, value_logits)`
- `model.py:run_many_encoded` — apply softmax/log_softmax here for MCTS callers
- `train.py` — apply masked log_softmax as above

---

## Phase 3: Model architecture improvements (next model iteration)

These changes alter the model architecture, so they require retraining from scratch (or careful weight migration). Bundle them together.

### 3.1 Add GNN residual connections
**File**: `model.py:340-348`

Add skip connections to the GAT layers for better gradient flow. The first layer needs a linear projection to match dimensions since `gnn_node_proj_dim` != `gnn_heads * gnn_hidden_dim_per_head`.

**Changes**:
- Add a `nn.Linear` projection for the first layer's skip connection
- For subsequent layers (where input dim == output dim), use identity skip: `node_repr = gat_output + node_repr`

### 3.2 Fix Kaiming init nonlinearity
**File**: `model.py:257`

Change `nonlinearity="relu"` to `nonlinearity="leaky_relu"` (closest supported approximation to GELU in PyTorch's kaiming init). Minor but trivial to fix.

### 3.3 Improve FactoredPolicyHead (MEDIUM)
**File**: `model.py:42-109`

The independence assumption `log P(h,t,r) = log P(h) + log P(t) + log P(r)` means the network cannot express (hex, tile) correlations. The legal mask compensates but wastes network capacity and slows policy learning.

**Proposed approach**: Autoregressive factoring
```
log P(h,t,r) = log P(h) + log P(t|h) + log P(r|h,t)
```

Implementation sketch:
- `hex_head`: unchanged, `trunk -> num_hexes`
- `tile_head`: takes `concat(trunk, hex_embedding)` -> `num_tiles`, where `hex_embedding` comes from a learned hex embedding table indexed by argmax(hex_head) during inference or teacher-forced during training
- `rotation_head`: takes `concat(trunk, hex_embedding, tile_embedding)` -> `num_rotations`
- During training, teacher-force with the actual hex/tile from the target policy
- During inference, use argmax or sample from each sub-head sequentially
- The joint log-prob is still the sum, but now each term is conditioned

**Complexity**: This is the most involved change. Requires careful handling of the training vs inference paths and ensuring gradients flow correctly through the teacher forcing.

**Alternative (simpler)**: Add a bilinear interaction term between hex and tile logits:
```python
hex_logits = self.hex_head(x)           # (B, H)
tile_logits = self.tile_head(x)         # (B, T)
interaction = self.bilinear(hex_logits, tile_logits)  # (B, H, T)
lay_tile_logits = interaction.unsqueeze(3) + rot_log.unsqueeze(1).unsqueeze(2)
```
This captures hex-tile correlation without full autoregressive complexity.

---

## Phase 4: Self-play throughput (before next training run)

These don't change the model — they make self-play faster so each training iteration generates more data.

### 4.1 Lazy encoding (defer to leaf evaluation)
**File**: `mcts.py:111-116`

Every `MCTSNode` encodes the game state at construction time. Most nodes created by `maybe_add_child` are never selected as leaves — their encoding work is wasted and then discarded when the subtree is pruned.

**Change**: Store only the game state at construction. Encode on demand when the node is selected as a leaf:
```python
def __init__(self, game_state, ...):
    self.game_object = game_state
    self.encoded_game_state = None  # lazy

def ensure_encoded(self):
    if self.encoded_game_state is None:
        if isinstance(self.game_object, RustGameAdapter):
            self.encoded_game_state = _rust_encode(self.game_object)
        else:
            self.encoded_game_state = Encoder_1830.get_encoder_for_model(
                self.config.network
            ).encode(self.game_object)
```

Call `ensure_encoded()` in `select_leaf` just before returning the leaf, and in `tree_search` before passing to the network. Also reduces memory since pruned subtrees never allocate tensor data.

### 4.2 Increase parallel readouts to saturate GPU
**File**: `config.py:114`, `self_play.py:143-210`

`parallel_readouts=8` means 25 network forward passes per move (200 readouts / 8). Batch size 8 almost certainly doesn't saturate the GPU.

**Change**: Increase `parallel_readouts` to 32-64. The virtual loss mechanism already prevents duplicate leaf selection. Scale `max_select_leaf_attempts` proportionally (currently `parallel_readouts * 2`). Benchmark to find the sweet spot where GPU utilization is high but leaf selection doesn't stall.

### 4.3 Adaptive readouts by position complexity
**File**: `self_play.py:425-438`

Every move gets exactly `num_readouts=200` simulations regardless of position complexity. A forced train purchase with 3 options gets the same budget as a tile lay with 150+ options.

**Change**: Scale readouts by position complexity:
```python
base = self.config.num_readouts
n_legal = player.root.num_legal_actions
if n_legal <= 5:
    readouts = base // 4
elif n_legal <= 20:
    readouts = base // 2
else:
    readouts = base
```

Optionally boost for late operating rounds where decisions have the highest leverage. Add a `min_readouts` floor to `SelfPlayConfig`.

### 4.4 Cache ActionMapper instance
**File**: `mcts.py:117`

`ActionMapper()` is re-instantiated for every node. It's stateless — cache a single instance on `SelfPlayConfig` or as a module-level singleton.

---

## Phase 5: Training signal improvements (next model iteration, bundle with Phase 3)

These change model architecture or loss function. Retrain from scratch alongside Phase 3 changes.

### 5.1 FiLM phase conditioning on shared trunk
**File**: `model.py:365-368`

The shared trunk processes all positions identically, but auction, stock, and operating positions need fundamentally different evaluation strategies. Round type is just one float in a 390-dim vector — easy for the network to ignore.

**Change**: Add a learned phase embedding with FiLM (Feature-wise Linear Modulation) applied per residual block:
```python
self.phase_embedding = nn.Embedding(num_phases, embed_dim)
self.film_layers = nn.ModuleList([
    nn.Linear(embed_dim, shared_trunk_dim * 2)  # gamma + beta
    for _ in range(num_res_blocks)
])

# In forward, per res block:
phase_embed = self.phase_embedding(round_type_idx)
gamma, beta = self.film_layers[i](phase_embed).chunk(2, dim=1)
current_features = gamma * current_features + beta
```

Negligible parameter cost. Gives the network an explicit mode switch for different game phases.

### 5.2 Per-player value head with active-player signal
**File**: `model.py:242-246`

The value head outputs 4 logits with no indication of whose turn it is. The active player's value estimate is more important, and evaluating your own corporations is qualitatively different from evaluating opponents'.

**Change**: Concatenate a one-hot active-player indicator to the trunk output before the value head:
```python
player_indicator = F.one_hot(active_player_idx, num_classes=4).float()
value_input = torch.cat([trunk_output, player_indicator], dim=1)
value_logits = self.value_head(value_input)
```

Requires the encoder to pass `active_player_idx` through to the model forward pass.

### 5.3 Deeper value head
**File**: `model.py:242-246`

The value head is `Linear -> GELU -> Linear`. For a 4-player game with complex economic dynamics, this is likely underpowered.

**Change**: Expand to 2-3 layers with a residual connection:
```python
self.value_head = nn.Sequential(
    nn.Linear(trunk_dim, head_hidden),
    nn.GELU(),
    nn.Linear(head_hidden, head_hidden),
    nn.GELU(),
    nn.Linear(head_hidden, value_size),
)
```

Or use a dedicated residual block. Small parameter increase, potentially large improvement in value prediction accuracy.

### 5.4 Auxiliary loss: predict legal action count
**File**: `train.py`

At each training position, `len(legal_action_indices)` is known. This is a free supervision signal that forces the trunk to learn representations encoding position complexity.

**Change**: Add a small auxiliary head and loss term:
```python
# In model
self.aux_action_count_head = nn.Linear(shared_trunk_dim, 1)

# In training
predicted_count = model.aux_action_count_head(trunk_features)
aux_loss = F.mse_loss(predicted_count.squeeze(), torch.log(legal_action_count.float()))
total_loss = policy_loss + value_loss_weight * value_loss + 0.1 * aux_loss
```

Requires passing `legal_action_count` through the data loader. Negligible compute cost.

### 5.5 LayerNorm after gated fusion
**File**: `model.py:362-363`

The MLP and GNN branches may have different scale distributions after fusion. Add a `LayerNorm` before the residual blocks to stabilize training:
```python
self.ln_fusion = nn.LayerNorm(shared_trunk_hidden_dim)
# after gated fusion and dropout:
fused_features = self.ln_fusion(fused_features)
```

### 5.6 Attention-based hex scoring for tile-placement policy
**File**: `model.py:42-109`

If the factored head is retained (even with the Phase 3.3 improvements), the hex sub-head computes `P(hex)` from a pooled vector that has lost per-hex identity.

**Change**: For the hex sub-head, compute attention scores directly from per-node GNN embeddings:
```python
hex_scores = self.hex_scorer(node_repr_proj)  # (num_nodes, 1)
hex_log = F.log_softmax(hex_scores.squeeze(), dim=0)
```

This lets the policy reason "hex E14 is good *because of what's on E14*" rather than decoding hex preferences from a global average. The tile and rotation sub-heads can remain factored from the pooled vector.

Requires threading `node_repr_proj` from the GNN branch into the policy head. More involved if batching across positions with different graph sizes (use `batch` index from PyG).

---

## Phase 6: MCTS algorithmic improvements (experimental)

These are lower-confidence improvements that should be validated with A/B testing against the baseline.

### 6.1 Depth-discounted value backup
**File**: `mcts.py:369-380`

The current backup adds the raw network value equally at every ancestor. A value estimate from depth 2 is more reliable than one from depth 20 in a 300+ move game.

**Change**: Apply a mild discount during backup:
```python
def backup_value(self, value, up_to, discount=0.99):
    self.N += 1
    self.W += value
    if self.parent is None or self is up_to:
        return
    self.parent.backup_value(value * discount, up_to, discount)
```

Needs A/B testing — may help or hurt depending on how deep the trees get. Start with `discount=0.995` to be conservative.

### 6.2 Per-round-type exploration constants
**File**: `mcts.py:223-229`

All positions use the same `c_puct_init=1.25`. But auction rounds have ~20 legal moves (benefit from higher exploration), tile placement has 100+ (benefit from more exploitation of the prior).

**Change**: Look up `c_puct_init` from a per-round-type table:
```python
C_PUCT_BY_ROUND = {
    "Auction": 1.5,
    "Stock": 1.25,
    "Operating": 1.0,
}
```

Requires the MCTS node to know the round type (already available via `game_object`). Needs empirical tuning — start with small deltas from the baseline.

### 6.3 Progressive widening
**File**: `mcts.py:271-305`

With 26,535 total actions and many legal per state, MCTS can waste visits on low-probability actions. Progressive widening limits expanded children to `k = c * N^alpha`, focusing search on promising moves.

**Change**: In `select_leaf`, only consider the top-k children by prior probability where k grows with visit count:
```python
k = max(1, int(self.config.pw_c * (self.N ** self.config.pw_alpha)))
top_k_indices = np.argsort(self.child_prior_compressed)[-k:]
# Only compute action scores for top_k_indices
```

Add `pw_c` (e.g., 1.0) and `pw_alpha` (e.g., 0.5) to `SelfPlayConfig`. This is most impactful for positions with 100+ legal moves.

### 6.4 Score prediction instead of win/loss
**File**: `mcts.py:385-399`, `train.py:133-139`

The current value target is `{-1, 0, +1}`. In 1830, winning by $200 vs $2000 requires different strategies. Predicting normalized final scores gives a richer gradient signal.

**Change**: Replace win/loss targets with normalized score fractions:
```python
# In game_result:
scores = np.array([result[pid] for pid in sorted(result.keys())], dtype=np.float32)
value = scores / scores.sum()  # fraction of total wealth

# In training: use KL divergence or MSE against score fractions
```

Requires changing value loss from cross-entropy on `{-1,0,+1}` to KL divergence on score distributions. MCTS backup and Q-value interpretation also need adjustment. More invasive than other changes — validate carefully.

### 6.5 FP16 inference during self-play
**File**: `self_play.py:186-188`

The network runs in FP32 during self-play. Using mixed precision for the forward pass roughly doubles GPU throughput.

**Change**:
```python
with torch.no_grad(), torch.cuda.amp.autocast():
    move_probs, _, values = self.network.run_many_encoded(...)
```

No model changes needed. Negligible quality impact for inference.

### 6.6 Policy entropy bonus during training
**File**: `train.py:127-131`

Early in training, the policy can collapse prematurely to a few actions. An entropy bonus encourages broader exploration:
```python
policy_probs = F.softmax(masked_logits, dim=1)
entropy = -(policy_probs * log_probs).sum(dim=1).mean()
total_loss = policy_loss + value_loss_weight * value_loss - entropy_weight * entropy
```

Add `entropy_weight` (e.g., 0.01) to `TrainingConfig`, decay it over training iterations.

### 6.7 Gradient accumulation for larger effective batch size
**File**: `train.py:73, 114`

Batch size is 256 for a policy space of 26,535. Accumulating gradients over 2-4 mini-batches (effective batch 512-1024) could improve training stability.

**Change**: Standard gradient accumulation pattern:
```python
accumulation_steps = 4
for i, batch in enumerate(train_loader):
    loss = compute_loss(batch) / accumulation_steps
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

Add `gradient_accumulation_steps` to `TrainingConfig`.

---

## Implementation order

```
Phase 1 (bug fixes) ──> Phase 2 (training fix) ──> Phase 4 (throughput) ──> Resume training
     ~1 day                    ~1 day                   ~2 days

Phase 3 + 5 (architecture, bundled) ──> Retrain from scratch
              ~5-7 days

Phase 6 (experimental) ──> A/B test individually
         ongoing
```

- Phase 1 and 2 fix bugs in the existing pipeline.
- Phase 4 improves self-play speed without model changes — do before resuming training to generate more data per iteration.
- Phase 3 and 5 both require retraining from scratch, so bundle all architecture changes together.
- Phase 6 items are experimental and should be validated individually with controlled comparisons against the baseline.
