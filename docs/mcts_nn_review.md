# MCTS & Neural Network Analysis

## MCTS Implementation Issues

### 1. BUG: `pick_move` temperature is effectively a no-op (CRITICAL)

`self_play.py:123-134` — `children_as_pi(squash=True)` applies `probs**0.98` — this is an extremely weak temperature. Standard AlphaZero uses temperature=1.0 early and temperature→0 late. `0.98` exponent is equivalent to temperature ≈ 50 (since `p^(1/T)` with `T=1/0.98≈1.02`), which barely changes the distribution. This likely has almost no exploration effect.

### 2. BUG: `tree_search_duration_this_move` timing is broken (MEDIUM)

`self_play.py:440` — `tree_search_end_time_this_move = time.time() - tree_search_duration_this_move` subtracts 0 (since `tree_search_duration_this_move` is initialized to 0 on line 401 and never updated). This means the metric `SelfPlay/Tree_Search_Time_ms` always reports `current_time * 1000`, not the actual duration.

### 3. BUG: `sim_count_this_move` referenced before assignment (MEDIUM)

`self_play.py:472` — `self.add_metric("SelfPlay/Num_MCTS_Moves", sim_count_this_move)` will raise `UnboundLocalError` when `num_legal_actions == 1` because `sim_count_this_move` is only defined inside the `else` block at line 434.

### 4. Value head output lacks activation (MEDIUM)

`model.py:375-376` — The value head outputs raw logits, but during MCTS backup (`mcts.py:332-367`), the value is used directly without softmax/tanh. In `incorporate_results`, the raw value tensor is backed up through the tree. If the value head produces large logits, this can cause Q-values to drift. The training code treats them as logits (applies `log_softmax` in the loss), but inference uses them raw.

**Fix**: Either apply `torch.softmax` to value output during inference, or apply `torch.tanh` and train with MSE loss like standard AlphaZero.

### 5. `child_Q` division uses `1 + N` instead of `N` (LOW-MEDIUM)

`mcts.py:208` — `child_W / (1 + child_N)` means the Q-value is diluted by one extra phantom visit. Standard MCTS uses `W/N`. The `1+` prevents division by zero but biases Q toward 0 for low-visit nodes. This is a common but debatable choice — it makes unvisited nodes have Q=0 instead of undefined, which can interact badly with the exploration term.

### 6. `N` counts are off by one at root

`mcts.py:227` — `n_s = max(1, self.N - 1)` in `child_U_compressed`. The `-1` compensates for the current visit, but combined with the `1+N` in Q, the exploration/exploitation balance is subtly shifted.

### 7. Memory: game state cloned at every MCTS node

`mcts.py:278` — `pickle_clone()` is called for every new child node. With 200 readouts and branching, this can mean hundreds of full game-state clones per move. Consider:
- **Lazy expansion**: Only clone when a node is actually visited, not when added
- **Action replay from root**: Store the action sequence and replay from root's state when needed (avoids storing N copies)

### 8. No tree reuse across moves

`self_play.py:102-104` — After `play_move`, the old tree is pruned but the new root retains its subtree. However, the root is re-expanded with noise injection on each move (`line 422`). This is correct for AlphaZero, but the N/W statistics from the previous search are carried over, which means `target_readouts_for_move = current_readouts + num_readouts` at line 431 effectively gives the root's subtree a head start. This is standard and correct.

### 9. Recursive tree pruning risks stack overflow

`self_play.py:268-290` — `_recursive_clear_references` is recursive. Deep MCTS trees (hundreds of levels) could hit Python's recursion limit. Use an iterative approach with an explicit stack.

---

## Neural Network Architecture Issues

### 10. Kaiming init uses `nonlinearity="relu"` but network uses GELU (MEDIUM)

`model.py:257` — `nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")` but all activations are GELU. The gain factor for ReLU is `sqrt(2)` while GELU is closer to `sqrt(2/pi) * sqrt(2)`. In practice this may not matter much, but it's inconsistent.

### 11. FactoredPolicyHead assumes independence of hex/tile/rotation (HIGH — architectural)

`model.py:42-109` — The factored head computes `log P(h,t,r) = log P(h) + log P(t) + log P(r)`. This assumes hex, tile, and rotation choices are conditionally independent given the trunk representation. In 1830, they are **strongly correlated** — certain tiles can only go on certain hexes, and rotation matters for connectivity. The legal action mask downstream will zero out impossible combos, but the network must learn to spread probability mass inefficiently because it can't express "tile X on hex Y" preferences without also spreading mass to "tile X on hex Z".

**Improvement options**:
- Add a lightweight cross-attention or bilinear interaction: `log P(h,t,r) = log P(h) + log P(t|h) + log P(r|h,t)` via autoregressive factoring
- Use a smaller MLP that takes `concat(hex_embed, tile_embed, rot_embed)` and outputs a scalar adjustment per (h,t,r) triple — only compute for legal triples

### 12. No GNN residual connections

`model.py:340-348` — The GNN layers don't use residual connections. GAT layers with 4 layers deep can suffer from oversmoothing. Adding skip connections (`node_repr = gat_output + node_repr`) per layer would help, though you'd need a linear projection to match dimensions on the first layer.

### 13. `global_mean_pool` loses spatial information

`model.py:354` — Mean pooling over all hex nodes collapses the spatial structure. For a board game, spatial relationships matter. Consider:
- **Multi-head pooling**: `[mean_pool || max_pool || attention_pool]`
- **Set Transformer / PMA**: Learnable pooling via cross-attention with a small number of seed vectors
- **Keep per-node embeddings**: Feed them into the policy head directly for tile-placement actions (the hex head could attend to specific node embeddings instead of the pooled vector)

### 14. Gated fusion has no layer norm before heads

`model.py:362-363` — After gated fusion and dropout, features go directly into res blocks. A LayerNorm here could stabilize training, especially since the two branches (MLP and GNN) may have different scale distributions.

### 15. Value head is too shallow

`model.py:242-246` — The value head is just `Linear -> GELU -> Linear`. For a 4-player game with complex economic dynamics, a deeper value head (2-3 layers with residual) or separate value trunk could improve value prediction. Standard AlphaZero uses a separate conv layer + FC layers for the value head.

### 16. No auxiliary losses or regularization signals

The training only uses policy cross-entropy + value cross-entropy. Consider:
- **Auxiliary prediction heads**: Predict current game phase, total assets, or round type as auxiliary tasks — these are "free" supervision signals from the encoder
- **Policy entropy bonus**: Prevent premature policy collapse during early training
- **KL divergence from previous iteration**: MCTS-filtered policy improvement, prevents catastrophic forgetting

---

## Training Issues

### 17. Policy loss masking is incorrect (HIGH)

`train.py:128-131`:
```python
masked_log_probs = move_log_probs * legal_action_mask
masked_log_probs = masked_log_probs + 1e-8
policy_loss = -torch.sum(pi * masked_log_probs, dim=1).mean()
```

This multiplies log-probs by a 0/1 mask, which sets illegal log-probs to 0. But `log(softmax)` for illegal actions can be large negative numbers, and zeroing them means the loss ignores them — which is correct conceptually. However, `+ 1e-8` is then added to **all** entries, including illegal ones, making their contribution `pi[illegal] * 1e-8`. Since `pi` for illegal actions should be 0, this is fine in theory, but if `pi` has any floating-point noise in illegal slots, it introduces a small spurious gradient.

**Better approach**: Use `log_softmax` with `-inf` masking before softmax, not after:
```python
masked_logits = policy_logits.masked_fill(legal_action_mask == 0, float('-inf'))
log_probs = F.log_softmax(masked_logits, dim=1)
policy_loss = -torch.sum(pi * log_probs, dim=1).mean()
```

### 18. No learning rate warmup for Adam's momentum (LOW)

`train.py:87` — The warmup starts at `0.01 * lr`. With Adam, the momentum buffers are cold at the start, so warmup should arguably be even more gradual or use RAdam/Lamb which handle this natively.

### 19. No gradient accumulation for large effective batch sizes

`train.py:73` — Batch size is 256. For a policy space of 26,535, this is quite small. Gradient accumulation over 2-4 mini-batches (effective batch 512-1024) would likely improve training stability.

---

## Suggested Architectural Improvements

### A. Transformer-based map encoder instead of GNN

The hex grid is small (93 nodes) — small enough for a Transformer. Benefits:
- Global attention captures long-range route dependencies (GATv2Conv is local)
- Positional encoding via hex coordinates gives spatial awareness
- No oversmoothing risk
- More parameter efficient than 4-layer GAT with 8 heads

### B. Autoregressive policy head for tile placement

Instead of the independence assumption in FactoredPolicyHead:
```
P(action) = P(hex) * P(tile | hex, trunk) * P(rotation | hex, tile, trunk)
```
This can be done with a small LSTM/Transformer decoder that autoregressively samples hex -> tile -> rotation. Only needed for LayTile actions; other actions use the flat head.

### C. Value head should predict expected scores, not just win/loss

The current value target is {-1, 0, +1}. In 1830, the margin matters — winning by $200 vs $2000 requires different strategies. Predicting normalized final scores would give the value head a richer signal.

### D. Progressive widening for MCTS

With 26,535 actions and many legal at each state, MCTS wastes visits on low-probability actions. Progressive widening limits the number of children expanded based on visit count: only expand the top-k actions where k grows as `c * N^alpha`. This focuses search on promising moves.

### E. Use mixed precision (FP16) for inference

The network runs in FP32 during self-play. Using `torch.cuda.amp.autocast()` for the forward pass in `tree_search` would roughly double throughput on GPU, with negligible quality loss.
