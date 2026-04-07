# MCTS Performance Optimization Plan

## Current Architecture

Each MCTS move does `num_readouts=200` simulations. Each simulation:
1. **select_leaf**: Walk tree from root → leaf (fast, O(depth) numpy ops)
2. **maybe_add_child**: Clone game + process action + create new MCTSNode
3. **MCTSNode.__init__**: Encode game state + get legal action indices
4. **incorporate_results**: Neural net forward pass on encoded state
5. **backup_value**: Walk back up tree (fast, O(depth) numpy ops)

Per game (~700 actions), total MCTS work ≈ 700 × 200 = 140,000 simulations.

## Profiling Targets

### Phase 1: Measure the Bottlenecks

**Goal**: Instrument the MCTS hot path to know exactly where time goes.

**Method**: Add `time.perf_counter()` around each component in `MCTSNode.__init__` and `maybe_add_child`, run a few MCTS moves, aggregate.

Components to measure:
- `pickle_clone()` — game state deep copy
- `process_action()` — apply one action
- `map_index_to_action()` — convert index to Action object (Python ActionHelper)
- `encoder.encode()` — convert game state to graph tensors
- `get_legal_action_indices()` — calls ActionHelper → enumerates all legal actions
- Neural net forward pass (already separated in `incorporate_results`)

**Expected findings** (based on bench_engines.py):
- Rust process_action: ~0.005ms (217K/s)
- Rust clone: ~0.035ms per clone (28.9K/s, 7× the action cost)
- Encoder: likely dominant — creates torch tensors, builds graph structure
- ActionHelper: significant — full Python action enumeration per node

### Phase 2: Optimize Clone (Rust-side)

**Current cost**: Each `clone_for_search()` deep-copies ~15 fields. The expensive ones:

| Field | Type | Est. Size | Clone Cost |
|-------|------|-----------|------------|
| `hexes` | `Vec<Hex>` (≈70 hexes, each with Tile, Cities, Paths) | ~50KB | HIGH |
| `corporations` | `Vec<Corporation>` (8 corps, 10 shares each) | ~5KB | MEDIUM |
| `stock_market` | `StockMarket` (grid of Option<SharePrice>) | ~2KB | LOW |
| `market_cell_corps` | `HashMap<(u8,u8), Vec<String>>` | ~1KB | LOW |
| `tile_counts_remaining` | `HashMap<String, u32>` | ~1KB | LOW |
| `corp_idx/company_idx/hex_idx` | `HashMap<String, usize>` | ~2KB | LOW |
| `hex_adjacency` | `Arc<...>` (shared) | 8 bytes | FREE |
| `tile_catalog` | `Arc<...>` (shared) | 8 bytes | FREE |

**Optimization ideas**:

1. **Arc-share lookup indices**: `corp_idx`, `company_idx`, `hex_idx` never change after game init. Wrap in `Arc` like `hex_adjacency`.
   - Expected impact: ~10% clone reduction

2. **COW (Copy-on-Write) hexes**: Most hexes don't change between clones (tiles are only upgraded occasionally). Use `Arc<Vec<Hex>>` with COW semantics — share until a tile is placed, then clone just the modified hex.
   - Expected impact: 30-50% clone reduction (hexes are the biggest field)
   - Complexity: HIGH — need to track which hexes changed

3. **Smaller Hex representation**: Strip non-essential data from Hex for search clones. Paths, edges, upgrades are only needed for graph computation which is cached.
   - Alternative: lazy-rebuild paths/edges from tile_catalog when needed

4. **Pool-based allocation**: Pre-allocate a pool of BaseGame objects, reuse instead of allocating fresh.
   - Expected impact: reduces allocator pressure, ~10-20% improvement

### Phase 3: Reduce Python Round-Trips (ActionHelper)

**Current cost**: `get_legal_action_indices()` calls Python's `ActionHelper.get_all_choices_limited()` which:
- Calls `game.round.actions_for(entity)` → goes through Rust adapter
- For each action type, enumerates all specific actions (tiles, shares, trains, bids...)
- Converts each to an Action object
- Maps each to an action index

This is the most expensive per-node operation because it's 100% Python.

**Optimization ideas**:

5. **Rust-native action enumeration**: Move ActionHelper logic into Rust. Return `Vec<u32>` of legal action indices directly. Eliminates the Python ActionHelper entirely.
   - Expected impact: 10-50× speedup for action generation
   - Complexity: HIGH — need to port all action enumeration logic
   - Priority: HIGHEST for overall throughput

6. **Cache legal actions for unchanged state**: If a node's parent has the same legal actions (only one child was expanded), reuse the parent's legal action list.
   - Expected impact: LOW — most nodes have different states

7. **Incremental action mask**: Instead of enumerating all actions, maintain a bitmask and update it incrementally when an action is taken.
   - Expected impact: HIGH if action enumeration is the bottleneck
   - Complexity: VERY HIGH — need delta-computation for every action type

### Phase 4: Optimize Encoder

**Current cost**: `encoder.encode()` builds graph tensors for PyTorch Geometric:
- Node features: 50 features × ~70 nodes
- Edge index: adjacency pairs
- Game state vector: 390 floats

This involves Python loops over hexes, corporations, companies, and creates numpy arrays then converts to torch tensors.

**Optimization ideas**:

8. **Rust-native encoder**: Compute all feature vectors in Rust, return flat arrays via PyO3. Python just wraps in torch tensors.
   - Expected impact: 5-20× encoder speedup
   - Complexity: MEDIUM — encoder logic is straightforward math

9. **Incremental encoding**: Track what changed since parent node's encoding. Only recompute affected features.
   - Expected impact: HIGH for deep tree traversals
   - Complexity: HIGH — need change tracking

10. **Batch encoding**: Collect multiple leaf nodes, encode all at once. Reduces per-call overhead.
    - Expected impact: MEDIUM (already done for GPU forward pass, extend to encoding)

### Phase 5: Neural Network Forward Pass

11. **Batch inference**: Collect `parallel_readouts=8` leaves, batch the forward pass.
    - Already implemented in the current MCTS code
    - Verify GPU utilization with torch profiler

12. **ONNX/TensorRT export**: Convert the GNN model to ONNX for optimized inference.
    - Expected impact: 2-5× inference speedup
    - Complexity: MEDIUM (GATv2Conv may need custom export)

13. **Smaller model for self-play**: Use a distilled/smaller model during self-play, full model for training.
    - Expected impact: proportional to model size reduction

## Recommended Priority Order

| Priority | Optimization | Impact | Effort |
|----------|-------------|--------|--------|
| 1 | **Profile the hot path** (Phase 1) | Prerequisite | LOW |
| 2 | **Rust-native action enum** (Phase 3, #5) | 10-50× action gen | HIGH |
| 3 | **Rust-native encoder** (Phase 4, #8) | 5-20× encoding | MEDIUM |
| 4 | **Arc-share lookup indices** (Phase 2, #1) | 10% clone | LOW |
| 5 | **COW hexes** (Phase 2, #2) | 30-50% clone | HIGH |
| 6 | **Batch inference** verification (Phase 5, #11) | Verify existing | LOW |
| 7 | **ONNX export** (Phase 5, #12) | 2-5× inference | MEDIUM |

## Quick Wins (implementable today)

- **Arc-share corp_idx/company_idx/hex_idx** — trivial change
- **Add profiling hooks** to MCTS — measure before optimizing
- **Reduce recent_actions clone** — cap at last 5 instead of unbounded
- **Skip graph_cache clone** — already creating fresh GraphCache::new()
- **String interning** — corp_idx/hex_idx keys are repeated strings; use u32 IDs internally
