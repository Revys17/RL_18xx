# Router Rewrite Plan

## Problem

The current Rust router uses a node-level DFS that generates invalid routes. It visits revenue nodes (cities/towns/offboards) as abstract graph nodes, but doesn't track the physical hex-level path the train takes. This causes:

1. **Non-linear routes** — The DFS visit order doesn't form a connected linear track path. Two nodes in the path may require traversing through a shared hex in incompatible directions.
2. **Invalid hexside conflict detection** — Hexside bits are assigned during neighbor tracing, not from the actual route path. Combined routes may claim non-conflicting hexsides that physically overlap.
3. **Bypass path confusion** — When a tile has both a city path and a bypass (edge↔edge) path (like H12 in 1830), the DFS treats both as valid stops. But the bypass means the train passes through WITHOUT stopping. A blocked city with a bypass must be bypassed, not stopped at.

**Result**: 0/50 "Rust higher" routes validated by Python's engine. The 27% "Rust higher" cases are all invalid.

## Root Cause

Python's autorouter walks at the **Path level** (tile path segments), not the node level. Each walk step is a physical tile path (`Path` object) connecting two endpoints (edge, city, town, junction). The walk accumulates a `visited_paths` dict that IS the route — an ordered sequence of physical track segments. This guarantees the route is a connected linear path by construction.

Rust's router walks at the **Node level**. `reachable_neighbors()` traces from a node through multiple hexes to find the next node, collapsing the hex-level path into a single hop. The DFS then explores a tree of node-to-node hops. The resulting "path" is a DFS visit sequence, not a physical track path.

## Architecture: Path-Level Walk (matching Python)

### Core Data Structure

Replace node-level DFS with a path-level walk that mirrors Python's `Node.walk()` → `Path.walk()` chain.

```rust
/// A single tile path segment — the atomic unit of route traversal.
/// Corresponds to one PathDef on a tile (e.g., Edge(0)↔City(0), Edge(1)↔Edge(4)).
#[derive(Clone, Hash, Eq, PartialEq)]
struct PathSegment {
    hex_id: String,
    path_index: usize,  // index into tile.paths
}

/// Accumulated state during a walk — the equivalent of Python's `vp` dict.
struct WalkState {
    /// Visited path segments (dedup + backtrack guard)
    visited_paths: HashSet<PathSegment>,
    /// Visited nodes (city/town blocking guard)  
    visited_nodes: HashSet<NodeId>,
    /// Edge traversal counter (prevents reusing same hexside)
    edge_counter: HashMap<(String, u8), u32>,
    /// Accumulated hexside bits for conflict detection
    hexside_bits: u128,
}
```

### Walk Algorithm

```
walk_from_node(hex_id, node, state) → yields WalkState snapshots
    mark node as visited
    for each path on this node's tile that connects to this node:
        if path already in visited_paths: skip
        if path's edge already traversed (edge_counter > 0): skip
        
        add path to visited_paths
        yield current state (this is a valid partial route)
        
        match path's other endpoint:
            Edge(e) →
                increment edge_counter for this edge
                find neighbor hex at edge e
                find matching path in neighbor hex entering from opposite edge
                walk_into_path(neighbor_hex, matching_path, state)
                decrement edge_counter
            
            City(i) / Town(i) →
                if not blocked:
                    walk_from_node(same_hex, node_i, state)  // intra-tile hop
            
            Junction →
                for each other junction path on same tile:
                    walk_into_path(same_hex, junction_path, state)
        
        remove path from visited_paths (backtrack)
    
    remove node from visited_nodes (backtrack / converging)
```

```
walk_into_path(hex_id, path, state) →
    if path already in visited_paths: return
    if path's edge already traversed: return
    
    add path to visited_paths
    yield current state
    
    match path's other endpoint:
        Edge(e) →
            // pass-through hex: enter one edge, exit another
            increment edge_counter
            find neighbor, find matching path
            walk_into_path(neighbor_hex, matching_path, state)
            decrement edge_counter
        
        City(i) / Town(i) / Offboard(i) →
            // reached a revenue node
            walk_from_node(hex_id, node_i, state)
        
        Junction → ...
    
    remove path from visited_paths (backtrack)
```

### Route Construction from Walk State

After each `yield`, the accumulated `visited_paths` in `WalkState` contains the full set of path segments traversed. To build a route candidate:

1. **Extract stops**: Collect all nodes that appear in the visited paths. These are the revenue stops.
2. **Build connection hex chains**: Group consecutive paths into chains between stops. Each chain is a list of hex IDs from one stop to the next.
3. **Compute revenue**: Sum `node_revenue()` for each stop, using phase-appropriate values.
4. **Compute hexside bits**: OR all edge bits from the walk.
5. **Filter**: Route must have ≥2 stops and include at least one tokened city.

### Combination Optimizer

Keep the existing `find_best_routes()` approach: for each train, collect all valid route candidates, then find the revenue-maximizing combination with no hexside conflicts (bitfield AND check).

The bitfield conflict detection is correct in principle — the issue is that the current implementation assigns bits during `trace_to_node` which doesn't correspond to the actual route path. With path-level walks, the bits are accumulated from the actual path segments traversed, so conflicts are accurate.

### What to Keep

- `RouteFinder` struct and `assign_hexside_pair()` — hexside bit assignment is fine
- `find_best_routes()` and `find_best_recursive()` — combination optimizer is correct  
- `RouteCandidate` struct (add `connections: Vec<Vec<String>>` field for hex chains)
- `node_revenue()` with phase-dependent offboard revenue
- `is_city_blocked_for_route()` — used during walk to skip blocked nodes
- `has_bypass_path()` — used to decide blocked+bypass behavior
- All unit tests (update to use new API)

### What to Replace

- `reachable_neighbors()` — replaced by path-level walk
- `trace_to_node()` / `trace_to_node_inner()` — replaced by path-level walk
- `enumerate_routes()` — rewritten to use walk + candidate extraction
- `dfs_routes()` — replaced by walk-based enumeration
- `DfsState` — replaced by `WalkState`

## Connection Hex Chain Output

The rewritten router should output connection hex chains for each route, matching Python's `connection_hexes` format:

```
Route {
    nodes: [NodeId],           // revenue stops visited
    revenue: i32,
    hexside_bits: u128,
    connections: Vec<Vec<String>>,  // hex chains between consecutive stops
}
```

Each connection is `[start_hex, intermediate_hex_1, ..., end_hex]`. This enables:
1. Direct construction of Python `Route` objects for validation
2. Construction of `RunRoutes` action dicts
3. Verification that routes are physically valid

## Validation Strategy

After rewriting, validate by:

1. **Unit tests**: Existing tests should pass with updated API
2. **Route construction**: For each Rust route, construct a Python `Route` object via `connection_hexes` and call `route.revenue()`. It should not throw.
3. **Revenue matching**: Compare Rust revenue with Python Route revenue. They should match exactly (same stops, same phase-appropriate revenue calculation).
4. **Cross-engine comparison**: Run the 100-seed comparison test. "Rust higher" routes that Python's Route constructor rejects → 0. "Py higher" cases → investigate individually.

## Effort Estimate

- Walk algorithm implementation: ~200 lines replacing ~150 lines
- Route candidate extraction (stops, hex chains, revenue): ~80 lines  
- Integration with `calculate_corp_routes` and `calculate_routes` PyO3 method: ~30 lines
- Update unit tests: ~50 lines
- Validation test script: ~50 lines

Total: ~400 lines changed, ~2-3 hours of focused work.

## Key Files

- `engine-rs/src/router.rs` — main rewrite target
- `engine-rs/src/game.rs` — `calculate_routes()` method (minor updates)
- `engine-rs/src/map.rs` — reference for walk algorithm (graph connectivity walk)
- `engine-rs/src/tiles.rs` — `PathDef`, `PathEndpoint` types (already correct)

## Reference: Python's Walk Architecture

The Python autorouter works as follows:

1. **Starting points**: All connected revenue nodes, sorted by priority (tokened > offboard > high revenue)

2. **Walk**: `node.walk()` recursively yields `(path, visited_paths_dict)` at each path step. The `visited_paths` dict accumulates ALL path segments traversed from the starting node to the current position. Backtracking removes paths (if `converging=True`), enabling alternative route branches.

3. **Chain assembly**: `process_path(vp)` takes the visited_paths dict and groups the paths into chains: `[{nodes: [left, right], paths: [Path, ...]}]`. Each chain runs between two revenue nodes.

4. **Route validation**: `Route(connection_data=chains)` validates: ≥2 stops, has token, connected (pairwise `connects_to()`), within distance, no cycles, no terminal in interior.

5. **Combination**: `js_evaluate_combos()` tries all combinations (one route per train), using bitfield AND for fast hexside conflict check, then `final_revenue_check()` re-validates the best combo with full overlap checking.

Key insight: Python's walk guarantees routes are **physically connected linear paths** because each step extends from the current position via an actual tile path. The DFS visit order IS the route path. There is no "reachable neighbor" abstraction — every hop is a concrete path segment on a tile.
