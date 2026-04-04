# Phase 4: Map Graph, Tile System, Token Placement, Connectivity, and Auto-Routing

## Context

Phase 3 (action processing) is blocked because Operating Round steps need graph connectivity information to function:
- `Track` step calls `can_lay_tile()` which needs `graph.connected_hexes(corp)` to determine reachable hexes
- `Token` step calls `can_place_token()` which needs `graph.tokenable_cities(corp)` for valid cities
- `Route` step needs the autorouter to calculate train revenues
- Many steps auto-skip when the entity has no legal actions, which requires connectivity queries

Phase 4 implements the full tile/graph/routing system so Phase 3's Operating Round can work.

## What This Phase Delivers

1. **Tile definition parsing** — load all 46 tile types from DSL strings, with paths/edges/cities/towns
2. **Tile placement** — `hex.lay(tile)` with token transfer, path validation, rotation checking
3. **Graph connectivity** — `Graph.compute(corp)` → reachable hexes/nodes/paths from token positions
4. **Token placement** — validate and place tokens on reachable cities
5. **Route calculation** — autorouter finds optimal revenue for multi-train corporations
6. **Legal move queries** — `can_lay_tile(entity)`, `upgradeable_tiles(entity, hex)`, `can_place_token(entity)`

---

## Architecture

### New Files

```
engine-rs/src/
├── tiles.rs          # Tile DSL parser, tile catalog, path/edge connectivity
├── map.rs            # Graph connectivity engine (compute, walk, reachability)
├── router.rs         # Auto-router for revenue calculation
```

### Modified Files

```
├── graph.rs          # Add Path struct, update Tile to reference parsed paths
├── game.rs           # Wire up graph, add tile/token processing methods
├── entities.rs       # Token placement on City
├── title/g1830.rs    # Add full tile catalog DSL strings, upgrade rules
```

---

## Step 1: Tile DSL Parser (`src/tiles.rs`)

### The DSL Format

Each tile is defined by a semicolon-separated string of parts:
```
"city=revenue:30,slots:2;path=a:0,b:_0;path=a:3,b:_0;label=B"
```

Parts:
- `city=revenue:V[,slots:N][,loc:P]` — city with revenue, token slots
- `town=revenue:V[,loc:P]` — town with revenue
- `offboard=revenue:yellow_V|brown_V[,groups:G]` — red hex offboard
- `path=a:REF,b:REF[,terminal:true]` — connection between two endpoints
  - REF is either an edge number `0-5` or a node reference `_N` (Nth city/town)
- `upgrade=cost:N,terrain:T` — terrain cost
- `label=TEXT` — display label (B, OO, NY)
- `border=edge:N,type:TYPE` — hex border restriction
- `junction` — hub allowing path branching
- `stub=edge:N` — dead-end stub

### Data Structures

```rust
/// A parsed tile definition with all connectivity information.
#[derive(Clone, Debug)]
pub struct TileDef {
    pub name: String,
    pub color: TileColor,
    pub paths: Vec<PathDef>,
    pub cities: Vec<CityDef>,
    pub towns: Vec<TownDef>,
    pub offboards: Vec<OffboardDef>,
    pub edges: Vec<u8>,           // unique edge numbers used by paths
    pub upgrades: Vec<UpgradeDef>,
    pub label: Option<String>,
    pub has_junction: bool,
}

#[derive(Clone, Debug)]
pub struct PathDef {
    pub a: PathEndpoint,
    pub b: PathEndpoint,
    pub terminal: bool,
}

#[derive(Clone, Debug)]
pub enum PathEndpoint {
    Edge(u8),           // hexside 0-5
    City(usize),        // index into tile's cities
    Town(usize),        // index into tile's towns
    Offboard(usize),    // index into tile's offboards
    Junction,           // connects to junction node
}

#[derive(Clone, Debug)]
pub struct CityDef {
    pub revenue: i32,
    pub slots: u8,
}

#[derive(Clone, Debug)]
pub struct TownDef {
    pub revenue: i32,
}

pub enum TileColor { Yellow, Green, Brown, Gray, Red, White }
```

### Parser Implementation

```rust
pub fn parse_tile(name: &str, code: &str, color: TileColor) -> TileDef {
    let parts: Vec<&str> = code.split(';').collect();
    let mut cities = Vec::new();
    let mut towns = Vec::new();
    let mut paths = Vec::new();
    // ... parse each part ...
    
    // Extract unique edges from paths
    let edges: Vec<u8> = paths.iter()
        .flat_map(|p| [&p.a, &p.b])
        .filter_map(|ep| match ep { PathEndpoint::Edge(n) => Some(*n), _ => None })
        .collect::<HashSet<_>>().into_iter().collect();
    
    TileDef { name, color, paths, cities, towns, edges, ... }
}
```

### Rotation

Rotation transforms edge numbers: `rotated_edge = (edge + rotation) % 6`

```rust
impl TileDef {
    /// Return a rotated copy of this tile definition.
    pub fn rotated(&self, rotation: u8) -> TileDef {
        let rotated_paths = self.paths.iter().map(|p| PathDef {
            a: p.a.rotated(rotation),
            b: p.b.rotated(rotation),
            terminal: p.terminal,
        }).collect();
        // edges also rotated
        ...
    }
}

impl PathEndpoint {
    fn rotated(&self, rotation: u8) -> PathEndpoint {
        match self {
            PathEndpoint::Edge(n) => PathEndpoint::Edge((*n + rotation) % 6),
            other => other.clone(),  // cities/towns don't rotate
        }
    }
}
```

### Tile Catalog

Build from the Python `TileConfig` catalog. For 1830, we need 46 tile types. Store as a static catalog:

```rust
/// Returns all tile definitions keyed by tile ID.
pub fn tile_catalog() -> HashMap<String, TileDef> {
    let mut catalog = HashMap::new();
    // Yellow tiles
    catalog.insert("1".into(), parse_tile("1", "town=revenue:10;path=a:0,b:_0;path=a:_0,b:4", Yellow));
    catalog.insert("2".into(), parse_tile("2", "town=revenue:10;path=a:0,b:_0;path=a:_0,b:3", Yellow));
    // ... all 46 tiles ...
    catalog
}
```

The DSL strings for all tiles must be extracted from `graph.py` lines 2134-2664 (`TileConfig` class).

---

## Step 2: Tile Placement on Hexes

### Upgrade Validation

A new tile can be placed on a hex if:
1. The hex's current tile color allows upgrade to the new tile's color (yellow→green→brown→gray)
2. All existing paths are preserved as a subset of the new tile's paths
3. City/town counts and positions map correctly
4. At least one rotation connects to the corporation's network

```rust
impl BaseGame {
    /// Check if `new_tile` can legally replace the tile on `hex` at `rotation`.
    pub fn legal_tile_placement(
        &self,
        hex_id: &str,
        new_tile_id: &str,
        rotation: u8,
        corp_sym: &str,
    ) -> bool {
        let hex = &self.hexes[self.hex_idx[hex_id]];
        let old_tile = &hex.tile;
        let new_tile = self.tile_catalog.get(new_tile_id)
            .expect("unknown tile").rotated(rotation);
        
        // 1. Color upgrade check
        if !is_valid_upgrade_color(old_tile.color, new_tile.color) { return false; }
        
        // 2. Path subset check: all old paths must exist in new tile
        if !old_paths_subset_of_new(&old_tile, &new_tile) { return false; }
        
        // 3. At least one new exit connects to corp's network
        let connected = self.graph_connected_hexes(corp_sym);
        let hex_edges = connected.get(hex_id);
        if let Some(edges) = hex_edges {
            new_tile.edges.iter().any(|e| edges.contains(e))
        } else {
            false // hex not reachable
        }
    }
}
```

### Executing Tile Placement

```rust
pub fn lay_tile(&mut self, hex_id: &str, tile_id: &str, rotation: u8) {
    let hex = &mut self.hexes[self.hex_idx[hex_id]];
    let new_tile_def = self.tile_catalog.get(tile_id).unwrap().rotated(rotation);
    
    // Map old cities → new cities (transfer tokens)
    let city_map = compute_city_map(&hex.tile, &new_tile_def);
    
    // Build new tile with tokens transferred
    let mut new_tile = Tile::from_def(&new_tile_def);
    for (old_idx, new_idx) in &city_map {
        // Transfer tokens from old city to new city
        let old_tokens = hex.tile.cities[*old_idx].tokens.clone();
        new_tile.cities[*new_idx].tokens = old_tokens;
    }
    
    hex.tile = new_tile;
    
    // Clear graph caches (connectivity changed)
    self.clear_graph_cache();
}
```

---

## Step 3: Graph Connectivity Engine (`src/map.rs`)

This is the most algorithmically complex component. It determines which hexes, nodes, and paths a corporation can reach from its placed tokens.

### Algorithm: `compute(corp)`

```
Starting nodes = all cities where corp has tokens
For each starting node:
    BFS/DFS walk outward through paths:
        For each path from current node:
            Find the exit edge of the path
            Look up the neighbor hex at that edge
            Find the matching path in neighbor hex (entering from opposite edge)
            If the path leads to a node:
                If node is not blocked (no other corp's token in a city with all slots full):
                    Add node to reachable set
                    Continue walk from that node
            If the path leads to another edge:
                Continue traversal through hex
Collect all reachable hexes, nodes, paths
```

### Data Structure

```rust
pub struct GraphCache {
    /// For each corporation: which hexes are reachable and via which edges.
    connected_hexes: HashMap<String, HashMap<String, HashSet<u8>>>,  // corp -> hex_id -> {edges}
    /// For each corporation: which nodes (cities/towns) are reachable.
    connected_nodes: HashMap<String, HashSet<NodeId>>,
    /// For each corporation: which cities can accept a new token.
    tokenable_cities: HashMap<String, Vec<TokenableCity>>,
    /// Dirty flag — cleared when tiles change.
    dirty: bool,
}

#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub struct NodeId {
    pub hex_id: String,
    pub node_type: NodeType,  // City(index) or Town(index)
    pub index: usize,
}

pub struct TokenableCity {
    pub hex_id: String,
    pub city_index: usize,
}
```

### Walk Implementation

The key algorithm — recursive path traversal from a node through connected hexes:

```rust
fn walk_from_node(
    game: &BaseGame,
    start_hex: &str,
    start_node: &NodeId,
    visited_nodes: &mut HashSet<NodeId>,
    visited_hexes: &mut HashMap<String, HashSet<u8>>,
    corp_sym: &str,
) {
    if visited_nodes.contains(start_node) { return; }
    visited_nodes.insert(start_node.clone());
    
    let hex = &game.hexes[game.hex_idx[start_hex]];
    let tile = &hex.tile;  // need parsed TileDef with paths
    
    // Find all paths that connect to this node
    for path in &tile.paths {
        let exit_edge = match path_exit_from_node(path, start_node) {
            Some(edge) => edge,
            None => continue,  // path doesn't connect to this node
        };
        
        // Follow exit to neighbor hex
        let neighbor_id = match game.hex_adjacency.get(start_hex)
            .and_then(|n| n.get(&exit_edge)) {
            Some(id) => id,
            None => continue,  // edge leads off-map
        };
        
        visited_hexes.entry(start_hex.to_string())
            .or_default().insert(exit_edge);
        
        let enter_edge = (exit_edge + 3) % 6;  // opposite direction
        
        // Find matching path in neighbor hex entering from opposite edge
        walk_into_hex(game, neighbor_id, enter_edge, visited_nodes, visited_hexes, corp_sym);
    }
}

fn walk_into_hex(
    game: &BaseGame,
    hex_id: &str,
    enter_edge: u8,
    visited_nodes: &mut HashSet<NodeId>,
    visited_hexes: &mut HashMap<String, HashSet<u8>>,
    corp_sym: &str,
) {
    let hex = &game.hexes[game.hex_idx[hex_id]];
    let tile = &hex.tile;
    
    visited_hexes.entry(hex_id.to_string())
        .or_default().insert(enter_edge);
    
    // Find paths that enter this hex from enter_edge
    for path in &tile.paths {
        if !path_enters_from(path, enter_edge) { continue; }
        
        // Where does this path lead?
        let dest = path_destination_from(path, enter_edge);
        match dest {
            PathEndpoint::Edge(exit_edge) => {
                // Path passes through hex — continue to next neighbor
                let neighbor_id = match game.hex_adjacency.get(hex_id)
                    .and_then(|n| n.get(&exit_edge)) {
                    Some(id) => id,
                    None => continue,
                };
                visited_hexes.entry(hex_id.to_string())
                    .or_default().insert(exit_edge);
                let next_enter = (exit_edge + 3) % 6;
                walk_into_hex(game, neighbor_id, next_enter, visited_nodes, visited_hexes, corp_sym);
            }
            PathEndpoint::City(idx) | PathEndpoint::Town(idx) => {
                // Path leads to a node
                let node = NodeId { hex_id: hex_id.to_string(), node_type: dest.node_type(), index: idx };
                
                // Check blocking: is this city fully tokened by other corps?
                if is_blocked(&tile, idx, corp_sym) { continue; }
                
                // Recurse from this node
                walk_from_node(game, hex_id, &node, visited_nodes, visited_hexes, corp_sym);
            }
            _ => {}
        }
    }
}
```

### Blocking Rules

A city blocks traversal if:
- All token slots are filled AND
- None of the tokens belong to the traversing corporation

```rust
fn is_blocked(tile: &Tile, city_index: usize, corp_sym: &str) -> bool {
    let city = &tile.cities[city_index];
    if city.tokens.iter().any(|t| t.is_none()) {
        return false;  // empty slot = not blocked
    }
    // All slots full — blocked unless corp has a token here
    !city.tokens.iter().any(|t| {
        t.as_ref().map_or(false, |tok| tok.corporation_id == corp_sym)
    })
}
```

---

## Step 4: Token Placement

### Legal Token Placement

```rust
impl BaseGame {
    pub fn can_place_token(&self, corp_sym: &str) -> bool {
        let corp = &self.corporations[self.corp_idx[corp_sym]];
        // Must have unplaced tokens
        if corp.next_token_index().is_none() { return false; }
        // Must be able to afford cheapest token
        let price = corp.next_token_price().unwrap();
        if corp.cash < price { return false; }
        // Must have at least one tokenable city reachable
        !self.tokenable_cities(corp_sym).is_empty()
    }
    
    pub fn tokenable_cities(&self, corp_sym: &str) -> Vec<TokenableCity> {
        // Compute graph if needed, then filter for cities with open slots
        // that the corp can reach and doesn't already have a token in (same hex)
        ...
    }
    
    pub fn place_token(&mut self, corp_sym: &str, hex_id: &str, city_index: usize) {
        let corp_idx = self.corp_idx[corp_sym];
        let token_idx = self.corporations[corp_idx].next_token_index().unwrap();
        let price = self.corporations[corp_idx].tokens[token_idx].price;
        
        // Pay for token
        self.corporations[corp_idx].cash -= price;
        self.corporations[corp_idx].tokens[token_idx].used = true;
        self.corporations[corp_idx].tokens[token_idx].city_hex_id = hex_id.to_string();
        
        // Place in city
        let hex_i = self.hex_idx[hex_id];
        let city = &mut self.hexes[hex_i].tile.cities[city_index];
        let empty_slot = city.tokens.iter().position(|t| t.is_none()).unwrap();
        city.tokens[empty_slot] = Some(Token::new(corp_sym.to_string(), price));
        
        // Clear graph cache
        self.clear_graph_cache();
    }
}
```

---

## Step 5: Auto-Router (`src/router.rs`)

### Algorithm Overview

The auto-router finds the revenue-maximizing combination of routes for all trains:

```
1. For each train:
   a. From each tokened city: enumerate all possible routes (DFS through connected paths)
   b. Each route is a sequence of nodes (cities/towns) visited
   c. Filter routes by: min 2 stops, no city visited twice, distance ≤ train distance
   d. Calculate revenue for each valid route

2. Find the combination of routes (one per train) that:
   a. Maximizes total revenue
   b. Has no overlapping track segments (hexside conflicts)
```

### Hexside Conflict Detection (Bitfield)

Each hexside gets a unique bit position. A route's "bitfield" is the OR of all hexside bits it uses. Two routes conflict if their bitfields AND to non-zero.

```rust
pub struct RouteFinder {
    hexside_bits: HashMap<(String, u8), u64>,  // (hex_id, edge) -> bit position
    next_bit: u64,
}

impl RouteFinder {
    fn assign_bit(&mut self, hex_id: &str, edge: u8) -> u64 {
        let key = (hex_id.to_string(), edge);
        *self.hexside_bits.entry(key).or_insert_with(|| {
            let bit = 1u64 << self.next_bit;
            self.next_bit += 1;
            bit
        })
    }
    
    fn routes_conflict(a: u64, b: u64) -> bool {
        a & b != 0
    }
}
```

### Route Enumeration (DFS)

```rust
fn enumerate_routes(
    game: &BaseGame,
    corp_sym: &str,
    train_distance: u32,
) -> Vec<RouteCandidate> {
    let mut candidates = Vec::new();
    let graph = game.compute_graph(corp_sym);
    
    // Start from each tokened city
    for start_node in &graph.token_nodes {
        let mut visited = HashSet::new();
        visited.insert(start_node.clone());
        
        dfs_routes(
            game, start_node, &mut visited,
            &mut vec![start_node.clone()],
            0,  // distance so far
            train_distance,
            &mut candidates,
        );
    }
    candidates
}

fn dfs_routes(
    game: &BaseGame,
    current: &NodeId,
    visited: &mut HashSet<NodeId>,
    path: &mut Vec<NodeId>,
    distance: u32,
    max_distance: u32,
    candidates: &mut Vec<RouteCandidate>,
) {
    // Record current path as candidate if 2+ stops
    if path.len() >= 2 {
        let revenue = calculate_revenue(game, path);
        candidates.push(RouteCandidate { nodes: path.clone(), revenue, ... });
    }
    
    if distance >= max_distance { return; }
    
    // Explore neighbors
    for neighbor in reachable_nodes_from(game, current) {
        if visited.contains(&neighbor) { continue; }
        visited.insert(neighbor.clone());
        path.push(neighbor.clone());
        dfs_routes(game, &neighbor, visited, path, distance + 1, max_distance, candidates);
        path.pop();
        visited.remove(&neighbor);
    }
}
```

### Optimal Combination Selection

```rust
fn find_best_routes(
    trains: &[Train],
    candidates_per_train: &[Vec<RouteCandidate>],
) -> Vec<RouteCandidate> {
    // For small numbers of trains (typically 1-3), brute-force all combos
    let mut best_revenue = 0;
    let mut best_combo = Vec::new();
    
    for combo in all_combinations(candidates_per_train) {
        // Check no hexside conflicts
        let mut combined_bits = 0u64;
        let mut conflict = false;
        for route in &combo {
            if combined_bits & route.bitfield != 0 {
                conflict = true;
                break;
            }
            combined_bits |= route.bitfield;
        }
        if conflict { continue; }
        
        let total = combo.iter().map(|r| r.revenue).sum::<i32>();
        if total > best_revenue {
            best_revenue = total;
            best_combo = combo;
        }
    }
    best_combo
}
```

---

## Step 6: Wire Everything into BaseGame

### Tile Catalog as Shared Data

```rust
pub struct BaseGame {
    // ... existing fields ...
    
    // NEW: parsed tile catalog (shared via Arc for cloning)
    pub(crate) tile_catalog: Arc<HashMap<String, TileDef>>,
    
    // NEW: graph connectivity cache (rebuilt after tile/token changes)
    pub(crate) graph_cache: Option<GraphCache>,
}
```

### Key Methods to Add

```rust
#[pymethods]
impl BaseGame {
    /// Get reachable hexes for a corporation (for encoder and legal move generation).
    fn connected_hexes(&self, corp_sym: &str) -> HashMap<String, Vec<u8>> { ... }
    
    /// Get cities where a corporation can place tokens.
    fn tokenable_cities_for(&self, corp_sym: &str) -> Vec<(String, usize)> { ... }
    
    /// Get legal tile placements for a corporation on a hex.
    fn legal_tiles_for_hex(&self, corp_sym: &str, hex_id: &str) -> Vec<(String, Vec<u8>)> { ... }
    
    /// Calculate optimal routes and revenue for a corporation.
    fn calculate_routes(&self, corp_sym: &str) -> (Vec<RouteData>, i32) { ... }
    
    /// Lay a tile on a hex.
    fn lay_tile(&mut self, hex_id: &str, tile_id: &str, rotation: u8) { ... }
    
    /// Place a token on a city.
    fn place_token(&mut self, corp_sym: &str, hex_id: &str, city_index: usize) { ... }
}
```

---

## Tile Catalog Data

The full tile DSL strings must be extracted from `graph.py` TileConfig (lines 2134-2664). There are ~200 tile definitions across all colors, but 1830 only uses 46. The relevant ones are listed in `g1830.py` TILES dict.

**Approach:** Rather than extracting all ~200 tiles, extract only the 46 used by 1830 plus their upgrade targets. The upgrade chain for 1830 is:

```
Yellow tiles (initial placement):
  1, 2, 3, 4, 7, 8, 9, 55, 56, 57, 58, 69

Green tiles (upgrade from yellow):
  14, 15, 16, 18, 19, 20, 23, 24, 25, 26, 27, 28, 29

Brown tiles (upgrade from green):
  39, 40, 41, 42, 43, 44, 45, 46, 47, 53, 54, 59, 61, 62, 63, 64, 65, 66, 67, 68, 70
```

Each tile's DSL string should be copied directly from the Python TileConfig catalog. The parser handles the rest.

---

## Validation

### Test 1: Tile parsing roundtrip
```python
# Parse all 46 tiles, verify edge/city/town counts match expectations
from engine_rs import TileCatalog
catalog = TileCatalog()
t = catalog.get("14")  # Green city tile
assert len(t.cities) == 1
assert len(t.paths) == 2
assert set(t.edges) == {0, 2}  # or whatever the correct edges are
```

### Test 2: Connectivity after tile placement
```python
# Place tiles and verify connectivity matches Python engine
# Use a known game state from human games
```

### Test 3: Route revenue matches Python
```python
# At specific game states, verify auto-router produces same revenue
# Compare against Python autorouter output
```

### Test 4: Replay validation continues
```python
# Extend the Phase 3 replay validator to cover Operating Round actions
# LayTile, PlaceToken, RunRoutes should now produce matching state
```

---

## Effort Estimate

| Component | Effort | Lines (est.) |
|-----------|--------|-------------|
| tiles.rs (DSL parser + catalog) | 3-4 days | ~600 |
| map.rs (graph connectivity) | 4-5 days | ~500 |
| router.rs (auto-routing) | 3-4 days | ~400 |
| graph.rs updates (Path on Tile) | 1-2 days | ~200 |
| game.rs integration | 2-3 days | ~300 |
| g1830.rs (tile DSL strings, upgrades) | 2-3 days | ~400 |
| Validation + debugging | 3-4 days | — |
| **Total** | **~3-4 weeks** | **~2,400** |

## Priority Order

1. **tiles.rs** — tile parsing is foundational, everything else depends on it
2. **g1830.rs tile catalog** — actual DSL strings for all 46 tiles
3. **graph.rs updates** — make Tile hold parsed PathDef data
4. **map.rs** — connectivity engine (walk algorithm)
5. **game.rs integration** — wire up lay_tile, place_token, connected_hexes
6. **router.rs** — auto-routing (can be last since RunRoutes can initially accept pre-computed routes from Python)
7. **Validation** — replay human games through both engines

### Dependency on Phase 3

Phase 4 unblocks Phase 3's Operating Round. The recommended workflow:
1. Implement Phase 4 first (tiles + graph + routing)
2. Then complete Phase 3's Operating Round steps using Phase 4's connectivity queries
3. Auction and Stock round logic from Phase 3 can proceed in parallel since they don't need graph connectivity
