use std::collections::{HashMap, HashSet};

use crate::graph::Hex;
use crate::map::{NodeId, NodeType};
use crate::tiles::PathEndpoint;

// ---------------------------------------------------------------------------
// Route candidate
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct RouteCandidate {
    pub nodes: Vec<NodeId>,
    pub revenue: i32,
    /// Bitfield of hexsides used by this route (for conflict detection).
    pub hexside_bits: u128,
}

// ---------------------------------------------------------------------------
// Route finder
// ---------------------------------------------------------------------------

pub struct RouteFinder {
    hexside_bits: HashMap<(String, u8), u128>,
    next_bit: u32,
}

impl Default for RouteFinder {
    fn default() -> Self {
        Self::new()
    }
}

impl RouteFinder {
    pub fn new() -> Self {
        RouteFinder {
            hexside_bits: HashMap::new(),
            next_bit: 0,
        }
    }

    fn assign_bit(&mut self, hex_id: &str, edge: u8) -> u128 {
        let key = (hex_id.to_string(), edge);
        *self.hexside_bits.entry(key).or_insert_with(|| {
            let bit = 1u128 << self.next_bit;
            self.next_bit = (self.next_bit + 1).min(127);
            bit
        })
    }

    /// Assign bit for both sides of a hexside (hex_id/edge and neighbor_id/opposite_edge).
    fn assign_hexside_pair(
        &mut self,
        hex_id: &str,
        edge: u8,
        hex_adjacency: &HashMap<String, HashMap<u8, String>>,
    ) -> u128 {
        // Both sides of the same hexside share one bit
        let key1 = (hex_id.to_string(), edge);
        if let Some(&bit) = self.hexside_bits.get(&key1) {
            return bit;
        }

        // Check if neighbor side was already assigned
        if let Some(neighbor_id) = hex_adjacency.get(hex_id).and_then(|n| n.get(&edge)) {
            let opposite = (edge + 3) % 6;
            let key2 = (neighbor_id.clone(), opposite);
            if let Some(&bit) = self.hexside_bits.get(&key2) {
                self.hexside_bits.insert(key1, bit);
                return bit;
            }
        }

        let bit = self.assign_bit(hex_id, edge);

        // Also assign the opposite side
        if let Some(neighbor_id) = hex_adjacency.get(hex_id).and_then(|n| n.get(&edge)) {
            let opposite = (edge + 3) % 6;
            let key2 = (neighbor_id.clone(), opposite);
            self.hexside_bits.entry(key2).or_insert(bit);
        }

        bit
    }
}

// ---------------------------------------------------------------------------
// Route enumeration
// ---------------------------------------------------------------------------

/// Get the revenue of a node (city/town/offboard).
fn node_revenue(hexes: &[Hex], hex_idx: &HashMap<String, usize>, node: &NodeId) -> i32 {
    let hi = match hex_idx.get(&node.hex_id) {
        Some(&i) => i,
        None => return 0,
    };
    let tile = &hexes[hi].tile;
    match node.node_type {
        NodeType::City => tile.cities.get(node.index).map_or(0, |c| c.revenue),
        NodeType::Town => tile.towns.get(node.index).map_or(0, |t| t.revenue),
        NodeType::Offboard => tile.offboards.get(node.index).map_or(0, |o| o.revenue),
    }
}

/// Find neighboring nodes reachable from `current_node` through tile paths,
/// returning (neighbor_node, hexsides_used_as_bits).
fn reachable_neighbors(
    hexes: &[Hex],
    hex_idx: &HashMap<String, usize>,
    hex_adjacency: &HashMap<String, HashMap<u8, String>>,
    current_node: &NodeId,
    finder: &mut RouteFinder,
) -> Vec<(NodeId, u128)> {
    let mut result = Vec::new();

    let hi = match hex_idx.get(&current_node.hex_id) {
        Some(&i) => i,
        None => return result,
    };
    let tile = &hexes[hi].tile;

    let node_ep = match current_node.node_type {
        NodeType::City => PathEndpoint::City(current_node.index),
        NodeType::Town => PathEndpoint::Town(current_node.index),
        NodeType::Offboard => PathEndpoint::Offboard(current_node.index),
    };

    // Find paths from this node to edges, then follow through neighboring hexes
    for path in &tile.paths {
        if path.terminal {
            continue;
        }

        let exit = if path.a == node_ep {
            &path.b
        } else if path.b == node_ep {
            &path.a
        } else {
            continue;
        };

        match exit {
            PathEndpoint::Edge(exit_edge) => {
                let bit =
                    finder.assign_hexside_pair(&current_node.hex_id, *exit_edge, hex_adjacency);

                // Follow to neighbor hex and find what node we reach
                if let Some(neighbor_id) = hex_adjacency
                    .get(&current_node.hex_id)
                    .and_then(|n| n.get(exit_edge))
                {
                    let enter_edge = (*exit_edge + 3) % 6;
                    // Trace through the neighbor (may pass through multiple hexes)
                    let traced = trace_to_node(
                        hexes,
                        hex_idx,
                        hex_adjacency,
                        neighbor_id,
                        enter_edge,
                        finder,
                    );
                    for (dest_node, extra_bits) in traced {
                        result.push((dest_node, bit | extra_bits));
                    }
                }
            }
            PathEndpoint::City(idx) => {
                // Same-tile city connection (no hexside used)
                let dest = NodeId {
                    hex_id: current_node.hex_id.clone(),
                    node_type: NodeType::City,
                    index: *idx,
                };
                result.push((dest, 0));
            }
            PathEndpoint::Town(idx) => {
                let dest = NodeId {
                    hex_id: current_node.hex_id.clone(),
                    node_type: NodeType::Town,
                    index: *idx,
                };
                result.push((dest, 0));
            }
            _ => {}
        }
    }

    result
}

/// Trace from an entry edge through a hex until we reach a node or dead end.
/// Returns all reachable (node, accumulated_hexside_bits) pairs.
fn trace_to_node(
    hexes: &[Hex],
    hex_idx: &HashMap<String, usize>,
    hex_adjacency: &HashMap<String, HashMap<u8, String>>,
    hex_id: &str,
    enter_edge: u8,
    finder: &mut RouteFinder,
) -> Vec<(NodeId, u128)> {
    let mut results = Vec::new();

    let hi = match hex_idx.get(hex_id) {
        Some(&i) => i,
        None => return results,
    };
    let tile = &hexes[hi].tile;
    let enter_ep = PathEndpoint::Edge(enter_edge);

    for path in &tile.paths {
        if path.terminal {
            continue;
        }

        let dest = if path.a == enter_ep {
            &path.b
        } else if path.b == enter_ep {
            &path.a
        } else {
            continue;
        };

        match dest {
            PathEndpoint::Edge(exit_edge) => {
                // Track passes through — continue to next hex
                let bit = finder.assign_hexside_pair(hex_id, *exit_edge, hex_adjacency);

                if let Some(neighbor_id) = hex_adjacency.get(hex_id).and_then(|n| n.get(exit_edge))
                {
                    let next_enter = (*exit_edge + 3) % 6;
                    let traced = trace_to_node(
                        hexes,
                        hex_idx,
                        hex_adjacency,
                        neighbor_id,
                        next_enter,
                        finder,
                    );
                    for (node, extra_bits) in traced {
                        results.push((node, bit | extra_bits));
                    }
                }
            }
            PathEndpoint::City(idx) => {
                results.push((
                    NodeId {
                        hex_id: hex_id.to_string(),
                        node_type: NodeType::City,
                        index: *idx,
                    },
                    0,
                ));
            }
            PathEndpoint::Town(idx) => {
                results.push((
                    NodeId {
                        hex_id: hex_id.to_string(),
                        node_type: NodeType::Town,
                        index: *idx,
                    },
                    0,
                ));
            }
            PathEndpoint::Offboard(idx) => {
                results.push((
                    NodeId {
                        hex_id: hex_id.to_string(),
                        node_type: NodeType::Offboard,
                        index: *idx,
                    },
                    0,
                ));
            }
            PathEndpoint::Junction => {
                // Follow through junction
                for other_path in &tile.paths {
                    if std::ptr::eq(path, other_path) || other_path.terminal {
                        continue;
                    }
                    let has_junction = other_path.a == PathEndpoint::Junction
                        || other_path.b == PathEndpoint::Junction;
                    if !has_junction {
                        continue;
                    }
                    let other_exit = if other_path.a == PathEndpoint::Junction {
                        &other_path.b
                    } else {
                        &other_path.a
                    };
                    if let PathEndpoint::Edge(e) = other_exit {
                        if *e == enter_edge {
                            continue;
                        }
                        let bit = finder.assign_hexside_pair(hex_id, *e, hex_adjacency);
                        if let Some(neighbor_id) = hex_adjacency.get(hex_id).and_then(|n| n.get(e))
                        {
                            let next_enter = (*e + 3) % 6;
                            let traced = trace_to_node(
                                hexes,
                                hex_idx,
                                hex_adjacency,
                                neighbor_id,
                                next_enter,
                                finder,
                            );
                            for (node, extra_bits) in traced {
                                results.push((node, bit | extra_bits));
                            }
                        }
                    }
                }
            }
        }
    }

    results
}

// ---------------------------------------------------------------------------
// DFS route enumeration
// ---------------------------------------------------------------------------

struct DfsState<'a> {
    hexes: &'a [Hex],
    hex_idx: &'a HashMap<String, usize>,
    hex_adjacency: &'a HashMap<String, HashMap<u8, String>>,
    finder: RouteFinder,
    candidates: Vec<RouteCandidate>,
}

/// Enumerate all valid routes for a train starting from tokened cities.
pub fn enumerate_routes(
    hexes: &[Hex],
    hex_idx: &HashMap<String, usize>,
    hex_adjacency: &HashMap<String, HashMap<u8, String>>,
    token_nodes: &[NodeId],
    train_distance: u32,
    is_d_train: bool,
) -> Vec<RouteCandidate> {
    let mut state = DfsState {
        hexes,
        hex_idx,
        hex_adjacency,
        finder: RouteFinder::new(),
        candidates: Vec::new(),
    };

    for start in token_nodes {
        let mut visited = HashSet::new();
        visited.insert(start.clone());
        let revenue = node_revenue(hexes, hex_idx, start);

        dfs_routes(
            &mut state,
            start,
            &mut visited,
            &mut vec![start.clone()],
            revenue,
            0u128,
            1, // distance = 1 stop so far
            train_distance,
            is_d_train,
        );
    }

    state.candidates
}

#[allow(clippy::too_many_arguments)]
fn dfs_routes(
    state: &mut DfsState<'_>,
    current: &NodeId,
    visited: &mut HashSet<NodeId>,
    path: &mut Vec<NodeId>,
    revenue: i32,
    bits: u128,
    stops: u32,
    max_distance: u32,
    is_d_train: bool,
) {
    // A valid route needs at least 2 stops (cities/towns/offboards)
    if path.len() >= 2 {
        state.candidates.push(RouteCandidate {
            nodes: path.clone(),
            revenue,
            hexside_bits: bits,
        });
    }

    if stops >= max_distance {
        return;
    }

    // Get reachable neighbors
    let neighbors = reachable_neighbors(
        state.hexes,
        state.hex_idx,
        state.hex_adjacency,
        current,
        &mut state.finder,
    );

    for (neighbor, edge_bits) in neighbors {
        if visited.contains(&neighbor) {
            continue;
        }
        // For D trains: towns don't count as stops but still add revenue
        let is_town = neighbor.node_type == NodeType::Town;
        let stop_cost = if is_d_train && is_town { 0 } else { 1 };
        let neighbor_rev = node_revenue(state.hexes, state.hex_idx, &neighbor);

        visited.insert(neighbor.clone());
        path.push(neighbor.clone());

        dfs_routes(
            state,
            &neighbor,
            visited,
            path,
            revenue + neighbor_rev,
            bits | edge_bits,
            stops + stop_cost,
            max_distance,
            is_d_train,
        );

        path.pop();
        visited.remove(&neighbor);
    }
}

// ---------------------------------------------------------------------------
// Optimal route combination
// ---------------------------------------------------------------------------

/// Find the revenue-maximizing combination of routes (one per train) with no
/// hexside conflicts.
pub fn find_best_routes(
    candidates_per_train: &[Vec<RouteCandidate>],
) -> (Vec<RouteCandidate>, i32) {
    if candidates_per_train.is_empty() {
        return (Vec::new(), 0);
    }

    if candidates_per_train.len() == 1 {
        // Single train: just pick the highest revenue route
        let best = candidates_per_train[0].iter().max_by_key(|r| r.revenue);
        return match best {
            Some(route) => (vec![route.clone()], route.revenue),
            None => (Vec::new(), 0),
        };
    }

    // For 2-3 trains: brute-force all combinations
    let mut best_revenue = 0i32;
    let mut best_combo: Vec<RouteCandidate> = Vec::new();

    find_best_recursive(
        candidates_per_train,
        0,
        &mut Vec::new(),
        0,
        0,
        &mut best_revenue,
        &mut best_combo,
    );

    (best_combo, best_revenue)
}

fn find_best_recursive(
    candidates_per_train: &[Vec<RouteCandidate>],
    train_index: usize,
    current_combo: &mut Vec<RouteCandidate>,
    current_revenue: i32,
    used_bits: u128,
    best_revenue: &mut i32,
    best_combo: &mut Vec<RouteCandidate>,
) {
    if train_index >= candidates_per_train.len() {
        if current_revenue > *best_revenue {
            *best_revenue = current_revenue;
            *best_combo = current_combo.clone();
        }
        return;
    }

    // Option: skip this train (run 0 routes for it)
    find_best_recursive(
        candidates_per_train,
        train_index + 1,
        current_combo,
        current_revenue,
        used_bits,
        best_revenue,
        best_combo,
    );

    // Try each candidate for this train
    for route in &candidates_per_train[train_index] {
        // Check hexside conflict
        if route.hexside_bits & used_bits != 0 {
            continue;
        }

        current_combo.push(route.clone());
        find_best_recursive(
            candidates_per_train,
            train_index + 1,
            current_combo,
            current_revenue + route.revenue,
            used_bits | route.hexside_bits,
            best_revenue,
            best_combo,
        );
        current_combo.pop();
    }
}

/// Calculate optimal routes and total revenue for a corporation.
pub fn calculate_corp_routes(
    hexes: &[Hex],
    hex_idx: &HashMap<String, usize>,
    hex_adjacency: &HashMap<String, HashMap<u8, String>>,
    token_nodes: &[NodeId],
    trains: &[(u32, bool)], // (distance, is_d_train)
) -> (Vec<RouteCandidate>, i32) {
    if trains.is_empty() || token_nodes.is_empty() {
        return (Vec::new(), 0);
    }

    let mut candidates_per_train = Vec::new();
    for &(distance, is_d) in trains {
        let candidates =
            enumerate_routes(hexes, hex_idx, hex_adjacency, token_nodes, distance, is_d);
        candidates_per_train.push(candidates);
    }

    find_best_routes(&candidates_per_train)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{City, Hex, Tile, Town};
    use crate::tiles::{parse_tile, TileColor};

    fn tile_from_dsl(name: &str, dsl: &str, color: TileColor) -> Tile {
        let tile_def = parse_tile(name, dsl, color);
        let mut tile = Tile::new(name.to_string(), name.to_string());
        tile.color = tile_def.color;
        tile.paths = tile_def.paths;
        for cd in &tile_def.cities {
            tile.cities.push(City::new(cd.revenue, cd.slots));
        }
        for td in &tile_def.towns {
            tile.towns.push(Town::new(td.revenue));
        }
        tile
    }

    fn place_token(tile: &mut Tile, city_index: usize, corp_sym: &str) {
        let city = &mut tile.cities[city_index];
        let slot = city.tokens.iter().position(|t| t.is_none()).unwrap();
        let mut tok = crate::entities::Token::new(corp_sym.to_string(), 0);
        tok.used = true;
        city.tokens[slot] = Some(tok);
    }

    /// Simple linear: A(city:20, token) -- B(track) -- C(city:30)
    /// A exits edge 1, B passes edge 4→1, C enters edge 4
    fn build_linear_for_routing() -> (
        Vec<Hex>,
        HashMap<String, usize>,
        HashMap<String, HashMap<u8, String>>,
        Vec<NodeId>,
    ) {
        let mut tile_a = tile_from_dsl("a", "city=revenue:20;path=a:1,b:_0", TileColor::Yellow);
        place_token(&mut tile_a, 0, "PRR");

        let tile_b = tile_from_dsl("b", "path=a:4,b:1", TileColor::Yellow);

        let tile_c = tile_from_dsl("c", "city=revenue:30;path=a:4,b:_0", TileColor::Yellow);

        let hexes = vec![
            Hex::new("A".to_string(), tile_a),
            Hex::new("B".to_string(), tile_b),
            Hex::new("C".to_string(), tile_c),
        ];
        let hex_idx: HashMap<String, usize> = [
            ("A".to_string(), 0),
            ("B".to_string(), 1),
            ("C".to_string(), 2),
        ]
        .into();
        let adjacency: HashMap<String, HashMap<u8, String>> = [
            ("A".to_string(), [(1u8, "B".to_string())].into()),
            (
                "B".to_string(),
                [(4u8, "A".to_string()), (1u8, "C".to_string())].into(),
            ),
            ("C".to_string(), [(4u8, "B".to_string())].into()),
        ]
        .into();
        let token_nodes = vec![NodeId {
            hex_id: "A".to_string(),
            node_type: NodeType::City,
            index: 0,
        }];

        (hexes, hex_idx, adjacency, token_nodes)
    }

    #[test]
    fn single_train_finds_best_route() {
        let (hexes, hex_idx, adjacency, token_nodes) = build_linear_for_routing();

        // Distance-2 train: can visit 2 cities → A(20) + C(30) = 50
        let (routes, revenue) =
            calculate_corp_routes(&hexes, &hex_idx, &adjacency, &token_nodes, &[(2, false)]);

        assert_eq!(revenue, 50, "Best route should be A+C = 50");
        assert_eq!(routes.len(), 1);
    }

    #[test]
    fn no_trains_no_revenue() {
        let (hexes, hex_idx, adjacency, token_nodes) = build_linear_for_routing();

        let (routes, revenue) =
            calculate_corp_routes(&hexes, &hex_idx, &adjacency, &token_nodes, &[]);

        assert_eq!(revenue, 0);
        assert!(routes.is_empty());
    }

    #[test]
    fn no_tokens_no_revenue() {
        let (hexes, hex_idx, adjacency, _) = build_linear_for_routing();

        let (routes, revenue) =
            calculate_corp_routes(&hexes, &hex_idx, &adjacency, &[], &[(2, false)]);

        assert_eq!(revenue, 0);
        assert!(routes.is_empty());
    }

    #[test]
    fn distance_1_train_stays_home() {
        let (hexes, hex_idx, adjacency, token_nodes) = build_linear_for_routing();

        // Distance-1 train can only visit 1 city — needs 2 stops minimum
        let candidates = enumerate_routes(&hexes, &hex_idx, &adjacency, &token_nodes, 1, false);

        // No valid route (need 2+ stops for a valid route)
        assert!(
            candidates.is_empty(),
            "Distance-1 train shouldn't find valid routes (needs 2 stops)"
        );
    }

    /// Build a diamond:  A -- B
    ///                     \  |
    ///                      C
    /// A(city:20, token), B(city:30), C(city:40)
    fn build_diamond_for_routing() -> (
        Vec<Hex>,
        HashMap<String, usize>,
        HashMap<String, HashMap<u8, String>>,
        Vec<NodeId>,
    ) {
        let mut tile_a = tile_from_dsl(
            "a",
            "city=revenue:20;path=a:1,b:_0;path=a:2,b:_0",
            TileColor::Green,
        );
        place_token(&mut tile_a, 0, "PRR");

        let tile_b = tile_from_dsl(
            "b",
            "city=revenue:30;path=a:4,b:_0;path=a:3,b:_0",
            TileColor::Green,
        );

        let tile_c = tile_from_dsl(
            "c",
            "city=revenue:40;path=a:5,b:_0;path=a:0,b:_0",
            TileColor::Green,
        );

        let hexes = vec![
            Hex::new("A".to_string(), tile_a),
            Hex::new("B".to_string(), tile_b),
            Hex::new("C".to_string(), tile_c),
        ];
        let hex_idx: HashMap<String, usize> = [
            ("A".to_string(), 0),
            ("B".to_string(), 1),
            ("C".to_string(), 2),
        ]
        .into();
        // A.1→B (B enters 4), A.2→C (C enters 5), B.3→C (C enters 0)
        let adjacency: HashMap<String, HashMap<u8, String>> = [
            (
                "A".to_string(),
                [(1u8, "B".to_string()), (2u8, "C".to_string())].into(),
            ),
            (
                "B".to_string(),
                [(4u8, "A".to_string()), (3u8, "C".to_string())].into(),
            ),
            (
                "C".to_string(),
                [(5u8, "A".to_string()), (0u8, "B".to_string())].into(),
            ),
        ]
        .into();
        let token_nodes = vec![NodeId {
            hex_id: "A".to_string(),
            node_type: NodeType::City,
            index: 0,
        }];

        (hexes, hex_idx, adjacency, token_nodes)
    }

    #[test]
    fn diamond_distance3_visits_all() {
        let (hexes, hex_idx, adjacency, token_nodes) = build_diamond_for_routing();

        // Distance-3 train: best route A(20)+B(30)+C(40)=90 or A+C+B=90
        let (routes, revenue) =
            calculate_corp_routes(&hexes, &hex_idx, &adjacency, &token_nodes, &[(3, false)]);

        assert_eq!(revenue, 90, "Should visit all 3 cities for 90 revenue");
        assert_eq!(routes.len(), 1);
        assert_eq!(routes[0].nodes.len(), 3);
    }

    #[test]
    fn diamond_distance2_picks_best_pair() {
        let (hexes, hex_idx, adjacency, token_nodes) = build_diamond_for_routing();

        // Distance-2 train: best is A(20)+C(40)=60
        let (routes, revenue) =
            calculate_corp_routes(&hexes, &hex_idx, &adjacency, &token_nodes, &[(2, false)]);

        assert_eq!(revenue, 60, "Best 2-stop route: A+C=60");
        assert_eq!(routes.len(), 1);
    }

    #[test]
    fn two_trains_no_hexside_conflict() {
        // A(token) -- B -- C, A(token) -- D
        // Two distance-2 trains should find non-conflicting routes
        let mut tile_a = tile_from_dsl(
            "a",
            "city=revenue:10;path=a:1,b:_0;path=a:3,b:_0",
            TileColor::Green,
        );
        place_token(&mut tile_a, 0, "PRR");

        let tile_b = tile_from_dsl("b", "path=a:4,b:1", TileColor::Yellow);

        let tile_c = tile_from_dsl("c", "city=revenue:30;path=a:4,b:_0", TileColor::Yellow);

        let tile_d = tile_from_dsl("d", "city=revenue:20;path=a:0,b:_0", TileColor::Yellow);

        let hexes = vec![
            Hex::new("A".to_string(), tile_a),
            Hex::new("B".to_string(), tile_b),
            Hex::new("C".to_string(), tile_c),
            Hex::new("D".to_string(), tile_d),
        ];
        let hex_idx: HashMap<String, usize> = [
            ("A".to_string(), 0),
            ("B".to_string(), 1),
            ("C".to_string(), 2),
            ("D".to_string(), 3),
        ]
        .into();
        // A.1→B, B.4→A, B.1→C, C.4→B, A.3→D, D.0→A
        let adjacency: HashMap<String, HashMap<u8, String>> = [
            (
                "A".to_string(),
                [(1u8, "B".to_string()), (3u8, "D".to_string())].into(),
            ),
            (
                "B".to_string(),
                [(4u8, "A".to_string()), (1u8, "C".to_string())].into(),
            ),
            ("C".to_string(), [(4u8, "B".to_string())].into()),
            ("D".to_string(), [(0u8, "A".to_string())].into()),
        ]
        .into();

        let token_nodes = vec![NodeId {
            hex_id: "A".to_string(),
            node_type: NodeType::City,
            index: 0,
        }];

        // Two distance-2 trains
        let (routes, revenue) = calculate_corp_routes(
            &hexes,
            &hex_idx,
            &adjacency,
            &token_nodes,
            &[(2, false), (2, false)],
        );

        // Best: train1: A(10)+C(30)=40, train2: A(10)+D(20)=30, total=70
        // They don't share hexsides since they go in different directions
        assert_eq!(revenue, 70, "Two non-conflicting routes: A-C=40 + A-D=30");
        assert_eq!(routes.len(), 2);
    }

    #[test]
    fn town_adds_revenue_on_route() {
        // A(city:20, token) → B(town:10) → C(city:30)
        let mut tile_a = tile_from_dsl("a", "city=revenue:20;path=a:1,b:_0", TileColor::Yellow);
        place_token(&mut tile_a, 0, "PRR");

        let tile_b = tile_from_dsl(
            "b",
            "town=revenue:10;path=a:4,b:_0;path=a:_0,b:1",
            TileColor::Yellow,
        );

        let tile_c = tile_from_dsl("c", "city=revenue:30;path=a:4,b:_0", TileColor::Yellow);

        let hexes = vec![
            Hex::new("A".to_string(), tile_a),
            Hex::new("B".to_string(), tile_b),
            Hex::new("C".to_string(), tile_c),
        ];
        let hex_idx: HashMap<String, usize> = [
            ("A".to_string(), 0),
            ("B".to_string(), 1),
            ("C".to_string(), 2),
        ]
        .into();
        let adjacency: HashMap<String, HashMap<u8, String>> = [
            ("A".to_string(), [(1u8, "B".to_string())].into()),
            (
                "B".to_string(),
                [(4u8, "A".to_string()), (1u8, "C".to_string())].into(),
            ),
            ("C".to_string(), [(4u8, "B".to_string())].into()),
        ]
        .into();
        let token_nodes = vec![NodeId {
            hex_id: "A".to_string(),
            node_type: NodeType::City,
            index: 0,
        }];

        // Distance-3 train: A(20) + B(10) + C(30) = 60
        let (routes, revenue) =
            calculate_corp_routes(&hexes, &hex_idx, &adjacency, &token_nodes, &[(3, false)]);

        assert_eq!(revenue, 60, "Route through town: 20+10+30=60");
        assert_eq!(routes.len(), 1);
    }

    #[test]
    fn route_must_start_from_tokened_city() {
        // Only the tokened city should be a valid starting point
        let (hexes, hex_idx, adjacency, token_nodes) = build_linear_for_routing();

        let candidates = enumerate_routes(&hexes, &hex_idx, &adjacency, &token_nodes, 3, false);

        // All candidates should start from A (the tokened city)
        for c in &candidates {
            assert_eq!(
                c.nodes[0].hex_id, "A",
                "Route must start from tokened city A"
            );
        }
    }

    #[test]
    fn find_best_routes_with_conflict() {
        // Two candidates that share a hexside
        let r1 = RouteCandidate {
            nodes: vec![],
            revenue: 50,
            hexside_bits: 0b0001,
        };
        let r2 = RouteCandidate {
            nodes: vec![],
            revenue: 40,
            hexside_bits: 0b0001, // conflicts with r1
        };
        let r3 = RouteCandidate {
            nodes: vec![],
            revenue: 30,
            hexside_bits: 0b0010, // no conflict with r1
        };

        let (routes, revenue) = find_best_routes(&[vec![r1, r3.clone()], vec![r2, r3]]);

        // Best non-conflicting combo: r1(50) + r3(30) = 80
        assert_eq!(revenue, 80);
        assert_eq!(routes.len(), 2);
    }

    #[test]
    fn single_candidate_list() {
        let r1 = RouteCandidate {
            nodes: vec![],
            revenue: 42,
            hexside_bits: 0,
        };

        let (routes, revenue) = find_best_routes(&[vec![r1]]);
        assert_eq!(revenue, 42);
        assert_eq!(routes.len(), 1);
    }

    #[test]
    fn empty_candidates() {
        let (routes, revenue) = find_best_routes(&[]);
        assert_eq!(revenue, 0);
        assert!(routes.is_empty());
    }
}
