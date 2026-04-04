use std::collections::{HashMap, HashSet};

use crate::graph::{Hex, Tile};
use crate::tiles::PathEndpoint;

// ---------------------------------------------------------------------------
// Node identification
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub struct NodeId {
    pub hex_id: String,
    pub node_type: NodeType,
    pub index: usize,
}

#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub enum NodeType {
    City,
    Town,
    Offboard,
}

// ---------------------------------------------------------------------------
// Tokenable city
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct TokenableCity {
    pub hex_id: String,
    pub city_index: usize,
}

// ---------------------------------------------------------------------------
// Graph cache
// ---------------------------------------------------------------------------

/// Cached connectivity data for a corporation.
#[derive(Clone, Debug)]
pub struct CorpGraph {
    /// Reachable hexes and the edges through which they connect.
    pub connected_hexes: HashMap<String, HashSet<u8>>,
    /// All reachable nodes (cities/towns/offboards).
    pub connected_nodes: HashSet<NodeId>,
    /// Cities where a new token can be placed.
    pub tokenable_cities: Vec<TokenableCity>,
}

/// Full graph cache holding per-corporation data.
#[derive(Clone, Debug, Default)]
pub struct GraphCache {
    cache: HashMap<String, CorpGraph>,
}

impl GraphCache {
    pub fn new() -> Self {
        GraphCache {
            cache: HashMap::new(),
        }
    }

    /// Clear the entire cache (call after tile/token changes).
    pub fn clear(&mut self) {
        self.cache.clear();
    }

    /// Check if the cache has a key (for debugging).
    pub fn has_key(&self, corp_sym: &str) -> bool {
        self.cache.contains_key(corp_sym)
    }

    /// Get or compute the graph for a corporation.
    /// `reservations` is a list of (hex_id, city_index, reserving_corp_sym) for
    /// home city reservations. In 1830 with TILE_RESERVATION_BLOCKS_OTHERS="always",
    /// a city reserved for another corp is not tokenable.
    pub fn get_or_compute(
        &mut self,
        corp_sym: &str,
        hexes: &[Hex],
        hex_idx: &HashMap<String, usize>,
        hex_adjacency: &HashMap<String, HashMap<u8, String>>,
        corp_token_hexes: &[(String, usize)],
        reservations: &[(String, usize, String)], // (hex_id, city_index, reserving_corp)
    ) -> &CorpGraph {
        if !self.cache.contains_key(corp_sym) {
            let graph = compute_corp_graph(
                corp_sym,
                hexes,
                hex_idx,
                hex_adjacency,
                corp_token_hexes,
                reservations,
            );
            self.cache.insert(corp_sym.to_string(), graph);
        }
        &self.cache[corp_sym]
    }
}

// ---------------------------------------------------------------------------
// Core graph computation
// ---------------------------------------------------------------------------

/// Compute connectivity graph for a single corporation.
fn compute_corp_graph(
    corp_sym: &str,
    hexes: &[Hex],
    hex_idx: &HashMap<String, usize>,
    hex_adjacency: &HashMap<String, HashMap<u8, String>>,
    corp_token_hexes: &[(String, usize)],
    reservations: &[(String, usize, String)],
) -> CorpGraph {
    let mut visited_nodes = HashSet::new();
    let mut visited_hexes: HashMap<String, HashSet<u8>> = HashMap::new();

    // Each token walk gets its own visited_paths (matching Python's per-token
    // fresh visited_paths dict).  The visited_nodes set for each walk is
    // pre-seeded with the OTHER token cities so the walk stops when it reaches
    // another token rather than traversing through it (Python does this too).
    // Results are merged across all token walks.
    for (token_hex_id, city_index) in corp_token_hexes {
        let start_node = NodeId {
            hex_id: token_hex_id.clone(),
            node_type: NodeType::City,
            index: *city_index,
        };

        // Pre-seed visited_nodes with other token cities
        let mut local_visited_nodes: HashSet<NodeId> = corp_token_hexes
            .iter()
            .filter(|(h, ci)| h != token_hex_id || *ci != *city_index)
            .map(|(h, ci)| NodeId {
                hex_id: h.clone(),
                node_type: NodeType::City,
                index: *ci,
            })
            .collect();
        let mut local_visited_hexes: HashMap<String, HashSet<u8>> = HashMap::new();
        let mut local_visited_paths: HashSet<(String, usize)> = HashSet::new();
        let mut edge_counter: HashMap<(String, u8), u32> = HashMap::new();

        walk_from_node(
            hexes,
            hex_idx,
            hex_adjacency,
            token_hex_id,
            &start_node,
            &mut local_visited_nodes,
            &mut local_visited_hexes,
            &mut local_visited_paths,
            corp_sym,
            false, // converging starts as false for the token node
            &mut edge_counter,
        );

        // Merge results
        for node in local_visited_nodes {
            visited_nodes.insert(node);
        }
        for (hex_id, edges) in local_visited_hexes {
            visited_hexes.entry(hex_id).or_default().extend(edges);
        }
    }

    // Determine tokenable cities: reachable cities with open slots that the corp
    // doesn't already have a token in.
    let mut tokenable = Vec::new();

    for node in &visited_nodes {
        if node.node_type != NodeType::City {
            continue;
        }
        // Skip cities where this corp already has a token in the same hex
        // (1830 rule: one token per hex per corp, but we check the city specifically)
        if corp_token_hexes
            .iter()
            .any(|(h, ci)| h == &node.hex_id && *ci == node.index)
        {
            continue;
        }

        if let Some(&hi) = hex_idx.get(&node.hex_id) {
            let city = &hexes[hi].tile.cities[node.index];
            // Must have at least one empty slot
            if city.tokens.iter().any(|t| t.is_none()) {
                // In 1830: one token per hex per corp
                let corp_already_in_hex = hexes[hi].tile.cities.iter().any(|c| {
                    c.tokens
                        .iter()
                        .any(|t| t.as_ref().is_some_and(|tok| tok.corporation_id == corp_sym))
                });
                if !corp_already_in_hex {
                    // Check reservations: in 1830 (reservation_blocks="always"),
                    // if ANY city on this hex is reserved for another corp,
                    // this corp can't place a token here.
                    let blocked_by_reservation = reservations
                        .iter()
                        .any(|(rh, _rc, rsym)| rh == &node.hex_id && rsym != corp_sym);
                    if !blocked_by_reservation {
                        tokenable.push(TokenableCity {
                            hex_id: node.hex_id.clone(),
                            city_index: node.index,
                        });
                    }
                }
            }
        }
    }

    CorpGraph {
        connected_hexes: visited_hexes,
        connected_nodes: visited_nodes,
        tokenable_cities: tokenable,
    }
}

// ---------------------------------------------------------------------------
// Walk algorithm
// ---------------------------------------------------------------------------

/// Check if an edge on a tile has more than one path using it (converging exit).
/// When true, paths walked through this edge should be temporarily visited
/// (removed from visited_paths after the recursive walk completes).
fn is_converging_exit(tile: &Tile, edge: u8) -> bool {
    // An edge is "converging" when multiple edge-to-edge paths share it.
    // City/town-to-edge paths don't count — they represent a single route
    // through the city, not a crossover.
    let edge_ep = PathEndpoint::Edge(edge);
    let count = tile
        .paths
        .iter()
        .filter(|p| {
            !p.terminal
                && (p.a == edge_ep || p.b == edge_ep)
                && p.a.edge_num().is_some()
                && p.b.edge_num().is_some()
        })
        .count();
    count > 1
}

/// Walk outward from a node (city/town/offboard) through connected paths.
///
/// `converging` mirrors the Python engine's converging_path flag: when true,
/// paths are temporarily marked visited during the DFS and removed afterwards,
/// allowing other walk branches to reuse them. This is critical for tiles
/// where multiple paths share an edge (crossover / junction tiles).
///
/// `edge_counter` tracks how many times each (hex, edge) has been traversed
/// in the current DFS stack, preventing infinite loops when paths are
/// un-visited due to converging.
#[allow(clippy::too_many_arguments)]
fn walk_from_node(
    hexes: &[Hex],
    hex_idx: &HashMap<String, usize>,
    hex_adjacency: &HashMap<String, HashMap<u8, String>>,
    hex_id: &str,
    node: &NodeId,
    visited_nodes: &mut HashSet<NodeId>,
    visited_hexes: &mut HashMap<String, HashSet<u8>>,
    visited_paths: &mut HashSet<(String, usize)>,
    corp_sym: &str,
    converging: bool,
    edge_counter: &mut HashMap<(String, u8), u32>,
) {
    if visited_nodes.contains(node) {
        return;
    }
    visited_nodes.insert(node.clone());

    // Add the hex itself to visited
    visited_hexes.entry(hex_id.to_string()).or_default();

    let hi = match hex_idx.get(hex_id) {
        Some(&i) => i,
        None => return,
    };
    let tile = &hexes[hi].tile;

    // The endpoint that matches this node
    let node_endpoint = match node.node_type {
        NodeType::City => PathEndpoint::City(node.index),
        NodeType::Town => PathEndpoint::Town(node.index),
        NodeType::Offboard => PathEndpoint::Offboard(node.index),
    };

    // Find all paths that connect to this node
    for (path_idx, path) in tile.paths.iter().enumerate() {
        if path.terminal {
            continue;
        }

        // Skip paths we've already walked
        let path_key = (hex_id.to_string(), path_idx);
        if visited_paths.contains(&path_key) {
            continue;
        }

        // Check if this path connects to our node
        let exit = if path.a == node_endpoint {
            &path.b
        } else if path.b == node_endpoint {
            &path.a
        } else {
            continue;
        };

        // Mark path as visited
        visited_paths.insert(path_key.clone());

        match exit {
            PathEndpoint::Edge(exit_edge) => {
                visited_hexes
                    .entry(hex_id.to_string())
                    .or_default()
                    .insert(*exit_edge);

                // Determine converging for the next walk: true if the current
                // tile has a converging exit at this edge.
                let next_converging =
                    converging || is_converging_exit(tile, *exit_edge);

                // Follow to neighbor hex
                if let Some(neighbor_id) = hex_adjacency.get(hex_id).and_then(|n| n.get(exit_edge))
                {
                    let edge_key = (hex_id.to_string(), *exit_edge);
                    let cnt = edge_counter.entry(edge_key.clone()).or_insert(0);
                    if *cnt == 0 {
                        *cnt += 1;
                        let enter_edge = (*exit_edge + 3) % 6;
                        walk_into_hex(
                            hexes,
                            hex_idx,
                            hex_adjacency,
                            neighbor_id,
                            enter_edge,
                            visited_nodes,
                            visited_hexes,
                            visited_paths,
                            corp_sym,
                            next_converging,
                            edge_counter,
                        );
                        *edge_counter.entry(edge_key).or_insert(0) -= 1;
                    }
                }
            }
            PathEndpoint::City(idx) => {
                let dest_node = NodeId {
                    hex_id: hex_id.to_string(),
                    node_type: NodeType::City,
                    index: *idx,
                };
                if !is_city_blocked(hexes, hex_idx, hex_id, *idx, corp_sym) {
                    walk_from_node(
                        hexes,
                        hex_idx,
                        hex_adjacency,
                        hex_id,
                        &dest_node,
                        visited_nodes,
                        visited_hexes,
                        visited_paths,
                        corp_sym,
                        converging,
                        edge_counter,
                    );
                }
            }
            PathEndpoint::Town(idx) => {
                let dest_node = NodeId {
                    hex_id: hex_id.to_string(),
                    node_type: NodeType::Town,
                    index: *idx,
                };
                walk_from_node(
                    hexes,
                    hex_idx,
                    hex_adjacency,
                    hex_id,
                    &dest_node,
                    visited_nodes,
                    visited_hexes,
                    visited_paths,
                    corp_sym,
                    converging,
                    edge_counter,
                );
            }
            PathEndpoint::Offboard(idx) => {
                let dest_node = NodeId {
                    hex_id: hex_id.to_string(),
                    node_type: NodeType::Offboard,
                    index: *idx,
                };
                visited_nodes.insert(dest_node);
            }
            PathEndpoint::Junction => {
                // Junction: follow all other paths through the junction
                for (other_idx, other_path) in tile.paths.iter().enumerate() {
                    if other_idx == path_idx || other_path.terminal {
                        continue;
                    }
                    let other_key = (hex_id.to_string(), other_idx);
                    if visited_paths.contains(&other_key) {
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
                        visited_paths.insert(other_key.clone());

                        let next_converging =
                            converging || is_converging_exit(tile, *e);

                        visited_hexes
                            .entry(hex_id.to_string())
                            .or_default()
                            .insert(*e);
                        if let Some(neighbor_id) = hex_adjacency.get(hex_id).and_then(|n| n.get(e))
                        {
                            let edge_key = (hex_id.to_string(), *e);
                            let cnt = edge_counter.entry(edge_key.clone()).or_insert(0);
                            if *cnt == 0 {
                                *cnt += 1;
                                let enter_edge = (*e + 3) % 6;
                                walk_into_hex(
                                    hexes,
                                    hex_idx,
                                    hex_adjacency,
                                    neighbor_id,
                                    enter_edge,
                                    visited_nodes,
                                    visited_hexes,
                                    visited_paths,
                                    corp_sym,
                                    next_converging,
                                    edge_counter,
                                );
                                *edge_counter.entry(edge_key).or_insert(0) -= 1;
                            }
                        }

                        // If converging, remove this junction path from visited
                        // so other walk branches can reuse it.
                        if converging {
                            visited_paths.remove(&other_key);
                        }
                    }
                }
            }
        }

        // If this path was entered via a converging walk, remove it from
        // visited_paths so other DFS branches can reuse it.
        if converging {
            visited_paths.remove(&path_key);
        }
    }

    // Note: we do NOT remove the node from visited_nodes during converging.
    // Paths are un-visited to allow other branches to reuse them, but nodes
    // (cities/towns) are genuine destinations that should remain visited.
    // The Python engine's `del visited[self]` affects path reachability,
    // not the final connected_nodes result.
}

/// Walk into a hex from a specific entry edge, following paths.
///
/// See `walk_from_node` for documentation on `converging` and `edge_counter`.
#[allow(clippy::too_many_arguments)]
fn walk_into_hex(
    hexes: &[Hex],
    hex_idx: &HashMap<String, usize>,
    hex_adjacency: &HashMap<String, HashMap<u8, String>>,
    hex_id: &str,
    enter_edge: u8,
    visited_nodes: &mut HashSet<NodeId>,
    visited_hexes: &mut HashMap<String, HashSet<u8>>,
    visited_paths: &mut HashSet<(String, usize)>,
    corp_sym: &str,
    converging: bool,
    edge_counter: &mut HashMap<(String, u8), u32>,
) {
    let hi = match hex_idx.get(hex_id) {
        Some(&i) => i,
        None => return,
    };
    let tile = &hexes[hi].tile;

    visited_hexes
        .entry(hex_id.to_string())
        .or_default()
        .insert(enter_edge);

    let enter_ep = PathEndpoint::Edge(enter_edge);

    // Find all paths that enter from this edge
    for (path_idx, path) in tile.paths.iter().enumerate() {
        if path.terminal {
            continue;
        }

        // Check edge counter: if any of this path's edges have already been
        // traversed in the current DFS stack, skip it to prevent cycles.
        let path_edges: Vec<u8> = [&path.a, &path.b]
            .iter()
            .filter_map(|ep| ep.edge_num())
            .collect();
        let edge_in_stack = path_edges
            .iter()
            .any(|e| *edge_counter.get(&(hex_id.to_string(), *e)).unwrap_or(&0) > 0);
        if edge_in_stack {
            if hex_id == "H10" {
                eprintln!("  path[{}] SKIPPED by edge_in_stack (path_edges={:?})", path_idx, path_edges);
            }
            continue;
        }

        // Skip paths we've already walked
        let path_key = (hex_id.to_string(), path_idx);
        if visited_paths.contains(&path_key) {
            continue;
        }

        let dest = if path.a == enter_ep {
            &path.b
        } else if path.b == enter_ep {
            &path.a
        } else {
            continue;
        };

        // Mark path as visited
        visited_paths.insert(path_key.clone());

        match dest {
            PathEndpoint::Edge(exit_edge) => {
                // Path passes through hex — continue to next neighbor
                visited_hexes
                    .entry(hex_id.to_string())
                    .or_default()
                    .insert(*exit_edge);

                let next_converging =
                    converging || is_converging_exit(tile, *exit_edge);

                if let Some(neighbor_id) = hex_adjacency.get(hex_id).and_then(|n| n.get(exit_edge))
                {
                    let edge_key = (hex_id.to_string(), *exit_edge);
                    let cnt = edge_counter.entry(edge_key.clone()).or_insert(0);
                    if *cnt == 0 {
                        *cnt += 1;
                        let next_enter = (*exit_edge + 3) % 6;
                        walk_into_hex(
                            hexes,
                            hex_idx,
                            hex_adjacency,
                            neighbor_id,
                            next_enter,
                            visited_nodes,
                            visited_hexes,
                            visited_paths,
                            corp_sym,
                            next_converging,
                            edge_counter,
                        );
                        *edge_counter.entry(edge_key).or_insert(0) -= 1;
                    }
                }
            }
            PathEndpoint::City(idx) => {
                let node = NodeId {
                    hex_id: hex_id.to_string(),
                    node_type: NodeType::City,
                    index: *idx,
                };
                if !is_city_blocked(hexes, hex_idx, hex_id, *idx, corp_sym) {
                    walk_from_node(
                        hexes,
                        hex_idx,
                        hex_adjacency,
                        hex_id,
                        &node,
                        visited_nodes,
                        visited_hexes,
                        visited_paths,
                        corp_sym,
                        converging,
                        edge_counter,
                    );
                } else {
                    visited_nodes.insert(node);
                }
            }
            PathEndpoint::Town(idx) => {
                let node = NodeId {
                    hex_id: hex_id.to_string(),
                    node_type: NodeType::Town,
                    index: *idx,
                };
                walk_from_node(
                    hexes,
                    hex_idx,
                    hex_adjacency,
                    hex_id,
                    &node,
                    visited_nodes,
                    visited_hexes,
                    visited_paths,
                    corp_sym,
                    converging,
                    edge_counter,
                );
            }
            PathEndpoint::Offboard(idx) => {
                let node = NodeId {
                    hex_id: hex_id.to_string(),
                    node_type: NodeType::Offboard,
                    index: *idx,
                };
                visited_nodes.insert(node);
            }
            PathEndpoint::Junction => {
                // Follow through junction to all other paths on this tile
                for (other_idx, other_path) in tile.paths.iter().enumerate() {
                    if other_idx == path_idx || other_path.terminal {
                        continue;
                    }
                    let other_key = (hex_id.to_string(), other_idx);
                    if visited_paths.contains(&other_key) {
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
                            continue; // don't go back
                        }
                        visited_paths.insert(other_key.clone());

                        let next_converging =
                            converging || is_converging_exit(tile, *e);

                        visited_hexes
                            .entry(hex_id.to_string())
                            .or_default()
                            .insert(*e);
                        if let Some(neighbor_id) = hex_adjacency.get(hex_id).and_then(|n| n.get(e))
                        {
                            let edge_key = (hex_id.to_string(), *e);
                            let cnt = edge_counter.entry(edge_key.clone()).or_insert(0);
                            if *cnt == 0 {
                                *cnt += 1;
                                let next_enter = (*e + 3) % 6;
                                walk_into_hex(
                                    hexes,
                                    hex_idx,
                                    hex_adjacency,
                                    neighbor_id,
                                    next_enter,
                                    visited_nodes,
                                    visited_hexes,
                                    visited_paths,
                                    corp_sym,
                                    next_converging,
                                    edge_counter,
                                );
                                *edge_counter.entry(edge_key).or_insert(0) -= 1;
                            }
                        }

                        if converging {
                            visited_paths.remove(&other_key);
                        }
                    }
                }
            }
        }

        // If converging, remove this path from visited so other DFS branches
        // can reuse it.
        if converging {
            visited_paths.remove(&path_key);
        }
    }
}

// ---------------------------------------------------------------------------
// Blocking
// ---------------------------------------------------------------------------

/// Check if `corp_sym` has a token placed in a specific city on a hex.
pub fn corp_has_token_in_city(
    hexes: &[Hex],
    hex_idx: &HashMap<String, usize>,
    hex_id: &str,
    city_index: usize,
    corp_sym: &str,
) -> bool {
    let hi = match hex_idx.get(hex_id) {
        Some(&i) => i,
        None => return false,
    };
    let tile = &hexes[hi].tile;
    if city_index >= tile.cities.len() {
        return false;
    }
    tile.cities[city_index]
        .tokens
        .iter()
        .any(|t| t.as_ref().is_some_and(|tok| tok.corporation_id == corp_sym))
}

/// A city blocks traversal if all token slots are filled and none belong to `corp_sym`.
fn is_city_blocked(
    hexes: &[Hex],
    hex_idx: &HashMap<String, usize>,
    hex_id: &str,
    city_index: usize,
    corp_sym: &str,
) -> bool {
    let hi = match hex_idx.get(hex_id) {
        Some(&i) => i,
        None => return false,
    };
    let tile = &hexes[hi].tile;
    if city_index >= tile.cities.len() {
        return false;
    }
    let city = &tile.cities[city_index];

    // If any slot is empty, not blocked
    if city.tokens.iter().any(|t| t.is_none()) {
        return false;
    }

    // All slots full — blocked unless corp has a token here
    !city
        .tokens
        .iter()
        .any(|t| t.as_ref().is_some_and(|tok| tok.corporation_id == corp_sym))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entities::Token;
    use crate::graph::{City, Hex, Tile, Town};
    use crate::tiles::{parse_tile, TileColor};

    /// Helper: build a Tile from a DSL string, placing it on a hex.
    fn tile_from_dsl(name: &str, dsl: &str, color: TileColor, rotation: u8) -> Tile {
        let tile_def = parse_tile(name, dsl, color);
        let rotated = tile_def.rotated(rotation);
        let mut tile = Tile::new(name.to_string(), name.to_string());
        tile.rotation = rotation;
        tile.color = rotated.color;
        tile.label = rotated.label.clone();
        tile.paths = rotated.paths;
        for cd in &rotated.cities {
            tile.cities.push(City::new(cd.revenue, cd.slots));
        }
        for td in &rotated.towns {
            tile.towns.push(Town::new(td.revenue));
        }
        tile
    }

    /// Place a token in a city slot.
    fn place_token(tile: &mut Tile, city_index: usize, corp_sym: &str) {
        let city = &mut tile.cities[city_index];
        let slot = city.tokens.iter().position(|t| t.is_none()).unwrap();
        let mut tok = Token::new(corp_sym.to_string(), 0);
        tok.used = true;
        city.tokens[slot] = Some(tok);
    }

    /// Build a simple linear map:
    ///   A -- B -- C
    /// Each hex connected via edge 1 (right) / edge 4 (left).
    /// A has a city with PRR token; B is a straight track; C has a city.
    fn build_linear_3hex() -> (
        Vec<Hex>,
        HashMap<String, usize>,
        HashMap<String, HashMap<u8, String>>,
    ) {
        // Hex A: city tile 57 at rotation 2 → edges 2,5 (city connects edge 2 and 5)
        // Actually let's keep it simple: city with paths to edge 1
        let mut tile_a = tile_from_dsl(
            "57",
            "city=revenue:20;path=a:1,b:_0;path=a:_0,b:4",
            TileColor::Yellow,
            0,
        );
        place_token(&mut tile_a, 0, "PRR");
        let hex_a = Hex::new("A".to_string(), tile_a);

        // Hex B: straight track edge 4 to edge 1 (tile 9 at rotation 1)
        let tile_b = tile_from_dsl("9", "path=a:4,b:1", TileColor::Yellow, 0);
        let hex_b = Hex::new("B".to_string(), tile_b);

        // Hex C: city tile 57 with edges 4,1
        let tile_c = tile_from_dsl(
            "57",
            "city=revenue:30;path=a:4,b:_0;path=a:_0,b:1",
            TileColor::Yellow,
            0,
        );
        let hex_c = Hex::new("C".to_string(), tile_c);

        let hexes = vec![hex_a, hex_b, hex_c];
        let hex_idx: HashMap<String, usize> = [
            ("A".to_string(), 0),
            ("B".to_string(), 1),
            ("C".to_string(), 2),
        ]
        .into();

        // A edge 1 → B, B edge 4 → A, B edge 1 → C, C edge 4 → B
        let adjacency: HashMap<String, HashMap<u8, String>> = [
            ("A".to_string(), [(1u8, "B".to_string())].into()),
            (
                "B".to_string(),
                [(4u8, "A".to_string()), (1u8, "C".to_string())].into(),
            ),
            ("C".to_string(), [(4u8, "B".to_string())].into()),
        ]
        .into();

        (hexes, hex_idx, adjacency)
    }

    #[test]
    fn linear_3hex_prr_reaches_city_c() {
        let (hexes, hex_idx, adjacency) = build_linear_3hex();
        let token_positions = vec![("A".to_string(), 0usize)];

        let mut cache = GraphCache::new();
        let graph =
            cache.get_or_compute("PRR", &hexes, &hex_idx, &adjacency, &token_positions, &[]);

        // PRR should reach all 3 hexes
        assert!(
            graph.connected_hexes.contains_key("A"),
            "A should be reachable"
        );
        assert!(
            graph.connected_hexes.contains_key("B"),
            "B should be reachable"
        );
        assert!(
            graph.connected_hexes.contains_key("C"),
            "C should be reachable"
        );

        // PRR should see city in C as a connected node
        let c_city = NodeId {
            hex_id: "C".to_string(),
            node_type: NodeType::City,
            index: 0,
        };
        assert!(
            graph.connected_nodes.contains(&c_city),
            "City C should be a connected node"
        );
    }

    #[test]
    fn linear_3hex_tokenable_city_c() {
        let (hexes, hex_idx, adjacency) = build_linear_3hex();
        let token_positions = vec![("A".to_string(), 0usize)];

        let mut cache = GraphCache::new();
        let graph =
            cache.get_or_compute("PRR", &hexes, &hex_idx, &adjacency, &token_positions, &[]);

        // C's city has an empty slot and PRR can reach it
        assert_eq!(graph.tokenable_cities.len(), 1);
        assert_eq!(graph.tokenable_cities[0].hex_id, "C");
        assert_eq!(graph.tokenable_cities[0].city_index, 0);
    }

    #[test]
    fn no_tokens_means_no_connectivity() {
        let (hexes, hex_idx, adjacency) = build_linear_3hex();
        let token_positions: Vec<(String, usize)> = vec![]; // no tokens placed

        let mut cache = GraphCache::new();
        let graph =
            cache.get_or_compute("NYC", &hexes, &hex_idx, &adjacency, &token_positions, &[]);

        assert!(graph.connected_hexes.is_empty());
        assert!(graph.connected_nodes.is_empty());
        assert!(graph.tokenable_cities.is_empty());
    }

    /// Build a Y-shaped map where A (city+token) connects to B and C (both cities),
    /// but C has a blocked city (all slots taken by NYC).
    fn build_y_with_blocked_city() -> (
        Vec<Hex>,
        HashMap<String, usize>,
        HashMap<String, HashMap<u8, String>>,
    ) {
        // A: city with PRR token, exits to edge 0 and edge 2
        let mut tile_a = tile_from_dsl(
            "14",
            "city=revenue:30,slots:2;path=a:0,b:_0;path=a:2,b:_0",
            TileColor::Green,
            0,
        );
        place_token(&mut tile_a, 0, "PRR");

        // B: city with empty slot, enters from edge 3
        let tile_b = tile_from_dsl(
            "57",
            "city=revenue:20;path=a:3,b:_0;path=a:_0,b:0",
            TileColor::Yellow,
            0,
        );

        // C: city with 1 slot, fully blocked by NYC
        let mut tile_c = tile_from_dsl(
            "57",
            "city=revenue:40;path=a:5,b:_0;path=a:_0,b:2",
            TileColor::Yellow,
            0,
        );
        place_token(&mut tile_c, 0, "NYC");

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

        // A edge 0 → B (B enters from edge 3), A edge 2 → C (C enters from edge 5)
        let adjacency: HashMap<String, HashMap<u8, String>> = [
            (
                "A".to_string(),
                [(0u8, "B".to_string()), (2u8, "C".to_string())].into(),
            ),
            ("B".to_string(), [(3u8, "A".to_string())].into()),
            ("C".to_string(), [(5u8, "A".to_string())].into()),
        ]
        .into();

        (hexes, hex_idx, adjacency)
    }

    #[test]
    fn blocked_city_is_not_traversable() {
        let (hexes, hex_idx, adjacency) = build_y_with_blocked_city();
        let token_positions = vec![("A".to_string(), 0usize)];

        let mut cache = GraphCache::new();
        let graph =
            cache.get_or_compute("PRR", &hexes, &hex_idx, &adjacency, &token_positions, &[]);

        // B should be reachable (open city)
        let b_city = NodeId {
            hex_id: "B".to_string(),
            node_type: NodeType::City,
            index: 0,
        };
        assert!(
            graph.connected_nodes.contains(&b_city),
            "B city should be reachable"
        );

        // C IS reachable as a destination (route can end at a blocked city),
        // but the walk should NOT traverse THROUGH it.
        let c_city = NodeId {
            hex_id: "C".to_string(),
            node_type: NodeType::City,
            index: 0,
        };
        assert!(
            graph.connected_nodes.contains(&c_city),
            "C city should be reachable (route can run TO a blocked city)"
        );

        // Only B should be tokenable (C is full, no open slots)
        assert_eq!(graph.tokenable_cities.len(), 1);
        assert_eq!(graph.tokenable_cities[0].hex_id, "B");
    }

    #[test]
    fn own_token_city_not_blocked() {
        // If PRR has a token in C, it should NOT be blocked for PRR
        let (mut hexes, hex_idx, adjacency) = build_y_with_blocked_city();
        // Replace C's token with PRR (city has 1 slot)
        hexes[2].tile.cities[0].tokens[0] = Some({
            let mut t = Token::new("PRR".to_string(), 0);
            t.used = true;
            t
        });

        let token_positions = vec![("A".to_string(), 0), ("C".to_string(), 0)];

        let mut cache = GraphCache::new();
        let graph =
            cache.get_or_compute("PRR", &hexes, &hex_idx, &adjacency, &token_positions, &[]);

        let c_city = NodeId {
            hex_id: "C".to_string(),
            node_type: NodeType::City,
            index: 0,
        };
        assert!(
            graph.connected_nodes.contains(&c_city),
            "PRR's own city should not be blocked"
        );
    }

    #[test]
    fn city_with_empty_slot_not_blocked() {
        // A 2-slot city with one NYC token has an empty slot → not blocked
        let (mut hexes, hex_idx, adjacency) = build_y_with_blocked_city();
        // Give C a 2-slot city with only 1 NYC token
        hexes[2].tile.cities[0].slots = 2;
        hexes[2].tile.cities[0].tokens.push(None); // add second empty slot

        let token_positions = vec![("A".to_string(), 0usize)];

        let mut cache = GraphCache::new();
        let graph =
            cache.get_or_compute("PRR", &hexes, &hex_idx, &adjacency, &token_positions, &[]);

        let c_city = NodeId {
            hex_id: "C".to_string(),
            node_type: NodeType::City,
            index: 0,
        };
        assert!(
            graph.connected_nodes.contains(&c_city),
            "2-slot city with 1 token should not be blocked"
        );
        // C should be tokenable (has empty slot, PRR not already there)
        assert!(
            graph.tokenable_cities.iter().any(|tc| tc.hex_id == "C"),
            "C should be tokenable"
        );
    }

    #[test]
    fn town_traversal_works() {
        // A (city+token) → B (town) → C (city)
        let mut tile_a = tile_from_dsl("57", "city=revenue:20;path=a:1,b:_0", TileColor::Yellow, 0);
        place_token(&mut tile_a, 0, "PRR");

        let tile_b = tile_from_dsl(
            "3",
            "town=revenue:10;path=a:4,b:_0;path=a:_0,b:1",
            TileColor::Yellow,
            0,
        );

        let tile_c = tile_from_dsl("57", "city=revenue:30;path=a:4,b:_0", TileColor::Yellow, 0);

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

        let token_positions = vec![("A".to_string(), 0usize)];
        let mut cache = GraphCache::new();
        let graph =
            cache.get_or_compute("PRR", &hexes, &hex_idx, &adjacency, &token_positions, &[]);

        // Should reach the town in B
        let b_town = NodeId {
            hex_id: "B".to_string(),
            node_type: NodeType::Town,
            index: 0,
        };
        assert!(graph.connected_nodes.contains(&b_town));

        // Should reach the city in C (through the town)
        let c_city = NodeId {
            hex_id: "C".to_string(),
            node_type: NodeType::City,
            index: 0,
        };
        assert!(graph.connected_nodes.contains(&c_city));
    }

    #[test]
    fn disconnected_hex_not_reachable() {
        // A (city+token, exits edge 1) and B (city, exits edge 0)
        // But they're connected A.1→B.4, and B has no path on edge 4 → disconnected
        let mut tile_a = tile_from_dsl("57", "city=revenue:20;path=a:1,b:_0", TileColor::Yellow, 0);
        place_token(&mut tile_a, 0, "PRR");

        // B has path only on edge 0 (not matching the entry from edge 4)
        let tile_b = tile_from_dsl("57", "city=revenue:30;path=a:0,b:_0", TileColor::Yellow, 0);

        let hexes = vec![
            Hex::new("A".to_string(), tile_a),
            Hex::new("B".to_string(), tile_b),
        ];
        let hex_idx: HashMap<String, usize> = [("A".to_string(), 0), ("B".to_string(), 1)].into();
        let adjacency: HashMap<String, HashMap<u8, String>> = [
            ("A".to_string(), [(1u8, "B".to_string())].into()),
            ("B".to_string(), [(4u8, "A".to_string())].into()),
        ]
        .into();

        let token_positions = vec![("A".to_string(), 0usize)];
        let mut cache = GraphCache::new();
        let graph =
            cache.get_or_compute("PRR", &hexes, &hex_idx, &adjacency, &token_positions, &[]);

        // B should NOT be reachable (no matching path on entry edge 4)
        let b_city = NodeId {
            hex_id: "B".to_string(),
            node_type: NodeType::City,
            index: 0,
        };
        assert!(
            !graph.connected_nodes.contains(&b_city),
            "B should not be reachable — no path on entry edge"
        );
    }

    #[test]
    fn cache_clears_properly() {
        let (hexes, hex_idx, adjacency) = build_linear_3hex();
        let token_positions = vec![("A".to_string(), 0usize)];

        let mut cache = GraphCache::new();
        let _ = cache.get_or_compute("PRR", &hexes, &hex_idx, &adjacency, &token_positions, &[]);
        assert!(!cache.cache.is_empty());

        cache.clear();
        assert!(cache.cache.is_empty());
    }

    #[test]
    fn offboard_reachable_from_edge() {
        // A (city+token, exit edge 1) → B (offboard, entry from edge 4)
        let mut tile_a = tile_from_dsl("57", "city=revenue:20;path=a:1,b:_0", TileColor::Yellow, 0);
        place_token(&mut tile_a, 0, "PRR");

        let tile_b = tile_from_dsl(
            "offboard",
            "offboard=revenue:yellow_40|brown_70;path=a:4,b:_0",
            TileColor::Red,
            0,
        );

        let hexes = vec![
            Hex::new("A".to_string(), tile_a),
            Hex::new("B".to_string(), tile_b),
        ];
        let hex_idx: HashMap<String, usize> = [("A".to_string(), 0), ("B".to_string(), 1)].into();
        let adjacency: HashMap<String, HashMap<u8, String>> = [
            ("A".to_string(), [(1u8, "B".to_string())].into()),
            ("B".to_string(), [(4u8, "A".to_string())].into()),
        ]
        .into();

        let token_positions = vec![("A".to_string(), 0usize)];
        let mut cache = GraphCache::new();
        let graph =
            cache.get_or_compute("PRR", &hexes, &hex_idx, &adjacency, &token_positions, &[]);

        let b_offboard = NodeId {
            hex_id: "B".to_string(),
            node_type: NodeType::Offboard,
            index: 0,
        };
        assert!(
            graph.connected_nodes.contains(&b_offboard),
            "Offboard should be reachable"
        );
    }

    #[test]
    fn already_tokened_hex_not_tokenable() {
        // PRR has token in A. A connects to B. B should not be tokenable if PRR already there.
        // Actually the rule is one token per hex per corp, not that own token makes it untokenable.
        // Let's test: PRR has token in A city 0, A also has city 1 → NOT tokenable (same hex).
        let mut tile_a = tile_from_dsl(
            "59",
            "city=revenue:40;city=revenue:40;path=a:0,b:_0;path=a:2,b:_1",
            TileColor::Green,
            0,
        );
        place_token(&mut tile_a, 0, "PRR");

        let hexes = vec![Hex::new("A".to_string(), tile_a)];
        let hex_idx: HashMap<String, usize> = [("A".to_string(), 0)].into();
        let adjacency: HashMap<String, HashMap<u8, String>> =
            [("A".to_string(), HashMap::new())].into();

        let token_positions = vec![("A".to_string(), 0usize)];
        let mut cache = GraphCache::new();
        let graph =
            cache.get_or_compute("PRR", &hexes, &hex_idx, &adjacency, &token_positions, &[]);

        // City 1 in the same hex should NOT be tokenable (1830: one token per hex per corp)
        assert!(
            graph.tokenable_cities.is_empty(),
            "Same-hex city should not be tokenable"
        );
    }
}
