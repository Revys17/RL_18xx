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
    /// Hex chains between consecutive stops (for Python Route construction).
    pub connections: Vec<Vec<String>>,
}

// ---------------------------------------------------------------------------
// Route finder (hexside bit assignment)
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
        let key1 = (hex_id.to_string(), edge);
        if let Some(&bit) = self.hexside_bits.get(&key1) {
            return bit;
        }

        if let Some(neighbor_id) = hex_adjacency.get(hex_id).and_then(|n| n.get(&edge)) {
            let opposite = (edge + 3) % 6;
            let key2 = (neighbor_id.clone(), opposite);
            if let Some(&bit) = self.hexside_bits.get(&key2) {
                self.hexside_bits.insert(key1, bit);
                return bit;
            }
        }

        let bit = self.assign_bit(hex_id, edge);

        if let Some(neighbor_id) = hex_adjacency.get(hex_id).and_then(|n| n.get(&edge)) {
            let opposite = (edge + 3) % 6;
            let key2 = (neighbor_id.clone(), opposite);
            self.hexside_bits.entry(key2).or_insert(bit);
        }

        bit
    }
}

// ---------------------------------------------------------------------------
// Path-level walk state
// ---------------------------------------------------------------------------

/// A single tile path segment — the atomic unit of route traversal.
#[derive(Clone, Hash, Eq, PartialEq, Debug)]
struct PathSegment {
    hex_id: String,
    path_index: usize,
}

/// Accumulated state during a path-level walk.
struct WalkState<'a> {
    hexes: &'a [Hex],
    hex_idx: &'a HashMap<String, usize>,
    hex_adjacency: &'a HashMap<String, HashMap<u8, String>>,
    phase_tiles: &'a [String],
    corp_sym: &'a str,
    finder: &'a mut RouteFinder,

    // Walk tracking
    visited_paths: HashSet<PathSegment>,
    visited_nodes: HashSet<NodeId>,
    edge_counter: HashMap<(String, u8), u32>,
    hexside_bits: u128,

    // Route collection
    /// Current ordered list of stops (revenue nodes) in the walk.
    stops: Vec<NodeId>,
    /// Current ordered list of hex IDs traversed between stops.
    /// Each entry corresponds to a connection between stops[i] and stops[i+1].
    current_chain: Vec<String>,
    /// Completed chains between consecutive stops.
    chains: Vec<Vec<String>>,
    /// Accumulated revenue.
    revenue: i32,

    // Train constraints
    train_distance: u32,
    is_d_train: bool,
    /// Number of stops that count toward train distance.
    stop_count: u32,

    // Results
    candidates: Vec<RouteCandidate>,
    token_set: HashSet<NodeId>,
}

impl<'a> WalkState<'a> {
    /// Emit the current walk state as a route candidate if valid.
    fn emit_candidate(&mut self) {
        if self.stops.len() < 2 {
            return;
        }
        // Must include at least one tokened city
        if !self.stops.iter().any(|n| self.token_set.contains(n)) {
            return;
        }

        // Build connections: chains[0..n-1] are complete, current_chain is the
        // in-progress chain for the last stop pair (if stops > chains + 1).
        let mut connections = self.chains.clone();
        if !self.current_chain.is_empty() {
            connections.push(self.current_chain.clone());
        }

        // Fix up chains: ensure each chain starts with its source stop's hex
        // and ends with its destination stop's hex. The walk's backtracking
        // can sometimes lose the first/last hex due to push/pop ordering.
        for (ci, chain) in connections.iter_mut().enumerate() {
            if chain.is_empty() {
                continue;
            }
            // Source stop hex (stops[ci] is the stop BEFORE this chain)
            let src_hex = &self.stops[ci].hex_id;
            if chain.first().map(|s| s.as_str()) != Some(src_hex) {
                chain.insert(0, src_hex.clone());
            }
            // Destination stop hex (stops[ci+1] is the stop AFTER this chain)
            if ci + 1 < self.stops.len() {
                let dst_hex = &self.stops[ci + 1].hex_id;
                if chain.last().map(|s| s.as_str()) != Some(dst_hex) {
                    chain.push(dst_hex.clone());
                }
            }
        }

        self.candidates.push(RouteCandidate {
            nodes: self.stops.clone(),
            revenue: self.revenue,
            hexside_bits: self.hexside_bits,
            connections,
        });
    }
}

// ---------------------------------------------------------------------------
// Revenue helpers
// ---------------------------------------------------------------------------

fn node_revenue(
    hexes: &[Hex],
    hex_idx: &HashMap<String, usize>,
    node: &NodeId,
    phase_tiles: &[String],
) -> i32 {
    let hi = match hex_idx.get(&node.hex_id) {
        Some(&i) => i,
        None => return 0,
    };
    let tile = &hexes[hi].tile;
    match node.node_type {
        NodeType::City => tile.cities.get(node.index).map_or(0, |c| c.revenue),
        NodeType::Town => tile.towns.get(node.index).map_or(0, |t| t.revenue),
        NodeType::Offboard => tile
            .offboards
            .get(node.index)
            .map_or(0, |o| o.phase_revenue(phase_tiles)),
    }
}

/// Check if a city blocks route traversal: all token slots filled by other corps.
fn is_city_blocked_for_route(
    hexes: &[Hex],
    hex_idx: &HashMap<String, usize>,
    node: &NodeId,
    corp_sym: &str,
) -> bool {
    if node.node_type != NodeType::City {
        return false;
    }
    let hi = match hex_idx.get(&node.hex_id) {
        Some(&i) => i,
        None => return false,
    };
    let city = match hexes[hi].tile.cities.get(node.index) {
        Some(c) => c,
        None => return false,
    };
    if city.tokens.iter().any(|t| t.is_none()) {
        return false;
    }
    !city
        .tokens
        .iter()
        .any(|t| t.as_ref().is_some_and(|tok| tok.corporation_id == corp_sym))
}

/// Check if a hex has any edge↔edge bypass path.
fn has_bypass_path(hexes: &[Hex], hex_idx: &HashMap<String, usize>, node: &NodeId) -> bool {
    let hi = match hex_idx.get(&node.hex_id) {
        Some(&i) => i,
        None => return false,
    };
    let tile = &hexes[hi].tile;
    tile.paths
        .iter()
        .any(|p| matches!((&p.a, &p.b), (PathEndpoint::Edge(_), PathEndpoint::Edge(_))))
}

// ---------------------------------------------------------------------------
// Path-level walk algorithm
// ---------------------------------------------------------------------------

/// Walk outward from a revenue node (city/town/offboard), exploring all paths
/// that connect to this node. Each path step is a concrete tile PathDef.
fn walk_from_node(state: &mut WalkState<'_>, hex_id: &str, node: &NodeId) {
    if state.visited_nodes.contains(node) {
        return;
    }
    state.visited_nodes.insert(node.clone());

    let hi = match state.hex_idx.get(hex_id) {
        Some(&i) => i,
        None => {
            state.visited_nodes.remove(node);
            return;
        }
    };
    let tile = &state.hexes[hi].tile;

    let node_ep = match node.node_type {
        NodeType::City => PathEndpoint::City(node.index),
        NodeType::Town => PathEndpoint::Town(node.index),
        NodeType::Offboard => PathEndpoint::Offboard(node.index),
    };

    // Collect path indices to avoid borrow issues
    let path_indices: Vec<(usize, PathEndpoint)> = tile
        .paths
        .iter()
        .enumerate()
        .filter(|(_, p)| !p.terminal)
        .filter_map(|(idx, p)| {
            if p.a == node_ep {
                Some((idx, p.b.clone()))
            } else if p.b == node_ep {
                Some((idx, p.a.clone()))
            } else {
                None
            }
        })
        .collect();

    for (path_idx, exit) in path_indices {
        let seg = PathSegment {
            hex_id: hex_id.to_string(),
            path_index: path_idx,
        };
        if state.visited_paths.contains(&seg) {
            continue;
        }

        // Check edge counter for edges in this path
        if let Some(e) = exit.edge_num() {
            let cnt = *state
                .edge_counter
                .get(&(hex_id.to_string(), e))
                .unwrap_or(&0);
            if cnt > 0 {
                continue;
            }
        }

        state.visited_paths.insert(seg.clone());

        match &exit {
            PathEndpoint::Edge(exit_edge) => {
                let bit = state.finder.assign_hexside_pair(
                    hex_id,
                    *exit_edge,
                    state.hex_adjacency,
                );
                state.hexside_bits |= bit;

                // Add hex to current chain (save length for reliable restore)
                let chain_len = state.current_chain.len();
                if state.current_chain.is_empty()
                    || state.current_chain.last().map(|s| s.as_str()) != Some(hex_id)
                {
                    state.current_chain.push(hex_id.to_string());
                }

                // Follow to neighbor
                let neighbor_id = state
                    .hex_adjacency
                    .get(hex_id)
                    .and_then(|n| n.get(exit_edge))
                    .cloned();

                if let Some(neighbor_id) = neighbor_id {
                    let enter_edge = (*exit_edge + 3) % 6;
                    let edge_key = (hex_id.to_string(), *exit_edge);
                    *state.edge_counter.entry(edge_key.clone()).or_insert(0) += 1;

                    walk_into_path(state, &neighbor_id, enter_edge);

                    *state.edge_counter.entry(edge_key).or_insert(0) -= 1;
                }

                state.hexside_bits &= !bit;
                state.current_chain.truncate(chain_len);
            }
            PathEndpoint::City(idx) => {
                let dest = NodeId {
                    hex_id: hex_id.to_string(),
                    node_type: NodeType::City,
                    index: *idx,
                };
                let blocked = is_city_blocked_for_route(
                    state.hexes,
                    state.hex_idx,
                    &dest,
                    state.corp_sym,
                );
                if blocked {
                    let has_bypass =
                        has_bypass_path(state.hexes, state.hex_idx, &dest);
                    if !has_bypass {
                        // Terminal stop at blocked city (no bypass)
                        let rev =
                            node_revenue(state.hexes, state.hex_idx, &dest, state.phase_tiles);
                        let cost = 1u32;
                        if state.stop_count + cost <= state.train_distance {
                            let chain_len = state.current_chain.len();
                            if state.current_chain.is_empty()
                                || state.current_chain.last().map(|s| s.as_str())
                                    != Some(hex_id)
                            {
                                state.current_chain.push(hex_id.to_string());
                            }
                            let chain = state.current_chain.clone();

                            state.stops.push(dest.clone());
                            state.chains.push(chain.clone());
                            let saved_chain =
                                std::mem::replace(&mut state.current_chain, Vec::new());
                            state.revenue += rev;
                            state.stop_count += cost;

                            state.emit_candidate();

                            state.stop_count -= cost;
                            state.revenue -= rev;
                            state.current_chain = saved_chain;
                            state.chains.pop();
                            state.stops.pop();

                            state.current_chain.truncate(chain_len);
                        }
                    }
                    // If bypass exists, skip — bypass paths are separate path segments
                } else {
                    // Unblocked intra-tile city: add as stop and continue walk
                    arrive_at_node(state, hex_id, &dest);
                }
            }
            PathEndpoint::Town(idx) => {
                let dest = NodeId {
                    hex_id: hex_id.to_string(),
                    node_type: NodeType::Town,
                    index: *idx,
                };
                arrive_at_node(state, hex_id, &dest);
            }
            PathEndpoint::Offboard(idx) => {
                let dest = NodeId {
                    hex_id: hex_id.to_string(),
                    node_type: NodeType::Offboard,
                    index: *idx,
                };
                arrive_at_node(state, hex_id, &dest);
            }
            PathEndpoint::Junction => {
                // Follow through junction to all other junction paths on this tile
                let tile = &state.hexes[hi].tile;
                let junction_paths: Vec<(usize, PathEndpoint)> = tile
                    .paths
                    .iter()
                    .enumerate()
                    .filter(|(idx, p)| {
                        *idx != path_idx
                            && !p.terminal
                            && (p.a == PathEndpoint::Junction || p.b == PathEndpoint::Junction)
                    })
                    .map(|(idx, p)| {
                        let exit = if p.a == PathEndpoint::Junction {
                            p.b.clone()
                        } else {
                            p.a.clone()
                        };
                        (idx, exit)
                    })
                    .collect();

                for (other_idx, other_exit) in junction_paths {
                    let other_seg = PathSegment {
                        hex_id: hex_id.to_string(),
                        path_index: other_idx,
                    };
                    if state.visited_paths.contains(&other_seg) {
                        continue;
                    }

                    match &other_exit {
                        PathEndpoint::Edge(e) => {
                            let edge_cnt = *state
                                .edge_counter
                                .get(&(hex_id.to_string(), *e))
                                .unwrap_or(&0);
                            if edge_cnt > 0 {
                                continue;
                            }

                            state.visited_paths.insert(other_seg.clone());
                            let bit = state.finder.assign_hexside_pair(
                                hex_id,
                                *e,
                                state.hex_adjacency,
                            );
                            state.hexside_bits |= bit;

                            let chain_len = state.current_chain.len();
                            if state.current_chain.is_empty()
                                || state.current_chain.last().map(|s| s.as_str()) != Some(hex_id)
                            {
                                state.current_chain.push(hex_id.to_string());
                            }

                            let neighbor_id = state
                                .hex_adjacency
                                .get(hex_id)
                                .and_then(|n| n.get(e))
                                .cloned();

                            if let Some(neighbor_id) = neighbor_id {
                                let enter_edge = (*e + 3) % 6;
                                let edge_key = (hex_id.to_string(), *e);
                                *state.edge_counter.entry(edge_key.clone()).or_insert(0) += 1;

                                walk_into_path(state, &neighbor_id, enter_edge);

                                *state.edge_counter.entry(edge_key).or_insert(0) -= 1;
                            }

                            state.hexside_bits &= !bit;
                            state.current_chain.truncate(chain_len);
                            state.visited_paths.remove(&other_seg);
                        }
                        PathEndpoint::City(idx) => {
                            state.visited_paths.insert(other_seg.clone());
                            let dest = NodeId {
                                hex_id: hex_id.to_string(),
                                node_type: NodeType::City,
                                index: *idx,
                            };
                            let blocked = is_city_blocked_for_route(
                                state.hexes, state.hex_idx, &dest, state.corp_sym,
                            );
                            if !blocked {
                                arrive_at_node(state, hex_id, &dest);
                            }
                            state.visited_paths.remove(&other_seg);
                        }
                        PathEndpoint::Town(idx) => {
                            state.visited_paths.insert(other_seg.clone());
                            let dest = NodeId {
                                hex_id: hex_id.to_string(),
                                node_type: NodeType::Town,
                                index: *idx,
                            };
                            arrive_at_node(state, hex_id, &dest);
                            state.visited_paths.remove(&other_seg);
                        }
                        PathEndpoint::Offboard(idx) => {
                            state.visited_paths.insert(other_seg.clone());
                            let dest = NodeId {
                                hex_id: hex_id.to_string(),
                                node_type: NodeType::Offboard,
                                index: *idx,
                            };
                            arrive_at_node(state, hex_id, &dest);
                            state.visited_paths.remove(&other_seg);
                        }
                        PathEndpoint::Junction => {
                            // Nested junctions — skip to avoid infinite recursion
                        }
                    }
                }
            }
        }

        state.visited_paths.remove(&seg);
    }

    state.visited_nodes.remove(node);
}

/// Walk into a hex from an entry edge, following the matching path.
/// This handles pass-through hexes (edge→edge) and arrival at revenue nodes.
fn walk_into_path(state: &mut WalkState<'_>, hex_id: &str, enter_edge: u8) {
    let hi = match state.hex_idx.get(hex_id) {
        Some(&i) => i,
        None => return,
    };
    let tile = &state.hexes[hi].tile;
    let enter_ep = PathEndpoint::Edge(enter_edge);

    // Collect matching paths
    let matching: Vec<(usize, PathEndpoint)> = tile
        .paths
        .iter()
        .enumerate()
        .filter(|(_, p)| !p.terminal)
        .filter_map(|(idx, p)| {
            if p.a == enter_ep {
                Some((idx, p.b.clone()))
            } else if p.b == enter_ep {
                Some((idx, p.a.clone()))
            } else {
                None
            }
        })
        .collect();

    for (path_idx, dest) in matching {
        let seg = PathSegment {
            hex_id: hex_id.to_string(),
            path_index: path_idx,
        };
        if state.visited_paths.contains(&seg) {
            continue;
        }

        // Check edge counter for the exit edge of this path
        if let Some(e) = dest.edge_num() {
            let cnt = *state
                .edge_counter
                .get(&(hex_id.to_string(), e))
                .unwrap_or(&0);
            if cnt > 0 {
                continue;
            }
        }

        state.visited_paths.insert(seg.clone());

        // Add hex to current chain. Save length for reliable restore —
        // nested arrive_at_node backtracking can corrupt the chain if we
        // rely on positional pop().
        let chain_len = state.current_chain.len();
        state.current_chain.push(hex_id.to_string());

        match &dest {
            PathEndpoint::Edge(exit_edge) => {
                // Pass-through hex: enter one edge, exit another
                let bit = state.finder.assign_hexside_pair(
                    hex_id,
                    *exit_edge,
                    state.hex_adjacency,
                );
                state.hexside_bits |= bit;

                let neighbor_id = state
                    .hex_adjacency
                    .get(hex_id)
                    .and_then(|n| n.get(exit_edge))
                    .cloned();

                if let Some(neighbor_id) = neighbor_id {
                    let next_enter = (*exit_edge + 3) % 6;
                    let edge_key = (hex_id.to_string(), *exit_edge);
                    *state.edge_counter.entry(edge_key.clone()).or_insert(0) += 1;

                    walk_into_path(state, &neighbor_id, next_enter);

                    *state.edge_counter.entry(edge_key).or_insert(0) -= 1;
                }

                state.hexside_bits &= !bit;
            }
            PathEndpoint::City(idx) => {
                let dest_node = NodeId {
                    hex_id: hex_id.to_string(),
                    node_type: NodeType::City,
                    index: *idx,
                };
                let blocked = is_city_blocked_for_route(
                    state.hexes,
                    state.hex_idx,
                    &dest_node,
                    state.corp_sym,
                );
                if blocked {
                    let has_bypass =
                        has_bypass_path(state.hexes, state.hex_idx, &dest_node);
                    if !has_bypass {
                        // Terminal stop at blocked city with no bypass
                        let rev = node_revenue(
                            state.hexes,
                            state.hex_idx,
                            &dest_node,
                            state.phase_tiles,
                        );
                        let cost = 1u32;
                        if state.stop_count + cost <= state.train_distance {
                            let chain = state.current_chain.clone();
                            state.stops.push(dest_node.clone());
                            state.chains.push(chain);
                            let saved_chain =
                                std::mem::replace(&mut state.current_chain, Vec::new());
                            state.revenue += rev;
                            state.stop_count += cost;

                            state.emit_candidate();

                            state.stop_count -= cost;
                            state.revenue -= rev;
                            state.current_chain = saved_chain;
                            state.chains.pop();
                            state.stops.pop();
                        }
                    }
                    // With bypass: the bypass edge→edge paths will be separate entries
                } else {
                    arrive_at_node(state, hex_id, &dest_node);
                }
            }
            PathEndpoint::Town(idx) => {
                let dest_node = NodeId {
                    hex_id: hex_id.to_string(),
                    node_type: NodeType::Town,
                    index: *idx,
                };
                arrive_at_node(state, hex_id, &dest_node);
            }
            PathEndpoint::Offboard(idx) => {
                let dest_node = NodeId {
                    hex_id: hex_id.to_string(),
                    node_type: NodeType::Offboard,
                    index: *idx,
                };
                arrive_at_node(state, hex_id, &dest_node);
            }
            PathEndpoint::Junction => {
                // Follow through junction
                let tile = &state.hexes[hi].tile;
                let junction_paths: Vec<(usize, PathEndpoint)> = tile
                    .paths
                    .iter()
                    .enumerate()
                    .filter(|(idx, p)| {
                        *idx != path_idx
                            && !p.terminal
                            && (p.a == PathEndpoint::Junction || p.b == PathEndpoint::Junction)
                    })
                    .map(|(idx, p)| {
                        let exit = if p.a == PathEndpoint::Junction {
                            p.b.clone()
                        } else {
                            p.a.clone()
                        };
                        (idx, exit)
                    })
                    .collect();

                for (other_idx, other_exit) in junction_paths {
                    let other_seg = PathSegment {
                        hex_id: hex_id.to_string(),
                        path_index: other_idx,
                    };
                    if state.visited_paths.contains(&other_seg) {
                        continue;
                    }

                    match &other_exit {
                        PathEndpoint::Edge(e) => {
                            let edge_cnt = *state
                                .edge_counter
                                .get(&(hex_id.to_string(), *e))
                                .unwrap_or(&0);
                            if edge_cnt > 0 {
                                continue;
                            }

                            state.visited_paths.insert(other_seg.clone());
                            let bit = state.finder.assign_hexside_pair(
                                hex_id,
                                *e,
                                state.hex_adjacency,
                            );
                            state.hexside_bits |= bit;

                            let neighbor_id = state
                                .hex_adjacency
                                .get(hex_id)
                                .and_then(|n| n.get(e))
                                .cloned();

                            if let Some(neighbor_id) = neighbor_id {
                                let next_enter = (*e + 3) % 6;
                                let edge_key = (hex_id.to_string(), *e);
                                *state.edge_counter.entry(edge_key.clone()).or_insert(0) += 1;

                                walk_into_path(state, &neighbor_id, next_enter);

                                *state.edge_counter.entry(edge_key).or_insert(0) -= 1;
                            }

                            state.hexside_bits &= !bit;
                            state.visited_paths.remove(&other_seg);
                        }
                        PathEndpoint::City(idx) => {
                            state.visited_paths.insert(other_seg.clone());
                            let dest = NodeId {
                                hex_id: hex_id.to_string(),
                                node_type: NodeType::City,
                                index: *idx,
                            };
                            let blocked = is_city_blocked_for_route(
                                state.hexes, state.hex_idx, &dest, state.corp_sym,
                            );
                            if !blocked {
                                arrive_at_node(state, hex_id, &dest);
                            }
                            state.visited_paths.remove(&other_seg);
                        }
                        PathEndpoint::Town(idx) => {
                            state.visited_paths.insert(other_seg.clone());
                            let dest = NodeId {
                                hex_id: hex_id.to_string(),
                                node_type: NodeType::Town,
                                index: *idx,
                            };
                            arrive_at_node(state, hex_id, &dest);
                            state.visited_paths.remove(&other_seg);
                        }
                        PathEndpoint::Offboard(idx) => {
                            state.visited_paths.insert(other_seg.clone());
                            let dest = NodeId {
                                hex_id: hex_id.to_string(),
                                node_type: NodeType::Offboard,
                                index: *idx,
                            };
                            arrive_at_node(state, hex_id, &dest);
                            state.visited_paths.remove(&other_seg);
                        }
                        PathEndpoint::Junction => {
                            // Nested junctions — skip
                        }
                    }
                }
            }
        }

        // Restore chain to pre-push state (reliable even after nested backtracking)
        state.current_chain.truncate(chain_len);
        state.visited_paths.remove(&seg);
    }
}

/// Arrive at a revenue node: add as stop, emit candidate, continue walking.
fn arrive_at_node(state: &mut WalkState<'_>, hex_id: &str, node: &NodeId) {
    if state.visited_nodes.contains(node) {
        return;
    }

    let is_town = node.node_type == NodeType::Town;
    let stop_cost = if state.is_d_train && is_town {
        0u32
    } else {
        1u32
    };

    if state.stop_count + stop_cost > state.train_distance {
        return;
    }

    let rev = node_revenue(state.hexes, state.hex_idx, node, state.phase_tiles);

    // Save chain length for reliable restore after backtracking
    let chain_len = state.current_chain.len();

    // Finalize the current chain (add this hex as endpoint)
    if state.current_chain.is_empty()
        || state.current_chain.last().map(|s| s.as_str()) != Some(hex_id)
    {
        state.current_chain.push(hex_id.to_string());
    }
    let chain = state.current_chain.clone();

    // Push stop
    state.stops.push(node.clone());
    state.chains.push(chain);
    let saved_chain = std::mem::replace(&mut state.current_chain, Vec::new());
    state.revenue += rev;
    state.stop_count += stop_cost;

    // Emit candidate
    state.emit_candidate();

    // Continue walking from this node if we haven't reached distance limit.
    // Offboards are always terminal — Python's Offboard.blocks() returns True,
    // preventing traversal past offboard nodes.
    let is_offboard = node.node_type == NodeType::Offboard;
    if !is_offboard && state.stop_count < state.train_distance {
        walk_from_node(state, hex_id, node);
    }

    // Backtrack
    state.stop_count -= stop_cost;
    state.revenue -= rev;
    state.current_chain = saved_chain;
    state.chains.pop();
    state.stops.pop();

    // Restore chain to pre-push state
    state.current_chain.truncate(chain_len);
}

// ---------------------------------------------------------------------------
// Route enumeration
// ---------------------------------------------------------------------------

/// Enumerate all valid routes for a train using path-level walk.
pub fn enumerate_routes(
    hexes: &[Hex],
    hex_idx: &HashMap<String, usize>,
    hex_adjacency: &HashMap<String, HashMap<u8, String>>,
    token_nodes: &[NodeId],
    connected_nodes: &[NodeId],
    train_distance: u32,
    is_d_train: bool,
    phase_tiles: &[String],
    corp_sym: &str,
    finder: &mut RouteFinder,
) -> Vec<RouteCandidate> {
    let token_set: HashSet<NodeId> = token_nodes.iter().cloned().collect();

    let mut state = WalkState {
        hexes,
        hex_idx,
        hex_adjacency,
        phase_tiles,
        corp_sym,
        finder,
        visited_paths: HashSet::new(),
        visited_nodes: HashSet::new(),
        edge_counter: HashMap::new(),
        hexside_bits: 0,
        stops: Vec::new(),
        current_chain: Vec::new(),
        chains: Vec::new(),
        revenue: 0,
        train_distance,
        is_d_train,
        stop_count: 0,
        candidates: Vec::new(),
        token_set,
    };

    for start in connected_nodes {
        // Don't skip blocked cities as starting points — they can be valid
        // terminal stops at the start of a route. Python walks from ALL
        // connected nodes; blocking only prevents traversal THROUGH a city,
        // not starting at one.

        let rev = node_revenue(hexes, hex_idx, start, phase_tiles);
        let is_town = start.node_type == NodeType::Town;
        let stop_cost = if is_d_train && is_town { 0u32 } else { 1u32 };

        state.stops.push(start.clone());
        state.revenue = rev;
        state.stop_count = stop_cost;

        // Start chain with this hex
        state.current_chain.clear();

        walk_from_node(&mut state, &start.hex_id, start);

        state.stops.pop();
        state.revenue = 0;
        state.stop_count = 0;
        state.current_chain.clear();
        state.chains.clear();
    }

    state.candidates
}

// ---------------------------------------------------------------------------
// Optimal route combination
// ---------------------------------------------------------------------------

/// Collect hex-pair edges (normalized) from a route's connection chains.
/// Each hex pair represents a hexside the route physically traverses.
fn route_hex_edges(route: &RouteCandidate) -> HashSet<(String, String)> {
    let mut edges = HashSet::new();
    for chain in &route.connections {
        for pair in chain.windows(2) {
            let (a, b) = (&pair[0], &pair[1]);
            if a < b {
                edges.insert((a.clone(), b.clone()));
            } else {
                edges.insert((b.clone(), a.clone()));
            }
        }
    }
    edges
}

/// Check if a combo of routes has any actual hexside overlaps.
fn combo_has_hex_overlap(combo: &[RouteCandidate]) -> bool {
    let edge_sets: Vec<HashSet<(String, String)>> =
        combo.iter().map(route_hex_edges).collect();
    for i in 0..edge_sets.len() {
        for j in (i + 1)..edge_sets.len() {
            if !edge_sets[i].is_disjoint(&edge_sets[j]) {
                return true;
            }
        }
    }
    false
}

/// Find the revenue-maximizing combination of routes (one per train) with no
/// hexside conflicts.
pub fn find_best_routes(candidates_per_train: &[Vec<RouteCandidate>]) -> (Vec<RouteCandidate>, i32) {
    if candidates_per_train.is_empty() {
        return (Vec::new(), 0);
    }

    if candidates_per_train.len() == 1 {
        let best = candidates_per_train[0].iter().max_by_key(|r| r.revenue);
        return match best {
            Some(route) => (vec![route.clone()], route.revenue),
            None => (Vec::new(), 0),
        };
    }

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

    // Secondary validation: verify the best combo has no actual hexside overlaps.
    // The bitfield check is a fast heuristic; this catches edge cases where bits
    // were assigned inconsistently due to walk order.
    if best_combo.len() > 1 && combo_has_hex_overlap(&best_combo) {
        // Fall back to exhaustive search with hex-edge overlap check
        best_revenue = 0;
        best_combo.clear();
        find_best_recursive_validated(
            candidates_per_train,
            0,
            &mut Vec::new(),
            0,
            0,
            &mut best_revenue,
            &mut best_combo,
        );
    }

    (best_combo, best_revenue)
}

fn find_best_recursive_validated(
    candidates_per_train: &[Vec<RouteCandidate>],
    train_index: usize,
    current_combo: &mut Vec<RouteCandidate>,
    current_revenue: i32,
    used_bits: u128,
    best_revenue: &mut i32,
    best_combo: &mut Vec<RouteCandidate>,
) {
    if train_index >= candidates_per_train.len() {
        if current_revenue > *best_revenue && !combo_has_hex_overlap(current_combo) {
            *best_revenue = current_revenue;
            *best_combo = current_combo.clone();
        }
        return;
    }

    // Option: skip this train
    find_best_recursive_validated(
        candidates_per_train,
        train_index + 1,
        current_combo,
        current_revenue,
        used_bits,
        best_revenue,
        best_combo,
    );

    for route in &candidates_per_train[train_index] {
        if route.hexside_bits & used_bits != 0 {
            continue;
        }

        current_combo.push(route.clone());
        find_best_recursive_validated(
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

    // Option: skip this train
    find_best_recursive(
        candidates_per_train,
        train_index + 1,
        current_combo,
        current_revenue,
        used_bits,
        best_revenue,
        best_combo,
    );

    for route in &candidates_per_train[train_index] {
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
    connected_nodes: &[NodeId],
    trains: &[(u32, bool)],
    phase_tiles: &[String],
    corp_sym: &str,
) -> (Vec<RouteCandidate>, i32) {
    if trains.is_empty() || token_nodes.is_empty() {
        return (Vec::new(), 0);
    }

    // Single RouteFinder shared across all trains so hexside bit assignments
    // are consistent — required for cross-train conflict detection.
    let mut finder = RouteFinder::new();

    let mut candidates_per_train = Vec::new();
    for &(distance, is_d) in trains {
        let candidates = enumerate_routes(
            hexes,
            hex_idx,
            hex_adjacency,
            token_nodes,
            connected_nodes,
            distance,
            is_d,
            phase_tiles,
            corp_sym,
            &mut finder,
        );
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

        let (routes, revenue) = calculate_corp_routes(
            &hexes,
            &hex_idx,
            &adjacency,
            &token_nodes,
            &token_nodes,
            &[(2, false)],
            &vec!["yellow".to_string()],
            "PRR",
        );

        assert_eq!(revenue, 50, "Best route should be A+C = 50");
        assert_eq!(routes.len(), 1);
    }

    #[test]
    fn no_trains_no_revenue() {
        let (hexes, hex_idx, adjacency, token_nodes) = build_linear_for_routing();

        let (routes, revenue) = calculate_corp_routes(
            &hexes,
            &hex_idx,
            &adjacency,
            &token_nodes,
            &token_nodes,
            &[],
            &vec!["yellow".to_string()],
            "PRR",
        );

        assert_eq!(revenue, 0);
        assert!(routes.is_empty());
    }

    #[test]
    fn no_tokens_no_revenue() {
        let (hexes, hex_idx, adjacency, _) = build_linear_for_routing();

        let (routes, revenue) = calculate_corp_routes(
            &hexes,
            &hex_idx,
            &adjacency,
            &[],
            &[],
            &[(2, false)],
            &vec!["yellow".to_string()],
            "PRR",
        );

        assert_eq!(revenue, 0);
        assert!(routes.is_empty());
    }

    #[test]
    fn distance_1_train_stays_home() {
        let (hexes, hex_idx, adjacency, token_nodes) = build_linear_for_routing();

        let mut finder = RouteFinder::new();
        let candidates = enumerate_routes(
            &hexes,
            &hex_idx,
            &adjacency,
            &token_nodes,
            &token_nodes,
            1,
            false,
            &vec!["yellow".to_string()],
            "PRR",
            &mut finder,
        );

        assert!(
            candidates.is_empty(),
            "Distance-1 train shouldn't find valid routes (needs 2 stops)"
        );
    }

    /// Build a diamond: A -- B, A -- C, B -- C
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

        let (routes, revenue) = calculate_corp_routes(
            &hexes,
            &hex_idx,
            &adjacency,
            &token_nodes,
            &token_nodes,
            &[(3, false)],
            &vec!["yellow".to_string()],
            "PRR",
        );

        assert_eq!(revenue, 90, "Should visit all 3 cities for 90 revenue");
        assert_eq!(routes.len(), 1);
        assert_eq!(routes[0].nodes.len(), 3);
    }

    #[test]
    fn diamond_distance2_picks_best_pair() {
        let (hexes, hex_idx, adjacency, token_nodes) = build_diamond_for_routing();

        let (routes, revenue) = calculate_corp_routes(
            &hexes,
            &hex_idx,
            &adjacency,
            &token_nodes,
            &token_nodes,
            &[(2, false)],
            &vec!["yellow".to_string()],
            "PRR",
        );

        assert_eq!(revenue, 60, "Best 2-stop route: A+C=60");
        assert_eq!(routes.len(), 1);
    }

    #[test]
    fn two_trains_no_hexside_conflict() {
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

        let (routes, revenue) = calculate_corp_routes(
            &hexes,
            &hex_idx,
            &adjacency,
            &token_nodes,
            &token_nodes,
            &[(2, false), (2, false)],
            &vec!["yellow".to_string()],
            "PRR",
        );

        assert_eq!(revenue, 70, "Two non-conflicting routes: A-C=40 + A-D=30");
        assert_eq!(routes.len(), 2);
    }

    #[test]
    fn town_adds_revenue_on_route() {
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

        let (routes, revenue) = calculate_corp_routes(
            &hexes,
            &hex_idx,
            &adjacency,
            &token_nodes,
            &token_nodes,
            &[(3, false)],
            &vec!["yellow".to_string()],
            "PRR",
        );

        assert_eq!(revenue, 60, "Route through town: 20+10+30=60");
        assert_eq!(routes.len(), 1);
    }

    #[test]
    fn find_best_routes_with_conflict() {
        let r1 = RouteCandidate {
            nodes: vec![],
            revenue: 50,
            hexside_bits: 0b0001,
            connections: vec![],
        };
        let r2 = RouteCandidate {
            nodes: vec![],
            revenue: 40,
            hexside_bits: 0b0001,
            connections: vec![],
        };
        let r3 = RouteCandidate {
            nodes: vec![],
            revenue: 30,
            hexside_bits: 0b0010,
            connections: vec![],
        };

        let (routes, revenue) = find_best_routes(&[vec![r1, r3.clone()], vec![r2, r3]]);

        assert_eq!(revenue, 80);
        assert_eq!(routes.len(), 2);
    }

    #[test]
    fn single_candidate_list() {
        let r1 = RouteCandidate {
            nodes: vec![],
            revenue: 42,
            hexside_bits: 0,
            connections: vec![],
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

    #[test]
    fn bypass_blocked_city_not_stopped() {
        // H12 tile: city(rev:10), paths: city↔edge1, city↔edge4, edge1↔edge4 (bypass)
        // When city is blocked, train should use bypass, not stop
        let mut tile_h12 = tile_from_dsl(
            "h12",
            "city=revenue:10;path=a:1,b:_0;path=a:4,b:_0;path=a:1,b:4",
            TileColor::Gray,
        );
        // Block the city: fill slot with another corp's token
        place_token(&mut tile_h12, 0, "NYC");

        let mut tile_src = tile_from_dsl("src", "city=revenue:20;path=a:1,b:_0", TileColor::Yellow);
        place_token(&mut tile_src, 0, "PRR");

        let tile_dest = tile_from_dsl("dest", "city=revenue:30;path=a:4,b:_0", TileColor::Yellow);

        let hexes = vec![
            Hex::new("SRC".to_string(), tile_src),
            Hex::new("H12".to_string(), tile_h12),
            Hex::new("DST".to_string(), tile_dest),
        ];
        let hex_idx: HashMap<String, usize> = [
            ("SRC".to_string(), 0),
            ("H12".to_string(), 1),
            ("DST".to_string(), 2),
        ]
        .into();
        let adjacency: HashMap<String, HashMap<u8, String>> = [
            ("SRC".to_string(), [(1u8, "H12".to_string())].into()),
            (
                "H12".to_string(),
                [(4u8, "SRC".to_string()), (1u8, "DST".to_string())].into(),
            ),
            ("DST".to_string(), [(4u8, "H12".to_string())].into()),
        ]
        .into();

        let token_nodes = vec![NodeId {
            hex_id: "SRC".to_string(),
            node_type: NodeType::City,
            index: 0,
        }];
        let connected = vec![
            NodeId {
                hex_id: "SRC".to_string(),
                node_type: NodeType::City,
                index: 0,
            },
            NodeId {
                hex_id: "H12".to_string(),
                node_type: NodeType::City,
                index: 0,
            },
            NodeId {
                hex_id: "DST".to_string(),
                node_type: NodeType::City,
                index: 0,
            },
        ];

        let (routes, revenue) = calculate_corp_routes(
            &hexes,
            &hex_idx,
            &adjacency,
            &token_nodes,
            &connected,
            &[(3, false)],
            &vec!["yellow".to_string()],
            "PRR",
        );

        // Should bypass H12 (blocked) and reach DST: SRC(20) + DST(30) = 50
        // Should NOT stop at H12 since it has a bypass
        assert_eq!(revenue, 50, "Should bypass blocked city and reach DST");
        assert!(routes.len() >= 1);
        // Verify no route includes H12 as a stop
        for r in &routes {
            assert!(
                !r.nodes.iter().any(|n| n.hex_id == "H12"),
                "Route should not stop at blocked city with bypass"
            );
        }
    }

    #[test]
    fn connections_have_hex_chains() {
        let (hexes, hex_idx, adjacency, token_nodes) = build_linear_for_routing();

        let mut finder = RouteFinder::new();
        let candidates = enumerate_routes(
            &hexes,
            &hex_idx,
            &adjacency,
            &token_nodes,
            &token_nodes,
            2,
            false,
            &vec!["yellow".to_string()],
            "PRR",
            &mut finder,
        );

        // Find the A→C route
        let route = candidates
            .iter()
            .find(|r| r.nodes.len() == 2 && r.revenue == 50)
            .expect("Should find A+C route");

        // Should have 1 connection (A to C through B)
        assert_eq!(route.connections.len(), 1, "Should have 1 connection chain");
        let chain = &route.connections[0];
        assert!(chain.contains(&"A".to_string()), "Chain should include A");
        assert!(chain.contains(&"B".to_string()), "Chain should include B");
        assert!(chain.contains(&"C".to_string()), "Chain should include C");
    }
}
