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
    /// True if any token walk found at least one additional revenue node
    /// beyond the starting token city (valid route exists).
    pub route_available: bool,
    /// True if the corp can purchase a train: 2+ mandatory nodes (cities)
    /// are reachable. This is stricter than route_available which also counts
    /// optional nodes (towns). Mirrors Python's route_train_purchase.
    pub route_train_purchase: bool,
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
            // While the 1867 generator runs, it and its validation clones
            // share one memo of these (pure) computations; anywhere else
            // `get_or` just computes.
            let graph = crate::g1867::legal::graph_memo::get_or(
                corp_sym,
                hexes,
                hex_idx,
                hex_adjacency,
                corp_token_hexes,
                reservations,
                || compute_corp_graph(corp_sym, hexes, hex_idx, hex_adjacency, corp_token_hexes, reservations),
            );
            self.cache.insert(corp_sym.to_string(), graph);
        }
        &self.cache[corp_sym]
    }
}

/// `city_index` of a tile-level reservation in a reservations list (a home
/// hex with several cities and no city given): it keeps one slot open.
pub const TILE_RESERVATION: usize = usize::MAX;
/// A tile-level reservation on a LAID tile of a title whose laid tiles block
/// (Ruby `TILE_RESERVATION_BLOCKS_OTHERS = :always`): no other corporation
/// may token the tile.
pub const TILE_RESERVATION_BLOCKING: usize = usize::MAX - 1;

// ---------------------------------------------------------------------------
// Core graph computation
// ---------------------------------------------------------------------------

/// Compute connectivity graph for a single corporation: the walk on flat arrays (§11.3), equal to
/// `compute_corp_graph_reference` by construction (and checked against it on every call in tests).
fn compute_corp_graph(
    corp_sym: &str,
    hexes: &[Hex],
    hex_idx: &HashMap<String, usize>,
    hex_adjacency: &HashMap<String, HashMap<u8, String>>,
    corp_token_hexes: &[(String, usize)],
    reservations: &[(String, usize, String)],
) -> CorpGraph {
    let graph = compute_corp_graph_flat(corp_sym, hexes, hex_idx, hex_adjacency, corp_token_hexes, reservations);
    #[cfg(test)]
    if graph_cross_check() {
        let reference = compute_corp_graph_reference(corp_sym, hexes, hex_idx, hex_adjacency, corp_token_hexes, reservations);
        assert_same_graph(&graph, &reference, corp_sym);
    }
    graph
}

fn compute_corp_graph_flat(
    corp_sym: &str,
    hexes: &[Hex],
    hex_idx: &HashMap<String, usize>,
    hex_adjacency: &HashMap<String, HashMap<u8, String>>,
    corp_token_hexes: &[(String, usize)],
    reservations: &[(String, usize, String)],
) -> CorpGraph {
    // a token on a hex the board doesn't index (never on a real board): the reference walk
    if corp_token_hexes.iter().any(|(h, _)| !hex_idx.contains_key(h)) {
        return compute_corp_graph_reference(corp_sym, hexes, hex_idx, hex_adjacency, corp_token_hexes, reservations);
    }
    let ctx = FlatCtx::new(hexes, hex_idx, hex_adjacency, corp_sym, corp_token_hexes);
    let n_nodes = ctx.n_nodes();
    let words = n_nodes.div_ceil(64).max(1);
    let token_nodes: Vec<(usize, usize)> = corp_token_hexes
        .iter()
        .map(|(h, ci)| {
            let hi = hex_idx[h];
            (hi, ctx.node(hi, &PathEndpoint::City(*ci)))
        })
        .collect();
    let mut all_nodes = vec![0u64; words];
    let mut all_hexes = vec![0u8; hexes.len()];
    let mut route_available = false;
    let mut route_train_purchase = false;
    let mut w = FlatWalk::new(&ctx);
    let mut pre_seed = vec![0u64; words];
    for (k, (token_hex_id, city_index)) in corp_token_hexes.iter().enumerate() {
        let (hi, start) = token_nodes[k];
        // a fresh walk, its visited nodes pre-seeded with the other token cities
        w.visited_paths.iter_mut().for_each(|x| *x = 0);
        w.edge_counter.iter_mut().for_each(|x| *x = 0);
        w.visited_nodes.iter_mut().for_each(|x| *x = 0);
        w.result_nodes.iter_mut().for_each(|x| *x = 0);
        w.hexes.iter_mut().for_each(|x| *x = 0);
        pre_seed.iter_mut().for_each(|x| *x = 0);
        for (j, (h, ci)) in corp_token_hexes.iter().enumerate() {
            if h != token_hex_id || *ci != *city_index {
                bit_set(&mut pre_seed, token_nodes[j].1);
                bit_set(&mut w.visited_nodes, token_nodes[j].1);
            }
        }
        walk_from_node(&ctx, &mut w, hi, start, false, false);

        if !route_available {
            let mut has_optional = false;
            let mut reached_other_token = false;
            let mut other_mandatory = false;
            for word in 0..words {
                let mut found = w.visited_nodes[word] | w.result_nodes[word];
                while found != 0 {
                    let n = word * 64 + found.trailing_zeros() as usize;
                    found &= found - 1;
                    let seeded = bit_get(&pre_seed, n);
                    let hex_seen = all_present(&w.hexes, ctx.node_hex[n] as usize);
                    if n != start && !seeded {
                        has_optional = true;
                    }
                    if seeded && hex_seen {
                        reached_other_token = true;
                    }
                    if n != start && (!seeded || hex_seen) {
                        other_mandatory = true;
                    }
                }
            }
            if has_optional || reached_other_token {
                route_available = true;
            }
            if !route_train_purchase && other_mandatory {
                route_train_purchase = true;
            }
        }

        for word in 0..words {
            all_nodes[word] |= w.visited_nodes[word] | w.result_nodes[word];
        }
        for (a, b) in all_hexes.iter_mut().zip(&w.hexes) {
            *a |= *b;
        }
    }

    let mut visited_nodes: HashSet<NodeId> = HashSet::new();
    for word in 0..words {
        let mut bits = all_nodes[word];
        while bits != 0 {
            let n = word * 64 + bits.trailing_zeros() as usize;
            bits &= bits - 1;
            visited_nodes.insert(ctx.node_id(n));
        }
    }
    let mut visited_hexes: HashMap<String, HashSet<u8>> = HashMap::new();
    for (hi, &m) in all_hexes.iter().enumerate() {
        if m & HEX_PRESENT != 0 {
            visited_hexes.insert(hexes[hi].id.clone(), (0..6u8).filter(|e| m & (1 << e) != 0).collect());
        }
    }
    let tokenable = tokenable_cities_of(corp_sym, hexes, hex_idx, corp_token_hexes, reservations, &visited_nodes);
    CorpGraph {
        connected_hexes: visited_hexes,
        connected_nodes: visited_nodes,
        tokenable_cities: tokenable,
        route_available,
        route_train_purchase,
    }
}

#[inline]
fn all_present(hexes: &[u8], hi: usize) -> bool {
    hexes[hi] & HEX_PRESENT != 0
}

/// Tests compare the flat walk with the reference on every computation (`GRAPH_CROSS_CHECK=0` turns it off).
#[cfg(test)]
fn graph_cross_check() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("GRAPH_CROSS_CHECK").map_or(true, |v| v != "0"))
}

#[cfg(test)]
pub(crate) static GRAPH_CROSS_CHECKS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

#[cfg(test)]
fn assert_same_graph(a: &CorpGraph, b: &CorpGraph, corp_sym: &str) {
    GRAPH_CROSS_CHECKS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let toks = |g: &CorpGraph| g.tokenable_cities.iter().map(|t| (t.hex_id.clone(), t.city_index)).collect::<Vec<_>>();
    assert_eq!(a.connected_hexes, b.connected_hexes, "{corp_sym}: connected_hexes");
    assert_eq!(a.connected_nodes, b.connected_nodes, "{corp_sym}: connected_nodes");
    assert_eq!(toks(a), toks(b), "{corp_sym}: tokenable_cities");
    assert_eq!((a.route_available, a.route_train_purchase), (b.route_available, b.route_train_purchase), "{corp_sym}: routes");
}

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

// ---------------------------------------------------------------------------
// The walk on flat arrays (docs/SITE_ENGINE_RS_STEP3.md §11.3)
// ---------------------------------------------------------------------------
//
// The same recursion as `compute_corp_graph_reference` / `walk_from_node_reference` / `walk_into_hex_reference`
// below (the walk as it was, kept verbatim: the fallback and the differential check), on integer state: hexes by
// index, the neighbour across each edge looked up once, nodes by number, visited paths as a bitmask per hex, the edge
// counter per hex side, visited / result nodes as bitsets, visited hexes as present + an edge mask. Every decision
// reads the same facts in the same order (`tile.paths` order; `is_converging_exit` and `is_city_blocked` precomputed
// per call, pure functions of the tile and the company), so the result is the same; only the containers changed:
// the String-keyed hash sets allocated and hashed on every step were most of the walk's time.

/// The walk's flat context for one company.
struct FlatCtx<'a> {
    hexes: &'a [Hex],
    /// The neighbour hex index across each edge (`hex_adjacency`, then `hex_idx`), per hex.
    nbr: Vec<[u32; 6]>,
    /// Bit e: `is_converging_exit(tile, e)`, per hex.
    converging: Vec<u8>,
    /// Per hex: its first node number and its numbers of city, town and offboard nodes (each at least the tile's
    /// count and every index its paths or the company's tokens name).
    node_base: Vec<u32>,
    counts: Vec<[u32; 3]>,
    /// Per node number: a city that blocks the company (`is_city_blocked`); its hex.
    blocked: Vec<bool>,
    node_hex: Vec<u32>,
    /// Words of the visited-paths bitmask per hex.
    path_words: usize,
}

const NO_NBR: u32 = u32::MAX;

impl<'a> FlatCtx<'a> {
    fn new(
        hexes: &'a [Hex],
        hex_idx: &HashMap<String, usize>,
        hex_adjacency: &HashMap<String, HashMap<u8, String>>,
        corp_sym: &str,
        corp_token_hexes: &[(String, usize)],
    ) -> Self {
        let n = hexes.len();
        let mut nbr = vec![[NO_NBR; 6]; n];
        let mut converging = vec![0u8; n];
        let mut node_base = Vec::with_capacity(n + 1);
        let mut counts = Vec::with_capacity(n);
        let mut max_paths = 1usize;
        let mut total = 0u32;
        for (hi, hex) in hexes.iter().enumerate() {
            let tile = &hex.tile;
            max_paths = max_paths.max(tile.paths.len());
            let mut c = [tile.cities.len() as u32, tile.towns.len() as u32, tile.offboards.len() as u32];
            for p in &tile.paths {
                for ep in [&p.a, &p.b] {
                    match ep {
                        PathEndpoint::City(i) => c[0] = c[0].max(*i as u32 + 1),
                        PathEndpoint::Town(i) => c[1] = c[1].max(*i as u32 + 1),
                        PathEndpoint::Offboard(i) => c[2] = c[2].max(*i as u32 + 1),
                        _ => {}
                    }
                }
            }
            for (h, ci) in corp_token_hexes {
                if *h == hex.id {
                    c[0] = c[0].max(*ci as u32 + 1);
                }
            }
            if let Some(adj) = hex_adjacency.get(&hex.id) {
                for (&e, nb) in adj {
                    if e < 6 {
                        if let Some(&x) = hex_idx.get(nb) {
                            nbr[hi][e as usize] = x as u32;
                        }
                    }
                }
            }
            for e in 0..6u8 {
                if is_converging_exit(tile, e) {
                    converging[hi] |= 1 << e;
                }
            }
            node_base.push(total);
            counts.push(c);
            total += c[0] + c[1] + c[2];
        }
        node_base.push(total);
        let mut blocked = vec![false; total as usize];
        let mut node_hex = vec![0u32; total as usize];
        for (hi, hex) in hexes.iter().enumerate() {
            for ci in 0..counts[hi][0] as usize {
                blocked[node_base[hi] as usize + ci] = is_city_blocked(hexes, hex_idx, &hex.id, ci, corp_sym);
            }
            for n in node_base[hi]..node_base[hi + 1] {
                node_hex[n as usize] = hi as u32;
            }
        }
        FlatCtx { hexes, nbr, converging, node_base, counts, blocked, node_hex, path_words: max_paths.div_ceil(64) }
    }

    fn n_nodes(&self) -> usize {
        *self.node_base.last().unwrap_or(&0) as usize
    }

    /// The node number of a node endpoint of hex `hi`.
    #[inline]
    fn node(&self, hi: usize, ep: &PathEndpoint) -> usize {
        let base = self.node_base[hi] as usize;
        let c = &self.counts[hi];
        match ep {
            PathEndpoint::City(i) => base + i,
            PathEndpoint::Town(i) => base + c[0] as usize + i,
            PathEndpoint::Offboard(i) => base + (c[0] + c[1]) as usize + i,
            _ => usize::MAX,
        }
    }

    /// A node number back as the walk's `NodeId`.
    fn node_id(&self, n: usize) -> NodeId {
        let hi = self.node_hex[n] as usize;
        let off = n as u32 - self.node_base[hi];
        let c = &self.counts[hi];
        let (node_type, index) = if off < c[0] {
            (NodeType::City, off)
        } else if off < c[0] + c[1] {
            (NodeType::Town, off - c[0])
        } else {
            (NodeType::Offboard, off - c[0] - c[1])
        };
        NodeId { hex_id: self.hexes[hi].id.clone(), node_type, index: index as usize }
    }
}

/// One token walk's state, flat.
struct FlatWalk {
    visited_paths: Vec<u64>,
    edge_counter: Vec<u32>,
    visited_nodes: Vec<u64>,
    result_nodes: Vec<u64>,
    /// Bit 6: the hex is in `visited_hexes`; bits 0-5: its edges.
    hexes: Vec<u8>,
}

const HEX_PRESENT: u8 = 1 << 6;

#[inline]
fn bit_get(v: &[u64], i: usize) -> bool {
    v[i >> 6] & (1u64 << (i & 63)) != 0
}
#[inline]
fn bit_set(v: &mut [u64], i: usize) {
    v[i >> 6] |= 1u64 << (i & 63);
}
#[inline]
fn bit_clear(v: &mut [u64], i: usize) {
    v[i >> 6] &= !(1u64 << (i & 63));
}

impl FlatWalk {
    fn new(ctx: &FlatCtx) -> Self {
        let n = ctx.hexes.len();
        let words = ctx.n_nodes().div_ceil(64).max(1);
        FlatWalk {
            visited_paths: vec![0; n * ctx.path_words],
            edge_counter: vec![0; n * 6],
            visited_nodes: vec![0; words],
            result_nodes: vec![0; words],
            hexes: vec![0; n],
        }
    }
    #[inline]
    fn path_visited(&self, ctx: &FlatCtx, hi: usize, pi: usize) -> bool {
        bit_get(&self.visited_paths, hi * ctx.path_words * 64 + pi)
    }
    #[inline]
    fn set_path(&mut self, ctx: &FlatCtx, hi: usize, pi: usize) {
        bit_set(&mut self.visited_paths, hi * ctx.path_words * 64 + pi);
    }
    #[inline]
    fn clear_path(&mut self, ctx: &FlatCtx, hi: usize, pi: usize) {
        bit_clear(&mut self.visited_paths, hi * ctx.path_words * 64 + pi);
    }
}

/// `walk_from_node_reference` on the flat state.
fn walk_from_node(ctx: &FlatCtx, w: &mut FlatWalk, hi: usize, node: usize, converging: bool, converging_path: bool) {
    if bit_get(&w.visited_nodes, node) {
        return;
    }
    bit_set(&mut w.visited_nodes, node);
    bit_set(&mut w.result_nodes, node);
    w.hexes[hi] |= HEX_PRESENT;
    let tile = &ctx.hexes[hi].tile;
    let node_endpoint = {
        let base = ctx.node_base[hi] as usize;
        let c = &ctx.counts[hi];
        let off = node - base;
        if off < c[0] as usize {
            PathEndpoint::City(off)
        } else if off < (c[0] + c[1]) as usize {
            PathEndpoint::Town(off - c[0] as usize)
        } else {
            PathEndpoint::Offboard(off - (c[0] + c[1]) as usize)
        }
    };
    for (path_idx, path) in tile.paths.iter().enumerate() {
        if path.terminal {
            continue;
        }
        if w.path_visited(ctx, hi, path_idx) {
            continue;
        }
        let exit = if path.a == node_endpoint {
            &path.b
        } else if path.b == node_endpoint {
            &path.a
        } else {
            continue;
        };
        w.set_path(ctx, hi, path_idx);
        match exit {
            PathEndpoint::Edge(exit_edge) => {
                w.hexes[hi] |= HEX_PRESENT | (1 << exit_edge);
                let next_converging = converging || ctx.converging[hi] & (1 << exit_edge) != 0;
                let nb = ctx.nbr[hi][*exit_edge as usize];
                if nb != NO_NBR {
                    let k = hi * 6 + *exit_edge as usize;
                    if w.edge_counter[k] == 0 {
                        w.edge_counter[k] += 1;
                        walk_into_hex(ctx, w, nb as usize, (*exit_edge + 3) % 6, next_converging);
                        w.edge_counter[k] -= 1;
                    }
                }
            }
            PathEndpoint::City(idx) => {
                let dest = ctx.node(hi, &PathEndpoint::City(*idx));
                if !ctx.blocked[dest] {
                    walk_from_node(ctx, w, hi, dest, converging, converging_path || converging);
                }
            }
            PathEndpoint::Town(idx) => {
                let dest = ctx.node(hi, &PathEndpoint::Town(*idx));
                walk_from_node(ctx, w, hi, dest, converging, converging_path || converging);
            }
            PathEndpoint::Offboard(idx) => {
                let dest = ctx.node(hi, &PathEndpoint::Offboard(*idx));
                bit_set(&mut w.result_nodes, dest);
                bit_set(&mut w.visited_nodes, dest);
            }
            PathEndpoint::Junction => {
                for (other_idx, other_path) in tile.paths.iter().enumerate() {
                    if other_idx == path_idx || other_path.terminal {
                        continue;
                    }
                    if w.path_visited(ctx, hi, other_idx) {
                        continue;
                    }
                    let has_junction = other_path.a == PathEndpoint::Junction || other_path.b == PathEndpoint::Junction;
                    if !has_junction {
                        continue;
                    }
                    let other_exit = if other_path.a == PathEndpoint::Junction { &other_path.b } else { &other_path.a };
                    if let PathEndpoint::Edge(e) = other_exit {
                        w.set_path(ctx, hi, other_idx);
                        let next_converging = converging || ctx.converging[hi] & (1 << e) != 0;
                        w.hexes[hi] |= HEX_PRESENT | (1 << e);
                        let nb = ctx.nbr[hi][*e as usize];
                        if nb != NO_NBR {
                            let k = hi * 6 + *e as usize;
                            if w.edge_counter[k] == 0 {
                                w.edge_counter[k] += 1;
                                walk_into_hex(ctx, w, nb as usize, (*e + 3) % 6, next_converging);
                                w.edge_counter[k] -= 1;
                            }
                        }
                        if converging {
                            w.clear_path(ctx, hi, other_idx);
                        }
                    }
                }
            }
        }
        if converging {
            w.clear_path(ctx, hi, path_idx);
        }
    }
    if converging_path {
        bit_clear(&mut w.visited_nodes, node);
    }
}

/// `walk_into_hex_reference` on the flat state.
fn walk_into_hex(ctx: &FlatCtx, w: &mut FlatWalk, hi: usize, enter_edge: u8, converging: bool) {
    let tile = &ctx.hexes[hi].tile;
    let enter_ep = PathEndpoint::Edge(enter_edge);
    for (path_idx, path) in tile.paths.iter().enumerate() {
        if path.terminal {
            continue;
        }
        let in_stack = |e: Option<u8>| e.is_some_and(|e| w.edge_counter[hi * 6 + e as usize] > 0);
        if in_stack(path.a.edge_num()) || in_stack(path.b.edge_num()) {
            continue;
        }
        if w.path_visited(ctx, hi, path_idx) {
            continue;
        }
        let dest = if path.a == enter_ep {
            &path.b
        } else if path.b == enter_ep {
            &path.a
        } else {
            continue;
        };
        w.hexes[hi] |= HEX_PRESENT | (1 << enter_edge);
        w.set_path(ctx, hi, path_idx);
        match dest {
            PathEndpoint::Edge(exit_edge) => {
                w.hexes[hi] |= HEX_PRESENT | (1 << exit_edge);
                let next_converging = converging || ctx.converging[hi] & (1 << exit_edge) != 0;
                let nb = ctx.nbr[hi][*exit_edge as usize];
                if nb != NO_NBR {
                    let k = hi * 6 + *exit_edge as usize;
                    if w.edge_counter[k] == 0 {
                        w.edge_counter[k] += 1;
                        walk_into_hex(ctx, w, nb as usize, (*exit_edge + 3) % 6, next_converging);
                        w.edge_counter[k] -= 1;
                    }
                }
            }
            PathEndpoint::City(idx) => {
                let node = ctx.node(hi, &PathEndpoint::City(*idx));
                if !ctx.blocked[node] {
                    walk_from_node(ctx, w, hi, node, converging, converging);
                } else {
                    bit_set(&mut w.result_nodes, node);
                    bit_set(&mut w.visited_nodes, node);
                }
            }
            PathEndpoint::Town(idx) => {
                let node = ctx.node(hi, &PathEndpoint::Town(*idx));
                walk_from_node(ctx, w, hi, node, converging, converging);
            }
            PathEndpoint::Offboard(idx) => {
                let node = ctx.node(hi, &PathEndpoint::Offboard(*idx));
                bit_set(&mut w.result_nodes, node);
                bit_set(&mut w.visited_nodes, node);
            }
            PathEndpoint::Junction => {
                for (other_idx, other_path) in tile.paths.iter().enumerate() {
                    if other_idx == path_idx || other_path.terminal {
                        continue;
                    }
                    if w.path_visited(ctx, hi, other_idx) {
                        continue;
                    }
                    let has_junction = other_path.a == PathEndpoint::Junction || other_path.b == PathEndpoint::Junction;
                    if !has_junction {
                        continue;
                    }
                    let other_exit = if other_path.a == PathEndpoint::Junction { &other_path.b } else { &other_path.a };
                    if let PathEndpoint::Edge(e) = other_exit {
                        if *e == enter_edge {
                            continue;
                        }
                        w.set_path(ctx, hi, other_idx);
                        let next_converging = converging || ctx.converging[hi] & (1 << e) != 0;
                        w.hexes[hi] |= HEX_PRESENT | (1 << e);
                        let nb = ctx.nbr[hi][*e as usize];
                        if nb != NO_NBR {
                            let k = hi * 6 + *e as usize;
                            if w.edge_counter[k] == 0 {
                                w.edge_counter[k] += 1;
                                walk_into_hex(ctx, w, nb as usize, (*e + 3) % 6, next_converging);
                                w.edge_counter[k] -= 1;
                            }
                        }
                        if converging {
                            w.clear_path(ctx, hi, other_idx);
                        }
                    }
                }
            }
        }
        if converging {
            w.clear_path(ctx, hi, path_idx);
        }
    }
}

/// The cities a company may token among the nodes it reaches (`compute_corp_graph`'s rule: an open slot, no token of
/// its in the hex, the reservations), sorted by hex id and city.
fn tokenable_cities_of(
    corp_sym: &str,
    hexes: &[Hex],
    hex_idx: &HashMap<String, usize>,
    corp_token_hexes: &[(String, usize)],
    reservations: &[(String, usize, String)],
    visited_nodes: &HashSet<NodeId>,
) -> Vec<TokenableCity> {
    let mut tokenable = Vec::new();
    for node in visited_nodes {
        if node.node_type != NodeType::City {
            continue;
        }
        if corp_token_hexes.iter().any(|(h, ci)| h == &node.hex_id && *ci == node.index) {
            continue;
        }
        if let Some(&hi) = hex_idx.get(&node.hex_id) {
            let city = &hexes[hi].tile.cities[node.index];
            if city.tokens.iter().any(|t| t.is_none()) {
                let corp_already_in_hex = hexes[hi]
                    .tile
                    .cities
                    .iter()
                    .any(|c| c.tokens.iter().any(|t| t.as_ref().is_some_and(|tok| tok.corporation_id == corp_sym)));
                if !corp_already_in_hex {
                    let tile_level_blocking = reservations
                        .iter()
                        .any(|(rh, rc, rsym)| rh == &node.hex_id && *rc == TILE_RESERVATION_BLOCKING && rsym != corp_sym);
                    let tile_level_other =
                        reservations.iter().any(|(rh, rc, rsym)| rh == &node.hex_id && *rc == TILE_RESERVATION && rsym != corp_sym);
                    let num_cities = hexes[hi].tile.cities.len();
                    let blocked = if tile_level_blocking || (tile_level_other && num_cities <= 1) {
                        true
                    } else if tile_level_other {
                        let total_slots: usize = hexes[hi].tile.cities.iter().map(|c| c.tokens.len()).sum();
                        let total_tokens: usize =
                            hexes[hi].tile.cities.iter().map(|c| c.tokens.iter().filter(|t| t.is_some()).count()).sum();
                        let other_tile_reservations = reservations
                            .iter()
                            .filter(|(rh, rc, rsym)| rh == &node.hex_id && *rc == TILE_RESERVATION && rsym != corp_sym)
                            .count();
                        let other_city_reservations = reservations
                            .iter()
                            .filter(|(rh, rc, rsym)| rh == &node.hex_id && *rc < TILE_RESERVATION_BLOCKING && rsym != corp_sym)
                            .count();
                        total_slots.saturating_sub(total_tokens + other_tile_reservations + other_city_reservations) == 0
                    } else {
                        let num_tokens = city.tokens.iter().filter(|t| t.is_some()).count();
                        let num_other_reservations = reservations
                            .iter()
                            .filter(|(rh, rc, rsym)| rh == &node.hex_id && *rc == node.index && rsym != corp_sym)
                            .count();
                        let total_slots = city.tokens.len();
                        total_slots.saturating_sub(num_tokens + num_other_reservations) == 0
                    };
                    if !blocked {
                        tokenable.push(TokenableCity { hex_id: node.hex_id.clone(), city_index: node.index });
                    }
                }
            }
        }
    }
    tokenable.sort_by(|a, b| (a.hex_id.as_str(), a.city_index).cmp(&(b.hex_id.as_str(), b.city_index)));
    tokenable
}

/// The walk as it was before the flat arrays (§11.3), kept verbatim: the fallback for a token on a hex the
/// board doesn't index, and what the differential check compares the flat walk with.
fn compute_corp_graph_reference(
    corp_sym: &str,
    hexes: &[Hex],
    hex_idx: &HashMap<String, usize>,
    hex_adjacency: &HashMap<String, HashMap<u8, String>>,
    corp_token_hexes: &[(String, usize)],
    reservations: &[(String, usize, String)],
) -> CorpGraph {
    let mut visited_nodes = HashSet::new();
    let mut visited_hexes: HashMap<String, HashSet<u8>> = HashMap::new();
    let mut route_available = false;
    let mut route_train_purchase = false;

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
        let pre_seed: HashSet<NodeId> = corp_token_hexes
            .iter()
            .filter(|(h, ci)| h != token_hex_id || *ci != *city_index)
            .map(|(h, ci)| NodeId {
                hex_id: h.clone(),
                node_type: NodeType::City,
                index: *ci,
            })
            .collect();
        let mut local_visited_nodes: HashSet<NodeId> = pre_seed.clone();
        let mut local_visited_hexes: HashMap<String, HashSet<u8>> = HashMap::new();
        let mut local_visited_paths: HashSet<(String, usize)> = HashSet::new();
        let mut edge_counter: HashMap<(String, u8), u32> = HashMap::new();
        let mut local_result_nodes: HashSet<NodeId> = HashSet::new();

        walk_from_node_reference(
            hexes,
            hex_idx,
            hex_adjacency,
            token_hex_id,
            &start_node,
            &mut local_visited_nodes,
            &mut local_result_nodes,
            &mut local_visited_hexes,
            &mut local_visited_paths,
            corp_sym,
            false, // converging starts as false for the token node
            &mut edge_counter,
            false, // converging_path starts as false
        );

        // Check if this walk found additional revenue nodes
        if !route_available {
            let pre_seed: HashSet<NodeId> = corp_token_hexes
                .iter()
                .filter(|(h, ci)| h != token_hex_id || *ci != *city_index)
                .map(|(h, ci)| NodeId {
                    hex_id: h.clone(),
                    node_type: NodeType::City,
                    index: *ci,
                })
                .collect();
            // Union of visited and result nodes — visited may miss nodes that
            // were un-visited by converging, result catches those.
            let all_found: HashSet<&NodeId> = local_visited_nodes
                .iter()
                .chain(local_result_nodes.iter())
                .collect();
            let has_optional = all_found.iter().any(|n| {
                **n != start_node
                    && !pre_seed.contains(n)
                    && (n.node_type == NodeType::City
                        || n.node_type == NodeType::Town
                        || n.node_type == NodeType::Offboard)
            });
            let reached_other_token = all_found.iter().any(|n| {
                pre_seed.contains(n) && local_visited_hexes.contains_key(&n.hex_id)
            });
            if has_optional || reached_other_token {
                route_available = true;
            }
            // route_train_purchase: 2+ mandatory nodes. In 1830 Cities,
            // Towns, and Offboards all count as mandatory revenue stops
            // (Python's ``Node.route == "mandatory"`` covers all three).
            // Start node counts as 1; any other City/Town/Offboard reached
            // counts as another.
            if !route_train_purchase {
                let other_mandatory = all_found.iter().any(|n| {
                    (n.node_type == NodeType::City
                        || n.node_type == NodeType::Town
                        || n.node_type == NodeType::Offboard)
                        && **n != start_node
                        && (!pre_seed.contains(n) || local_visited_hexes.contains_key(&n.hex_id))
                });
                if other_mandatory {
                    route_train_purchase = true;
                }
            }
        }

        // Merge results
        for node in local_visited_nodes {
            visited_nodes.insert(node);
        }
        for node in local_result_nodes {
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
                    // Reservation blocking, matching Ruby/Python
                    // ``Tile#token_blocked_by_reservation?``:
                    //
                    //   * A blocking tile-level reservation (a laid tile in a
                    //     `:always` title, TILE_RESERVATION_BLOCKING) blocks
                    //     all other corps from tokening the tile; so does one
                    //     on a single-city tile.
                    //   * A tile-level reservation on a preprinted multi-city
                    //     tile (ERIE's E11, `:never`) just needs ONE slot
                    //     somewhere on the tile to remain open; other corps may
                    //     token any other slot: slot-counting across the tile.
                    //   * City-level reservations: slot-level blocking on the
                    //     specific city.
                    let tile_level_blocking = reservations.iter().any(|(rh, rc, rsym)| {
                        rh == &node.hex_id && *rc == TILE_RESERVATION_BLOCKING && rsym != corp_sym
                    });
                    let tile_level_other = reservations.iter().any(|(rh, rc, rsym)| {
                        rh == &node.hex_id && *rc == TILE_RESERVATION && rsym != corp_sym
                    });
                    let num_cities = hexes[hi].tile.cities.len();
                    let blocked = if tile_level_blocking || (tile_level_other && num_cities <= 1) {
                        true
                    } else if tile_level_other {
                        // Multi-city tile with tile-level reservation: total
                        // free slots across the whole tile must remain >
                        // number of other corps' tile-level reservations
                        // (slot-counting). Mirrors Python's else branch which
                        // uses ``city.available_slots`` summed across cities.
                        let total_slots: usize =
                            hexes[hi].tile.cities.iter().map(|c| c.tokens.len()).sum();
                        let total_tokens: usize = hexes[hi]
                            .tile
                            .cities
                            .iter()
                            .map(|c| c.tokens.iter().filter(|t| t.is_some()).count())
                            .sum();
                        let other_tile_reservations = reservations
                            .iter()
                            .filter(|(rh, rc, rsym)| {
                                rh == &node.hex_id && *rc == TILE_RESERVATION && rsym != corp_sym
                            })
                            .count();
                        // Also account for other corps' city-level reservations
                        // on this tile so the slot bookkeeping matches Python.
                        let other_city_reservations = reservations
                            .iter()
                            .filter(|(rh, rc, rsym)| {
                                rh == &node.hex_id && *rc < TILE_RESERVATION_BLOCKING && rsym != corp_sym
                            })
                            .count();
                        total_slots
                            .saturating_sub(total_tokens + other_tile_reservations + other_city_reservations)
                            == 0
                    } else {
                        // City-level: slot-level blocking
                        let num_tokens = city.tokens.iter().filter(|t| t.is_some()).count();
                        let num_other_reservations = reservations
                            .iter()
                            .filter(|(rh, rc, rsym)| {
                                rh == &node.hex_id && *rc == node.index && rsym != corp_sym
                            })
                            .count();
                        let total_slots = city.tokens.len();
                        total_slots.saturating_sub(num_tokens + num_other_reservations) == 0
                    };
                    if !blocked {
                        tokenable.push(TokenableCity {
                            hex_id: node.hex_id.clone(),
                            city_index: node.index,
                        });
                    }
                }
            }
        }
    }

    // The walk visits nodes through hash sets, so the order cities were found in changes from run to
    // run; sort them (hex id, then city) so every caller, and the move lists built from them, see one
    // order (the Python ActionHelper sorts its token moves too). The set itself is unchanged.
    tokenable.sort_by(|a, b| (a.hex_id.as_str(), a.city_index).cmp(&(b.hex_id.as_str(), b.city_index)));

    CorpGraph {
        connected_hexes: visited_hexes,
        connected_nodes: visited_nodes,
        tokenable_cities: tokenable,
        route_available,
        route_train_purchase,
    }
}

// ---------------------------------------------------------------------------
// Walk algorithm
// ---------------------------------------------------------------------------

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
fn walk_from_node_reference(
    hexes: &[Hex],
    hex_idx: &HashMap<String, usize>,
    hex_adjacency: &HashMap<String, HashMap<u8, String>>,
    hex_id: &str,
    node: &NodeId,
    visited_nodes: &mut HashSet<NodeId>,
    result_nodes: &mut HashSet<NodeId>,
    visited_hexes: &mut HashMap<String, HashSet<u8>>,
    visited_paths: &mut HashSet<(String, usize)>,
    corp_sym: &str,
    converging: bool,
    edge_counter: &mut HashMap<(String, u8), u32>,
    converging_path: bool,
) {
    if visited_nodes.contains(node) {
        return;
    }
    visited_nodes.insert(node.clone());
    result_nodes.insert(node.clone());

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
                        walk_into_hex_reference(
                            hexes,
                            hex_idx,
                            hex_adjacency,
                            neighbor_id,
                            enter_edge,
                            visited_nodes,
                            result_nodes,
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
                    walk_from_node_reference(
                        hexes,
                        hex_idx,
                        hex_adjacency,
                        hex_id,
                        &dest_node,
                        visited_nodes,
                        result_nodes,
                        visited_hexes,
                        visited_paths,
                        corp_sym,
                        converging,
                        edge_counter,
                        converging_path || converging,
                    );
                }
            }
            PathEndpoint::Town(idx) => {
                let dest_node = NodeId {
                    hex_id: hex_id.to_string(),
                    node_type: NodeType::Town,
                    index: *idx,
                };
                walk_from_node_reference(
                    hexes,
                    hex_idx,
                    hex_adjacency,
                    hex_id,
                    &dest_node,
                    visited_nodes,
                    result_nodes,
                    visited_hexes,
                    visited_paths,
                    corp_sym,
                    converging,
                    edge_counter,
                    converging_path || converging,
                );
            }
            PathEndpoint::Offboard(idx) => {
                let dest_node = NodeId {
                    hex_id: hex_id.to_string(),
                    node_type: NodeType::Offboard,
                    index: *idx,
                };
                result_nodes.insert(dest_node.clone());
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
                                walk_into_hex_reference(
                                    hexes,
                                    hex_idx,
                                    hex_adjacency,
                                    neighbor_id,
                                    enter_edge,
                                    visited_nodes,
                                    result_nodes,
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

    // Un-visit the node when converging_path is true (matching Python's
    // `if converging_path: del visited[self]` in Node.walk). This allows
    // sibling DFS branches to re-enter this node. The node stays in
    // result_nodes permanently — only the cycle-prevention set is affected.
    if converging_path {
        visited_nodes.remove(node);
    }
}

/// Walk into a hex from a specific entry edge, following paths.
///
/// See `walk_from_node` for documentation on `converging` and `edge_counter`.
#[allow(clippy::too_many_arguments)]
fn walk_into_hex_reference(
    hexes: &[Hex],
    hex_idx: &HashMap<String, usize>,
    hex_adjacency: &HashMap<String, HashMap<u8, String>>,
    hex_id: &str,
    enter_edge: u8,
    visited_nodes: &mut HashSet<NodeId>,
    result_nodes: &mut HashSet<NodeId>,
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

    // Don't add hex to visited_hexes yet — only add after finding a matching
    // path from the entry edge. This prevents "phantom" connectivity through
    // hexes that have no path matching the entry direction.

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

        // Found a matching path — now add the hex to visited with entry edge
        visited_hexes
            .entry(hex_id.to_string())
            .or_default()
            .insert(enter_edge);

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
                        walk_into_hex_reference(
                            hexes,
                            hex_idx,
                            hex_adjacency,
                            neighbor_id,
                            next_enter,
                            visited_nodes,
                            result_nodes,
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
                    walk_from_node_reference(
                        hexes,
                        hex_idx,
                        hex_adjacency,
                        hex_id,
                        &node,
                        visited_nodes,
                        result_nodes,
                        visited_hexes,
                        visited_paths,
                        corp_sym,
                        converging,
                        edge_counter,
                        converging,
                    );
                } else {
                    result_nodes.insert(node.clone());
                    visited_nodes.insert(node);
                }
            }
            PathEndpoint::Town(idx) => {
                let node = NodeId {
                    hex_id: hex_id.to_string(),
                    node_type: NodeType::Town,
                    index: *idx,
                };
                walk_from_node_reference(
                    hexes,
                    hex_idx,
                    hex_adjacency,
                    hex_id,
                    &node,
                    visited_nodes,
                    result_nodes,
                    visited_hexes,
                    visited_paths,
                    corp_sym,
                    converging,
                    edge_counter,
                    converging,
                );
            }
            PathEndpoint::Offboard(idx) => {
                let node = NodeId {
                    hex_id: hex_id.to_string(),
                    node_type: NodeType::Offboard,
                    index: *idx,
                };
                result_nodes.insert(node.clone());
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
                                walk_into_hex_reference(
                                    hexes,
                                    hex_idx,
                                    hex_adjacency,
                                    neighbor_id,
                                    next_enter,
                                    visited_nodes,
                                    result_nodes,
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

    // A neutral token (1867's CN green placeholders) keeps the city open
    // for everyone (Ruby City#blocks?).
    if city
        .tokens
        .iter()
        .any(|t| t.as_ref().is_some_and(|tok| tok.token_type == "neutral"))
    {
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

    /// G1 (docs/SITE_ENGINE_RS_STEP3.md §11.3): every `compute_corp_graph` a test makes is compared with the walk as it
    /// was (`compute_corp_graph_reference`, `graph_cross_check`); these walks make thousands on maps denser than real
    /// games (1867's generator, both policies), and the three neutrality pins make the rest.
    #[test]
    fn the_flat_walk_equals_the_reference_on_generated_1867_games() {
        use std::sync::atomic::Ordering::Relaxed;
        let before = GRAPH_CROSS_CHECKS.load(Relaxed);
        for (players, seed, policy) in [
            (3, 2053, crate::title::CrossPlayerPolicy::RULES),
            (4, 7, crate::title::CrossPlayerPolicy::RULES),
            (5, 11, crate::title::CrossPlayerPolicy::RESTRICTED),
        ] {
            let r = crate::g1867::fuzz::walk(players, seed, 2500, policy);
            assert!(r.error.is_none(), "{:?}", r.error);
        }
        assert!(GRAPH_CROSS_CHECKS.load(Relaxed) - before > 1000, "{}", GRAPH_CROSS_CHECKS.load(Relaxed) - before);
    }

    /// The same check over stored site games (a scratch run, `GRAPH_CORPUS=<dir or .json>[:...]`): at every position of
    /// every game, every company's graph computed afresh by both walks.
    #[cfg(feature = "site")]
    #[test]
    #[ignore]
    fn the_flat_walk_equals_the_reference_on_a_corpus() {
        let Ok(spec) = std::env::var("GRAPH_CORPUS") else { return };
        let mut files = Vec::new();
        for part in spec.split(':') {
            let p = std::path::Path::new(part);
            if p.is_dir() {
                for e in std::fs::read_dir(p).unwrap().flatten() {
                    if e.path().extension().is_some_and(|x| x == "json") {
                        files.push(e.path());
                    }
                }
            } else {
                files.push(p.to_path_buf());
            }
        }
        let (mut positions, mut graphs) = (0u64, 0u64);
        for f in files {
            let data: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&f).unwrap()).unwrap();
            let Ok(mut g) = crate::site::game::SiteGame::load(&data) else { continue };
            g.site_log = false;
            while !g.done() {
                g.process_next();
                positions += 1;
                let syms: Vec<String> = g.game.corporations.iter().filter(|c| !c.closed).map(|c| c.sym.clone()).collect();
                for sym in syms {
                    let tokens = g.game.corp_token_positions(&sym);
                    let reservations = g.game.home_reservations();
                    let flat = compute_corp_graph_flat(&sym, &g.game.hexes, &g.game.hex_idx, &g.game.hex_adjacency, &tokens, &reservations);
                    let reference = compute_corp_graph_reference(&sym, &g.game.hexes, &g.game.hex_idx, &g.game.hex_adjacency, &tokens, &reservations);
                    assert_same_graph(&flat, &reference, &format!("{} @{} {sym}", f.display(), g.current_action_id));
                    graphs += 1;
                }
            }
        }
        eprintln!("graph corpus: {positions} positions, {graphs} graphs equal");
    }
}
