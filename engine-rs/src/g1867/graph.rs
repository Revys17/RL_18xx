//! Ruby's `Graph#compute` / `Node#walk` / `Path#walk` (lib/engine/graph.rb,
//! part/node.rb, part/path.rb) for 1867's legal-move generator — the 1822
//! agent's port (`g1822/graph.rs`) with 1867's neighbours (walls from
//! title data, impassable gray/red/blue hexes), on flat arrays (the Track
//! generator walks once per upgrade candidate).
//!
//! The shared `map.rs` walk (a port of the Python 1830 engine) is what the
//! REPLAY path uses; it treats track as undirected reachability. Ruby's
//! walk visits each path at most once per token walk (un-visiting only
//! behind converging exits), so a path reachable only by re-crossing a
//! hexside already walked is NOT connected — exactly what
//! `check_track_restrictions!` ("Must use new track or change city value")
//! and the lay frontier read. The generator uses this graph for those.

use std::collections::{BTreeMap, HashMap, HashSet};

use crate::game::BaseGame;
use crate::graph::Tile;
use crate::tiles::PathEndpoint;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum NodeKind {
    City,
    Town,
    Offboard,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Node {
    pub hex: usize,
    pub kind: NodeKind,
    pub index: usize,
}

pub(crate) type PathRef = (usize, usize);

/// What Ruby's Graph#compute produces for one corporation.
#[derive(Clone, Debug, Default)]
pub struct CorpGraph {
    /// hex index → edges (connected_hexes: where track may be laid).
    pub connected_hexes: BTreeMap<usize, Vec<u8>>,
    /// connected_nodes (where tokens may be placed / routes run).
    pub connected_nodes: HashSet<Node>,
    /// connected_paths ((hex, path index) of every walked path).
    pub connected_paths: HashSet<PathRef>,
}

/// Max paths per tile the flat arrays index (1867's richest tiles have 7).
const MAXP: usize = 16;

fn endpoint_node(e: &PathEndpoint, hex: usize) -> Option<Node> {
    match e {
        PathEndpoint::City(i) => Some(Node { hex, kind: NodeKind::City, index: *i }),
        PathEndpoint::Town(i) => Some(Node { hex, kind: NodeKind::Town, index: *i }),
        PathEndpoint::Offboard(i) => Some(Node { hex, kind: NodeKind::Offboard, index: *i }),
        _ => None,
    }
}

struct Ctx<'a> {
    game: &'a BaseGame,
    corp: &'a str,
    neighbors: &'a HashMap<usize, BTreeMap<u8, usize>>,
    /// A hex whose tile is replaced for this walk (a candidate lay).
    over: Option<(usize, &'a Tile)>,
}

impl<'a> Ctx<'a> {
    #[inline]
    fn tile(&self, hex: usize) -> &'a Tile {
        match self.over {
            Some((h, t)) if h == hex => t,
            _ => &self.game.hexes[hex].tile,
        }
    }
    #[inline]
    fn path(&self, p: PathRef) -> &'a crate::tiles::PathDef {
        &self.tile(p.0).paths[p.1]
    }
    #[inline]
    fn path_junction(&self, p: PathRef) -> bool {
        let pd = self.path(p);
        pd.a == PathEndpoint::Junction || pd.b == PathEndpoint::Junction
    }
    #[inline]
    fn path_exits(&self, p: PathRef) -> [Option<u8>; 2] {
        let pd = self.path(p);
        [pd.a.edge_num(), pd.b.edge_num()]
    }
    #[inline]
    fn path_nodes(&self, p: PathRef) -> [Option<Node>; 2] {
        let pd = self.path(p);
        [endpoint_node(&pd.a, p.0), endpoint_node(&pd.b, p.0)]
    }
    #[inline]
    fn path_has_exit(&self, p: PathRef, edge: u8) -> bool {
        self.path_exits(p).iter().any(|e| *e == Some(edge))
    }
    /// Tile#converging_exit?: more than one path exits through the edge.
    fn converging_exit(&self, hex: usize, edge: u8) -> bool {
        let tile = self.tile(hex);
        (0..tile.paths.len())
            .map(|i| self.path_exits((hex, i)).iter().filter(|e| **e == Some(edge)).count())
            .sum::<usize>()
            > 1
    }
    /// Node#blocks?(corporation): City#blocks? (all slots others' non-
    /// neutral tokens), Offboard#blocks? (always), towns never.
    fn blocks(&self, n: Node) -> bool {
        match n.kind {
            NodeKind::Offboard => true,
            NodeKind::Town => false,
            NodeKind::City => {
                let city = &self.tile(n.hex).cities[n.index];
                !(city.tokens.iter().flatten().any(|t| t.corporation_id == self.corp)
                    || city.tokens.iter().any(|t| t.is_none())
                    || city.tokens.iter().flatten().any(|t| t.token_type == "neutral"))
            }
        }
    }
}

struct Walk {
    visited_paths: Vec<bool>,
    counter_edges: Vec<i32>,
    counter_junctions: Vec<i32>,
}

/// The compute block's accumulators.
struct Found {
    seen: Vec<bool>,
    paths: Vec<PathRef>,
    nodes: HashSet<Node>,
}

impl Found {
    fn record(&mut self, ctx: &Ctx, p: PathRef) {
        let k = p.0 * MAXP + p.1;
        if self.seen[k] {
            return;
        }
        self.seen[k] = true;
        self.paths.push(p);
        for n in ctx.path_nodes(p).into_iter().flatten() {
            self.nodes.insert(n);
        }
    }
}

fn node_walk(ctx: &Ctx, node: Node, visited_nodes: &mut Vec<Node>, w: &mut Walk, converging_path: bool, found: &mut Found) {
    if visited_nodes.contains(&node) {
        return;
    }
    visited_nodes.push(node);
    let n_paths = ctx.tile(node.hex).paths.len();
    for i in 0..n_paths {
        let np = (node.hex, i);
        if ctx.path_nodes(np).contains(&Some(node)) {
            path_walk(ctx, np, None, None, w, converging_path, node, visited_nodes, converging_path, found);
        }
    }
    if converging_path {
        if let Some(pos) = visited_nodes.iter().position(|n| *n == node) {
            visited_nodes.swap_remove(pos);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn path_walk(
    ctx: &Ctx,
    p: PathRef,
    skip: Option<u8>,
    jskip: Option<usize>,
    w: &mut Walk,
    converging: bool,
    origin: Node,
    visited_nodes: &mut Vec<Node>,
    origin_converging_path: bool,
    found: &mut Found,
) {
    let pk = p.0 * MAXP + p.1;
    if w.visited_paths[pk] {
        return;
    }
    let junction = ctx.path_junction(p);
    if junction && w.counter_junctions[p.0] > 1 {
        return;
    }
    let exits = ctx.path_exits(p);
    if exits.iter().flatten().map(|&e| w.counter_edges[p.0 * 6 + e as usize]).sum::<i32>() > 0 {
        return;
    }
    let terminal = ctx.path(p).terminal;
    if junction && terminal {
        return;
    }
    w.visited_paths[pk] = true;
    if junction {
        w.counter_junctions[p.0] += 1;
    }

    // -- the Node#walk block --
    found.record(ctx, p);
    if !terminal {
        for next in ctx.path_nodes(p).into_iter().flatten() {
            if next == origin || ctx.blocks(next) {
                continue;
            }
            node_walk(ctx, next, visited_nodes, w, origin_converging_path || converging, found);
        }
    }

    if junction && jskip != Some(p.0) {
        let n_paths = ctx.tile(p.0).paths.len();
        for j in 0..n_paths {
            if ctx.path_junction((p.0, j)) {
                path_walk(ctx, (p.0, j), None, Some(p.0), w, converging, origin, visited_nodes, origin_converging_path, found);
            }
        }
    }
    for e in exits.into_iter().flatten() {
        if Some(e) == skip {
            continue;
        }
        let Some(&nb) = ctx.neighbors.get(&p.0).and_then(|m| m.get(&e)) else { continue };
        w.counter_edges[p.0 * 6 + e as usize] += 1;
        let np_edge = (e + 3) % 6;
        let conv = converging || ctx.converging_exit(p.0, e);
        let n_paths = ctx.tile(nb).paths.len();
        for q in 0..n_paths {
            if ctx.path_has_exit((nb, q), np_edge) {
                path_walk(ctx, (nb, q), Some(np_edge), None, w, conv, origin, visited_nodes, origin_converging_path, found);
            }
        }
        w.counter_edges[p.0 * 6 + e as usize] -= 1;
    }
    if converging {
        w.visited_paths[pk] = false;
    }
    if junction {
        w.counter_junctions[p.0] -= 1;
    }
}

impl BaseGame {
    /// The token cities of `corp` (Ruby `city_tokened_by?`).
    pub(crate) fn g1867_token_nodes(&self, corp: &str) -> Vec<Node> {
        self.g1867_token_nodes_with(corp, None)
    }

    fn g1867_token_nodes_with(&self, corp: &str, over: Option<(usize, &Tile)>) -> Vec<Node> {
        let mut out = Vec::new();
        for (hi, hex) in self.hexes.iter().enumerate() {
            let tile = match over {
                Some((h, t)) if h == hi => t,
                _ => &hex.tile,
            };
            for (ci, city) in tile.cities.iter().enumerate() {
                if city.tokens.iter().flatten().any(|t| t.corporation_id == corp) {
                    out.push(Node { hex: hi, kind: NodeKind::City, index: ci });
                }
            }
        }
        out
    }

    /// Ruby Graph#compute(corporation) (no home-as-token, no teleports —
    /// 1867 has neither).
    pub(crate) fn g1867_graph(&self, corp: &str) -> CorpGraph {
        self.g1867_graph_with(corp, None)
    }

    /// Graph#compute with hex `over.0`'s tile replaced by `over.1` (the
    /// state right after a candidate lay, without applying it).
    pub(crate) fn g1867_graph_with(&self, corp: &str, over: Option<(usize, &Tile)>) -> CorpGraph {
        let neighbors = self.g1867_neighbors();
        let ctx = Ctx { game: self, corp, neighbors, over };
        let n = self.hexes.len();
        debug_assert!(self.hexes.iter().all(|h| h.tile.paths.len() <= MAXP));
        let tokens = self.g1867_token_nodes_with(corp, over);
        let mut found = Found { seen: vec![false; n * MAXP], paths: Vec::new(), nodes: HashSet::new() };
        let mut w = Walk { visited_paths: vec![false; n * MAXP], counter_edges: vec![0; n * 6], counter_junctions: vec![0; n] };
        for &t in &tokens {
            found.nodes.insert(t);
            let mut visited_nodes: Vec<Node> = tokens.iter().copied().filter(|x| *x != t).collect();
            w.visited_paths.iter_mut().for_each(|v| *v = false);
            node_walk(&ctx, t, &mut visited_nodes, &mut w, false, &mut found);
        }
        // connected_hexes: every tokened hex on all its neighbour edges,
        // every walked path's exits and the inverse edge across each.
        let mut hexes: BTreeMap<usize, Vec<u8>> = BTreeMap::new();
        let mut add = |h: usize, e: u8| {
            let v = hexes.entry(h).or_default();
            if !v.contains(&e) {
                v.push(e);
            }
        };
        for t in &tokens {
            if let Some(nbs) = neighbors.get(&t.hex) {
                for &e in nbs.keys() {
                    add(t.hex, e);
                }
            }
        }
        for &p in &found.paths {
            for e in ctx.path_exits(p).into_iter().flatten() {
                add(p.0, e);
                if let Some(&nb) = neighbors.get(&p.0).and_then(|m| m.get(&e)) {
                    add(nb, (e + 3) % 6);
                }
            }
        }
        CorpGraph {
            connected_hexes: hexes,
            connected_nodes: found.nodes,
            connected_paths: found.paths.into_iter().collect(),
        }
    }

    /// Ruby `hex.neighbors` (connect_hexes): adjacency minus neighbours of
    /// an impassable colour (gray/red/blue) that don't target the hex,
    /// minus edges behind the hex's own walls. Ruby computes it once at
    /// setup from the PREPRINTED tiles (1867's gray, red and blue hexes are
    /// never laid on), so it is static title data: computed once per
    /// process.
    pub(crate) fn g1867_neighbors(&self) -> &'static HashMap<usize, BTreeMap<u8, usize>> {
        static CACHE: std::sync::OnceLock<HashMap<usize, BTreeMap<u8, usize>>> = std::sync::OnceLock::new();
        debug_assert_eq!(self.title, "1867");
        CACHE.get_or_init(|| self.g1867_neighbors_uncached())
    }

    fn g1867_neighbors_uncached(&self) -> HashMap<usize, BTreeMap<u8, usize>> {
        let mut out: HashMap<usize, BTreeMap<u8, usize>> = HashMap::new();
        for (hi, hex) in self.hexes.iter().enumerate() {
            let mut m = BTreeMap::new();
            if let Some(adj) = self.hex_adjacency.get(&hex.id) {
                for (&dir, nid) in adj {
                    let Some(&ni) = self.hex_idx.get(nid) else { continue };
                    let ntile = &self.hexes[ni].tile;
                    let impassable = self
                        .title_def()
                        .preprinted_hex_dsl(&self.hexes[ni].id)
                        .map_or(false, |(_, c)| matches!(c, "gray" | "red" | "blue"));
                    if impassable {
                        let inv = (dir + 3) % 6;
                        let targets =
                            ntile.paths.iter().any(|p| p.a.edge_num() == Some(inv) || p.b.edge_num() == Some(inv));
                        if !targets {
                            continue;
                        }
                    }
                    if self.walls.get(&hex.id).map_or(false, |w| w.contains(&dir)) {
                        continue;
                    }
                    m.insert(dir, ni);
                }
            }
            out.insert(hi, m);
        }
        out
    }
}
