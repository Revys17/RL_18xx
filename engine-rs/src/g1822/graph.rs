//! A direct port of Ruby's `Graph#compute` / `Node#walk` / `Path#walk`
//! for the 1822 family (lib/engine/graph.rb, part/node.rb, part/path.rb).
//!
//! The shared `map.rs` walk is a port of the Python 1830 engine and skips
//! terminal paths outright; 1822 has tokenable TERMINAL cities (E2, H1,
//! C34, F23, A42, E28 — minor homes and destinations), port hexes whose
//! junction paths are terminal (never walked), and `ignore:1` paths at
//! Merthyr Tydfil / Pontypool. Ruby walks OUT of a node along a terminal
//! path and reaches a terminal city's node without passing through it;
//! this module follows those rules exactly. 1822 does not model lanes
//! (`lanes:2` parses as one path): they never change which hexes or
//! nodes are reachable on the 1822 map, only route-overlap legality.
//! 1822PNW does (Ruby `Path.make_lanes`: lane copies `base + LANE_STRIDE ×
//! k`, lane-matched crossings, edge counters keyed by lane) — Portland's
//! `lanes:2` and P9's `b_lane` paths reach hexes one merged path cannot.

use std::collections::{BTreeMap, HashMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::game::BaseGame;
use crate::tiles::PathEndpoint;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum NodeKind {
    City,
    Town,
    Offboard,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Node {
    pub hex: usize,
    pub kind: NodeKind,
    pub index: usize,
}

pub(crate) type PathRef = (usize, usize);

/// What Ruby's Graph#compute produces for one corporation.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct CorpGraph {
    /// hex index → edges (connected_hexes: where track may be laid).
    pub connected_hexes: BTreeMap<usize, Vec<u8>>,
    /// connected_nodes (where tokens may be placed / routes run).
    pub connected_nodes: HashSet<Node>,
    /// reachable_hexes (hexes some walked path lies in).
    pub reachable_hexes: HashSet<usize>,
    /// connected_paths ((hex, path index) of every walked path).
    pub connected_paths: HashSet<PathRef>,
    pub route_available: bool,
    pub route_train_purchase: bool,
}

struct Ctx<'a> {
    game: &'a BaseGame,
    corp: &'a str,
    neighbors: &'a HashMap<usize, BTreeMap<u8, usize>>,
    /// 1822PNW: lanes modelled (Path.make_lanes copies, lane-matched hex
    /// crossings, per-lane edge counters); `lanes[hex][path]` = the lane
    /// specs (a end, b end) of each copy. None for 1822 (one path each).
    lanes: Option<Vec<Vec<Vec<LaneSpec>>>>,
}

/// A path copy's lanes at its (a, b) ends: (width, index) — Ruby
/// `Path#lanes`.
pub(crate) type LaneSpec = ((u8, u8), (u8, u8));

/// A lane copy's path index: base + LANE_STRIDE × copy.
const LANE_STRIDE: usize = 1000;

/// Ruby `Path.make_lanes` / `decode_lane_spec` for one `path=` DSL part.
pub(crate) fn lane_specs(attrs: &str) -> Vec<LaneSpec> {
    let mut lanes: Option<u8> = None;
    let mut a_lane = (1u8, 0u8);
    let mut b_lane = (1u8, 0u8);
    let mut a_edge = false;
    let mut b_edge = false;
    let decode = |v: &str| -> (u8, u8) {
        let (w, i) = v.split_once('.').unwrap_or((v, "0"));
        (w.parse().unwrap_or(1), i.parse().unwrap_or(0))
    };
    for kv in attrs.split(',') {
        let Some((k, v)) = kv.split_once(':') else { continue };
        match k {
            "lanes" => lanes = v.parse().ok(),
            "a_lane" => a_lane = decode(v),
            "b_lane" => b_lane = decode(v),
            "a" => a_edge = !v.starts_with('_') && v != "junction",
            "b" => b_edge = !v.starts_with('_') && v != "junction",
            _ => {}
        }
    }
    match lanes {
        Some(n) if n > 1 => (0..n)
            .map(|k| {
                let a = (n, k);
                let b = if a_edge && b_edge { (n, n - k - 1) } else { a };
                (a, b)
            })
            .collect(),
        _ => vec![(a_lane, b_lane)],
    }
}

/// Ruby `Path#lane_match?`.
fn lane_match(l0: (u8, u8), l1: (u8, u8)) -> bool {
    l1.0 == l0.0 && l1.1 == l0.0 - l0.1 - 1
}

impl<'a> Ctx<'a> {
    fn new(game: &'a BaseGame, corp: &'a str, neighbors: &'a HashMap<usize, BTreeMap<u8, usize>>) -> Ctx<'a> {
        let lanes = game.g1822_is_pnw().then(|| game.g1822_lane_table());
        Ctx { game, corp, neighbors, lanes }
    }
    fn base(p: PathRef) -> PathRef {
        (p.0, p.1 % LANE_STRIDE)
    }
    fn path(&self, p: PathRef) -> &crate::tiles::PathDef {
        &self.game.hexes[p.0].tile.paths[p.1 % LANE_STRIDE]
    }
    /// The copies of base path `pi` on `hex`.
    fn copies(&self, hex: usize, pi: usize) -> Vec<PathRef> {
        let n = self
            .lanes
            .as_ref()
            .and_then(|t| t.get(hex))
            .and_then(|v| v.get(pi))
            .map_or(1, |c| c.len().max(1));
        (0..n).map(|k| (hex, pi + k * LANE_STRIDE)).collect()
    }
    /// The lane of copy `p` at its exit `edge` ((1, 0) without lanes).
    fn exit_lane(&self, p: PathRef, edge: u8) -> (u8, u8) {
        let Some(t) = &self.lanes else { return (1, 0) };
        let spec = t
            .get(p.0)
            .and_then(|v| v.get(p.1 % LANE_STRIDE))
            .and_then(|c| c.get(p.1 / LANE_STRIDE))
            .copied()
            .unwrap_or(((1, 0), (1, 0)));
        let pd = self.path(p);
        if pd.a.edge_num() == Some(edge) {
            spec.0
        } else {
            spec.1
        }
    }
    fn path_terminal(&self, p: PathRef) -> bool {
        self.path(p).terminal
    }
    fn path_ignore(&self, p: PathRef) -> bool {
        self.game.g1822_path_ignored(p.0, p.1 % LANE_STRIDE)
    }
    fn path_junction(&self, p: PathRef) -> bool {
        let pd = self.path(p);
        pd.a == PathEndpoint::Junction || pd.b == PathEndpoint::Junction
    }
    fn path_exits(&self, p: PathRef) -> Vec<u8> {
        let pd = self.path(p);
        [&pd.a, &pd.b].iter().filter_map(|e| e.edge_num()).collect()
    }
    fn path_nodes(&self, p: PathRef) -> Vec<Node> {
        let pd = self.path(p);
        [&pd.a, &pd.b]
            .iter()
            .filter_map(|e| match e {
                PathEndpoint::City(i) => Some(Node { hex: p.0, kind: NodeKind::City, index: *i }),
                PathEndpoint::Town(i) => Some(Node { hex: p.0, kind: NodeKind::Town, index: *i }),
                PathEndpoint::Offboard(i) => Some(Node { hex: p.0, kind: NodeKind::Offboard, index: *i }),
                _ => None,
            })
            .collect()
    }
    fn node_paths(&self, n: Node) -> Vec<PathRef> {
        let tile = &self.game.hexes[n.hex].tile;
        tile.paths
            .iter()
            .enumerate()
            .filter(|(i, _)| self.path_nodes((n.hex, *i)).contains(&n))
            .flat_map(|(i, _)| self.copies(n.hex, i))
            .collect()
    }
    /// Ruby Hex#paths[edge] (a `track:thin` path never matches a walk's
    /// gauge: Path#tracks_match?).
    fn hex_paths_at(&self, hex: usize, edge: u8) -> Vec<PathRef> {
        let tile = &self.game.hexes[hex].tile;
        (0..tile.paths.len())
            .filter(|&i| self.path_exits((hex, i)).contains(&edge))
            .filter(|&i| !self.game.g1822_path_thin(hex, i))
            .flat_map(|i| self.copies(hex, i))
            .collect()
    }
    fn junction_paths(&self, hex: usize) -> Vec<PathRef> {
        let tile = &self.game.hexes[hex].tile;
        (0..tile.paths.len())
            .filter(|&i| self.path_junction((hex, i)))
            .flat_map(|i| self.copies(hex, i))
            .collect()
    }
    /// Tile#converging_exit?: more than one path exits through the edge.
    fn converging_exit(&self, hex: usize, edge: u8) -> bool {
        let tile = &self.game.hexes[hex].tile;
        (0..tile.paths.len())
            .flat_map(|i| self.copies(hex, i))
            .map(|p| self.path_exits(p).iter().filter(|&&e| e == edge).count())
            .sum::<usize>()
            > 1
    }
    /// City#blocks?(corporation).
    fn blocks(&self, n: Node) -> bool {
        if n.kind != NodeKind::City {
            return false;
        }
        let city = &self.game.hexes[n.hex].tile.cities[n.index];
        if city.tokens.iter().flatten().any(|t| t.corporation_id == self.corp) {
            return false;
        }
        if city.tokens.iter().any(|t| t.is_none()) {
            return false;
        }
        if city.tokens.iter().flatten().any(|t| t.token_type == "neutral") {
            return false;
        }
        true
    }
}

struct Walk {
    visited_paths: HashSet<PathRef>,
    /// Ruby `counter[edge.id]` — Edge#id carries the lane index.
    counter_edges: HashMap<(usize, u8, u8), i32>,
    counter_junctions: HashMap<usize, i32>,
}

/// One yielded path: the compute block records it.
struct Found<'b> {
    paths: &'b mut HashSet<PathRef>,
    nodes: &'b mut HashSet<Node>,
    local_nodes: &'b mut HashSet<Node>,
    hexes: &'b mut BTreeMap<usize, Vec<u8>>,
}

impl<'b> Found<'b> {
    fn record(&mut self, ctx: &Ctx, p: PathRef) {
        let p = Ctx::base(p);
        if !self.paths.insert(p) {
            return;
        }
        for n in ctx.path_nodes(p) {
            self.nodes.insert(n);
            self.local_nodes.insert(n);
        }
        for e in ctx.path_exits(p) {
            add_edge(self.hexes, p.0, e);
            if let Some(&nb) = ctx.neighbors.get(&p.0).and_then(|m| m.get(&e)) {
                add_edge(self.hexes, nb, (e + 3) % 6);
            }
        }
    }
}

fn add_edge(hexes: &mut BTreeMap<usize, Vec<u8>>, hex: usize, e: u8) {
    let v = hexes.entry(hex).or_default();
    if !v.contains(&e) {
        v.push(e);
    }
}

#[allow(clippy::too_many_arguments)]
fn node_walk(
    ctx: &Ctx,
    node: Node,
    visited_nodes: &mut HashSet<Node>,
    w: &mut Walk,
    converging_path: bool,
    found: &mut Found,
) {
    if visited_nodes.contains(&node) {
        return;
    }
    visited_nodes.insert(node);
    for np in ctx.node_paths(node) {
        if ctx.path_ignore(np) {
            continue;
        }
        path_walk(ctx, np, None, None, w, converging_path, node, visited_nodes, converging_path, found);
    }
    if converging_path {
        visited_nodes.remove(&node);
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
    visited_nodes: &mut HashSet<Node>,
    origin_converging_path: bool,
    found: &mut Found,
) {
    if w.visited_paths.contains(&p) {
        return;
    }
    let junction = ctx.path_junction(p);
    if junction && *w.counter_junctions.get(&p.0).unwrap_or(&0) > 1 {
        return;
    }
    let exits = ctx.path_exits(p);
    if exits.iter().map(|&e| *w.counter_edges.get(&(p.0, e, ctx.exit_lane(p, e).1)).unwrap_or(&0)).sum::<i32>() > 0 {
        return;
    }
    let terminal = ctx.path_terminal(p);
    if junction && terminal {
        return;
    }
    w.visited_paths.insert(p);
    if junction {
        *w.counter_junctions.entry(p.0).or_insert(0) += 1;
    }

    // -- the Node#walk block --
    found.record(ctx, p);
    if !terminal {
        for next in ctx.path_nodes(p) {
            if next == origin || ctx.blocks(next) {
                continue;
            }
            node_walk(ctx, next, visited_nodes, w, origin_converging_path || converging, found);
        }
    }

    if junction && jskip != Some(p.0) {
        for jp in ctx.junction_paths(p.0) {
            path_walk(ctx, jp, None, Some(p.0), w, converging, origin, visited_nodes, origin_converging_path, found);
        }
    }
    for e in exits {
        if Some(e) == skip {
            continue;
        }
        let Some(&nb) = ctx.neighbors.get(&p.0).and_then(|m| m.get(&e)) else { continue };
        let lane = ctx.exit_lane(p, e);
        *w.counter_edges.entry((p.0, e, lane.1)).or_insert(0) += 1;
        let np_edge = (e + 3) % 6;
        let conv = converging || ctx.converging_exit(p.0, e);
        for np in ctx.hex_paths_at(nb, np_edge) {
            if ctx.lanes.is_some() && !lane_match(lane, ctx.exit_lane(np, np_edge)) {
                continue;
            }
            path_walk(ctx, np, Some(np_edge), None, w, conv, origin, visited_nodes, origin_converging_path, found);
        }
        *w.counter_edges.entry((p.0, e, lane.1)).or_insert(0) -= 1;
    }
    if converging {
        w.visited_paths.remove(&p);
    }
    if junction {
        *w.counter_junctions.entry(p.0).or_insert(0) -= 1;
    }
}

impl BaseGame {
    /// The token cities of `corp` (Ruby `Game#city_tokened_by?`).
    pub(crate) fn g1822_token_nodes(&self, corp: &str) -> Vec<Node> {
        // Game#city_tokened_by?: 1822PNW counts :normal and :destination
        // tokens only (not the coal token).
        let pnw = self.g1822_is_pnw();
        let mut out = Vec::new();
        for (hi, hex) in self.hexes.iter().enumerate() {
            for (ci, city) in hex.tile.cities.iter().enumerate() {
                if city.tokens.iter().flatten().any(|t| t.corporation_id == corp && !(pnw && t.token_type == "coal")) {
                    out.push(Node { hex: hi, kind: NodeKind::City, index: ci });
                }
            }
        }
        out
    }

    /// Ruby `Graph#connected_*(corporation)`: 1822PNW answers from Ruby's
    /// cache when Rust mirrored the compute that filled it (see
    /// `G1822State::ruby_graphs`), else a fresh compute.
    pub(crate) fn g1822_graph(&self, corp: &str) -> CorpGraph {
        if self.g1822_is_pnw() {
            if let Some(g) = self.g().ruby_graphs.get(corp) {
                return g;
            }
        }
        self.g1822_graph_fresh(corp)
    }

    /// A full Ruby Graph#compute whose result Ruby keeps until its next
    /// Graph#clear (1822PNW only).
    pub(crate) fn g1822_ruby_graph_store(&self, corp: &str) {
        if self.g1822_is_pnw() && !self.g().ruby_graphs.contains(corp) {
            let g = self.g1822_graph_fresh(corp);
            self.g().ruby_graphs.insert(corp, g);
        }
    }

    /// Ruby Graph#clear / clear_graph_for_entity (1822PNW's cache).
    pub(crate) fn g1822_ruby_graph_clear(&self) {
        if self.g1822_is_pnw() {
            self.g().ruby_graphs.clear();
        }
    }

    /// Ruby Graph#compute(corporation) on the current map.
    pub(crate) fn g1822_graph_fresh(&self, corp: &str) -> CorpGraph {
        let neighbors = self.g1822_neighbors();
        let ctx = Ctx::new(self, corp, &neighbors);
        let mut hexes: BTreeMap<usize, Vec<u8>> = BTreeMap::new();
        let mut nodes: HashSet<Node> = HashSet::new();
        let tokens = self.g1822_token_nodes(corp);
        for t in &tokens {
            if let Some(nbs) = neighbors.get(&t.hex) {
                for &e in nbs.keys() {
                    add_edge(&mut hexes, t.hex, e);
                }
            }
            nodes.insert(*t);
        }
        let mut paths: HashSet<PathRef> = HashSet::new();
        let mut route_available = false;
        let mut route_train_purchase = false;
        for &t in &tokens {
            let mut visited_nodes: HashSet<Node> = tokens.iter().copied().filter(|n| *n != t).collect();
            let mut local_nodes: HashSet<Node> = HashSet::new();
            let mut w = Walk {
                visited_paths: HashSet::new(),
                counter_edges: HashMap::new(),
                counter_junctions: HashMap::new(),
            };
            {
                let mut found = Found {
                    paths: &mut paths,
                    nodes: &mut nodes,
                    local_nodes: &mut local_nodes,
                    hexes: &mut hexes,
                };
                node_walk(&ctx, t, &mut visited_nodes, &mut w, false, &mut found);
            }
            if route_train_purchase {
                continue;
            }
            // Every revenue center is :mandatory (RevenueCenter default).
            let mandatory = local_nodes.len();
            if mandatory > 1 {
                route_available = true;
                route_train_purchase = true;
            }
        }
        let reachable_hexes: HashSet<usize> = paths.iter().map(|p| p.0).collect();
        CorpGraph {
            connected_hexes: hexes,
            connected_nodes: nodes,
            reachable_hexes,
            connected_paths: paths,
            route_available,
            route_train_purchase,
        }
    }

    /// Ruby `node.walk(corporation:)` from one node: every path walked.
    pub(crate) fn g1822_walk_from(&self, corp: &str, start: Node) -> HashSet<PathRef> {
        let neighbors = self.g1822_neighbors();
        let ctx = Ctx::new(self, corp, &neighbors);
        let mut paths: HashSet<PathRef> = HashSet::new();
        let mut nodes: HashSet<Node> = HashSet::new();
        let mut local: HashSet<Node> = HashSet::new();
        let mut hexes: BTreeMap<usize, Vec<u8>> = BTreeMap::new();
        let mut visited: HashSet<Node> = HashSet::new();
        let mut w = Walk {
            visited_paths: HashSet::new(),
            counter_edges: HashMap::new(),
            counter_junctions: HashMap::new(),
        };
        let mut found = Found {
            paths: &mut paths,
            nodes: &mut nodes,
            local_nodes: &mut local,
            hexes: &mut hexes,
        };
        node_walk(&ctx, start, &mut visited, &mut w, false, &mut found);
        paths
    }

    /// 1822PNW: every hex's per-path lane copies (from the preprinted or
    /// the laid tile's DSL).
    pub(crate) fn g1822_lane_table(&self) -> Vec<Vec<Vec<LaneSpec>>> {
        let f = self.fam();
        self.hexes
            .iter()
            .map(|hex| {
                let dsl = if hex.tile.id.starts_with("preprinted_") {
                    (f.hex_dsl)(&hex.id).map(|(d, _)| d)
                } else {
                    let base = hex.tile.name.split('-').next().unwrap_or(&hex.tile.name);
                    (f.tile_dsl)(base).map(|(d, _)| d)
                };
                let Some(dsl) = dsl else { return vec![] };
                dsl.split(';').filter_map(|p| p.trim().strip_prefix("path=")).map(lane_specs).collect()
            })
            .collect()
    }

    /// The nodes on path `p` (hex, kind, index).
    pub(crate) fn g1822_path_nodes(&self, p: PathRef) -> Vec<Node> {
        let pd = &self.hexes[p.0].tile.paths[p.1];
        [&pd.a, &pd.b]
            .iter()
            .filter_map(|e| match e {
                PathEndpoint::City(i) => Some(Node { hex: p.0, kind: NodeKind::City, index: *i }),
                PathEndpoint::Town(i) => Some(Node { hex: p.0, kind: NodeKind::Town, index: *i }),
                PathEndpoint::Offboard(i) => Some(Node { hex: p.0, kind: NodeKind::Offboard, index: *i }),
                _ => None,
            })
            .collect()
    }

    /// Ruby `hex.neighbors` (connect_hexes): adjacency minus neighbors of an
    /// impassable colour (gray/red/blue) that don't target the hex, minus
    /// edges behind the hex's OWN impassable borders (asymmetric, as Ruby).
    pub(crate) fn g1822_neighbors(&self) -> HashMap<usize, BTreeMap<u8, usize>> {
        if self.g1822_is_pnw() || self.g1822_is_mx() {
            return self.g1822_pnw_neighbors();
        }
        let mut out: HashMap<usize, BTreeMap<u8, usize>> = HashMap::new();
        let walls = self.g1822_impassable_edges();
        for (hi, hex) in self.hexes.iter().enumerate() {
            let mut m = BTreeMap::new();
            if let Some(adj) = self.hex_adjacency.get(&hex.id) {
                for (&dir, nid) in adj {
                    let Some(&ni) = self.hex_idx.get(nid) else { continue };
                    let ntile = &self.hexes[ni].tile;
                    // connect_hexes runs once at setup: the PREPRINTED colour
                    // decides (a hex later upgraded to a gray tile, e.g. P2's
                    // 169, stays a neighbour).
                    let impassable_color = (self.fam().hex_dsl)(&self.hexes[ni].id)
                        .map_or(false, |(_, c)| matches!(c, "gray" | "red" | "blue"));
                    if impassable_color {
                        let inv = (dir + 3) % 6;
                        let targets = ntile.paths.iter().any(|p| {
                            p.a.edge_num() == Some(inv) || p.b.edge_num() == Some(inv)
                        });
                        if !targets {
                            continue;
                        }
                    }
                    if walls.get(&hex.id).map_or(false, |w| w.contains(&dir)) {
                        continue;
                    }
                    m.insert(dir, ni);
                }
            }
            out.insert(hi, m);
        }
        out
    }

    pub(crate) fn g1822_hex_is_blue(&self, coord: &str) -> bool {
        if self.g1822_is_pnw() || self.g1822_is_mx() {
            return self.hex_idx.get(coord).map_or(false, |&hi| self.g1822_tile_color_name(hi) == "blue");
        }
        (self.fam().hex_dsl)(coord).map_or(false, |(_, c)| c == "blue")
    }

    /// Impassable border edges per hex: the game's `walls`
    /// (`title::title_walls`, shared with every title — Ruby
    /// `connect_hexes` drops these edges from `hex.neighbors`).
    pub(crate) fn g1822_impassable_edges(&self) -> std::sync::Arc<HashMap<String, Vec<u8>>> {
        std::sync::Arc::clone(&self.walls)
    }

    /// Whether path `pi` of hex `hi` is `track:thin` (1822PNW's D9 / E8
    /// ferry lines; preprinted only).
    pub(crate) fn g1822_path_thin(&self, hi: usize, pi: usize) -> bool {
        if !(self.g1822_is_pnw() || self.g1822_is_mx()) {
            return false;
        }
        let hex = &self.hexes[hi];
        if !hex.tile.id.starts_with("preprinted_") {
            return false;
        }
        let Some((dsl, _)) = (self.fam().hex_dsl)(&hex.id) else { return false };
        dsl.split(';')
            .filter(|part| part.trim().starts_with("path="))
            .nth(pi)
            .map_or(false, |part| part.contains("track:thin"))
    }

    /// Whether path `pi` of hex `hi` carries `ignore:1` (only the static
    /// gray E34/F33 paths do; laid tiles never).
    pub(crate) fn g1822_path_ignored(&self, hi: usize, pi: usize) -> bool {
        let hex = &self.hexes[hi];
        if !hex.tile.id.starts_with("preprinted_") {
            return false;
        }
        let Some((dsl, _)) = (self.fam().hex_dsl)(&hex.id) else { return false };
        dsl.split(';')
            .filter(|part| part.trim().starts_with("path="))
            .nth(pi)
            .map_or(false, |part| part.contains("ignore:1"))
    }

    /// A Ruby Graph#compute for `corp` happened here: latch a
    /// purchase-grade route (see G1822State::route_purchase).
    pub(crate) fn g1822_note_graph(&self, corp: &str) {
        if self.g().route_purchase.contains(corp) {
            return;
        }
        if self.g1822_graph(corp).route_train_purchase {
            self.g().route_purchase.insert(corp);
        }
    }

    /// Ruby Graph#can_token?(corporation): some connected city is
    /// tokenable (free: true) by the corporation.
    pub(crate) fn g1822_can_token(&self, corp: &str) -> bool {
        let g = self.g1822_graph(corp);
        // A full Graph#compute also caches route_info (see route_purchase).
        if g.route_train_purchase {
            self.g().route_purchase.insert(corp);
        }
        g.connected_nodes
            .iter()
            .filter(|n| n.kind == NodeKind::City)
            .any(|n| self.g1822_city_tokenable(corp, n.hex, n.index, true, false))
    }
}

#[cfg(test)]
mod walls_tests {
    use std::collections::HashMap;

    use crate::title::GameTitle;

    /// The impassable borders 1822-family code read itself before it moved
    /// onto `BaseGame.walls` (the HexDef `impassable` flags) are exactly
    /// `title::title_walls`' (which also reads `border=edge:N,type:
    /// impassable` from the preprinted DSL): the move changed nothing.
    #[test]
    fn g1822_family_walls_are_title_walls() {
        let titles: [&dyn GameTitle; 2] = [&crate::title::g1822::G1822, &crate::title::g1822pnw::G1822Pnw];
        for t in titles {
            let mut own: HashMap<String, Vec<u8>> = HashMap::new();
            for d in t.hex_definitions() {
                for b in d.borders {
                    if b.impassable {
                        own.entry(d.coord.to_string()).or_default().push(b.edge);
                    }
                }
            }
            for v in own.values_mut() {
                v.sort_unstable();
                v.dedup();
            }
            let walls = crate::title::title_walls(t);
            assert!(!walls.is_empty(), "{}: no walls", t.name());
            assert_eq!(own, walls, "{}", t.name());
        }
    }
}
