//! A direct port of Ruby's `Graph#compute` / `Node#walk` / `Path#walk`
//! for the 1822 family (lib/engine/graph.rb, part/node.rb, part/path.rb).
//!
//! The shared `map.rs` walk is a port of the Python 1830 engine and skips
//! terminal paths outright; 1822 has tokenable TERMINAL cities (E2, H1,
//! C34, F23, A42, E28 — minor homes and destinations), port hexes whose
//! junction paths are terminal (never walked), and `ignore:1` paths at
//! Merthyr Tydfil / Pontypool. Ruby walks OUT of a node along a terminal
//! path and reaches a terminal city's node without passing through it;
//! this module follows those rules exactly. Lanes are not modelled
//! (`lanes:2` parses as one path): they never change which hexes or
//! nodes are reachable on the 1822 map, only route-overlap legality.

use std::collections::{BTreeMap, HashMap, HashSet};

use crate::game::BaseGame;
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
}

impl<'a> Ctx<'a> {
    fn path(&self, p: PathRef) -> &crate::tiles::PathDef {
        &self.game.hexes[p.0].tile.paths[p.1]
    }
    fn path_terminal(&self, p: PathRef) -> bool {
        self.path(p).terminal
    }
    fn path_ignore(&self, p: PathRef) -> bool {
        self.game.g1822_path_ignored(p.0, p.1)
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
            .map(|(i, _)| (n.hex, i))
            .collect()
    }
    /// Ruby Hex#paths[edge].
    fn hex_paths_at(&self, hex: usize, edge: u8) -> Vec<PathRef> {
        let tile = &self.game.hexes[hex].tile;
        (0..tile.paths.len())
            .filter(|&i| self.path_exits((hex, i)).contains(&edge))
            .map(|i| (hex, i))
            .collect()
    }
    fn junction_paths(&self, hex: usize) -> Vec<PathRef> {
        let tile = &self.game.hexes[hex].tile;
        (0..tile.paths.len())
            .filter(|&i| self.path_junction((hex, i)))
            .map(|i| (hex, i))
            .collect()
    }
    /// Tile#converging_exit?: more than one path exits through the edge.
    fn converging_exit(&self, hex: usize, edge: u8) -> bool {
        let tile = &self.game.hexes[hex].tile;
        (0..tile.paths.len())
            .map(|i| self.path_exits((hex, i)).iter().filter(|&&e| e == edge).count())
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
    counter_edges: HashMap<(usize, u8), i32>,
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
    if exits.iter().map(|&e| *w.counter_edges.get(&(p.0, e)).unwrap_or(&0)).sum::<i32>() > 0 {
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
        *w.counter_edges.entry((p.0, e)).or_insert(0) += 1;
        let np_edge = (e + 3) % 6;
        let conv = converging || ctx.converging_exit(p.0, e);
        for np in ctx.hex_paths_at(nb, np_edge) {
            path_walk(ctx, np, Some(np_edge), None, w, conv, origin, visited_nodes, origin_converging_path, found);
        }
        *w.counter_edges.entry((p.0, e)).or_insert(0) -= 1;
    }
    if converging {
        w.visited_paths.remove(&p);
    }
    if junction {
        *w.counter_junctions.entry(p.0).or_insert(0) -= 1;
    }
}

impl BaseGame {
    /// The token cities of `corp` (Ruby `city_tokened_by?`, any token type).
    pub(crate) fn g1822_token_nodes(&self, corp: &str) -> Vec<Node> {
        let mut out = Vec::new();
        for (hi, hex) in self.hexes.iter().enumerate() {
            for (ci, city) in hex.tile.cities.iter().enumerate() {
                if city.tokens.iter().flatten().any(|t| t.corporation_id == corp) {
                    out.push(Node { hex: hi, kind: NodeKind::City, index: ci });
                }
            }
        }
        out
    }

    /// Ruby Graph#compute(corporation).
    pub(crate) fn g1822_graph(&self, corp: &str) -> CorpGraph {
        let neighbors = self.g1822_neighbors();
        let ctx = Ctx {
            game: self,
            corp,
            neighbors: &neighbors,
        };
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
        let ctx = Ctx {
            game: self,
            corp,
            neighbors: &neighbors,
        };
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
                    let impassable_color = crate::title::g1822::hex_dsl(&self.hexes[ni].id)
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
        crate::title::g1822::hex_dsl(coord).map_or(false, |(_, c)| c == "blue")
    }

    /// Impassable border edges per hex (title data).
    pub(crate) fn g1822_impassable_edges(&self) -> HashMap<String, Vec<u8>> {
        let mut out: HashMap<String, Vec<u8>> = HashMap::new();
        for d in self.title_def().hex_definitions() {
            for b in d.borders {
                if b.impassable {
                    out.entry(d.coord.to_string()).or_default().push(b.edge);
                }
            }
        }
        out
    }

    /// Whether path `pi` of hex `hi` carries `ignore:1` (only the static
    /// gray E34/F33 paths do; laid tiles never).
    pub(crate) fn g1822_path_ignored(&self, hi: usize, pi: usize) -> bool {
        let hex = &self.hexes[hi];
        if !hex.tile.id.starts_with("preprinted_") {
            return false;
        }
        let Some((dsl, _)) = crate::title::g1822::hex_dsl(&hex.id) else { return false };
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
