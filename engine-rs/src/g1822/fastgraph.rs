//! Ruby's `Graph#compute` for the 1822 family on per-state arrays (the
//! generator speed-up of 2026-11-14): the same walk as `graph.rs`
//! (`node_walk` / `path_walk`, the same calls in the same order, the same
//! visited / counter rules), with everything the walk asks of a path, a
//! node or a hex answered from tables instead of re-derived (and
//! re-allocated) at every step: a path copy's exits, lanes at each exit,
//! nodes and flags; a hex's paths per edge, its junction paths, its
//! converging exits; the paths through each node. A hex's table depends on
//! its tile alone (paths, id, name, revenue-node counts) and the title's
//! static data, so it is cached per (title, hex, tile) and a compute only
//! looks the tables up. The `ignore:1` / `track:thin` flags of the
//! preprinted tiles and the lane specs of the DSL are read once per title.
//!
//! `graph.rs`'s walk stays the reference (`g1822::legal::reference`, the
//! differential tests; `BaseGame::g1822_graph_check_with` compares the two).

use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::{Arc, Mutex, OnceLock};

use super::graph::{lane_match, lane_specs, CorpGraph, LaneSpec, Neighbors, Node, NodeKind};
use crate::game::BaseGame;
use crate::tiles::{PathDef, PathEndpoint};

/// One path copy (Ruby `Path.make_lanes`: a `lanes:n` path is n paths).
#[derive(Clone)]
struct Copy {
    hex: usize,
    /// The base path's index on the tile.
    pi: usize,
    /// `path.exits` in [a, b] order, and the copy's lane at each.
    exits: [u8; 2],
    lanes: [(u8, u8); 2],
    n_exits: u8,
    /// The revenue nodes at its ends, [a, b] order.
    nodes: [Node; 2],
    n_nodes: u8,
    terminal: bool,
    junction: bool,
    ignore: bool,
}

/// What the walk reads of one hex: its copies (local ids, in path then
/// copy order) and the per-edge / per-node / junction lists over them.
struct HexTopo {
    copies: Vec<Copy>,
    /// Per edge: the non-thin copies with that exit and their lane at it
    /// (Ruby `hex.paths[edge]`).
    at_edge: [Vec<(u32, (u8, u8))>; 6],
    junctions: Vec<u32>,
    /// `Tile#converging_exit?` for each edge.
    converging: [bool; 6],
    /// (node, the copies whose ends include it).
    node_paths: Vec<(Node, Vec<u32>)>,
    n_base: usize,
    /// Node id ranges: cities, towns, offboards.
    dims: (usize, usize, usize),
    max_lane: usize,
}

/// What a hex's table is a function of (besides the title and the hex).
struct HexKey {
    id: String,
    name: String,
    paths: Vec<PathDef>,
    dims: (usize, usize, usize),
}

impl HexKey {
    fn matches(&self, t: &crate::graph::Tile) -> bool {
        self.id == t.id
            && self.name == t.name
            && self.dims == (t.cities.len(), t.towns.len(), t.offboards.len())
            && self.paths == t.paths
    }
}

/// Everything the walk reads of the map, for one state.
struct Topo {
    hexes: Vec<Arc<HexTopo>>,
    /// Per hex: the first copy id / base path id / node id.
    copy_off: Vec<usize>,
    base_off: Vec<usize>,
    node_off: Vec<usize>,
    copies_total: usize,
    nodes_total: usize,
    max_lane: usize,
}

/// The preprinted tiles' per-path flags of a title: (thin, ignore) per
/// `path=` part of the hex's DSL.
type PathFlags = Vec<Vec<(bool, bool)>>;

fn preprinted_flags(g: &BaseGame) -> Arc<PathFlags> {
    static TABLES: OnceLock<Mutex<HashMap<&'static str, Arc<PathFlags>>>> = OnceLock::new();
    let title = g.title_def().name();
    if let Some(t) = TABLES.get_or_init(Default::default).lock().ok().and_then(|m| m.get(title).cloned()) {
        if t.len() == g.hexes.len() {
            return t;
        }
    }
    let thin_title = g.g1822_is_pnw() || g.g1822_is_mx();
    let f = g.fam();
    let t: PathFlags = g
        .hexes
        .iter()
        .map(|hex| {
            let Some((dsl, _)) = (f.hex_dsl)(&hex.id) else { return Vec::new() };
            dsl.split(';')
                .filter(|part| part.trim().starts_with("path="))
                .map(|part| (thin_title && part.contains("track:thin"), part.contains("ignore:1")))
                .collect()
        })
        .collect();
    let t = Arc::new(t);
    if let Ok(mut m) = TABLES.get_or_init(Default::default).lock() {
        m.insert(title, Arc::clone(&t));
    }
    t
}

/// `g1822_lane_table()` for one hex (the preprinted hex's DSL, or the laid
/// tile's by name).
fn lanes_of(g: &BaseGame, hi: usize) -> Vec<Vec<LaneSpec>> {
    let hex = &g.hexes[hi];
    let f = g.fam();
    let dsl = if hex.tile.id.starts_with("preprinted_") {
        (f.hex_dsl)(&hex.id).map(|(d, _)| d)
    } else {
        let base = hex.tile.name.split('-').next().unwrap_or(&hex.tile.name);
        (f.tile_dsl)(base).map(|(d, _)| d)
    };
    match dsl {
        Some(dsl) => dsl.split(';').filter_map(|p| p.trim().strip_prefix("path=")).map(lane_specs).collect(),
        None => Vec::new(),
    }
}

fn end_node(hex: usize, e: &PathEndpoint) -> Option<Node> {
    match e {
        PathEndpoint::City(i) => Some(Node { hex, kind: NodeKind::City, index: *i }),
        PathEndpoint::Town(i) => Some(Node { hex, kind: NodeKind::Town, index: *i }),
        PathEndpoint::Offboard(i) => Some(Node { hex, kind: NodeKind::Offboard, index: *i }),
        _ => None,
    }
}

const NO_NODE: Node = Node { hex: usize::MAX, kind: NodeKind::City, index: 0 };

impl HexTopo {
    fn build(g: &BaseGame, hi: usize, flags: &PathFlags) -> HexTopo {
        let with_lanes = g.g1822_is_pnw() || g.g1822_is_ca();
        let tile = &g.hexes[hi].tile;
        // node ids: cities, then towns, then offboards (sized to cover
        // every index a path names)
        let (mut nc, mut nt, mut no) = (tile.cities.len(), tile.towns.len(), tile.offboards.len());
        for p in &tile.paths {
            for e in [&p.a, &p.b] {
                match e {
                    PathEndpoint::City(i) => nc = nc.max(i + 1),
                    PathEndpoint::Town(i) => nt = nt.max(i + 1),
                    PathEndpoint::Offboard(i) => no = no.max(i + 1),
                    _ => {}
                }
            }
        }
        let preprinted = tile.id.starts_with("preprinted_");
        let lanes = if with_lanes { Some(lanes_of(g, hi)) } else { None };
        let empty = Vec::new();
        let hflags = flags.get(hi).unwrap_or(&empty);
        let mut copies: Vec<Copy> = Vec::new();
        let mut at_edge: [Vec<(u32, (u8, u8))>; 6] = Default::default();
        let mut junctions = Vec::new();
        let mut exit_count = [0usize; 6];
        let mut node_paths: Vec<(Node, Vec<u32>)> = Vec::new();
        let mut max_lane = 0usize;
        for (pi, pd) in tile.paths.iter().enumerate() {
            let n_copies = lanes.as_ref().and_then(|t| t.get(pi)).map_or(1, |c| c.len().max(1));
            let (thin, ignore) = if preprinted { hflags.get(pi).copied().unwrap_or((false, false)) } else { (false, false) };
            let mut exits = [0u8; 2];
            let mut n_exits = 0u8;
            for e in [&pd.a, &pd.b] {
                if let Some(x) = e.edge_num() {
                    exits[n_exits as usize] = x;
                    n_exits += 1;
                }
            }
            let mut nodes = [NO_NODE; 2];
            let mut n_nodes = 0u8;
            for e in [&pd.a, &pd.b] {
                if let Some(x) = end_node(hi, e) {
                    nodes[n_nodes as usize] = x;
                    n_nodes += 1;
                }
            }
            let junction = pd.a == PathEndpoint::Junction || pd.b == PathEndpoint::Junction;
            for k in 0..n_copies {
                let spec =
                    lanes.as_ref().and_then(|t| t.get(pi)).and_then(|c| c.get(k)).copied().unwrap_or(((1, 0), (1, 0)));
                let mut ls = [(1u8, 0u8); 2];
                if with_lanes {
                    for s in 0..n_exits as usize {
                        ls[s] = if pd.a.edge_num() == Some(exits[s]) { spec.0 } else { spec.1 };
                        max_lane = max_lane.max(ls[s].1 as usize);
                    }
                }
                let id = copies.len() as u32;
                copies.push(Copy {
                    hex: hi,
                    pi,
                    exits,
                    lanes: ls,
                    n_exits,
                    nodes,
                    n_nodes,
                    terminal: pd.terminal,
                    junction,
                    ignore,
                });
                for s in 0..n_exits as usize {
                    exit_count[exits[s] as usize] += 1;
                    if !thin {
                        at_edge[exits[s] as usize].push((id, ls[s]));
                    }
                }
                if junction {
                    junctions.push(id);
                }
                for x in &nodes[..n_nodes as usize] {
                    match node_paths.iter_mut().find(|(n, _)| n == x) {
                        Some((_, v)) => {
                            // (a path naming the node at both ends is listed
                            // once, as `select` does)
                            if v.last() != Some(&id) {
                                v.push(id);
                            }
                        }
                        None => node_paths.push((*x, vec![id])),
                    }
                }
            }
        }
        let mut converging = [false; 6];
        for e in 0..6 {
            converging[e] = exit_count[e] > 1;
        }
        HexTopo { copies, at_edge, junctions, converging, node_paths, n_base: tile.paths.len(), dims: (nc, nt, no), max_lane }
    }
}

/// Per title and hex: the tables of the tiles seen there lately.
type HexCache = HashMap<&'static str, Vec<Vec<(HexKey, Arc<HexTopo>)>>>;

thread_local! {
    static HEX_CACHE: RefCell<HexCache> = RefCell::new(HashMap::new());
}

/// Tables kept per hex (the generator's lays try a few tiles per hex).
const HEX_CACHE_DEPTH: usize = 8;

impl Topo {
    fn build(g: &BaseGame) -> Topo {
        let title = g.title_def().name();
        let n = g.hexes.len();
        let flags = preprinted_flags(g);
        let hexes: Vec<Arc<HexTopo>> = HEX_CACHE.with(|c| {
            let mut c = c.borrow_mut();
            let per_hex = c.entry(title).or_default();
            if per_hex.len() != n {
                *per_hex = (0..n).map(|_| Vec::new()).collect();
            }
            (0..n)
                .map(|hi| {
                    let tile = &g.hexes[hi].tile;
                    let slot = &mut per_hex[hi];
                    if let Some(i) = slot.iter().position(|(k, _)| k.matches(tile)) {
                        if i != 0 {
                            let e = slot.remove(i);
                            slot.insert(0, e);
                        }
                        return Arc::clone(&slot[0].1);
                    }
                    let t = Arc::new(HexTopo::build(g, hi, &flags));
                    let key = HexKey {
                        id: tile.id.clone(),
                        name: tile.name.clone(),
                        paths: tile.paths.clone(),
                        dims: (tile.cities.len(), tile.towns.len(), tile.offboards.len()),
                    };
                    slot.insert(0, (key, Arc::clone(&t)));
                    slot.truncate(HEX_CACHE_DEPTH);
                    t
                })
                .collect()
        });
        let mut copy_off = Vec::with_capacity(n + 1);
        let mut base_off = Vec::with_capacity(n + 1);
        let mut node_off = Vec::with_capacity(n + 1);
        let (mut c, mut b, mut nd, mut max_lane) = (0, 0, 0, 0);
        for h in &hexes {
            copy_off.push(c);
            base_off.push(b);
            node_off.push(nd);
            c += h.copies.len();
            b += h.n_base;
            nd += h.dims.0 + h.dims.1 + h.dims.2;
            max_lane = max_lane.max(h.max_lane);
        }
        copy_off.push(c);
        base_off.push(b);
        node_off.push(nd);
        Topo { hexes, copy_off, base_off, node_off, copies_total: c, nodes_total: nd, max_lane }
    }

    fn node_id(&self, n: Node) -> usize {
        let (nc, nt, _) = self.hexes[n.hex].dims;
        self.node_off[n.hex]
            + match n.kind {
                NodeKind::City => n.index,
                NodeKind::Town => nc + n.index,
                NodeKind::Offboard => nc + nt + n.index,
            }
    }

    fn paths_of(&self, n: Node) -> &[u32] {
        self.hexes[n.hex].node_paths.iter().find(|(x, _)| *x == n).map_or(&[], |(_, v)| v.as_slice())
    }
}

struct Ctx<'a> {
    game: &'a BaseGame,
    corp: &'a str,
    neighbors: &'a Neighbors,
    topo: &'a Topo,
    with_lanes: bool,
    ca: bool,
}

impl<'a> Ctx<'a> {
    /// City#blocks?(corporation) (graph.rs `Ctx::blocks`).
    fn blocks(&self, n: Node) -> bool {
        if n.kind != NodeKind::City {
            return false;
        }
        let city = &self.game.hexes[n.hex].tile.cities[n.index];
        if city.tokened_by(self.corp) {
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

    fn edge_counter(&self, hex: usize, e: u8, lane: u8) -> usize {
        (hex * 6 + e as usize) * (self.topo.max_lane + 1) + lane as usize
    }
}

struct Walk {
    visited_paths: Vec<bool>,
    counter_edges: Vec<i32>,
    counter_junctions: Vec<i32>,
}

struct Found<'b> {
    paths: &'b mut Vec<bool>,
    nodes: &'b mut HashSet<Node>,
    local_nodes: &'b mut HashSet<Node>,
    hexes: &'b mut BTreeMap<usize, Vec<u8>>,
}

fn add_edge(hexes: &mut BTreeMap<usize, Vec<u8>>, hex: usize, e: u8) {
    let v = hexes.entry(hex).or_default();
    if !v.contains(&e) {
        v.push(e);
    }
}

impl<'b> Found<'b> {
    fn record(&mut self, ctx: &Ctx, c: &Copy) {
        let b = ctx.topo.base_off[c.hex] + c.pi;
        if self.paths[b] {
            return;
        }
        self.paths[b] = true;
        for n in &c.nodes[..c.n_nodes as usize] {
            self.nodes.insert(*n);
            self.local_nodes.insert(*n);
        }
        for &e in &c.exits[..c.n_exits as usize] {
            add_edge(self.hexes, c.hex, e);
            if let Some(&nb) = ctx.neighbors.get(&c.hex).and_then(|m| m.get(&e)) {
                add_edge(self.hexes, nb, (e + 3) % 6);
            }
        }
    }
}

fn node_walk(ctx: &Ctx, node: Node, visited_nodes: &mut Vec<bool>, w: &mut Walk, converging_path: bool, found: &mut Found) {
    let nid = ctx.topo.node_id(node);
    if visited_nodes[nid] {
        return;
    }
    visited_nodes[nid] = true;
    for &np in ctx.topo.paths_of(node) {
        if ctx.topo.hexes[node.hex].copies[np as usize].ignore {
            continue;
        }
        path_walk(ctx, node.hex, np, None, None, w, converging_path, node, visited_nodes, converging_path, found);
    }
    if converging_path {
        visited_nodes[nid] = false;
    }
}

#[allow(clippy::too_many_arguments)]
fn path_walk(
    ctx: &Ctx,
    hex: usize,
    local: u32,
    skip: Option<u8>,
    jskip: Option<usize>,
    w: &mut Walk,
    converging: bool,
    origin: Node,
    visited_nodes: &mut Vec<bool>,
    origin_converging_path: bool,
    found: &mut Found,
) {
    let pu = ctx.topo.copy_off[hex] + local as usize;
    if w.visited_paths[pu] {
        return;
    }
    let ht = &ctx.topo.hexes[hex];
    let c = &ht.copies[local as usize];
    let junction = c.junction;
    if junction && w.counter_junctions[hex] > 1 {
        return;
    }
    let ne = c.n_exits as usize;
    let mut sum = 0;
    for s in 0..ne {
        sum += w.counter_edges[ctx.edge_counter(hex, c.exits[s], c.lanes[s].1)];
    }
    if sum > 0 {
        return;
    }
    let terminal = c.terminal;
    if junction && terminal {
        return;
    }
    w.visited_paths[pu] = true;
    if junction {
        w.counter_junctions[hex] += 1;
    }

    // -- the Node#walk block --
    found.record(ctx, c);
    if !terminal {
        for next in &c.nodes[..c.n_nodes as usize] {
            if *next == origin || ctx.blocks(*next) {
                continue;
            }
            node_walk(ctx, *next, visited_nodes, w, origin_converging_path || converging, found);
        }
    }

    if junction && jskip != Some(hex) {
        for &jp in &ht.junctions {
            path_walk(ctx, hex, jp, None, Some(hex), w, converging, origin, visited_nodes, origin_converging_path, found);
        }
    }
    for s in 0..ne {
        let e = c.exits[s];
        if Some(e) == skip {
            continue;
        }
        let Some(&nb) = ctx.neighbors.get(&hex).and_then(|m| m.get(&e)) else { continue };
        let lane = c.lanes[s];
        let ci = ctx.edge_counter(hex, e, lane.1);
        w.counter_edges[ci] += 1;
        let np_edge = (e + 3) % 6;
        let conv = converging || ht.converging[e as usize];
        for &(np, nlane) in &ctx.topo.hexes[nb].at_edge[np_edge as usize] {
            if ctx.with_lanes && !lane_match(lane, nlane, ctx.ca) {
                continue;
            }
            path_walk(ctx, nb, np, Some(np_edge), None, w, conv, origin, visited_nodes, origin_converging_path, found);
        }
        w.counter_edges[ci] -= 1;
    }
    if converging {
        w.visited_paths[pu] = false;
    }
    if junction {
        w.counter_junctions[hex] -= 1;
    }
}

impl BaseGame {
    /// Ruby Graph#compute(corporation) on the current map, on per-state
    /// arrays (see the module docs): `g1822_graph_fresh`'s fast path.
    pub(crate) fn g1822_graph_fast(&self, corp: &str) -> CorpGraph {
        let neighbors = self.g1822_neighbors();
        let t = super::prof::start();
        let topo = Topo::build(self);
        super::prof::stop(t, || "graph:topo".into());
        let ctx = Ctx {
            game: self,
            corp,
            neighbors: &neighbors,
            topo: &topo,
            with_lanes: self.g1822_is_pnw() || self.g1822_is_ca(),
            ca: self.g1822_is_ca(),
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
        let mut paths = vec![false; *topo.base_off.last().unwrap_or(&0)];
        let mut route_available = false;
        let mut route_train_purchase = false;
        let n_counters = self.hexes.len() * 6 * (topo.max_lane + 1);
        for &t in &tokens {
            let mut visited_nodes = vec![false; topo.nodes_total];
            for o in tokens.iter().filter(|n| **n != t) {
                visited_nodes[topo.node_id(*o)] = true;
            }
            let mut local_nodes: HashSet<Node> = HashSet::new();
            let mut w = Walk {
                visited_paths: vec![false; topo.copies_total],
                counter_edges: vec![0; n_counters],
                counter_junctions: vec![0; self.hexes.len()],
            };
            {
                let mut found = Found { paths: &mut paths, nodes: &mut nodes, local_nodes: &mut local_nodes, hexes: &mut hexes };
                node_walk(&ctx, t, &mut visited_nodes, &mut w, false, &mut found);
            }
            if route_train_purchase {
                continue;
            }
            // Every revenue center is :mandatory (RevenueCenter default).
            if local_nodes.len() > 1 {
                route_available = true;
                route_train_purchase = true;
            }
        }
        let mut connected_paths: HashSet<(usize, usize)> = HashSet::new();
        for (hi, w) in topo.base_off.windows(2).enumerate() {
            for b in w[0]..w[1] {
                if paths[b] {
                    connected_paths.insert((hi, b - w[0]));
                }
            }
        }
        let reachable_hexes: HashSet<usize> = connected_paths.iter().map(|p| p.0).collect();
        CorpGraph {
            connected_hexes: hexes,
            connected_nodes: nodes,
            reachable_hexes,
            connected_paths,
            route_available,
            route_train_purchase,
        }
    }
}

impl BaseGame {
    /// The fast paths' pure helpers against their references on this state
    /// (the differential tests call it at every state): the neighbour map
    /// (`g1822_neighbors`'s table + memo against the walk over every hex)
    /// and every tokened corporation's graph (`fastgraph` against
    /// `graph.rs`'s walk). None when all agree, else the first difference.
    pub fn g1822_graph_check_with(&self) -> Option<String> {
        let fast = self.g1822_neighbors();
        let reference = super::legal::reference::with(true, || self.g1822_neighbors());
        if *fast != *reference {
            return Some("neighbour maps differ".to_string());
        }
        for c in &self.corporations {
            if self.g1822_token_nodes(&c.sym).is_empty() {
                continue;
            }
            let a = self.g1822_graph_fast(&c.sym);
            let b = self.g1822_graph_reference(&c.sym);
            let same = a.connected_hexes == b.connected_hexes
                && a.connected_nodes == b.connected_nodes
                && a.reachable_hexes == b.reachable_hexes
                && a.connected_paths == b.connected_paths
                && a.route_available == b.route_available
                && a.route_train_purchase == b.route_train_purchase;
            if !same {
                return Some(format!("{}: graphs differ: fast {a:?} reference {b:?}", c.sym));
            }
        }
        None
    }
}
