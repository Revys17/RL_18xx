//! Ruby's graph walk (`graph.rs`: `node_walk` / `path_walk`) on arrays
//! precomputed once per state, for the Track generator's many questions of
//! the form "after laying THIS tile on hex h, does the corporation's walk
//! reach one of these paths?" (`check_track_restrictions!`, once per lay
//! candidate).
//!
//! Exactness: the walk below makes the same calls in the same order as
//! `graph.rs`'s — every per-path / per-node fact it reads (exits, nodes,
//! junction, terminal, `converging_exit?`, `blocks?`, the paths through an
//! edge or touching a node, in path-index order) is taken from the tile
//! once instead of on every visit, and the hex neighbours come from the
//! static table. A candidate lay replaces one hex's facts ([`WalkMap::hex_facts`]
//! of the laid tile). It answers only "is a target path recorded", stopping
//! at the first one: the recorded set only grows, in a fixed order, so the
//! answer is that of the full walk.

use crate::game::BaseGame;
use crate::graph::Tile;
use crate::tiles::PathEndpoint;

use super::graph::{Node, NodeKind};

/// Max paths per tile (graph.rs `MAXP`).
const MAXP: usize = 16;
const NO_EDGE: u8 = u8::MAX;

/// One path's facts.
#[derive(Clone, Debug)]
struct PathW {
    exits: [u8; 2],
    nodes: [Option<Node>; 2],
    junction: bool,
    terminal: bool,
}

/// One hex's (tile's) facts.
#[derive(Clone, Debug, Default)]
pub(crate) struct HexW {
    paths: Vec<PathW>,
    /// Path indices exiting through each edge, ascending.
    by_exit: [Vec<u8>; 6],
    /// `Tile#converging_exit?` per edge.
    converging: [bool; 6],
    /// Junction path indices, ascending.
    junctions: Vec<u8>,
    /// Paths touching each node (by kind, then node index), ascending.
    city_paths: Vec<Vec<u8>>,
    town_paths: Vec<Vec<u8>>,
    off_paths: Vec<Vec<u8>>,
    /// `City#blocks?(corporation)` per city.
    city_blocks: Vec<bool>,
    /// The cities holding a token of the corporation (the walk's starts).
    token_cities: Vec<usize>,
}

fn endpoint_node(e: &PathEndpoint, hex: usize) -> Option<Node> {
    match e {
        PathEndpoint::City(i) => Some(Node { hex, kind: NodeKind::City, index: *i }),
        PathEndpoint::Town(i) => Some(Node { hex, kind: NodeKind::Town, index: *i }),
        PathEndpoint::Offboard(i) => Some(Node { hex, kind: NodeKind::Offboard, index: *i }),
        _ => None,
    }
}

/// The walk's per-state data for one corporation.
pub(crate) struct WalkMap<'a> {
    nb: &'a [[u32; 6]],
    hexes: Vec<HexW>,
    corp: String,
    // Scratch, reused by every walk.
    visited_paths: Vec<bool>,
    counter_edges: Vec<i32>,
    counter_junctions: Vec<i32>,
    seen_path: Vec<bool>,
}

struct Walk<'w, 'a> {
    map: &'w WalkMap<'a>,
    over: Option<(usize, &'w HexW)>,
    target: &'w dyn Fn(usize, usize) -> bool,
}

impl<'w, 'a> Walk<'w, 'a> {
    #[inline]
    fn hex(&self, h: usize) -> &'w HexW {
        match self.over {
            Some((o, hw)) if o == h => hw,
            _ => &self.map.hexes[h],
        }
    }
    #[inline]
    fn node_paths(&self, n: Node) -> &'w [u8] {
        let hw = self.hex(n.hex);
        let v = match n.kind {
            NodeKind::City => &hw.city_paths,
            NodeKind::Town => &hw.town_paths,
            NodeKind::Offboard => &hw.off_paths,
        };
        v.get(n.index).map_or(&[], |x| x.as_slice())
    }
    /// Node#blocks?(corporation) (graph.rs `Ctx::blocks`).
    #[inline]
    fn blocks(&self, n: Node) -> bool {
        match n.kind {
            NodeKind::Offboard => true,
            NodeKind::Town => false,
            NodeKind::City => self.hex(n.hex).city_blocks[n.index],
        }
    }
    #[inline]
    fn neighbor(&self, hex: usize, edge: u8) -> Option<usize> {
        let n = self.map.nb[hex][edge as usize];
        (n != u32::MAX).then_some(n as usize)
    }
}

/// Scratch state of one walk (graph.rs `Walk` + the stop flag).
struct State<'s> {
    visited_paths: &'s mut [bool],
    counter_edges: &'s mut [i32],
    counter_junctions: &'s mut [i32],
    hit: bool,
}

fn node_walk(w: &Walk, node: Node, visited_nodes: &mut Vec<Node>, st: &mut State, converging_path: bool) {
    if visited_nodes.contains(&node) {
        return;
    }
    visited_nodes.push(node);
    for &i in w.node_paths(node) {
        path_walk(w, (node.hex, i as usize), None, None, st, converging_path, node, visited_nodes, converging_path);
        if st.hit {
            return;
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
    w: &Walk,
    p: (usize, usize),
    skip: Option<u8>,
    jskip: Option<usize>,
    st: &mut State,
    converging: bool,
    origin: Node,
    visited_nodes: &mut Vec<Node>,
    origin_converging_path: bool,
) {
    let pk = p.0 * MAXP + p.1;
    if st.visited_paths[pk] {
        return;
    }
    let hw = w.hex(p.0);
    let pw = &hw.paths[p.1];
    let junction = pw.junction;
    if junction && st.counter_junctions[p.0] > 1 {
        return;
    }
    let mut sum = 0;
    for &e in &pw.exits {
        if e != NO_EDGE {
            sum += st.counter_edges[p.0 * 6 + e as usize];
        }
    }
    if sum > 0 {
        return;
    }
    let terminal = pw.terminal;
    if junction && terminal {
        return;
    }
    st.visited_paths[pk] = true;
    if junction {
        st.counter_junctions[p.0] += 1;
    }

    // -- the Node#walk block: record the path --
    if (w.target)(p.0, p.1) {
        st.hit = true;
        return;
    }
    if !terminal {
        for next in pw.nodes.iter().flatten() {
            if *next == origin || w.blocks(*next) {
                continue;
            }
            node_walk(w, *next, visited_nodes, st, origin_converging_path || converging);
            if st.hit {
                return;
            }
        }
    }

    if junction && jskip != Some(p.0) {
        for &j in &hw.junctions {
            path_walk(w, (p.0, j as usize), None, Some(p.0), st, converging, origin, visited_nodes, origin_converging_path);
            if st.hit {
                return;
            }
        }
    }
    for &e in &pw.exits {
        if e == NO_EDGE || Some(e) == skip {
            continue;
        }
        let Some(nb) = w.neighbor(p.0, e) else { continue };
        st.counter_edges[p.0 * 6 + e as usize] += 1;
        let np_edge = (e + 3) % 6;
        let conv = converging || hw.converging[e as usize];
        for &q in &w.hex(nb).by_exit[np_edge as usize] {
            path_walk(w, (nb, q as usize), Some(np_edge), None, st, conv, origin, visited_nodes, origin_converging_path);
            if st.hit {
                return;
            }
        }
        st.counter_edges[p.0 * 6 + e as usize] -= 1;
    }
    if converging {
        st.visited_paths[pk] = false;
    }
    if junction {
        st.counter_junctions[p.0] -= 1;
    }
}

impl<'a> WalkMap<'a> {
    /// The facts of `tile` on hex `hex` for `corp`.
    pub(crate) fn hex_facts(tile: &Tile, hex: usize, corp: &str) -> HexW {
        let mut hw = HexW {
            paths: Vec::with_capacity(tile.paths.len()),
            city_paths: vec![Vec::new(); tile.cities.len()],
            town_paths: vec![Vec::new(); tile.towns.len()],
            off_paths: vec![Vec::new(); tile.offboards.len()],
            ..HexW::default()
        };
        let mut exit_count = [0usize; 6];
        for (i, pd) in tile.paths.iter().enumerate() {
            let exits = [pd.a.edge_num().unwrap_or(NO_EDGE), pd.b.edge_num().unwrap_or(NO_EDGE)];
            let nodes = [endpoint_node(&pd.a, hex), endpoint_node(&pd.b, hex)];
            let junction = pd.a == PathEndpoint::Junction || pd.b == PathEndpoint::Junction;
            // `path_has_exit` / `contains`: each path once per edge / node.
            let mut listed: [bool; 6] = [false; 6];
            for &e in &exits {
                if e != NO_EDGE {
                    exit_count[e as usize] += 1;
                    if !listed[e as usize] {
                        listed[e as usize] = true;
                        hw.by_exit[e as usize].push(i as u8);
                    }
                }
            }
            let mut touched: Vec<Node> = Vec::new();
            for n in nodes.iter().flatten() {
                if touched.contains(n) {
                    continue;
                }
                touched.push(*n);
                let list = match n.kind {
                    NodeKind::City => &mut hw.city_paths,
                    NodeKind::Town => &mut hw.town_paths,
                    NodeKind::Offboard => &mut hw.off_paths,
                };
                if list.len() <= n.index {
                    list.resize(n.index + 1, Vec::new());
                }
                list[n.index].push(i as u8);
            }
            if junction {
                hw.junctions.push(i as u8);
            }
            hw.paths.push(PathW { exits, nodes, junction, terminal: pd.terminal });
        }
        for e in 0..6 {
            hw.converging[e] = exit_count[e] > 1;
        }
        hw.city_blocks = tile
            .cities
            .iter()
            .map(|city| {
                !(city.tokens.iter().flatten().any(|t| t.corporation_id == corp)
                    || city.tokens.iter().any(|t| t.is_none())
                    || city.tokens.iter().flatten().any(|t| t.token_type == "neutral"))
            })
            .collect();
        hw.token_cities = tile
            .cities
            .iter()
            .enumerate()
            .filter(|(_, c)| c.tokens.iter().flatten().any(|t| t.corporation_id == corp))
            .map(|(ci, _)| ci)
            .collect();
        hw
    }

    /// The walk data of `game`'s map for `corp`.
    pub(crate) fn new(game: &'a BaseGame, corp: &str, nb: &'a [[u32; 6]]) -> WalkMap<'a> {
        let n = game.hexes.len();
        debug_assert!(game.hexes.iter().all(|h| h.tile.paths.len() <= MAXP));
        WalkMap {
            nb,
            hexes: game.hexes.iter().enumerate().map(|(hi, h)| Self::hex_facts(&h.tile, hi, corp)).collect(),
            corp: corp.to_string(),
            visited_paths: vec![false; n * MAXP],
            counter_edges: vec![0; n * 6],
            counter_junctions: vec![0; n],
            seen_path: vec![false; n * MAXP],
        }
    }

    pub(crate) fn corp(&self) -> &str {
        &self.corp
    }

    /// The token cities in Ruby's order (`g1867_token_nodes_with`: hex
    /// order, then city order), hex `over.0` read from `over.1`.
    fn tokens(&self, over: Option<(usize, &HexW)>) -> Vec<Node> {
        let mut out = Vec::new();
        for (hi, hw) in self.hexes.iter().enumerate() {
            let hw = match over {
                Some((o, x)) if o == hi => x,
                _ => hw,
            };
            for &ci in &hw.token_cities {
                out.push(Node { hex: hi, kind: NodeKind::City, index: ci });
            }
        }
        out
    }

    /// [`BaseGame::g1867_graph_reaches`]: whether Graph#compute(corp), with
    /// hex `over.0` described by `over.1`, records a path `target(hex,
    /// path)` accepts.
    pub(crate) fn reaches(&mut self, over: Option<(usize, &HexW)>, target: &dyn Fn(usize, usize) -> bool) -> bool {
        let tokens = self.tokens(over);
        let mut visited_paths = std::mem::take(&mut self.visited_paths);
        let mut counter_edges = std::mem::take(&mut self.counter_edges);
        let mut counter_junctions = std::mem::take(&mut self.counter_junctions);
        counter_edges.iter_mut().for_each(|c| *c = 0);
        counter_junctions.iter_mut().for_each(|c| *c = 0);
        let mut hit = false;
        {
            let w = Walk { map: self, over, target };
            let mut st = State {
                visited_paths: &mut visited_paths,
                counter_edges: &mut counter_edges,
                counter_junctions: &mut counter_junctions,
                hit: false,
            };
            for &t in &tokens {
                let mut visited_nodes: Vec<Node> = tokens.iter().copied().filter(|x| *x != t).collect();
                st.visited_paths.iter_mut().for_each(|v| *v = false);
                node_walk(&w, t, &mut visited_nodes, &mut st, false);
                if st.hit {
                    hit = true;
                    break;
                }
            }
        }
        self.visited_paths = visited_paths;
        self.counter_edges = counter_edges;
        self.counter_junctions = counter_junctions;
        hit
    }

    /// [`BaseGame::g1867_graph_may_reach`]: the walk's moves without its
    /// dynamic limits; `false` = no walk records a target.
    pub(crate) fn may_reach(&mut self, over: Option<(usize, &HexW)>, target: &dyn Fn(usize, usize) -> bool) -> bool {
        let tokens = self.tokens(over);
        let mut seen_path = std::mem::take(&mut self.seen_path);
        seen_path.iter_mut().for_each(|v| *v = false);
        let found = {
            let w = Walk { map: self, over, target };
            let mut seen_node: Vec<Node> = Vec::new();
            let mut stack: Vec<(usize, usize)> = Vec::new();
            for &t in &tokens {
                if !seen_node.contains(&t) {
                    seen_node.push(t);
                    stack.extend(w.node_paths(t).iter().map(|&i| (t.hex, i as usize)));
                }
            }
            let mut found = false;
            while let Some(p) = stack.pop() {
                let k = p.0 * MAXP + p.1;
                if seen_path[k] {
                    continue;
                }
                seen_path[k] = true;
                let hw = w.hex(p.0);
                let pw = &hw.paths[p.1];
                if pw.junction && pw.terminal {
                    continue;
                }
                if target(p.0, p.1) {
                    found = true;
                    break;
                }
                if !pw.terminal {
                    for next in pw.nodes.iter().flatten() {
                        if w.blocks(*next) || seen_node.contains(next) {
                            continue;
                        }
                        seen_node.push(*next);
                        stack.extend(w.node_paths(*next).iter().map(|&i| (next.hex, i as usize)));
                    }
                }
                if pw.junction {
                    stack.extend(hw.junctions.iter().map(|&j| (p.0, j as usize)));
                }
                for &e in &pw.exits {
                    if e == NO_EDGE {
                        continue;
                    }
                    let Some(h) = w.neighbor(p.0, e) else { continue };
                    let np_edge = (e + 3) % 6;
                    stack.extend(w.hex(h).by_exit[np_edge as usize].iter().map(|&q| (h, q as usize)));
                }
            }
            found
        };
        self.seen_path = seen_path;
        found
    }
}
