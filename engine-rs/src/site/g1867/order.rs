//! The order Ruby's `Graph#compute` finds a company's nodes in (its `@connected_nodes` hash: the token cities first,
//! then each walked path's nodes as `Node#walk` / `Path#walk` meet them), which G1867::Step::Merge#mergeable_candidates
//! reads (`connected_cities(corp).flat_map { tokens' corporations }`), so a merger's partners are listed in Ruby's order
//! (docs/SITE_ENGINE_RS_STEP3.md §10: the menu listed them by name). The walk is `g1867/graph.rs`'s (Ruby's walk on
//! 1867's neighbours), with the nodes kept in the order found; nothing else about it changes.

use crate::game::BaseGame;
use crate::tiles::PathEndpoint as E;

/// A node: hex index, endpoint (City/Town/Offboard).
type Node = (usize, E);
type PathRef = (usize, usize);

struct Walk<'a> {
    game: &'a BaseGame,
    corp: &'a str,
    neighbors: &'a std::collections::HashMap<usize, std::collections::BTreeMap<u8, usize>>,
    visited_paths: Vec<bool>,
    counter_edges: Vec<i32>,
    counter_junctions: Vec<i32>,
    seen_paths: Vec<bool>,
    nodes: Vec<Node>,
    stride: usize,
}

fn is_node(e: &E) -> bool {
    matches!(e, E::City(_) | E::Town(_) | E::Offboard(_))
}

impl<'a> Walk<'a> {
    fn path(&self, p: PathRef) -> &'a crate::tiles::PathDef {
        &self.game.hexes[p.0].tile.paths[p.1]
    }
    fn junction(&self, p: PathRef) -> bool {
        let pd = self.path(p);
        pd.a == E::Junction || pd.b == E::Junction
    }
    fn exits(&self, p: PathRef) -> [Option<u8>; 2] {
        let pd = self.path(p);
        [pd.a.edge_num(), pd.b.edge_num()]
    }
    fn path_nodes(&self, p: PathRef) -> Vec<Node> {
        let pd = self.path(p);
        [&pd.a, &pd.b].into_iter().filter(|e| is_node(e)).map(|e| (p.0, e.clone())).collect()
    }
    fn converging_exit(&self, hex: usize, edge: u8) -> bool {
        let tile = &self.game.hexes[hex].tile;
        (0..tile.paths.len()).map(|i| self.exits((hex, i)).iter().filter(|e| **e == Some(edge)).count()).sum::<usize>() > 1
    }
    fn blocks(&self, n: &Node) -> bool {
        match &n.1 {
            E::Offboard(_) => true,
            E::City(i) => {
                let city = &self.game.hexes[n.0].tile.cities[*i];
                !(city.tokens.iter().flatten().any(|t| t.corporation_id == self.corp)
                    || city.tokens.iter().any(|t| t.is_none())
                    || city.tokens.iter().flatten().any(|t| t.token_type == "neutral"))
            }
            _ => false,
        }
    }
    fn record(&mut self, p: PathRef) {
        let k = p.0 * self.stride + p.1;
        if self.seen_paths[k] {
            return;
        }
        self.seen_paths[k] = true;
        for n in self.path_nodes(p) {
            if !self.nodes.contains(&n) {
                self.nodes.push(n);
            }
        }
    }

    fn node_walk(&mut self, node: Node, visited_nodes: &mut Vec<Node>, converging_path: bool) {
        if visited_nodes.contains(&node) {
            return;
        }
        visited_nodes.push(node.clone());
        let n_paths = self.game.hexes[node.0].tile.paths.len();
        for i in 0..n_paths {
            let np = (node.0, i);
            if self.path_nodes(np).contains(&node) {
                self.path_walk(np, None, None, converging_path, &node, visited_nodes, converging_path);
            }
        }
        if converging_path {
            if let Some(pos) = visited_nodes.iter().position(|n| *n == node) {
                visited_nodes.swap_remove(pos);
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn path_walk(&mut self, p: PathRef, skip: Option<u8>, jskip: Option<usize>, converging: bool, origin: &Node, visited_nodes: &mut Vec<Node>, origin_converging_path: bool) {
        let pk = p.0 * self.stride + p.1;
        if self.visited_paths[pk] {
            return;
        }
        let junction = self.junction(p);
        if junction && self.counter_junctions[p.0] > 1 {
            return;
        }
        let exits = self.exits(p);
        if exits.iter().flatten().map(|&e| self.counter_edges[p.0 * 6 + e as usize]).sum::<i32>() > 0 {
            return;
        }
        let terminal = self.path(p).terminal;
        if junction && terminal {
            return;
        }
        self.visited_paths[pk] = true;
        if junction {
            self.counter_junctions[p.0] += 1;
        }
        self.record(p);
        if !terminal {
            for next in self.path_nodes(p) {
                if next == *origin || self.blocks(&next) {
                    continue;
                }
                self.node_walk(next, visited_nodes, origin_converging_path || converging);
            }
        }
        if junction && jskip != Some(p.0) {
            let n_paths = self.game.hexes[p.0].tile.paths.len();
            for j in 0..n_paths {
                if self.junction((p.0, j)) {
                    self.path_walk((p.0, j), None, Some(p.0), converging, origin, visited_nodes, origin_converging_path);
                }
            }
        }
        for e in exits.into_iter().flatten() {
            if Some(e) == skip {
                continue;
            }
            let Some(&nb) = self.neighbors.get(&p.0).and_then(|m| m.get(&e)) else { continue };
            self.counter_edges[p.0 * 6 + e as usize] += 1;
            let np_edge = (e + 3) % 6;
            let conv = converging || self.converging_exit(p.0, e);
            let n_paths = self.game.hexes[nb].tile.paths.len();
            for q in 0..n_paths {
                if self.exits((nb, q)).contains(&Some(np_edge)) {
                    self.path_walk((nb, q), Some(np_edge), None, conv, origin, visited_nodes, origin_converging_path);
                }
            }
            self.counter_edges[p.0 * 6 + e as usize] -= 1;
        }
        if converging {
            self.visited_paths[pk] = false;
        }
        if junction {
            self.counter_junctions[p.0] -= 1;
        }
    }
}

/// Ruby's `graph.connected_nodes(corp).keys`, in its order.
pub fn connected_nodes_in_order(game: &BaseGame, corp: &str) -> Vec<Node> {
    let neighbors = game.g1867_neighbors();
    let n = game.hexes.len();
    let stride = game.hexes.iter().map(|h| h.tile.paths.len()).max().unwrap_or(1).max(1);
    let mut tokens: Vec<Node> = Vec::new();
    for (hi, hex) in game.hexes.iter().enumerate() {
        for (ci, city) in hex.tile.cities.iter().enumerate() {
            if city.tokens.iter().flatten().any(|t| t.corporation_id == corp) {
                tokens.push((hi, E::City(ci)));
            }
        }
    }
    let mut w = Walk {
        game,
        corp,
        neighbors,
        visited_paths: vec![false; n * stride],
        counter_edges: vec![0; n * 6],
        counter_junctions: vec![0; n],
        seen_paths: vec![false; n * stride],
        nodes: tokens.clone(),
        stride,
    };
    for t in &tokens {
        let mut visited_nodes: Vec<Node> = tokens.iter().filter(|x| *x != t).cloned().collect();
        w.visited_paths.iter_mut().for_each(|v| *v = false);
        w.node_walk(t.clone(), &mut visited_nodes, false);
    }
    w.nodes
}

/// G1867::Step::Merge#mergeable_candidates' order: for each company merging (or the minor), its connected cities in
/// Ruby's order, each city's stations' companies in slot order, each once. The caller filters (the generator's moves).
pub fn candidate_order(game: &BaseGame, merging: &[String]) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for corp in merging {
        for (hex, e) in connected_nodes_in_order(game, corp) {
            let E::City(ci) = e else { continue };
            let Some(city) = game.hexes[hex].tile.cities.get(ci) else { continue };
            for t in city.tokens.iter().flatten() {
                if !merging.contains(&t.corporation_id) && !out.contains(&t.corporation_id) {
                    out.push(t.corporation_id.clone());
                }
            }
        }
    }
    out
}
