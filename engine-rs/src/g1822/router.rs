//! 1822 auto-router: the revenue-maximising legal run for a corporation's
//! trains, under Ruby's route rules for 1822 (route.rb `revenue` checks +
//! G1822 `check_distance` / `check_overlap` / `check_local_cities` /
//! `revenue_for` / `destination_bonus`).
//!
//! 1. **Enumerate** routes: a depth-first walk from every revenue node the
//!    corporation reaches, node to node along tile paths (a chain passes
//!    plain track and junctions, never a node), keeping routes that visit
//!    a corporation token. A route never revisits a node (check_cycles!),
//!    never passes THROUGH a node that blocks (a city full of others'
//!    tokens, an offboard), uses terminal paths only as its first or last
//!    path (check_terminals!), and — unless it touches one of the
//!    DOUBLE_HEX hexes — visits each stop hex once (consecutive stops on
//!    one hex count once, as Ruby's `hexes` chunks them).
//! 2. **Price** each route for each train kind: L/LP (1 city + 1 town, or a
//!    single tokened city), numeric trains (visit cost ≤ distance; France
//!    costs 0), the E-train (pays tokened cities only, ×2, from and to
//!    tokened cities), a pullman-attached train ("X+": towns free); the
//!    English Channel needs France and vice versa; LP may not cross it.
//! 3. **Combine**: branch-and-bound over one route (or none) per train;
//!    track is used once per train type (E-trains and the rest separately,
//!    lanes counted), local trains use each city once (a single-city run counts: its one
//!    connection is a chain),
//!    only one of the two Merthyr Tydfil–Pontypool tracks, and the best
//!    single destination bonus (home + destination token on one route)
//!    joins the total.
//!
//! Lanes: a `lanes:2` path is two Ruby paths; the engine keeps one, so a
//! track key's capacity is its lane count.

use std::collections::{BTreeMap, HashMap, HashSet};

use crate::actions::RouteData;
use crate::game::BaseGame;
use crate::tiles::PathEndpoint;
use crate::title::g1822 as data;

use super::graph::{Node, NodeKind};

/// DOUBLE_HEX: hexes whose two stops a route may both visit.
const DOUBLE_HEX: &[&str] = &["D35", "F7", "H21", "H37"];
/// Bound on a route's stops for the unbounded trains (E, pullman towns):
/// keeps the walk finite on dense maps.
const MAX_STOPS_UNBOUNDED: usize = 32;
/// Routes kept per train kind for the combination search.
const TOP_K: usize = 4000;
/// Walk steps per routing call (dense fuzz maps can hold millions of
/// routes; the best ones are found early from the token cities).
const DFS_STEP_BUDGET: u64 = 3_000_000;

/// A walk's step allowance (one start node's fair share of the call's
/// DFS_STEP_BUDGET).
struct Budget {
    used: u64,
    limit: u64,
}

impl Budget {
    /// Each start gets an equal part of what the earlier ones left, so a
    /// dense corner of the map cannot starve the rest (each route is
    /// recorded from its lower-keyed end only, i.e. from ONE start).
    fn share(spent: u64, starts_left: usize) -> Budget {
        Budget { used: 0, limit: DFS_STEP_BUDGET.saturating_sub(spent) / starts_left.max(1) as u64 }
    }
}

#[derive(Clone, Copy, Debug)]
struct Bound {
    total: usize,
    cities: usize,
}

type PathRef = (usize, usize);

/// One node-to-node chain: the paths it uses and the track keys.
#[derive(Clone, Debug)]
struct Seg {
    to: Node,
    paths: Vec<PathRef>,
    /// (hex, edge) per path end on an edge.
    track: Vec<(usize, u8)>,
    /// Ends on a terminal path (the route must stop here).
    terminal_last: bool,
    /// Starts with a terminal path (only allowed as a route's first chain).
    terminal_first: bool,
    /// Uses a terminal path strictly inside the chain (never legal).
    terminal_inner: bool,
}

/// A walked route.
#[derive(Clone, Debug)]
struct Walked {
    stops: Vec<Node>,
    segs: Vec<Seg>,
}

/// A route as priced for one train.
#[derive(Clone, Debug)]
struct Priced {
    route: usize,
    revenue: i32,
    bonus: i32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Entry {
    Node,
    Edge(u8),
    Junction,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Kind {
    Local { lp: bool },
    Numeric(u32),
    E,
    Pullman(u32),
}

struct Ctx<'a> {
    g: &'a BaseGame,
    corp: &'a str,
    neighbors: HashMap<usize, BTreeMap<u8, usize>>,
    lanes: HashMap<PathRef, u8>,
    seg_cache: std::cell::RefCell<HashMap<Node, Vec<Seg>>>,
    london: Option<usize>,
}

fn path_lanes_from_dsl(dsl: &str) -> Vec<u8> {
    dsl.split(';')
        .filter_map(|p| p.trim().strip_prefix("path="))
        .map(|attrs| {
            attrs
                .split(',')
                .find_map(|kv| kv.strip_prefix("lanes:"))
                .and_then(|v| v.parse().ok())
                .unwrap_or(1)
        })
        .collect()
}

impl<'a> Ctx<'a> {
    fn new(g: &'a BaseGame, corp: &'a str) -> Ctx<'a> {
        let mut lanes = HashMap::new();
        for (hi, hex) in g.hexes.iter().enumerate() {
            let dsl = if hex.tile.id.starts_with("preprinted_") {
                data::hex_dsl(&hex.id).map(|(d, _)| d)
            } else {
                data::tile_dsl(hex.tile.name.split('-').next().unwrap_or(&hex.tile.name)).map(|(d, _)| d)
            };
            if let Some(d) = dsl {
                for (pi, l) in path_lanes_from_dsl(d).into_iter().enumerate() {
                    if l > 1 {
                        lanes.insert((hi, pi), l);
                    }
                }
            }
        }
        Ctx {
            g,
            corp,
            neighbors: g.g1822_neighbors(),
            lanes,
            seg_cache: std::cell::RefCell::new(HashMap::new()),
            london: g.hex_idx.get(data::LONDON_HEX).copied(),
        }
    }

    fn pd(&self, p: PathRef) -> &crate::tiles::PathDef {
        &self.g.hexes[p.0].tile.paths[p.1]
    }

    fn nodes_of(&self, p: PathRef) -> Vec<Node> {
        self.g.g1822_path_nodes(p)
    }

    fn is_junction(&self, p: PathRef) -> bool {
        let pd = self.pd(p);
        pd.a == PathEndpoint::Junction || pd.b == PathEndpoint::Junction
    }

    fn edges_of(&self, p: PathRef) -> Vec<u8> {
        let pd = self.pd(p);
        [&pd.a, &pd.b].iter().filter_map(|e| e.edge_num()).collect()
    }

    fn paths_at_edge(&self, hex: usize, edge: u8) -> Vec<PathRef> {
        let tile = &self.g.hexes[hex].tile;
        (0..tile.paths.len())
            .filter(|&i| self.edges_of((hex, i)).contains(&edge))
            .map(|i| (hex, i))
            .collect()
    }

    fn node_paths(&self, n: Node) -> Vec<PathRef> {
        let tile = &self.g.hexes[n.hex].tile;
        (0..tile.paths.len())
            .filter(|&i| self.nodes_of((n.hex, i)).contains(&n))
            .map(|i| (n.hex, i))
            .collect()
    }

    /// Node#blocks? for pass-through: a city full of other corporations'
    /// tokens; offboards are always ends.
    fn blocks(&self, n: Node) -> bool {
        match n.kind {
            NodeKind::Offboard => true,
            NodeKind::Town => false,
            NodeKind::City => {
                let city = &self.g.hexes[n.hex].tile.cities[n.index];
                if city.tokens.iter().flatten().any(|t| t.corporation_id == self.corp) {
                    return false;
                }
                if city.tokens.iter().any(|t| t.is_none()) {
                    return false;
                }
                true
            }
        }
    }

    /// Every chain from node `n`.
    fn segs_from(&self, n: Node) -> Vec<Seg> {
        if let Some(v) = self.seg_cache.borrow().get(&n) {
            return v.clone();
        }
        let mut out = Vec::new();
        for p in self.node_paths(n) {
            let mut paths = vec![p];
            let mut track: Vec<(usize, u8)> = Vec::new();
            self.follow(n, p, Entry::Node, &mut paths, &mut track, &mut out);
        }
        self.seg_cache.borrow_mut().insert(n, out.clone());
        out
    }

    /// `p` (the last of `paths`) was entered through `entry`: leave it by
    /// its other end.
    fn follow(
        &self,
        from: Node,
        p: PathRef,
        entry: Entry,
        paths: &mut Vec<PathRef>,
        track: &mut Vec<(usize, u8)>,
        out: &mut Vec<Seg>,
    ) {
        let pd = self.pd(p).clone();
        for end in [&pd.a, &pd.b] {
            // skip the end we came in by
            let is_entry = match (end, entry) {
                (PathEndpoint::Edge(e), Entry::Edge(x)) => *e == x,
                (PathEndpoint::Junction, Entry::Junction) => true,
                (PathEndpoint::City(_) | PathEndpoint::Town(_) | PathEndpoint::Offboard(_), Entry::Node) => {
                    self.nodes_of(p).first() == Some(&from) && self.node_at(p, end) == Some(from)
                }
                _ => false,
            };
            if is_entry {
                continue;
            }
            match end {
                PathEndpoint::City(_) | PathEndpoint::Town(_) | PathEndpoint::Offboard(_) => {
                    if let Some(m) = self.node_at(p, end) {
                        if m != from {
                            out.push(self.make_seg(m, paths, track));
                        }
                    }
                }
                PathEndpoint::Edge(e) => {
                    let e = *e;
                    if track.contains(&(p.0, e)) {
                        continue;
                    }
                    let Some(&nb) = self.neighbors.get(&p.0).and_then(|m| m.get(&e)) else { continue };
                    let inv = (e + 3) % 6;
                    if track.contains(&(nb, inv)) {
                        continue;
                    }
                    track.push((p.0, e));
                    track.push((nb, inv));
                    for q in self.paths_at_edge(nb, inv) {
                        if paths.contains(&q) {
                            continue;
                        }
                        // Path#walk: `return if @junction && counter[@junction]
                        // > 1` — a chain passes a junction once (in, out)
                        // (and `return if @junction && @terminal`)
                        if self.is_junction(q)
                            && (self.pd(q).terminal || paths.iter().any(|&o| o.0 == nb && self.is_junction(o)))
                        {
                            continue;
                        }
                        paths.push(q);
                        self.follow(from, q, Entry::Edge(inv), paths, track, out);
                        paths.pop();
                    }
                    track.pop();
                    track.pop();
                }
                PathEndpoint::Junction => {
                    let n = self.g.hexes[p.0].tile.paths.len();
                    for j in 0..n {
                        let q = (p.0, j);
                        if q == p || paths.contains(&q) || !self.is_junction(q) || self.pd(q).terminal {
                            continue;
                        }
                        paths.push(q);
                        self.follow(from, q, Entry::Junction, paths, track, out);
                        paths.pop();
                    }
                }
            }
        }
    }

    /// The node a path end names on its tile.
    fn node_at(&self, p: PathRef, end: &PathEndpoint) -> Option<Node> {
        let kind = match end {
            PathEndpoint::City(i) => (NodeKind::City, *i),
            PathEndpoint::Town(i) => (NodeKind::Town, *i),
            PathEndpoint::Offboard(i) => (NodeKind::Offboard, *i),
            _ => return None,
        };
        Some(Node { hex: p.0, kind: kind.0, index: kind.1 })
    }

    fn make_seg(&self, to: Node, paths: &[PathRef], track: &[(usize, u8)]) -> Seg {
        let n = paths.len();
        let term: Vec<bool> = paths.iter().map(|&p| self.pd(p).terminal).collect();
        // track keys: each path end on an edge, once per hexside side
        let mut keys: Vec<(usize, u8)> = Vec::new();
        for &p in paths {
            for e in self.edges_of(p) {
                if !keys.contains(&(p.0, e)) {
                    keys.push((p.0, e));
                }
            }
        }
        let _ = track;
        Seg {
            to,
            paths: paths.to_vec(),
            track: keys,
            terminal_first: term.first().copied().unwrap_or(false),
            terminal_last: n > 0 && term[n - 1],
            terminal_inner: n > 2 && term[1..n - 1].iter().any(|&t| t),
        }
    }

    fn tokened(&self, n: Node) -> bool {
        n.kind == NodeKind::City
            && self.g.hexes[n.hex].tile.cities[n.index]
                .tokens
                .iter()
                .flatten()
                .any(|t| t.corporation_id == self.corp)
    }

    /// Ruby `tokened_by?(entity, types: %i[normal destination])`.
    fn tokened_normal_or_dest(&self, n: Node) -> bool {
        n.kind == NodeKind::City
            && self.g.hexes[n.hex].tile.cities[n.index]
                .tokens
                .iter()
                .flatten()
                .any(|t| t.corporation_id == self.corp && (t.token_type == "normal" || t.token_type == "destination"))
    }

    fn stop_revenue(&self, n: Node) -> i32 {
        let hex = &self.g.hexes[n.hex];
        let phase = &self.g.phase.tiles;
        match n.kind {
            NodeKind::City => {
                let c = &hex.tile.cities[n.index];
                if hex.tile.id.starts_with("preprinted_") {
                    super::data_tiered_city_revenue(&hex.id, phase).unwrap_or(c.revenue)
                } else {
                    c.revenue
                }
            }
            NodeKind::Town => hex.tile.towns[n.index].revenue,
            NodeKind::Offboard => hex.tile.offboards[n.index].phase_revenue(phase),
        }
    }

    fn visit_cost(&self, n: Node) -> u32 {
        if self.g.hexes[n.hex].id == data::FRANCE_HEX {
            0
        } else {
            1
        }
    }
}

impl BaseGame {
    fn g1822_train_kind(t: &crate::entities::Train) -> Kind {
        if t.name == "L" || t.name == "LP" {
            Kind::Local { lp: t.name == "LP" }
        } else if t.name == "E" {
            Kind::E
        } else if t.name.ends_with('+') {
            Kind::Pullman(t.distance)
        } else {
            Kind::Numeric(t.distance)
        }
    }

    /// Ruby `Node#signature` ("HEX-<part index>"): the tile's parts list
    /// cities, then towns, then offboards.
    pub(crate) fn g1822_node_signature(&self, n: Node) -> String {
        let tile = &self.hexes[n.hex].tile;
        let idx = match n.kind {
            NodeKind::City => n.index,
            NodeKind::Town => tile.cities.len() + n.index,
            NodeKind::Offboard => tile.cities.len() + tile.towns.len() + n.index,
        };
        format!("{}-{}", self.hexes[n.hex].id, idx)
    }

    fn g1822_node_from_signature(&self, sig: &str) -> Option<Node> {
        let (hex, idx) = sig.rsplit_once('-')?;
        let idx: usize = idx.parse().ok()?;
        let &hi = self.hex_idx.get(hex)?;
        let tile = &self.hexes[hi].tile;
        let (nc, nt) = (tile.cities.len(), tile.towns.len());
        Some(if idx < nc {
            Node { hex: hi, kind: NodeKind::City, index: idx }
        } else if idx < nc + nt {
            Node { hex: hi, kind: NodeKind::Town, index: idx - nc }
        } else {
            Node { hex: hi, kind: NodeKind::Offboard, index: idx - nc - nt }
        })
    }

    /// Price recorded routes that carry no revenue (Ruby `revenue_for` per
    /// route + the run's best destination bonus on its route).
    pub(crate) fn g1822_price_recorded_routes(&self, c: &str, routes: &[RouteData]) -> Result<Vec<i32>, crate::actions::GameError> {
        let ctx = Ctx::new(self, c);
        let home = self.g1822_home_node(c);
        let dest = self.g1822_destination_node(c);
        let major = self.is_major(c);
        let ci = self.ci(c);
        let mut base = Vec::new();
        let mut bonus = Vec::new();
        for r in routes {
            let t = self.corporations[ci]
                .trains
                .iter()
                .find(|t| t.id == r.train_name)
                .ok_or_else(|| crate::actions::GameError::new(format!("unknown train {}", r.train_name)))?;
            let single = r.connections.len() == 1 && r.connections[0].first().map(String::as_str) == Some("local");
            let stops: Vec<Node> = r.nodes.iter().filter_map(|s| self.g1822_node_from_signature(s)).collect();
            let (rev, b) = self
                .g1822_price_route(&ctx, &stops, single, Self::g1822_train_kind(t), major, home, dest)
                .ok_or_else(|| crate::actions::GameError::new(format!("illegal route for {}", r.train_name)))?;
            base.push(rev);
            bonus.push(b);
        }
        // G1822#destination_bonus: `sort_by { revenue }.reverse.first` —
        // sort_by is stable in MRI, so among equal bonuses the LAST route
        // takes it (it matters to the 2P extra-train dividend choice).
        if let Some((i, &b)) = bonus.iter().enumerate().max_by_key(|(i, b)| (**b, *i)) {
            if b > 0 {
                base[i] += b;
            }
        }
        Ok(base)
    }

    /// Debug: for each recorded route (by its stop signatures), whether the
    /// router enumerated it and at what price for its train.
    /// Debug: the router's chains out of node `sig` ("<hex>-<i>") for `c`.
    pub(crate) fn g1822_router_segs_debug(&self, c: &str, sig: &str) -> Vec<String> {
        let ctx = Ctx::new(self, c);
        let Some(n) = self.g1822_node_from_signature(sig) else { return vec![format!("no node {sig}")] };
        ctx.segs_from(n)
            .iter()
            .map(|s| {
                let paths: Vec<String> = s
                    .paths
                    .iter()
                    .map(|&p| format!("{}{:?}{}", self.hexes[p.0].id, ctx.edges_of(p), if ctx.pd(p).terminal { "T" } else { "" }))
                    .collect();
                format!(
                    "-> {} [{}] first={} last={} inner={}",
                    self.g1822_node_signature(s.to),
                    paths.join(" "),
                    s.terminal_first,
                    s.terminal_last,
                    s.terminal_inner
                )
            })
            .collect()
    }

    pub(crate) fn g1822_router_explain(&self, c: &str, routes: &[RouteData]) -> Vec<String> {
        let ctx = Ctx::new(self, c);
        let graph = self.g1822_graph(c);
        let tokens: HashSet<Node> = self.g1822_token_nodes(c).into_iter().collect();
        let mut found: Vec<Walked> = Vec::new();
        let mut starts: Vec<Node> = graph.connected_nodes.iter().copied().collect();
        starts.sort();
        let mut spent = 0u64;
        for (si, &s) in starts.iter().enumerate() {
            let mut steps = Budget::share(spent, starts.len() - si);
            let mut stack = Walked { stops: vec![s], segs: Vec::new() };
            let mut used = HashMap::new();
            let b = Bound { total: MAX_STOPS_UNBOUNDED, cities: MAX_STOPS_UNBOUNDED };
            let mut sink = |w: &Walked| {
                if found.len() < 200_000 {
                    found.push(w.clone());
                }
            };
            self.g1822_route_dfs(&ctx, &mut stack, &mut used, b, &tokens, &mut sink, &mut steps);
            spent += steps.used.min(steps.limit);
        }
        let mut out = Vec::new();
        let ci = self.ci(c);
        for r in routes {
            let stops: Vec<Node> = r.nodes.iter().filter_map(|s| self.g1822_node_from_signature(s)).collect();
            let mut sorted = stops.clone();
            sorted.sort();
            let hit = found.iter().any(|w| {
                let mut v = w.stops.clone();
                v.sort();
                v == sorted
            });
            let t = self.corporations[ci].trains.iter().find(|t| t.id == r.train_name);
            let price = t.and_then(|t| {
                self.g1822_price_route(
                    &ctx,
                    &stops,
                    r.connections.len() == 1 && r.connections[0].first().map(String::as_str) == Some("local"),
                    Self::g1822_train_kind(t),
                    self.is_major(c),
                    self.g1822_home_node(c),
                    self.g1822_destination_node(c),
                )
            });
            let starts_connected = stops.iter().all(|n| graph.connected_nodes.contains(n));
            out.push(format!(
                "{}: enumerated={} price={:?} all_stops_connected={} found_total={}",
                r.train_name,
                hit,
                price,
                starts_connected,
                found.len()
            ));
        }
        out
    }

    /// The best run for `c`'s route trains: (RouteData per train run,
    /// total revenue incl. the destination bonus).
    pub(crate) fn g1822_best_routes(&self, c: &str) -> (Vec<RouteData>, i32) {
        let trains = self.g1822_route_trains(c);
        if trains.is_empty() {
            return (Vec::new(), 0);
        }
        let kinds: Vec<Kind> = trains.iter().map(Self::g1822_train_kind).collect();
        // Walk bounds: total stops, and stops that are not towns (cities and
        // offboards; France is free so allow one extra).
        let bound = kinds.iter().fold(Bound { total: 0, cities: 0 }, |b, k| {
            let (t, c) = match k {
                Kind::Local { .. } => (2, 1),
                Kind::Numeric(d) => (*d as usize + 1, *d as usize + 1),
                Kind::Pullman(d) => (MAX_STOPS_UNBOUNDED, *d as usize + 1),
                Kind::E => (MAX_STOPS_UNBOUNDED, MAX_STOPS_UNBOUNDED),
            };
            Bound { total: b.total.max(t), cities: b.cities.max(c) }
        });

        let ctx = Ctx::new(self, c);
        let graph = self.g1822_graph(c);
        let mut starts: Vec<Node> = graph.connected_nodes.iter().copied().collect();
        starts.sort();
        let tokens: HashSet<Node> = self.g1822_token_nodes(c).into_iter().collect();

        // -- enumerate + price -------------------------------------------
        // Each walked route is priced for every distinct train kind on the
        // spot; a kind keeps its TOP_K routes by value (memory stays
        // bounded on dense late-game maps; the DFS has a step budget too).
        let dest_bonus_ok = self.is_major(c);
        let home = self.g1822_home_node(c);
        let dest = self.g1822_destination_node(c);
        let mut distinct: Vec<Kind> = Vec::new();
        for k in &kinds {
            if !distinct.contains(k) {
                distinct.push(*k);
            }
        }
        let mut routes: Vec<Walked> = Vec::new();
        // per distinct kind: (value, revenue, bonus, route index)
        let mut tops: Vec<Vec<(i32, i32, i32, usize)>> = vec![Vec::new(); distinct.len()];
        let consider = |w: &Walked, routes: &mut Vec<Walked>, tops: &mut Vec<Vec<(i32, i32, i32, usize)>>| {
            let mut idx: Option<usize> = None;
            for (ki, k) in distinct.iter().enumerate() {
                let Some((rev, bonus)) =
                    self.g1822_price_route(&ctx, &w.stops, w.segs.is_empty(), *k, dest_bonus_ok, home, dest)
                else {
                    continue;
                };
                if rev + bonus <= 0 {
                    continue;
                }
                let v = &mut tops[ki];
                if v.len() >= TOP_K && v.last().map_or(false, |l| l.0 >= rev + bonus) {
                    continue;
                }
                let ri = *idx.get_or_insert_with(|| {
                    routes.push(w.clone());
                    routes.len() - 1
                });
                let pos = v.partition_point(|e| e.0 >= rev + bonus);
                v.insert(pos, (rev + bonus, rev, bonus, ri));
                if v.len() > TOP_K {
                    v.pop();
                }
            }
        };
        let mut spent = 0u64;
        for (si, &s) in starts.iter().enumerate() {
            let mut steps = Budget::share(spent, starts.len() - si);
            let mut stack = Walked { stops: vec![s], segs: Vec::new() };
            let mut used: HashMap<(usize, u8), u8> = HashMap::new();
            let mut sink = |w: &Walked| consider(w, &mut routes, &mut tops);
            self.g1822_route_dfs(&ctx, &mut stack, &mut used, bound, &tokens, &mut sink, &mut steps);
            spent += steps.used.min(steps.limit);
        }
        // E-trains pay tokened cities only, so their best runs are long
        // chains between token cities that the bounded walk cannot reach:
        // add token-to-token tours (each hop a shortest free connection).
        if kinds.contains(&Kind::E) {
            for w in self.g1822_e_tours(&ctx, &graph.connected_nodes) {
                consider(&w, &mut routes, &mut tops);
            }
        }
        // single-city local runs
        if kinds.iter().any(|k| matches!(k, Kind::Local { .. })) {
            let mut ts: Vec<Node> = tokens.iter().copied().collect();
            ts.sort();
            for t in ts {
                consider(&Walked { stops: vec![t], segs: Vec::new() }, &mut routes, &mut tops);
            }
        }
        let per_train: Vec<Vec<Priced>> = kinds
            .iter()
            .map(|k| {
                let ki = distinct.iter().position(|d| d == k).unwrap();
                tops[ki]
                    .iter()
                    .map(|&(_, rev, bonus, ri)| Priced { route: ri, revenue: rev, bonus })
                    .collect()
            })
            .collect();

        // -- combine -----------------------------------------------------
        let mut best_total = 0;
        let mut best_pick: Vec<Option<usize>> = vec![None; trains.len()];
        let mut pick: Vec<Option<usize>> = vec![None; trains.len()];
        let caps = |key: &(usize, u8)| -> u8 {
            // capacity: the lanes of the path(s) at this hexside
            let (h, e) = *key;
            let tile = &self.hexes[h].tile;
            (0..tile.paths.len())
                .filter(|&i| ctx.edges_of((h, i)).contains(&e))
                .map(|i| *ctx.lanes.get(&(h, i)).unwrap_or(&1))
                .max()
                .unwrap_or(1)
        };
        let mut cap_cache: HashMap<(usize, u8), u8> = HashMap::new();
        for r in &routes {
            for s in &r.segs {
                for k in &s.track {
                    cap_cache.entry(*k).or_insert_with(|| caps(k));
                }
            }
        }
        let upper: Vec<i32> = per_train.iter().map(|v| v.first().map_or(0, |p| p.revenue + p.bonus)).collect();
        let mut st = Search {
            routes: &routes,
            per_train: &per_train,
            kinds: &kinds,
            caps: &cap_cache,
            used_normal: HashMap::new(),
            used_e: HashMap::new(),
            local_cities: HashSet::new(),
            merthyr: [false, false],
            upper: &upper,
            merthyr_hex: self.hex_idx.get(data::MERTHYR_TYDFIL_PONTYPOOL_HEX).copied(),
        };
        st.search(0, 0, 0, &mut pick, &mut best_total, &mut best_pick);

        let mut out = Vec::new();
        for (ti, p) in best_pick.iter().enumerate() {
            let Some(pi) = p else { continue };
            let pr = &per_train[ti][*pi];
            let r = &routes[pr.route];
            out.push(self.g1822_route_data(&ctx, r, &trains[ti].id, pr.revenue));
        }
        // The destination bonus rides on the route that earns the biggest
        // (ties: the last one, as Ruby's stable sort_by + reverse picks).
        let mut bonus_best: Option<(usize, i32)> = None;
        for (oi, (ti, p)) in best_pick.iter().enumerate().filter_map(|(ti, p)| p.map(|p| (ti, p))).enumerate() {
            let b = per_train[ti][p].bonus;
            if b > 0 && bonus_best.map_or(true, |(_, bb)| b >= bb) {
                bonus_best = Some((oi, b));
            }
        }
        if let Some((oi, b)) = bonus_best {
            if let Some(rv) = out[oi].revenue.as_mut() {
                *rv += b;
            }
        }
        (out, best_total)
    }

    #[allow(clippy::too_many_arguments)]
    fn g1822_route_dfs(
        &self,
        ctx: &Ctx,
        cur: &mut Walked,
        used: &mut HashMap<(usize, u8), u8>,
        bound: Bound,
        tokens: &HashSet<Node>,
        sink: &mut dyn FnMut(&Walked),
        steps: &mut Budget,
    ) {
        steps.used += 1;
        if steps.used > steps.limit || cur.stops.len() >= bound.total {
            return;
        }
        let cities = cur.stops.iter().filter(|n| n.kind != NodeKind::Town).count();
        let last = *cur.stops.last().unwrap();
        // pass-through: an interior node must not block
        if cur.stops.len() > 1 && ctx.blocks(last) {
            return;
        }
        if let Some(s) = cur.segs.last() {
            if s.terminal_last {
                return;
            }
        }
        for seg in ctx.segs_from(last) {
            if seg.terminal_inner {
                continue;
            }
            if seg.terminal_first && !cur.segs.is_empty() {
                continue;
            }
            if cur.stops.contains(&seg.to) {
                continue;
            }
            if seg.to.kind != NodeKind::Town && cities + 1 > bound.cities {
                continue;
            }
            // groups:London (every London city; Hex#lay carries the group
            // to upgraded cities): one stop per group.
            if ctx.london == Some(seg.to.hex)
                && seg.to.kind == NodeKind::City
                && cur.stops.iter().any(|n| n.hex == seg.to.hex && n.kind == NodeKind::City)
            {
                continue;
            }
            if seg.track.iter().any(|k| used.get(k).copied().unwrap_or(0) > 0) {
                continue;
            }
            for k in &seg.track {
                *used.entry(*k).or_insert(0) += 1;
            }
            cur.stops.push(seg.to);
            cur.segs.push(seg.clone());
            // record each route once: from its lower-keyed end (a route
            // never revisits a node, so the two ends differ)
            if cur.stops.iter().any(|n| tokens.contains(n)) {
                let first = cur.stops[0];
                let lastn = *cur.stops.last().unwrap();
                if first <= lastn && self.g1822_hexes_ok(&cur.stops) {
                    sink(cur);
                }
            }
            self.g1822_route_dfs(ctx, cur, used, bound, tokens, sink, steps);
            cur.segs.pop();
            cur.stops.pop();
            for k in &seg.track {
                if let Some(v) = used.get_mut(k) {
                    *v -= 1;
                }
            }
        }
    }

    /// Token-to-token tours for an E-train: from every token city, visit
    /// further token cities one hop at a time, each hop the fewest-chain
    /// connection that reuses no node or track of the tour so far and
    /// passes no blocking node; every tour ending on a token city with two
    /// or more of them is a candidate (priced as Kind::E later).
    fn g1822_e_tours(&self, ctx: &Ctx, connected: &HashSet<Node>) -> Vec<Walked> {
        let mut toks: Vec<Node> = connected.iter().copied().filter(|n| ctx.tokened_normal_or_dest(*n)).collect();
        toks.sort();
        let mut out: Vec<Walked> = Vec::new();
        let mut budget = 20_000usize;
        for &t0 in &toks {
            let mut tour = Walked { stops: vec![t0], segs: Vec::new() };
            let mut used: HashSet<(usize, u8)> = HashSet::new();
            self.g1822_e_extend(ctx, &toks, &mut tour, &mut used, &mut out, &mut budget);
        }
        out
    }

    fn g1822_e_extend(
        &self,
        ctx: &Ctx,
        toks: &[Node],
        tour: &mut Walked,
        used: &mut HashSet<(usize, u8)>,
        out: &mut Vec<Walked>,
        budget: &mut usize,
    ) {
        if *budget == 0 {
            return;
        }
        *budget -= 1;
        let last = *tour.stops.last().unwrap();
        for &t in toks {
            if tour.stops.contains(&t) {
                continue;
            }
            let Some(hop) = self.g1822_shortest_hop(ctx, last, t, tour, used) else { continue };
            for s in &hop {
                for k in &s.track {
                    used.insert(*k);
                }
                tour.stops.push(s.to);
                tour.segs.push(s.clone());
            }
            let first = tour.stops[0];
            if first <= t && self.g1822_hexes_ok(&tour.stops) {
                out.push(tour.clone());
            }
            // a hop that arrives along a terminal path ends the tour
            // (check_terminals!: terminal paths only first or last)
            if !hop.last().map_or(false, |s| s.terminal_last) {
                self.g1822_e_extend(ctx, toks, tour, used, out, budget);
            }
            for s in hop.iter().rev() {
                tour.stops.pop();
                tour.segs.pop();
                for k in &s.track {
                    used.remove(k);
                }
            }
        }
    }

    /// BFS over chains from `from` to `to` avoiding the tour's nodes and
    /// track; interior nodes must not block; terminal paths only at the
    /// ends; one London city.
    fn g1822_shortest_hop(
        &self,
        ctx: &Ctx,
        from: Node,
        to: Node,
        tour: &Walked,
        used: &HashSet<(usize, u8)>,
    ) -> Option<Vec<Seg>> {
        use std::collections::VecDeque;
        let mut prev: HashMap<Node, (Node, Seg)> = HashMap::new();
        let mut q: VecDeque<Node> = VecDeque::new();
        q.push_back(from);
        let london_used = tour.stops.iter().any(|n| ctx.london == Some(n.hex) && n.kind == NodeKind::City);
        while let Some(n) = q.pop_front() {
            if n == to {
                break;
            }
            if n != from && ctx.blocks(n) {
                continue;
            }
            for seg in ctx.segs_from(n) {
                let m = seg.to;
                if m == from || prev.contains_key(&m) || tour.stops.contains(&m) {
                    continue;
                }
                if seg.terminal_inner || seg.track.iter().any(|k| used.contains(k)) {
                    continue;
                }
                if seg.terminal_last && m != to {
                    continue;
                }
                if seg.terminal_first && !(n == from && tour.segs.is_empty()) {
                    continue;
                }
                if london_used && ctx.london == Some(m.hex) && m.kind == NodeKind::City {
                    continue;
                }
                prev.insert(m, (n, seg));
                q.push_back(m);
            }
        }
        if !prev.contains_key(&to) {
            return None;
        }
        let mut hop: Vec<Seg> = Vec::new();
        let mut cur = to;
        while cur != from {
            let (p, seg) = prev.get(&cur)?.clone();
            hop.push(seg);
            cur = p;
        }
        hop.reverse();
        // the hop must not reuse its own track either
        let mut seen: HashSet<(usize, u8)> = HashSet::new();
        for s in &hop {
            for k in &s.track {
                if !seen.insert(*k) {
                    return None;
                }
            }
        }
        Some(hop)
    }

    /// revenue_for's 'Route visits same hex twice' (stop hexes chunked;
    /// skipped when a DOUBLE_HEX hex is on the route).
    fn g1822_hexes_ok(&self, stops: &[Node]) -> bool {
        let mut hexes: Vec<usize> = Vec::new();
        for n in stops {
            if hexes.last() != Some(&n.hex) {
                hexes.push(n.hex);
            }
        }
        if hexes.iter().any(|&h| DOUBLE_HEX.contains(&self.hexes[h].id.as_str())) {
            return true;
        }
        let set: HashSet<usize> = hexes.iter().copied().collect();
        set.len() == hexes.len()
    }

    /// The city holding the corporation's placed destination token.
    pub(crate) fn g1822_destination_node(&self, c: &str) -> Option<Node> {
        for (hi, hex) in self.hexes.iter().enumerate() {
            for (ci, city) in hex.tile.cities.iter().enumerate() {
                if city.tokens.iter().flatten().any(|t| t.corporation_id == c && t.token_type == "destination") {
                    return Some(Node { hex: hi, kind: NodeKind::City, index: ci });
                }
            }
        }
        None
    }

    /// Price route `r` for a train kind: Some((revenue, destination bonus))
    /// when the train may run it.
    #[allow(clippy::too_many_arguments)]
    fn g1822_price_route(
        &self,
        ctx: &Ctx,
        stops: &[Node],
        single: bool,
        kind: Kind,
        dest_ok: bool,
        home: Option<Node>,
        dest: Option<Node>,
    ) -> Option<(i32, i32)> {
        let _ = single;
        // "Cannot use group London more than once"
        if stops.iter().filter(|n| ctx.london == Some(n.hex) && n.kind == NodeKind::City).count() > 1 {
            return None;
        }
        let channel_hex = |n: &Node| {
            let id = self.hexes[n.hex].id.as_str();
            id == data::ENGLISH_CHANNEL_HEX || id == data::FRANCE_HEX
        };
        let channel = stops.iter().filter(|n| channel_hex(n)).count();
        let local = matches!(kind, Kind::Local { .. });
        // a single-node route is a local train's only
        if stops.len() < 2 && !local {
            return None;
        }
        // check_route_token: a city the corporation tokened
        if !stops.iter().any(|n| ctx.tokened(*n)) {
            return None;
        }
        // check_distance (G1822): channel rules first
        if channel > 0 {
            if local {
                return None;
            }
            if channel == 1 {
                return None;
            }
            if channel == 2 && stops.len() == 2 {
                return None;
            }
        }
        let multiplier = if kind == Kind::E { 2 } else { 1 };
        let mut revenue = 0;
        match kind {
            Kind::Local { .. } => {
                let cities = stops.iter().filter(|n| n.kind != NodeKind::Town).count();
                let towns = stops.iter().filter(|n| n.kind == NodeKind::Town).count();
                if cities > 1 || towns > 1 {
                    return None;
                }
                revenue = stops.iter().map(|n| ctx.stop_revenue(*n)).sum();
            }
            Kind::Numeric(d) => {
                let cost: u32 = stops.iter().map(|n| ctx.visit_cost(*n)).sum();
                if cost > d {
                    return None;
                }
                revenue = stops.iter().map(|n| ctx.stop_revenue(*n)).sum();
            }
            Kind::Pullman(d) => {
                let cost: u32 = stops.iter().filter(|n| n.kind != NodeKind::Town).map(|n| ctx.visit_cost(*n)).sum();
                if cost > d {
                    return None;
                }
                revenue = stops.iter().map(|n| ctx.stop_revenue(*n)).sum();
            }
            Kind::E => {
                let tokened: Vec<bool> = stops.iter().map(|n| ctx.tokened_normal_or_dest(*n)).collect();
                if tokened.iter().filter(|&&t| t).count() < 2 {
                    return None;
                }
                let ends = tokened[0] as usize + tokened[tokened.len() - 1] as usize;
                let valid_ends = match ends {
                    2 => true,
                    1 => {
                        // english_channel_etrain?: both channel stops and a
                        // token in the Channel city
                        channel == 2
                            && self.hex_idx.get(data::ENGLISH_CHANNEL_HEX).map_or(false, |&h| {
                                self.hexes[h].tile.cities.first().map_or(false, |c| {
                                    c.tokens.iter().flatten().any(|t| t.corporation_id == ctx.corp)
                                })
                            })
                    }
                    _ => false,
                };
                if !valid_ends {
                    return None;
                }
                let france = stops.iter().find(|n| self.hexes[n.hex].id == data::FRANCE_HEX && n.kind == NodeKind::Offboard);
                for (n, t) in stops.iter().zip(&tokened) {
                    if n.kind != NodeKind::City || !t {
                        continue;
                    }
                    if self.hexes[n.hex].id == data::ENGLISH_CHANNEL_HEX {
                        revenue += france.map_or(0, |f| ctx.stop_revenue(*f) * multiplier);
                    } else {
                        revenue += ctx.stop_revenue(*n) * multiplier;
                    }
                }
            }
        }
        // destination bonus: home + destination token on this route
        let mut bonus = 0;
        if dest_ok {
            if let (Some(h), Some(d)) = (home, dest) {
                if stops.contains(&h) && stops.contains(&d) {
                    bonus = ctx.stop_revenue(d) * multiplier;
                }
            }
        }
        Some((revenue, bonus))
    }

    /// The run_routes wire entry for a walked route.
    fn g1822_route_data(&self, ctx: &Ctx, r: &Walked, train_id: &str, revenue: i32) -> RouteData {
        let mut hexes: Vec<String> = Vec::new();
        for n in &r.stops {
            let id = self.hexes[n.hex].id.clone();
            if hexes.last() != Some(&id) {
                hexes.push(id);
            }
        }
        let connections: Vec<Vec<String>> = if r.segs.is_empty() {
            vec![vec!["local".to_string(), self.hexes[r.stops[0].hex].id.clone()]]
        } else {
            r.segs
                .iter()
                .map(|s| {
                    // Route#chain_id: one hex per path, skipping junction
                    // paths after the first through a junction.
                    let mut out: Vec<String> = Vec::new();
                    let mut junction_hexes: HashSet<usize> = HashSet::new();
                    for &p in &s.paths {
                        let j = ctx.is_junction(p);
                        if j && junction_hexes.contains(&p.0) {
                            continue;
                        }
                        if j {
                            junction_hexes.insert(p.0);
                        }
                        out.push(self.hexes[p.0].id.clone());
                    }
                    out
                })
                .collect()
        };
        let nodes: Vec<String> = r.stops.iter().map(|n| self.g1822_node_signature(*n)).collect();
        RouteData {
            train_name: train_id.to_string(),
            hexes,
            revenue: Some(revenue),
            connections,
            nodes,
        }
    }
}

struct Search<'s> {
    routes: &'s [Walked],
    per_train: &'s [Vec<Priced>],
    kinds: &'s [Kind],
    caps: &'s HashMap<(usize, u8), u8>,
    used_normal: HashMap<(usize, u8), u8>,
    used_e: HashMap<(usize, u8), u8>,
    local_cities: HashSet<Node>,
    merthyr: [bool; 2],
    upper: &'s [i32],
    merthyr_hex: Option<usize>,
}

impl Search<'_> {
    fn fits(&self, t: usize, r: &Walked) -> bool {
        let used = if self.kinds[t] == Kind::E { &self.used_e } else { &self.used_normal };
        for s in &r.segs {
            for k in &s.track {
                if used.get(k).copied().unwrap_or(0) >= self.caps.get(k).copied().unwrap_or(1) {
                    return false;
                }
            }
        }
        if matches!(self.kinds[t], Kind::Local { .. }) {
            for n in &r.stops {
                if n.kind == NodeKind::City && self.local_cities.contains(n) {
                    return false;
                }
            }
        }
        if let Some(mh) = self.merthyr_hex {
            let (m1, m2) = self.merthyr_use(r, mh);
            if (m1 || self.merthyr[0]) && (m2 || self.merthyr[1]) {
                return false;
            }
        }
        true
    }

    fn merthyr_use(&self, r: &Walked, mh: usize) -> (bool, bool) {
        let mut m = (false, false);
        for s in &r.segs {
            for &(h, e) in &s.track {
                if h == mh {
                    if e == 1 {
                        m.0 = true;
                    }
                    if e == 2 {
                        m.1 = true;
                    }
                }
            }
        }
        m
    }

    fn apply(&mut self, t: usize, r: &Walked, sign: i8) {
        let e = self.kinds[t] == Kind::E;
        for s in &r.segs {
            for k in &s.track {
                let used = if e { &mut self.used_e } else { &mut self.used_normal };
                let v = used.entry(*k).or_insert(0);
                if sign > 0 {
                    *v += 1;
                } else {
                    *v -= 1;
                }
            }
        }
        if matches!(self.kinds[t], Kind::Local { .. }) {
            for n in &r.stops {
                if n.kind == NodeKind::City {
                    if sign > 0 {
                        self.local_cities.insert(*n);
                    } else {
                        self.local_cities.remove(n);
                    }
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn search(
        &mut self,
        t: usize,
        revenue: i32,
        bonus: i32,
        pick: &mut Vec<Option<usize>>,
        best: &mut i32,
        best_pick: &mut Vec<Option<usize>>,
    ) {
        let total = revenue + bonus;
        if total > *best {
            *best = total;
            best_pick.clone_from(pick);
        }
        if t == self.per_train.len() {
            return;
        }
        let rest: i32 = self.upper[t..].iter().sum();
        if total + rest <= *best {
            return;
        }
        for pi in 0..self.per_train[t].len() {
            let p = self.per_train[t][pi].clone();
            let rest_after: i32 = self.upper[t + 1..].iter().sum();
            // monotone in the list order (sorted by revenue + bonus)
            if revenue + bonus + p.revenue + p.bonus + rest_after <= *best {
                break;
            }
            let r = &self.routes[p.route];
            if !self.fits(t, r) {
                continue;
            }
            let saved_m = self.merthyr;
            if let Some(mh) = self.merthyr_hex {
                let (m1, m2) = self.merthyr_use(r, mh);
                self.merthyr[0] |= m1;
                self.merthyr[1] |= m2;
            }
            self.apply(t, r, 1);
            pick[t] = Some(pi);
            self.search(t + 1, revenue + p.revenue, bonus.max(p.bonus), pick, best, best_pick);
            pick[t] = None;
            self.apply(t, r, -1);
            self.merthyr = saved_m;
        }
        // this train runs nothing
        self.search(t + 1, revenue, bonus, pick, best, best_pick);
    }
}
