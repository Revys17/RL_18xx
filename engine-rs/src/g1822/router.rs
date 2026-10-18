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
use super::family::Family;

use super::graph::{Node, NodeKind};

/// Bound on a route's stops for the unbounded trains (E, pullman towns):
/// keeps the walk finite on dense maps.
const MAX_STOPS_UNBOUNDED: usize = 32;
/// Routes kept per train kind for the combination search.
const TOP_K: usize = 4000;
/// 1822CA: its attached long trains hold many more routes above a
/// threshold.
const TOP_K_CA: usize = 16000;
/// 1822CA: search nodes one combination search may visit.
const COMBO_BUDGET_CA: u64 = 150_000_000;
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

/// A walked route (1822PNW: with its mill and lumber-baron bonus
/// candidates — each run pays only its biggest one, like the destination
/// bonus).
#[derive(Clone, Debug, Default)]
struct Walked {
    stops: Vec<Node>,
    segs: Vec<Seg>,
    mill: i32,
    lumber: i32,
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
    /// 1822CA: a train with the pullman attached ("<name>+<2d>"): at most
    /// `d` cities / offboards, towns up to 2d plus the cities' unused
    /// share; every stop pays.
    CaPullman(u32),
    /// 1822CA: a train with the grain train attached ("<name>G"): up to 12
    /// grain-elevator stops besides its own distance (`d`, or an L/LP's
    /// one city + one town); never a single-city local run.
    CaGrain { d: u32, local: bool },
}

/// 1822CA GRAIN_ELEVATOR_COUNT.
const CA_GRAIN_STOPS: u32 = 12;

/// 1822CA's revenue extras for one corporation's routing call:
/// `grain_and_port_bonus` (per route) and the sawmill (per run, the
/// biggest — like the destination bonus).
struct CaR {
    elev: Vec<bool>,
    east: Vec<bool>,
    west: Vec<bool>,
    major: bool,
    sawmill_hex: Option<usize>,
    /// `@sawmill_bonus` if this corporation receives it, else 0.
    sawmill_amount: i32,
}

impl CaR {
    /// `grain_and_port_bonus(train, stops)`.
    fn grain(&self, kind: Kind, stops: &[Node]) -> i32 {
        if !self.major || matches!(kind, Kind::E | Kind::CaPullman(_)) {
            return 0;
        }
        let elevators = stops.iter().filter(|n| self.elev[n.hex]).count() as i32;
        if elevators == 0 {
            return 0;
        }
        let ports = stops.iter().any(|n| self.east[n.hex]) as i32 + stops.iter().any(|n| self.west[n.hex]) as i32;
        match kind {
            Kind::CaGrain { .. } => 20 * ports,
            _ => 10 * elevators + 10 * ports,
        }
    }

    /// `calculate_sawmill_bonus(route)`: `dest_here` = this route takes the
    /// run's destination bonus.
    fn sawmill(&self, ctx: &Ctx, kind: Kind, stops: &[Node], dest_here: bool) -> i32 {
        let Some(h) = self.sawmill_hex else { return 0 };
        if self.sawmill_amount == 0 {
            return 0;
        }
        let Some(&stop) = stops.iter().find(|n| n.hex == h) else { return 0 };
        let e = kind == Kind::E;
        if e && !ctx.tokened(stop) {
            return 0;
        }
        let dest_tok = stop.kind == NodeKind::City
            && ctx.g.hexes[stop.hex].tile.cities[stop.index]
                .tokens
                .iter()
                .flatten()
                .any(|t| t.corporation_id == ctx.corp && t.token_type == "destination");
        let mut m = if dest_tok && dest_here { 2 } else { 1 };
        if e {
            m *= 2;
        }
        self.sawmill_amount * m
    }

    /// The most a route of `kind` can collect: (grain / port, sawmill).
    fn caps(&self, kind: Kind, elev_nodes: usize, max_stops: usize) -> (i32, i32) {
        let g = if !self.major || matches!(kind, Kind::E | Kind::CaPullman(_)) {
            0
        } else if matches!(kind, Kind::CaGrain { .. }) {
            40
        } else {
            10 * elev_nodes.min(max_stops) as i32 + 20
        };
        let m = if self.sawmill_hex.is_some() { self.sawmill_amount * if kind == Kind::E { 4 } else { 2 } } else { 0 };
        (g, m)
    }
}

struct Ctx<'a> {
    g: &'a BaseGame,
    corp: &'a str,
    neighbors: HashMap<usize, BTreeMap<u8, usize>>,
    /// Lane width of a path at one of its edges, (hex, path, edge) → width
    /// (only > 1 is stored): `lanes:N` at both ends; 1822PNW also reads
    /// `a_lane` / `b_lane` (Route#check_overlap keys track by lane).
    lanes: HashMap<(usize, usize, u8), u8>,
    seg_cache: std::cell::RefCell<HashMap<Node, Vec<Seg>>>,
    /// The route-group hexes (`groups:London`, ...): one city per group.
    groups: Vec<usize>,
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

/// Per path: the lane width at its a and b ends (`lanes:N` both,
/// `a_lane:W.i` / `b_lane:W.i` one end).
fn path_end_lanes_from_dsl(dsl: &str) -> Vec<(u8, u8)> {
    dsl.split(';')
        .filter_map(|p| p.trim().strip_prefix("path="))
        .map(|attrs| {
            let mut ab = (1u8, 1u8);
            for kv in attrs.split(',') {
                let Some((k, v)) = kv.split_once(':') else { continue };
                let w = || v.split('.').next().and_then(|x| x.parse::<u8>().ok()).unwrap_or(1);
                match k {
                    "lanes" => ab = (w(), w()),
                    "a_lane" => ab.0 = w(),
                    "b_lane" => ab.1 = w(),
                    _ => {}
                }
            }
            ab
        })
        .collect()
}

impl<'a> Ctx<'a> {
    fn new(g: &'a BaseGame, corp: &'a str) -> Ctx<'a> {
        let mut lanes = HashMap::new();
        for (hi, hex) in g.hexes.iter().enumerate() {
            let fam = g.fam();
            let dsl = if hex.tile.id.starts_with("preprinted_") {
                (fam.hex_dsl)(&hex.id).map(|(d, _)| d)
            } else {
                (fam.tile_dsl)(hex.tile.name.split('-').next().unwrap_or(&hex.tile.name)).map(|(d, _)| d)
            };
            if let Some(d) = dsl {
                let paths = &hex.tile.paths;
                if g.g1822_is_pnw() {
                    for (pi, (wa, wb)) in path_end_lanes_from_dsl(d).into_iter().enumerate() {
                        let Some(pd) = paths.get(pi) else { continue };
                        for (end, w) in [(&pd.a, wa), (&pd.b, wb)] {
                            if let (Some(e), true) = (end.edge_num(), w > 1) {
                                lanes.insert((hi, pi, e), w);
                            }
                        }
                    }
                } else {
                    for (pi, l) in path_lanes_from_dsl(d).into_iter().enumerate() {
                        let Some(pd) = paths.get(pi) else { continue };
                        if l > 1 {
                            for e in [&pd.a, &pd.b].iter().filter_map(|x| x.edge_num()) {
                                lanes.insert((hi, pi, e), l);
                            }
                        }
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
            groups: g.fam().route_group_hexes.iter().filter_map(|h| g.hex_idx.get(*h).copied()).collect(),
        }
    }

    /// Whether `hex` is a route-group hex.
    fn is_group(&self, hex: usize) -> bool {
        self.groups.contains(&hex)
    }

    fn pd(&self, p: PathRef) -> &crate::tiles::PathDef {
        &self.g.hexes[p.0].tile.paths[p.1]
    }

    /// How many times track (hex, edge) may be used: its widest lane.
    fn edge_cap(&self, h: usize, e: u8) -> u8 {
        let tile = &self.g.hexes[h].tile;
        (0..tile.paths.len())
            .filter(|&i| self.edges_of((h, i)).contains(&e))
            .map(|i| *self.lanes.get(&(h, i, e)).unwrap_or(&1))
            .max()
            .unwrap_or(1)
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
            .filter(|&i| !self.g.g1822_path_thin(hex, i))
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
                if city.tokened_by(self.corp) {
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
        n.kind == NodeKind::City && self.g.hexes[n.hex].tile.cities[n.index].tokened_by(self.corp)
    }

    /// Ruby `tokened_by?(entity, types: %i[normal destination])`.
    fn tokened_normal_or_dest(&self, n: Node) -> bool {
        n.kind == NodeKind::City
            && self.g.hexes[n.hex].tile.cities[n.index]
                .all_tokens()
                .any(|t| t.corporation_id == self.corp && (t.token_type == "normal" || t.token_type == "destination"))
    }

    fn stop_revenue(&self, n: Node) -> i32 {
        let hex = &self.g.hexes[n.hex];
        let phase = &self.g.phase.tiles;
        match n.kind {
            NodeKind::City => {
                let c = &hex.tile.cities[n.index];
                if hex.tile.id.starts_with("preprinted_") || self.g.g1822_is_pnw() || self.g.g1822_is_mx() {
                    self.g.g1822_tiered_city_revenue(n.hex).unwrap_or(c.revenue)
                } else {
                    c.revenue
                }
            }
            NodeKind::Town => hex.tile.towns[n.index].revenue,
            NodeKind::Offboard => hex.tile.offboards[n.index].phase_revenue(phase),
        }
    }

    fn visit_cost(&self, n: Node) -> u32 {
        let hex = &self.g.hexes[n.hex];
        if Family::is(self.g.fam().france_hex, &hex.id) {
            0
        } else if self.g.g1822_is_ca()
            && hex.tile.id.starts_with("preprinted_")
            && super::ca::DETROIT_TO_DULUTH_HEXES.contains(&hex.id.as_str())
        {
            // 1822CA Q19 / Y27 `visit_cost:99` (until the DD tiles)
            99
        } else {
            1
        }
    }
}

impl BaseGame {
    /// The routing kind of a (possibly attached) train; 1822CA's attached
    /// trains first.
    fn g1822_train_kind_of(&self, t: &crate::entities::Train) -> Kind {
        if self.g1822_is_ca() {
            if let Some((_, extra)) = t.name.split_once('+') {
                if !extra.is_empty() {
                    return Kind::CaPullman(t.distance);
                }
            }
            if t.name.len() > 1 && t.name.ends_with('G') {
                let base = t.name.strip_suffix("-G").or_else(|| t.name.strip_suffix('G')).unwrap_or(&t.name);
                let local = base == "L" || base == "LP";
                return Kind::CaGrain { d: if local { 2 } else { t.distance }, local };
            }
        }
        Self::g1822_train_kind(t)
    }

    /// 1822CA's extras info for `c` (None for other titles).
    fn g1822_ca_router_info(&self, c: &str) -> Option<CaR> {
        if !self.g1822_is_ca() {
            return None;
        }
        let n = self.hexes.len();
        let mark = |list: &[&str]| -> Vec<bool> { (0..n).map(|h| list.contains(&self.hexes[h].id.as_str())).collect() };
        let ca = &self.g().ca;
        let sawmill_hex = ca.sawmill_hex.as_ref().and_then(|h| self.hex_idx.get(h.as_str()).copied());
        // receives_sawmill_bonus?: the closed sawmill's owner only
        let receives = ca.sawmill_owner.as_ref().map_or(true, |o| o == c);
        Some(CaR {
            elev: mark(super::ca::GRAIN_ELEVATORS),
            east: mark(super::ca::EAST_PORTS),
            west: mark(super::ca::WEST_PORTS),
            major: self.is_major(c),
            sawmill_hex,
            sawmill_amount: if receives { ca.sawmill_bonus } else { 0 },
        })
    }

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
        let pnw = self.g1822_pnw_router_info(c);
        let ca = self.g1822_ca_router_info(c);
        let mut ca_runs: Vec<(Kind, Vec<Node>)> = Vec::new();
        let mut side: Vec<(i32, i32)> = Vec::new();
        let mut halved: Vec<bool> = Vec::new();
        for r in routes {
            let t = self.corporations[ci]
                .trains
                .iter()
                .find(|t| t.id == r.train_name)
                .ok_or_else(|| crate::actions::GameError::new(format!("unknown train {}", r.train_name)))?;
            halved.push(t.name.starts_with("3/2P"));
            let single = r.connections.len() == 1 && r.connections[0].first().map(String::as_str) == Some("local");
            let stops: Vec<Node> = r.nodes.iter().filter_map(|s| self.g1822_node_from_signature(s)).collect();
            let kind = self.g1822_train_kind_of(t);
            let (rev, b) = self
                .g1822_price_route(&ctx, &stops, single, kind, major, home, dest)
                .ok_or_else(|| crate::actions::GameError::new(format!("illegal route for {}", r.train_name)))?;
            // 1822PNW extras over Route#all_hexes (the connections' hexes)
            let (x, m, l) = match &pnw {
                Some(p) => {
                    let hexes: HashSet<usize> = r
                        .connections
                        .iter()
                        .flatten()
                        .filter_map(|h| self.hex_idx.get(h.as_str()).copied())
                        .collect();
                    let none = HashSet::new();
                    p.extras(kind, &stops, if single { &none } else { &hexes })
                }
                None => (0, 0, 0),
            };
            // 1822CA grain_and_port_bonus (per route)
            let gx = ca.as_ref().map_or(0, |p| p.grain(kind, &stops));
            ca_runs.push((kind, stops.clone()));
            base.push(rev + x + gx);
            bonus.push(b);
            side.push((m, l));
        }
        // 1822CA sawmill_bonus: the biggest, on its route (doubled for a
        // destination city when that route also takes the destination bonus)
        if let Some(p) = &ca {
            let dest_route = bonus.iter().enumerate().filter(|(_, b)| **b > 0).max_by_key(|(i, b)| (**b, *i)).map(|(i, _)| i);
            for (i, (kind, stops)) in ca_runs.iter().enumerate() {
                side[i].0 = p.sawmill(&ctx, *kind, stops, dest_route == Some(i));
            }
        }
        // 1822PNW mill / lumber-baron bonuses: the biggest, on its route
        for ch in 0..2 {
            let vals: Vec<i32> = side.iter().map(|v| if ch == 0 { v.0 } else { v.1 }).collect();
            if let Some((i, &v)) = vals.iter().enumerate().max_by_key(|(i, v)| (**v, *i)) {
                if v > 0 {
                    base[i] += v;
                }
            }
        }
        // G1822#destination_bonus: `sort_by { revenue }.reverse.first` —
        // sort_by is stable in MRI, so among equal bonuses the LAST route
        // takes it (it matters to the 2P extra-train dividend choice).
        if let Some((i, &b)) = bonus.iter().enumerate().max_by_key(|(i, b)| (**b, *i)) {
            if b > 0 {
                base[i] += b;
            }
        }
        // 1822MX revenue_for: a 3/2P earns half (bonus included)
        for (i, h) in halved.into_iter().enumerate() {
            if h {
                base[i] = half_revenue(base[i]);
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
            let mut stack = Walked { stops: vec![s], segs: Vec::new(), mill: 0, lumber: 0 };
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
                    self.g1822_train_kind_of(t),
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
        self.g1822_best_routes_v2(c)
    }

    /// The pre-2026-10-17 router (one walk bounded by the widest train,
    /// every route priced for every kind, a 3M-step budget). Kept for
    /// comparison (`g1822_router_best_legacy`).
    pub(crate) fn g1822_best_routes_legacy(&self, c: &str) -> (Vec<RouteData>, i32) {
        let trains = self.g1822_route_trains(c);
        if trains.is_empty() {
            return (Vec::new(), 0);
        }
        let kinds: Vec<Kind> = trains.iter().map(|t| self.g1822_train_kind_of(t)).collect();
        // Walk bounds: total stops, and stops that are not towns (cities and
        // offboards; France is free so allow one extra).
        let bound = kinds.iter().fold(Bound { total: 0, cities: 0 }, |b, k| {
            let (t, c) = match k {
                Kind::Local { .. } => (2, 1),
                Kind::Numeric(d) => (*d as usize + 1, *d as usize + 1),
                Kind::Pullman(d) => (MAX_STOPS_UNBOUNDED, *d as usize + 1),
                Kind::E => (MAX_STOPS_UNBOUNDED, MAX_STOPS_UNBOUNDED),
                Kind::CaPullman(d) => (3 * *d as usize, *d as usize + 1),
                Kind::CaGrain { d, .. } => ((*d + CA_GRAIN_STOPS) as usize + 1, (*d + CA_GRAIN_STOPS) as usize + 1),
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
            let mut stack = Walked { stops: vec![s], segs: Vec::new(), mill: 0, lumber: 0 };
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
                consider(&Walked { stops: vec![t], segs: Vec::new(), mill: 0, lumber: 0 }, &mut routes, &mut tops);
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
            ctx.edge_cap(h, e)
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
            merthyr_hex: self.hex_idx.get(self.fam().merthyr_tydfil_pontypool_hex).copied(),
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
            if ctx.is_group(seg.to.hex)
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
        if self.g1822_is_ca() {
            return self.g1822_ca_e_tours(ctx, connected);
        }
        let mut toks: Vec<Node> = connected.iter().copied().filter(|n| ctx.tokened_normal_or_dest(*n)).collect();
        toks.sort();
        // France: a tour may end there from the English Channel city
        // (english_channel_etrain?: one tokened end, both channel stops)
        let france = connected
            .iter()
            .copied()
            .find(|n| n.kind == NodeKind::Offboard && Family::is(self.fam().france_hex, &self.hexes[n.hex].id));
        let mut out: Vec<Walked> = Vec::new();
        let mut budget = 20_000usize;
        for &t0 in &toks {
            let mut tour = Walked { stops: vec![t0], segs: Vec::new(), mill: 0, lumber: 0 };
            let mut used: HashSet<(usize, u8)> = HashSet::new();
            self.g1822_e_extend(ctx, &toks, france, &mut tour, &mut used, &mut out, &mut budget);
        }
        out
    }

    #[allow(clippy::too_many_arguments)]
    fn g1822_e_extend(
        &self,
        ctx: &Ctx,
        toks: &[Node],
        france: Option<Node>,
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
            let terminal = hop.last().map_or(false, |s| s.terminal_last);
            // the Channel city on to France ends the tour there
            if let Some(f) = france.filter(|_| !terminal && Family::is(self.fam().english_channel_hex, &self.hexes[t.hex].id)) {
                if !tour.stops.contains(&f) {
                    if let Some(fhop) = self.g1822_shortest_hop(ctx, t, f, tour, used) {
                        let n0 = tour.stops.len();
                        for s in &fhop {
                            tour.stops.push(s.to);
                            tour.segs.push(s.clone());
                        }
                        if self.g1822_hexes_ok(&tour.stops) {
                            out.push(tour.clone());
                        }
                        tour.stops.truncate(n0);
                        tour.segs.truncate(n0 - 1);
                    }
                }
            }
            // a hop that arrives along a terminal path ends the tour
            // (check_terminals!: terminal paths only first or last)
            if !terminal {
                self.g1822_e_extend(ctx, toks, france, tour, used, out, budget);
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

    /// 1822CA's E tours: the same token-to-token tours, searched best-first
    /// (the next token by revenue) with a bound — the tour's paying cities
    /// plus every token city not yet on it — against the best tour found,
    /// and a larger budget (CA's late maps hold a dozen token cities across
    /// the continent).
    fn g1822_ca_e_tours(&self, ctx: &Ctx, connected: &HashSet<Node>) -> Vec<Walked> {
        let mut toks: Vec<Node> = connected.iter().copied().filter(|n| ctx.tokened_normal_or_dest(*n)).collect();
        toks.sort_by_key(|n| (std::cmp::Reverse(ctx.stop_revenue(*n)), *n));
        let mut out: Vec<Walked> = Vec::new();
        let mut budget = 400_000usize;
        let mut best = 0i32;
        for &t0 in &toks {
            let mut tour = Walked { stops: vec![t0], segs: Vec::new(), mill: 0, lumber: 0 };
            let mut used: HashSet<(usize, u8)> = HashSet::new();
            self.g1822_ca_e_extend(ctx, &toks, &mut tour, &mut used, &mut out, &mut budget, &mut best);
        }
        out
    }

    /// Twice the revenue of the tour's token cities (an E-train's pay).
    fn g1822_ca_e_value(&self, ctx: &Ctx, stops: &[Node]) -> i32 {
        stops
            .iter()
            .filter(|n| n.kind == NodeKind::City && ctx.tokened_normal_or_dest(**n))
            .map(|n| 2 * ctx.stop_revenue(*n).max(0))
            .sum()
    }

    #[allow(clippy::too_many_arguments)]
    fn g1822_ca_e_extend(
        &self,
        ctx: &Ctx,
        toks: &[Node],
        tour: &mut Walked,
        used: &mut HashSet<(usize, u8)>,
        out: &mut Vec<Walked>,
        budget: &mut usize,
        best: &mut i32,
    ) {
        if *budget == 0 {
            return;
        }
        *budget -= 1;
        let value = self.g1822_ca_e_value(ctx, &tour.stops);
        let rest: i32 = toks.iter().filter(|t| !tour.stops.contains(t)).map(|t| 2 * ctx.stop_revenue(*t).max(0)).sum();
        // (the destination bonus and the sawmill can add on top: keep a
        // margin of the biggest city's double revenue + 160)
        let margin = toks.iter().map(|t| 2 * ctx.stop_revenue(*t).max(0)).max().unwrap_or(0) + 160;
        if value + rest + margin <= *best {
            return;
        }
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
            if self.g1822_hexes_ok(&tour.stops) {
                let v = self.g1822_ca_e_value(ctx, &tour.stops);
                *best = (*best).max(v);
                if first <= t || v >= *best {
                    out.push(tour.clone());
                }
            }
            let terminal = hop.last().map_or(false, |s| s.terminal_last);
            if !terminal {
                self.g1822_ca_e_extend(ctx, toks, tour, used, out, budget, best);
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
        let groups_used: Vec<usize> =
            tour.stops.iter().filter(|n| ctx.is_group(n.hex) && n.kind == NodeKind::City).map(|n| n.hex).collect();
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
                if m.kind == NodeKind::City && groups_used.contains(&m.hex) {
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
        if hexes.iter().any(|&h| self.fam().double_hex.contains(&self.hexes[h].id.as_str())) {
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
        for &gh in &ctx.groups {
            if stops.iter().filter(|n| n.hex == gh && n.kind == NodeKind::City).count() > 1 {
                return None;
            }
        }
        let fam = self.fam();
        // english_channel_visit: 0 unless both hexes are set
        let channel_hex = |n: &Node| {
            let id = self.hexes[n.hex].id.as_str();
            !fam.english_channel_hex.is_empty()
                && !fam.france_hex.is_empty()
                && (id == fam.english_channel_hex || id == fam.france_hex)
        };
        let channel = stops.iter().filter(|n| channel_hex(n)).count();
        let local = matches!(kind, Kind::Local { .. });
        // (a grain-attached L / LP has no_local)
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
            Kind::CaPullman(d) => {
                // check_distance: cities / offboards first into the mixed
                // row, towns into the 2d town row then what is left of it
                let city_cost: u32 = stops.iter().filter(|n| n.kind != NodeKind::Town).map(|n| ctx.visit_cost(*n)).sum();
                let towns: u32 = stops.iter().filter(|n| n.kind == NodeKind::Town).map(|n| ctx.visit_cost(*n)).sum();
                if city_cost > d || towns > 2 * d + (d - city_cost) {
                    return None;
                }
                revenue = stops.iter().map(|n| ctx.stop_revenue(*n)).sum();
            }
            Kind::CaGrain { d, local: base_local } => {
                let elev = |n: &Node| super::ca::GRAIN_ELEVATORS.contains(&self.hexes[n.hex].id.as_str());
                let grain: u32 = stops.iter().filter(|n| elev(n)).map(|n| ctx.visit_cost(*n)).sum();
                if grain > CA_GRAIN_STOPS {
                    return None;
                }
                let rest: Vec<&Node> = stops.iter().filter(|n| !elev(n)).collect();
                if base_local {
                    // the L / LP's rows: one city, one town (no offboards)
                    let cities: u32 = rest.iter().filter(|n| n.kind == NodeKind::City).map(|n| ctx.visit_cost(**n)).sum();
                    let towns: u32 = rest.iter().filter(|n| n.kind == NodeKind::Town).map(|n| ctx.visit_cost(**n)).sum();
                    if cities > 1 || towns > 1 || rest.iter().any(|n| n.kind == NodeKind::Offboard) {
                        return None;
                    }
                } else {
                    let cost: u32 = rest.iter().map(|n| ctx.visit_cost(**n)).sum();
                    if cost > d {
                        return None;
                    }
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
                            && self.hex_idx.get(fam.english_channel_hex).map_or(false, |&h| {
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
                let france =
                    stops.iter().find(|n| Family::is(fam.france_hex, &self.hexes[n.hex].id) && n.kind == NodeKind::Offboard);
                for (n, t) in stops.iter().zip(&tokened) {
                    if n.kind != NodeKind::City || !t {
                        continue;
                    }
                    if Family::is(fam.english_channel_hex, &self.hexes[n.hex].id) {
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

// ===========================================================================
// Router v2 (2026-10-17): per-kind branch-and-bound walks.
//
// The legacy walk ran ONE depth-first enumeration bounded by the widest
// train (a pullman or E-train meant 32 stops) and priced every route for
// every kind under a 3M-step budget; dense late-game maps ran out of
// budget before the best pullman runs were reached. v2:
//
// * walks each distinct train kind on its own, with its own cost cap
//   (numeric: every stop but France costs 1; pullman: towns and France are
//   free; local: 1 city + 1 town);
// * prunes a partial route when its value plus an UPPER BOUND on what the
//   rest of the route can still add (a walk-relaxation table over the
//   chain graph, towns folded into their town components for pullman,
//   capped by the top-h stop revenues) cannot reach the kind's threshold;
// * thresholds come from the combination: a scouting pass keeps each
//   kind's SCOUT_K best routes, their best combination is a lower bound LB
//   on the optimum, and a route for train t can only be in a combination
//   at least as good as LB if its value ≥ LB − Σ(best single route of
//   every other train). The full pass keeps every route above that (up to
//   TOP_K, the best first), so the combination search below is exact
//   unless a kind holds more than TOP_K routes above its threshold or a
//   walk exhausts WALK_STEP_BUDGET (both counted in `stats`);
// * children are tried revenue-first and starts token cities first, so the
//   dynamic (list-full) thresholds rise early.
//
// Routes are deduplicated by (oriented stops, track) and recorded from
// either end. Numeric and pullman routes are priced incrementally on the
// walk; local and E routes by `g1822_price_route` (test builds cross-check
// the incremental prices against it).
// ===========================================================================

/// Steps one kind's walk may take (a safety net: the bounds keep real
/// walks far below it).
const WALK_STEP_BUDGET: u64 = 4_000_000;
/// Routes kept per kind in the scouting pass.
const SCOUT_K: usize = 64;
/// 1822CA's scouting depth.
const SCOUT_K_CA: usize = 512;
/// E-trains walk only this far: their routes come from the token-to-token
/// tours (`g1822_e_tours`); the walk adds the short ones.
const E_WALK_STEP_BUDGET: u64 = 300_000;
/// 1822CA's E walks (its E routes cross the continent).
const E_WALK_STEP_BUDGET_CA: u64 = 3_000_000;

/// `G1822_ROUTER_DEBUG=1`: per-call phase timings on stderr.
fn router_debug() -> bool {
    static D: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *D.get_or_init(|| std::env::var("G1822_ROUTER_DEBUG").map_or(false, |v| v == "1"))
}

/// Router counters (process-wide; debug and the router check read them).
pub mod stats {
    use std::sync::atomic::AtomicU64;
    pub static CALLS: AtomicU64 = AtomicU64::new(0);
    pub static WALK_STEPS: AtomicU64 = AtomicU64::new(0);
    /// Walks that hit WALK_STEP_BUDGET (the result may be short).
    pub static EXHAUSTED: AtomicU64 = AtomicU64::new(0);
    /// Full passes whose TOP_K list overflowed above the threshold.
    pub static OVERFLOW: AtomicU64 = AtomicU64::new(0);
}

/// 1822PNW's revenue extras for one corporation's routing call
/// (G1822PNW#revenue_for): +10 per forest hex the route's paths cross and
/// +30 for the Ski Haus hex (not E-trains), −10 per portage tile (PNW1 /
/// PNW2) unless the owner holds P16; per run, the biggest mill bonus (P15
/// hex as a stop: 10, 30 from phase 5; E-trains double only through their
/// own token there) and, with P14, the biggest route's forest revenue
/// again (lumber baron).
struct Pnw {
    forest: Vec<bool>,
    ski: Vec<bool>,
    portage: Vec<bool>,
    p16: bool,
    lumber: bool,
    mill_hex: Option<usize>,
    mill_amount: i32,
    mill_tok: bool,
}

impl Pnw {
    /// Weight of one forest hex in an upper bound (forest + lumber baron).
    fn forest_w(&self) -> i32 {
        if self.lumber {
            20
        } else {
            10
        }
    }

    /// The most mill bonus a route of `kind` can earn.
    fn mill_max(&self, kind: Kind) -> i32 {
        match (self.mill_hex, kind) {
            (None, _) => 0,
            (Some(_), Kind::E) => {
                if self.mill_tok {
                    2 * self.mill_amount
                } else {
                    0
                }
            }
            _ => self.mill_amount,
        }
    }

    /// (revenue change, mill bonus, lumber-baron bonus) from the counts.
    fn value(&self, kind: Kind, forest_n: i32, ski_n: i32, port_n: i32, mill_n: i32) -> (i32, i32, i32) {
        let e = kind == Kind::E;
        let forest = if e { 0 } else { 10 * forest_n };
        let ski = if !e && ski_n > 0 { 30 } else { 0 };
        let port = if self.p16 { 0 } else { 10 * port_n };
        let mill = if mill_n > 0 { self.mill_max(kind) } else { 0 };
        let lumber = if self.lumber { forest } else { 0 };
        (forest + ski - port, mill, lumber)
    }

    /// The same from scratch: the route's stops and the hexes its paths
    /// cross (Route#all_hexes).
    fn extras(&self, kind: Kind, stops: &[Node], hexes: &HashSet<usize>) -> (i32, i32, i32) {
        let cnt = |v: &Vec<bool>| hexes.iter().filter(|&&h| v[h]).count() as i32;
        let mill_n = self.mill_hex.map_or(0, |m| stops.iter().filter(|n| n.hex == m).count() as i32);
        self.value(kind, cnt(&self.forest), cnt(&self.ski), cnt(&self.portage), mill_n)
    }
}

fn walked_hexes(segs: &[Seg]) -> HashSet<usize> {
    segs.iter().flat_map(|s| s.paths.iter().map(|p| p.0)).collect()
}

impl BaseGame {
    fn g1822_pnw_router_info(&self, c: &str) -> Option<Pnw> {
        if !self.g1822_is_pnw() {
            return None;
        }
        let companies = self.companies_of(&crate::entities::EntityId::corporation(c));
        let has = |id: &str| companies.iter().any(|x| x == id);
        let n = self.hexes.len();
        let assigned = |key: &str| -> Vec<bool> { (0..n).map(|h| self.g1822_pnw_assigned(&self.hexes[h].id, key)).collect() };
        let portage = (0..n)
            .map(|h| matches!(self.hexes[h].tile.name.split('-').next(), Some("PNW1" | "PNW2")))
            .collect();
        let mill_hex = (0..n).find(|&h| self.g1822_pnw_assigned(&self.hexes[h].id, "P15"));
        let mill_amount = if super::operating::ruby_to_i(&self.phase.name) < 5 { 10 } else { 30 };
        // mill_hex.tile.cities[0].tokened_by?(owner) (the coal token has
        // no corporation)
        let mill_tok = mill_hex.map_or(false, |h| {
            self.hexes[h]
                .tile
                .cities
                .first()
                .map_or(false, |city| city.tokens.iter().flatten().any(|t| t.corporation_id == c && t.token_type != "coal"))
        });
        Some(Pnw {
            forest: assigned("forest"),
            ski: assigned("P17"),
            portage,
            p16: has("P16"),
            lumber: has("P14"),
            mill_hex,
            mill_amount,
            mill_tok,
        })
    }
}

/// One chain of the dense graph.
struct RSeg {
    to: usize,
    track: Vec<u32>,
    tfirst: bool,
    tlast: bool,
    seg: Seg,
    /// 1822PNW: the hexes its paths cross, and their forest / ski weight
    /// in an upper bound (0 elsewhere).
    hexes: Vec<usize>,
    ext_w: i32,
}

/// The router's dense node graph for one corporation.
struct RG {
    nodes: Vec<Node>,
    rev: Vec<i32>,
    town: Vec<bool>,
    france: Vec<bool>,
    /// Ruby `visit_cost` (France 0; 1822CA's Q19 / Y27 99; else 1).
    vcost: Vec<u32>,
    /// 1822CA: grain-elevator stops (free for a grain-attached train).
    elev: Vec<bool>,
    ca: Option<CaR>,
    pass: Vec<bool>,
    tok: Vec<bool>,
    tok_nd: Vec<bool>,
    london: Vec<bool>,
    channel: Vec<bool>,
    out: Vec<Vec<usize>>,
    segs: Vec<RSeg>,
    ntrack: usize,
    /// Per track id: how often ONE route may use it (1822PNW: its lane
    /// width; 1 elsewhere).
    tcap: Vec<u8>,
    /// Walk starts (the corporation's connected nodes), tokens first.
    starts: Vec<usize>,
    home: Option<usize>,
    dest: Option<usize>,
    major: bool,
    pnw: Option<Pnw>,
    /// Upper bound of the forest / ski extras any one route can collect.
    ext_cap: i32,
}

impl RG {
    fn build(g: &BaseGame, ctx: &Ctx, c: &str, connected: &HashSet<Node>) -> RG {
        let mut nodes: Vec<Node> = connected.iter().copied().collect();
        nodes.sort();
        let mut id: HashMap<Node, usize> = nodes.iter().enumerate().map(|(i, n)| (*n, i)).collect();
        let mut raw: Vec<Vec<Seg>> = Vec::new();
        let mut i = 0;
        while i < nodes.len() {
            let segs: Vec<Seg> = ctx.segs_from(nodes[i]).into_iter().filter(|s| !s.terminal_inner).collect();
            for s in &segs {
                if !id.contains_key(&s.to) {
                    id.insert(s.to, nodes.len());
                    nodes.push(s.to);
                }
            }
            raw.push(segs);
            i += 1;
        }
        let n = nodes.len();
        let rev: Vec<i32> = nodes.iter().map(|&x| ctx.stop_revenue(x)).collect();
        let hex_id = |x: &Node| g.hexes[x.hex].id.as_str();
        let fam = g.fam();
        let mut track_id: HashMap<(usize, u8), u32> = HashMap::new();
        let mut segs: Vec<RSeg> = Vec::new();
        let mut out: Vec<Vec<usize>> = vec![Vec::new(); n];
        let pnw = g.g1822_pnw_router_info(c);
        let mut all_hexes: HashSet<usize> = HashSet::new();
        for (u, list) in raw.into_iter().enumerate() {
            for s in list {
                let (hexes, ext_w) = match &pnw {
                    Some(p) => {
                        let mut hs: Vec<usize> = s.paths.iter().map(|q| q.0).collect();
                        hs.sort_unstable();
                        hs.dedup();
                        let w = hs.iter().filter(|&&h| p.forest[h]).count() as i32 * p.forest_w()
                            + if hs.iter().any(|&h| p.ski[h]) { 30 } else { 0 };
                        all_hexes.extend(hs.iter().copied());
                        (hs, w)
                    }
                    None => (Vec::new(), 0),
                };
                let track = s
                    .track
                    .iter()
                    .map(|k| {
                        let l = track_id.len() as u32;
                        *track_id.entry(*k).or_insert(l)
                    })
                    .collect();
                out[u].push(segs.len());
                segs.push(RSeg {
                    to: id[&s.to],
                    track,
                    tfirst: s.terminal_first,
                    tlast: s.terminal_last,
                    seg: s,
                    hexes,
                    ext_w,
                });
            }
        }
        let ext_cap = pnw.as_ref().map_or(0, |p| {
            all_hexes.iter().filter(|&&h| p.forest[h]).count() as i32 * p.forest_w()
                + if all_hexes.iter().any(|&h| p.ski[h]) { 30 } else { 0 }
        });
        // revenue-first children
        for o in &mut out {
            o.sort_by_key(|&si| (std::cmp::Reverse(rev[segs[si].to]), si));
        }
        // Route#revenue's token stop. 1822PNW (city_tokened_by? + its
        // check_other): a normal / destination token, in a city without the
        // coal token.
        let tok: Vec<bool> = nodes
            .iter()
            .map(|&x| {
                if pnw.is_some() {
                    ctx.tokened_normal_or_dest(x)
                        && !g.hexes[x.hex].tile.cities[x.index].tokens.iter().flatten().any(|t| t.token_type == "coal")
                } else {
                    ctx.tokened(x)
                }
            })
            .collect();
        let mut starts: Vec<usize> = (0..n).filter(|&i| connected.contains(&nodes[i])).collect();
        starts.sort_by_key(|&i| (!tok[i], std::cmp::Reverse(rev[i]), i));
        let home = g.g1822_home_node(c).and_then(|x| id.get(&x).copied());
        let dest = g.g1822_destination_node(c).and_then(|x| id.get(&x).copied());
        RG {
            rev,
            town: nodes.iter().map(|x| x.kind == NodeKind::Town).collect(),
            france: nodes.iter().map(|x| Family::is(fam.france_hex, hex_id(x))).collect(),
            vcost: nodes.iter().map(|&x| ctx.visit_cost(x)).collect(),
            elev: nodes
                .iter()
                .map(|x| g.g1822_is_ca() && super::ca::GRAIN_ELEVATORS.contains(&hex_id(x)))
                .collect(),
            ca: g.g1822_ca_router_info(c),
            pass: nodes.iter().map(|&x| !ctx.blocks(x)).collect(),
            tok,
            tok_nd: nodes.iter().map(|&x| ctx.tokened_normal_or_dest(x)).collect(),
            london: nodes.iter().map(|x| ctx.is_group(x.hex) && x.kind == NodeKind::City).collect(),
            channel: nodes
                .iter()
                .map(|x| {
                    !fam.english_channel_hex.is_empty()
                        && !fam.france_hex.is_empty()
                        && (hex_id(x) == fam.english_channel_hex || hex_id(x) == fam.france_hex)
                })
                .collect(),
            out,
            segs,
            ntrack: track_id.len(),
            tcap: {
                let mut v = vec![1u8; track_id.len()];
                if g.g1822_is_pnw() {
                    for (&(h, e), &id) in &track_id {
                        v[id as usize] = ctx.edge_cap(h, e);
                    }
                }
                v
            },
            starts,
            home,
            dest,
            major: g.is_major(c),
            nodes,
            pnw,
            ext_cap,
        }
    }

    /// Visit cost of node `v` for a train whose `free` stops cost nothing
    /// (a pullman's towns, a grain train's elevators).
    fn cost(&self, v: usize, free: &[bool]) -> u32 {
        if self.france[v] || free[v] {
            0
        } else {
            self.vcost[v]
        }
    }

    /// The stops a kind visits for free.
    fn free_mask(&self, kind: Kind) -> Vec<bool> {
        match kind {
            Kind::Pullman(_) | Kind::CaPullman(_) => self.town.clone(),
            Kind::CaGrain { .. } => self.elev.clone(),
            _ => vec![false; self.nodes.len()],
        }
    }

    /// ub[h][u]: an upper bound on the revenue a route standing at `u`
    /// (allowed to leave it) can still add with `h` cost left — the best
    /// WALK (nodes may repeat) of the chain graph, towns of a pullman train
    /// folded into their town components (each component's whole revenue,
    /// then any exit), capped by the h best costly stops plus every free
    /// stop.
    fn bound_table(&self, d: u32, free: &[bool]) -> Vec<Vec<i32>> {
        let n = self.nodes.len();
        let d = d as usize;
        let pullman = free.iter().any(|&f| f);
        // town components (pullman)
        let mut comp: Vec<usize> = (0..n).collect();
        fn find(c: &mut [usize], x: usize) -> usize {
            let mut r = x;
            while c[r] != r {
                r = c[r];
            }
            let mut y = x;
            while c[y] != r {
                let nx = c[y];
                c[y] = r;
                y = nx;
            }
            r
        }
        if pullman {
            for u in 0..n {
                if !free[u] {
                    continue;
                }
                for &si in &self.out[u] {
                    let v = self.segs[si].to;
                    if free[v] {
                        let (a, b) = (find(&mut comp, u), find(&mut comp, v));
                        if a != b {
                            comp[a] = b;
                        }
                    }
                }
            }
        }
        let root: Vec<usize> = (0..n).map(|u| find(&mut comp, u)).collect();
        let mut sum_c = vec![0i32; n];
        if pullman {
            for u in 0..n {
                if free[u] {
                    sum_c[root[u]] += self.rev[u].max(0);
                    // 1822PNW: the component's inner chains' extras
                    for &si in &self.out[u] {
                        if free[self.segs[si].to] {
                            sum_c[root[u]] += self.segs[si].ext_w;
                        }
                    }
                }
            }
        }
        // global caps
        let mut costly: Vec<i32> = (0..n).filter(|&v| self.cost(v, free) == 1).map(|v| self.rev[v]).collect();
        costly.sort_unstable_by(|a, b| b.cmp(a));
        let free_rev: i32 = (0..n).filter(|&v| self.cost(v, free) == 0).map(|v| self.rev[v].max(0)).sum();
        let mut cap = vec![free_rev + self.ext_cap; d + 1];
        for h in 1..=d {
            cap[h] = cap[h - 1] + costly.get(h - 1).copied().unwrap_or(0).max(0);
        }
        let neg = i32::MIN / 4;
        let mut ub: Vec<Vec<i32>> = Vec::with_capacity(d + 1);
        for h in 0..=d {
            // a chain into costly (or free offboard) `v`
            let g = |ub: &Vec<Vec<i32>>, s: &RSeg| -> i32 {
                let v = s.to;
                let c = self.cost(v, free) as usize;
                if c > h {
                    return neg;
                }
                // France (cost 0) is an offboard: never passed through
                let cont = if c == 1 && self.pass[v] && !s.tlast { ub[h - 1][v] } else { 0 };
                self.rev[v].max(0) + s.ext_w + cont
            };
            // per town component: its best exit
            let mut m_c = vec![0i32; n];
            if pullman {
                for u in 0..n {
                    if !free[u] {
                        continue;
                    }
                    let r = root[u];
                    for &si in &self.out[u] {
                        let s = &self.segs[si];
                        if !free[s.to] {
                            m_c[r] = m_c[r].max(g(&ub, s));
                        }
                    }
                }
            }
            let mut row = vec![0i32; n];
            for u in 0..n {
                if free[u] {
                    row[u] = (sum_c[root[u]] + m_c[root[u]]).min(cap[h]);
                    continue;
                }
                let mut best = 0;
                for &si in &self.out[u] {
                    let s = &self.segs[si];
                    let val = if free[s.to] {
                        s.ext_w + sum_c[root[s.to]] + m_c[root[s.to]]
                    } else {
                        g(&ub, s)
                    };
                    best = best.max(val);
                }
                row[u] = best.min(cap[h]);
            }
            ub.push(row);
        }
        ub
    }
}

/// A kind's route list: (value, revenue, bonus, route index), best first.
type KindList = Vec<(i32, i32, i32, usize)>;

/// One kind's walk.
struct KWalk<'r> {
    g: &'r BaseGame,
    ctx: &'r Ctx<'r>,
    rg: &'r RG,
    kind: Kind,
    /// Cost cap (numeric / pullman) and stop cap.
    d: u32,
    max_stops: usize,
    ub: Vec<Vec<i32>>,
    /// E-trains: the whole map's paying value (an upper bound).
    e_total: i32,
    thr: i32,
    cap: usize,
    list: KindList,
    routes: Vec<Walked>,
    seen: HashSet<Vec<u32>>,
    steps: u64,
    limit: u64,
    exhausted: bool,
    // walk state
    stops: Vec<usize>,
    segs: Vec<usize>,
    visited: Vec<bool>,
    used: Vec<u8>,
    rev: i32,
    cost: u32,
    n_tok: u32,
    bonus: i32,
    /// The stops this kind visits for free (RG::free_mask).
    free: Vec<bool>,
    /// 1822CA: grain-elevator / city / town stops on the route, and the
    /// extras' caps (grain / port, sawmill).
    ca_elev: u32,
    ca_city: u32,
    ca_town: u32,
    ca_caps: (i32, i32),
    // 1822PNW extras: per-hex use counts and the route's counts
    hcnt: Vec<u16>,
    forest_n: i32,
    ski_n: i32,
    port_n: i32,
    mill_n: i32,
}

impl<'r> KWalk<'r> {
    fn new(g: &'r BaseGame, ctx: &'r Ctx<'r>, rg: &'r RG, kind: Kind, thr: i32, cap: usize) -> KWalk<'r> {
        let (d, max_stops) = match kind {
            Kind::Local { .. } => (1, 2),
            Kind::Numeric(d) => (d, d as usize + 2),
            Kind::Pullman(d) => (d, MAX_STOPS_UNBOUNDED),
            Kind::E => (0, MAX_STOPS_UNBOUNDED),
            Kind::CaPullman(d) => (d, (3 * d) as usize),
            Kind::CaGrain { d, .. } => (d, (d + CA_GRAIN_STOPS) as usize),
        };
        let free = rg.free_mask(kind);
        let ub = match kind {
            Kind::Numeric(d) | Kind::Pullman(d) | Kind::CaPullman(d) | Kind::CaGrain { d, .. } => rg.bound_table(d, &free),
            _ => Vec::new(),
        };
        let n = rg.nodes.len();
        let ca_caps =
            rg.ca.as_ref().map_or((0, 0), |p| p.caps(kind, rg.elev.iter().filter(|&&e| e).count(), max_stops));
        let e_total = if kind == Kind::E {
            let france = (0..n).filter(|&v| rg.france[v]).map(|v| rg.rev[v]).max().unwrap_or(0);
            (0..n)
                .filter(|&v| rg.tok_nd[v] && rg.nodes[v].kind == NodeKind::City)
                .map(|v| 2 * if rg.channel[v] { france } else { rg.rev[v] }.max(0))
                .sum()
        } else {
            0
        };
        KWalk {
            g,
            ctx,
            rg,
            kind,
            d,
            max_stops,
            ub,
            e_total,
            thr: thr.max(1),
            cap,
            list: Vec::new(),
            routes: Vec::new(),
            seen: HashSet::new(),
            steps: 0,
            limit: if kind == Kind::E {
                if g.g1822_is_ca() {
                    E_WALK_STEP_BUDGET_CA
                } else {
                    E_WALK_STEP_BUDGET
                }
            } else {
                WALK_STEP_BUDGET
            },
            exhausted: false,
            stops: Vec::new(),
            segs: Vec::new(),
            visited: vec![false; n],
            used: vec![0; rg.ntrack],
            rev: 0,
            cost: 0,
            n_tok: 0,
            bonus: 0,
            free,
            ca_elev: 0,
            ca_city: 0,
            ca_town: 0,
            ca_caps,
            hcnt: if rg.pnw.is_some() { vec![0; g.hexes.len()] } else { Vec::new() },
            forest_n: 0,
            ski_n: 0,
            port_n: 0,
            mill_n: 0,
        }
    }

    /// 1822PNW: (revenue change, mill, lumber baron) of the current route.
    fn pnw_now(&self) -> (i32, i32, i32) {
        match &self.rg.pnw {
            Some(p) => p.value(self.kind, self.forest_n, self.ski_n, self.port_n, self.mill_n),
            None => (0, 0, 0),
        }
    }

    /// Mill bonus the route could still add.
    fn mill_pot(&self) -> i32 {
        match &self.rg.pnw {
            Some(p) if self.mill_n == 0 => p.mill_max(self.kind),
            _ => 0,
        }
    }

    fn pnw_track(&mut self, v: usize, seg: Option<usize>, add: bool) {
        let rg = self.rg;
        let Some(p) = &rg.pnw else { return };
        let d = if add { 1 } else { -1 };
        if p.mill_hex == Some(rg.nodes[v].hex) {
            self.mill_n += d;
        }
        if let Some(si) = seg {
            for &h in &rg.segs[si].hexes {
                let first = if add {
                    self.hcnt[h] += 1;
                    self.hcnt[h] == 1
                } else {
                    self.hcnt[h] -= 1;
                    self.hcnt[h] == 0
                };
                if first {
                    self.forest_n += d * p.forest[h] as i32;
                    self.ski_n += d * p.ski[h] as i32;
                    self.port_n += d * p.portage[h] as i32;
                }
            }
        }
    }


    /// The value a route must reach to be kept.
    fn eff_thr(&self) -> i32 {
        if self.list.len() >= self.cap {
            self.thr.max(self.list.last().map_or(0, |l| l.0) + 1)
        } else {
            self.thr
        }
    }

    fn mult(&self) -> i32 {
        if self.kind == Kind::E {
            2
        } else {
            1
        }
    }

    /// The destination bonus this route could still pick up.
    fn bonus_pot(&self) -> i32 {
        let (Some(h), Some(d)) = (self.rg.home, self.rg.dest) else { return 0 };
        if !self.rg.major || (self.visited[h] && self.visited[d]) {
            return 0;
        }
        self.rg.rev[d].max(0) * self.mult()
    }

    fn upper(&self, u: usize) -> i32 {
        let (x, m, l) = self.pnw_now();
        // (1822CA: a start node costing more than the train's distance —
        // Q19 / Y27's visit_cost 99 — leads nowhere)
        if self.cost > self.d && !matches!(self.kind, Kind::E | Kind::Local { .. }) {
            return i32::MIN / 2;
        }
        match self.kind {
            Kind::Numeric(_) | Kind::Pullman(_) => {
                self.rev
                    + x
                    + self.bonus
                    + m
                    + l
                    + self.ub[(self.d - self.cost) as usize][u]
                    + self.bonus_pot()
                    + self.mill_pot()
                    + self.ca_caps.0
                    + self.ca_caps.1
            }
            Kind::CaPullman(_) | Kind::CaGrain { .. } => {
                self.rev
                    + self.bonus
                    + self.ub[(self.d - self.cost) as usize][u]
                    + self.bonus_pot()
                    + self.ca_caps.0
                    + self.ca_caps.1
            }
            // (an E-train's only extra besides the mill is a penalty)
            Kind::E => self.e_total + self.bonus + self.bonus_pot() + m + self.mill_pot() + self.ca_caps.1,
            Kind::Local { .. } => i32::MAX,
        }
    }

    fn run(&mut self) {
        let starts = self.rg.starts.clone();
        for s in starts {
            if self.exhausted {
                break;
            }
            self.push(s, None);
            self.extend();
            self.pop(None);
        }
    }

    fn push(&mut self, v: usize, seg: Option<usize>) {
        let rg = self.rg;
        self.stops.push(v);
        self.visited[v] = true;
        self.rev += rg.rev[v];
        self.cost += rg.cost(v, &self.free);
        if rg.tok[v] {
            self.n_tok += 1;
        }
        if rg.elev[v] {
            self.ca_elev += 1;
        } else if rg.town[v] {
            self.ca_town += 1;
        } else {
            self.ca_city += 1;
        }
        if let Some(si) = seg {
            self.segs.push(si);
            for &k in &rg.segs[si].track {
                self.used[k as usize] += 1;
            }
        }
        self.pnw_track(v, seg, true);
        self.bonus = self.cur_bonus();
    }

    fn pop(&mut self, seg: Option<usize>) {
        let rg = self.rg;
        let v = self.stops.pop().unwrap();
        self.visited[v] = false;
        self.rev -= rg.rev[v];
        self.cost -= rg.cost(v, &self.free);
        if rg.tok[v] {
            self.n_tok -= 1;
        }
        if rg.elev[v] {
            self.ca_elev -= 1;
        } else if rg.town[v] {
            self.ca_town -= 1;
        } else {
            self.ca_city -= 1;
        }
        if let Some(si) = seg {
            self.segs.pop();
            for &k in &rg.segs[si].track {
                self.used[k as usize] -= 1;
            }
        }
        self.pnw_track(v, seg, false);
        self.bonus = self.cur_bonus();
    }

    fn cur_bonus(&self) -> i32 {
        match (self.rg.home, self.rg.dest) {
            (Some(h), Some(d)) if self.rg.major && self.visited[h] && self.visited[d] => self.rg.rev[d] * self.mult(),
            _ => 0,
        }
    }

    fn extend(&mut self) {
        self.steps += 1;
        if self.steps > self.limit {
            self.exhausted = true;
            return;
        }
        let rg = self.rg;
        let u = *self.stops.last().unwrap();
        if self.stops.len() > 1 && !rg.pass[u] {
            return;
        }
        if let Some(&si) = self.segs.last() {
            if rg.segs[si].tlast {
                return;
            }
        }
        if self.stops.len() >= self.max_stops {
            return;
        }
        if self.upper(u) < self.eff_thr() {
            return;
        }
        for &si in &rg.out[u] {
            if self.exhausted {
                return;
            }
            let s = &rg.segs[si];
            let v = s.to;
            if self.visited[v] || (s.tfirst && !self.segs.is_empty()) {
                continue;
            }
            // one city per route group (the group's hex)
            if rg.london[v] && self.stops.iter().any(|&x| rg.london[x] && rg.nodes[x].hex == rg.nodes[v].hex) {
                continue;
            }
            match self.kind {
                Kind::Numeric(_) | Kind::Pullman(_) | Kind::CaPullman(_) => {
                    if self.cost + rg.cost(v, &self.free) > self.d {
                        continue;
                    }
                }
                Kind::CaGrain { local: false, .. } => {
                    if self.cost + rg.cost(v, &self.free) > self.d || (rg.elev[v] && self.ca_elev >= CA_GRAIN_STOPS) {
                        continue;
                    }
                }
                Kind::CaGrain { local: true, .. } => {
                    let ok = if rg.elev[v] {
                        self.ca_elev < CA_GRAIN_STOPS
                    } else if rg.nodes[v].kind == NodeKind::Offboard {
                        false
                    } else if rg.town[v] {
                        self.ca_town == 0 && rg.vcost[v] <= 1
                    } else {
                        self.ca_city == 0 && rg.vcost[v] <= 1
                    };
                    if !ok {
                        continue;
                    }
                }
                // 1 city + 1 town
                Kind::Local { .. } => {
                    if rg.town[u] == rg.town[v] {
                        continue;
                    }
                }
                Kind::E => {}
            }
            if s.track.iter().any(|&k| self.used[k as usize] >= rg.tcap[k as usize]) {
                continue;
            }
            self.push(v, Some(si));
            self.record();
            self.extend();
            self.pop(Some(si));
        }
    }

    fn record(&mut self) {
        let rg = self.rg;
        if self.n_tok == 0 {
            return;
        }
        let thr = self.eff_thr();
        let (px, pm, pl) = self.pnw_now();
        let priced = match self.kind {
            Kind::Numeric(_) | Kind::Pullman(_) => {
                if self.rev + px + self.bonus + pm + pl + self.ca_caps.0 + self.ca_caps.1 < thr {
                    return;
                }
                // check_distance's channel rules
                let ch = self.stops.iter().filter(|&&x| rg.channel[x]).count();
                if ch == 1 || (ch == 2 && self.stops.len() == 2) {
                    return;
                }
                let stops: Vec<Node> = self.stops.iter().map(|&i| rg.nodes[i]).collect();
                if !self.g.g1822_hexes_ok(&stops) {
                    return;
                }
                let p = (self.rev, self.bonus);
                if cfg!(test) {
                    let q = self.g.g1822_price_route(
                        self.ctx,
                        &stops,
                        false,
                        self.kind,
                        rg.major,
                        self.home_node(),
                        self.dest_node(),
                    );
                    assert_eq!(q, Some(p), "router v2 incremental price for {:?}", self.kind);
                }
                p
            }
            _ => {
                if self.kind == Kind::E {
                    // both ends tokened, unless the Channel exception
                    let (a, b) = (self.stops[0], *self.stops.last().unwrap());
                    if !(rg.tok_nd[a] && rg.tok_nd[b]) && !self.stops.iter().any(|&x| rg.channel[x]) {
                        return;
                    }
                    if self.e_total + self.bonus + pm < thr {
                        return;
                    }
                }
                let stops: Vec<Node> = self.stops.iter().map(|&i| rg.nodes[i]).collect();
                if !self.g.g1822_hexes_ok(&stops) {
                    return;
                }
                let Some(p) = self.g.g1822_price_route(
                    self.ctx,
                    &stops,
                    false,
                    self.kind,
                    rg.major,
                    self.home_node(),
                    self.dest_node(),
                ) else {
                    return;
                };
                p
            }
        };
        if let Some(p) = &self.rg.pnw {
            if cfg!(test) {
                let stops: Vec<Node> = self.stops.iter().map(|&i| rg.nodes[i]).collect();
                let segs: Vec<Seg> = self.segs.iter().map(|&si| rg.segs[si].seg.clone()).collect();
                assert_eq!(p.extras(self.kind, &stops, &walked_hexes(&segs)), (px, pm, pl), "router v2 PNW extras");
            }
        }
        // 1822CA: grain / port bonus (revenue) and the sawmill (a side bonus)
        let (cx, cm) = match &self.rg.ca {
            Some(p) => {
                let stops: Vec<Node> = self.stops.iter().map(|&i| rg.nodes[i]).collect();
                (p.grain(self.kind, &stops), p.sawmill(self.ctx, self.kind, &stops, priced.1 > 0))
            }
            None => (0, 0),
        };
        let pm = pm + cm;
        let priced = (priced.0 + px + cx, priced.1);
        let value = priced.0 + priced.1 + pm + pl;
        if value < thr || value <= 0 {
            return;
        }
        // dedupe: oriented stops + sorted track
        let mut key: Vec<u32> = if self.stops[0] <= *self.stops.last().unwrap() {
            self.stops.iter().map(|&x| x as u32).collect()
        } else {
            self.stops.iter().rev().map(|&x| x as u32).collect()
        };
        key.push(u32::MAX);
        let mut tr: Vec<u32> = self.segs.iter().flat_map(|&si| rg.segs[si].track.iter().copied()).collect();
        tr.sort_unstable();
        key.extend(tr);
        if !self.seen.insert(key) {
            return;
        }
        let w = Walked {
            stops: self.stops.iter().map(|&i| rg.nodes[i]).collect(),
            segs: self.segs.iter().map(|&si| rg.segs[si].seg.clone()).collect(),
            mill: pm,
            lumber: pl,
        };
        self.insert(w, priced.0, priced.1);
    }

    fn insert(&mut self, w: Walked, rev: i32, bonus: i32) {
        let value = rev + bonus + w.mill + w.lumber;
        let pos = self.list.partition_point(|e| e.0 >= value);
        if self.list.len() >= self.cap && pos >= self.cap {
            return;
        }
        self.routes.push(w);
        self.list.insert(pos, (value, rev, bonus, self.routes.len() - 1));
        if self.list.len() > self.cap {
            self.list.pop();
        }
    }

    fn home_node(&self) -> Option<Node> {
        self.rg.home.map(|i| self.rg.nodes[i])
    }
    fn dest_node(&self) -> Option<Node> {
        self.rg.dest.map(|i| self.rg.nodes[i])
    }

    /// A valid upper bound on this kind's best single route.
    fn root_upper(&self) -> i32 {
        let found = self.list.first().map_or(0, |l| l.0);
        match self.kind {
            Kind::Numeric(_) | Kind::Pullman(_) => {
                let mut best = 0;
                for &s in &self.rg.starts {
                    let c = self.rg.cost(s, &self.free);
                    if c > self.d {
                        continue;
                    }
                    best = best.max(self.rg.rev[s].max(0) + self.ub[(self.d - c) as usize][s]);
                }
                found.max(best + self.bonus_pot() + self.mill_pot() + self.ca_caps.0 + self.ca_caps.1)
            }
            Kind::CaPullman(_) | Kind::CaGrain { .. } => {
                let mut best = 0;
                for &s in &self.rg.starts {
                    let c = self.rg.cost(s, &self.free);
                    if c > self.d {
                        continue;
                    }
                    best = best.max(self.rg.rev[s].max(0) + self.ub[(self.d - c) as usize][s]);
                }
                found.max(best + self.bonus_pot() + self.ca_caps.0 + self.ca_caps.1)
            }
            Kind::E => found.max(self.e_total + self.bonus_pot() + self.mill_pot() + self.ca_caps.1),
            Kind::Local { .. } => found,
        }
    }
}

impl BaseGame {
    /// Router v2 (see the section comment above).
    pub(crate) fn g1822_best_routes_v2(&self, c: &str) -> (Vec<RouteData>, i32) {
        use std::sync::atomic::Ordering::Relaxed;
        stats::CALLS.fetch_add(1, Relaxed);
        let trains = self.g1822_route_trains(c);
        if trains.is_empty() {
            return (Vec::new(), 0);
        }
        let kinds: Vec<Kind> = trains.iter().map(|t| self.g1822_train_kind_of(t)).collect();
        let mut distinct: Vec<Kind> = Vec::new();
        for k in &kinds {
            if !distinct.contains(k) {
                distinct.push(*k);
            }
        }
        let ctx = Ctx::new(self, c);
        let graph = self.g1822_graph(c);
        let rg = RG::build(self, &ctx, c, &graph.connected_nodes);
        let tokens: Vec<usize> = (0..rg.nodes.len()).filter(|&i| rg.tok[i]).collect();

        // one kind's walk (+ the E tours / single-city local runs):
        // (list, routes, list cut or walk exhausted, best-single bound)
        let walk = |k: Kind, thr: i32, cap: usize| -> (KindList, Vec<Walked>, bool, i32) {
            let mut w = KWalk::new(self, &ctx, &rg, k, thr, cap);
            w.run();
            if k == Kind::E {
                for mut t in self.g1822_e_tours(&ctx, &graph.connected_nodes) {
                    let th = w.eff_thr();
                    if let Some((r, b)) =
                        self.g1822_price_route(&ctx, &t.stops, false, k, rg.major, w.home_node(), w.dest_node())
                    {
                        let (x, m, l) = rg.pnw.as_ref().map_or((0, 0, 0), |p| p.extras(k, &t.stops, &walked_hexes(&t.segs)));
                        let (cx, cm) = rg
                            .ca
                            .as_ref()
                            .map_or((0, 0), |p| (p.grain(k, &t.stops), p.sawmill(&ctx, k, &t.stops, b > 0)));
                        let (x, m) = (x + cx, m + cm);
                        t.mill = m;
                        t.lumber = l;
                        if r + x + b + m + l >= th && r + x + b + m + l > 0 {
                            w.insert(t, r + x, b);
                        }
                    }
                }
            }
            if matches!(k, Kind::Local { .. }) {
                for &t in &tokens {
                    // Route#connection_data rebuilds a ["local", hex] run on
                    // the hex's FIRST node the corporation tokened: in a
                    // multi-city hex (1822MX: the NdeM's and Mexico City's
                    // several homes) only that city has a single-city run
                    if self.g1822_is_mx() || self.g1822_is_ca() {
                        let n = rg.nodes[t];
                        // (city_tokened_by?: 1822CA's P10 token sits in an
                        // extra slot)
                        let first = (0..self.hexes[n.hex].tile.cities.len())
                            .find(|&ci| self.hexes[n.hex].tile.cities[ci].tokened_by(c));
                        if first != Some(n.index) {
                            continue;
                        }
                    }
                    let st = vec![rg.nodes[t]];
                    if let Some((r, b)) =
                        self.g1822_price_route(&ctx, &st, true, k, rg.major, w.home_node(), w.dest_node())
                    {
                        // (no paths: only the mill can apply)
                        let (x, m, l) = rg.pnw.as_ref().map_or((0, 0, 0), |p| p.extras(k, &st, &HashSet::new()));
                        let (cx, cm) =
                            rg.ca.as_ref().map_or((0, 0), |p| (p.grain(k, &st), p.sawmill(&ctx, k, &st, b > 0)));
                        let (x, m) = (x + cx, m + cm);
                        if r + x + b + m + l >= w.eff_thr() && r + x + b + m + l > 0 {
                            w.insert(Walked { stops: st, segs: Vec::new(), mill: m, lumber: l }, r + x, b);
                        }
                    }
                }
            }
            stats::WALK_STEPS.fetch_add(w.steps, Relaxed);
            if router_debug() {
                eprintln!(
                    "  walk {:?} thr {} cap {}: steps {} exhausted {} kept {} best {:?} nodes {} segs {}",
                    k,
                    thr,
                    cap,
                    w.steps,
                    w.exhausted,
                    w.list.len(),
                    w.list.first().map(|l| l.0),
                    rg.nodes.len(),
                    rg.segs.len()
                );
            }
            if w.exhausted {
                stats::EXHAUSTED.fetch_add(1, Relaxed);
            }
            let up = w.root_upper();
            let up = if w.exhausted { up } else { w.list.first().map_or(0, |l| l.0) };
            let full = w.exhausted || w.list.len() >= cap;
            (std::mem::take(&mut w.list), std::mem::take(&mut w.routes), full, up)
        };

        // -- scouting pass: each kind's SCOUT_K best ------------------------
        let mut lists: Vec<(KindList, Vec<Walked>)> = Vec::new();
        let mut full: Vec<bool> = Vec::new();
        let mut best_single: Vec<i32> = Vec::new();
        for &k in &distinct {
            // (1822CA: a deeper scouting list — its long attached trains'
            // best routes mostly share track, so 64 give a weak bound)
            let (l, r, f, up) = walk(k, 1, if self.g1822_is_ca() { SCOUT_K_CA } else { SCOUT_K });
            lists.push((l, r));
            full.push(f);
            best_single.push(up);
        }
        let ki = |k: &Kind| distinct.iter().position(|d| d == k).unwrap();
        let t0 = std::time::Instant::now();
        // 1822MX revenue_for: a 3/2P train earns half (after bonuses,
        // rounded to 10)
        let half: Vec<bool> = trains.iter().map(|t| t.name.starts_with("3/2P")).collect();
        let (mut pick, mut lb) = self.g1822_combine(&ctx, &kinds, &distinct, &lists, &half);
        if router_debug() {
            eprintln!("  scout combine: lb {} in {:?}; best single {:?}", lb, t0.elapsed(), best_single);
        }

        // -- full pass for kinds whose scouting list was cut ----------------
        // (1822CA: repeated while the combination improves and a list was
        // cut — a better LB raises the thresholds; the other titles take
        // one pass)
        let ca = self.g1822_is_ca();
        let mut cut = full.clone();
        let mut passes = 0;
        loop {
            if !(cut.iter().any(|&f| f) && kinds.len() > 1) {
                break;
            }
            let before = if ca { lists.clone() } else { Vec::new() };
            let total_up: i32 = kinds.iter().map(|k| best_single[ki(k)]).sum();
            let mut changed = false;
            let mut cut_now = vec![false; distinct.len()];
            for (di, &k) in distinct.iter().enumerate() {
                if !cut[di] {
                    continue;
                }
                // LB − Σ the other trains' best (one train of this kind
                // taken out: the loosest of this kind's trains)
                let thr = lb - (total_up - best_single[di]);
                let (l, r, f, _) = walk(k, thr.max(1), if ca { TOP_K_CA } else { TOP_K });
                if f {
                    stats::OVERFLOW.fetch_add(1, Relaxed);
                }
                cut_now[di] = f;
                lists[di] = (l, r);
                changed = true;
            }
            let mut improved = false;
            if changed {
                let t0 = std::time::Instant::now();
                let (p2, t2) = self.g1822_combine(&ctx, &kinds, &distinct, &lists, &half);
                if router_debug() {
                    eprintln!("  full combine: {} in {:?}", t2, t0.elapsed());
                }
                // (1822CA: keep the previous run when a pass does worse —
                // a budget-limited E walk can lose the tour the scouting
                // pass found; the other titles keep their output)
                if t2 >= lb || !ca {
                    improved = t2 > lb;
                    pick = p2;
                    lb = t2;
                } else {
                    lists = before;
                }
            }
            passes += 1;
            if !ca || passes >= 3 || !improved {
                break;
            }
            cut = cut_now;
        }

        // -- output -----------------------------------------------------
        let mut out = Vec::new();
        // destination, mill, lumber baron
        let mut best3: [Option<(usize, i32)>; 3] = [None; 3];
        for (ti, p) in pick.iter().enumerate() {
            let Some(pi) = p else { continue };
            let (list, routes) = &lists[ki(&kinds[ti])];
            let (_, rev, bonus, ri) = list[*pi];
            let oi = out.len();
            out.push(self.g1822_route_data(&ctx, &routes[ri], &trains[ti].id, rev));
            // each bonus rides on the route that earns the biggest (ties:
            // the last one, as Ruby's stable sort_by + reverse picks)
            for (ch, v) in [bonus, routes[ri].mill, routes[ri].lumber].into_iter().enumerate() {
                if v > 0 && best3[ch].map_or(true, |(_, bb)| v >= bb) {
                    best3[ch] = Some((oi, v));
                }
            }
        }
        for (oi, b) in best3.into_iter().flatten() {
            if let Some(rv) = out[oi].revenue.as_mut() {
                *rv += b;
            }
        }
        // 3/2P: halved after its bonus
        let mut oi = 0;
        for (ti, p) in pick.iter().enumerate() {
            if p.is_none() {
                continue;
            }
            if half[ti] {
                if let Some(rv) = out[oi].revenue.as_mut() {
                    *rv = half_revenue(*rv);
                }
            }
            oi += 1;
        }
        (out, lb)
    }

    /// The best combination over per-kind route lists: (pick per train
    /// into its kind's list, total) — branch-and-bound over one route (or
    /// none) per train, on dense track counters (see [`Combo`]).
    fn g1822_combine(
        &self,
        ctx_lanes: &Ctx,
        kinds: &[Kind],
        distinct: &[Kind],
        lists: &[(KindList, Vec<Walked>)],
        half: &[bool],
    ) -> (Vec<Option<usize>>, i32) {
        // 1822CA: search the E-trains last — they share no track with the
        // other trains, so at the last level the first E route that fits is
        // the best and the bound stops the scan (the picks map back).
        if self.g1822_is_ca() && kinds.iter().any(|k| *k == Kind::E) && kinds.iter().any(|k| *k != Kind::E) {
            let mut perm: Vec<usize> = (0..kinds.len()).collect();
            perm.sort_by_key(|&i| kinds[i] == Kind::E);
            if perm.iter().enumerate().any(|(a, &b)| a != b) {
                let kp: Vec<Kind> = perm.iter().map(|&i| kinds[i]).collect();
                let hp: Vec<bool> = perm.iter().map(|&i| half[i]).collect();
                let (pp, total) = self.g1822_combine_ordered(ctx_lanes, &kp, distinct, lists, &hp);
                let mut pick = vec![None; kinds.len()];
                for (a, &b) in perm.iter().enumerate() {
                    pick[b] = pp[a];
                }
                return (pick, total);
            }
        }
        self.g1822_combine_ordered(ctx_lanes, kinds, distinct, lists, half)
    }

    fn g1822_combine_ordered(
        &self,
        ctx_lanes: &Ctx,
        kinds: &[Kind],
        distinct: &[Kind],
        lists: &[(KindList, Vec<Walked>)],
        half: &[bool],
    ) -> (Vec<Option<usize>>, i32) {
        let merthyr_hex = self.hex_idx.get(self.fam().merthyr_tydfil_pontypool_hex).copied();
        let mut track_id: HashMap<(usize, u8), u32> = HashMap::new();
        let mut caps: Vec<u8> = Vec::new();
        let mut city_id: HashMap<Node, u32> = HashMap::new();
        // per distinct kind, per route: (track ids, local city ids, merthyr)
        let mut dense: Vec<Vec<CRoute>> = Vec::new();
        for (_, routes) in lists {
            let mut v = Vec::with_capacity(routes.len());
            for r in routes {
                let mut track = Vec::new();
                let mut m = (false, false);
                for s in &r.segs {
                    for k in &s.track {
                        let id = *track_id.entry(*k).or_insert_with(|| {
                            caps.push(ctx_lanes.edge_cap(k.0, k.1));
                            caps.len() as u32 - 1
                        });
                        track.push(id);
                        if Some(k.0) == merthyr_hex {
                            m.0 |= k.1 == 1;
                            m.1 |= k.1 == 2;
                        }
                    }
                }
                let cities = r
                    .stops
                    .iter()
                    .filter(|n| n.kind == NodeKind::City)
                    .map(|n| {
                        let l = city_id.len() as u32;
                        *city_id.entry(*n).or_insert(l)
                    })
                    .collect();
                let multi = {
                    let mut t2 = track.clone();
                    t2.sort_unstable();
                    t2.windows(2).any(|w| w[0] == w[1])
                };
                v.push(CRoute { track, cities, merthyr: m, mill: r.mill, lumber: r.lumber, multi });
            }
            dense.push(v);
        }
        let kind_of: Vec<usize> = kinds.iter().map(|k| distinct.iter().position(|d| d == k).unwrap()).collect();
        let nt = kinds.len();
        let max_rev: Vec<i32> = (0..nt)
            .map(|t| {
                let m = lists[kind_of[t]].0.iter().map(|e| e.1).max().unwrap_or(0);
                if half[t] {
                    half_revenue(m)
                } else {
                    m
                }
            })
            .collect();
        let max_bonus: Vec<i32> = (0..nt).map(|t| lists[kind_of[t]].0.iter().map(|e| e.2).max().unwrap_or(0)).collect();
        let max_side = |f: fn(&CRoute) -> i32| -> Vec<i32> {
            (0..nt).map(|t| dense[kind_of[t]].iter().map(f).max().unwrap_or(0)).collect()
        };
        let max_mill = max_side(|r| r.mill);
        let max_lumber = max_side(|r| r.lumber);
        let mut rest_rev = vec![0i32; nt + 1];
        let mut rest_bonus = vec![[0i32; 3]; nt + 1];
        for t in (0..nt).rev() {
            rest_rev[t] = rest_rev[t + 1] + max_rev[t];
            let nx = rest_bonus[t + 1];
            rest_bonus[t] = [nx[0].max(max_bonus[t]), nx[1].max(max_mill[t]), nx[2].max(max_lumber[t])];
        }
        let mut c = Combo {
            lists,
            dense: &dense,
            kind_of: &kind_of,
            e: kinds.iter().map(|k| *k == Kind::E).collect(),
            local: kinds.iter().map(|k| matches!(k, Kind::Local { .. })).collect(),
            caps: &caps,
            used_normal: vec![0; caps.len()],
            used_e: vec![0; caps.len()],
            local_cities: vec![false; city_id.len()],
            merthyr: [false, false],
            rest_rev: &rest_rev,
            rest_bonus: &rest_bonus,
            pick: vec![None; nt],
            best: 0,
            best_pick: vec![None; nt],
            half: half.to_vec(),
            steps: 0,
            budget: if self.g1822_is_ca() { COMBO_BUDGET_CA } else { u64::MAX },
        };
        c.search(0, 0, [0; 3], (0, 0));
        (c.best_pick, c.best)
    }
}

/// A route as the combination search sees it.
struct CRoute {
    track: Vec<u32>,
    cities: Vec<u32>,
    merthyr: (bool, bool),
    mill: i32,
    lumber: i32,
    /// Uses some track id twice (two lanes of one hexside).
    multi: bool,
}

/// Σ of per-channel maxima (destination, mill, lumber baron).
fn bmax(a: [i32; 3], b: [i32; 3]) -> [i32; 3] {
    [a[0].max(b[0]), a[1].max(b[1]), a[2].max(b[2])]
}

fn bsum(a: [i32; 3]) -> i32 {
    a[0] + a[1] + a[2]
}

/// Branch-and-bound over one route (or none) per train: track used once
/// per train type (E-trains and the rest separately, lanes counted), local
/// trains use each city once, only one of the two Merthyr Tydfil–Pontypool
/// tracks; the total is Σ revenue + the biggest destination bonus (1822PNW:
/// + the biggest mill and lumber-baron bonuses).
struct Combo<'s> {
    lists: &'s [(KindList, Vec<Walked>)],
    dense: &'s [Vec<CRoute>],
    kind_of: &'s [usize],
    e: Vec<bool>,
    local: Vec<bool>,
    caps: &'s [u8],
    used_normal: Vec<u8>,
    used_e: Vec<u8>,
    local_cities: Vec<bool>,
    merthyr: [bool; 2],
    rest_rev: &'s [i32],
    rest_bonus: &'s [[i32; 3]],
    pick: Vec<Option<usize>>,
    best: i32,
    best_pick: Vec<Option<usize>>,
    /// 1822MX's 3/2P: the train earns half its run (bonus included).
    half: Vec<bool>,
    /// Search nodes visited and the cap (1822CA: its big lists can make
    /// the exact search slow; the best found so far stands).
    steps: u64,
    budget: u64,
}

/// 1822MX `(revenue / 2).round(-1)` (Integer#round: half up).
pub(crate) fn half_revenue(r: i32) -> i32 {
    let h = r.div_euclid(2);
    (h + 5).div_euclid(10) * 10
}

impl Combo<'_> {
    fn fits(&self, t: usize, r: &CRoute) -> bool {
        let used = if self.e[t] { &self.used_e } else { &self.used_normal };
        if r.track.iter().any(|&k| used[k as usize] >= self.caps[k as usize]) {
            return false;
        }
        if r.multi && r.track.iter().any(|&k| {
            used[k as usize] + r.track.iter().filter(|&&x| x == k).count() as u8 > self.caps[k as usize]
        }) {
            return false;
        }
        if self.local[t] && r.cities.iter().any(|&c| self.local_cities[c as usize]) {
            return false;
        }
        (!r.merthyr.0 && !self.merthyr[0]) || (!r.merthyr.1 && !self.merthyr[1])
    }

    fn apply(&mut self, t: usize, r: &CRoute, add: bool) {
        let used = if self.e[t] { &mut self.used_e } else { &mut self.used_normal };
        for &k in &r.track {
            if add {
                used[k as usize] += 1;
            } else {
                used[k as usize] -= 1;
            }
        }
        if self.local[t] {
            for &c in &r.cities {
                self.local_cities[c as usize] = add;
            }
        }
    }

    /// `bonus`: per-channel maxima (bounds); `hold`: the destination
    /// bonus's holder as (its raw bonus, what it adds) — the raw bonus for
    /// a normal route, half(run + bonus) − half(run) for a 3/2P.
    fn search(&mut self, t: usize, revenue: i32, bonus: [i32; 3], hold: (i32, i32)) {
        self.steps += 1;
        if self.steps > self.budget {
            return;
        }
        let value = revenue + hold.1 + bonus[1] + bonus[2];
        if value > self.best {
            self.best = value;
            self.best_pick.clone_from(&self.pick);
        }
        if t == self.kind_of.len() {
            return;
        }
        if revenue + self.rest_rev[t] + bsum(bmax(bonus, self.rest_bonus[t])) <= self.best {
            return;
        }
        let ki = self.kind_of[t];
        let lists = self.lists;
        let dense = self.dense;
        let (rr, rb) = (self.rest_rev[t + 1], self.rest_bonus[t + 1]);
        let open = bsum(bmax(bonus, rb));
        for (pi, &(value, rev, b, ri)) in lists[ki].0.iter().enumerate() {
            // monotone in list order (sorted by value = revenue + every
            // bonus): max(x, y, z) ≤ y + max(x, z) per channel
            if revenue + value + rr + open <= self.best {
                break;
            }
            self.steps += 1;
            if self.steps > self.budget {
                return;
            }
            let r = &dense[ki][ri];
            let nb = bmax(bonus, [b, r.mill, r.lumber]);
            if revenue + rev + rr + bsum(bmax(nb, rb)) <= self.best {
                continue;
            }
            if !self.fits(t, r) {
                continue;
            }
            let saved = self.merthyr;
            self.merthyr[0] |= r.merthyr.0;
            self.merthyr[1] |= r.merthyr.1;
            self.apply(t, r, true);
            self.pick[t] = Some(pi);
            let (gain, nh) = if self.half[t] {
                let h = half_revenue(rev);
                let nh = if b > 0 && b >= hold.0 { (b, half_revenue(rev + b) - h) } else { hold };
                (h, nh)
            } else {
                let nh = if b > 0 && b >= hold.0 { (b, b) } else { hold };
                (rev, nh)
            };
            self.search(t + 1, revenue + gain, nb, nh);
            self.pick[t] = None;
            self.apply(t, r, false);
            self.merthyr = saved;
        }
        // this train runs nothing
        self.search(t + 1, revenue, bonus, hold);
    }
}
