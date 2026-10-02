//! Native route search: the revenue-maximising set of legal routes for a
//! corporation's trains — the Rust counterpart of Python's `AutoRouter` (a
//! port of Ruby's auto_router.rb), used when self-play's native decode builds
//! a RunRoutes action.
//!
//! Two stages:
//! 1. **Enumerate** every legal route with a depth-first walk over tile paths
//!    from each revenue node the corporation reaches (Python `Node.walk` /
//!    `Path.walk`). Each route is walked from both of its ends; only the walk
//!    from the lower-keyed end is kept.
//! 2. **Combine**: branch-and-bound over one route (or none) per train,
//!    maximising total revenue with no track shared between routes — the
//!    maximum Python's `js_evaluate_combos` finds by brute force.
//!
//! Legal means passing the checks of Python's `Route.revenue()` (Ruby
//! route.rb), which the walk enforces as it goes:
//! - at least two stops, one a city the corporation has tokened
//!   (`check_route_token`);
//! - no stop visited twice (`check_cycles`) — every node a route passes is a
//!   stop, including the one it started from;
//! - no track used twice, within a route or across one run's routes
//!   (`check_overlap`, keyed per `(hex, edge)` path end; a route only ever
//!   uses a path end by crossing that hexside, so the walk tracks hexsides,
//!   plus node-to-node paths inside a tile);
//! - at most one stop per group (`groups:Canada`: 1830's two Canada
//!   offboards);
//! - offboards and cities blocked by other corporations' tokens can only end
//!   a route (`check_connected` → `Node.blocks`), and terminal paths can only
//!   be a route's first or last path (`check_terminals`);
//! - no more stops than the train's distance (`check_distance`).

use std::collections::HashMap;

use crate::graph::{Hex, Tile};
use crate::map::{NodeId, NodeType};
use crate::tiles::PathEndpoint;

// ---------------------------------------------------------------------------
// Public result
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct RouteCandidate {
    /// Stops in route order.
    pub nodes: Vec<NodeId>,
    pub revenue: i32,
    /// Hex chains between consecutive stops, one hex per tile path (Python
    /// `Route.chain_id`), in route order — what a logged run_routes records.
    pub connections: Vec<Vec<String>>,
    /// Index into the `trains` slice passed to [`calculate_corp_routes`] of
    /// the train that runs this route.
    pub train_index: usize,
}

// ---------------------------------------------------------------------------
// Node keys
// ---------------------------------------------------------------------------

/// A revenue node packed as `hex << 8 | kind << 6 | index`, so keys order by
/// hex, then kind, then index.
type NodeKey = u32;

const KIND_CITY: u32 = 0;
const KIND_TOWN: u32 = 1;
const KIND_OFFBOARD: u32 = 2;

fn endpoint_key(hex: usize, ep: &PathEndpoint) -> Option<NodeKey> {
    let (kind, index) = match ep {
        PathEndpoint::City(i) => (KIND_CITY, *i),
        PathEndpoint::Town(i) => (KIND_TOWN, *i),
        PathEndpoint::Offboard(i) => (KIND_OFFBOARD, *i),
        PathEndpoint::Edge(_) | PathEndpoint::Junction => return None,
    };
    if index >= 64 {
        return None;
    }
    Some((hex as u32) << 8 | kind << 6 | index as u32)
}

fn key_hex(key: NodeKey) -> usize {
    (key >> 8) as usize
}

fn key_endpoint(key: NodeKey) -> PathEndpoint {
    let index = (key & 63) as usize;
    match (key >> 6) & 3 {
        KIND_CITY => PathEndpoint::City(index),
        KIND_TOWN => PathEndpoint::Town(index),
        _ => PathEndpoint::Offboard(index),
    }
}

fn node_id_key(node: &NodeId, hex_idx: &HashMap<String, usize>) -> Option<NodeKey> {
    let hex = *hex_idx.get(&node.hex_id)?;
    let ep = match node.node_type {
        NodeType::City => PathEndpoint::City(node.index),
        NodeType::Town => PathEndpoint::Town(node.index),
        NodeType::Offboard => PathEndpoint::Offboard(node.index),
    };
    endpoint_key(hex, &ep)
}

fn key_node_id(key: NodeKey, hexes: &[Hex]) -> NodeId {
    let (node_type, index) = match key_endpoint(key) {
        PathEndpoint::City(i) => (NodeType::City, i),
        PathEndpoint::Town(i) => (NodeType::Town, i),
        PathEndpoint::Offboard(i) => (NodeType::Offboard, i),
        _ => unreachable!("node keys only encode revenue nodes"),
    };
    NodeId {
        hex_id: hexes[key_hex(key)].id.clone(),
        node_type,
        index,
    }
}

// ---------------------------------------------------------------------------
// Revenue nodes
// ---------------------------------------------------------------------------

fn node_exists(tile: &Tile, ep: &PathEndpoint) -> bool {
    match ep {
        PathEndpoint::City(i) => *i < tile.cities.len(),
        PathEndpoint::Town(i) => *i < tile.towns.len(),
        PathEndpoint::Offboard(i) => *i < tile.offboards.len(),
        _ => false,
    }
}

fn node_revenue(tile: &Tile, ep: &PathEndpoint, phase_tiles: &[String]) -> i32 {
    match ep {
        PathEndpoint::City(i) => tile.cities.get(*i).map_or(0, |c| c.revenue),
        PathEndpoint::Town(i) => tile.towns.get(*i).map_or(0, |t| t.revenue),
        PathEndpoint::Offboard(i) => tile
            .offboards
            .get(*i)
            .map_or(0, |o| o.phase_revenue(phase_tiles)),
        _ => 0,
    }
}

fn node_groups<'t>(tile: &'t Tile, ep: &PathEndpoint) -> &'t [String] {
    match ep {
        PathEndpoint::Offboard(i) => tile.offboards.get(*i).map_or(&[], |o| &o.groups),
        _ => &[],
    }
}

/// Ruby `Node#blocks?`: an offboard always ends a route; a city does when
/// every slot holds another corporation's (non-neutral) token.
fn node_blocks(tile: &Tile, ep: &PathEndpoint, corp_sym: &str) -> bool {
    match ep {
        PathEndpoint::Offboard(_) => true,
        PathEndpoint::City(i) => tile.cities.get(*i).is_some_and(|city| {
            city.tokens.iter().all(|t| {
                t.as_ref().is_some_and(|tok| {
                    tok.token_type != "neutral" && tok.corporation_id != corp_sym
                })
            })
        }),
        _ => false,
    }
}

// ---------------------------------------------------------------------------
// Stage 1: route enumeration
// ---------------------------------------------------------------------------

const NBR_UNKNOWN: u16 = u16::MAX;
const NBR_NONE: u16 = u16::MAX - 1;
const NO_TRACK: u16 = u16::MAX;

/// A legal route, stored as ranges into the walk's arenas.
struct Found {
    revenue: i32,
    /// Stops counted toward distance: all of them / towns free (D trains).
    cost: u32,
    cost_d: u32,
    /// Into `Walk::found_stops` and `Walk::found_stop_pos`.
    stops: std::ops::Range<usize>,
    /// Into `Walk::found_trail`; `found_stop_pos` indexes this slice.
    trail: std::ops::Range<usize>,
    /// Into `Walk::found_track`.
    track: std::ops::Range<usize>,
}

struct Walk<'a> {
    hexes: &'a [Hex],
    hex_idx: &'a HashMap<String, usize>,
    hex_adjacency: &'a HashMap<String, HashMap<u8, String>>,
    phase_tiles: &'a [String],
    corp_sym: &'a str,
    /// Most stops any train can count, every stop counted (`None`: no
    /// such train) / towns free (D trains).
    max_cost: Option<u32>,
    max_cost_d: Option<u32>,
    tokens: Vec<NodeKey>,
    /// Sorted: every node a walk starts from.
    starts: Vec<NodeKey>,

    /// Neighbor hex by (hex, edge), resolved on first use.
    neighbors: Vec<[u16; 6]>,
    /// Track id of each `(hex, edge)` path end — both sides of a hexside
    /// share one — assigned on first use.
    hexside_track: Vec<u16>,
    /// Track ids of node-to-node paths inside a tile, by (hex, path index).
    node_path_track: HashMap<(usize, usize), u16>,
    /// Whether the route being walked uses each track id.
    in_use: Vec<bool>,

    // The route being walked.
    stops: Vec<NodeKey>,
    /// Index into `trail` of each stop's hex.
    stop_pos: Vec<u16>,
    /// The route's hexes, one per tile path (consecutive paths in one hex
    /// share an entry).
    trail: Vec<u16>,
    track: Vec<u16>,
    junction_hexes: Vec<usize>,
    groups: Vec<&'a str>,
    revenue: i32,
    cost: u32,
    cost_d: u32,
    n_paths: u32,

    found: Vec<Found>,
    found_stops: Vec<NodeKey>,
    found_stop_pos: Vec<u16>,
    found_trail: Vec<u16>,
    found_track: Vec<u16>,
}

impl<'a> Walk<'a> {
    fn neighbor(&mut self, hex: usize, edge: u8) -> Option<usize> {
        let slot = self.neighbors[hex][edge as usize];
        if slot == NBR_UNKNOWN {
            let nb = self
                .hex_adjacency
                .get(&self.hexes[hex].id)
                .and_then(|m| m.get(&edge))
                .and_then(|id| self.hex_idx.get(id))
                .copied();
            self.neighbors[hex][edge as usize] = nb.map_or(NBR_NONE, |n| n as u16);
            return nb;
        }
        (slot != NBR_NONE).then_some(slot as usize)
    }

    fn new_track(&mut self) -> u16 {
        let id = self.in_use.len() as u16;
        self.in_use.push(false);
        id
    }

    fn hexside_track(&mut self, hex: usize, edge: u8, nb: usize) -> usize {
        let k = hex * 6 + edge as usize;
        if self.hexside_track[k] == NO_TRACK {
            let id = self.new_track();
            self.hexside_track[k] = id;
            self.hexside_track[nb * 6 + ((edge + 3) % 6) as usize] = id;
        }
        self.hexside_track[k] as usize
    }

    fn node_path_track(&mut self, hex: usize, path: usize) -> usize {
        if let Some(&id) = self.node_path_track.get(&(hex, path)) {
            return id as usize;
        }
        let id = self.new_track();
        self.node_path_track.insert((hex, path), id);
        id as usize
    }

    fn fits(&self, cost: u32, cost_d: u32) -> bool {
        self.max_cost.is_some_and(|m| cost <= m) || self.max_cost_d.is_some_and(|m| cost_d <= m)
    }

    /// Walk every route that starts at `start`.
    fn walk_routes_from(&mut self, start: NodeKey) {
        let hexes = self.hexes;
        let hex = key_hex(start);
        let ep = key_endpoint(start);
        let tile = &hexes[hex].tile;
        let cost_d = if matches!(ep, PathEndpoint::Town(_)) {
            0
        } else {
            1
        };
        if !node_exists(tile, &ep) || !self.fits(1, cost_d) {
            return;
        }
        self.stops.push(start);
        self.stop_pos.push(0);
        self.trail.push(hex as u16);
        self.groups
            .extend(node_groups(tile, &ep).iter().map(String::as_str));
        self.revenue = node_revenue(tile, &ep, self.phase_tiles);
        self.cost = 1;
        self.cost_d = cost_d;
        // A route may start at a blocked city or an offboard; it just
        // can't pass through one.
        self.walk_from(hex, &ep);
        self.stops.clear();
        self.stop_pos.clear();
        self.trail.clear();
        self.groups.clear();
    }

    /// Extend the route from the stop at `node` along each of its paths.
    fn walk_from(&mut self, hex: usize, node: &PathEndpoint) {
        let hexes = self.hexes;
        for (pi, p) in hexes[hex].tile.paths.iter().enumerate() {
            let other = if p.a == *node {
                &p.b
            } else if p.b == *node {
                &p.a
            } else {
                continue;
            };
            // A terminal path can be a route's first path, or its last.
            let last = p.terminal && self.n_paths > 0;
            self.n_paths += 1;
            self.follow(hex, pi, other, last);
            self.n_paths -= 1;
        }
    }

    /// Continue along path `path` of `hex` to its endpoint `other`. `last`:
    /// the path is terminal, so the route must end at its node.
    fn follow(&mut self, hex: usize, path: usize, other: &PathEndpoint, last: bool) {
        match other {
            PathEndpoint::Edge(e) => {
                if !last {
                    self.cross(hex, *e);
                }
            }
            PathEndpoint::Junction => {
                if !last {
                    self.through_junction(hex, path);
                }
            }
            node => {
                // A node-to-node path inside the tile: no hexside, so it is
                // its own piece of track.
                let id = self.node_path_track(hex, path);
                if self.in_use[id] {
                    return;
                }
                self.in_use[id] = true;
                self.track.push(id as u16);
                self.arrive(hex, node, last);
                self.track.pop();
                self.in_use[id] = false;
            }
        }
    }

    /// Cross from `hex` over its `edge` and follow each path of the
    /// neighbor that meets that hexside.
    fn cross(&mut self, hex: usize, edge: u8) {
        let Some(nb) = self.neighbor(hex, edge) else {
            return;
        };
        let id = self.hexside_track(hex, edge, nb);
        if self.in_use[id] {
            return;
        }
        self.in_use[id] = true;
        self.track.push(id as u16);
        self.trail.push(nb as u16);
        let enter = PathEndpoint::Edge((edge + 3) % 6);
        let hexes = self.hexes;
        for (qi, q) in hexes[nb].tile.paths.iter().enumerate() {
            let other = if q.a == enter {
                &q.b
            } else if q.b == enter {
                &q.a
            } else {
                continue;
            };
            self.n_paths += 1;
            self.follow(nb, qi, other, q.terminal);
            self.n_paths -= 1;
        }
        self.trail.pop();
        self.track.pop();
        self.in_use[id] = false;
    }

    /// Pass through `hex`'s junction (entered on `path`) onto each of its
    /// other paths — at most once per route (Python `Path.walk`).
    fn through_junction(&mut self, hex: usize, path: usize) {
        if self.junction_hexes.contains(&hex) {
            return;
        }
        self.junction_hexes.push(hex);
        let hexes = self.hexes;
        for (ji, j) in hexes[hex].tile.paths.iter().enumerate() {
            if ji == path {
                continue;
            }
            let other = if j.a == PathEndpoint::Junction {
                &j.b
            } else if j.b == PathEndpoint::Junction {
                &j.a
            } else {
                continue;
            };
            if *other == PathEndpoint::Junction {
                continue;
            }
            self.n_paths += 1;
            self.follow(hex, ji, other, j.terminal);
            self.n_paths -= 1;
        }
        self.junction_hexes.pop();
    }

    /// Stop at `node` on `hex`: record the route so far, then keep going
    /// unless the node (or a terminal path, `last`) ends the route.
    fn arrive(&mut self, hex: usize, node: &PathEndpoint, last: bool) {
        let Some(key) = endpoint_key(hex, node) else {
            return;
        };
        if self.stops.contains(&key) {
            return;
        }
        let hexes = self.hexes;
        let tile = &hexes[hex].tile;
        if !node_exists(tile, node) {
            return;
        }
        let cost = self.cost + 1;
        let cost_d = self.cost_d + u32::from(!matches!(node, PathEndpoint::Town(_)));
        if !self.fits(cost, cost_d) {
            return;
        }
        let groups = node_groups(tile, node);
        if groups.iter().any(|g| self.groups.contains(&g.as_str())) {
            return;
        }
        let (saved_cost, saved_cost_d) = (self.cost, self.cost_d);
        let revenue = node_revenue(tile, node, self.phase_tiles);
        self.stops.push(key);
        self.stop_pos.push((self.trail.len() - 1) as u16);
        self.groups.extend(groups.iter().map(String::as_str));
        self.revenue += revenue;
        self.cost = cost;
        self.cost_d = cost_d;

        self.record();
        if !last && !node_blocks(tile, node, self.corp_sym) {
            self.walk_from(hex, node);
        }

        self.cost = saved_cost;
        self.cost_d = saved_cost_d;
        self.revenue -= revenue;
        self.groups.truncate(self.groups.len() - groups.len());
        self.stop_pos.pop();
        self.stops.pop();
    }

    /// Keep the route walked so far if it is a route: two or more stops,
    /// one of them tokened.
    fn record(&mut self) {
        if self.stops.len() < 2 || !self.stops.iter().any(|s| self.tokens.contains(s)) {
            return;
        }
        // The walk from the route's other end finds it too; keep one.
        let (first, last) = (self.stops[0], self.stops[self.stops.len() - 1]);
        if last < first && self.starts.binary_search(&last).is_ok() {
            return;
        }
        let stops = self.found_stops.len()..self.found_stops.len() + self.stops.len();
        self.found_stops.extend_from_slice(&self.stops);
        self.found_stop_pos.extend_from_slice(&self.stop_pos);
        let trail = self.found_trail.len()..self.found_trail.len() + self.trail.len();
        self.found_trail.extend_from_slice(&self.trail);
        let track = self.found_track.len()..self.found_track.len() + self.track.len();
        self.found_track.extend_from_slice(&self.track);
        self.found.push(Found {
            revenue: self.revenue,
            cost: self.cost,
            cost_d: self.cost_d,
            stops,
            trail,
            track,
        });
    }

    fn route(&self, found: &Found, train_index: usize) -> RouteCandidate {
        let stops = &self.found_stops[found.stops.clone()];
        let pos = &self.found_stop_pos[found.stops.clone()];
        let trail = &self.found_trail[found.trail.clone()];
        RouteCandidate {
            nodes: stops.iter().map(|&k| key_node_id(k, self.hexes)).collect(),
            revenue: found.revenue,
            connections: pos
                .windows(2)
                .map(|w| {
                    trail[w[0] as usize..=w[1] as usize]
                        .iter()
                        .map(|&h| self.hexes[h as usize].id.clone())
                        .collect()
                })
                .collect(),
            train_index,
        }
    }
}

// ---------------------------------------------------------------------------
// Stage 2: best combination
// ---------------------------------------------------------------------------

/// Branch-and-bound over one route (or none) per train, trains ordered so
/// identical ones are adjacent.
struct Combo<'a> {
    revenue: &'a [i32],
    /// Track bitset of each found route, `words` u64s apiece.
    bits: &'a [u64],
    words: usize,
    /// Per ordered train: eligible routes, best revenue first.
    lists: Vec<&'a [u32]>,
    /// Per ordered train: for each track id, the bitset of the list
    /// positions whose route uses it (`list.len().div_ceil(64)` u64s each).
    uses: Vec<&'a [u64]>,
    /// Whether ordered train `t` is identical to train `t - 1`.
    same_as_prev: Vec<bool>,
    /// First ordered train after `t`'s run of identical trains.
    group_end: Vec<usize>,
    /// Track bitset of the routes chosen so far.
    used: Vec<u64>,
    /// Per ordered train: scratch bitset of its list positions that share
    /// track with the routes chosen so far.
    blocked: Vec<Vec<u64>>,
    chosen: Vec<Option<u32>>,
    best: i32,
    best_chosen: Vec<Option<u32>>,
}

impl Combo<'_> {
    fn toggle(&mut self, route: usize) {
        let b = &self.bits[route * self.words..(route + 1) * self.words];
        for (u, x) in self.used.iter_mut().zip(b) {
            *u ^= x;
        }
    }

    /// The most trains `t..` could add, ignoring overlap, when train `t`
    /// takes routes from list position `from` on: it and the identical
    /// trains after it take distinct routes, the rest their best.
    fn upper_bound(&self, t: usize, from: usize) -> i32 {
        let n = self.lists.len();
        let mut bound = 0;
        let (mut t, mut from) = (t, from);
        while t < n {
            let end = self.group_end[t];
            let list = self.lists[t];
            bound += list[from.min(list.len())..]
                .iter()
                .take(end - t)
                .map(|&r| self.revenue[r as usize])
                .sum::<i32>();
            (t, from) = (end, 0);
        }
        bound
    }

    /// Choose routes for trains `t..`, having earned `revenue` so far.
    /// `from`: identical trains take routes in list order, so this train
    /// starts at the list position after the previous train's.
    fn search(&mut self, t: usize, revenue: i32, from: usize) {
        if revenue > self.best {
            self.best = revenue;
            self.best_chosen.clone_from(&self.chosen);
        }
        let n = self.lists.len();
        if t == n || revenue + self.upper_bound(t, from) <= self.best {
            return;
        }
        let next_same = t + 1 < n && self.same_as_prev[t + 1];
        let list = self.lists[t];
        let rest_bound = if next_same {
            0
        } else {
            self.upper_bound(t + 1, 0)
        };

        // Rule out, 64 positions at a time, every route sharing track with
        // the ones chosen so far.
        let pw = list.len().div_ceil(64);
        let mut blocked = std::mem::take(&mut self.blocked[t]);
        blocked.clear();
        blocked.resize(pw, 0);
        for (w, &word) in self.used.iter().enumerate() {
            let mut word = word;
            while word != 0 {
                let id = w * 64 + word.trailing_zeros() as usize;
                word &= word - 1;
                for (b, u) in blocked
                    .iter_mut()
                    .zip(&self.uses[t][id * pw..(id + 1) * pw])
                {
                    *b |= u;
                }
            }
        }

        'scan: for w in from / 64..pw {
            let mut free = !blocked[w];
            if w == from / 64 {
                free &= !0u64 << (from % 64);
            }
            while free != 0 {
                let pos = w * 64 + free.trailing_zeros() as usize;
                free &= free - 1;
                if pos >= list.len() {
                    break 'scan;
                }
                let route = list[pos] as usize;
                let r = self.revenue[route];
                // Both terms shrink as `pos` grows, so nothing later can win.
                let rest = if next_same {
                    self.upper_bound(t + 1, pos + 1)
                } else {
                    rest_bound
                };
                if revenue + r + rest <= self.best {
                    break 'scan;
                }
                self.toggle(route);
                self.chosen[t] = Some(route as u32);
                self.search(t + 1, revenue + r, if next_same { pos + 1 } else { 0 });
                self.chosen[t] = None;
                self.toggle(route);
            }
        }
        self.blocked[t] = blocked;

        // This train runs nothing — and then neither do identical ones
        // after it (any other arrangement is a permutation of one tried).
        self.search(self.group_end[t], revenue, 0);
    }
}

/// The best set of `(train index, found route)` pairs and its revenue.
fn best_combination(walk: &Walk<'_>, trains: &[(u32, bool)]) -> (Vec<(usize, usize)>, i32) {
    let found = &walk.found;
    if found.is_empty() {
        return (Vec::new(), 0);
    }
    let n_track = walk.in_use.len();
    let words = n_track.div_ceil(64).max(1);
    let mut bits = vec![0u64; found.len() * words];
    for (i, f) in found.iter().enumerate() {
        for &id in &walk.found_track[f.track.clone()] {
            bits[i * words + id as usize / 64] |= 1u64 << (id % 64);
        }
    }
    let revenue: Vec<i32> = found.iter().map(|f| f.revenue).collect();
    let mut by_revenue: Vec<u32> = (0..found.len() as u32).collect();
    by_revenue.sort_by_key(|&i| std::cmp::Reverse(revenue[i as usize]));

    // Larger trains first; identical trains adjacent and sharing a list.
    let mut order: Vec<usize> = (0..trains.len()).collect();
    order.sort_by_key(|&i| std::cmp::Reverse((trains[i].1, trains[i].0)));
    let mut kinds: Vec<(u32, bool)> = Vec::new();
    let mut kind_lists: Vec<Vec<u32>> = Vec::new();
    let mut kind_uses: Vec<Vec<u64>> = Vec::new();
    for &i in &order {
        let (distance, is_d) = trains[i];
        if kinds.contains(&(distance, is_d)) {
            continue;
        }
        let list: Vec<u32> = by_revenue
            .iter()
            .copied()
            .filter(|&r| {
                let f = &found[r as usize];
                if is_d {
                    f.cost_d <= distance
                } else {
                    f.cost <= distance
                }
            })
            .collect();
        let pw = list.len().div_ceil(64);
        let mut uses = vec![0u64; n_track * pw];
        for (pos, &r) in list.iter().enumerate() {
            for &id in &walk.found_track[found[r as usize].track.clone()] {
                uses[id as usize * pw + pos / 64] |= 1u64 << (pos % 64);
            }
        }
        kinds.push((distance, is_d));
        kind_lists.push(list);
        kind_uses.push(uses);
    }
    let kind_of: Vec<usize> = order
        .iter()
        .map(|&i| kinds.iter().position(|&k| k == trains[i]).unwrap())
        .collect();
    let same_as_prev: Vec<bool> = (0..order.len())
        .map(|t| t > 0 && trains[order[t]] == trains[order[t - 1]])
        .collect();
    let mut group_end = vec![order.len(); order.len()];
    for t in (0..order.len().saturating_sub(1)).rev() {
        group_end[t] = if same_as_prev[t + 1] {
            group_end[t + 1]
        } else {
            t + 1
        };
    }

    let mut combo = Combo {
        revenue: &revenue,
        bits: &bits,
        words,
        lists: kind_of.iter().map(|&k| kind_lists[k].as_slice()).collect(),
        uses: kind_of.iter().map(|&k| kind_uses[k].as_slice()).collect(),
        same_as_prev,
        group_end,
        used: vec![0u64; words],
        blocked: vec![Vec::new(); order.len()],
        chosen: vec![None; order.len()],
        best: 0,
        best_chosen: vec![None; order.len()],
    };
    combo.search(0, 0, 0);

    let mut chosen: Vec<(usize, usize)> = combo
        .best_chosen
        .iter()
        .enumerate()
        .filter_map(|(t, r)| r.map(|r| (order[t], r as usize)))
        .collect();
    chosen.sort_unstable();
    (chosen, combo.best)
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// The revenue-maximising legal routes for a corporation's `trains` (each
/// `(distance, is_d)`; a D train counts towns free), and their total
/// revenue. `token_nodes` are the corporation's token cities;
/// `connected_nodes` the revenue nodes its track reaches.
#[allow(clippy::too_many_arguments)]
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
    let tokens: Vec<NodeKey> = token_nodes
        .iter()
        .filter_map(|n| node_id_key(n, hex_idx))
        .collect();
    let mut starts: Vec<NodeKey> = connected_nodes
        .iter()
        .filter_map(|n| node_id_key(n, hex_idx))
        .chain(tokens.iter().copied())
        .collect();
    starts.sort_unstable();
    starts.dedup();

    let mut walk = Walk {
        hexes,
        hex_idx,
        hex_adjacency,
        phase_tiles,
        corp_sym,
        max_cost: trains.iter().filter(|t| !t.1).map(|t| t.0).max(),
        max_cost_d: trains.iter().filter(|t| t.1).map(|t| t.0).max(),
        tokens,
        starts: starts.clone(),
        neighbors: vec![[NBR_UNKNOWN; 6]; hexes.len()],
        hexside_track: vec![NO_TRACK; hexes.len() * 6],
        node_path_track: HashMap::new(),
        in_use: Vec::new(),
        stops: Vec::new(),
        stop_pos: Vec::new(),
        trail: Vec::new(),
        track: Vec::new(),
        junction_hexes: Vec::new(),
        groups: Vec::new(),
        revenue: 0,
        cost: 0,
        cost_d: 0,
        n_paths: 0,
        found: Vec::new(),
        found_stops: Vec::new(),
        found_stop_pos: Vec::new(),
        found_trail: Vec::new(),
        found_track: Vec::new(),
    };
    for start in starts {
        walk.walk_routes_from(start);
    }

    let (chosen, revenue) = best_combination(&walk, trains);
    let routes = chosen
        .into_iter()
        .map(|(train, f)| walk.route(&walk.found[f], train))
        .collect();
    (routes, revenue)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{City, Hex, Offboard, Tile, Town};
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
        for od in &tile_def.offboards {
            let mut ob = Offboard::new(od.yellow_revenue);
            ob.brown_revenue = Some(od.brown_revenue);
            ob.groups = od.groups.clone();
            tile.offboards.push(ob);
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

    fn city(hex: &str) -> NodeId {
        NodeId {
            hex_id: hex.to_string(),
            node_type: NodeType::City,
            index: 0,
        }
    }

    /// A test map: hexes with tiles, adjacency from `(hex, edge, neighbor)`
    /// triples (the reverse direction is added).
    struct TestMap {
        hexes: Vec<Hex>,
        hex_idx: HashMap<String, usize>,
        adjacency: HashMap<String, HashMap<u8, String>>,
    }

    impl TestMap {
        fn new(tiles: Vec<(&str, Tile)>, links: &[(&str, u8, &str)]) -> Self {
            let hexes: Vec<Hex> = tiles
                .into_iter()
                .map(|(id, tile)| Hex::new(id.to_string(), tile))
                .collect();
            let hex_idx = hexes
                .iter()
                .enumerate()
                .map(|(i, h)| (h.id.clone(), i))
                .collect();
            let mut adjacency: HashMap<String, HashMap<u8, String>> = HashMap::new();
            for &(a, edge, b) in links {
                adjacency
                    .entry(a.to_string())
                    .or_default()
                    .insert(edge, b.to_string());
                adjacency
                    .entry(b.to_string())
                    .or_default()
                    .insert((edge + 3) % 6, a.to_string());
            }
            TestMap {
                hexes,
                hex_idx,
                adjacency,
            }
        }

        /// Every revenue node on the map counts as connected.
        fn routes(&self, tokens: &[NodeId], trains: &[(u32, bool)]) -> (Vec<RouteCandidate>, i32) {
            let mut connected = Vec::new();
            for h in &self.hexes {
                for i in 0..h.tile.cities.len() {
                    connected.push(NodeId {
                        hex_id: h.id.clone(),
                        node_type: NodeType::City,
                        index: i,
                    });
                }
                for i in 0..h.tile.towns.len() {
                    connected.push(NodeId {
                        hex_id: h.id.clone(),
                        node_type: NodeType::Town,
                        index: i,
                    });
                }
                for i in 0..h.tile.offboards.len() {
                    connected.push(NodeId {
                        hex_id: h.id.clone(),
                        node_type: NodeType::Offboard,
                        index: i,
                    });
                }
            }
            calculate_corp_routes(
                &self.hexes,
                &self.hex_idx,
                &self.adjacency,
                tokens,
                &connected,
                trains,
                &["yellow".to_string()],
                "PRR",
            )
        }
    }

    fn y(name: &str, dsl: &str) -> Tile {
        tile_from_dsl(name, dsl, TileColor::Yellow)
    }

    fn tokened(mut tile: Tile) -> Tile {
        place_token(&mut tile, 0, "PRR");
        tile
    }

    fn stops(route: &RouteCandidate) -> Vec<String> {
        route.nodes.iter().map(|n| n.hex_id.clone()).collect()
    }

    /// A(city 20, token) -- B(track) -- C(city 30)
    fn linear() -> TestMap {
        TestMap::new(
            vec![
                ("A", tokened(y("a", "city=revenue:20;path=a:1,b:_0"))),
                ("B", y("b", "path=a:4,b:1")),
                ("C", y("c", "city=revenue:30;path=a:4,b:_0")),
            ],
            &[("A", 1, "B"), ("B", 1, "C")],
        )
    }

    #[test]
    fn single_train_finds_best_route() {
        let (routes, revenue) = linear().routes(&[city("A")], &[(2, false)]);
        assert_eq!(revenue, 50, "A+C");
        assert_eq!(routes.len(), 1);
        assert_eq!(routes[0].connections, vec![vec!["A", "B", "C"]]);
    }

    #[test]
    fn no_trains_or_tokens_no_revenue() {
        let map = linear();
        assert_eq!(map.routes(&[city("A")], &[]).1, 0);
        assert_eq!(map.routes(&[], &[(2, false)]).1, 0);
    }

    #[test]
    fn route_needs_two_stops() {
        let (routes, revenue) = linear().routes(&[city("A")], &[(1, false)]);
        assert_eq!(revenue, 0);
        assert!(routes.is_empty());
    }

    #[test]
    fn route_needs_a_token() {
        // C--D runs on track the corp reaches, but neither stop is tokened.
        let map = TestMap::new(
            vec![
                ("A", tokened(y("a", "city=revenue:20;path=a:1,b:_0"))),
                ("C", y("c", "city=revenue:30;path=a:4,b:_0;path=a:1,b:_0")),
                ("D", y("d", "city=revenue:90;path=a:4,b:_0")),
            ],
            &[("A", 1, "C"), ("C", 1, "D")],
        );
        let (routes, revenue) = map.routes(&[city("A")], &[(2, false)]);
        assert_eq!(revenue, 50, "A-C, not the untokened C-D (120)");
        assert_eq!(stops(&routes[0]), vec!["A", "C"]);
    }

    /// Triangle A(token) -- B -- C -- A.
    fn triangle() -> TestMap {
        TestMap::new(
            vec![
                (
                    "A",
                    tokened(tile_from_dsl(
                        "a",
                        "city=revenue:20;path=a:1,b:_0;path=a:2,b:_0",
                        TileColor::Green,
                    )),
                ),
                (
                    "B",
                    tile_from_dsl(
                        "b",
                        "city=revenue:30;path=a:4,b:_0;path=a:3,b:_0",
                        TileColor::Green,
                    ),
                ),
                (
                    "C",
                    tile_from_dsl(
                        "c",
                        "city=revenue:40;path=a:5,b:_0;path=a:0,b:_0",
                        TileColor::Green,
                    ),
                ),
            ],
            &[("A", 1, "B"), ("A", 2, "C"), ("B", 3, "C")],
        )
    }

    #[test]
    fn triangle_visits_each_city_once() {
        let (routes, revenue) = triangle().routes(&[city("A")], &[(3, false)]);
        assert_eq!(revenue, 90);
        assert_eq!(routes[0].nodes.len(), 3);
        // A 4-train must not loop back to its start (A-B-C-A = 110).
        let (routes, revenue) = triangle().routes(&[city("A")], &[(4, false)]);
        assert_eq!(revenue, 90, "no stop counted twice: {:?}", routes);
    }

    #[test]
    fn triangle_two_stop_best_pair() {
        let (routes, revenue) = triangle().routes(&[city("A")], &[(2, false)]);
        assert_eq!(revenue, 60, "A-C; B-C (70) has no token");
        assert_eq!(stops(&routes[0]), vec!["A", "C"]);
    }

    #[test]
    fn route_never_revisits_a_blocked_start() {
        // B is full of another corp's tokens: a route may start or end there
        // but never visit it twice (B-A-C-B would count B twice).
        let mut b = tile_from_dsl(
            "b",
            "city=revenue:50;path=a:4,b:_0;path=a:3,b:_0",
            TileColor::Green,
        );
        place_token(&mut b, 0, "NYC");
        let map = TestMap::new(
            vec![
                (
                    "A",
                    tokened(tile_from_dsl(
                        "a",
                        "city=revenue:20;path=a:1,b:_0;path=a:2,b:_0",
                        TileColor::Green,
                    )),
                ),
                ("B", b),
                (
                    "C",
                    tile_from_dsl(
                        "c",
                        "city=revenue:40;path=a:5,b:_0;path=a:0,b:_0",
                        TileColor::Green,
                    ),
                ),
            ],
            &[("A", 1, "B"), ("A", 2, "C"), ("B", 3, "C")],
        );
        let (routes, revenue) = map.routes(&[city("A")], &[(4, false)]);
        assert_eq!(revenue, 110, "B-A-C or A-C-B, once each: {:?}", routes);
        for r in &routes {
            let mut s = stops(r);
            s.sort();
            s.dedup();
            assert_eq!(s.len(), r.nodes.len(), "repeated stop in {:?}", r);
        }
    }

    #[test]
    fn route_never_crosses_a_hexside_twice() {
        // X and Y each have two paths onto the X|Y hexside (like D20/D22
        // in 1830): S-X-Y-M, round the K loop back into Y, and over X|Y
        // again to E would pay S+M+E = 90.
        let map = TestMap::new(
            vec![
                ("S", tokened(y("s", "city=revenue:20;path=a:1,b:_0"))),
                ("X", y("x", "path=a:4,b:1;path=a:2,b:1")),
                ("Y", y("y", "path=a:4,b:0;path=a:4,b:2")),
                ("M", y("m", "city=revenue:30;path=a:3,b:_0;path=a:1,b:_0")),
                ("K", y("k", "path=a:5,b:4")),
                ("E", y("e", "city=revenue:40;path=a:5,b:_0")),
            ],
            &[
                ("S", 1, "X"),
                ("X", 1, "Y"),
                ("X", 2, "E"),
                ("Y", 0, "M"),
                ("Y", 2, "K"),
                ("M", 1, "K"),
            ],
        );
        let (routes, revenue) = map.routes(&[city("S")], &[(3, false)]);
        assert_eq!(revenue, 50, "S-M, not S-M-E over X|Y twice: {:?}", routes);
    }

    #[test]
    fn trains_never_share_a_hexside() {
        // X forks the track out of A (edge 4) to B and to C: two trains
        // can't both leave A through X, and one can't run B-X-A-X-C.
        let map = TestMap::new(
            vec![
                ("A", tokened(y("a", "city=revenue:20;path=a:1,b:_0"))),
                ("X", y("x", "path=a:4,b:1;path=a:4,b:2")),
                ("B", y("b", "city=revenue:30;path=a:4,b:_0")),
                ("C", y("c", "city=revenue:40;path=a:5,b:_0")),
            ],
            &[("A", 1, "X"), ("X", 1, "B"), ("X", 2, "C")],
        );
        let (routes, revenue) = map.routes(&[city("A")], &[(3, false)]);
        assert_eq!(revenue, 60, "A-X-C: {:?}", routes);
        let (routes, revenue) = map.routes(&[city("A")], &[(2, false), (2, false)]);
        assert_eq!(revenue, 60, "{:?}", routes);
        assert_eq!(routes.len(), 1);
    }

    #[test]
    fn group_counts_once() {
        // Two Canada offboards either side of the token city.
        let map = TestMap::new(
            vec![
                (
                    "W",
                    tile_from_dsl(
                        "w",
                        "offboard=revenue:yellow_30|brown_50,groups:Canada;path=a:1,b:_0",
                        TileColor::Red,
                    ),
                ),
                (
                    "A",
                    tokened(y("a", "city=revenue:20;path=a:4,b:_0;path=a:1,b:_0")),
                ),
                (
                    "E",
                    tile_from_dsl(
                        "e",
                        "offboard=revenue:yellow_30|brown_50,groups:Canada;path=a:4,b:_0",
                        TileColor::Red,
                    ),
                ),
            ],
            &[("W", 1, "A"), ("A", 1, "E")],
        );
        let (routes, revenue) = map.routes(&[city("A")], &[(3, false)]);
        assert_eq!(revenue, 50, "one Canada offboard: {:?}", routes);
        assert_eq!(routes[0].nodes.len(), 2);
    }

    #[test]
    fn offboard_ends_a_route() {
        // A(token) -- O(offboard, two paths) -- C: no running through O.
        let map = TestMap::new(
            vec![
                ("A", tokened(y("a", "city=revenue:20;path=a:1,b:_0"))),
                (
                    "O",
                    tile_from_dsl(
                        "o",
                        "offboard=revenue:yellow_30|brown_50;path=a:4,b:_0;path=a:1,b:_0",
                        TileColor::Red,
                    ),
                ),
                ("C", y("c", "city=revenue:40;path=a:4,b:_0")),
            ],
            &[("A", 1, "O"), ("O", 1, "C")],
        );
        let (_, revenue) = map.routes(&[city("A")], &[(3, false)]);
        assert_eq!(revenue, 50);
    }

    #[test]
    fn two_trains_no_shared_track() {
        let map = TestMap::new(
            vec![
                (
                    "A",
                    tokened(tile_from_dsl(
                        "a",
                        "city=revenue:10;path=a:1,b:_0;path=a:3,b:_0",
                        TileColor::Green,
                    )),
                ),
                ("B", y("b", "path=a:4,b:1")),
                ("C", y("c", "city=revenue:30;path=a:4,b:_0")),
                ("D", y("d", "city=revenue:20;path=a:0,b:_0")),
            ],
            &[("A", 1, "B"), ("B", 1, "C"), ("A", 3, "D")],
        );
        let (routes, revenue) = map.routes(&[city("A")], &[(2, false), (2, false)]);
        assert_eq!(revenue, 70, "A-C=40 + A-D=30");
        assert_eq!(routes.len(), 2);
        assert_ne!(routes[0].train_index, routes[1].train_index);
        // D-A-C (60) for the 3-train alone pays less than splitting.
        let (routes, revenue) = map.routes(&[city("A")], &[(2, false), (3, false)]);
        assert_eq!(revenue, 70, "{:?}", routes);
        assert_eq!(routes.len(), 2);
    }

    #[test]
    fn town_adds_revenue_on_route() {
        let map = TestMap::new(
            vec![
                ("A", tokened(y("a", "city=revenue:20;path=a:1,b:_0"))),
                ("B", y("b", "town=revenue:10;path=a:4,b:_0;path=a:_0,b:1")),
                ("C", y("c", "city=revenue:30;path=a:4,b:_0")),
            ],
            &[("A", 1, "B"), ("B", 1, "C")],
        );
        let (_, revenue) = map.routes(&[city("A")], &[(3, false)]);
        assert_eq!(revenue, 60, "20+10+30");
        // A 2-train can't skip the town.
        let (_, revenue) = map.routes(&[city("A")], &[(2, false)]);
        assert_eq!(revenue, 30, "A+town");
        // A D train counts the town free.
        let (_, revenue) = map.routes(&[city("A")], &[(2, true)]);
        assert_eq!(revenue, 60);
    }

    /// H12-like: a city with a bypass (edge 1 <-> edge 4) between SRC and DST.
    fn bypass(blocked: bool) -> TestMap {
        let mut h12 = tile_from_dsl(
            "h12",
            "city=revenue:10;path=a:1,b:_0;path=a:4,b:_0;path=a:1,b:4",
            TileColor::Gray,
        );
        if blocked {
            place_token(&mut h12, 0, "NYC");
        }
        TestMap::new(
            vec![
                ("SRC", tokened(y("src", "city=revenue:20;path=a:1,b:_0"))),
                ("H12", h12),
                ("DST", y("dst", "city=revenue:30;path=a:4,b:_0")),
            ],
            &[("SRC", 1, "H12"), ("H12", 1, "DST")],
        )
    }

    #[test]
    fn blocked_city_with_bypass() {
        // Through the blocked city only on the bypass.
        let (routes, revenue) = bypass(true).routes(&[city("SRC")], &[(3, false)]);
        assert_eq!(revenue, 50);
        assert_eq!(stops(&routes[0]), vec!["SRC", "DST"]);
        assert_eq!(routes[0].connections, vec![vec!["SRC", "H12", "DST"]]);
        // A blocked city still ends a route (Python stops there).
        let (routes, revenue) = bypass(true).routes(&[city("SRC")], &[(2, false), (2, false)]);
        assert_eq!(
            revenue, 50,
            "SRC-DST on the bypass; SRC-H12 would share track: {:?}",
            routes
        );
        let (_, revenue) = bypass(false).routes(&[city("SRC")], &[(3, false)]);
        assert_eq!(revenue, 60, "unblocked: SRC-H12-DST");
    }

    #[test]
    fn identical_trains_share_out_routes() {
        // Token city A with four spokes to cities worth 10/20/30/40.
        let map = TestMap::new(
            vec![
                ("A", tokened(tile_from_dsl("a", "city=revenue:0,slots:2;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0;path=a:3,b:_0", TileColor::Green))),
                ("B0", y("b0", "city=revenue:10;path=a:3,b:_0")),
                ("B1", y("b1", "city=revenue:20;path=a:4,b:_0")),
                ("B2", y("b2", "city=revenue:30;path=a:5,b:_0")),
                ("B3", y("b3", "city=revenue:40;path=a:0,b:_0")),
            ],
            &[("A", 0, "B0"), ("A", 1, "B1"), ("A", 2, "B2"), ("A", 3, "B3")],
        );
        let (routes, revenue) = map.routes(&[city("A")], &[(2, false), (2, false), (2, false)]);
        assert_eq!(revenue, 90, "40+30+20: {:?}", routes);
        let mut trains: Vec<usize> = routes.iter().map(|r| r.train_index).collect();
        trains.sort();
        assert_eq!(trains, vec![0, 1, 2]);
    }

    #[test]
    fn many_hexsides() {
        // A 301-hex line with cities at H0, H150 (token) and H300: the two
        // routes out of H150 use 300 hexsides between them, and must not
        // be mistaken for overlapping.
        let mut tiles = Vec::new();
        let mut links = Vec::new();
        for i in 0..=300 {
            let dsl = match i {
                0 => "city=revenue:10;path=a:1,b:_0",
                150 => "city=revenue:10;path=a:4,b:_0;path=a:1,b:_0",
                300 => "city=revenue:10;path=a:4,b:_0",
                _ => "path=a:4,b:1",
            };
            let tile = if i == 150 {
                tokened(y("t", dsl))
            } else {
                y("t", dsl)
            };
            tiles.push((format!("H{}", i), tile));
            if i > 0 {
                links.push((format!("H{}", i - 1), 1u8, format!("H{}", i)));
            }
        }
        let tiles: Vec<(&str, Tile)> = tiles.iter().map(|(h, t)| (h.as_str(), t.clone())).collect();
        let links: Vec<(&str, u8, &str)> = links
            .iter()
            .map(|(a, e, b)| (a.as_str(), *e, b.as_str()))
            .collect();
        let map = TestMap::new(tiles, &links);
        let (routes, revenue) = map.routes(&[city("H150")], &[(2, false), (2, false)]);
        assert_eq!(
            revenue,
            40,
            "{:?}",
            routes.iter().map(stops).collect::<Vec<_>>()
        );
    }
}
