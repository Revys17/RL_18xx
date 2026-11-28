//! Runs checked and priced by engine-rs (docs/SITE_ENGINE_RS_STEP3.md §6): any valid run is accepted, not only the
//! auto-router's. A posted run's routes are rebuilt from their `connections` (and `nodes`) on the position as Ruby's
//! `Route#connection_data` builds them, then Ruby's `Route#revenue` checks run in Ruby's order, then each route is
//! priced, and the stored action gets Ruby's fields (`hexes`, `revenue`, `revenue_str`, `nodes`).
//!
//! Ruby's rebuilding (lib/engine/route.rb): each connection is a hex chain, one hex id per tile path (a path through
//! a junction already passed adds none: `chain_id`); `find_matching_chains` walks from every node of the chain's first
//! hex to every node of its last (`get_node_chains`: `Path#walk` from the start node's paths, through hexsides and
//! junctions, never through a node, a hexside at most once) and keeps the walks whose `chain_id` is the chain. One
//! chain: the first match with a station of the company that shares no track with the other routes; several: each
//! pair of neighbouring chains joined at exactly one node (`find_pairwise_chain`, the posted `nodes` breaking ties).
//! Ruby resolves a run's routes last first: each excludes the track of the routes after it (`compute_other_paths`
//! sees the routes still being built as empty).
//!
//! Checks (`Route#revenue`): at least 2 stops; a station of the company among them (`check_route_token`); a group
//! once; no terminal path inside the route (`check_terminals!`); no node twice (`check_cycles!`); the train's
//! distance (`check_distance!`: 1830's n and D, 1867's distance rows); no track twice in the run (`check_overlap!`:
//! a hexside side, a path between two nodes); no blocked node passed through (`check_connected!`: `Path#connects_to?`
//! at each joint); 1867: no hex twice (`G1867#revenue_for`). And the Route step's: the company's own trains, each
//! once, none that has run. 1830 and 1867 have no lanes, double towns, halts or route abilities.

use super::game::{refused, SiteGame};
use crate::game::BaseGame;
use crate::tiles::PathEndpoint as E;
use serde_json::{json, Value};

/// A node on the map: hex index, its tile's endpoint (City/Town/Offboard).
type Node = (usize, E);
/// A tile path: hex index, index into the tile's paths.
type PathRef = (usize, usize);

#[derive(Clone, Debug)]
struct Chain {
    /// The node the walk started from and the one it ended at (Ruby `chain[:nodes]`)
    a: Node,
    b: Node,
    /// The paths in walk order
    paths: Vec<PathRef>,
}

/// One connection of a route as Ruby's `connection_data` keeps it.
#[derive(Clone, Debug)]
struct Conn {
    left: Node,
    right: Node,
    chain: Chain,
}

fn is_node(e: &E) -> bool {
    matches!(e, E::City(_) | E::Town(_) | E::Offboard(_))
}

/// Ruby `tile.nodes`: the paths' nodes in path order, each once.
fn tile_nodes(game: &BaseGame, hex: usize) -> Vec<E> {
    let mut out: Vec<E> = Vec::new();
    for p in &game.hexes[hex].tile.paths {
        for e in [&p.a, &p.b] {
            if is_node(e) && !out.contains(e) {
                out.push(e.clone());
            }
        }
    }
    out
}

/// Ruby `Node#signature`'s index: the node's place among its tile's parts (cities, then towns, then offboards in
/// 1830's and 1867's tiles), in Ruby's numbering (G19's preprinted cities swapped).
pub fn node_index(g: &SiteGame, hex: usize, e: &E) -> usize {
    let h = &g.game.hexes[hex];
    let tile = &h.tile;
    match e {
        E::City(i) => {
            if g.swapped.contains(&h.id.as_str()) && tile.cities.len() == 2 {
                1 - *i
            } else {
                *i
            }
        }
        E::Town(i) => tile.cities.len() + i,
        E::Offboard(i) => tile.cities.len() + tile.towns.len() + i,
        _ => 0,
    }
}

fn signature(g: &SiteGame, n: &Node) -> String {
    format!("{}-{}", g.game.hexes[n.0].id, node_index(g, n.0, &n.1))
}

struct Walker<'a> {
    game: &'a BaseGame,
    ids: &'a [String],
    end: Node,
    visited: Vec<PathRef>,
    /// hexsides crossed out of (hex, edge): Ruby's `counter[edge.id]`
    crossed: Vec<(usize, u8)>,
    /// junction paths entered per hex (Ruby's `counter[@junction]`)
    junctions: Vec<usize>,
    /// how much of `ids` the walk so far has produced, and the junctions passed (Ruby `chain_id`'s junction_map)
    id_pos: usize,
    seen_junction: Vec<usize>,
    found: Vec<Vec<PathRef>>,
}

impl<'a> Walker<'a> {
    fn neighbor(&self, hex: usize, edge: u8) -> Option<usize> {
        let id = &self.game.hexes[hex].id;
        let nb = self.game.hex_adjacency.get(id)?.get(&edge)?;
        self.game.hex_idx.get(nb).copied()
    }

    /// Ruby `Path#walk` (converging: every walk backtracks), pruned to walks whose `chain_id` stays a prefix of `ids`.
    fn walk(&mut self, p: PathRef, skip: Option<u8>, jskip: bool) {
        let (hex, pi) = p;
        let path = &self.game.hexes[hex].tile.paths[pi];
        if self.visited.contains(&p) {
            return;
        }
        let junction = path.a == E::Junction || path.b == E::Junction;
        if junction && self.junctions.iter().filter(|&&h| h == hex).count() > 1 {
            return;
        }
        let edges: Vec<u8> = [&path.a, &path.b].iter().filter_map(|e| e.edge_num()).collect();
        if edges.iter().any(|&e| self.crossed.contains(&(hex, e))) {
            return;
        }
        if junction && path.terminal {
            return;
        }
        // chain_id: this path's hex unless it touches a junction already passed
        let adds = !(junction && self.seen_junction.contains(&hex));
        if adds {
            if self.ids.get(self.id_pos) != Some(&self.game.hexes[hex].id) {
                return;
            }
            self.id_pos += 1;
        }
        let marked = junction && !self.seen_junction.contains(&hex);
        if marked {
            self.seen_junction.push(hex);
        }
        self.visited.push(p);
        if junction {
            self.junctions.push(hex);
        }
        // yield: a chain when the path reaches the end node
        let reaches = (path.a == self.end.1 || path.b == self.end.1) && hex == self.end.0;
        if reaches && self.id_pos == self.ids.len() {
            self.found.push(self.visited.clone());
        }
        if junction && !jskip {
            let n = self.game.hexes[hex].tile.paths.len();
            for qi in 0..n {
                let q = &self.game.hexes[hex].tile.paths[qi];
                if q.a == E::Junction || q.b == E::Junction {
                    self.walk((hex, qi), None, true);
                }
            }
        }
        for e in edges {
            if Some(e) == skip {
                continue;
            }
            let Some(nb) = self.neighbor(hex, e) else { continue };
            self.crossed.push((hex, e));
            let enter = (e + 3) % 6;
            let n = self.game.hexes[nb].tile.paths.len();
            for qi in 0..n {
                let q = &self.game.hexes[nb].tile.paths[qi];
                if q.a == E::Edge(enter) || q.b == E::Edge(enter) {
                    self.walk((nb, qi), Some(enter), false);
                }
            }
            self.crossed.pop();
        }
        self.visited.pop();
        if junction {
            self.junctions.pop();
        }
        if marked {
            self.seen_junction.pop();
        }
        if adds {
            self.id_pos -= 1;
        }
    }
}

/// Ruby `find_matching_chains(hex_ids)`: every walk between a node of the first hex and a node of the last whose
/// `chain_id` is `ids`, in Ruby's order (start nodes, end nodes, then the walks by length).
fn matching_chains(game: &BaseGame, ids: &[String]) -> Vec<Chain> {
    let mut out = Vec::new();
    if ids.len() < 2 && ids.first().is_none_or(|x| x.contains(' ')) {
        return out;
    }
    let (Some(&sh), Some(&eh)) = (game.hex_idx.get(&ids[0]), game.hex_idx.get(&ids[ids.len() - 1])) else { return out };
    for sn in tile_nodes(game, sh) {
        for en in tile_nodes(game, eh) {
            if sh == eh && sn == en {
                continue;
            }
            let mut w = Walker {
                game,
                ids,
                end: (eh, en.clone()),
                visited: Vec::new(),
                crossed: Vec::new(),
                junctions: Vec::new(),
                id_pos: 0,
                seen_junction: Vec::new(),
                found: Vec::new(),
            };
            let n = game.hexes[sh].tile.paths.len();
            for pi in 0..n {
                let p = &game.hexes[sh].tile.paths[pi];
                if p.a == sn || p.b == sn {
                    w.walk((sh, pi), None, false);
                }
            }
            let mut found = w.found;
            found.sort_by_key(|f| f.len());
            for paths in found {
                out.push(Chain { a: (sh, sn.clone()), b: (eh, en.clone()), paths });
            }
        }
    }
    out
}

fn disjoint(a: &[PathRef], b: &[PathRef]) -> bool {
    !a.iter().any(|p| b.contains(p))
}

/// A train's owner once discarded (Ruby's Depot: no city is tokened by it) and once rusted (nil: Ruby's
/// `City#tokened_by?(nil)` is true of a city with an empty slot, `nil&.corporation == nil`).
pub const DEPOT: &str = "\u{0}depot";
pub const NOBODY: &str = "\u{0}nil";

/// A city with a station of `corp` (Ruby `city_tokened_by?`); towns and offboards have none.
fn tokened_by(game: &BaseGame, n: &Node, corp: &str) -> bool {
    match &n.1 {
        E::City(i) => game.hexes[n.0].tile.cities.get(*i).is_some_and(|c| {
            if corp == NOBODY {
                return c.tokens.iter().any(|t| t.is_none());
            }
            c.tokens.iter().flatten().any(|t| t.corporation_id == corp) || c.extra_tokens.iter().any(|t| t.corporation_id == corp)
        }),
        _ => false,
    }
}

/// Ruby `Node#blocks?(corporation)`: an offboard always; a city whose every slot holds another company's token,
/// none of them neutral.
fn blocks(game: &BaseGame, n: &Node, corp: &str) -> bool {
    match &n.1 {
        E::Offboard(_) => true,
        E::City(i) => game.hexes[n.0].tile.cities.get(*i).is_some_and(|c| {
            !tokened_by(game, n, corp)
                && !c.tokens.iter().any(|t| t.is_none())
                && !c.tokens.iter().flatten().any(|t| t.token_type == "neutral")
        }),
        _ => false,
    }
}

fn groups(game: &BaseGame, n: &Node) -> Vec<String> {
    match &n.1 {
        E::Offboard(i) => game.hexes[n.0].tile.offboards.get(*i).map(|o| o.groups.clone()).unwrap_or_default(),
        _ => Vec::new(),
    }
}

fn node_revenue(game: &BaseGame, n: &Node) -> i32 {
    let tile = &game.hexes[n.0].tile;
    match &n.1 {
        E::City(i) => tile.cities.get(*i).map_or(0, |c| c.revenue),
        E::Town(i) => tile.towns.get(*i).map_or(0, |t| t.revenue),
        E::Offboard(i) => tile.offboards.get(*i).map_or(0, |o| o.phase_revenue(&game.phase.tiles)),
        _ => 0,
    }
}

fn kind(n: &Node) -> &'static str {
    match n.1 {
        E::City(_) => "city",
        E::Town(_) => "town",
        _ => "offboard",
    }
}

/// A route rebuilt: its connections, the posted parts kept for the stored action.
struct Built {
    conns: Vec<Conn>,
}

impl Built {
    fn paths(&self) -> Vec<PathRef> {
        self.conns.iter().flat_map(|c| c.chain.paths.iter().copied()).collect()
    }
    /// Ruby `visited_stops`: the connections' left and right nodes, each once.
    fn stops(&self) -> Vec<Node> {
        let mut out: Vec<Node> = Vec::new();
        for c in &self.conns {
            for n in [&c.left, &c.right] {
                if !out.contains(n) {
                    out.push(n.clone());
                }
            }
        }
        out
    }
    /// Ruby `ordered_paths`: each connection's paths from its left node.
    fn ordered_paths(&self, game: &BaseGame) -> Vec<PathRef> {
        let mut out = Vec::new();
        for c in &self.conns {
            let ps = &c.chain.paths;
            if ps.is_empty() {
                continue;
            }
            let first = &game.hexes[ps[0].0].tile.paths[ps[0].1];
            let from_left = ps[0].0 == c.left.0 && (first.a == c.left.1 || first.b == c.left.1);
            if from_left {
                out.extend(ps.iter().copied());
            } else {
                out.extend(ps.iter().rev().copied());
            }
        }
        out
    }
}

fn conn_lists(r: &Value) -> Vec<Vec<String>> {
    r.get("connections")
        .and_then(Value::as_array)
        .map(|cs| cs.iter().map(|c| c.as_array().map(|x| x.iter().filter_map(|y| y.as_str().map(str::to_string)).collect()).unwrap_or_default()).collect())
        .unwrap_or_default()
}

/// Ruby `connection_data` of one route, `other` the track of the routes after it.
fn build(g: &SiteGame, corp: &str, r: &Value, other: &[PathRef]) -> Result<Built, String> {
    let game = &g.game;
    let lists = conn_lists(r);
    if lists.iter().any(|l| l.iter().any(|h| h.contains(' ') || h == "local")) {
        return Err(super::game::unsupported("a local train's or an intra-tile connection (none in 1830 or 1867)"));
    }
    if lists.is_empty() {
        return Ok(Built { conns: Vec::new() });
    }
    let posted_nodes: Vec<String> =
        r.get("nodes").and_then(Value::as_array).map(|v| v.iter().filter_map(|x| x.as_str().map(str::to_string)).collect()).unwrap_or_default();
    let possibilities: Vec<Vec<Chain>> = lists.iter().map(|ids| matching_chains(game, ids)).collect();
    let mut conns = Vec::new();
    if possibilities.len() == 1 {
        let chain = possibilities[0]
            .iter()
            .find(|ch| (tokened_by(game, &ch.a, corp) || tokened_by(game, &ch.b, corp)) && disjoint(&ch.paths, other))
            .cloned();
        if let Some(ch) = chain {
            conns.push(Conn { left: ch.a.clone(), right: ch.b.clone(), chain: ch });
        }
        return Ok(Built { conns });
    }
    let mut other: Vec<PathRef> = other.to_vec();
    for (index, pair) in possibilities.windows(2).enumerate() {
        let a_list: Vec<&Chain> = pair[0].iter().filter(|c| disjoint(&c.paths, &other)).collect();
        let b_list: Vec<&Chain> = pair[1].iter().filter(|c| disjoint(&c.paths, &other)).collect();
        let mut cands: Vec<(&Chain, &Chain, Node, Node, Node)> = Vec::new();
        for a in &a_list {
            for b in &b_list {
                let an = [&a.a, &a.b];
                let bn = [&b.a, &b.b];
                let middle: Vec<&Node> = an.iter().copied().filter(|n| bn.contains(n)).collect();
                let mut mid_u: Vec<&Node> = Vec::new();
                for m in middle {
                    if !mid_u.contains(&m) {
                        mid_u.push(m);
                    }
                }
                if mid_u.len() != 1 || !disjoint(&a.paths, &b.paths) {
                    continue;
                }
                let m = mid_u[0].clone();
                let left = an.iter().find(|n| ***n != m).map(|n| (*n).clone());
                let right = bn.iter().find(|n| ***n != m).map(|n| (*n).clone());
                let (Some(left), Some(right)) = (left, right) else { continue };
                cands.push((a, b, left, right, m));
            }
        }
        if cands.is_empty() {
            return Ok(Built { conns: Vec::new() });
        }
        let pick = if cands.len() == 1 || posted_nodes.is_empty() {
            0
        } else {
            // max_by the posted nodes it passes (the first of the best)
            let score = |c: &(&Chain, &Chain, Node, Node, Node)| {
                [&c.2, &c.3, &c.4].iter().filter(|n| posted_nodes.contains(&signature(g, n))).count()
            };
            let best = cands.iter().map(score).max().unwrap_or(0);
            cands.iter().position(|c| score(c) == best).unwrap_or(0)
        };
        let (a, b, left, right, middle) = cands[pick].clone();
        if index == 0 {
            conns.push(Conn { left, right: middle.clone(), chain: a.clone() });
        }
        conns.push(Conn { left: middle, right, chain: b.clone() });
        other.extend(a.paths.iter().copied());
    }
    Ok(Built { conns })
}

/// A train's distance rows: (node kinds, visits allowed) (Ruby `train.distance`; a number is every kind).
fn distance_rows(g: &SiteGame, train_name: &str) -> Vec<(Vec<&'static str>, u32)> {
    if g.game.title == "1867" {
        match train_name {
            "5+5E" => vec![(vec!["offboard"], 5), (vec!["city", "town"], 99)],
            n => {
                let d: u32 = n.split('+').next().and_then(|x| x.parse().ok()).unwrap_or(0);
                vec![(vec!["city", "offboard"], d), (vec!["town"], 99)]
            }
        }
    } else {
        let d = g.game.title_def().trains().iter().find(|t| t.name == train_name).map_or(0, |t| t.distance);
        vec![(vec!["city", "town", "offboard"], d)]
    }
}

/// Ruby `check_distance` over the visited stops (each stop costs 1).
fn check_distance(g: &SiteGame, train_name: &str, stops: &[Node]) -> Result<(), String> {
    let rows = distance_rows(g, train_name);
    if rows.len() == 1 {
        let n = stops.len() as u32;
        if n > rows[0].1 {
            return Err(format!("{n} is too many stops for {} train", rows[0].1));
        }
        return Ok(());
    }
    let mut left: Vec<u32> = rows.iter().map(|r| r.1).collect();
    for k in ["city", "offboard", "town"] {
        let mut num = stops.iter().filter(|s| kind(s) == k).count() as u32;
        for (i, r) in rows.iter().enumerate() {
            if num == 0 {
                break;
            }
            if !r.0.contains(&k) || left[i] == 0 {
                continue;
            }
            let take = num.min(left[i]);
            left[i] -= take;
            num -= take;
        }
        if num > 0 {
            return Err("Route has too many stops".into());
        }
    }
    Ok(())
}

/// Ruby `check_overlap`'s keys of a path: each hexside side it ends at, and the path itself between two nodes.
fn overlap_keys(game: &BaseGame, p: PathRef) -> Vec<(usize, i32)> {
    let path = &game.hexes[p.0].tile.paths[p.1];
    let mut out = Vec::new();
    for e in [&path.a, &path.b] {
        if let Some(n) = e.edge_num() {
            out.push((p.0, n as i32));
        }
    }
    if is_node(&path.a) && is_node(&path.b) {
        out.push((p.0, 100 + p.1 as i32));
    }
    out
}

/// The routes of a posted run, rebuilt and checked; Ok(each route's stops) or the refusal's words.
fn check_routes(g: &SiteGame, corp: &str, routes: &[Value]) -> Result<Vec<Built>, String> {
    let game = &g.game;
    let ci = *game.corp_idx.get(corp).ok_or_else(|| format!("No company {corp}"))?;
    // the Route step's own checks (Step::Route#process_run_routes)
    let mut trains_seen: Vec<String> = Vec::new();
    for r in routes {
        let train = r.get("train").and_then(Value::as_str).unwrap_or("");
        let Some(t) = game.corporations[ci].trains.iter().find(|t| t.id == train) else {
            return Err("Cannot run another corporation's train. refresh".into());
        };
        if trains_seen.iter().any(|x| x == train) {
            return Err("Cannot run train twice".into());
        }
        if t.operated {
            return Err("Cannot run train that operated".into());
        }
        trains_seen.push(train.to_string());
    }
    // rebuilt last first, each excluding the track of the routes after it
    let mut built: Vec<Option<Built>> = (0..routes.len()).map(|_| None).collect();
    let mut after: Vec<PathRef> = Vec::new();
    for k in (0..routes.len()).rev() {
        let b = build(g, corp, &routes[k], &after)?;
        after.extend(b.paths());
        built[k] = Some(b);
    }
    let built: Vec<Built> = built.into_iter().map(Option::unwrap).collect();
    let mut overlap: Vec<(usize, i32)> = Vec::new();
    for (k, (r, b)) in routes.iter().zip(&built).enumerate() {
        let train_id = r.get("train").and_then(Value::as_str).unwrap_or("");
        let train_name = game.corporations[ci].trains.iter().find(|t| t.id == train_id).map(|t| t.name.clone()).unwrap_or_default();
        let stops = b.stops();
        let n = k + 1;
        if !b.conns.is_empty() && stops.len() < 2 {
            return Err(format!("Route {n}: Route must have at least 2 stops"));
        }
        if !stops.iter().any(|s| tokened_by(game, s, corp)) {
            return Err(format!("Route {n}: Route must contain token"));
        }
        let mut gs: Vec<String> = Vec::new();
        for s in &stops {
            for gname in groups(game, s) {
                if gs.contains(&gname) {
                    return Err(format!("Route {n}: Cannot use group {gname} more than once"));
                }
                gs.push(gname);
            }
        }
        let ordered = b.ordered_paths(game);
        if ordered.len() >= 3 && ordered[1..ordered.len() - 1].iter().any(|p| game.hexes[p.0].tile.paths[p.1].terminal) {
            return Err(format!("Route {n}: Route cannot pass through terminal"));
        }
        let mut cycles: Vec<&Node> = Vec::new();
        for c in &b.conns {
            if !cycles.contains(&&c.left) {
                cycles.push(&c.left);
            }
            if cycles.contains(&&c.right) {
                return Err(format!("Route {n}: Cannot use {} twice", game.hexes[c.right.0].id));
            }
            cycles.push(&c.right);
        }
        check_distance(g, &train_name, &stops).map_err(|e| format!("Route {n}: {e}"))?;
        // check_overlap: every route's track so far, this one's included
        for p in b.paths() {
            for key in overlap_keys(game, p) {
                if overlap.contains(&key) {
                    return Err(format!("Route {n}: Route cannot reuse track on {}", game.hexes[key.0].id));
                }
                overlap.push(key);
            }
        }
        // check_connected: a train passes through no node that blocks it
        for c in b.conns.iter().take(b.conns.len().saturating_sub(1)) {
            if blocks(game, &c.right, corp) {
                return Err(format!("Route {n}: Route is not connected"));
            }
        }
        if game.title == "1867" {
            let hexes = route_hexes(g, r, b);
            let mut seen: Vec<&String> = Vec::new();
            for h in &hexes {
                if seen.contains(&h) {
                    return Err(format!("Route {n}: Route visits same hex twice"));
                }
                seen.push(h);
            }
        }
    }
    Ok(built)
}

/// Ruby `Route#hexes`: the posted list unless stale (fewer hexes than nodes), else the stops' hexes in route order
/// (consecutive connections' shared node once).
fn route_hexes(g: &SiteGame, r: &Value, b: &Built) -> Vec<String> {
    let posted: Option<Vec<String>> =
        r.get("hexes").and_then(Value::as_array).map(|v| v.iter().filter_map(|x| x.as_str().map(str::to_string)).collect());
    let n_nodes = r.get("nodes").and_then(Value::as_array).map(Vec::len);
    let stale = matches!((&posted, n_nodes), (Some(h), Some(n)) if h.len() < n);
    if let (Some(h), false) = (posted, stale) {
        return h;
    }
    let mut nodes: Vec<&Node> = Vec::new();
    for c in &b.conns {
        for n in [&c.left, &c.right] {
            if nodes.last() != Some(&n) {
                nodes.push(n);
            }
        }
    }
    nodes.iter().map(|n| g.game.hexes[n.0].id.clone()).collect()
}

/// ports:last_run's routes as Opal's map rebuilds them on the board now (`Engine::Route.new(connection_hexes:, nodes:,
/// routes: built)` per train, then `Route#hexes`): each route excludes the track of the ones before it; None for a
/// route whose chains no longer fit the map (Ruby's `connection_data` comes back empty).
pub fn rebuilt_hexes(g: &SiteGame, corp: &str, routes: &[Value]) -> Vec<Option<Vec<String>>> {
    let mut before: Vec<PathRef> = Vec::new();
    let mut out = Vec::new();
    let _ = corp;
    for r in routes {
        // Route#corporation is the train's owner now (`train_owner(train)`): a train sold on is run for its new owner,
        // a discarded one for the depot, a rusted one for nil
        let train = r.get("train").and_then(Value::as_str).unwrap_or("");
        // (a route of several chains is rebuilt whoever owns the train: find_pairwise_chain asks for no station)
        let owner = g.train_owners.get(train).cloned().flatten().unwrap_or_else(|| DEPOT.to_string());
        let built = build(g, &owner, r, &before).ok().filter(|b| !b.conns.is_empty());
        match built {
            Some(b) => {
                before.extend(b.paths());
                let mut bare = r.clone();
                if let Some(m) = bare.as_object_mut() {
                    m.remove("hexes");
                }
                out.push(Some(route_hexes(g, &bare, &b)));
            }
            None => out.push(None),
        }
    }
    out
}

/// Ruby `revenue_str`: the stops' hexes in the order the train reaches them, each stop once (`visited_hexes`).
fn revenue_str(g: &SiteGame, b: &Built) -> String {
    b.stops().iter().map(|n| g.game.hexes[n.0].id.clone()).collect::<Vec<_>>().join("-")
}

/// Ruby `node_signatures` once rebuilt: the chains' nodes (each as walked: start, end), each once.
fn node_sigs(g: &SiteGame, b: &Built) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for c in &b.conns {
        for n in [&c.chain.a, &c.chain.b] {
            let s = signature(g, n);
            if !out.contains(&s) {
                out.push(s);
            }
        }
    }
    out
}

/// A route's revenue: 1830 the stops' revenues (every stop counts: n-trains to n, D unlimited); 1867 Ruby's
/// `compute_stops` + `revenue_for` over the stops the rebuilt route visits (`price_1867`).
fn price(g: &SiteGame, corp: &str, r: &Value, b: &Built) -> Result<i32, String> {
    if g.game.title == "1867" {
        return price_1867(g, corp, r.get("train").and_then(Value::as_str).unwrap_or(""), b);
    }
    Ok(b.stops().iter().map(|n| node_revenue(&g.game, n)).sum())
}

/// Ruby G1867 `compute_stops` + `revenue_for` over the route's `visited_stops`, each stop the node the route visits:
/// its own revenue and `stop.tokened_by?(route.corporation)` of that city (Toronto's and Montreal's cities apart: a
/// 5+5E's stops must include a city holding the company's station, not just a hex that has one; the replay pricer
/// `title::g1867::recorded_route_revenue` asks per hex, the step-2 "R6"). The rest as that pricer: the counted kinds
/// (5+5E: offboards; else cities and offboards) to the train's distance, the others filling best-first, the
/// multiplier, the companies' hex bonuses (per hex once) and the capitals bonus.
fn price_1867(g: &SiteGame, corp: &str, train: &str, b: &Built) -> Result<i32, String> {
    use crate::revenue::{RouteBonuses, Stop, StopKind};
    let game = &g.game;
    let name = train.rsplit_once('-').map_or(train, |(n, _)| n);
    let tdef = crate::title::g1867::trains().into_iter().find(|t| t.name == name).ok_or_else(|| format!("recorded route for unknown train {train:?}"))?;
    let counted: &[StopKind] = if tdef.name == "5+5E" { &[StopKind::Offboard] } else { &[StopKind::City, StopKind::Offboard] };
    let stops: Vec<Stop> = b
        .stops()
        .iter()
        .map(|n| Stop {
            hex: game.hexes[n.0].id.clone(),
            kind: match n.1 {
                E::City(_) => StopKind::City,
                E::Town(_) => StopKind::Town,
                _ => StopKind::Offboard,
            },
            revenue: node_revenue(game, n),
            tokened: tokened_by(game, n, corp),
        })
        .collect();
    let mut bonuses = RouteBonuses {
        capital_hexes: crate::title::g1867::BONUS_CAPITALS.iter().map(|s| s.to_string()).collect(),
        capital_partner: Some(crate::title::g1867::BONUS_REVENUE.to_string()),
        capital_amount: 40,
        ..Default::default()
    };
    let corp_eid = crate::entities::EntityId::corporation(corp);
    for company in &game.companies {
        if company.closed || company.owner != corp_eid {
            continue;
        }
        for (hexes, amount) in crate::abilities::hex_bonuses("1867", &company.sym) {
            bonuses.hex_bonuses.push((hexes.iter().map(|h| h.to_string()).collect(), amount));
        }
    }
    crate::revenue::price_bucketed_stops(&stops, tdef.distance as usize, tdef.multiplier as i32, counted, &bonuses).map_err(|e| e.message)
}

/// A stored 1867 run priced as the authority prices a posted one (the shadow key's `last_action`): its routes
/// rebuilt on the position before it (last first, as `check_routes`) and each priced by `price_1867`; None when a
/// route can't be rebuilt (the caller falls back on the replay pricer).
pub fn price_stored_1867(g: &SiteGame, corp: &str, routes: &[Value]) -> Option<Vec<i32>> {
    let mut built: Vec<Option<Built>> = (0..routes.len()).map(|_| None).collect();
    let mut after: Vec<PathRef> = Vec::new();
    for k in (0..routes.len()).rev() {
        let b = build(g, corp, &routes[k], &after).ok().filter(|b| !b.conns.is_empty())?;
        after.extend(b.paths());
        built[k] = Some(b);
    }
    routes
        .iter()
        .zip(built)
        .map(|(r, b)| price_1867(g, corp, r.get("train").and_then(Value::as_str).unwrap_or(""), &b?).ok())
        .collect()
}

/// A posted `run_routes` checked and priced: Ok(the action to process and store: each route's `revenue`,
/// `revenue_str`, `hexes`, `nodes` as Ruby stores them; `extra_revenue` / `subsidy` 0 where posted) or Err(refused).
pub fn checked_run(g: &mut SiteGame, a: &Value) -> Result<Value, String> {
    let corp = super::json::entity_of(a).map_err(|e| refused(e.message))?;
    // the Route step must be the company's now
    let types = if g.game.title == "1867" { super::g1867::ruby_actions(&mut g.game) } else { super::menu::ruby_types(g) };
    let current = if g.game.title == "1867" {
        super::g1867::active_step(&mut g.game).map(|(_, e)| super::g1867::entity_id(&e))
    } else {
        g.game.round_state.active_entity_id.corp_sym().map(str::to_string)
    };
    if !types.iter().any(|t| t == "run_routes") || current.as_deref() != Some(corp.as_str()) {
        return Err(refused(format!("{corp} cannot run routes now")));
    }
    let routes: Vec<Value> = a.get("routes").and_then(Value::as_array).cloned().unwrap_or_default();
    // G19's cities in the engine's numbering for the rebuilding (the posted `nodes` keep Ruby's)
    let built = check_routes(g, &corp, &routes).map_err(refused)?;
    let mut out = a.clone();
    let mut stored = Vec::new();
    for (r, b) in routes.iter().zip(&built) {
        let revenue = price(g, &corp, r, b).map_err(refused)?;
        let mut m = serde_json::Map::new();
        m.insert("train".into(), r.get("train").cloned().unwrap_or(Value::Null));
        m.insert("connections".into(), r.get("connections").cloned().unwrap_or(json!([])));
        m.insert("hexes".into(), json!(route_hexes(g, r, b)));
        m.insert("revenue".into(), json!(revenue));
        m.insert("revenue_str".into(), json!(revenue_str(g, b)));
        for k in ["halts", "abilities"] {
            if let Some(v) = r.get(k).filter(|v| !v.is_null()) {
                m.insert(k.into(), v.clone());
            }
        }
        let sigs = node_sigs(g, b);
        if !sigs.is_empty() {
            m.insert("nodes".into(), json!(sigs));
        }
        stored.push(Value::Object(m));
    }
    out["routes"] = json!(stored);
    for k in ["extra_revenue", "subsidy"] {
        if a.get(k).is_some_and(|v| !v.is_null()) {
            out[k] = json!(0);
        }
    }
    Ok(out)
}

/// lib/run_revenue.rb's `mismatches`: the posted numbers that differ from the stored ones.
pub fn mismatches(posted: &Value, stored: &Value) -> Vec<Value> {
    let mut out = Vec::new();
    let empty = Vec::new();
    let prs = posted.get("routes").and_then(Value::as_array).unwrap_or(&empty);
    let srs = stored.get("routes").and_then(Value::as_array).unwrap_or(&empty);
    for (i, sr) in srs.iter().enumerate() {
        let claim = prs.get(i).cloned().unwrap_or(json!({}));
        for k in ["revenue", "subsidy"] {
            let Some(c) = claim.get(k).filter(|v| !v.is_null()) else { continue };
            let v = if k == "subsidy" { sr.get(k).cloned().unwrap_or(json!(0)) } else { sr.get(k).cloned().unwrap_or(Value::Null) };
            if *c != v {
                out.push(json!({"path": format!("routes[{i}].{k}"), "posted": c, "stored": v}));
            }
        }
    }
    for k in ["extra_revenue", "subsidy"] {
        let Some(c) = posted.get(k).filter(|v| !v.is_null()) else { continue };
        let v = stored.get(k).cloned().unwrap_or(json!(0));
        if *c != v {
            out.push(json!({"path": k, "posted": c, "stored": v}));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distance_rows_follow_ruby() {
        // 1867's plain trains: cities and offboards to n, towns free; 5+5E: offboards to 5, the rest free
        let rows = |name: &str| -> Vec<(Vec<&'static str>, u32)> {
            match name {
                "5+5E" => vec![(vec!["offboard"], 5), (vec!["city", "town"], 99)],
                n => vec![(vec!["city", "offboard"], n.split('+').next().unwrap().parse().unwrap()), (vec!["town"], 99)],
            }
        };
        assert_eq!(rows("2+2")[0].1, 2);
        assert_eq!(rows("5+5E")[0].0, vec!["offboard"]);
    }

    /// A 5+5E's stops through Toronto: CNR's station is in F16's city 1, the national's in its city 0. A route at its
    /// four offboards with one city left to fill, through F16's city 0 and E15 (CNR's): Ruby's `compute_stops` takes
    /// E15 (the city 0 isn't CNR's); the per-hex replay pricer took Toronto, the dearer one.
    #[test]
    fn a_5_5e_counts_the_station_of_the_city_it_visits() {
        let data: Value = serde_json::from_str(include_str!("../../site/fixtures/g1867_nationalization_cash.json")).unwrap();
        let mut d = data.clone();
        let list = d["actions"].as_array().unwrap().clone();
        let at = list.iter().position(|a| a["id"] == json!(389)).unwrap();
        d["actions"] = json!(list[..at].to_vec());
        let mut g = SiteGame::load(&d).unwrap();
        g.site_log = false;
        g.run_to(None);
        let game = &g.game;
        let hex = |id: &str| game.hexes.iter().position(|h| h.id == id).unwrap();
        let (f16, e15) = (hex("F16"), hex("E15"));
        assert!(tokened_by(game, &(f16, E::City(1)), "CNR") && !tokened_by(game, &(f16, E::City(0)), "CNR"));
        assert!(tokened_by(game, &(e15, E::City(0)), "CNR"));
        let offboards: Vec<Node> = (0..game.hexes.len())
            .filter(|&h| !game.hexes[h].tile.offboards.is_empty() && game.hexes[h].id != "D2")
            .take(4)
            .map(|h| (h, E::Offboard(0)))
            .collect();
        assert_eq!(offboards.len(), 4);
        let mut nodes = offboards.clone();
        nodes.push((f16, E::City(0)));
        nodes.push((e15, E::City(0)));
        let chain = |a: &Node, b: &Node| Chain { a: a.clone(), b: b.clone(), paths: Vec::new() };
        let conns: Vec<Conn> = nodes.windows(2).map(|w| Conn { left: w[0].clone(), right: w[1].clone(), chain: chain(&w[0], &w[1]) }).collect();
        let b = Built { conns };
        let offs: i32 = offboards.iter().map(|n| node_revenue(game, n)).sum();
        let (toronto, e15_rev) = (node_revenue(game, &(f16, E::City(0))), node_revenue(game, &(e15, E::City(0))));
        assert!(toronto > e15_rev, "{toronto} {e15_rev}");
        assert_eq!(price_1867(&g, "CNR", "5+5E-0", &b), Ok((offs + e15_rev) * 2));
        // the per-hex pricer over the same stops
        let hexes: Vec<String> = nodes.iter().map(|n| game.hexes[n.0].id.clone()).collect();
        let rd = crate::actions::RouteData {
            train_name: "5+5E-0".into(),
            connections: hexes.windows(2).map(|w| w.to_vec()).collect(),
            hexes: Vec::new(),
            revenue: None,
            nodes: Vec::new(),
        };
        assert_eq!(crate::title::g1867::recorded_route_revenue(game, "CNR", &rd).unwrap(), (offs + toronto) * 2);
    }
}
