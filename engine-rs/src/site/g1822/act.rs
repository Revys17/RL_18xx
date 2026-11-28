//! A posted 1822-family move checked as the authority checks it (docs/SITE_ENGINE_RS_1822.md §4.2): it must match a
//! move of the legal-move generator's Ruby set (`CrossPlayerPolicy::RULES`: an Opal page or a bot may post a
//! cross-player move the site's own menus never offer, and Ruby accepts it), compared on what decides the move (the
//! RL importer's `action_key`: type, entity, company / corporation, the bundle's holders and size, hex / tile /
//! rotation / combo partners, city, train type and seller, choice, kind, target), a ranged move by range and step.
//! The stored action is the generator's own wire form with the posted price: what the engine and Ruby both take.
//! A run must be the router's (step 1's rule for the family: engine-rs's pricer is not a validator, §2.3 G2).

use super::super::game::{refused, SiteGame};
use crate::g1822::legal::Move;
use crate::title::CrossPlayerPolicy;
use serde_json::{json, Map, Value};

fn atype(a: &Value) -> &str {
    a.get("type").and_then(Value::as_str).unwrap_or("")
}

fn sval(v: Option<&Value>) -> String {
    match v {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Null) | None => String::new(),
        Some(x) => x.to_string(),
    }
}

fn ival(v: Option<&Value>) -> Option<i64> {
    match v {
        Some(Value::Number(n)) => n.as_i64().or_else(|| n.as_f64().filter(|f| f.fract() == 0.0).map(|f| f as i64)),
        Some(Value::String(s)) => s.split(',').next().and_then(|x| x.trim().parse().ok()),
        _ => None,
    }
}

/// The holder of a share certificate ("LNWR_3") as the engine names it.
fn share_owner(g: &SiteGame, id: &str) -> String {
    let Some((sym, idx)) = id.rsplit_once('_') else { return String::new() };
    let Ok(idx) = idx.parse::<usize>() else { return String::new() };
    let Some(&ci) = g.game.corp_idx.get(sym) else { return String::new() };
    g.game.corporations[ci].shares.get(idx).map(|s| s.owner.0.clone()).unwrap_or_default()
}

/// (name, holder) of a train id: a corporation's sym, or "depot".
pub fn train_info(g: &SiteGame, id: &str) -> (String, String) {
    for c in &g.game.corporations {
        if let Some(t) = c.trains.iter().find(|t| t.id == id) {
            return (t.name.clone(), c.sym.clone());
        }
    }
    for t in g.game.depot.trains.iter().chain(g.game.depot.discarded.iter()) {
        if t.id == id {
            return (t.name.clone(), "depot".into());
        }
    }
    (String::new(), String::new())
}

fn shares_key(g: &SiteGame, a: &Value) -> (String, Vec<String>, i64, bool) {
    let ids: Vec<String> = a.get("shares").and_then(Value::as_array).map(|v| v.iter().filter_map(|x| x.as_str().map(str::to_string)).collect()).unwrap_or_default();
    let mut owners: Vec<String> = ids.iter().map(|s| share_owner(g, s)).collect();
    owners.sort();
    let corp = ids.first().and_then(|i| i.rsplit_once('_').map(|x| x.0.to_string())).unwrap_or_else(|| sval(a.get("corporation")));
    let pres = ids.iter().any(|s| s.ends_with("_0"));
    (corp, owners, ival(a.get("percent")).unwrap_or(0), pres)
}

/// The parameters that decide an action's outcome (the importer's `action_key`), on the position before it.
fn key(g: &SiteGame, a: &Value) -> Value {
    match atype(a) {
        "bid" => json!([sval(a.get("company"))]),
        "buy_shares" | "sell_shares" => {
            let (c, o, p, pres) = shares_key(g, a);
            json!([c, o, p, pres])
        }
        "par" => json!([sval(a.get("corporation")), ival(a.get("share_price"))]),
        "lay_tile" => {
            let tile = sval(a.get("tile"));
            let base = tile.split('-').next().unwrap_or("").to_string();
            let mut combo: Vec<String> = a.get("combo_entities").and_then(Value::as_array).map(|v| v.iter().map(|x| sval(Some(x))).collect()).unwrap_or_default();
            combo.sort();
            json!([sval(a.get("hex")), base, ival(a.get("rotation")).unwrap_or(0), combo])
        }
        "place_token" | "remove_token" => json!([sval(a.get("city")), sval(a.get("hex"))]),
        "hex_token" => json!([sval(a.get("hex"))]),
        "dividend" => json!([sval(a.get("kind"))]),
        "choose" | "choose_ability" => json!([sval(a.get("choice"))]),
        "buy_train" => {
            let (name, owner) = train_info(g, &sval(a.get("train")));
            let variant = if owner == "depot" { sval(a.get("variant")) } else { String::new() };
            let exchange = a.get("exchange").is_some_and(|x| !x.is_null());
            json!([name, owner, variant, exchange])
        }
        "discard_train" => json!([train_info(g, &sval(a.get("train"))).0]),
        "merge" => json!([sval(a.get("corporation"))]),
        "acquire_company" => json!([sval(a.get("company"))]),
        "assign" => json!([sval(a.get("target")), sval(a.get("target_type"))]),
        _ => json!([]),
    }
}

/// The price a posted action chose: a bid's, a train's, a partial debt payoff's amount.
fn price_of(a: &Value) -> Option<i64> {
    match atype(a) {
        "bid" | "buy_train" => ival(a.get("price")),
        "payoff_player_debt_partial" => ival(a.get("amount")),
        _ => None,
    }
}

/// Whether the posted action (seated) is the generated move `m` (wire `w`).
fn same(g: &SiteGame, posted: &Value, w: &Value, m: &Move) -> bool {
    if atype(w) != atype(posted) || sval(w.get("entity")) != sval(posted.get("entity")) {
        return false;
    }
    if key(g, w) != key(g, posted) {
        return false;
    }
    let t = atype(posted);
    if (t == "buy_shares" || t == "sell_shares") && m.share_price.is_some() && ival(posted.get("share_price")) != m.share_price.map(|x| x as i64) {
        return false;
    }
    match (&m.price, price_of(posted)) {
        (Some(r), Some(p)) => {
            let p = p as i32;
            p >= r.lo && p <= r.hi && (p - r.lo) % r.step.max(1) == 0
        }
        (Some(_), None) => false,
        (None, p) => match t {
            "buy_train" => p == ival(w.get("price")),
            "payoff_player_debt_partial" => false,
            _ => true,
        },
    }
}

/// The stored form of a matched move: the generator's wire (the site's ids), the posted price kept.
fn canonical(g: &SiteGame, posted: &Value, w: &Value, m: &Move) -> Value {
    let mut out = super::to_site(g, w);
    if m.price.is_some() {
        match atype(posted) {
            "bid" | "buy_train" => out["price"] = posted["price"].clone(),
            "payoff_player_debt_partial" => out["amount"] = posted["amount"].clone(),
            _ => {}
        }
    }
    // the posted action's own metadata keys stay (`user`, `created_at`)
    if let (Some(o), Some(p)) = (out.as_object_mut(), posted.as_object()) {
        for k in ["user", "created_at"] {
            if let Some(v) = p.get(k) {
                o.insert(k.to_string(), v.clone());
            }
        }
    }
    out
}

/// The generator's moves now under Ruby's set, each with its wire form (engine ids).
pub fn rules_moves(g: &mut SiteGame) -> Vec<(Move, Value)> {
    let moves = g.game.g1822_legal_moves_with(CrossPlayerPolicy::RULES);
    moves.into_iter().map(|m| {
        let w = g.game.g1822_move_wire(&m);
        (m, w)
    }).collect()
}

/// Check a posted move (not a run): Ok(the action to store) or Err(refused: words).
pub fn check_move(g: &mut SiteGame, a: &Value) -> Result<Value, String> {
    let posted = super::to_engine(g, a);
    let moves = rules_moves(g);
    for (m, w) in &moves {
        if same(g, &posted, w, m) {
            return Ok(canonical(g, a, w, m));
        }
    }
    Err(refused(why_not(g, &posted, &moves)))
}

/// Why a posted move matches nothing: the engine's own words where a candidate it would be raises (the Ruby dead
/// ends), else which part is not on offer.
fn why_not(g: &mut SiteGame, posted: &Value, moves: &[(Move, Value)]) -> String {
    let t = atype(posted);
    let ent = sval(posted.get("entity"));
    let ent_site = match posted.get("entity_type").and_then(Value::as_str) {
        Some("player") => ent.parse::<u32>().map(|p| super::super::menu::player_name(g, g.site_id(p))).unwrap_or(ent.clone()),
        _ => ent.clone(),
    };
    if super::active(&g.game).is_none() {
        return "Nobody may act now".into();
    }
    if !moves.iter().any(|(_, w)| sval(w.get("entity")) == ent) {
        if let Some(e) = dead_end(g) {
            return e;
        }
        return format!("{ent_site} may not act now");
    }
    let same_type: Vec<&(Move, Value)> = moves.iter().filter(|(_, w)| atype(w) == t && sval(w.get("entity")) == ent).collect();
    if same_type.is_empty() {
        return format!("{ent_site} may not {} now", t.replace('_', " "));
    }
    let k = key(g, posted);
    if let Some((m, _)) = same_type.iter().find(|(_, w)| key(g, w) == k) {
        if let (Some(r), Some(p)) = (&m.price, price_of(posted)) {
            return format!("{p} is not a price on offer ({} to {}, in steps of {})", r.lo, r.hi, r.step);
        }
        return format!("{} is not on offer at that price", t.replace('_', " "));
    }
    format!("That {} is not on offer", t.replace('_', " "))
}

/// The engine's words when no move is left because every candidate raises (a Ruby dead end, docs/SITE_ENGINE_RS_1822.md
/// §5.2 4): the first "(Ruby raises)" message, else None.
pub fn dead_end(g: &mut SiteGame) -> Option<String> {
    if g.game.finished {
        return None;
    }
    let cands = g.game.g1822_candidates_debug();
    cands.into_iter().filter_map(|(_, e)| e).find(|e| e.contains("(Ruby raises)")).map(|e| e.trim_start_matches("GameError: ").to_string())
}

/// Ruby's revenue string of a route of the run (base `revenue_str` over `visited_hexes`, the site's Ruby: one entry
/// per stop; G1822 adds the destination bonus "(£20)"; 1822CA its sawmill and grain / port bonuses).
fn revenue_str(g: &SiteGame, corp: &str, hexes: &[String], dest_bonus: i32, sawmill: i32, grain: &str) -> String {
    let mut s = hexes.join("-");
    let cur = if super::is_ca(&g.game) { "$" } else { "£" };
    if dest_bonus > 0 {
        s.push_str(&format!(" ({cur}{dest_bonus})"));
    }
    if sawmill > 0 {
        s.push_str(&format!(" + Sawmill (${sawmill})"));
    }
    s.push_str(grain);
    let _ = corp;
    s
}

/// The hexes of a route's stops in the order the train reaches them (the site's Ruby `Route#visited_hexes`: the
/// connections' nodes in turn, one entry per stop; a local run's one stop).
fn visited(r: &crate::actions::RouteData) -> Vec<String> {
    if let Some(c) = r.connections.first().filter(|c| c.first().map(String::as_str) == Some("local")) {
        return c.get(1).cloned().into_iter().collect();
    }
    let ends: Vec<(String, String)> =
        r.connections.iter().filter_map(|c| Some((c.first()?.clone(), c.last()?.clone()))).collect();
    match ends.len() {
        0 => r.hexes.clone(),
        // one chain: its nodes as found, the first hex's then the last's (Route#connection_data)
        1 => vec![ends[0].0.clone(), ends[0].1.clone()],
        _ => {
            // (find_pairwise_chain: each chain's right node is the one it shares with the next)
            let (a, b) = &ends[0];
            let (c, d) = &ends[1];
            let (l, mut prev) = if b == c || b == d { (a.clone(), b.clone()) } else { (b.clone(), a.clone()) };
            let mut out = vec![l, prev.clone()];
            for (x, y) in &ends[1..] {
                let next = if *x == prev { y.clone() } else { x.clone() };
                out.push(next.clone());
                prev = next;
            }
            out
        }
    }
}

/// The destination bonus of each route of a run (G1822#destination_bonus: the biggest, on its route; ties to the
/// last): Ruby's `destination_token.city.route_revenue` when both the home and the destination station are stops.
fn destination_bonuses(g: &SiteGame, corp: &str, routes: &[crate::actions::RouteData]) -> Vec<i32> {
    let game = &g.game;
    let mut out = vec![0; routes.len()];
    if !game.is_major(corp) {
        return out;
    }
    let (Some(home), Some(dest)) = (game.g1822_home_node(corp), game.g1822_destination_node(corp)) else { return out };
    let (hs, ds) = (game.g1822_node_signature(home), game.g1822_node_signature(dest));
    let ci = game.ci(corp);
    let e = game.fam().e_train;
    let mut best: Option<(i32, usize)> = None;
    for (i, r) in routes.iter().enumerate() {
        if !(r.nodes.contains(&hs) && r.nodes.contains(&ds)) {
            continue;
        }
        let train = game.corporations[ci].trains.iter().find(|t| t.id == r.train_name);
        let mult = if train.is_some_and(|t| t.name == e) { 2 } else { 1 };
        let v = game.g1822_hex_stop_revenue(&game.hexes[dest.hex].id) * mult;
        if best.is_none_or(|(b, _)| v >= b) {
            best = Some((v, i));
        }
    }
    if let Some((v, i)) = best {
        out[i] = v;
    }
    out
}

/// 1822CA's `grain_and_port_bonus` description of a route.
fn grain_desc(g: &SiteGame, corp: &str, r: &crate::actions::RouteData) -> String {
    use crate::g1822::ca::{EAST_PORTS, GRAIN_ELEVATORS, WEST_PORTS};
    let game = &g.game;
    if !super::is_ca(game) || game.is_minor(corp) {
        return String::new();
    }
    let ci = game.ci(corp);
    let Some(t) = game.corporations[ci].trains.iter().find(|t| t.id == r.train_name) else { return String::new() };
    if game.rnd().pullman_train.as_deref() == Some(t.id.as_str()) || t.name == game.fam().e_train {
        return String::new();
    }
    let hexes: Vec<&str> = r.nodes.iter().map(|n| n.rsplit_once('-').map_or(n.as_str(), |x| x.0)).collect();
    let grain = hexes.iter().filter(|h| GRAIN_ELEVATORS.contains(h)).count() as i32;
    if grain == 0 {
        return String::new();
    }
    let ports = hexes.iter().any(|h| EAST_PORTS.contains(h)) as i32 + hexes.iter().any(|h| WEST_PORTS.contains(h)) as i32;
    let with_grain = game.rnd().ca.grain_train.as_deref() == Some(t.id.as_str());
    let mut s = String::new();
    if with_grain {
        let pb = 20 * ports;
        if pb > 0 {
            s.push_str(&format!(" + Port{} (${pb})", if ports == 1 { "" } else { "s" }));
        }
    } else {
        s.push_str(&format!(" + Grain (${})", 10 * grain));
        let pb = 10 * ports;
        if pb > 0 {
            s.push_str(&format!(" + Port{} (${pb})", if ports == 1 { "" } else { "s" }));
        }
    }
    s
}

/// 1822CA's sawmill bonus of each route (the biggest, on its route; doubled for an E train and for the destination
/// city of the route taking the destination bonus).
fn sawmill_bonuses(g: &SiteGame, corp: &str, routes: &[crate::actions::RouteData], dest: &[i32]) -> Vec<i32> {
    let game = &g.game;
    let mut out = vec![0; routes.len()];
    if !super::is_ca(game) {
        return out;
    }
    let ca = &game.g().ca;
    let Some(hex) = ca.sawmill_hex.clone() else { return out };
    if ca.sawmill_owner.as_deref().is_some_and(|o| o != corp) {
        return out;
    }
    let ci = game.ci(corp);
    let mut best: Option<(i32, usize)> = None;
    for (i, r) in routes.iter().enumerate() {
        let Some(stop) = r.nodes.iter().find(|n| n.rsplit_once('-').map_or(n.as_str(), |x| x.0) == hex) else { continue };
        let Some(t) = game.corporations[ci].trains.iter().find(|t| t.id == r.train_name) else { continue };
        let etrain = t.name == game.fam().e_train;
        let hi = game.hex_idx[hex.as_str()];
        let idx: usize = stop.rsplit_once('-').and_then(|x| x.1.parse().ok()).unwrap_or(0);
        let city = game.hexes[hi].tile.cities.get(idx);
        let tokened = city.is_some_and(|c| c.tokens.iter().flatten().any(|x| x.corporation_id == corp));
        if etrain && !tokened {
            continue;
        }
        let dest_here = city.is_some_and(|c| c.tokens.iter().flatten().any(|x| x.corporation_id == corp && x.token_type == "destination")) && dest[i] > 0;
        let mut v = ca.sawmill_bonus * if dest_here { 2 } else { 1 };
        if etrain {
            v *= 2;
        }
        if best.is_none_or(|(b, _)| v >= b) {
            best = Some((v, i));
        }
    }
    if let Some((v, i)) = best {
        out[i] = v;
    }
    out
}

/// A route with Ruby's `hexes` (`Route#hexes` from the track, one entry per stop: both cities of an OO hex count, as
/// RunRoutes#to_h stores them) where its list has fewer hexes than stops, or none (rl18xx-next's RunRoutes#h_to_args
/// takes them from the track then); `fallback`'s when the connections say nothing.
fn with_ruby_hexes(mut r: crate::actions::RouteData, fallback: &[String]) -> crate::actions::RouteData {
    r.revenue = None;
    if r.hexes.is_empty() || r.hexes.len() < r.nodes.len() {
        r.hexes = visited(&r);
        if r.hexes.is_empty() {
            r.hexes = fallback.to_vec();
        }
    }
    r
}

/// A route's single "local" connection (`[["local", hex]]`, Ruby's Route#connection_hexes of a local train's one stop).
fn is_local(r: &crate::actions::RouteData) -> bool {
    r.connections.len() == 1 && r.connections[0].first().map(String::as_str) == Some("local")
}

/// A connection's hex ("D35 0.1", an ambiguous one-hex connection, is D35's).
fn hex_of(h: &str) -> &str {
    h.split(' ').next().unwrap_or("")
}

/// The router's route `canon` (connections in travel order, `nodes` its stops in turn) in the form the posted
/// connections give it, when Ruby rebuilds those as this route (Route#connection_data): the router's connections in
/// turn or in reverse, each one's hexes either way round (a connection is found from either end, get_node_chains),
/// with the `nodes` Ruby writes for them (each connection's two stops in the order of its hexes, in turn, repeats
/// dropped: Route#node_signatures) and its `hexes` (Route#hexes: the stops' hexes in turn; a one-connection route's in
/// the connection's order). A local run, or a route with a connection inside one hex (both cities of an OO hex,
/// which Ruby's chain search finds both ways round), only as the router writes it. None for any other connections.
/// With it, the stops in the order the train reaches them (Ruby's `Route#stops`, which its pricing and the mail
/// contracts read; `node_signatures` need not be in that order): the router's, or their reverse.
fn ruby_form(conns: &[Vec<String>], canon: &crate::actions::RouteData) -> Option<(crate::actions::RouteData, Vec<String>)> {
    let k = canon.connections.len();
    let one_hex = canon.connections.iter().any(|c| c.first().map(|h| hex_of(h)) == c.last().map(|h| hex_of(h)));
    if is_local(canon) || one_hex || canon.nodes.len() != k + 1 {
        return (conns == canon.connections.as_slice()).then(|| (canon.clone(), canon.nodes.clone()));
    }
    if conns.len() != k {
        return None;
    }
    let stop_hex = |n: &String| n.rsplit_once('-').map_or(n.as_str(), |x| x.0).to_string();
    for forward in [true, false] {
        let mut nodes: Vec<String> = Vec::new();
        let mut ok = true;
        for (j, pc) in conns.iter().enumerate() {
            let i = if forward { j } else { k - 1 - j };
            let c = &canon.connections[i];
            let (a, b) = (&canon.nodes[i], &canon.nodes[i + 1]);
            let ends = if pc == c {
                [a, b]
            } else if pc.iter().rev().eq(c.iter()) {
                [b, a]
            } else {
                ok = false;
                break;
            };
            for n in ends {
                if !nodes.contains(n) {
                    nodes.push(n.clone());
                }
            }
        }
        if !ok {
            continue;
        }
        let hexes: Vec<String> = if k == 1 {
            nodes.iter().map(stop_hex).collect()
        } else if forward {
            canon.nodes.iter().map(stop_hex).collect()
        } else {
            canon.nodes.iter().rev().map(stop_hex).collect()
        };
        let travel: Vec<String> = if forward { canon.nodes.clone() } else { canon.nodes.iter().rev().cloned().collect() };
        return Some((crate::actions::RouteData { train_name: canon.train_name.clone(), hexes, revenue: None, connections: conns.to_vec(), nodes }, travel));
    }
    None
}

/// The hexes of a posted route's stops, sorted, from its connections (Ruby's Route#connection_data: each connection
/// runs from one stop to the next).
fn stop_hexes(r: &crate::actions::RouteData) -> Vec<String> {
    let mut bare = r.clone();
    bare.connections = r.connections.iter().map(|c| c.iter().map(|h| hex_of(h).to_string()).collect()).collect();
    let mut out = visited(&bare);
    out.sort();
    out
}

/// The hexes of a route's stops, sorted, from its nodes ("D35-1").
fn node_hexes(r: &crate::actions::RouteData) -> Vec<String> {
    let mut out: Vec<String> = r.nodes.iter().map(|n| n.rsplit_once('-').map_or(n.as_str(), |x| x.0).to_string()).collect();
    out.sort();
    out
}

/// A posted run: it must be the router's (the same trains, each over the same stops: a route's `nodes`, or, posted
/// without them as Ruby's RunRoutes allows and older records are, the hexes its connections end on); stored with
/// Ruby's fields (`hexes`, `revenue`, `revenue_str`, `nodes`, `connections`), `extra_revenue` 0 and the run's mail
/// subsidy. Each route is stored with the posted connections where Ruby rebuilds them as the router's route
/// (`ruby_form`, with the nodes and hexes Ruby writes for them), else in the router's form: never in a form Ruby
/// can't rebuild or rebuilds as another route (Ruby reads the connections; the engine prices the stops).
pub fn checked_run(g: &mut SiteGame, a: &Value) -> Result<Value, String> {
    let corp = sval(a.get("entity"));
    let moves = rules_moves(g);
    let Some((m, _)) = moves.iter().find(|(m, _)| matches!(&m.action, crate::actions::Action::RunRoutes { entity_id, .. } if *entity_id == corp)) else {
        return Err(refused(format!("{corp} may not run trains now")));
    };
    let crate::actions::Action::RunRoutes { routes, .. } = &m.action else { unreachable!() };
    let routes = routes.clone();
    let best = || {
        let total: i32 = routes.iter().map(|r| r.revenue.unwrap_or(0)).sum();
        refused(format!(
            "Only the best run can be posted for {corp} in this title on this site (the auto-router's, {}{total})",
            if super::is_ca(&g.game) { "$" } else { "£" }
        ))
    };
    let Ok(crate::actions::Action::RunRoutes { routes: posted_routes, .. }) = super::super::json::action_from_json(a) else {
        return Err(refused("A run needs its routes"));
    };
    // each posted route against the router's route of its train; the run as posted (its trains' order, each route's
    // connections where Ruby rebuilds them as the router's route), priced by the engine's Ruby pricer in that order
    // (the destination bonus goes to the last of the biggest)
    let mut run: Vec<crate::actions::RouteData> = Vec::new();
    // (each route's stops in the order the train reaches them, for the engine's pricing)
    let mut travel: Vec<Vec<String>> = Vec::new();
    for p in &posted_routes {
        let Some(router) = routes.iter().find(|r| r.train_name == p.train_name) else { return Err(best()) };
        let canon = with_ruby_hexes(router.clone(), &router.hexes);
        let same_stops = if p.nodes.is_empty() {
            // (no stops posted: Ruby takes them from the connections, whose ends are the stops' hexes)
            !p.connections.is_empty() && stop_hexes(p) == node_hexes(&canon)
        } else {
            let (mut ps, mut rs) = (p.nodes.clone(), canon.nodes.clone());
            ps.sort();
            rs.sort();
            ps == rs
        };
        let (stored, stops) = match ruby_form(&p.connections, &canon) {
            Some((mut f, stops)) => {
                // (Ruby keeps a stored route's hexes as stored: its own records list a one-connection route's two
                // stops in the order they were clicked, either way round)
                let posted_hexes = with_ruby_hexes(p.clone(), &f.hexes).hexes;
                if f.connections.len() == 1 && posted_hexes.iter().rev().eq(f.hexes.iter()) {
                    f.hexes = posted_hexes;
                }
                (f, stops)
            }
            None if same_stops => {
                let stops = canon.nodes.clone();
                (canon, stops)
            }
            None => return Err(best()),
        };
        if !p.nodes.is_empty() && !same_stops {
            return Err(best());
        }
        run.push(stored);
        travel.push(stops);
    }
    let (mut theirs, mut mine): (Vec<&str>, Vec<&str>) =
        (run.iter().map(|r| r.train_name.as_str()).collect(), routes.iter().map(|r| r.train_name.as_str()).collect());
    theirs.sort();
    mine.sort();
    if theirs != mine {
        return Err(best());
    }
    // (priced over the stops in turn: the engine's pricer reads its route's nodes in order, as Ruby's reads its stops)
    let in_turn: Vec<crate::actions::RouteData> =
        run.iter().zip(travel.iter()).map(|(r, t)| crate::actions::RouteData { nodes: t.clone(), ..r.clone() }).collect();
    let prices = g.game.g1822_price_recorded_routes(&corp, &in_turn).map_err(|e| refused(e.message))?;
    let run: Vec<crate::actions::RouteData> =
        run.into_iter().zip(prices.iter()).map(|(r, p)| crate::actions::RouteData { revenue: Some(*p), ..r }).collect();
    let dest = destination_bonuses(g, &corp, &run);
    let saw = sawmill_bonuses(g, &corp, &run, &dest);
    let rs: Vec<Value> = run
        .iter()
        .enumerate()
        .map(|(i, r)| {
            let grain = grain_desc(g, &corp, r);
            json!({"train": r.train_name, "connections": r.connections, "hexes": r.hexes, "revenue": r.revenue.unwrap_or(0),
                   "revenue_str": revenue_str(g, &corp, &visited(r), dest[i], saw[i], &grain), "nodes": r.nodes})
        })
        .collect();
    let routes = run;
    // the mail contracts' subsidy (G1822#routes_subsidy) of this run: the round holding its routes, on a copy
    let mut probe = g.game.clone_for_search_without_log();
    probe.rndm().routes = routes
        .iter()
        .map(|r| crate::g1822::RunRoute {
            train_id: r.train_name.clone(),
            revenue: r.revenue.unwrap_or(0),
            subsidy: 0,
            connections: r.connections.clone(),
            hexes: r.hexes.clone(),
            nodes: r.nodes.clone(),
        })
        .collect();
    let subsidy = probe.g1822_routes_subsidy(&corp);
    let mut out = Map::new();
    for k in ["type", "entity", "entity_type", "user", "created_at"] {
        if let Some(v) = a.get(k) {
            out.insert(k.into(), v.clone());
        }
    }
    out.insert("routes".into(), json!(rs));
    out.insert("extra_revenue".into(), json!(0));
    out.insert("subsidy".into(), json!(subsidy));
    Ok(Value::Object(out))
}
