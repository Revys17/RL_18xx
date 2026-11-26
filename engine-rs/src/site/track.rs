//! A run's track as the board draws it (types.ts `RunTrack`; adapter.ts `runTrack` reads it from Ruby's
//! `Route#connection_data`): the stops in the order the train reaches them and, for each leg between two stops, the
//! tile paths it takes, each from its end nearer the leg's start. Rebuilt here from the run's recorded `connections`
//! (one hex chain per leg, node to node) on the board as it is now: in each hex of a chain the path that joins the
//! hexside the chain crosses (or the stop it starts or ends at).

use super::game::SiteGame;
use crate::tiles::PathEndpoint as E;
use serde_json::{json, Value};

/// A path end in the board's shape (Ruby's city numbering: G19 swapped while preprinted).
fn end_json(g: &SiteGame, hex: &str, e: &E) -> Value {
    match e {
        E::Edge(n) => json!({"t": "e", "n": n}),
        E::City(i) => {
            let swapped = g.swapped.contains(&hex) && g.game.hex_ref(hex).is_some_and(|h| h.tile.cities.len() == 2);
            json!({"t": "c", "i": if swapped { 1 - *i } else { *i }})
        }
        E::Town(i) => json!({"t": "t", "i": i}),
        E::Offboard(i) => json!({"t": "o", "i": i}),
        E::Junction => json!({"t": "j"}),
    }
}

/// The hexside of `from` that faces `to`.
fn edge_between(g: &SiteGame, from: &str, to: &str) -> Option<u8> {
    g.game.hex_adjacency.get(from)?.iter().find(|(_, h)| h.as_str() == to).map(|(e, _)| *e)
}

/// The paths of hex `hex` that take a train from end `start` (an edge or a node) to `goal` (an edge; None: any
/// node), in order, each oriented from `start` towards `goal`. A junction may join two paths.
fn walk(g: &SiteGame, hex: &str, start: &E, goal: Option<u8>) -> Option<Vec<(E, E)>> {
    let tile = &g.game.hex_ref(hex)?.tile;
    let ends_ok = |b: &E| match goal {
        Some(n) => *b == E::Edge(n),
        None => b.is_node(),
    };
    // one path
    for p in &tile.paths {
        for (a, b) in [(&p.a, &p.b), (&p.b, &p.a)] {
            if a == start && ends_ok(b) {
                return Some(vec![(a.clone(), b.clone())]);
            }
        }
    }
    // two paths through a junction
    for p in &tile.paths {
        for (a, b) in [(&p.a, &p.b), (&p.b, &p.a)] {
            if a == start && *b == E::Junction {
                for q in &tile.paths {
                    for (c, d) in [(&q.a, &q.b), (&q.b, &q.a)] {
                        if *c == E::Junction && !std::ptr::eq(p, q) && ends_ok(d) {
                            return Some(vec![(a.clone(), b.clone()), (c.clone(), d.clone())]);
                        }
                    }
                }
            }
        }
    }
    None
}

/// One chain (node to node): its legs as `(hex, a, b)` and the nodes it starts and ends at.
fn chain_paths(g: &SiteGame, chain: &[String], start_node: Option<&E>) -> Option<(Vec<(String, E, E)>, E, E)> {
    if chain.len() < 2 {
        return None;
    }
    let mut legs = Vec::new();
    let k = chain.len();
    // first hex: from a node to the hexside facing the next hex
    let out0 = edge_between(g, &chain[0], &chain[1])?;
    let tile0 = &g.game.hex_ref(&chain[0])?.tile;
    let mut first_node = None;
    for p in &tile0.paths {
        for (a, b) in [(&p.a, &p.b), (&p.b, &p.a)] {
            if a.is_node() && *b == E::Edge(out0) && start_node.map_or(true, |s| s == a) {
                first_node = Some(a.clone());
            }
        }
    }
    let first_node = match first_node {
        Some(n) => n,
        None => {
            // through a junction (a node's path to a junction, the junction's to the edge)
            let mut found = None;
            for p in &tile0.paths {
                for (a, b) in [(&p.a, &p.b), (&p.b, &p.a)] {
                    if a.is_node() && *b == E::Junction && start_node.map_or(true, |s| s == a) {
                        found = Some(a.clone());
                    }
                }
            }
            found?
        }
    };
    for (a, b) in walk(g, &chain[0], &first_node, Some(out0))? {
        legs.push((chain[0].clone(), a, b));
    }
    for i in 1..k - 1 {
        let inn = edge_between(g, &chain[i], &chain[i - 1])?;
        let out = edge_between(g, &chain[i], &chain[i + 1])?;
        for (a, b) in walk(g, &chain[i], &E::Edge(inn), Some(out))? {
            legs.push((chain[i].clone(), a, b));
        }
    }
    let inn = edge_between(g, &chain[k - 1], &chain[k - 2])?;
    let last = walk(g, &chain[k - 1], &E::Edge(inn), None)?;
    let last_node = last.last()?.1.clone();
    for (a, b) in last {
        legs.push((chain[k - 1].clone(), a, b));
    }
    Some((legs, first_node, last_node))
}

/// The track of a run's route (its site JSON: `connections`), or None when it can't be rebuilt.
pub fn route_track(g: &SiteGame, route: &Value) -> Option<Value> {
    let chains: Vec<Vec<String>> = route
        .get("connections")?
        .as_array()?
        .iter()
        .map(|c| c.as_array().map(|xs| xs.iter().filter_map(|x| x.as_str().map(str::to_string)).collect()).unwrap_or_default())
        .collect();
    if chains.is_empty() {
        return None;
    }
    if chains.len() == 1 && chains[0].first().map(String::as_str) == Some("local") {
        let hex = chains[0].get(1)?;
        let tile = &g.game.hex_ref(hex)?.tile;
        let end = if !tile.cities.is_empty() { E::City(0) } else if !tile.towns.is_empty() { E::Town(0) } else { E::Offboard(0) };
        return Some(json!({"stops": [{"hex": hex, "end": end_json(g, hex, &end)}], "legs": []}));
    }
    // orient the chains: each starts where the one before ended
    let mut oriented: Vec<Vec<String>> = Vec::new();
    for (i, c) in chains.iter().enumerate() {
        let mut c = c.clone();
        if i == 0 {
            if let Some(next) = chains.get(1) {
                let (a, b) = (c.first()?.clone(), c.last()?.clone());
                let touches = |h: &String| next.first() == Some(h) || next.last() == Some(h);
                if touches(&a) && !touches(&b) {
                    c.reverse();
                }
            }
        } else {
            let prev_end = oriented.last()?.last()?.clone();
            if c.first() != Some(&prev_end) && c.last() == Some(&prev_end) {
                c.reverse();
            }
        }
        oriented.push(c);
    }
    let mut stops = Vec::new();
    let mut legs = Vec::new();
    let mut prev_node: Option<E> = None;
    for (i, c) in oriented.iter().enumerate() {
        let (paths, first, last) = chain_paths(g, c, if i == 0 { None } else { prev_node.as_ref() })?;
        if i == 0 {
            stops.push(json!({"hex": c[0], "end": end_json(g, &c[0], &first)}));
        }
        let last_hex = c.last()?.clone();
        stops.push(json!({"hex": last_hex, "end": end_json(g, &last_hex, &last)}));
        legs.push(json!(paths.iter().map(|(h, a, b)| json!({"hex": h, "a": end_json(g, h, a), "b": end_json(g, h, b)})).collect::<Vec<_>>()));
        prev_node = Some(last);
    }
    Some(json!({"stops": stops, "legs": legs}))
}

/// Ruby `Route#visited_hexes` (rl18xx-next c57803b): the hexes of the route's stops in the order the train reaches
/// them, one per stop (both cities of an OO hex: the hex twice), from the track, "-"-joined.
pub fn visited_hexes(g: &SiteGame, route: &Value) -> Option<String> {
    let t = route_track(g, route)?;
    let mut seen: Vec<(String, String)> = Vec::new();
    for s in t.get("stops")?.as_array()? {
        let node = (s.get("hex")?.as_str()?.to_string(), s.get("end")?.to_string());
        // `.uniq` over the nodes: a stop reached again isn't listed again
        if !seen.contains(&node) {
            seen.push(node);
        }
    }
    Some(seen.into_iter().map(|n| n.0).collect::<Vec<_>>().join("-"))
}
