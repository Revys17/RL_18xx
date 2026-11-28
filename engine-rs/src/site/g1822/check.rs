//! The RL session's binding check (docs/SITE_ENGINE_RS_1822.md §3.4): at every position of a game, the site's menus
//! (the players to act, own-only, no fallback) against engine-rs's RESTRICTED legal set (`g1822_legal_moves_with`), in
//! both directions; the generator's fallback to Ruby's set (when the restricted one is empty) counted apart. Also:
//! every fact's action is one of the menu's moves, every move has a fact (a lay through `tileHexes` / a private's
//! `abilities`, a run through `routes.pending`), and the key, the snapshot and the bot view build everywhere.

use super::super::game::SiteGame;
use crate::title::CrossPlayerPolicy;
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};

/// A move's site action as a comparable string.
pub fn canon(v: &Value) -> String {
    let mut v = v.clone();
    if let Some(o) = v.as_object_mut() {
        o.remove("created_at");
        o.remove("user");
    }
    v.to_string()
}

/// The actions every fact of a menu names (outside the flat `moves` list).
fn fact_actions(v: &Value, out: &mut BTreeSet<String>, top: bool) {
    match v {
        Value::Object(m) => {
            if let Some(a) = m.get("action").filter(|a| a.get("type").is_some()) {
                out.insert(canon(a));
            }
            for (k, x) in m {
                if top && k == "moves" {
                    continue;
                }
                if k != "action" {
                    fact_actions(x, out, false);
                }
            }
        }
        Value::Array(xs) => xs.iter().for_each(|x| fact_actions(x, out, false)),
        _ => {}
    }
}

/// The counts of one game (or several, added): positions, equal, fallback, the differences (a few examples each).
#[derive(Default, Debug)]
pub struct Report {
    pub positions: usize,
    pub equal: usize,
    pub fallback: usize,
    pub menu_only: Vec<String>,
    pub engine_only: Vec<String>,
    pub foreign: Vec<String>,
    pub uncovered: BTreeMap<String, usize>,
}

impl Report {
    pub fn json(&self) -> Value {
        json!({"positions": self.positions, "equal": self.equal, "fallback": self.fallback,
               "menuOnly": self.menu_only.len(), "engineOnly": self.engine_only.len(), "factsOutsideMoves": self.foreign.len(),
               "movesWithoutFact": self.uncovered, "examples": self.menu_only.iter().chain(self.engine_only.iter()).chain(self.foreign.iter()).take(10).collect::<Vec<_>>()})
    }
}

/// Run the check over a game's whole replay (`name` labels the examples).
pub fn check_game(name: &str, data: &Value, r: &mut Report) -> Result<(), String> {
    let mut g = SiteGame::load(data)?;
    g.site_log = false;
    loop {
        if !g.finished() {
            r.positions += 1;
            let before = crate::g1822::legal::stats::get(&crate::g1822::legal::stats::CROSS_FALLBACK);
            let moves = g.game.g1822_legal_moves_with(CrossPlayerPolicy::RESTRICTED);
            let fell_back = crate::g1822::legal::stats::get(&crate::g1822::legal::stats::CROSS_FALLBACK) > before;
            let engine: BTreeSet<String> = moves.iter().map(|m| canon(&super::to_site(&g, &g.game.g1822_move_wire(m)))).collect();
            let mut site: BTreeSet<String> = BTreeSet::new();
            for pid in super::active_players(&g) {
                let m = super::menu::menu(&mut g, pid);
                let listed: BTreeSet<String> =
                    m.get("moves").and_then(Value::as_array).map(|v| v.iter().map(|x| canon(&x["action"])).collect()).unwrap_or_default();
                let mut facts = BTreeSet::new();
                fact_actions(&m, &mut facts, true);
                for f in facts.difference(&listed) {
                    r.foreign.push(format!("{name} @{}: {f}", g.current_action_id));
                }
                let mut lay_hexes: BTreeSet<String> = BTreeSet::new();
                for h in m.get("tileHexes").and_then(Value::as_array).into_iter().flatten() {
                    lay_hexes.extend(h.as_str().map(str::to_string));
                }
                for a in m.get("abilities").and_then(Value::as_array).into_iter().flatten() {
                    for h in a["hexes"].as_array().into_iter().flatten() {
                        lay_hexes.extend(h.as_str().map(str::to_string));
                    }
                }
                for x in m.get("moves").and_then(Value::as_array).cloned().unwrap_or_default() {
                    let covered = match x["type"].as_str() {
                        Some("lay_tile") => x["action"]["hex"].as_str().is_some_and(|h| lay_hexes.contains(h)),
                        Some("run_routes") => m["routes"]["pending"] == json!(true),
                        _ => false,
                    };
                    if !covered && !facts.contains(&canon(&x["action"])) {
                        *r.uncovered.entry(x["type"].as_str().unwrap_or("").to_string()).or_default() += 1;
                    }
                }
                site.extend(listed);
            }
            if site == engine {
                r.equal += 1;
            } else if fell_back {
                r.fallback += 1;
            } else {
                for x in site.difference(&engine).take(3) {
                    r.menu_only.push(format!("{name} @{}: {x}", g.current_action_id));
                }
                for x in engine.difference(&site).take(3) {
                    r.engine_only.push(format!("{name} @{}: {x}", g.current_action_id));
                }
            }
            let _ = super::key::key(&mut g, None);
            let _ = super::view::snapshot(&g, g.current_action_id);
            let _ = super::super::authority::bot_view(&mut g);
        }
        if g.done() {
            break;
        }
        while !g.done() && g.process_next().is_none() {}
    }
    Ok(())
}
