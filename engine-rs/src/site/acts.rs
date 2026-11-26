//! Act-level cases (docs/SITE_ENGINE_RS.md §8.3, contract K11): at sampled decision positions of a game, every
//! option the Rust menu offers each player who must act -- the RawMenu's complete actions, a bid's minimum and
//! maximum, the tile options of the hexes it lists, the auto-router's run -- acted in Rust on a copy: accepted (the
//! posted site action and the shadow key after) or refused (the engine's words). B's `ruby_verdicts.rb` runs each
//! case through Ruby as `record_action` does, C through Opal; every option accepted by one and refused by another is a
//! disagreement.
//!
//! One JSON line per case: `{game_id, upto, player, action, rust: {verdict, posted?, error?, key?}}` (`upto`: the
//! last raw action id of the position; the case's game is the corpus file's actions up to it).

use super::game::SiteGame;
use serde_json::{json, Value};

/// Which positions get cases: every `every`-th decision position, tile options at every `tile_every`-th of those.
pub struct Sampling {
    pub every: usize,
    pub tile_every: usize,
}

fn bid_actions(menu: &Value, pid: u32) -> Vec<Value> {
    let mut out = Vec::new();
    for lot in menu.get("auction").and_then(Value::as_array).cloned().unwrap_or_default() {
        let sym = lot["sym"].clone();
        let (lo, hi) = (lot["minBid"].as_i64().unwrap_or(0), lot["maxBid"].as_i64().unwrap_or(0));
        for price in [lo, hi] {
            let a = json!({"type": "bid", "entity": pid, "entity_type": "player", "company": sym, "price": price});
            if !out.contains(&a) {
                out.push(a);
            }
        }
        if lot["buy"].as_bool() == Some(true) {
            let value = lot["value"].as_i64().unwrap_or(0);
            let a = json!({"type": "bid", "entity": pid, "entity_type": "player", "company": sym, "price": value});
            if !out.contains(&a) {
                out.push(a);
            }
        }
    }
    out
}

fn collect(v: &Value, out: &mut Vec<Value>) {
    match v {
        Value::Object(m) => {
            if let Some(a) = m.get("action") {
                // an option priced by the player (another company's train, a private): its minimum and maximum
                let priced = a.get("price").is_none() && m.get("min").is_some_and(Value::is_i64) && m.get("max").is_some_and(Value::is_i64);
                let list: Vec<Value> = if priced {
                    let mut v = Vec::new();
                    for k in ["min", "max"] {
                        let mut x = a.clone();
                        x["price"] = m[k].clone();
                        if !v.contains(&x) {
                            v.push(x);
                        }
                    }
                    v
                } else {
                    vec![a.clone()]
                };
                for a in list {
                    if a.get("type").is_some() && !out.contains(&a) {
                        out.push(a);
                    }
                }
            }
            for (k, x) in m {
                if k != "action" {
                    collect(x, out);
                }
            }
        }
        Value::Array(xs) => xs.iter().for_each(|x| collect(x, out)),
        _ => {}
    }
}

/// The options of the position now, per acting player.
pub fn options(g: &mut SiteGame, with_tiles: bool) -> Vec<(u32, Value)> {
    let mut out = Vec::new();
    for pid in super::view::active_players(g) {
        let m = super::menu::menu(g, pid);
        if m.is_null() {
            continue;
        }
        let mut acts = Vec::new();
        collect(&m, &mut acts);
        acts.extend(bid_actions(&m, pid));
        if with_tiles {
            for hex in m.get("tileHexes").and_then(Value::as_array).cloned().unwrap_or_default() {
                if let Some(h) = hex.as_str() {
                    if let Ok(opts) = super::menu::tile_options(g, h, None) {
                        collect(&opts, &mut acts);
                    }
                }
            }
        }
        if m.get("routes").is_some() {
            let r = super::menu::auto_route(g);
            if let Some(a) = r.get("action") {
                acts.push(a.clone());
            }
        }
        for a in acts {
            out.push((pid, a));
        }
    }
    out
}

/// One option acted on a copy of the position.
pub fn verdict(g: &SiteGame, action: &Value) -> Value {
    let mut copy = g.clone();
    copy.site_log = false;
    // the position as the server holds it at `upto`: no later raw actions
    copy.raw.truncate(copy.pos);
    copy.kept.truncate(copy.pos);
    match copy.act(action) {
        Ok((posted, _)) => {
            let key = super::shadow::key_now(&mut copy);
            json!({"verdict": "accepted", "posted": posted, "key": key})
        }
        Err(e) => json!({"verdict": if e.starts_with("refused:") { "refused" } else { "error" }, "error": e}),
    }
}

/// Every case of one game, as JSON lines.
pub fn game_cases(data: &Value, game_id: &str, s: &Sampling, out: &mut dyn FnMut(Value)) -> Result<usize, String> {
    let mut g = SiteGame::load(data)?;
    g.site_log = false;
    let mut decision = 0usize;
    let mut n = 0usize;
    loop {
        if g.stopped.is_some() || g.finished() {
            break;
        }
        if !super::view::active_players(&g).is_empty() {
            if decision % s.every.max(1) == 0 {
                let with_tiles = (decision / s.every.max(1)) % s.tile_every.max(1) == 0;
                for (pid, action) in options(&mut g, with_tiles) {
                    let rust = verdict(&g, &action);
                    out(json!({"game_id": game_id, "upto": g.current_action_id, "player": pid, "action": action, "rust": rust}));
                    n += 1;
                }
            }
            decision += 1;
        }
        // to the next kept position
        let mut stepped = false;
        while !g.done() {
            if g.process_next().is_some() {
                stepped = true;
                break;
            }
        }
        if !stepped {
            break;
        }
    }
    Ok(n)
}
