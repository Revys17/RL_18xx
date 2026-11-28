//! Generated 1822-family games (docs/SITE_ENGINE_RS_1822.md §6.1 A2): walks through the site layer as players at the
//! site's ids (1..n: colliding with the minors' syms on purpose), every move taken from the site's own menu
//! (`CrossPlayerPolicy::RESTRICTED`, what a Rust page offers) or from Ruby's whole set (`RULES`: cross-player deals a
//! bot or an Opal page could post), posted the way a Rust page posts it (`SiteGame::act`: the authority's check, the
//! router's run, the step auto actions and programs made here) and written as site JSON, for the site's Ruby to judge
//! move by move and for the replay check: the written game must replay to the same key.

use super::super::game::SiteGame;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde_json::{json, Value};

/// The optional-rule combinations of 1822 (1822CA: none).
pub const RULES_1822: [&[&str]; 6] = [
    &[],
    &["plus_expansion"],
    &["plus_expansion", "plus_expansion_no_removals"],
    &["plus_expansion", "plus_expansion_single_stack"],
    &["plus_expansion", "plus_expansion_no_removals", "plus_expansion_single_stack", "tax_haven_multiple"],
    &["tax_haven_multiple"],
];

/// One generated game: `title`, `players` (2-7), `rules`, `seed`; at most `max_moves` moves; `busy` the share of
/// picks that aren't a pass; `rules_policy`: pick from Ruby's set (else the site's menus).
pub fn generate(title: &str, players: usize, rules: &[&str], seed: u64, max_moves: usize, busy: f64, rules_policy: bool) -> Value {
    let mut rng = StdRng::seed_from_u64(seed);
    let ps: Vec<Value> = (1..=players).map(|i| json!({"id": i, "name": format!("Player {i}")})).collect();
    let base = json!({"id": format!("gen-{seed}"), "title": title, "description": "generated (engine-rs)", "status": "active",
        "players": ps, "settings": {"optional_rules": rules, "seed": seed}, "actions": []});
    let mut g = SiteGame::load(&base).expect("the family's game loads");
    g.site_log = false;
    let mut refused: Vec<Value> = Vec::new();
    let mut dead_end: Option<String> = None;
    let mut moves_made = 0usize;
    let mut cross = 0usize;
    let mut now = 1_700_000_000i64;
    while moves_made < max_moves && !g.finished() {
        let acting = super::active_players(&g);
        // the moves on offer: every acting player's menu, or Ruby's set
        let mut offered: Vec<(Value, Option<(i32, i32, i32)>, bool)> = Vec::new();
        if rules_policy {
            let (own, others) = super::menu::offers(&mut g);
            for o in own {
                offered.push((o.action.clone(), o.m.price.map(|r| (r.lo, r.hi, r.step)), false));
            }
            for o in others {
                offered.push((o.action.clone(), o.m.price.map(|r| (r.lo, r.hi, r.step)), true));
            }
        } else {
            for pid in &acting {
                let m = super::menu::menu(&mut g, *pid);
                for x in m.get("moves").and_then(Value::as_array).cloned().unwrap_or_default() {
                    let r = x.get("range").filter(|r| !r.is_null()).map(|r| {
                        (r["min"].as_i64().unwrap_or(0) as i32, r["max"].as_i64().unwrap_or(0) as i32, r["step"].as_i64().unwrap_or(1) as i32)
                    });
                    offered.push((x["action"].clone(), r, false));
                }
            }
        }
        if offered.is_empty() {
            dead_end = Some(super::act::dead_end(&mut g).unwrap_or_else(|| "no move".into()));
            break;
        }
        let non_pass: Vec<usize> = (0..offered.len()).filter(|&i| offered[i].0["type"] != "pass").collect();
        let mut order: Vec<usize> = if !non_pass.is_empty() && rng.gen_bool(busy) { non_pass } else { (0..offered.len()).collect() };
        let mut done = false;
        for _ in 0..order.len().min(8) {
            let k = rng.gen_range(0..order.len());
            let i = order.swap_remove(k);
            let (mut a, range, is_cross) = offered[i].clone();
            if a["type"] == "run_routes" {
                a = super::menu::auto_route(&mut g)["action"].clone();
                if a.is_null() {
                    continue;
                }
            }
            if let Some((lo, hi, step)) = range {
                let steps = ((hi - lo) / step.max(1)).max(0);
                // mostly near the low end, sometimes anywhere on the range
                let k = if rng.gen_bool(0.85) { rng.gen_range(0..=steps.min(4)) } else { rng.gen_range(0..=steps) };
                let p = lo + k * step.max(1);
                match a["type"].as_str() {
                    Some("payoff_player_debt_partial") => a["amount"] = json!(p),
                    _ => a["price"] = json!(p),
                }
            }
            now += 7;
            a["created_at"] = json!(now);
            match g.act(&a) {
                Ok((mut posted, _)) => {
                    if let Some(p) = acting.first() {
                        posted["user"] = json!(p);
                    }
                    if let Some(last) = g.raw.last_mut() {
                        *last = posted;
                    }
                    if is_cross {
                        cross += 1;
                    }
                    done = true;
                    break;
                }
                Err(e) => {
                    if refused.len() < 20 {
                        refused.push(json!({"at": g.current_action_id, "action": a, "error": e}));
                    }
                }
            }
        }
        if !done {
            break;
        }
        moves_made += 1;
    }
    let mut out = g.meta.clone();
    out["actions"] = json!(g.raw);
    out["status"] = json!(if g.finished() { "finished" } else { "active" });
    out["generated"] = json!({"title": title, "seed": seed, "players": players, "rules": rules, "policy": if rules_policy { "rules" } else { "restricted" },
        "moves": moves_made, "cross": cross, "finished": g.finished(), "phase": g.game.phase.name, "refused": refused, "deadEnd": dead_end,
        "endReason": g.end_reason(), "stopped": g.stopped.as_ref().map(|s| s.message.clone()),
        "key": super::super::shadow::key_now(&mut g)});
    out
}

/// A generated game replayed from its JSON: whether it ends where the walk ended (the same key).
pub fn replays(game: &Value) -> Result<(), String> {
    let mut g = SiteGame::load(game)?;
    g.site_log = false;
    g.run_to(None);
    if let Some(s) = &g.stopped {
        return Err(format!("stopped at {}: {}", s.action_id, s.message));
    }
    let key = super::super::shadow::key_now(&mut g);
    if key != game["generated"]["key"] {
        return Err("the replayed key differs from the walk's".into());
    }
    Ok(())
}
