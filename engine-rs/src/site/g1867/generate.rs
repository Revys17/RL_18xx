//! Generated 1867 games (docs/SITE_ENGINE_RS_1867.md §10.4): engine-rs walks under the site's own-only rule
//! (`CrossPlayerPolicy::RESTRICTED`), every move from the generator and posted the way a Rust page posts it
//! (`SiteGame::act`: Ruby's tile instances, the router's run, the predicted step auto actions), written as site JSON
//! for the site's Ruby to judge move by move as `record_action` does (`scripts/site_engine/ruby/live_replay.rb`:
//! `validate_auto_actions: true`). Every Ruby refusal is a Rust or conversion bug, or one of §5.4's known differences.

use super::super::game::SiteGame;
use crate::actions::Action;
use crate::title::CrossPlayerPolicy;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde_json::{json, Value};

/// One generated game: `seed`, `players` (3-6), at most `max_moves` moves; `busy` the share of non-pass picks.
pub fn generate(seed: u64, players: usize, max_moves: usize, busy: f64, ruby_dead_ends: bool) -> Value {
    let mut rng = StdRng::seed_from_u64(seed);
    let ps: Vec<Value> = (1..=players).map(|i| json!({"id": i, "name": format!("Player {i}")})).collect();
    let base = json!({"id": format!("gen-{seed}"), "title": "1867", "description": "generated (engine-rs)", "status": "active",
        "players": ps, "settings": {"optional_rules": []}, "actions": []});
    let mut g = SiteGame::load(&base).expect("1867 loads");
    g.site_log = false;
    let mut refused: Vec<Value> = Vec::new();
    let mut moves_made = 0usize;
    while moves_made < max_moves && !g.finished() {
        let mut moves = g.game.g1867_legal_moves_with(CrossPlayerPolicy::RESTRICTED);
        // Ruby's dead end (docs/g1867_remaining_work.md, SITE_ENGINE_RS_1867.md §5.4): a trainless major nationalized
        // while other majors are queued leaves Ruby's round on its old object, and no later action matches; the walk
        // keeps such a major instead (unless `ruby_dead_ends`)
        if !ruby_dead_ends && g.game.trainless_major.len() > 1 {
            moves.retain(|m| !matches!(&m.action, Action::Choose { choice, .. } if choice == "nationalize"));
        }
        if moves.is_empty() {
            break;
        }
        let acting = super::active_players(&mut g.game);
        // the busy share: prefer a move that isn't a pass
        let non_pass: Vec<usize> = (0..moves.len()).filter(|&i| moves[i].action.action_type() != "pass").collect();
        let mut order: Vec<usize> = if !non_pass.is_empty() && rng.gen_bool(busy) {
            non_pass.clone()
        } else {
            (0..moves.len()).collect()
        };
        // try picks until one is accepted (a refusal of a generated move by act is itself a finding)
        let mut done = false;
        for _ in 0..order.len().min(8) {
            let k = rng.gen_range(0..order.len());
            let i = order.swap_remove(k);
            let m = &moves[i];
            let mut a = if m.action.action_type() == "run_routes" {
                super::menu::auto_route(&mut g)["action"].clone()
            } else {
                g.game.g1867_move_wire(m)
            };
            if a.is_null() {
                continue;
            }
            if let Some(r) = m.price {
                let steps = ((r.hi - r.lo) / r.step.max(1)).max(0);
                // mostly near the low end, sometimes anywhere on the range
                let k = if rng.gen_bool(0.9) { rng.gen_range(0..=steps.min(6)) } else { rng.gen_range(0..=steps) };
                a["price"] = json!(r.lo + k * r.step.max(1));
            }
            if let Action::LayTile { tile_id, .. } = &m.action {
                let name = super::super::json::tile_name(tile_id).unwrap_or(tile_id).to_string();
                if let Some(id) = g.next_tile(&name) {
                    a["tile"] = json!(id);
                }
            }
            match g.act(&a) {
                Ok((mut posted, _)) => {
                    if let Some(p) = acting.first() {
                        posted["user"] = json!(p);
                    }
                    if let Some(last) = g.raw.last_mut() {
                        *last = posted;
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
    out["generated"] = json!({"seed": seed, "players": players, "moves": moves_made, "finished": g.finished(),
        "phase": g.game.phase.name, "refused": refused, "stopped": g.stopped.as_ref().map(|s| s.message.clone())});
    out
}
