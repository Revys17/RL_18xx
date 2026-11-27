//! The details drawers' engine parts and the packages' requests (docs/WASM_ENGINE_SPIKE.md §1 D1-D3, O1-O3):
//! `ports:restart` (restart_info), `ports:last_run` (last_run_info), `ports:meta` (meta), `details`.

use super::game::SiteGame;
use crate::rounds::Round;
use serde_json::{json, Value};

/// `ports:restart` (living-board ports/worker.ts `restartInfo`).
pub fn restart_info(g: &SiteGame) -> Value {
    json!({
        "turnStart": g.turn_start,
        "lastTurnStart": g.last_turn_start,
        "lastGameActionId": g.last_game_action_id,
        "undoPossible": g.last_game_action_id > 0,
        "auction": matches!(g.game.round, Round::Auction(_)),
    })
}

/// `ports:meta` (ports/worker.ts `metaOf`).
pub fn meta(g: &SiteGame) -> Value {
    let mut res = serde_json::Map::new();
    if g.finished() {
        for (k, v) in g.result() {
            res.insert(k.to_string(), json!(v));
        }
    }
    json!({
        "turn": g.game.turn,
        "round": super::view::round_info(g)["name"],
        "acting": if g.game.title == "1867" {
            super::g1867::active_players(&mut g.game.clone_for_search())
        } else {
            super::view::active_players(g)
        },
        "finished": g.finished(),
        "result": res,
        "manually_ended": if g.finished() { json!(g.manually_ended) } else { Value::Null },
        "game_end_reason": g.end_reason(),
    })
}

/// `ports:last_run` (ports/worker.ts `lastRun`): the companies that have operated with their last revenue, and one
/// company's last routes (with tracks) and the hexes it last laid.
pub fn last_run_info(g: &SiteGame, sym: Option<&str>) -> Value {
    use crate::rounds::OperatingStep as S;
    let offered = !matches!(&g.game.round, Round::Operating(s) if s.crowded_corps.is_empty() && matches!(s.step, S::RunRoutes | S::Dividend));
    let mut operators = Vec::new();
    // Ruby operated_operators: the game's corporations (1867: Ruby's @corporations, in its order) that have operated
    let corps: Vec<&crate::entities::Corporation> = if g.game.title == "1867" {
        super::g1867::ruby_corporations(&g.game).into_iter().map(|i| &g.game.corporations[i]).collect()
    } else {
        g.game.corporations.iter().collect()
    };
    if let Some(rec) = g.game.site.as_ref() {
        for c in corps {
            if let Some(e) = rec.op_history.get(&c.sym).and_then(|l| l.iter().max_by_key(|e| (e.turn, e.round))) {
                operators.push(json!({"sym": c.sym, "kind": "corporation", "revenue": e.revenue}));
            }
        }
    }
    let mut out = json!({"offered": offered, "operators": operators, "sym": null, "routes": [], "laid": [], "or": null});
    let Some(sym) = sym else { return out };
    if !offered {
        return out;
    }
    let Some(e) = g.game.site.as_ref().and_then(|r| r.op_history.get(sym)).and_then(|l| l.iter().max_by_key(|e| (e.turn, e.round))) else {
        return out;
    };
    out["sym"] = json!(sym);
    out["or"] = json!(format!("{}.{}", e.turn, e.round));
    if let Some((t, r, hexes)) = g.laid_hexes.get(sym) {
        if (*t, *r) == (e.turn, e.round) {
            out["laid"] = json!(hexes);
        }
    }
    if let Some(run) = g.last_runs.get(sym) {
        if (run.turn, run.round) == (e.turn, e.round) && !e.no_run {
            let routes: Vec<Value> = run
                .routes
                .iter()
                .map(|r| {
                    json!({"train": r.get("train"), "revenue": r.get("revenue"), "connections": r.get("connections"),
                           "hexes": r.get("hexes"), "track": super::track::route_track(g, r)})
                })
                .collect();
            out["routes"] = json!(routes);
        }
    }
    out
}

/// The engine parts of the details drawers (`parts`: ["info", "tiles", "sheet"]).
pub fn details(g: &mut SiteGame, parts: &Value) -> Result<Value, String> {
    let want = |p: &str| parts.as_array().is_some_and(|a| a.iter().any(|x| x.as_str() == Some(p)));
    let mut out = serde_json::Map::new();
    if want("tiles") {
        let mut counts: Vec<(String, u32)> = g.game.tile_counts_remaining.iter().map(|(k, v)| (k.clone(), *v)).collect();
        counts.sort();
        let total: std::collections::HashMap<&str, u32> = g.game.title_def().tile_counts().into_iter().collect();
        out.insert(
            "tiles".into(),
            json!({"tiles": counts.iter().map(|(k, v)| json!({"name": k, "left": v, "total": total.get(k.as_str()).copied().unwrap_or(0)})).collect::<Vec<_>>()}),
        );
    }
    if want("sheet") {
        let hist = g.game.site.as_ref().map(|r| r.player_history.clone()).unwrap_or_default();
        out.insert(
            "sheet".into(),
            json!({"history": hist.iter().map(|h| json!({"round": h.round, "turn": h.turn, "num": h.round_num,
                "values": h.values.iter().map(|(p, v)| (p.to_string(), json!(v))).collect::<serde_json::Map<_, _>>()})).collect::<Vec<_>>()}),
        );
    }
    if want("info") {
        out.insert("info".into(), json!({"phase": g.game.phase.name, "endReason": g.end_reason()}));
    }
    Ok(Value::Object(out))
}

/// The Auto tab's facts (living-board moves.ts `programInfo` / `ProgramInfo`): the program types 1830 offers (Ruby
/// `available_programmed_actions`: ProgramBuyShares, ProgramSharePass), whether several may run at once
/// (ALLOW_MULTIPLE_PROGRAMS false), whether stock passes are always unconditional (false), whether a stock round is
/// on, the player's own programs (from the recorded `program_*` actions) and, for "buy until", the floated or
/// parred corporations with shares in the IPO or the market (counts of certificates; `owned` in shares).
pub fn program_info(g: &SiteGame, pid: u32) -> Value {
    use crate::entities::EntityId;
    if g.finished() || !g.game.players.iter().any(|p| p.id == pid) {
        return Value::Null;
    }
    let pe = EntityId::player(pid);
    let mut buyable = Vec::new();
    for c in g.game.corporations.iter().filter(|c| !c.closed && c.ipo_price.is_some()) {
        let ipo = c.shares.iter().filter(|s| !s.owner.is_player() && !s.owner.is_market()).count();
        let market = c.shares.iter().filter(|s| s.owner.is_market()).count();
        if ipo == 0 && market == 0 {
            continue;
        }
        let owned_pct: u32 = c.shares.iter().filter(|s| s.owner == pe).map(|s| s.percent as u32).sum();
        let held: u32 = c.shares.iter().filter(|s| s.owner.is_player() || s.owner.is_market()).map(|s| s.percent as u32).sum();
        let ipo_pct = 100u32.saturating_sub(held) as i64;
        let to_float = if c.floated { 0 } else { ipo_pct - (100 - c.float_percent as i64) };
        // Ruby num_shares_of(corp, ceil: false): percent / share_percent (1830: 10%)
        let owned = owned_pct as f64 / 10.0;
        buyable.push(json!({"corp": c.sym, "floated": c.floated, "toFloat": to_float, "ipo": ipo, "market": market,
                            "owned": if owned.fract() == 0.0 { json!(owned as i64) } else { json!(owned) }}));
    }
    let mine = super::view::programs(g).and_then(|p| p.get(pid.to_string()).cloned()).unwrap_or(json!([]));
    // Ruby available_programmed_actions (1867: ProgramMergerPass too)
    let available = if g.game.title == "1867" {
        json!(["program_merger_pass", "program_buy_shares", "program_share_pass"])
    } else {
        json!(["program_buy_shares", "program_share_pass"])
    };
    json!({
        "available": available,
        "multiple": false,
        "forceUnconditional": false,
        "stock": matches!(g.game.round, Round::Stock(_)),
        "stockName": "Stock Round",
        "mine": mine,
        "buyable": buyable,
    })
}
