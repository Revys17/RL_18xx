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
    if let Some((_, _, hexes)) = g.laid_hexes.get(sym).and_then(|l| l.iter().find(|x| (x.0, x.1) == (e.turn, e.round))) {
        out["laid"] = json!(hexes);
    }
    if let Some(run) = g.last_runs.get(sym) {
        if (run.turn, run.round) == (e.turn, e.round) && !e.no_run {
            // the routes as Opal's map rebuilds them on today's board (`Engine::Route.new` per train, each excluding the
            // track of those before it): a route that no longer fits the map has no hexes and no track
            let rebuilt = super::route_check::rebuilt_hexes(g, sym, &run.routes);
            let routes: Vec<Value> = run
                .routes
                .iter()
                .zip(rebuilt)
                .map(|(r, hexes)| {
                    let track = if hexes.is_some() { super::track::route_track(g, r) } else { None };
                    json!({"train": r.get("train"), "revenue": r.get("revenue"), "connections": r.get("connections"),
                           "hexes": hexes.unwrap_or_default(), "track": track})
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
        // an unlimited tile (1867's 7, 8, 9): Ruby's supply holds one, and every one upgraded over comes back
        // (`add_extra_tile`, `update_tile_lists`; docs/SITE_ENGINE_RS_STEP3.md §10)
        for (name, left) in counts.iter_mut() {
            if *left >= u32::MAX / 2 {
                *left = g.supply.count(name);
            }
        }
        let total: std::collections::HashMap<&str, u32> = g.game.title_def().tile_counts().into_iter().collect();
        out.insert(
            "tiles".into(),
            json!({"tiles": counts.iter().map(|(k, v)| json!({"name": k, "left": v, "total": total.get(k.as_str()).copied().unwrap_or(0)})).collect::<Vec<_>>()}),
        );
    }
    if want("sheet") {
        let hist = g.game.site.as_ref().map(|r| r.player_history.clone()).unwrap_or_default();
        let liquidity: serde_json::Map<String, Value> = g.players.iter().map(|p| (p.0.to_string(), json!(liquidity(g, p.0)))).collect();
        out.insert(
            "sheet".into(),
            json!({"history": hist.iter().map(|h| json!({"round": h.round, "turn": h.turn, "num": h.round_num,
                "values": h.values.iter().map(|(p, v)| (p.to_string(), json!(v))).collect::<serde_json::Map<_, _>>()})).collect::<Vec<_>>(),
                "liquidity": liquidity}),
        );
    }
    if want("info") {
        out.insert("info".into(), json!({"phase": g.game.phase.name, "endReason": g.end_reason()}));
    }
    Ok(Value::Object(out))
}

/// Ruby `Game#liquidity(player)` (the Spreadsheet's Liquidity column, docs/SITE_ENGINE_RS_STEP3.md §10): the cash
/// alone before 1830's second stock round (`sellable_turn?`); else the cash and, for each company the player holds
/// shares of (1867: one that has operated, `SELL_AFTER :operate`), the dearest bundle Ruby's `bundles_for_corporation`
/// makes that may be dumped and fits in the market (`value_for_dumpable`: the shares by percent with the president's
/// certificate last, each prefix, and the president's partial bundles; a bundle's price `ceil(price * percent / unit)`).
pub fn liquidity(g: &SiteGame, pid: u32) -> i32 {
    use crate::entities::EntityId;
    let game = &g.game;
    let Some(p) = game.players.iter().find(|x| x.id == pid) else { return 0 };
    let is_1867 = game.title == "1867";
    let sellable_turn = is_1867 || game.turn > 1 || !matches!(game.round, Round::Stock(_));
    if !sellable_turn {
        return p.cash;
    }
    let me = EntityId::player(pid);
    let mut value = p.cash;
    for c in &game.corporations {
        let mut shares: Vec<&crate::entities::Share> = c.shares.iter().filter(|s| s.owner == me).collect();
        if shares.is_empty() || c.ipo_price.is_none() || c.closed {
            continue;
        }
        if is_1867 {
            let operated = game.site.as_ref().and_then(|r| r.op_history.get(&c.sym)).is_some_and(|l| !l.is_empty()) || c.ever_operated;
            if !operated {
                continue;
            }
        }
        let Some(price) = c.share_price.as_ref().map(|s| s.price) else { continue };
        shares.sort_by_key(|s| (s.president as u8, s.percent));
        let unit = c.share_unit_percent.max(1) as i64;
        let pool = c.market_shares_percent() as i64;
        let best_other = game.players.iter().filter(|x| x.id != pid).map(|x| c.percent_owned_by(&EntityId::player(x.id)) as i64).max().unwrap_or(0);
        let mut best = 0i64;
        let mut pct = 0i64;
        let mut bundles: Vec<(i64, bool)> = Vec::new();
        for s in &shares {
            pct += s.percent as i64;
            bundles.push((pct, s.president));
        }
        if let Some(last) = shares.last().filter(|s| s.president) {
            let extra = (last.percent as i64 - unit) / unit;
            for n in 1..=extra {
                bundles.push((pct - unit * n, true));
            }
        }
        for (percent, with_pres) in bundles {
            let pres_pct = shares.iter().find(|s| s.president).map_or(0, |s| s.percent as i64);
            let dumpable = !with_pres || best_other >= pres_pct;
            if !dumpable || percent + pool > 50 {
                continue;
            }
            let bundle_price = (price as i64 * percent + unit - 1) / unit;
            best = best.max(bundle_price);
        }
        value += best as i32;
    }
    value
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
