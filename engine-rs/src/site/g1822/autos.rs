//! Ruby's step auto actions in the 1822 family (docs/SITE_ENGINE_RS_1822.md §3.5): after every move Ruby's
//! `process_action` runs `round.auto_actions` (the active step's `auto_actions(current_entity)`) until there are none.
//!
//! | Ruby | Here |
//! |---|---|
//! | `G1822::Step::DestinationToken#auto_actions` (the token when connected, else pass; 1822CA: none when ICR reaches several Québec cities) | the blocking step's own generated moves: exactly one is that move |
//! | `G1822::Step::MinorAcquisition#auto_actions` (pass when nothing is mergeable, outside the pay / token choices) | the step's moves are only its pass |
//! | `G1822::Step::Choose#auto_actions` (P9's pass when doubling can't change the player order) | Ruby's cash rule |
//! | `BuySellParShares`' Programmer (`program_share_pass`, `program_buy_shares`) | `stock_programs` |
//!
//! The predictor equals Ruby's at every position of every site fixture but the old records of §2.3 G4.

use super::super::game::SiteGame;
use crate::g1822::{AcquireState, Ent, RoundKind, Step};
use crate::title::CrossPlayerPolicy;
use serde_json::{json, Value};

fn pass(entity: Value, entity_type: &str, now: i64) -> Value {
    json!({"type": "pass", "entity": entity, "entity_type": entity_type, "created_at": now})
}

/// The blocking step's own moves for its entity (the generator's, Ruby's set): the parallel companies' left out.
fn own_moves(g: &mut SiteGame, ent: &str) -> Vec<Value> {
    let moves = g.game.g1822_legal_moves_with(CrossPlayerPolicy::RULES);
    moves
        .iter()
        .map(|m| g.game.g1822_move_wire(m))
        .filter(|w| w.get("entity_type").and_then(Value::as_str) != Some("company"))
        .filter(|w| w.get("entity").map(|e| e.as_str().map(str::to_string).unwrap_or_else(|| e.to_string())).as_deref() == Some(ent))
        .collect()
}

/// Ruby's `round.auto_actions` now (empty: none).
pub fn auto_actions(g: &mut SiteGame, now: i64) -> Vec<Value> {
    let Some((_, step, ent)) = super::active(&g.game) else { return Vec::new() };
    match (step, &ent) {
        (Step::DestinationToken, Ent::Corp(c)) => {
            let c = c.clone();
            let moves = own_moves(g, &c);
            if moves.len() == 1 {
                let mut a = super::to_site(g, &moves[0]);
                // 1822CA's ICR in one of several Québec cities: Ruby's PlaceToken names `slot: city.get_slot(entity)`,
                // nil (left out of the stored action) when the city has no free slot and no reservation of its own (the
                // destination token then goes in as an extra one); the generator's wire names the slot it will take
                let slotless = a.get("type").and_then(Value::as_str) == Some("place_token")
                    && match super::super::json::action_from_json(&a) {
                        Ok(crate::actions::Action::PlaceToken { hex_id, city_index, .. }) => {
                            let hex = g.game.g1822_resolve_hex(&hex_id).unwrap_or(hex_id);
                            g.game.hex_idx.get(hex.as_str()).is_some_and(|&hi| g.game.g1822_get_slot(hi, city_index as usize, &c, false).is_none())
                        }
                        _ => false,
                    };
                if let Some(m) = a.as_object_mut() {
                    if slotless {
                        m.remove("slot");
                    }
                    m.insert("created_at".into(), json!(now));
                }
                vec![a]
            } else {
                Vec::new()
            }
        }
        (Step::MinorAcquisition, Ent::Corp(c)) => {
            if g.game.rnd().acquire_state != AcquireState::SelectMinor {
                return Vec::new();
            }
            let c = c.clone();
            let moves = own_moves(g, &c);
            if !moves.is_empty() && moves.iter().all(|w| w.get("type").and_then(Value::as_str) == Some("pass")) {
                vec![pass(json!(c), "corporation", now)]
            } else {
                Vec::new()
            }
        }
        (Step::Choose, Ent::Player(p)) if g.game.rnd().kind == RoundKind::Choices => {
            // find_company: P9 (CA P7) held by the player
            let dc = g.game.fam().company_double_cash;
            if dc.is_empty() || g.game.co_owner(dc).player_id() != Some(*p) || g.game.co_closed(dc) {
                return Vec::new();
            }
            // `players.max_by(&:cash) == entity || entity.cash * 2 < players.map(&:cash).sort[1]` (max_by: the first)
            let cash: Vec<(u32, i32)> = g.game.players.iter().map(|x| (x.id, x.cash)).collect();
            let mut richest: Option<(u32, i32)> = None;
            for &(id, c) in &cash {
                if richest.is_none_or(|(_, b)| c > b) {
                    richest = Some((id, c));
                }
            }
            let me = cash.iter().find(|x| x.0 == *p).map_or(0, |x| x.1);
            let mut sorted: Vec<i32> = cash.iter().map(|x| x.1).collect();
            sorted.sort();
            let second = sorted.get(1).copied().unwrap_or(i32::MAX);
            if richest.map(|r| r.0) == Some(*p) || me * 2 < second {
                vec![pass(json!(g.site_id(*p)), "player", now)]
            } else {
                Vec::new()
            }
        }
        (Step::BuySellParShares, Ent::Player(p)) => {
            let site = g.site_id(*p);
            stock_programs(g, site, now)
        }
        _ => Vec::new(),
    }
}

/// The current player's program's auto actions in the stock round (BuySellParShares' Programmer): Ruby's
/// `activate_program_share_pass` / `activate_program_buy_shares` on G1822's step.
fn stock_programs(g: &mut SiteGame, pid: u32, now: i64) -> Vec<Value> {
    let Some(list) = g.programs.get(&pid).cloned() else { return Vec::new() };
    let seat = g.seat(pid);
    let mut out = Vec::new();
    let acts = super::ruby_actions(&g.game);
    for prog in list {
        match prog.get("type").and_then(Value::as_str) {
            Some("program_share_pass") => {
                // normal_pass? (G1822: always), the step's actions include pass
                if !acts.iter().any(|t| t == "pass") {
                    continue;
                }
                let unconditional = prog.get("unconditional").and_then(Value::as_bool).unwrap_or(false);
                let reason = if unconditional { None } else { super::super::programs::should_stop_1822(g, pid, None, &acts) };
                match reason {
                    Some(r) => out.push(json!({"type": "program_disable", "entity": pid, "entity_type": "player", "created_at": now, "reason": r})),
                    None => out.push(pass(json!(pid), "player", now)),
                }
            }
            Some("program_buy_shares") => {
                let corp = prog.get("corporation").and_then(Value::as_str).unwrap_or("").to_string();
                let Some(&ci) = g.game.corp_idx.get(corp.as_str()) else { continue };
                let until = prog.get("until_condition").cloned().unwrap_or(Value::Null);
                let c = &g.game.corporations[ci];
                let held: u32 = c.shares.iter().filter(|x| x.owner.is_player() || x.owner.is_market()).map(|x| x.percent as u32).sum();
                let to_float = c.float_percent as i64 - held as i64;
                let finished = if until.as_str() == Some("float") {
                    (c.floated || to_float <= 0).then(|| format!("{} is floated", c.sym))
                } else {
                    let n = until.as_f64().or_else(|| until.as_str().and_then(|x| x.parse().ok())).unwrap_or(f64::MAX);
                    let mine = c.percent_owned_by(&crate::entities::EntityId::player(seat)) as f64 / c.share_unit_percent.max(1) as f64;
                    let us = match &until {
                        Value::String(x) => x.clone(),
                        o => o.to_string(),
                    };
                    (mine >= n).then(|| format!("{us} share(s) bought in {}, end condition met", c.sym))
                };
                if let Some(r) = finished {
                    out.push(json!({"type": "program_disable", "entity": pid, "entity_type": "player", "created_at": now, "reason": r}));
                    if prog.get("auto_pass_after").and_then(Value::as_bool).unwrap_or(false) {
                        out.push(json!({"type": "program_share_pass", "entity": pid, "entity_type": "player", "created_at": now,
                                        "unconditional": false, "indefinite": false}));
                    }
                    continue;
                }
                if acts.iter().any(|t| t == "buy_shares") {
                    let from_market = prog.get("from_market").and_then(Value::as_bool).unwrap_or(false);
                    let source = if from_market { "market" } else { "Treasury" };
                    let name = g.game.corporations[ci].sym.clone();
                    let cands = buy_candidates(g, seat, &corp, from_market);
                    if cands.is_empty() {
                        out.push(json!({"type": "program_disable", "entity": pid, "entity_type": "player", "created_at": now,
                                        "reason": format!("Cannot buy {name} from {source}")}));
                        continue;
                    }
                    let mut pcts: Vec<u32> = cands.iter().map(|x| x.1).collect();
                    pcts.sort();
                    pcts.dedup();
                    if pcts.len() != 1 {
                        out.push(json!({"type": "program_disable", "entity": pid, "entity_type": "player", "created_at": now,
                                        "reason": format!("Shares of different sizes exist, cannot auto buy {name} from {source}")}));
                        continue;
                    }
                    let share = cands[0].clone();
                    if let Some(r) = super::super::programs::should_stop_1822(g, pid, Some(&share), &acts) {
                        out.push(json!({"type": "program_disable", "entity": pid, "entity_type": "player", "created_at": now, "reason": r}));
                        continue;
                    }
                    out.push(json!({"type": "buy_shares", "entity": pid, "entity_type": "player", "created_at": now,
                                    "shares": [share.0], "percent": share.1}));
                } else if bought(g) && acts.iter().any(|t| t == "pass") {
                    if until.as_str() == Some("float") && g.game.corporations[ci].floated {
                        continue;
                    }
                    out.push(pass(json!(pid), "player", now));
                }
            }
            _ => {}
        }
    }
    out
}

/// The shares a buy program may take (Ruby: the IPO's or the market's shares the player can buy, by percent): the
/// generator's buys of that corporation from that source.
fn buy_candidates(g: &mut SiteGame, seat: u32, corp: &str, from_market: bool) -> Vec<(String, u32)> {
    let moves = g.game.g1822_legal_moves_with(CrossPlayerPolicy::RULES);
    let mut out: Vec<(String, u32)> = Vec::new();
    let ci = g.game.corp_idx[corp];
    for mv in &moves {
        let crate::actions::Action::BuyShares { corporation_sym, entity_id, percent, .. } = &mv.action else { continue };
        if corporation_sym != corp || entity_id != &seat.to_string() {
            continue;
        }
        let Some(id) = mv.shares.first() else { continue };
        let idx = id.rsplit_once('_').and_then(|(_, i)| i.parse::<usize>().ok());
        let in_market = idx.and_then(|i| g.game.corporations[ci].shares.get(i)).is_some_and(|s| s.owner.is_market());
        if in_market == from_market && !out.iter().any(|x| x.0 == *id) {
            out.push((id.clone(), *percent as u32));
        }
    }
    out
}

/// Ruby `bought?`: the turn has a purchase (a share bought, a par) by the player.
fn bought(g: &SiteGame) -> bool {
    g.turn_actions.iter().any(|a| {
        let t = a.get("type").and_then(Value::as_str).unwrap_or("");
        let by_player = a.get("entity_type").and_then(Value::as_str).unwrap_or("player") == "player";
        by_player && matches!(t, "buy_shares" | "par")
    })
}
