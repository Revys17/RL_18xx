//! One engine action of an 1822-family game (docs/SITE_ENGINE_RS_1822.md §4.2): the site's JSON in Ruby's terms
//! seated (`to_engine`), the extras the shared `Action` has no field for (`combo_entities`, a bundle's `share_price`)
//! handed to the engine as the Python bridge hands them, processed by the family's step machine, and around it the
//! records Ruby keeps that the engine doesn't (operating history with its revenue and kind, the players' values at
//! each round's end), the turn in progress, the companies' order, the last runs and lays, and the log.

use super::super::game::{RunRecord, SiteGame};
use super::super::json::action_from_json;
use crate::actions::{Action, GameError};
use crate::g1822::RoundKind;
use serde_json::Value;
use std::collections::BTreeMap;

fn atype(a: &Value) -> &str {
    a.get("type").and_then(Value::as_str).unwrap_or("")
}

fn str_list(v: Option<&Value>) -> Vec<String> {
    match v {
        Some(Value::Array(xs)) => xs.iter().filter_map(|x| x.as_str().map(str::to_string)).collect(),
        _ => Vec::new(),
    }
}

/// The round and turn identity (Ruby calls `next_turn!` when the round or its entity index changes).
#[derive(Clone, Debug, PartialEq)]
pub struct TurnKey {
    pub kind: RoundKind,
    pub turn: u32,
    pub round_num: u8,
    pub entity_index: usize,
}

pub fn turn_key(g: &SiteGame) -> TurnKey {
    let r = g.game.rnd();
    TurnKey { kind: r.kind, turn: g.game.turn, round_num: r.round_num, entity_index: r.entity_index }
}

/// What the records need from the position before an action.
pub struct Before {
    pub key: TurnKey,
    pub stock_turn: Option<(u32, Vec<u32>, usize)>,
    pub step: String,
    pub op_history: BTreeMap<String, u32>,
    /// The routes the round holds (the operating company's run awaiting its dividend): (operator, revenue, any)
    pub routes: Option<(String, i32, bool)>,
    pub finished: bool,
}

pub fn before(g: &SiteGame) -> Before {
    let game = &g.game;
    let r = game.rnd();
    let routes = r.current_operator.clone().or_else(|| r.entities.get(r.entity_index).cloned()).map(|op| {
        (op, game.g1822_routes_revenue(), !r.routes.is_empty())
    });
    Before {
        key: turn_key(g),
        stock_turn: super::super::programs::stock_turn(g),
        step: super::active(game).map(|(_, s, _)| super::step_class(s)).unwrap_or_default(),
        op_history: game.g().operating_history.clone(),
        routes,
        finished: game.finished,
    }
}

/// A site action's `share_price` as the engine takes it (Ruby `@share_price || price_per_share`: false is none).
fn share_price(a: &Value) -> Option<i32> {
    match a.get("share_price") {
        Some(Value::Number(n)) => n.as_i64().or_else(|| n.as_f64().map(|f| f as i64)).map(|x| x as i32),
        Some(Value::String(s)) => s.split(',').next().and_then(|x| x.parse().ok()),
        _ => None,
    }
}

/// Process one engine action (the site's JSON, the site's ids) and keep the records.
pub fn apply_engine(g: &mut SiteGame, a: &Value) -> Result<(), GameError> {
    let t = atype(a).to_string();
    let translated = super::to_engine(g, a);
    let action = action_from_json(&translated)?;
    let combo = str_list(a.get("combo_entities"));
    let sp = if t == "buy_shares" || t == "sell_shares" { share_price(a) } else { None };
    let b = before(g);
    let log_before = g.site_log.then(|| super::log::before(g, a));
    let tr_before = (g.game.turn, g.game.rnd().round_num);
    g.game.gm().pending_combo = combo;
    g.game.gm().pending_share_price = sp;
    let r = g.game.process_action_internal(&action);
    if g.game.g1822.is_some() {
        g.game.gm().pending_combo.clear();
        g.game.gm().pending_share_price = None;
    }
    r?;
    g.engine_count += 1;
    records(g, &b, a, &action);
    // the turn in progress (Ruby's round.current_actions), move by move, auto actions included
    let advanced = turn_key(g) != b.key;
    if advanced {
        g.turn_advanced = true;
        g.turn_actions.clear();
        g.turn_steps.clear();
    } else {
        g.turn_actions.push(a.clone());
        g.turn_steps.push(b.step.clone());
    }
    companies_moved(g);
    // the Programmer's history (Ruby's players_history; G1822 tracks bids too) and the stored winning bids at the end
    // of a stock turn (BuySellParShares#pass!)
    let hist_player = super::super::authority::entity_player(g, a).flatten();
    super::super::programs::observe(g, b.stock_turn.clone(), a, hist_player, None, advanced, false);
    if b.key.kind == RoundKind::Stock && b.step == "BuySellParShares" && advanced {
        if let Some(p) = hist_player.filter(|_| a.get("entity_type").and_then(Value::as_str) != Some("company")) {
            super::update_stored_winning(g, p);
        }
    }
    match &action {
        Action::LayTile { hex_id, entity_id, .. } => {
            let corp = g.operator_of(entity_id);
            let (lt, lr) = tr_before;
            let list = g.laid_hexes.entry(corp).or_default();
            if list.last().is_none_or(|e| (e.0, e.1) != (lt, lr)) {
                list.push((lt, lr, Vec::new()));
                if list.len() > 4 {
                    list.remove(0);
                }
            }
            if let Some(e) = list.last_mut() {
                e.2.push(hex_id.clone());
            }
        }
        Action::RunRoutes { entity_id, .. } => {
            let routes = a.get("routes").and_then(Value::as_array).cloned().unwrap_or_default();
            g.last_runs.insert(entity_id.clone(), RunRecord { routes, turn: tr_before.0, round: tr_before.1 });
        }
        _ => {}
    }
    if let Some(lb) = log_before {
        for l in super::log::explain(g, &lb, a, &action) {
            g.log_push(l);
        }
    }
    if !b.finished && g.game.finished {
        g.log_game_over();
    }
    Ok(())
}

/// Ruby's records around one action: an operating history entry for each company whose dividend step ran
/// (`process_dividend`: its run's revenue, the action's kind, a minor's automatic `half` / `withhold`), the players'
/// values when a round finishes (`store_player_info`: at each transition and at the end).
fn records(g: &mut SiteGame, b: &Before, raw: &Value, action: &Action) {
    let now: BTreeMap<String, u32> = g.game.g().operating_history.clone();
    let (turn, round) = (b.key.turn, b.key.round_num);
    for (sym, n) in &now {
        let was = b.op_history.get(sym).copied().unwrap_or(0);
        if *n <= was {
            continue;
        }
        // the run: this action's own routes, else the routes the round held before it
        let (revenue, ran) = match action {
            Action::RunRoutes { entity_id, routes, .. } if entity_id == sym => {
                let prices = g.last_run_prices.as_ref().filter(|(id, _)| Some(*id) == raw.get("id").and_then(Value::as_i64)).map(|(_, p)| p.clone());
                let total: i32 = routes
                    .iter()
                    .enumerate()
                    .map(|(k, r)| r.revenue.unwrap_or_else(|| prices.as_ref().and_then(|p| p.get(k)).copied().unwrap_or(0)))
                    .sum();
                (total, !routes.is_empty())
            }
            _ => match &b.routes {
                Some((op, rev, any)) if op == sym => (*rev, *any),
                _ => (0, false),
            },
        };
        let minor = g.game.is_minor(sym);
        let kind = match action {
            Action::Dividend { entity_id, kind } if entity_id == sym => match kind {
                crate::actions::DividendKind::Payout => "payout",
                crate::actions::DividendKind::Half => "half",
                crate::actions::DividendKind::Withhold => "withhold",
            },
            _ if minor && revenue > 0 => "half",
            _ => "withhold",
        };
        g.game.site_note_operated_at(sym, turn, round, revenue, kind, !ran && revenue == 0);
    }
    // the players' values at each round's end (one entry per finished round; the end of the game writes its round's)
    let after = turn_key(g);
    let round_changed = (after.kind, after.turn, after.round_num) != (b.key.kind, b.key.turn, b.key.round_num);
    let ended = !b.finished && g.game.finished;
    if round_changed && !ended {
        // Ruby's check_programmed_actions after the round it starts: the programs whose `disable?` holds there
        let kind = super::round_short(&g.game);
        for list in g.programs.values_mut() {
            list.retain(|p| !super::super::game::program_disabled(p, kind));
        }
        g.programs.retain(|_, l| !l.is_empty());
    }
    if round_changed || ended {
        let short = match b.key.kind {
            RoundKind::Stock => "SR",
            RoundKind::Choices => "Choices",
            RoundKind::Operating => "OR",
            RoundKind::Merger => "MR",
        };
        let num = if b.key.kind == RoundKind::Operating { b.key.round_num.max(1) } else { 1 };
        let values: Vec<(u32, i32)> =
            g.game.players.iter().map(|p| (g.site_id(p.id), g.game.g1822_player_value(p.id))).collect();
        if let Some(rec) = g.game.site.as_mut() {
            rec.player_history.push(crate::site::records::PlayerInfo { round: short.into(), turn: b.key.turn, round_num: num, values });
        }
    }
}

/// The companies that changed hands, in the order Ruby's `companies` lists keep (one at a time), and the share
/// holders in the order they first held a share (Ruby's `share_holders` keys).
fn companies_moved(g: &mut SiteGame) {
    for i in 0..g.game.companies.len() {
        let (sym, owner) = (g.game.companies[i].sym.clone(), g.game.companies[i].owner.0.clone());
        if g.company_owner.get(&sym) != Some(&owner) {
            g.seq += 1;
            g.company_seq.insert(sym.clone(), g.seq);
            g.company_owner.insert(sym, owner);
        }
    }
    for c in &g.game.corporations {
        let list = g.holders.entry(c.sym.clone()).or_default();
        let mut new: Vec<(u64, u32)> = c
            .shares
            .iter()
            .filter_map(|s| s.owner.player_id().map(|p| (s.acquired_seq, p)))
            .filter(|(_, p)| !list.contains(p))
            .collect();
        new.sort();
        for (_, p) in new {
            if !list.contains(&p) {
                list.push(p);
            }
        }
    }
}
