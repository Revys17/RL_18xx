//! The 1822 family's site layer (docs/SITE_ENGINE_RS_1822.md §4.2): titles `1822` (2-7 players, its four optional
//! rules) and `1822CA` (the base game, 2-7 players, no optional rules), dispatched by title at the entry points of
//! `SiteGame`, `WasmGame` and `site_engine`.
//!
//! The engine is `crate::g1822`, a step machine of its own that mirrors Ruby's `G1822` steps one by one: the site
//! layer reads its steps, entities and step state instead of the shared step machine, decides moves with its
//! legal-move generator (`g1822_legal_moves_with(RULES)`, Ruby's player-time checks; the replay trusts the record as
//! Ruby's `load` does) and keeps itself the records the shared step machine's hooks (H1-H5) keep for 1830 and 1867.
//!
//! **Seat ids**: the site's user ids collide with 1822's numeric minors ("1".."30"), which engine-rs refuses, so every
//! site player id `u` is the engine's player `u + SEAT_BASE` and back at every boundary: actions in, views, menus,
//! keys and results out (`SiteGame::seat_offset`; 1830 and 1867 keep the site's ids).

pub mod act;
pub mod apply;
pub mod autos;
pub mod generate;
pub mod key;
pub mod log;
pub mod menu;
pub mod view;
#[cfg(test)]
mod tests;

use super::game::SiteGame;
use crate::entities::EntityId;
use crate::g1822::{Ent, RoundKind, Step};
use crate::game::BaseGame;
use serde_json::Value;
use std::collections::BTreeMap;

/// The site titles of the family engine-rs plays (ERS / WRS are other titles: `unsupported:`).
pub const TITLES: &[&str] = &["1822", "1822CA"];

/// What every site player id is offset by in the engine (above the minors' syms and every company's).
pub const SEAT_BASE: u32 = 1000;

/// The game is one of the family's.
pub fn is_1822(g: &SiteGame) -> bool {
    g.game.g1822.is_some()
}

/// The game is 1822CA's.
pub fn is_ca(game: &BaseGame) -> bool {
    game.g1822.is_some() && game.title == "1822CA"
}

/// Ruby `initialize_seed`: `settings.seed`, else the game id's first digit run (0 without either).
pub fn ruby_seed(data: &Value) -> u64 {
    match data.pointer("/settings/seed") {
        Some(Value::Number(n)) => return n.as_u64().or_else(|| n.as_i64().map(|x| x as u64)).unwrap_or(0),
        Some(Value::String(s)) if !s.is_empty() => {
            if let Ok(x) = s.trim().parse::<u64>() {
                return x;
            }
        }
        _ => {}
    }
    let id = match data.get("id") {
        Some(Value::Number(n)) => n.to_string(),
        Some(Value::String(s)) => s.clone(),
        _ => String::new(),
    };
    let digits: String = id.chars().skip_while(|c| !c.is_ascii_digit()).take_while(|c| c.is_ascii_digit()).collect();
    digits.parse().unwrap_or(0)
}

/// The site layer's own records and step state for the family (what the shared step machine's hooks keep for 1830
/// and 1867, and what the engine doesn't keep the way Ruby shows it).
#[derive(Clone, Debug, Default)]
pub struct Records {
    /// The operating round each corporation's operating history entries were written in, by count (Ruby
    /// `operating_history.size`, which the engine counts): what the records of `game.site` already hold
    pub op_seen: BTreeMap<String, u32>,
    /// The revenue of the run each corporation made this operating turn (its `run_routes`), until its dividend
    pub run: Option<(String, i32, bool)>,
    /// G1822::Round::Stock's `@stored_winning_bids`: the bid-box lots each player held the high bid on when their
    /// last turn ended (`BuySellParShares#pass!`) or they enabled a program (`player_enabled_program`), site ids;
    /// with the stock round (its turn) they belong to
    pub stored_winning: BTreeMap<u32, Vec<String>>,
    pub stored_round: Option<u32>,
}

/// `update_stored_winning_bids(player)`: the bid-box lots whose highest bid is the player's (site id).
pub fn update_stored_winning(g: &mut SiteGame, pid: u32) {
    let game = &g.game;
    if game.g1822.is_none() || game.rnd().kind != RoundKind::Stock {
        return;
    }
    let seat = g.seat(pid);
    let mut lots = game.bidbox_minors();
    lots.extend(game.bidbox_concessions());
    lots.extend(game.bidbox_privates());
    let winning: Vec<String> = lots.into_iter().filter(|c| game.g1822_highest_bid(c).is_some_and(|b| b.player == seat)).collect();
    let turn = game.turn;
    if g.g1822.stored_round != Some(turn) {
        g.g1822.stored_winning.clear();
        g.g1822.stored_round = Some(turn);
    }
    g.g1822.stored_winning.insert(pid, winning);
}

/// `stored_winning_bids(player)` of the stock round in progress.
pub fn stored_winning(g: &SiteGame, pid: u32) -> Vec<String> {
    if g.g1822.stored_round != Some(g.game.turn) {
        return Vec::new();
    }
    g.g1822.stored_winning.get(&pid).cloned().unwrap_or_default()
}

impl SiteGame {
    /// The engine's player id of a site player id.
    pub fn seat(&self, site: u32) -> u32 {
        site + self.seat_offset
    }
    /// The site's player id of an engine player id.
    pub fn site_id(&self, seat: u32) -> u32 {
        seat.saturating_sub(self.seat_offset)
    }
}

/// Ruby's `round.active_step` and its current entity (`step.current_entity`): (index, step, entity).
pub fn active(game: &BaseGame) -> Option<(usize, Step, Ent)> {
    if game.finished || game.g1822.is_none() {
        return None;
    }
    let bi = game.g1822_active_step()?;
    let step = game.rnd().steps[bi];
    let ent = game.g1822_step_current_entity(bi)?;
    Some((bi, step, ent))
}

/// Ruby's step class name (`step.class.name.split('::').last`): the engine's steps are named as Ruby's.
pub fn step_class(step: Step) -> String {
    format!("{step:?}")
}

/// Ruby's round name (`round.name`): "Stock Round", G1822's Choices round "Choose", "Operating Round".
pub fn round_name(game: &BaseGame) -> &'static str {
    match game.rnd().kind {
        RoundKind::Stock => "Stock Round",
        RoundKind::Choices => "Choose",
        RoundKind::Operating => "Operating Round",
        RoundKind::Merger => "Merger Round",
    }
}

/// Ruby's round short name (`round.class.short_name`): "SR", "Choices", "OR".
pub fn round_short(game: &BaseGame) -> &'static str {
    match game.rnd().kind {
        RoundKind::Stock => "SR",
        RoundKind::Choices => "Choices",
        RoundKind::Operating => "OR",
        RoundKind::Merger => "MR",
    }
}

/// A corporation's president (Ruby `corporation.owner`): a minor's owner, a major's president.
pub fn corp_owner(game: &BaseGame, sym: &str) -> Option<u32> {
    game.g1822_corp_president(sym)
}

/// Ruby `acting_for_entity(entity)`: the player who decides for an entity (a corporation's president, a company's
/// owner or its owning corporation's president), as an engine player id.
pub fn acting_for(game: &BaseGame, e: &Ent) -> Option<u32> {
    match e {
        Ent::Player(p) => Some(*p),
        Ent::Corp(c) => corp_owner(game, c),
        Ent::Company(co) => {
            let owner = game.co_owner(co);
            match owner.corp_sym() {
                Some(c) if owner.is_corporation() => corp_owner(game, c),
                _ => owner.player_id(),
            }
        }
    }
}

/// Ruby `active_players_id`, as site ids: the active step's active entities' players (DiscardTrain: every crowded
/// company's president, in Ruby's `crowded_corps` order; elsewhere the step's current entity's).
pub fn active_players(g: &SiteGame) -> Vec<u32> {
    let game = &g.game;
    let Some((_, step, ent)) = active(game) else { return Vec::new() };
    let mut out: Vec<u32> = Vec::new();
    if step == Step::DiscardTrain {
        for c in game.g1822_crowded_corps() {
            if let Some(p) = corp_owner(game, &c) {
                let s = g.site_id(p);
                if !out.contains(&s) {
                    out.push(s);
                }
            }
        }
        return out;
    }
    acting_for(game, &ent).map(|p| g.site_id(p)).into_iter().collect()
}

/// Ruby `step.actions(entity)` of the active step for its current entity (bot_view's `actions`).
pub fn ruby_actions(game: &BaseGame) -> Vec<String> {
    let Some((bi, _, ent)) = active(game) else { return Vec::new() };
    game.g1822_step_actions(bi, &ent).into_iter().map(str::to_string).collect()
}

/// The site's id of an engine entity: a player's site id, a corporation's or company's sym.
pub fn entity_site_id(g: &SiteGame, e: &Ent) -> String {
    match e {
        Ent::Player(p) => g.site_id(*p).to_string(),
        Ent::Corp(s) | Ent::Company(s) => s.clone(),
    }
}

/// Ruby's `engine.corporations` in its order: every corporation in play that hasn't closed (the 1822+ ones without
/// the expansion are never made; a closed one leaves).
pub fn members(game: &BaseGame) -> Vec<usize> {
    // (1822CA's Detroit-Duluth blocker "DD" is a Corporation Ruby keeps outside `@corporations`:
    // G1822CA#block_detroit_duluth)
    (0..game.corporations.len()).filter(|&i| !game.corporations[i].closed && !(is_ca(game) && game.corporations[i].sym == "DD")).collect()
}

/// Ruby `@game_end_reason` once the game is over.
pub fn end_reason(game: &BaseGame) -> &'static str {
    match game.g().game_end_reason.as_deref() {
        Some("bank") => "bank",
        Some("stock_market") => "stock_market",
        Some("final_phase") => "final_phase",
        Some("final_or") => "final_or",
        Some("dnf") => "dnf",
        Some(_) | None => "unknown",
    }
}

/// An action as the engine reads it: the site's player ids seated (`entity` of a player's action).
pub fn to_engine(g: &SiteGame, a: &Value) -> Value {
    let mut out = a.clone();
    let is_player = match a.get("entity_type").and_then(Value::as_str) {
        Some(t) => t == "player",
        // (no entity_type: a number that is one of the game's players)
        None => a.get("entity").and_then(|e| e.as_u64().or_else(|| e.as_str().and_then(|s| s.parse().ok()))).is_some_and(|p| {
            g.players.iter().any(|x| x.0 as u64 == p)
        }),
    };
    if is_player {
        let id = a.get("entity").and_then(|e| e.as_u64().or_else(|| e.as_str().and_then(|s| s.parse().ok())));
        if let Some(id) = id {
            out["entity"] = Value::from(id + g.seat_offset as u64);
        }
    }
    out
}

/// A move's wire form (`g1822_move_wire`) as the site names it: a player's entity by its site id.
pub fn to_site(g: &SiteGame, w: &Value) -> Value {
    let mut out = w.clone();
    if w.get("entity_type").and_then(Value::as_str) == Some("player") {
        if let Some(p) = w.get("entity").and_then(Value::as_u64) {
            out["entity"] = Value::from(g.site_id(p as u32));
        }
    }
    out
}

/// A holder's code in the site's terms: "p<site id>", "c<sym>", "bank".
pub fn holder_code(g: &SiteGame, e: &EntityId) -> String {
    if let Some(p) = e.player_id() {
        format!("p{}", g.site_id(p))
    } else if let Some(c) = e.corp_sym().filter(|_| e.is_corporation()) {
        format!("c{c}")
    } else {
        "bank".into()
    }
}

/// The engine's price of each route of a site `run_routes` (the shadow key's `last_action`): Ruby's `revenue_for`
/// on the position before the run (the engine's pricer of a record without revenues), or the recorded revenues
/// where it can't price them.
pub fn price_routes(g: &SiteGame, a: &Value) -> Vec<i32> {
    let Some(corp) = a.get("entity").and_then(Value::as_str) else { return Vec::new() };
    let Ok(crate::actions::Action::RunRoutes { routes, .. }) = super::json::action_from_json(a) else { return Vec::new() };
    let bare: Vec<crate::actions::RouteData> = routes.iter().map(|r| crate::actions::RouteData { revenue: None, ..r.clone() }).collect();
    if !g.game.corp_idx.contains_key(corp) {
        return Vec::new();
    }
    match g.game.g1822_price_recorded_routes(corp, &bare) {
        Ok(p) => p,
        Err(_) => routes.iter().map(|r| r.revenue.unwrap_or(0)).collect(),
    }
}

/// The family's phases' Ruby statuses (the title data's `PhaseDef.status`, Ruby's own words).
pub fn phase_status(game: &BaseGame) -> Vec<String> {
    game.title_def()
        .phases()
        .iter()
        .find(|p| p.name == game.phase.name)
        .map(|p| p.status.iter().map(|s| s.to_string()).collect())
        .unwrap_or_default()
}

/// The bot API's view facts for the family (`authority::bot_view`, contract K12): Ruby's `bot_view` values on the
/// position, the site's ids: the step's entity (`type` corporation for minors: Ruby's minors are corporations), its
/// class name and actions, the corporations with trains in Ruby's order, the depot's `depot_trains`.
pub fn bot_view(g: &mut SiteGame, mut out: Value) -> Value {
    let game = &g.game;
    let Some((_, step, ent)) = active(game) else { return out };
    let (id, etype, player) = match &ent {
        Ent::Player(p) => (serde_json::json!(g.site_id(*p)), "player", Some(g.site_id(*p))),
        Ent::Corp(c) => (serde_json::json!(c), "corporation", corp_owner(game, c).map(|p| g.site_id(p))),
        Ent::Company(c) => (serde_json::json!(c), "company", acting_for(game, &ent).map(|p| g.site_id(p))),
    };
    out["entity"] = serde_json::json!({"id": id, "type": etype, "player": player});
    out["step"] = serde_json::json!(step_class(step));
    out["actions"] = serde_json::json!(ruby_actions(game));
    let trains: Vec<Value> = members(game)
        .into_iter()
        .map(|i| &game.corporations[i])
        .filter(|c| !c.trains.is_empty())
        .map(|c| serde_json::json!([c.sym, c.trains.iter().map(|t| serde_json::json!({"id": t.id, "name": t.name})).collect::<Vec<_>>()]))
        .collect();
    out["trains"] = serde_json::json!(trains);
    // Ruby depot_trains: the next train, one of each later type already available, one of each discarded type
    let mut depot: Vec<Value> = Vec::new();
    let mut names: Vec<String> = Vec::new();
    if let Some(t) = game.depot.trains.first() {
        names.push(t.name.clone());
        depot.push(serde_json::json!({"id": t.id, "name": t.name}));
    }
    for t in &game.depot.trains {
        if !names.contains(&t.name) && game.phase_available(t.available_on.as_deref()) {
            names.push(t.name.clone());
            depot.push(serde_json::json!({"id": t.id, "name": t.name}));
        }
    }
    let mut dn: Vec<String> = Vec::new();
    for t in &game.depot.discarded {
        if !dn.contains(&t.name) {
            dn.push(t.name.clone());
            depot.push(serde_json::json!({"id": t.id, "name": t.name}));
        }
    }
    out["depot"] = serde_json::json!(depot);
    out
}
