//! 1867's site layer (docs/SITE_ENGINE_RS_1867.md §3): what the site layer reads differently for 1867 than for 1830,
//! dispatched by title at the entry points of `SiteGame`, `WasmGame` and `site_engine`. 1867 runs through the shared
//! step machine (`steps.rs`: the title's step list, the blocking step, each step's actions) and the legal-move
//! generator (`g1867/legal.rs`), so this layer reads those instead of re-deriving Ruby's gates the way 1830's does.

pub mod act;
pub mod autos;
pub mod generate;
pub mod key;
pub mod log;
pub mod menu;
pub mod order;
pub mod view;

use super::game::SiteGame;
use crate::entities::Corporation;
use crate::game::BaseGame;
use crate::rounds::Round;
use crate::steps::{StepEntity, StepKind};
use crate::title::CorpType;

/// The game is 1867's.
pub fn is_1867(g: &SiteGame) -> bool {
    g.game.title == "1867"
}

/// A 1867 event has fired (engine-rs keys them "<train>:<event>", some plain).
pub fn event_fired(game: &BaseGame, name: &str) -> bool {
    game.events_fired.iter().any(|e| e == name || e.rsplit_once(':').is_some_and(|(_, n)| n == name))
}

/// Whether a corporation is in Ruby's `@corporations` (docs/SITE_ENGINE_RS_1867.md §3.1): the national never
/// (`setup` deletes it); before `green_minors_available` only the minors that are not green
/// (`setup_future_corporations!`); a closed one never (`close_corporation`); after `minors_cannot_start` no minor
/// without an owner (`event_minors_cannot_start!`: a minor can't be started later, so one unowned now stays out); after
/// `minors_nationalized` no minor (`event_minors_nationalized!`). A nationalized major keeps its place
/// (`reset_corporation`).
pub fn ruby_member(game: &BaseGame, c: &Corporation) -> bool {
    if c.closed || c.corp_type == CorpType::National {
        return false;
    }
    if c.corp_type != CorpType::Minor {
        return event_fired(game, "green_minors_available");
    }
    if event_fired(game, "minors_nationalized") {
        return false;
    }
    if !event_fired(game, "green_minors_available") && crate::title::g1867::GREEN_CORPORATIONS.contains(&c.sym.as_str()) {
        return false;
    }
    if event_fired(game, "minors_cannot_start") && !c.owner_id.is_player() {
        return false;
    }
    true
}

/// Ruby's `@corporations` in Ruby's order: the minors that are not green (title order), then at
/// `green_minors_available` the `@future_corporations` appended (title order: the majors, then the green minors),
/// less the ones that left (`ruby_member`).
pub fn ruby_corporations(game: &BaseGame) -> Vec<usize> {
    let mut first = Vec::new();
    let mut later = Vec::new();
    for (i, c) in game.corporations.iter().enumerate() {
        if !ruby_member(game, c) {
            continue;
        }
        if c.corp_type == CorpType::Minor && !crate::title::g1867::GREEN_CORPORATIONS.contains(&c.sym.as_str()) {
            first.push(i);
        } else {
            later.push(i);
        }
    }
    first.extend(later);
    first
}

/// Ruby's `round.active_step` and its first active entity (`step.active_entities.first || current_entity`): the
/// blocking step of the title's step list and its entity.
pub fn active_step(game: &mut BaseGame) -> Option<(StepKind, StepEntity)> {
    if game.finished {
        return None;
    }
    let snap = game.round.clone();
    let (i, e) = game.blocking_step_entity(&snap)?;
    let kind = game.round_step_descs(&snap).get(i)?.kind;
    if kind == StepKind::HomeToken && matches!(game.round, Round::Stock(_)) {
        // G1867::Step::HomeToken#active_entities: the ROUND's entity, which takes no action there (`actions` answers
        // only the corporation whose station is pending): in a stock round the player who started the company (the
        // par's player, or the auction's winner: BuySellParShares#win_bid's `goto_entity!`), its owner
        if let StepEntity::Corp(sym) = &e {
            // (a minor put up for auction places its station at once, HOME_TOKEN_TIMING :par, before it has an owner:
            // the round's entity is then the player who put it up, the stock round's current player)
            let owner = game.corp_idx.get(sym.as_str()).and_then(|&ci| {
                let c = &game.corporations[ci];
                c.owner_id.player_id().or_else(|| c.president_id())
            });
            let current = match &game.round {
                Round::Stock(s) => s.player_order.get(s.current_player_index).copied(),
                _ => None,
            };
            if let Some(p) = owner.or(current) {
                return Some((kind, StepEntity::Player(p)));
            }
        }
    }
    Some((kind, e))
}

/// Ruby `active_players_id` (`active_entities` mapped to players: a corporation's owner, a company's owner or its
/// owning corporation's).
pub fn active_players(game: &mut BaseGame) -> Vec<u32> {
    let Some((_, e)) = active_step(game) else { return Vec::new() };
    let owner_of_corp = |game: &BaseGame, sym: &str| {
        game.corp_idx.get(sym).and_then(|&ci| game.corporations[ci].owner_id.player_id())
    };
    match e {
        StepEntity::Player(p) => vec![p],
        StepEntity::Corp(s) => owner_of_corp(game, &s).into_iter().collect(),
        StepEntity::Company(s) => {
            let Some(&mi) = game.company_idx.get(s.as_str()) else { return Vec::new() };
            let owner = &game.companies[mi].owner;
            match owner.player_id() {
                Some(p) => vec![p],
                None => owner.corp_sym().and_then(|c| owner_of_corp(game, c)).into_iter().collect(),
            }
        }
    }
}

/// Ruby's step class name (`step.class.name.split('::').last`) for 1867's steps.
pub fn step_class(kind: StepKind) -> &'static str {
    match kind {
        StepKind::SingleItemAuction => "SingleItemAuction",
        StepKind::BuySellParShares => "BuySellParShares",
        StepKind::HomeToken => "HomeToken",
        StepKind::MajorTrainless => "MajorTrainless",
        StepKind::DiscardTrain => "DiscardTrain",
        StepKind::BuyCompany => "BuyCompany",
        StepKind::BuyCompanyPreloan => "BuyCompanyPreloan",
        StepKind::RedeemShares => "RedeemShares",
        StepKind::Track => "Track",
        StepKind::Token => "Token",
        StepKind::Route => "Route",
        StepKind::Dividend => "Dividend",
        StepKind::LoanOperations => "LoanOperations",
        StepKind::BuyTrain => "BuyTrain",
        StepKind::Merge => "Merge",
        StepKind::PostMergerShares => "PostMergerShares",
        StepKind::ReduceTokens => "ReduceTokens",
        StepKind::Bankrupt => "Bankrupt",
        StepKind::Exchange => "Exchange",
        StepKind::SpecialTrack => "SpecialTrack",
        StepKind::SpecialToken => "SpecialToken",
        StepKind::CompanyPendingPar => "CompanyPendingPar",
        StepKind::WaterfallAuction => "WaterfallAuction",
    }
}

/// The site's id of a step entity: a player's id, a corporation's or company's sym (Ruby `id.to_s`).
pub fn entity_id(e: &StepEntity) -> String {
    match e {
        StepEntity::Player(p) => p.to_string(),
        StepEntity::Corp(s) | StepEntity::Company(s) => s.clone(),
    }
}

/// Ruby's `step.actions(entity)` of the active step for its entity, in Ruby's order.
pub fn ruby_actions(game: &mut BaseGame) -> Vec<String> {
    let Some((kind, e)) = active_step(game) else { return Vec::new() };
    let snap = game.round.clone();
    let Some(desc) = game.round_step_descs(&snap).iter().find(|d| d.kind == kind).copied() else { return Vec::new() };
    let types: Vec<String> = game.step_actions(&desc, &e, &snap).into_iter().map(str::to_string).collect();
    ruby_order(kind, types)
}

/// Ruby's list of a step's actions where the engine's differs: the order, and the steps whose Ruby `actions` is all or
/// nothing while the engine lists only what is possible (the engine's step blocks either way, so Ruby's is the whole
/// list whenever it is the active step).
pub fn ruby_order_of(kind: StepKind, types: Vec<String>) -> Vec<String> {
    ruby_order(kind, types)
}

fn ruby_order(kind: StepKind, mut types: Vec<String>) -> Vec<String> {
    let fixed: Option<&[&str]> = match kind {
        // Engine::Step::Track#actions: ACTIONS unless !can_lay_tile? (a lay left this turn, its cost within the buying
        // power), whatever the map offers
        StepKind::Track => Some(&["lay_tile", "pass"]),
        // Engine::Step::BuyCompany#actions (and G1867's BuyCompanyPreloan): a blocking one lists ACTIONS when
        // can_buy_company? (any player's private, its minimum price within the cash), else nothing
        StepKind::BuyCompany | StepKind::BuyCompanyPreloan => Some(&["buy_company", "pass"]),
        _ => None,
    };
    if let Some(f) = fixed {
        // (Ruby's active step always has actions: `blocking?` needs them)
        return f.iter().map(|x| x.to_string()).collect();
    }
    let shares_first = types.iter().any(|t| matches!(t.as_str(), "buy_shares" | "par" | "sell_shares"));
    let order: &[&str] = match kind {
        // Engine::Step::BuySellParShares#actions (buy_shares, par, sell_shares, then pass if any), then
        // BuySellParSharesViaBid's bid, and its pass when the base list was empty (and `%w[bid pass]` in an auction)
        StepKind::BuySellParShares if shares_first => &["buy_shares", "par", "sell_shares", "pass", "bid"],
        StepKind::BuySellParShares => &["bid", "pass"],
        StepKind::MajorTrainless => &["pass", "choose"],
        StepKind::Merge => &["merge", "convert", "pass"],
        StepKind::PostMergerShares => &["buy_shares", "pass"],
        StepKind::BuyTrain => &["buy_train", "pass"],
        _ => return types,
    };
    types.sort_by_key(|t| order.iter().position(|x| x == t).unwrap_or(order.len()));
    types
}

/// The round's entity now (the snapshot's `current`, "p:<id>" / "c:<sym>" / "m:<sym>": Ruby `current_entity`, the
/// active step's: HomeToken's is the company whose station is pending, where its `active_entities` is the round's).
pub fn current_holder(game: &mut BaseGame) -> Option<String> {
    let snap = game.round.clone();
    let (i, raw) = game.blocking_step_entity(&snap)?;
    let kind = game.round_step_descs(&snap).get(i)?.kind;
    let e = if kind == StepKind::HomeToken { raw } else { active_step(game)?.1 };
    Some(match e {
        StepEntity::Player(p) => format!("p:{p}"),
        StepEntity::Corp(s) => format!("c:{s}"),
        StepEntity::Company(s) => format!("m:{s}"),
    })
}

/// The engine's own price of each route of a stored `run_routes` on the position before it (the shadow key's
/// `last_action`): as the authority prices a posted run (`route_check::price_stored_1867`: Ruby `compute_stops` +
/// `revenue_for` over the stops the rebuilt route visits, a stop's station per city); a run whose routes can't be
/// rebuilt by `title::g1867::recorded_route_revenue` (the same over the route's connection chains, per hex).
pub fn price_routes(g: &SiteGame, a: &serde_json::Value) -> Option<Vec<i32>> {
    let crate::actions::Action::RunRoutes { entity_id, routes, .. } = super::json::action_from_json(a).ok()? else {
        return None;
    };
    if let Some(raw) = a.get("routes").and_then(serde_json::Value::as_array) {
        if let Some(p) = super::route_check::price_stored_1867(g, &entity_id, raw) {
            return Some(p);
        }
    }
    let mut out = Vec::new();
    for r in &routes {
        let mut bare = r.clone();
        bare.revenue = None;
        out.push(crate::title::g1867::recorded_route_revenue(&g.game, &entity_id, &bare).ok()?);
    }
    Some(out)
}

/// The merger round's state the engine doesn't keep (Ruby's round_state of `G1867::Step::Merge`: `converts`, every
/// major converted or merged into this round; `merge_type`, the last one's kind), kept by the site layer.
#[derive(Clone, Debug, Default)]
pub struct MergerRecord {
    pub converts: Vec<String>,
    pub merge_type: Option<&'static str>,
    pub last_converted: Option<String>,
}

impl MergerRecord {
    /// After an engine action: a new merger round starts afresh; a major named for a conversion or a merger (the
    /// `merge` naming it: Merge#finish_convert / #finish_merge_to_major) joins `converts`, even when its shares are
    /// dealt within the same action.
    pub fn update(&mut self, before: Option<(bool, bool)>, action: &crate::actions::Action, round: &Round) {
        let Round::Merger(s) = round else {
            *self = MergerRecord::default();
            return;
        };
        let Some((converting, merge_major)) = before else {
            *self = MergerRecord::default();
            self.last_converted = s.converted.clone();
            return;
        };
        if let crate::actions::Action::Merge { corporation_sym, .. } = action {
            if converting || merge_major {
                self.converts.push(corporation_sym.clone());
                self.merge_type = Some(if converting { "convert" } else { "merge" });
            }
        }
        self.last_converted = s.converted.clone();
    }
}

/// The generator's moves now as wire dicts with their price ranges (debugging: `site_replay --dump`).
pub fn debug_moves(g: &mut SiteGame) -> Vec<serde_json::Value> {
    if !is_1867(g) {
        return Vec::new();
    }
    let moves = g.game.g1867_legal_moves_with(crate::title::CrossPlayerPolicy::RULES);
    moves
        .iter()
        .map(|m| serde_json::json!({"wire": g.game.g1867_move_wire(m), "range": m.price.map(|r| [r.lo, r.hi, r.step]), "step": format!("{:?}", m.step)}))
        .collect()
}

/// Which city of its hex each of a company's stations is in (Ruby's `token.city`; engine-rs's token knows only its
/// hex): kept by the site layer for the shadow key's `tokens` where a company holds two stations in one multi-city hex
/// (Montreal, Toronto). Per company, per token index: (hex, city index).
#[derive(Clone, Debug, Default)]
pub struct TokenCities {
    pub map: std::collections::HashMap<String, Vec<Option<(String, usize)>>>,
}

/// A company's place in Ruby's `Corporation#<=>` (the operating order's sort; G1867::Step::Merge sorts the merging
/// minors with it): the higher price, the column farther right, the higher row, then the earlier arrival on its share
/// price's cell (`share_price.corporations.find_index`), the name last; a company with no price after every other.
pub type CorpRank = (i32, i32, u8, usize, String);

pub fn corp_rank(game: &BaseGame, c: &crate::entities::Corporation) -> CorpRank {
    match c.share_price.as_ref() {
        Some(sp) => (-sp.price, -(sp.column as i32), sp.row, game.market_cell_position(&c.sym, sp.row, sp.column), c.name.clone()),
        None => (i32::MAX, 0, 0, 0, c.sym.clone()),
    }
}

/// `slot_owners` without its strings, for telling whether a move changed any: (hex index, city, slot, the
/// company's index); None when a station's company has no index (then the caller keeps `slot_owners` itself).
pub fn slot_owners_compact(game: &BaseGame) -> Option<Vec<(usize, usize, usize, usize)>> {
    let mut out = Vec::new();
    for (hi, h) in game.hexes.iter().enumerate() {
        for (ci, c) in h.tile.cities.iter().enumerate() {
            for (si, t) in c.tokens.iter().enumerate() {
                if let Some(t) = t {
                    out.push((hi, ci, si, *game.corp_idx.get(t.corporation_id.as_str())?));
                }
            }
        }
    }
    Some(out)
}

/// `slot_owners` from its compact form (the same board's hexes and companies).
pub fn slot_owners_expand(game: &BaseGame, compact: &[(usize, usize, usize, usize)]) -> std::collections::HashMap<(String, usize, usize), String> {
    compact.iter().map(|&(hi, ci, si, who)| ((game.hexes[hi].id.clone(), ci, si), game.corporations[who].sym.clone())).collect()
}

/// The companies occupying each city slot: (hex, city, slot) -> company.
pub fn slot_owners(game: &BaseGame) -> std::collections::HashMap<(String, usize, usize), String> {
    let mut out = std::collections::HashMap::new();
    for h in &game.hexes {
        for (ci, c) in h.tile.cities.iter().enumerate() {
            for (si, t) in c.tokens.iter().enumerate() {
                if let Some(t) = t {
                    out.insert((h.id.clone(), ci, si), t.corporation_id.clone());
                }
            }
        }
    }
    out
}

/// What `TokenCities::update` reads of the companies' tokens (per company, each token's use and hex), hashed.
pub fn token_sig(game: &BaseGame) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    for c in &game.corporations {
        c.sym.hash(&mut h);
        c.tokens.len().hash(&mut h);
        for t in &c.tokens {
            t.used.hash(&mut h);
            t.city_hex_id.hash(&mut h);
        }
    }
    h.finish()
}

impl TokenCities {
    /// Ruby `move_tokens_to_surviving` (`used + unused`): the company's tokens in use first, in their order.
    pub fn compact(&mut self, sym: &str) {
        if let Some(list) = self.map.get_mut(sym) {
            let n = list.len();
            let mut used: Vec<Option<(String, usize)>> = list.drain(..).filter(Option::is_some).collect();
            used.resize(n.max(used.len()), None);
            *list = used;
        }
    }

    /// After an engine action, per company, Ruby's tokens (the record's index is the token's place in Ruby's
    /// `corporation.tokens`; a record is a token in use): a station still standing keeps its token, also where
    /// engine-rs took another of the company's tokens on that hex off the map (it knows only the hex: Ruby's
    /// ReduceTokens removes the token in the city named, `token.remove!`, and the others keep their places); the
    /// stations newly placed (as many on a hex as the engine's tokens there beyond those kept) take Ruby's next tokens
    /// (`next_token`: the first not in use), several at once (a merger moving minors' stations) by their previous
    /// holders in Corporation#<=> order on the prices before (G1867::Step::Merge#move_tokens, `@merging.sort`).
    pub fn update(
        &mut self,
        before: &std::collections::HashMap<(String, usize, usize), String>,
        rank_before: &dyn Fn(&str) -> CorpRank,
        game: &BaseGame,
    ) {
        let after = slot_owners(game);
        for c in &game.corporations {
            let list = self.map.entry(c.sym.clone()).or_default();
            if list.len() < c.tokens.len() {
                list.resize(c.tokens.len(), None);
            }
            let occupied = |h: &str, ci: usize| after.iter().any(|((hh, cc, _), o)| hh == h && *cc == ci && o == &c.sym);
            // the stations still standing
            let mut taken: Vec<(String, usize)> = Vec::new();
            for rec in list.iter_mut() {
                let keep = rec.as_ref().is_some_and(|(h, ci)| occupied(h, *ci) && !taken.iter().any(|(th, tc)| th == h && tc == ci));
                if keep {
                    taken.push(rec.clone().unwrap());
                } else {
                    *rec = None;
                }
            }
            // the stations newly placed, hex by hex: as many as the engine's tokens there beyond those kept, each a
            // slot of the company's on the hex no token of its has, the newly occupied first, then by the previous
            // occupant's Ruby order
            let mut hexes: Vec<String> = Vec::new();
            for t in c.tokens.iter().filter(|t| t.used && !t.city_hex_id.is_empty()) {
                if !hexes.contains(&t.city_hex_id) {
                    hexes.push(t.city_hex_id.clone());
                }
            }
            let mut placed: Vec<((u8, CorpRank, usize), String, usize)> = Vec::new();
            for (hi, hex) in hexes.iter().enumerate() {
                let used = c.tokens.iter().filter(|t| t.used && &t.city_hex_id == hex).count();
                let kept = taken.iter().filter(|(h, _)| h == hex).count();
                if used <= kept {
                    continue;
                }
                let mut cands: Vec<((u8, CorpRank, usize), usize)> = after
                    .iter()
                    .filter(|((h, _, _), o)| h == hex && *o == &c.sym)
                    .filter(|((h, ci, _), _)| !taken.iter().any(|(th, tc)| th == h && tc == ci))
                    .map(|((h, ci, si), _)| {
                        let prev = before.get(&(h.clone(), *ci, *si));
                        let new = prev.map_or(true, |p| p != &c.sym) as u8;
                        let rank = prev.map_or((i32::MAX, 0, 0, 0, String::new()), |p| rank_before(p));
                        ((1 - new, rank, hi), *ci)
                    })
                    .collect();
                cands.sort();
                cands.dedup_by(|a, b| a.1 == b.1);
                placed.extend(cands.into_iter().take(used - kept).map(|(k, ci)| (k, hex.clone(), ci)));
            }
            // each takes Ruby's next token (`next_token`: the first not in use), in that order (a merger moves the
            // merging minors' stations one minor at a time)
            placed.sort();
            let mut free = (0..list.len()).filter(|&ti| list[ti].is_none()).collect::<Vec<_>>().into_iter();
            for (_, hex, ci) in placed {
                let ti = match free.next() {
                    Some(ti) => ti,
                    None => {
                        list.push(None);
                        list.len() - 1
                    }
                };
                list[ti] = Some((hex, ci));
            }
        }
    }
}
