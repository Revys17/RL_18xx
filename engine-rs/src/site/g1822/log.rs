//! The 1822 family's log lines (docs/SITE_ENGINE_RS_1822.md §4.2): Ruby's words (the templates of `Engine::Step::*`,
//! `Engine::SharePool`, `Engine::Phase` and `G1822::*`) for what an action did, read off the position before and
//! after it. The main beats: round and phase headers, bids and the bid boxes' resolution, passes, buys, sales, pars
//! and conversions, price moves, floats, presidencies, lays, stations, runs, dividends, trains, discards,
//! acquisitions, debt, closes, the new player order. What a beat Ruby writes is missing here shows as fewer lines,
//! never as wrong state (the log is display only).

use super::super::game::SiteGame;
use crate::actions::{Action, DividendKind};
use crate::entities::EntityId;
use crate::g1822::RoundKind;
use crate::game::BaseGame;
use serde_json::Value;

include!("locations.rs");

/// The position before an action, as the log reads it.
pub struct Before {
    pub game: BaseGame,
    /// The active step's description before the action
    pub step_label: Option<String>,
    /// The turn's actions so far (Ruby's round.current_actions, for log_pass)
    pub turn: Vec<Value>,
}

pub fn before(g: &SiteGame, _raw: &Value) -> Before {
    Before { game: g.game.clone_for_search_without_log(), step_label: super::view::step_label(g), turn: g.turn_actions.clone() }
}

fn cur(game: &BaseGame) -> &'static str {
    if super::is_ca(game) { "$" } else { "£" }
}

fn money(game: &BaseGame, x: i32) -> String {
    format!("{}{x}", cur(game))
}

fn pname(game: &BaseGame, pid: u32) -> String {
    game.players.iter().find(|p| p.id == pid).map(|p| p.name.clone()).unwrap_or_default()
}

fn coname(game: &BaseGame, sym: &str) -> String {
    game.title_def().companies().iter().find(|c| c.sym == sym).map(|c| c.name.to_string()).unwrap_or_else(|| sym.to_string())
}

fn holder_name(game: &BaseGame, e: &EntityId) -> String {
    if let Some(p) = e.player_id() {
        pname(game, p)
    } else if let Some(c) = e.corp_sym() {
        c.to_string()
    } else {
        "the bank".into()
    }
}

/// A hex as the log names it: "L1 (London)".
fn hex_desc(game: &BaseGame, hex: &str) -> String {
    let table = if super::is_ca(game) { LOCATIONS_1822CA } else { LOCATIONS_1822 };
    match table.iter().find(|(h, _)| *h == hex) {
        Some((_, loc)) => format!("{hex} ({loc})"),
        None => hex.to_string(),
    }
}

/// An action's entity's name (a player's, a corporation's sym, a company's name).
fn ent_name(game: &BaseGame, id: &str) -> String {
    match game.g1822_entity(id) {
        crate::g1822::Ent::Player(p) => pname(game, p),
        crate::g1822::Ent::Corp(c) => c,
        crate::g1822::Ent::Company(c) => coname(game, &c),
    }
}

/// "-- Phase 1 (Operating Rounds: 1 | Train Limit: minor: 2, major: 4 | Available Tiles: Yellow) --"
fn phase_line(game: &BaseGame) -> String {
    let title = game.title_def();
    let Some(p) = title.phases().into_iter().find(|p| p.name == game.phase.name) else { return String::new() };
    let limit = match p.minor_train_limit {
        Some(m) => format!("minor: {m}, major: {}", p.train_limit),
        None => p.train_limit.to_string(),
    };
    let tiles: Vec<String> = p
        .tiles
        .iter()
        .map(|t| {
            let mut c = t.chars();
            c.next().map(|f| f.to_uppercase().collect::<String>() + c.as_str()).unwrap_or_default()
        })
        .collect();
    format!("-- Phase {} (Operating Rounds: {} | Train Limit: {limit} | Available Tiles: {}) --", p.name, p.operating_rounds, tiles.join(", "))
}

/// The lines written before the first action: 1822CA's beta notice, the phase, the optional rules.
pub fn start(g: &SiteGame) -> Vec<String> {
    let game = &g.game;
    let mut out = Vec::new();
    if super::is_ca(game) {
        out.push("1822CA is currently considered 'beta', the rules implementation may allow illegal moves.".into());
        out.push("As the implementation improves, games that are not compatible with the latest version will be pinned but may be archived after 7 days.".into());
        out.push("Because of this we suggest not playing games that may take months to complete.".into());
    }
    out.push(phase_line(game));
    let rules: &[(&str, &str)] = &[
        ("plus_expansion", "1822+: 6 more minors and 3 more privates. The privates are categorized into trains (bidbox 1), track (bidbox 2) and other (bidbox 3) stacks."),
        ("plus_expansion_no_removals", "No Removals: (1822+) Use all 21 private companies instead of removing 3. "),
        ("plus_expansion_single_stack", "Single Stack: (1822+) The privates are not categorized, and fill the 3 bidboxes as in base 1822."),
        ("tax_haven_multiple", "Tax Haven Variant: P16 (Tax Haven) can use the cash it accumulates to buy 1 share per SR. Cannot own multiple shares of one corporation. To ensure P16 is present in 1822+, use the \"No Removals\" variant."),
    ];
    let used: Vec<&str> = rules.iter().filter(|(k, _)| game.optional_rules.iter().any(|r| r == k)).map(|(_, t)| *t).collect();
    if !used.is_empty() {
        out.push("Optional rules used in this game:".into());
        for t in used {
            out.push(format!(" * {t}"));
        }
    }
    out
}

/// "X's share price moves right from £A to £B", one line per corporation whose price moved.
fn price_moves(b: &BaseGame, a: &BaseGame, out: &mut Vec<String>) {
    for c in &a.corporations {
        let Some(sp) = c.share_price.as_ref() else { continue };
        let Some(&bi) = b.corp_idx.get(c.sym.as_str()) else { continue };
        let Some(bp) = b.corporations[bi].share_price.as_ref() else { continue };
        if (bp.row, bp.column) == (sp.row, sp.column) {
            continue;
        }
        let dir = if sp.column > bp.column {
            "right"
        } else if sp.column < bp.column {
            "left"
        } else if sp.row > bp.row {
            "down"
        } else {
            "up"
        };
        out.push(format!("{}'s share price moves {dir} from {} to {}", c.sym, money(a, bp.price), money(a, sp.price)));
    }
}

/// Floats and presidencies that changed.
fn floats_and_presidents(b: &BaseGame, a: &BaseGame, settled: &[String], out: &mut Vec<String>) {
    for c in &a.corporations {
        if settled.contains(&c.sym) {
            continue;
        }
        let Some(&bi) = b.corp_idx.get(c.sym.as_str()) else { continue };
        let bc = &b.corporations[bi];
        if c.floated && !bc.floated {
            out.push(format!("{} floats", c.sym));
        }
        if c.corp_type == crate::title::CorpType::Major {
            let (bp, ap) = (b.g1822_corp_president(&c.sym), a.g1822_corp_president(&c.sym));
            if ap.is_some() && bp.is_some() && ap != bp {
                out.push(format!("{} becomes the president of {}", pname(a, ap.unwrap()), c.sym));
            }
        }
    }
}

/// The trains that rusted, the phase that began, the events.
fn phases(b: &BaseGame, a: &BaseGame, out: &mut Vec<String>) {
    if a.phase.name != b.phase.name {
        out.push(phase_line(a));
    }
    let gone: Vec<String> = {
        let held = |g: &BaseGame| -> Vec<(String, String)> {
            g.corporations.iter().flat_map(|c| c.trains.iter().map(move |t| (t.id.clone(), t.name.clone()))).collect()
        };
        let after: Vec<String> = held(a).into_iter().map(|x| x.0).collect();
        let mut names: Vec<String> = Vec::new();
        for (id, name) in held(b) {
            let discarded = a.depot.discarded.iter().any(|t| t.id == id);
            if !after.contains(&id) && !discarded && !names.contains(&name) {
                names.push(name);
            }
        }
        names
    };
    if a.phase.name != b.phase.name && !gone.is_empty() {
        out.push(format!("-- Event: {} trains rust --", gone.join(", ")));
    }
}

/// The round that began (Ruby `new_stock_round` / `new_operating_round` / G1822's Choices), the new player order.
fn transitions(b: &BaseGame, a: &BaseGame, out: &mut Vec<String>) {
    let (rb, ra) = (b.rnd(), a.rnd());
    if (rb.kind, b.turn, rb.round_num) == (ra.kind, a.turn, ra.round_num) || a.finished {
        return;
    }
    if b.players.iter().map(|p| p.id).collect::<Vec<_>>() != a.players.iter().map(|p| p.id).collect::<Vec<_>>() {
        let order: Vec<String> = a.players.iter().map(|p| p.name.clone()).collect();
        out.push(format!("-- New player order: {}", order.join(", ")));
    }
    match ra.kind {
        RoundKind::Stock => out.push(format!("-- Stock Round {} --", a.turn)),
        RoundKind::Operating => {
            out.push(format!("-- Operating Round {}.{} (of {}) --", a.turn, ra.round_num, a.g1822_operating_rounds_in_set()))
        }
        _ => {}
    }
}

/// Ruby BuySellParShares#log_pass after a stock round move that passes the turn.
fn log_pass(game: &BaseGame, who: &str, turn: &[Value], out: &mut Vec<String>) {
    if turn.is_empty() {
        out.push(format!("{who} passes"));
        return;
    }
    let bought = turn.iter().any(|a| matches!(a.get("type").and_then(Value::as_str), Some("buy_shares" | "par")));
    let sold = turn.iter().any(|a| a.get("type").and_then(Value::as_str) == Some("sell_shares"));
    if bought && sold {
        return;
    }
    let _ = game;
    out.push(format!("{who} declines {} shares", if bought { "to sell" } else { "to buy" }));
}

/// The lines an action wrote.
pub fn explain(g: &SiteGame, b: &Before, raw: &Value, action: &Action) -> Vec<String> {
    let a = &g.game;
    let bg = &b.game;
    let mut out = Vec::new();
    let ent = action.entity_id().to_string();
    let who = ent_name(bg, &ent);
    let stock = bg.rnd().kind == RoundKind::Stock;
    match action {
        Action::Bid { company_sym, price, .. } => {
            out.push(format!("{who} bids {} for {}", money(a, *price), coname(a, company_sym)));
            if stock && bg.rnd().bid_actions + 1 >= bg.fam().bidding_tokens_per_action {
                let mut turn = b.turn.clone();
                turn.push(raw.clone());
                log_pass(a, &who, &turn, &mut out);
            }
        }
        Action::Pass { .. } if stock && matches!(bg.g1822_entity(&ent), crate::g1822::Ent::Player(_)) => {
            log_pass(a, &who, &b.turn, &mut out);
        }
        Action::Pass { .. } if bg.rnd().kind == RoundKind::Operating => {
            // Step::Base#log_pass: "X passes <the step's description>"
            if let Some(d) = b.step_label.as_deref() {
                out.push(format!("{who} passes {}", d.to_lowercase()));
            }
        }
        Action::BuyShares { corporation_sym, percent, .. } => {
            let price = spent(bg, a, &ent);
            let from = share_source(bg, raw);
            out.push(format!("{who} buys a {percent}% share of {corporation_sym} from {from} for {}", money(a, price)));
            if stock && matches!(bg.g1822_entity(&ent), crate::g1822::Ent::Player(_)) {
                let mut turn = b.turn.clone();
                turn.push(raw.clone());
                log_pass(a, &who, &turn, &mut out);
            }
        }
        Action::SellShares { corporation_sym, percent, share_indices, .. } => {
            let got = -spent(bg, a, &ent);
            let n = share_indices.len().max(1);
            let what = if n == 1 { format!("a {percent}% share") } else { format!("{n} shares") };
            let verb = if matches!(bg.g1822_entity(&ent), crate::g1822::Ent::Corp(_)) { "issues" } else { "sells" };
            out.push(format!("{who} {verb} {what} of {corporation_sym} and receives {}", money(a, got)));
        }
        Action::Par { corporation_sym, share_price, .. } => {
            out.push(format!("{who} pars {corporation_sym} at {}", money(a, *share_price)));
            let concession = bg.g().par_via_exchange.get(corporation_sym).cloned().or_else(|| {
                bg.fam().concessions.iter().find(|(_, m)| *m == corporation_sym).map(|(c, _)| c.to_string())
            });
            let paid = spent(bg, a, &ent);
            match concession.filter(|c| bg.co_owner(c).player_id().is_some()) {
                Some(c) if paid > 0 => out.push(format!("{who} exchanges {} and {} for a 20% share of {corporation_sym} from the Treasury", coname(a, &c), money(a, paid))),
                Some(c) => out.push(format!("{who} exchanges {} for a 20% share of {corporation_sym} from the Treasury", coname(a, &c))),
                None => out.push(format!("{who} buys a 20% share of {corporation_sym} from the Treasury for {}", money(a, paid))),
            }
            if stock {
                let mut turn = b.turn.clone();
                turn.push(raw.clone());
                log_pass(a, &who, &turn, &mut out);
            }
        }
        Action::LayTile { hex_id, tile_id, rotation, .. } => {
            let corp = g.operator_of(&ent);
            let cost = spent(bg, a, &corp);
            let name = tile_id.rsplit_once('-').map_or(tile_id.as_str(), |x| x.0);
            let spender = if matches!(bg.g1822_entity(&ent), crate::g1822::Ent::Company(_)) {
                let combo = raw.get("combo_entities").and_then(Value::as_array).map(|v| v.iter().filter_map(Value::as_str).collect::<Vec<_>>()).unwrap_or_default();
                let mut syms = vec![ent.clone()];
                syms.extend(combo.iter().map(|s| s.to_string()));
                format!("{corp} ({})", syms.join("+"))
            } else {
                corp.clone()
            };
            let spend = if cost > 0 { format!(" spends {} and", money(a, cost)) } else { String::new() };
            out.push(format!("{spender}{spend} lays tile #{name} with rotation {rotation} on {}", hex_desc(a, hex_id)));
        }
        Action::PlaceToken { hex_id, .. } => {
            let corp = g.operator_of(&ent);
            let cost = spent(bg, a, &corp);
            let price = if cost > 0 { format!(" for {}", money(a, cost)) } else { String::new() };
            out.push(format!("{corp} places a token on {}{price}", hex_desc(a, hex_id)));
        }
        Action::HexToken { hex_id, .. } => {
            out.push(format!("{who} places its destination token on {hex_id}"));
        }
        Action::RunRoutes { routes, .. } => {
            for r in routes {
                let train = bg.corporations.iter().flat_map(|c| c.trains.iter()).find(|t| t.id == r.train_name).map(|t| t.name.clone()).unwrap_or_default();
                let rs = raw
                    .get("routes")
                    .and_then(Value::as_array)
                    .and_then(|v| v.iter().find(|x| x.get("train").and_then(Value::as_str) == Some(r.train_name.as_str())))
                    .and_then(|x| x.get("revenue_str").and_then(Value::as_str))
                    .map(str::to_string)
                    .unwrap_or_else(|| r.hexes.join("-"));
                out.push(format!("{who} runs a {train} train for {}: {rs}", money(a, r.revenue.unwrap_or(0))));
            }
        }
        Action::Dividend { kind, .. } => {
            let revenue = bg.g1822_routes_revenue();
            let retained = cash_of(a, &ent) - cash_of(bg, &ent);
            match kind {
                DividendKind::Half => out.push(format!("{who} runs for {} and pays half", money(a, revenue))),
                DividendKind::Withhold if revenue == 0 => out.push(format!("{who} does not run")),
                _ => {}
            }
            if !matches!(kind, DividendKind::Payout) && retained > 0 && revenue > 0 {
                out.push(format!("{who} withholds {}", money(a, retained)));
            }
            payouts(bg, a, &ent, revenue, matches!(kind, DividendKind::Half), &mut out);
        }
        Action::BuyTrain { train_name, from, exchange, .. } => {
            let price = spent(bg, a, &ent);
            let name = a.corporations.iter().flat_map(|c| c.trains.iter()).find(|t| t.id == *train_name).map(|t| t.name.clone()).unwrap_or_default();
            if exchange.is_some() {
                out.push(format!("{who} upgrades an L train to a 2 train for {}", money(a, price)));
            } else {
                let source = if from == "depot" {
                    if bg.depot.discarded.iter().any(|t| t.id == *train_name) { "The Discard".to_string() } else { "The Depot".to_string() }
                } else {
                    from.clone()
                };
                out.push(format!("{who} buys a {name} train for {} from {source}", money(a, price)));
            }
        }
        Action::DiscardTrain { train_name, .. } => {
            let name = bg.corporations.iter().flat_map(|c| c.trains.iter()).find(|t| t.id == *train_name).map(|t| t.name.clone()).unwrap_or_default();
            out.push(format!("{who} discards {name}"));
        }
        Action::AcquireCompany { company_sym, .. } => {
            let owner = holder_name(bg, &bg.co_owner(company_sym));
            out.push(format!("{who} acquires {} from {owner}", coname(a, company_sym)));
        }
        Action::PayoffPlayerDebt { .. } | Action::PayoffPlayerDebtPartial { .. } => {
            let seat = ent.parse::<u32>().unwrap_or(0);
            let paid = bg.player_debt(seat) - a.player_debt(seat);
            out.push(if a.player_debt(seat) == 0 {
                format!("{who} pays off their loan of {}", money(a, paid))
            } else {
                format!("{who} decreases their loan by {} ({})", money(a, paid), money(a, a.player_debt(seat)))
            });
        }
        _ => {}
    }
    // a minor's dividend (G1822::Step::Dividend#skip!: half when it ran, else withhold) and the other companies'
    // automatic ones, within this action
    if !matches!(action, Action::Dividend { .. }) {
        for (sym, n) in &a.g().operating_history {
            if *n > bg.g().operating_history.get(sym).copied().unwrap_or(0) {
                let revenue = match action {
                    Action::RunRoutes { entity_id, routes, .. } if entity_id == sym => routes.iter().map(|r| r.revenue.unwrap_or(0)).sum(),
                    _ => 0,
                };
                let minor = a.is_minor(sym);
                if revenue > 0 && minor {
                    out.push(format!("{sym} runs for {} and pays half", money(a, revenue)));
                    let kept = revenue / 2;
                    if kept > 0 {
                        out.push(format!("{sym} withholds {}", money(a, kept)));
                    }
                    payouts(bg, a, sym, revenue, true, &mut out);
                } else if revenue == 0 {
                    out.push(format!("{sym} does not run"));
                }
            }
        }
    }
    let settled = bids_settled(bg, a, &mut out);
    floats_and_presidents(bg, a, &settled, &mut out);
    price_moves(bg, a, &mut out);
    phases(bg, a, &mut out);
    closes(bg, a, &mut out);
    transitions(bg, a, &mut out);
    out
}

/// What an entity's cash fell by (a player's or a corporation's).
fn spent(b: &BaseGame, a: &BaseGame, id: &str) -> i32 {
    cash_of(b, id) - cash_of(a, id)
}

fn cash_of(game: &BaseGame, id: &str) -> i32 {
    match game.g1822_entity(id) {
        crate::g1822::Ent::Player(p) => game.players.iter().find(|x| x.id == p).map_or(0, |x| x.cash),
        crate::g1822::Ent::Corp(c) => game.corp_idx.get(c.as_str()).map_or(0, |&ci| game.corporations[ci].cash),
        crate::g1822::Ent::Company(_) => 0,
    }
}

/// Where a bought share came from: "the Treasury" (the corporation's own), "the market".
fn share_source(b: &BaseGame, raw: &Value) -> String {
    let id = raw.get("shares").and_then(Value::as_array).and_then(|v| v.first()).and_then(Value::as_str).unwrap_or("");
    let Some((sym, idx)) = id.rsplit_once('_') else { return "the market".into() };
    let Some(&ci) = b.corp_idx.get(sym) else { return "the market".into() };
    match b.corporations[ci].shares.get(idx.parse::<usize>().unwrap_or(usize::MAX)) {
        Some(s) if s.owner.is_market() => "the market".into(),
        Some(_) => "the Treasury".into(),
        None => "the market".into(),
    }
}

/// "X pays out £R = £P per share (£a to A, £b to B)" (Ruby log_payout_shares; receivers by amount).
fn payouts(b: &BaseGame, a: &BaseGame, corp: &str, revenue: i32, half: bool, out: &mut Vec<String>) {
    let Some(&ci) = a.corp_idx.get(corp) else { return };
    let mut receivers: Vec<(i32, String)> = Vec::new();
    for p in &a.players {
        let before = b.players.iter().find(|x| x.id == p.id).map_or(p.cash, |x| x.cash);
        if p.cash > before {
            receivers.push((p.cash - before, p.name.clone()));
        }
    }
    if receivers.is_empty() {
        return;
    }
    receivers.sort_by(|x, y| y.0.cmp(&x.0));
    // (Ruby's total_shares: 100 / share_percent; a minor's director's certificate counts as two 50% shares)
    let shares = 100 / a.corporations[ci].share_unit_percent.max(1) as i32;
    let paid = if half { revenue - revenue / 2 } else { revenue };
    let per = paid as f64 / shares as f64;
    let per_s = if per.fract() == 0.0 { format!("{}", per as i64) } else { format!("{per}") };
    let list: Vec<String> = receivers.iter().map(|(x, n)| format!("{} to {n}", money(a, *x))).collect();
    out.push(format!("{corp} pays out {} = {}{per_s} per share ({})", money(a, paid), cur(a), list.join(", ")));
}

/// The bid boxes settled at the stock round's end (G1822::Round::Stock#finish_round): the concessions' and the
/// privates' winners in box order ("P wins the bid C for £X"), then the minors floated, the highest bid first (ties:
/// the earlier box): its par, its president's certificate, its float, what the bid paid beyond the treasury. Returns
/// the minors floated.
fn bids_settled(b: &BaseGame, a: &BaseGame, out: &mut Vec<String>) -> Vec<String> {
    let mut floated = Vec::new();
    if b.rnd().kind != RoundKind::Stock || (a.rnd().kind == RoundKind::Stock && a.turn == b.turn) {
        return floated;
    }
    let high = |co: &str| -> Option<crate::g1822::BidRec> { b.g1822_highest_bid(co) };
    for co in b.bidbox_concessions().iter().chain(b.bidbox_privates().iter()) {
        if let Some(h) = high(co) {
            out.push(format!("{} wins the bid {} for {}", pname(b, h.player), coname(b, co), money(a, h.price)));
        }
    }
    let minors = b.bidbox_minors();
    let mut wins: Vec<(i32, usize, String, crate::g1822::BidRec)> = Vec::new();
    for (i, co) in minors.iter().enumerate() {
        if let Some(h) = high(co) {
            wins.push((h.price, minors.len() - i, co.clone(), h));
        }
    }
    wins.sort_by(|x, y| (y.0, y.1).cmp(&(x.0, x.1)));
    for (_, _, co, h) in wins {
        let minor = BaseGame::minor_of_company(&co);
        let Some(&ci) = a.corp_idx.get(minor.as_str()) else { continue };
        let par = a.corporations[ci].ipo_price.as_ref().map_or(0, |s| s.price);
        let who = pname(b, h.player);
        out.push(format!("{who} pars {minor} at {}", money(a, par)));
        out.push(format!("{who} buys a 100% share of {minor} from the Treasury for {}", money(a, par * 2)));
        out.push(format!("{minor} floats"));
        let treasury = a.corporations[ci].cash;
        if h.price - treasury > 0 {
            out.push(format!("{who} pays {} to the bank", money(a, h.price - treasury)));
        }
        if treasury - par * 2 > 0 {
            out.push(format!("{minor} receives an additional {} from the bid", money(a, treasury - par * 2)));
        }
        floated.push(minor);
    }
    floated
}

/// Companies that closed.
fn closes(b: &BaseGame, a: &BaseGame, out: &mut Vec<String>) {
    for (i, c) in a.companies.iter().enumerate() {
        let was = &b.companies[i];
        if c.closed && !was.closed && !was.owner.is_none() && !a.g().deleted_open.contains(&c.sym) {
            out.push(format!("{} closes", coname(a, &c.sym)));
        }
    }
}
