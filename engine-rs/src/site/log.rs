//! Ruby's log lines (docs/WASM_ENGINE_SPIKE.md A3, `site_log`): engine-rs writes no log, so the site layer words each
//! processed action from the engine's state before and after it, in Ruby's words (the shapes the board's animations
//! parse, fx/beats.ts, first; then by frequency). The engine itself is untouched: self-play formats no strings.

use super::game::SiteGame;
use crate::entities::EntityId;
use crate::game::BaseGame;
use crate::rounds::{OperatingStep as S, Round};
use serde_json::Value;
use std::collections::HashMap;

/// What the log needs from the position before an action.
pub struct Before {
    pub game: BaseGame,
    auctioning: Option<String>,
    bids: HashMap<String, Vec<(u32, i32)>>,
    turn_acted: bool,
    turn_bought: bool,
    turn_sold: bool,
    /// The rounds finished before the action (the records' `store_player_info` entries)
    rounds_before: usize,
    /// The operating company's Buy Trains step has acted this turn (bought a train, or sold shares to): Ruby's
    /// step then never logs its skip
    buy_train_acted: bool,
}

pub fn before(g: &SiteGame) -> Before {
    // the position before without the map (the log reads the map only after the move)
    let mut game = g.game.clone_for_search_without_log();
    game.hexes = Vec::new();
    game.tile_counts_remaining = Default::default();
    game.site = None;
    let mut bids = HashMap::new();
    for c in game.auction_companies() {
        bids.insert(c.clone(), game.auction_bids(&c));
    }
    // Ruby's round.current_actions: BuySellParShares's own (a company's exchange is the Exchange step's)
    let own = |a: &&Value| a.get("entity_type").and_then(Value::as_str) != Some("company");
    let acted = g.turn_actions.iter().any(|a| own(&a));
    let bought = g.turn_actions.iter().filter(own).any(|a| matches!(a.get("type").and_then(Value::as_str), Some("buy_shares" | "par")));
    let sold = g.turn_actions.iter().filter(own).any(|a| a.get("type").and_then(Value::as_str) == Some("sell_shares"));
    let rounds_before = g.game.site.as_ref().map_or(0, |r| r.player_history.len());
    let buy_train_acted = g.turn_acted_at(Some("buy_train"), "BuyTrain") || g.turn_acted_at(Some("sell_shares"), "BuyTrain");
    Before { auctioning: game.auctioning_company(), bids, turn_acted: acted, turn_bought: bought, turn_sold: sold, rounds_before, buy_train_acted, game }
}

fn pname(game: &BaseGame, pid: u32) -> String {
    game.players.iter().find(|p| p.id == pid).map(|p| p.name.clone()).unwrap_or_default()
}

fn cname(game: &BaseGame, sym: &str) -> String {
    game.title_def().companies().iter().find(|c| c.sym == sym).map(|c| c.name.to_string()).unwrap_or_else(|| sym.to_string())
}

fn holder_name(game: &BaseGame, e: &EntityId) -> String {
    if let Some(p) = e.player_id() {
        pname(game, p)
    } else {
        e.corp_sym().unwrap_or("").to_string()
    }
}

fn hex_desc(game: &BaseGame, hex: &str) -> String {
    if game.title == "1867" {
        return match LOCATION_NAMES_1867.iter().find(|(h, _)| *h == hex) {
            Some((_, loc)) => format!("{hex} ({loc})"),
            None => hex.to_string(),
        };
    }
    match crate::title::g1830::location_names().get(hex) {
        Some(loc) if game.title == "1830" => format!("{hex} ({loc})"),
        _ => hex.to_string(),
    }
}

/// 1867's LOCATION_NAMES (g_1867/map.rb), as the log names a hex
const LOCATION_NAMES_1867: &[(&str, &str)] = &[
    ("D2", "Timmins ($80 if includes T/M/Q)"), ("D8", "Sudbury"), ("F8", "North Bay"), ("E13", "Barrie"),
    ("E15", "Guelph"), ("E17", "Hamilton"), ("D16", "Berlin"), ("C17", "London"), ("G15", "Peterborough"),
    ("I15", "Kingston"), ("J12", "Ottawa"), ("M9", "Trois-Rivières"), ("O7", "Quebec"), ("N12", "Sherbrooke"),
    ("C15", "Goderich"), ("B18", "Sarnia"), ("H14", "Belleville"), ("H10", "Pembroke"), ("K13", "Cornwall"),
    ("L10", "St. Jerome"), ("M13", "Granby"), ("L12", "Montreal"), ("F16", "Toronto"), ("A7", "Sault Ste. Marie"),
    ("F18", "Buffalo"), ("M15", "New England"), ("O13", "Maine"), ("P8", "Maritime Provinces"), ("A19", "Detroit"),
];

fn price_of(game: &BaseGame, sym: &str) -> Option<(i32, u8, u8)> {
    let c = game.corporations.iter().find(|c| c.sym == sym)?;
    c.share_price.as_ref().map(|s| (s.price, s.row, s.column))
}

/// "X's share price moves right from $A to $B" (one line per corporation whose price moved, Ruby's log_share_price).
fn price_moves(b: &BaseGame, a: &BaseGame, only: Option<&str>, out: &mut Vec<String>) {
    for c in &a.corporations {
        if only.is_some_and(|o| o != c.sym) {
            continue;
        }
        let (Some(p0), Some(p1)) = (price_of(b, &c.sym), price_of(a, &c.sym)) else { continue };
        if (p0.1, p0.2) == (p1.1, p1.2) {
            continue;
        }
        // Ruby log_share_price: any move to another cell, the same price included ("from $67 to $67"), its
        // directions up/down then left/right
        let mut dirs = Vec::new();
        if p1.1 < p0.1 {
            dirs.push("up");
        }
        if p1.1 > p0.1 {
            dirs.push("down");
        }
        if p1.2 < p0.2 {
            dirs.push("left");
        }
        if p1.2 > p0.2 {
            dirs.push("right");
        }
        out.push(format!("{}'s share price moves {} from ${} to ${}", c.sym, dirs.join(" and "), p0.0, p1.0));
    }
}

fn presidents(b: &BaseGame, a: &BaseGame, out: &mut Vec<String>) {
    for c in &a.corporations {
        let p0 = b.corporations.iter().find(|x| x.sym == c.sym).and_then(|x| x.president_id());
        let p1 = c.president_id();
        if let Some(p) = p1 {
            if p0 != p1 {
                out.push(format!("{} becomes the president of {}", pname(a, p), c.sym));
            }
        }
    }
}

fn floats(b: &BaseGame, a: &BaseGame, out: &mut Vec<String>) {
    for c in &a.corporations {
        let was = b.corporations.iter().find(|x| x.sym == c.sym).is_some_and(|x| x.floated);
        if c.floated && !was {
            out.push(format!("{} floats", c.sym));
            // (1867's companies are incremental: Ruby's float! pays nothing at once)
            if a.title != "1867" {
                out.push(format!("{} receives ${}", c.sym, c.cash - b.corporations.iter().find(|x| x.sym == c.sym).map_or(0, |x| x.cash)));
            }
        }
    }
}

/// What a company paid in a move: its cash before less after, and what the loans it took for it brought (1867's
/// AutomaticLoan: "X takes a loan and receives $45", then the lay or the station is paid).
fn spent_by(bg: &BaseGame, a: &BaseGame, corp: &str) -> i32 {
    let (Some(&i0), Some(&i1)) = (bg.corp_idx.get(corp), a.corp_idx.get(corp)) else { return 0 };
    let taken = a.corporations[i1].loans.saturating_sub(bg.corporations[i0].loans) as i32;
    let net = if a.title == "1867" { 45 } else { 0 };
    let mut spent = bg.corporations[i0].cash - a.corporations[i1].cash + taken * net;
    // (the rest of the turn skipped through its LoanOperations inside the move: the interest and the loans repaid
    // there aren't the lay's or the station's)
    if a.title == "1867" && loan_ops_ran(bg, a, corp) {
        let due = match &bg.round {
            Round::Operating(s0) => s0.interest_snapshot.get(corp).copied().unwrap_or(0) as i32,
            _ => 0,
        };
        let repaid = (bg.corporations[i0].loans as i32 + taken - a.corporations[i1].loans as i32).max(0);
        spent -= 5 * due + 50 * repaid;
    }
    spent
}

/// Whether the company's LoanOperations ran inside the move: it was operating before it, and its turn is past it now.
fn loan_ops_ran(bg: &BaseGame, a: &BaseGame, corp: &str) -> bool {
    let Round::Operating(s0) = &bg.round else { return false };
    if s0.current_corp_sym() != Some(corp) || matches!(s0.step, S::LoanOperations | S::DiscardTrain | S::BuyTrain | S::BuyCompany | S::Done) {
        return false;
    }
    match &a.round {
        Round::Operating(s1) if s1.round_num == s0.round_num && a.turn == bg.turn && s1.current_corp_sym() == Some(corp) => {
            matches!(s1.step, S::DiscardTrain | S::BuyTrain | S::BuyCompany | S::Done)
        }
        _ => true,
    }
}

/// Merge#move_assets' words: the minor's cash, companies, loans and trains as they move to the major.
fn receiving(bg: &BaseGame, minor: &str) -> String {
    receiving_in(bg, minor, None)
}

/// receiving() with the companies in the order the minor got them (Ruby's `corporation.companies`: the site layer's
/// record of when each changed hands)
fn receiving_in(bg: &BaseGame, minor: &str, seq: Option<&std::collections::HashMap<String, u64>>) -> String {
    let Some(&i) = bg.corp_idx.get(minor) else { return String::new() };
    let c = &bg.corporations[i];
    // (Merge#move_assets: the cash only when there is some; an array interpolated, as Opal writes it: "trains (2,2)")
    let mut parts = Vec::new();
    if c.cash > 0 {
        parts.push(format!("${}", c.cash));
    }
    let me = EntityId::corporation(minor);
    let mut owned: Vec<&crate::entities::Company> = bg.companies.iter().filter(|m| m.owner == me && !m.closed).collect();
    if let Some(seq) = seq {
        owned.sort_by_key(|m| seq.get(&m.sym).copied().unwrap_or(0));
    }
    let comps: Vec<String> = owned.iter().map(|m| cname(bg, &m.sym)).collect();
    if !comps.is_empty() {
        parts.push(format!("companies ({})", comps.join(", ")));
    }
    if c.loans > 0 {
        parts.push(format!("loans ({})", c.loans));
    }
    if !c.trains.is_empty() {
        parts.push(format!("trains ({})", c.trains.iter().map(|t| t.name.clone()).collect::<Vec<_>>().join(",")));
    }
    parts.join(", ")
}

/// Merge#finish_merge_to_major's lines: the merging minors in Corporation#<=> order (`@merging.sort`, on the prices
/// before), each one's owner given a share of the major (the presidency to the first who already holds 10% while it is
/// still in the treasury: "receives a 20% share", "becomes the president", "floats"), the minor's close and its assets
/// moved; then the token limit, or the major's tokens (`fix_token_count!`: "has tokens (3: 2 on hexes E15,F16 & 1 on
/// the charter)", Ruby's token order, the site layer's record of it).
fn merge_to_major_lines(g: &SiteGame, bg: &BaseGame, a: &BaseGame, target: &str, merging: &[String], out: &mut Vec<String>) {
    let mut held: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
    let mut presidency_open = true;
    for m in by_sort_order(bg, merging.to_vec()) {
        let Some(&mi) = bg.corp_idx.get(m.as_str()) else { continue };
        if let Some(p) = bg.corporations[mi].owner_id.player_id() {
            let nm = pname(a, p);
            let h = held.entry(p).or_insert(0);
            if presidency_open && *h == 10 {
                *h = 20;
                presidency_open = false;
                out.push(format!("{nm} receives a 20% share of {target}"));
                out.push(format!("{nm} becomes the president of {target}"));
                out.push(format!("{target} floats"));
            } else {
                *h += 10;
                out.push(format!("{nm} receives a 10% share of {target}"));
            }
        }
        out.push(format!("{m} closes"));
        out.push(format!("{m} merges into {target} receiving {}", receiving_in(bg, &m, Some(&g.company_seq))));
    }
    if matches!(&a.round, Round::Merger(s1) if s1.corporations_removing_tokens.is_some()) {
        out.push(format!("{target} will be above token limit and must decide which tokens to remove"));
    } else if let Some(&ti) = a.corp_idx.get(target) {
        let n = a.corporations[ti].tokens.len();
        let placed: Vec<String> = match g.token_cities.map.get(target) {
            Some(list) => list.iter().flatten().map(|(h, _)| h.clone()).collect(),
            None => a.corporations[ti].tokens.iter().filter(|t| t.used).map(|t| t.city_hex_id.clone()).collect(),
        };
        let charter = n.saturating_sub(placed.len());
        let on_charter = if charter > 0 { format!(" & {charter} on the charter") } else { String::new() };
        out.push(format!("{target} has tokens ({n}: {} on hexes {}{on_charter})", placed.len(), placed.join(",")));
    }
}

/// 1867's phase events after the phase line and the rusting (g_1867/game.rb event_*): the green minors, the majors,
/// the minors that can't start, the privates nationalized, and the trainless companies nationalized
/// (postevent_trainless_nationalization!, `nationalize!`'s lines).
fn phase_events_1867(g: &SiteGame, b: &Before, out: &mut Vec<String>) {
    let a = &g.game;
    let bg = &b.game;
    let fired = |name: &str| super::g1867::event_fired(a, name) && !super::g1867::event_fired(bg, name);
    if fired("green_minors_available") {
        out.push("Green minors are now available".into());
    }
    if fired("majors_can_ipo") {
        out.push("Majors can now be started".into());
    }
    // (event_minors_cannot_start!: only when an unstarted minor is taken out)
    let minor = |c: &crate::entities::Corporation| c.corp_type == crate::title::CorpType::Minor && !c.closed;
    if fired("minors_cannot_start") && bg.corporations.iter().any(|c| minor(c) && !c.floated && c.president_id().is_none()) {
        out.push("Minors can no longer be started".into());
    }
    if fired("nationalize_companies") {
        out.push("-- Event: Private companies are nationalized --".into());
        for (i, c) in a.companies.iter().enumerate() {
            let was = &bg.companies[i];
            if c.sym == "3" || was.closed || was.owner.corp_sym() == Some("CN") {
                continue;
            }
            if c.owner.corp_sym() == Some("CN") {
                out.push(format!("{} nationalized from {} for ${}", cname(a, &c.sym), holder_name(bg, &was.owner), c.value));
            }
        }
    }
    // (event_minors_nationalized!: only when a minor is left)
    if fired("minors_nationalized") && bg.corporations.iter().any(|c| minor(c) && c.floated) {
        out.push("Minors nationalized".into());
    }
    // the companies nationalized now: minors closed, majors started afresh, each with its lines
    let gone: Vec<String> = bg
        .corporations
        .iter()
        .filter(|c| c.floated && c.corp_type != crate::title::CorpType::National)
        .filter(|c| {
            a.corp_idx.get(c.sym.as_str()).is_some_and(|&i| {
                let n = &a.corporations[i];
                n.closed && c.corp_type == crate::title::CorpType::Minor || (!n.closed && n.ipo_price.is_none())
            })
        })
        .map(|c| c.sym.clone())
        .collect();
    for sym in by_sort_order(bg, gone) {
        nationalize_lines(g, bg, &sym, out);
    }
}

/// 1867's auctions (G1867::Step::SingleItemAuction, BuySellParSharesViaBid): the lines of a bid or a pass, and of
/// the auction settling. Returns false where step 1's lines apply.
fn auction_lines_1867(g: &SiteGame, b: &Before, raw: &Value, out: &mut Vec<String>) -> bool {
    let a = &g.game;
    let bg = &b.game;
    let t = raw_type(raw);
    let ent = super::json::entity_of(raw).unwrap_or_default();
    let Ok(pid) = ent.parse::<u32>() else { return false };
    let who = pname(a, pid);
    let price = raw.get("price").and_then(Value::as_i64).unwrap_or(0);
    let cash_of = |game: &BaseGame, p: u32| game.players.iter().find(|x| x.id == p).map_or(0, |x| x.cash);
    match (&bg.round, t) {
        (Round::Auction(s0), "bid" | "pass") if s0.single_item.is_some() => {
            let Some(ci) = s0.auctioning else { return false };
            let name = cname(a, &bg.companies[ci].sym);
            let dutch = s0.single_item.as_ref().is_some_and(|x| x.dutch_mode);
            let had_bids = s0.bids.get(&ci).is_some_and(|v| !v.is_empty());
            if t == "bid" && !dutch {
                out.push(format!("{who} bids ${price} for {name}"));
                // PassableAuction#add_bid: those who can't afford the next bid leave the auction, in its order
                let after: Vec<u32> = match &a.round {
                    Round::Auction(s1) if s1.auctioning == Some(ci) => s1.single_item.as_ref().map(|x| x.active_bidders.clone()).unwrap_or_default(),
                    _ => Vec::new(),
                };
                let still = matches!(&a.round, Round::Auction(s1) if s1.auctioning == Some(ci));
                if still {
                    for p in s0.single_item.as_ref().map(|x| x.active_bidders.clone()).unwrap_or_default() {
                        if p != pid && !after.contains(&p) {
                            out.push(format!("{} cannot bid ${} and is out of the auction for {name}", pname(a, p), price + 5));
                        }
                    }
                }
            } else if t == "pass" {
                if had_bids || dutch {
                    out.push(format!("{who} passes on {name}"));
                } else {
                    out.push(format!("{who} declined to bid on {name}"));
                }
            }
            // the lot settled: its winner, the next lot up; or the price drops (enter_dutch)
            let won = a.companies[ci].owner.player_id();
            if let Some(w) = won.filter(|_| bg.companies[ci].owner.player_id().is_none()) {
                let paid = cash_of(bg, w) - cash_of(a, w);
                out.push(format!("{} wins the auction for {name} with a bid of ${paid}", pname(a, w)));
                if let Round::Auction(s1) = &a.round {
                    if let Some(n) = s1.auctioning {
                        out.push(format!("{} is up for auction, minimum bid is ${}", cname(a, &a.companies[n].sym), a.sa_min_bid(s1, n)));
                    }
                }
            } else if let Round::Auction(s1) = &a.round {
                let d0 = bg.companies[ci].discount;
                let d1 = a.companies[ci].discount;
                if d1 > d0 {
                    let tense = if dutch { "bought" } else { "bid on" };
                    out.push(format!("Nobody {tense} {name}, reducing price to ${}", a.companies[ci].min_bid()));
                }
                let _ = s1;
            }
            true
        }
        (Round::Stock(s0), "bid" | "pass") if t == "bid" || s0.bid_auction.is_some() => {
            let sym = raw.get("corporation").and_then(Value::as_str).map(str::to_string)
                .or_else(|| s0.bid_auction.as_ref().map(|x| x.corporation_sym.clone()))
                .unwrap_or_default();
            if t == "bid" && s0.bid_auction.is_none() {
                out.push(format!("{who} auctions {sym} for ${price}"));
                // HOME_TOKEN_TIMING :par: the minor's station is placed as it goes up (place_home_token)
                if matches!(&a.round, Round::Stock(s1) if s1.pending_home_tokens.iter().any(|(c, _)| c == &sym)) {
                    out.push(format!("{sym} must choose city for home token"));
                }
            } else if t == "bid" {
                out.push(format!("{who} bids ${price} for {sym}"));
            } else {
                out.push(format!("{who} passes on {sym}"));
            }
            // the players who can't raise the bid leave the auction (PassableAuction#resolve_bids)
            let still = matches!(&a.round, Round::Stock(s1) if s1.bid_auction.is_some());
            let high = match &a.round {
                Round::Stock(s1) => s1.bid_auction.as_ref().and_then(|x| x.bids.iter().map(|b| b.price).max()),
                _ => None,
            }
            .unwrap_or(price as i32);
            let before: Vec<u32> = s0.bid_auction.as_ref().map(|x| x.active_bidders.clone()).unwrap_or_else(|| s0.player_order.clone());
            let after: Vec<u32> = match &a.round {
                Round::Stock(s1) => s1.bid_auction.as_ref().map(|x| x.active_bidders.clone()).unwrap_or_default(),
                _ => Vec::new(),
            };
            let owner = a.corp_idx.get(sym.as_str()).and_then(|&ci| a.corporations[ci].owner_id.player_id());
            for p in before {
                if p == pid || Some(p) == owner || after.contains(&p) {
                    continue;
                }
                if still || owner.is_some() {
                    out.push(format!("{} cannot bid ${} and is out of the auction for {sym}", pname(a, p), high + 5));
                }
            }
            if let (false, Some(w)) = (still, owner) {
                let ci = a.corp_idx[sym.as_str()];
                let par = a.corporations[ci].share_price.as_ref().map_or(0, |s| s.price);
                let won_at = a.corporations[ci].cash;
                out.push(format!("{} wins bid on {sym} for ${won_at}", pname(a, w)));
                out.push(format!("{} pars {sym} at ${par}", pname(a, w)));
                out.push(format!("{} buys a 100% share of {sym} from the Treasury for ${}", pname(a, w), 2 * par));
                out.push(format!("{} becomes the president of {sym}", pname(a, w)));
                out.push(format!("{sym} floats"));
            }
            true
        }
        _ => false,
    }
}

fn phase_line(game: &BaseGame) -> String {
    let def = game.title_def().phases().into_iter().find(|p| p.name == game.phase.name);
    let tiles: Vec<String> = game
        .phase
        .tiles
        .iter()
        .map(|t| {
            let mut c = t.chars();
            c.next().map(|f| f.to_uppercase().collect::<String>() + c.as_str()).unwrap_or_default()
        })
        .collect();
    format!(
        "-- Phase {} (Operating Rounds: {} | Train Limit: {} | Available Tiles: {}) --",
        game.phase.name,
        def.map_or(game.phase.operating_rounds, |d| d.operating_rounds),
        if game.title == "1867" {
            // the phase's train limit by kind (Ruby's PHASES hash, "minor: 2, major: 4")
            super::g1867::view::phase_limits(&game.phase.name).iter().map(|(k, l)| format!("{k}: {l}")).collect::<Vec<_>>().join(", ")
        } else {
            game.phase.train_limit.to_string()
        },
        tiles.join(", ")
    )
}

/// The game's opening lines (before the first action).
pub fn start(g: &SiteGame) -> Vec<String> {
    // Ruby Game::Base#initialize: init_phase (the phase's line) before log_optional_rules
    let mut out = vec![phase_line(&g.game)];
    let rules: &[(&str, &str)] = &[
        ("multiple_brown_from_ipo", "Buy Multiple Brown Shares From IPO: Multiple brown shares may be bought from IPO as well as from pool"),
        ("optional_6_train", "Optional extra 6-Train: Adds a 3rd 6-train"),
    ];
    let used: Vec<&str> = rules.iter().filter(|(k, _)| g.game.optional_rules.iter().any(|r| r == k)).map(|(_, t)| *t).collect();
    if !used.is_empty() {
        out.push("Optional rules used in this game:".to_string());
        for t in used {
            out.push(format!(" * {t}"));
        }
    }
    // 1867's SingleItemAuction#setup: the first lot goes up (auction_entity_log)
    if let Round::Auction(s) = &g.game.round {
        if let (Some(ci), true) = (s.auctioning, s.single_item.is_some()) {
            out.push(format!("{} is up for auction, minimum bid is ${}", cname(&g.game, &g.game.companies[ci].sym), g.game.sa_min_bid(s, ci)));
        }
    }
    out
}

fn op_corp(game: &BaseGame) -> Option<(String, S, usize, u8)> {
    match &game.round {
        Round::Operating(s) => s.current_corp_sym().map(|c| (c.to_string(), s.step.clone(), s.entity_index, s.round_num)),
        _ => None,
    }
}

const STEP_ORDER: &[S] = &[S::LayTile, S::PlaceToken, S::RunRoutes, S::Dividend, S::BuyTrain, S::BuyCompany];
/// 1867's operating steps that log (g_1867/game.rb operating_round: RedeemShares first, BuyCompanyPreloan and
/// LoanOperations between Dividend and BuyTrain)
const STEP_ORDER_1867: &[S] = &[
    S::RedeemShares, S::LayTile, S::PlaceToken, S::RunRoutes, S::Dividend, S::BuyCompanyPreloan, S::LoanOperations,
    S::BuyTrain, S::BuyCompany,
];
thread_local! {
    /// The loans the move itself took (AutomaticLoan), written with the move: LoanOperations' own are the rest
    static MOVE_LOANS: std::cell::RefCell<(String, u32)> = const { std::cell::RefCell::new((String::new(), 0)) };
    /// The title whose step list the log walks (set by `explain` / `turn_start`): 1830's unless 1867's
    static IS_1867: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

fn steps_now() -> &'static [S] {
    if IS_1867.with(|c| c.get()) { STEP_ORDER_1867 } else { STEP_ORDER }
}

fn step_pos(s: &S) -> usize {
    let order = steps_now();
    order.iter().position(|x| x == s).unwrap_or(order.len())
}

/// The run's revenue for a 1867 minor's automatic payout (the run_routes being explained, else 0).
fn run_revenue(raw: &Value, corp: &str) -> i32 {
    if raw.get("type").and_then(Value::as_str) != Some("run_routes") || super::json::entity_of(raw).ok().as_deref() != Some(corp) {
        return 0;
    }
    raw.get("routes").and_then(Value::as_array).map_or(0, |rs| rs.iter().filter_map(|r| r.get("revenue").and_then(Value::as_i64)).sum::<i64>() as i32)
}

/// The skip lines of a corporation's steps from `from` (inclusive) to `to` (exclusive): Ruby's log_skip and the
/// dividend step's automatic withhold (1867: a minor's automatic payout, `run_rev` its run's revenue).
fn skips(g: &SiteGame, b: &BaseGame, corp: &str, from: usize, to: usize, out: &mut Vec<String>) {
    skips_with(g, b, corp, from, to, 0, out)
}

fn skips_with(g: &SiteGame, b: &BaseGame, corp: &str, from: usize, to: usize, run_rev: i32, out: &mut Vec<String>) {
    let a = &g.game;
    let order = steps_now();
    let is_1867 = a.title == "1867";
    let minor = is_1867 && a.corp_idx.get(corp).is_some_and(|&i| a.corporations[i].corp_type == crate::title::CorpType::Minor);
    for s in order.iter().take(to.min(order.len())).skip(from) {
        match s {
            // G1867 RedeemShares#log_skip and Token#log_skip: majors only
            S::RedeemShares => {
                if !minor {
                    out.push(format!("{corp} skips redeem shares"));
                }
            }
            S::PlaceToken if minor => {}
            S::BuyCompanyPreloan => out.push(format!("{corp} skips buy companies")),
            S::LoanOperations => {
                // InterestOnLoans#pay_interest!: "X pays $N interest for K loans" on the loans due (the count the
                // operating round took, `@interest`: a loan taken since owes nothing yet; nothing owed, nothing said);
                // not when it can't pay (nationalized: `interest_unpaid!`)
                let loans = b.corp_idx.get(corp).map_or(0, |&i| b.corporations[i].loans);
                let due = match &b.round {
                    Round::Operating(s0) => s0.interest_snapshot.get(corp).copied().unwrap_or(0),
                    _ => 0,
                };
                let reset = a.corp_idx.get(corp).is_some_and(|&i| a.corporations[i].ipo_price.is_none() || a.corporations[i].closed);
                let left = a.corp_idx.get(corp).map_or(0, |&i| a.corporations[i].loans);
                // the loans it takes to pay (pay_interest!: while it owes more than it has; take_loan's line), beyond
                // those the move took itself
                let loans = loans + MOVE_LOANS.with(|c| { let m = c.borrow(); if m.0 == corp { m.1 } else { 0 } });
                for _ in loans..left {
                    out.push(format!("{corp} takes a loan and receives $45"));
                }
                if due > 0 && !reset {
                    // (InterestOnLoans#pay_interest!'s words, its empty field and all)
                    out.push(format!("{corp} pays ${} interest for  {due} loan{}", 5 * due, if due == 1 { "" } else { "s" }));
                } else if due > 0 && reset {
                    // LoanOperations#interest_unpaid!: what it owes and has, then nationalize! (G1867 nationalizes it)
                    let has = b.corp_idx.get(corp).map_or(0, |&i| b.corporations[i].cash);
                    out.push(format!("{corp} owes ${} in loan interest but only has ${has}", 5 * due));
                    nationalize_lines(g, b, corp, out);
                }
                // LoanOperations#skip!: then the loans it can pay off (repay_loan; a company nationalized for its
                // interest has said what it paid off: nationalize_lines)
                if !reset {
                    for _ in left..loans {
                        out.push(format!("{corp} pays off a loan for $50"));
                    }
                }
            }
            S::Dividend if minor && run_rev > 0 => {
                // G1867::Step::Dividend#skip!: a minor pays half to its treasury, half to its owner
                let half = run_rev / 2;
                out.push(format!("{corp} withholds ${half}"));
                if let Some(p) = a.corp_idx.get(corp).and_then(|&i| a.corporations[i].owner_id.player_id()) {
                    out.push(format!("{} receives ${}", pname(a, p), run_rev - half));
                }
                price_moves(b, a, Some(corp), out);
            }
            S::LayTile => out.push(format!("{corp} skips lay track")),
            S::PlaceToken => out.push(format!("{corp} skips place a token")),
            S::RunRoutes => out.push(format!("{corp} skips run routes")),
            S::Dividend => {
                out.push(format!("{corp} does not run"));
                price_moves(b, a, Some(corp), out);
            }
            S::BuyTrain => {
                // (the skip comes before an export at the round's end and the rusting it sets off: then the company's
                // trains and limit as they were)
                let n_b = b.corp_idx.get(corp).map_or(0, |&i| b.corporations[i].trains.len());
                let n_a = a.corp_idx.get(corp).map_or(0, |&i| a.corporations[i].trains.len());
                let g = if is_1867 && a.phase.name != b.phase.name && n_a < n_b { b } else { a };
                let limit = if is_1867 {
                    g.corp_idx.get(corp).map_or(g.phase.train_limit as usize, |&i| g.corp_train_limit(i))
                } else {
                    g.phase.train_limit as usize
                };
                let full = g.corp_idx.get(corp).is_some_and(|&i| g.corporations[i].trains.len() >= limit);
                if full {
                    out.push(format!("{corp} automatically skips buy trains because it has no room for more trains"));
                } else {
                    out.push(format!("{corp} skips buy trains"));
                }
            }
            S::BuyCompany => out.push(format!("{corp} skips buy companies")),
            _ => {}
        }
    }
}

/// The lines of one processed action (`raw`: the site JSON as recorded).
pub fn explain(g: &SiteGame, b: &Before, raw: &Value) -> Vec<String> {
    IS_1867.with(|c| c.set(g.game.title == "1867"));
    let a = &g.game;
    let bg = &b.game;
    let mut out = Vec::new();
    let t = raw.get("type").and_then(Value::as_str).unwrap_or("");
    let ent = super::json::entity_of(raw).unwrap_or_default();
    let pid: Option<u32> = ent.parse().ok();
    let who = pid.map(|p| pname(a, p)).unwrap_or_else(|| ent.clone());
    // the acting player (a corporation's president) and Ruby's "via Master Mode" line
    let acting = pid.or_else(|| bg.corp_idx.get(ent.as_str()).and_then(|&i| bg.corporations[i].president_id()))
        .or_else(|| {
            // a company: its owner, or its owning corporation's president
            let owner = &bg.companies[*bg.company_idx.get(ent.as_str())?].owner;
            owner.player_id().or_else(|| owner.corp_sym().and_then(|c| bg.corp_idx.get(c)).and_then(|&i| bg.corporations[i].president_id()))
        });
    // (Ruby Action::Base.from_h keeps the user only when the entity has a player and it isn't them: an entity nobody
    // owns, a minor at auction choosing its home city, writes no such line)
    if let Some(user) = raw.get("user").and_then(Value::as_u64) {
        if acting.is_some_and(|p| p != user as u32) {
            let name = g.players.iter().find(|p| p.0 == user as u32).map(|p| p.1.clone()).unwrap_or_else(|| "Owner".into());
            out.push(format!("• Action({t}) via Master Mode by: {name}"));
        }
    }
    let n = out.len();
    // where Ruby's Bank#spend writes "-- The bank has broken --" (the spend that breaks it), and where a private's
    // close on its corporation's first train goes (Game::Base#buy_train, before the phase)
    let mut bank_at: Option<usize> = None;
    let mut bought_at: Option<usize> = None;
    let price = raw.get("price").and_then(Value::as_i64).unwrap_or(0);
    let auction_1867 = a.title == "1867" && auction_lines_1867(g, b, raw, &mut out);
    let mut pending_home: Option<String> = None;
    match (t, &bg.round) {
        _ if auction_1867 => {}
        ("bid", Round::Auction(_)) => {
            let comp = raw.get("company").and_then(Value::as_str).unwrap_or("");
            let first = bg.auction_companies().first().cloned();
            let bought = a.company_idx.get(comp).is_some_and(|&i| a.companies[i].owner.player_id() == pid);
            if bought && first.as_deref() == Some(comp) && b.auctioning.is_none() && b.bids.get(comp).map_or(true, |v| v.is_empty()) {
                out.push(format!("{who} buys {} for ${price}", cname(a, comp)));
                if let Some(share) = crate::abilities::share_grant(&a.title, comp) {
                    let (corp, idx) = share.rsplit_once('_').unwrap_or((share, "1"));
                    if idx != "0" {
                        out.push(format!("{who} receives a 10% share of {corp}"));
                    }
                }
            } else {
                out.push(format!("{who} bids ${price} for {}", cname(a, comp)));
            }
        }
        ("pass", Round::Auction(_)) => match &b.auctioning {
            Some(c) => out.push(format!("{who} passes on {}", cname(a, c))),
            None => {
                out.push(format!("{who} passes bidding"));
                // WaterfallAuction#all_passed!: every player has passed -- the cheapest private's minimum bid
                // drops (increase_discount!) while it is unsold, else the privates pay (payout_companies)
                if let Some(c) = bg.auction_companies().first() {
                    let (m0, m1) = (bg.auction_min_bid(c), a.auction_min_bid(c));
                    if m1 < m0 && a.company_idx.get(c.as_str()).is_some_and(|&i| a.companies[i].owner.player_id().is_none()) {
                        out.push(format!("{} minimum bid decreases from ${m0} to ${m1}", cname(a, c)));
                    }
                }
                let paid = a.players.iter().any(|p| bg.players.iter().any(|q| q.id == p.id && p.cash > q.cash));
                if paid {
                    for p in super::view::player_order(g) {
                        let mut list: Vec<(i32, String)> = a
                            .companies
                            .iter()
                            .filter(|c| c.owner == EntityId::player(p) && !c.closed && c.revenue > 0)
                            .map(|c| (c.revenue, cname(a, &c.sym)))
                            .collect();
                        list.sort();
                        for (rev, name) in list {
                            out.push(format!("{} collects ${rev} from {name}", pname(a, p)));
                        }
                    }
                }
            }
        },
        ("par", _) => {
            let corp = raw.get("corporation").and_then(Value::as_str).unwrap_or("");
            let p = a.corporations.iter().find(|c| c.sym == corp).and_then(|c| c.ipo_price.as_ref().map(|s| s.price)).unwrap_or(0);
            out.push(format!("{who} pars {corp} at ${p}"));
            if matches!(bg.round, Round::Auction(_)) {
                out.push(format!("{who} receives a 20% share of {corp}"));
            } else {
                let ipo = if a.title == "1867" { "the Treasury" } else { "the IPO" };
                out.push(format!("{who} buys a 20% share of {corp} from {ipo} for ${}", 2 * p));
                if a.title == "1867" && matches!(&a.round, Round::Stock(s1) if s1.pending_home_tokens.iter().any(|(c, _)| c == corp)) {
                    pending_home = Some(corp.to_string());
                }
            }
        }
        ("buy_shares", _) if raw.get("entity_type").and_then(Value::as_str) == Some("company") => {
            let shares = raw.get("shares").and_then(Value::as_array).cloned().unwrap_or_default();
            let sid = shares.first().and_then(Value::as_str).unwrap_or("");
            let corp = sid.rfind('_').map_or("", |p| &sid[..p]);
            let owner = bg.company_idx.get(ent.as_str()).map(|&i| holder_name(bg, &bg.companies[i].owner)).unwrap_or_default();
            let from = bg.corporations.iter().find(|c| c.sym == corp).and_then(|c| c.shares.iter().find(|s| format!("{}_{}", corp, s.index) == sid)).map_or("the IPO", |s| if s.owner.is_market() { "the market" } else { "the IPO" });
            out.push(format!("{owner} exchanges {} for a 10% share of {corp} from {from}", cname(a, &ent)));
        }
        ("buy_shares", _) => {
            let shares = raw.get("shares").and_then(Value::as_array).cloned().unwrap_or_default();
            let sid = shares.first().and_then(Value::as_str).unwrap_or("");
            let corp = sid.rfind('_').map_or("", |p| &sid[..p]);
            let pct = raw.get("percent").and_then(Value::as_i64).unwrap_or(10);
            let from_market = bg.corporations.iter().find(|c| c.sym == corp).and_then(|c| c.shares.iter().find(|s| format!("{}_{}", corp, s.index) == sid)).is_some_and(|s| s.owner.is_market());
            let corp_b = bg.corporations.iter().find(|c| c.sym == corp);
            // (a company always_market_price, 1867's, sells its treasury shares at the market price)
            let at_market = from_market || corp_b.is_some_and(|c| c.always_market_price);
            let unit = if at_market { corp_b.and_then(|c| c.share_price.as_ref()).map_or(0, |s| s.price) } else { corp_b.and_then(|c| c.ipo_price.as_ref()).map_or(0, |s| s.price) };
            let paid = unit * pct as i32 / 10;
            // Ruby ipo_name: 1867's IPO is its "Treasury"
            let ipo = if a.title == "1867" { "the Treasury" } else { "the IPO" };
            if a.title == "1867" && raw.get("entity_type").and_then(Value::as_str) == Some("corporation") {
                // G1867::Step::RedeemShares: the company buys its own share back
                out.push(format!("{ent} redeems a {pct}% share  from the market for ${paid}"));
            } else {
                out.push(format!("{who} buys a {pct}% share of {corp} from {} for ${paid}", if from_market { "the market" } else { ipo }));
            }
        }
        ("sell_shares", _) => {
            let shares = raw.get("shares").and_then(Value::as_array).cloned().unwrap_or_default();
            let sid = shares.first().and_then(Value::as_str).unwrap_or("");
            let corp = sid.rfind('_').map_or("", |p| &sid[..p]);
            let pct = raw.get("percent").and_then(Value::as_i64).unwrap_or(10);
            let got = pid.map_or(0, |p| a.players.iter().find(|x| x.id == p).map_or(0, |x| x.cash) - bg.players.iter().find(|x| x.id == p).map_or(0, |x| x.cash));
            let what = if pct == 10 { "a 10% share".to_string() } else { format!("{} shares", pct / 10) };
            out.push(format!("{who} sells {what} of {corp} and receives ${got}"));
            bank_at = Some(out.len());
        }
        ("pass", Round::Stock(_)) => {
            if !b.turn_acted {
                out.push(format!("{who} passes"));
            } else if b.turn_bought && !b.turn_sold {
                out.push(format!("{who} declines to sell shares"));
            } else if b.turn_sold && !b.turn_bought {
                out.push(format!("{who} declines to buy shares"));
            }
        }
        ("lay_tile", _) => {
            let hex = raw.get("hex").and_then(Value::as_str).unwrap_or("");
            let tile = raw.get("tile").and_then(Value::as_str).unwrap_or("");
            let name = super::json::tile_name(tile).unwrap_or(tile);
            let rot = raw.get("rotation").and_then(Value::as_i64).unwrap_or(0);
            let corp = g.operator_of(&ent);
            let spent = spent_by(bg, a, &corp);
            let spend = if spent > 0 { format!("spends ${spent} and ") } else { String::new() };
            // Ruby Tracker#pay_tile_cost!: a private's lay names it, "C&O (CS) lays tile ..."
            let via = if corp != ent { format!(" ({ent})") } else { String::new() };
            out.push(format!("{corp}{via} {spend}lays tile #{name} with rotation {rot} on {}", hex_desc(a, hex)));
            // G1867::Game#place_639_token: Montreal's green tile takes the national's reserved station
            if a.title == "1867" {
                let used = |game: &BaseGame| game.corp_idx.get("CN").map_or(0, |&i| game.corporations[i].tokens.iter().filter(|t| t.used).count());
                if used(a) > used(bg) {
                    let place = LOCATION_NAMES_1867.iter().find(|(h, _)| *h == hex).map_or(hex, |x| x.1);
                    out.push(format!("CN places a token on {place}"));
                }
            }
        }
        ("place_token", _) => {
            let corp = g.operator_of(&ent);
            let city = raw.get("city").and_then(Value::as_str).unwrap_or("");
            // the hex the token went to: the city's tile
            let hex = a.hexes.iter().find(|h| city.starts_with(&format!("{}-", h.tile.name)) || city.starts_with(&format!("{}-0-", h.id))).map(|h| h.id.clone()).unwrap_or_default();
            let spent = spent_by(bg, a, &corp);
            // Ruby: "places a token on <hex> (<location>) <price>" (the price part empty for a free one)
            let price_part = if spent > 0 { format!(" for ${spent}") } else { String::new() };
            // Ruby Tokener#place_token: a private's ability names it, "B&O (DH) places a token on ..."
            let via = if corp != ent { format!(" ({ent})") } else { String::new() };
            out.push(format!("{corp}{via} places a token on {} {price_part}", hex_desc(a, &hex)));
        }
        ("run_routes", _) => {
            for r in raw.get("routes").and_then(Value::as_array).cloned().unwrap_or_default() {
                let train = r.get("train").and_then(Value::as_str).unwrap_or("");
                let tname = train.rsplit_once('-').map_or(train, |x| x.0);
                let rev = r.get("revenue").and_then(Value::as_i64).unwrap_or(0);
                // Ruby revenue_str (rl18xx-next c57803b): Route#visited_hexes, the stops' hexes as the train reaches them,
                // a hex stopped at twice (both cities of an OO hex) twice; a route stored with fewer hexes than stops
                // takes them from the track (Action::RunRoutes.h_to_args): its nodes' hexes
                let hexes: Vec<&str> = r.get("hexes").and_then(Value::as_array).map(|h| h.iter().filter_map(Value::as_str).collect()).unwrap_or_default();
                let nodes: Vec<&str> = r.get("nodes").and_then(Value::as_array).map(|h| h.iter().filter_map(Value::as_str).collect()).unwrap_or_default();
                // (Route#revenue_str: the stored string unless stale, else Game#revenue_str from the track)
                let stale = !hexes.is_empty() && !nodes.is_empty() && hexes.len() < nodes.len();
                let stored = r.get("revenue_str").and_then(Value::as_str).filter(|x| !x.is_empty());
                let s = match stored {
                    Some(x) if !stale => x.to_string(),
                    _ => super::track::visited_hexes(g, &r)
                        .or_else(|| (!nodes.is_empty()).then(|| nodes.iter().map(|n| n.rsplit_once('-').map_or(*n, |x| x.0)).collect::<Vec<_>>().join("-")))
                        .unwrap_or_else(|| hexes.join("-")),
                };
                out.push(format!("{ent} runs a {tname} train for ${rev}: {s}"));
            }
        }
        ("dividend", _) if a.title == "1867" => {
            // G1867::Step::Dividend (HalfPay): payout, half (half withheld, to the $10 below, the rest paid out per
            // share) or withhold; every share not a player's pays the company (its treasury's and the market's)
            let kind = raw.get("kind").and_then(Value::as_str).unwrap_or("");
            let rev = match &bg.round {
                Round::Operating(s) => s.revenue,
                _ => 0,
            };
            let withheld = match kind {
                "payout" => 0,
                "half" => rev / 2 / 10 * 10,
                _ => rev,
            };
            if kind == "half" {
                out.push(format!("{ent} runs for ${rev} and pays half"));
            }
            if withheld > 0 {
                out.push(format!("{ent} withholds ${withheld}"));
            }
            if kind != "withhold" {
                let paid = rev - withheld;
                // Dividend#dividends_for_entity: (shares * per share).ceil, the per share a Float
                let amount = |n: i32| (n * paid + 9).div_euclid(10);
                let mut pays: Vec<(i32, usize, String)> = Vec::new();
                let mut corp_shares = 0;
                if let Some(&ci) = bg.corp_idx.get(ent.as_str()) {
                    for (k, pid2) in super::view::player_order(g).into_iter().enumerate() {
                        let n: i32 = bg.corporations[ci].shares.iter().filter(|s| s.owner == EntityId::player(pid2)).map(|s| s.percent as i32 / 10).sum();
                        if amount(n) > 0 {
                            pays.push((-amount(n), k, pname(a, pid2)));
                        }
                    }
                    // (Dividend#holder_for_corporation: an incremental company is paid for its treasury's shares; the
                    // market's pay nobody)
                    corp_shares = bg.corporations[ci].shares.iter().filter(|s| s.owner.is_ipo() || s.owner == EntityId::corporation(&ent)).map(|s| s.percent as i32 / 10).sum::<i32>();
                }
                if amount(corp_shares) > 0 {
                    pays.push((-amount(corp_shares), usize::MAX, ent.clone()));
                }
                pays.sort();
                let list: Vec<String> = pays.iter().map(|(amt, _, nm)| format!("${} to {nm}", -amt)).collect();
                // (Dividend#payout_per_share is a Float: Ruby writes "$19.0", the site's Opal page "$19", as JavaScript
                // prints a whole number; Rust pages write Opal's)
                let f = paid as f64 / 10.0;
                let per_s = if f.fract() == 0.0 { format!("{}", f as i64) } else { format!("{f}") };
                bank_at = Some(out.len());
                out.push(format!("{ent} pays out ${paid} = ${per_s} per share ({})", list.join(", ")));
            }
            price_moves(bg, a, Some(&ent), &mut out);
        }
        ("dividend", _) => {
            let kind = raw.get("kind").and_then(Value::as_str).unwrap_or("");
            let rev = match &bg.round {
                Round::Operating(s) => s.revenue,
                _ => 0,
            };
            if kind == "payout" {
                let per = rev / 10;
                // Ruby Dividend#payout_shares: the payees in (@game.players + @game.corporations) order, the market's
                // shares paying the company (1830), then `sort_by { -cash }` -- stable in Opal, so equal amounts keep
                // the payees' order
                let mut pays: Vec<(i32, usize, String)> = Vec::new();
                for (k, pid2) in super::view::player_order(g).into_iter().enumerate() {
                    let n: i32 = bg.corp_idx.get(ent.as_str()).map_or(0, |&ci| bg.corporations[ci].shares.iter().filter(|s| s.owner == EntityId::player(pid2)).map(|s| s.percent as i32 / 10).sum());
                    if n * per > 0 {
                        pays.push((-(n * per), k, pname(a, pid2)));
                    }
                }
                let pool: i32 = bg.corp_idx.get(ent.as_str()).map_or(0, |&ci| bg.corporations[ci].shares.iter().filter(|s| s.owner.is_market()).map(|s| s.percent as i32 / 10).sum());
                if pool * per > 0 {
                    pays.push((-(pool * per), usize::MAX, ent.clone()));
                }
                pays.sort();
                let list: Vec<String> = pays.iter().map(|(amt, _, nm)| format!("${} to {nm}", -amt)).collect();
                // (Dividend#payout_shares pays, then writes the line)
                bank_at = Some(out.len());
                out.push(format!("{ent} pays out ${rev} = ${per} per share ({})", list.join(", ")));
            } else {
                // (log_run_payout writes the line, then payout_corporation pays)
                out.push(format!("{ent} withholds ${rev}"));
                bank_at = Some(out.len());
            }
            price_moves(bg, a, Some(&ent), &mut out);
        }
        ("buy_train", _) => {
            let train = raw.get("train").and_then(Value::as_str).unwrap_or("");
            let tname = train.rsplit_once('-').map_or(train, |x| x.0);
            let seller = bg.corporations.iter().find(|c| c.trains.iter().any(|x| x.id == train)).map(|c| c.sym.clone());
            // Ruby Step::Train: a discarded train comes "from The Discard"
            let from = seller.unwrap_or_else(|| {
                if bg.depot.discarded.iter().any(|t| t.id == train) { "The Discard".into() } else { "The Depot".into() }
            });
            if raw.get("exchange").is_some() {
                let old = raw.get("exchange").and_then(Value::as_str).unwrap_or("");
                let oname = old.rsplit_once('-').map_or(old, |x| x.0);
                out.push(format!("{ent} exchanges a {oname} for a {tname} train for ${price} from {from}"));
            } else {
                out.push(format!("{ent} buys a {tname} train for ${price} from {from}"));
            }
            if let Some(p) = bg.corp_idx.get(ent.as_str()).and_then(|&i| bg.corporations[i].president_id()) {
                let c0 = bg.players.iter().find(|x| x.id == p).map_or(0, |x| x.cash);
                let c1 = a.players.iter().find(|x| x.id == p).map_or(0, |x| x.cash);
                if c1 < c0 {
                    let pos = out.len() - 1;
                    out.insert(pos, format!("{} contributes ${}", pname(a, p), c0 - c1));
                }
            }
            bought_at = Some(out.len());
        }
        ("discard_train", _) => {
            let train = raw.get("train").and_then(Value::as_str).unwrap_or("");
            out.push(format!("{ent} discards {}", train.rsplit_once('-').map_or(train, |x| x.0)));
        }
        ("buy_company", Round::Operating(_)) => {
            let comp = raw.get("company").and_then(Value::as_str).unwrap_or("");
            let owner = bg.company_idx.get(comp).map(|&i| holder_name(bg, &bg.companies[i].owner)).unwrap_or_default();
            out.push(format!("{ent} buys {} from {owner} for ${price}", cname(a, comp)));
        }
        ("buy_company", _) => {
            // Ruby BuySellParShares#process_buy_company: from another player, "-- X buys ... from Y for $n"
            let comp = raw.get("company").and_then(Value::as_str).unwrap_or("");
            let from_player = bg.company_idx.get(comp).is_some_and(|&i| bg.companies[i].owner.is_player());
            let owner = bg.company_idx.get(comp).map(|&i| holder_name(bg, &bg.companies[i].owner)).unwrap_or_default();
            let dash = if from_player { "-- " } else { "" };
            out.push(format!("{dash}{who} buys {} from {owner} for ${price}", cname(a, comp)));
            // Game::Base#after_buy_company (site_rules S6): the private's share (C&A's PRR_1) out of the market,
            // "<player> receives a 10% share of PRR"
            if let (Some(p), Some(share)) = (pid, crate::abilities::share_grant(&a.title, comp)) {
                let me = EntityId::player(p);
                let held = |game: &BaseGame| {
                    game.corporations.iter().find(|c| c.sym == share).map_or(0, |c| c.percent_owned_by(&me))
                };
                let got = held(a) as i32 - held(bg) as i32;
                if got > 0 {
                    out.push(format!("{who} receives a {got}% share of {share}"));
                }
            }
        }
        ("pass", Round::Operating(_)) | ("pass", Round::Merger(_)) | ("pass", Round::Stock(_))
            if a.title == "1867" && bg.trainless_major.first().is_some_and(|m| m == &ent) =>
        {
            // G1867::Step::MajorTrainless#process_pass
            out.push(format!("{ent} declines to nationalize"));
        }
        ("pass", Round::Operating(s)) if a.title == "1867" => {
            // Ruby log_pass: "X passes <the step's description>"
            let label: Option<String> = match s.step {
                S::RedeemShares => Some("redeem shares".into()),
                S::LayTile => Some(if s.num_laid_track == 1 && s.upgraded_this_turn { "lay track".into() } else if s.num_laid_track >= 2 { "lay track".into() } else { "lay/upgrade track".into() }),
                S::PlaceToken => Some("place a token".into()),
                S::BuyTrain => Some("buy trains".into()),
                S::BuyCompany | S::BuyCompanyPreloan => Some("buy companies".into()),
                _ => None,
            };
            if let Some(l) = label {
                out.push(format!("{ent} passes {l}"));
            }
            // G1867::Step::BuyTrain#pass!: a trainless major that buys none is nationalized
            let trainless = bg.corp_idx.get(ent.as_str()).is_some_and(|&i| bg.corporations[i].trains.is_empty() && bg.corporations[i].floated);
            let reset = a.corp_idx.get(ent.as_str()).is_some_and(|&i| !a.corporations[i].floated || a.corporations[i].closed);
            if s.step == S::BuyTrain && trainless && reset {
                nationalize_lines(g, bg, &ent, &mut out);
            }
        }
        ("choose", _) if a.title == "1867" && bg.trainless_major.first().is_some_and(|m| m == &ent) => {
            // G1867::Step::MajorTrainless#process_choose: nationalize!
            nationalize_lines(g, bg, &ent, &mut out);
        }
        ("merge", Round::Merger(s)) if a.title == "1867" => {
            let target = raw.get("corporation").and_then(Value::as_str).unwrap_or("").to_string();
            if s.converting {
                // Merge#finish_convert: the par, the free 20%, the minor's assets moved, its close
                let owner = bg.corp_idx.get(ent.as_str()).and_then(|&i| bg.corporations[i].owner_id.player_id());
                let par = a.corp_idx.get(target.as_str()).and_then(|&i| a.corporations[i].share_price.as_ref().map(|p| p.price)).unwrap_or(0);
                if let Some(p) = owner {
                    let nm = pname(a, p);
                    out.push(format!("{nm} pars {target} at ${par}"));
                    out.push(format!("{nm} receives a 20% share of {target}"));
                    out.push(format!("{nm} becomes the president of {target}"));
                }
                out.push(format!("{target} floats"));
                out.push(format!("{ent} closes"));
                out.push(format!("{ent} converts into {target} receiving {}", receiving_in(bg, &ent, Some(&g.company_seq))));
            } else if !s.merge_major {
                // Merge#process_merge: a minor added (and the merge finished when no other minor can join it)
                let list = if s.merging.is_empty() { vec![ent.clone()] } else { s.merging.clone() };
                out.push(format!("Adding {target} to merge of {}", list.join(",")));
                if let Round::Merger(s1) = &a.round {
                    if s1.merge_major {
                        out.push(format!("Finish merge of {}", s1.merging.join(",")));
                    }
                }
            } else {
                merge_to_major_lines(g, bg, a, &target, &s.merging, &mut out);
            }
        }
        ("remove_token", Round::Merger(_)) if a.title == "1867" => {
            // ReduceTokens#process_remove_token: "<company> removes token from <hex>"
            let city = raw.get("city").and_then(Value::as_str).unwrap_or("");
            let parts: Vec<&str> = city.split('-').collect();
            let inst = if parts.len() >= 3 { format!("{}-{}", parts[0], parts[1]) } else { parts.first().unwrap_or(&"").to_string() };
            // (the position before is kept without its map: the hexes after, whose tiles the removal doesn't change)
            let hex = a.hexes.iter().find(|h| h.tile.id == inst).or_else(|| a.hexes.iter().find(|h| Some(&h.id.as_str()) == parts.first())).map(|h| h.id.clone());
            if let Some(hex) = hex {
                out.push(format!("{ent} removes token from {hex}"));
            }
        }
        ("pass", Round::Merger(s)) if a.title == "1867" && !s.merging.is_empty() && s.converted.is_none() => {
            // Merge#process_pass while merging: finish_merge
            out.push(format!("Finish merge of {}", s.merging.join(",")));
        }
        ("pass", Round::Merger(s)) if a.title == "1867" => {
            if s.converted.is_some() && pid.is_some() {
                // PostMergerShares#process_pass (log_pass)
                out.push(format!("{who} passes buy shares post merge"));
            } else if s.converted.is_none() && s.merging.is_empty() && !s.converting && !s.merge_major {
                out.push(format!("{ent} passes convert or merge minor corporation"));
            }
        }
        ("pass", Round::Operating(s)) => {
            let label = if s.teleport_pending {
                None
            } else {
                match s.step {
                    S::LayTile => Some("lay/upgrade track"),
                    S::PlaceToken => Some("place a token"),
                    S::BuyTrain => Some("buy trains"),
                    S::BuyCompany => Some("buy companies"),
                    _ => None,
                }
            };
            if let Some(l) = label {
                out.push(format!("{ent} passes {l}"));
            }
        }
        ("bankrupt", _) => {
            if let Some(p) = acting {
                out.push(format!("-- {} goes bankrupt and sells remaining shares --", pname(a, p)));
                // Step::Bankrupt#sell_bankrupt_shares: the player's corporations sorted (Corporation#<=>, on the
                // prices before), each one's largest sellable bundle sold (sell_shares_and_change_price): the sale,
                // the presidency it moves, the price
                let me = EntityId::player(p);
                let held: Vec<String> = bg.corporations.iter().filter(|c| c.share_price.is_some()).map(|c| c.sym.clone()).collect();
                for sym in by_sort_order(bg, held) {
                    let (Some(&i0), Some(&i1)) = (bg.corp_idx.get(sym.as_str()), a.corp_idx.get(sym.as_str())) else { continue };
                    let (c0, c1) = (&bg.corporations[i0], &a.corporations[i1]);
                    let sold = c0.percent_owned_by(&me) as i32 - c1.percent_owned_by(&me) as i32;
                    if sold <= 0 {
                        continue;
                    }
                    let unit = c0.share_price.as_ref().map_or(0, |s| s.price);
                    let what = if sold == 10 { "a 10% share".to_string() } else { format!("{} shares", sold / 10) };
                    out.push(format!("{} sells {what} of {sym} and receives ${}", pname(a, p), unit * sold / 10));
                    if let Some(np) = c1.president_id() {
                        if c0.president_id() != Some(np) {
                            out.push(format!("{} becomes the president of {sym}", pname(a, np)));
                        }
                    }
                    price_moves(bg, a, Some(&sym), &mut out);
                }
            }
        }
        _ => {}
    }
    let _ = n;
    // consequences: presidencies, floats, price moves of a sale, privates granted or closed, phase, rusting
    if t == "sell_shares" || t == "buy_shares" || t == "par" || t == "buy_company" {
        presidents(bg, a, &mut out);
    }
    let floats_done = auction_1867 || (a.title == "1867" && t == "merge" && matches!(&bg.round, Round::Merger(s) if s.converting || s.merge_major));
    // 1867: the loans the company took for the move (AutomaticLoan#try_take_loan, before it pays: take_loan's line)
    // (a move before the company's LoanOperations -- a pass, a run, a dividend -- takes none itself: the loans then are
    // pay_interest!'s, written with the interest in the skipped steps)
    let before_loan_ops = matches!(t, "pass" | "run_routes" | "dividend") && matches!(&bg.round, Round::Operating(s0) if !matches!(s0.step, S::LoanOperations | S::DiscardTrain | S::BuyTrain | S::BuyCompany | S::Done));
    MOVE_LOANS.with(|c| *c.borrow_mut() = (String::new(), 0));
    if a.title == "1867" && !before_loan_ops {
        let corp = g.operator_of(&ent);
        let (l0, l1) = (
            bg.corp_idx.get(corp.as_str()).map_or(0, |&i| bg.corporations[i].loans),
            a.corp_idx.get(corp.as_str()).map_or(0, |&i| a.corporations[i].loans),
        );
        for _ in l0..l1 {
            out.insert(n, format!("{corp} takes a loan and receives $45"));
        }
        MOVE_LOANS.with(|c| *c.borrow_mut() = (corp.clone(), l1.saturating_sub(l0)));
    }
    if t == "sell_shares" {
        let shares = raw.get("shares").and_then(Value::as_array).cloned().unwrap_or_default();
        let sid = shares.first().and_then(Value::as_str).unwrap_or("");
        let corp = sid.rfind('_').map_or("", |p| &sid[..p]).to_string();
        // Ruby log_share_price: nothing when the price stays in its cell (`return unless from != to`)
        price_moves(bg, a, Some(&corp), &mut out);
    }
    if !floats_done {
        floats(bg, a, &mut out);
    }
    if let Some(c) = pending_home {
        out.push(format!("{c} must choose city for home token"));
    }
    if a.bank_broken && !bg.bank_broken {
        let line = "-- The bank has broken --".to_string();
        match bank_at {
            Some(i) if i <= out.len() => out.insert(i, line),
            _ => out.push(line),
        }
    }
    // a private that closes as its corporation buys a train (B&O's): Ruby closes it before the phase changes
    let mut closed_early: Option<String> = None;
    if let (Some(i), Some(c)) = (bought_at, crate::abilities::close_on_bought_train(&a.title, &ent)) {
        let now = a.company_idx.get(c).is_some_and(|&k| a.companies[k].closed && !bg.companies[k].closed);
        if now && i <= out.len() {
            out.insert(i, format!("{} closes", cname(a, c)));
            closed_early = Some(c.to_string());
        }
    }
    // (a phase an export at the operating round's end sets off comes after the export's line: `transitions`)
    let exported_now = export_phase(b, a, raw);
    if exported_now.is_none() {
        phase_block(g, b, raw, None, &mut out);
    }
    // (a private exchanged for a share closes without a line of its own: Step::Exchange's company.close!)
    let exchanged = t == "buy_shares" && raw.get("entity_type").and_then(Value::as_str) == Some("company");
    let closing: Vec<String> = a
        .companies
        .iter()
        .enumerate()
        .filter(|(i, c)| c.closed && !bg.companies[*i].closed && !(exchanged && c.sym == ent) && closed_early.as_deref() != Some(c.sym.as_str()))
        // (1867's hidden company "3" closes without a word at phase 3: G1867::Game#event_green_minors!, `close!`; a
        // nationalized minor's companies close with its own line: nationalize_lines)
        .filter(|(_, c)| !(a.title == "1867" && c.sym == "3"))
        .filter(|(i, _)| !(a.title == "1867" && bg.companies[*i].owner.corp_sym().is_some_and(|cs| a.corp_idx.get(cs).is_some_and(|&ci| a.corporations[ci].closed))))
        .map(|(_, c)| c.sym.clone())
        .collect();
    if closing.len() > 1 && a.phase.name != bg.phase.name {
        out.push("-- Event: Private companies close --".into());
    } else {
        for sym in closing {
            out.push(format!("{} closes", cname(a, &sym)));
        }
    }
    // the auction's resolutions
    if let (Round::Auction(_), false) = (&bg.round, auction_1867) {
        for (i, c) in a.companies.iter().enumerate() {
            let was = bg.companies[i].owner.player_id();
            let now = c.owner.player_id();
            if let (None, Some(p)) = (was, now) {
                let bought_now = t == "bid" && raw.get("company").and_then(Value::as_str) == Some(c.sym.as_str()) && b.bids.get(&c.sym).map_or(true, |v| v.is_empty());
                if bought_now {
                    continue;
                }
                let bids = b.bids.get(&c.sym).cloned().unwrap_or_default();
                let price = bids.iter().filter(|x| x.0 == p).map(|x| x.1).max().unwrap_or(0);
                if bids.len() <= 1 {
                    out.push(format!("{} wins the auction for {} with the only bid of ${price}", pname(a, p), cname(a, &c.sym)));
                } else {
                    out.push(format!("{} wins the auction for {} with a bid of ${price}", pname(a, p), cname(a, &c.sym)));
                }
                if let Some(share) = crate::abilities::share_grant(&a.title, &c.sym) {
                    let (corp, idx) = share.rsplit_once('_').unwrap_or((share, "1"));
                    if idx != "0" {
                        out.push(format!("{} receives a 10% share of {corp}", pname(a, p)));
                    }
                }
            }
        }
        if let Round::Auction(_) = &a.round {
            if let Some(c) = a.auctioning_company() {
                if b.auctioning.as_deref() != Some(c.as_str()) {
                    out.push(format!("{} goes up for auction", cname(a, &c)));
                }
            }
        }
    }
    // the turn and round changes
    transitions(g, b, raw, &mut out);
    out
}

/// G1867::Game#nationalize!'s lines: "X is nationalized", the loans it pays off from its cash, its price one step left
/// and two more per loan still held, the settlement with its players at the new price, and a minor's close (with its
/// companies: close_corporation).
fn nationalize_lines(g: &SiteGame, bg: &BaseGame, sym: &str, out: &mut Vec<String>) {
    let a = &g.game;
    let Some(&i0) = bg.corp_idx.get(sym) else { return };
    let c0 = &bg.corporations[i0];
    out.push(format!("{sym} is nationalized"));
    // repay_loan while the cash covers a loan's $50
    let repaid = if c0.cash > 0 { (c0.cash / 50).min(c0.loans as i32).max(0) as u32 } else { 0 };
    for _ in 0..repaid {
        out.push(format!("{sym} pays off a loan for $50"));
    }
    // the price it settles at: one step left, two more per loan left (nationalization_loan_movement)
    let from = c0.share_price.as_ref().map_or(0, |s| s.price);
    let col = c0.share_price.as_ref().map_or(0, |s| s.column) as i32;
    let row = c0.share_price.as_ref().map_or(0, |s| s.row);
    let steps = 1 + 2 * (c0.loans - repaid) as i32;
    let to = a.stock_market.share_price_at(row, (col - steps).max(0) as u8).or_else(|| a.stock_market.share_price_at(0, (col - steps).max(0) as u8)).map_or(from, |s| s.price);
    if to != from {
        out.push(format!("{sym}'s share price moves left from ${from} to ${to}"));
    }
    let mut pays: Vec<(i32, usize, String)> = Vec::new();
    for (k, pid) in super::view::player_order(g).into_iter().enumerate() {
        let pct: i32 = c0.shares.iter().filter(|s| s.owner == EntityId::player(pid)).map(|s| s.percent as i32).sum();
        let n = pct / c0.share_unit_percent.max(1) as i32;
        if n > 0 {
            pays.push((-(n * to), k, pname(a, pid)));
        }
    }
    pays.sort();
    if !pays.is_empty() {
        let total = (100 / c0.share_unit_percent.max(1) as i32) * to;
        let list: Vec<String> = pays.iter().map(|(amt, _, nm)| format!("${} to {nm}", -amt)).collect();
        out.push(format!("{sym} settles with shareholders ${total} = ${to} ({})", list.join(", ")));
    }
    if c0.corp_type == crate::title::CorpType::Minor {
        out.push(format!("{sym} closes"));
        let me = EntityId::corporation(sym);
        let mut owned: Vec<&crate::entities::Company> = bg
            .companies
            .iter()
            .enumerate()
            .filter(|(i, m)| m.owner == me && !m.closed && a.companies.get(*i).is_some_and(|x| x.closed))
            .map(|(_, m)| m)
            .collect();
        // (Ruby's `corporation.companies`: in the order it got them)
        owned.sort_by_key(|m| g.company_seq.get(&m.sym).copied().unwrap_or(0));
        let comps: Vec<String> = owned.iter().map(|m| m.sym.clone()).collect();
        if !comps.is_empty() {
            out.push(format!("{sym}'s companies close: {}", comps.join(", ")));
        }
    }
}

/// What the train that exports at an operating round's end sets off (1867, Depot#export!: its line, then
/// `phase.buying_train!`): the exported train's id when the new phase is the one it starts (its `on`).
fn export_phase(b: &Before, a: &BaseGame, raw: &Value) -> Option<String> {
    let bg = &b.game;
    if a.title != "1867" || !matches!(bg.round, Round::Operating(_)) || !bg.phase_has_status("export_train") || a.phase.name == bg.phase.name {
        return None;
    }
    let t = exported_train(bg, a, raw)?;
    let on = a.title_def().phases().into_iter().find(|p| p.name == a.phase.name).and_then(|p| p.on)?;
    // (a train the move bought that starts the phase itself: the phase is the move's)
    let bought = raw.get("variant").and_then(Value::as_str).or_else(|| {
        let id = raw.get("train").and_then(Value::as_str)?;
        bg.depot.trains.iter().chain(bg.depot.discarded.iter()).chain(bg.corporations.iter().flat_map(|c| c.trains.iter())).find(|x| x.id == id).map(|x| x.name.as_str())
    });
    if bought == Some(on) {
        return None;
    }
    (on == t.name).then(|| t.id.clone())
}

/// The train Depot#export! took at an operating round's end: the depot's next train once the move's own purchase is
/// made, gone from the depot without a buyer.
fn exported_train(bg: &BaseGame, a: &BaseGame, raw: &Value) -> Option<crate::entities::Train> {
    let bought = raw.get("train").and_then(Value::as_str).unwrap_or("");
    let head = bg.depot.trains.iter().find(|t| t.id != bought)?;
    let gone = !a.depot.trains.iter().any(|x| x.id == head.id) && !a.corporations.iter().any(|c| c.trains.iter().any(|x| x.id == head.id));
    gone.then(|| head.clone())
}

/// The lines of a new phase (the phase, the trains that rust, 1867's events), the exported train left out of the
/// rusting ones.
fn phase_block(g: &SiteGame, b: &Before, raw: &Value, exported: Option<&str>, out: &mut Vec<String>) {
    let a = &g.game;
    let bg = &b.game;
    if a.phase.name != bg.phase.name {
        out.push(phase_line(a));
        // Ruby rust_trains!: the rusting trains in @depot.trains order (each type's trains by number), each owner
        // counted in the order its first rusting train comes, an unsold or discarded train's owner The Depot (a
        // train traded in for a D was reclaimed by the depot first)
        let bought = raw.get("train").and_then(Value::as_str).unwrap_or("");
        let traded = raw.get("exchange").and_then(Value::as_str).unwrap_or("");
        let mut held: Vec<(String, String, String)> = Vec::new(); // (id, name, owner)
        let buyer = raw.get("entity").and_then(Value::as_str).unwrap_or("");
        for c in &bg.corporations {
            for t in &c.trains {
                // (at an export's phase the move's purchase was made first: a train bought from another company is
                // its buyer's when it rusts)
                let owner = if t.id == traded {
                    "The Depot".to_string()
                } else if exported.is_some() && t.id == bought {
                    buyer.to_string()
                } else {
                    c.sym.clone()
                };
                held.push((t.id.clone(), t.name.clone(), owner));
            }
        }
        // (the train an export at the round's end took, whichever phase it started)
        let exported_now: Option<String> = if a.title == "1867" && matches!(bg.round, Round::Operating(_)) && bg.phase_has_status("export_train") {
            exported_train(bg, a, raw).map(|t| t.id)
        } else {
            None
        };
        for t in bg.depot.trains.iter().chain(bg.depot.discarded.iter()) {
            if Some(t.id.as_str()) != exported && Some(&t.id) != exported_now.as_ref() {
                held.push((t.id.clone(), t.name.clone(), "The Depot".to_string()));
            }
        }
        // the trains exported earlier (1867): Ruby's depot keeps them, its own, and they rust with their type
        // (engine-rs drops them; the site layer remembers them)
        if a.title == "1867" {
            let on = a.title_def().phases().into_iter().find(|p| p.name == a.phase.name).and_then(|p| p.on);
            let rusting: Vec<&str> = a.title_def().trains().iter().filter(|t| on.is_some() && t.rusts_on == on).map(|t| t.name).collect();
            for (id, name) in &g.exported {
                if rusting.contains(&name.as_str()) {
                    held.push((id.clone(), name.clone(), "The Depot".to_string()));
                }
            }
        }
        let exists = |id: &str| {
            a.corporations.iter().any(|c| c.trains.iter().any(|t| t.id == id))
                || a.depot.trains.iter().chain(a.depot.discarded.iter()).any(|t| t.id == id)
        };
        let order: Vec<&str> = a.title_def().trains().iter().map(|t| t.name).collect();
        let rank = |id: &str, name: &str| {
            let n: u32 = id.rsplit_once('-').and_then(|x| x.1.parse().ok()).unwrap_or(0);
            (order.iter().position(|x| *x == name).unwrap_or(usize::MAX), n)
        };
        let mut rusted: Vec<(String, String, String)> =
            held.into_iter().filter(|(id, _, _)| (exported.is_some() || id != bought) && !exists(id)).collect();
        rusted.sort_by_key(|(id, name, _)| rank(id, name));
        rusted.dedup_by(|x, y| x.0 == y.0);
        if !rusted.is_empty() {
            let mut names: Vec<String> = Vec::new();
            let mut owners: Vec<(String, usize)> = Vec::new();
            for (_, name, owner) in &rusted {
                if !names.contains(name) {
                    names.push(name.clone());
                }
                match owners.iter_mut().find(|o| &o.0 == owner) {
                    Some(o) => o.1 += 1,
                    None => owners.push((owner.clone(), 1)),
                }
            }
            let list: Vec<String> = owners.iter().map(|(s, k)| format!("{s} x{k}")).collect();
            out.push(format!("-- Event: {} trains rust ({}) --", names.join(", "), list.join(", ")));
        }
    }
    if a.title == "1867" && a.phase.name != bg.phase.name {
        phase_events_1867(g, b, out);
    }
}

/// What happens between the action and the next decision: skipped steps, other companies' turns, skipped players,
/// round ends and starts.
fn transitions(g: &SiteGame, b: &Before, _raw: &Value, out: &mut Vec<String>) {
    let a = &g.game;
    let bg = &b.game;
    let same_round = bg.round.round_type_str() == a.round.round_type_str() && bg.round.round_num() == a.round.round_num() && bg.turn == a.turn;
    if let (Round::Operating(s0), Round::Operating(s1)) = (&bg.round, &a.round) {
        if s0.pending_tokens.is_empty() {
            if let Some((sym, 0, _)) = s1.pending_tokens.first() {
                // a company's home token at its turn's start is turn_start's ("must choose city for home token");
                // a token a tile lay displaced is Tracker's
                if !home_choice_now(bg, a) {
                    out.push(format!("{sym} must choose city for token"));
                }
            }
        }
    }
    // the operating company's remaining steps
    if let Some((corp, step, idx, _)) = op_corp(bg) {
        let home_choice = matches!(&bg.round, Round::Operating(s0) if !s0.pending_tokens.is_empty() && step == S::PlaceToken);
        // Ruby's step acted only if it processed the action itself (Round::Base#process_action sets `acted` on the
        // step that processes it): a private bought meanwhile is the first, non-blocking BuyCompany step's, a
        // private's lay or token its own step's, and 1830's blocking BuyCompany step at the end never acts (the
        // first one processes every buy_company); a step that didn't act logs its skip
        let t = raw_type(_raw);
        let by_company = _raw.get("entity_type").and_then(Value::as_str) == Some("company");
        // (1867's MajorTrainless asks another major before the operator's turn has begun: its pass isn't the
        // operator's first step's)
        let other_major = a.title == "1867"
            && step == S::RedeemShares
            && bg.trainless_major.first().is_some_and(|m| Some(m.as_str()) == _raw.get("entity").and_then(Value::as_str));
        let own = !by_company
            && !other_major
            && match step {
                S::LayTile => matches!(t, "lay_tile" | "pass"),
                S::PlaceToken => matches!(t, "place_token" | "pass"),
                S::RunRoutes => t == "run_routes",
                S::Dividend => t == "dividend",
                S::BuyTrain => b.buy_train_acted || matches!(t, "buy_train" | "sell_shares" | "pass" | "bankrupt"),
                S::BuyCompany => t == "pass",
                S::RedeemShares => matches!(t, "buy_shares" | "pass"),
                S::BuyCompanyPreloan => matches!(t, "pass" | "buy_company"),
                _ => true,
            };
        // D&H's teleport: while its token is pending (SpecialToken blocks) Ruby's skip_steps stops there, so the
        // Track step the teleport's lay used up is skipped (and logged) only once D&H's token is placed or declined
        let teleport_done = by_company && matches!(&bg.round, Round::Operating(s0) if s0.teleport_pending);
        let next_from = if home_choice {
            step_pos(&S::PlaceToken)
        } else if teleport_done {
            step_pos(&S::LayTile)
        } else if t == "discard_train" {
            // a discard follows the operating company's train purchase (its BuyTrain step acted)
            step_pos(&S::BuyTrain) + 1
        } else if own {
            step_pos(&step) + 1
        } else {
            step_pos(&step)
        };
        match op_corp(a) {
            Some((c2, s2, i2, _)) if same_round && c2 == corp && i2 == idx => {
                // (a discard pending -- another company over the train limit -- blocks Ruby's skip_steps until it's
                // made: the skips come with the discard; so does D&H's teleport token while it is pending)
                let teleport_pending = matches!(&a.round, Round::Operating(s1) if s1.teleport_pending);
                if step_pos(&s2) > next_from && step_pos(&s2) < steps_now().len() && !teleport_pending {
                    skips_with(g, bg, &corp, next_from, step_pos(&s2), run_revenue(_raw, &corp), out);
                }
            }
            _ => {
                // (a minor nationalized in the move ends its turn there: nationalize! closes it, force_next_entity!;
                // a major started afresh goes on with its skips)
                let gone = a.title == "1867" && a.corp_idx.get(corp.as_str()).is_some_and(|&ci| a.corporations[ci].closed);
                // (closed by its LoanOperations -- interest it can't pay -- its skips go up to there)
                let to = if gone { step_pos(&S::LoanOperations) + 1 } else { steps_now().len() };
                if (step_pos(&step) < steps_now().len() || t == "discard_train") && next_from < to {
                    skips_with(g, bg, &corp, next_from, to, run_revenue(_raw, &corp), out);
                }
            }
        }
    }
    // stock round: players skipped because they can't act (Ruby Round::Stock#start_entity -> skip_steps ->
    // BuySellParShares#skip! -> log_skip), from the player after the one whose turn ended up to the next player who
    // can act -- all the way round when that is the same player again -- or, when the round ends, until every
    // player has passed (each skip passes the player; the player whose turn just ended passed only if it did
    // nothing else this turn)
    if let Round::Stock(s0) = &bg.round {
        let order = &s0.player_order;
        let n = order.len();
        let mut from = s0.current_player_index;
        let t = raw_type(_raw);
        let mut by_player = _raw.get("entity_type").and_then(Value::as_str) == Some("player")
            || (a.title == "1867" && t == "place_token");
        // 1867's minors' auctions (BuySellParSharesViaBid): nobody's turn ends while one runs or a home city is to be
        // chosen (the step's own active entities bid; the round's entity stays); the move that ends one hands the
        // turn on from its winner (G1867 BuySellParShares#win_bid: goto_entity!(winner), pass!)
        let mut won_by: Option<usize> = None;
        if a.title == "1867" {
            if matches!(&a.round, Round::Stock(s1) if s1.bid_auction.is_some() || !s1.pending_home_tokens.is_empty()) {
                by_player = false;
            } else if let Some(au) = &s0.bid_auction {
                let winner = a.corp_idx.get(au.corporation_sym.as_str()).and_then(|&ci| a.corporations[ci].president_id());
                if let Some(i) = winner.and_then(|w| order.iter().position(|p| *p == w)) {
                    from = i;
                    won_by = Some(i);
                }
            } else if let Some((_, chooser)) = s0.pending_home_tokens.first() {
                // the home city chosen once the auction was won (or a major parred): the turn goes on from the one who
                // chose it, the winner
                if let Some(i) = order.iter().position(|p| p == chooser) {
                    from = i;
                }
            }
        }
        match &a.round {
            Round::Stock(s1) if same_round && by_player => {
                let to = s1.current_player_index;
                let ended = to != from || !s1.acted_this_turn;
                if ended && n > 0 {
                    let mut k = (from + 1) % n;
                    while k != to {
                        out.push(format!("{} has no valid actions and passes", pname(a, order[k])));
                        k = (k + 1) % n;
                    }
                }
            }
            _ if !same_round && by_player && n > 0 => {
                let mut passed: Vec<bool> = (0..n).map(|i| s0.player_passed.get(i).copied().unwrap_or(false)).collect();
                passed[from % n] = won_by.is_none() && t == "pass" && !b.turn_acted;
                let bankrupt = |pid: u32| bg.players.iter().any(|p| p.id == pid && p.bankrupt);
                let mut k = (from + 1) % n;
                let mut guard = 0;
                while (0..n).any(|i| !passed[i] && !bankrupt(order[i])) && guard < 2 * n {
                    if !bankrupt(order[k]) {
                        out.push(format!("{} has no valid actions and passes", pname(a, order[k])));
                        passed[k] = true;
                    }
                    k = (k + 1) % n;
                    guard += 1;
                }
            }
            _ => {}
        }
    }
    if same_round {
        return;
    }
    // a round ended: the stock round's sold-out price rises, in Ruby's order (Round::Stock#finish_round sorts the
    // corporations, Corporation#<=>: operating order on the prices before)
    if let Round::Stock(_) = &bg.round {
        for sym in by_sort_order(bg, bg.corporations.iter().map(|c| c.sym.clone()).collect()) {
            price_moves(bg, a, Some(&sym), out);
        }
    }
    // 1867's export at an operating round's end (G1867::Game#or_round_finished, phases with export_train:
    // Depot#export! writes "-- Event: A 4 train exports --")
    if a.title == "1867" && matches!(bg.round, Round::Operating(_)) && bg.phase_has_status("export_train") && !same_round {
        if let Some(t) = exported_train(bg, a, _raw) {
            out.push(format!("-- Event: A {} train exports --", t.name));
            // the phase the export starts (Depot#export!: phase.buying_train!)
            if let Some(id) = export_phase(b, a, _raw) {
                phase_block(g, b, _raw, Some(&id), out);
            }
        }
    }
    // the rounds the move went through: every round that finished during it (the records' store_player_info
    // entries) but the first began and ended inside it -- an operating round in which no company operates (only the
    // privates pay), between a stock round and the next. Ruby's next_round! writes each one's lines.
    let mut done: Vec<super::records::PlayerInfo> =
        a.site.as_ref().map(|r| r.player_history.iter().skip(b.rounds_before).cloned().collect()).unwrap_or_default();
    if a.finished && !bg.finished {
        // end_game!'s entry is the round the game ended in: the round now
        done.pop();
    }
    let mut prev = super::records::round_short(&bg.round).to_string();
    for e in done.iter().skip(1) {
        new_round_lines(g, &prev, &e.round, e.turn, e.round_num, ors_of_phase(a), out);
        prev = e.round.clone();
    }
    // the new round
    let (num, total) = match &a.round {
        Round::Operating(s) => (s.round_num, s.total_ors),
        Round::Merger(s) => (s.round_num, s.total_ors),
        _ => (1, ors_of_phase(a)),
    };
    new_round_lines(g, &prev, super::records::round_short(&a.round), a.turn, num, total, out);
}

/// The number of operating rounds of the phase (Ruby's `@operating_rounds`, set as a set of them begins).
fn ors_of_phase(game: &BaseGame) -> u8 {
    let def = game.title_def().phases().into_iter().find(|p| p.name == game.phase.name);
    def.map_or(game.phase.operating_rounds, |d| d.operating_rounds)
}

/// Corporations in Ruby's `Corporation#<=>` order (`sort_order_key`: price high to low, rightmost column, top row,
/// first in its cell, name); those without a price after them in their order here.
fn by_sort_order(game: &BaseGame, syms: Vec<String>) -> Vec<String> {
    let mut keyed: Vec<(Option<(i32, i32, u8, usize, String)>, usize, String)> = syms
        .into_iter()
        .enumerate()
        .map(|(i, sym)| {
            let key = game.corp_idx.get(sym.as_str()).and_then(|&ci| game.corporations[ci].share_price.as_ref()).map(|sp| {
                let pos = game.market_cell_corps.get(&(sp.row, sp.column)).and_then(|l| l.iter().position(|s| s == &sym)).unwrap_or(0);
                (-sp.price, -(sp.column as i32), sp.row, pos, sym.clone())
            });
            (key, i, sym)
        })
        .collect();
    // Some before None (Corporation#<=>: a corporation without a price goes last)
    keyed.sort_by(|x, y| match (&x.0, &y.0) {
        (Some(a), Some(b)) => a.cmp(b),
        (Some(_), None) => std::cmp::Ordering::Less,
        (None, Some(_)) => std::cmp::Ordering::Greater,
        (None, None) => x.1.cmp(&y.1),
    });
    keyed.into_iter().map(|k| k.2).collect()
}

/// Ruby's lines as a round begins (`next_round!`): the priority deal after a stock round or the private auction
/// (`reorder_players`), the round's heading, and at an operating round's start the privates' revenue
/// (`payout_companies`: the players' in the players' order, then the corporations').
fn new_round_lines(g: &SiteGame, prev: &str, kind: &str, turn: u32, num: u8, total: u8, out: &mut Vec<String>) {
    let a = &g.game;
    match kind {
        "SR" => {
            if prev == "PR" {
                out.push(format!("{} has priority deal", pname(a, a.priority_deal_player)));
            }
            out.push(format!("-- Stock Round {turn} --"));
            // 1867: a stock round that began and ended inside the move: nobody could act (each one's skip, from the
            // priority deal)
            let current = turn == a.turn && matches!(a.round, Round::Stock(_));
            if a.title == "1867" && !current {
                let order = super::view::player_order(g);
                let start = order.iter().position(|p| *p == a.priority_deal_player).unwrap_or(0);
                for k in 0..order.len() {
                    out.push(format!("{} has no valid actions and passes", pname(a, order[(start + k) % order.len()])));
                }
            }
            // 1867: the players before the first who can act, from the priority deal (Round::Stock#start_entity ->
            // skip_steps -> log_skip)
            if let Round::Stock(s1) = &a.round {
                let n = s1.player_order.len();
                if a.title == "1867" && turn == a.turn && n > 0 {
                    if let Some(mut k) = s1.player_order.iter().position(|p| *p == a.priority_deal_player) {
                        let mut guard = 0;
                        while k != s1.current_player_index % n && guard < n {
                            out.push(format!("{} has no valid actions and passes", pname(a, s1.player_order[k])));
                            k = (k + 1) % n;
                            guard += 1;
                        }
                    }
                }
            }
        }
        "MR" => {
            // G1867::Game#new_merger_round: round_description('Merger', round_num)
            out.push(format!("-- Merger Round {turn}.{num} (of {total}) --"));
        }
        "OR" => {
            if prev == "SR" {
                out.push(format!("{} has priority deal", pname(a, a.priority_deal_player)));
            }
            out.push(format!("-- Operating Round {turn}.{num} (of {total}) --"));
            // Game::Base#payout_companies: sorted by owner (the players' in their order, then the corporations' by
            // Corporation#<=>), then revenue, then name
            let paying = |owner: &EntityId| {
                let mut list: Vec<(i32, String, String)> = a
                    .companies
                    .iter()
                    .filter(|c| &c.owner == owner && !c.closed && c.revenue > 0)
                    .map(|c| (c.revenue, cname(a, &c.sym), c.sym.clone()))
                    .collect();
                list.sort();
                list
            };
            for pid in super::view::player_order(g) {
                for (rev, name, _) in paying(&EntityId::player(pid)) {
                    out.push(format!("{} collects ${rev} from {name}", pname(a, pid)));
                }
            }
            for sym in by_sort_order(a, a.corporations.iter().map(|c| c.sym.clone()).collect()) {
                let mut list = paying(&EntityId::corporation(&sym));
                // (a corporation without a share price -- 1867's national -- compares above itself in Corporation#<=>,
                // so the sort keys never tie on its owner: Opal pays its companies highest revenue first)
                if a.corp_idx.get(sym.as_str()).is_some_and(|&ci| a.corporations[ci].share_price.is_none()) {
                    list.reverse();
                }
                for (rev, name, _) in list {
                    out.push(format!("{sym} collects ${rev} from {name}"));
                }
            }
        }
        _ => {}
    }
}

fn raw_type(raw: &Value) -> &str {
    raw.get("type").and_then(Value::as_str).unwrap_or("")
}

/// A company whose turn the move started must choose the city of its home token (Game::Base#place_home_token, as
/// its turn starts: ERIE on an upgraded E11).
fn home_choice_now(bg: &BaseGame, a: &BaseGame) -> bool {
    let Round::Operating(s1) = &a.round else { return false };
    let Some((sym, 0, _)) = s1.pending_tokens.first() else { return false };
    let was_pending = matches!(&bg.round, Round::Operating(s0) if !s0.pending_tokens.is_empty());
    let started = match op_corp(bg) {
        Some((c0, _, i0, r0)) => {
            !(bg.round.round_type_str() == a.round.round_type_str() && r0 == s1.round_num && bg.turn == a.turn && i0 == s1.entity_index && &c0 == sym)
        }
        None => true,
    };
    !was_pending && started && s1.current_corp_sym() == Some(sym.as_str())
}

/// G1867::Game#end_game!: the loans "paid off" by moving each company's price one step left a loan, in the game's
/// corporation order, before the game-over line.
pub fn game_end_1867(g: &SiteGame, b: &Before) -> Vec<String> {
    let mut out = Vec::new();
    let a = &g.game;
    if a.title != "1867" {
        return out;
    }
    for c0 in b.game.corporations.iter().filter(|c| c.loans > 0 && !c.closed) {
        if out.is_empty() {
            out.push("-- Loans are \"paid off\" by moving share price left one step per loan --".to_string());
        }
        let from = c0.share_price.as_ref().map(|s| s.price);
        let to = a.corp_idx.get(c0.sym.as_str()).and_then(|&i| a.corporations[i].share_price.as_ref()).map(|s| s.price);
        if let (Some(from), Some(to)) = (from, to) {
            if from != to {
                let n = c0.loans;
                out.push(format!("{}'s share price moves left from ${from} to ${to} ({n} step{})", c0.sym, if n == 1 { "" } else { "s" }));
            }
        }
    }
    out
}

/// The lines of a company's turn starting (after `explain`): "X operates Y", its home station.
pub fn turn_start(g: &SiteGame, b: &Before) -> Vec<String> {
    IS_1867.with(|c| c.set(g.game.title == "1867"));
    let a = &g.game;
    let mut out = Vec::new();
    let before = op_corp(&b.game);
    let Round::Operating(s) = &a.round else { return out };
    let Some(now) = s.current_corp_sym() else { return out };
    let same_round = b.game.round.round_type_str() == "Operating" && b.game.round.round_num() == s.round_num && b.game.turn == a.turn;
    let start_idx = match &before {
        Some((_, _, i, _)) if same_round => i + 1,
        _ => 0,
    };
    for i in start_idx..=s.entity_index.min(s.operating_order.len().saturating_sub(1)) {
        let corp = &s.operating_order[i];
        // (a company closed since the round began -- 1867's minors nationalized at phase 4 -- is passed over without a
        // word: Round::Operating#next_entity! skips a closed entity)
        if corp != now && a.corp_idx.get(corp.as_str()).is_some_and(|&ci| a.corporations[ci].closed || (a.title == "1867" && !a.corporations[ci].floated)) {
            continue;
        }
        let pres = a.corp_idx.get(corp.as_str()).and_then(|&ci| a.corporations[ci].president_id());
        if let Some(p) = pres {
            out.push(format!("{} operates {corp}", pname(a, p)));
        }
        let had = b.game.corp_idx.get(corp.as_str()).is_some_and(|&ci| b.game.corporations[ci].tokens.first().is_some_and(|t| t.used));
        let has = a.corp_idx.get(corp.as_str()).is_some_and(|&ci| a.corporations[ci].tokens.first().is_some_and(|t| t.used));
        if has && !had {
            let hex = a.corp_idx.get(corp.as_str()).map(|&ci| a.corporations[ci].tokens[0].city_hex_id.clone()).unwrap_or_default();
            out.push(format!("{corp} places a token on {hex}"));
        }
        if corp == now && home_choice_now(&b.game, a) {
            out.push(format!("{corp} must choose city for home token"));
        }
        if corp != now {
            skips(g, &b.game, corp, 0, steps_now().len(), &mut out);
        } else if step_pos(&s.step) > 0 {
            // (1867's turn starts at RedeemShares, its first step: skipped there too)
            let from = if a.title == "1867" { 0 } else { 1 };
            skips(g, &b.game, corp, from, step_pos(&s.step), &mut out);
        }
    }
    out
}
