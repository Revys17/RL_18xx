//! 1867's move menu as facts (RawMenu, docs/SITE_ENGINE_RS_1867.md §3.5): what the player to act may do now, as
//! complete site actions with the numbers the hand needs, read off the legal-move generator's moves (Ruby's whole set,
//! `CrossPlayerPolicy::RULES`) with the site's own-only rule applied (another player's train, private or minor is
//! never offered; the minors left out of a merger are named). The shared parts keep step 1's shapes (living-board's
//! `rawmenu.ts`, as Opal's `moves.ts rawMenu` makes them); 1867's own parts are facts the client words
//! (`words.ts`): `merger`, `reduceTokens`, `redeem`, `nationalize`, `passKind`, `passNationalizes`, `autoLoans`,
//! `tilePower`, `trains[].loans`, `auctionStart` and the minors' lots' `par`.

use super::super::game::SiteGame;
use super::super::menu as m1;
use crate::actions::Action;
use crate::entities::EntityId;
use crate::g1867::legal::Move;
use crate::rounds::Round;
use crate::steps::{StepEntity, StepKind};
use crate::title::{CorpType, CrossPlayerPolicy};
use serde_json::{json, Map, Value};

/// 1867's hexes in Ruby's order (`game.hexes`: g_1867/map.rb's HEXES as the site lists them; its menus follow it).
pub const RUBY_HEX_ORDER: &[&str] = &[
    "B6", "B8", "C5", "C7", "C19", "D4", "D6", "D14", "E3", "E5", "E7", "E9", "F2", "F4", "F6", "F10", "F12", "F14", "G3",
    "G5", "G7", "G9", "G11", "G13", "H4", "H6", "H8", "H12", "I5", "I7", "I9", "I11", "I13", "J6", "J8", "J10", "J14",
    "K5", "K7", "K9", "L6", "L8", "M5", "M7", "N6", "O11", "D18", "C9", "D10", "E11", "C11", "D12", "C13", "K11", "N8",
    "N10", "M11", "O9", "M9", "D8", "F8", "E13", "E15", "C17", "I15", "N12", "G15", "E17", "D16", "O7", "J12", "L10",
    "H14", "C15", "B18", "H10", "M13", "K13", "D2", "C3", "E1", "B16", "L14", "L12", "F16", "A7", "F18", "M15", "O13",
    "P8", "A17", "A19",
];

pub fn hex_pos(hex: &str) -> usize {
    RUBY_HEX_ORDER.iter().position(|h| *h == hex).unwrap_or(usize::MAX)
}

/// Ruby's MAX_MINOR_PAR (G1867::Step::BuySellParShares): a minor won at auction pars at most at $135.
const MAX_MINOR_PAR: i32 = 135;
/// A loan brings its value less $5 (Ruby `take_loan`), and costs $5 interest an operating round.
const LOAN_VALUE: i32 = 50;
const LOAN_NET: i32 = 45;
const INTEREST: i32 = 5;

fn ent_json(e: &StepEntity) -> Value {
    match e {
        StepEntity::Player(p) => json!({"entity": p, "entity_type": "player"}),
        StepEntity::Corp(s) => json!({"entity": s, "entity_type": "corporation"}),
        StepEntity::Company(s) => json!({"entity": s, "entity_type": "company"}),
    }
}

fn corp_full_name(g: &SiteGame, sym: &str) -> String {
    g.game.corp_idx.get(sym).map(|&ci| g.game.corporations[ci].name.clone()).unwrap_or_else(|| sym.to_string())
}

fn max_loans(g: &SiteGame, ci: usize) -> u32 {
    if g.game.corporations[ci].corp_type == CorpType::Minor { 2 } else { 5 }
}

/// Ruby `buying_power(entity, full: true)`: the cash and what the loans still to take bring.
fn full_power(g: &SiteGame, ci: usize) -> i32 {
    let c = &g.game.corporations[ci];
    let left = max_loans(g, ci).saturating_sub(c.loans) as i32;
    let left = if g.game.loans_remaining == 0 { 0 } else { left.min(g.game.loans_remaining as i32) };
    c.cash + left * LOAN_NET
}

/// Ruby's `pass_description` of the active step.
fn pass_description(g: &SiteGame, kind: StepKind) -> String {
    match kind {
        StepKind::SingleItemAuction => match &g.game.round {
            Round::Auction(s) => match s.auctioning.and_then(|ci| g.game.companies.get(ci)) {
                Some(c) => format!("Pass (on {})", c.sym),
                None => "Pass".into(),
            },
            _ => "Pass".into(),
        },
        StepKind::BuySellParShares => match &g.game.round {
            Round::Stock(s) if s.bid_auction.is_some() => "Pass (Bid)".into(),
            Round::Stock(s) if s.acted_this_turn => "Done (Share)".into(),
            _ => "Pass (Share)".into(),
        },
        StepKind::Track => match &g.game.round {
            Round::Operating(s) if s.num_laid_track > 0 => "Done (Track)".into(),
            _ => "Skip (Track)".into(),
        },
        StepKind::Token => "Skip (Token)".into(),
        StepKind::BuyTrain => if g.turn_acted("buy_train") { "Done (Trains)".into() } else { "Skip (Trains)".into() },
        StepKind::BuyCompany | StepKind::BuyCompanyPreloan => "Skip (Companies)".into(),
        StepKind::RedeemShares => if g.turn_acted("buy_shares") { "Done (Redeem)".into() } else { "Skip (Redeem)".into() },
        StepKind::Merge => match &g.game.round {
            Round::Merger(s) if !s.merging.is_empty() => "Done Adding Corporations".into(),
            _ => "Pass".into(),
        },
        _ => "Pass".into(),
    }
}

/// The generator's moves now (Ruby's set) and their wire forms.
struct Moves {
    list: Vec<(Move, Value)>,
}

impl Moves {
    fn of(g: &mut SiteGame) -> Moves {
        let moves = g.game.g1867_legal_moves_with(CrossPlayerPolicy::RULES);
        let list = moves.into_iter().map(|m| {
            let w = g.game.g1867_move_wire(&m);
            (m, w)
        }).collect();
        Moves { list }
    }
    fn of_type<'a>(&'a self, t: &'a str) -> impl Iterator<Item = &'a (Move, Value)> + 'a {
        self.list.iter().filter(move |(m, _)| m.action.action_type() == t)
    }
}

fn wire_action(w: &Value) -> Value {
    let mut a = w.clone();
    if let Some(m) = a.as_object_mut() {
        m.remove("price");
        m.remove("routes");
    }
    a
}

/// The opening auction's lot and the minors' auctions (Ruby's SingleItemAuction, BuySellParSharesViaBid).
fn auction_parts(g: &SiteGame, pid: u32, out: &mut Map<String, Value>, moves: &Moves) {
    let cash = g.game.players.iter().find(|p| p.id == pid).map_or(0, |p| p.cash);
    match &g.game.round {
        Round::Auction(s) => {
            // the lot up for auction is the only one (SingleItemAuction: one company at a time)
            let Some(ci) = s.auctioning else { return };
            let c = &g.game.companies[ci];
            let dutch = s.single_item.as_ref().is_some_and(|x| x.dutch_mode);
            out.insert(
                "auction".into(),
                json!([{"sym": c.sym, "name": m1::company_name(g, &c.sym), "value": c.value,
                        "minBid": g.game.sa_min_bid(s, ci), "maxBid": cash, "buy": dutch}]),
            );
            out.insert("bidStep".into(), json!({"increment": 5, "multiple": true}));
            let mut bidders: Vec<(i32, u32)> = s.bids.get(&ci).map(|b| b.iter().map(|x| (x.price, x.player_id)).collect()).unwrap_or_default();
            bidders.sort_by(|a, b| b.0.cmp(&a.0));
            let list: Vec<Value> =
                bidders.iter().map(|(price, p)| json!({"id": p, "name": m1::player_name(g, *p), "price": price})).collect();
            out.insert("auctioning".into(), json!({"sym": c.sym, "name": m1::company_name(g, &c.sym), "bidders": list}));
        }
        Round::Stock(s) => {
            let par = |g: &SiteGame| {
                // the par a winning bid gives: the highest minor par at most min(bid / 2, $135)
                let mut prices: Vec<i32> = g
                    .game
                    .stock_market
                    .share_prices_with_types(g.game.title_def().par_price_types(CorpType::Minor))
                    .into_iter()
                    .map(|sp| sp.price)
                    .collect();
                prices.sort_by(|a, b| b.cmp(a));
                prices.dedup();
                json!({"prices": prices, "cap": MAX_MINOR_PAR})
            };
            if let Some(a) = &s.bid_auction {
                let high = a.bids.iter().map(|b| b.price).max().unwrap_or(0);
                let sym = &a.corporation_sym;
                out.insert(
                    "auction".into(),
                    json!([{"sym": sym, "name": corp_full_name(g, sym), "value": 0, "target": "corporation",
                            "minBid": high + 5, "maxBid": cash, "buy": false, "par": par(g)}]),
                );
                out.insert("bidStep".into(), json!({"increment": 5, "multiple": true}));
                let mut bidders: Vec<(i32, u32)> = a.bids.iter().map(|b| (b.price, b.player_id)).collect();
                bidders.sort_by(|x, y| y.0.cmp(&x.0));
                let list: Vec<Value> =
                    bidders.iter().map(|(price, p)| json!({"id": p, "name": m1::player_name(g, *p), "price": price})).collect();
                out.insert("auctioning".into(), json!({"sym": sym, "name": corp_full_name(g, sym), "bidders": list}));
                return;
            }
            // a stock-round turn may put a minor up for auction: the minors the generator lets this player start
            let mut lots: Vec<(usize, Value)> = Vec::new();
            for (mv, _) in moves.of_type("bid") {
                let Action::CorporationBid { corporation_sym, .. } = &mv.action else { continue };
                if lots.iter().any(|(_, l)| l["sym"] == json!(corporation_sym)) {
                    continue;
                }
                let lo = mv.price.map_or(100, |r| r.lo);
                let pos = super::ruby_corporations(&g.game)
                    .iter()
                    .position(|&ci| g.game.corporations[ci].sym == *corporation_sym)
                    .unwrap_or(usize::MAX);
                lots.push((pos, json!({"sym": corporation_sym, "name": corp_full_name(g, corporation_sym), "value": 0,
                    "target": "corporation", "minBid": lo, "maxBid": cash, "buy": false, "par": par(g)})));
            }
            if lots.is_empty() {
                return;
            }
            lots.sort_by_key(|l| l.0);
            out.insert("auctionStart".into(), json!(true));
            out.insert("auction".into(), json!(lots.into_iter().map(|l| l.1).collect::<Vec<_>>()));
            out.insert("bidStep".into(), json!({"increment": 5, "multiple": true}));
        }
        _ => {}
    }
}

/// Shares the player may buy (living-board RawBuy): the generator's buys, in Ruby's corporation order, the treasury's
/// share before the market's.
fn buy_options(g: &SiteGame, pid: u32, moves: &Moves) -> Value {
    let order = super::ruby_corporations(&g.game);
    let mut out: Vec<((usize, u8), Value)> = Vec::new();
    for (mv, w) in moves.of_type("buy_shares") {
        let Action::BuyShares { corporation_sym, source, percent, .. } = &mv.action else { continue };
        if w["entity"] != json!(pid) {
            continue;
        }
        let Some(&ci) = g.game.corp_idx.get(corporation_sym.as_str()) else { continue };
        let c = &g.game.corporations[ci];
        let from_market = source == "market"
            || mv.shares.first().and_then(|id| id.rsplit_once('_')).and_then(|(_, i)| i.parse::<usize>().ok())
                .and_then(|i| c.shares.iter().find(|s| s.index == i)).is_some_and(|s| s.owner.is_market());
        let kind = if from_market { "market" } else { "IPO" };
        if out.iter().any(|(_, o)| o["corp"] == json!(corporation_sym) && o["kind"] == json!(kind)) {
            continue;
        }
        let price = c.share_price.as_ref().map_or(0, |s| s.price) * (*percent as i32) / c.share_unit_percent.max(1) as i32;
        let held: u32 = c.shares.iter().filter(|s| s.owner.is_player() || s.owner.is_market()).map(|s| s.percent as u32).sum();
        let floats = kind == "IPO" && !c.floated && held + *percent as u32 >= c.float_percent as u32;
        let pos = order.iter().position(|&x| x == ci).unwrap_or(usize::MAX);
        out.push((
            (pos, (kind == "market") as u8),
            json!({"corp": corporation_sym, "kind": kind, "percent": percent, "price": price, "floats": floats,
                   "action": {"type": "buy_shares", "entity": pid, "entity_type": "player", "shares": mv.shares, "percent": percent}}),
        ));
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    json!(out.into_iter().map(|x| x.1).collect::<Vec<_>>())
}

/// Majors the player may start (phase 4 on): the generator's pars, prices high to low.
fn par_options(g: &SiteGame, pid: u32, moves: &Moves) -> Value {
    let cash = g.game.players.iter().find(|p| p.id == pid).map_or(0, |p| p.cash);
    let order = super::ruby_corporations(&g.game);
    let mut by_corp: Vec<(usize, String, Vec<(i32, Value)>)> = Vec::new();
    for (mv, w) in moves.of_type("par") {
        let Action::Par { corporation_sym, share_price, .. } = &mv.action else { continue };
        let Some(&ci) = g.game.corp_idx.get(corporation_sym.as_str()) else { continue };
        let pos = order.iter().position(|&x| x == ci).unwrap_or(usize::MAX);
        let entry = match by_corp.iter_mut().find(|x| &x.1 == corporation_sym) {
            Some(e) => e,
            None => {
                by_corp.push((pos, corporation_sym.clone(), Vec::new()));
                by_corp.last_mut().unwrap()
            }
        };
        entry.2.push((*share_price, json!({"price": share_price, "action": wire_action(w)})));
    }
    by_corp.sort_by_key(|x| x.0);
    let out: Vec<Value> = by_corp
        .into_iter()
        .map(|(_, sym, mut prices)| {
            prices.sort_by(|a, b| b.0.cmp(&a.0));
            let ci = g.game.corp_idx[sym.as_str()];
            let c = &g.game.corporations[ci];
            json!({"corp": sym, "cash": cash, "floatPercent": c.float_percent, "presPercent": c.president_percent(),
                   "sharePercent": c.share_unit_percent, "prices": prices.into_iter().map(|p| p.1).collect::<Vec<_>>()})
        })
        .collect();
    json!(out)
}

/// Track (G1867::Step::Track with AutomaticLoan): the hexes with a lay the company can pay with its loans, those it
/// can't (`tileDear`: the terrain and the second lay's $20 beyond its power), and that power (`tilePower`).
fn track_parts(g: &mut SiteGame, corp: &str, moves: &Moves, out: &mut Map<String, Value>) {
    let Some(&ci) = g.game.corp_idx.get(corp) else { return };
    let power = full_power(g, ci);
    let fee = match &g.game.round {
        Round::Operating(s) if s.num_laid_track >= 1 => 20,
        _ => 0,
    };
    let mut legal: Vec<String> = Vec::new();
    for (mv, _) in moves.of_type("lay_tile") {
        if let Action::LayTile { hex_id, .. } = &mv.action {
            if !legal.contains(hex_id) {
                legal.push(hex_id.clone());
            }
        }
    }
    // hexes with a legal lay the company can't pay for even with its loans: the generator on a copy whose treasury
    // is topped up, less the real set; each with its cheapest lay's cost (the terrain, the second lay's $20 and the
    // water borders the new track completes, as the engine charges them)
    let mut rich = g.game.clone_for_search();
    rich.corporations[ci].cash += 100_000;
    let mut rich_cost: Vec<(String, i32)> = Vec::new();
    for m in rich.g1867_legal_moves_with(CrossPlayerPolicy::RULES) {
        if let Action::LayTile { hex_id, tile_id, rotation, .. } = &m.action {
            if legal.contains(hex_id) && !rich_cost.iter().any(|x| &x.0 == hex_id) {
                continue;
            }
            let base = tile_id.split('-').next().unwrap_or(tile_id);
            let cost = m1::terrain_of(g, hex_id).0 + fee + g.game.tile_border_cost(hex_id, base, *rotation).0;
            match rich_cost.iter_mut().find(|x| &x.0 == hex_id) {
                Some(x) => x.1 = x.1.min(cost),
                None => {
                    rich_cost.push((hex_id.clone(), cost));
                    legal.push(hex_id.clone());
                }
            }
        }
    }
    legal.sort_by_key(|h| hex_pos(h));
    let mut hexes = Vec::new();
    let mut dear = Vec::new();
    for hex in legal {
        let (t, mut terrain) = m1::terrain_of(g, &hex);
        let mut cost = t + fee;
        // a hex only the topped-up treasury can lay on: dear at its cheapest lay's cost (a water border the
        // terrain doesn't show: Ruby refuses every tile there, "AE has 65 and cannot spend 80")
        if let Some(&(_, c)) = rich_cost.iter().find(|x| x.0 == hex) {
            if c > power {
                if c > cost && !terrain.iter().any(|x| x == "water") {
                    terrain.push("water".into());
                }
                cost = c;
            }
        }
        if cost > power {
            dear.push(json!({"hex": hex, "cost": cost, "terrain": terrain}));
        } else {
            hexes.push(hex);
        }
    }
    out.insert("tilePower".into(), json!(power));
    out.insert("tileHexes".into(), json!(hexes));
    if !dear.is_empty() {
        out.insert("tileDear".into(), json!(dear));
    }
}

/// Stations (HomeToken: any open city, $0; Token: $40 × the hexes from the nearest own station, cash only).
fn token_options(g: &SiteGame, moves: &Moves, kind: StepKind) -> Value {
    let mut out: Vec<((usize, u8), Value)> = Vec::new();
    for (mv, w) in moves.of_type("place_token") {
        if mv.step != kind {
            continue;
        }
        let Action::PlaceToken { hex_id, city_index, entity_id } = &mv.action else { continue };
        let cities = g.game.hex_ref(hex_id).map_or(0, |h| h.tile.cities.len());
        let price = if kind == StepKind::HomeToken { 0 } else { token_price(g, entity_id, hex_id) };
        out.push(((hex_pos(hex_id), *city_index), json!({"hex": hex_id, "city": city_index, "cities": cities, "price": price, "action": w})));
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    json!(out.into_iter().map(|x| x.1).collect::<Vec<_>>())
}

/// G1867::Step::Token#adjust_token_price_ability!: the token's price times the hex distance from the company's
/// nearest placed station.
fn token_price(g: &SiteGame, corp: &str, hex: &str) -> i32 {
    let Some(&ci) = g.game.corp_idx.get(corp) else { return 0 };
    let c = &g.game.corporations[ci];
    let Some(base) = c.tokens.iter().find(|t| !t.used).map(|t| t.price) else { return 0 };
    base * g.game.min_token_hex_distance(ci, hex).unwrap_or(1)
}

/// Dividends of a major (G1867::Step::Dividend: payout, half, withhold), each previewed.
fn dividend_options(g: &SiteGame, corp: &str, ent: &Value) -> Value {
    let Some(&ci) = g.game.corp_idx.get(corp) else { return json!([]) };
    let revenue = match &g.game.round {
        Round::Operating(s) => s.revenue,
        _ => 0,
    };
    let c = &g.game.corporations[ci];
    let sp = c.share_price.clone();
    let mut out = Vec::new();
    for kind in ["payout", "half", "withhold"] {
        let action = m1::with(m1::with(json!({"type": "dividend"}), ent.clone()), json!({"kind": kind}));
        let mut clone = g.game.clone_for_search();
        let ok = super::super::json::action_from_json(&action).is_ok_and(|a| clone.process_action_internal(&a).is_ok());
        let preview = if ok {
            // Engine::Step::HalfPay#half: half (to the $10 below) withheld, the rest paid out per share
            let withheld = match kind {
                "payout" => 0,
                "half" => revenue / 2 / 10 * 10,
                _ => revenue,
            };
            let per = (revenue - withheld) / 10;
            let players: Vec<Value> = super::super::view::player_order(g)
                .into_iter()
                .filter_map(|pid| {
                    let n = c.shares.iter().filter(|s| s.owner == EntityId::player(pid)).map(|s| s.percent as i32 / 10).sum::<i32>();
                    (per * n > 0).then(|| json!({"id": pid, "amount": per * n}))
                })
                .collect();
            let to = clone.corporations[ci].share_price.as_ref().map(|s| s.price);
            json!({"kind": kind, "revenue": revenue, "perShare": per, "toTreasury": withheld, "players": players,
                   "priceFrom": sp.as_ref().map(|s| s.price), "priceTo": to})
        } else {
            Value::Null
        };
        out.push(json!({"action": action, "kind": kind, "preview": preview}));
    }
    json!(out)
}

/// Ruby's `depot.depot_trains`: the next train, the later types already available, one of each discarded type.
fn depot_trains(g: &SiteGame) -> Vec<(String, String, i32, bool)> {
    let game = &g.game;
    let mut out: Vec<(String, String, i32, bool)> = Vec::new();
    let mut names: Vec<String> = Vec::new();
    for t in game.depot.trains.first().into_iter().chain(game.depot.trains.iter().filter(|t| game.phase_available(t.available_on.as_deref()))) {
        if !names.contains(&t.name) {
            names.push(t.name.clone());
            out.push((t.id.clone(), t.name.clone(), t.price, false));
        }
    }
    let mut dn: Vec<String> = Vec::new();
    for t in &game.depot.discarded {
        if !dn.contains(&t.name) {
            dn.push(t.name.clone());
            out.push((t.id.clone(), t.name.clone(), t.price, true));
        }
    }
    out
}

/// Trains (G1867::Step::BuyTrain with AutomaticLoan): the depot's (each at its price, those beyond the buying power
/// too, `short`; the loans the company takes for one it must buy), and `trainBuy` (who pays, the president's other
/// companies' trains, the phase-8 trade-ins).
fn train_parts(g: &mut SiteGame, corp: &str, types: &[String], moves: &Moves, out: &mut Map<String, Value>) {
    let Some(&ci) = g.game.corp_idx.get(corp) else { return };
    let cash = g.game.corporations[ci].cash;
    let pres = g.game.corporations[ci].president_id();
    let pres_cash = pres.and_then(|p| g.game.players.iter().find(|x| x.id == p)).map_or(0, |p| p.cash);
    // G1867 BuyTrain#must_buy_train? (trainless, a depot train affordable with every loan): its actions are then
    // `buy_train` alone, and only then the loans count in the buying power
    let must = types.iter().any(|t| t == "buy_train") && !types.iter().any(|t| t == "pass");
    let power = if must { full_power(g, ci) } else { cash };
    let depot = depot_trains(g);
    // the generator's room check (`g67_gen_buy_train`: a train beyond the limit is never bought, only traded in):
    // at its limit a company (phase 8, the step open for a trade-in) is offered no depot train, as act's check
    // refuses one (Ruby's buy_train_action would take it and then ask for a discard; Opal's hand lists it)
    let room = g.game.corporations[ci].trains.len() < g.game.corp_train_limit(ci);
    let mut trains = Vec::new();
    let mut seen: Vec<(String, i32)> = Vec::new();
    for (id, name, price, discarded) in depot.iter().filter(|_| room) {
        if seen.contains(&(name.clone(), *price)) {
            continue;
        }
        seen.push((name.clone(), *price));
        let short = (price - power).max(0);
        let loans = if must && *price > cash && *price <= power { ((price - cash) as f64 / LOAN_NET as f64).ceil() as i32 } else { 0 };
        let mut t = json!({"train": id, "name": name, "price": price, "discarded": discarded, "short": short,
            "mayContribute": false, "presCash": pres_cash, "cheapest": null,
            "action": {"type": "buy_train", "entity": corp, "entity_type": "corporation", "train": id, "price": price, "variant": name}});
        if loans > 0 {
            t["loans"] = json!({"n": loans, "per": LOAN_NET});
        }
        trains.push(t);
    }
    // another company's trains: only the president's own companies' (the site's rule), one offer per company and
    // type (its first such train), at the generator's range
    let mut others: Vec<Value> = Vec::new();
    for (mv, _) in moves.of_type("buy_train") {
        let Action::BuyTrain { train_name, exchange: None, .. } = &mv.action else { continue };
        let Some(oi) = g.game.corporations.iter().position(|c| c.sym != corp && c.trains.iter().any(|t| &t.id == train_name)) else { continue };
        if pres.is_none() || g.game.corporations[oi].president_id() != pres {
            continue;
        }
        let from = g.game.corporations[oi].sym.clone();
        let name = g.game.corporations[oi].trains.iter().find(|t| &t.id == train_name).map(|t| t.name.clone()).unwrap_or_default();
        if others.iter().any(|o| o["from"] == json!(from) && o["name"] == json!(name)) {
            continue;
        }
        // (Ruby's Depot#other_trains: the game's trains in the order they were made, so the lowest instance first)
        let idx = |id: &str| id.rsplit_once('-').and_then(|x| x.1.parse::<u32>().ok()).unwrap_or(u32::MAX);
        let first = g.game.corporations[oi].trains.iter().filter(|t| t.name == name).min_by_key(|t| idx(&t.id)).map(|t| t.id.clone()).unwrap_or_default();
        let count = g.game.corporations[oi].trains.iter().filter(|t| t.name == name).count();
        let face = g.game.title_def().trains().iter().find(|t| t.name == name).map_or(0, |t| t.price);
        let (lo, hi) = mv.price.map_or((1, cash), |r| (r.lo, r.hi));
        others.push(json!({"train": first, "name": name, "count": count, "from": from, "owner": pres,
            "ownerName": pres.map(|p| m1::player_name(g, p)), "min": lo, "max": hi, "face": face, "fixed": null,
            "action": {"type": "buy_train", "entity": corp, "entity_type": "corporation", "train": first}}));
    }
    others.sort_by(|a, b| a["from"].as_str().cmp(&b["from"].as_str()).then(a["name"].as_str().cmp(&b["name"].as_str())));
    // trade-ins (G1867 BuyTrain#discountable_trains_allowed?: phase 8): Ruby discountable_trains_for, the company's
    // trains each against the depot's discounted ones
    let mut exchanges = Vec::new();
    if g.game.phase.name == "8" {
        let mut seen_ex: Vec<String> = Vec::new();
        let own: Vec<(String, String)> = g.game.corporations[ci].trains.iter().map(|t| (t.id.clone(), t.name.clone())).collect();
        for (oid, oname) in &own {
            for (did, dname, dprice, _) in &depot {
                let Some(dt) = g.game.depot.trains.iter().chain(g.game.depot.discarded.iter()).find(|t| &t.id == did) else { continue };
                let Some((_, d)) = dt.discount.iter().find(|(n, _)| n == oname) else { continue };
                let price = dprice - d;
                // a 5+5E for a 2+2 "costs" $600 - $750: Ruby's discountable_trains_for lists it and Spender#spend
                // refuses the negative amount; the generator never offers it (g67_gen_buy_train)
                if price <= 0 {
                    continue;
                }
                let key = format!("{oname}|{dname}|{price}");
                if seen_ex.contains(&key) {
                    continue;
                }
                seen_ex.push(key);
                exchanges.push(json!({"old": oname, "variant": dname, "price": price, "full": dprice, "buyingPower": cash,
                    "action": {"type": "buy_train", "entity": corp, "entity_type": "corporation", "train": did, "price": price,
                               "variant": dname, "exchange": oid}}));
            }
        }
    }
    let cheapest = depot.iter().map(|t| t.2).min();
    out.insert("trains".into(), json!(trains));
    out.insert(
        "trainBuy".into(),
        json!({"corp": corp, "cash": cash, "buyingPower": cash, "mustBuy": must, "presidentMayContribute": false,
               "president": pres.map(|p| json!({"id": p, "name": m1::player_name(g, p), "cash": pres_cash})),
               "others": others, "exchanges": exchanges, "sell": [], "cheapest": cheapest}),
    );
}

/// The privates the company may buy now (the site's rule: only its president's own), when a BuyCompany step is open.
fn company_buy(g: &SiteGame, corp: &str, blocking: bool) -> Option<Value> {
    let game = &g.game;
    if !game.phase_has_status("can_buy_companies") {
        return None;
    }
    let ci = *game.corp_idx.get(corp)?;
    let c = &game.corporations[ci];
    let cash = c.cash;
    let pres = c.president_id();
    let mut offers = Vec::new();
    let mut comps: Vec<&crate::entities::Company> = game
        .companies
        .iter()
        .filter(|m| !m.closed && m.owner.is_player() && pres.is_some_and(|p| m.owner == EntityId::player(p)))
        .collect();
    comps.sort_by_key(|m| m.value);
    for m in comps {
        let max = m.value.min(cash);
        if max < 1 {
            continue;
        }
        offers.push(json!({"sym": m.sym, "name": m1::company_name(g, &m.sym), "value": m.value, "revenue": m.revenue,
            "min": 1, "max": max, "action": {"type": "buy_company", "entity": corp, "entity_type": "corporation", "company": m.sym}}));
    }
    Some(json!({"corp": corp, "cash": cash, "offers": offers, "blocking": blocking}))
}

/// The company's loans (facts the hand words: how many, the most, what one brings now, the interest a loan).
fn auto_loans(g: &SiteGame, ci: usize, kind: StepKind, must_buy: bool) -> Value {
    let c = &g.game.corporations[ci];
    let max = max_loans(g, ci);
    // what one loan brings where the step counts the loans in its buying power (Track's AutomaticLoan; BuyTrain's
    // when the company must buy): its $45; else the loan's value
    let full = kind == StepKind::Track || (kind == StepKind::BuyTrain && must_buy);
    let per = if full && max > c.loans { LOAN_NET } else { LOAN_VALUE };
    json!({"corp": c.sym, "loans": c.loans, "max": max, "per": per, "interest": INTEREST})
}

/// 1867's merger round, for the minor acting (G1867::Step::Merge): facts (§3.5).
fn merger_facts(g: &mut SiteGame, minor: &str, types: &[String], moves: &Moves) -> Value {
    let Round::Merger(s) = &g.game.round else { return Value::Null };
    let s = s.clone();
    let phase = if s.converting || s.merge_major { "major" } else if !s.merging.is_empty() { "adding" } else { "start" };
    let mut out = json!({"minor": minor, "phase": phase, "merging": s.merging, "partners": [], "othersLeftOut": [], "majors": []});
    let price_of = |g: &SiteGame, sym: &str| g.game.corp_idx.get(sym).and_then(|&ci| g.game.corporations[ci].share_price.as_ref().map(|p| p.price));
    // the conversion's par: the highest par / par_2 price at most the minor's price
    let convert_par = |g: &SiteGame, own: i32| -> Option<i32> {
        let mut ps: Vec<i32> = g.game.stock_market.share_prices_with_types(&["par", "par_2"]).into_iter().map(|p| p.price).filter(|p| *p <= own).collect();
        ps.sort();
        ps.last().copied()
    };
    if phase == "major" {
        out["mode"] = json!(if s.converting { "convert" } else { "merge" });
        let par = if s.converting {
            price_of(g, minor).and_then(|p| convert_par(g, p))
        } else {
            // Merge#merged_par_price: the merging minors' prices (Ruby's rule), snapped down to a par / par_2 cell
            let prices: Vec<i32> = s.merging.iter().filter_map(|x| price_of(g, x)).collect();
            match (prices.iter().min(), prices.iter().max()) {
                (Some(&lo), Some(&hi)) => convert_par(g, crate::rounds::merger::merged_major_par_target(lo, hi)),
                _ => None,
            }
        };
        out["par"] = json!(par);
        let majors: Vec<Value> = moves
            .of_type("merge")
            .filter_map(|(mv, w)| match &mv.action {
                Action::Merge { corporation_sym, .. } => Some(json!({"sym": corporation_sym, "action": w})),
                _ => None,
            })
            .collect();
        out["majors"] = json!(majors);
        return out;
    }
    let owner = g.game.corp_idx.get(minor).and_then(|&ci| g.game.corporations[ci].owner_id.player_id());
    let mut partners = Vec::new();
    let mut left = Vec::new();
    for (mv, w) in moves.of_type("merge") {
        let Action::Merge { corporation_sym, .. } = &mv.action else { continue };
        let other = g.game.corp_idx.get(corporation_sym.as_str()).and_then(|&ci| g.game.corporations[ci].owner_id.player_id());
        if other != owner {
            left.push(json!({"sym": corporation_sym, "owner": other, "ownerName": other.map(|p| m1::player_name(g, p))}));
        } else {
            partners.push(json!({"sym": corporation_sym, "price": price_of(g, corporation_sym), "action": w}));
        }
    }
    // Ruby's order (Merge#mergeable_candidates: the merging companies' connected cities as Graph#compute finds them,
    // each city's stations), not the generator's
    let base: Vec<String> = if s.merging.is_empty() { vec![minor.to_string()] } else { s.merging.clone() };
    let order = super::order::candidate_order(&g.game, &base);
    let pos = |v: &Value| order.iter().position(|x| Some(x.as_str()) == v["sym"].as_str()).unwrap_or(usize::MAX);
    partners.sort_by_key(|v| pos(v));
    left.sort_by_key(|v| pos(v));
    out["partners"] = json!(partners);
    out["othersLeftOut"] = json!(left);
    if phase == "start" && types.iter().any(|t| t == "convert") {
        let par = price_of(g, minor).and_then(|p| convert_par(g, p));
        out["convert"] = json!({"par": par, "action": {"type": "convert", "entity": minor, "entity_type": "corporation"}});
    }
    out
}

/// Stations the merged major must take off (G1867::Step::ReduceTokens): facts (§3.5).
fn reduce_tokens(g: &SiteGame, corp: &str, moves: &Moves) -> Value {
    let mut options: Vec<((usize, u8, u8), Value)> = Vec::new();
    for (mv, w) in moves.of_type("remove_token") {
        let Action::RemoveToken { hex_id, city_index, slot, .. } = &mv.action else { continue };
        let owner = g
            .game
            .hex_ref(hex_id)
            .and_then(|h| h.tile.cities.get(*city_index as usize))
            .and_then(|c| c.tokens.get(*slot as usize).cloned().flatten())
            .map(|t| t.corporation_id.clone());
        options.push(((hex_pos(hex_id), *city_index, *slot), json!({"hex": hex_id, "city": w["city"], "slot": slot, "owner": owner, "action": w})));
    }
    options.sort_by(|a, b| a.0.cmp(&b.0));
    // Ruby ReduceTokens#help: the limit, and two stations in one hex
    let same_hex = {
        let mut hexes: Vec<&str> = Vec::new();
        let mut dup = false;
        if let Some(&ci) = g.game.corp_idx.get(corp) {
            for t in g.game.corporations[ci].tokens.iter().filter(|t| t.used) {
                if hexes.contains(&t.city_hex_id.as_str()) {
                    dup = true;
                }
                hexes.push(&t.city_hex_id);
            }
        }
        dup
    };
    json!({"corp": corp, "issues": if same_hex { json!(["limit", "sameHex"]) } else { json!(["limit"]) },
           "options": options.into_iter().map(|x| x.1).collect::<Vec<_>>()})
}

/// The menu (RawMenu) of a 1867 player, or null when they needn't act.
pub fn menu(g: &mut SiteGame, pid: u32) -> Value {
    if g.finished() || !super::active_players(&mut g.game).contains(&pid) {
        return Value::Null;
    }
    let snap = g.game.round.clone();
    let Some((bi, entity)) = g.game.blocking_step_entity(&snap) else { return Value::Null };
    let Some(desc) = g.game.round_step_descs(&snap).get(bi).copied() else { return Value::Null };
    let kind = desc.kind;
    let raw: Vec<String> = g.game.step_actions(&desc, &entity, &snap).into_iter().map(str::to_string).collect();
    let types = super::ruby_order_of(kind, raw);
    let ent = ent_json(&entity);
    let actor = match &entity {
        StepEntity::Player(p) => json!({"type": "player", "id": p, "name": m1::player_name(g, *p)}),
        StepEntity::Corp(s) => json!({"type": "corporation", "id": s, "name": s}),
        StepEntity::Company(s) => json!({"type": "company", "id": s, "name": m1::company_name(g, s)}),
    };
    let moves = Moves::of(g);
    let mut m = Map::new();
    m.insert("actor".into(), actor);
    m.insert("description".into(), json!(super::view::step_label(&mut g.game)));
    m.insert("roundName".into(), super::view::round_info(g)["name"].clone());
    m.insert("passDescription".into(), json!(pass_description(g, kind)));
    m.insert("types".into(), json!(types));
    m.insert("unsupported".into(), json!([]));
    let is_player = matches!(entity, StepEntity::Player(_));
    if is_player {
        m.insert("programs".into(), json!(["program_merger_pass", "program_buy_shares", "program_share_pass"]));
    }
    let corp = match &entity {
        StepEntity::Corp(s) => Some(s.clone()),
        _ => None,
    };
    let mut unsupported: Vec<String> = Vec::new();
    for t in types.clone() {
        match t.as_str() {
            "bid" => auction_parts(g, pid, &mut m, &moves),
            "buy_shares" if is_player => {
                m.insert("buy".into(), buy_options(g, pid, &moves));
            }
            "buy_shares" => {
                // RedeemShares: the company buys its own shares back from the market
                let c = corp.clone().unwrap_or_default();
                let redeem: Vec<Value> = moves
                    .of_type("buy_shares")
                    .filter_map(|(mv, w)| match &mv.action {
                        Action::BuyShares { corporation_sym, percent, .. } => {
                            let price = g.game.corp_idx.get(corporation_sym.as_str()).and_then(|&ci| g.game.corporations[ci].share_price.as_ref().map(|s| s.price)).unwrap_or(0) * (*percent as i32) / 10;
                            Some(json!({"corp": c, "percent": percent, "price": price, "action": wire_action(w)}))
                        }
                        _ => None,
                    })
                    .collect();
                m.insert("redeem".into(), json!(redeem));
            }
            "sell_shares" if is_player => {
                m.insert("sell".into(), m1::sell_options(g, pid));
            }
            "par" => {
                m.insert("par".into(), par_options(g, pid, &moves));
            }
            "lay_tile" => {
                if let Some(c) = &corp {
                    track_parts(g, c, &moves, &mut m);
                }
            }
            "place_token" => {
                m.insert("tokens".into(), token_options(g, &moves, kind));
            }
            "run_routes" => {
                m.insert("routes".into(), json!({"pending": true}));
            }
            "dividend" => {
                if let Some(c) = &corp {
                    m.insert("dividend".into(), dividend_options(g, c, &ent));
                }
            }
            "buy_train" => {
                if let Some(c) = &corp {
                    train_parts(g, c, &types, &moves, &mut m);
                }
            }
            "discard_train" => {
                let list: Vec<Value> = moves
                    .of_type("discard_train")
                    .filter_map(|(mv, w)| match &mv.action {
                        Action::DiscardTrain { entity_id, train_name } => {
                            let name = train_name.rsplit_once('-').map_or(train_name.clone(), |x| x.0.to_string());
                            Some(json!({"corp": entity_id, "train": name, "own": Some(entity_id) == corp.as_ref(), "action": w}))
                        }
                        _ => None,
                    })
                    .collect();
                m.insert("discard".into(), json!(list));
            }
            "pass" => {
                m.insert("pass".into(), json!({"action": m1::with(json!({"type": "pass"}), ent.clone())}));
                match kind {
                    StepKind::Merge => {
                        let done = matches!(&g.game.round, Round::Merger(s) if !s.merging.is_empty());
                        m.insert("passKind".into(), json!(if done { "merger_done" } else { "merger_stay" }));
                    }
                    StepKind::MajorTrainless => {
                        m.insert("passKind".into(), json!("trainless_keep"));
                    }
                    StepKind::BuyTrain => {
                        // G1867::Step::BuyTrain#pass! nationalizes a company that ends its turn without a train
                        if let Some(c) = &corp {
                            let ci = g.game.corp_idx[c.as_str()];
                            if g.game.corporations[ci].trains.is_empty() && g.game.corporations[ci].floated {
                                m.insert("passNationalizes".into(), json!(c));
                            }
                        }
                    }
                    _ => {}
                }
            }
            "merge" | "convert" => {
                if !m.contains_key("merger") {
                    let minor = corp.clone().unwrap_or_default();
                    let f = merger_facts(g, &minor, &types, &moves);
                    m.insert("merger".into(), f);
                }
            }
            "remove_token" => {
                if let Some(c) = &corp {
                    m.insert("reduceTokens".into(), reduce_tokens(g, c, &moves));
                }
            }
            "choose" => {
                if let Some(c) = &corp {
                    m.insert("nationalize".into(), json!({"corp": c, "verb": "nationalize",
                        "action": {"type": "choose", "entity": c, "entity_type": "corporation", "choice": "nationalize"}}));
                }
            }
            "buy_company" => {}
            other => unsupported.push(other.to_string()),
        }
    }
    m.insert("unsupported".into(), json!(unsupported));
    if is_player && kind == StepKind::BuySellParShares && !matches!(&g.game.round, Round::Stock(s) if s.bid_auction.is_some()) {
        let (turn, blocks) = m1::share_turn(g, pid, m.get("buy").cloned().unwrap_or(json!([])));
        m.insert("turn".into(), turn);
        m.insert("blocks".into(), blocks);
    }
    if let Some(c) = &corp {
        let ci = g.game.corp_idx[c.as_str()];
        let must_buy = kind == StepKind::BuyTrain && types.iter().any(|t| t == "buy_train") && !types.iter().any(|t| t == "pass");
        m.insert("autoLoans".into(), auto_loans(g, ci, kind, must_buy));
        // a BuyCompany step offering buy_company to the company now (Ruby round.actions_for: the non-blocking one open
        // beside the turn's steps when can_buy_company?, or the blocking ones)
        let blocking = types.iter().any(|t| t == "buy_company");
        let me = StepEntity::Corp(c.clone());
        let open = blocking
            || g.game.step_offers().iter().any(|o| {
                matches!(o.kind, StepKind::BuyCompany | StepKind::BuyCompanyPreloan) && o.entity == me && o.types.contains(&"buy_company")
            });
        if open {
            if let Some(cb) = company_buy(g, c, blocking) {
                m.insert("companyBuy".into(), cb);
            }
        }
    }
    Value::Object(m)
}

/// Every tile the generator allows on a hex now, in every legal rotation (living-board TileOption): Ruby's next
/// instance of each name in Ruby's supply order, `cost` the hex's terrain plus the lay's own fee (the second lay's $20,
/// `fee`), as Opal computes them.
pub fn tile_options(g: &mut SiteGame, hex: &str) -> Result<Value, String> {
    let Some(corp) = g.game.round_state.active_entity_id.corp_sym().map(str::to_string) else {
        return Err("refused: no company is laying track".into());
    };
    let moves = g.game.g1867_legal_moves_with(CrossPlayerPolicy::RULES);
    let mut lays: Vec<(String, u8)> = Vec::new();
    for m in &moves {
        if let Action::LayTile { hex_id, tile_id, rotation, .. } = &m.action {
            if hex_id == hex {
                let name = super::super::json::tile_name(tile_id).unwrap_or(tile_id).to_string();
                if !lays.contains(&(name.clone(), *rotation)) {
                    lays.push((name, *rotation));
                }
            }
        }
    }
    let (terrain_cost, terrain) = m1::terrain_of(g, hex);
    let fee = match &g.game.round {
        Round::Operating(s) if s.num_laid_track >= 1 => 20,
        _ => 0,
    };
    let mut names: Vec<String> = Vec::new();
    for (n, _) in &lays {
        if !names.contains(n) {
            names.push(n.clone());
        }
    }
    names.sort_by_key(|n| g.supply.rank(n));
    let mut out = Vec::new();
    for name in names {
        let Some(id) = g.next_tile(&name) else { continue };
        let color = g.game.tile_catalog.get(&name).map(|t| format!("{:?}", t.color).to_lowercase()).unwrap_or_default();
        let mut rots: Vec<u8> = lays.iter().filter(|(n, _)| *n == name).map(|(_, r)| *r).collect();
        rots.sort();
        for r in rots {
            let mut o = json!({"tile": id, "name": name, "color": color, "rotation": r, "cost": terrain_cost + fee,
                "terrain": terrain, "action": {"type": "lay_tile", "entity": corp, "entity_type": "corporation", "hex": hex,
                "tile": id, "rotation": r}});
            if fee > 0 {
                o["fee"] = json!(fee);
            }
            out.push(o);
        }
    }
    Ok(json!(out))
}

/// The exact router's run for the acting company (`g1867_auto_routes`): {action (run_routes as the site posts it),
/// revenue, routes (each with its track)}, or null.
pub fn auto_route(g: &mut SiteGame) -> Value {
    let Some(corp) = g.game.round_state.active_entity_id.corp_sym().map(str::to_string) else { return Value::Null };
    if super::active_step(&mut g.game).map(|x| x.0) != Some(StepKind::Route) {
        return Value::Null;
    }
    let (routes, revenue) = g.game.clone_for_search().g1867_auto_routes(&corp);
    if routes.is_empty() {
        return Value::Null;
    }
    let site_routes: Vec<Value> = routes
        .iter()
        .map(|r| {
            json!({"train": r.train_name, "connections": r.connections, "hexes": r.hexes, "revenue": r.revenue,
                   "revenue_str": r.hexes.join("-"), "nodes": r.nodes})
        })
        .collect();
    let action = json!({"type": "run_routes", "entity": corp, "entity_type": "corporation", "routes": site_routes});
    let with_tracks: Vec<Value> = site_routes
        .iter()
        .map(|r| {
            let mut r = r.clone();
            if let Some(t) = super::super::track::route_track(g, &r) {
                r["track"] = t;
            }
            r
        })
        .collect();
    json!({"action": action, "revenue": revenue, "routes": with_tracks})
}
