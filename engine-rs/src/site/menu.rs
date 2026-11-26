//! The move menu as facts (RawMenu, docs/WASM_ENGINE_SPIKE.md §3.3; living-board `src/engine/rawmenu.ts`, whose Opal
//! twin is moves.ts `rawMenu`): what the player to act may do now, as complete site actions with the numbers, reason
//! codes and previews the hand needs. Previews are computed by processing the move on a clone (exact by
//! construction). Same own-only rule as the Opal menu: another player's trains and privates are never offered.

use super::game::SiteGame;
use super::json::SWAPPED_PREPRINTED_CITIES;
use crate::entities::EntityId;
use crate::rounds::{OperatingStep as S, Round};
use serde_json::{json, Map, Value};

/// 1830's hexes in Ruby's order (`game.hexes`: the order the site lists them in, which its menus follow).
const RUBY_HEX_ORDER: &[&str] = &[
    "F2", "I1", "J2", "A9", "A11", "K13", "B24", "D2", "F6", "E9", "H12", "D14", "C15", "K15", "A17", "A19", "I19", "F24",
    "D24", "F4", "J14", "F22", "E7", "F8", "C11", "C13", "D12", "B16", "C17", "B20", "D4", "F10", "I13", "D18", "B12",
    "B14", "B22", "C7", "C9", "C23", "D8", "D16", "D20", "E3", "E13", "E15", "F12", "F14", "F18", "G3", "G5", "G9", "G11",
    "H2", "H6", "H8", "H14", "I3", "I5", "I7", "I9", "J4", "J6", "J8", "G15", "C21", "D22", "E17", "E21", "G13", "I11",
    "J10", "J12", "E19", "H4", "B10", "H10", "H16", "F16", "G7", "G17", "F20", "D6", "I17", "B18", "C19", "E5", "D10",
    "E11", "H18", "I15", "G19", "E23",
];

fn ruby_hex_pos(hex: &str) -> usize {
    RUBY_HEX_ORDER.iter().position(|h| *h == hex).unwrap_or(usize::MAX)
}

fn pent(pid: u32) -> Value {
    json!({"entity": pid, "entity_type": "player"})
}

fn with(base: Value, extra: Value) -> Value {
    let mut m = base.as_object().cloned().unwrap_or_default();
    if let Value::Object(e) = extra {
        for (k, v) in e {
            m.insert(k, v);
        }
    }
    Value::Object(m)
}

fn player_name(g: &SiteGame, pid: u32) -> String {
    g.players.iter().find(|p| p.0 == pid).map(|p| p.1.clone()).unwrap_or_default()
}

fn company_name(g: &SiteGame, sym: &str) -> String {
    g.game.title_def().companies().iter().find(|c| c.sym == sym).map(|c| c.name.to_string()).unwrap_or_default()
}

/// The city id the site names (Ruby's numbering: G19 swapped while preprinted): "<tile id>-<city>".
pub fn city_id(g: &SiteGame, hex: &str, city: usize) -> String {
    let Some(h) = g.game.hex_ref(hex) else { return String::new() };
    let swapped = g.swapped.contains(&hex) && h.tile.cities.len() == 2 && SWAPPED_PREPRINTED_CITIES.contains(&hex);
    let c = if swapped { 1 - city } else { city };
    let tile = if super::json::tile_name(&h.tile.name).is_some() { h.tile.name.clone() } else { format!("{hex}-0") };
    format!("{tile}-{c}")
}

/// Ruby's step.actions(entity), ordered as Ruby lists them.
pub fn ruby_types(g: &mut SiteGame) -> Vec<String> {
    let mut types = g.game.legal_action_types();
    let order = [
        "bid", "par", "buy_shares", "buy_company", "sell_shares", "lay_tile", "place_token", "run_routes", "dividend",
        "discard_train", "buy_train", "bankrupt", "pass",
    ];
    // Ruby's stock step: buy_shares, par, buy_company, sell_shares, pass; Buy Trains with a must: sell_shares, buy_train
    let stock = matches!(g.game.round, Round::Stock(_));
    let rank = |t: &str| -> usize {
        if stock {
            ["buy_shares", "par", "buy_company", "sell_shares", "pass"].iter().position(|x| *x == t).unwrap_or(9)
        } else {
            order.iter().position(|x| *x == t).unwrap_or(20)
        }
    };
    types.sort_by_key(|t| rank(t));
    types.dedup();
    // Ruby's step.actions(entity) is the ACTIVE step's: the steps open beside it (a private's lay, buying a private
    // during the turn, M&H's exchange) are offered by their own parts of the menu
    let allowed: &[&str] = match &g.game.round {
        Round::Operating(s) if !s.crowded_corps.is_empty() => &["discard_train"],
        Round::Operating(s) if s.teleport_pending => &["place_token", "pass"],
        Round::Operating(s) => match s.step {
            S::LayTile => &["lay_tile", "pass"],
            S::PlaceToken => &["place_token", "pass"],
            S::RunRoutes => &["run_routes"],
            S::Dividend => &["dividend"],
            S::BuyTrain => &["sell_shares", "buy_train", "pass"],
            S::BuyCompany => &["buy_company", "pass"],
            _ => &[],
        },
        Round::Stock(_) => &["buy_shares", "par", "buy_company", "sell_shares", "pass"],
        Round::Auction(_) => &["bid", "par", "pass"],
        Round::Merger(_) => &[],
    };
    types.retain(|t| allowed.contains(&t.as_str()));
    if let Round::Auction(a) = &g.game.round {
        if a.pending_par.is_none() {
            types = vec!["bid".into(), "pass".into()];
        }
    }
    if let Round::Operating(st) = &g.game.round {
        if st.crowded_corps.is_empty() && !st.teleport_pending {
            // Ruby's operating steps list their actions whatever the position
            let fixed: Option<&[&str]> = match st.step {
                S::LayTile => Some(&["lay_tile", "pass"]),
                S::PlaceToken if st.pending_tokens.is_empty() => Some(&["place_token", "pass"]),
                S::PlaceToken => Some(&["place_token"]),
                S::RunRoutes => Some(&["run_routes"]),
                S::Dividend => Some(&["dividend"]),
                S::BuyCompany => None,
                _ => None,
            };
            if let Some(f) = fixed {
                types = f.iter().map(|x| x.to_string()).collect();
            }
            if st.step == S::BuyCompany {
                let corp = st.current_corp_sym().unwrap_or("").to_string();
                types = if ruby_buy_company_open(g, &corp) { vec!["buy_company".into(), "pass".into()] } else { vec!["pass".into()] };
            }
            if st.step == S::BuyTrain {
                if let Some(corp) = st.current_corp_sym().map(str::to_string) {
                    types = if ruby_must_buy(g, &corp) {
                        vec!["sell_shares".into(), "buy_train".into()]
                    } else {
                        vec!["buy_train".into(), "pass".into()]
                    };
                }
            }
        }
    }
    types
}

/// Ruby Game#must_buy_train?: no train, trains in the depot, and a route to run (president_may_contribute? too).
fn ruby_must_buy(g: &mut SiteGame, corp: &str) -> bool {
    let Some(&ci) = g.game.corp_idx.get(corp) else { return false };
    g.game.corporations[ci].trains.is_empty() && !g.game.depot.trains.is_empty() && g.game.must_buy_train(corp)
}

/// Ruby's depot_trains: the next train, those of later types already available, and one of each discarded type.
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

/// The legal action types of every step open now (Ruby's round.actions_for(entity)).
fn open_types(g: &mut SiteGame) -> Vec<String> {
    g.game.legal_action_types()
}

pub fn menu(g: &mut SiteGame, pid: u32) -> Value {
    if g.finished() || !super::view::active_players(g).contains(&pid) {
        return Value::Null;
    }
    let types = ruby_types(g);
    let cur = super::view::current_holder(g).unwrap_or_default();
    let (actor, ent) = if let Some(p) = cur.strip_prefix("p:") {
        let id: u32 = p.parse().unwrap_or(0);
        (json!({"type": "player", "id": id, "name": player_name(g, id)}), pent(id))
    } else if let Some(c) = cur.strip_prefix("c:") {
        (json!({"type": "corporation", "id": c, "name": c}), json!({"entity": c, "entity_type": "corporation"}))
    } else if let Some(m) = cur.strip_prefix("m:") {
        (json!({"type": "company", "id": m, "name": company_name(g, m)}), json!({"entity": m, "entity_type": "company"}))
    } else {
        return Value::Null;
    };
    let round_name = super::view::round_info(g)["name"].clone();
    let description = super::view::step_label(g);
    let mut m = Map::new();
    m.insert("actor".into(), actor.clone());
    m.insert("description".into(), json!(description));
    m.insert("roundName".into(), round_name);
    m.insert("passDescription".into(), json!(pass_description(g)));
    m.insert("types".into(), json!(types));
    m.insert("unsupported".into(), json!([]));
    let is_player = actor["type"] == "player";
    if is_player {
        m.insert("programs".into(), json!(["program_buy_shares", "program_share_pass"]));
    }
    let mut types = types;
    if matches!(g.game.round, Round::Stock(_)) && g.game.buyable_shares(pid).is_empty() {
        types.retain(|t| t != "buy_shares");
        m.insert("types".into(), json!(types));
    }
    let mut unsupported: Vec<String> = Vec::new();
    for t in types.clone() {
        match t.as_str() {
            "bid" => {
                m.insert("auction".into(), auction(g, pid));
                // Ruby WaterfallAuction: bids rise by at least $5, any amount above
                m.insert("bidStep".into(), json!({"increment": 5, "multiple": false}));
                if let Some(up) = g.game.auctioning_company() {
                    let mut bidders: Vec<(i32, u32)> = g.game.auction_bids(&up).into_iter().map(|(p, price)| (price, p)).collect();
                    bidders.sort_by(|a, b| b.0.cmp(&a.0));
                    let list: Vec<Value> =
                        bidders.iter().map(|(price, p)| json!({"id": p, "name": player_name(g, *p), "price": price})).collect();
                    m.insert("auctioning".into(), json!({"sym": up, "name": company_name(g, &up), "bidders": list}));
                }
            }
            "buy_shares" => {
                m.insert("buy".into(), buy_options(g, pid));
            }
            "sell_shares" if is_player => {
                m.insert("sell".into(), sell_options(g, pid));
            }
            "par" => {
                m.insert("par".into(), par_options(g, pid));
            }
            "lay_tile" => {
                let (hexes, dear) = tile_hexes(g);
                m.insert("tileHexes".into(), json!(hexes));
                if !dear.is_empty() {
                    m.insert("tileDear".into(), json!(dear));
                }
            }
            "place_token" => {
                m.insert("tokens".into(), token_options(g, &ent));
            }
            "run_routes" => {
                m.insert("routes".into(), json!({"pending": true}));
            }
            "dividend" => {
                m.insert("dividend".into(), dividend_options(g, &ent));
            }
            "buy_train" => {
                let (trains, buy) = train_options(g, pid);
                m.insert("trains".into(), trains);
                m.insert("trainBuy".into(), buy);
            }
            "sell_shares" => {}
            "discard_train" => {
                m.insert("discard".into(), discard_options(g));
            }
            "pass" => {
                m.insert("pass".into(), json!({"action": with(json!({"type": "pass"}), ent.clone())}));
            }
            "buy_company" if !is_player => {}
            "buy_company" => {}
            "bankrupt" => {}
            other => unsupported.push(other.to_string()),
        }
    }
    m.insert("unsupported".into(), json!(unsupported));
    if is_player && matches!(g.game.round, Round::Stock(_)) {
        let (turn, blocks) = share_turn(g, pid, m.get("buy").cloned().unwrap_or(json!([])));
        m.insert("turn".into(), turn);
        m.insert("blocks".into(), blocks);
    }
    if actor["type"] == "corporation" {
        let corp = actor["id"].as_str().unwrap_or("").to_string();
        let blocking = types.iter().any(|t| t == "buy_company");
        if ruby_buy_company_open(g, &corp) {
            if let Some(cb) = company_buy(g, &corp, blocking) {
                m.insert("companyBuy".into(), cb);
            }
        }
        let ab = ability_uses(g, &corp);
        if !ab.is_empty() {
            m.insert("abilities".into(), json!(ab));
        }
        if let Some(b) = bankrupt_offer(g, &corp) {
            m.insert("bankrupt".into(), b);
        }
    }
    let ex = exchange_offers(g, pid);
    if !ex.is_empty() {
        m.insert("exchange".into(), json!(ex));
    }
    Value::Object(m)
}

/// Ruby's pass_description of the active step.
fn pass_description(g: &SiteGame) -> Value {
    match &g.game.round {
        Round::Auction(s) => {
            if s.pending_par.is_some() {
                json!("Pass")
            } else {
                match g.game.auctioning_company() {
                    Some(c) => json!(format!("Pass (on {c})")),
                    None => json!("Pass"),
                }
            }
        }
        Round::Stock(s) => json!(if s.acted_this_turn { "Done (Share)" } else { "Pass (Share)" }),
        Round::Operating(s) => {
            if !s.crowded_corps.is_empty() {
                return json!("Pass");
            }
            if s.teleport_pending {
                return json!("Pass (Token)");
            }
            if s.step == S::PlaceToken && !s.pending_tokens.is_empty() {
                return json!("Pass");
            }
            json!(match s.step {
                S::LayTile => if s.num_laid_track > 0 { "Done (Track)" } else { "Skip (Track)" },
                S::PlaceToken => "Skip (Token)",
                // Ruby's BuyTrain step acted when it processed a purchase or an emergency sale (a private bought
                // meanwhile is the BuyCompany step's)
                S::BuyTrain => {
                    if g.turn_acted_at(Some("buy_train"), "BuyTrain") || g.turn_acted_at(Some("sell_shares"), "BuyTrain") {
                        "Done (Trains)"
                    } else {
                        "Skip (Trains)"
                    }
                }
                // 1830's OR has two BuyCompany steps; a buy_company is always processed by the first (non-blocking,
                // never passed), so the blocking one at the end never acted: Ruby's pass_description is "Skip"
                S::BuyCompany => "Skip (Companies)",
                _ => "Pass",
            })
        }
        Round::Merger(_) => json!("Pass"),
    }
}

/// The waterfall auction's lots (living-board MoveMenu `auction`): while a company is being auctioned it is the only
/// lot (the site's Ruby refuses bids on another private then: rl18xx-next 1b5c534, the auction lock).
fn auction(g: &SiteGame, pid: u32) -> Value {
    let game = &g.game;
    let comps = game.auction_companies();
    let auctioning = game.auctioning_company();
    let list: Vec<Value> = comps
        .iter()
        .enumerate()
        .filter(|(_, sym)| auctioning.as_ref().map_or(true, |up| up == *sym))
        .map(|(i, sym)| {
            let value = game.companies.iter().find(|c| &c.sym == sym).map_or(0, |c| c.value);
            json!({"sym": sym, "name": company_name(g, sym), "value": value, "minBid": game.auction_min_bid(sym),
                   "maxBid": game.auction_max_bid(pid, sym), "buy": i == 0 && auctioning.is_none()})
        })
        .collect();
    json!(list)
}

fn buy_options(g: &SiteGame, pid: u32) -> Value {
    let game = &g.game;
    let mut out = Vec::new();
    let mut list = game.buyable_shares(pid);
    // Ruby: corporations in order, the IPO's share before the market's
    let corp_pos = |s: &str| game.corporations.iter().position(|c| c.sym == s).unwrap_or(99);
    list.sort_by_key(|(c, src, _, _)| (corp_pos(c), if src == "ipo" { 0 } else { 1 }));
    for (corp, source, idx, price) in list {
        let Some(c) = game.corporations.iter().find(|c| c.sym == corp) else { continue };
        // the market's first share is the first to have arrived there (Ruby's pool list)
        let share = if source == "market" {
            c.shares.iter().filter(|s| s.owner.is_market() && !s.president).min_by_key(|s| s.acquired_seq).unwrap_or(&c.shares[idx])
        } else {
            &c.shares[idx]
        };
        let held: u32 = c.shares.iter().filter(|s| s.owner.is_player() || s.owner.is_market()).map(|s| s.percent as u32).sum();
        let ipo_left = 100 - held;
        let floats = source == "ipo" && !c.floated && 100 - ipo_left + share.percent as u32 >= c.float_percent as u32;
        out.push(json!({
            "corp": corp, "kind": if source == "ipo" { "IPO" } else { "market" }, "percent": share.percent, "price": price,
            "floats": floats,
            "action": {"type": "buy_shares", "entity": pid, "entity_type": "player", "shares": [format!("{}_{}", corp, share.index)], "percent": share.percent},
        }));
    }
    json!(out)
}

/// The share ids of a sale bundle, in Ruby's bundles_for_corporation order (ordinary shares as acquired, then the
/// president's certificate).
fn bundle_share_ids(g: &SiteGame, pid: u32, corp: &str, num: usize) -> Vec<String> {
    let me = EntityId::player(pid);
    let Some(c) = g.game.corporations.iter().find(|c| c.sym == corp) else { return vec![] };
    let mut shares: Vec<&crate::entities::Share> = c.shares.iter().filter(|s| s.owner == me).collect();
    shares.sort_by_key(|s| (s.president as u8, s.percent, s.acquired_seq));
    shares.iter().take(num).map(|s| format!("{}_{}", corp, s.index)).collect()
}

fn sell_options(g: &SiteGame, pid: u32) -> Value {
    let mut out = Vec::new();
    let corp_pos = |s: &str| g.game.corporations.iter().position(|c| c.sym == s).unwrap_or(99);
    let mut list = g.game.sellable_bundles(pid);
    list.sort_by_key(|(c, n, pct)| (corp_pos(c), *n, *pct));
    for (corp, num, pct) in list {
        let shares = bundle_share_ids(g, pid, &corp, num);
        let action = json!({"type": "sell_shares", "entity": pid, "entity_type": "player", "shares": shares, "percent": pct});
        let Some(c) = g.game.corporations.iter().find(|c| c.sym == corp) else { continue };
        let price = c.share_price.as_ref().map_or(0, |s| s.price) * pct as i32 / c.share_unit_percent.max(1) as i32;
        let sale = sale_preview(g, pid, &corp, pct, &action);
        if sale.is_null() && g.game.corporations.iter().any(|c| c.sym == corp && c.share_price.is_some()) {
            // the engine refuses this sale (an emergency sale beyond what's needed, the operating company's
            // presidency): not offered, exact by construction
            continue;
        }
        out.push(json!({"corp": corp, "percent": pct, "price": price, "sale": sale, "action": action}));
    }
    json!(out)
}

/// What a sale does, by processing it on a clone: the price and cell after, the player's percent left, the presidency.
fn sale_preview(g: &SiteGame, pid: u32, corp: &str, pct: u8, action: &Value) -> Value {
    let Some(c) = g.game.corporations.iter().find(|c| c.sym == corp) else { return Value::Null };
    let Some(sp) = c.share_price.clone() else { return Value::Null };
    let me = EntityId::player(pid);
    let mine = c.percent_owned_by(&me) as i64;
    let left = mine - pct as i64;
    let was_pres = c.president_id() == Some(pid);
    let mut clone = g.game.clone_for_search();
    let Ok(a) = super::json::action_from_json(action) else { return Value::Null };
    if clone.process_action_internal(&a).is_err() {
        return Value::Null;
    }
    let after = clone.corporations.iter().find(|x| x.sym == corp);
    let to = after.and_then(|x| x.share_price.clone());
    let pres = if !was_pres {
        json!({"kind": "none"})
    } else {
        let others: Vec<(u32, i64)> = g
            .game
            .players
            .iter()
            .filter(|p| p.id != pid)
            .map(|p| (p.id, c.percent_owned_by(&EntityId::player(p.id)) as i64))
            .filter(|x| x.1 > 0)
            .collect();
        let max = others.iter().map(|x| x.1).max().unwrap_or(0);
        let new_pres = after.and_then(|x| x.president_id());
        if left >= max || new_pres == Some(pid) {
            let tie: Vec<String> = if left == max {
                // Ruby's player_share_holders order: the order players took shares (the players' order here)
                let holders = g.holders.get(corp).cloned().unwrap_or_default();
                let mut t: Vec<(usize, u32)> = others
                    .iter()
                    .filter(|o| o.1 == max)
                    .map(|o| (holders.iter().position(|h| *h == o.0).unwrap_or(usize::MAX), o.0))
                    .collect();
                t.sort();
                t.into_iter().map(|(_, p)| player_name(g, p)).collect()
            } else {
                vec![]
            };
            json!({"kind": "keep", "tie": tie, "next": max})
        } else if let Some(np) = new_pres {
            let pct_np = others.iter().find(|o| o.0 == np).map_or(0, |o| o.1);
            json!({"kind": "pass", "to": {"id": np, "name": player_name(g, np), "pct": pct_np}})
        } else {
            json!({"kind": "keep", "tie": [], "next": max})
        }
    };
    json!({"priceFrom": sp.price, "priceTo": to.as_ref().map(|t| t.price), "cellFrom": [sp.row, sp.column],
           "cellTo": to.as_ref().map(|t| json!([t.row, t.column])), "left": left, "pres": pres})
}

fn par_options(g: &SiteGame, pid: u32) -> Value {
    let game = &g.game;
    let cash = game.players.iter().find(|p| p.id == pid).map_or(0, |p| p.cash);
    let mut prices = game.par_prices_with_coords();
    prices.sort_by_key(|p| -p.0);
    let action = |corp: &str, p: &(i32, usize, usize)| {
        json!({"type": "par", "entity": pid, "entity_type": "player", "corporation": corp, "share_price": format!("{},{},{}", p.0, p.1, p.2)})
    };
    let mut out = Vec::new();
    if let Round::Auction(s) = &game.round {
        if let Some((corp, _)) = &s.pending_par {
            let via = crate::abilities::par_trigger_company_for(&game.title, corp).unwrap_or("");
            let list: Vec<Value> = prices.iter().map(|p| json!({"price": p.0, "action": action(corp, p)})).collect();
            out.push(json!({"corp": corp, "via": via, "cash": cash, "floatPercent": 60, "presPercent": 20, "sharePercent": 10, "prices": list}));
            return json!(out);
        }
    }
    for c in &game.corporations {
        if c.ipo_price.is_some() || c.closed {
            continue;
        }
        let list: Vec<Value> = prices
            .iter()
            .filter(|p| p.0 * c.president_share_units() <= cash)
            .map(|p| json!({"price": p.0, "action": action(&c.sym, p)}))
            .collect();
        if !list.is_empty() {
            out.push(json!({"corp": c.sym, "cash": cash, "floatPercent": c.float_percent, "presPercent": c.president_percent(),
                            "sharePercent": c.share_unit_percent, "prices": list}));
        }
    }
    json!(out)
}

/// The stock turn so far and every share on the table that isn't offered, with the checks that fail.
fn share_turn(g: &SiteGame, pid: u32, buy: Value) -> (Value, Value) {
    let game = &g.game;
    let Round::Stock(s) = &game.round else { return (Value::Null, json!([])) };
    let mut bought = Vec::new();
    let mut sold = Vec::new();
    // Ruby's round.current_actions: BuySellParShares's own (a company's exchange is the Exchange step's)
    for a in g.turn_actions.iter().filter(|a| a.get("entity_type").and_then(Value::as_str) != Some("company")) {
        match a.get("type").and_then(Value::as_str) {
            Some("buy_shares") => bought.push(json!({"corp": share_corp(a), "what": "share"})),
            Some("par") => bought.push(json!({"corp": a.get("corporation"), "what": "par"})),
            Some("sell_shares") => sold.push(json!({"corp": share_corp(a), "pct": a.get("percent")})),
            _ => {}
        }
    }
    let certs = game.num_certs(pid);
    let limit = game.cert_limit as u32;
    let sellable = game.sellable_bundles(pid);
    let must_sell = if game.stock_must_sell(pid, &sellable, certs) {
        if certs > limit {
            json!({"kind": "certs"})
        } else {
            let me = EntityId::player(pid);
            let over: Vec<Value> = game
                .corporations
                .iter()
                .filter(|c| c.percent_owned_by(&me) > c.max_ownership_percent && !game.corp_holding_ok(c, pid))
                .map(|c| json!({"corp": c.sym, "pct": c.percent_owned_by(&me), "max": c.max_ownership_percent}))
                .collect();
            if over.is_empty() { json!({"kind": "other"}) } else { json!({"kind": "hold", "over": over}) }
        }
    } else {
        Value::Null
    };
    let did_buy = s.bought_this_turn || s.parred_this_turn;
    let multi_only_market = game.title_def().multiple_buy_only_from_market(&game.optional_rules);
    let turn = json!({"bought": bought, "sold": sold, "mustSell": must_sell, "certs": certs, "certLimit": limit,
                      "didBuy": did_buy, "multiOnlyMarket": multi_only_market});
    let offered: Vec<(String, String)> = buy
        .as_array()
        .map(|v| v.iter().map(|o| (o["corp"].as_str().unwrap_or("").to_string(), o["kind"].as_str().unwrap_or("").to_string())).collect())
        .unwrap_or_default();
    let me = EntityId::player(pid);
    let cash = game.players.iter().find(|p| p.id == pid).map_or(0, |p| p.cash);
    let mut blocks = Vec::new();
    let last_all_one_corp = |sym: &str| {
        !bought.is_empty() && bought.iter().all(|b| b["corp"] == sym && b["what"] == "share")
    };
    for c in &game.corporations {
        if c.ipo_price.is_none() || c.closed {
            continue;
        }
        let mut cands: Vec<(&str, i32)> = Vec::new();
        if c.shares.iter().any(|sh| !sh.president && !sh.owner.is_player() && !sh.owner.is_market()) {
            cands.push(("IPO", c.ipo_price.as_ref().map_or(0, |p| p.price)));
        }
        if c.shares.iter().any(|sh| sh.owner.is_market()) {
            cands.push(("market", c.share_price.as_ref().map_or(0, |p| p.price)));
        }
        for (kind, price) in cands {
            if offered.iter().any(|(oc, ok)| oc == &c.sym && ok == kind) {
                continue;
            }
            let mut codes: Vec<&str> = Vec::new();
            let mut brown = Value::Null;
            if !must_sell.is_null() {
                codes.push("mustsell");
            }
            if s.players_sold.get(&pid).is_some_and(|m| m.contains_key(&c.sym)) {
                codes.push("sold");
            }
            let in_brown = c.share_price.as_ref().is_some_and(|p| p.types.iter().any(|t| t == "multiple_buy"));
            if did_buy && !game.stock_can_buy_multiple(s, c, kind == "market") {
                if in_brown && last_all_one_corp(&c.sym) {
                    brown = if multi_only_market && kind == "IPO" {
                        json!("ipo")
                    } else if multi_only_market && s.bought_from_ipo {
                        json!("boughtIpo")
                    } else {
                        json!("nomore")
                    };
                    codes.push("brown");
                } else {
                    codes.push("onebuy");
                }
            }
            if cash < price {
                codes.push("cash");
            }
            if !holding_ok(c, pid) {
                codes.push("hold");
            }
            let counts = !c.share_price.as_ref().is_some_and(|p| p.types.iter().any(|t| t == "no_cert_limit" || t == "unlimited" || t == "multiple_buy"));
            if counts && certs >= limit {
                codes.push("certlimit");
            }
            if codes.is_empty() {
                codes.push("other");
            }
            blocks.push(json!({"corp": c.sym, "kind": kind, "codes": codes, "brown": brown, "price": price, "cash": cash,
                               "holding": c.percent_owned_by(&me), "maxPct": c.max_ownership_percent}));
        }
    }
    (turn, json!(blocks))
}

fn share_corp(a: &Value) -> Value {
    if let Some(c) = a.get("corporation") {
        return c.clone();
    }
    a.get("shares")
        .and_then(Value::as_array)
        .and_then(|v| v.first())
        .and_then(Value::as_str)
        .and_then(|s| s.rfind('_').map(|p| json!(&s[..p])))
        .unwrap_or(Value::Null)
}

/// Hexes the company may lay on now, and those whose terrain it can't pay for.
fn tile_hexes(g: &mut SiteGame) -> (Vec<String>, Vec<Value>) {
    let Some(corp) = g.game.round_state.active_entity_id.corp_sym().map(str::to_string) else { return (vec![], vec![]) };
    let cash = g.game.corp_idx.get(&corp).map_or(0, |&i| g.game.corporations[i].cash);
    let cands = g.game.lay_tile_candidate_hexes(&corp);
    let mut ok = Vec::new();
    let mut dear = Vec::new();
    let mut legal: Vec<String> = Vec::new();
    let lays: Vec<String> = g
        .game
        .get_factored_choices_impl()
        .into_iter()
        .filter(|a| a.action_type == "LayTile" && !a.entity.contains_key("private"))
        .filter_map(|a| a.params.get("hex").and_then(|h| h.as_str()).map(str::to_string))
        .collect();
    for (hex, _) in &cands {
        if blocked(g, hex) {
            continue;
        }
        let (cost, _) = terrain_of(g, hex);
        if lays.contains(hex) || (cost > cash && !g.game.upgradeable_tiles_for(hex).is_empty()) {
            legal.push(hex.clone());
        }
    }
    legal.sort_by_key(|h| ruby_hex_pos(h));
    for hex in legal {
        let (cost, terrain) = terrain_of(g, &hex);
        if cost > cash {
            dear.push(json!({"hex": hex, "cost": cost, "terrain": terrain}));
        } else {
            ok.push(hex);
        }
    }
    (ok, dear)
}

/// A hex a private blocks while a player owns it (Ruby `blocks_hexes`).
fn blocked(g: &SiteGame, hex: &str) -> bool {
    g.game.companies.iter().any(|c| {
        !c.closed && c.owner.is_player() && crate::abilities::blocks_hexes(&g.game.title, &c.sym).is_some_and(|(_, h)| h.contains(&hex))
    })
}

/// Ruby Corporation#holding_ok?(player, one more share): within the holding limit, or in a zone without one.
fn holding_ok(c: &crate::entities::Corporation, pid: u32) -> bool {
    let after = c.percent_owned_by(&EntityId::player(pid)) as u32 + c.share_unit_percent as u32;
    after <= c.max_ownership_percent as u32
        || c.share_price.as_ref().is_some_and(|p| p.types.iter().any(|t| t == "unlimited" || t == "multiple_buy"))
}

fn terrain_of(g: &SiteGame, hex: &str) -> (i32, Vec<String>) {
    let Some(h) = g.game.hex_ref(hex) else { return (0, vec![]) };
    let mut cost = 0;
    let mut terrain: Vec<String> = Vec::new();
    for u in &h.tile.upgrades {
        cost += u.cost;
        if !u.terrain.is_empty() && !terrain.contains(&u.terrain) {
            terrain.push(u.terrain.clone());
        }
    }
    (cost, terrain)
}

fn token_options(g: &mut SiteGame, ent: &Value) -> Value {
    let Some(corp) = g.game.round_state.active_entity_id.corp_sym().map(str::to_string) else { return json!([]) };
    let teleport = matches!(&g.game.round, Round::Operating(s) if s.teleport_pending);
    let pending = match &g.game.round {
        Round::Operating(s) => s.pending_tokens.first().cloned(),
        _ => None,
    };
    let ci = g.game.corp_idx[&corp];
    let price = if pending.is_some() || teleport { 0 } else { g.game.corporations[ci].next_token_price().unwrap_or(0) };
    let tele_hexes: Vec<String> = if teleport {
        let me = EntityId::corporation(&corp);
        g.game
            .companies
            .iter()
            .filter(|c| c.owner == me && !c.closed)
            .filter_map(|c| crate::abilities::teleport(&g.game.title, &c.sym))
            .flat_map(|(h, _)| h.iter().map(|x| x.to_string()))
            .collect()
    } else {
        vec![]
    };
    let mut cities: Vec<(String, usize)> = if teleport {
        tele_hexes
            .iter()
            .flat_map(|hex| {
                let n = g.game.hex_ref(hex).map_or(0, |h| h.tile.cities.len());
                (0..n).map(move |i| (hex.clone(), i))
            })
            .filter(|(hex, i)| g.game.hex_ref(hex).is_some_and(|h| h.tile.cities[*i].tokens.iter().any(|t| t.is_none())))
            .collect()
    } else if let Some((_, _, hex)) = &pending {
        let n = g.game.hex_ref(hex).map_or(0, |h| h.tile.cities.len());
        (0..n).filter(|&i| g.game.hex_ref(hex).is_some_and(|h| h.tile.cities[i].tokens.iter().any(|t| t.is_none()))).map(|i| (hex.clone(), i)).collect()
    } else {
        g.game.tokenable_cities_for(&corp)
    };
    let swapped = g.swapped.clone();
    cities.sort_by_key(|(h, c)| {
        let rc = if swapped.contains(&h.as_str()) { 1 - *c } else { *c };
        (ruby_hex_pos(h), rc)
    });
    let mut out = Vec::new();
    for (hex, city) in cities {
        let Some(h) = g.game.hex_ref(&hex) else { continue };
        let n = h.tile.cities.len();
        // Ruby City#get_slot: the first free slot not held for another company's home
        let reserved = g.game.home_reservations().iter().filter(|(hx, c, who)| hx == &hex && *c == city && who != &corp).count();
        let slot = h.tile.cities[city].tokens.iter().enumerate().position(|(i, t)| t.is_none() && i >= reserved)
            .or_else(|| h.tile.cities[city].tokens.iter().position(|t| t.is_none()))
            .unwrap_or(0);
        let rc = if g.swapped.contains(&hex.as_str()) && n == 2 { 1 - city } else { city };
        let action = with(json!({"type": "place_token"}), ent.clone());
        let action = with(action, json!({"city": city_id(g, &hex, city), "slot": slot, "tokener": corp}));
        out.push(json!({"hex": hex, "city": rc, "cities": n, "price": price, "action": action}));
    }
    json!(out)
}

fn dividend_options(g: &mut SiteGame, ent: &Value) -> Value {
    let Some(corp) = g.game.round_state.active_entity_id.corp_sym().map(str::to_string) else { return json!([]) };
    let revenue = match &g.game.round {
        Round::Operating(s) => s.revenue,
        _ => 0,
    };
    let ci = g.game.corp_idx[&corp];
    let c = &g.game.corporations[ci];
    let sp = c.share_price.clone();
    let mut out = Vec::new();
    for kind in ["payout", "withhold"] {
        let action = with(json!({"type": "dividend"}), ent.clone());
        let action = with(action, json!({"kind": kind}));
        // the preview: the move processed on a clone
        let mut clone = g.game.clone_for_search();
        let before: Vec<(u32, i32)> = clone.players.iter().map(|p| (p.id, p.cash)).collect();
        let corp_before = clone.corporations[ci].cash;
        let a = super::json::action_from_json(&action).ok();
        let ok = a.is_some_and(|a| clone.process_action_internal(&a).is_ok());
        let _ = (&before, corp_before);
        let preview = if ok {
            // Ruby dividend_options: per share (payout), each player's shares times it, the rest to the treasury
            let per = if kind == "payout" { revenue / 10 } else { 0 };
            let players: Vec<Value> = super::view::player_order(g)
                .into_iter()
                .filter_map(|pid| {
                    let n = c.shares.iter().filter(|s| s.owner == EntityId::player(pid)).map(|s| s.percent as i32 / 10).sum::<i32>();
                    (per * n > 0).then(|| json!({"id": pid, "amount": per * n}))
                })
                .collect();
            let to = clone.corporations[ci].share_price.as_ref().map(|s| s.price);
            json!({"kind": kind, "revenue": revenue, "perShare": per, "toTreasury": if kind == "payout" { 0 } else { revenue },
                   "players": players, "priceFrom": sp.as_ref().map(|s| s.price), "priceTo": to})
        } else {
            Value::Null
        };
        out.push(json!({"action": action, "kind": kind, "preview": preview}));
    }
    json!(out)
}

fn train_options(g: &mut SiteGame, pid: u32) -> (Value, Value) {
    let Some(corp) = g.game.round_state.active_entity_id.corp_sym().map(str::to_string) else { return (json!([]), Value::Null) };
    let ci = g.game.corp_idx[&corp];
    let cash = g.game.corporations[ci].cash;
    let pres = g.game.corporations[ci].president_id();
    let pres_cash = pres.and_then(|p| g.game.players.iter().find(|x| x.id == p)).map_or(0, |p| p.cash);
    let must = ruby_must_buy(g, &corp);
    let may = must;
    let power = cash;
    let list = g.game.buyable_trains_for(&corp);
    let mut depot = depot_trains(g);
    if let Some(minp) = depot.iter().map(|t| t.2).min() {
        if cash < minp {
            // Ruby EBUY_DEPOT_TRAIN_MUST_BE_CHEAPEST: the cheapest train only
            let first = depot.iter().find(|t| t.2 == minp).cloned();
            depot = first.into_iter().collect();
        }
    }
    let mut trains = Vec::new();
    let mut seen: Vec<(String, i32)> = Vec::new();
    let cheapest = depot.iter().min_by_key(|t| t.2).cloned();
    let last_sold = match &g.game.round {
        Round::Operating(s) => s.last_share_sold_price,
        _ => None,
    };
    for (id, name, price, discarded) in &depot {
        if seen.contains(&(name.clone(), *price)) {
            continue;
        }
        seen.push((name.clone(), *price));
        let short = (price - power).max(0);
        let cheapest = match &cheapest {
            Some(c) if short > 0 && may && c.1 != *name => json!({"name": c.1}),
            _ => Value::Null,
        };
        trains.push(json!({"train": id, "name": name, "price": price, "discarded": discarded, "short": short,
            "mayContribute": may, "presCash": pres_cash, "cheapest": cheapest,
            "action": {"type": "buy_train", "entity": corp, "entity_type": "corporation", "train": id, "price": price, "variant": name}}));
    }
    // trains the president's other companies hold
    let mut others: Vec<Value> = Vec::new();
    let mut groups: Vec<(String, String, Vec<String>)> = Vec::new();
    for (id, name, _, source) in &list {
        if source == "depot" || source == "discard" {
            continue;
        }
        let Some(&oi) = g.game.corp_idx.get(source.as_str()) else { continue };
        if g.game.corporations[oi].president_id() != pres || pres.is_none() {
            continue;
        }
        match groups.iter_mut().find(|x| &x.0 == source && &x.1 == name) {
            Some(gr) => gr.2.push(id.clone()),
            None => groups.push((source.clone(), name.clone(), vec![id.clone()])),
        }
    }
    groups.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    for (from, name, ids) in groups {
        let face = g.game.title_def().trains().iter().find(|t| t.name == name).map_or(0, |t| t.price);
        // Ruby spend_minmax (1830: EBUY_FROM_OTHERS :value); the treasury alone unless the president must help
        let (min, mut max) = if power < face {
            let possible = power + pres_cash;
            (last_sold.map_or(1, |l| possible - l + 1), face.min(possible))
        } else {
            (1, power)
        };
        if !may {
            max = max.min(power);
        }
        let min_depot = depot.iter().map(|t| t.2).min().unwrap_or(0);
        if cash < min_depot && last_sold.is_some() && face < min {
            continue;
        }
        others.push(json!({"train": ids[0], "name": name, "count": ids.len(), "from": from, "owner": pres,
            "ownerName": pres.map(|p| player_name(g, p)), "min": min, "max": max, "face": face, "fixed": null,
            "action": {"type": "buy_train", "entity": corp, "entity_type": "corporation", "train": ids[0]}}));
    }
    // trade-ins (1830: a 4, 5 or 6 for a D)
    let mut exchanges = Vec::new();
    let owned: Vec<(String, String)> = g.game.corporations[ci].trains.iter().map(|t| (t.id.clone(), t.name.clone())).collect();
    let mut seen_ex: Vec<String> = Vec::new();
    for dt in g.game.depot.trains.iter().chain(g.game.depot.discarded.iter()) {
        if dt.discount.is_empty() || !g.game.phase_available(dt.available_on.as_deref()) {
            continue;
        }
        for (oid, oname) in &owned {
            if let Some((_, d)) = dt.discount.iter().find(|(n, _)| n == oname) {
                let price = dt.price - d;
                let key = format!("{oname}|{}|{price}", dt.name);
                if seen_ex.contains(&key) {
                    continue;
                }
                seen_ex.push(key);
                exchanges.push(json!({"old": oname, "variant": dt.name, "price": price, "full": dt.price, "buyingPower": power,
                    "action": {"type": "buy_train", "entity": corp, "entity_type": "corporation", "train": dt.id, "price": price, "variant": dt.name, "exchange": oid}}));
            }
        }
        break;
    }
    let min_depot = depot.iter().map(|t| t.2).min();
    let needs_money = min_depot.is_some_and(|m| cash + pres_cash < m);
    let sell = if may && needs_money && pres == Some(pid) { sell_options(g, pid) } else { json!([]) };
    let buy = json!({"corp": corp, "cash": cash, "buyingPower": power, "mustBuy": must, "presidentMayContribute": may,
        "president": pres.map(|p| json!({"id": p, "name": player_name(g, p), "cash": pres_cash})), "others": others,
        "exchanges": exchanges, "sell": sell, "cheapest": min_depot});
    (json!(trains), buy)
}

fn discard_options(g: &SiteGame) -> Value {
    let Round::Operating(s) = &g.game.round else { return json!([]) };
    let Some(first) = s.crowded_corps.first() else { return json!([]) };
    let fi = g.game.corp_idx[first];
    let pres = g.game.corporations[fi].president_id();
    let mut out = Vec::new();
    for (k, sym) in s.crowded_corps.iter().enumerate() {
        let ci = g.game.corp_idx[sym];
        if k > 0 && (pres.is_none() || g.game.corporations[ci].president_id() != pres) {
            continue;
        }
        for t in &g.game.corporations[ci].trains {
            out.push(json!({"corp": sym, "train": t.name, "own": k == 0,
                "action": {"type": "discard_train", "entity": sym, "entity_type": "corporation", "train": t.id}}));
        }
    }
    json!(out)
}

/// Ruby BuyCompany#actions beside the active step: a private any player holds that the company can pay the
/// minimum price for, in a phase that allows it.
fn ruby_buy_company_open(g: &SiteGame, corp: &str) -> bool {
    let game = &g.game;
    let Some(&ci) = game.corp_idx.get(corp) else { return false };
    let cash = game.corporations[ci].cash;
    game.phase_has_status("can_buy_companies")
        && matches!(&game.round, Round::Operating(s) if s.crowded_corps.is_empty())
        && game.companies.iter().any(|c| {
            !c.closed && c.owner.is_player() && !c.no_buy && cash >= game.title_def().company_buy_price_range(c.value).0
        })
}

fn company_buy(g: &SiteGame, corp: &str, blocking: bool) -> Option<Value> {
    let game = &g.game;
    if !game.phase_has_status("can_buy_companies") {
        return None;
    }
    let ci = *game.corp_idx.get(corp)?;
    let c = &game.corporations[ci];
    let pres = c.president_id()?;
    let pe = EntityId::player(pres);
    let cash = c.cash;
    let mut offers = Vec::new();
    let mut dear = Vec::new();
    let mut comps: Vec<&crate::entities::Company> = game.companies.iter().filter(|m| !m.closed && m.owner == pe && !m.no_buy).collect();
    comps.sort_by_key(|m| m.value);
    for m in comps {
        let (min, max) = game.title_def().company_buy_price_range(m.value);
        let max = max.min(cash);
        if max < min {
            dear.push(json!({"sym": m.sym, "min": min}));
            continue;
        }
        offers.push(json!({"sym": m.sym, "name": company_name(g, &m.sym), "value": m.value, "revenue": m.revenue, "min": min, "max": max,
            "action": {"type": "buy_company", "entity": corp, "entity_type": "corporation", "company": m.sym}}));
    }
    let mut out = json!({"corp": corp, "cash": cash, "offers": offers, "blocking": blocking});
    if !dear.is_empty() {
        out["tooDear"] = json!(dear);
    }
    Some(out)
}

fn ability_uses(g: &mut SiteGame, corp: &str) -> Vec<Value> {
    let mut out = Vec::new();
    let me = EntityId::corporation(corp);
    let at_track = matches!(&g.game.round, Round::Operating(s) if s.step == S::LayTile && !s.teleport_pending && s.crowded_corps.is_empty());
    let syms: Vec<String> = g
        .game
        .companies
        .iter()
        .filter(|co| !co.closed && !co.ability_used && co.owner == me)
        .map(|co| co.sym.clone())
        .collect();
    for sym in syms {
        let (hexes, tp): (Vec<String>, bool) = if let Some((h, _, _, _)) = crate::abilities::tile_lay(&g.game.title, &sym) {
            (h.iter().map(|x| x.to_string()).collect(), false)
        } else if let Some((h, _)) = crate::abilities::teleport(&g.game.title, &sym) {
            if !at_track {
                continue;
            }
            (h.iter().map(|x| x.to_string()).collect(), true)
        } else {
            continue;
        };
        let hexes: Vec<String> = hexes.into_iter().filter(|h| !g.game.upgradeable_tiles_for(h).is_empty()).collect();
        if hexes.is_empty() {
            continue;
        }
        let has_token = g.game.corp_idx.get(corp).is_some_and(|&i| g.game.corporations[i].next_token_index().is_some());
        out.push(json!({"company": sym, "name": company_name(g, &sym), "owner": corp, "hexes": hexes, "teleport": tp && has_token, "extra": !tp}));
    }
    out
}

/// Bankruptcy as the site offers it: voluntary bankruptcy is kept for the site's players (X3, Caleb's decision,
/// 2026-11-26; docs/SITE_ENGINE_RS.md §11.2). The site's hand (living-board `manual.ts` `bankruptRaw`, on Opal) offers
/// it in Buy Trains to the operating company when Ruby's `round.actions_for` has `bankrupt` (Step::Bankrupt: the
/// current entity, at every operating step), `Game#can_go_bankrupt?` holds (total_emr_buying_power below
/// `depot.min_depot_price`) and the active step's `must_buy_train?` isn't `false` -- Opal's `nil` is true in
/// JavaScript, so a company with no train facing a depot with trains is offered it whether or not it has a route to
/// run (Game#must_buy_train?'s `route_info&.dig(:route_train_purchase)` is `nil` without one). Ruby's
/// Step::Bankrupt#process_bankrupt accepts it there (it checks `can_go_bankrupt?` only), and so does engine-rs's
/// process_action (its Buy Trains guard is the same `can_go_bankrupt_emr`). The bots keep the printed rule: the
/// shared legal-move enumeration (`legal_action_types`, the factored enumeration) offers `bankrupt` only to a
/// company that must buy a train; nothing here changes it.
fn bankrupt_offer(g: &mut SiteGame, corp: &str) -> Option<Value> {
    if !matches!(&g.game.round, Round::Operating(s) if s.step == S::BuyTrain && s.crowded_corps.is_empty()) {
        return None;
    }
    let ci = *g.game.corp_idx.get(corp)?;
    let pid = g.game.corporations[ci].president_id()?;
    // Game#must_buy_train?'s first two terms (`entity.trains.empty? && !depot.depot_trains.empty?`): the only ones that
    // can answer `false`
    if !g.game.corporations[ci].trains.is_empty() || depot_trains(g).is_empty() {
        return None;
    }
    if !g.game.can_go_bankrupt_emr(pid, corp) {
        return None;
    }
    let min = g.game.min_depot_price_for_emr();
    let power = emr_power(g, pid, corp);
    Some(json!({"action": {"type": "bankrupt", "entity": corp, "entity_type": "corporation"}, "player": player_name(g, pid),
        "corp": corp, "min": min, "power": power, "minText": format!("${min}"), "powerText": format!("${power}"),
        "ends": " The game ends immediately when any player is bankrupt."}))
}

/// Ruby total_emr_buying_power: the president's cash and what they may sell, and the company's cash (as
/// `can_go_bankrupt_emr` counts them).
fn emr_power(g: &SiteGame, pid: u32, corp: &str) -> i32 {
    let game = &g.game;
    let needed = game.min_depot_price_for_emr();
    let cash = game.players.iter().find(|p| p.id == pid).map_or(0, |p| p.cash);
    let ccash = game.corp_idx.get(corp).map_or(0, |&i| game.corporations[i].cash);
    let extra = needed - (cash + ccash);
    let mut per: std::collections::HashMap<String, i32> = std::collections::HashMap::new();
    for b in game.sellable_bundles_detailed(pid) {
        if b.corp_sym == corp && game.causes_president_swap(&b.corp_sym, pid, b.percent) {
            continue;
        }
        if b.bundle_price - b.min_share_price >= extra {
            continue;
        }
        let e = per.entry(b.corp_sym.clone()).or_insert(0);
        *e = (*e).max(b.bundle_price);
    }
    cash + per.values().sum::<i32>() + ccash
}

fn exchange_offers(g: &SiteGame, pid: u32) -> Vec<Value> {
    if matches!(g.game.round, Round::Auction(_)) {
        return vec![];
    }
    let me = EntityId::player(pid);
    let mut out = Vec::new();
    for m in &g.game.companies {
        if m.closed || m.owner != me {
            continue;
        }
        let Some((corps, _)) = crate::abilities::exchange(&g.game.title, &m.sym) else { continue };
        if !g.game.company_exchange_available(&m.sym) {
            continue;
        }
        for corp in corps {
            let Some(&ci) = g.game.corp_idx.get(*corp) else { continue };
            let c = &g.game.corporations[ci];
            let ipo = c.shares.iter().find(|s| !s.president && !s.owner.is_player() && !s.owner.is_market());
            let market = c.shares.iter().find(|s| s.owner.is_market());
            for (from, sh) in [("IPO", ipo), ("market", market)] {
                if let Some(sh) = sh {
                    out.push(json!({"company": m.sym, "companyName": company_name(g, &m.sym), "corp": corp, "from": from, "percent": sh.percent,
                        "action": {"type": "buy_shares", "entity": m.sym, "entity_type": "company", "shares": [format!("{}_{}", corp, sh.index)], "percent": sh.percent}}));
                }
            }
        }
    }
    out
}

/// Every tile the engine allows on a hex, in every legal rotation: [{tile, name, color, rotation, cost, terrain, action}].
pub fn tile_options(g: &mut SiteGame, hex: &str, via: Option<&str>) -> Result<Value, String> {
    let entity = match via {
        Some(c) => json!({"entity": c, "entity_type": "company"}),
        None => match g.game.round_state.active_entity_id.corp_sym() {
            Some(c) => json!({"entity": c, "entity_type": "corporation"}),
            None => return Err("refused: no company is laying track".into()),
        },
    };
    // a private's lay pays the terrain too (1830's D&H on F16), unless its ability makes it free (none in 1830)
    let (cost, terrain) = terrain_of(g, hex);
    // the engine's own lays on this hex (its legal tiles in their legal rotations, connected for the company)
    let lays: Vec<(String, u32)> = g
        .game
        .get_factored_choices_impl()
        .into_iter()
        .filter(|a| a.action_type == "LayTile")
        .filter(|a| a.entity.get("private").and_then(Value::as_str) == via)
        .filter(|a| a.params.get("hex").and_then(Value::as_str) == Some(hex))
        .filter_map(|a| {
            let t = a.params.get("tile").and_then(Value::as_str)?.to_string();
            let r = a.params.get("rotation").and_then(Value::as_u64)? as u32;
            Some((t, r))
        })
        .collect();
    let mut names: Vec<String> = Vec::new();
    for (name, _) in &lays {
        if !names.contains(name) {
            names.push(name.clone());
        }
    }
    // Ruby's tile list order (1830's TILES are numbered in order)
    names.sort_by_key(|n| g.supply.rank(n));
    let mut out = Vec::new();
    for name in names {
        let Some(id) = g.next_tile(&name) else { continue };
        let color = g.game.tile_catalog.get(&name).map(|t| format!("{:?}", t.color).to_lowercase()).unwrap_or_default();
        let mut rots: Vec<u32> = lays.iter().filter(|(n, _)| *n == name).map(|(_, r)| *r).collect();
        rots.sort();
        rots.dedup();
        for r in rots {
            let action = with(json!({"type": "lay_tile"}), entity.clone());
            let action = with(action, json!({"hex": hex, "tile": id, "rotation": r}));
            out.push(json!({"tile": id, "name": name, "color": color, "rotation": r, "cost": cost, "terrain": terrain, "action": action}));
        }
    }
    Ok(json!(out))
}

/// The auto-router's best run for the acting company: {action (run_routes as the site posts it), revenue, routes
/// (each with its track)}, or null.
pub fn auto_route(g: &mut SiteGame) -> Value {
    let Some(corp) = g.game.round_state.active_entity_id.corp_sym().map(str::to_string) else { return Value::Null };
    if !matches!(&g.game.round, Round::Operating(s) if s.step == S::RunRoutes) {
        return Value::Null;
    }
    let (routes, train_ids, revenue) = g.game.clone_for_search().optimal_routes(&corp);
    if routes.is_empty() {
        return Value::Null;
    }
    let mut site_routes = Vec::new();
    let mut ordered: Vec<&crate::router::RouteCandidate> = routes.iter().collect();
    ordered.sort_by_key(|r| r.train_index);
    for r in ordered {
        let train = train_ids.get(r.train_index).cloned().unwrap_or_default();
        let swapped = g.swapped.clone();
        let nodes: Vec<String> = r
            .nodes
            .iter()
            .map(|n| {
                let idx = if swapped.contains(&n.hex_id.as_str()) && n.node_type == crate::map::NodeType::City && g.game.hex_ref(&n.hex_id).is_some_and(|h| h.tile.cities.len() == 2) { 1 - n.index } else { n.index };
                format!("{}-{}", n.hex_id, idx)
            })
            .collect();
        let hexes: Vec<String> = r.nodes.iter().map(|n| n.hex_id.clone()).collect();
        let route = json!({"train": train, "connections": r.connections, "hexes": hexes, "revenue": r.revenue,
                           "revenue_str": hexes.join("-"), "nodes": nodes});
        site_routes.push(route);
    }
    let action = json!({"type": "run_routes", "entity": corp, "entity_type": "corporation", "routes": site_routes});
    let with_tracks: Vec<Value> = site_routes
        .iter()
        .map(|r| {
            let mut r = r.clone();
            if let Some(t) = super::track::route_track(g, &r) {
                r["track"] = t;
            }
            r
        })
        .collect();
    json!({"action": action, "revenue": revenue, "routes": with_tracks})
}

/// Every legal action (the RL enumeration, own-only) as site JSON: the menu's actions flattened.
pub fn legal_actions(g: &mut SiteGame) -> Value {
    let pids = super::view::active_players(g);
    let mut out = Vec::new();
    for pid in pids {
        let m = menu(g, pid);
        collect_actions(&m, &mut out);
    }
    json!(out)
}

fn collect_actions(v: &Value, out: &mut Vec<Value>) {
    match v {
        Value::Object(m) => {
            if let Some(a) = m.get("action") {
                if a.get("type").is_some() && !out.contains(a) {
                    out.push(a.clone());
                }
            }
            for (k, x) in m {
                if k != "action" {
                    collect_actions(x, out);
                }
            }
        }
        Value::Array(xs) => xs.iter().for_each(|x| collect_actions(x, out)),
        _ => {}
    }
}
