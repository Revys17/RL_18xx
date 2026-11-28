//! What the board renders for an 1822-family game (docs/SITE_ENGINE_RS_1822.md §4.2): the Snapshot (living-board
//! `types.ts`, the shared fields with their step-1 meanings, players by their site ids) plus `g1822`, the family's own
//! facts (bid boxes, stacks, bidding tokens, debt, the tax haven, credits, exchange tokens, destinations,
//! reservations, 1822CA's sawmill), and the engine part of StaticInfo.

use super::super::game::SiteGame;
use super::super::json::tile_name;
use crate::entities::EntityId;
use crate::g1822::{Ent, RoundKind, Step};
use serde_json::{json, Map, Value};

fn holder(g: &SiteGame, e: &EntityId) -> Option<String> {
    if let Some(p) = e.player_id() {
        Some(format!("p:{}", g.site_id(p)))
    } else if e.is_corporation() {
        e.corp_sym().map(|s| format!("c:{s}"))
    } else {
        None
    }
}

fn currency(g: &SiteGame) -> &'static str {
    if super::is_ca(&g.game) { "$" } else { "£" }
}

/// The round's name, short name, label, turn and number (adapter.ts `roundInfo`).
pub fn round_info(g: &SiteGame) -> Value {
    let game = &g.game;
    let turn = game.turn;
    let r = game.rnd();
    match r.kind {
        RoundKind::Stock => json!({"name": "Stock Round", "short": "SR", "label": format!("Stock Round {turn}"), "turn": turn, "num": 1}),
        RoundKind::Choices => json!({"name": "Choose", "short": "Choices", "label": "Choose", "turn": turn, "num": r.round_num}),
        RoundKind::Operating => {
            let total = game.g1822_operating_rounds_in_set();
            json!({"name": "Operating Round", "short": "OR", "label": format!("Operating Round {turn}.{} of {total}", r.round_num),
                   "turn": turn, "num": r.round_num})
        }
        RoundKind::Merger => json!({"name": "Merger Round", "short": "MR", "label": "Merger Round", "turn": turn, "num": r.round_num}),
    }
}

/// Ruby's active step's description (`round.active_step.description`).
pub fn step_label(g: &SiteGame) -> Option<String> {
    let game = &g.game;
    let (bi, step, ent) = super::active(game)?;
    Some(match step {
        Step::DiscardTrain => "Discard Trains".into(),
        Step::BuySellParShares => "Bid, convert concession or buy/sell shares".into(),
        Step::Choose | Step::SpecialChoose => "Choose".into(),
        Step::PendingToken => {
            // HomeToken#description
            let pend = game.rnd().pending_tokens.first().cloned();
            let cur = game.rnd().entities.get(game.rnd().entity_index).cloned();
            match pend {
                Some(p) if cur.as_deref() != Some(p.corp.as_str()) && game.rnd().kind == RoundKind::Operating => {
                    format!("Place {} Home Token", corp_name(g, &p.corp))
                }
                Some(p) if p.token_idx == 0 => "Place Home Token".into(),
                _ => "Place Token".into(),
            }
        }
        Step::FirstTurnHousekeeping | Step::BuyTrain => "Buy Trains".into(),
        Step::AcquireCompany => "Acquire private companies".into(),
        Step::SpecialTrack => match &ent {
            Ent::Company(c) => format!("Lay Track for {}", super::super::menu::company_name(g, c)),
            _ => "Lay Track".into(),
        },
        Step::SpecialToken => "Place teleport token".into(),
        Step::Track => {
            if game.rnd().num_laid_track > 0 || game.rnd().upgraded_track {
                "Lay Track".into()
            } else {
                "Lay/Upgrade Track".into()
            }
        }
        Step::DestinationToken => "Place the Destination Token".into(),
        Step::Token => "Place a Token".into(),
        Step::Route => "Run Routes".into(),
        Step::Dividend => "Pay or Withhold Dividends".into(),
        Step::MinorAcquisition => "Acquire a minor".into(),
        Step::IssueShares => "Issue or Redeem Shares".into(),
        Step::AssignSawmill => "sawmill".into(),
        Step::AcquisitionTrack => "Lay/Upgrade Track (Acquisition Bonus)".into(),
        _ => {
            let _ = bi;
            super::step_class(step)
        }
    })
}

/// Ruby's `pass_description` of the active step.
pub fn pass_description(g: &SiteGame) -> String {
    let game = &g.game;
    let Some((bi, step, _)) = super::active(game) else { return "Pass".into() };
    let acted = game.rnd().acted.get(bi).copied().unwrap_or(false);
    match step {
        Step::BuySellParShares => if game.rnd().bid_actions > 0 { "Pass (Bids)".into() } else { "Pass".into() },
        Step::Track | Step::AcquisitionTrack => if acted { "Done (Track)".into() } else { "Skip (Track)".into() },
        Step::Token => "Skip (Token)".into(),
        Step::SpecialToken => "Pass (Token)".into(),
        Step::BuyTrain | Step::FirstTurnHousekeeping => if acted { "Done (Trains)".into() } else { "Skip (Trains)".into() },
        Step::DestinationToken => "Skip (Destination Token)".into(),
        Step::MinorAcquisition => "Skip (Minor acquisition)".into(),
        Step::IssueShares => "Skip (Issue/Redeem)".into(),
        Step::AcquireCompany => if acted { "Done (Acquire companies)".into() } else { "Skip (Acquire companies)".into() },
        _ => "Pass".into(),
    }
}

pub fn corp_name(g: &SiteGame, sym: &str) -> String {
    g.game.corp_idx.get(sym).map(|&ci| g.game.corporations[ci].name.clone()).unwrap_or_else(|| sym.to_string())
}

/// The entity acting now as the snapshot names it ("p:13", "c:LNWR", "m:P2").
pub fn current_holder(g: &SiteGame) -> Option<String> {
    let (_, _, ent) = super::active(&g.game)?;
    Some(match ent {
        Ent::Player(p) => format!("p:{}", g.site_id(p)),
        Ent::Corp(c) => format!("c:{c}"),
        Ent::Company(c) => format!("m:{c}"),
    })
}

/// The bidding tokens a player has left (G1822's `bidding_token_per_player` less their bids standing).
pub fn tokens_left(g: &SiteGame, seat: u32) -> i32 {
    let game = &g.game;
    let total = (game.fam().bidding_tokens)(game.players.len());
    let mine = game.rnd().bids.values().flatten().filter(|b| b.player == seat).count() as i32;
    total - mine
}

/// The companies an owner holds, in the order it got them (Ruby's `companies` lists); a company owned only by name
/// (1822MX's) left out.
pub fn owned_companies(g: &SiteGame, owner: &EntityId) -> Vec<String> {
    let st = g.game.g();
    let mut list: Vec<(u64, usize, String)> = g
        .game
        .companies
        .iter()
        .enumerate()
        .filter(|(_, c)| &c.owner == owner && !c.closed && !st.owner_only.contains(&c.sym) && !st.bank_owned.contains(&c.sym))
        .map(|(i, c)| (g.company_seq.get(&c.sym).copied().unwrap_or(0), i, c.sym.clone()))
        .collect();
    list.sort();
    list.into_iter().map(|x| x.2).collect()
}

/// A corporation's latest operating result.
pub fn last_run(g: &SiteGame, sym: &str) -> Value {
    super::super::view::last_run(g, sym)
}

/// The board's hexes: tile key, tokens per city slot, reservations per city.
pub fn hex_states(g: &SiteGame) -> Value {
    let game = &g.game;
    let st = game.g();
    let mut out = Map::new();
    for h in &game.hexes {
        let mut m = Map::new();
        let name = match tile_name(&h.tile.name) {
            Some(n) => n.to_string(),
            None => h.id.clone(),
        };
        m.insert("k".into(), json!(format!("{name}@{}", h.tile.rotation)));
        if !h.tile.cities.is_empty() {
            let mut any = false;
            let tok: Vec<Value> = h
                .tile
                .cities
                .iter()
                .map(|c| {
                    json!(c.tokens.iter().map(|t| match t {
                        Some(t) => {
                            any = true;
                            json!(t.corporation_id)
                        }
                        None => Value::Null,
                    }).collect::<Vec<_>>())
                })
                .collect();
            if any {
                m.insert("tok".into(), Value::Array(tok));
            }
            let mut any_res = false;
            let res: Vec<Value> = (0..h.tile.cities.len())
                .map(|k| {
                    let r: Vec<String> = st.city_reservations.get(&format!("{}#{k}", h.id)).map(|v| v.iter().flatten().cloned().collect()).unwrap_or_default();
                    if !r.is_empty() {
                        any_res = true;
                    }
                    json!(r)
                })
                .collect();
            if any_res {
                m.insert("res".into(), Value::Array(res));
            }
        }
        if let Some(r) = st.tile_reservations.get(&h.id).filter(|v| !v.is_empty()) {
            m.insert("tileRes".into(), json!(r));
        }
        out.insert(h.id.clone(), Value::Object(m));
    }
    Value::Object(out)
}

/// The depot: the upcoming trains (Ruby holds one 7 at a time: it is unlimited), the discarded ones, and what may be
/// bought now (the next train, one of each later type already available, one of each discarded type).
fn depot(g: &SiteGame) -> Value {
    let game = &g.game;
    let card = |t: &crate::entities::Train| json!({"id": t.id, "name": t.name});
    let mut upcoming = Vec::new();
    for t in &game.depot.trains {
        upcoming.push(card(t));
        if t.name == "7" {
            break;
        }
    }
    let mut available: Vec<String> = Vec::new();
    let mut names: Vec<String> = Vec::new();
    if let Some(t) = game.depot.trains.first() {
        names.push(t.name.clone());
        available.push(t.id.clone());
    }
    for t in &game.depot.trains {
        if !names.contains(&t.name) && game.phase_available(t.available_on.as_deref()) {
            names.push(t.name.clone());
            available.push(t.id.clone());
        }
    }
    let mut dn: Vec<String> = Vec::new();
    for t in &game.depot.discarded {
        if !dn.contains(&t.name) {
            dn.push(t.name.clone());
            available.push(t.id.clone());
        }
    }
    json!({"upcoming": upcoming, "discarded": game.depot.discarded.iter().map(card).collect::<Vec<_>>(), "available": available})
}

/// The family's own facts (`Snapshot.g1822`).
pub fn family(g: &SiteGame) -> Value {
    let game = &g.game;
    let st = game.g();
    let mut bids = Vec::new();
    for (co, list) in &st.round.bids {
        for b in list {
            bids.push(json!({"company": co, "player": g.site_id(b.player), "price": b.price}));
        }
    }
    let mut tokens = Map::new();
    let mut committed = Map::new();
    let mut debt = Map::new();
    for p in &game.players {
        let id = g.site_id(p.id).to_string();
        tokens.insert(id.clone(), json!(tokens_left(g, p.id)));
        committed.insert(id.clone(), json!(game.g1822_committed_cash(p.id)));
        let d = game.player_debt(p.id);
        if d != 0 {
            debt.insert(id, json!(d));
        }
    }
    let th = EntityId(crate::g1822::TAX_HAVEN_OWNER.to_string());
    let mut th_shares = Map::new();
    for c in super::members(game).into_iter().map(|i| &game.corporations[i]) {
        let pct = c.percent_owned_by(&th);
        if pct > 0 {
            th_shares.insert(c.sym.clone(), json!(pct));
        }
    }
    let osth = game.fam().company_osth;
    let credits: Map<String, Value> = st.phase_revenue.iter().filter_map(|(c, v)| v.map(|x| (c.clone(), json!(x)))).collect();
    let ex: Map<String, Value> = st.exchange_tokens.iter().filter(|(_, n)| **n > 0).map(|(c, n)| (c.clone(), json!(n))).collect();
    let mut dest = Map::new();
    for c in super::members(game).into_iter().map(|i| &game.corporations[i]) {
        if c.corp_type == crate::title::CorpType::Major {
            if let Some(d) = game.g1822_destination(&c.sym) {
                dest.insert(c.sym.clone(), json!({"hex": d, "placed": !st.destination_ability.contains(&c.sym) || game.g1822_destination_node(&c.sym).is_some()}));
            }
        }
    }
    let mut out = json!({
        "bidBoxes": {"minors": game.bidbox_minors(), "concessions": game.bidbox_concessions(), "privates": game.bidbox_privates()},
        "bids": bids,
        "stacks": st.company_order,
        "bankOwned": st.bank_owned,
        "biddingTokens": tokens,
        "biddingTokensTotal": (game.fam().bidding_tokens)(game.players.len()),
        "bidsPerTurn": game.fam().bidding_tokens_per_action,
        "bidStep": game.fam().min_bid_increment,
        "committed": committed,
        "debt": debt,
        "taxHaven": {"company": osth, "owner": holder(g, &game.co_owner(osth)), "cash": st.tax_haven_cash, "shares": th_shares},
        "credits": credits,
        "exchangeTokens": ex,
        "destinations": dest,
        "plus": game.g1822_plus(),
        "currency": currency(g),
    });
    if super::is_ca(game) {
        out["sawmill"] = json!({"hex": st.ca.sawmill_hex, "owner": st.ca.sawmill_owner, "bonus": st.ca.sawmill_bonus});
    }
    out
}

pub fn snapshot(g: &SiteGame, action_id: i64) -> Value {
    let game = &g.game;
    let finished = game.finished;
    let members = super::members(game);
    let mut players: Vec<Map<String, Value>> = Vec::new();
    for p in &game.players {
        let pe = EntityId::player(p.id);
        let mut certs = Map::new();
        let mut count = Map::new();
        for c in members.iter().map(|&i| &game.corporations[i]) {
            let mut pct = 0u32;
            let mut n = 0u32;
            for s in c.shares.iter().filter(|s| s.owner == pe) {
                pct += s.percent as u32;
                n += 1;
            }
            if n > 0 {
                certs.insert(c.sym.clone(), json!(pct));
                count.insert(c.sym.clone(), json!(n));
            }
        }
        let mut m = Map::new();
        m.insert("numCerts".into(), json!(game.g1822_num_certs(p.id)));
        m.insert("id".into(), json!(g.site_id(p.id)));
        m.insert("cash".into(), json!(p.cash));
        m.insert("value".into(), json!(game.g1822_player_value(p.id)));
        m.insert("certs".into(), Value::Object(certs));
        m.insert("certCount".into(), Value::Object(count));
        m.insert("president".into(), json!([]));
        m.insert("companies".into(), json!(owned_companies(g, &pe)));
        m.insert("bankrupt".into(), json!(p.bankrupt));
        m.insert("debt".into(), json!(game.player_debt(p.id)));
        players.push(m);
    }
    let r = game.rnd();
    let is_or = r.kind == RoundKind::Operating;
    let cur_corp = if is_or { r.entities.get(r.entity_index).cloned() } else { None };
    let cur_pos = cur_corp.as_ref().and_then(|c| r.entities.iter().position(|x| x == c));
    let mut corps = Vec::new();
    for c in members.iter().map(|&i| &game.corporations[i]) {
        let president = super::corp_owner(game, &c.sym).map(|p| g.site_id(p));
        if let Some(pid) = president {
            if let Some(pm) = players.iter_mut().find(|m| m["id"] == json!(pid)) {
                pm.get_mut("president").unwrap().as_array_mut().unwrap().push(json!(c.sym));
            }
        }
        let sp = c.share_price.as_ref();
        let stack = sp
            .and_then(|sp| game.market_cell_corps.get(&(sp.row, sp.column)))
            .and_then(|list| list.iter().position(|s| s == &c.sym))
            .unwrap_or(0);
        let unused: Vec<&crate::entities::Token> = c.tokens.iter().filter(|t| !t.used).collect();
        let ipo_pct: u32 = c.shares.iter().filter(|s| s.owner == EntityId::ipo(&c.sym) || s.owner == EntityId::corporation(&c.sym)).map(|s| s.percent as u32).sum();
        let pool_pct: u32 = c.shares.iter().filter(|s| s.owner.is_market()).map(|s| s.percent as u32).sum();
        let operated = is_or && cur_pos.is_some_and(|ci| matches!(r.entities.iter().position(|s| s == &c.sym), Some(i) if i < ci));
        let mut cm = Map::new();
        cm.insert("sym".into(), json!(c.sym));
        cm.insert("type".into(), json!(if c.corp_type == crate::title::CorpType::Minor { "minor" } else { "major" }));
        cm.insert("cash".into(), json!(c.cash));
        cm.insert("price".into(), json!(sp.map(|s| s.price)));
        cm.insert("cell".into(), sp.map_or(Value::Null, |s| json!([s.row, s.column])));
        cm.insert("stack".into(), json!(stack));
        cm.insert("par".into(), json!(c.ipo_price.as_ref().map(|s| s.price)));
        cm.insert("ipoed".into(), json!(c.ipo_price.is_some()));
        cm.insert("floated".into(), json!(c.floated));
        cm.insert("toFloat".into(), json!(if c.floated { 0 } else { (c.float_percent as i64 - (100 - ipo_pct as i64)).max(0) }));
        cm.insert("closed".into(), json!(c.closed));
        cm.insert("ipoPct".into(), json!(ipo_pct));
        cm.insert("poolPct".into(), json!(pool_pct));
        cm.insert("president".into(), json!(president));
        cm.insert("trains".into(), json!(c.trains.iter().map(|t| json!({"id": t.id, "name": t.name})).collect::<Vec<_>>()));
        cm.insert("tokensLeft".into(), json!(unused.len()));
        cm.insert("nextTokenPrice".into(), json!(unused.first().map(|t| t.price)));
        cm.insert("companies".into(), json!(owned_companies(g, &EntityId::corporation(&c.sym))));
        cm.insert("operated".into(), json!(operated));
        cm.insert("lastRun".into(), last_run(g, &c.sym));
        corps.push(Value::Object(cm));
    }
    let st = game.g();
    let companies: Vec<Value> = st
        .company_order
        .iter()
        .filter_map(|sym| game.company_idx.get(sym.as_str()).map(|&i| &game.companies[i]))
        .map(|m| {
            let closed = m.closed && !st.deleted_open.contains(&m.sym);
            let owner = if closed || st.bank_owned.contains(&m.sym) { Value::Null } else { holder(g, &m.owner).map_or(Value::Null, Value::from) };
            json!({"sym": m.sym, "owner": owner, "closed": closed, "bank": st.bank_owned.contains(&m.sym)})
        })
        .collect();
    let bids: Vec<Value> = family(g)["bids"].as_array().cloned().unwrap_or_default();
    let (result, end_reason) = if finished {
        let mut rm = Map::new();
        for (k, v) in g.result() {
            rm.insert(k.to_string(), json!(v));
        }
        (Value::Object(rm), json!(g.end_reason()))
    } else {
        (Value::Null, Value::Null)
    };
    let (order, order_index) = match r.kind {
        RoundKind::Operating | RoundKind::Merger => (r.entities.iter().map(|c| format!("c:{c}")).collect::<Vec<_>>(), r.entity_index as i64),
        _ => (
            r.entities.iter().filter_map(|p| p.parse::<u32>().ok()).map(|p| format!("p:{}", g.site_id(p))).collect::<Vec<_>>(),
            r.entity_index as i64,
        ),
    };
    let op_order: Vec<String> = if is_or { r.entities.clone() } else { Vec::new() };
    let mut out = Map::new();
    if let Some(p) = super::super::view::programs(g) {
        out.insert("programs".into(), p);
    }
    out.insert("actionId".into(), json!(action_id));
    out.insert("round".into(), round_info(g));
    out.insert("stepLabel".into(), if finished { Value::Null } else { json!(step_label(g)) });
    out.insert("phase".into(), json!(game.phase.name));
    out.insert("trainLimit".into(), json!(game.phase.train_limit));
    out.insert("bank".into(), json!(game.bank.cash));
    out.insert("activePlayers".into(), json!(if finished { Vec::new() } else { super::active_players(g) }));
    out.insert("current".into(), if finished { Value::Null } else { json!(current_holder(g)) });
    out.insert("priority".into(), json!(super::key::priority(g)));
    out.insert("players".into(), Value::Array(players.into_iter().map(Value::Object).collect()));
    out.insert("corps".into(), Value::Array(corps));
    out.insert("companies".into(), Value::Array(companies));
    out.insert("hexes".into(), hex_states(g));
    out.insert("depot".into(), depot(g));
    out.insert("bids".into(), Value::Array(bids));
    out.insert("auction".into(), json!([]));
    out.insert("finished".into(), json!(finished));
    out.insert("result".into(), result);
    out.insert("endReason".into(), end_reason);
    out.insert("ending".into(), ending(g));
    out.insert("order".into(), json!(order));
    out.insert("orderIndex".into(), json!(order_index));
    out.insert("opOrder".into(), json!(op_order));
    out.insert("waiting".into(), json!([]));
    out.insert("g1822".into(), family(g));
    Value::Object(out)
}

/// Whether the game will end, and when (Ruby `game_ending_description`): the latched trigger.
pub fn ending(g: &SiteGame) -> Value {
    let game = &g.game;
    if game.finished {
        return Value::Null;
    }
    let Some((reason, timing)) = game.g().game_end_trigger.clone() else { return Value::Null };
    let text_reason = match reason.as_str() {
        "bank" => "Bank Broken",
        "stock_market" => "Company hit max stock value",
        "final_phase" => "Final phase was reached",
        _ => "Game end triggered",
    };
    let turn = game.turn;
    let r = game.rnd();
    let (after, when) = match timing.as_str() {
        "current_or" if r.kind == RoundKind::Operating => (format!("this OR ({turn}.{})", r.round_num), format!("after OR {turn}.{}", r.round_num)),
        "current_or" => ("the next OR".to_string(), "after the next OR".to_string()),
        _ => {
            let n = game.g1822_operating_rounds_in_set();
            (format!("OR {turn}.{n}"), format!("after OR {turn}.{n}"))
        }
    };
    let text = format!("{text_reason} : Game Ends at conclusion of {after}");
    json!({"when": when, "why": text_reason.to_lowercase(), "text": text})
}

/// The step's action as the board shows it (adapter.ts ACTION_KEYS plus the family's).
pub fn action_lite(g: &SiteGame, raw: &Value) -> Value {
    let mut out = super::super::view::action_lite(g, raw);
    if let Some(o) = out.as_object_mut() {
        for k in ["choice", "amount", "target", "target_type", "combo_entities", "token_type", "tokener", "slot", "exchange"] {
            if let Some(v) = raw.get(k).filter(|v| !v.is_null()) {
                o.insert(k.to_string(), v.clone());
            }
        }
    }
    out
}

/// The engine part of StaticInfo for the family.
pub fn static_info(g: &SiteGame) -> Value {
    let game = &g.game;
    let title = game.title_def();
    let fam = game.fam();
    let players: Vec<Value> = g.players.iter().enumerate().map(|(i, (id, name, bot))| json!({"id": id, "name": name, "bot": bot, "seat": i})).collect();
    let corporations: Vec<Value> = title
        .corporations()
        .iter()
        .filter(|c| game.corp_idx.get(c.sym).is_some_and(|&ci| !game.corporations[ci].closed || game.corporations[ci].ipo_price.is_some()))
        .map(|c| {
            let ci = game.corp_idx[c.sym];
            let co = &game.corporations[ci];
            // (a major's destination token is one of its tokens: Ruby's setup_destinations)
            let tokens = c.token_prices.len() + (fam.destination)(c.sym).is_some() as usize;
            json!({"sym": c.sym, "name": c.name, "home": c.home_hex, "tokenCount": tokens,
                   "floatPercent": c.float_percent, "capitalization": "incremental", "sharePercent": co.share_unit_percent,
                   "type": if co.corp_type == crate::title::CorpType::Minor { "minor" } else { "major" },
                   "destination": (fam.destination)(c.sym), "exchangeTokens": (fam.exchange_tokens)(c.sym)})
        })
        .collect();
    let companies: Vec<Value> = title
        .companies()
        .iter()
        .filter(|c| game.g().company_order.iter().any(|x| x == c.sym) || game.company_idx.get(c.sym).is_some_and(|&i| !game.companies[i].closed))
        .map(|c| {
            let kind = match c.sym.chars().next() {
                Some('M') => "minor",
                Some('C') => "concession",
                _ => "private",
            };
            let category = fam.plus_expansion_bidbox.iter().position(|b| b.contains(&c.sym)).map(|i| ["train", "track", "other"][i]);
            json!({"sym": c.sym, "name": c.name, "value": c.value, "revenue": c.revenue, "kind": kind, "category": category,
                   "blocks": [], "comesWith": []})
        })
        .collect();
    let market: Vec<Value> = game
        .stock_market
        .grid
        .iter()
        .map(|row| json!(row.iter().map(|c| c.as_ref().map_or(Value::Null, |c| json!({"price": c.price,
            "types": c.types.iter().filter(|t| t.as_str() != "normal").collect::<Vec<_>>()}))).collect::<Vec<_>>()))
        .collect();
    let trains: Vec<Value> = title
        .trains()
        .iter()
        .map(|t| {
            let unlimited = t.count == u32::MAX;
            json!({"name": t.name, "distance": t.distance, "price": t.price, "rustsOn": t.rusts_on,
                   "num": if unlimited { 1 } else { title.num_trains(t, &game.optional_rules) }, "unlimited": unlimited,
                   "multiplier": t.multiplier, "events": t.events})
        })
        .collect();
    let phases: Vec<Value> = title
        .phases()
        .iter()
        .map(|p| json!({"name": p.name, "trainLimit": p.train_limit, "minorTrainLimit": p.minor_train_limit, "tiles": p.tiles,
                        "operatingRounds": p.operating_rounds, "status": p.status, "on": p.on}))
        .collect();
    let hexes: Vec<Value> = game.hexes.iter().map(|h| json!({"id": h.id})).collect();
    let n = g.players.len() as u8;
    json!({
        "gameId": g.meta.get("id").cloned().unwrap_or(Value::Null),
        "title": game.title,
        "description": g.meta.get("description").cloned().unwrap_or(json!("")),
        "status": g.meta.get("status").cloned().unwrap_or(json!("")),
        "layout": "flat",
        "players": players,
        "corporations": corporations,
        "companies": companies,
        "market": market,
        "hexes": hexes,
        "trainTypes": trains,
        "phases": phases,
        "bankStart": title.bank_cash() - n as i32 * title.starting_cash(n),
        "certLimit": game.cert_limit,
        "currency": format!("{}%s", currency(g)),
        "optionalRules": game.optional_rules,
        "biddingTokens": (fam.bidding_tokens)(game.players.len()),
        "packKeys": ["corporations.name", "corporations.color", "corporations.textColor", "corporations.logo",
                     "companies.name", "companies.desc", "hexes.x", "hexes.y", "hexes.location", "zones", "description"],
    })
}
