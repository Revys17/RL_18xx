//! What the board renders, in the client's shapes (living-board `src/engine/types.ts`, built there by
//! `adapter.ts` from Opal objects): the engine part of StaticInfo, the Snapshot after every kept action, and a step's
//! ActionLite with its run's track. Names, colours, texts and tile drawings come from the title pack (B's).

use super::game::SiteGame;
use super::json::tile_name;
use crate::entities::EntityId;
use crate::rounds::Round;
use crate::title::HexType;
use serde_json::{json, Map, Value};
use std::collections::BTreeMap;

/// Ruby's name for a preprinted hex's tile: a hex whose code is one of the generic tiles' codes takes that tile's
/// name ("blank", "town", "city"); any other is named by its coordinate.
pub fn preprinted_name(game: &crate::game::BaseGame, hex: &str) -> String {
    let title = game.title_def();
    if title.preprinted_hex_dsl(hex).is_some() {
        return hex.to_string();
    }
    if let Some(d) = title.hex_definitions().into_iter().find(|d| d.coord == hex) {
        if d.terrain_cost == 0 && d.borders.is_empty() && title.hex_label(hex).is_none() {
            match d.hex_type {
                HexType::Blank => return "blank".into(),
                HexType::Town { revenue: 0 } => return "town".into(),
                HexType::City { revenue: 0, slots: 1 } => return "city".into(),
                _ => {}
            }
        }
    }
    hex.to_string()
}

/// The TileSpec key of a hex's tile: "<Ruby tile name>@<rotation>".
pub fn tile_key(g: &SiteGame, hex: &str) -> String {
    let Some(h) = g.game.hex_ref(hex) else { return String::new() };
    let name = match tile_name(&h.tile.name) {
        Some(n) => n.to_string(),
        None => preprinted_name(&g.game, hex),
    };
    format!("{name}@{}", h.tile.rotation)
}

fn holder(e: &EntityId) -> Option<String> {
    if let Some(p) = e.player_id() {
        Some(format!("p:{p}"))
    } else if e.is_corporation() {
        e.corp_sym().map(|s| format!("c:{s}"))
    } else {
        None
    }
}

fn train_card(t: &crate::entities::Train) -> Value {
    json!({"id": t.id, "name": t.name})
}

/// The round's name, short name, label, turn and number (adapter.ts `roundInfo`).
pub fn round_info(g: &SiteGame) -> Value {
    let turn = g.game.turn;
    match &g.game.round {
        Round::Auction(_) => json!({"name": "Auction Round", "short": "PR", "label": "Private Auction", "turn": turn, "num": 1}),
        Round::Stock(_) => json!({"name": "Stock Round", "short": "SR", "label": format!("Stock Round {turn}"), "turn": turn, "num": 1}),
        Round::Operating(s) => json!({"name": "Operating Round", "short": "OR",
            "label": format!("Operating Round {turn}.{} of {}", s.round_num, s.total_ors), "turn": turn, "num": s.round_num}),
        Round::Merger(s) => json!({"name": "Merger Round", "short": "MR", "label": "Merger Round", "turn": turn, "num": s.round_num}),
    }
}

/// Ruby's active step's description (`round.active_step.description`), or None.
pub fn step_label(g: &SiteGame) -> Option<String> {
    use crate::rounds::OperatingStep as S;
    let game = &g.game;
    Some(match &game.round {
        Round::Auction(s) => {
            if s.pending_par.is_some() {
                "Choose Corporation Par Value".into()
            } else {
                "Bid on Companies".into()
            }
        }
        Round::Stock(_) => "Sell/Buy/Sell Shares".into(),
        Round::Operating(s) => {
            if !s.crowded_corps.is_empty() {
                return Some("Discard Trains".into());
            }
            if s.teleport_pending {
                return Some("Place teleport token".into());
            }
            if let (S::PlaceToken, Some((sym, idx, _))) = (&s.step, s.pending_tokens.first()) {
                let cur = s.current_corp_sym().unwrap_or("");
                return Some(if sym != cur {
                    format!("Place {sym} Home Token")
                } else if *idx == 0 {
                    "Place Home Token".into()
                } else {
                    "Place Token".into()
                });
            }
            match s.step {
                S::LayTile => "Lay/Upgrade Track".into(),
                S::PlaceToken => "Place a Token".into(),
                S::RunRoutes => "Run Routes".into(),
                S::Dividend => "Pay or Withhold Dividends".into(),
                S::BuyTrain => "Buy Trains".into(),
                S::BuyCompany => "Buy Companies".into(),
                _ => return None,
            }
        }
        Round::Merger(_) => return None,
    })
}

/// The entity acting now as the snapshot names it ("p:13", "c:PRR").
pub fn current_holder(g: &SiteGame) -> Option<String> {
    // a teleporting private places its station itself (Ruby's SpecialToken step: the company acts)
    if let Round::Operating(s) = &g.game.round {
        if s.teleport_pending {
            if let Some(cur) = s.current_corp_sym() {
                let me = EntityId::corporation(cur);
                if let Some(c) = g.game.companies.iter().find(|c| c.owner == me && !c.closed && crate::abilities::teleport(&g.game.title, &c.sym).is_some()) {
                    return Some(format!("m:{}", c.sym));
                }
            }
        }
    }
    holder(&g.game.round_state.active_entity_id)
}

/// The players who must act (Ruby `active_players`: a corporation's president in an operating round).
pub fn active_players(g: &SiteGame) -> Vec<u32> {
    if g.game.finished {
        return vec![];
    }
    g.game.acting_player_id_pub().into_iter().collect()
}

/// Ruby `priority_deal_player`: in a round of players, the engine's; otherwise the first player not bankrupt.
pub fn priority(g: &SiteGame) -> u32 {
    match &g.game.round {
        Round::Auction(_) | Round::Stock(_) => g.game.priority_deal_player_py().id,
        _ => player_order(g)
            .into_iter()
            .find(|p| g.game.players.iter().any(|x| x.id == *p && !x.bankrupt))
            .unwrap_or(g.game.priority_deal_player),
    }
}

/// The seating order now (Ruby rotates `@players` after the auction and each stock round).
pub fn player_order(g: &SiteGame) -> Vec<u32> {
    let seats = &g.game.player_order;
    let first = seats.iter().position(|p| *p == g.game.priority_deal_player).unwrap_or(0);
    seats[first..].iter().chain(seats[..first].iter()).copied().collect()
}

pub fn snapshot(g: &SiteGame, action_id: i64) -> Value {
    let game = &g.game;
    let results = game.calculate_results();
    let order = player_order(g);
    let mut players: Vec<Map<String, Value>> = Vec::new();
    for pid in &order {
        let Some(p) = game.players.iter().find(|p| p.id == *pid) else { continue };
        let pe = EntityId::player(p.id);
        let mut certs = Map::new();
        let mut count = Map::new();
        for c in &game.corporations {
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
        m.insert("numCerts".into(), json!(game.num_certs(p.id)));
        m.insert("id".into(), json!(p.id));
        m.insert("cash".into(), json!(p.cash));
        m.insert("value".into(), json!(results.get(&p.id).copied().unwrap_or(p.cash)));
        m.insert("certs".into(), Value::Object(certs));
        m.insert("certCount".into(), Value::Object(count));
        m.insert("president".into(), json!([]));
        m.insert("companies".into(), json!(owned_companies(g, &pe)));
        m.insert("bankrupt".into(), json!(p.bankrupt));
        players.push(m);
    }

    let (is_or, or_order, or_index) = match &game.round {
        Round::Operating(s) => (true, s.operating_order.clone(), s.entity_index),
        _ => (false, vec![], 0),
    };
    let cur_pos: Option<usize> = if !is_or || or_index >= or_order.len() {
        None
    } else {
        match current_holder(g) {
            Some(h) if h.starts_with("c:") => or_order.iter().position(|s| *s == h[2..]),
            Some(_) => None,
            None => Some(or_index),
        }
    };
    let mut corps = Vec::new();
    for c in &game.corporations {
        let president = c.president_id();
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
        // Ruby: shares not held by players or the market are the corporation's (its IPO), from the start
        let held: u32 = c.shares.iter().filter(|s| s.owner.is_player() || s.owner.is_market()).map(|s| s.percent as u32).sum();
        let ipo_pct: u32 = 100u32.saturating_sub(held);
        let pool_pct: u32 = c.shares.iter().filter(|s| s.owner.is_market()).map(|s| s.percent as u32).sum();
        let ipoed = c.ipo_price.is_some();
        // Ruby percent_to_float: the IPO's percent less what the IPO keeps when it floats (adapter.ts: >= 0)
        let to_float = if c.floated { 0 } else { (ipo_pct as i64 - (100 - c.float_percent as i64)).max(0) };
        // adapter.ts: the companies before the round's current entity in its order (none when that is a private)
        let operated = is_or && cur_pos.is_some_and(|ci| matches!(or_order.iter().position(|s| s == &c.sym), Some(i) if i < ci));
        let mut cm = Map::new();
        cm.insert("sym".into(), json!(c.sym));
        cm.insert("cash".into(), json!(c.cash));
        cm.insert("price".into(), json!(sp.map(|s| s.price)));
        cm.insert("cell".into(), sp.map_or(Value::Null, |s| json!([s.row, s.column])));
        cm.insert("stack".into(), json!(stack));
        cm.insert("par".into(), json!(c.ipo_price.as_ref().map(|s| s.price)));
        cm.insert("ipoed".into(), json!(ipoed));
        cm.insert("floated".into(), json!(c.floated));
        cm.insert("toFloat".into(), json!(to_float));
        cm.insert("closed".into(), json!(c.closed));
        cm.insert("ipoPct".into(), json!(ipo_pct));
        cm.insert("poolPct".into(), json!(pool_pct));
        cm.insert("president".into(), json!(president));
        cm.insert("trains".into(), json!(c.trains.iter().map(train_card).collect::<Vec<_>>()));
        cm.insert("tokensLeft".into(), json!(unused.len()));
        cm.insert("nextTokenPrice".into(), json!(unused.first().map(|t| t.price)));
        cm.insert("companies".into(), json!(owned_companies(g, &EntityId::corporation(&c.sym))));
        cm.insert("operated".into(), json!(operated));
        cm.insert("lastRun".into(), last_run(g, &c.sym));
        corps.push(Value::Object(cm));
    }

    let companies: Vec<Value> = game
        .companies
        .iter()
        .map(|m| json!({"sym": m.sym, "owner": if m.closed { Value::Null } else { holder(&m.owner).map_or(Value::Null, Value::from) }, "closed": m.closed}))
        .collect();

    let hexes = hex_states(g);

    let mut available: Vec<String> = Vec::new();
    {
        let mut names: Vec<String> = Vec::new();
        let first = game.depot.trains.first();
        for t in first.into_iter().chain(game.depot.trains.iter().filter(|t| game.phase_available(t.available_on.as_deref()))) {
            if !names.contains(&t.name) {
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
    }

    let mut bids = Vec::new();
    let mut auction = Vec::new();
    if let Round::Auction(s) = &game.round {
        if s.pending_par.is_none() && !game.finished {
            for sym in game.auction_companies() {
                let mut list = game.auction_bids(&sym);
                list.sort_by_key(|b| b.1);
                for (pid, price) in list {
                    bids.push(json!({"company": sym, "player": pid, "price": price}));
                }
            }
            for sym in game.auction_companies() {
                auction.push(json!({"sym": sym, "minBid": game.auction_min_bid(&sym)}));
            }
        }
    }

    let finished = game.finished;
    let (result, end_reason) = if finished {
        let mut r = Map::new();
        for (k, v) in g.result() {
            r.insert(k.to_string(), json!(v));
        }
        (Value::Object(r), json!(g.end_reason()))
    } else {
        (Value::Null, Value::Null)
    };

    let (order_list, order_index) = round_order(g);
    let (op_order, waiting) = operating_order(g);

    let mut out = Map::new();
    if let Some(p) = programs(g) {
        out.insert("programs".into(), p);
    }
    if let Some(s) = sold(g) {
        out.insert("sold".into(), s);
    }
    out.insert("actionId".into(), json!(action_id));
    out.insert("round".into(), round_info(g));
    out.insert("stepLabel".into(), if finished { Value::Null } else { json!(step_label(g)) });
    out.insert("phase".into(), json!(game.phase.name));
    out.insert("trainLimit".into(), json!(game.phase.train_limit));
    out.insert("bank".into(), json!(game.bank.cash));
    out.insert("activePlayers".into(), json!(active_players(g)));
    out.insert("current".into(), if finished { Value::Null } else { json!(current_holder(g)) });
    out.insert("priority".into(), json!(priority(g)));
    out.insert("players".into(), Value::Array(players.into_iter().map(Value::Object).collect()));
    out.insert("corps".into(), Value::Array(corps));
    out.insert("companies".into(), Value::Array(companies));
    out.insert("hexes".into(), hexes);
    out.insert(
        "depot".into(),
        json!({"upcoming": upcoming(g),
               "discarded": game.depot.discarded.iter().map(train_card).collect::<Vec<_>>(), "available": available}),
    );
    out.insert("bids".into(), Value::Array(bids));
    out.insert("auction".into(), Value::Array(auction));
    out.insert("finished".into(), json!(finished));
    out.insert("result".into(), result);
    out.insert("endReason".into(), end_reason);
    out.insert("order".into(), json!(order_list));
    out.insert("orderIndex".into(), json!(order_index));
    out.insert("opOrder".into(), json!(op_order));
    out.insert("waiting".into(), json!(waiting));
    Value::Object(out)
}

/// The board's hexes: tile key, tokens per city slot, home reservations per city, blocking privates. City order as
/// Ruby numbers it (G19 swapped while preprinted).
pub fn hex_states(g: &SiteGame) -> Value {
    let game = &g.game;
    let reservations = game.home_reservations();
    let mut blockers: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for c in &game.companies {
        if c.closed || !c.owner.is_player() {
            continue;
        }
        if let Some((_, hexes)) = crate::abilities::blocks_hexes(&game.title, &c.sym) {
            for h in hexes {
                blockers.entry(h.to_string()).or_default().push(c.sym.clone());
            }
        }
    }
    let mut out = Map::new();
    for h in &game.hexes {
        let mut st = Map::new();
        st.insert("k".into(), json!(tile_key(g, &h.id)));
        let swap = g.swapped.contains(&h.id.as_str()) && h.tile.cities.len() == 2;
        let order: Vec<usize> = if swap { vec![1, 0] } else { (0..h.tile.cities.len()).collect() };
        if !h.tile.cities.is_empty() {
            let mut any = false;
            let tok: Vec<Value> = order
                .iter()
                .map(|&ci| {
                    let city = &h.tile.cities[ci];
                    json!(city.tokens.iter().map(|t| match t {
                        Some(t) => {
                            any = true;
                            json!(t.corporation_id)
                        }
                        None => Value::Null,
                    }).collect::<Vec<_>>())
                })
                .collect();
            if any {
                st.insert("tok".into(), Value::Array(tok));
            }
            let mut any_res = false;
            let res: Vec<Value> = order
                .iter()
                .map(|&ci| {
                    let r: Vec<String> = reservations
                        .iter()
                        .filter(|(hx, c, _)| hx == &h.id && *c == ci)
                        .map(|(_, _, corp)| corp.clone())
                        .collect();
                    if !r.is_empty() {
                        any_res = true;
                    }
                    json!(r)
                })
                .collect();
            if any_res {
                st.insert("res".into(), Value::Array(res));
            }
        }
        if let Some(b) = blockers.get(&h.id) {
            st.insert("blk".into(), json!(b));
        }
        out.insert(h.id.clone(), Value::Object(st));
    }
    Value::Object(out)
}

/// The round's turn order and the index of the one acting (adapter.ts: `round.entities`, `round.entity_index`).
pub fn round_order(g: &SiteGame) -> (Vec<String>, i64) {
    let game = &g.game;
    let (list, idx) = round_order_raw(g);
    // past the last entity (a finished round): Ruby's index stays on the last one
    let idx = if !list.is_empty() && idx >= list.len() as i64 { list.len() as i64 - 1 } else { idx };
    let _ = game;
    (list, idx)
}

fn round_order_raw(g: &SiteGame) -> (Vec<String>, i64) {
    let game = &g.game;
    match &game.round {
        Round::Auction(s) => (s.player_order.iter().map(|p| format!("p:{p}")).collect(), s.entity_index as i64),
        Round::Stock(s) => (s.player_order.iter().map(|p| format!("p:{p}")).collect(), s.current_player_index as i64),
        Round::Operating(s) => (s.operating_order.iter().map(|c| format!("c:{c}")).collect(), s.entity_index as i64),
        Round::Merger(_) => (vec![], -1),
    }
}

/// The Order column: the companies that operate, in order, and the started ones that don't yet (by share price).
pub fn operating_order(g: &SiteGame) -> (Vec<String>, Vec<String>) {
    let game = &g.game;
    let op: Vec<String> = match &game.round {
        Round::Operating(s) => s.operating_order.clone(),
        _ => game.compute_operating_order(),
    };
    let mut keyed: Vec<((i32, i32, u8, usize, String), String)> = Vec::new();
    for c in &game.corporations {
        if op.contains(&c.sym) || c.closed {
            continue;
        }
        let Some(sp) = c.share_price.as_ref() else { continue };
        let pos = game.market_cell_corps.get(&(sp.row, sp.column)).and_then(|l| l.iter().position(|s| s == &c.sym)).unwrap_or(0);
        keyed.push(((-sp.price, -(sp.column as i32), sp.row, pos, c.sym.clone()), c.sym.clone()));
    }
    keyed.sort();
    (op, keyed.into_iter().map(|k| k.1).collect())
}

/// A corporation's latest operating result (adapter.ts `lastRun`).
pub fn last_run(g: &SiteGame, sym: &str) -> Value {
    let Some(rec) = g.game.site.as_ref() else { return Value::Null };
    let Some(list) = rec.op_history.get(sym) else { return Value::Null };
    let Some(best) = list.iter().max_by_key(|e| (e.turn, e.round)) else { return Value::Null };
    json!({"revenue": best.revenue, "kind": if best.no_run { "no_run" } else { best.kind.as_str() }, "or": format!("{}.{}", best.turn, best.round)})
}

/// Who sold what this stock round, by corporation.
pub fn sold(g: &SiteGame) -> Option<Value> {
    let Round::Stock(s) = &g.game.round else { return None };
    let mut out: BTreeMap<String, Vec<u32>> = BTreeMap::new();
    for pid in &player_order(g) {
        if let Some(m) = s.players_sold.get(pid) {
            let mut syms: Vec<&String> = m.keys().collect();
            syms.sort();
            for sym in syms {
                out.entry(sym.clone()).or_default().push(*pid);
            }
        }
    }
    Some(json!(out))
}

/// The players' programmed actions (display: the snapshot's `programs`).
pub fn programs(g: &SiteGame) -> Option<Value> {
    if g.programs.is_empty() {
        return None;
    }
    let mut out = Map::new();
    for (pid, list) in &g.programs {
        out.insert(pid.to_string(), json!(list.iter().map(program_of).collect::<Vec<_>>()));
    }
    Some(Value::Object(out))
}

/// living-board `programs.ts` `programFrom`: a program's display form.
fn program_of(a: &Value) -> Value {
    let t = a.get("type").and_then(Value::as_str).unwrap_or("");
    let f = |k: &str| a.get(k).filter(|v| !v.is_null());
    let truthy = |k: &str| f(k).is_some_and(|v| v.as_bool().unwrap_or(true));
    let (label, short) = match t {
        "program_share_pass" => (
            format!("Auto-pass {}{}", if truthy("indefinite") { "in every stock round" } else { "for this stock round" },
                if truthy("unconditional") { ", whatever others do" } else { "" }),
            "auto-pass".to_string(),
        ),
        "program_buy_shares" => {
            let corp = f("corporation").and_then(Value::as_str).unwrap_or("?").to_string();
            let until = match f("until_condition") {
                Some(Value::String(x)) if x == "float" => "it floats".to_string(),
                Some(v) => {
                    let n = v.as_i64().unwrap_or(0);
                    format!("you hold {} share{}", v.as_str().map(str::to_string).unwrap_or(n.to_string()), if n == 1 { "" } else { "s" })
                }
                None => "you hold undefined shares".to_string(),
            };
            (format!("Buy {corp} from the {} until {until}{}", if truthy("from_market") { "market" } else { "IPO" },
                if truthy("auto_pass_after") { ", then auto-pass" } else { "" }), format!("auto-buy {corp}"))
        }
        "program_auction_bid" => (format!("Auto-bid on {}", f("bid_target").and_then(Value::as_str).unwrap_or("?")), "auto-bid".into()),
        "program_close_pass" => ("Auto-pass in closing rounds".into(), "auto-pass".into()),
        "program_merger_pass" => ("Auto-pass in mergers".into(), "auto-pass".into()),
        _ => (t.replacen("program_", "auto ", 1).replace('_', " "), "auto".into()),
    };
    let mut p = Map::new();
    p.insert("type".into(), json!(t));
    p.insert("label".into(), json!(label));
    p.insert("short".into(), json!(short));
    if let Some(v) = f("corporation") {
        p.insert("corporation".into(), v.clone());
    }
    if let Some(v) = f("until_condition") {
        p.insert("until".into(), v.clone());
    }
    for (k, out) in [("from_market", "fromMarket"), ("auto_pass_after", "autoPassAfter"), ("unconditional", "unconditional"), ("indefinite", "indefinite")] {
        if f(k).is_some() {
            p.insert(out.into(), json!(truthy(k)));
        }
    }
    Value::Object(p)
}

/// The companies an entity holds, in the order it acquired them (Ruby's `companies` lists).
pub fn owned_companies(g: &SiteGame, owner: &EntityId) -> Vec<String> {
    let mut list: Vec<(u64, usize, String)> = g
        .game
        .companies
        .iter()
        .enumerate()
        .filter(|(_, c)| &c.owner == owner && !c.closed)
        .map(|(i, c)| (g.company_seq.get(&c.sym).copied().unwrap_or(0), i, c.sym.clone()))
        .collect();
    list.sort();
    list.into_iter().map(|x| x.2).collect()
}

/// The depot's upcoming trains (Ruby holds one D at a time: it is unlimited).
fn upcoming(g: &SiteGame) -> Vec<Value> {
    let mut out = Vec::new();
    let mut seen_d = false;
    for t in &g.game.depot.trains {
        if t.name == "D" {
            if seen_d {
                continue;
            }
            seen_d = true;
        }
        out.push(train_card(t));
    }
    out
}

/// The step's action as the board shows it: the site action's own fields (adapter.ts ACTION_KEYS), a run's routes
/// with their track.
pub fn action_lite(g: &SiteGame, raw: &Value) -> Value {
    const KEYS: &[&str] = &[
        "id", "type", "entity", "entity_type", "user", "price", "company", "corporation", "share_price", "shares", "percent",
        "hex", "tile", "rotation", "city", "train", "variant", "kind", "routes", "message", "created_at",
        "unconditional", "indefinite", "until_condition", "from_market", "auto_pass_after", "reason", "original_type",
    ];
    const AUTO: &[&str] = &[
        "type", "entity", "entity_type", "corporation", "shares", "reason", "original_type", "until_condition", "from_market",
        "auto_pass_after", "unconditional", "indefinite", "price", "company",
    ];
    let mut out = Map::new();
    for k in KEYS {
        if let Some(v) = raw.get(*k).filter(|v| !v.is_null()) {
            out.insert(k.to_string(), v.clone());
        }
    }
    if let Some(Value::Array(autos)) = raw.get("auto_actions") {
        if !autos.is_empty() {
            let list: Vec<Value> = autos
                .iter()
                .map(|x| {
                    let mut a = Map::new();
                    for k in AUTO {
                        if let Some(v) = x.get(*k).filter(|v| !v.is_null()) {
                            a.insert(k.to_string(), v.clone());
                        }
                    }
                    Value::Object(a)
                })
                .collect();
            out.insert("auto_actions".into(), Value::Array(list));
        }
    }
    if out.get("type").and_then(Value::as_str) == Some("run_routes") {
        if let Some(Value::Array(routes)) = out.get_mut("routes") {
            for r in routes.iter_mut() {
                if let Some(track) = super::track::route_track(g, r) {
                    r["track"] = track;
                }
            }
        }
    }
    Value::Object(out)
}

/// The engine part of StaticInfo (types.ts): what depends on the players, the optional rules or the title's rules
/// data. Names, colours, logos, descriptions, hex geometry and texts come from the title pack.
pub fn static_info(g: &SiteGame) -> Value {
    let game = &g.game;
    let title = game.title_def();
    let players: Vec<Value> = g
        .players
        .iter()
        .enumerate()
        .map(|(i, (id, name, bot))| json!({"id": id, "name": name, "bot": bot, "seat": i}))
        .collect();
    let corporations: Vec<Value> = title
        .corporations()
        .iter()
        .map(|c| {
            json!({"sym": c.sym, "name": c.name, "home": c.home_hex, "tokenCount": c.token_prices.len(),
                   "floatPercent": c.float_percent, "capitalization": "full", "sharePercent": 10})
        })
        .collect();
    let companies: Vec<Value> = title
        .companies()
        .iter()
        .map(|c| {
            let blocks: Vec<&str> = crate::abilities::blocks_hexes(&game.title, c.sym).map(|(_, h)| h.to_vec()).unwrap_or_default();
            let comes: Vec<Value> = match crate::abilities::share_grant(&game.title, c.sym) {
                Some(share) => {
                    let (corp, idx) = share.rsplit_once('_').unwrap_or((share, "1"));
                    let president = idx == "0";
                    vec![json!({"corp": corp, "pct": if president { 20 } else { 10 }, "president": president})]
                }
                None => vec![],
            };
            json!({"sym": c.sym, "name": c.name, "value": c.value, "revenue": c.revenue, "blocks": blocks, "comesWith": comes})
        })
        .collect();
    let market: Vec<Value> = game
        .stock_market
        .grid
        .iter()
        .map(|row| json!(row.iter().map(|c| c.as_ref().map_or(Value::Null, |c| json!({"price": c.price, "types": c.types}))).collect::<Vec<_>>()))
        .collect();
    let trains: Vec<Value> = title
        .trains()
        .iter()
        .map(|t| {
            let num = title.num_trains(t, &game.optional_rules);
            json!({"name": t.name, "distance": if t.name == "D" { json!(999) } else { json!(t.distance) }, "price": t.price,
                   "rustsOn": t.rusts_on, "num": num, "unlimited": t.name == "D"})
        })
        .collect();
    let phases: Vec<Value> = title
        .phases()
        .iter()
        .map(|p| json!({"name": p.name, "trainLimit": p.train_limit, "tiles": p.tiles}))
        .collect();
    let hexes: Vec<Value> = game.hexes.iter().map(|h| json!({"id": h.id})).collect();
    json!({
        "gameId": g.meta.get("id").cloned().unwrap_or(Value::Null),
        "title": "1830",
        "description": g.meta.get("description").cloned().unwrap_or(json!("")),
        "status": g.meta.get("status").cloned().unwrap_or(json!("")),
        "layout": "pointy",
        "players": players,
        "corporations": corporations,
        "companies": companies,
        "market": market,
        "hexes": hexes,
        "trainTypes": trains,
        "phases": phases,
        "bankStart": title.bank_cash(),
        "certLimit": game.cert_limit,
        "currency": "$%s",
        "packKeys": ["corporations.name", "corporations.color", "corporations.textColor", "corporations.logo",
                     "companies.name", "companies.desc", "hexes.x", "hexes.y", "hexes.location", "zones", "description"],
    })
}

/// A step as the worker posts it: {index, actionId, action, log, snap}.
pub fn step_json(g: &SiteGame, s: &super::game::StepOut) -> Value {
    json!({"index": s.index, "actionId": s.action_id, "action": action_lite(g, &s.action), "log": s.log,
           "snap": snapshot(g, s.action_id)})
}
