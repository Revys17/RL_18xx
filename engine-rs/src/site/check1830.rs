//! 1830's live move check (docs/SITE_ENGINE_RS_STEP3.md §5.2). engine-rs's 1830 processing trusts what it is given
//! the way Ruby's `load` does (a lay's rotation and track, a station's connection, a blocked hex are checked only
//! while Ruby isn't loading: `Tracker#lay_tile`, `Token#process_place_token`, `Tokener#check_connected`). As the
//! authority, a posted move must be a move of the site menu of a seat that must act (the RawMenu's complete actions,
//! a bid's or a priced option's range, the tile options of the hex it lays on, the station options) or of Ruby's
//! cross-player set the menus leave out (another president's company's train; another player's private in a stock
//! round while `site_interplayer_purchase_open`), matched on an outcome key like 1867's (`g1867/act.rs`): type, entity,
//! company / corporation, a bundle's corporation, holder, percent and president's certificate, hex, tile name and
//! rotation, city, train, kind, price on its range.

use super::game::{refused, SiteGame};
use crate::rounds::{OperatingStep as S, Round};
use serde_json::{json, Value};

fn s(v: &Value, k: &str) -> Option<String> {
    v.get(k).filter(|x| !x.is_null()).map(|x| x.as_str().map(str::to_string).unwrap_or_else(|| x.to_string()))
}

/// A tile or train instance's name ("57-1" -> "57", "D-3" -> "D").
fn base(id: &str) -> String {
    match id.rsplit_once('-') {
        Some((n, i)) if i.parse::<u32>().is_ok() => n.to_string(),
        _ => id.to_string(),
    }
}

/// An option a seat may take: a complete site action, and the price range when the player names the price.
#[derive(Clone, Debug)]
struct Opt {
    action: Value,
    range: Option<(i64, i64)>,
}

/// Every option of a menu (any `action` object; a sibling `min` / `max` makes its price the player's).
fn collect(v: &Value, out: &mut Vec<Opt>) {
    match v {
        Value::Object(m) => {
            if let Some(a) = m.get("action").filter(|a| a.get("type").is_some()) {
                let ranged = a.get("price").is_none() && m.get("min").is_some_and(Value::is_i64) && m.get("max").is_some_and(Value::is_i64);
                let range = ranged.then(|| (m["min"].as_i64().unwrap_or(0), m["max"].as_i64().unwrap_or(0)));
                out.push(Opt { action: a.clone(), range });
            }
            for (k, x) in m {
                if k != "action" {
                    collect(x, out);
                }
            }
        }
        Value::Array(xs) => xs.iter().for_each(|x| collect(x, out)),
        _ => {}
    }
}

/// The auction's bids: each lot at any price from its minimum to the player's maximum (and its face value when it
/// may be bought outright).
fn bids(menu: &Value, pid: u32, out: &mut Vec<Opt>) {
    for lot in menu.get("auction").and_then(Value::as_array).cloned().unwrap_or_default() {
        let (lo, hi) = (lot["minBid"].as_i64().unwrap_or(0), lot["maxBid"].as_i64().unwrap_or(0));
        let a = json!({"type": "bid", "entity": pid, "entity_type": "player", "company": lot["sym"]});
        out.push(Opt { action: a.clone(), range: Some((lo, hi)) });
        if lot["buy"].as_bool() == Some(true) {
            let v = lot["value"].as_i64().unwrap_or(0);
            out.push(Opt { action: a, range: Some((v, v)) });
        }
    }
}

/// Where a share is: "ipo", "market", or its holder's entity id.
fn share_place(g: &SiteGame, id: &str) -> Option<(String, String, u8, bool)> {
    let (corp, idx) = id.rsplit_once('_')?;
    let idx: usize = idx.parse().ok()?;
    let &ci = g.game.corp_idx.get(corp)?;
    let sh = g.game.corporations[ci].shares.iter().find(|x| x.index == idx)?;
    let place = if sh.owner.is_market() {
        "market".to_string()
    } else if sh.owner.is_player() {
        sh.owner.0.clone()
    } else {
        "ipo".to_string()
    };
    Some((corp.to_string(), place, sh.percent, sh.president))
}

/// Where a train is (its holder's sym, "depot" for the depot's) and its name.
fn train_place(g: &SiteGame, id: &str) -> Option<(String, String)> {
    let game = &g.game;
    if let Some(t) = game.depot.trains.iter().chain(game.depot.discarded.iter()).find(|t| t.id == id) {
        return Some(("depot".into(), t.name.clone()));
    }
    game.corporations.iter().find_map(|c| c.trains.iter().find(|t| t.id == id).map(|t| (c.sym.clone(), t.name.clone())))
}

/// Whether the posted `a` is the option `w` (its price apart).
fn same(g: &SiteGame, a: &Value, w: &Value) -> bool {
    if s(a, "type") != s(w, "type") || s(a, "entity") != s(w, "entity") {
        return false;
    }
    if let (Some(x), Some(y)) = (s(a, "entity_type"), s(w, "entity_type")) {
        if x != y {
            return false;
        }
    }
    for k in ["corporation", "company", "hex", "kind"] {
        if let Some(wv) = s(w, k) {
            if s(a, k) != Some(wv) {
                return false;
            }
        }
    }
    if let Some(wv) = s(w, "share_price") {
        let av = s(a, "share_price").unwrap_or_default();
        let price = |x: &str| x.split(',').next().unwrap_or("").to_string();
        if av != wv && !(av.split(',').count() == 1 && price(&av) == price(&wv)) {
            return false;
        }
    }
    if let Some(wv) = w.get("rotation").and_then(Value::as_i64) {
        if a.get("rotation").and_then(Value::as_i64) != Some(wv) {
            return false;
        }
    }
    if let Some(wt) = s(w, "tile") {
        let Some(at) = s(a, "tile") else { return false };
        // the same tile, an instance not on the map
        if base(&at) != base(&wt) || (at != wt && g.game.hexes.iter().any(|h| h.tile.name == at)) {
            return false;
        }
    }
    if let Some(wc) = s(w, "city") {
        if s(a, "city") != Some(wc) {
            return false;
        }
    }
    if let Some(wt) = s(w, "train") {
        let Some(at) = s(a, "train") else { return false };
        if at != wt {
            // another company's train: any of the same type it holds (the depot's: that one)
            match (train_place(g, &at), train_place(g, &wt)) {
                (Some((ha, na)), Some((hw, nw))) if ha != "depot" && ha == hw && na == nw => {}
                _ => return false,
            }
        }
        if let Some(wv) = s(w, "variant") {
            if s(a, "variant").unwrap_or_else(|| base(&at)) != wv {
                return false;
            }
        }
    }
    if let Some(we) = s(w, "exchange") {
        let Some(ae) = s(a, "exchange") else { return false };
        if ae != we && train_place(g, &ae) != train_place(g, &we) {
            return false;
        }
    } else if s(a, "exchange").is_some() {
        return false;
    }
    if let Some(Value::Array(ws)) = w.get("shares") {
        let xs: Vec<String> = a.get("shares").and_then(Value::as_array).map(|v| v.iter().filter_map(|y| y.as_str().map(str::to_string)).collect()).unwrap_or_default();
        let ys: Vec<String> = ws.iter().filter_map(|v| v.as_str().map(str::to_string)).collect();
        if xs.len() != ys.len() || xs.is_empty() {
            return false;
        }
        let mut px: Vec<(String, String, u8, bool)> = Vec::new();
        for x in &xs {
            match share_place(g, x) {
                Some(p) => px.push(p),
                None => return false,
            }
        }
        let mut py: Vec<(String, String, u8, bool)> = ys.iter().filter_map(|y| share_place(g, y)).collect();
        // the same corporation, holder, percents and president's certificate: the same bundle
        px.sort();
        py.sort();
        if px != py {
            return false;
        }
        let mut dedup = xs.clone();
        dedup.sort();
        dedup.dedup();
        if dedup.len() != xs.len() {
            return false;
        }
        if let (Some(ap), Some(wp)) = (a.get("percent").and_then(Value::as_i64), w.get("percent").and_then(Value::as_i64)) {
            if ap != wp {
                return false;
            }
        }
    }
    true
}

/// The price on the option's range, or its own price.
fn price_ok(a: &Value, o: &Opt) -> Result<(), String> {
    let ap = a.get("price").and_then(Value::as_i64);
    if let Some((lo, hi)) = o.range {
        let Some(p) = ap else { return Err("The move needs a price".into()) };
        if p < lo || p > hi {
            return Err(format!("Price must be between ${lo} and ${hi}"));
        }
        return Ok(());
    }
    if let Some(wp) = o.action.get("price").and_then(Value::as_i64) {
        if ap != Some(wp) {
            return Err(format!("The price is ${wp}"));
        }
    }
    Ok(())
}

/// Ruby's moves the site menus never offer but Ruby accepts (§5.2): another president's company's train
/// (`Depot#other_trains`, `spend_minmax`), another player's private bought by a company (`purchasable_companies`, the
/// private's price range) or by a player in a stock round (`site_interplayer_purchase_open`), M&H's exchange whoever's
/// turn it is (a company's step is never blocking: `Round#active_step(company)`), any crowded company's discard.
fn cross_player(g: &mut SiteGame, a: &Value, out: &mut Vec<Opt>) {
    let t = s(a, "type").unwrap_or_default();
    match (t.as_str(), &g.game.round) {
        ("buy_train", Round::Operating(st)) if st.step == S::BuyTrain && st.crowded_corps.is_empty() => {
            let Some(corp) = st.current_corp_sym().map(str::to_string) else { return };
            let last_sold = st.last_share_sold_price;
            let ci = g.game.corp_idx[&corp];
            let power = g.game.corporations[ci].cash;
            let pres = g.game.corporations[ci].president_id();
            let pres_cash = pres.and_then(|p| g.game.players.iter().find(|x| x.id == p)).map_or(0, |p| p.cash);
            let types = super::menu::ruby_types(g);
            if !types.iter().any(|x| x == "buy_train") {
                return;
            }
            for (oi, other) in g.game.corporations.iter().enumerate() {
                if oi == ci || other.closed || other.president_id() == pres {
                    continue;
                }
                for tr in &other.trains {
                    let face = tr.price;
                    let (min, max) = if power < face {
                        let possible = power + pres_cash;
                        (last_sold.map_or(1, |l| possible - l + 1), face.min(possible))
                    } else {
                        (1, power)
                    };
                    out.push(Opt {
                        action: json!({"type": "buy_train", "entity": corp, "entity_type": "corporation", "train": tr.id}),
                        range: Some((min as i64, max as i64)),
                    });
                }
            }
        }
        ("buy_company", Round::Operating(st)) if st.crowded_corps.is_empty() => {
            let Some(corp) = st.current_corp_sym().map(str::to_string) else { return };
            if !super::menu::ruby_buy_company_open(g, &corp) {
                return;
            }
            let ci = g.game.corp_idx[&corp];
            let cash = g.game.corporations[ci].cash;
            for c in &g.game.companies {
                if c.closed || !c.owner.is_player() || c.no_buy {
                    continue;
                }
                let (min, max) = g.game.title_def().company_buy_price_range(c.value);
                out.push(Opt {
                    action: json!({"type": "buy_company", "entity": corp, "entity_type": "corporation", "company": c.sym}),
                    range: Some((min as i64, max.min(cash) as i64)),
                });
            }
        }
        ("buy_company", Round::Stock(st)) => {
            let Some(pid) = g.game.round_state.active_entity_id.player_id() else { return };
            if !g.game.site_interplayer_purchase_open(pid, st.bought_this_turn || st.parred_this_turn) {
                return;
            }
            let cash = g.game.players.iter().find(|p| p.id == pid).map_or(0, |p| p.cash);
            let me = crate::entities::EntityId::player(pid);
            for c in &g.game.companies {
                if c.closed || !c.owner.is_player() || c.owner == me || crate::abilities::no_buy(&g.game.title, &c.sym) {
                    continue;
                }
                out.push(Opt {
                    action: json!({"type": "buy_company", "entity": pid, "entity_type": "player", "company": c.sym}),
                    range: Some((0, cash as i64)),
                });
            }
        }
        ("buy_shares", _) if s(a, "entity_type").as_deref() == Some("company") => {
            let owner = s(a, "entity").and_then(|c| g.game.company_idx.get(c.as_str()).copied()).and_then(|mi| g.game.companies[mi].owner.player_id());
            if let Some(pid) = owner {
                for o in super::menu::exchange_offers(g, pid) {
                    out.push(Opt { action: o["action"].clone(), range: None });
                }
            }
        }
        ("discard_train", Round::Operating(st)) => {
            for sym in st.crowded_corps.clone() {
                let ci = g.game.corp_idx[&sym];
                for tr in &g.game.corporations[ci].trains {
                    out.push(Opt {
                        action: json!({"type": "discard_train", "entity": sym, "entity_type": "corporation", "train": tr.id}),
                        range: None,
                    });
                }
            }
        }
        _ => {}
    }
}

/// A posted sale's bundle as Ruby's `SellShares.from_h` builds it (`ShareBundle.new(shares, percent)`): distinct
/// shares of one corporation, all the seller's, the percent their sum, or less by whole shares when the president's
/// certificate is among them (a partial bundle). Then Ruby's `can_sell?` is the menu's for that corporation and
/// percent, and `can_dump?` for the president's certificate: another holder with as much as it is.
fn sale_ok(g: &SiteGame, a: &Value, w: &Value) -> bool {
    if s(a, "type").as_deref() != Some("sell_shares") || s(w, "type").as_deref() != Some("sell_shares") || s(a, "entity") != s(w, "entity") {
        return false;
    }
    let Some(pid) = s(a, "entity").and_then(|e| e.parse::<u32>().ok()) else { return false };
    let me = format!("player:{pid}");
    let ids: Vec<String> = a.get("shares").and_then(Value::as_array).map(|v| v.iter().filter_map(|y| y.as_str().map(str::to_string)).collect()).unwrap_or_default();
    let mut dedup = ids.clone();
    dedup.sort();
    dedup.dedup();
    if ids.is_empty() || dedup.len() != ids.len() {
        return false;
    }
    let mut corp = None;
    let (mut sum, mut pres_pct) = (0i64, None);
    for id in &ids {
        let Some((c, place, pct, pres)) = share_place(g, id) else { return false };
        if place != me || corp.as_ref().is_some_and(|x| *x != c) {
            return false;
        }
        corp = Some(c);
        sum += pct as i64;
        if pres {
            pres_pct = Some(pct as i64);
        }
    }
    let corp = corp.unwrap_or_default();
    let w_corp = w.get("shares").and_then(Value::as_array).and_then(|v| v.first()).and_then(Value::as_str).and_then(|x| x.rsplit_once('_')).map(|x| x.0.to_string());
    if w_corp.as_deref() != Some(corp.as_str()) {
        return false;
    }
    let pct = a.get("percent").and_then(Value::as_i64).unwrap_or(sum);
    let Some(&ci) = g.game.corp_idx.get(corp.as_str()) else { return false };
    let unit = g.game.corporations[ci].share_unit_percent.max(1) as i64;
    let shape = pct == sum || pres_pct.is_some_and(|p| pct > 0 && pct < sum && (sum - pct) % unit == 0 && sum - pct <= p - unit);
    if !shape || w.get("percent").and_then(Value::as_i64) != Some(pct) {
        return false;
    }
    if let Some(p) = pres_pct {
        let c = &g.game.corporations[ci];
        let best_other = g.game.players.iter().filter(|x| x.id != pid).map(|x| c.percent_owned_by(&crate::entities::EntityId::player(x.id)) as i64).max().unwrap_or(0);
        if best_other < p {
            return false;
        }
    }
    true
}

/// The options of the position for the posted move's kind: every acting seat's menu, a lay's tile options on its
/// hex, the cross-player set.
fn options(g: &mut SiteGame, a: &Value) -> Vec<Opt> {
    let t = s(a, "type").unwrap_or_default();
    let mut out = Vec::new();
    for pid in super::view::active_players(g) {
        let m = super::menu::menu(g, pid);
        if m.is_null() {
            continue;
        }
        collect(&m, &mut out);
        bids(&m, pid, &mut out);
        if t == "lay_tile" {
            let hex = s(a, "hex").unwrap_or_default();
            let ent = s(a, "entity").unwrap_or_default();
            if s(a, "entity_type").as_deref() == Some("company") {
                let offered = m.get("abilities").and_then(Value::as_array).is_some_and(|abs| {
                    abs.iter().any(|ab| ab["company"] == json!(ent) && ab["hexes"].as_array().is_some_and(|h| h.iter().any(|x| x == &json!(hex))))
                });
                if offered {
                    if let Ok(v) = super::menu::tile_options(g, &hex, Some(&ent)) {
                        collect(&v, &mut out);
                    }
                }
            } else if m.get("tileHexes").and_then(Value::as_array).is_some_and(|h| h.iter().any(|x| x == &json!(hex))) {
                if let Ok(v) = super::menu::tile_options(g, &hex, None) {
                    collect(&v, &mut out);
                }
            }
        }
    }
    cross_player(g, a, &mut out);
    out
}

/// The verdict on a posted 1830 engine move (not a run: the route check's): Ok, or the refusal's words.
pub fn check_move(g: &mut SiteGame, a: &Value) -> Result<(), String> {
    let t = s(a, "type").unwrap_or_default();
    let opts = options(g, a);
    let mut price_err = None;
    for o in &opts {
        if same(g, a, &o.action) || sale_ok(g, a, &o.action) {
            match price_ok(a, o) {
                Ok(()) => return Ok(()),
                Err(e) => price_err = Some(e),
            }
        }
    }
    if let Some(e) = price_err {
        return Err(refused(e));
    }
    let who = s(a, "entity").unwrap_or_default();
    let same_type = opts.iter().any(|o| s(&o.action, "type").as_deref() == Some(t.as_str()));
    Err(refused(if same_type {
        format!("Not a legal {} for {who} now", t.replace('_', " "))
    } else {
        format!("{who} cannot {} now", t.replace('_', " "))
    }))
}
