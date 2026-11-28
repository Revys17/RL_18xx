//! The shadow key of an 1822-family position (docs/SITE_ENGINE_RS_1822.md §4.3): v2's fields in Ruby's terms (the
//! twin of rl18xx-next `lib/engine_shadow.rb` `capture`) and the fields only the family has. Players by their site ids.
//!
//! | Field | Ruby |
//! |---|---|
//! | `bidboxes` `{minors, concessions, privates}` | `engine.bidbox_minors` / `bidbox_concessions` / `bidbox_privates` ids |
//! | `stacks` | `engine.companies.map(&:id)` (the stacks' order: P17 moves cards) |
//! | `auction` | `{lot: null, bids: [[company, player, price], ...]}` from the stock round's `@bids`, sorted by (company, price, player) |
//! | `debt` | `{player id => player.debt}`, the non-zero ones |
//! | `tax_haven` | `engine.tax_haven` (P16's dummy player; 1822CA P11): `{cash, shares: {corp => percent}}` |
//! | `credits` | the phase-revenue companies' dummy players' cash (P15 / P20; 1822CA P8 / P9), while in play |
//! | `exchange_tokens` | `{major => engine.exchange_tokens(major)}`, the non-zero ones |
//! | `reservations` | `{"<hex>-<city index>" => [reserved ids]}` for every city with a reservation (Ruby `city.reservations` compacted) |
//! | `tile_reservations` | `{hex => [ids]}` (London's tile reservation: minor 14) |
//! | `sawmill` (1822CA) | `{hex, owner}`: `@sawmill_hex`, `@sawmill_owner` (a closed sawmill's) |
//! | `destinated` (1822CA) | `@destinated` ids, sorted |

use super::super::game::SiteGame;
use crate::entities::EntityId;
use crate::g1822::RoundKind;
use serde_json::{json, Map, Value};

/// A station's place "<hex>-<city index>" (Ruby `token_place`: `token.city.hex.id`, `token.city.index`).
fn token_places(g: &SiteGame, sym: &str) -> Vec<String> {
    let game = &g.game;
    let Some(&ci) = game.corp_idx.get(sym) else { return Vec::new() };
    let mut out = Vec::new();
    // (two of a corporation's tokens can share a hex but not a city: each used token takes the next city of its hex
    // holding one of its tokens)
    let mut seen: Vec<(String, usize)> = Vec::new();
    for t in game.corporations[ci].tokens.iter().filter(|t| t.used) {
        let hex = &t.city_hex_id;
        let Some(h) = game.hex_ref(hex) else {
            out.push(format!("{hex}-?"));
            continue;
        };
        let idx = h.tile.cities.iter().enumerate().position(|(k, c)| {
            !seen.contains(&(hex.clone(), k))
                && c.tokens.iter().flatten().any(|x| x.corporation_id == sym && x.token_type == t.token_type)
        });
        let idx = idx.or_else(|| h.tile.cities.iter().position(|c| c.tokens.iter().flatten().any(|x| x.corporation_id == sym)));
        match idx {
            Some(k) => {
                seen.push((hex.clone(), k));
                out.push(format!("{hex}-{k}"));
            }
            None => out.push(format!("{hex}-?")),
        }
    }
    out
}

/// Ruby `priority_deal_player`: in a round of players the one after the last to act (the first when nobody has),
/// otherwise the first player.
pub fn priority(g: &SiteGame) -> Option<u32> {
    let game = &g.game;
    let players: Vec<u32> = game.players.iter().filter(|p| !p.bankrupt).map(|p| p.id).collect();
    if players.is_empty() {
        return None;
    }
    let r = game.rnd();
    let player_round = matches!(r.kind, RoundKind::Stock | RoundKind::Choices);
    let p = if player_round {
        match r.last_to_act.and_then(|l| players.iter().position(|x| *x == l)) {
            Some(i) => players[(i + 1) % players.len()],
            None => players[0],
        }
    } else {
        players[0]
    };
    Some(g.site_id(p))
}

/// The shadow key of the position now, after `last` (the raw action it follows).
pub fn key(g: &mut SiteGame, last: Option<&Value>) -> Value {
    let finished = g.game.finished;
    let game = &g.game;
    let active = super::active(game);
    let acting: Vec<u32> = if finished { Vec::new() } else { super::active_players(g) };
    let entity = active.as_ref().map(|(_, _, e)| super::entity_site_id(g, e));
    let step = active.as_ref().map(|(_, s, _)| super::step_class(*s));
    let actions = if finished { Vec::new() } else { super::ruby_actions(game) };
    let members = super::members(game);
    let mut players = Vec::new();
    for p in &game.players {
        let pe = EntityId::player(p.id);
        let mut shares = Map::new();
        for c in members.iter().map(|&i| &game.corporations[i]) {
            let pct: u32 = c.shares.iter().filter(|s| s.owner == pe).map(|s| s.percent as u32).sum();
            if pct > 0 {
                shares.insert(c.sym.clone(), json!(pct));
            }
        }
        let mut companies: Vec<String> = game.companies.iter().filter(|c| c.owner == pe && !c.closed && !game.g().owner_only.contains(&c.sym)).map(|c| c.sym.clone()).collect();
        companies.sort();
        players.push(json!({"id": g.site_id(p.id), "cash": p.cash, "shares": shares, "companies": companies, "bankrupt": p.bankrupt}));
    }
    let mut corps = Map::new();
    for c in members.iter().map(|&i| &game.corporations[i]) {
        let sp = c.share_price.as_ref();
        let ipo: u32 = c.shares.iter().filter(|s| s.owner == EntityId::ipo(&c.sym) || s.owner == EntityId::corporation(&c.sym)).map(|s| s.percent as u32).sum();
        let market: u32 = c.shares.iter().filter(|s| s.owner.is_market()).map(|s| s.percent as u32).sum();
        let ce = EntityId::corporation(&c.sym);
        let mut companies: Vec<String> = game.companies.iter().filter(|m| m.owner == ce && !m.closed).map(|m| m.sym.clone()).collect();
        companies.sort();
        let last_run = game
            .site
            .as_ref()
            .and_then(|r| r.op_history.get(&c.sym))
            .and_then(|l| l.iter().max_by_key(|e| (e.turn, e.round)))
            .map_or(Value::Null, |e| json!([e.turn, e.round, e.revenue, e.kind]));
        let president = super::corp_owner(game, &c.sym).map(|p| g.site_id(p));
        corps.insert(
            c.sym.clone(),
            json!({
                "cash": c.cash,
                "price": sp.map(|s| s.price),
                "cell": sp.map(|s| json!([s.row, s.column])),
                // Ruby Corporation#par_price: the share price for a company `always_market_price` (1822's minors)
                "par": if c.always_market_price { sp.map(|s| s.price) } else { c.ipo_price.as_ref().map(|s| s.price) },
                "floated": c.floated,
                "closed": c.closed,
                "ipo": ipo,
                "market": market,
                "president": president,
                "trains": c.trains.iter().map(|t| t.id.clone()).collect::<Vec<_>>(),
                "tokens": token_places(g, &c.sym),
                "companies": companies,
                "last_run": last_run,
            }),
        );
    }
    let st = game.g();
    let mut companies = Map::new();
    for sym in &st.company_order {
        let Some(&mi) = game.company_idx.get(sym.as_str()) else { continue };
        let m = &game.companies[mi];
        let code = if m.closed && !st.deleted_open.contains(sym) {
            "closed".to_string()
        } else if st.bank_owned.contains(sym) {
            "bank".to_string()
        } else {
            super::holder_code(g, &m.owner)
        };
        companies.insert(sym.clone(), json!(code));
    }
    let upcoming = view_upcoming(game);
    let discarded: Vec<String> = game.depot.discarded.iter().map(|t| t.id.clone()).collect();
    let mut hexes = Map::new();
    for h in &game.hexes {
        // (a laid tile: Ruby's `h.tile != h.original_tile`; the preprinted ones are the engine's "preprinted_*")
        if h.tile.id.starts_with("preprinted_") {
            continue;
        }
        if let Some(name) = super::super::json::tile_name(&h.tile.name) {
            hexes.insert(h.id.clone(), json!(format!("{name}@{}", h.tile.rotation)));
        }
    }
    let mut result = Map::new();
    for (id, v) in g.result() {
        result.insert(id.to_string(), json!(v));
    }
    let last_action = match last {
        None => Value::Null,
        Some(a) => {
            let t = a.get("type").and_then(Value::as_str).unwrap_or("");
            if t == "run_routes" {
                let id = a.get("id").and_then(Value::as_i64).unwrap_or(0);
                let revenues = match &g.last_run_prices {
                    Some((pid, revs)) if *pid == id => json!(revs),
                    _ => Value::Null,
                };
                json!({"revenues": revenues, "extra_revenue": a.get("extra_revenue").and_then(Value::as_i64).unwrap_or(0),
                       "subsidy": a.get("subsidy").and_then(Value::as_i64).unwrap_or(0)})
            } else {
                json!({"type": t})
            }
        }
    };
    let mut out = json!({
        "v": super::super::KEY_VERSION,
        "round": super::round_name(game),
        "turn": game.turn,
        "round_num": game.rnd().round_num,
        "phase": game.phase.name,
        "bank": game.bank.cash,
        "acting": acting,
        "entity": entity,
        "step": step,
        "actions": actions,
        "priority": priority(g),
        "players": players,
        "corps": corps,
        "companies": companies,
        "depot": {"upcoming": upcoming, "discarded": discarded},
        "hexes": hexes,
        "finished": finished,
        "result": result,
        "manually_ended": finished && g.manually_ended,
        "game_end_reason": g.end_reason(),
        "last_action": last_action,
    });
    if let Some(m) = out.as_object_mut() {
        family_fields(g, m);
    }
    out
}

/// Ruby's depot.upcoming: the trains still to be sold, in order, one card of the unlimited type (the 7).
pub fn view_upcoming(game: &crate::game::BaseGame) -> Vec<String> {
    let mut out = Vec::new();
    for t in &game.depot.trains {
        out.push(t.id.clone());
        if t.name == "7" {
            break;
        }
    }
    out
}

/// The family's own fields (the table in the module doc).
fn family_fields(g: &SiteGame, m: &mut Map<String, Value>) {
    let game = &g.game;
    let st = game.g();
    m.insert(
        "bidboxes".into(),
        json!({"minors": game.bidbox_minors(), "concessions": game.bidbox_concessions(), "privates": game.bidbox_privates()}),
    );
    m.insert("stacks".into(), json!(st.company_order));
    if !game.finished && game.rnd().kind == RoundKind::Stock {
        let mut bids: Vec<(String, u32, i32)> = Vec::new();
        for (co, list) in &st.round.bids {
            for b in list {
                bids.push((co.clone(), g.site_id(b.player), b.price));
            }
        }
        if !bids.is_empty() {
            bids.sort_by(|a, b| (&a.0, a.2, a.1).cmp(&(&b.0, b.2, b.1)));
            m.insert("auction".into(), json!({"lot": null, "bids": bids.iter().map(|(c, p, x)| json!([c, p, x])).collect::<Vec<_>>()}));
        }
    }
    let debt: Map<String, Value> = st.player_debt.iter().filter(|(_, d)| **d != 0).map(|(p, d)| (g.site_id(*p).to_string(), json!(d))).collect();
    m.insert("debt".into(), Value::Object(debt));
    let th = EntityId(crate::g1822::TAX_HAVEN_OWNER.to_string());
    let mut th_shares = Map::new();
    for c in super::members(game).into_iter().map(|i| &game.corporations[i]) {
        let pct = c.percent_owned_by(&th);
        if pct > 0 {
            th_shares.insert(c.sym.clone(), json!(pct));
        }
    }
    m.insert("tax_haven".into(), json!({"cash": st.tax_haven_cash, "shares": th_shares}));
    let credits: Map<String, Value> = st.phase_revenue.iter().filter_map(|(c, v)| v.map(|x| (c.clone(), json!(x)))).collect();
    m.insert("credits".into(), Value::Object(credits));
    let ex: Map<String, Value> = st.exchange_tokens.iter().filter(|(_, n)| **n > 0).map(|(c, n)| (c.clone(), json!(n))).collect();
    m.insert("exchange_tokens".into(), Value::Object(ex));
    let mut res = Map::new();
    for (k, slots) in &st.city_reservations {
        let ids: Vec<&String> = slots.iter().flatten().collect();
        if ids.is_empty() {
            continue;
        }
        let place = k.replacen('#', "-", 1);
        res.insert(place, json!(ids));
    }
    m.insert("reservations".into(), Value::Object(res));
    let tres: Map<String, Value> = st.tile_reservations.iter().filter(|(_, v)| !v.is_empty()).map(|(h, v)| (h.clone(), json!(v))).collect();
    m.insert("tile_reservations".into(), Value::Object(tres));
    if super::is_ca(game) {
        m.insert("sawmill".into(), json!({"hex": st.ca.sawmill_hex, "owner": st.ca.sawmill_owner}));
        let mut d = st.ca.destinated.clone();
        d.sort();
        m.insert("destinated".into(), json!(d));
    }
}
