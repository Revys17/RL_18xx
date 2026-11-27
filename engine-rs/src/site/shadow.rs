//! The shadow key, v1 (docs/SITE_ENGINE_RS.md §5.3, contract K4): what decides a game -- money, holdings, track,
//! tokens, trains, who acts and what they may do -- in Ruby's terms, so the server can compare engine-rs with its Ruby
//! engine after every 1830 action. The Ruby twin is rl18xx-next `lib/engine_shadow.rb`; both print the same JSON.
//!
//! Hash keys sorted (serde_json's map), arrays in the order given, integers as integers. Players in Ruby's
//! `engine.players` order (it rotates); corporations keyed by sym, closed ones left out (Ruby's `close_corporation`
//! deletes them from `engine.corporations`). G19's cities in Ruby's numbering. Once the game is finished nobody acts:
//! `acting` [], `entity` / `step` null, `actions` [] (bot_view returns before reading the step).

use super::game::SiteGame;
use super::json::tile_name;
use crate::entities::EntityId;
use crate::rounds::Round;
use serde_json::{json, Map, Value};

/// Ruby's active step's class name (`step.class.name.split('::').last`) in 1830's rounds, from the step the site
/// layer already names (`view::step_label`, Ruby's `description`).
pub fn ruby_step_class(g: &SiteGame) -> Option<&'static str> {
    if g.game.finished {
        return None;
    }
    let label = super::view::step_label(g)?;
    Some(match label.as_str() {
        "Bid on Companies" => "WaterfallAuction",
        "Choose Corporation Par Value" => "CompanyPendingPar",
        "Sell/Buy/Sell Shares" => "BuySellParShares",
        "Discard Trains" => "DiscardTrain",
        "Place teleport token" => "SpecialToken",
        "Lay/Upgrade Track" => "Track",
        "Place a Token" => "Token",
        "Run Routes" => "Route",
        "Pay or Withhold Dividends" => "Dividend",
        "Buy Trains" => "BuyTrain",
        "Buy Companies" => "BuyCompany",
        l if l == "Place Home Token" || l == "Place Token" || (l.starts_with("Place ") && l.ends_with(" Home Token")) => {
            "HomeToken"
        }
        _ => return None,
    })
}

/// Ruby `step.actions(entity)` of the active step for its first active entity (bot_view's `actions`), in Ruby's
/// order: the menu's `types`.
pub fn ruby_actions(g: &mut SiteGame) -> Vec<String> {
    if g.game.finished {
        return Vec::new();
    }
    let mut types = super::menu::ruby_types(g);
    if let Round::Stock(_) = g.game.round {
        let pid = super::view::current_holder(g).and_then(|h| h.strip_prefix("p:").and_then(|p| p.parse::<u32>().ok()));
        if let Some(pid) = pid {
            if g.game.buyable_shares(pid).is_empty() {
                types.retain(|t| t != "buy_shares");
            }
        }
    }
    types
}

/// The entity bot_view names: the active step's first active entity, or the round's current entity (`id.to_s`).
pub fn ruby_entity(g: &SiteGame) -> Option<String> {
    if g.game.finished {
        return None;
    }
    let h = super::view::current_holder(g)?;
    Some(match h.split_once(':') {
        Some((_, id)) => id.to_string(),
        None => h,
    })
}

/// A corporation's used tokens as "<hex>-<city index on its tile>" (Ruby's numbering), in token order.
fn token_places(g: &SiteGame, sym: &str, tokens: &[crate::entities::Token]) -> Vec<String> {
    let mut out = Vec::new();
    if g.game.title == "1867" {
        // the site layer's record of each station's city (Ruby's token.city)
        if let Some(list) = g.token_cities.map.get(sym) {
            for (ti, t) in tokens.iter().enumerate().filter(|(_, t)| t.used) {
                match list.get(ti).cloned().flatten() {
                    Some((h, c)) => out.push(format!("{h}-{c}")),
                    None => out.push(format!("{}-?", t.city_hex_id)),
                }
            }
            return out;
        }
    }
    for t in tokens.iter().filter(|t| t.used) {
        let hex = &t.city_hex_id;
        let Some(h) = g.game.hex_ref(hex) else {
            out.push(format!("{hex}-?"));
            continue;
        };
        let idx = h.tile.cities.iter().position(|c| c.tokens.iter().flatten().any(|x| x.corporation_id == sym));
        let idx = match idx {
            Some(i) if g.swapped.contains(&hex.as_str()) && h.tile.cities.len() == 2 => 1 - i,
            Some(i) => i,
            None => {
                out.push(format!("{hex}-?"));
                continue;
            }
        };
        out.push(format!("{hex}-{idx}"));
    }
    out
}

/// The engine's own price of each route of a site `run_routes` (Ruby `Route#revenue` with nothing posted, 1830: the
/// sum of its stops' revenues in the phase, no bonuses): each stop the route's `nodes` name ("<hex>-<index>" in
/// Ruby's numbering; both cities of an OO hex count), or, for a route stored without nodes (older games), the hexes
/// its connection chains end at.
pub fn price_routes(g: &SiteGame, a: &Value) -> Option<Vec<i32>> {
    let entity = a.get("entity").and_then(Value::as_str)?;
    let routes = a.get("routes").and_then(Value::as_array)?;
    let mut out = Vec::new();
    for r in routes {
        if let Some(nodes) = r.get("nodes").and_then(Value::as_array).filter(|n| !n.is_empty()) {
            let mut sum = 0;
            for n in nodes {
                sum += node_revenue(g, n.as_str()?)?;
            }
            out.push(sum);
            continue;
        }
        let conns: Vec<Vec<String>> = r
            .get("connections")
            .and_then(Value::as_array)?
            .iter()
            .map(|c| c.as_array().map(|xs| xs.iter().filter_map(|x| x.as_str().map(str::to_string)).collect()).unwrap_or_default())
            .collect();
        // a route is a path of chains: a stop inside it ends two chains, a stop at an end one, so a hex ending n
        // chains is visited ceil(n / 2) times (both cities of an OO hex: "F20-G19-F20")
        let mut ends: Vec<(String, u32)> = Vec::new();
        for c in conns.iter().filter(|c| c.len() >= 2) {
            for h in [&c[0], &c[c.len() - 1]] {
                match ends.iter_mut().find(|e| &e.0 == h) {
                    Some(e) => e.1 += 1,
                    None => ends.push((h.clone(), 1)),
                }
            }
        }
        let mut sum = 0;
        for (h, n) in ends {
            sum += crate::revenue::classify_stop(&g.game, &h, entity).ok()?.revenue * ((n + 1) / 2) as i32;
        }
        out.push(sum);
    }
    Some(out)
}

/// A stop's revenue in the phase from its node signature "<hex>-<index>" (Ruby's numbering: G19's preprinted cities
/// swapped). 1830's tiles carry one kind of stop each (cities, towns or an offboard).
fn node_revenue(g: &SiteGame, sig: &str) -> Option<i32> {
    let (hex, idx) = sig.rsplit_once('-')?;
    let idx: usize = idx.parse().ok()?;
    let h = g.game.hex_ref(hex)?;
    let tile = &h.tile;
    if let Some(ob) = tile.offboards.get(idx).or_else(|| tile.offboards.first()) {
        return Some(ob.phase_revenue(&g.game.phase.tiles));
    }
    if !tile.cities.is_empty() {
        let i = if g.swapped.contains(&hex) && tile.cities.len() == 2 { 1 - idx.min(1) } else { idx };
        return tile.cities.get(i).map(|c| c.revenue);
    }
    tile.towns.get(idx).map(|t| t.revenue)
}

fn owner_code(e: &EntityId) -> String {
    if let Some(p) = e.player_id() {
        format!("p{p}")
    } else if let Some(c) = e.corp_sym().filter(|_| e.is_corporation()) {
        format!("c{c}")
    } else {
        "bank".to_string()
    }
}

/// The shadow key of the position now. `last` is the raw action the position follows (None at the start).
pub fn key(g: &mut SiteGame, last: Option<&Value>) -> Value {
    let finished = g.game.finished;
    let is_1867 = super::g1867::is_1867(g);
    let info = super::view::round_info(g);
    let acting: Vec<u32> = if finished {
        Vec::new()
    } else if is_1867 {
        super::g1867::active_players(&mut g.game)
    } else {
        super::view::active_players(g)
    };
    let (entity, step, actions) = if is_1867 {
        let active = super::g1867::active_step(&mut g.game);
        let actions = if finished { Vec::new() } else { super::g1867::ruby_actions(&mut g.game) };
        (
            active.as_ref().map(|(_, e)| super::g1867::entity_id(e)),
            active.map(|(k, _)| super::g1867::step_class(k)),
            actions,
        )
    } else {
        (ruby_entity(g), ruby_step_class(g), ruby_actions(g))
    };
    let priority = super::view::priority(g);
    let auction = super::g1867::key::auction(g);
    // Ruby's `engine.corporations` (1830: every corporation not closed; 1867: §3.1's membership)
    let members: Vec<usize> = if is_1867 {
        super::g1867::ruby_corporations(&g.game)
    } else {
        (0..g.game.corporations.len()).filter(|&i| !g.game.corporations[i].closed).collect()
    };
    let game = &g.game;

    let mut players = Vec::new();
    for pid in super::view::player_order(g) {
        let Some(p) = game.players.iter().find(|p| p.id == pid) else { continue };
        let pe = EntityId::player(pid);
        let mut shares = Map::new();
        for c in members.iter().map(|&i| &game.corporations[i]) {
            let pct: u32 = c.shares.iter().filter(|s| s.owner == pe).map(|s| s.percent as u32).sum();
            if pct > 0 {
                shares.insert(c.sym.clone(), json!(pct));
            }
        }
        let mut companies: Vec<String> =
            game.companies.iter().filter(|c| c.owner == pe && !c.closed).map(|c| c.sym.clone()).collect();
        companies.sort();
        players.push(json!({"id": pid, "cash": p.cash, "shares": shares, "companies": companies, "bankrupt": p.bankrupt}));
    }

    let mut corps = Map::new();
    for c in members.iter().map(|&i| &game.corporations[i]) {
        let sp = c.share_price.as_ref();
        let held: u32 = c.shares.iter().filter(|s| s.owner.is_player() || s.owner.is_market()).map(|s| s.percent as u32).sum();
        let market: u32 = c.shares.iter().filter(|s| s.owner.is_market()).map(|s| s.percent as u32).sum();
        let ce = EntityId::corporation(&c.sym);
        let mut companies: Vec<String> =
            game.companies.iter().filter(|m| m.owner == ce && !m.closed).map(|m| m.sym.clone()).collect();
        companies.sort();
        let last_run = game
            .site
            .as_ref()
            .and_then(|r| r.op_history.get(&c.sym))
            .and_then(|l| l.iter().max_by_key(|e| (e.turn, e.round)))
            .map_or(Value::Null, |e| json!([e.turn, e.round, e.revenue, e.kind]));
        let mut cm = json!({
                "cash": c.cash,
                "price": sp.map(|s| s.price),
                "cell": sp.map(|s| json!([s.row, s.column])),
                // Ruby Corporation#par_price: the share price for a company `always_market_price` (1867's)
                "par": if c.always_market_price { sp.map(|s| s.price) } else { c.ipo_price.as_ref().map(|s| s.price) },
                "floated": c.floated,
                "closed": c.closed,
                "ipo": 100u32.saturating_sub(held),
                "market": market,
                "president": c.president_id(),
                "trains": c.trains.iter().map(|t| t.id.clone()).collect::<Vec<_>>(),
                "tokens": token_places(g, &c.sym, &c.tokens),
                "companies": companies,
                "last_run": last_run,
            });
        if is_1867 {
            if let Some(m) = cm.as_object_mut() {
                super::g1867::key::corp_fields(g, &c.sym, m);
            }
        }
        corps.insert(c.sym.clone(), cm);
    }

    let mut companies = Map::new();
    for m in &game.companies {
        companies.insert(m.sym.clone(), json!(if m.closed { "closed".to_string() } else { owner_code(&m.owner) }));
    }

    let upcoming: Vec<Value> = if is_1867 {
        super::g1867::view::upcoming(g).into_iter().map(|t| t["id"].clone()).collect()
    } else {
        super::view::upcoming(g).into_iter().map(|t| t["id"].clone()).collect()
    };
    let discarded: Vec<String> = game.depot.discarded.iter().map(|t| t.id.clone()).collect();

    let mut hexes = Map::new();
    for h in &game.hexes {
        if let Some(name) = tile_name(&h.tile.name) {
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
                json!({"revenues": revenues, "extra_revenue": 0, "subsidy": 0})
            } else {
                json!({"type": t})
            }
        }
    };

    let mut out = json!({
        "v": super::KEY_VERSION,
        "round": info["name"],
        "turn": game.turn,
        "round_num": info["num"],
        "phase": game.phase.name,
        "bank": game.bank.cash,
        "acting": acting,
        "entity": entity,
        "step": step,
        "actions": actions,
        "priority": priority,
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
    // v2 (docs/SITE_ENGINE_RS_1867.md §6): the fields only some titles have, and every title's auction in progress
    if let Some(m) = out.as_object_mut() {
        if is_1867 {
            super::g1867::key::game_fields(g, m);
        }
        if let Some(a) = auction {
            m.insert("auction".into(), a);
        }
    }
    out
}

/// The raw action the position now follows: the last raw action processed (any type), None at the start.
pub fn last_raw(g: &SiteGame) -> Option<&Value> {
    // a refused action was counted (`pos` is past it) but not applied
    let done = if g.stopped.is_some() { g.pos.saturating_sub(1) } else { g.pos };
    if done == 0 {
        None
    } else {
        g.raw.get(done - 1)
    }
}

/// The key of the position now, after the last raw action processed.
pub fn key_now(g: &mut SiteGame) -> Value {
    let last = last_raw(g).cloned();
    key(g, last.as_ref())
}

/// FNV-1a 64 (stable on every platform: the native and the wasm build must agree byte for byte).
struct Fnv(u64);
impl Fnv {
    fn add(&mut self, s: &str) {
        for &x in (s.len() as u64).to_le_bytes().iter().chain(s.as_bytes()) {
            self.0 = (self.0 ^ x as u64).wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
}

/// The digest of a whole replay as the browser sees it (docs/SITE_ENGINE_RS.md §3.1 determinism, §8.2 native vs
/// wasm): the start log, every step's JSON (action, log, snapshot) and shadow key, and at the end the snapshot, every
/// seated player's menu and the replay's stop. `{digest, steps}`; equal in the native build and the wasm one.
pub fn replay_digest(data: &Value) -> Result<Value, String> {
    let mut g = SiteGame::load(data)?;
    let mut h = Fnv(0xcbf2_9ce4_8422_2325);
    h.add(&json!(g.start_log).to_string());
    let mut steps = 0usize;
    while !g.done() {
        if let Some(s) = g.process_next() {
            steps += 1;
            h.add(&super::view::step_json(&g, &s).to_string());
            h.add(&key(&mut g, Some(&s.action)).to_string());
        }
    }
    h.add(&super::view::snapshot(&g, g.current_action_id).to_string());
    let pids: Vec<u32> = g.players.iter().map(|p| p.0).collect();
    for pid in pids {
        h.add(&super::menu::menu(&mut g, pid).to_string());
    }
    h.add(&format!("{:?}", g.stopped.as_ref().map(|s| (s.action_id, &s.message))));
    Ok(json!({"digest": format!("{:016x}", h.0), "steps": steps}))
}
