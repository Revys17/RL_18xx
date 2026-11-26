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
    let bought = g.turn_actions.iter().any(|a| matches!(a.get("type").and_then(Value::as_str), Some("buy_shares" | "par")));
    let sold = g.turn_actions.iter().any(|a| a.get("type").and_then(Value::as_str) == Some("sell_shares"));
    Before { auctioning: game.auctioning_company(), bids, turn_acted: !g.turn_actions.is_empty(), turn_bought: bought, turn_sold: sold, game }
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
    match crate::title::g1830::location_names().get(hex) {
        Some(loc) if game.title == "1830" => format!("{hex} ({loc})"),
        _ => hex.to_string(),
    }
}

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
        let dir = if p1.2 > p0.2 {
            "right"
        } else if p1.2 < p0.2 {
            "left"
        } else if p1.1 < p0.1 {
            "up"
        } else {
            "down"
        };
        if p0.0 != p1.0 {
            out.push(format!("{}'s share price moves {dir} from ${} to ${}", c.sym, p0.0, p1.0));
        }
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
            out.push(format!("{} receives ${}", c.sym, c.cash - b.corporations.iter().find(|x| x.sym == c.sym).map_or(0, |x| x.cash)));
        }
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
        game.phase.train_limit,
        tiles.join(", ")
    )
}

/// The game's opening lines (before the first action).
pub fn start(g: &SiteGame) -> Vec<String> {
    let mut out = Vec::new();
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
    out.push(phase_line(&g.game));
    out
}

fn op_corp(game: &BaseGame) -> Option<(String, S, usize, u8)> {
    match &game.round {
        Round::Operating(s) => s.current_corp_sym().map(|c| (c.to_string(), s.step.clone(), s.entity_index, s.round_num)),
        _ => None,
    }
}

const STEP_ORDER: &[S] = &[S::LayTile, S::PlaceToken, S::RunRoutes, S::Dividend, S::BuyTrain, S::BuyCompany];

fn step_pos(s: &S) -> usize {
    STEP_ORDER.iter().position(|x| x == s).unwrap_or(STEP_ORDER.len())
}

/// The skip lines of a corporation's steps from `from` (inclusive) to `to` (exclusive): Ruby's log_skip and the
/// dividend step's automatic withhold.
fn skips(b: &BaseGame, a: &BaseGame, corp: &str, from: usize, to: usize, out: &mut Vec<String>) {
    for s in STEP_ORDER.iter().take(to.min(STEP_ORDER.len())).skip(from) {
        match s {
            S::LayTile => out.push(format!("{corp} skips lay track")),
            S::PlaceToken => out.push(format!("{corp} skips place a token")),
            S::RunRoutes => out.push(format!("{corp} skips run routes")),
            S::Dividend => {
                out.push(format!("{corp} does not run"));
                price_moves(b, a, Some(corp), out);
            }
            S::BuyTrain => {
                let full = a.corp_idx.get(corp).is_some_and(|&i| a.corporations[i].trains.len() >= a.phase.train_limit as usize);
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
    let a = &g.game;
    let bg = &b.game;
    let mut out = Vec::new();
    let t = raw.get("type").and_then(Value::as_str).unwrap_or("");
    let ent = super::json::entity_of(raw).unwrap_or_default();
    let pid: Option<u32> = ent.parse().ok();
    let who = pid.map(|p| pname(a, p)).unwrap_or_else(|| ent.clone());
    // the acting player (a corporation's president) and Ruby's "via Master Mode" line
    let acting = pid.or_else(|| bg.corp_idx.get(ent.as_str()).and_then(|&i| bg.corporations[i].president_id()))
        .or_else(|| bg.company_idx.get(ent.as_str()).and_then(|&i| bg.companies[i].owner.player_id()));
    if let Some(user) = raw.get("user").and_then(Value::as_u64) {
        if Some(user as u32) != acting {
            let name = g.players.iter().find(|p| p.0 == user as u32).map(|p| p.1.clone()).unwrap_or_else(|| "Owner".into());
            out.push(format!("• Action({t}) via Master Mode by: {name}"));
        }
    }
    let n = out.len();
    let price = raw.get("price").and_then(Value::as_i64).unwrap_or(0);
    match (t, &bg.round) {
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
            None => out.push(format!("{who} passes bidding")),
        },
        ("par", _) => {
            let corp = raw.get("corporation").and_then(Value::as_str).unwrap_or("");
            let p = a.corporations.iter().find(|c| c.sym == corp).and_then(|c| c.ipo_price.as_ref().map(|s| s.price)).unwrap_or(0);
            out.push(format!("{who} pars {corp} at ${p}"));
            if matches!(bg.round, Round::Auction(_)) {
                out.push(format!("{who} receives a 20% share of {corp}"));
            } else {
                out.push(format!("{who} buys a 20% share of {corp} from the IPO for ${}", 2 * p));
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
            let unit = if from_market { corp_b.and_then(|c| c.share_price.as_ref()).map_or(0, |s| s.price) } else { corp_b.and_then(|c| c.ipo_price.as_ref()).map_or(0, |s| s.price) };
            let paid = unit * pct as i32 / 10;
            out.push(format!("{who} buys a {pct}% share of {corp} from {} for ${paid}", if from_market { "the market" } else { "the IPO" }));
        }
        ("sell_shares", _) => {
            let shares = raw.get("shares").and_then(Value::as_array).cloned().unwrap_or_default();
            let sid = shares.first().and_then(Value::as_str).unwrap_or("");
            let corp = sid.rfind('_').map_or("", |p| &sid[..p]);
            let pct = raw.get("percent").and_then(Value::as_i64).unwrap_or(10);
            let got = pid.map_or(0, |p| a.players.iter().find(|x| x.id == p).map_or(0, |x| x.cash) - bg.players.iter().find(|x| x.id == p).map_or(0, |x| x.cash));
            let what = if pct == 10 { "a 10% share".to_string() } else { format!("{} shares", pct / 10) };
            out.push(format!("{who} sells {what} of {corp} and receives ${got}"));
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
            let spent = bg.corp_idx.get(corp.as_str()).map_or(0, |&i| bg.corporations[i].cash) - a.corp_idx.get(corp.as_str()).map_or(0, |&i| a.corporations[i].cash);
            let spend = if spent > 0 { format!("spends ${spent} and ") } else { String::new() };
            out.push(format!("{corp} {spend}lays tile #{name} with rotation {rot} on {}", hex_desc(a, hex)));
        }
        ("place_token", _) => {
            let corp = g.operator_of(&ent);
            let city = raw.get("city").and_then(Value::as_str).unwrap_or("");
            // the hex the token went to: the city's tile
            let hex = a.hexes.iter().find(|h| city.starts_with(&format!("{}-", h.tile.name)) || city.starts_with(&format!("{}-0-", h.id))).map(|h| h.id.clone()).unwrap_or_default();
            let spent = bg.corp_idx.get(corp.as_str()).map_or(0, |&i| bg.corporations[i].cash) - a.corp_idx.get(corp.as_str()).map_or(0, |&i| a.corporations[i].cash);
            // Ruby: "places a token on <hex> (<location>) <price>" (the price part empty for a free one)
            let price_part = if spent > 0 { format!(" for ${spent}") } else { String::new() };
            out.push(format!("{corp} places a token on {} {price_part}", hex_desc(a, &hex)));
        }
        ("run_routes", _) => {
            for r in raw.get("routes").and_then(Value::as_array).cloned().unwrap_or_default() {
                let train = r.get("train").and_then(Value::as_str).unwrap_or("");
                let tname = train.rsplit_once('-').map_or(train, |x| x.0);
                let rev = r.get("revenue").and_then(Value::as_i64).unwrap_or(0);
                // the site's revenue_str: the stops' hexes as the train reaches them (a hex stopped at twice, twice)
                let s = r.get("hexes").and_then(Value::as_array).map(|h| h.iter().filter_map(Value::as_str).collect::<Vec<_>>().join("-"))
                    .filter(|x| !x.is_empty())
                    .or_else(|| r.get("revenue_str").and_then(Value::as_str).map(str::to_string))
                    .unwrap_or_default();
                out.push(format!("{ent} runs a {tname} train for ${rev}: {s}"));
            }
        }
        ("dividend", _) => {
            let kind = raw.get("kind").and_then(Value::as_str).unwrap_or("");
            let rev = match &bg.round {
                Round::Operating(s) => s.revenue,
                _ => 0,
            };
            if kind == "payout" {
                let per = rev / 10;
                // each holder's dividend, the market's shares paying the company (1830), largest first; among equal
                // amounts the players in the order they took the company's shares, then the company
                let mut pays: Vec<(i32, usize, String)> = Vec::new();
                let holders = g.holders.get(&ent).cloned().unwrap_or_default();
                for pid2 in super::view::player_order(g) {
                    let n: i32 = bg.corp_idx.get(ent.as_str()).map_or(0, |&ci| bg.corporations[ci].shares.iter().filter(|s| s.owner == EntityId::player(pid2)).map(|s| s.percent as i32 / 10).sum());
                    if n * per > 0 {
                        pays.push((-(n * per), holders.iter().position(|h| *h == pid2).unwrap_or(usize::MAX - 1), pname(a, pid2)));
                    }
                }
                let pool: i32 = bg.corp_idx.get(ent.as_str()).map_or(0, |&ci| bg.corporations[ci].shares.iter().filter(|s| s.owner.is_market()).map(|s| s.percent as i32 / 10).sum());
                if pool * per > 0 {
                    pays.push((-(pool * per), usize::MAX, ent.clone()));
                }
                pays.sort();
                let list: Vec<String> = pays.iter().map(|(amt, _, nm)| format!("${} to {nm}", -amt)).collect();
                out.push(format!("{ent} pays out ${rev} = ${per} per share ({})", list.join(", ")));
            } else {
                out.push(format!("{ent} withholds ${rev}"));
            }
            price_moves(bg, a, Some(&ent), &mut out);
        }
        ("buy_train", _) => {
            let train = raw.get("train").and_then(Value::as_str).unwrap_or("");
            let tname = train.rsplit_once('-').map_or(train, |x| x.0);
            let seller = bg.corporations.iter().find(|c| c.trains.iter().any(|x| x.id == train)).map(|c| c.sym.clone());
            let from = seller.unwrap_or_else(|| "The Depot".into());
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
            let comp = raw.get("company").and_then(Value::as_str).unwrap_or("");
            let owner = bg.company_idx.get(comp).map(|&i| holder_name(bg, &bg.companies[i].owner)).unwrap_or_default();
            out.push(format!("{who} buys {} from {owner} for ${price}", cname(a, comp)));
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
            }
        }
        _ => {}
    }
    let _ = n;
    // consequences: presidencies, floats, price moves of a sale, privates granted or closed, phase, rusting
    if t == "sell_shares" || t == "buy_shares" || t == "par" || t == "bankrupt" {
        presidents(bg, a, &mut out);
    }
    if t == "sell_shares" {
        let shares = raw.get("shares").and_then(Value::as_array).cloned().unwrap_or_default();
        let sid = shares.first().and_then(Value::as_str).unwrap_or("");
        let corp = sid.rfind('_').map_or("", |p| &sid[..p]).to_string();
        let n0 = out.len();
        price_moves(bg, a, Some(&corp), &mut out);
        if out.len() == n0 {
            if let Some(p) = price_of(a, &corp) {
                out.push(format!("{corp}'s share price moves down from ${} to ${}", p.0, p.0));
            }
        }
    }
    floats(bg, a, &mut out);
    if a.bank_broken && !bg.bank_broken {
        out.push("-- The bank has broken --".into());
    }
    if a.phase.name != bg.phase.name {
        out.push(phase_line(a));
        let rusted: Vec<(String, usize)> = bg
            .corporations
            .iter()
            .filter_map(|c| {
                let after = a.corporations.iter().find(|x| x.sym == c.sym)?;
                let lost = c.trains.iter().filter(|t| !after.trains.iter().any(|x| x.id == t.id) && Some(t.id.as_str()) != raw.get("train").and_then(Value::as_str)).count();
                (lost > 0).then(|| (c.sym.clone(), lost))
            })
            .collect();
        if !rusted.is_empty() {
            let name = bg.corporations.iter().flat_map(|c| c.trains.iter()).find(|t| rusted.iter().any(|(s, _)| bg.corporations.iter().any(|c| &c.sym == s && c.trains.iter().any(|x| x.id == t.id)))).map(|t| t.name.clone()).unwrap_or_default();
            let mut r = rusted.clone();
            r.sort();
            let list: Vec<String> = r.iter().map(|(s, k)| format!("{s} x{k}")).collect();
            out.push(format!("-- Event: {name} trains rust ({}) --", list.join(", ")));
        }
    }
    let closing: Vec<String> = a.companies.iter().enumerate().filter(|(i, c)| c.closed && !bg.companies[*i].closed).map(|(_, c)| c.sym.clone()).collect();
    if closing.len() > 1 && a.phase.name != bg.phase.name {
        out.push("-- Event: Private companies close --".into());
    } else {
        for sym in closing {
            out.push(format!("{} closes", cname(a, &sym)));
        }
    }
    // the auction's resolutions
    if let Round::Auction(_) = &bg.round {
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

/// What happens between the action and the next decision: skipped steps, other companies' turns, skipped players,
/// round ends and starts.
fn transitions(g: &SiteGame, b: &Before, _raw: &Value, out: &mut Vec<String>) {
    let a = &g.game;
    let bg = &b.game;
    let same_round = bg.round.round_type_str() == a.round.round_type_str() && bg.round.round_num() == a.round.round_num() && bg.turn == a.turn;
    if let (Round::Operating(s0), Round::Operating(s1)) = (&bg.round, &a.round) {
        if s0.pending_tokens.is_empty() {
            if let Some((sym, 0, _)) = s1.pending_tokens.first() {
                out.push(format!("{sym} must choose city for token"));
            }
        }
    }
    // the operating company's remaining steps
    if let Some((corp, step, idx, _)) = op_corp(bg) {
        let home_choice = matches!(&bg.round, Round::Operating(s0) if !s0.pending_tokens.is_empty() && step == S::PlaceToken);
        let next_from = if home_choice { step_pos(&S::PlaceToken) } else { step_pos(&step) + 1 };
        match op_corp(a) {
            Some((c2, s2, i2, _)) if same_round && c2 == corp && i2 == idx => {
                if step_pos(&s2) > next_from {
                    skips(bg, a, &corp, next_from, step_pos(&s2), out);
                }
            }
            _ => {
                if step_pos(&step) < STEP_ORDER.len() {
                    skips(bg, a, &corp, next_from, STEP_ORDER.len(), out);
                }
            }
        }
    }
    // stock round: players skipped because they can't act
    if let (Round::Stock(s0), Round::Stock(s1)) = (&bg.round, &a.round) {
        if same_round {
            let order = &s0.player_order;
            let from = s0.current_player_index;
            let to = s1.current_player_index;
            if from != to {
                let n = order.len();
                let mut k = (from + 1) % n;
                while k != to {
                    out.push(format!("{} has no valid actions and passes", pname(a, order[k])));
                    k = (k + 1) % n;
                }
            }
        }
    }
    if same_round {
        return;
    }
    // a round ended: the stock round's sold-out price rises
    if let Round::Stock(_) = &bg.round {
        price_moves(bg, a, None, out);
    }
    // a new round
    match &a.round {
        Round::Stock(_) => {
            if matches!(bg.round, Round::Auction(_)) {
                out.push(format!("{} has priority deal", pname(a, a.priority_deal_player)));
            }
            out.push(format!("-- Stock Round {} --", a.turn));
        }
        Round::Operating(s) => {
            if matches!(bg.round, Round::Stock(_)) {
                out.push(format!("{} has priority deal", pname(a, a.priority_deal_player)));
            }
            out.push(format!("-- Operating Round {}.{} (of {}) --", a.turn, s.round_num, s.total_ors));
            // private revenues, in the players' order then the companies'
            for pid in super::view::player_order(g) {
                for sym in super::view::owned_companies(g, &EntityId::player(pid)) {
                    if let Some(c) = a.companies.iter().find(|c| c.sym == sym) {
                        if c.revenue > 0 {
                            out.push(format!("{} collects ${} from {}", pname(a, pid), c.revenue, cname(a, &sym)));
                        }
                    }
                }
            }
            for corp in &a.corporations {
                for sym in super::view::owned_companies(g, &EntityId::corporation(&corp.sym)) {
                    if let Some(c) = a.companies.iter().find(|c| c.sym == sym) {
                        if c.revenue > 0 {
                            out.push(format!("{} collects ${} from {}", corp.sym, c.revenue, cname(a, &sym)));
                        }
                    }
                }
            }
        }
        _ => {}
    }
}

/// The lines of a company's turn starting (after `explain`): "X operates Y", its home station.
pub fn turn_start(g: &SiteGame, b: &Before) -> Vec<String> {
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
        if corp != now {
            skips(&b.game, a, corp, 0, STEP_ORDER.len(), &mut out);
        } else if step_pos(&s.step) > 0 {
            skips(&b.game, a, corp, 1, step_pos(&s.step), &mut out);
        }
    }
    out
}
