//! 1867's parts of what the board renders (docs/SITE_ENGINE_RS_1867.md §3.4): the Snapshot and the engine part of
//! StaticInfo in living-board's shapes (`types.ts`), equal to Opal's adapter (`adapter.ts`, which reads them off
//! Ruby's objects). Step 1's 1830 code (`view.rs`) is not touched: 1867's snapshot is built here, from the same
//! helpers where Ruby's data agrees.

use super::super::game::SiteGame;
use super::super::view as v1;
use crate::entities::EntityId;
use crate::rounds::Round;
use crate::steps::StepKind;
use crate::title::CorpType;
use serde_json::{json, Map, Value};

/// Whether a train type is Ruby's `num: 'unlimited'` (1867's 8, 2+2 and 5+5E: one at a time in the depot,
/// `Depot#clone_unlimited`).
pub fn unlimited(g: &SiteGame, name: &str) -> bool {
    let title = g.game.title_def();
    title.trains().iter().any(|t| t.name == name && title.num_trains(t, &g.game.optional_rules) == u32::MAX)
}

fn train_card(t: &crate::entities::Train) -> Value {
    json!({"id": t.id, "name": t.name})
}

/// The depot's upcoming trains (Ruby `depot.upcoming`): an unlimited type shows one train at a time.
pub fn upcoming(g: &SiteGame) -> Vec<Value> {
    let mut out = Vec::new();
    let mut seen: Vec<&str> = Vec::new();
    for t in &g.game.depot.trains {
        if unlimited(g, &t.name) {
            if seen.contains(&t.name.as_str()) {
                continue;
            }
            seen.push(&t.name);
        }
        out.push(train_card(t));
    }
    out
}

/// Ruby's PHASES `train_limit` hashes (g_1867/game.rb), by phase: the snapshot's `trainLimits` (the engine's phase
/// data carries the two numbers, not which kinds Ruby lists).
pub fn phase_limits(phase: &str) -> &'static [(&'static str, u8)] {
    match phase {
        "2" => &[("minor", 2)],
        "3" => &[("minor", 2), ("major", 4)],
        "4" | "5" => &[("minor", 1), ("major", 3)],
        "6" | "7" => &[("minor", 1), ("major", 2)],
        _ => &[("major", 2)],
    }
}

/// Ruby `train_limit(corporation)`: the phase's limit for the company's kind (0 where the phase names none).
pub fn train_limit(g: &SiteGame, ci: usize) -> u8 {
    let kind = match g.game.corporations[ci].corp_type {
        CorpType::Minor => "minor",
        _ => "major",
    };
    phase_limits(&g.game.phase.name).iter().find(|(k, _)| *k == kind).map_or(0, |x| x.1)
}

/// The round (adapter.ts `roundInfo`): 1867's auction keeps its own name; an operating or merger round says its
/// place in the set ("Merger Round 5.2 of 2", Ruby `round_description`).
pub fn round_info(g: &SiteGame) -> Value {
    let turn = g.game.turn;
    match &g.game.round {
        Round::Auction(_) => json!({"name": "Auction Round", "short": "PR", "label": "Auction Round", "turn": turn, "num": 1}),
        Round::Stock(_) => {
            json!({"name": "Stock Round", "short": "SR", "label": format!("Stock Round {turn}"), "turn": turn, "num": 1})
        }
        Round::Operating(s) => json!({"name": "Operating Round", "short": "OR",
            "label": format!("Operating Round {turn}.{} of {}", s.round_num, s.total_ors), "turn": turn, "num": s.round_num}),
        Round::Merger(s) => json!({"name": "Merger Round", "short": "MR",
            "label": format!("Merger Round {turn}.{} of {}", s.round_num, s.total_ors), "turn": turn, "num": s.round_num}),
    }
}

/// Ruby's active step's `description` (1867's steps).
pub fn step_label(game: &mut crate::game::BaseGame) -> Option<String> {
    let (kind, _) = super::active_step(game)?;
    Some(match kind {
        StepKind::SingleItemAuction => "Bid on Companies".into(),
        // SELL_BUY_ORDER :sell_buy
        StepKind::BuySellParShares => "Sell then Buy Shares".into(),
        StepKind::HomeToken => {
            // Engine::Step::HomeToken#description: the pending token is the company's first, or another
            let first = match &game.round {
                Round::Stock(s) => s.pending_home_tokens.first().map(|(sym, _)| sym.clone()),
                _ => None,
            };
            let placed = first.as_ref().and_then(|sym| game.corp_idx.get(sym.as_str())).map_or(0, |&ci| {
                game.corporations[ci].tokens.iter().filter(|t| t.used).count()
            });
            if placed == 0 { "Place Home Token".into() } else { "Place Token".into() }
        }
        StepKind::MajorTrainless => "Choose if Major is nationalized".into(),
        StepKind::DiscardTrain => "Discard Trains".into(),
        StepKind::BuyCompany | StepKind::BuyCompanyPreloan => "Buy Companies".into(),
        StepKind::RedeemShares => "Redeem Shares".into(),
        StepKind::Track => {
            // Engine::Step::Track#description over the turn's next lay (TILE_LAYS: the first lays or upgrades; the
            // second ($20) lays, and upgrades unless the first was an upgrade)
            match &game.round {
                Round::Operating(s) if s.num_laid_track == 0 => "Lay/Upgrade Track".into(),
                Round::Operating(s) if s.num_laid_track == 1 && !s.upgraded_this_turn => "Lay/Upgrade Track".into(),
                _ => "Lay Track".into(),
            }
        }
        StepKind::Token => "Place a Token".into(),
        StepKind::Route => "Run Routes".into(),
        StepKind::Dividend => "Pay or Withhold Dividends".into(),
        StepKind::BuyTrain => "Buy Trains".into(),
        StepKind::Merge => match &game.round {
            Round::Merger(s) if s.converting || s.merge_major => "Choose Major Corporation".into(),
            Round::Merger(s) if !s.merging.is_empty() => "Merge Minor Corporations".into(),
            _ => "Convert or Merge Minor Corporation".into(),
        },
        StepKind::PostMergerShares => "Buy Shares Post Merge".into(),
        StepKind::ReduceTokens => "Choose tokens to remove".into(),
        k => crate::steps::step_description(k).into(),
    })
}

/// The round's turn order and the index of the one acting (adapter.ts: `round.entities`, `round.entity_index`).
pub fn round_order(g: &SiteGame) -> (Vec<String>, i64) {
    let game = &g.game;
    match &game.round {
        Round::Auction(s) => {
            let list: Vec<String> = s.player_order.iter().map(|p| format!("p:{p}")).collect();
            let idx = s.entity_index as i64;
            let idx = if !list.is_empty() && idx >= list.len() as i64 { list.len() as i64 - 1 } else { idx };
            (list, idx)
        }
        Round::Merger(s) => {
            // G1867::Round::Merger#entities: the players while a new major's shares are dealt, else the minors
            if s.converted.is_some() {
                let order = v1::player_order(g);
                (order.iter().map(|p| format!("p:{p}")).collect(), s.entity_index as i64)
            } else {
                (s.entities.iter().map(|c| format!("c:{c}")).collect(), s.entity_index as i64)
            }
        }
        Round::Stock(_) => {
            let (list, idx) = v1::round_order(g);
            // a pending home station keeps the round on the player who started the company (`goto_entity!`)
            let mut bg = g.game.clone_for_search();
            if let Some((StepKind::HomeToken, crate::steps::StepEntity::Player(p))) = super::active_step(&mut bg) {
                if let Some(i) = list.iter().position(|x| *x == format!("p:{p}")) {
                    return (list, i as i64);
                }
            }
            (list, idx)
        }
        _ => v1::round_order(g),
    }
}

/// The snapshot (adapter.ts `snapshot`) of a 1867 game.
pub fn snapshot(g: &SiteGame, action_id: i64) -> Value {
    let finished = g.game.finished;
    // (the step machine's reads take the game mutably: a copy answers them)
    let mut bg = g.game.clone_for_search();
    let label = if finished { None } else { step_label(&mut bg) };
    let current = if finished { None } else { super::current_holder(&mut bg) };
    let active = if finished { Vec::new() } else { super::active_players(&mut bg) };
    let members = super::ruby_corporations(&g.game);
    let auction = auction_view(g, &mut bg);
    let game = &g.game;
    let results = game.calculate_results();
    let order = v1::player_order(g);
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
        m.insert("companies".into(), json!(v1::owned_companies(g, &pe)));
        m.insert("bankrupt".into(), json!(p.bankrupt));
        players.push(m);
    }

    let first_limit = members.first().map_or(0, |&ci| train_limit(g, ci));
    let (is_or, or_order, cur_pos) = match &game.round {
        Round::Operating(s) => {
            let pos = current.as_deref().and_then(|h| h.strip_prefix("c:")).and_then(|c| s.operating_order.iter().position(|x| x == c));
            (true, s.operating_order.clone(), pos)
        }
        _ => (false, Vec::new(), None),
    };
    let mut corps = Vec::new();
    for &ci in &members {
        let c = &game.corporations[ci];
        let president = c.owner_id.player_id().filter(|_| c.ipo_price.is_some() || c.president_id().is_some()).or(c.president_id());
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
        let held: u32 = c.shares.iter().filter(|s| s.owner.is_player() || s.owner.is_market()).map(|s| s.percent as u32).sum();
        let ipo_pct: u32 = 100u32.saturating_sub(held);
        let pool_pct: u32 = c.shares.iter().filter(|s| s.owner.is_market()).map(|s| s.percent as u32).sum();
        let to_float = if c.floated { 0 } else { (ipo_pct as i64 - (100 - c.float_percent as i64)).max(0) };
        let operated = is_or && cur_pos.is_some_and(|cp| matches!(or_order.iter().position(|s| s == &c.sym), Some(i) if i < cp));
        let mut cm = Map::new();
        cm.insert("sym".into(), json!(c.sym));
        cm.insert("cash".into(), json!(c.cash));
        cm.insert("price".into(), json!(sp.map(|s| s.price)));
        cm.insert("cell".into(), sp.map_or(Value::Null, |s| json!([s.row, s.column])));
        cm.insert("stack".into(), json!(stack));
        // Ruby Corporation#par_price: the share price (always_market_price)
        cm.insert("par".into(), json!(sp.map(|s| s.price)));
        cm.insert("ipoed".into(), json!(c.ipo_price.is_some()));
        cm.insert("floated".into(), json!(c.floated));
        cm.insert("toFloat".into(), json!(to_float));
        cm.insert("closed".into(), json!(c.closed));
        cm.insert("ipoPct".into(), json!(ipo_pct));
        cm.insert("poolPct".into(), json!(pool_pct));
        cm.insert("president".into(), json!(president));
        cm.insert("trains".into(), json!(c.trains.iter().map(train_card).collect::<Vec<_>>()));
        cm.insert("tokensLeft".into(), json!(unused.len()));
        cm.insert("nextTokenPrice".into(), json!(unused.first().map(|t| t.price)));
        cm.insert("companies".into(), json!(v1::owned_companies(g, &EntityId::corporation(&c.sym))));
        cm.insert("operated".into(), json!(operated));
        cm.insert("lastRun".into(), v1::last_run(g, &c.sym));
        if c.loans > 0 {
            cm.insert("loans".into(), json!(c.loans));
            cm.insert("loanMax".into(), json!(if c.corp_type == CorpType::Minor { 2 } else { 5 }));
            // Ruby interest_owed: $5 a loan the last operating round counted (@interest outlives the round)
            cm.insert("interest".into(), json!(5 * g.interest_due(&c.sym)));
        }
        let tl = train_limit(g, ci);
        if tl != first_limit {
            cm.insert("trainLimit".into(), json!(tl));
        }
        corps.push(Value::Object(cm));
    }

    let companies: Vec<Value> = game
        .companies
        .iter()
        .map(|m| {
            let owner = if m.closed {
                Value::Null
            } else if let Some(p) = m.owner.player_id() {
                json!(format!("p:{p}"))
            } else if m.owner.is_corporation() {
                m.owner.corp_sym().map_or(Value::Null, |s| json!(format!("c:{s}")))
            } else {
                Value::Null
            };
            json!({"sym": m.sym, "owner": owner, "closed": m.closed})
        })
        .collect();

    let hexes = v1::hex_states(g);

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
    let (op_order, waiting) = operating_order(g, &members);

    let mut out = Map::new();
    if let Some(p) = v1::programs(g) {
        out.insert("programs".into(), p);
    }
    if let Some(s) = v1::sold(g) {
        out.insert("sold".into(), s);
    }
    out.insert("actionId".into(), json!(action_id));
    out.insert("round".into(), round_info(g));
    out.insert("stepLabel".into(), json!(label));
    out.insert("phase".into(), json!(game.phase.name));
    out.insert("trainLimit".into(), json!(first_limit));
    out.insert(
        "trainLimits".into(),
        json!(phase_limits(&game.phase.name).iter().map(|(k, l)| json!({"kind": k, "limit": l})).collect::<Vec<_>>()),
    );
    out.insert("bank".into(), json!(game.bank.cash));
    out.insert("activePlayers".into(), json!(active));
    out.insert("current".into(), json!(current));
    out.insert("priority".into(), json!(v1::priority(g)));
    out.insert("players".into(), Value::Array(players.into_iter().map(Value::Object).collect()));
    out.insert("corps".into(), Value::Array(corps));
    out.insert("companies".into(), Value::Array(companies));
    out.insert("hexes".into(), hexes);
    out.insert(
        "depot".into(),
        json!({"upcoming": upcoming(g), "discarded": game.depot.discarded.iter().map(train_card).collect::<Vec<_>>(),
               "available": available}),
    );
    out.insert("bids".into(), auction.0);
    out.insert("auction".into(), auction.1);
    if let Some(a) = auction.2 {
        out.insert("auctioning".into(), json!(a));
    }
    out.insert("finished".into(), json!(finished));
    out.insert("result".into(), result);
    out.insert("endReason".into(), end_reason);
    out.insert("ending".into(), v1::ending(g));
    out.insert("order".into(), json!(order_list));
    out.insert("orderIndex".into(), json!(order_index));
    out.insert("opOrder".into(), json!(op_order));
    out.insert("waiting".into(), json!(waiting));
    Value::Object(out)
}

/// The snapshot's auction parts (adapter.ts: the active step's `bids`, its `companies` with `min_bid`, and
/// `auctionedNow`): the opening auction's lots and the lot up; a stock round's minor auction's bids.
fn auction_view(g: &SiteGame, bg: &mut crate::game::BaseGame) -> (Value, Value, Option<String>) {
    let mut bids = Vec::new();
    let mut lots = Vec::new();
    let mut up = None;
    let active = super::active_step(bg).map(|x| x.0);
    match &g.game.round {
        Round::Auction(s) if active == Some(StepKind::SingleItemAuction) => {
            if let Some(ci) = s.auctioning {
                let sym = g.game.companies[ci].sym.clone();
                let mut list: Vec<(u32, i32)> = s.bids.get(&ci).map(|b| b.iter().map(|x| (x.player_id, x.price)).collect()).unwrap_or_default();
                list.sort_by_key(|b| b.1);
                for (pid, price) in list {
                    bids.push(json!({"company": sym, "player": pid, "price": price}));
                }
                up = Some(sym);
            }
            for &ci in &s.remaining_companies {
                lots.push(json!({"sym": g.game.companies[ci].sym, "minBid": g.game.sa_min_bid(s, ci)}));
            }
        }
        Round::Stock(s) if active == Some(StepKind::BuySellParShares) => {
            if let Some(a) = &s.bid_auction {
                let mut list: Vec<(u32, i32)> = a.bids.iter().map(|b| (b.player_id, b.price)).collect();
                list.sort_by_key(|b| b.1);
                for (pid, price) in list {
                    bids.push(json!({"company": a.corporation_sym, "player": pid, "price": price}));
                }
                up = Some(a.corporation_sym.clone());
            }
        }
        _ => {}
    }
    (json!(bids), json!(lots), up)
}

/// The Order column (adapter.ts `operatingOrder`): the operating round's own order, else Ruby's `operating_order`
/// (the floated minors, then the floated majors, each by price); then the started companies that don't operate.
fn operating_order(g: &SiteGame, members: &[usize]) -> (Vec<String>, Vec<String>) {
    let game = &g.game;
    let op: Vec<String> = match &game.round {
        Round::Operating(s) => s.operating_order.clone(),
        _ => game.compute_operating_order(),
    };
    let mut keyed: Vec<((i32, i32, u8, usize, String), String)> = Vec::new();
    for &ci in members {
        let c = &game.corporations[ci];
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

/// The engine part of StaticInfo for 1867 (types.ts; adapter.ts `staticInfo`): what depends on the players and the
/// title's rules data. Names, colours, logos, descriptions, hex geometry and texts come from the title pack.
pub fn static_info(g: &SiteGame) -> Value {
    let game = &g.game;
    let title = game.title_def();
    let players: Vec<Value> = g
        .players
        .iter()
        .enumerate()
        .map(|(i, (id, name, bot))| json!({"id": id, "name": name, "bot": bot, "seat": i}))
        .collect();
    let mut kinds = Map::new();
    let corporations: Vec<Value> = title
        .corporations()
        .iter()
        .filter(|c| c.corp_type != CorpType::National)
        .map(|c| {
            let minor = c.corp_type == CorpType::Minor;
            if minor {
                kinds.insert(c.sym.to_string(), json!("minor"));
            }
            let mut m = json!({"sym": c.sym, "name": c.name, "home": null, "tokenCount": c.token_prices.len(),
                "floatPercent": c.float_percent, "capitalization": "incremental",
                "sharePercent": if minor { 50 } else { 10 }});
            if minor {
                m["kind"] = json!("minor");
                m["presidentPercent"] = json!(100);
                m["shares"] = json!(2);
            }
            m
        })
        .collect();
    let companies: Vec<Value> = title
        .companies()
        .iter()
        .map(|c| {
            let hidden = c.sym == "3";
            let mut m = json!({"sym": c.sym, "name": c.name, "value": c.value, "revenue": c.revenue,
                "blocks": if hidden { json!(["M13"]) } else { json!([]) }, "comesWith": []});
            if hidden {
                m["hidden"] = json!(true);
            }
            m
        })
        .collect();
    let market: Vec<Value> = game
        .stock_market
        .grid
        .iter()
        .map(|row| {
            json!(row
                .iter()
                .map(|c| c.as_ref().map_or(Value::Null, |c| json!({"price": c.price,
                    "types": c.types.iter().filter(|t| t.as_str() != "normal").collect::<Vec<_>>()})))
                .collect::<Vec<_>>())
        })
        .collect();
    let trains: Vec<Value> = title
        .trains()
        .iter()
        .map(|t| {
            let unl = title.num_trains(t, &game.optional_rules) == u32::MAX;
            json!({"name": t.name, "distance": t.name, "price": t.price, "rustsOn": t.rusts_on,
                   "num": if unl { 1 } else { title.num_trains(t, &game.optional_rules) }, "unlimited": unl})
        })
        .collect();
    let phases: Vec<Value> =
        title.phases().iter().map(|p| json!({"name": p.name, "trainLimit": 0, "tiles": p.tiles})).collect();
    let hexes: Vec<Value> = game.hexes.iter().map(|h| json!({"id": h.id})).collect();
    json!({
        "gameId": g.meta.get("id").cloned().unwrap_or(Value::Null),
        "title": "1867",
        "description": g.meta.get("description").cloned().unwrap_or(json!("")),
        "status": g.meta.get("status").cloned().unwrap_or(json!("")),
        "layout": "flat",
        "players": players,
        "corporations": corporations,
        "companies": companies,
        "market": market,
        "marketKind": "1d",
        "hexes": hexes,
        "trainTypes": trains,
        "phases": phases,
        "bankStart": title.bank_cash() - g.players.len() as i32 * title.starting_cash(g.players.len() as u8),
        "certLimit": game.cert_limit,
        "currency": "$%s",
        "kinds": kinds,
        "packKeys": ["corporations.name", "corporations.color", "corporations.textColor", "corporations.logo",
                     "companies.name", "companies.desc", "hexes.x", "hexes.y", "hexes.location", "zones", "description"],
    })
}
