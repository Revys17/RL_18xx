//! The 1822 family's move menu as facts (RawMenu, docs/SITE_ENGINE_RS_1822.md §3.3): what the player to act may do
//! now, as complete site actions with the numbers the hands need, read off the legal-move generator's moves (Ruby's
//! whole set, `CrossPlayerPolicy::RULES`) with the site's own-only rule applied by the engine's own predicate
//! (`g1822_cross_player_excluded`: no train from another player's company, no private another player owns, no minor
//! another player owns; bank lots stay) and **no fallback** to Ruby's set when nothing is left (the engine's
//! generator falls back; the site says why instead: `othersLeftOut`, `deadEnd`).
//!
//! Shared facts keep step 1's shapes (`buy`, `sell`, `par`, `turn`, `tileHexes`, `abilities`, `tokens`, `routes`,
//! `dividend`, `trains`, `trainBuy`, `discard`, `redeem`, `pass`, `programs`); the family's own are new facts the
//! client words (`words.ts`): `bidBoxes`, `taxHaven`, `debt`, `choices`, `abilityChoices`, `acquire`, `hexToken`,
//! `routeChoices`, `extraDividend`, `acquireMinor`, `sawmill`, `issue`, `deadEnd`, `othersLeftOut`, and every offered
//! move in `moves` (each with its price range), so no decision is ever without an action.

use super::super::game::SiteGame;
use super::super::menu::{company_name, player_name};
use crate::actions::Action;
use crate::entities::EntityId;
use crate::g1822::legal::Move;
use crate::g1822::{AcquireState, Ent, RoundKind, Step};
use crate::title::CrossPlayerPolicy;
use serde_json::{json, Map, Value};

fn atype(a: &Value) -> &str {
    a.get("type").and_then(Value::as_str).unwrap_or("")
}

fn sval(v: Option<&Value>) -> String {
    match v {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Null) | None => String::new(),
        Some(x) => x.to_string(),
    }
}

/// One offered move: the generator's move, its wire form in the site's ids, the price range.
pub struct Offer {
    pub m: Move,
    pub action: Value,
}

impl Offer {
    fn range(&self) -> Value {
        self.m.price.map_or(Value::Null, |r| json!({"min": r.lo, "max": r.hi, "step": r.step}))
    }
}

/// The moves now, split by the site's own-only rule: (offered, left out).
pub fn offers(g: &mut SiteGame) -> (Vec<Offer>, Vec<Offer>) {
    let moves = g.game.g1822_legal_moves_with(CrossPlayerPolicy::RULES);
    let mut keep = Vec::new();
    let mut out = Vec::new();
    for m in moves {
        let w = g.game.g1822_move_wire(&m);
        let o = Offer { action: super::to_site(g, &w), m };
        if g.game.g1822_cross_player_excluded(&o.m, CrossPlayerPolicy::RESTRICTED) {
            out.push(o);
        } else {
            keep.push(o);
        }
    }
    (keep, out)
}

fn of<'a>(list: &'a [Offer], t: &'a str) -> impl Iterator<Item = &'a Offer> + 'a {
    list.iter().filter(move |o| atype(&o.action) == t)
}

fn corp_name(g: &SiteGame, sym: &str) -> String {
    super::view::corp_name(g, sym)
}

/// A share's price for a holder (Ruby `Share#price`): the market price for its percent.
fn share_price(g: &SiteGame, corp: &str, percent: u8) -> i32 {
    let Some(&ci) = g.game.corp_idx.get(corp) else { return 0 };
    g.game.g1822_share_price(&g.game.corporations[ci], percent)
}

/// The move applied on a copy (previews: exact by construction).
fn after(g: &SiteGame, o: &Offer) -> Option<crate::game::BaseGame> {
    let mut c = g.game.clone_for_search_without_log();
    c.g1822_apply_move(&o.m).ok()?;
    Some(c)
}

fn player_json(g: &SiteGame, seat: u32) -> Value {
    let site = g.site_id(seat);
    json!({"id": site, "name": player_name(g, site), "cash": g.game.players.iter().find(|p| p.id == seat).map_or(0, |p| p.cash)})
}

pub fn menu(g: &mut SiteGame, pid: u32) -> Value {
    if g.game.finished || !super::active_players(g).contains(&pid) {
        return Value::Null;
    }
    let Some((bi, step, ent)) = super::active(&g.game) else { return Value::Null };
    let seat = g.seat(pid);
    let (all, left_out) = offers(g);
    // the player's own moves (the step's entity and the companies acting beside it are all theirs)
    let mine: Vec<Offer> = all
        .into_iter()
        .filter(|o| {
            let e = g.game.g1822_entity(&sval(super::to_engine(g, &o.action).get("entity")));
            super::acting_for(&g.game, &e) == Some(seat)
        })
        .collect();
    let (actor, ent_json) = match &ent {
        Ent::Player(p) => {
            let s = g.site_id(*p);
            (json!({"type": "player", "id": s, "name": player_name(g, s)}), json!({"entity": s, "entity_type": "player"}))
        }
        Ent::Corp(c) => (json!({"type": "corporation", "id": c, "name": c}), json!({"entity": c, "entity_type": "corporation"})),
        Ent::Company(c) => (json!({"type": "company", "id": c, "name": company_name(g, c)}), json!({"entity": c, "entity_type": "company"})),
    };
    if mine.is_empty() && !matches!(&ent, Ent::Player(p) if *p == seat) && super::acting_for(&g.game, &ent) != Some(seat) {
        // (a crowded company's president whose company isn't the first to discard: Ruby's view lets them, the
        // authority doesn't; nothing to offer)
        return Value::Null;
    }
    let mut m = Map::new();
    m.insert("actor".into(), actor);
    m.insert("description".into(), json!(super::view::step_label(g)));
    m.insert("roundName".into(), json!(super::round_name(&g.game)));
    m.insert("passDescription".into(), json!(super::view::pass_description(g)));
    m.insert("step".into(), json!(super::step_class(step)));
    m.insert("types".into(), json!(super::ruby_actions(&g.game)));
    m.insert("unsupported".into(), json!([]));
    if g.game.rnd().kind == RoundKind::Stock && matches!(ent, Ent::Player(_)) {
        m.insert("programs".into(), json!(["program_buy_shares", "program_share_pass"]));
    }
    // every move offered, with its range (the plain list: nothing is ever without an action)
    m.insert("moves".into(), json!(mine.iter().map(|o| json!({"type": atype(&o.action), "action": o.action, "range": o.range()})).collect::<Vec<_>>()));
    if mine.is_empty() {
        // (the engine's generator would fall back to Ruby's set here; the site says why instead: a Ruby dead end's
        // words, or that only deals with other players' companies are left)
        let why = super::act::dead_end(g).unwrap_or_else(|| {
            if left_out.is_empty() {
                "No move is open to you here".into()
            } else {
                "The only moves left are deals with other players' companies, which this site doesn't offer".into()
            }
        });
        m.insert("deadEnd".into(), json!(why));
    }
    let lo: Vec<Value> = left_out
        .iter()
        .map(|o| {
            let (kind, target, owner) = match &o.m.action {
                Action::BuyTrain { from, train_name, .. } => ("train", train_name.clone(), g.game.g1822_corp_president(from)),
                Action::AcquireCompany { company_sym, .. } => ("private", company_sym.clone(), g.game.co_owner(company_sym).player_id()),
                Action::Merge { corporation_sym, .. } => ("minor", corporation_sym.clone(), g.game.g1822_corp_president(corporation_sym)),
                _ => ("other", String::new(), None),
            };
            json!({"kind": kind, "sym": target, "owner": owner.map(|p| g.site_id(p)), "ownerName": owner.map(|p| player_name(g, g.site_id(p)))})
        })
        .collect();
    if !lo.is_empty() {
        m.insert("othersLeftOut".into(), json!(lo));
    }
    if let Some(p) = of(&mine, "pass").find(|o| sval(o.action.get("entity")) == sval(ent_json.get("entity"))).or_else(|| of(&mine, "pass").next()) {
        m.insert("pass".into(), json!({"action": p.action}));
    }
    match step {
        Step::BuySellParShares => stock_parts(g, seat, &mine, &mut m),
        Step::Choose => {
            choices(g, "double_cash", g.game.fam().company_double_cash, &mine, &mut m);
        }
        _ => operating_parts(g, bi, step, &ent, &mine, &mut m),
    }
    // the debt of a player who still owes some (paid off in a stock round)
    Value::Object(m)
}

/// The bid boxes (G1822's BidboxAuction in the stock round): each lot with its high bid, the player's own, the price
/// range and why a lot can't be bid on.
fn bid_boxes(g: &SiteGame, seat: u32, mine: &[Offer]) -> Value {
    let game = &g.game;
    let st = game.g();
    let cash = game.players.iter().find(|p| p.id == seat).map_or(0, |p| p.cash);
    let committed = game.g1822_committed_cash(seat);
    let left = super::view::tokens_left(g, seat);
    let total = (game.fam().bidding_tokens)(game.players.len());
    let per_turn = game.fam().bidding_tokens_per_action;
    let debt = game.player_debt(seat);
    let inc = game.fam().min_bid_increment;
    let lot = |sym: &str| -> Value {
        let ci = game.company_idx.get(sym).copied();
        let value = ci.map_or(0, |i| game.companies[i].value);
        let high = game.g1822_highest_bid(sym);
        let my = st.round.bids.get(sym).and_then(|l| l.iter().rev().find(|b| b.player == seat)).map(|b| b.price);
        let min = match &high {
            Some(h) => h.price + inc,
            None => ci.map_or(0, |i| game.companies[i].min_bid()),
        };
        let offer = of(mine, "bid").find(|o| sval(o.action.get("company")) == sym);
        let why: Value = if offer.is_some() {
            Value::Null
        } else if high.as_ref().is_some_and(|h| h.player == seat) {
            json!("high")
        } else if debt > 0 {
            json!("debt")
        } else if game.rnd().bid_actions >= per_turn {
            json!("turn")
        } else if left <= 0 && my.is_none() {
            json!("tokens")
        } else if min > cash - committed {
            json!("cash")
        } else if game.g1822_num_certs_with_bids(seat) >= game.cert_limit as i32 {
            json!("certs")
        } else {
            json!("other")
        };
        let kind = match sym.chars().next() {
            Some('M') => "minor",
            Some('C') => "concession",
            _ => "private",
        };
        json!({"sym": sym, "kind": kind, "name": company_name(g, sym), "value": value,
               "minBid": offer.and_then(|o| o.m.price.map(|r| r.lo)).unwrap_or(min),
               "maxBid": offer.and_then(|o| o.m.price.map(|r| r.hi)).unwrap_or(cash - committed),
               "high": high.map(|h| json!({"player": g.site_id(h.player), "name": player_name(g, g.site_id(h.player)), "price": h.price})),
               "mine": my, "canBid": offer.is_some(), "why": why,
               "action": offer.map(|o| o.action.clone()), "range": offer.map(|o| o.range())})
    };
    let mut boxes = vec![
        json!({"kind": "minor", "lots": game.bidbox_minors().iter().map(|s| lot(s)).collect::<Vec<_>>()}),
        json!({"kind": "concession", "lots": game.bidbox_concessions().iter().map(|s| lot(s)).collect::<Vec<_>>()}),
    ];
    let privates = game.bidbox_privates();
    let plus_boxes = game.g1822_plus() && !game.optional_rules.iter().any(|r| r == "plus_expansion_single_stack");
    if plus_boxes {
        for (i, cat) in ["train", "track", "other"].iter().enumerate() {
            let lots: Vec<Value> = privates.iter().filter(|p| game.fam().plus_expansion_bidbox[i].contains(&p.as_str())).map(|s| lot(s)).collect();
            boxes.push(json!({"kind": "private", "category": cat, "lots": lots}));
        }
    } else {
        boxes.push(json!({"kind": "private", "lots": privates.iter().map(|s| lot(s)).collect::<Vec<_>>()}));
    }
    json!({"boxes": boxes, "tokens": {"left": left, "total": total}, "committed": committed, "cash": cash,
           "available": cash - committed, "bidsThisTurn": game.rnd().bid_actions, "perTurn": per_turn, "step": inc})
}

fn stock_parts(g: &mut SiteGame, seat: u32, mine: &[Offer], m: &mut Map<String, Value>) {
    let game = &g.game;
    let cash = game.players.iter().find(|p| p.id == seat).map_or(0, |p| p.cash);
    if of(mine, "bid").next().is_some() || super::ruby_actions(game).iter().any(|t| t == "bid") {
        m.insert("bidBoxes".into(), bid_boxes(g, seat, mine));
    }
    // buys (majors' shares: the treasury, called IPO, or the market)
    let buys: Vec<Value> = of(mine, "buy_shares")
        .filter(|o| o.action.get("entity_type").and_then(Value::as_str) == Some("player"))
        .filter_map(|o| {
            let Action::BuyShares { corporation_sym, percent, .. } = &o.m.action else { return None };
            let ci = *game.corp_idx.get(corporation_sym.as_str())?;
            let c = &game.corporations[ci];
            let id = o.m.shares.first()?;
            let idx: usize = id.rsplit_once('_')?.1.parse().ok()?;
            let market = c.shares.get(idx).is_some_and(|s| s.owner.is_market());
            let price = o.m.share_price.unwrap_or_else(|| share_price(g, corporation_sym, *percent));
            Some(json!({"corp": corporation_sym, "kind": if market { "market" } else { "IPO" }, "percent": percent, "price": price,
                        "floats": false, "action": o.action}))
        })
        .collect();
    if !buys.is_empty() {
        m.insert("buy".into(), json!(buys));
    }
    let sells = sell_list(g, mine, true);
    if !sells.is_empty() {
        m.insert("sell".into(), json!(sells));
    }
    // pars: a concession's conversion (its major at a par price; the concession counts towards the president's
    // certificate) or a major parred from the market (1822CA from phase 5)
    let mut pars: Vec<(String, Vec<Value>)> = Vec::new();
    for o in of(mine, "par") {
        let Action::Par { corporation_sym, share_price, .. } = &o.m.action else { continue };
        match pars.iter_mut().find(|p| p.0 == *corporation_sym) {
            Some(p) => p.1.push(json!({"price": share_price, "action": o.action})),
            None => pars.push((corporation_sym.clone(), vec![json!({"price": share_price, "action": o.action})])),
        }
    }
    if !pars.is_empty() {
        let list: Vec<Value> = pars
            .into_iter()
            .map(|(corp, prices)| {
                let ci = game.corp_idx[corp.as_str()];
                let c = &game.corporations[ci];
                let via = game.fam().concessions.iter().find(|(_, major)| *major == corp).map(|(co, _)| co.to_string())
                    .filter(|co| game.co_owner(co).player_id() == Some(seat));
                let value = via.as_ref().and_then(|co| game.company_idx.get(co.as_str()).map(|&i| game.companies[i].value));
                let pres = c.shares.iter().find(|s| s.president).map_or(20, |s| s.percent);
                json!({"corp": corp, "via": via, "exchangeValue": value, "cash": cash, "floatPercent": c.float_percent,
                       "presPercent": pres, "sharePercent": c.share_unit_percent, "prices": prices})
            })
            .collect();
        m.insert("par".into(), json!(list));
    }
    // the turn so far (Ruby's round.current_actions)
    let mut bought = Vec::new();
    let mut sold = Vec::new();
    for a in &g.turn_actions {
        let pid_site = g.site_id(seat);
        if a.get("entity").and_then(Value::as_u64) != Some(pid_site as u64) {
            continue;
        }
        let corp = || a.get("corporation").and_then(Value::as_str).map(str::to_string).or_else(|| {
            a.get("shares").and_then(Value::as_array).and_then(|v| v.first()).and_then(Value::as_str).and_then(|x| x.rsplit_once('_').map(|y| y.0.to_string()))
        });
        match atype(a) {
            "buy_shares" => bought.push(json!({"corp": corp(), "what": "share"})),
            "par" => bought.push(json!({"corp": corp(), "what": "par"})),
            "sell_shares" => sold.push(json!({"corp": corp(), "pct": a.get("percent")})),
            "bid" => bought.push(json!({"corp": a.get("company"), "what": "bid"})),
            _ => {}
        }
    }
    let must_sell = super::ruby_actions(game) == ["sell_shares"];
    m.insert(
        "turn".into(),
        json!({"bought": bought, "sold": sold, "mustSell": if must_sell { json!({"kind": "certs"}) } else { Value::Null },
               "certs": game.g1822_num_certs(seat), "certsWithBids": game.g1822_num_certs_with_bids(seat), "certLimit": game.cert_limit,
               "didBuy": !bought.is_empty(), "multiOnlyMarket": false, "bids": game.rnd().bid_actions}),
    );
    // the tax haven (P16; 1822CA P11): a share into the haven, from the treasury or the market, in its owner's turn
    let osth = game.fam().company_osth;
    let th: Vec<Value> = of(mine, "choose_ability")
        .filter(|o| sval(o.action.get("entity")) == osth)
        .map(|o| {
            let choice = sval(o.action.get("choice"));
            let (corp, from) = choice.rsplit_once('_').map(|(c, f)| (c.to_string(), f.to_string())).unwrap_or((choice.clone(), String::new()));
            let price = share_price(g, &corp, game.corp_idx.get(corp.as_str()).map_or(10, |&ci| game.corporations[ci].share_unit_percent));
            json!({"corp": corp, "from": if from == "market" { "market" } else { "IPO" }, "price": price, "choice": choice, "action": o.action})
        })
        .collect();
    if !th.is_empty() {
        let multiple = game.g1822_tax_haven_multiple();
        let haven = multiple && game.g1822_tax_haven_value() > 0;
        m.insert(
            "taxHaven".into(),
            json!({"company": osth, "name": company_name(g, osth), "spender": if haven { "haven" } else { "owner" },
                   "cash": if haven { game.g().tax_haven_cash } else { cash }, "multiple": multiple, "options": th}),
        );
    }
    debt_part(g, mine, m);
    ability_choices(g, mine, m);
}

/// Player debt (EBUY's loan, 50% interest at each stock round's end): what is owed and its payoff, whole or in part.
fn debt_part(g: &SiteGame, mine: &[Offer], m: &mut Map<String, Value>) {
    let Some(o) = of(mine, "payoff_player_debt").next().or_else(|| of(mine, "payoff_player_debt_partial").next()) else { return };
    let seat = sval(super::to_engine(g, &o.action).get("entity")).parse::<u32>().unwrap_or(0);
    let full = of(mine, "payoff_player_debt").next().map(|o| json!({"action": o.action}));
    let part = of(mine, "payoff_player_debt_partial").next().map(|o| json!({"min": o.m.price.map_or(1, |r| r.lo),
        "max": o.m.price.map_or(1, |r| r.hi), "step": o.m.price.map_or(1, |r| r.step), "action": o.action}));
    m.insert("debt".into(), json!({"amount": g.game.player_debt(seat), "interest": g.game.fam().player_loan_interest_rate,
                                   "payoff": full, "partial": part}));
}

/// The president's or a player's sales with their previews (Ruby's sell bundles).
fn sell_list(g: &SiteGame, mine: &[Offer], by_player: bool) -> Vec<Value> {
    let game = &g.game;
    of(mine, "sell_shares")
        .filter(|o| (o.action.get("entity_type").and_then(Value::as_str) == Some("player")) == by_player)
        .filter_map(|o| {
            let Action::SellShares { corporation_sym, percent, entity_id, .. } = &o.m.action else { return None };
            let ci = *game.corp_idx.get(corporation_sym.as_str())?;
            let c = &game.corporations[ci];
            let sp = c.share_price.as_ref()?;
            let price = o.m.share_price.map(|p| p * (*percent as i32) / c.share_unit_percent.max(1) as i32).unwrap_or_else(|| share_price(g, corporation_sym, *percent));
            let sale = after(g, o).map(|a| {
                let ac = &a.corporations[ci];
                let seller = EntityId(if by_player { format!("player:{}", entity_id) } else { String::new() });
                let left = ac.percent_owned_by(&EntityId::player(entity_id.parse().unwrap_or(0)));
                let pres_before = game.g1822_corp_president(corporation_sym);
                let pres_after = a.g1822_corp_president(corporation_sym);
                let pres = if pres_before.map(|p| p.to_string()) != Some(entity_id.clone()) {
                    json!({"kind": "none"})
                } else if pres_after == pres_before {
                    json!({"kind": "keep", "tie": [], "next": 0})
                } else {
                    let to = pres_after.unwrap_or(0);
                    json!({"kind": "pass", "to": {"id": g.site_id(to), "name": player_name(g, g.site_id(to)),
                           "pct": ac.percent_owned_by(&EntityId::player(to))}})
                };
                let _ = seller;
                json!({"priceFrom": sp.price, "priceTo": ac.share_price.as_ref().map(|s| s.price), "left": left,
                       "cellFrom": [sp.row, sp.column], "cellTo": ac.share_price.as_ref().map(|s| json!([s.row, s.column])), "pres": pres})
            });
            Some(json!({"corp": corporation_sym, "percent": percent, "price": price, "sale": sale, "action": o.action}))
        })
        .collect()
}

/// A step's named choices (`choose`): the context the client words, each option with its action.
fn choices(g: &SiteGame, context: &str, company: &str, mine: &[Offer], m: &mut Map<String, Value>) {
    let opts: Vec<Value> = of(mine, "choose").map(|o| json!({"choice": sval(o.action.get("choice")), "action": o.action})).collect();
    if opts.is_empty() {
        return;
    }
    let mut c = json!({"context": context, "options": opts});
    if !company.is_empty() {
        c["company"] = json!(company);
        c["companyName"] = json!(company_name(g, company));
    }
    m.insert("choices".into(), c);
}

/// The companies' own named choices (`choose_ability`): P17's card moves ("M26_top"), P18's / 1822CA P28's station
/// swaps, P5's exchange token; the tax haven's are `taxHaven`.
fn ability_choices(g: &SiteGame, mine: &[Offer], m: &mut Map<String, Value>) {
    let fam = g.game.fam();
    let mut groups: Vec<(String, Vec<Value>)> = Vec::new();
    for o in of(mine, "choose_ability") {
        let co = sval(o.action.get("entity"));
        if co == fam.company_osth {
            continue;
        }
        let choice = sval(o.action.get("choice"));
        let mut opt = json!({"choice": choice, "action": o.action});
        if let Some((card, place)) = choice.rsplit_once('_') {
            if place == "top" || place == "bottom" {
                opt["card"] = json!(card);
                opt["where"] = json!(place);
            }
        }
        match groups.iter_mut().find(|x| x.0 == co) {
            Some(x) => x.1.push(opt),
            None => groups.push((co, vec![opt])),
        }
    }
    if groups.is_empty() {
        return;
    }
    let list: Vec<Value> = groups
        .into_iter()
        .map(|(co, options)| {
            let kind = if co == fam.company_lur {
                "lur"
            } else if co == fam.company_chpr || (super::is_ca(&g.game) && co == "P28") {
                "chpr"
            } else if co == fam.company_lcdr {
                "lcdr"
            } else {
                "other"
            };
            json!({"company": co, "name": company_name(g, &co), "kind": kind, "options": options})
        })
        .collect();
    m.insert("abilityChoices".into(), json!(list));
}

fn operating_parts(g: &mut SiteGame, bi: usize, step: Step, ent: &Ent, mine: &[Offer], m: &mut Map<String, Value>) {
    let corp = match ent {
        Ent::Corp(c) => c.clone(),
        Ent::Company(co) => g.game.co_owner(co).corp_sym().map(str::to_string).unwrap_or_default(),
        Ent::Player(_) => String::new(),
    };
    let _ = bi;
    // privates the company may acquire (the president's own: another player's left out)
    let acq: Vec<Value> = of(mine, "acquire_company")
        .map(|o| {
            let co = sval(o.action.get("company"));
            let value = g.game.company_idx.get(co.as_str()).map_or(0, |&i| g.game.companies[i].value);
            let owner = g.game.co_owner(&co).player_id().map(|p| g.site_id(p));
            json!({"company": co, "name": company_name(g, &co), "value": value, "owner": owner, "action": o.action})
        })
        .collect();
    if !acq.is_empty() {
        m.insert("acquire".into(), json!(acq));
    }
    // track: the company's own lays and its privates' (with the privates they combine with)
    let lays: Vec<&Offer> = of(mine, "lay_tile").collect();
    if !lays.is_empty() {
        let mut hexes: Vec<String> = Vec::new();
        let mut by_company: Vec<(String, Vec<String>, Vec<Vec<String>>)> = Vec::new();
        for o in &lays {
            let hex = sval(o.action.get("hex"));
            let e = sval(o.action.get("entity"));
            if o.action.get("entity_type").and_then(Value::as_str) == Some("company") {
                let combo = o.m.combo.clone();
                match by_company.iter_mut().find(|x| x.0 == e) {
                    Some(x) => {
                        if !x.1.contains(&hex) {
                            x.1.push(hex);
                        }
                        if !combo.is_empty() && !x.2.contains(&combo) {
                            x.2.push(combo);
                        }
                    }
                    None => by_company.push((e, vec![hex], if combo.is_empty() { vec![] } else { vec![combo] })),
                }
            } else if !hexes.contains(&hex) {
                hexes.push(hex);
            }
        }
        m.insert("tileHexes".into(), json!(hexes));
        if !by_company.is_empty() {
            let ab: Vec<Value> = by_company
                .into_iter()
                .map(|(co, hexes, combos)| {
                    json!({"company": co, "name": company_name(g, &co), "owner": corp, "hexes": hexes, "teleport": false,
                           "extra": true, "desc": "", "combos": combos})
                })
                .collect();
            m.insert("abilities".into(), json!(ab));
        }
        if let Some(&ci) = g.game.corp_idx.get(corp.as_str()) {
            m.insert("tilePower".into(), json!(g.game.corporations[ci].cash));
        }
    }
    // stations: the company's, its privates' (P5's English Channel station, 1822CA P10's Winnipeg exchange station),
    // a destination token named by city (1822CA ICR), a pending home token
    let toks: Vec<Value> = of(mine, "place_token")
        .filter_map(|o| {
            let Action::PlaceToken { hex_id, city_index, .. } = &o.m.action else { return None };
            let cities = g.game.hex_ref(hex_id).map_or(1, |h| h.tile.cities.len());
            let price = after(g, o).map_or(0, |a| {
                let payer = a.corp_idx.get(corp.as_str()).map_or(0, |&ci| g.game.corporations[ci].cash - a.corporations[ci].cash);
                payer
            });
            let mut t = json!({"hex": hex_id, "city": city_index, "cities": cities, "price": price, "action": o.action});
            if o.action.get("entity_type").and_then(Value::as_str) == Some("company") {
                t["via"] = o.action["entity"].clone();
            }
            if let Some(x) = o.action.get("tokener") {
                t["tokener"] = x.clone();
            }
            if let Some(x) = o.action.get("token_type") {
                t["tokenType"] = x.clone();
            }
            Some(t)
        })
        .collect();
    if !toks.is_empty() {
        m.insert("tokens".into(), json!(toks));
    }
    let hex_tokens: Vec<Value> = of(mine, "hex_token")
        .map(|o| json!({"hex": o.action.get("hex"), "tokenType": o.action.get("token_type"), "action": o.action}))
        .collect();
    if !hex_tokens.is_empty() {
        m.insert("hexToken".into(), json!(hex_tokens));
    }
    match step {
        Step::Route => {
            if of(mine, "run_routes").next().is_some() {
                m.insert("routes".into(), json!({"pending": true}));
            }
            let opts: Vec<Value> = of(mine, "choose")
                .map(|o| {
                    let choice = sval(o.action.get("choice"));
                    let train = route_choice_train(g, &corp, &choice);
                    json!({"choice": choice, "train": train, "action": o.action})
                })
                .collect();
            if !opts.is_empty() {
                let kind = if super::is_ca(&g.game) && g.game.g1822_ca_choosing(&corp).is_some_and(|c| format!("{c:?}").contains("Grain")) {
                    "grain"
                } else {
                    "pullman"
                };
                m.insert("routeChoices".into(), json!({"kind": kind, "options": opts}));
            }
        }
        Step::Dividend => {
            let divs: Vec<Value> = of(mine, "dividend").map(|o| json!({"kind": sval(o.action.get("kind")), "action": o.action, "preview": dividend_preview(g, &corp, o)})).collect();
            if !divs.is_empty() {
                m.insert("dividend".into(), json!(divs));
            }
            let extra: Vec<Value> = of(mine, "choose").map(|o| json!({"kind": sval(o.action.get("choice")), "action": o.action})).collect();
            if !extra.is_empty() {
                let train = g.game.g1822_find_extra_train(&corp);
                let revenue = train.as_ref().and_then(|t| g.game.rnd().routes.iter().find(|r| &r.train_id == t).map(|r| r.revenue));
                m.insert("extraDividend".into(), json!({"train": train, "revenue": revenue, "options": extra}));
            }
        }
        Step::MinorAcquisition => {
            let r = g.game.rnd();
            let phase = match r.acquire_state {
                AcquireState::SelectMinor => "select",
                AcquireState::ChoosePay => "pay",
                AcquireState::ChooseToken => "token",
            };
            let candidates: Vec<Value> = of(mine, "merge")
                .map(|o| {
                    let minor = sval(o.action.get("corporation"));
                    let owner = g.game.g1822_corp_president(&minor).map(|p| g.site_id(p));
                    json!({"minor": minor, "name": corp_name(g, &minor), "owner": owner, "bank": owner.is_none(), "action": o.action})
                })
                .collect();
            let opts: Vec<Value> = of(mine, "choose").map(|o| json!({"choice": sval(o.action.get("choice")), "action": o.action})).collect();
            let mut am = json!({"phase": phase, "selected": r.selected_minor, "candidates": candidates});
            if phase == "pay" {
                am["pay"] = json!(opts);
            } else if phase == "token" {
                am["token"] = json!(opts);
            }
            m.insert("acquireMinor".into(), am);
        }
        Step::AssignSawmill => {
            let then: Vec<Value> = of(mine, "choose").map(|o| json!({"choice": sval(o.action.get("choice")), "action": o.action})).collect();
            m.insert("sawmill".into(), json!({"hexes": [], "then": then}));
        }
        Step::SpecialChoose => {
            let co = g.game.rnd().special_choose_company.clone().unwrap_or_default();
            choices(g, "special", &co, mine, m);
        }
        Step::IssueShares => {
            let issue: Vec<Value> = of(mine, "sell_shares")
                .filter(|o| o.action.get("entity_type").and_then(Value::as_str) == Some("corporation"))
                .map(|o| {
                    let Action::SellShares { percent, .. } = &o.m.action else { unreachable!() };
                    json!({"corp": corp, "percent": percent, "price": o.m.share_price, "priceAfter": after(g, o).and_then(|a| a.corp_idx.get(corp.as_str()).and_then(|&ci| a.corporations[ci].share_price.as_ref().map(|s| s.price))), "action": o.action})
                })
                .collect();
            if !issue.is_empty() {
                m.insert("issue".into(), json!(issue));
            }
            let redeem: Vec<Value> = of(mine, "buy_shares")
                .filter(|o| o.action.get("entity_type").and_then(Value::as_str) == Some("corporation"))
                .map(|o| {
                    let Action::BuyShares { percent, .. } = &o.m.action else { unreachable!() };
                    json!({"corp": corp, "percent": percent, "price": o.m.share_price.unwrap_or_else(|| share_price(g, &corp, *percent)), "action": o.action})
                })
                .collect();
            if !redeem.is_empty() {
                m.insert("redeem".into(), json!(redeem));
            }
        }
        _ => {}
    }
    // 1822CA's sawmill (P13): placed on a hex any time in its owner's operating turn, then open or closed
    let assign: Vec<Value> = of(mine, "assign").map(|o| json!({"hex": o.action.get("target"), "company": o.action.get("entity"), "action": o.action})).collect();
    if !assign.is_empty() {
        let then = m.get("sawmill").and_then(|s| s.get("then")).cloned().unwrap_or(json!([]));
        m.insert("sawmill".into(), json!({"hexes": assign, "then": then}));
    }
    // the president's sales (emergency money before a train purchase, or what the step allows outside one)
    let sells = sell_list(g, mine, true);
    if !sells.is_empty() {
        m.insert("sell".into(), json!(sells));
    }
    debt_part(g, mine, m);
    // trains: the depot's (an L or its 2, 3-7, the 7's E, the L's upgrade to a 2), another of the president's companies'
    if of(mine, "buy_train").next().is_some() {
        let (trains, buy) = train_parts(g, &corp, mine);
        m.insert("trains".into(), trains);
        m.insert("trainBuy".into(), buy);
    }
    let discards: Vec<Value> = of(mine, "discard_train")
        .map(|o| json!({"corp": o.action.get("entity"), "train": o.action.get("train"), "own": sval(o.action.get("entity")) == corp, "action": o.action}))
        .collect();
    if !discards.is_empty() {
        m.insert("discard".into(), json!(discards));
    }
    ability_choices(g, mine, m);
}

/// The train a Route step choice names: a pullman's index into the company's trains it may attach to (G1822), 1822CA's
/// grain / pullman train ids (or "None").
fn route_choice_train(g: &SiteGame, corp: &str, choice: &str) -> Value {
    if let Ok(i) = choice.parse::<usize>() {
        let ids = g.game.g1822_pullman_choices(corp);
        if let Some(id) = ids.get(i) {
            let name = g.game.corp_idx.get(corp).and_then(|&ci| g.game.corporations[ci].trains.iter().find(|t| &t.id == id)).map(|t| t.name.clone());
            return json!({"id": id, "name": name});
        }
    }
    if choice == "None" {
        return Value::Null;
    }
    let name = g.game.corp_idx.get(corp).and_then(|&ci| g.game.corporations[ci].trains.iter().find(|t| t.id == choice)).map(|t| t.name.clone());
    json!({"id": choice, "name": name})
}

/// What a dividend does, on a copy: revenue, per share, to the treasury, each player's, the price move.
fn dividend_preview(g: &SiteGame, corp: &str, o: &Offer) -> Value {
    let game = &g.game;
    let Some(&ci) = game.corp_idx.get(corp) else { return Value::Null };
    let revenue = game.g1822_routes_revenue();
    let Some(a) = after(g, o) else { return Value::Null };
    let players: Vec<Value> = game
        .players
        .iter()
        .filter_map(|p| {
            let after_cash = a.players.iter().find(|x| x.id == p.id).map_or(p.cash, |x| x.cash);
            (after_cash != p.cash).then(|| json!({"id": g.site_id(p.id), "amount": after_cash - p.cash}))
        })
        .collect();
    let c = &game.corporations[ci];
    let unit = c.share_unit_percent.max(1) as i32;
    let per = match sval(o.action.get("kind")).as_str() {
        "payout" => revenue / (100 / unit),
        "half" => (revenue - revenue / 2) / (100 / unit),
        _ => 0,
    };
    json!({"kind": sval(o.action.get("kind")), "revenue": revenue, "perShare": per, "toTreasury": a.corporations[ci].cash - c.cash,
           "players": players, "priceFrom": c.share_price.as_ref().map(|s| s.price),
           "priceTo": a.corporations[ci].share_price.as_ref().map(|s| s.price)})
}

/// The depot's trains and the train purchase's facts.
fn train_parts(g: &mut SiteGame, corp: &str, mine: &[Offer]) -> (Value, Value) {
    let game = &g.game;
    let Some(&ci) = game.corp_idx.get(corp) else { return (json!([]), Value::Null) };
    let c = &game.corporations[ci];
    let cash = c.cash;
    let pres = super::corp_owner(game, corp);
    let pres_cash = pres.and_then(|p| game.players.iter().find(|x| x.id == p)).map_or(0, |p| p.cash);
    let discarded: Vec<String> = game.depot.discarded.iter().map(|t| t.id.clone()).collect();
    let mut trains = Vec::new();
    let mut exchanges = Vec::new();
    let mut others = Vec::new();
    for o in of(mine, "buy_train") {
        let Action::BuyTrain { train_name, price, from, variant, exchange, .. } = &o.m.action else { continue };
        let (name, owner) = super::act::train_info(g, train_name);
        if owner == "depot" || from == "depot" {
            let shown = variant.clone().unwrap_or(name.clone());
            if exchange.is_some() {
                exchanges.push(json!({"old": exchange, "variant": shown, "price": price, "full": price, "buyingPower": cash, "action": o.action}));
            } else {
                trains.push(json!({"train": train_name, "name": shown, "variant": variant, "price": price,
                                   "discarded": discarded.contains(train_name), "short": (price - cash).max(0),
                                   "mayContribute": price > &cash, "presCash": pres_cash, "cheapest": null, "action": o.action}));
            }
        } else {
            let owner_p = game.g1822_corp_president(&owner);
            let face = game.title_def().trains().iter().find(|t| t.name == name).map_or(0, |t| t.price);
            let (min, max) = o.m.price.map_or((*price, *price), |r| (r.lo, r.hi));
            others.push(json!({"train": train_name, "name": name, "count": 1, "from": owner, "owner": owner_p.map(|p| g.site_id(p)),
                               "ownerName": owner_p.map(|p| player_name(g, g.site_id(p))), "min": min, "max": max, "face": face,
                               "fixed": if o.m.price.is_none() { json!(price) } else { Value::Null }, "action": o.action,
                               "range": o.range()}));
        }
    }
    let sells = sell_list(g, mine, true);
    let must = of(mine, "pass").next().is_none() && !trains.is_empty();
    let debt_now = pres.map_or(0, |p| game.player_debt(p));
    let buy = json!({"corp": corp, "cash": cash, "buyingPower": cash, "mustBuy": must, "presidentMayContribute": must,
                     "president": pres.map(|p| player_json(g, p)), "others": others, "exchanges": exchanges, "sell": sells,
                     "cheapest": null, "debt": {"afterSell": true, "owed": debt_now, "interest": game.fam().player_loan_interest_rate}});
    (json!(trains), buy)
}

/// The tile lays on a hex for the operating company or one of its privates (`via`): each tile (Ruby's instance) in
/// each rotation, with its action.
pub fn tile_options(g: &mut SiteGame, hex: &str, via: Option<&str>) -> Result<Value, String> {
    let (mine, _) = offers(g);
    let mut out = Vec::new();
    for o in of(&mine, "lay_tile") {
        if sval(o.action.get("hex")) != hex {
            continue;
        }
        let is_company = o.action.get("entity_type").and_then(Value::as_str) == Some("company");
        match via {
            Some(c) if !(is_company && sval(o.action.get("entity")) == c) => continue,
            None if is_company => continue,
            _ => {}
        }
        let id = sval(o.action.get("tile"));
        let name = id.rsplit_once('-').map_or(id.clone(), |x| x.0.to_string());
        let cost = after(g, o).map_or(0, |a| {
            let corp = g.operator_of(&sval(o.action.get("entity")));
            a.corp_idx.get(corp.as_str()).map_or(0, |&ci| g.game.corporations[ci].cash - a.corporations[ci].cash)
        });
        out.push(json!({"tile": id, "name": name, "rotation": o.action.get("rotation"), "cost": cost, "terrain": [],
                        "combo": o.m.combo, "action": o.action}));
    }
    Ok(json!(out))
}

/// The router's best run for the operating company (the one run `act` takes): {action, revenue, routes (+track)}.
pub fn auto_route(g: &mut SiteGame) -> Value {
    let Some((_, Step::Route, Ent::Corp(corp))) = super::active(&g.game) else { return Value::Null };
    let a = json!({"type": "run_routes", "entity": corp, "entity_type": "corporation", "routes": []});
    let probe = {
        let (mine, _) = offers(g);
        let run = of(&mine, "run_routes").next().map(|o| o.action.clone());
        run
    };
    let Some(mut posted) = probe else { return Value::Null };
    let _ = a;
    // the router's run with Ruby's fields, as `act` stores it
    if let Some(o) = posted.as_object_mut() {
        o.remove("subsidy");
    }
    let checked = match super::act::checked_run(g, &posted) {
        Ok(x) => x,
        Err(_) => return Value::Null,
    };
    let revenue: i64 = checked.get("routes").and_then(Value::as_array).map(|rs| rs.iter().filter_map(|r| r.get("revenue").and_then(Value::as_i64)).sum()).unwrap_or(0);
    let routes: Vec<Value> = checked
        .get("routes")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
        .into_iter()
        .map(|mut r| {
            if let Some(t) = super::super::track::route_track(g, &r) {
                r["track"] = t;
            }
            r
        })
        .collect();
    json!({"action": checked, "revenue": revenue, "subsidy": checked.get("subsidy"), "routes": routes})
}

/// Every offered move of the players to act (own-only), as site JSON.
pub fn legal_actions(g: &mut SiteGame) -> Value {
    let mut out: Vec<Value> = Vec::new();
    for pid in super::active_players(g) {
        let m = menu(g, pid);
        if let Some(list) = m.get("moves").and_then(Value::as_array) {
            for x in list {
                if let Some(a) = x.get("action") {
                    if !out.contains(a) {
                        out.push(a.clone());
                    }
                }
            }
        }
    }
    json!(out)
}
