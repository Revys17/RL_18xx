//! Ruby's auto actions in the site layer (docs/SITE_ENGINE_RS_STEP3.md §5.3): after every move Ruby's
//! `process_action` runs `round.auto_actions` (the active step's `auto_actions(current_entity)`) until there are none
//! and appends them to the move. The steps 1830 and 1867 have:
//!
//! | Ruby | Here |
//! |---|---|
//! | `Step::Programmer#programmed_auto_actions` (BuySellParShares) | the current player's program (one at a time): `activate_program_share_pass`, `activate_program_buy_shares` |
//! | `BuySellParShares#should_stop_applying_program` / `#action_is_shenanigan?` | `must_sell?`, then the round's `players_history` (`StockHistory`, kept by `SiteGame`) since the program was enabled |
//! | `ProgrammerMergerPass#activate_program_merger_pass` (1867's Merge) | the merger record's `converts`, `corporations_by_round` |
//! | 1867's Token / Merge / BuyCompanyPreloan passes | `g1867::autos` (step 2) |
//!
//! Reasons are Ruby's strings word for word (they are stored). `created_at` of every auto action is the time the move
//! was made (Ruby's `Time.now`).

use super::game::SiteGame;
use crate::rounds::Round;
use serde_json::{json, Value};

/// One action in Ruby's `@round.players_history` (`track_action`): its type, its entity (a player's id or, for an
/// exchange, the company's sym), and what orders it against a program (`Action#<=>`: ids when both have one, else
/// `created_at`).
#[derive(Clone, Debug)]
pub struct HistItem {
    pub kind: String,
    pub entity: String,
    pub id: Option<i64>,
    pub created_at: i64,
}

/// Ruby's `@round.players_history` of the stock round in progress: player -> [(corporation (a company for a private
/// bought), [actions])], keys in the order Ruby's hash got them (a player's key appears when their turn starts, or
/// when an action of theirs is tracked first).
#[derive(Clone, Debug, Default)]
pub struct StockHistory {
    /// The stock round these entries belong to (its turn); None outside a stock round
    pub round: Option<u32>,
    pub players: Vec<(u32, Vec<(String, Vec<HistItem>)>)>,
}

impl StockHistory {
    fn entry(&mut self, pid: u32) -> &mut Vec<(String, Vec<HistItem>)> {
        if let Some(i) = self.players.iter().position(|p| p.0 == pid) {
            return &mut self.players[i].1;
        }
        self.players.push((pid, Vec::new()));
        &mut self.players.last_mut().unwrap().1
    }

    /// `players_history[player].clear` (BuySellParShares#setup: each turn start).
    pub fn clear(&mut self, pid: u32) {
        self.entry(pid).clear();
    }

    pub fn track(&mut self, pid: u32, corp: &str, item: HistItem) {
        let e = self.entry(pid);
        match e.iter_mut().find(|x| x.0 == corp) {
            Some(x) => x.1.push(item),
            None => e.push((corp.to_string(), vec![item])),
        }
    }
}

/// The stock round's turn order and whose turn it is (Ruby's `@entities`, `@entity_index`), with the round's turn.
pub fn stock_turn(g: &SiteGame) -> Option<(u32, Vec<u32>, usize)> {
    match &g.game.round {
        Round::Stock(s) => Some((g.game.turn, s.player_order.clone(), s.current_player_index)),
        _ => None,
    }
}

/// Keep `players_history` after an engine action (called by `SiteGame::apply_engine`): `before` is the stock turn
/// before it, `raw` the site action, `ctx` what the site layer saw (the acting entity's player, a minor won in a
/// 1867 auction).
pub fn observe(g: &mut SiteGame, before: Option<(u32, Vec<u32>, usize)>, raw: &Value, entity_player: Option<u32>, won_minor: Option<(u32, String)>, turn_advanced: bool, auction_before: bool) {
    let after = stock_turn(g);
    let Some((turn, order, idx)) = after.clone() else {
        g.stock_history = StockHistory::default();
        return;
    };
    let t = raw.get("type").and_then(Value::as_str).unwrap_or("").to_string();
    let created_at = raw.get("created_at").and_then(Value::as_i64).unwrap_or(0);
    let id = raw.get("id").and_then(Value::as_i64);
    // a new stock round: a fresh history, every entity visited from the first to the current one set up
    let new_round = g.stock_history.round != Some(turn) || before.as_ref().is_none_or(|b| b.0 != turn);
    if new_round {
        g.stock_history = StockHistory { round: Some(turn), players: Vec::new() };
        for k in 0..=idx.min(order.len().saturating_sub(1)) {
            if let Some(&p) = order.get(k) {
                g.stock_history.clear(p);
            }
        }
        return;
    }
    // the action itself (track_action), before the turn moves on
    let corp_of_shares = || -> Option<String> {
        raw.get("corporation").and_then(Value::as_str).map(str::to_string).or_else(|| {
            raw.get("shares").and_then(Value::as_array).and_then(|v| v.first()).and_then(Value::as_str).and_then(|x| x.rsplit_once('_').map(|y| y.0.to_string()))
        })
    };
    let by_company = raw.get("entity_type").and_then(Value::as_str) == Some("company");
    let item = |kind: &str| HistItem { kind: kind.to_string(), entity: raw.get("entity").map(|e| e.as_str().map(str::to_string).unwrap_or_else(|| e.to_string())).unwrap_or_default(), id, created_at };
    if let Some(pid) = entity_player {
        match t.as_str() {
            "buy_shares" | "sell_shares" | "par" => {
                if let Some(c) = corp_of_shares() {
                    if !by_company || t == "buy_shares" {
                        g.stock_history.track(pid, &c, item(&t));
                    }
                }
            }
            "buy_company" if !by_company && raw.get("entity_type").and_then(Value::as_str) == Some("player") => {
                if let Some(c) = raw.get("company").and_then(Value::as_str) {
                    g.stock_history.track(pid, c, item("buy_company"));
                }
            }
            _ => {}
        }
    }
    // 1867: a minor won at auction is parred by the step (an internal Par: no id, made when the move was)
    if let Some((winner, minor)) = won_minor {
        g.stock_history.track(winner, &minor, HistItem { kind: "par".into(), entity: winner.to_string(), id: None, created_at });
    }
    // the turn moved on: each entity the round visited had its turn set up (Round::Stock#start_entity)
    if turn_advanced {
        if let Some((_, _, bidx)) = before {
            let n = order.len().max(1);
            if auction_before || (g.game.title == "1867" && t == "bid") {
                // after an auction the round goes to the winner (goto_entity!: no setup), then on to the next player
                g.stock_history.clear(order[idx % n]);
            } else {
                let mut k = bidx;
                for _ in 0..n {
                    k = (k + 1) % n;
                    g.stock_history.clear(order[k]);
                    if k == idx {
                        break;
                    }
                }
            }
        }
    }
}

fn s(v: &Value, k: &str) -> Option<String> {
    v.get(k).filter(|x| !x.is_null()).map(|x| x.as_str().map(str::to_string).unwrap_or_else(|| x.to_string()))
}

fn pass(entity: Value, entity_type: &str, now: i64) -> Value {
    json!({"type": "pass", "entity": entity, "entity_type": entity_type, "created_at": now})
}

fn disable(pid: u32, reason: &str, now: i64) -> Value {
    json!({"type": "program_disable", "entity": pid, "entity_type": "player", "created_at": now, "reason": reason})
}

fn player_name(g: &SiteGame, pid: u32) -> String {
    g.players.iter().find(|p| p.0 == pid).map(|p| p.1.clone()).unwrap_or_default()
}

/// A corporation's president (Ruby `corporation.owner`).
fn owner(g: &SiteGame, corp: &str) -> Option<u32> {
    let &ci = g.game.corp_idx.get(corp)?;
    let c = &g.game.corporations[ci];
    if g.game.title == "1867" {
        c.owner_id.player_id().or_else(|| c.president_id())
    } else {
        c.president_id()
    }
}

fn percent_of(g: &SiteGame, holder: &crate::entities::EntityId, corp: &str) -> i32 {
    g.game.corp_idx.get(corp).map_or(0, |&ci| g.game.corporations[ci].percent_owned_by(holder) as i32)
}

/// Ruby `check_sale_timing(entity, Share.new(corporation).to_bundle)` in a stock round: 1830 (`SELL_AFTER :first`)
/// after the first stock round; 1867 (`:operate`) once the company has operated.
fn sale_timing(g: &SiteGame, corp: &str) -> bool {
    if g.game.title == "1867" {
        let rec = g.game.site.as_ref().and_then(|r| r.op_history.get(corp)).is_some_and(|l| !l.is_empty());
        rec || g.game.corp_idx.get(corp).is_some_and(|&ci| g.game.corporations[ci].ever_operated)
    } else {
        g.game.turn > 1
    }
}

/// Ruby `action_is_shenanigan?`.
fn shenanigan(g: &SiteGame, me: u32, other: u32, item: &HistItem, corp: &str, share: Option<&(String, u32)>) -> Option<String> {
    // the share's corporation ("CPR_2" -> "CPR")
    let corp_buying = share.and_then(|s| s.0.rsplit_once('_')).map(|x| x.0);
    match item.kind.as_str() {
        "par" => (corp_buying.is_none() || sale_timing(g, corp)).then(|| format!("Corporation {corp} parred")),
        "buy_shares" => {
            if item.entity == corp {
                return Some(format!("{corp} redeemed a share."));
            }
            if owner(g, corp) != Some(me) {
                return None;
            }
            let me_e = crate::entities::EntityId::player(me);
            let mine = percent_of(g, &me_e, corp);
            if mine >= 50 {
                return None;
            }
            if corp_buying != Some(corp) {
                return Some(format!("{} bought on corporation {corp} and is unsecure", player_name(g, other)));
            }
            let pct = share.map_or(0, |s| s.1 as i32);
            let percentage = mine + pct;
            if percentage > 50 {
                return None;
            }
            let ci = *g.game.corp_idx.get(corp)?;
            let c = &g.game.corporations[ci];
            let bigger = c.shares.iter().filter(|x| (x.percent as i32) > pct && (x.owner != me_e || owner(g, corp) != Some(me))).map(|x| x.percent as i32).max();
            if let Some(b) = bigger {
                // the buyer: a player, or a company's owner (an exchange)
                let (holder, name) = match item.entity.parse::<u32>() {
                    Ok(p) => (crate::entities::EntityId::player(p), player_name(g, p)),
                    Err(_) => {
                        let comp_owner = g.game.company_idx.get(item.entity.as_str()).and_then(|&mi| g.game.companies[mi].owner.player_id());
                        (crate::entities::EntityId::none(), comp_owner.map(|p| player_name(g, p)).unwrap_or_default())
                    }
                };
                let other_percent = percent_of(g, &holder, corp) + b;
                if percentage < other_percent {
                    return Some(format!("{name} has bought, shares exist that could allow them to gain presidency"));
                }
            }
            None
        }
        "sell_shares" => Some("Shares were sold".into()),
        "take_loan" => Some(format!("{corp} took a loan")),
        other => Some(format!("Unknown action {other} disabling for safety")),
    }
}

/// `Action#<=>`: ids when both have one, else the time.
fn before_program(item: &HistItem, prog: &(Option<i64>, i64)) -> bool {
    match (item.id, prog.0) {
        (Some(a), Some(b)) => a < b,
        _ => item.created_at < prog.1,
    }
}

/// The step's actions for the current entity (Ruby `step.actions(entity)`).
fn step_actions(g: &mut SiteGame) -> Vec<String> {
    if g.game.title == "1867" {
        super::g1867::ruby_actions(&mut g.game)
    } else {
        super::shadow::ruby_actions(g)
    }
}

/// Ruby `should_stop_applying_program`.
fn should_stop(g: &mut SiteGame, me: u32, prog: &(Option<i64>, i64), share: Option<&(String, u32)>, unconditional_actions: &[String]) -> Option<String> {
    if unconditional_actions == ["sell_shares"] {
        return Some(format!("{} must sell shares", player_name(g, me)));
    }
    let hist = g.stock_history.players.clone();
    for (other, corps) in &hist {
        if *other == me {
            continue;
        }
        for (corp, items) in corps {
            for item in items {
                if before_program(item, prog) {
                    continue;
                }
                if let Some(r) = shenanigan(g, me, *other, item, corp, share) {
                    return Some(r);
                }
            }
        }
    }
    None
}

/// Ruby `normal_pass?` (BuySellParSharesViaBid: not inside a minor's auction).
fn normal_pass(g: &SiteGame) -> bool {
    !matches!(&g.game.round, Round::Stock(s) if s.bid_auction.is_some())
}

/// The shares a buy program may take: the first of the IPO's (or the market's) that the player can buy, by percent.
fn buy_candidates(g: &mut SiteGame, pid: u32, corp: &str, from_market: bool) -> Vec<(String, u32)> {
    let want = if from_market { "market" } else { "ipo" };
    if g.game.title == "1867" {
        let moves = g.game.g1867_legal_moves_with(crate::title::CrossPlayerPolicy::RULES);
        let mut out = Vec::new();
        for mv in &moves {
            let crate::actions::Action::BuyShares { corporation_sym, entity_id, percent, .. } = &mv.action else { continue };
            if corporation_sym != corp || entity_id != &pid.to_string() {
                continue;
            }
            let Some(id) = mv.shares.first() else { continue };
            let idx = id.rsplit_once('_').and_then(|(_, i)| i.parse::<usize>().ok());
            let ci = g.game.corp_idx[corp];
            let in_market = idx.and_then(|i| g.game.corporations[ci].shares.iter().find(|s| s.index == i)).is_some_and(|s| s.owner.is_market());
            if in_market == from_market && !out.iter().any(|x: &(String, u32)| x.0 == *id) {
                out.push((id.clone(), *percent as u32));
            }
        }
        return out;
    }
    let mut out = Vec::new();
    for (c, src, idx, _) in g.game.buyable_shares(pid) {
        if c != corp || src != want {
            continue;
        }
        let ci = g.game.corp_idx[corp];
        let co = &g.game.corporations[ci];
        // the market's first share is the first to have arrived there (Ruby's pool list)
        let share = if src == "market" {
            co.shares.iter().filter(|s| s.owner.is_market() && !s.president).min_by_key(|s| s.acquired_seq).unwrap_or(&co.shares[idx])
        } else {
            &co.shares[idx]
        };
        let id = format!("{}_{}", corp, share.index);
        if !out.iter().any(|x: &(String, u32)| x.0 == id) {
            out.push((id, share.percent as u32));
        }
    }
    out
}

/// The current entity's program's auto actions in the stock round (BuySellParShares' Programmer).
fn stock_programs(g: &mut SiteGame, pid: u32, now: i64) -> Vec<Value> {
    let Some(list) = g.programs.get(&pid).cloned() else { return Vec::new() };
    let mut out = Vec::new();
    for prog in list {
        let meta = g.program_meta.get(&pid).cloned().unwrap_or((None, 0));
        match s(&prog, "type").as_deref() {
            Some("program_share_pass") => {
                let acts = step_actions(g);
                if !acts.iter().any(|t| t == "pass") || !normal_pass(g) {
                    continue;
                }
                let unconditional = prog.get("unconditional").and_then(Value::as_bool).unwrap_or(false);
                let reason = if unconditional { None } else { should_stop(g, pid, &meta, None, &acts) };
                match reason {
                    Some(r) => out.push(disable(pid, &r, now)),
                    None => out.push(pass(json!(pid), "player", now)),
                }
            }
            Some("program_buy_shares") => {
                let corp = s(&prog, "corporation").unwrap_or_default();
                let Some(&ci) = g.game.corp_idx.get(corp.as_str()) else { continue };
                let until = prog.get("until_condition").cloned().unwrap_or(Value::Null);
                let c = &g.game.corporations[ci];
                let held: u32 = c.shares.iter().filter(|x| x.owner.is_player() || x.owner.is_market()).map(|x| x.percent as u32).sum();
                let to_float = c.float_percent as i64 - held as i64;
                let finished = if until.as_str() == Some("float") {
                    (c.floated || to_float <= 0).then(|| format!("{corp} is floated"))
                } else {
                    let n = until.as_f64().or_else(|| until.as_str().and_then(|x| x.parse().ok())).unwrap_or(f64::MAX);
                    let mine = c.percent_owned_by(&crate::entities::EntityId::player(pid)) as f64 / c.share_unit_percent.max(1) as f64;
                    (mine >= n).then(|| format!("{} share(s) bought in {corp}, end condition met", until_str(&until)))
                };
                if let Some(r) = finished {
                    out.push(disable(pid, &r, now));
                    if prog.get("auto_pass_after").and_then(Value::as_bool).unwrap_or(false) {
                        // ProgramSharePass.new(entity): its defaults are stored (false isn't nil)
                        out.push(json!({"type": "program_share_pass", "entity": pid, "entity_type": "player", "created_at": now,
                                        "unconditional": false, "indefinite": false}));
                    }
                    continue;
                }
                let acts = step_actions(g);
                if acts.iter().any(|t| t == "buy_shares") {
                    let from_market = prog.get("from_market").and_then(Value::as_bool).unwrap_or(false);
                    let source = if from_market { "market" } else if g.game.title == "1867" { "Treasury" } else { "IPO" };
                    let cands = buy_candidates(g, pid, &corp, from_market);
                    if cands.is_empty() {
                        out.push(disable(pid, &format!("Cannot buy {corp} from {source}"), now));
                        continue;
                    }
                    let mut pcts: Vec<u32> = cands.iter().map(|x| x.1).collect();
                    pcts.dedup();
                    if pcts.len() != 1 {
                        out.push(disable(pid, &format!("Shares of different sizes exist, cannot auto buy {corp} from {source}"), now));
                        continue;
                    }
                    let share = cands[0].clone();
                    if let Some(r) = should_stop(g, pid, &meta, Some(&share), &acts) {
                        out.push(disable(pid, &r, now));
                        continue;
                    }
                    out.push(json!({"type": "buy_shares", "entity": pid, "entity_type": "player", "created_at": now,
                                    "shares": [share.0], "percent": share.1}));
                } else if bought(g) && acts.iter().any(|t| t == "pass") {
                    let floated = g.game.corporations[ci].floated;
                    if until.as_str() == Some("float") && floated {
                        continue;
                    }
                    out.push(pass(json!(pid), "player", now));
                }
            }
            _ => {}
        }
    }
    out
}

fn until_str(v: &Value) -> String {
    match v {
        Value::String(x) => x.clone(),
        other => other.to_string(),
    }
}

/// Ruby `bought?`: the turn has a purchase (a share bought, a par, a private bought) by the player.
fn bought(g: &SiteGame) -> bool {
    g.turn_actions.iter().any(|a| {
        let t = a.get("type").and_then(Value::as_str).unwrap_or("");
        let by_player = a.get("entity_type").and_then(Value::as_str).unwrap_or("player") == "player";
        by_player && matches!(t, "buy_shares" | "par" | "buy_company")
    }) || match &g.game.round {
        Round::Stock(s) => s.bought_this_turn || s.parred_this_turn,
        _ => false,
    }
}

/// 1867's Merge step: Ruby's `auto_actions` (a pass when the minor has nothing to merge with, else the merger pass
/// program).
fn merge_autos(g: &mut SiteGame, minor: &str, now: i64) -> Vec<Value> {
    let Round::Merger(st) = &g.game.round else { return Vec::new() };
    let st = st.clone();
    let programs = |g: &mut SiteGame| -> Vec<Value> {
        let Some(pid) = owner(g, minor) else { return Vec::new() };
        let Some(list) = g.programs.get(&pid).cloned() else { return Vec::new() };
        let meta = g.program_meta.get(&pid).cloned().unwrap_or((None, 0));
        let mut out = Vec::new();
        for prog in list {
            if s(&prog, "type").as_deref() != Some("program_merger_pass") {
                continue;
            }
            let disable_others = prog.get("options").and_then(Value::as_array).is_some_and(|o| o.iter().any(|x| x.as_str() == Some("disable_others")));
            let just_enabled = meta.0.is_some() && meta.0 == Some(g.current_action_id);
            if !just_enabled && disable_others && !g.merger.converts.is_empty() {
                out.push(disable(pid, "Other players have acted and requested to stop", now));
                continue;
            }
            if st.converting || st.merge_major {
                continue;
            }
            let names = prog.pointer("/corporations_by_round/MR").and_then(Value::as_array).cloned().unwrap_or_default();
            if names.iter().any(|x| x.as_str() == Some(minor)) {
                out.push(pass(json!(minor), "corporation", now));
            }
        }
        out
    };
    if st.converting || st.merge_major || !st.merging.is_empty() || g.game.merger_can_convert(minor) {
        return programs(g);
    }
    if g.game.merger_mergeable_candidates(&st, minor).is_empty() {
        return vec![pass(json!(minor), "corporation", now)];
    }
    programs(g)
}

/// Ruby's `round.auto_actions` now: the active step's auto actions for the current entity (empty: none).
pub fn auto_actions(g: &mut SiteGame, now: i64) -> Vec<Value> {
    if g.game.finished {
        return Vec::new();
    }
    if g.game.title == "1867" {
        use crate::steps::{StepEntity, StepKind};
        let Some((kind, e)) = super::g1867::active_step(&mut g.game) else { return Vec::new() };
        return match (kind, e) {
            (StepKind::Token | StepKind::BuyCompanyPreloan, StepEntity::Corp(_)) => {
                match super::g1867::autos::next_auto(&mut g.game) {
                    Some(ent) => vec![pass(json!(ent), "corporation", now)],
                    None => Vec::new(),
                }
            }
            (StepKind::Merge, StepEntity::Corp(c)) => merge_autos(g, &c, now),
            (StepKind::BuySellParShares, StepEntity::Player(p)) => stock_programs(g, p, now),
            _ => Vec::new(),
        };
    }
    match &g.game.round {
        Round::Stock(_) => match g.game.round_state.active_entity_id.player_id() {
            Some(p) => stock_programs(g, p, now),
            None => Vec::new(),
        },
        _ => Vec::new(),
    }
}

/// Ruby's loop (`until (actions = round.auto_actions).empty?`): every auto action applied to `g` as the replay
/// applies a recorded one; returns them as stored.
pub fn run_autos(g: &mut SiteGame, now: i64) -> Result<Vec<Value>, String> {
    let mut out = Vec::new();
    for _ in 0..500 {
        let next = auto_actions(g, now);
        if next.is_empty() {
            return Ok(out);
        }
        for a in next {
            g.apply_auto(&a).map_err(|e| format!("internal: an auto action failed: {e}"))?;
            out.push(a);
        }
    }
    Err("internal: auto actions without end".into())
}
