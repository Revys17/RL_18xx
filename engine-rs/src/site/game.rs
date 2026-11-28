//! `SiteGame`: a site game (GET /api/game/<id>) replayed in engine-rs the way the site's Ruby replays it
//! (docs/WASM_ENGINE_SPIKE.md §3.2): Ruby's undo filter over all raw actions, players by their site ids, recorded
//! auto actions after their action, `message` / `log` / `program_*` / `end_game` handled here (they are not engine
//! actions), and the boundary translations of `json.rs`. Native and testable; `wasm.rs` wraps it for the browser.

use super::json::{action_from_json, swap_city_refs, tile_name, TileSupply, SWAPPED_PREPRINTED_CITIES};
use crate::actions::{Action, GameError};
use crate::game::BaseGame;
use serde_json::Value;
use std::collections::{BTreeMap, HashMap};

/// Error codes the worker maps to its messages (§3.2).
pub fn unsupported(what: impl std::fmt::Display) -> String {
    format!("unsupported: {what}")
}
pub fn refused(what: impl std::fmt::Display) -> String {
    format!("refused: {what}")
}

/// Ruby `Game::Base.filtered_actions`: which raw actions are kept once undo / redo are resolved, and whether undos
/// are still open (the site then offers Redo). Ruby's rules exactly: a bare `undo` takes back the last kept
/// non-message action; `undo` with `action_id` takes back everything after that action (0: everything); `redo`
/// restores the last undo's actions; a new game action closes every open undo; messages are never undone.
pub fn filtered_actions(actions: &[Value]) -> (Vec<bool>, bool) {
    let n = actions.len();
    let mut kept: Vec<bool> = vec![false; n];
    let mut active_undos: Vec<Vec<usize>> = Vec::new();
    let is_msg = |i: usize| actions[i].get("type").and_then(Value::as_str) == Some("message");
    for (index, a) in actions.iter().enumerate() {
        match a.get("type").and_then(Value::as_str).unwrap_or("") {
            "undo" => {
                let undo_to = match a.get("action_id").and_then(Value::as_i64) {
                    Some(0) => 0,
                    Some(id) => {
                        actions.iter().position(|x| x.get("id").and_then(Value::as_i64) == Some(id)).map_or(0, |p| p + 1)
                    }
                    None => (0..index).rev().find(|&i| kept[i] && !is_msg(i)).unwrap_or(0),
                };
                let mut cleared = Vec::new();
                for i in undo_to..index {
                    if kept[i] && !is_msg(i) {
                        kept[i] = false;
                        cleared.push(i);
                    }
                }
                active_undos.push(cleared);
            }
            "redo" => {
                if let Some(list) = active_undos.pop() {
                    for i in list {
                        kept[i] = true;
                    }
                }
            }
            "message" => kept[index] = true,
            _ => {
                active_undos.clear();
                kept[index] = true;
            }
        }
    }
    (kept, !active_undos.is_empty())
}

/// Where a replay stopped: the engine refused a recorded action.
#[derive(Clone, Debug)]
pub struct Stop {
    pub action_id: i64,
    pub index: usize,
    pub message: String,
}

/// One kept site action and what it did (the snapshot is taken by the caller: `view.rs`).
#[derive(Clone, Debug)]
pub struct StepOut {
    pub index: usize,
    pub action_id: i64,
    /// The action as the site stores it (G19 and tile ids in Ruby's numbering)
    pub action: Value,
    pub log: Vec<String>,
}

/// A run as the board needs it: the routes of the last `run_routes` (site JSON) and its OR.
#[derive(Clone, Debug, Default)]
pub struct RunRecord {
    pub routes: Vec<Value>,
    pub turn: u32,
    pub round: u8,
}

pub struct SiteGame {
    pub game: BaseGame,
    /// The game JSON without its actions (id, title, players, settings, status, result...)
    pub meta: Value,
    pub players: Vec<(u32, String, bool)>,
    pub raw: Vec<Value>,
    pub kept: Vec<bool>,
    pub redo: bool,
    /// The next raw action to process
    pub pos: usize,
    /// Steps produced so far (0 = the start)
    pub index: usize,
    pub current_action_id: i64,
    pub last_game_action_id: i64,
    pub stopped: Option<Stop>,
    pub supply: TileSupply,
    /// Preprinted hexes whose cities are still swapped between the engines (`json.rs`)
    pub swapped: Vec<&'static str>,
    /// Each player's programmed actions (Ruby `programmed_actions`), as the site JSON of their `program_*` action
    pub programs: BTreeMap<u32, Vec<Value>>,
    pub manually_ended: bool,
    /// Each corporation's last run (`ports:last_run`) and the hexes it laid in its last operating turn
    pub last_runs: HashMap<String, RunRecord>,
    /// Each corporation's lays by operating turn (the latest few): Ruby's `OperatingInfo#laid_hexes`
    pub laid_hexes: HashMap<String, Vec<(u32, u8, Vec<String>)>>,
    /// How far the log has been handed out (`take_log`)
    log_pos: usize,
    /// The log lines written before the first action (step 0's)
    pub start_log: Vec<String>,
    /// Ruby's `turn_start_action_id` / `last_turn_start_action_id` (`next_turn!`: a new entity's turn or round)
    pub turn_start: i64,
    pub last_turn_start: i64,
    /// When each company last changed hands (the order Ruby's `companies` lists keep)
    pub company_seq: HashMap<String, u64>,
    company_owner: HashMap<String, String>,
    seq: u64,
    /// The players in the order they first sold this stock round (Ruby's `players_sold` keys), and that round's turn
    pub sold_order: Vec<u32>,
    sold_turn: u32,
    /// The actions of the turn in progress (Ruby's `round.current_actions`, for the menu), with the engine's active
    /// step when each was processed
    pub turn_actions: Vec<Value>,
    pub turn_steps: Vec<String>,
    /// Each corporation's share holders in the order they first held a share (Ruby's `share_holders` keys)
    pub holders: HashMap<String, Vec<u32>>,
    /// Ruby's log lines (`site/log.rs`); off, the log stays empty
    pub site_log: bool,
    /// The log lines written so far, in Ruby's words
    pub log: Vec<String>,
    /// The engine's own price of the routes of the last `run_routes` processed (its action id, each route's
    /// revenue), read on the position before it: the shadow key's `last_action` (§5.3)
    pub last_run_prices: Option<(i64, Vec<i32>)>,
    /// Engine actions processed so far (moves and their auto actions)
    pub engine_count: u64,
    /// The action being processed advanced the round's entity (Ruby next_turn!)
    turn_advanced: bool,
    /// 1867's merger round state the engine doesn't keep (Ruby's `converts`, `merge_type`)
    pub merger: super::g1867::MergerRecord,
    /// 1867: the loans each company owed interest on in the last operating round (Ruby's `@interest`, which outlives
    /// the round; the engine keeps it in the round's state)
    pub interest: HashMap<String, u32>,
    /// 1867: the city of each station (Ruby's `token.city`)
    pub token_cities: super::g1867::TokenCities,
    /// 1867's trains exported so far, (id, name): Ruby's depot keeps them (they rust with their type, "The Depot"
    /// in the rust event); engine-rs drops them
    pub exported: Vec<(String, String)>,
    /// 1867's trains of a company nationalized while holding them, (id, name, company): Ruby's `close_corporation`
    /// (a minor: `CLOSED_CORP_TRAINS_REMOVED` only makes them unbuyable) and `reset_corporation` (a major: the old
    /// object keeps them) leave them owned by it, so they rust with their type under its name ("AE x2" in the rust
    /// event); engine-rs drops them
    pub orphan_trains: Vec<(String, String, String)>,
    /// Ruby's `@round.players_history` of the stock round in progress (the Programmer's shenanigan checks,
    /// docs/SITE_ENGINE_RS_STEP3.md §5.3)
    pub stock_history: super::programs::StockHistory,
    /// Each player's program's (id, created_at): what Ruby's `action < program` compares
    pub program_meta: BTreeMap<u32, (Option<i64>, i64)>,
    /// The players who were to act when a manual end finished the game (Ruby's round is left as it was: bot_view's
    /// `acting` of a finished game)
    pub acting_at_end: Option<Vec<u32>>,
    /// Ruby's `train.owner` of every train a company has held: its holder now, the depot once discarded
    /// (`route_check::DEPOT`), nobody once rusted (Ruby's `rust` sets it nil: `route_check::NOBODY`): what a last run's
    /// route is rebuilt for (`train_owner(train)`)
    pub train_owners: HashMap<String, Option<String>>,
}

impl Clone for SiteGame {
    fn clone(&self) -> Self {
        SiteGame {
            game: self.game.clone_for_search(),
            meta: self.meta.clone(),
            players: self.players.clone(),
            raw: self.raw.clone(),
            kept: self.kept.clone(),
            redo: self.redo,
            pos: self.pos,
            index: self.index,
            current_action_id: self.current_action_id,
            last_game_action_id: self.last_game_action_id,
            stopped: self.stopped.clone(),
            supply: self.supply.clone(),
            swapped: self.swapped.clone(),
            programs: self.programs.clone(),
            manually_ended: self.manually_ended,
            last_runs: self.last_runs.clone(),
            laid_hexes: self.laid_hexes.clone(),
            log_pos: self.log_pos,
            start_log: self.start_log.clone(),
            turn_start: self.turn_start,
            last_turn_start: self.last_turn_start,
            company_seq: self.company_seq.clone(),
            company_owner: self.company_owner.clone(),
            seq: self.seq,
            sold_order: self.sold_order.clone(),
            sold_turn: self.sold_turn,
            turn_actions: self.turn_actions.clone(),
            turn_steps: self.turn_steps.clone(),
            holders: self.holders.clone(),
            site_log: self.site_log,
            log: self.log.clone(),
            last_run_prices: self.last_run_prices.clone(),
            engine_count: self.engine_count,
            turn_advanced: self.turn_advanced,
            merger: self.merger.clone(),
            interest: self.interest.clone(),
            token_cities: self.token_cities.clone(),
            exported: self.exported.clone(),
            orphan_trains: self.orphan_trains.clone(),
            stock_history: self.stock_history.clone(),
            program_meta: self.program_meta.clone(),
            acting_at_end: self.acting_at_end.clone(),
            train_owners: self.train_owners.clone(),
        }
    }
}

fn atype(a: &Value) -> &str {
    a.get("type").and_then(Value::as_str).unwrap_or("")
}

/// Ruby's `disable?` of each program (`Engine::Action::Program*`) in a round of this kind (the records' short
/// names: "SR" stock, "OR" operating, "PR" the private auction, "MR" merger).
fn program_disabled(p: &Value, kind: &str) -> bool {
    let indefinite = p.get("indefinite").and_then(Value::as_bool).unwrap_or(false);
    match atype(p) {
        "program_buy_shares" => kind != "SR",
        "program_share_pass" => kind != "SR" && !indefinite,
        "program_auction_bid" | "program_harzbahn_draft_pass" => kind != "PR",
        "program_merger_pass" => kind != "MR",
        "program_independent_mines" => kind != "OR" && !indefinite,
        "program_close_pass" => false,
        // Engine::Action::ProgramEnable#disable?
        _ => true,
    }
}

fn aid(a: &Value) -> i64 {
    a.get("id").and_then(Value::as_i64).unwrap_or(0)
}

impl SiteGame {
    /// The site's game JSON. Nothing processed yet; undo/redo already resolved over all raw actions.
    pub fn load(data: &Value) -> Result<SiteGame, String> {
        let title = data.get("title").and_then(Value::as_str).unwrap_or("");
        if !super::TITLES.contains(&title) {
            return Err(unsupported(format!("title {title}")));
        }
        let rules: Vec<String> = data
            .pointer("/settings/optional_rules")
            .and_then(Value::as_array)
            .map(|xs| xs.iter().filter_map(|x| x.as_str().map(str::to_string)).collect())
            .unwrap_or_default();
        // 1867's optional rule (grid_market) is not in engine-rs: none is supported
        let supported = crate::title::resolve(title).optional_rules();
        if let Some(bad) = rules.iter().find(|r| !supported.contains(&r.as_str())) {
            return Err(unsupported(format!("optional rule {bad}")));
        }
        let mut players = Vec::new();
        for p in data.get("players").and_then(Value::as_array).cloned().unwrap_or_default() {
            let id = p.get("id").and_then(Value::as_u64).ok_or_else(|| unsupported("a player without an id"))? as u32;
            let name = p.get("name").and_then(Value::as_str).unwrap_or("").to_string();
            let bot = p.get("bot").and_then(Value::as_bool).unwrap_or(false);
            players.push((id, name, bot));
        }
        if title == "1867" && players.len() == 2 {
            // Ruby's setup_for_2_players (fewer trains, two privates removed at random): engine-rs has no 2-player
            // 1867 (docs/SITE_ENGINE_RS_1867.md §3.1)
            return Err(unsupported("1867 for 2 players"));
        }
        if !(2..=6).contains(&players.len()) {
            return Err(unsupported(format!("{} players", players.len())));
        }
        let ids: Vec<u32> = players.iter().map(|p| p.0).collect();
        let names: HashMap<u32, String> = players.iter().map(|p| (p.0, p.1.clone())).collect();
        let mut game = BaseGame::build_with_rules(title, ids, names, rules);
        game.site_rules = true;
        game.site = Some(Box::default());
        let supply = if title == "1830" {
            let mut counts: Vec<(String, u32)> =
                game.tile_counts_remaining.iter().map(|(k, v)| (k.clone(), *v)).collect();
            // Ruby's TILES order (1830's are numbered in order)
            counts.sort_by_key(|(k, _)| (k.parse::<u32>().unwrap_or(u32::MAX), k.clone()));
            TileSupply::new(&counts)
        } else {
            // Ruby's TILES order is the title's own list (g_1867/map.rb); 'unlimited' tiles (7, 8, 9) are one
            // instance at a time (Game::Base#add_extra_tile)
            let counts: Vec<(String, u32)> =
                game.title_def().tile_counts().into_iter().map(|(k, v)| (k.to_string(), v)).collect();
            TileSupply::new(&counts)
        };
        let raw: Vec<Value> = data.get("actions").and_then(Value::as_array).cloned().unwrap_or_default();
        let (kept, redo) = filtered_actions(&raw);
        let mut meta = data.clone();
        if let Some(m) = meta.as_object_mut() {
            m.remove("actions");
        }
        let mut g = SiteGame {
            game,
            meta,
            players,
            raw,
            kept,
            redo,
            pos: 0,
            index: 0,
            current_action_id: 0,
            last_game_action_id: 0,
            stopped: None,
            supply,
            // G19's two cities are numbered the other way round in 1830 only (1867's preprinted cities agree)
            swapped: if title == "1830" { SWAPPED_PREPRINTED_CITIES.to_vec() } else { Vec::new() },
            programs: BTreeMap::new(),
            manually_ended: false,
            last_runs: HashMap::new(),
            laid_hexes: HashMap::new(),
            log_pos: 0,
            start_log: Vec::new(),
            turn_start: 0,
            last_turn_start: 0,
            company_seq: HashMap::new(),
            company_owner: HashMap::new(),
            seq: 0,
            sold_order: Vec::new(),
            sold_turn: 0,
            turn_actions: Vec::new(),
            turn_steps: Vec::new(),
            holders: HashMap::new(),
            site_log: true,
            log: Vec::new(),
            last_run_prices: None,
            engine_count: 0,
            turn_advanced: false,
            merger: Default::default(),
            interest: HashMap::new(),
            token_cities: Default::default(),
            exported: Vec::new(),
            orphan_trains: Vec::new(),
            stock_history: Default::default(),
            program_meta: BTreeMap::new(),
            acting_at_end: None,
            train_owners: HashMap::new(),
        };
        g.log_start();
        g.start_log = g.take_log();
        Ok(g)
    }

    pub fn finished(&self) -> bool {
        self.game.finished
    }

    pub fn total(&self) -> usize {
        self.raw.len()
    }

    pub fn done(&self) -> bool {
        self.stopped.is_some() || self.pos >= self.raw.len()
    }

    /// Process the next raw action. Returns its step when it is a kept action that makes one (not a chat message,
    /// an undo or a redo, nor an action taken back). Sets `stopped` when the engine refuses it.
    pub fn process_next(&mut self) -> Option<StepOut> {
        if self.done() {
            return None;
        }
        let i = self.pos;
        self.pos += 1;
        let a = self.raw[i].clone();
        let id = aid(&a);
        let t = atype(&a).to_string();
        if !self.kept[i] {
            self.current_action_id = id;
            return None;
        }
        if t == "message" {
            // Ruby's process_action runs a message's recorded auto actions too (docs/SITE_ENGINE_RS_1867.md §3.1);
            // a message makes no step
            if a.get("auto_actions").and_then(Value::as_array).is_some_and(|x| !x.is_empty()) {
                if let Err(e) = self.apply_autos(&a) {
                    self.stopped = Some(Stop { action_id: id, index: self.index, message: e });
                    return None;
                }
            }
            self.current_action_id = id;
            return None;
        }
        if t == "undo" || t == "redo" {
            // only reached when appended live (load resolved the recorded ones): the caller re-opens the game
            self.current_action_id = id;
            return None;
        }
        self.turn_advanced = false;
        match self.apply(&a) {
            Ok(()) => {
                // Ruby next_turn!: once per action whose processing advanced the round's entity
                if self.turn_advanced {
                    self.last_turn_start = self.turn_start;
                    self.turn_start = id;
                }
                self.current_action_id = id;
                self.last_game_action_id = id;
                self.index += 1;
                Some(StepOut { index: self.index, action_id: id, action: a, log: self.take_log() })
            }
            Err(e) => {
                self.stopped = Some(Stop { action_id: id, index: self.index, message: e });
                None
            }
        }
    }

    /// Process every raw action up to and including `until` (all of them when None).
    pub fn run_to(&mut self, until: Option<i64>) {
        while !self.done() {
            if let Some(u) = until {
                if aid(&self.raw[self.pos]) > u {
                    break;
                }
            }
            self.process_next();
        }
    }

    /// A kept site action: the engine's move, its recorded auto actions after it, or one of the site's own kinds.
    pub fn apply(&mut self, a: &Value) -> Result<(), String> {
        let t = atype(a).to_string();
        if t == "log" {
            if let Some(m) = a.get("message").and_then(Value::as_str) {
                self.log_push(m.to_string());
            }
            return Ok(());
        }
        if t.starts_with("program_") {
            // Ruby's master-mode line for a program set for someone else's seat
            if let (Some(user), Ok(e)) = (a.get("user").and_then(Value::as_u64), super::json::entity_of(a)) {
                if e.parse::<u64>().ok() != Some(user) {
                    let name = self.players.iter().find(|p| p.0 as u64 == user).map(|p| p.1.clone()).unwrap_or_else(|| "Owner".into());
                    self.log_push(format!("• Action({t}) via Master Mode by: {name}"));
                }
            }
            self.note_program(a);
            return self.apply_autos(a);
        }
        if t == "end_game" {
            if self.game.finished {
                return Err(refused("Game is already finished"));
            }
            // Ruby process_single_action's master-mode line: a user who isn't the action's player (end_game has none)
            if let Some(user) = a.get("user").and_then(Value::as_u64) {
                let player = super::json::entity_of(a).ok().and_then(|e| e.parse::<u64>().ok());
                if player != Some(user) {
                    let name = self.players.iter().find(|p| p.0 as u64 == user).map(|p| p.1.clone()).unwrap_or_else(|| "Owner".into());
                    self.log_push(format!("• Action({t}) via Master Mode by: {name}"));
                }
            }
            let who = self.entity_name(a);
            self.log_push(format!("Game ended manually by {who}"));
            self.acting_at_end = Some(super::authority::ruby_active_players(self));
            self.manually_ended = true;
            self.game.end_game();
            self.log_game_over();
            return Ok(());
        }
        if t == "run_routes" {
            let prices = if self.game.title == "1867" {
                super::g1867::price_routes(self, a).unwrap_or_default()
            } else {
                super::shadow::price_routes(self, a).unwrap_or_default()
            };
            self.last_run_prices = Some((aid(a), prices));
        }
        self.apply_engine(a).map_err(|e| e.message)?;
        self.apply_autos(a)
    }

    /// The moves a recorded action set off for players with programs (its `auto_actions`), in order.
    fn apply_autos(&mut self, a: &Value) -> Result<(), String> {
        if let Some(Value::Array(autos)) = a.get("auto_actions") {
            for au in autos.clone() {
                self.apply_auto(&au)?;
            }
        }
        Ok(())
    }

    /// One auto action: a program's note, or an engine action.
    pub fn apply_auto(&mut self, au: &Value) -> Result<(), String> {
        let t = atype(au).to_string();
        if t.starts_with("program_") {
            self.note_program(au);
            Ok(())
        } else {
            self.apply_engine(au).map_err(|e| format!("auto action {t}: {}", e.message))
        }
    }

    fn entity_name(&self, a: &Value) -> String {
        let e = super::json::entity_of(a).unwrap_or_default();
        self.players.iter().find(|p| p.0.to_string() == e).map(|p| p.1.clone()).unwrap_or(e)
    }

    /// An engine action in the site's JSON (Ruby's numbering): translated, processed, recorded.
    pub fn apply_engine(&mut self, a: &Value) -> Result<(), GameError> {
        let t = atype(a).to_string();
        let translated = if self.swapped.is_empty() { a.clone() } else { swap_city_refs(a, &self.swapped) };
        let action = action_from_json(&translated)?;
        let before = self.site_before(&action);
        // the operating turn the action is made in (a company's last lay can end the round)
        let tr_before = self.turn_round();
        let was_finished = self.game.finished;
        // the log's view of the position before (a copy without the map and the records)
        let log_before = self.site_log.then(|| super::log::before(self, a));
        let key_before = self.turn_key();
        let step_before = self.game.active_step_type();
        let rounds_before = self.finished_rounds();
        let is_1867 = self.game.title == "1867";
        let or_before = match &self.game.round {
            crate::rounds::Round::Operating(s) if is_1867 => Some((self.game.turn, s.round_num)),
            _ => None,
        };
        let parred_before: Vec<String> = if is_1867 {
            self.game.corporations.iter().filter(|c| c.ipo_price.is_some() && !c.closed).map(|c| c.sym.clone()).collect()
        } else {
            Vec::new()
        };
        let merger_before = matches!(self.game.round, crate::rounds::Round::Merger(_));
        let depot_before: Vec<(String, String)> =
            if is_1867 { self.game.depot.trains.iter().map(|t| (t.id.clone(), t.name.clone())).collect() } else { Vec::new() };
        let corp_trains_before: Vec<(String, String, String)> = if is_1867 {
            self.game.corporations.iter().flat_map(|c| c.trains.iter().map(|t| (t.id.clone(), t.name.clone(), c.sym.clone()))).collect()
        } else {
            Vec::new()
        };
        let phase_before = self.game.phase.name.clone();
        // the company operating while its LoanOperations is still ahead in its turn: if this action ends the operating
        // round, its LoanOperations ran inside it (G1867::Step::LoanOperations#skip!: interest, repayments, then
        // `calculate_corporation_interest`) and the round's own count went with the round
        let loan_ops_ahead: Option<String> = match &self.game.round {
            crate::rounds::Round::Operating(s) if is_1867 => {
                use crate::rounds::OperatingStep as P;
                matches!(s.step, P::RedeemShares | P::LayTile | P::PlaceToken | P::RunRoutes | P::Dividend | P::BuyCompanyPreloan)
                    .then(|| s.current_corp_sym().map(str::to_string))
                    .flatten()
            }
            _ => None,
        };
        let slots_before = if is_1867 { super::g1867::slot_owners(&self.game) } else { HashMap::new() };
        let ranks_before: HashMap<String, super::g1867::CorpRank> = if is_1867 {
            self.game.corporations.iter().map(|c| (c.sym.clone(), super::g1867::corp_rank(&self.game, c))).collect()
        } else {
            HashMap::new()
        };
        let merge_state_before = match &self.game.round {
            crate::rounds::Round::Merger(s) => Some((s.converting, s.merge_major)),
            _ => None,
        };
        let converted_before = match &self.game.round {
            crate::rounds::Round::Merger(s) => s.converted.clone(),
            _ => None,
        };
        // a 1867 stock round's state before the action: a minor's auction under way, the pending home station's company
        let stock_before = match &self.game.round {
            crate::rounds::Round::Stock(s) if is_1867 => {
                Some((s.bid_auction.is_some(), s.pending_home_tokens.first().map(|(sym, _)| sym.clone())))
            }
            _ => None,
        };
        // the companies' owners' order before a 1867 merger (Ruby `Corporation#<=>`: price, column, row, name), keyed
        // by the owner's entity id
        let corp_ranks = (is_1867 && matches!(action, Action::Merge { .. })).then(|| {
            self.game
                .corporations
                .iter()
                .filter(|c| c.share_price.is_some())
                .map(|c| (crate::entities::EntityId::corporation(&c.sym).0, super::g1867::corp_rank(&self.game, c)))
                .collect::<HashMap<_, _>>()
        });
        // the Programmer's history (Ruby's players_history): the stock turn before, whose move it is, 1867's minors
        let stock_turn_before = super::programs::stock_turn(self);
        let hist_player = super::authority::entity_player(self, a).flatten();
        let auction_before = matches!(&self.game.round, crate::rounds::Round::Stock(s) if s.bid_auction.is_some());
        let unparred_minors: Vec<String> = if is_1867 && stock_turn_before.is_some() {
            self.game.corporations.iter().filter(|c| c.corp_type == crate::title::CorpType::Minor && c.ipo_price.is_none()).map(|c| c.sym.clone()).collect()
        } else {
            Vec::new()
        };
        self.game.process_action_internal(&action)?;
        self.engine_count += 1;
        if is_1867 {
            self.records_1867(&translated, &action, or_before, &parred_before);
            self.merger.update(merge_state_before, &action, &self.game.round);
            let rank = |sym: &str| ranks_before.get(sym).cloned().unwrap_or((i32::MAX, 0, 0, 0, sym.to_string()));
            self.token_cities.update(&slots_before, &rank, &self.game);
            // the depot's trains gone without a buyer: exported (Depot#export!; a type's last trains rust only once
            // sold, so nothing unsold rusts)
            for (id, name) in depot_before {
                let g = &self.game;
                let gone = !g.depot.trains.iter().chain(g.depot.discarded.iter()).any(|t| t.id == id)
                    && !g.corporations.iter().any(|c| c.trains.iter().any(|t| t.id == id));
                if gone {
                    self.exported.push((id, name));
                }
            }
            // a company's trains gone with it (nationalized: closed or reset), not rusted by this move's new phase
            let rusting: Vec<&str> = if self.game.phase.name != phase_before {
                let on = self.game.title_def().phases().into_iter().find(|p| p.name == self.game.phase.name).and_then(|p| p.on);
                self.game.title_def().trains().iter().filter(|t| on.is_some() && t.rusts_on == on).map(|t| t.name).collect()
            } else {
                Vec::new()
            };
            for (id, name, sym) in corp_trains_before {
                let g = &self.game;
                // (a company still running keeps whatever it hasn't sold, and what it sold is someone's)
                let alive = g.corp_idx.get(sym.as_str()).is_some_and(|&ci| !g.corporations[ci].closed && g.corporations[ci].ipo_price.is_some());
                if alive {
                    continue;
                }
                let gone = !g.depot.trains.iter().chain(g.depot.discarded.iter()).any(|t| t.id == id)
                    && !g.corporations.iter().any(|c| c.trains.iter().any(|t| t.id == id));
                if gone && !rusting.contains(&name.as_str()) && !self.orphan_trains.iter().any(|o| o.0 == id) {
                    self.orphan_trains.push((id, name, sym));
                }
            }
            // the last station a merged major takes off (Ruby ReduceTokens#move_tokens_to_surviving): its tokens in use
            // first, in their order, the others after them
            if let Action::RemoveToken { entity_id, .. } = &action {
                let removing = matches!(&self.game.round, crate::rounds::Round::Merger(s) if s.corporations_removing_tokens.as_ref().is_some_and(|l| l.contains(entity_id)));
                if !removing {
                    self.token_cities.compact(entity_id);
                }
            }
            if let crate::rounds::Round::Operating(s) = &self.game.round {
                self.interest = s.interest_snapshot.clone();
            } else if let Some(c) = loan_ops_ahead {
                // Ruby's @interest[corporation] after its LoanOperations: the loans it holds once repaid (nothing after
                // that step changes them in a turn that ended the round: its BuyTrain and BuyCompany were skipped)
                if let Some(&ci) = self.game.corp_idx.get(&c) {
                    self.interest.insert(c, self.game.corporations[ci].loans);
                }
            }
        }
        self.check_programs(rounds_before, was_finished);
        // the turn in progress (Ruby's round.current_actions), move by move, auto actions included: Ruby advances the
        // round's entity index (next_turn!) also when it comes back round to the same player (the others skipped,
        // "has no valid actions and passes"), so in a stock round a turn that hasn't acted is a new one
        // A private's move (M&H's exchange) is the Exchange step's: it never starts a turn (Ruby's step calls no
        // next_turn!, and the stock turn's current_actions are BuySellParShares's own)
        let key_after = self.turn_key();
        let by_company = a.get("entity_type").and_then(Value::as_str) == Some("company");
        let stock_new = !is_1867 && !by_company && matches!(&self.game.round, crate::rounds::Round::Stock(s) if !s.acted_this_turn);
        // 1867's stock round (Ruby next_turn!, once an action): a pass outside a minor's auction (after_process ->
        // next_entity!), an auction settled (BuySellParShares#win_bid's goto_entity!), a won minor's home station
        // placed (the step then has nothing left: next_entity!); the round's or the turn's change as everywhere. A
        // minor's auction and its station put down when it goes up stay within the turn of the player who put it up.
        let stock_1867 = is_1867
            && stock_before.as_ref().is_some_and(|(auction, _)| {
                let after_auction = matches!(&self.game.round, crate::rounds::Round::Stock(s) if s.bid_auction.is_some());
                let pass_outside = matches!(action, Action::Pass { .. }) && !*auction;
                let settled = (*auction || matches!(action, Action::CorporationBid { .. })) && !after_auction;
                let won_station = matches!(action, Action::PlaceToken { .. })
                    && stock_before.as_ref().and_then(|(_, pending)| pending.clone()).is_some_and(|sym| {
                        self.game.corp_idx.get(sym.as_str()).is_some_and(|&ci| {
                            let c = &self.game.corporations[ci];
                            c.corp_type == crate::title::CorpType::Minor && c.owner_id.is_player()
                        })
                    });
                pass_outside || settled || won_station
            });
        let key_changed = key_after != key_before
            // 1867's merger round: a merger of minors done (Merge#finish_merge_to_major; a conversion starts no turn)
            || (merger_before
                && converted_before.is_none()
                && matches!(&self.game.round, crate::rounds::Round::Merger(s) if s.converted.is_some() && !s.merge_type_convert));
        let advanced = key_changed || stock_new || stock_1867;
        if advanced {
            self.turn_advanced = true;
            self.turn_actions.clear();
            self.turn_steps.clear();
        } else {
            self.turn_actions.push(a.clone());
            self.turn_steps.push(step_before);
        }
        let won_minor = unparred_minors.iter().find_map(|sym| {
            let &ci = self.game.corp_idx.get(sym.as_str())?;
            let c = &self.game.corporations[ci];
            if c.ipo_price.is_some() {
                c.owner_id.player_id().map(|p| (p, sym.clone()))
            } else {
                None
            }
        });
        super::programs::observe(self, stock_turn_before, a, hist_player, won_minor, advanced, auction_before);
        // a bankruptcy ends the game with Ruby's round where it was: its active entity's player, the bankrupt
        // president, is who `active_players_id` names (bot_view's `acting` of the finished game)
        if !was_finished && self.game.finished && matches!(action, Action::Bankrupt { .. }) {
            self.acting_at_end = hist_player.map(|p| vec![p]);
        }
        self.site_after(&t, a, &action, before, corp_ranks, tr_before);
        if let Some(b) = log_before {
            let mut lines = super::log::explain(self, &b, a);
            lines.extend(super::log::turn_start(self, &b));
            if !was_finished && self.game.finished {
                lines.extend(super::log::game_end_1867(self, &b));
            }
            for l in lines {
                self.log_push(l);
            }
        }
        if !was_finished && self.game.finished {
            self.log_game_over();
        }
        Ok(())
    }

    /// What a move changes that the site layer keeps track of, read before the engine processes it.
    fn site_before(&self, action: &Action) -> Option<String> {
        match action {
            Action::LayTile { hex_id, .. } => self.game.hex_ref(hex_id).map(|h| h.tile.name.clone()),
            _ => None,
        }
    }

    /// 1867's records the engine keeps no hook for (docs/SITE_ENGINE_RS_1867.md §3.1), around one engine action:
    /// a minor's automatic payout (`G1867::Step::Dividend#skip!`: `payout` when its run paid, processed inside its
    /// `run_routes`; the revenue-0 withhold is H2's) and a nationalized major's fresh start (`reset_corporation`: its
    /// operating history empty).
    fn records_1867(&mut self, raw: &Value, action: &Action, or_before: Option<(u32, u8)>, parred_before: &[String]) {
        if let (Action::RunRoutes { entity_id, .. }, Some((turn, round))) = (action, or_before) {
            let minor = self.game.corp_idx.get(entity_id.as_str()).is_some_and(|&ci| {
                self.game.corporations[ci].corp_type == crate::title::CorpType::Minor
            });
            // each route's recorded revenue, else the engine's own price of it (what the engine paid)
            let prices = self.last_run_prices.as_ref().filter(|(id, _)| *id == aid(raw)).map(|(_, p)| p.clone());
            let revenue: i32 = raw
                .get("routes")
                .and_then(Value::as_array)
                .map(|rs| {
                    rs.iter()
                        .enumerate()
                        .map(|(k, r)| match r.get("revenue").and_then(Value::as_i64) {
                            Some(v) => v as i32,
                            None => prices.as_ref().and_then(|p| p.get(k)).copied().unwrap_or(0),
                        })
                        .sum::<i32>()
                })
                .unwrap_or(0);
            if minor && revenue > 0 {
                self.game.site_note_operated_at(entity_id, turn, round, revenue, "payout", false);
            }
        }
        for sym in parred_before {
            let Some(&ci) = self.game.corp_idx.get(sym.as_str()) else { continue };
            let c = &self.game.corporations[ci];
            if !c.closed && c.ipo_price.is_none() {
                self.game.site_forget_operated(sym);
            }
        }
    }

    fn site_after(
        &mut self,
        _t: &str,
        raw: &Value,
        action: &Action,
        before: Option<String>,
        corp_ranks: Option<HashMap<String, super::g1867::CorpRank>>,
        tr_before: (u32, u8),
    ) {
        let (turn, round) = self.turn_round();
        for c in &self.game.corporations {
            let list = self.holders.entry(c.sym.clone()).or_default();
            let mut new: Vec<(u64, u32)> = c
                .shares
                .iter()
                .filter_map(|s| s.owner.player_id().map(|p| (s.acquired_seq, p)))
                .filter(|(_, p)| !list.contains(p))
                .collect();
            new.sort();
            for (_, p) in new {
                if !list.contains(&p) {
                    list.push(p);
                }
            }
        }
        // the companies that changed hands, in the order Ruby moves them: one at a time but for 1867's mergers, where
        // Merge#move_assets takes each merging minor's companies in turn, the minors in Corporation#<=> order
        let mut moved: Vec<(Option<super::g1867::CorpRank>, u64, usize)> = Vec::new();
        for (i, c) in self.game.companies.iter().enumerate() {
            let owner = c.owner.0.clone();
            if self.company_owner.get(&c.sym) != Some(&owner) {
                let prev_owner = self.company_owner.get(&c.sym).cloned().unwrap_or_default();
                let rank = corp_ranks.as_ref().and_then(|r| r.get(&prev_owner).cloned());
                // (within a merging minor, its own list's order; elsewhere the companies' order, as step 1)
                let within = if rank.is_some() { self.company_seq.get(&c.sym).copied().unwrap_or(0) } else { 0 };
                moved.push((rank, within, i));
            }
        }
        moved.sort();
        for (_, _, i) in moved {
            let c = &self.game.companies[i];
            self.seq += 1;
            self.company_seq.insert(c.sym.clone(), self.seq);
            self.company_owner.insert(c.sym.clone(), c.owner.0.clone());
        }
        if let crate::rounds::Round::Stock(_) = &self.game.round {
            if self.sold_turn != self.game.turn {
                self.sold_turn = self.game.turn;
                self.sold_order.clear();
            }
            if let Action::SellShares { entity_id, .. } = action {
                if let Ok(pid) = entity_id.parse::<u32>() {
                    if !self.sold_order.contains(&pid) {
                        self.sold_order.push(pid);
                    }
                }
            }
        }
        // (a train no company holds any more rusted: Ruby's `rust` sets its owner to nil; a discarded one is the depot's)
        let mut held: std::collections::HashSet<&str> = std::collections::HashSet::new();
        for c in &self.game.corporations {
            for t in &c.trains {
                held.insert(t.id.as_str());
                if self.train_owners.get(&t.id).and_then(|o| o.as_deref()) != Some(c.sym.as_str()) {
                    self.train_owners.insert(t.id.clone(), Some(c.sym.clone()));
                }
            }
        }
        let discarded: std::collections::HashSet<&str> = self.game.depot.discarded.iter().map(|t| t.id.as_str()).collect();
        for (id, owner) in self.train_owners.iter_mut() {
            if !held.contains(id.as_str()) {
                let now = if discarded.contains(id.as_str()) { super::route_check::DEPOT } else { super::route_check::NOBODY };
                if owner.as_deref() != Some(now) {
                    *owner = Some(now.to_string());
                }
            }
        }
        match action {
            Action::LayTile { hex_id, tile_id, entity_id, .. } => {
                self.supply.laid(tile_id, before.as_deref().unwrap_or(""));
                self.swapped.retain(|h| h != hex_id);
                let corp = self.operator_of(entity_id);
                let (lt, lr) = tr_before;
                let list = self.laid_hexes.entry(corp).or_default();
                if list.last().is_none_or(|e| (e.0, e.1) != (lt, lr)) {
                    list.push((lt, lr, Vec::new()));
                    if list.len() > 4 {
                        list.remove(0);
                    }
                }
                if let Some(e) = list.last_mut() {
                    e.2.push(hex_id.clone());
                }
            }
            Action::RunRoutes { entity_id, .. } => {
                let routes = raw.get("routes").and_then(Value::as_array).cloned().unwrap_or_default();
                self.last_runs.insert(entity_id.clone(), RunRecord { routes, turn, round });
            }
            _ => {}
        }
    }

    /// The corporation a move is made for (a private's ability lays for its owner).
    pub fn operator_of(&self, entity: &str) -> String {
        if let Some(&ci) = self.game.company_idx.get(entity) {
            if let Some(sym) = self.game.companies[ci].owner.corp_sym() {
                return sym.to_string();
            }
        }
        entity.to_string()
    }

    /// What identifies whose turn it is (Ruby calls `next_turn!` when the round or its entity index changes).
    fn turn_key(&self) -> (u32, &'static str, u8, i64) {
        // (1867's merger round: its minors in turn, Ruby's @entity_index)
        let idx = match &self.game.round {
            crate::rounds::Round::Merger(s) => s.entity_index as i64,
            // 1867's SingleItemAuction: a lot won moves the round on (goto_entity! + next_entity_index!), whoever wins
            crate::rounds::Round::Auction(s) if s.single_item.is_some() => s.remaining_companies.len() as i64,
            _ => super::view::round_order(self).1,
        };
        let kind = self.game.round.round_type_str();
        (self.game.turn, kind, self.game.round.round_num(), idx)
    }

    /// Ruby's (turn, round_num)
    pub fn turn_round(&self) -> (u32, u8) {
        (self.game.turn, self.game.round.round_num())
    }

    /// A `program_*` action: Ruby's Step::Program (`ALLOW_MULTIPLE_PROGRAMS` false, 1830 as the base): enabling a
    /// program first drops the player's last one (`remove_programmed_action`: `programmed_actions[entity].last`), so a
    /// player has one program at a time; `program_disable` drops the last one.
    fn note_program(&mut self, a: &Value) {
        let pid = match super::json::entity_of(a).ok().and_then(|e| e.parse::<u32>().ok()) {
            Some(p) => p,
            None => return,
        };
        let t = atype(a).to_string();
        if t == "program_disable" {
            self.program_meta.remove(&pid);
        } else {
            let created = a.get("created_at").and_then(Value::as_i64).unwrap_or(0);
            self.program_meta.insert(pid, (a.get("id").and_then(Value::as_i64), created));
        }
        let list = self.programs.entry(pid).or_default();
        list.pop();
        if t != "program_disable" {
            let mut p = a.clone();
            if let Some(m) = p.as_object_mut() {
                for k in ["id", "user", "created_at"] {
                    m.remove(k);
                }
            }
            list.push(p);
        }
        if list.is_empty() {
            self.programs.remove(&pid);
        }
    }

    /// How many rounds have finished so far (the records' `store_player_info` entries: one per
    /// `transition_to_next_round!`, and `end_game!`'s).
    fn finished_rounds(&self) -> usize {
        self.game.site.as_ref().map_or(0, |r| r.player_history.len())
    }

    /// Ruby's `check_programmed_actions`, which `transition_to_next_round!` runs after each round it starts: the
    /// programs whose `disable?` holds in the new round are dropped. One move can start several rounds (a stock
    /// round that ends into an operating round where no company operates, and on into the next stock round): each
    /// is checked in turn. The rounds a move started are read from the records: every round that finished during
    /// it but the first was started by it, and so is the round now unless the move ended the game (`end_game!`
    /// writes the last round's entry and starts nothing, nor checks).
    fn check_programs(&mut self, rounds_before: usize, was_finished: bool) {
        let Some(rec) = self.game.site.as_ref() else { return };
        let mut started: Vec<String> = rec.player_history.iter().skip(rounds_before).skip(1).map(|e| e.round.clone()).collect();
        if rec.player_history.len() > rounds_before && !(self.game.finished && !was_finished) {
            started.push(super::records::round_short(&self.game.round).to_string());
        }
        for kind in started {
            for list in self.programs.values_mut() {
                list.retain(|p| !program_disabled(p, &kind));
            }
            self.programs.retain(|_, l| !l.is_empty());
        }
    }

    /// Any player has a program running (the server would expect the auto actions it sets off).
    pub fn programs_enabled(&self) -> bool {
        !self.programs.is_empty()
    }

    // ------------------------------------------------------------------------------------------ log

    fn log_start(&mut self) {
        for l in super::log::start(self) {
            self.log_push(l);
        }
    }

    pub fn log_push(&mut self, line: String) {
        if self.site_log {
            self.log.push(line);
        }
    }

    fn log_game_over(&mut self) {
        let res = self.result();
        let scores: Vec<String> = res
            .iter()
            .map(|(id, v)| {
                let name = self.players.iter().find(|p| p.0 == *id).map(|p| p.1.clone()).unwrap_or_default();
                format!("{name} (${v})")
            })
            .collect();
        self.log_push(format!("-- Game over: {} --", scores.join(", ")));
    }

    /// The log lines written since the last call.
    pub fn take_log(&mut self) -> Vec<String> {
        let out = self.log.get(self.log_pos..).map(<[String]>::to_vec).unwrap_or_default();
        self.log_pos = self.log.len();
        out
    }

    // ------------------------------------------------------------------------------------------ results

    /// Ruby `result`: player id -> value, highest first (ties keep the engine's player order reversed, as Ruby's
    /// `sort_by.reverse` does).
    pub fn result(&self) -> Vec<(u32, i32)> {
        let r = self.game.calculate_results();
        let mut out: Vec<(u32, i32)> = self.game.players.iter().map(|p| (p.id, r.get(&p.id).copied().unwrap_or(0))).collect();
        out.sort_by_key(|x| x.1);
        out.reverse();
        out
    }

    /// Ruby `game_end_reason` once finished.
    pub fn end_reason(&self) -> Option<&'static str> {
        if !self.game.finished {
            return None;
        }
        if self.manually_ended {
            Some("manually_ended")
        } else if self.game.title == "1867" {
            // Ruby game_end_check over GAME_END_CHECK {bank: :current_or, final_phase: :one_more_full_or_set}: the
            // latched trigger whose timing comes first in GAME_END_TIMING_PRIORITY (1867: the final phase's)
            let title = self.game.title_def();
            let final_phase = title.phases().last().is_some_and(|p| p.name == self.game.phase.name);
            let mut latched: Vec<(&'static str, crate::title::GameEndTiming)> = Vec::new();
            if self.game.bank_broken {
                latched.push(("bank", title.bank_game_end_timing()));
            }
            if final_phase {
                latched.push(("final_phase", crate::title::GameEndTiming::OneMoreFullOrSet));
            }
            let priority = title.game_end_timing_priority();
            Some(
                latched
                    .into_iter()
                    .min_by_key(|(_, t)| priority.iter().position(|p| p == t).unwrap_or(usize::MAX))
                    .map_or("unknown", |(r, _)| r),
            )
        } else if self.game.players.iter().any(|p| p.bankrupt) {
            Some("bankrupt")
        } else if self.game.bank_broken {
            Some("bank")
        } else {
            Some("unknown")
        }
    }

    // ------------------------------------------------------------------------------------------ live

    /// A live action as the site stores it (with its id): processed, its step returned. An undo or redo, or an id
    /// that isn't after the last one, answers `resync:` (the worker re-opens the game).
    pub fn append(&mut self, a: &Value) -> Result<Option<StepOut>, String> {
        let t = atype(a);
        if t == "undo" || t == "redo" {
            return Err("resync: an undo or redo re-opens the game".into());
        }
        if aid(a) <= self.current_action_id {
            return Err(format!("resync: action {} is not after {}", aid(a), self.current_action_id));
        }
        if !self.done() {
            return Err("resync: the replay hasn't finished".into());
        }
        if self.stopped.is_some() {
            return Err("resync: the game stopped".into());
        }
        self.raw.push(a.clone());
        self.kept.push(true);
        let step = self.process_next();
        if let Some(s) = self.stopped.take() {
            // the engine refused it: forget it
            self.raw.pop();
            self.kept.pop();
            self.pos -= 1;
            return Err(refused(s.message));
        }
        Ok(step)
    }

    /// A move as the page posts it (WasmGame API 3, docs/SITE_ENGINE_RS_STEP3.md §10): checked as the server's
    /// authority checks it (1830: the site menu or Ruby's cross-player set; 1867: the generator; a run: the route
    /// check, any valid run, priced by the engine), then processed with Ruby's auto actions (the Programmer, 1867's
    /// step passes) made here. Returns the action as the server will store it (id = the current id + 1, Rust's
    /// `auto_actions`) and its step. Chat, `log`, `end_game` and programs as the site stores them.
    pub fn act(&mut self, a: &Value) -> Result<(Value, Option<StepOut>), String> {
        let t = atype(a);
        if t == "undo" || t == "redo" {
            return Err("resync: an undo or redo re-opens the game".into());
        }
        if !self.done() {
            return Err("resync: the replay hasn't finished".into());
        }
        if let Some(st) = &self.stopped {
            return Err(format!("resync: the game stopped at action {}", st.action_id));
        }
        let now = super::authority::now_secs();
        let d = super::authority::decide(self, a, None, now)?;
        Ok((d.action, d.step))
    }

    /// 1867: the loans a company owes interest on (Ruby `loans_due_interest`: the operating round's count, which
    /// outlives the round; a company started afresh owes none).
    pub fn interest_due(&self, sym: &str) -> u32 {
        let reset = self.game.corp_idx.get(sym).is_some_and(|&ci| self.game.corporations[ci].ipo_price.is_none());
        if reset {
            return 0;
        }
        match &self.game.round {
            crate::rounds::Round::Operating(s) => s.interest_snapshot.get(sym).copied().unwrap_or(0),
            _ => self.interest.get(sym).copied().unwrap_or(0),
        }
    }

    /// Whether the turn in progress has an action of this type processed at this engine step.
    pub fn turn_acted_at(&self, t: Option<&str>, step: &str) -> bool {
        self.turn_actions.iter().zip(self.turn_steps.iter()).any(|(a, s)| t.map_or(true, |t| atype(a) == t) && s == step)
    }

    /// Whether the turn in progress has an action of this type (a pass description's "Done").
    pub fn turn_acted(&self, t: &str) -> bool {
        self.turn_actions.iter().any(|a| atype(a) == t)
    }

    /// The tile instance Ruby would lay next for a tile name.
    pub fn next_tile(&self, name: &str) -> Option<String> {
        self.supply.next(name).map(str::to_string)
    }

    /// A position from the site's tile id: is the tile name a laid instance?
    pub fn tile_name_of(id: &str) -> &str {
        tile_name(id).unwrap_or(id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn acts(xs: &[(i64, &str, Option<i64>)]) -> Vec<Value> {
        xs.iter()
            .map(|(id, t, to)| match to {
                Some(to) => json!({"id": id, "type": t, "action_id": to}),
                None => json!({"id": id, "type": t}),
            })
            .collect()
    }

    #[test]
    fn filtered_actions_follow_ruby() {
        // a bare undo takes back the last game action, not the chat after it
        let a = acts(&[(1, "bid", None), (2, "pass", None), (3, "message", None), (4, "undo", None)]);
        assert_eq!(filtered_actions(&a), (vec![true, false, true, false], true));
        // redo restores it; a new move closes the open undos
        let b = acts(&[(1, "bid", None), (2, "pass", None), (3, "undo", None), (4, "redo", None)]);
        assert_eq!(filtered_actions(&b), (vec![true, true, false, false], false));
        let c = acts(&[(1, "bid", None), (2, "pass", None), (3, "undo", None), (4, "pass", None), (5, "redo", None)]);
        assert_eq!(filtered_actions(&c), (vec![true, false, false, true, false], false));
        // undo to an action id: everything after it
        let d = acts(&[(1, "bid", None), (2, "pass", None), (3, "pass", None), (4, "undo", Some(1))]);
        assert_eq!(filtered_actions(&d), (vec![true, false, false, false], true));
        // undo to 0: everything
        let e = acts(&[(1, "bid", None), (2, "message", None), (3, "undo", Some(0))]);
        assert_eq!(filtered_actions(&e), (vec![false, true, false], true));
    }
}
