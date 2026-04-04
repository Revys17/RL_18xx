use std::collections::HashMap;
use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::actions::{Action, GameError};
use crate::core::{Phase, StockMarket};
use crate::entities::{Bank, Company, Corporation, Depot, EntityId, Player, Share, Token, Train};
use crate::graph::{City, Edge, Hex, Offboard, Tile, Town, Upgrade};
use crate::map::{GraphCache, NodeId, NodeType};
use crate::rounds::Round;
use crate::tiles::{self, TileColor, TileDef};
use crate::title::g1830::{self, HexType};

// ---------------------------------------------------------------------------
// RoundState
// ---------------------------------------------------------------------------

#[pyclass]
#[derive(Clone, Debug)]
pub struct RoundState {
    #[pyo3(get)]
    pub round_type: String,
    #[pyo3(get)]
    pub round_num: u8,
    pub active_entity_id: EntityId,
}

#[pymethods]
impl RoundState {
    #[new]
    pub fn new(round_type: String, round_num: u8) -> Self {
        RoundState {
            round_type,
            round_num,
            active_entity_id: EntityId::none(),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "RoundState(type='{}', num={})",
            self.round_type, self.round_num
        )
    }
}

// ---------------------------------------------------------------------------
// BaseGame
// ---------------------------------------------------------------------------

#[pyclass]
pub struct BaseGame {
    // Mutable game state (cloned per search)
    pub(crate) players: Vec<Player>,
    pub(crate) corporations: Vec<Corporation>,
    pub(crate) companies: Vec<Company>,
    pub(crate) bank: Bank,
    pub(crate) depot: Depot,
    pub(crate) phase: Phase,
    pub(crate) round_state: RoundState,
    pub(crate) round: Round,
    pub(crate) stock_market: StockMarket,
    /// Market cell occupants: (row, col) -> ordered list of corp syms.
    /// Corps are added when their token lands on a cell.
    pub(crate) market_cell_corps: HashMap<(u8, u8), Vec<String>>,
    pub(crate) hexes: Vec<Hex>,
    pub(crate) tile_counts_remaining: HashMap<String, u32>,

    // Immutable static data (shared via Arc for fast clone)
    pub(crate) hex_adjacency: Arc<HashMap<String, HashMap<u8, String>>>,
    pub(crate) tile_catalog: Arc<HashMap<String, TileDef>>,
    pub(crate) starting_cash: i32,
    pub(crate) cert_limit: u8,

    // Graph connectivity cache (cleared when tiles/tokens change)
    pub(crate) graph_cache: GraphCache,

    // Lookup caches
    pub(crate) corp_idx: HashMap<String, usize>,
    pub(crate) company_idx: HashMap<String, usize>,
    pub(crate) hex_idx: HashMap<String, usize>,

    // Metadata
    pub(crate) title: String,
    pub(crate) finished: bool,
    pub(crate) move_number: usize,
    /// Turn counter: incremented each time a new round starts (Auction=0, Stock1=1, OR=2, Stock2=3, ...).
    /// Used for SELL_AFTER="first" rule: selling is only allowed when turn > 1.
    pub(crate) turn: u32,

    // Game end tracking
    pub(crate) game_end_triggered: bool,
    /// The player order for the current game (ids, in seating order).
    pub(crate) player_order: Vec<u32>,
    /// Priority deal player id.
    pub(crate) priority_deal_player: u32,
}

impl BaseGame {
    /// Fast clone for MCTS search — shares immutable data via Arc.
    pub fn clone_for_search(&self) -> BaseGame {
        BaseGame {
            players: self.players.clone(),
            corporations: self.corporations.clone(),
            companies: self.companies.clone(),
            bank: self.bank.clone(),
            depot: self.depot.clone(),
            phase: self.phase.clone(),
            round_state: self.round_state.clone(),
            round: self.round.clone(),
            stock_market: self.stock_market.clone(),
            market_cell_corps: self.market_cell_corps.clone(),
            hexes: self.hexes.clone(),
            tile_counts_remaining: self.tile_counts_remaining.clone(),

            hex_adjacency: Arc::clone(&self.hex_adjacency),
            tile_catalog: Arc::clone(&self.tile_catalog),
            starting_cash: self.starting_cash,
            cert_limit: self.cert_limit,

            graph_cache: GraphCache::new(), // fresh cache for cloned state

            corp_idx: self.corp_idx.clone(),
            company_idx: self.company_idx.clone(),
            hex_idx: self.hex_idx.clone(),

            title: self.title.clone(),
            finished: self.finished,
            move_number: self.move_number,
            turn: self.turn,

            game_end_triggered: self.game_end_triggered,
            player_order: self.player_order.clone(),
            priority_deal_player: self.priority_deal_player,
        }
    }

    /// Internal action dispatch — routes to the correct round processor.
    fn process_action_internal(&mut self, action: &Action) -> Result<(), GameError> {
        if self.finished {
            return Err(GameError::new("Game is already finished"));
        }

        // Handle bankruptcy immediately
        if let Action::Bankrupt { entity_id } = action {
            // The bankrupt entity is the corporation. Its president's cash goes to the bank.
            let president_id = if let Some(&ci) = self.corp_idx.get(entity_id.as_str()) {
                self.corporations[ci].president_id()
            } else {
                entity_id.parse::<u32>().ok()
            };
            if let Some(pid) = president_id {
                if let Some(idx) = self.player_index(pid) {
                    self.bank.cash += self.players[idx].cash;
                    self.players[idx].cash = 0;
                }
            }
            self.end_game();
            return Ok(());
        }

        // Handle company exchange actions (e.g., MH exchange for NYC share).
        // These can happen in any round and bypass normal round dispatch.
        if self.try_process_company_exchange(action)? {
            self.move_number += 1;
            self.check_game_end();
            return Ok(());
        }

        // Handle company ability actions (CS lay_tile, DH lay_tile/place_token).
        // These happen during the OR but the entity is a company, not a corp.
        if self.try_process_company_ability(action)? {
            self.move_number += 1;
            self.check_game_end();

            // DH's lay_tile consumes the corp's tile lay AND token placement.
            // After DH lay_tile, the corp may optionally place the DH token
            // (DH place_token) or decline (DH pass). Either way, both Track
            // and PlaceToken steps are consumed.
            //
            // CS's lay_tile is a BONUS tile — doesn't consume anything.
            let entity_id = action.entity_id();
            let mut needs_skip_steps = false;
            if entity_id == "DH" {
                if let Round::Operating(ref mut s) = self.round {
                    match action {
                        Action::LayTile { .. } if s.step == crate::rounds::OperatingStep::LayTile => {
                            s.num_laid_track += 1;
                            s.num_placed_token += 1;
                            // Advance past Track and PlaceToken to RunRoutes.
                            // DH place_token (if it comes) is a no-op on step.
                            s.step = crate::rounds::OperatingStep::RunRoutes;
                        }
                        Action::Pass { .. } | Action::PlaceToken { .. } => {
                            // DH token placed or declined — skip_steps from
                            // current position to advance through RunRoutes etc.
                            needs_skip_steps = true;
                        }
                        _ => {}
                    }
                }
            }
            if needs_skip_steps {
                if let Round::Operating(_) = &self.round {
                    self.skip_steps();
                    let is_done = matches!(
                        &self.round,
                        Round::Operating(s) if !s.finished && s.step == crate::rounds::OperatingStep::Done
                    );
                    if is_done {
                        if let Round::Operating(ref mut s) = self.round {
                            s.advance_to_next_corp();
                        }
                        if !matches!(&self.round, Round::Operating(s) if s.finished) {
                            self.start_operating();
                        }
                    }
                }
            }

            return Ok(());
        }

        match &self.round {
            Round::Auction(_) => self.process_auction_action(action)?,
            Round::Stock(_) => self.process_stock_action(action)?,
            Round::Operating(_) => self.process_operating_action(action)?,
        }

        self.move_number += 1;

        // Check game end conditions after every action
        self.check_game_end();

        // Check for round transitions (may chain: e.g., Stock→OR1→OR2→Stock→OR1→OR2→Stock)
        // Need enough iterations for multiple empty Stock→OR→Stock cycles.
        for _ in 0..20 {
            let round_finished = match &self.round {
                Round::Auction(s) => s.finished,
                Round::Stock(s) => s.finished,
                Round::Operating(s) => s.finished,
            };
            if !round_finished {
                break;
            }
            if self.game_end_triggered && self.should_end_now() {
                self.end_game();
                return Ok(());
            }
            self.transition_to_next_round();
        }

        Ok(())
    }

    /// Check if the game should end now based on the end timing.
    /// 1830: bankruptcy = immediate (handled above), bank = full_or.
    fn should_end_now(&self) -> bool {
        // "full_or" timing: end after the last OR of the current set
        match &self.round {
            Round::Operating(s) => s.round_num >= s.total_ors,
            _ => false,
        }
    }

    /// Synchronize the `round_state` (PyO3-visible) from the internal `round` enum.
    pub(crate) fn update_round_state(&mut self) {
        self.round_state.round_type = self.round.round_type_str().to_string();
        self.round_state.round_num = self.round.round_num();

        match &self.round {
            Round::Auction(s) => {
                self.round_state.active_entity_id = EntityId::player(s.active_player_id());
            }
            Round::Stock(s) => {
                self.round_state.active_entity_id = EntityId::player(s.current_player_id());
            }
            Round::Operating(s) => {
                // If a corp must discard trains (crowded), it's the active entity.
                if let Some(crowded) = s.crowded_corps.first() {
                    self.round_state.active_entity_id = EntityId::corporation(crowded);
                } else if let Some(sym) = s.current_corp_sym() {
                    self.round_state.active_entity_id = EntityId::corporation(sym);
                }
            }
        }
    }

    /// Transition to the next round after the current round finishes.
    fn transition_to_next_round(&mut self) {
        match &self.round {
            Round::Auction(_) => {
                // Auction → Stock Round (still turn 1 — first stock round)
                let state =
                    crate::rounds::StockState::new(&self.player_order, self.priority_deal_player);
                self.round = Round::Stock(state);
                // Check if the first player can act; if not, auto-advance
                // (mirrors Python's Stock.setup() → skip_steps → next_entity)
                self.stock_start_entity();
            }
            Round::Stock(s) => {
                // Stock → Operating Round(s)
                self.priority_deal_player = s.priority_deal_player;
                let operating_order = self.compute_operating_order();
                let total_ors = self.phase.operating_rounds;
                self.round = Round::Operating(crate::rounds::OperatingState::new(
                    1,
                    total_ors,
                    operating_order,
                ));
                // OR setup: pay company revenues, start first corp's turn
                self.payout_companies();
                self.start_operating();
            }
            Round::Operating(s) => {
                if s.round_num < s.total_ors {
                    // More ORs in this set
                    let total_ors = s.total_ors;
                    let operating_order = self.compute_operating_order();
                    self.round = Round::Operating(crate::rounds::OperatingState::new(
                        s.round_num + 1,
                        total_ors,
                        operating_order,
                    ));
                    // OR setup: pay company revenues, start first corp's turn
                    self.payout_companies();
                    self.start_operating();
                } else {
                    // Back to Stock Round — increment turn (matches Python's turn counter)
                    self.turn += 1;
                    let state = crate::rounds::StockState::new(
                        &self.player_order,
                        self.priority_deal_player,
                    );
                    self.round = Round::Stock(state);
                    self.stock_start_entity();
                }
            }
        }
        self.update_round_state();
    }

    /// Check if buying a certain train triggers a phase change.
    pub(crate) fn check_phase_advance(&mut self, train_name: &str) {
        let phase_defs = g1830::phases();

        // Strip instance suffix (e.g., "3-0" → "3")
        let base_name = train_name.split('-').next().unwrap_or(train_name);

        // Find the phase that this train triggers
        let new_phase_name = match base_name {
            "3" => Some("3"),
            "4" => Some("4"),
            "5" => Some("5"),
            "6" => Some("6"),
            "D" => Some("D"),
            _ => None,
        };

        if let Some(name) = new_phase_name {
            // Only advance if we're in an earlier phase
            let current_phase_idx = phase_defs
                .iter()
                .position(|p| p.name == self.phase.name)
                .unwrap_or(0);
            let new_phase_idx = phase_defs.iter().position(|p| p.name == name).unwrap_or(0);

            if new_phase_idx <= current_phase_idx {
                return; // Already at or past this phase
            }

            let phase_def = &phase_defs[new_phase_idx];
            self.phase = Phase::new(
                phase_def.name.to_string(),
                phase_def.operating_rounds,
                phase_def.train_limit,
                phase_def.tiles.iter().map(|s| s.to_string()).collect(),
            );

            // Rust trains
            match name {
                "4" => self.rust_trains("2"),
                "6" => self.rust_trains("3"),
                "D" => self.rust_trains("4"),
                _ => {}
            }

            // Phase 5: close all private companies
            if name == "5" {
                self.close_all_companies();
            }

            // Check for corps over the new train limit (they must discard)
            let train_limit = phase_def.train_limit as usize;
            let crowded: Vec<String> = self
                .corporations
                .iter()
                .filter(|c| c.floated && c.trains.len() > train_limit)
                .map(|c| c.sym.clone())
                .collect();
            if !crowded.is_empty() {
                if let Round::Operating(ref mut s) = self.round {
                    s.crowded_corps = crowded;
                }
            }
        }
    }

    /// Remove all trains of a given name from corporations.
    fn rust_trains(&mut self, train_name: &str) {
        for corp in &mut self.corporations {
            corp.trains.retain(|t| t.name != train_name);
        }
        // Also remove from depot and discarded pile
        self.depot.trains.retain(|t| t.name != train_name);
        self.depot.discarded.retain(|t| t.name != train_name);
    }

    /// Close all private companies.
    fn close_all_companies(&mut self) {
        for company in &mut self.companies {
            company.closed = true;
        }
    }

    /// Handle company exchange abilities (e.g., MH → NYC share).
    /// Returns Ok(true) if the action was a company exchange and was processed,
    /// Ok(false) if it's not a company exchange (caller should dispatch normally).
    fn try_process_company_exchange(&mut self, action: &Action) -> Result<bool, GameError> {
        let (entity_id, corp_sym, percent) = match action {
            Action::BuyShares {
                entity_id,
                corporation_sym,
                percent,
                ..
            } => (entity_id.as_str(), corporation_sym.as_str(), *percent),
            _ => return Ok(false),
        };

        // Check if entity is a company (not a player or corp)
        let company_idx = match self.company_idx.get(entity_id) {
            Some(&idx) => idx,
            None => return Ok(false), // Not a company entity — normal action
        };

        // The company's owner (a player) receives the share
        let owner_player_id = self.companies[company_idx]
            .owner
            .player_id()
            .ok_or_else(|| GameError::new("Exchange company not owned by a player"))?;

        let corp_idx = *self
            .corp_idx
            .get(corp_sym)
            .ok_or_else(|| GameError::new(format!("Unknown corporation: {}", corp_sym)))?;

        let player_eid = EntityId::player(owner_player_id);

        // Find a share to transfer: prefer IPO, then market, then uninitialized
        let mut transferred = false;
        let ipo_eid = EntityId::ipo(corp_sym);
        let market_eid = EntityId::market();

        // Try IPO first
        for share in &mut self.corporations[corp_idx].shares {
            if !share.president && share.percent == percent && share.owner == ipo_eid {
                share.owner = player_eid.clone();
                transferred = true;
                break;
            }
        }
        // Then market
        if !transferred {
            for share in &mut self.corporations[corp_idx].shares {
                if !share.president && share.percent == percent && share.owner == market_eid {
                    share.owner = player_eid.clone();
                    transferred = true;
                    break;
                }
            }
        }
        // Then uninitialized (before corp is parred)
        if !transferred {
            for share in &mut self.corporations[corp_idx].shares {
                if !share.president && share.percent == percent && share.owner.is_none() {
                    share.owner = player_eid.clone();
                    transferred = true;
                    break;
                }
            }
        }

        if !transferred {
            return Err(GameError::new(format!(
                "No {}% share of {} available for exchange",
                percent, corp_sym
            )));
        }

        // Close the company
        self.companies[company_idx].closed = true;

        // Check float
        self.check_float(corp_idx);

        Ok(true)
    }

    /// Handle company ability actions (CS free tile lay, DH teleport tile/token).
    /// Returns Ok(true) if the action was a company ability and was processed.
    fn try_process_company_ability(&mut self, action: &Action) -> Result<bool, GameError> {
        let entity_id = action.entity_id();

        // Check if entity is a company
        let company_idx = match self.company_idx.get(entity_id) {
            Some(&idx) => idx,
            None => return Ok(false),
        };

        // Only handle if the company is still open
        if self.companies[company_idx].closed {
            return Ok(false);
        }

        match action {
            Action::LayTile {
                hex_id,
                tile_id,
                rotation,
                ..
            } => {
                // Company lay_tile ability (e.g., CS on B20, DH on F16)
                // The owning corporation pays terrain costs
                let base_tile_id = tile_id.split('-').next().unwrap_or(tile_id);

                // Find the owning corporation and charge terrain cost
                let corp_sym = self.companies[company_idx]
                    .owner
                    .corp_sym()
                    .map(|s| s.to_string());
                if let Some(ref sym) = corp_sym {
                    if let Some(&hex_idx) = self.hex_idx.get(hex_id.as_str()) {
                        let terrain_cost: i32 = self.hexes[hex_idx]
                            .tile
                            .upgrades
                            .iter()
                            .map(|u| u.cost)
                            .sum();
                        if terrain_cost > 0 {
                            let ci = self.corp_idx[sym.as_str()];
                            self.corporations[ci].cash -= terrain_cost;
                            self.bank.cash += terrain_cost;
                        }
                    }
                }

                if let Some(&hex_idx) = self.hex_idx.get(hex_id.as_str()) {
                    // Return old tile to supply
                    let old_tile_name = self.hexes[hex_idx].tile.name.clone();
                    let old_base = old_tile_name.split('-').next().unwrap_or(&old_tile_name);
                    if !old_base.starts_with("preprinted") {
                        *self
                            .tile_counts_remaining
                            .entry(old_base.to_string())
                            .or_insert(0) += 1;
                    }

                    let old_cities = self.hexes[hex_idx].tile.cities.clone();

                    // Build new tile from catalog (with paths) or fallback
                    let mut new_tile = if let Some(tile_def) = self.tile_catalog.get(base_tile_id) {
                        let mut t = BaseGame::tile_from_def(tile_def, *rotation);
                        t.id = tile_id.to_string();
                        t.name = tile_id.to_string();
                        t
                    } else {
                        let mut t =
                            crate::graph::Tile::new(tile_id.to_string(), tile_id.to_string());
                        t.rotation = *rotation;
                        t
                    };

                    // Transfer tokens from old cities
                    if !old_cities.is_empty() {
                        if new_tile.cities.is_empty() {
                            if let Some(city_slots) = crate::title::g1830::tile_cities(base_tile_id)
                            {
                                for (i, &slots) in city_slots.iter().enumerate() {
                                    if i < old_cities.len() {
                                        let mut city = old_cities[i].clone();
                                        while city.tokens.len() < slots as usize {
                                            city.tokens.push(None);
                                        }
                                        city.slots = slots;
                                        new_tile.cities.push(city);
                                    } else {
                                        new_tile.cities.push(City::new(0, slots));
                                    }
                                }
                            } else {
                                new_tile.cities = old_cities;
                            }
                        } else {
                            for (i, new_city) in new_tile.cities.iter_mut().enumerate() {
                                if i < old_cities.len() {
                                    for (j, old_tok) in old_cities[i].tokens.iter().enumerate() {
                                        if let Some(tok) = old_tok {
                                            if j < new_city.tokens.len() {
                                                new_city.tokens[j] = Some(tok.clone());
                                            } else {
                                                new_city.tokens.push(Some(tok.clone()));
                                                new_city.slots += 1;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    self.hexes[hex_idx].tile = new_tile;
                    self.clear_graph_cache();

                    // Decrement tile count
                    let base_id = base_tile_id.to_string();
                    if let Some(count) = self.tile_counts_remaining.get_mut(&base_id) {
                        *count -= 1;
                        if *count == 0 {
                            self.tile_counts_remaining.remove(&base_id);
                        }
                    }
                }

                Ok(true)
            }
            Action::PlaceToken {
                hex_id, city_index, ..
            } => {
                // Company place_token ability (e.g., DH teleport)
                // Find the corporation that owns this company
                let corp_eid = &self.companies[company_idx].owner;
                let corp_sym = corp_eid
                    .corp_sym()
                    .ok_or_else(|| GameError::new("Company not owned by a corporation"))?
                    .to_string();
                let corp_idx = self.corp_idx[&corp_sym];

                // Resolve hex
                let resolved_hex_id = if let Some(tile_instance) = hex_id.strip_prefix("__tile:") {
                    let base_name = tile_instance.split('-').next().unwrap_or(tile_instance);
                    self.hexes
                        .iter()
                        .find(|h| {
                            h.tile.name == tile_instance
                                || h.tile.id == tile_instance
                                || h.tile.name == base_name
                                || h.id == base_name
                        })
                        .map(|h| h.id.clone())
                        .ok_or_else(|| {
                            GameError::new(format!("No hex with tile {}", tile_instance))
                        })?
                } else {
                    hex_id.clone()
                };

                let hex_idx = *self
                    .hex_idx
                    .get(resolved_hex_id.as_str())
                    .ok_or_else(|| GameError::new(format!("Unknown hex: {}", resolved_hex_id)))?;

                // Place token (free — no cost for ability tokens)
                let token_idx = self.corporations[corp_idx]
                    .next_token_index()
                    .ok_or_else(|| GameError::new("No tokens available"))?;

                let city = self.hexes[hex_idx]
                    .tile
                    .cities
                    .get_mut(*city_index as usize)
                    .ok_or_else(|| GameError::new("Invalid city index"))?;

                let slot_idx = city
                    .tokens
                    .iter()
                    .position(|t| t.is_none())
                    .ok_or_else(|| GameError::new("No empty token slots"))?;

                let mut token = self.corporations[corp_idx].tokens[token_idx].clone();
                token.used = true;
                token.city_hex_id = resolved_hex_id.clone();
                city.tokens[slot_idx] = Some(token);

                self.corporations[corp_idx].tokens[token_idx].used = true;
                self.corporations[corp_idx].tokens[token_idx].city_hex_id = resolved_hex_id;

                self.clear_graph_cache();

                Ok(true)
            }
            Action::Pass { .. } => {
                // Pass from a company entity (e.g., declining DH token placement).
                // This is a no-op — the company ability was offered but declined.
                // Handled by the caller to trigger skip_steps if needed.
                Ok(true)
            }
            _ => Ok(false),
        }
    }

    /// Check if the game should end.
    fn check_game_end(&mut self) {
        // Bank breaks → end after current set of ORs
        if self.bank.cash <= 0 && !self.game_end_triggered {
            self.game_end_triggered = true;
        }
    }

    /// End the game and calculate final scores.
    fn end_game(&mut self) {
        self.finished = true;
    }

    /// Calculate final results: player_id -> total value.
    /// Value = cash + share values at current market price + face value of owned companies.
    pub fn calculate_results(&self) -> HashMap<u32, i32> {
        let mut results = HashMap::new();
        for player in &self.players {
            let value = player.value(&self.corporations) + player.company_value(&self.companies);
            results.insert(player.id, value);
        }
        results
    }

    fn force_or_completion(&mut self) {
        // Complete the current OR and advance through remaining ORs to reach Stock.
        // Each new OR gets company payouts and state machine advancement.
        for _ in 0..5 {
            // Complete current OR
            if let Round::Operating(ref mut s) = self.round {
                s.finished = true;
            }
            self.transition_to_next_round();
            if !matches!(&self.round, Round::Operating(_)) {
                break;
            }
        }
        self.update_round_state();
    }

    /// Update market cell occupancy when a corp's share price changes.
    /// Removes from old cell, adds to new cell.
    pub(crate) fn update_market_cell(
        &mut self,
        corp_sym: &str,
        old_row: u8,
        old_col: u8,
        new_row: u8,
        new_col: u8,
    ) {
        if (old_row, old_col) != (new_row, new_col) || old_row == 0 && old_col == 0 {
            // Remove from old cell
            if let Some(corps) = self.market_cell_corps.get_mut(&(old_row, old_col)) {
                corps.retain(|c| c != corp_sym);
            }
        }
        // Add to new cell (if not already there)
        let cell = self
            .market_cell_corps
            .entry((new_row, new_col))
            .or_default();
        if !cell.contains(&corp_sym.to_string()) {
            cell.push(corp_sym.to_string());
        }
    }

    /// Get a corp's position within its market cell (for operating order tiebreak).
    pub(crate) fn market_cell_position(&self, corp_sym: &str, row: u8, col: u8) -> usize {
        self.market_cell_corps
            .get(&(row, col))
            .and_then(|corps| corps.iter().position(|c| c == corp_sym))
            .unwrap_or(0)
    }

    /// Clear the graph cache (call after tile/token changes).
    pub(crate) fn clear_graph_cache(&mut self) {
        self.graph_cache.clear();
    }

    /// Get the token placements for a corporation: (hex_id, city_index) pairs.
    pub(crate) fn corp_token_positions(&self, corp_sym: &str) -> Vec<(String, usize)> {
        let ci = match self.corp_idx.get(corp_sym) {
            Some(&i) => i,
            None => return Vec::new(),
        };
        let mut positions = Vec::new();
        for token in &self.corporations[ci].tokens {
            if token.used && !token.city_hex_id.is_empty() {
                // Find which city index this token is in
                if let Some(&hi) = self.hex_idx.get(&token.city_hex_id) {
                    for (ci_idx, city) in self.hexes[hi].tile.cities.iter().enumerate() {
                        if city
                            .tokens
                            .iter()
                            .any(|t| t.as_ref().is_some_and(|tok| tok.corporation_id == corp_sym))
                        {
                            positions.push((token.city_hex_id.clone(), ci_idx));
                            break;
                        }
                    }
                }
            }
        }
        positions
    }

    /// Build the home city reservations list: (hex_id, city_index, corp_sym).
    /// In 1830 each corp's home city is reserved for that corp until they place
    /// their home token. After the token is placed, the reservation is consumed.
    pub(crate) fn home_reservations(&self) -> Vec<(String, usize, String)> {
        let corp_defs = crate::title::g1830::corporations();
        let mut reservations = Vec::new();
        for cd in &corp_defs {
            let sym = cd.sym.to_string();
            // Only reserve if the corp hasn't placed its home token yet
            if let Some(&ci) = self.corp_idx.get(sym.as_str()) {
                let home_token_placed =
                    self.corporations[ci].tokens.first().is_some_and(|t| t.used);
                if !home_token_placed {
                    reservations.push((cd.home_hex.to_string(), cd.home_city_index as usize, sym));
                }
            }
        }
        reservations
    }

    /// Build a tile from a TileDef, creating the graph.rs Tile with paths and cities.
    pub(crate) fn tile_from_def(tile_def: &TileDef, rotation: u8) -> Tile {
        let rotated = tile_def.rotated(rotation);
        let mut tile = Tile::new(tile_def.name.clone(), tile_def.name.clone());
        tile.rotation = rotation;
        tile.color = rotated.color;
        tile.label = rotated.label.clone();
        tile.paths = rotated.paths.clone();
        tile.edges = rotated.edges.iter().map(|&e| Edge::new(e)).collect();
        for cd in &rotated.cities {
            tile.cities.push(City::new(cd.revenue, cd.slots));
        }
        for td in &rotated.towns {
            tile.towns.push(Town::new(td.revenue));
        }
        for od in &rotated.offboards {
            tile.offboards.push(Offboard::new(od.yellow_revenue));
        }
        for ud in &rotated.upgrades {
            tile.upgrades
                .push(Upgrade::new(ud.cost, ud.terrain.clone()));
        }
        tile
    }

    /// Compute the operating order.
    /// Sort key matches Python: [-price, -column, row, cell_position, name]
    pub fn compute_operating_order(&self) -> Vec<String> {
        let mut corps: Vec<(String, i32, u8, u8, usize, String)> = self
            .corporations
            .iter()
            .filter(|c| c.floated)
            .map(|c| {
                let (price, col, row) = c
                    .share_price
                    .as_ref()
                    .map_or((0, 0, 0), |sp| (sp.price, sp.column, sp.row));
                let cell_pos = self.market_cell_position(&c.sym, row, col);
                (c.sym.clone(), price, col, row, cell_pos, c.name.clone())
            })
            .collect();
        corps.sort_by(|a, b| {
            b.1.cmp(&a.1) // highest price first
                .then(b.2.cmp(&a.2)) // rightmost column first
                .then(a.3.cmp(&b.3)) // lowest row first
                .then(a.4.cmp(&b.4)) // earlier cell arrival first
                .then(a.5.cmp(&b.5)) // alphabetical name
        });
        corps.into_iter().map(|(sym, ..)| sym).collect()
    }
}

// ---------------------------------------------------------------------------
// Hex construction helpers
// ---------------------------------------------------------------------------

fn build_hex_from_def(def: &g1830::HexDef) -> Hex {
    let coord = def.coord.to_string();

    // Check if this hex has a preprinted DSL definition with path data
    if let Some((dsl, color_str)) = g1830::preprinted_hex_dsl(def.coord) {
        let color = match color_str {
            "red" => TileColor::Red,
            "gray" => TileColor::Gray,
            "yellow" => TileColor::Yellow,
            _ => TileColor::White,
        };
        let tile_def = tiles::parse_preprinted_tile(def.coord, dsl, color);
        let mut tile = BaseGame::tile_from_def(&tile_def, 0);
        tile.id = format!("preprinted_{}", coord);
        tile.name = coord.clone();

        // Add terrain upgrade cost if applicable
        if def.terrain_cost > 0 {
            let terrain = if def.terrain_cost >= 120 {
                "mountain".to_string()
            } else {
                "water".to_string()
            };
            tile.upgrades.push(Upgrade::new(def.terrain_cost, terrain));
        }

        return Hex::new(coord, tile);
    }

    // Non-preprinted hexes (white blanks, white cities/towns without paths)
    let tile = match &def.hex_type {
        HexType::Blank => {
            let mut t = Tile::new(String::new(), String::new());
            if def.terrain_cost > 0 {
                let terrain = if def.terrain_cost >= 120 {
                    "mountain".to_string()
                } else {
                    "water".to_string()
                };
                t.upgrades.push(Upgrade::new(def.terrain_cost, terrain));
            }
            t
        }
        HexType::City { revenue, slots } => {
            let mut t = Tile::new(format!("preprinted_{}", coord), coord.clone());
            t.cities.push(City::new(*revenue, *slots));
            if def.terrain_cost > 0 {
                let terrain = if def.terrain_cost >= 120 {
                    "mountain".to_string()
                } else {
                    "water".to_string()
                };
                t.upgrades.push(Upgrade::new(def.terrain_cost, terrain));
            }
            t
        }
        HexType::Town { revenue } => {
            let mut t = Tile::new(format!("preprinted_{}", coord), coord.clone());
            t.towns.push(Town::new(*revenue));
            t
        }
        HexType::DoubleCity { revenue } => {
            let mut t = Tile::new(format!("preprinted_{}", coord), coord.clone());
            t.cities.push(City::new(*revenue, 1));
            t.cities.push(City::new(*revenue, 1));
            if def.terrain_cost > 0 {
                t.upgrades
                    .push(Upgrade::new(def.terrain_cost, "water".to_string()));
            }
            t
        }
        HexType::DoubleTown => {
            let mut t = Tile::new(format!("preprinted_{}", coord), coord.clone());
            t.towns.push(Town::new(0));
            t.towns.push(Town::new(0));
            t
        }
        HexType::Offboard {
            yellow_revenue,
            brown_revenue: _,
        } => {
            let mut t = Tile::new(format!("offboard_{}", coord), coord.clone());
            t.offboards.push(Offboard::new(*yellow_revenue));
            t
        }
        HexType::Path => Tile::new(format!("path_{}", coord), coord.clone()),
    };

    Hex::new(coord, tile)
}

// ---------------------------------------------------------------------------
// PyO3 methods
// ---------------------------------------------------------------------------

#[pymethods]
impl BaseGame {
    #[new]
    fn new(player_names: HashMap<u32, String>) -> Self {
        let num_players = player_names.len() as u8;
        let cash = g1830::starting_cash(num_players);
        let cert_lim = g1830::cert_limit(num_players);

        // 1. Players (sorted by ID for deterministic order)
        let mut player_ids: Vec<u32> = player_names.keys().copied().collect();
        player_ids.sort();
        let players: Vec<Player> = player_ids
            .iter()
            .map(|&id| Player::new(id, player_names[&id].clone(), cash))
            .collect();

        // 2. Bank (starting cash is deducted for players)
        let total_player_cash = cash * num_players as i32;
        let bank = Bank::new(g1830::BANK_CASH - total_player_cash);

        // 3. Companies
        let company_defs = g1830::companies();
        let companies: Vec<Company> = company_defs
            .iter()
            .map(|cd| {
                let mut company = Company::new(
                    cd.sym.to_string(),
                    cd.name.to_string(),
                    cd.value,
                    cd.revenue,
                );
                // BO (Baltimore & Ohio private) has a no_buy ability —
                // it cannot be purchased by corporations during the OR.
                if cd.sym == "BO" {
                    company.no_buy = true;
                }
                company
            })
            .collect();

        // 4. Corporations (with tokens and shares)
        let corp_defs = g1830::corporations();
        let corporations: Vec<Corporation> = corp_defs
            .iter()
            .map(|cd| {
                let tokens: Vec<Token> = cd
                    .token_prices
                    .iter()
                    .map(|&price| Token::new(cd.sym.to_string(), price))
                    .collect();

                let mut shares = Vec::with_capacity(9);
                // President share (20%)
                shares.push(Share::new(cd.sym.to_string(), 20, true));
                // 8 normal shares (10%)
                for _ in 0..8 {
                    shares.push(Share::new(cd.sym.to_string(), 10, false));
                }

                Corporation::new(cd.sym.to_string(), cd.name.to_string(), tokens, shares)
            })
            .collect();

        // 5. Depot (trains)
        let train_defs = g1830::trains();
        let mut depot = Depot::new();
        let mut train_instance_counters: std::collections::HashMap<String, u32> =
            std::collections::HashMap::new();
        for td in &train_defs {
            for _ in 0..td.count {
                let instance = train_instance_counters
                    .entry(td.name.to_string())
                    .or_insert(0);
                let mut train = Train::new(td.name.to_string(), td.distance, td.price);
                train.id = format!("{}-{}", td.name, instance);
                *instance += 1;
                depot.trains.push(train);
            }
        }

        // 6. Hexes
        let hex_defs = g1830::hex_definitions();
        let hexes: Vec<Hex> = hex_defs.iter().map(build_hex_from_def).collect();

        // 7. Hex adjacency
        let coords: Vec<&str> = hex_defs.iter().map(|h| h.coord).collect();
        let adjacency = g1830::compute_adjacency(&coords);

        // 8. Phase
        let phase_defs = g1830::phases();
        let first_phase = &phase_defs[0];
        let phase = Phase::new(
            first_phase.name.to_string(),
            first_phase.operating_rounds,
            first_phase.train_limit,
            first_phase.tiles.iter().map(|s| s.to_string()).collect(),
        );

        // 9. Round state + Round enum
        let first_player_id = player_ids[0];
        let round_state = RoundState {
            round_type: "Auction".to_string(),
            round_num: 0,
            active_entity_id: EntityId::player(first_player_id),
        };
        let auction_state = crate::rounds::AuctionState::new(&player_ids, companies.len());
        let round = Round::Auction(auction_state);

        // 9b. Stock market
        let stock_market = StockMarket::new_1830();

        // 10. Tile counts
        let tile_counts_remaining: HashMap<String, u32> = g1830::tile_counts()
            .into_iter()
            .map(|(id, count)| (id.to_string(), count))
            .collect();

        // 10b. Tile catalog (shared immutable data)
        let tile_catalog = tiles::tile_catalog_1830();

        // 11. Lookup caches
        let corp_idx: HashMap<String, usize> = corporations
            .iter()
            .enumerate()
            .map(|(i, c)| (c.sym.clone(), i))
            .collect();

        let company_idx: HashMap<String, usize> = companies
            .iter()
            .enumerate()
            .map(|(i, c)| (c.sym.clone(), i))
            .collect();

        let hex_idx: HashMap<String, usize> = hexes
            .iter()
            .enumerate()
            .map(|(i, h)| (h.id.clone(), i))
            .collect();

        BaseGame {
            players,
            corporations,
            companies,
            bank,
            depot,
            phase,
            round_state,
            round,
            stock_market,
            market_cell_corps: HashMap::new(),
            hexes,
            tile_counts_remaining,
            hex_adjacency: Arc::new(adjacency),
            tile_catalog,
            starting_cash: cash,
            cert_limit: cert_lim,
            graph_cache: GraphCache::new(),
            corp_idx,
            company_idx,
            hex_idx,
            title: "1830".to_string(),
            finished: false,
            move_number: 0,
            turn: 1, // Start at 1 (Auction round is turn 1, first Stock round is still turn 1)
            game_end_triggered: false,
            player_order: player_ids.clone(),
            priority_deal_player: first_player_id,
        }
    }

    // -- Getters --

    #[getter]
    fn players(&self) -> Vec<Player> {
        self.players.clone()
    }

    #[getter]
    fn corporations(&self) -> Vec<Corporation> {
        self.corporations.clone()
    }

    #[getter]
    fn companies(&self) -> Vec<Company> {
        self.companies.clone()
    }

    #[getter]
    fn bank(&self) -> Bank {
        self.bank.clone()
    }

    #[getter]
    fn depot(&self) -> Depot {
        self.depot.clone()
    }

    #[getter]
    fn hexes(&self) -> Vec<Hex> {
        self.hexes.clone()
    }

    #[getter]
    fn phase(&self) -> Phase {
        self.phase.clone()
    }

    #[getter]
    fn finished(&self) -> bool {
        self.finished
    }

    #[getter]
    fn title(&self) -> String {
        self.title.clone()
    }

    #[getter]
    fn move_number(&self) -> usize {
        self.move_number
    }

    #[getter]
    fn round(&self) -> RoundState {
        self.round_state.clone()
    }

    /// Debug: return (round_type, round_num, step_name, active_corp)
    fn debug_state(&self) -> (String, u8, String, String) {
        match &self.round {
            crate::rounds::Round::Auction(_) => ("Auction".into(), 0, "".into(), "".into()),
            crate::rounds::Round::Stock(_) => ("Stock".into(), 0, "".into(), "".into()),
            crate::rounds::Round::Operating(s) => {
                let step = format!("{:?}", s.step);
                let corp = s.current_corp_sym().unwrap_or("none").to_string();
                ("Operating".into(), s.round_num, step, corp)
            }
        }
    }

    /// Unplaced tiles as a list (for encoder's depot_tiles feature).
    /// The encoder iterates game.tiles and counts by tile name/id.
    #[getter]
    fn tiles(&self) -> Vec<Tile> {
        let mut result = Vec::new();
        for (id, &count) in &self.tile_counts_remaining {
            for _ in 0..count {
                result.push(Tile::new(id.clone(), id.clone()));
            }
        }
        result
    }

    // -- Lookup methods --

    fn corporation_by_id(&self, sym: &str) -> Option<Corporation> {
        self.corp_idx
            .get(sym)
            .map(|&i| self.corporations[i].clone())
    }

    fn company_by_id(&self, sym: &str) -> Option<Company> {
        self.company_idx
            .get(sym)
            .map(|&i| self.companies[i].clone())
    }

    fn hex_by_id(&self, coord: &str) -> Option<Hex> {
        self.hex_idx.get(coord).map(|&i| self.hexes[i].clone())
    }

    /// Returns the currently active player(s).
    fn active_players(&self) -> Vec<Player> {
        let active_id = &self.round_state.active_entity_id.0;
        self.players
            .iter()
            .filter(|p| &EntityId::player(p.id).0 == active_id)
            .cloned()
            .collect()
    }

    /// Returns the priority deal player.
    fn priority_deal_player_py(&self) -> Player {
        let idx = self
            .players
            .iter()
            .position(|p| p.id == self.priority_deal_player)
            .unwrap_or(0);
        self.players[idx].clone()
    }

    /// Count certificates owned by a player (shares + companies).
    fn num_certs(&self, player_id: u32) -> u32 {
        self.num_certs_internal(player_id)
    }

    /// Get the certificate limit for this game.
    #[getter]
    fn cert_limit_py(&self) -> u8 {
        self.cert_limit
    }

    /// Process a game action from a Python dict.
    ///
    /// This is the main entry point for advancing the game state.
    /// The dict should have at minimum a "type" and "entity" key.
    fn process_action(&mut self, action_dict: &Bound<'_, PyDict>) -> PyResult<()> {
        let action = Action::from_py_dict(action_dict)?;
        self.process_action_internal(&action)?;
        Ok(())
    }

    /// Get the connected hexes for a corporation (from graph cache).
    fn get_connected_hexes(&mut self, corp_sym: String) -> Vec<String> {
        let token_positions = self.corp_token_positions(&corp_sym);
        let reservations = self.home_reservations();
        let graph = self.graph_cache.get_or_compute(
            &corp_sym,
            &self.hexes,
            &self.hex_idx,
            &self.hex_adjacency,
            &token_positions,
            &reservations,
        );
        let mut hexes: Vec<String> = graph.connected_hexes.keys().cloned().collect();
        hexes.sort();
        hexes
    }

    /// Get the current OR step as a string (for debugging).
    fn get_or_step(&self) -> String {
        match &self.round {
            crate::rounds::Round::Operating(s) => {
                let active = if let Some(crowded) = s.crowded_corps.first() {
                    crowded.as_str()
                } else {
                    s.current_corp_sym().unwrap_or("?")
                };
                format!("{:?} entity_idx={} corp={}", s.step, s.entity_index, active)
            }
            _ => format!("not in OR (round={})", self.round.round_type_str()),
        }
    }

    /// Get tokenable cities for a corporation: list of (hex_id, city_index).
    fn get_tokenable_cities(&mut self, corp_sym: String) -> Vec<(String, usize)> {
        self.tokenable_cities_for(&corp_sym)
    }

    /// Returns the set of valid action type strings at the current game state.
    fn legal_action_types(&self) -> Vec<String> {
        if self.finished {
            return Vec::new();
        }

        match &self.round {
            Round::Auction(s) => {
                if s.remaining_companies.is_empty() {
                    return Vec::new();
                }
                vec!["bid".to_string(), "pass".to_string()]
            }
            Round::Stock(s) => {
                let player_id = s.current_player_id();
                let mut types = Vec::new();

                // Check must_sell (over cert limit)
                let certs = self.num_certs_internal(player_id);
                if certs > self.cert_limit as u32 {
                    // Can only sell if must_sell
                    types.push("sell_shares".to_string());
                    return types;
                }

                // Can sell (1830: sell_buy_sell = no ordering restriction)
                let player_eid = EntityId::player(player_id);
                let has_sellable = self
                    .corporations
                    .iter()
                    .any(|c| c.floated && c.percent_owned_by(&player_eid) > 0);
                if has_sellable {
                    types.push("sell_shares".to_string());
                }

                if !s.bought_this_turn {
                    // Can buy shares
                    let can_buy = certs < self.cert_limit as u32
                        && self.corporations.iter().any(|c| {
                            if !c.floated {
                                return false;
                            }
                            if s.sold_corp_this_round(player_id, &c.sym) {
                                return false;
                            }
                            let ipo_eid = EntityId::ipo(&c.sym);
                            let market_eid = EntityId::market();
                            c.shares.iter().any(|sh| {
                                !sh.president && (sh.owner == ipo_eid || sh.owner == market_eid)
                            })
                        });
                    if can_buy {
                        types.push("buy_shares".to_string());
                    }

                    // Can par
                    let can_par = certs < self.cert_limit as u32
                        && self.corporations.iter().any(|c| c.ipo_price.is_none());
                    if can_par {
                        types.push("par".to_string());
                    }
                }

                types.push("pass".to_string());
                types
            }
            Round::Operating(s) => {
                let mut types = Vec::new();
                match s.step {
                    crate::rounds::OperatingStep::DiscardTrain => {
                        types.push("discard_train".to_string());
                    }
                    crate::rounds::OperatingStep::LayTile => {
                        types.push("lay_tile".to_string());
                        types.push("pass".to_string());
                    }
                    crate::rounds::OperatingStep::PlaceToken => {
                        types.push("place_token".to_string());
                        types.push("pass".to_string());
                    }
                    crate::rounds::OperatingStep::RunRoutes => {
                        types.push("run_routes".to_string());
                        types.push("pass".to_string());
                    }
                    crate::rounds::OperatingStep::Dividend => {
                        types.push("dividend".to_string());
                    }
                    crate::rounds::OperatingStep::BuyTrain => {
                        types.push("buy_train".to_string());
                        types.push("pass".to_string());
                    }
                    crate::rounds::OperatingStep::BuyCompany => {
                        types.push("buy_company".to_string());
                        types.push("pass".to_string());
                    }
                    crate::rounds::OperatingStep::Done => {}
                }
                types
            }
        }
    }

    /// Get the game result: player_id -> total value (cash + share values).
    fn result(&self) -> HashMap<u32, i32> {
        self.calculate_results()
    }

    /// Get all valid par prices for the stock market.
    fn par_prices(&self) -> Vec<i32> {
        self.stock_market.par_prices()
    }

    /// Fast clone for MCTS (exposed to Python as pickle_clone for compat).
    fn pickle_clone(&self) -> BaseGame {
        self.clone_for_search()
    }

    // -- Phase 4: Connectivity, Tile, Token, Routing --

    /// Get reachable hexes for a corporation: hex_id -> list of connected edges.
    fn connected_hexes(&mut self, corp_sym: &str) -> HashMap<String, Vec<u8>> {
        let token_positions = self.corp_token_positions(corp_sym);
        let reservations = self.home_reservations();
        let graph = self.graph_cache.get_or_compute(
            corp_sym,
            &self.hexes,
            &self.hex_idx,
            &self.hex_adjacency,
            &token_positions,
            &reservations,
        );
        graph
            .connected_hexes
            .iter()
            .map(|(k, v)| {
                let mut edges: Vec<u8> = v.iter().copied().collect();
                edges.sort();
                (k.clone(), edges)
            })
            .collect()
    }

    /// Get cities where a corporation can place tokens: list of (hex_id, city_index).
    fn tokenable_cities_for(&mut self, corp_sym: &str) -> Vec<(String, usize)> {
        let token_positions = self.corp_token_positions(corp_sym);
        let reservations = self.home_reservations();
        let graph = self.graph_cache.get_or_compute(
            corp_sym,
            &self.hexes,
            &self.hex_idx,
            &self.hex_adjacency,
            &token_positions,
            &reservations,
        );
        graph
            .tokenable_cities
            .iter()
            .map(|tc| (tc.hex_id.clone(), tc.city_index))
            .collect()
    }

    /// Calculate optimal routes and revenue for a corporation.
    /// Returns (routes_as_dicts, total_revenue).
    fn calculate_routes(&self, corp_sym: &str) -> (Vec<HashMap<String, String>>, i32) {
        let ci = match self.corp_idx.get(corp_sym) {
            Some(&i) => i,
            None => return (Vec::new(), 0),
        };
        let corp = &self.corporations[ci];

        // Get token nodes
        let token_positions = self.corp_token_positions(corp_sym);
        let token_nodes: Vec<NodeId> = token_positions
            .iter()
            .map(|(hex_id, city_idx)| NodeId {
                hex_id: hex_id.clone(),
                node_type: NodeType::City,
                index: *city_idx,
            })
            .collect();

        // Get trains
        let trains: Vec<(u32, bool)> = corp
            .trains
            .iter()
            .map(|t| (t.distance, t.name == "D"))
            .collect();

        let (routes, revenue) = crate::router::calculate_corp_routes(
            &self.hexes,
            &self.hex_idx,
            &self.hex_adjacency,
            &token_nodes,
            &trains,
        );

        // Convert to Python-friendly format
        let route_dicts: Vec<HashMap<String, String>> = routes
            .iter()
            .map(|r| {
                let mut d = HashMap::new();
                d.insert("revenue".to_string(), r.revenue.to_string());
                let nodes_str: Vec<String> = r
                    .nodes
                    .iter()
                    .map(|n| format!("{}:{:?}:{}", n.hex_id, n.node_type, n.index))
                    .collect();
                d.insert("nodes".to_string(), nodes_str.join(","));
                d
            })
            .collect();

        (route_dicts, revenue)
    }

    /// Check if a corporation can run any route.
    /// Requires at least 2 connected revenue nodes (one must be a corp token city).
    pub(crate) fn can_run_route(&mut self, corp_sym: &str) -> bool {
        let token_positions = self.corp_token_positions(corp_sym);
        let reservations = self.home_reservations();
        let graph = self.graph_cache.get_or_compute(
            corp_sym,
            &self.hexes,
            &self.hex_idx,
            &self.hex_adjacency,
            &token_positions,
            &reservations,
        );
        // Need at least 2 connected nodes to form a route
        graph.connected_nodes.len() >= 2
    }

    pub(crate) fn can_place_token(&mut self, corp_sym: &str) -> bool {
        let ci = match self.corp_idx.get(corp_sym) {
            Some(&i) => i,
            None => return false,
        };
        let corp = &self.corporations[ci];
        // Must have unplaced tokens
        let token_idx = match corp.next_token_index() {
            Some(idx) => idx,
            None => return false,
        };
        // Must afford cheapest token
        let price = corp.tokens[token_idx].price;
        if corp.cash < price {
            return false;
        }
        // If the corp has no tokens on the board (home token not yet placed),
        // it can always place on its home hex — no reachability needed.
        let has_placed_token = corp.tokens.iter().any(|t| t.used);
        if !has_placed_token {
            return true;
        }
        // Must have at least one reachable tokenable city
        !self.tokenable_cities_for(corp_sym).is_empty()
    }

    /// Compute the set of edge exits for each city index in a tile, using the
    /// tile catalog's path definitions. Returns a Vec where index i contains the
    /// set of edge numbers connected to city i.
    pub(crate) fn city_exits_from_catalog(&self, tile_name: &str, rotation: u8) -> Vec<Vec<u8>> {
        let tile_def = match self.tile_catalog.get(tile_name) {
            Some(td) => td.rotated(rotation),
            None => return Vec::new(),
        };
        let num_cities = tile_def.cities.len();
        let mut exits: Vec<Vec<u8>> = vec![Vec::new(); num_cities];
        for path in &tile_def.paths {
            let (city_idx, edge) = match (&path.a, &path.b) {
                (crate::tiles::PathEndpoint::City(ci), crate::tiles::PathEndpoint::Edge(e)) => {
                    (Some(*ci), Some(*e))
                }
                (crate::tiles::PathEndpoint::Edge(e), crate::tiles::PathEndpoint::City(ci)) => {
                    (Some(*ci), Some(*e))
                }
                _ => (None, None),
            };
            if let (Some(ci), Some(e)) = (city_idx, edge) {
                if ci < num_cities {
                    exits[ci].push(e);
                }
            }
        }
        exits
    }

    fn __repr__(&self) -> String {
        format!(
            "BaseGame(title='{}', players={}, corps={}, hexes={}, finished={})",
            self.title,
            self.players.len(),
            self.corporations.len(),
            self.hexes.len(),
            self.finished
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::actions::Action;
    use crate::rounds::Round;

    fn new_4p_game() -> BaseGame {
        let mut players = HashMap::new();
        players.insert(1, "Alice".to_string());
        players.insert(2, "Bob".to_string());
        players.insert(3, "Carol".to_string());
        players.insert(4, "Dave".to_string());
        BaseGame::new(players)
    }

    #[test]
    fn construction_produces_valid_state() {
        let game = new_4p_game();
        assert_eq!(game.players.len(), 4);
        assert_eq!(game.corporations.len(), 8);
        assert_eq!(game.companies.len(), 6);
        assert_eq!(game.bank.cash, 12000 - 600 * 4); // Bank minus starting cash
        assert!(!game.finished);
        assert_eq!(game.move_number, 0);
        for player in &game.players {
            assert_eq!(player.cash, 600);
        }
    }

    #[test]
    fn initial_round_is_auction() {
        let game = new_4p_game();
        assert!(matches!(game.round, Round::Auction(_)));
        assert_eq!(game.round_state.round_type, "Auction");
    }

    #[test]
    fn auction_state_has_6_companies() {
        let game = new_4p_game();
        if let Round::Auction(ref state) = game.round {
            assert_eq!(state.remaining_companies.len(), 6);
            assert_eq!(state.player_order.len(), 4);
            assert!(state.auctioning.is_none());
            assert!(!state.finished);
            assert_eq!(state.discount, 0);
        } else {
            panic!("Expected Auction round");
        }
    }

    #[test]
    fn auction_buy_cheapest_at_face_value() {
        let mut game = new_4p_game();
        let first_player = game.player_order[0];
        let initial_cash = game.players[0].cash;

        // Buy SV for face value (20)
        game.process_action_internal(&Action::Bid {
            entity_id: first_player.to_string(),
            company_sym: "SV".to_string(),
            price: 20,
        })
        .unwrap();

        // Player paid 20
        assert_eq!(game.players[0].cash, initial_cash - 20);

        // SV owned by player
        assert_eq!(game.companies[0].owner, EntityId::player(first_player));

        // 5 remaining companies
        if let Round::Auction(ref state) = game.round {
            assert_eq!(state.remaining_companies.len(), 5);
        }
    }

    #[test]
    fn auction_all_pass_reduces_cheapest_price() {
        let mut game = new_4p_game();
        let players: Vec<u32> = game.player_order.clone();

        // All 4 players pass
        for &pid in &players {
            game.process_action_internal(&Action::Pass {
                entity_id: pid.to_string(),
            })
            .unwrap();
        }

        // Discount increased by 5
        if let Round::Auction(ref state) = game.round {
            assert_eq!(state.discount, 5);
        }
    }

    #[test]
    fn auction_price_reduces_to_zero_gives_free() {
        let mut game = new_4p_game();

        // SV value = 20. Need 4 rounds of all-pass to reach 0.
        // Must use the active player each time since entity_index advances.
        for _ in 0..4 {
            for _ in 0..4 {
                let active = match &game.round {
                    Round::Auction(s) => s.active_player_id(),
                    _ => panic!("Expected auction"),
                };
                game.process_action_internal(&Action::Pass {
                    entity_id: active.to_string(),
                })
                .unwrap();
                // Check if SV was given away (stops the loop)
                if game.companies[0].owner.player_id().is_some() {
                    break;
                }
            }
            if game.companies[0].owner.player_id().is_some() {
                break;
            }
        }

        // SV should have been given away for free
        let sv_owned = game.companies[0].owner.player_id().is_some();
        assert!(
            sv_owned,
            "SV should be owned by a player after price drops to 0"
        );

        // Player didn't pay anything
        let owner_id = game.companies[0].owner.player_id().unwrap();
        let owner = game.players.iter().find(|p| p.id == owner_id).unwrap();
        assert_eq!(owner.cash, 600, "Owner should still have starting cash");
    }

    #[test]
    fn stock_market_par_prices_exist() {
        let game = new_4p_game();
        let par_prices = game.stock_market.par_prices();
        assert!(par_prices.contains(&67));
        assert!(par_prices.contains(&100));
        assert!(par_prices.len() >= 6);
    }

    #[test]
    fn stock_market_move_right() {
        let market = crate::core::StockMarket::new_1830();
        // Par price 100 at (0, 6)
        let sp = market.par_price(100).unwrap();
        assert_eq!((sp.row, sp.column), (0, 6));

        let (r, c) = market.move_right(0, 6);
        assert_eq!(market.cell_at(r, c).unwrap().price, 112);
    }

    #[test]
    fn stock_market_move_down() {
        let market = crate::core::StockMarket::new_1830();
        // Move down from par 100 at (0, 6) -> (1, 6) = 90
        let (r, c) = market.move_down(0, 6);
        assert_eq!(market.cell_at(r, c).unwrap().price, 90);
    }

    #[test]
    fn num_certs_initially_zero() {
        let game = new_4p_game();
        for player in &game.players {
            assert_eq!(game.num_certs_internal(player.id), 0);
        }
    }

    #[test]
    fn no_floated_corps_initially() {
        let game = new_4p_game();
        assert!(game.compute_operating_order().is_empty());
    }

    #[test]
    fn initial_player_values() {
        let game = new_4p_game();
        let results = game.calculate_results();
        for &value in results.values() {
            assert_eq!(value, 600);
        }
    }

    #[test]
    fn full_auction_completes_to_stock_round() {
        let mut game = new_4p_game();

        // Buy all 6 companies at face value, cycling through players
        let company_syms = ["SV", "CS", "DH", "MH", "CA", "BO"];
        let company_prices = [20, 40, 70, 110, 160, 220];

        for (i, (&sym, &price)) in company_syms.iter().zip(company_prices.iter()).enumerate() {
            let active_player = match &game.round {
                Round::Auction(s) => s.active_player_id(),
                _ => panic!("Expected auction round at company {}", i),
            };

            game.process_action_internal(&Action::Bid {
                entity_id: active_player.to_string(),
                company_sym: sym.to_string(),
                price,
            })
            .unwrap();
        }

        // After BO purchase, there's a pending par for B&O
        if let Round::Auction(ref s) = game.round {
            if s.pending_par.is_some() {
                let active = s.active_player_id();
                game.process_action_internal(&Action::Par {
                    entity_id: active.to_string(),
                    corporation_sym: "B&O".to_string(),
                    share_price: 100,
                })
                .unwrap();
            }
        }

        assert!(
            matches!(game.round, Round::Stock(_)),
            "Expected Stock round after auction, got {}",
            game.round.round_type_str()
        );
    }

    /// Helper: skip through an operating round by passing all blocking steps.
    fn skip_or(game: &mut BaseGame) {
        while matches!(&game.round, Round::Operating(_)) {
            let entity_id = game.round_state.active_entity_id.0.clone();
            if entity_id.is_empty() {
                break;
            }
            game.process_action_internal(&Action::Pass { entity_id })
                .unwrap_or_else(|_| {});
        }
    }

    /// Helper: buy all auction companies at face value, returning game in Stock round.
    fn game_in_stock_round() -> BaseGame {
        let mut game = new_4p_game();
        let company_syms = ["SV", "CS", "DH", "MH", "CA", "BO"];
        let company_prices = [20, 40, 70, 110, 160, 220];

        for (&sym, &price) in company_syms.iter().zip(company_prices.iter()) {
            let active = match &game.round {
                Round::Auction(s) => s.active_player_id(),
                _ => panic!(
                    "Expected auction round, got {}",
                    game.round.round_type_str()
                ),
            };
            game.process_action_internal(&Action::Bid {
                entity_id: active.to_string(),
                company_sym: sym.to_string(),
                price,
            })
            .unwrap();
        }

        // After BO purchase, there's a pending par for B&O — resolve it
        if let Round::Auction(ref s) = game.round {
            if s.pending_par.is_some() {
                let active = s.active_player_id();
                game.process_action_internal(&Action::Par {
                    entity_id: active.to_string(),
                    corporation_sym: "B&O".to_string(),
                    share_price: 100,
                })
                .unwrap();
            }
        }

        assert!(
            matches!(game.round, Round::Stock(_)),
            "Expected Stock round, got {}",
            game.round.round_type_str()
        );
        game
    }

    #[test]
    fn stock_round_par_corporation() {
        let mut game = game_in_stock_round();
        let player_id = match &game.round {
            Round::Stock(s) => s.current_player_id(),
            _ => panic!("Expected stock round"),
        };
        let player_idx = game.player_index(player_id).unwrap();
        let initial_cash = game.players[player_idx].cash;

        // Par PRR at 100
        game.process_action_internal(&Action::Par {
            entity_id: player_id.to_string(),
            corporation_sym: "PRR".to_string(),
            share_price: 100,
        })
        .unwrap();

        // Player paid 200 (2 * 100 for president share)
        assert_eq!(game.players[player_idx].cash, initial_cash - 200);

        // PRR should have IPO and share price set
        let prr_idx = game.corp_idx["PRR"];
        assert!(game.corporations[prr_idx].ipo_price.is_some());
        assert_eq!(
            game.corporations[prr_idx]
                .share_price
                .as_ref()
                .unwrap()
                .price,
            100
        );

        // President share owned by player
        assert_eq!(
            game.corporations[prr_idx].shares[0].owner,
            EntityId::player(player_id)
        );

        // PRR not yet floated (only 20% sold)
        assert!(!game.corporations[prr_idx].floated);
    }

    #[test]
    fn stock_round_float_at_60_percent() {
        let mut game = game_in_stock_round();

        // Par PRR at 100 (20% sold — president share, auto-passes to next player)
        let p1 = match &game.round {
            Round::Stock(s) => s.current_player_id(),
            _ => panic!(),
        };
        game.process_action_internal(&Action::Par {
            entity_id: p1.to_string(),
            corporation_sym: "PRR".to_string(),
            share_price: 100,
        })
        .unwrap();

        // Player 0 already has 10% from CA company ability.
        // President share = 20%, so 30% sold. Need 3 more buys to reach 60%.
        // After par, current player bought, so pass first.
        let parrer = match &game.round {
            Round::Stock(s) => s.current_player_id(),
            _ => panic!(),
        };
        game.process_action_internal(&Action::Pass {
            entity_id: parrer.to_string(),
        })
        .unwrap();

        for i in 0..3 {
            let buyer = match &game.round {
                Round::Stock(s) => s.current_player_id(),
                _ => panic!("Expected Stock round at buy {}", i),
            };
            game.process_action_internal(&Action::BuyShares {
                entity_id: buyer.to_string(),
                corporation_sym: "PRR".to_string(),
                shares: vec![],
                percent: 10,
                source: "ipo".to_string(),
                share_indices: vec![],
            })
            .unwrap_or_else(|e| panic!("Float buy {} by player {} failed: {}", i, buyer, e));
            game.process_action_internal(&Action::Pass {
                entity_id: buyer.to_string(),
            })
            .unwrap();
            skip_or(&mut game);
        }

        // PRR should be floated now (60% sold from IPO)
        let prr_idx = game.corp_idx["PRR"];
        assert!(
            game.corporations[prr_idx].floated,
            "PRR should have floated at 60%"
        );

        // Corp should have received 100 * 10 = 1000 from bank
        assert_eq!(game.corporations[prr_idx].cash, 1000);
    }

    #[test]
    fn stock_round_sell_moves_price_down() {
        let mut game = game_in_stock_round();

        // Par PRR at 67 (cheapest par) and buy shares until someone can sell
        let first = match &game.round {
            Round::Stock(s) => s.current_player_id(),
            _ => panic!(),
        };
        game.process_action_internal(&Action::Par {
            entity_id: first.to_string(),
            corporation_sym: "PRR".to_string(),
            share_price: 67,
        })
        .unwrap();
        game.process_action_internal(&Action::Pass {
            entity_id: first.to_string(),
        })
        .unwrap();
        skip_or(&mut game);

        // Buy 3 more shares to float (player 0 already has 10% from CA = 60% total)
        for i in 0..3 {
            let pid = match &game.round {
                Round::Stock(s) => s.current_player_id(),
                _ => panic!("Expected Stock round at buy {}", i),
            };
            game.process_action_internal(&Action::BuyShares {
                entity_id: pid.to_string(),
                corporation_sym: "PRR".to_string(),
                shares: vec![],
                percent: 10,
                source: "ipo".to_string(),
                share_indices: vec![],
            })
            .unwrap_or_else(|e| panic!("Buy {} by player {} failed: {}", i, pid, e));
            game.process_action_internal(&Action::Pass {
                entity_id: pid.to_string(),
            })
            .unwrap();
            skip_or(&mut game);
        }

        // Now find a player who owns a non-president PRR share and sell it
        let seller = match &game.round {
            Round::Stock(s) => s.current_player_id(),
            _ => panic!(
                "Expected Stock round after OR completion, got {}",
                game.round.round_type_str()
            ),
        };

        let prr_idx = game.corp_idx["PRR"];
        let seller_eid = EntityId::player(seller);
        let has_share = game.corporations[prr_idx]
            .shares
            .iter()
            .any(|s| s.owner == seller_eid && !s.president);

        if has_share {
            let sp_before = game.corporations[prr_idx]
                .share_price
                .as_ref()
                .unwrap()
                .clone();

            game.process_action_internal(&Action::SellShares {
                entity_id: seller.to_string(),
                corporation_sym: "PRR".to_string(),
                shares: vec![],
                percent: 10,
                share_indices: vec![],
            })
            .unwrap();

            let sp_after = game.corporations[prr_idx]
                .share_price
                .as_ref()
                .unwrap()
                .clone();
            // In 1830, sell moves price DOWN (row increases)
            assert!(
                sp_after.row > sp_before.row || sp_after.price <= sp_before.price,
                "Price should move down after sell: row {}->{}, price {}->{}",
                sp_before.row,
                sp_after.row,
                sp_before.price,
                sp_after.price
            );
        }
    }

    #[test]
    fn stock_round_cant_buy_corp_sold_this_round() {
        let mut game = game_in_stock_round();

        // Par PRR at 100 then pass
        let p1 = match &game.round {
            Round::Stock(s) => s.current_player_id(),
            _ => panic!(),
        };
        game.process_action_internal(&Action::Par {
            entity_id: p1.to_string(),
            corporation_sym: "PRR".to_string(),
            share_price: 100,
        })
        .unwrap();
        game.process_action_internal(&Action::Pass {
            entity_id: p1.to_string(),
        })
        .unwrap();
        skip_or(&mut game);

        // Other players buy shares then pass
        for _ in 0..2 {
            let pid = match &game.round {
                Round::Stock(s) => s.current_player_id(),
                _ => panic!("Expected Stock round"),
            };
            game.process_action_internal(&Action::BuyShares {
                entity_id: pid.to_string(),
                corporation_sym: "PRR".to_string(),
                shares: vec![],
                percent: 10,
                source: "ipo".to_string(),
                share_indices: vec![],
            })
            .unwrap();
            game.process_action_internal(&Action::Pass {
                entity_id: pid.to_string(),
            })
            .unwrap();
            skip_or(&mut game);
        }

        // P1's turn: buy another share
        let current = match &game.round {
            Round::Stock(s) => s.current_player_id(),
            _ => panic!("Expected Stock round"),
        };
        game.process_action_internal(&Action::BuyShares {
            entity_id: current.to_string(),
            corporation_sym: "PRR".to_string(),
            shares: vec![],
            percent: 10,
            source: "ipo".to_string(),
            share_indices: vec![],
        })
        .unwrap();

        // Sell 10%
        game.process_action_internal(&Action::SellShares {
            entity_id: current.to_string(),
            corporation_sym: "PRR".to_string(),
            shares: vec![],
            percent: 10,
            share_indices: vec![],
        })
        .unwrap();

        // Now try to buy PRR — should fail (sold this round)
        let result = game.process_action_internal(&Action::BuyShares {
            entity_id: current.to_string(),
            corporation_sym: "PRR".to_string(),
            shares: vec![],
            percent: 10,
            source: "market".to_string(),
            share_indices: vec![],
        });
        assert!(
            result.is_err(),
            "Should not be able to buy a corp sold this round"
        );
    }

    #[test]
    fn company_revenue_counted_in_player_value() {
        let game = game_in_stock_round();
        let results = game.calculate_results();

        // After auction + pending par:
        // - Companies cancel out (cash paid = face value in score)
        // - Player 1 got B&O 20% president share for FREE (BO company ability)
        //   B&O parred at 100, so 20% = $200 value
        // - Player 0 got PRR 10% share for FREE (CA company ability)
        //   PRR is not parred yet, so it has no market value
        // Total = starting_cash + B&O free share value
        let total: i32 = results.values().sum();
        let starting_total = 600 * 4;
        let free_share_value = 200; // 20% * $100 par
        assert_eq!(total, starting_total + free_share_value);
    }

    #[test]
    fn clone_for_search_preserves_state() {
        let game = new_4p_game();
        let clone = game.clone_for_search();

        assert_eq!(clone.players.len(), game.players.len());
        assert_eq!(clone.bank.cash, game.bank.cash);
        assert_eq!(clone.corporations.len(), game.corporations.len());
        assert_eq!(clone.move_number, game.move_number);
    }

    #[test]
    fn payout_companies_distributes_revenue() {
        let mut game = new_4p_game();

        // Manually assign SV to player 1 (skip auction)
        game.companies[0].owner = EntityId::player(game.players[0].id);
        let cash_before = game.players[0].cash;
        let bank_before = game.bank.cash;

        game.payout_companies();

        // SV has revenue 5
        assert_eq!(
            game.players[0].cash - cash_before,
            5,
            "SV owner should receive $5 revenue"
        );
        assert_eq!(game.bank.cash, bank_before - 5);
    }

    #[test]
    fn phase_advance_rusts_trains() {
        let mut game = new_4p_game();

        // Manually set up a corp with 2-trains to test rusting
        let prr_idx = game.corp_idx["PRR"];
        game.corporations[prr_idx].floated = true;
        game.corporations[prr_idx]
            .trains
            .push(crate::entities::Train::new("2".to_string(), 2, 80));
        game.corporations[prr_idx]
            .trains
            .push(crate::entities::Train::new("2".to_string(), 2, 80));
        assert_eq!(game.corporations[prr_idx].trains.len(), 2);

        // Advance to phase 4 (should rust all 2-trains)
        game.check_phase_advance("4");
        assert_eq!(game.phase.name, "4");
        assert_eq!(
            game.corporations[prr_idx].trains.len(),
            0,
            "All 2-trains should have rusted"
        );
    }

    #[test]
    fn close_all_companies_on_phase_5() {
        let mut game = game_in_stock_round();

        // Verify companies are not closed
        assert!(!game.companies[0].closed);

        game.check_phase_advance("5");
        assert_eq!(game.phase.name, "5");

        // All companies should be closed
        for c in &game.companies {
            assert!(c.closed, "Company {} should be closed after phase 5", c.sym);
        }
    }

    #[test]
    fn preprinted_hexes_have_paths() {
        let game = new_4p_game();

        // H12 (gray city: Altoona) should have paths from DSL
        let h12 = &game.hexes[game.hex_idx["H12"]];
        assert!(
            !h12.tile.paths.is_empty(),
            "H12 should have paths from preprinted DSL, got none"
        );
        // H12 DSL: "city=revenue:10;path=a:1,b:_0;path=a:4,b:_0;path=a:1,b:4"
        // 3 paths: edge1→city0, edge4→city0, edge1→edge4
        assert_eq!(h12.tile.paths.len(), 3, "H12 should have 3 paths");
        assert_eq!(h12.tile.cities.len(), 1);
        assert_eq!(h12.tile.cities[0].revenue, 10);

        // F2 (red offboard: Chicago) should have paths
        let f2 = &game.hexes[game.hex_idx["F2"]];
        assert!(
            !f2.tile.paths.is_empty(),
            "F2 should have paths from preprinted DSL"
        );
        // F2 DSL has 3 paths: edge3→offboard, edge4→offboard, edge5→offboard
        assert_eq!(f2.tile.paths.len(), 3, "F2 should have 3 paths");
        assert_eq!(f2.tile.offboards.len(), 1);

        // I15 (yellow preprinted: Baltimore) should have paths and label
        let i15 = &game.hexes[game.hex_idx["I15"]];
        assert!(
            !i15.tile.paths.is_empty(),
            "I15 should have paths from preprinted DSL"
        );
        assert_eq!(i15.tile.label, Some("B".to_string()));

        // G19 (yellow preprinted: NY) should have label and 2 cities
        let g19 = &game.hexes[game.hex_idx["G19"]];
        assert_eq!(g19.tile.label, Some("NY".to_string()));
        assert_eq!(g19.tile.cities.len(), 2);
        assert!(!g19.tile.paths.is_empty());
    }

    #[test]
    fn h12_is_adjacent_to_h10() {
        let game = new_4p_game();

        // H12 and H10 should be adjacent. Same letter row 'H' (index 7).
        // H12: (7, 12), H10: (7, 10). Diff = (0, -2) which is direction 4 (left).
        // So H12 direction 4 → H10, and H10 direction 1 → H12.
        let h12_neighbors = game.hex_adjacency.get("H12");
        assert!(h12_neighbors.is_some(), "H12 should have neighbors");
        let h12_n = h12_neighbors.unwrap();

        // Check that H10 is a neighbor (direction should be left=4 based on HEX_DELTAS)
        let has_h10 = h12_n.values().any(|v| v == "H10");
        assert!(
            has_h10,
            "H12 should be adjacent to H10, but neighbors are: {:?}",
            h12_n
        );
    }

    #[test]
    fn laid_tile_has_paths_from_catalog() {
        let game = new_4p_game();

        // Tile 57 should be in the catalog
        assert!(
            game.tile_catalog.contains_key("57"),
            "Tile 57 should be in catalog"
        );
        let t57 = &game.tile_catalog["57"];
        assert_eq!(t57.paths.len(), 2, "Tile 57 should have 2 paths");
        assert_eq!(t57.cities.len(), 1, "Tile 57 should have 1 city");
    }

    #[test]
    fn tile_from_def_produces_paths() {
        let game = new_4p_game();

        // Build tile 57 at rotation 1 via tile_from_def
        let tile_def = &game.tile_catalog["57"];
        let tile = BaseGame::tile_from_def(tile_def, 1);

        // tile 57 DSL: "city=revenue:20;path=a:0,b:_0;path=a:_0,b:3"
        // Rotated by 1: edge 0→1, edge 3→4
        // So paths should be: Edge(1)→City(0) and City(0)→Edge(4)
        assert_eq!(
            tile.paths.len(),
            2,
            "Tile 57 should have 2 paths after tile_from_def"
        );
        assert_eq!(tile.cities.len(), 1, "Tile 57 should have 1 city");
        assert_eq!(tile.rotation, 1);

        // Verify the actual path endpoints
        use crate::tiles::PathEndpoint;
        let has_edge_1_to_city = tile.paths.iter().any(|p| {
            (p.a == PathEndpoint::Edge(1) && p.b == PathEndpoint::City(0))
                || (p.a == PathEndpoint::City(0) && p.b == PathEndpoint::Edge(1))
        });
        let has_city_to_edge_4 = tile.paths.iter().any(|p| {
            (p.a == PathEndpoint::Edge(4) && p.b == PathEndpoint::City(0))
                || (p.a == PathEndpoint::City(0) && p.b == PathEndpoint::Edge(4))
        });
        assert!(has_edge_1_to_city, "Should have path Edge(1)↔City(0)");
        assert!(has_city_to_edge_4, "Should have path City(0)↔Edge(4)");
    }

    #[test]
    fn f18_adjacency_matches_python() {
        let game = new_4p_game();
        let f18_adj = game.hex_adjacency.get("F18").unwrap();
        // Python: F18.edge0→G17, edge1→F16, edge2→E17, edge3→E19, edge4→F20, edge5→G19
        assert_eq!(
            f18_adj.get(&0),
            Some(&"G17".to_string()),
            "F18.0 should be G17"
        );
        assert_eq!(
            f18_adj.get(&1),
            Some(&"F16".to_string()),
            "F18.1 should be F16"
        );
        assert_eq!(
            f18_adj.get(&2),
            Some(&"E17".to_string()),
            "F18.2 should be E17"
        );
        assert_eq!(
            f18_adj.get(&3),
            Some(&"E19".to_string()),
            "F18.3 should be E19"
        );
        assert_eq!(
            f18_adj.get(&4),
            Some(&"F20".to_string()),
            "F18.4 should be F20"
        );
        assert_eq!(
            f18_adj.get(&5),
            Some(&"G19".to_string()),
            "F18.5 should be G19"
        );
    }

    #[test]
    fn e19_to_g19_connectivity_through_f18() {
        // Manually construct the exact scenario:
        // E19: tile 57 at rotation 0 (edges 0, 3), NYC token
        // F18: tile 8 at rotation 3 (edges 3, 5)
        // G19: tile 54 at rotation 0 (edges 0, 1, 2, 3), 2 cities
        // Walk: E19.city→edge0 → F18(enter edge3→exit edge5) → G19(enter edge2→City1)
        use crate::tiles;
        let catalog = tiles::tile_catalog_1830();

        let mut tile_e19 = BaseGame::tile_from_def(&catalog["57"], 0);
        tile_e19.id = "57-1".into();
        tile_e19.name = "57-1".into();
        {
            let city = &mut tile_e19.cities[0];
            let mut tok = Token::new("NYC".into(), 0);
            tok.used = true;
            tok.city_hex_id = "E19".into();
            city.tokens[0] = Some(tok);
        }

        let mut tile_f18 = BaseGame::tile_from_def(&catalog["8"], 3);
        tile_f18.id = "8-2".into();
        tile_f18.name = "8-2".into();

        let tile_g19 = BaseGame::tile_from_def(&catalog["54"], 0);
        // tile_g19 should have 2 cities and paths on edges 0,1,2,3

        let hexes = vec![
            crate::graph::Hex::new("E19".into(), tile_e19),
            crate::graph::Hex::new("F18".into(), tile_f18),
            crate::graph::Hex::new("G19".into(), tile_g19),
        ];
        let hex_idx: HashMap<String, usize> =
            [("E19".into(), 0usize), ("F18".into(), 1), ("G19".into(), 2)].into();

        // Adjacency matching Python:
        // E19.edge0→F18, F18.edge3→E19, F18.edge5→G19, G19.edge2→F18
        let adjacency: HashMap<String, HashMap<u8, String>> = [
            (
                "E19".to_string(),
                [(0u8, "F18".to_string()), (3u8, "D20".to_string())].into(),
            ),
            (
                "F18".to_string(),
                [(3u8, "E19".to_string()), (5u8, "G19".to_string())].into(),
            ),
            ("G19".to_string(), [(2u8, "F18".to_string())].into()),
        ]
        .into();

        let token_positions = vec![("E19".to_string(), 0usize)];
        let mut cache = crate::map::GraphCache::new();
        let graph =
            cache.get_or_compute("NYC", &hexes, &hex_idx, &adjacency, &token_positions, &[]);

        // E19 city should be found
        let e19_city = crate::map::NodeId {
            hex_id: "E19".into(),
            node_type: crate::map::NodeType::City,
            index: 0,
        };
        assert!(
            graph.connected_nodes.contains(&e19_city),
            "E19 city should be connected"
        );

        // G19 city 1 should be found (via F18 pass-through)
        let g19_city1 = crate::map::NodeId {
            hex_id: "G19".into(),
            node_type: crate::map::NodeType::City,
            index: 1,
        };
        assert!(
            graph.connected_nodes.contains(&g19_city1),
            "G19 city 1 should be reachable through F18. Nodes: {:?}",
            graph.connected_nodes
        );
    }

    #[test]
    fn connectivity_after_manual_tile_place_and_token() {
        // Simulate: E19 has tile 57 at rot 0, NYC token in city 0
        // D20 has tile 57 at rot 0 (city with no token)
        // E19.edge3 → D20, D20.edge0 → E19
        // Connectivity from E19 should find D20's city

        let game = new_4p_game();

        // Build tiles from catalog
        let tile_def_57 = &game.tile_catalog["57"];

        // E19: tile 57 at rotation 0 → edges 0,3. City(0) connected to Edge(0) and Edge(3)
        let mut tile_e19 = BaseGame::tile_from_def(tile_def_57, 0);
        tile_e19.id = "57-0".to_string();
        tile_e19.name = "57-0".to_string();
        // Place NYC token
        {
            let city = &mut tile_e19.cities[0];
            let mut tok = Token::new("NYC".to_string(), 0);
            tok.used = true;
            tok.city_hex_id = "E19".to_string();
            city.tokens[0] = Some(tok);
        }

        // D20: tile 57 at rotation 0 → city connected to Edge(0) and Edge(3)
        let mut tile_d20 = BaseGame::tile_from_def(tile_def_57, 0);
        tile_d20.id = "57-1".to_string();
        tile_d20.name = "57-1".to_string();

        let hexes = vec![
            crate::graph::Hex::new("E19".to_string(), tile_e19),
            crate::graph::Hex::new("D20".to_string(), tile_d20),
        ];
        let hex_idx: HashMap<String, usize> =
            [("E19".to_string(), 0), ("D20".to_string(), 1)].into();

        // E19 edge 3 → D20, D20 edge 0 → E19
        let adjacency: HashMap<String, HashMap<u8, String>> = [
            ("E19".to_string(), [(3u8, "D20".to_string())].into()),
            ("D20".to_string(), [(0u8, "E19".to_string())].into()),
        ]
        .into();

        let token_positions = vec![("E19".to_string(), 0usize)];

        let mut graph_cache = crate::map::GraphCache::new();
        let graph =
            graph_cache.get_or_compute("NYC", &hexes, &hex_idx, &adjacency, &token_positions, &[]);

        // NYC should see both its own city and D20's city
        assert!(
            graph.connected_nodes.len() >= 2,
            "NYC should reach at least 2 nodes (E19 city + D20 city), got {}",
            graph.connected_nodes.len()
        );

        let d20_city = crate::map::NodeId {
            hex_id: "D20".to_string(),
            node_type: crate::map::NodeType::City,
            index: 0,
        };
        assert!(
            graph.connected_nodes.contains(&d20_city),
            "NYC should reach D20's city. Connected nodes: {:?}",
            graph.connected_nodes
        );
    }
}
