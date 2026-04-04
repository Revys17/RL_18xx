//! Operating round logic for 1830.
//!
//! Each floated corporation operates in share price order (highest first).
//! Steps: LayTile → PlaceToken → RunRoutes → Dividend → BuyTrain.
//! At each step, the corporation can also buy a company from its president.

use crate::actions::{Action, DividendKind, GameError, RouteData};
use crate::entities::EntityId;
use crate::game::BaseGame;
use crate::graph::City;
use crate::rounds::{OperatingState, OperatingStep};

impl BaseGame {
    /// Process an action during the operating round.
    ///
    /// The OR step order for 1830 is:
    ///   Non-blocking: Bankrupt, Exchange, SpecialTrack, SpecialToken, BuyCompany, HomeToken
    ///   Blocking:     Track, Token, Route, Dividend, DiscardTrain, BuyTrain, BuyCompany(blocking)
    ///
    /// Non-blocking steps are handled elsewhere (company exchanges, abilities).
    /// BuyCompany is accepted at any step (non-blocking version). The blocking
    /// BuyCompany at the end requires an explicit pass to end the corp's turn.
    pub fn process_operating_action(&mut self, action: &Action) -> Result<(), GameError> {
        let state = match &self.round {
            crate::rounds::Round::Operating(s) => s.clone(),
            _ => return Err(GameError::new("Not in operating round")),
        };

        // If a corp is crowded (over train limit after phase change), it MUST
        // discard before anything else happens. Only accept discard_train from
        // the first crowded corp; reject all other actions.
        let crowded_corp = match &self.round {
            crate::rounds::Round::Operating(s) if !s.crowded_corps.is_empty() => {
                Some(s.crowded_corps[0].clone())
            }
            _ => None,
        };
        if let Some(ref required_corp) = crowded_corp {
            if let Action::DiscardTrain {
                entity_id,
                train_name,
            } = action
            {
                if entity_id != required_corp {
                    return Err(GameError::new(format!(
                        "Expected discard_train from {} (crowded), got {}",
                        required_corp, entity_id
                    )));
                }
                self.or_process_discard_train(&state, entity_id, train_name)?;
                // If no more crowded corps, go back to BuyTrain for the operating
                // corp (matching Python's "unpass BuyTrain" after discard).
                let still_crowded = match &self.round {
                    crate::rounds::Round::Operating(s) => !s.crowded_corps.is_empty(),
                    _ => false,
                };
                if !still_crowded {
                    // Go back to BuyTrain for the operating corp (Python's
                    // "unpass BuyTrain"). Then run skip_steps starting from
                    // BuyTrain — if the corp can't buy, it auto-advances
                    // through BuyCompany → Done → next corp.
                    if let crate::rounds::Round::Operating(ref mut s) = self.round {
                        s.step = OperatingStep::BuyTrain;
                    }
                    self.skip_steps();
                    let is_done = matches!(
                        &self.round,
                        crate::rounds::Round::Operating(s) if !s.finished && s.step == OperatingStep::Done
                    );
                    if is_done {
                        if let crate::rounds::Round::Operating(ref mut s) = self.round {
                            s.advance_to_next_corp();
                        }
                        if !matches!(&self.round, crate::rounds::Round::Operating(s) if s.finished) {
                            self.start_operating();
                        }
                    }
                }
                return Ok(());
            } else {
                return Err(GameError::new(format!(
                    "{} must discard a train (over train limit)",
                    required_corp
                )));
            }
        }

        // Emergency sell: president sells shares during BuyTrain step to fund
        // a forced train purchase. Accepted when a player entity sells shares
        // while the OR is at BuyTrain step.
        if let Action::SellShares {
            entity_id,
            corporation_sym,
            percent,
            ..
        } = action
        {
            if state.step == OperatingStep::BuyTrain {
                return self.or_emergency_sell(&state, entity_id, corporation_sym, *percent);
            }
        }

        // Process the action. BuyCompany is accepted at any step (non-blocking),
        // but still needs the skip_steps/Done check after processing.
        match action {
            Action::BuyCompany {
                entity_id,
                company_sym,
                price,
            } => self.or_process_buy_company(&state, entity_id, company_sym, *price),
            Action::LayTile {
                entity_id,
                hex_id,
                tile_id,
                rotation,
            } => self.or_process_lay_tile(&state, entity_id, hex_id, tile_id, *rotation),
            Action::PlaceToken {
                entity_id,
                hex_id,
                city_index,
            } => self.or_process_place_token(&state, entity_id, hex_id, *city_index),
            Action::RunRoutes {
                entity_id,
                routes,
                extra_revenue,
            } => self.or_process_run_routes(&state, entity_id, routes, *extra_revenue),
            Action::Dividend { entity_id, kind } => {
                self.or_process_dividend(&state, entity_id, kind)
            }
            Action::BuyTrain {
                entity_id,
                train_name,
                price,
                from,
                exchange,
                ..
            } => self.or_process_buy_train(
                &state,
                entity_id,
                train_name,
                *price,
                from,
                exchange.as_deref(),
            ),
            Action::DiscardTrain {
                entity_id,
                train_name,
            } => self.or_process_discard_train(&state, entity_id, train_name),
            Action::Pass { entity_id } => self.or_process_pass(&state, entity_id),
            _ => Err(GameError::new(format!(
                "Invalid action in operating round: {}",
                action.action_type()
            ))),
        }?;

        // After each action, advance past non-blocking steps
        self.skip_steps();

        // If all steps complete (reached Done), advance to next corp.
        // Then start_operating handles setup + skip for subsequent corps.
        let is_done = matches!(
            &self.round,
            crate::rounds::Round::Operating(s) if !s.finished && s.step == OperatingStep::Done
        );
        if is_done {
            if let crate::rounds::Round::Operating(ref mut s) = self.round {
                s.advance_to_next_corp();
            }
            if !matches!(&self.round, crate::rounds::Round::Operating(s) if s.finished) {
                self.start_operating();
            }
        }

        Ok(())
    }

    fn or_process_lay_tile(
        &mut self,
        state: &OperatingState,
        _entity_id: &str,
        hex_id: &str,
        tile_id: &str,
        rotation: u8,
    ) -> Result<(), GameError> {
        let mut new_state = state.clone();

        if new_state.step != OperatingStep::LayTile {
            return Err(GameError::new("Not in LayTile step"));
        }

        if new_state.num_laid_track >= 1 {
            return Err(GameError::new("Already laid a tile this turn"));
        }

        let hex_idx = *self
            .hex_idx
            .get(hex_id)
            .ok_or_else(|| GameError::new(format!("Unknown hex: {}", hex_id)))?;

        // Pay terrain cost
        let terrain_cost: i32 = self.hexes[hex_idx]
            .tile
            .upgrades
            .iter()
            .map(|u| u.cost)
            .sum();

        if terrain_cost > 0 {
            let corp_sym = new_state
                .current_corp_sym()
                .ok_or_else(|| GameError::new("No current corp"))?
                .to_string();
            let corp_idx = self.corp_idx[&corp_sym];
            if self.corporations[corp_idx].cash < terrain_cost {
                return Err(GameError::new(format!(
                    "{} cannot afford terrain cost {}",
                    corp_sym, terrain_cost
                )));
            }
            self.corporations[corp_idx].cash -= terrain_cost;
            self.bank.cash += terrain_cost;
        }

        // Replace the tile. Tile IDs may have variant suffix like "57-0"; strip it.
        let base_tile_id = tile_id.split('-').next().unwrap_or(tile_id);
        let tile_count = self
            .tile_counts_remaining
            .get(base_tile_id)
            .copied()
            .unwrap_or(0);
        if tile_count == 0 {
            return Err(GameError::new(format!(
                "No tiles of type {} remaining",
                base_tile_id
            )));
        }

        // Return the old tile to the supply (if it's a placed tile, not a preprinted one)
        let old_tile_name = self.hexes[hex_idx].tile.name.clone();
        let old_base = old_tile_name.split('-').next().unwrap_or(&old_tile_name);
        if !old_base.starts_with("preprinted") {
            *self
                .tile_counts_remaining
                .entry(old_base.to_string())
                .or_insert(0) += 1;
        }

        // Create new tile from catalog (with paths, color, label) or fallback
        let old_tile = &self.hexes[hex_idx].tile;
        let old_cities = old_tile.cities.clone();
        let old_tile_rotation = old_tile.rotation;

        let mut new_tile = if let Some(tile_def) = self.tile_catalog.get(base_tile_id) {
            let mut t = crate::game::BaseGame::tile_from_def(tile_def, rotation);
            t.id = tile_id.to_string();
            t.name = tile_id.to_string();
            t
        } else {
            let mut t = crate::graph::Tile::new(tile_id.to_string(), tile_id.to_string());
            t.rotation = rotation;
            t
        };

        // Determine if tokens on the old tile had an ambiguous destination.
        // When the old tile has cities with no exits (placeholder preprinted cities
        // like E11's 0-revenue unconnected cities) and the new tile has multiple
        // cities, the corp must choose which city the token goes in (via explicit
        // place_token action). Cities WITH exits have well-defined positions that
        // map unambiguously to the new tile.
        let old_has_token = old_cities
            .iter()
            .any(|c| c.tokens.iter().any(|t| t.is_some()));
        let old_cities_are_placeholders = old_cities.iter().all(|c| c.revenue == 0);
        let new_has_multiple_cities = new_tile.cities.len() > 1;
        let needs_token_choice =
            old_has_token && old_cities_are_placeholders && new_has_multiple_cities;

        // Preserve existing tokens: map old cities to new cities
        if !old_cities.is_empty() && !needs_token_choice {
            if new_tile.cities.is_empty() {
                // Catalog had no cities; check tile_cities fallback
                if let Some(city_slots) = crate::title::g1830::tile_cities(base_tile_id) {
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
                    new_tile.cities = old_cities.clone();
                }
            } else {
                // Transfer tokens from old cities to new cities.
                // Use exit-based mapping: match old city to new city by finding
                // the new city whose exits are a superset of the old city's exits.
                // This handles OO tile upgrades where city ordering changes.
                let old_tile_base = old_tile_name.split('-').next().unwrap_or(&old_tile_name);
                let old_exits = self.city_exits_from_catalog(old_tile_base, old_tile_rotation);
                let new_exits = self.city_exits_from_catalog(base_tile_id, rotation);

                for (old_ci, old_city) in old_cities.iter().enumerate() {
                    let has_token = old_city.tokens.iter().any(|t| t.is_some());
                    if !has_token {
                        continue;
                    }

                    // Find the matching new city by exit overlap
                    let old_city_exits = old_exits.get(old_ci).cloned().unwrap_or_default();
                    let target_new_ci = if old_city_exits.is_empty() {
                        // No exit info — fall back to positional mapping
                        Some(old_ci)
                    } else {
                        new_exits.iter().enumerate().find_map(|(nci, ne)| {
                            if old_city_exits.iter().any(|oe| ne.contains(oe)) {
                                Some(nci)
                            } else {
                                None
                            }
                        })
                    };

                    let dest_ci = target_new_ci.unwrap_or(old_ci);
                    if dest_ci < new_tile.cities.len() {
                        for (j, old_tok) in old_city.tokens.iter().enumerate() {
                            if let Some(tok) = old_tok {
                                let new_city = &mut new_tile.cities[dest_ci];
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

        // Reset corp tokens that were on the old tile, then re-mark those that
        // survived the transfer onto the new tile.
        for old_city in &old_cities {
            for old_tok in old_city.tokens.iter().flatten() {
                if let Some(&ci) = self.corp_idx.get(old_tok.corporation_id.as_str()) {
                    for ct in &mut self.corporations[ci].tokens {
                        if ct.used && ct.city_hex_id == hex_id {
                            ct.used = false;
                            ct.city_hex_id = String::new();
                        }
                    }
                }
            }
        }

        self.hexes[hex_idx].tile = new_tile;

        // Re-mark corp tokens that survived the transfer onto the new tile.
        for city in &self.hexes[hex_idx].tile.cities {
            for tok in city.tokens.iter().flatten() {
                if let Some(&ci) = self.corp_idx.get(tok.corporation_id.as_str()) {
                    for ct in &mut self.corporations[ci].tokens {
                        if !ct.used && ct.corporation_id == tok.corporation_id {
                            ct.used = true;
                            ct.city_hex_id = hex_id.to_string();
                            break;
                        }
                    }
                }
            }
        }

        self.clear_graph_cache();

        // Decrement tile count
        let base_id = base_tile_id.to_string();
        if let Some(count) = self.tile_counts_remaining.get_mut(&base_id) {
            *count -= 1;
            if *count == 0 {
                self.tile_counts_remaining.remove(&base_id);
            }
        }

        new_state.num_laid_track += 1;

        // Auto-pass Track step if no more tiles can be laid (1 per turn in 1830)
        if new_state.num_laid_track >= 1 {
            new_state.step = new_state.step.next();
        }

        self.round = crate::rounds::Round::Operating(new_state);
        self.update_round_state();
        Ok(())
    }

    fn or_process_place_token(
        &mut self,
        state: &OperatingState,
        _entity_id: &str,
        hex_id: &str,
        city_index: u8,
    ) -> Result<(), GameError> {
        let mut new_state = state.clone();

        if new_state.step != OperatingStep::PlaceToken {
            return Err(GameError::new("Not in PlaceToken step"));
        }

        if new_state.num_placed_token >= 1 {
            return Err(GameError::new("Already placed a token this turn"));
        }

        let corp_sym = new_state
            .current_corp_sym()
            .ok_or_else(|| GameError::new("No current corp"))?
            .to_string();
        let corp_idx = self.corp_idx[&corp_sym];

        // Find the next available token
        let token_idx = self.corporations[corp_idx]
            .next_token_index()
            .ok_or_else(|| GameError::new("No tokens available"))?;
        let token_cost = self.corporations[corp_idx].tokens[token_idx].price;

        if self.corporations[corp_idx].cash < token_cost {
            return Err(GameError::new(format!(
                "{} cannot afford token cost {}",
                corp_sym, token_cost
            )));
        }

        // Resolve hex_id: may be a direct hex ID or "__tile:<name>" from city-based parsing
        let resolved_hex_id = if let Some(tile_instance) = hex_id.strip_prefix("__tile:") {
            // Try matching tile instance (e.g., "57-0"), tile name, or hex ID
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
                .ok_or_else(|| GameError::new(format!("No hex with tile {}", tile_instance)))?
        } else {
            hex_id.to_string()
        };

        // Place token in city
        let hex_idx = *self
            .hex_idx
            .get(resolved_hex_id.as_str())
            .ok_or_else(|| GameError::new(format!("Unknown hex: {}", resolved_hex_id)))?;

        let city = self.hexes[hex_idx]
            .tile
            .cities
            .get_mut(city_index as usize)
            .ok_or_else(|| GameError::new("Invalid city index"))?;

        // Find empty slot
        let slot_idx = city
            .tokens
            .iter()
            .position(|t| t.is_none())
            .ok_or_else(|| GameError::new("No empty token slots"))?;

        let mut token = self.corporations[corp_idx].tokens[token_idx].clone();
        token.used = true;
        token.city_hex_id = resolved_hex_id.clone();
        city.tokens[slot_idx] = Some(token);

        // Update corp token tracking
        self.corporations[corp_idx].tokens[token_idx].used = true;
        self.corporations[corp_idx].tokens[token_idx].city_hex_id = resolved_hex_id.clone();

        // Pay
        self.corporations[corp_idx].cash -= token_cost;
        self.bank.cash += token_cost;

        // Clear graph cache (new token changes connectivity)
        self.clear_graph_cache();

        new_state.num_placed_token += 1;

        // Auto-pass Token step (1 token per turn in 1830)
        // Advance to RunRoutes regardless of which step we were in
        if new_state.num_placed_token >= 1 {
            new_state.step = OperatingStep::RunRoutes;
        }

        self.round = crate::rounds::Round::Operating(new_state);
        self.update_round_state();
        Ok(())
    }

    fn or_process_run_routes(
        &mut self,
        state: &OperatingState,
        _entity_id: &str,
        routes: &[RouteData],
        extra_revenue: i32,
    ) -> Result<(), GameError> {
        let mut new_state = state.clone();

        // Accept routes (route validation is Phase 4)
        let total_revenue: i32 = routes.iter().map(|r| r.revenue).sum::<i32>() + extra_revenue;

        new_state.routes = routes.to_vec();
        new_state.revenue = total_revenue;
        new_state.step = OperatingStep::Dividend;

        self.round = crate::rounds::Round::Operating(new_state);
        self.update_round_state();
        Ok(())
    }

    fn or_process_dividend(
        &mut self,
        state: &OperatingState,
        _entity_id: &str,
        kind: &DividendKind,
    ) -> Result<(), GameError> {
        let mut new_state = state.clone();

        if new_state.step != OperatingStep::Dividend {
            return Err(GameError::new("Not in Dividend step"));
        }

        let corp_sym = new_state
            .current_corp_sym()
            .ok_or_else(|| GameError::new("No current corp"))?
            .to_string();
        let corp_idx = self.corp_idx[&corp_sym];
        let revenue = new_state.revenue;

        match kind {
            DividendKind::Payout => {
                self.distribute_revenue(corp_idx, revenue)?;
                // Move share price right
                if let Some(sp) = self.corporations[corp_idx].share_price.clone() {
                    let (new_row, new_col) = self.stock_market.move_right(sp.row, sp.column);
                    if let Some(new_sp) = self.stock_market.share_price_at(new_row, new_col) {
                        self.corporations[corp_idx].share_price = Some(new_sp);
                        self.update_market_cell(&corp_sym, sp.row, sp.column, new_row, new_col);
                    }
                }
            }
            DividendKind::Withhold => {
                // Corp keeps all revenue
                self.corporations[corp_idx].cash += revenue;
                self.bank.cash -= revenue;
                // Move share price left
                if let Some(sp) = self.corporations[corp_idx].share_price.clone() {
                    let (new_row, new_col) = self.stock_market.move_left(sp.row, sp.column);
                    if let Some(new_sp) = self.stock_market.share_price_at(new_row, new_col) {
                        self.corporations[corp_idx].share_price = Some(new_sp);
                        self.update_market_cell(&corp_sym, sp.row, sp.column, new_row, new_col);
                    }
                }
            }
        }

        new_state.step = OperatingStep::BuyTrain;

        self.round = crate::rounds::Round::Operating(new_state);
        self.update_round_state();
        Ok(())
    }

    /// Distribute revenue proportionally to shareholders.
    /// In 1830 (full capitalization): payout gives per_share to each player holding shares.
    /// IPO shares generate NO revenue. Market pool shares generate NO revenue to the corp.
    /// The corporation itself receives 0 from payout.
    fn distribute_revenue(&mut self, corp_idx: usize, revenue: i32) -> Result<(), GameError> {
        if revenue <= 0 {
            return Ok(());
        }

        let corp = &self.corporations[corp_idx];
        let total_shares = 10i32; // 1830: all corps have 10 shares (president = 2 shares)
        let per_share = revenue / total_shares;

        // Pay each shareholder based on their number of shares
        for player in &mut self.players {
            let eid = EntityId::player(player.id);
            let percent = corp.percent_owned_by(&eid);
            if percent > 0 {
                // percent / 10 = number of shares (10% = 1 share, 20% = 2 shares)
                let num_shares = percent as i32 / 10;
                let payout = num_shares * per_share;
                player.cash += payout;
                self.bank.cash -= payout;
            }
        }

        // In 1830, market pool shares generate revenue paid to the corporation.
        // IPO shares generate no revenue.
        let market_eid = EntityId::market();
        let market_pct = corp.percent_owned_by(&market_eid);
        if market_pct > 0 {
            let market_shares = market_pct as i32 / 10;
            let corp_payout = market_shares * per_share;
            self.corporations[corp_idx].cash += corp_payout;
            self.bank.cash -= corp_payout;
        }

        Ok(())
    }

    fn or_process_buy_train(
        &mut self,
        state: &OperatingState,
        _entity_id: &str,
        train_name: &str,
        price: i32,
        from: &str,
        exchange: Option<&str>,
    ) -> Result<(), GameError> {
        let mut new_state = state.clone();

        if new_state.step != OperatingStep::BuyTrain {
            return Err(GameError::new("Not in BuyTrain step"));
        }

        let corp_sym = new_state
            .current_corp_sym()
            .ok_or_else(|| GameError::new("No current corp"))?
            .to_string();
        let corp_idx = self.corp_idx[&corp_sym];

        // Determine source: "depot", explicit corp sym, or auto-detect.
        // When "from" is unspecified (defaults to "depot"), use these heuristics:
        // 1. If the price differs from the depot price for this train type → inter-corp
        // 2. If the depot has this train type → depot
        // 3. Otherwise → search corporations
        let base_name = train_name.split('-').next().unwrap_or(train_name);
        let actual_from = if from == "depot" {
            let depot_price = self
                .depot
                .trains
                .iter()
                .find(|t| t.name == base_name)
                .map(|t| t.price);

            // Find the specific train instance by its full ID.
            // Check: (1) inter-corp by ID, (2) discarded pile by ID,
            // (3) depot by price, (4) inter-corp by base name, (5) fallback depot.
            let inter_corp = self
                .corporations
                .iter()
                .find(|c| c.sym != corp_sym && c.trains.iter().any(|t| t.id == train_name));

            let in_discard = self.depot.discarded.iter().any(|t| t.id == train_name);

            if let Some(seller) = inter_corp {
                seller.sym.clone()
            } else if in_discard {
                "depot".to_string() // bought from discard (bank pool)
            } else if let Some(dp) = depot_price {
                if price == dp {
                    "depot".to_string()
                } else {
                    self.corporations
                        .iter()
                        .find(|c| c.sym != corp_sym && c.trains.iter().any(|t| t.name == base_name))
                        .map(|c| c.sym.clone())
                        .unwrap_or_else(|| "depot".to_string())
                }
            } else {
                self.corporations
                    .iter()
                    .find(|c| c.sym != corp_sym && c.trains.iter().any(|t| t.name == base_name))
                    .map(|c| c.sym.clone())
                    .unwrap_or_else(|| "depot".to_string())
            }
        } else {
            from.to_string()
        };

        if actual_from == "depot" {
            // Check upcoming trains first, then discarded (bank pool).
            // Discarded trains are available at face value.
            let from_discarded = !self.depot.trains.iter().any(|t| t.name == base_name)
                && self.depot.discarded.iter().any(|t| t.name == base_name);

            let train_idx = if from_discarded {
                self.depot
                    .discarded
                    .iter()
                    .position(|t| t.name == base_name)
                    .ok_or_else(|| GameError::new(format!("Train {} not in depot or discard", train_name)))?
            } else {
                self.depot
                    .trains
                    .iter()
                    .position(|t| t.name == base_name)
                    .ok_or_else(|| GameError::new(format!("Train {} not in depot", train_name)))?
            };

            // If exchanging a train (e.g., 4→D), remove the old train first
            // and use the discounted price from the action.
            if let Some(exchange_name) = exchange {
                let exchange_base = exchange_name.split('-').next().unwrap_or(exchange_name);
                let ex_idx = self.corporations[corp_idx]
                    .trains
                    .iter()
                    .position(|t| t.name == exchange_base)
                    .ok_or_else(|| {
                        GameError::new(format!(
                            "Exchange train {} not owned by {}",
                            exchange_name, corp_sym
                        ))
                    })?;
                let old_train = self.corporations[corp_idx].trains.remove(ex_idx);
                self.depot.discarded.push(old_train);
            }

            // When exchanging, the action price is the discounted price.
            // Otherwise, must pay full depot/discard price.
            let train_source = if from_discarded {
                &self.depot.discarded
            } else {
                &self.depot.trains
            };
            let actual_price = if exchange.is_some() {
                price
            } else {
                train_source[train_idx].price
            };

            // Check if corp can afford; if not, president contributes
            let corp_pays = self.corporations[corp_idx].cash.min(actual_price);

            if corp_pays < actual_price {
                let president_pays = actual_price - corp_pays;
                if let Some(pres_id) = self.corporations[corp_idx].president_id() {
                    let pres_idx = self.player_index(pres_id).unwrap();
                    if self.players[pres_idx].cash < president_pays {
                        // President can't afford — bankruptcy
                        return Err(GameError::new(format!(
                            "President of {} cannot afford forced train buy",
                            corp_sym
                        )));
                    }
                    self.players[pres_idx].cash -= president_pays;
                }
            }

            self.corporations[corp_idx].cash -= corp_pays;
            self.bank.cash += actual_price;

            // Move train to corporation (marked as operated — can't run this turn)
            let mut train = if from_discarded {
                self.depot.discarded.remove(train_idx)
            } else {
                self.depot.trains.remove(train_idx)
            };
            train.owner = EntityId::corporation(&corp_sym);
            train.operated = true;
            self.corporations[corp_idx].trains.push(train);

            // Check phase advance
            self.check_phase_advance(train_name);
        } else {
            // Buy from another corporation
            let from_corp_idx = *self
                .corp_idx
                .get(actual_from.as_str())
                .ok_or_else(|| GameError::new(format!("Unknown corporation: {}", actual_from)))?;

            // Find train by full ID first, fall back to base name
            let train_idx = self.corporations[from_corp_idx]
                .trains
                .iter()
                .position(|t| t.id == train_name)
                .or_else(|| {
                    self.corporations[from_corp_idx]
                        .trains
                        .iter()
                        .position(|t| t.name == base_name)
                })
                .ok_or_else(|| {
                    GameError::new(format!("Train {} not owned by {}", train_name, actual_from))
                })?;

            // Transfer money — president contributes if corp can't afford (emergency buy)
            let corp_pays = self.corporations[corp_idx].cash.min(price);
            if corp_pays < price {
                let president_pays = price - corp_pays;
                if let Some(pres_id) = self.corporations[corp_idx].president_id() {
                    let pres_idx = self.player_index(pres_id).unwrap();
                    if self.players[pres_idx].cash < president_pays {
                        return Err(GameError::new(format!(
                            "Cannot afford train (corp has {}, president has {})",
                            self.corporations[corp_idx].cash, self.players[pres_idx].cash
                        )));
                    }
                    self.players[pres_idx].cash -= president_pays;
                } else {
                    return Err(GameError::new("Cannot afford train and no president"));
                }
            }
            self.corporations[corp_idx].cash -= corp_pays;
            self.corporations[from_corp_idx].cash += price;

            // Transfer train (marked as operated — can't run this turn)
            let mut train = self.corporations[from_corp_idx].trains.remove(train_idx);
            train.owner = EntityId::corporation(&corp_sym);
            train.operated = true;
            self.corporations[corp_idx].trains.push(train);
        }

        // Special rule: B&O private company closes when B&O buys its first train
        if corp_sym == "B&O" && self.corporations[corp_idx].trains.len() == 1 {
            if let Some(&bo_idx) = self.company_idx.get("BO") {
                if !self.companies[bo_idx].closed {
                    self.companies[bo_idx].closed = true;
                }
            }
        }

        // Preserve crowded_corps set by check_phase_advance (which modifies
        // self.round directly, but new_state was cloned before the phase change).
        // If any corps are crowded, jump to DiscardTrain step.
        if let crate::rounds::Round::Operating(ref current) = self.round {
            if !current.crowded_corps.is_empty() {
                new_state.crowded_corps = current.crowded_corps.clone();
                new_state.step = OperatingStep::DiscardTrain;
            }
        }

        self.round = crate::rounds::Round::Operating(new_state);
        self.update_round_state();
        Ok(())
    }

    /// Emergency sell: president sells shares during BuyTrain step to fund
    /// a forced train purchase. Uses the same share transfer logic as the
    /// stock round sell, but without stock round state management.
    fn or_emergency_sell(
        &mut self,
        _state: &OperatingState,
        entity_id: &str,
        corporation_sym: &str,
        percent: u8,
    ) -> Result<(), GameError> {
        let player_id: u32 = entity_id
            .parse()
            .map_err(|_| GameError::new(format!("Invalid player id: {}", entity_id)))?;

        let corp_idx = *self
            .corp_idx
            .get(corporation_sym)
            .ok_or_else(|| GameError::new(format!("Unknown corporation: {}", corporation_sym)))?;

        let share_price = self.corporations[corp_idx]
            .share_price
            .as_ref()
            .ok_or_else(|| GameError::new(format!("{} has not been parred", corporation_sym)))?
            .clone();

        let player_eid = crate::entities::EntityId::player(player_id);
        let market_eid = crate::entities::EntityId::market();
        let player_idx = self.player_index(player_id).unwrap();

        // Transfer shares to market
        let mut transferred_pct = 0u8;
        for share in &mut self.corporations[corp_idx].shares {
            if transferred_pct >= percent {
                break;
            }
            if share.owner == player_eid && !share.president {
                share.owner = market_eid.clone();
                transferred_pct += share.percent;
            }
        }

        // Player receives money
        let revenue = (percent as i32 * share_price.price) / 10;
        self.players[player_idx].cash += revenue;
        self.bank.cash -= revenue;

        // Share price drops: move DOWN once per 10% share sold
        let num_shares = percent as u32 / 10;
        let (mut row, mut col) = (share_price.row, share_price.column);
        for _ in 0..num_shares {
            let (nr, nc) = self.stock_market.move_down(row, col);
            row = nr;
            col = nc;
        }
        if let Some(new_sp) = self.stock_market.share_price_at(row, col) {
            self.corporations[corp_idx].share_price = Some(new_sp);
            self.update_market_cell(
                corporation_sym,
                share_price.row,
                share_price.column,
                row,
                col,
            );
        }

        // Check president change
        self.check_president_change(corp_idx);

        self.update_round_state();
        Ok(())
    }

    fn or_process_discard_train(
        &mut self,
        state: &OperatingState,
        entity_id: &str,
        train_name: &str,
    ) -> Result<(), GameError> {
        let mut new_state = state.clone();

        // DiscardTrain can target any corp (not just the current operating one),
        // e.g., when a phase change forces multiple corps to discard.
        let corp_sym = entity_id.to_string();
        let corp_idx = *self
            .corp_idx
            .get(&corp_sym)
            .ok_or_else(|| GameError::new(format!("Unknown corporation: {}", corp_sym)))?;

        let base_name = train_name.split('-').next().unwrap_or(train_name);
        let train_idx = self.corporations[corp_idx]
            .trains
            .iter()
            .position(|t| t.name == base_name)
            .ok_or_else(|| {
                GameError::new(format!("Train {} not owned by {}", train_name, corp_sym))
            })?;

        let train = self.corporations[corp_idx].trains.remove(train_idx);
        self.depot.discarded.push(train);

        // Remove this corp from crowded_corps if it's now within the limit
        let train_limit = self.phase.train_limit as usize;
        if self.corporations[corp_idx].trains.len() <= train_limit {
            new_state.crowded_corps.retain(|s| s != &corp_sym);
        }

        self.round = crate::rounds::Round::Operating(new_state);
        self.update_round_state();
        Ok(())
    }

    fn or_process_buy_company(
        &mut self,
        state: &OperatingState,
        _entity_id: &str,
        company_sym: &str,
        price: i32,
    ) -> Result<(), GameError> {
        let new_state = state.clone();

        let corp_sym = new_state
            .current_corp_sym()
            .ok_or_else(|| GameError::new("No current corp"))?
            .to_string();
        let corp_idx = self.corp_idx[&corp_sym];

        let company_idx = *self
            .company_idx
            .get(company_sym)
            .ok_or_else(|| GameError::new(format!("Unknown company: {}", company_sym)))?;

        // Validate: company must be owned by the corporation's president
        let president_id = self.corporations[corp_idx]
            .president_id()
            .ok_or_else(|| GameError::new("Corporation has no president"))?;

        let company_owner_id = self.companies[company_idx]
            .owner
            .player_id()
            .ok_or_else(|| GameError::new("Company not owned by a player"))?;

        if company_owner_id != president_id {
            return Err(GameError::new(
                "Company must be owned by the corporation's president",
            ));
        }

        // Validate price: between 1 and 2x face value
        let face_value = self.companies[company_idx].value;
        if price < 1 || price > face_value * 2 {
            return Err(GameError::new(format!(
                "Price {} must be between 1 and {} (2x face value)",
                price,
                face_value * 2
            )));
        }

        if self.corporations[corp_idx].cash < price {
            return Err(GameError::new("Corporation cannot afford company"));
        }

        // Transfer
        self.corporations[corp_idx].cash -= price;
        let pres_idx = self.player_index(president_id).unwrap();
        self.players[pres_idx].cash += price;
        self.companies[company_idx].owner = EntityId::corporation(&corp_sym);

        self.round = crate::rounds::Round::Operating(new_state);
        self.update_round_state();
        Ok(())
    }

    /// Start a corporation's operating turn.
    /// Places home token (if needed) and advances past non-blocking steps.
    /// If the corp has nothing to do, advances to the next corp.
    /// Set up a single corp's operating turn: reset train flags, clear graph
    /// cache, place home token if needed. Does NOT call skip_steps or advance.
    fn setup_corp_turn(&mut self) {
        let sym = match &self.round {
            crate::rounds::Round::Operating(s) => {
                s.current_corp_sym().map(|x| x.to_string())
            }
            _ => return,
        };
        let sym = match sym {
            Some(s) => s,
            None => return,
        };
        let ci = match self.corp_idx.get(sym.as_str()) {
            Some(&i) => i,
            None => return,
        };

        // Reset train operated flags
        for train in &mut self.corporations[ci].trains {
            train.operated = false;
        }

        // Clear graph cache (tile/token state may have changed since last corp)
        self.clear_graph_cache();

        // Place home token if the corp hasn't placed one yet
        if !self.corporations[ci].tokens.is_empty()
            && !self.corporations[ci].tokens[0].used
        {
            self.place_home_token(ci);
        }
    }

    /// Start the operating round: iterate through corps in operating order.
    /// For each corp, set up its turn and skip non-blocking steps. If a corp
    /// has nothing to do (all steps skip to Done), advance to the next corp.
    /// Stops when a corp has a blocking step (waiting for player action) or
    /// all corps have been processed (round finished).
    pub(crate) fn start_operating(&mut self) {
        // If no corps to operate, mark the round as finished immediately
        if let crate::rounds::Round::Operating(ref s) = self.round {
            if s.operating_order.is_empty() {
                if let crate::rounds::Round::Operating(ref mut s) = self.round {
                    s.finished = true;
                }
                return;
            }
        }

        let max_corps = match &self.round {
            crate::rounds::Round::Operating(s) => s.operating_order.len(),
            _ => return,
        };

        for _ in 0..max_corps {
            self.setup_corp_turn();
            self.skip_steps();

            let (is_done, is_finished) = match &self.round {
                crate::rounds::Round::Operating(s) => {
                    (s.step == OperatingStep::Done, s.finished)
                }
                _ => return,
            };

            if is_finished || !is_done {
                // Either round is over or this corp has a blocking step
                break;
            }

            // This corp had nothing to do — advance to next
            if let crate::rounds::Round::Operating(ref mut s) = self.round {
                s.advance_to_next_corp();
            }

            if matches!(&self.round, crate::rounds::Round::Operating(s) if s.finished) {
                break;
            }
        }

        self.update_round_state();
    }

    /// Advance past non-blocking steps in the current corp's operating turn.
    /// Stops at the first blocking step that has possible actions.
    ///
    /// Skip conditions:
    /// - Track: always blocking (requires pass)
    /// - Token: skip if no available tokens or can't afford
    /// - Route: skip if corp has no trains
    /// - Dividend: skip if revenue == 0 (auto-withhold, move price left)
    /// - DiscardTrain: skip if within train limit
    /// - BuyTrain: always blocking
    /// - BuyCompany(blocking): always blocking
    pub(crate) fn skip_steps(&mut self) {
        for _iteration in 0..20 {
            let (step, corp_sym) = match &self.round {
                crate::rounds::Round::Operating(s) => {
                    if s.finished {
                        return;
                    }
                    match s.current_corp_sym() {
                        Some(sym) => (s.step.clone(), sym.to_string()),
                        None => return,
                    }
                }
                _ => return,
            };

            let corp_idx = match self.corp_idx.get(&corp_sym) {
                Some(&idx) => idx,
                None => return,
            };

            let should_skip = match step {
                OperatingStep::LayTile => {
                    // Always blocking — requires explicit lay_tile or pass
                    false
                }
                OperatingStep::PlaceToken => {
                    // Skip if the corp can't place a token anywhere.
                    !self.can_place_token(&corp_sym)
                }
                OperatingStep::RunRoutes => {
                    let has_runnable_train = self.corporations[corp_idx]
                        .trains
                        .iter()
                        .any(|t| !t.operated);
                    if !has_runnable_train {
                        true
                    } else {
                        !self.can_run_route(&corp_sym)
                    }
                }
                OperatingStep::Dividend => {
                    // Skip if revenue == 0: auto-withhold (move price left)
                    let revenue = match &self.round {
                        crate::rounds::Round::Operating(s) => s.revenue,
                        _ => 0,
                    };
                    if revenue == 0 {
                        if let Some(sp) = self.corporations[corp_idx].share_price.clone() {
                            let (nr, nc) = self.stock_market.move_left(sp.row, sp.column);
                            if let Some(new_sp) = self.stock_market.share_price_at(nr, nc) {
                                self.corporations[corp_idx].share_price = Some(new_sp);
                                self.update_market_cell(&corp_sym, sp.row, sp.column, nr, nc);
                            }
                        }
                        true
                    } else {
                        false
                    }
                }
                OperatingStep::DiscardTrain => {
                    // Block if ANY corp is over the train limit (crowded_corps)
                    // or if the current corp is over the limit.
                    let has_crowded = match &self.round {
                        crate::rounds::Round::Operating(s) => !s.crowded_corps.is_empty(),
                        _ => false,
                    };
                    if has_crowded {
                        false // blocking — a corp needs to discard
                    } else {
                        let train_limit = self.phase.train_limit as usize;
                        self.corporations[corp_idx].trains.len() <= train_limit
                    }
                }
                OperatingStep::BuyCompany => {
                    // BuyCompany during OR is only available from Phase 3+.
                    // Any player-owned company can be sold to the corp.
                    let phase_num: u8 = self.phase.name.parse().unwrap_or(0);
                    if phase_num < 3 {
                        true
                    } else {
                        let corp_cash = self.corporations[corp_idx].cash;
                        !self.companies.iter().any(|c| {
                            !c.closed && !c.no_buy && c.owner.is_player() && corp_cash >= c.value / 2
                        })
                    }
                }
                OperatingStep::BuyTrain => {
                    // A corp must buy a train only if it has no trains AND has a
                    // legal revenue route. No legal route = no obligation to own a
                    // train, so BuyTrain is optional (and skippable if unaffordable).
                    let corp_cash = self.corporations[corp_idx].cash;
                    let has_trains = !self.corporations[corp_idx].trains.is_empty();

                    let must_buy = !has_trains && self.can_run_route(&corp_sym);

                    if must_buy {
                        false // blocking — forced buy, president must sell shares
                    } else {
                        // Can buy from depot (upcoming or discarded bank pool)?
                        let can_buy_from_depot = self
                            .depot
                            .trains
                            .first()
                            .map(|t| corp_cash >= t.price)
                            .unwrap_or(false);
                        let can_buy_from_discard = self
                            .depot
                            .discarded
                            .iter()
                            .any(|t| corp_cash >= t.price);

                        // Can exchange for discounted D?
                        // D-train exchanges are only available when D-trains are
                        // purchasable (phase D). The depot.trains list includes
                        // future trains; only the first of each type is available.
                        // D-trains become available when phase "D" or "6" is reached
                        // (the "D" train's "on" field is "D", available after first
                        // D purchase, but also available once "6" trains are out).
                        let d_available = self.phase.name == "D"
                            || self.depot.trains.first().map_or(false, |t| t.name == "D");
                        let can_exchange = if d_available {
                            if let Some(d_train) =
                                self.depot.trains.iter().find(|t| t.name == "D")
                            {
                                let discount = 300;
                                let exchange_price = d_train.price - discount;
                                corp_cash >= exchange_price
                                    && self.corporations[corp_idx]
                                        .trains
                                        .iter()
                                        .any(|t| {
                                            t.name == "4" || t.name == "5" || t.name == "6"
                                        })
                            } else {
                                false
                            }
                        } else {
                            false
                        };

                        // Can buy inter-corp (same president)?
                        let pres_id = self.corporations[corp_idx].president_id();
                        let can_buy_inter_corp = corp_cash > 0
                            && pres_id.is_some_and(|pid| {
                                self.corporations.iter().any(|other| {
                                    other.sym != corp_sym
                                        && other.president_id() == Some(pid)
                                        && !other.trains.is_empty()
                                })
                            });

                        let has_room = self.corporations[corp_idx].trains.len()
                            < self.phase.train_limit as usize;

                        !((has_room && (can_buy_from_depot || can_buy_from_discard || can_buy_inter_corp)) || can_exchange)
                    }
                }
                // Done: always blocking
                _ => false,
            };

            if !should_skip {
                return;
            }

            // Advance to next step
            if let crate::rounds::Round::Operating(ref mut s) = self.round {
                s.step = s.step.next();
            }
        }
    }

    /// Check if a corporation can place a token anywhere on the board.
    /// Requires: at least one unplaced token, enough cash to pay for it,
    /// and at least one reachable city with an open token slot.
    fn or_process_pass(
        &mut self,
        state: &OperatingState,
        _entity_id: &str,
    ) -> Result<(), GameError> {
        let mut new_state = state.clone();

        if new_state.step == OperatingStep::BuyCompany {
            // Pass from final BuyCompany ends the corp's turn
            new_state.advance_to_next_corp();
            self.round = crate::rounds::Round::Operating(new_state);
            self.update_round_state();
            if !matches!(&self.round, crate::rounds::Round::Operating(s) if s.finished) {
                self.start_operating();
            }
        } else {
            // Pass from any blocking step: advance to next step
            new_state.step = new_state.step.next();
            self.round = crate::rounds::Round::Operating(new_state);
            self.update_round_state();
            // skip_steps is called by process_operating_action after this returns
        }

        Ok(())
    }
}
