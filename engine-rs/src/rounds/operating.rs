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
        // discard before anything else happens. Accept discard_train from any
        // corp in the crowded list (Python's DiscardTrain step accepts the
        // action from any entity in ``crowded_corps``, not just the first —
        // round.py:2686). Reject all non-discard actions while crowded.
        let crowded_list: Vec<String> = match &self.round {
            crate::rounds::Round::Operating(s) if !s.crowded_corps.is_empty() => {
                s.crowded_corps.clone()
            }
            _ => Vec::new(),
        };
        if !crowded_list.is_empty() {
            if let Action::DiscardTrain {
                entity_id,
                train_name,
            } = action
            {
                if !crowded_list.iter().any(|c| c == entity_id) {
                    return Err(GameError::new(format!(
                        "Expected discard_train from one of {:?} (crowded), got {}",
                        crowded_list, entity_id
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
            } else if let Action::BuyCompany {
                entity_id,
                company_sym,
                price,
            } = action
            {
                // The blocking DiscardTrain step sits AFTER the non-blocking
                // BuyCompany step in the 1830 OR step list (g1830.py:612-626).
                // Python's process_action loop (round.py:5344-5369) lets that
                // earlier non-blocking step process a BuyCompanyAction and
                // RETURN before reaching the blocking DiscardTrain, so Python
                // ACCEPTS BuyCompany while a corp is crowded. Mirror that:
                // dispatch through the normal buy-company handler. (MH BuyShares
                // and CS LayTile while crowded are already handled earlier in
                // process_action_internal via try_process_company_exchange /
                // try_process_company_ability — those paths short-circuit BEFORE
                // this gate, and their skip_steps stays at DiscardTrain because a
                // corp is still crowded. Only BuyCompany reaches here.)
                self.or_process_buy_company(&state, entity_id, company_sym, *price)?;
                // skip_steps stays at the blocking DiscardTrain while any corp is
                // crowded (operating.rs DiscardTrain branch returns false), so the
                // game correctly remains at DiscardTrain after the buy. Do NOT run
                // the discard-specific "unpass BuyTrain / advance" logic.
                self.skip_steps();
                return Ok(());
            } else {
                return Err(GameError::new(format!(
                    "{:?} must discard a train (over train limit)",
                    crowded_list
                )));
            }
        }

        // Handle pending token re-placement (displaced by OO tile upgrade).
        // The operating corp acts to place the displaced corp's token.
        // Action entity_id is the operating corp, but the token placed belongs
        // to the displaced corp. This is an extra action (doesn't consume the
        // operating corp's regular token step).
        let pending_token = match &self.round {
            crate::rounds::Round::Operating(s) if !s.pending_tokens.is_empty() => {
                Some(s.pending_tokens[0].clone())
            }
            _ => None,
        };
        if let Some((ref pending_corp, pending_token_idx, ref pending_hex)) = pending_token {
            if let Action::PlaceToken {
                hex_id,
                city_index,
                ..
            } = action
            {
                // Resolve hex_id
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
                        .ok_or_else(|| GameError::new(format!("No hex with tile {}", tile_instance)))?
                } else {
                    hex_id.clone()
                };

                // Validate the hex matches the pending token's expected hex.
                // Python's HomeToken.process_place_token raises
                // "Cannot place token on X as the hex is not available" when
                // the chosen hex isn't in pending_token['hexes'].
                if !pending_hex.is_empty() && resolved_hex_id != *pending_hex {
                    return Err(GameError::new(format!(
                        "Cannot place token on {} as the hex is not available",
                        resolved_hex_id
                    )));
                }

                let hex_idx = *self
                    .hex_idx
                    .get(resolved_hex_id.as_str())
                    .ok_or_else(|| GameError::new(format!("Unknown hex: {}", resolved_hex_id)))?;
                let corp_idx = self.corp_idx[pending_corp.as_str()];

                // Reservation-aware slot selection mirroring Python's
                // `City.exchange_token` -> `get_slot(token.corporation)`
                // (graph.py:1025-1031). A plain "first None token" pick ignores
                // reservations and would write the wrong slot when another corp's
                // home reservation sits ahead of an empty slot. Compute before
                // taking the mutable city borrow.
                let slot_idx = self
                    .token_slot_for(&resolved_hex_id, *city_index as usize, pending_corp)
                    .ok_or_else(|| GameError::new("No empty token slots"))?;

                let city = self.hexes[hex_idx]
                    .tile
                    .cities
                    .get_mut(*city_index as usize)
                    .ok_or_else(|| GameError::new("Invalid city index"))?;

                let mut token = self.corporations[corp_idx].tokens[pending_token_idx].clone();
                token.used = true;
                token.city_hex_id = resolved_hex_id.clone();
                city.tokens[slot_idx] = Some(token);

                self.corporations[corp_idx].tokens[pending_token_idx].used = true;
                self.corporations[corp_idx].tokens[pending_token_idx].city_hex_id = resolved_hex_id;

                self.clear_graph_cache();

                // Remove from pending_tokens
                let was_home_token = if let crate::rounds::Round::Operating(ref s) = self.round {
                    // Home/pre-step token: no tiles laid yet in this corp's turn
                    s.num_laid_track == 0
                } else {
                    false
                };
                if let crate::rounds::Round::Operating(ref mut s) = self.round {
                    s.pending_tokens.remove(0);
                    // Home token: reset to LayTile for normal operating turn
                    if was_home_token && s.pending_tokens.is_empty() {
                        s.step = OperatingStep::LayTile;
                    }
                }
                self.update_round_state();

                // Run skip_steps and advance logic (same as post-action flow)
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
                return Ok(());
            }
        }

        // Home token placement on an upgraded OO hex: the corp must choose
        // which city. Accept place_token at the start of the turn (before Track).
        if let Action::PlaceToken {
            entity_id,
            hex_id,
            city_index,
        } = action
        {
            if let Some(corp_sym) = state.current_corp_sym() {
                if entity_id == corp_sym {
                    if let Some(&ci) = self.corp_idx.get(corp_sym) {
                        let needs_home = !self.corporations[ci].tokens.is_empty()
                            && !self.corporations[ci].tokens[0].used;
                        // Handle as HomeToken if the home token needs placement
                        // AND the target hex is the corp's home hex.
                        let corp_defs = crate::title::g1830::corporations();
                        let is_home_hex = corp_defs.iter()
                            .find(|cd| cd.sym == corp_sym)
                            .map_or(false, |cd| {
                                // Resolve hex_id for comparison
                                let target = if let Some(ti) = hex_id.strip_prefix("__tile:") {
                                    self.hexes.iter().find(|h| {
                                        let base = ti.split('-').next().unwrap_or(ti);
                                        h.tile.name == ti || h.tile.id == ti
                                            || h.tile.name == base || h.id == base
                                    }).map(|h| h.id.as_str())
                                } else {
                                    Some(hex_id.as_str())
                                };
                                target.map_or(false, |t| t == cd.home_hex)
                            });
                        let is_home_token = needs_home
                            && state.step == OperatingStep::PlaceToken
                            && is_home_hex;
                    if is_home_token {
                            // Resolve hex_id (may be "__tile:59-1" format)
                            let resolved = if let Some(ti) = hex_id.strip_prefix("__tile:") {
                                let base = ti.split('-').next().unwrap_or(ti);
                                self.hexes.iter().find(|h| {
                                    h.tile.name == ti || h.tile.id == ti
                                        || h.tile.name == base || h.id == base
                                }).map(|h| h.id.clone())
                            } else {
                                Some(hex_id.to_string())
                            };
                            if let Some(rhex) = resolved {
                                let city_idx = *city_index as usize;
                                if let Some(&hi) = self.hex_idx.get(&rhex) {
                                    if let Some(city) = self.hexes[hi].tile.cities.get_mut(city_idx) {
                                        for token_slot in &mut city.tokens {
                                            if token_slot.is_none() {
                                                let mut token =
                                                    self.corporations[ci].tokens[0].clone();
                                                token.used = true;
                                                token.city_hex_id = rhex.clone();
                                                *token_slot = Some(token);
                                                self.corporations[ci].tokens[0].used = true;
                                                self.corporations[ci].tokens[0].city_hex_id =
                                                    rhex;
                                                self.corporations[ci].home_token_ever_placed = true;
                                                break;
                                            }
                                        }
                                    }
                                }
                            }
                            self.clear_graph_cache();
                            if let crate::rounds::Round::Operating(ref mut s) = self.round {
                                if s.num_laid_track == 0 {
                                    // Start of turn — advance to LayTile
                                    s.step = OperatingStep::LayTile;
                                }
                                // Mid-turn: step stays at PlaceToken.
                                // Don't increment num_placed_token — home token
                                // doesn't consume the regular token step.
                            }
                            self.update_round_state();
                            // After mid-turn home token, run skip_steps to
                            // advance past PlaceToken if no regular token
                            // placement is possible.
                            if state.num_laid_track > 0 {
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
                        }
                    }
                }
            }
        }

        // Emergency sell: the operating corp's president sells shares at the
        // Buy Trains step (Python BuyTrain.actions gives the owner
        // [SellShares], round.py:798-799; processed by
        // EmergencyMoney.process_sell_shares, round.py:450-457). The dispatch
        // gate has already enforced the timing (pc == BuyTrain) and the
        // seller's identity (the corp's president) — a sell at any other OR
        // step is rejected there, exactly like Python's blocking-step guard.
        if let Action::SellShares {
            entity_id,
            corporation_sym,
            percent,
            share_indices,
            ..
        } = action
        {
            return self.or_emergency_sell(
                &state,
                entity_id,
                corporation_sym,
                *percent,
                share_indices,
            );
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
        entity_id: &str,
        hex_id: &str,
        tile_id: &str,
        rotation: u8,
    ) -> Result<(), GameError> {
        let mut new_state = state.clone();

        if new_state.step != OperatingStep::LayTile {
            return Err(GameError::new("Not in LayTile step"));
        }

        // Match Python's strict dispatch: action.entity must equal the
        // current operating corporation (round.py BaseStep). Catches
        // Ruby-engine quirks where an action records a non-operator entity
        // (master-mode actions, mis-recorded undo / redo sequences, etc.).
        let cur = new_state.current_corp_sym().ok_or_else(|| GameError::new("No current corp"))?;
        if entity_id != cur {
            return Err(GameError::new(format!(
                "lay_tile entity {} does not match current operator {}",
                entity_id, cur
            )));
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
        if !old_base.starts_with("preprinted") && !old_base.is_empty() && old_base != hex_id {
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

        // When upgrading an OO hex (preprinted with pathless 0-revenue cities)
        // to a tile with multiple cities AND paths, tokens must be re-placed
        // explicitly (Python's update_token / pending_tokens mechanism).
        // Tokens on tiles that already have paths are transferred automatically.
        let old_has_token = old_cities
            .iter()
            .any(|c| c.tokens.iter().any(|t| t.is_some()));
        let old_tile_has_no_paths = self.hexes[hex_idx].tile.paths.is_empty();
        let new_has_multiple_cities = new_tile.cities.len() > 1;
        let new_has_paths = !new_tile.paths.is_empty();
        let needs_token_choice =
            old_has_token && old_tile_has_no_paths && new_has_paths && new_has_multiple_cities;

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

        // Collect tokens displaced by OO upgrade (needs_token_choice) for
        // re-placement via pending_tokens. Must be done BEFORE resetting below.
        let mut displaced_tokens: Vec<(String, usize, String)> = Vec::new();
        if needs_token_choice {
            for old_city in &old_cities {
                for old_tok in old_city.tokens.iter().flatten() {
                    let corp_sym = &old_tok.corporation_id;
                    if let Some(&ci) = self.corp_idx.get(corp_sym.as_str()) {
                        // Find the token index in the corporation's token list
                        for (ti, ct) in self.corporations[ci].tokens.iter().enumerate() {
                            if ct.used && ct.city_hex_id == hex_id {
                                displaced_tokens.push((corp_sym.clone(), ti, hex_id.to_string()));
                                break;
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

        // Add displaced tokens to pending_tokens for re-placement
        if !displaced_tokens.is_empty() {
            new_state.pending_tokens.extend(displaced_tokens);
        }

        new_state.num_laid_track += 1;

        // Auto-pass Track step if no more tiles can be laid (1 per turn in 1830)
        if new_state.num_laid_track >= 1 {
            new_state.step = crate::steps::next_operating_pc(self.operating_step_descs(), &new_state.step);
        }

        self.round = crate::rounds::Round::Operating(new_state);
        self.update_round_state();
        Ok(())
    }

    fn or_process_place_token(
        &mut self,
        state: &OperatingState,
        entity_id: &str,
        hex_id: &str,
        city_index: u8,
    ) -> Result<(), GameError> {
        let mut new_state = state.clone();

        if new_state.step != OperatingStep::PlaceToken {
            return Err(GameError::new("Not in PlaceToken step"));
        }

        // Strict dispatch (see or_process_lay_tile). The operating corp
        // places its own token (or a displaced corp's token via OO upgrade —
        // either way action.entity is the operating corp per Python).
        let cur = new_state.current_corp_sym().ok_or_else(|| GameError::new("No current corp"))?;
        if entity_id != cur {
            return Err(GameError::new(format!(
                "place_token entity {} does not match current operator {}",
                entity_id, cur
            )));
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

        // Resolve the slot the SAME way Python's `City.get_slot` does (skipping
        // slots reserved for OTHER corps, honoring this corp's own reservation).
        // Computed before the mutable city borrow below so the borrows don't
        // overlap. Mirrors Python's `exchange_token` recomputing the slot from
        // the corporation rather than trusting the action's slot field.
        let slot_idx = self
            .token_slot_for(&resolved_hex_id, city_index as usize, &corp_sym)
            .ok_or_else(|| GameError::new("No empty token slots"))?;

        let city = self.hexes[hex_idx]
            .tile
            .cities
            .get_mut(city_index as usize)
            .ok_or_else(|| GameError::new("Invalid city index"))?;
        if city.tokens.get(slot_idx).map_or(true, |t| t.is_some()) {
            return Err(GameError::new("No empty token slots"));
        }

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
        entity_id: &str,
        routes: &[RouteData],
        extra_revenue: i32,
    ) -> Result<(), GameError> {
        let mut new_state = state.clone();

        // Strict dispatch — action.entity must match current operator.
        let cur = new_state.current_corp_sym().ok_or_else(|| GameError::new("No current corp"))?;
        if entity_id != cur {
            return Err(GameError::new(format!(
                "run_routes entity {} does not match current operator {}",
                entity_id, cur
            )));
        }

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
        entity_id: &str,
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

        // Strict dispatch — action.entity must match current operator. Same
        // logic as Python's process_dividend (which now uses
        // self.current_entity instead of action.entity).
        if entity_id != corp_sym {
            return Err(GameError::new(format!(
                "dividend entity {} does not match current operator {}",
                entity_id, corp_sym
            )));
        }
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
        entity_id: &str,
        train_name: &str,
        price: i32,
        from: &str,
        exchange: Option<&str>,
    ) -> Result<(), GameError> {
        let mut new_state = state.clone();

        if new_state.step != OperatingStep::BuyTrain {
            return Err(GameError::new("Not in BuyTrain step"));
        }

        // Strict dispatch — action.entity (the buying corp) must match the
        // current operator.
        let cur = new_state.current_corp_sym().ok_or_else(|| GameError::new("No current corp"))?;
        if entity_id != cur {
            return Err(GameError::new(format!(
                "buy_train entity {} does not match current operator {}",
                entity_id, cur
            )));
        }

        let corp_sym = new_state
            .current_corp_sym()
            .ok_or_else(|| GameError::new("No current corp"))?
            .to_string();
        let corp_idx = self.corp_idx[&corp_sym];

        // Validate that the action's EXACT train (matched by full ID) is
        // actually buyable in the current situation, mirroring Python's
        // `buy_train_action` (round.py:566-570):
        //
        //     if train not in (depot.available(entity) + buyable_trains(entity)):
        //         raise Exception("Not a buyable train")
        //
        // Membership is by object identity, so a same-NAMED train at a
        // different/own corp or a later upcoming copy does NOT satisfy the
        // check. Since `buyable_trains(entity)` is always a subset of
        // `depot.available(entity)` (it only further restricts the same
        // pools), the union reduces, membership-wise, to
        // `depot.available(entity)` =
        //     depot.depot_trains() + depot.other_trains(entity)
        // with no affordability filter (affordability is enforced separately
        // below and during enumeration). Build the legal exact-train-ID set
        // exactly:
        //
        //   depot_trains() (entities.py:739-753):
        //     [upcoming[0]]                      # head-of-queue, always visible
        //     + [t for t in upcoming if phase.available(t.available_on)]
        //     + discarded
        //   other_trains(entity) (entities.py:758-762):
        //     trains on OTHER corps that are buyable (always True in 1830),
        //     owner not in [corp, depot, None], and — since
        //     ALLOW_TRAIN_BUY_FROM_OTHER_PLAYERS == False — only where the
        //     seller corp's president == the buyer corp's president.
        //
        // For exchanges (exchange.is_some()) the variant/discount legality is
        // validated separately by enumeration/Python, so skip this membership
        // check (the discounted exchange train comes from the depot anyway).
        if exchange.is_none() {
            let mut legal_ids: std::collections::HashSet<String> =
                std::collections::HashSet::new();

            // depot_trains(): head-of-queue is always visible.
            if let Some(t) = self.depot.trains.first() {
                legal_ids.insert(t.id.clone());
            }
            // depot_trains(): upcoming trains whose available_on phase is
            // reached. `phase_available(None)` returns false (matching Python's
            // `phase.available(None) == False`), so non-D trains only become
            // visible via the head rule above.
            for t in &self.depot.trains {
                if self.phase_available(t.available_on.as_deref()) {
                    legal_ids.insert(t.id.clone());
                }
            }
            // depot_trains(): discarded pool, always visible.
            for t in &self.depot.discarded {
                legal_ids.insert(t.id.clone());
            }
            // other_trains(entity): other-corp trains, same president only
            // (ALLOW_TRAIN_BUY_FROM_OTHER_PLAYERS == False for 1830).
            let buyer_pres = self.corporations[corp_idx].president_id();
            for other in &self.corporations {
                if other.sym == corp_sym {
                    continue;
                }
                if buyer_pres.is_none() || other.president_id() != buyer_pres {
                    continue;
                }
                for t in &other.trains {
                    legal_ids.insert(t.id.clone());
                }
            }

            if !legal_ids.contains(train_name) {
                return Err(GameError::new("Not a buyable train"));
            }
        }

        // Determine source: "depot", explicit corp sym, or auto-detect.
        // When "from" is unspecified (defaults to "depot"), use these heuristics:
        // 1. If the price differs from the depot price for this train type → inter-corp
        // 2. If the depot has this train type → depot
        // 3. Otherwise → search corporations
        let base_name = train_name.split('-').next().unwrap_or(train_name);
        // When exchanging a train (e.g., 4→D), the source is always depot
        // even though the price differs from depot price (it's discounted).
        let actual_from = if exchange.is_some() {
            "depot".to_string()
        } else if from == "depot" {
            let depot_price = self
                .depot
                .trains
                .iter()
                .find(|t| t.name == base_name)
                .map(|t| t.price);

            // Source detection priority (mirrors Python's ``train.owner``
            // semantics — the train's actual owner determines the source,
            // regardless of action.price):
            // 1. Exact train ID in depot.trains → depot (definitive)
            // 2. Exact train ID in depot.discarded → depot (definitive)
            // 3. Exact train ID on another corp → inter-corp (definitive)
            // 4. Base name in depot.discarded → depot
            // 5. Base name in depot.trains → depot
            // 6. Base name on another corp → inter-corp
            let in_depot_by_id = self.depot.trains.iter().any(|t| t.id == train_name);
            let in_discard_by_id = self.depot.discarded.iter().any(|t| t.id == train_name);
            let inter_corp_by_id = self
                .corporations
                .iter()
                .find(|c| c.sym != corp_sym && c.trains.iter().any(|t| t.id == train_name));
            let in_depot_by_name = self.depot.trains.iter().any(|t| t.name == base_name);
            let in_discard_by_name = self.depot.discarded.iter().any(|t| t.name == base_name);

            if in_depot_by_id || in_discard_by_id {
                "depot".to_string()
            } else if let Some(seller) = inter_corp_by_id {
                seller.sym.clone()
            } else if in_discard_by_name || in_depot_by_name {
                "depot".to_string()
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
            // Prefer exact-id match (so the discarded train with id "5-0"
            // is removed when the action specifies "5-0", matching Python's
            // ``game.train_by_id(...)`` followed by ``remove_train`` which
            // deletes that specific train from whichever depot list holds it).
            // Fall back to name-based: only after checking ID, prefer
            // discarded (Python's ``min_depot_train`` considers discarded as
            // the cheaper alternative for a same-named train).
            let id_in_trains = self.depot.trains.iter().position(|t| t.id == train_name);
            let id_in_discard = self.depot.discarded.iter().position(|t| t.id == train_name);
            let from_discarded = if id_in_discard.is_some() {
                true
            } else if id_in_trains.is_some() {
                false
            } else {
                // No ID match — fall back to name. Prefer discarded if it
                // has this name and depot doesn't (the only remaining
                // legal source).
                !self.depot.trains.iter().any(|t| t.name == base_name)
                    && self.depot.discarded.iter().any(|t| t.name == base_name)
            };

            let train_idx = if from_discarded {
                id_in_discard
                    .or_else(|| self.depot.discarded.iter().position(|t| t.name == base_name))
                    .ok_or_else(|| {
                        GameError::new(format!("Train {} not in depot or discard", train_name))
                    })?
            } else {
                id_in_trains
                    .or_else(|| self.depot.trains.iter().position(|t| t.name == base_name))
                    .ok_or_else(|| GameError::new(format!("Train {} not in depot", train_name)))?
            };

            // Resolve the trade-in train (validation only — the removal is
            // deferred until every check below passes, so a failed buy
            // leaves state untouched).
            let ex_idx = if let Some(exchange_name) = exchange {
                let exchange_base = exchange_name.split('-').next().unwrap_or(exchange_name);
                // Match by full ID first, then fall back to base name
                Some(
                    self.corporations[corp_idx]
                        .trains
                        .iter()
                        .position(|t| t.id == exchange_name)
                        .or_else(|| {
                            self.corporations[corp_idx]
                                .trains
                                .iter()
                                .position(|t| t.name == exchange_base)
                        })
                        .ok_or_else(|| {
                            GameError::new(format!(
                                "Exchange train {} not owned by {}",
                                exchange_name, corp_sym
                            ))
                        })?,
                )
            } else {
                None
            };

            // Use the action's price — Python honors ``action.price`` for
            // all depot purchases, matching what the human game recorded.
            // Depot trains normally sell at face value, but the recorded
            // action's ``price`` is the source of truth.
            let actual_price = price;

            // Check if corp can afford; if not, president contributes.
            let corp_pays = self.corporations[corp_idx].cash.min(actual_price);
            let president_pays = actual_price - corp_pays;
            if president_pays > 0 {
                // Python forbids president contribution on an exchange buy
                // (round.py:579-581).
                if ex_idx.is_some() {
                    return Err(GameError::new("Cannot contribute funds when exchanging"));
                }
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
                } else {
                    return Err(GameError::new("Cannot afford train and no president"));
                }
            }

            // All checks passed — commit. The trade-in goes to the discard pile.
            if let Some(ex_idx) = ex_idx {
                let mut old_train = self.corporations[corp_idx].trains.remove(ex_idx);
                // Reset ownership when train returns to the bank pool — otherwise
                // ``train.owner`` still reports the previous corporation, which breaks
                // the cleaning pipeline's cross-player check (it sees the discarded
                // train as still owned by the old corp).
                old_train.owner = EntityId::none();
                self.depot.discarded.push(old_train);
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

        // Close ability (when: bought_train): the linked private company
        // closes when its corporation buys its first train (1830: BO closes
        // on B&O's first train).
        if self.corporations[corp_idx].trains.len() == 1 {
            if let Some(co_sym) = crate::abilities::close_on_bought_train(&corp_sym) {
                if let Some(&co_idx) = self.company_idx.get(co_sym) {
                    if !self.companies[co_idx].closed {
                        self.companies[co_idx].closed = true;
                    }
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

        // Also check if the buying corp is now over the train limit
        // (can happen when buying at the limit — Python allows buy+discard).
        let train_limit = self.phase.train_limit as usize;
        if self.corporations[corp_idx].trains.len() > train_limit {
            if !new_state.crowded_corps.contains(&corp_sym) {
                new_state.crowded_corps.push(corp_sym.clone());
            }
            new_state.step = OperatingStep::DiscardTrain;
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
        state: &OperatingState,
        entity_id: &str,
        corporation_sym: &str,
        percent: u8,
        share_indices: &[usize],
    ) -> Result<(), GameError> {
        let player_id: u32 = entity_id
            .parse()
            .map_err(|_| GameError::new(format!("Invalid player id: {}", entity_id)))?;

        // Validate-then-mutate: the Buy Trains step's `can_sell`
        // (Train.can_sell -> EmergencyMoney.can_sell, round.py:459-505,626-631)
        // before any state change — ownership of the named certs, the 50%
        // market cap, president-dump legality, the operating-corp
        // president-swap concern, and `selling_minimum_shares`.
        let op_corp = state
            .current_corp_sym()
            .ok_or_else(|| GameError::new("No current corp"))?;
        self.validate_sell_bundle(player_id, corporation_sym, percent, share_indices, Some(op_corp))?;

        // Perform the actual bundle sale (share transfer to market, price drop,
        // president change, partial-president return). Mirrors Python's
        // ``sell_shares_and_change_price``. The named share_indices keep the
        // sold certs aligned with Python's recorded share ids.
        let pre_sale_price =
            self.sell_share_bundle(player_id, corporation_sym, percent, share_indices)?;

        // Emergency sale dropped a (possibly future-operating) corp's share
        // price. Mirror Python's ``Operating.recalculate_order`` so the
        // not-yet-operated tail of operating_order is resorted by current
        // price (round.py:5667).
        self.recalculate_operating_order();

        self.update_round_state();
        // Record the per-share price of this emergency sale (captured pre-sale,
        // like Python's `bundle.price_per_share()` at action time) so the
        // subsequent emergency train buy's `spend_minmax` computes the correct
        // minimum. Set after update_round_state so it isn't clobbered; reset on
        // the next corp's turn via `advance_to_next_corp`.
        if let crate::rounds::Round::Operating(ref mut s) = self.round {
            s.last_share_sold_price = Some(pre_sale_price);
        }
        Ok(())
    }

    /// Which Python step's `can_sell` governs a SellShares action.
    pub(crate) fn validate_sell_bundle(
        &self,
        player_id: u32,
        corporation_sym: &str,
        percent: u8,
        share_indices: &[usize],
        operating_corp: Option<&str>,
    ) -> Result<(), GameError> {
        // Faithful port of Python's sell-side validation, run BEFORE any
        // mutation (validate-then-mutate):
        //   * Stock round (`operating_corp == None`):
        //     `BuySellParShares.sell_shares` -> `can_sell` (round.py:1839-1844,
        //     1601-1621): ownership, `check_sale_timing` (SELL_AFTER="first"),
        //     `can_sell_order` (always true for 1830's sell_buy_sell),
        //     `share_pool.fit_in_bank` (50% market cap), `bundle.can_dump`.
        //   * OR Buy Trains step (`operating_corp == Some(op)`):
        //     `Train.can_sell` -> `EmergencyMoney.can_sell` (round.py:626-631,
        //     459-505): ownership, `check_sale_timing` (always passes in an
        //     OR), `sellable_bundle` (can_dump + fit_in_bank +
        //     president-swap concern for the OPERATING corp), and
        //     `selling_minimum_shares` (EBUY_SELL_MORE_THAN_NEEDED == False).
        // Every rejection is Python's "Cannot sell shares of {corp}".
        let reject = || GameError::new(format!("Cannot sell shares of {}", corporation_sym));

        let ci = *self
            .corp_idx
            .get(corporation_sym)
            .ok_or_else(|| GameError::new(format!("Unknown corporation: {}", corporation_sym)))?;
        let corp = &self.corporations[ci];
        let share_price = corp
            .share_price
            .as_ref()
            .ok_or_else(|| GameError::new(format!("{} has not been parred", corporation_sym)))?
            .price;
        let player_eid = crate::entities::EntityId::player(player_id);

        let pres_pct: u8 = corp
            .shares
            .iter()
            .find(|s| s.president)
            .map(|s| s.percent)
            .unwrap_or(20);
        let player_total = corp.percent_owned_by(&player_eid);

        // -- resolve the bundle's certs ------------------------------------
        // Named certs: every index must exist and be owned by the seller
        // (Python: `share_by_id` + `ShareBundle` same-owner invariant +
        // `can_sell`'s `entity != bundle.owner`), and the action's percent
        // must be consistent with the named certs (equal, or the generated
        // partial-president reduction — `partial_bundles_for_presidents_share`,
        // base.py:1599-1603).
        let includes_president: bool;
        let bundle_cert_pcts: Vec<u8>;
        if !share_indices.is_empty() {
            let mut named_pcts: Vec<u8> = Vec::new();
            let mut named_pres = false;
            for &i in share_indices {
                let sh = corp
                    .shares
                    .get(i)
                    .ok_or_else(|| GameError::new(format!(
                        "Unknown share {}_{}",
                        corporation_sym, i
                    )))?;
                if sh.owner != player_eid {
                    return Err(reject());
                }
                named_pcts.push(sh.percent);
                named_pres = named_pres || sh.president;
            }
            let sum_pct: u8 = named_pcts.iter().sum();
            let percent_ok = percent == sum_pct
                || (named_pres
                    && percent < sum_pct
                    && (sum_pct - percent) % 10 == 0
                    && percent >= sum_pct - (pres_pct - 10));
            if !percent_ok {
                return Err(GameError::new(format!(
                    "SellShares percent {} does not match named shares totalling {}%",
                    percent, sum_pct
                )));
            }
            includes_president = named_pres;
            bundle_cert_pcts = named_pcts;
        } else {
            // No named certs (the native-decode path): the bundle is implied
            // by the percent — the seller's non-president certs by ascending
            // index, plus the president cert when the sale would leave the
            // seller below the president percent (the same selection the
            // mutation below performs). The seller must actually hold it.
            if percent > player_total {
                return Err(reject());
            }
            let has_pres = corp
                .shares
                .iter()
                .any(|s| s.president && s.owner == player_eid);
            includes_president = has_pres && player_total - percent < pres_pct;
            let non_pres_needed = if includes_president {
                percent.saturating_sub(pres_pct)
            } else {
                percent
            };
            let non_pres_held: u8 = corp
                .shares
                .iter()
                .filter(|s| s.owner == player_eid && !s.president)
                .map(|s| s.percent)
                .sum();
            if non_pres_held < non_pres_needed {
                return Err(reject());
            }
            let mut pcts: Vec<u8> = Vec::new();
            let mut taken = 0u8;
            for s in corp.shares.iter() {
                if taken >= non_pres_needed {
                    break;
                }
                if s.owner == player_eid && !s.president {
                    pcts.push(s.percent);
                    taken += s.percent;
                }
            }
            if includes_president {
                pcts.push(pres_pct);
            }
            bundle_cert_pcts = pcts;
        }

        // -- check_sale_timing (base.py:1511-1515, SELL_AFTER = "first") ----
        // `turn > 1 or round.operating`: stock-round sales are forbidden in
        // the first stock round; OR sales always pass the timing check.
        if operating_corp.is_none() && self.turn <= 1 {
            return Err(reject());
        }

        // -- fit_in_bank (entities.py:455-458): pool capped at 50% ----------
        let market_pct: u8 = corp
            .shares
            .iter()
            .filter(|s| s.owner.is_market())
            .map(|s| s.percent)
            .sum();
        if market_pct + percent > 50 {
            return Err(reject());
        }

        // -- can_dump (entities.py:141-146): dumping the presidency requires
        // another holder at or above the president percent ------------------
        if includes_president {
            let max_other = self
                .players
                .iter()
                .filter(|p| p.id != player_id)
                .map(|p| corp.percent_owned_by(&crate::entities::EntityId::player(p.id)))
                .max()
                .unwrap_or(0);
            if max_other < pres_pct {
                return Err(reject());
            }
        }

        // -- OR Buy Trains extras (EmergencyMoney, round.py:459-505) --------
        if let Some(op_corp) = operating_corp {
            // president_swap_concern (EBUY_PRES_SWAP == True): only the
            // OPERATING corp's shares are guarded against a president swap.
            if corporation_sym == op_corp
                && self.causes_president_swap(corporation_sym, player_id, percent)
            {
                return Err(reject());
            }
            // selling_minimum_shares (round.py:474-478,
            // EBUY_SELL_MORE_THAN_NEEDED == False): the next-smaller bundle
            // (this bundle minus its cheapest cert) must leave the buyer
            // short. needed_cash = depot.min_depot_price
            // (EBUY_DEPOT_TRAIN_MUST_BE_CHEAPEST), available_cash =
            // seller.cash + operating corp cash (round.py:664-668).
            let price_for_pct =
                |pct: u8| -> i32 { ((share_price as i64 * pct as i64 + 9) / 10) as i32 };
            let bundle_price = price_for_pct(percent);
            let min_share_price = bundle_cert_pcts
                .iter()
                .map(|&p| price_for_pct(p))
                .min()
                .unwrap_or(0);
            let seller_cash = self
                .players
                .iter()
                .find(|p| p.id == player_id)
                .map_or(0, |p| p.cash);
            let op_corp_cash = self
                .corp_idx
                .get(op_corp)
                .map_or(0, |&i| self.corporations[i].cash);
            let additional_cash_needed =
                self.min_depot_price_for_emr() - (seller_cash + op_corp_cash);
            if !(bundle_price - min_share_price < additional_cash_needed) {
                return Err(reject());
            }
        }

        Ok(())
    }

    /// Sell a `percent`% bundle of `corporation_sym` held by `player_id` into
    /// the market. Mirrors Python's ``sell_shares_and_change_price`` for 1830's
    /// ``SELL_MOVEMENT = "down_share"``: transfer the shares to the share pool,
    /// pay the player, drop the price one step per 10% sold, route the
    /// presidency if the president share was dumped, and return any leftover
    /// half-president slice to the seller. Returns the pre-sale per-share price
    /// (for ``last_share_sold_price`` bookkeeping).
    pub(crate) fn sell_share_bundle(
        &mut self,
        player_id: u32,
        corporation_sym: &str,
        percent: u8,
        share_indices: &[usize],
    ) -> Result<i32, GameError> {
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

        // Snapshot pre-action owners of every share BEFORE any owner mutation.
        // When this sale triggers a president change, the swap must pick the new
        // president's OLDEST pre-action shares (by acquisition order), matching
        // Python's ``shares_for_presidency_swap(president.shares_of(corp))``.
        // The stock-round sell path snapshots this too (stock.rs); the OR
        // emergency-sale path previously called ``check_president_change_with_prev``
        // without a snapshot, so its swap fell back to Vec-index order and
        // picked the wrong specific certs — diverging from Python.
        let pre_action_owners: Vec<crate::entities::EntityId> = self.corporations[corp_idx]
            .shares
            .iter()
            .map(|s| s.owner.clone())
            .collect();

        // Determine if the president share is being dumped — same condition
        // as the stock-round sell logic: player has the president and would
        // be left with < 20% after the sale.
        let player_total = self.corporations[corp_idx].percent_owned_by(&player_eid);
        let remaining_after = player_total.saturating_sub(percent);
        let includes_president = self.corporations[corp_idx]
            .shares
            .iter()
            .any(|s| s.president && s.owner == player_eid)
            && remaining_after < 20;

        // Transfer shares to market. When dumping the president, transfer the
        // non-president portion (= percent - president_face_value, i.e.
        // percent - 20 in 1830) first, then move the 20% president share to
        // market. ``check_president_change`` below will then route the
        // president to the new president and balance with two of their
        // normal shares moving to market.
        //
        // Like the stock-round sell, the action's named ``share_indices`` ARE
        // the exact certs to move (for every 1830 share ``corp.shares[i].index
        // == i``, set once at init and never reordered). Python's
        // ``SharePool.transfer_shares`` moves precisely the bundle's certs, so
        // we honor every named NON-president index on EVERY sell (the president
        // cert, if named, is routed separately below). This keeps Rust's
        // per-cert → owner mapping in lockstep with Python's recorded share ids
        // so a later sell that names a specific id resolves to the same owner in
        // both engines. We fall back to owner-based ascending-index selection
        // when no indices are supplied — which is NOT rare: the native decode
        // emits empty ``share_indices`` for EVERY SellShares (decode.rs), and
        // the bankruptcy liquidation constructs its own bundles. Either way
        // the correct PERCENT reaches the market.
        let pres_pct: u8 = self.corporations[corp_idx]
            .shares
            .iter()
            .find(|s| s.president)
            .map(|s| s.percent)
            .unwrap_or(20);
        let non_pres_to_sell = if includes_president {
            percent.saturating_sub(pres_pct)
        } else {
            percent
        };

        // Honor the EXACT certs named by the action (Python's transfer_shares
        // moves precisely the bundle's non-president certs; any over-move for a
        // partial president-dump is returned by the partial handling below).
        // The president cert, if named, is routed separately. Moving exactly the
        // named ids keeps Rust's per-cert -> owner map aligned with Python; the
        // president-swap snapshot fixes (acquired_seq ordering) keep the seller
        // owning these certs, so identity never drifts onto another owner's cert.
        let mut to_market: Vec<usize> = Vec::new();
        if !share_indices.is_empty() {
            for &i in share_indices {
                if i < self.corporations[corp_idx].shares.len()
                    && !self.corporations[corp_idx].shares[i].president
                {
                    to_market.push(i);
                }
            }
        } else {
            let mut transferred_pct = 0u8;
            for (i, share) in self.corporations[corp_idx].shares.iter().enumerate() {
                if transferred_pct >= non_pres_to_sell {
                    break;
                }
                if share.owner == player_eid && !share.president {
                    to_market.push(i);
                    transferred_pct += share.percent;
                }
            }
        }
        for i in to_market {
            self.corporations[corp_idx]
                .set_share_owner(i, market_eid.clone());
        }
        // The president cert is moved last (only when it is being dumped).
        if includes_president {
            if let Some(pres_idx) = self.corporations[corp_idx]
                .shares
                .iter()
                .position(|s| s.owner == player_eid && s.president)
            {
                self.corporations[corp_idx]
                    .set_share_owner(pres_idx, market_eid.clone());
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

        // Check president change — pass the seller as previous_president so
        // the clockwise-from-prev tiebreaker works (mirrors stock.rs), AND the
        // pre-action owner snapshot so the presidency swap picks the new
        // president's oldest pre-action shares (matches Python's
        // ``possible_reorder(president.shares_of(corp))`` insertion order).
        self.check_president_change_with_snapshot(corp_idx, Some(player_id), pre_action_owners);

        // Handle partial bundles: when selling a partial president bundle
        // (percent < president face value), the seller keeps the leftover
        // half-president as a normal share. Mirrors stock.rs:461-480.
        // Uses the corporation's `market_order` (insertion order) so that the
        // OLDEST market share is returned, matching Python.
        if includes_president {
            let actual_pct = self.corporations[corp_idx].percent_owned_by(&player_eid);
            let target_pct = remaining_after;
            if actual_pct < target_pct {
                let deficit = target_pct - actual_pct;
                let shares_to_return = deficit / 10;
                let mut returned = 0u8;
                while returned < shares_to_return {
                    let oldest = self.corporations[corp_idx].oldest_market_share_index();
                    match oldest {
                        Some(idx) => {
                            self.corporations[corp_idx]
                                .set_share_owner(idx, player_eid.clone());
                            returned += 1;
                        }
                        None => break,
                    }
                }
            }
        }

        Ok(share_price.price)
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

        // Match by full ID first (e.g., "3-4") so the right specific train is
        // discarded when the corp owns multiple of the same type. Fall back to
        // base name (e.g., "3") for backwards compat with actions that only
        // record the train type.
        let base_name = train_name.split('-').next().unwrap_or(train_name);
        let train_idx = self.corporations[corp_idx]
            .trains
            .iter()
            .position(|t| t.id == train_name)
            .or_else(|| {
                self.corporations[corp_idx]
                    .trains
                    .iter()
                    .position(|t| t.name == base_name)
            })
            .ok_or_else(|| {
                GameError::new(format!("Train {} not owned by {}", train_name, corp_sym))
            })?;

        let mut train = self.corporations[corp_idx].trains.remove(train_idx);
        // Reset ownership: discarded trains are owned by the bank pool, not the
        // corp that discarded them. See discussion at the exchange-discard path.
        train.owner = EntityId::none();
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
        entity_id: &str,
        company_sym: &str,
        price: i32,
    ) -> Result<(), GameError> {
        let new_state = state.clone();

        let corp_sym = new_state
            .current_corp_sym()
            .ok_or_else(|| GameError::new("No current corp"))?
            .to_string();
        let corp_idx = self.corp_idx[&corp_sym];

        // Strict dispatch — action.entity (the buying corp) must match
        // current operator.
        if entity_id != corp_sym {
            return Err(GameError::new(format!(
                "buy_company entity {} does not match current operator {}",
                entity_id, corp_sym
            )));
        }

        let company_idx = *self
            .company_idx
            .get(company_sym)
            .ok_or_else(|| GameError::new(format!("Unknown company: {}", company_sym)))?;

        // Validate ownership: Python's purchasable_companies (base.py:1439-1447)
        // plus company_sellable (base.py:2109-2110) require that the company be
        // owned by a PLAYER (any player), and not by the buying corporation. A
        // company owned by a Corporation is not sellable. There is NO
        // president restriction in Python.
        let company_owner_id = self.companies[company_idx]
            .owner
            .player_id()
            .ok_or_else(|| GameError::new(format!("Cannot buy {} (not owned by a player)", company_sym)))?;

        // Validate price: between ceil(value/2) and 2x face value.
        // Python: Company.min_price = (value//2)+(value%2) = ceil(value/2),
        // max_price = value*2 (entities.py:938-939, get_max_price entities.py:1020).
        let face_value = self.companies[company_idx].value;
        let min_price = (face_value + 1) / 2;
        let max_price = face_value * 2;
        if price < min_price || price > max_price {
            return Err(GameError::new(format!(
                "Price {} must be between {} and {} (2x face value)",
                price, min_price, max_price
            )));
        }

        if self.corporations[corp_idx].cash < price {
            return Err(GameError::new("Corporation cannot afford company"));
        }

        // Transfer. Python pays the seller `owner = company.owner` (round.py:1483),
        // which is the company's current player-owner (not the president).
        self.corporations[corp_idx].cash -= price;
        let owner_idx = self.player_index(company_owner_id).unwrap();
        self.players[owner_idx].cash += price;
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

        // Place home token if the corp hasn't placed one yet (tokens[0] unused).
        if !self.corporations[ci].tokens.is_empty()
            && !self.corporations[ci].tokens[0].used
        {
            // Python's place_home_token blocks (pending_tokens) when:
            //   tile.reserved_by(corporation) AND any(tile.paths)
            // In 1830, only E11 has a tile-level reservation (ERIE). All other
            // corps use city-level reservations which don't trigger tile.reserved_by.
            // So we block only when: home hex is E11, tile has paths (upgraded),
            // and the tile has multiple cities (choice is meaningful).
            let corp_defs = crate::title::g1830::corporations();
            let mut needs_choice = false;
            if let Some(cd) = corp_defs.iter().find(|cd| cd.sym == sym) {
                let has_tile_reservation = cd.home_hex == "E11";
                if has_tile_reservation {
                    if let Some(&hi) = self.hex_idx.get(cd.home_hex) {
                        let tile = &self.hexes[hi].tile;
                        let has_paths = !tile.paths.is_empty();
                        if has_paths && tile.cities.len() > 1 {
                            needs_choice = true;
                        }
                    }
                }
            }
            if needs_choice {
                // Add to pending_tokens AND set step to PlaceToken.
                // This mirrors Python's HomeToken step which runs before Track.
                // The hex_id is the corp's home hex (only E11 reaches this path
                // in 1830 — ERIE's tile-reserved home).
                let home_hex = corp_defs
                    .iter()
                    .find(|cd| cd.sym == sym)
                    .map(|cd| cd.home_hex.to_string())
                    .unwrap_or_default();
                if let crate::rounds::Round::Operating(ref mut s) = self.round {
                    s.pending_tokens.push((sym.clone(), 0, home_hex));
                    s.step = OperatingStep::PlaceToken;
                }
            } else {
                self.place_home_token(ci);
            }
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

    /// Advance past auto-skippable steps in the current corp's operating
    /// turn (Python `BaseRound.skip_steps` + the per-step `skip!` hooks).
    /// Stops at the first pc whose listed step blocks at the current state.
    ///
    /// Table-driven: the title's OR step list supplies both the pc SEQUENCE
    /// (`crate::steps::next_operating_pc`) and the step kind whose skip
    /// predicate runs at each pc (`operating_step_auto_skips`). Adding a step
    /// to a title = write its predicate arm + list it in the title's round
    /// description; this loop never changes.
    pub(crate) fn skip_steps(&mut self) {
        let descs = self.operating_step_descs();
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

            // pc -> the listed step that executes it. A pc not in the list
            // (Done) blocks: the caller's advance machinery takes over.
            let kind = match descs
                .iter()
                .find_map(|d| (d.operating_pc() == Some(step.clone())).then_some(d.kind))
            {
                Some(k) => k,
                None => return,
            };

            if !self.operating_step_auto_skips(kind, corp_idx, &corp_sym) {
                return;
            }

            // Advance to the next pc in the title's step list.
            if let crate::rounds::Round::Operating(ref mut s) = self.round {
                s.step = crate::steps::next_operating_pc(descs, &step);
            }
        }
    }

    /// Whether the listed step auto-skips (Python `step.skip!`) when the
    /// turn's pc reaches it — i.e. it does NOT block at the current state.
    /// May carry the step's skip side effect (Dividend auto-withholds: 0
    /// revenue moves the share price left). Skip conditions:
    /// - Track: always blocking (requires explicit lay_tile or pass)
    /// - Token: skip if no available tokens or can't afford (pending home/OO
    ///   tokens block)
    /// - Route: skip if corp has no runnable train or no route
    /// - Dividend: skip if revenue == 0 (auto-withhold, move price left)
    /// - DiscardTrain: skip if no corp is over the train limit
    /// - BuyTrain: blocking when a train can be bought or must be bought
    /// - BuyCompany(blocking): blocking when a purchasable company or an
    ///   unused lay ability is open
    fn operating_step_auto_skips(
        &mut self,
        kind: crate::steps::StepKind,
        corp_idx: usize,
        corp_sym: &str,
    ) -> bool {
        use crate::steps::StepKind;
        let corp_sym = corp_sym.to_string();
        {
            let should_skip = match kind {
                StepKind::Track => {
                    // Always blocking — requires explicit lay_tile or pass
                    false
                }
                StepKind::Token => {
                    // Check for pending tokens from OO upgrade
                    let has_pending = match &self.round {
                        crate::rounds::Round::Operating(s) => !s.pending_tokens.is_empty(),
                        _ => false,
                    };
                    if has_pending {
                        false // blocking — displaced tokens must be re-placed
                    } else {
                        // Skip if the corp can't place a token anywhere.
                        !self.can_place_token(&corp_sym)
                    }
                }
                StepKind::Route => {
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
                StepKind::Dividend => {
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
                StepKind::DiscardTrain => {
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
                StepKind::BuyCompany => {
                    let phase_num: u8 = self.phase.name.parse().unwrap_or(0);
                    if phase_num < 3 {
                        true
                    } else {
                        let corp_cash = self.corporations[corp_idx].cash;
                        let can_buy_company = self.companies.iter().any(|c| {
                            !c.closed && !c.no_buy && c.owner.is_player()
                                && corp_cash >= c.value / 2
                        });
                        // A corp holding a company with an unused bonus
                        // tile_lay ability (CS) keeps the step blocking; a
                        // teleport ability (DH) doesn't (it's filtered by
                        // Python's abilities() timing check).
                        let has_ability = self.corp_has_unused_lay_ability(&corp_sym);
                        !can_buy_company && !has_ability
                    }
                }
                StepKind::BuyTrain => {
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
                        // D-trains become purchasable once phase 6 is reached
                        // (D-trains have available_on="6"). They're also available
                        // in phase D. Check phase name for "6" or "D".
                        let d_available = self.phase.name == "D" || self.phase.name == "6";
                        // Python's `discountable_trains_for` looks at
                        // `depot.depot_trains()` — the VISIBLE upcoming trains
                        // PLUS the discarded pool — not just the upcoming queue.
                        // Once the last upcoming D is bought the queue empties,
                        // but discarded D-trains remain exchangeable, so the
                        // BuyTrain step must stay blocking (the corp can still
                        // exchange an owned 4/5/6 for a discounted discarded D).
                        // Find a D across both upcoming and discarded.
                        let d_train_price = self
                            .depot
                            .trains
                            .iter()
                            .chain(self.depot.discarded.iter())
                            .find(|t| t.name == "D")
                            .map(|t| t.price);
                        let can_exchange = if d_available {
                            if let Some(d_price) = d_train_price {
                                let discount = 300;
                                let exchange_price = d_price - discount;
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
                // Steps with no auto-skip hook (HomeToken; future kinds)
                // block when the pc lands on them.
                _ => false,
            };
            should_skip
        }
    }

    /// Check if a corporation can place a token anywhere on the board.
    /// Requires: at least one unplaced token, enough cash to pay for it,
    /// and at least one reachable city with an open token slot.
    fn or_process_pass(
        &mut self,
        state: &OperatingState,
        entity_id: &str,
    ) -> Result<(), GameError> {
        let mut new_state = state.clone();

        // Strict dispatch — pass action must come from the current operator.
        let cur = new_state.current_corp_sym().ok_or_else(|| GameError::new("No current corp"))?;
        if entity_id != cur {
            return Err(GameError::new(format!(
                "pass entity {} does not match current operator {}",
                entity_id, cur
            )));
        }

        if new_state.step == OperatingStep::BuyCompany {
            // Pass from final BuyCompany ends the corp's turn
            new_state.advance_to_next_corp();
            self.round = crate::rounds::Round::Operating(new_state);
            self.update_round_state();
            if !matches!(&self.round, crate::rounds::Round::Operating(s) if s.finished) {
                self.start_operating();
            }
        } else {
            // When the corp MUST buy a train (Python's `president_may_contribute`
            // == `must_buy_train`), the BuyTrain step's legal actions are
            // [SellShares, BuyTrain] — Pass is EXCLUDED (round.py:805-810). A
            // Pass at that point is rejected by Python's blocking-step guard
            // (round.py:5356-5357) and now propagates as a real error (neither
            // engine swallows failed passes). The enumerator never offers this
            // pass, so reaching here signals a bad caller; reject it.
            if new_state.step == OperatingStep::BuyTrain
                && self.president_may_contribute_pub(cur)
            {
                return Err(GameError::new(
                    "Blocking step Buy Trains cannot process action Pass",
                ));
            }
            // Pass from any blocking step: advance to the next pc per the
            // title's step list.
            new_state.step = crate::steps::next_operating_pc(self.operating_step_descs(), &new_state.step);
            self.round = crate::rounds::Round::Operating(new_state);
            self.update_round_state();
            // skip_steps is called by process_operating_action after this returns
        }

        Ok(())
    }
}
