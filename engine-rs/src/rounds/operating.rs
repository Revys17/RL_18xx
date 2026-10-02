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
                        self.or_advance_to_next_corp();
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
                let first_pc = crate::steps::first_operating_pc(self.operating_step_descs());
                if let crate::rounds::Round::Operating(ref mut s) = self.round {
                    s.pending_tokens.remove(0);
                    // Home token: back to the start of the normal operating
                    // turn (the title's first pc — 1830 LayTile, 1867
                    // RedeemShares).
                    if was_home_token && s.pending_tokens.is_empty() {
                        s.step = first_pc;
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
                    self.or_advance_to_next_corp();
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
                        let corp_defs = self.title_def().corporations();
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
                            let first_pc =
                                crate::steps::first_operating_pc(self.operating_step_descs());
                            if let crate::rounds::Round::Operating(ref mut s) = self.round {
                                if s.num_laid_track == 0 {
                                    // Start of turn — back to the title's
                                    // first pc (1830 LayTile, 1867
                                    // RedeemShares).
                                    s.step = first_pc;
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
                                    self.or_advance_to_next_corp();
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
            Action::BuyShares {
                entity_id,
                corporation_sym,
                percent,
                share_indices,
                ..
            } => self.or_process_redeem_shares(
                &state,
                entity_id,
                corporation_sym,
                *percent,
                share_indices,
            ),
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
            self.or_advance_to_next_corp();
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

        // The title's per-turn lay allowance (Ruby TILE_LAYS; 1830: one
        // unrestricted lay, 1867: two with a constrained second).
        let lays = self.title_def().tile_lays();
        let Some(slot) = lays.get(new_state.num_laid_track as usize).copied() else {
            return Err(GameError::new("Already laid a tile this turn"));
        };
        if slot.cannot_reuse_same_hex && new_state.laid_hexes.iter().any(|h| h == hex_id) {
            return Err(GameError::new(format!(
                "Cannot lay on {} again this turn",
                hex_id
            )));
        }
        // An upgrade = laying a non-yellow tile (yellow only ever goes on
        // untiled white hexes; green/brown/gray replace existing track).
        let is_upgrade = {
            let base = tile_id.split('-').next().unwrap_or(tile_id);
            self.tile_catalog
                .get(base)
                .map_or(false, |td| td.color != crate::tiles::TileColor::Yellow)
        };
        match slot.upgrade {
            crate::title::UpgradeAllowance::Yes => {}
            crate::title::UpgradeAllowance::No if is_upgrade => {
                return Err(GameError::new("This lay may not be an upgrade"));
            }
            crate::title::UpgradeAllowance::NotIfUpgraded
                if is_upgrade && new_state.upgraded_this_turn =>
            {
                return Err(GameError::new(
                    "Cannot upgrade twice in one turn",
                ));
            }
            _ => {}
        }
        if !slot.lay && !is_upgrade {
            return Err(GameError::new("This lay may not place a new tile"));
        }

        let hex_idx = *self
            .hex_idx
            .get(hex_id)
            .ok_or_else(|| GameError::new(format!("Unknown hex: {}", hex_id)))?;

        // Pay terrain cost (+ the lay slot's surcharge — 1867's $20 second
        // lay).
        let terrain_cost: i32 = self.hexes[hex_idx]
            .tile
            .upgrades
            .iter()
            .map(|u| u.cost)
            .sum::<i32>()
            + slot.cost;

        if terrain_cost > 0 {
            let corp_sym = new_state
                .current_corp_sym()
                .ok_or_else(|| GameError::new("No current corp"))?
                .to_string();
            let corp_idx = self.corp_idx[&corp_sym];
            // 1867's Track step includes AutomaticLoan: lay costs beyond
            // cash are loan-funded (no-op for titles without loans).
            if self.title_def().loan_value() > 0 {
                self.auto_take_loans(corp_idx, terrain_cost);
            }
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
                if let Some(city_slots) = self.title_def().tile_cities(base_tile_id) {
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
                // Transfer tokens from old cities to new cities — Ruby
                // Hex#lay (hex.rb:115-153) + city_map_for (hex.rb:258-292):
                //
                //   1. With NO exits on any old city (and equal counts),
                //      cities map by index.
                //   2. Otherwise each old city maps to the first new city
                //      whose exits are a SUPERSET of its own (old ⊆ new) —
                //      subset, not mere overlap: 1867's X3→X5/X6/X7 brown
                //      Montreal merges three 1-slot cities into 2+1 slots,
                //      and a partial-overlap first-match can pick the wrong
                //      survivor city.
                //   3. Unmapped old cities fall back to the same index when
                //      that city is still unclaimed, else the first
                //      unclaimed new city (hex.rb:279-287).
                //
                // Tokens then move per DESTINATION city in old-city order
                // via City#exchange_token → get_slot (city.rb:125-133,
                // 172-190): the corp's own reservation slot if any, else
                // the first open un-reserved slot — NOT the token's old
                // slot index (two old cities merging into one would
                // collide and silently drop a token). When the merge
                // overflows the city, Ruby places a "cheater" token in an
                // appended extra slot (move_tokens_to_new_tile_multi_city!,
                // hex.rb:294-330).
                let old_tile_base = old_tile_name.split('-').next().unwrap_or(&old_tile_name);
                let mut old_exits = self.city_exits_from_catalog(old_tile_base, old_tile_rotation);
                if old_exits.is_empty() {
                    // Preprinted tiles aren't in the catalog — derive from
                    // the live tile (1867 Montreal/Ottawa: the preprinted
                    // city order differs from the green tiles', so the
                    // positional fallback would misplace home tokens).
                    old_exits = crate::game::BaseGame::city_exits_from_tile(&self.hexes[hex_idx].tile);
                }
                let new_exits = self.city_exits_from_catalog(base_tile_id, rotation);
                let n_new = new_tile.cities.len();

                // city_map[old_ci] = Some(new_ci) | None (no destination).
                let mut city_map: Vec<Option<usize>> = vec![None; old_cities.len()];
                if old_exits.iter().all(|e| e.is_empty()) && old_cities.len() == n_new {
                    for (old_ci, slot) in city_map.iter_mut().enumerate() {
                        *slot = Some(old_ci);
                    }
                } else {
                    for (old_ci, oe) in old_exits.iter().enumerate().take(old_cities.len()) {
                        if !oe.is_empty() {
                            city_map[old_ci] = new_exits
                                .iter()
                                .position(|ne| oe.iter().all(|e| ne.contains(e)));
                        }
                    }
                    let mut claimed: Vec<usize> = city_map.iter().flatten().copied().collect();
                    for old_ci in 0..old_cities.len() {
                        if city_map[old_ci].is_some() {
                            continue;
                        }
                        let dest = if old_ci < n_new && !claimed.contains(&old_ci) {
                            Some(old_ci)
                        } else {
                            (0..n_new).find(|j| !claimed.contains(j))
                        };
                        city_map[old_ci] = dest;
                        if let Some(d) = dest {
                            claimed.push(d);
                        }
                    }
                }

                // Group the moving tokens by destination city, old-city order.
                let mut moved: Vec<Vec<crate::entities::Token>> = vec![Vec::new(); n_new];
                for (old_ci, old_city) in old_cities.iter().enumerate() {
                    let toks: Vec<_> = old_city.tokens.iter().flatten().cloned().collect();
                    if toks.is_empty() {
                        continue;
                    }
                    let Some(dest_ci) = city_map[old_ci] else {
                        // Ruby raises here too (hex.rb:303-307).
                        return Err(GameError::new(format!(
                            "No city found on new tile {} for tokens from {} city {}",
                            tile_id, hex_id, old_ci
                        )));
                    };
                    moved[dest_ci].extend(toks);
                }

                // Positional reservation model (see token_slot_for): a live
                // home reservation occupies slot 0 of its city.
                let home_res = self.home_reservations();
                for (nci, toks) in moved.into_iter().enumerate() {
                    let mut reservations: Vec<Option<String>> =
                        vec![None; new_tile.cities[nci].tokens.len()];
                    for (rh, rc, rsym) in &home_res {
                        if rh == hex_id && *rc == nci && !reservations.is_empty() {
                            reservations[0] = Some(rsym.clone());
                        }
                    }
                    for tok in toks {
                        let city = &mut new_tile.cities[nci];
                        let own_res = reservations
                            .iter()
                            .position(|r| r.as_deref() == Some(tok.corporation_id.as_str()));
                        let slot = own_res.or_else(|| {
                            city.tokens
                                .iter()
                                .enumerate()
                                .position(|(i, t)| t.is_none() && reservations[i].is_none())
                        });
                        match slot {
                            Some(i) => city.tokens[i] = Some(tok),
                            None => {
                                // Cheater token: an appended extra slot
                                // beyond normal_slots (city.rb:130,190 —
                                // @tokens[@tokens.size]); `slots` stays the
                                // printed slot count, Ruby's normal_slots.
                                city.tokens.push(Some(tok));
                                reservations.push(None);
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

        // Laying the gray Montreal tile claims the CN's reserved token
        // (Ruby G1867::Step::Track: `place_639_token if tile.name == '639'`).
        if base_tile_id == "639" {
            self.place_639_token(hex_id);
        }

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
        new_state.laid_hexes.push(hex_id.to_string());
        if is_upgrade {
            new_state.upgraded_this_turn = true;
        }

        // Advance off the Track step once the title's lay allowance is
        // exhausted (1830: after the single lay; 1867: after the second —
        // an explicit pass moves on earlier).
        if (new_state.num_laid_track as usize) >= self.title_def().tile_lays().len() {
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

        // 1867 (G1867::Step::Token adjust_token_price_ability!): the token
        // costs its charter price × the straight-line hex distance from the
        // corp's NEAREST placed token. Re-check affordability at the real
        // cost (the early check used the base price).
        let token_cost = if self.title_def().token_price_by_distance() {
            let dist = self
                .min_token_hex_distance(corp_idx, &resolved_hex_id)
                .unwrap_or(1);
            token_cost * dist
        } else {
            token_cost
        };
        if self.corporations[corp_idx].cash < token_cost {
            return Err(GameError::new(format!(
                "{} cannot afford token cost {}",
                corp_sym, token_cost
            )));
        }

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

    /// Straight-line hex distance from the corp's nearest placed token to
    /// `target_hex` (Ruby Hex#distance via [`hex_crow_distance`]).
    fn min_token_hex_distance(&self, corp_idx: usize, target_hex: &str) -> Option<i32> {
        let flat = self.title_def().hex_layout() == crate::title::HexLayout::Flat;
        self.corporations[corp_idx]
            .tokens
            .iter()
            .filter(|t| t.used && !t.city_hex_id.is_empty())
            .filter_map(|t| hex_crow_distance(&t.city_hex_id, target_hex, flat))
            .min()
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

        // Price the routes. Titles whose imports record connections without
        // revenue (1867) compute every route themselves via the title hook;
        // otherwise the recorded revenue is authoritative (1830 — route
        // validation is Phase 4).
        let title = self.title_def();
        let mut total_revenue: i32 = extra_revenue;
        for r in routes {
            total_revenue += match title.recorded_route_revenue(self, cur, r) {
                Some(computed) => computed?,
                None => r.revenue.unwrap_or(0),
            };
        }

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

        // Ruby writes operating_history at process_dividend — the corp has
        // now "operated" (gates SELL_AFTER=:operate sales).
        self.corporations[corp_idx].ever_operated = true;

        // The corporation-withheld part per kind (Ruby dividend_options:
        // withhold keeps all, half keeps `(rev/2/total).floor × total` —
        // Step::HalfPay — and payout keeps nothing); the rest distributes
        // per share.
        let withheld = match kind {
            DividendKind::Payout => 0,
            DividendKind::Withhold => revenue,
            DividendKind::Half => {
                let total = self.corporations[corp_idx].num_share_units();
                (revenue / 2 / total) * total
            }
        };
        if withheld > 0 {
            self.corporations[corp_idx].cash += withheld;
            self.bank.cash -= withheld;
        }
        let distributed = revenue - withheld;
        if distributed > 0 {
            self.distribute_revenue(corp_idx, distributed)?;
        }

        // Share-price movement on the DISTRIBUTED amount (Ruby
        // change_share_price gets `revenue - payout[:corporation]`).
        let movement = match self.title_def().dividend_movement() {
            crate::title::DividendMovement::Standard => {
                if distributed > 0 {
                    Some(true) // right
                } else {
                    Some(false) // left
                }
            }
            crate::title::DividendMovement::RightIfGePrice => {
                let price = self.corporations[corp_idx]
                    .share_price
                    .as_ref()
                    .map_or(0, |sp| sp.price);
                if distributed <= 0 {
                    Some(false)
                } else if distributed >= price {
                    Some(true)
                } else {
                    None
                }
            }
        };
        if let Some(right) = movement {
            if let Some(sp) = self.corporations[corp_idx].share_price.clone() {
                let (new_row, new_col) = if right {
                    self.stock_market.move_right(sp.row, sp.column)
                } else {
                    self.stock_market.move_left(sp.row, sp.column)
                };
                if let Some(new_sp) = self.stock_market.share_price_at(new_row, new_col) {
                    self.corporations[corp_idx].share_price = Some(new_sp);
                    self.update_market_cell(&corp_sym, sp.row, sp.column, new_row, new_col);
                }
            }
        }

        // Next pc from the title's step list (1830: BuyTrain via the
        // auto-skipped DiscardTrain; 1867: the BuyCompanyPreloan window +
        // LoanOperations come first for majors). The post-action skip_steps
        // pass advances through whatever doesn't ask.
        new_state.step =
            crate::steps::next_operating_pc(self.operating_step_descs(), &OperatingStep::Dividend);

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
        let share_unit = corp.share_unit();
        let total_shares = corp.num_share_units();
        let per_share = revenue / total_shares;

        // Pay each shareholder based on their number of share units
        for player in &mut self.players {
            let eid = EntityId::player(player.id);
            let percent = corp.percent_owned_by(&eid);
            if percent > 0 {
                let num_shares = percent as i32 / share_unit;
                let payout = num_shares * per_share;
                player.cash += payout;
                self.bank.cash -= payout;
            }
        }

        // The corporation's own cut (Ruby Step::Dividend
        // `holder_for_corporation`): full capitalization (1830) → the
        // market pool's shares pay the corp and IPO shares pay no one;
        // incremental (1867) → the corp's TREASURY shares pay the corp and
        // pool shares pay no one.
        let holder_eid = match corp.capitalization {
            crate::title::Capitalization::Full => EntityId::market(),
            crate::title::Capitalization::Incremental => EntityId::ipo(&corp.sym),
        };
        let holder_pct = corp.percent_owned_by(&holder_eid);
        if holder_pct > 0 {
            let holder_shares = holder_pct as i32 / share_unit;
            let corp_payout = holder_shares * per_share;
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
            // other_trains(entity): other-corp trains. 1830 restricts to
            // sellers with the SAME president (ALLOW_TRAIN_BUY_FROM_OTHER_
            // PLAYERS == false); 1867 trades trains between any corps.
            let cross_president = self.title_def().train_buy_from_other_players();
            let buyer_pres = self.corporations[corp_idx].president_id();
            for other in &self.corporations {
                if other.sym == corp_sym {
                    continue;
                }
                if !cross_president && (buyer_pres.is_none() || other.president_id() != buyer_pres)
                {
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
            // action's ``price`` is the source of truth (no face-value check
            // when the corp pays on its own — Python only enforces "Cannot
            // buy for more than cost" inside the president-contribution
            // branch, round.py:582-583).
            let actual_price = price;

            // 1867 AutomaticLoan (step/buy_train.rb try_take_loan): when the
            // corp MUST buy a train and it comes from the DEPOT, loans are
            // taken implicitly until the price is covered. Other shortfalls
            // (optional buys, inter-corp) stay errors for loans titles.
            if self.title_def().loan_value() > 0
                && self.corporations[corp_idx].cash < actual_price
                && self.operating_must_buy_train(&corp_sym)
            {
                self.auto_take_loans(corp_idx, actual_price);
            }

            // Check if corp can afford; if not, president contributes.
            let corp_pays = self.corporations[corp_idx].cash.min(actual_price);
            let president_pays = actual_price - corp_pays;
            if president_pays > 0 {
                // Python only lets the president fund the shortfall when
                // `president_may_contribute(entity)` — i.e. `must_buy_train`
                // (round.py:550-551, 576-577). Otherwise the contribution
                // block is skipped and `entity.spend(price)` raises on the
                // corp's insufficient cash. Loans-EMR titles (1867) never
                // let the president contribute.
                if !self.title_def().ebuy_president_may_contribute()
                    || !self.president_may_contribute_pub(&corp_sym)
                {
                    return Err(GameError::new(format!(
                        "{} cannot afford {} for the {} train (has {}) and the president may not contribute",
                        corp_sym, actual_price, train_name, self.corporations[corp_idx].cash
                    )));
                }
                // Python forbids president contribution on an exchange buy
                // (round.py:579-581).
                if ex_idx.is_some() {
                    return Err(GameError::new("Cannot contribute funds when exchanging"));
                }
                // check_for_cheapest_train (round.py:765-773): when the
                // president contributes, a from-depot train must be the
                // cheapest train in the depot (EBUY_DEPOT_TRAIN_MUST_BE_CHEAPEST,
                // base.py:532; EBUY_OTHER_VALUE exempts corp-owned trains only).
                let head_name = self.depot.trains.first().map(|t| t.name.clone());
                let cheapest: Option<(i32, String)> = self
                    .depot
                    .trains
                    .iter()
                    .filter(|t| {
                        self.phase_available(t.available_on.as_deref())
                            || Some(&t.name) == head_name.as_ref()
                    })
                    .chain(self.depot.discarded.iter())
                    .map(|t| (t.price, t.name.clone()))
                    .min_by_key(|(p, _)| *p);
                if let Some((_, ref cheapest_name)) = cheapest {
                    if cheapest_name != base_name {
                        return Err(GameError::new(format!(
                            "Cannot purchase {} train: cheaper train available ({})",
                            base_name, cheapest_name
                        )));
                    }
                }
                // "Cannot buy for more than cost" (round.py:582-583): with the
                // president contributing, the price is capped at the train's
                // face value.
                let train_face = if from_discarded {
                    self.depot.discarded[train_idx].price
                } else {
                    self.depot.trains[train_idx].price
                };
                if actual_price > train_face {
                    return Err(GameError::new("Cannot buy for more than cost"));
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
            if actual_from == corp_sym {
                // Python buy_train_action (round.py:573-574).
                return Err(GameError::new(
                    "An entity cannot buy a train from itself",
                ));
            }
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

            // check_spend (round.py:824-839): a corp-owned train's price must
            // fall inside `spend_minmax` (round.py:749-757). For 1830,
            // EBUY_OTHER_VALUE == True and buying_power(corp) == corp.cash:
            //   corp.cash < face (emergency):
            //     min = 1, or corp.cash + president.cash - last_share_sold_price + 1
            //           after an emergency share sale this turn (752-753);
            //     max = min(face, corp.cash + president.cash)
            //   otherwise: min = 1, max = corp.cash (no face cap).
            // Python runs this BEFORE buy_train_action and regardless of
            // president_may_contribute — the contribution gate below then
            // rejects shortfalls the president may not fund.
            {
                let face = self.corporations[from_corp_idx].trains[train_idx].price;
                let bp = self.corporations[corp_idx].cash;
                let pres_cash = self.corporations[corp_idx]
                    .president_id()
                    .and_then(|pid| self.players.iter().find(|p| p.id == pid))
                    .map_or(0, |p| p.cash);
                let last_sold = match &self.round {
                    crate::rounds::Round::Operating(s) => s.last_share_sold_price,
                    _ => None,
                };
                let (min_p, max_p) = if bp < face {
                    let min_p = match last_sold {
                        Some(lssp) => bp + pres_cash - lssp + 1,
                        None => 1,
                    };
                    (min_p, face.min(bp + pres_cash))
                } else {
                    (1, bp)
                };
                if !(min_p <= price && price <= max_p) {
                    // The "may not buy a train from another corporation"
                    // branch requires `not EBUY_OTHER_VALUE` (round.py:832) —
                    // never for 1830, so always the range message.
                    return Err(GameError::new(format!(
                        "{} may not spend {} on {}'s {} train; may only spend between {} and {}.",
                        corp_sym, price, actual_from, base_name, min_p, max_p
                    )));
                }
            }

            // Transfer money — president contributes ONLY when Python's
            // `president_may_contribute` (== must_buy_train, round.py:550-551,
            // 576-577) holds; otherwise the corp pays alone and an unfunded
            // shortfall is an error (Python: entity.spend raises).
            let corp_pays = self.corporations[corp_idx].cash.min(price);
            if corp_pays < price {
                let president_pays = price - corp_pays;
                if !self.title_def().ebuy_president_may_contribute()
                    || !self.president_may_contribute_pub(&corp_sym)
                {
                    return Err(GameError::new(format!(
                        "{} cannot afford {} for the {} train (has {}) and the president may not contribute",
                        corp_sym, price, train_name, self.corporations[corp_idx].cash
                    )));
                }
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
            if let Some(co_sym) = crate::abilities::close_on_bought_train(&self.title, &corp_sym) {
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
        let train_limit = self.corp_train_limit(corp_idx);
        if self.corporations[corp_idx].trains.len() > train_limit {
            if !new_state.crowded_corps.contains(&corp_sym) {
                new_state.crowded_corps.push(corp_sym.clone());
            }
            new_state.step = OperatingStep::DiscardTrain;
        }

        self.round = crate::rounds::Round::Operating(new_state);

        // Ruby runs `post_train_buy` after EVERY purchase (depot or
        // inter-corp), once events/phase/rusting are settled — the 1867
        // trainless-nationalization consumer (no-op elsewhere). It runs
        // AFTER the new state is installed: nationalizing minors must not
        // be clobbered by the `new_state` write, and while the resulting
        // MajorTrainless queue is non-empty the round freezes as-is.
        self.post_train_buy();

        // If the purchase's events nationalized the BUYING corp itself
        // (`minors_nationalized` on the first 8 can hit a minor buyer),
        // its turn is over — Ruby `@round.force_next_entity!`.
        let buyer_dead = self
            .corp_idx
            .get(corp_sym.as_str())
            .map_or(false, |&ci| {
                self.corporations[ci].closed || !self.corporations[ci].floated
            });
        if buyer_dead {
            if let crate::rounds::Round::Operating(ref mut s) = self.round {
                if s.current_corp_sym() == Some(corp_sym.as_str()) {
                    s.step = OperatingStep::Done;
                }
            }
        }

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

    /// The certificates a sale of `percent`% of corporation `corp_idx` by
    /// `player_eid` moves when the action names none (the native decode never
    /// does): the seller's non-president certs by ascending index until the
    /// non-president portion is covered — `percent` minus the president
    /// cert's face value when the presidency is dumped — and whether the
    /// president cert goes too (the seller would drop below the president's
    /// percent). Python's `all_bundles_for_corporation` builds the same
    /// [normals..., president] bundle. Shared by the sell validation, both
    /// sale paths, and the native action log, so the logged share ids are
    /// exactly the certs the sale moves.
    pub(crate) fn implied_sell_bundle(
        &self,
        corp_idx: usize,
        player_eid: &crate::entities::EntityId,
        percent: u8,
    ) -> (Vec<usize>, bool) {
        let corp = &self.corporations[corp_idx];
        let pres_pct: u8 = corp
            .shares
            .iter()
            .find(|s| s.president)
            .map(|s| s.percent)
            .unwrap_or(20);
        let remaining_after = corp.percent_owned_by(player_eid).saturating_sub(percent);
        let includes_president = corp
            .shares
            .iter()
            .any(|s| s.president && s.owner == *player_eid)
            && remaining_after < corp.president_percent();
        let non_pres_to_sell = if includes_president {
            percent.saturating_sub(pres_pct)
        } else {
            percent
        };
        let mut normals: Vec<usize> = Vec::new();
        let mut taken = 0u8;
        for (i, share) in corp.shares.iter().enumerate() {
            if taken >= non_pres_to_sell {
                break;
            }
            if share.owner == *player_eid && !share.president {
                normals.push(i);
                taken += share.percent;
            }
        }
        (normals, includes_president)
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
            let (normals, dumps_president) = self.implied_sell_bundle(ci, &player_eid, percent);
            includes_president = dumps_president;
            let non_pres_needed = if includes_president {
                percent.saturating_sub(pres_pct)
            } else {
                percent
            };
            let mut pcts: Vec<u8> = normals.iter().map(|&i| corp.shares[i].percent).collect();
            // The selection stops once the non-president portion is covered,
            // so falling short means the seller doesn't hold enough of it.
            if pcts.iter().sum::<u8>() < non_pres_needed {
                return Err(reject());
            }
            if includes_president {
                pcts.push(pres_pct);
            }
            bundle_cert_pcts = pcts;
        }

        // -- check_sale_timing (base.rb:1171-1188) --------------------------
        // SELL_AFTER :first (1830): `turn > 1 or round.operating` — no
        // first-stock-round sales, OR sales always pass. :operate (1867):
        // only shares of corporations that have operated, in any round.
        match self.title_def().sell_after() {
            crate::title::SellAfter::FirstStockRound => {
                if operating_corp.is_none() && self.turn <= 1 {
                    return Err(reject());
                }
            }
            crate::title::SellAfter::Operate => {
                if !corp.ever_operated {
                    return Err(reject());
                }
            }
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
            // ceil(units × price): price is per share unit
            let unit = self
                .corp_idx
                .get(corporation_sym)
                .map_or(10, |&i| self.corporations[i].share_unit()) as i64;
            let price_for_pct =
                |pct: u8| -> i32 { ((share_price as i64 * pct as i64 + unit - 1) / unit) as i32 };
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
        // be left with less than the president's percent after the sale.
        let player_total = self.corporations[corp_idx].percent_owned_by(&player_eid);
        let remaining_after = player_total.saturating_sub(percent);
        let includes_president = self.corporations[corp_idx]
            .shares
            .iter()
            .any(|s| s.president && s.owner == player_eid)
            && remaining_after < self.corporations[corp_idx].president_percent();

        // Transfer shares to market. When dumping the president, transfer the
        // non-president portion (= percent - president_face_value, i.e.
        // percent - 20 in 1830) first, then move the 20% president share to
        // market. ``check_president_change_with_snapshot`` below will then route the
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
        // (`implied_sell_bundle`) when no indices are supplied — which is NOT
        // rare: the native decode emits empty ``share_indices`` for EVERY
        // SellShares (decode.rs), and the bankruptcy liquidation constructs
        // its own bundles. Either way the correct PERCENT reaches the market.

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
            to_market = self.implied_sell_bundle(corp_idx, &player_eid, percent).0;
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

        // Player receives money (units sold × price-per-unit)
        let share_unit = self.corporations[corp_idx].share_unit();
        let revenue = (percent as i32 * share_price.price) / share_unit;
        self.players[player_idx].cash += revenue;
        self.bank.cash -= revenue;

        // Share price drops: move DOWN once per share unit sold
        let num_shares = percent as u32 / share_unit as u32;
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
                let shares_to_return = deficit / share_unit as u8;
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
        let train_limit = self.corp_train_limit(corp_idx);
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

        // Validate price against the title's bounds (1830: ceil(value/2) ..
        // 2× face, Python entities.py:938-939/get_max_price; 1867: $1 ..
        // face via CompanyPriceUpToFace).
        let face_value = self.companies[company_idx].value;
        let (min_price, max_price) = self.title_def().company_buy_price_range(face_value);
        if price < min_price || price > max_price {
            return Err(GameError::new(format!(
                "Price {} must be between {} and {}",
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
            let corp_defs = self.title_def().corporations();
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
            self.or_advance_to_next_corp();

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
        // 1867 MajorTrainless: while trainless majors owe a decision the
        // step (listed FIRST) blocks the whole round — Ruby's skip walk
        // `break if step.blocking?` stops before anything auto-skips. The
        // frozen turn resumes from the choice interceptor once the queue
        // drains (game.rs process_action_internal).
        if !self.trainless_major.is_empty() {
            return;
        }
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
                StepKind::RedeemShares => {
                    // Ruby G1867::Step::RedeemShares#actions: empty (→ the
                    // base skip) unless redeemable_shares(corp) is non-empty.
                    !self.corp_can_redeem_share(corp_idx)
                }
                StepKind::Track => {
                    // Ruby Tracker#can_lay_tile?: the step blocks while the
                    // NEXT lay-allowance slot is usable and its slot cost is
                    // within buying power — FULL power on AutomaticLoan
                    // titles (probed on 21268: minor CA, $10 cash + 2
                    // takeable loans, IS asked for the $20 second lay; minor
                    // BO, $1 cash + 0 takeable, is NOT). 1830's single free
                    // slot always blocks pre-lay; its pc advances on the lay
                    // itself, so this arm is 1830-inert.
                    let (laid, upgraded) = match &self.round {
                        crate::rounds::Round::Operating(s) => {
                            (s.num_laid_track as usize, s.upgraded_this_turn)
                        }
                        _ => (0, false),
                    };
                    match self.title_def().tile_lays().get(laid) {
                        None => true, // allowance exhausted
                        Some(entry) => {
                            let upgrade_ok = match entry.upgrade {
                                crate::title::UpgradeAllowance::Yes => true,
                                crate::title::UpgradeAllowance::No => false,
                                crate::title::UpgradeAllowance::NotIfUpgraded => !upgraded,
                            };
                            let usable = entry.lay || upgrade_ok;
                            !usable || self.corp_buying_power_full(corp_idx) < entry.cost
                        }
                    }
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
                    // Both auto arms ARE the dividend step running — Ruby
                    // Dividend#skip! calls process_dividend, which writes
                    // operating_history ("operated", SELL_AFTER=:operate).
                    self.corporations[corp_idx].ever_operated = true;
                    if revenue == 0 {
                        if let Some(sp) = self.corporations[corp_idx].share_price.clone() {
                            let (nr, nc) = self.stock_market.move_left(sp.row, sp.column);
                            if let Some(new_sp) = self.stock_market.share_price_at(nr, nc) {
                                self.corporations[corp_idx].share_price = Some(new_sp);
                                self.update_market_cell(&corp_sym, sp.row, sp.column, nr, nc);
                            }
                        }
                        true
                    } else if self.corporations[corp_idx].corp_type
                        == crate::title::CorpType::Minor
                    {
                        // 1867 minors have NO dividend choice (G1867 Dividend
                        // step: actions [] + skip! auto-payout): revenue/2 to
                        // the treasury, the rest to the owner, base price
                        // movement (right on payout).
                        let corp_half = revenue / 2;
                        let owner_half = revenue - corp_half;
                        self.corporations[corp_idx].cash += corp_half;
                        self.bank.cash -= corp_half;
                        if let Some(pid) = self.corporations[corp_idx].owner_id.player_id() {
                            if let Some(pi) = self.player_index(pid) {
                                self.players[pi].cash += owner_half;
                                self.bank.cash -= owner_half;
                            }
                        }
                        if let Some(sp) = self.corporations[corp_idx].share_price.clone() {
                            let (nr, nc) = self.stock_market.move_right(sp.row, sp.column);
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
                        let train_limit = self.corp_train_limit(corp_idx);
                        self.corporations[corp_idx].trains.len() <= train_limit
                    }
                }
                StepKind::BuyCompany => {
                    // phase.status gate: skip outright unless the phase
                    // allows company purchases (1830: phases 3-4). In later
                    // phases all companies are closed, so the else-branch
                    // checks below would skip anyway.
                    if !self.phase_has_status("can_buy_companies") {
                        true
                    } else {
                        let corp_cash = self.corporations[corp_idx].cash;
                        let title = self.title_def();
                        let can_buy_company = self.companies.iter().any(|c| {
                            !c.closed
                                && !c.no_buy
                                && c.owner.is_player()
                                && corp_cash >= title.company_buy_price_range(c.value).0
                        });
                        // A corp holding a company with an unused bonus
                        // tile_lay ability (CS) keeps the step blocking; a
                        // teleport ability (DH) doesn't (it's filtered by
                        // Python's abilities() timing check).
                        let has_ability = self.corp_has_unused_lay_ability(&corp_sym);
                        !can_buy_company && !has_ability
                    }
                }
                StepKind::BuyCompanyPreloan => {
                    // Base BuyCompany actionability at the pre-LoanOperations
                    // window (1867). Ruby auto-passes loan-free corps but
                    // that pass is server-generated INTO the record — replay
                    // blocks here either way and consumes it.
                    if !self.phase_has_status("can_buy_companies") {
                        true
                    } else {
                        let corp_cash = self.corporations[corp_idx].cash;
                        let title = self.title_def();
                        let can_buy_company = self.companies.iter().any(|c| {
                            !c.closed
                                && !c.no_buy
                                && c.owner.is_player()
                                && corp_cash >= title.company_buy_price_range(c.value).0
                        });
                        !can_buy_company
                    }
                }
                StepKind::LoanOperations => {
                    // The skip IS the mechanic: pay interest on the OR-start
                    // snapshot, then forced repayment, then re-snapshot.
                    let loans_at_start = match &self.round {
                        crate::rounds::Round::Operating(s) => {
                            s.interest_snapshot.get(&corp_sym).copied().unwrap_or(0)
                        }
                        _ => 0,
                    };
                    match self.loan_operations_auto(corp_idx, loans_at_start) {
                        Some(new_count) => {
                            if let crate::rounds::Round::Operating(ref mut s) = self.round {
                                s.interest_snapshot.insert(corp_sym.clone(), new_count);
                            }
                            true
                        }
                        None => {
                            // Unpayable interest NATIONALIZED the corp mid-
                            // turn (Ruby interest_unpaid! → nationalize! →
                            // force_next_entity!). Its turn is over: park
                            // the pc at Done and stop skipping — the
                            // caller's advance machinery moves to the next
                            // living corp.
                            if let crate::rounds::Round::Operating(ref mut s) = self.round {
                                s.step = OperatingStep::Done;
                            }
                            false
                        }
                    }
                }
                StepKind::BuyTrain => {
                    // A corp must buy a train only if it has no trains AND has a
                    // legal revenue route. No legal route = no obligation to own a
                    // train, so BuyTrain is optional (and skippable if unaffordable).
                    // NOTE: OPTIONAL buys use plain cash even for loans
                    // titles (Ruby 1867 buy_train.rb buying_power: full only
                    // when must_buy — loans only back the obligation).
                    let corp_cash = self.corporations[corp_idx].cash;
                    let has_trains = !self.corporations[corp_idx].trains.is_empty();

                    // The shared obligation predicate (steps.rs): Python's
                    // must_buy_train + the loans-EMR affordability arm (Ruby
                    // G1867 must_buy_train?, needed_cash = min_depot_price).
                    let must_buy = self.operating_must_buy_train(&corp_sym);

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

                        // Can exchange for a discounted train (1830: the D,
                        // available_on="6", trades in a 4/5/6 for 300 off)?
                        // Python's `discountable_trains_for` looks at
                        // `depot.depot_trains()` — the VISIBLE upcoming trains
                        // PLUS the discarded pool — not just the upcoming
                        // queue. Once the last upcoming D is bought the queue
                        // empties, but discarded D-trains remain exchangeable,
                        // so the BuyTrain step must stay blocking (the corp
                        // can still exchange an owned 4/5/6 for a discounted
                        // discarded D). Both the discount map and the
                        // availability phase come from the train data.
                        let can_exchange = self
                            .depot
                            .trains
                            .iter()
                            .chain(self.depot.discarded.iter())
                            .filter(|t| !t.discount.is_empty())
                            .any(|dt| {
                                let available = self
                                    .phase_available(dt.available_on.as_deref());
                                available
                                    && self.corporations[corp_idx].trains.iter().any(|owned| {
                                        dt.discount.iter().any(|(name, disc)| {
                                            name == &owned.name && corp_cash >= dt.price - disc
                                        })
                                    })
                            });

                        // Can buy inter-corp? 1830: same president only;
                        // 1867: any corp's trains are buyable (from $1).
                        let cross = self.title_def().train_buy_from_other_players();
                        let pres_id = self.corporations[corp_idx].president_id();
                        let can_buy_inter_corp = corp_cash > 0
                            && self.corporations.iter().any(|other| {
                                other.sym != corp_sym
                                    && !other.trains.is_empty()
                                    && (cross
                                        || (pres_id.is_some()
                                            && other.president_id() == pres_id))
                            });

                        let has_room = self.corporations[corp_idx].trains.len()
                            < self.corp_train_limit(corp_idx);

                        let can_buy_any = (has_room
                            && (can_buy_from_depot || can_buy_from_discard || can_buy_inter_corp))
                            || can_exchange;
                        if !can_buy_any
                            && !has_trains
                            && self.title_def().buy_train_pass_nationalizes()
                        {
                            // Ruby Base#skip! == pass! (step/base.rb:60-63),
                            // and G1867 BuyTrain#pass! nationalizes a
                            // trainless corp: one that can buy NOTHING (no
                            // obligation, nothing affordable — not even a $1
                            // inter-corp train) folds into the CN without
                            // any recorded action. Park the pc at Done
                            // (turn over) exactly like the LoanOperations
                            // unpayable-interest arm.
                            self.nationalize_corporation(&corp_sym);
                            if let crate::rounds::Round::Operating(ref mut s) = self.round {
                                s.step = OperatingStep::Done;
                            }
                            false
                        } else {
                            !can_buy_any
                        }
                    }
                }
                // Steps with no auto-skip hook (HomeToken; future kinds)
                // block when the pc lands on them.
                _ => false,
            };
            should_skip
        }
    }

    /// Ruby G1867 `redeemable_shares` (g_1867/game.rb:484-489): bundles of
    /// the corp's OWN shares held by the market pool (percent > 0), minus
    /// those the treasury cannot afford. Bundles are cumulative over shares
    /// sorted by [president?, percent] (base.rb `all_bundles_for_corporation`),
    /// so a non-empty result == the corp affords its CHEAPEST single pool
    /// cert at market price (Share#price = market price × percent/unit).
    pub(crate) fn corp_can_redeem_share(&self, corp_idx: usize) -> bool {
        let corp = &self.corporations[corp_idx];
        let Some(price) = corp.share_price.as_ref().map(|sp| sp.price) else {
            return false;
        };
        let unit = corp.share_unit_percent.max(1) as i32;
        corp.shares
            .iter()
            .filter(|s| s.owner.is_market() && s.percent > 0)
            .map(|s| price * s.percent as i32 / unit)
            .min()
            .map_or(false, |cheapest| corp.cash >= cheapest)
    }

    /// 1867 RedeemShares (redeem_shares.rb < step/issue_shares.rb): the
    /// operating corp buys a bundle of its OWN shares from the market pool.
    /// Money: corp → BANK (share_pool.rb:104-126: the pool isn't a
    /// corporation or player, so the receiver falls through to the bank) at
    /// market price per share. Shares: pool → treasury (transfer to the
    /// corp; 1867's default `ipo_owner == self`, corporation.rb:39, makes
    /// the corp-owned bucket the IPO/treasury bucket — `EntityId::ipo` here,
    /// same convention as the merger round's treasury dealing). One bundle
    /// per turn: `process_buy_shares` ends with `pass!`.
    fn or_process_redeem_shares(
        &mut self,
        state: &OperatingState,
        entity_id: &str,
        corporation_sym: &str,
        percent: u8,
        share_indices: &[usize],
    ) -> Result<(), GameError> {
        let mut new_state = state.clone();

        if new_state.step != OperatingStep::RedeemShares {
            return Err(GameError::new("Not in Redeem Shares step"));
        }
        let cur = new_state
            .current_corp_sym()
            .ok_or_else(|| GameError::new("No current corp"))?
            .to_string();
        if entity_id != cur {
            return Err(GameError::new(format!(
                "buy_shares entity {} does not match current operator {}",
                entity_id, cur
            )));
        }
        // Ruby redeemable_shares only ever bundles the corp's own shares.
        if corporation_sym != cur {
            return Err(GameError::new(format!(
                "{} may only redeem its own shares, not {}",
                cur, corporation_sym
            )));
        }
        let corp_idx = self.corp_idx[cur.as_str()];

        // Honor the recorded certs when they are redeemable pool shares —
        // keeps per-cert identity aligned with Ruby. Otherwise (defensive:
        // recorded games always carry the certs) rebuild the bundle the way
        // Ruby sorts it: president last, percent ascending, market pool
        // insertion order as the tiebreak.
        let market = crate::entities::EntityId::market();
        let unit = self.corporations[corp_idx].share_unit_percent.max(1) as i32;
        let mut chosen: Vec<usize> = share_indices
            .iter()
            .copied()
            .filter(|&i| {
                self.corporations[corp_idx]
                    .shares
                    .get(i)
                    .map_or(false, |sh| sh.owner == market && sh.percent > 0)
            })
            .collect();
        let chosen_percent: i32 = chosen
            .iter()
            .map(|&i| self.corporations[corp_idx].shares[i].percent as i32)
            .sum();
        if chosen_percent != percent as i32 {
            let mut pool: Vec<usize> = self.corporations[corp_idx]
                .market_order
                .iter()
                .copied()
                .filter(|&i| {
                    self.corporations[corp_idx]
                        .shares
                        .get(i)
                        .map_or(false, |sh| sh.owner == market && sh.percent > 0)
                })
                .collect();
            pool.sort_by_key(|&i| {
                let sh = &self.corporations[corp_idx].shares[i];
                (sh.president, sh.percent)
            });
            chosen.clear();
            let mut acc = 0i32;
            for i in pool {
                if acc >= percent as i32 {
                    break;
                }
                acc += self.corporations[corp_idx].shares[i].percent as i32;
                chosen.push(i);
            }
            if acc != percent as i32 {
                return Err(GameError::new(format!(
                    "{} cannot redeem {}% — the market holds {}%",
                    cur, percent, acc
                )));
            }
        }

        // Bundle price at MARKET price (Share#price_per_share: pool shares
        // use corporation.share_price, share.rb:47-50).
        let market_price = self.corporations[corp_idx]
            .share_price
            .as_ref()
            .map(|sp| sp.price)
            .ok_or_else(|| GameError::new("Corporation has no share price"))?;
        let price: i32 = chosen
            .iter()
            .map(|&i| market_price * self.corporations[corp_idx].shares[i].percent as i32 / unit)
            .sum();
        if self.corporations[corp_idx].cash < price {
            return Err(GameError::new(format!(
                "{} cannot afford to redeem: price {} > cash {}",
                cur, price, self.corporations[corp_idx].cash
            )));
        }

        self.corporations[corp_idx].cash -= price;
        self.bank.cash += price;
        let treasury = crate::entities::EntityId::ipo(&cur);
        for &i in &chosen {
            self.corporations[corp_idx].set_share_owner(i, treasury.clone());
        }

        // Ruby process_buy_shares ends with pass!: one bundle per turn,
        // then the turn moves on toward Track.
        new_state.step =
            crate::steps::next_operating_pc(self.operating_step_descs(), &new_state.step);
        self.round = crate::rounds::Round::Operating(new_state);
        self.update_round_state();
        Ok(())
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
            // Pass from final BuyCompany ends the corp's turn (advance via
            // the dead-corp-skipping wrapper, after installing the state).
            self.round = crate::rounds::Round::Operating(new_state);
            self.or_advance_to_next_corp();
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
            // engine swallows failed passes). The gate uses the SAME predicate
            // as the enumeration/dispatch arms (`operating_must_buy_train` ==
            // Python's `must_buy_train`, round.py:550-551 — graph
            // `route_train_purchase`, not the looser `route_available`; for
            // loans-EMR titles it carries the affordable-with-max-loans arm of
            // Ruby G1867 must_buy_train?, so a corp that cannot raise
            // `min_depot_price` even with max loans MAY pass).
            if new_state.step == OperatingStep::BuyTrain {
                let cur_sym = cur.to_string();
                if self.operating_must_buy_train(&cur_sym) {
                    return Err(GameError::new(
                        "Blocking step Buy Trains cannot process action Pass",
                    ));
                }
                // Ruby G1867::Step::BuyTrain#pass! (step/buy_train.rb:28-31):
                // a corp that passes the buy still trainless is nationalized
                // on the spot. Its turn is over — the closed minor / reset
                // major has no further steps (Ruby's round walks past a
                // closed entity) — so park the pc at Done and let the
                // caller's advance machinery move to the next living corp
                // (same shape as the LoanOperations unpayable-interest arm).
                if self.title_def().buy_train_pass_nationalizes()
                    && self
                        .corp_idx
                        .get(cur_sym.as_str())
                        .map_or(false, |&ci| self.corporations[ci].trains.is_empty())
                {
                    self.nationalize_corporation(&cur_sym);
                    new_state.step = OperatingStep::Done;
                    self.round = crate::rounds::Round::Operating(new_state);
                    self.update_round_state();
                    return Ok(());
                }
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

/// Ruby Hex#distance ("as the crow flies"): with dx = |letter Δ| and
/// dy = |number Δ|, pointy layouts (double-width numbers) give
/// dy + max(0,(dx-dy)/2) and flat layouts (double-height numbers) give
/// dx + max(0,(dy-dx)/2) — the non-doubled axis steps once per row, the
/// doubled axis twice per hex. Drives 1867's distance-priced tokens.
pub(crate) fn hex_crow_distance(a: &str, b: &str, flat: bool) -> Option<i32> {
    fn parse(coord: &str) -> Option<(i32, i32)> {
        let letter_len = coord.chars().take_while(|c| c.is_ascii_alphabetic()).count();
        if letter_len == 0 || letter_len > 2 {
            return None;
        }
        let letters: Vec<char> = coord[..letter_len].chars().collect();
        let x = if letters.len() == 1 {
            letters[0] as i32 - 'A' as i32
        } else {
            26 + (letters[1] as i32 - 'A' as i32)
        };
        let y: i32 = coord[letter_len..].parse().ok()?;
        Some((x, y))
    }
    let (ax, ay) = parse(a)?;
    let (bx, by) = parse(b)?;
    let dx = (ax - bx).abs();
    let dy = (ay - by).abs();
    Some(if flat {
        dx + 0.max((dy - dx) / 2)
    } else {
        dy + 0.max((dx - dy) / 2)
    })
}

#[cfg(test)]
mod hex_distance_tests {
    use super::hex_crow_distance;

    /// Flat-top (1867): adjacency deltas are (0,±2) and (±1,±1) in
    /// (letter, number) space — all six neighbors are distance 1.
    #[test]
    fn flat_neighbors_are_distance_one() {
        for n in ["L10", "L14", "K11", "K13", "M11", "M13"] {
            assert_eq!(hex_crow_distance("L12", n, true), Some(1), "L12->{n}");
        }
    }

    #[test]
    fn flat_straight_line_and_diagonals() {
        // Same letter row: two number steps per hex.
        assert_eq!(hex_crow_distance("L12", "L20", true), Some(4));
        // Pure letter steps.
        assert_eq!(hex_crow_distance("D2", "H2", true), Some(4));
        // Mixed: the 1867 CNR probe case shape (e.g. M9 -> I13).
        assert_eq!(hex_crow_distance("M9", "I13", true), Some(4));
        assert_eq!(hex_crow_distance("A1", "A1", true), Some(0));
    }

    // No pointy-top assertions: that branch is Ruby Hex#distance verbatim
    // and no supported pointy title (1830) prices anything by hex distance.
}

#[cfg(test)]
mod g1867_or_tests {
    use super::*;
    use crate::entities::EntityId;
    use crate::game::BaseGame;
    use std::collections::HashMap;

    fn new_4p_game() -> BaseGame {
        let mut players = HashMap::new();
        players.insert(1, "Alice".to_string());
        players.insert(2, "Bob".to_string());
        players.insert(3, "Carol".to_string());
        players.insert(4, "Dave".to_string());
        BaseGame::build_titled("1867", vec![1, 2, 3, 4], players)
    }

    /// Float `sym` by surgery: priced at market column `col`, all certs to
    /// player 1 (tests move them around afterwards).
    fn rig_floated(game: &mut BaseGame, sym: &str, col: u8) {
        let ci = game.corp_idx[sym];
        let sp = game.stock_market.share_price_at(0, col).unwrap();
        game.corporations[ci].share_price = Some(sp.clone());
        game.corporations[ci].ipo_price = Some(sp.clone());
        game.update_market_cell(sym, 0, col, 0, col);
        game.corporations[ci].floated = true;
        game.corporations[ci].ever_operated = true;
        let n = game.corporations[ci].shares.len();
        for i in 0..n {
            game.corporations[ci].set_share_owner(i, EntityId::player(1));
        }
    }

    /// Park the game in an OR whose only operator is `sym`, at the 1867
    /// turn-start pc (RedeemShares — `first_operating_pc`).
    fn enter_or(game: &mut BaseGame, sym: &str) {
        let mut s = crate::rounds::OperatingState::new(1, 2, vec![sym.to_string()]);
        s.step = crate::steps::first_operating_pc(game.operating_step_descs());
        game.round = crate::rounds::Round::Operating(s);
        game.update_round_state();
    }

    /// 1867 operating turns START at RedeemShares (game.rb:839-856 lists it
    /// before Track); 1830 turns still start at LayTile.
    #[test]
    fn first_pc_is_redeem_for_1867_lay_tile_for_1830() {
        assert_eq!(
            crate::steps::first_operating_pc(crate::title::g1867::operating_steps()),
            OperatingStep::RedeemShares
        );
        assert_eq!(
            crate::steps::first_operating_pc(crate::title::g1830::operating_steps()),
            OperatingStep::LayTile
        );
    }

    /// redeem_shares.rb + share_pool.rb:104-126: the corp buys its own pool
    /// cert at market price, pays the BANK, the share joins the treasury,
    /// and the step passes (pc moves on toward Track).
    #[test]
    fn redeem_buys_pool_cert_at_market_price_then_passes() {
        let mut game = new_4p_game();
        rig_floated(&mut game, "CPR", 10);
        let ci = game.corp_idx["CPR"];
        let price = game.corporations[ci].share_price.as_ref().unwrap().price;
        game.corporations[ci].set_share_owner(3, EntityId::market());
        game.corporations[ci].cash = price + 7;
        enter_or(&mut game, "CPR");
        let bank0 = game.bank.cash;

        game.process_action_internal(&crate::actions::Action::BuyShares {
            entity_id: "CPR".into(),
            corporation_sym: "CPR".into(),
            shares: Vec::new(),
            percent: 10,
            source: "market".into(),
            share_indices: vec![3],
        })
        .unwrap();

        let corp = &game.corporations[game.corp_idx["CPR"]];
        assert_eq!(corp.cash, 7);
        assert_eq!(game.bank.cash, bank0 + price);
        assert_eq!(corp.shares[3].owner, EntityId::ipo("CPR"));
        assert_eq!(corp.market_shares_percent(), 0);
        // pass! — the turn moved past RedeemShares (Track blocks or later).
        if let crate::rounds::Round::Operating(s) = &game.round {
            assert_ne!(s.step, OperatingStep::RedeemShares);
        } else {
            panic!("expected operating round");
        }
    }

    /// actions(entity) gate: with NO affordable pool share the step is
    /// empty and auto-skips (the turn opens at Track); with one, the step
    /// BLOCKS and a recorded pass is consumed by RedeemShares, advancing
    /// the pc without touching cash.
    #[test]
    fn redeem_blocks_only_while_pool_share_affordable() {
        let mut game = new_4p_game();
        rig_floated(&mut game, "CPR", 10);
        let ci = game.corp_idx["CPR"];

        // No pool shares: auto-skip to LayTile.
        game.corporations[ci].cash = 1000;
        enter_or(&mut game, "CPR");
        game.skip_steps();
        if let crate::rounds::Round::Operating(s) = &game.round {
            assert_eq!(s.step, OperatingStep::LayTile);
        }

        // Pool share but unaffordable: still skips.
        let price = game.corporations[ci].share_price.as_ref().unwrap().price;
        game.corporations[ci].set_share_owner(3, EntityId::market());
        game.corporations[ci].cash = price - 1;
        enter_or(&mut game, "CPR");
        game.skip_steps();
        if let crate::rounds::Round::Operating(s) = &game.round {
            assert_eq!(s.step, OperatingStep::LayTile);
        }

        // Affordable: blocks at RedeemShares; the pass is consumed there.
        game.corporations[ci].cash = price;
        enter_or(&mut game, "CPR");
        game.skip_steps();
        if let crate::rounds::Round::Operating(s) = &game.round {
            assert_eq!(s.step, OperatingStep::RedeemShares);
        }
        let cash0 = game.corporations[ci].cash;
        game.process_action_internal(&crate::actions::Action::Pass {
            entity_id: "CPR".into(),
        })
        .unwrap();
        assert_eq!(game.corporations[game.corp_idx["CPR"]].cash, cash0);
        assert_eq!(
            game.corporations[game.corp_idx["CPR"]].market_shares_percent(),
            10
        );
    }

    /// The X3→X6 brown-Montreal merge, pinned against the actual Ruby
    /// engine on nationalization_cash file id 449 (X3 rot 2 → X6 rot 2):
    /// old cities {2,0}/{3,4}/{5,1} map by EXIT SUBSET into X6's
    /// {2,5,0,1}(2 slots)/{3,4}(1 slot) — Ruby city_map_for, hex.rb:258-292
    /// — so city 0 ends [TGB, NO] and city 1 [CPR]. The pre-port slot
    /// logic wrote both city-0 tokens to slot 0 and silently DROPPED one.
    #[test]
    fn x3_to_x6_token_transfer_matches_ruby() {
        let mut game = new_4p_game();
        for sym in ["TGB", "CPR", "NO"] {
            rig_floated(&mut game, sym, 8);
        }
        // Install X3 rot 2 on L12 with TGB/CPR/NO tokens in cities 0/1/2.
        let hi = game.hex_idx["L12"];
        let x3 = game.tile_catalog.get("X3").unwrap().clone();
        game.hexes[hi].tile = BaseGame::tile_from_def(&x3, 2);
        game.hexes[hi].tile.name = "X3-0".into();
        for (city_i, sym) in [(0usize, "TGB"), (1, "CPR"), (2, "NO")] {
            let ci = game.corp_idx[sym];
            let ti = game.corporations[ci].next_token_index().unwrap();
            game.corporations[ci].tokens[ti].used = true;
            game.corporations[ci].tokens[ti].city_hex_id = "L12".into();
            let tok = game.corporations[ci].tokens[ti].clone();
            game.hexes[hi].tile.cities[city_i].tokens[0] = Some(tok);
        }

        enter_or(&mut game, "TGB");
        let state = match &game.round {
            crate::rounds::Round::Operating(s) => {
                let mut s = s.clone();
                s.step = OperatingStep::LayTile;
                s
            }
            _ => unreachable!(),
        };
        game.corporations[game.corp_idx["TGB"]].cash = 100;
        game.or_process_lay_tile(&state, "TGB", "L12", "X6-0", 2)
            .unwrap();

        let tile = &game.hexes[game.hex_idx["L12"]].tile;
        let names = |ci: usize| -> Vec<String> {
            tile.cities[ci]
                .tokens
                .iter()
                .map(|t| t.as_ref().map_or("-".into(), |t| t.corporation_id.clone()))
                .collect()
        };
        assert_eq!(names(0), vec!["TGB".to_string(), "NO".to_string()]);
        assert_eq!(names(1), vec!["CPR".to_string()]);
        // All three corp token records survived onto the new tile.
        for sym in ["TGB", "CPR", "NO"] {
            let ci = game.corp_idx[sym];
            assert!(
                game.corporations[ci]
                    .tokens
                    .iter()
                    .any(|t| t.used && t.city_hex_id == "L12"),
                "{} token lost in the X3→X6 transfer",
                sym
            );
        }
        // Nothing was displaced into pending_tokens.
        if let crate::rounds::Round::Operating(s) = &game.round {
            assert!(s.pending_tokens.is_empty());
        }
    }

    /// Park `sym`'s turn at the BuyTrain pc.
    fn enter_buy_train(game: &mut BaseGame, sym: &str) {
        let mut s = crate::rounds::OperatingState::new(1, 2, vec![sym.to_string()]);
        s.step = OperatingStep::BuyTrain;
        game.round = crate::rounds::Round::Operating(s);
        game.update_round_state();
    }

    /// Ruby G1867 BuyTrain (step/buy_train.rb): a trainless corp whose FULL
    /// buying power (cash + takeable loans × $45) cannot reach
    /// `min_depot_price` is NOT obliged (`must_buy_train?` :41-44) — it may
    /// pass while a $1 inter-corp train keeps the step blocking — and the
    /// pass nationalizes it (`pass!` :28-31). Probed on 20229 file id 211:
    /// minor CA, $105 + 2 loans = $195 < $225 depot 3-train, recorded pass,
    /// CA closed with its token turned into a CN token.
    #[test]
    fn buy_train_pass_without_obligation_nationalizes_trainless_corp() {
        let mut game = new_4p_game();
        rig_floated(&mut game, "BO", 8); // minor: max 2 loans
        rig_floated(&mut game, "CPR", 8);
        let ci = game.corp_idx["BO"];
        // Depot head = 2-train ($100); full power 5 + 2×45 = 95 < 100.
        game.corporations[ci].cash = 5;
        assert!(!game.operating_must_buy_train("BO"));
        // A cross-corp train keeps BuyTrain blocking ($1 buys are legal).
        let mut train = game.depot.trains[0].clone();
        train.owner = EntityId::corporation("CPR");
        let cpr = game.corp_idx["CPR"];
        game.corporations[cpr].trains.push(train);

        enter_buy_train(&mut game, "BO");
        let price0 = game.corporations[ci].share_price.as_ref().unwrap().price;
        let p1_0 = game.players[0].cash;
        game.process_action_internal(&crate::actions::Action::Pass {
            entity_id: "BO".into(),
        })
        .unwrap();

        let ci = game.corp_idx["BO"];
        assert!(game.corporations[ci].closed, "trainless pass must nationalize");
        // Owner paid 2 × the once-left price (minor = two share units).
        assert!(game.players[0].cash > p1_0);
        assert!(game.players[0].cash - p1_0 < 2 * price0);
    }

    /// Ruby Base#skip! == pass! (step/base.rb:60-63): a trainless corp with
    /// NOTHING buyable (no obligation, no affordable depot/discard/inter-corp
    /// train) never blocks at BuyTrain — the auto-skip itself nationalizes
    /// it, with no recorded action.
    #[test]
    fn buy_train_auto_skip_nationalizes_trainless_corp() {
        let mut game = new_4p_game();
        rig_floated(&mut game, "BO", 8);
        let ci = game.corp_idx["BO"];
        game.corporations[ci].cash = 0; // can't even take a $1 inter-corp train

        enter_buy_train(&mut game, "BO");
        game.skip_steps();

        let ci = game.corp_idx["BO"];
        assert!(game.corporations[ci].closed, "skip must nationalize");
        if let crate::rounds::Round::Operating(s) = &game.round {
            assert_eq!(s.step, OperatingStep::Done, "turn parked at Done");
        } else {
            panic!("expected operating round");
        }
    }

    /// A corp with trains that passes BuyTrain is NOT nationalized (the
    /// pass!-hook fires only on `trains.empty?`).
    #[test]
    fn buy_train_pass_with_trains_does_not_nationalize() {
        let mut game = new_4p_game();
        rig_floated(&mut game, "BO", 8);
        rig_floated(&mut game, "CPR", 8);
        let ci = game.corp_idx["BO"];
        game.corporations[ci].cash = 500;
        let mut train = game.depot.trains[0].clone();
        train.owner = EntityId::corporation("BO");
        game.corporations[ci].trains.push(train);
        // Another corp's train keeps the step blocking (optional buy).
        let mut other = game.depot.trains[0].clone();
        other.owner = EntityId::corporation("CPR");
        let cpr = game.corp_idx["CPR"];
        game.corporations[cpr].trains.push(other);

        enter_buy_train(&mut game, "BO");
        game.process_action_internal(&crate::actions::Action::Pass {
            entity_id: "BO".into(),
        })
        .unwrap();
        assert!(!game.corporations[game.corp_idx["BO"]].closed);
    }

    /// `needed_cash` for the obligation is Ruby Depot#min_depot_price
    /// (depot.rb:61-65): the cheapest of the upcoming HEAD, phase-available
    /// later upcoming trains and the discarded pool — not the head alone.
    /// Probed on 20289 file id 654: depot [8 @1000, 2+2 @600] in phase 8,
    /// CNR $519 + 5 loans = $744 ≥ 600 IS obliged (and loan-funds the 2+2);
    /// the old head-only check read $1000 and wrongly let it pass.
    #[test]
    fn must_buy_obligation_uses_ruby_min_depot_price() {
        let mut game = new_4p_game();
        rig_floated(&mut game, "BO", 8); // minor: full power = cash + 2×45
        let ci = game.corp_idx["BO"];
        game.corporations[ci].cash = 200; // full = 290

        // Price the head (every 2-train: the visible-upcoming filter matches
        // by NAME) out of reach; a discarded train IS reachable.
        let mut cheap = game.depot.trains[0].clone();
        for t in game.depot.trains.iter_mut() {
            if t.name == "2" {
                t.price = 1000;
            }
        }
        cheap.price = 80;
        game.depot.discarded.push(cheap);
        assert!(
            game.operating_must_buy_train("BO"),
            "min_depot_price = $80 via the discard pool; full power 290 covers it"
        );

        // Discard also out of reach → the obligation lapses (may pass).
        game.depot.discarded[0].price = 400;
        assert!(!game.operating_must_buy_train("BO"));
    }
}
