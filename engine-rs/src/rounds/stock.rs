//! Stock round logic for 1830.
//!
//! Players take turns buying and selling shares. 1830 uses sell_buy_sell order,
//! meaning there's no restriction on interleaving buys and sells.
//! A player can sell multiple bundles, buy one bundle (IPO or market),
//! par one corporation, or pass to end their turn.
//! The round ends when all players pass without acting.

use crate::actions::{Action, GameError};
use crate::entities::EntityId;
use crate::game::BaseGame;
use crate::rounds::StockState;

impl BaseGame {
    /// Process an action during the stock round.
    pub fn process_stock_action(&mut self, action: &Action) -> Result<(), GameError> {
        let state = match &self.round {
            crate::rounds::Round::Stock(s) => s.clone(),
            _ => return Err(GameError::new("Not in stock round")),
        };

        match action {
            Action::Par {
                entity_id,
                corporation_sym,
                share_price,
            } => self.stock_process_par(&state, entity_id, corporation_sym, *share_price),
            Action::BuyShares {
                entity_id,
                corporation_sym,
                percent,
                source,
                share_indices,
                ..
            } => self.stock_process_buy_shares(
                &state,
                entity_id,
                corporation_sym,
                *percent,
                source,
                share_indices,
            ),
            Action::SellShares {
                entity_id,
                corporation_sym,
                percent,
                share_indices,
                ..
            } => self.stock_process_sell_shares(
                &state,
                entity_id,
                corporation_sym,
                *percent,
                share_indices,
            ),
            Action::BuyCompany {
                entity_id,
                company_sym,
                price,
            } => self.stock_process_buy_company(&state, entity_id, company_sym, *price),
            Action::Pass { entity_id } => self.stock_process_pass(&state, entity_id),
            _ => Err(GameError::new(format!(
                "Invalid action in stock round: {}",
                action.action_type()
            ))),
        }
    }

    fn stock_process_par(
        &mut self,
        state: &StockState,
        entity_id: &str,
        corporation_sym: &str,
        share_price: i32,
    ) -> Result<(), GameError> {
        let player_id = self.validate_stock_turn(state, entity_id)?;
        let mut new_state = state.clone();

        if new_state.bought_this_turn {
            return Err(GameError::new("Already bought this turn"));
        }

        // Validate the par price
        let par_sp = self
            .stock_market
            .par_price(share_price)
            .ok_or_else(|| GameError::new(format!("Invalid par price: {}", share_price)))?;

        let corp_idx = *self
            .corp_idx
            .get(corporation_sym)
            .ok_or_else(|| GameError::new(format!("Unknown corporation: {}", corporation_sym)))?;

        if self.corporations[corp_idx].ipo_price.is_some() {
            return Err(GameError::new(format!(
                "{} has already been parred",
                corporation_sym
            )));
        }

        // Player must afford 2x par price (president share is 20%)
        let cost = share_price * 2;
        let player_idx = self.player_index(player_id).unwrap();
        if self.players[player_idx].cash < cost {
            return Err(GameError::new(format!(
                "Cannot afford par cost {} (has {})",
                cost, self.players[player_idx].cash
            )));
        }

        // Check cert limit before buying
        let current_certs = self.num_certs_internal(player_id);
        if current_certs >= self.cert_limit as u32 {
            return Err(GameError::new("At certificate limit"));
        }

        // Set IPO and share price
        self.corporations[corp_idx].ipo_price = Some(par_sp.clone());
        self.corporations[corp_idx].share_price = Some(par_sp.clone());
        self.update_market_cell(corporation_sym, 0, 0, par_sp.row, par_sp.column);

        // Initialize all unowned shares as IPO
        let ipo_eid = EntityId::ipo(corporation_sym);
        for share in &mut self.corporations[corp_idx].shares {
            if share.owner.is_none() {
                share.owner = ipo_eid.clone();
            }
        }

        // Transfer president share to player
        let player_eid = EntityId::player(player_id);
        self.corporations[corp_idx].shares[0].owner = player_eid.clone();

        // Player pays
        self.players[player_idx].cash -= cost;
        self.bank.cash += cost;

        // Set president
        self.corporations[corp_idx].owner_id = player_eid;

        // Check float
        self.check_float(corp_idx);

        // Update state
        new_state.bought_this_turn = true;
        new_state.parred_this_turn = true;
        new_state.acted_this_turn = true;
        new_state.consecutive_passes = 0;

        // Set priority deal: next player after this one
        new_state.priority_deal_player = self.next_player_id(player_id);

        self.round = crate::rounds::Round::Stock(new_state);
        self.update_round_state();
        self.stock_after_process();
        Ok(())
    }

    fn stock_process_buy_shares(
        &mut self,
        state: &StockState,
        entity_id: &str,
        corporation_sym: &str,
        percent: u8,
        source: &str,
        share_indices: &[usize],
    ) -> Result<(), GameError> {
        let player_id = self.validate_stock_turn(state, entity_id)?;
        let mut new_state = state.clone();

        // Check if this buy is allowed (one buy per turn, unless buy_multiple applies)
        if new_state.bought_this_turn {
            // Check can_buy_multiple: allowed if the corp's share price has "multiple_buy" type,
            // no par was done this turn, and prior buys were of the same corp.
            let corp_idx_check = self.corp_idx.get(corporation_sym).copied();
            let can_buy_multiple = if let Some(ci) = corp_idx_check {
                let corp = &self.corporations[ci];
                let has_multiple_buy = corp
                    .share_price
                    .as_ref()
                    .map_or(false, |sp| sp.types.iter().any(|t| t == "multiple_buy"));
                has_multiple_buy
                    && !new_state.parred_this_turn
                    && new_state
                        .bought_corp_this_turn
                        .as_ref()
                        .map_or(true, |c| c == corporation_sym)
            } else {
                false
            };

            if !can_buy_multiple {
                return Err(GameError::new("Already bought a share this turn"));
            }
        }

        // Can't buy a corp you sold this round
        if new_state.sold_corp_this_round(player_id, corporation_sym) {
            return Err(GameError::new(format!(
                "Cannot buy {} — sold it this round",
                corporation_sym
            )));
        }

        // Check cert limit
        let current_certs = self.num_certs_internal(player_id);
        if current_certs >= self.cert_limit as u32 {
            return Err(GameError::new("At certificate limit"));
        }

        let corp_idx = *self
            .corp_idx
            .get(corporation_sym)
            .ok_or_else(|| GameError::new(format!("Unknown corporation: {}", corporation_sym)))?;

        let ipo_eid = EntityId::ipo(corporation_sym);
        let market_eid = EntityId::market();

        // Determine the source. When share_indices are provided, check the PYTHON
        // state of that share to infer the intended source (IPO vs market).
        // Since specific share positions may diverge between Python and Rust,
        // we use the inferred source to find ANY available share from that source.
        let inferred_source = if !share_indices.is_empty() {
            let idx = share_indices[0];
            if idx < self.corporations[corp_idx].shares.len() {
                let share = &self.corporations[corp_idx].shares[idx];
                if share.owner == ipo_eid {
                    Some("ipo")
                } else if share.owner == market_eid {
                    Some("market")
                } else {
                    // Share at this index is owned by someone - fall back to auto
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        let effective_source = inferred_source.unwrap_or(source);

        let (share_idx, actual_source) = if effective_source == "ipo" {
            self.corporations[corp_idx]
                .shares
                .iter()
                .position(|s| s.owner == ipo_eid && !s.president)
                .map(|i| (i, "ipo"))
                .ok_or_else(|| GameError::new("No IPO shares available"))?
        } else if source == "market" {
            self.corporations[corp_idx]
                .shares
                .iter()
                .position(|s| s.owner == market_eid && !s.president)
                .map(|i| (i, "market"))
                .ok_or_else(|| GameError::new("No market shares available"))?
        } else {
            // "auto": try market first (uses market price), then IPO (uses par price)
            self.corporations[corp_idx]
                .shares
                .iter()
                .position(|s| s.owner == market_eid && !s.president)
                .map(|i| (i, "market"))
                .or_else(|| {
                    self.corporations[corp_idx]
                        .shares
                        .iter()
                        .position(|s| s.owner == ipo_eid && !s.president)
                        .map(|i| (i, "ipo"))
                })
                .ok_or_else(|| GameError::new("No shares available"))?
        };

        // Enforce multiple_buy_only_from_market: if this is a second buy (multiple),
        // the source must be market, not IPO.
        if new_state.bought_this_turn && actual_source == "ipo" {
            return Err(GameError::new(
                "Cannot buy from IPO during multiple buy — must buy from market",
            ));
        }

        // IPO shares are sold at par price, market shares at current market price
        let unit_price = if actual_source == "ipo" {
            self.corporations[corp_idx]
                .ipo_price
                .as_ref()
                .ok_or_else(|| GameError::new(format!("{} has not been parred", corporation_sym)))?
                .price
        } else {
            self.corporations[corp_idx]
                .share_price
                .as_ref()
                .ok_or_else(|| GameError::new(format!("{} has not been parred", corporation_sym)))?
                .price
        };

        let price = (percent as i32 * unit_price) / 10;
        let player_idx = self.player_index(player_id).unwrap();

        if self.players[player_idx].cash < price {
            return Err(GameError::new(format!(
                "Cannot afford {} (has {})",
                price, self.players[player_idx].cash
            )));
        }

        // Check ownership limit (60% normally, unlimited in multiple_buy/unlimited zones)
        let player_eid = EntityId::player(player_id);
        let current_pct = self.corporations[corp_idx].percent_owned_by(&player_eid);
        let in_unlimited_zone =
            self.corporations[corp_idx]
                .share_price
                .as_ref()
                .map_or(false, |sp| {
                    sp.types
                        .iter()
                        .any(|t| t == "multiple_buy" || t == "unlimited")
                });
        if !in_unlimited_zone && current_pct + percent > 60 {
            return Err(GameError::new(format!(
                "Would exceed 60% ownership of {}",
                corporation_sym
            )));
        }

        // Transfer
        self.corporations[corp_idx].shares[share_idx].owner = player_eid;
        self.players[player_idx].cash -= price;
        self.bank.cash += price;

        // Check float
        self.check_float(corp_idx);

        // Check president change
        self.check_president_change(corp_idx);

        new_state.bought_this_turn = true;
        new_state.bought_corp_this_turn = Some(corporation_sym.to_string());
        if actual_source == "ipo" {
            new_state.bought_from_ipo = true;
        }
        new_state.acted_this_turn = true;
        new_state.consecutive_passes = 0;
        new_state.priority_deal_player = self.next_player_id(player_id);

        self.round = crate::rounds::Round::Stock(new_state);
        self.update_round_state();
        self.stock_after_process();
        Ok(())
    }

    fn stock_process_sell_shares(
        &mut self,
        state: &StockState,
        entity_id: &str,
        corporation_sym: &str,
        percent: u8,
        share_indices: &[usize],
    ) -> Result<(), GameError> {
        let player_id = self.validate_stock_turn(state, entity_id)?;
        let mut new_state = state.clone();

        let corp_idx = *self
            .corp_idx
            .get(corporation_sym)
            .ok_or_else(|| GameError::new(format!("Unknown corporation: {}", corporation_sym)))?;

        let share_price = self.corporations[corp_idx]
            .share_price
            .as_ref()
            .ok_or_else(|| GameError::new(format!("{} has not been parred", corporation_sym)))?
            .clone();

        let player_eid = EntityId::player(player_id);
        let market_eid = EntityId::market();
        let player_idx = self.player_index(player_id).unwrap();

        // Determine if the president share is included
        let player_total = self.corporations[corp_idx].percent_owned_by(&player_eid);
        let remaining_after = player_total.saturating_sub(percent);
        let includes_president = self.corporations[corp_idx]
            .shares
            .iter()
            .any(|s| s.president && s.owner == player_eid)
            && remaining_after < 20;

        // Transfer shares to market.
        // Note: share_indices are NOT used for sells because specific share positions
        // may diverge between Python and Rust. We use the owner-based logic instead.
        {
            let mut transferred_pct = 0u8;
            if includes_president {
                let non_pres_to_sell = percent.saturating_sub(10);
                for share in &mut self.corporations[corp_idx].shares {
                    if transferred_pct >= non_pres_to_sell {
                        break;
                    }
                    if share.owner == player_eid && !share.president {
                        share.owner = market_eid.clone();
                        transferred_pct += share.percent;
                    }
                }
                for share in &mut self.corporations[corp_idx].shares {
                    if share.owner == player_eid && share.president {
                        share.owner = market_eid.clone();
                        break;
                    }
                }
            } else {
                for share in &mut self.corporations[corp_idx].shares {
                    if transferred_pct >= percent {
                        break;
                    }
                    if share.owner == player_eid && !share.president {
                        share.owner = market_eid.clone();
                        transferred_pct += share.percent;
                    }
                }
            }
        }

        // Player receives money based on the percent value
        let revenue = (percent as i32 * share_price.price) / 10;
        self.players[player_idx].cash += revenue;
        self.bank.cash -= revenue;

        // Share price drops: move DOWN once per 10% share sold (1830 SELL_MOVEMENT = "down_share")
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

        // Record sold corp for this round
        new_state.record_sell(player_id, corporation_sym);
        new_state.acted_this_turn = true;
        new_state.consecutive_passes = 0;

        // Update priority deal: next player after the seller
        new_state.priority_deal_player = self.next_player_id(player_id);

        // Check president change — pass the seller as previous president
        // so the tiebreaker works correctly when the president share is in the market.
        self.check_president_change_with_prev(corp_idx, Some(player_id));

        // Handle partial bundles: when selling with president share, the face value
        // of certs transferred exceeds the bundle percent. Move excess shares back.
        if includes_president {
            let player_eid_check = EntityId::player(player_id);
            let actual_pct = self.corporations[corp_idx].percent_owned_by(&player_eid_check);
            let target_pct = remaining_after;
            if actual_pct < target_pct {
                let deficit = target_pct - actual_pct;
                let shares_to_return = deficit / 10;
                let market_eid_check = EntityId::market();
                let mut returned = 0u8;
                for share in &mut self.corporations[corp_idx].shares {
                    if returned >= shares_to_return {
                        break;
                    }
                    if share.owner == market_eid_check && !share.president {
                        share.owner = player_eid_check.clone();
                        returned += 1;
                    }
                }
            }
        }

        self.round = crate::rounds::Round::Stock(new_state);
        self.update_round_state();
        self.stock_after_process();
        Ok(())
    }

    fn stock_process_buy_company(
        &mut self,
        state: &StockState,
        entity_id: &str,
        company_sym: &str,
        price: i32,
    ) -> Result<(), GameError> {
        let player_id = self.validate_stock_turn(state, entity_id)?;
        let mut new_state = state.clone();

        let company_idx = *self
            .company_idx
            .get(company_sym)
            .ok_or_else(|| GameError::new(format!("Unknown company: {}", company_sym)))?;

        let seller_id = self.companies[company_idx]
            .owner
            .player_id()
            .ok_or_else(|| GameError::new("Company not owned by a player"))?;

        let player_idx = self.player_index(player_id).unwrap();
        let seller_idx = self.player_index(seller_id).unwrap();

        if self.players[player_idx].cash < price {
            return Err(GameError::new("Cannot afford company"));
        }

        self.players[player_idx].cash -= price;
        self.players[seller_idx].cash += price;
        self.companies[company_idx].owner = EntityId::player(player_id);

        new_state.acted_this_turn = true;
        new_state.consecutive_passes = 0;
        new_state.priority_deal_player = self.next_player_id(player_id);

        self.round = crate::rounds::Round::Stock(new_state);
        self.update_round_state();
        self.stock_after_process();
        Ok(())
    }

    fn stock_process_pass(&mut self, state: &StockState, entity_id: &str) -> Result<(), GameError> {
        let _player_id = self.validate_stock_turn(state, entity_id)?;
        let mut new_state = state.clone();

        // Mirrors Python BuySellParShares.pass_():
        // If the player took actions this turn (current_actions non-empty), unpass them.
        // If they didn't take any actions (pure pass), mark as passed.
        if new_state.acted_this_turn {
            new_state.unpass_current_player();
            new_state.consecutive_passes = 0;
        } else {
            new_state.mark_current_player_passed();
            new_state.consecutive_passes += 1;
        }

        self.round = crate::rounds::Round::Stock(new_state);
        self.update_round_state();

        // After the pass, advance to next entity (like after_process → next_entity)
        self.stock_next_entity();

        Ok(())
    }

    // -- Stock round helpers --

    fn validate_stock_turn(&self, state: &StockState, entity_id: &str) -> Result<u32, GameError> {
        let player_id: u32 = entity_id
            .parse()
            .map_err(|_| GameError::new(format!("Invalid player id: {}", entity_id)))?;

        if player_id != state.current_player_id() {
            return Err(GameError::new(format!(
                "Not player {}'s turn, expected {}",
                player_id,
                state.current_player_id()
            )));
        }

        Ok(player_id)
    }

    // stock_advance_to_player removed — auto-skip logic in stock_after_process
    // and stock_start_entity should handle all player advancement correctly.

    /// Internal cert counting (used by stock round logic).
    /// In 1830, shares of corps in the Yellow zone (no_cert_limit) don't count.
    pub(crate) fn num_certs_internal(&self, player_id: u32) -> u32 {
        let player_eid = EntityId::player(player_id);
        let share_certs: u32 = self
            .corporations
            .iter()
            .filter(|c| {
                // Exclude corps whose share price is in a no_cert_limit zone (Yellow)
                !c.share_price
                    .as_ref()
                    .is_some_and(|sp| sp.types.iter().any(|t| t == "no_cert_limit"))
            })
            .flat_map(|c| c.shares.iter())
            .filter(|s| s.owner == player_eid)
            .count() as u32;
        // CERT_LIMIT_INCLUDES_PRIVATES = true in 1830
        let company_certs: u32 = self
            .companies
            .iter()
            .filter(|c| c.owner == player_eid && !c.closed)
            .count() as u32;
        share_certs + company_certs
    }

    /// Check if a corporation should float (60%+ sold from IPO).
    pub(crate) fn check_float(&mut self, corp_idx: usize) {
        let corp = &self.corporations[corp_idx];
        if !corp.floated && corp.check_floated() {
            // Float: corporation receives par_price * total_shares (10) from bank
            if let Some(ref ipo_price) = corp.ipo_price {
                let treasury = ipo_price.price * 10; // 10 total shares
                self.corporations[corp_idx].cash = treasury;
                self.bank.cash -= treasury;
                self.corporations[corp_idx].floated = true;
                // Home token is placed at the start of the corp's operating turn,
                // not here during the stock round.
            }
        }
    }

    /// Place the corporation's first token on its home city.
    pub(crate) fn place_home_token(&mut self, corp_idx: usize) {
        let corp_defs = crate::title::g1830::corporations();
        let corp_sym = &self.corporations[corp_idx].sym;
        let corp_def = corp_defs.iter().find(|cd| cd.sym == corp_sym);

        if let Some(cd) = corp_def {
            let home_hex = cd.home_hex.to_string();
            let home_city_idx = cd.home_city_index as usize;

            if let Some(hex_idx) = self.hex_idx.get(&home_hex) {
                let hex = &mut self.hexes[*hex_idx];
                if let Some(city) = hex.tile.cities.get_mut(home_city_idx) {
                    for token_slot in &mut city.tokens {
                        if token_slot.is_none() {
                            let mut token = self.corporations[corp_idx].tokens[0].clone();
                            token.used = true;
                            token.city_hex_id = home_hex.clone();
                            *token_slot = Some(token);
                            self.corporations[corp_idx].tokens[0].used = true;
                            self.corporations[corp_idx].tokens[0].city_hex_id = home_hex;
                            break;
                        }
                    }
                }
            }
        }
    }

    /// Check if president should change.
    /// In 1830, the current president keeps the certificate on ties —
    /// a new president is only assigned when another player has STRICTLY more shares.
    pub(crate) fn check_president_change(&mut self, corp_idx: usize) {
        self.check_president_change_with_prev(corp_idx, None);
    }

    pub(crate) fn check_president_change_with_prev(&mut self, corp_idx: usize, previous_president: Option<u32>) {
        let corp = &self.corporations[corp_idx];
        let current_president = corp.president_id().or(previous_president);

        // Find the player with the most shares. If tied, the current president
        // keeps it. If the current president is no longer a candidate (sold),
        // the tie goes to the player closest clockwise to the previous president.
        let mut max_percent = 0u8;
        for player in &self.players {
            let eid = EntityId::player(player.id);
            let pct = corp.percent_owned_by(&eid);
            if pct > max_percent {
                max_percent = pct;
            }
        }

        if max_percent < 20 {
            return;
        }

        // Collect all players at max_percent
        let candidates: Vec<u32> = self
            .players
            .iter()
            .filter(|p| {
                let eid = EntityId::player(p.id);
                corp.percent_owned_by(&eid) == max_percent
            })
            .map(|p| p.id)
            .collect();

        if candidates.is_empty() {
            return;
        }

        // If the current president is among the candidates, they keep it
        if let Some(pres_id) = current_president {
            if candidates.contains(&pres_id) {
                return;
            }
        }

        // Tiebreaker: closest clockwise from the previous president.
        // Find the previous president's position in player order.
        let max_player = if candidates.len() == 1 {
            candidates[0]
        } else if let Some(pres_id) = current_president {
            let pres_pos = self
                .player_order
                .iter()
                .position(|&id| id == pres_id)
                .unwrap_or(0);
            let n = self.player_order.len();
            // Walk clockwise from pres_pos+1, pick first candidate
            let mut winner = candidates[0];
            for offset in 1..=n {
                let check_id = self.player_order[(pres_pos + offset) % n];
                if candidates.contains(&check_id) {
                    winner = check_id;
                    break;
                }
            }
            winner
        } else {
            candidates[0]
        };

        let new_pres = max_player;

        // If president share is in the market (previous president sold it),
        // transfer it to the new president and move 2 normal shares to market.
        let pres_share_in_market = self.corporations[corp_idx]
            .shares
            .iter()
            .any(|s| s.president && s.owner.is_market());

        if pres_share_in_market {
            let new_pres_eid = EntityId::player(new_pres);
            let market_eid = EntityId::market();

            // Give president share to new president
            if let Some(pres_idx) = self.corporations[corp_idx].president_share_index() {
                self.corporations[corp_idx].shares[pres_idx].owner = new_pres_eid.clone();
            }

            // Move 2 normal shares from new president to market
            let mut swapped = 0;
            for share in &mut self.corporations[corp_idx].shares {
                if swapped >= 2 {
                    break;
                }
                if share.owner == new_pres_eid && !share.president {
                    share.owner = market_eid.clone();
                    swapped += 1;
                }
            }

            self.corporations[corp_idx].owner_id = new_pres_eid;
            return;
        }

        // Normal president change: president share is held by a player
        if let Some(old_pres) = current_president {
            if new_pres != old_pres {
                let new_pres_eid = EntityId::player(new_pres);
                let old_pres_eid = EntityId::player(old_pres);

                // Give president share to new president
                if let Some(pres_idx) = self.corporations[corp_idx].president_share_index() {
                    self.corporations[corp_idx].shares[pres_idx].owner = new_pres_eid.clone();
                }

                // Swap 2 normal shares from new president to old president
                let mut swapped = 0;
                for share in &mut self.corporations[corp_idx].shares {
                    if swapped >= 2 {
                        break;
                    }
                    if share.owner == new_pres_eid && !share.president {
                        share.owner = old_pres_eid.clone();
                        swapped += 1;
                    }
                }

                self.corporations[corp_idx].owner_id = EntityId::player(new_pres);
            }
        }
    }

    /// After stock round: corps with no shares in IPO or market get price increase.
    pub(crate) fn check_sold_out_price_increases(&mut self) {
        for corp_idx in 0..self.corporations.len() {
            let corp = &self.corporations[corp_idx];
            if !corp.floated {
                continue;
            }
            let ipo_pct = corp.ipo_shares_percent();
            let market_pct = corp.market_shares_percent();
            if ipo_pct == 0 && market_pct == 0 {
                if let Some(ref sp) = corp.share_price.clone() {
                    // 1830: sold_out_stock_movement is move_up
                    let (new_row, new_col) = self.stock_market.move_up(sp.row, sp.column);
                    if let Some(new_sp) = self.stock_market.share_price_at(new_row, new_col) {
                        let sym = self.corporations[corp_idx].sym.clone();
                        self.corporations[corp_idx].share_price = Some(new_sp);
                        self.update_market_cell(&sym, sp.row, sp.column, new_row, new_col);
                    }
                }
            }
        }
    }

    /// Get the next player ID in order.
    pub(crate) fn next_player_id(&self, player_id: u32) -> u32 {
        let idx = self
            .players
            .iter()
            .position(|p| p.id == player_id)
            .unwrap_or(0);
        let next_idx = (idx + 1) % self.players.len();
        self.players[next_idx].id
    }

    /// Stock round after_process: called after every stock action.
    /// Mirrors Python's Stock.after_process → next_entity → start_entity chain.
    /// If the current player has no more valid actions, advance to the next player.
    /// If that player also can't act, skip them (marking as passed). Continue until
    /// a player CAN act or all players are passed (round finishes).
    pub(crate) fn stock_after_process(&mut self) {
        // Check if the current step is still "blocking" (player has valid actions).
        // In Python, after any action, `actions(player)` is rechecked. If it returns
        // non-empty (player can sell, buy, par, or at least pass-with-actions), the
        // step is blocking and the player must explicitly pass.
        // If it returns empty (player has no valid actions at all), the step is
        // non-blocking and the player is auto-advanced.
        let still_blocking = {
            let (player_id, acted, bought) = match &self.round {
                crate::rounds::Round::Stock(s) => {
                    if s.finished {
                        return;
                    }
                    (s.current_player_id(), s.acted_this_turn, s.bought_this_turn)
                }
                _ => return,
            };
            self.player_has_actions(player_id, acted, bought)
        };

        if still_blocking {
            return;
        }

        // Player's turn is over — process their pass.
        // If they acted (current_actions non-empty), unpass them.
        // If they didn't act, mark them as passed.
        if let crate::rounds::Round::Stock(ref mut s) = self.round {
            if s.acted_this_turn {
                // Player acted then auto-passed: unpass (Python: entity.unpass())
                s.unpass_current_player();
                s.consecutive_passes = 0;
            } else {
                // Player had no actions: mark as passed (Python: entity.pass_())
                s.mark_current_player_passed();
                s.consecutive_passes += 1;
            }
        }

        // next_entity: check if round is finished, else start next entity
        self.stock_next_entity();
    }

    /// Advance to the next entity in the stock round.
    /// If all players are passed, finish the round.
    /// Otherwise, start the next player's turn and auto-skip if they can't act.
    fn stock_next_entity(&mut self) {
        // Check if round is finished (all players passed)
        let all_passed = match &self.round {
            crate::rounds::Round::Stock(s) => s.all_players_passed(),
            _ => return,
        };

        if all_passed {
            self.check_sold_out_price_increases();
            if let crate::rounds::Round::Stock(ref mut s) = self.round {
                s.finished = true;
            }
            return;
        }

        // Advance to next player
        if let crate::rounds::Round::Stock(ref mut s) = self.round {
            s.advance_to_next_player();
        }

        // start_entity: reset step state, check if player can act
        self.stock_start_entity();
    }

    /// Start a new player's turn in the stock round.
    /// Resets per-turn state and checks if the player has valid actions.
    /// If not, marks them as passed and recurses to next_entity.
    pub(crate) fn stock_start_entity(&mut self) {
        // Reset step state for new player (Python: step.unpass(), step.setup())
        // bought_this_turn and acted_this_turn are already reset by advance_to_next_player

        // Check if this player has any valid actions
        let (player_id, acted, bought) = match &self.round {
            crate::rounds::Round::Stock(s) => {
                if s.finished {
                    return;
                }
                (s.current_player_id(), s.acted_this_turn, s.bought_this_turn)
            }
            _ => return,
        };

        let has_actions = self.player_has_actions(player_id, acted, bought);

        if has_actions {
            return; // Player can act, wait for their action
        }

        // Player has no valid actions — auto-skip (Python: step.skip() → step.pass_())
        // Since current_actions is empty, mark player as passed
        if let crate::rounds::Round::Stock(ref mut s) = self.round {
            s.mark_current_player_passed();
            s.consecutive_passes += 1;
        }

        // Recurse: try next entity
        self.stock_next_entity();
    }

    /// Check whether a player has any valid actions in the stock round.
    /// Returns true if they can buy, sell, or par (i.e., the step is "blocking").
    /// In Python, after acting, the step is blocking if `actions()` returns non-empty.
    /// `actions()` includes sell options (if allowed) even after buying.
    fn player_has_actions(&self, player_id: u32, _acted: bool, bought: bool) -> bool {
        let player_idx = match self.player_index(player_id) {
            Some(idx) => idx,
            None => return false,
        };
        let player_cash = self.players[player_idx].cash;
        let player_eid = EntityId::player(player_id);
        let current_certs = self.num_certs_internal(player_id);
        let at_cert_limit = current_certs >= self.cert_limit as u32;

        // 1830 SELL_AFTER="first": selling blocked in first stock round (turn == 1)
        let sell_allowed = self.turn > 1;

        // Can sell?
        // Mirrors Python's bundles_for_corporation + can_sell logic.
        // Enumerates possible sell bundles (including partial president bundles)
        // and checks fit_in_bank + can_dump for each.
        if sell_allowed {
            let can_sell = self.corporations.iter().any(|corp| {
                if corp.ipo_price.is_none() || corp.share_price.is_none() {
                    return false;
                }
                let pct = corp.percent_owned_by(&player_eid);
                if pct == 0 {
                    return false;
                }
                let market_pct = corp.market_shares_percent();
                let has_president = corp
                    .shares
                    .iter()
                    .any(|s| s.president && s.owner == player_eid);
                let num_normal = corp
                    .shares
                    .iter()
                    .filter(|s| s.owner == player_eid && !s.president)
                    .count() as u8;

                // Generate bundles like Python: accumulate normal shares, then president.
                // Each cumulative step is a bundle. President bundles also get partial variants.
                //
                // Normal-only bundles: 10%, 20%, 30%, ... (each adding one 10% share)
                // With president: all the above plus (normal_total + 20%), and
                //   partial variants at (normal_total + 20% - 10%) = (normal_total + 10%)

                // Check normal-only bundles (no president involved)
                for n in 1..=num_normal {
                    let bundle_pct = n * 10;
                    if market_pct + bundle_pct as u8 <= 50 {
                        return true; // Can sell n normal shares
                    }
                }

                // Check bundles including the president share (requires can_dump)
                if has_president {
                    let pres_pct = corp
                        .shares
                        .iter()
                        .find(|s| s.president)
                        .map_or(20, |s| s.percent);
                    let can_dump = self.players.iter().any(|p| {
                        p.id != player_id
                            && corp.percent_owned_by(&EntityId::player(p.id)) >= pres_pct
                    });
                    if can_dump {
                        // Full bundle: all normal + president
                        let full_pct = num_normal * 10 + pres_pct;
                        if market_pct + full_pct as u8 <= 50 {
                            return true;
                        }
                        // Partial president bundles: reduce percent by 10 per step
                        // In 1830: president is 20%, normal share is 10%, so 1 partial
                        // bundle at (full_pct - 10)
                        let normal_share_pct: u8 = 10; // corp.share_percent in Python
                        let num_partials = (pres_pct - normal_share_pct) / normal_share_pct;
                        for p in 1..=num_partials {
                            let partial_pct = full_pct - p * normal_share_pct;
                            if market_pct + partial_pct as u8 <= 50 {
                                return true;
                            }
                        }
                    }
                }

                false
            });
            if can_sell {
                return true;
            }
        }

        // Already bought — can only sell (checked above), buy_multiple, or pass
        if bought {
            // Check if can_buy_multiple: any corp with multiple_buy share price
            // that the player could still buy from (market only in 1830)
            let (bought_corp, bought_from_ipo, parred) = match &self.round {
                crate::rounds::Round::Stock(s) => (
                    s.bought_corp_this_turn.clone(),
                    s.bought_from_ipo,
                    s.parred_this_turn,
                ),
                _ => return false,
            };

            // Can't buy multiple if parred this turn
            if parred {
                return false;
            }

            if !at_cert_limit {
                let sold_corps = match &self.round {
                    crate::rounds::Round::Stock(s) => &s.players_sold,
                    _ => return false,
                };

                let can_buy_multiple = self.corporations.iter().any(|corp| {
                    let has_multiple_buy = corp
                        .share_price
                        .as_ref()
                        .map_or(false, |sp| sp.types.iter().any(|t| t == "multiple_buy"));
                    if !has_multiple_buy {
                        return false;
                    }
                    // Must be the same corp bought earlier (or first buy)
                    if let Some(ref bc) = bought_corp {
                        if bc != &corp.sym {
                            return false;
                        }
                    }
                    // Can't buy if sold this round
                    if sold_corps
                        .get(&player_id)
                        .and_then(|m| m.get(&corp.sym))
                        .is_some()
                    {
                        return false;
                    }
                    // Must buy from market (multiple_buy_only_from_market = true in 1830)
                    let market_eid = EntityId::market();
                    let has_market = corp
                        .shares
                        .iter()
                        .any(|s| s.owner == market_eid && !s.president);
                    if !has_market {
                        return false;
                    }
                    // In multiple_buy zone, 60% limit is lifted (holding_ok returns true)
                    // Check price affordable
                    let price = corp.share_price.as_ref().map_or(0, |sp| sp.price);
                    player_cash >= price
                });
                if can_buy_multiple {
                    return true;
                }
            }

            return false;
        }

        // Already acted (sold) but hasn't bought — can still buy/par
        // (In sell_buy_sell order, selling first doesn't prevent buying)

        if at_cert_limit {
            return false;
        }

        let sold_corps = match &self.round {
            crate::rounds::Round::Stock(s) => &s.players_sold,
            _ => return false,
        };

        // Can buy?
        let can_buy = self.corporations.iter().any(|corp| {
            if corp.ipo_price.is_none() {
                return false;
            }
            let corp_sym = &corp.sym;
            let sold = sold_corps
                .get(&player_id)
                .and_then(|m| m.get(corp_sym.as_str()))
                .is_some();
            if sold {
                return false;
            }
            // 60% limit — lifted in multiple_buy/unlimited zones
            let in_unlimited_zone = corp.share_price.as_ref().map_or(false, |sp| {
                sp.types
                    .iter()
                    .any(|t| t == "multiple_buy" || t == "unlimited")
            });
            if !in_unlimited_zone && corp.percent_owned_by(&player_eid) >= 60 {
                return false;
            }
            let ipo_eid = EntityId::ipo(corp_sym);
            let market_eid = EntityId::market();
            let has_ipo = corp
                .shares
                .iter()
                .any(|s| s.owner == ipo_eid && !s.president);
            let has_market = corp
                .shares
                .iter()
                .any(|s| s.owner == market_eid && !s.president);
            if !has_ipo && !has_market {
                return false;
            }
            let price = if has_market {
                corp.share_price.as_ref().map_or(0, |sp| sp.price)
            } else {
                corp.ipo_price.as_ref().map_or(0, |sp| sp.price)
            };
            player_cash >= price
        });
        if can_buy {
            return true;
        }

        // Can par?
        let par_prices = self.stock_market.par_prices();
        if let Some(&min_par) = par_prices.iter().min() {
            let min_par_cost = min_par * 2;
            if player_cash >= min_par_cost {
                let has_unparred = self.corporations.iter().any(|c| c.ipo_price.is_none());
                if has_unparred {
                    return true;
                }
            }
        }

        false
    }
}
