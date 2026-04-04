//! Waterfall auction round for 1830 private companies.
//!
//! Companies are auctioned in value order (cheapest first).
//! - The cheapest company can be purchased outright ("placement bid").
//! - Non-cheapest companies accumulate bids.
//! - When the cheapest is sold, resolve_bids cascades to the new cheapest.
//! - If 2+ bids on cheapest, it enters auction mode (lowest bidder acts next).
//! - If all players pass, cheapest company's min_bid drops by 5 (to 0 → free).

use crate::actions::{Action, GameError};
use crate::entities::EntityId;
use crate::game::BaseGame;
use crate::rounds::{AuctionState, Bid};

impl BaseGame {
    /// Process an action during the auction round.
    pub fn process_auction_action(&mut self, action: &Action) -> Result<(), GameError> {
        // Check if there's a pending par (CompanyPendingPar step)
        let has_pending_par =
            matches!(&self.round, crate::rounds::Round::Auction(s) if s.pending_par.is_some());

        if has_pending_par {
            return match action {
                Action::Par {
                    entity_id,
                    corporation_sym,
                    share_price,
                } => self.auction_process_pending_par(entity_id, corporation_sym, *share_price),
                _ => Err(GameError::new(format!(
                    "Expected par action for pending corporation, got {}",
                    action.action_type()
                ))),
            };
        }

        match action {
            Action::Bid {
                entity_id,
                company_sym,
                price,
            } => self.auction_process_bid(entity_id, company_sym, *price),
            Action::Pass { entity_id } => self.auction_process_pass(entity_id),
            _ => Err(GameError::new(format!(
                "Invalid action in auction: {}",
                action.action_type()
            ))),
        }
    }

    /// Process a par action for a corporation that was triggered by a company ability.
    /// The par sets the share price and transfers the president share for FREE.
    fn auction_process_pending_par(
        &mut self,
        entity_id: &str,
        corporation_sym: &str,
        share_price: i32,
    ) -> Result<(), GameError> {
        let mut state = self.get_auction_state()?;

        let (pending_corp, pending_player) = state
            .pending_par
            .as_ref()
            .ok_or_else(|| GameError::new("No pending par"))?
            .clone();

        let player_id: u32 = entity_id
            .parse()
            .map_err(|_| GameError::new(format!("Invalid player id: {}", entity_id)))?;

        if player_id != pending_player {
            return Err(GameError::new(format!(
                "Not player {}'s turn for pending par",
                player_id
            )));
        }

        if corporation_sym != pending_corp {
            return Err(GameError::new(format!(
                "Expected par for {}, got {}",
                pending_corp, corporation_sym
            )));
        }

        // Set par price
        let par_sp = self
            .stock_market
            .par_price(share_price)
            .ok_or_else(|| GameError::new(format!("Invalid par price: {}", share_price)))?;

        let corp_idx = *self
            .corp_idx
            .get(corporation_sym)
            .ok_or_else(|| GameError::new(format!("Unknown corporation: {}", corporation_sym)))?;

        self.corporations[corp_idx].ipo_price = Some(par_sp.clone());
        self.corporations[corp_idx].share_price = Some(par_sp.clone());
        let sym = corporation_sym.to_string();
        self.update_market_cell(&sym, 0, 0, par_sp.row, par_sp.column);

        // Initialize all shares as IPO
        let ipo_eid = EntityId::ipo(corporation_sym);
        for share in &mut self.corporations[corp_idx].shares {
            if share.owner.is_none() {
                share.owner = ipo_eid.clone();
            }
        }

        // Transfer president share to player for FREE (exchange="free" in Python)
        let player_eid = EntityId::player(player_id);
        self.corporations[corp_idx].shares[0].owner = player_eid.clone();
        self.corporations[corp_idx].owner_id = player_eid;

        // Clear pending par
        state.pending_par = None;

        // Check if auction should finish
        if state.remaining_companies.is_empty() {
            state.finished = true;
            // Set priority based on entity_index (after_last_to_act ordering)
            let ei = state.entity_index % state.player_order.len();
            self.priority_deal_player = state.player_order[ei];
        }

        self.set_auction_state(state);
        Ok(())
    }

    fn auction_process_bid(
        &mut self,
        entity_id: &str,
        company_sym: &str,
        price: i32,
    ) -> Result<(), GameError> {
        let player_id: u32 = entity_id
            .parse()
            .map_err(|_| GameError::new(format!("Invalid player id: {}", entity_id)))?;

        let state = self.get_auction_state()?;

        // Validate it's this player's turn
        if player_id != state.active_player_id() {
            return Err(GameError::new(format!(
                "Not player {}'s turn, expected {}",
                player_id,
                state.active_player_id()
            )));
        }

        // Find the company index
        let company_idx = self
            .company_idx
            .get(company_sym)
            .copied()
            .ok_or_else(|| GameError::new(format!("Unknown company: {}", company_sym)))?;

        // Validate company is still available
        if !state.remaining_companies.contains(&company_idx) {
            return Err(GameError::new(format!("{} is not available", company_sym)));
        }

        let company_value = self.companies[company_idx].value;

        // Validate bid amount
        let min_required = state.min_bid_for(company_idx, company_value);
        if price < min_required {
            return Err(GameError::new(format!(
                "Bid {} is below minimum {}",
                price, min_required
            )));
        }

        let player_cash = self.player_cash(player_id);
        let max_allowed = state.max_bid(player_id, company_idx, player_cash);
        if price > max_allowed {
            return Err(GameError::new(format!(
                "Bid {} exceeds max {} (cash {} - committed)",
                price, max_allowed, player_cash
            )));
        }

        // Unpass the player
        let mut new_state = state.clone();
        if let Some(pidx) = new_state
            .player_order
            .iter()
            .position(|&id| id == player_id)
        {
            new_state.passed[pidx] = false;
        }

        if let Some((auction_company, _)) = state.active_auction() {
            // Active auction: must bid on the auctioning company
            if company_idx != auction_company {
                return Err(GameError::new(format!(
                    "Must bid on the company being auctioned, not {}",
                    company_sym
                )));
            }
            // Update existing bid or add new one
            let bids = new_state.bids.entry(company_idx).or_default();
            if let Some(existing) = bids.iter_mut().find(|b| b.player_id == player_id) {
                existing.price = price;
            } else {
                bids.push(Bid { player_id, price });
            }
        } else {
            // Placement phase: bid on a company
            if state.may_purchase(company_idx) {
                // Direct purchase of cheapest company
                self.auction_buy_company(&mut new_state, company_idx, player_id, price)?;
                self.auction_resolve_bids(&mut new_state)?;
                new_state.advance_entity();
            } else {
                // Bid on a non-cheapest company (recorded for later)
                let bids = new_state.bids.entry(company_idx).or_default();
                if let Some(existing) = bids.iter_mut().find(|b| b.player_id == player_id) {
                    existing.price = price;
                } else {
                    bids.push(Bid { player_id, price });
                }
                new_state.advance_entity();
            }
        }

        self.set_auction_state(new_state);
        Ok(())
    }

    fn auction_process_pass(&mut self, entity_id: &str) -> Result<(), GameError> {
        let player_id: u32 = entity_id
            .parse()
            .map_err(|_| GameError::new(format!("Invalid player id: {}", entity_id)))?;

        let state = self.get_auction_state()?;

        if player_id != state.active_player_id() {
            return Err(GameError::new(format!(
                "Not player {}'s turn, expected {}",
                player_id,
                state.active_player_id()
            )));
        }

        let mut new_state = state.clone();

        if let Some((auction_company, _)) = state.active_auction() {
            // Pass during active auction: remove player's bid from the auction
            if let Some(bids) = new_state.bids.get_mut(&auction_company) {
                bids.retain(|b| b.player_id != player_id);
                // If only one bidder left, they win
                if bids.len() == 1 {
                    let winner_id = bids[0].player_id;
                    let winning_price = bids[0].price;
                    new_state.auctioning = None;
                    self.auction_buy_company(
                        &mut new_state,
                        auction_company,
                        winner_id,
                        winning_price,
                    )?;
                    self.auction_resolve_bids(&mut new_state)?;
                }
            }
        } else {
            // Waterfall phase: mark player as passed
            if let Some(pidx) = new_state
                .player_order
                .iter()
                .position(|&id| id == player_id)
            {
                new_state.passed[pidx] = true;
            }

            if new_state.all_passed() {
                // Check if the current auction target is still in the remaining companies
                let target_in_remaining = new_state
                    .current_auction_company
                    .map_or(false, |t| new_state.remaining_companies.contains(&t));

                if target_in_remaining {
                    // Normal path: reduce cheapest company price
                    if let Some(cheapest_idx) = new_state.cheapest_company() {
                        new_state.discount += 5;
                        let company_value = self.companies[cheapest_idx].value;
                        let new_min = (company_value - new_state.discount).max(0);

                        if new_min <= 0 {
                            // Give to current entity for free
                            new_state.advance_entity();
                            let buyer_id = new_state.current_player_id();
                            self.auction_buy_company(&mut new_state, cheapest_idx, buyer_id, 0)?;
                            self.auction_resolve_bids(&mut new_state)?;
                        }

                        new_state.unpass_all();
                        new_state.advance_entity();
                    } else {
                        new_state.finished = true;
                    }
                } else if !new_state.remaining_companies.is_empty() {
                    // Target was already sold — pay company revenues and continue
                    // (Python: self.game.payout_companies(), self.game.or_set_finished())
                    // Don't update current_auction_company — it stays as the original
                    // target (already sold), so subsequent all-pass cycles also trigger payouts.
                    self.payout_companies();
                    new_state.unpass_all();
                    new_state.advance_entity();
                } else {
                    // All companies sold, pay and finish
                    self.payout_companies();
                    new_state.finished = true;
                }
            } else {
                new_state.advance_entity();
            }
        }

        self.set_auction_state(new_state);
        Ok(())
    }

    /// Buy a company at auction: transfer money and ownership, process abilities.
    fn auction_buy_company(
        &mut self,
        state: &mut AuctionState,
        company_idx: usize,
        player_id: u32,
        price: i32,
    ) -> Result<(), GameError> {
        let player_idx = self
            .player_index(player_id)
            .ok_or_else(|| GameError::new(format!("Player {} not found", player_id)))?;

        if price > 0 {
            self.players[player_idx].cash -= price;
            self.bank.cash += price;
        }
        self.companies[company_idx].owner = EntityId::player(player_id);

        // Process company abilities
        let company_sym = self.companies[company_idx].sym.clone();
        self.process_company_abilities(&company_sym, player_id, state);

        state.remove_company(company_idx);

        if state.remaining_companies.is_empty() && state.pending_par.is_none() {
            state.finished = true;
        }

        Ok(())
    }

    /// Process abilities triggered when a company is bought.
    fn process_company_abilities(
        &mut self,
        company_sym: &str,
        player_id: u32,
        state: &mut AuctionState,
    ) {
        let company_defs = crate::title::g1830::companies();
        let def = company_defs.iter().find(|c| c.sym == company_sym);

        if let Some(def) = def {
            // grants_share: give a free share of a corporation to the buyer
            if let Some((corp_sym, _percent)) = &def.grants_share {
                let player_eid = EntityId::player(player_id);
                if let Some(&corp_idx) = self.corp_idx.get(*corp_sym) {
                    // Find a non-president share (from IPO or uninitialized) and transfer it
                    let ipo_eid = EntityId::ipo(corp_sym);
                    for share in &mut self.corporations[corp_idx].shares {
                        if !share.president && (share.owner == ipo_eid || share.owner.is_none()) {
                            share.owner = player_eid;
                            break;
                        }
                    }
                }
            }

            // triggers_par: the buyer must set the par price for a corporation
            if let Some(corp_sym) = def.triggers_par {
                state.pending_par = Some((corp_sym.to_string(), player_id));
            }
        }
    }

    /// After buying a company, try to resolve bids on the new cheapest company.
    fn auction_resolve_bids(&mut self, state: &mut AuctionState) -> Result<(), GameError> {
        loop {
            let cheapest = match state.cheapest_company() {
                Some(c) => c,
                None => {
                    // Only finish if no pending par
                    if state.pending_par.is_none() {
                        state.finished = true;
                        let ei = state.entity_index % state.player_order.len();
                        self.priority_deal_player = state.player_order[ei];
                    }
                    return Ok(());
                }
            };

            let bids = state.bids.get(&cheapest).cloned().unwrap_or_default();

            if bids.len() == 1 {
                // Single bidder: auto-win
                let winner_id = bids[0].player_id;
                let winning_price = bids[0].price;
                state.auctioning = None;
                self.auction_buy_company(state, cheapest, winner_id, winning_price)?;
                // Continue loop to check next cheapest
            } else if bids.len() > 1 {
                // Multiple bidders: enter auction mode
                state.auctioning = Some(cheapest);
                return Ok(());
            } else {
                // No bids on this company: wait for waterfall
                state.auctioning = None;
                return Ok(());
            }
        }
    }

    // -- Helpers --

    fn get_auction_state(&self) -> Result<AuctionState, GameError> {
        match &self.round {
            crate::rounds::Round::Auction(s) => Ok(s.clone()),
            _ => Err(GameError::new("Not in auction round")),
        }
    }

    fn set_auction_state(&mut self, state: AuctionState) {
        self.round = crate::rounds::Round::Auction(state);
        self.update_round_state();
    }

    fn player_cash(&self, player_id: u32) -> i32 {
        self.players
            .iter()
            .find(|p| p.id == player_id)
            .map_or(0, |p| p.cash)
    }

    pub(crate) fn player_index(&self, player_id: u32) -> Option<usize> {
        self.players.iter().position(|p| p.id == player_id)
    }

    /// Pay private company revenues to their owners.
    pub(crate) fn payout_companies(&mut self) {
        for company in &self.companies {
            if company.closed || company.revenue <= 0 {
                continue;
            }
            if let Some(pid) = company.owner.player_id() {
                if let Some(idx) = self.player_index(pid) {
                    self.players[idx].cash += company.revenue;
                    self.bank.cash -= company.revenue;
                }
            } else if let Some(sym) = company.owner.corp_sym() {
                if let Some(&idx) = self.corp_idx.get(sym) {
                    self.corporations[idx].cash += company.revenue;
                    self.bank.cash -= company.revenue;
                }
            }
        }
    }
}
