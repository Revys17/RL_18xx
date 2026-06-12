//! Single-item auction round (1867's opener).
//!
//! Port of Ruby G1867::Step::SingleItemAuction on Engine::Step::
//! PassableAuction / Auctioner — vendored at
//! docs/ruby_reference/g_1867/step/single_item_auction.rb and
//! docs/ruby_reference/base_step/{auctioner,passable_auction}.rb.
//!
//! Companies are auctioned ONE AT A TIME, cheapest first (the queue is
//! value-sorted at game construction; non-auctionable companies like 1867's
//! hidden '3' never enter it). For each company:
//!
//!   * Ascending auction: the opening bid is `company.min_bid()`
//!     (value - discount); raises are highest + $5, on the $5 grid
//!     (MUST_BID_INCREMENT_MULTIPLE), capped by the player's cash.
//!     With a bid standing, a pass leaves the auction for good; when one
//!     bidder remains, they win at their bid.
//!   * Dutch fallback: if EVERY active player declines to open the bidding,
//!     the company's price drops $5 (`company.discount += 5`) and play
//!     continues in dutch mode — any bid now buys outright; if everyone
//!     passes again the price drops another $5 and everyone re-enters,
//!     until at $0 the first eligible player force-buys (Ruby fakes the
//!     bid). Leftover companies therefore cannot exist and Ruby's
//!     `all_passed!` removal path is unreachable.
//!
//! The winner's right-hand neighbour opens the bidding on the next company
//! (`goto_entity!(winner)` + `next_entity_index!`); when the queue empties,
//! priority deal goes to the player after the last winner (base
//! `reorder_players`, NEXT_SR_PLAYER_ORDER = :after_last_to_act).

use crate::actions::GameError;
use crate::entities::EntityId;
use crate::game::BaseGame;
use crate::rounds::{AuctionState, Bid};

/// Ruby `MIN_BID_INCREMENT` (base default; 1867 keeps it).
const MIN_BID_INCREMENT: i32 = 5;

impl BaseGame {
    /// Put the first company up (the Ruby step's `setup`). Called once at
    /// game construction — affordability partitioning needs player cash, so
    /// this cannot happen inside `AuctionState::new_single_item`.
    pub(crate) fn single_auction_start(&mut self) {
        let mut state = match &self.round {
            crate::rounds::Round::Auction(s) if s.single_item.is_some() => s.clone(),
            _ => return,
        };
        match state.remaining_companies.first().copied() {
            Some(ci) => {
                self.sa_auction_entity(&mut state, ci);
                self.sa_resolve(&mut state);
            }
            None => state.finished = true,
        }
        self.set_auction_state(state);
    }

    /// Ruby `process_bid`.
    pub(crate) fn process_single_auction_bid(
        &mut self,
        entity_id: &str,
        company_sym: &str,
        price: i32,
    ) -> Result<(), GameError> {
        let player_id: u32 = entity_id
            .parse()
            .map_err(|_| GameError::new(format!("Invalid player id: {}", entity_id)))?;
        let mut state = self.get_auction_state()?;

        if player_id != state.active_player_id() {
            return Err(GameError::new(format!(
                "Not player {}'s turn, expected {}",
                player_id,
                state.active_player_id()
            )));
        }

        let ci = self
            .company_idx
            .get(company_sym)
            .copied()
            .ok_or_else(|| GameError::new(format!("Unknown company: {}", company_sym)))?;
        if state.auctioning != Some(ci) {
            return Err(GameError::new(format!(
                "{} is not up for auction",
                company_sym
            )));
        }

        // Auctioner#add_bid validations: min bid, $5-grid increment
        // (MUST_BID_INCREMENT_MULTIPLE), max bid (= player cash; the
        // single-item override has no committed-cash deduction).
        let min = self.sa_min_bid(&state, ci);
        if price < min {
            return Err(GameError::new(format!(
                "Minimum bid is {} for {}",
                min, company_sym
            )));
        }
        if (price - min) % MIN_BID_INCREMENT != 0 {
            return Err(GameError::new(format!(
                "Must increase bid by a multiple of {}",
                MIN_BID_INCREMENT
            )));
        }
        let cash = self.player_cash(player_id);
        if price > cash {
            return Err(GameError::new(format!(
                "Cannot afford bid. Maximum possible bid is {}",
                cash
            )));
        }

        let dutch = state.single_item.as_ref().expect("single-item").dutch_mode;
        if dutch {
            // Any bid in dutch mode buys outright: the bidder becomes the
            // only active bidder and resolve awards the lot.
            {
                let si = state.single_item.as_mut().expect("single-item");
                si.active_bidders = vec![player_id];
                si.declined.clear();
            }
            state.bids.insert(ci, vec![Bid { player_id, price }]);
        } else {
            {
                let si = state.single_item.as_mut().expect("single-item");
                si.declined.clear();
                if !si.active_bidders.contains(&player_id) {
                    si.active_bidders.push(player_id);
                }
            }
            {
                let bids = state.bids.entry(ci).or_default();
                bids.retain(|b| b.player_id != player_id);
                bids.push(Bid { player_id, price });
            }
            // PassableAuction#add_bid: drop active bidders who can no longer
            // meet the new minimum, removing their standing bids with them.
            let new_min = price + MIN_BID_INCREMENT;
            let cant: Vec<u32> = state
                .single_item
                .as_ref()
                .expect("single-item")
                .active_bidders
                .iter()
                .copied()
                .filter(|&p| p != player_id && self.player_cash(p) < new_min)
                .collect();
            if !cant.is_empty() {
                state
                    .single_item
                    .as_mut()
                    .expect("single-item")
                    .active_bidders
                    .retain(|p| !cant.contains(p));
                if let Some(bids) = state.bids.get_mut(&ci) {
                    bids.retain(|b| !cant.contains(&b.player_id));
                }
            }
        }

        self.sa_resolve(&mut state);
        self.set_auction_state(state);
        Ok(())
    }

    /// Ruby `process_pass`.
    pub(crate) fn process_single_auction_pass(&mut self, entity_id: &str) -> Result<(), GameError> {
        let player_id: u32 = entity_id
            .parse()
            .map_err(|_| GameError::new(format!("Invalid player id: {}", entity_id)))?;
        let mut state = self.get_auction_state()?;

        if player_id != state.active_player_id() {
            return Err(GameError::new(format!(
                "Not player {}'s turn, expected {}",
                player_id,
                state.active_player_id()
            )));
        }
        let Some(ci) = state.auctioning else {
            return Err(GameError::new("No company is up for auction"));
        };

        let has_bid = state.bids.get(&ci).map_or(false, |b| !b.is_empty());
        let dutch = state.single_item.as_ref().expect("single-item").dutch_mode;
        if has_bid || dutch {
            // pass_auction: leave this company's auction for good (any
            // standing bid goes too).
            state
                .single_item
                .as_mut()
                .expect("single-item")
                .active_bidders
                .retain(|&p| p != player_id);
            if let Some(bids) = state.bids.get_mut(&ci) {
                bids.retain(|b| b.player_id != player_id);
            }
        } else {
            // Decline to open the bidding; once everyone has declined the
            // lot enters the dutch auction $5 cheaper (the player stays
            // active — declines do not eliminate).
            let all_declined = {
                let si = state.single_item.as_mut().expect("single-item");
                if !si.declined.contains(&player_id) {
                    si.declined.push(player_id);
                }
                si.declined.len() == si.active_bidders.len()
            };
            if all_declined {
                self.sa_enter_dutch(&mut state, ci);
            }
        }

        self.sa_resolve(&mut state);
        self.set_auction_state(state);
        Ok(())
    }

    // -- internals -----------------------------------------------------------

    /// Ruby SingleItemAuction#min_bid: dutch mode → the company's dropped
    /// price; otherwise highest standing bid + $5, or the company's
    /// `min_bid()` (value - discount) opening price when no bid stands.
    fn sa_min_bid(&self, state: &AuctionState, ci: usize) -> i32 {
        let company_min = self.companies[ci].min_bid();
        if state.single_item.as_ref().expect("single-item").dutch_mode {
            return company_min;
        }
        match state
            .bids
            .get(&ci)
            .and_then(|bids| bids.iter().map(|b| b.price).max())
        {
            Some(high) => high + MIN_BID_INCREMENT,
            None => company_min,
        }
    }

    /// PassableAuction#auction_entity: put `ci` up and seed the active
    /// bidders — the rotation order from `entity_index`, minus players who
    /// cannot afford the opening bid.
    fn sa_auction_entity(&mut self, state: &mut AuctionState, ci: usize) {
        state.auctioning = Some(ci);
        state.current_auction_company = Some(ci);
        let min = self.sa_min_bid(state, ci);
        let n = state.player_order.len();
        let eligible: Vec<u32> = (0..n)
            .map(|i| state.player_order[(state.entity_index + i) % n])
            .filter(|&pid| self.player_cash(pid) >= min)
            .collect();
        let si = state.single_item.as_mut().expect("single-item");
        si.active_bidders = eligible;
        si.declined.clear();
    }

    /// 1867 `enter_dutch`: the lot failed to sell at the current price —
    /// drop it $5 and switch to (or stay in) dutch mode.
    fn sa_enter_dutch(&mut self, state: &mut AuctionState, ci: usize) {
        self.companies[ci].discount += MIN_BID_INCREMENT;
        let si = state.single_item.as_mut().expect("single-item");
        si.dutch_mode = true;
        si.declined.clear();
        state.bids.remove(&ci);
    }

    /// PassableAuction#resolve_bids plus the 1867 win/dutch hooks, made
    /// iterative (Ruby recurses through win_bid/post_win_bid).
    fn sa_resolve(&mut self, state: &mut AuctionState) {
        loop {
            let Some(ci) = state.auctioning else { return };
            let active = state
                .single_item
                .as_ref()
                .expect("single-item")
                .active_bidders
                .clone();
            let bids = state.bids.get(&ci).cloned().unwrap_or_default();

            if active.is_empty() {
                // win_bid(nil): nobody can or will buy at this price —
                // dutch drop, then post_win_bid(nil) re-opens the lot at the
                // lower price with everyone eligible back in.
                self.sa_enter_dutch(state, ci);
                self.sa_auction_entity(state, ci);
                if self.companies[ci].min_bid() == 0 {
                    // At $0 Ruby fakes a bid by the first eligible player,
                    // which (being a dutch bid) buys outright.
                    let first = state
                        .single_item
                        .as_ref()
                        .expect("single-item")
                        .active_bidders
                        .first()
                        .copied();
                    if let Some(first) = first {
                        state.single_item.as_mut().expect("single-item").active_bidders =
                            vec![first];
                        state.bids.insert(
                            ci,
                            vec![Bid {
                                player_id: first,
                                price: 0,
                            }],
                        );
                    }
                }
                continue;
            }

            if active.len() == 1 && !bids.is_empty() {
                // The last player standing wins at their bid (their bid is
                // the only one left — passes remove bids with the passer).
                let winner = bids
                    .iter()
                    .find(|b| b.player_id == active[0])
                    .cloned()
                    .unwrap_or_else(|| bids[0].clone());
                self.sa_win(state, ci, winner.player_id, winner.price);
                match state.remaining_companies.first().copied() {
                    Some(next_ci) => {
                        self.sa_auction_entity(state, next_ci);
                        continue;
                    }
                    None => {
                        // Ruby pass!: the round is over; priority deal goes
                        // to the player after the last winner
                        // (reorder_players :after_last_to_act).
                        state.auctioning = None;
                        state.current_auction_company = None;
                        state.finished = true;
                        let n = state.player_order.len();
                        self.priority_deal_player = state.player_order[state.entity_index % n];
                        return;
                    }
                }
            }

            return;
        }
    }

    /// 1867 `win_bid` + `post_win_bid` (winner half): transfer the company,
    /// take the money, and point the rotation at the winner's right-hand
    /// neighbour.
    fn sa_win(&mut self, state: &mut AuctionState, ci: usize, winner: u32, price: i32) {
        if price > 0 {
            if let Some(pi) = self.player_index(winner) {
                self.players[pi].cash -= price;
                self.bank.cash += price;
            }
        }
        self.companies[ci].owner = EntityId::player(winner);
        state.remove_company(ci);
        state.auctioning = None;
        let n = state.player_order.len();
        if let Some(wi) = state.player_order.iter().position(|&p| p == winner) {
            state.entity_index = (wi + 1) % n;
        }
        let si = state.single_item.as_mut().expect("single-item");
        si.dutch_mode = false;
        si.declined.clear();
        si.active_bidders.clear();
    }
}
