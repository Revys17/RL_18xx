//! In-stock-round corporation bidding (1867 minor founding).
//!
//! Port of Ruby Engine::Step::BuySellParSharesViaBid (vendored at
//! docs/ruby_reference/base_step/buy_sell_par_shares_via_bid.rb) +
//! G1867::Step::BuySellParShares (g_1867/step/buy_sell_par_shares.rb):
//!
//!   * A minor is STARTED by auction: a player's `bid` on it (min $100,
//!     $5 grid) is their buy action for the turn and opens a live auction —
//!     the opening bidder immediately chooses the minor's home city
//!     (HOME_TOKEN_TIMING = :par → a pending home token, the HomeToken
//!     step blocks until placed).
//!   * While the auction runs only bid/pass are legal; the active player is
//!     the active bidder after the standing high bidder; a pass leaves the
//!     auction for good; when one bidder remains they win.
//!   * Win (G1867 win_bid): the minor pars at min(bid/2, $135) snapped DOWN
//!     to a par_1/par cell, the winner takes the 100% president cert and
//!     pays their FULL bid into the minor's treasury (Ruby's temporary
//!     2×par bank grant + claw-back nets to exactly that), and the turn
//!     passes to the player after the winner.

use crate::actions::GameError;
use crate::entities::EntityId;
use crate::game::BaseGame;
use crate::rounds::{Bid, StockBidAuction, StockState};
use crate::title::CorpType;

/// Ruby G1867::Step::BuySellParShares constants.
const MIN_BID: i32 = 100;
const MAX_MINOR_PAR: i32 = 135;
const MIN_BID_INCREMENT: i32 = 5;

impl BaseGame {
    /// The player who acts while an SR auction is live (ViaBid
    /// `active_entities`): the active bidder AFTER the standing high bidder.
    pub(crate) fn stock_bid_active_player(&self, auction: &StockBidAuction) -> Option<u32> {
        let high = auction.bids.iter().max_by_key(|b| b.price)?;
        let i = auction
            .active_bidders
            .iter()
            .position(|&p| p == high.player_id)?;
        Some(auction.active_bidders[(i + 1) % auction.active_bidders.len()])
    }

    /// Whether `player_id` could open a bid auction on any minor right now
    /// (ViaBid `can_bid_any?`) — gates the "bid" offer in the step arms.
    pub(crate) fn stock_can_bid_minor(&self, state: &StockState, player_id: u32) -> bool {
        if state.bought_this_turn || state.parred_this_turn || state.bid_auction.is_some() {
            return false;
        }
        // `minors_cannot_start` (first 5-train) closes the founding window.
        if self.events_fired.iter().any(|e| e == "minors_cannot_start") {
            return false;
        }
        if self.player_cash(player_id) < MIN_BID {
            return false;
        }
        let title = self.title_def();
        self.corporations.iter().any(|c| {
            c.corp_type == CorpType::Minor
                && c.ipo_price.is_none()
                && title.corporation_startable(&c.sym, &self.phase.name)
        })
    }

    /// Ruby ViaBid `process_bid` — `selection_bid` when no auction is live,
    /// `add_bid` when one is.
    pub(crate) fn process_corporation_bid(
        &mut self,
        entity_id: &str,
        corporation_sym: &str,
        price: i32,
    ) -> Result<(), GameError> {
        let state = match &self.round {
            crate::rounds::Round::Stock(s) => s.clone(),
            _ => return Err(GameError::new("Not in stock round")),
        };
        if !state.pending_home_tokens.is_empty() {
            return Err(GameError::new("A home token must be placed first"));
        }
        let player_id: u32 = entity_id
            .parse()
            .map_err(|_| GameError::new(format!("Invalid player id: {}", entity_id)))?;

        let corp_idx = *self
            .corp_idx
            .get(corporation_sym)
            .ok_or_else(|| GameError::new(format!("Unknown corporation: {}", corporation_sym)))?;

        let mut new_state = state.clone();
        if let Some(auction) = state.bid_auction.as_ref() {
            // -- raise on the live auction --
            if auction.corporation_sym != corporation_sym {
                return Err(GameError::new(format!(
                    "{} is not up for auction",
                    corporation_sym
                )));
            }
            let expected = self.stock_bid_active_player(auction);
            if expected != Some(player_id) {
                return Err(GameError::new(format!(
                    "Not player {}'s turn to bid, expected {:?}",
                    player_id, expected
                )));
            }
            let high = auction.bids.iter().map(|b| b.price).max().unwrap_or(0);
            let min = high + MIN_BID_INCREMENT;
            self.validate_corp_bid(player_id, price, min)?;

            let a = new_state.bid_auction.as_mut().expect("auction");
            a.bids.retain(|b| b.player_id != player_id);
            a.bids.push(Bid { player_id, price });
            // PassableAuction#add_bid: drop active bidders who can't meet
            // the new minimum (their bids go too).
            let new_min = price + MIN_BID_INCREMENT;
            let cant: Vec<u32> = a
                .active_bidders
                .iter()
                .copied()
                .filter(|&p| p != player_id && self.player_cash(p) < new_min)
                .collect();
            if !cant.is_empty() {
                let a = new_state.bid_auction.as_mut().expect("auction");
                a.active_bidders.retain(|p| !cant.contains(p));
                a.bids.retain(|b| !cant.contains(&b.player_id));
            }
        } else {
            // -- selection bid: open the auction (the player's buy action) --
            if player_id != state.current_player_id() {
                return Err(GameError::new(format!(
                    "Not player {}'s turn, expected {}",
                    player_id,
                    state.current_player_id()
                )));
            }
            if state.bought_this_turn || state.parred_this_turn {
                return Err(GameError::new("Already bought/started this turn"));
            }
            let corp = &self.corporations[corp_idx];
            if corp.corp_type != CorpType::Minor {
                return Err(GameError::new(format!(
                    "{} cannot be started by bid (not a minor)",
                    corporation_sym
                )));
            }
            if corp.ipo_price.is_some() {
                return Err(GameError::new(format!(
                    "{} has already been started",
                    corporation_sym
                )));
            }
            if !self
                .title_def()
                .corporation_startable(corporation_sym, &self.phase.name)
                || self.events_fired.iter().any(|e| e == "minors_cannot_start")
            {
                return Err(GameError::new(format!(
                    "{} is not available in phase {}",
                    corporation_sym, self.phase.name
                )));
            }
            self.validate_corp_bid(player_id, price, MIN_BID)?;

            // PassableAuction#auction_entity: active bidders = the rotation
            // from the triggerer, minus players who can't meet the next
            // minimum (the triggerer always stays).
            let n = state.player_order.len();
            let start = state
                .player_order
                .iter()
                .position(|&p| p == player_id)
                .unwrap_or(0);
            let min_next = price + MIN_BID_INCREMENT;
            let active: Vec<u32> = (0..n)
                .map(|i| state.player_order[(start + i) % n])
                .filter(|&p| p == player_id || self.player_cash(p) >= min_next)
                .collect();
            new_state.bid_auction = Some(StockBidAuction {
                corporation_sym: corporation_sym.to_string(),
                bids: vec![Bid { player_id, price }],
                active_bidders: active,
            });
            // HOME_TOKEN_TIMING = :par — the opening bidder chooses the
            // minor's home city NOW (ViaBid add_bid → place_home_token →
            // pending token; the HomeToken step blocks until placed).
            new_state
                .pending_home_tokens
                .push((corporation_sym.to_string(), player_id));
        }

        let won = self.stock_bid_resolve(&mut new_state);
        self.round = crate::rounds::Round::Stock(new_state);
        self.update_round_state();
        if won {
            // The winner's turn is consumed (Ruby goto_entity! + pass!):
            // advance to the player after them with fresh turn flags.
            self.stock_next_entity();
        }
        Ok(())
    }

    /// A pass while the SR auction is live (ViaBid `pass!` →
    /// `pass_auction`): leave the auction for good.
    pub(crate) fn process_stock_bid_pass(&mut self, entity_id: &str) -> Result<(), GameError> {
        let state = match &self.round {
            crate::rounds::Round::Stock(s) => s.clone(),
            _ => return Err(GameError::new("Not in stock round")),
        };
        let Some(auction) = state.bid_auction.as_ref() else {
            return Err(GameError::new("No corporation auction is live"));
        };
        let player_id: u32 = entity_id
            .parse()
            .map_err(|_| GameError::new(format!("Invalid player id: {}", entity_id)))?;
        let expected = self.stock_bid_active_player(auction);
        if expected != Some(player_id) {
            return Err(GameError::new(format!(
                "Not player {}'s turn to bid, expected {:?}",
                player_id, expected
            )));
        }

        let mut new_state = state.clone();
        {
            let a = new_state.bid_auction.as_mut().expect("auction");
            a.active_bidders.retain(|&p| p != player_id);
            a.bids.retain(|b| b.player_id != player_id);
        }
        let won = self.stock_bid_resolve(&mut new_state);
        self.round = crate::rounds::Round::Stock(new_state);
        self.update_round_state();
        if won {
            self.stock_next_entity();
        }
        Ok(())
    }

    /// Place a pending STOCK-round home token (1867 HomeToken step; the
    /// action's entity is the CORPORATION, the chooser is the player who
    /// triggered it).
    pub(crate) fn process_stock_home_token(
        &mut self,
        entity_id: &str,
        hex_id: &str,
        city_index: u8,
    ) -> Result<(), GameError> {
        let state = match &self.round {
            crate::rounds::Round::Stock(s) => s.clone(),
            _ => return Err(GameError::new("Not in stock round")),
        };
        let Some((pending_sym, _chooser)) = state.pending_home_tokens.first().cloned() else {
            return Err(GameError::new("No home token is pending"));
        };
        if entity_id != pending_sym {
            return Err(GameError::new(format!(
                "Expected home token for {}, got {}",
                pending_sym, entity_id
            )));
        }
        let corp_idx = *self
            .corp_idx
            .get(&pending_sym)
            .ok_or_else(|| GameError::new(format!("Unknown corporation: {}", pending_sym)))?;

        // Resolve "__tile:M9-0"-style refs the same way the operating-round
        // token step does (preprinted tiles are named by their hex coord).
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
            hex_id.to_string()
        };
        let hex_idx = *self
            .hex_idx
            .get(resolved_hex_id.as_str())
            .ok_or_else(|| GameError::new(format!("Unknown hex: {}", resolved_hex_id)))?;

        // TODO(1867-home): full home_token_locations legality (minors: any
        // city with a free slot incl. disconnected Toronto/Montreal; majors:
        // no normal token anywhere on the hex). Replay-grade validation
        // here: the chosen city must exist and have a free slot.
        let slot_idx = self
            .token_slot_for(&resolved_hex_id, city_index as usize, &pending_sym)
            .ok_or_else(|| GameError::new("No empty token slots"))?;
        let city = self.hexes[hex_idx]
            .tile
            .cities
            .get_mut(city_index as usize)
            .ok_or_else(|| GameError::new("Invalid city index"))?;
        if city.tokens.get(slot_idx).map_or(true, |t| t.is_some()) {
            return Err(GameError::new("No empty token slots"));
        }
        let mut token = self.corporations[corp_idx].tokens[0].clone();
        token.used = true;
        token.city_hex_id = resolved_hex_id.clone();
        city.tokens[slot_idx] = Some(token);
        self.corporations[corp_idx].tokens[0].used = true;
        self.corporations[corp_idx].tokens[0].city_hex_id = resolved_hex_id;
        self.corporations[corp_idx].home_token_ever_placed = true;
        self.clear_graph_cache();

        let mut new_state = state;
        new_state.pending_home_tokens.remove(0);
        let round_over = new_state.pending_home_tokens.is_empty()
            && new_state.bid_auction.is_none()
            && new_state.all_players_passed();
        self.round = crate::rounds::Round::Stock(new_state);
        self.update_round_state();
        if round_over {
            // The pending token was the only thing keeping the round open
            // (everyone had already passed) — finish it now.
            self.stock_next_entity();
        }
        Ok(())
    }

    // -- internals -----------------------------------------------------------

    /// Shared bid validation (Auctioner#add_bid): minimum, the $5 grid
    /// (MUST_BID_INCREMENT_MULTIPLE), and the player's cash (ViaBid max_bid).
    fn validate_corp_bid(&self, player_id: u32, price: i32, min: i32) -> Result<(), GameError> {
        if price < min {
            return Err(GameError::new(format!("Minimum bid is {}", min)));
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
        Ok(())
    }

    /// PassableAuction#resolve_bids: one active bidder left → they win.
    /// Returns true when the auction resolved (the caller hands the turn on).
    fn stock_bid_resolve(&mut self, state: &mut StockState) -> bool {
        let Some(auction) = state.bid_auction.clone() else { return false };
        if auction.active_bidders.len() != 1 || auction.bids.is_empty() {
            return false;
        }
        let winner = auction.bids[0].clone();
        self.stock_bid_win(state, &auction.corporation_sym, winner.player_id, winner.price);
        true
    }

    /// G1867 `win_bid`: par at min(bid/2, 135) snapped down to a par_1/par
    /// cell, president cert + full bid to the minor, turn to the player
    /// after the winner.
    fn stock_bid_win(&mut self, state: &mut StockState, corp_sym: &str, winner: u32, price: i32) {
        let corp_idx = self.corp_idx[corp_sym];

        // Par: the greatest par_1/par cell ≤ min(bid/2, MAX_MINOR_PAR)
        // (share_prices_with_types is sorted descending).
        let target = (price / 2).min(MAX_MINOR_PAR);
        let sp = self
            .stock_market
            .share_prices_with_types(&["par_1", "par"])
            .into_iter()
            .find(|sp| sp.price <= target)
            .expect("a minor par cell at or below the bid floor always exists");
        self.corporations[corp_idx].ipo_price = Some(sp.clone());
        self.corporations[corp_idx].share_price = Some(sp.clone());
        let sym = corp_sym.to_string();
        self.update_market_cell(&sym, 0, 0, sp.row, sp.column);

        // The 100% president cert to the winner; the FULL bid into the
        // minor's treasury (Ruby's 2×par bank-grant dance nets to this).
        let player_eid = EntityId::player(winner);
        self.corporations[corp_idx].set_share_owner(0, player_eid.clone());
        self.corporations[corp_idx].owner_id = player_eid;
        if let Some(pi) = self.player_index(winner) {
            self.players[pi].cash -= price;
        }
        self.corporations[corp_idx].cash += price;
        // 100% sold → floats (incremental: no bank lump sum).
        self.check_float(corp_idx);

        state.bid_auction = None;

        // Ruby: @round.goto_entity!(winner) + pass! — the bid was the
        // winner's turn; the caller advances to the player after them
        // (stock_next_entity, with fresh turn flags).
        if let Some(wi) = state.player_order.iter().position(|&p| p == winner) {
            state.current_player_index = wi;
        }
        state.acted_this_turn = true;
        state.consecutive_passes = 0;
        state.unpass_current_player();
        state.priority_deal_player = self.next_player_id(winner);
    }
}
