//! Stock round logic for 1830.
//!
//! Players take turns buying and selling shares. 1830 uses sell_buy_sell order,
//! meaning there's no restriction on interleaving buys and sells.
//! A player can sell multiple bundles, buy one bundle (IPO or market),
//! par one corporation, or pass to end their turn.
//! The round ends when all players pass without acting.

use crate::actions::{Action, GameError};
use crate::entities::{Corporation, EntityId};
use crate::game::BaseGame;
use crate::rounds::StockState;

impl BaseGame {
    /// Process an action during the stock round.
    pub fn process_stock_action(&mut self, action: &Action) -> Result<(), GameError> {
        let state = match &self.round {
            crate::rounds::Round::Stock(s) => s.clone(),
            _ => return Err(GameError::new("Not in stock round")),
        };

        // 1867 overlays on the stock round (rounds/stock_bid.rs): a pending
        // home token blocks everything; a live minor auction narrows the
        // round to bid/pass.
        if !state.pending_home_tokens.is_empty() {
            return match action {
                Action::PlaceToken {
                    entity_id,
                    hex_id,
                    city_index,
                } => self.process_stock_home_token(entity_id, hex_id, *city_index),
                _ => Err(GameError::new(format!(
                    "A home token must be placed before {}",
                    action.action_type()
                ))),
            };
        }
        if state.bid_auction.is_some() {
            return match action {
                Action::CorporationBid {
                    entity_id,
                    corporation_sym,
                    price,
                } => self.process_corporation_bid(entity_id, corporation_sym, *price),
                Action::Pass { entity_id } => self.process_stock_bid_pass(entity_id),
                _ => Err(GameError::new(format!(
                    "Invalid action during corporation auction: {}",
                    action.action_type()
                ))),
            };
        }

        match action {
            Action::CorporationBid {
                entity_id,
                corporation_sym,
                price,
            } => self.process_corporation_bid(entity_id, corporation_sym, *price),
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

        let corp_idx = *self
            .corp_idx
            .get(corporation_sym)
            .ok_or_else(|| GameError::new(format!("Unknown corporation: {}", corporation_sym)))?;

        // Minors are never parred directly (1867 founds them by bid in the
        // stock round); titles may gate startability by phase (1867 majors:
        // phase 4+). Both are no-ops for 1830.
        if self.corporations[corp_idx].corp_type == crate::title::CorpType::Minor {
            return Err(GameError::new(format!(
                "{} cannot be started by par (founded by bid)",
                corporation_sym
            )));
        }
        if !self
            .title_def()
            .corporation_startable(corporation_sym, &self.phase.name)
        {
            return Err(GameError::new(format!(
                "{} is not available in phase {}",
                corporation_sym, self.phase.name
            )));
        }

        // Validate the par price against the title's legal par cells for
        // this corporation class (Ruby G1867::Step::BuySellParShares
        // #get_all_par_prices: majors par on `%i[par_2 par]` — the
        // $150-200 z cells are major-legal; 1830 keeps plain `par` cells).
        let par_types = self
            .title_def()
            .par_price_types(self.corporations[corp_idx].corp_type);
        let par_sp = self
            .stock_market
            .share_prices_with_types(par_types)
            .into_iter()
            .find(|sp| sp.price == share_price)
            .ok_or_else(|| GameError::new(format!("Invalid par price: {}", share_price)))?;

        if self.corporations[corp_idx].ipo_price.is_some() || self.corporations[corp_idx].closed {
            return Err(GameError::new(format!(
                "{} has already been parred",
                corporation_sym
            )));
        }

        // Player must afford the president's certificate (its unit count ×
        // par; 1830: 2 × par for the 20% cert)
        let cost = share_price * self.corporations[corp_idx].president_share_units();
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
        let n = self.corporations[corp_idx].shares.len();
        for i in 0..n {
            if self.corporations[corp_idx].shares[i].owner.is_none() {
                self.corporations[corp_idx]
                    .set_share_owner(i, ipo_eid.clone());
            }
        }

        // Transfer president share to player
        let player_eid = EntityId::player(player_id);
        self.corporations[corp_idx]
            .set_share_owner(0, player_eid.clone());

        // Player pays. Under incremental capitalization (1867) the
        // president's-cert payment goes to the CORPORATION (Ruby
        // SharePool#buy_shares: receiver = bundle.owner when capitalization
        // is :incremental and the cert sits in the corp's own treasury-IPO,
        // share_pool.rb:104-115); full-cap titles (1830) pay the bank.
        self.players[player_idx].cash -= cost;
        if self.corporations[corp_idx].capitalization == crate::title::Capitalization::Incremental {
            self.corporations[corp_idx].cash += cost;
        } else {
            self.bank.cash += cost;
        }

        // Set president
        self.corporations[corp_idx].owner_id = player_eid;

        // Check float
        self.check_float(corp_idx);

        // HOME_TOKEN_TIMING = :par with no fixed home (1867 majors,
        // CorporationDef.home_hex == ""): the parring player chooses the
        // home city NOW — queue the pending token and let the HomeToken
        // step block the round until the corp's place_token lands (the
        // same machinery as bid-founded minors, rounds/stock_bid.rs).
        let needs_home_choice = self
            .title_def()
            .corporations()
            .iter()
            .find(|cd| cd.sym == corporation_sym)
            .map_or(false, |cd| cd.home_hex.is_empty());
        if needs_home_choice {
            new_state
                .pending_home_tokens
                .push((corporation_sym.to_string(), player_id));
        }

        // Update state
        new_state.bought_this_turn = true;
        new_state.parred_this_turn = true;
        new_state.acted_this_turn = true;
        new_state.consecutive_passes = 0;

        // Set priority deal: next player after this one
        new_state.priority_deal_player = self.next_player_id(player_id);

        self.round = crate::rounds::Round::Stock(new_state);
        self.update_round_state();
        if !needs_home_choice {
            // With a home choice pending the turn does NOT auto-advance —
            // process_stock_home_token resumes it after the token lands.
            self.stock_after_process();
        }
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

        let corp_idx = *self
            .corp_idx
            .get(corporation_sym)
            .ok_or_else(|| GameError::new(format!("Unknown corporation: {}", corporation_sym)))?;

        // Check cert limit — bypass if the target corporation is in a cert-limit-exempt zone
        let target_counts_for_limit = !Self::corp_exempt_from_cert_limit(&self.corporations[corp_idx]);
        if target_counts_for_limit {
            let current_certs = self.num_certs_internal(player_id);
            if current_certs >= self.cert_limit as u32 {
                return Err(GameError::new("At certificate limit"));
            }
        }

        let ipo_eid = EntityId::ipo(corporation_sym);
        let market_eid = EntityId::market();

        // Determine the source. When share_indices are provided, check Rust's
        // state of that share to infer the intended source (IPO vs market).
        let inferred_source = if !share_indices.is_empty() {
            let idx = share_indices[0];
            if idx < self.corporations[corp_idx].shares.len() {
                let share = &self.corporations[corp_idx].shares[idx];
                if share.owner == ipo_eid {
                    Some("ipo")
                } else if share.owner == market_eid {
                    Some("market")
                } else {
                    None // Share owned by player/other — index diverged
                }
            } else {
                None
            }
        } else {
            None
        };

        let effective_source = inferred_source.unwrap_or(source);

        // If the action specifies a share index whose Rust-side owner matches
        // the effective source, use that EXACT index. This keeps Rust's share
        // assignments aligned with Python's (Python honors the share_id in
        // the action when moving shares). Fall back to `position(...)` when
        // the index is unusable.
        let (share_idx, actual_source) = if let (Some(src), false) = (inferred_source, share_indices.is_empty()) {
            let idx = share_indices[0];
            (idx, src)
        } else if effective_source == "ipo" {
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

        // IPO shares are sold at par price, market shares at current market
        // price — except `always_market_price` corps (1867), whose treasury
        // shares also sell at the CURRENT market price (Ruby
        // share.rb `price_per_share`).
        let unit_price = if actual_source == "ipo" && !self.corporations[corp_idx].always_market_price {
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

        // units bought × price-per-unit
        let price = (percent as i32 * unit_price) / self.corporations[corp_idx].share_unit();
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
        let max_ownership = self.corporations[corp_idx].max_ownership_percent;
        if !in_unlimited_zone && current_pct + percent > max_ownership {
            return Err(GameError::new(format!(
                "Would exceed {}% ownership of {}",
                max_ownership, corporation_sym
            )));
        }

        // Snapshot pre-action owners of every share in this corporation. This
        // mirrors Python's insertion-order semantics for `shares_for_presidency_swap`:
        // when a buy triggers a president change, the swap picks the new president's
        // OLDEST shares (i.e. ones owned BEFORE the buy), not the just-bought one.
        let pre_action_owners: Vec<EntityId> = self.corporations[corp_idx]
            .shares
            .iter()
            .map(|s| s.owner.clone())
            .collect();

        // Transfer. Under incremental capitalization (1867) a treasury-share
        // buy pays the CORPORATION (Ruby SharePool#buy_shares routes to the
        // corporation when `corporation.capitalization != :full`); market
        // shares and full-cap titles pay the bank.
        self.corporations[corp_idx].set_share_owner(share_idx, player_eid.clone());
        self.players[player_idx].cash -= price;
        let incremental_treasury_buy = actual_source == "ipo"
            && self.corporations[corp_idx].capitalization
                == crate::title::Capitalization::Incremental;
        if incremental_treasury_buy {
            self.corporations[corp_idx].cash += price;
        } else {
            self.bank.cash += price;
        }

        // Check float
        self.check_float(corp_idx);

        // Check president change
        self.check_president_change_with_snapshot(corp_idx, None, pre_action_owners);

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

        // Validate-then-mutate: BuySellParShares.sell_shares -> can_sell
        // (round.py:1839-1844, 1601-1621) before any state change — ownership
        // of the named certs / percent consistency, check_sale_timing
        // (SELL_AFTER="first": no sales in the first stock round), the 50%
        // market cap (fit_in_bank), and president-dump legality (can_dump).
        self.validate_sell_bundle(player_id, corporation_sym, percent, share_indices, None)?;

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
            && remaining_after < self.corporations[corp_idx].president_percent();

        // Snapshot pre-action owners of every share. This is used in two places:
        //   1. `handle_partial`: pick shares that were ALREADY in market before
        //      this sell (oldest market shares) — matches Python's
        //      `share_pool.shares_of(corp)[0]` semantics.
        //   2. President swap (when the seller dumps the pres cert): pick the new
        //      president's pre-existing shares for swap — matches Python's
        //      `shares_for_presidency_swap`.
        let pre_action_owners: Vec<EntityId> = self.corporations[corp_idx]
            .shares
            .iter()
            .map(|s| s.owner.clone())
            .collect();
        // Transfer shares to market. Python's SellShares carries the EXACT
        // shares being sold (``[game.share_by_id(id) for id in args["shares"]]``)
        // and ``SharePool.transfer_shares`` moves precisely those certs to the
        // pool (``for share in bundle.shares: self.move_share(share, to_entity)``).
        // The cert ids map directly onto Rust's share-Vec positions: for every
        // 1830 share ``corp.shares[i].index == i`` (set once at init in
        // game.rs and never reordered — only the owner field mutates), so
        // ``share_indices`` ARE the exact certs to move. We must honor those
        // indices on EVERY sell; otherwise Rust's per-cert → owner mapping
        // drifts from Python's recorded ids and a later sell that names a
        // specific id resolves to a cert with a different owner (raising
        // "All shares must be owned by the same owner").
        //
        // The named bundle includes the president cert when it is being dumped;
        // that cert is routed separately (moved last, then
        // ``check_president_change_with_snapshot`` swaps the presidency), so here we move only
        // the named NON-president certs. When no indices are supplied we fall
        // back to owner-based selection by ascending index. The empty-indices
        // fallback is NOT a rare path: the native decode emits empty
        // ``share_indices`` for EVERY SellShares (decode.rs), and the
        // bankruptcy liquidation constructs its own bundles — so this
        // selection must match Python's bundle composition exactly. A dumped
        // bundle's non-president portion is ``percent`` minus the PRESIDENT
        // CERT'S FACE VALUE (20% in 1830) — Python's
        // ``all_bundles_for_corporation`` builds [normals..., president], so
        // the certs moved alongside the president are the remaining
        // ``percent - 20`` of normals (the partial-dump leftover is returned
        // from the pool afterwards, mirroring ``handle_partial``).
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
        {
            let mut to_market: Vec<usize> = Vec::new();
            if !share_indices.is_empty() {
                // Honor the EXACT certs named by the action — Python's
                // SharePool.transfer_shares moves precisely the bundle's
                // non-president certs to the pool (the president cert, if named,
                // is routed separately below; any over-move for a partial
                // president-dump bundle is returned by the partial handling
                // further down). Moving exactly the named ids keeps Rust's
                // per-cert index -> owner map aligned with Python's recorded ids.
                // The seller owns these certs in Python; the president-change
                // snapshot fixes (acquired_seq ordering on every president swap)
                // keep that true in Rust, so identity never drifts onto another
                // owner's cert.
                for &i in share_indices {
                    if i < self.corporations[corp_idx].shares.len()
                        && !self.corporations[corp_idx].shares[i].president
                    {
                        to_market.push(i);
                    }
                }
            } else {
                // No indices supplied (the native-decode path — EVERY decoded
                // SellShares — and the bankruptcy liquidation): take the
                // seller's non-president certs by ascending index until the
                // percent is covered.
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
            // The president cert is always moved last (when dumped).
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
        }

        // Player receives money (units sold × price-per-unit)
        let share_unit = self.corporations[corp_idx].share_unit();
        let revenue = (percent as i32 * share_price.price) / share_unit;
        self.players[player_idx].cash += revenue;
        self.bank.cash -= revenue;

        // Share price drops: move DOWN once per share unit sold (1830 SELL_MOVEMENT = "down_share")
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

        // Record sold corp for this round
        new_state.record_sell(player_id, corporation_sym);
        new_state.acted_this_turn = true;
        new_state.consecutive_passes = 0;

        // Update priority deal: next player after the seller
        new_state.priority_deal_player = self.next_player_id(player_id);

        // Check president change — pass the seller as previous president
        // so the tiebreaker works correctly when the president share is in the market.
        // Pass the pre-action owners snapshot so that swap picks the new president's
        // OLDEST (pre-action) shares first (matches Python's insertion-order).
        self.check_president_change_with_snapshot(corp_idx, Some(player_id), pre_action_owners);

        // Handle partial bundles: when selling with president share, the face value
        // of certs transferred exceeds the bundle percent. Move excess shares back.
        // Python's ``handle_partial`` picks ``share_pool.shares_of(corp)[0]`` —
        // the OLDEST market share by insertion order. We mirror that via the
        // corporation's ``market_order`` queue.
        if includes_president {
            let player_eid_check = EntityId::player(player_id);
            let actual_pct = self.corporations[corp_idx].percent_owned_by(&player_eid_check);
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
                                .set_share_owner(idx, player_eid_check.clone());
                            returned += 1;
                        }
                        None => break,
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
    /// In 1830, shares of corps in cert-limit-exempt zones don't count.
    /// Exempt types: "no_cert_limit", "multiple_buy", "unlimited".
    pub(crate) fn num_certs_internal(&self, player_id: u32) -> u32 {
        let player_eid = EntityId::player(player_id);
        let share_certs: u32 = self
            .corporations
            .iter()
            .filter(|c| {
                // Exclude corps whose share price is in a cert-limit-exempt zone
                !Self::corp_exempt_from_cert_limit(c)
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

    /// Check if a corporation is in a cert-limit-exempt price zone.
    /// In 1830, CERT_LIMIT_TYPES = ["multiple_buy", "unlimited", "no_cert_limit"].
    pub(crate) fn corp_exempt_from_cert_limit(corp: &Corporation) -> bool {
        const EXEMPT_TYPES: &[&str] = &["no_cert_limit", "multiple_buy", "unlimited"];
        corp.share_price
            .as_ref()
            .is_some_and(|sp| sp.types.iter().any(|t| EXEMPT_TYPES.contains(&t.as_str())))
    }

    /// Mirror of Python `Corporation.holding_ok(player, extra_percent=0)`
    /// (entities.py:1319-1323): the ownership (60%) limit is lifted only in the
    /// "multiple_buy"/"unlimited" zones; otherwise the player's common percent of
    /// this corp must be <= max_ownership_percent (60 in 1830). An un-parred corp
    /// (no share_price) holds 0% for the player and is therefore always ok.
    /// NOTE: this is DISTINCT from the cert-limit exemption — `no_cert_limit` is
    /// NOT an ownership exemption (only multiple_buy/unlimited are).
    pub(crate) fn corp_holding_ok(&self, corp: &Corporation, player_id: u32) -> bool {
        const OWNERSHIP_EXEMPT: &[&str] = &["multiple_buy", "unlimited"];
        let ownership_exempt = corp
            .share_price
            .as_ref()
            .is_some_and(|sp| sp.types.iter().any(|t| OWNERSHIP_EXEMPT.contains(&t.as_str())));
        if ownership_exempt {
            return true;
        }
        let player_eid = EntityId::player(player_id);
        // Per-corp cap (1830: 60 everywhere; 1867 minors are single
        // 100%-cert corps with max_ownership_percent 100).
        corp.percent_owned_by(&player_eid) <= corp.max_ownership_percent
    }

    /// Faithful port of Python `BuySellParShares.must_sell(entity)`
    /// (rl18xx/game/engine/round.py:1592-1599). When this returns true, the
    /// Stock-round step offers ONLY `SellShares` — Par/BuyShares/BuyCompany/Pass
    /// are all suppressed.
    ///
    ///   def must_sell(self, entity):
    ///       if not self.can_sell_any(entity): return False
    ///       if self.game.num_certs(entity) > self.game.cert_limit(entity): return True
    ///       if not self.game.can_hold_above_corp_limit(entity):
    ///           return any(not corp.holding_ok(entity) for corp in self.game.corporations)
    ///       return False
    ///
    /// `sellable` is the `can_sell_any(entity)` analog (== `!sellable_bundles().is_empty()`),
    /// passed in by the caller which has already computed it. `certs` is
    /// `num_certs(entity)` (== `num_certs_internal`). For 1830
    /// `can_hold_above_corp_limit` is unconditionally false (base.py:2483), so the
    /// holding-limit branch is always evaluated — encoded explicitly here so the
    /// correspondence to Python is exact and survives a future title that lifts it.
    pub(crate) fn stock_must_sell(&self, player_id: u32, sellable: bool, certs: u32) -> bool {
        if !sellable {
            return false;
        }
        if certs > self.cert_limit as u32 {
            return true;
        }
        if !self.can_hold_above_corp_limit() {
            return self
                .corporations
                .iter()
                .any(|corp| !self.corp_holding_ok(corp, player_id));
        }
        false
    }

    /// Mirror of Python `BaseGame.can_hold_above_corp_limit(entity)`
    /// (rl18xx/game/engine/game/base.py:2483-2484): false for 1830.
    pub(crate) fn can_hold_above_corp_limit(&self) -> bool {
        false
    }

    /// Check if a corporation should float (`float_percent`+ sold from IPO).
    pub(crate) fn check_float(&mut self, corp_idx: usize) {
        let corp = &self.corporations[corp_idx];
        if !corp.floated && corp.check_floated() {
            if let Some(ref ipo_price) = corp.ipo_price {
                match corp.capitalization {
                    // Full capitalization: par × total share units from the
                    // bank on float (1830).
                    crate::title::Capitalization::Full => {
                        let treasury = ipo_price.price * corp.num_share_units();
                        self.corporations[corp_idx].cash = treasury;
                        self.bank.cash -= treasury;
                    }
                    // Incremental capitalization (1867): the treasury accrues
                    // per share SOLD (the buy-side routing below) — floating
                    // brings no lump sum and must not clobber cash already
                    // there (a minor's founding bid).
                    crate::title::Capitalization::Incremental => {}
                }
                self.corporations[corp_idx].floated = true;
                // Home token is placed at the start of the corp's operating turn,
                // not here during the stock round.
            }
        }
    }

    /// Place the corporation's first token on its home city.
    pub(crate) fn place_home_token(&mut self, corp_idx: usize) {
        let corp_defs = self.title_def().corporations();
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
                            self.corporations[corp_idx].home_token_ever_placed = true;
                            break;
                        }
                    }
                }
            }
        }
    }

    /// Check if president should change, given `previous_president` and a
    /// `pre_action_owners` snapshot: for each share index in this corp, the
    /// EntityId that owned it before the current action started. In 1830 the
    /// current president keeps the certificate on ties — a new president is
    /// only assigned when another player has STRICTLY more shares. When the
    /// president changes and we need to pick 2 shares from the new president
    /// to swap, we prefer shares that the new president already owned
    /// (filtered through this snapshot) — this matches Python's
    /// insertion-order semantics
    /// (`shares_for_presidency_swap(P.shares_of(corp)[:2])`).
    pub(crate) fn check_president_change_with_snapshot(
        &mut self,
        corp_idx: usize,
        previous_president: Option<u32>,
        pre_action_owners: Vec<EntityId>,
    ) {
        self.check_president_change_inner(corp_idx, previous_president, Some(pre_action_owners));
    }

    fn check_president_change_inner(
        &mut self,
        corp_idx: usize,
        previous_president: Option<u32>,
        pre_action_owners: Option<Vec<EntityId>>,
    ) {
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

        if max_percent < corp.president_percent() {
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

        // Build preferred swap list (share indices the new president owned BEFORE
        // this action), if a pre-action snapshot was provided. Sort by
        // ``acquired_seq`` so the OLDEST acquisitions are picked first,
        // matching Python's ``possible_reorder(president.shares_of(corporation))``
        // which returns shares in per-owner insertion order.
        let new_pres_eid_calc = EntityId::player(new_pres);
        let preferred_swap: Option<Vec<usize>> = pre_action_owners.as_ref().map(|owners| {
            let mut idxs: Vec<usize> = owners
                .iter()
                .enumerate()
                .filter(|(idx, owner)| {
                    **owner == new_pres_eid_calc
                        && *idx < self.corporations[corp_idx].shares.len()
                        && !self.corporations[corp_idx].shares[*idx].president
                })
                .map(|(idx, _)| idx)
                .collect();
            idxs.sort_by_key(|&i| self.corporations[corp_idx].shares[i].acquired_seq);
            idxs
        });

        if pres_share_in_market {
            let new_pres_eid = EntityId::player(new_pres);
            let market_eid = EntityId::market();

            // Give president share to new president
            if let Some(pres_idx) = self.corporations[corp_idx].president_share_index() {
                self.corporations[corp_idx]
                    .set_share_owner(pres_idx, new_pres_eid.clone());
            }

            // Move 2 normal shares from new president to market.
            // To match Python's insertion-order semantics, prefer shares that the
            // new president owned BEFORE this action (oldest first).
            let mut to_swap: Vec<usize> = Vec::new();
            if let Some(ref preferred) = preferred_swap {
                for &idx in preferred {
                    if to_swap.len() >= 2 {
                        break;
                    }
                    let share = &self.corporations[corp_idx].shares[idx];
                    if share.owner == new_pres_eid && !share.president {
                        to_swap.push(idx);
                    }
                }
            }
            if to_swap.len() < 2 {
                for (i, share) in self.corporations[corp_idx].shares.iter().enumerate() {
                    if to_swap.len() >= 2 {
                        break;
                    }
                    if share.owner == new_pres_eid && !share.president && !to_swap.contains(&i) {
                        to_swap.push(i);
                    }
                }
            }
            for idx in to_swap {
                self.corporations[corp_idx]
                    .set_share_owner(idx, market_eid.clone());
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
                    self.corporations[corp_idx]
                        .set_share_owner(pres_idx, new_pres_eid.clone());
                }

                // Swap 2 normal shares from new president to old president.
                // To match Python's insertion-order semantics, prefer shares that the
                // new president owned BEFORE this action (oldest first).
                let mut to_swap: Vec<usize> = Vec::new();
                if let Some(ref preferred) = preferred_swap {
                    for &idx in preferred {
                        if to_swap.len() >= 2 {
                            break;
                        }
                        let share = &self.corporations[corp_idx].shares[idx];
                        if share.owner == new_pres_eid && !share.president {
                            to_swap.push(idx);
                        }
                    }
                }
                for (i, share) in self.corporations[corp_idx].shares.iter().enumerate() {
                    if to_swap.len() >= 2 {
                        break;
                    }
                    if share.owner == new_pres_eid && !share.president && !to_swap.contains(&i) {
                        to_swap.push(i);
                    }
                }
                for idx in to_swap {
                    self.corporations[corp_idx]
                        .set_share_owner(idx, old_pres_eid.clone());
                }

                self.corporations[corp_idx].owner_id = EntityId::player(new_pres);
            }
        }
    }

    /// After stock round: corps with no shares in IPO or market get price increase.
    ///
    /// Iterates corps in *operating-order* (matches Python's
    /// ``Stock.finish_round`` at round.py:5734 which sorts ``corporations``
    /// before iterating). The iteration order matters: when two corps move
    /// up into the same destination cell, the order they're added to
    /// ``market_cell_corps`` determines the tiebreak in subsequent
    /// operating-order computations.
    pub(crate) fn check_sold_out_price_increases(&mut self) {
        // Compute operating order over *all* floated corps (the property is
        // ``share_price.coordinates``-based — same key as ``compute_operating_order``).
        let order = self.compute_operating_order();
        for sym in &order {
            let corp_idx = match self.corp_idx.get(sym.as_str()) {
                Some(&idx) => idx,
                None => continue,
            };
            let corp = &self.corporations[corp_idx];
            if !corp.floated {
                continue;
            }
            // Ruby Stock#corporations_to_move_price skips :minor types
            // (round/stock.rb:94-96) and G1867::Round::Stock#sold_out?
            // additionally requires :major — 1867 minors (100% player
            // ownable) never get the sold-out bump. 1830-inert (all Major).
            if corp.corp_type != crate::title::CorpType::Major {
                continue;
            }
            let ipo_pct = corp.ipo_shares_percent();
            let market_pct = corp.market_shares_percent();
            if ipo_pct == 0 && market_pct == 0 {
                if let Some(ref sp) = corp.share_price.clone() {
                    // sold_out_stock_movement is move_up (base.rb:1347-1349);
                    // on 1867's 1-D market `up` resolves to RIGHT (core.rs).
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
    pub(crate) fn stock_next_entity(&mut self) {
        // Check if round is finished (all players passed). A pending home
        // token or live auction keeps the round open even when every player
        // has passed (1867: the HomeToken step stays active — Ruby's
        // `finished? = !active_step`): the round WAITS for the blocking
        // action instead of finishing — and must not advance/auto-skip
        // either, or this would recurse forever. `process_stock_home_token`
        // re-runs the finish check once the token lands.
        let (all_passed, overlay_blocked) = match &self.round {
            crate::rounds::Round::Stock(s) => (
                s.all_players_passed(),
                !s.pending_home_tokens.is_empty() || s.bid_auction.is_some(),
            ),
            _ => return,
        };

        if all_passed {
            if overlay_blocked {
                return;
            }
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
        // 1867 MajorTrainless: while trainless majors owe a decision, the
        // step (listed FIRST in the stock list too) blocks the whole round
        // — no player may be auto-passed before the queue drains. A stock
        // round can START frozen (e.g. an 8 exported during the OR→SR
        // transition rusts the 4s); the choice interceptor re-runs this
        // once the queue drains (game.rs process_action_internal).
        if !self.trainless_major.is_empty() {
            return;
        }
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

        // Ruby SELL_AFTER (check_sale_timing): :first (1830) blocks all
        // sales during the first stock round; :operate (1867) gates per
        // corp on `corporation.operated?` (checked inside the loop below).
        // Ruby SELL_BUY_ORDER: in :sell_buy titles (1867) no sell may follow
        // the turn's buy, so a post-buy turn has nothing left.
        let sell_after = self.title_def().sell_after();
        let sell_allowed = match sell_after {
            crate::title::SellAfter::FirstStockRound => self.turn > 1,
            crate::title::SellAfter::Operate => true,
        } && (!bought || self.title_def().stock_sell_after_buy());

        // Can sell?
        // Mirrors Python's bundles_for_corporation + can_sell logic.
        // Enumerates possible sell bundles (including partial president bundles)
        // and checks fit_in_bank + can_dump for each.
        if sell_allowed {
            let can_sell = self.corporations.iter().any(|corp| {
                if corp.ipo_price.is_none() || corp.share_price.is_none() {
                    return false;
                }
                if sell_after == crate::title::SellAfter::Operate && !corp.ever_operated {
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
                let unit_pct = corp.share_unit_percent;
                for n in 1..=num_normal {
                    let bundle_pct = n * unit_pct;
                    if market_pct + bundle_pct as u8 <= 50 {
                        return true; // Can sell n normal shares
                    }
                }

                // Check bundles including the president share (requires can_dump)
                if has_president {
                    let pres_pct = corp.president_percent();
                    let can_dump = self.players.iter().any(|p| {
                        p.id != player_id
                            && corp.percent_owned_by(&EntityId::player(p.id)) >= pres_pct
                    });
                    if can_dump {
                        // Full bundle: all normal + president
                        let full_pct = num_normal * unit_pct + pres_pct;
                        if market_pct + full_pct as u8 <= 50 {
                            return true;
                        }
                        // Partial president bundles: reduce percent by one
                        // unit per step. In 1830: president is 20%, unit is
                        // 10%, so 1 partial bundle at (full_pct - 10)
                        let normal_share_pct = unit_pct; // corp.share_percent in Python
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
            let (bought_corp, parred) = match &self.round {
                crate::rounds::Round::Stock(s) => (
                    s.bought_corp_this_turn.clone(),
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

        // Can par? (parring buys the president's cert: its unit count × par)
        // Minors are never parred directly (1867 founds them by bid) and a
        // title may gate startability by phase (1867 majors: phase 4+).
        let par_prices = self.stock_market.par_prices();
        if let Some(&min_par) = par_prices.iter().min() {
            let title = self.title_def();
            let can_par = self.corporations.iter().any(|c| {
                c.ipo_price.is_none()
                    && c.corp_type != crate::title::CorpType::Minor
                    && title.corporation_startable(&c.sym, &self.phase.name)
                    && player_cash >= min_par * c.president_share_units()
            });
            if can_par {
                return true;
            }
        }

        // 1867: opening a bid auction on a startable minor is a buy action
        // (ViaBid `can_bid_any?`). No-op for titles without minors.
        if let crate::rounds::Round::Stock(s) = &self.round {
            if self.stock_can_bid_minor(s, player_id) {
                return true;
            }
        }

        false
    }
}

#[cfg(test)]
mod g1867_par_tests {
    use crate::game::BaseGame;
    use crate::rounds::{Round, StockState};
    use std::collections::HashMap;

    fn new_4p_1867() -> BaseGame {
        let mut players = HashMap::new();
        players.insert(1, "Alice".to_string());
        players.insert(2, "Bob".to_string());
        players.insert(3, "Carol".to_string());
        players.insert(4, "Dave".to_string());
        BaseGame::build_titled("1867", vec![1, 2, 3, 4], players)
    }

    fn set_phase(game: &mut BaseGame, name: &str) {
        let pd = game
            .title_def()
            .phases()
            .into_iter()
            .find(|p| p.name == name)
            .unwrap();
        game.phase = crate::core::Phase::new(
            pd.name.to_string(),
            pd.operating_rounds,
            pd.train_limit,
            pd.tiles.iter().map(|s| s.to_string()).collect(),
        );
    }

    fn enter_stock_round(game: &mut BaseGame) {
        let order = game.player_order.clone();
        game.round = Round::Stock(StockState::new(&order, order[0]));
        game.update_round_state();
    }

    fn try_par(game: &mut BaseGame, sym: &str, price: i32) -> Result<(), crate::actions::GameError> {
        let state = match &game.round {
            Round::Stock(s) => s.clone(),
            _ => panic!("not in stock round"),
        };
        let pid = state.current_player_id().to_string();
        game.stock_process_par(&state, &pid, sym, price)
    }

    /// The 1867 major par ladder is CLASS-typed, not phase-widened
    /// (g_1867/step/buy_sell_par_shares.rb get_all_par_prices): majors
    /// par on z+p cells $70-200. A $200 par pays the CORP 2 × par
    /// (incremental capitalization) and queues the SR home-token choice
    /// (HOME_TOKEN_TIMING = :par, no fixed home hex); the turn does not
    /// advance until the corp's place_token lands, then auto-advances
    /// (sell_buy — no recorded pass).
    #[test]
    fn major_pars_on_z_cell_pays_corp_and_pends_home_choice() {
        let mut game = new_4p_1867();
        set_phase(&mut game, "4");
        enter_stock_round(&mut game);
        game.players[0].cash = 1000;
        let bank0 = game.bank.cash;

        try_par(&mut game, "CPR", 200).unwrap();

        let ci = game.corp_idx["CPR"];
        assert_eq!(game.corporations[ci].ipo_price.as_ref().unwrap().price, 200);
        assert_eq!(game.corporations[ci].cash, 400, "par pays the corp, not the bank");
        assert_eq!(game.players[0].cash, 1000 - 400);
        assert_eq!(game.bank.cash, bank0);
        let Round::Stock(s) = &game.round else { panic!() };
        assert_eq!(s.pending_home_tokens, vec![("CPR".to_string(), 1)]);
        assert_eq!(s.current_player_id(), 1, "turn waits for the home choice");

        // The parring player chooses the home city; the turn then
        // auto-advances to the next player.
        game.process_stock_home_token("CPR", "M9", 0).unwrap();
        let Round::Stock(s) = &game.round else { panic!() };
        assert!(s.pending_home_tokens.is_empty());
        assert_eq!(s.current_player_id(), 2);
        assert!(game.corporations[ci].tokens[0].used);
    }

    /// Off-ladder prices stay rejected: $50 is a minor-only x cell
    /// (par_1), $145 is no market cell at all, minors are never parred
    /// directly, and majors stay locked before phase 4.
    #[test]
    fn par_ladder_rejections() {
        let mut game = new_4p_1867();
        set_phase(&mut game, "4");
        enter_stock_round(&mut game);
        game.players[0].cash = 1000;

        let err = try_par(&mut game, "CPR", 50).unwrap_err();
        assert!(err.to_string().contains("Invalid par price"), "{}", err);
        let err = try_par(&mut game, "CPR", 145).unwrap_err();
        assert!(err.to_string().contains("Invalid par price"), "{}", err);
        let err = try_par(&mut game, "BBG", 70).unwrap_err();
        assert!(err.to_string().contains("founded by bid"), "{}", err);

        set_phase(&mut game, "3");
        let err = try_par(&mut game, "CPR", 200).unwrap_err();
        assert!(err.to_string().contains("not available in phase"), "{}", err);
    }

    /// The full major ladder per Ruby get_all_par_prices(%i[par_2 par]):
    /// all seven p cells ($70-135) plus the four z cells ($150-200).
    #[test]
    fn major_par_ladder_is_z_plus_p() {
        let game = new_4p_1867();
        let types = game.title_def().par_price_types(crate::title::CorpType::Major);
        let prices: Vec<i32> = game
            .stock_market
            .share_prices_with_types(types)
            .iter()
            .map(|sp| sp.price)
            .collect();
        assert_eq!(prices, vec![200, 180, 165, 150, 135, 120, 110, 100, 90, 80, 70]);
    }
}
