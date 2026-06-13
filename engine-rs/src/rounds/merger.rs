//! 1867's merger round (G1867::Round::Merger + its steps).
//!
//! Runs after EVERY operating round in phases 3-7 (game.rb:876-897
//! `next_round!`). Entities are the floated minors in operating order
//! (round/merger.rb `select_entities` = `merge_corporations.sort`); each
//! gets one turn on the Merge step (step/merge.rb):
//!
//!   * `convert` — a minor whose price is in the convert range announces a
//!     conversion; the follow-up `merge` action names an unfloated major.
//!     Par = the greatest par/par_2 cell ≤ the minor's price; the owner
//!     receives the president cert free and the minor's assets/token move
//!     into the major.
//!   * `merge` — accumulate connected minors (≤10 total, ≤6 per owner) one
//!     action at a time; when candidates run out (or the initiator passes)
//!     the NEXT `merge` names the major. Par = min(200, max(100, cheapest +
//!     priciest)) snapped down to a par/par_2 cell; each merged minor's
//!     owner receives a 10% share (presidency assembled via the 20%-swap),
//!     tokens move with duplicates removed, and the minors close.
//!   * `pass` — done (or, while a merge set is open, "Done Adding
//!     Corporations" → choose the major next).
//!
//! A completed convert/merge replaces the acting minor with the major in
//! the round's entity list and opens the follow-up steps in list order
//! (game.rb:826-837 — the step ORDER is ReduceTokens, PostMergerShares,
//! DiscardTrain, with Merge last):
//!
//!   * ReduceTokens (step/reduce_tokens.rb): while the merged major holds
//!     more than 2 on-map tokens (or 2 in one hex), `remove_token` actions
//!     drop them; afterwards the token list is normalized to ≤2 on-map
//!     tokens and topped back up to 3 with a $40 charter token
//!     (game.rb:794-798 `fix_token_count!`).
//!   * PostMergerShares (step/post_merger_shares.rb): eligible players buy
//!     10% treasury shares at par, paid to the CORP (incremental cap).
//!     Convert: every player in order from the owner, only the old owner
//!     may buy multiple. Merge: the involved owners may buy multiple (rule
//!     9.2 C); when none can buy any more, every player may buy one (rule
//!     9.2 D). A player holding 10% who buys while the president cert is
//!     still in the treasury swaps into the 20% cert (net +10%).
//!   * DiscardTrain: the major discards down to its class train limit.
//!
//! When no step has actions left, the round clears the dealing state and
//! moves to the next minor (round/merger.rb `after_process`); when the
//! entity list is exhausted the round ends and the title's flow function
//! resumes the OR set.

use crate::actions::{Action, GameError};
use crate::entities::EntityId;
use crate::rounds::{MergerState, Round};
use crate::game::BaseGame;
use crate::title::CorpType;

// Merge limits (g_1867/step/merge.rb LIMIT_MERGE / LIMIT_OWNED_BY_ONE_ENTITY).
const LIMIT_MERGE: usize = 10;
const LIMIT_OWNED_BY_ONE_ENTITY: usize = 6;
// Token limit after a merger (game.rb:335 LIMIT_TOKENS_AFTER_MERGER).
const LIMIT_TOKENS_AFTER_MERGER: usize = 2;

impl BaseGame {
    // -- round construction ---------------------------------------------------

    /// Ruby `merge_corporations` (game.rb:417-419) in `select_entities`
    /// order: the floated minors, operating order.
    pub(crate) fn merge_corporations(&self) -> Vec<String> {
        self.compute_operating_order()
            .into_iter()
            .filter(|sym| {
                self.corp_idx
                    .get(sym.as_str())
                    .map_or(false, |&ci| self.corporations[ci].corp_type == CorpType::Minor)
            })
            .collect()
    }

    // -- step predicates (used by steps.rs and the processing below) ----------

    /// Corps over their class train limit (Ruby DiscardTrain `crowded_corps`
    /// computed live — in the merger round only the merged major can be).
    pub(crate) fn merger_crowded_corps(&self) -> Vec<String> {
        self.corporations
            .iter()
            .enumerate()
            .filter(|(ci, c)| c.floated && c.trains.len() > self.corp_train_limit(*ci))
            .map(|(_, c)| c.sym.clone())
            .collect()
    }

    /// Ruby Merge#can_convert?: price in the convert range, type minor.
    pub(crate) fn merger_can_convert(&self, corp_sym: &str) -> bool {
        let Some(&ci) = self.corp_idx.get(corp_sym) else {
            return false;
        };
        let corp = &self.corporations[ci];
        corp.corp_type == CorpType::Minor
            && corp
                .share_price
                .as_ref()
                .map_or(false, |sp| sp.types.iter().any(|t| t == "convert_range"))
    }

    /// Ruby PostMergerShares#eligible_players: the share-dealing rotation
    /// filtered to players who haven't passed and can still buy.
    pub(crate) fn merger_eligible_players(&self, s: &MergerState) -> Vec<u32> {
        s.share_dealing_players
            .iter()
            .copied()
            .filter(|pid| !s.passed_players.contains(pid) && self.merger_can_buy_any(s, *pid))
            .collect()
    }

    /// Ruby PostMergerShares#can_buy_any?: a 10% treasury share of the
    /// converted corp the player can afford and may gain.
    pub(crate) fn merger_can_buy_any(&self, s: &MergerState, pid: u32) -> bool {
        let Some(target) = s.converted.as_deref() else {
            return false;
        };
        self.merger_buyable_share(target, pid).is_some()
    }

    /// The first treasury (IPO-owned) non-president share of `target` that
    /// `pid` can afford and gain — None when no buy is possible.
    fn merger_buyable_share(&self, target: &str, pid: u32) -> Option<usize> {
        let &ci = self.corp_idx.get(target)?;
        let corp = &self.corporations[ci];
        let price = corp.share_price.as_ref()?.price;
        let cash = self.players.iter().find(|p| p.id == pid)?.cash;
        if cash < price {
            return None;
        }
        // can_gain?: cert limit + the corp's ownership cap.
        if self.num_certs_internal(pid) >= self.cert_limit as u32 {
            return None;
        }
        if self.player_percent_of(pid, ci) + corp.share_unit_percent as i32
            > corp.max_ownership_percent as i32
        {
            return None;
        }
        let ipo = EntityId::ipo(target);
        corp.shares
            .iter()
            .position(|sh| sh.owner == ipo && !sh.president)
    }

    // -- action processing ----------------------------------------------------

    pub(crate) fn process_merger_action(&mut self, action: &Action) -> Result<(), GameError> {
        let Round::Merger(state) = &self.round else {
            return Err(GameError::new("Not in a merger round"));
        };
        let mut s = state.clone();

        match action {
            Action::Convert { entity_id } => self.merger_process_convert(&mut s, entity_id)?,
            Action::Merge {
                entity_id,
                corporation_sym,
            } => self.merger_process_merge(&mut s, entity_id, corporation_sym)?,
            Action::Pass { entity_id } => self.merger_process_pass(&mut s, entity_id)?,
            Action::BuyShares {
                entity_id,
                corporation_sym,
                share_indices,
                ..
            } => self.merger_process_buy_shares(&mut s, entity_id, corporation_sym, share_indices)?,
            Action::RemoveToken {
                entity_id,
                hex_id,
                city_index,
                slot,
            } => self.merger_process_remove_token(&mut s, entity_id, hex_id, *city_index, *slot)?,
            Action::DiscardTrain {
                entity_id,
                train_name,
            } => self.merger_process_discard_train(&mut s, entity_id, train_name)?,
            _ => {
                return Err(GameError::new(format!(
                    "Invalid action in merger round: {}",
                    action.action_type()
                )))
            }
        }

        // Ruby Merger#after_process: when no step has anything left, clear
        // the dealing state and move to the next minor.
        self.merger_after_process(&mut s);

        self.round = Round::Merger(s);
        self.update_round_state();
        Ok(())
    }

    fn merger_after_process(&self, s: &mut MergerState) {
        if self.merger_active_step(s) {
            return;
        }
        s.converted = None;
        s.merge_type_convert = false;
        s.share_dealing_players.clear();
        s.share_dealing_multiple.clear();
        s.passed_players.clear();
        self.merger_next_entity(s);
    }

    /// Ruby `round.active_step` over the merger step list: does any step
    /// still have actions? Mirrors PostMergerShares#active_entities, which
    /// lazily runs `check_merge` (the C→D dealing-phase flip) when the
    /// eligible list empties.
    fn merger_active_step(&self, s: &mut MergerState) -> bool {
        if s.corporations_removing_tokens.is_some() {
            return true;
        }
        if s.converted.is_some() {
            if !self.merger_eligible_players(s).is_empty() {
                return true;
            }
            self.merger_check_merge(s);
            if !self.merger_eligible_players(s).is_empty() {
                return true;
            }
        }
        if !self.merger_crowded_corps().is_empty() {
            return true;
        }
        // The Merge step itself (inactive once passed or once a
        // convert/merge completed — `converted` is still set here).
        s.converted.is_none() && !s.merge_passed && s.entity_index < s.entities.len()
    }

    fn merger_next_entity(&self, s: &mut MergerState) {
        s.merge_passed = false;
        s.converting = false;
        debug_assert!(s.merging.is_empty() && !s.merge_major);
        s.entity_index += 1;
        if s.entity_index >= s.entities.len() {
            s.finished = true;
        }
    }

    fn merger_current_entity_check(&self, s: &MergerState, entity_id: &str) -> Result<usize, GameError> {
        if s.current_entity_sym() != Some(entity_id) {
            return Err(GameError::new(format!(
                "Not {}'s turn in the merger round",
                entity_id
            )));
        }
        self.corp_idx
            .get(entity_id)
            .copied()
            .ok_or_else(|| GameError::new(format!("Unknown corporation: {}", entity_id)))
    }

    // -- Merge step -----------------------------------------------------------

    fn merger_process_convert(&mut self, s: &mut MergerState, entity_id: &str) -> Result<(), GameError> {
        self.merger_current_entity_check(s, entity_id)?;
        if s.converting || s.merge_major || !s.merging.is_empty() || s.converted.is_some() {
            return Err(GameError::new("Cannot convert now"));
        }
        if !self.merger_can_convert(entity_id) {
            return Err(GameError::new(format!(
                "{} cannot convert (price not in the convert range)",
                entity_id
            )));
        }
        s.converting = true;
        Ok(())
    }

    fn merger_process_merge(
        &mut self,
        s: &mut MergerState,
        entity_id: &str,
        corporation_sym: &str,
    ) -> Result<(), GameError> {
        self.merger_current_entity_check(s, entity_id)?;
        if s.converting {
            return self.merger_finish_convert(s, entity_id, corporation_sym);
        }
        if s.merge_major {
            return self.merger_finish_merge_to_major(s, corporation_sym);
        }

        // Accumulate a minor into the merge set (Ruby process_merge).
        let candidates = self.merger_mergeable_candidates(s, entity_id);
        if !candidates.iter().any(|c| c == corporation_sym) {
            return Err(GameError::new(format!(
                "Cannot merge with {}",
                corporation_sym
            )));
        }
        if s.merging.is_empty() {
            s.merging.push(entity_id.to_string());
        }
        s.merging.push(corporation_sym.to_string());

        // No more potential merges → the next `merge` names the major
        // (Ruby finish_merge).
        if self.merger_mergeable_candidates(s, entity_id).is_empty() {
            s.merge_major = true;
        }
        Ok(())
    }

    fn merger_process_pass(&mut self, s: &mut MergerState, entity_id: &str) -> Result<(), GameError> {
        // A player pass during post-merger share dealing.
        if let Ok(pid) = entity_id.parse::<u32>() {
            if self.players.iter().any(|p| p.id == pid) {
                return self.merger_post_shares_pass(s, pid);
            }
        }
        // The current minor's pass on the Merge step.
        self.merger_current_entity_check(s, entity_id)?;
        if s.converting || s.merge_major {
            return Err(GameError::new("Must choose a major corporation"));
        }
        if !s.merging.is_empty() {
            // "Done Adding Corporations" → choose the major next.
            s.merge_major = true;
        } else {
            s.merge_passed = true;
        }
        Ok(())
    }

    /// Ruby Merge#mergeable_candidates: minors connected by track to the
    /// current merge set (or to `corp` when no set is open), within the
    /// 10-corp / 6-per-owner limits.
    pub(crate) fn merger_mergeable_candidates(&mut self, s: &MergerState, corp_sym: &str) -> Vec<String> {
        let base: Vec<String> = if s.merging.is_empty() {
            vec![corp_sym.to_string()]
        } else {
            s.merging.clone()
        };
        if base.len() >= LIMIT_MERGE {
            return Vec::new();
        }
        // Owners already contributing 6 minors can't add more.
        let mut owner_counts: Vec<(EntityId, usize)> = Vec::new();
        for sym in &base {
            if let Some(&ci) = self.corp_idx.get(sym.as_str()) {
                let owner = self.corporations[ci].owner_id.clone();
                match owner_counts.iter_mut().find(|(o, _)| *o == owner) {
                    Some((_, n)) => *n += 1,
                    None => owner_counts.push((owner, 1)),
                }
            }
        }
        let owners_at_limit: Vec<EntityId> = owner_counts
            .into_iter()
            .filter(|(_, n)| *n >= LIMIT_OWNED_BY_ONE_ENTITY)
            .map(|(o, _)| o)
            .collect();

        // Corps with a token in a city connected (by track) to any member.
        let mut available: Vec<String> = Vec::new();
        for sym in &base {
            let token_positions = self.corp_token_positions(sym);
            let reservations = self.home_reservations();
            let graph = self.graph_cache.get_or_compute(
                sym,
                &self.hexes,
                &self.hex_idx,
                &self.hex_adjacency,
                &token_positions,
                &reservations,
            );
            let mut connected: Vec<(String, usize)> = graph
                .connected_nodes
                .iter()
                .filter(|n| matches!(n.node_type, crate::map::NodeType::City))
                .map(|n| (n.hex_id.clone(), n.index))
                .collect();
            connected.sort();
            for (hex_id, city_idx) in connected {
                let Some(&hi) = self.hex_idx.get(hex_id.as_str()) else {
                    continue;
                };
                let Some(city) = self.hexes[hi].tile.cities.get(city_idx) else {
                    continue;
                };
                for tok in city.tokens.iter().flatten() {
                    let c = &tok.corporation_id;
                    if !base.iter().any(|b| b == c) && !available.iter().any(|a| a == c) {
                        available.push(c.clone());
                    }
                }
            }
        }
        available.retain(|sym| {
            self.corp_idx.get(sym.as_str()).map_or(false, |&ci| {
                let corp = &self.corporations[ci];
                corp.corp_type == CorpType::Minor
                    && !owners_at_limit.contains(&corp.owner_id)
            })
        });
        available
    }

    /// Ruby Merge#finish_convert: par the major at the greatest par/par_2
    /// cell ≤ the minor's price, hand the president cert to the minor's
    /// owner, move tokens/assets, close the minor and open share dealing
    /// for ALL players (only the old owner may buy multiple).
    fn merger_finish_convert(
        &mut self,
        s: &mut MergerState,
        minor_sym: &str,
        target_sym: &str,
    ) -> Result<(), GameError> {
        self.merger_validate_target(target_sym)?;
        let minor_idx = self.corp_idx[minor_sym];
        let target_idx = self.corp_idx[target_sym];
        let minor_price = self.corporations[minor_idx]
            .share_price
            .as_ref()
            .map(|sp| sp.price)
            .ok_or_else(|| GameError::new("Converting minor has no share price"))?;

        self.merger_set_par(target_idx, minor_price);

        // The owner receives the president cert free.
        let owner_eid = self.corporations[minor_idx].owner_id.clone();
        let owner_pid = owner_eid
            .player_id()
            .ok_or_else(|| GameError::new("Converting minor has no player owner"))?;
        self.corporations[target_idx].set_share_owner(0, owner_eid.clone());
        self.corporations[target_idx].owner_id = owner_eid;

        self.merger_move_tokens(minor_sym, target_sym);
        self.merger_move_assets(minor_idx, target_idx);
        self.merger_close_minor(minor_idx);
        self.check_float(target_idx);

        // Replace the minor with the major in the round.
        s.entities[s.entity_index] = target_sym.to_string();
        s.converting = false;
        s.converted = Some(target_sym.to_string());
        s.merge_type_convert = true;
        // All players are eligible, in order from the new president.
        s.share_dealing_players = self.players_rotated_from(owner_pid);
        s.share_dealing_multiple = vec![owner_pid];
        s.passed_players.clear();
        Ok(())
    }

    /// Ruby Merge#finish_merge_to_major.
    fn merger_finish_merge_to_major(
        &mut self,
        s: &mut MergerState,
        target_sym: &str,
    ) -> Result<(), GameError> {
        self.merger_validate_target(target_sym)?;
        let target_idx = self.corp_idx[target_sym];

        // Sort the merge set into operating order (Ruby `@merging.sort`).
        let order = self.compute_operating_order();
        let mut merging = s.merging.clone();
        merging.sort_by_key(|sym| order.iter().position(|o| o == sym).unwrap_or(usize::MAX));

        // Par = min(200, max(100, cheapest + priciest)) snapped down.
        let prices: Vec<i32> = merging
            .iter()
            .filter_map(|sym| {
                self.corp_idx
                    .get(sym.as_str())
                    .and_then(|&ci| self.corporations[ci].share_price.as_ref().map(|sp| sp.price))
            })
            .collect();
        let (min, max) = (
            *prices.iter().min().unwrap_or(&0),
            *prices.iter().max().unwrap_or(&0),
        );
        let new_price = merged_major_par_target(min, max);
        let merged_par = self
            .stock_market
            .share_prices_with_types(&["par", "par_2"])
            .into_iter()
            .find(|sp| sp.price <= new_price)
            .expect("a major par cell at or below $200 always exists");

        // Involved players in order from the first (operating-order) owner.
        let owners: Vec<u32> = merging
            .iter()
            .filter_map(|sym| {
                self.corp_idx
                    .get(sym.as_str())
                    .and_then(|&ci| self.corporations[ci].owner_id.player_id())
            })
            .collect();
        let mut players: Vec<u32> = self
            .players
            .iter()
            .map(|p| p.id)
            .filter(|pid| owners.contains(pid))
            .collect();
        if let Some(first) = owners.first() {
            if let Some(pos) = players.iter().position(|p| p == first) {
                players.rotate_left(pos);
            }
        }
        // Someone must be able to assemble the 20% presidency (rule check
        // in Ruby BEFORE any state moves).
        let possible = players.iter().any(|pid| {
            let cash = self.players.iter().find(|p| p.id == *pid).map_or(0, |p| p.cash);
            cash >= merged_par.price || owners.iter().filter(|o| *o == pid).count() >= 2
        });
        if !possible {
            return Err(GameError::new(
                "Merge impossible, no player can become president",
            ));
        }

        self.merger_set_par_cell(target_idx, merged_par);

        // Replace the initiating minor with the major in the round.
        s.entities[s.entity_index] = target_sym.to_string();

        // Transfer shares and assets in operating order.
        for minor_sym in &merging {
            let minor_idx = self.corp_idx[minor_sym.as_str()];
            let owner_eid = self.corporations[minor_idx].owner_id.clone();

            // A 10% share to the owner — or the presidency swap when this
            // is the owner's second 10%.
            self.merger_grant_share(target_idx, &owner_eid);

            self.merger_remove_duplicate_tokens(target_sym, &merging);
            self.merger_move_tokens(minor_sym, target_sym);
            self.merger_move_assets(minor_idx, target_idx);
            self.merger_close_minor(minor_idx);
            s.entities.retain(|e| e != minor_sym);
        }
        self.check_float(target_idx);

        if self.merger_tokens_above_limits(target_idx) {
            s.corporations_removing_tokens = Some(vec![target_sym.to_string()]);
        } else {
            self.merger_fix_token_count(target_idx);
        }

        // Deleting entities changed the order — restore it to the target
        // (Ruby `goto_entity!(target)`).
        if let Some(pos) = s.entities.iter().position(|e| e == target_sym) {
            s.entity_index = pos;
        }

        s.merging.clear();
        s.merge_major = false;
        s.converted = Some(target_sym.to_string());
        s.merge_type_convert = false;
        s.share_dealing_players = players.clone();
        s.share_dealing_multiple = players;
        s.passed_players.clear();
        Ok(())
    }

    /// The convert/merge target must be an unfloated major (Ruby
    /// Merge#mergeable for the converting/merge_major phases).
    fn merger_validate_target(&self, target_sym: &str) -> Result<(), GameError> {
        let &ci = self
            .corp_idx
            .get(target_sym)
            .ok_or_else(|| GameError::new(format!("Unknown corporation: {}", target_sym)))?;
        let corp = &self.corporations[ci];
        if corp.corp_type != CorpType::Major || corp.floated || corp.closed {
            return Err(GameError::new(format!(
                "Choose an unstarted major corporation to merge into (not {})",
                target_sym
            )));
        }
        Ok(())
    }

    /// Set the major's par to the greatest par/par_2 cell ≤ `target_price`
    /// (Ruby `share_prices_with_types(%i[par par_2]).find { <= price }`).
    fn merger_set_par(&mut self, corp_idx: usize, target_price: i32) {
        let cell = self
            .stock_market
            .share_prices_with_types(&["par", "par_2"])
            .into_iter()
            .find(|sp| sp.price <= target_price)
            .expect("a par cell at or below the convert range always exists");
        self.merger_set_par_cell(corp_idx, cell);
    }

    fn merger_set_par_cell(&mut self, corp_idx: usize, cell: crate::core::SharePrice) {
        let sym = self.corporations[corp_idx].sym.clone();
        self.corporations[corp_idx].ipo_price = Some(cell.clone());
        self.corporations[corp_idx].share_price = Some(cell.clone());
        self.update_market_cell(&sym, 0, 0, cell.row, cell.column);
        // Initialize all unowned shares as treasury/IPO (the major was
        // never parred through the stock round).
        let ipo = EntityId::ipo(&sym);
        let n = self.corporations[corp_idx].shares.len();
        for i in 0..n {
            if self.corporations[corp_idx].shares[i].owner.is_none() {
                self.corporations[corp_idx].set_share_owner(i, ipo.clone());
            }
        }
    }

    /// Give `owner` a free 10% share of the merged major — the LAST
    /// treasury share (Ruby `target.shares.last`), or the presidency swap:
    /// an owner already holding 10% while the president cert is still in
    /// the treasury returns their share and takes the 20% cert instead.
    fn merger_grant_share(&mut self, target_idx: usize, owner_eid: &EntityId) {
        let target_sym = self.corporations[target_idx].sym.clone();
        let ipo = EntityId::ipo(&target_sym);
        let president_in_treasury = self.corporations[target_idx]
            .shares
            .first()
            .map_or(false, |sh| sh.president && sh.owner == ipo);
        let owner_pid = owner_eid.player_id();
        let owner_percent = owner_pid
            .map(|pid| self.player_percent_of(pid, target_idx))
            .unwrap_or(0);

        if president_in_treasury
            && owner_percent == self.corporations[target_idx].share_unit_percent as i32
        {
            // Swap: the held 10% back to the treasury, the 20% out.
            let held = self.corporations[target_idx]
                .shares_owned_by_in_order(owner_eid)
                .first()
                .copied();
            if let Some(held_idx) = held {
                self.corporations[target_idx].set_share_owner(held_idx, ipo);
            }
            self.corporations[target_idx].set_share_owner(0, owner_eid.clone());
            self.corporations[target_idx].owner_id = owner_eid.clone();
        } else {
            // The last treasury non-president share.
            let share_idx = self.corporations[target_idx]
                .shares
                .iter()
                .rposition(|sh| sh.owner == ipo && !sh.president);
            if let Some(i) = share_idx {
                self.corporations[target_idx].set_share_owner(i, owner_eid.clone());
            }
        }
    }

    // -- asset/token movement and closing --------------------------------------

    /// Ruby Merge#move_tokens: each of the minor's placed tokens is
    /// replaced in its city slot by one of the major's tokens (a fresh $0
    /// token if the charter is empty).
    fn merger_move_tokens(&mut self, from_sym: &str, to_sym: &str) {
        let from_idx = self.corp_idx[from_sym];
        let placed: Vec<String> = self.corporations[from_idx]
            .tokens
            .iter()
            .filter(|t| t.used && !t.city_hex_id.is_empty())
            .map(|t| t.city_hex_id.clone())
            .collect();

        for hex_id in placed {
            let Some(&hi) = self.hex_idx.get(hex_id.as_str()) else {
                continue;
            };
            // Locate the minor's token on the hex (city + slot).
            let mut found: Option<(usize, usize)> = None;
            'outer: for (cidx, city) in self.hexes[hi].tile.cities.iter().enumerate() {
                for (sidx, tok) in city.tokens.iter().enumerate() {
                    if tok.as_ref().map_or(false, |t| t.corporation_id == from_sym) {
                        found = Some((cidx, sidx));
                        break 'outer;
                    }
                }
            }
            let Some((cidx, sidx)) = found else { continue };

            // The major's next charter token, or a fresh $0 one.
            let to_idx = self.corp_idx[to_sym];
            let token_idx = match self.corporations[to_idx].next_token_index() {
                Some(i) => i,
                None => {
                    self.corporations[to_idx]
                        .tokens
                        .push(crate::entities::Token::new(to_sym.to_string(), 0));
                    self.corporations[to_idx].tokens.len() - 1
                }
            };
            self.corporations[to_idx].tokens[token_idx].used = true;
            self.corporations[to_idx].tokens[token_idx].city_hex_id = hex_id.clone();
            let mut new_token = self.corporations[to_idx].tokens[token_idx].clone();
            new_token.used = true;
            new_token.city_hex_id = hex_id.clone();
            self.hexes[hi].tile.cities[cidx].tokens[sidx] = Some(new_token);

            // The minor side: unplace.
            for t in &mut self.corporations[from_idx].tokens {
                if t.used && t.city_hex_id == hex_id {
                    t.used = false;
                    t.city_hex_id = String::new();
                    break;
                }
            }
        }
        self.clear_graph_cache();
    }

    /// Ruby Merge#move_assets: cash, companies, loans and trains move to
    /// the major.
    fn merger_move_assets(&mut self, from_idx: usize, to_idx: usize) {
        let from_eid = EntityId::corporation(&self.corporations[from_idx].sym);
        let to_sym = self.corporations[to_idx].sym.clone();
        let to_eid = EntityId::corporation(&to_sym);

        let cash = self.corporations[from_idx].cash;
        self.corporations[from_idx].cash = 0;
        self.corporations[to_idx].cash += cash;

        for company in &mut self.companies {
            if company.owner == from_eid {
                company.owner = to_eid.clone();
            }
        }

        let loans = self.corporations[from_idx].loans;
        self.corporations[from_idx].loans = 0;
        self.corporations[to_idx].loans += loans;

        let mut trains = std::mem::take(&mut self.corporations[from_idx].trains);
        for t in &mut trains {
            t.owner = to_eid.clone();
        }
        self.corporations[to_idx].trains.extend(trains);
    }

    /// Ruby `close_corporation` for a merged/converted minor: the corp
    /// leaves play — certs vanish, the market cell is vacated. Also the
    /// bookkeeping tail of nationalization's `close_corporation`
    /// (rounds/national.rs), which handles map tokens/trains/companies
    /// first (the merger flow has already moved or removed those).
    pub(crate) fn merger_close_minor(&mut self, corp_idx: usize) {
        let sym = self.corporations[corp_idx].sym.clone();
        if let Some(sp) = self.corporations[corp_idx].share_price.clone() {
            if let Some(corps) = self.market_cell_corps.get_mut(&(sp.row, sp.column)) {
                corps.retain(|c| c != &sym);
            }
        }
        let n = self.corporations[corp_idx].shares.len();
        for i in 0..n {
            self.corporations[corp_idx].set_share_owner(i, EntityId::none());
        }
        let corp = &mut self.corporations[corp_idx];
        corp.floated = false;
        corp.closed = true;
        corp.share_price = None;
        corp.ipo_price = None;
        corp.owner_id = EntityId::none();
        corp.tokens.clear();
    }

    // -- token limits (TokenMerger + G1867::Step::ReduceTokens) ---------------

    /// Ruby TokenMerger#tokens_above_limits? after the minors closed: the
    /// survivor has two tokens on one hex, or more than 2 placed.
    fn merger_tokens_above_limits(&self, corp_idx: usize) -> bool {
        let hexes: Vec<&str> = self.corporations[corp_idx]
            .tokens
            .iter()
            .filter(|t| t.used && !t.city_hex_id.is_empty())
            .map(|t| t.city_hex_id.as_str())
            .collect();
        let mut uniq = hexes.clone();
        uniq.sort();
        uniq.dedup();
        uniq.len() != hexes.len() || hexes.len() > LIMIT_TOKENS_AFTER_MERGER
    }

    /// Ruby game.fix_token_count!: top the charter back up to 3 tokens
    /// with a $40 token.
    fn merger_fix_token_count(&mut self, corp_idx: usize) {
        if self.corporations[corp_idx].tokens.len() == 3 {
            return;
        }
        let sym = self.corporations[corp_idx].sym.clone();
        self.corporations[corp_idx]
            .tokens
            .push(crate::entities::Token::new(sym, 40));
    }

    /// Ruby TokenMerger#remove_duplicate_tokens: drop the survivor's tokens
    /// from cities where a still-unmerged minor also has one.
    fn merger_remove_duplicate_tokens(&mut self, target_sym: &str, merging: &[String]) {
        // Cities (hex, city_idx) holding tokens of the other merging minors.
        let mut other_cities: Vec<(String, usize)> = Vec::new();
        for (hi, hex) in self.hexes.iter().enumerate() {
            for (cidx, city) in hex.tile.cities.iter().enumerate() {
                for tok in city.tokens.iter().flatten() {
                    if merging.iter().any(|m| *m == tok.corporation_id) {
                        other_cities.push((self.hexes[hi].id.clone(), cidx));
                    }
                }
            }
        }
        for (hex_id, cidx) in other_cities {
            let Some(&hi) = self.hex_idx.get(hex_id.as_str()) else {
                continue;
            };
            let city = &mut self.hexes[hi].tile.cities[cidx];
            for slot in city.tokens.iter_mut() {
                if slot.as_ref().map_or(false, |t| t.corporation_id == target_sym) {
                    *slot = None;
                    // Corp side: unplace one matching token.
                    let ti = self.corp_idx[target_sym];
                    if let Some(t) = self.corporations[ti]
                        .tokens
                        .iter_mut()
                        .find(|t| t.used && t.city_hex_id == hex_id)
                    {
                        t.used = false;
                        t.city_hex_id = String::new();
                    }
                }
            }
        }
    }

    // -- ReduceTokens step ------------------------------------------------------

    fn merger_process_remove_token(
        &mut self,
        s: &mut MergerState,
        entity_id: &str,
        hex_ref: &str,
        city_index: u8,
        slot: u8,
    ) -> Result<(), GameError> {
        let Some(removing) = s.corporations_removing_tokens.clone() else {
            return Err(GameError::new("No tokens to remove"));
        };
        if removing.first().map(|s| s.as_str()) != Some(entity_id) {
            return Err(GameError::new(format!(
                "{} is not removing tokens",
                entity_id
            )));
        }
        let corp_idx = self.corp_idx[entity_id];

        // Resolve "__tile:<instance>" refs the way place_token does.
        let hex_id = if let Some(tile_instance) = hex_ref.strip_prefix("__tile:") {
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
            hex_ref.to_string()
        };
        let &hi = self
            .hex_idx
            .get(hex_id.as_str())
            .ok_or_else(|| GameError::new(format!("Unknown hex: {}", hex_id)))?;

        let hexes_with_tokens = self.merger_token_hex_count(corp_idx);

        let city = self.hexes[hi]
            .tile
            .cities
            .get_mut(city_index as usize)
            .ok_or_else(|| GameError::new("Invalid city index"))?;
        let tok = city
            .tokens
            .get_mut(slot as usize)
            .ok_or_else(|| GameError::new("Invalid token slot"))?;
        match tok {
            Some(t) if t.corporation_id == entity_id => {}
            Some(t) => {
                return Err(GameError::new(format!(
                    "Cannot remove {} token",
                    t.corporation_id
                )))
            }
            None => return Err(GameError::new("No token in that slot")),
        }
        *tok = None;
        if let Some(t) = self.corporations[corp_idx]
            .tokens
            .iter_mut()
            .find(|t| t.used && t.city_hex_id == hex_id)
        {
            t.used = false;
            t.city_hex_id = String::new();
        }
        self.clear_graph_cache();

        // 1867: a merger survivor must keep two tokens in DIFFERENT hexes
        // (step/reduce_tokens.rb process_remove_token).
        let now = self.merger_token_hex_count(corp_idx);
        if hexes_with_tokens > 1 && now == 1 {
            return Err(GameError::new(format!(
                "{} must have two tokens in different hexes",
                entity_id
            )));
        }

        if !self.merger_tokens_above_limits(corp_idx) {
            // Ruby move_tokens_to_surviving + fix_token_count!: normalize
            // the charter to the placed tokens (≤2) and top up to 3.
            let placed: Vec<crate::entities::Token> = self.corporations[corp_idx]
                .tokens
                .iter()
                .filter(|t| t.used)
                .cloned()
                .collect();
            let mut unplaced: Vec<crate::entities::Token> = self.corporations[corp_idx]
                .tokens
                .iter()
                .filter(|t| !t.used)
                .cloned()
                .collect();
            unplaced.sort_by_key(|t| t.price);
            let mut rebuilt = placed;
            rebuilt.extend(unplaced);
            rebuilt.truncate(LIMIT_TOKENS_AFTER_MERGER);
            self.corporations[corp_idx].tokens = rebuilt;
            self.merger_fix_token_count(corp_idx);
            s.corporations_removing_tokens = None;
        }
        Ok(())
    }

    fn merger_token_hex_count(&self, corp_idx: usize) -> usize {
        let mut hexes: Vec<&str> = self.corporations[corp_idx]
            .tokens
            .iter()
            .filter(|t| t.used && !t.city_hex_id.is_empty())
            .map(|t| t.city_hex_id.as_str())
            .collect();
        hexes.sort();
        hexes.dedup();
        hexes.len()
    }

    // -- PostMergerShares step ---------------------------------------------------

    fn merger_process_buy_shares(
        &mut self,
        s: &mut MergerState,
        entity_id: &str,
        corporation_sym: &str,
        share_indices: &[usize],
    ) -> Result<(), GameError> {
        let pid: u32 = entity_id
            .parse()
            .map_err(|_| GameError::new("Post-merger shares are bought by players"))?;
        let target = s
            .converted
            .clone()
            .ok_or_else(|| GameError::new("No post-merger share dealing in progress"))?;
        if corporation_sym != target {
            return Err(GameError::new(format!(
                "Only {} shares may be bought now",
                target
            )));
        }
        let corp_idx = self.corp_idx[target.as_str()];
        let ipo = EntityId::ipo(&target);
        let player_eid = EntityId::player(pid);

        let Some(fallback_idx) = self.merger_buyable_share(&target, pid) else {
            return Err(GameError::new(format!(
                "{} cannot buy a {} share",
                pid, target
            )));
        };
        // Honor the recorded cert when it is a buyable treasury share —
        // keeps per-cert identity aligned with Ruby.
        let share_idx = share_indices
            .first()
            .copied()
            .filter(|&i| {
                self.corporations[corp_idx]
                    .shares
                    .get(i)
                    .map_or(false, |sh| sh.owner == ipo && !sh.president)
            })
            .unwrap_or(fallback_idx);

        let price = self.corporations[corp_idx]
            .share_price
            .as_ref()
            .map(|sp| sp.price)
            .unwrap_or(0);

        // Incremental capitalization: treasury-share payments go to the corp.
        let player_idx = self
            .player_index(pid)
            .ok_or_else(|| GameError::new(format!("Unknown player: {}", pid)))?;
        self.players[player_idx].cash -= price;
        self.corporations[corp_idx].cash += price;

        let president_in_treasury = self.corporations[corp_idx]
            .shares
            .first()
            .map_or(false, |sh| sh.president && sh.owner == ipo);
        let unit = self.corporations[corp_idx].share_unit_percent as i32;
        if president_in_treasury && self.player_percent_of(pid, corp_idx) == unit {
            // Ruby's 20% swap: the bought share and the held share return
            // to the treasury, the presidency comes out.
            let held = self.corporations[corp_idx]
                .shares_owned_by_in_order(&player_eid)
                .first()
                .copied();
            if let Some(held_idx) = held {
                self.corporations[corp_idx].set_share_owner(held_idx, ipo.clone());
            }
            self.corporations[corp_idx].set_share_owner(0, player_eid.clone());
            self.corporations[corp_idx].owner_id = player_eid;
        } else {
            self.corporations[corp_idx].set_share_owner(share_idx, player_eid);
        }

        // One-buy players auto-pass; so does anyone who can't buy again.
        if !s.share_dealing_multiple.contains(&pid) || !self.merger_can_buy_any(s, pid) {
            if !s.passed_players.contains(&pid) {
                s.passed_players.push(pid);
            }
        }
        self.merger_check_merge(s);
        Ok(())
    }

    fn merger_post_shares_pass(&mut self, s: &mut MergerState, pid: u32) -> Result<(), GameError> {
        if s.converted.is_none() {
            return Err(GameError::new("No post-merger share dealing in progress"));
        }
        if !s.passed_players.contains(&pid) {
            s.passed_players.push(pid);
        }
        self.merger_check_merge(s);
        // Ruby: don't re-offer the phase-D rotation to the player whose
        // pass ended phase C.
        if self.merger_eligible_players(s).first() == Some(&pid) {
            s.share_dealing_players.retain(|p| *p != pid);
        }
        Ok(())
    }

    /// Ruby PostMergerShares#check_merge: when nobody eligible remains,
    /// either flag the broken merge (president cert unsold) or flip a
    /// merge's dealing from phase C (owners, multiple buys) to phase D
    /// (everyone, one buy).
    fn merger_check_merge(&self, s: &mut MergerState) {
        if !self.merger_eligible_players(s).is_empty() {
            return;
        }
        let Some(target) = s.converted.as_deref() else {
            return;
        };
        let corp_idx = self.corp_idx[target];
        let ipo = EntityId::ipo(target);
        let president_in_treasury = self.corporations[corp_idx]
            .shares
            .first()
            .map_or(false, |sh| sh.president && sh.owner == ipo);
        if president_in_treasury {
            // Ruby logs "Merge failed ... please undo" and stalls for a
            // manual undo — unreachable in valid records (the president
            // check in finish_merge_to_major guards it).
            unimplemented!("1867 merge failed: no player took the 20% presidency (undo-only state)");
        }
        if !s.merge_type_convert && !s.share_dealing_multiple.is_empty() {
            // Rule 9.2 D — everyone may buy one share, in order from the
            // president.
            s.passed_players.clear();
            let president = self.corporations[corp_idx].owner_id.player_id();
            s.share_dealing_players = president
                .map(|pid| self.players_rotated_from(pid))
                .unwrap_or_default();
            s.share_dealing_multiple.clear();
        }
    }

    fn players_rotated_from(&self, first: u32) -> Vec<u32> {
        let mut ids: Vec<u32> = self.players.iter().map(|p| p.id).collect();
        if let Some(pos) = ids.iter().position(|p| *p == first) {
            ids.rotate_left(pos);
        }
        ids
    }

    // -- DiscardTrain step --------------------------------------------------------

    fn merger_process_discard_train(
        &mut self,
        _s: &mut MergerState,
        entity_id: &str,
        train_name: &str,
    ) -> Result<(), GameError> {
        let &corp_idx = self
            .corp_idx
            .get(entity_id)
            .ok_or_else(|| GameError::new(format!("Unknown corporation: {}", entity_id)))?;
        if self.corporations[corp_idx].trains.len() <= self.corp_train_limit(corp_idx) {
            return Err(GameError::new(format!(
                "{} is not over its train limit",
                entity_id
            )));
        }
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
                GameError::new(format!("Train {} not owned by {}", train_name, entity_id))
            })?;
        let mut train = self.corporations[corp_idx].trains.remove(train_idx);
        train.owner = EntityId::none();
        self.depot.discarded.push(train);
        Ok(())
    }
}

/// Merged-major par target (step/merge.rb finish_merge_to_major): the sum
/// of the cheapest and priciest merging minors, clamped to $100..$200; the
/// par cell is then the greatest par/par_2 cell at or below it.
pub(crate) fn merged_major_par_target(min_price: i32, max_price: i32) -> i32 {
    200.min(100.max(min_price + max_price))
}

#[cfg(test)]
mod tests {
    use super::merged_major_par_target;
    use crate::core::StockMarket;
    use crate::title::{g1867, GameTitle, MarketMovement};

    fn market() -> StockMarket {
        StockMarket::new(g1867::G1867.market_grid(), MarketMovement::OneDimensional)
    }

    fn snap_par(target: i32) -> i32 {
        market()
            .share_prices_with_types(&["par", "par_2"])
            .into_iter()
            .find(|sp| sp.price <= target)
            .expect("par cell")
            .price
    }

    /// The 1867 column market's par/par_2 cells are 70..135 (p) and
    /// 150..200 (z); convert/merge snapping always lands on one of them.
    #[test]
    fn merged_par_clamps_and_snaps() {
        // Two cheap minors: 50+55 = 105 -> clamp stays -> snap DOWN to 100.
        assert_eq!(merged_major_par_target(50, 55), 105);
        assert_eq!(snap_par(105), 100);
        // Below the floor: clamped up to exactly 100 (a cell).
        assert_eq!(merged_major_par_target(40, 45), 100);
        assert_eq!(snap_par(100), 100);
        // Rich merge: clamped to the 200 ceiling (a cell).
        assert_eq!(merged_major_par_target(135, 135), 200);
        assert_eq!(snap_par(200), 200);
        // Mid-range: 60+90 = 150 lands exactly on the z cell.
        assert_eq!(snap_par(merged_major_par_target(60, 90)), 150);
        // A gap target snaps down across it: 140-149 -> 135.
        assert_eq!(snap_par(merged_major_par_target(55, 90)), 135);
    }

    /// Conversion par (finish_convert): greatest par/par_2 cell at or
    /// below the minor's price — convert-range prices are 100-165.
    #[test]
    fn convert_par_snaps_down() {
        assert_eq!(snap_par(110), 110);
        assert_eq!(snap_par(165), 165); // 165zCm IS a par_2 cell
        assert_eq!(snap_par(120), 120);
    }
}

/// The ReduceTokens / PostMergerShares step-ORDER fork. Current Ruby lists
/// ReduceTokens first (g_1867/game.rb:826-837, per the GTG errata in
/// tobymao/18xx#9655) — the engine's canonical blocking order, pinned by
/// the vendored fixtures. Games recorded BEFORE the upstream swap dealt
/// shares first; the dispatch gate admits that interleaving while BOTH
/// phases pend (probed on corpus game 20693: current Ruby rejects its own
/// record at file id 366). See steps.rs `step_actions_dispatch`.
#[cfg(test)]
mod g1867_merger_order_tests {
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

    /// The post-merge moment with BOTH phases pending: merged major CNR
    /// holds three placed tokens (above the keep-2 limit ⇒
    /// corporations_removing_tokens) and the share dealing is open
    /// (converted + the dealing rotation). Player 1 took the presidency
    /// (a president cert still in treasury stalls check_merge).
    fn rig_both_pending(game: &mut BaseGame) {
        let ci = game.corp_idx["CNR"];
        let sp = game.stock_market.share_price_at(0, 10).unwrap();
        game.corporations[ci].share_price = Some(sp.clone());
        game.corporations[ci].ipo_price = Some(sp.clone());
        game.update_market_cell("CNR", 0, 10, 0, 10);
        game.corporations[ci].floated = true;
        game.corporations[ci].set_share_owner(0, EntityId::player(1));
        game.corporations[ci].owner_id = EntityId::player(1);
        // The rest of the certs sit in the treasury (merger_set_par_cell
        // initializes a merge target's unowned shares as IPO).
        let ipo = EntityId::ipo("CNR");
        let n = game.corporations[ci].shares.len();
        for i in 1..n {
            game.corporations[ci].set_share_owner(i, ipo.clone());
        }
        for hex in ["M9", "J12", "I15"] {
            let ti = game.corporations[ci].next_token_index().unwrap();
            game.corporations[ci].tokens[ti].used = true;
            game.corporations[ci].tokens[ti].city_hex_id = hex.to_string();
            let tok = game.corporations[ci].tokens[ti].clone();
            let hi = game.hex_idx[hex];
            let city = &mut game.hexes[hi].tile.cities[0];
            let slot = city.tokens.iter().position(|t| t.is_none()).unwrap();
            city.tokens[slot] = Some(tok);
        }
        for p in game.players.iter_mut() {
            p.cash = 1000;
        }
        let mut s = crate::rounds::MergerState::new(1, 2, vec!["CNR".to_string()]);
        s.converted = Some("CNR".to_string());
        s.share_dealing_players = vec![1, 2, 3, 4];
        s.share_dealing_multiple = vec![1, 2, 3, 4];
        s.corporations_removing_tokens = Some(vec!["CNR".to_string()]);
        game.round = crate::rounds::Round::Merger(s);
        game.update_round_state();
    }

    fn buy_shares(pid: u32) -> crate::actions::Action {
        crate::actions::Action::BuyShares {
            entity_id: pid.to_string(),
            corporation_sym: "CNR".into(),
            shares: Vec::new(),
            percent: 10,
            source: "ipo".into(),
            share_indices: vec![1],
        }
    }

    /// Pre-#9655 records: buy_shares / pass while the token removal still
    /// pends are ACCEPTED, change only the dealing state, and the removal
    /// still completes afterwards.
    #[test]
    fn old_order_share_dealing_accepted_while_tokens_pend() {
        let mut game = new_4p_game();
        rig_both_pending(&mut game);
        let price = game.corporations[game.corp_idx["CNR"]]
            .share_price
            .as_ref()
            .unwrap()
            .price;

        // ENUMERATION stays canonical (current rules): the blocking step is
        // ReduceTokens, offering remove_token alone.
        assert_eq!(game.step_action_types_impl(), vec!["remove_token".to_string()]);

        game.process_action_internal(&buy_shares(2)).unwrap();
        assert_eq!(game.players[1].cash, 1000 - price);
        game.process_action_internal(&crate::actions::Action::Pass {
            entity_id: "3".into(),
        })
        .unwrap();

        // Both still pending; the buy went to the treasury share.
        let crate::rounds::Round::Merger(s) = &game.round else {
            panic!("expected merger round")
        };
        assert!(s.corporations_removing_tokens.is_some());
        assert_eq!(s.converted.as_deref(), Some("CNR"));
        assert!(s.passed_players.contains(&3));

        // The removal completes as usual (down to 2 hexes), dealing stays
        // open for the remaining players.
        let slot = {
            let hi = game.hex_idx["M9"];
            game.hexes[hi].tile.cities[0]
                .tokens
                .iter()
                .position(|t| t.as_ref().map_or(false, |t| t.corporation_id == "CNR"))
                .unwrap()
        };
        game.process_action_internal(&crate::actions::Action::RemoveToken {
            entity_id: "CNR".into(),
            hex_id: "M9".into(),
            city_index: 0,
            slot: slot as u8,
        })
        .unwrap();
        let crate::rounds::Round::Merger(s) = &game.round else {
            panic!("expected merger round")
        };
        assert!(s.corporations_removing_tokens.is_none());
        assert_eq!(s.converted.as_deref(), Some("CNR"));
        // fix_token_count!: the charter is topped back up to 3 tokens.
        assert_eq!(game.corporations[game.corp_idx["CNR"]].tokens.len(), 3);
    }

    /// The accommodation is NARROW: without an open share dealing, player
    /// actions at a pending ReduceTokens stay rejected.
    #[test]
    fn dealing_actions_rejected_when_only_tokens_pend() {
        let mut game = new_4p_game();
        rig_both_pending(&mut game);
        if let crate::rounds::Round::Merger(ref mut s) = game.round {
            s.converted = None;
            s.share_dealing_players.clear();
            s.share_dealing_multiple.clear();
        }
        game.update_round_state();
        let err = game.process_action_internal(&buy_shares(2)).unwrap_err();
        assert!(
            err.message
                .contains("Blocking step Choose tokens to remove cannot process action buy_shares"),
            "got: {}",
            err.message
        );
    }
}
