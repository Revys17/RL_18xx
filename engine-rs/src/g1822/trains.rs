//! 1822 trains: the BuyTrain / FirstTurnHousekeeping / DiscardTrain steps
//! (G1822 overrides over Engine::Step::BuyTrain + Train + EmergencyMoney),
//! Depot#min_price / depot_trains / other_trains, Phase#buying_train!
//! (phase advance, rusting, train events) and the G1822 crowded-corp rule.

use crate::actions::{Action, GameError};
use crate::entities::{EntityId, Train};
use crate::game::BaseGame;

use super::shares::Bundle;
use super::{err, Ent, Step, E_TRAIN, EXTRA_TRAINS, EXTRA_TRAIN_PERMANENTS, EXTRA_TRAIN_PULLMAN, UPGRADE_COST_L_TO_2};
use crate::title::g1822 as data;

impl BaseGame {
    pub(crate) fn g1822_is_extra_train(name: &str) -> bool {
        EXTRA_TRAINS.contains(&name)
    }

    /// Phase#train_limit(entity) by corporation type.
    pub(crate) fn g1822_train_limit(&self, c: &str) -> u32 {
        let pd = self
            .title_def()
            .phases()
            .into_iter()
            .find(|p| p.name == self.phase.name)
            .expect("phase");
        if self.is_minor(c) {
            pd.minor_train_limit.unwrap_or(pd.train_limit) as u32
        } else {
            pd.train_limit as u32
        }
    }

    /// G1822 `crowded_corp?`.
    ///
    /// Ruby memoizes `crowded_corps` (reset by buy_train / remove_train /
    /// transfer(:trains) / rust_trains!). The only state change between
    /// resets that alters the answer is the Route step's pullman
    /// attachment, which renames a train "<name>+" (no longer an extra
    /// train when it was a 2P) until run_routes detaches it: the memo
    /// still holds the pre-attachment answer, so judge the attached
    /// train by its original name (233265: "2P+" must not crowd LNWR).
    pub(crate) fn g1822_crowded_corp(&self, c: &str) -> bool {
        let co = &self.corporations[self.ci(c)];
        let attached = self.rnd().pullman_train.as_deref();
        let original = self.rnd().pullman_original.as_ref().map(|(n, _)| n.as_str());
        let name_of = |t: &Train| -> String {
            match (attached, original) {
                (Some(id), Some(orig)) if t.id == id => orig.to_string(),
                _ => t.name.clone(),
            }
        };
        let normal = co.trains.iter().filter(|t| !Self::g1822_is_extra_train(&name_of(t))).count() as u32;
        let perm = co
            .trains
            .iter()
            .filter(|t| EXTRA_TRAIN_PERMANENTS.contains(&name_of(t).as_str()))
            .count();
        let pullman = co.trains.iter().filter(|t| name_of(t) == EXTRA_TRAIN_PULLMAN).count();
        normal > self.g1822_train_limit(c) || perm > 1 || pullman > 1
    }

    /// G1822 `crowded_corps` (all corporations, in @corporations order).
    pub(crate) fn g1822_crowded_corps(&self) -> Vec<String> {
        self.corporations
            .iter()
            .filter(|c| !c.closed && c.ipo_price.is_some())
            .filter(|c| self.g1822_crowded_corp(&c.sym))
            .map(|c| c.sym.clone())
            .collect()
    }

    // -- depot ---------------------------------------------------------------

    /// Depot#depot_trains: the upcoming head (+ phase-available later
    /// upcoming trains, unique by name) + the discard pile unique by name.
    pub(crate) fn g1822_depot_trains(&self) -> Vec<Train> {
        let mut out: Vec<Train> = Vec::new();
        if let Some(head) = self.depot.trains.first() {
            out.push(head.clone());
        }
        for t in &self.depot.trains {
            if self.phase_available(t.available_on.as_deref()) && !out.iter().any(|o| o.name == t.name) {
                out.push(t.clone());
            }
        }
        for t in &self.depot.discarded {
            if !out.iter().any(|o| o.name == t.name && o.owner.is_none()) || !out.iter().any(|o| o.id == t.id) {
                if !self.depot.discarded.iter().take_while(|d| d.id != t.id).any(|d| d.name == t.name) {
                    out.push(t.clone());
                }
            }
        }
        out
    }

    /// Depot#other_trains(corporation): trains owned by OTHER corporations
    /// that are buyable (company-issued permanents are not).
    pub(crate) fn g1822_other_trains(&self, c: &str) -> Vec<(String, Train)> {
        let mut out = Vec::new();
        for co in &self.corporations {
            if co.closed || co.sym == c {
                continue;
            }
            for t in &co.trains {
                if Self::g1822_train_buyable(t) {
                    out.push((co.sym.clone(), t.clone()));
                }
            }
        }
        out
    }

    /// Ruby `train.buyable`: the company-issued 2P/LP/P+ are not.
    pub(crate) fn g1822_train_buyable(t: &Train) -> bool {
        !(t.name == "2P" || t.name == "LP" || t.name == "P+" || t.name.ends_with('+'))
    }

    /// A depot card's variants (name, price): the card's own list, but a
    /// discarded train lost its variants (G1822 DiscardTrain calls
    /// `remove_variants!`): only its current name.
    pub(crate) fn g1822_card_variants(&self, t: &Train) -> Vec<(String, i32)> {
        if self.depot.discarded.iter().any(|d| d.id == t.id) {
            return vec![(t.name.clone(), t.price)];
        }
        let base = Self::g1822_base_name(t);
        let vs: Vec<(String, i32)> = data::train_variants(base)
            .iter()
            .filter_map(|v| data::train_variant(base, v).map(|x| (x.0.to_string(), x.2)))
            .collect();
        if vs.is_empty() {
            vec![(t.name.clone(), t.price)]
        } else {
            vs
        }
    }

    /// Train#min_price for a depot train: min over its card's variants.
    fn g1822_depot_min_price(&self, t: &Train) -> i32 {
        self.g1822_card_variants(t).iter().map(|v| v.1).min().unwrap_or(t.price)
    }

    /// The physical card's base name (Ruby train.sym): "L-3" → "L".
    pub(crate) fn g1822_base_name(t: &Train) -> &str {
        t.id.rsplit_once('-').map(|(n, _)| n).unwrap_or(&t.name)
    }

    /// Depot#min_price(corporation) over depot + other corps' trains.
    fn g1822_min_price(&self, c: &str) -> Option<i32> {
        let mut prices: Vec<i32> = self.g1822_depot_trains().iter().map(|t| self.g1822_depot_min_price(t)).collect();
        if !self.g1822_other_trains(c).is_empty() {
            prices.push(1);
        }
        prices.into_iter().min()
    }

    /// Depot#min_depot_price (min variant price of the cheapest depot card).
    pub(crate) fn g1822_min_depot_price(&self) -> i32 {
        let trains = self.g1822_depot_trains();
        let Some(t) = trains.iter().min_by_key(|t| t.price) else { return 0 };
        self.g1822_depot_min_price(t)
    }

    /// G1822 `discountable_trains_for`: each owned L can be upgraded to a 2
    /// for £80.
    fn g1822_l_upgrades(&self, c: &str) -> Vec<String> {
        let co = &self.corporations[self.ci(c)];
        co.trains.iter().filter(|t| t.name == "L").map(|t| t.id.clone()).collect()
    }

    /// Step::Train#can_buy_train? with the BASE room? (FirstTurn
    /// Housekeeping inherits it).
    pub(crate) fn g1822_can_buy_train_base(&self, c: &str) -> bool {
        let co = &self.corporations[self.ci(c)];
        let room = (co.trains.len() as u32) < self.g1822_train_limit(c);
        self.g1822_can_buy_train_with_room(c, room)
    }

    /// G1822 BuyTrain `can_buy_train?` (room? counts non-extra trains).
    pub(crate) fn g1822_can_buy_train(&self, c: &str) -> bool {
        let co = &self.corporations[self.ci(c)];
        let normal = co.trains.iter().filter(|t| !Self::g1822_is_extra_train(&t.name)).count() as u32;
        let room = normal < self.g1822_train_limit(c);
        self.g1822_can_buy_train_with_room(c, room)
    }

    fn g1822_can_buy_train_with_room(&self, c: &str, room: bool) -> bool {
        let cash = self.corporations[self.ci(c)].cash;
        let normal = room && self.g1822_min_price(c).map_or(false, |m| cash >= m);
        normal || (!self.g1822_l_upgrades(c).is_empty() && cash >= UPGRADE_COST_L_TO_2)
    }

    /// G1822 `must_buy_train?`: no non-extra trains and a depot train.
    pub(crate) fn g1822_must_buy_train(&self, c: &str) -> bool {
        let co = &self.corporations[self.ci(c)];
        co.trains.iter().all(|t| Self::g1822_is_extra_train(&t.name)) && !self.g1822_depot_trains().is_empty()
    }

    /// G1822 BuyTrain#buyable_trains: depot only before 'can_buy_trains'.
    fn g1822_buyable_trains(&self, c: &str) -> Vec<(String, Train)> {
        let mut out: Vec<(String, Train)> = self
            .g1822_depot_trains()
            .into_iter()
            .map(|t| ("depot".to_string(), t))
            .collect();
        if self.phase_has_status("can_buy_trains") {
            let cash = self.corporations[self.ci(c)].cash;
            // EBUY_FROM_OTHERS :never: no other-corp trains with no cash.
            if cash > 0 {
                out.extend(self.g1822_other_trains(c));
            }
        }
        out
    }

    pub(crate) fn g1822_buy_train_actions(&self, ent: &Ent) -> Vec<&'static str> {
        let op = self.g1822_operator();
        match ent {
            Ent::Player(p) => {
                // `['sell_shares'] if entity == current_entity&.owner &&
                // can_ebuy_sell_shares?` (always true for 1822).
                let Some(op) = op else { return vec![] };
                if self.g1822_president(&op) == Some(EntityId::player(*p)) {
                    vec!["sell_shares"]
                } else {
                    vec![]
                }
            }
            Ent::Corp(c) if Some(c) == op.as_ref() => {
                if self.g1822_must_buy_train(c) {
                    vec!["sell_shares", "buy_train"]
                } else if self.g1822_can_buy_train(c) {
                    vec!["buy_train", "pass"]
                } else {
                    vec![]
                }
            }
            _ => vec![],
        }
    }

    pub(crate) fn g1822_buy_train_process(&mut self, i: usize, ent: &Ent, action: &Action) -> Result<(), GameError> {
        match action {
            Action::BuyTrain {
                train_name,
                price,
                variant,
                exchange,
                ..
            } => {
                let Ent::Corp(c) = ent else { return err("buy_train by a non-corporation") };
                let c = c.clone();
                // check_spend: other corps' trains within [1, buying power].
                let owned_by_corp = self.corporations.iter().any(|co| co.trains.iter().any(|t| &t.id == train_name));
                if owned_by_corp && exchange.is_none() {
                    let cash = self.corporations[self.ci(&c)].cash;
                    if *price < 1 || *price > cash {
                        return err(format!("{c} may not spend {price} on that train"));
                    }
                }
                if exchange.is_some() {
                    self.g1822_upgrade_l_train(&c, train_name, *price, variant.as_deref())?;
                } else {
                    if variant.as_deref() == Some(E_TRAIN) {
                        let ci = self.ci(&c);
                        if self.corporations[ci].trains.iter().any(|t| t.name == E_TRAIN) {
                            return err(format!("{c} can only own one E-train"));
                        }
                    }
                    self.g1822_buy_train_action(&c, train_name, *price, variant.as_deref(), Some(i), true)?;
                }
                if !self.g1822_can_buy_train(&c) {
                    self.rndm().passed[i] = true;
                }
                // Phase-1 special case: the first 2 (bought or upgraded)
                // advances the phase (the card's sym is still L).
                let bought_name = self
                    .corporations
                    .iter()
                    .flat_map(|co| co.trains.iter())
                    .find(|t| &t.id == train_name)
                    .map(|t| t.name.clone());
                if self.phase_num() <= 1 && bought_name.as_deref() == Some("2") {
                    self.g1822_phase_advance_on("2");
                }
                Ok(())
            }
            Action::SellShares {
                corporation_sym,
                percent,
                share_indices,
                entity_id,
                ..
            } => {
                let pid: u32 = entity_id.parse().map_err(|_| GameError::new("sell by non-player"))?;
                self.g1822_emergency_sell(pid, corporation_sym, *percent, share_indices)
            }
            Action::Pass { .. } => {
                self.rndm().passed[i] = true;
                self.rndm().last_share_sold_price = None;
                Ok(())
            }
            _ => err("BuyTrain cannot process this action"),
        }
    }

    /// G1822 `upgrade_train_action`: the L card flips to its 2 side; the
    /// corp pays the bank.
    fn g1822_upgrade_l_train(&mut self, c: &str, train_id: &str, price: i32, variant: Option<&str>) -> Result<(), GameError> {
        let ci = self.ci(c);
        let Some(t) = self.corporations[ci].trains.iter().find(|t| t.id == train_id) else {
            return err(format!("{c} does not own {train_id}"));
        };
        if t.name != "L" {
            return err("Only an L train can be upgraded");
        }
        let v = variant.unwrap_or("2");
        let (name, dist, vprice, _) = data::train_variant("L", v).ok_or_else(|| GameError::new("bad variant"))?;
        self.corp_spend(c, price)?;
        let t = self.corporations[ci].trains.iter_mut().find(|t| t.id == train_id).unwrap();
        t.name = name.to_string();
        t.distance = dist;
        t.price = vprice;
        Ok(())
    }

    /// Step::Train#buy_train_action (+ Game#buy_train + Phase#buying_train!).
    pub(crate) fn g1822_buy_train_action(
        &mut self,
        c: &str,
        train_id: &str,
        price: i32,
        variant: Option<&str>,
        step: Option<usize>,
        president_may_contribute: bool,
    ) -> Result<(), GameError> {
        // Locate the train: depot head/discards or another corporation.
        let in_depot = self.g1822_depot_trains().iter().any(|t| t.id == train_id);
        let seller: Option<String> = self
            .corporations
            .iter()
            .find(|co| co.trains.iter().any(|t| t.id == train_id))
            .map(|co| co.sym.clone());
        if !in_depot && seller.is_none() {
            return err(format!("Not a buyable train: {train_id}"));
        }
        if seller.as_deref() == Some(c) {
            return err("An entity cannot buy a train from itself");
        }
        if let Some(s) = &seller {
            if !self.g1822_buyable_trains(c).iter().any(|(o, t)| o == s && t.id == train_id) {
                return err(format!("Not a buyable train: {train_id}"));
            }
        }
        let mut train = if let Some(s) = &seller {
            let sci = self.ci(s);
            let pos = self.corporations[sci].trains.iter().position(|t| t.id == train_id).unwrap();
            self.corporations[sci].trains[pos].clone()
        } else if let Some(pos) = self.depot.trains.iter().position(|t| t.id == train_id) {
            self.depot.trains[pos].clone()
        } else {
            let pos = self.depot.discarded.iter().position(|t| t.id == train_id).unwrap();
            self.depot.discarded[pos].clone()
        };
        // train.variant = action.variant (a discarded card has no variants)
        let base = Self::g1822_base_name(&train).to_string();
        let discarded = self.depot.discarded.iter().any(|d| d.id == train.id);
        if let Some(v) = variant {
            if discarded {
                if v != train.name {
                    return err(format!("{v} is not a variant of {}", train.id));
                }
            } else if let Some((name, dist, vprice, _)) = data::train_variant(&base, v) {
                train.name = name.to_string();
                train.distance = dist;
                train.price = vprice;
            } else if v != train.name {
                return err(format!("{v} is not a variant of {}", train.id));
            }
        }
        let cash = self.corporations[self.ci(c)].cash;
        let remaining = price - cash;
        if remaining > 0 && president_may_contribute && self.g1822_must_buy_train(c) {
            // check_for_cheapest_train (EBUY_DEPOT_TRAIN_MUST_BE_CHEAPEST,
            // EBUY_FROM_OTHERS :never): with the president's money only
            // the cheapest depot card's cheapest variant(s) may be bought.
            let depot = self.g1822_depot_trains();
            if let Some(cheap) = depot.iter().min_by_key(|t| t.price) {
                let vs = self.g1822_card_variants(cheap);
                let min = vs.iter().map(|v| v.1).min().unwrap_or(0);
                let names: Vec<String> = vs.into_iter().filter(|v| v.1 == min).map(|v| v.0).collect();
                if !names.contains(&train.name) {
                    return err(format!(
                        "Cannot purchase {} train: cheaper train available ({})",
                        train.name, names[0]
                    ));
                }
            }
            if price > train.price && seller.is_none() {
                return err("Cannot buy for more than cost");
            }
            let pres = self
                .g1822_president(c)
                .and_then(|e| e.player_id())
                .ok_or_else(|| GameError::new("no president"))?;
            self.g1822_try_take_player_loan(pres, remaining)?;
            let pi = self.player_index(pres).unwrap();
            if self.players[pi].cash < remaining {
                return err("President cannot contribute enough");
            }
            self.players[pi].cash -= remaining;
            let ci = self.ci(c);
            self.corporations[ci].cash += remaining;
        }
        // Game#buy_train: operator pays the train's owner.
        self.corp_spend_to(c, seller.as_deref(), price)?;
        // remove from wherever, give to the buyer
        if let Some(s) = &seller {
            let sci = self.ci(s);
            self.corporations[sci].trains.retain(|t| t.id != train_id);
        } else {
            self.depot.trains.retain(|t| t.id != train_id);
            self.depot.discarded.retain(|t| t.id != train_id);
        }
        train.owner = EntityId::corporation(c);
        train.operated = false;
        let ci = self.ci(c);
        self.corporations[ci].trains.push(train.clone());
        self.g1822_phase_buying_train(&train, seller.as_deref());
        if let Some(i) = step {
            if !self.g1822_can_buy_train(c) {
                self.rndm().passed[i] = true;
            }
        }
        Ok(())
    }

    /// Operator → seller (bank for depot trains).
    fn corp_spend_to(&mut self, c: &str, seller: Option<&str>, amount: i32) -> Result<(), GameError> {
        let ci = self.ci(c);
        if self.corporations[ci].cash < amount {
            return err(format!("{c} cannot afford {amount}"));
        }
        self.corporations[ci].cash -= amount;
        match seller {
            Some(s) => {
                let si = self.ci(s);
                self.corporations[si].cash += amount;
            }
            None => self.g1822_bank_receive(amount),
        }
        Ok(())
    }

    /// EmergencyMoney#try_take_player_loan (EBUY_CAN_TAKE_PLAYER_LOAN
    /// :after_sell — only once nothing is left to sell).
    fn g1822_try_take_player_loan(&mut self, pid: u32, cost: i32) -> Result<(), GameError> {
        let cash = self.player_cash(pid);
        if cost <= 0 || cost <= cash {
            return Ok(());
        }
        if self.g1822_sellable_value(pid) > 0 {
            return err("still needs to sell shares before a loan can be granted");
        }
        let amount = cost - cash;
        // take_cash_loan: bank pays, debt += amount × 1.5 (ceil).
        self.g1822_bank_pay(&EntityId::player(pid), amount);
        let debt = amount as i64 + (amount as i64 * super::PLAYER_LOAN_INTEREST_RATE as i64 + 99) / 100;
        let e = self.gm().player_debt.entry(pid).or_insert(0);
        *e = (*e as i64 + debt).min(i32::MAX as i64) as i32;
        Ok(())
    }

    /// liquidity(player, emergency: true) - cash: the best sellable
    /// bundle value per corporation.
    fn g1822_sellable_value(&self, pid: u32) -> i32 {
        let eid = EntityId::player(pid);
        let mut v = 0;
        for co in &self.corporations {
            if co.closed {
                continue;
            }
            let best = self
                .g1822_bundles_for(&eid, &co.sym)
                .into_iter()
                .filter(|b| self.g1822_emr_can_sell(pid, b))
                .map(|b| self.g1822_bundle_price(&b))
                .max()
                .unwrap_or(0);
            v += best;
        }
        v
    }

    /// EmergencyMoney#can_sell? (EBUY_SELL_MORE_THAN_NEEDED false).
    pub(crate) fn g1822_emr_can_sell(&self, pid: u32, b: &Bundle) -> bool {
        let eid = EntityId::player(pid);
        let c = &self.corporations[self.ci(&b.corp)];
        if b.shares.iter().any(|&i| c.shares[i].owner != eid) {
            return false;
        }
        if !self.g1822_check_sale_timing(&b.corp) {
            return false;
        }
        if !self.g1822_can_dump(&eid, b) || !self.g1822_fit_in_bank(b) {
            return false;
        }
        // president_swap_concern?: EBUY_PRES_SWAP true → only for the
        // operating corp itself.
        if self.g1822_president(&b.corp) == Some(eid.clone())
            && self.g1822_operator().as_deref() == Some(b.corp.as_str())
        {
            let holders = self.g1822_player_share_holders(&b.corp);
            let mine = holders.iter().find(|(e, _)| *e == eid).map_or(0, |(_, p)| *p) as i32;
            let next = holders.iter().filter(|(e, _)| *e != eid).map(|(_, p)| *p).max().unwrap_or(0) as i32;
            if mine - (b.percent as i32) < next {
                return false;
            }
        }
        // selling_minimum_shares?: the next smaller bundle is insufficient.
        let Some(op) = self.g1822_operator() else { return true };
        let need = self.g1822_min_depot_price() - (self.corporations[self.ci(&op)].cash + self.player_cash(pid));
        let smallest = b
            .shares
            .iter()
            .map(|&i| {
                self.g1822_bundle_price(&Bundle {
                    corp: b.corp.clone(),
                    shares: vec![i],
                    percent: c.shares[i].percent,
                    price_override: None,
                })
            })
            .min()
            .unwrap_or(0);
        self.g1822_bundle_price(b) - smallest < need
    }

    fn g1822_emergency_sell(&mut self, pid: u32, sym: &str, percent: u8, idx: &[usize]) -> Result<(), GameError> {
        let b = self.g1822_action_bundle_pub(sym, percent, idx)?;
        if !self.g1822_emr_can_sell(pid, &b) {
            return err(format!("Cannot sell shares of {sym}"));
        }
        let pps = self.g1822_price_per_share(sym, &EntityId::player(pid));
        self.rndm().last_share_sold_price = Some(pps);
        self.g1822_sell_shares_and_change_price(&b);
        self.rndm().corporations_sold.push(sym.to_string());
        // recalculate_order: re-sort the not-yet-operated tail.
        let idx_now = self.rnd().entity_index;
        let mut tail: Vec<String> = self.rnd().entities[idx_now + 1..].to_vec();
        let order = self.g1822_operating_order();
        tail.sort_by_key(|s| order.iter().position(|o| o == s).unwrap_or(usize::MAX));
        let r = self.rndm();
        r.entities.truncate(idx_now + 1);
        r.entities.extend(tail);
        Ok(())
    }

    // -- discard ------------------------------------------------------------

    pub(crate) fn g1822_process_discard_train(&mut self, c: &str, train_id: &str) -> Result<(), GameError> {
        let ci = self.ci(c);
        let pos = self.corporations[ci]
            .trains
            .iter()
            .position(|t| t.id == train_id)
            .ok_or_else(|| GameError::new(format!("{c} does not own {train_id}")))?;
        let mut t = self.corporations[ci].trains.remove(pos);
        let removed = EXTRA_TRAIN_PERMANENTS.contains(&t.name.as_str()) || t.name == EXTRA_TRAIN_PULLMAN;
        if !removed {
            // remove_variants! + reclaim_train → the discard pile.
            t.owner = EntityId::none();
            t.operated = false;
            self.depot.discarded.push(t);
        }
        Ok(())
    }

    // -- phase / rust / events ---------------------------------------------

    /// Advance phases while the next phase's `on` names `sym`.
    pub(crate) fn g1822_phase_advance_on(&mut self, sym: &str) {
        loop {
            let defs = self.title_def().phases();
            let idx = defs.iter().position(|p| p.name == self.phase.name).unwrap_or(0);
            let Some(next) = defs.get(idx + 1) else { break };
            // Phase 2 is `on: %w[2 3]`.
            let matches = next.on == Some(sym) || (next.name == "2" && sym == "3");
            if !matches {
                break;
            }
            self.phase = crate::core::Phase::new(
                next.name.to_string(),
                next.operating_rounds,
                next.train_limit,
                next.tiles.iter().map(|s| s.to_string()).collect(),
            );
        }
    }

    /// Phase#buying_train!(entity, train, source).
    pub(crate) fn g1822_phase_buying_train(&mut self, train: &Train, _source: Option<&str>) {
        let sym = Self::g1822_base_name(train).to_string();
        self.g1822_phase_advance_on(&sym);
        self.g1822_rust_trains(&sym);
        // Events fire for the type's first card (index 0) only.
        if train.id.ends_with("-0") {
            let key = format!("{sym}:events");
            if !self.events_fired.contains(&key) {
                self.events_fired.push(key);
                let events = self
                    .title_def()
                    .trains()
                    .into_iter()
                    .find(|t| t.name == sym)
                    .map(|t| t.events)
                    .unwrap_or(&[]);
                for e in events {
                    self.g1822_event(e);
                }
            }
        }
    }

    /// rust_trains!: every train whose rusts_on is the bought sym.
    fn g1822_rust_trains(&mut self, sym: &str) {
        let rusts = |t: &Train| -> bool {
            let base = Self::g1822_base_name(t);
            let rusts_on = if t.name == base {
                data::trains().into_iter().find(|d| d.name == base).and_then(|d| d.rusts_on)
            } else {
                data::train_variant(base, &t.name).and_then(|v| v.3)
            };
            rusts_on == Some(sym)
        };
        for co in &mut self.corporations {
            co.trains.retain(|t| !rusts(t));
        }
        self.depot.trains.retain(|t| !rusts(t));
        self.depot.discarded.retain(|t| !rusts(t));
    }

    fn g1822_event(&mut self, event: &str) {
        match event {
            "close_concessions" => {
                let concs: Vec<String> = self
                    .companies
                    .iter()
                    .filter(|c| c.sym.starts_with('C') && !c.closed)
                    .map(|c| c.sym.clone())
                    .collect();
                for c in concs {
                    self.close_company(&c);
                }
                let majors: Vec<String> = self
                    .corporations
                    .iter()
                    .filter(|c| !c.closed && !c.floated && c.corp_type == crate::title::CorpType::Major)
                    .map(|c| c.sym.clone())
                    .collect();
                for m in majors {
                    self.gm().par_via_exchange.remove(&m);
                    let ci = self.ci(&m);
                    self.corporations[ci].float_percent = 50;
                }
            }
            "full_capitalisation" => {
                for co in &mut self.corporations {
                    if !co.closed && !co.floated && co.corp_type == crate::title::CorpType::Major {
                        co.capitalization = crate::title::Capitalization::Full;
                    }
                }
            }
            "phase_revenue" => {
                for c in super::PRIVATE_PHASE_REVENUE {
                    if self.co_closed(c) {
                        continue;
                    }
                    let Some(Some(cash)) = self.g().phase_revenue.get(*c).cloned() else { continue };
                    if cash > 0 {
                        self.g1822_bank_receive(cash);
                    }
                    self.gm().phase_revenue.insert(c.to_string(), None);
                    self.close_company(c);
                }
            }
            other => panic!("1822: unknown train event {other}"),
        }
    }

    /// Ruby `remove_train` + `close_corporation` for minors that leave play.
    pub(crate) fn g1822_close_corporation(&mut self, sym: &str) {
        let ci = self.ci(sym);
        // CLOSED_CORP_TOKENS_REMOVED: tokens leave the map.
        for hex in &mut self.hexes {
            for city in &mut hex.tile.cities {
                for t in city.tokens.iter_mut() {
                    if t.as_ref().is_some_and(|tok| tok.corporation_id == sym) {
                        *t = None;
                    }
                }
            }
        }
        // CLOSED_CORP_RESERVATIONS_REMOVED.
        self.g1822_remove_reservations(sym);
        // set_cash(0, bank)
        let cash = self.corporations[ci].cash;
        self.corporations[ci].cash = 0;
        self.bank.cash += cash;
        // CLOSED_CORP_TRAINS_REMOVED: trains unbuyable — drop them.
        self.corporations[ci].trains.clear();
        let owned: Vec<String> = self.companies_of(&EntityId::corporation(sym));
        for c in owned {
            self.close_company(&c);
        }
        // shares leave their holders
        let n = self.corporations[ci].shares.len();
        for i in 0..n {
            self.corporations[ci].set_share_owner(i, EntityId::none());
        }
        if let Some(sp) = self.corporations[ci].share_price.clone() {
            if let Some(v) = self.market_cell_corps.get_mut(&(sp.row, sp.column)) {
                v.retain(|c| c != sym);
            }
        }
        self.corporations[ci].closed = true;
        self.corporations[ci].floated = false;
        self.corporations[ci].owner_id = EntityId::none();
        for t in &mut self.corporations[ci].tokens {
            t.used = false;
            t.city_hex_id.clear();
        }
        self.clear_graph_cache();
    }

    /// The operating round may need to skip a closed current entity
    /// (Ruby force_next_entity! when the closing corp is the current one).
    #[allow(dead_code)]
    pub(crate) fn g1822_force_next_entity(&mut self) {
        let n = self.rnd().steps.len();
        self.rndm().passed = vec![true; n];
    }

    #[allow(dead_code)]
    fn g1822_step_index(&self, s: Step) -> Option<usize> {
        self.rnd().step_index(s)
    }
}
