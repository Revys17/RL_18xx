//! Share ownership for 1822 — a port of Ruby's SharePool#buy_shares /
//! sell_shares / transfer_shares (presidency changes included),
//! Game#bundles_for_corporation and ShareBundle#can_dump?.
//!
//! Owners: `player:<id>`, `ipo:<sym>` (the corporation's treasury — 1822 is
//! incremental, Ruby's ipo_owner IS the corporation), `market`, and the
//! tax haven (`taxhaven:P16`, a Ruby Player: it counts as a player share
//! holder for presidency).

use crate::actions::GameError;
use crate::entities::EntityId;
use crate::game::BaseGame;

use super::{err, TAX_HAVEN_OWNER};

/// A share bundle (Ruby ShareBundle): certificate indices + percent (the
/// percent may be below the certs' sum for a partial president sale).
#[derive(Clone, Debug)]
pub struct Bundle {
    pub corp: String,
    pub shares: Vec<usize>,
    pub percent: u8,
    /// Ruby `bundle.share_price` (an action may name the price, e.g. the
    /// reduced price of issued shares); None = the share's own price.
    pub price_override: Option<i32>,
}

impl BaseGame {
    pub(crate) fn g1822_owner_is_player_like(owner: &EntityId) -> bool {
        owner.is_player() || owner.0 == TAX_HAVEN_OWNER
    }

    /// Ruby `share_holder.percent_of(corporation)`.
    pub(crate) fn g1822_percent_of(&self, owner: &EntityId, corp: &str) -> u8 {
        let c = &self.corporations[self.ci(corp)];
        c.percent_owned_by(owner)
    }

    /// Ruby `corporation.player_share_holders` (players + the tax haven),
    /// in a stable order: seated players, then the tax haven.
    pub(crate) fn g1822_player_share_holders(&self, corp: &str) -> Vec<(EntityId, u8)> {
        let c = &self.corporations[self.ci(corp)];
        let mut out = Vec::new();
        for p in &self.players {
            let e = EntityId::player(p.id);
            let pct = c.percent_owned_by(&e);
            if pct > 0 {
                out.push((e, pct));
            }
        }
        let th = EntityId(TAX_HAVEN_OWNER.to_string());
        let pct = c.percent_owned_by(&th);
        if pct > 0 {
            out.push((th, pct));
        }
        out
    }

    /// Ruby `corporation.owner` for majors/minors: the president's holder.
    pub(crate) fn g1822_president(&self, corp: &str) -> Option<EntityId> {
        let c = &self.corporations[self.ci(corp)];
        if c.owner_id.is_none() {
            None
        } else {
            Some(c.owner_id.clone())
        }
    }

    /// Ruby `bundle.price` (price_per_share × num_shares, ceiled).
    pub(crate) fn g1822_bundle_price(&self, b: &Bundle) -> i32 {
        let c = &self.corporations[self.ci(&b.corp)];
        let pps = b.price_override.unwrap_or_else(|| {
            self.g1822_price_per_share(&b.corp, &self.corporations[self.ci(&b.corp)].shares[b.shares[0]].owner)
        });
        let num = b.percent as f64 / c.share_unit_percent as f64;
        (pps as f64 * num).ceil() as i32
    }

    /// Ruby `Share#price_per_share`: always_market_price → share price;
    /// an un-parred corp prices at `min_price` (the cheapest par cell).
    pub(crate) fn g1822_price_per_share(&self, corp: &str, _owner: &EntityId) -> i32 {
        let c = &self.corporations[self.ci(corp)];
        match c.share_price.as_ref() {
            Some(sp) => sp.price,
            None => self
                .stock_market
                .par_prices()
                .first()
                .copied()
                .unwrap_or(0),
        }
    }

    /// Ruby `Game#bundles_for_corporation(share_holder, corp)`
    /// (all_bundles_for_corporation): cumulative bundles over the holder's
    /// shares sorted non-president first by percent, plus the partial
    /// president bundles; sorted by percent.
    pub(crate) fn g1822_bundles_for(&self, owner: &EntityId, corp: &str) -> Vec<Bundle> {
        let c = &self.corporations[self.ci(corp)];
        if c.ipo_price.is_none() {
            return Vec::new();
        }
        let mut shares = c.shares_owned_by_in_order(owner);
        if shares.is_empty() {
            return Vec::new();
        }
        // sort_by { [president ? 1 : 0, percent] } — stable on ties.
        shares.sort_by_key(|&i| (c.shares[i].president as u8, c.shares[i].percent));
        let mut bundles = Vec::new();
        let mut acc: Vec<usize> = Vec::new();
        let mut pct: u32 = 0;
        for &i in &shares {
            acc.push(i);
            pct += c.shares[i].percent as u32;
            bundles.push(Bundle {
                corp: corp.to_string(),
                shares: acc.clone(),
                percent: pct as u8,
                price_override: None,
            });
        }
        if c.shares[*shares.last().unwrap()].president {
            let normal = c.share_unit_percent as u32;
            let difference = c.president_percent() as u32 - normal;
            let n = difference / normal;
            for k in 1..=n {
                bundles.push(Bundle {
                    corp: corp.to_string(),
                    shares: acc.clone(),
                    percent: (pct - normal * k) as u8,
                    price_override: None,
                });
            }
        }
        // sort_by(&:percent) — Ruby's sort_by is not stable, but percents
        // here are distinct except a partial president bundle tying a
        // plain one; keep the build order for ties.
        bundles.sort_by_key(|b| b.percent);
        bundles
    }

    /// Ruby `ShareBundle#can_dump?(entity)`.
    pub(crate) fn g1822_can_dump(&self, owner: &EntityId, b: &Bundle) -> bool {
        let c = &self.corporations[self.ci(&b.corp)];
        let pres = b.shares.iter().find(|&&i| c.shares[i].president);
        let Some(&pi) = pres else { return true };
        // player_share_holders(corporate: true): players + corps (not self).
        let mut best = 0u8;
        for (e, pct) in self.g1822_player_share_holders(&b.corp) {
            if &e != owner {
                best = best.max(pct);
            }
        }
        for other in &self.corporations {
            if other.sym == b.corp {
                continue;
            }
            let e = EntityId::corporation(&other.sym);
            let pct = c.percent_owned_by(&e);
            if &e != owner {
                best = best.max(pct);
            }
        }
        best >= c.shares[pi].percent
    }

    /// SharePool#fit_in_bank?: pool percent + bundle ≤ 50.
    pub(crate) fn g1822_fit_in_bank(&self, b: &Bundle) -> bool {
        let c = &self.corporations[self.ci(&b.corp)];
        b.percent as u32 + c.percent_owned_by(&EntityId::market()) as u32
            <= self.title_def().market_pool_limit() as u32
    }

    /// Move certificates to `to` (Ruby move_share, appending to the new
    /// owner's per-corp list).
    pub(crate) fn g1822_move_shares(&mut self, corp: &str, shares: &[usize], to: &EntityId) {
        let ci = self.ci(corp);
        for &i in shares {
            self.corporations[ci].set_share_owner(i, to.clone());
        }
    }

    /// Ruby SharePool#transfer_shares, minus the money (callers pay).
    /// `prev_president` is the corporation's owner BEFORE the transfer.
    pub(crate) fn g1822_transfer_shares(
        &mut self,
        b: &Bundle,
        to: &EntityId,
        allow_president_change: bool,
    ) {
        let ci = self.ci(&b.corp);
        let owner = self.corporations[ci].shares[b.shares[0]].owner.clone();
        let previous_president = self.g1822_president(&b.corp);
        let bundle_has_pres = b.shares.iter().any(|&i| self.corporations[ci].shares[i].president);
        let bundle_sum: u32 = b.shares.iter().map(|&i| self.corporations[ci].shares[i].percent as u32).sum();
        let partial = bundle_sum != b.percent as u32;
        self.g1822_move_shares(&b.corp, &b.shares, to);
        if !allow_president_change {
            return;
        }
        let holders = self.g1822_player_share_holders(&b.corp);
        let max_shares = holders.iter().map(|(_, p)| *p).max().unwrap_or(0);
        let majority: Vec<EntityId> = holders
            .iter()
            .filter(|(_, p)| *p == max_shares)
            .map(|(e, _)| e.clone())
            .collect();
        if let Some(pp) = &previous_president {
            if majority.iter().any(|e| e == pp) {
                return;
            }
        }
        let pres_cert = self.corporations[ci].president_share_index();
        let presidents_percent = self.corporations[ci].president_percent();
        let mut president: Option<EntityId> = None;
        if previous_president.is_none() {
            let pres_owner = pres_cert.map(|i| self.corporations[ci].shares[i].owner.clone());
            president = majority.iter().find(|e| Some(*e) == pres_owner.as_ref()).cloned();
        }
        if president.is_none() {
            let order: Vec<EntityId> = self.players.iter().map(|p| EntityId::player(p.id)).collect();
            let distance = |a: &Option<EntityId>, bb: &EntityId| -> usize {
                let Some(a) = a else { return 0 };
                let ia = order.iter().position(|e| e == a);
                let ib = order.iter().position(|e| e == bb);
                match (ia, ib) {
                    (Some(ia), Some(ib)) => {
                        if ia < ib {
                            ib - ia
                        } else {
                            ib + order.len() - ia
                        }
                    }
                    _ => usize::MAX,
                }
            };
            let candidates: Vec<&EntityId> = majority
                .iter()
                .filter(|e| self.corporations[ci].percent_owned_by(e) >= presidents_percent)
                .collect();
            // Ruby's min_by ranks every candidate by SharePool#distance over
            // possible_presidents (the seated players): a tax haven among
            // them makes `a < b` compare with nil, an ArgumentError, so
            // Ruby refuses the action. Mirror that as a rejection.
            if previous_president.is_some() && candidates.iter().any(|e| e.0 == TAX_HAVEN_OWNER) {
                self.gm().pending_error = Some(format!(
                    "{}: the presidency would be ranked against the tax haven (Ruby raises)",
                    b.corp
                ));
            }
            president = candidates
                .into_iter()
                .min_by_key(|e| distance(&previous_president, e))
                .cloned();
        }
        let Some(president) = president else { return };
        self.corporations[ci].owner_id = president.clone();

        if *to == president
            && previous_president.as_ref() == Some(&owner)
            && bundle_has_pres
            && !partial
        {
            return;
        }
        let mut previous_president = previous_president;
        if owner.0 == format!("ipo:{}", b.corp) && !bundle_has_pres {
            // can_swap_for_presidents_share_directly_from_corporation?
            if previous_president.is_none() {
                previous_president = Some(owner.clone());
            }
        }
        if owner == president || previous_president.is_none() {
            return;
        }
        let previous_president = previous_president.unwrap();
        // presidents_share = bundle.presidents_share || previous president's
        let pres_idx = if bundle_has_pres {
            b.shares.iter().copied().find(|&i| self.corporations[ci].shares[i].president)
        } else {
            pres_cert.filter(|&i| self.corporations[ci].shares[i].owner == previous_president)
        };
        let Some(pres_idx) = pres_idx else { return };
        let pres_pct = self.corporations[ci].shares[pres_idx].percent;
        let (transfer_to, swap_to) = if (owner.is_player() && to.is_player()) && bundle_has_pres {
            (to.clone(), to.clone())
        } else {
            let transfer_to = EntityId::market();
            let swap_to = if self.corporations[ci].percent_owned_by(&previous_president) >= pres_pct {
                previous_president.clone()
            } else {
                transfer_to.clone()
            };
            (transfer_to, swap_to)
        };
        // change_president: the new president hands num_shares of their
        // OWN shares (insertion order, first N) to swap_to, takes the cert.
        let unit = self.corporations[ci].share_unit_percent;
        let num = (pres_pct / unit) as usize;
        let mine: Vec<usize> = self.corporations[ci]
            .shares_owned_by_in_order(&president)
            .into_iter()
            .filter(|&i| i != pres_idx)
            .take(num)
            .collect();
        self.g1822_move_shares(&b.corp, &mine, &swap_to);
        self.g1822_move_shares(&b.corp, &[pres_idx], &president);
        if partial {
            // handle_partial(bundle, transfer_to, owner)
            let difference = bundle_sum - b.percent as u32;
            let n = difference / unit as u32;
            for _ in 0..n {
                if let Some(&first) = self.corporations[ci].shares_owned_by_in_order(&transfer_to).first() {
                    self.g1822_move_shares(&b.corp, &[first], &owner);
                }
            }
        }
    }

    /// Ruby `SharePool#buy_shares` (no exchange): price to the treasury
    /// for treasury certs (incremental), to the bank for pool certs.
    pub(crate) fn g1822_pool_buy_shares(
        &mut self,
        buyer: &EntityId,
        b: &Bundle,
        exchange_price: Option<i32>,
        is_exchange: bool,
        allow_president_change: bool,
    ) -> Result<(), GameError> {
        let ci = self.ci(&b.corp);
        let owner = self.corporations[ci].shares[b.shares[0]].owner.clone();
        let was_floated = self.corporations[ci].floated;
        let price = if is_exchange {
            exchange_price.unwrap_or(0)
        } else {
            self.g1822_bundle_price(b)
        };
        if price > 0 {
            // spender = buyer; receiver = treasury (owner is the corp's
            // IPO / a corporation) or player, else the bank.
            self.g1822_entity_spend(buyer, price)?;
            // SharePool#buy_shares receiver: the treasury only while the
            // corp is incrementally capitalized (full capitalisation — 1822
            // phases 6+ before floating — pays the bank).
            let incremental = self.corporations[ci].capitalization == crate::title::Capitalization::Incremental;
            if owner.is_ipo() && incremental {
                self.corporations[ci].cash += price;
            } else if let Some(pid) = owner.player_id() {
                self.pay_player(pid, price);
            } else {
                self.g1822_bank_receive(price);
            }
        }
        self.g1822_transfer_shares(b, buyer, allow_president_change);
        self.g1822_check_float(&b.corp, was_floated);
        Ok(())
    }

    /// Ruby `@game.share_pool.sell_shares(bundle)`: bank pays the seller,
    /// shares go to the pool.
    pub(crate) fn g1822_pool_sell_shares(&mut self, b: &Bundle, allow_president_change: bool) {
        let ci = self.ci(&b.corp);
        let owner = self.corporations[ci].shares[b.shares[0]].owner.clone();
        let price = self.g1822_bundle_price(b);
        self.g1822_bank_pay(&owner, price);
        self.g1822_transfer_shares(b, &EntityId::market(), allow_president_change);
    }

    /// Ruby `sell_shares_and_change_price` with SELL_MOVEMENT :down_share.
    pub(crate) fn g1822_sell_shares_and_change_price(&mut self, b: &Bundle) {
        let ci = self.ci(&b.corp);
        let unit = self.corporations[ci].share_unit_percent as u32;
        let num_shares = (b.percent as u32).div_ceil(unit);
        self.g1822_pool_sell_shares(b, true);
        for _ in 0..num_shares {
            self.g1822_move_price(&b.corp, Dir::Down);
        }
    }

    /// Spend from a player / corp / tax haven to the bank-side receiver
    /// (the caller credits the receiver).
    pub(crate) fn g1822_entity_spend(&mut self, who: &EntityId, amount: i32) -> Result<(), GameError> {
        if let Some(pid) = who.player_id() {
            let i = self.player_index(pid).expect("player");
            if self.players[i].cash < amount {
                return err(format!(
                    "{} has {} and cannot spend {}",
                    self.players[i].name, self.players[i].cash, amount
                ));
            }
            self.players[i].cash -= amount;
            Ok(())
        } else if who.0 == TAX_HAVEN_OWNER {
            if self.g().tax_haven_cash < amount {
                return err("Tax haven cannot afford");
            }
            self.gm().tax_haven_cash -= amount;
            Ok(())
        } else if let Some(sym) = who.corp_sym() {
            let ci = self.ci(sym);
            if self.corporations[ci].cash < amount {
                return err(format!("{} cannot afford {}", sym, amount));
            }
            self.corporations[ci].cash -= amount;
            Ok(())
        } else {
            err(format!("unknown spender {}", who.0))
        }
    }

    /// Bank pays an entity (latches the bank break on overdraft).
    pub(crate) fn g1822_bank_pay(&mut self, who: &EntityId, amount: i32) {
        if amount <= 0 {
            return;
        }
        self.g1822_bank_spend(amount);
        if let Some(pid) = who.player_id() {
            self.pay_player(pid, amount);
        } else if who.0 == TAX_HAVEN_OWNER {
            self.gm().tax_haven_cash += amount;
        } else if let Some(sym) = who.corp_sym() {
            let ci = self.ci(sym);
            self.corporations[ci].cash += amount;
        }
    }

    /// Ruby Bank#spend's check_cash: overdrawing breaks the bank (latched).
    pub(crate) fn g1822_bank_spend(&mut self, amount: i32) {
        if self.bank.cash - amount < 0 {
            self.bank_broken = true;
        }
        self.bank.cash -= amount;
    }

    pub(crate) fn g1822_bank_receive(&mut self, amount: i32) {
        self.bank.cash += amount;
    }

    /// Ruby `float_corporation` hook after a buy (`floatable && floated
    /// changed`): G1822#float_corporation.
    pub(crate) fn g1822_check_float(&mut self, corp: &str, was_floated: bool) {
        let ci = self.ci(corp);
        if self.corporations[ci].floated {
            return;
        }
        // Corporation#floated?: treasury percent ≤ 100 - float_percent.
        let ipo = self.corporations[ci].percent_owned_by(&EntityId::ipo(corp)) as i32;
        if ipo > 100 - self.corporations[ci].float_percent as i32 {
            return;
        }
        self.corporations[ci].floated = true;
        if was_floated {
            return;
        }
        self.g1822_float_corporation(corp);
    }

    /// G1822 `float_corporation`: in full-capitalisation phases a major's
    /// treasury cash returns to the bank and its remaining treasury shares
    /// move to the market, then base float (incremental: no lump sum…
    /// unless capitalization is :full), then capitalization reverts to
    /// incremental for majors.
    pub(crate) fn g1822_float_corporation(&mut self, corp: &str) {
        let ci = self.ci(corp);
        let major = self.corporations[ci].corp_type == crate::title::CorpType::Major;
        if self.phase_has_status("full_capitalisation") && major {
            let cash = self.corporations[ci].cash;
            if cash > 0 {
                self.corporations[ci].cash = 0;
                self.g1822_bank_receive(cash);
            }
            let treasury: Vec<usize> = self.corporations[ci].shares_owned_by_in_order(&EntityId::ipo(corp));
            self.g1822_move_shares(corp, &treasury, &EntityId::market());
        }
        if self.corporations[ci].capitalization == crate::title::Capitalization::Full {
            let par = self.corporations[ci].ipo_price.as_ref().map_or(0, |p| p.price);
            let total = self.corporations[ci].num_share_units();
            let amount = par * total;
            self.g1822_bank_spend(amount);
            self.corporations[ci].cash += amount;
        }
        if major {
            self.corporations[ci].capitalization = crate::title::Capitalization::Incremental;
        }
    }

    // -- market movement ----------------------------------------------------

    /// Move a corp's share price one step, maintaining the cell order.
    pub(crate) fn g1822_move_price(&mut self, corp: &str, dir: Dir) {
        let ci = self.ci(corp);
        let Some(sp) = self.corporations[ci].share_price.clone() else { return };
        let (r, c) = match dir {
            Dir::Up => self.stock_market.move_up(sp.row, sp.column),
            Dir::Down => self.stock_market.move_down(sp.row, sp.column),
            Dir::Left => self.stock_market.move_left(sp.row, sp.column),
            Dir::Right => self.stock_market.move_right(sp.row, sp.column),
        };
        if (r, c) == (sp.row, sp.column) {
            return;
        }
        if let Some(new_sp) = self.stock_market.share_price_at(r, c) {
            if new_sp.types.iter().any(|t| t == "endgame") {
                self.gm().stock_market_max_reached = true;
            }
            self.corporations[ci].share_price = Some(new_sp);
            self.update_market_cell(corp, sp.row, sp.column, r, c);
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dir {
    Up,
    Down,
    Left,
    Right,
}
