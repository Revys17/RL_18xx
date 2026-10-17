//! 1822 stock round (G1822::Round::Stock + G1822::Step::BuySellParShares
//! with the BidboxAuction mixin) and the choices round (G1822::Round::
//! Choices + G1822::Step::Choose).

use crate::actions::{Action, GameError};
use crate::entities::EntityId;
use crate::game::BaseGame;
use crate::title::CorpType;

use super::shares::Bundle;
use super::{
    err, BidRec, CurAct, Ent, RoundKind, RoundSt, SoldWhen, Step, BIDDING_TOKENS_PER_ACTION,
    COMPANY_DOUBLE_CASH, COMPANY_OSTH, MINOR_START_PAR_PRICE, MIN_BID_INCREMENT,
    PLAYER_LOAN_INTEREST_RATE, TAX_HAVEN_OWNER,
};

impl BaseGame {
    // =====================================================================
    // Round lifecycle
    // =====================================================================

    /// `new_stock_round` → G1822::Round::Stock.new + `@round.setup`.
    pub(crate) fn g1822_start_stock_round(&mut self) {
        let entities: Vec<String> = self
            .players
            .iter()
            .filter(|p| !p.bankrupt)
            .map(|p| p.id.to_string())
            .collect();
        let round = RoundSt::new(
            RoundKind::Stock,
            1,
            entities,
            vec![Step::DiscardTrain, Step::BuySellParShares],
        );
        self.gm().round = round;
        // Round::Base#initialize runs each step's setup.
        self.g1822_bsps_setup();
        // G1822::Round::Stock#setup: setup_bidboxes, then Stock#setup.
        self.setup_bidboxes();
        self.g1822_skip_steps();
        if self.g1822_active_step().is_none() {
            // A finished? round is handled by the caller's transition loop;
            // otherwise move to the next entity.
            let _ = self.g1822_stock_next_entity();
        }
    }

    /// BSPS#setup (+ G1822 bidbox setup).
    fn g1822_bsps_setup(&mut self) {
        let r = self.rndm();
        for corps in r.players_sold.values_mut() {
            for w in corps.values_mut() {
                *w = SoldWhen::Prev;
            }
        }
        r.current_actions.clear();
        r.bought_from_ipo = false;
        r.bid_actions = 0;
    }

    fn g1822_set_player_passed(&mut self, pid: u32, passed: bool) {
        let v = &mut self.gm().round_passed_players;
        v.retain(|&p| p != pid);
        if passed {
            v.push(pid);
        }
    }

    /// Stock#finished?: every player entity has passed.
    pub(crate) fn g1822_stock_finished(&self) -> bool {
        if self.finished {
            return true;
        }
        let passed = &self.g().round_passed_players;
        self.rnd()
            .entities
            .iter()
            .all(|e| e.parse::<u32>().map_or(false, |p| passed.contains(&p)))
    }

    /// Stock#next_entity!.
    pub(crate) fn g1822_stock_next_entity(&mut self) -> Result<(), GameError> {
        if self.g1822_stock_finished() {
            self.g1822_next_entity_index();
            return self.g1822_stock_finish_round();
        }
        self.g1822_next_entity_index();
        self.g1822_stock_start_entity()
    }

    fn g1822_next_entity_index(&mut self) {
        let r = self.rndm();
        if !r.entities.is_empty() {
            r.entity_index = (r.entity_index + 1) % r.entities.len();
        }
    }

    /// Stock#start_entity.
    fn g1822_stock_start_entity(&mut self) -> Result<(), GameError> {
        let n = self.rnd().steps.len();
        let r = self.rndm();
        r.passed = vec![false; n];
        r.acted = vec![false; n];
        self.g1822_bsps_setup();
        self.g1822_skip_steps();
        if self.g1822_active_step().is_none() {
            return self.g1822_stock_next_entity();
        }
        Ok(())
    }

    /// G1822::Round::Stock#finish_round.
    fn g1822_stock_finish_round(&mut self) -> Result<(), GameError> {
        if self.g().nothing_sold_in_sr {
            self.g1822_end_game("dnf");
            return Ok(());
        }
        let mut float_minors: Vec<(BidRec, String, usize)> = Vec::new();
        let mut remove_l_count = 0u32;
        let mut remove_minor: Option<String> = None;
        let boxed_minors = self.bidbox_minors();
        let minor_count = boxed_minors.len();
        for (index, minor) in boxed_minors.iter().enumerate() {
            if let Some(bid) = self.g1822_highest_bid(minor) {
                float_minors.push((bid, minor.clone(), index));
            } else {
                // minor.owner = nil
                self.gm().bank_owned.retain(|c| c != minor);
                remove_l_count += 1;
                if index == 0 {
                    remove_minor = Some(minor.clone());
                }
            }
        }
        for conc in self.bidbox_concessions() {
            if let Some(bid) = self.g1822_highest_bid(&conc) {
                self.g1822_buy_company(&bid, &conc)?;
            } else {
                self.gm().bank_owned.retain(|c| c != &conc);
            }
        }
        for co in self.bidbox_privates() {
            if let Some(bid) = self.g1822_highest_bid(&co) {
                self.g1822_buy_company(&bid, &co)?;
            } else {
                self.gm().bank_owned.retain(|c| c != &co);
            }
        }
        // sort_by { [price, minor_count - index] }.reverse_each: highest bid
        // first; ties → lower index first.
        float_minors.sort_by_key(|(b, _, idx)| (b.price, minor_count as i64 - *idx as i64));
        for (bid, company, _) in float_minors.into_iter().rev() {
            self.g1822_float_minor(&bid, &company)?;
        }
        if remove_l_count > 0
            && self.depot.trains.first().map_or(false, |t| t.name == "L")
        {
            self.g1822_remove_l_trains(remove_l_count);
        }
        if let Some(m) = remove_minor {
            self.g1822_remove_minor_and_first_train(&m);
        }
        self.bidbox_minors_refill();
        self.g1822_add_interest_player_loans();
        // Base Stock#finish_round: sold-out majors move up (G1822 sold_out?
        // is majors only); POOL_SHARE_DROP :none.
        let mut corps: Vec<String> = self
            .corporations
            .iter()
            .filter(|c| !c.closed && c.floated && c.corp_type != CorpType::Minor)
            .map(|c| c.sym.clone())
            .collect();
        self.g1822_sort_operating(&mut corps);
        for sym in corps {
            if self.g1822_sold_out(&sym) {
                self.g1822_move_price(&sym, super::shares::Dir::Up);
            }
        }
        Ok(())
    }

    /// Base `sold_out?` (player share holders incl. the tax haven hold
    /// 100%) restricted to majors.
    fn g1822_sold_out(&self, sym: &str) -> bool {
        if !self.is_major(sym) {
            return false;
        }
        let total: u32 = self
            .g1822_player_share_holders(sym)
            .iter()
            .map(|(_, p)| *p as u32)
            .sum();
        total == 100
    }

    /// Ruby `G1822::Round::Stock#buy_company`.
    fn g1822_buy_company(&mut self, bid: &BidRec, company: &str) -> Result<(), GameError> {
        let ci = self.co_idx(company).expect("company");
        self.companies[ci].owner = EntityId::player(bid.player);
        self.gm().bank_owned.retain(|c| c != company);
        if bid.price > 0 {
            self.player_spend(bid.player, bid.price)?;
        }
        Ok(())
    }

    /// `minor_float_share_price(bid)`.
    pub(crate) fn g1822_minor_float_share_price(&self, bid_price: i32) -> crate::core::SharePrice {
        let mut pars = self.stock_market.share_prices_with_types(&["par", "par_1"]);
        // par_prices sort: [price, column, row] descending.
        pars.sort_by(|a, b| (b.price, b.column, b.row).cmp(&(a.price, a.column, a.row)));
        let max_par = pars.iter().map(|p| p.price).max().unwrap_or(0);
        let mut target = if self.phase_num() == 1 {
            MINOR_START_PAR_PRICE
        } else {
            bid_price / 2
        };
        if target > max_par {
            target = max_par;
        }
        pars.into_iter()
            .find(|p| p.price <= target)
            .expect("a par price for the minor")
    }

    /// `minor_float_starting_cash`.
    fn g1822_minor_float_starting_cash(&self, bid_amount: i32, presidency_price: i32) -> i32 {
        if self.phase_num() < 3 {
            presidency_price
        } else {
            bid_amount
        }
    }

    /// G1822::Round::Stock#float_minor.
    fn g1822_float_minor(&mut self, bid: &BidRec, company: &str) -> Result<(), GameError> {
        let minor = Self::minor_of_company(company);
        let share_price = self.g1822_minor_float_share_price(bid.price);
        let presidency_price = share_price.price * 2;
        // par_corporation + buy the 100% presidency from the treasury.
        self.g1822_par_corporation(&minor, share_price);
        let player = EntityId::player(bid.player);
        let b = Bundle {
            corp: minor.clone(),
            shares: vec![0],
            percent: 100,
            price_override: None,
        };
        self.g1822_pool_buy_shares(&player, &b, None, false, true)?;
        // after_par: nothing for 1822 (no :shares companies, HOME_TOKEN
        // timing :operate).
        // "Clear the corporation of par cash".
        let mci = self.ci(&minor);
        let cash = self.corporations[mci].cash;
        if cash > 0 {
            self.corporations[mci].cash = 0;
            self.g1822_bank_receive(cash);
        }
        let treasury = self.g1822_minor_float_starting_cash(bid.price, presidency_price);
        let excess = bid.price - presidency_price;
        if excess != 0 {
            // player.spend(excess, bank) — a negative excess (a phase-1 bid
            // under the 100 presidency can't happen: min bid is 100).
            self.player_spend(bid.player, excess)?;
        }
        self.g1822_bank_spend(treasury);
        self.corporations[mci].cash += treasury;
        self.delete_company(company);
        Ok(())
    }

    /// Base `par_corporation` (STOCK_MARKET_TOKEN_TIMING :par).
    pub(crate) fn g1822_par_corporation(&mut self, sym: &str, sp: crate::core::SharePrice) {
        let ci = self.ci(sym);
        self.corporations[ci].ipo_price = Some(sp.clone());
        self.corporations[ci].share_price = Some(sp.clone());
        self.update_market_cell(sym, u8::MAX, u8::MAX, sp.row, sp.column);
        let ipo = EntityId::ipo(sym);
        let n = self.corporations[ci].shares.len();
        for i in 0..n {
            if self.corporations[ci].shares[i].owner.is_none() {
                self.corporations[ci].set_share_owner(i, ipo.clone());
            }
        }
    }

    /// `remove_l_trains(count)`.
    fn g1822_remove_l_trains(&mut self, mut count: u32) {
        while count > 0 && self.depot.trains.first().map_or(false, |t| t.name == "L") {
            self.depot.trains.remove(0);
            count -= 1;
        }
    }

    /// `remove_minor_and_first_train(company)`: export the next train (as
    /// if bought: phase change, events, rusting) and close the minor.
    fn g1822_remove_minor_and_first_train(&mut self, company: &str) {
        if !self.depot.trains.is_empty() {
            let train = self.depot.trains.remove(0);
            self.g1822_phase_buying_train(&train, None);
        }
        let minor = Self::minor_of_company(company);
        self.g1822_close_corporation(&minor);
        self.close_company(company);
        self.delete_company(company);
    }

    /// `add_interest_player_loans!` (50%, ceiled).
    fn g1822_add_interest_player_loans(&mut self) {
        let debts: Vec<(u32, i32)> = self.g().player_debt.iter().map(|(&p, &d)| (p, d)).collect();
        for (p, d) in debts {
            if d > 0 {
                let added = (d * PLAYER_LOAN_INTEREST_RATE + 99) / 100;
                self.gm().player_debt.insert(p, d + added);
            }
        }
    }

    /// `reorder_players` (most cash, the P9 double-cash chooser doubled;
    /// ties keep the current order).
    pub(crate) fn g1822_reorder_players(&mut self) {
        let double = self.g().double_cash_choice;
        let n = self.players.len();
        let mut keyed: Vec<(i64, usize, crate::entities::Player)> = self
            .players
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let mut cash = p.cash as i64;
                if Some(p.id) == double {
                    cash *= 2;
                }
                // current_order = players.reverse → index n-1-i
                (cash, n - 1 - i, p.clone())
            })
            .collect();
        keyed.sort_by_key(|(c, idx, _)| (*c, *idx));
        keyed.reverse();
        self.players = keyed.into_iter().map(|(_, _, p)| p).collect();
        self.player_order = self.players.iter().map(|p| p.id).collect();
        self.gm().double_cash_choice = None;
    }

    // =====================================================================
    // BuySellParShares (G1822)
    // =====================================================================

    /// `committed_cash(player)`: the player's WINNING bids.
    pub(crate) fn g1822_committed_cash(&self, pid: u32) -> i32 {
        self.g1822_winning_bids(pid).iter().map(|b| b.price).sum()
    }

    /// `bids_for_player(player, only_committed_bids: true)`.
    fn g1822_winning_bids(&self, pid: u32) -> Vec<BidRec> {
        self.rnd()
            .bids
            .values()
            .filter_map(|bids| {
                let high = Self::max_bid_of(bids)?;
                (high.player == pid).then(|| high.clone())
            })
            .collect()
    }

    /// Ruby `bids.max_by(&:price)` (first max wins ties).
    fn max_bid_of(bids: &[BidRec]) -> Option<&BidRec> {
        let mut best: Option<&BidRec> = None;
        for b in bids {
            if best.map_or(true, |x| b.price > x.price) {
                best = Some(b);
            }
        }
        best
    }

    pub(crate) fn g1822_highest_bid(&self, company: &str) -> Option<BidRec> {
        self.rnd()
            .bids
            .get(company)
            .and_then(|b| Self::max_bid_of(b))
            .cloned()
    }

    fn g1822_available_cash(&self, who: &EntityId) -> i32 {
        if let Some(pid) = who.player_id() {
            self.player_cash(pid) - self.g1822_committed_cash(pid)
        } else if who.0 == TAX_HAVEN_OWNER {
            self.g().tax_haven_cash
        } else {
            0
        }
    }

    /// Ruby `num_certs(entity)` + 1822's P16 -1.
    pub(crate) fn g1822_num_certs(&self, pid: u32) -> i32 {
        let eid = EntityId::player(pid);
        let mut n = 0i32;
        for c in &self.corporations {
            if c.closed {
                continue;
            }
            let counts = c
                .share_price
                .as_ref()
                .map_or(true, |sp| {
                    !sp.types
                        .first()
                        .map_or(false, |t| t == "multiple_buy" || t == "unlimited" || t == "no_cert_limit")
                });
            if !counts {
                continue;
            }
            n += c.shares.iter().filter(|s| s.owner == eid).count() as i32;
        }
        let companies = self.companies.iter().filter(|c| !c.closed && c.owner == eid).count() as i32;
        n += companies;
        if self.companies.iter().any(|c| !c.closed && c.owner == eid && c.sym == COMPANY_OSTH) {
            n -= 1;
        }
        n
    }

    pub(crate) fn g1822_num_certs_with_bids(&self, pid: u32) -> i32 {
        self.g1822_num_certs(pid) + self.g1822_winning_bids(pid).len() as i32
    }

    fn g1822_bought(&self) -> bool {
        self.rnd()
            .current_actions
            .iter()
            .any(|a| matches!(a, CurAct::BuyShares | CurAct::Par))
    }

    fn g1822_corp_counts_for_limit(&self, sym: &str) -> bool {
        let c = &self.corporations[self.ci(sym)];
        c.share_price.as_ref().map_or(true, |sp| {
            !sp.types
                .first()
                .map_or(false, |t| t == "multiple_buy" || t == "unlimited" || t == "no_cert_limit")
        })
    }

    /// Corporation#holding_ok?(holder, extra).
    fn g1822_holding_ok(&self, holder: &EntityId, sym: &str, extra: u8) -> bool {
        let c = &self.corporations[self.ci(sym)];
        if holder.0 == format!("ipo:{sym}") {
            return true;
        }
        let pct = c.percent_owned_by(holder) as u32 + extra as u32;
        let typ = c.share_price.as_ref().and_then(|sp| sp.types.first().cloned());
        matches!(typ.as_deref(), Some("multiple_buy") | Some("unlimited"))
            || pct <= c.max_ownership_percent as u32
    }

    /// G1822 `can_gain?(entity, bundle, exchange:)`.
    fn g1822_can_gain(&self, pid: u32, b: &Bundle, exchange: bool) -> bool {
        let c = &self.corporations[self.ci(&b.corp)];
        let owner = &c.shares[b.shares[0]].owner;
        if owner.is_player() {
            return false;
        }
        self.g1822_holding_ok(&EntityId::player(pid), &b.corp, b.percent)
            && (!self.g1822_corp_counts_for_limit(&b.corp)
                || exchange
                || self.g1822_num_certs_with_bids(pid) < self.cert_limit as i32)
    }

    /// G1822 `can_buy?(entity, bundle, available_cash:)`.
    fn g1822_can_buy(&self, pid: u32, b: &Bundle, available: Option<i32>) -> bool {
        let c = &self.corporations[self.ci(&b.corp)];
        if c.corp_type != CorpType::Major {
            return false;
        }
        let has_pres = b.shares.iter().any(|&i| c.shares[i].president);
        let exchange = has_pres && self.phase_has_status("can_convert_concessions");
        let cash = available.unwrap_or_else(|| self.g1822_available_cash(&EntityId::player(pid)));
        let sold = self
            .rnd()
            .players_sold
            .get(&pid)
            .map_or(false, |m| m.contains_key(&b.corp));
        cash >= self.g1822_bundle_price(b)
            && !sold
            && (self.g1822_can_buy_multiple(&b.corp) || !self.g1822_bought())
            && self.g1822_can_gain(pid, b, exchange)
    }

    /// Base `can_buy_multiple?` — 1822 has no multiple_buy cells.
    fn g1822_can_buy_multiple(&self, sym: &str) -> bool {
        let c = &self.corporations[self.ci(sym)];
        let buy_multiple = c
            .share_price
            .as_ref()
            .map_or(false, |sp| sp.types.first().map_or(false, |t| t == "multiple_buy"));
        buy_multiple
            && !self.rnd().current_actions.iter().any(|a| *a == CurAct::Par)
    }

    /// Base `can_buy_shares?(entity, shares)` over a holder's certs.
    fn g1822_can_buy_shares(&self, pid: u32, sym: &str, owner: &EntityId) -> bool {
        let c = &self.corporations[self.ci(sym)];
        let certs = c.shares_owned_by_in_order(owner);
        if certs.is_empty() {
            return false;
        }
        let sold = self
            .rnd()
            .players_sold
            .get(&pid)
            .map_or(false, |m| m.contains_key(sym));
        if sold || (self.g1822_bought() && !self.g1822_can_buy_multiple(sym)) {
            return false;
        }
        // min_share: smallest percent among buyable certs (first wins).
        let mut min: Option<usize> = None;
        for &i in &certs {
            if min.map_or(true, |m| c.shares[i].percent < c.shares[m].percent) {
                min = Some(i);
            }
        }
        let Some(m) = min else { return false };
        let b = Bundle {
            corp: sym.to_string(),
            shares: vec![m],
            percent: c.shares[m].percent,
            price_override: None,
        };
        self.g1822_available_cash(&EntityId::player(pid)) >= self.g1822_bundle_price(&b)
            && self.g1822_can_gain(pid, &b, false)
    }

    fn g1822_can_buy_any(&self, pid: u32) -> bool {
        // can_buy_any_from_market?: pool shares grouped by corporation.
        for c in &self.corporations {
            if c.closed {
                continue;
            }
            if c.shares.iter().any(|s| s.owner.is_market())
                && self.g1822_can_buy_shares(pid, &c.sym, &EntityId::market())
            {
                return true;
            }
        }
        // G1822 can_buy_any_from_ipo?: ipoed majors' treasury certs
        // (`corporation.shares` = the treasury's shares).
        for c in &self.corporations {
            if c.closed || c.corp_type != CorpType::Major || c.ipo_price.is_none() {
                continue;
            }
            if self.g1822_can_buy_shares(pid, &c.sym, &EntityId::ipo(&c.sym)) {
                return true;
            }
        }
        false
    }

    /// G1822 `can_par?(corporation, parrer)`.
    fn g1822_can_par(&self, sym: &str, pid: u32) -> bool {
        let c = &self.corporations[self.ci(sym)];
        if c.closed || c.corp_type == CorpType::Minor {
            return false;
        }
        if !(self.phase_has_status("can_convert_concessions") || self.phase_has_status("can_par")) {
            return false;
        }
        if let Some(conc) = self.g().par_via_exchange.get(sym) {
            if self.co_owner(conc) != EntityId::player(pid) || self.co_closed(conc) {
                return false;
            }
        }
        c.ipo_price.is_none()
    }

    /// `available_par_cash(entity, corporation, share_price:)`.
    pub(crate) fn g1822_available_par_cash(&self, pid: u32, sym: &str, share_price: Option<i32>) -> i32 {
        let mut available = self.g1822_available_cash(&EntityId::player(pid));
        if let Some(conc) = self.g().par_via_exchange.get(sym) {
            if !self.co_closed(conc) {
                available += match share_price {
                    Some(p) if sym == "LNWR" => p,
                    _ => self.companies[self.co_idx(conc).unwrap()].value,
                };
            }
        }
        available
    }

    fn g1822_president_bundle(&self, sym: &str) -> Bundle {
        let c = &self.corporations[self.ci(sym)];
        let pi = c.president_share_index().unwrap_or(0);
        Bundle {
            corp: sym.to_string(),
            shares: vec![pi],
            percent: c.shares[pi].percent,
            price_override: None,
        }
    }

    fn g1822_can_ipo_any(&self, pid: u32) -> bool {
        if self.g1822_bought() {
            return false;
        }
        self.corporations.iter().any(|c| {
            c.corp_type == CorpType::Major
                && !c.closed
                && self.g1822_can_par(&c.sym, pid)
                && self.g1822_can_buy(
                    pid,
                    &self.g1822_president_bundle(&c.sym),
                    Some(self.g1822_available_par_cash(pid, &c.sym, None)),
                )
        })
    }

    /// Base `check_sale_timing` for SELL_AFTER = :full_or_turn.
    pub(crate) fn g1822_check_sale_timing(&self, sym: &str) -> bool {
        let ops = self.g().operating_history.get(sym).copied().unwrap_or(0);
        if self.rnd().kind == RoundKind::Operating
            && self.rnd().current_operator.as_deref() == Some(sym)
        {
            ops > 1
        } else {
            ops > 0
        }
    }

    /// Base BSPS `can_sell?(entity, bundle)` (SELL_BUY_ORDER :sell_buy).
    fn g1822_can_sell(&self, owner: &EntityId, b: &Bundle) -> bool {
        let c = &self.corporations[self.ci(&b.corp)];
        if b.shares.iter().any(|&i| c.shares[i].owner != *owner) {
            return false;
        }
        self.g1822_check_sale_timing(&b.corp)
            && !self.g1822_bought()
            && self.g1822_fit_in_bank(b)
            && self.g1822_can_dump(owner, b)
    }

    fn g1822_can_sell_any(&self, pid: u32) -> bool {
        let eid = EntityId::player(pid);
        self.corporations.iter().filter(|c| !c.closed).any(|c| {
            self.g1822_bundles_for(&eid, &c.sym)
                .iter()
                .any(|b| self.g1822_can_sell(&eid, b))
        })
    }

    fn g1822_must_sell(&self, pid: u32) -> bool {
        self.g1822_can_sell_any(pid) && self.g1822_num_certs(pid) > self.cert_limit as i32
    }

    /// The round's current player (BSPS current_entity).
    fn g1822_current_player(&self) -> Option<u32> {
        match self.g1822_round_entity()? {
            Ent::Player(p) => Some(p),
            _ => None,
        }
    }

    /// G1822 BSPS `actions(entity)`.
    pub(crate) fn g1822_bsps_actions(&self, ent: &Ent) -> Vec<&'static str> {
        if let Ent::Company(sym) = ent {
            if !self.g1822_tax_haven_choices(sym).is_empty() {
                return vec!["choose_ability"];
            }
            return vec![];
        }
        let Ent::Player(pid) = ent else { return vec![] };
        if Some(*pid) != self.g1822_current_player() {
            return vec![];
        }
        if self.g1822_must_sell(*pid) {
            return vec!["sell_shares"];
        }
        let debt = self.player_debt(*pid);
        let bid_actions = self.rnd().bid_actions;
        let mut a = Vec::new();
        if bid_actions == 0 && debt == 0 && self.g1822_can_buy_any(*pid) {
            a.push("buy_shares");
        }
        if bid_actions == 0 && debt == 0 && self.g1822_can_ipo_any(*pid) {
            a.push("par");
        }
        if self.g1822_can_sell_any(*pid) {
            a.push("sell_shares");
        }
        if debt == 0 {
            a.push("bid");
        }
        if debt > 0 && self.player_cash(*pid) > 0 {
            a.push("payoff_player_debt");
            a.push("payoff_player_debt_partial");
        }
        if !a.is_empty() {
            a.push("pass");
        }
        a
    }

    /// `company_choices_osth(company, :stock_round)` + the BSPS
    /// `choices_ability` gate (no P16 buys after bids / with debt).
    pub(crate) fn g1822_tax_haven_choices(&self, company: &str) -> Vec<String> {
        if company != COMPANY_OSTH || self.co_closed(COMPANY_OSTH) {
            return vec![];
        }
        let owner = self.co_owner(COMPANY_OSTH);
        let Some(pid) = owner.player_id() else { return vec![] };
        if self.rnd().kind != RoundKind::Stock || self.rnd().tax_haven_bought {
            return vec![];
        }
        // Without tax_haven_multiple the haven holds one share, ever.
        let multiple = self.g1822_tax_haven_multiple();
        if self.g1822_tax_haven_value() > 0 && !multiple {
            return vec![];
        }
        if self.rnd().bid_actions > 0 || self.player_debt(pid) > 0 {
            return vec![];
        }
        // tax_haven_spender: the owner, or the haven itself once it holds
        // value under tax_haven_multiple.
        let spender_cash = if multiple && self.g1822_tax_haven_value() > 0 {
            self.g().tax_haven_cash
        } else {
            self.player_cash(pid)
        };
        let th = EntityId(TAX_HAVEN_OWNER.to_string());
        let mut out = Vec::new();
        for c in &self.corporations {
            if c.closed || c.corp_type != CorpType::Major {
                continue;
            }
            let price = c.share_price.as_ref().map_or(0, |sp| sp.price);
            if price <= 0 || c.percent_owned_by(&th) > 0 || price > spender_cash {
                continue;
            }
            if c.shares.iter().any(|s| s.owner == EntityId::ipo(&c.sym)) {
                out.push(format!("{}_ipo", c.sym));
            }
            if c.shares.iter().any(|s| s.owner.is_market()) {
                out.push(format!("{}_market", c.sym));
            }
        }
        out
    }

    /// Ruby BSPS `pass!` (G1822: store bids first).
    pub(crate) fn g1822_bsps_pass(&mut self) {
        let Some(i) = self.rnd().step_index(Step::BuySellParShares) else { return };
        self.rndm().passed[i] = true;
        let Some(pid) = self.g1822_current_player() else { return };
        if !self.rnd().current_actions.is_empty() {
            self.rndm().pass_order.retain(|&p| p != pid);
            self.g1822_set_player_passed(pid, false);
        } else {
            if !self.rnd().pass_order.contains(&pid) {
                self.rndm().pass_order.push(pid);
            }
            self.g1822_set_player_passed(pid, true);
        }
    }

    fn g1822_track_action(&mut self, pid: u32, kind: CurAct) {
        self.rndm().last_to_act = Some(pid);
        self.rndm().current_actions.push(kind);
    }

    pub(crate) fn g1822_bsps_process(&mut self, ent: &Ent, action: &Action) -> Result<(), GameError> {
        match action {
            Action::Bid {
                company_sym,
                price,
                ..
            } => {
                let Ent::Player(pid) = ent else { return err("bid by a non-player") };
                self.g1822_process_bid(*pid, company_sym, *price)
            }
            Action::Pass { .. } => {
                self.g1822_bsps_pass();
                Ok(())
            }
            Action::BuyShares {
                corporation_sym,
                percent,
                share_indices,
                ..
            } => {
                let Ent::Player(pid) = ent else { return err("buy by a non-player") };
                self.g1822_process_buy_shares(*pid, corporation_sym, *percent, share_indices)
            }
            Action::SellShares {
                corporation_sym,
                percent,
                share_indices,
                ..
            } => {
                let Ent::Player(pid) = ent else { return err("sell by a non-player") };
                self.g1822_process_sell_shares(*pid, corporation_sym, *percent, share_indices)
            }
            Action::Par {
                corporation_sym,
                share_price,
                ..
            } => {
                let Ent::Player(pid) = ent else { return err("par by a non-player") };
                self.g1822_process_par(*pid, corporation_sym, *share_price)
            }
            Action::ChooseAbility { choice, .. } => {
                let Ent::Company(sym) = ent else { return err("choose_ability by a non-company") };
                self.g1822_process_tax_haven(sym, choice)
            }
            Action::PayoffPlayerDebt { .. } => {
                let Ent::Player(pid) = ent else { return err("payoff by a non-player") };
                self.g1822_payoff_player_loan(*pid, None);
                self.g1822_track_action(*pid, CurAct::Payoff);
                Ok(())
            }
            Action::PayoffPlayerDebtPartial { amount, .. } => {
                let Ent::Player(pid) = ent else { return err("payoff by a non-player") };
                self.g1822_payoff_player_loan(*pid, Some(*amount));
                self.g1822_track_action(*pid, CurAct::Payoff);
                Ok(())
            }
            _ => err(format!("BuySellParShares cannot process {}", action.action_type())),
        }
    }

    /// `payoff_player_loan(player, payoff_amount:)`.
    pub(crate) fn g1822_payoff_player_loan(&mut self, pid: u32, amount: Option<i32>) {
        let cash = self.player_cash(pid);
        let amount = amount.map(|a| a.min(cash));
        let debt = self.player_debt(pid);
        let pay = amount.unwrap_or(cash).min(debt);
        if pay <= 0 {
            return;
        }
        let _ = self.player_spend(pid, pay);
        self.gm().player_debt.insert(pid, debt - pay);
    }

    /// G1822 `process_bid` → `add_bid`.
    fn g1822_process_bid(&mut self, pid: u32, company: &str, price: i32) -> Result<(), GameError> {
        let ci = self
            .co_idx(company)
            .ok_or_else(|| GameError::new(format!("Unknown company {company}")))?;
        self.gm().nothing_sold_in_sr = false;
        self.g1822_set_player_passed(pid, false);
        // Auctioner#add_bid with BidboxAuction's min/max.
        let min = match self.g1822_highest_bid(company) {
            Some(h) => h.price + MIN_BID_INCREMENT,
            None => self.companies[ci].min_bid(),
        };
        if price < min {
            return err(format!("Minimum bid is {} for {}", min, company));
        }
        if (price - min) % MIN_BID_INCREMENT != 0 {
            return err(format!("Must increase bid by a multiple of {MIN_BID_INCREMENT}"));
        }
        let max = self.player_cash(pid) - self.g1822_committed_cash(pid);
        if price > max {
            return err(format!("Cannot afford bid. Maximum possible bid is {max}"));
        }
        let bids = self.rndm().bids.entry(company.to_string()).or_default();
        bids.retain(|b| b.player != pid);
        bids.push(BidRec { player: pid, price });
        self.g1822_track_action(pid, CurAct::Bid);
        self.rndm().bid_actions += 1;
        if self.rnd().bid_actions >= BIDDING_TOKENS_PER_ACTION {
            self.g1822_bsps_pass();
        }
        Ok(())
    }

    /// The bundle a buy/sell action names (Ruby ShareBundle.new(shares,
    /// percent)).
    pub(crate) fn g1822_action_bundle(&self, sym: &str, percent: u8, idx: &[usize]) -> Result<Bundle, GameError> {
        let ci = *self
            .corp_idx
            .get(sym)
            .ok_or_else(|| GameError::new(format!("Unknown corporation {sym}")))?;
        if idx.is_empty() {
            return err("share action names no certificates");
        }
        let c = &self.corporations[ci];
        if idx.iter().any(|&i| i >= c.shares.len()) {
            return err("unknown certificate");
        }
        let owner = &c.shares[idx[0]].owner;
        if idx.iter().any(|&i| c.shares[i].owner != *owner) {
            return err("All shares must be owned by the same owner");
        }
        let sum: u32 = idx.iter().map(|&i| c.shares[i].percent as u32).sum();
        let percent = if percent == 0 { sum as u8 } else { percent };
        Ok(Bundle {
            corp: sym.to_string(),
            shares: idx.to_vec(),
            percent,
            price_override: self.g().pending_share_price,
        })
    }

    fn g1822_process_buy_shares(
        &mut self,
        pid: u32,
        sym: &str,
        percent: u8,
        idx: &[usize],
    ) -> Result<(), GameError> {
        let b = self.g1822_action_bundle(sym, percent, idx)?;
        if !self.g1822_can_buy(pid, &b, None) {
            return err(format!("Cannot buy a share of {sym}"));
        }
        let from_treasury = {
            let c = &self.corporations[self.ci(sym)];
            c.shares[b.shares[0]].owner.is_ipo()
        };
        *self
            .rndm()
            .players_bought
            .entry(pid)
            .or_default()
            .entry(sym.to_string())
            .or_default() += b.percent as u32;
        if from_treasury {
            self.rndm().bought_from_ipo = true;
        }
        self.g1822_pool_buy_shares(&EntityId::player(pid), &b, None, false, true)?;
        self.g1822_track_action(pid, CurAct::BuyShares);
        self.g1822_bsps_pass();
        Ok(())
    }

    fn g1822_process_sell_shares(
        &mut self,
        pid: u32,
        sym: &str,
        percent: u8,
        idx: &[usize],
    ) -> Result<(), GameError> {
        let b = self.g1822_action_bundle(sym, percent, idx)?;
        let eid = EntityId::player(pid);
        if !self.g1822_can_sell(&eid, &b) {
            return err(format!("Cannot sell shares of {sym}"));
        }
        self.rndm()
            .players_sold
            .entry(pid)
            .or_default()
            .insert(sym.to_string(), SoldWhen::Now);
        self.g1822_sell_shares_and_change_price(&b);
        self.g1822_track_action(pid, CurAct::SellShares);
        Ok(())
    }

    fn g1822_process_par(&mut self, pid: u32, sym: &str, price: i32) -> Result<(), GameError> {
        if !self.corp_idx.contains_key(sym) {
            return err(format!("Unknown corporation {sym}"));
        }
        if !self.g1822_can_par(sym, pid) {
            return err(format!("{sym} can't be parred"));
        }
        let sp = self
            .stock_market
            .share_prices_with_types(&["par", "par_1"])
            .into_iter()
            .find(|p| p.price == price)
            .ok_or_else(|| GameError::new(format!("Invalid par price {price}")))?;
        let bundle = self.g1822_president_bundle(sym);
        let conc = self
            .g()
            .par_via_exchange
            .get(sym)
            .cloned()
            .filter(|c| !self.co_closed(c));
        let player = EntityId::player(pid);
        if let Some(conc) = conc {
            if !self.g1822_can_buy(pid, &bundle, Some(self.g1822_available_par_cash(pid, sym, None))) {
                return err(format!("Can't buy a share of {sym}"));
            }
            let mult = (bundle.percent / 10) as i32;
            let conc_value = self.companies[self.co_idx(&conc).unwrap()].value;
            let exchange_price = price * mult - conc_value;
            let exchange_price = (exchange_price >= 0).then_some(exchange_price);
            self.g1822_par_corporation(sym, sp.clone());
            self.g1822_pool_buy_shares(&player, &bundle, exchange_price, true, true)?;
            let concession_money = if exchange_price.is_some() {
                conc_value
            } else {
                price
            };
            self.g1822_bank_spend(concession_money);
            let ci = self.ci(sym);
            self.corporations[ci].cash += concession_money;
            self.close_company(&conc);
        } else {
            // Base process_par: buy the president's cert at par.
            self.g1822_par_corporation(sym, sp);
            *self
                .rndm()
                .players_bought
                .entry(pid)
                .or_default()
                .entry(sym.to_string())
                .or_default() += bundle.percent as u32;
            if !self.g1822_can_buy(pid, &bundle, None) {
                return err(format!("Cannot buy a share of {sym}"));
            }
            self.g1822_pool_buy_shares(&player, &bundle, None, false, true)?;
        }
        self.g1822_track_action(pid, CurAct::Par);
        self.g1822_bsps_pass();
        Ok(())
    }

    /// BSPS `process_choose_ability` for P16 (the tax haven).
    fn g1822_process_tax_haven(&mut self, company: &str, choice: &str) -> Result<(), GameError> {
        if company != COMPANY_OSTH {
            return err(format!("{company} has no stock-round choice"));
        }
        if !self.g1822_tax_haven_choices(company).iter().any(|c| c == choice) {
            return err(format!("Can't buy a share for {choice}"));
        }
        let owner_pid = self.co_owner(COMPANY_OSTH).player_id().expect("player owner");
        let (corp, from) = choice
            .rsplit_once('_')
            .ok_or_else(|| GameError::new("bad tax haven choice"))?;
        let ci = self.ci(corp);
        let src = if from == "ipo" {
            EntityId::ipo(corp)
        } else {
            EntityId::market()
        };
        let share = match from {
            "ipo" => self.corporations[ci].shares_owned_by_in_order(&src).first().copied(),
            _ => self.corporations[ci].oldest_market_share_index(),
        }
        .ok_or_else(|| GameError::new("no share"))?;
        let b = Bundle {
            corp: corp.to_string(),
            shares: vec![share],
            percent: self.corporations[ci].shares[share].percent,
            price_override: None,
        };
        let price = self.g1822_bundle_price(&b);
        let sold = self
            .rnd()
            .players_sold
            .get(&owner_pid)
            .map_or(false, |m| m.contains_key(corp));
        let haven_pays = self.g1822_tax_haven_multiple() && self.g1822_tax_haven_value() > 0;
        let spender = if haven_pays {
            EntityId(TAX_HAVEN_OWNER.to_string())
        } else {
            EntityId::player(owner_pid)
        };
        if self.g1822_available_cash(&spender) < price || sold {
            return err(format!("Can't buy a share of {corp}"));
        }
        self.rndm().tax_haven_bought = true;
        let was_floated = self.corporations[ci].floated;
        // receiver = bank for pool shares, else the corp's treasury.
        self.g1822_entity_spend(&spender, price)?;
        if src.is_market() {
            self.g1822_bank_receive(price);
        } else {
            self.corporations[ci].cash += price;
        }
        let th = EntityId(TAX_HAVEN_OWNER.to_string());
        self.g1822_transfer_shares(&b, &th, false);
        self.g1822_check_float(corp, was_floated);
        self.g1822_track_action(owner_pid, CurAct::Choose);
        self.g1822_bsps_pass();
        Ok(())
    }

    // =====================================================================
    // Choices round (P9 double cash)
    // =====================================================================

    pub(crate) fn g1822_start_choices_round(&mut self) -> Result<(), GameError> {
        let entities: Vec<String> = match self.co_owner(COMPANY_DOUBLE_CASH).player_id() {
            Some(p) if !self.co_closed(COMPANY_DOUBLE_CASH) => vec![p.to_string()],
            _ => vec![],
        };
        let round_num = self.rnd().round_num;
        self.gm().round = RoundSt::new(RoundKind::Choices, round_num, entities, vec![Step::Choose]);
        Ok(())
    }

    /// G1822::Step::Choose#actions.
    pub(crate) fn g1822_choose_actions(&self, ent: &Ent) -> Vec<&'static str> {
        let Ent::Player(pid) = ent else { return vec![] };
        if self.g1822_round_entity() != Some(ent.clone()) {
            return vec![];
        }
        if self.co_closed(COMPANY_DOUBLE_CASH) || self.co_owner(COMPANY_DOUBLE_CASH) != EntityId::player(*pid) {
            return vec![];
        }
        vec!["choose", "pass"]
    }

    pub(crate) fn g1822_choose_process(&mut self, ent: &Ent, action: &Action) -> Result<(), GameError> {
        let Ent::Player(pid) = ent else { return err("choose by a non-player") };
        let i = self.rnd().step_index(Step::Choose).unwrap();
        match action {
            Action::Choose { choice, .. } => {
                // company_choices_double_cash: only "double" while unchosen.
                if choice != "double" || self.g().double_cash_choice.is_some() {
                    return err(format!("{choice} is not a valid choice"));
                }
                self.gm().double_cash_choice = Some(*pid);
            }
            Action::Pass { .. } => {}
            _ => return err("Choose step cannot process this action"),
        }
        self.rndm().passed[i] = true;
        Ok(())
    }
}
