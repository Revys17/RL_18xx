//! 1822 company acquisition (AcquireCompany + G1822#company_bought), the
//! company choice abilities (P5 / P8 / P9 / P17 / P18 `company_choices` +
//! `company_made_choice`) and the MinorAcquisition step.

use crate::actions::{Action, GameError};
use crate::entities::{EntityId, Token};
use crate::game::BaseGame;

use super::track::AbKind;
use super::family::Family;
use super::{err, AcquireState, Ent};

impl BaseGame {
    // =====================================================================
    // AcquireCompany
    // =====================================================================

    /// G1822 `purchasable_companies(entity)`: player-owned, open, not the
    /// entity's, acquirable by its class in this phase.
    pub(crate) fn g1822_purchasable_companies(&self, c: &str) -> Vec<String> {
        let ctype = self.corporations[self.ci(c)].corp_type;
        let phase = self.phase_num();
        self.g()
            .company_order
            .iter()
            .filter(|co| {
                let Some(i) = self.co_idx(co) else { return false };
                let company = &self.companies[i];
                if company.closed || !company.owner.is_player() {
                    return false;
                }
                match (self.fam().private_acquisition)(co) {
                    Some((types, ph)) => phase >= ph && types.contains(&ctype),
                    None => false,
                }
            })
            .cloned()
            .collect()
    }

    pub(crate) fn g1822_process_acquire_company(&mut self, c: &str, company: &str) -> Result<(), GameError> {
        // Ruby Step::AcquireCompany#process_acquire_company does not check
        // the company is purchasable (the UI only offers those): records
        // that re-acquire a company the corporation already owns replay
        // in Ruby (241523 @347 re-acquires P2), so replay accepts any open
        // company. The legal-move generator offers purchasable ones only.
        let Some(i) = self.co_idx(company) else { return err(format!("Unknown company {company}")) };
        if self.co_closed(company) {
            return err(format!("{company} is closed"));
        }
        self.companies[i].owner = EntityId::corporation(c);
        self.rndm().acquired_companies.push(company.to_string());
        self.g1822_company_bought(company, c);
        Ok(())
    }

    /// G1822#company_bought: train privates flip into their train,
    /// remove-revenue privates stop paying, phase-revenue credits move to
    /// the acquirer, P9 pays its corporate revenue.
    fn g1822_company_bought(&mut self, company: &str, c: &str) {
        let f = self.fam();
        if f.private_trains.contains(&company) {
            if let Some(mut t) = self.gm().company_trains.remove(company) {
                t.owner = EntityId::corporation(c);
                t.operated = false;
                let ci = self.ci(c);
                self.corporations[ci].trains.push(t);
            }
            self.close_company(company);
        }
        if f.variant == super::family::Variant::Pnw {
            // PNW company_bought: only the cube company, P14 and P9 stop
            // paying (no remove-revenue / phase-revenue / double-cash).
            if matches!(company, "P10" | "P14" | "P9") {
                if let Some(i) = self.co_idx(company) {
                    self.companies[i].revenue = 0;
                }
            }
            return;
        }
        if f.private_remove_revenue.contains(&company) {
            if let Some(i) = self.co_idx(company) {
                self.companies[i].revenue = 0;
            }
        }
        if f.private_phase_revenue.contains(&company) {
            if let Some(Some(cash)) = self.g().phase_revenue.get(company).cloned() {
                if cash > 0 {
                    let ci = self.ci(c);
                    self.corporations[ci].cash += cash;
                }
            }
            self.gm().phase_revenue.insert(company.to_string(), None);
            self.close_company(company);
        }
        if Family::is(f.company_double_cash, company) {
            if let Some(i) = self.co_idx(company) {
                self.companies[i].revenue = f.company_double_cash_revenue[self.phase_num() as usize];
            }
        }
    }

    // =====================================================================
    // Company choices
    // =====================================================================

    /// The SpecialChoose company: a just-acquired P8 still holding its
    /// three abilities.
    pub(crate) fn g1822_special_choose_company(&self, c: &str) -> Option<String> {
        self.rnd()
            .acquired_companies
            .iter()
            .find(|co| {
                self.co_owner(co) == EntityId::corporation(c) && !self.g1822_company_choices(co, "special_choose").is_empty()
            })
            .cloned()
    }

    /// G1822#company_choices(company, time) — the choice keys.
    pub(crate) fn g1822_company_choices(&self, co: &str, time: &str) -> Vec<String> {
        if self.co_closed(co) {
            return vec![];
        }
        if self.g1822_is_pnw() {
            // PNW company_choices: P21 (Credit Mobilier) only
            return if co == "P21" { self.g1822_pnw_p21_choices(time) } else { vec![] };
        }
        let owner = self.co_owner(co);
        let f = self.fam();
        match co {
            x if Family::is(f.company_chpr, x) => {
                if !matches!(time, "token" | "track") {
                    return vec![];
                }
                let Some(c) = owner.corp_sym() else { return vec![] };
                let mut v = Vec::new();
                let n = self.g1822_exchange_tokens(c);
                if n > 0 {
                    v.push("exchange".to_string());
                }
                let cix = self.ci(c);
                let has_available = self.corporations[cix].tokens.iter().any(|t| !t.used);
                if has_available && n < (f.exchange_tokens)(c) {
                    v.push("available".to_string());
                }
                v
            }
            x if Family::is(f.company_egr, x) => {
                if time != "special_choose" || self.g1822_abilities(co).len() != 3 {
                    return vec![];
                }
                vec!["token".to_string(), "discount".to_string()]
            }
            x if Family::is(f.company_lcdr, x) => {
                if time != "token" {
                    return vec![];
                }
                match owner.corp_sym() {
                    Some(c) if self.g1822_exchange_tokens(c) > 0 => vec!["exchange".to_string()],
                    _ => vec![],
                }
            }
            x if Family::is(f.company_lur, x) => {
                if !matches!(time, "token" | "track" | "issue") || !owner.is_corporation() {
                    return vec![];
                }
                let mut v = Vec::new();
                for (prefix, boxed) in [
                    ('M', self.bidbox_minors()),
                    ('C', self.bidbox_concessions()),
                    ('P', self.bidbox_privates()),
                ] {
                    for c in self.bank_companies(prefix) {
                        if boxed.contains(&c) {
                            continue;
                        }
                        v.push(format!("{c}_top"));
                        v.push(format!("{c}_bottom"));
                    }
                }
                v
            }
            x if Family::is(f.company_double_cash, x) => {
                if self.g().double_cash_choice.is_some() || !owner.is_player() || time != "choose" {
                    return vec![];
                }
                vec!["double".to_string()]
            }
            _ => vec![],
        }
    }

    /// G1822#company_made_choice.
    pub(crate) fn g1822_company_made_choice(&mut self, co: &str, choice: &str, time: &str) -> Result<(), GameError> {
        if !self.g1822_company_choices(co, time).iter().any(|c| c == choice) {
            return err(format!("{choice} is not a valid choice for {co}"));
        }
        let owner = self.co_owner(co);
        let f = self.fam();
        if self.g1822_is_pnw() && co == "P21" && choice == "exchange" {
            // company_made_choice_p21: an exchange token becomes available
            let c = owner.corp_sym().unwrap().to_string();
            self.g1822_move_exchange_token(&c);
            self.close_company(co);
            return Ok(());
        }
        match co {
            x if Family::is(f.company_egr, x) => {
                let remove = if choice == "token" { AbKind::TileDiscount } else { AbKind::TileLay };
                if let Some(v) = self.gm().abilities.get_mut(co) {
                    v.retain(|a| a.kind != remove);
                }
            }
            x if Family::is(f.company_lcdr, x) => {
                let c = owner.corp_sym().unwrap().to_string();
                self.g1822_move_exchange_token(&c);
                self.close_company(co);
            }
            x if Family::is(f.company_chpr, x) => {
                let c = owner.corp_sym().unwrap().to_string();
                if choice == "exchange" {
                    self.g1822_move_exchange_token(&c);
                } else {
                    // find_token_by_type.destroy! + add_exchange_token. The
                    // offer tests tokens_by_type (unused tokens of ANY
                    // type), so with only the destination token unused
                    // find_token_by_type(:normal) is nil and Ruby crashes.
                    let cix = self.ci(&c);
                    let Some(pos) =
                        self.corporations[cix].tokens.iter().position(|t| !t.used && t.token_type == "normal")
                    else {
                        return err(format!("{c} has no available normal token for {co} (Ruby raises)"));
                    };
                    self.corporations[cix].tokens.remove(pos);
                    *self.gm().exchange_tokens.entry(c.clone()).or_insert(0) += 1;
                }
                self.close_company(co);
            }
            x if Family::is(f.company_lur, x) => {
                let (target, where_) = choice.split_once('_').ok_or_else(|| GameError::new("bad choice"))?;
                let target = target.to_string();
                let top = where_ == "top";
                self.gm().company_order.retain(|c| c != &target);
                if top {
                    let last_boxed = match target.chars().next() {
                        Some('M') => self.bidbox_minors().last().cloned(),
                        Some('C') => self.bidbox_concessions().last().cloned(),
                        _ => self.bidbox_privates().last().cloned(),
                    };
                    let idx = last_boxed
                        .and_then(|l| self.g().company_order.iter().position(|c| *c == l))
                        .map_or(0, |i| i + 1);
                    self.gm().company_order.insert(idx, target);
                } else {
                    self.gm().company_order.push(target);
                }
                self.close_company(co);
            }
            x if Family::is(f.company_double_cash, x) => {
                self.gm().double_cash_choice = owner.player_id();
            }
            _ => return err(format!("{co} has no choices")),
        }
        Ok(())
    }

    // =====================================================================
    // MinorAcquisition
    // =====================================================================

    fn g1822_minor_price(&self, m: &str) -> i32 {
        self.corporations[self.ci(m)].share_price.as_ref().map_or(0, |sp| sp.price)
    }

    fn g1822_minor_from_bank(&self, m: &str) -> bool {
        // A bidbox minor: not floated, owned by nobody (`!minor.owner`;
        // 1822PNW's regionals are owned without floating).
        if self.g1822_is_pnw() {
            return self.corporations[self.ci(m)].owner_id.is_none();
        }
        !self.corporations[self.ci(m)].floated
    }

    /// The pay choice keys (`pay_choices(entity, minor)`).
    pub(crate) fn g1822_pay_choices(&self, c: &str, m: &str) -> Vec<String> {
        let cash = self.corporations[self.ci(c)].cash;
        if self.g1822_minor_from_bank(m) {
            return if cash >= self.fam().minor_bidbox_price { vec!["money".to_string()] } else { vec![] };
        }
        let mut v = Vec::new();
        let mprice = self.g1822_minor_price(m) * 2;
        if cash >= mprice {
            v.push("money".to_string());
        }
        let ipo = self.corporations[self.ci(c)].percent_owned_by(&EntityId::ipo(c)) / 10;
        let owner = self.g1822_president(m);
        let owner_cash = owner.as_ref().and_then(|o| o.player_id()).map_or(0, |p| self.player_cash(p));
        let cprice = self.corporations[self.ci(c)].share_price.as_ref().map_or(0, |sp| sp.price);
        let ok = |n: i32| -> bool {
            let diff = mprice - cprice * n;
            (diff < 0 && -diff <= owner_cash) || diff == 0 || (diff > 0 && diff <= cash)
        };
        if ipo > 0 && ok(1) {
            v.push("one_share".to_string());
        }
        let counts_for_limit = self.corporations[self.ci(c)].share_price.as_ref().map_or(true, |sp| {
            !sp.types.first().map_or(false, |t| t == "no_cert_limit" || t == "unlimited" || t == "multiple_buy")
        });
        let owner_certs_ok = owner
            .as_ref()
            .and_then(|o| o.player_id())
            .map_or(false, |p| self.g1822_num_certs(p) < self.cert_limit as i32);
        if ipo > 1 && (!counts_for_limit || owner_certs_ok) && ok(2) {
            v.push("two_shares".to_string());
        }
        v
    }

    pub(crate) fn g1822_potentially_mergeable(&self, c: &str) -> Vec<String> {
        let mut out: Vec<String> = self
            .corporations
            .iter()
            .filter(|m| {
                !m.closed
                    && m.corp_type == crate::title::CorpType::Minor
                    && m.floated
                    && self.g().operating_history.get(&m.sym).copied().unwrap_or(0) > 1
            })
            .map(|m| m.sym.clone())
            .filter(|m| !self.g1822_pay_choices(c, m).is_empty())
            .collect();
        if self.phase_has_status("can_acquire_minor_bidbox") {
            for co in self.bidbox_minors() {
                let m = Self::minor_of_company(&co);
                if !self.g1822_pay_choices(c, &m).is_empty() {
                    out.push(m);
                }
            }
        }
        if self.g1822_is_pnw() {
            // PNW: no associated minors; regionals after two payouts
            out.retain(|m| !self.g1822_pnw_is_associated(m));
            for r in super::pnw::REGIONALS {
                if self.corp_idx.contains_key(*r)
                    && !self.corporations[self.ci(r)].closed
                    && self.g().regional_payout_count.get(*r).copied().unwrap_or(0) > 1
                {
                    out.push(r.to_string());
                }
            }
        }
        out
    }

    pub(crate) fn g1822_minor_acquisition_actions(&self, ent: &Ent) -> Vec<&'static str> {
        let Ent::Corp(c) = ent else { return vec![] };
        if Some(c) != self.g1822_operator().as_ref() {
            return vec![];
        }
        match self.rnd().acquire_state {
            AcquireState::ChoosePay | AcquireState::ChooseToken => return vec!["choose"],
            AcquireState::SelectMinor => {}
        }
        if !self.is_major(c) || self.g().operating_history.get(c).copied().unwrap_or(0) <= 1 {
            return vec![];
        }
        if self.g1822_potentially_mergeable(c).is_empty() {
            vec![]
        } else {
            vec!["merge", "pass"]
        }
    }

    pub(crate) fn g1822_minor_acquisition_process(&mut self, i: usize, ent: &Ent, action: &Action) -> Result<(), GameError> {
        let Ent::Corp(c) = ent else { return err("acquisition by a non-corporation") };
        let c = c.clone();
        match action {
            Action::Merge { corporation_sym, .. } => {
                if !self.corp_idx.contains_key(corporation_sym.as_str()) {
                    return err(format!("Unknown minor {corporation_sym}"));
                }
                // process_merge's entity_connects? computes the major's graph.
                self.g1822_note_graph(&c);
                let pay = self.g1822_pay_choices(&c, corporation_sym);
                let tokens = if self.g1822_exchange_tokens(&c) > 0 {
                    vec!["replace".to_string(), "exchange".to_string()]
                } else {
                    vec!["remove".to_string()]
                };
                let r = self.rndm();
                r.selected_minor = Some(corporation_sym.clone());
                r.pay_choices = pay;
                r.token_choices = tokens;
                r.acquire_state = AcquireState::ChoosePay;
                Ok(())
            }
            Action::Choose { choice, .. } => match self.rnd().acquire_state {
                AcquireState::ChoosePay => {
                    let n = match choice.as_str() {
                        "money" => 0,
                        "one_share" => 1,
                        "two_shares" => 2,
                        _ => return err(format!("{choice} is not a pay choice")),
                    };
                    let r = self.rndm();
                    r.selected_share_num = n;
                    r.acquire_state = AcquireState::ChooseToken;
                    Ok(())
                }
                AcquireState::ChooseToken => {
                    let m = self.rnd().selected_minor.clone().ok_or_else(|| GameError::new("no minor selected"))?;
                    if self.g1822_minor_from_bank(&m) {
                        self.g1822_acquire_bank_minor(&c, &m, choice)?;
                    } else {
                        self.g1822_acquire_entity_minor(&c, &m, choice)?;
                    }
                    self.rndm().acquire_state = AcquireState::SelectMinor;
                    self.rndm().passed[i] = true;
                    Ok(())
                }
                AcquireState::SelectMinor => err("nothing to choose"),
            },
            Action::Pass { .. } => {
                self.rndm().passed[i] = true;
                Ok(())
            }
            _ => err("MinorAcquisition cannot process this action"),
        }
    }

    fn g1822_acquire_bank_minor(&mut self, c: &str, m: &str, token_choice: &str) -> Result<(), GameError> {
        self.corp_spend(c, self.fam().minor_bidbox_price)?;
        match token_choice {
            "replace" => {
                if m == self.fam().minor_14_id {
                    self.g1822_remove_exchange_token(c);
                    let cix = self.ci(c);
                    self.corporations[cix].tokens.push(Token::new(c.to_string(), 0));
                    let tidx = self.corporations[cix].tokens.len() - 1;
                    // minor_14_token_ability: a £20 token anywhere.
                    self.gm().corp_token_abilities.entry(c.to_string()).or_default().push((vec![], 20));
                    let home = self.fam().minor_14_home_hex.to_string();
                    self.rndm().pending_tokens.push(super::PendingTok {
                        corp: c.to_string(),
                        hexes: vec![home],
                        token_idx: tidx,
                    });
                } else {
                    let home = self.title_def().corporations().into_iter().find(|d| d.sym == m).unwrap().home_hex;
                    let hi = self.hex_idx[home];
                    let n = self.hexes[hi].tile.cities.len();
                    let ci = (0..n)
                        .find(|&k| self.g1822_city_res(home, k).iter().any(|r| r.as_deref() == Some(m)))
                        .unwrap_or(0);
                    // minor_city.reservations.delete(minor)
                    if let Some(v) = self.gm().city_reservations.get_mut(&format!("{home}#{ci}")) {
                        v.retain(|r| r.as_deref() != Some(m));
                    }
                    if self.g1822_city_tokened_by(hi, ci, c) {
                        self.g1822_move_exchange_token(c);
                    } else {
                        self.g1822_remove_exchange_token(c);
                        let cix = self.ci(c);
                        self.corporations[cix].tokens.push(Token::new(c.to_string(), 0));
                        let tidx = self.corporations[cix].tokens.len() - 1;
                        self.g1822_put_token(c, tidx, hi, ci, false)?;
                        self.g1822_ruby_graph_clear();
                    }
                }
            }
            "exchange" => self.g1822_move_exchange_token(c),
            _ => {}
        }
        let company = format!("M{m}");
        self.delete_company(&company);
        self.g1822_close_corporation(m);
        Ok(())
    }

    fn g1822_acquire_entity_minor(&mut self, c: &str, m: &str, token_choice: &str) -> Result<(), GameError> {
        let n = self.rnd().selected_share_num as i32;
        let mprice = self.g1822_minor_price(m) * 2;
        let cprice = self.corporations[self.ci(c)].share_price.as_ref().map_or(0, |sp| sp.price);
        let diff = mprice - cprice * n;
        let owner = self.g1822_president(m).ok_or_else(|| GameError::new("minor has no owner"))?;
        let owner_pid = owner.player_id().ok_or_else(|| GameError::new("minor owner not a player"))?;
        if diff < 0 {
            let pi = self.player_index(owner_pid).unwrap();
            if self.players[pi].cash < -diff {
                return err("minor owner cannot pay the difference");
            }
            self.players[pi].cash += diff;
            let cix = self.ci(c);
            self.corporations[cix].cash -= diff;
        } else if diff > 0 {
            let cix = self.ci(c);
            if self.corporations[cix].cash < diff {
                return err(format!("{c} cannot pay {diff}"));
            }
            self.corporations[cix].cash -= diff;
            self.pay_player(owner_pid, diff);
        }
        // Treasury shares to the minor's owner (Share#transfer — no
        // presidency check).
        let cix = self.ci(c);
        for _ in 0..n {
            let ipo = EntityId::ipo(c);
            let Some(&s) = self.corporations[cix].shares_owned_by_in_order(&ipo).first() else {
                return err(format!("{c} has no treasury shares"));
            };
            self.corporations[cix].set_share_owner(s, owner.clone());
        }
        // Assets: cash, companies, trains.
        let mci = self.ci(m);
        let cash = self.corporations[mci].cash;
        self.corporations[mci].cash = 0;
        self.corporations[cix].cash += cash;
        for co in self.companies_of(&EntityId::corporation(m)) {
            let i = self.co_idx(&co).unwrap();
            self.companies[i].owner = EntityId::corporation(c);
        }
        let trains: Vec<crate::entities::Train> = std::mem::take(&mut self.corporations[mci].trains);
        for mut t in trains {
            t.owner = EntityId::corporation(c);
            self.corporations[cix].trains.push(t);
        }
        if self.g1822_is_pnw() {
            // PNW extra_transfers: the coal token
            self.g1822_pnw_move_coal_token(m, c);
        }
        match token_choice {
            "replace" => {
                let (hi, ci) = self
                    .g1822_token_city(m, None)
                    .ok_or_else(|| GameError::new("minor has no token"))?;
                if self.g1822_city_tokened_by(hi, ci, c) {
                    self.g1822_move_exchange_token(c);
                    self.g1822_remove_minor_token(m);
                } else {
                    self.g1822_remove_exchange_token(c);
                    // move_tokens_to_surviving: the minor's token becomes a
                    // new (price 0) token of the major in the same city;
                    // used tokens first, then unused by price.
                    self.g1822_remove_token_at(hi, ci, m, None);
                    let cix = self.ci(c);
                    let mut used: Vec<Token> = Vec::new();
                    let mut unused: Vec<Token> = Vec::new();
                    for t in self.corporations[cix].tokens.drain(..) {
                        if t.used {
                            used.push(t);
                        } else {
                            unused.push(t);
                        }
                    }
                    used.push(Token::new(c.to_string(), 0));
                    let new_idx = used.len() - 1;
                    unused.sort_by_key(|t| t.price);
                    let mut all = used;
                    all.extend(unused);
                    all.truncate(9);
                    self.corporations[cix].tokens = all;
                    self.g1822_put_token(c, new_idx, hi, ci, false)?;
                    // TokenMerger: clear_graph_for(surviving)
                    self.g1822_ruby_graph_clear();
                }
            }
            "exchange" => {
                self.g1822_move_exchange_token(c);
                self.g1822_remove_minor_token(m);
            }
            _ => {}
        }
        let company = format!("M{m}");
        self.delete_company(&company);
        self.g1822_close_corporation(m);
        Ok(())
    }

    /// remove_minor_token: delete the minor's token, dropping an extra
    /// (cheater) slot when the city has more than its minimum.
    fn g1822_remove_minor_token(&mut self, m: &str) {
        let Some((hi, ci)) = self.g1822_token_city(m, None) else { return };
        let min_slots = self.g1822_min_city_slots(hi, ci);
        let city = &self.hexes[hi].tile.cities[ci];
        let remove_slot = city.tokens.len() > min_slots;
        let pos = city.tokens.iter().position(|t| t.as_ref().is_some_and(|tok| tok.corporation_id == m));
        let hex_id = self.hexes[hi].id.clone();
        if let Some(pos) = pos {
            if remove_slot {
                self.hexes[hi].tile.cities[ci].tokens.remove(pos);
                if let Some(v) = self.gm().city_reservations.get_mut(&format!("{hex_id}#{ci}")) {
                    if pos < v.len() {
                        v.remove(pos);
                        v.push(None);
                    }
                }
            } else {
                self.hexes[hi].tile.cities[ci].tokens[pos] = None;
            }
        }
        self.clear_graph_cache();
    }

    /// G1822#min_city_slots: minor 14's London city keeps one extra slot.
    fn g1822_min_city_slots(&self, hi: usize, ci: usize) -> usize {
        let city = &self.hexes[hi].tile.cities[ci];
        let normal = city.slots as usize;
        if self.hexes[hi].id != self.fam().minor_14_home_hex {
            return normal;
        }
        let exit = self.hexes[hi].tile.paths.iter().find_map(|p| match (&p.a, &p.b) {
            (crate::tiles::PathEndpoint::City(c), crate::tiles::PathEndpoint::Edge(e))
            | (crate::tiles::PathEndpoint::Edge(e), crate::tiles::PathEndpoint::City(c))
                if *c == ci =>
            {
                Some(*e)
            }
            _ => None,
        });
        if exit.is_some() && exit == self.g().minor_14_city_exit {
            normal + 1
        } else {
            normal
        }
    }
}
