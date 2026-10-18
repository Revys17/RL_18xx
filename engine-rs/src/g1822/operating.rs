//! 1822 operating round: Engine::Round::Operating with the G1822 step list
//! (game.rb `operating_round`):
//!
//!   DiscardTrain, PendingToken, FirstTurnHousekeeping, AcquireCompany,
//!   SpecialChoose, SpecialTrack, SpecialToken, Track, DestinationToken,
//!   Token, Route, Dividend, BuyTrain, MinorAcquisition, PendingToken,
//!   IssueShares
//!
//! Each step's `actions` / `process_*` / `skip!` is ported in this file
//! or its siblings (track.rs, trains.rs, tokens.rs, acquisition.rs).

use crate::actions::{Action, GameError};
use crate::entities::EntityId;
use crate::game::BaseGame;
use crate::title::CorpType;

use super::shares::{Bundle, Dir};
use super::family::Family;
use super::{err, Ent, RoundKind, RoundSt, RunRoute, Step};

impl BaseGame {
    // =====================================================================
    // Lifecycle
    // =====================================================================

    /// `@operating_rounds` for the current set (phase.operating_rounds at
    /// set start; Choices → OR sets it).
    pub(crate) fn g1822_operating_rounds_in_set(&self) -> u32 {
        self.g().operating_rounds as u32
    }

    /// G1822 `operating_order`: floated corporations sorted (price desc,
    /// column desc, row asc, cell position, name), minors first.
    pub(crate) fn g1822_operating_order(&self) -> Vec<String> {
        let mut floated: Vec<String> = self
            .corporations
            .iter()
            .filter(|c| !c.closed && c.floated)
            .map(|c| c.sym.clone())
            .collect();
        self.g1822_sort_operating(&mut floated);
        let (minors, majors): (Vec<String>, Vec<String>) =
            floated.into_iter().partition(|s| self.is_minor(s));
        let order: Vec<String> = minors.into_iter().chain(majors).collect();
        if self.g1822_is_mx() {
            // MX operating_order: the NdeM's place depends on its state
            return self.g1822_mx_operating_order(order);
        }
        order
    }

    /// Corporation#<=> (sort_order_key).
    pub(crate) fn g1822_sort_operating(&self, syms: &mut [String]) {
        syms.sort_by_key(|s| {
            let c = &self.corporations[self.ci(s)];
            match &c.share_price {
                Some(sp) => (
                    0i32,
                    -sp.price,
                    -(sp.column as i32),
                    sp.row as i32,
                    self.market_cell_position(s, sp.row, sp.column) as i32,
                    s.clone(),
                ),
                None => (1, 0, 0, 0, 0, s.clone()),
            }
        });
    }

    pub(crate) fn g1822_or_steps(&self) -> Vec<Step> {
        let mut v = vec![
            Step::DiscardTrain,
            Step::PendingToken,
            Step::FirstTurnHousekeeping,
            Step::AcquireCompany,
            Step::SpecialChoose,
            Step::SpecialTrack,
            Step::SpecialToken,
            Step::Track,
            Step::DestinationToken,
            Step::Token,
            Step::Route,
            Step::Dividend,
            Step::BuyTrain,
            Step::MinorAcquisition,
            Step::PendingToken,
            Step::IssueShares,
        ];
        if self.g1822_is_pnw() {
            // G1822PNW: Assign after AcquireCompany
            v.insert(4, Step::Assign);
        }
        if self.g1822_is_mx() {
            // G1822MX: the NdeM's privatization steps close the list
            v.push(Step::CashOutNdem);
            v.push(Step::AuctionNdemTokens);
        }
        if self.g1822_is_ca() {
            // G1822CA: AssignSawmill after AcquireCompany, a PendingToken
            // after Track (QMOO's home), AcquisitionTrack after
            // MinorAcquisition
            v = vec![
                Step::DiscardTrain,
                Step::PendingToken,
                Step::FirstTurnHousekeeping,
                Step::AcquireCompany,
                Step::AssignSawmill,
                Step::SpecialChoose,
                Step::SpecialTrack,
                Step::SpecialToken,
                Step::Track,
                Step::PendingToken,
                Step::DestinationToken,
                Step::Token,
                Step::Route,
                Step::Dividend,
                Step::BuyTrain,
                Step::MinorAcquisition,
                Step::AcquisitionTrack,
                Step::PendingToken,
                Step::IssueShares,
            ];
        }
        v
    }

    /// `new_operating_round(round_num)` + Operating#setup.
    pub(crate) fn g1822_start_operating_round(&mut self, round_num: u8) -> Result<(), GameError> {
        if round_num == 1 {
            // Choices → OR: `@operating_rounds = @phase.operating_rounds`.
            self.gm().operating_rounds = self.phase.operating_rounds;
        }
        let entities = self.g1822_operating_order();
        let round = RoundSt::new(RoundKind::Operating, round_num, entities, self.g1822_or_steps());
        self.gm().round = round;
        self.g1822_or_steps_setup();
        // Operating#setup: payout_companies (G1822 set_private_revenues
        // first), trains un-operated, then start_operating if any to act.
        self.g1822_payout_companies();
        let syms = self.rnd().entities.clone();
        for s in &syms {
            let ci = self.ci(s);
            for t in &mut self.corporations[ci].trains {
                t.operated = false;
            }
        }
        // reset_ability_count_this_or! (1822CA's count_per_or privates)
        for v in self.gm().abilities.values_mut() {
            for a in v.iter_mut() {
                a.count_this_or = 0;
            }
        }
        if self.g1822_any_to_act() {
            self.g1822_start_operating()?;
        }
        Ok(())
    }

    fn g1822_any_to_act(&self) -> bool {
        self.rnd()
            .entities
            .iter()
            .any(|s| !self.corporations[self.ci(s)].closed)
    }

    pub(crate) fn g1822_or_finished(&self) -> bool {
        self.finished || self.g1822_active_step().is_none() || !self.g1822_any_to_act()
    }

    /// Steps' `setup` for a new operator (Tracker / Tokener / BuyTrain /
    /// BSPS-free OR steps).
    fn g1822_or_steps_setup(&mut self) {
        let r = self.rndm();
        r.num_laid_track = 0;
        r.upgraded_track = false;
        r.laid_hexes.clear();
        r.tokened = false;
        r.last_share_sold_price = None;
        r.corporations_sold.clear();
        r.num_laid_portage = 0;
        if self.g1822_is_ca() {
            self.g1822_ca_route_setup();
        }
        // MX Track#setup / AuctionNdemTokens#setup for the NdeM's turn
        if self.g1822_is_mx() && self.g1822_operator().as_deref() == Some(super::mx::NDEM) {
            self.g1822_mx_track_setup();
            self.g1822_mx_auction_setup();
        }
    }

    /// Operating#start_operating.
    pub(crate) fn g1822_start_operating(&mut self) -> Result<(), GameError> {
        let sym = self.rnd().entities[self.rnd().entity_index].clone();
        if self.corporations[self.ci(&sym)].closed {
            return self.g1822_or_next_entity();
        }
        self.rndm().current_operator = Some(sym.clone());
        self.g1822_place_home_token(&sym);
        self.g1822_skip_steps();
        if self.g1822_or_finished() {
            return self.g1822_or_next_entity();
        }
        Ok(())
    }

    /// Operating#next_entity!.
    pub(crate) fn g1822_or_next_entity(&mut self) -> Result<(), GameError> {
        let r = self.rnd();
        if r.entity_index + 1 >= r.entities.len() {
            return Ok(());
        }
        self.rndm().entity_index += 1;
        let sym = self.rnd().entities[self.rnd().entity_index].clone();
        if self.corporations[self.ci(&sym)].closed {
            return self.g1822_or_next_entity();
        }
        let n = self.rnd().steps.len();
        self.rndm().passed = vec![false; n];
        self.rndm().acted = vec![false; n];
        self.g1822_or_steps_setup();
        self.g1822_start_operating()
    }

    /// Operating#after_process (+ the process_single_action round-end loop).
    pub(crate) fn g1822_or_after_process(&mut self) -> Result<(), GameError> {
        if self.finished {
            return Ok(());
        }
        if self.g1822_active_step().is_none() {
            self.g1822_or_next_entity()?;
        }
        Ok(())
    }

    /// Ruby `or_round_finished` — base no-op for 1822.
    pub(crate) fn g1822_or_round_finished(&mut self) {}

    /// Unused hook kept for the next_round! flow (end_now? is evaluated by
    /// the transition loop before next_round!).
    pub(crate) fn g1822_end_now_after_or(&self) -> bool {
        self.finished
    }

    /// payout_companies with set_private_revenues (P15/P20 phase revenue,
    /// P9's corporate revenue).
    fn g1822_payout_companies(&mut self) {
        let phase = self.phase_num() as i32;
        let f = self.fam();
        let syms: Vec<String> = self.g().company_order.clone();
        // PNW set_private_revenues is a no-op
        let set_private_revenues = !self.g1822_is_pnw();
        for c in syms.iter().filter(|_| set_private_revenues) {
            let Some(i) = self.co_idx(c) else { continue };
            if self.companies[i].closed || self.companies[i].owner.is_none() {
                continue;
            }
            if f.private_phase_revenue.contains(&c.as_str()) {
                let mult = if Family::is(f.company_10x_revenue, c) { 10 } else { 5 };
                let rev = phase * mult;
                self.companies[i].revenue = rev;
                self.g1822_bank_spend(rev);
                if let Some(Some(cash)) = self.gm().phase_revenue.get_mut(c.as_str()) {
                    *cash += rev;
                }
            }
            if Family::is(f.company_double_cash, c) && self.companies[i].owner.is_corporation() {
                self.companies[i].revenue = f.company_double_cash_revenue[phase as usize];
            }
        }
        // Base payout_companies: owned, positive revenue; players (seat
        // order) first, then corporations; then revenue, name.
        let mut payees: Vec<(u32, usize, i32, String, String)> = Vec::new();
        for c in &syms {
            let Some(i) = self.co_idx(c) else { continue };
            let co = &self.companies[i];
            if co.closed || co.owner.is_none() || co.revenue <= 0 {
                continue;
            }
            let key = if let Some(pid) = co.owner.player_id() {
                (0u32, self.player_index(pid).unwrap_or(0))
            } else {
                (1u32, 0)
            };
            payees.push((key.0, key.1, co.revenue, co.name.clone(), c.clone()));
        }
        payees.sort_by(|a, b| (a.0, a.1, a.2, &a.3).cmp(&(b.0, b.1, b.2, &b.3)));
        for (_, _, rev, _, c) in payees {
            let owner = self.co_owner(&c);
            self.g1822_bank_pay(&owner, rev);
        }
        if self.g1822_is_mx() {
            // MX payout_companies: P16's £10 (or its closing)
            self.g1822_mx_p16_payment();
        }
        if self.g1822_is_pnw() {
            // PNW payout_companies: count the regionals' payouts
            for r in super::pnw::REGIONALS {
                let owned = self.corp_idx.get(*r).map_or(false, |&ri| {
                    !self.corporations[ri].closed && !self.corporations[ri].owner_id.is_none()
                });
                if owned {
                    *self.gm().regional_payout_count.entry(r.to_string()).or_insert(0) += 1;
                }
            }
        }
    }

    // =====================================================================
    // Step dispatch
    // =====================================================================

    /// The operating corp (Ruby `current_entity` of the round).
    pub(crate) fn g1822_operator(&self) -> Option<String> {
        if self.rnd().kind != RoundKind::Operating {
            return None;
        }
        self.rnd().entities.get(self.rnd().entity_index).cloned()
    }

    fn is_operator(&self, ent: &Ent) -> bool {
        matches!(ent, Ent::Corp(c) if Some(c) == self.g1822_operator().as_ref())
    }

    pub(crate) fn g1822_or_step_actions(&self, i: usize, ent: &Ent) -> Vec<&'static str> {
        match self.rnd().steps[i] {
            Step::DiscardTrain => match ent {
                Ent::Corp(c) if self.g1822_crowded_corps().contains(c) => vec!["discard_train"],
                _ => vec![],
            },
            Step::PendingToken => match (ent, self.rnd().pending_tokens.first()) {
                (Ent::Corp(c), Some(p)) if &p.corp == c => vec!["place_token"],
                _ => vec![],
            },
            Step::FirstTurnHousekeeping => {
                let Ent::Corp(c) = ent else { return vec![] };
                if !self.is_operator(ent) {
                    // Ruby FTH#actions has no current-entity check, but its
                    // conditions only ever hold for the operator in practice.
                }
                if self.is_minor(c)
                    && !self.g1822_operated(c)
                    && self.depot.trains.first().map_or(false, |t| t.name == "L")
                    && self.g1822_can_buy_train_base(c)
                {
                    vec!["buy_train", "pass"]
                } else {
                    vec![]
                }
            }
            Step::AcquireCompany => {
                if self.g1822_is_pnw() && !self.rnd().p20_choices.is_empty() {
                    return vec!["choose"];
                }
                if !self.is_operator(ent) {
                    return vec![];
                }
                let Ent::Corp(c) = ent else { return vec![] };
                if self.g1822_purchasable_companies(c).is_empty() {
                    vec![]
                } else {
                    vec!["acquire_company", "pass"]
                }
            }
            // Engine::Step::SpecialChoose (PNW): no :choose_ability companies
            Step::SpecialChoose if self.g1822_is_pnw() => vec![],
            // G1822MX::Step::SpecialChoose: P16's close_p16 while its owner
            // operates (`owning_corp_or_turn`)
            Step::SpecialChoose if self.g1822_is_mx() => match ent {
                Ent::Company(co) if co == "P16" && self.g1822_mx_p16_usable() => vec!["choose_ability"],
                _ => vec![],
            },
            Step::SpecialChoose => {
                if !self.is_operator(ent) {
                    return vec![];
                }
                let Ent::Corp(c) = ent else { return vec![] };
                if self.g1822_special_choose_company(c).is_some() {
                    vec!["choose"]
                } else {
                    vec![]
                }
            }
            Step::SpecialTrack => self.g1822_special_track_actions(ent),
            Step::SpecialToken => self.g1822_special_token_actions(ent),
            Step::Track => self.g1822_track_actions(ent),
            Step::AssignSawmill => self.g1822_ca_sawmill_actions(ent),
            Step::AcquisitionTrack => self.g1822_ca_acquisition_track_actions(ent),
            Step::DestinationToken => {
                if !self.is_operator(ent) {
                    return vec![];
                }
                let Ent::Corp(c) = ent else { return vec![] };
                if self.g1822_is_ca() {
                    // CA DestinationToken#actions
                    if self.g1822_ca_destinated(c) {
                        return vec![];
                    }
                    if self.is_major(c) && self.g1822_unplaced_destination_token(c).is_some() {
                        return vec!["hex_token", "place_token", "pass"];
                    }
                    return vec![];
                }
                if self.is_major(c) && self.g1822_unplaced_destination_token(c).is_some() {
                    vec!["hex_token", "pass"]
                } else {
                    vec![]
                }
            }
            Step::Token => self.g1822_token_actions(ent),
            Step::Route => self.g1822_route_actions(ent),
            Step::Dividend => {
                if !self.is_operator(ent) {
                    return vec![];
                }
                let Ent::Corp(c) = ent else { return vec![] };
                // MX Dividend#actions: nothing for the NdeM (it pays out by itself)
                if !self.is_major(c) || self.g1822_routes_revenue() == 0 || self.g1822_mx_is_ndem(c) {
                    return vec![];
                }
                let mut a = vec!["dividend"];
                if self.rnd().extra_train_choice.is_empty()
                    && self.g1822_find_extra_train(c).is_some()
                    && self.corporations[self.ci(c)].trains.len() > 1
                {
                    a.push("choose");
                }
                a
            }
            Step::BuyTrain => self.g1822_buy_train_actions(ent),
            Step::MinorAcquisition => self.g1822_minor_acquisition_actions(ent),
            Step::IssueShares => self.g1822_issue_shares_actions(ent),
            Step::CashOutNdem => {
                if self.g1822_mx_cash_out_active() {
                    vec!["pass"]
                } else {
                    vec![]
                }
            }
            Step::AuctionNdemTokens => self.g1822_mx_auction_actions(ent),
            _ => vec![],
        }
    }

    pub(crate) fn g1822_or_step_process(&mut self, i: usize, ent: &Ent, action: &Action) -> Result<(), GameError> {
        let step = self.rnd().steps[i];
        match step {
            Step::DiscardTrain => match action {
                Action::DiscardTrain { train_name, .. } => {
                    let Ent::Corp(c) = ent else { return err("discard by a non-corporation") };
                    self.g1822_process_discard_train(c, train_name)
                }
                _ => err("DiscardTrain cannot process this action"),
            },
            Step::PendingToken => match action {
                Action::PlaceToken { hex_id, city_index, .. } => {
                    self.g1822_process_pending_token(hex_id, *city_index)
                }
                _ => err("PendingToken cannot process this action"),
            },
            Step::FirstTurnHousekeeping => match action {
                Action::BuyTrain { price, variant, .. } => {
                    let Ent::Corp(c) = ent else { return err("buy_train by a non-corporation") };
                    let head = self
                        .depot
                        .trains
                        .first()
                        .map(|t| t.id.clone())
                        .ok_or_else(|| GameError::new("empty depot"))?;
                    self.g1822_buy_train_action(c, &head, *price, variant.as_deref(), None, false)?;
                    self.rndm().passed[i] = true;
                    Ok(())
                }
                Action::Pass { .. } => {
                    self.rndm().passed[i] = true;
                    Ok(())
                }
                _ => err("FirstTurnHousekeeping cannot process this action"),
            },
            Step::AcquireCompany => match action {
                Action::Choose { choice, .. } if self.g1822_is_pnw() => self.g1822_pnw_p20_choose(choice),
                Action::AcquireCompany { company_sym, .. } if self.g1822_is_pnw() && company_sym == "P20" => {
                    let Ent::Corp(c) = ent else { return err("acquire by a non-corporation") };
                    self.g1822_pnw_acquire_p20(c)?;
                    if self.g1822_purchasable_companies(c).is_empty() && self.rnd().p20_choices.is_empty() {
                        self.rndm().passed[i] = true;
                    }
                    Ok(())
                }
                Action::AcquireCompany { company_sym, .. } => {
                    let Ent::Corp(c) = ent else { return err("acquire by a non-corporation") };
                    self.g1822_process_acquire_company(c, company_sym)?;
                    if self.g1822_purchasable_companies(c).is_empty() {
                        self.rndm().passed[i] = true;
                    }
                    Ok(())
                }
                Action::Pass { .. } => {
                    self.rndm().passed[i] = true;
                    Ok(())
                }
                _ => err("AcquireCompany cannot process this action"),
            },
            Step::SpecialChoose if self.g1822_is_mx() => match action {
                Action::ChooseAbility { choice, .. } => {
                    // MX process_choose_ability: `return unless close_p16`
                    if choice == "close_p16" {
                        self.g1822_mx_close_p16();
                    }
                    Ok(())
                }
                _ => err("SpecialChoose cannot process this action"),
            },
            Step::SpecialChoose => match action {
                Action::Choose { choice, .. } => {
                    let Ent::Corp(c) = ent else { return err("choose by a non-corporation") };
                    let co = self
                        .g1822_special_choose_company(c)
                        .ok_or_else(|| GameError::new("nothing to choose"))?;
                    self.g1822_company_made_choice(&co, choice, "special_choose")
                }
                _ => err("SpecialChoose cannot process this action"),
            },
            Step::SpecialTrack => self.g1822_special_track_process(i, ent, action),
            Step::SpecialToken => self.g1822_special_token_process(ent, action),
            Step::Track | Step::AcquisitionTrack => self.g1822_track_process(i, ent, action),
            Step::AssignSawmill => self.g1822_ca_sawmill_process(i, ent, action),
            Step::DestinationToken if self.g1822_is_ca() => match action {
                Action::HexToken { .. } | Action::PlaceToken { .. } => {
                    let Ent::Corp(c) = ent else { return err("destination token by a non-corporation") };
                    let c = c.clone();
                    self.g1822_ca_process_destination(&c, action)?;
                    self.rndm().passed[i] = true;
                    Ok(())
                }
                Action::Pass { .. } => {
                    self.rndm().passed[i] = true;
                    Ok(())
                }
                _ => err("DestinationToken cannot process this action"),
            },
            Step::DestinationToken => match action {
                Action::HexToken { hex_id, .. } => {
                    let Ent::Corp(c) = ent else { return err("hex_token by a non-corporation") };
                    self.g1822_process_destination_token(c, hex_id)?;
                    self.rndm().passed[i] = true;
                    Ok(())
                }
                Action::Pass { .. } => {
                    self.rndm().passed[i] = true;
                    Ok(())
                }
                _ => err("DestinationToken cannot process this action"),
            },
            Step::Token => self.g1822_token_process(i, ent, action),
            Step::Route => self.g1822_route_process(i, ent, action),
            Step::Dividend => match action {
                Action::Dividend { kind, .. } => {
                    let Ent::Corp(c) = ent else { return err("dividend by a non-corporation") };
                    self.g1822_process_dividend(c, kind.as_str())?;
                    self.rndm().passed[i] = true;
                    self.rndm().extra_train_choice.clear();
                    Ok(())
                }
                Action::Choose { choice, .. } => {
                    if !matches!(choice.as_str(), "payout" | "withhold" | "half") {
                        return err(format!("{choice} is an illegal choice"));
                    }
                    self.rndm().extra_train_choice = choice.clone();
                    Ok(())
                }
                _ => err("Dividend cannot process this action"),
            },
            Step::BuyTrain => self.g1822_buy_train_process(i, ent, action),
            Step::MinorAcquisition => self.g1822_minor_acquisition_process(i, ent, action),
            Step::IssueShares => self.g1822_issue_shares_process(i, ent, action),
            Step::CashOutNdem => match action {
                Action::Pass { .. } => {
                    self.g1822_mx_cash_out();
                    Ok(())
                }
                _ => err("CashOutNdem cannot process this action"),
            },
            Step::AuctionNdemTokens => self.g1822_mx_auction_process(action),
            _ => err(format!("{step:?} cannot process actions")),
        }
    }

    /// Ruby `skip!` per OR step (default `pass!`).
    pub(crate) fn g1822_or_step_skip(&mut self, i: usize) {
        match self.rnd().steps[i] {
            Step::Dividend => {
                let Some(c) = self.g1822_operator() else {
                    self.rndm().passed[i] = true;
                    return;
                };
                // G1822 Dividend#skip!: majors → base skip! (withhold dummy);
                // minors → half if revenue positive else withhold; MX: the
                // NdeM pays out whatever it earned.
                let kind = if self.g1822_mx_is_ndem(&c) {
                    if self.g1822_routes_revenue() > 0 {
                        "payout"
                    } else {
                        "withhold"
                    }
                } else if self.is_major(&c) {
                    "withhold"
                } else if self.g1822_routes_revenue() > 0 {
                    "half"
                } else {
                    "withhold"
                };
                let _ = self.g1822_process_dividend(&c, kind);
                self.rndm().passed[i] = true;
                self.rndm().extra_train_choice.clear();
            }
            Step::MinorAcquisition => {
                self.rndm().passed[i] = true;
            }
            Step::Track if self.g1822_is_mx() && self.g1822_operator().as_deref() == Some(super::mx::NDEM) => {
                self.g1822_mx_track_pass(i);
            }
            _ => self.rndm().passed[i] = true,
        }
    }

    // =====================================================================
    // Route (G1822::Step::Route)
    // =====================================================================

    fn g1822_route_actions(&self, ent: &Ent) -> Vec<&'static str> {
        // G1822 Route#actions drops the base `entity == current_entity`
        // check: any operator with runnable trains may submit run_routes
        // while the Route step is active (248432 @1074: minor 10's empty
        // run_routes ends minor 27's Route step).
        let Ent::Corp(c) = ent else { return vec![] };
        if self.corp_idx.get(c).map_or(true, |&i| self.corporations[i].closed) {
            return vec![];
        }
        let trains = self.g1822_route_trains(c);
        if trains.is_empty() || !self.g1822_can_run_route(c) {
            return vec![];
        }
        if self.is_minor(c) && trains.iter().all(|t| t.name == self.fam().e_train) {
            return vec![];
        }
        let mut a = vec!["run_routes"];
        let choosing = if self.g1822_is_ca() {
            self.g1822_ca_choosing(c).is_some()
        } else {
            self.g1822_choosing_pullman(c)
        };
        if choosing {
            a.push("choose");
        }
        a
    }

    /// `route_trains`: runnable trains minus the pullman.
    pub(crate) fn g1822_route_trains(&self, c: &str) -> Vec<crate::entities::Train> {
        let ci = self.ci(c);
        let ca = self.g1822_is_ca();
        self.corporations[ci]
            .trains
            .iter()
            .filter(|t| t.name != self.fam().extra_train_pullman && !t.operated)
            // 1822CA route_trains: never the grain train
            .filter(|t| !(ca && Self::g1822_ca_grain_train(t)))
            .cloned()
            .collect()
    }

    /// G1822 `can_run_route?`: any local train, else the graph's
    /// route_available.
    pub(crate) fn g1822_can_run_route(&self, c: &str) -> bool {
        let ci = self.ci(c);
        if self.corporations[ci]
            .trains
            .iter()
            .any(|t| self.fam().local_trains.contains(&t.name.as_str()))
        {
            return true;
        }
        // Graph#route_info: a cached purchase-grade route survives
        // Graph#clear (1822 never calls clear_graph_for), so once a
        // corporation's route_info was computed with one it stays.
        if self.g().route_purchase.contains(c) {
            return true;
        }
        let g = self.g1822_graph(c);
        if g.route_train_purchase {
            self.g().route_purchase.insert(c);
        }
        g.route_available
    }

    pub(crate) fn g1822_pullman_choices(&self, c: &str) -> Vec<String> {
        self.g1822_route_trains(c)
            .into_iter()
            .filter(|t| !self.fam().local_trains.contains(&t.name.as_str()) && t.name != self.fam().e_train)
            .map(|t| t.id)
            .collect()
    }

    fn g1822_choosing_pullman(&self, c: &str) -> bool {
        let ci = self.ci(c);
        self.rnd().pullman_train.is_none()
            && self.corporations[ci].trains.iter().any(|t| t.name == self.fam().extra_train_pullman)
            && !self.g1822_pullman_choices(c).is_empty()
    }

    fn g1822_route_process(&mut self, i: usize, ent: &Ent, action: &Action) -> Result<(), GameError> {
        let Ent::Corp(c) = ent else { return err("route by a non-corporation") };
        match action {
            Action::Choose { choice, .. } if self.g1822_is_ca() => {
                let c = c.clone();
                self.g1822_ca_route_choose(&c, choice)
            }
            Action::Choose { choice, .. } => {
                // pullman_train_choices(entity)[action.choice.to_i]: Ruby's
                // String#to_i (a non-number is 0, a negative counts from
                // the end; out of range raises on nil.name)
                let choices = self.g1822_pullman_choices(c);
                let n = ruby_to_i(choice);
                let idx = if n < 0 { choices.len() as i64 + n } else { n };
                let tid = usize::try_from(idx)
                    .ok()
                    .and_then(|i| choices.get(i).cloned())
                    .ok_or_else(|| GameError::new("bad pullman choice"))?;
                // attach_pullman: the train becomes "<name>+" (city visit =
                // its distance, everything pays) for this run.
                let ci = self.ci(c);
                let t = self.corporations[ci].trains.iter_mut().find(|t| t.id == tid).unwrap();
                let orig = (t.name.clone(), t.distance);
                t.name = format!("{}+", t.name);
                self.gm().round.pullman_original = Some(orig);
                self.gm().round.pullman_train = Some(tid);
                Ok(())
            }
            Action::RunRoutes { routes, extra_revenue, .. } => {
                let ci = self.ci(c);
                let mut seen: Vec<String> = Vec::new();
                let mut run = Vec::new();
                // Routes recorded without revenue are priced (and checked).
                let priced: Option<Vec<i32>> = if routes.iter().any(|r| r.revenue.is_none()) {
                    Some(self.g1822_price_recorded_routes(c, routes)?)
                } else {
                    None
                };
                for (ri, r) in routes.iter().enumerate() {
                    let Some(t) = self.corporations[ci].trains.iter().find(|t| t.id == r.train_name) else {
                        return err(format!("Cannot run another corporation's train {}", r.train_name));
                    };
                    if seen.contains(&t.id) {
                        return err("Cannot run train twice");
                    }
                    if t.operated {
                        return err("Cannot run train that operated");
                    }
                    seen.push(t.id.clone());
                    let revenue = match (&priced, r.revenue) {
                        (Some(p), _) => p[ri],
                        (None, Some(v)) => v,
                        (None, None) => 0,
                    };
                    run.push(RunRoute {
                        train_id: t.id.clone(),
                        revenue,
                        subsidy: 0,
                        connections: r.connections.clone(),
                        hexes: r.hexes.clone(),
                        nodes: r.nodes.clone(),
                    });
                }
                self.rndm().routes = run;
                self.rndm().extra_revenue = *extra_revenue;
                self.rndm().passed[i] = true;
                // detach_pullman: the step's @pullman_train, whoever owns it
                // (Route#actions offers `choose` to any corporation, so
                // another one's choose can attach its train meanwhile)
                if let (Some(tid), Some((name, _))) = (self.rnd().pullman_train.clone(), self.rnd().pullman_original.clone()) {
                    if let Some(t) = self.corporations.iter_mut().flat_map(|co| co.trains.iter_mut()).find(|t| t.id == tid) {
                        t.name = name;
                    }
                }
                self.rndm().pullman_train = None;
                self.rndm().pullman_original = None;
                if self.g1822_is_ca() {
                    self.g1822_ca_detach_grain();
                }
                Ok(())
            }
            _ => err("Route cannot process this action"),
        }
    }

    // =====================================================================
    // Dividend (G1822::Step::Dividend)
    // =====================================================================

    pub(crate) fn g1822_routes_revenue(&self) -> i32 {
        self.rnd().routes.iter().map(|r| r.revenue).sum()
    }

    /// `find_extra_train`: the corp's permanent extra train (2P/LP) whose
    /// route revenue is positive and not the whole run.
    pub(crate) fn g1822_find_extra_train(&self, c: &str) -> Option<String> {
        let ci = self.ci(c);
        let t = self.corporations[ci]
            .trains
            .iter()
            .find(|t| self.fam().extra_train_permanents.contains(&t.name.as_str()))?;
        let rev = self
            .rnd()
            .routes
            .iter()
            .find(|r| r.train_id == t.id)
            .map_or(0, |r| r.revenue);
        (rev > 0 && rev != self.g1822_routes_revenue()).then(|| t.id.clone())
    }

    fn total_shares(&self, c: &str) -> i32 {
        let co = &self.corporations[self.ci(c)];
        100 / co.share_unit_percent as i32
    }

    /// `half_pay_withhold_amount`.
    fn half_withhold(&self, c: &str, revenue: i32) -> i32 {
        let half = revenue / 2;
        if self.is_minor(c) {
            half
        } else {
            let ts = self.total_shares(c);
            (half / ts) * ts
        }
    }

    /// (corporation retained, per-share payout) for one dividend type.
    fn payout_of(&self, c: &str, kind: &str, revenue: i32) -> (i32, f64) {
        let ts = self.total_shares(c) as f64;
        match kind {
            "payout" => (0, revenue as f64 / ts),
            "half" => {
                let w = self.half_withhold(c, revenue);
                (w, (revenue - w) as f64 / ts)
            }
            _ => (revenue, 0.0),
        }
    }

    /// G1822 `process_dividend`.
    pub(crate) fn g1822_process_dividend(&mut self, c: &str, kind: &str) -> Result<(), GameError> {
        if self.g1822_is_pnw() {
            self.g1822_pnw_pay_portage_fee();
        }
        let total = self.g1822_routes_revenue();
        let subsidy = self.g1822_routes_subsidy(c);
        let extra_choice = self.rnd().extra_train_choice.clone();
        let (mut corp_part, mut per_share) = (0, 0.0);
        if !extra_choice.is_empty() {
            if let Some(tid) = self.g1822_find_extra_train(c) {
                let extra_rev = self.rnd().routes.iter().find(|r| r.train_id == tid).map_or(0, |r| r.revenue);
                let (ec, eps) = self.payout_of(c, &extra_choice, extra_rev);
                let (bc, bps) = self.payout_of(c, kind, total - extra_rev);
                corp_part = ec + bc;
                per_share = eps + bps;
            } else {
                let (bc, bps) = self.payout_of(c, kind, total);
                corp_part = bc;
                per_share = bps;
            }
        } else {
            let (bc, bps) = self.payout_of(c, kind, total);
            corp_part = bc;
            per_share = bps;
        }
        // operating_history entry; trains operated.
        *self.gm().operating_history.entry(c.to_string()).or_insert(0) += 1;
        let ci = self.ci(c);
        for t in &mut self.corporations[ci].trains {
            t.operated = true;
        }
        self.rndm().routes.clear();
        // payout_corporation(corporation + subsidy)
        let to_corp = corp_part + subsidy;
        if to_corp > 0 {
            self.g1822_bank_pay(&EntityId::corporation(c), to_corp);
        }
        if per_share > 0.0 {
            self.g1822_payout_shares(c, per_share);
        }
        // share price change on the distributed amount
        let distributed = total - corp_part;
        self.g1822_dividend_price_change(c, distributed);
        Ok(())
    }

    /// Base payout_shares (players + corporations; treasury shares pay the
    /// corp itself — holder_for_corporation = entity) + the tax haven.
    fn g1822_payout_shares(&mut self, c: &str, per_share: f64) {
        let ci = self.ci(c);
        let unit = self.corporations[ci].share_unit_percent as f64;
        let minor = self.is_minor(c);
        let holders: Vec<EntityId> = self
            .players
            .iter()
            .map(|p| EntityId::player(p.id))
            .collect();
        for h in holders {
            let pct = self.corporations[ci].percent_owned_by(&h) as f64;
            let amount = ((pct / unit) * per_share).ceil() as i32;
            if amount != 0 {
                self.g1822_bank_pay(&h, amount);
            }
        }
        // corporations: the corp's own treasury shares (incremental).
        if !minor {
            let pct = self.corporations[ci].percent_owned_by(&EntityId::ipo(c)) as f64;
            let amount = ((pct / unit) * per_share).ceil() as i32;
            if amount != 0 {
                self.g1822_bank_pay(&EntityId::corporation(c), amount);
            }
        }
        // company_tax_haven_payout: num_shares_of (ceiled count) × per
        // share, spent as an integer (to_i).
        if self.g1822_tax_haven_value() > 0 {
            let th = EntityId(super::TAX_HAVEN_OWNER.to_string());
            let pct = self.corporations[ci].percent_owned_by(&th) as f64;
            let n = (pct / unit).ceil();
            let amount = (n * per_share) as i32;
            if amount > 0 {
                self.g1822_bank_spend(amount);
                self.gm().tax_haven_cash += amount;
            }
        }
    }

    /// G1822 `share_price_change` + base `change_share_price`.
    fn g1822_dividend_price_change(&mut self, c: &str, revenue: i32) {
        if self.g1822_mx_is_ndem(c) {
            return self.g1822_mx_ndem_price_change(revenue);
        }
        let ci = self.ci(c);
        let Some(sp) = self.corporations[ci].share_price.clone() else { return };
        if revenue <= 0 {
            self.g1822_move_price(c, Dir::Left);
            return;
        }
        let major = self.is_major(c);
        let mut times = 0;
        if revenue >= sp.price || !major {
            times = 1;
        }
        if revenue >= sp.price * 2 && major {
            times = 2;
        }
        // PNW Dividend#share_price_change: 3 → for a triple dividend ≤ 150
        if self.g1822_is_pnw() && revenue >= sp.price * 3 && sp.price <= 150 && major {
            times = 3;
        }
        for _ in 0..times {
            self.g1822_move_price(c, Dir::Right);
        }
    }

    /// G1822 `routes_subsidy`: mail contracts (P6/P7) pay half the base
    /// value of the first and last stops of the best route(s).
    fn g1822_routes_subsidy(&self, c: &str) -> i32 {
        let owner = EntityId::corporation(c);
        let contracts = self.fam().private_mail_contracts
            .iter()
            .filter(|p| !self.co_closed(p) && self.co_owner(p) == owner)
            .count();
        if self.g1822_is_ca() && !self.rnd().routes.is_empty() {
            // CA mail_contract_bonus: G1822's (large) bonuses + the small ones
            let large = if contracts == 0 { Vec::new() } else { self.g1822_large_mail_bonuses(c, contracts) };
            return self.g1822_ca_mail_subsidy(c, large);
        }
        if contracts == 0 || self.rnd().routes.is_empty() {
            return 0;
        }
        self.g1822_large_mail_bonuses(c, contracts).into_iter().sum()
    }

    /// G1822 `mail_contract_bonus`'s subsidies: per route half the base
    /// value of its first and last stops, best first, one per contract.
    fn g1822_large_mail_bonuses(&self, c: &str, contracts: usize) -> Vec<i32> {
        let mut subsidies: Vec<i32> = Vec::new();
        for r in &self.rnd().routes {
            let ci = self.ci(c);
            let Some(t) = self.corporations[ci].trains.iter().find(|t| t.id == r.train_id) else { continue };
            let stops = self.g1822_route_stop_revenues(r);
            if stops.is_empty() {
                continue;
            }
            // LOCAL_TRAIN_CAN_CARRY_MAIL (1822PNW) lets L trains and
            // single-stop runs carry mail too.
            if !self.fam().local_train_can_carry_mail && (t.name.starts_with('L') || stops.len() < 2) {
                continue;
            }
            let first = stops[0];
            let last = if stops.len() < 2 { 0 } else { *stops.last().unwrap() };
            subsidies.push((first + last) / 2);
        }
        subsidies.sort_by(|a, b| b.cmp(a));
        subsidies.into_iter().take(contracts).collect()
    }

    // =====================================================================
    // IssueShares (G1822::Step::IssueShares)
    // =====================================================================

    /// G1822 `issuable_shares`: majors with treasury shares that have
    /// operated twice; bundles of its own treasury that fit the pool.
    pub(crate) fn g1822_issuable_bundles(&self, c: &str) -> Vec<Bundle> {
        if !self.is_major(c) {
            return vec![];
        }
        let ci = self.ci(c);
        let ipo = EntityId::ipo(c);
        let n_ipo = self.corporations[ci].percent_owned_by(&ipo);
        if n_ipo == 0 || self.g().operating_history.get(c).copied().unwrap_or(0) < 2 {
            return vec![];
        }
        self.g1822_bundles_for(&ipo, c)
            .into_iter()
            .filter(|b| self.g1822_fit_in_bank(b))
            .collect()
    }

    /// G1822 `redeemable_shares`: pool bundles the treasury can afford.
    pub(crate) fn g1822_redeemable_bundles(&self, c: &str) -> Vec<Bundle> {
        if !self.is_major(c) {
            return vec![];
        }
        let cash = self.corporations[self.ci(c)].cash;
        self.g1822_bundles_for(&EntityId::market(), c)
            .into_iter()
            .filter(|b| self.g1822_bundle_price(b) <= cash)
            .collect()
    }

    fn g1822_issue_shares_actions(&self, ent: &Ent) -> Vec<&'static str> {
        let mut a: Vec<&'static str> = Vec::new();
        // PNW IssueShares#actions: nothing for minors
        if self.g1822_is_pnw() && matches!(ent, Ent::Corp(c) if self.is_minor(c)) {
            return a;
        }
        match ent {
            Ent::Corp(c) if self.is_operator(ent) => {
                if !self.g1822_redeemable_bundles(c).is_empty() {
                    a.push("buy_shares");
                }
                if !self.g1822_issuable_bundles(c).is_empty() {
                    a.push("sell_shares");
                }
                if !a.is_empty() {
                    a.push("pass");
                }
                // LUR special case: a corp owning P17 stops here.
                if a.is_empty() && self.g1822_owns(c, self.fam().company_lur) {
                    a.push("choose_ability");
                }
            }
            Ent::Company(co) => {
                if !self.g1822_company_choices(co, "issue").is_empty() {
                    a.push("choose_ability");
                }
            }
            _ => {}
        }
        if !a.is_empty() && !a.contains(&"pass") {
            a.push("pass");
        }
        a
    }

    fn g1822_issue_shares_process(&mut self, i: usize, ent: &Ent, action: &Action) -> Result<(), GameError> {
        match action {
            Action::SellShares {
                corporation_sym,
                percent,
                share_indices,
                ..
            } => {
                let Ent::Corp(c) = ent else { return err("issue by a non-corporation") };
                if corporation_sym != c {
                    return err("can only issue own shares");
                }
                let b = self.g1822_action_bundle_pub(c, *percent, share_indices)?;
                if self.g1822_is_pnw() || self.g1822_is_mx() || self.g1822_is_ca() {
                    // PNW / MX / CA: sell, then one step left per certificate
                    self.g1822_pool_sell_shares(&b, true);
                    for _ in 0..b.shares.len() {
                        self.g1822_move_price(c, super::shares::Dir::Left);
                    }
                } else {
                    self.g1822_sell_shares_and_change_price(&b);
                }
                self.rndm().passed[i] = true;
                Ok(())
            }
            Action::BuyShares {
                corporation_sym,
                percent,
                share_indices,
                ..
            } => {
                let Ent::Corp(c) = ent else { return err("redeem by a non-corporation") };
                if corporation_sym != c {
                    return err("can only redeem own shares");
                }
                let b = self.g1822_action_bundle_pub(c, *percent, share_indices)?;
                self.g1822_pool_buy_shares(&EntityId::ipo(c), &b, None, false, true)?;
                self.rndm().passed[i] = true;
                Ok(())
            }
            Action::ChooseAbility { choice, .. } => {
                let Ent::Company(co) = ent else { return err("choose_ability by a non-company") };
                self.g1822_company_made_choice(co, choice, "issue")
            }
            Action::Pass { .. } => {
                self.rndm().passed[i] = true;
                Ok(())
            }
            _ => err("IssueShares cannot process this action"),
        }
    }

    pub(crate) fn g1822_operated(&self, c: &str) -> bool {
        self.g().operating_history.get(c).copied().unwrap_or(0) > 0
    }

    pub(crate) fn g1822_owns(&self, corp: &str, company: &str) -> bool {
        !self.co_closed(company) && self.co_owner(company) == EntityId::corporation(corp)
    }

    #[allow(dead_code)]
    fn corp_type_of(&self, c: &str) -> CorpType {
        self.corporations[self.ci(c)].corp_type
    }
}

/// Ruby `String#to_i`: leading whitespace, an optional sign, then digits
/// (underscores between them allowed); anything else ends it (no digits: 0).
pub(crate) fn ruby_to_i(s: &str) -> i64 {
    let t = s.trim_start();
    let (neg, rest) = match t.as_bytes().first() {
        Some(b'-') => (true, &t[1..]),
        Some(b'+') => (false, &t[1..]),
        _ => (false, t),
    };
    let mut v: i64 = 0;
    let mut prev_digit = false;
    for ch in rest.chars() {
        if let Some(d) = ch.to_digit(10) {
            v = v.saturating_mul(10).saturating_add(d as i64);
            prev_digit = true;
        } else if ch == '_' && prev_digit {
            prev_digit = false;
        } else {
            break;
        }
    }
    if neg {
        -v
    } else {
        v
    }
}
