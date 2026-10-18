//! 1822 legal-move generation (the RL-readiness contract, item 1).
//!
//! `g1822_legal_moves()` enumerates, for the live state, the moves Ruby's
//! UI offers the acting entity — the current entity of the active step and
//! the companies it owns that act in parallel — one [`Move`] per distinct
//! choice. Price-bearing moves (bids, buying another corporation's train,
//! partial debt payoff) carry a [`PriceRange`]; the move's own action holds
//! the range's low end.
//!
//! Soundness: each generator applies the rules Ruby checks when a PLAYER
//! acts (connectivity, tile legality, the UI's offers) — replay skips them
//! like Ruby's `@game.loading` — and every candidate is then applied to a
//! clone, so the process-side rules (cash, slots, step gates) filter too.
//! `g1822_generator_status` names, per step and action type, whether a
//! generator exists or the type is replay-only; the in-crate fuzzer
//! (tests below) fails on any panic, any generated move that does not
//! apply, and any unfinished state with no moves.

use std::collections::{BTreeMap, HashMap};

use serde_json::{json, Map, Value};

use crate::actions::{Action, DividendKind, GameError, RouteData};
use crate::entities::EntityId;
use crate::game::BaseGame;
use crate::title::{CorpType, CrossPlayerPolicy};

use super::shares::Bundle;
use super::tile_rules::RotRule;
use super::track::AbKind;
use super::family::Family;
use super::{AcquireState, Ent, RoundKind, Step};

/// An inclusive price range with a step (`lo + k*step <= hi`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PriceRange {
    pub lo: i32,
    pub hi: i32,
    pub step: i32,
}

/// One legal move.
#[derive(Clone, Debug)]
pub struct Move {
    pub action: Action,
    /// lay_tile `combo_entities` (Ruby SpecialTrack company combos).
    pub combo: Vec<String>,
    /// buy/sell `share_price` (issued bundles name their reduced price).
    pub share_price: Option<i32>,
    /// The price-bearing parameter's range (the action holds `lo`).
    pub price: Option<PriceRange>,
    /// The share certificates' Ruby ids, for the wire form.
    pub shares: Vec<String>,
    /// The generator already checked everything the process checks (tile
    /// lays): skip the apply-on-a-clone filter.
    pub validated: bool,
}

impl Move {
    pub(crate) fn plain(action: Action) -> Move {
        Move {
            action,
            combo: Vec::new(),
            share_price: None,
            price: None,
            shares: Vec::new(),
            validated: false,
        }
    }

    /// The move with its price-bearing parameter set to `p`.
    pub fn with_price(&self, p: i32) -> Move {
        let mut m = self.clone();
        match &mut m.action {
            Action::Bid { price, .. } => *price = p,
            Action::BuyTrain { price, .. } => *price = p,
            Action::PayoffPlayerDebtPartial { amount, .. } => *amount = p,
            _ => {}
        }
        m
    }
}

/// Whether a (step, action type) pair has a generator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GenStatus {
    Generated,
    /// Never generated; the reason says why replay still accepts it.
    ReplayOnly(&'static str),
}

/// The RL-readiness table: every action type an 1822 step can offer.
pub fn g1822_generator_status(step: Step, action_type: &str) -> GenStatus {
    use GenStatus::*;
    match (step, action_type) {
        (Step::BuySellParShares, "bid" | "buy_shares" | "sell_shares" | "par" | "pass")
        | (Step::BuySellParShares, "payoff_player_debt" | "payoff_player_debt_partial" | "choose_ability") => {
            Generated
        }
        // 1822PNW: the conversion's exchange choice (pnw_legal)
        (Step::BuySellParShares, "choose") => Generated,
        (Step::Choose, "choose" | "pass") => Generated,
        (Step::DiscardTrain, "discard_train") => Generated,
        (Step::PendingToken, "place_token") => Generated,
        (Step::FirstTurnHousekeeping, "buy_train" | "pass") => Generated,
        (Step::AcquireCompany, "acquire_company" | "pass") => Generated,
        // 1822PNW (pnw_legal): P20's associated-minor choice, the Assign
        // step's paper mill / ski haus, the Merger round
        (Step::AcquireCompany, "choose") => Generated,
        (Step::Assign, "assign") => Generated,
        (Step::Merge, "merge" | "choose" | "pass") => Generated,
        (Step::SpecialChoose, "choose") => Generated,
        (Step::SpecialTrack, "lay_tile" | "pass") => Generated,
        (Step::SpecialToken, "place_token" | "pass") => Generated,
        (Step::Track, "lay_tile" | "pass" | "choose_ability") => Generated,
        (Step::DestinationToken, "hex_token" | "pass") => Generated,
        (Step::Token, "place_token" | "pass" | "choose_ability") => Generated,
        (Step::Route, "run_routes") => Generated,
        (Step::Route, "choose") => Generated,
        (Step::Dividend, "dividend" | "choose") => Generated,
        (Step::BuyTrain, "buy_train" | "sell_shares" | "pass") => Generated,
        (Step::MinorAcquisition, "merge" | "choose" | "pass") => Generated,
        (Step::IssueShares, "buy_shares" | "sell_shares" | "pass" | "choose_ability") => Generated,
        // 1822MX (mx_legal): P16's close_p16, the NdeM's cash-out and its
        // token auction
        (Step::SpecialChoose, "choose_ability") => Generated,
        (Step::CashOutNdem, "pass") => Generated,
        (Step::AuctionNdemTokens, "remove_token" | "bid" | "merge" | "pass") => Generated,
        _ => ReplayOnly("not offered by this step"),
    }
}

/// Generator profiling counters (cheap; read by the fuzz probe).
pub mod stats {
    use std::sync::atomic::{AtomicU64, Ordering};
    pub static CANDS: AtomicU64 = AtomicU64::new(0);
    pub static VALIDATE_NS: AtomicU64 = AtomicU64::new(0);
    pub static LAYS_NS: AtomicU64 = AtomicU64::new(0);
    pub static RESTRICT_NS: AtomicU64 = AtomicU64::new(0);
    pub static RESTRICT_N: AtomicU64 = AtomicU64::new(0);
    /// States where the cross-player policy left no move (rules' set used).
    pub static CROSS_FALLBACK: AtomicU64 = AtomicU64::new(0);
    pub fn add(c: &AtomicU64, v: u64) {
        c.fetch_add(v, Ordering::Relaxed);
    }
    pub fn get(c: &AtomicU64) -> u64 {
        c.load(Ordering::Relaxed)
    }
}

fn player_ent(pid: u32) -> String {
    pid.to_string()
}

impl BaseGame {
    // =====================================================================
    // Entry points
    // =====================================================================

    /// Every legal move for the acting entity (see the module docs),
    /// under the title's cross-player policy
    /// (`GameTitle::generator_cross_player`).
    pub fn g1822_legal_moves(&mut self) -> Vec<Move> {
        let policy = self.title_def().generator_cross_player();
        self.g1822_legal_moves_with(policy)
    }

    /// Every legal move under `policy`: moves it excludes (a purchase from
    /// another player's holdings) are dropped -- unless nothing else is
    /// legal, when the rules' set is returned rather than a dead end.
    /// `CrossPlayerPolicy::RULES` = everything Ruby accepts.
    pub fn g1822_legal_moves_with(&mut self, policy: CrossPlayerPolicy) -> Vec<Move> {
        if self.finished || self.g1822.is_none() {
            return Vec::new();
        }
        // The UI-side graph computes the generator makes (e.g. a minor
        // acquisition's connection checks) are not Ruby process computes:
        // they must not fill the route latch or the Ruby graph cache.
        let memo = (self.g().route_purchase.clone(), self.g().ruby_graphs.clone());
        let out = self.g1822_legal_moves_inner(policy);
        let gm = self.gm();
        gm.route_purchase = memo.0;
        gm.ruby_graphs = memo.1;
        out
    }

    fn g1822_legal_moves_inner(&mut self, policy: CrossPlayerPolicy) -> Vec<Move> {
        let Some(bi) = self.g1822_active_step() else { return Vec::new() };
        let step = self.rnd().steps[bi];
        let Some(ent) = self.g1822_step_current_entity(bi) else { return Vec::new() };
        let mut cands: Vec<Move> = match step {
            Step::BuySellParShares => self.gen_bsps(&ent),
            Step::Choose => self.gen_choose_round(&ent),
            _ => self.gen_or(bi, step, &ent),
        };
        // Companies acting in parallel at the earlier steps (Ruby
        // actions_for over steps up to the blocking one): SpecialTrack /
        // SpecialToken company lays and tokens during the operator's turn.
        if self.rnd().kind == RoundKind::Operating && matches!(ent, Ent::Corp(_)) {
            cands.extend(self.gen_parallel_companies(bi, &ent));
        }
        let (kept, dropped): (Vec<Move>, Vec<Move>) =
            cands.into_iter().partition(|m| !self.g1822_cross_player_excluded(m, policy));
        let out = self.g1822_validate(kept);
        if out.is_empty() && !dropped.is_empty() {
            stats::add(&stats::CROSS_FALLBACK, 1);
            return self.g1822_validate(dropped);
        }
        out
    }

    /// Whether `policy` excludes `m`: a purchase whose seller is not the
    /// buying corporation's president (a train from a corporation or
    /// minor another player presides, a private another player owns, a
    /// minor another player owns). Bank sellers (depot, bid-box minors)
    /// are never excluded; a missing president counts as another player.
    pub(crate) fn g1822_cross_player_excluded(&self, m: &Move, policy: CrossPlayerPolicy) -> bool {
        let other = |buyer: &str, seller: Option<u32>| {
            let b = self.g1822_corp_president(buyer);
            b.is_none() || seller != b
        };
        match &m.action {
            Action::BuyTrain { entity_id, from, .. } if !policy.trains && from != "depot" => {
                self.corp_idx.contains_key(from.as_str()) && other(entity_id, self.g1822_corp_president(from))
            }
            Action::AcquireCompany { entity_id, company_sym } if !policy.companies => {
                other(entity_id, self.co_owner(company_sym).player_id())
            }
            Action::Merge { entity_id, corporation_sym } if !policy.minors => {
                // bid-box minors belong to the bank: never excluded
                match self.g1822_corp_president(corporation_sym) {
                    Some(owner) => other(entity_id, Some(owner)),
                    None => false,
                }
            }
            _ => false,
        }
    }

    /// Debug: every candidate before the apply filter, with its error.
    pub fn g1822_candidates_debug(&mut self) -> Vec<(String, Option<String>)> {
        let memo = (self.g().route_purchase.clone(), self.g().ruby_graphs.clone());
        let out = self.g1822_candidates_debug_inner();
        let gm = self.gm();
        gm.route_purchase = memo.0;
        gm.ruby_graphs = memo.1;
        out
    }

    fn g1822_candidates_debug_inner(&mut self) -> Vec<(String, Option<String>)> {
        let Some(bi) = self.g1822_active_step() else { return Vec::new() };
        let step = self.rnd().steps[bi];
        let Some(ent) = self.g1822_step_current_entity(bi) else { return Vec::new() };
        let mut cands: Vec<Move> = match step {
            Step::BuySellParShares => self.gen_bsps(&ent),
            Step::Choose => self.gen_choose_round(&ent),
            _ => self.gen_or(bi, step, &ent),
        };
        if self.rnd().kind == RoundKind::Operating && matches!(ent, Ent::Corp(_)) {
            cands.extend(self.gen_parallel_companies(bi, &ent));
        }
        cands
            .into_iter()
            .map(|m| {
                let mut g = self.clone_for_search();
                let e = g.g1822_apply_move_nobackup(&m).err().map(|e| e.to_string());
                (format!("{:?}", m.action), e)
            })
            .collect()
    }

    /// Keep the candidates that apply (on a clone).
    fn g1822_validate(&self, cands: Vec<Move>) -> Vec<Move> {
        let t = std::time::Instant::now();
        stats::add(&stats::CANDS, cands.len() as u64);
        let mut out: Vec<Move> = Vec::new();
        for m in cands {
            if out.iter().any(|o| same_move(o, &m)) {
                continue;
            }
            if m.validated {
                out.push(m);
                continue;
            }
            let mut g = self.clone_for_search();
            if g.g1822_apply_move_nobackup(&m).is_ok() {
                out.push(m);
            }
        }
        stats::add(&stats::VALIDATE_NS, t.elapsed().as_nanos() as u64);
        out
    }

    /// Apply a move on a throwaway clone (no rollback copy: the caller
    /// discards the game on error).
    fn g1822_apply_move_nobackup(&mut self, m: &Move) -> Result<(), GameError> {
        self.gm().pending_combo = m.combo.clone();
        self.gm().pending_share_price = m.share_price;
        self.g1822_process_action_nobackup(&m.action)
    }

    /// Apply a move (the in-crate path; Python sends the wire dict).
    pub fn g1822_apply_move(&mut self, m: &Move) -> Result<(), GameError> {
        self.gm().pending_combo = m.combo.clone();
        self.gm().pending_share_price = m.share_price;
        let r = self.g1822_process_action(&m.action);
        if self.g1822.is_some() {
            self.gm().pending_combo.clear();
            self.gm().pending_share_price = None;
        }
        r
    }

    // =====================================================================
    // Stock round (G1822 BuySellParShares, BidboxAuction)
    // =====================================================================

    fn gen_bsps(&mut self, ent: &Ent) -> Vec<Move> {
        let Ent::Player(pid) = ent else { return Vec::new() };
        let pid = *pid;
        // 1822PNW: converting an associated minor → its share-count choice
        if self.g1822_is_pnw() && self.rnd().converting_major.is_some() {
            // (an exchange that leaves the player unable to pay the bids
            // they hold dead-ends the round — Ruby raises there; offered
            // only when nothing else is)
            let all: Vec<Move> = self
                .g1822_pnw_convert_choices()
                .into_iter()
                .map(|choice| Move::plain(Action::Choose { entity_id: player_ent(pid), choice }))
                .collect();
            let ok: Vec<Move> = all
                .iter()
                .filter(|m| match &m.action {
                    Action::Choose { choice, .. } => self.pnw_exchange_affordable(pid, choice.parse().unwrap_or(0), None),
                    _ => true,
                })
                .cloned()
                .collect();
            return if ok.is_empty() { all } else { ok };
        }
        let types = self.g1822_bsps_actions(ent);
        let me = player_ent(pid);
        let mut out = Vec::new();
        // P16 tax haven (the current player's): choose_ability.
        let osth = self.fam().company_osth;
        if !self.co_closed(osth) && self.co_owner(osth) == EntityId::player(pid) {
            for choice in self.g1822_tax_haven_choices(osth) {
                out.push(Move::plain(Action::ChooseAbility {
                    entity_id: osth.to_string(),
                    choice,
                }));
            }
        }
        let has = |t: &str| types.iter().any(|x| *x == t);
        if has("sell_shares") {
            out.extend(self.gen_player_sales(pid, false));
            if types.len() == 1 {
                // must_sell?: only sales.
                return out;
            }
        }
        if has("buy_shares") {
            out.extend(self.gen_player_buys(pid));
        }
        if has("par") {
            out.extend(self.gen_pars(pid));
        }
        if has("bid") {
            out.extend(self.gen_bids(pid));
        }
        if has("payoff_player_debt") {
            out.push(Move::plain(Action::PayoffPlayerDebt { entity_id: me.clone() }));
            let cap = self.player_cash(pid).min(self.player_debt(pid));
            if cap >= 1 {
                let mut m = Move::plain(Action::PayoffPlayerDebtPartial {
                    entity_id: me.clone(),
                    amount: 1,
                });
                m.price = Some(PriceRange { lo: 1, hi: cap, step: 1 });
                out.push(m);
            }
        }
        if has("pass") {
            out.push(Move::plain(Action::Pass { entity_id: me }));
        }
        out
    }

    fn bundle_move_ids(&self, b: &Bundle) -> Vec<String> {
        b.shares.iter().map(|i| format!("{}_{}", b.corp, i)).collect()
    }

    /// A player's sellable bundles: the stock round's can_sell? (Ruby
    /// checks it only for live players) or, in the BuyTrain step, the
    /// emergency-money rules.
    fn gen_player_sales(&self, pid: u32, emergency: bool) -> Vec<Move> {
        let eid = EntityId::player(pid);
        let mut out = Vec::new();
        for c in &self.corporations {
            if c.closed {
                continue;
            }
            for b in self.g1822_bundles_for(&eid, &c.sym) {
                let ok = if emergency { self.g1822_emr_can_sell(pid, &b) } else { self.g1822_can_sell(&eid, &b) };
                if !ok {
                    continue;
                }
                let mut m = Move::plain(Action::SellShares {
                    entity_id: player_ent(pid),
                    corporation_sym: c.sym.clone(),
                    shares: Vec::new(),
                    percent: b.percent,
                    share_indices: b.shares.clone(),
                });
                m.shares = self.bundle_move_ids(&b);
                out.push(m);
            }
        }
        out
    }

    /// The first certificate of each percent a holder offers (the UI's
    /// "buy IPO share" / "buy market share" buttons).
    fn offered_certs(&self, sym: &str, owner: &EntityId) -> Vec<usize> {
        let c = &self.corporations[self.ci(sym)];
        let mut out: Vec<usize> = Vec::new();
        for i in c.shares_owned_by_in_order(owner) {
            if !out.iter().any(|&j| c.shares[j].percent == c.shares[i].percent && c.shares[j].president == c.shares[i].president) {
                out.push(i);
            }
        }
        out
    }

    fn gen_player_buys(&self, pid: u32) -> Vec<Move> {
        let mut out = Vec::new();
        for c in &self.corporations {
            if c.closed || c.corp_type != CorpType::Major {
                continue;
            }
            let mut sources = vec![EntityId::market()];
            if c.ipo_price.is_some() {
                sources.push(EntityId::ipo(&c.sym));
            }
            // 1822MX: the NdeM's first (pool) certificate is a plain 10%
            let ndem = self.g1822_mx_is_ndem(&c.sym);
            for src in sources {
                for i in self.offered_certs(&c.sym, &src) {
                    if c.shares[i].president && !ndem {
                        continue;
                    }
                    let b = Bundle {
                        corp: c.sym.clone(),
                        shares: vec![i],
                        percent: c.shares[i].percent,
                        price_override: None,
                    };
                    let mut m = Move::plain(Action::BuyShares {
                        entity_id: player_ent(pid),
                        corporation_sym: c.sym.clone(),
                        shares: Vec::new(),
                        percent: b.percent,
                        source: if src.is_market() { "market".into() } else { "ipo".into() },
                        share_indices: vec![i],
                    });
                    m.shares = self.bundle_move_ids(&b);
                    out.push(m);
                }
            }
        }
        out
    }

    fn gen_pars(&self, pid: u32) -> Vec<Move> {
        let mut out = Vec::new();
        let par_cells: Vec<i32> = {
            let mut v: Vec<i32> = self
                .stock_market
                .share_prices_with_types(&["par_1"])
                .into_iter()
                .map(|sp| sp.price)
                .collect();
            v.sort();
            v.dedup();
            v
        };
        for c in &self.corporations {
            if c.closed || c.corp_type != CorpType::Major {
                continue;
            }
            // 1822PNW get_par_prices: converting one's floated associated
            // minor — the par prices (not 50) some exchange works at
            if self.g1822_is_pnw()
                && self.g1822_pnw_formation(&c.sym, Some(pid)) == super::pnw::Formation::Convertable
            {
                let Some(minor) = self.g1822_pnw_assoc_minor(&c.sym) else { continue };
                let m = &self.corporations[self.ci(&minor)];
                if m.floated {
                    let value = m.share_price.as_ref().map_or(0, |sp| sp.price) * 2;
                    let mut pars: Vec<i32> = self.g1822_pnw_par_prices().into_iter().map(|sp| sp.price).collect();
                    pars.dedup();
                    for p in pars {
                        let shares = Self::g1822_pnw_possible_shares(p, m.cash, value, self.player_cash(pid));
                        if p != 50 && shares.iter().any(|&n| self.pnw_exchange_affordable(pid, n, Some((p, value, m.cash)))) {
                            out.push(Move::plain(Action::Par {
                                entity_id: player_ent(pid),
                                corporation_sym: c.sym.clone(),
                                share_price: p,
                            }));
                        }
                    }
                    continue;
                }
            }
            // 1822PNW ipo_type: a major is parable only while its associated
            // minor is unowned (Ruby's process_par does not check: parring
            // another player's minor's major crashes in after_par)
            if self.g1822_is_pnw() && self.g1822_pnw_formation(&c.sym, Some(pid)) != super::pnw::Formation::Parable {
                continue;
            }
            let mult = c.president_percent() as i32 / 10;
            let avail = self.g1822_available_par_cash(pid, &c.sym, None);
            for &p in &par_cells {
                if p * mult > avail {
                    continue;
                }
                out.push(Move::plain(Action::Par {
                    entity_id: player_ent(pid),
                    corporation_sym: c.sym.clone(),
                    share_price: p,
                }));
            }
        }
        out
    }

    /// BIDDING_TOKENS (the 18xx.games client's lookup works — Opal
    /// strings are symbols — so the UI limits bids by tokens; the server
    /// only checks prices).
    fn bidding_tokens_per_player(&self) -> i32 {
        (self.fam().bidding_tokens)(self.players.len())
    }

    fn gen_bids(&self, pid: u32) -> Vec<Move> {
        let mut out = Vec::new();
        let mut targets = self.bidbox_minors();
        targets.extend(self.bidbox_concessions());
        targets.extend(self.bidbox_privates());
        let committed = self.g1822_committed_cash(pid);
        let max = self.player_cash(pid) - committed;
        let my_bids = self.rnd().bids.values().filter(|bs| bs.iter().any(|b| b.player == pid)).count() as i32;
        let tokens = self.bidding_tokens_per_player() - my_bids;
        let certs_ok = self.g1822_num_certs_with_bids(pid) < self.cert_limit as i32;
        for co in targets {
            let Some(ci) = self.co_idx(&co) else { continue };
            let high = self.g1822_highest_bid(&co);
            let inc = self.fam().min_bid_increment;
            let min = match &high {
                Some(h) => h.price + inc,
                None => self.companies[ci].min_bid(),
            };
            // can_bid_company?
            if !certs_ok || max < min {
                continue;
            }
            let mine = self.rnd().bids.get(&co).and_then(|bs| bs.iter().find(|b| b.player == pid)).cloned();
            if let (Some(m), Some(h)) = (&mine, &high) {
                if m.price >= h.price {
                    continue; // already the highest bidder
                }
            }
            if mine.is_none() && tokens <= 0 {
                continue;
            }
            let hi = min + ((max - min) / inc) * inc;
            let mut m = Move::plain(Action::Bid {
                entity_id: player_ent(pid),
                company_sym: co.clone(),
                price: min,
            });
            m.price = Some(PriceRange { lo: min, hi, step: inc });
            out.push(m);
        }
        out
    }

    // =====================================================================
    // Choices round (G1822::Step::Choose — P9 double cash)
    // =====================================================================

    fn gen_choose_round(&mut self, ent: &Ent) -> Vec<Move> {
        let Ent::Player(pid) = ent else { return Vec::new() };
        if self.g1822_choose_actions(ent).is_empty() {
            return Vec::new();
        }
        vec![
            Move::plain(Action::Choose {
                entity_id: player_ent(*pid),
                choice: "double".to_string(),
            }),
            Move::plain(Action::Pass {
                entity_id: player_ent(*pid),
            }),
        ]
    }

    // =====================================================================
    // Operating round
    // =====================================================================

    fn gen_or(&mut self, bi: usize, step: Step, ent: &Ent) -> Vec<Move> {
        let types = self.g1822_step_actions(bi, ent);
        if types.is_empty() {
            return Vec::new();
        }
        let has = |t: &str| types.iter().any(|x| *x == t);
        let eid = ent.id();
        let pass = || Move::plain(Action::Pass { entity_id: eid.clone() });
        let mut out = Vec::new();
        match step {
            // 1822PNW: the Merger round (Assign never blocks: its
            // companies act in parallel, see gen_parallel_companies)
            Step::Merge => out.extend(self.gen_pnw_merge(ent)),
            Step::Assign => {}
            Step::DiscardTrain => {
                if let Ent::Corp(c) = ent {
                    out.extend(self.gen_discards(c));
                }
            }
            Step::PendingToken => out.extend(self.gen_pending_tokens()),
            Step::FirstTurnHousekeeping => {
                if let (Ent::Corp(c), Some(head)) = (ent, self.depot.trains.first().cloned()) {
                    // The UI offers a fresh clone of the head L card (Ruby
                    // `Train.new(name: 'L')`, id "L-0"); FirstTurnHousekeeping
                    // ignores the id and buys the depot head.
                    out.push(Move::plain(Action::BuyTrain {
                        entity_id: c.clone(),
                        train_name: format!("{}-0", head.name),
                        price: head.price,
                        from: "depot".into(),
                        variant: Some(head.name.clone()),
                        exchange: None,
                    }));
                }
                out.push(pass());
            }
            Step::AcquireCompany => {
                if let Ent::Corp(c) = ent {
                    // 1822PNW: after P20, only its associated-minor choice
                    if self.g1822_is_pnw() && !self.rnd().p20_choices.is_empty() {
                        return self.gen_pnw_p20_choices(c);
                    }
                    for co in self.g1822_purchasable_companies(c) {
                        out.push(Move::plain(Action::AcquireCompany {
                            entity_id: c.clone(),
                            company_sym: co,
                        }));
                    }
                }
                out.push(pass());
            }
            Step::SpecialChoose => {
                if let Ent::Corp(c) = ent {
                    if let Some(co) = self.g1822_special_choose_company(c) {
                        for choice in self.g1822_company_choices(&co, "special_choose") {
                            out.push(Move::plain(Action::Choose {
                                entity_id: c.clone(),
                                choice,
                            }));
                        }
                    }
                }
            }
            Step::SpecialTrack => {
                if let Ent::Company(co) = ent {
                    out.extend(self.gen_company_lays(co, true));
                    if has("pass") {
                        out.push(pass());
                    }
                }
            }
            Step::SpecialToken => {
                if let Ent::Company(co) = ent {
                    out.extend(self.gen_special_tokens(co));
                    if has("pass") {
                        out.push(pass());
                    }
                }
            }
            Step::Track => {
                if let Ent::Corp(c) = ent {
                    if has("lay_tile") {
                        out.extend(self.gen_corp_lays(c));
                    }
                    if has("pass") {
                        out.push(pass());
                    }
                    out.extend(self.gen_company_choices(c, "track"));
                }
            }
            Step::DestinationToken => {
                if let Ent::Corp(c) = ent {
                    if self.g1822_destination_connected(c) {
                        if let Some(hex) = self.g1822_destination(c) {
                            out.push(Move::plain(Action::HexToken {
                                entity_id: c.clone(),
                                hex_id: hex.to_string(),
                                token_type: Some("destination".to_string()),
                            }));
                        }
                    } else {
                        out.push(pass());
                    }
                }
            }
            Step::Token => {
                if let Ent::Corp(c) = ent {
                    if has("place_token") {
                        out.extend(self.gen_tokens(c));
                    }
                    if has("pass") {
                        out.push(pass());
                    }
                    out.extend(self.gen_company_choices(c, "token"));
                }
            }
            Step::Route => {
                if let Ent::Corp(c) = ent {
                    if has("choose") {
                        out.extend(self.gen_pullman_choices(c));
                    }
                    if has("run_routes") {
                        out.push(Move::plain(Action::RunRoutes {
                            entity_id: c.clone(),
                            routes: self.g1822_auto_routes(c),
                            extra_revenue: 0,
                        }));
                    }
                }
            }
            Step::Dividend => {
                if let Ent::Corp(c) = ent {
                    for kind in [DividendKind::Payout, DividendKind::Half, DividendKind::Withhold] {
                        out.push(Move::plain(Action::Dividend {
                            entity_id: c.clone(),
                            kind,
                        }));
                    }
                    if has("choose") {
                        for choice in ["payout", "half", "withhold"] {
                            out.push(Move::plain(Action::Choose {
                                entity_id: c.clone(),
                                choice: choice.to_string(),
                            }));
                        }
                    }
                }
            }
            Step::BuyTrain => out.extend(self.gen_buy_train(ent, &types)),
            Step::MinorAcquisition => {
                if let Ent::Corp(c) = ent {
                    out.extend(self.gen_minor_acquisition(c, has("pass")));
                }
            }
            Step::IssueShares => {
                if let Ent::Corp(c) = ent {
                    out.extend(self.gen_issue_shares(c, &types));
                    out.extend(self.gen_company_choices(c, "issue"));
                }
            }
            // 1822MX (mx_legal): the NdeM's cash-out and token auction
            Step::CashOutNdem => out.push(pass()),
            Step::AuctionNdemTokens => out.extend(self.gen_mx_auction(ent)),
            Step::BuySellParShares | Step::Choose => {}
        }
        out
    }

    fn gen_discards(&self, c: &str) -> Vec<Move> {
        let co = &self.corporations[self.ci(c)];
        let perm = co.trains.iter().filter(|t| self.fam().extra_train_permanents.contains(&t.name.as_str())).count();
        let pull = co.trains.iter().filter(|t| t.name == self.fam().extra_train_pullman).count();
        let eligible: Vec<&crate::entities::Train> = if perm > 1 {
            co.trains.iter().filter(|t| self.fam().extra_train_permanents.contains(&t.name.as_str())).collect()
        } else if pull > 1 {
            co.trains.iter().filter(|t| t.name == self.fam().extra_train_pullman).collect()
        } else {
            co.trains.iter().filter(|t| !self.g1822_is_extra_train(&t.name)).collect()
        };
        let mut seen: Vec<String> = Vec::new();
        let mut out = Vec::new();
        for t in eligible {
            if seen.contains(&t.name) {
                continue;
            }
            seen.push(t.name.clone());
            out.push(Move::plain(Action::DiscardTrain {
                entity_id: c.to_string(),
                train_name: t.id.clone(),
            }));
        }
        out
    }

    fn gen_pending_tokens(&self) -> Vec<Move> {
        use super::graph::{Node, NodeKind};
        let Some(p) = self.rnd().pending_tokens.first().cloned() else { return Vec::new() };
        // G1822 PendingToken: `connected: !pending_home_tokeners.include?`
        // — only minor 14's own home token skips Tokener#check_connected;
        // a major's London token after acquiring bid-box minor 14 must go
        // to a city it connects to.
        let graph = (p.corp != self.fam().minor_14_id).then(|| self.g1822_graph(&p.corp));
        let mut out = Vec::new();
        for hex in &p.hexes {
            let Some(&hi) = self.hex_idx.get(hex) else { continue };
            for ci in 0..self.hexes[hi].tile.cities.len() {
                if self.g1822_city_tokened_by(hi, ci, &p.corp) {
                    continue;
                }
                if let Some(g) = &graph {
                    if !g.connected_nodes.contains(&Node { hex: hi, kind: NodeKind::City, index: ci }) {
                        continue;
                    }
                }
                out.push(Move::plain(Action::PlaceToken {
                    entity_id: p.corp.clone(),
                    hex_id: hex.clone(),
                    city_index: ci as u8,
                }));
            }
        }
        out
    }

    fn gen_company_choices(&self, c: &str, time: &str) -> Vec<Move> {
        let mut out = Vec::new();
        for co in self.companies_of(&EntityId::corporation(c)) {
            let f = self.fam();
            if ![f.company_chpr, f.company_lcdr, f.company_lur].iter().any(|x| Family::is(x, &co)) {
                continue;
            }
            for choice in self.g1822_company_choices(&co, time) {
                out.push(Move::plain(Action::ChooseAbility {
                    entity_id: co.clone(),
                    choice,
                }));
            }
        }
        out
    }

    /// Companies of the operator that act in parallel (SpecialTrack lays,
    /// P5's SpecialToken) while an earlier-or-equal step blocks.
    fn gen_parallel_companies(&mut self, bi: usize, ent: &Ent) -> Vec<Move> {
        let Ent::Corp(c) = ent else { return Vec::new() };
        let mut out = Vec::new();
        let steps = self.rnd().steps.clone();
        for co in self.companies_of(&EntityId::corporation(c)) {
            let ce = Ent::Company(co.clone());
            // Ruby step walk for the company's action types.
            let mut lay = false;
            let mut tok = false;
            let mut assign = false;
            let mut special_choose = false;
            for (i, s) in steps.iter().enumerate().take(bi + 1) {
                if !self.g1822_step_active(i) {
                    continue;
                }
                let acts = self.g1822_step_actions(i, &ce);
                match s {
                    Step::SpecialTrack if acts.contains(&"lay_tile") => lay = true,
                    Step::SpecialToken if acts.contains(&"place_token") => tok = true,
                    Step::Assign if acts.contains(&"assign") => assign = true,
                    Step::SpecialChoose if acts.contains(&"choose_ability") => special_choose = true,
                    _ => {}
                }
            }
            if assign {
                out.extend(self.gen_pnw_assigns(&co));
            }
            if special_choose {
                out.extend(self.gen_mx_special_choose(&co));
            }
            if lay {
                out.extend(self.gen_company_lays(&co, false));
            }
            if tok {
                out.extend(self.gen_special_tokens(&co));
            }
        }
        out
    }

    // =====================================================================
    // Tile lays
    // =====================================================================

    fn tile_instance_for(&self, name: &str) -> String {
        let on_map: std::collections::HashSet<&str> = self.hexes.iter().map(|h| h.tile.id.as_str()).collect();
        for n in 0..10_000u32 {
            let id = format!("{name}-{n}");
            if !on_map.contains(id.as_str()) {
                return id;
            }
        }
        format!("{name}-0")
    }

    /// Track step lays for the operating corporation.
    fn gen_corp_lays(&mut self, c: &str) -> Vec<Move> {
        let t0 = std::time::Instant::now();
        let out = self.gen_corp_lays_inner(c);
        stats::add(&stats::LAYS_NS, t0.elapsed().as_nanos() as u64);
        out
    }

    fn gen_corp_lays_inner(&mut self, c: &str) -> Vec<Move> {
        if self.g1822_is_pnw() {
            return self.gen_pnw_corp_lays(c);
        }
        if self.g1822_is_mx() {
            return self.gen_mx_corp_lays(c);
        }
        let graph = self.g1822_graph(c);
        let nbrs = self.g1822_neighbors();
        let Some((lay, upgrade, no_reuse)) = self.g1822_tile_lay(c, false) else { return Vec::new() };
        let mut out = Vec::new();
        for (&hi, edges) in &graph.connected_hexes {
            let color = self.hexes[hi].tile.color;
            let white = color == crate::tiles::TileColor::White;
            if white && !lay {
                continue;
            }
            if !white && !upgrade {
                continue;
            }
            if !white && no_reuse && self.rnd().laid_hexes.contains(&self.hexes[hi].id) {
                continue;
            }
            let tiles = self.g1822_potential_tiles(c, hi);
            out.extend(self.lays_for(c, c, hi, tiles, edges, &nbrs, RotRule::Normal, &[], true, &graph.connected_nodes));
        }
        out
    }

    /// (tile, rotation) candidates on one hex, with the UI's max-exits
    /// filter and (for a corporation's own lay) check_track_restrictions!.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn lays_for(
        &mut self,
        entity: &str,
        operator: &str,
        hi: usize,
        tiles: Vec<String>,
        connected: &[u8],
        nbrs: &HashMap<usize, BTreeMap<u8, usize>>,
        rule: RotRule,
        combo: &[String],
        restrict: bool,
        connected_nodes: &std::collections::HashSet<super::graph::Node>,
    ) -> Vec<Move> {
        let mut with_rot: Vec<(String, Vec<u8>)> = Vec::new();
        for t in tiles {
            let def = self.tile_catalog[&t].clone();
            let rots: Vec<u8> = (0..6u8)
                .filter(|&r| self.g1822_legal_rotation(rule, operator, hi, &def, r, connected, nbrs))
                .collect();
            if !rots.is_empty() {
                with_rot.push((t, rots));
            }
        }
        let keep = self.g1822_max_exits(hi, with_rot.iter().map(|(t, _)| t.clone()).collect());
        let hex_id = self.hexes[hi].id.clone();
        // A corporation's own lay (Track step): every process-side check is
        // the generator's (slot rules, supply, rotation) except the cost,
        // checked here, so the lay skips the apply-on-a-clone filter.
        // (1822PNW: every lay goes through the apply filter)
        let own = restrict && entity == operator && combo.is_empty() && !self.g1822_is_pnw();
        let cash = self.corp_idx.get(operator).map_or(0, |&i| self.corporations[i].cash);
        let mut out = Vec::new();
        for (t, rots) in with_rot {
            if !keep.contains(&t) {
                continue;
            }
            let inst = self.tile_instance_for(&t);
            let def = self.tile_catalog[&t].clone();
            for r in rots {
                let mut m = Move::plain(Action::LayTile {
                    entity_id: entity.to_string(),
                    hex_id: hex_id.clone(),
                    tile_id: inst.clone(),
                    rotation: r,
                });
                m.combo = combo.to_vec();
                if own {
                    let exits = def.rotated(r).edges.clone();
                    if self.g1822_corp_lay_cost(operator, hi, &exits, nbrs) > cash {
                        continue;
                    }
                    m.validated = true;
                }
                if restrict && !self.lay_passes_restrictions(&m, operator, hi, &def, r, connected_nodes, connected) {
                    continue;
                }
                out.push(m);
            }
        }
        out
    }

    /// check_track_restrictions! on the laid tile (a clone).
    #[allow(clippy::too_many_arguments)]
    fn lay_passes_restrictions(
        &self,
        m: &Move,
        operator: &str,
        hi: usize,
        def: &crate::tiles::TileDef,
        rotation: u8,
        connected_nodes: &std::collections::HashSet<super::graph::Node>,
        connected_edges: &[u8],
    ) -> bool {
        let hex = &self.hexes[hi];
        if hex.tile.paths.is_empty()
            || Family::is(self.fam().english_channel_hex, &hex.id)
            || Family::is(self.fam().cardiff_hex, &hex.id)
        {
            return true; // laying on track-less ground always uses new track
        }
        // Fast path: the operator already reaches a city of this hex whose
        // revenue the new tile changes. Old paths are kept and reachability
        // only grows, so after the lay the walk reaches that city and walks
        // its paths: a connected path with a changed city value exists.
        let new_tile = BaseGame::tile_from_def(def, rotation);
        let map = BaseGame::g1822_city_map(&hex.tile, &new_tile);
        for (oi, ni) in map.iter().enumerate() {
            let Some(ni) = ni else { continue };
            let node = super::graph::Node { hex: hi, kind: super::graph::NodeKind::City, index: oi };
            if connected_nodes.contains(&node)
                && hex.tile.cities[oi].revenue != new_tile.cities[*ni].revenue
                && !new_tile.cities[*ni].revenue.eq(&0)
            {
                // The 'New track must override old one' check still runs.
                let new_tv = super::tile_rules::TV::from_def(def, rotation);
                let old_tv = super::tile_rules::TV::from_tile(&hex.tile);
                if super::tile_rules::cities_towns_kept(&old_tv, &new_tv) {
                    return true;
                }
            }
        }
        // Fast path 2: new track through an edge the operator's walk enters
        // the hex by (not a token hex, whose connected edges are all its
        // neighbours rather than walked ones): that path is walked after
        // the lay and is new track.
        let token_hex = self.g1822_token_nodes(operator).iter().any(|n| n.hex == hi);
        if !token_hex {
            let new_tv = super::tile_rules::TV::from_def(def, rotation);
            let old_tv = super::tile_rules::TV::from_tile(&hex.tile);
            if super::tile_rules::new_track_at(&old_tv, &new_tv, connected_edges)
                && super::tile_rules::cities_towns_kept(&old_tv, &new_tv)
            {
                return true;
            }
        }
        let t0 = std::time::Instant::now();
        stats::add(&stats::RESTRICT_N, 1);
        let r = self.lay_passes_restrictions_inner(m, operator, hi);
        stats::add(&stats::RESTRICT_NS, t0.elapsed().as_nanos() as u64);
        r
    }

    fn lay_passes_restrictions_inner(&self, m: &Move, operator: &str, hi: usize) -> bool {
        let old = self.hexes[hi].tile.clone();
        let mut g = self.clone_for_search();
        if g.g1822_apply_move(m).is_err() {
            return false;
        }
        g.g1822_track_restrictions_ok(operator, hi, &old)
    }

    /// SpecialTrack lays by a company (alone and in Ruby-valid combos).
    fn gen_company_lays(&mut self, co: &str, in_step: bool) -> Vec<Move> {
        let _ = in_step;
        if self.g1822_is_pnw() {
            if let Some(v) = self.gen_pnw_company_lays(co) {
                return v;
            }
            let mut v = self.gen_company_lays_1822(co);
            v.extend(self.pnw_company_extras(co));
            return v;
        }
        if self.g1822_is_mx() {
            if let Some(v) = self.gen_mx_company_lays(co) {
                return v;
            }
            let mut v = self.gen_company_lays_1822(co);
            v.extend(self.mx_company_extras(co));
            return v;
        }
        self.gen_company_lays_1822(co)
    }

    fn gen_company_lays_1822(&mut self, co: &str) -> Vec<Move> {
        let Some((_, ab)) = self.g1822_track_ability(co) else { return Vec::new() };
        let Some(owner) = self.co_owner(co).corp_sym().map(str::to_string) else { return Vec::new() };
        // Combos: companies this one combines with (owned by the same
        // corporation, with a live tile_lay ability).
        let partners: Vec<String> = ab
            .combo
            .iter()
            .filter(|p| self.co_owner(p) == EntityId::corporation(&owner) && self.g1822_track_ability(p).is_some())
            .cloned()
            .collect();
        let mut combos: Vec<Vec<String>> = vec![Vec::new()];
        for p in &partners {
            combos.push(vec![p.clone()]);
        }
        let mut out = Vec::new();
        for combo in combos {
            if !self.valid_combo(co, &combo) {
                continue;
            }
            out.extend(self.gen_company_lays_combo(co, &owner, &combo));
        }
        out
    }

    fn valid_combo(&self, co: &str, combo: &[String]) -> bool {
        let mut all = vec![co.to_string()];
        all.extend(combo.iter().cloned());
        for (i, a) in all.iter().enumerate() {
            let Some((_, ab)) = self.g1822_track_ability(a) else { return false };
            if all[i + 1..].iter().any(|b| !ab.combo.contains(b)) {
                return false;
            }
        }
        true
    }

    fn gen_company_lays_combo(&mut self, co: &str, owner: &str, combo: &[String]) -> Vec<Move> {
        let Some((_, ab)) = self.g1822_track_ability(co) else { return Vec::new() };
        let nbrs = self.g1822_neighbors();
        let mut entities = vec![co.to_string()];
        entities.extend(combo.iter().cloned());
        let f = self.fam();
        let rule = if Family::is(f.company_lcdr, co) {
            RotRule::LcdrChannel
        } else if entities.iter().any(|e| self.g1822_must_remove_town(e)) {
            RotRule::RemoveTown
        } else {
            RotRule::Normal
        };
        let graph = self.g1822_graph(owner);
        let extra = entities.iter().any(|e| self.g1822_company_ability_extra_track(e));
        let tile_lay = self.g1822_tile_lay(owner, Family::is(f.company_hsbc, co));
        let mut out = Vec::new();
        let hexes: Vec<usize> = if ab.kind == AbKind::Teleport {
            ab.hexes.iter().filter_map(|h| self.hex_idx.get(h).copied()).collect()
        } else if !ab.hexes.is_empty() {
            ab.hexes.iter().filter_map(|h| self.hex_idx.get(h).copied()).collect()
        } else {
            graph.connected_hexes.keys().copied().collect()
        };
        for hi in hexes {
            // 1822PNW SpecialTrack#available_hex: never a mountain-icon hex
            if self.g1822_is_pnw() && self.pnw_mountain_icon(hi) {
                continue;
            }
            let hex = &self.hexes[hi];
            let connected: Vec<u8> = if ab.kind == AbKind::Teleport {
                nbrs.get(&hi).map(|m| m.keys().copied().collect()).unwrap_or_default()
            } else {
                match graph.connected_hexes.get(&hi) {
                    Some(e) => e.clone(),
                    None => continue,
                }
            };
            if ab.kind != AbKind::Teleport && !Family::is(f.company_hsbc, co) {
                // available_hex: the operator's lay slot, unless P12 adds one.
                let white = hex.tile.color == crate::tiles::TileColor::White;
                match tile_lay {
                    Some((lay, up, no_reuse)) if lay || up => {
                        if white && !lay {
                            continue;
                        }
                        if !white && !up {
                            continue;
                        }
                        if !white && no_reuse && self.rnd().laid_hexes.contains(&hex.id) {
                            continue;
                        }
                    }
                    _ => {
                        if !extra {
                            continue;
                        }
                    }
                }
                if entities.iter().any(|e| self.g1822_must_remove_town(e)) && hex.tile.towns.len() != 1 {
                    continue;
                }
                if entities.iter().any(|e| self.g1822_can_only_lay_plain_or_towns(e))
                    && !self.fam().track_plain.contains(&base(&hex.tile.name))
                    && !self.fam().track_town.contains(&base(&hex.tile.name))
                {
                    continue;
                }
                if entities.iter().any(|e| self.g1822_can_upgrade_one_phase_ahead(e)) {
                    let last = self.phase.tiles.last().cloned().unwrap_or_default();
                    if super::tile_rules::color_name(hex.tile.color) != last {
                        continue;
                    }
                }
                if entities.iter().any(|e| self.g1822_must_be_on_terrain(e)) {
                    let terrain = hex.tile.upgrades.iter().any(|u| u.terrain == "mountain" || u.terrain == "hill");
                    if !terrain {
                        continue;
                    }
                }
                if entities.iter().any(|e| self.g1822_must_be_on_estuary(e))
                    && !self.g().borders.get(&hex.id).map_or(false, |v| v.iter().any(|b| b.2 == "water"))
                {
                    continue;
                }
            }
            let mut tiles = self.special_potential_tiles(co, combo, owner, hi);
            if self.g1822_is_pnw() {
                tiles = self.pnw_potential_tiles(hi, tiles);
            }
            out.extend(self.lays_for(co, owner, hi, tiles, &connected, &nbrs, rule, combo, false, &graph.connected_nodes));
        }
        out
    }

    /// SpecialTrack#potential_tiles over the company + combo abilities.
    fn special_potential_tiles(&self, co: &str, combo: &[String], owner: &str, hi: usize) -> Vec<String> {
        let mut abilities: Vec<(String, super::track::Ab)> = Vec::new();
        if let Some((_, a)) = self.g1822_track_ability(co) {
            abilities.push((co.to_string(), a));
        }
        for c in combo {
            if let Some((_, a)) = self.g1822_track_ability(c) {
                abilities.push((c.clone(), a));
            }
        }
        let mut use_default = true;
        let mut tiles: Vec<String> = Vec::new();
        for (company, ab) in abilities {
            if use_default && !ab.tiles.is_empty() {
                use_default = false;
                tiles.clear();
            }
            if use_default {
                tiles.extend(self.g1822_potential_tiles(owner, hi));
            } else {
                tiles.extend(self.potential_tiles_for_entity(&company, owner, hi, &ab));
            }
        }
        tiles.sort();
        tiles.dedup();
        tiles
    }

    fn potential_tiles_for_entity(&self, company: &str, owner: &str, hi: usize, ab: &super::track::Ab) -> Vec<String> {
        use crate::tiles::TileColor::*;
        let hex_color = self.hexes[hi].tile.color;
        let minor = self.is_minor(owner);
        if minor && matches!(hex_color, Green | Brown | Gray) {
            return Vec::new();
        }
        let advanced = self.g1822_can_upgrade_one_phase_ahead(company);
        if advanced && minor && hex_color != Yellow {
            return Vec::new();
        }
        let special = ab.kind == AbKind::TileLay && ab.special;
        let mut out = Vec::new();
        for name in &ab.tiles {
            if self.tile_counts_remaining.get(name).copied().unwrap_or(0) == 0 {
                continue;
            }
            let Some(def) = self.tile_catalog.get(name) else { continue };
            if !self.g1822_upgrades_to(hi, def, special) {
                continue;
            }
            let in_phase = self.phase.tiles.iter().any(|c| c == super::tile_rules::color_name(def.color));
            if advanced == in_phase {
                continue;
            }
            out.push(name.clone());
        }
        out
    }

    // =====================================================================
    // Tokens
    // =====================================================================

    fn gen_tokens(&self, c: &str) -> Vec<Move> {
        let graph = self.g1822_graph(c);
        let mut out = Vec::new();
        let mut nodes: Vec<&super::graph::Node> = graph
            .connected_nodes
            .iter()
            .filter(|n| n.kind == super::graph::NodeKind::City)
            .collect();
        nodes.sort();
        for n in nodes {
            if self.g1822_city_tokened_by(n.hex, n.index, c) {
                continue;
            }
            out.push(Move::plain(Action::PlaceToken {
                entity_id: c.to_string(),
                hex_id: self.hexes[n.hex].id.clone(),
                city_index: n.index as u8,
            }));
        }
        out
    }

    fn gen_special_tokens(&self, co: &str) -> Vec<Move> {
        let f = self.fam();
        if !Family::is(f.company_lcdr, co) {
            return Vec::new();
        }
        let Some(&hi) = self.hex_idx.get(f.english_channel_hex) else { return Vec::new() };
        (0..self.hexes[hi].tile.cities.len())
            .map(|ci| {
                Move::plain(Action::PlaceToken {
                    entity_id: co.to_string(),
                    hex_id: f.english_channel_hex.to_string(),
                    city_index: ci as u8,
                })
            })
            .collect()
    }

    // =====================================================================
    // Route / Dividend
    // =====================================================================

    fn gen_pullman_choices(&self, c: &str) -> Vec<Move> {
        let n = self.g1822_pullman_choices(c).len();
        (0..n)
            .map(|i| {
                Move::plain(Action::Choose {
                    entity_id: c.to_string(),
                    choice: i.to_string(),
                })
            })
            .collect()
    }

    /// The routes the generator runs: the auto-router's best run.
    pub(crate) fn g1822_auto_routes(&self, c: &str) -> Vec<RouteData> {
        self.g1822_best_routes(c).0
    }

    // =====================================================================
    // BuyTrain
    // =====================================================================

    fn gen_buy_train(&mut self, ent: &Ent, types: &[&'static str]) -> Vec<Move> {
        let mut out = Vec::new();
        let has = |t: &str| types.iter().any(|x| *x == t);
        match ent {
            Ent::Player(pid) => {
                if has("sell_shares") {
                    out.extend(self.gen_player_sales(*pid, true));
                }
            }
            Ent::Corp(c) => {
                if has("buy_train") {
                    // depot (head, phase-available, discards) × variants
                    for t in self.g1822_depot_trains() {
                        let variants = self.g1822_card_variants(&t);
                        for (v, price) in variants {
                            out.push(Move::plain(Action::BuyTrain {
                                entity_id: c.clone(),
                                train_name: t.id.clone(),
                                price,
                                from: "depot".into(),
                                variant: Some(v),
                                exchange: None,
                            }));
                        }
                    }
                    // other corporations' trains: £1 .. cash
                    if self.phase_has_status("can_buy_trains") {
                        let cash = self.corporations[self.ci(c)].cash;
                        if cash >= 1 {
                            let mut seen: Vec<(String, String)> = Vec::new();
                            for (seller, t) in self.g1822_other_trains(c) {
                                if seen.contains(&(seller.clone(), t.name.clone())) {
                                    continue;
                                }
                                seen.push((seller.clone(), t.name.clone()));
                                let mut m = Move::plain(Action::BuyTrain {
                                    entity_id: c.clone(),
                                    train_name: t.id.clone(),
                                    price: 1,
                                    from: seller.clone(),
                                    variant: Some(t.name.clone()),
                                    exchange: None,
                                });
                                m.price = Some(PriceRange { lo: 1, hi: cash, step: 1 });
                                out.push(m);
                            }
                        }
                    }
                    // L → 2 upgrades (£80)
                    let ls: Vec<String> = self.corporations[self.ci(c)]
                        .trains
                        .iter()
                        .filter(|t| t.name == "L")
                        .map(|t| t.id.clone())
                        .collect();
                    if let Some(l) = ls.first() {
                        out.push(Move::plain(Action::BuyTrain {
                            entity_id: c.clone(),
                            train_name: l.clone(),
                            price: self.fam().upgrade_cost_l_to_2,
                            from: "depot".into(),
                            variant: Some("2".into()),
                            exchange: Some(l.clone()),
                        }));
                    }
                }
                if has("sell_shares") {
                    // must buy: the president's emergency sales.
                    if let Some(p) = self.g1822_president(c).and_then(|o| o.player_id()) {
                        out.extend(self.gen_player_sales(p, true));
                    }
                }
                if has("pass") {
                    out.push(Move::plain(Action::Pass { entity_id: c.clone() }));
                }
            }
            Ent::Company(_) => {}
        }
        out
    }

    // =====================================================================
    // MinorAcquisition / IssueShares
    // =====================================================================

    fn gen_minor_acquisition(&self, c: &str, can_pass: bool) -> Vec<Move> {
        let mut out = Vec::new();
        match self.rnd().acquire_state {
            AcquireState::ChoosePay | AcquireState::ChooseToken => {
                let choices = if self.rnd().acquire_state == AcquireState::ChoosePay {
                    self.rnd().pay_choices.clone()
                } else {
                    self.rnd().token_choices.clone()
                };
                let moves: Vec<Move> = choices
                    .into_iter()
                    .map(|ch| Move::plain(Action::Choose { entity_id: c.to_string(), choice: ch }))
                    .collect();
                // Dead-end lookahead: Ruby accepts every choice, but e.g.
                // replacing minor 14's token leaves a pending London token
                // costing £20 that a broke major cannot pay (no pass) —
                // drop choices whose next state has no legal move, unless
                // all do.
                let live: Vec<Move> = moves
                    .iter()
                    .filter(|m| {
                        let mut g = self.clone_for_search();
                        g.g1822_apply_move_nobackup(m).is_ok() && (g.finished || !g.g1822_legal_moves().is_empty())
                    })
                    .cloned()
                    .collect();
                out.extend(if live.is_empty() { moves } else { live });
            }
            AcquireState::SelectMinor => {
                for m in self.g1822_mergeable(c) {
                    // 1822PNW's regionals join potentially_mergeable without
                    // the pay-choice filter: one the major cannot pay for
                    // leaves a choose with no choices (a Ruby dead end)
                    if self.g1822_is_pnw() && self.g1822_pay_choices(c, &m).is_empty() {
                        continue;
                    }
                    out.push(Move::plain(Action::Merge {
                        entity_id: c.to_string(),
                        corporation_sym: m,
                    }));
                }
                if can_pass {
                    out.push(Move::plain(Action::Pass { entity_id: c.to_string() }));
                }
            }
        }
        out
    }

    fn gen_issue_shares(&self, c: &str, types: &[&'static str]) -> Vec<Move> {
        let mut out = Vec::new();
        let has = |t: &str| types.iter().any(|x| *x == t);
        if has("sell_shares") {
            for b in self.g1822_issuable_bundles(c) {
                let mut m = Move::plain(Action::SellShares {
                    entity_id: c.to_string(),
                    corporation_sym: c.to_string(),
                    shares: Vec::new(),
                    percent: b.percent,
                    share_indices: b.shares.clone(),
                });
                m.shares = self.bundle_move_ids(&b);
                m.share_price = Some(self.g1822_price_after_drops(c, b.shares.len()));
                out.push(m);
            }
        }
        if has("buy_shares") {
            for b in self.g1822_redeemable_bundles(c) {
                let mut m = Move::plain(Action::BuyShares {
                    entity_id: c.to_string(),
                    corporation_sym: c.to_string(),
                    shares: Vec::new(),
                    percent: b.percent,
                    source: "market".into(),
                    share_indices: b.shares.clone(),
                });
                m.shares = self.bundle_move_ids(&b);
                out.push(m);
            }
        }
        if has("pass") {
            out.push(Move::plain(Action::Pass { entity_id: c.to_string() }));
        }
        out
    }

    /// `reduced_bundle_price_for_market_drop`: the price `n` cells down.
    pub(crate) fn g1822_price_after_drops(&self, c: &str, n: usize) -> i32 {
        let ci = self.ci(c);
        let Some(sp) = self.corporations[ci].share_price.clone() else { return 0 };
        let (mut r, mut col) = (sp.row, sp.column);
        let pnw = self.g1822_is_pnw() || self.g1822_is_mx();
        for _ in 0..n {
            // PNW / MX reduced_bundle_price_for_market_drop: [:left] * n
            let (nr, nc) = if pnw { self.stock_market.move_left(r, col) } else { self.stock_market.move_down(r, col) };
            r = nr;
            col = nc;
        }
        self.stock_market.share_price_at(r, col).map_or(sp.price, |p| p.price)
    }

    // =====================================================================
    // Wire form (18xx.games action dicts) — for Python and Ruby replay.
    // =====================================================================

    /// The 18xx.games action dict for a move (state BEFORE applying it).
    pub fn g1822_move_wire(&self, m: &Move) -> Value {
        let a = &m.action;
        let mut o = Map::new();
        o.insert("type".into(), json!(a.action_type()));
        let eid = a.entity_id().to_string();
        let (ent_val, ent_type) = match self.g1822_entity(&eid) {
            Ent::Player(p) => (json!(p), "player"),
            Ent::Corp(c) => (json!(c), "corporation"),
            Ent::Company(c) => (json!(c), "company"),
        };
        o.insert("entity".into(), ent_val);
        o.insert("entity_type".into(), json!(ent_type));
        match a {
            Action::Bid { company_sym, price, .. } => {
                // (1822MX's NdeM token bids name no company)
                if !company_sym.is_empty() {
                    o.insert("company".into(), json!(company_sym));
                }
                o.insert("price".into(), json!(price));
            }
            Action::RemoveToken { hex_id, city_index, slot, .. } => {
                if let Some(&hi) = self.hex_idx.get(hex_id.as_str()) {
                    let tid = self.g1822_wire_tile_id(hi);
                    o.insert("city".into(), json!(format!("{tid}-{city_index}")));
                }
                o.insert("slot".into(), json!(slot));
            }
            Action::Par { corporation_sym, share_price, .. } => {
                o.insert("corporation".into(), json!(corporation_sym));
                // Ruby SharePrice#id: "price,row,column".
                let cell = self
                    .stock_market
                    .share_prices_with_types(&["par_1", "par"])
                    .into_iter()
                    .find(|sp| sp.price == *share_price);
                match cell {
                    Some(sp) => o.insert("share_price".into(), json!(format!("{},{},{}", sp.price, sp.row, sp.column))),
                    None => o.insert("share_price".into(), json!(share_price)),
                };
            }
            Action::BuyShares { percent, .. } | Action::SellShares { percent, .. } => {
                o.insert("shares".into(), json!(m.shares));
                o.insert("percent".into(), json!(percent));
                if let Some(p) = m.share_price {
                    o.insert("share_price".into(), json!(p));
                }
            }
            Action::LayTile { hex_id, tile_id, rotation, .. } => {
                o.insert("hex".into(), json!(hex_id));
                o.insert("tile".into(), json!(tile_id));
                o.insert("rotation".into(), json!(rotation));
                if !m.combo.is_empty() {
                    o.insert("combo_entities".into(), json!(m.combo));
                }
            }
            Action::PlaceToken { hex_id, city_index, .. } => {
                if let Some(&hi) = self.hex_idx.get(hex_id.as_str()) {
                    let tid = self.g1822_wire_tile_id(hi);
                    o.insert("city".into(), json!(format!("{tid}-{city_index}")));
                    let slot = self.g1822_token_slot_preview(&eid, hi, *city_index as usize);
                    o.insert("slot".into(), json!(slot));
                    if let Ent::Company(co) = self.g1822_entity(&eid) {
                        if let Some(c) = self.co_owner(&co).corp_sym() {
                            o.insert("tokener".into(), json!(c));
                        }
                    }
                }
            }
            Action::RunRoutes { routes, extra_revenue, .. } => {
                let rs: Vec<Value> = routes
                    .iter()
                    .map(|r| {
                        json!({
                            "train": r.train_name,
                            "hexes": r.hexes,
                            "connections": r.connections,
                            "nodes": r.nodes,
                            "revenue": r.revenue,
                        })
                    })
                    .collect();
                o.insert("routes".into(), json!(rs));
                o.insert("extra_revenue".into(), json!(extra_revenue));
                o.insert("subsidy".into(), json!(0));
            }
            Action::Dividend { kind, .. } => {
                let k = match kind {
                    DividendKind::Payout => "payout",
                    DividendKind::Half => "half",
                    DividendKind::Withhold => "withhold",
                };
                o.insert("kind".into(), json!(k));
            }
            Action::BuyTrain { train_name, price, variant, exchange, .. } => {
                o.insert("train".into(), json!(train_name));
                o.insert("price".into(), json!(price));
                if let Some(v) = variant {
                    o.insert("variant".into(), json!(v));
                }
                if let Some(x) = exchange {
                    o.insert("exchange".into(), json!(x));
                }
            }
            Action::DiscardTrain { train_name, .. } => {
                o.insert("train".into(), json!(train_name));
            }
            Action::Merge { corporation_sym, .. } => {
                o.insert("corporation".into(), json!(corporation_sym));
            }
            Action::Choose { choice, .. } | Action::ChooseAbility { choice, .. } => {
                o.insert("choice".into(), json!(choice));
            }
            Action::AcquireCompany { company_sym, .. } => {
                o.insert("company".into(), json!(company_sym));
            }
            Action::HexToken { hex_id, token_type, .. } => {
                o.insert("hex".into(), json!(hex_id));
                if let Some(t) = token_type {
                    o.insert("token_type".into(), json!(t));
                }
            }
            Action::PayoffPlayerDebtPartial { amount, .. } => {
                o.insert("amount".into(), json!(amount));
            }
            Action::Assign { target, target_type, .. } => {
                o.insert("target".into(), json!(target));
                o.insert("target_type".into(), json!(target_type));
            }
            _ => {}
        }
        Value::Object(o)
    }

    /// The Ruby tile id of a hex's tile ("<HEX>-0" for a preprinted one).
    pub(crate) fn g1822_wire_tile_id(&self, hi: usize) -> String {
        let t = &self.hexes[hi].tile;
        if t.id.starts_with("preprinted_") || !t.id.contains('-') {
            format!("{}-0", self.hexes[hi].id)
        } else {
            t.id.clone()
        }
    }
}

fn base(name: &str) -> &str {
    name.split('-').next().unwrap_or(name)
}

fn same_move(a: &Move, b: &Move) -> bool {
    format!("{:?}{:?}{:?}", a.action, a.combo, a.share_price) == format!("{:?}{:?}{:?}", b.action, b.combo, b.share_price)
}

impl BaseGame {
    /// DestinationToken#destination_node_check?: a walk from the
    /// destination hex's first node reaches the home token's city.
    pub(crate) fn g1822_destination_connected(&self, c: &str) -> bool {
        use super::graph::{Node, NodeKind};
        let Some(dest) = self.g1822_destination(c) else { return false };
        let Some(&dhi) = self.hex_idx.get(dest.as_str()) else { return false };
        let tile = &self.hexes[dhi].tile;
        let start = if !tile.cities.is_empty() {
            Node { hex: dhi, kind: NodeKind::City, index: 0 }
        } else if !tile.towns.is_empty() {
            Node { hex: dhi, kind: NodeKind::Town, index: 0 }
        } else if !tile.offboards.is_empty() {
            Node { hex: dhi, kind: NodeKind::Offboard, index: 0 }
        } else {
            return false;
        };
        // The home token: the corporation's first token's city.
        let Some(home) = self.g1822_home_node(c) else { return false };
        self.g1822_walk_from(c, start)
            .into_iter()
            .any(|p| self.g1822_path_nodes(p).contains(&home))
    }

    /// `entity.tokens.first.city` as a node.
    pub(crate) fn g1822_home_node(&self, c: &str) -> Option<super::graph::Node> {
        use super::graph::{Node, NodeKind};
        let ci = *self.corp_idx.get(c)?;
        let first = self.corporations[ci].tokens.first()?;
        if !first.used {
            return None;
        }
        if let Some((h, k)) = self.g().home_city.get(c) {
            let &hi = self.hex_idx.get(h.as_str())?;
            let ok = self.hexes[hi]
                .tile
                .cities
                .get(*k)
                .map_or(false, |city| city.tokens.iter().flatten().any(|t| t.corporation_id == c));
            if ok {
                return Some(Node { hex: hi, kind: NodeKind::City, index: *k });
            }
        }
        let &hi = self.hex_idx.get(first.city_hex_id.as_str())?;
        // The city of that hex holding the corporation's first-type token.
        let cities = &self.hexes[hi].tile.cities;
        let idx = (0..cities.len()).find(|&k| {
            cities[k]
                .tokens
                .iter()
                .flatten()
                .any(|t| t.corporation_id == c && t.token_type == first.token_type)
        })?;
        Some(Node { hex: hi, kind: NodeKind::City, index: idx })
    }

    /// MinorAcquisition#mergeable: potentially_mergeable minors the major
    /// connects to (entity_connects?).
    pub(crate) fn g1822_mergeable(&self, c: &str) -> Vec<String> {
        use super::graph::{Node, NodeKind};
        if self.g1822_is_pnw() {
            // PNW MinorAcquisition: Connections#entity_connects?
            return self
                .g1822_potentially_mergeable(c)
                .into_iter()
                .filter(|m| self.g1822_pnw_entity_connects(c, m))
                .collect();
        }
        let graph = self.g1822_graph(c);
        // 1822MX entity_connects?: or both reach a port
        let mx_port = self.g1822_is_mx() && self.g1822_mx_connected_to_port(c);
        let mut out = Vec::new();
        for m in self.g1822_potentially_mergeable(c) {
            if mx_port && self.g1822_mx_connected_to_port(&m) {
                out.push(m);
                continue;
            }
            let floated = self.corporations[self.ci(&m)].floated;
            let connected = if !floated && m == self.fam().minor_14_id {
                let Some(&hi) = self.hex_idx.get(self.fam().minor_14_home_hex) else { continue };
                (0..self.hexes[hi].tile.cities.len())
                    .any(|k| graph.connected_nodes.contains(&Node { hex: hi, kind: NodeKind::City, index: k }))
            } else if !floated {
                let home = self
                    .title_def()
                    .corporations()
                    .into_iter()
                    .find(|d| d.sym == m)
                    .map(|d| d.home_hex);
                let Some(home) = home else { continue };
                let Some(&hi) = self.hex_idx.get(home) else { continue };
                let city = (0..self.hexes[hi].tile.cities.len())
                    .find(|&k| self.g1822_city_res(home, k).iter().any(|r| r.as_deref() == Some(m.as_str())));
                city.map_or(false, |k| graph.connected_nodes.contains(&Node { hex: hi, kind: NodeKind::City, index: k }))
            } else {
                self.g1822_token_nodes(&m).first().map_or(false, |n| graph.connected_nodes.contains(n))
            };
            if connected {
                out.push(m);
            }
        }
        out
    }

    /// The slot a token placed by `entity` in city (hi, ci) would take.
    pub(crate) fn g1822_token_slot_preview(&self, entity: &str, hi: usize, ci: usize) -> usize {
        let corp = match self.g1822_entity(entity) {
            Ent::Company(co) => self.co_owner(&co).corp_sym().unwrap_or(entity).to_string(),
            _ => entity.to_string(),
        };
        self.g1822_get_slot(hi, ci, &corp, false)
            .unwrap_or(self.hexes[hi].tile.cities[ci].tokens.len())
    }
}

