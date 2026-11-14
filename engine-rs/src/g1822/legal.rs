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

/// An inclusive price range with a step (`lo + k*step <= hi`): the same
/// type as 1867's generator, so the RL index API (`rl::generator`) prices
/// both titles' moves alike.
pub use crate::g1867::legal::PriceRange;

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
        // 1822CA (ca_legal): the sawmill, the acquisition bonus lay, ICR's
        // city choice, P10's Winnipeg token
        (Step::AssignSawmill, "assign" | "choose") => Generated,
        (Step::AcquisitionTrack, "lay_tile" | "pass") => Generated,
        (Step::DestinationToken, "place_token") => Generated,
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

/// The reference generator: the code paths as they were before the
/// 2026-11-14 speed-ups, kept callable so a differential test can compare
/// the two move lists state by state (`g1822_legal_moves_reference`;
/// Python `BaseGame.g1822_set_reference_generator(on)`). Thread-local, so
/// parallel cargo tests don't see each other's flag.
pub mod reference {
    use std::cell::Cell;
    thread_local! {
        static ON: Cell<bool> = const { Cell::new(false) };
    }
    /// Whether the reference paths are in force on this thread.
    #[inline]
    pub fn on() -> bool {
        ON.with(|c| c.get())
    }
    /// Set the flag on this thread (until set again); returns the previous.
    pub fn set(enabled: bool) -> bool {
        ON.with(|c| c.replace(enabled))
    }
    /// Run `f` with the reference paths on (or off); the flag is restored
    /// afterwards, also on a panic.
    pub fn with<T>(enabled: bool, f: impl FnOnce() -> T) -> T {
        struct Restore(bool);
        impl Drop for Restore {
            fn drop(&mut self) {
                ON.with(|c| c.set(self.0));
            }
        }
        let _restore = Restore(ON.with(|c| c.replace(enabled)));
        f()
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
        let t = super::prof::start();
        // The fast paths share their (pure) graph computations -- this
        // state's own and its validation clones' (`graph::graph_memo`).
        let _graphs = (!reference::on()).then(|| super::graph::graph_memo::install(self));
        let out = self.g1822_legal_moves_inner(policy);
        super::prof::stop(t, || {
            let step = self.g1822_active_step().map_or("-".to_string(), |bi| format!("{:?}", self.rnd().steps[bi]));
            format!("total:{step}")
        });
        let gm = self.gm();
        gm.route_purchase = memo.0;
        gm.ruby_graphs = memo.1;
        out
    }

    /// [`BaseGame::g1822_legal_moves_with`] through the reference paths
    /// (the generator before the 2026-11-14 speed-ups): the differential
    /// test's oracle. Same moves -- or a bug in the fast paths.
    pub fn g1822_legal_moves_reference(&mut self, policy: CrossPlayerPolicy) -> Vec<Move> {
        reference::with(true, || self.g1822_legal_moves_with(policy))
    }

    /// One move as a comparable string: the action, its combo, share
    /// price, price range and certificate ids (everything a move carries
    /// but the internal `validated` flag).
    pub fn g1822_move_string(m: &Move) -> String {
        format!("{:?}|{:?}|{:?}|{:?}|{:?}", m.action, m.combo, m.share_price, m.price, m.shares)
    }

    /// The state the generator must leave as it found it (the family's
    /// state incl. the route latch and Ruby's graph cache, the map, the
    /// corporations, the players).
    fn g1822_state_fingerprint(&self) -> String {
        format!(
            "{}|{}|{}|{}",
            serde_json::to_string(&self.g1822).unwrap_or_default(),
            serde_json::to_string(&self.hexes).unwrap_or_default(),
            serde_json::to_string(&self.corporations).unwrap_or_default(),
            serde_json::to_string(&self.players).unwrap_or_default()
        )
    }

    /// The differential test of one state: the generator's moves under
    /// `policy` against the reference generator's ([`reference`]): (1) the
    /// sorted move strings ([`BaseGame::g1822_move_string`]), (2) the RL
    /// index rows (index, wire dict, price range; ascending), (3) the state
    /// after each call equals the state before. Returns `(moves compared,
    /// None)` when all agree, else a description of the first difference.
    pub fn g1822_reference_diff_with(&mut self, policy: CrossPlayerPolicy) -> (usize, Option<String>) {
        let before = self.g1822_state_fingerprint();
        let fast = self.g1822_legal_moves_with(policy);
        if self.g1822_state_fingerprint() != before {
            return (fast.len(), Some("the generator changed the state".into()));
        }
        let slow = self.g1822_legal_moves_reference(policy);
        if self.g1822_state_fingerprint() != before {
            return (fast.len(), Some("the reference generator changed the state".into()));
        }
        (fast.len(), self.g1822_compare_move_lists(&fast, &slow))
    }

    /// [`BaseGame::g1822_reference_diff_with`] under the title's policy AND
    /// the rules' at once, running the reference generator once (under the
    /// rules'): its list under the title's policy is derived from that one
    /// exactly as `g1822_legal_moves_inner` builds it -- the candidates do
    /// not depend on the policy, the policy only partitions them before the
    /// validation, which decides each candidate on its own clone (no class,
    /// dedup by the first occurrence of a key, and duplicates share their
    /// exclusion) -- so the reference's title-policy list is its rules' list
    /// without the excluded moves, or, when that leaves nothing, the
    /// excluded ones. The fast generator runs under both policies.
    pub fn g1822_reference_diff_both_policies(&mut self) -> (usize, Option<String>) {
        let title = self.title_def().generator_cross_player();
        let before = self.g1822_state_fingerprint();
        let fast_title = self.g1822_legal_moves_with(title);
        let fast_rules = self.g1822_legal_moves_with(CrossPlayerPolicy::RULES);
        if self.g1822_state_fingerprint() != before {
            return (0, Some("the generator changed the state".into()));
        }
        let slow_rules = self.g1822_legal_moves_reference(CrossPlayerPolicy::RULES);
        if self.g1822_state_fingerprint() != before {
            return (0, Some("the reference generator changed the state".into()));
        }
        let kept: Vec<Move> =
            slow_rules.iter().filter(|m| !self.g1822_cross_player_excluded(m, title)).cloned().collect();
        let slow_title: Vec<Move> = if kept.is_empty() {
            slow_rules.iter().filter(|m| self.g1822_cross_player_excluded(m, title)).cloned().collect()
        } else {
            kept
        };
        let n = fast_title.len() + fast_rules.len();
        if let Some(d) = self.g1822_compare_move_lists(&fast_rules, &slow_rules) {
            return (n, Some(format!("policy rules: {d}")));
        }
        if let Some(d) = self.g1822_compare_move_lists(&fast_title, &slow_title) {
            return (n, Some(format!("policy title: {d}")));
        }
        (n, None)
    }

    /// The fast and the reference generator's lists compared: sorted move
    /// strings, then the RL index rows (index, wire dict, price range;
    /// ascending; every move an index, no index twice).
    fn g1822_compare_move_lists(&self, fast: &[Move], slow: &[Move]) -> Option<String> {
        let strings = |ms: &[Move]| -> Vec<String> {
            let mut v: Vec<String> = ms.iter().map(Self::g1822_move_string).collect();
            v.sort();
            v
        };
        let (a, b) = (strings(fast), strings(slow));
        if a != b {
            let only_new: Vec<&String> = a.iter().filter(|x| !b.contains(x)).collect();
            let only_ref: Vec<&String> = b.iter().filter(|x| !a.contains(x)).collect();
            return Some(format!(
                "moves differ ({} new vs {} reference); only new: {only_new:?}; only reference: {only_ref:?}",
                a.len(),
                b.len()
            ));
        }
        if self.title_def().rl_layout_kind() == crate::title::RlLayoutKind::Generator {
            let rows = |g: &BaseGame, ms: &[Move]| -> Result<Vec<String>, GameError> {
                let mut v: Vec<(u32, String)> = Vec::with_capacity(ms.len());
                for m in ms {
                    let r = m.price.map(|r| (r.lo, r.hi, r.step));
                    v.push((g.rl_index_of::<Move>(m)?, format!("{}|{r:?}", g.g1822_move_wire(m))));
                }
                v.sort();
                if let Some(w) = v.windows(2).find(|w| w[0].0 == w[1].0) {
                    return Err(GameError::new(format!("index {} names two moves: {} / {}", w[0].0, w[0].1, w[1].1)));
                }
                Ok(v.into_iter().map(|(i, s)| format!("{i}|{s}")).collect())
            };
            match (rows(self, fast), rows(self, slow)) {
                (Ok(ia), Ok(ib)) if ia != ib => {
                    return Some(format!("index rows differ: new {ia:?} reference {ib:?}"));
                }
                (Ok(_), Ok(_)) => {}
                (ra, rb) => {
                    let e = |r: &Result<Vec<String>, GameError>| r.as_ref().err().map(|e| e.to_string());
                    return Some(format!("index API: new {:?} reference {:?}", e(&ra), e(&rb)));
                }
            }
        }
        None
    }

    /// `!self.g1822_legal_moves().is_empty()`, without generating every
    /// move when a pass the generator offers applies (the fast path's
    /// [`BaseGame::g1822_pass_probe`]).
    pub(crate) fn g1822_has_legal_move(&mut self) -> bool {
        if !reference::on() && self.g1822_pass_probe() {
            return true;
        }
        !self.g1822_legal_moves().is_empty()
    }

    /// Whether the generator's candidates certainly include the acting
    /// entity's `pass` (the conditions under which `gen_bsps` /
    /// `gen_choose_round` / `gen_or` emit it) and it applies on a clone: then
    /// the generator's list is not empty (a pass is never excluded by the
    /// cross-player policy and survives validation). False says nothing.
    pub(crate) fn g1822_pass_probe(&mut self) -> bool {
        if self.finished || self.g1822.is_none() {
            return false;
        }
        let Some(bi) = self.g1822_active_step() else { return false };
        let step = self.rnd().steps[bi];
        let Some(ent) = self.g1822_step_current_entity(bi) else { return false };
        let offered = match step {
            Step::BuySellParShares => {
                matches!(ent, Ent::Player(_))
                    && !(self.g1822_is_pnw() && self.rnd().converting_major.is_some())
                    && self.g1822_bsps_actions(&ent).contains(&"pass")
            }
            Step::Choose => matches!(ent, Ent::Player(_)) && !self.g1822_choose_actions(&ent).is_empty(),
            _ => {
                let types = self.g1822_step_actions(bi, &ent);
                let has_pass = types.contains(&"pass");
                let corp = matches!(ent, Ent::Corp(_));
                !types.is_empty()
                    && match step {
                        // (not BuyTrain: whether its pass applies reads the
                        // route latch, which the generation of the other
                        // candidates may still fill before the reference
                        // validates the pass)
                        Step::Track | Step::Token | Step::IssueShares => corp && has_pass,
                        Step::SpecialTrack | Step::SpecialToken => matches!(ent, Ent::Company(_)) && has_pass,
                        Step::AcquireCompany => corp && !(self.g1822_is_pnw() && !self.rnd().p20_choices.is_empty()),
                        Step::AcquisitionTrack => corp,
                        Step::FirstTurnHousekeeping | Step::CashOutNdem => true,
                        Step::MinorAcquisition => {
                            corp && has_pass && self.rnd().acquire_state == AcquireState::SelectMinor
                        }
                        _ => false,
                    }
            }
        };
        if !offered {
            return false;
        }
        let pass = Move::plain(Action::Pass { entity_id: ent.id() });
        let mut g = self.g1822_validation_clone();
        g.g1822_apply_move_nobackup(&pass).is_ok()
    }

    /// A throwaway clone to apply a candidate on: the reference copies the
    /// JSON action log, which the family's processing never reads.
    pub(crate) fn g1822_validation_clone(&self) -> BaseGame {
        if reference::on() {
            self.clone_for_search()
        } else {
            self.clone_for_search_without_log()
        }
    }

    fn g1822_legal_moves_inner(&mut self, policy: CrossPlayerPolicy) -> Vec<Move> {
        let Some(bi) = self.g1822_active_step() else { return Vec::new() };
        let step = self.rnd().steps[bi];
        let Some(ent) = self.g1822_step_current_entity(bi) else { return Vec::new() };
        let t = super::prof::start();
        let mut cands: Vec<Move> = match step {
            Step::BuySellParShares => self.gen_bsps(&ent),
            Step::Choose => self.gen_choose_round(&ent),
            _ => self.gen_or(bi, step, &ent),
        };
        super::prof::stop(t, || format!("gen:{step:?}"));
        // Companies acting in parallel at the earlier steps (Ruby
        // actions_for over steps up to the blocking one): SpecialTrack /
        // SpecialToken company lays and tokens during the operator's turn.
        if self.rnd().kind == RoundKind::Operating && matches!(ent, Ent::Corp(_)) {
            let t = super::prof::start();
            cands.extend(self.gen_parallel_companies(bi, &ent));
            super::prof::stop(t, || "gen:parallel_companies".into());
        }
        let (kept, dropped): (Vec<Move>, Vec<Move>) =
            cands.into_iter().partition(|m| !self.g1822_cross_player_excluded(m, policy));
        let t = super::prof::start();
        let out = self.g1822_validate(kept);
        super::prof::stop(t, || "validate:all".into());
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
        // (the fast path dedups through a set of the same keys `same_move`
        // compares; the reference compares every pair)
        let fast = !reference::on();
        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
        // The stock round's bid-box bids form one validation class (see
        // `g1822_bid_own_checks`): the first bid whose own checks pass is
        // applied on a clone, the others take its verdict.
        let mut bid_class: Option<bool> = None;
        for m in cands {
            if fast {
                if !seen.insert(move_key(&m)) {
                    continue;
                }
            } else if out.iter().any(|o| same_move(o, &m)) {
                continue;
            }
            if m.validated {
                out.push(m);
                continue;
            }
            let bid_own = if fast { self.g1822_bid_own_checks(&m) } else { None };
            match (bid_own, bid_class) {
                (Some(false), _) => continue,
                (Some(true), Some(v)) => {
                    if v {
                        out.push(m);
                    }
                    continue;
                }
                _ => {}
            }
            let tp = super::prof::start();
            let mut g = self.g1822_validation_clone();
            super::prof::stop(tp, || "validate:clone".into());
            let tp = super::prof::start();
            let ok = g.g1822_apply_move_nobackup(&m).is_ok();
            super::prof::stop(tp, || {
                let kind = match self.g1822_entity(m.action.entity_id()) {
                    Ent::Player(_) => "player",
                    Ent::Corp(_) => "corp",
                    Ent::Company(_) => "company",
                };
                format!("validate:process:{}/{}:{}", m.action.action_type(), kind, if ok { "ok" } else { "rejected" })
            });
            if bid_own == Some(true) {
                bid_class = Some(ok);
            }
            if ok {
                out.push(m);
            }
        }
        stats::add(&stats::VALIDATE_NS, t.elapsed().as_nanos() as u64);
        out
    }

    /// The validation class of the stock round's bid-box bids: Some(own
    /// checks) for a player's `bid` on a company while the round's
    /// BuySellParShares step is the active one, else None. Proof that a
    /// member applies iff the class's shared conditions hold and its own
    /// checks pass: the step dispatch reads only the action's type and
    /// entity (the same for every member); `process_bid` fails exactly on
    /// its own checks -- an unknown company, a price below the minimum
    /// (the highest bid + the increment, or the company's minimum bid), off
    /// the increment, or above the player's cash less the cash their other
    /// bids commit -- and otherwise mutates only the company's bid list and
    /// the round's bid counters (the same for every member); what follows
    /// (`bsps_pass` on the bid count, the next entity, the transition loop)
    /// reads the round's counters and the players' passed flags, which every
    /// member sets alike (a bid unsets its bidder's pass, so the round never
    /// ends on a bid). So the first member with its own checks passing
    /// decides the shared part for all.
    fn g1822_bid_own_checks(&self, m: &Move) -> Option<bool> {
        let Action::Bid { entity_id, company_sym, price } = &m.action else { return None };
        if company_sym.is_empty() {
            return None;
        }
        let bi = self.g1822_active_step()?;
        if self.rnd().steps[bi] != Step::BuySellParShares {
            return None;
        }
        let Ent::Player(pid) = self.g1822_entity(entity_id) else { return None };
        let Some(ci) = self.co_idx(company_sym) else { return Some(false) };
        let inc = self.fam().min_bid_increment;
        let min = match self.g1822_highest_bid(company_sym) {
            Some(h) => h.price + inc,
            None => self.companies[ci].min_bid(),
        };
        let max = self.player_cash(pid) - self.g1822_committed_cash(pid);
        Some(*price >= min && (*price - min) % inc == 0 && *price <= max)
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
            // (fast path: a par of a corporation `can_par?` refuses fails in
            // process_par -- or at the step gate before it -- whatever the
            // price, so its candidates would all be dropped by the apply)
            if !reference::on() && !self.g1822_can_par(&c.sym, pid) {
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
        let t = super::prof::start();
        let types = self.g1822_step_actions(bi, ent);
        super::prof::stop(t, || "gen:step_actions".into());
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
            // 1822CA (ca_legal): the sawmill's choice once assigned (the
            // assignment itself is a parallel company move)
            Step::AssignSawmill => out.extend(self.gen_ca_sawmill_choices()),
            Step::AcquisitionTrack => {
                if let Ent::Corp(c) = ent {
                    out.extend(self.gen_corp_lays(c));
                    out.push(pass());
                }
            }
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
            Step::DestinationToken if self.g1822_is_ca() => {
                if let Ent::Corp(c) = ent {
                    out.extend(self.gen_ca_destination(c));
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
                    if has("choose") && self.g1822_is_ca() {
                        out.extend(self.gen_ca_route_choices(c));
                    } else if has("choose") {
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
        let grain = co.trains.iter().filter(|t| self.g1822_is_ca() && Self::g1822_ca_grain_train(t)).count();
        let eligible: Vec<&crate::entities::Train> = if grain > 1 {
            // 1822CA DiscardTrain#trains: the grain trains first
            co.trains.iter().filter(|t| Self::g1822_ca_grain_train(t)).collect()
        } else if perm > 1 {
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
        // (1822CA pending_home_tokeners: QMOO too, once Quebec has a tile)
        let home_tokener = p.corp == self.fam().minor_14_id || (self.g1822_is_ca() && p.corp == "QMOO");
        let graph = (!home_tokener).then(|| self.g1822_graph(&p.corp));
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
            let mut ca_assign = false;
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
                    Step::AssignSawmill if acts.contains(&"assign") => ca_assign = true,
                    Step::SpecialChoose if acts.contains(&"choose_ability") => special_choose = true,
                    _ => {}
                }
            }
            if assign {
                out.extend(self.gen_pnw_assigns(&co));
            }
            if ca_assign {
                out.extend(self.gen_ca_assigns(&co));
            }
            if special_choose {
                out.extend(self.gen_mx_special_choose(&co));
            }
            if lay {
                out.extend(self.gen_company_lays(&co, false));
            }
            if tok {
                if self.g1822_is_ca() {
                    out.extend(self.gen_ca_special_tokens(&co));
                } else {
                    out.extend(self.gen_special_tokens(&co));
                }
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
        let tp = super::prof::start();
        let out = self.gen_corp_lays_inner(c);
        super::prof::stop(tp, || "lays:corp".into());
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
        let t = super::prof::start();
        let graph = self.g1822_graph(c);
        super::prof::stop(t, || "lays:graph".into());
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
            let t = super::prof::start();
            let tiles = self.g1822_potential_tiles(c, hi);
            super::prof::stop(t, || "lays:potential_tiles".into());
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
        let tp = super::prof::start();
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
        super::prof::stop(tp, || "lays:rotations".into());
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
                    let tp = super::prof::start();
                    let exits = def.rotated(r).edges.clone();
                    let cost = self.g1822_corp_lay_cost(operator, hi, &exits, nbrs);
                    super::prof::stop(tp, || "lays:cost".into());
                    if cost > cash {
                        continue;
                    }
                    m.validated = true;
                }
                // 1822PNW (fast path): a plain lay decided without applying
                // it, then checked like 1822's own lays (see
                // `g1822_pnw_lay_prevalidate`)
                let pnw_pre = if restrict && self.g1822_is_pnw() && !reference::on() {
                    self.g1822_pnw_lay_prevalidate(operator, hi, &t, &def, r, &inst, nbrs)
                } else {
                    None
                };
                if cfg!(test) {
                    if let Some(v) = pnw_pre {
                        let mut g = self.g1822_validation_clone();
                        let applied = g.g1822_apply_move_nobackup(&m).is_ok();
                        let short = self.lay_restrictions_shortcut(operator, hi, &def, r, connected_nodes, connected);
                        // (an apply that fails only on the restriction check is
                        // the restriction verdict's business)
                        let restricted = !applied && v && short != Some(true);
                        assert!(applied == v || restricted, "1822PNW lay pre-validation {v} vs apply {applied}: {:?}", m.action);
                        if short == Some(true) {
                            assert_eq!(applied, v, "1822PNW lay pre-validation vs apply (shortcut true): {:?}", m.action);
                        }
                    }
                }
                match pnw_pre {
                    Some(false) => continue,
                    Some(true) => {
                        m.validated = true;
                        if !self.lay_passes_restrictions(&m, operator, hi, &def, r, connected_nodes, connected) {
                            continue;
                        }
                        out.push(m);
                        continue;
                    }
                    None => {}
                }
                if restrict && self.g1822_is_pnw() && !reference::on() {
                    // 1822PNW (fast path): the reference checks the
                    // restrictions (its shortcuts, else apply + check on a
                    // clone) and then applies the lay again to validate it;
                    // here one apply decides both. Reference result:
                    // shortcut_or(applied && check) && applied
                    //   = applied && (shortcut || check); the apply is
                    // deterministic, so the validation is skipped.
                    let short = self.lay_restrictions_shortcut(operator, hi, &def, r, connected_nodes, connected);
                    if short == Some(false) {
                        continue;
                    }
                    let old = self.hexes[hi].tile.clone();
                    let mut g = self.g1822_validation_clone();
                    let applied = g.g1822_apply_move_nobackup(&m).is_ok();
                    if !(applied && (short == Some(true) || g.g1822_track_restrictions_ok(operator, hi, &old))) {
                        continue;
                    }
                    m.validated = true;
                    out.push(m);
                    continue;
                }
                if restrict && !self.lay_passes_restrictions(&m, operator, hi, &def, r, connected_nodes, connected) {
                    continue;
                }
                out.push(m);
            }
        }
        out
    }

    /// 1822PNW: whether a corporation's own Track lay of `def` at `rotation`
    /// (instance `inst`, a rotation `lays_for` found legal) applies, decided
    /// without applying it -- Some(verdict) for the plain cases, None for
    /// the cases with effects of their own (then the lay is applied on a
    /// clone, as the reference does). Some(true) claims what 1822's own
    /// lays claim (every check the process makes but the restrictions,
    /// which `lay_passes_restrictions` settles): the Track step's portage
    /// gate; `lay_tile_action`'s slot rules (an upgrade = a non-white
    /// colour off the token cities); `lay_tile`'s forest, `upgrades_to?`
    /// and supply checks; Hex#lay's token move (an old city's tokens need a
    /// new city); and the cost: the hex's terrain (`upgrade_cost`) plus the
    /// water borders the new track crosses, less 40 per builder cube the
    /// new tile carries -- its own or the old tile's, whichever is more
    /// (`carry_cubes`) -- never below 0, within the corporation's cash.
    /// None: token cities (their own cost and upgrade rule), an old
    /// blue / gray / red tile (the lay opens neighbour links the border
    /// cost reads), H11 / H13 (Seattle's border), the PNW tiles, H19's
    /// city+town tiles, a hex next to or holding multi-lane track (the
    /// restriction shortcuts assume single lanes) and a corporation whose
    /// graph Ruby's cache answers stale (the shortcuts read the generator's
    /// graph).
    pub(crate) fn g1822_pnw_lay_prevalidate(
        &self,
        c: &str,
        hi: usize,
        base: &str,
        def: &crate::tiles::TileDef,
        rotation: u8,
        inst: &str,
        nbrs: &HashMap<usize, BTreeMap<u8, usize>>,
    ) -> Option<bool> {
        let hex = &self.hexes[hi];
        let hex_id = hex.id.as_str();
        if Self::g1822_pnw_tokencity(hex_id)
            || matches!(hex_id, "H11" | "H13")
            || base.starts_with("PNW")
            || Self::g1822_pnw_city_and_town_tile(hex_id, base)
        {
            return None;
        }
        let old_color = self.g1822_tile_color_name(hi);
        if matches!(old_color, "blue" | "gray" | "red") || self.g().ruby_graphs.contains(c) {
            return None;
        }
        if !self.g1822_pnw_single_lanes_around(hi, base) {
            return None;
        }
        let ci = *self.corp_idx.get(c)?;
        if self.rnd().num_laid_portage > 0 {
            return Some(false);
        }
        let upgrade = old_color != "white";
        let Some((lay, can_upgrade, no_reuse)) = self.g1822_tile_lay(c, false) else { return Some(false) };
        if (upgrade && !can_upgrade)
            || (def.color == crate::tiles::TileColor::Yellow && !lay)
            || (no_reuse && self.rnd().laid_hexes.iter().any(|h| h == hex_id))
            || self.g1822_pnw_assigned(hex_id, "forest")
            || !self.g1822_upgrades_to_sel(hi, def, false, None)
            || self.tile_counts_remaining.get(base).copied().unwrap_or(0) == 0
        {
            return Some(false);
        }
        let new_tile = BaseGame::tile_from_def(def, rotation);
        if hex.tile.cities.len() > 1 {
            let map = BaseGame::g1822_city_map(&hex.tile, &new_tile);
            if map
                .iter()
                .enumerate()
                .any(|(oi, m)| m.is_none() && hex.tile.cities[oi].tokens.iter().any(|t| t.is_some()))
            {
                return Some(false);
            }
        }
        let exits: Vec<u8> =
            new_tile.paths.iter().flat_map(|p| [p.a.edge_num(), p.b.edge_num()]).flatten().collect();
        let border: i32 = self
            .g()
            .borders
            .get(hex_id)
            .map(|v| {
                v.iter()
                    .filter(|(edge, _, _)| nbrs.get(&hi).map_or(false, |m| m.contains_key(edge)) && exits.contains(edge))
                    .map(|(_, cost, _)| *cost)
                    .sum()
            })
            .unwrap_or(0);
        let cubes = |id: &str| self.g().builder_cubes.get(id).copied().unwrap_or(0);
        let carried = cubes(&hex.tile.id).max(cubes(inst)) as i32;
        let cost = (self.g1822_upgrade_cost(hi, c) + border - 40 * carried).max(0);
        Some(self.corporations[ci].cash >= cost)
    }

    /// Whether hex `hi` (its current tile and a `new_base` tile) and its
    /// neighbours' tiles hold single-lane track only (1822PNW's lanes).
    fn g1822_pnw_single_lanes_around(&self, hi: usize, new_base: &str) -> bool {
        let f = self.fam();
        let single = |dsl: Option<&str>| {
            dsl.map_or(true, |d| {
                d.split(';').filter_map(|p| p.trim().strip_prefix("path=")).all(|attrs| {
                    super::graph::lane_specs(attrs).iter().all(|&(a, b)| a == (1, 0) && b == (1, 0))
                })
            })
        };
        let tile_ok = |h: usize| {
            let hex = &self.hexes[h];
            let dsl = if hex.tile.id.starts_with("preprinted_") {
                (f.hex_dsl)(&hex.id).map(|(d, _)| d)
            } else {
                let base = hex.tile.name.split('-').next().unwrap_or(&hex.tile.name);
                (f.tile_dsl)(base).map(|(d, _)| d)
            };
            single(dsl)
        };
        if !single((f.tile_dsl)(new_base).map(|(d, _)| d)) || !tile_ok(hi) {
            return false;
        }
        self.hex_adjacency
            .get(&self.hexes[hi].id)
            .map_or(true, |adj| adj.values().all(|nid| self.hex_idx.get(nid).map_or(true, |&ni| tile_ok(ni))))
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
        if let Some(v) = self.lay_restrictions_shortcut(operator, hi, def, rotation, connected_nodes, connected_edges) {
            return v;
        }
        let t0 = std::time::Instant::now();
        stats::add(&stats::RESTRICT_N, 1);
        let tp = super::prof::start();
        let r = self.lay_passes_restrictions_inner(m, operator, hi);
        super::prof::stop(tp, || "lays:restriction_walk".into());
        stats::add(&stats::RESTRICT_NS, t0.elapsed().as_nanos() as u64);
        r
    }

    /// The restriction verdicts that need no lay on a clone: Some(verdict)
    /// when one of the shortcuts settles it, None when only the lay and the
    /// walk after it can (`lay_passes_restrictions_inner`).
    fn lay_restrictions_shortcut(
        &self,
        operator: &str,
        hi: usize,
        def: &crate::tiles::TileDef,
        rotation: u8,
        connected_nodes: &std::collections::HashSet<super::graph::Node>,
        connected_edges: &[u8],
    ) -> Option<bool> {
        let hex = &self.hexes[hi];
        if hex.tile.paths.is_empty()
            || Family::is(self.fam().english_channel_hex, &hex.id)
            || Family::is(self.fam().cardiff_hex, &hex.id)
        {
            return Some(true); // laying on track-less ground always uses new track
        }
        if self.g1822_is_ca() {
            // TRACK_RESTRICTION :permissive: only 'New track must override'
            let new_tv = super::tile_rules::TV::from_def(def, rotation);
            let old_tv = super::tile_rules::TV::from_tile(&hex.tile);
            return Some(super::tile_rules::cities_towns_kept(&old_tv, &new_tv));
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
                    return Some(true);
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
                return Some(true);
            }
        }
        None
    }

    fn lay_passes_restrictions_inner(&self, m: &Move, operator: &str, hi: usize) -> bool {
        let old = self.hexes[hi].tile.clone();
        let mut g = self.g1822_validation_clone();
        // (the fast path applies without the rollback copy: the clone is
        // thrown away on an error either way)
        let applied = if reference::on() { g.g1822_apply_move(m).is_ok() } else { g.g1822_apply_move_nobackup(m).is_ok() };
        if !applied {
            return false;
        }
        g.g1822_track_restrictions_ok(operator, hi, &old)
    }

    /// SpecialTrack lays by a company (alone and in Ruby-valid combos).
    fn gen_company_lays(&mut self, co: &str, in_step: bool) -> Vec<Move> {
        let tp = super::prof::start();
        let out = self.gen_company_lays_inner(co, in_step);
        super::prof::stop(tp, || "lays:company".into());
        out
    }

    fn gen_company_lays_inner(&mut self, co: &str, in_step: bool) -> Vec<Move> {
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
            // 1822CA SpecialTrack#available_hex (sawmill hex, big cities)
            let ca_hex = self.g1822_ca_special_hex(&entities, hi);
            if ca_hex == Some(true) {
                continue;
            }
            if ab.kind != AbKind::Teleport && !Family::is(f.company_hsbc, co) && ca_hex.is_none() {
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
                let tp = super::prof::start();
                let live: Vec<Move> = moves
                    .iter()
                    .filter(|m| {
                        let mut g = self.g1822_validation_clone();
                        g.g1822_apply_move_nobackup(m).is_ok() && (g.finished || g.g1822_has_legal_move())
                    })
                    .cloned()
                    .collect();
                super::prof::stop(tp, || "minor_acq:lookahead".into());
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
        // (CA keeps G1822's [:down] * n, which a one-row market moves left)
        let pnw = self.g1822_is_pnw() || self.g1822_is_mx() || self.g1822_is_ca();
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
                    // 1822CA DestinationToken#process_place_token takes
                    // `action.token`: Ruby's PlaceToken picks the token by
                    // token_type (nil → a normal one), so name the
                    // destination token
                    if self.g1822_is_ca()
                        && self.g1822_active_step().map_or(false, |i| self.rnd().steps[i] == Step::DestinationToken)
                    {
                        o.insert("token_type".into(), json!("destination"));
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
            format!("{}-{}", self.hexes[hi].id, (self.fam().hex_index)(&self.hexes[hi].id))
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

/// The key `same_move` compares.
fn move_key(m: &Move) -> String {
    format!("{:?}{:?}{:?}", m.action, m.combo, m.share_price)
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

