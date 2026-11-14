//! 1867 legal-move generation (the RL-readiness contract, item 1; bridge
//! design T3-T6).
//!
//! `g1867_legal_moves()` enumerates, for the live state, the moves Ruby's
//! UI offers the acting entity: the shared step machine's offers
//! (`BaseGame::step_offers` — the title's step list, the blocking step and
//! the action types each active step accepts, attributed to the step the
//! dispatch hands them to) expanded into concrete actions, one [`Move`]
//! per distinct choice. Price-bearing moves (bids, a company bought for
//! $1..face, a train bought from another corporation) carry a
//! [`PriceRange`]; the move's own action holds the range's low end.
//!
//! Soundness: each generator applies the rules Ruby checks when a PLAYER
//! acts that the replay handler skips like Ruby's `@game.loading` (tile
//! legality, home-token locations, trade-in timing, ...), and every
//! candidate is then applied to a clone, so the process-side rules (cash,
//! slots, step gates) filter too. `g1867_generator_status` names, per step
//! kind and action type, whether a generator exists or the type is
//! replay-only; the fuzzer (`super::fuzz`) fails on any panic, any
//! generated move that does not apply and any unfinished state without
//! moves.

use serde_json::{json, Map, Value};

use crate::actions::{Action, DividendKind, GameError};
use crate::entities::EntityId;
use crate::game::BaseGame;
use crate::rounds::Round;
use crate::steps::{StepEntity, StepKind};
use crate::title::{CorpType, CrossPlayerPolicy};

/// Ruby G1867::Step::BuySellParShares MIN_BID (minor founding bids).
const MINOR_MIN_BID: i32 = 100;
/// Ruby MIN_BID_INCREMENT (both auctions; MUST_BID_INCREMENT_MULTIPLE).
const BID_STEP: i32 = 5;

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
    /// The step kind the move is offered by (diagnostics / status table).
    pub step: StepKind,
    /// The price-bearing parameter's range (the action holds `lo`).
    pub price: Option<PriceRange>,
    /// The share certificates' Ruby ids, for the wire form.
    pub shares: Vec<String>,
    /// The generator already checked everything the process checks:
    /// skip the apply-on-a-clone filter.
    pub validated: bool,
}

impl Move {
    fn plain(step: StepKind, action: Action) -> Move {
        Move { action, step, price: None, shares: Vec::new(), validated: false }
    }

    fn ranged(step: StepKind, action: Action, lo: i32, hi: i32, step_size: i32) -> Move {
        let hi = lo + (hi - lo).max(0) / step_size * step_size;
        Move { action, step, price: Some(PriceRange { lo, hi, step: step_size }), shares: Vec::new(), validated: false }
    }

    /// The move with its price-bearing parameter set to `p`.
    pub fn with_price(&self, p: i32) -> Move {
        let mut m = self.clone();
        match &mut m.action {
            Action::Bid { price, .. }
            | Action::CorporationBid { price, .. }
            | Action::BuyTrain { price, .. }
            | Action::BuyCompany { price, .. } => *price = p,
            _ => {}
        }
        m
    }
}

/// Candidates whose validation outcome is shared
/// (`BaseGame::g67_validation_class`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ValidationClass {
    Par = 0,
    MinorOpeningBid = 1,
    TokenPlace = 2,
    StockHomeToken = 3,
    StockSell = 4,
}

impl ValidationClass {
    const COUNT: usize = 5;
}

/// Per-state data the validation classes' own checks share.
#[derive(Default)]
struct ClassCtx {
    /// A corporation's tokenable cities on its freshly computed graph.
    tokenable: Option<(String, Vec<(String, usize)>)>,
}

/// Whether a (step, action type) pair has a generator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GenStatus {
    Generated,
    /// Never generated; the reason says why replay still accepts it.
    ReplayOnly(&'static str),
}

/// The RL-readiness table: every action type a 1867 step can offer.
pub fn g1867_generator_status(kind: StepKind, action_type: &str) -> GenStatus {
    use GenStatus::*;
    use StepKind as K;
    match (kind, action_type) {
        (K::SingleItemAuction, "bid" | "pass") => Generated,
        (K::MajorTrainless, "choose" | "pass") => Generated,
        (K::DiscardTrain, "discard_train") => Generated,
        (K::HomeToken, "place_token") => Generated,
        (K::BuySellParShares, "bid" | "par" | "buy_shares" | "sell_shares" | "pass") => Generated,
        (K::BuyCompany | K::BuyCompanyPreloan, "buy_company" | "pass") => Generated,
        (K::RedeemShares, "buy_shares" | "pass") => Generated,
        (K::Track, "lay_tile" | "pass") => Generated,
        (K::Token, "place_token" | "pass") => Generated,
        (K::Route, "run_routes") => Generated,
        (K::Dividend, "dividend") => Generated,
        (K::BuyTrain, "buy_train" | "pass") => Generated,
        (K::Merge, "merge" | "convert" | "pass") => Generated,
        (K::PostMergerShares, "buy_shares" | "pass") => Generated,
        (K::ReduceTokens, "remove_token") => Generated,
        (K::Bankrupt, "bankrupt") => ReplayOnly(
            "Ruby 1867 lists no Bankrupt step: a corporation that must buy a train can always \
             afford one with loans (must_buy_train? requires it), else it passes into nationalization",
        ),
        (K::BuyTrain, "sell_shares") => ReplayOnly("1867 EMR never sells shares (BuyTrain#can_sell? false)"),
        _ => ReplayOnly("not offered by this step in 1867"),
    }
}

/// Generator counters (cheap; read by the fuzz probe).
pub mod stats {
    use std::sync::atomic::{AtomicU64, Ordering};
    pub static CANDS: AtomicU64 = AtomicU64::new(0);
    pub static REJECTED: AtomicU64 = AtomicU64::new(0);
    /// States where the cross-player policy left no move (rules' set used).
    pub static CROSS_FALLBACK: AtomicU64 = AtomicU64::new(0);
    /// Merge-step adds the policy dropped (a minor of another owner).
    pub static CROSS_DROPPED_MERGES: AtomicU64 = AtomicU64::new(0);
    /// Time (ns) in step_offers / candidate generation / validation, and
    /// in the Track generator's tile search vs its lay checks.
    pub static OFFERS_NS: AtomicU64 = AtomicU64::new(0);
    pub static GEN_NS: AtomicU64 = AtomicU64::new(0);
    pub static VALIDATE_NS: AtomicU64 = AtomicU64::new(0);
    pub static LAY_SEARCH_NS: AtomicU64 = AtomicU64::new(0);
    pub static LAY_CHECK_NS: AtomicU64 = AtomicU64::new(0);
    pub static LAY_GRAPH_NS: AtomicU64 = AtomicU64::new(0);
    pub fn add(c: &AtomicU64, v: u64) {
        c.fetch_add(v, Ordering::Relaxed);
    }
    pub fn get(c: &AtomicU64) -> u64 {
        c.load(Ordering::Relaxed)
    }

    // -- the profile (off by default): (count, ns) per named section --------
    use std::collections::BTreeMap;
    use std::sync::atomic::AtomicBool;
    use std::sync::Mutex;
    static PROFILING: AtomicBool = AtomicBool::new(false);
    static PROFILE: Mutex<BTreeMap<String, (u64, u64)>> = Mutex::new(BTreeMap::new());
    /// Turn the per-section profile on or off (`g1867_profile` in Python).
    pub fn set_profiling(on: bool) {
        PROFILING.store(on, Ordering::Relaxed);
    }
    #[inline]
    pub fn profiling() -> bool {
        PROFILING.load(Ordering::Relaxed)
    }
    /// A section's start (None while the profile is off).
    #[inline]
    pub fn start() -> Option<std::time::Instant> {
        profiling().then(std::time::Instant::now)
    }
    /// Credit the time since `t0` to section `key` (a no-op while off).
    pub fn stop(t0: Option<std::time::Instant>, key: impl FnOnce() -> String) {
        let Some(t0) = t0 else { return };
        record(key, t0.elapsed().as_nanos() as u64);
    }
    /// Credit `ns` to section `key` (a no-op while off).
    pub fn record(key: impl FnOnce() -> String, ns: u64) {
        if !profiling() {
            return;
        }
        let mut p = PROFILE.lock().unwrap_or_else(|e| e.into_inner());
        let e = p.entry(key()).or_insert((0, 0));
        e.0 += 1;
        e.1 += ns;
    }
    /// Count an event in section `key` without time.
    pub fn count(key: impl FnOnce() -> String) {
        if !profiling() {
            return;
        }
        let mut p = PROFILE.lock().unwrap_or_else(|e| e.into_inner());
        p.entry(key()).or_insert((0, 0)).0 += 1;
    }
    /// The profile so far, then cleared: `(section, count, ns)`.
    pub fn take() -> Vec<(String, u64, u64)> {
        let mut p = PROFILE.lock().unwrap_or_else(|e| e.into_inner());
        std::mem::take(&mut *p).into_iter().map(|(k, (c, ns))| (k, c, ns)).collect()
    }
}

/// The reference generator: the code paths as they were before the
/// 2026-11-13 speed-ups, kept callable so a differential test can compare
/// the two move lists state by state (`g1867_legal_moves_reference`;
/// Python `g1867_moves_debug(policy, reference=True)`). Thread-local, so
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

/// Memoised replay-graph computations while the generator runs (the fast
/// path installs it for a whole `g1867_legal_moves_with` call): every
/// validation clone of one state starts from the state's map, and each
/// recomputed the operating corporation's graph (`map::GraphCache` is per
/// game, empty in a clone; the state itself computes it too when its cache
/// was cleared) — ~0.4-0.7 ms apiece on a late map.
/// `map::compute_corp_graph` is a pure function of its inputs
/// (the corporation, the hexes, the static hex index and adjacency, the
/// token cities, the reservations); a memoised result is handed out only
/// for inputs EQUAL to the ones it was computed from (the hexes compared
/// in full), so it is the graph the clone would have computed. Only active
/// while a [`graph_memo::Guard`] lives (thread-local): never for 1830 or
/// replay.
pub mod graph_memo {
    use std::cell::RefCell;
    use std::collections::HashMap;

    use crate::graph::Hex;
    use crate::map::CorpGraph;

    type Tokens = Vec<(String, usize)>;
    type Reservations = Vec<(String, usize, String)>;

    struct Memo {
        /// The map every entry was computed on: the map of the first
        /// computation made while the memo is installed (copied then, so a
        /// state that computes no graph pays nothing).
        hexes: Option<Vec<Hex>>,
        hex_idx: *const HashMap<String, usize>,
        hex_adjacency: *const HashMap<String, HashMap<u8, String>>,
        entries: Vec<(String, Tokens, Reservations, CorpGraph)>,
    }

    thread_local! {
        static MEMO: RefCell<Option<Memo>> = const { RefCell::new(None) };
    }

    /// While the guard lives, graph computations on the game's static hex
    /// index and adjacency are memoised. Dropping it restores the previous
    /// memo (none).
    pub struct Guard(Option<Memo>);
    impl Drop for Guard {
        fn drop(&mut self) {
            let prev = self.0.take();
            MEMO.with(|m| *m.borrow_mut() = prev);
        }
    }

    /// Install a memo for graphs computed with `hex_idx` / `hex_adjacency`
    /// (only compared by address: the game's `Arc`s, shared by its clones).
    pub fn install(
        hex_idx: &HashMap<String, usize>,
        hex_adjacency: &HashMap<String, HashMap<u8, String>>,
    ) -> Guard {
        let memo = Memo { hexes: None, hex_idx, hex_adjacency, entries: Vec::new() };
        Guard(MEMO.with(|m| m.borrow_mut().replace(memo)))
    }

    /// `compute()` = `compute_corp_graph(corp, hexes, hex_idx, hex_adjacency,
    /// tokens, reservations)`, or the memoised result for the same inputs.
    pub fn get_or(
        corp: &str,
        hexes: &[Hex],
        hex_idx: &HashMap<String, usize>,
        hex_adjacency: &HashMap<String, HashMap<u8, String>>,
        tokens: &[(String, usize)],
        reservations: &[(String, usize, String)],
        compute: impl FnOnce() -> CorpGraph,
    ) -> CorpGraph {
        // None: no memo (or another game's); Some(false): the memo is of
        // another map; Some(true): usable.
        let usable = MEMO.with(|m| {
            m.borrow_mut().as_mut().and_then(|memo| {
                if !std::ptr::eq(memo.hex_idx, hex_idx) || !std::ptr::eq(memo.hex_adjacency, hex_adjacency) {
                    return None;
                }
                Some(match &memo.hexes {
                    Some(h) => h.as_slice() == hexes,
                    None => {
                        memo.hexes = Some(hexes.to_vec());
                        true
                    }
                })
            })
        });
        if usable != Some(true) {
            let t = super::stats::start();
            let g = compute();
            super::stats::stop(t, || {
                if usable.is_some() { "graph:validation_changed_map" } else { "graph:outside_validation" }.into()
            });
            return g;
        }
        let hit = MEMO.with(|m| {
            m.borrow().as_ref().and_then(|memo| {
                memo.entries
                    .iter()
                    .find(|(c, t, r, _)| c == corp && t.as_slice() == tokens && r.as_slice() == reservations)
                    .map(|(_, _, _, g)| g.clone())
            })
        });
        if let Some(g) = hit {
            super::stats::count(|| "graph_memo:hit".into());
            return g;
        }
        let t = super::stats::start();
        let g = compute();
        super::stats::stop(t, || "graph:validation_memo_miss".into());
        MEMO.with(|m| {
            if let Some(memo) = m.borrow_mut().as_mut() {
                memo.entries.push((corp.to_string(), tokens.to_vec(), reservations.to_vec(), g.clone()));
            }
        });
        g
    }
}

fn base_name(id: &str) -> &str {
    id.rsplit_once('-').map_or(id, |(n, _)| n)
}

impl BaseGame {
    // =====================================================================
    // Entry points
    // =====================================================================

    /// Every legal move under the title's cross-player policy
    /// (`GameTitle::generator_cross_player`, default: no purchases from
    /// another player's holdings, no mergers with another player's minors).
    pub fn g1867_legal_moves(&mut self) -> Vec<Move> {
        let policy = self.title_def().generator_cross_player();
        self.g1867_legal_moves_with(policy)
    }

    /// Every legal move under `policy`: moves it excludes are dropped —
    /// unless nothing else is legal, when the rules' set is returned
    /// rather than a dead end. `CrossPlayerPolicy::RULES` = everything
    /// Ruby accepts.
    pub fn g1867_legal_moves_with(&mut self, policy: CrossPlayerPolicy) -> Vec<Move> {
        if self.finished || self.title != "1867" {
            return Vec::new();
        }
        let t_all = stats::start();
        // The fast paths share their (pure) replay-graph computations — this
        // state's own and its validation clones' (`graph_memo`).
        let _memo = (!reference::on()).then(|| graph_memo::install(&self.hex_idx, &self.hex_adjacency));
        let cands = self.g1867_candidates();
        let (kept, dropped): (Vec<Move>, Vec<Move>) =
            cands.into_iter().partition(|m| !self.g1867_cross_player_excluded(m, policy));
        let merges = dropped.iter().filter(|m| matches!(m.action, Action::Merge { .. })).count();
        stats::add(&stats::CROSS_DROPPED_MERGES, merges as u64);
        let mut out = self.g1867_validate(kept);
        if out.is_empty() && !dropped.is_empty() {
            stats::add(&stats::CROSS_FALLBACK, 1);
            out = self.g1867_validate(dropped);
        }
        // A deterministic order (some candidate sources iterate hash sets):
        // the same state always lists the same moves in the same order.
        let t = stats::start();
        out.sort_by_cached_key(|m| format!("{:?}|{:?}", m.step, m.action));
        stats::stop(t, || "sort".into());
        stats::stop(t_all, || "total".into());
        out
    }

    /// [`BaseGame::g1867_legal_moves_with`] through the reference paths
    /// (the generator before the 2026-11-13 speed-ups): the differential
    /// test's oracle. Same moves, same order — or a bug in the fast paths.
    pub fn g1867_legal_moves_reference(&mut self, policy: CrossPlayerPolicy) -> Vec<Move> {
        reference::with(true, || self.g1867_legal_moves_with(policy))
    }

    /// A move list as one comparable string per move (step, action, price
    /// range, certificate ids — everything a move carries but the internal
    /// `validated` flag), in list order.
    pub fn g1867_moves_strings(moves: &[Move]) -> Vec<String> {
        moves.iter().map(|m| format!("{:?}|{:?}|{:?}|{:?}", m.step, m.action, m.price, m.shares)).collect()
    }

    /// Every candidate before the apply filter, with its error (debug).
    pub fn g1867_candidates_debug(&mut self) -> Vec<(String, Option<String>)> {
        self.g1867_candidates()
            .into_iter()
            .map(|m| {
                let mut g = self.clone_for_search();
                let e = g.process_action_internal(&m.action).err().map(|e| e.to_string());
                (format!("{:?} {:?}", m.step, m.action), e)
            })
            .collect()
    }

    fn g1867_candidates(&mut self) -> Vec<Move> {
        let t0 = std::time::Instant::now();
        let t = stats::start();
        let offers = self.step_offers();
        stats::stop(t, || "offers".into());
        stats::add(&stats::OFFERS_NS, t0.elapsed().as_nanos() as u64);
        let t0 = std::time::Instant::now();
        let out = self.g1867_candidates_for(offers);
        stats::add(&stats::GEN_NS, t0.elapsed().as_nanos() as u64);
        out
    }

    fn g1867_candidates_for(&mut self, offers: Vec<crate::steps::StepOffer>) -> Vec<Move> {
        let mut out = Vec::new();
        for o in offers {
            for t in &o.types {
                let t0 = stats::start();
                out.extend(self.g1867_gen(o.kind, &o.entity, t));
                stats::stop(t0, || format!("gen:{:?}/{t}", o.kind));
            }
        }
        out
    }

    /// Whether `policy` excludes `m`: a purchase whose seller is not the
    /// buying corporation's president (a train from a corporation another
    /// player presides; a private another player owns), or (`!policy.
    /// minors`) a Merge-step add of a minor another player owns — so every
    /// merger self-play makes is of ONE player's minors (the user's ruling
    /// of 2026-10-18: ~99% of human mergers are). Bank sellers (the depot)
    /// are never excluded; a missing president or owner counts as another
    /// player. Naming the merged major (an unfloated major, owned by
    /// nobody) and conversions are never excluded.
    pub(crate) fn g1867_cross_player_excluded(&self, m: &Move, policy: CrossPlayerPolicy) -> bool {
        let pres = |c: &str| self.corp_idx.get(c).and_then(|&i| self.corporations[i].president_id());
        let other = |buyer: &str, seller: Option<u32>| {
            let b = pres(buyer);
            b.is_none() || seller != b
        };
        // A minor's owning player (Some(None): a minor without one).
        let minor_owner = |c: &str| {
            self.corp_idx
                .get(c)
                .map(|&i| &self.corporations[i])
                .filter(|c| c.corp_type == CorpType::Minor)
                .map(|c| c.owner_id.player_id())
        };
        match &m.action {
            Action::BuyTrain { entity_id, from, .. } if !policy.trains && from != "depot" => {
                other(entity_id, pres(from))
            }
            Action::BuyCompany { entity_id, company_sym, .. } if !policy.companies => {
                let seller = self.company_idx.get(company_sym.as_str()).and_then(|&i| self.companies[i].owner.player_id());
                other(entity_id, seller)
            }
            // The acting minor is the merge's initiator (the round's current
            // entity); each add must be of a minor its owner also owns.
            Action::Merge { entity_id, corporation_sym } if !policy.minors => match minor_owner(corporation_sym) {
                Some(target) => {
                    let initiator = minor_owner(entity_id).flatten();
                    initiator.is_none() || target != initiator
                }
                None => false,
            },
            _ => false,
        }
    }

    /// Keep the candidates that apply (on a clone); dedup.
    fn g1867_validate(&mut self, cands: Vec<Move>) -> Vec<Move> {
        let t0 = std::time::Instant::now();
        let out = if reference::on() { self.g1867_validate_reference(cands) } else { self.g1867_validate_inner(cands) };
        stats::add(&stats::VALIDATE_NS, t0.elapsed().as_nanos() as u64);
        out
    }

    /// The candidates in order, each kept unless an equal action was
    /// already kept (the first kept wins); a `validated` one as is, any
    /// other only if it applies on a clone (and, in the Merge step, the
    /// merge set can still finish) — or, for a member of a validation class
    /// (`g67_validation_class`) whose class is already known to pass, if its
    /// own checks pass.
    fn g1867_validate_inner(&mut self, cands: Vec<Move>) -> Vec<Move> {
        stats::add(&stats::CANDS, cands.len() as u64);
        let mut out: Vec<Move> = Vec::new();
        // Keys of the KEPT moves (a rejected candidate's duplicate is
        // checked again, as the reference does).
        let mut kept: std::collections::HashSet<String> = std::collections::HashSet::new();
        // Per class: do its shared conditions hold (None: not known yet)?
        let mut class_ok: [Option<bool>; ValidationClass::COUNT] = [None; ValidationClass::COUNT];
        let mut ctx = ClassCtx::default();
        for m in cands {
            let key = format!("{:?}", m.action);
            if kept.contains(&key) {
                continue;
            }
            if m.validated {
                kept.insert(key);
                out.push(m);
                continue;
            }
            let class = self.g67_validation_class(&m, &mut ctx);
            let ok = match class {
                // An own check fails: the handler refuses it whatever the
                // shared part (each own check is one of its refusals).
                Some((c, Some(false))) => {
                    stats::count(|| format!("validate:class_own_fails:{c:?}"));
                    false
                }
                Some((c, Some(true))) if class_ok[c as usize].is_some() => {
                    stats::count(|| format!("validate:class:{c:?}"));
                    class_ok[c as usize] == Some(true)
                }
                _ => {
                    let t = stats::start();
                    let mut g = self.g1867_validation_clone();
                    stats::stop(t, || "validate:clone".into());
                    let t = stats::start();
                    let ok = g.process_action_internal(&m.action).is_ok();
                    stats::stop(t, || format!("validate:process:{:?}/{}", m.step, m.action.action_type()));
                    let t = stats::start();
                    let ok = ok && (m.step != StepKind::Merge || g.g67_merge_can_finish(0));
                    if m.step == StepKind::Merge {
                        stats::stop(t, || "validate:merge_can_finish".into());
                    }
                    // valid = shared ∧ own: a member whose own checks pass
                    // tells whether the shared conditions hold.
                    if let Some((c, Some(true))) = class {
                        class_ok[c as usize] = Some(ok);
                    }
                    ok
                }
            };
            if ok {
                kept.insert(key);
                out.push(m);
            } else {
                stats::add(&stats::REJECTED, 1);
            }
        }
        out
    }

    /// The validation class of candidate `m` and its OWN checks: for the
    /// members of a class, as the generator builds them, the process
    /// succeeds iff the class's SHARED conditions hold (the same for every
    /// member: they read only the state and the action's type and entity)
    /// and the member's own checks pass. `(class, None)`: own checks not
    /// decidable here (the member is applied on a clone). The shared part
    /// of every class: the dispatch gate (it reads the action's type and
    /// `action_step_entity`, one entity per class); `process_action_internal`'s
    /// hooks (trainless choices take only `pass`/`choose`, company
    /// exchanges a company's `buy_shares`, company abilities a company
    /// entity: none of these members); and nothing after a handler returns
    /// an error (`skip_steps`, `stock_after_process`, `stock_next_entity`,
    /// `check_game_end`, the round transitions return no `Result`).
    ///
    /// * `Par` (`g67_gen_pars`; own checks: none): `process_stock_action`
    ///   rejects every par alike while a home token is pending or a minor
    ///   auction is live; `stock_process_par` checks the turn,
    ///   `bought_this_turn` and the cert limit (player-level), and per member:
    ///   a known major (`corp_type` Major — the generator's filter),
    ///   startable in this phase, not parred, not closed (its filters, the
    ///   same predicates), a par cell of the price (the generator takes the
    ///   price from that same cell list) and `price × president units ≤
    ///   cash` (its filter; `player_cash` = the player's cash).
    /// * `MinorOpeningBid` (no auction live; `g67_gen_minor_bids`, all at
    ///   $100; own checks: none): `process_corporation_bid` checks the
    ///   pending home tokens, the turn, `bought/parred_this_turn` and the
    ///   `minors_cannot_start` event (player- or game-level), and per
    ///   member: a minor, not started, not closed, startable (the
    ///   generator's filters) and `validate_corp_bid` at $100 (the minimum,
    ///   on the grid, ≤ `stock_bid_max` — the generator's `hi ≥
    ///   MINOR_MIN_BID`); `stock_bid_resolve` has no error path.
    /// * `StockSell` (`g67_gen_player_sales`): shared — `process_stock_action`'s
    ///   pending-home-token and live-auction refusals, `validate_stock_turn`;
    ///   own — `validate_sell_bundle` (a `&self` check the handler runs on
    ///   the unchanged state: ownership, sale timing, the pool cap, dump
    ///   legality), the corporation known and parred. The handler has no
    ///   refusal after it starts moving certificates.
    /// * `TokenPlace` (the Token step's `place_token`s): shared —
    ///   `process_operating_action`'s crowded-corps refusal, the OO
    ///   re-placement branch (pending tokens: no Token-step offer then),
    ///   `or_process_place_token`'s step, operator, one-token-a-turn, an
    ///   unplaced token and its base price; own — [`Self::g67_token_place_own_checks`].
    /// * `StockHomeToken` (a stock round's HomeToken `place_token`s): shared —
    ///   `process_stock_home_token`'s pending token and its corporation;
    ///   own — the hex, a slot (`token_slot_for`), the city, the slot empty.
    fn g67_validation_class(&mut self, m: &Move, ctx: &mut ClassCtx) -> Option<(ValidationClass, Option<bool>)> {
        match (&m.action, &self.round) {
            (Action::Par { .. }, Round::Stock(_)) if m.step == StepKind::BuySellParShares => {
                Some((ValidationClass::Par, Some(true)))
            }
            (Action::CorporationBid { price, .. }, Round::Stock(s))
                if m.step == StepKind::BuySellParShares && s.bid_auction.is_none() && *price == MINOR_MIN_BID =>
            {
                Some((ValidationClass::MinorOpeningBid, Some(true)))
            }
            (Action::SellShares { entity_id, corporation_sym, percent, share_indices, .. }, Round::Stock(_))
                if m.step == StepKind::BuySellParShares =>
            {
                let own = entity_id.parse::<u32>().is_ok_and(|pid| {
                    self.validate_sell_bundle(pid, corporation_sym, *percent, share_indices, None).is_ok()
                }) && self.corp_idx.get(corporation_sym.as_str()).is_some_and(|&ci| self.corporations[ci].share_price.is_some());
                Some((ValidationClass::StockSell, Some(own)))
            }
            (Action::PlaceToken { entity_id, hex_id, city_index }, Round::Operating(_)) if m.step == StepKind::Token => {
                let own = self.g67_token_place_own_checks(entity_id, hex_id, *city_index as usize, ctx);
                Some((ValidationClass::TokenPlace, own))
            }
            (Action::PlaceToken { entity_id, hex_id, city_index }, Round::Stock(_)) if m.step == StepKind::HomeToken => {
                let city = *city_index as usize;
                let own = self.hex_idx.get(hex_id.as_str()).is_some_and(|&hi| {
                    self.token_slot_for(hex_id, city, entity_id).is_some_and(|slot| {
                        self.hexes[hi].tile.cities.get(city).is_some_and(|c| c.tokens.get(slot).is_some_and(|t| t.is_none()))
                    })
                });
                Some((ValidationClass::StockHomeToken, Some(own)))
            }
            _ => None,
        }
    }

    /// A Token-step `place_token`'s own checks — `or_process_place_token`'s
    /// per-city ones, in its order, on the state before the action (the
    /// handler mutates nothing before them): the city is among the
    /// corporation's tokenable cities (`tokenable_cities_for` on the
    /// replay graph computed afresh — the graph the handler reads), it has
    /// a slot for the corporation (`token_slot_for`), the corporation can
    /// pay the distance price (`token_price_by_distance`: the next token's
    /// price × the distance from its nearest placed token, 1 if none), the
    /// city exists and that slot is empty. `None` where the handler would
    /// take `process_operating_action`'s home-token branch instead (an
    /// unplaced first token on the corporation's fixed home hex; 1867
    /// corporations have none).
    fn g67_token_place_own_checks(&mut self, corp: &str, hex_id: &str, city: usize, ctx: &mut ClassCtx) -> Option<bool> {
        let &ci = self.corp_idx.get(corp)?;
        let c = &self.corporations[ci];
        let needs_home = !c.tokens.is_empty() && !c.tokens[0].used;
        let fixed_home = self.title_def().corporations().iter().find(|cd| cd.sym == corp).map(|cd| cd.home_hex);
        if needs_home && fixed_home.map_or(true, |h| h == hex_id) {
            return None;
        }
        let Some(&hi) = self.hex_idx.get(hex_id) else { return Some(false) };
        if ctx.tokenable.as_ref().map_or(true, |(c, _)| c != corp) {
            // A pristine clone's cache is empty: its graph is computed
            // afresh (or taken from the memo of this very map).
            let mut g = self.g1867_validation_clone();
            let v = g.tokenable_cities_for(corp);
            ctx.tokenable = Some((corp.to_string(), v));
        }
        if !ctx.tokenable.as_ref().is_some_and(|(_, v)| v.iter().any(|(h, x)| h == hex_id && *x == city)) {
            return Some(false);
        }
        let Some(slot) = self.token_slot_for(hex_id, city, corp) else { return Some(false) };
        let Some(ti) = self.corporations[ci].next_token_index() else { return Some(false) };
        let base = self.corporations[ci].tokens[ti].price;
        let cost = if self.title_def().token_price_by_distance() {
            base * self.min_token_hex_distance(ci, hex_id).unwrap_or(1)
        } else {
            base
        };
        if self.corporations[ci].cash < cost {
            return Some(false);
        }
        Some(self.hexes[hi].tile.cities.get(city).is_some_and(|x| x.tokens.get(slot).is_some_and(|t| t.is_none())))
    }

    /// `clone_for_search` without the JSON action log, for applying a
    /// candidate to see whether it is legal: processing never reads the
    /// log (only `process_action` and the RL apply path append to it), and
    /// copying it was most of a late-game clone.
    fn g1867_validation_clone(&mut self) -> BaseGame {
        let log = std::mem::take(&mut self.action_log);
        let g = self.clone_for_search();
        self.action_log = log;
        g
    }

    /// The validation before 2026-11-13 (the differential test's
    /// reference): a full `clone_for_search` per candidate, dedup by a scan
    /// of the kept moves.
    fn g1867_validate_reference(&self, cands: Vec<Move>) -> Vec<Move> {
        stats::add(&stats::CANDS, cands.len() as u64);
        let mut out: Vec<Move> = Vec::new();
        for m in cands {
            let key = format!("{:?}", m.action);
            if out.iter().any(|o| format!("{:?}", o.action) == key) {
                continue;
            }
            if m.validated {
                out.push(m);
                continue;
            }
            let mut g = self.clone_for_search();
            if g.process_action_internal(&m.action).is_ok() && (m.step != StepKind::Merge || g.g67_merge_can_finish(0)) {
                out.push(m);
            } else {
                stats::add(&stats::REJECTED, 1);
            }
        }
        out
    }

    /// T6 (states humans never reach): after a Merge-step move, can the
    /// open merge set still finish? Ruby `finish_merge_to_major` raises
    /// "Merge impossible, no player can become president" for every major
    /// when no owner can take the 20% presidency, leaving the players to
    /// undo — so the move that commits the set to that is not offered: an
    /// add (or the "done adding" pass) is legal only if SOME completion
    /// (finish now, or after further adds) names a major successfully; a
    /// `convert` only if some unfloated major can take the conversion
    /// (Ruby offers `convert` whenever the price is in range, even with
    /// every major started). The search walks the rules' adds whatever the
    /// cross-player policy; under the single-owner merge policy that is
    /// still exact: a set of one player's minors always has an owner with
    /// two of them (Ruby's presidency check passes), so only naming the
    /// major can fail, and that does not depend on the set.
    fn g67_merge_can_finish(&mut self, depth: usize) -> bool {
        let Round::Merger(s) = &self.round else { return true };
        let s = s.clone();
        if (s.merging.is_empty() && !s.converting) || s.converted.is_some() {
            return true;
        }
        let Some(cur) = s.current_entity_sym().map(str::to_string) else { return true };
        let try_move = |g: &BaseGame, a: Action| -> Option<BaseGame> {
            let mut c = g.clone_for_search();
            c.process_action_internal(&a).ok().map(|_| c)
        };
        // Naming the major (a conversion's or a finished set's): some
        // unfloated major must take it.
        if s.merge_major || s.converting {
            return self
                .corporations
                .iter()
                .filter(|t| t.corp_type == CorpType::Major && !t.floated && !t.closed)
                .any(|t| try_move(self, Action::Merge { entity_id: cur.clone(), corporation_sym: t.sym.clone() }).is_some());
        }
        if let Some(mut g) = try_move(self, Action::Pass { entity_id: cur.clone() }) {
            if g.g67_merge_can_finish(depth + 1) {
                return true;
            }
        }
        if depth >= 10 {
            return false;
        }
        for cand in self.merger_mergeable_candidates(&s, &cur) {
            if let Some(mut g) = try_move(self, Action::Merge { entity_id: cur.clone(), corporation_sym: cand }) {
                if g.g67_merge_can_finish(depth + 1) {
                    return true;
                }
            }
        }
        false
    }

    // =====================================================================
    // Per-step generators
    // =====================================================================

    fn g1867_gen(&mut self, kind: StepKind, ent: &StepEntity, t: &str) -> Vec<Move> {
        use StepKind as K;
        match (kind, ent, t) {
            (_, StepEntity::Corp(c), "pass") => vec![Move::plain(kind, Action::Pass { entity_id: c.clone() })],
            (_, StepEntity::Player(p), "pass") => {
                vec![Move::plain(kind, Action::Pass { entity_id: p.to_string() })]
            }
            (K::SingleItemAuction, StepEntity::Player(p), "bid") => self.g67_gen_auction_bid(*p),
            (K::MajorTrainless, StepEntity::Corp(c), "choose") => vec![Move::plain(
                kind,
                Action::Choose { entity_id: c.clone(), choice: "nationalize".into() },
            )],
            (K::DiscardTrain, StepEntity::Corp(c), "discard_train") => self.g67_gen_discards(c),
            (K::HomeToken, StepEntity::Corp(c), "place_token") => self.g67_gen_home_tokens(kind, c),
            (K::BuySellParShares, StepEntity::Player(p), _) => self.g67_gen_bsps(*p, t),
            (K::BuyCompany | K::BuyCompanyPreloan, StepEntity::Corp(c), "buy_company") => {
                self.g67_gen_buy_company(kind, c)
            }
            (K::RedeemShares, StepEntity::Corp(c), "buy_shares") => self.g67_gen_redeem(c),
            (K::Track, StepEntity::Corp(c), "lay_tile") => self.g67_gen_lays(c),
            (K::Token, StepEntity::Corp(c), "place_token") => self.g67_gen_tokens(c),
            (K::Route, StepEntity::Corp(c), "run_routes") => self.g67_gen_routes(c),
            (K::Dividend, StepEntity::Corp(c), "dividend") => [DividendKind::Payout, DividendKind::Half, DividendKind::Withhold]
                .into_iter()
                .map(|kind_| Move::plain(kind, Action::Dividend { entity_id: c.clone(), kind: kind_ }))
                .collect(),
            (K::BuyTrain, StepEntity::Corp(c), "buy_train") => self.g67_gen_buy_train(c),
            (K::Merge, StepEntity::Corp(c), "merge") => self.g67_gen_merge(c),
            (K::Merge, StepEntity::Corp(c), "convert") => {
                vec![Move::plain(kind, Action::Convert { entity_id: c.clone() })]
            }
            (K::PostMergerShares, StepEntity::Player(p), "buy_shares") => self.g67_gen_post_merger_buy(*p),
            (K::ReduceTokens, StepEntity::Corp(c), "remove_token") => self.g67_gen_remove_tokens(c),
            _ => Vec::new(),
        }
    }

    // -- auction ------------------------------------------------------------

    /// SingleItemAuction bid: the lot up for auction, from its minimum on
    /// the $5 grid up to the player's cash; in dutch mode only the asking
    /// price (any bid buys outright, so a higher one only loses money).
    fn g67_gen_auction_bid(&self, pid: u32) -> Vec<Move> {
        let Round::Auction(s) = &self.round else { return Vec::new() };
        let Some(ci) = s.auctioning else { return Vec::new() };
        let min = self.sa_min_bid(s, ci);
        let cash = self.player_cash(pid);
        if cash < min {
            return Vec::new();
        }
        let dutch = s.single_item.as_ref().map_or(false, |si| si.dutch_mode);
        let hi = if dutch { min } else { cash };
        let a = Action::Bid { entity_id: pid.to_string(), company_sym: self.companies[ci].sym.clone(), price: min };
        vec![Move::ranged(StepKind::SingleItemAuction, a, min, hi, BID_STEP)]
    }

    // -- stock round --------------------------------------------------------

    fn g67_gen_bsps(&mut self, pid: u32, t: &str) -> Vec<Move> {
        let k = StepKind::BuySellParShares;
        let Round::Stock(s) = &self.round else { return Vec::new() };
        if let Some(auc) = &s.bid_auction {
            if t != "bid" {
                return Vec::new();
            }
            let high = auc.bids.iter().map(|b| b.price).max().unwrap_or(0);
            let lo = high + BID_STEP;
            let hi = self.stock_bid_max(pid);
            if hi < lo {
                return Vec::new();
            }
            let a = Action::CorporationBid {
                entity_id: pid.to_string(),
                corporation_sym: auc.corporation_sym.clone(),
                price: lo,
            };
            return vec![Move::ranged(k, a, lo, hi, BID_STEP)];
        }
        match t {
            "sell_shares" => self.g67_gen_player_sales(pid),
            "buy_shares" => self.g67_gen_player_buys(pid),
            "par" => self.g67_gen_pars(pid),
            "bid" => self.g67_gen_minor_bids(pid),
            _ => Vec::new(),
        }
    }

    /// Ruby bundle order for a holder's certs of a corp: `[president?,
    /// percent]` ascending (base.rb all_bundles_for_corporation).
    fn g67_sorted_certs(&self, ci: usize, owner: &EntityId) -> Vec<usize> {
        let corp = &self.corporations[ci];
        let mut v: Vec<usize> = (0..corp.shares.len()).filter(|&i| corp.shares[i].owner == *owner).collect();
        v.sort_by_key(|&i| (corp.shares[i].president, corp.shares[i].percent, i));
        v
    }

    fn g67_cert_id(&self, ci: usize, idx: usize) -> String {
        format!("{}_{}", self.corporations[ci].sym, idx)
    }

    fn g67_gen_player_sales(&self, pid: u32) -> Vec<Move> {
        let me = EntityId::player(pid);
        let mut out = Vec::new();
        for b in self.sellable_bundles_detailed(pid) {
            let Some(&ci) = self.corp_idx.get(b.corp_sym.as_str()) else { continue };
            let mut idx = Vec::new();
            let mut acc = 0u8;
            for i in self.g67_sorted_certs(ci, &me) {
                if acc >= b.percent {
                    break;
                }
                acc += self.corporations[ci].shares[i].percent;
                idx.push(i);
            }
            let mut m = Move::plain(
                StepKind::BuySellParShares,
                Action::SellShares {
                    entity_id: pid.to_string(),
                    corporation_sym: b.corp_sym.clone(),
                    shares: Vec::new(),
                    percent: b.percent,
                    share_indices: idx.clone(),
                },
            );
            m.shares = idx.iter().map(|&i| self.g67_cert_id(ci, i)).collect();
            out.push(m);
        }
        out
    }

    fn g67_gen_player_buys(&self, pid: u32) -> Vec<Move> {
        let mut out = Vec::new();
        for (sym, source, idx, _price) in self.buyable_shares_for_factored(pid) {
            let Some(&ci) = self.corp_idx.get(sym.as_str()) else { continue };
            let percent = self.corporations[ci].shares[idx].percent;
            let mut m = Move::plain(
                StepKind::BuySellParShares,
                Action::BuyShares {
                    entity_id: pid.to_string(),
                    corporation_sym: sym.clone(),
                    shares: Vec::new(),
                    percent,
                    source,
                    share_indices: vec![idx],
                },
            );
            m.shares = vec![self.g67_cert_id(ci, idx)];
            out.push(m);
        }
        out
    }

    /// Major pars (G1867 BuySellParShares get_par_prices: par_2 + par cells
    /// with 2×price ≤ cash, phase ≥ 4, a home-token location must exist).
    fn g67_gen_pars(&mut self, pid: u32) -> Vec<Move> {
        let cash = self.player_cash(pid);
        let title = self.title_def();
        let majors: Vec<usize> = (0..self.corporations.len())
            .filter(|&ci| {
                let c = &self.corporations[ci];
                c.corp_type == CorpType::Major
                    && c.ipo_price.is_none()
                    && !c.closed
                    && title.corporation_startable(&c.sym, &self.phase.name)
            })
            .collect();
        let cells = self.stock_market.share_prices_with_types(title.par_price_types(CorpType::Major));
        let mut out = Vec::new();
        for ci in majors {
            if self.g1867_home_token_locations(ci).is_empty() {
                continue;
            }
            let units = self.corporations[ci].president_share_units();
            for sp in &cells {
                if sp.price * units > cash {
                    continue;
                }
                out.push(Move::plain(
                    StepKind::BuySellParShares,
                    Action::Par {
                        entity_id: pid.to_string(),
                        corporation_sym: self.corporations[ci].sym.clone(),
                        share_price: sp.price,
                    },
                ));
            }
        }
        out
    }

    /// Opening bids on unstarted minors (ViaBid `can_bid_any?`): $100 on
    /// the $5 grid up to the player's cash (0 at the cert limit). A minor
    /// with no home-token location is not offered: Ruby accepts the bid
    /// and then waits forever on a HomeToken step with no legal city (a
    /// dead end only an undo leaves; every city on the map full).
    fn g67_gen_minor_bids(&mut self, pid: u32) -> Vec<Move> {
        let Round::Stock(s) = &self.round else { return Vec::new() };
        if !self.stock_can_bid_minor(s, pid) {
            return Vec::new();
        }
        let hi = self.stock_bid_max(pid);
        if hi < MINOR_MIN_BID {
            return Vec::new();
        }
        let title = self.title_def();
        let minors: Vec<usize> = (0..self.corporations.len())
            .filter(|&ci| {
                let c = &self.corporations[ci];
                c.corp_type == CorpType::Minor
                    && c.ipo_price.is_none()
                    && !c.closed
                    && title.corporation_startable(&c.sym, &self.phase.name)
            })
            .collect();
        let minors: Vec<usize> =
            minors.into_iter().filter(|&ci| !self.g1867_home_token_locations(ci).is_empty()).collect();
        minors
            .into_iter()
            .map(|ci| &self.corporations[ci])
            .map(|c| {
                Move::ranged(
                    StepKind::BuySellParShares,
                    Action::CorporationBid {
                        entity_id: pid.to_string(),
                        corporation_sym: c.sym.clone(),
                        price: MINOR_MIN_BID,
                    },
                    MINOR_MIN_BID,
                    hi,
                    BID_STEP,
                )
            })
            .collect()
    }

    /// The pending home token's cities: Ruby `home_token_locations`
    /// (the hexes) × the cities of each that the corporation can token.
    fn g67_gen_home_tokens(&mut self, kind: StepKind, c: &str) -> Vec<Move> {
        let Some(&ci) = self.corp_idx.get(c) else { return Vec::new() };
        let hexes: Vec<usize> = match &self.round {
            Round::Stock(_) => self.g1867_home_token_locations(ci),
            // An operating-round pending token (an OO-style re-placement):
            // the hex it was displaced from.
            Round::Operating(s) => s
                .pending_tokens
                .iter()
                .filter(|(sym, _, _)| sym == c)
                .filter_map(|(_, _, h)| self.hex_idx.get(h.as_str()).copied())
                .collect(),
            _ => Vec::new(),
        };
        let mut out = Vec::new();
        for hi in hexes {
            for city in 0..self.hexes[hi].tile.cities.len() {
                if !self.g1867_city_tokenable(hi, city, c) {
                    continue;
                }
                out.push(Move::plain(
                    kind,
                    Action::PlaceToken { entity_id: c.to_string(), hex_id: self.hexes[hi].id.clone(), city_index: city as u8 },
                ));
            }
        }
        out
    }

    /// Ruby G1867 `home_token_locations(corporation)`: minors — a city with
    /// a free tokenable slot and no tokens; majors — a hex with a tokenable
    /// city and no NORMAL token anywhere on it, preferring hexes whose
    /// track reaches no other node (`unconnected_hexes`) when any exist.
    pub(crate) fn g1867_home_token_locations(&self, ci: usize) -> Vec<usize> {
        let sym = self.corporations[ci].sym.clone();
        let minor = self.corporations[ci].corp_type == CorpType::Minor;
        let mut open = Vec::new();
        for hi in 0..self.hexes.len() {
            let cities = &self.hexes[hi].tile.cities;
            if cities.is_empty() {
                continue;
            }
            let ok = if minor {
                (0..cities.len()).any(|c| {
                    cities[c].tokens.iter().all(|t| t.is_none()) && self.g1867_city_tokenable(hi, c, &sym)
                })
            } else {
                (0..cities.len()).any(|c| self.g1867_city_tokenable(hi, c, &sym))
                    && cities.iter().all(|c| {
                        c.tokens.iter().flatten().all(|t| t.token_type == "neutral")
                    })
            };
            if ok {
                open.push(hi);
            }
        }
        if minor {
            return open;
        }
        let unconnected: Vec<usize> = open.iter().copied().filter(|&hi| !self.g1867_city_track_reaches_node(hi)).collect();
        if unconnected.is_empty() {
            open
        } else {
            unconnected
        }
    }

    /// Ruby `City#tokenable?(corporation, free: true)` for a corporation's
    /// (next) token: a free un-reserved slot (or its own reservation), no
    /// token of the corporation elsewhere on the tile, not blocked by a
    /// reservation.
    pub(crate) fn g1867_city_tokenable(&self, hi: usize, city: usize, corp: &str) -> bool {
        let hex = &self.hexes[hi];
        let tile = &hex.tile;
        if tile.cities.iter().any(|c| c.tokens.iter().flatten().any(|t| t.corporation_id == corp)) {
            return false;
        }
        let Some(slot) = self.token_slot_for(&hex.id, city, corp) else { return false };
        tile.cities[city].tokens.get(slot).map_or(false, |t| t.is_none())
    }

    // -- operating round ----------------------------------------------------

    fn g67_gen_discards(&self, c: &str) -> Vec<Move> {
        let Some(&ci) = self.corp_idx.get(c) else { return Vec::new() };
        let mut out = Vec::new();
        let mut names: Vec<&str> = Vec::new();
        for t in &self.corporations[ci].trains {
            if names.contains(&t.name.as_str()) {
                continue;
            }
            names.push(&t.name);
            out.push(Move::plain(
                StepKind::DiscardTrain,
                Action::DiscardTrain { entity_id: c.to_string(), train_name: t.id.clone() },
            ));
        }
        out
    }

    /// Ruby BuyCompany: privates owned by a player (`purchasable_companies`),
    /// in the phases with `can_buy_companies`, for $1..face
    /// (CompanyPriceUpToFace), within the corporation's cash.
    fn g67_gen_buy_company(&self, kind: StepKind, c: &str) -> Vec<Move> {
        if !self.phase_has_status("can_buy_companies") {
            return Vec::new();
        }
        let Some(&ci) = self.corp_idx.get(c) else { return Vec::new() };
        let cash = self.corporations[ci].cash;
        let title = self.title_def();
        let mut out = Vec::new();
        for co in &self.companies {
            if co.closed || co.no_buy || !co.owner.is_player() {
                continue;
            }
            let (lo, hi) = title.company_buy_price_range(co.value);
            let hi = hi.min(cash);
            if hi < lo {
                continue;
            }
            out.push(Move::ranged(
                kind,
                Action::BuyCompany { entity_id: c.to_string(), company_sym: co.sym.clone(), price: lo },
                lo,
                hi,
                1,
            ));
        }
        out
    }

    /// Ruby G1867 RedeemShares (`redeemable_shares`): bundles of the
    /// corporation's own pool shares, cumulative in Ruby's bundle order,
    /// affordable at market price.
    fn g67_gen_redeem(&self, c: &str) -> Vec<Move> {
        let Some(&ci) = self.corp_idx.get(c) else { return Vec::new() };
        let corp = &self.corporations[ci];
        let Some(price) = corp.share_price.as_ref().map(|sp| sp.price) else { return Vec::new() };
        let unit = corp.share_unit_percent.max(1) as i32;
        let pool: Vec<usize> = self
            .g67_sorted_certs(ci, &EntityId::market())
            .into_iter()
            .filter(|&i| corp.shares[i].percent > 0)
            .collect();
        let mut out = Vec::new();
        let (mut pct, mut cost) = (0u8, 0i32);
        for (n, &i) in pool.iter().enumerate() {
            pct += corp.shares[i].percent;
            cost += price * corp.shares[i].percent as i32 / unit;
            if cost > corp.cash {
                break;
            }
            let idx: Vec<usize> = pool[..=n].to_vec();
            let mut m = Move::plain(
                StepKind::RedeemShares,
                Action::BuyShares {
                    entity_id: c.to_string(),
                    corporation_sym: c.to_string(),
                    shares: Vec::new(),
                    percent: pct,
                    source: "market".into(),
                    share_indices: idx.clone(),
                },
            );
            m.shares = idx.iter().map(|&i| self.g67_cert_id(ci, i)).collect();
            out.push(m);
        }
        out
    }

    /// Token step: every city the corporation can token (connected, free
    /// slot, not reserved, not already in the hex); the distance price and
    /// cash are checked by the process on the clone.
    fn g67_gen_tokens(&mut self, c: &str) -> Vec<Move> {
        // Ruby Tokener#check_connected: the city must be one of
        // `graph.connected_nodes(entity)` (the Ruby walk — stricter than
        // the replay graph the process checks).
        let connected: std::collections::HashSet<(usize, usize)> = self
            .g1867_graph(c)
            .connected_nodes
            .into_iter()
            .filter(|n| n.kind == super::graph::NodeKind::City)
            .map(|n| (n.hex, n.index))
            .collect();
        self.tokenable_cities_for(c)
            .into_iter()
            .filter(|(h, city)| self.hex_idx.get(h.as_str()).map_or(false, |&hi| connected.contains(&(hi, *city))))
            .map(|(hex_id, city)| {
                Move::plain(StepKind::Token, Action::PlaceToken { entity_id: c.to_string(), hex_id, city_index: city as u8 })
            })
            .collect()
    }

    /// Route step: the auto-router's optimum (one move).
    fn g67_gen_routes(&mut self, c: &str) -> Vec<Move> {
        let (routes, _) = self.g1867_auto_routes(c);
        vec![Move::plain(
            StepKind::Route,
            Action::RunRoutes { entity_id: c.to_string(), routes, extra_revenue: 0 },
        )]
    }

    /// Ruby G1867::Step::BuyTrain + Train: depot trains (the upcoming head,
    /// phase-available later trains, the discard pool — one per name) at
    /// face; trade-ins of an owned train against a discounted phase-8 train
    /// (only in phase 8 — `discountable_trains_allowed?`, so never the
    /// first 8-train); another corporation's train for $1 up to the cash
    /// (loans fund only depot trains a corporation must buy).
    fn g67_gen_buy_train(&mut self, c: &str) -> Vec<Move> {
        let Some(&ci) = self.corp_idx.get(c) else { return Vec::new() };
        let must = self.operating_must_buy_train(c);
        let cash = self.corporations[ci].cash;
        let power = if must { self.corp_buying_power_full(ci) } else { cash };
        let room = self.corporations[ci].trains.len() < self.corp_train_limit(ci);
        let k = StepKind::BuyTrain;
        let mut out = Vec::new();
        // depot_trains: head + phase-available upcoming + discarded, uniq by name.
        let mut depot: Vec<crate::entities::Train> = Vec::new();
        if let Some(h) = self.depot.trains.first() {
            depot.push(h.clone());
        }
        for t in &self.depot.trains {
            if self.phase_available(t.available_on.as_deref()) && !depot.iter().any(|d| d.name == t.name) {
                depot.push(t.clone());
            }
        }
        let mut discarded: Vec<crate::entities::Train> = Vec::new();
        for t in &self.depot.discarded {
            if !discarded.iter().any(|d| d.name == t.name) {
                discarded.push(t.clone());
            }
        }
        if room {
            for t in depot.iter().chain(discarded.iter()) {
                if t.price <= power {
                    out.push(Move::plain(
                        k,
                        Action::BuyTrain {
                            entity_id: c.to_string(),
                            train_name: t.id.clone(),
                            price: t.price,
                            from: "depot".into(),
                            variant: Some(t.name.clone()),
                            exchange: None,
                        },
                    ));
                }
            }
        }
        // Trade-ins (phase 8 only).
        if self.g1867_trade_in_allowed() {
            // One trade-in per owned train NAME (two 5s trade in alike).
            let mut owned: Vec<crate::entities::Train> = Vec::new();
            for t in &self.corporations[ci].trains {
                if !owned.iter().any(|o| o.name == t.name) {
                    owned.push(t.clone());
                }
            }
            for dt in depot.iter().chain(discarded.iter()) {
                for ot in &owned {
                    let Some(&(_, disc)) = dt.discount.iter().find(|(n, _)| *n == ot.name) else { continue };
                    let price = dt.price - disc;
                    // A 5+5E traded for a 2+2 "costs" $600 - $750: Ruby's
                    // discountable_trains_for lists it, then Spender#spend
                    // raises on the negative amount — never offered.
                    if price > cash || price <= 0 {
                        continue;
                    }
                    out.push(Move::plain(
                        k,
                        Action::BuyTrain {
                            entity_id: c.to_string(),
                            train_name: dt.id.clone(),
                            price,
                            from: "depot".into(),
                            variant: Some(dt.name.clone()),
                            exchange: Some(ot.id.clone()),
                        },
                    ));
                }
            }
        }
        // Other corporations' trains.
        if room && cash >= 1 && self.title_def().train_buy_from_other_players() {
            for (oi, other) in self.corporations.iter().enumerate() {
                if oi == ci || other.trains.is_empty() {
                    continue;
                }
                let mut names: Vec<&str> = Vec::new();
                for t in &other.trains {
                    if names.contains(&t.name.as_str()) {
                        continue;
                    }
                    names.push(&t.name);
                    out.push(Move::ranged(
                        k,
                        Action::BuyTrain {
                            entity_id: c.to_string(),
                            train_name: t.id.clone(),
                            price: 1,
                            from: other.sym.clone(),
                            variant: Some(t.name.clone()),
                            exchange: None,
                        },
                        1,
                        cash,
                        1,
                    ));
                }
            }
        }
        out
    }

    /// Ruby G1867 BuyTrain#discountable_trains_allowed?: phase 8.
    pub(crate) fn g1867_trade_in_allowed(&self) -> bool {
        self.title_def().trade_in_allowed(&self.phase.name)
    }

    // -- track ----------------------------------------------------------------

    /// Hexes blocked by an open company's `blocks_hexes` ability. Ruby
    /// `ability_blocking_hex` checks every open company whatever its owner
    /// (1867's hidden '3' — owned by nobody — blocks Granby M13 until the
    /// first 3-train closes it; its title data carries a placeholder
    /// owner_type, so the owner is ignored here).
    pub(crate) fn g1867_blocked_hexes(&self) -> std::collections::HashSet<&'static str> {
        let mut out = std::collections::HashSet::new();
        for co in &self.companies {
            if co.closed {
                continue;
            }
            if let Some((_, hexes)) = crate::abilities::blocks_hexes(&self.title, &co.sym) {
                out.extend(hexes.iter().copied());
            }
        }
        out
    }

    /// The first tile instance of `name` not on the map (Ruby keeps laid
    /// tiles out of `@tiles`; an upgraded-away tile comes back).
    fn g1867_tile_instance(&self, name: &str) -> String {
        let on_map: std::collections::HashSet<&str> = self.hexes.iter().map(|h| h.tile.id.as_str()).collect();
        (0..10_000u32)
            .map(|n| format!("{name}-{n}"))
            .find(|id| !on_map.contains(id.as_str()))
            .unwrap_or_else(|| format!("{name}-0"))
    }

    /// Track step lays (Ruby Track#available_hex → tracker_available_hex,
    /// Tracker#upgradeable_tiles: potential tiles × legal rotations, the
    /// max-exits filter, then check_track_restrictions! on the laid tile).
    fn g67_gen_lays(&mut self, c: &str) -> Vec<Move> {
        if reference::on() {
            self.g67_gen_lays_reference(c)
        } else {
            self.g67_gen_lays_fast(c)
        }
    }

    /// The Track generator before 2026-11-13 (the differential test's
    /// reference; [`BaseGame::g67_gen_lays_fast`] must give the same moves):
    /// Track step lays (Ruby Track#available_hex → tracker_available_hex,
    /// Tracker#upgradeable_tiles: potential tiles × legal rotations, the
    /// max-exits filter, then check_track_restrictions! on the laid tile).
    fn g67_gen_lays_reference(&mut self, c: &str) -> Vec<Move> {
        use crate::title::UpgradeAllowance;
        let Round::Operating(s) = &self.round else { return Vec::new() };
        let laid = s.num_laid_track as usize;
        let upgraded = s.upgraded_this_turn;
        let laid_hexes = s.laid_hexes.clone();
        let Some(slot) = self.title_def().tile_lays().get(laid).copied() else { return Vec::new() };
        let upgrade_ok = match slot.upgrade {
            UpgradeAllowance::Yes => true,
            UpgradeAllowance::No => false,
            UpgradeAllowance::NotIfUpgraded => !upgraded,
        };
        let Some(&ci) = self.corp_idx.get(c) else { return Vec::new() };
        let power = self.corp_buying_power_full(ci);
        let blocked = self.g1867_blocked_hexes();
        // Ruby `graph.connected_hexes(entity)` (the lay targets and the
        // edges each is reached by).
        let tg = std::time::Instant::now();
        let graph = self.g1867_graph(c);
        stats::add(&stats::LAY_GRAPH_NS, tg.elapsed().as_nanos() as u64);
        let targets = graph.connected_hexes.clone();
        let mut out = Vec::new();
        for (hi, edges) in targets {
            let hex_id = self.hexes[hi].id.clone();
            if blocked.contains(hex_id.as_str()) {
                continue;
            }
            let white = self.hexes[hi].tile.color == crate::tiles::TileColor::White;
            if (white && !slot.lay) || (!white && !upgrade_ok) {
                continue;
            }
            // tracker_available_hex (upgrades) / lay_tile_action (any lay):
            // the second lay never reuses a hex laid this turn.
            if slot.cannot_reuse_same_hex && laid_hexes.contains(&hex_id) {
                continue;
            }
            let ts = std::time::Instant::now();
            let nb = self.passable_exits_for(&hex_id);
            let mut with_rot: Vec<(String, Vec<u8>)> = Vec::new();
            for t in self.g1867_potential_tiles(hi) {
                let def = self.tile_catalog[&t].clone();
                let rots: Vec<u8> =
                    (0..6u8).filter(|&r| self.g1867_legal_rotation(hi, &def, r, &edges, &nb)).collect();
                if !rots.is_empty() {
                    with_rot.push((t, rots));
                }
            }
            let keep = self.g1867_max_exits(hi, with_rot.iter().map(|(t, _)| t.clone()).collect());
            stats::add(&stats::LAY_SEARCH_NS, ts.elapsed().as_nanos() as u64);
            let tc = std::time::Instant::now();
            let terrain: i32 = self.hexes[hi].tile.upgrades.iter().map(|u| u.cost).sum();
            for (t, rots) in with_rot {
                if !keep.contains(&t) {
                    continue;
                }
                let inst = self.g1867_tile_instance(&t);
                for r in rots {
                    let (border, _) = self.tile_border_cost(&hex_id, &t, r);
                    if terrain + slot.cost + border > power {
                        continue;
                    }
                    let mut m = Move::plain(
                        StepKind::Track,
                        Action::LayTile { entity_id: c.to_string(), hex_id: hex_id.clone(), tile_id: inst.clone(), rotation: r },
                    );
                    // A lay on untiled ground always uses new track, and
                    // the generator already checked what the process
                    // checks (slot rules, cost, supply): no clone needed.
                    if self.hexes[hi].tile.paths.is_empty() {
                        m.validated = true;
                        out.push(m);
                        continue;
                    }
                    // check_track_restrictions! on the tile as laid: the
                    // walk after the lay (the hex's tile replaced, tokens
                    // moved by Hex#lay's city map). The generator checked
                    // what the process checks (slot rules, cost, supply,
                    // rotation), so the generic filter skips the move.
                    if !self.g67_lay_changes_reached_city(hi, &t, r, &graph.connected_nodes) {
                        let old = &self.hexes[hi].tile;
                        let laid = self.g67_tile_as_laid(hi, &t, r);
                        let paths: Vec<usize> = self
                            .g1867_graph_with(c, Some((hi, &laid)))
                            .connected_paths
                            .into_iter()
                            .filter(|(h, _)| *h == hi)
                            .map(|(_, p)| p)
                            .collect();
                        if !super::tile_rules::g1867_track_restrictions_ok_for(old, &laid, &paths) {
                            continue;
                        }
                    }
                    m.validated = true;
                    out.push(m);
                }
            }
            stats::add(&stats::LAY_CHECK_NS, tc.elapsed().as_nanos() as u64);
        }
        out
    }


    /// [`BaseGame::g67_gen_lays_reference`] computed once per state where it
    /// recomputed per hex or per candidate: the catalog tiles' facts per
    /// rotation (`g1867_catalog_facts`), each hex's facts once, the tiles in
    /// supply and colour once, the map's tile ids once (tile instances), and
    /// `check_track_restrictions!` on one precomputed walk map: the
    /// candidates whose verdict needs the walk are first put to a cheap
    /// over-approximation of it (`WalkMap::may_reach`: unreachable even
    /// without the walk's limits = not connected), then to the exact walk,
    /// stopping at the first connected path that decides the lay. Same
    /// moves, same order.
    fn g67_gen_lays_fast(&mut self, c: &str) -> Vec<Move> {
        use super::tile_rules::{HexFacts, TrackRestriction};
        use crate::title::UpgradeAllowance;
        let Round::Operating(s) = &self.round else { return Vec::new() };
        let laid = s.num_laid_track as usize;
        let upgraded = s.upgraded_this_turn;
        let laid_hexes = s.laid_hexes.clone();
        let Some(slot) = self.title_def().tile_lays().get(laid).copied() else { return Vec::new() };
        let upgrade_ok = match slot.upgrade {
            UpgradeAllowance::Yes => true,
            UpgradeAllowance::No => false,
            UpgradeAllowance::NotIfUpgraded => !upgraded,
        };
        let Some(&ci) = self.corp_idx.get(c) else { return Vec::new() };
        let power = self.corp_buying_power_full(ci);
        let blocked = self.g1867_blocked_hexes();
        let tp = stats::start();
        let graph = self.g1867_graph(c);
        stats::stop(tp, || "lay:graph".into());
        let cat = self.g1867_catalog_facts();
        // Tracker#potential_tiles' supply and phase filters (per state).
        let avail: Vec<&str> = cat
            .sorted
            .iter()
            .filter(|n| self.tile_counts_remaining.get(*n).copied().unwrap_or(0) > 0)
            .filter(|n| self.phase.tiles.iter().any(|col| col == cat.tiles[*n].color_name))
            .map(|n| n.as_str())
            .collect();
        let on_map: std::collections::HashSet<&str> = self.hexes.iter().map(|h| h.tile.id.as_str()).collect();
        let mut walk: Option<super::fastwalk::WalkMap> = None;
        let mut out = Vec::new();
        for (&hi, edges) in &graph.connected_hexes {
            let hex_id = self.hexes[hi].id.clone();
            if blocked.contains(hex_id.as_str()) {
                continue;
            }
            let white = self.hexes[hi].tile.color == crate::tiles::TileColor::White;
            if (white && !slot.lay) || (!white && !upgrade_ok) {
                continue;
            }
            if slot.cannot_reuse_same_hex && laid_hexes.contains(&hex_id) {
                continue;
            }
            let tp = stats::start();
            let nb = self.passable_exits_for(&hex_id);
            let hf = HexFacts::of(&self.hexes[hi].tile);
            let mut with_rot: Vec<(&str, Vec<u8>)> = Vec::new();
            for &t in &avail {
                let tf = &cat.tiles[t];
                if !self.g1867_upgrades_to_fast(hi, &hf, &tf.def, &tf.rot[0]) {
                    continue;
                }
                let rots: Vec<u8> = (0..6u8)
                    .filter(|&r| self.g1867_legal_rotation_fast(hi, &hf, &tf.rot[r as usize], edges, &nb))
                    .collect();
                if !rots.is_empty() {
                    with_rot.push((t, rots));
                }
            }
            let keep = self.g1867_max_exits(hi, with_rot.iter().map(|(t, _)| t.to_string()).collect());
            stats::stop(tp, || "lay:search".into());
            let tp = stats::start();
            let terrain: i32 = self.hexes[hi].tile.upgrades.iter().map(|u| u.cost).sum();
            let untiled = self.hexes[hi].tile.paths.is_empty();
            for (t, rots) in with_rot {
                if !keep.iter().any(|k| k == t) {
                    continue;
                }
                let tf = &cat.tiles[t];
                let inst = (0..10_000u32)
                    .map(|n| format!("{t}-{n}"))
                    .find(|id| !on_map.contains(id.as_str()))
                    .unwrap_or_else(|| format!("{t}-0"));
                for r in rots {
                    let (border, _) = self.tile_border_cost(&hex_id, t, r);
                    if terrain + slot.cost + border > power {
                        continue;
                    }
                    let mut m = Move::plain(
                        StepKind::Track,
                        Action::LayTile { entity_id: c.to_string(), hex_id: hex_id.clone(), tile_id: inst.clone(), rotation: r },
                    );
                    m.validated = true;
                    if untiled {
                        out.push(m);
                        continue;
                    }
                    let rf = &tf.rot[r as usize];
                    if !self.g67_lay_changes_reached_city_fast(hi, &hf, rf, &graph.connected_nodes) {
                        let tw = stats::start();
                        let ok = match super::tile_rules::g1867_track_restriction_targets_fast(&hf, rf) {
                            TrackRestriction::Decided(v) => {
                                stats::count(|| format!("lay:restr:decided_{v}"));
                                v
                            }
                            TrackRestriction::IfReached(mask) => {
                                let laid = self.g67_tile_as_laid_from(hi, &rf.tile);
                                let walk = walk.get_or_insert_with(|| {
                                    let t = stats::start();
                                    let w = super::fastwalk::WalkMap::new(self, c, self.g1867_neighbor_table());
                                    stats::stop(t, || "lay:walkmap".into());
                                    w
                                });
                                let over = super::fastwalk::WalkMap::hex_facts(&laid, hi, walk.corp());
                                let target = |h: usize, p: usize| h == hi && mask.get(p).copied().unwrap_or(false);
                                let t = stats::start();
                                let may = walk.may_reach(Some((hi, &over)), &target);
                                stats::stop(t, || format!("lay:restr:bfs_{may}"));
                                may && {
                                    let t = stats::start();
                                    let v = walk.reaches(Some((hi, &over)), &target);
                                    stats::stop(t, || format!("lay:restr:walk_{v}"));
                                    v
                                }
                            }
                        };
                        stats::stop(tw, || "lay:restriction_walk".into());
                        if !ok {
                            continue;
                        }
                    }
                    out.push(m);
                }
            }
            stats::stop(tp, || "lay:check".into());
        }
        out
    }

    /// [`BaseGame::g67_tile_as_laid`] from the tile `tile_from_def` gives.
    fn g67_tile_as_laid_from(&self, hi: usize, fresh: &crate::graph::Tile) -> crate::graph::Tile {
        let old = &self.hexes[hi].tile;
        let mut t = fresh.clone();
        let map = super::tile_rules::g1867_city_map(old, &t);
        for (oi, ni) in map.iter().enumerate() {
            let Some(ni) = ni else { continue };
            for tok in old.cities[oi].tokens.iter().flatten() {
                let city = &mut t.cities[*ni];
                match city.tokens.iter().position(|x| x.is_none()) {
                    Some(i) => city.tokens[i] = Some(tok.clone()),
                    None => city.tokens.push(Some(tok.clone())),
                }
            }
        }
        t
    }

    /// [`BaseGame::g67_lay_changes_reached_city`] on precomputed facts.
    fn g67_lay_changes_reached_city_fast(
        &self,
        hi: usize,
        hf: &super::tile_rules::HexFacts,
        rf: &super::tile_rules::RotFacts,
        reached: &std::collections::HashSet<super::graph::Node>,
    ) -> bool {
        let hex = &self.hexes[hi];
        let map = super::tile_rules::g1867_city_map(&hex.tile, &rf.tile);
        if !super::tile_rules::cities_towns_kept(&hf.tv, &rf.tv) {
            return false;
        }
        map.iter().enumerate().any(|(oi, ni)| {
            let Some(ni) = ni else { return false };
            let node = super::graph::Node { hex: hi, kind: super::graph::NodeKind::City, index: oi };
            reached.contains(&node) && hex.tile.cities[oi].revenue != rf.tile.cities[*ni].revenue
        })
    }

    /// The tile `name` at `rotation` as Hex#lay leaves it on hex `hi`: the
    /// old cities' tokens moved to their mapped new cities (City#
    /// exchange_token → the first free slot; an overflow appends a slot).
    fn g67_tile_as_laid(&self, hi: usize, name: &str, rotation: u8) -> crate::graph::Tile {
        let old = &self.hexes[hi].tile;
        let mut t = BaseGame::tile_from_def(&self.tile_catalog[name], rotation);
        let map = super::tile_rules::g1867_city_map(old, &t);
        for (oi, ni) in map.iter().enumerate() {
            let Some(ni) = ni else { continue };
            for tok in old.cities[oi].tokens.iter().flatten() {
                let city = &mut t.cities[*ni];
                match city.tokens.iter().position(|x| x.is_none()) {
                    Some(i) => city.tokens[i] = Some(tok.clone()),
                    None => city.tokens.push(Some(tok.clone())),
                }
            }
        }
        t
    }

    /// check_track_restrictions! fast path: the corporation reaches a city
    /// of hex `hi` and the new tile (`name` at `rotation`) gives that city
    /// a different revenue, keeping every old city/town.
    fn g67_lay_changes_reached_city(
        &self,
        hi: usize,
        name: &str,
        rotation: u8,
        reached: &std::collections::HashSet<super::graph::Node>,
    ) -> bool {
        let hex = &self.hexes[hi];
        let def = &self.tile_catalog[name];
        let new_tile = BaseGame::tile_from_def(def, rotation);
        let map = super::tile_rules::g1867_city_map(&hex.tile, &new_tile);
        let new_tv = super::tile_rules::TV::from_def(def, rotation);
        let old_tv = super::tile_rules::TV::from_tile(&hex.tile);
        if !super::tile_rules::cities_towns_kept(&old_tv, &new_tv) {
            return false;
        }
        map.iter().enumerate().any(|(oi, ni)| {
            let Some(ni) = ni else { return false };
            let node = super::graph::Node { hex: hi, kind: super::graph::NodeKind::City, index: oi };
            reached.contains(&node) && hex.tile.cities[oi].revenue != new_tile.cities[*ni].revenue
        })
    }

    // -- merger round ---------------------------------------------------------

    fn g67_gen_merge(&mut self, c: &str) -> Vec<Move> {
        let Round::Merger(s) = &self.round else { return Vec::new() };
        let s = s.clone();
        let targets: Vec<String> = if s.converting || s.merge_major {
            // Ruby Merge#mergeable: unfloated majors.
            self.corporations
                .iter()
                .filter(|t| t.corp_type == CorpType::Major && !t.floated && !t.closed)
                .map(|t| t.sym.clone())
                .collect()
        } else {
            // Ruby Merge#mergeable_candidates reads `graph.connected_nodes`
            // (the Ruby walk): keep the process's candidates whose token
            // sits in a city some member of the set reaches that way.
            let base: Vec<String> = if s.merging.is_empty() { vec![c.to_string()] } else { s.merging.clone() };
            let mut reach: std::collections::HashSet<String> = std::collections::HashSet::new();
            for m in &base {
                for n in self.g1867_graph(m).connected_nodes {
                    if n.kind != super::graph::NodeKind::City {
                        continue;
                    }
                    for t in self.hexes[n.hex].tile.cities[n.index].tokens.iter().flatten() {
                        reach.insert(t.corporation_id.clone());
                    }
                }
            }
            self.merger_mergeable_candidates(&s, c).into_iter().filter(|x| reach.contains(x)).collect()
        };
        targets
            .into_iter()
            .map(|t| Move::plain(StepKind::Merge, Action::Merge { entity_id: c.to_string(), corporation_sym: t }))
            .collect()
    }

    fn g67_gen_post_merger_buy(&self, pid: u32) -> Vec<Move> {
        let Round::Merger(s) = &self.round else { return Vec::new() };
        let Some(target) = s.converted.clone() else { return Vec::new() };
        let Some(idx) = self.merger_buyable_share(&target, pid) else { return Vec::new() };
        let ci = self.corp_idx[target.as_str()];
        let mut m = Move::plain(
            StepKind::PostMergerShares,
            Action::BuyShares {
                entity_id: pid.to_string(),
                corporation_sym: target.clone(),
                shares: Vec::new(),
                percent: self.corporations[ci].shares[idx].percent,
                source: "ipo".into(),
                share_indices: vec![idx],
            },
        );
        m.shares = vec![self.g67_cert_id(ci, idx)];
        vec![m]
    }

    /// ReduceTokens: any placed token of the merged major (the process
    /// keeps two tokens in different hexes).
    fn g67_gen_remove_tokens(&self, c: &str) -> Vec<Move> {
        let mut out = Vec::new();
        for hex in &self.hexes {
            for (city, ct) in hex.tile.cities.iter().enumerate() {
                for (slot, t) in ct.tokens.iter().enumerate() {
                    if t.as_ref().map_or(false, |t| t.corporation_id == c) {
                        out.push(Move::plain(
                            StepKind::ReduceTokens,
                            Action::RemoveToken {
                                entity_id: c.to_string(),
                                hex_id: hex.id.clone(),
                                city_index: city as u8,
                                slot: slot as u8,
                            },
                        ));
                    }
                }
            }
        }
        out
    }

    // =====================================================================
    // Wire form (18xx.games action dicts) — for Python and Ruby replay.
    // =====================================================================

    /// The Ruby tile id of a hex's tile (a preprinted one: "<hex>-<index
    /// in its map.rb HEXES list>", e.g. "C17-4").
    pub(crate) fn g1867_wire_tile_id(&self, hi: usize) -> String {
        let t = &self.hexes[hi].tile;
        if t.id.starts_with("preprinted_") || !t.id.contains('-') {
            crate::title::g1867::preprinted_tile_id(&self.hexes[hi].id)
        } else {
            t.id.clone()
        }
    }

    /// The 18xx.games action dict for a move (state BEFORE applying it).
    pub fn g1867_move_wire(&self, m: &Move) -> Value {
        let a = &m.action;
        let mut o = Map::new();
        o.insert("type".into(), json!(a.action_type()));
        let eid = a.entity_id().to_string();
        let is_player = eid.parse::<u32>().ok().map_or(false, |p| self.players.iter().any(|x| x.id == p));
        if is_player {
            o.insert("entity".into(), json!(eid.parse::<u32>().unwrap()));
            o.insert("entity_type".into(), json!("player"));
        } else if self.company_idx.contains_key(&eid) {
            o.insert("entity".into(), json!(eid));
            o.insert("entity_type".into(), json!("company"));
        } else {
            o.insert("entity".into(), json!(eid));
            o.insert("entity_type".into(), json!("corporation"));
        }
        match a {
            Action::Bid { company_sym, price, .. } => {
                o.insert("company".into(), json!(company_sym));
                o.insert("price".into(), json!(price));
            }
            Action::CorporationBid { corporation_sym, price, .. } => {
                o.insert("corporation".into(), json!(corporation_sym));
                o.insert("price".into(), json!(price));
            }
            Action::Par { corporation_sym, share_price, .. } => {
                o.insert("corporation".into(), json!(corporation_sym));
                let types = self
                    .corp_idx
                    .get(corporation_sym.as_str())
                    .map(|&ci| self.title_def().par_price_types(self.corporations[ci].corp_type))
                    .unwrap_or(&["par"]);
                let cell = self.stock_market.share_prices_with_types(types).into_iter().find(|sp| sp.price == *share_price);
                match cell {
                    Some(sp) => o.insert("share_price".into(), json!(format!("{},{},{}", sp.price, sp.row, sp.column))),
                    None => o.insert("share_price".into(), json!(share_price)),
                };
            }
            Action::BuyShares { percent, .. } | Action::SellShares { percent, .. } => {
                o.insert("shares".into(), json!(m.shares));
                o.insert("percent".into(), json!(percent));
            }
            Action::LayTile { hex_id, tile_id, rotation, .. } => {
                o.insert("hex".into(), json!(hex_id));
                o.insert("tile".into(), json!(tile_id));
                o.insert("rotation".into(), json!(rotation));
            }
            Action::PlaceToken { hex_id, city_index, .. } => {
                if let Some(&hi) = self.hex_idx.get(hex_id.as_str()) {
                    let tid = self.g1867_wire_tile_id(hi);
                    o.insert("city".into(), json!(format!("{tid}-{city_index}")));
                    let slot = self.token_slot_for(hex_id, *city_index as usize, &eid).unwrap_or(0);
                    o.insert("slot".into(), json!(slot));
                    o.insert("tokener".into(), json!(eid));
                }
            }
            Action::RemoveToken { hex_id, city_index, slot, .. } => {
                if let Some(&hi) = self.hex_idx.get(hex_id.as_str()) {
                    let tid = self.g1867_wire_tile_id(hi);
                    o.insert("city".into(), json!(format!("{tid}-{city_index}")));
                    o.insert("slot".into(), json!(slot));
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
                o.insert("variant".into(), json!(variant.clone().unwrap_or_else(|| base_name(train_name).to_string())));
                if let Some(x) = exchange {
                    o.insert("exchange".into(), json!(x));
                }
            }
            Action::DiscardTrain { train_name, .. } => {
                o.insert("train".into(), json!(train_name));
            }
            Action::BuyCompany { company_sym, price, .. } => {
                o.insert("company".into(), json!(company_sym));
                o.insert("price".into(), json!(price));
            }
            Action::Merge { corporation_sym, .. } => {
                o.insert("corporation".into(), json!(corporation_sym));
            }
            Action::Choose { choice, .. } => {
                o.insert("choice".into(), json!(choice));
            }
            _ => {}
        }
        Value::Object(o)
    }

    /// Apply a move (the in-crate path; Python sends the wire dict).
    pub fn g1867_apply_move(&mut self, m: &Move) -> Result<(), GameError> {
        self.process_action_internal(&m.action)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rounds::MergerState;

    /// Float minor `sym` for player `owner` at market column `col` (8 =
    /// $80, below the convert range) with its token in `hex`'s city 0.
    fn rig_minor(g: &mut BaseGame, sym: &str, owner: u32, col: u8, hex: &str) {
        let ci = g.corp_idx[sym];
        let sp = g.stock_market.share_price_at(0, col).unwrap();
        g.corporations[ci].share_price = Some(sp.clone());
        g.corporations[ci].ipo_price = Some(sp);
        g.update_market_cell(sym, 0, col, 0, col);
        g.corporations[ci].floated = true;
        g.corporations[ci].ever_operated = true;
        for i in 0..g.corporations[ci].shares.len() {
            g.corporations[ci].set_share_owner(i, EntityId::player(owner));
        }
        g.corporations[ci].owner_id = EntityId::player(owner);
        let ti = g.corporations[ci].next_token_index().unwrap();
        g.corporations[ci].tokens[ti].used = true;
        g.corporations[ci].tokens[ti].city_hex_id = hex.to_string();
        let tok = g.corporations[ci].tokens[ti].clone();
        let hi = g.hex_idx[hex];
        let city = &mut g.hexes[hi].tile.cities[0];
        let slot = city.tokens.iter().position(|t| t.is_none()).unwrap();
        city.tokens[slot] = Some(tok);
    }

    /// A merger round with `order` to act: BBG and BO (player 1's) and CS
    /// (player 2's) share the 3-slot city of a gray X8 laid on I15, so
    /// each reaches the other two (Ruby `mergeable_candidates`).
    fn rig_merger(order: &[&str]) -> BaseGame {
        let mut g = super::super::fuzz::new_game(4);
        let hi = g.hex_idx["I15"];
        g.hexes[hi].tile = BaseGame::tile_from_def(&g.tile_catalog["X8"], 0);
        rig_minor(&mut g, "BBG", 1, 8, "I15");
        rig_minor(&mut g, "BO", 1, 8, "I15");
        rig_minor(&mut g, "CS", 2, 8, "I15");
        g.clear_graph_cache();
        g.round = Round::Merger(MergerState::new(1, 2, order.iter().map(|s| s.to_string()).collect()));
        g.update_round_state();
        g
    }

    fn merge_targets(moves: &[Move]) -> Vec<String> {
        moves
            .iter()
            .filter_map(|m| match &m.action {
                Action::Merge { corporation_sym, .. } => Some(corporation_sym.clone()),
                _ => None,
            })
            .collect()
    }

    fn types(moves: &[Move]) -> Vec<&'static str> {
        moves.iter().map(|m| m.action.action_type()).collect()
    }

    /// Apply the generated move whose action prints as `want`.
    fn apply(g: &mut BaseGame, policy: CrossPlayerPolicy, want: &Action) {
        let key = format!("{want:?}");
        let moves = g.g1867_legal_moves_with(policy);
        let Some(m) = moves.iter().find(|m| format!("{:?}", m.action) == key) else {
            panic!("{key} not generated; got {:?}", moves.iter().map(|m| format!("{:?}", m.action)).collect::<Vec<_>>())
        };
        g.g1867_apply_move(m).unwrap();
    }

    fn merge(entity: &str, target: &str) -> Action {
        Action::Merge { entity_id: entity.into(), corporation_sym: target.into() }
    }

    /// The title's policy (RESTRICTED) offers a merge only with a minor the
    /// initiator's owner also owns; the rules' set (Ruby) offers any
    /// connected minor whatever its owner.
    #[test]
    fn restricted_policy_merges_only_one_players_minors() {
        let mut g = rig_merger(&["BBG", "BO", "CS"]);
        let rules = merge_targets(&g.g1867_legal_moves_with(CrossPlayerPolicy::RULES));
        assert_eq!(rules, vec!["BO", "CS"], "Ruby: every connected minor");
        let dropped = stats::get(&stats::CROSS_DROPPED_MERGES);
        let restricted = g.g1867_legal_moves_with(CrossPlayerPolicy::RESTRICTED);
        assert_eq!(merge_targets(&restricted), vec!["BO"], "player 2's CS is not offered");
        assert!(types(&restricted).contains(&"pass"));
        assert!(stats::get(&stats::CROSS_DROPPED_MERGES) > dropped, "the dropped merge is counted");
        assert_eq!(merge_targets(&g.g1867_legal_moves()), vec!["BO"], "the title's policy is RESTRICTED");

        // Player 2's CS initiating: both connected minors are player 1's.
        // Nothing to merge with; the pass remains, so the policy needs no
        // fallback to the rules' set.
        let mut g = rig_merger(&["CS", "BBG", "BO"]);
        assert_eq!(merge_targets(&g.g1867_legal_moves_with(CrossPlayerPolicy::RULES)), vec!["BBG", "BO"]);
        assert_eq!(types(&g.g1867_legal_moves_with(CrossPlayerPolicy::RESTRICTED)), vec!["pass"]);
    }

    /// A merger of one player's minors under the title's policy, through
    /// naming the major and the post-merger dealing to the next minor's
    /// turn: the owner gets the 20% presidency (Ruby's 10% + swap), no
    /// other player is involved until rule 9.2 D's one-share round, and
    /// every state on the way has generated moves that apply.
    #[test]
    fn single_owner_merge_runs_through_the_post_merger_steps() {
        let p = CrossPlayerPolicy::RESTRICTED;
        let mut g = rig_merger(&["BBG", "BO", "CS"]);
        apply(&mut g, p, &merge("BBG", "BO"));
        // CS is still a rules candidate (no automatic finish), but only the
        // "Done Adding Corporations" pass is offered.
        assert_eq!(types(&g.g1867_legal_moves_with(p)), vec!["pass"]);
        assert_eq!(merge_targets(&g.g1867_legal_moves_with(CrossPlayerPolicy::RULES)), vec!["CS"]);
        apply(&mut g, p, &Action::Pass { entity_id: "BBG".into() });
        // Naming the major: every unfloated major.
        let targets = merge_targets(&g.g1867_legal_moves_with(p));
        assert!(targets.contains(&"CNR".to_string()) && targets.len() == 8, "{targets:?}");
        apply(&mut g, p, &merge("BBG", "CNR"));

        let ci = g.corp_idx["CNR"];
        let me = EntityId::player(1);
        assert!(g.corporations[g.corp_idx["BBG"]].closed && g.corporations[g.corp_idx["BO"]].closed);
        assert_eq!(g.corporations[ci].president_id(), Some(1));
        assert_eq!(g.corporations[ci].owner_id, me);
        assert_eq!(g.corporations[ci].percent_owned_by(&me), 20, "two minors: 10% + the swap to the 20% cert");
        assert!((2..=4).all(|pid| g.corporations[ci].percent_owned_by(&EntityId::player(pid)) == 0));
        // $80 + $80 = 160 snaps down to the $150 par_2 cell.
        assert_eq!(g.corporations[ci].share_price.as_ref().map(|sp| sp.price), Some(150));
        let Round::Merger(s) = &g.round else { panic!("merger round") };
        assert_eq!(s.converted.as_deref(), Some("CNR"));
        assert_eq!(s.share_dealing_players, vec![1]);
        assert_eq!(s.share_dealing_multiple, vec![1]);

        // Drive the follow-up steps with the first generated move (buys
        // sort before passes) until CS's Merge turn.
        let mut dealt = std::collections::HashSet::new();
        for _ in 0..100 {
            let Round::Merger(s) = &g.round else { panic!("left the merger round early") };
            if s.converted.is_none() && s.current_entity_sym() == Some("CS") {
                break;
            }
            let moves = g.g1867_legal_moves_with(p);
            assert!(!moves.is_empty(), "stuck: {}", g.g1867_debug_state());
            if moves[0].step == StepKind::PostMergerShares {
                dealt.insert(moves[0].action.entity_id().to_string());
            }
            g.g1867_apply_move(&moves[0]).unwrap();
        }
        let Round::Merger(s) = &g.round else { panic!("merger round") };
        assert_eq!(s.current_entity_sym(), Some("CS"), "{}", g.g1867_debug_state());
        assert!(s.converted.is_none());
        assert_eq!(g.corporations[ci].president_id(), Some(1), "one-share buys never overtake the owner");
        assert!(dealt.len() > 1, "rule 9.2 D opened the dealing to the other players: {dealt:?}");
        // CS (player 2's) has no minor of its owner to merge with.
        assert!(merge_targets(&g.g1867_legal_moves_with(p)).is_empty());
    }
}
