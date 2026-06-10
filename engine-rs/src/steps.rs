//! Per-title step/round machinery — a port of Ruby/Python's round architecture.
//!
//! The Python reference (`rl18xx/game/engine/round.py`) models a round as an
//! ORDERED LIST of step objects. `BaseRound.actions_for(entity)` accumulates
//! the actions of every ACTIVE step in list order and stops at (including) the
//! first BLOCKING step, where `blocking = blocks && current_actions`. Titles
//! compose by listing steps (`g1830.py::operating_round`), so adding a step or
//! round type never edits the shared dispatch loop.
//!
//! This module is the Rust equivalent:
//!   * [`StepKind`] — one variant per step class. Each kind implements
//!     `actions` (which action types it offers a given entity), `active`,
//!     a `blocking` override hook, and its own `current_entity`.
//!   * [`StepDesc`] — a step listed in a title's round description, with the
//!     per-title `blocks` flag (mirrors Python's `[BuyCompany, {"blocks":
//!     True}]`).
//!   * [`BaseGame::step_action_types_impl`] — THE shared accumulation loop
//!     (written once; per-title step lists drive it).
//!
//! The per-kind `actions` bodies are faithful ports of the hand-derived gates
//! that previously lived in `game.rs::legal_action_types` — this module
//! RESTRUCTURES dispatch, it does not change rules logic. The 1830 step lists
//! live in `crate::title::g1830` (`operating_steps`, `stock_steps`,
//! `auction_steps`), mirroring the Python lists exactly, including order.
//!
//! This IS the production dispatch: `legal_action_types` /
//! `legal_action_types_for_factored` (and therefore the factored enumeration,
//! decode, and MCTS) all route through [`BaseGame::step_action_types_impl`],
//! and the operating round's `skip_steps` / pc advances derive the turn
//! sequence from the title's step list ([`next_operating_pc`]). The historical
//! hand-derived dispatch survives only as the test-only differential oracle
//! (`game.rs::legacy_legal_action_types_oracle`).
//!
//! How future titles plug in (design notes, not implemented):
//!   * 1867's merger (M&A) round: add a `Merger` round variant + step kinds
//!     (`Merge`, `PostMergerShares`, `ReduceTokens`, ...), list them in
//!     `g1867::merger_steps()`, and insert the round into the title's
//!     round-sequence description. No edits to the loop below.
//!   * 1822's minor-first-OR `BuyTrain`: list an extra `BuyTrain`-like kind at
//!     the front of `g1822::operating_steps()` whose `actions`/`active` gate on
//!     "minor's first OR turn" — the accumulation loop and skip machinery pick
//!     it up automatically because they iterate the title's list.

use crate::entities::EntityId;
use crate::game::BaseGame;
use crate::rounds::{OperatingState, OperatingStep, Round};

// ---------------------------------------------------------------------------
// Step kinds + round descriptions
// ---------------------------------------------------------------------------

/// One step class. Mirrors the Python step classes used by 1830's rounds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StepKind {
    // Operating-round steps (Python round.py class names)
    Bankrupt,
    Exchange,
    SpecialTrack,
    SpecialToken,
    BuyCompany,
    HomeToken,
    Track,
    Token,
    Route,
    Dividend,
    DiscardTrain,
    BuyTrain,
    // Stock-round blocking step
    BuySellParShares,
    // Auction-round steps
    CompanyPendingPar,
    WaterfallAuction,
}

/// A step as listed in a title's round description.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StepDesc {
    pub kind: StepKind,
    /// Python's `BaseStep.blocks` (or the `{"blocks": ...}` option):
    /// whether this step can be the round's blocking step.
    pub blocks: bool,
}

impl StepDesc {
    pub const fn blocking(kind: StepKind) -> Self {
        StepDesc { kind, blocks: true }
    }
    pub const fn non_blocking(kind: StepKind) -> Self {
        StepDesc {
            kind,
            blocks: false,
        }
    }

    /// Which Rust operating-turn program counter ([`OperatingStep`]) this
    /// listed step corresponds to, if any. The Rust engine flattens the
    /// per-step `passed` flags of the Python model into a single pc; the
    /// title's step list is the source of truth for the pc SEQUENCE (see
    /// [`next_operating_pc`]). Steps without a pc (HomeToken, DiscardTrain
    /// out-of-turn, the non-blocking specials) overlay other pcs.
    pub fn operating_pc(&self) -> Option<OperatingStep> {
        match self.kind {
            StepKind::Track => Some(OperatingStep::LayTile),
            StepKind::Token => Some(OperatingStep::PlaceToken),
            StepKind::Route => Some(OperatingStep::RunRoutes),
            StepKind::Dividend => Some(OperatingStep::Dividend),
            StepKind::DiscardTrain => Some(OperatingStep::DiscardTrain),
            StepKind::BuyTrain => Some(OperatingStep::BuyTrain),
            StepKind::BuyCompany if self.blocks => Some(OperatingStep::BuyCompany),
            _ => None,
        }
    }
}

/// The next operating-turn pc after `cur`, DERIVED from the title's step list:
/// the pc of the next listed step (after the one mapping to `cur`) that has a
/// pc of its own; `Done` when the list is exhausted. Replaces the old
/// hardcoded `OperatingStep::next()` enum-ordering function.
pub fn next_operating_pc(steps: &[StepDesc], cur: &OperatingStep) -> OperatingStep {
    let pos = steps.iter().position(|d| d.operating_pc().as_ref() == Some(cur));
    let start = match pos {
        Some(i) => i + 1,
        None => return OperatingStep::Done, // Done (or unknown) → stays Done
    };
    for d in &steps[start..] {
        if let Some(pc) = d.operating_pc() {
            return pc;
        }
    }
    OperatingStep::Done
}

/// One entry in a title's round-sequence description.
///
/// A title's game flow = an opening round (1830: the waterfall auction,
/// constructed at game start) followed by the repeating
/// [`round_cycle`](crate::title::g1830::round_cycle); the game's `turn`
/// counter increments each time the cycle wraps. Future titles add kinds
/// here (1867's `Merger` round between OR sets, 1822's `Choices` round) and
/// list them in their cycle — `transition_to_next_round` only walks the
/// title's list.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RoundKind {
    /// The opening (waterfall) auction. Constructed once at game start;
    /// no 1830 cycle entry.
    Auction,
    /// A stock round.
    Stock,
    /// A set of operating rounds (`phase.operating_rounds` of them; the
    /// set-internal repetition lives in the OperatingState, not the cycle).
    OperatingSet,
}

/// The entity a step acts for / offers actions to. Players act in auction and
/// stock rounds, corporations in operating rounds, and companies when a
/// private's special ability is the actor (teleport token, exchange, ...).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StepEntity {
    Player(u32),
    Corp(String),
    Company(String),
}

fn operating<'a>(snap: &'a Round) -> Option<&'a OperatingState> {
    match snap {
        Round::Operating(s) => Some(s),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// The shared accumulation loop + per-kind step logic
// ---------------------------------------------------------------------------

impl BaseGame {
    /// The title's step list for the current round. (1830 is the only title;
    /// when a second title lands this dispatches through the game's title.)
    pub(crate) fn round_step_descs(&self, snap: &Round) -> &'static [StepDesc] {
        match snap {
            Round::Auction(_) => crate::title::g1830::auction_steps(),
            Round::Stock(_) => crate::title::g1830::stock_steps(),
            Round::Operating(_) => self.operating_step_descs(),
        }
    }

    /// The title's OPERATING-round step list — the single title-dispatch point
    /// for everything that walks the OR turn sequence (`skip_steps`, the pc
    /// advance after a blocking Pass, the post-lay auto-advance). 1830-only
    /// today; a second title dispatches here.
    pub(crate) fn operating_step_descs(&self) -> &'static [StepDesc] {
        crate::title::g1830::operating_steps()
    }

    /// The title's repeating round cycle — everything after the opening
    /// auction. Drives `transition_to_next_round`; the title-dispatch point
    /// for the round SEQUENCE. 1830: Stock → OR set → (wrap, `turn` += 1) →
    /// Stock → ...
    pub(crate) fn round_cycle(&self) -> &'static [RoundKind] {
        crate::title::g1830::round_cycle()
    }

    /// THE shared `actions_for` accumulation loop (Python
    /// `BaseRound.actions_for` + `active_step` + the ActionHelper's parallel
    /// company-actions union), written once for all rounds and titles:
    ///
    ///   1. find the BLOCKING step: the first listed step that is active and
    ///      `blocks` and whose `current_actions` (for the step's OWN current
    ///      entity) are non-empty — or whose per-kind blocking override fires
    ///      (SpecialToken while a teleport is pending);
    ///   2. the round's acting entity is that step's current entity;
    ///   3. accumulate `step.actions(entity)` over the active steps in list
    ///      order up to and including the blocking step;
    ///   4. union in the parallel company actions: for each company that may
    ///      act alongside the entity (its owner's companies + the exchange
    ///      company), run the SAME loop with the company as entity.
    ///
    /// Returns the deduplicated action-type strings. Equals the historical
    /// hand-derived `legal_action_types` output as a SET at every state (the
    /// differential tests pin this).
    pub(crate) fn step_action_types_impl(&mut self) -> Vec<String> {
        if self.finished {
            return Vec::new();
        }
        let snap = self.round.clone();
        let steps = self.round_step_descs(&snap);

        // 1+2: blocking step + acting entity.
        let mut blocking: Option<(usize, StepEntity)> = None;
        for (i, desc) in steps.iter().enumerate() {
            if !self.step_active(desc, &snap) {
                continue;
            }
            if self.step_blocking_override(desc, &snap) {
                if let Some(e) = self.step_current_entity(desc, &snap) {
                    blocking = Some((i, e));
                }
                break;
            }
            if !desc.blocks {
                continue;
            }
            if let Some(e) = self.step_current_entity(desc, &snap) {
                if !self.step_actions(desc, &e, &snap).is_empty() {
                    blocking = Some((i, e));
                    break;
                }
            }
        }
        let Some((bidx, entity)) = blocking else {
            return Vec::new();
        };

        let mut out: Vec<String> = Vec::new();
        let push_all = |ts: Vec<&'static str>, out: &mut Vec<String>| {
            for t in ts {
                if !out.iter().any(|x| x == t) {
                    out.push(t.to_string());
                }
            }
        };

        // 3: accumulate for the acting entity.
        for desc in steps.iter().take(bidx + 1) {
            if !self.step_active(desc, &snap) {
                continue;
            }
            let ts = self.step_actions(desc, &entity, &snap);
            push_all(ts, &mut out);
        }

        // 4: parallel company actions (Python FactoredActionHelper
        // `_company_actions` / ActionHelper `get_company_actions`).
        if !self.suppress_parallel_companies(&entity, &snap) {
            for csym in self.parallel_companies(&entity) {
                let ce = StepEntity::Company(csym);
                if ce == entity {
                    continue; // already accumulated above
                }
                for desc in steps.iter().take(bidx + 1) {
                    if !self.step_active(desc, &snap) {
                        continue;
                    }
                    let ts = self.step_actions(desc, &ce, &snap);
                    push_all(ts, &mut out);
                }
            }
        }
        out
    }

    // -- step predicates ----------------------------------------------------

    /// Python `step.active` (default `True`; overridden by DiscardTrain,
    /// HomeToken, CompanyPendingPar to gate on their pending state).
    fn step_active(&self, desc: &StepDesc, snap: &Round) -> bool {
        match desc.kind {
            StepKind::DiscardTrain => {
                operating(snap).map_or(false, |s| !s.crowded_corps.is_empty())
            }
            StepKind::HomeToken => {
                operating(snap).map_or(false, |s| !s.pending_tokens.is_empty())
            }
            StepKind::CompanyPendingPar => match snap {
                Round::Auction(s) => s.pending_par.is_some(),
                _ => false,
            },
            _ => true,
        }
    }

    /// Python's per-step `blocking` overrides. SpecialToken (round.py:4062)
    /// blocks unconditionally while a teleport is pending — that is what
    /// suppresses the regular Token/BuyCompany steps until the teleport
    /// completes.
    fn step_blocking_override(&self, desc: &StepDesc, snap: &Round) -> bool {
        match desc.kind {
            // NOTE the pc gate: in this engine `teleport_pending` can outlive
            // the PlaceToken pc (a regular corp Pass advances the pc without
            // clearing the flag), and the teleport only blocks while the turn
            // is actually at the token placement.
            StepKind::SpecialToken => operating(snap)
                .map_or(false, |s| s.teleport_pending && s.step == OperatingStep::PlaceToken),
            _ => false,
        }
    }

    /// Python `step.current_entity` (`active_entities[0]`): the round's entity
    /// at `entity_index` by default; DiscardTrain acts for the first crowded
    /// corp, SpecialToken for the teleported company, CompanyPendingPar for
    /// the player owing a par.
    fn step_current_entity(&self, desc: &StepDesc, snap: &Round) -> Option<StepEntity> {
        match desc.kind {
            StepKind::DiscardTrain => operating(snap)
                .and_then(|s| s.crowded_corps.first())
                .map(|c| StepEntity::Corp(c.clone())),
            StepKind::SpecialToken => {
                let s = operating(snap)?;
                if !s.teleport_pending {
                    return None;
                }
                self.teleport_company(s).map(StepEntity::Company)
            }
            StepKind::CompanyPendingPar => match snap {
                Round::Auction(s) => s.pending_par.as_ref().map(|(_, pid)| StepEntity::Player(*pid)),
                _ => None,
            },
            _ => match snap {
                Round::Auction(s) => Some(StepEntity::Player(s.active_player_id())),
                Round::Stock(s) => Some(StepEntity::Player(s.current_player_id())),
                Round::Operating(s) => s
                    .current_corp_sym()
                    .map(|sym| StepEntity::Corp(sym.to_string())),
            },
        }
    }

    /// The open teleport company owned by the operating corp whose teleport
    /// tile has been laid (`ability_used`) — Python's `round.teleported`.
    fn teleport_company(&self, s: &OperatingState) -> Option<String> {
        let corp_eid = EntityId::corporation(s.current_corp_sym()?);
        self.companies
            .iter()
            .find(|co| {
                !co.closed
                    && co.ability_used
                    && co.owner == corp_eid
                    && crate::abilities::teleport(&co.sym).is_some()
            })
            .map(|co| co.sym.clone())
    }

    // -- parallel company actions --------------------------------------------

    /// The companies whose special abilities may act in parallel with the
    /// round's entity: the entity's own companies (Python iterates
    /// `current_entity.companies`) plus the exchange company (MH) whenever an
    /// exchange is available (the ActionHelper's MH-special branch).
    fn parallel_companies(&self, entity: &StepEntity) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        let owner_eid = match entity {
            StepEntity::Corp(sym) => Some(EntityId::corporation(sym)),
            StepEntity::Player(pid) => Some(EntityId::player(*pid)),
            StepEntity::Company(sym) => {
                out.push(sym.clone());
                None
            }
        };
        if let Some(eid) = owner_eid {
            for co in &self.companies {
                if !co.closed && co.owner == eid {
                    out.push(co.sym.clone());
                }
            }
        }
        for co in &self.companies {
            if self.company_exchange_available(&co.sym) && !out.iter().any(|s| s == &co.sym) {
                out.push(co.sym.clone());
            }
        }
        out
    }

    /// Historical engine gates that suppress the parallel-company union
    /// entirely (ported verbatim from the superseded hand-derived arms —
    /// behavior-preserving for 1830):
    ///   * Stock round, `must_sell`: only SellShares is legal
    ///     (BuySellParShares.actions short-circuit, round.py:1527-1528).
    ///   * Operating round, a corp crowded OUT OF TURN (discarder != operator):
    ///     the parallel steps key off the OPERATOR and return nothing for the
    ///     discarder (see the game-30642 note in the old DiscardTrain arm).
    fn suppress_parallel_companies(&self, entity: &StepEntity, snap: &Round) -> bool {
        match snap {
            Round::Stock(_) => {
                if let StepEntity::Player(pid) = entity {
                    let sellable = !self.sellable_bundles_for_factored(*pid).is_empty();
                    let certs = self.num_certs_internal(*pid);
                    self.stock_must_sell(*pid, sellable, certs)
                } else {
                    false
                }
            }
            Round::Operating(s) => match s.crowded_corps.first() {
                Some(c) => Some(c.as_str()) != s.current_corp_sym(),
                None => false,
            },
            Round::Auction(_) => false,
        }
    }

    /// Per-company exchange availability (1830: MH → NYC): company open, has
    /// an exchange ability, and a target corporation still holds
    /// non-president shares outside player hands.
    fn company_exchange_available(&self, company_sym: &str) -> bool {
        let Some(co) = self.companies.iter().find(|c| c.sym == company_sym) else {
            return false;
        };
        if co.closed {
            return false;
        }
        let Some((corporations, _)) = crate::abilities::exchange(&co.sym) else {
            return false;
        };
        corporations.iter().any(|corp_sym| {
            self.corp_idx.get(*corp_sym).map_or(false, |&ci| {
                self.corporations[ci]
                    .shares
                    .iter()
                    .any(|s| !s.president && !s.owner.is_player())
            })
        })
    }

    // -- step.actions(entity) -------------------------------------------------

    /// Python `step.actions(entity)` mapped to the engine's action-type
    /// strings. Each arm is a faithful port of the corresponding gate from the
    /// superseded hand-derived `legal_action_types` arms.
    fn step_actions(
        &mut self,
        desc: &StepDesc,
        entity: &StepEntity,
        snap: &Round,
    ) -> Vec<&'static str> {
        match desc.kind {
            // -- operating: non-blocking specials --------------------------
            StepKind::Bankrupt => {
                // Bankrupt surfaces only during a forced train buy where the
                // president cannot raise the price (round.py:1339-1352 +
                // can_go_bankrupt). Ported from the old BuyTrain arm.
                let Some(s) = operating(snap) else { return vec![] };
                if s.step != OperatingStep::BuyTrain {
                    return vec![];
                }
                let StepEntity::Corp(sym) = entity else { return vec![] };
                if Some(sym.as_str()) != s.current_corp_sym() {
                    return vec![];
                }
                if !self.operating_must_buy_train(sym) {
                    return vec![];
                }
                let pres = self
                    .corp_idx
                    .get(sym.as_str())
                    .and_then(|&ci| self.corporations[ci].president_id());
                match pres {
                    Some(pid) if self.can_go_bankrupt_emr(pid, sym) => vec!["bankrupt"],
                    _ => vec![],
                }
            }
            StepKind::Exchange => {
                // Exchange acts for COMPANY entities only (round.py:2942-2952;
                // a corp/player entity gets nothing from this step).
                let StepEntity::Company(sym) = entity else { return vec![] };
                if self.company_exchange_available(sym) {
                    vec!["buy_shares"]
                } else {
                    vec![]
                }
            }
            StepKind::SpecialTrack => {
                // A private's special tile lay, acting for the company: CS's
                // bonus lay at any step of the owning corp's turn; DH's
                // teleport lay at the Track step only.
                let Some(s) = operating(snap) else { return vec![] };
                let StepEntity::Company(sym) = entity else { return vec![] };
                let Some(op) = s.current_corp_sym() else { return vec![] };
                let corp_eid = EntityId::corporation(op);
                let usable = self.companies.iter().any(|co| {
                    co.sym == *sym
                        && !co.closed
                        && !co.ability_used
                        && co.owner == corp_eid
                        && (crate::abilities::tile_lay(&co.sym).is_some()
                            || (crate::abilities::teleport(&co.sym).is_some()
                                && s.step == OperatingStep::LayTile))
                });
                if usable {
                    vec!["lay_tile"]
                } else {
                    vec![]
                }
            }
            StepKind::SpecialToken => {
                // The teleport company's station-token placement, offered only
                // while the teleport is PENDING (tile laid, token not yet
                // placed/declined). Pass = decline (`teleport_complete()`).
                let Some(s) = operating(snap) else { return vec![] };
                if !s.teleport_pending || s.step != OperatingStep::PlaceToken {
                    return vec![];
                }
                let StepEntity::Company(sym) = entity else { return vec![] };
                if self.teleport_company(s).as_deref() != Some(sym.as_str()) {
                    return vec![];
                }
                let mut v = Vec::new();
                if self.teleport_token_placeable(s, sym) {
                    v.push("place_token");
                }
                v.push("pass");
                v
            }
            StepKind::BuyCompany => {
                // Python BuyCompany.actions (round.py:1394-1425): the
                // operating corp buys president-owned privates in phases 3-4.
                // The blocking instance also offers a bare Pass when the corp
                // holds a company with an unused (tile-lay) ability.
                let Some(s) = operating(snap) else { return vec![] };
                if desc.blocks && s.step != OperatingStep::BuyCompany {
                    return vec![];
                }
                let StepEntity::Corp(sym) = entity else { return vec![] };
                if Some(sym.as_str()) != s.current_corp_sym() {
                    return vec![];
                }
                if self.has_buyable_companies(s) {
                    if desc.blocks {
                        vec!["buy_company", "pass"]
                    } else {
                        vec!["buy_company"]
                    }
                } else if desc.blocks {
                    // The blocking instance is only reached when the skip
                    // machinery stopped the turn at this step (a purchasable
                    // company exists or an unused lay ability is open) — the
                    // corp may always decline with Pass. Mirrors both Python's
                    // `[Pass]`-when-abilities branch (round.py:1411-1412) and
                    // the historical hand-derived arm, which offered Pass
                    // unconditionally at this pc.
                    vec!["pass"]
                } else {
                    vec![]
                }
            }
            StepKind::HomeToken => {
                // Pending home-token choice (1830: ERIE's E11 OO hex). The
                // placement is mandatory — no Pass.
                let Some(s) = operating(snap) else { return vec![] };
                let StepEntity::Corp(sym) = entity else { return vec![] };
                if Some(sym.as_str()) != s.current_corp_sym() {
                    return vec![];
                }
                if s.pending_tokens.is_empty() {
                    vec![]
                } else {
                    vec!["place_token"]
                }
            }
            // -- operating: the regular turn-sequence steps -----------------
            StepKind::Track => {
                let Some(s) = operating(snap) else { return vec![] };
                if s.step != OperatingStep::LayTile {
                    return vec![];
                }
                let StepEntity::Corp(sym) = entity else { return vec![] };
                if Some(sym.as_str()) != s.current_corp_sym() {
                    return vec![];
                }
                let mut v = Vec::new();
                if self.track_has_layable_tile(sym) {
                    v.push("lay_tile");
                }
                v.push("pass");
                v
            }
            StepKind::Token => {
                let Some(s) = operating(snap) else { return vec![] };
                if s.step != OperatingStep::PlaceToken
                    || !s.pending_tokens.is_empty()
                    || s.teleport_pending
                {
                    return vec![];
                }
                let StepEntity::Corp(sym) = entity else { return vec![] };
                if Some(sym.as_str()) != s.current_corp_sym() {
                    return vec![];
                }
                let tokenable = self.tokenable_cities_for_factored(sym);
                // Home token still unplaced: placement is offered without
                // connectivity (mirrors the old needs_home_token gate).
                let needs_home_token = self
                    .corp_idx
                    .get(sym.as_str())
                    .map_or(false, |&ci| !self.corporations[ci].home_token_ever_placed);
                let mut v = Vec::new();
                if !tokenable.is_empty() || needs_home_token {
                    v.push("place_token");
                }
                v.push("pass");
                v
            }
            StepKind::Route => {
                let Some(s) = operating(snap) else { return vec![] };
                if s.step != OperatingStep::RunRoutes {
                    return vec![];
                }
                let StepEntity::Corp(sym) = entity else { return vec![] };
                if Some(sym.as_str()) != s.current_corp_sym() {
                    return vec![];
                }
                vec!["run_routes"]
            }
            StepKind::Dividend => {
                let Some(s) = operating(snap) else { return vec![] };
                if s.step != OperatingStep::Dividend {
                    return vec![];
                }
                let StepEntity::Corp(sym) = entity else { return vec![] };
                if Some(sym.as_str()) != s.current_corp_sym() {
                    return vec![];
                }
                vec!["dividend"]
            }
            StepKind::DiscardTrain => {
                // Python DiscardTrain.actions (round.py:2685-2686): any corp
                // in crowded_corps may discard.
                let Some(s) = operating(snap) else { return vec![] };
                let StepEntity::Corp(sym) = entity else { return vec![] };
                if s.crowded_corps.iter().any(|c| c == sym) {
                    vec!["discard_train"]
                } else {
                    vec![]
                }
            }
            StepKind::BuyTrain => {
                let Some(s) = operating(snap) else { return vec![] };
                if s.step != OperatingStep::BuyTrain {
                    return vec![];
                }
                let StepEntity::Corp(sym) = entity else { return vec![] };
                if Some(sym.as_str()) != s.current_corp_sym() {
                    return vec![];
                }
                // Python BuyTrain.actions (round.py:792-805): when the corp
                // MUST buy a train, `president_may_contribute` holds and the
                // step returns [SellShares, BuyTrain] — Pass is excluded.
                if self.operating_must_buy_train(sym) {
                    vec!["buy_train", "sell_shares"]
                } else {
                    vec!["buy_train", "pass"]
                }
            }
            // -- stock ------------------------------------------------------
            StepKind::BuySellParShares => {
                let Round::Stock(s) = snap else { return vec![] };
                let StepEntity::Player(pid) = entity else { return vec![] };
                if *pid != s.current_player_id() {
                    return vec![];
                }
                let pid = *pid;
                let buyable = !self.buyable_shares_for_factored(pid).is_empty();
                let sellable = !self.sellable_bundles_for_factored(pid).is_empty();
                let certs = self.num_certs_internal(pid);
                // must_sell short-circuit (round.py:1527-1528): ONLY SellShares.
                if self.stock_must_sell(pid, sellable, certs) {
                    return vec!["sell_shares"];
                }
                let mut v = Vec::new();
                if sellable {
                    v.push("sell_shares");
                }
                if buyable {
                    v.push("buy_shares");
                }
                if !s.bought_this_turn {
                    let can_par = certs < self.cert_limit as u32
                        && self.corporations.iter().any(|c| {
                            c.ipo_price.is_none()
                                && self.players.iter().find(|p| p.id == pid).map_or(false, |p| {
                                    p.cash
                                        >= self
                                            .stock_market
                                            .par_prices()
                                            .first()
                                            .copied()
                                            .unwrap_or(0)
                                            * 2
                                })
                        });
                    if can_par {
                        v.push("par");
                    }
                }
                v.push("pass");
                v
            }
            // -- auction ----------------------------------------------------
            StepKind::CompanyPendingPar => {
                let Round::Auction(s) = snap else { return vec![] };
                let Some((_, pending_pid)) = &s.pending_par else { return vec![] };
                let StepEntity::Player(pid) = entity else { return vec![] };
                if pid == pending_pid {
                    vec!["par"]
                } else {
                    vec![]
                }
            }
            StepKind::WaterfallAuction => {
                let Round::Auction(s) = snap else { return vec![] };
                if s.pending_par.is_some() || s.remaining_companies.is_empty() {
                    return vec![];
                }
                let StepEntity::Player(pid) = entity else { return vec![] };
                if *pid != s.active_player_id() {
                    return vec![];
                }
                let player_cash = self
                    .players
                    .iter()
                    .find(|p| p.id == *pid)
                    .map_or(0, |p| p.cash);
                let biddable: Vec<usize> = if let Some(auc_ci) = s.auctioning {
                    vec![auc_ci]
                } else {
                    s.remaining_companies.clone()
                };
                let can_bid = biddable.iter().any(|&ci| {
                    let value = self.companies.get(ci).map_or(0, |c| c.value);
                    let min_bid = s.min_bid_for(ci, value);
                    let max_bid = s.max_bid(*pid, ci, player_cash);
                    max_bid >= min_bid
                });
                let mut v = Vec::new();
                if can_bid {
                    v.push("bid");
                }
                v.push("pass");
                v
            }
        }
    }

    // -- ported gate helpers ---------------------------------------------------

    /// Whether the operating corp can lay at least one tile somewhere
    /// (candidate hexes minus ability-blocked hexes, terrain affordable).
    /// Ported verbatim from the old LayTile arm.
    fn track_has_layable_tile(&mut self, corp_sym: &str) -> bool {
        let corp_cash = self
            .corp_idx
            .get(corp_sym)
            .map_or(0, |&ci| self.corporations[ci].cash);
        let candidates = self.lay_tile_candidate_hexes(corp_sym);
        let blocked_hexes = self.ability_blocked_hexes();
        for (hex_id, edges) in &candidates {
            if blocked_hexes.contains(hex_id.as_str()) {
                continue;
            }
            if self.has_layable_tile_for_corp(hex_id, edges, corp_cash) {
                return true;
            }
        }
        false
    }

    /// Whether the corp must buy a train: no trains, depot (upcoming or
    /// discarded) non-empty, and a legal revenue route exists. Ported from the
    /// old BuyTrain arm (= Python's `must_buy_train`).
    pub(crate) fn operating_must_buy_train(&mut self, corp_sym: &str) -> bool {
        let no_trains = self
            .corp_idx
            .get(corp_sym)
            .map_or(true, |&ci| self.corporations[ci].trains.is_empty());
        let depot_has_trains = !self.depot.trains.is_empty() || !self.depot.discarded.is_empty();
        no_trains && depot_has_trains && self.must_buy_train_pub(corp_sym)
    }

    /// Whether the teleport company's token can actually be placed: a free
    /// slot on a teleport hex city + the corp has an unplaced token. Ported
    /// from the old PlaceToken arm's `dh_token` gate.
    fn teleport_token_placeable(&self, s: &OperatingState, company_sym: &str) -> bool {
        let Some(op) = s.current_corp_sym() else { return false };
        let slot_free = crate::abilities::teleport(company_sym).map_or(false, |(hexes, _)| {
            hexes.iter().any(|h| {
                self.hex_idx.get(*h).map_or(false, |&hi| {
                    self.hexes[hi]
                        .tile
                        .cities
                        .iter()
                        .any(|c| c.tokens.iter().any(|t| t.is_none()))
                })
            })
        });
        slot_free
            && self
                .corp_idx
                .get(op)
                .map_or(false, |&ci| self.corporations[ci].next_token_index().is_some())
    }

    /// Whether the corp owns a company with an unused bonus tile-lay ability
    /// (CS). Ported from the old BuyCompany skip/blocking gate (a teleport
    /// ability does not keep BuyCompany blocking — Python's `abilities()`
    /// timing check filters it).
    pub(crate) fn corp_has_unused_lay_ability(&self, corp_sym: &str) -> bool {
        let corp_eid = EntityId::corporation(corp_sym);
        self.companies.iter().any(|c| {
            !c.closed
                && !c.ability_used
                && c.owner == corp_eid
                && crate::abilities::tile_lay(&c.sym).is_some()
        })
    }
}

// ---------------------------------------------------------------------------
// PyO3 surface
// ---------------------------------------------------------------------------

use pyo3::prelude::*;

#[pymethods]
impl BaseGame {
    /// Legal action types via the step-list machinery (the table-driven
    /// `actions_for` loop). Differential-tested against the historical
    /// hand-derived `legal_action_types` at every state.
    fn step_action_types(&mut self) -> Vec<String> {
        self.step_action_types_impl()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rounds::OperatingStep;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use std::collections::HashMap;

    fn new_4p_game() -> BaseGame {
        let mut players = HashMap::new();
        players.insert(1, "Alice".to_string());
        players.insert(2, "Bob".to_string());
        players.insert(3, "Carol".to_string());
        players.insert(4, "Dave".to_string());
        BaseGame::build(vec![1, 2, 3, 4], players)
    }

    fn sorted_set(v: Vec<String>) -> Vec<String> {
        let mut v: Vec<String> = v;
        v.sort();
        v.dedup();
        v
    }

    /// Differential: at every state of a random self-played game, the
    /// step-machinery enumeration must equal the historical hand-derived
    /// dispatch (retained as the test-only `legacy_legal_action_types_oracle`)
    /// exactly (as a set — the historical Vec order is hand-arranged; every
    /// consumer is set/index-based).
    fn run_differential_walk(seed: u64, max_actions: usize) {
        let mut game = new_4p_game();
        let mut rng = StdRng::seed_from_u64(seed);
        let company_offset = crate::action_index::layout().action_offsets["CompanyBuyShares"];

        for step in 0..max_actions {
            if game.is_finished_pub() {
                break;
            }
            let old = sorted_set(game.legacy_legal_action_types_oracle());
            let new = sorted_set(game.step_action_types_impl());
            assert_eq!(
                old, new,
                "seed {} step {}: legal_action_types {:?} != step machinery {:?} \
                 (round {:?})",
                seed, step, old, new, game.round
            );

            let choices = game.get_factored_choices_impl();
            if choices.is_empty() {
                break;
            }
            // Defensive filter: pending-token entries with `slot=None` used
            // to be enumerated (parity-faithfully) by BOTH engines and then
            // rejected at process time. Both enumerations now skip full
            // cities (factored.rs / factored_action_helper.py), so this
            // filter should never match — kept so a regression degrades the
            // walk's coverage instead of crashing it (the enumeration
            // equality assert above still sees every entry).
            let applicable: Vec<&crate::factored::LegalAction> = choices
                .iter()
                .filter(|c| {
                    !(c.action_type == "PlaceToken"
                        && c.params.get("slot").map_or(false, |s| s.is_null()))
                })
                .collect();
            if applicable.is_empty() {
                break;
            }
            let la = applicable[rng.gen_range(0..applicable.len())];
            let Some(idx) = crate::action_index::legal_action_to_index(la) else {
                panic!("seed {} step {}: unindexable {:?}", seed, step, la);
            };
            let price = la.price_range.map(|(lo, hi)| rng.gen_range(lo..=hi));
            let action = game
                .build_action(la, idx, idx >= company_offset, price)
                .unwrap_or_else(|e| panic!("seed {} step {}: build_action: {}", seed, step, e));
            game.process_action_internal(&action)
                .unwrap_or_else(|e| panic!("seed {} step {}: process {:?}: {}", seed, step, action, e));
        }
    }

    #[test]
    fn differential_step_machinery_random_walks() {
        for seed in 0..12u64 {
            run_differential_walk(seed, 4000);
        }
    }

    #[test]
    fn differential_step_machinery_long_walks() {
        for seed in [100u64, 101, 102, 103] {
            run_differential_walk(seed, 100_000);
        }
    }

    /// Pin the 1830 round descriptions to the Python reference lists
    /// (g1830.py::operating_round, base.py::stock_round/new_auction_round).
    #[test]
    fn g1830_step_lists_mirror_python() {
        use StepKind::*;
        let ops: Vec<(StepKind, bool)> = crate::title::g1830::operating_steps()
            .iter()
            .map(|d| (d.kind, d.blocks))
            .collect();
        assert_eq!(
            ops,
            vec![
                (Bankrupt, false),
                (Exchange, false),
                (SpecialTrack, false),
                (SpecialToken, false),
                (BuyCompany, false),
                (HomeToken, true),
                (Track, true),
                (Token, true),
                (Route, true),
                (Dividend, true),
                (DiscardTrain, true),
                (BuyTrain, true),
                (BuyCompany, true),
            ]
        );
        let stock: Vec<(StepKind, bool)> = crate::title::g1830::stock_steps()
            .iter()
            .map(|d| (d.kind, d.blocks))
            .collect();
        assert_eq!(
            stock,
            vec![
                (DiscardTrain, true),
                (Exchange, false),
                (SpecialTrack, false),
                (BuySellParShares, true),
            ]
        );
        let auction: Vec<(StepKind, bool)> = crate::title::g1830::auction_steps()
            .iter()
            .map(|d| (d.kind, d.blocks))
            .collect();
        assert_eq!(auction, vec![(CompanyPendingPar, true), (WaterfallAuction, true)]);
    }

    /// Pin 1830's round cycle to the Python reference (base.py::next_round!):
    /// opening auction (pre-cycle), then Stock → OR set, turn += 1 on wrap.
    #[test]
    fn g1830_round_cycle_mirrors_python() {
        assert_eq!(
            crate::title::g1830::round_cycle(),
            &[RoundKind::Stock, RoundKind::OperatingSet]
        );
    }

    /// The derived pc sequence equals the historical hardcoded
    /// `OperatingStep::next()` ordering.
    #[test]
    fn derived_pc_sequence_matches_1830_or_order() {
        let steps = crate::title::g1830::operating_steps();
        let seq = [
            OperatingStep::LayTile,
            OperatingStep::PlaceToken,
            OperatingStep::RunRoutes,
            OperatingStep::Dividend,
            OperatingStep::DiscardTrain,
            OperatingStep::BuyTrain,
            OperatingStep::BuyCompany,
            OperatingStep::Done,
        ];
        for w in seq.windows(2) {
            assert_eq!(
                next_operating_pc(steps, &w[0]),
                w[1],
                "next pc after {:?}",
                w[0]
            );
        }
        assert_eq!(next_operating_pc(steps, &OperatingStep::Done), OperatingStep::Done);
    }
}
