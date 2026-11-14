//! `BaseGame::encode_tokens`: one position as typed tokens (see the module
//! docs and `docs/rl_encoder.md`).

use std::collections::HashMap;

use super::schema::*;
use super::spec::*;
use super::*;
use crate::entities::EntityId;
use crate::game::BaseGame;
use crate::rounds::{Round, SoldTiming};

/// One encoded position. Flat row-major buffers; `rel` holds `(src, dst,
/// kind)` triples.
#[derive(Clone, Debug, PartialEq)]
pub struct TokenEncoding {
    pub title_id: i32,
    pub num_hexes: usize,
    pub num_entities: usize,
    /// `[H, F_HEX]`
    pub hex_feat: Vec<f32>,
    /// `[H, S_MAX]`: entity index of the token in each slot; `PTR_NONE` empty
    /// / no slot, `PTR_NEUTRAL` a token of no corporation.
    pub hex_ptr: Vec<i32>,
    /// `[H, S_MAX]`: entity index of the corporation a slot is reserved for.
    pub hex_res: Vec<i32>,
    /// `[H, S_MAX]`: the city each slot belongs to, -1 = no slot.
    pub hex_slot_city: Vec<i32>,
    /// `[H, 2]`: (label id, tile id).
    pub hex_ids: Vec<i32>,
    /// `[E]`
    pub ent_type: Vec<i32>,
    /// `[E, F_ENT]`
    pub ent_feat: Vec<f32>,
    /// `[E, P_ENT]`
    pub ent_ptr: Vec<i32>,
    /// `[E, P_HEX]`
    pub ent_hex: Vec<i32>,
    /// `[R, 3]`
    pub rel: Vec<i32>,
    /// `[R]`
    pub rel_val: Vec<f32>,
    pub meta: [i32; META_LEN],
    /// Tokens that did not fit `S_MAX` slots (0 on every registered map).
    pub truncated_slots: u32,
}

impl TokenEncoding {
    pub fn num_relations(&self) -> usize {
        self.rel_val.len()
    }
    /// The hex part, for de-duplicating maps across a batch (pointers name
    /// corporations, whose entity indices do not depend on the seat rotation,
    /// so positions of one map share it whoever is to move).
    pub fn hex_hash(&self) -> u64 {
        // FNV-style over the raw bits in four independent lanes (a single
        // multiply chain over ~10k floats cost as much as the encoding);
        // `same_hexes` confirms every hit, so the hash only has to be fast.
        const P: u64 = 0x0000_0100_0000_01b3;
        let mut lanes = [0xcbf2_9ce4_8422_2325u64 ^ self.title_id as u64; 4];
        let mut mix4 = |xs: [u64; 4]| {
            for k in 0..4 {
                lanes[k] = (lanes[k] ^ xs[k]).wrapping_mul(P).rotate_left(5);
            }
        };
        let bits = |v: &[f32], i: usize| v.get(i).map_or(0, |x| x.to_bits() as u64);
        let f = &self.hex_feat;
        for c in (0..f.len()).step_by(8) {
            mix4([
                bits(f, c) | bits(f, c + 1) << 32,
                bits(f, c + 2) | bits(f, c + 3) << 32,
                bits(f, c + 4) | bits(f, c + 5) << 32,
                bits(f, c + 6) | bits(f, c + 7) << 32,
            ]);
        }
        let ints = |v: &[i32], i: usize| v.get(i).map_or(0, |x| *x as u32 as u64);
        for v in [&self.hex_ptr, &self.hex_res, &self.hex_slot_city, &self.hex_ids] {
            for c in (0..v.len()).step_by(4) {
                mix4([ints(v, c), ints(v, c + 1), ints(v, c + 2), ints(v, c + 3)]);
            }
        }
        lanes.iter().fold(0u64, |h, &l| (h ^ l).wrapping_mul(P).rotate_left(17))
    }
    pub fn same_hexes(&self, other: &TokenEncoding) -> bool {
        self.title_id == other.title_id
            && self.hex_ptr == other.hex_ptr
            && self.hex_res == other.hex_res
            && self.hex_slot_city == other.hex_slot_city
            && self.hex_ids == other.hex_ids
            && self.hex_feat.len() == other.hex_feat.len()
            && self.hex_feat.iter().zip(&other.hex_feat).all(|(a, b)| a.to_bits() == b.to_bits())
    }
}

/// Sequential feature writer (debug builds check the schema's count).
struct W<'a> {
    buf: &'a mut [f32],
    i: usize,
}

impl<'a> W<'a> {
    fn new(buf: &'a mut [f32]) -> Self {
        W { buf, i: 0 }
    }
    #[inline]
    fn put(&mut self, v: f32) {
        self.buf[self.i] = v;
        self.i += 1;
    }
    #[inline]
    fn flag(&mut self, b: bool) {
        self.put(if b { 1.0 } else { 0.0 });
    }
    #[inline]
    fn ratio(&mut self, num: f32, den: f32) {
        self.put(if den != 0.0 { num / den } else { 0.0 });
    }
    fn slice(&mut self, n: usize) -> &mut [f32] {
        let s = &mut self.buf[self.i..self.i + n];
        self.i += n;
        s
    }
    fn done(&self, expected: usize) {
        debug_assert_eq!(self.i, expected, "feature count does not match the schema");
    }
}

/// Per-corporation share holdings, gathered in one pass.
#[derive(Default, Clone)]
struct Holdings {
    seat_pct: [u16; MAX_SEATS],
    /// Certificates held per seat that count toward the cert limit.
    seat_certs: [u16; MAX_SEATS],
    ipo_pct: u16,
    market_pct: u16,
    other_pct: u16,
    /// (holding corporation entity, percent)
    corp_holders: Vec<(i32, u16)>,
    /// (holding company entity, percent): the 1822 family's tax haven.
    company_holders: Vec<(i32, u16)>,
}

/// A slot reservation: (spec hex, city or -1 = the hex, slot or -1 = the
/// first free slot, corporation entity).
type Reservation = (usize, i32, i32, i32);

/// The 1822 family's per-position state beyond the shared engine's
/// (encoder version 2), gathered once per encoding.
struct FamilyState {
    /// Corporations / companies the current decision is about, in order of
    /// importance (`subject` features; the first is the global lot pointer).
    subject_corps: Vec<String>,
    subject_companies: Vec<String>,
    /// Players in the current sub-step's queue, in order.
    queue: Vec<u32>,
    /// Bid boxes: (company, slot k, boxes in its row, the first minor box).
    bidbox: Vec<(String, usize, usize, bool)>,
    /// Hex assignments: (spec hex, key: a company sym or "forest").
    assignments: Vec<(usize, String)>,
    /// 1822MX's NdeM token auction while it runs: (spec hex, price) of each
    /// token for sale, the one up for bid, the high bid and its bidder.
    tokens_for_sale: Vec<(usize, i32)>,
    token_up: Option<usize>,
    high_bid: Option<(u32, i32)>,
    /// 1822PNW's paper-mill bonus in this phase.
    mill_amount: i32,
}

struct Ctx<'a> {
    g: &'a BaseGame,
    spec: &'static RlSpec,
    sc: f32,
    np: usize,
    seat_ids: Vec<u32>,
    rotation: usize,
    mover: Option<u32>,
    /// game corporation index -> entity index (-1 = not in the spec).
    corp_ent: Vec<i32>,
    /// game company index -> entity index.
    company_ent: Vec<i32>,
    /// game corporation index -> spec corporation index.
    corp_spec: Vec<i32>,
    phase_idx: i32,
    /// The 1822 family's extra state (None for other titles).
    fam: Option<FamilyState>,
}

impl<'a> Ctx<'a> {
    fn seat_of(&self, pid: u32) -> Option<usize> {
        let pos = self.seat_ids.iter().position(|&x| x == pid)?;
        Some((pos + self.np - self.rotation) % self.np)
    }
    fn seat_ent(&self, pid: u32) -> i32 {
        self.seat_of(pid).map_or(PTR_NONE, |s| RlSpec::seat_entity(s) as i32)
    }
    fn corp_ent_sym(&self, sym: &str) -> i32 {
        self.g.corp_idx.get(sym).map_or(PTR_NONE, |&i| self.corp_ent[i])
    }
    fn company_ent_sym(&self, sym: &str) -> i32 {
        self.g.company_idx.get(sym).map_or(PTR_NONE, |&i| self.company_ent[i])
    }
    /// Entity index of an owner id: "player:N", "corp:SYM", "company:SYM".
    fn owner_ent(&self, eid: &EntityId) -> i32 {
        if let Some(pid) = eid.player_id() {
            return self.seat_ent(pid);
        }
        if let Some(sym) = eid.0.strip_prefix("corp:") {
            return self.corp_ent_sym(sym);
        }
        if let Some(sym) = eid.0.strip_prefix("company:") {
            return self.company_ent_sym(sym);
        }
        PTR_NONE
    }
}

#[inline]
fn clean(v: f32) -> f32 {
    if v.is_finite() {
        v.clamp(-FEATURE_CLIP, FEATURE_CLIP)
    } else {
        0.0
    }
}

impl BaseGame {
    /// The player deciding the current move (seat 0 of the encoding): the
    /// rule of `mover_index` (`acting_player_id_pub`) for the shared engine,
    /// `g1822_acting_player`'s for the 1822 family. None when the game is
    /// over or nobody acts.
    pub(crate) fn rl_mover_id(&self) -> Option<u32> {
        self.rl_mover_at(self.rl_g1822_active_step())
    }

    /// The 1822 family's active step (Ruby `active_step`), None for other
    /// titles or a finished game. An encoding evaluates it once and hands it
    /// to the mover and step-kind readers (each evaluation runs the steps'
    /// `actions`: ~10-17 us).
    fn rl_g1822_active_step(&self) -> Option<usize> {
        if self.g1822.is_none() || self.finished {
            return None;
        }
        self.g1822_active_step()
    }

    fn rl_mover_at(&self, active: Option<usize>) -> Option<u32> {
        if self.finished {
            return None;
        }
        if self.g1822.is_none() {
            return self.acting_player_id_pub();
        }
        use crate::g1822::Ent;
        let bi = active?;
        match self.g1822_step_current_entity(bi)? {
            Ent::Player(p) => Some(p),
            Ent::Corp(c) if self.g1822_mx_is_ndem(&c) => self.g1822_mx_ndem_acting_player(),
            Ent::Corp(c) => self.g1822_corp_president(&c),
            Ent::Company(co) => match self.co_owner(&co).corp_sym() {
                Some(c) => self.g1822_corp_president(c),
                None => self.co_owner(&co).player_id(),
            },
        }
    }

    /// `ROUND_KINDS` name of the live round.
    pub(crate) fn rl_round_kind(&self) -> &'static str {
        if let Some(st) = self.g1822.as_deref() {
            use crate::g1822::RoundKind;
            return match st.round.kind {
                RoundKind::Stock => "Stock",
                RoundKind::Choices => "Choices",
                RoundKind::Operating => "Operating",
                RoundKind::Merger => "Merger",
            };
        }
        self.round.round_type_str()
    }

    /// `STEP_KINDS` name of the step the current decision belongs to. For
    /// the 1822 family Ruby's `active_step`; for 1830/1867 read off the round
    /// state (the blocking step as the step machinery would find it, without
    /// its `&mut` evaluation): pending side steps first (MajorTrainless,
    /// pending home tokens, discards, a DH teleport), then the round's own
    /// step / the operating turn's program counter.
    pub(crate) fn rl_step_kind(&self) -> &'static str {
        self.rl_step_kind_at(self.rl_g1822_active_step())
    }

    fn rl_step_kind_at(&self, active: Option<usize>) -> &'static str {
        if self.finished {
            return "none";
        }
        if self.g1822.is_some() {
            use crate::g1822::Step as S;
            let Some(i) = active else { return "none" };
            return match self.rnd().steps[i] {
                S::DiscardTrain => "DiscardTrain",
                S::BuySellParShares => "BuySellParShares",
                S::Choose => "Choose",
                S::PendingToken => "PendingToken",
                S::FirstTurnHousekeeping => "FirstTurnHousekeeping",
                S::AcquireCompany => "AcquireCompany",
                S::SpecialChoose => "SpecialChoose",
                S::SpecialTrack => "SpecialTrack",
                S::SpecialToken => "SpecialToken",
                S::Track => "Track",
                S::DestinationToken => "DestinationToken",
                S::Token => "Token",
                S::Route => "Route",
                S::Dividend => "Dividend",
                S::BuyTrain => "BuyTrain",
                S::MinorAcquisition => "MinorAcquisition",
                S::IssueShares => "IssueShares",
                S::Assign => "Assign",
                S::Merge => "Merge",
                S::CashOutNdem => "CashOutNdem",
                S::AuctionNdemTokens => "AuctionNdemTokens",
                S::AssignSawmill => "AssignSawmill",
                S::AcquisitionTrack => "AcquisitionTrack",
            };
        }
        let trainless = !self.trainless_major.is_empty();
        match &self.round {
            Round::Auction(a) => {
                if a.pending_par.is_some() {
                    "CompanyPendingPar"
                } else if a.single_item.is_some() {
                    "SingleItemAuction"
                } else {
                    "WaterfallAuction"
                }
            }
            Round::Stock(s) => {
                if trainless {
                    "MajorTrainless"
                } else if !s.pending_home_tokens.is_empty() {
                    "HomeToken"
                } else if s.bid_auction.is_some() {
                    "BuySellParSharesViaBid"
                } else {
                    "BuySellParShares"
                }
            }
            Round::Operating(o) => {
                use crate::rounds::OperatingStep as P;
                if trainless {
                    "MajorTrainless"
                } else if !o.crowded_corps.is_empty() {
                    "DiscardTrain"
                } else if !o.pending_tokens.is_empty() {
                    "HomeToken"
                } else if o.teleport_pending {
                    "SpecialToken"
                } else {
                    match o.step {
                        P::RedeemShares => "RedeemShares",
                        P::LayTile => "Track",
                        P::PlaceToken => "Token",
                        P::RunRoutes => "Route",
                        P::Dividend => "Dividend",
                        P::BuyCompanyPreloan => "BuyCompanyPreloan",
                        P::LoanOperations => "LoanOperations",
                        P::DiscardTrain => "DiscardTrain",
                        P::BuyTrain => "BuyTrain",
                        P::BuyCompany => "BuyCompany",
                        P::Done => "none",
                    }
                }
            }
            Round::Merger(m) => {
                if trainless {
                    "MajorTrainless"
                } else if m.corporations_removing_tokens.is_some() {
                    "ReduceTokens"
                } else if m.converted.is_some() && !self.merger_eligible_players(m).is_empty() {
                    "PostMergerShares"
                } else if !self.merger_crowded_corps().is_empty() {
                    "DiscardTrain"
                } else {
                    "Merge"
                }
            }
        }
    }

    /// The 1822 family's extra state for version 2's features.
    fn rl_family_state(&self, spec: &RlSpec, active: Option<usize>) -> Option<FamilyState> {
        use crate::g1822::{mx, AcquireState, MergeState, Step};
        let st = self.g1822.as_deref()?;
        let r = &st.round;
        let fam = self.fam();
        let active_step = active.map(|i| r.steps[i]);
        let ndem_auction = r.ndem_auction.as_ref().filter(|_| self.g1822_mx_auction_active());

        let mut subject_corps: Vec<String> = Vec::new();
        let mut subject_companies: Vec<String> = Vec::new();
        let add = |s: &Option<String>, v: &mut Vec<String>| {
            if let Some(s) = s {
                if !v.contains(s) {
                    v.push(s.clone());
                }
            }
        };
        if ndem_auction.is_some() {
            add(&Some(mx::NDEM.to_string()), &mut subject_corps);
        }
        if r.merge_state != MergeState::None {
            for s in [&r.merge_new, &r.merge_assoc, &r.merge_unassoc] {
                add(s, &mut subject_corps);
            }
        }
        if r.acquire_state != AcquireState::SelectMinor {
            add(&r.selected_minor, &mut subject_corps);
        }
        if active_step == Some(Step::AcquisitionTrack) {
            add(&r.ca.acquiring_major, &mut subject_corps);
        }
        add(&r.converting_major, &mut subject_corps);
        add(&r.p20_minor, &mut subject_corps);
        add(&r.special_track_company, &mut subject_companies);
        if active_step == Some(Step::SpecialChoose) {
            // the just-acquired company still holding its choices (P8)
            if let Some(op) = self.g1822_operator() {
                add(&self.g1822_special_choose_company(&op), &mut subject_companies);
            }
        }
        if r.ca.sawmill_state == crate::g1822::ca::SawmillState::Assigned {
            add(&Some(crate::g1822::ca::COMPANY_SAWMILL.to_string()), &mut subject_companies);
        }

        // the sub-step's queue of deciding players (1822MX's NdeM)
        let mut queue: Vec<u32> = Vec::new();
        let mut tokens_for_sale = Vec::new();
        let (mut token_up, mut high_bid) = (None, None);
        if let Some(a) = ndem_auction {
            queue = if a.up.is_some() { a.remaining_bidders.clone() } else { a.remaining_choosers.clone() };
            for (k, t) in a.available.iter().enumerate() {
                if let Some(&hi) = spec.hex_index.get(t.hex.as_str()) {
                    tokens_for_sale.push((hi, t.cost));
                    if a.up == Some(k) {
                        token_up = Some(hi);
                    }
                }
            }
            if let (Some(_), Some(p)) = (a.up, a.high_bidder) {
                high_bid = Some((p, a.high_bid));
            }
        } else if active_step == Some(Step::Track)
            && self.g1822_is_mx()
            && self.g1822_operator().as_deref() == Some(mx::NDEM)
        {
            queue = r.ndem_layers.clone();
        }

        // bid boxes: minors, concessions, privates
        // (shown in every round: the boxes the next stock round auctions)
        let mut bidbox = Vec::new();
        let rows = [
            (self.bidbox_minors(), fam.bidding_box_minor_count),
            (self.bidbox_concessions(), fam.bidding_box_concession_count),
            (self.bidbox_privates(), fam.bidding_box_private_count),
        ];
        for (row, (list, size)) in rows.into_iter().enumerate() {
            let size = size.max(list.len()).max(1);
            for (k, co) in list.into_iter().enumerate() {
                bidbox.push((co, k, size, row == 0 && k == 0));
            }
        }

        let mut assignments = Vec::new();
        for (hex, keys) in &st.hex_assignments {
            if let Some(&hi) = spec.hex_index.get(hex.as_str()) {
                assignments.extend(keys.iter().map(|k| (hi, k.clone())));
            }
        }
        let mill_amount = if crate::g1822::operating::ruby_to_i(&self.phase.name) < 5 { 10 } else { 30 };
        Some(FamilyState {
            subject_corps,
            subject_companies,
            queue,
            bidbox,
            assignments,
            tokens_for_sale,
            token_up,
            high_bid,
            mill_amount,
        })
    }

    /// Route bonus of an 1822 hex assignment (the router's values): a
    /// 1822PNW forest $10 a crossing, its ski haus (P17) $30, its paper mill
    /// (P15) the phase's mill amount, 1822CA's sawmill (P13) its chosen bonus.
    fn rl_assignment_bonus(&self, key: &str, fs: &FamilyState) -> i32 {
        let ca = self.g1822_is_ca();
        match key {
            "forest" => 10,
            "P17" if !ca => 30,
            "P15" if !ca => fs.mill_amount,
            k if ca && k == crate::g1822::ca::COMPANY_SAWMILL => self.g1822.as_deref().map_or(0, |st| st.ca.sawmill_bonus),
            _ => 0,
        }
    }

    fn rl_priority_deal_player(&self) -> u32 {
        match &self.round {
            Round::Stock(s) if self.g1822.is_none() => s.priority_deal_player,
            Round::Auction(a) => match a.last_purchaser_id {
                Some(pid) => self.next_player_id(pid),
                None => self.priority_deal_player,
            },
            _ => self.priority_deal_player,
        }
    }

    /// Encode this position as typed tokens.
    pub fn encode_tokens(&self) -> TokenEncoding {
        let spec = rl_spec_for(&self.title);
        let np = self.players.len();
        // seats: the seating order (`BaseGame.players`, the descriptor vocabulary's
        // players), rotated so that seat 0 is the mover
        let seat_ids: Vec<u32> = self.players.iter().map(|p| p.id).collect();
        let active = self.rl_g1822_active_step();
        let mover = self.rl_mover_at(active);
        let rotation = mover.and_then(|m| seat_ids.iter().position(|&x| x == m)).unwrap_or(0);

        // Game index -> spec index: the same order on every game of a title
        // (checked per element, looked up otherwise).
        let mut corp_ent = vec![PTR_NONE; self.corporations.len()];
        let mut corp_spec = vec![PTR_NONE; self.corporations.len()];
        for (i, c) in self.corporations.iter().enumerate() {
            let j = match spec.corps.get(i) {
                Some(cs) if cs.sym == c.sym => Some(i),
                _ => spec.corp_index.get(c.sym.as_str()).copied(),
            };
            if let Some(j) = j {
                corp_ent[i] = (spec.e_corp0 + j) as i32;
                corp_spec[i] = j as i32;
            }
        }
        let mut company_ent = vec![PTR_NONE; self.companies.len()];
        for (i, c) in self.companies.iter().enumerate() {
            let j = match spec.companies.get(i) {
                Some(cs) if cs.sym == c.sym => Some(i),
                _ => spec.company_index.get(c.sym.as_str()).copied(),
            };
            if let Some(j) = j {
                company_ent[i] = (spec.e_company0 + j) as i32;
            }
        }
        let phase_idx = spec.phase_index.get(self.phase.name.as_str()).map_or(PTR_NONE, |&i| i as i32);
        let ctx = Ctx {
            g: self,
            spec,
            sc: (self.starting_cash.max(1)) as f32,
            np,
            seat_ids,
            rotation,
            mover,
            corp_ent,
            company_ent,
            corp_spec,
            phase_idx,
            fam: self.rl_family_state(spec, active),
        };

        let h = spec.num_hexes();
        let e = spec.num_entities;
        let mut out = TokenEncoding {
            title_id: spec.title_id,
            num_hexes: h,
            num_entities: e,
            hex_feat: vec![0.0; h * F_HEX],
            hex_ptr: vec![PTR_NONE; h * S_MAX],
            hex_res: vec![PTR_NONE; h * S_MAX],
            hex_slot_city: vec![PTR_NONE; h * S_MAX],
            hex_ids: vec![0; h * 2],
            ent_type: vec![ENT_PAD; e],
            ent_feat: vec![0.0; e * F_ENT],
            ent_ptr: vec![PTR_NONE; e * P_ENT],
            ent_hex: vec![PTR_NONE; e * P_HEX],
            rel: Vec::with_capacity(96),
            rel_val: Vec::with_capacity(32),
            meta: [0; META_LEN],
            truncated_slots: 0,
        };

        let holdings = self.rl_holdings(&ctx);
        let reservations = self.rl_reservations(&ctx);
        let reserved_corps = self.rl_encode_hexes(&ctx, &reservations, &mut out);
        self.rl_encode_entities(&ctx, &holdings, &reserved_corps, &mut out);
        self.rl_encode_relations(&ctx, &holdings, &mut out);

        let acting = ctx.owner_ent(&self.round_state.active_entity_id);
        let round_num = match &self.round {
            Round::Operating(o) => o.round_num as i32,
            Round::Merger(m) => m.round_num as i32,
            _ => 0,
        };
        out.meta = [
            spec.title_id,
            round_kind_id(self.rl_round_kind()),
            step_kind_id(self.rl_step_kind_at(active)),
            if self.finished { PTR_NONE } else { acting },
            np as i32,
            rotation as i32,
            self.finished as i32,
            round_num,
        ];
        for v in out.hex_feat.iter_mut().chain(out.ent_feat.iter_mut()).chain(out.rel_val.iter_mut()) {
            *v = clean(*v);
        }
        out
    }

    // -- holdings ------------------------------------------------------------

    fn rl_holdings(&self, ctx: &Ctx) -> Vec<Holdings> {
        // the 1822 family's tax haven holds shares for its company's owner
        let tax_haven = match self.g1822 {
            Some(_) if !self.fam().company_osth.is_empty() => ctx.company_ent_sym(self.fam().company_osth),
            _ => PTR_NONE,
        };
        self.corporations
            .iter()
            .map(|c| {
                let mut h = Holdings::default();
                let sym = c.sym.as_str();
                // `num_certs_internal`: certificates outside the cert-limit-exempt zones count
                let counts = !BaseGame::corp_exempt_from_cert_limit(c) as u16;
                for s in &c.shares {
                    let pct = s.percent as u16;
                    let o = s.owner.0.as_str();
                    let own = o.strip_prefix("ipo:").or_else(|| o.strip_prefix("corp:")) == Some(sym);
                    if o.is_empty() || own {
                        h.ipo_pct += pct;
                    } else if o == "market" {
                        h.market_pct += pct;
                    } else if let Some(pid) = s.owner.player_id() {
                        match ctx.seat_of(pid) {
                            Some(seat) => {
                                h.seat_pct[seat] += pct;
                                h.seat_certs[seat] += counts;
                            }
                            None => h.other_pct += pct,
                        }
                    } else if let Some(sym) = o.strip_prefix("corp:") {
                        h.other_pct += pct;
                        let ent = ctx.corp_ent_sym(sym);
                        if ent >= 0 {
                            match h.corp_holders.iter_mut().find(|(e, _)| *e == ent) {
                                Some(x) => x.1 += pct,
                                None => h.corp_holders.push((ent, pct)),
                            }
                        }
                    } else {
                        h.other_pct += pct;
                        if tax_haven >= 0 && o == crate::g1822::TAX_HAVEN_OWNER {
                            match h.company_holders.iter_mut().find(|(e, _)| *e == tax_haven) {
                                Some(x) => x.1 += pct,
                                None => h.company_holders.push((tax_haven, pct)),
                            }
                        }
                    }
                }
                h
            })
            .collect()
    }

    // -- reservations ----------------------------------------------------------

    fn rl_reservations(&self, ctx: &Ctx) -> Vec<Reservation> {
        let spec = ctx.spec;
        let mut out: Vec<Reservation> = Vec::new();
        if let Some(st) = self.g1822.as_deref() {
            for (key, slots) in &st.city_reservations {
                let Some((hex, city)) = key.rsplit_once('#') else { continue };
                let (Some(&hi), Ok(ci)) = (spec.hex_index.get(hex), city.parse::<i32>()) else { continue };
                for (si, r) in slots.iter().enumerate() {
                    if let Some(sym) = r {
                        let ent = ctx.corp_ent_sym(sym);
                        if ent >= 0 {
                            out.push((hi, ci, si as i32, ent));
                        }
                    }
                }
            }
            for (hex, syms) in &st.tile_reservations {
                let Some(&hi) = spec.hex_index.get(hex.as_str()) else { continue };
                for sym in syms {
                    let ent = ctx.corp_ent_sym(sym);
                    if ent >= 0 {
                        out.push((hi, -1, -1, ent));
                    }
                }
            }
            return out;
        }
        for (gi, corp) in self.corporations.iter().enumerate() {
            let sj = ctx.corp_spec[gi];
            if sj < 0 || corp.closed || corp.home_token_ever_placed {
                continue;
            }
            let cs = &spec.corps[sj as usize];
            if cs.home_hex < 0 {
                continue;
            }
            let coord = spec.hex_coords[cs.home_hex as usize];
            if corp.tokens.iter().any(|t| t.used && t.city_hex_id == coord) {
                continue;
            }
            out.push((cs.home_hex as usize, cs.home_city as i32, -1, ctx.corp_ent[gi]));
        }
        if let Some(ni) = spec.national {
            let ent = (spec.e_corp0 + ni) as i32;
            for hex in &self.national_reservations {
                if let Some(&hi) = spec.hex_index.get(hex.as_str()) {
                    // held in the hex's last city (1867: Montreal's third)
                    out.push((hi, i32::MAX, -1, ent));
                }
            }
        }
        out
    }

    // -- hexes -----------------------------------------------------------------

    /// Writes the hex arrays; returns the corporation entities whose
    /// reservations are on the map.
    fn rl_encode_hexes(&self, ctx: &Ctx, reservations: &[Reservation], out: &mut TokenEncoding) -> Vec<i32> {
        let spec = ctx.spec;
        let blocked = if self.g1822.is_none() { self.ability_blocked_hexes() } else { Default::default() };
        // Route bonuses of open companies: (spec hex, amount).
        let mut bonuses: Vec<(usize, i32)> = Vec::new();
        for (gi, co) in self.companies.iter().enumerate() {
            if co.closed {
                continue;
            }
            let ent = ctx.company_ent[gi];
            if ent < 0 {
                continue;
            }
            bonuses.extend(spec.companies[ent as usize - spec.e_company0].hex_bonus.iter().copied());
        }
        let laid: &[String] = match (&self.round, self.g1822.as_deref()) {
            (_, Some(st)) => &st.round.laid_hexes,
            (Round::Operating(o), None) => &o.laid_hexes,
            _ => &[],
        };
        let empty_borders: Vec<(u8, i32)> = Vec::new();
        let mut reserved_corps: Vec<i32> = Vec::new();

        for i in 0..spec.num_hexes() {
            let coord = spec.hex_coords[i];
            let gh = match self.hexes.get(i) {
                Some(hx) if hx.id == coord => i,
                _ => match self.hex_idx.get(coord) {
                    Some(&j) => j,
                    None => continue,
                },
            };
            let hex = &self.hexes[gh];
            let tile = &hex.tile;
            let preprinted = tile.id.starts_with("preprinted_")
                || tile.id.starts_with("offboard_")
                || tile.id.starts_with("path_");
            let laid_tile = !tile.id.is_empty() && !preprinted;

            // -- slots, pointers, reservations --------------------------------
            let base = i * S_MAX;
            let mut k = 0usize;
            let mut city_first_slot = [usize::MAX; 8];
            let mut city_slots = [0usize; 8];
            let (mut n_tokens, mut n_neutral, mut n_extra, mut free) = (0usize, 0usize, 0usize, 0usize);
            let (mut n_dest, mut n_coal) = (0usize, 0usize);
            for (ci, city) in tile.cities.iter().enumerate() {
                if ci < 8 {
                    city_first_slot[ci] = k;
                    city_slots[ci] = city.tokens.len();
                }
                let extras = city.extra_tokens.iter().map(Some);
                for tok in city.tokens.iter().map(|t| t.as_ref()).chain(extras) {
                    if k >= S_MAX {
                        out.truncated_slots += 1;
                        continue;
                    }
                    out.hex_slot_city[base + k] = ci as i32;
                    match tok {
                        Some(t) => {
                            let ent = if t.token_type == "neutral" { PTR_NEUTRAL } else { ctx.corp_ent_sym(&t.corporation_id) };
                            let ent = if ent == PTR_NONE { PTR_NEUTRAL } else { ent };
                            out.hex_ptr[base + k] = ent;
                            n_tokens += 1;
                            if ent == PTR_NEUTRAL {
                                n_neutral += 1;
                            }
                            match t.token_type.as_str() {
                                "destination" => n_dest += 1,
                                "coal" => n_coal += 1,
                                _ => {}
                            }
                        }
                        None => free += 1,
                    }
                    k += 1;
                }
                n_extra += city.extra_tokens.len();
            }
            let mut n_reserved = 0usize;
            for &(_, city, slot, ent) in reservations.iter().filter(|r| r.0 == i) {
                let nc = tile.cities.len().min(8);
                if nc == 0 {
                    continue;
                }
                let city = match city {
                    i32::MAX => nc - 1,
                    c if c >= 0 && (c as usize) < nc => c as usize,
                    _ => 0,
                };
                let first = city_first_slot[city];
                if first == usize::MAX {
                    continue;
                }
                let target = if slot >= 0 && (slot as usize) < city_slots[city] {
                    Some(first + slot as usize)
                } else {
                    (first..first + city_slots[city])
                        .find(|&s| s < S_MAX && out.hex_ptr[base + s] == PTR_NONE && out.hex_res[base + s] == PTR_NONE)
                };
                if let Some(s) = target.filter(|&s| s < S_MAX) {
                    if out.hex_res[base + s] == PTR_NONE {
                        out.hex_res[base + s] = ent;
                        n_reserved += 1;
                        if !reserved_corps.contains(&ent) {
                            reserved_corps.push(ent);
                        }
                    }
                }
            }

            // -- ids ---------------------------------------------------------------
            out.hex_ids[i * 2] = label_id(tile.label.as_deref());
            out.hex_ids[i * 2 + 1] = if !laid_tile {
                0
            } else {
                let id = tile.name.as_str();
                let id = if spec.tile_index.contains_key(id) { id } else { id.rsplit_once('-').map_or(id, |(n, _)| n) };
                spec.tile_index.get(id).map_or(spec.num_tiles() as i32 + 1, |&t| t as i32 + 1)
            };

            // -- features -------------------------------------------------------------
            let mut w = W::new(&mut out.hex_feat[i * F_HEX..(i + 1) * F_HEX]);
            w.put(spec.hex_pos[i][0]);
            w.put(spec.hex_pos[i][1]);
            let c = w.slice(6);
            c[color_slot(tile.color)] = 1.0;
            w.flag(laid_tile);
            w.flag(preprinted);
            w.flag(tile.label.is_some());
            let r = w.slice(6);
            if laid_tile {
                r[tile.rotation as usize % 6] = 1.0;
            }
            let ms = spec.max_stop_revenue;
            w.put(tile.cities.len() as f32 / 6.0);
            w.put(tile.cities.iter().map(|c| c.tokens.len()).sum::<usize>() as f32 / 6.0);
            w.put(free as f32 / 6.0);
            w.put(tile.towns.len() as f32 / 2.0);
            w.flag(!tile.offboards.is_empty());
            w.put(tile.cities.iter().map(|c| c.revenue).max().unwrap_or(0) as f32 / ms);
            w.put(tile.towns.iter().map(|c| c.revenue).max().unwrap_or(0) as f32 / ms);
            match tile.offboards.first() {
                Some(ob) => {
                    let yellow = ob.revenue;
                    let green = ob.green_revenue.unwrap_or(yellow);
                    let brown = ob.brown_revenue.unwrap_or(green);
                    let gray = ob.gray_revenue.unwrap_or(brown);
                    w.put(ob.phase_revenue(&self.phase.tiles) as f32 / ms);
                    for v in [yellow, green, brown, gray] {
                        w.put(v as f32 / ms);
                    }
                    w.flag(!ob.groups.is_empty());
                }
                None => {
                    w.slice(6);
                }
            }
            w.put(tile.upgrades.iter().map(|u| u.cost).max().unwrap_or(0) as f32 / spec.max_lay_cost);
            let t = w.slice(TERRAINS.len() + 1);
            for u in &tile.upgrades {
                if u.terrain.is_empty() {
                    continue;
                }
                match vocab_index(TERRAINS, &u.terrain) {
                    Some(ti) => t[ti] = 1.0,
                    None => t[TERRAINS.len()] = 1.0,
                }
            }
            for e in 0..6 {
                w.flag(spec.hex_walls[i][e]);
            }
            let bc = w.slice(6);
            if let Some(st) = self.g1822.as_deref() {
                for (edge, cost, _) in st.borders.get(coord).map(|v| v.as_slice()).unwrap_or(&[]) {
                    bc[*edge as usize % 6] = *cost as f32 / spec.max_lay_cost;
                }
            } else {
                for (edge, cost) in self.border_costs.get(coord).unwrap_or(&empty_borders) {
                    let paid = self.paid_borders.iter().any(|(h, e)| h == coord && e == edge);
                    if !paid {
                        bc[*edge as usize % 6] = *cost as f32 / spec.max_lay_cost;
                    }
                }
            }
            write_track_features(&tile.paths, w.slice(49));
            w.flag(blocked.contains(coord));
            w.put(bonuses.iter().filter(|b| b.0 == i).map(|b| b.1).sum::<i32>() as f32 / ms);
            w.put(n_tokens as f32 / 6.0);
            w.put(n_neutral as f32 / 6.0);
            w.put(n_extra as f32 / 2.0);
            w.put(n_reserved as f32 / 6.0);
            w.flag(laid.iter().any(|h| h == coord));
            // -- version 2: the 1822 family's map state (0 elsewhere) ---------------
            match (ctx.fam.as_ref(), self.g1822.as_deref()) {
                (Some(fs), Some(st)) => {
                    w.put(st.builder_cubes.get(&tile.id).copied().unwrap_or(0) as f32 / 2.0);
                    let (mut forest, mut n_assigned, mut bonus) = (false, 0usize, 0i32);
                    for (_, key) in fs.assignments.iter().filter(|a| a.0 == i) {
                        if key == "forest" {
                            forest = true;
                        } else {
                            n_assigned += 1;
                        }
                        bonus += self.rl_assignment_bonus(key, fs);
                    }
                    w.flag(forest);
                    w.put(n_assigned as f32);
                    w.put(bonus as f32 / ms);
                    let marks = spec.hex_marks[i];
                    w.flag(marks & MARK_GRAIN != 0);
                    w.flag(marks & MARK_PORT_WEST != 0);
                    w.flag(marks & MARK_PORT_EAST != 0);
                    w.put(n_dest as f32 / 2.0);
                    w.put(n_coal as f32 / 2.0);
                    let sale = fs.tokens_for_sale.iter().filter(|t| t.0 == i).map(|t| t.1).max();
                    w.put(sale.unwrap_or(0) as f32 / ctx.sc);
                    w.flag(fs.token_up == Some(i));
                }
                _ => {
                    w.slice(11);
                }
            }
            w.done(F_HEX);
        }
        reserved_corps
    }

    // -- entities ----------------------------------------------------------------

    fn rl_encode_entities(&self, ctx: &Ctx, holdings: &[Holdings], reserved: &[i32], out: &mut TokenEncoding) {
        let spec = ctx.spec;
        let sc = ctx.sc;
        let np = ctx.np;
        let title = self.title_def();
        let g1822 = self.g1822.as_deref();
        let phase_spec = (ctx.phase_idx >= 0).then(|| &spec.phases[ctx.phase_idx as usize]);
        let (major_limit, minor_limit) = match phase_spec {
            Some(p) => (p.train_limit, p.minor_train_limit.unwrap_or(p.train_limit)),
            None => (self.phase.train_limit, self.phase.train_limit),
        };

        // Operating order of the live OR (1822: the synced view).
        let (or_order, or_index): (&[String], usize) = match &self.round {
            Round::Operating(o) => (&o.operating_order, o.entity_index),
            _ => (&[], 0),
        };
        let acting = &self.round_state.active_entity_id;

        // Auction / bid state.
        let auction = match &self.round {
            Round::Auction(a) => Some(a),
            _ => None,
        };
        let stock = match &self.round {
            Round::Stock(s) if g1822.is_none() => Some(s),
            _ => None,
        };
        let bid_step = title.bid_price_step() as i32;
        // the 1822 family's bid boxes (minors, concessions, privates)
        let bidbox: Vec<&str> = ctx.fam.as_ref().map_or(Vec::new(), |fs| fs.bidbox.iter().map(|b| b.0.as_str()).collect());
        // Standing bids per company (game index): (player, price).
        let company_bids = |gi: usize| -> Vec<(u32, i32)> {
            if let Some(a) = auction {
                return a.bids.get(&gi).map(|b| b.iter().map(|x| (x.player_id, x.price)).collect()).unwrap_or_default();
            }
            if let Some(st) = g1822 {
                return st
                    .round
                    .bids
                    .get(&self.companies[gi].sym)
                    .map(|b| b.iter().map(|x| (x.player, x.price)).collect())
                    .unwrap_or_default();
            }
            Vec::new()
        };

        // -- per-player aggregates --------------------------------------------------
        let mut committed: HashMap<u32, i32> = HashMap::new();
        for gi in 0..self.companies.len() {
            for (pid, price) in company_bids(gi) {
                *committed.entry(pid).or_insert(0) += price;
            }
        }
        if let Some(ba) = stock.and_then(|s| s.bid_auction.as_ref()) {
            for b in &ba.bids {
                *committed.entry(b.player_id).or_insert(0) += b.price;
            }
        }
        let priority = self.rl_priority_deal_player();

        // -- global ------------------------------------------------------------------
        out.ent_type[0] = ENT_GLOBAL;
        {
            let mut w = W::new(&mut out.ent_feat[0..F_ENT]);
            w.put(self.bank.cash as f32 / spec.bank_cash);
            w.flag(self.bank_broken);
            match g1822 {
                // the family latches its own trigger: "current_or" ends the
                // game after this (or the next) OR, "full_or" after the set
                Some(st) => {
                    w.flag(st.game_end_trigger.is_some());
                    w.flag(st.game_end_trigger.as_ref().map_or(false, |(_, t)| t == "current_or"));
                }
                None => {
                    w.flag(self.game_end_triggered);
                    w.flag(self.final_turn.is_some());
                }
            }
            w.flag(self.finished);
            w.put(self.turn as f32 / 20.0);
            let (rn, tot) = match &self.round {
                Round::Operating(o) => (o.round_num as f32, o.total_ors as f32),
                Round::Merger(m) => (m.round_num as f32, m.total_ors as f32),
                _ => (0.0, self.phase.operating_rounds as f32),
            };
            w.ratio(rn, tot);
            w.put(tot / 3.0);
            w.ratio(ctx.phase_idx.max(0) as f32, (spec.phases.len() as f32 - 1.0).max(1.0));
            w.put(np as f32 / MAX_SEATS as f32);
            w.put(self.cert_limit as f32 / 30.0);
            w.put(sc / spec.bank_cash);
            w.ratio(self.loans_remaining as f32, spec.num_loans);
            w.put(spec.loan_value / sc);
            w.ratio(spec.loan_interest, spec.loan_value);
            w.put(spec.market_pool_limit / 100.0);
            let (laid_n, tokened, revenue) = match (g1822, &self.round) {
                (Some(st), _) => (
                    st.round.num_laid_track as f32,
                    st.round.tokened as u8 as f32,
                    st.round.routes.iter().map(|r| r.revenue).sum::<i32>() as f32,
                ),
                (None, Round::Operating(o)) => (o.num_laid_track as f32, o.num_placed_token as f32, o.revenue as f32),
                _ => (0.0, 0.0, 0.0),
            };
            w.put(laid_n / 2.0);
            w.put(tokened);
            w.put(revenue / sc);
            w.put(auction.map_or(0.0, |a| a.discount as f32 / sc));
            let lot_running = auction.map_or(false, |a| a.auctioning.is_some())
                || stock.map_or(false, |s| s.bid_auction.is_some());
            w.flag(lot_running);
            let (sold, bought, parred) = match (g1822, stock) {
                (Some(st), _) => {
                    use crate::g1822::CurAct;
                    let ca = &st.round.current_actions;
                    (ca.contains(&CurAct::SellShares), ca.contains(&CurAct::BuyShares), ca.contains(&CurAct::Par))
                }
                (None, Some(s)) => {
                    let pid = s.current_player_id();
                    let sold = s
                        .players_sold
                        .get(&pid)
                        .map_or(false, |m| m.values().any(|t| *t == SoldTiming::Now));
                    (sold, s.bought_this_turn, s.parred_this_turn)
                }
                _ => (false, false, false),
            };
            w.flag(sold);
            w.flag(bought);
            w.flag(parred);
            w.put(stock.map_or(0.0, |s| s.consecutive_passes as f32 / np.max(1) as f32));
            w.put(self.events_fired.len() as f32 / 8.0);
            w.put(self.trainless_major.len() as f32 / 4.0);
            let pending = match (g1822, &self.round) {
                (Some(st), _) => st.round.pending_tokens.len(),
                (None, Round::Stock(s)) => s.pending_home_tokens.len(),
                (None, Round::Operating(o)) => o.pending_tokens.len(),
                _ => 0,
            };
            w.put(pending as f32 / 2.0);
            let crowded = match (g1822, &self.round) {
                (Some(_), _) => self.g1822_crowded_corps().len(),
                (None, Round::Operating(o)) => o.crowded_corps.len(),
                _ => 0,
            };
            w.put(crowded as f32 / 2.0);
            // -- version 2: the 1822 family (0 elsewhere) ------------------------------
            match g1822 {
                Some(st) => {
                    let f = self.fam();
                    let r = &st.round;
                    let stock = r.kind == crate::g1822::RoundKind::Stock;
                    w.flag(stock && st.nothing_sold_in_sr);
                    w.put(f.player_loan_interest_rate as f32 / 100.0);
                    w.put((f.bidding_tokens)(np) as f32 / 10.0);
                    w.ratio(if stock { r.bid_actions as f32 } else { 0.0 }, f.bidding_tokens_per_action as f32);
                    w.flag(r.upgraded_track);
                    w.flag(r.pullman_train.is_some());
                    w.flag(r.ca.grain_train.is_some());
                    w.put((r.ca.skipped_pullman as u8 + r.ca.skipped_grain as u8) as f32 / 2.0);
                }
                None => {
                    w.slice(8);
                }
            }
            // -- version 3: 1822's optional rules (0 elsewhere) ------------------------
            if self.title == "1822" {
                let has = |r: &str| self.optional_rules.iter().any(|x| x == r);
                for (_, rule, needs) in RULE_FLAGS {
                    w.flag(has(rule) && needs.map_or(true, |n| has(n)));
                }
            } else {
                w.slice(RULE_FLAGS.len());
            }
            w.done(GLOBAL_FEATURES.len());
        }
        let lot = if let Some(a) = auction {
            match a.auctioning.or_else(|| if a.single_item.is_some() { None } else { a.cheapest_company() }) {
                Some(ci) if ci < self.companies.len() => ctx.company_ent[ci],
                _ => PTR_NONE,
            }
        } else if let Some(ba) = stock.and_then(|s| s.bid_auction.as_ref()) {
            ctx.corp_ent_sym(&ba.corporation_sym)
        } else if let Some(fs) = ctx.fam.as_ref() {
            // the 1822 family: the entity the decision is about
            match (fs.subject_corps.first(), fs.subject_companies.first()) {
                (Some(c), _) => ctx.corp_ent_sym(c),
                (None, Some(co)) => ctx.company_ent_sym(co),
                _ => PTR_NONE,
            }
        } else {
            PTR_NONE
        };
        out.ent_ptr[0..P_ENT].copy_from_slice(&[
            if self.finished { PTR_NONE } else { ctx.owner_ent(acting) },
            ctx.seat_ent(priority),
            if ctx.phase_idx >= 0 { (spec.e_phase0 + ctx.phase_idx as usize) as i32 } else { PTR_NONE },
            lot,
        ]);

        // -- seats ---------------------------------------------------------------------
        let presidents: Vec<Option<u32>> = self.corporations.iter().map(|c| c.president_id()).collect();
        for seat in 0..np.min(MAX_SEATS) {
            let pid = ctx.seat_ids[(seat + ctx.rotation) % np];
            let Some(player) = self.players.iter().find(|p| p.id == pid) else { continue };
            let ent = RlSpec::seat_entity(seat);
            out.ent_type[ent] = ENT_PLAYER;
            let pid_eid = EntityId::player(pid);
            let company_certs = self.companies.iter().filter(|c| !c.closed && c.owner == pid_eid).count();
            let certs = (holdings.iter().map(|h| h.seat_certs[seat] as usize).sum::<usize>() + company_certs) as f32;
            let cl = (self.cert_limit as f32).max(1.0);
            let share_value: f32 = self
                .corporations
                .iter()
                .zip(holdings)
                .filter_map(|(c, h)| {
                    let pct = h.seat_pct[seat] as i32;
                    let price = c.share_price.as_ref()?.price;
                    (pct > 0).then(|| (pct * price / c.share_unit().max(1)) as f32)
                })
                .sum();
            let company_value: f32 = self
                .companies
                .iter()
                .filter(|c| !c.closed && c.owner == pid_eid)
                .map(|c| c.value as f32)
                .sum();
            let order_pos = match (g1822, &self.round) {
                (Some(st), _) if matches!(st.round.kind, crate::g1822::RoundKind::Stock | crate::g1822::RoundKind::Choices) => {
                    let ents = &st.round.entities;
                    ents.iter().position(|e| e.parse::<u32>().ok() == Some(pid)).map(|p| {
                        ((p + ents.len() - st.round.entity_index.min(ents.len())) % ents.len().max(1)) as f32
                    })
                }
                (None, Round::Stock(s)) => s.player_order.iter().position(|&x| x == pid).map(|p| p as f32),
                (None, Round::Auction(a)) => a.player_order.iter().position(|&x| x == pid).map(|p| {
                    ((p + a.player_order.len() - a.entity_index % a.player_order.len().max(1)) % a.player_order.len().max(1)) as f32
                }),
                _ => None,
            };
            let passed = match (g1822, &self.round) {
                (Some(st), _) => st.round_passed_players.contains(&pid),
                (None, Round::Stock(s)) => {
                    s.player_order.iter().position(|&x| x == pid).map_or(false, |i| s.player_passed.get(i).copied().unwrap_or(false))
                }
                (None, Round::Auction(a)) => match &a.single_item {
                    Some(si) => si.declined.contains(&pid),
                    None => a.player_order.iter().position(|&x| x == pid).map_or(false, |i| a.passed.get(i).copied().unwrap_or(false)),
                },
                (None, Round::Merger(m)) => m.passed_players.contains(&pid),
                _ => false,
            };
            let sold_corps = match (g1822, stock) {
                (Some(st), _) => st.round.players_sold.get(&pid).map_or(0, |m| m.len()),
                (None, Some(s)) => s.players_sold.get(&pid).map_or(0, |m| m.len()),
                _ => 0,
            };
            let debt = if g1822.is_some() { self.player_debt(pid) as f32 } else { 0.0 };
            // the 1822 family: its own cert count (P16 / P11 -1, the NdeM and
            // owner-only companies exempt), Ruby's committed cash (winning
            // bids only) and its score (debt off, the tax haven on)
            let (certs, committed_v, net_worth) = match g1822 {
                Some(_) => (
                    self.g1822_num_certs(pid) as f32,
                    self.g1822_committed_cash(pid),
                    self.g1822_player_value(pid) as f32,
                ),
                None => (
                    certs,
                    committed.get(&pid).copied().unwrap_or(0),
                    player.cash as f32 + share_value + company_value,
                ),
            };
            let mut w = W::new(&mut out.ent_feat[ent * F_ENT..(ent + 1) * F_ENT]);
            w.put(1.0);
            w.flag(seat == 0 && ctx.mover.is_some());
            w.put(player.cash as f32 / sc);
            w.put(certs / cl);
            w.put((cl - certs) / cl);
            w.flag(pid == priority);
            w.put(order_pos.map_or(0.0, |p| p / ((np as f32 - 1.0).max(1.0))));
            w.put(committed_v as f32 / sc);
            w.put(net_worth / sc / 4.0);
            w.put(share_value / sc / 4.0);
            w.put(company_value / sc);
            w.flag(player.bankrupt);
            w.flag(passed);
            w.put(presidents.iter().filter(|p| **p == Some(pid)).count() as f32 / 4.0);
            w.put(sold_corps as f32 / 4.0);
            w.put(debt / sc);
            // -- version 2: the 1822 family (0 elsewhere) ------------------------------
            match (g1822, ctx.fam.as_ref()) {
                (Some(st), Some(fs)) => {
                    let tokens = (self.fam().bidding_tokens)(np) as f32;
                    let my_bids = st.round.bids.values().filter(|bs| bs.iter().any(|b| b.player == pid)).count() as f32;
                    w.ratio(tokens - my_bids, tokens);
                    w.put(self.g1822_num_certs_with_bids(pid) as f32 / cl);
                    w.flag(st.double_cash_choice == Some(pid));
                    let q = fs.queue.iter().position(|&p| p == pid);
                    w.flag(q.is_some());
                    w.put(q.map_or(0.0, |p| p as f32 / ((np as f32 - 1.0).max(1.0))));
                    let osth = self.fam().company_osth;
                    let haven = !osth.is_empty() && !self.co_closed(osth) && self.co_owner(osth) == pid_eid;
                    w.put(if haven { self.g1822_tax_haven_value() as f32 / sc } else { 0.0 });
                }
                _ => {
                    w.slice(6);
                }
            }
            w.done(PLAYER_FEATURES.len());
        }

        // -- corporations -----------------------------------------------------------
        for (gi, corp) in self.corporations.iter().enumerate() {
            let ent = ctx.corp_ent[gi];
            if ent < 0 {
                continue;
            }
            let ent = ent as usize;
            let cs = &spec.corps[ctx.corp_spec[gi] as usize];
            out.ent_type[ent] = cs.ent_type;
            let h = &holdings[gi];
            let unit = corp.share_unit_percent.max(1) as f32;
            let units = corp.num_share_units();
            let total_units = units.max(1) as f32;
            let is_minor = corp.corp_type == crate::title::CorpType::Minor;
            let limit = if is_minor { minor_limit } else { major_limit } as f32;
            let max_loans = match corp.corp_type {
                crate::title::CorpType::Major => spec.max_loans_major,
                crate::title::CorpType::Minor => spec.max_loans_minor,
                crate::title::CorpType::National => 0.0,
            };
            let or_pos = or_order.iter().position(|s| *s == corp.sym);
            let operated = corp.ever_operated
                || g1822.map_or(false, |st| st.operating_history.get(&corp.sym).copied().unwrap_or(0) > 0);
            let mut w = W::new(&mut out.ent_feat[ent * F_ENT..(ent + 1) * F_ENT]);
            w.flag(corp.ipo_price.is_some() || corp.share_price.is_some());
            w.flag(corp.floated);
            w.flag(operated);
            w.flag(corp.closed);
            w.put(corp.cash as f32 / sc);
            match &corp.share_price {
                Some(sp) => {
                    w.put(sp.price as f32 / spec.max_market_price);
                    w.put(corp.ipo_price.as_ref().map_or(0, |p| p.price) as f32 / spec.max_market_price);
                    w.put(sp.row as f32 / spec.market_rows);
                    w.put(sp.column as f32 / spec.market_cols);
                    let z = w.slice(MARKET_ZONES.len());
                    for t in &sp.types {
                        if let Some(zi) = vocab_index(MARKET_ZONES, t) {
                            z[zi] = 1.0;
                        }
                    }
                }
                None => {
                    w.put(0.0);
                    w.put(corp.ipo_price.as_ref().map_or(0, |p| p.price) as f32 / spec.max_market_price);
                    w.slice(2 + MARKET_ZONES.len());
                }
            }
            w.put(h.ipo_pct as f32 / unit / total_units);
            w.put(h.market_pct as f32 / unit / total_units);
            w.put(h.other_pct as f32 / unit / total_units);
            w.put(units as f32 / 10.0);
            w.put(if units > 0 { corp.president_share_units() as f32 / total_units } else { 0.0 });
            w.put(unit / 100.0);
            w.put(corp.float_percent as f32 / 100.0);
            w.flag(corp.capitalization == crate::title::Capitalization::Incremental);
            let n_tok = corp.tokens.len() as f32;
            w.put(n_tok / 8.0);
            w.ratio(corp.tokens.iter().filter(|t| t.used).count() as f32, n_tok);
            w.put(corp.next_token_price().unwrap_or(0) as f32 / spec.max_token_price);
            w.ratio(corp.loans as f32, max_loans);
            w.put(max_loans / 10.0);
            let n_trains = corp.trains.len() as f32;
            w.put(n_trains / 4.0);
            w.put((limit - n_trains) / 4.0);
            w.put(limit / 4.0);
            w.put(or_pos.map_or(0.0, |p| p as f32 / ((or_order.len() as f32 - 1.0).max(1.0))));
            w.flag(or_pos.is_some());
            w.flag(or_pos.map_or(false, |p| p < or_index));
            w.flag(acting.0.strip_prefix("corp:") == Some(corp.sym.as_str()));
            w.flag(presidents[gi].is_some());
            w.put(corp.max_ownership_percent as f32 / 100.0);
            w.flag(corp.always_market_price);
            w.flag(title.corporation_startable(&corp.sym, &self.phase.name));
            w.flag(reserved.contains(&(ent as i32)));
            w.flag(g1822.map_or(false, |st| st.destination_ability.iter().any(|s| *s == corp.sym)));
            w.put(g1822.map_or(0, |st| st.exchange_tokens.get(&corp.sym).copied().unwrap_or(0)) as f32 / 4.0);
            for s in 0..MAX_SEATS {
                w.put(h.seat_pct[s] as f32 / 100.0);
            }
            w.put(h.market_pct as f32 / 100.0);
            w.put(h.ipo_pct as f32 / 100.0);
            // -- version 2: the 1822 family (0 elsewhere) ------------------------------
            match (g1822, ctx.fam.as_ref()) {
                (Some(st), Some(fs)) => {
                    let sym = corp.sym.as_str();
                    w.put(st.regional_payout_count.get(sym).copied().unwrap_or(0) as f32 / 2.0);
                    w.flag(self.g1822_mx_is_ndem(sym) && st.ndem_state == crate::g1822::mx::NdemState::Closing);
                    w.flag(fs.subject_corps.iter().any(|s| s == sym));
                    w.put(st.corp_token_abilities.get(sym).map_or(0, |v| v.len()) as f32 / 2.0);
                    w.flag(st.round.pending_tokens.iter().any(|p| p.corp == sym));
                }
                _ => {
                    w.slice(5);
                }
            }
            w.done(CORPORATION_FEATURES.len());

            // pointers
            let link = if let Some(st) = g1822 {
                st.minor_associations
                    .iter()
                    .find(|(m, _)| *m == corp.sym)
                    .map(|(_, major)| ctx.corp_ent_sym(major))
                    .or_else(|| cs.concession.as_deref().map(|co| ctx.company_ent_sym(co)))
                    .unwrap_or(PTR_NONE)
            } else {
                PTR_NONE
            };
            out.ent_ptr[ent * P_ENT] = presidents[gi].map_or(PTR_NONE, |p| ctx.seat_ent(p));
            out.ent_ptr[ent * P_ENT + 1] = link;
            let home = g1822
                .and_then(|st| st.home_override.get(&corp.sym))
                .and_then(|h| spec.hex_index.get(h.as_str()).map(|&i| i as i32))
                .unwrap_or(cs.home_hex);
            let home = if home >= 0 {
                home
            } else {
                corp.tokens
                    .first()
                    .filter(|t| t.used)
                    .and_then(|t| spec.hex_index.get(t.city_hex_id.as_str()))
                    .map_or(PTR_NONE, |&i| i as i32)
            };
            out.ent_hex[ent * P_HEX] = home;
            out.ent_hex[ent * P_HEX + 1] = if g1822.is_some() {
                self.g1822_destination(&corp.sym)
                    .and_then(|d| spec.hex_index.get(d.as_str()).map(|&i| i as i32))
                    .unwrap_or(PTR_NONE)
            } else {
                PTR_NONE
            };
        }

        // -- companies ----------------------------------------------------------------
        for (gi, co) in self.companies.iter().enumerate() {
            let ent = ctx.company_ent[gi];
            if ent < 0 {
                continue;
            }
            let ent = ent as usize;
            let cs = &spec.companies[ent - spec.e_company0];
            out.ent_type[ent] = cs.ent_type;
            let owner = ctx.owner_ent(&co.owner);
            let owner_seat = co.owner.is_player();
            let owner_corp = co.owner.is_corporation();
            let bids = company_bids(gi);
            let high = bids.iter().map(|b| b.1).max();
            let up = if let Some(a) = auction {
                match &a.single_item {
                    Some(_) => a.auctioning == Some(gi),
                    None => a.remaining_companies.contains(&gi),
                }
            } else {
                bidbox.iter().any(|s| *s == co.sym)
            };
            let min_bid = if !up {
                0
            } else if let (Some(a), None) = (auction, auction.and_then(|a| a.single_item.as_ref())) {
                a.min_bid_for(gi, co.value)
            } else {
                match high {
                    Some(hb) => hb + bid_step,
                    None => co.min_bid(),
                }
            };
            let revenue = g1822
                .and_then(|st| st.company_revenue.get(&co.sym).copied())
                .unwrap_or(co.revenue);
            let ability_used = co.ability_used
                || g1822.map_or(false, |st| st.abilities.get(&co.sym).map_or(false, |v| !v.is_empty() && v.iter().all(|a| a.used)));
            let mut w = W::new(&mut out.ent_feat[ent * F_ENT..(ent + 1) * F_ENT]);
            w.put(co.value as f32 / sc);
            w.put(revenue as f32 / spec.max_company_revenue);
            w.flag(co.closed);
            w.flag(owner_seat);
            w.flag(owner_corp);
            w.flag(!owner_seat && !owner_corp);
            w.flag(cs.auctionable);
            w.flag(up);
            w.put(min_bid as f32 / sc);
            w.put(high.unwrap_or(0) as f32 / sc);
            w.put(bids.len() as f32 / np.max(1) as f32);
            w.put(co.discount as f32 / sc);
            w.flag(co.no_buy);
            w.flag(ability_used);
            for a in cs.abilities {
                w.flag(a);
            }
            let seat_bids = w.slice(MAX_SEATS);
            for (pid, price) in &bids {
                if let Some(s) = ctx.seat_of(*pid) {
                    seat_bids[s] = *price as f32 / sc;
                }
            }
            // -- version 2: the 1822 family (0 elsewhere) ------------------------------
            let mut owner = owner;
            let mut ability_hex = cs.ability_hex;
            match (g1822, ctx.fam.as_ref()) {
                (Some(st), Some(fs)) => {
                    let sym = co.sym.as_str();
                    let f = self.fam();
                    let haven = !f.company_osth.is_empty() && sym == f.company_osth;
                    let cash = if haven {
                        st.tax_haven_cash
                    } else {
                        st.phase_revenue.get(sym).copied().flatten().unwrap_or(0)
                    };
                    w.put(cash as f32 / sc);
                    w.put(if haven { (self.g1822_tax_haven_value() - st.tax_haven_cash) as f32 / sc } else { 0.0 });
                    match fs.bidbox.iter().find(|b| b.0 == sym) {
                        Some((_, k, size, first)) => {
                            w.put((k + 1) as f32 / *size as f32);
                            w.flag(*first);
                        }
                        None => {
                            w.slice(2);
                        }
                    }
                    let abilities = st.abilities.get(sym).map(|v| v.as_slice()).unwrap_or(&[]);
                    let mut uses: i32 = abilities.iter().filter(|a| !a.used).map(|a| a.count.unwrap_or(1).max(0)).sum();
                    if sym == f.company_lcdr && st.lcdr_token_ability {
                        uses += 1;
                    }
                    w.put(uses as f32 / 3.0);
                    let per_or: i32 = abilities.iter().filter_map(|a| a.count_per_or).sum();
                    let this_or: i32 = abilities.iter().filter(|a| a.count_per_or.is_some()).map(|a| a.count_this_or).sum();
                    w.ratio(this_or as f32, per_or as f32);
                    let assigned = fs.assignments.iter().find(|a| a.1 == sym);
                    w.flag(assigned.is_some());
                    if let Some((hi, _)) = assigned {
                        ability_hex = *hi as i32;
                    }
                    w.flag(fs.subject_companies.iter().any(|s| s == sym));
                    w.flag(st.company_trains.contains_key(sym));
                    // 1822CA: a closed sawmill pays its bonus to the corporation that assigned it
                    if self.g1822_is_ca() && sym == crate::g1822::ca::COMPANY_SAWMILL {
                        if let Some(o) = &st.ca.sawmill_owner {
                            owner = ctx.corp_ent_sym(o);
                        }
                    }
                }
                _ => {
                    w.slice(9);
                }
            }
            w.done(COMPANY_FEATURES.len());
            out.ent_ptr[ent * P_ENT] = owner;
            out.ent_ptr[ent * P_ENT + 1] = cs.link.as_deref().map_or(PTR_NONE, |s| ctx.corp_ent_sym(s));
            out.ent_hex[ent * P_HEX] = ability_hex;
        }

        // -- train types --------------------------------------------------------------
        let nt = spec.trains.len();
        let mut depot = vec![0u32; nt];
        let mut pool = vec![0u32; nt];
        let mut held = vec![0u32; nt];
        let tidx = |name: &str| spec.train_index.get(name).copied();
        for t in &self.depot.trains {
            if let Some(i) = tidx(&t.name) {
                depot[i] += 1;
            }
        }
        for t in &self.depot.discarded {
            if let Some(i) = tidx(&t.name) {
                pool[i] += 1;
            }
        }
        for c in &self.corporations {
            for t in &c.trains {
                if let Some(i) = tidx(&t.name) {
                    held[i] += 1;
                }
            }
        }
        let next_up = self.depot.trains.first().and_then(|t| tidx(&t.name));
        let ln21 = (21.0f32).ln();
        for (ti, ts) in spec.trains.iter().enumerate() {
            let ent = spec.e_train0 + ti;
            out.ent_type[ent] = ENT_TRAIN;
            let unlimited = ts.count == u32::MAX;
            let reached = |p: i32| p >= 0 && ctx.phase_idx >= p;
            let mut w = W::new(&mut out.ent_feat[ent * F_ENT..(ent + 1) * F_ENT]);
            if ts.distance >= 99 {
                w.put(1.0);
                w.put(1.0);
            } else {
                w.put(ts.distance.min(8) as f32 / 8.0);
                w.put(0.0);
            }
            w.put(ts.multiplier as f32 / 2.0);
            w.put(ts.price as f32 / spec.max_train_price);
            let d = depot[ti] as f32;
            w.put(if unlimited { (d > 0.0) as u8 as f32 } else if ts.count > 0 { d / ts.count as f32 } else { 0.0 });
            w.put((1.0 + d).ln() / ln21);
            w.put(pool[ti] as f32 / 4.0);
            w.flag(unlimited);
            w.put(if unlimited { 1.0 } else { ts.count as f32 / 10.0 });
            w.put(held[ti] as f32 / 8.0);
            w.flag(reached(ts.rusts_on));
            w.flag(reached(ts.obsolete_on));
            w.flag(next_up == Some(ti) || (reached(ts.available_on) && depot[ti] > 0));
            w.flag(next_up == Some(ti));
            w.ratio(ts.max_discount as f32, ts.price as f32);
            w.put(ts.n_events as f32 / 3.0);
            w.flag(ts.count == 0);
            w.done(TRAIN_FEATURES.len());
            let pe = |p: i32| if p >= 0 { (spec.e_phase0 + p as usize) as i32 } else { PTR_NONE };
            out.ent_ptr[ent * P_ENT + 2] = pe(ts.rusts_on);
            out.ent_ptr[ent * P_ENT + 3] = if ts.obsolete_on >= 0 { pe(ts.obsolete_on) } else { pe(ts.available_on) };
        }

        // -- phases ---------------------------------------------------------------------
        let n_ph = spec.phases.len();
        for (pi, ps) in spec.phases.iter().enumerate() {
            let ent = spec.e_phase0 + pi;
            out.ent_type[ent] = ENT_PHASE;
            let mut w = W::new(&mut out.ent_feat[ent * F_ENT..(ent + 1) * F_ENT]);
            w.flag(pi as i32 == ctx.phase_idx);
            w.flag((pi as i32) < ctx.phase_idx);
            w.ratio(pi as f32, (n_ph as f32 - 1.0).max(1.0));
            w.put(ps.train_limit as f32 / 4.0);
            w.put(ps.minor_train_limit.unwrap_or(ps.train_limit) as f32 / 4.0);
            w.put(ps.operating_rounds as f32 / 3.0);
            for c in ps.colors {
                w.flag(c);
            }
            let st = w.slice(PHASE_STATUSES.len() + 1);
            for &s in &ps.statuses {
                if s >= 0 {
                    st[s as usize] = 1.0;
                } else {
                    st[PHASE_STATUSES.len()] = 1.0;
                }
            }
            w.done(PHASE_FEATURES.len());
            if ps.on_train >= 0 {
                out.ent_ptr[ent * P_ENT + 2] = (spec.e_train0 + ps.on_train as usize) as i32;
            }
        }
    }

    // -- relations ---------------------------------------------------------------------

    fn rl_encode_relations(&self, ctx: &Ctx, holdings: &[Holdings], out: &mut TokenEncoding) {
        let spec = ctx.spec;
        let sc = ctx.sc;
        let mut push = |src: i32, dst: i32, kind: i32, val: f32| {
            if src >= 0 && dst >= 0 {
                out.rel.extend_from_slice(&[src, dst, kind]);
                out.rel_val.push(val);
            }
        };
        for (gi, corp) in self.corporations.iter().enumerate() {
            let ce = ctx.corp_ent[gi];
            if ce < 0 {
                continue;
            }
            let h = &holdings[gi];
            for s in 0..ctx.np.min(MAX_SEATS) {
                if h.seat_pct[s] > 0 {
                    push(RlSpec::seat_entity(s) as i32, ce, REL_SHARES, h.seat_pct[s] as f32 / 100.0);
                }
            }
            for &(holder, pct) in h.corp_holders.iter().chain(&h.company_holders) {
                push(holder, ce, REL_SHARES, pct as f32 / 100.0);
            }
            if let Some(p) = corp.president_id() {
                push(ctx.seat_ent(p), ce, REL_PRESIDENT, 1.0);
            }
            // trains held, by type in spec order
            let mut counts: Vec<(usize, u32)> = Vec::new();
            for t in &corp.trains {
                if let Some(&ti) = spec.train_index.get(t.name.as_str()) {
                    match counts.iter_mut().find(|(i, _)| *i == ti) {
                        Some(x) => x.1 += 1,
                        None => counts.push((ti, 1)),
                    }
                }
            }
            counts.sort_unstable();
            for (ti, n) in counts {
                push(ce, (spec.e_train0 + ti) as i32, REL_TRAINS, n as f32 / 4.0);
            }
        }
        for (gi, co) in self.companies.iter().enumerate() {
            let ent = ctx.company_ent[gi];
            if co.closed || ent < 0 {
                continue;
            }
            push(ctx.owner_ent(&co.owner), ent, REL_OWNS, 1.0);
        }
        // the 1822 family's train privates: the train each reserves
        if let Some(st) = self.g1822.as_deref() {
            for (gi, co) in self.companies.iter().enumerate() {
                let ent = ctx.company_ent[gi];
                if co.closed || ent < 0 {
                    continue;
                }
                if let Some(&ti) = st.company_trains.get(&co.sym).and_then(|t| spec.train_index.get(t.name.as_str())) {
                    push(ent, (spec.e_train0 + ti) as i32, REL_TRAINS, 1.0 / 4.0);
                }
            }
        }
        // standing bids
        match (&self.round, self.g1822.as_deref()) {
            (_, Some(st)) => {
                for (gi, co) in self.companies.iter().enumerate() {
                    if let Some(bids) = st.round.bids.get(&co.sym) {
                        for b in bids {
                            push(ctx.seat_ent(b.player), ctx.company_ent[gi], REL_BID, b.price as f32 / sc);
                        }
                    }
                }
                // 1822MX: the high bid for the NdeM token up for auction
                if let Some((p, price)) = ctx.fam.as_ref().and_then(|fs| fs.high_bid) {
                    push(ctx.seat_ent(p), ctx.corp_ent_sym(crate::g1822::mx::NDEM), REL_BID, price as f32 / sc);
                }
            }
            (Round::Auction(a), None) => {
                for gi in 0..self.companies.len() {
                    if let Some(bids) = a.bids.get(&gi) {
                        for b in bids {
                            push(ctx.seat_ent(b.player_id), ctx.company_ent[gi], REL_BID, b.price as f32 / sc);
                        }
                    }
                }
            }
            (Round::Stock(s), None) => {
                if let Some(ba) = &s.bid_auction {
                    let ce = ctx.corp_ent_sym(&ba.corporation_sym);
                    for b in &ba.bids {
                        push(ctx.seat_ent(b.player_id), ce, REL_BID, b.price as f32 / sc);
                    }
                }
            }
            _ => {}
        }
        // sales this stock round (seat order, then corporation order)
        for s in 0..ctx.np.min(MAX_SEATS) {
            let pid = ctx.seat_ids[(s + ctx.rotation) % ctx.np];
            let seat = RlSpec::seat_entity(s) as i32;
            match (&self.round, self.g1822.as_deref()) {
                (_, Some(st)) => {
                    if let Some(m) = st.round.players_sold.get(&pid) {
                        for (gi, corp) in self.corporations.iter().enumerate() {
                            if let Some(when) = m.get(&corp.sym) {
                                let v = if *when == crate::g1822::SoldWhen::Now { 1.0 } else { 0.5 };
                                push(seat, ctx.corp_ent[gi], REL_SOLD, v);
                            }
                        }
                    }
                }
                (Round::Stock(st), None) => {
                    if let Some(m) = st.players_sold.get(&pid) {
                        for (gi, corp) in self.corporations.iter().enumerate() {
                            if let Some(when) = m.get(&corp.sym) {
                                let v = if *when == SoldTiming::Now { 1.0 } else { 0.5 };
                                push(seat, ctx.corp_ent[gi], REL_SOLD, v);
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }
}
