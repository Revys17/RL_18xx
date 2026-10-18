//! 1822 (and, later, its `G1822::Game` subclasses) — a Ruby-faithful step
//! machine over the shared `BaseGame` state.
//!
//! Why a separate machine: 1822's rounds differ from 1830/1867 at almost
//! every step boundary (bid boxes in the stock round, a choices round, a
//! 16-step operating round with first-turn housekeeping, company
//! acquisition, special track/token/choose steps, destination tokens,
//! minor acquisition and share issuing), and replay fidelity depends on
//! exactly WHICH step consumes each recorded action (a `pass` belongs to
//! the first blocking step). Rather than flattening those steps into the
//! shared program-counter OR, this module mirrors Ruby's
//! `Round::Base#process_action` / `skip_steps` / `after_process` with per-
//! step `passed` flags and per-step state, so each Ruby step ports as one
//! unit (`actions`, `active?`, `blocks?`, `process_*`, `skip!`, `pass!`).
//!
//! Shared state lives where the other titles keep it (players, corps,
//! companies, hexes, depot, stock market on `BaseGame`); the 1822-only
//! state (bid boxes, stack order, tax haven, phase-revenue credits,
//! exchange tokens, player debt, the live round's step states) lives in
//! [`G1822State`] on `BaseGame::g1822`. Shared code reaches this module
//! through a handful of title-gated hooks (`process_action_internal`,
//! `step_action_types_impl`, `home_reservations`, the constructor) keyed on
//! `BaseGame::g1822.is_some()` — 1830/1867 never take them.

pub mod acquisition;
pub mod family;
pub mod graph;
pub mod legal;
#[cfg(test)]
mod fuzz;
pub mod mx;
pub mod mx_legal;
pub mod operating;
pub mod pnw;
pub mod pnw_legal;
pub mod py;
pub mod router;
pub mod shares;
pub mod stock;
pub mod tile_rules;
pub mod tokens;
pub mod track;
pub mod trains;

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::actions::{Action, GameError};
use crate::entities::{EntityId, Token, Train};
use crate::game::BaseGame;
use crate::title::CorpType;

// ---------------------------------------------------------------------------
// Ruby's game RNG (base.rb initialize_seed / rand): a 31-bit LCG.
// ---------------------------------------------------------------------------

pub const RAND_A: u64 = 1_103_515_245;
pub const RAND_C: u64 = 12_345;
pub const RAND_M: u64 = 1 << 31;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RubyRand {
    pub state: u64,
}

impl RubyRand {
    pub fn new(seed: u64) -> Self {
        RubyRand { state: seed % RAND_M }
    }
    /// `rand` — advances and returns the new state.
    pub fn next(&mut self) -> u64 {
        self.state = (RAND_A * self.state + RAND_C) % RAND_M;
        self.state
    }
}

// ---------------------------------------------------------------------------
// Constants: Ruby's `self.class::` constants live in the title's family
// table (`family::Family`, reached through `BaseGame::fam()`).
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

/// Ruby step classes used by the 1822 rounds (round lists in game.rb
/// `stock_round`, `choose_step`, `operating_round`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Step {
    DiscardTrain,
    BuySellParShares,
    Choose,
    PendingToken,
    FirstTurnHousekeeping,
    AcquireCompany,
    SpecialChoose,
    SpecialTrack,
    SpecialToken,
    Track,
    DestinationToken,
    Token,
    Route,
    Dividend,
    BuyTrain,
    MinorAcquisition,
    IssueShares,
    /// 1822PNW: G1822PNW::Step::Assign (P15 paper mill, P17 ski haus).
    Assign,
    /// 1822PNW: the Merger round's G1822PNW::Step::Merge.
    Merge,
    /// 1822MX: G1822MX::Step::CashOutNdem.
    CashOutNdem,
    /// 1822MX: G1822MX::Step::AuctionNdemTokens.
    AuctionNdemTokens,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RoundKind {
    Stock,
    Choices,
    Operating,
    /// 1822PNW: G1822PNW::Round::Merger (after each OR from phase 2).
    Merger,
}

/// 1822PNW Merge `@merge_state`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum MergeState {
    #[default]
    None,
    SelectingPar,
    SelectingShares,
    SelectingToken,
}

/// Ruby `players_sold[player][corp]` values.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SoldWhen {
    Now,
    Prev,
}

/// The kinds of action Ruby keeps in `@round.current_actions` (BSPS).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CurAct {
    Bid,
    BuyShares,
    Par,
    SellShares,
    Choose,
    /// payoff_player_debt(_partial): Ruby appends these too, so a player
    /// who pays debt then passes is NOT passed (227938).
    Payoff,
}

/// One standing bid (Ruby Action::Bid kept in `@bids[company]`).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BidRec {
    pub player: u32,
    pub price: i32,
}

/// A pending home-token placement (Ruby `@round.pending_tokens` entry).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PendingTok {
    pub corp: String,
    pub hexes: Vec<String>,
    /// Index into the corp's `tokens`.
    pub token_idx: usize,
}

/// A recorded route, as kept for the dividend step (Ruby `@round.routes`).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RunRoute {
    pub train_id: String,
    pub revenue: i32,
    pub subsidy: i32,
    pub connections: Vec<Vec<String>>,
    pub hexes: Vec<String>,
    pub nodes: Vec<String>,
}

/// MinorAcquisition `@acquire_state`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AcquireState {
    SelectMinor,
    ChoosePay,
    ChooseToken,
}

/// The live round (Ruby `@round`): entities, entity index, steps with
/// their `passed` / `acted` flags, and the round_state ivars the steps
/// declare.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RoundSt {
    pub kind: RoundKind,
    pub round_num: u8,
    /// Player ids (stock / choices) or corporation syms (operating), as
    /// strings.
    pub entities: Vec<String>,
    pub entity_index: usize,
    pub steps: Vec<Step>,
    pub passed: Vec<bool>,
    pub acted: Vec<bool>,

    // -- stock (BuySellParShares round_state + G1822 additions) --
    pub players_sold: BTreeMap<u32, BTreeMap<String, SoldWhen>>,
    pub players_bought: BTreeMap<u32, BTreeMap<String, u32>>,
    pub current_actions: Vec<CurAct>,
    pub bought_from_ipo: bool,
    pub last_to_act: Option<u32>,
    pub pass_order: Vec<u32>,
    pub tax_haven_bought: bool,
    /// Bids per bid-box company, in placement order (Ruby `@round.bids`).
    pub bids: BTreeMap<String, Vec<BidRec>>,
    /// BSPS `@bid_actions` (reset each turn by `setup`).
    pub bid_actions: u32,

    // -- operating --
    pub current_operator: Option<String>,
    pub num_laid_track: u32,
    pub upgraded_track: bool,
    pub laid_hexes: Vec<String>,
    pub tokened: bool,
    pub teleported: Option<String>,
    pub acquired_companies: Vec<String>,
    pub pending_tokens: Vec<PendingTok>,
    pub routes: Vec<RunRoute>,
    pub extra_revenue: i32,
    /// SpecialTrack `@company` (an extra-tile-lay company mid-use) and
    /// `@extra_laided_track`.
    pub special_track_company: Option<String>,
    pub extra_laided_track: bool,
    /// SpecialChoose `@company`.
    pub special_choose_company: Option<String>,
    /// Route `@pullman_train` (train id) and the original name/distance.
    pub pullman_train: Option<String>,
    pub pullman_original: Option<(String, u32)>,
    /// Dividend `@extra_train_choice` ("" = none yet).
    pub extra_train_choice: String,
    /// BuyTrain `@last_share_sold_price`.
    pub last_share_sold_price: Option<i32>,
    /// MinorAcquisition step state (persists across entities like the
    /// Ruby step object's ivars).
    pub acquire_state: AcquireState,
    pub selected_minor: Option<String>,
    pub selected_share_num: u32,
    pub pay_choices: Vec<String>,
    pub token_choices: Vec<String>,
    /// BuyTrain `@corporations_sold`.
    pub corporations_sold: Vec<String>,
    /// SpecialTrack `@in_process`.
    pub special_track_in_process: bool,
    /// 1822PNW Merge step: `@merge_state`, the associated / unassociated
    /// minors, the new major, the merging player, the chosen par.
    #[serde(default)]
    pub merge_state: MergeState,
    #[serde(default)]
    pub merge_assoc: Option<String>,
    #[serde(default)]
    pub merge_unassoc: Option<String>,
    #[serde(default)]
    pub merge_new: Option<String>,
    #[serde(default)]
    pub merge_player: Option<u32>,
    #[serde(default)]
    pub merge_par: i32,
    /// 1822PNW SpecialTrack `@round.num_laid_portage`.
    #[serde(default)]
    pub num_laid_portage: u32,
    /// 1822PNW AcquireCompany: P20's `@choices` (minor ids) and
    /// `@new_associated_minor`.
    #[serde(default)]
    pub p20_choices: Vec<String>,
    #[serde(default)]
    pub p20_minor: Option<String>,
    /// 1822PNW BSPS: `@converting_major` and its `@par_price`.
    #[serde(default)]
    pub converting_major: Option<String>,
    #[serde(default)]
    pub convert_par: i32,
    /// 1822MX Track: the NdeM's remaining tile layers (`@ndem_tile_layers`,
    /// players in priority order; set at the NdeM's turn).
    #[serde(default)]
    pub ndem_layers: Vec<u32>,
    #[serde(default)]
    pub ndem_layers_set: bool,
    /// 1822MX AuctionNdemTokens' state.
    #[serde(default)]
    pub ndem_auction: Option<mx::NdemAuction>,
}

impl RoundSt {
    pub fn new(kind: RoundKind, round_num: u8, entities: Vec<String>, steps: Vec<Step>) -> Self {
        let n = steps.len();
        RoundSt {
            kind,
            round_num,
            entities,
            entity_index: 0,
            steps,
            passed: vec![false; n],
            acted: vec![false; n],
            players_sold: BTreeMap::new(),
            players_bought: BTreeMap::new(),
            current_actions: Vec::new(),
            bought_from_ipo: false,
            last_to_act: None,
            pass_order: Vec::new(),
            tax_haven_bought: false,
            bids: BTreeMap::new(),
            bid_actions: 0,
            current_operator: None,
            num_laid_track: 0,
            upgraded_track: false,
            laid_hexes: Vec::new(),
            tokened: false,
            teleported: None,
            acquired_companies: Vec::new(),
            pending_tokens: Vec::new(),
            routes: Vec::new(),
            extra_revenue: 0,
            special_track_company: None,
            extra_laided_track: false,
            special_choose_company: None,
            pullman_train: None,
            pullman_original: None,
            extra_train_choice: String::new(),
            last_share_sold_price: None,
            acquire_state: AcquireState::SelectMinor,
            selected_minor: None,
            selected_share_num: 0,
            pay_choices: Vec::new(),
            token_choices: Vec::new(),
            corporations_sold: Vec::new(),
            special_track_in_process: false,
            merge_state: MergeState::None,
            merge_assoc: None,
            merge_unassoc: None,
            merge_new: None,
            merge_player: None,
            merge_par: 0,
            num_laid_portage: 0,
            p20_choices: Vec::new(),
            p20_minor: None,
            converting_major: None,
            convert_par: 0,
            ndem_layers: Vec::new(),
            ndem_layers_set: false,
            ndem_auction: None,
        }
    }

    pub fn step_index(&self, step: Step) -> Option<usize> {
        self.steps.iter().position(|s| *s == step)
    }
}

/// The tax haven (P16): a dummy shareholder with its own cash.
pub const TAX_HAVEN_OWNER: &str = "taxhaven:P16";

/// The 1822 family's own state (see the module docs).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct G1822State {
    pub seed: u64,
    /// Ruby `@companies` order (the bid-box stacks read it): companies
    /// deleted from Ruby's list (floated / acquired minors' proxies,
    /// non-starting ones) are absent.
    pub company_order: Vec<String>,
    pub bidbox_minors_cache: Vec<String>,
    /// Ruby `company.owner == @bank` (bid-box companies up for auction).
    pub bank_owned: Vec<String>,
    pub nothing_sold_in_sr: bool,
    /// P15/P20 dummy players' cash; `None` once the credit left play.
    pub phase_revenue: BTreeMap<String, Option<i32>>,
    pub tax_haven_cash: i32,
    pub double_cash_choice: Option<u32>,
    pub minor_14_city_exit: Option<u8>,
    /// Trains reserved for the train privates (P1 5P, P3/P4 2P, P13/P14 P+).
    pub company_trains: BTreeMap<String, Train>,
    /// `exchange_token` ability counts per major.
    pub exchange_tokens: BTreeMap<String, u32>,
    /// Ruby `corporation.par_via_exchange` (concession company sym).
    pub par_via_exchange: BTreeMap<String, String>,
    /// Majors still holding their unplaced destination token ability.
    pub destination_ability: Vec<String>,
    pub player_debt: BTreeMap<u32, i32>,
    /// Ruby City#reservations, positional by slot, keyed "HEX#city".
    pub city_reservations: BTreeMap<String, Vec<Option<String>>>,
    /// Ruby Tile#reservations (minor 14's London), keyed by hex.
    pub tile_reservations: BTreeMap<String, Vec<String>>,
    /// Company abilities still live (tile lays, teleport, discounts).
    pub abilities: BTreeMap<String, Vec<track::Ab>>,
    /// Corporation-level :token abilities: (hexes, price) — SECR's free
    /// English Channel token; the £20 any-hex token from minor 14.
    pub corp_token_abilities: BTreeMap<String, Vec<(Vec<String>, i32)>>,
    /// P5's own token ability (count 1) still unused.
    pub lcdr_token_ability: bool,
    /// Live water borders with a cost: hex → (edge, cost, type).
    pub borders: BTreeMap<String, Vec<(u8, i32, String)>>,
    /// Next instance index per unlimited tile name.
    pub tile_instances: BTreeMap<String, u32>,
    /// combo_entities of the action being processed.
    pub pending_combo: Vec<String>,
    /// `share_price` of the buy/sell action being processed.
    pub pending_share_price: Option<i32>,
    /// Ruby Player#passed? for the stock-round entities.
    pub round_passed_players: Vec<u32>,
    /// `@operating_rounds` of the current OR set.
    pub operating_rounds: u8,
    /// Companies' current revenue (Ruby mutates `company.revenue`).
    pub company_revenue: BTreeMap<String, i32>,
    /// Corps' operating history entries (Ruby `operating_history.size`).
    pub operating_history: BTreeMap<String, u32>,
    /// Game end: Ruby `@game_end_triggered` reason/timing once latched.
    pub game_end_trigger: Option<(String, String)>,
    pub game_end_reason: Option<String>,
    pub stock_market_max_reached: bool,
    pub round: RoundSt,
    /// An error a deep helper raised the way Ruby raises a non-GameError
    /// mid-action (the action is refused and rolled back).
    #[serde(default)]
    pub pending_error: Option<String>,
    /// Corporations whose Ruby `Graph#route_info` is latched with
    /// `route_train_purchase` (Graph#clear keeps such entries and 1822
    /// never calls clear_graph_for): once a corporation could run a route
    /// it counts as able to for the rest of the game.
    #[serde(default)]
    pub route_purchase: Latch,
    /// 1822PNW: Ruby's Graph cache — the connected hexes/nodes/paths of
    /// each corporation whose graph Ruby fully computed since its last
    /// Graph#clear. Ruby never clears after a home token, a regional's
    /// float token or a closure, so a graph computed in the Merger round
    /// (Merge#mergeable) answers stale through the stock round into the
    /// next operating round until the first tile lay / token. Filled only
    /// where Rust mirrors such a Ruby compute; anything else is fresh.
    #[serde(default)]
    pub ruby_graphs: GraphMemo,
    /// The city holding each corporation's FIRST token (Ruby
    /// `tokens.first.city`: the destination bonus and destination check
    /// need that very token, not any of the corporation's): (hex, city),
    /// remapped when the hex is upgraded.
    #[serde(default)]
    pub home_city: BTreeMap<String, (String, usize)>,
    /// Companies Ruby deleted from `@companies` without closing (minor
    /// proxies after a float or an acquisition): their abilities live on
    /// (e.g. consent blockers), their owner is the bank.
    #[serde(default)]
    pub deleted_open: Vec<String>,

    // -- 1822PNW (empty for the other titles) -----------------------------
    /// `@minor_associations`: minor → major, in Ruby hash order.
    #[serde(default)]
    pub minor_associations: Vec<(String, String)>,
    /// `@empty_bidboxes` / `@any_bidboxes_empty` (a major parred straight
    /// from the bid box closes its associated minor, leaving its box empty
    /// until the round's refill).
    #[serde(default)]
    pub empty_bidboxes: Vec<bool>,
    #[serde(default)]
    pub any_bidboxes_empty: bool,
    /// `@regional_payout_count` per regional railway.
    #[serde(default)]
    pub regional_payout_count: BTreeMap<String, u32>,
    /// Hex assignments (Ruby Assignable#assignments keys, in order):
    /// 'forest', 'P15' (paper mill), 'P17' (ski haus).
    #[serde(default)]
    pub hex_assignments: BTreeMap<String, Vec<String>>,
    /// Builder cubes by TILE id (`block` icons; see `g1822_pnw_cubes`).
    #[serde(default)]
    pub builder_cubes: BTreeMap<String, u32>,
    /// Neighbour links a lay on a blue / gray / red hex opened (Tracker
    /// #lay_tile): (hex, direction, neighbour).
    #[serde(default)]
    pub opened_neighbors: Vec<(usize, u8, usize)>,
    /// P20 (backroom negotiations) moved a major's home and destination:
    /// `major.coordinates` / `destination_coordinates` overrides.
    #[serde(default)]
    pub home_override: BTreeMap<String, String>,
    #[serde(default)]
    pub destination_override: BTreeMap<String, String>,

    // -- 1822MX (defaults for the other titles) ----------------------------
    /// `@ndem_state`.
    #[serde(default)]
    pub ndem_state: mx::NdemState,
    /// `@number_ndem_shares` (the next NdeM share's index).
    #[serde(default)]
    pub ndem_shares: u32,
    /// Companies whose `owner` Ruby set without adding them to the owner's
    /// `companies` (MX's M18 after C1's purchase): the owner shows, but they
    /// are no certificate and no value.
    #[serde(default)]
    pub owner_only: Vec<String>,
}

/// A memo Ruby fills lazily while it evaluates step actions (read-side
/// code sets it, so it needs interior mutability). Send + Sync.
#[derive(Debug, Default)]
pub struct Latch(std::sync::Mutex<Vec<String>>);

impl Latch {
    pub fn contains(&self, s: &str) -> bool {
        self.0.lock().map(|v| v.iter().any(|x| x == s)).unwrap_or(false)
    }
    pub fn insert(&self, s: &str) {
        if let Ok(mut v) = self.0.lock() {
            if !v.iter().any(|x| x == s) {
                v.push(s.to_string());
            }
        }
    }
    fn snapshot(&self) -> Vec<String> {
        self.0.lock().map(|v| v.clone()).unwrap_or_default()
    }
}

impl Clone for Latch {
    fn clone(&self) -> Self {
        Latch(std::sync::Mutex::new(self.snapshot()))
    }
}

/// Ruby's per-corporation Graph cache (see `G1822State::ruby_graphs`).
#[derive(Debug, Default)]
pub struct GraphMemo(std::sync::Mutex<Vec<(String, graph::CorpGraph)>>);

impl GraphMemo {
    pub fn contains(&self, s: &str) -> bool {
        self.0.lock().map(|v| v.iter().any(|(k, _)| k == s)).unwrap_or(false)
    }
    pub fn get(&self, s: &str) -> Option<graph::CorpGraph> {
        self.0.lock().ok().and_then(|v| v.iter().find(|(k, _)| k == s).map(|(_, g)| g.clone()))
    }
    pub fn insert(&self, s: &str, g: graph::CorpGraph) {
        if let Ok(mut v) = self.0.lock() {
            if !v.iter().any(|(k, _)| k == s) {
                v.push((s.to_string(), g));
            }
        }
    }
    pub fn clear(&self) {
        if let Ok(mut v) = self.0.lock() {
            v.clear();
        }
    }
    fn snapshot(&self) -> Vec<(String, graph::CorpGraph)> {
        self.0.lock().map(|v| v.clone()).unwrap_or_default()
    }
}

impl Clone for GraphMemo {
    fn clone(&self) -> Self {
        GraphMemo(std::sync::Mutex::new(self.snapshot()))
    }
}

impl Serialize for GraphMemo {
    fn serialize<S: serde::Serializer>(&self, ser: S) -> Result<S::Ok, S::Error> {
        self.snapshot().serialize(ser)
    }
}

impl<'de> Deserialize<'de> for GraphMemo {
    fn deserialize<D: serde::Deserializer<'de>>(de: D) -> Result<Self, D::Error> {
        Ok(GraphMemo(std::sync::Mutex::new(Vec::<(String, graph::CorpGraph)>::deserialize(de)?)))
    }
}

impl Serialize for Latch {
    fn serialize<S: serde::Serializer>(&self, ser: S) -> Result<S::Ok, S::Error> {
        self.snapshot().serialize(ser)
    }
}

impl<'de> Deserialize<'de> for Latch {
    fn deserialize<D: serde::Deserializer<'de>>(de: D) -> Result<Self, D::Error> {
        Ok(Latch(std::sync::Mutex::new(Vec::<String>::deserialize(de)?)))
    }
}

// ---------------------------------------------------------------------------
// Entities
// ---------------------------------------------------------------------------

/// A Ruby entity as resolved from an action's `entity` id.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Ent {
    Player(u32),
    Corp(String),
    Company(String),
}

impl Ent {
    pub fn id(&self) -> String {
        match self {
            Ent::Player(p) => p.to_string(),
            Ent::Corp(s) | Ent::Company(s) => s.clone(),
        }
    }
}

pub fn err<T>(msg: impl Into<String>) -> Result<T, GameError> {
    Err(GameError::new(msg))
}

impl BaseGame {
    /// The title's family table (Ruby `self.class::` constants).
    pub(crate) fn fam(&self) -> &'static family::Family {
        self.title_def().g1822_family().expect("1822-family title")
    }
    /// A major's destination (`destination_coordinates`, P20-aware).
    pub(crate) fn g1822_destination(&self, c: &str) -> Option<String> {
        if let Some(d) = self.g().destination_override.get(c) {
            return Some(d.clone());
        }
        (self.fam().destination)(c).map(str::to_string)
    }

    /// The Ruby subclass (for the methods a variant overrides).
    pub(crate) fn variant(&self) -> family::Variant {
        self.fam().variant
    }

    // -- G1822 predicate methods a variant overrides ------------------------

    /// `can_upgrade_one_phase_ahead?(entity)` (G1822: COMPANY_BER).
    pub(crate) fn g1822_can_upgrade_one_phase_ahead(&self, company: &str) -> bool {
        match self.variant() {
            family::Variant::Pnw => false,
            family::Variant::Mx => company == "P8",
            _ => family::Family::is(self.fam().company_ber, company),
        }
    }
    /// `can_only_lay_plain_or_towns?(entity)` (G1822: COMPANY_BER).
    pub(crate) fn g1822_can_only_lay_plain_or_towns(&self, company: &str) -> bool {
        match self.variant() {
            family::Variant::Pnw => false,
            family::Variant::Mx => company == "P8",
            _ => family::Family::is(self.fam().company_ber, company),
        }
    }
    /// `must_be_on_terrain?(entity)` (G1822: COMPANY_EGR).
    pub(crate) fn g1822_must_be_on_terrain(&self, company: &str) -> bool {
        match self.variant() {
            family::Variant::Pnw | family::Variant::Mx => false,
            _ => family::Family::is(self.fam().company_egr, company),
        }
    }
    /// `must_be_on_estuary?(entity)` (G1822: COMPANY_GSWR).
    pub(crate) fn g1822_must_be_on_estuary(&self, company: &str) -> bool {
        family::Family::is(self.fam().company_gswr, company)
    }
    /// `must_remove_town?(entity)` (G1822: COMPANY_MTONR; 1822PNW: P7, P8).
    pub(crate) fn g1822_must_remove_town(&self, company: &str) -> bool {
        match self.variant() {
            family::Variant::Pnw => matches!(company, "P7" | "P8"),
            family::Variant::Mx => matches!(company, "P12" | "P13"),
            _ => family::Family::is(self.fam().company_mtonr, company),
        }
    }
    /// `company_ability_extra_track?(company)` (G1822: COMPANY_LSR;
    /// 1822PNW: P11, P12, P13).
    pub(crate) fn g1822_company_ability_extra_track(&self, company: &str) -> bool {
        match self.variant() {
            family::Variant::Pnw => matches!(company, "P11" | "P12" | "P13"),
            family::Variant::Mx => matches!(company, "P9" | "P17" | "P18"),
            _ => family::Family::is(self.fam().company_lsr, company),
        }
    }
    pub(crate) fn g(&self) -> &G1822State {
        self.g1822.as_deref().expect("1822 state")
    }
    pub(crate) fn gm(&mut self) -> &mut G1822State {
        self.g1822.as_deref_mut().expect("1822 state")
    }
    pub(crate) fn rnd(&self) -> &RoundSt {
        &self.g().round
    }
    pub(crate) fn rndm(&mut self) -> &mut RoundSt {
        &mut self.gm().round
    }

    /// Resolve an action entity id. 1822 corporation syms are numeric
    /// ("1".."30" minors), so player ids MUST NOT collide with them (the
    /// replay harness seats players at 1001+); corporations resolve first.
    pub(crate) fn g1822_entity(&self, id: &str) -> Ent {
        if let Some(&ci) = self.corp_idx.get(id) {
            if !self.corporations[ci].closed {
                return Ent::Corp(id.to_string());
            }
        }
        if self.company_idx.contains_key(id) {
            return Ent::Company(id.to_string());
        }
        if let Ok(pid) = id.parse::<u32>() {
            if self.player_index(pid).is_some() {
                return Ent::Player(pid);
            }
        }
        if self.corp_idx.contains_key(id) {
            return Ent::Corp(id.to_string());
        }
        Ent::Company(format!("__unknown:{id}"))
    }

    // -- company helpers --------------------------------------------------

    pub(crate) fn co_idx(&self, sym: &str) -> Option<usize> {
        self.company_idx.get(sym).copied()
    }
    pub(crate) fn co_owner(&self, sym: &str) -> EntityId {
        self.co_idx(sym)
            .map(|i| self.companies[i].owner.clone())
            .unwrap_or_else(EntityId::none)
    }
    pub(crate) fn co_closed(&self, sym: &str) -> bool {
        self.co_idx(sym).map_or(true, |i| self.companies[i].closed)
    }
    /// Ruby `company.close!`.
    pub(crate) fn close_company(&mut self, sym: &str) {
        if let Some(i) = self.co_idx(sym) {
            self.companies[i].closed = true;
            self.companies[i].owner = EntityId::none();
        }
        self.gm().bank_owned.retain(|c| c != sym);
        self.gm().deleted_open.retain(|c| c != sym);
    }
    /// Ruby `@companies.delete(company)` — gone from the stacks and from
    /// every lookup.
    pub(crate) fn delete_company(&mut self, sym: &str) {
        self.gm().company_order.retain(|c| c != sym);
        self.gm().bank_owned.retain(|c| c != sym);
        if !self.co_closed(sym) && !self.g().deleted_open.iter().any(|c| c == sym) {
            self.gm().deleted_open.push(sym.to_string());
        }
        if let Some(i) = self.co_idx(sym) {
            self.companies[i].closed = true;
            self.companies[i].owner = EntityId::none();
        }
    }
    pub(crate) fn company_exists(&self, sym: &str) -> bool {
        self.g().company_order.iter().any(|c| c == sym)
    }
    /// Companies owned by `owner` in Ruby `entity.companies` order
    /// (acquisition order is approximated by the stack order).
    pub(crate) fn companies_of(&self, owner: &EntityId) -> Vec<String> {
        self.g()
            .company_order
            .iter()
            .filter(|c| {
                self.co_idx(c)
                    .map_or(false, |i| !self.companies[i].closed && self.companies[i].owner == *owner)
            })
            .cloned()
            .collect()
    }

    /// Ruby `bank_companies(prefix)`: un-owned (or bank-owned) open
    /// companies of a prefix, in `@companies` order.
    pub(crate) fn bank_companies(&self, prefix: char) -> Vec<String> {
        let st = self.g();
        st.company_order
            .iter()
            .filter(|c| c.starts_with(prefix))
            .filter(|c| {
                self.co_idx(c).map_or(false, |i| {
                    let co = &self.companies[i];
                    !co.closed && (co.owner.is_none() || st.bank_owned.iter().any(|b| b == *c))
                })
            })
            .cloned()
            .collect()
    }
    pub(crate) fn bidbox_minors(&self) -> Vec<String> {
        let cache = &self.g().bidbox_minors_cache;
        let minors: Vec<String> = self
            .bank_companies('M')
            .into_iter()
            .take(self.fam().bidding_box_minor_count)
            .filter(|c| cache.contains(c))
            .collect();
        if !self.g().any_bidboxes_empty {
            return minors;
        }
        // 1822PNW: an emptied box keeps its place (EMPTY placeholder)
        let empty = &self.g().empty_bidboxes;
        let mut out: Vec<String> = Vec::new();
        for m in minors {
            while empty.get(out.len()).copied().unwrap_or(false) {
                out.push(pnw::EMPTY_BIDBOX.to_string());
            }
            out.push(m);
        }
        out
    }
    pub(crate) fn bidbox_concessions(&self) -> Vec<String> {
        self.bank_companies('C')
            .into_iter()
            .take(self.fam().bidding_box_concession_count)
            .collect()
    }
    pub(crate) fn bidbox_privates(&self) -> Vec<String> {
        let privates = self.bank_companies('P');
        if self.g1822_plus() && !self.optional_rules.iter().any(|r| r == "plus_expansion_single_stack") {
            // One bid box per private category (first of each stack).
            return self
                .fam()
                .plus_expansion_bidbox
                .iter()
                .filter_map(|bx| privates.iter().find(|c| bx.contains(&c.as_str())).cloned())
                .collect();
        }
        privates.into_iter().take(self.fam().bidding_box_private_count).collect()
    }

    /// Whether the 1822+ expansion is in play.
    pub(crate) fn g1822_plus(&self) -> bool {
        self.optional_rules.iter().any(|r| r == "plus_expansion")
    }

    /// TAX_HAVEN_MULTIPLE / the tax_haven_multiple optional rule.
    pub(crate) fn g1822_tax_haven_multiple(&self) -> bool {
        self.optional_rules.iter().any(|r| r == "tax_haven_multiple")
    }
    pub(crate) fn bidbox_minors_refill(&mut self) {
        // 1822PNW init_empty_bidboxes!
        if self.g().any_bidboxes_empty {
            let n = self.fam().bidding_box_minor_count;
            self.gm().empty_bidboxes = vec![false; n];
        }
        self.gm().any_bidboxes_empty = false;
        let cache: Vec<String> = self
            .bank_companies('M')
            .into_iter()
            .take(self.fam().bidding_box_minor_count)
            .collect();
        self.gm().bidbox_minors_cache = cache;
    }
    /// Ruby `setup_bidboxes` (G1822::Round::Stock#setup).
    pub(crate) fn setup_bidboxes(&mut self) {
        self.bidbox_minors_refill();
        let mut boxed = self.bidbox_minors();
        boxed.extend(self.bidbox_concessions());
        boxed.extend(self.bidbox_privates());
        for c in boxed {
            if !self.g().bank_owned.contains(&c) {
                self.gm().bank_owned.push(c);
            }
        }
        self.gm().double_cash_choice = None;
    }

    // -- corporation helpers ----------------------------------------------

    pub(crate) fn ci(&self, sym: &str) -> usize {
        *self
            .corp_idx
            .get(sym)
            .unwrap_or_else(|| panic!("unknown corporation {sym}"))
    }
    pub(crate) fn is_major(&self, sym: &str) -> bool {
        self.corp_idx
            .get(sym)
            .map_or(false, |&i| self.corporations[i].corp_type == CorpType::Major)
    }
    pub(crate) fn is_minor(&self, sym: &str) -> bool {
        self.corp_idx
            .get(sym)
            .map_or(false, |&i| self.corporations[i].corp_type == CorpType::Minor)
    }
    /// Ruby `find_corporation(company)` (proxy company M<n> → minor <n>).
    pub(crate) fn minor_of_company(company: &str) -> String {
        company[1..].to_string()
    }
    pub(crate) fn phase_num(&self) -> u32 {
        self.phase.name.parse().unwrap_or(0)
    }
    pub(crate) fn player_debt(&self, pid: u32) -> i32 {
        self.g().player_debt.get(&pid).copied().unwrap_or(0)
    }
    pub(crate) fn pay_player(&mut self, pid: u32, amount: i32) {
        let i = self.player_index(pid).expect("player");
        self.players[i].cash += amount;
    }
    /// Player → bank.
    pub(crate) fn player_spend(&mut self, pid: u32, amount: i32) -> Result<(), GameError> {
        let i = self.player_index(pid).expect("player");
        if self.players[i].cash < amount {
            return err(format!(
                "{} has {} and cannot spend {}",
                self.players[i].name, self.players[i].cash, amount
            ));
        }
        self.players[i].cash -= amount;
        self.bank.cash += amount;
        Ok(())
    }
    pub(crate) fn bank_pay_player(&mut self, pid: u32, amount: i32) {
        self.bank.cash -= amount;
        self.pay_player(pid, amount);
    }
    pub(crate) fn bank_pay_corp(&mut self, sym: &str, amount: i32) {
        let ci = self.ci(sym);
        self.bank.cash -= amount;
        self.corporations[ci].cash += amount;
    }
    pub(crate) fn corp_spend(&mut self, sym: &str, amount: i32) -> Result<(), GameError> {
        let ci = self.ci(sym);
        if self.corporations[ci].cash < amount {
            return err(format!(
                "{} has {} and cannot spend {}",
                sym, self.corporations[ci].cash, amount
            ));
        }
        self.corporations[ci].cash -= amount;
        self.bank.cash += amount;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Construction (G1822 setup)
// ---------------------------------------------------------------------------

impl BaseGame {
    /// Ruby `G1822::Game#setup` + the base init steps 1822 customizes
    /// (init_companies / init_corporations starting lists, reservations,
    /// the seeded company shuffle, company trains, exchange tokens,
    /// destinations) and the initial stock round. Called once by the
    /// constructor for 1822-family titles.
    pub(crate) fn g1822_setup(&mut self, seed: u64) {
        // Non-starting companies / corporations (the 1822+ ones) leave play
        // (Ruby never instantiates them).
        let plus = self.g1822_plus();
        let fam = self.fam();
        // setup_optional_rules runs before setup (base initialize)
        if fam.variant == family::Variant::Pnw {
            self.g1822_pnw_setup_optional_rules();
        }
        let starting_companies: Vec<String> = if plus {
            fam.starting_companies_plus
        } else {
            fam.starting_companies
        }
        .iter()
        .map(|s| s.to_string())
        .collect();
        let starting_corps: &[&str] = if plus {
            fam.starting_corporations_plus
        } else {
            fam.starting_corporations
        };
        for co in &mut self.companies {
            if !starting_companies.contains(&co.sym) {
                co.closed = true;
            }
        }
        for corp in &mut self.corporations {
            if !starting_corps.contains(&corp.sym.as_str()) {
                corp.closed = true;
            }
        }

        // setup_companies: `@companies.sort_by! { rand }` over the
        // init_companies order (COMPANIES order filtered to the starting
        // list), then minors / concessions / privates with M24, C1, P1
        // moved to the front of their stacks.
        let mut rng = RubyRand::new(seed);
        let mut keyed: Vec<(u64, usize, String)> = self
            .companies
            .iter()
            .filter(|c| starting_companies.contains(&c.sym))
            .enumerate()
            .map(|(i, c)| (rng.next(), i, c.sym.clone()))
            .collect();
        keyed.sort_by_key(|(k, i, _)| (*k, *i));
        let shuffled: Vec<String> = keyed.into_iter().map(|(_, _, s)| s).collect();
        let company_order: Vec<String> = if fam.variant == family::Variant::Pnw {
            // PNW's own setup_companies
            self.g1822_pnw_order_companies(&shuffled, &mut rng)
        } else if fam.variant == family::Variant::Mx {
            // MX's own setup_companies
            Self::g1822_mx_order_companies(&shuffled)
        } else {
        let by_prefix = |p: char| -> Vec<String> {
            shuffled.iter().filter(|s| s.starts_with(p)).cloned().collect()
        };
        let mut minors = by_prefix('M');
        let mut concessions = by_prefix('C');
        let mut privates = by_prefix('P');
        for (list, first) in [
            (&mut minors, fam.bidding_box_start_minor),
            (&mut concessions, fam.bidding_box_start_concession),
            (&mut privates, fam.bidding_box_start_private),
        ] {
            if let Some(pos) = list.iter().position(|s| s == first) {
                let s = list.remove(pos);
                list.insert(0, s);
            }
        }
        // 1822+ (without No Removals): privates regroup by bid box, then
        // one box-2 private (never P5) and two box-3 privates leave play.
        if plus && !self.optional_rules.iter().any(|r| r == "plus_expansion_no_removals") {
            let in_box = |b: &[&str], s: &String| b.contains(&s.as_str());
            let mut grouped: Vec<String> = Vec::new();
            for bx in fam.plus_expansion_bidbox {
                grouped.extend(privates.iter().filter(|s| in_box(bx, s)).cloned());
            }
            privates = grouped;
            let mut removed: Vec<String> = Vec::new();
            if let Some(pos) = privates
                .iter()
                .position(|s| s != fam.company_lcdr && in_box(fam.plus_expansion_bidbox[1], s))
            {
                removed.push(privates.remove(pos));
            }
            for _ in 0..2 {
                if let Some(pos) = privates.iter().position(|s| in_box(fam.plus_expansion_bidbox[2], s)) {
                    removed.push(privates.remove(pos));
                }
            }
            for r in removed {
                if let Some(i) = self.co_idx(&r) {
                    self.companies[i].closed = true;
                }
            }
        }
        let mut company_order = minors;
        company_order.extend(concessions);
        company_order.extend(privates);
        company_order
        };

        // Company trains (find_and_remove_train_by_id): reserved, out of
        // the depot. P1's 5P stays buyable; the 2P/P+ are not.
        let mut company_trains = BTreeMap::new();
        for &(co, tid) in fam.company_trains {
            if let Some(pos) = self.depot.trains.iter().position(|t| t.id == tid) {
                let t = self.depot.trains.remove(pos);
                company_trains.insert(co.to_string(), t);
            }
        }

        // Destination tokens (setup_destinations): one extra token of type
        // :destination per major (price 0).
        let mut destination_ability = Vec::new();
        for corp in &mut self.corporations {
            if corp.closed || (fam.destination)(&corp.sym).is_none() {
                continue;
            }
            let mut t = Token::new(corp.sym.clone(), 0);
            t.token_type = "destination".to_string();
            corp.tokens.push(t);
            if corp.sym == fam.two_home_corporation {
                destination_ability.push(corp.sym.clone());
            }
        }

        // Exchange tokens (setup_exchange_tokens).
        let exchange_tokens: BTreeMap<String, u32> = self
            .corporations
            .iter()
            .filter(|c| !c.closed && (fam.exchange_tokens)(&c.sym) > 0)
            .map(|c| (c.sym.clone(), (fam.exchange_tokens)(&c.sym)))
            .collect();

        // init_company_abilities: each concession's exchange-from-par
        // ability sets its major's par_via_exchange.
        let mut par_via_exchange = BTreeMap::new();
        for &(conc, corp) in fam.concessions {
            par_via_exchange.insert(corp.to_string(), conc.to_string());
        }

        let company_revenue = self
            .companies
            .iter()
            .map(|c| (c.sym.clone(), c.revenue))
            .collect();

        let mut phase_revenue = BTreeMap::new();
        for c in fam.private_phase_revenue {
            phase_revenue.insert(c.to_string(), Some(0));
        }

        let players: Vec<String> = self.players.iter().map(|p| p.id.to_string()).collect();
        let state = G1822State {
            seed,
            company_order,
            bidbox_minors_cache: Vec::new(),
            bank_owned: Vec::new(),
            nothing_sold_in_sr: true,
            phase_revenue,
            tax_haven_cash: 0,
            double_cash_choice: None,
            minor_14_city_exit: None,
            company_trains,
            exchange_tokens,
            par_via_exchange,
            destination_ability,
            player_debt: BTreeMap::new(),
            city_reservations: BTreeMap::new(),
            tile_reservations: BTreeMap::new(),
            abilities: (fam.abilities)(),
            corp_token_abilities: fam
                .corp_token_abilities
                .iter()
                .map(|(c, hexes, price)| (c.to_string(), vec![(hexes.iter().map(|h| h.to_string()).collect(), *price)]))
                .collect(),
            lcdr_token_ability: true,
            borders: BTreeMap::new(),
            tile_instances: BTreeMap::new(),
            pending_combo: Vec::new(),
            pending_share_price: None,
            round_passed_players: Vec::new(),
            operating_rounds: 1,
            company_revenue,
            operating_history: BTreeMap::new(),
            game_end_trigger: None,
            game_end_reason: None,
            stock_market_max_reached: false,
            pending_error: None,
            route_purchase: Latch::default(),
            ruby_graphs: GraphMemo::default(),
            home_city: BTreeMap::new(),
            deleted_open: Vec::new(),
            minor_associations: if fam.variant == family::Variant::Pnw {
                pnw::ASSOCIATIONS.iter().map(|(m, j)| (m.to_string(), j.to_string())).collect()
            } else {
                Vec::new()
            },
            empty_bidboxes: vec![false; fam.bidding_box_minor_count],
            any_bidboxes_empty: false,
            regional_payout_count: if fam.variant == family::Variant::Pnw {
                pnw::REGIONALS.iter().map(|r| (r.to_string(), 0)).collect()
            } else {
                BTreeMap::new()
            },
            hex_assignments: BTreeMap::new(),
            builder_cubes: BTreeMap::new(),
            opened_neighbors: Vec::new(),
            home_override: BTreeMap::new(),
            destination_override: BTreeMap::new(),
            ndem_state: mx::NdemState::Open,
            ndem_shares: 0,
            owner_only: Vec::new(),
            round: RoundSt::new(RoundKind::Stock, 1, players, vec![Step::DiscardTrain, Step::BuySellParShares]),
        };
        self.g1822 = Some(Box::new(state));
        self.turn = 1;
        // Home reservations (init_hexes, reservation_corporations order):
        // a city slot (Tile#add_reservation! → City#get_slot), or the
        // whole tile for minor 14 (London, no `city:`).
        let corp_defs = self.title_def().corporations();
        for cd in &corp_defs {
            if !starting_corps.contains(&cd.sym) || cd.home_hex.is_empty() {
                continue;
            }
            // 1822PNW reservation_corporations: corporations.reject(major)
            if fam.variant == family::Variant::Pnw && cd.corp_type == CorpType::Major {
                continue;
            }
            let city = (cd.home_city_index != u8::MAX).then_some(cd.home_city_index as usize);
            self.g1822_add_reservation(cd.home_hex, city, cd.sym);
        }
        // Live water borders (border=edge:N,type:water,cost:C).
        let mut borders = BTreeMap::new();
        for (coord, dsl, _) in fam.hex_dsl_all {
            let mut v = Vec::new();
            for part in dsl.split(';') {
                let Some(attrs) = part.trim().strip_prefix("border=") else { continue };
                let (mut edge, mut cost, mut ty) = (0u8, None, String::new());
                for kv in attrs.split(',') {
                    let (k, val) = kv.split_once(':').unwrap_or((kv, ""));
                    match k {
                        "edge" => edge = val.parse().unwrap_or(0),
                        "cost" => cost = val.parse().ok(),
                        "type" => ty = val.to_string(),
                        _ => {}
                    }
                }
                if let Some(c) = cost {
                    v.push((edge, c, ty));
                }
            }
            if !v.is_empty() {
                borders.insert(coord.to_string(), v);
            }
        }
        self.gm().borders = borders;
        if fam.variant == family::Variant::Mx {
            // MX game_cert_limit: CERT_LIMIT_INCREASED under higher_cert_limit
            if self.optional_rules.iter().any(|r| r == "higher_cert_limit") {
                if let Some(l) = crate::title::g1822mx::cert_limit_increased(self.players.len() as u8) {
                    self.cert_limit = l;
                }
            }
            // setup_game_specific → setup_ndem
            self.g1822_mx_setup_ndem();
        }
        // init_round = the first stock round (1822PNW's starting packet:
        // the packets, then straight into an operating round).
        if self.g1822_is_pnw() && self.g1822_pnw_starting_packet() {
            self.g1822_pnw_setup_starting_packets();
            let _ = self.g1822_start_operating_round(1);
            return;
        }
        self.g1822_start_stock_round();
    }

}

// ---------------------------------------------------------------------------
// The step machine (Ruby Round::Base)
// ---------------------------------------------------------------------------

impl BaseGame {
    /// Ruby `Round::Base#process_action` for an 1822-family game.
    /// Validate-and-mutate like Ruby, but atomically: a rejected action
    /// leaves the game exactly as it was.
    pub(crate) fn g1822_process_action(&mut self, action: &Action) -> Result<(), GameError> {
        if self.finished {
            return err("Game is already finished");
        }
        let backup = self.clone_for_search();
        match self.g1822_process_inner(action) {
            Ok(()) => {
                self.move_number += 1;
                self.g1822_sync_round_view();
                Ok(())
            }
            Err(e) => {
                *self = backup;
                Err(e)
            }
        }
    }

    /// `g1822_process_action` without the rollback copy, for throwaway
    /// clones (the legal-move filter): the state is garbage on error.
    pub(crate) fn g1822_process_action_nobackup(&mut self, action: &Action) -> Result<(), GameError> {
        if self.finished {
            return err("Game is already finished");
        }
        self.g1822_process_inner(action)?;
        self.move_number += 1;
        Ok(())
    }

    fn g1822_process_inner(&mut self, action: &Action) -> Result<(), GameError> {
        let ent = self.g1822_entity(action.entity_id());
        let atype = action.action_type();
        let steps = self.rnd().steps.clone();
        let mut chosen = None;
        for (i, step) in steps.iter().enumerate() {
            if !self.g1822_step_active(i) {
                continue;
            }
            let process = self.g1822_step_actions(i, &ent).contains(&atype);
            let blocking = self.g1822_step_blocking(i);
            if blocking && !process {
                return err(format!(
                    "Blocking step {:?} cannot process action {} (entity {})",
                    step,
                    atype,
                    action.entity_id()
                ));
            }
            if blocking || process {
                chosen = Some(i);
                break;
            }
        }
        let Some(i) = chosen else {
            return err(format!(
                "No step found for action {} (entity {})",
                atype,
                action.entity_id()
            ));
        };
        self.rndm().acted[i] = true;
        self.gm().pending_error = None;
        self.g1822_step_process(i, &ent, action)?;
        if let Some(e) = self.gm().pending_error.take() {
            return err(e);
        }
        self.g1822_skip_steps();
        self.g1822_after_process(action)?;
        // process_single_action: game_end_check, then the round loop.
        self.g1822_game_end_check();
        self.g1822_transition_loop()
    }

    /// Ruby `game_end_check` (GAME_END_LOCK_FIRST_TRIGGER): bank break
    /// (:full_or during an OR, :current_or otherwise) or the stock market's
    /// end cell (:current_or).
    pub(crate) fn g1822_game_end_check(&mut self) {
        if !self.fam().game_end_lock_first_trigger {
            return self.g1822_game_end_check_unlocked();
        }
        if self.g().game_end_trigger.is_some() {
            return;
        }
        if self.bank_broken || self.bank.cash < 0 {
            self.bank_broken = true;
            let timing = if self.rnd().kind == RoundKind::Operating { "full_or" } else { "current_or" };
            self.gm().game_end_trigger = Some(("bank".to_string(), timing.to_string()));
            return;
        }
        if self.g().stock_market_max_reached {
            self.gm().game_end_trigger = Some(("stock_market".to_string(), "current_or".to_string()));
        }
    }

    /// Base `game_end_check` without GAME_END_LOCK_FIRST_TRIGGER (1822PNW):
    /// every check re-evaluates the triggers (GAME_END_CHECK order: bank,
    /// stock_market) and a more urgent timing replaces the latched one.
    fn g1822_game_end_check_unlocked(&mut self) {
        const PRIORITY: [&str; 5] = ["immediate", "current_round", "current_or", "full_or", "one_more_full_or_set"];
        let prio = |a: &str| PRIORITY.iter().position(|p| *p == a).unwrap_or(PRIORITY.len());
        if self.bank_broken || self.bank.cash < 0 {
            self.bank_broken = true;
        }
        let mut triggered: Vec<(&str, &str)> = Vec::new();
        if self.bank_broken {
            triggered.push(("bank", if self.rnd().kind == RoundKind::Operating { "full_or" } else { "current_or" }));
        }
        if self.g().stock_market_max_reached {
            triggered.push(("stock_market", "current_or"));
        }
        for (reason, after) in triggered {
            if let Some((_, cur)) = &self.g().game_end_trigger {
                if prio(after) >= prio(cur) {
                    continue;
                }
            }
            self.gm().game_end_trigger = Some((reason.to_string(), after.to_string()));
        }
    }

    /// Ruby `end_now?(after)` for the round that just finished.
    pub(crate) fn g1822_end_now(&self) -> bool {
        let Some((_, timing)) = &self.g().game_end_trigger else { return false };
        if self.rnd().kind != RoundKind::Operating {
            return false;
        }
        match timing.as_str() {
            "current_or" => true,
            "full_or" => self.rnd().round_num as u32 >= self.g1822_operating_rounds_in_set(),
            _ => false,
        }
    }

    /// Whether the live round is finished (Ruby `@round.finished?`).
    pub(crate) fn g1822_round_is_finished(&self) -> bool {
        if self.finished {
            return true;
        }
        match self.rnd().kind {
            RoundKind::Stock => self.g1822_stock_finished(),
            RoundKind::Choices | RoundKind::Merger => self.g1822_active_step().is_none(),
            RoundKind::Operating => self.g1822_or_finished(),
        }
    }

    /// The `while @round.finished? && !@finished` loop.
    pub(crate) fn g1822_transition_loop(&mut self) -> Result<(), GameError> {
        for _ in 0..64 {
            if self.finished || !self.g1822_round_is_finished() {
                return Ok(());
            }
            self.gm().round_passed_players.clear();
            if self.g1822_end_now() {
                let reason = self
                    .g()
                    .game_end_trigger
                    .as_ref()
                    .map(|(r, _)| r.clone())
                    .unwrap_or_default();
                self.g1822_end_game(&reason);
                return Ok(());
            }
            if self.turn >= 100 {
                self.g1822_end_game("dnf");
                return Ok(());
            }
            self.g1822_next_round()?;
        }
        err("1822: round transitions did not settle")
    }

    /// Ruby `step.active?`.
    pub(crate) fn g1822_step_active(&self, i: usize) -> bool {
        let r = self.rnd();
        match r.steps[i] {
            Step::DiscardTrain => !self.g1822_crowded_corps().is_empty(),
            Step::PendingToken => !r.pending_tokens.is_empty(),
            // 1822MX: active while the closing NdeM operates
            Step::CashOutNdem => self.g1822_mx_cash_out_active(),
            Step::AuctionNdemTokens => self.g1822_mx_auction_active(),
            // MX Track#active? for the NdeM: super && (layers left || !acted)
            Step::Track if self.g1822_is_mx() && self.g1822_operator().as_deref() == Some(mx::NDEM) => {
                !r.passed[i] && (!r.ndem_layers.is_empty() || !r.acted[i])
            }
            _ => !r.passed[i],
        }
    }

    /// Ruby `step.blocks?`.
    pub(crate) fn g1822_step_blocks(&self, i: usize) -> bool {
        let r = self.rnd();
        match r.steps[i] {
            Step::SpecialTrack => r.special_track_company.is_some(),
            Step::SpecialToken => self.g1822_can_token_after_teleport(),
            // Engine::Step::Assign / SpecialChoose: never block
            Step::Assign => false,
            Step::SpecialChoose if self.g1822_is_pnw() || self.g1822_is_mx() => false,
            _ => true,
        }
    }

    /// Ruby `step.current_entity` (`active_entities[0]`).
    pub(crate) fn g1822_step_current_entity(&self, i: usize) -> Option<Ent> {
        let r = self.rnd();
        match r.steps[i] {
            Step::DiscardTrain => self.g1822_crowded_corps().first().map(|c| Ent::Corp(c.clone())),
            Step::PendingToken => r.pending_tokens.first().map(|p| Ent::Corp(p.corp.clone())),
            Step::SpecialTrack if r.special_track_company.is_some() => {
                r.special_track_company.clone().map(Ent::Company)
            }
            Step::SpecialToken if r.teleported.is_some() => r.teleported.clone().map(Ent::Company),
            _ => self.g1822_round_entity(),
        }
    }

    /// `entities[entity_index]` as an entity.
    pub(crate) fn g1822_round_entity(&self) -> Option<Ent> {
        let r = self.rnd();
        let id = r.entities.get(r.entity_index)?;
        Some(match r.kind {
            RoundKind::Operating | RoundKind::Merger => Ent::Corp(id.clone()),
            _ => Ent::Player(id.parse().ok()?),
        })
    }

    /// Ruby `step.blocking?` = `blocks? && !current_actions.empty?`
    /// (SpecialToken overrides `blocking?` with its `blocks?`).
    pub(crate) fn g1822_step_blocking(&self, i: usize) -> bool {
        if !self.g1822_step_blocks(i) {
            return false;
        }
        if self.rnd().steps[i] == Step::SpecialToken {
            return true;
        }
        match self.g1822_step_current_entity(i) {
            Some(e) => {
                if self.g1822_entity_closed(&e) {
                    return false;
                }
                !self.g1822_step_actions(i, &e).is_empty()
            }
            None => false,
        }
    }

    pub(crate) fn g1822_entity_closed(&self, e: &Ent) -> bool {
        match e {
            Ent::Corp(s) => self.corp_idx.get(s).map_or(true, |&i| self.corporations[i].closed),
            Ent::Company(s) => self.co_closed(s),
            Ent::Player(_) => false,
        }
    }

    /// Ruby `round.active_step` (the first active blocking step).
    pub(crate) fn g1822_active_step(&self) -> Option<usize> {
        (0..self.rnd().steps.len()).find(|&i| self.g1822_step_active(i) && self.g1822_step_blocking(i))
    }

    /// Ruby `step.actions(entity)`.
    pub(crate) fn g1822_step_actions(&self, i: usize, ent: &Ent) -> Vec<&'static str> {
        match self.rnd().steps[i] {
            Step::BuySellParShares => self.g1822_bsps_actions(ent),
            Step::Choose => self.g1822_choose_actions(ent),
            Step::Merge => self.g1822_pnw_merge_actions(ent),
            Step::Assign => self.g1822_pnw_assign_actions(ent),
            _ => self.g1822_or_step_actions(i, ent),
        }
    }

    fn g1822_step_process(&mut self, i: usize, ent: &Ent, action: &Action) -> Result<(), GameError> {
        match self.rnd().steps[i] {
            Step::BuySellParShares => self.g1822_bsps_process(ent, action),
            Step::Choose => self.g1822_choose_process(ent, action),
            Step::Merge => self.g1822_pnw_merge_process(i, ent, action),
            Step::Assign => self.g1822_pnw_assign_process(ent, action),
            _ => self.g1822_or_step_process(i, ent, action),
        }
    }

    /// Ruby `Round::Base#skip_steps`.
    pub(crate) fn g1822_skip_steps(&mut self) {
        for i in 0..self.rnd().steps.len() {
            if !self.g1822_step_active(i) || !self.g1822_step_blocks(i) {
                continue;
            }
            if let Some(Ent::Corp(c)) = self.g1822_round_entity() {
                if self.corp_idx.get(&c).map_or(true, |&ci| self.corporations[ci].closed) {
                    continue;
                }
            }
            if self.g1822_step_blocking(i) {
                break;
            }
            self.g1822_step_skip(i);
        }
    }

    /// Ruby `step.skip!` (default: `pass!`).
    fn g1822_step_skip(&mut self, i: usize) {
        match self.rnd().steps[i] {
            Step::BuySellParShares => self.g1822_bsps_pass(),
            Step::Choose | Step::Merge | Step::Assign => self.rndm().passed[i] = true,
            _ => self.g1822_or_step_skip(i),
        }
    }

    fn g1822_after_process(&mut self, _action: &Action) -> Result<(), GameError> {
        match self.rnd().kind {
            RoundKind::Stock => {
                if self.g1822_active_step().is_none() {
                    self.g1822_stock_next_entity()?;
                }
            }
            RoundKind::Choices => {}
            RoundKind::Merger => self.g1822_pnw_merger_after_process(),
            RoundKind::Operating => self.g1822_or_after_process()?,
        }
        Ok(())
    }

    /// game.rb `next_round!`: SR → Choices → OR set → SR.
    pub(crate) fn g1822_next_round(&mut self) -> Result<(), GameError> {
        if self.finished {
            return Ok(());
        }
        match self.rnd().kind {
            RoundKind::Choices => {
                self.g1822_reorder_players();
                self.g1822_start_operating_round(1)?;
            }
            RoundKind::Stock => self.g1822_start_choices_round()?,
            RoundKind::Operating if self.g1822_is_pnw() && self.g1822_pnw_starting_packet() && self.phase_num() == 1 => {
                // PNW starting packet: phase-1 ORs repeat (round 1 again)
                self.turn += 1;
                self.g1822_start_operating_round(1)?;
            }
            RoundKind::Operating if self.g1822_is_pnw() && self.phase_num() >= 2 => {
                let round_num = self.rnd().round_num;
                self.g1822_pnw_start_merger_round(round_num)?;
            }
            RoundKind::Operating | RoundKind::Merger => {
                let round_num = self.rnd().round_num;
                let total = self.g1822_operating_rounds_in_set();
                if (round_num as u32) < total {
                    self.g1822_start_operating_round(round_num + 1)?;
                } else {
                    self.turn += 1;
                    self.g1822_start_stock_round();
                }
            }
        }
        Ok(())
    }

    /// Mirror the live 1822 round into the shared `Round` enum so generic
    /// readers (PyO3 `round()`, `debug_state`, active entity) see it.
    pub(crate) fn g1822_sync_round_view(&mut self) {
        let r = self.rnd().clone();
        match r.kind {
            RoundKind::Stock | RoundKind::Choices => {
                let ids: Vec<u32> = self.players.iter().map(|p| p.id).collect();
                let current = self
                    .g1822_active_step()
                    .and_then(|i| self.g1822_step_current_entity(i));
                let cur_pid = match current {
                    Some(Ent::Player(p)) => p,
                    _ => r.entities.get(r.entity_index).and_then(|s| s.parse().ok()).unwrap_or(ids[0]),
                };
                let mut s = crate::rounds::StockState::new(&ids, ids[0]);
                s.current_player_index = s.player_order.iter().position(|&p| p == cur_pid).unwrap_or(0);
                s.turn = self.turn;
                self.round = crate::rounds::Round::Stock(s);
                self.round_state.round_type = match r.kind {
                    RoundKind::Choices => "Choices".to_string(),
                    _ => "Stock".to_string(),
                };
                self.round_state.round_num = r.round_num;
                self.round_state.active_entity_id = EntityId::player(cur_pid);
            }
            RoundKind::Operating | RoundKind::Merger => {
                let mut s = crate::rounds::OperatingState::new(
                    r.round_num,
                    self.g1822_operating_rounds_in_set() as u8,
                    r.entities.clone(),
                );
                s.entity_index = r.entity_index;
                self.round = crate::rounds::Round::Operating(s);
                self.round_state.round_type =
                    if r.kind == RoundKind::Merger { "Merger" } else { "Operating" }.to_string();
                self.round_state.round_num = r.round_num;
                let current = self
                    .g1822_active_step()
                    .and_then(|i| self.g1822_step_current_entity(i));
                self.round_state.active_entity_id = match current {
                    Some(Ent::Corp(c)) => EntityId::corporation(&c),
                    Some(Ent::Player(p)) => EntityId::player(p),
                    Some(Ent::Company(c)) => EntityId(format!("company:{c}")),
                    None => EntityId::none(),
                };
            }
        }
    }

    /// The legal action types for the acting entity (Ruby `actions_for`
    /// over the round's current entity, unioned with the current entity's
    /// companies — the companies act in parallel).
    pub(crate) fn g1822_action_types(&self) -> Vec<String> {
        if self.finished {
            return Vec::new();
        }
        let Some(bi) = self.g1822_active_step() else { return Vec::new() };
        let Some(ent) = self.g1822_step_current_entity(bi) else { return Vec::new() };
        let mut out: Vec<String> = Vec::new();
        let mut push = |ts: Vec<&'static str>| {
            for t in ts {
                if !out.iter().any(|x| x == t) {
                    out.push(t.to_string());
                }
            }
        };
        for i in 0..=bi {
            if self.g1822_step_active(i) {
                push(self.g1822_step_actions(i, &ent));
            }
        }
        out
    }

    /// Ruby `@game.end_game!` bookkeeping for 1822.
    pub(crate) fn g1822_end_game(&mut self, reason: &str) {
        self.gm().game_end_reason = Some(reason.to_string());
        self.finished = true;
    }

    /// Ruby `G1822::Game#player_value` (player.value + tax haven value).
    pub(crate) fn g1822_player_value(&self, pid: u32) -> i32 {
        let Some(pi) = self.player_index(pid) else { return 0 };
        let p = &self.players[pi];
        let eid = EntityId::player(pid);
        let mut v = p.cash;
        for c in &self.corporations {
            if c.closed || c.ipo_price.is_none() {
                continue;
            }
            for s in &c.shares {
                if s.owner == eid {
                    v += self.g1822_share_price(c, s.percent);
                }
            }
        }
        for co in &self.companies {
            if !co.closed && co.owner == eid && !self.g().owner_only.contains(&co.sym) {
                v += co.value;
            }
        }
        v -= self.player_debt(pid);
        // tax_haven_value: P16's owner gets the haven's cash + shares.
        let osth = self.fam().company_osth;
        if self.co_owner(osth) == eid && !self.co_closed(osth) {
            v += self.g1822_tax_haven_value();
        }
        v
    }

    /// Ruby `@tax_haven.value` (cash + share prices).
    pub(crate) fn g1822_tax_haven_value(&self) -> i32 {
        let th = EntityId(TAX_HAVEN_OWNER.to_string());
        let mut v = self.g().tax_haven_cash;
        for c in &self.corporations {
            if c.closed || c.ipo_price.is_none() {
                continue;
            }
            for s in &c.shares {
                if s.owner == th {
                    v += self.g1822_share_price(c, s.percent);
                }
            }
        }
        v
    }

    /// Ruby `Share#price` for a holder (market price; `always_market_price`
    /// makes IPO and pool shares the same): `price × percent / share_percent`.
    pub(crate) fn g1822_share_price(&self, c: &crate::entities::Corporation, percent: u8) -> i32 {
        let price = c.share_price.as_ref().map_or(0, |sp| sp.price);
        // num_shares(ceil: false) × price, ceiled.
        let num = percent as f64 / c.share_unit_percent as f64;
        (price as f64 * num).ceil() as i32
    }

    pub(crate) fn g1822_results(&self) -> std::collections::HashMap<u32, i32> {
        self.players
            .iter()
            .map(|p| (p.id, self.g1822_player_value(p.id)))
            .collect()
    }
}

impl BaseGame {
    pub(crate) fn g1822_action_bundle_pub(&self, sym: &str, percent: u8, idx: &[usize]) -> Result<shares::Bundle, GameError> {
        self.g1822_action_bundle(sym, percent, idx)
    }

    /// Price a route whose record carries no revenue — not supported yet
    /// for 1822 (every 18xx.games 1822 record carries revenue).
    pub(crate) fn g1822_compute_route_revenue(&self, _corp: &str, r: &crate::actions::RouteData) -> Result<i32, GameError> {
        err(format!("1822: route for {} carries no revenue", r.train_name))
    }

    /// The base revenues of a recorded route's first and last stops (mail
    /// contracts): the first chain's first hex, the last chain's last hex.
    pub(crate) fn g1822_route_stop_revenues(&self, r: &RunRoute) -> Vec<i32> {
        // The record's `hexes` lists the visited stops in route order
        // (Ruby Route#hexes over visited_stops).
        r.hexes.iter().map(|h| self.g1822_hex_stop_revenue(h)).collect()
    }

    /// A hex's stop revenue at the current phase (the best node on it).
    pub(crate) fn g1822_hex_stop_revenue(&self, hex: &str) -> i32 {
        let Some(&hi) = self.hex_idx.get(hex) else { return 0 };
        let tile = &self.hexes[hi].tile;
        let tiers = self.g1822_tiered_city_revenue(hi);
        let mut best = 0;
        for c in &tile.cities {
            best = best.max(tiers.unwrap_or(c.revenue));
        }
        for t in &tile.towns {
            best = best.max(t.revenue);
        }
        for o in &tile.offboards {
            best = best.max(o.phase_revenue(&self.phase.tiles));
        }
        best
    }
}

impl BaseGame {
    /// The phase-tiered city revenue of the tile on `hi`: a preprinted
    /// hex's DSL, or (1822PNW's port tiles) a laid tile's own DSL.
    pub(crate) fn g1822_tiered_city_revenue(&self, hi: usize) -> Option<i32> {
        let hex = &self.hexes[hi];
        if !(self.g1822_is_pnw() || self.g1822_is_mx()) || hex.tile.id.starts_with("preprinted_") {
            return data_tiered_city_revenue(self.fam(), &hex.id, &self.phase.tiles);
        }
        let base = hex.tile.name.split('-').next().unwrap_or(&hex.tile.name);
        let (dsl, _) = (self.fam().tile_dsl)(base)?;
        tiered_from_dsl(dsl, &self.phase.tiles)
    }
}

/// A gray hex's phase-tiered city revenue ("revenue:yellow_40|green_50|...").
pub(crate) fn data_tiered_city_revenue(fam: &family::Family, hex: &str, phase_tiles: &[String]) -> Option<i32> {
    let (dsl, _) = (fam.hex_dsl)(hex)?;
    tiered_from_dsl(dsl, phase_tiles)
}

fn tiered_from_dsl(dsl: &str, phase_tiles: &[String]) -> Option<i32> {
    let part = dsl.split(';').find(|p| p.starts_with("city=revenue:") && p.contains('|'))?;
    let rev = part.strip_prefix("city=revenue:")?.split(',').next()?;
    let mut out = None;
    for tier in rev.split('|') {
        let (color, v) = tier.split_once('_')?;
        if phase_tiles.iter().any(|t| t == color) {
            out = v.parse().ok();
        }
    }
    out
}

