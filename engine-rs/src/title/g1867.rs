//! Static game data + title impl for **1867: Railways of Canada**.
//!
//! Transcribed verbatim from the vendored Ruby reference at
//! docs/ruby_reference/g_1867/ (tobymao/18xx lib/engine/game/g_1867/):
//!   entities.rb  — COMPANIES, CORPORATIONS (8 majors + 16 minors + CN)
//!   game.rb      — TRAINS, PHASES, COLUMN_MARKET, GRID_MARKET, cash/cert/
//!                  bank, loans, round flow, events
//!   map.rb       — TILES, HEXES
//!   stock_market.rb, round/*.rb, step/*.rb — referenced in comments
//!
//! Promoted from the analysis draft (g1867_draft.rs) when the title
//! registered. The title is **rules-incomplete while Phase 1 lands** — the
//! opening single-item auction is implemented (rounds/single_auction.rs);
//! the remaining mechanics arrive seam by seam against the fixture-replay
//! harness (tests/test_g1867_fixture_replay.py prefix watermarks). Open
//! engine gaps are marked `TODO(1867-...)` and catalogued in
//! docs/g1867_port_notes.md.

use std::collections::HashMap;
use std::sync::Arc;

use super::{
    AbilityDef, BorderDef, Capitalization, CompanyDef, CorpType, CorporationDef, GameTitle,
    HexDef, HexType, MarketCell, MarketMovement, MarketZone, OwnerType, PhaseDef, TrainDef,
};
use crate::steps::{FinishedRound, RoundStart, RoundTransition, StepDesc, StepKind};
use crate::tiles::TileDef;

/// 1867: Railways of Canada.
pub struct G1867;

impl GameTitle for G1867 {
    fn name(&self) -> &'static str {
        "1867"
    }
    fn auction_steps(&self) -> &'static [StepDesc] {
        auction_steps()
    }
    fn stock_steps(&self) -> &'static [StepDesc] {
        stock_steps()
    }
    fn operating_steps(&self) -> &'static [StepDesc] {
        operating_steps()
    }
    fn next_round(&self, finished: FinishedRound, phase_operating_rounds: u8) -> RoundTransition {
        next_round(finished, phase_operating_rounds)
    }
    fn starting_cash(&self, num_players: u8) -> i32 {
        starting_cash(num_players)
    }
    fn cert_limit(&self, num_players: u8) -> u8 {
        cert_limit(num_players)
    }
    fn bank_cash(&self) -> i32 {
        BANK_CASH
    }
    fn companies(&self) -> Vec<CompanyDef> {
        companies()
    }
    fn corporations(&self) -> Vec<CorporationDef> {
        corporations()
    }
    fn trains(&self) -> Vec<TrainDef> {
        trains()
    }
    fn phases(&self) -> Vec<PhaseDef> {
        phases()
    }
    fn hex_definitions(&self) -> Vec<HexDef> {
        hex_definitions()
    }
    fn preprinted_hex_dsl(&self, coord: &str) -> Option<(&'static str, &'static str)> {
        preprinted_hex_dsl(coord)
    }
    fn tile_counts(&self) -> Vec<(&'static str, u32)> {
        tile_counts()
    }
    /// The 1867 catalog carries full city geometry for every id (parsed from
    /// the DSL), so the fallback is never consulted.
    fn tile_cities(&self, _tile_id: &str) -> Option<Vec<u8>> {
        None
    }
    fn tile_catalog(&self) -> Arc<HashMap<String, TileDef>> {
        crate::tiles::tile_catalog_1867()
    }
    /// The default COLUMN market (the GRID 2-D market is an optional rule —
    /// meta.rb `:grid_market` — not plumbed; the corpus audit skips those
    /// games).
    fn market_grid(&self) -> Vec<Vec<Option<MarketCell>>> {
        market_grid()
    }
    fn market_movement(&self) -> MarketMovement {
        MarketMovement::OneDimensional
    }
    /// Rules-first registration: the AlphaZero bridge (flat action layout +
    /// encoder spec) is a later roadmap phase — the 1830-shaped slot layout
    /// cannot express 1867's three discounted trains or the merger-round
    /// actions yet.
    fn alphazero_bridge_ready(&self) -> bool {
        false
    }
    /// Phase availability windows (game.rb): green minors join when the
    /// first 3-train fires `green_minors_available`; majors can IPO from
    /// phase 4 (`majors_can_ipo`).
    /// TODO(1867-events): `minors_cannot_start` (first 5-train) closes the
    /// minor window again — needs event state, lands with the event work.
    fn corporation_startable(&self, sym: &str, phase_name: &str) -> bool {
        let phase: u8 = phase_name.parse().unwrap_or(2);
        if GREEN_CORPORATIONS.contains(&sym) {
            return phase >= 3;
        }
        if corporations()
            .iter()
            .any(|cd| cd.sym == sym && cd.corp_type == CorpType::Major)
        {
            return phase >= MAJOR_PHASE;
        }
        true
    }
    /// 1867 is a FLAT-top map (game.rb:220 `LAYOUT = :flat`) — every edge
    /// number in the map DSL, borders and stubs is relative to it.
    fn hex_layout(&self) -> super::HexLayout {
        super::HexLayout::Flat
    }
    /// Loans (game.rb:376-378, 444-446, 499-530, 899-907): 72 × $50, taking
    /// nets $45; majors hold ≤5, minors ≤2.
    fn loan_value(&self) -> i32 {
        LOAN_VALUE
    }
    fn loan_interest_rate(&self) -> i32 {
        INTEREST_RATE
    }
    fn num_loans(&self) -> u32 {
        NUM_LOANS
    }
    fn max_loans(&self, corp_type: CorpType) -> u32 {
        match corp_type {
            CorpType::Major => MAXIMUM_LOANS_MAJOR,
            CorpType::Minor => MAXIMUM_LOANS_MINOR,
            CorpType::National => 0,
        }
    }
    /// Trains trade between ANY corporations (base default; 1830 restricts
    /// to same-president).
    fn train_buy_from_other_players(&self) -> bool {
        true
    }
    /// 1867's EMR is loans-only: no share selling, no president cash
    /// (step/buy_train.rb `president_may_contribute?` / `can_sell?` false).
    fn ebuy_president_may_contribute(&self) -> bool {
        false
    }
    /// MUST_BUY_TRAIN = :always (game.rb:308) — no route requirement.
    fn must_buy_train_always(&self) -> bool {
        true
    }
    /// Two lays per OR turn: the second never an upgrade after an upgrade,
    /// costs $20, and must target a fresh hex (game.rb:330-333 TILE_LAYS).
    fn tile_lays(&self) -> &'static [super::TileLayDef] {
        use super::{TileLayDef, UpgradeAllowance};
        const TWO: &[TileLayDef] = &[
            TileLayDef {
                lay: true,
                upgrade: UpgradeAllowance::Yes,
                cost: 0,
                cannot_reuse_same_hex: false,
            },
            TileLayDef {
                lay: true,
                upgrade: UpgradeAllowance::NotIfUpgraded,
                cost: 20,
                cannot_reuse_same_hex: true,
            },
        ];
        TWO
    }
    // NOTE: no action_hex_order / action_tile_order overrides — 1867 uses
    // the trait's clean derivations (no trained checkpoints to preserve).
}

// ---------------------------------------------------------------------------
// Round descriptions (game.rb:806-856)
// ---------------------------------------------------------------------------

/// 1867's auction round (game.rb:806-810 `new_auction_round`):
///   Engine::Round::Auction.new(self, [G1867::Step::SingleItemAuction])
pub fn auction_steps() -> &'static [StepDesc] {
    const STEPS: &[StepDesc] = &[StepDesc::blocking(StepKind::SingleItemAuction)];
    STEPS
}

/// 1867's stock round (game.rb:812-819 `stock_round`), Ruby order:
///   MajorTrainless, DiscardTrain, HomeToken, BuySellParShares (via bid).
///
/// The via-bid behavior (in-SR minor founding) and the stock-round
/// HomeToken choice live in rounds/stock_bid.rs.
/// TODO(1867-trainless): the MajorTrainless choose step lands with CN
/// nationalization.
pub fn stock_steps() -> &'static [StepDesc] {
    const STEPS: &[StepDesc] = &[
        StepDesc::blocking(StepKind::DiscardTrain),
        StepDesc::blocking(StepKind::HomeToken),
        StepDesc::blocking(StepKind::BuySellParShares),
    ];
    STEPS
}

/// 1867's operating round (game.rb:839-856 `operating_round`), Ruby order:
///   MajorTrainless, BuyCompany, RedeemShares, Track, Token, Route,
///   Dividend, BuyCompanyPreloan(blocks), LoanOperations, DiscardTrain,
///   BuyTrain, BuyCompany(blocks).
///
/// TODO(1867-or): placeholder with the steps the engine has today; the
/// 1867-specific steps (RedeemShares, BuyCompanyPreloan, LoanOperations,
/// MajorTrainless) and the rule deltas (two tile lays, distance token
/// pricing, half-pay dividends, loan-EMR BuyTrain) land mechanic by
/// mechanic.
pub fn operating_steps() -> &'static [StepDesc] {
    const STEPS: &[StepDesc] = &[
        StepDesc::non_blocking(StepKind::Bankrupt),
        StepDesc::non_blocking(StepKind::BuyCompany),
        StepDesc::blocking(StepKind::HomeToken),
        StepDesc::blocking(StepKind::Track),
        StepDesc::blocking(StepKind::Token),
        StepDesc::blocking(StepKind::Route),
        StepDesc::blocking(StepKind::Dividend),
        StepDesc::blocking(StepKind::DiscardTrain),
        StepDesc::blocking(StepKind::BuyTrain),
        StepDesc::blocking(StepKind::BuyCompany),
    ];
    STEPS
}

/// 1867's round flow (game.rb:876-897 `next_round!`):
///   auction → SR; SR → OR set (count from phase, final set 3);
///   OR → next OR / SR (turn++).
///
/// TODO(1867-merger): in phases 3-7 a MERGER ROUND follows every OR
/// (SR → OR1 → MR → OR2 → MR → SR); needs a `RoundStart::Merger` variant +
/// `FinishedRound::Merger` arm here. Until the merger round lands this is
/// the phase-2/8 flow for all phases.
/// TODO(1867-export): `or_round_finished` exports a train to the CN in
/// phases 4-7 (depot.export! + phase change as if purchased) — lands with
/// the train-export work.
pub fn next_round(finished: FinishedRound, phase_operating_rounds: u8) -> RoundTransition {
    let (increment_turn, start) = match finished {
        FinishedRound::Auction => (false, RoundStart::Stock),
        FinishedRound::Stock => (
            false,
            RoundStart::Operating {
                round_num: 1,
                total_ors: phase_operating_rounds,
            },
        ),
        FinishedRound::Operating { round_num, total_ors } if round_num < total_ors => (
            false,
            RoundStart::Operating {
                round_num: round_num + 1,
                total_ors,
            },
        ),
        FinishedRound::Operating { .. } => (true, RoundStart::Stock),
    };
    RoundTransition { increment_turn, start }
}

// ---------------------------------------------------------------------------
// Misc rule constants from game.rb (verbatim; consumers land with their
// mechanics — see docs/g1867_port_notes.md for the full inventory)
// ---------------------------------------------------------------------------

/// Revenue bonus hexes: a route that includes one of Toronto/Montreal/Quebec
/// AND Timmins earns +$40 (× train multiplier). (game.rb:318-319, 673-680)
pub const BONUS_CAPITALS: &[&str] = &["F16", "L12", "O7"];
pub const BONUS_REVENUE: &str = "D2";

/// A token slot in Montreal is reserved for the CN national. (game.rb:360)
pub const NATIONAL_RESERVATIONS: &[&str] = &["L12"];

/// Minors only startable once the first 3-train fires
/// `green_minors_available`. (game.rb:361)
pub const GREEN_CORPORATIONS: &[&str] = &["BBG", "LPS", "QLS", "SLA", "TGB", "THB"];

/// Loans (game.rb:376-378, 444-446, 499-530, 899-907): 72 × $50; take nets
/// $45, repay costs $50; interest $5/loan/OR; majors hold ≤5, minors ≤2.
pub const LOAN_VALUE: i32 = 50;
pub const NUM_LOANS: u32 = 72;
pub const INTEREST_RATE: i32 = 5;
pub const MAXIMUM_LOANS_MAJOR: u32 = 5;
pub const MAXIMUM_LOANS_MINOR: u32 = 2;

/// After a merger the survivor keeps ≤2 tokens (different hexes); a $40
/// third token is re-added (game.rb:335, 794-798, step/reduce_tokens.rb).
pub const LIMIT_TOKENS_AFTER_MERGER: u32 = 2;

/// Minimum par/min_price for minors (game.rb:336, init_corporations).
pub const MINIMUM_MINOR_PRICE: i32 = 50;

/// Stock-round bid rules (step/buy_sell_par_shares.rb:32-34).
pub const MIN_BID: i32 = 100;
pub const MAX_MINOR_PAR: i32 = 135;
pub const MAJOR_PHASE: u8 = 4;

/// Merge limits (step/merge.rb:411-412): ≤10 minors per merger; one player
/// contributes ≤6.
pub const LIMIT_MERGE: u32 = 10;
pub const LIMIT_OWNED_BY_ONE_ENTITY: u32 = 6;

// ---------------------------------------------------------------------------
// Corporation data (entities.rb CORPORATIONS: 8 majors + 16 minors;
// the CN national entry is NOT built as a corporation here —
// TODO(1867-national): it never operates and holds no shares; it lands as a
// special entity with the nationalization work)
// ---------------------------------------------------------------------------

/// entities.rb omits `shares:` for the majors, so the tobymao base default
/// applies: 20% president + eight 10% certificates — 1867 majors are
/// 10-share corporations.
const SHARES_1867_MAJOR: &[u8] = &[20, 10, 10, 10, 10, 10, 10, 10, 10];

/// A minor is a single 100% certificate (entities.rb `shares: [100]`);
/// the game treats it as size 2 for dividends and conversion
/// (game.rb:358 CORPORATION_SIZES, 368 "Minors are done as corporations
/// with a size of 2").
const SHARES_1867_MINOR: &[u8] = &[100];

/// Verbatim major fields: float_percent 20, always_market_price, tokens
/// [0, 20, 40], type major. No preprinted home — the home city is CHOSEN
/// when the corporation pars (HOME_TOKEN_TIMING = :par, game.rb:448-482);
/// `home_hex: ""` + a home-choice step is that representation —
/// TODO(1867-home): the choice step lands with the stock-round work.
fn major(sym: &'static str, name: &'static str) -> CorporationDef {
    CorporationDef {
        sym,
        name,
        token_prices: &[0, 20, 40],
        home_hex: "",
        home_city_index: 0,
        reserved: false,
        shares: SHARES_1867_MAJOR,
        float_percent: 20,
        capitalization: Capitalization::Incremental,
        corp_type: CorpType::Major,
        // game.rb:643-649/936-941 bump majors to 70% in the 2-player
        // variant — TODO(1867-2p), out of replay scope.
        max_ownership_percent: 60,
        always_market_price: true,
    }
}

/// Verbatim minor fields: float_percent 100, always_market_price, tokens
/// [0], shares [100], max_ownership_percent 100, type minor. Started by BID
/// in the stock round (min $100), parring at min(bid/2, 135) on a
/// par_1/par cell with the whole bid as treasury — TODO(1867-stock).
fn minor(sym: &'static str, name: &'static str) -> CorporationDef {
    CorporationDef {
        sym,
        name,
        token_prices: &[0],
        home_hex: "",
        home_city_index: 0,
        reserved: false,
        shares: SHARES_1867_MINOR,
        float_percent: 100,
        capitalization: Capitalization::Incremental,
        corp_type: CorpType::Minor,
        max_ownership_percent: 100,
        always_market_price: true,
    }
}

pub fn corporations() -> Vec<CorporationDef> {
    vec![
        // -- the 8 majors (entities.rb:97-188) --
        major("CNR", "Canadian Northern Railway"),
        major("CPR", "Canadian Pacific Railway"),
        major("C&O", "Chesapeake and Ohio Railway"),
        major("GTR", "Grand Trunk Railway"),
        major("GWR", "Great Western Railway"),
        major("ICR", "Intercolonial Railway"),
        major("NTR", "National Transcontinental Railway"),
        major("NYC", "New York Central Railroad"),
        // -- the 16 minors (entities.rb:189-380); BBG/LPS/QLS/SLA/TGB/THB
        // are the GREEN_CORPORATIONS (startable from phase 3) --
        minor("BBG", "Buffalo, Brantford, and Goderich"),
        minor("BO", "Brockville and Ottawa"),
        minor("CS", "Canada Southern"),
        minor("CV", "Credit Valley Railway"),
        minor("KP", "Kingston and Pembroke"),
        minor("LPS", "London and Port Stanley"),
        minor("OP", "Ottawa and Prescott"),
        minor("SLA", "St. Lawrence and Atlantic"),
        minor("TGB", "Toronto, Grey, and Bruce"),
        minor("TN", "Toronto and Nipissing"),
        minor("AE", "Algoma Eastern Railway"),
        minor("CA", "Canada Atlantic Railway"),
        minor("NO", "New York and Ottawa"),
        minor("PM", "Pere Marquette Railway"),
        minor("QLS", "Quebec and Lake St. John"),
        minor("THB", "Toronto, Hamilton and Buffalo"),
    ]
}

// ---------------------------------------------------------------------------
// Private company data (entities.rb COMPANIES, 6 companies)
//
// Every real private carries a `discount:` that sets the opening auction
// bid at `value - discount` (= $20 for C&SL, then $30/$40/$50/$60); the
// dutch single-item auction lowers it further by $5 per all-pass round.
// CompanyPriceUpToFace (game.rb:365): corporations may buy a private for
// $1..face — TODO(1867-companies). On the first 6-train,
// `nationalize_companies` closes every private paying the owner face value
// from the bank (game.rb:1047-1062) — TODO(1867-events).
// ---------------------------------------------------------------------------

pub fn companies() -> Vec<CompanyDef> {
    vec![
        // { name: 'rules until start of phase 3', sym: '3', value: 3,
        //   revenue: 0, desc: 'Hidden corporation',
        //   abilities: [{ type: 'blocks_hexes', hexes: ['M13'] }] }
        // The hidden company (game.rb:960-961): never auctioned, never
        // owned; blocks Granby M13 until the first 3-train closes it
        // (event_green_minors_available!, game.rb:980-991).
        CompanyDef {
            sym: "3",
            name: "rules until start of phase 3",
            value: 3,
            revenue: 0,
            discount: 0,
            auctionable: false,
            // TODO(1867-blocker): the Ruby ability has NO owner_type — it
            // blocks unconditionally while the (unowned) company is open.
            // Player is the closest existing gate; the track step must
            // treat an UNOWNED open blocker as live when the lay work lands.
            abilities: &[AbilityDef::BlocksHexes {
                owner_type: OwnerType::Player,
                hexes: &["M13"],
            }],
        },
        CompanyDef {
            sym: "C&SL",
            name: "Champlain & St. Lawrence",
            value: 30,
            revenue: 10,
            discount: 10,
            auctionable: true,
            abilities: &[],
        },
        // "they gain $10 extra revenue for each of their routes that
        // include Buffalo" (consumed by revenue_for, game.rb:667-671 —
        // TODO(1867-router)).
        CompanyDef {
            sym: "NFB",
            name: "Niagara Falls Bridge",
            value: 45,
            revenue: 15,
            discount: 15,
            auctionable: true,
            abilities: &[AbilityDef::HexBonus {
                owner_type: OwnerType::Corporation,
                hexes: &["F18"],
                amount: 10,
            }],
        },
        CompanyDef {
            sym: "MB",
            name: "Montreal Bridge",
            value: 60,
            revenue: 20,
            discount: 20,
            auctionable: true,
            abilities: &[AbilityDef::HexBonus {
                owner_type: OwnerType::Corporation,
                hexes: &["L12"],
                amount: 10,
            }],
        },
        CompanyDef {
            sym: "QB",
            name: "Quebec Bridge",
            value: 75,
            revenue: 25,
            discount: 25,
            auctionable: true,
            abilities: &[AbilityDef::HexBonus {
                owner_type: OwnerType::Corporation,
                hexes: &["O7"],
                amount: 10,
            }],
        },
        CompanyDef {
            sym: "SCT",
            name: "St. Clair Tunnel",
            value: 90,
            revenue: 30,
            discount: 30,
            auctionable: true,
            abilities: &[AbilityDef::HexBonus {
                owner_type: OwnerType::Corporation,
                hexes: &["A19", "A17"],
                amount: 10,
            }],
        },
    ]
}

// ---------------------------------------------------------------------------
// Train data (game.rb:200-304 TRAINS, 9 types)
//
// Every 1867 train distance is BUCKETED, e.g. for the 2-train:
//   distance: [{ nodes: [city offboard], pay: 2, visit: 2 },
//              { nodes: [town], pay: 0, visit: 99 }]
// i.e. it pays/visits N cities-or-offboards and runs through ANY number of
// towns free; the 5+5E inverts the buckets (pays offboards ONLY) and also
// requires a tokened counted stop. TrainDef.distance keeps the city-count
// N; the bucket semantics are router work — TODO(1867-router)
// (compute_stops, game.rb:706-739).
// ---------------------------------------------------------------------------

/// Trade-in discounts on the three phase-8 trains (identical map on each);
/// usable only in phase 8 (`train_trade_allowed` —
/// step/buy_train.rb `discountable_trains_allowed?`) — TODO(1867-trains).
const PHASE8_DISCOUNT: &[(&str, i32)] = &[
    ("5", 275),
    ("6", 325),
    ("7", 400),
    ("8", 500),
    ("2+2", 300),
    ("5+5E", 750),
];

/// Ruby `num: 'unlimited'` (the 8, 2+2 and 5+5E).
const UNLIMITED: u32 = u32::MAX;

pub fn trains() -> Vec<TrainDef> {
    vec![
        TrainDef {
            name: "2",
            distance: 2,
            price: 100,
            count: 10,
            rusts_on: Some("4"),
            multiplier: 1,
            obsolete_on: None,
            events: &[],
            available_on: None,
            discount: &[],
        },
        TrainDef {
            name: "3",
            distance: 3,
            price: 225,
            count: 7,
            rusts_on: Some("6"),
            multiplier: 1,
            obsolete_on: None,
            events: &["green_minors_available"],
            available_on: None,
            discount: &[],
        },
        TrainDef {
            name: "4",
            distance: 4,
            price: 350,
            count: 4,
            rusts_on: Some("8"),
            multiplier: 1,
            obsolete_on: None,
            events: &["majors_can_ipo", "trainless_nationalization"],
            available_on: None,
            discount: &[],
        },
        TrainDef {
            name: "5",
            distance: 5,
            price: 550,
            count: 4,
            rusts_on: None,
            multiplier: 1,
            obsolete_on: None,
            events: &["minors_cannot_start"],
            available_on: None,
            discount: &[],
        },
        TrainDef {
            name: "6",
            distance: 6,
            price: 650,
            count: 2,
            rusts_on: None,
            multiplier: 1,
            obsolete_on: None,
            events: &["nationalize_companies", "trainless_nationalization"],
            available_on: None,
            discount: &[],
        },
        TrainDef {
            name: "7",
            distance: 7,
            price: 800,
            count: 2,
            rusts_on: None,
            multiplier: 1,
            obsolete_on: None,
            events: &[],
            available_on: None,
            discount: &[],
        },
        // The permanent end-game train: unlimited supply, rusts the 4s,
        // nationalizes all remaining minors, opens trade-ins.
        TrainDef {
            name: "8",
            distance: 8,
            price: 1000,
            count: UNLIMITED,
            rusts_on: None,
            multiplier: 1,
            obsolete_on: None,
            events: &[
                "minors_nationalized",
                "trainless_nationalization",
                "train_trade_allowed",
            ],
            available_on: None,
            discount: PHASE8_DISCOUNT,
        },
        TrainDef {
            name: "2+2",
            distance: 2,
            price: 600,
            count: UNLIMITED,
            rusts_on: None,
            multiplier: 2,
            obsolete_on: None,
            events: &[],
            available_on: Some("8"),
            discount: PHASE8_DISCOUNT,
        },
        // Pays OFFBOARDS only (inverted buckets), ×2, needs a tokened
        // counted stop — TODO(1867-router).
        TrainDef {
            name: "5+5E",
            distance: 5,
            price: 1500,
            count: UNLIMITED,
            rusts_on: None,
            multiplier: 2,
            obsolete_on: None,
            events: &[],
            available_on: Some("8"),
            discount: PHASE8_DISCOUNT,
        },
    ]
}

// ---------------------------------------------------------------------------
// Phase data (game.rb:144-198 PHASES, 7 phases)
//
// Ruby train_limit is per-type ({minor: 2, major: 4}); `train_limit` holds
// the MAJOR limit, `minor_train_limit` the minor one. Phase '2' has no
// major limit in Ruby (majors cannot exist before phase 4) — the minor
// value is mirrored into both fields. Every phase has operating_rounds 2;
// phases 3-7 interleave a merger round after each OR
// (TODO(1867-merger)) and the final OR set is 3
// (game_end_set_final_turn!, game.rb:787-790 — TODO(1867-endgame)).
// 'export_train' status (phases 4-7): at the end of each OR the next depot
// train is exported to the CN, triggering phase change as if purchased
// (game.rb:323-327, 858-864 — TODO(1867-export)).
// ---------------------------------------------------------------------------

pub fn phases() -> Vec<PhaseDef> {
    vec![
        PhaseDef {
            name: "2",
            on: None,
            train_limit: 2, // Ruby: { minor: 2 } — no major limit
            minor_train_limit: Some(2),
            tiles: &["yellow"],
            operating_rounds: 2,
            status: &[],
        },
        PhaseDef {
            name: "3",
            on: Some("3"),
            train_limit: 4, // Ruby: { minor: 2, major: 4 }
            minor_train_limit: Some(2),
            tiles: &["yellow", "green"],
            operating_rounds: 2,
            status: &["can_buy_companies"],
        },
        PhaseDef {
            name: "4",
            on: Some("4"),
            train_limit: 3, // Ruby: { minor: 1, major: 3 }
            minor_train_limit: Some(1),
            tiles: &["yellow", "green"],
            operating_rounds: 2,
            status: &["can_buy_companies", "export_train"],
        },
        PhaseDef {
            name: "5",
            on: Some("5"),
            train_limit: 3, // Ruby: { minor: 1, major: 3 }
            minor_train_limit: Some(1),
            tiles: &["yellow", "green", "brown"],
            operating_rounds: 2,
            status: &["can_buy_companies", "export_train"],
        },
        PhaseDef {
            name: "6",
            on: Some("6"),
            train_limit: 2, // Ruby: { minor: 1, major: 2 }
            minor_train_limit: Some(1),
            tiles: &["yellow", "green", "brown"],
            operating_rounds: 2,
            status: &["export_train"],
        },
        PhaseDef {
            name: "7",
            on: Some("7"),
            train_limit: 2, // Ruby: { minor: 1, major: 2 }
            minor_train_limit: Some(1),
            tiles: &["yellow", "green", "brown", "gray"],
            operating_rounds: 2,
            status: &["export_train"],
        },
        PhaseDef {
            name: "8",
            on: Some("8"),
            train_limit: 2, // Ruby: { major: 2 } — no minors remain
            minor_train_limit: None,
            tiles: &["yellow", "green", "brown", "gray"],
            operating_rounds: 2,
            status: &[],
        },
    ]
}

// ---------------------------------------------------------------------------
// Starting cash, cert limits, bank cash (game.rb:25-29)
// ---------------------------------------------------------------------------

/// STARTING_CASH = { 2 => 420, 3 => 420, 4 => 315, 5 => 252, 6 => 210 }
pub fn starting_cash(num_players: u8) -> i32 {
    match num_players {
        2 | 3 => 420,
        4 => 315,
        5 => 252,
        6 => 210,
        _ => panic!("Invalid player count: {}", num_players),
    }
}

/// CERT_LIMIT = { 2 => 21, 3 => 21, 4 => 16, 5 => 13, 6 => 11 }.
/// CERT_LIMIT_CHANGE_ON_BANKRUPTCY = true (game.rb:321): on a bankruptcy
/// the limit is recomputed for one fewer player — TODO(1867-cert-limit).
pub fn cert_limit(num_players: u8) -> u8 {
    match num_players {
        2 | 3 => 21,
        4 => 16,
        5 => 13,
        6 => 11,
        _ => panic!("Invalid player count: {}", num_players),
    }
}

pub const BANK_CASH: i32 = 15_000;

// ---------------------------------------------------------------------------
// Stock market — the default 1-D COLUMN_MARKET (game.rb:37-66)
//
// Cell suffixes are COMBINABLE type flags (share_price.rb TYPE_MAP +
// MARKET_TEXT game.rb:352-357):
//   x = par_1 (minor par, $50-65)      z = par_2 (major par, $150-200)
//   p = par   (both, $70-135)          C = convert_range ($100-165)
//   m = max_price (minor cap, $165)
// Par-price queries by corp class (majors par on z+p, minors on x+p,
// capped at MAX_MINOR_PAR) and the G1867 movement quirks (minor on an `m`
// cell doesn't move right) land with the stock-round work —
// TODO(1867-market).
// ---------------------------------------------------------------------------

pub fn market_grid() -> Vec<Vec<Option<MarketCell>>> {
    use MarketZone::*;
    fn mc(price: i32, zones: &[MarketZone]) -> Option<MarketCell> {
        Some(MarketCell {
            price,
            zones: zones.to_vec(),
        })
    }

    // COLUMN_MARKET: a single row, 28 cells.
    vec![vec![
        mc(35, &[Normal]),
        mc(40, &[Normal]),
        mc(45, &[Normal]),
        mc(50, &[Par1]),                          // "50x"
        mc(55, &[Par1]),                          // "55x"
        mc(60, &[Par1]),                          // "60x"
        mc(65, &[Par1]),                          // "65x"
        mc(70, &[Par]),                           // "70p"
        mc(80, &[Par]),                           // "80p"
        mc(90, &[Par]),                           // "90p"
        mc(100, &[Par, ConvertRange]),            // "100pC"
        mc(110, &[Par, ConvertRange]),            // "110pC"
        mc(120, &[Par, ConvertRange]),            // "120pC"
        mc(135, &[Par, ConvertRange]),            // "135pC" (MAX_MINOR_PAR)
        mc(150, &[Par2, ConvertRange]),           // "150zC"
        mc(165, &[Par2, ConvertRange, MaxPrice]), // "165zCm"
        mc(180, &[Par2]),                         // "180z"
        mc(200, &[Par2]),                         // "200z"
        mc(220, &[Normal]),
        mc(245, &[Normal]),
        mc(270, &[Normal]),
        mc(300, &[Normal]),
        mc(330, &[Normal]),
        mc(360, &[Normal]),
        mc(400, &[Normal]),
        mc(440, &[Normal]),
        mc(490, &[Normal]),
        mc(540, &[Normal]),
    ]]
}

/// The OPTIONAL 2-D market variant (game.rb:68-142 GRID_MARKET; meta.rb
/// optional rule `:grid_market`). Not selectable yet — kept verbatim so the
/// optional-rule plumbing has the data when it lands (161 corpus games use
/// it). G1867::StockMarket 2-D quirks (ceiling up = right+down-one for
/// majors; minor on max_price up instead of right) — TODO(1867-market).
pub fn grid_market() -> Vec<Vec<Option<MarketCell>>> {
    use MarketZone::*;
    fn mc(price: i32, zones: &[MarketZone]) -> Option<MarketCell> {
        Some(MarketCell {
            price,
            zones: zones.to_vec(),
        })
    }

    vec![
        vec![
            None,
            None,
            None,
            None,
            mc(135, &[Normal]),
            mc(150, &[Normal]),
            mc(165, &[MaxPrice, ConvertRange]), // "165mC"
            mc(180, &[Normal]),
            mc(200, &[Par2]), // "200z"
            mc(220, &[Normal]),
            mc(245, &[Normal]),
            mc(270, &[Normal]),
            mc(300, &[Normal]),
            mc(330, &[Normal]),
            mc(360, &[Normal]),
            mc(400, &[Normal]),
            mc(440, &[Normal]),
            mc(490, &[Normal]),
            mc(540, &[Normal]),
        ],
        vec![
            None,
            None,
            None,
            mc(110, &[Normal]),
            mc(120, &[Normal]),
            mc(135, &[Normal]),
            mc(150, &[MaxPrice, ConvertRange]), // "150mC"
            mc(165, &[Par2]),                   // "165z"
            mc(180, &[Par2]),                   // "180z"
            mc(200, &[Normal]),
            mc(220, &[Normal]),
            mc(245, &[Normal]),
            mc(270, &[Normal]),
            mc(300, &[Normal]),
            mc(330, &[Normal]),
            mc(360, &[Normal]),
            mc(400, &[Normal]),
            mc(440, &[Normal]),
            mc(490, &[Normal]),
        ],
        vec![
            None,
            None,
            mc(90, &[Normal]),
            mc(100, &[Normal]),
            mc(110, &[Normal]),
            mc(120, &[Normal]),
            mc(135, &[Par, MaxPrice, ConvertRange]), // "135pmC"
            mc(150, &[Par2]),                        // "150z"
            mc(165, &[Normal]),
            mc(180, &[Normal]),
            mc(200, &[Normal]),
            mc(220, &[Normal]),
            mc(245, &[Normal]),
            mc(270, &[Normal]),
            mc(300, &[Normal]),
            mc(330, &[Normal]),
            mc(360, &[Normal]),
            mc(400, &[Normal]),
            mc(440, &[Normal]),
        ],
        vec![
            None,
            mc(70, &[Normal]),
            mc(80, &[Normal]),
            mc(90, &[Normal]),
            mc(100, &[Normal]),
            mc(110, &[Par]),                         // "110p"
            mc(120, &[Par, MaxPrice, ConvertRange]), // "120pmC"
            mc(135, &[Normal]),
            mc(150, &[Normal]),
            mc(165, &[Normal]),
            mc(180, &[Normal]),
            mc(200, &[Normal]),
        ],
        vec![
            mc(60, &[Normal]),
            mc(65, &[Normal]),
            mc(70, &[Normal]),
            mc(80, &[Normal]),
            mc(90, &[Par]),                     // "90p"
            mc(100, &[Par]),                    // "100p"
            mc(110, &[MaxPrice, ConvertRange]), // "110mC"
            mc(120, &[Normal]),
            mc(135, &[Normal]),
            mc(150, &[Normal]),
        ],
        vec![
            mc(55, &[Normal]),
            mc(60, &[Normal]),
            mc(65, &[Normal]),
            mc(70, &[Par]), // "70p"
            mc(80, &[Par]), // "80p"
            mc(90, &[Normal]),
            mc(100, &[MaxPrice, ConvertRange]), // "100mC"
            mc(110, &[Normal]),
        ],
        vec![
            mc(50, &[Normal]),
            mc(55, &[Normal]),
            mc(60, &[Par1]), // "60x"
            mc(65, &[Par1]), // "65x"
            mc(70, &[Normal]),
            mc(80, &[Normal]),
        ],
        vec![
            mc(45, &[Normal]),
            mc(50, &[Par1]), // "50x"
            mc(55, &[Par1]), // "55x"
            mc(60, &[Normal]),
            mc(65, &[Normal]),
        ],
        vec![
            mc(40, &[Normal]),
            mc(45, &[Normal]),
            mc(50, &[Normal]),
            mc(55, &[Normal]),
        ],
        vec![mc(35, &[Normal]), mc(40, &[Normal]), mc(45, &[Normal])],
    ]
}

// ---------------------------------------------------------------------------
// Tile counts (map.rb TILES) — definitions in tiles::tile_catalog_1867
// ---------------------------------------------------------------------------

pub fn tile_counts() -> Vec<(&'static str, u32)> {
    vec![
        ("3", 2),
        ("4", 4),
        ("5", 2),
        ("6", 2),
        ("7", u32::MAX), // Ruby: 'unlimited'
        ("8", u32::MAX),
        ("9", u32::MAX),
        ("14", 2),
        ("15", 4),
        ("16", 2),
        ("17", 2),
        ("18", 2),
        ("19", 2),
        ("20", 2),
        ("21", 2),
        ("22", 2),
        ("23", 5),
        ("24", 5),
        ("25", 4),
        ("26", 2),
        ("27", 2),
        ("28", 2),
        ("29", 2),
        ("30", 2),
        ("31", 2),
        ("39", 2),
        ("40", 2),
        ("41", 2),
        ("42", 2),
        ("43", 2),
        ("44", 2),
        ("45", 2),
        ("46", 2),
        ("47", 2),
        ("57", 2),
        ("58", 4),
        ("63", 3),
        ("70", 2),
        ("87", 2),
        ("88", 2),
        ("120", 1),
        ("122", 1),
        ("124", 1),
        ("201", 3),
        ("202", 3),
        ("204", 2),
        ("207", 5),
        ("208", 2),
        ("611", 3),
        ("619", 2),
        ("621", 2),
        ("622", 2),
        ("623", 3),
        ("624", 1),
        ("625", 1),
        ("626", 1),
        ("637", 1),
        ("639", 1),
        ("801", 2),
        ("911", 3),
        ("X1", 1),
        ("X2", 1),
        ("X3", 1),
        ("X4", 1),
        ("X5", 1),
        ("X6", 1),
        ("X7", 1),
        ("X8", 1),
    ]
}

// ---------------------------------------------------------------------------
// Hex grid definitions (map.rb HEXES)
//
// 94 hexes: 78 white, 5 gray, 2 yellow, 7 red, 2 blue. White-hex DSL in
// map.rb is hex ATTRIBUTES (untiled cities/towns, borders, stubs, labels),
// NOT preprinted track — those live here as HexDef data;
// `preprinted_hex_dsl` covers only the colored (fixed-track) hexes.
//
// Still-open map features — TODO(1867-map): stubs (G15 edge 1, L10 edge 0,
// K13 edge 4; StubsAreRestricted), labels Y/M/T/O + future_label (J12 gray
// O → tile X8) gating upgrades, the L12 TRIPLE city (three separate 1-slot
// cities — DoubleCity is the closest current fit; the third city holds a
// neutral CN token + the national reservation), grouped Detroit
// (A17+A19 count once), 4-tier offboard revenue (green/gray tiers),
// border-cost/impassable enforcement in the lay/graph code (the DATA is
// in `borders` below).
// ---------------------------------------------------------------------------

/// Plain white hexes (map.rb HEXES white => '').
const PLAIN_WHITE_HEXES: &[&str] = &[
    "B6", "B8", "C5", "C7", "C19", "D4", "D6", "D14", "E3", "E5", "E7", "E9", "F2", "F4", "F6",
    "F10", "F12", "F14", "G3", "G5", "G7", "G9", "G11", "G13", "H4", "H6", "H8", "H12", "I5",
    "I7", "I9", "I11", "I13", "J6", "J8", "J10", "J14", "K5", "K7", "K9", "L6", "L8", "M5", "M7",
    "N6", "O11",
];

pub fn hex_definitions() -> Vec<HexDef> {
    use HexType::*;

    const NO_BORDERS: &[BorderDef] = &[];
    const fn wall(edge: u8) -> BorderDef {
        BorderDef {
            edge,
            cost: None,
            impassable: true,
        }
    }
    const fn water80(edge: u8) -> BorderDef {
        BorderDef {
            edge,
            cost: Some(80),
            impassable: false,
        }
    }
    // Named border sets (const items so the slices are 'static).
    const WALL_0: &[BorderDef] = &[wall(0)];
    const WALL_2: &[BorderDef] = &[wall(2)];
    const WALL_3: &[BorderDef] = &[wall(3)];
    const WALL_5: &[BorderDef] = &[wall(5)];
    const WALL_0_5: &[BorderDef] = &[wall(0), wall(5)];
    const WALL_2_1: &[BorderDef] = &[wall(2), wall(1)];
    const WALL_3_4: &[BorderDef] = &[wall(3), wall(4)];
    const WALL_0_3_4: &[BorderDef] = &[wall(0), wall(3), wall(4)];
    const WALL_2_1_0_5: &[BorderDef] = &[wall(2), wall(1), wall(0), wall(5)];
    const WATER_2: &[BorderDef] = &[water80(2)];
    const WATER_5: &[BorderDef] = &[water80(5)];
    const WATER_0_5: &[BorderDef] = &[water80(0), water80(5)];
    const WATER_2_3: &[BorderDef] = &[water80(2), water80(3)];
    const WATER_5_0: &[BorderDef] = &[water80(5), water80(0)];

    let mut hexes: Vec<HexDef> = PLAIN_WHITE_HEXES
        .iter()
        .map(|&coord| HexDef {
            coord,
            hex_type: Blank,
            terrain_cost: 0,
            borders: NO_BORDERS,
        })
        .collect();

    hexes.extend(vec![
        // --- White hexes with impassable borders ---
        HexDef { coord: "D18", hex_type: Blank, terrain_cost: 0, borders: WALL_5 },
        HexDef { coord: "C9", hex_type: Blank, terrain_cost: 0, borders: WALL_0_5 },
        HexDef {
            coord: "D10",
            hex_type: Blank,
            terrain_cost: 0,
            borders: WALL_2_1_0_5,
        },
        HexDef { coord: "E11", hex_type: Blank, terrain_cost: 0, borders: WALL_2_1 },
        HexDef {
            coord: "C11",
            hex_type: Blank,
            terrain_cost: 0,
            borders: WALL_0_3_4,
        },
        HexDef { coord: "D12", hex_type: Blank, terrain_cost: 0, borders: WALL_3_4 },
        HexDef { coord: "C13", hex_type: Blank, terrain_cost: 0, borders: WALL_3 },
        // --- White hexes with water costs ---
        // K11: 'upgrade=cost:20,terrain:water' — a whole-hex lay cost.
        HexDef { coord: "K11", hex_type: Blank, terrain_cost: 20, borders: NO_BORDERS },
        // Per-EDGE water crossings of $80 (paid when track crosses the edge).
        HexDef { coord: "N8", hex_type: Blank, terrain_cost: 0, borders: WATER_0_5 },
        HexDef { coord: "N10", hex_type: Blank, terrain_cost: 0, borders: WATER_2_3 },
        HexDef { coord: "M11", hex_type: Blank, terrain_cost: 0, borders: WATER_2_3 },
        HexDef { coord: "O9", hex_type: Blank, terrain_cost: 0, borders: WATER_2 },
        // --- White city hexes ---
        HexDef {
            coord: "M9", // Trois-Rivières
            hex_type: City { revenue: 0, slots: 1 },
            terrain_cost: 0,
            borders: WATER_5_0,
        },
        HexDef { coord: "D8", hex_type: City { revenue: 0, slots: 1 }, terrain_cost: 0, borders: NO_BORDERS },
        HexDef { coord: "F8", hex_type: City { revenue: 0, slots: 1 }, terrain_cost: 0, borders: NO_BORDERS },
        HexDef { coord: "E13", hex_type: City { revenue: 0, slots: 1 }, terrain_cost: 0, borders: NO_BORDERS },
        HexDef { coord: "E15", hex_type: City { revenue: 0, slots: 1 }, terrain_cost: 0, borders: NO_BORDERS },
        HexDef { coord: "C17", hex_type: City { revenue: 0, slots: 1 }, terrain_cost: 0, borders: NO_BORDERS },
        HexDef { coord: "I15", hex_type: City { revenue: 0, slots: 1 }, terrain_cost: 0, borders: NO_BORDERS },
        HexDef { coord: "N12", hex_type: City { revenue: 0, slots: 1 }, terrain_cost: 0, borders: NO_BORDERS },
        // G15 Peterborough: 'city=revenue:0;stub=edge:1' — TODO(1867-map).
        HexDef { coord: "G15", hex_type: City { revenue: 0, slots: 1 }, terrain_cost: 0, borders: NO_BORDERS },
        // E17/D16/O7: 'city=revenue:0;label=Y' — TODO(1867-map) labels.
        HexDef { coord: "E17", hex_type: City { revenue: 0, slots: 1 }, terrain_cost: 0, borders: NO_BORDERS },
        HexDef { coord: "D16", hex_type: City { revenue: 0, slots: 1 }, terrain_cost: 0, borders: NO_BORDERS },
        HexDef { coord: "O7", hex_type: City { revenue: 0, slots: 1 }, terrain_cost: 0, borders: NO_BORDERS },
        // J12 Ottawa: label=Y + future_label O (gray X8) + $20 water.
        HexDef { coord: "J12", hex_type: City { revenue: 0, slots: 1 }, terrain_cost: 20, borders: NO_BORDERS },
        // --- White town hexes ---
        // L10 St. Jerome: also 'stub=edge:0' — TODO(1867-map).
        HexDef { coord: "L10", hex_type: Town { revenue: 0 }, terrain_cost: 0, borders: WATER_5 },
        HexDef { coord: "H14", hex_type: Town { revenue: 0 }, terrain_cost: 0, borders: WALL_0 },
        HexDef { coord: "C15", hex_type: Town { revenue: 0 }, terrain_cost: 0, borders: NO_BORDERS },
        HexDef { coord: "B18", hex_type: Town { revenue: 0 }, terrain_cost: 0, borders: NO_BORDERS },
        HexDef { coord: "H10", hex_type: Town { revenue: 0 }, terrain_cost: 0, borders: NO_BORDERS },
        HexDef { coord: "M13", hex_type: Town { revenue: 0 }, terrain_cost: 0, borders: NO_BORDERS },
        // K13 Cornwall: 'stub=edge:4' — TODO(1867-map).
        HexDef { coord: "K13", hex_type: Town { revenue: 0 }, terrain_cost: 0, borders: NO_BORDERS },
        // --- Gray hexes (preprinted) ---
        // D2 Timmins: the capital-bonus partner hex ($40 base + $40 bonus
        // with T/M/Q); holds a neutral CN green token until phase 3 —
        // TODO(1867-national).
        HexDef { coord: "D2", hex_type: City { revenue: 40, slots: 1 }, terrain_cost: 0, borders: NO_BORDERS },
        HexDef { coord: "C3", hex_type: Path, terrain_cost: 0, borders: NO_BORDERS },
        HexDef { coord: "E1", hex_type: Path, terrain_cost: 0, borders: NO_BORDERS },
        HexDef { coord: "B16", hex_type: Path, terrain_cost: 0, borders: NO_BORDERS },
        HexDef { coord: "L14", hex_type: Path, terrain_cost: 0, borders: NO_BORDERS },
        // --- Yellow hexes (preprinted) ---
        // L12 Montreal: a TRIPLE city in Ruby (third 1-slot city holds a
        // neutral CN token + the national reservation) — DoubleCity is the
        // closest current fit, TODO(1867-map).
        HexDef { coord: "L12", hex_type: DoubleCity { revenue: 40 }, terrain_cost: 20, borders: NO_BORDERS },
        HexDef { coord: "F16", hex_type: DoubleCity { revenue: 30 }, terrain_cost: 0, borders: NO_BORDERS },
        // --- Red hexes (offboard; 4-tier revenue, yellow/brown kept) ---
        HexDef { coord: "A7", hex_type: Offboard { yellow_revenue: 20, brown_revenue: 40 }, terrain_cost: 0, borders: NO_BORDERS },
        HexDef { coord: "F18", hex_type: Offboard { yellow_revenue: 30, brown_revenue: 50 }, terrain_cost: 0, borders: NO_BORDERS },
        HexDef { coord: "M15", hex_type: Offboard { yellow_revenue: 30, brown_revenue: 50 }, terrain_cost: 0, borders: NO_BORDERS },
        HexDef { coord: "O13", hex_type: Offboard { yellow_revenue: 20, brown_revenue: 40 }, terrain_cost: 0, borders: NO_BORDERS },
        HexDef { coord: "P8", hex_type: Offboard { yellow_revenue: 30, brown_revenue: 40 }, terrain_cost: 0, borders: NO_BORDERS },
        // A17+A19 are ONE grouped offboard (groups:Detroit, A17 hide:1) —
        // TODO(1867-router) counts it once.
        HexDef { coord: "A17", hex_type: Offboard { yellow_revenue: 30, brown_revenue: 50 }, terrain_cost: 0, borders: WALL_0 },
        HexDef { coord: "A19", hex_type: Offboard { yellow_revenue: 30, brown_revenue: 50 }, terrain_cost: 0, borders: WALL_3 },
        // --- Blue hexes (lake ports, flat revenue 10 at all phases) ---
        HexDef { coord: "E19", hex_type: Offboard { yellow_revenue: 10, brown_revenue: 10 }, terrain_cost: 0, borders: WALL_2 },
        HexDef { coord: "H16", hex_type: Offboard { yellow_revenue: 10, brown_revenue: 10 }, terrain_cost: 0, borders: WALL_3 },
    ]);

    hexes
}

// ---------------------------------------------------------------------------
// Preprinted (fixed-track) hex DSL — the COLORED hexes only (map.rb HEXES
// gray/yellow/red/blue, verbatim). White-hex attributes live in
// hex_definitions above.
// ---------------------------------------------------------------------------

pub fn preprinted_hex_dsl(coord: &str) -> Option<(&'static str, &'static str)> {
    // Returns (dsl_string, color_str)
    match coord {
        // --- Gray hexes ---
        "D2" => Some((
            "city=revenue:40;path=a:0,b:_0;path=a:1,b:_0;path=a:4,b:_0;\
             path=a:5,b:_0;border=edge:1;border=edge:4",
            "gray",
        )),
        "C3" => Some(("path=a:0,b:4;border=edge:4", "gray")),
        "E1" => Some(("path=a:1,b:5;border=edge:1", "gray")),
        "B16" => Some(("path=a:0,b:5", "gray")),
        "L14" => Some(("path=a:2,b:3", "gray")),
        // --- Yellow hexes ---
        "L12" => Some((
            "city=revenue:40;city=revenue:40;city=revenue:40,loc:5;path=a:1,b:_0;\
             path=a:3,b:_1;label=M;upgrade=cost:20,terrain:water",
            "yellow",
        )),
        "F16" => Some((
            "city=revenue:30;city=revenue:30;path=a:1,b:_0;path=a:4,b:_1;label=T",
            "yellow",
        )),
        // --- Red offboard hexes ---
        "A7" => Some((
            "offboard=revenue:yellow_20|green_30|brown_40|gray_40;path=a:4,b:_0;path=a:5,b:_0",
            "red",
        )),
        "F18" => Some((
            "offboard=revenue:yellow_30|green_40|brown_50|gray_60;path=a:2,b:_0",
            "red",
        )),
        "M15" => Some((
            "offboard=revenue:yellow_30|green_40|brown_50|gray_60;path=a:3,b:_0",
            "red",
        )),
        "O13" => Some((
            "offboard=revenue:yellow_20|green_30|brown_40|gray_40;path=a:2,b:_0;path=a:3,b:_0",
            "red",
        )),
        "P8" => Some((
            "offboard=revenue:yellow_30|green_30|brown_40|gray_40;path=a:2,b:_0;path=a:1,b:_0",
            "red",
        )),
        "A17" => Some((
            "offboard=revenue:yellow_30|green_40|brown_50|gray_70,hide:1,groups:Detroit;\
             path=a:5,b:_0;border=edge:0",
            "red",
        )),
        "A19" => Some((
            "offboard=revenue:yellow_30|green_40|brown_50|gray_70,groups:Detroit;\
             path=a:4,b:_0;border=edge:3",
            "red",
        )),
        // --- Blue hexes (lake ports; mapped to the gray tile color: fixed
        // track, never upgradeable) ---
        "E19" => Some((
            "offboard=revenue:10;path=a:3,b:_0;border=edge:2,type:impassable",
            "blue",
        )),
        "H16" => Some((
            "offboard=revenue:10;path=a:2,b:_0;path=a:4,b:_0;border=edge:3,type:impassable",
            "blue",
        )),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// In-crate tests — the TEST-5SHARE pattern: construction + the single-item
// auction driven through process_action_internal (dispatch gate included).
// The Python-side oracle is the fixture replay harness
// (tests/test_g1867_fixture_replay.py).
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use crate::actions::Action;
    use crate::entities::EntityId;
    use crate::game::BaseGame;
    use crate::rounds::Round;
    use crate::title::{CorpType, MarketZone};
    use std::collections::HashMap;

    const TITLE: &str = "1867";

    fn new_4p_game() -> BaseGame {
        let mut players = HashMap::new();
        players.insert(1, "Alice".to_string());
        players.insert(2, "Bob".to_string());
        players.insert(3, "Carol".to_string());
        players.insert(4, "Dave".to_string());
        BaseGame::build_titled(TITLE, vec![1, 2, 3, 4], players)
    }

    fn bid(game: &mut BaseGame, pid: u32, company: &str, price: i32) {
        game.process_action_internal(&Action::Bid {
            entity_id: pid.to_string(),
            company_sym: company.to_string(),
            price,
        })
        .unwrap_or_else(|e| panic!("bid {pid} {company} {price}: {e}"));
    }

    fn pass(game: &mut BaseGame, pid: u32) {
        game.process_action_internal(&Action::Pass {
            entity_id: pid.to_string(),
        })
        .unwrap_or_else(|e| panic!("pass {pid}: {e}"));
    }

    fn auction(game: &BaseGame) -> &crate::rounds::AuctionState {
        match &game.round {
            Round::Auction(s) => s,
            other => panic!("expected auction round, got {:?}", other.round_type_str()),
        }
    }

    fn active_player(game: &BaseGame) -> u32 {
        auction(game).active_player_id()
    }

    fn auctioning_sym(game: &BaseGame) -> String {
        let s = auction(game);
        let ci = s.auctioning.expect("a company up for auction");
        game.companies[ci].sym.clone()
    }

    #[test]
    fn construction_smoke() {
        let game = new_4p_game();
        assert_eq!(game.title, TITLE);
        // 8 majors + 16 minors (no CN yet — lands with nationalization).
        assert_eq!(game.corporations.len(), 24);
        let majors = game.corporations.iter().filter(|c| c.corp_type == CorpType::Major).count();
        let minors = game.corporations.iter().filter(|c| c.corp_type == CorpType::Minor).count();
        assert_eq!((majors, minors), (8, 16));
        // Majors: 10-share, president 20%, float at 20%. Minors: one 100%
        // cert, unit 100, float at 100%.
        let cnr = &game.corporations[game.corp_idx["CNR"]];
        assert_eq!(cnr.num_share_units(), 10);
        assert_eq!(cnr.float_percent, 20);
        let no = &game.corporations[game.corp_idx["NO"]];
        assert_eq!(no.shares.len(), 1);
        // Ruby's president/2 rule for single-cert corps: the 100% president
        // cert is TWO 50% shares ("Minors are done as corporations with a
        // size of 2", game.rb:368) — par costs 2×par, dividends split /2.
        assert_eq!(no.share_unit(), 50);
        assert_eq!(no.num_share_units(), 2);
        assert_eq!(no.float_percent, 100);
        // 4p economics.
        assert_eq!(game.starting_cash, 315);
        assert_eq!(game.cert_limit, 16);
        assert_eq!(game.players.iter().map(|p| p.cash).sum::<i32>() + game.bank.cash, 15_000);
        // The 1-D market: one row, combinable zone flags on the 165 cell.
        let grid = super::market_grid();
        assert_eq!(grid.len(), 1);
        assert_eq!(grid[0].len(), 28);
        let cell165 = grid[0][15].as_ref().unwrap();
        assert_eq!(cell165.price, 165);
        assert!(cell165.zones.contains(&MarketZone::Par2));
        assert!(cell165.zones.contains(&MarketZone::ConvertRange));
        assert!(cell165.zones.contains(&MarketZone::MaxPrice));
        // Map: 94 hexes; the catalog resolves every purchasable tile id.
        assert_eq!(game.hexes.len(), 94);
        for (id, _) in super::tile_counts() {
            assert!(game.tile_catalog.contains_key(id), "tile {id} missing from catalog");
        }
    }

    /// The opening lot: companies one at a time, cheapest first, hidden '3'
    /// excluded, opening bid = value - discount = $20.
    #[test]
    fn auction_opens_with_cheapest_company() {
        let game = new_4p_game();
        let s = auction(&game);
        assert!(s.single_item.is_some());
        // Queue: C&SL(30) NFB(45) MB(60) QB(75) SCT(90); '3' not offered.
        let syms: Vec<&str> = s
            .remaining_companies
            .iter()
            .map(|&ci| game.companies[ci].sym.as_str())
            .collect();
        assert_eq!(syms, vec!["C&SL", "NFB", "MB", "QB", "SCT"]);
        assert_eq!(auctioning_sym(&game), "C&SL");
        assert_eq!(game.companies[game.company_idx["C&SL"]].min_bid(), 20);
        assert_eq!(active_player(&game), 1);
    }

    /// The ascending auction, traced against fixture 21268's opening
    /// (docs/ruby_reference + tests/fixtures/1867/21268.json): bid 20 →
    /// pass → bid 25 → pass → pass resolves the lot to the $25 bidder, and
    /// the winner's right-hand neighbour opens the next company.
    #[test]
    fn ascending_auction_matches_fixture_prefix() {
        let mut game = new_4p_game();
        bid(&mut game, 1, "C&SL", 20);
        // With a bid standing, the next ACTIVE bidder after the high bidder
        // acts: player 2.
        assert_eq!(active_player(&game), 2);
        pass(&mut game, 2); // leaves the auction
        assert_eq!(active_player(&game), 3);
        bid(&mut game, 3, "C&SL", 25);
        assert_eq!(active_player(&game), 4);
        pass(&mut game, 4);
        assert_eq!(active_player(&game), 1);
        pass(&mut game, 1); // only player 3 remains → wins at 25
        let csl = &game.companies[game.company_idx["C&SL"]];
        assert_eq!(csl.owner, EntityId::player(3));
        assert_eq!(game.players[2].cash, 315 - 25);
        assert_eq!(game.bank.cash, 15_000 - 4 * 315 + 25);
        // Next lot: NFB at min 30, opened by the winner's right-hand
        // neighbour (player 4).
        assert_eq!(auctioning_sym(&game), "NFB");
        assert_eq!(game.companies[game.company_idx["NFB"]].min_bid(), 30);
        assert_eq!(active_player(&game), 4);
    }

    /// Bids must raise by $5 steps from the standing high bid and respect
    /// cash; turn order rejects out-of-turn bids.
    #[test]
    fn bid_validation() {
        let mut game = new_4p_game();
        // Out of turn.
        assert!(game
            .process_action_internal(&Action::Bid {
                entity_id: "2".into(),
                company_sym: "C&SL".into(),
                price: 20
            })
            .is_err());
        // Below minimum.
        assert!(game
            .process_action_internal(&Action::Bid {
                entity_id: "1".into(),
                company_sym: "C&SL".into(),
                price: 15
            })
            .is_err());
        // Off the $5 grid.
        assert!(game
            .process_action_internal(&Action::Bid {
                entity_id: "1".into(),
                company_sym: "C&SL".into(),
                price: 22
            })
            .is_err());
        // Not the company up for auction.
        assert!(game
            .process_action_internal(&Action::Bid {
                entity_id: "1".into(),
                company_sym: "NFB".into(),
                price: 30
            })
            .is_err());
        // Beyond cash.
        assert!(game
            .process_action_internal(&Action::Bid {
                entity_id: "1".into(),
                company_sym: "C&SL".into(),
                price: 320
            })
            .is_err());
        bid(&mut game, 1, "C&SL", 20);
        // A raise must be ≥ 25 for the next player.
        assert!(game
            .process_action_internal(&Action::Bid {
                entity_id: "2".into(),
                company_sym: "C&SL".into(),
                price: 20
            })
            .is_err());
        bid(&mut game, 2, "C&SL", 25);
    }

    /// Everyone declines → dutch auction: price drops $5; a bid then buys
    /// outright.
    #[test]
    fn all_decline_enters_dutch_and_bid_buys() {
        let mut game = new_4p_game();
        for pid in [1, 2, 3, 4] {
            assert_eq!(active_player(&game), pid);
            pass(&mut game, pid);
        }
        // Dutch: C&SL price dropped 20 → 15; rotation restarts at player 1.
        let s = auction(&game);
        assert!(s.single_item.as_ref().unwrap().dutch_mode);
        assert_eq!(game.companies[game.company_idx["C&SL"]].min_bid(), 15);
        assert_eq!(active_player(&game), 1);
        pass(&mut game, 1); // dutch pass eliminates
        assert_eq!(active_player(&game), 2);
        bid(&mut game, 2, "C&SL", 15); // dutch bid buys outright
        let csl = &game.companies[game.company_idx["C&SL"]];
        assert_eq!(csl.owner, EntityId::player(2));
        assert_eq!(game.players[1].cash, 315 - 15);
        // Player 3 (winner's neighbour) opens NFB; dutch mode reset.
        assert_eq!(auctioning_sym(&game), "NFB");
        assert_eq!(active_player(&game), 3);
        assert!(!auction(&game).single_item.as_ref().unwrap().dutch_mode);
    }

    /// The dutch price grinds to $0 → the first eligible player force-buys
    /// free (Ruby fakes the bid).
    #[test]
    fn dutch_grinds_to_zero_force_buy() {
        let mut game = new_4p_game();
        // Decline round (20), then dutch all-pass rounds at 15, 10, 5.
        for _ in 0..4 {
            for _ in 0..4 {
                let pid = active_player(&game);
                pass(&mut game, pid);
            }
        }
        // At $0 player 1 (rotation head) force-bought C&SL for free.
        let csl = &game.companies[game.company_idx["C&SL"]];
        assert_eq!(csl.owner, EntityId::player(1));
        assert_eq!(game.players[0].cash, 315);
        assert_eq!(csl.min_bid(), 0);
        // The next lot opens normally with player 2.
        assert_eq!(auctioning_sym(&game), "NFB");
        assert_eq!(active_player(&game), 2);
        assert_eq!(game.companies[game.company_idx["NFB"]].min_bid(), 30);
    }

    /// The whole auction runs to completion and hands the stock round to
    /// the player after the last winner.
    #[test]
    fn full_auction_reaches_stock_round() {
        let mut game = new_4p_game();
        // Each lot: the opener bids minimum, everyone else passes.
        // Winners rotate: P1 gets C&SL, P2 NFB, P3 MB, P4 QB, P1 SCT.
        let mut expected_owners = Vec::new();
        for _ in 0..5 {
            let opener = active_player(&game);
            let sym = auctioning_sym(&game);
            let min = game.companies[game.company_idx[&sym]].min_bid();
            bid(&mut game, opener, &sym, min);
            for _ in 0..3 {
                let pid = active_player(&game);
                pass(&mut game, pid);
            }
            expected_owners.push((sym, opener, min));
            if matches!(&game.round, Round::Stock(_)) {
                break;
            }
        }
        assert!(matches!(&game.round, Round::Stock(_)), "auction should hand to the stock round");
        assert_eq!(expected_owners.len(), 5);
        for (sym, owner, _price) in &expected_owners {
            assert_eq!(
                game.companies[game.company_idx[sym]].owner,
                EntityId::player(*owner),
                "{sym}"
            );
        }
        // Last winner was P1 (SCT) → priority deal P2.
        assert_eq!(expected_owners[4].1, 1);
        assert_eq!(game.priority_deal_player, 2);
        // Money conserved: players paid 20+30+40+50+60 = 200 to the bank.
        let paid: i32 = expected_owners.iter().map(|(_, _, p)| p).sum();
        assert_eq!(paid, 200);
        assert_eq!(game.bank.cash, 15_000 - 4 * 315 + paid);
        // 1830 frozen pin: a 1830 game built side-by-side still opens with
        // the waterfall format.
        let mut players = HashMap::new();
        players.insert(1, "A".to_string());
        players.insert(2, "B".to_string());
        let g1830 = BaseGame::build(vec![1, 2], players);
        assert!(auction(&g1830).single_item.is_none());
    }

    /// Incremental capitalization: floating brings NO lump sum and must not
    /// clobber treasury cash already there (a minor's founding bid).
    #[test]
    fn incremental_float_keeps_treasury_no_bank_grant() {
        let mut game = new_4p_game();
        let ci = game.corp_idx["NO"];
        let par = game.stock_market.par_price(100).expect("par 100");
        game.corporations[ci].ipo_price = Some(par.clone());
        game.corporations[ci].share_price = Some(par);
        game.corporations[ci].cash = 110; // the founding bid
        // The single 100% cert sold: 100% from IPO → floats.
        game.corporations[ci].set_share_owner(0, crate::entities::EntityId::player(2));
        assert!(game.corporations[ci].check_floated());
        let bank_before = game.bank.cash;
        game.check_float(ci);
        assert!(game.corporations[ci].floated);
        assert_eq!(game.corporations[ci].cash, 110, "no lump sum, bid kept");
        assert_eq!(game.bank.cash, bank_before, "bank untouched on incremental float");
    }

    /// Treasury-share buys pay the CORPORATION at the current MARKET price
    /// (incremental capitalization + always_market_price), not the bank at
    /// par.
    #[test]
    fn treasury_buy_pays_corporation_at_market_price() {
        let mut game = new_4p_game();
        let ci = game.corp_idx["CNR"];
        assert!(game.corporations[ci].always_market_price);
        // Par CNR at 80; the market price has since dropped to 70.
        let par = game.stock_market.par_price(80).expect("par 80");
        let market = game.stock_market.par_price(70).expect("cell 70");
        game.corporations[ci].ipo_price = Some(par);
        game.corporations[ci].share_price = Some(market);
        let ipo = crate::entities::EntityId::ipo("CNR");
        let n = game.corporations[ci].shares.len();
        for i in 0..n {
            game.corporations[ci].set_share_owner(i, ipo.clone());
        }
        game.corporations[ci].set_share_owner(0, crate::entities::EntityId::player(2));
        game.corporations[ci].owner_id = crate::entities::EntityId::player(2);
        // Player 1's stock turn.
        game.round = Round::Stock(crate::rounds::StockState::new(&[1, 2, 3, 4], 1));
        game.update_round_state();
        let bank_before = game.bank.cash;
        game.process_action_internal(&Action::BuyShares {
            entity_id: "1".into(),
            corporation_sym: "CNR".into(),
            shares: Vec::new(),
            percent: 10,
            source: "ipo".into(),
            share_indices: vec![1],
        })
        .expect("treasury buy");
        // Paid the market price (70, not par 80), into the corp's treasury.
        assert_eq!(game.players[0].cash, 315 - 70);
        assert_eq!(game.corporations[ci].cash, 70);
        assert_eq!(game.bank.cash, bank_before, "bank untouched on treasury buy");
        // 20% president + 10% = 30% sold ≥ float 20% → floated, still no
        // lump sum.
        assert!(game.corporations[ci].floated);
    }

    fn corp_bid(game: &mut BaseGame, pid: u32, corp: &str, price: i32) {
        game.process_action_internal(&Action::CorporationBid {
            entity_id: pid.to_string(),
            corporation_sym: corp.to_string(),
            price,
        })
        .unwrap_or_else(|e| panic!("corp bid {pid} {corp} {price}: {e}"));
    }

    /// Drive the opening auction to completion (each opener buys at
    /// minimum): P1 C&SL $20, P2 NFB $30, P3 MB $40, P4 QB $50, P1 SCT $60;
    /// priority deal to P2.
    fn game_in_stock_round() -> BaseGame {
        let mut game = new_4p_game();
        for _ in 0..5 {
            let opener = active_player(&game);
            let sym = auctioning_sym(&game);
            let min = game.companies[game.company_idx[&sym]].min_bid();
            bid(&mut game, opener, &sym, min);
            for _ in 0..3 {
                let pid = active_player(&game);
                pass(&mut game, pid);
            }
            if matches!(&game.round, Round::Stock(_)) {
                break;
            }
        }
        assert!(matches!(&game.round, Round::Stock(_)));
        game
    }

    /// The full in-SR minor founding cycle: selection bid → immediate home
    /// token choice → live auction → win pars at min(bid/2, 135) snapped
    /// down to a par_1/par cell, with the FULL bid as the minor's treasury.
    #[test]
    fn minor_founded_by_bid_in_stock_round() {
        let mut game = game_in_stock_round();
        // Priority deal: P2 opens (P1 won the last auction lot).
        let s = match &game.round {
            Round::Stock(s) => s,
            _ => unreachable!(),
        };
        assert_eq!(s.current_player_id(), 2);

        corp_bid(&mut game, 2, "CV", 100);
        // The opening bidder chooses the home city NOW; everything else is
        // blocked until the token lands.
        {
            let s = match &game.round {
                Round::Stock(s) => s,
                _ => unreachable!(),
            };
            assert_eq!(s.pending_home_tokens, vec![("CV".to_string(), 2)]);
        }
        assert!(game
            .process_action_internal(&Action::Pass { entity_id: "3".into() })
            .is_err());
        game.process_action_internal(&Action::PlaceToken {
            entity_id: "CV".into(),
            hex_id: "E15".into(),
            city_index: 0,
        })
        .expect("home token");
        // Token on the map, auction live: rotation from P2 → P3 acts.
        assert!(game.hexes[game.hex_idx["E15"]].tile.cities[0]
            .tokens
            .iter()
            .flatten()
            .any(|t| t.corporation_id == "CV"));
        corp_bid(&mut game, 3, "CV", 110);
        pass(&mut game, 4);
        pass(&mut game, 1);
        pass(&mut game, 2); // P3 alone → wins at 110
        let ci = game.corp_idx["CV"];
        let cv = &game.corporations[ci];
        // Par: min(110/2, 135) = 55 → the 55x (par_1) cell.
        assert_eq!(cv.ipo_price.as_ref().unwrap().price, 55);
        assert_eq!(cv.share_price.as_ref().unwrap().price, 55);
        assert!(cv.shares[0].owner == EntityId::player(3));
        assert_eq!(cv.owner_id, EntityId::player(3));
        assert_eq!(cv.cash, 110, "the FULL bid funds the treasury");
        assert!(cv.floated);
        // P3 paid the bid (auction QB? no — P3 won MB at 40 in the opener).
        assert_eq!(game.players[2].cash, 315 - 40 - 110);
        // The winner's turn is consumed: P4 acts next.
        let s = match &game.round {
            Round::Stock(s) => s,
            _ => unreachable!(),
        };
        assert!(s.bid_auction.is_none());
        assert_eq!(s.current_player_id(), 4);
        // Cash conservation across auction + founding.
        let total: i64 = game.players.iter().map(|p| p.cash as i64).sum::<i64>()
            + game.corporations.iter().map(|c| c.cash as i64).sum::<i64>()
            + game.bank.cash as i64;
        assert_eq!(total, 15_000);
    }

    /// Flat-top adjacency (LAYOUT = :flat): the map relationships that
    /// pinned the delta table — G15's stub edge 1 faces Toronto F16;
    /// Toronto's city paths exit a:1 → E17 Hamilton and a:4 → G15;
    /// D2 Timmins edge 1 pairs with C3's edge 4.
    #[test]
    fn flat_layout_adjacency() {
        let game = new_4p_game();
        let n = |hex: &str, dir: u8| game.hex_adjacency[hex].get(&dir).cloned();
        assert_eq!(n("G15", 1), Some("F16".to_string()));
        assert_eq!(n("F16", 4), Some("G15".to_string()));
        assert_eq!(n("F16", 1), Some("E17".to_string()));
        assert_eq!(n("D2", 1), Some("C3".to_string()));
        assert_eq!(n("C3", 4), Some("D2".to_string()));
        assert_eq!(n("D2", 0), Some("D4".to_string()));
        // 1830 stays pointy: H12 dir 0 (upper-right) is I11 there.
        let mut players = HashMap::new();
        players.insert(1, "A".to_string());
        players.insert(2, "B".to_string());
        let g1830 = BaseGame::build(vec![1, 2], players);
        assert_eq!(g1830.hex_adjacency["H12"].get(&0).cloned(), Some("I11".to_string()));
    }

    /// The 1867 two-lay rule: a second yellow lay costs $20 and must hit a
    /// fresh hex; a third lay is rejected; an explicit pass after one lay
    /// moves the turn on (the fixture pattern).
    #[test]
    fn two_tile_lays_with_surcharge() {
        let mut game = game_in_stock_round();
        // Found CV (home E15 Guelph) and let everyone else pass the SR.
        corp_bid(&mut game, 2, "CV", 100);
        game.process_action_internal(&Action::PlaceToken {
            entity_id: "CV".into(),
            hex_id: "E15".into(),
            city_index: 0,
        })
        .expect("home token");
        for _ in 0..3 {
            let s = match &game.round {
                Round::Stock(s) => s.clone(),
                _ => unreachable!(),
            };
            let pid = s
                .bid_auction
                .as_ref()
                .and_then(|a| game.stock_bid_active_player(a))
                .unwrap();
            pass(&mut game, pid);
        }
        // Everyone passes out of the SR → OR1 with CV operating.
        loop {
            match &game.round {
                Round::Stock(s) => {
                    let pid = s.current_player_id();
                    pass(&mut game, pid);
                }
                Round::Operating(_) => break,
                _ => unreachable!(),
            }
        }
        let lay = |game: &mut BaseGame, hex: &str, tile: &str, rot: u8| {
            game.process_action_internal(&Action::LayTile {
                entity_id: "CV".into(),
                hex_id: hex.into(),
                tile_id: tile.into(),
                rotation: rot,
            })
        };
        // First lay: yellow city on the home hex (free).
        lay(&mut game, "E15", "5", 0).expect("first lay");
        assert_eq!(game.corporations[game.corp_idx["CV"]].cash, 100);
        // Reusing the same hex is forbidden for the second slot.
        assert!(lay(&mut game, "E15", "6", 0).is_err());
        // Second lay on a fresh connected hex costs $20.
        // E15 tile 5 rot 0 exits edges 0 (→E17) and 1 (→D16).
        lay(&mut game, "E17", "5", 3).expect("second lay");
        assert_eq!(game.corporations[game.corp_idx["CV"]].cash, 80);
        // The allowance is exhausted: pc moved off Track (a third lay is
        // rejected by the dispatch gate).
        assert!(lay(&mut game, "D14", "9", 0).is_err());
    }

    /// Phase windows: green minors are not biddable in phase 2; majors are
    /// not parrable before phase 4.
    #[test]
    fn phase_windows_gate_founding() {
        let mut game = game_in_stock_round();
        assert!(game
            .process_action_internal(&Action::CorporationBid {
                entity_id: "2".into(),
                corporation_sym: "BBG".into(),
                price: 100,
            })
            .is_err());
        assert!(game
            .process_action_internal(&Action::Par {
                entity_id: "2".into(),
                corporation_sym: "CNR".into(),
                share_price: 80,
            })
            .is_err());
        // A normal (non-green) minor bid still works afterwards.
        corp_bid(&mut game, 2, "CV", 100);
    }

    /// A bidder who can no longer afford the raised minimum is dropped from
    /// the auction automatically (PassableAuction#add_bid).
    #[test]
    fn unaffordable_raise_drops_bidder() {
        let mut game = new_4p_game();
        // Impoverish player 2 so they can open at 20 but not follow at 30+.
        let p2 = game.players.iter().position(|p| p.id == 2).unwrap();
        game.players[p2].cash = 27;
        game.bank.cash += 315 - 27;
        bid(&mut game, 1, "C&SL", 20);
        bid(&mut game, 2, "C&SL", 25);
        // P3 raises to 30 → min becomes 35; P2 (cash 27) is auto-dropped,
        // their bid removed.
        bid(&mut game, 3, "C&SL", 30);
        let s = auction(&game);
        let si = s.single_item.as_ref().unwrap();
        assert!(!si.active_bidders.contains(&2));
        let ci = game.company_idx["C&SL"];
        assert!(s.bids[&ci].iter().all(|b| b.player_id != 2));
        // P4 then P1 pass → P3 wins at 30 (their standing bid).
        pass(&mut game, 4);
        pass(&mut game, 1);
        assert_eq!(
            game.companies[ci].owner,
            EntityId::player(3)
        );
        assert_eq!(game.players[2].cash, 315 - 30);
    }
}
