//! Static game data for the 1830: Railways & Robber Barons title.
//!
//! All data is derived from the Python engine's g1830.py.
//! These are plain Rust structs used as templates during game construction —
//! they are NOT pyclass types.

use std::collections::HashMap;

use super::{
    AbilityDef, AbilityWhen, Capitalization, CompanyDef, CorporationDef, HexDef, HexType,
    MarketCell, MarketZone, OwnerType, PhaseDef, ShareSource, TrainDef,
};
use crate::steps::{FinishedRound, RoundStart, RoundTransition, StepDesc, StepKind};

// ---------------------------------------------------------------------------
// Round descriptions — the per-title ordered step lists
// ---------------------------------------------------------------------------

/// 1830's operating-round step list, mirroring the PYTHON reference exactly
/// (g1830.py::operating_round, including order):
/// Bankrupt, Exchange, SpecialTrack, SpecialToken, BuyCompany, HomeToken,
/// Track, Token, Route, Dividend, DiscardTrain, BuyTrain,
/// [BuyCompany, {"blocks": True}].
pub fn operating_steps() -> &'static [StepDesc] {
    const STEPS: &[StepDesc] = &[
        StepDesc::non_blocking(StepKind::Bankrupt),
        StepDesc::non_blocking(StepKind::Exchange),
        StepDesc::non_blocking(StepKind::SpecialTrack),
        StepDesc::non_blocking(StepKind::SpecialToken),
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

/// 1830's stock-round step list (base.py::stock_round):
/// DiscardTrain, Exchange, SpecialTrack, BuySellParShares.
pub fn stock_steps() -> &'static [StepDesc] {
    const STEPS: &[StepDesc] = &[
        StepDesc::blocking(StepKind::DiscardTrain),
        StepDesc::non_blocking(StepKind::Exchange),
        StepDesc::non_blocking(StepKind::SpecialTrack),
        StepDesc::blocking(StepKind::BuySellParShares),
    ];
    STEPS
}

/// 1830's auction-round step list (base.py::new_auction_round):
/// CompanyPendingPar, WaterfallAuction.
pub fn auction_steps() -> &'static [StepDesc] {
    const STEPS: &[StepDesc] = &[
        StepDesc::blocking(StepKind::CompanyPendingPar),
        StepDesc::blocking(StepKind::WaterfallAuction),
    ];
    STEPS
}

/// 1830's round flow (base.py::next_round!): the opening waterfall auction
/// flows into the first stock round; a stock round starts a set of
/// `phase.operating_rounds` operating rounds (the count fixed at set start);
/// the set's last OR hands back to a stock round and the game's `turn`
/// counter increments.
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
// Definition structs
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Corporation data (8 corporations)
// ---------------------------------------------------------------------------

/// Every 1830 corporation is a 10-share corp: a 20% president's certificate
/// + eight 10% certificates (g1830.py `shares: [20] + [10]*8`).
const SHARES_1830: &[u8] = &[20, 10, 10, 10, 10, 10, 10, 10, 10];

pub fn corporations() -> Vec<CorporationDef> {
    vec![
        CorporationDef {
            sym: "PRR",
            name: "Pennsylvania Railroad",
            token_prices: &[0, 40, 100, 100],
            home_hex: "H12",
            home_city_index: 0,
            reserved: false,
            shares: SHARES_1830,
            float_percent: 60,
            capitalization: Capitalization::Full,
        },
        CorporationDef {
            sym: "NYC",
            name: "New York Central Railroad",
            token_prices: &[0, 40, 100, 100],
            home_hex: "E19",
            home_city_index: 0,
            reserved: false,
            shares: SHARES_1830,
            float_percent: 60,
            capitalization: Capitalization::Full,
        },
        CorporationDef {
            sym: "CPR",
            name: "Canadian Pacific Railroad",
            token_prices: &[0, 40, 100, 100],
            home_hex: "A19",
            home_city_index: 0,
            reserved: false,
            shares: SHARES_1830,
            float_percent: 60,
            capitalization: Capitalization::Full,
        },
        CorporationDef {
            sym: "B&O",
            name: "Baltimore & Ohio Railroad",
            token_prices: &[0, 40, 100],
            home_hex: "I15",
            home_city_index: 0,
            reserved: false,
            shares: SHARES_1830,
            float_percent: 60,
            capitalization: Capitalization::Full,
        },
        CorporationDef {
            sym: "C&O",
            name: "Chesapeake & Ohio Railroad",
            token_prices: &[0, 40, 100],
            home_hex: "F6",
            home_city_index: 0,
            reserved: false,
            shares: SHARES_1830,
            float_percent: 60,
            capitalization: Capitalization::Full,
        },
        CorporationDef {
            sym: "ERIE",
            name: "Erie Railroad",
            token_prices: &[0, 40, 100],
            home_hex: "E11",
            home_city_index: 0,
            reserved: true,
            shares: SHARES_1830,
            float_percent: 60,
            capitalization: Capitalization::Full,
        },
        CorporationDef {
            sym: "NYNH",
            name: "New York, New Haven & Hartford Railroad",
            token_prices: &[0, 40],
            home_hex: "G19",
            home_city_index: 1,
            reserved: false,
            shares: SHARES_1830,
            float_percent: 60,
            capitalization: Capitalization::Full,
        },
        CorporationDef {
            sym: "B&M",
            name: "Boston & Maine Railroad",
            token_prices: &[0, 40],
            home_hex: "E23",
            home_city_index: 0,
            reserved: false,
            shares: SHARES_1830,
            float_percent: 60,
            capitalization: Capitalization::Full,
        },
    ]
}

// ---------------------------------------------------------------------------
// Private company data (6 companies)
// ---------------------------------------------------------------------------

pub fn companies() -> Vec<CompanyDef> {
    vec![
        CompanyDef {
            sym: "SV",
            name: "Schuylkill Valley",
            value: 20,
            revenue: 5,
            abilities: &[AbilityDef::BlocksHexes {
                owner_type: OwnerType::Player,
                hexes: &["G15"],
            }],
        },
        CompanyDef {
            sym: "CS",
            name: "Champlain & St.Lawrence",
            value: 40,
            revenue: 10,
            abilities: &[
                AbilityDef::BlocksHexes {
                    owner_type: OwnerType::Player,
                    hexes: &["B20"],
                },
                AbilityDef::TileLay {
                    owner_type: OwnerType::Corporation,
                    hexes: &["B20"],
                    tiles: &["3", "4", "58"],
                    when: AbilityWhen::OwningCorpOrTurn,
                    count: 1,
                },
            ],
        },
        CompanyDef {
            sym: "DH",
            name: "Delaware & Hudson",
            value: 70,
            revenue: 15,
            abilities: &[
                AbilityDef::BlocksHexes {
                    owner_type: OwnerType::Player,
                    hexes: &["F16"],
                },
                AbilityDef::Teleport {
                    owner_type: OwnerType::Corporation,
                    hexes: &["F16"],
                    tiles: &["57"],
                },
            ],
        },
        CompanyDef {
            sym: "MH",
            name: "Mohawk & Hudson",
            value: 110,
            revenue: 20,
            abilities: &[
                AbilityDef::BlocksHexes {
                    owner_type: OwnerType::Player,
                    hexes: &["D18"],
                },
                AbilityDef::Exchange {
                    owner_type: OwnerType::Player,
                    corporations: &["NYC"],
                    from: &[ShareSource::Ipo, ShareSource::Market],
                    when: AbilityWhen::Any,
                },
            ],
        },
        CompanyDef {
            sym: "CA",
            name: "Camden & Amboy",
            value: 160,
            revenue: 25,
            abilities: &[
                AbilityDef::BlocksHexes {
                    owner_type: OwnerType::Player,
                    hexes: &["H18"],
                },
                // "shares": "PRR_1" — a normal 10% share granted on purchase.
                AbilityDef::Shares {
                    corporation: "PRR",
                    share_index: 1,
                },
            ],
        },
        CompanyDef {
            sym: "BO",
            name: "Baltimore & Ohio",
            value: 220,
            revenue: 30,
            abilities: &[
                AbilityDef::BlocksHexes {
                    owner_type: OwnerType::Player,
                    hexes: &["I13", "I15"],
                },
                AbilityDef::Close {
                    when: AbilityWhen::BoughtTrain,
                    corporation: "B&O",
                },
                AbilityDef::NoBuy,
                // "shares": "B&O_0" — the president's certificate; granting it
                // triggers the pending B&O par.
                AbilityDef::Shares {
                    corporation: "B&O",
                    share_index: 0,
                },
            ],
        },
    ]
}

// ---------------------------------------------------------------------------
// Train data (6 types, 40 total)
// ---------------------------------------------------------------------------

pub fn trains() -> Vec<TrainDef> {
    vec![
        TrainDef {
            name: "2",
            events: &[],
            distance: 2,
            price: 80,
            count: 6,
            rusts_on: Some("4"),
            available_on: None,
            discount: &[],
        },
        TrainDef {
            name: "3",
            events: &[],
            distance: 3,
            price: 180,
            count: 5,
            rusts_on: Some("6"),
            available_on: None,
            discount: &[],
        },
        TrainDef {
            name: "4",
            events: &[],
            distance: 4,
            price: 300,
            count: 4,
            rusts_on: Some("D"),
            available_on: None,
            discount: &[],
        },
        TrainDef {
            name: "5",
            events: &["close_companies"],
            distance: 5,
            price: 450,
            count: 3,
            rusts_on: None,
            available_on: None,
            discount: &[],
        },
        TrainDef {
            name: "6",
            events: &[],
            distance: 6,
            price: 630,
            count: 2,
            rusts_on: None,
            available_on: None,
            discount: &[],
        },
        TrainDef {
            name: "D",
            events: &[],
            distance: 999,
            price: 1100,
            count: 20,
            rusts_on: None,
            available_on: Some("6"),
            discount: &[("4", 300), ("5", 300), ("6", 300)],
        },
    ]
}

// ---------------------------------------------------------------------------
// Phase data (6 phases)
// ---------------------------------------------------------------------------

pub fn phases() -> Vec<PhaseDef> {
    vec![
        PhaseDef {
            name: "2",
            on: None,
            status: &[],
            train_limit: 4,
            tiles: &["yellow"],
            operating_rounds: 1,
        },
        PhaseDef {
            name: "3",
            on: Some("3"),
            status: &["can_buy_companies"],
            train_limit: 4,
            tiles: &["yellow", "green"],
            operating_rounds: 2,
        },
        PhaseDef {
            name: "4",
            on: Some("4"),
            status: &["can_buy_companies"],
            train_limit: 3,
            tiles: &["yellow", "green"],
            operating_rounds: 2,
        },
        PhaseDef {
            name: "5",
            on: Some("5"),
            status: &[],
            train_limit: 2,
            tiles: &["yellow", "green", "brown"],
            operating_rounds: 3,
        },
        PhaseDef {
            name: "6",
            on: Some("6"),
            status: &[],
            train_limit: 2,
            tiles: &["yellow", "green", "brown"],
            operating_rounds: 3,
        },
        PhaseDef {
            name: "D",
            on: Some("D"),
            status: &[],
            train_limit: 2,
            tiles: &["yellow", "green", "brown"],
            operating_rounds: 3,
        },
    ]
}

// ---------------------------------------------------------------------------
// Starting cash, cert limits, bank cash
// ---------------------------------------------------------------------------

pub fn starting_cash(num_players: u8) -> i32 {
    match num_players {
        2 => 1200,
        3 => 800,
        4 => 600,
        5 => 480,
        6 => 400,
        _ => panic!("Invalid player count: {}", num_players),
    }
}

pub fn cert_limit(num_players: u8) -> u8 {
    match num_players {
        2 => 28,
        3 => 20,
        4 => 16,
        5 => 13,
        6 => 11,
        _ => panic!("Invalid player count: {}", num_players),
    }
}

pub const BANK_CASH: i32 = 12000;

// ---------------------------------------------------------------------------
// Stock market grid
// ---------------------------------------------------------------------------

/// Returns the stock market grid. Each inner vec is a row (left to right).
/// `None` = empty cell (below-market dead zone).
pub fn market_grid() -> Vec<Vec<Option<MarketCell>>> {
    fn mc(price: i32, zone: MarketZone) -> Option<MarketCell> {
        Some(MarketCell { price, zone })
    }

    use MarketZone::*;

    vec![
        // Row 0
        vec![
            mc(60, Yellow),
            mc(67, Normal),
            mc(71, Normal),
            mc(76, Normal),
            mc(82, Normal),
            mc(90, Normal),
            mc(100, Par),
            mc(112, Normal),
            mc(126, Normal),
            mc(142, Normal),
            mc(160, Normal),
            mc(180, Normal),
            mc(200, Normal),
            mc(225, Normal),
            mc(250, Normal),
            mc(275, Normal),
            mc(300, Normal),
            mc(325, Normal),
            mc(350, Normal),
        ],
        // Row 1
        vec![
            mc(53, Yellow),
            mc(60, Yellow),
            mc(66, Normal),
            mc(70, Normal),
            mc(76, Normal),
            mc(82, Normal),
            mc(90, Par),
            mc(100, Normal),
            mc(112, Normal),
            mc(126, Normal),
            mc(142, Normal),
            mc(160, Normal),
            mc(180, Normal),
            mc(200, Normal),
            mc(220, Normal),
            mc(240, Normal),
            mc(260, Normal),
            mc(280, Normal),
            mc(300, Normal),
        ],
        // Row 2
        vec![
            mc(46, Yellow),
            mc(55, Yellow),
            mc(60, Yellow),
            mc(65, Normal),
            mc(70, Normal),
            mc(76, Normal),
            mc(82, Par),
            mc(90, Normal),
            mc(100, Normal),
            mc(111, Normal),
            mc(125, Normal),
            mc(140, Normal),
            mc(155, Normal),
            mc(170, Normal),
            mc(185, Normal),
            mc(200, Normal),
        ],
        // Row 3
        vec![
            mc(39, Orange),
            mc(48, Yellow),
            mc(54, Yellow),
            mc(60, Yellow),
            mc(66, Normal),
            mc(71, Normal),
            mc(76, Par),
            mc(82, Normal),
            mc(90, Normal),
            mc(100, Normal),
            mc(110, Normal),
            mc(120, Normal),
            mc(130, Normal),
        ],
        // Row 4
        vec![
            mc(32, Orange),
            mc(41, Orange),
            mc(48, Yellow),
            mc(55, Yellow),
            mc(62, Normal),
            mc(67, Normal),
            mc(71, Par),
            mc(76, Normal),
            mc(82, Normal),
            mc(90, Normal),
            mc(100, Normal),
        ],
        // Row 5
        vec![
            mc(25, Brown),
            mc(34, Orange),
            mc(42, Orange),
            mc(50, Yellow),
            mc(58, Yellow),
            mc(65, Normal),
            mc(67, Par),
            mc(71, Normal),
            mc(75, Normal),
            mc(80, Normal),
        ],
        // Row 6
        vec![
            mc(18, Brown),
            mc(27, Brown),
            mc(36, Orange),
            mc(45, Orange),
            mc(54, Yellow),
            mc(63, Normal),
            mc(67, Normal),
            mc(69, Normal),
            mc(70, Normal),
        ],
        // Row 7
        vec![
            mc(10, Brown),
            mc(20, Brown),
            mc(30, Brown),
            mc(40, Orange),
            mc(50, Yellow),
            mc(60, Yellow),
            mc(67, Normal),
            mc(68, Normal),
        ],
        // Row 8
        vec![
            None,
            mc(10, Brown),
            mc(20, Brown),
            mc(30, Brown),
            mc(40, Orange),
            mc(50, Yellow),
            mc(60, Yellow),
        ],
        // Row 9
        vec![
            None,
            None,
            mc(10, Brown),
            mc(20, Brown),
            mc(30, Brown),
            mc(40, Orange),
            mc(50, Yellow),
        ],
        // Row 10
        vec![
            None,
            None,
            None,
            mc(10, Brown),
            mc(20, Brown),
            mc(30, Brown),
            mc(40, Orange),
        ],
    ]
}

// ---------------------------------------------------------------------------
// Tile counts (for unplaced-tiles encoder feature)
// ---------------------------------------------------------------------------

pub fn tile_counts() -> Vec<(&'static str, u32)> {
    vec![
        ("1", 1),
        ("2", 1),
        ("3", 2),
        ("4", 2),
        ("7", 4),
        ("8", 8),
        ("9", 7),
        ("14", 3),
        ("15", 2),
        ("16", 1),
        ("18", 1),
        ("19", 1),
        ("20", 1),
        ("23", 3),
        ("24", 3),
        ("25", 1),
        ("26", 1),
        ("27", 1),
        ("28", 1),
        ("29", 1),
        ("39", 1),
        ("40", 1),
        ("41", 2),
        ("42", 2),
        ("43", 2),
        ("44", 1),
        ("45", 2),
        ("46", 2),
        ("47", 1),
        ("53", 2),
        ("54", 1),
        ("55", 1),
        ("56", 1),
        ("57", 4),
        ("58", 2),
        ("59", 2),
        ("61", 2),
        ("62", 1),
        ("63", 3),
        ("64", 1),
        ("65", 1),
        ("66", 1),
        ("67", 1),
        ("68", 1),
        ("69", 1),
        ("70", 1),
    ]
}

/// Tile city definitions: tile_id -> list of city slot counts.
/// Only tiles that have cities are listed.
pub fn tile_cities(tile_id: &str) -> Option<Vec<u8>> {
    match tile_id {
        // Yellow city tiles
        "5" | "6" | "53" | "57" | "61" => Some(vec![1]),
        "14" | "15" | "63" => Some(vec![2]),
        "54" | "59" | "64" | "65" | "66" | "67" | "68" => Some(vec![1, 1]),
        // Green city tiles
        "12" => Some(vec![1]),
        "205" | "206" => Some(vec![2]),
        "619" => Some(vec![2]),
        // Brown city tiles
        "39" | "40" => Some(vec![2]),
        "41" | "42" | "43" | "44" | "45" | "46" | "47" => Some(vec![2]),
        "611" => Some(vec![2]),
        // Gray city tiles
        "51" => Some(vec![2]),
        // Double city tiles
        "62" => Some(vec![2, 2]),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Hex grid definitions
// ---------------------------------------------------------------------------

/// Returns all hex definitions for the 1830 map.
pub fn hex_definitions() -> Vec<HexDef> {
    use HexType::*;

    vec![
        // --- Red hexes (offboard) ---
        HexDef {
            coord: "F2",
            hex_type: Offboard {
                yellow_revenue: 40,
                brown_revenue: 70,
            },
            terrain_cost: 0,
        },
        HexDef {
            coord: "I1",
            hex_type: Offboard {
                yellow_revenue: 30,
                brown_revenue: 60,
            },
            terrain_cost: 0,
        },
        HexDef {
            coord: "J2",
            hex_type: Offboard {
                yellow_revenue: 30,
                brown_revenue: 60,
            },
            terrain_cost: 0,
        },
        HexDef {
            coord: "A9",
            hex_type: Offboard {
                yellow_revenue: 30,
                brown_revenue: 50,
            },
            terrain_cost: 0,
        },
        HexDef {
            coord: "A11",
            hex_type: Offboard {
                yellow_revenue: 30,
                brown_revenue: 50,
            },
            terrain_cost: 0,
        },
        HexDef {
            coord: "K13",
            hex_type: Offboard {
                yellow_revenue: 30,
                brown_revenue: 40,
            },
            terrain_cost: 0,
        },
        HexDef {
            coord: "B24",
            hex_type: Offboard {
                yellow_revenue: 20,
                brown_revenue: 30,
            },
            terrain_cost: 0,
        },
        // --- Gray hexes (preprinted, not upgradable) ---
        HexDef {
            coord: "D2",
            hex_type: City {
                revenue: 20,
                slots: 1,
            },
            terrain_cost: 0,
        },
        HexDef {
            coord: "F6",
            hex_type: City {
                revenue: 30,
                slots: 1,
            },
            terrain_cost: 0,
        },
        HexDef {
            coord: "E9",
            hex_type: Path,
            terrain_cost: 0,
        },
        HexDef {
            coord: "H12",
            hex_type: City {
                revenue: 10,
                slots: 1,
            },
            terrain_cost: 0,
        },
        HexDef {
            coord: "D14",
            hex_type: City {
                revenue: 20,
                slots: 1,
            },
            terrain_cost: 0,
        },
        HexDef {
            coord: "C15",
            hex_type: Town { revenue: 10 },
            terrain_cost: 0,
        },
        HexDef {
            coord: "K15",
            hex_type: City {
                revenue: 20,
                slots: 1,
            },
            terrain_cost: 0,
        },
        HexDef {
            coord: "A17",
            hex_type: Path,
            terrain_cost: 0,
        },
        HexDef {
            coord: "A19",
            hex_type: City {
                revenue: 40,
                slots: 1,
            },
            terrain_cost: 0,
        },
        HexDef {
            coord: "I19",
            hex_type: Town { revenue: 10 },
            terrain_cost: 0,
        },
        HexDef {
            coord: "F24",
            hex_type: Town { revenue: 10 },
            terrain_cost: 0,
        },
        HexDef {
            coord: "D24",
            hex_type: Path,
            terrain_cost: 0,
        },
        // --- Yellow hexes (preprinted upgradable) ---
        HexDef {
            coord: "E5",
            hex_type: DoubleCity { revenue: 0 },
            terrain_cost: 80,
        },
        HexDef {
            coord: "D10",
            hex_type: DoubleCity { revenue: 0 },
            terrain_cost: 80,
        },
        HexDef {
            coord: "E11",
            hex_type: DoubleCity { revenue: 0 },
            terrain_cost: 0,
        },
        HexDef {
            coord: "H18",
            hex_type: DoubleCity { revenue: 0 },
            terrain_cost: 0,
        },
        HexDef {
            coord: "I15",
            hex_type: City {
                revenue: 30,
                slots: 1,
            },
            terrain_cost: 0,
        },
        HexDef {
            coord: "G19",
            hex_type: DoubleCity { revenue: 40 },
            terrain_cost: 80,
        },
        HexDef {
            coord: "E23",
            hex_type: City {
                revenue: 30,
                slots: 1,
            },
            terrain_cost: 0,
        },
        // --- White hexes: cities ---
        HexDef {
            coord: "F4",
            hex_type: City {
                revenue: 0,
                slots: 1,
            },
            terrain_cost: 80,
        },
        HexDef {
            coord: "J14",
            hex_type: City {
                revenue: 0,
                slots: 1,
            },
            terrain_cost: 80,
        },
        HexDef {
            coord: "F22",
            hex_type: City {
                revenue: 0,
                slots: 1,
            },
            terrain_cost: 80,
        },
        HexDef {
            coord: "B16",
            hex_type: City {
                revenue: 0,
                slots: 1,
            },
            terrain_cost: 0,
        },
        HexDef {
            coord: "E19",
            hex_type: City {
                revenue: 0,
                slots: 1,
            },
            terrain_cost: 0,
        },
        HexDef {
            coord: "H4",
            hex_type: City {
                revenue: 0,
                slots: 1,
            },
            terrain_cost: 0,
        },
        HexDef {
            coord: "B10",
            hex_type: City {
                revenue: 0,
                slots: 1,
            },
            terrain_cost: 0,
        },
        HexDef {
            coord: "H10",
            hex_type: City {
                revenue: 0,
                slots: 1,
            },
            terrain_cost: 0,
        },
        HexDef {
            coord: "H16",
            hex_type: City {
                revenue: 0,
                slots: 1,
            },
            terrain_cost: 0,
        },
        HexDef {
            coord: "F16",
            hex_type: City {
                revenue: 0,
                slots: 1,
            },
            terrain_cost: 120,
        },
        // --- White hexes: towns ---
        HexDef {
            coord: "E7",
            hex_type: Town { revenue: 0 },
            terrain_cost: 0,
        },
        HexDef {
            coord: "B20",
            hex_type: Town { revenue: 0 },
            terrain_cost: 0,
        },
        HexDef {
            coord: "D4",
            hex_type: Town { revenue: 0 },
            terrain_cost: 0,
        },
        HexDef {
            coord: "F10",
            hex_type: Town { revenue: 0 },
            terrain_cost: 0,
        },
        // Double towns
        HexDef {
            coord: "G7",
            hex_type: DoubleTown,
            terrain_cost: 0,
        },
        HexDef {
            coord: "G17",
            hex_type: DoubleTown,
            terrain_cost: 0,
        },
        HexDef {
            coord: "F20",
            hex_type: DoubleTown,
            terrain_cost: 0,
        },
        // --- White hexes: blank ---
        HexDef {
            coord: "I13",
            hex_type: Blank,
            terrain_cost: 0,
        },
        HexDef {
            coord: "D18",
            hex_type: Blank,
            terrain_cost: 0,
        },
        HexDef {
            coord: "B12",
            hex_type: Blank,
            terrain_cost: 0,
        },
        HexDef {
            coord: "B14",
            hex_type: Blank,
            terrain_cost: 0,
        },
        HexDef {
            coord: "B22",
            hex_type: Blank,
            terrain_cost: 0,
        },
        HexDef {
            coord: "C7",
            hex_type: Blank,
            terrain_cost: 0,
        },
        HexDef {
            coord: "C9",
            hex_type: Blank,
            terrain_cost: 0,
        },
        HexDef {
            coord: "C23",
            hex_type: Blank,
            terrain_cost: 0,
        },
        HexDef {
            coord: "D8",
            hex_type: Blank,
            terrain_cost: 0,
        },
        HexDef {
            coord: "D16",
            hex_type: Blank,
            terrain_cost: 0,
        },
        HexDef {
            coord: "D20",
            hex_type: Blank,
            terrain_cost: 0,
        },
        HexDef {
            coord: "E3",
            hex_type: Blank,
            terrain_cost: 0,
        },
        HexDef {
            coord: "E13",
            hex_type: Blank,
            terrain_cost: 0,
        },
        HexDef {
            coord: "E15",
            hex_type: Blank,
            terrain_cost: 0,
        },
        HexDef {
            coord: "F12",
            hex_type: Blank,
            terrain_cost: 0,
        },
        HexDef {
            coord: "F14",
            hex_type: Blank,
            terrain_cost: 0,
        },
        HexDef {
            coord: "F18",
            hex_type: Blank,
            terrain_cost: 0,
        },
        HexDef {
            coord: "G3",
            hex_type: Blank,
            terrain_cost: 0,
        },
        HexDef {
            coord: "G5",
            hex_type: Blank,
            terrain_cost: 0,
        },
        HexDef {
            coord: "G9",
            hex_type: Blank,
            terrain_cost: 0,
        },
        HexDef {
            coord: "G11",
            hex_type: Blank,
            terrain_cost: 0,
        },
        HexDef {
            coord: "H2",
            hex_type: Blank,
            terrain_cost: 0,
        },
        HexDef {
            coord: "H6",
            hex_type: Blank,
            terrain_cost: 0,
        },
        HexDef {
            coord: "H8",
            hex_type: Blank,
            terrain_cost: 0,
        },
        HexDef {
            coord: "H14",
            hex_type: Blank,
            terrain_cost: 0,
        },
        HexDef {
            coord: "I3",
            hex_type: Blank,
            terrain_cost: 0,
        },
        HexDef {
            coord: "I5",
            hex_type: Blank,
            terrain_cost: 0,
        },
        HexDef {
            coord: "I7",
            hex_type: Blank,
            terrain_cost: 0,
        },
        HexDef {
            coord: "I9",
            hex_type: Blank,
            terrain_cost: 0,
        },
        HexDef {
            coord: "J4",
            hex_type: Blank,
            terrain_cost: 0,
        },
        HexDef {
            coord: "J6",
            hex_type: Blank,
            terrain_cost: 0,
        },
        HexDef {
            coord: "J8",
            hex_type: Blank,
            terrain_cost: 0,
        },
        // --- Mountains (cost=120) ---
        HexDef {
            coord: "G15",
            hex_type: Blank,
            terrain_cost: 120,
        },
        HexDef {
            coord: "C21",
            hex_type: Blank,
            terrain_cost: 120,
        },
        HexDef {
            coord: "D22",
            hex_type: Blank,
            terrain_cost: 120,
        },
        HexDef {
            coord: "E17",
            hex_type: Blank,
            terrain_cost: 120,
        },
        HexDef {
            coord: "E21",
            hex_type: Blank,
            terrain_cost: 120,
        },
        HexDef {
            coord: "G13",
            hex_type: Blank,
            terrain_cost: 120,
        },
        HexDef {
            coord: "I11",
            hex_type: Blank,
            terrain_cost: 120,
        },
        HexDef {
            coord: "J10",
            hex_type: Blank,
            terrain_cost: 120,
        },
        HexDef {
            coord: "J12",
            hex_type: Blank,
            terrain_cost: 120,
        },
        HexDef {
            coord: "C17",
            hex_type: Blank,
            terrain_cost: 120,
        },
        // --- Water (cost=80) ---
        HexDef {
            coord: "D6",
            hex_type: Blank,
            terrain_cost: 80,
        },
        HexDef {
            coord: "I17",
            hex_type: Blank,
            terrain_cost: 80,
        },
        HexDef {
            coord: "B18",
            hex_type: Blank,
            terrain_cost: 80,
        },
        HexDef {
            coord: "C19",
            hex_type: Blank,
            terrain_cost: 80,
        },
        // --- Special border hexes (blank with impassable borders) ---
        HexDef {
            coord: "F8",
            hex_type: Blank,
            terrain_cost: 0,
        },
        HexDef {
            coord: "C11",
            hex_type: Blank,
            terrain_cost: 0,
        },
        HexDef {
            coord: "C13",
            hex_type: Blank,
            terrain_cost: 0,
        },
        HexDef {
            coord: "D12",
            hex_type: Blank,
            terrain_cost: 0,
        },
    ]
}

// ---------------------------------------------------------------------------
// Frozen action-layout orders (AlphaZero bridge)
// ---------------------------------------------------------------------------

/// Hex order of the flat action layout. A FROZEN artifact of the Python
/// ActionMapper's original construction (mostly map order, with the
/// multi-city/home hexes grouped at the end) — trained 1830 checkpoints
/// depend on these exact slots, so this list must never change. Pinned by
/// the frozen-layout cargo test.
pub fn action_hex_order() -> Vec<&'static str> {
    vec![
        "F2", "I1", "J2", "A9", "A11", "K13", "B24", "D2", "F6", "E9", "H12", "D14", "C15", "K15",
        "A17", "A19", "I19", "F24", "D24", "F4", "J14", "F22", "E7", "F8", "C11", "C13", "D12",
        "B16", "C17", "B20", "D4", "F10", "I13", "D18", "B12", "B14", "B22", "C7", "C9", "C23",
        "D8", "D16", "D20", "E3", "E13", "E15", "F12", "F14", "F18", "G3", "G5", "G9", "G11",
        "H2", "H6", "H8", "H14", "I3", "I5", "I7", "I9", "J4", "J6", "J8", "G15", "C21", "D22",
        "E17", "E21", "G13", "I11", "J10", "J12", "E19", "H4", "B10", "H10", "H16", "F16", "G7",
        "G17", "F20", "D6", "I17", "B18", "C19", "E5", "D10", "E11", "H18", "I15", "G19", "E23",
    ]
}

/// Tile order of the flat action layout. Like [`action_hex_order`], a FROZEN
/// Python-ActionMapper artifact (set-iteration order) that checkpoints
/// depend on. Same tile SET as `tile_counts()`, different order.
pub fn action_tile_order() -> Vec<&'static str> {
    vec![
        "42", "4", "16", "70", "23", "7", "18", "24", "3", "55", "61", "54", "9", "41", "26",
        "68", "57", "45", "1", "56", "44", "62", "63", "64", "40", "66", "20", "27", "39", "19",
        "59", "25", "46", "28", "65", "43", "2", "53", "58", "14", "47", "8", "29", "69", "15",
        "67",
    ]
}

// ---------------------------------------------------------------------------
// Location names
// ---------------------------------------------------------------------------

pub fn location_names() -> HashMap<&'static str, &'static str> {
    [
        ("D2", "Lansing"),
        ("F2", "Chicago"),
        ("J2", "Gulf"),
        ("F4", "Toledo"),
        ("J14", "Washington"),
        ("F22", "Providence"),
        ("E5", "Detroit & Windsor"),
        ("D10", "Hamilton & Toronto"),
        ("F6", "Cleveland"),
        ("E7", "London"),
        ("A11", "Canadian West"),
        ("K13", "Deep South"),
        ("E11", "Dunkirk & Buffalo"),
        ("H12", "Altoona"),
        ("D14", "Rochester"),
        ("C15", "Kingston"),
        ("I15", "Baltimore"),
        ("K15", "Richmond"),
        ("B16", "Ottawa"),
        ("F16", "Scranton"),
        ("H18", "Philadelphia & Trenton"),
        ("A19", "Montreal"),
        ("E19", "Albany"),
        ("G19", "New York & Newark"),
        ("I19", "Atlantic City"),
        ("F24", "Mansfield"),
        ("B20", "Burlington"),
        ("E23", "Boston"),
        ("B24", "Maritime Provinces"),
        ("D4", "Flint"),
        ("F10", "Erie"),
        ("G7", "Akron & Canton"),
        ("G17", "Reading & Allentown"),
        ("F20", "New Haven & Hartford"),
        ("H4", "Columbus"),
        ("B10", "Barrie"),
        ("H10", "Pittsburgh"),
        ("H16", "Lancaster"),
    ]
    .iter()
    .copied()
    .collect()
}

// ---------------------------------------------------------------------------
// Preprinted hex DSL strings
// ---------------------------------------------------------------------------

/// Returns the DSL string for preprinted hexes that have path connectivity.
/// Red (offboard) and gray hexes have fixed paths; yellow preprinted hexes also
/// have initial paths/labels that matter for graph connectivity and upgrades.
pub fn preprinted_hex_dsl(coord: &str) -> Option<(&'static str, &'static str)> {
    // Returns (dsl_string, color_str)
    match coord {
        // Red offboard hexes
        "F2" => Some((
            "offboard=revenue:yellow_40|brown_70;path=a:3,b:_0;path=a:4,b:_0;path=a:5,b:_0",
            "red",
        )),
        "I1" => Some(("offboard=revenue:yellow_30|brown_60;path=a:4,b:_0", "red")),
        "J2" => Some((
            "offboard=revenue:yellow_30|brown_60;path=a:3,b:_0;path=a:4,b:_0",
            "red",
        )),
        "A9" => Some(("offboard=revenue:yellow_30|brown_50;path=a:5,b:_0", "red")),
        "A11" => Some((
            "offboard=revenue:yellow_30|brown_50;path=a:5,b:_0;path=a:0,b:_0",
            "red",
        )),
        "K13" => Some((
            "offboard=revenue:yellow_30|brown_40;path=a:2,b:_0;path=a:3,b:_0",
            "red",
        )),
        "B24" => Some((
            "offboard=revenue:yellow_20|brown_30;path=a:1,b:_0;path=a:0,b:_0",
            "red",
        )),
        // Gray hexes with paths
        "D2" => Some(("city=revenue:20;path=a:5,b:_0;path=a:4,b:_0", "gray")),
        "F6" => Some(("city=revenue:30;path=a:5,b:_0;path=a:0,b:_0", "gray")),
        "E9" => Some(("path=a:2,b:3", "gray")),
        "H12" => Some((
            "city=revenue:10;path=a:1,b:_0;path=a:4,b:_0;path=a:1,b:4",
            "gray",
        )),
        "D14" => Some((
            "city=revenue:20;path=a:1,b:_0;path=a:4,b:_0;path=a:0,b:_0",
            "gray",
        )),
        "C15" => Some(("town=revenue:10;path=a:1,b:_0;path=a:3,b:_0", "gray")),
        "K15" => Some(("city=revenue:20;path=a:2,b:_0", "gray")),
        "A17" => Some(("path=a:0,b:5", "gray")),
        "A19" => Some(("city=revenue:40;path=a:5,b:_0;path=a:0,b:_0", "gray")),
        "I19" | "F24" => Some(("town=revenue:10;path=a:1,b:_0;path=a:2,b:_0", "gray")),
        "D24" => Some(("path=a:1,b:0", "gray")),
        // Yellow preprinted hexes with paths/labels
        "I15" => Some((
            "city=revenue:30;path=a:4,b:_0;path=a:0,b:_0;label=B",
            "yellow",
        )),
        "G19" => Some((
            "city=revenue:40;city=revenue:40;path=a:3,b:_1;path=a:0,b:_0;label=NY",
            "yellow",
        )),
        "E23" => Some((
            "city=revenue:30;path=a:3,b:_0;path=a:5,b:_0;label=B",
            "yellow",
        )),
        // Yellow OO hexes (double cities, no initial paths but have labels)
        "E5" | "D10" => Some(("city=revenue:0;city=revenue:0;label=OO", "yellow")),
        "E11" | "H18" => Some(("city=revenue:0;city=revenue:0;label=OO", "yellow")),
        _ => None,
    }
}
