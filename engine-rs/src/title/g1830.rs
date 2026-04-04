//! Static game data for the 1830: Railways & Robber Barons title.
//!
//! All data is derived from the Python engine's g1830.py.
//! These are plain Rust structs used as templates during game construction —
//! they are NOT pyclass types.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Definition structs
// ---------------------------------------------------------------------------

pub struct CorporationDef {
    pub sym: &'static str,
    pub name: &'static str,
    pub token_prices: &'static [i32],
    pub home_hex: &'static str,
    pub home_city_index: u8,
    /// Whether the home hex tile is reserved for this corp (token placed after tile upgrade).
    pub reserved: bool,
}

pub struct CompanyDef {
    pub sym: &'static str,
    pub name: &'static str,
    pub value: i32,
    pub revenue: i32,
    pub blocked_hexes: &'static [&'static str],
    /// If set, buying this company grants a free share of this corporation.
    pub grants_share: Option<(&'static str, u8)>, // (corp_sym, percent)
    /// If set, buying this company triggers a pending par for this corporation.
    pub triggers_par: Option<&'static str>, // corp_sym
}

pub struct TrainDef {
    pub name: &'static str,
    pub distance: u32,
    pub price: i32,
    pub count: u32,
    pub rusts_on: Option<&'static str>,
}

pub struct PhaseDef {
    pub name: &'static str,
    pub train_limit: u8,
    pub tiles: &'static [&'static str],
    pub operating_rounds: u8,
}

pub struct MarketCell {
    pub price: i32,
    pub zone: MarketZone,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MarketZone {
    Normal,
    Par,
    Yellow,
    Orange,
    Brown,
}

pub struct HexDef {
    pub coord: &'static str,
    pub hex_type: HexType,
    pub terrain_cost: i32,
}

pub enum HexType {
    Blank,
    City {
        revenue: i32,
        slots: u8,
    },
    Town {
        revenue: i32,
    },
    DoubleCity {
        revenue: i32,
    },
    DoubleTown,
    Offboard {
        yellow_revenue: i32,
        brown_revenue: i32,
    },
    Path,
}

// ---------------------------------------------------------------------------
// Corporation data (8 corporations)
// ---------------------------------------------------------------------------

pub fn corporations() -> Vec<CorporationDef> {
    vec![
        CorporationDef {
            sym: "PRR",
            name: "Pennsylvania Railroad",
            token_prices: &[0, 40, 100, 100],
            home_hex: "H12",
            home_city_index: 0,
            reserved: false,
        },
        CorporationDef {
            sym: "NYC",
            name: "New York Central Railroad",
            token_prices: &[0, 40, 100, 100],
            home_hex: "E19",
            home_city_index: 0,
            reserved: false,
        },
        CorporationDef {
            sym: "CPR",
            name: "Canadian Pacific Railroad",
            token_prices: &[0, 40, 100, 100],
            home_hex: "A19",
            home_city_index: 0,
            reserved: false,
        },
        CorporationDef {
            sym: "B&O",
            name: "Baltimore & Ohio Railroad",
            token_prices: &[0, 40, 100],
            home_hex: "I15",
            home_city_index: 0,
            reserved: false,
        },
        CorporationDef {
            sym: "C&O",
            name: "Chesapeake & Ohio Railroad",
            token_prices: &[0, 40, 100],
            home_hex: "F6",
            home_city_index: 0,
            reserved: false,
        },
        CorporationDef {
            sym: "ERIE",
            name: "Erie Railroad",
            token_prices: &[0, 40, 100],
            home_hex: "E11",
            home_city_index: 0,
            reserved: true,
        },
        CorporationDef {
            sym: "NYNH",
            name: "New York, New Haven & Hartford Railroad",
            token_prices: &[0, 40],
            home_hex: "G19",
            home_city_index: 1,
            reserved: false,
        },
        CorporationDef {
            sym: "B&M",
            name: "Boston & Maine Railroad",
            token_prices: &[0, 40],
            home_hex: "E23",
            home_city_index: 0,
            reserved: false,
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
            blocked_hexes: &["G15"],
            grants_share: None,
            triggers_par: None,
        },
        CompanyDef {
            sym: "CS",
            name: "Champlain & St.Lawrence",
            value: 40,
            revenue: 10,
            blocked_hexes: &["B20"],
            grants_share: None,
            triggers_par: None,
        },
        CompanyDef {
            sym: "DH",
            name: "Delaware & Hudson",
            value: 70,
            revenue: 15,
            blocked_hexes: &["F16"],
            grants_share: None,
            triggers_par: None,
        },
        CompanyDef {
            sym: "MH",
            name: "Mohawk & Hudson",
            value: 110,
            revenue: 20,
            blocked_hexes: &["D18"],
            grants_share: None,
            triggers_par: None,
        },
        CompanyDef {
            sym: "CA",
            name: "Camden & Amboy",
            value: 160,
            revenue: 25,
            blocked_hexes: &["H18"],
            grants_share: Some(("PRR", 10)),
            triggers_par: None,
        },
        CompanyDef {
            sym: "BO",
            name: "Baltimore & Ohio",
            value: 220,
            revenue: 30,
            blocked_hexes: &["I13", "I15"],
            grants_share: None,
            triggers_par: Some("B&O"),
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
            distance: 2,
            price: 80,
            count: 6,
            rusts_on: Some("4"),
        },
        TrainDef {
            name: "3",
            distance: 3,
            price: 180,
            count: 5,
            rusts_on: Some("6"),
        },
        TrainDef {
            name: "4",
            distance: 4,
            price: 300,
            count: 4,
            rusts_on: Some("D"),
        },
        TrainDef {
            name: "5",
            distance: 5,
            price: 450,
            count: 3,
            rusts_on: None,
        },
        TrainDef {
            name: "6",
            distance: 6,
            price: 630,
            count: 2,
            rusts_on: None,
        },
        TrainDef {
            name: "D",
            distance: 999,
            price: 1100,
            count: 20,
            rusts_on: None,
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
            train_limit: 4,
            tiles: &["yellow"],
            operating_rounds: 1,
        },
        PhaseDef {
            name: "3",
            train_limit: 4,
            tiles: &["yellow", "green"],
            operating_rounds: 2,
        },
        PhaseDef {
            name: "4",
            train_limit: 3,
            tiles: &["yellow", "green"],
            operating_rounds: 2,
        },
        PhaseDef {
            name: "5",
            train_limit: 2,
            tiles: &["yellow", "green", "brown"],
            operating_rounds: 3,
        },
        PhaseDef {
            name: "6",
            train_limit: 2,
            tiles: &["yellow", "green", "brown"],
            operating_rounds: 3,
        },
        PhaseDef {
            name: "D",
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

/// Parses a hex coordinate like "H12" into (row_letter_index, number).
/// Letter part → x: A=0..K=10.  Number part → y (raw number).
pub fn parse_coord(coord: &str) -> (i32, i32) {
    let bytes = coord.as_bytes();
    let letter = (bytes[0] - b'A') as i32;
    let number: i32 = coord[1..].parse().expect("invalid hex coordinate number");
    (letter, number)
}

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
// Hex adjacency computation
// ---------------------------------------------------------------------------

/// Pointy-top hex direction deltas in (d_letter, d_number) space.
///
/// 1830 uses axes: x=number, y=letter. The Python engine's pointy-top
/// direction deltas are in (dx, dy) = (d_number, d_letter) space:
///   0:(-1,1), 1:(-2,0), 2:(-1,-1), 3:(1,-1), 4:(2,0), 5:(1,1)
///
/// We store coordinates as (letter_index, number), so we swap to (dy, dx):
const HEX_DELTAS: [(i32, i32); 6] = [
    (1, -1),  // 0: upper-right  (Python dx=-1, dy=+1)
    (0, -2),  // 1: right        (Python dx=-2, dy= 0)
    (-1, -1), // 2: lower-right  (Python dx=-1, dy=-1)
    (-1, 1),  // 3: lower-left   (Python dx=+1, dy=-1)
    (0, 2),   // 4: left         (Python dx=+2, dy= 0)
    (1, 1),   // 5: upper-left   (Python dx=+1, dy=+1)
];

/// Format (letter_index, number) back to a coordinate string like "H12".
fn format_coord(letter: i32, number: i32) -> String {
    let ch = (b'A' + letter as u8) as char;
    format!("{}{}", ch, number)
}

/// Compute hex adjacency from a set of hex coordinates.
/// Returns hex_id -> { direction -> neighbor_hex_id } for all valid neighbors.
pub fn compute_adjacency(coords: &[&str]) -> HashMap<String, HashMap<u8, String>> {
    let coord_set: std::collections::HashSet<String> =
        coords.iter().map(|c| c.to_string()).collect();

    let mut adjacency: HashMap<String, HashMap<u8, String>> = HashMap::new();

    for &coord in coords {
        let (letter, number) = parse_coord(coord);
        let mut neighbors = HashMap::new();

        for (dir, (dl, dn)) in HEX_DELTAS.iter().enumerate() {
            let nl = letter + dl;
            let nn = number + dn;
            if nl >= 0 {
                let neighbor = format_coord(nl, nn);
                if coord_set.contains(&neighbor) {
                    neighbors.insert(dir as u8, neighbor);
                }
            }
        }

        adjacency.insert(coord.to_string(), neighbors);
    }

    adjacency
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
