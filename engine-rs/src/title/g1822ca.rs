//! **1822CA** (The Railways of Canada), Ruby `G1822CA::Game < G1822::Game`
//! (tobymao/18xx `lib/engine/game/g_1822_ca/`).
//!
//! The data (companies, corporations, trains, phases, market, tiles, map)
//! is GENERATED from the Ruby constants into `g1822ca_data.rs`
//! (scripts/g1822_title_data.{rb,py}); this module holds the `GameTitle`
//! registration and the family table — CA's `self.class::` constants, with
//! everything CA does not override copied from 1822's (Ruby inheritance; a
//! constant CA sets to `nil` is ""). The rules CA overrides (the big
//! cities, grain trains, the sawmill, the Detroit-Duluth blocker, QMOO's
//! Quebec home, ICR's destination, the acquisition track bonus, small mail
//! contracts, ...) branch on `Variant::Ca` in `crate::g1822`.

use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, OnceLock};

use super::g1822ca_data as data;
use super::{
    BorderDef, CompanyDef, CorpType, CorporationDef, GameTitle, HexDef, HexType, MarketCell, MarketMovement,
    MarketZone, PhaseDef, TrainDef,
};
use crate::g1822::family::{Acquisition, Family, Variant};
use crate::g1822::track::{Ab, AbKind};
use crate::steps::{FinishedRound, RoundTransition, StepDesc};
use crate::tiles::{parse_tile, TileDef};

pub use data::TITLE;

/// The Detroit-Duluth blocker (`block_detroit_duluth`): a pseudo
/// corporation whose two tokens fill Q19 and Y27 until the first 3-train.
pub const DD: &str = "DD";

pub struct G1822Ca;

impl GameTitle for G1822Ca {
    fn name(&self) -> &'static str {
        TITLE
    }
    // Rounds are driven by crate::g1822 (see title::g1822).
    fn auction_steps(&self) -> &'static [StepDesc] {
        &[]
    }
    fn stock_steps(&self) -> &'static [StepDesc] {
        &[]
    }
    fn operating_steps(&self) -> &'static [StepDesc] {
        &[]
    }
    fn next_round(&self, finished: FinishedRound, phase_name: &str, phase_operating_rounds: u8) -> RoundTransition {
        super::g1822::G1822.next_round(finished, phase_name, phase_operating_rounds)
    }
    fn minors_operate_first(&self) -> bool {
        true
    }
    fn stock_sell_after_buy(&self) -> bool {
        false
    }
    fn starting_cash(&self, num_players: u8) -> i32 {
        data::starting_cash(num_players)
    }
    fn cert_limit(&self, num_players: u8) -> u8 {
        data::cert_limit(num_players)
    }
    fn bank_cash(&self) -> i32 {
        data::BANK_CASH
    }
    fn companies(&self) -> Vec<CompanyDef> {
        data::companies()
    }
    /// CORPORATIONS + the Detroit-Duluth blocker (`block_detroit_duluth`
    /// creates it at setup; it never floats and holds two tokens).
    fn corporations(&self) -> Vec<CorporationDef> {
        let mut v = data::corporations();
        v.push(CorporationDef {
            sym: DD,
            name: "Detroit-Duluth blocker",
            token_prices: &[0, 0],
            home_hex: "",
            home_city_index: u8::MAX,
            reserved: false,
            shares: &[10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
            float_percent: 60,
            capitalization: super::Capitalization::Incremental,
            corp_type: CorpType::Major,
            max_ownership_percent: 60,
            always_market_price: false,
        });
        v
    }
    fn trains(&self) -> Vec<TrainDef> {
        data::trains()
    }
    fn phases(&self) -> Vec<PhaseDef> {
        data::phases()
    }
    fn optional_rules(&self) -> &'static [&'static str] {
        data::OPTIONAL_RULES
    }
    fn hex_definitions(&self) -> Vec<HexDef> {
        hex_definitions()
    }
    /// map.rb `future_label=label:L,color:C`.
    fn hex_future_label(&self, coord: &str) -> Option<(&'static str, &'static str)> {
        future_label(coord)
    }
    fn preprinted_hex_dsl(&self, coord: &str) -> Option<(&'static str, &'static str)> {
        let (dsl, color) = hex_dsl(coord)?;
        if dsl.is_empty() {
            return None;
        }
        Some((dsl, color))
    }
    fn tile_counts(&self) -> Vec<(&'static str, u32)> {
        data::TILES.iter().map(|(n, c, _, _, _)| (*n, *c)).collect()
    }
    fn tile_cities(&self, _tile_id: &str) -> Option<Vec<u8>> {
        None
    }
    fn tile_catalog(&self) -> Arc<HashMap<String, TileDef>> {
        tile_catalog()
    }
    fn market_grid(&self) -> Vec<Vec<Option<MarketCell>>> {
        data::MARKET
            .iter()
            .map(|row| row.iter().map(|c| market_cell(c)).collect())
            .collect()
    }
    /// G1822::StockMarket (CA's market is one row).
    fn market_movement(&self) -> MarketMovement {
        MarketMovement::TwoDimensionalUpDownRight
    }
    /// map.rb LAYOUT = :flat.
    fn hex_layout(&self) -> super::HexLayout {
        super::HexLayout::Flat
    }
    fn train_buy_from_other_players(&self) -> bool {
        true
    }
    fn generator_cross_player(&self) -> super::CrossPlayerPolicy {
        super::CrossPlayerPolicy::RESTRICTED
    }
    fn must_buy_train_always(&self) -> bool {
        true
    }
    fn alphazero_bridge_ready(&self) -> bool {
        false
    }
    fn is_g1822_family(&self) -> bool {
        true
    }
    fn g1822_family(&self) -> Option<&'static Family> {
        Some(&FAMILY)
    }
}

/// Parse one Ruby market code (p = par, x = par_1, y = yellow, e = endgame).
fn market_cell(code: &str) -> Option<MarketCell> {
    if code.is_empty() {
        return None;
    }
    let digits: String = code.chars().take_while(|c| c.is_ascii_digit()).collect();
    let price: i32 = digits.parse().expect("market price");
    let mut zones = Vec::new();
    for ch in code[digits.len()..].chars() {
        zones.push(match ch {
            'p' => MarketZone::Par,
            'x' => MarketZone::Par1,
            'y' => MarketZone::Yellow,
            'e' => MarketZone::EndGame,
            other => panic!("1822CA market: unknown cell type {other}"),
        });
    }
    if zones.is_empty() {
        zones.push(MarketZone::Normal);
    }
    Some(MarketCell { price, zones })
}

pub fn tile_catalog() -> Arc<HashMap<String, TileDef>> {
    static CATALOG: OnceLock<Arc<HashMap<String, TileDef>>> = OnceLock::new();
    CATALOG
        .get_or_init(|| {
            Arc::new(
                data::TILES
                    .iter()
                    .map(|(id, _, color, dsl, _)| {
                        (id.to_string(), parse_tile(id, dsl, super::g1822::color_of(color)))
                    })
                    .collect(),
            )
        })
        .clone()
}

/// The tile DSL for a catalog id: (DSL, color).
pub fn tile_dsl(id: &str) -> Option<(&'static str, &'static str)> {
    data::TILES.iter().find(|(t, _, _, _, _)| *t == id).map(|(_, _, c, d, _)| (*d, *c))
}

pub fn hex_dsl(coord: &str) -> Option<(&'static str, &'static str)> {
    data::HEX_DSL.iter().find(|(c, _, _)| *c == coord).map(|(_, d, col)| (*d, *col))
}

/// A hex's `future_label=label:L,color:C` (label, colour).
pub fn future_label(coord: &str) -> Option<(&'static str, &'static str)> {
    let (dsl, _) = hex_dsl(coord)?;
    let part = dsl.split(';').find_map(|p| p.trim().strip_prefix("future_label="))?;
    let mut label = None;
    let mut color = None;
    for kv in part.split(',') {
        match kv.split_once(':') {
            Some(("label", v)) => label = Some(v),
            Some(("color", v)) => color = Some(v),
            _ => {}
        }
    }
    Some((label?, color?))
}

fn parsed_borders() -> &'static HashMap<&'static str, Vec<BorderDef>> {
    static BORDERS: OnceLock<HashMap<&'static str, Vec<BorderDef>>> = OnceLock::new();
    BORDERS.get_or_init(|| {
        let mut out = HashMap::new();
        for (coord, dsl, _) in data::HEX_DSL {
            let mut v = Vec::new();
            for part in dsl.split(';') {
                let Some(attrs) = part.trim().strip_prefix("border=") else { continue };
                let mut edge = 0u8;
                let mut cost = None;
                let mut kind = "";
                for kv in attrs.split(',') {
                    let (k, val) = kv.split_once(':').unwrap_or((kv, ""));
                    match k {
                        "edge" => edge = val.parse().unwrap(),
                        "cost" => cost = Some(val.parse().unwrap()),
                        "type" => kind = val,
                        _ => {}
                    }
                }
                v.push(BorderDef { edge, cost, impassable: kind == "impassable" });
            }
            out.insert(*coord, v);
        }
        out
    })
}

pub fn hex_definitions() -> Vec<HexDef> {
    let borders = parsed_borders();
    data::HEX_DSL
        .iter()
        .map(|(coord, dsl, color)| {
            let hex_type = if *color == "gray" && dsl.starts_with("offboard") {
                HexType::Offboard { yellow_revenue: 0, brown_revenue: 0 }
            } else if dsl.contains("city=") {
                HexType::City { revenue: 0, slots: 1 }
            } else if dsl.contains("town=") {
                HexType::Town { revenue: 0 }
            } else if dsl.contains("path=") {
                HexType::Path
            } else {
                HexType::Blank
            };
            HexDef {
                coord,
                hex_type,
                terrain_cost: 0,
                borders: borders.get(coord).map(|v| -> &'static [BorderDef] { v.as_slice() }).unwrap_or(&[]),
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// The family table
// ---------------------------------------------------------------------------

/// PRIVATE_COMPANIES_ACQUISITION (game.rb).
pub fn private_acquisition(sym: &str) -> Acquisition {
    use CorpType::{Major as J, Minor as N};
    const MAJ: &[CorpType] = &[J];
    const BOTH: &[CorpType] = &[J, N];
    const NONE: &[CorpType] = &[];
    Some(match sym {
        "P1" => (MAJ, 5),
        "P2" => (BOTH, 1),
        "P3" => (MAJ, 2),
        "P4" => (MAJ, 2),
        "P5" => (MAJ, 5),
        "P6" => (MAJ, 5),
        "P7" => (MAJ, 3),
        "P8" => (BOTH, 2),
        "P9" => (BOTH, 2),
        "P10" => (MAJ, 3),
        "P11" => (NONE, 8),
        "P12" => (BOTH, 1),
        "P13" => (MAJ, 3),
        "P14" => (BOTH, 1),
        "P15" => (BOTH, 1),
        "P16" => (BOTH, 1),
        "P17" => (BOTH, 1),
        "P18" => (BOTH, 1),
        "P19" => (MAJ, 3),
        "P20" => (MAJ, 3),
        "P21" => (MAJ, 3),
        "P22" => (MAJ, 3),
        "P23" => (MAJ, 3),
        "P24" => (MAJ, 3),
        "P25" => (MAJ, 3),
        "P26" => (MAJ, 3),
        "P27" => (MAJ, 3),
        "P28" => (BOTH, 3),
        "P29" => (BOTH, 1),
        "P30" => (BOTH, 1),
        _ => return None,
    })
}

fn strs(v: &[&str]) -> Vec<String> {
    v.iter().map(|s| s.to_string()).collect()
}

fn lay(count: Option<i32>) -> Ab {
    Ab {
        kind: AbKind::TileLay,
        count,
        used: false,
        hexes: vec![],
        tiles: vec![],
        free: false,
        special: true,
        reachable: false,
        closed_when_used_up: false,
        must_lay_together: false,
        consume_tile_lay: false,
        lay_count: None,
        upgrade_count: None,
        combo: vec![],
        discount: 0,
        terrain: None,
        track_only: false,
        count_per_or: None,
        count_this_or: 0,
    }
}

/// COMPANIES_BIG_CITY_UPGRADES and their cities (BIG_CITY_HEXES_TO_COMPANIES).
pub const BIG_CITY_COMPANIES: &[(&str, &str)] =
    &[("P14", "AC21"), ("P15", "AE15"), ("P16", "AF12"), ("P17", "AH8"), ("P18", "N16")];

/// P21's 3-tile grant hexes (entities.rb).
const P21_HEXES: &[&str] = &[
    "A7", "A9", "B6", "B8", "B10", "B12", "B14", "C7", "C9", "C11", "C13", "D6", "D8", "D10", "D12", "D14", "D16", "E7",
    "E9", "E15", "F6", "F8", "F10", "F12", "G7", "G9", "G11", "G13", "G17", "G15", "H6", "H8", "H10", "H12", "H14", "H16",
    "I7", "I9", "I11", "I13", "I15", "I17", "J6", "J8", "J10", "J12", "J14", "J16", "K7", "K9", "K11", "K13", "K15",
    "K17", "L6", "L8", "L10", "L12", "L14", "L16", "M7", "M9", "M11", "M13", "M15", "M17", "N6", "N8", "N10", "N12",
    "N14", "N18", "O7", "O9", "O11", "O13", "O15", "O17", "P8", "P10", "P12", "P14", "P16", "Q7", "Q9", "Q11", "Q13",
    "Q15", "Q17", "R8", "R10", "R12", "R14", "R16", "S9", "S11", "S13", "S15", "U9", "U11", "U13", "U15", "U17", "V8",
    "V10", "V12", "V14", "V16", "V18", "W7", "W9", "W11", "W13", "W15", "W17", "W19", "X8", "X10", "X12", "X14", "X16",
    "X18", "Y7", "Y9", "Y11", "Y13", "Y15", "Y17", "Z8", "Z10", "Z12", "Z14", "Z16", "Z18", "Z22", "Z24", "Z26", "Z28",
    "AA7", "AA9", "AA11", "AA13", "AA15", "AA17", "AA19", "AA21", "AA23", "AA25", "AA27", "AB8", "AB10", "AB12", "AB14",
    "AB16", "AB18", "AB20", "AB22", "AB24", "AC7", "AC9", "AC11", "AC13", "AC15", "AC17", "AC19", "AD8", "AD10", "AD12",
    "AD14", "AD16", "AD18", "AD20", "AE7", "AE9", "AE11", "AE13", "AE17", "AE19", "AF6", "AF8", "AF10", "AF14", "AF16",
    "AG3", "AG5", "AG7", "AG9", "AG11", "AG13", "AG15", "AH2", "AH4", "AH6", "AH10", "AH12", "AH14", "AI3", "AI5", "AI7",
    "AI9", "AI11", "AI13", "AJ2", "AJ4", "AJ6", "AJ8", "AJ10", "AJ12", "AK3", "AK5", "AK7", "AK9", "AK11", "AL2", "AL4",
    "AL6", "AL8", "AL10", "AM3", "AM5", "AM7", "AM9", "AN2", "AN4", "AN6", "AO3", "AO5", "AO7", "AO9", "AP2", "AP6",
    "AP8",
];

/// entities.rb tile_lay abilities of the CA privates (P10's token, P13's
/// sawmill assignment and P28's swap live in their steps).
pub fn abilities_ca() -> BTreeMap<String, Vec<Ab>> {
    let plain = ["7", "8", "9", "80", "81", "82", "83", "544", "545", "546", "60", "169"];
    let mut m = BTreeMap::new();
    m.insert(
        "P12".to_string(),
        vec![Ab {
            reachable: true,
            closed_when_used_up: true,
            tiles: strs(&["80", "81", "82", "83", "544", "545", "546", "60", "169", "141", "142", "143", "144", "767", "768", "769"]),
            combo: strs(&["P29", "P30"]),
            consume_tile_lay: true,
            ..lay(Some(1))
        }],
    );
    let big: [(&str, &[&str]); 5] = [
        ("P14", &["T1", "T2", "T3", "T4", "T5", "T6", "T7"]),
        ("P15", &["O1", "O2", "O3", "O4", "O5", "O6", "O7", "O8"]),
        ("P16", &["M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8"]),
        ("P17", &["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8"]),
        ("P18", &["W1", "W2", "W3", "W4", "W5", "W6", "W7"]),
    ];
    for (co, tiles) in big {
        let hex = BIG_CITY_COMPANIES.iter().find(|(c, _)| *c == co).unwrap().1;
        m.insert(
            co.to_string(),
            vec![Ab {
                hexes: strs(&[hex]),
                tiles: strs(tiles),
                free: true,
                special: false,
                reachable: true,
                count_per_or: Some(1),
                ..lay(None)
            }],
        );
    }
    for (co, hex) in [("P19", "F16"), ("P20", "E11")] {
        m.insert(
            co.to_string(),
            vec![Ab {
                hexes: strs(&[hex]),
                tiles: strs(&["7", "8", "9"]),
                free: true,
                special: false,
                reachable: true,
                consume_tile_lay: true,
                closed_when_used_up: true,
                ..lay(Some(1))
            }],
        );
    }
    m.insert(
        "P21".to_string(),
        vec![Ab {
            reachable: true,
            closed_when_used_up: true,
            special: false,
            free: true,
            tiles: strs(&["1", "2", "3", "4", "5", "6", "7", "8", "9", "55", "56", "57", "58", "69", "201", "202", "621", "630", "631", "632", "633"]),
            hexes: strs(P21_HEXES),
            ..lay(Some(3))
        }],
    );
    for p in ["P29", "P30"] {
        m.insert(
            p.to_string(),
            vec![Ab {
                reachable: true,
                closed_when_used_up: true,
                tiles: strs(&plain),
                combo: strs(&["P12"]),
                consume_tile_lay: true,
                ..lay(Some(1))
            }],
        );
    }
    m
}

pub static FAMILY: Family = Family {
    variant: Variant::Ca,
    player_range: (data::MIN_PLAYERS, data::MAX_PLAYERS),
    // -- inherited from G1822 unless noted --------------------------------
    bidding_box_minor_count: 4,
    bidding_box_concession_count: 3,
    bidding_box_private_count: 3,
    bidding_box_start_minor: "M6",
    bidding_box_start_concession: "C2",
    bidding_box_start_private: "P1",
    bidding_tokens: data::bidding_tokens,
    bidding_tokens_per_action: 3,
    min_bid_increment: 5,
    minor_start_par_price: 50,
    minor_bidbox_price: 200,
    token_price: 100,
    upgrade_cost_l_to_2: 80,
    player_loan_interest_rate: 50,
    starting_companies: data::STARTING_COMPANIES,
    // no 1822+ option in CA: the _PLUS lists are never read
    starting_companies_plus: data::STARTING_COMPANIES,
    starting_corporations: data::STARTING_CORPORATIONS,
    starting_corporations_plus: data::STARTING_CORPORATIONS,
    plus_expansion_bidbox: [
        &["P1", "P3", "P4", "P13", "P14", "P19"],
        &["P2", "P5", "P8", "P10", "P11", "P12", "P21"],
        &["P6", "P7", "P9", "P15", "P16", "P17", "P18", "P20"],
    ],
    two_home_corporation: "CPR",
    minor_14_id: "13",
    minor_14_home_hex: "AC21",
    company_mtonr: "",
    company_lcdr: "",
    company_egr: "",
    company_double_cash: "P7",
    company_double_cash_revenue: &[0, 0, 0, 20, 20, 40, 40, 60],
    company_gswr: "",
    company_gswr_discount: 40,
    company_ber: "P12",
    company_lsr: "P21",
    company_10x_revenue: "P8",
    company_osth: "P11",
    company_lur: "",
    company_chpr: "P28",
    company_5x_revenue: "P9",
    company_hsbc: "",
    company_hsbc_tiles: &["N21", "N23"],
    private_mail_contracts: &["P22", "P23"],
    private_remove_revenue: &[
        "P1", "P5", "P6", "P7", "P14", "P15", "P16", "P17", "P18", "P22", "P23", "P24", "P25", "P26", "P27", "P28",
    ],
    private_phase_revenue: &["P8", "P9"],
    private_trains: &["P1", "P2", "P3", "P4", "P5", "P6", "P26", "P27"],
    private_acquisition,
    company_trains: &[
        ("P3", "2P-0"),
        ("P4", "2P-1"),
        ("P1", "5P-0"),
        ("P5", "P-0"),
        ("P6", "P-1"),
        ("P2", "LP-0"),
        ("P26", "G-0"),
        ("P27", "G-1"),
    ],
    concessions: &[
        ("C1", "CNoR"),
        ("C2", "CPR"),
        ("C3", "GNWR"),
        ("C4", "GT"),
        ("C5", "GTP"),
        ("C6", "GWR"),
        ("C7", "ICR"),
        ("C8", "NTR"),
        ("C9", "PGE"),
        ("C10", "QMOO"),
    ],
    abilities: abilities_ca,
    corp_token_abilities: &[],
    consent_blockers: &[],
    extra_trains: &["2P", "P", "LP", "G"],
    extra_train_pullman: "P",
    extra_train_permanents: &["2P", "LP"],
    local_trains: &["L", "LP"],
    e_train: "E",
    local_train_can_carry_mail: false,
    trains: data::trains,
    train_variant: data::train_variant,
    train_variants: data::train_variants,
    destination: data::destination,
    exchange_tokens: data::exchange_tokens,
    london_hex: "M38",
    route_group_hexes: &["AC21", "AE15", "AF12", "AH8", "N16"],
    english_channel_hex: "",
    france_hex: "",
    cardiff_hex: "F35",
    merthyr_tydfil_pontypool_hex: "F33",
    upgradeable_s_hex_name: "D35",
    upgradeable_s_yellow_city_tile: "57",
    upgradeable_s_yellow_rotations: &[2, 5],
    double_hex: &["G13", "I15", "AB22", "AG13", "AH10"],
    track_plain: &["7", "8", "9", "80", "81", "82", "83", "544", "545", "546", "60", "169"],
    track_town: &["3", "4", "58", "141", "142", "143", "144", "767", "768", "769", "X17"],
    minor_green_upgrade: &["yellow", "green"],
    hex_dsl,
    hex_dsl_all: data::HEX_DSL,
    tile_dsl,
    hex_index: data::hex_index,
    must_sell_in_blocks: true,
    game_end_lock_first_trigger: true,
};
