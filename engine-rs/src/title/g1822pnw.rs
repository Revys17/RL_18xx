//! **1822PNW** (Pacific Northwest), Ruby `G1822PNW::Game < G1822::Game`
//! (tobymao/18xx `lib/engine/game/g_1822_pnw/`).
//!
//! The data (companies, corporations, trains, phases, market, tiles, map)
//! is GENERATED from the Ruby constants into `g1822pnw_data.rs`
//! (scripts/g1822_title_data.{rb,py}); this module holds the `GameTitle`
//! registration and the family table — PNW's `self.class::` constants,
//! with everything PNW does not override copied from 1822's (Ruby
//! inheritance; a constant PNW sets to `nil` is ""). The rules PNW
//! overrides (associated minors, regional railways, the merger round,
//! builder cubes, ports, portage, the mill / ski / forest revenues, the
//! coal mine, the boomtown, token cities, ...) branch on
//! `Variant::Pnw` in `crate::g1822`.

use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, OnceLock};

use super::g1822pnw_data as data;
use super::{
    BorderDef, CompanyDef, CorpType, CorporationDef, GameTitle, HexDef, HexType, MarketCell, MarketMovement,
    MarketZone, PhaseDef, TrainDef,
};
use crate::g1822::family::{Acquisition, Family, Variant};
use crate::g1822::track::{Ab, AbKind};
use crate::steps::{FinishedRound, RoundTransition, StepDesc};
use crate::tiles::{parse_tile, TileDef};

pub use data::TITLE;

pub struct G1822Pnw;

impl GameTitle for G1822Pnw {
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
    fn corporations(&self) -> Vec<CorporationDef> {
        data::corporations()
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
    /// G1822::StockMarket (PNW keeps it; its market is one row).
    fn market_movement(&self) -> MarketMovement {
        MarketMovement::TwoDimensionalUpDownRight
    }
    fn hex_layout(&self) -> super::HexLayout {
        super::HexLayout::Pointy
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

/// Parse one Ruby market code (p = par, x = par_1, e = endgame).
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
            other => panic!("1822PNW market: unknown cell type {other}"),
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
    Some(match sym {
        "P1" => (MAJ, 5),
        "P2" => (MAJ, 2),
        "P3" => (BOTH, 1),
        "P4" => (BOTH, 1),
        "P5" => (BOTH, 3),
        "P6" => (BOTH, 3),
        "P7" => (BOTH, 1),
        "P8" => (BOTH, 1),
        "P9" => (MAJ, 3),
        "P10" => (BOTH, 1),
        "P11" => (BOTH, 3),
        "P12" => (BOTH, 2),
        "P13" => (MAJ, 3),
        "P14" => (BOTH, 3),
        "P15" => (BOTH, 3),
        "P16" => (BOTH, 2),
        "P17" => (BOTH, 5),
        "P18" => (BOTH, 3),
        "P19" => (BOTH, 3),
        "P20" => (BOTH, 1),
        "P21" => (BOTH, 2),
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
    }
}

/// entities.rb abilities of the PNW privates (tile_lay; P15 / P17's
/// assign_hexes live in the Assign step).
pub fn abilities_pnw() -> BTreeMap<String, Vec<Ab>> {
    let plain = ["7", "8", "9", "80", "81", "82", "83", "544", "545", "546", "60", "169"];
    let ports = ["B7", "D9", "E8", "I10", "N9"];
    let mut m = BTreeMap::new();
    for p in ["P7", "P8"] {
        m.insert(
            p.to_string(),
            vec![Ab { reachable: true, closed_when_used_up: true, tiles: strs(&plain), combo: strs(&["P11"]), ..lay(Some(1)) }],
        );
    }
    m.insert("P10".to_string(), vec![Ab { reachable: true, closed_when_used_up: true, track_only: true, ..lay(Some(3)) }]);
    m.insert(
        "P11".to_string(),
        vec![Ab {
            must_lay_together: true,
            lay_count: Some(2),
            upgrade_count: Some(1),
            reachable: true,
            closed_when_used_up: true,
            combo: strs(&["P7", "P8"]),
            // TileLay#setup: @count ||= @lay_count
            ..lay(Some(2))
        }],
    );
    m.insert(
        "P12".to_string(),
        vec![Ab { closed_when_used_up: true, track_only: true, hexes: strs(&ports), tiles: strs(&["P1"]), ..lay(Some(1)) }],
    );
    m.insert(
        "P13".to_string(),
        vec![Ab { closed_when_used_up: true, track_only: true, hexes: strs(&ports), tiles: strs(&["P2"]), ..lay(Some(1)) }],
    );
    m.insert(
        "P16".to_string(),
        vec![Ab {
            hexes: strs(&["E10", "F11", "G10", "H9", "I10", "M6", "M8", "N9", "O16", "O18"]),
            tiles: strs(&["PNW1", "PNW2"]),
            track_only: true,
            ..lay(Some(2))
        }],
    );
    m.insert("P18".to_string(), vec![Ab { closed_when_used_up: true, track_only: true, tiles: strs(&["PNW3"]), ..lay(Some(1)) }]);
    m.insert(
        "P19".to_string(),
        vec![Ab {
            hexes: strs(&[
                "A16", "B15", "C14", "D15", "E16", "F17", "G4", "G18", "H5", "I6", "H17", "I18", "J17", "K16", "L15",
                "M14", "N13",
            ]),
            tiles: strs(&["PNW4"]),
            free: true,
            track_only: true,
            ..lay(Some(1))
        }],
    );
    m
}

pub static FAMILY: Family = Family {
    variant: Variant::Pnw,
    player_range: (data::MIN_PLAYERS, data::MAX_PLAYERS),
    // -- inherited from G1822 unless noted --------------------------------
    bidding_box_minor_count: 4,
    bidding_box_concession_count: 3,
    bidding_box_private_count: 3,
    // PNW's own setup_companies never reads these (inherited values kept)
    bidding_box_start_minor: "M24",
    bidding_box_start_concession: "C1",
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
    // no 1822+ option in PNW: the _PLUS lists are never read
    starting_companies_plus: data::STARTING_COMPANIES,
    starting_corporations: data::STARTING_CORPORATIONS,
    starting_corporations_plus: data::STARTING_CORPORATIONS,
    plus_expansion_bidbox: [
        &["P1", "P3", "P4", "P13", "P14", "P19"],
        &["P2", "P5", "P8", "P10", "P11", "P12", "P21"],
        &["P6", "P7", "P9", "P15", "P16", "P17", "P18", "P20"],
    ],
    two_home_corporation: "LNWR",
    minor_14_id: "",
    minor_14_home_hex: "M38",
    company_mtonr: "P2",
    company_lcdr: "",
    company_egr: "P8",
    company_double_cash: "P9",
    company_double_cash_revenue: &[0, 0, 0, 20, 20, 40, 40, 60],
    company_gswr: "P10",
    company_gswr_discount: 40,
    company_ber: "P11",
    company_lsr: "P12",
    company_10x_revenue: "",
    company_osth: "",
    company_lur: "P21",
    company_chpr: "",
    company_5x_revenue: "P20",
    company_hsbc: "P21",
    company_hsbc_tiles: &["N21", "N23"],
    private_mail_contracts: &["P9"],
    private_remove_revenue: &["P5", "P6", "P7", "P8", "P10", "P17", "P18", "P21"],
    private_phase_revenue: &[],
    private_trains: &["P1", "P2", "P3", "P4", "P5", "P6"],
    private_acquisition,
    company_trains: &[("P1", "5P-0"), ("P2", "2P-0"), ("P3", "LP-0"), ("P4", "LP-1"), ("P5", "P+-0"), ("P6", "P+-1")],
    concessions: &[],
    abilities: abilities_pnw,
    corp_token_abilities: &[],
    consent_blockers: &[],
    extra_trains: &["2P", "P+", "LP"],
    extra_train_pullman: "P+",
    extra_train_permanents: &["2P", "LP"],
    local_trains: &["L", "LP"],
    e_train: "E",
    local_train_can_carry_mail: true,
    trains: data::trains,
    train_variant: data::train_variant,
    train_variants: data::train_variants,
    destination: data::destination,
    exchange_tokens: data::exchange_tokens,
    london_hex: "M38",
    // (the router's group hex was LONDON_HEX before the field existed)
    route_group_hexes: &["M38"],
    english_channel_hex: "",
    france_hex: "",
    cardiff_hex: "F35",
    merthyr_tydfil_pontypool_hex: "F33",
    upgradeable_s_hex_name: "D35",
    upgradeable_s_yellow_city_tile: "57",
    upgradeable_s_yellow_rotations: &[2, 5],
    double_hex: &["H19", "M4"],
    track_plain: &["7", "8", "9", "80", "81", "82", "83", "544", "545", "546", "60", "169"],
    track_town: &["3", "4", "58", "141", "142", "143", "144", "767", "768", "769", "X17"],
    minor_green_upgrade: &["yellow", "green"],
    hex_dsl,
    hex_dsl_all: data::HEX_DSL,
    tile_dsl,
    hex_index: crate::g1822::family::hex_index_zero,
    must_sell_in_blocks: true,
    game_end_lock_first_trigger: false,
};
