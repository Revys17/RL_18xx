//! **1822MX** (The Railways of Mexico), Ruby `G1822MX::Game < G1822::Game`
//! (tobymao/18xx `lib/engine/game/g_1822_mx/`).
//!
//! The data (companies, corporations, trains, phases, market, tiles, map)
//! is GENERATED from the Ruby constants into `g1822mx_data.rs`
//! (scripts/g1822_title_data.{rb,py}); this module holds the `GameTitle`
//! registration and the family table — MX's `self.class::` constants, with
//! everything MX does not override copied from 1822's (Ruby inheritance; a
//! constant MX sets to `nil` is ""). The rules MX overrides (the NdeM, its
//! token auction and cash-out, the C1/M18 pairing, P7's double cash, P16's
//! stock drop, builder cubes, ports, the 3/2P train, Mexico City's M22
//! connection rule, ...) branch on `Variant::Mx` in `crate::g1822`.

use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, OnceLock};

use super::g1822mx_data as data;
use super::{
    BorderDef, CompanyDef, CorpType, CorporationDef, GameTitle, HexDef, HexType, MarketCell, MarketMovement,
    MarketZone, PhaseDef, TrainDef,
};
use crate::g1822::family::{Acquisition, Family, Variant};
use crate::g1822::track::{Ab, AbKind};
use crate::steps::{FinishedRound, RoundTransition, StepDesc};
use crate::tiles::{parse_tile, TileDef};

pub use data::TITLE;

pub struct G1822Mx;

impl GameTitle for G1822Mx {
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
    /// CERT_LIMIT (the higher_cert_limit option's CERT_LIMIT_INCREASED is
    /// applied at setup: `g1822_mx_setup`).
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
    /// map.rb `future_label=label:T,color:green` (B1 J9 L37 L39 P23).
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
    /// G1822::StockMarket (MX's market is one row).
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

/// CERT_LIMIT_INCREASED (the higher_cert_limit option; 3-5 players).
pub fn cert_limit_increased(num_players: u8) -> Option<u8> {
    match num_players {
        3 => Some(18),
        4 => Some(14),
        5 => Some(11),
        _ => None,
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
            other => panic!("1822MX market: unknown cell type {other}"),
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
    Some(match sym {
        "P1" => (MAJ, 5),
        "P2" => (MAJ, 2),
        "P3" => (BOTH, 1),
        "P4" => (BOTH, 1),
        "P5" => (BOTH, 3),
        "P6" => (BOTH, 3),
        "P7" => (MAJ, 3),
        "P8" => (BOTH, 1),
        "P9" => (BOTH, 3),
        "P10" => (BOTH, 1),
        "P11" => (BOTH, 1),
        "P12" => (BOTH, 1),
        "P13" => (BOTH, 1),
        "P14" => (MAJ, 3),
        "P15" => (MAJ, 3),
        "P16" => (MAJ, 2),
        "P17" => (BOTH, 2),
        "P18" => (MAJ, 3),
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

/// The port spikes P17 / P18 may replace (entities.rb `hexes`).
pub const PORT_SPIKES: &[&str] = &["F7", "G8", "H5", "H9", "I10", "J7", "J11", "M28", "M40", "Q28"];

/// entities.rb abilities of the MX privates (tile_lay; P16's
/// choose_ability lives in the SpecialChoose step).
pub fn abilities_mx() -> BTreeMap<String, Vec<Ab>> {
    let plain = ["7", "8", "9", "80", "81", "82", "83", "544", "545", "546", "60", "169"];
    let mut m = BTreeMap::new();
    m.insert(
        "P8".to_string(),
        vec![Ab {
            reachable: true,
            closed_when_used_up: true,
            tiles: strs(&[
                "80", "81", "82", "83", "544", "545", "546", "60", "169", "141", "142", "143", "144", "767", "768", "769",
                "X17",
            ]),
            combo: strs(&["P9", "P12", "P13"]),
            ..lay(Some(1))
        }],
    );
    m.insert(
        "P9".to_string(),
        vec![Ab {
            must_lay_together: true,
            lay_count: Some(2),
            upgrade_count: Some(1),
            reachable: true,
            closed_when_used_up: true,
            combo: strs(&["P8", "P12", "P13"]),
            // TileLay#setup: @count ||= @lay_count
            ..lay(Some(2))
        }],
    );
    for p in ["P10", "P11"] {
        m.insert(
            p.to_string(),
            vec![Ab { reachable: true, closed_when_used_up: true, track_only: true, ..lay(Some(3)) }],
        );
    }
    for p in ["P12", "P13"] {
        m.insert(
            p.to_string(),
            vec![Ab {
                reachable: true,
                closed_when_used_up: true,
                tiles: strs(&plain),
                combo: strs(&["P8", "P9"]),
                ..lay(Some(1))
            }],
        );
    }
    m.insert(
        "P17".to_string(),
        vec![Ab { closed_when_used_up: true, track_only: true, hexes: strs(PORT_SPIKES), tiles: strs(&["P1"]), ..lay(Some(1)) }],
    );
    m.insert(
        "P18".to_string(),
        vec![Ab { closed_when_used_up: true, track_only: true, hexes: strs(PORT_SPIKES), tiles: strs(&["P2"]), ..lay(Some(1)) }],
    );
    m
}

pub static FAMILY: Family = Family {
    variant: Variant::Mx,
    player_range: (data::MIN_PLAYERS, data::MAX_PLAYERS),
    // -- inherited from G1822 unless noted --------------------------------
    bidding_box_minor_count: 4,
    bidding_box_concession_count: 3,
    bidding_box_private_count: 3,
    // MX: BIDDING_BOX_START_MINOR = nil (MX's setup_companies moves M18
    // to the end of the minors instead)
    bidding_box_start_minor: "",
    bidding_box_start_concession: "C1",
    bidding_box_start_private: "P1",
    bidding_tokens: data::bidding_tokens,
    bidding_tokens_per_action: 3,
    min_bid_increment: 5,
    minor_start_par_price: 50,
    minor_bidbox_price: 200,
    token_price: 100,
    // UPGRADE_COST_L_TO_2 and MX's UPGRADE_COST_L_TO_2_PHASE_2 are both 80
    upgrade_cost_l_to_2: 80,
    player_loan_interest_rate: 50,
    starting_companies: data::STARTING_COMPANIES,
    // no 1822+ option in MX: the _PLUS lists are never read
    starting_companies_plus: data::STARTING_COMPANIES,
    starting_corporations: data::STARTING_CORPORATIONS,
    starting_corporations_plus: data::STARTING_CORPORATIONS,
    plus_expansion_bidbox: [
        &["P1", "P3", "P4", "P13", "P14", "P19"],
        &["P2", "P5", "P8", "P10", "P11", "P12", "P21"],
        &["P6", "P7", "P9", "P15", "P16", "P17", "P18", "P20"],
    ],
    two_home_corporation: "LNWR",
    // MINOR_14_ID = nil (MINOR_14_HOME_HEX inherited)
    minor_14_id: "",
    minor_14_home_hex: "M38",
    company_mtonr: "P2",
    company_lcdr: "",
    company_egr: "P8",
    // G1822's double-cash machinery IS MX's P7 (P7_REVENUE, @p7_choice):
    // MX overrides every method that reads COMPANY_DOUBLE_CASH ('P9') with
    // the same code for 'P7'.
    company_double_cash: "P7",
    company_double_cash_revenue: &[0, 0, 0, 20, 20, 40, 40, 60],
    company_gswr: "P10",
    company_gswr_discount: 40,
    company_ber: "P11",
    company_lsr: "P12",
    company_10x_revenue: "P15",
    company_osth: "",
    company_lur: "",
    company_chpr: "P18",
    company_5x_revenue: "P20",
    company_hsbc: "P21",
    company_hsbc_tiles: &["N21", "N23"],
    private_mail_contracts: &["P14", "P15"],
    private_remove_revenue: &["P5", "P6", "P7", "P8", "P10", "P17", "P18", "P21"],
    private_phase_revenue: &[],
    private_trains: &["P1", "P2", "P3", "P4", "P5", "P6"],
    private_acquisition,
    company_trains: &[("P1", "5P-0"), ("P2", "2P-0"), ("P3", "3/2P-0"), ("P4", "LP-0"), ("P5", "P+-0"), ("P6", "P+-1")],
    concessions: &[
        ("C1", "FCM"),
        ("C2", "MC"),
        ("C3", "CHP"),
        ("C4", "FNM"),
        ("C5", "MIR"),
        ("C6", "FCP"),
        ("C7", "IRM"),
    ],
    abilities: abilities_mx,
    corp_token_abilities: &[],
    consent_blockers: &[
        ("C2", "MC", "M22"),
        ("C4", "FNM", "M24"),
        ("C7", "IRM", "O24"),
        ("M14", "14", "N21"),
        ("M15", "15", "O22"),
    ],
    extra_trains: &["2P", "P+", "LP", "3/2P"],
    extra_train_pullman: "P+",
    extra_train_permanents: &["2P", "LP", "3/2P"],
    local_trains: &["L", "LP"],
    e_train: "E",
    local_train_can_carry_mail: false,
    trains: data::trains,
    train_variant: data::train_variant,
    train_variants: data::train_variants,
    destination: data::destination,
    exchange_tokens: data::exchange_tokens,
    london_hex: "M38",
    english_channel_hex: "",
    france_hex: "",
    cardiff_hex: "F35",
    merthyr_tydfil_pontypool_hex: "F33",
    upgradeable_s_hex_name: "D35",
    upgradeable_s_yellow_city_tile: "57",
    upgradeable_s_yellow_rotations: &[2, 5],
    double_hex: &["L19", "M22", "M26"],
    track_plain: &["7", "8", "9", "80", "81", "82", "83", "544", "545", "546", "60", "169"],
    track_town: &["3", "4", "58", "141", "142", "143", "144", "767", "768", "769", "X17"],
    minor_green_upgrade: &["yellow", "green"],
    hex_dsl,
    hex_dsl_all: data::HEX_DSL,
    tile_dsl,
    must_sell_in_blocks: true,
    game_end_lock_first_trigger: true,
};
