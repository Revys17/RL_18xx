//! Static game data + title impl for **1822: The Railways of Great Britain**.
//!
//! Transcribed from tobymao/18xx `lib/engine/game/g_1822/` (clone at
//! logs/ruby/18xx, HEAD 46a592e when written):
//!   entities.rb — COMPANIES (21 privates, 10 concessions, 30 minors incl.
//!                 the 1822+ ones) and CORPORATIONS (30 minors + 10 majors)
//!   game.rb     — TRAINS, PHASES, MARKET, cash/cert/bank, rule constants
//!   map.rb      — TILES, HEXES
//!
//! The 1822 RULES (bid boxes, choices round, the 16-step operating round,
//! minor acquisition, destination tokens, ...) live in `crate::g1822`; this
//! module is data plus the `GameTitle` registration. 1822's round machinery
//! does not go through the shared 1830/1867 step lists — the title's step
//! lists below are empty and `BaseGame::process_action_internal` hands every
//! action of an 1822-family game to `crate::g1822::process_action`, which is
//! a Ruby-faithful step machine (Round::Base#process_action / skip_steps /
//! per-step `passed` flags). The 1822 variants (1822CA, 1822MX, 1822PNW —
//! `G1822::Game` subclasses in Ruby) are meant to register as further titles
//! that reuse that machinery with their own data.

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use super::{
    BorderDef, Capitalization, CompanyDef, CorpType, CorporationDef, GameTitle, HexDef, HexType,
    MarketCell, MarketMovement, MarketZone, PhaseDef, TrainDef,
};
use crate::steps::{FinishedRound, RoundStart, RoundTransition, StepDesc};
use crate::tiles::{parse_tile, TileColor, TileDef};

pub const TITLE: &str = "1822";

/// 1822: The Railways of Great Britain.
pub struct G1822;

impl GameTitle for G1822 {
    fn name(&self) -> &'static str {
        TITLE
    }
    // 1822's rounds are driven by crate::g1822 (a Ruby-faithful step
    // machine), not by the shared step lists — see the module docs.
    fn auction_steps(&self) -> &'static [StepDesc] {
        &[]
    }
    fn stock_steps(&self) -> &'static [StepDesc] {
        &[]
    }
    fn operating_steps(&self) -> &'static [StepDesc] {
        &[]
    }
    /// Never consulted (crate::g1822 sequences its own rounds — game.rb
    /// next_round!: SR → Choices → OR set → SR); kept total for the trait.
    fn next_round(
        &self,
        finished: FinishedRound,
        _phase_name: &str,
        phase_operating_rounds: u8,
    ) -> RoundTransition {
        let (increment_turn, start) = match finished {
            FinishedRound::Auction | FinishedRound::Merger { .. } => (false, RoundStart::Stock),
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
    fn minors_operate_first(&self) -> bool {
        true
    }
    /// SELL_BUY_ORDER = :sell_buy.
    fn stock_sell_after_buy(&self) -> bool {
        false
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
    fn optional_rules(&self) -> &'static [&'static str] {
        OPTIONAL_RULES
    }
    fn hex_definitions(&self) -> Vec<HexDef> {
        hex_definitions()
    }
    fn hex_future_label(&self, coord: &str) -> Option<(&'static str, &'static str)> {
        match coord {
            "D35" => Some(("S", "green")),
            "B43" | "K42" | "M42" => Some(("T", "green")),
            _ => None,
        }
    }
    fn preprinted_hex_dsl(&self, coord: &str) -> Option<(&'static str, &'static str)> {
        let (dsl, color) = hex_dsl(coord)?;
        // Plain white hexes are blank (no preprinted tile).
        if dsl.is_empty() {
            return None;
        }
        Some((dsl, color))
    }
    fn tile_counts(&self) -> Vec<(&'static str, u32)> {
        tile_counts()
    }
    fn tile_cities(&self, _tile_id: &str) -> Option<Vec<u8>> {
        None
    }
    fn tile_catalog(&self) -> Arc<HashMap<String, TileDef>> {
        tile_catalog()
    }
    fn market_grid(&self) -> Vec<Vec<Option<MarketCell>>> {
        market_grid()
    }
    /// G1822::StockMarket: 2-D, but `up` from a cell with no cell above
    /// moves DOWN-RIGHT (r+1, c+1) — the compressed upper market.
    fn market_movement(&self) -> MarketMovement {
        MarketMovement::TwoDimensionalUpDownRight
    }
    fn hex_layout(&self) -> super::HexLayout {
        super::HexLayout::Flat
    }
    /// ALLOW_TRAIN_BUY_FROM_OTHER_PLAYERS (base default true): replay
    /// accepts trains bought from another player's corporation.
    fn train_buy_from_other_players(&self) -> bool {
        true
    }
    /// The generator offers nothing across players (the project's policy,
    /// as for 1830; docs/g1822_remaining_work.md lists the cases for the
    /// user to confirm). Replay stays Ruby-faithful.
    fn generator_cross_player(&self) -> super::CrossPlayerPolicy {
        super::CrossPlayerPolicy::RESTRICTED
    }
    fn must_buy_train_always(&self) -> bool {
        true
    }
    /// Rules-first registration: the RL bridge comes later.
    fn alphazero_bridge_ready(&self) -> bool {
        false
    }
    fn is_g1822_family(&self) -> bool {
        true
    }
}

// ---------------------------------------------------------------------------
// Optional rules (meta.rb OPTIONAL_RULES), all implemented: 1822+ (six
// more minors, three more privates, privates in three category bid boxes,
// three privates removed at setup), its No Removals / Single Stack
// modifiers (inert without plus_expansion, as in Ruby) and the multi-share
// Tax Haven.
// ---------------------------------------------------------------------------

pub const OPTIONAL_RULES: &[&str] = &[
    "plus_expansion",
    "plus_expansion_no_removals",
    "plus_expansion_single_stack",
    "tax_haven_multiple",
];

// ---------------------------------------------------------------------------
// Rule constants (game.rb)
// ---------------------------------------------------------------------------

pub const BANK_CASH: i32 = 12_000;

/// CERT_LIMIT = { 2 => 33, 3 => 26, 4 => 20, 5 => 16, 6 => 13, 7 => 11 }
pub fn cert_limit(num_players: u8) -> u8 {
    match num_players {
        2 => 33,
        3 => 26,
        4 => 20,
        5 => 16,
        6 => 13,
        7 => 11,
        _ => panic!("1822: invalid player count {num_players}"),
    }
}

/// STARTING_CASH = { 2 => 900, 3 => 700, 4 => 525, 5 => 420, 6 => 350, 7 => 300 }
pub fn starting_cash(num_players: u8) -> i32 {
    match num_players {
        2 => 900,
        3 => 700,
        4 => 525,
        5 => 420,
        6 => 350,
        7 => 300,
        _ => panic!("1822: invalid player count {num_players}"),
    }
}

/// PLAYER_RANGE (meta.rb).
pub const MIN_PLAYERS: u8 = 2;
pub const MAX_PLAYERS: u8 = 7;

// ---------------------------------------------------------------------------
// Companies (entities.rb COMPANIES) — Ruby order: privates, concessions,
// minors. Values: minors/concessions 100 (their bid-box min price), privates
// 0 (min bid 0). Abilities are implemented by crate::g1822 keyed on the
// game.rb COMPANY_* constants, exactly as G1822::Game does.
// ---------------------------------------------------------------------------

const fn co(sym: &'static str, name: &'static str, value: i32, revenue: i32) -> CompanyDef {
    CompanyDef {
        sym,
        name,
        value,
        revenue,
        abilities: &[],
        discount: 0,
        auctionable: true,
    }
}

pub fn companies() -> Vec<CompanyDef> {
    vec![
        co("P1", "P1-BEC (5-Train)", 0, 5),
        co("P2", "P2-MtonR (Remove Town)", 0, 10),
        co("P3", "P3-S&HR (Permanent 2T)", 0, 0),
        co("P4", "P4-SDR (Permanent 2T)", 0, 0),
        co("P5", "P5-LC&DR (English Channel)", 0, 10),
        co("P6", "P6-L&SR (Mail Contract)", 0, 10),
        co("P7", "P7-S&BR (Mail Contract)", 0, 10),
        co("P8", "P8-E&GR (Hill Discount)", 0, 10),
        co("P9", "P9-M&GNR (Double Cash)", 0, 10),
        co("P10", "P10-G&SWR (River Discount)", 0, 10),
        co("P11", "P11-B&ER (Adv. Tile Lay)", 0, 10),
        co("P12", "P12-L&SR (Extra Tile Lay)", 0, 10),
        co("P13", "P13-YN&BR (Pullman)", 0, 10),
        co("P14", "P14-K&TR (Pullman)", 0, 10),
        co("P15", "P15-HR (£10x Phase)", 0, 0),
        co("P16", "P16-TH (Tax Haven)", 0, 0),
        co("P17", "P17-LUR (Move Card)", 0, 10),
        co("P18", "P18-C&HPR (Station Swap)", 0, 10),
        co("P19", "P19-AEC (Perm. L Train)", 0, 0),
        co("P20", "P20-C&WR (£5x Phase)", 0, 0),
        co("P21", "P21-HSBC (Grimsby/Hull Bridge)", 0, 10),
        co("C1", "CONCESSION: LNWR", 100, 10),
        co("C2", "CONCESSION: GWR", 100, 10),
        co("C3", "CONCESSION: LBSCR", 100, 10),
        co("C4", "CONCESSION: SECR", 100, 10),
        co("C5", "CONCESSION: CR", 100, 10),
        co("C6", "CONCESSION: MR", 100, 10),
        co("C7", "CONCESSION: LYR", 100, 10),
        co("C8", "CONCESSION: NBR", 100, 10),
        co("C9", "CONCESSION: SWR", 100, 10),
        co("C10", "CONCESSION: NER", 100, 10),
        co("M1", "MINOR: 1. Great North of Scotland Railway", 100, 0),
        co("M2", "MINOR: 2. Lanarkshire & Dumbartonshire Railway", 100, 0),
        co("M3", "MINOR: 3. Edinburgh & Dalkeith Railway", 100, 0),
        co("M4", "MINOR: 4. Newcastle & North Shields Railway", 100, 0),
        co("M5", "MINOR: 5. Stockton and Darlington Railway", 100, 0),
        co("M6", "MINOR: 6. Furness railway", 100, 0),
        co("M7", "MINOR: 7. Warrington & Newton Railway", 100, 0),
        co("M8", "MINOR: 8. Manchester Sheffield & Lincolnshire Railway", 100, 0),
        co("M9", "MINOR: 9. East Lincolnshire Railway", 100, 0),
        co("M10", "MINOR: 10. Grand Junction Railway", 100, 0),
        co("M11", "MINOR: 11. Great Northern Railway", 100, 0),
        co("M12", "MINOR: 12. Eastern Union Railway", 100, 0),
        co("M13", "MINOR: 13. Headcorn & Maidstone Junction Light Railway", 100, 0),
        co("M14", "MINOR: 14. Metropolitan Railway", 100, 0),
        co("M15", "MINOR: 15. London Tilbury & Southend Railway", 100, 0),
        co("M16", "MINOR: 16. Wycombe Railway", 100, 0),
        co("M17", "MINOR: 17. London & Southampton Railway", 100, 0),
        co("M18", "MINOR: 18. Somerset & Dorset Joint Railway", 100, 0),
        co("M19", "MINOR: 19. Penarth Harbour & Dock Railway Company", 100, 0),
        co("M20", "MINOR: 20. Monmouthshire Railway & Canal Company", 100, 0),
        co("M21", "MINOR: 21. Taff Vale railway", 100, 0),
        co("M22", "MINOR: 22. Exeter and Crediton Railway", 100, 0),
        co("M23", "MINOR: 23. West Cornwall Railway", 100, 0),
        co("M24", "MINOR: 24. The Swansea and Mumbles Railway", 100, 0),
        co("M25", "MINOR: 25. Yarmouth & Norwich Railway", 100, 0),
        co("M26", "MINOR: 26. Hull and Selby Railway", 100, 0),
        co("M27", "MINOR: 27. City of Glasgow Union Railway", 100, 0),
        co("M28", "MINOR: 28. Maryport and Carlisle Railway", 100, 0),
        co("M29", "MINOR: 29. Shropshire and Montgomeryshire Railway", 100, 0),
        co("M30", "MINOR: 30. Plymouth and Dartmoor Railway", 100, 0),
    ]
}

/// game.rb STARTING_COMPANIES (base game). The Ruby list also carries a
/// stray "16" (no such company) — harmless there, omitted here.
pub const STARTING_COMPANIES: &[&str] = &[
    "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10", "P11", "P12", "P13", "P14", "P15",
    "P16", "P17", "P18", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "M1", "M2",
    "M3", "M4", "M5", "M6", "M7", "M8", "M9", "M10", "M11", "M12", "M13", "M14", "M15", "M16",
    "M17", "M18", "M19", "M20", "M21", "M22", "M23", "M24",
];

/// game.rb STARTING_COMPANIES_PLUS (1822+; the stray "16" again omitted).
pub const STARTING_COMPANIES_PLUS: &[&str] = &[
    "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10", "P11", "P12", "P13", "P14", "P15",
    "P16", "P17", "P18", "P19", "P20", "P21", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9",
    "C10", "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9", "M10", "M11", "M12", "M13", "M14",
    "M15", "M16", "M17", "M18", "M19", "M20", "M21", "M22", "M23", "M24", "M25", "M26", "M27",
    "M28", "M29", "M30",
];

/// game.rb STARTING_CORPORATIONS_PLUS.
pub const STARTING_CORPORATIONS_PLUS: &[&str] = &[
    "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17",
    "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "LNWR", "GWR",
    "LBSCR", "SECR", "CR", "MR", "LYR", "NBR", "SWR", "NER",
];

/// game.rb STARTING_CORPORATIONS (base game).
pub const STARTING_CORPORATIONS: &[&str] = &[
    "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17",
    "18", "19", "20", "21", "22", "23", "24", "LNWR", "GWR", "LBSCR", "SECR", "CR", "MR", "LYR",
    "NBR", "SWR", "NER",
];

// ---------------------------------------------------------------------------
// Corporations (entities.rb CORPORATIONS)
// ---------------------------------------------------------------------------

const SHARES_MINOR: &[u8] = &[100];
/// Majors omit `shares:` (base default 20% director + 8 × 10%) except the
/// LNWR, whose director certificate is 10% (shares: [10] × 10).
const SHARES_MAJOR: &[u8] = &[20, 10, 10, 10, 10, 10, 10, 10, 10];
const SHARES_LNWR: &[u8] = &[10, 10, 10, 10, 10, 10, 10, 10, 10, 10];

fn minor(sym: &'static str, name: &'static str, home: &'static str, city: u8) -> CorporationDef {
    CorporationDef {
        sym,
        name,
        // M14's home token costs £20 (tokens: [20]); every other minor [0].
        token_prices: if sym == "14" { &[20] } else { &[0] },
        home_hex: home,
        home_city_index: city,
        reserved: true,
        shares: SHARES_MINOR,
        float_percent: 100,
        capitalization: Capitalization::Incremental,
        corp_type: CorpType::Minor,
        max_ownership_percent: 100,
        always_market_price: true,
    }
}

fn major(sym: &'static str, name: &'static str, home: &'static str, city: u8) -> CorporationDef {
    let lnwr = sym == "LNWR";
    CorporationDef {
        sym,
        name,
        token_prices: &[0, 100],
        home_hex: home,
        home_city_index: city,
        reserved: true,
        shares: if lnwr { SHARES_LNWR } else { SHARES_MAJOR },
        float_percent: if lnwr { 10 } else { 20 },
        capitalization: Capitalization::Incremental,
        corp_type: CorpType::Major,
        max_ownership_percent: 60,
        always_market_price: true,
    }
}

pub fn corporations() -> Vec<CorporationDef> {
    vec![
        minor("1", "Great North of Scotland Railway", "H1", 0),
        minor("2", "Lanarkshire & Dumbartonshire Railway", "E2", 0),
        minor("3", "Edinburgh & Dalkeith Railway", "H5", 0),
        minor("4", "Newcastle & North Shields Railway", "K10", 0),
        minor("5", "Stockton and Darlington Railway", "J15", 0),
        minor("6", "Furness railway", "G16", 0),
        minor("7", "Warrington & Newton Railway", "H23", 0),
        minor("8", "Manchester Sheffield & Lincolnshire Railway", "K24", 0),
        minor("9", "East Lincolnshire Railway", "N23", 0),
        minor("10", "Grand Junction Railway", "I30", 0),
        minor("11", "Great Northern Railway", "M30", 0),
        minor("12", "Eastern Union Railway", "P35", 0),
        minor("13", "Headcorn & Maidstone Junction Light Railway", "O40", 0),
        // Minor 14 has no `city:` — its London home city is CHOSEN
        // (PENDING_HOME_TOKENERS; home_token_locations → London).
        minor("14", "Metropolitan Railway", "M38", u8::MAX),
        minor("15", "London Tilbury & Southend Railway", "M38", 4),
        minor("16", "Wycombe Railway", "M38", 2),
        minor("17", "London & Southampton Railway", "J41", 0),
        minor("18", "Somerset & Dorset Joint Railway", "I42", 0),
        minor("19", "Penarth Harbour & Dock Railway Company", "F35", 0),
        minor("20", "Monmouthshire Railway & Canal Company", "F33", 0),
        minor("21", "Taff Vale railway", "E34", 0),
        minor("22", "Exeter and Crediton Railway", "D41", 0),
        minor("23", "West Cornwall Railway", "A42", 0),
        minor("24", "The Swansea and Mumbles Railway", "D35", 0),
        minor("25", "Yarmouth & Norwich Railway", "Q30", 0),
        minor("26", "Hull and Selby Railway", "N21", 0),
        minor("27", "City of Glasgow Union Railway", "E6", 0),
        minor("28", "Maryport and Carlisle Railway", "G12", 0),
        minor("29", "Shropshire and Montgomeryshire Railway", "E28", 0),
        minor("30", "Plymouth and Dartmoor Railway", "B43", 0),
        major("LNWR", "London and North Western Railway", "M38", 3),
        major("GWR", "Great Western Railway", "M38", 1),
        major("LBSCR", "London, Brighton and South Coast Railway", "M38", 0),
        major("SECR", "South Eastern & Chatham Railway", "M38", 5),
        major("CR", "Caledonian Railway", "E6", 0),
        major("MR", "Midland Railway", "J29", 0),
        major("LYR", "Lancashire & Yorkshire Railway", "G22", 0),
        major("NBR", "North British Railway", "H5", 0),
        major("SWR", "South Wales Railway", "H33", 0),
        major("NER", "North Eastern Railway", "L19", 0),
    ]
}

/// Major destinations (entities.rb `destination_coordinates`).
pub fn destination(sym: &str) -> Option<&'static str> {
    Some(match sym {
        "LNWR" => "I22",
        "GWR" => "G36",
        "LBSCR" => "M42",
        "SECR" => "P41",
        "CR" => "G12",
        "MR" => "L19",
        "LYR" => "I22",
        "NBR" => "H1",
        "SWR" => "C34",
        "NER" => "H5",
        _ => return None,
    })
}

/// game.rb EXCHANGE_TOKENS.
pub fn exchange_tokens(sym: &str) -> u32 {
    match sym {
        "LNWR" => 4,
        "GWR" | "LBSCR" | "SECR" | "CR" | "MR" | "LYR" | "NBR" | "SWR" | "NER" => 3,
        _ => 0,
    }
}

// ---------------------------------------------------------------------------
// Trains (game.rb TRAINS). TrainDef.distance is the CITY count; the L/LP's
// bucketed distance (1 city + 1 town) and the P+'s 99/99 live in
// crate::g1822. Variants (L→'2', 7→'E') get pseudo-defs with count 0 so the
// name-keyed rust/multiplier lookups see them ('2' rusts on the 4).
// ---------------------------------------------------------------------------

const UNLIMITED: u32 = u32::MAX;

const fn train(
    name: &'static str,
    distance: u32,
    price: i32,
    count: u32,
    rusts_on: Option<&'static str>,
    events: &'static [&'static str],
) -> TrainDef {
    TrainDef {
        name,
        distance,
        price,
        count,
        rusts_on,
        multiplier: 1,
        obsolete_on: None,
        events,
        available_on: None,
        discount: &[],
    }
}

pub fn trains() -> Vec<TrainDef> {
    vec![
        train("L", 1, 60, 22, Some("3"), &[]),
        train("3", 3, 200, 9, Some("6"), &[]),
        train("4", 4, 300, 6, Some("7"), &[]),
        train("5", 5, 500, 3, None, &["close_concessions"]),
        train("6", 6, 600, 3, None, &["full_capitalisation"]),
        train("7", 7, 750, UNLIMITED, None, &["phase_revenue"]),
        train("2P", 2, 0, 2, None, &[]),
        train("LP", 1, 0, 1, None, &[]),
        train("5P", 5, 500, 1, None, &[]),
        train("P+", 99, 0, 2, None, &[]),
        // -- variants (no physical cards of their own) --
        TrainDef {
            available_on: Some("1"),
            ..train("2", 2, 120, 0, Some("4"), &[])
        },
        TrainDef {
            multiplier: 2,
            ..train("E", 99, 1000, 0, None, &[])
        },
    ]
}

/// The variant a train card may be bought as (L ↔ 2, 7 ↔ E): (variant name,
/// distance, price, rusts_on).
pub fn train_variant(base: &str, variant: &str) -> Option<(&'static str, u32, i32, Option<&'static str>)> {
    match (base, variant) {
        ("L", "L") => Some(("L", 1, 60, Some("3"))),
        ("L", "2") => Some(("2", 2, 120, Some("4"))),
        ("7", "7") => Some(("7", 7, 750, None)),
        ("7", "E") => Some(("E", 99, 1000, None)),
        _ => None,
    }
}

/// The variants of a base train card (Ruby `train.variants`), base first.
pub fn train_variants(base: &str) -> &'static [&'static str] {
    match base {
        "L" => &["L", "2"],
        "7" => &["7", "E"],
        _ => &[],
    }
}

// ---------------------------------------------------------------------------
// Phases (game.rb PHASES). Ruby `train_limit: { minor:, major: }` →
// train_limit = major, minor_train_limit = minor. Phase 2 is `on: %w[2 3]`:
// the '2' is the L card's VARIANT (BuyTrain special case), a 3 bought in
// phase 1 advances straight to phase 3 (Phase#next! while next_on includes
// the sym) — `on: "2"` + the jump-to-matching-phase advance reproduce both.
// ---------------------------------------------------------------------------

pub fn phases() -> Vec<PhaseDef> {
    vec![
        PhaseDef {
            name: "1",
            on: None,
            train_limit: 4,
            minor_train_limit: Some(2),
            tiles: &["yellow"],
            operating_rounds: 1,
            status: &["minor_float_phase1"],
        },
        PhaseDef {
            name: "2",
            on: Some("2"),
            train_limit: 4,
            minor_train_limit: Some(2),
            tiles: &["yellow"],
            operating_rounds: 2,
            status: &["can_convert_concessions", "minor_float_phase2"],
        },
        PhaseDef {
            name: "3",
            on: Some("3"),
            train_limit: 4,
            minor_train_limit: Some(2),
            tiles: &["yellow", "green"],
            operating_rounds: 2,
            status: &["can_buy_trains", "can_convert_concessions", "minor_float_phase3on"],
        },
        PhaseDef {
            name: "4",
            on: Some("4"),
            train_limit: 3,
            minor_train_limit: Some(1),
            tiles: &["yellow", "green"],
            operating_rounds: 2,
            status: &["can_buy_trains", "can_convert_concessions", "minor_float_phase3on"],
        },
        PhaseDef {
            name: "5",
            on: Some("5"),
            train_limit: 2,
            minor_train_limit: Some(1),
            tiles: &["yellow", "green", "brown"],
            operating_rounds: 2,
            status: &[
                "can_buy_trains",
                "can_acquire_minor_bidbox",
                "can_par",
                "minors_green_upgrade",
                "minor_float_phase3on",
            ],
        },
        PhaseDef {
            name: "6",
            on: Some("6"),
            train_limit: 2,
            minor_train_limit: Some(1),
            tiles: &["yellow", "green", "brown"],
            operating_rounds: 2,
            status: &[
                "can_buy_trains",
                "can_acquire_minor_bidbox",
                "can_par",
                "full_capitalisation",
                "minors_green_upgrade",
                "minor_float_phase3on",
            ],
        },
        PhaseDef {
            name: "7",
            on: Some("7"),
            train_limit: 2,
            minor_train_limit: Some(1),
            tiles: &["yellow", "green", "brown", "gray"],
            operating_rounds: 2,
            status: &[
                "can_buy_trains",
                "can_acquire_minor_bidbox",
                "can_par",
                "full_capitalisation",
                "minors_green_upgrade",
                "minor_float_phase3on",
            ],
        },
    ]
}

// ---------------------------------------------------------------------------
// Stock market (game.rb MARKET; '' = no cell). Codes: p = par (minors), x =
// par_1 (majors & minors), y = no_cert_limit, e = endgame (game end when a
// share price reaches it — GAME_END_CHECK stock_market: :current_or).
// ---------------------------------------------------------------------------

const MARKET: &[&[&str]] = &[
    &["", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "550", "600", "650", "700e"],
    &["", "", "", "", "", "", "", "", "", "", "", "", "", "330", "360", "400", "450", "500", "550", "600", "650"],
    &["", "", "", "", "", "", "", "", "", "200", "220", "245", "270", "300", "330", "360", "400", "450", "500", "550", "600"],
    &["70", "80", "90", "100", "110", "120", "135", "150", "165", "180", "200", "220", "245", "270", "300", "330", "360", "400", "450", "500", "550"],
    &["60", "70", "80", "90", "100xp", "110", "120", "135", "150", "165", "180", "200", "220", "245", "270", "300", "330", "360", "400", "450", "500"],
    &["50", "60", "70", "80", "90xp", "100", "110", "120", "135", "150", "165", "180", "200", "220", "245", "270", "300", "330"],
    &["45y", "50", "60", "70", "80xp", "90", "100", "110", "120", "135", "150", "165", "180", "200", "220", "245"],
    &["40y", "45y", "50", "60", "70xp", "80", "90", "100", "110", "120", "135", "150", "165", "180"],
    &["35y", "40y", "45y", "50", "60xp", "70", "80", "90", "100", "110", "120", "135"],
    &["30y", "35y", "40y", "45y", "50p", "60", "70", "80", "90", "100"],
    &["25y", "30y", "35y", "40y", "45y", "50", "60", "70", "80"],
    &["20y", "25y", "30y", "35y", "40y", "45y", "50y", "60y"],
    &["15y", "20y", "25y", "30y", "35y", "40y", "45y"],
    &["10y", "15y", "20y", "25y", "30y", "35y"],
    &["5y", "10y", "15y", "20y", "25y"],
];

/// Parse one Ruby market code (share_price.rb from_code / TYPE_MAP).
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
            other => panic!("1822 market: unknown cell type {other}"),
        });
    }
    if zones.is_empty() {
        zones.push(MarketZone::Normal);
    }
    Some(MarketCell { price, zones })
}

pub fn market_grid() -> Vec<Vec<Option<MarketCell>>> {
    MARKET
        .iter()
        .map(|row| row.iter().map(|c| market_cell(c)).collect())
        .collect()
}

// ---------------------------------------------------------------------------
// Tiles (map.rb TILES; standard DSL from lib/engine/config/tile.rb)
// ---------------------------------------------------------------------------

pub fn tile_counts() -> Vec<(&'static str, u32)> {
    vec![
        ("1", 1),
        ("2", 1),
        ("3", 6),
        ("4", 6),
        ("5", 6),
        ("6", 8),
        ("7", u32::MAX),
        ("8", u32::MAX),
        ("9", u32::MAX),
        ("55", 1),
        ("56", 1),
        ("57", 6),
        ("58", 6),
        ("69", 1),
        ("14", 6),
        ("15", 6),
        ("80", 6),
        ("81", 6),
        ("82", 8),
        ("83", 8),
        ("141", 4),
        ("142", 4),
        ("143", 4),
        ("144", 4),
        ("207", 2),
        ("208", 1),
        ("619", 6),
        ("622", 1),
        ("63", 8),
        ("544", 6),
        ("545", 6),
        ("546", 8),
        ("611", 4),
        ("60", 2),
        ("X20", 1),
        ("405", 3),
        ("X1", 1),
        ("X2", 2),
        ("X3", 1),
        ("X4", 1),
        ("X21", 1),
        ("768", 4),
        ("767", 4),
        ("769", 6),
        ("X5", 3),
        ("X6", 1),
        ("X7", 2),
        ("X8", 1),
        ("X9", 1),
        ("X10", 3),
        ("X22", 1),
        ("169", 2),
        ("X11", 2),
        ("X12", 1),
        ("X13", 2),
        ("X14", 1),
        ("X15", 1),
        ("X16", 2),
        ("X17", 2),
        ("X18", 2),
        ("X19", 4),
        ("X23", 1),
    ]
}

/// (tile id, color, DSL) — standard tiles verbatim from config/tile.rb,
/// custom ones verbatim from map.rb TILES.
const TILE_DSL: &[(&str, &str, &str)] = &[
    ("1", "yellow", "town=revenue:10;town=revenue:10;path=a:1,b:_0;path=a:_0,b:3;path=a:0,b:_1;path=a:_1,b:4"),
    ("2", "yellow", "town=revenue:10;town=revenue:10;path=a:0,b:_0;path=a:_0,b:3;path=a:1,b:_1;path=a:_1,b:2"),
    ("3", "yellow", "town=revenue:10;path=a:0,b:_0;path=a:_0,b:1"),
    ("4", "yellow", "town=revenue:10;path=a:0,b:_0;path=a:_0,b:3"),
    ("5", "yellow", "city=revenue:20;path=a:0,b:_0;path=a:1,b:_0"),
    ("6", "yellow", "city=revenue:20;path=a:0,b:_0;path=a:2,b:_0"),
    ("7", "yellow", "path=a:0,b:1"),
    ("8", "yellow", "path=a:0,b:2"),
    ("9", "yellow", "path=a:0,b:3"),
    ("55", "yellow", "town=revenue:10;town=revenue:10;path=a:0,b:_0;path=a:_0,b:3;path=a:1,b:_1;path=a:_1,b:4"),
    ("56", "yellow", "town=revenue:10;town=revenue:10;path=a:0,b:_0;path=a:_0,b:2;path=a:1,b:_1;path=a:_1,b:3"),
    ("57", "yellow", "city=revenue:20;path=a:0,b:_0;path=a:_0,b:3"),
    ("58", "yellow", "town=revenue:10;path=a:0,b:_0;path=a:_0,b:2"),
    ("69", "yellow", "town=revenue:10;town=revenue:10;path=a:0,b:_0;path=a:_0,b:3;path=a:2,b:_1;path=a:_1,b:4"),
    ("14", "green", "city=revenue:30,slots:2;path=a:0,b:_0;path=a:1,b:_0;path=a:3,b:_0;path=a:4,b:_0"),
    ("15", "green", "city=revenue:30,slots:2;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0;path=a:3,b:_0"),
    ("80", "green", "junction;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0"),
    ("81", "green", "junction;path=a:0,b:_0;path=a:2,b:_0;path=a:4,b:_0"),
    ("82", "green", "junction;path=a:0,b:_0;path=a:1,b:_0;path=a:3,b:_0"),
    ("83", "green", "junction;path=a:0,b:_0;path=a:5,b:_0;path=a:3,b:_0"),
    ("141", "green", "town=revenue:10;path=a:0,b:_0;path=a:3,b:_0;path=a:1,b:_0"),
    ("142", "green", "town=revenue:10;path=a:0,b:_0;path=a:5,b:_0;path=a:3,b:_0"),
    ("143", "green", "town=revenue:10;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0"),
    ("144", "green", "town=revenue:10;path=a:0,b:_0;path=a:2,b:_0;path=a:4,b:_0"),
    ("207", "green", "city=revenue:40,slots:2;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0;path=a:3,b:_0;label=Y"),
    ("208", "green", "city=revenue:40,slots:2;path=a:0,b:_0;path=a:1,b:_0;path=a:3,b:_0;path=a:4,b:_0;label=Y"),
    ("619", "green", "city=revenue:30,slots:2;path=a:0,b:_0;path=a:2,b:_0;path=a:3,b:_0;path=a:4,b:_0"),
    ("622", "green", "city=revenue:40,slots:2;path=a:0,b:_0;path=a:2,b:_0;path=a:3,b:_0;path=a:4,b:_0;label=Y"),
    ("63", "brown", "city=revenue:40,slots:2;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0;path=a:3,b:_0;path=a:4,b:_0;path=a:5,b:_0"),
    ("544", "brown", "junction;path=a:0,b:_0;path=a:1,b:_0;path=a:3,b:_0;path=a:4,b:_0"),
    ("545", "brown", "junction;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0;path=a:3,b:_0"),
    ("546", "brown", "junction;path=a:0,b:_0;path=a:2,b:_0;path=a:3,b:_0;path=a:4,b:_0"),
    ("611", "brown", "city=revenue:40,slots:2;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0;path=a:3,b:_0;path=a:4,b:_0"),
    ("60", "gray", "junction;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0;path=a:3,b:_0;path=a:4,b:_0;path=a:5,b:_0"),
    // -- 1822 custom tiles (map.rb) --
    ("X20", "yellow", "city=revenue:40;city=revenue:40;city=revenue:40;city=revenue:40;city=revenue:40;city=revenue:40;path=a:0,b:_0;path=a:1,b:_1;path=a:2,b:_2;path=a:3,b:_3;path=a:4,b:_4;path=a:5,b:_5;upgrade=cost:20;label=L"),
    ("405", "green", "city=revenue:40,slots:2;path=a:0,b:_0;path=a:1,b:_0;path=a:5,b:_0;label=T"),
    ("X1", "green", "city=revenue:30,slots:3;path=a:1,b:_0;path=a:2,b:_0;path=a:4,b:_0;label=C"),
    ("X2", "green", "city=revenue:50,slots:3;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0;path=a:3,b:_0;path=a:4,b:_0;label=BM"),
    ("X3", "green", "city=revenue:30,slots:2;path=a:1,b:_0;path=a:4,b:_0;label=S"),
    ("X4", "green", "city=revenue:0;path=a:2,b:_0;path=a:3,b:_0;path=a:5,b:_0;upgrade=cost:100;label=EC"),
    ("X21", "green", "city=revenue:60;city=revenue:60;city=revenue:60;city=revenue:60;city=revenue:60;city=revenue:60;path=a:0,b:_0;path=a:1,b:_1;path=a:2,b:_2;path=a:3,b:_3;path=a:4,b:_4;path=a:5,b:_5;upgrade=cost:20;label=L"),
    ("768", "brown", "town=revenue:10;path=a:0,b:_0;path=a:2,b:_0;path=a:3,b:_0;path=a:5,b:_0"),
    ("767", "brown", "town=revenue:10;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0;path=a:3,b:_0"),
    ("769", "brown", "town=revenue:10;path=a:0,b:_0;path=a:2,b:_0;path=a:3,b:_0;path=a:4,b:_0"),
    ("X5", "brown", "city=revenue:50,slots:3;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0;path=a:3,b:_0;path=a:4,b:_0;label=Y"),
    ("X6", "brown", "city=revenue:40,slots:3;path=a:1,b:_0;path=a:2,b:_0;path=a:4,b:_0;label=C"),
    ("X7", "brown", "city=revenue:60,slots:4;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0;path=a:3,b:_0;path=a:4,b:_0;path=a:5,b:_0;label=BM"),
    ("X8", "brown", "city=revenue:40,slots:2;path=a:1,b:_0;path=a:4,b:_0;label=S"),
    ("X9", "brown", "city=revenue:0,slots:2;path=a:2,b:_0;path=a:3,b:_0;path=a:5,b:_0,lanes:2;upgrade=cost:100;label=EC"),
    ("X10", "brown", "city=revenue:50,slots:2;path=a:0,b:_0;path=a:1,b:_0;path=a:5,b:_0;label=T"),
    ("X22", "brown", "city=revenue:80;city=revenue:80;city=revenue:80;city=revenue:80;city=revenue:80;city=revenue:80;path=a:0,b:_0;path=a:1,b:_1;path=a:2,b:_2;path=a:3,b:_3;path=a:4,b:_4;path=a:5,b:_5;upgrade=cost:20;label=L"),
    ("169", "gray", "junction;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0;path=a:3,b:_0;path=a:4,b:_0"),
    ("X11", "gray", "city=revenue:60,slots:3;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0;path=a:3,b:_0;path=a:4,b:_0;label=Y"),
    ("X12", "gray", "city=revenue:60,slots:3;path=a:1,b:_0;path=a:2,b:_0;path=a:4,b:_0;label=C"),
    ("X13", "gray", "city=revenue:80,slots:4;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0;path=a:3,b:_0;path=a:4,b:_0;path=a:5,b:_0;label=BM"),
    ("X14", "gray", "city=revenue:60,slots:3;path=a:1,b:_0;path=a:4,b:_0;label=S"),
    ("X15", "gray", "city=revenue:0,slots:3;path=a:2,b:_0;path=a:3,b:_0;path=a:5,b:_0,lanes:2;label=EC"),
    ("X16", "gray", "city=revenue:60,slots:3;path=a:0,b:_0;path=a:1,b:_0;path=a:5,b:_0;label=T"),
    ("X17", "gray", "town=revenue:10;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0;path=a:3,b:_0;path=a:4,b:_0;path=a:5,b:_0"),
    ("X18", "gray", "city=revenue:50,slots:3;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0;path=a:3,b:_0;path=a:4,b:_0"),
    ("X19", "gray", "city=revenue:50,slots:3;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0;path=a:3,b:_0;path=a:4,b:_0;path=a:5,b:_0"),
    ("X23", "gray", "city=revenue:100;city=revenue:100;city=revenue:100;city=revenue:100;city=revenue:100;city=revenue:100;path=a:0,b:_0;path=a:1,b:_1;path=a:2,b:_2;path=a:3,b:_3;path=a:4,b:_4;path=a:5,b:_5;label=L"),
];

pub fn color_of(name: &str) -> TileColor {
    match name {
        "yellow" => TileColor::Yellow,
        "green" => TileColor::Green,
        "brown" => TileColor::Brown,
        "gray" => TileColor::Gray,
        "red" => TileColor::Red,
        _ => TileColor::White,
    }
}

pub fn tile_catalog() -> Arc<HashMap<String, TileDef>> {
    static CATALOG: OnceLock<Arc<HashMap<String, TileDef>>> = OnceLock::new();
    CATALOG
        .get_or_init(|| {
            Arc::new(
                TILE_DSL
                    .iter()
                    .map(|(id, color, dsl)| (id.to_string(), parse_tile(id, dsl, color_of(color))))
                    .collect(),
            )
        })
        .clone()
}

/// The tile DSL for a catalog id (Ruby `Tile.for`).
pub fn tile_dsl(id: &str) -> Option<(&'static str, &'static str)> {
    TILE_DSL.iter().find(|(t, _, _)| *t == id).map(|(_, c, d)| (*d, *c))
}

// ---------------------------------------------------------------------------
// Map (map.rb HEXES, generated verbatim by logs/tools/gen_map.py from the
// Ruby hash: coord => (DSL, color)). Plain white hexes carry "".
// ---------------------------------------------------------------------------

pub fn hex_dsl(coord: &str) -> Option<(&'static str, &'static str)> {
    HEX_DSL.iter().find(|(c, _, _)| *c == coord).map(|(_, d, col)| (*d, *col))
}

pub const HEX_DSL: &[(&str, &str, &str)] = &[
    ("B39", "", "white"),
    ("C10", "", "white"),
    ("D9", "", "white"),
    ("E8", "", "white"),
    ("E12", "", "white"),
    ("F41", "", "white"),
    ("G2", "", "white"),
    ("G6", "", "white"),
    ("G26", "", "white"),
    ("G38", "", "white"),
    ("G40", "", "white"),
    ("H11", "", "white"),
    ("H27", "", "white"),
    ("H29", "", "white"),
    ("H31", "", "white"),
    ("H41", "", "white"),
    ("I6", "", "white"),
    ("I28", "", "white"),
    ("I34", "", "white"),
    ("I36", "", "white"),
    ("J9", "", "white"),
    ("J11", "", "white"),
    ("J13", "", "white"),
    ("J17", "", "white"),
    ("J27", "", "white"),
    ("J33", "", "white"),
    ("J35", "", "white"),
    ("J37", "", "white"),
    ("K8", "", "white"),
    ("K16", "", "white"),
    ("K18", "", "white"),
    ("K22", "", "white"),
    ("K26", "", "white"),
    ("K32", "", "white"),
    ("K34", "", "white"),
    ("K40", "", "white"),
    ("L15", "", "white"),
    ("L17", "", "white"),
    ("L23", "", "white"),
    ("L25", "", "white"),
    ("L27", "", "white"),
    ("L29", "", "white"),
    ("L31", "", "white"),
    ("L35", "", "white"),
    ("L41", "", "white"),
    ("M18", "", "white"),
    ("M20", "", "white"),
    ("M24", "", "white"),
    ("M32", "", "white"),
    ("M34", "", "white"),
    ("N19", "", "white"),
    ("N25", "", "white"),
    ("N31", "", "white"),
    ("N35", "", "white"),
    ("N41", "", "white"),
    ("O32", "", "white"),
    ("O34", "", "white"),
    ("P27", "", "white"),
    ("P29", "", "white"),
    ("P31", "", "white"),
    ("P33", "", "white"),
    ("Q28", "", "white"),
    ("Q32", "", "white"),
    ("Q34", "", "white"),
    ("H43", "border=edge:4,type:impassable", "white"),
    ("D37", "border=edge:3,type:impassable;border=edge:4,type:impassable", "white"),
    ("N27", "border=edge:0,type:impassable;border=edge:5,type:impassable", "white"),
    ("E36", "border=edge:0,type:impassable;border=edge:1,type:impassable;border=edge:5,type:impassable", "white"),
    ("G10", "border=edge:0,type:water,cost:40", "white"),
    ("O38", "border=edge:2,type:water,cost:40;border=edge:3,type:water,cost:40;border=edge:5,type:impassable", "white"),
    ("N37", "border=edge:0,type:water,cost:40;border=edge:5,type:water,cost:40;stub=edge:1", "white"),
    ("L37", "stub=edge:5", "white"),
    ("L39", "stub=edge:4", "white"),
    ("M40", "stub=edge:3", "white"),
    ("C42", "upgrade=cost:20,terrain:swamp", "white"),
    ("F39", "upgrade=cost:20,terrain:swamp", "white"),
    ("L21", "upgrade=cost:20,terrain:swamp", "white"),
    ("M28", "upgrade=cost:20,terrain:swamp", "white"),
    ("O28", "upgrade=cost:20,terrain:swamp;border=edge:1,type:impassable;border=edge:2,type:impassable", "white"),
    ("E38", "upgrade=cost:20,terrain:swamp;border=edge:3,type:impassable", "white"),
    ("H35", "upgrade=cost:20,terrain:swamp;border=edge:2,type:water,cost:40", "white"),
    ("N39", "upgrade=cost:20,terrain:swamp;border=edge:3,type:water,cost:40;stub=edge:2", "white"),
    ("F37", "upgrade=cost:20,terrain:swamp;border=edge:2,type:impassable;border=edge:3,type:impassable", "white"),
    ("D43", "upgrade=cost:40,terrain:swamp", "white"),
    ("I32", "upgrade=cost:40,terrain:swamp", "white"),
    ("M22", "upgrade=cost:40,terrain:swamp", "white"),
    ("N29", "upgrade=cost:40,terrain:swamp;border=edge:3,type:impassable;border=edge:4,type:impassable", "white"),
    ("B41", "upgrade=cost:40,terrain:hill", "white"),
    ("D39", "upgrade=cost:40,terrain:hill", "white"),
    ("G14", "upgrade=cost:40,terrain:hill", "white"),
    ("G30", "upgrade=cost:40,terrain:hill", "white"),
    ("H39", "upgrade=cost:40,terrain:hill", "white"),
    ("I12", "upgrade=cost:40,terrain:hill", "white"),
    ("I24", "upgrade=cost:40,terrain:hill", "white"),
    ("I38", "upgrade=cost:40,terrain:hill", "white"),
    ("J39", "upgrade=cost:40,terrain:hill", "white"),
    ("C40", "upgrade=cost:60,terrain:hill", "white"),
    ("E10", "upgrade=cost:60,terrain:hill", "white"),
    ("F9", "upgrade=cost:60,terrain:hill", "white"),
    ("G8", "upgrade=cost:60,terrain:hill", "white"),
    ("H7", "upgrade=cost:60,terrain:hill", "white"),
    ("H9", "upgrade=cost:60,terrain:hill", "white"),
    ("H15", "upgrade=cost:60,terrain:hill", "white"),
    ("I8", "upgrade=cost:60,terrain:hill", "white"),
    ("I10", "upgrade=cost:60,terrain:hill", "white"),
    ("J7", "upgrade=cost:60,terrain:hill", "white"),
    ("J23", "upgrade=cost:60,terrain:hill", "white"),
    ("J25", "upgrade=cost:60,terrain:hill", "white"),
    ("I14", "upgrade=cost:80,terrain:mountain", "white"),
    ("I16", "upgrade=cost:80,terrain:mountain", "white"),
    ("I18", "upgrade=cost:80,terrain:mountain", "white"),
    ("I20", "upgrade=cost:80,terrain:mountain", "white"),
    ("J19", "upgrade=cost:80,terrain:mountain", "white"),
    ("C38", "town=revenue:0", "white"),
    ("D11", "town=revenue:0", "white"),
    ("E40", "town=revenue:0", "white"),
    ("F3", "town=revenue:0", "white"),
    ("F5", "town=revenue:0", "white"),
    ("G20", "town=revenue:0", "white"),
    ("G28", "town=revenue:0", "white"),
    ("G32", "town=revenue:0", "white"),
    ("G42", "town=revenue:0", "white"),
    ("H13", "town=revenue:0", "white"),
    ("H25", "town=revenue:0", "white"),
    ("I26", "town=revenue:0", "white"),
    ("J31", "town=revenue:0", "white"),
    ("K12", "town=revenue:0", "white"),
    ("K36", "town=revenue:0", "white"),
    ("M16", "town=revenue:0", "white"),
    ("M26", "town=revenue:0", "white"),
    ("N33", "town=revenue:0", "white"),
    ("O42", "town=revenue:0", "white"),
    ("H17", "town=revenue:0;border=edge:2,type:impassable", "white"),
    ("P39", "town=revenue:0;border=edge:2,type:impassable", "white"),
    ("H3", "town=revenue:0;border=edge:1,type:impassable;border=edge:0,type:water,cost:40", "white"),
    ("F11", "town=revenue:0;border=edge:5,type:impassable", "white"),
    ("O36", "town=revenue:0;border=edge:0,type:water,cost:40", "white"),
    ("M36", "town=revenue:0;stub=edge:0", "white"),
    ("F7", "town=revenue:0;town=revenue:0", "white"),
    ("H21", "town=revenue:0;town=revenue:0", "white"),
    ("H37", "town=revenue:0;town=revenue:0;upgrade=cost:20,terrain:swamp", "white"),
    ("O30", "town=revenue:0;upgrade=cost:20,terrain:swamp", "white"),
    ("G34", "town=revenue:0;upgrade=cost:20,terrain:swamp;border=edge:0,type:water,cost:40;border=edge:5,type:water,cost:40", "white"),
    ("G24", "town=revenue:0;upgrade=cost:40,terrain:swamp", "white"),
    ("I40", "town=revenue:0;upgrade=cost:40,terrain:hill", "white"),
    ("J21", "town=revenue:0;upgrade=cost:60,terrain:hill", "white"),
    ("D41", "city=revenue:0", "white"),
    ("H19", "city=revenue:0", "white"),
    ("J15", "city=revenue:0", "white"),
    ("J29", "city=revenue:0", "white"),
    ("J41", "city=revenue:0", "white"),
    ("K10", "city=revenue:0", "white"),
    ("K14", "city=revenue:0", "white"),
    ("K20", "city=revenue:0", "white"),
    ("K24", "city=revenue:0", "white"),
    ("K28", "city=revenue:0", "white"),
    ("K30", "city=revenue:0", "white"),
    ("K38", "city=revenue:0", "white"),
    ("L33", "city=revenue:0", "white"),
    ("M30", "city=revenue:0", "white"),
    ("P35", "city=revenue:0", "white"),
    ("P41", "city=revenue:0", "white"),
    ("I42", "city=revenue:0;border=edge:1,type:impassable", "white"),
    ("G4", "city=revenue:0;border=edge:4,type:impassable", "white"),
    ("G16", "city=revenue:0;border=edge:5,type:impassable", "white"),
    ("G12", "city=revenue:0;border=edge:2,type:impassable;border=edge:3,type:water,cost:40", "white"),
    ("D35", "city=revenue:20,loc:center;town=revenue:10,loc:1;path=a:_0,b:_1;border=edge:0,type:impassable;future_label=label:S,color:green", "white"),
    ("M38", "city=revenue:20,groups:London;city=revenue:20,groups:London;city=revenue:20,groups:London;city=revenue:20,groups:London;city=revenue:20,groups:London;city=revenue:20,groups:London;path=a:0,b:_0;path=a:1,b:_1;path=a:2,b:_2;path=a:3,b:_3;path=a:4,b:_4;path=a:5,b:_5;upgrade=cost:20;label=L", "white"),
    ("B43", "city=revenue:0;future_label=label:T,color:green", "white"),
    ("K42", "city=revenue:0;future_label=label:T,color:green", "white"),
    ("M42", "city=revenue:0;future_label=label:T,color:green", "white"),
    ("L19", "city=revenue:0;upgrade=cost:20,terrain:swamp", "white"),
    ("Q30", "city=revenue:0;upgrade=cost:20,terrain:swamp", "white"),
    ("H23", "city=revenue:0;upgrade=cost:40,terrain:swamp", "white"),
    ("H33", "city=revenue:0;upgrade=cost:40,terrain:swamp", "white"),
    ("O40", "city=revenue:0;upgrade=cost:40,terrain:hill", "white"),
    ("N21", "city=revenue:0;upgrade=cost:20,terrain:swamp;border=edge:0,type:water,cost:40", "white"),
    ("N23", "city=revenue:0;upgrade=cost:20,terrain:swamp;border=edge:3,type:water,cost:40", "white"),
    ("F35", "city=revenue:30,slots:2;path=a:1,b:_0;path=a:2,b:_0;path=a:4,b:_0;border=edge:0,type:impassable;border=edge:5,type:impassable;label=C", "yellow"),
    ("G22", "city=revenue:30,slots:2;path=a:0,b:_0;path=a:3,b:_0;path=a:5,b:_0;label=Y", "yellow"),
    ("G36", "city=revenue:30,slots:2;path=a:0,b:_0;path=a:4,b:_0;path=a:5,b:_0;border=edge:2,type:impassable;border=edge:3,type:water,cost:40;upgrade=cost:20,terrain:swamp;label=Y", "yellow"),
    ("H5", "city=revenue:30,slots:2;path=a:1,b:_0;path=a:2,b:_0;path=a:5,b:_0;border=edge:3,type:water,cost:40;label=Y", "yellow"),
    ("I22", "city=revenue:40,slots:2;path=a:0,b:_0;path=a:1,b:_0;path=a:4,b:_0;upgrade=cost:60,terrain:hill;label=BM", "yellow"),
    ("I30", "city=revenue:40,slots:2;path=a:0,b:_0;path=a:2,b:_0;path=a:4,b:_0;upgrade=cost:40,terrain:swamp;label=BM", "yellow"),
    ("P43", "city=revenue:0;upgrade=cost:100;label=EC", "yellow"),
    ("A42", "city=revenue:yellow_40|green_30|brown_30|gray_40,slots:2,loc:1.5;path=a:4,b:_0,terminal:1;path=a:5,b:_0,terminal:1", "gray"),
    ("C34", "city=revenue:yellow_10|green_20|brown_30|gray_40,slots:2;path=a:5,b:_0,terminal:1", "gray"),
    ("E2", "city=revenue:yellow_10|green_10|brown_20|gray_20,slots:2;path=a:0,b:_0,terminal:1;path=a:5,b:_0,terminal:1", "gray"),
    ("E4", "path=a:0,b:3", "gray"),
    ("E6", "city=revenue:yellow_40|green_50|brown_60|gray_70,slots:3,loc:1;path=a:0,b:_0;path=a:3,b:_0;path=a:4,b:_0;path=a:5,b:_0", "gray"),
    ("E26", "path=a:0,b:4,lanes:2", "gray"),
    ("E28", "city=revenue:yellow_10|green_20|brown_20|gray_30,slots:3,loc:1.5;path=a:0,b:_0,lanes:2,terminal:1;path=a:3,b:_0,lanes:2,terminal:1;path=a:4,b:_0,lanes:2,terminal:1;path=a:5,b:_0,lanes:2,terminal:1", "gray"),
    ("E30", "path=a:3,b:5,lanes:2", "gray"),
    ("E32", "path=a:0,b:5", "gray"),
    ("E34", "city=revenue:yellow_30|green_40|brown_30|gray_10,slots:2,loc:0;path=a:3,b:_0;path=a:4,b:_0,terminal:1,ignore:1;path=a:5,b:_0", "gray"),
    ("F23", "city=revenue:yellow_20|green_20|brown_30|gray_40,slots:2;path=a:5,b:_0,terminal:1", "gray"),
    ("F25", "path=a:1,b:4,a_lane:2.0;path=a:1,b:5,a_lane:2.1", "gray"),
    ("F27", "path=a:1,b:4,a_lane:2.0;path=a:1,b:5,a_lane:2.1", "gray"),
    ("F29", "path=a:2,b:4,a_lane:2.0;path=a:2,b:5,a_lane:2.1", "gray"),
    ("F31", "path=a:2,b:4,a_lane:2.0;path=a:2,b:5,a_lane:2.1", "gray"),
    ("F33", "city=revenue:yellow_20|green_40|brown_30|gray_10,slots:2,loc:4;path=a:1,b:_0;path=a:2,b:_0,terminal:1,ignore:1;path=a:5,b:_0", "gray"),
    ("H1", "city=revenue:yellow_30|green_40|brown_50|gray_60,slots:2;path=a:0,b:_0,terminal:1;path=a:1,b:_0,terminal:1", "gray"),
    ("Q44", "offboard=revenue:yellow_0|green_60|brown_90|gray_120,visit_cost:0;path=a:2,b:_0,lanes:2", "gray"),
    ("L11", "junction;path=a:2,b:_0,terminal:1", "blue"),
    ("J43", "junction;path=a:2,b:_0,terminal:1", "blue"),
    ("Q36", "junction;path=a:2,b:_0,terminal:1", "blue"),
    ("Q42", "junction;path=a:2,b:_0,terminal:1", "blue"),
    ("R31", "junction;path=a:2,b:_0,terminal:1", "blue"),
    ("F17", "junction;path=a:4,b:_0,terminal:1", "blue"),
    ("F15", "junction;path=a:5,b:_0,terminal:1", "blue"),
    ("F21", "junction;path=a:5,b:_0,terminal:1", "blue"),
];

/// Per-hex border list parsed once from the DSL (`border=edge:N[,type:T][,cost:C]`).
fn parsed_borders() -> &'static HashMap<&'static str, Vec<BorderDef>> {
    static BORDERS: OnceLock<HashMap<&'static str, Vec<BorderDef>>> = OnceLock::new();
    BORDERS.get_or_init(|| {
        let mut out = HashMap::new();
        for (coord, dsl, _) in HEX_DSL {
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
                v.push(BorderDef {
                    edge,
                    cost,
                    impassable: kind == "impassable",
                });
            }
            out.insert(*coord, v);
        }
        out
    })
}

pub fn hex_definitions() -> Vec<HexDef> {
    let borders = parsed_borders();
    HEX_DSL
        .iter()
        .map(|(coord, dsl, color)| {
            let hex_type = if *color == "gray" && dsl.starts_with("offboard") {
                HexType::Offboard {
                    yellow_revenue: 0,
                    brown_revenue: 0,
                }
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
                borders: borders
                    .get(coord)
                    .map(|v| -> &'static [BorderDef] { v.as_slice() })
                    .unwrap_or(&[]),
            }
        })
        .collect()
}

/// game.rb LOCATION_NAMES subset used in logs/tests (Ruby location_name).
pub const LONDON_HEX: &str = "M38";
pub const ENGLISH_CHANNEL_HEX: &str = "P43";
pub const FRANCE_HEX: &str = "Q44";
pub const CARDIFF_HEX: &str = "F35";
pub const MERTHYR_TYDFIL_PONTYPOOL_HEX: &str = "F33";
pub const UPGRADEABLE_S_HEX_NAME: &str = "D35";
