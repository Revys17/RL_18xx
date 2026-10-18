//! Per-title constants of the 1822 family: Ruby's `self.class::X` class
//! constants (G1822 uses `self.class::` throughout, so a subclass's
//! override applies inside inherited code — the `Family` table is that
//! lookup) plus the title data the step machine reads directly (map DSL,
//! trains, the companies' abilities).
//!
//! A Ruby constant a variant sets to `nil` is the empty string here (no
//! company, hex or corporation has an empty id, so comparisons and lookups
//! fail exactly as Ruby's `== nil` / `hex_by_id(nil)` do). Each title's
//! table lives with its data (`title::g1822::FAMILY`, ...); everything not
//! overridden is copied from 1822's (Ruby inheritance).

use std::collections::BTreeMap;

use crate::title::{CorpType, TrainDef};

use super::track::Ab;

/// The variants whose Ruby subclass overrides methods (behaviour branches
/// test this; constants never do).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Variant {
    G1822,
    Pnw,
    Mx,
}

/// PRIVATE_COMPANIES_ACQUISITION: (acquirer classes, phase).
pub type Acquisition = Option<(&'static [CorpType], u32)>;

pub struct Family {
    pub variant: Variant,
    /// meta.rb PLAYER_RANGE (the constructor rejects other counts).
    pub player_range: (u8, u8),

    // -- stock round, bid boxes (game.rb) ---------------------------------
    pub bidding_box_minor_count: usize,
    pub bidding_box_concession_count: usize,
    pub bidding_box_private_count: usize,
    /// BIDDING_BOX_START_MINOR / _CONCESSION / _PRIVATE ("" = none).
    pub bidding_box_start_minor: &'static str,
    pub bidding_box_start_concession: &'static str,
    pub bidding_box_start_private: &'static str,
    /// BIDDING_TOKENS by player count (the UI's limit; the server's lookup
    /// misses — see legal.rs).
    pub bidding_tokens: fn(usize) -> i32,
    pub bidding_tokens_per_action: u32,
    pub min_bid_increment: i32,
    pub minor_start_par_price: i32,
    pub minor_bidbox_price: i32,
    pub token_price: i32,
    pub upgrade_cost_l_to_2: i32,
    pub player_loan_interest_rate: i32,
    pub starting_companies: &'static [&'static str],
    pub starting_companies_plus: &'static [&'static str],
    pub starting_corporations: &'static [&'static str],
    pub starting_corporations_plus: &'static [&'static str],
    pub plus_expansion_bidbox: [&'static [&'static str]; 3],
    pub two_home_corporation: &'static str,
    pub minor_14_id: &'static str,
    pub minor_14_home_hex: &'static str,

    // -- companies ---------------------------------------------------------
    pub company_mtonr: &'static str,
    pub company_lcdr: &'static str,
    pub company_egr: &'static str,
    pub company_double_cash: &'static str,
    pub company_double_cash_revenue: &'static [i32],
    pub company_gswr: &'static str,
    pub company_gswr_discount: i32,
    pub company_ber: &'static str,
    pub company_lsr: &'static str,
    pub company_10x_revenue: &'static str,
    pub company_osth: &'static str,
    pub company_lur: &'static str,
    pub company_chpr: &'static str,
    pub company_5x_revenue: &'static str,
    pub company_hsbc: &'static str,
    pub company_hsbc_tiles: &'static [&'static str],
    pub private_mail_contracts: &'static [&'static str],
    pub private_remove_revenue: &'static [&'static str],
    pub private_phase_revenue: &'static [&'static str],
    pub private_trains: &'static [&'static str],
    pub private_acquisition: fn(&str) -> Acquisition,
    /// setup_companies' `@company_trains`: (company, train id).
    pub company_trains: &'static [(&'static str, &'static str)],
    /// The concessions' exchange-from-par abilities: (company, major).
    pub concessions: &'static [(&'static str, &'static str)],
    /// The companies' tile-lay / teleport / discount abilities.
    pub abilities: fn() -> BTreeMap<String, Vec<Ab>>,
    /// Corporation-level :token abilities at setup: (corp, hexes, price).
    pub corp_token_abilities: &'static [(&'static str, &'static [&'static str], i32)],
    /// `blocks_hexes_consent` (entities.rb): (company, its corporation, hex).
    pub consent_blockers: &'static [(&'static str, &'static str, &'static str)],

    // -- trains -------------------------------------------------------------
    pub extra_trains: &'static [&'static str],
    pub extra_train_pullman: &'static str,
    pub extra_train_permanents: &'static [&'static str],
    pub local_trains: &'static [&'static str],
    pub e_train: &'static str,
    pub local_train_can_carry_mail: bool,
    pub trains: fn() -> Vec<TrainDef>,
    pub train_variant: fn(&str, &str) -> Option<(&'static str, u32, i32, Option<&'static str>)>,
    pub train_variants: fn(&str) -> &'static [&'static str],

    // -- corporations -------------------------------------------------------
    pub destination: fn(&str) -> Option<&'static str>,
    pub exchange_tokens: fn(&str) -> u32,

    // -- map ----------------------------------------------------------------
    pub london_hex: &'static str,
    pub english_channel_hex: &'static str,
    pub france_hex: &'static str,
    pub cardiff_hex: &'static str,
    pub merthyr_tydfil_pontypool_hex: &'static str,
    pub upgradeable_s_hex_name: &'static str,
    pub upgradeable_s_yellow_city_tile: &'static str,
    pub upgradeable_s_yellow_rotations: &'static [u8],
    pub double_hex: &'static [&'static str],
    pub track_plain: &'static [&'static str],
    pub track_town: &'static [&'static str],
    pub minor_green_upgrade: &'static [&'static str],
    /// Preprinted hex DSL by coordinate ("" for a plain hex) and the full
    /// table (coord, dsl, color).
    pub hex_dsl: fn(&str) -> Option<(&'static str, &'static str)>,
    pub hex_dsl_all: &'static [(&'static str, &'static str, &'static str)],
    pub tile_dsl: fn(&str) -> Option<(&'static str, &'static str)>,

    // -- game end -----------------------------------------------------------
    pub game_end_lock_first_trigger: bool,

    // -- shares ---------------------------------------------------------------
    /// MUST_SELL_IN_BLOCKS: one sale per corporation per turn (stock round)
    /// / per train purchase (emergency sales).
    pub must_sell_in_blocks: bool,
}

impl Family {
    /// `x == self.class::CONST` for a possibly-nil constant.
    #[inline]
    pub fn is(c: &str, x: &str) -> bool {
        !c.is_empty() && c == x
    }
}
