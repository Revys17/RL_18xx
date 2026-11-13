//! Title-generic typed-token state encoder (RL bridge milestone B1;
//! `docs/rl_bridge_design.md` §3, schema in `docs/rl_encoder.md`).
//!
//! The 1830 encoder (`crate::encoder`) is a flat vector whose layout is tied
//! to 1830's corporations, privates and map. This one encodes any title's
//! position as a bundle of typed tokens that a transformer reads:
//!
//! * **hex tokens** (one per map hex, in the title's action-layout hex order)
//!   with title-generic tile, track, stop, revenue, terrain/cost and border
//!   features; the tokens in a hex's city slots are POINTERS to the owning
//!   corporation's entity token (`hex_ptr`), not per-corporation one-hots;
//! * **entity tokens**: one global token, `MAX_SEATS` seat tokens (seat 0 =
//!   the player to move; absent seats are padding), the title's
//!   corporations (major / minor / national), companies (private /
//!   concession / minor proxy), train types and phases -- each entity type
//!   with a fixed, title-independent feature schema ([`schema`]);
//! * a sparse **relation list** (shares held, presidencies, companies
//!   owned, trains held, standing bids, sales this round) for attention
//!   biases;
//! * `meta` ints (title, round kind, step kind, acting entity, seats, ...).
//!
//! What is title-specific -- the entity lists, vocabularies and
//! normalizers -- is derived from the title data into an [`spec::RlSpec`]
//! per registered title. Vocabularies that the model embeds (titles, round
//! and step kinds, labels, terrains, phase statuses, market zones) are
//! global, append-only lists here, so ids never move when a title is added.
//!
//! Self-contained: nothing outside this module depends on it, and 1830's
//! frozen encoder, layout and model are untouched.

pub mod encode;
pub mod py;
pub mod schema;
pub mod spec;
#[cfg(test)]
mod tests;

pub use encode::TokenEncoding;
pub use spec::{rl_spec_for, RlSpec};

/// Bumped whenever a feature's meaning, order or normalization changes
/// (checkpoints record it; a mismatch must refuse to load).
pub const ENCODER_VERSION: u32 = 1;

/// Seat tokens per position (= the value head's width, `encoder::MAX_PLAYERS`).
pub const MAX_SEATS: usize = crate::encoder::MAX_PLAYERS;

/// Token slots per hex: every city's slots then its extra tokens, in city
/// order (1822's London: 6 cities x 1 slot; 1822MX's Mexico City likewise).
pub const S_MAX: usize = 8;

/// Entity pointer columns (`ent_ptr`): see [`schema::ENT_PTR_COLUMNS`].
pub const P_ENT: usize = 4;
/// Entity -> hex pointer columns (`ent_hex`): home, destination.
pub const P_HEX: usize = 2;
/// `meta` columns: see [`schema::META_COLUMNS`].
pub const META_LEN: usize = 8;

/// `hex_ptr` value for a token that belongs to no corporation entity (1867's
/// neutral placeholder tokens, 1822PNW's coal tokens without an owner).
pub const PTR_NEUTRAL: i32 = -2;
/// Pointer value for "nothing".
pub const PTR_NONE: i32 = -1;

/// Every feature is clipped to [-FEATURE_CLIP, FEATURE_CLIP] and NaN/inf
/// become 0 (normalizers keep almost everything in [-1, 2]; the clip only
/// guards against a title's data exceeding its normalizer).
pub const FEATURE_CLIP: f32 = 16.0;

// ---------------------------------------------------------------------------
// Entity types
// ---------------------------------------------------------------------------

pub const ENT_PAD: i32 = 0;
pub const ENT_GLOBAL: i32 = 1;
pub const ENT_PLAYER: i32 = 2;
pub const ENT_MAJOR: i32 = 3;
pub const ENT_MINOR: i32 = 4;
pub const ENT_NATIONAL: i32 = 5;
pub const ENT_PRIVATE: i32 = 6;
pub const ENT_CONCESSION: i32 = 7;
pub const ENT_MINOR_PROXY: i32 = 8;
pub const ENT_TRAIN: i32 = 9;
pub const ENT_PHASE: i32 = 10;
pub const ENTITY_TYPES: &[&str] = &[
    "pad",
    "global",
    "player",
    "major",
    "minor",
    "national",
    "private",
    "concession",
    "minor_proxy",
    "train_type",
    "phase",
];

// ---------------------------------------------------------------------------
// Relation kinds (`rel[:, 2]`; value in `rel_val`)
// ---------------------------------------------------------------------------

pub const REL_PAD: i32 = 0;
/// holder (seat or corporation) -> corporation: percent held / 100.
pub const REL_SHARES: i32 = 1;
/// seat -> corporation it presides: 1.
pub const REL_PRESIDENT: i32 = 2;
/// seat or corporation -> company it owns: 1.
pub const REL_OWNS: i32 = 3;
/// corporation -> train type: trains of that type held / 4.
pub const REL_TRAINS: i32 = 4;
/// seat -> company or corporation it has a standing bid on: price / starting cash.
pub const REL_BID: i32 = 5;
/// seat -> corporation it sold this stock round: 1 this turn, 0.5 earlier.
pub const REL_SOLD: i32 = 6;
pub const RELATION_KINDS: &[&str] = &["pad", "shares", "president", "owns_company", "trains", "bid", "sold"];

// ---------------------------------------------------------------------------
// Global vocabularies (append-only: ids are embedded by trained models)
// ---------------------------------------------------------------------------

/// Title ids (`meta[0]`): index in this list + 1; 0 = unknown.
pub const TITLES: &[&str] = &["1830", "1867", "1822", "1822PNW", "1822MX", "1822CA"];

/// Round kinds (`meta[1]`). 1822's merger round (PNW) shares 1867's id.
pub const ROUND_KINDS: &[&str] = &["none", "Auction", "Stock", "Operating", "Merger", "Choices"];

/// Step kinds (`meta[2]`): the shared engine's `StepKind` names, the 1822
/// family's `g1822::Step` names (a name both use is one id), plus
/// `BuySellParSharesViaBid` (1867's in-stock-round minor auction).
pub const STEP_KINDS: &[&str] = &[
    "none",
    "WaterfallAuction",
    "SingleItemAuction",
    "CompanyPendingPar",
    "BuySellParShares",
    "BuySellParSharesViaBid",
    "HomeToken",
    "MajorTrainless",
    "Bankrupt",
    "Exchange",
    "SpecialTrack",
    "SpecialToken",
    "BuyCompany",
    "BuyCompanyPreloan",
    "LoanOperations",
    "RedeemShares",
    "Track",
    "Token",
    "Route",
    "Dividend",
    "DiscardTrain",
    "BuyTrain",
    "Merge",
    "PostMergerShares",
    "ReduceTokens",
    "Choose",
    "PendingToken",
    "FirstTurnHousekeeping",
    "AcquireCompany",
    "SpecialChoose",
    "DestinationToken",
    "MinorAcquisition",
    "IssueShares",
    "Assign",
    "CashOutNdem",
    "AuctionNdemTokens",
    "AssignSawmill",
    "AcquisitionTrack",
];

/// Tile / hex labels (`hex_ids[..., 0]`): 0 = no label, 1 = a label not in
/// this list, then index + 2. Pinned by `vocab_covers_every_title`.
pub const LABELS: &[&str] = &[
    // 1830
    "B", "NY", "OO",
    // 1867
    "M", "O", "T", "Y",
    // 1822 family
    "BM", "C", "EC", "L", "S", "Por", "Sea", "MC", "A", "CP", "Q", "W", "YP",
];

/// Terrain kinds of tile upgrade costs (hex features `terrain_*`).
pub const TERRAINS: &[&str] = &["mountain", "water", "hill", "swamp", "forest", "river"];

/// Phase status flags (phase features `st_*`); unknown flags -> `st_other`.
pub const PHASE_STATUSES: &[&str] = &[
    "can_buy_companies",
    "export_train",
    "can_buy_trains",
    "can_acquire_minor_bidbox",
    "can_par",
    "full_capitalisation",
    "minors_green_upgrade",
    "minor_float_phase1",
    "minor_float_phase2",
    "minor_float_phase3on",
    "can_convert_concessions",
];

/// Stock-market cell types (`SharePrice.types` strings, `MarketZone::type_str`)
/// flagged per corporation.
pub const MARKET_ZONES: &[&str] = &[
    "par",
    "par_1",
    "par_2",
    "convert_range",
    "max_price",
    "no_cert_limit",
    "unlimited",
    "multiple_buy",
    "endgame",
];

/// Tile colors, one-hot in hex and tile features.
pub const TILE_COLORS: &[&str] = &["white", "yellow", "green", "brown", "gray", "red"];

/// Index of `s` in `list`, if any.
pub(crate) fn vocab_index(list: &[&str], s: &str) -> Option<usize> {
    list.iter().position(|x| *x == s)
}

/// `TITLES` id of a title name (0 = unknown).
pub fn title_id(name: &str) -> i32 {
    vocab_index(TITLES, name).map_or(0, |i| i as i32 + 1)
}

/// `LABELS` id: 0 none, 1 unknown, else index + 2.
pub fn label_id(label: Option<&str>) -> i32 {
    match label {
        None | Some("") => 0,
        Some(l) => vocab_index(LABELS, l).map_or(1, |i| i as i32 + 2),
    }
}

/// `STEP_KINDS` id (0 = none / unknown).
pub fn step_kind_id(name: &str) -> i32 {
    vocab_index(STEP_KINDS, name).map_or(0, |i| i as i32)
}

/// `ROUND_KINDS` id (0 = none / unknown).
pub fn round_kind_id(name: &str) -> i32 {
    vocab_index(ROUND_KINDS, name).map_or(0, |i| i as i32)
}
