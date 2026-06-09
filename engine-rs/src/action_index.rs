//! Slot layout + index <-> action dict for the Rust MCTS (Phase 4a).
//!
//! Mirrors the slot layout defined by Python `ActionMapper.init_actions` in
//! `rl18xx/agent/alphazero/action_mapper.py`. The two public functions are:
//!
//! - [`legal_action_to_index`]: pure-Rust encode (mirrors Python's
//!   `index_for_factored`).
//!
//! The inverse (index -> action) is native Rust in `decode.rs`
//! (`BaseGame::decode_index`) — no Python round-trip.

use std::collections::HashMap;
use std::sync::OnceLock;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use crate::factored::LegalAction;
use crate::game::BaseGame;

pub const POLICY_SIZE: u32 = 26537;

// ----------------------------------------------------------------------------
// Static offset tables — mirror ActionMapper.init_actions exactly.
// ----------------------------------------------------------------------------

pub struct SlotLayout {
    pub company_offsets: Vec<&'static str>,
    pub corporation_offsets: Vec<&'static str>,
    pub par_price_offsets: Vec<i32>,
    pub share_location_offsets: Vec<&'static str>,
    pub train_price_offsets: Vec<&'static str>,
    pub dividend_offsets: Vec<&'static str>,
    pub buy_company_price_offsets: Vec<&'static str>,
    pub train_type_offsets: Vec<&'static str>,
    pub hex_offsets: Vec<&'static str>,
    pub tile_offsets: Vec<&'static str>,
    pub city_offsets: HashMap<(&'static str, usize), u32>,
    pub city_count: HashMap<&'static str, usize>,
    /// Per-company sub-blocks of the CompanyLayTile slot range, derived from
    /// the title's Teleport / TileLay ability data.
    pub company_lay_tile_blocks: Vec<CompanyLayBlock>,
    /// Per-company offsets within the CompanyPlaceToken slot range (one slot
    /// per teleport company).
    pub company_place_token_offsets: Vec<(&'static str, u32)>,
    pub action_offsets: HashMap<&'static str, u32>,
}

/// One company's sub-block within the CompanyLayTile slot range:
/// `tiles.len() * 6` rotation slots starting at `offset` (relative to the
/// CompanyLayTile base), addressed as `offset + tile_pos * 6 + rotation`.
pub struct CompanyLayBlock {
    pub sym: &'static str,
    pub hexes: &'static [&'static str],
    pub tiles: &'static [&'static str],
    pub offset: u32,
    /// True for Teleport-derived blocks (DH), false for TileLay (CS). Only
    /// affects how unmatched tiles are handled during encode (mirroring the
    /// Python ActionMapper's fall-through vs error behavior).
    pub teleport: bool,
}

static LAYOUT: OnceLock<SlotLayout> = OnceLock::new();

pub fn layout() -> &'static SlotLayout {
    LAYOUT.get_or_init(build_layout)
}

fn build_layout() -> SlotLayout {
    let company_offsets = vec!["SV", "CS", "DH", "MH", "CA", "BO"];
    let corporation_offsets = vec!["PRR", "NYC", "CPR", "B&O", "C&O", "ERIE", "NYNH", "B&M"];
    let par_price_offsets: Vec<i32> = vec![67, 71, 76, 82, 90, 100];
    let share_location_offsets = vec!["ipo", "market"];
    let train_price_offsets = vec![
        "1", "20", "50", "100", "200", "300", "400", "500", "600", "700", "800", "900",
        "all-but-one", "all",
    ];
    let dividend_offsets = vec!["payout", "withhold"];
    let buy_company_price_offsets = vec!["min", "max"];
    let train_type_offsets = vec!["2", "3", "4", "5", "6", "D"];

    let hex_offsets = vec![
        "F2", "I1", "J2", "A9", "A11", "K13", "B24", "D2", "F6", "E9", "H12", "D14", "C15", "K15",
        "A17", "A19", "I19", "F24", "D24", "F4", "J14", "F22", "E7", "F8", "C11", "C13", "D12",
        "B16", "C17", "B20", "D4", "F10", "I13", "D18", "B12", "B14", "B22", "C7", "C9", "C23",
        "D8", "D16", "D20", "E3", "E13", "E15", "F12", "F14", "F18", "G3", "G5", "G9", "G11",
        "H2", "H6", "H8", "H14", "I3", "I5", "I7", "I9", "J4", "J6", "J8", "G15", "C21", "D22",
        "E17", "E21", "G13", "I11", "J10", "J12", "E19", "H4", "B10", "H10", "H16", "F16", "G7",
        "G17", "F20", "D6", "I17", "B18", "C19", "E5", "D10", "E11", "H18", "I15", "G19", "E23",
    ];

    let tile_offsets = vec![
        "42", "4", "16", "70", "23", "7", "18", "24", "3", "55", "61", "54", "9", "41", "26",
        "68", "57", "45", "1", "56", "44", "62", "63", "64", "40", "66", "20", "27", "39", "19",
        "59", "25", "46", "28", "65", "43", "2", "53", "58", "14", "47", "8", "29", "69", "15",
        "67",
    ];

    let city_count_entries: Vec<(&'static str, usize)> = vec![
        ("D2", 1), ("F6", 1), ("H12", 1), ("D14", 1), ("K15", 1), ("A19", 1),
        ("F4", 1), ("J14", 1), ("F22", 1), ("B16", 1), ("E19", 1), ("H4", 1),
        ("B10", 1), ("H10", 1), ("H16", 1), ("F16", 1), ("E5", 2), ("D10", 2),
        ("E11", 2), ("H18", 2), ("I15", 1), ("G19", 2), ("E23", 1),
    ];
    let city_count: HashMap<&'static str, usize> = city_count_entries.iter().copied().collect();

    let city_offset_entries: Vec<((&'static str, usize), u32)> = vec![
        (("D2", 0), 0), (("F6", 0), 1), (("H12", 0), 2), (("D14", 0), 3),
        (("K15", 0), 4), (("A19", 0), 5), (("F4", 0), 6), (("J14", 0), 7),
        (("F22", 0), 8), (("B16", 0), 9), (("E19", 0), 10), (("H4", 0), 11),
        (("B10", 0), 12), (("H10", 0), 13), (("H16", 0), 14), (("F16", 0), 15),
        (("E5", 0), 16), (("E5", 1), 17), (("D10", 0), 18), (("D10", 1), 19),
        (("E11", 0), 20), (("E11", 1), 21), (("H18", 0), 22), (("H18", 1), 23),
        (("I15", 0), 24), (("G19", 0), 25), (("G19", 1), 26), (("E23", 0), 27),
    ];
    let city_offsets: HashMap<(&'static str, usize), u32> = city_offset_entries.into_iter().collect();

    // Company special-lay sub-blocks, derived from the ability data. Mirrors
    // ActionMapper.init_actions: teleport lays first (DH: 1 tile x 6
    // rotations), then bonus tile lays (CS: 3 tiles x 6 rotations), each in
    // title company order.
    let mut company_lay_tile_blocks: Vec<CompanyLayBlock> = Vec::new();
    let mut company_lay_n: u32 = 0;
    for sym in &company_offsets {
        if let Some((hexes, tiles)) = crate::abilities::teleport(sym) {
            company_lay_tile_blocks.push(CompanyLayBlock {
                sym,
                hexes,
                tiles,
                offset: company_lay_n,
                teleport: true,
            });
            company_lay_n += (tiles.len() * 6) as u32;
        }
    }
    for sym in &company_offsets {
        if let Some((hexes, tiles, _, _)) = crate::abilities::tile_lay(sym) {
            company_lay_tile_blocks.push(CompanyLayBlock {
                sym,
                hexes,
                tiles,
                offset: company_lay_n,
                teleport: false,
            });
            company_lay_n += (tiles.len() * 6) as u32;
        }
    }

    // One CompanyPlaceToken slot per teleport company (DH on F16).
    let mut company_place_token_offsets: Vec<(&'static str, u32)> = Vec::new();
    for sym in &company_offsets {
        if crate::abilities::teleport(sym).is_some() {
            company_place_token_offsets.push((sym, company_place_token_offsets.len() as u32));
        }
    }

    // ----- Compute action_offsets and total slot count, mirroring init_actions -----
    let mut action_offsets: HashMap<&'static str, u32> = HashMap::new();
    let mut idx: u32 = 0;

    // Pass / CompanyPass (single shared slot)
    action_offsets.insert("Pass", idx);
    action_offsets.insert("CompanyPass", idx);
    idx += 1;

    action_offsets.insert("Bid", idx);
    idx += company_offsets.len() as u32; // 6

    action_offsets.insert("Par", idx);
    idx += (corporation_offsets.len() * par_price_offsets.len()) as u32; // 8 * 6

    action_offsets.insert("BuyShares", idx);
    idx += (corporation_offsets.len() * share_location_offsets.len()) as u32; // 8 * 2

    action_offsets.insert("SellShares", idx);
    idx += (corporation_offsets.len() * 5) as u32; // 8 * 5

    action_offsets.insert("PlaceToken", idx);
    // PlaceToken: sum of city_count over hexes that appear in city_count
    let mut place_token_n: u32 = 0;
    for h in &hex_offsets {
        if let Some(n) = city_count.get(h) {
            place_token_n += *n as u32;
        }
    }
    idx += place_token_n;

    action_offsets.insert("LayTile", idx);
    idx += (hex_offsets.len() * tile_offsets.len() * 6) as u32;

    action_offsets.insert("BuyTrain", idx);
    // 1 (depot) + train_type (market discarded) + corp*train_type*price (cross-corp)
    let buy_train_n =
        1u32
            + train_type_offsets.len() as u32
            + (corporation_offsets.len() * train_type_offsets.len() * train_price_offsets.len())
                as u32;
    idx += buy_train_n;

    action_offsets.insert("DiscardTrain", idx);
    idx += train_type_offsets.len() as u32;

    action_offsets.insert("Dividend", idx);
    idx += dividend_offsets.len() as u32;

    action_offsets.insert("BuyCompany", idx);
    idx += (company_offsets.len() * buy_company_price_offsets.len()) as u32;

    action_offsets.insert("Bankrupt", idx);
    idx += 1;

    action_offsets.insert("RunRoutes", idx);
    idx += 1;

    // CompanyBuyShares: 2 slots (ipo, market for MH→NYC).
    action_offsets.insert("CompanyBuyShares", idx);
    idx += share_location_offsets.len() as u32;

    // CompanyLayTile: the ability-derived sub-blocks (1830: 6 DH teleport
    // slots + 3 CS tiles * 6 rotations = 24).
    action_offsets.insert("CompanyLayTile", idx);
    idx += company_lay_n;

    // CompanyPlaceToken: one slot per teleport company (1830: 1, DH F16).
    action_offsets.insert("CompanyPlaceToken", idx);
    idx += company_place_token_offsets.len() as u32;

    // D-train depot disambiguation slots (appended last for backward-compat).
    action_offsets.insert("BuyTrainDFull", idx);
    idx += 1;
    action_offsets.insert("BuyTrainDTradeIn", idx);
    idx += 1;

    debug_assert_eq!(
        idx, POLICY_SIZE,
        "action_index slot count diverges from Python ActionMapper"
    );

    SlotLayout {
        company_offsets,
        corporation_offsets,
        par_price_offsets,
        share_location_offsets,
        train_price_offsets,
        dividend_offsets,
        buy_company_price_offsets,
        train_type_offsets,
        hex_offsets,
        tile_offsets,
        city_offsets,
        city_count,
        company_lay_tile_blocks,
        company_place_token_offsets,
        action_offsets,
    }
}

// ----------------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------------

fn pos<T: PartialEq>(v: &[T], needle: &T) -> Option<usize> {
    v.iter().position(|x| x == needle)
}

fn s_to_json_str(v: &serde_json::Value) -> Option<&str> {
    v.as_str()
}

// ----------------------------------------------------------------------------
// Encode: legal action -> flat policy index
// ----------------------------------------------------------------------------

/// Rust port of Python `ActionMapper.index_for_factored`.
///
/// Returns `None` for unrecognized inputs (instead of panicking).
pub fn legal_action_to_index(la: &LegalAction) -> Option<u32> {
    let lo = layout();
    let t = la.action_type.as_str();

    if t == "Pass" {
        return Some(lo.action_offsets["Pass"]);
    }
    if t == "Bankrupt" {
        return Some(lo.action_offsets["Bankrupt"]);
    }
    if t == "RunRoutes" {
        return Some(lo.action_offsets["RunRoutes"]);
    }

    if t == "Bid" {
        let company_sym = la.entity.get("private").and_then(s_to_json_str)?;
        let ci = pos(&lo.company_offsets, &company_sym)?;
        return Some(lo.action_offsets["Bid"] + ci as u32);
    }

    if t == "Par" {
        let corp_sym = la.entity.get("corp").and_then(s_to_json_str)?;
        let price = la.params.get("par_price").and_then(|v| v.as_i64())? as i32;
        let corp_i = pos(&lo.corporation_offsets, &corp_sym)?;
        let price_i = pos(&lo.par_price_offsets, &price)?;
        return Some(
            lo.action_offsets["Par"]
                + (corp_i * lo.par_price_offsets.len()) as u32
                + price_i as u32,
        );
    }

    if t == "BuyShares" {
        let corp_sym = la.entity.get("corp").and_then(s_to_json_str)?;
        let source = la.params.get("source").and_then(s_to_json_str)?;
        // CompanyBuyShares (MH → NYC) routes through a different block.
        if la.entity.get("private").and_then(s_to_json_str) == Some("MH") && corp_sym == "NYC" {
            let li = pos(&lo.share_location_offsets, &source)?;
            return Some(lo.action_offsets["CompanyBuyShares"] + li as u32);
        }
        let ci = pos(&lo.corporation_offsets, &corp_sym)?;
        let li = pos(&lo.share_location_offsets, &source)?;
        return Some(
            lo.action_offsets["BuyShares"]
                + (ci * lo.share_location_offsets.len()) as u32
                + li as u32,
        );
    }

    if t == "CompanyBuyShares" {
        let source = la.params.get("source").and_then(s_to_json_str)?;
        let li = pos(&lo.share_location_offsets, &source)?;
        return Some(lo.action_offsets["CompanyBuyShares"] + li as u32);
    }

    if t == "SellShares" {
        let corp_sym = la.entity.get("corp").and_then(s_to_json_str)?;
        let count = la.params.get("count").and_then(|v| v.as_i64())? as i32;
        let ci = pos(&lo.corporation_offsets, &corp_sym)?;
        return Some(lo.action_offsets["SellShares"] + (ci * 5) as u32 + (count as u32) - 1);
    }

    if t == "PlaceToken" {
        // A teleport company's token routes to its CompanyPlaceToken slot.
        if let Some(private) = la.entity.get("private").and_then(s_to_json_str) {
            if let Some((_, off)) = lo
                .company_place_token_offsets
                .iter()
                .find(|(sym, _)| *sym == private)
            {
                return Some(lo.action_offsets["CompanyPlaceToken"] + off);
            }
        }
        let hex_id = la.params.get("hex").and_then(s_to_json_str)?;
        let city_idx = la
            .params
            .get("city")
            .and_then(|v| v.as_i64())
            .map(|n| n as usize)
            .unwrap_or(0);
        let key: (&'static str, usize) = {
            // Need a 'static-keyed lookup; find the entry by string equality.
            let mut found: Option<((&'static str, usize), u32)> = None;
            for (&(h, c), &off) in &lo.city_offsets {
                if h == hex_id && c == city_idx {
                    found = Some(((h, c), off));
                    break;
                }
            }
            match found {
                Some((k, _)) => k,
                None => return None,
            }
        };
        let off = *lo.city_offsets.get(&key)?;
        return Some(lo.action_offsets["PlaceToken"] + off);
    }

    if t == "LayTile" {
        let private = la.entity.get("private").and_then(s_to_json_str);
        let tile_name = la.params.get("tile").and_then(s_to_json_str)?;
        let hex_id = la.params.get("hex").and_then(s_to_json_str)?;
        let rotation = la.params.get("rotation").and_then(|v| v.as_i64())? as u32;
        // A company special lay (teleport / bonus tile_lay) routes to the
        // company's CompanyLayTile sub-block. Mirrors Python
        // `index_for_factored`: a teleport lay with an unmatched tile falls
        // through to the regular hex/tile encode, while a bonus tile_lay with
        // an unmatched tile is an error (None).
        if let Some(private) = private {
            if let Some(block) = lo
                .company_lay_tile_blocks
                .iter()
                .find(|b| b.sym == private && b.hexes.contains(&hex_id))
            {
                match pos(block.tiles, &tile_name) {
                    Some(ti) => {
                        return Some(
                            lo.action_offsets["CompanyLayTile"]
                                + block.offset
                                + (ti * 6) as u32
                                + rotation,
                        )
                    }
                    None if block.teleport => {} // fall through (Python parity)
                    None => return None,
                }
            }
        }
        let hi = pos(&lo.hex_offsets, &hex_id)?;
        let ti = pos(&lo.tile_offsets, &tile_name)?;
        return Some(
            lo.action_offsets["LayTile"]
                + (hi * lo.tile_offsets.len() * 6) as u32
                + (ti * 6) as u32
                + rotation,
        );
    }

    if t == "BuyTrain" {
        return index_for_factored_buy_train(la);
    }

    if t == "DiscardTrain" {
        let train = la.params.get("train").and_then(s_to_json_str)?;
        let ti = pos(&lo.train_type_offsets, &train)?;
        return Some(lo.action_offsets["DiscardTrain"] + ti as u32);
    }

    if t == "Dividend" {
        let kind = la.params.get("kind").and_then(s_to_json_str)?;
        let di = pos(&lo.dividend_offsets, &kind)?;
        return Some(lo.action_offsets["Dividend"] + di as u32);
    }

    if t == "BuyCompany" {
        let company_sym = la.entity.get("private").and_then(s_to_json_str)?;
        let ci = pos(&lo.company_offsets, &company_sym)?;
        return Some(
            lo.action_offsets["BuyCompany"] + (ci * lo.buy_company_price_offsets.len()) as u32,
        );
    }

    None
}

fn index_for_factored_buy_train(la: &LegalAction) -> Option<u32> {
    let lo = layout();
    let offset = lo.action_offsets["BuyTrain"];
    let source = la.entity.get("source").and_then(s_to_json_str)?;
    let train_name = la.entity.get("train").and_then(s_to_json_str)?;

    if source == "discard" {
        let ti = pos(&lo.train_type_offsets, &train_name)?;
        return Some(offset + 1 + ti as u32);
    }

    if source == "depot" {
        // Mirror Python `ActionMapper._index_for_factored_buy_train` exactly:
        // a depot train whose *name* is in the discarded (face-value) pool routes
        // to its per-train-type discard slot — and this discard check runs BEFORE
        // the D-train/exchange disambiguation, so even a trade-in D lands in the
        // discard slot when a discarded D exists. `depot_discarded` is stamped by
        // the factored enumeration (`factored.rs`), which has the live depot in
        // hand; the descriptor itself still reads `source="depot"` (matching
        // Python), so this is the only place the upcoming-vs-discarded split is
        // resolved.
        if la.depot_discarded {
            let ti = pos(&lo.train_type_offsets, &train_name)?;
            return Some(offset + 1 + ti as u32);
        }
        if train_name == "D" {
            if la.entity.get("exchange").is_some() {
                return Some(lo.action_offsets["BuyTrainDTradeIn"]);
            }
            return Some(lo.action_offsets["BuyTrainDFull"]);
        }
        return Some(offset);
    }

    // Cross-corp: source is the seller corp's sym.
    let ci = pos(&lo.corporation_offsets, &source)?;
    let ti = pos(&lo.train_type_offsets, &train_name)?;
    let first_price = 0u32;
    Some(
        offset
            + 1
            + lo.train_type_offsets.len() as u32
            + (ci * lo.train_price_offsets.len() * lo.train_type_offsets.len()) as u32
            + (ti * lo.train_price_offsets.len()) as u32
            + first_price,
    )
}

// ----------------------------------------------------------------------------
// PyO3 surface — exposed for tests / introspection
// ----------------------------------------------------------------------------

/// Return the layout's `action_offsets` map (for Python-side debugging).
#[pyfunction]
pub fn action_offsets_py(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let lo = layout();
    let d = PyDict::new(py);
    for (k, v) in &lo.action_offsets {
        d.set_item(*k, *v)?;
    }
    Ok(d)
}

/// Total policy size (matches Python `ActionMapper().action_encoding_size`).
#[pyfunction]
pub fn policy_size_py() -> u32 {
    POLICY_SIZE
}

/// Compatibility shim so external callers can resolve a Python-side
/// LegalAction-style dict to a flat index without crossing the LegalAction
/// type. Accepts a (type, entity_dict, params_dict) tuple.
#[pyfunction]
pub fn legal_action_to_index_py(t: &str, entity: &Bound<'_, PyDict>, params: &Bound<'_, PyDict>) -> PyResult<Option<u32>> {
    let mut la = LegalAction {
        action_type: t.to_string(),
        entity: HashMap::new(),
        params: HashMap::new(),
        price_range: None,
        depot_discarded: false,
    };
    for (k, v) in entity.iter() {
        let key: String = k.extract()?;
        let val = py_to_json_value(&v)?;
        la.entity.insert(key, val);
    }
    for (k, v) in params.iter() {
        let key: String = k.extract()?;
        let val = py_to_json_value(&v)?;
        la.params.insert(key, val);
    }
    // This stateless shim has no depot in hand, so the upcoming-vs-discarded
    // split can only be resolved if the caller passes an explicit
    // `_depot_discarded` hint in the entity dict (the live Rust enumeration sets
    // the struct field directly and never emits this key). Defaults to the
    // legacy "fresh depot" behavior when absent.
    if let Some(v) = la.entity.get("_depot_discarded") {
        la.depot_discarded = v.as_bool().unwrap_or(false);
    }
    Ok(legal_action_to_index(&la))
}

fn py_to_json_value(v: &Bound<'_, PyAny>) -> PyResult<serde_json::Value> {
    if let Ok(s) = v.extract::<String>() {
        return Ok(serde_json::Value::String(s));
    }
    if let Ok(b) = v.extract::<bool>() {
        return Ok(serde_json::Value::Bool(b));
    }
    if let Ok(i) = v.extract::<i64>() {
        return Ok(serde_json::Value::Number(i.into()));
    }
    if let Ok(f) = v.extract::<f64>() {
        if let Some(n) = serde_json::Number::from_f64(f) {
            return Ok(serde_json::Value::Number(n));
        }
    }
    Ok(serde_json::Value::Null)
}

// `BaseGame` and `PyTuple` aren't directly used by this module's public API,
// but keep the imports so future native decode work (Phase 4c) can wire in
// without re-shuffling the prelude.
#[allow(dead_code)]
fn _force_unused(_g: Option<&BaseGame>, _t: Option<&Bound<'_, PyTuple>>) {}

// ----------------------------------------------------------------------------
// Layout-equivalence oracle
// ----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// The 1830 flat action layout is frozen: Python `ActionMapper` and
    /// already-trained checkpoints depend on these exact offsets. Any
    /// refactor of `build_layout` (e.g. deriving the company sub-blocks from
    /// ability data) must keep this test green — it pins every block offset
    /// and the total POLICY_SIZE to the historical constants.
    #[test]
    fn layout_matches_frozen_1830_constants() {
        let lo = layout();
        let expected: &[(&str, u32)] = &[
            ("Pass", 0),
            ("CompanyPass", 0),
            ("Bid", 1),
            ("Par", 7),
            ("BuyShares", 55),
            ("SellShares", 71),
            ("PlaceToken", 111),
            ("LayTile", 139),
            ("BuyTrain", 25807),
            ("DiscardTrain", 26486),
            ("Dividend", 26492),
            ("BuyCompany", 26494),
            ("Bankrupt", 26506),
            ("RunRoutes", 26507),
            ("CompanyBuyShares", 26508),
            ("CompanyLayTile", 26510),
            ("CompanyPlaceToken", 26534),
            ("BuyTrainDFull", 26535),
            ("BuyTrainDTradeIn", 26536),
        ];
        for (k, v) in expected {
            assert_eq!(lo.action_offsets[k], *v, "offset for {}", k);
        }
        assert_eq!(POLICY_SIZE, 26537);

        // Company-ability sub-block sizes (1830):
        // CompanyBuyShares: 2 (MH -> NYC from ipo|market)
        assert_eq!(
            lo.action_offsets["CompanyLayTile"] - lo.action_offsets["CompanyBuyShares"],
            2
        );
        // CompanyLayTile: 6 (DH teleport, 1 tile x 6 rotations)
        //               + 18 (CS tile_lay, 3 tiles x 6 rotations)
        assert_eq!(
            lo.action_offsets["CompanyPlaceToken"] - lo.action_offsets["CompanyLayTile"],
            24
        );
        // CompanyPlaceToken: 1 (DH teleport token)
        assert_eq!(
            lo.action_offsets["BuyTrainDFull"] - lo.action_offsets["CompanyPlaceToken"],
            1
        );

        // The ability-derived company sub-blocks must reproduce the frozen
        // 1830 ordering exactly: DH teleport (offset 0, tile 57 on F16),
        // then CS bonus lays (offset 6, tiles 3/4/58 on B20).
        let blocks: Vec<(&str, &[&str], &[&str], u32, bool)> = lo
            .company_lay_tile_blocks
            .iter()
            .map(|b| (b.sym, b.hexes, b.tiles, b.offset, b.teleport))
            .collect();
        assert_eq!(
            blocks,
            vec![
                ("DH", &["F16"][..], &["57"][..], 0, true),
                ("CS", &["B20"][..], &["3", "4", "58"][..], 6, false),
            ]
        );
        assert_eq!(lo.company_place_token_offsets, vec![("DH", 0)]);
    }
}
