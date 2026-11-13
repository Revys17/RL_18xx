//! PyO3 surface of the token encoder: a batched encode into stacked numpy
//! arrays (one call per batch, as 1830's `encode_games`), the schema and the
//! per-title spec.

use std::collections::HashMap;

use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::schema::*;
use super::*;
use crate::game::BaseGame;

/// Stack encodings into padded batch buffers.
pub struct Batch {
    pub b: usize,
    pub h: usize,
    pub e: usize,
    pub r: usize,
    /// Distinct hex blocks (= b without de-duplication).
    pub m: usize,
    pub hex_feat: Vec<f32>,
    pub hex_ptr: Vec<i32>,
    pub hex_res: Vec<i32>,
    pub hex_slot_city: Vec<i32>,
    pub hex_ids: Vec<i32>,
    pub hex_mask: Vec<bool>,
    pub map_index: Option<Vec<i64>>,
    pub ent_type: Vec<i32>,
    pub ent_feat: Vec<f32>,
    pub ent_ptr: Vec<i32>,
    pub ent_hex: Vec<i32>,
    pub ent_mask: Vec<bool>,
    pub rel: Vec<i32>,
    pub rel_val: Vec<f32>,
    pub rel_mask: Vec<bool>,
    pub meta: Vec<i32>,
}

impl Batch {
    pub fn stack(encs: &[TokenEncoding], dedupe_maps: bool) -> Batch {
        let b = encs.len();
        let h = encs.iter().map(|x| x.num_hexes).max().unwrap_or(0);
        let e = encs.iter().map(|x| x.num_entities).max().unwrap_or(0);
        let r = encs.iter().map(|x| x.num_relations()).max().unwrap_or(0);

        // Hex blocks: one per position, or each distinct one once.
        let mut blocks: Vec<usize> = Vec::with_capacity(b); // position -> source encoding of its block
        let mut map_index: Vec<i64> = Vec::with_capacity(b);
        if dedupe_maps {
            let mut by_hash: HashMap<u64, Vec<usize>> = HashMap::new();
            for (i, enc) in encs.iter().enumerate() {
                let hh = enc.hex_hash();
                let slots = by_hash.entry(hh).or_default();
                match slots.iter().find(|&&k| encs[blocks[k]].same_hexes(enc)) {
                    Some(&k) => map_index.push(k as i64),
                    None => {
                        slots.push(blocks.len());
                        map_index.push(blocks.len() as i64);
                        blocks.push(i);
                    }
                }
            }
        } else {
            blocks.extend(0..b);
        }
        let m = blocks.len();

        let mut out = Batch {
            b,
            h,
            e,
            r,
            m,
            hex_feat: vec![0.0; m * h * F_HEX],
            hex_ptr: vec![PTR_NONE; m * h * S_MAX],
            hex_res: vec![PTR_NONE; m * h * S_MAX],
            hex_slot_city: vec![PTR_NONE; m * h * S_MAX],
            hex_ids: vec![0; m * h * 2],
            hex_mask: vec![false; m * h],
            map_index: dedupe_maps.then_some(map_index),
            ent_type: vec![ENT_PAD; b * e],
            ent_feat: vec![0.0; b * e * F_ENT],
            ent_ptr: vec![PTR_NONE; b * e * P_ENT],
            ent_hex: vec![PTR_NONE; b * e * P_HEX],
            ent_mask: vec![false; b * e],
            rel: vec![0; b * r * 3],
            rel_val: vec![0.0; b * r],
            rel_mask: vec![false; b * r],
            meta: vec![0; b * META_LEN],
        };
        for (k, &src) in blocks.iter().enumerate() {
            let x = &encs[src];
            let n = x.num_hexes;
            out.hex_feat[k * h * F_HEX..(k * h + n) * F_HEX].copy_from_slice(&x.hex_feat);
            out.hex_ptr[k * h * S_MAX..(k * h + n) * S_MAX].copy_from_slice(&x.hex_ptr);
            out.hex_res[k * h * S_MAX..(k * h + n) * S_MAX].copy_from_slice(&x.hex_res);
            out.hex_slot_city[k * h * S_MAX..(k * h + n) * S_MAX].copy_from_slice(&x.hex_slot_city);
            out.hex_ids[k * h * 2..(k * h + n) * 2].copy_from_slice(&x.hex_ids);
            out.hex_mask[k * h..k * h + n].iter_mut().for_each(|v| *v = true);
        }
        for (i, x) in encs.iter().enumerate() {
            let n = x.num_entities;
            out.ent_type[i * e..i * e + n].copy_from_slice(&x.ent_type);
            out.ent_feat[i * e * F_ENT..(i * e + n) * F_ENT].copy_from_slice(&x.ent_feat);
            out.ent_ptr[i * e * P_ENT..(i * e + n) * P_ENT].copy_from_slice(&x.ent_ptr);
            out.ent_hex[i * e * P_HEX..(i * e + n) * P_HEX].copy_from_slice(&x.ent_hex);
            for j in 0..n {
                out.ent_mask[i * e + j] = x.ent_type[j] != ENT_PAD;
            }
            let nr = x.num_relations();
            out.rel[i * r * 3..(i * r + nr) * 3].copy_from_slice(&x.rel);
            out.rel_val[i * r..i * r + nr].copy_from_slice(&x.rel_val);
            out.rel_mask[i * r..i * r + nr].iter_mut().for_each(|v| *v = true);
            out.meta[i * META_LEN..(i + 1) * META_LEN].copy_from_slice(&x.meta);
        }
        out
    }

    pub fn into_dict<'py>(self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let (b, h, e, r, m) = (self.b, self.h, self.e, self.r, self.m);
        let d = PyDict::new(py);
        d.set_item("hex_feat", PyArray1::from_vec(py, self.hex_feat).reshape([m, h, F_HEX])?)?;
        d.set_item("hex_ptr", PyArray1::from_vec(py, self.hex_ptr).reshape([m, h, S_MAX])?)?;
        d.set_item("hex_res", PyArray1::from_vec(py, self.hex_res).reshape([m, h, S_MAX])?)?;
        d.set_item("hex_slot_city", PyArray1::from_vec(py, self.hex_slot_city).reshape([m, h, S_MAX])?)?;
        d.set_item("hex_ids", PyArray1::from_vec(py, self.hex_ids).reshape([m, h, 2])?)?;
        d.set_item("hex_mask", PyArray1::from_vec(py, self.hex_mask).reshape([m, h])?)?;
        match self.map_index {
            Some(mi) => d.set_item("map_index", PyArray1::from_vec(py, mi))?,
            None => d.set_item("map_index", py.None())?,
        }
        d.set_item("ent_type", PyArray1::from_vec(py, self.ent_type).reshape([b, e])?)?;
        d.set_item("ent_feat", PyArray1::from_vec(py, self.ent_feat).reshape([b, e, F_ENT])?)?;
        d.set_item("ent_ptr", PyArray1::from_vec(py, self.ent_ptr).reshape([b, e, P_ENT])?)?;
        d.set_item("ent_hex", PyArray1::from_vec(py, self.ent_hex).reshape([b, e, P_HEX])?)?;
        d.set_item("ent_mask", PyArray1::from_vec(py, self.ent_mask).reshape([b, e])?)?;
        d.set_item("rel", PyArray1::from_vec(py, self.rel).reshape([b, r, 3])?)?;
        d.set_item("rel_val", PyArray1::from_vec(py, self.rel_val).reshape([b, r])?)?;
        d.set_item("rel_mask", PyArray1::from_vec(py, self.rel_mask).reshape([b, r])?)?;
        d.set_item("meta", PyArray1::from_vec(py, self.meta).reshape([b, META_LEN])?)?;
        Ok(d)
    }
}

/// Encode a batch of positions (any titles and player counts) as stacked,
/// padded numpy arrays: hex arrays `[B or M, H, ...]` (with `dedupe_maps`
/// each distinct hex block once, `M` rows, and `map_index` (B,) naming each
/// position's), entity arrays `[B, E, ...]`, relations `[B, R, ...]` and
/// `meta [B, 8]`; H / E / R are the batch maxima, padding masked by
/// `hex_mask` / `ent_mask` / `rel_mask`. Schema: `rl_encoder_schema()`,
/// `docs/rl_encoder.md`.
#[pyfunction]
#[pyo3(signature = (games, dedupe_maps=false))]
pub fn rl_encode_batch<'py>(
    py: Python<'py>,
    games: Vec<PyRef<'py, BaseGame>>,
    dedupe_maps: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let encs: Vec<TokenEncoding> = games.iter().map(|g| g.encode_tokens()).collect();
    Batch::stack(&encs, dedupe_maps).into_dict(py)
}

fn strs<'py>(py: Python<'py>, xs: &[&str]) -> PyResult<PyObject> {
    Ok(xs.to_vec().into_pyobject(py)?.into_any().unbind())
}

/// The encoder's schema: widths, feature names per token type, pointer and
/// meta columns, and every global vocabulary.
#[pyfunction]
pub fn rl_encoder_schema<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("version", ENCODER_VERSION)?;
    d.set_item("max_seats", MAX_SEATS)?;
    d.set_item("s_max", S_MAX)?;
    d.set_item("f_hex", F_HEX)?;
    d.set_item("f_ent", F_ENT)?;
    d.set_item("f_tile", F_TILE)?;
    d.set_item("p_ent", P_ENT)?;
    d.set_item("p_hex", P_HEX)?;
    d.set_item("meta_len", META_LEN)?;
    d.set_item("ptr_none", PTR_NONE)?;
    d.set_item("ptr_neutral", PTR_NEUTRAL)?;
    d.set_item("feature_clip", FEATURE_CLIP)?;
    d.set_item("hex_features", strs(py, HEX_FEATURES)?)?;
    d.set_item("tile_features", strs(py, TILE_FEATURES)?)?;
    let ent = PyDict::new(py);
    for (i, name) in ENTITY_TYPES.iter().enumerate() {
        ent.set_item(*name, strs(py, entity_features(i as i32))?)?;
    }
    d.set_item("entity_features", ent)?;
    d.set_item("ent_ptr_columns", strs(py, ENT_PTR_COLUMNS)?)?;
    d.set_item("ent_hex_columns", strs(py, ENT_HEX_COLUMNS)?)?;
    d.set_item("meta_columns", strs(py, META_COLUMNS)?)?;
    d.set_item("entity_types", strs(py, ENTITY_TYPES)?)?;
    d.set_item("relation_kinds", strs(py, RELATION_KINDS)?)?;
    d.set_item("titles", strs(py, TITLES)?)?;
    d.set_item("round_kinds", strs(py, ROUND_KINDS)?)?;
    d.set_item("step_kinds", strs(py, STEP_KINDS)?)?;
    d.set_item("labels", strs(py, LABELS)?)?;
    d.set_item("terrains", strs(py, TERRAINS)?)?;
    d.set_item("phase_statuses", strs(py, PHASE_STATUSES)?)?;
    d.set_item("market_zones", strs(py, MARKET_ZONES)?)?;
    d.set_item("tile_colors", strs(py, TILE_COLORS)?)?;
    Ok(d)
}

/// A title's spec: hex order and geometry (`hex_axial` (H, 2),
/// `hex_neighbors` (H, 6), `hex_pos` (H, 2)), entity names and static
/// types, the tile table (`tile_ids`, `tile_feat` (T, F_TILE)) and the
/// normalizers.
#[pyfunction]
pub fn rl_spec_info<'py>(py: Python<'py>, title: &str) -> PyResult<Bound<'py, PyDict>> {
    if !crate::title::all_titles().iter().any(|t| t.name() == title) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!("unknown title {title:?}")));
    }
    let s = rl_spec_for(title);
    let h = s.num_hexes();
    let d = PyDict::new(py);
    d.set_item("title", s.title)?;
    d.set_item("title_id", s.title_id)?;
    d.set_item("layout", format!("{:?}", s.layout).to_lowercase())?;
    d.set_item("num_hexes", h)?;
    d.set_item("num_entities", s.num_entities)?;
    d.set_item("num_tiles", s.num_tiles())?;
    d.set_item("hex_coords", s.hex_coords.clone())?;
    d.set_item(
        "hex_axial",
        PyArray1::from_vec(py, s.hex_axial.iter().flat_map(|a| a.iter().copied()).collect::<Vec<i32>>()).reshape([h, 2])?,
    )?;
    d.set_item(
        "hex_pos",
        PyArray1::from_vec(py, s.hex_pos.iter().flat_map(|a| a.iter().copied()).collect::<Vec<f32>>()).reshape([h, 2])?,
    )?;
    d.set_item(
        "hex_neighbors",
        PyArray1::from_vec(py, s.hex_neighbors.iter().flat_map(|a| a.iter().copied()).collect::<Vec<i32>>())
            .reshape([h, 6])?,
    )?;
    d.set_item("entity_names", s.entity_names())?;
    d.set_item("entity_types", PyArray1::from_vec(py, s.entity_types()))?;
    d.set_item(
        "entity_offsets",
        [("global", 0usize), ("seat", 1), ("corp", s.e_corp0), ("company", s.e_company0), ("train", s.e_train0), ("phase", s.e_phase0)]
            .into_iter()
            .collect::<HashMap<&str, usize>>(),
    )?;
    d.set_item("tile_ids", s.tile_ids.clone())?;
    d.set_item("tile_feat", PyArray1::from_vec(py, s.tile_feat.clone()).reshape([s.num_tiles(), F_TILE])?)?;
    let n = PyDict::new(py);
    for (k, v) in [
        ("bank_cash", s.bank_cash),
        ("max_market_price", s.max_market_price),
        ("market_rows", s.market_rows),
        ("market_cols", s.market_cols),
        ("max_stop_revenue", s.max_stop_revenue),
        ("max_lay_cost", s.max_lay_cost),
        ("max_train_price", s.max_train_price),
        ("max_company_value", s.max_company_value),
        ("max_company_revenue", s.max_company_revenue),
        ("max_token_price", s.max_token_price),
        ("num_loans", s.num_loans),
        ("loan_value", s.loan_value),
        ("loan_interest", s.loan_interest),
        ("max_loans_major", s.max_loans_major),
        ("max_loans_minor", s.max_loans_minor),
        ("market_pool_limit", s.market_pool_limit),
    ] {
        n.set_item(k, v)?;
    }
    d.set_item("normalizers", n)?;
    Ok(d)
}
