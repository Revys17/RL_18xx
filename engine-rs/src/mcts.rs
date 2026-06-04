//! Rust MCTS (Phase 4a scaffold + Phase 4c progressive widening).
//!
//! Arena-based MCTS tree mirroring the categorical-descent semantics of
//! Python `MCTSNode` / `MCTSPlayer` (rl18xx/agent/alphazero/mcts.py +
//! self_play.py).
//!
//! Phase 4c (this revision) adds progressive widening (PW) with continuous
//! price grandchildren for the price-bearing categorical slots
//! (Bid / BuyTrain / BuyCompany). A "price grandchild" is a child of a
//! price-bearing categorical slot whose own statistics (N, W) live in the
//! parent's ``price_child_n`` / ``price_child_w`` dicts rather than the
//! parent's compressed categorical arrays. Backups still mirror up into the
//! categorical slot so categorical-level PUCT integrates over all
//! grandchildren under one slot. See Python's ``MCTSNode`` for the reference.

use std::collections::HashMap;

use numpy::PyReadonlyArray1;
use pyo3::exceptions::{PyIndexError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rand::SeedableRng;
use rand_distr::{Dirichlet, Distribution, Normal};

use crate::action_index::{index_to_action_dict, POLICY_SIZE};
use crate::factored::LegalAction;
use crate::game::BaseGame;

const VALUE_SIZE: usize = 6;
const C_PUCT_BASE: f32 = 19652.0;
const C_PUCT_INIT: f32 = 1.25;

// ----------------------------------------------------------------------------
// Price components (per-leaf NN price-head output)
// ----------------------------------------------------------------------------

/// Per-leaf ContinuousPriceHead output sliced from the model's batched
/// ``last_price_components`` dict. Indices into ``price_mean`` /
/// ``price_log_std`` come from ``slot_index``.
#[derive(Debug, Clone, Default)]
pub struct PriceComponents {
    pub price_mean: Vec<f32>,
    pub price_log_std: Vec<f32>,
    /// Map (action_type, entity_key_parts) → slot index.
    pub slot_index: HashMap<(String, Vec<String>), usize>,
    pub num_slots: usize,
}

// ----------------------------------------------------------------------------
// Arena node
// ----------------------------------------------------------------------------

pub struct RustMCTSNode {
    pub game: BaseGame,
    pub fmove: Option<u32>,
    pub parent: Option<usize>,
    pub parent_compressed_idx: usize,
    pub is_expanded: bool,
    pub losses_applied: i32,

    pub legal_action_indices: Vec<u32>,
    pub price_ranges: HashMap<u32, (i64, i64)>,
    /// Action-type name per legal slot (e.g. "Bid"/"BuyTrain"/"BuyCompany"/
    /// "Pass"/...). Populated alongside ``price_ranges`` at node construction.
    pub action_types: HashMap<u32, String>,
    pub child_n: Vec<f32>,           // [num_legal]
    pub child_w: Vec<[f32; VALUE_SIZE]>,
    pub child_prior: Vec<f32>,
    pub original_prior: Vec<f32>,

    pub children: HashMap<u32, usize>,

    // PW / price-grandchild bookkeeping. Layered as: slot_idx -> price ->
    // arena_idx / N / W. Only populated for price-bearing slots with a
    // non-degenerate price_range.
    pub price_children: HashMap<u32, HashMap<i64, usize>>,
    pub price_child_n: HashMap<u32, HashMap<i64, f32>>,
    pub price_child_w: HashMap<u32, HashMap<i64, [f32; VALUE_SIZE]>>,

    pub forced_action_chain: Vec<u32>,
    pub active_player_index: usize,
    pub num_players: usize,
    pub depth: u32,

    pub is_terminal: bool,

    /// Set on this node iff it was materialized as a PW price grandchild of
    /// its parent. Used during backup / virtual-loss to also mirror the
    /// update into the parent's categorical compressed arrays.
    pub sampled_price: Option<i64>,
    pub is_price_grandchild: bool,

    /// Continuous-price head output stashed at incorporate-results time.
    /// Consulted by ``_sample_price_for_slot`` for future PW expansions.
    pub price_components: Option<PriceComponents>,
}

impl RustMCTSNode {
    fn legal_index_in_compressed(&self, fmove: u32) -> Option<usize> {
        self.legal_action_indices.iter().position(|&x| x == fmove)
    }
}

// ----------------------------------------------------------------------------
// Price head entity-key resolver (mirrors Python ``_price_head_entity_key``).
// ----------------------------------------------------------------------------

/// Mirrors ``model_transformer.ContinuousPriceHead``'s slot layout. Returns
/// the ``(action_type, entity_key_parts)`` tuple used to index into the
/// price head's slot_index dict.
///
/// Only the head-modelled slots return ``Some``; depot trains and exchange
/// trains return ``None`` (they are fixed-price and never reach the sampler).
fn price_head_entity_key(action_index: u32, action_type: &str) -> Option<(String, Vec<String>)> {
    // We mirror the layout from `model_transformer.py::ContinuousPriceHead`
    // and `action_mapper.py::_PRICE_HEAD_*`. Order matches Python.
    const COMPANIES: [&str; 6] = ["SV", "CS", "DH", "MH", "CA", "BO"];
    const CORPORATIONS: [&str; 8] = ["PRR", "NYC", "CPR", "B&O", "C&O", "ERIE", "NYNH", "B&M"];
    const TRAIN_TYPES: [&str; 6] = ["2", "3", "4", "5", "6", "D"];

    // Re-fetch the layout offsets via the Rust action_index module. The
    // layout is global (memoized once); the lookup is cheap.
    let lo = crate::action_index::layout();
    let bid_start = *lo.action_offsets.get("Bid")? as i64;
    let buy_train_start = *lo.action_offsets.get("BuyTrain")? as i64;
    let buy_company_start = *lo.action_offsets.get("BuyCompany")? as i64;

    let idx = action_index as i64;

    if action_type == "Bid" {
        let bid_end = bid_start + COMPANIES.len() as i64;
        if idx >= bid_start && idx < bid_end {
            let i = (idx - bid_start) as usize;
            return Some(("Bid".to_string(), vec![COMPANIES[i].to_string()]));
        }
        return None;
    }

    if action_type == "BuyTrain" {
        // First slot is depot; next ``train_type_offsets.len()`` are market
        // discarded — both fixed-price. The cross-corp block follows.
        let train_price_count = lo.train_price_offsets.len() as i64;
        let train_type_count = TRAIN_TYPES.len() as i64;
        let cross_corp_start = buy_train_start + 1 + train_type_count;
        let per_corp = train_type_count * train_price_count;
        let cross_corp_end = cross_corp_start + (CORPORATIONS.len() as i64) * per_corp;
        if idx >= cross_corp_start && idx < cross_corp_end {
            let rel = idx - cross_corp_start;
            let corp_idx = (rel / per_corp) as usize;
            let train_idx = ((rel % per_corp) / train_price_count) as usize;
            return Some((
                "BuyTrain".to_string(),
                vec![
                    CORPORATIONS[corp_idx].to_string(),
                    TRAIN_TYPES[train_idx].to_string(),
                ],
            ));
        }
        return None;
    }

    if action_type == "BuyCompany" {
        let price_count = lo.buy_company_price_offsets.len() as i64;
        let block_end = buy_company_start + (COMPANIES.len() as i64) * price_count;
        if idx >= buy_company_start && idx < block_end {
            let rel = idx - buy_company_start;
            let company_idx = (rel / price_count) as usize;
            return Some((
                "BuyCompany".to_string(),
                vec![COMPANIES[company_idx].to_string()],
            ));
        }
        return None;
    }

    None
}

// ----------------------------------------------------------------------------
// Price snapping + sampling helpers (mirror Python ``_snap_price`` /
// ``sample_price_for_pw``).
// ----------------------------------------------------------------------------

fn price_grid_step(action_type: &str) -> i64 {
    match action_type {
        "Bid" => 5,
        _ => 1,
    }
}

fn snap_price(price: f32, action_type: &str, price_min: i64, price_max: i64) -> i64 {
    let step = price_grid_step(action_type);
    let mut snapped = ((price / step as f32).round() as i64) * step;
    if snapped < price_min {
        let rem = price_min.rem_euclid(step);
        snapped = if rem == 0 { price_min } else { price_min + (step - rem) };
    }
    if snapped > price_max {
        if step > 1 {
            snapped = price_max - price_max.rem_euclid(step);
        } else {
            snapped = price_max;
        }
    }
    snapped.max(price_min).min(price_max)
}

fn sample_price_for_pw(
    price_mean: f32,
    price_log_std: f32,
    action_type: &str,
    price_range: (i64, i64),
) -> i64 {
    let (p_min, p_max) = price_range;
    if p_min == p_max {
        return p_min;
    }
    let clipped = price_log_std.clamp(-1.0_f32, 8.5_f32);
    let sigma = clipped.exp().max(1e-3);
    let mu = price_mean;
    let mut rng = rand::rngs::StdRng::from_entropy();
    let dist = match Normal::new(mu as f64, sigma as f64) {
        Ok(d) => d,
        Err(_) => {
            // Fallback: uniform sample on the snap grid.
            let step = price_grid_step(action_type);
            let n_choices = (((p_max - p_min) / step) + 1).max(1);
            use rand::Rng;
            let k = rng.gen_range(0..n_choices);
            return p_min + k * step;
        }
    };
    for _ in 0..8 {
        let sample = dist.sample(&mut rng) as f32;
        if (p_min as f32 - sigma) <= sample && sample <= (p_max as f32 + sigma) {
            return snap_price(sample, action_type, p_min, p_max);
        }
    }
    let step = price_grid_step(action_type);
    let n_choices = (((p_max - p_min) / step) + 1).max(1);
    use rand::Rng;
    let k = rng.gen_range(0..n_choices);
    p_min + k * step
}

fn pw_target_children(visits: f32, pw_c: f32, pw_alpha: f32, min_children: usize) -> usize {
    let target = pw_c * (visits.max(0.0)).powf(pw_alpha);
    let rounded = target.ceil() as i64;
    std::cmp::max(min_children as i64, rounded) as usize
}

// ----------------------------------------------------------------------------
// Player (PyO3 surface)
// ----------------------------------------------------------------------------

#[pyclass]
pub struct RustMCTSPlayer {
    pub arena: Vec<RustMCTSNode>,
    pub root_idx: usize,
    pub num_players: usize,
    pub c_puct_init: f32,
    pub c_puct_base: f32,
    pub backup_discount: f32,
    /// Root-level visit count (the root has no parent to read from).
    pub root_n: f32,
    pub root_w: [f32; VALUE_SIZE],

    // PW knobs (defaults mirror Python ``SelfPlayConfig``).
    pub pw_c: f32,
    pub pw_alpha: f32,
    pub min_price_children: usize,
}

impl RustMCTSPlayer {
    fn build_node(
        &self,
        game: BaseGame,
        fmove: Option<u32>,
        parent: Option<usize>,
    ) -> PyResult<RustMCTSNode> {
        // Enumerate factored legal actions; encode each to a flat policy
        // slot; dedupe; also record price ranges + action types.
        let mut game = game;
        let choices: Vec<LegalAction> = game.get_factored_choices_impl();

        let mut legal_action_indices: Vec<u32> = Vec::new();
        let mut price_ranges: HashMap<u32, (i64, i64)> = HashMap::new();
        let mut action_types: HashMap<u32, String> = HashMap::new();
        let mut seen: std::collections::HashSet<u32> = std::collections::HashSet::new();
        for la in &choices {
            let idx = match crate::action_index::legal_action_to_index(la) {
                Some(i) => i,
                None => continue,
            };
            if seen.insert(idx) {
                legal_action_indices.push(idx);
                if let Some(pr) = la.price_range {
                    price_ranges.insert(idx, pr);
                }
                action_types.insert(idx, la.action_type.clone());
            } else if let Some(pr) = la.price_range {
                let existing = price_ranges.entry(idx).or_insert(pr);
                *existing = (existing.0.min(pr.0), existing.1.max(pr.1));
            }
        }
        legal_action_indices.sort_unstable();

        let num_legal = legal_action_indices.len();
        let mut player_ids: Vec<u32> = game.players_for_factored().iter().map(|p| p.id).collect();
        player_ids.sort_unstable();
        let active_pid = game
            .active_players_pub()
            .first()
            .map(|p| p.id)
            .unwrap_or(player_ids[0]);
        let active_player_index = player_ids.iter().position(|&p| p == active_pid).unwrap_or(0);
        let num_players = player_ids.len();

        let is_terminal = game.is_finished_pub();
        let depth = match parent {
            Some(_) => 0, // set by caller
            None => 0,
        };

        Ok(RustMCTSNode {
            game,
            fmove,
            parent,
            parent_compressed_idx: 0,
            is_expanded: false,
            losses_applied: 0,
            legal_action_indices,
            price_ranges,
            action_types,
            child_n: vec![0.0; num_legal],
            child_w: vec![[0.0; VALUE_SIZE]; num_legal],
            child_prior: vec![0.0; num_legal],
            original_prior: vec![0.0; num_legal],
            children: HashMap::new(),
            price_children: HashMap::new(),
            price_child_n: HashMap::new(),
            price_child_w: HashMap::new(),
            forced_action_chain: Vec::new(),
            active_player_index,
            num_players,
            depth,
            is_terminal,
            sampled_price: None,
            is_price_grandchild: false,
            price_components: None,
        })
    }

    /// True iff the slot at ``action_index`` is a PW slot (price-bearing
    /// with non-degenerate range).
    fn is_pw_slot(node: &RustMCTSNode, action_index: u32) -> bool {
        match node.price_ranges.get(&action_index) {
            Some(&(lo, hi)) => lo != hi,
            None => false,
        }
    }
}

#[pymethods]
impl RustMCTSPlayer {
    /// Build a fresh player from a Python-supplied BaseGame.
    ///
    /// Accepts either an `engine_rs.BaseGame` directly or a `RustGameAdapter`
    /// (we unwrap via its `_game` attribute).
    #[new]
    #[pyo3(signature = (game_obj, pw_c=None, pw_alpha=None, min_price_children=None))]
    pub fn new(
        py: Python<'_>,
        game_obj: PyObject,
        pw_c: Option<f32>,
        pw_alpha: Option<f32>,
        min_price_children: Option<usize>,
    ) -> PyResult<Self> {
        // Resolve the inner Rust BaseGame. If we got a RustGameAdapter, pull
        // out `._game`; otherwise treat the arg as the BaseGame itself.
        let bound = game_obj.into_bound(py);
        let base_obj = if bound.hasattr("_game")? {
            bound.getattr("_game")?
        } else {
            bound
        };

        // Extract a clone of the Rust BaseGame via `pickle_clone` so the
        // Python-side game keeps its state.
        let cloned = base_obj.call_method0("pickle_clone")?;
        let borrowed: PyRef<BaseGame> = cloned.extract()?;
        let cloned_inner: BaseGame = borrowed.clone_for_search();
        drop(borrowed);

        let mut player = RustMCTSPlayer {
            arena: Vec::new(),
            root_idx: 0,
            num_players: 0,
            c_puct_init: C_PUCT_INIT,
            c_puct_base: C_PUCT_BASE,
            backup_discount: 1.0,
            root_n: 0.0,
            root_w: [0.0; VALUE_SIZE],
            pw_c: pw_c.unwrap_or(1.0),
            pw_alpha: pw_alpha.unwrap_or(0.5),
            min_price_children: min_price_children.unwrap_or(1),
        };
        let root = player.build_node(cloned_inner, None, None)?;
        player.num_players = root.num_players;
        player.arena.push(root);
        player.root_idx = 0;
        Ok(player)
    }

    /// Set PW knobs after construction (mirrors a config update in Python).
    pub fn set_pw_config(&mut self, pw_c: f32, pw_alpha: f32, min_price_children: usize) {
        self.pw_c = pw_c;
        self.pw_alpha = pw_alpha;
        self.min_price_children = min_price_children;
    }

    /// Number of players in the root game.
    #[getter]
    fn num_players(&self) -> usize {
        self.num_players
    }

    /// Visit count at the root.
    fn n_at_root(&self) -> f32 {
        self.root_n
    }

    /// Per-player Q vector at the root. Mirrors Python ``MCTSNode.Q`` truncated
    /// to ``num_players`` (Phase 2 ``check_resign`` reads this to detect
    /// consensus leader/gap on a rolling window of recent moves).
    ///
    /// ``Q[i] = root_w[i] / (1 + root_n)`` — same formula as Python.
    fn root_q_vector(&self) -> Vec<f32> {
        let denom = 1.0 + self.root_n;
        self.root_w
            .iter()
            .take(self.num_players)
            .map(|&w| w / denom)
            .collect()
    }

    /// Full-size visit-count vector at the root (length POLICY_SIZE).
    /// Non-legal slots stay 0.
    fn child_n_at_root(&self) -> Vec<f32> {
        let root = &self.arena[self.root_idx];
        let mut out = vec![0.0f32; POLICY_SIZE as usize];
        for (i, &flat) in root.legal_action_indices.iter().enumerate() {
            out[flat as usize] = root.child_n[i];
        }
        out
    }

    /// Number of legal actions at the root.
    fn num_legal_at_root(&self) -> usize {
        self.arena[self.root_idx].legal_action_indices.len()
    }

    /// Legal action indices at the root (for parity diagnostics).
    fn legal_action_indices_at_root(&self) -> Vec<u32> {
        self.arena[self.root_idx].legal_action_indices.clone()
    }

    /// Legal action indices at an arbitrary arena node. Used by Phase 1
    /// PlayoutTrace finalization to compute the leaf's prior entropy over
    /// just its legal slots.
    fn legal_action_indices_for_idx(&self, arena_idx: usize) -> PyResult<Vec<u32>> {
        if arena_idx >= self.arena.len() {
            return Err(PyIndexError::new_err(format!(
                "arena_idx {} out of range (arena has {} nodes)",
                arena_idx, self.arena.len()
            )));
        }
        Ok(self.arena[arena_idx].legal_action_indices.clone())
    }

    /// Visit counts per (slot, price) for every PW slot at the root. Returns
    /// ``{slot_idx -> {price -> N}}`` so the Python adapter can extract price
    /// targets for training.
    pub fn price_grandchildren_at_root(&self) -> HashMap<u32, HashMap<i64, f32>> {
        self.arena[self.root_idx].price_child_n.clone()
    }

    /// Most-visited grandchild price under a root PW slot. Returns ``None``
    /// if no grandchildren are tracked.
    pub fn most_visited_price_for_slot(&self, action_index: u32) -> Option<i64> {
        let root = &self.arena[self.root_idx];
        let entries = root.price_child_n.get(&action_index)?;
        if entries.is_empty() {
            return None;
        }
        // Tie-break by price (stable ordering, matches Python's
        // ``max(..., key=(N, price))``).
        let mut best: Option<(f32, i64)> = None;
        for (&price, &n) in entries.iter() {
            let key = (n, price);
            best = match best {
                None => Some(key),
                Some(b) => {
                    if key.0 > b.0 || (key.0 == b.0 && key.1 > b.1) {
                        Some(key)
                    } else {
                        Some(b)
                    }
                }
            };
        }
        Some(best.unwrap().1)
    }

    /// PUCT descent from the root to a leaf. Returns the leaf's arena index.
    pub fn select_leaf(&mut self, py: Python<'_>) -> PyResult<usize> {
        let (leaf, _, _, _) = self._select_leaf_inner(py, false)?;
        Ok(leaf)
    }

    /// Tracing variant of ``select_leaf`` — same PUCT/PW descent, but also
    /// returns the descent path so callers can populate a Phase 1
    /// ``PlayoutTrace``. Returns ``(leaf_idx, action_path, pw_grandchild_path,
    /// forced_chain_lengths)``. Trace vectors are parallel: index ``i`` is the
    /// step from depth ``i`` to depth ``i+1``.
    pub fn select_leaf_with_trace(
        &mut self,
        py: Python<'_>,
    ) -> PyResult<(usize, Vec<u32>, Vec<bool>, Vec<u32>)> {
        self._select_leaf_inner(py, true)
    }

    fn _select_leaf_inner(
        &mut self,
        py: Python<'_>,
        record: bool,
    ) -> PyResult<(usize, Vec<u32>, Vec<bool>, Vec<u32>)> {
        let mut current = self.root_idx;
        let mut action_path: Vec<u32> = Vec::new();
        let mut pw_path: Vec<bool> = Vec::new();
        let mut forced_lens: Vec<u32> = Vec::new();
        loop {
            if !self.arena[current].is_expanded {
                return Ok((current, action_path, pw_path, forced_lens));
            }
            if self.arena[current].legal_action_indices.is_empty() {
                return Ok((current, action_path, pw_path, forced_lens));
            }
            let best_idx = self.argmax_action_score(current);
            let best_move = self.arena[current].legal_action_indices[best_idx];

            // PW dispatch: price-bearing slots with a non-degenerate range
            // descend through ``_select_or_expand_price_child``; everything
            // else uses the categorical ``maybe_add_child``.
            let is_pw = Self::is_pw_slot(&self.arena[current], best_move);
            current = if is_pw {
                self._select_or_expand_price_child(py, current, best_move)?
            } else {
                self.maybe_add_child(py, current, best_move, None)?
            };
            if record {
                action_path.push(best_move);
                pw_path.push(is_pw);
                forced_lens.push(self.arena[current].forced_action_chain.len() as u32);
            }
        }
    }

    /// PW descent for a price-bearing categorical slot.
    ///
    /// If the grandchild count is below the target ``k = max(min, ceil(pw_c *
    /// N^pw_alpha))``, sample a fresh price and either return an existing
    /// grandchild (collision) or expand a new one. Otherwise PUCT-select
    /// among existing grandchildren.
    pub fn _select_or_expand_price_child(
        &mut self,
        py: Python<'_>,
        arena_idx: usize,
        action_index: u32,
    ) -> PyResult<usize> {
        let slot_visits = {
            let node = &self.arena[arena_idx];
            let compressed = node
                .legal_index_in_compressed(action_index)
                .ok_or_else(|| {
                    PyValueError::new_err(format!(
                        "action_index {} not legal at arena_idx {}",
                        action_index, arena_idx
                    ))
                })?;
            node.child_n[compressed]
        };
        let target = pw_target_children(
            slot_visits,
            self.pw_c,
            self.pw_alpha,
            self.min_price_children,
        );
        let existing_count = self.arena[arena_idx]
            .price_children
            .get(&action_index)
            .map(|m| m.len())
            .unwrap_or(0);

        if existing_count < target {
            // Sample a fresh price; if it collides with an existing
            // grandchild, return that one (it will accrue a fresh visit on
            // backup). Otherwise materialize a new grandchild.
            let price_range = *self.arena[arena_idx]
                .price_ranges
                .get(&action_index)
                .ok_or_else(|| PyRuntimeError::new_err("price_range missing on PW slot"))?;
            let action_type = self.arena[arena_idx]
                .action_types
                .get(&action_index)
                .cloned()
                .unwrap_or_default();
            let sampled =
                self._sample_price_with_index(arena_idx, action_index, &action_type, price_range);
            if let Some(slot_grandchildren) = self.arena[arena_idx].price_children.get(&action_index) {
                if let Some(&existing_idx) = slot_grandchildren.get(&sampled) {
                    return Ok(existing_idx);
                }
            }
            return self.maybe_add_child(py, arena_idx, action_index, Some(sampled));
        }

        // PW cap reached — PUCT among existing grandchildren using the
        // categorical slot's prior split among them.
        let (slot_prior, ap, idx_compressed) = {
            let node = &self.arena[arena_idx];
            let i = node.legal_index_in_compressed(action_index).unwrap();
            (node.child_prior[i], node.active_player_index, i)
        };
        let n_existing = self.arena[arena_idx]
            .price_children
            .get(&action_index)
            .map(|m| m.len())
            .unwrap_or(1)
            .max(1);
        let per_grandchild_prior = slot_prior / (n_existing as f32);
        let _ = idx_compressed;

        let c_puct = 2.0
            * ((1.0 + slot_visits + self.c_puct_base) / self.c_puct_base).ln()
            + 2.0 * self.c_puct_init;
        let n_s = (slot_visits - 1.0).max(1.0);
        let sqrt_n_s = n_s.sqrt();

        let grandchildren_clone: HashMap<i64, usize> = self
            .arena[arena_idx]
            .price_children
            .get(&action_index)
            .cloned()
            .unwrap_or_default();
        let nw_clone = self.arena[arena_idx]
            .price_child_n
            .get(&action_index)
            .cloned()
            .unwrap_or_default();
        let ww_clone = self.arena[arena_idx]
            .price_child_w
            .get(&action_index)
            .cloned()
            .unwrap_or_default();

        let mut best_score = f32::NEG_INFINITY;
        let mut best_idx: usize = *grandchildren_clone.values().next().unwrap();
        for (&price, &gc_idx) in grandchildren_clone.iter() {
            let n_sa = *nw_clone.get(&price).unwrap_or(&0.0);
            let w_vec = *ww_clone.get(&price).unwrap_or(&[0.0; VALUE_SIZE]);
            let q_sa = if n_sa > 0.0 { w_vec[ap] / (1.0 + n_sa) } else { 0.0 };
            let u_sa = c_puct * per_grandchild_prior * sqrt_n_s / (1.0 + n_sa);
            let score = q_sa + u_sa;
            if score > best_score {
                best_score = score;
                best_idx = gc_idx;
            }
        }
        Ok(best_idx)
    }

    /// Look up or create the child for `action_index` under `arena_idx`.
    /// Implements the forced-chain collapse from Python's `maybe_add_child`.
    ///
    /// ``price`` semantics:
    ///   - ``None`` for non-PW slots; falls back to fixed-price (the engine
    ///     picks via ``map_index_to_action`` / ``..._with_price`` at min).
    ///   - ``None`` for a PW slot: sample via the price head's posterior.
    ///   - ``Some(p)`` for any slot: snap and use (PW grandchild path).
    #[pyo3(signature = (arena_idx, action_index, price=None))]
    pub fn maybe_add_child(
        &mut self,
        py: Python<'_>,
        arena_idx: usize,
        action_index: u32,
        price: Option<i64>,
    ) -> PyResult<usize> {
        // Categorical existing-child fast path.
        let is_pw = Self::is_pw_slot(&self.arena[arena_idx], action_index);
        if !is_pw {
            if let Some(&child_idx) = self.arena[arena_idx].children.get(&action_index) {
                return Ok(child_idx);
            }
        }

        let parent_compressed_idx = self.arena[arena_idx]
            .legal_index_in_compressed(action_index)
            .ok_or_else(|| {
                PyValueError::new_err(format!(
                    "action_index {} not legal at arena_idx {}",
                    action_index, arena_idx
                ))
            })?;

        // Resolve / snap the price for PW slots; existing-grandchild fast
        // path lives below the action_type lookup.
        let (sampled_price, expansion_price_for_apply): (Option<i64>, Option<i64>) = if is_pw {
            let price_range = *self.arena[arena_idx]
                .price_ranges
                .get(&action_index)
                .ok_or_else(|| PyRuntimeError::new_err("price_range missing on PW slot"))?;
            let action_type = self.arena[arena_idx]
                .action_types
                .get(&action_index)
                .cloned()
                .unwrap_or_default();
            let p = match price {
                Some(p) => snap_price(p as f32, &action_type, price_range.0, price_range.1),
                None => self._sample_price_with_index(
                    arena_idx,
                    action_index,
                    &action_type,
                    price_range,
                ),
            };
            // Existing-grandchild fast path.
            if let Some(slot_grandchildren) = self.arena[arena_idx].price_children.get(&action_index) {
                if let Some(&existing_idx) = slot_grandchildren.get(&p) {
                    return Ok(existing_idx);
                }
            }
            (Some(p), Some(p))
        } else if let Some((lo, _)) = self.arena[arena_idx].price_ranges.get(&action_index) {
            // Fixed-price categorical (depot trains): use engine min price.
            (Some(*lo), Some(*lo))
        } else {
            (None, None)
        };

        // Clone the parent's game and apply the action via Python ActionMapper.
        let mut new_game = self.arena[arena_idx].game.clone_for_search();
        apply_action(py, &mut new_game, action_index, expansion_price_for_apply)?;

        // Forced-chain collapse: while exactly one legal action, apply it.
        let mut forced_chain: Vec<u32> = Vec::new();
        loop {
            if new_game.is_finished_pub() {
                break;
            }
            // Enumerate factored choices, dedupe to flat indices.
            let choices = new_game.get_factored_choices_impl();
            let mut flat_indices: Vec<u32> = Vec::new();
            let mut seen: std::collections::HashSet<u32> = std::collections::HashSet::new();
            let mut forced_price_ranges: HashMap<u32, (i64, i64)> = HashMap::new();
            for la in &choices {
                if let Some(idx) = crate::action_index::legal_action_to_index(la) {
                    if seen.insert(idx) {
                        flat_indices.push(idx);
                    }
                    if let Some(pr) = la.price_range {
                        let existing = forced_price_ranges.entry(idx).or_insert(pr);
                        *existing = (existing.0.min(pr.0), existing.1.max(pr.1));
                    }
                }
            }
            if flat_indices.len() != 1 {
                break;
            }
            let forced_idx = flat_indices[0];
            let forced_price = forced_price_ranges.get(&forced_idx).map(|(lo, _)| *lo);
            apply_action(py, &mut new_game, forced_idx, forced_price)?;
            forced_chain.push(forced_idx);
        }

        let depth = self.arena[arena_idx].depth + 1;
        let mut child = self.build_node(new_game, Some(action_index), Some(arena_idx))?;
        child.parent_compressed_idx = parent_compressed_idx;
        child.forced_action_chain = forced_chain;
        child.depth = depth;
        child.sampled_price = sampled_price;
        child.is_price_grandchild = is_pw;

        let child_idx = self.arena.len();
        self.arena.push(child);

        if is_pw {
            let p = sampled_price.unwrap();
            let entry = self.arena[arena_idx]
                .price_children
                .entry(action_index)
                .or_insert_with(HashMap::new);
            entry.insert(p, child_idx);
            self.arena[arena_idx]
                .price_child_n
                .entry(action_index)
                .or_insert_with(HashMap::new)
                .insert(p, 0.0);
            self.arena[arena_idx]
                .price_child_w
                .entry(action_index)
                .or_insert_with(HashMap::new)
                .insert(p, [0.0; VALUE_SIZE]);
        } else {
            self.arena[arena_idx].children.insert(action_index, child_idx);
        }

        Ok(child_idx)
    }

    /// Apply a virtual loss along the path from `arena_idx` up to (but not
    /// including) the root. Matches Python's `add_virtual_loss`. Mirrors
    /// the update into the categorical slot for price grandchildren so PUCT
    /// at the categorical level sees the loss too.
    pub fn add_virtual_loss(&mut self, arena_idx: usize) {
        let up_to = self.root_idx;
        let mut current = arena_idx;
        loop {
            if current == up_to {
                return;
            }
            let parent = match self.arena[current].parent {
                Some(p) => p,
                None => return,
            };
            let prev_player = self.arena[parent].active_player_index;
            let pidx = self.arena[current].parent_compressed_idx;
            // Grandchild update first.
            if self.arena[current].is_price_grandchild {
                let fmove = self.arena[current].fmove.unwrap();
                let sp = self.arena[current].sampled_price.unwrap();
                if let Some(map) = self.arena[parent].price_child_w.get_mut(&fmove) {
                    if let Some(w) = map.get_mut(&sp) {
                        w[prev_player] -= 1.0;
                    }
                }
                // Mirror into categorical compressed W.
                self.arena[parent].child_w[pidx][prev_player] -= 1.0;
            } else {
                self.arena[parent].child_w[pidx][prev_player] -= 1.0;
            }
            self.arena[current].losses_applied += 1;
            current = parent;
        }
    }

    pub fn revert_virtual_loss(&mut self, arena_idx: usize) {
        let up_to = self.root_idx;
        let mut current = arena_idx;
        loop {
            if current == up_to {
                return;
            }
            let parent = match self.arena[current].parent {
                Some(p) => p,
                None => return,
            };
            let prev_player = self.arena[parent].active_player_index;
            let pidx = self.arena[current].parent_compressed_idx;
            if self.arena[current].is_price_grandchild {
                let fmove = self.arena[current].fmove.unwrap();
                let sp = self.arena[current].sampled_price.unwrap();
                if let Some(map) = self.arena[parent].price_child_w.get_mut(&fmove) {
                    if let Some(w) = map.get_mut(&sp) {
                        w[prev_player] += 1.0;
                    }
                }
                self.arena[parent].child_w[pidx][prev_player] += 1.0;
            } else {
                self.arena[parent].child_w[pidx][prev_player] += 1.0;
            }
            self.arena[current].losses_applied -= 1;
            current = parent;
        }
    }

    /// Backup a per-player value vector along the path from `arena_idx` to
    /// the root. For price grandchildren, both the per-price N/W and the
    /// categorical compressed N/W are updated (the latter so categorical-
    /// level PUCT integrates over all grandchildren).
    pub fn backup_value(&mut self, arena_idx: usize, value: Vec<f32>) -> PyResult<()> {
        if value.len() != VALUE_SIZE {
            return Err(PyValueError::new_err(format!(
                "value must have length {}",
                VALUE_SIZE
            )));
        }
        let mut v = [0.0f32; VALUE_SIZE];
        for i in 0..VALUE_SIZE {
            v[i] = value[i];
        }

        let up_to = self.root_idx;
        let mut current = arena_idx;
        let mut current_value = v;
        loop {
            // Update this node's N/W.
            if current == self.root_idx {
                self.root_n += 1.0;
                for i in 0..VALUE_SIZE {
                    self.root_w[i] += current_value[i];
                }
                return Ok(());
            }
            let parent = match self.arena[current].parent {
                Some(p) => p,
                None => return Ok(()),
            };
            let pidx = self.arena[current].parent_compressed_idx;
            if self.arena[current].is_price_grandchild {
                let fmove = self.arena[current].fmove.unwrap();
                let sp = self.arena[current].sampled_price.unwrap();
                if let Some(map) = self.arena[parent].price_child_n.get_mut(&fmove) {
                    if let Some(n) = map.get_mut(&sp) {
                        *n += 1.0;
                    }
                }
                if let Some(map) = self.arena[parent].price_child_w.get_mut(&fmove) {
                    if let Some(w) = map.get_mut(&sp) {
                        for i in 0..VALUE_SIZE {
                            w[i] += current_value[i];
                        }
                    }
                }
                // Mirror into categorical compressed N/W.
                self.arena[parent].child_n[pidx] += 1.0;
                for i in 0..VALUE_SIZE {
                    self.arena[parent].child_w[pidx][i] += current_value[i];
                }
            } else {
                self.arena[parent].child_n[pidx] += 1.0;
                for i in 0..VALUE_SIZE {
                    self.arena[parent].child_w[pidx][i] += current_value[i];
                }
            }
            if current == up_to {
                return Ok(());
            }
            // Apply backup discount.
            for i in 0..VALUE_SIZE {
                current_value[i] *= self.backup_discount;
            }
            current = parent;
        }
    }

    /// Incorporate priors + value at the leaf (Python's `incorporate_results`).
    ///
    /// `probs`: numpy float32 vector of length POLICY_SIZE (full policy).
    /// `value`: numpy float32 vector of length VALUE_SIZE.
    /// `price_components` (optional): dict with keys
    ///   - ``price_mean`` (np.float32 array, length num_slots)
    ///   - ``price_log_std`` (np.float32 array, length num_slots)
    ///   - ``slot_index`` ({(action_type_str, (entity_key_parts,)): int})
    ///   - ``num_slots`` (int)
    #[pyo3(signature = (arena_idx, probs, value, price_components=None))]
    pub fn incorporate_results(
        &mut self,
        py: Python<'_>,
        arena_idx: usize,
        probs: PyReadonlyArray1<'_, f32>,
        value: PyReadonlyArray1<'_, f32>,
        price_components: Option<PyObject>,
    ) -> PyResult<()> {
        let probs_slice = probs.as_slice()?;
        if probs_slice.len() != POLICY_SIZE as usize {
            return Err(PyValueError::new_err(format!(
                "probs length {} != POLICY_SIZE {}",
                probs_slice.len(),
                POLICY_SIZE
            )));
        }
        let value_slice = value.as_slice()?;
        if value_slice.len() != VALUE_SIZE {
            return Err(PyValueError::new_err(format!(
                "value length {} != VALUE_SIZE {}",
                value_slice.len(),
                VALUE_SIZE
            )));
        }
        let mut value_arr = [0.0f32; VALUE_SIZE];
        for i in 0..VALUE_SIZE {
            value_arr[i] = value_slice[i];
        }

        // Decode price_components (if present) into the Rust struct.
        if let Some(pc_obj) = price_components.as_ref() {
            let bound = pc_obj.bind(py);
            // Allow ``None`` to mean "no price components".
            if !bound.is_none() {
                let decoded = decode_price_components(py, bound)?;
                self.arena[arena_idx].price_components = Some(decoded);
            }
        }

        if self.arena[arena_idx].is_expanded {
            // Re-visited node — just back up the value again.
            self.backup_value(arena_idx, value_arr.to_vec())?;
            return Ok(());
        }
        self.arena[arena_idx].is_expanded = true;

        // Extract legal-prior compressed array, normalize.
        let legal: Vec<u32> = self.arena[arena_idx].legal_action_indices.clone();
        let mut legal_probs: Vec<f32> = legal
            .iter()
            .map(|&i| probs_slice[i as usize])
            .collect();
        let s: f32 = legal_probs.iter().sum();
        if s > 0.0 {
            for p in legal_probs.iter_mut() {
                *p /= s;
            }
        }
        self.arena[arena_idx].original_prior = legal_probs.clone();
        self.arena[arena_idx].child_prior = legal_probs;
        // child_w stays zeroed (standard AlphaZero).
        let n = self.arena[arena_idx].legal_action_indices.len();
        self.arena[arena_idx].child_w = vec![[0.0; VALUE_SIZE]; n];

        self.backup_value(arena_idx, value_arr.to_vec())?;
        Ok(())
    }

    /// Wrap the leaf's encoded game state for Python-side NN inference.
    /// Returns the bare BaseGame python object so Python can call its
    /// existing `Encoder_GNN` flow.
    pub fn get_game_for_idx(&self, py: Python<'_>, arena_idx: usize) -> PyResult<PyObject> {
        if arena_idx >= self.arena.len() {
            return Err(PyIndexError::new_err("arena_idx out of bounds"));
        }
        // Clone the Rust game so the Python side gets a fresh handle (the
        // arena retains ownership).
        let g = self.arena[arena_idx].game.clone_for_search();
        Ok(Py::new(py, g)?.into_any())
    }

    /// Diagnostic: depth of `arena_idx`.
    fn depth_of(&self, arena_idx: usize) -> PyResult<u32> {
        if arena_idx >= self.arena.len() {
            return Err(PyIndexError::new_err("arena_idx out of bounds"));
        }
        Ok(self.arena[arena_idx].depth)
    }

    /// Diagnostic: how many nodes are in the arena.
    fn arena_size(&self) -> usize {
        self.arena.len()
    }

    /// Whether the leaf at `arena_idx` is terminal (finished game).
    fn is_terminal(&self, arena_idx: usize) -> PyResult<bool> {
        if arena_idx >= self.arena.len() {
            return Err(PyIndexError::new_err("arena_idx out of bounds"));
        }
        Ok(self.arena[arena_idx].is_terminal)
    }

    /// Whether the leaf at `arena_idx` is already expanded.
    fn is_expanded(&self, arena_idx: usize) -> PyResult<bool> {
        if arena_idx >= self.arena.len() {
            return Err(PyIndexError::new_err("arena_idx out of bounds"));
        }
        Ok(self.arena[arena_idx].is_expanded)
    }

    /// Advance the search root to the child for `action_index`. Mirrors
    /// the bookkeeping that Python's ``MCTSPlayer.play_move`` does when it
    /// reassigns ``self.root``: rebase ``root_idx``/``root_n``/``root_w`` to
    /// the chosen child so subsequent ``select_leaf`` calls descend from it.
    ///
    /// For PW slots, the new root is the most-visited price grandchild (so
    /// the next move starts under the committed price). Otherwise the
    /// regular categorical child is used.
    pub fn advance_root(&mut self, py: Python<'_>, action_index: u32) -> PyResult<()> {
        let is_pw = Self::is_pw_slot(&self.arena[self.root_idx], action_index);
        let child_idx = if is_pw {
            let price_opt = self.most_visited_price_for_slot(action_index);
            if let Some(price) = price_opt {
                // Find the grandchild with this price.
                let gc = self.arena[self.root_idx]
                    .price_children
                    .get(&action_index)
                    .and_then(|m| m.get(&price))
                    .copied();
                match gc {
                    Some(idx) => idx,
                    None => self.maybe_add_child(py, self.root_idx, action_index, Some(price))?,
                }
            } else {
                self.maybe_add_child(py, self.root_idx, action_index, None)?
            }
        } else {
            self.maybe_add_child(py, self.root_idx, action_index, None)?
        };
        let pidx = self.arena[child_idx].parent_compressed_idx;
        let new_n = self.arena[self.root_idx].child_n[pidx];
        let new_w = self.arena[self.root_idx].child_w[pidx];
        self.root_idx = child_idx;
        self.root_n = new_n;
        self.root_w = new_w;
        // The new root is no longer a "child" of anyone — sever the parent
        // backlink so future virtual-loss / backup walks stop here.
        self.arena[self.root_idx].parent = None;
        // Also clear the price-grandchild flag — it should not mirror into
        // a (now-gone) parent slot during subsequent backups.
        self.arena[self.root_idx].is_price_grandchild = false;
        Ok(())
    }

    /// Compute a length-POLICY_SIZE visit-count policy at the root with
    /// temperature scaling. Mirrors Python ``MCTSNode.children_as_pi``.
    pub fn pi_at_root(&self, temperature: f32) -> Vec<f32> {
        let root = &self.arena[self.root_idx];
        let mut out = vec![0.0f32; POLICY_SIZE as usize];
        let num_legal = root.legal_action_indices.len();
        if num_legal == 0 {
            return out;
        }
        if temperature < 1e-8 {
            let mut best_i = 0usize;
            let mut best_v = f32::NEG_INFINITY;
            for (i, &n) in root.child_n.iter().enumerate() {
                if n > best_v {
                    best_v = n;
                    best_i = i;
                }
            }
            out[root.legal_action_indices[best_i] as usize] = 1.0;
            return out;
        }
        let mut probs = vec![0.0f64; num_legal];
        if (temperature - 1.0).abs() < 1e-8 {
            for i in 0..num_legal {
                probs[i] = root.child_n[i] as f64;
            }
        } else {
            let inv_t = 1.0 / temperature as f64;
            for i in 0..num_legal {
                let n = root.child_n[i] as f64;
                probs[i] = n.powf(inv_t);
            }
        }
        let sum: f64 = probs.iter().sum();
        if sum <= 0.0 {
            let u = 1.0 / num_legal as f32;
            for &flat in root.legal_action_indices.iter() {
                out[flat as usize] = u;
            }
            return out;
        }
        for (i, &flat) in root.legal_action_indices.iter().enumerate() {
            out[flat as usize] = (probs[i] / sum) as f32;
        }
        out
    }

    /// Pick the best action at the root: argmax of child_N, tie-broken by
    /// the PUCT action score (Python ``MCTSNode.best_child()``).
    pub fn pick_best_action(&self) -> PyResult<u32> {
        let root = &self.arena[self.root_idx];
        if root.legal_action_indices.is_empty() {
            return Err(PyRuntimeError::new_err(
                "pick_best_action: root has no legal actions",
            ));
        }
        let n_this = self.root_n;
        let c_puct = 2.0
            * ((1.0 + n_this + self.c_puct_base) / self.c_puct_base).ln()
            + 2.0 * self.c_puct_init;
        let n_s = (n_this - 1.0).max(1.0);
        let sqrt_n_s = n_s.sqrt();
        let ap = root.active_player_index;
        let mut best_i = 0usize;
        let mut best_key = (f32::NEG_INFINITY, f32::NEG_INFINITY);
        for i in 0..root.legal_action_indices.len() {
            let n_sa = root.child_n[i];
            let q_sa = root.child_w[i][ap] / (1.0 + n_sa);
            let u_sa = c_puct * root.child_prior[i] * sqrt_n_s / (1.0 + n_sa);
            let key0 = n_sa + (q_sa + u_sa) / 10000.0;
            let key = (key0, 0.0f32);
            if key.0 > best_key.0 {
                best_key = key;
                best_i = i;
            }
        }
        Ok(root.legal_action_indices[best_i])
    }

    /// Inject Dirichlet noise on the root's prior.
    pub fn inject_noise(&mut self, noise_weight: f32, concentration: f32) -> PyResult<()> {
        let root = &mut self.arena[self.root_idx];
        let num_legal = root.legal_action_indices.len();
        if num_legal == 0 || noise_weight <= 0.0 {
            return Ok(());
        }
        let alpha = (concentration as f64) / (num_legal.max(1) as f64);
        let noise: Vec<f64> = if num_legal == 1 {
            vec![1.0]
        } else {
            let mut rng = rand::rngs::StdRng::from_entropy();
            let dist = Dirichlet::new_with_size(alpha, num_legal).map_err(|e| {
                PyRuntimeError::new_err(format!("Dirichlet::new failed: {}", e))
            })?;
            dist.sample(&mut rng)
        };
        let w = noise_weight as f64;
        for i in 0..num_legal {
            let original = root.original_prior[i] as f64;
            let blended = original * (1.0 - w) + noise[i] * w;
            root.child_prior[i] = blended as f32;
        }
        Ok(())
    }

    /// Active player index at the root (used by check_resign + parity).
    fn root_active_player_index(&self) -> usize {
        self.arena[self.root_idx].active_player_index
    }

    /// Return the root's BaseGame as a Python object (a fresh clone — the
    /// arena retains its own copy). Mirrors ``get_game_for_idx(root_idx)``.
    fn root_game_object(&self, py: Python<'_>) -> PyResult<PyObject> {
        let g = self.arena[self.root_idx].game.clone_for_search();
        Ok(Py::new(py, g)?.into_any())
    }
}

impl RustMCTSPlayer {
    /// Internal: sample a price for a PW slot given the explicit action
    /// index, so we can resolve the price-head slot via the entity-key
    /// resolver. Falls back to a midpoint-Normal when no components or no
    /// slot mapping is available.
    fn _sample_price_with_index(
        &self,
        arena_idx: usize,
        action_index: u32,
        action_type: &str,
        price_range: (i64, i64),
    ) -> i64 {
        let (p_min, p_max) = price_range;
        if p_min == p_max {
            return p_min;
        }
        let midpoint = (p_min + p_max) as f32 / 2.0;
        let spread = (((p_max - p_min) as f32) / 4.0).max(1.0);
        let mut price_mean = midpoint;
        let mut price_log_std = spread.ln();

        if let Some(pc) = self.arena[arena_idx].price_components.as_ref() {
            if let Some(key) = price_head_entity_key(action_index, action_type) {
                if let Some(&slot) = pc.slot_index.get(&key) {
                    if slot < pc.price_mean.len() && slot < pc.price_log_std.len() {
                        price_mean = pc.price_mean[slot];
                        price_log_std = pc.price_log_std[slot];
                    }
                }
            }
        }
        sample_price_for_pw(price_mean, price_log_std, action_type, price_range)
    }

    /// Score: Q + U for each child compressed slot.
    fn argmax_action_score(&self, arena_idx: usize) -> usize {
        let node = &self.arena[arena_idx];
        let n_this = if arena_idx == self.root_idx {
            self.root_n
        } else {
            // Node's own N is stored on parent.
            let parent = node.parent.unwrap();
            self.arena[parent].child_n[node.parent_compressed_idx]
        };
        let c_puct = 2.0
            * ((1.0 + n_this + self.c_puct_base) / self.c_puct_base).ln()
            + 2.0 * self.c_puct_init;
        let n_s = (n_this - 1.0).max(1.0);
        let sqrt_n_s = n_s.sqrt();
        let ap = node.active_player_index;
        let mut best_i = 0usize;
        let mut best_score = f32::NEG_INFINITY;
        for i in 0..node.legal_action_indices.len() {
            let n_sa = node.child_n[i];
            let q_sa = node.child_w[i][ap] / (1.0 + n_sa);
            let u_sa = c_puct * node.child_prior[i] * sqrt_n_s / (1.0 + n_sa);
            let score = q_sa + u_sa;
            if score > best_score {
                best_score = score;
                best_i = i;
            }
        }
        best_i
    }
}

// ----------------------------------------------------------------------------
// Action application helper
// ----------------------------------------------------------------------------

/// Apply `flat_idx` to `game` via the Python ActionMapper. The mapper
/// produces a Python Action object that we convert to a dict and route
/// through the BaseGame's PyO3-exposed `process_action`.
fn apply_action(
    py: Python<'_>,
    game: &mut BaseGame,
    flat_idx: u32,
    sampled_price: Option<i64>,
) -> PyResult<()> {
    let cloned: BaseGame = game.clone_for_search();
    let py_game = Py::new(py, cloned)?;
    let py_game_obj: Bound<'_, PyAny> = py_game.into_bound(py).into_any();

    let rust_adapter_mod = py.import("rl18xx.rust_adapter")?;
    let adapter_cls = rust_adapter_mod.getattr("RustGameAdapter")?;
    let adapter = adapter_cls.call1((py_game_obj,))?;

    let dict_opt = index_to_action_dict(py, &adapter, flat_idx, sampled_price)?;
    let dict = dict_opt.ok_or_else(|| {
        PyRuntimeError::new_err(format!("index_to_action_dict returned None for idx={}", flat_idx))
    })?;

    game.process_action_dict_inner(py, &dict).map_err(|e| {
        PyRuntimeError::new_err(format!("process_action failed for idx={}: {}", flat_idx, e))
    })?;
    Ok(())
}

// ----------------------------------------------------------------------------
// Price-components decoder
// ----------------------------------------------------------------------------

/// Decode a Python dict {price_mean, price_log_std, slot_index, num_slots}
/// into the Rust ``PriceComponents`` struct. ``price_mean`` and
/// ``price_log_std`` should be numpy float32 arrays of length ``num_slots``.
/// ``slot_index`` keys are tuples of (action_type_str, (entity_key_parts...)).
fn decode_price_components(py: Python<'_>, dict: &Bound<'_, PyAny>) -> PyResult<PriceComponents> {
    let d: &Bound<'_, PyDict> = dict.downcast::<PyDict>().map_err(|_| {
        PyValueError::new_err("price_components must be a dict")
    })?;

    let means_obj = d.get_item("price_mean")?;
    let log_stds_obj = d.get_item("price_log_std")?;
    let slot_index_obj = d.get_item("slot_index")?;
    let num_slots_obj = d.get_item("num_slots")?;

    let means_obj = means_obj
        .ok_or_else(|| PyValueError::new_err("price_components missing price_mean"))?;
    let log_stds_obj = log_stds_obj
        .ok_or_else(|| PyValueError::new_err("price_components missing price_log_std"))?;
    let slot_index_obj = slot_index_obj
        .ok_or_else(|| PyValueError::new_err("price_components missing slot_index"))?;

    // ``price_mean`` / ``price_log_std`` may be numpy arrays or torch tensors.
    // Use numpy.asarray to normalize, then read as float32 vectors.
    let np = py.import("numpy")?;
    let asarray = np.getattr("asarray")?;
    let means_np = asarray.call1((means_obj, "float32"))?;
    let log_stds_np = asarray.call1((log_stds_obj, "float32"))?;

    // Use the numpy crate's PyReadonlyArray1 to extract slices.
    let means_arr: PyReadonlyArray1<'_, f32> = means_np.extract()?;
    let log_stds_arr: PyReadonlyArray1<'_, f32> = log_stds_np.extract()?;
    let price_mean: Vec<f32> = means_arr.as_slice()?.to_vec();
    let price_log_std: Vec<f32> = log_stds_arr.as_slice()?.to_vec();

    let num_slots = if let Some(n) = num_slots_obj {
        n.extract::<usize>().unwrap_or(price_mean.len())
    } else {
        price_mean.len()
    };

    // Decode slot_index: dict of (action_type_str, tuple_of_strs) -> int.
    // The Python side passes tuples-of-strings (e.g. ("SV",) or
    // ("B&O", "4")). We convert to Vec<String> for hashing.
    let mut slot_index: HashMap<(String, Vec<String>), usize> = HashMap::new();
    let si: &Bound<'_, PyDict> = slot_index_obj.downcast::<PyDict>().map_err(|_| {
        PyValueError::new_err("slot_index must be a dict")
    })?;
    for (key, value) in si.iter() {
        // Each key is a 2-tuple (action_type, entity_key_tuple).
        let key_tup = key.downcast::<pyo3::types::PyTuple>().map_err(|_| {
            PyValueError::new_err("slot_index key must be a 2-tuple")
        })?;
        if key_tup.len() != 2 {
            return Err(PyValueError::new_err("slot_index key must have 2 elements"));
        }
        let action_type: String = key_tup.get_item(0)?.extract()?;
        let entity_key_tup = key_tup.get_item(1)?;
        // entity_key may be a tuple of strings (typical) — convert each.
        let mut parts: Vec<String> = Vec::new();
        if let Ok(inner_tup) = entity_key_tup.downcast::<pyo3::types::PyTuple>() {
            for j in 0..inner_tup.len() {
                let s: String = inner_tup.get_item(j)?.extract()?;
                parts.push(s);
            }
        } else if let Ok(s) = entity_key_tup.extract::<String>() {
            parts.push(s);
        } else {
            // Unknown shape — best-effort string repr.
            parts.push(entity_key_tup.str()?.to_string());
        }
        let slot: usize = value.extract()?;
        slot_index.insert((action_type, parts), slot);
    }

    Ok(PriceComponents {
        price_mean,
        price_log_std,
        slot_index,
        num_slots,
    })
}
