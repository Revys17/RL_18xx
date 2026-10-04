//! Rust MCTS (Phase 4a scaffold + Phase 4c progressive widening).
//!
//! Arena-based MCTS tree mirroring the categorical-descent semantics of
//! Python `MCTSNode` / `MCTSPlayer` (rl18xx/agent/alphazero/mcts.py +
//! self_play.py).
//!
//! Phase 4c adds progressive widening (PW) with price grandchildren for the
//! price-bearing categorical slots (Bid / BuyTrain / BuyCompany); prices are
//! proposed from the price head's distribution over every legal price
//! (`price_pmf`). A "price grandchild" is a child of a
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
use rand::{Rng, SeedableRng};
use rand_distr::{Dirichlet, Distribution};

use crate::factored::LegalAction;
use crate::game::BaseGame;
use crate::price_pmf::{proposal_order, PriceCells, NUM_CELLS};

/// Value-head width — single-sourced from the encoder's player-slot cap.
const VALUE_SIZE: usize = crate::encoder::MAX_PLAYERS;
const C_PUCT_BASE: f32 = 19652.0;
const C_PUCT_INIT: f32 = 1.25;

// ----------------------------------------------------------------------------
// Price components (per-leaf NN price-head output)
// ----------------------------------------------------------------------------

/// Per-leaf price-head output sliced from the model's batched
/// ``last_price_components`` dict: ``NUM_CELLS`` cell logits per slot,
/// row-major, with slot rows located via ``slot_index``.
#[derive(Debug, Clone, Default)]
pub struct PriceComponents {
    pub price_logits: Vec<f32>,
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
    /// Virtual losses in flight through each child (``mean_q`` uses them to
    /// count an in-flight descent as a zero-value visit).
    pub child_vl: Vec<f32>,
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

    /// Price-head output stashed at incorporate-results time. Consulted when
    /// widening this node's price-bearing slots.
    pub price_components: Option<PriceComponents>,
    /// Per price-bearing slot: the cells PW will still materialize, in order
    /// (``price_pmf::proposal_order``), built on the slot's first widening.
    pub price_proposals: HashMap<u32, Vec<usize>>,
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
    // and `action_mapper.py::_PRICE_HEAD_*`. Order matches Python; the title
    // lists (company / corporation / train-type order) come from the shared
    // action layout rather than duplicated literals.
    //
    // Re-fetch the layout offsets via the Rust action_index module. The
    // layout is global (memoized once); the lookup is cheap.
    let lo = crate::action_index::layout();
    let companies = &lo.company_offsets;
    let corporations = &lo.corporation_offsets;
    let train_types = &lo.train_type_offsets;
    let bid_start = *lo.action_offsets.get("Bid")? as i64;
    let buy_train_start = *lo.action_offsets.get("BuyTrain")? as i64;
    let buy_company_start = *lo.action_offsets.get("BuyCompany")? as i64;

    let idx = action_index as i64;

    if action_type == "Bid" {
        let bid_end = bid_start + companies.len() as i64;
        if idx >= bid_start && idx < bid_end {
            let i = (idx - bid_start) as usize;
            return Some(("Bid".to_string(), vec![companies[i].to_string()]));
        }
        return None;
    }

    if action_type == "BuyTrain" {
        // First slot is depot; next ``train_type_offsets.len()`` are market
        // discarded — both fixed-price. The cross-corp block follows.
        let train_price_count = lo.train_price_offsets.len() as i64;
        let train_type_count = train_types.len() as i64;
        let cross_corp_start = buy_train_start + 1 + train_type_count;
        let per_corp = train_type_count * train_price_count;
        let cross_corp_end = cross_corp_start + (corporations.len() as i64) * per_corp;
        if idx >= cross_corp_start && idx < cross_corp_end {
            let rel = idx - cross_corp_start;
            let corp_idx = (rel / per_corp) as usize;
            let train_idx = ((rel % per_corp) / train_price_count) as usize;
            return Some((
                "BuyTrain".to_string(),
                vec![
                    corporations[corp_idx].to_string(),
                    train_types[train_idx].to_string(),
                ],
            ));
        }
        return None;
    }

    if action_type == "BuyCompany" {
        let price_count = lo.buy_company_price_offsets.len() as i64;
        let block_end = buy_company_start + (companies.len() as i64) * price_count;
        if idx >= buy_company_start && idx < block_end {
            let rel = idx - buy_company_start;
            let company_idx = (rel / price_count) as usize;
            return Some((
                "BuyCompany".to_string(),
                vec![companies[company_idx].to_string()],
            ));
        }
        return None;
    }

    None
}

// ----------------------------------------------------------------------------
// Price helpers
// ----------------------------------------------------------------------------

/// The auction bid ladder step for the price cells, from the title data
/// (1830: $5). The MCTS runs against the global 1830 layout today; a
/// per-title MCTS threads its own layout here.
fn bid_step() -> i64 {
    crate::title::resolve(crate::action_index::layout().title_name).bid_price_step()
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
    /// Exploration floor mixed into price proposals (``price_explore_eps``).
    pub price_explore_eps: f32,

    /// Per-round-type c_puct_init override (Python ``config.c_puct_by_round``,
    /// keyed by the round class name "Auction"/"Stock"/"Operating"). Falls
    /// back to ``c_puct_init`` for unknown round names.
    pub c_puct_by_round: HashMap<String, f32>,
    /// Forced-chain guard (Python ``config.max_game_length``): the collapse
    /// loop stops once the engine move number (action-log length) reaches
    /// this, exactly like Python's ``while not (finished or move_number >=
    /// max_game_length)``.
    pub max_game_length: usize,
    /// Selection Q. false (default): minigo's ``W / (1 + N)`` with unvisited
    /// children at 0 and a virtual loss of -1 to W. true: the mean value
    /// ``W / N``, an in-flight descent counted as a zero-value visit, and
    /// unvisited children at the node's own mean value minus
    /// ``fpu_reduction * sqrt(visited prior mass)`` (first-play urgency).
    pub mean_q: bool,
    pub fpu_reduction: f32,
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
            child_vl: vec![0.0; num_legal],
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
            price_proposals: HashMap::new(),
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
    #[pyo3(signature = (game_obj, pw_c=None, pw_alpha=None, min_price_children=None, price_explore_eps=None))]
    pub fn new(
        py: Python<'_>,
        game_obj: PyObject,
        pw_c: Option<f32>,
        pw_alpha: Option<f32>,
        min_price_children: Option<usize>,
        price_explore_eps: Option<f32>,
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
            price_explore_eps: price_explore_eps.unwrap_or(0.05),
            c_puct_by_round: HashMap::new(),
            max_game_length: 1000,
            mean_q: false,
            fpu_reduction: 0.0,
        };
        let root = player.build_node(cloned_inner, None, None)?;
        player.num_players = root.num_players;
        player.arena.push(root);
        player.root_idx = 0;
        Ok(player)
    }

    /// Set PW knobs after construction (mirrors a config update in Python).
    #[pyo3(signature = (pw_c, pw_alpha, min_price_children, price_explore_eps=None))]
    pub fn set_pw_config(&mut self, pw_c: f32, pw_alpha: f32, min_price_children: usize, price_explore_eps: Option<f32>) {
        self.pw_c = pw_c;
        self.pw_alpha = pw_alpha;
        self.min_price_children = min_price_children;
        if let Some(eps) = price_explore_eps {
            self.price_explore_eps = eps;
        }
    }

    /// Set the PUCT / forced-chain knobs from the Python ``SelfPlayConfig``
    /// (c_puct_init, c_puct_base, per-round c_puct_init overrides, and the
    /// forced-chain ``max_game_length`` guard).
    pub fn set_search_config(
        &mut self,
        c_puct_init: f32,
        c_puct_base: f32,
        c_puct_by_round: HashMap<String, f32>,
        max_game_length: usize,
    ) {
        self.c_puct_init = c_puct_init;
        self.c_puct_base = c_puct_base;
        self.c_puct_by_round = c_puct_by_round;
        self.max_game_length = max_game_length;
    }

    /// Selection-Q mode (see the ``mean_q`` field).
    pub fn set_q_config(&mut self, mean_q: bool, fpu_reduction: f32) {
        self.mean_q = mean_q;
        self.fpu_reduction = fpu_reduction;
    }

    /// Engine move number at the root (length of the full action log) —
    /// Python's ``game_object.move_number`` (= ``len(raw_actions)``). Used by
    /// the Python wrapper for softpick / temperature cutoffs so both players
    /// key off the same counter (which includes forced-chain actions).
    pub fn root_move_number(&self) -> usize {
        self.arena[self.root_idx].game.action_log.len()
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

    /// Full-size visit-count vector at the root (length = the layout's
    /// policy size). Non-legal slots stay 0.
    fn child_n_at_root(&self) -> Vec<f32> {
        let root = &self.arena[self.root_idx];
        let mut out = vec![0.0f32; crate::action_index::layout().total as usize];
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
    pub fn select_leaf(&mut self) -> PyResult<usize> {
        let (leaf, _, _, _) = self._select_leaf_inner(false)?;
        Ok(leaf)
    }

    /// Tracing variant of ``select_leaf`` — same PUCT/PW descent, but also
    /// returns the descent path so callers can populate a Phase 1
    /// ``PlayoutTrace``. Returns ``(leaf_idx, action_path, pw_grandchild_path,
    /// forced_chain_lengths)``. Trace vectors are parallel: index ``i`` is the
    /// step from depth ``i`` to depth ``i+1``.
    pub fn select_leaf_with_trace(
        &mut self,
    ) -> PyResult<(usize, Vec<u32>, Vec<bool>, Vec<u32>)> {
        self._select_leaf_inner(true)
    }

    fn _select_leaf_inner(
        &mut self,
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
                self._select_or_expand_price_child(current, best_move)?
            } else {
                self.maybe_add_child(current, best_move, None)?
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
    /// N^pw_alpha))``, expand a grandchild at a uniform price inside the next
    /// not-yet-expanded cell of the slot's proposal order (mirrors Python
    /// ``_select_or_expand_price_child``). Otherwise — or once every cell has
    /// a grandchild — PUCT-select among the existing grandchildren with the
    /// price head's cell probabilities as priors.
    pub fn _select_or_expand_price_child(
        &mut self,
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

        let action_type = self.arena[arena_idx]
            .action_types
            .get(&action_index)
            .cloned()
            .unwrap_or_default();
        let price_range = *self.arena[arena_idx]
            .price_ranges
            .get(&action_index)
            .ok_or_else(|| PyRuntimeError::new_err("price_range missing on PW slot"))?;
        if existing_count < target {
            if let Some(price) = self.next_proposed_price(arena_idx, action_index, &action_type, price_range) {
                return self.maybe_add_child(arena_idx, action_index, Some(price));
            }
        }

        // PW cap reached: PUCT among existing grandchildren, each with the
        // price head's probability of its cell (renormalized over the
        // expanded grandchildren) as its prior.
        let ap = self.arena[arena_idx].active_player_index;
        let cell_mass: HashMap<i64, f32> = match self.slot_cells(arena_idx, action_index, &action_type, price_range) {
            Some((cells, probs)) => self.arena[arena_idx]
                .price_children
                .get(&action_index)
                .map(|m| m.keys().map(|&p| (p, cells.cell_of(p).map_or(0.0, |c| probs[c] as f32))).collect())
                .unwrap_or_default(),
            None => HashMap::new(),
        };
        let total_mass: f32 = cell_mass.values().sum::<f32>();
        let total_mass = if total_mass > 0.0 { total_mass } else { 1.0 };

        let c_puct = 2.0
            * ((1.0 + slot_visits + self.c_puct_base) / self.c_puct_base).ln()
            + 2.0 * self.c_puct_init_for(arena_idx);
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
            let prior = cell_mass.get(&price).copied().unwrap_or(0.0) / total_mass;
            let u_sa = c_puct * prior * sqrt_n_s / (1.0 + n_sa);
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
    ///   - ``None`` for a PW slot: the slot's next proposed price (a draw
    ///     from the price head once every cell is expanded).
    ///   - ``Some(p)`` for a PW slot: clamped to the legal range and used —
    ///     any legal price is playable.
    #[pyo3(signature = (arena_idx, action_index, price=None))]
    pub fn maybe_add_child(
        &mut self,
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

        // Resolve the price for PW slots; existing-grandchild fast path lives
        // below the action_type lookup.
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
                Some(p) => p,
                None => match self.next_proposed_price(arena_idx, action_index, &action_type, price_range) {
                    Some(p) => p,
                    None => self.sample_price_from_head(arena_idx, action_index, &action_type, price_range),
                },
            }
            .clamp(price_range.0, price_range.1);
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

        // Clone the parent's game and apply the action via the native decode.
        let mut new_game = self.arena[arena_idx].game.clone_for_search();
        apply_action(&mut new_game, action_index, expansion_price_for_apply)?;

        // Forced-chain collapse: while exactly one legal action, apply it.
        // Mirrors Python's loop guard ``while not (finished or move_number >=
        // max_game_length)`` — move_number is the action-log length.
        let mut forced_chain: Vec<u32> = Vec::new();
        loop {
            if new_game.is_finished_pub() || new_game.action_log.len() >= self.max_game_length {
                break;
            }
            // Enumerate factored choices, dedupe to flat indices.
            let choices = new_game.get_factored_choices_impl();
            let mut flat_indices: Vec<u32> = Vec::new();
            let mut seen: std::collections::HashSet<u32> = std::collections::HashSet::new();
            let mut forced_price_ranges: HashMap<u32, (i64, i64)> = HashMap::new();
            let mut forced_action_types: HashMap<u32, String> = HashMap::new();
            for la in &choices {
                if let Some(idx) = crate::action_index::legal_action_to_index(la) {
                    if seen.insert(idx) {
                        flat_indices.push(idx);
                    }
                    if let Some(pr) = la.price_range {
                        let existing = forced_price_ranges.entry(idx).or_insert(pr);
                        *existing = (existing.0.min(pr.0), existing.1.max(pr.1));
                    }
                    forced_action_types
                        .entry(idx)
                        .or_insert_with(|| la.action_type.clone());
                }
            }
            if flat_indices.len() != 1 {
                break;
            }
            let forced_idx = flat_indices[0];
            // Python's forced loop draws a price from the price head for
            // non-degenerate ranges (reading this node's stashed
            // price_components); degenerate ranges apply the engine minimum.
            let forced_price = match forced_price_ranges.get(&forced_idx) {
                Some(&(lo, hi)) if lo != hi => {
                    let at = forced_action_types
                        .get(&forced_idx)
                        .cloned()
                        .unwrap_or_default();
                    Some(self.sample_price_from_head(arena_idx, forced_idx, &at, (lo, hi)))
                }
                Some(&(lo, _)) => Some(lo),
                None => None,
            };
            apply_action(&mut new_game, forced_idx, forced_price)?;
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
            self.arena[parent].child_vl[pidx] += 1.0;
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
            self.arena[parent].child_vl[pidx] -= 1.0;
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
    ///   - ``price_logits`` (np.float32, num_slots x NUM_CELLS, flat or 2-D)
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
        let policy_size = crate::action_index::layout().total;
        if probs_slice.len() != policy_size as usize {
            return Err(PyValueError::new_err(format!(
                "probs length {} != policy size {}",
                probs_slice.len(),
                policy_size
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

    /// Arena index of the current search root. `advance_root` moves it to the
    /// chosen child; arena slot 0 stays the position the player started from.
    #[getter(root_idx)]
    fn get_root_idx(&self) -> usize {
        self.root_idx
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
    pub fn advance_root(&mut self, action_index: u32) -> PyResult<()> {
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
                    None => self.maybe_add_child(self.root_idx, action_index, Some(price))?,
                }
            } else {
                // No price-grandchild was ever expanded for this PW slot at
                // commit time: commit the slot's first proposal — a price in
                // the price head's most likely cell (Python's
                // ``_select_most_visited_price_grandchild`` →
                // ``maybe_add_child(action_index)``). The range minimum would
                // be the wrong default for e.g. BuyCompany, which humans
                // play at the maximum 94% of the time.
                self.maybe_add_child(self.root_idx, action_index, None)?
            }
        } else {
            self.maybe_add_child(self.root_idx, action_index, None)?
        };
        // Rebase the root tally from the committed child's OWN stats. For a
        // PW price grandchild that is the per-price entry — the categorical
        // slot aggregates ALL price siblings and would inflate n_at_root and
        // mix sibling values into root_q_vector (Python's root-as-grandchild
        // reads parent.price_child_N/W[fmove][price]).
        let pidx = self.arena[child_idx].parent_compressed_idx;
        let (new_n, new_w) = if self.arena[child_idx].is_price_grandchild {
            let fmove = self.arena[child_idx].fmove.unwrap();
            let sp = self.arena[child_idx].sampled_price.unwrap();
            (
                self.arena[self.root_idx]
                    .price_child_n
                    .get(&fmove)
                    .and_then(|m| m.get(&sp))
                    .copied()
                    .unwrap_or(0.0),
                self.arena[self.root_idx]
                    .price_child_w
                    .get(&fmove)
                    .and_then(|m| m.get(&sp))
                    .copied()
                    .unwrap_or([0.0; VALUE_SIZE]),
            )
        } else {
            (
                self.arena[self.root_idx].child_n[pidx],
                self.arena[self.root_idx].child_w[pidx],
            )
        };
        self.root_idx = child_idx;
        self.root_n = new_n;
        self.root_w = new_w;
        // The new root is no longer a "child" of anyone — sever the parent
        // backlink so future virtual-loss / backup walks stop here.
        self.arena[self.root_idx].parent = None;
        // Also clear the price-grandchild flag — it should not mirror into
        // a (now-gone) parent slot during subsequent backups.
        self.arena[self.root_idx].is_price_grandchild = false;
        self.compact_to_root();
        Ok(())
    }

    /// Compute a policy-size-length visit-count policy at the root with
    /// temperature scaling. Mirrors Python ``MCTSNode.children_as_pi``.
    pub fn pi_at_root(&self, temperature: f32) -> Vec<f32> {
        let root = &self.arena[self.root_idx];
        let mut out = vec![0.0f32; crate::action_index::layout().total as usize];
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
            + 2.0 * self.c_puct_init_for(self.root_idx);
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
    /// Drop every arena node outside the current root's subtree and renumber
    /// the survivors, root first (root_idx becomes 0). Without this the arena
    /// keeps every node created during the game — each holding a full game
    /// clone — and a self-play game grew to ~18 GB before the OOM killer
    /// stepped in. Only called between searches (no leaf indices in flight).
    fn compact_to_root(&mut self) {
        let old_len = self.arena.len();
        let mut new_index = vec![usize::MAX; old_len];
        let mut order: Vec<usize> = Vec::new();
        let mut stack = vec![self.root_idx];
        while let Some(i) = stack.pop() {
            if new_index[i] != usize::MAX {
                continue;
            }
            new_index[i] = order.len();
            order.push(i);
            let node = &self.arena[i];
            stack.extend(node.children.values().copied());
            for by_price in node.price_children.values() {
                stack.extend(by_price.values().copied());
            }
        }
        if self.root_idx == 0 && order.len() == old_len {
            return;
        }
        let mut old: Vec<Option<RustMCTSNode>> = std::mem::take(&mut self.arena).into_iter().map(Some).collect();
        let mut arena = Vec::with_capacity(order.len());
        for &i in &order {
            let mut node = old[i].take().expect("arena node visited twice");
            node.parent = node.parent.and_then(|p| (new_index[p] != usize::MAX).then(|| new_index[p]));
            for child in node.children.values_mut() {
                *child = new_index[*child];
            }
            for by_price in node.price_children.values_mut() {
                for child in by_price.values_mut() {
                    *child = new_index[*child];
                }
            }
            arena.push(node);
        }
        self.arena = arena;
        self.root_idx = 0;
    }

    /// The price head's ``NUM_CELLS`` logits for ``action_index``'s slot at
    /// ``arena_idx``, if the node has price components and the slot maps.
    fn slot_logits(&self, arena_idx: usize, action_index: u32, action_type: &str) -> Option<&[f32]> {
        let pc = self.arena[arena_idx].price_components.as_ref()?;
        let key = price_head_entity_key(action_index, action_type)?;
        let &slot = pc.slot_index.get(&key)?;
        pc.price_logits.get(slot * NUM_CELLS..(slot + 1) * NUM_CELLS)
    }

    /// ``(PriceCells, cell probabilities)`` of a price-bearing slot — mirrors
    /// Python ``_slot_cell_probs``. Without logits (or with non-finite ones)
    /// the cells are uniform.
    fn slot_cells(
        &self,
        arena_idx: usize,
        action_index: u32,
        action_type: &str,
        price_range: (i64, i64),
    ) -> Option<(PriceCells, [f64; NUM_CELLS])> {
        let cells = PriceCells::new(action_type, price_range.0, price_range.1, bid_step())?;
        let probs = cells.cell_probs(self.slot_logits(arena_idx, action_index, action_type));
        Some((cells, probs))
    }

    /// The slot's next PW price: a uniform price inside the next cell of its
    /// proposal order without a grandchild yet, or ``None`` once every
    /// non-empty cell has one — mirrors Python ``_next_proposed_price``.
    fn next_proposed_price(
        &mut self,
        arena_idx: usize,
        action_index: u32,
        action_type: &str,
        price_range: (i64, i64),
    ) -> Option<i64> {
        let (cells, probs) = self.slot_cells(arena_idx, action_index, action_type, price_range)?;
        let mut rng = rand::thread_rng();
        let eps = self.price_explore_eps as f64;
        let taken: std::collections::HashSet<usize> = self.arena[arena_idx]
            .price_children
            .get(&action_index)
            .map(|m| m.keys().filter_map(|&p| cells.cell_of(p)).collect())
            .unwrap_or_default();
        let order = self.arena[arena_idx]
            .price_proposals
            .entry(action_index)
            .or_insert_with(|| proposal_order(&probs, eps, &mut rng));
        while !order.is_empty() {
            let cell = order.remove(0);
            if !taken.contains(&cell) {
                return Some(cells.sample_price(cell, &mut rng));
            }
        }
        None
    }

    /// One draw from the price head's distribution for a slot, with the
    /// exploration floor mixed in: a cell, then a uniform price inside it.
    /// Used for forced-chain prices and fully-expanded slots — mirrors
    /// Python ``_sample_price_for_slot``.
    fn sample_price_from_head(
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
        let (cells, probs) = match self.slot_cells(arena_idx, action_index, action_type, price_range) {
            Some(x) => x,
            None => return p_min,
        };
        let eps = self.price_explore_eps as f64;
        let live = (0..NUM_CELLS).filter(|&c| cells.nonempty(c)).count() as f64;
        let mut rng = rand::thread_rng();
        let mut u = rng.gen::<f64>();
        let mut chosen = None;
        for c in (0..NUM_CELLS).filter(|&c| cells.nonempty(c)) {
            chosen = Some(c);
            let mass = (1.0 - eps) * probs[c] + eps / live;
            if u < mass {
                break;
            }
            u -= mass;
        }
        chosen.map_or(p_min, |c| cells.sample_price(c, &mut rng))
    }

    /// A node's own visit count — Python ``MCTSNode.N``. The root reads its
    /// dedicated tally; a PW price grandchild reads its per-price entry in
    /// the parent's ``price_child_n`` (NOT the categorical slot, which
    /// aggregates all price siblings); a regular child reads the parent's
    /// compressed slot.
    fn node_n(&self, arena_idx: usize) -> f32 {
        if arena_idx == self.root_idx {
            return self.root_n;
        }
        let node = &self.arena[arena_idx];
        let parent = node.parent.unwrap();
        if node.is_price_grandchild {
            let fmove = node.fmove.unwrap();
            let sp = node.sampled_price.unwrap();
            self.arena[parent]
                .price_child_n
                .get(&fmove)
                .and_then(|m| m.get(&sp))
                .copied()
                .unwrap_or(0.0)
        } else {
            self.arena[parent].child_n[node.parent_compressed_idx]
        }
    }

    /// Mean backed-up value of ``arena_idx`` for ``player`` (in-flight
    /// virtual losses excluded); ``1 / num_players`` before any backup.
    fn node_mean_value(&self, arena_idx: usize, player: usize) -> f32 {
        let fallback = 1.0 / self.num_players.max(1) as f32;
        if arena_idx == self.root_idx {
            return if self.root_n > 0.0 { self.root_w[player] / self.root_n } else { fallback };
        }
        let node = &self.arena[arena_idx];
        let parent = &self.arena[node.parent.unwrap()];
        let pidx = node.parent_compressed_idx;
        let (n, mut w) = (parent.child_n[pidx], parent.child_w[pidx][player]);
        if player == parent.active_player_index {
            w += parent.child_vl[pidx];
        }
        if n > 0.0 {
            w / n
        } else {
            fallback
        }
    }

    /// c_puct_init for the node's round type — Python's
    /// ``config.c_puct_by_round.get(round_name, config.c_puct_init)`` keyed
    /// by the round class name.
    fn c_puct_init_for(&self, arena_idx: usize) -> f32 {
        let name = match &self.arena[arena_idx].game.round {
            crate::rounds::Round::Auction(_) => "Auction",
            crate::rounds::Round::Stock(_) => "Stock",
            crate::rounds::Round::Operating(_) => "Operating",
            // 1867-only; MCTS is 1830-pinned, but the key falls through to
            // the default c_puct anyway.
            crate::rounds::Round::Merger(_) => "Merger",
        };
        self.c_puct_by_round
            .get(name)
            .copied()
            .unwrap_or(self.c_puct_init)
    }

    /// Score: Q + U for each child compressed slot, with Python's top-k
    /// categorical progressive widening at wide nodes (``select_leaf``:
    /// when >20 legal actions, restrict to the k = max(1, pw_c * N^pw_alpha)
    /// highest-prior slots).
    fn argmax_action_score(&self, arena_idx: usize) -> usize {
        let node = &self.arena[arena_idx];
        let n_this = self.node_n(arena_idx);
        let c_puct = 2.0
            * ((1.0 + n_this + self.c_puct_base) / self.c_puct_base).ln()
            + 2.0 * self.c_puct_init_for(arena_idx);
        let n_s = (n_this - 1.0).max(1.0);
        let sqrt_n_s = n_s.sqrt();
        let ap = node.active_player_index;
        let num_legal = node.legal_action_indices.len();

        // Top-k categorical PW mask (by current — possibly noised — prior).
        let allowed: Option<Vec<bool>> = if num_legal > 20 {
            let k = ((self.pw_c * n_this.powf(self.pw_alpha)) as usize).max(1);
            if k < num_legal {
                let mut order: Vec<usize> = (0..num_legal).collect();
                order.sort_unstable_by(|&a, &b| {
                    node.child_prior[b]
                        .partial_cmp(&node.child_prior[a])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                let mut mask = vec![false; num_legal];
                for &i in order.iter().take(k) {
                    mask[i] = true;
                }
                Some(mask)
            } else {
                None
            }
        } else {
            None
        };

        let mut best_i: Option<usize> = None;
        let mut best_score = f32::NEG_INFINITY;
        // First-play urgency (Leela/KataGo): an unvisited child starts at the
        // node's own mean value, lowered by fpu_reduction * sqrt(prior mass
        // of the children already visited) -- the more of the policy the
        // search has explored, the less it expects from what is left.
        let fpu = if self.mean_q {
            let visited_mass: f32 = (0..num_legal)
                .filter(|&i| node.child_n[i] + node.child_vl[i] > 0.0)
                .map(|i| node.child_prior[i])
                .sum();
            self.node_mean_value(arena_idx, ap) - self.fpu_reduction * visited_mass.sqrt()
        } else {
            0.0
        };
        for i in 0..num_legal {
            if let Some(ref mask) = allowed {
                if !mask[i] {
                    continue;
                }
            }
            let n_sa = node.child_n[i];
            let q_sa = if self.mean_q {
                // Undo the -1 virtual losses and count each as a 0-value visit.
                let vl = node.child_vl[i];
                if n_sa + vl > 0.0 {
                    (node.child_w[i][ap] + vl) / (n_sa + vl)
                } else {
                    fpu
                }
            } else {
                node.child_w[i][ap] / (1.0 + n_sa)
            };
            let u_sa = c_puct * node.child_prior[i] * sqrt_n_s / (1.0 + n_sa);
            let score = q_sa + u_sa;
            if best_i.is_none() || score > best_score {
                best_score = score;
                best_i = Some(i);
            }
        }
        best_i.unwrap_or(0)
    }
}

// ----------------------------------------------------------------------------
// Action application helper
// ----------------------------------------------------------------------------

/// Decode `flat_idx` to a concrete `Action` and apply it natively. Pure Rust —
/// no Python round-trip — so it needs no GIL token.
fn apply_action(
    game: &mut BaseGame,
    flat_idx: u32,
    sampled_price: Option<i64>,
) -> PyResult<()> {
    // Native decode + apply — no Python ActionMapper round-trip. The decode is
    // verified behaviorally-equivalent to Python's map_index_to_action (see
    // tests/decode_parity_check.py + the strict lockstep's decode_diff), and
    // `process_action_native` logs a faithful, replayable dict to `raw_actions`
    // so the driver's forced-chain recovery / extract_data replay still work.
    let action = game
        .decode_index(flat_idx, sampled_price)
        .map_err(|e| PyRuntimeError::new_err(format!("decode_index failed for idx={}: {}", flat_idx, e)))?;
    game.process_action_native(&action).map_err(|e| {
        PyRuntimeError::new_err(format!("process_action failed for idx={}: {}", flat_idx, e))
    })?;
    Ok(())
}

// ----------------------------------------------------------------------------
// Price-components decoder
// ----------------------------------------------------------------------------

/// Decode a Python dict {price_logits, slot_index, num_slots} into the Rust
/// ``PriceComponents`` struct. ``price_logits`` is a numpy float32 array (or
/// torch tensor) of ``num_slots x NUM_CELLS`` cell logits, flat or 2-D.
/// ``slot_index`` keys are tuples of (action_type_str, (entity_key_parts...)).
fn decode_price_components(py: Python<'_>, dict: &Bound<'_, PyAny>) -> PyResult<PriceComponents> {
    let d: &Bound<'_, PyDict> = dict.downcast::<PyDict>().map_err(|_| {
        PyValueError::new_err("price_components must be a dict")
    })?;

    let logits_obj = d.get_item("price_logits")?;
    let slot_index_obj = d.get_item("slot_index")?;
    let num_slots_obj = d.get_item("num_slots")?;

    let logits_obj = logits_obj
        .ok_or_else(|| PyValueError::new_err("price_components missing price_logits"))?;
    let slot_index_obj = slot_index_obj
        .ok_or_else(|| PyValueError::new_err("price_components missing slot_index"))?;

    // ``price_logits`` may be a numpy array or torch tensor, (num_slots,
    // NUM_CELLS) or already flat: normalize to a flat float32 vector.
    let np = py.import("numpy")?;
    let flat = np.getattr("ravel")?.call1((np.getattr("asarray")?.call1((logits_obj, "float32"))?,))?;
    let logits_arr: PyReadonlyArray1<'_, f32> = flat.extract()?;
    let price_logits: Vec<f32> = logits_arr.as_slice()?.to_vec();

    let num_slots = match num_slots_obj {
        Some(n) => n.extract::<usize>().unwrap_or(price_logits.len() / NUM_CELLS),
        None => price_logits.len() / NUM_CELLS,
    };
    if price_logits.len() < num_slots * NUM_CELLS {
        return Err(PyValueError::new_err(format!(
            "price_logits has {} values, expected {} slots x {} cells",
            price_logits.len(),
            num_slots,
            NUM_CELLS
        )));
    }

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
        price_logits,
        slot_index,
        num_slots,
    })
}
