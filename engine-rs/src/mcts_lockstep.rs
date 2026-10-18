//! Many search trees in lockstep: one network request for the leaves of
//! several trees, and no Python per leaf.
//!
//! ``RustMCTSPlayer.tree_search`` (Python) selects a batch of leaves from one
//! tree, wraps each leaf's game in a ``RustGameAdapter`` and encodes it in
//! Python (``mcts._rust_encode``), runs the network, and hands each leaf a
//! dense policy-size prior vector. Here a worker gives
//! ``mcts_collect_leaves`` all its trees that are searching: each selects its
//! batch exactly as ``tree_search`` does (virtual losses, terminal leaves
//! backed up on the spot, ``dedupe_leaves``, the same attempt cap) and every
//! leaf is encoded straight into one stacked array the model takes, with its
//! legal moves. ``mcts_apply_results`` takes the network's outputs for those
//! rows -- the legal priors (the inference server's sparse reply), canonical
//! values and price-head rows -- and scatters them back: virtual losses
//! reverted, priors tempered (``prior_temperature``), values unrotated, leaves
//! expanded and backed up, as ``tree_search`` + ``incorporate_results`` do.
//! ``rl18xx/agent/alphazero/lockstep.py`` drives it.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::encoder::{spec_for, MAX_PLAYERS};
use crate::game::BaseGame;
use crate::mcts::{decode_slot_index, PriceComponents, RustMCTSPlayer};
use crate::price_pmf::NUM_CELLS;

const VALUE_SIZE: usize = MAX_PLAYERS;

/// The value a finished game backs up, in absolute seat order -- Python
/// ``RustMCTSPlayer._compute_terminal_value`` followed (with
/// ``use_score_values``) by ``mcts.terminal_backup_value``: a share of the
/// win split among the net-worth leaders; without it minigo's +1 / -1 (0 for
/// each of several tied leaders, -1 in the padded seats).
pub(crate) fn terminal_value(game: &BaseGame, use_score_values: bool) -> [f32; VALUE_SIZE] {
    let results = game.calculate_results();
    let mut ids: Vec<u32> = results.keys().copied().collect();
    ids.sort_unstable();
    let scores: Vec<f32> = ids.iter().map(|id| results[id] as f32).collect();
    let n = scores.len().min(VALUE_SIZE);
    if !use_score_values {
        let mut wl = [-1.0f32; VALUE_SIZE];
        if n == 0 {
            return wl;
        }
        let best = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let winners: Vec<usize> = (0..n).filter(|&i| scores[i] == best).collect();
        let mark = if winners.len() > 1 { 0.0 } else { 1.0 };
        for i in winners {
            wl[i] = mark;
        }
        return wl;
    }
    let mut out = [0.0f32; VALUE_SIZE];
    if n == 0 {
        return out;
    }
    let total: f32 = scores.iter().sum();
    let shares: Vec<f32> = if total > 0.0 {
        scores.iter().map(|&s| s / total).collect()
    } else {
        vec![1.0 / n as f32; n]
    };
    let best = shares.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let winners: Vec<bool> = shares.iter().map(|&s| s >= best - 1e-6).collect();
    let count = winners.iter().filter(|&&w| w).count() as f32;
    for i in 0..n {
        out[i] = if winners[i] { 1.0 / count } else { 0.0 };
    }
    out
}

impl RustMCTSPlayer {
    /// Select this tree's next batch of leaves (``tree_search``'s loop) into
    /// ``pending`` / ``pending_dups``; returns how many terminal leaves were
    /// backed up on the spot. A root not yet expanded is the batch's only
    /// leaf (the callers' first-expansion evaluation).
    pub(crate) fn collect_pending(&mut self, max_leaves: usize, dedupe: bool, use_score_values: bool) -> PyResult<usize> {
        if !self.pending.is_empty() || !self.pending_dups.is_empty() {
            return Err(PyRuntimeError::new_err(
                "mcts_collect_leaves: this tree's previous batch was not applied (mcts_apply_results)",
            ));
        }
        let root = self.root_idx;
        if self.arena[root].is_terminal {
            return Ok(0);
        }
        if !self.arena[root].is_expanded {
            let rotation = self.arena[root].active_player_index;
            self.pending.push((root, rotation));
            return Ok(0);
        }
        let gumbel = self.gumbel_search.as_ref().is_some_and(|gs| gs.root_idx == root);
        let mut terminal = 0usize;
        let mut failsafe = 0usize;
        let max_attempts = max_leaves * 2;
        let mut seen: HashSet<usize> = HashSet::new();
        while self.pending.len() < max_leaves && failsafe < max_attempts {
            // A Gumbel search backs up exactly its budget.
            if gumbel && self.gumbel_remaining_inner() == 0 {
                break;
            }
            failsafe += 1;
            let idx = self.select_leaf()?;
            if self.arena[idx].is_terminal {
                let value = terminal_value(&self.arena[idx].game, use_score_values);
                self.backup(idx, value);
                terminal += 1;
                continue;
            }
            self.add_virtual_loss(idx);
            if dedupe && !seen.insert(idx) {
                if gumbel {
                    self.gumbel_note_duplicate(idx, 1.0);
                }
                self.pending_dups.push(idx);
                continue;
            }
            let rotation = self.arena[idx].active_player_index;
            self.pending.push((idx, rotation));
        }
        Ok(terminal)
    }
}

/// A hash of a node-feature block's bit patterns (equal blocks hash equal;
/// a hit is confirmed by comparing the bits, so the quality only matters for
/// speed).
fn map_hash(features: &[f32]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325 ^ features.len() as u64;
    for &x in features {
        h = (h ^ x.to_bits() as u64).wrapping_mul(0x0000_0100_0000_01b3).rotate_left(5);
    }
    h
}

/// Stacked network inputs for a list of positions, as numpy arrays. With
/// ``dedupe_maps``, a position whose node features (the hex map -- the map
/// encoder's only input) are bit-for-bit an earlier row's stores no copy:
/// ``node_features`` holds each distinct map once, in order of first
/// appearance, and ``map_index`` gives each row's map. A search's leaves
/// mostly share their tree's map (all of them in a stock round).
struct Stacked {
    game_states: Vec<f32>,
    node_features: Vec<f32>,
    round_types: Vec<i64>,
    rotations: Vec<i64>,
    rows: usize,
    width: usize,
    num_players: usize,
    dedupe_maps: bool,
    map_index: Vec<i64>,
    num_maps: usize,
    maps_by_hash: HashMap<u64, Vec<usize>>,
}

impl Stacked {
    fn new(width: usize, num_players: usize, dedupe_maps: bool) -> Self {
        Stacked {
            game_states: Vec::new(),
            node_features: Vec::new(),
            round_types: Vec::new(),
            rotations: Vec::new(),
            rows: 0,
            width,
            num_players,
            dedupe_maps,
            map_index: Vec::new(),
            num_maps: 0,
            maps_by_hash: HashMap::new(),
        }
    }

    /// The map row of ``features``: an equal map already stored, or a new row.
    fn map_slot(&mut self, features: &[f32]) -> usize {
        let n = features.len();
        if self.dedupe_maps {
            let h = map_hash(features);
            if let Some(slots) = self.maps_by_hash.get(&h) {
                for &slot in slots {
                    let stored = &self.node_features[slot * n..(slot + 1) * n];
                    if stored.iter().zip(features).all(|(a, b)| a.to_bits() == b.to_bits()) {
                        return slot;
                    }
                }
            }
            self.maps_by_hash.entry(h).or_default().push(self.num_maps);
        }
        self.node_features.extend_from_slice(features);
        self.num_maps += 1;
        self.num_maps - 1
    }

    fn push(&mut self, game: &BaseGame) -> PyResult<usize> {
        let enc = game.encode_canonical();
        if enc.num_players != self.num_players || enc.game_state.len() != self.width {
            return Err(PyValueError::new_err(format!(
                "one batch must hold one player count: got {} players after {}",
                enc.num_players, self.num_players
            )));
        }
        self.game_states.extend_from_slice(&enc.game_state);
        let slot = self.map_slot(&enc.node_features);
        self.map_index.push(slot as i64);
        self.round_types.push(enc.round_type);
        self.rotations.push(enc.rotation as i64);
        self.rows += 1;
        Ok(enc.rotation)
    }

    fn into_dict<'py>(self, py: Python<'py>, title: &str) -> PyResult<Bound<'py, PyDict>> {
        let spec = spec_for(title);
        let d = PyDict::new(py);
        d.set_item("game_states", PyArray1::from_vec(py, self.game_states).reshape([self.rows, self.width])?)?;
        d.set_item(
            "node_features",
            PyArray1::from_vec(py, self.node_features).reshape([
                self.num_maps,
                spec.num_hexes(),
                spec.num_node_features(),
            ])?,
        )?;
        d.set_item("round_types", PyArray1::from_vec(py, self.round_types))?;
        d.set_item("rotations", PyArray1::from_vec(py, self.rotations))?;
        d.set_item("num_players", self.num_players)?;
        if self.dedupe_maps {
            d.set_item("map_index", PyArray1::from_vec(py, self.map_index))?;
        }
        Ok(d)
    }
}

/// Select a batch of leaves from each tree and encode them all for one
/// network request. ``max_leaves[t]`` caps tree ``t``'s batch (its
/// ``parallel_readouts``; a Gumbel search also stops at its budget),
/// ``dedupe[t]`` / ``use_score_values[t]`` are its ``SelfPlayConfig``
/// settings. Every tree must have the same player count. Returns a dict:
/// ``counts`` (leaves per tree, int64), ``terminal`` (terminal leaves backed
/// up per tree), ``game_states`` (B, D) and ``node_features`` (B, hexes,
/// features) float32 in the encoder's canonical frame, ``round_types`` /
/// ``rotations`` (B,) int64, ``num_players``, ``legal`` (int32, each leaf's
/// legal policy indices back to back) and ``legal_lengths`` (B,) int64. Each
/// tree keeps its leaves pending until ``mcts_apply_results``. With
/// ``dedupe_maps``, ``node_features`` is (M, hexes, features), each distinct
/// map once, and ``map_index`` (B,) int64 each leaf's row in it
/// (``node_features[map_index]`` is the plain layout).
#[pyfunction]
#[pyo3(signature = (trees, max_leaves, dedupe, use_score_values, dedupe_maps=false))]
pub fn mcts_collect_leaves<'py>(
    py: Python<'py>,
    trees: Vec<Bound<'py, RustMCTSPlayer>>,
    max_leaves: Vec<usize>,
    dedupe: Vec<bool>,
    use_score_values: Vec<bool>,
    dedupe_maps: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let t = trees.len();
    if max_leaves.len() != t || dedupe.len() != t || use_score_values.len() != t {
        return Err(PyValueError::new_err("mcts_collect_leaves: one max_leaves / dedupe / use_score_values per tree"));
    }
    let mut counts: Vec<i64> = Vec::with_capacity(t);
    let mut terminal: Vec<i64> = Vec::with_capacity(t);
    let mut legal: Vec<i32> = Vec::new();
    let mut legal_lengths: Vec<i64> = Vec::new();
    let mut stacked: Option<Stacked> = None;
    let mut title = String::from("1830");
    for (k, tree) in trees.iter().enumerate() {
        let mut tree = tree.borrow_mut();
        terminal.push(tree.collect_pending(max_leaves[k], dedupe[k], use_score_values[k])? as i64);
        counts.push(tree.pending.len() as i64);
        let tree = &mut *tree;
        for entry in tree.pending.iter_mut() {
            let node = &tree.arena[entry.0];
            let s = stacked.get_or_insert_with(|| {
                title = node.game.title.clone();
                let np = node.game.players_for_factored().len();
                Stacked::new(spec_for(&node.game.title).encoding_size(np), np, dedupe_maps)
            });
            entry.1 = s.push(&node.game)?;
            legal.extend(node.legal_action_indices.iter().map(|&i| i as i32));
            legal_lengths.push(node.legal_action_indices.len() as i64);
        }
    }
    let d = match stacked {
        Some(s) => s.into_dict(py, &title)?,
        None => {
            // Nothing to evaluate: empty arrays of the first tree's width.
            let (width, np) = match trees.first() {
                Some(tree) => {
                    let tree = tree.borrow();
                    let game = &tree.arena[tree.root_idx].game;
                    title = game.title.clone();
                    let np = game.players_for_factored().len();
                    (spec_for(&game.title).encoding_size(np), np)
                }
                None => (0, 0),
            };
            Stacked::new(width, np, dedupe_maps).into_dict(py, &title)?
        }
    };
    d.set_item("counts", PyArray1::from_vec(py, counts))?;
    d.set_item("terminal", PyArray1::from_vec(py, terminal))?;
    d.set_item("legal", PyArray1::from_vec(py, legal))?;
    d.set_item("legal_lengths", PyArray1::from_vec(py, legal_lengths))?;
    Ok(d)
}

/// Encode positions that are not search leaves (e.g. a policy player's
/// moves) into the same stacked layout as ``mcts_collect_leaves``:
/// ``game_states``, ``node_features``, ``round_types``, ``rotations``,
/// ``num_players`` (with ``dedupe_maps``, each distinct map once and
/// ``map_index``). One player count per call.
#[pyfunction]
#[pyo3(signature = (games, dedupe_maps=false))]
pub fn encode_games<'py>(
    py: Python<'py>,
    games: Vec<PyRef<'py, BaseGame>>,
    dedupe_maps: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let Some(first) = games.first() else {
        return Stacked::new(0, 0, dedupe_maps).into_dict(py, "1830");
    };
    let np = first.players_for_factored().len();
    let title = first.title.clone();
    let mut stacked = Stacked::new(spec_for(&title).encoding_size(np), np, dedupe_maps);
    for game in &games {
        stacked.push(game)?;
    }
    stacked.into_dict(py, &title)
}

/// Scatter one request's network outputs back into the trees that
/// ``mcts_collect_leaves`` built it from (same trees, same order).
/// ``priors``: the legal priors of every leaf back to back (the order of
/// ``legal``); ``values`` (B, VALUE_SIZE): the network's values in each leaf's
/// canonical frame; ``prior_temperature[t]``: tree ``t``'s
/// ``prior_temperature``; ``price_logits`` (B, slots, cells) or (B, slots *
/// cells) and ``slot_index``: the price head's rows and slot layout, if the
/// model has one.
#[pyfunction]
#[pyo3(signature = (trees, priors, values, prior_temperature, price_logits=None, slot_index=None))]
pub fn mcts_apply_results<'py>(
    trees: Vec<Bound<'py, RustMCTSPlayer>>,
    priors: PyReadonlyArray1<'py, f32>,
    values: PyReadonlyArray2<'py, f32>,
    prior_temperature: Vec<f32>,
    price_logits: Option<PyReadonlyArrayDyn<'py, f32>>,
    slot_index: Option<Bound<'py, PyAny>>,
) -> PyResult<()> {
    if prior_temperature.len() != trees.len() {
        return Err(PyValueError::new_err("mcts_apply_results: one prior_temperature per tree"));
    }
    let priors = priors.as_slice()?;
    let value_shape = values.shape().to_vec();
    if value_shape.len() != 2 || value_shape[1] < VALUE_SIZE {
        return Err(PyValueError::new_err(format!("values must be (B, {VALUE_SIZE}), got {value_shape:?}")));
    }
    let value_width = value_shape[1];
    let values = values.as_slice()?;
    let rows = value_shape[0];
    let total_pending: usize = trees.iter().map(|t| t.borrow().pending.len()).sum();
    if rows != total_pending {
        return Err(PyValueError::new_err(format!(
            "mcts_apply_results: {rows} value rows for {total_pending} pending leaves"
        )));
    }
    // Price rows: slots * cells floats per leaf, one shared slot layout.
    let price = match (&price_logits, &slot_index) {
        (Some(logits), Some(index)) if !index.is_none() && rows > 0 => {
            let flat = logits.as_slice()?;
            if flat.len() % rows != 0 || (flat.len() / rows) % NUM_CELLS != 0 {
                return Err(PyValueError::new_err(format!(
                    "price_logits: {} floats for {rows} leaves of {NUM_CELLS}-cell slots",
                    flat.len()
                )));
            }
            Some((flat, flat.len() / rows, Arc::new(decode_slot_index(index)?)))
        }
        _ => None,
    };

    let mut row = 0usize;
    let mut offset = 0usize;
    for (k, tree) in trees.iter().enumerate() {
        let mut tree = tree.borrow_mut();
        let inv_t = 1.0 / prior_temperature[k];
        for idx in std::mem::take(&mut tree.pending_dups) {
            tree.gumbel_note_duplicate(idx, -1.0);
            tree.revert_virtual_loss(idx);
        }
        let np = tree.num_players;
        for (idx, rotation) in std::mem::take(&mut tree.pending) {
            tree.revert_virtual_loss(idx);
            let n_legal = tree.arena[idx].legal_action_indices.len();
            if offset + n_legal > priors.len() {
                return Err(PyValueError::new_err("mcts_apply_results: fewer priors than the leaves' legal moves"));
            }
            let mut legal_probs = priors[offset..offset + n_legal].to_vec();
            offset += n_legal;
            if prior_temperature[k] != 1.0 {
                for p in legal_probs.iter_mut() {
                    *p = p.max(0.0).powf(inv_t);
                }
            }
            // Canonical frame (the leaf's mover first) -> absolute seats,
            // rotating only the real seats (``mcts.unrotate_value``).
            let v = &values[row * value_width..row * value_width + VALUE_SIZE];
            let mut value = [0.0f32; VALUE_SIZE];
            value.copy_from_slice(v);
            if rotation != 0 && np > 0 {
                for i in 0..np {
                    value[i] = v[(i + np - rotation % np) % np];
                }
            }
            let pc = price.as_ref().map(|(flat, width, index)| PriceComponents {
                price_logits: flat[row * width..(row + 1) * width].to_vec(),
                slot_index: Arc::clone(index),
                num_slots: width / NUM_CELLS,
            });
            tree.incorporate(idx, legal_probs, value, pc);
            row += 1;
        }
    }
    if offset != priors.len() {
        return Err(PyValueError::new_err(format!(
            "mcts_apply_results: {} priors for {offset} legal moves",
            priors.len()
        )));
    }
    Ok(())
}

/// The value a search backs up at a finished ``game`` (``terminal_value``).
#[pyfunction]
pub fn mcts_terminal_value(game: PyRef<'_, BaseGame>, use_score_values: bool) -> Vec<f32> {
    terminal_value(&game, use_score_values).to_vec()
}

/// Forget the trees' pending batches without applying them (virtual losses
/// reverted), e.g. after a failed request.
#[pyfunction]
pub fn mcts_discard_pending(trees: Vec<Bound<'_, RustMCTSPlayer>>) {
    for tree in trees {
        let mut tree = tree.borrow_mut();
        for idx in std::mem::take(&mut tree.pending_dups) {
            tree.gumbel_note_duplicate(idx, -1.0);
            tree.revert_virtual_loss(idx);
        }
        for (idx, _) in std::mem::take(&mut tree.pending) {
            tree.revert_virtual_loss(idx);
        }
    }
}
