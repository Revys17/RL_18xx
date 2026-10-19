//! Gumbel root selection and completed-Q policy targets for the Rust MCTS
//! (Danihelka et al. 2022, "Policy improvement by planning with Gumbel";
//! the reference implementation is DeepMind's ``mctx``:
//! ``gumbel_muzero_policy``, ``seq_halving``, ``qtransform_completed_by_mix_value``).
//!
//! An option of ``RustMCTSPlayer`` (``set_gumbel_config``; off by default).
//! At the root of a search with budget ``n`` (``begin_gumbel_search``):
//!
//! - draw Gumbel noise ``g`` for every legal move and keep the top
//!   ``m = min(k, legal)`` moves by ``g + logits`` (logits = log of the
//!   tree's priors, i.e. after ``prior_temperature``);
//! - spend the budget by sequential halving: simulation ``s`` goes to a kept
//!   move whose visit count (in this search) equals ``considered_visits[s]``
//!   (mctx's ``get_sequence_of_considered_visits``), the one with the highest
//!   ``g + logits + sigma(q)``, so each phase visits the surviving moves
//!   equally and keeps the better half for the next;
//! - play ``argmax(g + logits + sigma(q))`` over the most-visited kept moves;
//! - the improved policy target is ``softmax(logits + sigma(completed q))``
//!   over every legal move, an unvisited move's q completed by the root's
//!   ``v_mix`` (its network value mixed with the prior-weighted q of the
//!   visited moves).
//!
//! ``sigma(q) = (c_visit + max_b N(b)) * c_scale * q`` with q the mover's
//! mean 0-1 win share (the value vector read at the root's
//! ``active_player_index``, as PUCT reads it); no min-max rescaling.
//!
//! Our specifics:
//!
//! - **Below the root** nothing changes: PUCT (mean-Q / FPU / c_puct options),
//!   the >20-legal-move top-k widening and the price grandchildren. At the
//!   root the Gumbel top-m replaces the top-k widening mask.
//! - **Prices**: a root move is a policy slot. A price-bearing slot (Bid /
//!   BuyTrain from a corporation / BuyCompany) is one candidate; a
//!   simulation through it descends into its price grandchildren by the usual
//!   progressive widening. Its q is the slot's aggregate over all of its
//!   prices (what the categorical level sees, and what the policy head
//!   predicts), the move played commits the most-visited price
//!   (``advance_root``), and the price target stays the grandchildren's visit
//!   distribution (``price_grandchildren_at_root``).
//! - **Forced moves**: a root move's child is the position after its forced
//!   chain (``maybe_add_child`` collapses it), so it is one candidate like any
//!   other; a root with a single legal move needs no search (the callers play
//!   it directly).
//! - **Terminal leaves** back up their final value immediately and count as a
//!   simulation of the root move they went through.
//! - **Batches**: a simulation in flight counts toward its move's visits
//!   (its virtual loss), so one batch of leaves follows the schedule exactly;
//!   q reads only completed backups. A ``dedupe_leaves`` duplicate keeps its
//!   virtual loss for the batch (steering later descents off its path) but is
//!   not counted, so its slot is taken by the next selection; a search always
//!   backs up exactly ``n`` simulations (``gumbel_remaining`` caps each
//!   batch).
//! - **Tree reuse**: visit counts for the schedule are counted from the start
//!   of this search (a reused subtree's earlier visits don't count toward the
//!   budget); q, ``v_mix`` and ``max_b N(b)`` use every visit the tree has.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use rand::Rng;

use crate::mcts::RustMCTSPlayer;

/// Floor for log-priors and scores (mctx's ``low_logit``): a zero prior is
/// finite, so scores and the target stay well defined.
const LOW_LOGIT: f32 = -1e9;

#[derive(Debug, Clone)]
pub struct GumbelConfig {
    /// Moves kept at the root (``m = min(k, legal moves)``).
    pub k: usize,
    pub c_visit: f32,
    pub c_scale: f32,
    /// Scale of the Gumbel noise; 0 keeps the top-k by prior and plays
    /// deterministically (mctx's ``gumbel_scale``).
    pub noise_scale: f32,
}

#[derive(Debug, Clone)]
pub struct GumbelSearch {
    /// The arena root this search was begun at.
    pub root_idx: usize,
    pub budget: usize,
    /// Gumbel noise per root move (compressed index), times ``noise_scale``.
    pub noise: Vec<f32>,
    /// Root child visit counts when the search began (a reused subtree's).
    pub base_n: Vec<f32>,
    /// Per root move: whether it is among the m kept moves.
    pub considered: Vec<bool>,
    pub num_considered: usize,
    /// ``considered_visits[s]``: the visit count a move must have for
    /// simulation ``s`` to go to it.
    pub sequence: Vec<u32>,
    /// ``dedupe_leaves`` duplicates in flight per root move: their virtual
    /// losses steer the rest of the batch away from their path, but they are
    /// not simulations, so the schedule doesn't count them.
    pub duplicates: Vec<f32>,
}

/// mctx ``seq_halving.get_sequence_of_considered_visits``: for each
/// simulation index, the (search-local) visit count of the moves it may go
/// to. ``m`` moves are visited equally in ``ceil(log2 m)`` phases of roughly
/// ``n / ceil(log2 m)`` simulations, halving the moves after each phase.
pub fn considered_visits_sequence(m: usize, n: usize) -> Vec<u32> {
    if m <= 1 {
        return (0..n as u32).collect();
    }
    let log2max = (m as f64).log2().ceil() as usize;
    let mut sequence: Vec<u32> = Vec::with_capacity(n + m);
    let mut visits = vec![0u32; m];
    let mut num_considered = m;
    while sequence.len() < n {
        let extra = std::cmp::max(1, n / (log2max * num_considered));
        for _ in 0..extra {
            sequence.extend_from_slice(&visits[..num_considered]);
            for v in visits.iter_mut().take(num_considered) {
                *v += 1;
            }
        }
        num_considered = std::cmp::max(2, num_considered / 2);
    }
    sequence.truncate(n);
    sequence
}

fn softmax(xs: &[f32]) -> Vec<f64> {
    let max = xs.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let exps: Vec<f64> = xs.iter().map(|&x| ((x as f64) - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.into_iter().map(|e| e / sum).collect()
}

/// First index of the maximum of ``score`` over the indices ``allowed``.
fn argmax_over(score: &[f32], allowed: impl Iterator<Item = usize>) -> Option<usize> {
    let mut best: Option<(usize, f32)> = None;
    for i in allowed {
        if best.map_or(true, |(_, b)| score[i] > b) {
            best = Some((i, score[i]));
        }
    }
    best.map(|(i, _)| i)
}

impl RustMCTSPlayer {
    fn gumbel_search_here(&self) -> Option<&GumbelSearch> {
        self.gumbel_search.as_ref().filter(|gs| gs.root_idx == self.root_idx)
    }

    /// Log-priors of the root's moves, floored at ``LOW_LOGIT``.
    fn root_logits(&self) -> Vec<f32> {
        self.arena[self.root_idx]
            .child_prior
            .iter()
            .map(|&p| if p > 0.0 { p.ln().max(LOW_LOGIT) } else { LOW_LOGIT })
            .collect()
    }

    /// The root's completed q per move, for the mover: a visited move's mean
    /// backed-up value (in-flight virtual losses undone), an unvisited one's
    /// the root's ``v_mix`` (mctx ``_compute_mixed_value``).
    pub(crate) fn root_completed_q(&self) -> Vec<f32> {
        let q = self.root_visited_q();
        let v_mix = self.v_mix_of(&q);
        q.into_iter().map(|qi| qi.unwrap_or(v_mix)).collect()
    }

    /// The root's ``v_mix`` (mctx ``_compute_mixed_value``): its network value
    /// mixed with the prior-weighted mean q of the visited moves.
    pub(crate) fn root_v_mix(&self) -> f32 {
        let q = self.root_visited_q();
        self.v_mix_of(&q)
    }

    /// Each root move's mean backed-up value for the mover (in-flight virtual
    /// losses undone), ``None`` where unvisited.
    fn root_visited_q(&self) -> Vec<Option<f32>> {
        let root = &self.arena[self.root_idx];
        let ap = root.active_player_index;
        let num_legal = root.legal_action_indices.len();
        (0..num_legal)
            .map(|i| {
                let n = root.child_n[i];
                (n > 0.0).then(|| (root.child_w[i][ap] + root.child_vl[i]) / n)
            })
            .collect()
    }

    fn v_mix_of(&self, q: &[Option<f32>]) -> f32 {
        let root = &self.arena[self.root_idx];
        let ap = root.active_player_index;
        let num_legal = root.legal_action_indices.len();
        let sum_n: f32 = root.child_n.iter().sum();
        let tiny = f32::MIN_POSITIVE;
        let mut sum_p = 0.0f32;
        let mut weighted = 0.0f32;
        for i in 0..num_legal {
            if let Some(qi) = q[i] {
                let p = root.child_prior[i].max(tiny);
                sum_p += p;
                weighted += p * qi;
            }
        }
        let weighted_q = if sum_p > 0.0 { weighted / sum_p } else { 0.0 };
        (root.raw_value[ap] + sum_n * weighted_q) / (sum_n + 1.0)
    }

    /// ``sigma(q) = sigma_scale * q``: ``(c_visit + max_b N(b)) * c_scale``.
    fn gumbel_sigma_scale(&self, cfg: &GumbelConfig) -> f32 {
        let max_n = self.arena[self.root_idx].child_n.iter().copied().fold(0.0f32, f32::max);
        (cfg.c_visit + max_n) * cfg.c_scale
    }

    /// ``g + logits + sigma(completed q)`` per root move (mctx
    /// ``score_considered`` before its visit-count penalty).
    fn gumbel_scores(&self, gs: &GumbelSearch, cfg: &GumbelConfig) -> Vec<f32> {
        let logits = self.root_logits();
        let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let q = self.root_completed_q();
        let scale = self.gumbel_sigma_scale(cfg);
        (0..logits.len())
            .map(|i| (gs.noise[i] + (logits[i] - max_logit) + scale * q[i]).max(LOW_LOGIT))
            .collect()
    }

    /// Root move ``i``'s simulations in this search, completed (``in_flight``
    /// false) or also counting the descents in flight through it (but not
    /// duplicates).
    fn search_visits(&self, gs: &GumbelSearch, i: usize, in_flight: bool) -> f32 {
        let root = &self.arena[self.root_idx];
        let n = root.child_n[i] - gs.base_n[i];
        if in_flight {
            n + root.child_vl[i] - gs.duplicates[i]
        } else {
            n
        }
    }

    /// The root move (compressed index) whose subtree holds ``arena_idx``.
    fn root_move_of(&self, arena_idx: usize) -> Option<usize> {
        let mut current = arena_idx;
        loop {
            let parent = self.arena[current].parent?;
            if parent == self.root_idx {
                return Some(self.arena[current].parent_compressed_idx);
            }
            current = parent;
        }
    }

    /// A ``dedupe_leaves`` duplicate (virtual loss applied, no backup to come)
    /// was selected at ``arena_idx``: keep it out of the Gumbel schedule.
    pub(crate) fn gumbel_note_duplicate(&mut self, arena_idx: usize, delta: f32) {
        let Some(i) = self.root_move_of(arena_idx) else {
            return;
        };
        let root_idx = self.root_idx;
        if let Some(gs) = self.gumbel_search.as_mut().filter(|gs| gs.root_idx == root_idx) {
            if let Some(d) = gs.duplicates.get_mut(i) {
                *d = (*d + delta).max(0.0);
            }
        }
    }

    /// Simulations of the budget not yet completed or in flight.
    pub(crate) fn gumbel_remaining_inner(&self) -> usize {
        let Some(gs) = self.gumbel_search_here() else {
            return 0;
        };
        let used: f32 = (0..gs.noise.len()).map(|i| self.search_visits(gs, i, true)).sum();
        gs.budget.saturating_sub(used.round().max(0.0) as usize)
    }

    /// The root move the next simulation goes through, when a Gumbel search
    /// runs at ``node_idx`` (the root); ``None`` elsewhere (PUCT decides).
    pub(crate) fn gumbel_root_choice(&self, node_idx: usize) -> Option<usize> {
        if node_idx != self.root_idx {
            return None;
        }
        let gs = self.gumbel_search_here()?;
        let cfg = self.gumbel.as_ref()?;
        if gs.noise.len() != self.arena[self.root_idx].legal_action_indices.len() {
            return None;
        }
        let scores = self.gumbel_scores(gs, cfg);
        let counts: Vec<f32> = (0..gs.noise.len()).map(|i| self.search_visits(gs, i, true)).collect();
        let considered = || (0..gs.noise.len()).filter(|&i| gs.considered[i]);
        let simulation = counts.iter().sum::<f32>().round().max(0.0) as usize;
        if let Some(&level) = gs.sequence.get(simulation) {
            let at_level = considered().filter(|&i| counts[i] == level as f32);
            if let Some(i) = argmax_over(&scores, at_level) {
                return Some(i);
            }
        }
        // Off the schedule (past the budget, or the counts left it): the
        // least-visited kept moves.
        let min = considered().map(|i| counts[i]).fold(f32::INFINITY, f32::min);
        argmax_over(&scores, considered().filter(|&i| counts[i] == min))
    }

    fn gumbel_config_or_err(&self) -> PyResult<&GumbelConfig> {
        self.gumbel
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("Gumbel root selection is off (set_gumbel_config)"))
    }

    fn gumbel_search_or_err(&self) -> PyResult<&GumbelSearch> {
        self.gumbel_search_here()
            .ok_or_else(|| PyRuntimeError::new_err("no Gumbel search at this root (begin_gumbel_search)"))
    }
}

#[pymethods]
impl RustMCTSPlayer {
    /// Turn Gumbel root selection on (``k`` > 0: moves kept at the root) or
    /// off (``k`` = 0). See the module docs of ``gumbel.rs``.
    #[pyo3(signature = (k, c_visit=50.0, c_scale=1.0, noise_scale=1.0))]
    pub fn set_gumbel_config(&mut self, k: usize, c_visit: f32, c_scale: f32, noise_scale: f32) {
        self.gumbel = (k > 0).then(|| GumbelConfig { k, c_visit, c_scale, noise_scale });
        self.gumbel_search = None;
    }

    /// Whether Gumbel root selection is on.
    #[getter]
    pub fn gumbel_enabled(&self) -> bool {
        self.gumbel.is_some()
    }

    /// Whether a Gumbel search was begun at the current root (cleared when the
    /// root advances).
    #[getter]
    pub fn gumbel_search_active(&self) -> bool {
        self.gumbel_search_here().is_some()
    }

    /// Start a Gumbel search of ``budget`` simulations at the current root
    /// (which must be expanded): draws the noise and keeps the top-m moves.
    pub fn begin_gumbel_search(&mut self, budget: usize) -> PyResult<()> {
        let cfg = self.gumbel_config_or_err()?.clone();
        let root_idx = self.root_idx;
        if !self.arena[root_idx].is_expanded {
            return Err(PyRuntimeError::new_err("begin_gumbel_search: the root is not expanded"));
        }
        let num_legal = self.arena[root_idx].legal_action_indices.len();
        if num_legal == 0 {
            return Err(PyValueError::new_err("begin_gumbel_search: the root has no legal moves"));
        }
        let noise: Vec<f32> = (0..num_legal)
            .map(|_| {
                if cfg.noise_scale == 0.0 {
                    return 0.0;
                }
                // u in (0, 1): -ln(-ln u) is a standard Gumbel draw.
                let u: f64 = self.rng.gen_range(f64::MIN_POSITIVE..1.0);
                (-(-u.ln()).ln()) as f32 * cfg.noise_scale
            })
            .collect();
        let logits = self.root_logits();
        let m = cfg.k.min(num_legal);
        let mut order: Vec<usize> = (0..num_legal).collect();
        // Stable: ties keep the lower action index first.
        order.sort_by(|&a, &b| {
            (noise[b] + logits[b])
                .partial_cmp(&(noise[a] + logits[a]))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut considered = vec![false; num_legal];
        for &i in order.iter().take(m) {
            considered[i] = true;
        }
        self.gumbel_search = Some(GumbelSearch {
            root_idx,
            budget,
            noise,
            base_n: self.arena[root_idx].child_n.clone(),
            considered,
            num_considered: m,
            sequence: considered_visits_sequence(m, budget),
            duplicates: vec![0.0; num_legal],
        });
        Ok(())
    }

    /// ``tree_search`` with ``dedupe_leaves``: the leaf at ``arena_idx`` was
    /// selected again in this batch (its virtual loss applied). It is not a
    /// simulation of the Gumbel schedule.
    pub fn gumbel_mark_duplicate(&mut self, arena_idx: usize) {
        self.gumbel_note_duplicate(arena_idx, 1.0);
    }

    /// Revert a duplicate's virtual loss at the end of its batch (undoing
    /// ``gumbel_mark_duplicate``).
    pub fn revert_duplicate_virtual_loss(&mut self, arena_idx: usize) {
        self.gumbel_note_duplicate(arena_idx, -1.0);
        self.revert_virtual_loss(arena_idx);
    }

    /// Simulations of the current Gumbel search not yet completed or in
    /// flight (0 without one): a batch should select at most this many leaves.
    pub fn gumbel_remaining(&self) -> usize {
        self.gumbel_remaining_inner()
    }

    /// The move a Gumbel search plays: ``argmax(g + logits + sigma(q))`` over
    /// the kept moves with the most visits in this search.
    pub fn gumbel_action(&self) -> PyResult<u32> {
        let cfg = self.gumbel_config_or_err()?;
        let gs = self.gumbel_search_or_err()?;
        let scores = self.gumbel_scores(gs, cfg);
        let visits: Vec<f32> = (0..gs.noise.len()).map(|i| self.search_visits(gs, i, false)).collect();
        let considered = || (0..gs.noise.len()).filter(|&i| gs.considered[i]);
        let max = considered().map(|i| visits[i]).fold(f32::NEG_INFINITY, f32::max);
        let best = argmax_over(&scores, considered().filter(|&i| visits[i] == max))
            .ok_or_else(|| PyRuntimeError::new_err("gumbel_action: no kept moves"))?;
        Ok(self.arena[self.root_idx].legal_action_indices[best])
    }

    /// The improved policy at the root, ``softmax(logits + sigma(completed
    /// q))`` over the legal moves, as a dense policy-size vector (zero off
    /// the legal moves). Needs Gumbel on, not a running search: it reads only
    /// the root's priors and statistics.
    pub fn gumbel_policy_target(&self) -> PyResult<Vec<f32>> {
        let cfg = self.gumbel_config_or_err()?;
        let root = &self.arena[self.root_idx];
        let mut out = vec![0.0f32; crate::action_index::layout().total as usize];
        if root.legal_action_indices.is_empty() {
            return Ok(out);
        }
        let logits = self.root_logits();
        let q = self.root_completed_q();
        let scale = self.gumbel_sigma_scale(cfg);
        let z: Vec<f32> = (0..logits.len()).map(|i| logits[i] + scale * q[i]).collect();
        for (i, p) in softmax(&z).into_iter().enumerate() {
            out[root.legal_action_indices[i] as usize] = p as f32;
        }
        Ok(out)
    }

    /// Diagnostics of the root's Gumbel state, per legal move (in
    /// ``legal_action_indices_at_root`` order): ``{"legal", "noise",
    /// "logits", "completed_q", "considered", "search_visits",
    /// "sigma_scale", "budget", "num_considered"}``.
    pub fn gumbel_root_stats(&self, py: Python<'_>) -> PyResult<PyObject> {
        let cfg = self.gumbel_config_or_err()?;
        let gs = self.gumbel_search_or_err()?;
        let d = pyo3::types::PyDict::new(py);
        let n = gs.noise.len();
        d.set_item("legal", self.arena[self.root_idx].legal_action_indices.clone())?;
        d.set_item("noise", gs.noise.clone())?;
        d.set_item("logits", self.root_logits())?;
        d.set_item("completed_q", self.root_completed_q())?;
        d.set_item("considered", gs.considered.clone())?;
        d.set_item("search_visits", (0..n).map(|i| self.search_visits(gs, i, false)).collect::<Vec<f32>>())?;
        d.set_item("sigma_scale", self.gumbel_sigma_scale(cfg))?;
        d.set_item("budget", gs.budget)?;
        d.set_item("num_considered", gs.num_considered)?;
        Ok(d.into_any().unbind())
    }
}

#[cfg(test)]
mod tests {
    use super::considered_visits_sequence;

    #[test]
    fn sequence_matches_mctx() {
        // mctx seq_halving.get_sequence_of_considered_visits(4, 10): 4 moves
        // once, then the best 2 four more times.
        assert_eq!(considered_visits_sequence(4, 10), vec![0, 0, 0, 0, 1, 1, 2, 2, 3, 3]);
        // m = 1: every simulation to the one move.
        assert_eq!(considered_visits_sequence(1, 4), vec![0, 1, 2, 3]);
        // 16 moves, 32 simulations: 16 once, the best 8 once more, the best 4
        // twice more.
        let s = considered_visits_sequence(16, 32);
        assert_eq!(s.len(), 32);
        assert_eq!(&s[..16], &[0u32; 16]);
        assert_eq!(&s[16..24], &[1u32; 8]);
        assert_eq!(&s[24..32], &[2, 2, 2, 2, 3, 3, 3, 3]);
    }
}
