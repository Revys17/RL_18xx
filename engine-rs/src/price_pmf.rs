//! Exact price distribution for price-bearing actions — mirrors
//! `rl18xx/agent/alphazero/price_pmf.py` (integer arithmetic only, so both
//! engines assign every price to the same cell).
//!
//! The legal range `[lo, hi]` of a Bid / cross-corp BuyTrain / BuyCompany is
//! partitioned into *atoms* (single prices humans pile onto) and range-relative
//! *bins* whose mass is spread uniformly over their integers. The price head
//! emits one logit per cell; `P(price) = softmax(nonempty logits)[cell] / |cell|`.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::Rng;

pub const NUM_CELLS: usize = 24;
pub const BID_LADDER: usize = 20;

/// `(atom cells, bin cells)` per price-bearing action type.
pub fn layout(action_type: &str) -> Option<(usize, usize)> {
    match action_type {
        "Bid" => Some((BID_LADDER, 1)),
        "BuyTrain" => Some((3, 20)),
        "BuyCompany" => Some((2, 8)),
        _ => None,
    }
}

pub struct PriceCells {
    lo: i64,
    hi: i64,
    n_atoms: usize,
    n_bins: usize,
    /// Live atoms as `(price, cell)`, in atom-spec order.
    atoms: Vec<(i64, usize)>,
    atoms_sorted: Vec<i64>,
    pub counts: [i64; NUM_CELLS],
}

impl PriceCells {
    pub fn new(action_type: &str, lo: i64, hi: i64, bid_step: i64) -> Option<Self> {
        let (n_atoms, n_bins) = layout(action_type)?;
        if hi < lo {
            return None;
        }
        let spec: Vec<i64> = match action_type {
            "Bid" => (0..BID_LADDER as i64).map(|j| lo + bid_step * j).collect(),
            "BuyTrain" => vec![lo, hi, hi - 1],
            _ => vec![lo, hi],
        };
        let mut atoms: Vec<(i64, usize)> = Vec::new();
        for (j, &price) in spec.iter().enumerate() {
            if lo <= price && price <= hi && !atoms.iter().any(|&(p, _)| p == price) {
                atoms.push((price, j));
            }
        }
        let mut atoms_sorted: Vec<i64> = atoms.iter().map(|&(p, _)| p).collect();
        atoms_sorted.sort_unstable();
        let mut cells = PriceCells { lo, hi, n_atoms, n_bins, atoms, atoms_sorted, counts: [0; NUM_CELLS] };
        for &(_, j) in &cells.atoms {
            cells.counts[j] = 1;
        }
        for b in 0..n_bins {
            let (start, end) = cells.bin_bounds(b);
            let inside = cells.atoms_sorted.iter().filter(|&&a| start <= a && a <= end).count() as i64;
            cells.counts[n_atoms + b] = (end - start + 1 - inside).max(0);
        }
        Some(cells)
    }

    pub fn nonempty(&self, cell: usize) -> bool {
        self.counts[cell] > 0
    }

    /// Integer prices `[start, end]` whose bin is `b` (atoms included).
    fn bin_bounds(&self, b: usize) -> (i64, i64) {
        let (width, n, b) = (self.hi - self.lo + 1, self.n_bins as i64, b as i64);
        (self.lo + (b * width + n - 1) / n, self.lo + ((b + 1) * width + n - 1) / n - 1)
    }

    pub fn cell_of(&self, price: i64) -> Option<usize> {
        if price < self.lo || price > self.hi {
            return None;
        }
        if let Some(&(_, j)) = self.atoms.iter().find(|&&(p, _)| p == price) {
            return Some(j);
        }
        Some(self.n_atoms + (((price - self.lo) * self.n_bins as i64) / (self.hi - self.lo + 1)) as usize)
    }

    /// The `u`-th (0-based, ascending) price in `cell`; `0 <= u < counts[cell]`.
    pub fn member(&self, cell: usize, u: i64) -> Option<i64> {
        if cell >= NUM_CELLS || u < 0 || u >= self.counts[cell] {
            return None;
        }
        if cell < self.n_atoms {
            return self.atoms.iter().find(|&&(_, j)| j == cell).map(|&(p, _)| p);
        }
        let (start, _) = self.bin_bounds(cell - self.n_atoms);
        let mut price = start + u;
        for &atom in &self.atoms_sorted {
            if start <= atom && atom <= price {
                price += 1;
            }
        }
        Some(price)
    }

    pub fn sample_price<R: Rng>(&self, cell: usize, rng: &mut R) -> i64 {
        let u = rng.gen_range(0..self.counts[cell].max(1));
        self.member(cell, u).unwrap_or(self.lo)
    }

    /// Softmax of the head's logits over the non-empty cells; missing or
    /// non-finite logits fall back to uniform over them.
    pub fn cell_probs(&self, logits: Option<&[f32]>) -> [f64; NUM_CELLS] {
        let mut z = [0.0_f64; NUM_CELLS];
        if let Some(l) = logits {
            if l.len() >= NUM_CELLS && (0..NUM_CELLS).all(|c| !self.nonempty(c) || l[c].is_finite()) {
                for c in 0..NUM_CELLS {
                    z[c] = l[c] as f64;
                }
            }
        }
        let max = (0..NUM_CELLS).filter(|&c| self.nonempty(c)).map(|c| z[c]).fold(f64::NEG_INFINITY, f64::max);
        let mut p = [0.0_f64; NUM_CELLS];
        let mut total = 0.0;
        for c in 0..NUM_CELLS {
            if self.nonempty(c) {
                p[c] = (z[c] - max).exp();
                total += p[c];
            }
        }
        for v in p.iter_mut() {
            *v /= total;
        }
        p
    }
}

/// Order in which progressive widening materializes a slot's cells — mirrors
/// Python `proposal_order`: the most likely cell first, then Gumbel-top-k
/// (sampling without replacement) from `(1 - eps) * probs + eps * uniform`.
pub fn proposal_order<R: Rng>(probs: &[f64; NUM_CELLS], explore_eps: f64, rng: &mut R) -> Vec<usize> {
    let live: Vec<usize> = (0..NUM_CELLS).filter(|&c| probs[c] > 0.0).collect();
    if live.is_empty() {
        return Vec::new();
    }
    // argmax with the lowest index winning ties (matches numpy argmax).
    let mut first = live[0];
    for &c in &live {
        if probs[c] > probs[first] {
            first = c;
        }
    }
    let n = live.len() as f64;
    let mut keyed: Vec<(f64, usize)> = live
        .iter()
        .filter(|&&c| c != first)
        .map(|&c| {
            let mixed = (1.0 - explore_eps) * probs[c] + explore_eps / n;
            let u: f64 = rng.gen_range(f64::MIN_POSITIVE..1.0);
            (mixed.ln() - (-u.ln()).ln(), c)
        })
        .collect();
    keyed.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    std::iter::once(first).chain(keyed.into_iter().map(|(_, c)| c)).collect()
}

fn cells_or_err(action_type: &str, lo: i64, hi: i64, bid_step: i64) -> PyResult<PriceCells> {
    PriceCells::new(action_type, lo, hi, bid_step)
        .ok_or_else(|| PyValueError::new_err(format!("no price cells for {action_type} [{lo}, {hi}]")))
}

/// Per-cell price counts for `[lo, hi]` (parity tests).
#[pyfunction]
pub fn price_pmf_counts(action_type: &str, lo: i64, hi: i64, bid_step: i64) -> PyResult<Vec<i64>> {
    Ok(cells_or_err(action_type, lo, hi, bid_step)?.counts.to_vec())
}

/// The cell holding `price` (parity tests).
#[pyfunction]
pub fn price_pmf_cell_of(action_type: &str, price: i64, lo: i64, hi: i64, bid_step: i64) -> PyResult<usize> {
    cells_or_err(action_type, lo, hi, bid_step)?
        .cell_of(price)
        .ok_or_else(|| PyValueError::new_err(format!("price {price} outside [{lo}, {hi}]")))
}

/// The `u`-th price of `cell` (parity tests).
#[pyfunction]
pub fn price_pmf_member(action_type: &str, cell: usize, u: i64, lo: i64, hi: i64, bid_step: i64) -> PyResult<i64> {
    cells_or_err(action_type, lo, hi, bid_step)?
        .member(cell, u)
        .ok_or_else(|| PyValueError::new_err(format!("no member {u} in cell {cell}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cells_partition_the_range() {
        for &(t, lo, hi) in &[("Bid", 225, 600), ("Bid", 25, 40), ("BuyTrain", 1, 441), ("BuyTrain", 1, 3), ("BuyCompany", 80, 320), ("BuyTrain", 7, 8)] {
            let cells = PriceCells::new(t, lo, hi, 5).unwrap();
            assert_eq!(cells.counts.iter().sum::<i64>(), hi - lo + 1, "{t} [{lo}, {hi}]");
            let mut seen = std::collections::HashSet::new();
            for c in 0..NUM_CELLS {
                for u in 0..cells.counts[c] {
                    let p = cells.member(c, u).unwrap();
                    assert_eq!(cells.cell_of(p), Some(c), "{t} [{lo}, {hi}] price {p}");
                    assert!(seen.insert(p));
                }
            }
        }
    }

    #[test]
    fn proposal_order_starts_at_the_mode_and_covers_every_nonempty_cell() {
        let cells = PriceCells::new("BuyTrain", 1, 441, 5).unwrap();
        let mut logits = [0.0_f32; NUM_CELLS];
        logits[1] = 5.0;
        let probs = cells.cell_probs(Some(&logits));
        let order = proposal_order(&probs, 0.05, &mut rand::thread_rng());
        assert_eq!(order[0], 1);
        assert_eq!(order.len(), (0..NUM_CELLS).filter(|&c| cells.nonempty(c)).count());
    }
}
