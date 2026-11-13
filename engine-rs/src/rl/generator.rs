//! The index API of the titles whose legal moves come from their own
//! generator (`RlLayoutKind::Generator`; 1867, layout `1867-v1`): a state's
//! legal indices, index -> move (with its price), apply by index, the
//! price range of an index -- in Rust and in Python (`rl_legal_indices`,
//! `rl_legal_moves_json`, `rl_decode_index`, `rl_apply_index`,
//! `rl_price_range`).
//!
//! An index means the same move whatever the cross-player policy: the
//! policy only removes moves (`policy=None` = the title's own policy,
//! `"rules"` = everything Ruby accepts). A price-bearing index is ONE
//! index; its legal `[lo, hi]` and step come from the move
//! (`PriceRange`), never from the layout.

use pyo3::prelude::*;
use serde_json::{json, Value};

use super::layout::RlLayout;
use crate::actions::GameError;
use crate::g1867::legal::Move;
use crate::game::BaseGame;
use crate::title::{CrossPlayerPolicy, GameTitle, RlLayoutKind};

/// The registry layout of a `Generator` title.
pub(crate) fn registry_layout(title: &'static dyn GameTitle) -> Result<RlLayout, String> {
    match title.name() {
        "1867" => super::g1867::registry(title),
        other => Err(format!("{other}: no generator layout")),
    }
}

/// The price cells of a priced index (`docs/rl_descriptors.md` §6): the
/// kind's `price_pmf` cell type over the move's range in STEP UNITS, `k =
/// (p - lo) / step` in `[0, (hi - lo) / step]`, so every cell holds legal
/// prices only. `None` for a kind without fixed cells.
pub fn price_cells(kind: super::PriceKind, r: crate::g1867::legal::PriceRange) -> Option<crate::price_pmf::PriceCells> {
    let top = ((r.hi - r.lo) / r.step.max(1)) as i64;
    crate::price_pmf::PriceCells::new(kind.cell_type()?, 0, top, 1)
}

/// The price of step `k` of a range (the inverse of [`price_step`]).
pub fn price_at(r: crate::g1867::legal::PriceRange, k: i64) -> i32 {
    r.lo + r.step.max(1) * k as i32
}

/// The step of a legal price of a range (`None` off the range or its grid).
pub fn price_step(r: crate::g1867::legal::PriceRange, p: i32) -> Option<i64> {
    let s = r.step.max(1);
    (r.lo <= p && p <= r.hi && (p - r.lo) % s == 0).then(|| ((p - r.lo) / s) as i64)
}

/// A legal move and its index.
#[derive(Clone, Debug)]
pub struct IndexedMove {
    pub index: u32,
    pub mv: Move,
}

/// `m` with its price set to `price` (checked against the move's range and
/// step); `None` keeps the range's low end (the move as generated). A move
/// without a price ignores `price`.
pub fn priced(m: Move, price: Option<i32>) -> Result<Move, GameError> {
    match (m.price, price) {
        (Some(r), Some(p)) => {
            if p < r.lo || p > r.hi || (p - r.lo) % r.step.max(1) != 0 {
                Err(GameError::new(format!(
                    "price {p} is outside the move's range [{}, {}] step {} ({:?})",
                    r.lo, r.hi, r.step, m.action
                )))
            } else {
                Ok(m.with_price(p))
            }
        }
        _ => Ok(m),
    }
}

impl BaseGame {
    /// The game's generator layout; an error for a title of another kind
    /// (1830 indexes through `factored_legal_indices`; the 1822 family has
    /// no layout yet).
    pub fn rl_generator_layout(&self) -> Result<&'static RlLayout, GameError> {
        let kind = self.title_def().rl_layout_kind();
        if kind != RlLayoutKind::Generator {
            return Err(GameError::new(format!(
                "title {} has no generator layout (RL layout kind {:?}): {}",
                self.title,
                kind.name(),
                if kind == RlLayoutKind::Factored {
                    "index it through factored_legal_indices / apply_action_index"
                } else {
                    "it is not bridge-ready"
                }
            )));
        }
        super::rl_layout(&self.title)
    }

    fn rl_generator_moves(&mut self, policy: CrossPlayerPolicy) -> Result<Vec<Move>, GameError> {
        match self.title.as_str() {
            "1867" => Ok(self.g1867_legal_moves_with(policy)),
            t => Err(GameError::new(format!("{t}: no legal-move generator behind the RL index API"))),
        }
    }

    /// The index of one of the title generator's moves (the state names
    /// what the move refers to by id).
    pub fn rl_index_of_move(&self, m: &Move) -> Result<u32, GameError> {
        let lo = self.rl_generator_layout()?;
        match self.title.as_str() {
            "1867" => super::g1867::move_index(lo, self, m),
            t => Err(GameError::new(format!("{t}: no generator index function"))),
        }
    }

    /// Every legal move under `policy` with its index, ascending by index.
    /// Errs if a move has no index or two moves share one: either is a
    /// layout bug, which the fuzz and corpus checks rule out.
    pub fn rl_indexed_moves(&mut self, policy: CrossPlayerPolicy) -> Result<Vec<IndexedMove>, GameError> {
        self.rl_generator_layout()?;
        let moves = self.rl_generator_moves(policy)?;
        let mut out = Vec::with_capacity(moves.len());
        for mv in moves {
            out.push(IndexedMove { index: self.rl_index_of_move(&mv)?, mv });
        }
        out.sort_by_key(|x| x.index);
        if let Some(w) = out.windows(2).find(|w| w[0].index == w[1].index) {
            return Err(GameError::new(format!(
                "{}: index {} names two legal moves: {:?} and {:?}",
                self.title, w[0].index, w[0].mv.action, w[1].mv.action
            )));
        }
        Ok(out)
    }

    /// The legal move of index `idx` with its price set ([`priced`]).
    pub fn rl_move_for_index(
        &mut self,
        idx: u32,
        price: Option<i32>,
        policy: CrossPlayerPolicy,
    ) -> Result<Move, GameError> {
        let total = self.rl_generator_layout()?.total;
        if idx >= total {
            return Err(GameError::new(format!("index {idx} out of bounds (0-{})", total - 1)));
        }
        let mv = self
            .rl_indexed_moves(policy)?
            .into_iter()
            .find(|x| x.index == idx)
            .ok_or_else(|| GameError::new(format!("index {idx} is not legal at the current state")))?
            .mv;
        priced(mv, price)
    }

    /// The 18xx.games action dict of a generator move (state BEFORE it).
    pub fn rl_move_wire(&self, m: &Move) -> Result<Value, GameError> {
        match self.title.as_str() {
            "1867" => Ok(self.g1867_move_wire(m)),
            t => Err(GameError::new(format!("{t}: no wire form for generator moves"))),
        }
    }

    /// Apply a generator move and append its wire dict to the action log
    /// (so `raw_actions` replays the game, as after `process_action`).
    /// Returns the dict.
    pub fn rl_apply_move(&mut self, m: &Move) -> Result<Value, GameError> {
        let wire = self.rl_move_wire(m)?;
        self.process_action_internal(&m.action)?;
        self.action_log.push(wire.clone());
        Ok(wire)
    }

    /// Apply the move of index `idx` at `price` ([`BaseGame::rl_move_for_index`],
    /// [`BaseGame::rl_apply_move`]).
    pub fn rl_apply_index(
        &mut self,
        idx: u32,
        price: Option<i32>,
        policy: CrossPlayerPolicy,
    ) -> Result<Value, GameError> {
        let m = self.rl_move_for_index(idx, price, policy)?;
        self.rl_apply_move(&m)
    }
}

/// The policy argument of the Python surface: None = the title's,
/// "rules" = everything Ruby accepts, "restricted".
fn policy(game: &BaseGame, p: Option<&str>) -> PyResult<CrossPlayerPolicy> {
    crate::g1867::py::policy_of(game, p)
}

#[pymethods]
impl BaseGame {
    /// The state's legal indices in the title's RL layout (a generator
    /// title: 1867), ascending. `policy`: None = the title's cross-player
    /// policy, "rules" = every move Ruby accepts.
    #[pyo3(signature = (policy=None))]
    fn rl_legal_indices(&mut self, policy: Option<&str>) -> PyResult<Vec<u32>> {
        let p = self::policy(self, policy)?;
        Ok(self.rl_indexed_moves(p)?.into_iter().map(|x| x.index).collect())
    }

    /// The legal moves with their indices, as JSON: `[{"index", "block",
    /// "wire": <18xx.games action dict>, "range": [lo, hi, step] | null,
    /// "step": <step kind>}]`, ascending by index. A ranged move's dict
    /// holds `lo`; any price on the range's step is legal.
    #[pyo3(signature = (policy=None))]
    fn rl_legal_moves_json(&mut self, policy: Option<&str>) -> PyResult<String> {
        let p = self::policy(self, policy)?;
        let lo = self.rl_generator_layout()?;
        let mut out: Vec<Value> = Vec::new();
        for x in self.rl_indexed_moves(p)? {
            out.push(json!({
                "index": x.index,
                "block": lo.block_of(x.index).map(|b| b.spec.name.as_str()),
                "wire": self.rl_move_wire(&x.mv)?,
                "range": x.mv.price.map(|r| vec![r.lo, r.hi, r.step]),
                "step": format!("{:?}", x.mv.step),
            }));
        }
        Ok(serde_json::to_string(&out).unwrap_or_default())
    }

    /// The move of a legal index as an 18xx.games action dict (JSON), with
    /// `price` (on the move's range and step; None = its low end) for a
    /// price-bearing index. RuntimeError if the index is not legal.
    #[pyo3(signature = (idx, price=None, policy=None))]
    fn rl_decode_index(&mut self, idx: u32, price: Option<i32>, policy: Option<&str>) -> PyResult<String> {
        let p = self::policy(self, policy)?;
        let m = self.rl_move_for_index(idx, price, p)?;
        Ok(self.rl_move_wire(&m)?.to_string())
    }

    /// Apply the move of a legal index (at `price` for a price-bearing one)
    /// and log its action dict, as `process_action` would.
    #[pyo3(name = "rl_apply_index", signature = (idx, price=None, policy=None))]
    fn rl_apply_index_py(&mut self, idx: u32, price: Option<i32>, policy: Option<&str>) -> PyResult<()> {
        let p = self::policy(self, policy)?;
        self.rl_apply_index(idx, price, p)?;
        Ok(())
    }

    /// `(lo, hi, step)` of a legal price-bearing index; None for an index
    /// without a price; RuntimeError if the index is not legal.
    #[pyo3(signature = (idx, policy=None))]
    fn rl_price_range(&mut self, idx: u32, policy: Option<&str>) -> PyResult<Option<(i32, i32, i32)>> {
        let p = self::policy(self, policy)?;
        let m = self.rl_move_for_index(idx, None, p)?;
        Ok(m.price.map(|r| (r.lo, r.hi, r.step)))
    }
}
