//! PyO3 surface of the 1867 legal-move generator.

#[cfg(not(feature = "python"))]
#[allow(unused_imports)]
use crate::pyerr::{PyErr, PyResult};
#[cfg(not(feature = "python"))]
#[allow(unused_imports)]
use pyo3_noop::{pyclass, pymethods};
#[cfg(feature = "python")]

use pyo3::prelude::*;
use serde_json::{json, Value};

use crate::game::BaseGame;
use crate::steps::StepKind;
use crate::title::CrossPlayerPolicy;

use super::legal::{g1867_generator_status, GenStatus};

/// `None` = the title's policy, "rules" = everything Ruby allows,
/// "restricted" = no cross-player purchases.
pub(crate) fn policy_of(game: &BaseGame, policy: Option<&str>) -> PyResult<CrossPlayerPolicy> {
    match policy {
        None | Some("title") => Ok(game.title_def().generator_cross_player()),
        Some("rules") => Ok(CrossPlayerPolicy::RULES),
        Some("restricted") => Ok(CrossPlayerPolicy::RESTRICTED),
        Some(other) => Err(crate::pyerr::exceptions::PyValueError::new_err(format!("unknown policy {other:?}"))),
    }
}

/// Every 1867 step kind (the status table's rows).
const KINDS: &[StepKind] = &[
    StepKind::SingleItemAuction,
    StepKind::MajorTrainless,
    StepKind::DiscardTrain,
    StepKind::HomeToken,
    StepKind::BuySellParShares,
    StepKind::Bankrupt,
    StepKind::BuyCompany,
    StepKind::RedeemShares,
    StepKind::Track,
    StepKind::Token,
    StepKind::Route,
    StepKind::Dividend,
    StepKind::BuyCompanyPreloan,
    StepKind::LoanOperations,
    StepKind::BuyTrain,
    StepKind::ReduceTokens,
    StepKind::PostMergerShares,
    StepKind::Merge,
];

/// Every action type a step can offer (`steps.rs` step_actions arms).
const TYPES: &[&str] = &[
    "bid", "pass", "par", "buy_shares", "sell_shares", "choose", "discard_train", "place_token", "buy_company",
    "lay_tile", "run_routes", "dividend", "buy_train", "bankrupt", "merge", "convert", "remove_token",
];

#[pymethods]
impl BaseGame {
    /// The 1867 legal moves as JSON: `[{"wire": <18xx.games action dict>,
    /// "range": [lo, hi, step] | null, "step": <step kind>}]`. A ranged
    /// move's dict holds `lo`.
    #[pyo3(signature = (policy=None))]
    pub(crate) fn g1867_legal_moves_json(&mut self, policy: Option<&str>) -> PyResult<String> {
        let policy = policy_of(self, policy)?;
        let moves = self.g1867_legal_moves_with(policy);
        let out: Vec<Value> = moves
            .iter()
            .map(|m| {
                json!({
                    "wire": self.g1867_move_wire(m),
                    "range": m.price.map(|p| vec![p.lo, p.hi, p.step]),
                    "step": format!("{:?}", m.step),
                })
            })
            .collect();
        Ok(serde_json::to_string(&out).unwrap_or_default())
    }

    /// Debug: every candidate before the apply filter, with its error.
    pub(crate) fn g1867_candidates_debug_py(&mut self) -> Vec<(String, Option<String>)> {
        self.g1867_candidates_debug()
    }

    /// The step machine's offers right now: `[(step kind, entity, types)]`.
    pub(crate) fn g1867_step_offers(&mut self) -> Vec<(String, String, Vec<String>)> {
        self.step_offers()
            .into_iter()
            .map(|o| {
                let ent = match &o.entity {
                    crate::steps::StepEntity::Player(p) => p.to_string(),
                    crate::steps::StepEntity::Corp(c) => c.clone(),
                    crate::steps::StepEntity::Company(c) => format!("company:{c}"),
                };
                (format!("{:?}", o.kind), ent, o.types.iter().map(|t| t.to_string()).collect())
            })
            .collect()
    }

    /// Whether processing `choice` (a MajorTrainless `choose` by `corp`)
    /// now walks Ruby HEAD into its stale-operator dead end: the corp is
    /// the operating round's current operator and other trainless majors
    /// are still queued, so Ruby's nationalize! does not force the next
    /// entity and the round keeps the pre-reset object as operator (no
    /// action by id matches it again). The engine ends the turn instead.
    pub(crate) fn g1867_ruby_dead_end_choice(&self, corp: &str) -> bool {
        let crate::rounds::Round::Operating(s) = &self.round else { return false };
        s.current_corp_sym() == Some(corp)
            && self.trainless_major.iter().any(|m| m == corp)
            && self.trainless_major.len() > 1
    }

    /// The owner of a share certificate by Ruby id ("CNR_3"):
    /// "player:<id>", "market", "ipo:<corp>" (treasury) or "corp:<sym>".
    pub(crate) fn g1867_share_owner(&self, cert: &str) -> String {
        let Some((sym, idx)) = cert.rsplit_once('_') else { return String::new() };
        let (Some(&ci), Ok(i)) = (self.corp_idx.get(sym), idx.parse::<usize>()) else { return String::new() };
        match self.corporations[ci].shares.get(i) {
            Some(sh) => sh.owner.0.clone(),
            None => String::new(),
        }
    }

    /// A train id's (name, owner): owner "depot" (upcoming), "discard"
    /// (the bank pool) or the holding corporation's sym ("" if unknown).
    pub(crate) fn g1867_train_info(&self, train: &str) -> (String, String) {
        let name = train.rsplit_once('-').map_or(train, |(n, _)| n).to_string();
        if self.depot.trains.iter().any(|t| t.id == train) {
            return (name, "depot".into());
        }
        if self.depot.discarded.iter().any(|t| t.id == train) {
            return (name, "discard".into());
        }
        for c in &self.corporations {
            if c.trains.iter().any(|t| t.id == train) {
                return (name, c.sym.clone());
            }
        }
        (name, String::new())
    }

    /// A Ruby city id ("C17-4-0", "57-1-0") as (hex, city index).
    pub(crate) fn g1867_city_ref(&self, city: &str) -> (String, u8) {
        let parts: Vec<&str> = city.split('-').collect();
        if parts.len() < 3 {
            return (city.to_string(), 0);
        }
        let tile = format!("{}-{}", parts[0], parts[1]);
        let ci = parts[2].parse::<u8>().unwrap_or(0);
        for (hi, h) in self.hexes.iter().enumerate() {
            if self.g1867_wire_tile_id(hi) == tile || h.tile.id == tile {
                return (h.id.clone(), ci);
            }
        }
        // A preprinted tile named by its hex.
        (parts[0].to_string(), ci)
    }

    /// The legal moves (`g1867_legal_moves_json`'s list) as one string per
    /// move — step, action, price range, certificate ids — for exact
    /// comparisons; `reference=True` runs the generator's reference paths
    /// (the code before the 2026-11-13 speed-ups: the differential test's
    /// oracle, `scripts/g1867_generator_diff.py`).
    #[pyo3(signature = (policy=None, reference=false))]
    pub(crate) fn g1867_moves_debug(&mut self, policy: Option<&str>, reference: bool) -> PyResult<Vec<String>> {
        let policy = policy_of(self, policy)?;
        let moves = super::legal::reference::with(reference, || self.g1867_legal_moves_with(policy));
        Ok(BaseGame::g1867_moves_strings(&moves))
    }

    /// `rl_legal_moves_json` through the generator's reference paths (or
    /// not): indices, blocks, wire dicts and ranges of the oracle's list.
    #[pyo3(signature = (policy=None, reference=false))]
    pub(crate) fn rl_legal_moves_json_with(&mut self, policy: Option<&str>, reference: bool) -> PyResult<String> {
        super::legal::reference::with(reference, || self.rl_legal_moves_json(policy))
    }

    /// The differential check of one state under `policy`: the generator's
    /// move list against the reference generator's, (1) move for move
    /// ([`BaseGame::g1867_moves_strings`], in list order) and (2) at the RL
    /// index level (each move's `1867-v1` index, wire dict and price range,
    /// ascending by index). Returns `(moves compared, None)` when both
    /// agree, else a description of the first difference.
    #[pyo3(signature = (policy=None))]
    pub(crate) fn g1867_reference_diff(&mut self, policy: Option<&str>) -> PyResult<(usize, Option<String>)> {
        let p = policy_of(self, policy)?;
        let fast = self.g1867_legal_moves_with(p);
        let slow = self.g1867_legal_moves_reference(p);
        let (a, b) = (BaseGame::g1867_moves_strings(&fast), BaseGame::g1867_moves_strings(&slow));
        if a != b {
            let only_new: Vec<&String> = a.iter().filter(|x| !b.contains(x)).collect();
            let only_ref: Vec<&String> = b.iter().filter(|x| !a.contains(x)).collect();
            return Ok((
                a.len(),
                Some(format!(
                    "moves differ ({} new vs {} reference); only new: {only_new:?}; only reference: {only_ref:?}; \
                     same set, other order: {}",
                    a.len(),
                    b.len(),
                    only_new.is_empty() && only_ref.is_empty()
                )),
            ));
        }
        let rows = |g: &BaseGame, ms: &[super::legal::Move]| -> PyResult<Vec<(u32, String)>> {
            let mut v = Vec::with_capacity(ms.len());
            for m in ms {
                let range = m.price.map(|r| (r.lo, r.hi, r.step));
                v.push((g.rl_index_of_move(m)?, format!("{}|{range:?}", g.rl_move_wire(m)?)));
            }
            v.sort();
            Ok(v)
        };
        let (ia, ib) = (rows(self, &fast)?, rows(self, &slow)?);
        if ia != ib {
            return Ok((a.len(), Some(format!("index rows differ: new {ia:?} reference {ib:?}"))));
        }
        if let Some(w) = ia.windows(2).find(|w| w[0].0 == w[1].0) {
            return Ok((a.len(), Some(format!("index {} names two moves: {} / {}", w[0].0, w[0].1, w[1].1))));
        }
        Ok((a.len(), None))
    }

    /// Run every later 1867 generator call on this thread through the
    /// reference paths (True) or the fast ones (False, the default) — for
    /// timing whole pipelines before / after (`scripts/bench_generic_server.py
    /// worker --title 1867 --reference`). Returns the previous setting.
    #[staticmethod]
    pub(crate) fn g1867_set_reference_generator(on: bool) -> bool {
        super::legal::reference::set(on)
    }

    /// Turn the generator's per-section profile on or off (diagnostics:
    /// `scripts/g1867_generator_profile.py`).
    #[staticmethod]
    pub(crate) fn g1867_profile(on: bool) {
        super::legal::stats::set_profiling(on);
    }

    /// The profile since the last call: `[(section, count, ns)]`; cleared.
    #[staticmethod]
    pub(crate) fn g1867_profile_take() -> Vec<(String, u64, u64)> {
        super::legal::stats::take()
    }

    /// The RL-readiness table: `{step: {action type: "generated" |
    /// "replay-only: <reason>"}}` for every 1867 step kind × action type.
    #[staticmethod]
    pub(crate) fn g1867_generator_status_json() -> String {
        let mut out = serde_json::Map::new();
        for k in KINDS {
            let mut row = serde_json::Map::new();
            for t in TYPES {
                let s = match g1867_generator_status(*k, t) {
                    GenStatus::Generated => "generated".to_string(),
                    GenStatus::ReplayOnly(r) => format!("replay-only: {r}"),
                };
                row.insert(t.to_string(), json!(s));
            }
            out.insert(format!("{k:?}"), Value::Object(row));
        }
        Value::Object(out).to_string()
    }
}
