//! PyO3 surface of the 1867 legal-move generator.

use pyo3::prelude::*;
use serde_json::{json, Value};

use crate::game::BaseGame;
use crate::steps::StepKind;
use crate::title::CrossPlayerPolicy;

use super::legal::{g1867_generator_status, GenStatus};

/// `None` = the title's policy, "rules" = everything Ruby allows,
/// "restricted" = no cross-player purchases.
fn policy_of(game: &BaseGame, policy: Option<&str>) -> PyResult<CrossPlayerPolicy> {
    match policy {
        None | Some("title") => Ok(game.title_def().generator_cross_player()),
        Some("rules") => Ok(CrossPlayerPolicy::RULES),
        Some("restricted") => Ok(CrossPlayerPolicy::RESTRICTED),
        Some(other) => Err(pyo3::exceptions::PyValueError::new_err(format!("unknown policy {other:?}"))),
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

    /// A train id's (name, owner): owner "depot" (upcoming or discarded)
    /// or the holding corporation's sym ("" if unknown).
    pub(crate) fn g1867_train_info(&self, train: &str) -> (String, String) {
        let name = train.rsplit_once('-').map_or(train, |(n, _)| n).to_string();
        if self.depot.trains.iter().chain(self.depot.discarded.iter()).any(|t| t.id == train) {
            return (name, "depot".into());
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
