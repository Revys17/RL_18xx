//! PyO3 surface for 1822-family games (inspection for the replay
//! harness / Ruby diff rig, and the legal-action generator).

use pyo3::prelude::*;

use crate::entities::EntityId;
use crate::game::BaseGame;

use super::{Ent, RoundKind};

#[pymethods]
impl BaseGame {
    /// A JSON snapshot of the 1822 state in the shape of
    /// scripts/ruby_dump.rb's per-action rows (round, step, entity, phase,
    /// corps' price/cash/trains, player cash by id, bank).
    fn g1822_snapshot(&self) -> PyResult<String> {
        let Some(st) = self.g1822.as_deref() else {
            return Err(pyo3::exceptions::PyValueError::new_err("not an 1822-family game"));
        };
        let r = &st.round;
        let round = match r.kind {
            RoundKind::Stock => "Stock",
            RoundKind::Choices => "Choices",
            RoundKind::Operating => "Operating",
        };
        let active = self.g1822_active_step();
        let step = active.map(|i| format!("{:?}", r.steps[i]));
        let ent = active.and_then(|i| self.g1822_step_current_entity(i)).map(|e| match e {
            Ent::Player(p) => p.to_string(),
            Ent::Corp(c) | Ent::Company(c) => c,
        });
        let mut corps = serde_json::Map::new();
        for c in &self.corporations {
            if c.closed && c.ipo_price.is_none() {
                continue;
            }
            corps.insert(
                c.sym.clone(),
                serde_json::json!({
                    "p": c.share_price.as_ref().map(|s| s.price),
                    "cash": c.cash,
                    "fl": c.floated,
                    "closed": c.closed,
                    "tr": c.trains.iter().map(|t| t.name.clone()).collect::<Vec<_>>(),
                }),
            );
        }
        let pcash: serde_json::Map<String, serde_json::Value> = self
            .players
            .iter()
            .map(|p| (p.id.to_string(), serde_json::json!(p.cash)))
            .collect();
        let debt: serde_json::Map<String, serde_json::Value> = st
            .player_debt
            .iter()
            .map(|(p, d)| (p.to_string(), serde_json::json!(d)))
            .collect();
        let mut companies: serde_json::Map<String, serde_json::Value> = self
            .companies
            .iter()
            .filter(|c| !c.closed && !c.owner.is_none())
            .map(|c| (c.sym.clone(), serde_json::json!(c.owner.0)))
            .collect();
        for c in &st.bank_owned {
            if !self.co_closed(c) {
                companies.insert(c.clone(), serde_json::json!("The Bank"));
            }
        }
        // Ruby keeps ONE card of an unlimited type in `upcoming`; the
        // engine materializes the pool — show up to the first 7.
        let mut depot: Vec<String> = Vec::new();
        for t in self.depot.trains.iter().take(3) {
            depot.push(t.id.clone());
            if t.id.starts_with("7-") {
                break;
            }
        }
        let v = serde_json::json!({
            "round": round,
            "round_num": r.round_num,
            "turn": self.turn,
            "step": step,
            "ent": ent,
            "phase": self.phase.name,
            "corps": corps,
            "pcash": pcash,
            "debt": debt,
            "bank": self.bank.cash,
            "finished": self.finished,
            "companies": companies,
            "order": self.players.iter().map(|p| p.id).collect::<Vec<_>>(),
            "depot": depot,
            "game_end_reason": st.game_end_reason,
        });
        Ok(v.to_string())
    }

    /// The live company stack order (Ruby `@companies`) and bid boxes.
    fn g1822_companies_order(&self) -> Vec<String> {
        self.g1822.as_deref().map(|s| s.company_order.clone()).unwrap_or_default()
    }

    fn g1822_bidboxes(&self) -> (Vec<String>, Vec<String>, Vec<String>) {
        if self.g1822.is_none() {
            return (vec![], vec![], vec![]);
        }
        (self.bidbox_minors(), self.bidbox_concessions(), self.bidbox_privates())
    }

    /// The legal action types for the acting entity.
    fn g1822_legal_types(&self) -> Vec<String> {
        if self.g1822.is_none() {
            return vec![];
        }
        self.g1822_action_types()
    }

    /// Player value (Ruby player_value) by player id.
    fn g1822_player_values(&self) -> Vec<(u32, i32)> {
        self.players.iter().map(|p| (p.id, self.g1822_player_value(p.id))).collect()
    }

    #[allow(dead_code)]
    fn g1822_owner_of(&self, company: &str) -> String {
        self.co_owner(company).0
    }
}

#[allow(dead_code)]
fn _unused(_: EntityId) {}
