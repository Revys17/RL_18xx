//! 1867's step auto actions (docs/SITE_ENGINE_RS_1867.md §3.3): the passes Ruby makes after a move without any
//! program (`round.auto_actions`: the active step's `auto_actions(current_entity)`, repeated until none), predicted
//! here so a Rust page can apply a move at once and post it; the server's stored action must carry the same ones.
//!
//! | Step | Ruby | Rust predicate (on the position after the move) |
//! |---|---|---|
//! | `G1867::Step::Token` | `pass` unless `can_really_place_token?` | the active step is Token and the generator offers no `place_token` |
//! | `G1867::Step::Merge` | `pass` when not converting / naming a major / merging, the minor can't convert and `mergeable_candidates` is empty | the same, from the engine's merger state |
//! | `G1867::Step::BuyCompanyPreloan` | `pass` when the company has no loans | the active step is BuyCompanyPreloan and the company's `loans == 0` |
//!
//! Programs' auto actions (`program_merger_pass`, `program_buy_shares`, `program_share_pass`) stay the server's.

use crate::actions::Action;
use crate::game::BaseGame;
use crate::rounds::Round;
use crate::steps::{StepEntity, StepKind};
use crate::title::CrossPlayerPolicy;
use serde_json::{json, Value};

/// The step auto action Ruby would make now, if any: the passing entity (a corporation's sym).
fn next_auto(g: &mut BaseGame) -> Option<String> {
    if g.finished {
        return None;
    }
    let (kind, e) = super::active_step(g)?;
    let StepEntity::Corp(ent) = e else { return None };
    let auto = match kind {
        StepKind::Token => {
            let moves = g.g1867_legal_moves_with(CrossPlayerPolicy::RULES);
            !moves.iter().any(|m| m.action.action_type() == "place_token")
        }
        StepKind::BuyCompanyPreloan => g.corp_idx.get(ent.as_str()).is_some_and(|&ci| g.corporations[ci].loans == 0),
        StepKind::Merge => match &g.round {
            Round::Merger(s) if !s.converting && !s.merge_major && s.merging.is_empty() => {
                let s = s.clone();
                !g.merger_can_convert(&ent) && g.merger_mergeable_candidates(&s, &ent).is_empty()
            }
            _ => false,
        },
        _ => false,
    };
    auto.then_some(ent)
}

/// The step auto actions after the position `g` is in, as the site stores them (`{type: "pass", entity,
/// entity_type: "corporation"}`), processed on a copy.
pub fn predict(g: &BaseGame) -> Vec<Value> {
    let mut c = g.clone_for_search();
    let mut out = Vec::new();
    for _ in 0..16 {
        let Some(ent) = next_auto(&mut c) else { break };
        if c.process_action_internal(&Action::Pass { entity_id: ent.clone() }).is_err() {
            break;
        }
        out.push(json!({"type": "pass", "entity": ent, "entity_type": "corporation"}));
    }
    out
}
