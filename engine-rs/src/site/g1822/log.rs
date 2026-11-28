//! The 1822 family's log lines (docs/SITE_ENGINE_RS_1822.md §4.2): Ruby's words for what an action did, read off the
//! position before and after it.

use super::super::game::SiteGame;
use crate::actions::Action;
use serde_json::Value;

/// The position before an action, as the log reads it.
pub struct Before {
    pub game: crate::game::BaseGame,
}

pub fn before(g: &SiteGame, _raw: &Value) -> Before {
    Before { game: g.game.clone_for_search_without_log() }
}

/// The lines written before the first action.
pub fn start(_g: &SiteGame) -> Vec<String> {
    Vec::new()
}

/// The lines an action wrote.
pub fn explain(_g: &SiteGame, _b: &Before, _raw: &Value, _action: &Action) -> Vec<String> {
    Vec::new()
}
