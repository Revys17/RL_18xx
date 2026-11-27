//! 1867's parts of what the board renders (docs/SITE_ENGINE_RS_1867.md §3.4): the snapshot's fields beside step 1's
//! (`view.rs`), in living-board's shapes (`types.ts`), equal to Opal's adapter.

use super::super::game::SiteGame;
use serde_json::{json, Value};

/// Whether a train type is Ruby's `num: 'unlimited'` (1867's 8, 2+2 and 5+5E: one at a time in the depot,
/// `Depot#clone_unlimited`).
pub fn unlimited(g: &SiteGame, name: &str) -> bool {
    let title = g.game.title_def();
    title.trains().iter().any(|t| t.name == name && title.num_trains(t, &g.game.optional_rules) == u32::MAX)
}

/// The depot's upcoming trains (Ruby `depot.upcoming`): an unlimited type shows one train at a time.
pub fn upcoming(g: &SiteGame) -> Vec<Value> {
    let mut out = Vec::new();
    let mut seen: Vec<&str> = Vec::new();
    for t in &g.game.depot.trains {
        if unlimited(g, &t.name) {
            if seen.contains(&t.name.as_str()) {
                continue;
            }
            seen.push(&t.name);
        }
        out.push(json!({"id": t.id, "name": t.name}));
    }
    out
}
