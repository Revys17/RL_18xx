//! 1867 auto-router (bridge design T2): the revenue-maximising legal run
//! for a corporation's unoperated trains under Ruby G1867's route rules.
//!
//! STUB — the interface the legal-move generator calls; the
//! implementation lands separately.

use crate::actions::RouteData;
use crate::game::BaseGame;

impl BaseGame {
    /// The revenue-maximising legal routes for `corp`'s unoperated trains
    /// (one `RouteData` per running train: train instance id, stop hexes,
    /// node-to-node `connections`, `nodes` signatures, the engine's price
    /// as `revenue`) and their total. Empty when nothing can run.
    pub(crate) fn g1867_auto_routes(&mut self, _corp: &str) -> (Vec<RouteData>, i32) {
        (Vec::new(), 0)
    }
}
