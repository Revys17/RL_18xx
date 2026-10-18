//! 1867 self-play support on top of the shared 1830/1867 rules engine
//! (`steps.rs`, `rounds/*`): the pieces only enumeration needs — the
//! replay path trusts a record for them (Ruby's `@game.loading`).
//!
//! * `legal` — the legal-move generator (every step × action type, or
//!   listed replay-only: `legal::g1867_generator_status`), its wire form
//!   (18xx.games action dicts) and the cross-player purchase policy.
//! * `tile_rules` — Ruby-faithful tile-lay legality for the generator.
//! * `graph` — Ruby's Graph#compute walk (connected hexes / nodes / paths
//!   as Ruby's player-time checks read them).
//! * `router` — 1867's auto-router (bucketed train distances, x2 trains,
//!   hex bonuses, the Timmins capitals bonus, grouped Detroit), the
//!   engine's own `run_routes` for self-play.
//! * `fuzz` — random walks over the generator (cargo tests).
//! * `py` — the PyO3 surface (`g1867_legal_moves_json`, ...).
//!
//! Everything here is 1867-only: shared code reaches it through
//! title-gated calls, so 1830 and 1822 never take these paths.

pub mod fuzz;
pub mod graph;
pub mod legal;
pub mod py;
pub mod router;
pub mod tile_rules;
