//! 1867 self-play support on top of the shared 1830/1867 rules engine
//! (`steps.rs`, `rounds/*`): the pieces only enumeration needs — the
//! replay path trusts a record for them (Ruby's `@game.loading`).
//!
//! * `router` — 1867's auto-router (bucketed train distances, x2 trains,
//!   hex bonuses, the Timmins capitals bonus, grouped Detroit), the
//!   engine's own `run_routes` for self-play.
//!
//! Everything here is 1867-only: shared code reaches it through
//! title-gated calls, so 1830 and 1822 never take these paths.

pub mod router;
