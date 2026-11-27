//! Wall-clock readings for diagnostics only (the 1867 generator's and router's timing counters). On native builds it
//! is `std::time::Instant` itself; on wasm32-unknown-unknown, where `std::time::Instant::now` panics, a stand-in whose
//! readings are all zero (docs/SITE_ENGINE_RS_1867.md §5.3, P3). Nothing the engine decides reads it.

#[cfg(not(target_arch = "wasm32"))]
pub use std::time::Instant;

/// The wasm stand-in: every reading is the same instant, every duration zero.
#[cfg(target_arch = "wasm32")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Instant;

#[cfg(target_arch = "wasm32")]
impl Instant {
    pub fn now() -> Instant {
        Instant
    }
    pub fn elapsed(&self) -> std::time::Duration {
        std::time::Duration::ZERO
    }
}

#[cfg(target_arch = "wasm32")]
impl std::ops::Sub for Instant {
    type Output = std::time::Duration;
    fn sub(self, _rhs: Instant) -> std::time::Duration {
        std::time::Duration::ZERO
    }
}
