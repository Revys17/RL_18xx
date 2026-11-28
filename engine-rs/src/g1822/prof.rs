//! The 1822 family generator's per-section profile (off by default; the
//! same shape as 1867's `g1867::legal::stats` profile): `(count, ns)` per
//! named section, read and cleared by `g1822_profile_take` (Python
//! `BaseGame.g1822_profile(on)` / `g1822_profile_take()`;
//! `scripts/g1822_generator_profile.py`).

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

static PROFILING: AtomicBool = AtomicBool::new(false);
static PROFILE: Mutex<BTreeMap<String, (u64, u64)>> = Mutex::new(BTreeMap::new());

/// Turn the profile on or off.
pub fn set_profiling(on: bool) {
    PROFILING.store(on, Ordering::Relaxed);
}

#[inline]
pub fn profiling() -> bool {
    PROFILING.load(Ordering::Relaxed)
}

/// A section's start (None while the profile is off).
#[inline]
pub fn start() -> Option<crate::clock::Instant> {
    profiling().then(crate::clock::Instant::now)
}

/// Credit the time since `t0` to section `key` (a no-op while off).
#[inline]
pub fn stop(t0: Option<crate::clock::Instant>, key: impl FnOnce() -> String) {
    let Some(t0) = t0 else { return };
    record(key, t0.elapsed().as_nanos() as u64);
}

/// Credit `ns` to section `key` (a no-op while off).
pub fn record(key: impl FnOnce() -> String, ns: u64) {
    if !profiling() {
        return;
    }
    // (the key first: computing it may itself record a section)
    let k = key();
    let mut p = PROFILE.lock().unwrap_or_else(|e| e.into_inner());
    let e = p.entry(k).or_insert((0, 0));
    e.0 += 1;
    e.1 += ns;
}

/// Count an event in section `key` without time.
pub fn count(key: impl FnOnce() -> String) {
    if !profiling() {
        return;
    }
    let k = key();
    let mut p = PROFILE.lock().unwrap_or_else(|e| e.into_inner());
    p.entry(k).or_insert((0, 0)).0 += 1;
}

/// The profile so far, then cleared: `(section, count, ns)`.
pub fn take() -> Vec<(String, u64, u64)> {
    let mut p = PROFILE.lock().unwrap_or_else(|e| e.into_inner());
    std::mem::take(&mut *p).into_iter().map(|(k, (c, ns))| (k, c, ns)).collect()
}
