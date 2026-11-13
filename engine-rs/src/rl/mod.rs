//! The title-generic RL bridge (design `docs/rl_bridge_design.md` §2,
//! schema `docs/rl_descriptors.md`).
//!
//! * [`descriptor`]: the descriptor schema and its global vocabularies.
//! * [`layout`]: the block registry (families over axes from title data),
//!   descriptors per index, the layout fingerprint.
//! * [`factored_layout`]: the registry form of the frozen 1830 layout.
//! * [`py`]: the Python surface (`rl_layout_info`, `rl_title_status`,
//!   `rl_descriptor_vocab`).
//!
//! Titles whose legal moves come from their own generator (1867,
//! `g1867/legal.rs`; the 1822 family) get registry layouts here in a later
//! milestone; until then their `rl_layout_kind` is `None` and every index
//! path raises "not bridge-ready".

use std::collections::HashMap;
use std::sync::OnceLock;

use crate::actions::GameError;
use crate::title::RlLayoutKind;

pub mod descriptor;
pub mod factored_layout;
pub mod layout;
pub mod py;

pub use descriptor::{Descriptor, EntKind, Family, PriceKind};
pub use layout::{Axis, Block, BlockSpec, RlLayout, Vocab};

/// The registry layout of every bridge-ready title, built once.
fn registry() -> &'static HashMap<&'static str, Result<RlLayout, String>> {
    static LAYOUTS: OnceLock<HashMap<&'static str, Result<RlLayout, String>>> = OnceLock::new();
    LAYOUTS.get_or_init(|| {
        crate::title::all_titles()
            .iter()
            .filter_map(|t| match t.rl_layout_kind() {
                RlLayoutKind::None => None,
                RlLayoutKind::Factored => {
                    let built = crate::action_index::try_layout_for(t.name())
                        .map_err(|e| e.message)
                        .and_then(|lo| factored_layout::factored_registry(*t, lo));
                    Some((t.name(), built))
                }
            })
            .collect()
    })
}

/// A title's RL layout (blocks, descriptors, vocabularies, fingerprint), or
/// the "not bridge-ready" error.
pub fn rl_layout(title: &str) -> Result<&'static RlLayout, GameError> {
    match registry().get(title) {
        Some(Ok(lo)) => Ok(lo),
        Some(Err(e)) => Err(GameError::new(format!("{title}: the RL layout does not build: {e}"))),
        None => {
            // A registered title without a layout, or an unknown one.
            crate::action_index::try_layout_for(title)?;
            Err(crate::action_index::not_bridge_ready(title))
        }
    }
}

#[cfg(test)]
mod tests;
