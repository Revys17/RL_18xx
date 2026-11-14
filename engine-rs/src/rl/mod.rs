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
//! * [`g1867`]: 1867's registry layout (`1867-v1`) and its move -> index
//!   function; [`g1822`]: the 1822 family's (`1822-v1`, `1822PNW-v1`,
//!   `1822MX-v1`, `1822CA-v1`); [`generator`]: the index API of generator
//!   titles (legal indices, decode, apply, price range;
//!   `rl_legal_moves_json`, ...).
//!
//! Titles whose legal moves come from their own generator get a registry
//! layout (`RlLayoutKind::Generator`; 1867 and the 1822 family).

use std::collections::HashMap;
use std::sync::OnceLock;

use crate::actions::GameError;
use crate::title::RlLayoutKind;

pub mod descriptor;
/// The title-generic typed-token state encoder (milestone B1, `docs/rl_encoder.md`).
pub mod encode;
pub mod factored_layout;
pub mod g1822;
pub mod g1867;
pub mod generator;
pub mod layout;
pub mod py;

pub use descriptor::{Descriptor, EntKind, Family, PriceKind};
pub use layout::{Axis, Block, BlockSpec, RlLayout, Vocab};

type Slot = (&'static dyn crate::title::GameTitle, OnceLock<Result<RlLayout, String>>);

/// The registry layout of every bridge-ready title, each built once, on
/// its first use (the 1822 family's layouts hold 0.2-0.75M descriptors: a
/// process that plays 1830 alone never builds them).
fn registry() -> &'static HashMap<&'static str, Slot> {
    static LAYOUTS: OnceLock<HashMap<&'static str, Slot>> = OnceLock::new();
    LAYOUTS.get_or_init(|| {
        crate::title::all_titles()
            .iter()
            .filter(|t| t.rl_layout_kind() != RlLayoutKind::None)
            .map(|t| (t.name(), (*t, OnceLock::new())))
            .collect()
    })
}

fn build(t: &'static dyn crate::title::GameTitle) -> Result<RlLayout, String> {
    match t.rl_layout_kind() {
        RlLayoutKind::None => Err(format!("{}: no RL layout", t.name())),
        RlLayoutKind::Factored => crate::action_index::try_layout_for(t.name())
            .map_err(|e| e.message)
            .and_then(|lo| factored_layout::factored_registry(t, lo)),
        RlLayoutKind::Generator => generator::registry_layout(t),
    }
}

/// A title's RL layout (blocks, descriptors, vocabularies, fingerprint), or
/// the "not bridge-ready" error.
pub fn rl_layout(title: &str) -> Result<&'static RlLayout, GameError> {
    match registry().get(title).map(|(t, cell)| cell.get_or_init(|| build(*t))) {
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
