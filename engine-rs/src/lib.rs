// Builds without Python (the site layer's tools and the wasm package, docs/SITE_ENGINE_RS.md §4.1) leave the
// Python-only call sites out, so code only they reach is dead there; the wheel build keeps every warning.
#![cfg_attr(not(feature = "python"), allow(dead_code))]
#[cfg(feature = "python")]
use pyo3::prelude::*;

/// The titles this engine build can construct (`BaseGame.new_titled`).
#[cfg(feature = "python")]
#[pyfunction]
fn supported_titles_py() -> Vec<&'static str> {
    title::all_titles().iter().map(|t| t.name()).collect()
}

pub mod pyerr;
pub mod abilities;
pub mod action_index;
pub mod clock;
pub mod actions;
pub mod core;
pub mod decode;
pub mod encoder;
pub mod entities;
pub mod factored;
pub mod g1822;
pub mod g1867;
pub mod game;
pub mod graph;
pub mod gumbel;
pub mod map;
pub mod mcts;
pub mod mcts_lockstep;
/// Training neutrality: pinned 1830 self-play walks (docs/SITE_ENGINE_RS.md §8.4).
#[cfg(test)]
mod neutrality;
pub mod price_pmf;
pub mod revenue;
pub mod rl;
pub mod rounds;
pub mod router;
/// The site layer (docs/SITE_ENGINE_RS.md): `records` and `rules` in every build, the rest behind `site`.
pub mod site;
pub mod steps;
pub mod tiles;
pub mod title;

/// Rust-accelerated game engine for 18xx board games.
#[cfg(feature = "python")]
#[pymodule]
fn engine_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Game
    m.add_class::<game::BaseGame>()?;
    m.add_class::<game::RoundState>()?;

    // Core
    m.add_class::<core::SharePrice>()?;
    m.add_class::<core::Phase>()?;

    // Entities
    m.add_class::<entities::Player>()?;
    m.add_class::<entities::Corporation>()?;
    m.add_class::<entities::Company>()?;
    m.add_class::<entities::Bank>()?;
    m.add_class::<entities::Depot>()?;
    m.add_class::<entities::Train>()?;
    m.add_class::<entities::Token>()?;
    m.add_class::<entities::Share>()?;

    // Graph
    m.add_class::<graph::Hex>()?;
    m.add_class::<graph::Tile>()?;
    m.add_class::<graph::City>()?;
    m.add_class::<graph::Town>()?;
    m.add_class::<graph::Offboard>()?;
    m.add_class::<graph::Edge>()?;
    m.add_class::<graph::Upgrade>()?;

    // Rust MCTS (Phase 4a scaffold)
    m.add_class::<mcts::RustMCTSPlayer>()?;
    m.add_function(wrap_pyfunction!(mcts_lockstep::mcts_collect_leaves, m)?)?;
    m.add_function(wrap_pyfunction!(mcts_lockstep::mcts_collect_leaf_games, m)?)?;
    m.add_function(wrap_pyfunction!(mcts_lockstep::mcts_apply_results, m)?)?;
    m.add_function(wrap_pyfunction!(mcts_lockstep::mcts_discard_pending, m)?)?;
    m.add_function(wrap_pyfunction!(mcts_lockstep::encode_games, m)?)?;
    m.add_function(wrap_pyfunction!(mcts_lockstep::mcts_terminal_value, m)?)?;
    m.add_function(wrap_pyfunction!(mcts_lockstep::mcts_decider_id, m)?)?;
    m.add_function(wrap_pyfunction!(action_index::action_offsets_py, m)?)?;
    m.add_function(wrap_pyfunction!(action_index::policy_size_py, m)?)?;
    m.add_function(wrap_pyfunction!(action_index::policy_size_for_py, m)?)?;
    m.add_function(wrap_pyfunction!(supported_titles_py, m)?)?;
    m.add_function(wrap_pyfunction!(action_index::legal_action_to_index_py, m)?)?;
    m.add_function(wrap_pyfunction!(price_pmf::price_pmf_counts, m)?)?;
    m.add_function(wrap_pyfunction!(price_pmf::price_pmf_cell_of, m)?)?;
    m.add_function(wrap_pyfunction!(price_pmf::price_pmf_member, m)?)?;
    rl::py::register(m)?;

    Ok(())
}
