use pyo3::prelude::*;

pub mod actions;
pub mod core;
pub mod entities;
pub mod game;
pub mod graph;
pub mod map;
pub mod rounds;
pub mod router;
pub mod tiles;
pub mod title;

/// Rust-accelerated game engine for 18xx board games.
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

    Ok(())
}
