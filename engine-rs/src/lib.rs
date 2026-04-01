use pyo3::prelude::*;

mod game;

/// Rust-accelerated game engine for 18xx board games.
#[pymodule]
fn engine_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<game::BaseGame>()?;
    Ok(())
}
