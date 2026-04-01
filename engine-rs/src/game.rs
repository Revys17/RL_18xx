use pyo3::prelude::*;

/// Stub BaseGame for the 1830 board game.
///
/// This is the Phase 0 scaffold — it proves the Rust/PyO3/maturin
/// build pipeline works end-to-end. Real game logic will be added
/// in subsequent phases.
#[pyclass]
pub struct BaseGame {
    #[pyo3(get)]
    pub title: String,
    #[pyo3(get)]
    pub finished: bool,
}

#[pymethods]
impl BaseGame {
    #[new]
    fn new() -> Self {
        BaseGame {
            title: "1830".to_string(),
            finished: false,
        }
    }

    fn __repr__(&self) -> String {
        format!("BaseGame(title='{}', finished={})", self.title, self.finished)
    }
}
