use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

/// A share price cell on the stock market.
#[pyclass]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SharePrice {
    #[pyo3(get)]
    pub price: i32,
    #[pyo3(get)]
    pub row: u8,
    #[pyo3(get)]
    pub column: u8,
    #[pyo3(get)]
    pub types: Vec<String>,
}

#[pymethods]
impl SharePrice {
    #[new]
    pub fn new(price: i32, row: u8, column: u8, types: Vec<String>) -> Self {
        SharePrice { price, row, column, types }
    }

    fn __repr__(&self) -> String {
        format!("SharePrice(price={}, types={:?})", self.price, self.types)
    }
}

/// Game phase (e.g. "2", "3", "4", "5", "6", "D").
#[pyclass]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Phase {
    #[pyo3(get)]
    pub name: String,
    #[pyo3(get)]
    pub operating_rounds: u8,
    #[pyo3(get, name = "_train_limit")]
    pub train_limit: u8,
    #[pyo3(get)]
    pub tiles: Vec<String>,
}

#[pymethods]
impl Phase {
    #[new]
    pub fn new(name: String, operating_rounds: u8, train_limit: u8, tiles: Vec<String>) -> Self {
        Phase { name, operating_rounds, train_limit, tiles }
    }

    fn __repr__(&self) -> String {
        format!("Phase(name='{}', or={}, tl={})", self.name, self.operating_rounds, self.train_limit)
    }
}
