use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

use crate::core::SharePrice;

/// Identifies who owns something: a player, corporation, or the bank/market.
/// Stored as a simple string for PyO3 compatibility: "player:1", "corp:PRR", "bank", "market", "".
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct EntityId(pub String);

impl EntityId {
    pub fn player(id: u32) -> Self { EntityId(format!("player:{}", id)) }
    pub fn corporation(sym: &str) -> Self { EntityId(format!("corp:{}", sym)) }
    pub fn bank() -> Self { EntityId("bank".to_string()) }
    pub fn market() -> Self { EntityId("market".to_string()) }
    pub fn none() -> Self { EntityId(String::new()) }
    pub fn is_none(&self) -> bool { self.0.is_empty() }
}

/// A train card.
#[pyclass]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Train {
    #[pyo3(get)]
    pub name: String,
    #[pyo3(get)]
    pub distance: u32,
    #[pyo3(get)]
    pub price: i32,
    #[pyo3(get)]
    pub operated: bool,
    pub owner: EntityId,
}

#[pymethods]
impl Train {
    #[new]
    pub fn new(name: String, distance: u32, price: i32) -> Self {
        Train {
            name,
            distance,
            price,
            operated: false,
            owner: EntityId::none(),
        }
    }

    fn __repr__(&self) -> String {
        format!("Train(name='{}', price={})", self.name, self.price)
    }
}

/// A corporation token (placed on cities).
#[pyclass]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Token {
    #[pyo3(get)]
    pub price: i32,
    #[pyo3(get)]
    pub used: bool,
    #[pyo3(get)]
    pub token_type: String,
    /// ID of the corporation that owns this token.
    #[pyo3(get)]
    pub corporation_id: String,
    /// ID of the city this token is placed in (empty if unplaced).
    #[pyo3(get)]
    pub city_hex_id: String,
}

#[pymethods]
impl Token {
    #[new]
    pub fn new(corporation_id: String, price: i32) -> Self {
        Token {
            price,
            used: false,
            token_type: "normal".to_string(),
            corporation_id,
            city_hex_id: String::new(),
        }
    }
}

/// A single share certificate.
#[pyclass]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Share {
    #[pyo3(get)]
    pub corporation_id: String,
    #[pyo3(get)]
    pub percent: u8,
    #[pyo3(get)]
    pub president: bool,
    pub owner: EntityId,
}

#[pymethods]
impl Share {
    #[new]
    pub fn new(corporation_id: String, percent: u8, president: bool) -> Self {
        Share {
            corporation_id,
            percent,
            president,
            owner: EntityId::none(),
        }
    }
}

/// A private company.
#[pyclass]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Company {
    #[pyo3(get)]
    pub id: String,
    #[pyo3(get)]
    pub sym: String,
    #[pyo3(get)]
    pub name: String,
    #[pyo3(get)]
    pub value: i32,
    #[pyo3(get)]
    pub revenue: i32,
    #[pyo3(get)]
    pub closed: bool,
    pub owner: EntityId,
}

#[pymethods]
impl Company {
    #[new]
    pub fn new(sym: String, name: String, value: i32, revenue: i32) -> Self {
        Company {
            id: sym.clone(),
            sym,
            name,
            value,
            revenue,
            closed: false,
            owner: EntityId::none(),
        }
    }

    fn __repr__(&self) -> String {
        format!("Company(sym='{}', value={})", self.sym, self.value)
    }
}

/// A public corporation.
#[pyclass]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Corporation {
    #[pyo3(get)]
    pub id: String,
    #[pyo3(get)]
    pub sym: String,
    #[pyo3(get)]
    pub name: String,
    #[pyo3(get)]
    pub cash: i32,
    #[pyo3(get)]
    pub floated: bool,
    #[pyo3(get)]
    pub trains: Vec<Train>,
    #[pyo3(get)]
    pub tokens: Vec<Token>,
    #[pyo3(get)]
    pub shares: Vec<Share>,
    pub share_price: Option<SharePrice>,
    pub ipo_price: Option<SharePrice>,
    pub owner_id: EntityId,
}

#[pymethods]
impl Corporation {
    #[new]
    pub fn new(sym: String, name: String, tokens: Vec<Token>, shares: Vec<Share>) -> Self {
        Corporation {
            id: sym.clone(),
            sym,
            name,
            cash: 0,
            floated: false,
            trains: Vec::new(),
            tokens,
            shares,
            share_price: None,
            ipo_price: None,
            owner_id: EntityId::none(),
        }
    }

    #[getter]
    fn share_price(&self) -> Option<SharePrice> {
        self.share_price.clone()
    }

    #[getter]
    fn ipo_price(&self) -> Option<SharePrice> {
        self.ipo_price.clone()
    }

    fn __repr__(&self) -> String {
        format!("Corporation(sym='{}', floated={})", self.sym, self.floated)
    }
}

/// A player.
#[pyclass]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Player {
    #[pyo3(get)]
    pub id: u32,
    #[pyo3(get)]
    pub name: String,
    #[pyo3(get)]
    pub cash: i32,
}

#[pymethods]
impl Player {
    #[new]
    pub fn new(id: u32, name: String, cash: i32) -> Self {
        Player { id, name, cash }
    }

    fn __repr__(&self) -> String {
        format!("Player(id={}, name='{}', cash={})", self.id, self.name, self.cash)
    }
}

/// The bank.
#[pyclass]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Bank {
    #[pyo3(get)]
    pub cash: i32,
}

#[pymethods]
impl Bank {
    #[new]
    pub fn new(cash: i32) -> Self {
        Bank { cash }
    }
}

/// The train depot.
#[pyclass]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Depot {
    #[pyo3(get)]
    pub trains: Vec<Train>,
    #[pyo3(get)]
    pub discarded: Vec<Train>,
}

#[pymethods]
impl Depot {
    #[new]
    pub fn new() -> Self {
        Depot {
            trains: Vec::new(),
            discarded: Vec::new(),
        }
    }
}
