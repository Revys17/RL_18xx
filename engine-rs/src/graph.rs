use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::entities::Token;
use crate::tiles::{PathDef, TileColor};

/// A city on a tile (provides revenue, has token slots).
#[pyclass]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct City {
    #[pyo3(get)]
    pub revenue: i32,
    #[pyo3(get)]
    pub slots: u8,
    #[pyo3(get)]
    pub tokens: Vec<Option<Token>>,
}

#[pymethods]
impl City {
    #[new]
    pub fn new(revenue: i32, slots: u8) -> Self {
        let tokens = vec![None; slots as usize];
        City {
            revenue,
            slots,
            tokens,
        }
    }
}

/// A town on a tile (provides revenue, no token slots).
#[pyclass]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Town {
    #[pyo3(get)]
    pub revenue: i32,
}

#[pymethods]
impl Town {
    #[new]
    pub fn new(revenue: i32) -> Self {
        Town { revenue }
    }
}

/// An offboard location (red hex, phase-dependent revenue).
#[pyclass]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Offboard {
    #[pyo3(get)]
    pub revenue: i32,
}

#[pymethods]
impl Offboard {
    #[new]
    pub fn new(revenue: i32) -> Self {
        Offboard { revenue }
    }
}

/// A tile edge (direction 0-5).
#[pyclass]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Edge {
    #[pyo3(get)]
    pub num: u8,
}

#[pymethods]
impl Edge {
    #[new]
    pub fn new(num: u8) -> Self {
        Edge { num }
    }
}

/// A tile upgrade option.
#[pyclass]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Upgrade {
    #[pyo3(get)]
    pub cost: i32,
    #[pyo3(get)]
    pub terrain: String,
}

#[pymethods]
impl Upgrade {
    #[new]
    pub fn new(cost: i32, terrain: String) -> Self {
        Upgrade { cost, terrain }
    }
}

/// A tile placed on the map.
#[pyclass]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Tile {
    #[pyo3(get)]
    pub id: String,
    #[pyo3(get)]
    pub name: String,
    #[pyo3(get)]
    pub rotation: u8,
    #[pyo3(get)]
    pub cities: Vec<City>,
    #[pyo3(get)]
    pub towns: Vec<Town>,
    #[pyo3(get)]
    pub edges: Vec<Edge>,
    #[pyo3(get)]
    pub offboards: Vec<Offboard>,
    #[pyo3(get)]
    pub upgrades: Vec<Upgrade>,

    // Phase 4: connectivity data
    /// Parsed path definitions for graph traversal.
    pub paths: Vec<PathDef>,
    /// Tile color (white/yellow/green/brown/gray/red).
    pub color: TileColor,
    /// Label (e.g., "B", "NY", "OO") for upgrade matching.
    pub label: Option<String>,
}

#[pymethods]
impl Tile {
    #[new]
    pub fn new(id: String, name: String) -> Self {
        Tile {
            id,
            name,
            rotation: 0,
            cities: Vec::new(),
            towns: Vec::new(),
            edges: Vec::new(),
            offboards: Vec::new(),
            upgrades: Vec::new(),
            paths: Vec::new(),
            color: TileColor::White,
            label: None,
        }
    }

    fn __repr__(&self) -> String {
        format!("Tile(id='{}', rotation={})", self.id, self.rotation)
    }
}

/// A hex on the game map.
#[pyclass]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Hex {
    #[pyo3(get)]
    pub id: String,
    #[pyo3(get)]
    pub tile: Tile,
    /// Neighbor hex IDs by direction (0-5). Only includes neighbors that exist.
    pub neighbors: HashMap<u8, String>,
    /// All neighbor hex IDs including across-border connections.
    pub all_neighbors: HashMap<u8, String>,
}

#[pymethods]
impl Hex {
    #[new]
    pub fn new(id: String, tile: Tile) -> Self {
        Hex {
            id,
            tile,
            neighbors: HashMap::new(),
            all_neighbors: HashMap::new(),
        }
    }

    /// Get all_neighbors as a Python dict.
    #[getter]
    fn all_neighbors(&self) -> HashMap<u8, String> {
        self.all_neighbors.clone()
    }

    fn __repr__(&self) -> String {
        format!("Hex(id='{}', tile={})", self.id, self.tile.id)
    }
}
