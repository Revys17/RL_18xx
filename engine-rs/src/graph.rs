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
    /// Brown-phase revenue (used when phase allows brown tiles).
    #[pyo3(get)]
    pub brown_revenue: Option<i32>,
    /// Green/gray-phase revenue tiers (4-tier titles like 1867; None for
    /// 1830's yellow|brown offboards).
    #[pyo3(get)]
    #[serde(default)]
    pub green_revenue: Option<i32>,
    #[pyo3(get)]
    #[serde(default)]
    pub gray_revenue: Option<i32>,
    /// Ruby `groups:` — a route may stop at only one revenue center of a
    /// group (1830's two Canada offboards).
    #[pyo3(get)]
    #[serde(default)]
    pub groups: Vec<String>,
}

#[pymethods]
impl Offboard {
    #[new]
    pub fn new(revenue: i32) -> Self {
        Offboard {
            revenue,
            brown_revenue: None,
            green_revenue: None,
            gray_revenue: None,
            groups: Vec::new(),
        }
    }

}

impl Offboard {
    /// Get phase-appropriate revenue based on available tile colors.
    /// Ruby `RevenueCenter#revenue_for`: the revenue at the LATEST tile
    /// color available in the phase that this center defines (yellow is the
    /// base; an undefined tier falls through to the last defined one
    /// below it).
    pub fn phase_revenue(&self, phase_tiles: &[String]) -> i32 {
        let mut rev = self.revenue;
        for color in phase_tiles {
            let tier = match color.as_str() {
                "green" => self.green_revenue,
                "brown" => self.brown_revenue,
                "gray" => self.gray_revenue,
                _ => None,
            };
            if let Some(v) = tier {
                rev = v;
            }
        }
        rev
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

    /// Tile color as string: "white", "yellow", "green", "brown", "gray", "red".
    #[getter]
    fn color(&self) -> String {
        format!("{:?}", self.color).to_lowercase()
    }

    /// Tile label (e.g., "B", "NY", "OO") or None.
    #[getter]
    fn label(&self) -> Option<String> {
        self.label.clone()
    }

    /// Path definitions as list of (endpoint_a, endpoint_b, terminal) tuples.
    /// Each endpoint is a string like "Edge(0)", "City(0)", "Town(1)", "Junction".
    #[getter]
    fn path_defs(&self) -> Vec<(String, String, bool)> {
        self.paths
            .iter()
            .map(|p| {
                let a = format!("{:?}", p.a);
                let b = format!("{:?}", p.b);
                (a, b, p.terminal)
            })
            .collect()
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
