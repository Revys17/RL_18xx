use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

use crate::title::{g1830, MarketMovement, MarketZone};

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
        SharePrice {
            price,
            row,
            column,
            types,
        }
    }

    /// Primary type string (e.g., "par", "multiple_buy"). Matches Python's .type attribute.
    #[getter(r#type)]
    fn share_type(&self) -> Option<String> {
        self.types.first().cloned()
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
        Phase {
            name,
            operating_rounds,
            train_limit,
            tiles,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "Phase(name='{}', or={}, tl={})",
            self.name, self.operating_rounds, self.train_limit
        )
    }
}

// ---------------------------------------------------------------------------
// Stock Market
// ---------------------------------------------------------------------------

/// A cell in the stock market grid.
#[derive(Clone, Debug)]
pub struct MarketCell {
    pub price: i32,
    pub row: u8,
    pub column: u8,
    pub zone: String,
}

/// The stock market grid. Handles price lookups and share price movements.
#[derive(Clone, Debug)]
pub struct StockMarket {
    /// Grid[row][col] = Option<MarketCell>. None = empty/invalid cell.
    pub grid: Vec<Vec<Option<MarketCell>>>,
    /// The title's movement policy (2-D fallback moves vs a 1-D row).
    pub movement: MarketMovement,
}

impl StockMarket {
    /// Build from a title's raw market grid + movement policy.
    pub fn new(raw: Vec<Vec<Option<crate::title::MarketCell>>>, movement: MarketMovement) -> Self {
        let grid: Vec<Vec<Option<MarketCell>>> = raw
            .iter()
            .enumerate()
            .map(|(row_idx, row)| {
                row.iter()
                    .enumerate()
                    .map(|(col_idx, cell)| {
                        cell.as_ref().map(|c| MarketCell {
                            price: c.price,
                            row: row_idx as u8,
                            column: col_idx as u8,
                            zone: match c.zone {
                                MarketZone::Normal => "normal".to_string(),
                                MarketZone::Par => "par".to_string(),
                                MarketZone::Yellow => "no_cert_limit".to_string(),
                                MarketZone::Orange => "unlimited".to_string(),
                                MarketZone::Brown => "multiple_buy".to_string(),
                            },
                        })
                    })
                    .collect()
            })
            .collect();
        StockMarket { grid, movement }
    }

    /// The 1830 market (kept for tests; production goes through the title's
    /// `market_grid()`/`market_movement()`).
    pub fn new_1830() -> Self {
        Self::new(g1830::market_grid(), MarketMovement::TwoDimensional)
    }

    /// Find the cell at a given row/column.
    pub fn cell_at(&self, row: u8, col: u8) -> Option<&MarketCell> {
        self.grid
            .get(row as usize)
            .and_then(|r| r.get(col as usize))
            .and_then(|c| c.as_ref())
    }

    /// Find a SharePrice from a row/column.
    pub fn share_price_at(&self, row: u8, col: u8) -> Option<SharePrice> {
        self.cell_at(row, col).map(|c| SharePrice {
            price: c.price,
            row: c.row,
            column: c.column,
            types: vec![c.zone.clone()],
        })
    }

    /// Find the par price SharePrice for a given target price.
    /// Returns the cell in the par column with matching price.
    pub fn par_price(&self, price: i32) -> Option<SharePrice> {
        // Par prices are in column 6 (the "Par" zone column in 1830)
        for row in &self.grid {
            for cell in row.iter().flatten() {
                if cell.price == price && cell.zone == "par" {
                    return Some(SharePrice {
                        price: cell.price,
                        row: cell.row,
                        column: cell.column,
                        types: vec![cell.zone.clone()],
                    });
                }
            }
        }
        None
    }

    /// Get all valid par prices.
    pub fn par_prices(&self) -> Vec<i32> {
        let mut prices = Vec::new();
        for row in &self.grid {
            for cell in row.iter().flatten() {
                if cell.zone == "par" {
                    prices.push(cell.price);
                }
            }
        }
        prices.sort_unstable();
        prices
    }

    /// Move share price right (price increase). Returns new position.
    /// Used when: corporation's shares are sold out, or payout dividend.
    /// TwoDimensional (1830): at the right edge, fall back to move_up.
    /// OneDimensional (1867): stay at the right edge.
    pub fn move_right(&self, row: u8, col: u8) -> (u8, u8) {
        let new_col = col + 1;
        if self.cell_at(row, new_col).is_some() {
            (row, new_col)
        } else {
            match self.movement {
                MarketMovement::TwoDimensional => self.move_up(row, col),
                MarketMovement::OneDimensional => (row, col),
            }
        }
    }

    /// Move share price left (price decrease). Returns new position.
    /// Used when: shares are sold to the market, or corporation withholds.
    /// TwoDimensional (1830): at the left edge, fall back to move_down.
    /// OneDimensional (1867): stay at the left edge.
    pub fn move_left(&self, row: u8, col: u8) -> (u8, u8) {
        let fallback = |s: &Self| match s.movement {
            MarketMovement::TwoDimensional => s.move_down(row, col),
            MarketMovement::OneDimensional => (row, col),
        };
        if col == 0 {
            return fallback(self);
        }
        let new_col = col - 1;
        if self.cell_at(row, new_col).is_some() {
            (row, new_col)
        } else {
            fallback(self)
        }
    }

    /// Move share price down (price decrease). Returns new position.
    /// Used when: corporation withholds revenue. A 1-D market has no
    /// vertical movement.
    pub fn move_down(&self, row: u8, col: u8) -> (u8, u8) {
        if self.movement == MarketMovement::OneDimensional {
            return (row, col);
        }
        let new_row = row + 1;
        if self.cell_at(new_row, col).is_some() {
            (new_row, col)
        } else {
            (row, col) // Stay at bottom
        }
    }

    /// Move share price up (price increase). A 1-D market has no vertical
    /// movement.
    pub fn move_up(&self, row: u8, col: u8) -> (u8, u8) {
        if row == 0 || self.movement == MarketMovement::OneDimensional {
            return (row, col);
        }
        let new_row = row - 1;
        if self.cell_at(new_row, col).is_some() {
            (new_row, col)
        } else {
            (row, col) // Stay at top
        }
    }
}
