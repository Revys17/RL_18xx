use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

use crate::core::SharePrice;

/// Identifies who owns something: a player, corporation, or the bank/market.
/// Stored as a simple string for PyO3 compatibility: "player:1", "corp:PRR", "bank", "market", "".
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct EntityId(pub String);

impl EntityId {
    pub fn player(id: u32) -> Self {
        EntityId(format!("player:{}", id))
    }
    pub fn corporation(sym: &str) -> Self {
        EntityId(format!("corp:{}", sym))
    }
    pub fn ipo(sym: &str) -> Self {
        EntityId(format!("ipo:{}", sym))
    }
    pub fn bank() -> Self {
        EntityId("bank".to_string())
    }
    pub fn market() -> Self {
        EntityId("market".to_string())
    }
    pub fn none() -> Self {
        EntityId(String::new())
    }
    pub fn is_none(&self) -> bool {
        self.0.is_empty()
    }

    pub fn is_player(&self) -> bool {
        self.0.starts_with("player:")
    }
    pub fn is_corporation(&self) -> bool {
        self.0.starts_with("corp:")
    }
    pub fn is_ipo(&self) -> bool {
        self.0.starts_with("ipo:")
    }
    pub fn is_market(&self) -> bool {
        self.0 == "market"
    }

    /// Extract player id from "player:<id>".
    pub fn player_id(&self) -> Option<u32> {
        self.0.strip_prefix("player:").and_then(|s| s.parse().ok())
    }

    /// Extract corporation sym from "corp:<sym>" or "ipo:<sym>".
    pub fn corp_sym(&self) -> Option<&str> {
        self.0
            .strip_prefix("corp:")
            .or_else(|| self.0.strip_prefix("ipo:"))
    }
}

/// A train card.
#[pyclass]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Train {
    #[pyo3(get)]
    pub name: String,
    /// Unique train ID including instance suffix (e.g., "2-0", "3-1").
    #[pyo3(get)]
    pub id: String,
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
        let id = name.clone(); // Will be overridden with instance ID during init
        Train {
            name,
            id,
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

impl std::fmt::Display for EntityId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
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

    #[getter]
    fn owner(&self) -> String {
        self.owner.0.clone()
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
    /// If true, this company cannot be purchased by a corporation during OR.
    pub no_buy: bool,
    /// True once the company's special ability has been used (e.g., CS tile_lay).
    pub ability_used: bool,
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
            no_buy: false,
            ability_used: false,
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
    /// True once the home token has been placed at least once (even if later
    /// removed by tile upgrade). Used to determine if the home city reservation
    /// has been consumed.
    pub home_token_ever_placed: bool,
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
            home_token_ever_placed: false,
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
        format!(
            "Player(id={}, name='{}', cash={})",
            self.id, self.name, self.cash
        )
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
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        Depot {
            trains: Vec::new(),
            discarded: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Non-PyO3 helper methods for game logic
// ---------------------------------------------------------------------------

impl Corporation {
    /// Calculate the percent of shares owned by a given entity.
    pub fn percent_owned_by(&self, owner: &EntityId) -> u8 {
        self.shares
            .iter()
            .filter(|s| s.owner == *owner)
            .map(|s| s.percent)
            .sum()
    }

    /// Count shares in the IPO (owner = ipo:<sym>).
    pub fn ipo_shares_percent(&self) -> u8 {
        let ipo_id = EntityId::ipo(&self.sym);
        self.percent_owned_by(&ipo_id)
    }

    /// Count shares in the market pool.
    pub fn market_shares_percent(&self) -> u8 {
        self.percent_owned_by(&EntityId::market())
    }

    /// Check if the corporation has floated (60%+ sold from IPO).
    /// A corp floats when players own >= 60% of its shares.
    pub fn check_floated(&self) -> bool {
        let ipo_percent = self.ipo_shares_percent();
        // 100% total - IPO remaining = sold percent. Float at 60%+.
        (100 - ipo_percent) >= 60
    }

    /// Find the president share.
    pub fn president_share_index(&self) -> Option<usize> {
        self.shares.iter().position(|s| s.president)
    }

    /// Find the player who owns the president share.
    pub fn president_id(&self) -> Option<u32> {
        self.shares
            .iter()
            .find(|s| s.president)
            .and_then(|s| s.owner.player_id())
    }

    /// Get the next available (unplaced) token.
    pub fn next_token_index(&self) -> Option<usize> {
        self.tokens.iter().position(|t| !t.used)
    }

    /// Get the price of the next token to place.
    pub fn next_token_price(&self) -> Option<i32> {
        self.next_token_index().map(|i| self.tokens[i].price)
    }
}

impl Player {
    /// Count the total number of certificates this player holds.
    /// Includes shares (each share = 1 cert, president = 1 cert) and companies.
    pub fn count_certs(&self, corps: &[Corporation], companies: &[Company]) -> u32 {
        let player_eid = EntityId::player(self.id);
        let share_certs: u32 = corps
            .iter()
            .flat_map(|c| c.shares.iter())
            .filter(|s| s.owner == player_eid)
            .count() as u32;
        let company_certs: u32 = companies
            .iter()
            .filter(|c| c.owner == player_eid && !c.closed)
            .count() as u32;
        share_certs + company_certs
    }

    /// Calculate this player's total value: cash + share values (at market price).
    /// Does not include company face values — use company_value() separately.
    pub fn value(&self, corps: &[Corporation]) -> i32 {
        let player_eid = EntityId::player(self.id);
        let share_value: i32 = corps
            .iter()
            .filter_map(|c| {
                c.ipo_price.as_ref()?;
                let percent = c.percent_owned_by(&player_eid);
                if percent > 0 {
                    c.share_price
                        .as_ref()
                        .map(|sp| (percent as i32 * sp.price) / 10)
                } else {
                    None
                }
            })
            .sum();
        self.cash + share_value
    }

    /// Face value of private companies owned by this player.
    pub fn company_value(&self, companies: &[Company]) -> i32 {
        let player_eid = EntityId::player(self.id);
        companies
            .iter()
            .filter(|c| c.owner == player_eid && !c.closed)
            .map(|c| c.value)
            .sum()
    }
}
