//! Action types for the 1830 game engine.
//!
//! Each variant represents a player or entity decision.
//! Actions are parsed from Python dicts and dispatched to round processors.

use std::collections::HashMap;
use std::fmt;

use pyo3::prelude::*;
use pyo3::types::PyDict;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct GameError {
    pub message: String,
}

impl GameError {
    pub fn new(msg: impl Into<String>) -> Self {
        GameError {
            message: msg.into(),
        }
    }
}

impl fmt::Display for GameError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GameError: {}", self.message)
    }
}

impl std::error::Error for GameError {}

impl From<GameError> for PyErr {
    fn from(err: GameError) -> PyErr {
        pyo3::exceptions::PyRuntimeError::new_err(err.message)
    }
}

// ---------------------------------------------------------------------------
// Route data (for RunRoutes action)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RouteData {
    pub train_name: String,
    pub hexes: Vec<String>,
    pub revenue: i32,
}

// ---------------------------------------------------------------------------
// Dividend kind
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum DividendKind {
    Payout,
    Withhold,
}

impl DividendKind {
    pub fn parse(s: &str) -> Result<Self, GameError> {
        match s {
            "payout" => Ok(DividendKind::Payout),
            "withhold" => Ok(DividendKind::Withhold),
            _ => Err(GameError::new(format!("Unknown dividend kind: {}", s))),
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            DividendKind::Payout => "payout",
            DividendKind::Withhold => "withhold",
        }
    }
}

// ---------------------------------------------------------------------------
// Share reference (identifies shares in buy/sell actions)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct ShareRef {
    pub corporation_sym: String,
    pub percent: u8,
    pub president: bool,
    /// Where the share comes from/goes to: "ipo", "market", "player:<id>"
    pub source: String,
}

// ---------------------------------------------------------------------------
// Action enum
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub enum Action {
    Pass {
        entity_id: String,
    },
    Bid {
        entity_id: String,
        company_sym: String,
        price: i32,
    },
    Par {
        entity_id: String,
        corporation_sym: String,
        share_price: i32,
    },
    BuyShares {
        entity_id: String,
        corporation_sym: String,
        shares: Vec<ShareRef>,
        percent: u8,
        /// "ipo" or "market"
        source: String,
        /// Specific share indices from action dict (e.g., "NYC_3" → [3])
        share_indices: Vec<usize>,
    },
    SellShares {
        entity_id: String,
        corporation_sym: String,
        shares: Vec<ShareRef>,
        percent: u8,
        /// Specific share indices from action dict (e.g., "CPR_3" → [3])
        share_indices: Vec<usize>,
    },
    LayTile {
        entity_id: String,
        hex_id: String,
        tile_id: String,
        rotation: u8,
    },
    PlaceToken {
        entity_id: String,
        hex_id: String,
        city_index: u8,
    },
    RunRoutes {
        entity_id: String,
        routes: Vec<RouteData>,
        extra_revenue: i32,
    },
    Dividend {
        entity_id: String,
        kind: DividendKind,
    },
    BuyTrain {
        entity_id: String,
        train_name: String,
        price: i32,
        /// "depot" or a corporation sym
        from: String,
        variant: Option<String>,
        /// Train being exchanged (e.g., "4-0" when upgrading 4→D for discount)
        exchange: Option<String>,
    },
    DiscardTrain {
        entity_id: String,
        train_name: String,
    },
    BuyCompany {
        entity_id: String,
        company_sym: String,
        price: i32,
    },
    Bankrupt {
        entity_id: String,
    },
}

impl Action {
    /// Returns the entity_id for any action variant.
    pub fn entity_id(&self) -> &str {
        match self {
            Action::Pass { entity_id } => entity_id,
            Action::Bid { entity_id, .. } => entity_id,
            Action::Par { entity_id, .. } => entity_id,
            Action::BuyShares { entity_id, .. } => entity_id,
            Action::SellShares { entity_id, .. } => entity_id,
            Action::LayTile { entity_id, .. } => entity_id,
            Action::PlaceToken { entity_id, .. } => entity_id,
            Action::RunRoutes { entity_id, .. } => entity_id,
            Action::Dividend { entity_id, .. } => entity_id,
            Action::BuyTrain { entity_id, .. } => entity_id,
            Action::DiscardTrain { entity_id, .. } => entity_id,
            Action::BuyCompany { entity_id, .. } => entity_id,
            Action::Bankrupt { entity_id } => entity_id,
        }
    }

    /// Returns the action type as a string (matches Python action type names).
    pub fn action_type(&self) -> &'static str {
        match self {
            Action::Pass { .. } => "pass",
            Action::Bid { .. } => "bid",
            Action::Par { .. } => "par",
            Action::BuyShares { .. } => "buy_shares",
            Action::SellShares { .. } => "sell_shares",
            Action::LayTile { .. } => "lay_tile",
            Action::PlaceToken { .. } => "place_token",
            Action::RunRoutes { .. } => "run_routes",
            Action::Dividend { .. } => "dividend",
            Action::BuyTrain { .. } => "buy_train",
            Action::DiscardTrain { .. } => "discard_train",
            Action::BuyCompany { .. } => "buy_company",
            Action::Bankrupt { .. } => "bankrupt",
        }
    }

    /// Parse an Action from a Python dict.
    ///
    /// Expected keys vary by action type:
    /// - "type": action type string (required)
    /// - "entity": entity id (player id as int, or corp sym as string)
    /// - Other keys depend on the action type
    pub fn from_py_dict(dict: &Bound<'_, PyDict>) -> Result<Action, GameError> {
        let action_type: String = dict
            .get_item("type")
            .map_err(|e| GameError::new(format!("Missing 'type' key: {}", e)))?
            .ok_or_else(|| GameError::new("Missing 'type' key"))?
            .extract()
            .map_err(|e| GameError::new(format!("'type' must be a string: {}", e)))?;

        let entity_id = extract_entity_id(dict)?;

        match action_type.as_str() {
            "pass" => Ok(Action::Pass { entity_id }),

            "bid" => {
                let company_sym = extract_string(dict, "company")?;
                let price = extract_i32(dict, "price")?;
                Ok(Action::Bid {
                    entity_id,
                    company_sym,
                    price,
                })
            }

            "par" => {
                let corporation_sym = extract_string(dict, "corporation")?;
                // share_price can be int or "price,row,col" string
                let share_price = extract_share_price(dict)?;
                Ok(Action::Par {
                    entity_id,
                    corporation_sym,
                    share_price,
                })
            }

            "buy_shares" => {
                // shares field: ["PRR_2", "PRR_3"] — corp sym is prefix before '_'
                // OR corporation field may be present directly
                let (corporation_sym, share_ids) = extract_shares_info(dict)?;
                let percent = extract_optional_i32(dict, "percent")?.unwrap_or(10) as u8;
                let source =
                    extract_optional_string(dict, "source")?.unwrap_or_else(|| "auto".to_string());

                // Extract specific share indices from share IDs (e.g., "NYC_3" → 3)
                let share_indices: Vec<usize> = share_ids
                    .iter()
                    .filter_map(|id| id.rsplit('_').next()?.parse::<usize>().ok())
                    .collect();

                Ok(Action::BuyShares {
                    entity_id,
                    corporation_sym,
                    shares: Vec::new(),
                    percent,
                    source,
                    share_indices,
                })
            }

            "sell_shares" => {
                let (corporation_sym, share_ids) = extract_shares_info(dict)?;
                let percent = extract_optional_i32(dict, "percent")?.unwrap_or(10) as u8;

                let share_indices: Vec<usize> = share_ids
                    .iter()
                    .filter_map(|id| id.rsplit('_').next()?.parse::<usize>().ok())
                    .collect();

                Ok(Action::SellShares {
                    entity_id,
                    corporation_sym,
                    shares: Vec::new(),
                    percent,
                    share_indices,
                })
            }

            "lay_tile" => {
                let hex_id = extract_string(dict, "hex")?;
                let tile_id = extract_string(dict, "tile")?;
                let rotation = extract_optional_i32(dict, "rotation")?.unwrap_or(0) as u8;
                Ok(Action::LayTile {
                    entity_id,
                    hex_id,
                    tile_id,
                    rotation,
                })
            }

            "place_token" => {
                // Action may have 'hex' key directly, or 'city' key
                // City format: "tile_base-tile_instance-city_index" e.g. "57-0-0"
                let (hex_id, city_index) = if let Ok(hex) = extract_string(dict, "hex") {
                    let ci = extract_optional_i32(dict, "city_index")?.unwrap_or(0) as u8;
                    (hex, ci)
                } else {
                    let city_str = extract_string(dict, "city")?;
                    let parts: Vec<&str> = city_str.split('-').collect();
                    if parts.len() >= 3 {
                        // "57-0-0" → tile instance "57-0", city index 0
                        let tile_instance = format!("{}-{}", parts[0], parts[1]);
                        let ci = parts[2].parse::<u8>().unwrap_or(0);
                        (format!("__tile:{}", tile_instance), ci)
                    } else if parts.len() == 2 {
                        // "57-0" → tile name "57", city index 0
                        let ci = parts[1].parse::<u8>().unwrap_or(0);
                        (format!("__tile:{}", parts[0]), ci)
                    } else {
                        (format!("__tile:{}", city_str), 0)
                    }
                };
                Ok(Action::PlaceToken {
                    entity_id,
                    hex_id,
                    city_index,
                })
            }

            "run_routes" => {
                let extra_revenue = extract_optional_i32(dict, "extra_revenue")?.unwrap_or(0);
                let routes = extract_routes(dict)?;
                Ok(Action::RunRoutes {
                    entity_id,
                    routes,
                    extra_revenue,
                })
            }

            "dividend" => {
                let kind_str = extract_string(dict, "kind")?;
                let kind = DividendKind::parse(&kind_str)?;
                Ok(Action::Dividend { entity_id, kind })
            }

            "buy_train" => {
                let train_name = extract_string(dict, "train")?;
                let price = extract_i32(dict, "price")?;
                let from =
                    extract_optional_string(dict, "from")?.unwrap_or_else(|| "depot".to_string());
                let variant = extract_optional_string(dict, "variant")?;
                let exchange = extract_optional_string(dict, "exchange")?;
                Ok(Action::BuyTrain {
                    entity_id,
                    train_name,
                    price,
                    from,
                    variant,
                    exchange,
                })
            }

            "discard_train" => {
                let train_name = extract_string(dict, "train")?;
                Ok(Action::DiscardTrain {
                    entity_id,
                    train_name,
                })
            }

            "buy_company" => {
                let company_sym = extract_string(dict, "company")?;
                let price = extract_i32(dict, "price")?;
                Ok(Action::BuyCompany {
                    entity_id,
                    company_sym,
                    price,
                })
            }

            "bankrupt" => Ok(Action::Bankrupt { entity_id }),

            _ => Err(GameError::new(format!(
                "Unknown action type: {}",
                action_type
            ))),
        }
    }

    /// Convert this action to a HashMap suitable for Python consumption.
    pub fn to_map(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert("type".to_string(), self.action_type().to_string());
        map.insert("entity".to_string(), self.entity_id().to_string());

        match self {
            Action::Pass { .. } => {}
            Action::Bid {
                company_sym, price, ..
            } => {
                map.insert("company".to_string(), company_sym.clone());
                map.insert("price".to_string(), price.to_string());
            }
            Action::Par {
                corporation_sym,
                share_price,
                ..
            } => {
                map.insert("corporation".to_string(), corporation_sym.clone());
                map.insert("share_price".to_string(), share_price.to_string());
            }
            Action::BuyShares {
                corporation_sym,
                percent,
                source,
                ..
            } => {
                map.insert("corporation".to_string(), corporation_sym.clone());
                map.insert("percent".to_string(), percent.to_string());
                map.insert("source".to_string(), source.clone());
            }
            Action::SellShares {
                corporation_sym,
                percent,
                ..
            } => {
                map.insert("corporation".to_string(), corporation_sym.clone());
                map.insert("percent".to_string(), percent.to_string());
            }
            Action::LayTile {
                hex_id,
                tile_id,
                rotation,
                ..
            } => {
                map.insert("hex".to_string(), hex_id.clone());
                map.insert("tile".to_string(), tile_id.clone());
                map.insert("rotation".to_string(), rotation.to_string());
            }
            Action::PlaceToken {
                hex_id, city_index, ..
            } => {
                map.insert("hex".to_string(), hex_id.clone());
                map.insert("city_index".to_string(), city_index.to_string());
            }
            Action::RunRoutes {
                extra_revenue,
                routes,
                ..
            } => {
                map.insert("extra_revenue".to_string(), extra_revenue.to_string());
                map.insert("num_routes".to_string(), routes.len().to_string());
            }
            Action::Dividend { kind, .. } => {
                map.insert("kind".to_string(), kind.as_str().to_string());
            }
            Action::BuyTrain {
                train_name,
                price,
                from,
                variant,
                ..
            } => {
                map.insert("train".to_string(), train_name.clone());
                map.insert("price".to_string(), price.to_string());
                map.insert("from".to_string(), from.clone());
                if let Some(v) = variant {
                    map.insert("variant".to_string(), v.clone());
                }
            }
            Action::DiscardTrain { train_name, .. } => {
                map.insert("train".to_string(), train_name.clone());
            }
            Action::BuyCompany {
                company_sym, price, ..
            } => {
                map.insert("company".to_string(), company_sym.clone());
                map.insert("price".to_string(), price.to_string());
            }
            Action::Bankrupt { .. } => {}
        }

        map
    }
}

// ---------------------------------------------------------------------------
// Dict extraction helpers
// ---------------------------------------------------------------------------

fn extract_entity_id(dict: &Bound<'_, PyDict>) -> Result<String, GameError> {
    let item = dict
        .get_item("entity")
        .map_err(|e| GameError::new(format!("Error reading 'entity': {}", e)))?
        .ok_or_else(|| GameError::new("Missing 'entity' key"))?;

    // Entity can be an int (player id) or string (corp sym)
    if let Ok(id) = item.extract::<u32>() {
        Ok(id.to_string())
    } else if let Ok(s) = item.extract::<String>() {
        Ok(s)
    } else {
        Err(GameError::new("'entity' must be int or string"))
    }
}

fn extract_string(dict: &Bound<'_, PyDict>, key: &str) -> Result<String, GameError> {
    dict.get_item(key)
        .map_err(|e| GameError::new(format!("Error reading '{}': {}", key, e)))?
        .ok_or_else(|| GameError::new(format!("Missing '{}' key", key)))?
        .extract::<String>()
        .map_err(|e| GameError::new(format!("'{}' must be a string: {}", key, e)))
}

fn extract_i32(dict: &Bound<'_, PyDict>, key: &str) -> Result<i32, GameError> {
    dict.get_item(key)
        .map_err(|e| GameError::new(format!("Error reading '{}': {}", key, e)))?
        .ok_or_else(|| GameError::new(format!("Missing '{}' key", key)))?
        .extract::<i32>()
        .map_err(|e| GameError::new(format!("'{}' must be an integer: {}", key, e)))
}

fn extract_optional_string(
    dict: &Bound<'_, PyDict>,
    key: &str,
) -> Result<Option<String>, GameError> {
    match dict.get_item(key) {
        Ok(Some(item)) => {
            if item.is_none() {
                Ok(None)
            } else {
                Ok(Some(item.extract::<String>().map_err(|e| {
                    GameError::new(format!("'{}' must be a string: {}", key, e))
                })?))
            }
        }
        Ok(None) => Ok(None),
        Err(e) => Err(GameError::new(format!("Error reading '{}': {}", key, e))),
    }
}

fn extract_optional_i32(dict: &Bound<'_, PyDict>, key: &str) -> Result<Option<i32>, GameError> {
    match dict.get_item(key) {
        Ok(Some(item)) => {
            if item.is_none() {
                Ok(None)
            } else {
                Ok(Some(item.extract::<i32>().map_err(|e| {
                    GameError::new(format!("'{}' must be an integer: {}", key, e))
                })?))
            }
        }
        Ok(None) => Ok(None),
        Err(e) => Err(GameError::new(format!("Error reading '{}': {}", key, e))),
    }
}

fn extract_routes(dict: &Bound<'_, PyDict>) -> Result<Vec<RouteData>, GameError> {
    let routes_item = match dict.get_item("routes") {
        Ok(Some(item)) => item,
        Ok(None) => return Ok(Vec::new()),
        Err(e) => return Err(GameError::new(format!("Error reading 'routes': {}", e))),
    };

    if routes_item.is_none() {
        return Ok(Vec::new());
    }

    let routes_list: Vec<Bound<'_, PyDict>> = routes_item
        .extract()
        .map_err(|e| GameError::new(format!("'routes' must be a list of dicts: {}", e)))?;

    let mut result = Vec::with_capacity(routes_list.len());
    for route_dict in &routes_list {
        let train_name = extract_string(route_dict, "train")?;
        let revenue = extract_optional_i32(route_dict, "revenue")?.unwrap_or(0);

        let hexes: Vec<String> = match route_dict.get_item("hexes") {
            Ok(Some(item)) => item.extract().unwrap_or_default(),
            _ => Vec::new(),
        };

        result.push(RouteData {
            train_name,
            hexes,
            revenue,
        });
    }

    Ok(result)
}

/// Parse share_price which can be:
/// - An integer (direct price)
/// - A string like "100,0,6" (price,row,col) — extract just the price
fn extract_share_price(dict: &Bound<'_, PyDict>) -> Result<i32, GameError> {
    let item = dict
        .get_item("share_price")
        .map_err(|e| GameError::new(format!("Error reading 'share_price': {}", e)))?
        .ok_or_else(|| GameError::new("Missing 'share_price' key"))?;

    // Try integer first
    if let Ok(price) = item.extract::<i32>() {
        return Ok(price);
    }

    // Try string "price,row,col"
    if let Ok(s) = item.extract::<String>() {
        let parts: Vec<&str> = s.split(',').collect();
        if let Some(price_str) = parts.first() {
            return price_str
                .parse::<i32>()
                .map_err(|_| GameError::new(format!("Invalid share_price format: {}", s)));
        }
    }

    Err(GameError::new(
        "'share_price' must be int or 'price,row,col' string",
    ))
}

/// Extract corporation sym and share IDs from a buy_shares/sell_shares dict.
/// The dict may have "shares": ["PRR_2", "PRR_3"] and/or "corporation": "PRR".
fn extract_shares_info(dict: &Bound<'_, PyDict>) -> Result<(String, Vec<String>), GameError> {
    // Try "corporation" field first
    if let Ok(Some(corp_item)) = dict.get_item("corporation") {
        if let Ok(corp) = corp_item.extract::<String>() {
            let share_ids = extract_share_ids(dict);
            return Ok((corp, share_ids));
        }
    }

    // Derive corporation from share IDs: "PRR_2" → "PRR"
    let share_ids = extract_share_ids(dict);
    if let Some(first_id) = share_ids.first() {
        if let Some(corp) = first_id.rsplit('_').nth(1) {
            // Handle multi-part names like "B&O_2" — everything before last '_'
            let corp = if let Some(pos) = first_id.rfind('_') {
                &first_id[..pos]
            } else {
                corp
            };
            return Ok((corp.to_string(), share_ids));
        }
    }

    Err(GameError::new(
        "Cannot determine corporation from shares or corporation field",
    ))
}

fn extract_share_ids(dict: &Bound<'_, PyDict>) -> Vec<String> {
    match dict.get_item("shares") {
        Ok(Some(item)) => item.extract::<Vec<String>>().unwrap_or_default(),
        _ => Vec::new(),
    }
}
