//! Factored legal-action enumeration mirroring `rl18xx/game/factored_action_helper.py`.
//!
//! Returns categorical-only legal actions plus a `price_range` field for
//! price-bearing types (Bid, BuyCompany, BuyTrain). The factored output is the
//! canonical representation that the new AlphaZero policy head, MCTS, and
//! pretraining consume.
//!
//! The Python reference (`FactoredActionHelper`) is the source of truth; this
//! file is a clean translation of its enumeration logic, leveraging the Rust
//! engine's existing query methods (`buyable_shares`, `sellable_bundles`,
//! `auction_min_bid`, ...).

use std::collections::HashMap;

use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde_json::json;

use crate::entities::EntityId;
use crate::game::BaseGame;
use crate::rounds::Round;

/// One legal action in factored form. Mirrors the Python dataclass at
/// `rl18xx/game/factored_action_helper.py::LegalAction`.
#[derive(Debug, Clone)]
pub struct LegalAction {
    pub action_type: String,
    pub entity: HashMap<String, serde_json::Value>,
    pub params: HashMap<String, serde_json::Value>,
    pub price_range: Option<(i64, i64)>,
}

impl LegalAction {
    fn new(action_type: &str) -> Self {
        LegalAction {
            action_type: action_type.to_string(),
            entity: HashMap::new(),
            params: HashMap::new(),
            price_range: None,
        }
    }

    /// Convert to a Python dict.
    pub fn to_py_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("type", &self.action_type)?;
        let ent = PyDict::new(py);
        for (k, v) in &self.entity {
            ent.set_item(k, json_to_py(py, v)?)?;
        }
        dict.set_item("entity", ent)?;
        let par = PyDict::new(py);
        for (k, v) in &self.params {
            par.set_item(k, json_to_py(py, v)?)?;
        }
        dict.set_item("params", par)?;
        match self.price_range {
            Some((lo, hi)) => dict.set_item("price_range", (lo, hi))?,
            None => dict.set_item("price_range", py.None())?,
        }
        Ok(dict.into())
    }
}

fn json_to_py(py: Python<'_>, v: &serde_json::Value) -> PyResult<PyObject> {
    use pyo3::IntoPyObject;
    Ok(match v {
        serde_json::Value::Null => py.None(),
        serde_json::Value::Bool(b) => (*b).into_pyobject(py)?.to_owned().into(),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                i.into_pyobject(py)?.into()
            } else if let Some(u) = n.as_u64() {
                u.into_pyobject(py)?.into()
            } else if let Some(f) = n.as_f64() {
                f.into_pyobject(py)?.into_any().into()
            } else {
                py.None()
            }
        }
        serde_json::Value::String(s) => s.as_str().into_pyobject(py)?.into_any().into(),
        serde_json::Value::Array(arr) => {
            let list: Vec<PyObject> = arr
                .iter()
                .map(|x| json_to_py(py, x))
                .collect::<PyResult<Vec<_>>>()?;
            list.into_pyobject(py)?.into_any().into()
        }
        serde_json::Value::Object(obj) => {
            let d = PyDict::new(py);
            for (k, val) in obj {
                d.set_item(k, json_to_py(py, val)?)?;
            }
            d.into_any().into()
        }
    })
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

/// Player name lookup by ID.
fn player_name(game: &BaseGame, pid: u32) -> String {
    game.players
        .iter()
        .find(|p| p.id == pid)
        .map(|p| p.name.clone())
        .unwrap_or_else(|| format!("player:{}", pid))
}

/// Descriptor for an entity (player, corp, company).
fn entity_descriptor_player(game: &BaseGame, pid: u32) -> HashMap<String, serde_json::Value> {
    let mut m = HashMap::new();
    m.insert("player".to_string(), json!(player_name(game, pid)));
    m
}

fn entity_descriptor_corp(corp_sym: &str) -> HashMap<String, serde_json::Value> {
    let mut m = HashMap::new();
    m.insert("corp".to_string(), json!(corp_sym));
    m
}

#[allow(dead_code)]
fn entity_descriptor_company(company_sym: &str) -> HashMap<String, serde_json::Value> {
    let mut m = HashMap::new();
    m.insert("private".to_string(), json!(company_sym));
    m
}

/// Descriptor for the currently acting entity (player or corp).
fn current_entity_descriptor(game: &BaseGame) -> HashMap<String, serde_json::Value> {
    let eid = &game.round_state.active_entity_id;
    if eid.is_player() {
        if let Some(pid) = eid.player_id() {
            return entity_descriptor_player(game, pid);
        }
    }
    if eid.is_corporation() {
        if let Some(sym) = eid.corp_sym() {
            return entity_descriptor_corp(sym);
        }
    }
    HashMap::new()
}

// ---------------------------------------------------------------------------
// main entry point
// ---------------------------------------------------------------------------

impl BaseGame {
    /// Top-level factored enumeration.
    pub fn get_factored_choices_impl(&mut self) -> Vec<LegalAction> {
        if self.finished {
            return Vec::new();
        }

        let mut out: Vec<LegalAction> = Vec::new();

        // Get the legal action types for the current state. We use the
        // game's own typed enumeration as the master gate.
        let action_types = self.legal_action_types_internal();

        for at in &action_types {
            match at.as_str() {
                "pass" => out.extend(self.factored_pass()),
                "bid" => out.extend(self.factored_bid()),
                "par" => out.extend(self.factored_par()),
                "buy_shares" => out.extend(self.factored_buy_shares()),
                "sell_shares" => out.extend(self.factored_sell_shares()),
                "place_token" => out.extend(self.factored_place_token()),
                "lay_tile" => out.extend(self.factored_lay_tile()),
                "buy_train" => out.extend(self.factored_buy_train()),
                "discard_train" => out.extend(self.factored_discard_train()),
                "run_routes" => out.extend(self.factored_run_routes()),
                "dividend" => out.extend(self.factored_dividend()),
                "buy_company" => out.extend(self.factored_buy_company()),
                "bankrupt" => out.extend(self.factored_bankrupt()),
                _ => {}
            }
        }

        // Company actions (MH exchange branch). Surfaced even when the
        // current entity is a player/corp, mirroring Python's get_company_actions.
        out.extend(self.factored_company_actions());

        // Bankruptcy fallback: if no actions emitted at all and we're in
        // a state where the entity can go bankrupt, emit a Bankrupt option.
        // (legal_action_types_internal already includes "bankrupt" when the
        // engine thinks bankruptcy is required, so this is rare in practice.)
        out
    }

    /// Get legal action type strings for the current state.
    /// Wraps the existing `legal_action_types()` pymethod logic.
    fn legal_action_types_internal(&mut self) -> Vec<String> {
        // The PyO3-exposed legal_action_types is a `&mut self` method that
        // returns Vec<String>. We can call it directly.
        self.legal_action_types_for_factored()
    }

    fn factored_pass(&self) -> Vec<LegalAction> {
        let mut a = LegalAction::new("Pass");
        a.entity = current_entity_descriptor(self);
        vec![a]
    }

    fn factored_bid(&self) -> Vec<LegalAction> {
        let state = match &self.round {
            Round::Auction(s) => s.clone(),
            _ => return Vec::new(),
        };
        if state.pending_par.is_some() {
            return Vec::new();
        }

        let pid = state.active_player_id();
        let player_cash = self
            .players
            .iter()
            .find(|p| p.id == pid)
            .map_or(0, |p| p.cash);
        let entity_desc = entity_descriptor_player(self, pid);

        let biddable: Vec<usize> = if let Some(auc_ci) = state.auctioning {
            vec![auc_ci]
        } else {
            state.remaining_companies.clone()
        };

        let mut out = Vec::new();
        for ci in biddable {
            let company = match self.companies.get(ci) {
                Some(c) => c,
                None => continue,
            };
            let value = company.value;
            let min_bid = state.min_bid_for(ci, value);
            let mut max_bid = state.max_bid(pid, ci, player_cash);
            if max_bid < min_bid {
                continue;
            }
            // may_purchase: cheapest, no active auction → price fixed at min_bid.
            if state.auctioning.is_none() && state.may_purchase(ci) {
                max_bid = min_bid;
            }
            let mut a = LegalAction::new("Bid");
            a.entity = entity_desc.clone();
            a.entity
                .insert("private".to_string(), json!(company.sym.clone()));
            a.price_range = Some((min_bid as i64, max_bid as i64));
            out.push(a);
        }
        out
    }

    fn factored_par(&self) -> Vec<LegalAction> {
        // Two flavours:
        // 1. Pending par (company-triggered like BO → B&O): a single corp + range of par prices.
        // 2. Stock round par: any unpar'd corp × any par price affordable by buyer.
        let mut out: Vec<LegalAction> = Vec::new();
        let par_prices = self.par_prices_internal();

        // Pending par
        if let Round::Auction(s) = &self.round {
            if let Some((corp_sym, pid)) = s.pending_par.clone() {
                // Find the company that triggered this par (e.g., BO → B&O).
                let company_sym = self
                    .companies
                    .iter()
                    .find(|c| {
                        // Hard-coded for 1830: only BO triggers B&O par.
                        c.sym == "BO" && corp_sym == "B&O"
                    })
                    .map(|c| c.sym.clone());
                let entity_desc = entity_descriptor_player(self, pid);
                for price in &par_prices {
                    let mut a = LegalAction::new("Par");
                    a.entity = entity_desc.clone();
                    a.entity.insert("corp".to_string(), json!(corp_sym.clone()));
                    a.params.insert("par_price".to_string(), json!(*price));
                    if let Some(cs) = &company_sym {
                        a.params.insert("private".to_string(), json!(cs.clone()));
                    }
                    out.push(a);
                }
                return out;
            }
        }

        // Stock round par
        let player_id = match self.round_state.active_entity_id.player_id() {
            Some(pid) => pid,
            None => return out,
        };
        let buying_power = self
            .players
            .iter()
            .find(|p| p.id == player_id)
            .map_or(0, |p| p.cash);

        // Sorted list of parable corp syms.
        let mut parable: Vec<&str> = self
            .corporations
            .iter()
            .filter(|c| c.ipo_price.is_none())
            .map(|c| c.name.as_str())
            .collect();
        parable.sort();
        // map name → sym
        let entity_desc = entity_descriptor_player(self, player_id);
        for name in parable {
            if let Some(corp) = self.corporations.iter().find(|c| c.name == name) {
                for price in &par_prices {
                    if 2 * price > buying_power {
                        continue;
                    }
                    let mut a = LegalAction::new("Par");
                    a.entity = entity_desc.clone();
                    a.entity
                        .insert("corp".to_string(), json!(corp.sym.clone()));
                    a.params.insert("par_price".to_string(), json!(*price));
                    out.push(a);
                }
            }
        }
        out
    }

    fn par_prices_internal(&self) -> Vec<i32> {
        self.stock_market.par_prices()
    }

    fn factored_buy_shares(&self) -> Vec<LegalAction> {
        let player_id = match self.round_state.active_entity_id.player_id() {
            Some(pid) => pid,
            None => return Vec::new(),
        };
        let tuples = self.buyable_shares_internal(player_id);
        let mut out: Vec<LegalAction> = Vec::new();
        let entity_desc = entity_descriptor_player(self, player_id);

        // Deduplicate by (corp_sym, source) — buyable_shares can have multiple
        // entries per group but we emit one per categorical key.
        let mut seen: std::collections::HashSet<(String, String)> = std::collections::HashSet::new();
        // Sort for determinism: corp_sym ASC, then source (ipo before market).
        let mut sorted: Vec<(String, String, usize, i32)> = tuples;
        sorted.sort_by(|a, b| {
            a.0.cmp(&b.0).then_with(|| {
                // ipo < market matches the Python sort (which uses
                // owner.__class__.__name__: Corporation < SharePool/Market alphabetically).
                a.1.cmp(&b.1)
            })
        });
        for (corp_sym, source, _idx, _price) in sorted {
            if !seen.insert((corp_sym.clone(), source.clone())) {
                continue;
            }
            let mut a = LegalAction::new("BuyShares");
            a.entity = entity_desc.clone();
            a.entity.insert("corp".to_string(), json!(corp_sym));
            a.params.insert("source".to_string(), json!(source));
            a.params.insert("percent".to_string(), json!(10));
            out.push(a);
        }
        out
    }

    fn buyable_shares_internal(
        &self,
        player_id: u32,
    ) -> Vec<(String, String, usize, i32)> {
        // Call the PyO3 method via the same implementation path.
        // It's defined as a pymethod; we duplicate the body of the public
        // API logic via the pub re-export already present.
        self.buyable_shares_for_factored(player_id)
    }

    fn factored_sell_shares(&self) -> Vec<LegalAction> {
        let player_id = match self.round_state.active_entity_id.player_id() {
            // OR sell_shares: the active entity is a corp but the *seller*
            // is the corp's president. Fall through below.
            Some(pid) => Some(pid),
            None => None,
        };

        // Determine actual seller: usually current entity if player; if in OR
        // and entity is a corp, seller is the president.
        let (seller_pid, sell_entity_desc) = if let Some(pid) = player_id {
            (pid, entity_descriptor_player(self, pid))
        } else if let Some(sym) = self.round_state.active_entity_id.corp_sym() {
            // In OR, sell_shares means the president sells.
            let ci = match self.corp_idx.get(sym) {
                Some(&i) => i,
                None => return Vec::new(),
            };
            let pid = match self.corporations[ci].president_id() {
                Some(pid) => pid,
                None => return Vec::new(),
            };
            (pid, entity_descriptor_player(self, pid))
        } else {
            return Vec::new();
        };

        let bundles = self.sellable_bundles_for_factored(seller_pid);
        let mut out: Vec<LegalAction> = Vec::new();
        for (corp_sym, count, percent) in bundles {
            let mut a = LegalAction::new("SellShares");
            a.entity = sell_entity_desc.clone();
            a.entity.insert("corp".to_string(), json!(corp_sym));
            a.params.insert("count".to_string(), json!(count));
            a.params.insert("percent".to_string(), json!(percent as i64));
            out.push(a);
        }
        out
    }

    fn factored_place_token(&mut self) -> Vec<LegalAction> {
        let corp_sym = match self.round_state.active_entity_id.corp_sym() {
            Some(s) => s.to_string(),
            None => return Vec::new(),
        };
        let entity_desc = entity_descriptor_corp(&corp_sym);

        let mut out: Vec<LegalAction> = Vec::new();

        // Pending tokens (OO upgrade displacements)
        let pending = match &self.round {
            Round::Operating(s) => s.pending_tokens.clone(),
            _ => Vec::new(),
        };
        if !pending.is_empty() {
            // The pending token enumeration in Python iterates the hexes the
            // tile-lay step recorded as needing tokens. The Rust engine
            // tracks these by (corp_sym, token_idx); the actual hex is the
            // corp's home or whichever was displaced. We re-enumerate via
            // tokenable_cities for now (this covers the common case; pending
            // tokens are rare in 1830).
            let cities = self.tokenable_cities_for_factored(&corp_sym);
            for (hex_id, city_idx) in cities {
                let slot = self.first_empty_slot(&hex_id, city_idx).unwrap_or(0);
                let mut a = LegalAction::new("PlaceToken");
                a.entity = entity_desc.clone();
                a.params.insert("hex".to_string(), json!(hex_id));
                a.params.insert("city".to_string(), json!(city_idx));
                a.params.insert("slot".to_string(), json!(slot));
                out.push(a);
            }
            return out;
        }

        let cities = self.tokenable_cities_for_factored(&corp_sym);
        for (hex_id, city_idx) in cities {
            let slot = self.first_empty_slot(&hex_id, city_idx).unwrap_or(0);
            let mut a = LegalAction::new("PlaceToken");
            a.entity = entity_desc.clone();
            a.params.insert("hex".to_string(), json!(hex_id));
            a.params.insert("city".to_string(), json!(city_idx));
            a.params.insert("slot".to_string(), json!(slot));
            out.push(a);
        }
        out
    }

    fn first_empty_slot(&self, hex_id: &str, city_idx: usize) -> Option<usize> {
        let hi = *self.hex_idx.get(hex_id)?;
        let hex = &self.hexes[hi];
        let city = hex.tile.cities.get(city_idx)?;
        for (i, t) in city.tokens.iter().enumerate() {
            if t.is_none() {
                return Some(i);
            }
        }
        Some(0)
    }

    fn factored_lay_tile(&mut self) -> Vec<LegalAction> {
        let corp_sym = match self.round_state.active_entity_id.corp_sym() {
            Some(s) => s.to_string(),
            None => return Vec::new(),
        };
        let entity_desc = entity_descriptor_corp(&corp_sym);
        let corp_cash = self
            .corp_idx
            .get(corp_sym.as_str())
            .map_or(0, |&i| self.corporations[i].cash);

        // Build blocked-hexes set (private companies still owned by a player)
        let mut blocked: std::collections::HashSet<String> = std::collections::HashSet::new();
        for co in &self.companies {
            if co.closed || !co.owner.is_player() {
                continue;
            }
            let h = match co.sym.as_str() {
                "SV" => vec!["G15"],
                "CS" => vec!["B20"],
                "DH" => vec!["F16"],
                "MH" => vec!["D18"],
                "CA" => vec!["H18"],
                "BO" => vec!["I13", "I15"],
                _ => vec![],
            };
            for x in h {
                blocked.insert(x.to_string());
            }
        }

        let connected = self.connected_hexes_for_factored(&corp_sym);

        // For Python parity: include both directly-connected hexes AND their
        // outward neighbors (entry edges) — matches the legal_action_types code.
        let mut targets: Vec<(String, Vec<u8>)> = Vec::new();
        let mut visited: std::collections::HashSet<String> = std::collections::HashSet::new();
        for (hex_id, edges) in connected.iter() {
            if visited.insert(hex_id.clone()) {
                targets.push((hex_id.clone(), edges.clone()));
            }
            if let Some(neighbors) = self.hex_adjacency.get(hex_id) {
                for &edge in edges {
                    if let Some(nid) = neighbors.get(&edge) {
                        if !visited.contains(nid) {
                            let entry_edge = (edge + 3) % 6;
                            targets.push((nid.clone(), vec![entry_edge]));
                            visited.insert(nid.clone());
                        }
                    }
                }
            }
        }

        // Sort targets for determinism (by hex_id).
        targets.sort_by(|a, b| a.0.cmp(&b.0));

        let mut out: Vec<LegalAction> = Vec::new();
        for (hex_id, edges) in targets {
            if blocked.contains(&hex_id) {
                continue;
            }
            // Terrain cost check
            let hi = match self.hex_idx.get(hex_id.as_str()) {
                Some(&i) => i,
                None => continue,
            };
            let terrain_cost: i32 = self.hexes[hi].tile.upgrades.iter().map(|u| u.cost).sum();
            if terrain_cost > corp_cash {
                continue;
            }
            // Enumerate (tile, rotation) pairs
            let pairs = self.factored_layable_tiles_for(&hex_id, &edges);
            for (tile_name, rotation) in pairs {
                let mut a = LegalAction::new("LayTile");
                a.entity = entity_desc.clone();
                a.params.insert("hex".to_string(), json!(hex_id.clone()));
                a.params.insert("tile".to_string(), json!(tile_name));
                a.params.insert("rotation".to_string(), json!(rotation));
                out.push(a);
            }
        }
        out.sort_by(|a, b| {
            let ah = a.params.get("hex").map(|v| v.to_string()).unwrap_or_default();
            let bh = b.params.get("hex").map(|v| v.to_string()).unwrap_or_default();
            let at = a.params.get("tile").map(|v| v.to_string()).unwrap_or_default();
            let bt = b.params.get("tile").map(|v| v.to_string()).unwrap_or_default();
            let ar = a.params.get("rotation").and_then(|v| v.as_u64()).unwrap_or(0);
            let br = b.params.get("rotation").and_then(|v| v.as_u64()).unwrap_or(0);
            (ah, at, ar).cmp(&(bh, bt, br))
        });
        out
    }

    /// Helper: enumerate (tile_name, rotation) layable pairs for one hex,
    /// constrained to the corp's connected edges on the hex.
    fn factored_layable_tiles_for(
        &self,
        hex_id: &str,
        corp_connected_edges: &[u8],
    ) -> Vec<(String, u8)> {
        let hi = match self.hex_idx.get(hex_id) {
            Some(&i) => i,
            None => return Vec::new(),
        };

        let hex = &self.hexes[hi];
        let current_tile = &hex.tile;

        let old_tile_def = self.tile_catalog.get(&current_tile.name);
        let old_tile_def = match old_tile_def {
            Some(def) => def.rotated(current_tile.rotation),
            None => {
                crate::tiles::TileDef {
                    name: current_tile.name.clone(),
                    color: current_tile.color,
                    paths: current_tile.paths.clone(),
                    cities: current_tile.cities.iter().map(|c| crate::tiles::CityDef {
                        revenue: c.revenue,
                        slots: c.slots as u8,
                    }).collect(),
                    towns: current_tile.towns.iter().map(|t| crate::tiles::TownDef {
                        revenue: t.revenue,
                    }).collect(),
                    offboards: Vec::new(),
                    edges: crate::tiles::TileDef::compute_edges_pub(&current_tile.paths),
                    upgrades: Vec::new(),
                    label: current_tile.label.clone(),
                    has_junction: current_tile.paths.iter().any(|p|
                        p.a == crate::tiles::PathEndpoint::Junction
                            || p.b == crate::tiles::PathEndpoint::Junction
                    ),
                }
            }
        };

        let valid_exits: Vec<u8> = self
            .hex_adjacency
            .get(hex_id)
            .map(|n| n.keys().copied().collect())
            .unwrap_or_default();

        let next_color = match old_tile_def.color.next_color() {
            Some(c) => c,
            None => return Vec::new(),
        };

        let next_color_str = format!("{:?}", next_color).to_lowercase();
        if !self.phase.tiles.iter().any(|t| t == &next_color_str) {
            return Vec::new();
        }

        let mut out = Vec::new();
        for (tile_name, tile_def) in self.tile_catalog.iter() {
            if tile_def.color != next_color {
                continue;
            }
            let remaining = self
                .tile_counts_remaining
                .get(tile_name)
                .copied()
                .unwrap_or(0);
            if remaining == 0 {
                continue;
            }
            if !tile_def.is_valid_upgrade_for(&old_tile_def) {
                continue;
            }
            let rotations = tile_def.legal_rotations_for(&old_tile_def, &valid_exits);
            for &rot in &rotations {
                let rotated = tile_def.rotated(rot);
                // Must reach a new exit through corp's connected edges.
                if rotated.edges.iter().any(|e| corp_connected_edges.contains(e)) {
                    out.push((tile_name.clone(), rot));
                }
            }
        }
        out
    }

    fn factored_discard_train(&self) -> Vec<LegalAction> {
        let corp_sym = match self.round_state.active_entity_id.corp_sym() {
            Some(s) => s.to_string(),
            None => return Vec::new(),
        };
        let entity_desc = entity_descriptor_corp(&corp_sym);
        let ci = match self.corp_idx.get(corp_sym.as_str()) {
            Some(&i) => i,
            None => return Vec::new(),
        };
        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut out: Vec<LegalAction> = Vec::new();
        for train in &self.corporations[ci].trains {
            if !seen.insert(train.name.clone()) {
                continue;
            }
            let mut a = LegalAction::new("DiscardTrain");
            a.entity = entity_desc.clone();
            a.params.insert("train".to_string(), json!(train.name.clone()));
            a.params.insert("train_id".to_string(), json!(train.id.clone()));
            out.push(a);
        }
        out
    }

    fn factored_run_routes(&self) -> Vec<LegalAction> {
        let mut a = LegalAction::new("RunRoutes");
        a.entity = current_entity_descriptor(self);
        vec![a]
    }

    fn factored_dividend(&self) -> Vec<LegalAction> {
        let entity_desc = current_entity_descriptor(self);
        ["payout", "withhold"]
            .iter()
            .map(|kind| {
                let mut a = LegalAction::new("Dividend");
                a.entity = entity_desc.clone();
                a.params.insert("kind".to_string(), json!(kind));
                a
            })
            .collect()
    }

    fn factored_bankrupt(&self) -> Vec<LegalAction> {
        let mut a = LegalAction::new("Bankrupt");
        a.entity = current_entity_descriptor(self);
        vec![a]
    }

    fn factored_buy_company(&self) -> Vec<LegalAction> {
        // Corp buys a private from its president in OR.
        let corp_sym = match self.round_state.active_entity_id.corp_sym() {
            Some(s) => s.to_string(),
            None => return Vec::new(),
        };
        let ci = match self.corp_idx.get(corp_sym.as_str()) {
            Some(&i) => i,
            None => return Vec::new(),
        };
        let corp = &self.corporations[ci];
        let pres_id = match corp.president_id() {
            Some(pid) => pid,
            None => return Vec::new(),
        };
        let buying_power = corp.cash;
        let entity_desc = entity_descriptor_corp(&corp_sym);

        let pres_eid = EntityId::player(pres_id);
        let mut out: Vec<LegalAction> = Vec::new();
        for co in &self.companies {
            if co.closed || co.no_buy || co.owner != pres_eid {
                continue;
            }
            let min_price = co.value / 2;
            let max_price = (co.value * 2).min(buying_power);
            if max_price < min_price {
                continue;
            }
            let mut a = LegalAction::new("BuyCompany");
            a.entity = entity_desc.clone();
            a.entity
                .insert("private".to_string(), json!(co.sym.clone()));
            a.price_range = Some((min_price as i64, max_price as i64));
            out.push(a);
        }
        out
    }

    fn factored_buy_train(&mut self) -> Vec<LegalAction> {
        let corp_sym = match self.round_state.active_entity_id.corp_sym() {
            Some(s) => s.to_string(),
            None => return Vec::new(),
        };
        let entity_desc = entity_descriptor_corp(&corp_sym);

        let ci = match self.corp_idx.get(corp_sym.as_str()) {
            Some(&i) => i,
            None => return Vec::new(),
        };
        let corp_cash = self.corporations[ci].cash;
        let pres_id = self.corporations[ci].president_id();
        let pres_cash = pres_id
            .and_then(|pid| self.players.iter().find(|p| p.id == pid))
            .map_or(0, |p| p.cash);
        let pres_may_contribute = self.president_may_contribute_pub(&corp_sym);

        // Determine cheapest depot train name for ebuy filter. The
        // cheapest-only restriction (Python's
        // ``check_for_cheapest_train``) only applies when the corp's cash
        // is less than the train price (i.e. president must actually
        // contribute for the specific train). When the corp can afford a
        // depot train on its own, any depot train is buyable.
        let cheapest_name: Option<String> = if pres_may_contribute {
            self.depot.trains.first().map(|t| t.name.clone())
        } else {
            None
        };

        let mut trains = self.buyable_trains_for_factored(&corp_sym);
        // Phase availability filter for depot trains: only the first upcoming
        // train + trains whose name is <= current phase are visible. This
        // matches Python's `Depot.depot_trains` which gates by
        // `phase.available(train.available_on)`. The Rust Train struct
        // doesn't track `available_on`, so we infer from the name vs.
        // current phase: a train "X" is visible if X equals the current
        // phase name or one of the previously-passed phases.
        let phase_name = self.phase.name.clone();
        // 1830 phase order: "2", "3", "4", "5", "6", "D".
        let phase_order = ["2", "3", "4", "5", "6", "D"];
        let phase_idx = phase_order
            .iter()
            .position(|&p| p == phase_name)
            .unwrap_or(0);
        let depot_first_name: Option<String> =
            self.depot.trains.first().map(|t| t.name.clone());
        trains.retain(|(_id, name, _price, source)| {
            if source != "depot" {
                return true;
            }
            // Always show the head-of-depot train (next to sell).
            if Some(name) == depot_first_name.as_ref() {
                return true;
            }
            // Otherwise must be at-or-before current phase.
            let train_idx = phase_order
                .iter()
                .position(|&p| p == name.as_str())
                .unwrap_or(usize::MAX);
            train_idx <= phase_idx
        });

        // Deduplicate by (source, train_name).
        let mut seen: std::collections::HashSet<(String, String)> = std::collections::HashSet::new();

        let mut out: Vec<LegalAction> = Vec::new();
        for (_train_id, name, price, source) in &trains {
            if !seen.insert((source.clone(), name.clone())) {
                continue;
            }
            if source == "depot" {
                // Depot trains: fixed price (min == max)
                let min_p = *price;
                // Cheapest-only restriction applies only when this train
                // requires president contribution (corp can't afford on
                // its own).
                let requires_pres_help = min_p > corp_cash;
                if requires_pres_help {
                    if let Some(cn) = &cheapest_name {
                        if cn != name {
                            continue;
                        }
                    }
                }
                let affordable = if pres_may_contribute {
                    min_p <= corp_cash + pres_cash
                } else {
                    min_p <= corp_cash
                };
                if !affordable {
                    continue;
                }
                let mut a = LegalAction::new("BuyTrain");
                a.entity = entity_desc.clone();
                a.entity.insert("source".to_string(), json!("depot"));
                a.entity.insert("train".to_string(), json!(name.clone()));
                a.price_range = Some((min_p as i64, min_p as i64));
                out.push(a);
            } else {
                // Cross-corp trains: min 1, max = corp_cash (+ pres if ebuy)
                let min_p: i32 = 1;
                let mut max_p = if pres_may_contribute {
                    corp_cash + pres_cash
                } else {
                    corp_cash
                };
                // Cap at train face price (Python's spend_minmax caps at face).
                if max_p > *price {
                    max_p = *price;
                }
                if max_p < min_p {
                    continue;
                }
                let mut a = LegalAction::new("BuyTrain");
                a.entity = entity_desc.clone();
                a.entity.insert("source".to_string(), json!(source.clone()));
                a.entity.insert("train".to_string(), json!(name.clone()));
                a.price_range = Some((min_p as i64, max_p as i64));
                out.push(a);
            }
        }
        out
    }

    /// Company-as-actor branch: MH exchange (CompanyBuyShares).
    fn factored_company_actions(&self) -> Vec<LegalAction> {
        let mut out: Vec<LegalAction> = Vec::new();

        // MH exchange: NYC IPO share for free if MH owner is the current player.
        let mh = self.companies.iter().find(|c| c.sym == "MH" && !c.closed);
        if let Some(mh) = mh {
            if mh.owner.is_player() {
                let mh_owner_pid = mh.owner.player_id();
                // Restrict surfacing to the current actor:
                // - if current entity is a player, must equal MH owner
                // - if current entity is a corp, the corp's president must equal MH owner
                let current_pid = if self.round_state.active_entity_id.is_player() {
                    self.round_state.active_entity_id.player_id()
                } else if let Some(sym) = self.round_state.active_entity_id.corp_sym() {
                    self.corp_idx
                        .get(sym)
                        .and_then(|&i| self.corporations[i].president_id())
                } else {
                    None
                };
                let surface = match (current_pid, mh_owner_pid) {
                    (Some(cpid), Some(mpid)) => cpid == mpid,
                    _ => true,
                };
                if surface {
                    // NYC must be parred and have a non-president, non-player IPO share.
                    if let Some(&ci) = self.corp_idx.get("NYC") {
                        let nyc = &self.corporations[ci];
                        if nyc.ipo_price.is_some() {
                            let has_ipo_share = nyc.shares.iter().any(|s| {
                                !s.president && !s.owner.is_player() && s.owner.is_ipo()
                            });
                            let has_market_share = nyc.shares.iter().any(|s| {
                                !s.president && !s.owner.is_player() && s.owner.is_market()
                            });
                            if has_ipo_share {
                                let mut a = LegalAction::new("CompanyBuyShares");
                                a.entity.insert("private".to_string(), json!("MH"));
                                a.entity.insert("corp".to_string(), json!("NYC"));
                                a.params.insert("source".to_string(), json!("ipo"));
                                a.params.insert("percent".to_string(), json!(10));
                                out.push(a);
                            }
                            if has_market_share {
                                let mut a = LegalAction::new("CompanyBuyShares");
                                a.entity.insert("private".to_string(), json!("MH"));
                                a.entity.insert("corp".to_string(), json!("NYC"));
                                a.params.insert("source".to_string(), json!("market"));
                                a.params.insert("percent".to_string(), json!(10));
                                out.push(a);
                            }
                        }
                    }
                }
            }
        }

        // Deferred branches: the Python `FactoredActionHelper` surfaces
        // three more company-as-actor branches that are not yet ported:
        //
        //   1. ``LayTile`` via ``CS`` special-track ability on B20.
        //   2. ``LayTile`` + ``PlaceToken`` via ``DH`` teleport ability on F16.
        //   3. ``CompanyBuyShares`` (MH -> NYC exchange) *before* NYC pars —
        //      the pre-par reserved-shares path. The post-par path is
        //      already handled above (gated by ``nyc.ipo_price.is_some()``).
        //
        // The pipeline tolerates these gaps:
        //  * `tests/test_factored_action_helper_rust_parity.py::_is_ignored_key`
        //    marks ``CompanyBuyShares`` as out-of-scope.
        //  * MCTS / pretraining run through the Python `FactoredActionHelper`
        //    when the game state is a Python `BaseGame`; the gaps only matter
        //    while running self-play on `RustGameAdapter`.
        //  * The Rust engine still *processes* these actions correctly if a
        //    higher-level helper emits them — the gap is purely enumeration.
        //
        // Porting these requires Rust equivalents of `SpecialTrackStep.
        // potential_tiles` / `legal_tile_rotations` and the engine's
        // pre-par reserved-share bookkeeping, both of which are non-trivial
        // and warrant a dedicated follow-up.

        out
    }
}

// ---------------------------------------------------------------------------
// PyO3 entry point
// ---------------------------------------------------------------------------

#[pymethods]
impl BaseGame {
    /// Return the factored legal-action list as a `Vec<dict>`.
    ///
    /// Each entry has keys ``type``, ``entity``, ``params``, ``price_range``
    /// matching the Python ``FactoredActionHelper.LegalAction`` dataclass.
    fn get_factored_choices(&mut self, py: Python<'_>) -> PyResult<Vec<Py<PyDict>>> {
        let choices = self.get_factored_choices_impl();
        choices.iter().map(|c| c.to_py_dict(py)).collect()
    }
}
