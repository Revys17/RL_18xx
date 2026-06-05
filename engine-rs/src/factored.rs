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

        // Pending tokens (OO upgrade displacements & home-token on tile-reserved hex).
        // Python's HomeToken step restricts enumeration to the pending hex list:
        //   `for hex in pending_token['hexes'] for city in hex.tile.cities if city.tokenable(...)`
        // The Rust state stores the hex on the pending entry, so we iterate the
        // cities on that hex and filter to ones that still have a free slot.
        let pending = match &self.round {
            Round::Operating(s) => s.pending_tokens.clone(),
            _ => Vec::new(),
        };
        if !pending.is_empty() {
            let (_pending_corp, _pending_idx, pending_hex) = &pending[0];
            if let Some(&hi) = self.hex_idx.get(pending_hex.as_str()) {
                let cities = self.hexes[hi].tile.cities.clone();
                for (city_idx, city) in cities.iter().enumerate() {
                    // City must have at least one empty slot
                    if !city.tokens.iter().any(|t| t.is_none()) {
                        continue;
                    }
                    let slot = self.first_empty_slot(pending_hex, city_idx).unwrap_or(0);
                    let mut a = LegalAction::new("PlaceToken");
                    a.entity = entity_desc.clone();
                    a.params.insert("hex".to_string(), json!(pending_hex));
                    a.params.insert("city".to_string(), json!(city_idx));
                    a.params.insert("slot".to_string(), json!(slot));
                    out.push(a);
                }
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

        // DH teleport token: after the DH tile lay (`ability_used`), the owning
        // corp may place a station token on F16 even though it isn't connected.
        // Mirrors Python's `_company_place_token_choices` for the DH SpecialToken
        // ability; gate condition matches `legal_action_types`'s dh_token check.
        let corp_eid = crate::entities::EntityId::corporation(&corp_sym);
        let dh_token = self.companies.iter().any(|co| {
            co.sym == "DH" && !co.closed && co.ability_used && co.owner == corp_eid
        });
        if dh_token {
            if let Some(&ci) = self.corp_idx.get(corp_sym.as_str()) {
                if self.corporations[ci].next_token_index().is_some() {
                    if let Some(&hi) = self.hex_idx.get("F16") {
                        let cities = self.hexes[hi].tile.cities.clone();
                        for (city_idx, city) in cities.iter().enumerate() {
                            if !city.tokens.iter().any(|t| t.is_none()) {
                                continue;
                            }
                            // Don't duplicate an F16 city already emitted above.
                            let already = out.iter().any(|a| {
                                a.params.get("hex").and_then(|v| v.as_str()) == Some("F16")
                                    && a.params.get("city").and_then(|v| v.as_u64())
                                        == Some(city_idx as u64)
                            });
                            if already {
                                continue;
                            }
                            let slot = self.first_empty_slot("F16", city_idx).unwrap_or(0);
                            let mut a = LegalAction::new("PlaceToken");
                            a.entity = entity_desc.clone();
                            a.params.insert("hex".to_string(), json!("F16"));
                            a.params.insert("city".to_string(), json!(city_idx));
                            a.params.insert("slot".to_string(), json!(slot));
                            out.push(a);
                        }
                    }
                }
            }
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

    /// Candidate `(hex_id, connected_edges)` pairs for tile-laying, replicating
    /// Python's `Graph.compute` population of `connected_hexes`
    /// (graph.py:1405-1482):
    ///
    ///   * base: the walked path-exit edges the canonical graph recorded;
    ///   * (B) frontier (graph.py:1479-1482): for every walked exit edge on a
    ///     hex, the neighbour hex across it is connected at the inverted edge;
    ///   * (A) tokened-city hex (graph.py:1415-1416): a city the corp has
    ///     tokened is connected on ALL its passable neighbour edges.
    ///
    /// Shared by `factored_lay_tile` (the enumerator) and the
    /// `legal_action_types` gate so both agree on reachability. `connected_hexes`
    /// is consumed only by these enumeration paths (never route availability /
    /// revenue), so this can't perturb replay parity.
    pub(crate) fn lay_tile_candidate_hexes(&mut self, corp_sym: &str) -> Vec<(String, Vec<u8>)> {
        let connected = self.connected_hexes_for_factored(corp_sym);
        let mut conn: HashMap<String, std::collections::HashSet<u8>> = HashMap::new();
        for (h, edges) in &connected {
            conn.entry(h.clone()).or_default().extend(edges.iter().copied());
        }
        // (B) frontier expansion from the walked exits.
        for (h, edges) in &connected {
            if let Some(adj) = self.hex_adjacency.get(h) {
                for &edge in edges {
                    if let Some(nid) = adj.get(&edge) {
                        let inv = (edge + 3) % 6;
                        conn.entry(nid.clone()).or_default().insert(inv);
                    }
                }
            }
        }
        // (A) tokened-city hexes connect on all passable neighbour edges.
        for (token_hex, _ci) in self.corp_token_positions(corp_sym) {
            let exits = self.passable_exits_for(&token_hex);
            let entry = conn.entry(token_hex).or_default();
            for e in exits {
                entry.insert(e);
            }
        }
        let mut targets: Vec<(String, Vec<u8>)> = conn
            .into_iter()
            .map(|(h, set)| {
                let mut v: Vec<u8> = set.into_iter().collect();
                v.sort();
                (h, v)
            })
            .collect();
        targets.sort_by(|a, b| a.0.cmp(&b.0));
        targets
    }

    fn factored_lay_tile(&mut self) -> Vec<LegalAction> {
        let corp_sym = match self.round_state.active_entity_id.corp_sym() {
            Some(s) => s.to_string(),
            None => return Vec::new(),
        };

        // Determine the current OR step. The corp can lay tiles only at the
        // LayTile step. At other steps, "lay_tile" is in legal_action_types
        // because CS (and possibly DH) has an unused ability.
        let or_step = match &self.round {
            Round::Operating(s) => Some(s.step.clone()),
            _ => None,
        };
        let at_lay_tile_step = matches!(or_step, Some(crate::rounds::OperatingStep::LayTile));

        // Always check for CS company lay availability. CS fires at any step
        // during the owning corp's OR turn (count=1, hexes=B20, tiles=3/4/58).
        let mut out: Vec<LegalAction> = Vec::new();
        out.extend(self.factored_cs_lay_tile(&corp_sym));
        out.extend(self.factored_dh_lay_tile(&corp_sym));

        // Skip corp-side enumeration when not at the LayTile step. Python's
        // `game.round.actions_for(corp)` does not include LayTile at
        // BuyCompany/BuyTrain/Dividend/RunRoutes/PlaceToken steps.
        if !at_lay_tile_step {
            return out;
        }

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

        // Candidate (hex, connected-edge) pairs, replicating Python's
        // `Graph.connected_hexes` exactly. Shared with the `legal_action_types`
        // gate via `lay_tile_candidate_hexes` so the gate and this enumerator
        // agree on reachability.
        let targets = self.lay_tile_candidate_hexes(&corp_sym);

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

    /// Enumerate CS company tile-lay actions on B20.
    ///
    /// CS rules (1830):
    /// * Owner must be the active operating corp.
    /// * Company must be open and ability unused.
    /// * Only hex B20, only yellow tiles "3", "4", "58".
    /// * Tile must be a valid upgrade of B20's current tile (yellow over the
    ///   pre-printed blank, with at least one rotation that connects to an
    ///   adjacent neighbor edge).
    /// * Free of charge (no terrain cost on B20).
    fn factored_cs_lay_tile(&self, corp_sym: &str) -> Vec<LegalAction> {
        // Find a CS company owned by the active corp with unused ability.
        let corp_eid = crate::entities::EntityId::corporation(corp_sym);
        let cs_available = self
            .companies
            .iter()
            .any(|co| co.sym == "CS" && !co.closed && !co.ability_used && co.owner == corp_eid);
        if !cs_available {
            return Vec::new();
        }

        let hex_id = "B20";
        let hi = match self.hex_idx.get(hex_id) {
            Some(&i) => i,
            None => return Vec::new(),
        };
        let hex = &self.hexes[hi];
        let current_tile = &hex.tile;

        // CS only lays on the blank pre-printed hex (yellow tiles are valid
        // upgrades). If a tile is already laid, the CS ability is moot.
        // We still validate via `is_valid_upgrade_for` below for safety.
        let old_tile_def = match self.tile_catalog.get(&current_tile.name) {
            Some(def) => def.rotated(current_tile.rotation),
            None => crate::tiles::TileDef {
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
            },
        };

        let valid_exits = self.passable_exits_for(hex_id);

        let mut out: Vec<LegalAction> = Vec::new();
        let cs_tile_names = ["3", "4", "58"];
        for tile_name in cs_tile_names.iter() {
            let tile_def = match self.tile_catalog.get(*tile_name) {
                Some(def) => def,
                None => continue,
            };
            // Phase must allow this tile's color (always yellow for CS).
            let color_str = format!("{:?}", tile_def.color).to_lowercase();
            if !self.phase.tiles.iter().any(|t| t == &color_str) {
                continue;
            }
            let remaining = self
                .tile_counts_remaining
                .get(*tile_name)
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
                let mut a = LegalAction::new("LayTile");
                a.entity
                    .insert("private".to_string(), json!("CS"));
                a.params.insert("hex".to_string(), json!(hex_id));
                a.params.insert("tile".to_string(), json!((*tile_name).to_string()));
                a.params.insert("rotation".to_string(), json!(rot));
                out.push(a);
            }
        }
        out
    }

    /// Helper: return the set of edges on `hex_id` that are "passable" — i.e.,
    /// have an adjacent hex that is NOT impassable (gray/blue/red) unless the
    /// adjacent hex's tile already targets us back, AND the edge is not
    /// blocked by a hardcoded impassable border on the starting tile. Mirrors
    /// Python's ``Hex.neighbors`` (vs ``all_neighbors``) which filters out
    /// impassable neighbors during ``connect_hexes`` and tile-laying logic.
    /// Enumerate DH special-track tile lays on F16 (teleport ability).
    ///
    /// DH rules (1830, g1830.py:54-68): a corp owning DH may lay tile "57" on
    /// hex F16 (and later place a station token there) for the $120 mountain
    /// cost. The ability is a teleport — no track connectivity is required.
    /// Mirrors Python's `_company_lay_tile_choices` for the DH SpecialTrack
    /// ability: emit one LayTile per legal rotation, entity `{private: "DH"}`,
    /// gated on the corp being able to afford the F16 upgrade cost.
    fn factored_dh_lay_tile(&self, corp_sym: &str) -> Vec<LegalAction> {
        let corp_eid = crate::entities::EntityId::corporation(corp_sym);
        let dh_available = self.companies.iter().any(|co| {
            co.sym == "DH" && !co.closed && !co.ability_used && co.owner == corp_eid
        });
        if !dh_available {
            return Vec::new();
        }

        let hi = match self.hex_idx.get("F16") {
            Some(&i) => i,
            None => return Vec::new(),
        };
        // Python gates the DH lay on `upgrade_cost <= buying_power(corp)`; the
        // F16 mountain upgrade costs $120.
        let corp_cash = self
            .corp_idx
            .get(corp_sym)
            .map_or(0, |&i| self.corporations[i].cash);
        let f16_cost: i32 = self.hexes[hi].tile.upgrades.iter().map(|u| u.cost).sum();
        if f16_cost > corp_cash {
            return Vec::new();
        }

        let current_tile = &self.hexes[hi].tile;
        let old_tile_def = match self.tile_catalog.get(&current_tile.name) {
            Some(def) => def.rotated(current_tile.rotation),
            None => crate::tiles::TileDef {
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
                has_junction: current_tile.paths.iter().any(|p| {
                    p.a == crate::tiles::PathEndpoint::Junction
                        || p.b == crate::tiles::PathEndpoint::Junction
                }),
            },
        };

        let valid_exits = self.passable_exits_for("F16");
        let tile_name = "57";
        let tile_def = match self.tile_catalog.get(tile_name) {
            Some(d) => d,
            None => return Vec::new(),
        };
        let color_str = format!("{:?}", tile_def.color).to_lowercase();
        if !self.phase.tiles.iter().any(|t| t == &color_str) {
            return Vec::new();
        }
        if self.tile_counts_remaining.get(tile_name).copied().unwrap_or(0) == 0 {
            return Vec::new();
        }
        if !tile_def.is_valid_upgrade_for(&old_tile_def) {
            return Vec::new();
        }

        let mut out: Vec<LegalAction> = Vec::new();
        for &rot in &tile_def.legal_rotations_for(&old_tile_def, &valid_exits) {
            let mut a = LegalAction::new("LayTile");
            a.entity.insert("private".to_string(), json!("DH"));
            a.params.insert("hex".to_string(), json!("F16"));
            a.params.insert("tile".to_string(), json!(tile_name.to_string()));
            a.params.insert("rotation".to_string(), json!(rot));
            out.push(a);
        }
        out
    }

    pub(crate) fn passable_exits_for(&self, hex_id: &str) -> Vec<u8> {
        let neighbors = match self.hex_adjacency.get(hex_id) {
            Some(n) => n,
            None => return Vec::new(),
        };
        let impassable_border_edges: &[u8] = match hex_id {
            "E7" => &[5],
            "F8" => &[2],
            "C11" => &[5],
            "C13" => &[0],
            "D12" => &[2, 3],
            "B16" => &[5],
            "C17" => &[2],
            _ => &[],
        };
        let mut out: Vec<u8> = Vec::new();
        for (&edge, neighbor_id) in neighbors.iter() {
            if impassable_border_edges.contains(&edge) {
                continue;
            }
            let ni = match self.hex_idx.get(neighbor_id) {
                Some(&i) => i,
                None => continue,
            };
            let neighbor_tile = &self.hexes[ni].tile;
            let impassable = matches!(
                neighbor_tile.color,
                crate::tiles::TileColor::Gray
                    | crate::tiles::TileColor::Red
            );
            if impassable {
                // Skip unless neighbor targets us back — its tile must have an
                // exit on the inverse edge `(edge + 3) % 6`.
                let inverse = (edge + 3) % 6;
                let targets_back = neighbor_tile.paths.iter().any(|p| {
                    matches!(p.a, crate::tiles::PathEndpoint::Edge(e) if e == inverse)
                        || matches!(p.b, crate::tiles::PathEndpoint::Edge(e) if e == inverse)
                });
                if !targets_back {
                    continue;
                }
            }
            out.push(edge);
        }
        out.sort();
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

        let valid_exits = self.passable_exits_for(hex_id);

        let next_color = match old_tile_def.color.next_color() {
            Some(c) => c,
            None => return Vec::new(),
        };

        let next_color_str = format!("{:?}", next_color).to_lowercase();
        if !self.phase.tiles.iter().any(|t| t == &next_color_str) {
            return Vec::new();
        }

        // Sort tile_catalog by tile name for deterministic iteration order.
        let mut catalog_sorted: Vec<(&String, &crate::tiles::TileDef)> =
            self.tile_catalog.iter().collect();
        catalog_sorted.sort_by(|a, b| a.0.cmp(b.0));

        let mut out = Vec::new();
        for (tile_name, tile_def) in catalog_sorted.iter() {
            if tile_def.color != next_color {
                continue;
            }
            let remaining = self
                .tile_counts_remaining
                .get(*tile_name)
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
                    out.push((tile_name.to_string(), rot));
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
        //
        // Python's ``depot.min_depot_train`` considers BOTH upcoming and
        // discarded depot trains. Use the lowest-priced across both.
        let cheapest_name: Option<String> = if pres_may_contribute {
            let mut cheapest: Option<(i32, String)> = None;
            for t in &self.depot.trains {
                match &cheapest {
                    None => cheapest = Some((t.price, t.name.clone())),
                    Some((p, _)) if t.price < *p => {
                        cheapest = Some((t.price, t.name.clone()))
                    }
                    _ => {}
                }
            }
            for t in &self.depot.discarded {
                match &cheapest {
                    None => cheapest = Some((t.price, t.name.clone())),
                    Some((p, _)) if t.price < *p => {
                        cheapest = Some((t.price, t.name.clone()))
                    }
                    _ => {}
                }
            }
            cheapest.map(|(_, n)| n)
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
            // Both "depot" (upcoming) and "discard" (face-value discarded pool)
            // trains are owned by the Depot in Python, so `Train.from_depot()`
            // is True for both and Python's `_train_source_descriptor` labels
            // them identically as "depot". Normalize before dedup so an
            // upcoming and a discarded train of the same name collapse to one
            // categorical option (matching Python's `_unique_trains`, which
            // dedups by (name, owner) and both share the depot owner).
            let is_depot_like = source == "depot" || source == "discard";
            let out_source = if is_depot_like {
                "depot".to_string()
            } else {
                source.clone()
            };
            if !seen.insert((out_source.clone(), name.clone())) {
                continue;
            }
            if is_depot_like {
                // Depot/discard trains: fixed price (min == max).
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
                a.entity.insert("source".to_string(), json!(out_source.clone()));
                a.entity.insert("train".to_string(), json!(name.clone()));
                a.price_range = Some((min_p as i64, min_p as i64));
                out.push(a);
            } else {
                // Cross-corp (inter-corporation) train. Mirror Python's
                // `BuyTrain.spend_minmax` exactly. For 1830:
                // `EBUY_OTHER_VALUE == True` and `buying_power(corp) == corp.cash`.
                //
                //   if corp.cash < face:        # emergency branch
                //       min = 1 (or, after an emergency share sale,
                //              corp.cash + owner.cash - last_share_sold_price + 1)
                //       max = min(face, corp.cash + owner.cash)
                //   else:                        # normal branch — NO face cap
                //       min = 1; max = corp.cash
                //
                // then, if the president may not contribute, cap max at corp.cash.
                //
                let face = *price;
                let bp = corp_cash; // buying_power(entity) == corp.cash in 1830
                // After an emergency share sale this turn, Python narrows the
                // minimum to `corp.cash + owner.cash - last_share_sold_price + 1`
                // (round.py:752-753); otherwise the emergency min is 1.
                let last_sold = match &self.round {
                    Round::Operating(s) => s.last_share_sold_price,
                    _ => None,
                };
                let (min_p, mut max_p) = if bp < face {
                    let emergency_min = match last_sold {
                        Some(lssp) => bp + pres_cash - lssp + 1,
                        None => 1,
                    };
                    (emergency_min, face.min(bp + pres_cash))
                } else {
                    (1_i32, bp)
                };
                if !pres_may_contribute && max_p > corp_cash {
                    max_p = corp_cash;
                }
                if max_p < min_p {
                    continue;
                }
                let mut a = LegalAction::new("BuyTrain");
                a.entity = entity_desc.clone();
                a.entity.insert("source".to_string(), json!(out_source.clone()));
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
        // Python surfaces this via `round.actions_for(MH)`, which exposes the
        // Exchange step only in the Stock and Operating rounds — never during the
        // initial private Auction (a player can own MH mid-auction, but the
        // exchange isn't available until the auction resolves).
        let in_auction = matches!(self.round, Round::Auction(_));
        let mh = self.companies.iter().find(|c| c.sym == "MH" && !c.closed);
        if let Some(mh) = mh {
            if mh.owner.is_player() && !in_auction {
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
                    if let Some(&ci) = self.corp_idx.get("NYC") {
                        let nyc = &self.corporations[ci];
                        // Python's `Exchange.exchangeable_shares`: MH (from =
                        // [ipo, market], when = any) can claim an available NYC
                        // ipo/market share at ANY time — including BEFORE NYC pars
                        // — filtered by `can_gain`, whose binding 1830 rule is "the
                        // MH owner does not already hold 60% of NYC". No par gate
                        // (the IPO shares exist in `nyc.shares` pre-par).
                        let owner_pct = mh_owner_pid
                            .map(|pid| nyc.percent_owned_by(&crate::entities::EntityId::player(pid)))
                            .unwrap_or(0);
                        if owner_pct < 60 {
                            // Pre-par, this engine hasn't materialised the corp's
                            // shares into `nyc.shares` yet (they're created at par),
                            // but the IPO conceptually still holds all 100% — so a
                            // 10% IPO share is available to claim, matching Python's
                            // `available_share` (which reads the bank/IPO, not the
                            // corp treasury).
                            let has_ipo_share = if nyc.ipo_price.is_none() {
                                true
                            } else {
                                nyc.shares.iter().any(|s| {
                                    !s.president && !s.owner.is_player() && s.owner.is_ipo()
                                })
                            };
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

        // The other company-as-actor branches the Python `FactoredActionHelper`
        // surfaces are now all ported and exercised by the strict parity test:
        //   * ``LayTile`` via ``CS`` (B20) — `factored_cs_lay_tile`.
        //   * ``LayTile`` + ``PlaceToken`` via ``DH`` teleport (F16) —
        //     `factored_dh_lay_tile` and the DH branch in `factored_place_token`.
        //   * ``CompanyBuyShares`` (MH -> NYC) including the pre-par claim,
        //     handled above (no par gate; the IPO holds the share pre-par).

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
