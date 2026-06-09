//! Native index → `Action` decode.
//!
//! Mirrors `rl18xx/agent/alphazero/action_mapper.py::map_index_to_action` /
//! `map_index_to_action_with_price`, but builds a Rust [`crate::actions::Action`]
//! directly from the engine state — removing the Python `ActionMapper` from the
//! MCTS / self-play hot loop.
//!
//! Strategy: bridge through the factored enumeration. The Rust
//! `get_factored_choices_impl()` already resolves the concrete entity / hex /
//! train / source for every legal action, and `legal_action_to_index` maps each
//! to its flat policy slot. So to decode an index we enumerate, find the
//! `LegalAction` whose index matches, and translate it into an `Action`. This
//! reuses the forward index function (no second copy of the layout inverse) and
//! all the state-resolution the enumerator already performs.
//!
//! The decoded `Action` need not be field-identical to Python's — Python is no
//! longer in the loop. It must be BEHAVIORALLY identical: feeding it to
//! `process_action_internal` must reproduce the state the old
//! (Python-decode → dict → `from_py_dict`) path produced. Behavioral parity is
//! verified against that old path in `tests/decode_parity_corpus.py`.

use crate::actions::{Action, DividendKind, GameError, RouteData};
use crate::factored::LegalAction;
use crate::game::BaseGame;
use pyo3::prelude::*;
use serde_json::json;

/// Read a string field from a JSON map.
fn s(map: &std::collections::HashMap<String, serde_json::Value>, key: &str) -> Option<String> {
    map.get(key).and_then(|v| v.as_str()).map(|s| s.to_string())
}

/// Read an i64 field from a JSON map.
fn i(map: &std::collections::HashMap<String, serde_json::Value>, key: &str) -> Option<i64> {
    map.get(key).and_then(|v| v.as_i64())
}

impl BaseGame {
    /// The actor id string for a non-company action: the current entity as
    /// `Action.entity_id()` expects it (player → numeric id string; corp → sym).
    fn current_actor_id(&self) -> Result<String, GameError> {
        let e = &self.round_state.active_entity_id;
        if let Some(pid) = e.player_id() {
            return Ok(pid.to_string());
        }
        if let Some(sym) = e.corp_sym() {
            return Ok(sym.to_string());
        }
        Err(GameError::new(format!(
            "current entity {:?} is neither player nor corporation",
            e
        )))
    }

    /// Find the deduplicated factored [`LegalAction`] whose flat policy index
    /// matches `idx` at the current state, re-running the factored enumeration.
    /// Shared by `decode_index`, `price_range_for_index`, and
    /// `price_head_slot_for_index` (after dedup there is at most one entry per
    /// index, so the first match is the only match).
    pub(crate) fn legal_action_for_index(&mut self, idx: u32) -> Option<LegalAction> {
        self.get_factored_choices_impl()
            .into_iter()
            .find(|c| crate::action_index::legal_action_to_index(c) == Some(idx))
    }

    /// Decode a flat policy index into a concrete [`Action`] at the current
    /// state. `sampled_price` supplies the price for continuous-price slots
    /// (Bid / BuyCompany / cross-corp BuyTrain); ignored for fixed-price slots.
    pub(crate) fn decode_index(
        &mut self,
        idx: u32,
        sampled_price: Option<i64>,
    ) -> Result<Action, GameError> {
        let total = crate::action_index::POLICY_SIZE;
        if idx >= total {
            return Err(GameError::new(format!(
                "Action index {} out of bounds (0-{})",
                idx,
                total - 1
            )));
        }

        // Find the factored LegalAction whose flat index matches. (After dedup
        // there is at most one categorical entry per index; price-bearing slots
        // collapse the price dimension to a single entry carrying a range.)
        let la = self.legal_action_for_index(idx).ok_or_else(|| {
            GameError::new(format!("index {} is not legal at the current state", idx))
        })?;

        let company_offset = crate::action_index::layout().action_offsets["CompanyBuyShares"];
        let is_company = idx >= company_offset;

        self.build_action(&la, idx, is_company, sampled_price)
    }

    /// Process a natively-decoded action AND append a faithful, replayable JSON
    /// dict to the action log, so the driver's `raw_actions` (which
    /// `extract_data` replays) stay intact — but without any Python
    /// `ActionMapper` round-trip. The apply itself uses the
    /// decode-parity-verified `process_action_internal` path directly; the JSON
    /// is for logging only (never re-parsed for this apply).
    pub(crate) fn process_action_native(&mut self, action: &Action) -> Result<(), GameError> {
        let logged = self.action_to_json(action);
        self.process_action_internal(action)?;
        self.action_log.push(logged);
        Ok(())
    }

    /// (entity JSON value, entity_type) for an action's entity id — player ids
    /// become JSON numbers, corp/company syms strings (matching what the action
    /// dicts recorded by the cleaning pipeline look like, so replay works).
    fn entity_json(&self, eid: &str) -> (serde_json::Value, &'static str) {
        if let Ok(pid) = eid.parse::<u32>() {
            (json!(pid), "player")
        } else if self.corp_idx.contains_key(eid) {
            (json!(eid), "corporation")
        } else {
            (json!(eid), "company")
        }
    }

    /// Build a faithful, replayable action dict (as JSON) from a decoded action.
    /// Field shapes match what `Action::from_py_dict` parses, so re-processing
    /// the logged dict reproduces the same action/state.
    fn action_to_json(&self, action: &Action) -> serde_json::Value {
        let (entity, entity_type) = self.entity_json(action.entity_id());
        let mut m = serde_json::Map::new();
        m.insert("type".to_string(), json!(action.action_type()));
        m.insert("entity".to_string(), entity);
        m.insert("entity_type".to_string(), json!(entity_type));
        match action {
            Action::Pass { .. } | Action::Bankrupt { .. } => {}
            Action::Bid { company_sym, price, .. } => {
                m.insert("company".to_string(), json!(company_sym));
                m.insert("price".to_string(), json!(price));
            }
            Action::Par { corporation_sym, share_price, .. } => {
                m.insert("corporation".to_string(), json!(corporation_sym));
                m.insert("share_price".to_string(), json!(share_price));
            }
            Action::BuyShares { corporation_sym, percent, source, share_indices, .. } => {
                m.insert("corporation".to_string(), json!(corporation_sym));
                m.insert("percent".to_string(), json!(percent));
                m.insert("source".to_string(), json!(source));
                if !share_indices.is_empty() {
                    let shares: Vec<String> = share_indices
                        .iter()
                        .map(|i| format!("{}_{}", corporation_sym, i))
                        .collect();
                    m.insert("shares".to_string(), json!(shares));
                }
            }
            Action::SellShares { corporation_sym, percent, share_indices, .. } => {
                m.insert("corporation".to_string(), json!(corporation_sym));
                m.insert("percent".to_string(), json!(percent));
                if !share_indices.is_empty() {
                    let shares: Vec<String> = share_indices
                        .iter()
                        .map(|i| format!("{}_{}", corporation_sym, i))
                        .collect();
                    m.insert("shares".to_string(), json!(shares));
                }
            }
            Action::LayTile { hex_id, tile_id, rotation, .. } => {
                m.insert("hex".to_string(), json!(hex_id));
                m.insert("tile".to_string(), json!(tile_id));
                m.insert("rotation".to_string(), json!(rotation));
            }
            Action::PlaceToken { hex_id, city_index, .. } => {
                m.insert("hex".to_string(), json!(hex_id));
                m.insert("city_index".to_string(), json!(city_index));
            }
            Action::RunRoutes { routes, extra_revenue, .. } => {
                let rlist: Vec<serde_json::Value> = routes
                    .iter()
                    .map(|r| json!({"train": r.train_name, "revenue": r.revenue, "hexes": r.hexes}))
                    .collect();
                m.insert("routes".to_string(), json!(rlist));
                m.insert("extra_revenue".to_string(), json!(extra_revenue));
            }
            Action::Dividend { kind, .. } => {
                m.insert("kind".to_string(), json!(kind.as_str()));
            }
            Action::BuyTrain { train_name, price, from, variant, exchange, .. } => {
                m.insert("train".to_string(), json!(train_name));
                m.insert("price".to_string(), json!(price));
                m.insert("from".to_string(), json!(from));
                if let Some(v) = variant {
                    m.insert("variant".to_string(), json!(v));
                }
                if let Some(e) = exchange {
                    m.insert("exchange".to_string(), json!(e));
                }
            }
            Action::DiscardTrain { train_name, .. } => {
                m.insert("train".to_string(), json!(train_name));
            }
            Action::BuyCompany { company_sym, price, .. } => {
                m.insert("company".to_string(), json!(company_sym));
                m.insert("price".to_string(), json!(price));
            }
        }
        serde_json::Value::Object(m)
    }

    fn build_action(
        &mut self,
        la: &LegalAction,
        idx: u32,
        is_company: bool,
        sampled_price: Option<i64>,
    ) -> Result<Action, GameError> {
        let t = la.action_type.as_str();
        match t {
            "Pass" => Ok(Action::Pass {
                entity_id: self.current_actor_id()?,
            }),

            "Bid" => {
                // entity = the bidding player; company = bid target (entity.private).
                let entity_id = self.current_actor_id()?;
                let company_sym = s(&la.entity, "private")
                    .ok_or_else(|| GameError::new("Bid LegalAction missing 'private'"))?;
                let price = sampled_price
                    .or_else(|| la.price_range.map(|(lo, _)| lo))
                    .ok_or_else(|| GameError::new("Bid has no price"))?;
                Ok(Action::Bid {
                    entity_id,
                    company_sym,
                    price: price as i32,
                })
            }

            "Par" => {
                let entity_id = self.current_actor_id()?;
                let corporation_sym = s(&la.entity, "corp")
                    .ok_or_else(|| GameError::new("Par LegalAction missing 'corp'"))?;
                let share_price = i(&la.params, "par_price")
                    .ok_or_else(|| GameError::new("Par LegalAction missing 'par_price'"))?;
                Ok(Action::Par {
                    entity_id,
                    corporation_sym,
                    share_price: share_price as i32,
                })
            }

            "BuyShares" | "CompanyBuyShares" => {
                let corporation_sym = s(&la.entity, "corp")
                    .ok_or_else(|| GameError::new("BuyShares LegalAction missing 'corp'"))?;
                let source = s(&la.params, "source").unwrap_or_else(|| "auto".to_string());
                let percent = i(&la.params, "percent").unwrap_or(10) as u8;
                let entity_id = if is_company {
                    s(&la.entity, "private").ok_or_else(|| {
                        GameError::new("company BuyShares LegalAction missing 'private'")
                    })?
                } else {
                    self.current_actor_id()?
                };
                // Regular BuyShares: leave share_indices empty — the buy handler
                // resolves the lowest-index non-president cert in the pool, which
                // equals Python's `pool[0]`. For the COMPANY exchange (MH→NYC),
                // the handler's empty-indices fallback always tries IPO first and
                // ignores `source`, so a `source="market"` exchange would grab an
                // IPO cert. Resolve the exact cert matching the source here, the
                // way Python's `exchangeable_shares(...)[owner][0]` does.
                let share_indices = if is_company {
                    use crate::entities::EntityId;
                    let target = match source.as_str() {
                        "ipo" => EntityId::ipo(&corporation_sym),
                        "market" => EntityId::market(),
                        _ => EntityId::ipo(&corporation_sym),
                    };
                    let ci = *self.corp_idx.get(corporation_sym.as_str()).ok_or_else(|| {
                        GameError::new(format!("exchange: unknown corp {}", corporation_sym))
                    })?;
                    let found = self.corporations[ci]
                        .shares
                        .iter()
                        .position(|sh| !sh.president && sh.percent == percent && sh.owner == target);
                    match found {
                        Some(idx) => vec![idx],
                        None => Vec::new(),
                    }
                } else {
                    Vec::new()
                };
                Ok(Action::BuyShares {
                    entity_id,
                    corporation_sym,
                    shares: Vec::new(),
                    percent,
                    source,
                    share_indices,
                })
            }

            "SellShares" => {
                // The seller is the active player in a Stock round, but in an
                // Operating round the active entity is the operating
                // corporation and the seller is its PRESIDENT (emergency money
                // — round.py / factored_sell_shares). Mirror the enumerator's
                // seller resolution so the decoded entity is the president's
                // numeric id that `or_emergency_sell` expects; `current_actor_id()`
                // would yield the corp sym here and trip "Invalid player id".
                let entity_id = match self.round_state.active_entity_id.player_id() {
                    Some(pid) => pid.to_string(),
                    None => {
                        let sym = self
                            .round_state
                            .active_entity_id
                            .corp_sym()
                            .ok_or_else(|| {
                                GameError::new(
                                    "SellShares: active entity is neither player nor corporation",
                                )
                            })?;
                        let ci = *self.corp_idx.get(sym).ok_or_else(|| {
                            GameError::new(format!("SellShares: unknown corp {}", sym))
                        })?;
                        self.corporations[ci]
                            .president_id()
                            .ok_or_else(|| {
                                GameError::new(format!(
                                    "SellShares: corp {} has no president",
                                    sym
                                ))
                            })?
                            .to_string()
                    }
                };
                let corporation_sym = s(&la.entity, "corp")
                    .ok_or_else(|| GameError::new("SellShares LegalAction missing 'corp'"))?;
                // The factored sell slot encodes the share count under "count"
                // (see factored_sell_shares); fall back to percent/10 for any
                // producer that only carries the percent.
                let num = i(&la.params, "count")
                    .or_else(|| i(&la.params, "percent").map(|p| p / 10))
                    .ok_or_else(|| {
                        GameError::new("SellShares LegalAction missing 'count'/'percent'")
                    })?;
                Ok(Action::SellShares {
                    entity_id,
                    corporation_sym,
                    shares: Vec::new(),
                    percent: (num * 10) as u8,
                    share_indices: Vec::new(),
                })
            }

            "PlaceToken" => {
                let entity_id = if is_company {
                    s(&la.entity, "private").ok_or_else(|| {
                        GameError::new("company PlaceToken LegalAction missing 'private'")
                    })?
                } else {
                    self.current_actor_id()?
                };
                let hex_id = s(&la.params, "hex")
                    .ok_or_else(|| GameError::new("PlaceToken LegalAction missing 'hex'"))?;
                let city_index = i(&la.params, "city").unwrap_or(0) as u8;
                Ok(Action::PlaceToken {
                    entity_id,
                    hex_id,
                    city_index,
                })
            }

            "LayTile" => {
                let entity_id = if is_company {
                    s(&la.entity, "private").ok_or_else(|| {
                        GameError::new("company LayTile LegalAction missing 'private'")
                    })?
                } else {
                    self.current_actor_id()?
                };
                let hex_id = s(&la.params, "hex")
                    .ok_or_else(|| GameError::new("LayTile LegalAction missing 'hex'"))?;
                let tile_id = s(&la.params, "tile")
                    .ok_or_else(|| GameError::new("LayTile LegalAction missing 'tile'"))?;
                let rotation = i(&la.params, "rotation").unwrap_or(0) as u8;
                Ok(Action::LayTile {
                    entity_id,
                    hex_id,
                    tile_id,
                    rotation,
                })
            }

            "BuyTrain" => {
                let entity_id = self.current_actor_id()?;
                let name = s(&la.entity, "train")
                    .ok_or_else(|| GameError::new("BuyTrain LegalAction missing 'train'"))?;
                let source = s(&la.entity, "source").unwrap_or_else(|| "depot".to_string());
                let exchange_name = s(&la.entity, "exchange"); // donor NAME (trade-in)
                let is_depot = source == "depot" || source == "discard";

                // The buy_train handler validates by EXACT train instance id
                // (it builds the legal set from depot/other-corp train `.id`s).
                // The factored LegalAction only carries the train NAME, so resolve
                // it to the concrete buyable instance, mirroring Python's
                // map_index_to_action (which returns the train object → its id).
                let (train_id, instance_price) = if !is_depot {
                    // Cross-corp: the named train on the selling corporation.
                    let ci = *self.corp_idx.get(source.as_str()).ok_or_else(|| {
                        GameError::new(format!("BuyTrain: unknown seller corp {}", source))
                    })?;
                    let tr = self.corporations[ci]
                        .trains
                        .iter()
                        .find(|t| t.name == name)
                        .ok_or_else(|| {
                            GameError::new(format!("BuyTrain: {} has no train {}", source, name))
                        })?;
                    (tr.id.clone(), tr.price)
                } else if la.depot_discarded {
                    // Discarded (face-value) pool train of this name.
                    let tr = self
                        .depot
                        .discarded
                        .iter()
                        .find(|t| t.name == name)
                        .ok_or_else(|| {
                            GameError::new(format!("BuyTrain: no discarded {} train", name))
                        })?;
                    (tr.id.clone(), tr.price)
                } else {
                    // Fresh depot train: first upcoming instance of this name
                    // (head-of-queue / phase-visible), matching Python's
                    // depot.depot_trains() first non-discarded entry.
                    let tr = self
                        .depot
                        .trains
                        .iter()
                        .find(|t| t.name == name)
                        .ok_or_else(|| {
                            GameError::new(format!("BuyTrain: no depot {} train", name))
                        })?;
                    (tr.id.clone(), tr.price)
                };

                let from = if is_depot { "depot".to_string() } else { source.clone() };

                // Trade-in: several donor variants (exchange="4"/"5"/"6") collapse
                // onto the single BuyTrainDTradeIn slot, so the matched
                // LegalAction's specific donor is arbitrary. Mirror Python's
                // map_index_to_action, which AUTO-PICKS the lowest-tier owned
                // donor (4, then 5, then 6) — all yield the same $300 discount in
                // 1830, so the price is unchanged. Resolve to that donor's
                // instance id.
                let exchange_id = if exchange_name.is_some() {
                    let ci = *self.corp_idx.get(entity_id.as_str()).ok_or_else(|| {
                        GameError::new(format!("BuyTrain: unknown buyer corp {}", entity_id))
                    })?;
                    let donor_id = ["4", "5", "6"].iter().find_map(|dn| {
                        self.corporations[ci]
                            .trains
                            .iter()
                            .find(|t| t.name == *dn)
                            .map(|t| t.id.clone())
                    });
                    Some(donor_id.ok_or_else(|| {
                        GameError::new(format!(
                            "BuyTrain trade-in: buyer {} owns no 4/5/6 donor",
                            entity_id
                        ))
                    })?)
                } else {
                    None
                };

                // Price: trade-in / cross-corp take the slot's range (or sampled);
                // a plain depot buy uses the instance's fixed price.
                let price = if exchange_id.is_some() {
                    la.price_range
                        .map(|(lo, _)| lo)
                        .unwrap_or(instance_price as i64)
                } else if is_depot {
                    instance_price as i64
                } else {
                    sampled_price
                        .or_else(|| la.price_range.map(|(lo, _)| lo))
                        .ok_or_else(|| GameError::new("cross-corp BuyTrain has no price"))?
                };

                Ok(Action::BuyTrain {
                    entity_id,
                    train_name: train_id,
                    price: price as i32,
                    from,
                    variant: None,
                    exchange: exchange_id,
                })
            }

            "DiscardTrain" => {
                let entity_id = self.current_actor_id()?;
                let train_name = s(&la.entity, "train")
                    .or_else(|| s(&la.params, "train"))
                    .ok_or_else(|| GameError::new("DiscardTrain LegalAction missing 'train'"))?;
                Ok(Action::DiscardTrain {
                    entity_id,
                    train_name,
                })
            }

            "Dividend" => {
                let entity_id = self.current_actor_id()?;
                let kind_str = s(&la.params, "kind")
                    .ok_or_else(|| GameError::new("Dividend LegalAction missing 'kind'"))?;
                Ok(Action::Dividend {
                    entity_id,
                    kind: DividendKind::parse(&kind_str)?,
                })
            }

            "BuyCompany" => {
                let entity_id = self.current_actor_id()?;
                let company_sym = s(&la.entity, "private")
                    .ok_or_else(|| GameError::new("BuyCompany LegalAction missing 'private'"))?;
                // Fixed min/max price slots carry their value in the range;
                // a sampled price (continuous) overrides.
                let price = sampled_price
                    .or_else(|| la.price_range.map(|(lo, _)| lo))
                    .ok_or_else(|| GameError::new("BuyCompany has no price"))?;
                Ok(Action::BuyCompany {
                    entity_id,
                    company_sym,
                    price: price as i32,
                })
            }

            "Bankrupt" => Ok(Action::Bankrupt {
                entity_id: self.current_actor_id()?,
            }),

            "RunRoutes" => {
                let corp_sym = self
                    .round_state
                    .active_entity_id
                    .corp_sym()
                    .ok_or_else(|| GameError::new("RunRoutes: current entity is not a corp"))?
                    .to_string();
                // Native optimal routing — replaces Python's
                // ActionHelper.auto_route_action. `or_process_run_routes` only
                // consumes the summed revenue, so a single RouteData carrying the
                // total revenue is behaviorally equivalent.
                let (_route_dicts, total_revenue) = self.calculate_routes(corp_sym.clone());
                let routes = vec![RouteData {
                    train_name: String::new(),
                    hexes: Vec::new(),
                    revenue: total_revenue,
                }];
                Ok(Action::RunRoutes {
                    entity_id: corp_sym,
                    routes,
                    extra_revenue: 0,
                })
            }

            other => Err(GameError::new(format!(
                "decode_index: unhandled LegalAction type {:?} at index {}",
                other, idx
            ))),
        }
    }
}

#[pymethods]
impl BaseGame {
    /// Decode a flat policy index into an action and apply it natively — the
    /// Rust replacement for `apply_action`'s Python `ActionMapper` round-trip.
    /// `price` supplies the sampled price for continuous-price slots.
    fn apply_action_index(&mut self, idx: u32, price: Option<i64>) -> PyResult<()> {
        let action = self.decode_index(idx, price).map_err(PyErr::from)?;
        self.process_action_native(&action).map_err(PyErr::from)?;
        Ok(())
    }

    /// Decode a flat policy index into an action's `to_map()` representation
    /// WITHOUT applying it. Used by the decode-parity harness to compare the
    /// native decode against Python's `map_index_to_action(...).to_dict()`.
    fn decode_index_to_map(
        &mut self,
        idx: u32,
        price: Option<i64>,
    ) -> PyResult<std::collections::HashMap<String, String>> {
        let action = self.decode_index(idx, price).map_err(PyErr::from)?;
        Ok(action.to_map())
    }

    /// The legal `(price_min, price_max)` range for a flat index, or `None` if
    /// the index is illegal or categorical (no price). Native replacement for
    /// reading `ActionMapper.get_legal_actions_factored(...)[1][idx]`.
    fn price_range_for_index(&mut self, idx: u32) -> Option<(i64, i64)> {
        self.legal_action_for_index(idx).and_then(|c| c.price_range)
    }

    /// Resolve the `ContinuousPriceHead` slot for a flat index:
    /// `(slot_index, price_min, price_max)`, or `None` for categorical /
    /// fixed-price slots that don't reach the head. Native replacement for
    /// `ActionMapper.price_head_slot_for_action`. The slot layout matches the
    /// Python head: [Bid×6 companies][BuyTrain×(8 corps·6 train types)]
    /// [BuyCompany×6 companies].
    fn price_head_slot_for_index(&mut self, idx: u32) -> Option<(u32, i64, i64)> {
        const COMPANIES: [&str; 6] = ["SV", "CS", "DH", "MH", "CA", "BO"];
        const CORPS: [&str; 8] = ["PRR", "NYC", "CPR", "B&O", "C&O", "ERIE", "NYNH", "B&M"];
        const TRAINS: [&str; 6] = ["2", "3", "4", "5", "6", "D"];
        let la = self.legal_action_for_index(idx)?;
        // Any price-bearing slot maps to the head; a degenerate (min == max)
        // range still yields a slot with price_min == price_max, matching
        // Python's `price_head_slot_for_action`. (The sole caller,
        // `_extract_price_targets`, short-circuits on min == max before reaching
        // here, so this only matters for direct / parity-audit callers.)
        let (min, max) = match la.price_range {
            Some((lo, hi)) => (lo, hi),
            None => return None,
        };
        let slot = match la.action_type.as_str() {
            "Bid" => {
                let c = la.entity.get("private").and_then(|v| v.as_str())?;
                COMPANIES.iter().position(|x| *x == c)? as u32
            }
            "BuyCompany" => {
                let c = la.entity.get("private").and_then(|v| v.as_str())?;
                let ci = COMPANIES.iter().position(|x| *x == c)?;
                (COMPANIES.len() + CORPS.len() * TRAINS.len() + ci) as u32
            }
            "BuyTrain" => {
                // Only cross-corp buys reach the head; depot/discard are fixed.
                let src = la.entity.get("source").and_then(|v| v.as_str())?;
                if src == "depot" || src == "discard" {
                    return None;
                }
                let ci = CORPS.iter().position(|x| *x == src)?;
                let train = la.entity.get("train").and_then(|v| v.as_str())?;
                let ti = TRAINS.iter().position(|x| *x == train)?;
                (COMPANIES.len() + ci * TRAINS.len() + ti) as u32
            }
            _ => return None,
        };
        Some((slot, min, max))
    }
}
