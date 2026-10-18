//! PyO3 surface for 1822-family games (inspection for the replay
//! harness / Ruby diff rig, and the legal-action generator).

use pyo3::prelude::*;

use crate::entities::EntityId;
use crate::game::BaseGame;

use super::{Ent, RoundKind};

#[pymethods]
impl BaseGame {
    /// A JSON snapshot of the 1822 state in the shape of
    /// scripts/ruby_dump.rb's per-action rows (round, step, entity, phase,
    /// corps' price/cash/trains, player cash by id, bank).
    pub(crate) fn g1822_snapshot(&self) -> PyResult<String> {
        let Some(st) = self.g1822.as_deref() else {
            return Err(pyo3::exceptions::PyValueError::new_err("not an 1822-family game"));
        };
        let r = &st.round;
        let round = match r.kind {
            RoundKind::Stock => "Stock",
            RoundKind::Choices => "Choices",
            RoundKind::Operating => "Operating",
            RoundKind::Merger => "Merger",
        };
        let active = self.g1822_active_step();
        let step = active.map(|i| format!("{:?}", r.steps[i]));
        let ent = active.and_then(|i| self.g1822_step_current_entity(i)).map(|e| match e {
            Ent::Player(p) => p.to_string(),
            Ent::Corp(c) | Ent::Company(c) => c,
        });
        let mut corps = serde_json::Map::new();
        for c in &self.corporations {
            if c.closed && c.ipo_price.is_none() {
                continue;
            }
            corps.insert(
                c.sym.clone(),
                serde_json::json!({
                    "p": c.share_price.as_ref().map(|s| s.price),
                    "cash": c.cash,
                    "fl": c.floated,
                    "closed": c.closed,
                    "tr": c.trains.iter().map(|t| t.name.clone()).collect::<Vec<_>>(),
                }),
            );
        }
        let pcash: serde_json::Map<String, serde_json::Value> = self
            .players
            .iter()
            .map(|p| (p.id.to_string(), serde_json::json!(p.cash)))
            .collect();
        let debt: serde_json::Map<String, serde_json::Value> = st
            .player_debt
            .iter()
            .map(|(p, d)| (p.to_string(), serde_json::json!(d)))
            .collect();
        let mut companies: serde_json::Map<String, serde_json::Value> = self
            .companies
            .iter()
            .filter(|c| !c.closed && !c.owner.is_none())
            .map(|c| (c.sym.clone(), serde_json::json!(c.owner.0)))
            .collect();
        for c in &st.bank_owned {
            if !self.co_closed(c) {
                companies.insert(c.clone(), serde_json::json!("The Bank"));
            }
        }
        // Ruby keeps ONE card of an unlimited type in `upcoming`; the
        // engine materializes the pool — show up to the first 7.
        let mut depot: Vec<String> = Vec::new();
        for t in self.depot.trains.iter().take(3) {
            depot.push(t.id.clone());
            if t.id.starts_with("7-") {
                break;
            }
        }
        let v = serde_json::json!({
            "round": round,
            "round_num": r.round_num,
            "turn": self.turn,
            "step": step,
            "ent": ent,
            "phase": self.phase.name,
            "corps": corps,
            "pcash": pcash,
            "debt": debt,
            "bank": self.bank.cash,
            "finished": self.finished,
            "companies": companies,
            "order": self.players.iter().map(|p| p.id).collect::<Vec<_>>(),
            "depot": depot,
            "game_end_reason": st.game_end_reason,
        });
        Ok(v.to_string())
    }

    /// The live company stack order (Ruby `@companies`) and bid boxes.
    fn g1822_companies_order(&self) -> Vec<String> {
        self.g1822.as_deref().map(|s| s.company_order.clone()).unwrap_or_default()
    }

    fn g1822_bidboxes(&self) -> (Vec<String>, Vec<String>, Vec<String>) {
        if self.g1822.is_none() {
            return (vec![], vec![], vec![]);
        }
        (self.bidbox_minors(), self.bidbox_concessions(), self.bidbox_privates())
    }

    /// The legal action types for the acting entity.
    fn g1822_legal_types(&self) -> Vec<String> {
        if self.g1822.is_none() {
            return vec![];
        }
        self.g1822_action_types()
    }

    /// The player who decides the current move (Ruby `active_players` /
    /// `acting_for_entity`): a player entity itself, a corporation's
    /// president, a company's owner's president; 1822MX's presidentless
    /// NdeM: its current tile layer / the auction's chooser, bidder or
    /// winner / else its first shareholder in priority (the priority player
    /// if none). None when nobody acts.
    fn g1822_acting_player(&self) -> Option<u32> {
        self.g1822.as_ref()?;
        let bi = self.g1822_active_step()?;
        match self.g1822_step_current_entity(bi)? {
            Ent::Player(p) => Some(p),
            Ent::Corp(c) if self.g1822_mx_is_ndem(&c) => self.g1822_mx_ndem_acting_player(),
            Ent::Corp(c) => self.g1822_corp_president(&c),
            Ent::Company(co) => match self.co_owner(&co).corp_sym() {
                Some(c) => self.g1822_corp_president(c),
                None => self.co_owner(&co).player_id(),
            },
        }
    }

    /// Player value (Ruby player_value) by player id.
    fn g1822_player_values(&self) -> Vec<(u32, i32)> {
        self.players.iter().map(|p| (p.id, self.g1822_player_value(p.id))).collect()
    }

    #[allow(dead_code)]
    fn g1822_owner_of(&self, company: &str) -> String {
        self.co_owner(company).0
    }

    /// The legal moves (see `g1822::legal`) as JSON: a list of
    /// `{"wire": <18xx.games action dict>, "range": [lo, hi, step] | null}`.
    /// A ranged move's dict holds `lo` in its price field (`price` for
    /// bids and train buys, `amount` for partial debt payoff).
    /// `policy`: None = the title's cross-player policy, "rules" = every
    /// move the rules allow (Ruby-faithful), "restricted" = none across
    /// players.
    #[pyo3(signature = (policy=None))]
    fn g1822_legal_moves_json(&mut self, policy: Option<&str>) -> PyResult<String> {
        if self.g1822.is_none() {
            return Err(pyo3::exceptions::PyValueError::new_err("not an 1822-family game"));
        }
        let policy = match policy {
            None => self.title_def().generator_cross_player(),
            Some("rules") => crate::title::CrossPlayerPolicy::RULES,
            Some("restricted") => crate::title::CrossPlayerPolicy::RESTRICTED,
            Some(p) => return Err(pyo3::exceptions::PyValueError::new_err(format!("unknown policy {p:?}"))),
        };
        let moves = self.g1822_legal_moves_with(policy);
        let out: Vec<serde_json::Value> = moves
            .iter()
            .map(|m| {
                serde_json::json!({
                    "wire": self.g1822_move_wire(m),
                    "range": m.price.map(|r| vec![r.lo, r.hi, r.step]),
                })
            })
            .collect();
        Ok(serde_json::Value::Array(out).to_string())
    }

    /// The auto-router's best run for `corp`: (total revenue incl. the
    /// destination bonus, routes as run_routes JSON).
    fn g1822_router_best(&self, corp: &str) -> (i32, String) {
        let (routes, total) = self.g1822_best_routes(corp);
        let rs: Vec<serde_json::Value> = routes
            .iter()
            .map(|r| {
                serde_json::json!({
                    "train": r.train_name, "hexes": r.hexes, "connections": r.connections,
                    "nodes": r.nodes, "revenue": r.revenue,
                })
            })
            .collect();
        (total, serde_json::Value::Array(rs).to_string())
    }

    /// The legacy (pre-2026-10-17) router's best total, for comparison.
    fn g1822_router_best_legacy(&self, corp: &str) -> i32 {
        self.g1822_best_routes_legacy(corp).1
    }

    /// Router v2 counters: (calls, walk steps, exhausted walks, TOP_K
    /// overflows) since the process started.
    #[staticmethod]
    fn g1822_router_stats() -> (u64, u64, u64, u64) {
        use super::router::stats;
        use std::sync::atomic::Ordering::Relaxed;
        (
            stats::CALLS.load(Relaxed),
            stats::WALK_STEPS.load(Relaxed),
            stats::EXHAUSTED.load(Relaxed),
            stats::OVERFLOW.load(Relaxed),
        )
    }

    /// Debug: explain recorded routes (JSON list of run_routes entries).
    fn g1822_router_explain_json(&self, corp: &str, routes_json: &str) -> Vec<String> {
        let v: Vec<serde_json::Value> = serde_json::from_str(routes_json).unwrap_or_default();
        let routes: Vec<crate::actions::RouteData> = v
            .iter()
            .map(|r| crate::actions::RouteData {
                train_name: r["train"].as_str().unwrap_or("").to_string(),
                hexes: serde_json::from_value(r["hexes"].clone()).unwrap_or_default(),
                revenue: r["revenue"].as_i64().map(|x| x as i32),
                connections: serde_json::from_value(r["connections"].clone()).unwrap_or_default(),
                nodes: serde_json::from_value(r["nodes"].clone()).unwrap_or_default(),
            })
            .collect();
        self.g1822_router_explain(corp, &routes)
    }

    /// Debug: the router's chains out of a node ("<hex>-<index>").
    fn g1822_router_segs_json(&self, corp: &str, node: &str) -> Vec<String> {
        self.g1822_router_segs_debug(corp, node)
    }

    /// Graph#compute for a corporation (debug): route_available, the
    /// connected nodes ("HEX-kind-index"), connected hexes with edges.
    fn g1822_graph_debug(&self, corp: &str) -> (bool, Vec<String>, Vec<(String, Vec<u8>)>) {
        let g = self.g1822_graph(corp);
        let mut nodes: Vec<String> = g
            .connected_nodes
            .iter()
            .map(|n| format!("{}-{:?}-{}", self.hexes[n.hex].id, n.kind, n.index))
            .collect();
        nodes.sort();
        let hexes = g
            .connected_hexes
            .iter()
            .map(|(h, e)| {
                let mut e = e.clone();
                e.sort();
                (self.hexes[*h].id.clone(), e)
            })
            .collect();
        (g.route_available, nodes, hexes)
    }

    /// Debug: a hex's Ruby `neighbors` (edge, hex) and all adjacency.
    fn g1822_neighbors_debug(&self, hex: &str) -> (Vec<(u8, String)>, Vec<(u8, String)>) {
        let Some(&hi) = self.hex_idx.get(hex) else { return (vec![], vec![]) };
        let n = self.g1822_neighbors();
        let a: Vec<(u8, String)> = n
            .get(&hi)
            .map(|m| m.iter().map(|(e, h)| (*e, self.hexes[*h].id.clone())).collect())
            .unwrap_or_default();
        let mut b: Vec<(u8, String)> = self
            .hex_adjacency
            .get(hex)
            .map(|m| m.iter().map(|(e, h)| (*e, h.clone())).collect())
            .unwrap_or_default();
        b.sort();
        (a, b)
    }

    /// Debug: a hex's tile, its cities' tokens (corp, type, cheater, serial)
    /// and extra tokens, the city and tile reservations (JSON).
    fn g1822_hex_debug(&self, hex: &str) -> String {
        let Some(&hi) = self.hex_idx.get(hex) else { return String::new() };
        let h = &self.hexes[hi];
        let tok = |t: &crate::entities::Token| {
            serde_json::json!([t.corporation_id, t.token_type, t.cheater, t.serial])
        };
        let cities: Vec<serde_json::Value> = h
            .tile
            .cities
            .iter()
            .enumerate()
            .map(|(ci, c)| {
                serde_json::json!({
                    "slots": c.slots,
                    "tokens": c.tokens.iter().map(|t| t.as_ref().map(tok)).collect::<Vec<_>>(),
                    "extra": c.extra_tokens.iter().map(tok).collect::<Vec<_>>(),
                    "res": self.g1822_city_res(hex, ci),
                })
            })
            .collect();
        serde_json::json!({
            "tile": h.tile.id,
            "rotation": h.tile.rotation,
            "cities": cities,
            "tile_res": self.g().tile_reservations.get(hex).cloned().unwrap_or_default(),
        })
        .to_string()
    }

    /// Debug: (upgrades_to, legal rotations) of a catalog tile on a hex for
    /// a corporation's own lay.
    fn g1822_rotations_debug(&self, corp: &str, hex: &str, tile: &str) -> (bool, Vec<u8>, Vec<u8>) {
        let Some(&hi) = self.hex_idx.get(hex) else { return (false, vec![], vec![]) };
        let Some(def) = self.tile_catalog.get(tile).cloned() else { return (false, vec![], vec![]) };
        let up = self.g1822_upgrades_to(hi, &def, false);
        let connected = self.g1822_graph(corp).connected_hexes.get(&hi).cloned().unwrap_or_default();
        let nbrs = self.g1822_neighbors();
        let rots = (0..6u8)
            .filter(|&r| self.g1822_legal_rotation(super::tile_rules::RotRule::Normal, corp, hi, &def, r, &connected, &nbrs))
            .collect();
        (up, rots, connected)
    }

    /// The holder of a share certificate ("LNWR_3"): "market", "ipo:LNWR",
    /// a player id, "taxhaven:P16" ... ("" when unknown).
    fn g1822_share_owner(&self, share_id: &str) -> String {
        let Some((sym, idx)) = share_id.rsplit_once('_') else { return String::new() };
        let Ok(idx) = idx.parse::<usize>() else { return String::new() };
        let Some(&ci) = self.corp_idx.get(sym) else { return String::new() };
        self.corporations[ci].shares.get(idx).map(|s| s.owner.0.clone()).unwrap_or_default()
    }

    /// Resolve a wire `city` ("<tile id>-<city>", e.g. "X4-0-0", "P43-0-0")
    /// to (hex, city index).
    fn g1822_city_ref(&self, city: &str) -> (String, u8) {
        let parts: Vec<&str> = city.split('-').collect();
        if parts.len() < 3 {
            return (String::new(), 0);
        }
        let inst = format!("{}-{}", parts[0], parts[1]);
        let ci: u8 = parts[2].parse().unwrap_or(0);
        let hex = self.g1822_resolve_hex(&format!("__tile:{inst}")).unwrap_or_default();
        (hex, ci)
    }

    /// (name, owner) of a train id ("3-1"); owner "depot" for the depot.
    fn g1822_train_info(&self, train_id: &str) -> (String, String) {
        for c in &self.corporations {
            if let Some(t) = c.trains.iter().find(|t| t.id == train_id) {
                return (t.name.clone(), c.sym.clone());
            }
        }
        for t in self.depot.trains.iter().chain(self.depot.discarded.iter()) {
            if t.id == train_id {
                return (t.name.clone(), "depot".to_string());
            }
        }
        (String::new(), String::new())
    }

    /// Debug: the generator's candidates before the apply filter, each
    /// with the error applying it raises (None = applies).
    fn g1822_candidates(&mut self) -> Vec<(String, Option<String>)> {
        if self.g1822.is_none() {
            return vec![];
        }
        self.g1822_candidates_debug()
    }

    /// Whether a state with no legal moves is one Ruby cannot continue
    /// either (see g1822::fuzz::ruby_dead_end).
    fn g1822_ruby_dead_end(&mut self) -> bool {
        if self.g1822.is_none() {
            return false;
        }
        let cands = self.g1822_candidates_debug();
        let raises = cands.iter().any(|(_, e)| e.as_deref().map_or(false, |e| e.contains("(Ruby raises)")));
        let broke_token = !cands.is_empty()
            && cands
                .iter()
                .all(|(a, e)| a.starts_with("PlaceToken") && e.as_deref().map_or(false, |e| e.contains("cannot spend")));
        raises || broke_token
    }

    /// `(step, action type)` → "generated" or "replay-only: <reason>" for
    /// every pair an 1822 step can offer (the RL-readiness table).
    #[staticmethod]
    fn g1822_generator_table() -> Vec<(String, String, String)> {
        use super::legal::{g1822_generator_status, GenStatus};
        use super::Step::*;
        let pairs: &[(super::Step, &[&str])] = &[
            // (1822PNW: BSPS `choose` = the conversion's exchange,
            // AcquireCompany `choose` = P20's minor, Assign, Merge)
            (BuySellParShares, &["bid", "buy_shares", "sell_shares", "par", "pass", "payoff_player_debt", "payoff_player_debt_partial", "choose_ability", "choose"]),
            (Choose, &["choose", "pass"]),
            (DiscardTrain, &["discard_train"]),
            (PendingToken, &["place_token"]),
            (FirstTurnHousekeeping, &["buy_train", "pass"]),
            (AcquireCompany, &["acquire_company", "pass", "choose"]),
            (Assign, &["assign"]),
            (Merge, &["merge", "choose", "pass"]),
            // (1822MX: P16's close_p16, the NdeM's cash-out and auction)
            (SpecialChoose, &["choose", "choose_ability"]),
            (CashOutNdem, &["pass"]),
            (AuctionNdemTokens, &["remove_token", "bid", "merge", "pass"]),
            (SpecialTrack, &["lay_tile", "pass"]),
            (SpecialToken, &["place_token", "pass"]),
            (Track, &["lay_tile", "pass", "choose_ability"]),
            (DestinationToken, &["hex_token", "pass"]),
            (Token, &["place_token", "pass", "choose_ability"]),
            (Route, &["run_routes", "choose"]),
            (Dividend, &["dividend", "choose"]),
            (BuyTrain, &["buy_train", "sell_shares", "pass"]),
            (MinorAcquisition, &["merge", "choose", "pass"]),
            (IssueShares, &["buy_shares", "sell_shares", "pass", "choose_ability"]),
        ];
        let mut out = Vec::new();
        for (step, types) in pairs {
            for t in *types {
                let s = match g1822_generator_status(*step, t) {
                    GenStatus::Generated => "generated".to_string(),
                    GenStatus::ReplayOnly(why) => format!("replay-only: {why}"),
                };
                out.push((format!("{step:?}"), t.to_string(), s));
            }
        }
        out
    }
}

#[allow(dead_code)]
fn _unused(_: EntityId) {}
