//! GNN encoder: converts BaseGame state into flat feature vectors.
//!
//! Returns raw `Vec<f32>` arrays — Python wraps them in torch tensors.
//! This eliminates the ~2ms proxy overhead of the Python encoder on the Rust adapter.

use crate::entities::EntityId;
use crate::game::BaseGame;
use crate::tiles::PathEndpoint;

// ---------------------------------------------------------------------------
// Static ordered lists (must match Python encoder exactly)
// ---------------------------------------------------------------------------

pub const CORPORATION_IDS: &[&str] = &["PRR", "NYC", "CPR", "B&O", "C&O", "ERIE", "NYNH", "B&M"];
pub const PRIVATE_IDS: &[&str] = &["SV", "CS", "DH", "MH", "CA", "BO"];
pub const TRAIN_TYPES: &[&str] = &["2", "3", "4", "5", "6", "D"];
pub const PHASE_NAMES: &[&str] = &["2", "3", "4", "5", "6", "D"];
pub const TILE_IDS: &[&str] = &[
    "1", "2", "3", "4", "7", "8", "9", "14", "15", "16", "18", "19", "20", "23", "24", "25",
    "26", "27", "28", "29", "39", "40", "41", "42", "43", "44", "45", "46", "47", "53", "54",
    "55", "56", "57", "58", "59", "61", "62", "63", "64", "65", "66", "67", "68", "69", "70",
];
pub const HEX_COORDS: &[&str] = &[
    "A11", "A17", "A19", "A9", "B10", "B12", "B14", "B16", "B18", "B20", "B22", "B24", "C11",
    "C13", "C15", "C17", "C19", "C21", "C23", "C7", "C9", "D10", "D12", "D14", "D16", "D18",
    "D2", "D20", "D22", "D24", "D4", "D6", "D8", "E11", "E13", "E15", "E17", "E19", "E21",
    "E23", "E3", "E5", "E7", "E9", "F10", "F12", "F14", "F16", "F18", "F2", "F20", "F22",
    "F24", "F4", "F6", "F8", "G11", "G13", "G15", "G17", "G19", "G3", "G5", "G7", "G9", "H10",
    "H12", "H14", "H16", "H18", "H2", "H4", "H6", "H8", "I1", "I11", "I13", "I15", "I17",
    "I19", "I3", "I5", "I7", "I9", "J10", "J12", "J14", "J2", "J4", "J6", "J8", "K13", "K15",
];

pub const TRAIN_COUNTS: &[u32] = &[6, 5, 4, 3, 2, 20]; // 2, 3, 4, 5, 6, D

pub const NUM_CORPORATIONS: usize = CORPORATION_IDS.len();
pub const NUM_PRIVATES: usize = PRIVATE_IDS.len();
pub const NUM_TRAIN_TYPES: usize = TRAIN_TYPES.len();
pub const NUM_HEXES: usize = HEX_COORDS.len();
pub const NUM_TILE_IDS: usize = TILE_IDS.len();
pub const NUM_PHASES: usize = PHASE_NAMES.len();
pub const NUM_TILE_EDGES: usize = 6;
pub const NUM_PORT_PAIRS: usize = 15;

pub const BANK_CASH: f32 = 12000.0;
pub const MAX_SHARE_PRICE: f32 = 350.0;
pub const MAX_PRIVATE_REVENUE: f32 = 30.0;
pub const MAX_HEX_REVENUE: f32 = 80.0;
pub const MAX_LAY_COST: f32 = 120.0;

pub const CERT_LIMIT: &[(u8, u8)] = &[(2, 28), (3, 20), (4, 16), (5, 13), (6, 11)];
pub const STARTING_CASH: &[(u8, i32)] = &[(2, 1200), (3, 800), (4, 600), (5, 480), (6, 400)];

pub const NUM_NODE_FEATURES: usize =
    1 + 4 + 1 + 1 + NUM_CORPORATIONS * 2 + NUM_PORT_PAIRS + NUM_TILE_EDGES * 2;

const MAX_ROUND_TYPE: f32 = 2.0;

// Tile initial counts (TILE_IDS order):
// 1:1, 2:1, 3:2, 4:2, 7:4, 8:8, 9:7, 14:3, 15:2, 16:1, 18:1, 19:1, 20:1, 23:3, 24:3, 25:1,
// 26:1, 27:1, 28:1, 29:1, 39:1, 40:1, 41:2, 42:2, 43:2, 44:1, 45:2, 46:2, 47:1,
// 53:2, 54:1, 55:1, 56:1, 57:4, 58:2, 59:2, 61:2, 62:1, 63:3, 64:1, 65:1, 66:1, 67:1, 68:1, 69:1, 70:1
const TILE_INITIAL_COUNTS: &[u32] = &[
    1, 1, 2, 2, 4, 8, 7, 3, 2, 1, 1, 1, 1, 3, 3, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 2, 2, 1,
    2, 1, 1, 1, 4, 2, 2, 2, 1, 3, 1, 1, 1, 1, 1, 1, 1,
];

// ---------------------------------------------------------------------------
// Lookup helpers
// ---------------------------------------------------------------------------

fn corp_idx(sym: &str) -> Option<usize> {
    CORPORATION_IDS.iter().position(|&s| s == sym)
}
fn private_idx(sym: &str) -> Option<usize> {
    PRIVATE_IDS.iter().position(|&s| s == sym)
}
fn train_idx(name: &str) -> Option<usize> {
    TRAIN_TYPES.iter().position(|&s| s == name)
}
fn hex_idx(coord: &str) -> Option<usize> {
    HEX_COORDS.iter().position(|&s| s == coord)
}
fn phase_idx(name: &str) -> Option<usize> {
    PHASE_NAMES.iter().position(|&s| s == name)
}
fn cert_limit_for(n: usize) -> u8 {
    CERT_LIMIT.iter().find(|(k, _)| *k as usize == n).map(|(_, v)| *v).unwrap_or(16)
}
fn starting_cash_for(n: usize) -> i32 {
    STARTING_CASH.iter().find(|(k, _)| *k as usize == n).map(|(_, v)| *v).unwrap_or(600)
}

pub fn encoding_size(num_players: usize) -> usize {
    let np = num_players;
    let nc = NUM_CORPORATIONS;
    let npv = NUM_PRIVATES;
    let nt = NUM_TRAIN_TYPES;
    let ntile = NUM_TILE_IDS;

    (np + nc) + np + 1 + 1 + np + 1 + np + np + np * nc + npv * (np + nc) + npv
        + nc + nc + nc * nt + nc + nc * 2 + 2 * nc + 4 * nc + nt + nt + ntile
        + npv * np + npv + npv + npv + 2 + 1 + npv + np
}

// ---------------------------------------------------------------------------
// Game state encoder
// ---------------------------------------------------------------------------

impl BaseGame {
    /// Encode the full game state as flat f32 vectors.
    pub fn encode_state(&self) -> (Vec<f32>, Vec<f32>) {
        (self.encode_game_state(), self.encode_node_features())
    }

    fn encode_game_state(&self) -> Vec<f32> {
        let np = self.players.len();
        let size = encoding_size(np);
        let mut enc = vec![0.0f32; size];
        let mut off = 0;
        let sc = starting_cash_for(np) as f32;
        let cl = cert_limit_for(np) as f32;

        // Player id -> index (sorted by id)
        let mut pids: Vec<u32> = self.players.iter().map(|p| p.id).collect();
        pids.sort();
        let pidx = |id: u32| pids.iter().position(|&x| x == id);

        // Helper: encode an entity id into active_entity one-hot
        let encode_eid = |enc: &mut Vec<f32>, off: usize, eid: &EntityId| {
            if let Some(pid) = eid.player_id() {
                if let Some(i) = pidx(pid) { enc[off + i] = 1.0; }
            } else if let Some(sym) = eid.corp_sym() {
                if let Some(i) = corp_idx(sym) { enc[off + np + i] = 1.0; }
            }
        };

        // S1: Active Entity
        if !self.finished {
            let eid = &self.round_state.active_entity_id;
            if eid.is_player() || eid.0.starts_with("corp:") {
                encode_eid(&mut enc, off, eid);
            } else if eid.0.starts_with("company:") {
                // Company active: encode owner
                let sym = eid.0.strip_prefix("company:").unwrap_or("");
                if let Some(&ci) = self.company_idx.get(sym) {
                    encode_eid(&mut enc, off, &self.companies[ci].owner);
                }
            }
        }
        off += np + NUM_CORPORATIONS;

        // S2: Active President
        if !self.finished && self.round_state.active_entity_id.0.starts_with("corp:") {
            let sym = self.round_state.active_entity_id.corp_sym().unwrap_or("");
            if let Some(&ci) = self.corp_idx.get(sym) {
                if let Some(pid) = self.corporations[ci].president_id() {
                    if let Some(i) = pidx(pid) { enc[off + i] = 1.0; }
                }
            }
        }
        off += np;

        // S3: Round Type
        let rv = match &self.round {
            crate::rounds::Round::Auction(_) => 2.0,
            crate::rounds::Round::Stock(_) => 0.0,
            crate::rounds::Round::Operating(_) => 1.0,
        };
        enc[off] = rv / MAX_ROUND_TYPE;
        off += 1;

        // S4: Game Phase
        if let Some(i) = phase_idx(&self.phase.name) {
            enc[off] = i as f32 / (NUM_PHASES as f32 - 1.0);
        }
        off += 1;

        // S5: Priority Deal Player (use round-aware computation)
        let pd_id = self.effective_priority_deal_player();
        if let Some(i) = pidx(pd_id) {
            enc[off + i] = 1.0;
        }
        off += np;

        // S6: Bank Cash
        enc[off] = self.bank.cash as f32 / BANK_CASH;
        off += 1;

        // S7: Certs Remaining
        for p in &self.players {
            if let Some(i) = pidx(p.id) {
                let held = self.num_certs_for_player(p.id);
                enc[off + i] = (cl - held as f32) / cl;
            }
        }
        off += np;

        // S8: Player Cash
        for p in &self.players {
            if let Some(i) = pidx(p.id) {
                enc[off + i] = p.cash as f32 / sc;
            }
        }
        off += np;

        // S9: Player Share Ownership
        for (ci_raw, corp) in self.corporations.iter().enumerate() {
            if let Some(ci) = corp_idx(&corp.sym) {
                for p in &self.players {
                    if let Some(pi) = pidx(p.id) {
                        let pct = self.player_percent_of(p.id, ci_raw);
                        enc[off + pi * NUM_CORPORATIONS + ci] = pct as f32 / 100.0;
                    }
                }
            }
        }
        off += np * NUM_CORPORATIONS;

        // S10: Private Ownership (skip closed companies — matches Python adapter)
        let os = np + NUM_CORPORATIONS;
        for co in &self.companies {
            if co.closed { continue; }
            if let Some(pi) = private_idx(&co.sym) {
                if let Some(pid) = co.owner.player_id() {
                    if let Some(i) = pidx(pid) { enc[off + pi * os + i] = 1.0; }
                } else if let Some(sym) = co.owner.corp_sym() {
                    if let Some(i) = corp_idx(sym) { enc[off + pi * os + np + i] = 1.0; }
                }
            }
        }
        off += NUM_PRIVATES * os;

        // S11: Private Revenue
        for co in &self.companies {
            if let Some(i) = private_idx(&co.sym) {
                enc[off + i] = co.revenue as f32 / MAX_PRIVATE_REVENUE;
            }
        }
        off += NUM_PRIVATES;

        // S12: Corp Floated
        for c in &self.corporations {
            if let Some(i) = corp_idx(&c.sym) {
                enc[off + i] = if c.floated { 1.0 } else { 0.0 };
            }
        }
        off += NUM_CORPORATIONS;

        // S13: Corp Cash
        for c in &self.corporations {
            if let Some(i) = corp_idx(&c.sym) { enc[off + i] = c.cash as f32 / sc; }
        }
        off += NUM_CORPORATIONS;

        // S14: Corp Trains
        for c in &self.corporations {
            if let Some(ci) = corp_idx(&c.sym) {
                for t in &c.trains {
                    if let Some(ti) = train_idx(&t.name) {
                        enc[off + ci * NUM_TRAIN_TYPES + ti] += 1.0 / TRAIN_COUNTS[ti] as f32;
                    }
                }
            }
        }
        off += NUM_CORPORATIONS * NUM_TRAIN_TYPES;

        // S15: Corp Tokens Remaining
        for c in &self.corporations {
            if let Some(i) = corp_idx(&c.sym) {
                let total = c.tokens.len() as f32;
                let unused = c.tokens.iter().filter(|t| !t.used).count() as f32;
                enc[off + i] = if total > 0.0 { unused / total } else { 0.0 };
            }
        }
        off += NUM_CORPORATIONS;

        // S16: Corp Share Price
        for c in &self.corporations {
            if let Some(i) = corp_idx(&c.sym) {
                if let Some(ref sp) = c.share_price {
                    let ipo = c.ipo_price.as_ref().map(|p| p.price).unwrap_or(0);
                    enc[off + i * 2] = ipo as f32 / MAX_SHARE_PRICE;
                    enc[off + i * 2 + 1] = sp.price as f32 / MAX_SHARE_PRICE;
                }
            }
        }
        off += NUM_CORPORATIONS * 2;

        // S17: Corp Shares (IPO + Market) — count in share UNITS (president=2)
        for c in &self.corporations {
            if let Some(i) = corp_idx(&c.sym) {
                let share_units = |s: &crate::entities::Share| -> f32 {
                    if s.president { (s.percent / 10) as f32 } else { 1.0 }
                };
                let total: f32 = c.shares.iter().map(|s| share_units(s)).sum();
                if total > 0.0 {
                    let ipo_eid = EntityId::ipo(&c.sym);
                    let ipo: f32 = c.shares.iter()
                        .filter(|s| s.owner == ipo_eid || s.owner.is_none())
                        .map(|s| share_units(s)).sum();
                    let mkt: f32 = c.shares.iter()
                        .filter(|s| s.owner == EntityId::market())
                        .map(|s| share_units(s)).sum();
                    enc[off + 2 * i] = ipo / total;
                    enc[off + 2 * i + 1] = mkt / total;
                }
            }
        }
        off += 2 * NUM_CORPORATIONS;

        // S18: Corp Market Zone
        for c in &self.corporations {
            if let Some(i) = corp_idx(&c.sym) {
                if let Some(ref sp) = c.share_price {
                    let zone = self.market_zone_for(sp.row, sp.column);
                    let zi = match zone.as_str() {
                        "no_cert_limit" => 1,
                        "unlimited" => 2,
                        "multiple_buy" => 3,
                        _ => 0,
                    };
                    enc[off + i * 4 + zi] = 1.0;
                }
            }
        }
        off += 4 * NUM_CORPORATIONS;

        // S19: Depot Trains
        for t in &self.depot.trains {
            if let Some(ti) = train_idx(&t.name) {
                enc[off + ti] += 1.0 / TRAIN_COUNTS[ti] as f32;
            }
        }
        off += NUM_TRAIN_TYPES;

        // S20: Market Pool Trains (Discarded)
        for t in &self.depot.discarded {
            if let Some(ti) = train_idx(&t.name) {
                enc[off + ti] += 1.0 / TRAIN_COUNTS[ti] as f32;
            }
        }
        off += NUM_TRAIN_TYPES;

        // S21: Depot Tiles
        for (ti, name) in TILE_IDS.iter().enumerate() {
            let ct = self.tile_counts_remaining.get(*name).copied().unwrap_or(0);
            enc[off + ti] = ct as f32 / TILE_INITIAL_COUNTS[ti] as f32;
        }
        off += NUM_TILE_IDS;

        // S22-24: Auction state
        let is_auction = matches!(&self.round, crate::rounds::Round::Auction(_));
        if is_auction {
            if let crate::rounds::Round::Auction(ref a) = &self.round {
                // S22: Bids
                for (co_i, co) in self.companies.iter().enumerate() {
                    if let Some(pi) = private_idx(&co.sym) {
                        for bid in a.bids.get(&co_i).unwrap_or(&Vec::new()) {
                            let (bid_pid, price) = (bid.player_id, bid.price);
                            if let Some(bi) = pidx(bid_pid) {
                                enc[off + pi * np + bi] = price as f32 / sc;
                            }
                        }
                    }
                }
            }
        }
        off += NUM_PRIVATES * np;

        if is_auction {
            if let crate::rounds::Round::Auction(ref a) = &self.round {
                // S23: Min bid
                for (co_i, co) in self.companies.iter().enumerate() {
                    if let Some(pi) = private_idx(&co.sym) {
                        if co.owner != EntityId::none() && !co.owner.0.is_empty() {
                            enc[off + pi] = -1.0;
                        } else {
                            let mb = a.min_bid_for(co_i, self.companies[co_i].value);
                            enc[off + pi] = mb as f32 / sc;
                        }
                    }
                }
                // S24: Available company
                let cur = a.remaining_companies.first().copied().unwrap_or(usize::MAX);
                if cur < self.companies.len() {
                    if let Some(pi) = private_idx(&self.companies[cur].sym) {
                        enc[off + NUM_PRIVATES + pi] = 1.0;
                    }
                }
            }
        }
        off += NUM_PRIVATES; // min_bid
        off += NUM_PRIVATES; // available

        // S25: Face Value (always)
        for co in &self.companies {
            if let Some(i) = private_idx(&co.sym) {
                enc[off + i] = co.value as f32 / sc;
            }
        }
        off += NUM_PRIVATES;

        // S26: OR Structure
        if let crate::rounds::Round::Operating(_) = &self.round {
            let total_or = self.phase.operating_rounds.max(1) as f32;
            enc[off] = total_or / 3.0;
            enc[off + 1] = self.round_state.round_num as f32 / total_or;
        }
        off += 2;

        // S27: Train Limit
        enc[off] = self.phase.train_limit as f32 / 4.0;
        off += 1;

        // S28: Private Closed
        for co in &self.companies {
            if let Some(i) = private_idx(&co.sym) {
                if co.closed { enc[off + i] = 1.0; }
            }
        }
        off += NUM_PRIVATES;

        // S29: Player Turn Order
        if matches!(&self.round, crate::rounds::Round::Stock(_)) && np > 1 {
            let pp = self.effective_priority_deal_player();
            let nb: Vec<u32> = self.players.iter().filter(|p| !p.bankrupt).map(|p| p.id).collect();
            let pp_pos = nb.iter().position(|&id| id == pp).unwrap_or(0);
            for (pos, &pid) in nb.iter().cycle().skip(pp_pos).take(nb.len()).enumerate() {
                if let Some(i) = pidx(pid) {
                    enc[off + i] = pos as f32 / (np as f32 - 1.0);
                }
            }
        }
        off += np;

        debug_assert_eq!(off, size);
        for v in &mut enc { if v.is_nan() || v.is_infinite() { *v = 0.0; } }
        enc
    }

    fn encode_node_features(&self) -> Vec<f32> {
        let mut f = vec![0.0f32; NUM_HEXES * NUM_NODE_FEATURES];

        for hex in &self.hexes {
            let hi = match hex_idx(&hex.id) { Some(i) => i, None => continue };
            let base = hi * NUM_NODE_FEATURES;
            let tile = &hex.tile;
            let mut o = 0;

            // Revenue
            let rev = if !tile.cities.is_empty() {
                tile.cities[0].revenue
            } else if !tile.towns.is_empty() {
                tile.towns[0].revenue
            } else if !tile.offboards.is_empty() {
                // Use base revenue to match Python encoder behavior
                // (Python adapter doesn't expose dict-based phase revenue)
                tile.offboards[0].revenue
            } else { 0 };
            f[base + o] = rev as f32 / MAX_HEX_REVENUE;
            o += 1;

            // Type flags
            f[base + o] = if !tile.cities.is_empty() { 1.0 } else { 0.0 };
            f[base + o + 1] = if tile.cities.len() > 1 || tile.towns.len() > 1 { 1.0 } else { 0.0 };
            f[base + o + 2] = if !tile.towns.is_empty() { 1.0 } else { 0.0 };
            f[base + o + 3] = if !tile.offboards.is_empty() { 1.0 } else { 0.0 };
            o += 4;

            // Upgrade cost
            let cost = tile.upgrades.first().map(|u| u.cost).unwrap_or(0);
            f[base + o] = cost as f32 / MAX_LAY_COST;
            o += 1;

            // Rotation
            f[base + o] = tile.rotation as f32;
            o += 1;

            // Token presence
            let ts = o;
            for (ci, city) in tile.cities.iter().enumerate() {
                if ci >= 2 { break; }
                for tok in city.tokens.iter().flatten() {
                    if let Some(idx) = corp_idx(&tok.corporation_id) {
                        f[base + ts + idx * 2 + ci] = 1.0;
                    }
                }
            }
            o += NUM_CORPORATIONS * 2;

            // Edge connectivity + revenue connectivity
            let mut ec = [[0.0f32; NUM_TILE_EDGES]; NUM_TILE_EDGES];
            let mut rc = [[0.0f32; 2]; NUM_TILE_EDGES];
            let is_oo = tile.cities.len() > 1 || tile.towns.len() > 1;

            for path in &tile.paths {
                match (&path.a, &path.b) {
                    (PathEndpoint::Edge(a), PathEndpoint::Edge(b)) => {
                        ec[*a as usize][*b as usize] = 1.0;
                        ec[*b as usize][*a as usize] = 1.0;
                    }
                    (PathEndpoint::Edge(e), rev) | (rev, PathEndpoint::Edge(e)) => {
                        let ri = match rev {
                            PathEndpoint::City(i) | PathEndpoint::Town(i) | PathEndpoint::Offboard(i) => {
                                if is_oo { (*i).min(1) } else { 0 }
                            }
                            _ => 0,
                        };
                        rc[*e as usize][ri] = 1.0;
                    }
                    _ => {}
                }
            }

            for i in 0..NUM_TILE_EDGES {
                for j in 0..i {
                    f[base + o + (i * (i - 1) / 2) + j] = ec[i][j];
                }
            }
            o += NUM_PORT_PAIRS;

            for i in 0..NUM_TILE_EDGES {
                for j in 0..2 {
                    f[base + o + i * 2 + j] = rc[i][j];
                }
            }
            o += NUM_TILE_EDGES * 2;

            debug_assert_eq!(o, NUM_NODE_FEATURES);
        }
        f
    }

    /// Get the effective priority deal player (round-aware, matches PyO3 method).
    fn effective_priority_deal_player(&self) -> u32 {
        match &self.round {
            crate::rounds::Round::Stock(s) => s.priority_deal_player,
            crate::rounds::Round::Auction(s) => {
                match s.last_purchaser_id {
                    Some(pid) => self.next_player_id(pid),
                    None => self.priority_deal_player,
                }
            }
            _ => self.priority_deal_player,
        }
    }

    fn market_zone_for(&self, row: u8, col: u8) -> String {
        self.stock_market.cell_at(row, col)
            .map(|c| c.zone.clone())
            .unwrap_or_else(|| "normal".to_string())
    }

    fn num_certs_for_player(&self, player_id: u32) -> u32 {
        // Use the same implementation as num_certs_internal in stock.rs
        self.num_certs_internal(player_id)
    }

    fn player_percent_of(&self, player_id: u32, corp_index: usize) -> i32 {
        let pid = EntityId::player(player_id);
        self.corporations[corp_index].shares.iter()
            .filter(|s| s.owner == pid)
            .map(|s| s.percent as i32)
            .sum()
    }
}
