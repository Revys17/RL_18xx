//! 1822CA tokens: QMOO's Quebec home (`update_home`), ICR's multi-city
//! destination, and the CA Token / PendingToken / DestinationToken step
//! overrides.

use crate::actions::{Action, GameError};
use crate::game::BaseGame;
use crate::tiles::TileColor;

use super::ca::{
    destination_exits, BIG_CITY_HEXES_TO_COMPANIES, COMPANY_WINNIPEG_TOKEN, MOUNTAIN_PASS_HEXES_TO_COMPANIES, QUEBEC_HEX,
};
use super::err;

impl BaseGame {
    /// `destination_city(hex, entity)` as a list: the cities reached by the
    /// corporation's `destination_exits` (in exit, path order), else the
    /// tile's first city.
    pub(crate) fn g1822_ca_destination_cities(&self, corp: &str, hi: usize) -> Vec<usize> {
        let tile = &self.hexes[hi].tile;
        let Some(exits) = destination_exits(corp) else {
            return if tile.cities.is_empty() { Vec::new() } else { vec![0] };
        };
        let mut out = Vec::new();
        for &e in exits {
            for p in &tile.paths {
                use crate::tiles::PathEndpoint::{City, Edge};
                let city = match (&p.a, &p.b) {
                    (Edge(x), City(c)) | (City(c), Edge(x)) if *x == e => Some(*c),
                    _ => None,
                };
                if let Some(c) = city {
                    if !out.contains(&c) {
                        out.push(c);
                    }
                }
            }
        }
        out
    }

    /// CA DestinationToken#destination_node_check?: the destination cities
    /// connected (a walk from the city) to the home token's city.
    pub(crate) fn g1822_ca_connected_destination_cities(&self, corp: &str) -> Vec<usize> {
        use super::graph::{Node, NodeKind};
        let Some(dest) = self.g1822_destination(corp) else { return Vec::new() };
        let Some(&dhi) = self.hex_idx.get(dest.as_str()) else { return Vec::new() };
        let Some(home) = self.g1822_home_node(corp) else { return Vec::new() };
        self.g1822_ca_destination_cities(corp, dhi)
            .into_iter()
            .filter(|&ci| {
                let start = Node { hex: dhi, kind: NodeKind::City, index: ci };
                self.g1822_walk_from(corp, start)
                    .into_iter()
                    .any(|p| self.g1822_path_nodes(p).contains(&home))
            })
            .collect()
    }

    /// CA `place_destination_token(entity, hex, token, city)`: G1822's
    /// (free, unchecked, a cheater) into `city`, then
    /// `@destinated[entity] = true`.
    pub(crate) fn g1822_ca_place_destination(&mut self, corp: &str, hi: usize, ci: usize, tidx: usize) -> Result<(), GameError> {
        self.g1822_put_token(corp, tidx, hi, ci, true)?;
        self.g1822_ruby_graph_clear();
        self.gm().destination_ability.retain(|c| c != corp);
        if !self.g().ca.destinated.iter().any(|c| c == corp) {
            self.gm().ca.destinated.push(corp.to_string());
        }
        Ok(())
    }

    /// CA DestinationToken#actions: nothing once destinated.
    pub(crate) fn g1822_ca_destinated(&self, corp: &str) -> bool {
        self.g().ca.destinated.iter().any(|c| c == corp)
    }

    /// CA DestinationToken#process_hex_token / #process_place_token
    /// (connectivity is not re-checked when loading).
    pub(crate) fn g1822_ca_process_destination(&mut self, corp: &str, action: &Action) -> Result<(), GameError> {
        let tidx = self
            .g1822_unplaced_destination_token(corp)
            .ok_or_else(|| GameError::new("Corporation does not have a destination token unused"))?;
        let dest = self.g1822_destination(corp).ok_or_else(|| GameError::new(format!("{corp} has no destination")))?;
        let (hi, ci) = match action {
            Action::HexToken { hex_id, .. } => {
                if *hex_id != dest {
                    return err(format!("{hex_id} is not {corp}'s destination"));
                }
                if self.g1822_ca_connected_destination_cities(corp).len() > 1 {
                    return err(format!(
                        "{corp} is connected to multiple cities on the hex. Click on the city where it should place \
                         its destination token."
                    ));
                }
                let hi = self.hex_idx[dest.as_str()];
                let cities = self.g1822_ca_destination_cities(corp, hi);
                if cities.len() != 1 {
                    // destination_city returns an Array: Ruby raises
                    // (Array#place_token)
                    return err(format!("{corp}'s destination city on {dest} is ambiguous (Ruby raises)"));
                }
                (hi, cities[0])
            }
            Action::PlaceToken { hex_id, city_index, .. } => {
                let hex = self.g1822_resolve_hex(hex_id)?;
                let hi = *self.hex_idx.get(&hex).ok_or_else(|| GameError::new("unknown hex"))?;
                let ci = *city_index as usize;
                if ci >= self.hexes[hi].tile.cities.len() {
                    return err("Invalid city");
                }
                (hi, ci)
            }
            _ => return err("DestinationToken cannot process this action"),
        };
        self.g1822_ca_place_destination(corp, hi, ci, tidx)?;
        self.g1822_remove_extra_tokens(hi);
        // interactions with ICR's destination and QMOO's home
        if matches!(action, Action::PlaceToken { .. }) && corp == "ICR" {
            self.g1822_ca_update_home(false);
        }
        Ok(())
    }

    /// Tile#remove_reservation!(corp): the first city reserved by it (that
    /// slot becomes nil), else the tile's list.
    fn g1822_ca_tile_remove_reservation(&mut self, hi: usize, corp: &str) {
        let hex = self.hexes[hi].id.clone();
        let n = self.hexes[hi].tile.cities.len();
        for c in 0..n {
            let key = format!("{hex}#{c}");
            if let Some(v) = self.gm().city_reservations.get_mut(&key) {
                if let Some(pos) = v.iter().position(|r| r.as_deref() == Some(corp)) {
                    v[pos] = None;
                    return;
                }
            }
        }
        if let Some(v) = self.gm().tile_reservations.get_mut(&hex) {
            v.retain(|r| r != corp);
        }
    }

    /// `add_corp_reservation!(corp)`: a city reservation if exactly one
    /// city has an available slot (Tile#add_reservation! → City#get_slot),
    /// else (or with no slot) the tile's.
    fn g1822_ca_add_corp_reservation(&mut self, hi: usize, corp: &str) {
        let hex = self.hexes[hi].id.clone();
        let n = self.hexes[hi].tile.cities.len();
        let open: Vec<usize> = (0..n).filter(|&c| self.g1822_available_slots(hi, c) > 0).collect();
        let city = if n == 1 {
            Some(0)
        } else if open.len() == 1 {
            Some(open[0])
        } else {
            None
        };
        let slot = city.and_then(|c| self.g1822_get_slot(hi, c, corp, false).map(|s| (c, s)));
        match slot {
            Some((c, s)) => {
                let v = self.gm().city_reservations.entry(format!("{hex}#{c}")).or_default();
                while v.len() < s {
                    v.push(None);
                }
                v.insert(s, Some(corp.to_string()));
            }
            None => self.gm().tile_reservations.entry(hex).or_default().push(corp.to_string()),
        }
    }

    /// `update_home(qmoo, tile_trigger:)`: QMOO's home reservation follows
    /// Quebec's cities; a yellow lay under its placed home token takes the
    /// token back up to be placed again.
    pub(crate) fn g1822_ca_update_home(&mut self, tile_trigger: bool) {
        let corp = "QMOO";
        let Some(&cix) = self.corp_idx.get(corp) else { return };
        let Some(&hi) = self.hex_idx.get(QUEBEC_HEX) else { return };
        let Some(first) = self.corporations[cix].tokens.first().cloned() else { return };
        let operated = first.used && first.city_hex_id == QUEBEC_HEX;
        if operated && tile_trigger && self.hexes[hi].tile.color == TileColor::Yellow {
            // corp.tokens.first.remove!
            for city in &mut self.hexes[hi].tile.cities {
                for t in city.tokens.iter_mut() {
                    if t.as_ref().is_some_and(|t| t.corporation_id == corp && t.serial == first.serial) {
                        *t = None;
                    }
                }
            }
            self.corporations[cix].tokens[0].used = false;
            self.corporations[cix].tokens[0].city_hex_id.clear();
            self.gm().home_city.remove(corp);
            self.clear_graph_cache();
            self.g1822_ruby_graph_clear();
            self.g1822_ca_add_corp_reservation(hi, corp);
            self.g1822_place_home_token(corp);
        } else if !operated {
            self.g1822_ca_tile_remove_reservation(hi, corp);
            self.g1822_ca_add_corp_reservation(hi, corp);
        }
    }

    /// CA PendingToken#process_place_token's QMOO check.
    pub(crate) fn g1822_ca_pending_token_check(&self, corp: &str, hi: usize, ci: usize) -> Result<(), GameError> {
        if self.g1822_is_ca() && corp == "QMOO" && self.hexes[hi].id == QUEBEC_HEX && self.g1822_available_slots(hi, ci) == 0
        {
            return err("Cannot place QMOO's home token in occupied or reserved city");
        }
        Ok(())
    }

    /// CA `after_place_token(entity, city)`.
    pub(crate) fn g1822_ca_after_place_token(&mut self, hi: usize) {
        if self.g1822_is_ca() && self.hexes[hi].id == QUEBEC_HEX {
            self.g1822_ca_update_home(false);
        }
    }

    /// CA `after_lay_tile` (after G1822's): the big-city privates lose their
    /// lay once their city is gray, the pass privates once the pass is
    /// yellow; a Quebec lay updates QMOO's home.
    pub(crate) fn g1822_ca_after_lay_tile(&mut self, hi: usize) {
        let hex = self.hexes[hi].id.clone();
        let color = self.hexes[hi].tile.color;
        let co = if color == TileColor::Yellow {
            MOUNTAIN_PASS_HEXES_TO_COMPANIES.iter().find(|(h, _)| *h == hex).map(|(_, c)| *c)
        } else if color == TileColor::Gray {
            BIG_CITY_HEXES_TO_COMPANIES.iter().find(|(h, _)| *h == hex).map(|(_, c)| *c)
        } else {
            None
        };
        if let Some(co) = co {
            self.gm().abilities.remove(co);
        }
        if hex == QUEBEC_HEX {
            self.g1822_ca_update_home(true);
        }
    }

    // -- destination tokens across lays (preprocess_action / action_processed)

    /// CA `preprocess_action(LayTile)`: on a destination hex, the cheater
    /// destination tokens are picked up (their slot becomes nil) before the
    /// lay; returns (corporation, token serial, its city on the new tile).
    pub(crate) fn g1822_ca_preprocess_lay(&mut self, action: &Action) -> Vec<(String, u32, Option<usize>)> {
        let Action::LayTile { hex_id, tile_id, rotation, .. } = action else { return Vec::new() };
        let Some(&hi) = self.hex_idx.get(hex_id.as_str()) else { return Vec::new() };
        let is_dest = self
            .corporations
            .iter()
            .any(|c| self.g1822_destination(&c.sym).as_deref() == Some(hex_id.as_str()));
        if !is_dest {
            return Vec::new();
        }
        let base = tile_id.split('-').next().unwrap_or(tile_id);
        let Some(def) = self.tile_catalog.get(base).cloned() else { return Vec::new() };
        let new_tile = BaseGame::tile_from_def(&def, *rotation);
        let map = BaseGame::g1822_city_map(&self.hexes[hi].tile, &new_tile);
        let mut out = Vec::new();
        let n = self.hexes[hi].tile.cities.len();
        for ci in 0..n {
            let picked: Vec<(usize, String, u32)> = self.hexes[hi].tile.cities[ci]
                .tokens
                .iter()
                .enumerate()
                .filter_map(|(slot, t)| {
                    t.as_ref()
                        .filter(|t| t.token_type == "destination" && t.cheater)
                        .map(|t| (slot, t.corporation_id.clone(), t.serial))
                })
                .collect();
            for (slot, corp, serial) in picked {
                // Token#remove!
                self.hexes[hi].tile.cities[ci].tokens[slot] = None;
                if let Some(&cix) = self.corp_idx.get(&corp) {
                    if let Some(ct) = self.corporations[cix].tokens.iter_mut().find(|t| t.serial == serial) {
                        ct.used = false;
                        ct.city_hex_id.clear();
                    }
                }
                out.push((corp, serial, map.get(ci).copied().flatten()));
            }
        }
        if !out.is_empty() {
            self.clear_graph_cache();
        }
        out
    }

    /// CA `action_processed(LayTile)`: the picked-up destination tokens go
    /// back down (place_destination_token, a cheater only if it needs one).
    pub(crate) fn g1822_ca_action_processed_lay(&mut self, hex_id: &str, pending: Vec<(String, u32, Option<usize>)>) {
        let Some(&hi) = self.hex_idx.get(hex_id) else { return };
        for (corp, serial, city) in pending {
            let Some(&cix) = self.corp_idx.get(&corp) else { continue };
            let Some(tidx) = self.corporations[cix].tokens.iter().position(|t| t.serial == serial) else { continue };
            let ci = city.unwrap_or_else(|| self.g1822_ca_destination_cities(&corp, hi).first().copied().unwrap_or(0));
            let _ = self.g1822_ca_place_destination(&corp, hi, ci, tidx);
        }
    }

    // -- P10 (Winnipeg station token) -----------------------------------------

    /// P10's token ability as `@game.abilities(P10, :token)` sees it: open,
    /// corporation-owned, its corporation operating (owning_corp_or_turn;
    /// same_hex_allowed, so never dropped by ability_usable?). The owner.
    pub(crate) fn g1822_ca_p10_owner(&self) -> Option<String> {
        if !self.g1822_is_ca() || self.co_closed(COMPANY_WINNIPEG_TOKEN) {
            return None;
        }
        let owner = self.co_owner(COMPANY_WINNIPEG_TOKEN).corp_sym()?.to_string();
        (self.g1822_operator().as_deref() == Some(owner.as_str())).then_some(owner)
    }

    /// CA SpecialToken#actions for P10.
    pub(crate) fn g1822_ca_special_token_actions(&self, co: &str) -> Vec<&'static str> {
        if co != COMPANY_WINNIPEG_TOKEN {
            return vec![];
        }
        match self.g1822_ca_p10_owner() {
            Some(owner) if self.g1822_exchange_tokens(&owner) > 0 => vec!["place_token"],
            _ => vec![],
        }
    }

    /// CA SpecialToken#process_place_token for P10: a fresh, free token of
    /// the owner in an extra slot of a Winnipeg city (connected: false,
    /// unchecked, an extra action), P10 closes, an exchange token is used.
    pub(crate) fn g1822_ca_special_token_process(&mut self, co: &str, action: &Action) -> Result<(), GameError> {
        let Action::PlaceToken { hex_id, city_index, .. } = action else {
            return err("SpecialToken cannot process this action");
        };
        let owner = self.g1822_ca_p10_owner().ok_or_else(|| GameError::new(format!("{co} cannot place a token now")))?;
        let hex = self.g1822_resolve_hex(hex_id)?;
        let hi = *self.hex_idx.get(&hex).ok_or_else(|| GameError::new("unknown hex"))?;
        let ci = *city_index as usize;
        if ci >= self.hexes[hi].tile.cities.len() {
            return err("Invalid city");
        }
        if self.g1822_city_tokened_by(hi, ci, &owner) {
            return err(format!("{owner} already has a token in {hex} city {ci}"));
        }
        if hex != "N16" {
            return err(format!("Cannot place token on {hex}"));
        }
        let cix = self.ci(&owner);
        let mut tok = crate::entities::Token::new(owner.clone(), 0);
        self.gm().token_serial += 1;
        tok.serial = self.g().token_serial;
        tok.used = true;
        tok.city_hex_id = hex.clone();
        self.corporations[cix].tokens.push(tok.clone());
        self.hexes[hi].tile.cities[ci].extra_tokens.push(tok);
        self.g1822_consume_reservation(&hex, ci, &owner);
        self.clear_graph_cache();
        self.g1822_ruby_graph_clear();
        // ability.use! (count 1, closed_when_used_up)
        self.close_company(co);
        self.g1822_remove_exchange_token(&owner);
        Ok(())
    }

    /// CA Track#process_pass: P21 closes once its grant was used.
    pub(crate) fn g1822_ca_track_pass(&mut self, corp: &str) {
        let p21 = self.fam().company_lsr;
        if !self.g1822_owns(corp, p21) {
            return;
        }
        let used = self.g1822_abilities(p21).first().map_or(false, |a| a.used);
        if used {
            self.close_company(p21);
        }
    }
}
