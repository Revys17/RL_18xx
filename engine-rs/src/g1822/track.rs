//! 1822 track: the Track and SpecialTrack steps (G1822 Tracker over
//! Engine::Step::Tracker), Hex#lay (city map, reservations, tokens),
//! terrain/border costs with the 1822 discounts, and the private
//! companies' tile-lay abilities (P2, P5 teleport, P8, P10, P11, P12, P21).

use crate::actions::{Action, GameError};
use crate::entities::EntityId;
use crate::game::BaseGame;
use crate::graph::City;

use super::family::Family;
use super::{err, Ent};

/// A company ability (Ruby Ability::TileLay / Teleport / TileDiscount /
/// Token), the subset 1822 uses.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Ab {
    pub kind: AbKind,
    pub count: Option<i32>,
    pub used: bool,
    pub hexes: Vec<String>,
    pub tiles: Vec<String>,
    pub free: bool,
    pub special: bool,
    pub reachable: bool,
    pub closed_when_used_up: bool,
    pub must_lay_together: bool,
    pub consume_tile_lay: bool,
    pub lay_count: Option<i32>,
    pub upgrade_count: Option<i32>,
    pub combo: Vec<String>,
    pub discount: i32,
    pub terrain: Option<String>,
    /// `when: 'track'` only (1822PNW's P10, P12, P13, P16, P18, P19):
    /// usable while the Track step blocks, by its owner as operator.
    #[serde(default)]
    pub track_only: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum AbKind {
    TileLay,
    Teleport,
    TileDiscount,
}

impl Ab {
    fn tile_lay(count: Option<i32>) -> Ab {
        Ab {
            kind: AbKind::TileLay,
            count,
            used: false,
            hexes: vec![],
            tiles: vec![],
            free: false,
            special: true,
            reachable: true,
            closed_when_used_up: true,
            must_lay_together: false,
            consume_tile_lay: false,
            lay_count: None,
            upgrade_count: None,
            combo: vec![],
            discount: 0,
            terrain: None,
            track_only: false,
        }
    }
    fn discount(d: i32, terrain: Option<&str>, hexes: &[&str]) -> Ab {
        Ab {
            kind: AbKind::TileDiscount,
            count: None,
            discount: d,
            terrain: terrain.map(str::to_string),
            hexes: hexes.iter().map(|s| s.to_string()).collect(),
            ..Ab::tile_lay(None)
        }
    }
}

fn strs(v: &[&str]) -> Vec<String> {
    v.iter().map(|s| s.to_string()).collect()
}

/// entities.rb abilities of the 1822 privates (owner_type corporation).
pub fn abilities_1822() -> std::collections::BTreeMap<String, Vec<Ab>> {
    let mut m = std::collections::BTreeMap::new();
    let plain = ["7", "8", "9", "80", "81", "82", "83", "544", "545", "546", "60", "169"];
    m.insert(
        "P2".to_string(),
        vec![Ab {
            tiles: strs(&plain),
            combo: strs(&["P8", "P10", "P11", "P12"]),
            ..Ab::tile_lay(Some(1))
        }],
    );
    m.insert(
        "P5".to_string(),
        vec![Ab {
            kind: AbKind::Teleport,
            hexes: strs(&["P43"]),
            tiles: strs(&["X9", "X15"]),
            reachable: false,
            closed_when_used_up: false,
            ..Ab::tile_lay(None)
        }],
    );
    let p8_hexes = [
        "B41", "C40", "D39", "E10", "F9", "G14", "G30", "G8", "H15", "H39", "H7", "H9", "I10", "I12", "I14", "I16",
        "I18", "I20", "I22", "I24", "I38", "I40", "I8", "J19", "J21", "J23", "J25", "J39", "J7", "O40",
    ];
    m.insert(
        "P8".to_string(),
        vec![
            Ab {
                hexes: strs(&p8_hexes),
                free: true,
                special: false,
                combo: strs(&["P2", "P12"]),
                ..Ab::tile_lay(Some(1))
            },
            Ab::discount(
                20,
                None,
                &["C40", "E10", "F9", "G8", "H7", "H9", "H15", "I8", "I10", "I22", "J7", "J21", "J23", "J25"],
            ),
            Ab::discount(20, Some("mountain"), &[]),
        ],
    );
    m.insert(
        "P10".to_string(),
        vec![
            Ab {
                combo: strs(&["P2", "P11", "P12"]),
                ..Ab::tile_lay(Some(2))
            },
            Ab::discount(10, Some("swamp"), &[]),
        ],
    );
    m.insert(
        "P11".to_string(),
        vec![Ab {
            tiles: strs(&[
                "80", "81", "82", "83", "544", "545", "546", "60", "169", "141", "142", "143", "144", "767", "768", "769",
                "X17",
            ]),
            combo: strs(&["P2", "P10", "P12"]),
            ..Ab::tile_lay(Some(1))
        }],
    );
    m.insert(
        "P12".to_string(),
        vec![Ab {
            must_lay_together: true,
            lay_count: Some(2),
            upgrade_count: Some(1),
            combo: strs(&["P2", "P8", "P10", "P11"]),
            ..Ab::tile_lay(Some(2))
        }],
    );
    m.insert(
        "P21".to_string(),
        vec![Ab {
            hexes: strs(&["N21", "N23"]),
            tiles: strs(&["5", "6", "57", "15"]),
            free: true,
            must_lay_together: true,
            consume_tile_lay: true,
            ..Ab::tile_lay(Some(2))
        }],
    );
    m
}

impl BaseGame {
    pub(crate) fn g1822_abilities(&self, company: &str) -> Vec<Ab> {
        self.g().abilities.get(company).cloned().unwrap_or_default()
    }

    /// owner_type corporation: the company is owned by a corporation.
    pub(crate) fn g1822_ability_owner_ok(&self, company: &str) -> bool {
        !self.co_closed(company) && self.co_owner(company).is_corporation()
    }

    /// The usable tile_lay / teleport ability of a company (G1822
    /// SpecialTrack#abilities incl. the P5 and P21 gates).
    pub(crate) fn g1822_track_ability(&self, company: &str) -> Option<(usize, Ab)> {
        if !self.g1822_ability_owner_ok(company) {
            return None;
        }
        let owner = self.co_owner(company).corp_sym()?.to_string();
        let f = self.fam();
        if Family::is(f.company_lcdr, company) && !self.rnd().special_track_in_process {
            // P5 may only teleport-upgrade when the Channel is full and an
            // upgrade in the current phase would open a slot.
            let hi = self.hex_idx[f.english_channel_hex];
            let tile_color = self.hexes[hi].tile.color;
            let phase_color = self.phase.tiles.last().cloned().unwrap_or_default();
            if self.g1822_available_slots(hi, 0) > 0
                || self.g1822_exchange_tokens(&owner) == 0
                || (tile_color == crate::tiles::TileColor::Green && !matches!(phase_color.as_str(), "brown" | "gray"))
                || (tile_color == crate::tiles::TileColor::Brown && phase_color != "gray")
            {
                return None;
            }
        }
        if Family::is(f.company_hsbc, company) {
            if self.rnd().num_laid_track > 0 && self.rnd().special_track_company.as_deref() != Some(company) {
                // `entity != current_entity` — the company is never the
                // round's current entity unless it is mid-use.
                return None;
            }
            let cix = self.ci(&owner);
            let has = self.corporations[cix]
                .tokens
                .iter()
                .any(|t| t.used && f.company_hsbc_tiles.contains(&t.city_hex_id.as_str()));
            if !has {
                return None;
            }
        }
        let track_time = self.g1822_track_is_blocking_step() && self.g1822_operator().as_deref() == Some(owner.as_str());
        self.g1822_abilities(company)
            .into_iter()
            .enumerate()
            .find(|(_, a)| match a.kind {
                AbKind::TileLay if a.track_only && !track_time => false,
                AbKind::TileLay => a.count.map_or(true, |c| c > 0),
                AbKind::Teleport => !a.used,
                _ => false,
            })
    }

    /// The active step responds to process_lay_tile (Track / SpecialTrack).
    fn g1822_active_lays(&self) -> bool {
        self.g1822_active_step().map_or(false, |a| {
            matches!(self.rnd().steps[a], super::Step::Track | super::Step::SpecialTrack)
        })
    }

    pub(crate) fn g1822_special_track_actions(&self, ent: &Ent) -> Vec<&'static str> {
        let Ent::Company(co) = ent else { return vec![] };
        if self.rnd().special_track_company.as_deref() == Some(co.as_str()) {
            return vec!["lay_tile", "pass"];
        }
        if self.g1822_track_ability(co).is_some() && self.g1822_active_lays() {
            vec!["lay_tile"]
        } else {
            vec![]
        }
    }

    /// G1822 tile_lays(entity) → the allowance slot for the current lay.
    pub(crate) fn g1822_tile_lay(&self, operator: &str, hsbc: bool) -> Option<(bool, bool, bool)> {
        // (lay, upgrade, cannot_reuse_same_hex)
        let n = self.rnd().num_laid_track;
        let upgraded = self.rnd().upgraded_track;
        if hsbc {
            return match n {
                0 => Some((true, true, false)),
                1 => Some((true, !upgraded, true)),
                _ => None,
            };
        }
        if self.phase_num() >= 3 && self.is_major(operator) {
            return match n {
                0 => Some((true, true, false)),
                1 => Some((!upgraded, false, false)),
                _ => None,
            };
        }
        match n {
            0 => Some((true, true, false)),
            _ => None,
        }
    }

    /// G1822 Tracker#can_lay_tile? for the operating corporation.
    pub(crate) fn g1822_can_lay_tile(&self, c: &str) -> bool {
        if c == self.fam().minor_14_id && !self.g1822_operated(c) {
            return false;
        }
        // tile_lay_abilities_should_block?: an owned company's tile_lay
        // ability usable at 'track' that does not consume the lay.
        if self.g1822_track_is_blocking_step() {
            let owner = EntityId::corporation(c);
            for co in self.companies_of(&owner) {
                if let Some((_, ab)) = self.g1822_track_ability(&co) {
                    if ab.kind == AbKind::TileLay && !ab.consume_tile_lay {
                        return true;
                    }
                }
            }
        }
        let Some((lay, upgrade, _)) = self.g1822_tile_lay(c, false) else { return false };
        let cix = self.ci(c);
        !self.corporations[cix].tokens.is_empty() && (lay || upgrade)
    }

    /// Ruby `ability_blocking_step` is the Track step (not passed).
    pub(crate) fn g1822_track_is_blocking_step(&self) -> bool {
        let r = self.rnd();
        for (i, s) in r.steps.iter().enumerate() {
            match s {
                super::Step::SpecialTrack => {
                    if r.special_track_company.is_some() {
                        return false;
                    }
                }
                super::Step::FirstTurnHousekeeping | super::Step::Token | super::Step::Route
                | super::Step::Dividend | super::Step::BuyTrain => {
                    if !r.passed[i] {
                        return false;
                    }
                }
                super::Step::Track => return !r.passed[i],
                _ => {}
            }
        }
        false
    }

    pub(crate) fn g1822_track_actions(&self, ent: &Ent) -> Vec<&'static str> {
        if let Ent::Company(co) = ent {
            if !self.g1822_company_choices(co, "track").is_empty() {
                return vec!["choose_ability"];
            }
            return vec![];
        }
        let Ent::Corp(c) = ent else { return vec![] };
        // MX Track#actions: the NdeM lays only through its shareholders
        if self.g1822_mx_is_ndem(c) && self.g1822_operator().as_deref() == Some(c.as_str()) && self.rnd().ndem_layers.is_empty() {
            return vec![];
        }
        if Some(c) != self.g1822_operator().as_ref() || !self.g1822_can_lay_tile(c) {
            return vec![];
        }
        vec!["lay_tile", "pass"]
    }

    pub(crate) fn g1822_track_process(&mut self, i: usize, ent: &Ent, action: &Action) -> Result<(), GameError> {
        match action {
            Action::LayTile { hex_id, tile_id, rotation, .. } => {
                let Ent::Corp(c) = ent else { return err("lay_tile by a non-corporation") };
                let c = c.clone();
                let pnw = self.g1822_is_pnw();
                let mx = self.g1822_is_mx();
                if mx && tile_id == "BC-0" {
                    // MX Track#process_lay_tile: a builder cube uses the lay
                    let lay = self.g1822_tile_lay(&c, false).map_or(false, |t| t.0);
                    if !lay {
                        return err("Cannot lay a builder cube now");
                    }
                    if !self.hex_idx.contains_key(hex_id.as_str()) {
                        return err(format!("Unknown hex {hex_id}"));
                    }
                    self.g1822_pnw_place_cube(hex_id);
                    self.rndm().num_laid_track += 1;
                    self.rndm().laid_hexes.push(hex_id.clone());
                    if !self.g1822_can_lay_tile(&c) {
                        self.g1822_mx_track_pass_any(i, &c);
                    }
                    return Ok(());
                }
                if pnw {
                    if self.rnd().num_laid_portage > 0 {
                        return err("Cannot place a tile or cube now");
                    }
                    if tile_id.split('-').next() == Some("BC") {
                        // a builder cube: uses the lay, keeps the step
                        let lay = self.g1822_tile_lay(&c, false).map_or(false, |t| t.0);
                        if !lay {
                            return err("Cannot lay a builder cube now");
                        }
                        self.g1822_pnw_place_cube(hex_id);
                        self.rndm().num_laid_track += 1;
                        self.rndm().laid_hexes.push(hex_id.clone());
                        return Ok(());
                    }
                }
                let hi = *self.hex_idx.get(hex_id.as_str()).ok_or_else(|| GameError::new(format!("Unknown hex {hex_id}")))?;
                let forest = pnw && self.g1822_pnw_forest_tile(hi);
                // Ruby computes the operator's graph in legal_tile_rotation?
                // (before) and check_track_restrictions! (after the lay) —
                // PNW's rotation check returns early (no graph) for the same
                // tile and rotation, the cube and H19's city+town tiles.
                let base = tile_id.split('-').next().unwrap_or(tile_id);
                let cur = self.hexes[hi].tile.name.split('-').next().unwrap_or("").to_string();
                let early = pnw
                    && ((!self.hexes[hi].tile.id.starts_with("preprinted_") && cur == base && self.hexes[hi].tile.rotation == *rotation)
                        || Self::g1822_pnw_city_and_town_tile(hex_id, base));
                if !early {
                    self.g1822_note_graph(&c);
                }
                self.g1822_lay_tile_action(&c, &c, &[], hex_id, tile_id, *rotation, false)?;
                self.g1822_note_graph(&c);
                if !self.g1822_can_lay_tile(&c) {
                    if mx {
                        self.g1822_mx_track_pass_any(i, &c);
                    } else {
                        self.rndm().passed[i] = true;
                    }
                }
                if mx {
                    // MX: the new tile carries no builder cubes
                    self.g1822_pnw_drop_cubes(hex_id.as_str());
                }
                if pnw {
                    self.g1822_pnw_drop_cubes(hex_id.as_str());
                    if forest {
                        self.g1822_pnw_assign(hex_id, "forest");
                    }
                }
                Ok(())
            }
            Action::ChooseAbility { choice, .. } => {
                let Ent::Company(co) = ent else { return err("choose_ability by a non-company") };
                self.g1822_company_made_choice(co, choice, "track")
            }
            Action::Pass { .. } => {
                match ent {
                    Ent::Corp(c) if self.g1822_is_mx() => {
                        let c = c.clone();
                        self.g1822_mx_track_pass_any(i, &c);
                    }
                    _ => self.rndm().passed[i] = true,
                }
                Ok(())
            }
            _ => err("Track cannot process this action"),
        }
    }

    /// Tracker#lay_tile_action (the slot rules) + lay_tile.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn g1822_lay_tile_action(
        &mut self,
        entity: &str,
        spender: &str,
        combo: &[String],
        hex_id: &str,
        tile_id: &str,
        rotation: u8,
        hsbc: bool,
    ) -> Result<(), GameError> {
        let hi = *self.hex_idx.get(hex_id).ok_or_else(|| GameError::new(format!("Unknown hex {hex_id}")))?;
        let base = tile_id.split('-').next().unwrap_or(tile_id).to_string();
        let def = self
            .tile_catalog
            .get(&base)
            .cloned()
            .ok_or_else(|| GameError::new(format!("Unknown tile {tile_id}")))?;
        let operator = if self.corp_idx.contains_key(entity) {
            entity.to_string()
        } else {
            self.co_owner(entity).corp_sym().unwrap_or(spender).to_string()
        };
        // track_upgrade? (PNW: token cities always; blue counts as colour)
        let upgrade = if self.g1822_is_pnw() {
            self.g1822_tile_color_name(hi) != "white" || Self::g1822_pnw_tokencity(hex_id)
        } else {
            self.hexes[hi].tile.color != crate::tiles::TileColor::White
        };
        // 1822PNW SpecialTrack#get_tile_lay: a portage lay uses up the
        // turn's lay for every other company
        let portage_used = self.g1822_is_pnw()
            && self.rnd().num_laid_portage > 0
            && !self.corp_idx.contains_key(entity)
            && entity != "P16";
        let (lay, can_upgrade, no_reuse) = (if portage_used { None } else { self.g1822_tile_lay(&operator, hsbc) })
            .ok_or_else(|| GameError::new("No tile lay available"))?;
        if upgrade && !can_upgrade {
            return err("Cannot lay an upgrade now");
        }
        if def.color == crate::tiles::TileColor::Yellow && !lay {
            return err("Cannot lay a yellow now");
        }
        if no_reuse && self.rnd().laid_hexes.iter().any(|h| h == hex_id) {
            return err(format!("{hex_id} cannot be laid as this hex was already laid on this turn"));
        }
        self.g1822_lay_tile(entity, spender, combo, hi, tile_id, rotation)?;
        if upgrade {
            self.rndm().upgraded_track = true;
        }
        self.rndm().num_laid_track += 1;
        self.rndm().laid_hexes.push(hex_id.to_string());
        Ok(())
    }

    /// Tracker#lay_tile: tile counts, Hex#lay, costs, token clean-up.
    pub(crate) fn g1822_lay_tile(
        &mut self,
        entity: &str,
        spender: &str,
        combo: &[String],
        hi: usize,
        tile_id: &str,
        rotation: u8,
    ) -> Result<(), GameError> {
        let base = tile_id.split('-').next().unwrap_or(tile_id).to_string();
        let def = self
            .tile_catalog
            .get(&base)
            .cloned()
            .ok_or_else(|| GameError::new(format!("Unknown tile {tile_id}")))?;
        let mut entities: Vec<String> = vec![entity.to_string()];
        entities.extend(combo.iter().cloned());
        let hex_id = self.hexes[hi].id.clone();
        // Tracker#lay_tile's tile legality — what Ruby checks for a live
        // player (process_action of a live game): upgrades_to? (always),
        // legal_tile_rotation? (the step's own rule), and after the lay
        // check_track_restrictions! for a corporation's own lay. Shared with
        // the legal-move generator (g1822::tile_rules).
        let is_company = self.company_idx.contains_key(entity);
        let pnw = self.g1822_is_pnw();
        // PNW Track/SpecialTrack#lay_tile: 'Cannot upgrade forests'
        if pnw && self.g1822_pnw_assigned(&hex_id, "forest") {
            return err("Cannot upgrade forests");
        }
        let selected = is_company.then_some(entity);
        if !self.g1822_upgrades_to_sel(hi, &def, is_company, selected) {
            return err(format!("{} is not upgradeable to {}", self.hexes[hi].tile.name, base));
        }
        {
            use super::tile_rules::RotRule;
            let f = self.fam();
            let rule = if Family::is(f.company_lcdr, entity) && hex_id == f.english_channel_hex {
                RotRule::LcdrChannel
            } else if is_company && self.g1822_is_mx() && Self::g1822_mx_port_company(entity) {
                RotRule::MxPort
            } else if pnw && is_company && self.g1822_pnw_rot_rule(entity).is_some() {
                self.g1822_pnw_rot_rule(entity).unwrap()
            } else if entities.iter().any(|e| self.g1822_must_remove_town(e)) {
                RotRule::RemoveTown
            } else {
                RotRule::Normal
            };
            let connected: Vec<u8> = if rule == RotRule::LcdrChannel {
                Vec::new()
            } else {
                self.g1822_graph(spender).connected_hexes.get(&hi).cloned().unwrap_or_default()
            };
            let nbrs = self.g1822_neighbors();
            if !self.g1822_legal_rotation(rule, spender, hi, &def, rotation, &connected, &nbrs) {
                return err(format!("{} is not legally rotated for {}", self.hexes[hi].tile.name, base));
            }
        }
        let old_tile_for_check = (!is_company).then(|| self.hexes[hi].tile.clone());
        // free / discount from the company abilities in play.
        let mut free = false;
        for e in &entities {
            if !self.company_idx.contains_key(e.as_str()) {
                continue;
            }
            if let Some((_, ab)) = self.g1822_track_ability(e) {
                if !ab.hexes.is_empty() && !ab.hexes.contains(&hex_id) {
                    continue;
                }
                if !ab.tiles.is_empty() && !ab.tiles.contains(&base) {
                    continue;
                }
                if ab.kind == AbKind::TileLay {
                    free |= ab.free;
                }
            }
        }
        // update_tile_lists
        let remaining = self.tile_counts_remaining.get(&base).copied().unwrap_or(0);
        if remaining == 0 {
            return err(format!("No tiles of type {base} remaining"));
        }
        let old_cost = match pnw.then(|| self.g1822_pnw_tokencity_cost(hi, &base)).flatten() {
            Some(c) => c,
            None => self.g1822_upgrade_cost(hi, spender),
        };
        let old_color = self.g1822_tile_color_name(hi);
        let old_name = self.hexes[hi].tile.name.clone();
        let old_base = old_name.split('-').next().unwrap_or(&old_name).to_string();
        if !self.hexes[hi].tile.id.starts_with("preprinted_") && !old_base.is_empty() && old_base != hex_id {
            let n = self.tile_counts_remaining.entry(old_base).or_insert(0);
            if *n != u32::MAX {
                *n += 1;
            }
        }
        if let Some(n) = self.tile_counts_remaining.get_mut(&base) {
            if *n != u32::MAX {
                *n -= 1;
            }
        }
        // Instance id: the record names Ruby's tile instance ("8-0");
        // otherwise the next free index for the tile name.
        let instance = if tile_id.contains('-') {
            tile_id.to_string()
        } else {
            self.g1822_next_tile_instance(&base)
        };
        let old_tile_id = self.hexes[hi].tile.id.clone();
        self.g1822_hex_lay(hi, &def, rotation, &instance)?;
        if pnw {
            // Tracker#lay_tile: a lay on an impassable colour opens it
            self.g1822_pnw_open_neighbors(hi, old_color);
            self.g1822_pnw_carry_cubes(&old_tile_id, &instance);
        }
        // costs
        let (border, _) = self.g1822_remove_border_cost(hi, &entities)?;
        let cost = if free {
            0
        } else if pnw {
            // PNW tile_cost_with_discount: the boomtown is free, each
            // builder cube on the hex takes 40 off
            if base == "PNW3" {
                0
            } else {
                (old_cost + border - 40 * self.g1822_pnw_cubes(&hex_id) as i32).max(0)
            }
        } else {
            // Game#tile_cost_with_discount: the operator's terrain-less
            // tile_discount abilities naming the hex (P8 "discount" choice:
            // £20 off at its hill hexes) come off the whole cost.
            let base = old_cost + border;
            let d: i32 = self
                .companies_of(&EntityId::corporation(spender))
                .iter()
                .flat_map(|co| self.g1822_abilities(co))
                .filter(|a| {
                    a.kind == AbKind::TileDiscount && a.terrain.is_none() && (a.hexes.is_empty() || a.hexes.contains(&hex_id))
                })
                .map(|a| a.discount)
                .sum();
            if d == 0 {
                base
            } else {
                base - base.min(d)
            }
        };
        if cost > 0 {
            self.corp_spend(spender, cost)?;
        }
        if let Some(old_tile) = &old_tile_for_check {
            if !self.g1822_track_restrictions_ok(entity, hi, old_tile) {
                return err("Must use new track or change city value");
            }
        }
        // update_token!: G1822 remove_extra_tokens! first.
        self.g1822_remove_extra_tokens(hi);
        if pnw {
            self.g1822_pnw_after_lay_tile(hi);
        }
        // after_lay_tile: minor 14's extra London slot survives upgrades.
        if hex_id == self.fam().minor_14_home_hex {
            if let Some(exit) = self.g().minor_14_city_exit {
                let tile = &self.hexes[hi].tile;
                let city = tile.paths.iter().find_map(|p| match (&p.a, &p.b) {
                    (crate::tiles::PathEndpoint::Edge(e), crate::tiles::PathEndpoint::City(c))
                    | (crate::tiles::PathEndpoint::City(c), crate::tiles::PathEndpoint::Edge(e))
                        if *e == exit =>
                    {
                        Some(*c)
                    }
                    _ => None,
                });
                if let Some(ci) = city {
                    let city = &mut self.hexes[hi].tile.cities[ci];
                    if city.tokens.len() <= city.slots as usize {
                        let ns = city.slots as usize;
                        while city.tokens.len() <= ns {
                            city.tokens.push(None);
                        }
                    }
                }
            }
        }
        self.clear_graph_cache();
        Ok(())
    }

    fn g1822_next_tile_instance(&mut self, base: &str) -> String {
        let n = self.gm().tile_instances.entry(base.to_string()).or_insert(0);
        let id = format!("{base}-{n}");
        *n += 1;
        id
    }

    /// G1822 `upgrade_cost`: the hex's terrain upgrades minus the
    /// operator's matching tile_discount abilities (exact terrain match).
    fn g1822_upgrade_cost(&self, hi: usize, operator: &str) -> i32 {
        let base = self.g1822_upgrade_cost_g1822(hi, operator);
        if self.g1822_is_mx() {
            // MX upgrade_cost: the builder cubes on the hex
            return base - self.g1822_mx_cube_discount(hi);
        }
        base
    }

    fn g1822_upgrade_cost_g1822(&self, hi: usize, operator: &str) -> i32 {
        let hex_id = &self.hexes[hi].id;
        let discounts: Vec<Ab> = self
            .companies_of(&EntityId::corporation(operator))
            .iter()
            .flat_map(|co| self.g1822_abilities(co))
            .filter(|a| a.kind == AbKind::TileDiscount && (a.hexes.is_empty() || a.hexes.contains(hex_id)))
            .collect();
        self.hexes[hi]
            .tile
            .upgrades
            .iter()
            .map(|u| {
                let mut total = u.cost;
                for a in &discounts {
                    let terrains: Vec<&str> = if u.terrain.is_empty() { vec![] } else { vec![u.terrain.as_str()] };
                    if a.terrain.as_deref().map(|t| vec![t]) == Some(terrains.clone()) {
                        total -= a.discount;
                    }
                }
                total
            })
            .sum()
    }

    /// G1822 remove_border_calculate_cost!: water borders the new tile
    /// crosses (it targets the neighbour) are paid and deleted; the
    /// neighbour's copy only when its tile exits there. P10 knocks £40 off.
    fn g1822_remove_border_cost(&mut self, hi: usize, entities: &[String]) -> Result<(i32, Vec<String>), GameError> {
        let hex_id = self.hexes[hi].id.clone();
        let borders = self.g().borders.get(&hex_id).cloned().unwrap_or_default();
        let exits: Vec<u8> = self.hexes[hi]
            .tile
            .paths
            .iter()
            .flat_map(|p| [p.a.edge_num(), p.b.edge_num()])
            .flatten()
            .collect();
        let neighbors = self.g1822_neighbors();
        let mut total = 0;
        let mut types = Vec::new();
        for (edge, cost, ty) in borders {
            let Some(&nb) = neighbors.get(&hi).and_then(|m| m.get(&edge)) else { continue };
            if !exits.contains(&edge) {
                continue;
            }
            self.gm().borders.get_mut(&hex_id).unwrap().retain(|b| b.0 != edge);
            let nb_id = self.hexes[nb].id.clone();
            let n_edge = (edge + 3) % 6;
            let n_exits: Vec<u8> = self.hexes[nb]
                .tile
                .paths
                .iter()
                .flat_map(|p| [p.a.edge_num(), p.b.edge_num()])
                .flatten()
                .collect();
            if n_exits.contains(&n_edge) {
                if let Some(v) = self.gm().borders.get_mut(&nb_id) {
                    v.retain(|b| b.0 != n_edge);
                }
            }
            types.push(ty);
            // border_cost_discount (PNW: Seattle's water border is free)
            let discount = if self.g1822_is_pnw() && hex_id == "H11" { 75 } else { 0 };
            total += cost - discount;
        }
        // P10 must cross an estuary; P21 must point into the Humber
        // (Ruby raises these even while loading).
        let f = self.fam();
        if entities.iter().any(|e| Family::is(f.company_gswr, e)) {
            if total == 0 {
                return err("Must lay tile with one path over an estuary crossing");
            }
            total -= f.company_gswr_discount;
        }
        if entities.first().map_or(false, |e| Family::is(f.company_hsbc, e)) && total == 0 {
            return err("Must lay tile with one path over the Hull / Grimsby estuary");
        }
        Ok((total, types))
    }

    /// The cost `g1822_lay_tile` would charge a corporation's OWN lay (no
    /// company abilities) of a tile whose exits are `new_exits` on `hi`,
    /// without laying it: terrain upgrades (with discounts) + the water
    /// borders the tile crosses − the terrain-less hex discounts.
    pub(crate) fn g1822_corp_lay_cost(
        &self,
        spender: &str,
        hi: usize,
        new_exits: &[u8],
        neighbors: &std::collections::HashMap<usize, std::collections::BTreeMap<u8, usize>>,
    ) -> i32 {
        let hex_id = &self.hexes[hi].id;
        let old_cost = self.g1822_upgrade_cost(hi, spender);
        let border: i32 = self
            .g()
            .borders
            .get(hex_id)
            .map(|v| {
                v.iter()
                    .filter(|(edge, _, _)| {
                        neighbors.get(&hi).map_or(false, |m| m.contains_key(edge)) && new_exits.contains(edge)
                    })
                    .map(|(_, cost, _)| *cost)
                    .sum()
            })
            .unwrap_or(0);
        let base = old_cost + border;
        let d: i32 = self
            .companies_of(&EntityId::corporation(spender))
            .iter()
            .flat_map(|co| self.g1822_abilities(co))
            .filter(|a| {
                a.kind == AbKind::TileDiscount && a.terrain.is_none() && (a.hexes.is_empty() || a.hexes.contains(hex_id))
            })
            .map(|a| a.discount)
            .sum();
        if d == 0 {
            base
        } else {
            base - base.min(d)
        }
    }

    /// Ruby Hex#lay: city map, reservations then tokens move to the new tile.
    fn g1822_hex_lay(&mut self, hi: usize, def: &crate::tiles::TileDef, rotation: u8, instance: &str) -> Result<(), GameError> {
        let hex_id = self.hexes[hi].id.clone();
        let old = self.hexes[hi].tile.clone();
        let mut new_tile = BaseGame::tile_from_def(def, rotation);
        new_tile.id = instance.to_string();
        new_tile.name = instance.to_string();
        let n_old = old.cities.len();
        let n_new = new_tile.cities.len();
        // city_map_for
        let map = BaseGame::g1822_city_map(&old, &new_tile);
        // first tokens on this hex follow their city
        let homes: Vec<(String, usize)> = self
            .g()
            .home_city
            .iter()
            .filter(|(_, (h, _))| *h == hex_id)
            .map(|(c, (_, ci))| (c.clone(), *ci))
            .collect();
        for (c, ci) in homes {
            if let Some(Some(nc)) = map.get(ci) {
                self.gm().home_city.insert(c, (hex_id.clone(), *nc));
            }
        }
        // 1822MX: the NdeM tokens up for auction follow their city
        if let Some(a) = self.gm().round.ndem_auction.as_mut() {
            for t in a.available.iter_mut().filter(|t| t.hex == hex_id) {
                if let Some(Some(nc)) = map.get(t.city) {
                    t.city = *nc;
                }
            }
        }
        // reservations: new_city.reservations.concat(old's); old cleared.
        let mut new_res: Vec<Vec<Option<String>>> = vec![Vec::new(); n_new];
        for (oi, m) in map.iter().enumerate() {
            let key = format!("{hex_id}#{oi}");
            let old_res = self.gm().city_reservations.remove(&key).unwrap_or_default();
            if let Some(nc) = m {
                new_res[*nc].extend(old_res);
            }
        }
        for (nc, r) in new_res.iter().enumerate() {
            if !r.is_empty() {
                self.gm().city_reservations.insert(format!("{hex_id}#{nc}"), r.clone());
            }
        }
        // tokens
        self.hexes[hi].tile = new_tile;
        let get_slot = |city: &City, res: &[Option<String>], corp: &str, cheater: bool| -> usize {
            let reservation = res.iter().position(|r| r.as_deref() == Some(corp));
            let open = (0..city.tokens.len()).find(|&i| city.tokens[i].is_none() && res.get(i).map_or(true, |r| r.is_none()));
            if cheater {
                return open.unwrap_or(city.tokens.len());
            }
            reservation.or(open).unwrap_or(city.tokens.len())
        };
        if n_old == 1 && map.iter().flatten().count() == 1 {
            let nc = map[0].unwrap();
            for (idx, t) in old.cities[0].tokens.iter().enumerate() {
                let Some(t) = t else { continue };
                let cheater = idx >= old.cities[0].slots as usize;
                let res = new_res[nc].clone();
                let city = &mut self.hexes[hi].tile.cities[nc];
                let slot = get_slot(city, &res, &t.corporation_id, cheater);
                while city.tokens.len() <= slot {
                    city.tokens.push(None);
                }
                city.tokens[slot] = Some(t.clone());
            }
        } else if n_old > 1 {
            let mut per_new: Vec<Vec<(crate::entities::Token, bool)>> = vec![Vec::new(); n_new];
            for (oi, m) in map.iter().enumerate() {
                let toks: Vec<(crate::entities::Token, bool)> = old.cities[oi]
                    .tokens
                    .iter()
                    .enumerate()
                    .filter_map(|(i, t)| t.clone().map(|t| (t, i >= old.cities[oi].slots as usize)))
                    .collect();
                match m {
                    Some(nc) => per_new[*nc].extend(toks),
                    None if toks.is_empty() => {}
                    None => return err(format!("No city found on new tile for tokens on {hex_id}")),
                }
            }
            for (nc, mut toks) in per_new.into_iter().enumerate() {
                // (Ruby's sort of the cheaters to the end is a no-op —
                // `transform_values` builds a new hash nobody reads; the
                // variants with token identity follow that)
                if !self.g1822_token_identity() {
                    toks.sort_by_key(|(_, ch)| *ch);
                }
                let reserved = new_res[nc].len();
                for (index, (t, _)) in toks.into_iter().enumerate() {
                    let res = new_res[nc].clone();
                    let city = &mut self.hexes[hi].tile.cities[nc];
                    let cheater = reserved + index >= city.slots as usize;
                    let slot = get_slot(city, &res, &t.corporation_id, cheater);
                    while city.tokens.len() <= slot {
                        city.tokens.push(None);
                    }
                    city.tokens[slot] = Some(t);
                }
            }
        }
        self.clear_graph_cache();
        // Tracker#lay_tile: clear_graph_for_entity right after Hex#lay
        self.g1822_ruby_graph_clear();
        Ok(())
    }

    // -- SpecialTrack ---------------------------------------------------------

    pub(crate) fn g1822_special_track_process(&mut self, _i: usize, ent: &Ent, action: &Action) -> Result<(), GameError> {
        let Ent::Company(co) = ent else { return err("special track by a non-company") };
        let co = co.clone();
        match action {
            Action::LayTile { hex_id, tile_id, rotation, .. } => {
                if self.g1822_is_pnw() {
                    if let Some(r) = self.g1822_pnw_special_lay(&co, hex_id, tile_id, *rotation) {
                        return r;
                    }
                }
                if self.g1822_is_mx() {
                    if let Some(r) = self.g1822_mx_special_cube(&co, hex_id, tile_id) {
                        return r;
                    }
                }
                let (ab_idx, ab) = self
                    .g1822_track_ability(&co)
                    .ok_or_else(|| GameError::new(format!("{co} has no tile lay ability")))?;
                let spender = self
                    .co_owner(&co)
                    .corp_sym()
                    .map(str::to_string)
                    .ok_or_else(|| GameError::new("company not corporation-owned"))?;
                let combo = self.g().pending_combo.clone();
                // legal_tile_rotation? walks the operator's graph (not for
                // P5's Channel rotation rule).
                if !Family::is(self.fam().company_lcdr, &co) {
                    self.g1822_note_graph(&spender);
                }
                let tile_color = {
                    let base = tile_id.split('-').next().unwrap_or(tile_id);
                    self.tile_catalog.get(base).map(|d| d.color)
                };
                if self.is_minor(&spender)
                    && matches!(tile_color, Some(crate::tiles::TileColor::Brown) | Some(crate::tiles::TileColor::Gray))
                {
                    return err("Minor cannot upgrade past green");
                }
                self.rndm().special_track_in_process = true;
                let mut minor_single_use = false;
                let hi = *self.hex_idx.get(hex_id.as_str()).ok_or_else(|| GameError::new("unknown hex"))?;
                let pnw_forest = self.g1822_is_pnw() && self.g1822_pnw_forest_tile(hi);
                let upgrade = self.hexes[hi].tile.color != crate::tiles::TileColor::White;
                let extra_track =
            self.g1822_company_ability_extra_track(&co) || combo.iter().any(|c| self.g1822_company_ability_extra_track(c));
                let res = if extra_track {
                    let consume = ab.consume_tile_lay;
                    if upgrade && self.rnd().extra_laided_track && consume {
                        Err(GameError::new("Cannot lay an extra upgrade"))
                    } else {
                        let r = self.g1822_lay_tile(&co, &spender, &combo, hi, tile_id, *rotation);
                        if r.is_ok() {
                            self.rndm().laid_hexes.push(hex_id.clone());
                            if self.is_minor(&spender) {
                                minor_single_use = true;
                            } else {
                                self.rndm().extra_laided_track = true;
                            }
                        }
                        r
                    }
                } else {
                    let hsbc = Family::is(self.fam().company_hsbc, &co);
                    self.g1822_lay_tile_action(&co, &spender, &combo, hex_id, tile_id, *rotation, hsbc)
                };
                self.rndm().special_track_in_process = false;
                res?;
                // use the ability (+ combo abilities)
                let up = matches!(
                    tile_color,
                    Some(crate::tiles::TileColor::Green) | Some(crate::tiles::TileColor::Brown) | Some(crate::tiles::TileColor::Gray)
                );
                let mut users: Vec<(String, usize)> = vec![(co.clone(), ab_idx)];
                for c in &combo {
                    if let Some((i, _)) = self.g1822_track_ability(c) {
                        users.push((c.clone(), i));
                    }
                }
                // active_abilities.each: use!, close when used up, then
                // handle_extra_tile_lay_company — per ability, with ITS OWN
                // flags; the last ability processed decides @company (a
                // combo partner without must_lay_together clears it: 232041,
                // P12+P10 returns to the Track step).
                for (c, idx) in users {
                    let this = self.g1822_abilities(&c).get(idx).cloned();
                    self.g1822_use_ability(&c, idx, up)?;
                    if minor_single_use {
                        self.g1822_use_ability(&c, idx, up)?;
                    }
                    let Some(this) = this else { continue };
                    if this.kind != AbKind::TileLay {
                        continue;
                    }
                    let count_left = self.g1822_abilities(&c).get(idx).and_then(|a| a.count).unwrap_or(0);
                    if this.closed_when_used_up && count_left <= 0 {
                        self.close_company(&c);
                    }
                    if this.must_lay_together {
                        if self.rnd().num_laid_track == 1 && !this.consume_tile_lay {
                            self.rndm().num_laid_track += 1;
                        }
                        self.rndm().special_track_company =
                            if count_left > 0 && !self.co_closed(&c) { Some(c.clone()) } else { None };
                    } else {
                        self.rndm().special_track_company = None;
                    }
                }
                if self.g1822_is_mx() {
                    // MX SpecialTrack#process_lay_tile after super
                    self.g1822_pnw_drop_cubes(hex_id.as_str());
                }
                if self.g1822_is_pnw() {
                    // PNW SpecialTrack#process_lay_tile after super
                    self.g1822_pnw_drop_cubes(hex_id.as_str());
                    if pnw_forest {
                        self.g1822_pnw_assign(hex_id, "forest");
                    }
                }
                if ab.kind == AbKind::Teleport {
                    // the teleport is used; its token step follows.
                    if let Some(v) = self.gm().abilities.get_mut(&co) {
                        if let Some(a) = v.get_mut(ab_idx) {
                            a.used = true;
                        }
                    }
                    self.rndm().teleported = Some(co.clone());
                }
                self.gm().pending_combo.clear();
                Ok(())
            }
            Action::Pass { .. } => {
                if self.rnd().special_track_company.as_deref() != Some(co.as_str()) {
                    return err(format!("Not {co}'s turn"));
                }
                // remove the ability; @company = nil; the company closes.
                if let Some(v) = self.gm().abilities.get_mut(&co) {
                    v.retain(|a| a.kind != AbKind::TileLay);
                }
                self.rndm().special_track_company = None;
                self.close_company(&co);
                Ok(())
            }
            _ => err("SpecialTrack cannot process this action"),
        }
    }

    /// Ability#use! (TileLay with lay/upgrade counts).
    pub(crate) fn g1822_use_ability(&mut self, co: &str, idx: usize, upgrade: bool) -> Result<(), GameError> {
        let Some(v) = self.gm().abilities.get_mut(co) else { return Ok(()) };
        let Some(a) = v.get_mut(idx) else { return Ok(()) };
        if a.count.map_or(false, |c| c <= 0) {
            return Ok(());
        }
        a.used = true;
        if let Some(c) = a.count.as_mut() {
            *c -= 1;
        }
        if let (Some(_), Some(_)) = (a.upgrade_count, a.lay_count) {
            if upgrade {
                let uc = a.upgrade_count.unwrap();
                if uc <= 0 {
                    return err("Cannot use this ability to upgrade a tile now");
                }
                a.lay_count = Some(0);
                a.upgrade_count = Some(uc - 1);
                if uc - 1 <= 0 {
                    a.count = Some(0);
                }
            } else {
                let lc = a.lay_count.unwrap();
                if lc <= 0 {
                    return err("Cannot use this ability to lay a tile now");
                }
                a.upgrade_count = Some(0);
                a.lay_count = Some(lc - 1);
                if lc - 1 <= 0 {
                    a.count = Some(0);
                }
            }
        }
        Ok(())
    }
}
