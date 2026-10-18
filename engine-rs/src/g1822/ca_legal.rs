//! 1822CA legal-move generation: the CA steps and abilities the shared
//! generator (legal.rs) does not cover.

use crate::actions::Action;
use crate::game::BaseGame;

use super::ca::{SawmillState, COMPANY_SAWMILL, COMPANY_WINNIPEG_TOKEN, SAWMILL_HEXES};
use super::ca_route::{CaChoosing, CHOICE_SKIP};
use super::legal::Move;

impl BaseGame {
    /// AssignSawmill#choices once assigned: open / closed (the operator).
    pub(crate) fn gen_ca_sawmill_choices(&self) -> Vec<Move> {
        if self.rnd().ca.sawmill_state != SawmillState::Assigned {
            return Vec::new();
        }
        let Some(op) = self.g1822_operator() else { return Vec::new() };
        ["open", "closed"]
            .iter()
            .map(|c| Move::plain(Action::Choose { entity_id: op.clone(), choice: c.to_string() }))
            .collect()
    }

    /// AssignSawmill#available_hex: P13's hexes not yet assigned that still
    /// hold a city or a town (not one removed by P29 / P30).
    pub(crate) fn gen_ca_assigns(&self, co: &str) -> Vec<Move> {
        if co != COMPANY_SAWMILL {
            return Vec::new();
        }
        SAWMILL_HEXES
            .iter()
            .filter_map(|h| self.hex_idx.get(*h).map(|&hi| (h, hi)))
            .filter(|(h, hi)| {
                let t = &self.hexes[*hi].tile;
                !self.g1822_pnw_assigned(h, co) && (!t.cities.is_empty() || !t.towns.is_empty())
            })
            .map(|(h, _)| {
                Move::plain(Action::Assign {
                    entity_id: co.to_string(),
                    target: h.to_string(),
                    target_type: "hex".to_string(),
                })
            })
            .collect()
    }

    /// P10's Winnipeg token: any city of N16 its owner has no token in.
    pub(crate) fn gen_ca_special_tokens(&self, co: &str) -> Vec<Move> {
        if co != COMPANY_WINNIPEG_TOKEN {
            return Vec::new();
        }
        let Some(owner) = self.g1822_ca_p10_owner() else { return Vec::new() };
        let Some(&hi) = self.hex_idx.get("N16") else { return Vec::new() };
        (0..self.hexes[hi].tile.cities.len())
            .filter(|&ci| !self.g1822_city_tokened_by(hi, ci, &owner))
            .map(|ci| {
                Move::plain(Action::PlaceToken {
                    entity_id: co.to_string(),
                    hex_id: "N16".to_string(),
                    city_index: ci as u8,
                })
            })
            .collect()
    }

    /// CA DestinationToken: connected to one destination city → the hex
    /// token (a single-city destination) or that city's token; to several →
    /// one per city; to none → pass (pass is illegal once connected).
    pub(crate) fn gen_ca_destination(&self, c: &str) -> Vec<Move> {
        let connected = self.g1822_ca_connected_destination_cities(c);
        let Some(dest) = self.g1822_destination(c) else { return Vec::new() };
        let Some(&hi) = self.hex_idx.get(dest.as_str()) else { return Vec::new() };
        if connected.is_empty() {
            return vec![Move::plain(Action::Pass { entity_id: c.to_string() })];
        }
        if connected.len() == 1 && self.g1822_ca_destination_cities(c, hi).len() == 1 {
            return vec![Move::plain(Action::HexToken {
                entity_id: c.to_string(),
                hex_id: dest,
                token_type: Some("destination".to_string()),
            })];
        }
        connected
            .into_iter()
            .map(|ci| Move::plain(Action::PlaceToken { entity_id: c.to_string(), hex_id: dest.clone(), city_index: ci as u8 }))
            .collect()
    }

    /// CA Route#choices: "None" and each train the pullman / grain train
    /// can attach to.
    pub(crate) fn gen_ca_route_choices(&self, c: &str) -> Vec<Move> {
        let n = match self.g1822_ca_choosing(c) {
            Some(CaChoosing::Pullman) => self.g1822_pullman_choices(c).len(),
            Some(CaChoosing::Grain) => self.g1822_ca_grain_choices(c).len(),
            None => return Vec::new(),
        };
        let mut out = vec![Move::plain(Action::Choose { entity_id: c.to_string(), choice: CHOICE_SKIP.to_string() })];
        out.extend((0..n).map(|i| Move::plain(Action::Choose { entity_id: c.to_string(), choice: i.to_string() })));
        out
    }

    /// CA SpecialTrack#available_hex's own rules: never the sawmill hex for
    /// a town remover; a big-city private only on its city (connected),
    /// whatever the operator's lay slot. Some(skip?) for those cases.
    pub(crate) fn g1822_ca_special_hex(&self, entities: &[String], hi: usize) -> Option<bool> {
        if !self.g1822_is_ca() {
            return None;
        }
        let hex = &self.hexes[hi].id;
        if self.g().ca.sawmill_hex.as_deref() == Some(hex.as_str()) {
            return if entities.iter().any(|e| self.g1822_must_remove_town(e)) { Some(true) } else { None };
        }
        let co = entities.first()?;
        if super::ca::BIG_CITY_HEXES_TO_COMPANIES.iter().any(|(_, c)| c == co) {
            let mine = super::ca::BIG_CITY_HEXES_TO_COMPANIES.iter().any(|(h, c)| c == co && h == hex);
            return Some(!mine);
        }
        None
    }
}
