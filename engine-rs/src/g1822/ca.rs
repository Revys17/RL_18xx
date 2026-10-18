//! 1822CA (The Railways of Canada): the rules Ruby's `G1822CA::Game`
//! overrides (lib/engine/game/g_1822_ca/), on top of the G1822 step machine.
//!
//! Each function names the Ruby method it ports.

use serde::{Deserialize, Serialize};

use crate::actions::{Action, GameError};
use crate::game::BaseGame;
use crate::tiles::TileColor;

use super::{err, Ent};

/// 1822CA game state.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct CaState {
    /// `@sawmill_hex` / `@sawmill_owner` (set only for a closed sawmill) /
    /// `@sawmill_bonus`.
    pub sawmill_hex: Option<String>,
    pub sawmill_owner: Option<String>,
    pub sawmill_bonus: i32,
    /// `@destinated` (the corporations whose destination token is down).
    pub destinated: Vec<String>,
}

/// AssignSawmill `@sawmill_state`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum SawmillState {
    #[default]
    None,
    Assigned,
    Chosen,
}

/// 1822CA per-round step state.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct CaRound {
    /// AssignSawmill `@sawmill_state` and its `@sawmill_owner` (the
    /// assigning corporation).
    pub sawmill_state: SawmillState,
    pub sawmill_assigner: Option<String>,
    /// MinorAcquisition `@round.acquiring_major`.
    pub acquiring_major: Option<String>,
    /// Route: the grain / pullman choices skipped this turn, the train the
    /// grain train is attached to (`@grain_train`, a train id) and its
    /// original name.
    pub skipped_pullman: bool,
    pub skipped_grain: bool,
    pub grain_train: Option<String>,
    pub grain_original: Option<String>,
}

/// The CORPORATIONS' `destination_exits`.
pub fn destination_exits(corp: &str) -> Option<&'static [u8]> {
    Some(match corp {
        "GNWR" => &[3],
        "GT" => &[0],
        "ICR" => &[0, 1, 2, 3, 4, 5],
        "NTR" => &[5],
        _ => return None,
    })
}

/// BIG_CITY_ILLEGAL_TILE_UPGRADES (from => to).
pub const BIG_CITY_ILLEGAL_TILE_UPGRADES: &[(&str, &str)] =
    &[("M2", "M6"), ("M3", "M4"), ("O2", "O3"), ("O4", "O5"), ("Q4", "Q5"), ("T5", "T6"), ("W5", "W6")];

/// MOUNTAIN_PASS_HEXES / _TILES / _HEXES_TO_COMPANIES.
pub const MOUNTAIN_PASS_HEXES: &[&str] = &["E11", "F16"];
pub const MOUNTAIN_PASS_TILES: &[&str] = &["7", "8", "9"];
pub const MOUNTAIN_PASS_HEXES_TO_COMPANIES: &[(&str, &str)] = &[("E11", "P20"), ("F16", "P19")];

/// BIG_CITY_HEXES_TO_COMPANIES.
pub const BIG_CITY_HEXES_TO_COMPANIES: &[(&str, &str)] =
    &[("AH8", "P17"), ("N16", "P18"), ("AC21", "P14"), ("AE15", "P15"), ("AF12", "P16")];

/// COMPANIES_CONSUME_TILE_LAY.
pub const COMPANIES_CONSUME_TILE_LAY: &[&str] = &["P19", "P20", "P29", "P30"];

/// GRAIN_ELEVATORS, GRAIN_ELEVATOR_COUNT, WEST_PORTS, EAST_PORTS.
pub const GRAIN_ELEVATORS: &[&str] = &["G13", "H16", "I11", "I15", "I17", "J16", "K13", "K17", "L16", "M17"];
pub const GRAIN_ELEVATOR_COUNT: u32 = 12;
pub const WEST_PORTS: &[&str] = &["A7", "C15"];
pub const EAST_PORTS: &[&str] = &["N6", "R16"];

/// DETROIT_TO_DULUTH_HEXES.
pub const DETROIT_TO_DULUTH_HEXES: &[&str] = &["Q19", "Y27"];

/// PRIVATE_SMALL_MAIL_CONTRACTS.
pub const PRIVATE_SMALL_MAIL_CONTRACTS: &[&str] = &["P24", "P25"];

/// COMPANY_WINNIPEG_TOKEN.
pub const COMPANY_WINNIPEG_TOKEN: &str = "P10";

/// The sawmill company (P13's assign_hexes ability).
pub const COMPANY_SAWMILL: &str = "P13";

/// QMOO's home hex (`quebec_hex`).
pub const QUEBEC_HEX: &str = "AH8";

/// `legal_tile_rotation?`'s big-city rotation table.
fn big_city_rotations(tile: &str) -> Option<&'static [u8]> {
    Some(match tile {
        "M5" | "M6" | "M8" | "O2" | "O6" | "O8" | "Q6" | "Q8" | "T1" | "T2" | "T3" | "T4" | "T5" | "W2" | "W4" | "W5"
        | "W6" | "W7" => &[0],
        "M1" | "M2" | "M3" | "M4" | "O7" => &[0, 5],
        "M7" | "O1" | "O3" | "O4" | "O5" => &[0, 1],
        "Q1" | "Q2" | "Q3" | "Q4" | "Q5" | "Q7" => &[0, 3, 5],
        "W3" => &[0, 2, 3],
        _ => return None,
    })
}

impl BaseGame {
    pub(crate) fn g1822_is_ca(&self) -> bool {
        self.g1822.is_some() && self.fam().variant == super::family::Variant::Ca
    }

    /// CA `upgrades_to?` before G1822's: the white AG13 (a city and a town)
    /// takes 5, 6 or 57 only; BIG_CITY_ILLEGAL_TILE_UPGRADES are refused.
    pub(crate) fn g1822_ca_upgrades_to(&self, hi: usize, to_base: &str) -> Option<bool> {
        let hex = &self.hexes[hi];
        if hex.id == "AG13" && hex.tile.color == TileColor::White {
            return Some(matches!(to_base, "5" | "6" | "57"));
        }
        let from = hex.tile.name.split('-').next().unwrap_or(&hex.tile.name);
        if BIG_CITY_ILLEGAL_TILE_UPGRADES.iter().any(|(f, t)| *f == from && *t == to_base) {
            return Some(false);
        }
        None
    }

    /// CA `upgrades_to_correct_label?`: a mountain pass hex (its label YP
    /// / CP) also takes the plain 7, 8 and 9.
    pub(crate) fn g1822_ca_mountain_pass_label(&self, hi: usize, to_base: &str) -> bool {
        self.g1822_is_ca()
            && MOUNTAIN_PASS_HEXES.contains(&self.hexes[hi].id.as_str())
            && MOUNTAIN_PASS_TILES.contains(&to_base)
    }

    /// CA `legal_tile_rotation?`'s big-city rotation table (the game-level
    /// check, before Tracker's).
    pub(crate) fn g1822_ca_rotation_allowed(&self, to_base: &str, rotation: u8) -> bool {
        !self.g1822_is_ca() || big_city_rotations(to_base).map_or(true, |r| r.contains(&rotation))
    }

    /// CA Tracker `old_paths_maintained?`: the white AG13 → 5 / 6 / 57
    /// drops its town, so any of those keeps the old paths.
    pub(crate) fn g1822_ca_old_paths_override(&self, hi: usize, to_base: &str) -> Option<bool> {
        let hex = &self.hexes[hi];
        (self.g1822_is_ca() && hex.id == "AG13" && hex.tile.color == TileColor::White)
            .then(|| matches!(to_base, "5" | "6" | "57"))
    }
}

/// P13's assign_hexes ability hexes.
pub const SAWMILL_HEXES: &[&str] = &[
    "A9", "AA15", "AA9", "AB22", "AD18", "AD20", "AE17", "AF16", "AG13", "AG3", "AG9", "AH10", "AH12", "AK3", "AK5",
    "AL2", "AM5", "AM9", "AO9", "AP6", "C9", "D10", "D12", "D14", "D16", "F10", "F8", "G11", "G13", "G17", "H16", "H8",
    "I11", "I15", "I17", "J12", "J16", "K11", "K13", "K17", "K9", "L10", "L16", "L8", "M17", "M9", "N6", "O15", "O9",
    "P14", "Q7", "S15", "U11", "W9", "X12", "Y17", "Z10", "Z26",
];

impl BaseGame {
    // -- AssignSawmill --------------------------------------------------------

    /// AssignSawmill#actions: `choose` (any entity) once assigned, else
    /// Assign#actions — P13's assign_hexes ability (owner_type corporation,
    /// when owning_corp_or_turn: its corporation is operating).
    pub(crate) fn g1822_ca_sawmill_actions(&self, ent: &Ent) -> Vec<&'static str> {
        if self.rnd().ca.sawmill_state == SawmillState::Assigned {
            return vec!["choose"];
        }
        let Ent::Company(co) = ent else { return vec![] };
        if co != COMPANY_SAWMILL || self.co_closed(co) {
            return vec![];
        }
        match (self.co_owner(co).corp_sym(), self.g1822_operator()) {
            (Some(o), Some(op)) if o == op => vec!["assign"],
            _ => vec![],
        }
    }

    /// AssignSawmill#process_assign / #process_choose.
    pub(crate) fn g1822_ca_sawmill_process(&mut self, i: usize, ent: &Ent, action: &Action) -> Result<(), GameError> {
        match action {
            Action::Assign { target, target_type, .. } => {
                let Ent::Company(co) = ent else { return err("assign by a non-company") };
                if target_type != "hex" {
                    return err(format!("Invalid target {target} for assigning company {co}"));
                }
                if !self.hex_idx.contains_key(target.as_str()) {
                    return err(format!("Unknown hex {target}"));
                }
                if self.g1822_pnw_assigned(target, co) {
                    return err(format!("{co} is already assigned to {target}"));
                }
                let owner = self.co_owner(co).corp_sym().map(str::to_string);
                let co = co.clone();
                let r = self.rndm();
                r.ca.sawmill_assigner = owner;
                r.ca.sawmill_state = SawmillState::Assigned;
                self.gm().ca.sawmill_hex = Some(target.clone());
                // Assign#process_assign: assign!, use! (count 1), close
                self.g1822_pnw_assign(target, &co);
                self.close_company(&co);
                Ok(())
            }
            Action::Choose { choice, .. } => {
                if self.rnd().ca.sawmill_state != SawmillState::Assigned {
                    return err("Cannot choose if Sawmill is not assigned");
                }
                match choice.as_str() {
                    "open" => self.gm().ca.sawmill_bonus = 20,
                    "closed" => {
                        self.gm().ca.sawmill_bonus = 10;
                        let a = self.rnd().ca.sawmill_assigner.clone();
                        self.gm().ca.sawmill_owner = a;
                    }
                    other => return err(format!("Invalid choice for Sawmill: {other}")),
                }
                self.rndm().ca.sawmill_state = SawmillState::Chosen;
                self.rndm().passed[i] = true;
                Ok(())
            }
            _ => err("AssignSawmill cannot process this action"),
        }
    }

    // -- MinorAcquisition / AcquisitionTrack ----------------------------------

    /// CA MinorAcquisition#after_acquire_*_minor.
    pub(crate) fn g1822_ca_after_acquire_minor(&mut self, major: &str) {
        if self.g1822_is_ca() {
            self.rndm().ca.acquiring_major = Some(major.to_string());
        }
    }

    /// The AcquisitionTrack step is the one acting: MinorAcquisition is
    /// done for the acquiring operator and the bonus lay not yet taken.
    /// (Evaluated from the steps' flags: asking for the active step from
    /// the tile-lay rules would recurse.)
    pub(crate) fn g1822_ca_acquisition_track_live(&self) -> bool {
        if !self.g1822_is_ca() {
            return false;
        }
        let r = self.rnd();
        let (Some(ma), Some(at)) = (r.step_index(super::Step::MinorAcquisition), r.step_index(super::Step::AcquisitionTrack))
        else {
            return false;
        };
        r.passed[ma] && !r.passed[at] && r.ca.acquiring_major.is_some() && r.ca.acquiring_major == self.g1822_operator()
    }

    /// AcquisitionTrack#actions: the acquiring major lays or passes.
    pub(crate) fn g1822_ca_acquisition_track_actions(&self, ent: &Ent) -> Vec<&'static str> {
        let Ent::Corp(c) = ent else { return vec![] };
        if Some(c) != self.g1822_operator().as_ref() || self.rnd().ca.acquiring_major.as_ref() != Some(c) {
            return vec![];
        }
        vec!["lay_tile", "pass"]
    }

    /// CA `setup_game_specific` (the parts with state): `block_detroit_duluth`
    /// — the DD blocker's two tokens fill Q19's and Y27's cities.
    pub(crate) fn g1822_ca_setup(&mut self) {
        let dd = crate::title::g1822ca::DD;
        for (n, hex) in DETROIT_TO_DULUTH_HEXES.iter().enumerate() {
            let Some(&hi) = self.hex_idx.get(*hex) else { continue };
            let _ = self.g1822_put_token(dd, n, hi, 0, false);
        }
        self.gm().home_city.remove(dd);
    }

    /// `event_open_detroit_duluth!`: the blocker's tokens are destroyed and
    /// the DD tiles laid on Q19 and Y27 (Hex#lay only: no tile supply).
    pub(crate) fn g1822_ca_event_open_detroit_duluth(&mut self) {
        let dd = crate::title::g1822ca::DD;
        for hex in DETROIT_TO_DULUTH_HEXES {
            let Some(&hi) = self.hex_idx.get(*hex) else { continue };
            for city in &mut self.hexes[hi].tile.cities {
                for t in city.tokens.iter_mut() {
                    if t.as_ref().is_some_and(|t| t.corporation_id == dd) {
                        *t = None;
                    }
                }
            }
        }
        // (Token#destroy! deletes them from the blocker's list)
        let ci = self.ci(dd);
        self.corporations[ci].tokens.clear();
        for hex in DETROIT_TO_DULUTH_HEXES {
            let Some(&hi) = self.hex_idx.get(*hex) else { continue };
            let name = format!("DD-{hex}");
            let Some(def) = self.tile_catalog.get(&name).cloned() else { continue };
            let _ = self.g1822_hex_lay(hi, &def, 0, &format!("{name}-0"));
        }
        self.clear_graph_cache();
        self.g1822_ruby_graph_clear();
    }
}
