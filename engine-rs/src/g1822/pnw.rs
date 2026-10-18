//! 1822PNW (`G1822PNW::Game < G1822::Game`): the methods PNW overrides or
//! adds, for the step machine in `crate::g1822` to branch into
//! (`BaseGame::variant() == Variant::Pnw`). Ruby file references are
//! `lib/engine/game/g_1822_pnw/…`.

use crate::actions::{Action, GameError};
use crate::entities::EntityId;
use crate::game::BaseGame;
use crate::title::CorpType;

use super::graph::{Node, NodeKind};
use super::shares::Bundle;
use super::{err, Ent, MergeState, RoundKind, RoundSt, RubyRand, Step};

/// `setup_associated_minors`: minor → major (Ruby hash order;
/// `replace_associated_minor` moves an entry to the end).
pub const ASSOCIATIONS: &[(&str, &str)] = &[
    ("1", "CPR"),
    ("5", "GNR"),
    ("7", "CMPS"),
    ("8", "SWW"),
    ("17", "SPS"),
    ("18", "ORNC"),
    ("20", "NP"),
];

/// `setup_tokencity_tiles`: Portland and Seattle's tile sequences.
pub const TOKENCITY: &[(&str, &[&str])] = &[("O8", &["X24", "X25", "X26", "X27"]), ("H11", &["X20", "X21", "X22", "X23"])];

/// `@empty_bidbox_minor`'s sym (a placeholder in an emptied bid box).
pub const EMPTY_BIDBOX: &str = "EMPTY";

/// `major_formation_status`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Formation {
    Formed,
    Convertable,
    Parable,
    None,
}

/// The regional railways (`regional_railway?`) and their companies.
pub const REGIONALS: &[&str] = &["A", "B", "C"];
pub const REGIONAL_COMPANIES: &[&str] = &["MA", "MB", "MC"];

/// game.rb `setup_starting_packets`: per seat, (minors, privates).
pub const STARTING_PACKETS: &[(&[&str], &[&str])] = &[
    (&["M18", "M21"], &["P8", "P16"]),
    (&["M7", "M9"], &["P18", "P19"]),
    (&["M1", "M2"], &["P12", "P17"]),
    (&["M6", "M8"], &["P14", "P15"]),
    (&["M10", "M20"], &["P10", "P11"]),
];

/// `sort_by! { rand }` (Ruby's game LCG; keys are distinct in practice,
/// ties keep the input order).
fn shuffle(list: Vec<String>, rng: &mut RubyRand) -> Vec<String> {
    let mut keyed: Vec<(u64, usize, String)> = list.into_iter().enumerate().map(|(i, s)| (rng.next(), i, s)).collect();
    keyed.sort_by_key(|(k, i, _)| (*k, *i));
    keyed.into_iter().map(|(_, _, s)| s).collect()
}

impl BaseGame {
    pub(crate) fn g1822_is_pnw(&self) -> bool {
        self.g1822.is_some() && self.variant() == super::family::Variant::Pnw
    }

    /// `starting_packet?`.
    pub(crate) fn g1822_pnw_starting_packet(&self) -> bool {
        self.optional_rules.iter().any(|r| r == "starting_packet")
    }

    /// PNW `setup_companies` after `@companies.sort_by! { rand }` (the
    /// `shuffled` input): P1 first among the privates; minors as M6, then
    /// three shuffled stacks of associated + unassociated minors (just the
    /// minors under the starting packet); then the privates.
    pub(crate) fn g1822_pnw_order_companies(&self, shuffled: &[String], rng: &mut RubyRand) -> Vec<String> {
        let mut privates: Vec<String> = shuffled.iter().filter(|s| s.starts_with('P')).cloned().collect();
        if let Some(pos) = privates.iter().position(|s| s == "P1") {
            let p1 = privates.remove(pos);
            privates.insert(0, p1);
        }
        let minors: Vec<String> = shuffled.iter().filter(|s| s.starts_with('M')).cloned().collect();
        let mut out: Vec<String> = Vec::new();
        if self.g1822_pnw_starting_packet() {
            out.extend(minors);
        } else {
            let (minor_6, minors): (Vec<String>, Vec<String>) = minors.into_iter().partition(|s| s == "M6");
            let assoc_of = |s: &String| ASSOCIATIONS.iter().any(|(m, _)| *m == &s[1..]);
            let (assoc, minors): (Vec<String>, Vec<String>) = minors.into_iter().partition(assoc_of);
            // Ruby a[i..j] (inclusive, clipped to the array)
            let slice = |v: &Vec<String>, i: usize, j: usize| -> Vec<String> {
                v.iter().skip(i).take(j + 1 - i).cloned().collect()
            };
            out.extend(minor_6);
            let mut s1 = slice(&assoc, 0, 2);
            s1.extend(slice(&minors, 0, 5));
            out.extend(shuffle(s1, rng));
            let mut s2 = slice(&assoc, 3, 4);
            s2.extend(slice(&minors, 6, 10));
            out.extend(shuffle(s2, rng));
            let mut s3 = slice(&assoc, 5, 6);
            s3.extend(slice(&minors, 11, 15));
            out.extend(shuffle(s3, rng));
        }
        out.extend(privates);
        out
    }

    /// PNW setup_optional_rules: remove_three_ls / remove_two_ls discard
    /// the depot's first L/2 trains.
    pub(crate) fn g1822_pnw_setup_optional_rules(&mut self) {
        let n = if self.optional_rules.iter().any(|r| r == "remove_three_ls") {
            3
        } else if self.optional_rules.iter().any(|r| r == "remove_two_ls") {
            2
        } else {
            0
        };
        for _ in 0..n {
            if !self.depot.trains.is_empty() {
                self.depot.trains.remove(0);
            }
        }
    }

    /// PNW Round::Stock#float_minor for a regional railway: the winner buys
    /// the company (now worth 200) and owns the regional, whose token goes
    /// to its reserved home city; its value is fixed at the 100 par.
    pub(crate) fn g1822_pnw_float_regional(&mut self, bid: &super::BidRec, company: &str) -> Result<(), GameError> {
        let minor = Self::minor_of_company(company);
        self.g1822_buy_company(bid, company)?;
        if let Some(i) = self.co_idx(company) {
            self.companies[i].value = 200;
        }
        let mi = self.ci(&minor);
        self.corporations[mi].owner_id = EntityId::player(bid.player);
        if let Some(city) = self.g1822_pnw_minor_city_reserved(&minor) {
            self.g1822_put_token(&minor, 0, city.hex, city.index, false)?;
        }
        let sp = self
            .stock_market
            .share_prices_with_types(&["par", "par_1"])
            .into_iter()
            .find(|p| p.price == 100)
            .ok_or_else(|| GameError::new("no 100 par price"))?;
        // `minor.share_price =` (no market cell entry)
        self.corporations[mi].share_price = Some(sp);
        Ok(())
    }

    /// The city of `m`'s home hex reserved for it.
    fn g1822_pnw_minor_city_reserved(&self, m: &str) -> Option<Node> {
        let home = self.title_def().corporations().into_iter().find(|d| d.sym == m)?.home_hex;
        let &hi = self.hex_idx.get(home)?;
        let n = self.hexes[hi].tile.cities.len();
        (0..n)
            .find(|&k| self.g1822_city_res(home, k).iter().flatten().any(|r| r == m))
            .map(|k| Node { hex: hi, kind: NodeKind::City, index: k })
    }

    // =====================================================================
    // Map: token cities, builder cubes, forests, blue hexes, neighbours
    // =====================================================================

    /// `tokencity_tile_index_of(hex, tile)` (-1 off the list).
    pub(crate) fn g1822_pnw_tokencity_index(hex: &str, tile_base: &str) -> i32 {
        TOKENCITY
            .iter()
            .find(|(h, _)| *h == hex)
            .and_then(|(_, list)| list.iter().position(|t| *t == tile_base))
            .map_or(-1, |i| i as i32)
    }

    /// `tokencity?(hex)`.
    pub(crate) fn g1822_pnw_tokencity(hex: &str) -> bool {
        TOKENCITY.iter().any(|(h, _)| *h == hex)
    }

    /// The colour of the tile on `hi` as Ruby names it (blue included,
    /// which the engine's `TileColor` folds into white).
    pub(crate) fn g1822_tile_color_name(&self, hi: usize) -> &'static str {
        let tile = &self.hexes[hi].tile;
        let f = self.fam();
        if tile.id.starts_with("preprinted_") || tile.name.is_empty() {
            return (f.hex_dsl)(&self.hexes[hi].id).map_or("white", |(_, c)| c);
        }
        let base = tile.name.split('-').next().unwrap_or(&tile.name);
        (f.tile_dsl)(base).map_or("white", |(_, c)| c)
    }

    /// A catalog tile's Ruby colour name.
    pub(crate) fn g1822_def_color_name(&self, base: &str) -> &'static str {
        (self.fam().tile_dsl)(base).map_or("white", |(_, c)| c)
    }

    /// `total_terrain_cost(tile)`: upgrades with a terrain.
    fn g1822_pnw_total_terrain_cost(&self, hi: usize) -> i32 {
        self.hexes[hi].tile.upgrades.iter().filter(|u| !u.terrain.is_empty()).map(|u| u.cost).sum()
    }

    /// `can_place_river(tile)`: M4 / N5 / H13 while no path leaves by the
    /// river edge first.
    fn g1822_pnw_can_place_river(&self, hi: usize) -> bool {
        let dir = match self.hexes[hi].id.as_str() {
            "M4" => 5,
            "N5" => 2,
            "H13" => 1,
            _ => return false,
        };
        !self.hexes[hi].tile.paths.iter().any(|p| {
            let first = p.a.edge_num().or(p.b.edge_num());
            first == Some(dir)
        })
    }

    /// `max_builder_cubes(tile)`.
    pub(crate) fn g1822_pnw_max_cubes(&self, hi: usize) -> u32 {
        let total = self.g1822_pnw_total_terrain_cost(hi) + if self.g1822_pnw_can_place_river(hi) { 75 } else { 0 };
        ((total as f64) / 40.0).ceil() as u32
    }

    /// `current_builder_cubes(tile)` of the tile on `hex`. Cubes are
    /// icons on the TILE (sticky, preprinted): a tile upgraded away keeps
    /// its cubes in the supply, and Hex#lay copies the old tile's cubes
    /// onto the new one only beyond those it already carries — so a
    /// re-laid tile can bring stale cubes (and their £40 discounts).
    pub(crate) fn g1822_pnw_cubes(&self, hex: &str) -> u32 {
        let Some(&hi) = self.hex_idx.get(hex) else { return 0 };
        self.g().builder_cubes.get(&self.hexes[hi].tile.id).copied().unwrap_or(0)
    }

    /// Hex#lay's sticky icons: the new tile carries max(its own cubes,
    /// the old tile's); the old tile keeps its own.
    pub(crate) fn g1822_pnw_carry_cubes(&mut self, old_tile: &str, new_tile: &str) {
        let old = self.g().builder_cubes.get(old_tile).copied().unwrap_or(0);
        let new = self.g().builder_cubes.get(new_tile).copied().unwrap_or(0);
        if old > new {
            self.gm().builder_cubes.insert(new_tile.to_string(), old);
        }
    }

    /// process_lay_tile after super: `hex.tile.icons.reject! block`.
    pub(crate) fn g1822_pnw_drop_cubes(&mut self, hex: &str) {
        let Some(&hi) = self.hex_idx.get(hex) else { return };
        let id = self.hexes[hi].tile.id.clone();
        self.gm().builder_cubes.remove(&id);
    }

    /// `can_hold_builder_cubes?(tile)`.
    pub(crate) fn g1822_pnw_can_hold_cubes(&self, hi: usize) -> bool {
        self.g1822_pnw_cubes(&self.hexes[hi].id) < self.g1822_pnw_max_cubes(hi)
    }

    pub(crate) fn g1822_pnw_place_cube(&mut self, hex: &str) {
        let Some(&hi) = self.hex_idx.get(hex) else { return };
        let id = self.hexes[hi].tile.id.clone();
        *self.gm().builder_cubes.entry(id).or_insert(0) += 1;
    }

    /// `hex.assigned?(key)`.
    pub(crate) fn g1822_pnw_assigned(&self, hex: &str, key: &str) -> bool {
        self.g().hex_assignments.get(hex).map_or(false, |v| v.iter().any(|k| k == key))
    }

    pub(crate) fn g1822_pnw_assign(&mut self, hex: &str, key: &str) {
        let v = self.gm().hex_assignments.entry(hex.to_string()).or_default();
        if !v.iter().any(|k| k == key) {
            v.push(key.to_string());
        }
    }

    /// `forest?(tile)`: the tile's upgrades carry forest terrain.
    pub(crate) fn g1822_pnw_forest_tile(&self, hi: usize) -> bool {
        self.hexes[hi].tile.upgrades.iter().any(|u| u.terrain == "forest")
    }

    /// `legal_city_and_town_tile(hex, tile)`: H19 takes yellow 5/6/57.
    pub(crate) fn g1822_pnw_city_and_town_tile(hex: &str, tile_base: &str) -> bool {
        hex == "H19" && matches!(tile_base, "5" | "6" | "57")
    }

    /// PNW `upgrades_to?` — Some(answer) where PNW decides, None to fall
    /// through to G1822 / base.
    pub(crate) fn g1822_pnw_upgrades_to(&self, hi: usize, to_base: &str, selected: Option<&str>) -> Option<bool> {
        let hex = &self.hexes[hi];
        let from_color = self.g1822_tile_color_name(hi);
        if Self::g1822_pnw_city_and_town_tile(&hex.id, to_base) && from_color == "white" {
            return Some(true);
        }
        let to_color = self.g1822_def_color_name(to_base);
        // blue only upgrades to blue (blue is past every colour step)
        if from_color == "blue" {
            return Some(to_color == "blue");
        }
        let boomtown = selected == Some("P18");
        if to_base == "PNW3" || boomtown {
            return Some(to_base == "PNW3" && boomtown);
        }
        if to_base == "PNW4" {
            if selected != Some("P19") {
                return Some(false);
            }
            if from_color == "brown" || from_color == "white" {
                return Some(true);
            }
        }
        let from_base = hex.tile.name.split('-').next().unwrap_or(&hex.tile.name);
        if from_base == "PNW4" {
            return Some(to_base == "PNW5");
        }
        if Self::g1822_pnw_tokencity(&hex.id) {
            let from_base = if hex.tile.id.starts_with("preprinted_") { "" } else { from_base };
            let fi = Self::g1822_pnw_tokencity_index(&hex.id, from_base);
            let ti = Self::g1822_pnw_tokencity_index(&hex.id, to_base);
            return Some(ti > fi);
        }
        None
    }

    /// `upgrade_ignore_num_cities(from)`: O14 in yellow.
    pub(crate) fn g1822_pnw_ignore_num_cities(&self, hi: usize) -> bool {
        self.g1822_is_pnw() && self.hexes[hi].id == "O14" && self.g1822_tile_color_name(hi) == "yellow"
    }

    /// PNW `upgrade_cost` for a token city: 20 per step along its list.
    pub(crate) fn g1822_pnw_tokencity_cost(&self, hi: usize, to_base: &str) -> Option<i32> {
        let hex = &self.hexes[hi];
        if !Self::g1822_pnw_tokencity(&hex.id) {
            return None;
        }
        let from_base = if hex.tile.id.starts_with("preprinted_") {
            ""
        } else {
            hex.tile.name.split('-').next().unwrap_or(&hex.tile.name)
        };
        let fi = Self::g1822_pnw_tokencity_index(&hex.id, from_base);
        let ti = Self::g1822_pnw_tokencity_index(&hex.id, to_base);
        Some(20 * (ti - fi))
    }

    /// The neighbour map (connect_hexes once at setup from the preprinted
    /// tiles, plus the edges a lay on a blue/gray/red hex opened since).
    pub(crate) fn g1822_pnw_neighbors(&self) -> std::collections::HashMap<usize, std::collections::BTreeMap<u8, usize>> {
        use std::collections::{BTreeMap, HashMap};
        use std::sync::{Arc, Mutex, OnceLock};
        type Nbrs = HashMap<usize, BTreeMap<u8, usize>>;
        // one map per title (1822PNW and 1822MX share this code)
        static STATIC: OnceLock<Mutex<HashMap<&'static str, Arc<Nbrs>>>> = OnceLock::new();
        let key = self.title_def().name();
        let cached = STATIC.get_or_init(Default::default).lock().ok().and_then(|m| m.get(key).cloned());
        let base: Arc<Nbrs> = cached.unwrap_or_else(|| {
            let built = Arc::new(self.g1822_static_neighbors());
            if let Ok(mut m) = STATIC.get_or_init(Default::default).lock() {
                m.insert(key, built.clone());
            }
            built
        });
        let mut m: Nbrs = (*base).clone();
        for &(h, d, n) in &self.g().opened_neighbors {
            m.entry(h).or_default().insert(d, n);
        }
        m
    }

    /// Ruby `connect_hexes` (once, at setup) over the preprinted map.
    fn g1822_static_neighbors(&self) -> std::collections::HashMap<usize, std::collections::BTreeMap<u8, usize>> {
        use std::collections::{BTreeMap, HashMap};
        let base = {
            let f = self.fam();
            let exits_of = |coord: &str| -> (Vec<u8>, Vec<u8>, &'static str) {
                let Some((dsl, color)) = (f.hex_dsl)(coord) else { return (vec![], vec![], "white") };
                let mut exits = Vec::new();
                let mut stubs = Vec::new();
                for part in dsl.split(';') {
                    let part = part.trim();
                    if let Some(attrs) = part.strip_prefix("path=") {
                        for kv in attrs.split(',') {
                            if let Some((k, v)) = kv.split_once(':') {
                                if (k == "a" || k == "b") && !v.starts_with('_') {
                                    if let Ok(e) = v.parse::<u8>() {
                                        exits.push(e);
                                    }
                                }
                            }
                        }
                    } else if let Some(v) = part.strip_prefix("stub=edge:") {
                        if let Ok(e) = v.parse::<u8>() {
                            stubs.push(e);
                        }
                    }
                }
                (exits, stubs, color)
            };
            let walls = self.g1822_impassable_edges();
            let mut out: HashMap<usize, BTreeMap<u8, usize>> = HashMap::new();
            for (hi, hex) in self.hexes.iter().enumerate() {
                let mut m = BTreeMap::new();
                if let Some(adj) = self.hex_adjacency.get(&hex.id) {
                    for (&dir, nid) in adj {
                        let Some(&ni) = self.hex_idx.get(nid) else { continue };
                        let inv = (dir + 3) % 6;
                        let (n_exits, n_stubs, n_color) = exits_of(nid);
                        if matches!(n_color, "blue" | "gray" | "red") && !n_exits.contains(&inv) && !n_stubs.contains(&inv) {
                            continue;
                        }
                        if walls.get(&hex.id).map_or(false, |w| w.contains(&dir)) {
                            continue;
                        }
                        m.insert(dir, ni);
                    }
                }
                out.insert(hi, m);
            }
            out
        };
        base
    }

    /// Tracker#lay_tile on an impassable-coloured old tile: the new tile's
    /// exits open both ways (unless an impassable border sits there).
    pub(crate) fn g1822_pnw_open_neighbors(&mut self, hi: usize, old_color: &str) {
        if !matches!(old_color, "blue" | "gray" | "red") {
            return;
        }
        let walls = self.g1822_impassable_edges();
        let hex_id = self.hexes[hi].id.clone();
        let exits: Vec<u8> = self.hexes[hi].tile.paths.iter().flat_map(|p| [p.a.edge_num(), p.b.edge_num()]).flatten().collect();
        let adj: Vec<(u8, String)> = self.hex_adjacency.get(&hex_id).map(|m| m.iter().map(|(d, n)| (*d, n.clone())).collect()).unwrap_or_default();
        for (dir, nid) in adj {
            if walls.get(&hex_id).map_or(false, |w| w.contains(&dir)) || !exits.contains(&dir) {
                continue;
            }
            let Some(&ni) = self.hex_idx.get(&nid) else { continue };
            let inv = (dir + 3) % 6;
            self.gm().opened_neighbors.push((ni, inv, hi));
            self.gm().opened_neighbors.push((hi, dir, ni));
        }
    }

    /// `after_lay_tile` (PNW part): an H13 tile exiting at 1 removes the
    /// first border of its edge-1 neighbour (Seattle's water border).
    pub(crate) fn g1822_pnw_after_lay_tile(&mut self, hi: usize) {
        if self.hexes[hi].id != "H13" {
            return;
        }
        let exits1 = self.hexes[hi].tile.paths.iter().any(|p| p.a.edge_num() == Some(1) || p.b.edge_num() == Some(1));
        if !exits1 {
            return;
        }
        let Some(nid) = self.hex_adjacency.get("H13").and_then(|m| m.get(&1)).cloned() else { return };
        if let Some(v) = self.gm().borders.get_mut(&nid) {
            if !v.is_empty() {
                v.remove(0);
            }
        }
    }

    // =====================================================================
    // SpecialTrack (step/special_track.rb)
    // =====================================================================

    /// The rotation rule a PNW company's lay follows (None: G1822's).
    pub(crate) fn g1822_pnw_rot_rule(&self, company: &str) -> Option<super::tile_rules::RotRule> {
        use super::tile_rules::RotRule;
        match company {
            "P12" | "P13" => Some(RotRule::PnwPort),
            "P16" => Some(RotRule::PnwPortage),
            "P18" => Some(RotRule::PnwBoomtown),
            "P19" => Some(RotRule::PnwCoal),
            _ => None,
        }
    }

    /// legal_tile_rotation_{port,portage,boomtown,coal}_company?
    pub(crate) fn g1822_pnw_special_rotation(&self, rule: super::tile_rules::RotRule, operator: &str, hi: usize, exits: &[u8]) -> bool {
        use super::tile_rules::RotRule;
        let hex_id = self.hexes[hi].id.clone();
        let all_nb: Vec<(u8, String)> = self
            .hex_adjacency
            .get(&hex_id)
            .map(|m| m.iter().map(|(d, n)| (*d, n.clone())).collect())
            .unwrap_or_default();
        match rule {
            RotRule::PnwPort => {
                // hex.tile.paths.any? { terminal == '1' && exits == tile.exits }
                let Some((dsl, _)) = (self.fam().hex_dsl)(&hex_id) else { return false };
                let mut ex = exits.to_vec();
                ex.sort_unstable();
                ex.dedup();
                dsl.split(';').filter_map(|p| p.trim().strip_prefix("path=")).any(|attrs| {
                    let kv: Vec<(&str, &str)> = attrs.split(',').filter_map(|x| x.split_once(':')).collect();
                    if !kv.iter().any(|(k, v)| *k == "terminal" && *v == "1") {
                        return false;
                    }
                    let mut pe: Vec<u8> = kv
                        .iter()
                        .filter(|(k, v)| (*k == "a" || *k == "b") && !v.starts_with('_'))
                        .filter_map(|(_, v)| v.parse().ok())
                        .collect();
                    pe.sort_unstable();
                    pe.dedup();
                    pe == ex
                })
            }
            RotRule::PnwPortage => exits.iter().all(|e| {
                all_nb.iter().find(|(d, _)| d == e).map_or(false, |(_, n)| n != "H11")
            }),
            RotRule::PnwBoomtown => true,
            RotRule::PnwCoal => {
                let g = self.g1822_graph(operator);
                let nbrs = self.g1822_neighbors();
                exits.iter().any(|e| {
                    nbrs.get(&hi).and_then(|m| m.get(e)).map_or(false, |n| g.connected_hexes.contains_key(n))
                })
            }
            _ => true,
        }
    }

    /// PNW SpecialTrack#process_lay_tile for the companies with their own
    /// handling (P10 cubes, extra-track cubes, P16 portage, P18 boomtown,
    /// P19 coal); None → the G1822 path (P7/P8/P11 and the port tiles).
    pub(crate) fn g1822_pnw_special_lay(&mut self, co: &str, hex_id: &str, tile_id: &str, rotation: u8) -> Option<Result<(), GameError>> {
        let base = tile_id.split('-').next().unwrap_or(tile_id).to_string();
        let is_cube = base == "BC";
        let handled = co == "P10" || (is_cube && self.g1822_company_ability_extra_track(co)) || matches!(co, "P16" | "P18" | "P19");
        let tokencity = Self::g1822_pnw_tokencity(hex_id);
        if !handled {
            // handle_tokencity: a major's P11 on a token city uses up once more
            if tokencity && co == "P11" {
                if let Some(owner) = self.co_owner(co).corp_sym().map(str::to_string) {
                    if self.is_major(&owner) {
                        if let Some((idx, _)) = self.g1822_track_ability(co) {
                            if let Err(e) = self.g1822_use_ability(co, idx, false) {
                                return Some(Err(e));
                            }
                        }
                    }
                }
            }
            return None;
        }
        Some(self.g1822_pnw_special_lay_inner(co, hex_id, tile_id, rotation, &base, tokencity))
    }

    fn g1822_pnw_special_lay_inner(
        &mut self,
        co: &str,
        hex_id: &str,
        tile_id: &str,
        rotation: u8,
        base: &str,
        tokencity: bool,
    ) -> Result<(), GameError> {
        let (idx, ab) = self.g1822_track_ability(co).ok_or_else(|| GameError::new(format!("{co} has no tile lay ability")))?;
        let spender = self
            .co_owner(co)
            .corp_sym()
            .map(str::to_string)
            .ok_or_else(|| GameError::new("company not corporation-owned"))?;
        let _ = tokencity;
        let check_closing = |g: &mut BaseGame, c: &str, i: usize| {
            let count = g.g1822_abilities(c).get(i).and_then(|a| a.count).unwrap_or(0);
            if ab.closed_when_used_up && count <= 0 {
                g.close_company(c);
            }
        };
        let hi = *self.hex_idx.get(hex_id).ok_or_else(|| GameError::new(format!("Unknown hex {hex_id}")))?;
        if co == "P10" {
            // process_lay_tile_cube_company
            self.g1822_pnw_place_cube(hex_id);
            self.g1822_use_ability(co, idx, false)?;
            check_closing(self, co, idx);
            return Ok(());
        }
        if base == "BC" {
            // process_lay_tile_extra_track_cube
            self.g1822_pnw_place_cube(hex_id);
            self.g1822_use_ability(co, idx, false)?;
            if self.is_minor(&spender) {
                self.g1822_use_ability(co, idx, false)?;
            } else {
                self.rndm().extra_laided_track = true;
            }
            check_closing(self, co, idx);
            // handle_extra_tile_lay_company
            let count = self.g1822_abilities(co).get(idx).and_then(|a| a.count).unwrap_or(0);
            if ab.must_lay_together {
                if self.rnd().num_laid_track == 1 && !ab.consume_tile_lay {
                    self.rndm().num_laid_track += 1;
                }
                self.rndm().special_track_company = if count > 0 && !self.co_closed(co) { Some(co.to_string()) } else { None };
            } else {
                self.rndm().special_track_company = None;
            }
            return Ok(());
        }
        match co {
            "P16" => {
                // process_lay_tile_portage_company
                if self.rnd().num_laid_track > 0 {
                    return err("Cannot play portage now");
                }
                // (no graph: portage rotation reads the neighbours only)
                self.g1822_lay_tile(co, &spender, &[], hi, tile_id, rotation)?;
                self.g1822_use_ability(co, idx, false)?;
                self.rndm().num_laid_portage += 1;
                if let Some(i) = self.co_idx(co) {
                    self.companies[i].revenue = 0;
                }
            }
            "P18" => {
                self.g1822_lay_tile(co, &spender, &[], hi, tile_id, rotation)?;
                self.g1822_use_ability(co, idx, false)?;
                check_closing(self, co, idx);
            }
            "P19" => {
                // process_lay_tile_coal_company
                if self.g1822_pnw_assigned(hex_id, "P15") {
                    return err(format!("Cannot lay Rockport Coal Mine in Paper Mill hex {hex_id}"));
                }
                if self.g1822_pnw_assigned(hex_id, "P17") {
                    return err(format!("Cannot lay Rockport Coal Mine in Ski Haus hex {hex_id}"));
                }
                // (get_tile_lay is nil once a portage lay was made)
                let lay = self.rnd().num_laid_portage == 0 && self.g1822_tile_lay(&spender, false).map_or(false, |t| t.0);
                if !lay {
                    return err("Cannot lay coal company now");
                }
                self.g1822_note_graph(&spender);
                self.g1822_lay_tile(co, &spender, &[], hi, tile_id, rotation)?;
                self.rndm().num_laid_track += 1;
                // the coal token in the new tile's city
                let si = self.ci(&spender);
                let mut tok = crate::entities::Token::new(spender.clone(), 0);
                tok.token_type = "coal".to_string();
                self.corporations[si].tokens.push(tok);
                let ti = self.corporations[si].tokens.len() - 1;
                self.g1822_put_token(&spender, ti, hi, 0, false)?;
                if let Some(i) = self.co_idx("P19") {
                    self.companies[i].revenue = 0;
                }
                self.g1822_use_ability(co, idx, false)?;
            }
            _ => return err(format!("{co}: no PNW special lay")),
        }
        Ok(())
    }

    // =====================================================================
    // Revenue extras and fees
    // =====================================================================

    /// The hexes a recorded route passes (`route.all_hexes`, unique).
    pub(crate) fn g1822_pnw_route_hexes(r: &super::RunRoute) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        for chain in &r.connections {
            for h in chain {
                if h != "local" && !out.contains(h) {
                    out.push(h.clone());
                }
            }
        }
        for h in &r.hexes {
            if !out.contains(h) {
                out.push(h.clone());
            }
        }
        out
    }

    /// `portage_penalty(route)`: 10 per portage tile passed, unless the
    /// train's owner holds P16.
    pub(crate) fn g1822_pnw_portage_penalty(&self, c: &str, hexes: &[String]) -> i32 {
        if self.g1822_owns(c, "P16") {
            return 0;
        }
        10 * hexes
            .iter()
            .filter(|h| {
                self.hex_idx.get(h.as_str()).map_or(false, |&hi| {
                    let name = &self.hexes[hi].tile.name;
                    matches!(name.split('-').next(), Some("PNW1" | "PNW2"))
                })
            })
            .count() as i32
    }

    /// PNW Dividend#process_dividend: the portage owner collects the
    /// usage fees of the routes run (from the bank).
    pub(crate) fn g1822_pnw_pay_portage_fee(&mut self) {
        let Some(c) = self.g1822_operator() else { return };
        let fee: i32 = self
            .rnd()
            .routes
            .iter()
            .map(|r| self.g1822_pnw_portage_penalty(&c, &Self::g1822_pnw_route_hexes(r)))
            .sum();
        if fee <= 0 || self.co_closed("P16") {
            return;
        }
        if let Some(owner) = self.co_owner("P16").corp_sym().map(str::to_string) {
            self.g1822_bank_pay(&EntityId::corporation(&owner), fee);
        }
    }

    /// A corporation's `coordinates` (P20 moves a major's home).
    pub(crate) fn g1822_pnw_home(&self, c: &str) -> Option<String> {
        if let Some(h) = self.g().home_override.get(c) {
            return Some(h.clone());
        }
        self.title_def().corporations().into_iter().find(|d| d.sym == c).map(|d| d.home_hex.to_string())
    }

    // =====================================================================
    // Assign (step/assign.rb): P15 paper mill, P17 ski haus
    // =====================================================================

    /// The assign_hexes ability's hexes (entities.rb).
    pub(crate) fn g1822_pnw_assign_hexes(co: &str) -> &'static [&'static str] {
        match co {
            "P15" => &["D11", "D19", "F9", "F13", "G14", "G16", "H19", "H21", "J5", "L11", "L19", "N5", "O14"],
            "P17" => &["A16", "G4", "G18", "H5", "I6", "J17", "M14", "N13"],
            _ => &[],
        }
    }

    /// `coal_hex?(hex)`: PNW4 / PNW5 laid there.
    pub(crate) fn g1822_pnw_coal_hex(&self, hex: &str) -> bool {
        self.hex_idx.get(hex).map_or(false, |&hi| {
            let n = &self.hexes[hi].tile.name;
            matches!(n.split('-').next(), Some("PNW4" | "PNW5"))
        })
    }

    /// Assign#actions: an open P15 / P17 owned by the operating
    /// corporation (`when: owning_corp_or_turn`).
    pub(crate) fn g1822_pnw_assign_actions(&self, ent: &Ent) -> Vec<&'static str> {
        let Ent::Company(co) = ent else { return vec![] };
        if !matches!(co.as_str(), "P15" | "P17") || self.co_closed(co) || self.rnd().kind != RoundKind::Operating {
            return vec![];
        }
        let owner = self.co_owner(co);
        match (owner.corp_sym(), self.g1822_operator()) {
            (Some(o), Some(op)) if o == op => vec!["assign"],
            _ => vec![],
        }
    }

    /// Assign#process_assign (PNW checks, then the base step).
    pub(crate) fn g1822_pnw_assign_process(&mut self, ent: &Ent, action: &Action) -> Result<(), GameError> {
        let Ent::Company(co) = ent else { return err("assign by a non-company") };
        let Action::Assign { target, target_type, .. } = action else { return err("Assign cannot process this action") };
        if target_type != "hex" {
            return err(format!("Invalid target {target} for assigning company {co}"));
        }
        if co == "P15" {
            if self.g1822_pnw_coal_hex(target) {
                return err(format!("Cannot place Paper Mill in Rockport Coal Mine hex {target}"));
            }
            if self.g1822_pnw_assigned(target, "P17") {
                return err(format!("Cannot place Paper Mill in Ski Haus hex {target}"));
            }
        }
        if co == "P17" {
            if self.g1822_pnw_coal_hex(target) {
                return err(format!("Cannot place Ski Haus in Rockport Coal Mine hex {target}"));
            }
            if self.g1822_pnw_assigned(target, "P15") {
                return err(format!("Cannot place Ski Haus in Paper Mill hex {target}"));
            }
        }
        if self.g1822_pnw_assigned(target, co) {
            return err(format!("{co} is already assigned to {target}"));
        }
        if !self.hex_idx.contains_key(target.as_str()) {
            return err(format!("Unknown hex {target}"));
        }
        // Assignable.remove_from_all!(ability hexes, company), assign!, use!
        for h in Self::g1822_pnw_assign_hexes(co) {
            if let Some(v) = self.gm().hex_assignments.get_mut(*h) {
                v.retain(|k| k != co);
            }
        }
        let co = co.clone();
        self.g1822_pnw_assign(target, &co);
        // count 1, closed_when_used_up
        self.close_company(&co);
        Ok(())
    }

    // =====================================================================
    // AcquireCompany (step/acquire_company.rb): P20 backroom negotiations
    // =====================================================================

    /// `p20_targets`: associated minors still in the bank (not in a bid
    /// box, or already discarded) whose major is unstarted.
    pub(crate) fn g1822_pnw_p20_targets(&self) -> Vec<String> {
        let boxed = self.bidbox_minors();
        let mut out = Vec::new();
        for (m, j) in &self.g().minor_associations {
            let company = format!("M{m}");
            let Some(&mi) = self.corp_idx.get(m) else { continue };
            let Some(&ji) = self.corp_idx.get(j) else { continue };
            let minor = &self.corporations[mi];
            let major = &self.corporations[ji];
            let company_owned = self.co_idx(&company).map_or(false, |i| !self.companies[i].owner.is_none())
                || self.g().bank_owned.contains(&company);
            let already_discarded = !company_owned && self.co_closed(&company) && minor.closed;
            let minor_available = minor.owner_id.is_none() && (already_discarded || !boxed.contains(&company));
            let major_available = major.owner_id.is_none() && !major.floated;
            if minor_available && major_available {
                out.push(m.clone());
            }
        }
        out
    }

    /// PNW AcquireCompany#process_acquire_company for P20: the company
    /// closes; an unassociated minor may take an associated minor's place.
    pub(crate) fn g1822_pnw_acquire_p20(&mut self, c: &str) -> Result<(), GameError> {
        if self.co_closed("P20") {
            return err("P20 is closed");
        }
        self.close_company("P20");
        if !self.g1822_pnw_minors(false).iter().any(|m| m == c) {
            return Ok(());
        }
        let targets = self.g1822_pnw_p20_targets();
        if targets.is_empty() {
            return Ok(());
        }
        self.rndm().p20_minor = Some(c.to_string());
        self.rndm().p20_choices = targets;
        Ok(())
    }

    /// PNW AcquireCompany#process_choose.
    pub(crate) fn g1822_pnw_p20_choose(&mut self, choice: &str) -> Result<(), GameError> {
        if !self.rnd().p20_choices.iter().any(|c| c == choice) {
            return err(format!("{choice} is an illegal choice"));
        }
        let new_minor = self.rnd().p20_minor.clone().ok_or_else(|| GameError::new("no P20 minor"))?;
        let major = self.g1822_pnw_assoc_major(choice).ok_or_else(|| GameError::new("no major"))?;
        // replace_associated_minor: the new entry goes to the end
        let st = self.gm();
        st.minor_associations.retain(|(m, _)| m != choice);
        st.minor_associations.push((new_minor.clone(), major.clone()));
        // the old minor leaves the game (its reservation goes)
        self.g1822_remove_reservations(choice);
        if let Some(&oi) = self.corp_idx.get(choice) {
            self.corporations[oi].closed = true;
        }
        let original_home = self.g1822_pnw_home(&major).unwrap_or_default();
        let new_home = self.g1822_pnw_home(&new_minor).unwrap_or_default();
        self.gm().home_override.insert(major.clone(), new_home.clone());
        if self.g1822_destination(&major).as_deref() == Some(new_home.as_str()) {
            self.gm().destination_override.insert(major.clone(), original_home);
        }
        self.delete_company(&format!("M{choice}"));
        let r = self.rndm();
        r.p20_choices.clear();
        r.p20_minor = None;
        Ok(())
    }

    // =====================================================================
    // Majors: formation, par, conversion (game.rb, step/buy_sell_par_shares.rb)
    // =====================================================================

    /// `major_formation_status(major, player:)`.
    pub(crate) fn g1822_pnw_formation(&self, major: &str, player: Option<u32>) -> Formation {
        let Some(&ji) = self.corp_idx.get(major) else { return Formation::None };
        if self.corporations[ji].floated {
            return Formation::Formed;
        }
        let Some(minor) = self.g1822_pnw_assoc_minor(major) else { return Formation::Parable };
        if let (Some(&mi), Some(p)) = (self.corp_idx.get(&minor), player) {
            let m = &self.corporations[mi];
            if !m.closed && !m.owner_id.is_none() {
                return if m.owner_id == EntityId::player(p) { Formation::Convertable } else { Formation::None };
            }
        }
        let company = format!("M{minor}");
        let in_companies = self.g().company_order.iter().any(|c| *c == company);
        let has_bids = self.rnd().kind == RoundKind::Stock && self.rnd().bids.get(&company).map_or(false, |b| !b.is_empty());
        if !in_companies || !has_bids {
            return Formation::Parable;
        }
        Formation::None
    }

    /// PNW `after_par(major)`: its associated minor's company closes (an
    /// emptied bid box keeps its place) and the minor's reservation goes.
    pub(crate) fn g1822_pnw_after_par(&mut self, major: &str) {
        if !self.is_major(major) {
            return;
        }
        let Some(minor) = self.g1822_pnw_assoc_minor(major) else { return };
        let company = format!("M{minor}");
        if self.co_closed(&company) {
            return;
        }
        if let Some(i) = self.bidbox_minors().iter().position(|c| *c == company) {
            if i < self.g().empty_bidboxes.len() {
                self.gm().empty_bidboxes[i] = true;
            }
            self.gm().any_bidboxes_empty = true;
        }
        self.g1822_remove_city_reservation(&minor);
        self.close_company(&company);
    }

    /// `minor_city.reservations.delete(minor)` on its home hex.
    pub(crate) fn g1822_remove_city_reservation(&mut self, m: &str) {
        let Some(home) = self.g1822_pnw_home(m) else { return };
        let Some(&hi) = self.hex_idx.get(home.as_str()) else { return };
        for k in 0..self.hexes[hi].tile.cities.len() {
            if let Some(v) = self.gm().city_reservations.get_mut(&format!("{home}#{k}")) {
                if let Some(pos) = v.iter().position(|r| r.as_deref() == Some(m)) {
                    v.remove(pos);
                    return;
                }
            }
        }
    }

    /// The conversion's share-count choices.
    pub(crate) fn g1822_pnw_convert_choices(&self) -> Vec<String> {
        let Some(major) = self.rnd().converting_major.clone() else { return vec![] };
        let Some(minor) = self.g1822_pnw_assoc_minor(&major) else { return vec![] };
        let mi = self.ci(&minor);
        let value = self.g1822_share_price_of(&minor) * 2;
        let cash = self.corporations[mi].cash;
        let pcash = self.corporations[mi].owner_id.player_id().map_or(0, |p| self.player_cash(p));
        Self::g1822_pnw_possible_shares(self.rnd().convert_par, cash, value, pcash)
            .iter()
            .map(|n| n.to_string())
            .collect()
    }

    /// PNW BSPS#process_choose: the associated minor becomes its major.
    pub(crate) fn g1822_pnw_convert_choose(&mut self, pid: u32, choice: &str) -> Result<(), GameError> {
        let major = self.rnd().converting_major.clone().ok_or_else(|| GameError::new("nothing to convert"))?;
        let n: i32 = choice.parse().map_err(|_| GameError::new(format!("bad choice {choice}")))?;
        let minor = self.g1822_pnw_assoc_minor(&major).ok_or_else(|| GameError::new("no associated minor"))?;
        let par = self.rnd().convert_par;
        let minor_value = self.g1822_share_price_of(&minor) * 2;
        self.g1822_pnw_transfer_possessions(&minor, &major);
        let ji = self.ci(&major);
        self.corporations[ji].floated = true;
        self.corporations[ji].capitalization = crate::title::Capitalization::Incremental;
        let sp = self
            .stock_market
            .share_prices_with_types(&["par", "par_1"])
            .into_iter()
            .find(|p| p.price == par)
            .ok_or_else(|| GameError::new(format!("no par price {par}")))?;
        self.g1822_par_corporation(&major, sp);
        let shares_value = n * par;
        if shares_value > minor_value {
            let diff = shares_value - minor_value;
            self.player_spend(pid, diff)?;
            self.bank.cash -= diff;
            self.corporations[ji].cash += diff;
        } else if shares_value < minor_value {
            let diff = minor_value - shares_value;
            if self.corporations[ji].cash < diff {
                return err(format!("{major} has {} and cannot spend {diff}", self.corporations[ji].cash));
            }
            self.corporations[ji].cash -= diff;
            self.pay_player(pid, diff);
        }
        let ipo = EntityId::ipo(&major);
        let idx: Vec<usize> = (0..self.corporations[ji].shares.len())
            .filter(|&i| self.corporations[ji].shares[i].owner == ipo)
            .take((n - 1).max(0) as usize)
            .collect();
        let percent: u32 = idx.iter().map(|&i| self.corporations[ji].shares[i].percent as u32).sum();
        let b = Bundle { corp: major.clone(), shares: idx, percent: percent as u8, price_override: None };
        self.g1822_transfer_shares(&b, &EntityId::player(pid), true);
        if let Some(at) = self.g1822_pnw_delete_first_token(&minor) {
            self.g1822_pnw_place_major_token(&major, at, false)?;
        }
        self.g1822_pnw_close_minor(&minor);
        self.g1822_pnw_track_choose(pid);
        self.rndm().converting_major = None;
        self.g1822_bsps_pass();
        Ok(())
    }

    /// `company_choices_p21(company, time)`: move a bank minor / private
    /// (not in a bid box) to the top or bottom of its stack, or (a major
    /// with exchange tokens) free an exchange token.
    pub(crate) fn g1822_pnw_p21_choices(&self, time: &str) -> Vec<String> {
        let owner = self.co_owner("P21");
        let Some(c) = owner.corp_sym() else { return vec![] };
        if !matches!(time, "token" | "track" | "issue") {
            return vec![];
        }
        let mut v = Vec::new();
        for (prefix, boxed) in [('M', self.bidbox_minors()), ('P', self.bidbox_privates())] {
            for co in self.bank_companies(prefix) {
                if boxed.contains(&co) {
                    continue;
                }
                v.push(format!("{co}_top"));
                v.push(format!("{co}_bottom"));
            }
        }
        if self.is_major(c) && self.g1822_exchange_tokens(c) > 0 {
            v.push("exchange".to_string());
        }
        v
    }

    /// `setup_starting_packets`: every player's cash goes to the bank; each
    /// seat floats its two packet minors at the lowest par (the 100%
    /// share free, $100 from the bank) and owns its two privates.
    pub(crate) fn g1822_pnw_setup_starting_packets(&mut self) {
        let par = self.g1822_pnw_par_prices().into_iter().next();
        let Some(par) = par else { return };
        let pids: Vec<u32> = self.players.iter().map(|p| p.id).collect();
        for (index, pid) in pids.into_iter().enumerate() {
            let Some(&(minors, privates)) = STARTING_PACKETS.get(index) else { continue };
            let pi = self.player_index(pid).unwrap();
            let cash = self.players[pi].cash;
            self.players[pi].cash = 0;
            self.bank.cash += cash;
            for company in minors {
                let m = Self::minor_of_company(company);
                self.g1822_par_corporation(&m, par.clone());
                let b = Bundle { corp: m.clone(), shares: vec![0], percent: 100, price_override: Some(0) };
                let _ = self.g1822_pool_buy_shares(&EntityId::player(pid), &b, None, false, true);
                self.g1822_bank_spend(100);
                let mi = self.ci(&m);
                self.corporations[mi].cash += 100;
                self.delete_company(company);
            }
            for company in privates {
                if let Some(i) = self.co_idx(company) {
                    self.companies[i].owner = EntityId::player(pid);
                }
            }
        }
    }

    /// `major_name_for_associated_minor(id)`.
    pub(crate) fn g1822_pnw_assoc_major(&self, minor: &str) -> Option<String> {
        self.g().minor_associations.iter().find(|(m, _)| m == minor).map(|(_, j)| j.clone())
    }

    /// `associated_minor(major)`'s id.
    pub(crate) fn g1822_pnw_assoc_minor(&self, major: &str) -> Option<String> {
        self.g().minor_associations.iter().find(|(_, j)| j == major).map(|(m, _)| m.clone())
    }

    /// `associated_minor?(entity)`.
    pub(crate) fn g1822_pnw_is_associated(&self, minor: &str) -> bool {
        self.g().minor_associations.iter().any(|(m, _)| m == minor)
    }

    /// `regional_railway?(entity)`.
    pub(crate) fn g1822_pnw_is_regional(sym: &str) -> bool {
        REGIONALS.contains(&sym)
    }

    /// `associated_minors` / `unassociated_minors` (floated, @corporations
    /// order).
    pub(crate) fn g1822_pnw_minors(&self, associated: bool) -> Vec<String> {
        self.corporations
            .iter()
            .filter(|c| !c.closed && c.floated && c.corp_type == CorpType::Minor)
            .filter(|c| self.g1822_pnw_is_associated(&c.sym) == associated)
            .map(|c| c.sym.clone())
            .collect()
    }

    /// Connections#connected_to_port?: the entity's graph reaches E8 or D9.
    fn g1822_pnw_connected_to_port(&self, c: &str) -> bool {
        // (a full Graph#compute: cached, latches route_train_purchase)
        self.g1822_ruby_graph_store(c);
        self.g1822_note_graph(c);
        let g = self.g1822_graph(c);
        ["E8", "D9"].iter().any(|h| self.hex_idx.get(*h).map_or(false, |hi| g.connected_hexes.contains_key(hi)))
    }

    /// A minor's city: its reserved home city while unowned (a bid-box
    /// minor), else its (single) token's city.
    fn g1822_pnw_minor_city(&self, m: &str) -> Option<Node> {
        let ci = self.ci(m);
        let corp = &self.corporations[ci];
        if corp.owner_id.is_none() {
            return self.g1822_pnw_minor_city_reserved(m);
        }
        // minor.tokens.first: the home token (a coal token comes later)
        self.g1822_token_city(m, Some("normal")).map(|(hi, k)| Node { hex: hi, kind: NodeKind::City, index: k })
    }

    /// Connections#entity_connects?(entity, minor).
    pub(crate) fn g1822_pnw_entity_connects(&self, entity: &str, minor: &str) -> bool {
        self.g1822_ruby_graph_store(entity);
        self.g1822_note_graph(entity);
        if let Some(city) = self.g1822_pnw_minor_city(minor) {
            if self.g1822_graph(entity).connected_nodes.contains(&city) {
                return true;
            }
        }
        self.g1822_pnw_connected_to_port(minor) && self.g1822_pnw_connected_to_port(entity)
    }

    // =====================================================================
    // Conversion (step/conversion.rb)
    // =====================================================================

    /// `possible_exchanged_shares(par, minors_cash, minors_value, player_cash)`.
    pub(crate) fn g1822_pnw_possible_shares(par: i32, cash: i32, value: i32, player_cash: i32) -> Vec<i32> {
        if 6 * par < value {
            return if par == 100 { vec![6] } else { vec![] };
        }
        // Ruby integer division floors
        let min = ((value - cash - 1).div_euclid(par) + 1).max(2);
        let max = ((value + player_cash.min(par - 1)).div_euclid(par)).min(6);
        (min..=max).collect()
    }

    /// The par prices (`stock_market.par_prices`), ascending.
    pub(crate) fn g1822_pnw_par_prices(&self) -> Vec<crate::core::SharePrice> {
        let mut v = self.stock_market.share_prices_with_types(&["par", "par_1"]);
        v.sort_by_key(|p| (p.price, p.column, p.row));
        v
    }

    fn g1822_share_price_of(&self, c: &str) -> i32 {
        self.corporations[self.ci(c)].share_price.as_ref().map_or(0, |s| s.price)
    }

    // =====================================================================
    // The Merger round (round/merger.rb + step/merge.rb)
    // =====================================================================

    /// `new` Merger round: the floated minors in operating order, one Merge
    /// step; setup skips steps and moves past minors with nothing to do.
    pub(crate) fn g1822_pnw_start_merger_round(&mut self, round_num: u8) -> Result<(), GameError> {
        let entities: Vec<String> = self.g1822_operating_order().into_iter().filter(|c| self.is_minor(c)).collect();
        self.gm().round = RoundSt::new(RoundKind::Merger, round_num, entities, vec![Step::Merge]);
        self.g1822_skip_steps();
        if self.g1822_active_step().is_none() {
            self.g1822_pnw_merger_next_entity();
        }
        Ok(())
    }

    /// Merger#next_entity!.
    pub(crate) fn g1822_pnw_merger_next_entity(&mut self) {
        for _ in 0..64 {
            let r = self.rnd();
            if r.entities.is_empty() || r.entity_index + 1 >= r.entities.len() {
                return;
            }
            let n = r.steps.len();
            let r = self.rndm();
            r.entity_index += 1;
            r.passed = vec![false; n];
            r.acted = vec![false; n];
            // Merge#setup
            r.merge_state = MergeState::None;
            self.g1822_skip_steps();
            if self.g1822_active_step().is_some() {
                return;
            }
        }
    }

    /// Merger#after_process.
    pub(crate) fn g1822_pnw_merger_after_process(&mut self) {
        if self.g1822_active_step().is_none() {
            self.g1822_pnw_merger_next_entity();
        }
    }

    /// Merge#mergeable(corporation).
    pub(crate) fn g1822_pnw_mergeable(&self, c: &str) -> Vec<String> {
        if self.rnd().merge_state != MergeState::None {
            return vec![];
        }
        let others = self.g1822_pnw_minors(!self.g1822_pnw_is_associated(c));
        let owner = self.corporations[self.ci(c)].owner_id.clone();
        others
            .into_iter()
            .filter(|m| {
                (self.g1822_pnw_entity_connects(c, m) || self.g1822_pnw_entity_connects(m, c))
                    && !self.g1822_pnw_valid_pars(c, m).is_empty()
                    && self.corporations[self.ci(m)].owner_id == owner
            })
            .collect()
    }

    /// Merge#valid_par_prices(minor_one, minor_two).
    pub(crate) fn g1822_pnw_valid_pars(&self, m1: &str, m2: &str) -> Vec<i32> {
        let value = (self.g1822_share_price_of(m1) + self.g1822_share_price_of(m2)) * 2;
        let cash = if self.rnd().merge_state == MergeState::None {
            self.corporations[self.ci(m1)].cash + self.corporations[self.ci(m2)].cash
        } else {
            self.rnd().merge_new.as_ref().map_or(0, |n| self.corporations[self.ci(n)].cash)
        };
        let pcash = self.corporations[self.ci(m1)].owner_id.player_id().map_or(0, |p| self.player_cash(p));
        self.g1822_pnw_par_prices()
            .into_iter()
            .map(|p| p.price)
            .filter(|&par| par != 50 && !Self::g1822_pnw_possible_shares(par, cash, value, pcash).is_empty())
            .collect()
    }

    /// Merge#actions(entity).
    pub(crate) fn g1822_pnw_merge_actions(&self, ent: &Ent) -> Vec<&'static str> {
        let Ent::Corp(c) = ent else { return vec![] };
        if self.g1822_round_entity().as_ref() != Some(ent) {
            return vec![];
        }
        if self.rnd().merge_state != MergeState::None {
            return vec!["choose"];
        }
        if !self.g1822_pnw_mergeable(c).is_empty() {
            return vec!["merge", "pass"];
        }
        vec![]
    }

    /// Merge#choices (the keys).
    pub(crate) fn g1822_pnw_merge_choices(&self) -> Vec<String> {
        let r = self.rnd();
        let (Some(a), Some(u)) = (r.merge_assoc.clone(), r.merge_unassoc.clone()) else { return vec![] };
        match r.merge_state {
            MergeState::SelectingPar => self.g1822_pnw_valid_pars(&a, &u).iter().map(|p| p.to_string()).collect(),
            MergeState::SelectingShares => {
                let value = (self.g1822_share_price_of(&a) + self.g1822_share_price_of(&u)) * 2;
                let cash = r.merge_new.as_ref().map_or(0, |n| self.corporations[self.ci(n)].cash);
                let pcash = r.merge_player.map_or(0, |p| self.player_cash(p));
                Self::g1822_pnw_possible_shares(r.merge_par, cash, value, pcash).iter().map(|s| s.to_string()).collect()
            }
            MergeState::SelectingToken => vec!["replace".to_string(), "exchange".to_string()],
            MergeState::None => vec![],
        }
    }

    /// `transfer_posessions(from, to)`: cash, companies, trains, the coal
    /// token.
    pub(crate) fn g1822_pnw_transfer_possessions(&mut self, from: &str, to: &str) {
        let fi = self.ci(from);
        let ti = self.ci(to);
        let cash = self.corporations[fi].cash;
        if cash > 0 {
            self.corporations[fi].cash = 0;
            self.corporations[ti].cash += cash;
        }
        let to_id = EntityId::corporation(to);
        for co in self.companies_of(&EntityId::corporation(from)) {
            if let Some(i) = self.co_idx(&co) {
                self.companies[i].owner = to_id.clone();
            }
        }
        let trains: Vec<crate::entities::Train> = std::mem::take(&mut self.corporations[fi].trains);
        for mut t in trains {
            t.owner = to_id.clone();
            self.corporations[ti].trains.push(t);
        }
        self.g1822_pnw_move_coal_token(from, to);
    }

    /// The coal token (type "coal") changes hands with its corporation.
    pub(crate) fn g1822_pnw_move_coal_token(&mut self, from: &str, to: &str) {
        let fi = self.ci(from);
        let Some(pos) = self.corporations[fi].tokens.iter().position(|t| t.token_type == "coal") else { return };
        let mut tok = self.corporations[fi].tokens.remove(pos);
        tok.corporation_id = to.to_string();
        let hex = tok.city_hex_id.clone();
        if tok.used {
            if let Some(&hi) = self.hex_idx.get(hex.as_str()) {
                for city in &mut self.hexes[hi].tile.cities {
                    for t in city.tokens.iter_mut().flatten() {
                        if t.corporation_id == from && t.token_type == "coal" {
                            t.corporation_id = to.to_string();
                        }
                    }
                }
            }
        }
        let ti = self.ci(to);
        self.corporations[ti].tokens.push(tok);
        self.clear_graph_cache();
    }

    /// `close_minor(minor)`: the owner's share goes, the minor closes and
    /// leaves `@corporations` (its tokens were already lifted).
    pub(crate) fn g1822_pnw_close_minor(&mut self, m: &str) {
        let ci = self.ci(m);
        let n = self.corporations[ci].shares.len();
        for i in 0..n {
            self.corporations[ci].set_share_owner(i, EntityId::none());
        }
        if let Some(sp) = self.corporations[ci].share_price.clone() {
            if let Some(v) = self.market_cell_corps.get_mut(&(sp.row, sp.column)) {
                v.retain(|c| c != m);
            }
        }
        let c = &mut self.corporations[ci];
        c.closed = true;
        c.floated = false;
        c.ipo_price = None;
        c.owner_id = EntityId::none();
        self.clear_graph_cache();
    }

    /// Lift a corporation's first token off the map (`city.delete_token!`).
    pub(crate) fn g1822_pnw_delete_first_token(&mut self, m: &str) -> Option<(usize, usize)> {
        let (hi, ci) = self.g1822_token_city(m, Some("normal"))?;
        let city = &mut self.hexes[hi].tile.cities[ci];
        for t in city.tokens.iter_mut() {
            if t.as_ref().is_some_and(|tok| tok.corporation_id == m) {
                *t = None;
                break;
            }
        }
        let mi = self.ci(m);
        if let Some(t) = self.corporations[mi].tokens.first_mut() {
            t.used = false;
            t.city_hex_id.clear();
        }
        self.clear_graph_cache();
        Some((hi, ci))
    }

    /// `city.place_token(major, major.find_token_by_type, check_tokenable:
    /// false)` (type "normal"), or the destination token as a cheater.
    pub(crate) fn g1822_pnw_place_major_token(&mut self, major: &str, at: (usize, usize), destination: bool) -> Result<(), GameError> {
        let ty = if destination { "destination" } else { "normal" };
        let idx = self
            .g1822_find_token(major, ty)
            .ok_or_else(|| GameError::new(format!("{major} has no {ty} token")))?;
        self.g1822_put_token(major, idx, at.0, at.1, destination)
    }

    /// Merge#process_merge.
    fn g1822_pnw_process_merge(&mut self, entity: &str, other: &str) -> Result<(), GameError> {
        let (assoc, unassoc) = if self.g1822_pnw_is_associated(entity) {
            (entity.to_string(), other.to_string())
        } else {
            (other.to_string(), entity.to_string())
        };
        if !self.corp_idx.contains_key(&unassoc) || self.corporations[self.ci(&unassoc)].closed {
            return err(format!("Choose a corporation to merge with {assoc}"));
        }
        let major = self
            .g1822_pnw_assoc_major(&assoc)
            .ok_or_else(|| GameError::new(format!("{assoc} is not an associated minor")))?;
        let player = self.corporations[self.ci(&assoc)].owner_id.player_id();
        self.g1822_pnw_transfer_possessions(&assoc, &major);
        self.g1822_pnw_transfer_possessions(&unassoc, &major);
        let ji = self.ci(&major);
        self.corporations[ji].floated = true;
        self.corporations[ji].capitalization = crate::title::Capitalization::Incremental;
        let r = self.rndm();
        r.merge_assoc = Some(assoc);
        r.merge_unassoc = Some(unassoc);
        r.merge_new = Some(major);
        r.merge_player = player;
        r.merge_state = MergeState::SelectingPar;
        Ok(())
    }

    /// Merge#process_choose (par → shares → token).
    fn g1822_pnw_merge_choose(&mut self, choice: &str) -> Result<(), GameError> {
        let r = self.rnd().clone();
        let (Some(assoc), Some(unassoc), Some(major)) = (r.merge_assoc.clone(), r.merge_unassoc.clone(), r.merge_new.clone())
        else {
            return err("no merge in progress");
        };
        match r.merge_state {
            MergeState::SelectingPar => {
                let par: i32 = choice.parse().map_err(|_| GameError::new(format!("bad par {choice}")))?;
                // par_corporation(new, par_prices.find { price == par })
                let sp = self
                    .stock_market
                    .share_prices_with_types(&["par", "par_1"])
                    .into_iter()
                    .find(|p| p.price == par)
                    .ok_or_else(|| GameError::new(format!("no par price {par}")))?;
                self.g1822_par_corporation(&major, sp);
                self.rndm().merge_par = par;
                self.rndm().merge_state = MergeState::SelectingShares;
            }
            MergeState::SelectingShares => {
                let n: i32 = choice.parse().map_err(|_| GameError::new(format!("bad share count {choice}")))?;
                let pid = r.merge_player.ok_or_else(|| GameError::new("no merging player"))?;
                // shares.first(n - 1): the major's own (IPO) certificates
                let ji = self.ci(&major);
                let ipo = EntityId::ipo(&major);
                let idx: Vec<usize> = (0..self.corporations[ji].shares.len())
                    .filter(|&i| self.corporations[ji].shares[i].owner == ipo)
                    .take((n - 1).max(0) as usize)
                    .collect();
                let percent: u32 = idx.iter().map(|&i| self.corporations[ji].shares[i].percent as u32).sum();
                let b = Bundle { corp: major.clone(), shares: idx, percent: percent as u8, price_override: None };
                self.g1822_transfer_shares(&b, &EntityId::player(pid), true);
                let value = (self.g1822_share_price_of(&assoc) + self.g1822_share_price_of(&unassoc)) * 2;
                let shares_value = n * r.merge_par;
                if shares_value > value {
                    let diff = shares_value - value;
                    self.player_spend(pid, diff)?;
                    self.bank.cash -= diff;
                    self.corporations[ji].cash += diff;
                } else if shares_value < value {
                    let to_pres = (value - shares_value).min(self.corporations[ji].cash);
                    self.corporations[ji].cash -= to_pres;
                    self.pay_player(pid, to_pres);
                }
                self.rndm().merge_state = MergeState::SelectingToken;
                let dest = self.g1822_destination(&major);
                let assoc_home = self.g1822_pnw_home(&assoc);
                if dest.is_some() && dest == assoc_home {
                    self.g1822_pnw_minor_on_destination(&assoc, &unassoc, &major)?;
                }
            }
            MergeState::SelectingToken => {
                // Ruby takes any other choice as 'exchange' (no validation)
                if let Some(at) = self.g1822_pnw_delete_first_token(&assoc) {
                    self.g1822_pnw_place_major_token(&major, at, false)?;
                }
                let at = self.g1822_pnw_delete_first_token(&unassoc);
                self.g1822_move_exchange_token(&major);
                if choice == "replace" {
                    if let Some(at) = at {
                        self.g1822_pnw_place_major_token(&major, at, false)?;
                    }
                }
                self.g1822_pnw_finish_merge(&assoc, &unassoc);
            }
            MergeState::None => return err("no merge in progress"),
        }
        Ok(())
    }

    /// Merge#process_minor_on_destination: the unassociated minor's city
    /// gets a normal token, the associated minor's (the destination) the
    /// destination token.
    fn g1822_pnw_minor_on_destination(&mut self, assoc: &str, unassoc: &str, major: &str) -> Result<(), GameError> {
        if let Some(at) = self.g1822_pnw_delete_first_token(unassoc) {
            self.g1822_pnw_place_major_token(major, at, false)?;
        }
        if let Some(at) = self.g1822_pnw_delete_first_token(assoc) {
            self.g1822_pnw_place_major_token(major, at, true)?;
        }
        self.gm().destination_ability.retain(|c| c != major);
        self.g1822_pnw_finish_merge(assoc, unassoc);
        Ok(())
    }

    /// Merge#finish_merge.
    fn g1822_pnw_finish_merge(&mut self, assoc: &str, unassoc: &str) {
        self.g1822_ruby_graph_clear();
        self.g1822_pnw_close_minor(assoc);
        self.g1822_pnw_close_minor(unassoc);
        let r = self.rndm();
        r.merge_state = MergeState::None;
        let i = r.step_index(Step::Merge).expect("merge step");
        r.passed[i] = true;
    }

    /// Merge step dispatch.
    pub(crate) fn g1822_pnw_merge_process(&mut self, i: usize, ent: &Ent, action: &Action) -> Result<(), GameError> {
        let Ent::Corp(c) = ent else { return err("merge: not a corporation") };
        match action {
            Action::Merge { corporation_sym, .. } => self.g1822_pnw_process_merge(c, corporation_sym),
            Action::Choose { choice, .. } => self.g1822_pnw_merge_choose(choice),
            Action::Pass { .. } => {
                self.rndm().passed[i] = true;
                Ok(())
            }
            _ => err(format!("Merge cannot process {}", action.action_type())),
        }
    }
}
