//! 1822 tile-lay legality — the checks Ruby runs when a player (not a
//! saved game) lays a tile, in one place for the legal-move generator and
//! any validating caller:
//!
//! * `Game#upgrades_to?` (G1822: the D35 "S" town-removing upgrade, P2's
//!   town→plain upgrade; base: colour step, `Tile#paths_are_subset_of?`
//!   over all rotations, labels incl. `future_label`, city/town counts,
//!   `city_town_edges_are_subset_of?`);
//! * `Tracker#legal_tile_rotation?` (G1822: D35/57 rotations 2 and 5, the
//!   operator for company lays; consent rights / stubs (as left by
//!   `preprocess_action`); base: new exits face a neighbour, reach the
//!   operator's connected edges, old paths kept, added-city count, multi-
//!   city mapping, city sizes);
//! * SpecialTrack's P2 remove-town rotations and P5's rotation 0 at P43;
//! * `Tracker#potential_tiles` (phase colours, minors' green cap) and the
//!   UI's `TILE_UPGRADES_MUST_USE_MAX_EXITS = [:cities]` filter;
//! * `check_track_restrictions!` (:semi_restrictive — new track or a
//!   changed city value, evaluated on the laid tile's connected paths).
//!
//! Replay keeps Ruby's `@game.loading` semantics: process_action trusts a
//! record's rotation and track (Ruby skips these checks while loading),
//! so a stale record never fails on a rule Ruby itself skips. Everything
//! here works on the engine's placed `Tile` (paths already rotated) and
//! catalog `TileDef`s.

use std::collections::{BTreeMap, HashMap, HashSet};

use crate::game::BaseGame;
use crate::graph::Tile;
use crate::tiles::{PathDef, PathEndpoint, TileColor, TileDef};
use crate::title::g1822 as data;

/// G1822 TRACK_PLAIN / TRACK_TOWN (P2 Middleton Railway's town removal).
pub const TRACK_PLAIN: &[&str] = &["7", "8", "9", "80", "81", "82", "83", "544", "545", "546", "60", "169"];
pub const TRACK_TOWN: &[&str] = &["3", "4", "58", "141", "142", "143", "144", "767", "768", "769", "X17"];
pub const UPGRADEABLE_S_YELLOW_CITY_TILE: &str = "57";
pub const UPGRADEABLE_S_YELLOW_ROTATIONS: &[u8] = &[2, 5];
/// MINOR_GREEN_UPGRADE colours (phase status `minors_green_upgrade`).
pub const MINOR_GREEN_UPGRADE: &[&str] = &["yellow", "green"];

/// `blocks_hexes_consent` (entities.rb): (company, its corporation, hex).
pub const CONSENT_BLOCKERS: &[(&str, &str, &str)] = &[
    ("C1", "LNWR", "M36"),
    ("C2", "GWR", "L39"),
    ("C3", "LBSCR", "M40"),
    ("C4", "SECR", "N39"),
    ("M15", "15", "N37"),
    ("M16", "16", "L37"),
];

/// A path end as `Part#<=` sees it: edges by number, nodes by class.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum End {
    Edge(u8),
    City,
    Town,
    Offboard,
}

fn end_le(a: End, b: End) -> bool {
    a == b
}

/// A city/town (Ruby node identity on one tile).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Ct {
    City(usize),
    Town(usize),
}

/// The parts of a tile the predicates read.
#[derive(Clone, Debug)]
pub(crate) struct TV {
    pub name: String,
    pub color: TileColor,
    pub label: Option<String>,
    pub cities: Vec<(i32, u8)>, // (revenue, slots)
    pub towns: Vec<i32>,
    pub paths: Vec<PathDef>,
}

fn base_name(name: &str) -> &str {
    name.split('-').next().unwrap_or(name)
}

fn color_index(c: TileColor) -> Option<u8> {
    // Engine::Tile::COLORS order; red is not in the upgrade chain.
    match c {
        TileColor::Red => None,
        other => Some(other.index()),
    }
}

pub(crate) fn color_name(c: TileColor) -> &'static str {
    match c {
        TileColor::White => "white",
        TileColor::Yellow => "yellow",
        TileColor::Green => "green",
        TileColor::Brown => "brown",
        TileColor::Gray => "gray",
        TileColor::Red => "red",
    }
}

impl TV {
    pub fn from_tile(t: &Tile) -> TV {
        TV {
            name: base_name(&t.name).to_string(),
            color: t.color,
            label: t.label.clone(),
            cities: t.cities.iter().map(|c| (c.revenue, c.slots)).collect(),
            towns: t.towns.iter().map(|t| t.revenue).collect(),
            paths: t.paths.clone(),
        }
    }

    pub fn from_def(d: &TileDef, rotation: u8) -> TV {
        let r = d.rotated(rotation);
        TV {
            name: d.name.clone(),
            color: r.color,
            label: r.label.clone(),
            cities: r.cities.iter().map(|c| (c.revenue, c.slots)).collect(),
            towns: r.towns.iter().map(|t| t.revenue).collect(),
            paths: r.paths,
        }
    }

    fn has_junction(&self) -> bool {
        self.paths.iter().any(path_junction)
    }

    /// `Tile#exits` (unique edge numbers, path order).
    pub fn exits(&self) -> Vec<u8> {
        let mut out = Vec::new();
        for p in &self.paths {
            for e in [&p.a, &p.b] {
                if let PathEndpoint::Edge(n) = e {
                    if !out.contains(n) {
                        out.push(*n);
                    }
                }
            }
        }
        out
    }

    /// `Path#ends` of path `i` (a junction end expands to the other
    /// junction paths' non-junction ends).
    fn ends(&self, i: usize) -> Vec<End> {
        let p = &self.paths[i];
        let mut out = Vec::new();
        for part in [&p.a, &p.b] {
            if *part == PathEndpoint::Junction {
                for (j, q) in self.paths.iter().enumerate() {
                    if j == i || !path_junction(q) {
                        continue;
                    }
                    for e in [&q.a, &q.b] {
                        if let Some(x) = to_end(e) {
                            out.push(x);
                        }
                    }
                }
            } else if let Some(x) = to_end(part) {
                out.push(x);
            }
        }
        out
    }

    fn rotated(&self, ticks: u8) -> TV {
        let mut t = self.clone();
        for p in &mut t.paths {
            p.a = p.a.rotated(ticks);
            p.b = p.b.rotated(ticks);
        }
        t
    }

    /// `Tile#city_town_edges`: per city/town reached by a path (first
    /// appearance order), the edges of the paths through it.
    fn ct_edges(&self) -> Vec<(Ct, Vec<u8>)> {
        let mut out: Vec<(Ct, Vec<u8>)> = Vec::new();
        for p in &self.paths {
            let ct = path_ct(p);
            let Some(ct) = ct else { continue };
            for e in [&p.a, &p.b] {
                if let PathEndpoint::Edge(n) = e {
                    match out.iter_mut().find(|(c, _)| *c == ct) {
                        Some((_, v)) => v.push(*n),
                        None => out.push((ct, vec![*n])),
                    }
                }
            }
        }
        out
    }

    /// `Node#exits` of each city/town (`@city_towns` order: cities, towns).
    fn city_town_exits(&self) -> Vec<Vec<u8>> {
        let mut out = Vec::new();
        let nodes: Vec<Ct> = (0..self.cities.len())
            .map(Ct::City)
            .chain((0..self.towns.len()).map(Ct::Town))
            .collect();
        for n in nodes {
            let mut ex = Vec::new();
            for p in &self.paths {
                if path_nodes(p).contains(&n) {
                    for e in [&p.a, &p.b] {
                        if let PathEndpoint::Edge(x) = e {
                            ex.push(*x);
                        }
                    }
                }
            }
            out.push(ex);
        }
        out
    }

    /// Ruby `max_revenue` of a path's nodes, sorted.
    fn path_revenues(&self, i: usize) -> Vec<i32> {
        let mut v: Vec<i32> = path_nodes(&self.paths[i])
            .iter()
            .map(|n| match n {
                Ct::City(c) => self.cities.get(*c).map_or(0, |x| x.0),
                Ct::Town(t) => self.towns.get(*t).copied().unwrap_or(0),
            })
            .collect();
        v.sort();
        v
    }
}

fn path_junction(p: &PathDef) -> bool {
    p.a == PathEndpoint::Junction || p.b == PathEndpoint::Junction
}

fn to_end(e: &PathEndpoint) -> Option<End> {
    match e {
        PathEndpoint::Edge(n) => Some(End::Edge(*n)),
        PathEndpoint::City(_) => Some(End::City),
        PathEndpoint::Town(_) => Some(End::Town),
        PathEndpoint::Offboard(_) => Some(End::Offboard),
        PathEndpoint::Junction => None,
    }
}

/// `path.city || path.town` (Path#separate_parts keeps the LAST city /
/// town of a, b).
fn path_ct(p: &PathDef) -> Option<Ct> {
    let mut city = None;
    let mut town = None;
    for e in [&p.a, &p.b] {
        match e {
            PathEndpoint::City(i) => city = Some(Ct::City(*i)),
            PathEndpoint::Town(i) => town = Some(Ct::Town(*i)),
            _ => {}
        }
    }
    city.or(town)
}

fn path_nodes(p: &PathDef) -> Vec<Ct> {
    let mut v = Vec::new();
    for e in [&p.a, &p.b] {
        match e {
            PathEndpoint::City(i) => v.push(Ct::City(*i)),
            PathEndpoint::Town(i) => v.push(Ct::Town(*i)),
            _ => {}
        }
    }
    v
}

/// `Path#<=`: every end of `a` (on tile `ta`) matches an end of `b`.
fn path_le(ta: &TV, a: usize, tb: &TV, b: usize) -> bool {
    let be = tb.ends(b);
    ta.ends(a).iter().all(|t| be.iter().any(|o| end_le(*t, *o)))
}

/// `Tile#paths_are_subset_of?(other.paths)` (any rotation of `from`).
fn paths_are_subset_of(from: &TV, to: &TV) -> bool {
    if from.has_junction() && to.paths.iter().any(path_junction) {
        // Lawson-to-Lawson: compare exits only.
        let other: Vec<u8> = to.exits();
        let exits = from.exits();
        return (0..6u8).any(|ticks| {
            let rot: Vec<u8> = other.iter().map(|e| (e + ticks) % 6).collect();
            exits.iter().all(|e| rot.contains(e))
        });
    }
    (0..6u8).any(|ticks| {
        let r = from.rotated(ticks);
        (0..r.paths.len()).all(|i| (0..to.paths.len()).any(|j| path_le(&r, i, to, j)))
    })
}

/// `Tile#city_town_edges_are_subset_of?`.
fn cte_subset(from: &TV, to: &TV) -> bool {
    let cte = from.ct_edges();
    let other = to.ct_edges();
    (0..6u8).any(|rot| {
        cte.iter()
            .all(|(_, city)| other.iter().any(|(_, oc)| city.iter().all(|e| oc.contains(&((e + rot) % 6)))))
    })
}

/// The number of Ruby `Part::Edge`s of a catalog tile (`Tile#edges.size`):
/// one per path edge end, a `lanes:N` path counting N times.
pub(crate) fn tile_edge_count(name: &str) -> usize {
    let Some((dsl, _)) = data::tile_dsl(name) else { return 0 };
    let mut n = 0;
    for part in dsl.split(';') {
        let Some(attrs) = part.trim().strip_prefix("path=") else { continue };
        let mut lanes = 1usize;
        let mut edges = 0usize;
        for kv in attrs.split(',') {
            let mut it = kv.splitn(2, ':');
            let k = it.next().unwrap_or("");
            let v = it.next().unwrap_or("");
            match k {
                "a" | "b" if v.parse::<u8>().is_ok() => edges += 1,
                "lanes" => lanes = v.parse().unwrap_or(1),
                _ => {}
            }
        }
        n += edges * lanes;
    }
    n
}

/// Stub edges of a hex's preprinted tile (map DSL `stub=edge:N`).
fn hex_stubs(coord: &str) -> Vec<u8> {
    let Some((dsl, _)) = data::hex_dsl(coord) else { return Vec::new() };
    dsl.split(';')
        .filter_map(|p| p.trim().strip_prefix("stub=edge:"))
        .filter_map(|v| v.parse().ok())
        .collect()
}

/// check_track_restrictions!'s "New track must override old one": every
/// old city/town's exits survive on some new city/town.
pub(crate) fn cities_towns_kept(old: &TV, new: &TV) -> bool {
    let new_ct = new.city_town_exits();
    old.city_town_exits()
        .iter()
        .all(|oc| new_ct.iter().any(|nc| oc.iter().all(|e| nc.contains(e))))
}

/// Whether the new tile has a path through one of `edges` that is not
/// `<=` any old path (check_track_restrictions!'s "new track").
pub(crate) fn new_track_at(old: &TV, new: &TV, edges: &[u8]) -> bool {
    (0..new.paths.len()).any(|np| {
        let p = &new.paths[np];
        let touches = [&p.a, &p.b].iter().any(|e| matches!(e, PathEndpoint::Edge(x) if edges.contains(x)));
        touches && !(0..old.paths.len()).any(|o| path_le(new, np, old, o))
    })
}

/// Which rotation predicate a lay uses (Track vs SpecialTrack variants).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RotRule {
    /// G1822 Tracker#legal_tile_rotation? (Track, and SpecialTrack lays
    /// without a special rotation rule).
    Normal,
    /// SpecialTrack P2 (Middleton, removes a town): exits only.
    RemoveTown,
    /// SpecialTrack P5 teleport onto the English Channel: rotation 0.
    LcdrChannel,
}

impl BaseGame {
    /// Ruby `future_label` still on the hex's tile: it transfers to each new
    /// tile until one of its colour is laid.
    fn g1822_future_label(&self, hi: usize) -> Option<(&'static str, &'static str)> {
        let hex = &self.hexes[hi];
        let (label, color) = self.title_def().hex_future_label(&hex.id)?;
        let fut = match color {
            "yellow" => TileColor::Yellow,
            "green" => TileColor::Green,
            "brown" => TileColor::Brown,
            "gray" => TileColor::Gray,
            _ => return None,
        };
        (hex.tile.color.index() < fut.index()).then_some((label, color))
    }

    /// `Game#upgrades_to?(hex.tile, to, special)` for the tile on `hi`.
    pub(crate) fn g1822_upgrades_to(&self, hi: usize, to: &TileDef, special: bool) -> bool {
        let hex = &self.hexes[hi];
        // Blue (sea / port) hexes: COLORS has no tile colour after blue
        // that a catalog tile carries (the engine stores blue as white).
        if hex.tile.id.starts_with("preprinted_") && self.g1822_hex_is_blue(&hex.id) {
            return false;
        }
        let from = TV::from_tile(&hex.tile);
        // G1822: the D35 S tile removes the town in yellow.
        if hex.id == data::UPGRADEABLE_S_HEX_NAME && from.color == TileColor::White {
            return to.name == UPGRADEABLE_S_YELLOW_CITY_TILE;
        }
        let step_ok = match (color_index(from.color), color_index(to.color)) {
            (Some(a), Some(b)) => b == a + 1,
            _ => false,
        };
        // G1822: P2 Middleton removes a town (colour step only).
        if TRACK_TOWN.contains(&from.name.as_str()) && TRACK_PLAIN.contains(&to.name.as_str()) {
            return step_ok;
        }
        if !step_ok {
            return false;
        }
        let to_tv = TV::from_def(to, 0);
        if !paths_are_subset_of(&from, &to_tv) {
            return false;
        }
        if special {
            return true;
        }
        // upgrades_to_correct_label?
        let label_ok = match self.g1822_future_label(hi) {
            Some((label, color)) if color_name(to.color) == color => to.label.as_deref() == Some(label),
            _ => from.label == to.label,
        };
        if !label_ok {
            return false;
        }
        // upgrades_to_correct_city_town?
        if from.towns.len() != to_tv.towns.len() {
            return false;
        }
        if from.label.is_none() && from.cities.len() != to_tv.cities.len() {
            return false;
        }
        if from.cities.len() > 1 && to_tv.cities.len() > 1 && !cte_subset(&from, &to_tv) {
            return false;
        }
        if from.label.is_some() && !from.cities.is_empty() && to_tv.cities.is_empty() {
            return false;
        }
        if from.color == TileColor::White
            && from.label.as_deref() == Some("OO")
            && from.cities.len() != to_tv.cities.len()
        {
            return false;
        }
        true
    }

    /// `Game#legal_tile_rotation?` (G1822) as process_action sees it.
    ///
    /// The rights owners are the players behind the hex's consent blockers
    /// (`blocks_hexes_consent`: concession C1-C4 or minor proxy M15/M16,
    /// then its corporation, then minor 14's London-exit right). Before a
    /// lay, G1822#preprocess_action strips the blockers' consent abilities
    /// WHILE iterating `tile.blockers`, and each removal deletes from that
    /// same array, so every second blocker survives (Ruby Array#each over
    /// a shrinking array). Then: the acting player among the surviving
    /// rights owners, or any surviving rights owner → legal; none → the
    /// stubs must be kept (legal_if_stubbed?).
    fn g1822_game_legal_rotation(&self, operator: &str, hi: usize, to: &TV) -> bool {
        let hex = &self.hexes[hi];
        if !hex.tile.id.starts_with("preprinted_") {
            return true;
        }
        let stubs = hex_stubs(&hex.id);
        if stubs.is_empty() {
            return true;
        }
        // blockers, in Ruby's add order
        let mut blockers: Vec<Option<u32>> = Vec::new();
        for (company, corp, h) in CONSENT_BLOCKERS {
            if *h != hex.id {
                continue;
            }
            if self.company_idx.contains_key(*company) {
                if !self.co_closed(company) {
                    let owner = self.co_owner(company);
                    let p = owner.player_id().or_else(|| owner.corp_sym().and_then(|c| self.g1822_corp_president(c)));
                    blockers.push(p);
                } else if self.g().deleted_open.iter().any(|c| c == company) {
                    blockers.push(None); // deleted, still a blocker; the bank's
                }
            }
            if self.corp_idx.contains_key(*corp) {
                blockers.push(self.g1822_corp_president(corp));
            }
        }
        if let (Some(exit), Some(&lh)) = (self.g().minor_14_city_exit, self.hex_idx.get(data::LONDON_HEX)) {
            let nb = self.g1822_neighbors().get(&lh).and_then(|m| m.get(&exit)).copied();
            if nb == Some(hi) {
                blockers.push(self.g1822_corp_president(super::MINOR_14_ID));
            }
        }
        // preprocess_action removes every other one
        let survivors: Vec<Option<u32>> = blockers.into_iter().skip(1).step_by(2).collect();
        let mut rights: Vec<u32> = survivors.into_iter().flatten().collect();
        rights.dedup();
        let acting = self.g1822_corp_president(operator);
        if acting.map_or(false, |p| rights.contains(&p)) || !rights.is_empty() {
            return true;
        }
        let exits = to.exits();
        stubs.iter().all(|s| exits.contains(s))
    }

    /// `Tracker#legal_tile_rotation?` for laying `to` (already rotated by
    /// `rotation`) on `hi` for `operator`, whose graph reaches the hex
    /// through `connected` edges (`hex_neighbors(operator, hex)`).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn g1822_legal_rotation(
        &self,
        rule: RotRule,
        operator: &str,
        hi: usize,
        to_def: &TileDef,
        rotation: u8,
        connected: &[u8],
        neighbors: &HashMap<usize, BTreeMap<u8, usize>>,
    ) -> bool {
        let hex = &self.hexes[hi];
        let to = TV::from_def(to_def, rotation);
        let from = TV::from_tile(&hex.tile);
        let nbrs = neighbors.get(&hi);
        let has_nb = |e: &u8| nbrs.map_or(false, |m| m.contains_key(e));
        match rule {
            RotRule::LcdrChannel => return rotation == 0,
            RotRule::RemoveTown => {
                if !self.g1822_game_legal_rotation(operator, hi, &to) {
                    return false;
                }
                let new_exits = to.exits();
                let old_exits = from.exits();
                return new_exits.iter().all(has_nb)
                    && new_exits.iter().any(|e| connected.contains(e))
                    && old_exits.iter().all(|e| new_exits.contains(e));
            }
            RotRule::Normal => {}
        }
        // G1822 Tracker: the white S hex's 57 at rotation 2 or 5 is legal.
        if hex.id == data::UPGRADEABLE_S_HEX_NAME
            && to_def.name == UPGRADEABLE_S_YELLOW_CITY_TILE
            && UPGRADEABLE_S_YELLOW_ROTATIONS.contains(&rotation)
        {
            return true;
        }
        if !self.g1822_game_legal_rotation(operator, hi, &to) {
            return false;
        }
        let old_ct = from.ct_edges();
        let new_ct = to.ct_edges();
        let new_exits = to.exits();
        let added_cities = new_ct.len().saturating_sub(old_ct.len());
        let multi_city_upgrade = to.cities.len() > 1 && from.cities.len() > 1;
        if !new_exits.iter().all(has_nb) {
            return false;
        }
        if !new_exits.iter().any(|e| connected.contains(e)) {
            return false;
        }
        // old_paths_maintained? (at this rotation)
        if !(0..from.paths.len()).all(|i| (0..to.paths.len()).any(|j| path_le(&from, i, &to, j))) {
            return false;
        }
        let fresh = new_ct
            .iter()
            .filter(|(_, n)| old_ct.iter().all(|(_, o)| !n.iter().any(|e| o.contains(e))))
            .count();
        if added_cities < fresh {
            return false;
        }
        if multi_city_upgrade {
            let ok = old_ct.iter().all(|(_, o)| {
                let mut dedup = o.clone();
                dedup.dedup();
                let uniq: HashSet<u8> = o.iter().copied().collect();
                let no_dups = uniq.len() == o.len();
                new_ct
                    .iter()
                    .filter(|(_, n)| no_dups && o.iter().all(|e| n.contains(e)))
                    .count()
                    == 1
            });
            if !ok {
                return false;
            }
        }
        // city_sizes_maintained (mixed slot counts only).
        let slots: HashSet<u8> = from.cities.iter().map(|c| c.1).collect();
        if slots.len() > 1 {
            let new_tile = BaseGame::tile_from_def(to_def, rotation);
            let map = BaseGame::g1822_city_map(&hex.tile, &new_tile);
            for (oi, m) in map.iter().enumerate() {
                if let Some(ni) = m {
                    if to.cities[*ni].1 < from.cities[oi].1 {
                        return false;
                    }
                }
            }
        }
        true
    }

    /// Ruby `Hex#city_map_for(tile)`: old city index → new city index.
    pub(crate) fn g1822_city_map(old: &Tile, new: &Tile) -> Vec<Option<usize>> {
        let old_exits = BaseGame::city_exits_from_tile(old);
        let new_exits = BaseGame::city_exits_from_tile(new);
        let n_old = old.cities.len();
        let n_new = new.cities.len();
        let mut map: Vec<Option<usize>> = vec![None; n_old];
        if old_exits.iter().all(|e| e.is_empty()) && n_old == n_new {
            for (i, m) in map.iter_mut().enumerate() {
                *m = Some(i);
            }
        } else {
            for (i, oe) in old_exits.iter().enumerate() {
                if !oe.is_empty() {
                    map[i] = new_exits.iter().position(|ne| oe.iter().all(|e| ne.contains(e)));
                }
            }
        }
        let mut claimed: Vec<usize> = map.iter().flatten().copied().collect();
        for i in 0..n_old {
            if map[i].is_some() {
                continue;
            }
            let mut nc = if i < n_new { Some(i) } else { None };
            if nc.map_or(false, |c| claimed.contains(&c)) {
                nc = (0..n_new).find(|c| !claimed.contains(c));
            }
            map[i] = nc;
            if let Some(c) = nc {
                claimed.push(c);
            }
        }
        map
    }

    /// The corporation's president (Ruby `corporation.owner`/`.player`).
    pub(crate) fn g1822_corp_president(&self, c: &str) -> Option<u32> {
        let &ci = self.corp_idx.get(c)?;
        self.corporations[ci].owner_id.player_id()
    }

    /// G1822 Tracker#potential_tiles for a corporation lay on `hi`:
    /// tile names in supply whose colour the phase allows (minors stop at
    /// green while `minors_green_upgrade` holds) and that upgrade the hex.
    pub(crate) fn g1822_potential_tiles(&self, operator: &str, hi: usize) -> Vec<String> {
        let colors: Vec<String> = if self.is_minor(operator) && self.phase_has_status("minors_green_upgrade") {
            MINOR_GREEN_UPGRADE.iter().map(|s| s.to_string()).collect()
        } else {
            self.phase.tiles.clone()
        };
        let mut names: Vec<&String> = self.tile_catalog.keys().collect();
        names.sort();
        names
            .into_iter()
            .filter(|n| self.tile_counts_remaining.get(*n).copied().unwrap_or(0) > 0)
            .filter(|n| {
                let d = &self.tile_catalog[*n];
                colors.iter().any(|c| c == color_name(d.color)) && self.g1822_upgrades_to(hi, d, false)
            })
            .cloned()
            .collect()
    }

    /// The UI's `max_exits` filter (TILE_UPGRADES_MUST_USE_MAX_EXITS =
    /// [:cities]): on a hex with cities keep, per colour, the tiles with the
    /// most edges among those with a legal rotation.
    pub(crate) fn g1822_max_exits(&self, hi: usize, tiles: Vec<String>) -> Vec<String> {
        if self.hexes[hi].tile.cities.is_empty() {
            return tiles;
        }
        let mut best: BTreeMap<u8, usize> = BTreeMap::new();
        for t in &tiles {
            let c = self.tile_catalog[t].color.index();
            let n = tile_edge_count(t);
            let e = best.entry(c).or_insert(0);
            *e = (*e).max(n);
        }
        tiles
            .into_iter()
            .filter(|t| best.get(&self.tile_catalog[t].color.index()) == Some(&tile_edge_count(t)))
            .collect()
    }

    /// `check_track_restrictions!` (:semi_restrictive, no town removal),
    /// evaluated AFTER the lay (`self` holds the laid tile on `hi`; `old`
    /// is the tile it replaced). English Channel and Cardiff are exempt.
    pub(crate) fn g1822_track_restrictions_ok(&self, operator: &str, hi: usize, old: &Tile) -> bool {
        let hex = &self.hexes[hi];
        if hex.id == data::ENGLISH_CHANNEL_HEX || hex.id == data::CARDIFF_HEX {
            return true;
        }
        let new = TV::from_tile(&hex.tile);
        let old = TV::from_tile(old);
        // 'New track must override old one'
        if !cities_towns_kept(&old, &new) {
            return false;
        }
        let graph = self.g1822_graph(operator);
        let mut used_new_track = old.paths.is_empty();
        let mut changed_city = false;
        for np in 0..new.paths.len() {
            if !graph.connected_paths.contains(&(hi, np)) {
                continue;
            }
            let op = (0..old.paths.len()).find(|&o| path_le(&new, np, &old, o));
            if op.is_none() {
                used_new_track = true;
            }
            let old_rev = op.map(|o| old.path_revenues(o));
            if old_rev.as_ref() != Some(&new.path_revenues(np)) {
                changed_city = true;
            }
        }
        used_new_track || changed_city
    }
}
