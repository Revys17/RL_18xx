//! 1867 tile-lay legality (bridge design T3) — the checks Ruby runs when a
//! PLAYER lays a tile (Ruby skips most of them while `@game.loading`, so
//! the replay handler trusts a record's tile and rotation):
//!
//! * `Game#upgrades_to?` (base): colour step, `Tile#paths_are_subset_of?`
//!   over all rotations, labels incl. `future_label` (J12's O on gray),
//!   city/town counts, `city_town_edges_are_subset_of?`, labelled cities
//!   never downgraded;
//! * `Tracker#legal_tile_rotation?` (base) + G1867's `StubsAreRestricted`
//!   (`legal_if_stubbed?`: a stub's edge must be an exit of the new tile —
//!   G15 edge 1, L10 edge 0, K13 edge 4): new exits face a neighbour
//!   (`hex.neighbors`: walls and untargeting red/gray/blue hexes excluded),
//!   some new exit reaches the corporation's connected edges, old paths
//!   kept at this rotation, the added-city count, the multi-city mapping
//!   (1867's Montreal X tiles), city sizes;
//! * `Tracker#potential_tiles` (phase colours, supply) and the UI's
//!   `TILE_UPGRADES_MUST_USE_MAX_EXITS = %i[cities]` filter;
//! * `check_track_restrictions!` (`:semi_restrictive`: new track or a
//!   changed city value on the laid tile's connected paths, and no
//!   city/town dropped).
//!
//! The predicates are a port of the 1822 agent's Ruby port
//! (`g1822/tile_rules.rs`) without 1822's special cases; they work on the
//! engine's placed `Tile` (paths already rotated) and catalog `TileDef`s.

use std::collections::{BTreeMap, HashSet};

use crate::game::BaseGame;
use crate::graph::Tile;
use crate::tiles::{PathDef, PathEndpoint, TileColor, TileDef};

/// Map DSL stubs (`stub=edge:N`, StubsAreRestricted): white hexes whose
/// first lay must keep the stub edge (g_1867/map.rb).
pub const STUBS: &[(&str, u8)] = &[("G15", 1), ("L10", 0), ("K13", 4)];

/// A path end as `Part#<=` sees it: edges by number, nodes by class.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum End {
    Edge(u8),
    City,
    Town,
    Offboard,
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
    pub color: TileColor,
    pub label: Option<String>,
    pub cities: Vec<(i32, u8)>, // (revenue, slots)
    pub towns: Vec<i32>,
    pub paths: Vec<PathDef>,
}

/// Ruby `Engine::Tile::COLORS` index; red (and the blue ports, built red)
/// are not in the upgrade chain.
fn color_index(c: TileColor) -> Option<u8> {
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
            let Some(ct) = path_ct(p) else { continue };
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
    ta.ends(a).iter().all(|t| be.iter().any(|o| t == o))
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

/// Ruby `Tile#edges.size` of a catalog tile: one `Part::Edge` per path
/// edge end (1867 has no lanes).
pub(crate) fn tile_edge_count(d: &TileDef) -> usize {
    d.paths
        .iter()
        .map(|p| usize::from(p.a.is_edge()) + usize::from(p.b.is_edge()))
        .sum()
}

/// check_track_restrictions!'s "New track must override old one": every
/// old city/town's exits survive on some new city/town.
pub(crate) fn cities_towns_kept(old: &TV, new: &TV) -> bool {
    let new_ct = new.city_town_exits();
    old.city_town_exits()
        .iter()
        .all(|oc| new_ct.iter().any(|nc| oc.iter().all(|e| nc.contains(e))))
}

impl BaseGame {
    /// Ruby `future_label` still on the hex's tile (J12: O on the gray
    /// tile): it stays on each new tile until one of its colour is laid.
    fn g1867_future_label(&self, hi: usize) -> Option<(&'static str, &'static str)> {
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

    /// `Game#upgrades_to?(hex.tile, to)` for the tile on `hi`.
    pub(crate) fn g1867_upgrades_to(&self, hi: usize, to: &TileDef) -> bool {
        let hex = &self.hexes[hi];
        let from = TV::from_tile(&hex.tile);
        let step_ok = match (color_index(from.color), color_index(to.color)) {
            (Some(a), Some(b)) => b == a + 1,
            _ => false,
        };
        if !step_ok {
            return false;
        }
        let to_tv = TV::from_def(to, 0);
        if !paths_are_subset_of(&from, &to_tv) {
            return false;
        }
        // upgrades_to_correct_label?
        let label_ok = match self.g1867_future_label(hi) {
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

    /// `Tracker#legal_tile_rotation?` (base + StubsAreRestricted) for
    /// laying `to_def` at `rotation` on `hi` for a corporation whose graph
    /// reaches the hex through `connected` edges
    /// (`hex_neighbors(entity, hex)`). `nb_exits` = the hex's
    /// `hex.neighbors` edges ([`BaseGame::passable_exits_for`]).
    pub(crate) fn g1867_legal_rotation(
        &self,
        hi: usize,
        to_def: &TileDef,
        rotation: u8,
        connected: &[u8],
        nb_exits: &[u8],
    ) -> bool {
        let hex = &self.hexes[hi];
        let to = TV::from_def(to_def, rotation);
        let from = TV::from_tile(&hex.tile);
        let new_exits = to.exits();
        // StubsAreRestricted#legal_if_stubbed?: the hex's stubs (on its
        // untiled white tile only — a laid tile carries none) are exits.
        if hex.tile.id.starts_with("preprinted_") || hex.tile.name.is_empty() || hex.tile.name == hex.id {
            for (h, e) in STUBS {
                if *h == hex.id && !new_exits.contains(e) {
                    return false;
                }
            }
        }
        let old_ct = from.ct_edges();
        let new_ct = to.ct_edges();
        let added_cities = new_ct.len().saturating_sub(old_ct.len());
        let multi_city_upgrade = to.cities.len() > 1 && from.cities.len() > 1;
        if !new_exits.iter().all(|e| nb_exits.contains(e)) {
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
        // 1867: every old city corresponds to exactly one new city
        // (`(oldct & newct) == oldct`: Array#& drops duplicates).
        if multi_city_upgrade {
            let ok = old_ct.iter().all(|(_, o)| {
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
            let map = g1867_city_map(&hex.tile, &new_tile);
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

    /// `Tracker#potential_tiles`: catalog tiles in supply whose colour the
    /// phase allows and that upgrade the hex (sorted by name).
    pub(crate) fn g1867_potential_tiles(&self, hi: usize) -> Vec<String> {
        let mut names: Vec<&String> = self.tile_catalog.keys().collect();
        names.sort();
        names
            .into_iter()
            .filter(|n| self.tile_counts_remaining.get(*n).copied().unwrap_or(0) > 0)
            .filter(|n| {
                let d = &self.tile_catalog[*n];
                self.phase.tiles.iter().any(|c| c == color_name(d.color)) && self.g1867_upgrades_to(hi, d)
            })
            .cloned()
            .collect()
    }

    /// The UI's `max_exits` filter (TILE_UPGRADES_MUST_USE_MAX_EXITS =
    /// [:cities]): on a hex with cities keep, per colour, the tiles with
    /// the most edges among those with a legal rotation.
    pub(crate) fn g1867_max_exits(&self, hi: usize, tiles: Vec<String>) -> Vec<String> {
        if self.hexes[hi].tile.cities.is_empty() {
            return tiles;
        }
        let mut best: BTreeMap<u8, usize> = BTreeMap::new();
        for t in &tiles {
            let d = &self.tile_catalog[t];
            let e = best.entry(d.color.index()).or_insert(0);
            *e = (*e).max(tile_edge_count(d));
        }
        tiles
            .into_iter()
            .filter(|t| {
                let d = &self.tile_catalog[t];
                best.get(&d.color.index()) == Some(&tile_edge_count(d))
            })
            .collect()
    }

    /// `check_track_restrictions!` (:semi_restrictive, no town removal),
    /// evaluated AFTER the lay: `self` holds the laid tile on `hi`, `old` is
    /// the tile it replaced, `connected_paths` the corporation's walked
    /// paths on `hi` after the lay (Ruby clears the graph first).
    pub(crate) fn g1867_track_restrictions_ok(&self, hi: usize, old: &Tile, connected_paths: &[usize]) -> bool {
        g1867_track_restrictions_ok_for(old, &self.hexes[hi].tile, connected_paths)
    }
}

/// `check_track_restrictions!` for `new` replacing `old` (see
/// [`BaseGame::g1867_track_restrictions_ok`]).
pub(crate) fn g1867_track_restrictions_ok_for(old: &Tile, new: &Tile, connected_paths: &[usize]) -> bool {
    {
        let new = TV::from_tile(new);
        let old = TV::from_tile(old);
        if !cities_towns_kept(&old, &new) {
            return false;
        }
        let mut used_new_track = old.paths.is_empty();
        let mut changed_city = false;
        for &np in connected_paths {
            if np >= new.paths.len() {
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

/// Ruby `Hex#city_map_for(tile)`: old city index → new city index.
pub(crate) fn g1867_city_map(old: &Tile, new: &Tile) -> Vec<Option<usize>> {
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
