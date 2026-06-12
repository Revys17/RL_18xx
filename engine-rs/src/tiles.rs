use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Tile color
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TileColor {
    White,
    Yellow,
    Green,
    Brown,
    Gray,
    Red,
}

impl TileColor {
    /// Color index in the upgrade chain: white(0) → yellow(1) → green(2) → brown(3) → gray(4).
    pub fn index(self) -> u8 {
        match self {
            TileColor::White => 0,
            TileColor::Yellow => 1,
            TileColor::Green => 2,
            TileColor::Brown => 3,
            TileColor::Gray => 4,
            TileColor::Red => 5,
        }
    }

    /// Returns true if upgrading from `self` to `to` is a valid color step.
    pub fn can_upgrade_to(self, to: TileColor) -> bool {
        to.index() == self.index() + 1
    }

    /// The next color in the upgrade chain, or None if at the top.
    pub fn next_color(self) -> Option<TileColor> {
        match self {
            TileColor::White => Some(TileColor::Yellow),
            TileColor::Yellow => Some(TileColor::Green),
            TileColor::Green => Some(TileColor::Brown),
            TileColor::Brown => Some(TileColor::Gray),
            TileColor::Gray | TileColor::Red => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Path endpoints
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PathEndpoint {
    Edge(u8),
    City(usize),
    Town(usize),
    Offboard(usize),
    Junction,
}

impl PathEndpoint {
    /// Rotate edge references by `rotation` (mod 6). Non-edge endpoints unchanged.
    pub fn rotated(&self, rotation: u8) -> PathEndpoint {
        match self {
            PathEndpoint::Edge(n) => PathEndpoint::Edge((*n + rotation) % 6),
            other => other.clone(),
        }
    }

    pub fn is_edge(&self) -> bool {
        matches!(self, PathEndpoint::Edge(_))
    }

    pub fn edge_num(&self) -> Option<u8> {
        match self {
            PathEndpoint::Edge(n) => Some(*n),
            _ => None,
        }
    }

    pub fn is_node(&self) -> bool {
        matches!(
            self,
            PathEndpoint::City(_) | PathEndpoint::Town(_) | PathEndpoint::Offboard(_)
        )
    }
}

/// Lenient endpoint matching for path subset checks.
/// Edges must match exactly. City/Town/Offboard match by TYPE only (ignoring index).
/// Junction matches Junction.
fn endpoints_match_lenient(a: &PathEndpoint, b: &PathEndpoint) -> bool {
    match (a, b) {
        (PathEndpoint::Edge(e1), PathEndpoint::Edge(e2)) => e1 == e2,
        (PathEndpoint::City(_), PathEndpoint::City(_)) => true,
        (PathEndpoint::Town(_), PathEndpoint::Town(_)) => true,
        (PathEndpoint::Offboard(_), PathEndpoint::Offboard(_)) => true,
        (PathEndpoint::Junction, PathEndpoint::Junction) => true,
        _ => false,
    }
}

// ---------------------------------------------------------------------------
// Tile definition structs
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PathDef {
    pub a: PathEndpoint,
    pub b: PathEndpoint,
    pub terminal: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CityDef {
    pub revenue: i32,
    pub slots: u8,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TownDef {
    pub revenue: i32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OffboardDef {
    pub yellow_revenue: i32,
    pub brown_revenue: i32,
    /// 4-tier titles (1867) also price the green/gray phases; None = tier
    /// not defined (the highest defined tier at or below the phase's latest
    /// tile color applies).
    #[serde(default)]
    pub green_revenue: Option<i32>,
    #[serde(default)]
    pub gray_revenue: Option<i32>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UpgradeDef {
    pub cost: i32,
    pub terrain: String,
}

/// A fully parsed tile definition with all connectivity information.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TileDef {
    pub name: String,
    pub color: TileColor,
    pub paths: Vec<PathDef>,
    pub cities: Vec<CityDef>,
    pub towns: Vec<TownDef>,
    pub offboards: Vec<OffboardDef>,
    pub edges: Vec<u8>,
    pub upgrades: Vec<UpgradeDef>,
    pub label: Option<String>,
    pub has_junction: bool,
}

impl TileDef {
    /// Return a rotated copy of this tile definition.
    pub fn rotated(&self, rotation: u8) -> TileDef {
        if rotation == 0 {
            return self.clone();
        }
        let rotated_paths = self
            .paths
            .iter()
            .map(|p| PathDef {
                a: p.a.rotated(rotation),
                b: p.b.rotated(rotation),
                terminal: p.terminal,
            })
            .collect();
        let rotated_edges = self.edges.iter().map(|e| (*e + rotation) % 6).collect();
        TileDef {
            name: self.name.clone(),
            color: self.color,
            paths: rotated_paths,
            cities: self.cities.clone(),
            towns: self.towns.clone(),
            offboards: self.offboards.clone(),
            edges: rotated_edges,
            upgrades: self.upgrades.clone(),
            label: self.label.clone(),
            has_junction: self.has_junction,
        }
    }

    /// Collect unique edge numbers from all paths (public for external callers).
    pub fn compute_edges_pub(paths: &[PathDef]) -> Vec<u8> {
        Self::compute_edges(paths)
    }

    /// Collect unique edge numbers from all paths.
    fn compute_edges(paths: &[PathDef]) -> Vec<u8> {
        let set: HashSet<u8> = paths
            .iter()
            .flat_map(|p| [&p.a, &p.b])
            .filter_map(|ep| ep.edge_num())
            .collect();
        let mut edges: Vec<u8> = set.into_iter().collect();
        edges.sort();
        edges
    }

    /// Check if all paths of `old` exist in `self` at a specific rotation.
    /// Uses lenient matching: edge endpoints must match exactly, but city/town
    /// indices are treated as equivalent (City(0) matches City(1)).
    pub fn paths_are_superset_of(&self, old: &TileDef) -> bool {
        old.paths.iter().all(|op| {
            self.paths.iter().any(|np| {
                (endpoints_match_lenient(&np.a, &op.a) && endpoints_match_lenient(&np.b, &op.b))
                    || (endpoints_match_lenient(&np.a, &op.b)
                        && endpoints_match_lenient(&np.b, &op.a))
            })
        })
    }

    /// Check if all paths of `old` exist in `self` at ANY of the 6 rotations.
    /// Returns true if there's at least one rotation where all old paths are present.
    pub fn paths_are_superset_of_any_rotation(&self, old: &TileDef) -> bool {
        if old.paths.is_empty() {
            return true;
        }
        for rotation in 0..6u8 {
            let rotated = self.rotated(rotation);
            if rotated.paths_are_superset_of(old) {
                return true;
            }
        }
        false
    }

    /// Check if `self` (at a specific rotation) is a valid upgrade for `old_tile`.
    /// Mirrors Python's `upgrades_to(from_tile, to_tile)`.
    pub fn is_valid_upgrade_for(&self, old: &TileDef) -> bool {
        // 1. Color must be exactly one step up
        if !old.color.can_upgrade_to(self.color) {
            return false;
        }

        // 2. Old paths must exist in new tile (at any rotation)
        if !self.paths_are_superset_of_any_rotation(old) {
            return false;
        }

        // 3. Label match (None == None is OK)
        if old.label != self.label {
            return false;
        }

        // 4. Town count must match
        if old.towns.len() != self.towns.len() {
            return false;
        }

        // 5. City count must match (when no label)
        if old.label.is_none() && old.cities.len() != self.cities.len() {
            return false;
        }

        // 6. If old has label but no cities, new can't have cities
        if old.label.is_some() && old.cities.is_empty() && !self.cities.is_empty() {
            return false;
        }

        true
    }

    /// Return per-(city|town) edges. For each node, the set of hex edges that
    /// have a path ending at that node. Used to validate multi-city upgrades.
    /// Index ordering: cities first, then towns (matching path endpoint refs).
    pub fn city_town_edges(&self) -> Vec<std::collections::HashSet<u8>> {
        let mut out: Vec<std::collections::HashSet<u8>> =
            (0..self.cities.len() + self.towns.len())
                .map(|_| std::collections::HashSet::new())
                .collect();
        for p in &self.paths {
            let mut endpoints = vec![&p.a, &p.b];
            // For each endpoint, if it is City/Town, the OTHER endpoint may be
            // an Edge — record that edge for this node.
            for _ in 0..2 {
                let ep = endpoints.remove(0);
                let other = endpoints[0];
                let node_idx = match ep {
                    PathEndpoint::City(i) => Some(*i),
                    PathEndpoint::Town(i) => Some(self.cities.len() + *i),
                    _ => None,
                };
                if let (Some(idx), PathEndpoint::Edge(e)) = (node_idx, other) {
                    if idx < out.len() {
                        out[idx].insert(*e);
                    }
                }
                endpoints.push(ep);
            }
        }
        out
    }

    /// Find all rotations (0-5) at which `self` maintains all old paths.
    pub fn legal_rotations_for(&self, old: &TileDef, valid_exits: &[u8]) -> Vec<u8> {
        let mut rotations = Vec::new();
        let multi_city_upgrade = self.cities.len() > 1 && old.cities.len() > 1;
        let old_ct_edges_full = if multi_city_upgrade {
            Some(old.city_town_edges())
        } else {
            None
        };
        for rotation in 0..6u8 {
            let rotated = self.rotated(rotation);

            // All old paths must be maintained
            if !rotated.paths_are_superset_of(old) {
                continue;
            }

            // All new exits must be valid hex edges (have neighbors)
            if !rotated.edges.iter().all(|e| valid_exits.contains(e)) {
                continue;
            }

            // For multi-city upgrades (e.g. OO/NY tiles): each NEW city
            // can absorb at most ONE old city's edges. Mirrors a stricter
            // interpretation of Python's per-city mapping — two old cities
            // both mapping to the same new city means the upgrade collapses
            // them, which Python's ``Hex.city_map_for`` does not allow when
            // they are distinct cities.
            if let Some(ref old_edges) = old_ct_edges_full {
                let new_edges = rotated.city_town_edges();
                let n_cities_old = old.cities.len();
                let n_cities_new = rotated.cities.len();
                // Build per-old-city candidate new cities.
                // Use bipartite matching: try to assign each old city to a
                // distinct new city whose edges are a superset.
                let mut candidates: Vec<Vec<usize>> = Vec::new();
                for i in 0..n_cities_old {
                    let oe = &old_edges[i];
                    if oe.is_empty() {
                        // Old city without exits — anything goes; skip from
                        // matching to be permissive.
                        candidates.push((0..n_cities_new).collect());
                        continue;
                    }
                    let mut cands: Vec<usize> = Vec::new();
                    for j in 0..n_cities_new {
                        let ne = &new_edges[j];
                        if oe.is_subset(ne) {
                            cands.push(j);
                        }
                    }
                    candidates.push(cands);
                }
                // Greedy with backtracking (sizes are tiny — up to ~3).
                fn assign(
                    i: usize,
                    cands: &Vec<Vec<usize>>,
                    used: &mut Vec<bool>,
                ) -> bool {
                    if i == cands.len() {
                        return true;
                    }
                    for &j in &cands[i] {
                        if j < used.len() && !used[j] {
                            used[j] = true;
                            if assign(i + 1, cands, used) {
                                return true;
                            }
                            used[j] = false;
                        }
                    }
                    false
                }
                let mut used = vec![false; n_cities_new];
                if !assign(0, &candidates, &mut used) {
                    continue;
                }
            }

            rotations.push(rotation);
        }
        rotations
    }
}

// ---------------------------------------------------------------------------
// DSL parser
// ---------------------------------------------------------------------------

/// Parse a key=value pair from a part segment like "revenue:30" or "slots:2".
fn parse_kv(segment: &str) -> (&str, &str) {
    let mut parts = segment.splitn(2, ':');
    let key = parts.next().unwrap_or("");
    let val = parts.next().unwrap_or("");
    (key, val)
}

/// Parse a path endpoint reference like "0" (edge), "_0" (city/town ref), or "junction".
fn parse_path_ref(
    s: &str,
    city_count: usize,
    town_count: usize,
    offboard_count: usize,
) -> PathEndpoint {
    if s == "junction" {
        return PathEndpoint::Junction;
    }
    if let Some(idx_str) = s.strip_prefix('_') {
        let idx: usize = idx_str.parse().unwrap_or(0);
        // Determine if this references a city, town, or offboard based on index
        // Node references: _0, _1, ... map to cities first, then towns, then offboards
        if idx < city_count {
            PathEndpoint::City(idx)
        } else if idx < city_count + town_count {
            PathEndpoint::Town(idx - city_count)
        } else if idx < city_count + town_count + offboard_count {
            PathEndpoint::Offboard(idx - city_count - town_count)
        } else {
            // Fallback: if only offboards exist, map directly
            if city_count == 0 && town_count == 0 && offboard_count > 0 {
                PathEndpoint::Offboard(idx)
            } else {
                PathEndpoint::City(idx)
            }
        }
    } else {
        let edge: u8 = s.parse().unwrap_or(0);
        PathEndpoint::Edge(edge)
    }
}

/// Parse a tile DSL string into a TileDef.
pub fn parse_tile(name: &str, code: &str, color: TileColor) -> TileDef {
    let parts: Vec<&str> = code.split(';').collect();

    // First pass: count cities, towns, offboards (needed for path ref resolution)
    let mut city_count = 0usize;
    let mut town_count = 0usize;
    let mut offboard_count = 0usize;

    for part in &parts {
        let trimmed = part.trim();
        if trimmed.starts_with("city=") {
            city_count += 1;
        } else if trimmed.starts_with("town=") {
            town_count += 1;
        } else if trimmed.starts_with("offboard=") {
            offboard_count += 1;
        }
    }

    let mut cities = Vec::new();
    let mut towns = Vec::new();
    let mut offboards = Vec::new();
    let mut paths = Vec::new();
    let mut upgrades_list = Vec::new();
    let mut label = None;
    let mut has_junction = false;

    for part in &parts {
        let trimmed = part.trim();

        if let Some(attrs) = trimmed.strip_prefix("city=") {
            let mut revenue = 0i32;
            let mut slots = 1u8;
            for segment in attrs.split(',') {
                let (k, v) = parse_kv(segment);
                match k {
                    "revenue" => revenue = v.parse().unwrap_or(0),
                    "slots" => slots = v.parse().unwrap_or(1),
                    _ => {} // loc, hide, groups — ignored for connectivity
                }
            }
            cities.push(CityDef { revenue, slots });
        } else if let Some(attrs) = trimmed.strip_prefix("town=") {
            let mut revenue = 0i32;
            for segment in attrs.split(',') {
                let (k, v) = parse_kv(segment);
                if k == "revenue" {
                    revenue = v.parse().unwrap_or(0);
                }
            }
            towns.push(TownDef { revenue });
        } else if let Some(attrs) = trimmed.strip_prefix("offboard=") {
            let mut yellow_revenue = 0i32;
            let mut brown_revenue = 0i32;
            let mut green_revenue = None;
            let mut gray_revenue = None;
            for segment in attrs.split(',') {
                let (k, v) = parse_kv(segment);
                if k == "revenue" {
                    // Format: "yellow_V|green_V|brown_V|gray_V" (any subset;
                    // 1830 uses yellow|brown, 1867 all four), or a bare flat
                    // value "V" (1867's blue lake ports pay the same at every
                    // phase).
                    for phase_rev in v.split('|') {
                        if let Some(val_str) = phase_rev.strip_prefix("yellow_") {
                            yellow_revenue = val_str.parse().unwrap_or(0);
                        } else if let Some(val_str) = phase_rev.strip_prefix("green_") {
                            green_revenue = val_str.parse().ok();
                        } else if let Some(val_str) = phase_rev.strip_prefix("brown_") {
                            brown_revenue = val_str.parse().unwrap_or(0);
                        } else if let Some(val_str) = phase_rev.strip_prefix("gray_") {
                            gray_revenue = val_str.parse().ok();
                        } else if let Ok(flat) = phase_rev.parse::<i32>() {
                            yellow_revenue = flat;
                            brown_revenue = flat;
                        }
                    }
                }
            }
            offboards.push(OffboardDef {
                yellow_revenue,
                brown_revenue,
                green_revenue,
                gray_revenue,
            });
        } else if let Some(attrs) = trimmed.strip_prefix("path=") {
            let mut a_ref = "";
            let mut b_ref = "";
            let mut terminal = false;
            for segment in attrs.split(',') {
                let (k, v) = parse_kv(segment);
                match k {
                    "a" => a_ref = v,
                    "b" => b_ref = v,
                    "terminal" => terminal = v == "true" || v == "1",
                    _ => {}
                }
            }
            let a = parse_path_ref(a_ref, city_count, town_count, offboard_count);
            let b = parse_path_ref(b_ref, city_count, town_count, offboard_count);
            paths.push(PathDef { a, b, terminal });
        } else if let Some(attrs) = trimmed.strip_prefix("upgrade=") {
            let mut cost = 0i32;
            let mut terrain = String::new();
            for segment in attrs.split(',') {
                let (k, v) = parse_kv(segment);
                match k {
                    "cost" => cost = v.parse().unwrap_or(0),
                    "terrain" => terrain = v.to_string(),
                    _ => {}
                }
            }
            upgrades_list.push(UpgradeDef { cost, terrain });
        } else if let Some(attrs) = trimmed.strip_prefix("label=") {
            label = Some(attrs.to_string());
        } else if trimmed == "junction" {
            has_junction = true;
        }
        // stub, border — ignored for connectivity purposes
    }

    let edges = TileDef::compute_edges(&paths);

    TileDef {
        name: name.to_string(),
        color,
        paths,
        cities,
        towns,
        offboards,
        edges,
        upgrades: upgrades_list,
        label,
        has_junction,
    }
}

// ---------------------------------------------------------------------------
// Tile catalog for 1830
// ---------------------------------------------------------------------------

/// Build the complete tile catalog for 1830: all 46 purchasable tiles plus
/// preprinted hex tiles (red/gray).
pub fn tile_catalog_1830() -> Arc<HashMap<String, TileDef>> {
    let mut catalog = HashMap::new();

    // === Yellow tiles ===
    let y = TileColor::Yellow;
    catalog.insert(
        "1".into(),
        parse_tile(
            "1",
            "town=revenue:10;town=revenue:10;path=a:1,b:_0;path=a:_0,b:3;path=a:0,b:_1;path=a:_1,b:4",
            y,
        ),
    );
    catalog.insert(
        "2".into(),
        parse_tile(
            "2",
            "town=revenue:10;town=revenue:10;path=a:0,b:_0;path=a:_0,b:3;path=a:1,b:_1;path=a:_1,b:2",
            y,
        ),
    );
    catalog.insert(
        "3".into(),
        parse_tile("3", "town=revenue:10;path=a:0,b:_0;path=a:_0,b:1", y),
    );
    catalog.insert(
        "4".into(),
        parse_tile("4", "town=revenue:10;path=a:0,b:_0;path=a:_0,b:3", y),
    );
    catalog.insert("7".into(), parse_tile("7", "path=a:0,b:1", y));
    catalog.insert("8".into(), parse_tile("8", "path=a:0,b:2", y));
    catalog.insert("9".into(), parse_tile("9", "path=a:0,b:3", y));
    catalog.insert(
        "55".into(),
        parse_tile(
            "55",
            "town=revenue:10;town=revenue:10;path=a:0,b:_0;path=a:_0,b:3;path=a:1,b:_1;path=a:_1,b:4",
            y,
        ),
    );
    catalog.insert(
        "56".into(),
        parse_tile(
            "56",
            "town=revenue:10;town=revenue:10;path=a:0,b:_0;path=a:_0,b:2;path=a:1,b:_1;path=a:_1,b:3",
            y,
        ),
    );
    catalog.insert(
        "57".into(),
        parse_tile("57", "city=revenue:20;path=a:0,b:_0;path=a:_0,b:3", y),
    );
    catalog.insert(
        "58".into(),
        parse_tile("58", "town=revenue:10;path=a:0,b:_0;path=a:_0,b:2", y),
    );
    catalog.insert(
        "69".into(),
        parse_tile(
            "69",
            "town=revenue:10;town=revenue:10;path=a:0,b:_0;path=a:_0,b:3;path=a:2,b:_1;path=a:_1,b:4",
            y,
        ),
    );

    // === Green tiles ===
    let g = TileColor::Green;
    catalog.insert(
        "14".into(),
        parse_tile(
            "14",
            "city=revenue:30,slots:2;path=a:0,b:_0;path=a:1,b:_0;path=a:3,b:_0;path=a:4,b:_0",
            g,
        ),
    );
    catalog.insert(
        "15".into(),
        parse_tile(
            "15",
            "city=revenue:30,slots:2;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0;path=a:3,b:_0",
            g,
        ),
    );
    catalog.insert(
        "16".into(),
        parse_tile("16", "path=a:0,b:2;path=a:1,b:3", g),
    );
    catalog.insert(
        "18".into(),
        parse_tile("18", "path=a:0,b:3;path=a:1,b:2", g),
    );
    catalog.insert(
        "19".into(),
        parse_tile("19", "path=a:0,b:3;path=a:2,b:4", g),
    );
    catalog.insert(
        "20".into(),
        parse_tile("20", "path=a:0,b:3;path=a:1,b:4", g),
    );
    catalog.insert(
        "23".into(),
        parse_tile("23", "path=a:0,b:3;path=a:0,b:4", g),
    );
    catalog.insert(
        "24".into(),
        parse_tile("24", "path=a:0,b:3;path=a:0,b:2", g),
    );
    catalog.insert(
        "25".into(),
        parse_tile("25", "path=a:0,b:2;path=a:0,b:4", g),
    );
    catalog.insert(
        "26".into(),
        parse_tile("26", "path=a:0,b:3;path=a:0,b:5", g),
    );
    catalog.insert(
        "27".into(),
        parse_tile("27", "path=a:0,b:3;path=a:0,b:1", g),
    );
    catalog.insert(
        "28".into(),
        parse_tile("28", "path=a:0,b:4;path=a:0,b:5", g),
    );
    catalog.insert(
        "29".into(),
        parse_tile("29", "path=a:0,b:2;path=a:0,b:1", g),
    );
    catalog.insert(
        "53".into(),
        parse_tile(
            "53",
            "city=revenue:50;path=a:0,b:_0;path=a:2,b:_0;path=a:4,b:_0;label=B",
            g,
        ),
    );
    catalog.insert(
        "54".into(),
        parse_tile(
            "54",
            "city=revenue:60,loc:0.5;city=revenue:60,loc:2.5;path=a:0,b:_0;path=a:_0,b:1;path=a:2,b:_1;path=a:_1,b:3;label=NY",
            g,
        ),
    );
    catalog.insert(
        "59".into(),
        parse_tile(
            "59",
            "city=revenue:40;city=revenue:40;path=a:0,b:_0;path=a:2,b:_1;label=OO",
            g,
        ),
    );

    // === Brown tiles ===
    let b = TileColor::Brown;
    catalog.insert(
        "39".into(),
        parse_tile("39", "path=a:0,b:2;path=a:0,b:1;path=a:1,b:2", b),
    );
    catalog.insert(
        "40".into(),
        parse_tile("40", "path=a:0,b:2;path=a:2,b:4;path=a:0,b:4", b),
    );
    catalog.insert(
        "41".into(),
        parse_tile("41", "path=a:0,b:3;path=a:0,b:1;path=a:1,b:3", b),
    );
    catalog.insert(
        "42".into(),
        parse_tile("42", "path=a:0,b:3;path=a:3,b:5;path=a:0,b:5", b),
    );
    catalog.insert(
        "43".into(),
        parse_tile(
            "43",
            "path=a:0,b:3;path=a:0,b:2;path=a:1,b:3;path=a:1,b:2",
            b,
        ),
    );
    catalog.insert(
        "44".into(),
        parse_tile(
            "44",
            "path=a:0,b:3;path=a:1,b:4;path=a:0,b:1;path=a:3,b:4",
            b,
        ),
    );
    catalog.insert(
        "45".into(),
        parse_tile(
            "45",
            "path=a:0,b:3;path=a:2,b:4;path=a:0,b:4;path=a:2,b:3",
            b,
        ),
    );
    catalog.insert(
        "46".into(),
        parse_tile(
            "46",
            "path=a:0,b:3;path=a:2,b:4;path=a:3,b:4;path=a:0,b:2",
            b,
        ),
    );
    catalog.insert(
        "47".into(),
        parse_tile(
            "47",
            "path=a:0,b:3;path=a:1,b:4;path=a:1,b:3;path=a:0,b:4",
            b,
        ),
    );
    catalog.insert(
        "61".into(),
        parse_tile(
            "61",
            "city=revenue:60;path=a:0,b:_0;path=a:2,b:_0;path=a:3,b:_0;path=a:4,b:_0;label=B",
            b,
        ),
    );
    catalog.insert(
        "62".into(),
        parse_tile(
            "62",
            "city=revenue:80,slots:2;city=revenue:80,slots:2;path=a:0,b:_0;path=a:_0,b:1;path=a:2,b:_1;path=a:_1,b:3;label=NY",
            b,
        ),
    );
    catalog.insert(
        "63".into(),
        parse_tile(
            "63",
            "city=revenue:40,slots:2;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0;path=a:3,b:_0;path=a:4,b:_0;path=a:5,b:_0",
            b,
        ),
    );
    catalog.insert(
        "64".into(),
        parse_tile(
            "64",
            "city=revenue:50;city=revenue:50,loc:3.5;path=a:0,b:_0;path=a:_0,b:2;path=a:3,b:_1;path=a:_1,b:4;label=OO",
            b,
        ),
    );
    catalog.insert(
        "65".into(),
        parse_tile(
            "65",
            "city=revenue:50;city=revenue:50,loc:2.5;path=a:0,b:_0;path=a:_0,b:4;path=a:2,b:_1;path=a:_1,b:3;label=OO",
            b,
        ),
    );
    catalog.insert(
        "66".into(),
        parse_tile(
            "66",
            "city=revenue:50;city=revenue:50,loc:1.5;path=a:0,b:_0;path=a:_0,b:3;path=a:1,b:_1;path=a:_1,b:2;label=OO",
            b,
        ),
    );
    catalog.insert(
        "67".into(),
        parse_tile(
            "67",
            "city=revenue:50;city=revenue:50;path=a:0,b:_0;path=a:_0,b:3;path=a:2,b:_1;path=a:_1,b:4;label=OO",
            b,
        ),
    );
    catalog.insert(
        "68".into(),
        parse_tile(
            "68",
            "city=revenue:50;city=revenue:50;path=a:0,b:_0;path=a:_0,b:3;path=a:1,b:_1;path=a:_1,b:4;label=OO",
            b,
        ),
    );
    catalog.insert(
        "70".into(),
        parse_tile(
            "70",
            "path=a:0,b:1;path=a:0,b:2;path=a:1,b:3;path=a:2,b:3",
            b,
        ),
    );

    Arc::new(catalog)
}

// ---------------------------------------------------------------------------
// Tile catalog for 1867
// ---------------------------------------------------------------------------

/// Build the complete tile catalog for 1867: 60 standard tile types (the ids
/// in g1867::tile_counts; DSL transcribed verbatim from tobymao/18xx
/// lib/engine/config/tile.rb) plus the 8 custom 1867 tiles X1-X8 (map.rb
/// TILES; X1-X4 green Montreal, X5-X7 brown Montreal, X8 gray Ottawa).
/// Tile ids 1867 shares with 1830 are cloned from `tile_catalog_1830` so the
/// definitions can never drift apart.
pub fn tile_catalog_1867() -> Arc<HashMap<String, TileDef>> {
    let base = tile_catalog_1830();
    let mut catalog = HashMap::new();

    // === Standard ids shared with 1830 (clone, don't re-transcribe) ===
    const SHARED_WITH_1830: &[&str] = &[
        "3", "4", "7", "8", "9", "14", "15", "16", "18", "19", "20", "23", "24", "25", "26", "27",
        "28", "29", "39", "40", "41", "42", "43", "44", "45", "46", "47", "57", "58", "63", "70",
    ];
    for &id in SHARED_WITH_1830 {
        catalog.insert(
            id.to_string(),
            base.get(id)
                .unwrap_or_else(|| panic!("tile {id} missing from the 1830 catalog"))
                .clone(),
        );
    }

    // === Yellow tiles (1867-only ids) ===
    let y = TileColor::Yellow;
    catalog.insert(
        "5".into(),
        parse_tile("5", "city=revenue:20;path=a:0,b:_0;path=a:1,b:_0", y),
    );
    catalog.insert(
        "6".into(),
        parse_tile("6", "city=revenue:20;path=a:0,b:_0;path=a:2,b:_0", y),
    );
    catalog.insert(
        "201".into(),
        parse_tile("201", "city=revenue:30;path=a:0,b:_0;path=a:1,b:_0;label=Y", y),
    );
    catalog.insert(
        "202".into(),
        parse_tile("202", "city=revenue:30;path=a:0,b:_0;path=a:2,b:_0;label=Y", y),
    );
    catalog.insert(
        "621".into(),
        parse_tile("621", "city=revenue:30;path=a:0,b:_0;path=a:_0,b:3;label=Y", y),
    );

    // === Green tiles (1867-only ids) ===
    let g = TileColor::Green;
    catalog.insert("17".into(), parse_tile("17", "path=a:1,b:3;path=a:0,b:4", g));
    catalog.insert("21".into(), parse_tile("21", "path=a:0,b:2;path=a:3,b:4", g));
    catalog.insert("22".into(), parse_tile("22", "path=a:0,b:4;path=a:2,b:3", g));
    catalog.insert("30".into(), parse_tile("30", "path=a:0,b:4;path=a:0,b:1", g));
    catalog.insert("31".into(), parse_tile("31", "path=a:0,b:2;path=a:0,b:5", g));
    catalog.insert(
        "87".into(),
        parse_tile(
            "87",
            "town=revenue:10;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0;path=a:3,b:_0",
            g,
        ),
    );
    catalog.insert(
        "88".into(),
        parse_tile(
            "88",
            "town=revenue:10;path=a:0,b:_0;path=a:1,b:_0;path=a:3,b:_0;path=a:4,b:_0",
            g,
        ),
    );
    catalog.insert(
        "120".into(),
        parse_tile(
            "120",
            "city=revenue:60;city=revenue:60;path=a:0,b:_0;path=a:_0,b:1;path=a:2,b:_1;path=a:_1,b:3;label=T",
            g,
        ),
    );
    catalog.insert(
        "204".into(),
        parse_tile(
            "204",
            "town=revenue:10;path=a:0,b:_0;path=a:2,b:_0;path=a:3,b:_0;path=a:4,b:_0",
            g,
        ),
    );
    catalog.insert(
        "207".into(),
        parse_tile(
            "207",
            "city=revenue:40,slots:2;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0;path=a:3,b:_0;label=Y",
            g,
        ),
    );
    catalog.insert(
        "208".into(),
        parse_tile(
            "208",
            "city=revenue:40,slots:2;path=a:0,b:_0;path=a:1,b:_0;path=a:3,b:_0;path=a:4,b:_0;label=Y",
            g,
        ),
    );
    catalog.insert(
        "619".into(),
        parse_tile(
            "619",
            "city=revenue:30,slots:2;path=a:0,b:_0;path=a:2,b:_0;path=a:3,b:_0;path=a:4,b:_0",
            g,
        ),
    );
    catalog.insert(
        "622".into(),
        parse_tile(
            "622",
            "city=revenue:40,slots:2;path=a:0,b:_0;path=a:2,b:_0;path=a:3,b:_0;path=a:4,b:_0;label=Y",
            g,
        ),
    );
    catalog.insert("624".into(), parse_tile("624", "path=a:0,b:1;path=a:1,b:2", g));
    catalog.insert("625".into(), parse_tile("625", "path=a:0,b:1;path=a:2,b:3", g));
    catalog.insert("626".into(), parse_tile("626", "path=a:0,b:1;path=a:3,b:4", g));
    catalog.insert(
        "637".into(),
        parse_tile(
            "637",
            "city=revenue:50,loc:0.5;city=revenue:50,loc:2.5;city=revenue:50,loc:4.5;path=a:0,b:_0;path=a:_0,b:1;path=a:4,b:_2;path=a:_2,b:5;path=a:2,b:_1;path=a:_1,b:3;label=M",
            g,
        ),
    );
    catalog.insert(
        "X1".into(),
        parse_tile(
            "X1",
            "city=revenue:50;city=revenue:50;city=revenue:50;path=a:0,b:_0;path=a:_0,b:3;path=a:1,b:_1;path=a:_1,b:4;path=a:2,b:_2;path=a:_2,b:5;label=M",
            g,
        ),
    );
    catalog.insert(
        "X2".into(),
        parse_tile(
            "X2",
            "city=revenue:50;city=revenue:50;city=revenue:50;path=a:0,b:_0;path=a:_0,b:3;path=a:1,b:_1;path=a:_1,b:5;path=a:2,b:_2;path=a:_2,b:4;label=M",
            g,
        ),
    );
    catalog.insert(
        "X3".into(),
        parse_tile(
            "X3",
            "city=revenue:50;city=revenue:50;city=revenue:50;path=a:0,b:_0;path=a:_0,b:4;path=a:1,b:_1;path=a:_1,b:2;path=a:3,b:_2;path=a:_2,b:5;label=M",
            g,
        ),
    );
    catalog.insert(
        "X4".into(),
        parse_tile(
            "X4",
            "city=revenue:50;city=revenue:50;city=revenue:50;path=a:0,b:_0;path=a:_0,b:3;path=a:1,b:_1;path=a:_1,b:2;path=a:4,b:_2;path=a:_2,b:5;label=M",
            g,
        ),
    );

    // === Brown tiles (1867-only ids) ===
    let b = TileColor::Brown;
    catalog.insert(
        "122".into(),
        parse_tile(
            "122",
            "city=revenue:80,slots:2;city=revenue:80,slots:2;path=a:0,b:_0;path=a:_0,b:1;path=a:2,b:_1;path=a:_1,b:3;label=T",
            b,
        ),
    );
    catalog.insert(
        "611".into(),
        parse_tile(
            "611",
            "city=revenue:40,slots:2;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0;path=a:3,b:_0;path=a:4,b:_0",
            b,
        ),
    );
    catalog.insert(
        "623".into(),
        parse_tile(
            "623",
            "city=revenue:50,slots:2;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0;path=a:3,b:_0;path=a:4,b:_0;path=a:5,b:_0;label=Y",
            b,
        ),
    );
    catalog.insert(
        "801".into(),
        parse_tile(
            "801",
            "city=revenue:50,slots:2;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0;path=a:3,b:_0;label=Y",
            b,
        ),
    );
    catalog.insert(
        "911".into(),
        parse_tile(
            "911",
            "town=revenue:10;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0;path=a:3,b:_0;path=a:4,b:_0",
            b,
        ),
    );
    catalog.insert(
        "X5".into(),
        parse_tile(
            "X5",
            "city=revenue:70,slots:2;city=revenue:70;path=a:0,b:_1;path=a:1,b:_0;path=a:2,b:_0;path=a:3,b:_1;path=a:4,b:_0;path=a:5,b:_0;label=M",
            b,
        ),
    );
    catalog.insert(
        "X6".into(),
        parse_tile(
            "X6",
            "city=revenue:70,slots:2;city=revenue:70;path=a:0,b:_0;path=a:3,b:_0;path=a:4,b:_0;path=a:5,b:_0;path=a:1,b:_1;path=a:2,b:_1;label=M",
            b,
        ),
    );
    catalog.insert(
        "X7".into(),
        parse_tile(
            "X7",
            "city=revenue:70,slots:2;city=revenue:70;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_1;path=a:3,b:_0;path=a:4,b:_1;path=a:5,b:_0;label=M",
            b,
        ),
    );

    // === Gray tiles (1867-only ids) ===
    let gr = TileColor::Gray;
    catalog.insert(
        "124".into(),
        parse_tile(
            "124",
            "city=revenue:100,slots:4;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0;path=a:3,b:_0;label=T",
            gr,
        ),
    );
    // Laying 639 on Montreal triggers the CN national-reservation token
    // (game.rb:651-660 place_639_token) — hook lands with the CN work.
    catalog.insert(
        "639".into(),
        parse_tile(
            "639",
            "city=revenue:100,slots:4;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0;path=a:3,b:_0;path=a:4,b:_0;path=a:5,b:_0;label=M",
            gr,
        ),
    );
    catalog.insert(
        "X8".into(),
        parse_tile(
            "X8",
            "city=revenue:60,slots:3;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0;path=a:3,b:_0;path=a:4,b:_0;path=a:5,b:_0;label=O",
            gr,
        ),
    );

    Arc::new(catalog)
}

/// Parse a preprinted hex DSL string (for red/gray/yellow hexes on initial map).
/// Returns a TileDef with the appropriate color.
pub fn parse_preprinted_tile(hex_id: &str, code: &str, color: TileColor) -> TileDef {
    parse_tile(&format!("preprinted_{}", hex_id), code, color)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// The 1867 catalog parses completely and carries the load-bearing
    /// geometry: every id present, shared ids identical to 1830's, the
    /// custom Montreal/Ottawa/Toronto tiles with the right city counts,
    /// slots and labels.
    #[test]
    fn tile_catalog_1867_complete() {
        let catalog = tile_catalog_1867();
        assert_eq!(catalog.len(), 68);

        // Shared ids are clones of the 1830 definitions.
        let base = tile_catalog_1830();
        for id in ["3", "7", "14", "23", "39", "57", "63", "70"] {
            assert_eq!(catalog[id].paths.len(), base[id].paths.len(), "tile {id}");
            assert_eq!(catalog[id].color, base[id].color, "tile {id}");
        }

        // X1-X4: green Montreal triple cities (3 × 1-slot, label M).
        for id in ["X1", "X2", "X3", "X4"] {
            let t = &catalog[id];
            assert_eq!(t.color, TileColor::Green, "{id}");
            assert_eq!(t.cities.len(), 3, "{id}");
            assert!(t.cities.iter().all(|c| c.slots == 1 && c.revenue == 50), "{id}");
            assert_eq!(t.label.as_deref(), Some("M"), "{id}");
            assert_eq!(t.paths.len(), 6, "{id}");
        }
        // X5-X7: brown Montreal (2+1 slots, revenue 70).
        for id in ["X5", "X6", "X7"] {
            let t = &catalog[id];
            assert_eq!(t.color, TileColor::Brown, "{id}");
            assert_eq!(t.cities.len(), 2, "{id}");
            assert_eq!(t.cities[0].slots, 2, "{id}");
            assert_eq!(t.cities[1].slots, 1, "{id}");
        }
        // X8: gray Ottawa, 3 slots, label O.
        let x8 = &catalog["X8"];
        assert_eq!(x8.color, TileColor::Gray);
        assert_eq!(x8.cities[0].slots, 3);
        assert_eq!(x8.label.as_deref(), Some("O"));
        // 639: the gray Montreal tile whose lay claims the CN reservation.
        let t639 = &catalog["639"];
        assert_eq!(t639.cities[0].slots, 4);
        assert_eq!(t639.label.as_deref(), Some("M"));
        assert_eq!(t639.edges, vec![0, 1, 2, 3, 4, 5]);
        // 637: green Montreal triple city from the standard set.
        assert_eq!(catalog["637"].cities.len(), 3);

        // 1830-only ids must NOT leak in (1867 has no 1/2/55/56/69, no
        // B/NY/OO label tiles).
        for id in ["1", "2", "53", "54", "55", "56", "59", "61", "62", "64", "69"] {
            assert!(!catalog.contains_key(id), "1830-only tile {id} leaked into 1867");
        }
    }

    #[test]
    fn parse_simple_path_tile() {
        let t = parse_tile("7", "path=a:0,b:1", TileColor::Yellow);
        assert_eq!(t.paths.len(), 1);
        assert_eq!(t.paths[0].a, PathEndpoint::Edge(0));
        assert_eq!(t.paths[0].b, PathEndpoint::Edge(1));
        assert_eq!(t.edges, vec![0, 1]);
        assert!(t.cities.is_empty());
        assert!(t.towns.is_empty());
    }

    #[test]
    fn parse_city_tile() {
        let t = parse_tile(
            "57",
            "city=revenue:20;path=a:0,b:_0;path=a:_0,b:3",
            TileColor::Yellow,
        );
        assert_eq!(t.cities.len(), 1);
        assert_eq!(t.cities[0].revenue, 20);
        assert_eq!(t.cities[0].slots, 1);
        assert_eq!(t.paths.len(), 2);
        assert_eq!(t.paths[0].a, PathEndpoint::Edge(0));
        assert_eq!(t.paths[0].b, PathEndpoint::City(0));
        assert_eq!(t.edges, vec![0, 3]);
    }

    #[test]
    fn parse_double_town_tile() {
        let t = parse_tile(
            "1",
            "town=revenue:10;town=revenue:10;path=a:1,b:_0;path=a:_0,b:3;path=a:0,b:_1;path=a:_1,b:4",
            TileColor::Yellow,
        );
        assert_eq!(t.towns.len(), 2);
        assert_eq!(t.paths.len(), 4);
        // _0 should be Town(0), _1 should be Town(1)
        assert_eq!(t.paths[0].a, PathEndpoint::Edge(1));
        assert_eq!(t.paths[0].b, PathEndpoint::Town(0));
    }

    #[test]
    fn parse_labeled_tile() {
        let t = parse_tile(
            "53",
            "city=revenue:50;path=a:0,b:_0;path=a:2,b:_0;path=a:4,b:_0;label=B",
            TileColor::Green,
        );
        assert_eq!(t.label, Some("B".to_string()));
        assert_eq!(t.cities.len(), 1);
        assert_eq!(t.paths.len(), 3);
    }

    #[test]
    fn parse_offboard_tile() {
        let t = parse_tile(
            "F2",
            "offboard=revenue:yellow_40|brown_70;path=a:3,b:_0;path=a:4,b:_0;path=a:5,b:_0",
            TileColor::Red,
        );
        assert_eq!(t.offboards.len(), 1);
        assert_eq!(t.offboards[0].yellow_revenue, 40);
        assert_eq!(t.offboards[0].brown_revenue, 70);
        assert_eq!(t.paths.len(), 3);
        assert_eq!(t.paths[0].a, PathEndpoint::Edge(3));
        assert_eq!(t.paths[0].b, PathEndpoint::Offboard(0));
    }

    #[test]
    fn rotation_transforms_edges() {
        let t = parse_tile("7", "path=a:0,b:1", TileColor::Yellow);
        let r = t.rotated(2);
        assert_eq!(r.paths[0].a, PathEndpoint::Edge(2));
        assert_eq!(r.paths[0].b, PathEndpoint::Edge(3));
        assert_eq!(r.edges, vec![2, 3]);
    }

    #[test]
    fn rotation_preserves_nodes() {
        let t = parse_tile(
            "57",
            "city=revenue:20;path=a:0,b:_0;path=a:_0,b:3",
            TileColor::Yellow,
        );
        let r = t.rotated(1);
        assert_eq!(r.paths[0].a, PathEndpoint::Edge(1));
        assert_eq!(r.paths[0].b, PathEndpoint::City(0)); // city ref unchanged
        assert_eq!(r.paths[1].a, PathEndpoint::City(0));
        assert_eq!(r.paths[1].b, PathEndpoint::Edge(4));
    }

    #[test]
    fn catalog_has_all_tiles() {
        let catalog = tile_catalog_1830();
        // Yellow: 12, Green: 16 (14,15,16,18,19,20,23,24,25,26,27,28,29,53,54,59), Brown: 18
        assert_eq!(catalog.len(), 46);

        // Spot-check specific tiles
        assert!(catalog.contains_key("1"));
        assert!(catalog.contains_key("57"));
        assert!(catalog.contains_key("14"));
        assert!(catalog.contains_key("53"));
        assert!(catalog.contains_key("62"));
        assert!(catalog.contains_key("70"));
    }

    #[test]
    fn parse_double_city_ny() {
        let t = parse_tile(
            "54",
            "city=revenue:60,loc:0.5;city=revenue:60,loc:2.5;path=a:0,b:_0;path=a:_0,b:1;path=a:2,b:_1;path=a:_1,b:3;label=NY",
            TileColor::Green,
        );
        assert_eq!(t.cities.len(), 2);
        assert_eq!(t.cities[0].revenue, 60);
        assert_eq!(t.cities[1].revenue, 60);
        assert_eq!(t.label, Some("NY".to_string()));
        assert_eq!(t.paths.len(), 4);
        assert_eq!(t.paths[0].a, PathEndpoint::Edge(0));
        assert_eq!(t.paths[0].b, PathEndpoint::City(0));
        assert_eq!(t.paths[2].a, PathEndpoint::Edge(2));
        assert_eq!(t.paths[2].b, PathEndpoint::City(1));
    }

    #[test]
    fn path_superset_check() {
        // Tile 57 (yellow city: edges 0,3) should be subset of tile 14 (green city: edges 0,1,3,4)
        let t57 = parse_tile(
            "57",
            "city=revenue:20;path=a:0,b:_0;path=a:_0,b:3",
            TileColor::Yellow,
        );
        let t14 = parse_tile(
            "14",
            "city=revenue:30,slots:2;path=a:0,b:_0;path=a:1,b:_0;path=a:3,b:_0;path=a:4,b:_0",
            TileColor::Green,
        );
        assert!(t14.paths_are_superset_of(&t57));
    }

    #[test]
    fn parse_upgrade_part() {
        let t = parse_tile(
            "test",
            "city=revenue:0;upgrade=cost:80,terrain:water",
            TileColor::White,
        );
        assert_eq!(t.upgrades.len(), 1);
        assert_eq!(t.upgrades[0].cost, 80);
        assert_eq!(t.upgrades[0].terrain, "water");
    }
}
