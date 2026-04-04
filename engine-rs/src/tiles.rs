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

    /// Check if all paths of `old` exist in `self` (path subset for upgrade validation).
    pub fn paths_are_superset_of(&self, old: &TileDef) -> bool {
        old.paths.iter().all(|op| {
            self.paths
                .iter()
                .any(|np| (np.a == op.a && np.b == op.b) || (np.a == op.b && np.b == op.a))
        })
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
            for segment in attrs.split(',') {
                let (k, v) = parse_kv(segment);
                if k == "revenue" {
                    // Format: "yellow_V|brown_V"
                    for phase_rev in v.split('|') {
                        if let Some(val_str) = phase_rev.strip_prefix("yellow_") {
                            yellow_revenue = val_str.parse().unwrap_or(0);
                        } else if let Some(val_str) = phase_rev.strip_prefix("brown_") {
                            brown_revenue = val_str.parse().unwrap_or(0);
                        }
                    }
                }
            }
            offboards.push(OffboardDef {
                yellow_revenue,
                brown_revenue,
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
