//! `RlSpec`: what is title-specific about the token encoding -- the
//! entity lists and their order, the hex order and geometry, the tile table
//! and the normalizers -- derived once per registered title from its data
//! (`GameTitle`), memoized like `encoder::spec_for`.

use std::collections::HashMap;
use std::sync::OnceLock;

use super::schema::{F_TILE, TILE_FEATURES};
use super::*;
use crate::tiles::{PathDef, PathEndpoint, TileColor, TileDef};
use crate::title::{AbilityDef, CorpType, GameTitle, HexLayout, HexType};

pub struct CorpSpec {
    pub sym: &'static str,
    pub ent_type: i32,
    pub corp_type: CorpType,
    /// Home hex (spec hex index) and city from the title data; -1 when the
    /// corporation chooses its home (1867).
    pub home_hex: i32,
    pub home_city: usize,
    pub max_token_price: i32,
    /// 1822 family: the concession company that founds this major.
    pub concession: Option<String>,
}

pub struct CompanySpec {
    pub sym: &'static str,
    pub ent_type: i32,
    pub value: i32,
    pub revenue: i32,
    pub auctionable: bool,
    /// blocks_hexes, tile_lay, teleport, exchange, no_buy, shares, hex_bonus, close.
    pub abilities: [bool; 8],
    /// The first hex its abilities name (spec hex index), -1 none.
    pub ability_hex: i32,
    /// 1822 family: the corporation a proxy / concession stands for.
    pub link: Option<String>,
    /// Route bonuses: (spec hex index, amount) (1867's bridges and tunnel).
    pub hex_bonus: Vec<(usize, i32)>,
}

pub struct TrainSpec {
    pub name: &'static str,
    pub distance: u32,
    pub price: i32,
    pub count: u32,
    pub multiplier: u32,
    pub rusts_on: i32,
    pub obsolete_on: i32,
    pub available_on: i32,
    pub max_discount: i32,
    pub n_events: usize,
}

pub struct PhaseSpec {
    pub name: &'static str,
    /// The train type (index) whose purchase starts the phase, -1 none.
    pub on_train: i32,
    pub train_limit: u8,
    pub minor_train_limit: Option<u8>,
    pub operating_rounds: u8,
    /// yellow, green, brown, gray tiles available.
    pub colors: [bool; 4],
    /// `PHASE_STATUSES` indices; -1 for a status not in the vocabulary.
    pub statuses: Vec<i32>,
}

pub struct RlSpec {
    pub title: &'static str,
    pub title_id: i32,
    pub layout: HexLayout,
    pub is_g1822_family: bool,

    // -- map ---------------------------------------------------------------
    /// Hex coordinates in map order (`hex_definitions()` = `BaseGame.hexes`
    /// = the descriptor vocabulary's `hex`).
    pub hex_coords: Vec<&'static str>,
    pub hex_index: HashMap<&'static str, usize>,
    /// Normalized (x, y) of each hex's centre: x across, y down the map.
    pub hex_pos: Vec<[f32; 2]>,
    /// Axial coordinates (q, r): hex distance = (|dq| + |dr| + |dq + dr|) / 2.
    pub hex_axial: Vec<[i32; 2]>,
    /// Neighbor (spec hex index) across each edge 0-5, -1 off the map.
    pub hex_neighbors: Vec<[i32; 6]>,
    /// Impassable edges.
    pub hex_walls: Vec<[bool; 6]>,

    // -- entities ------------------------------------------------------------
    pub corps: Vec<CorpSpec>,
    pub companies: Vec<CompanySpec>,
    pub trains: Vec<TrainSpec>,
    pub phases: Vec<PhaseSpec>,
    pub corp_index: HashMap<&'static str, usize>,
    pub company_index: HashMap<&'static str, usize>,
    pub train_index: HashMap<&'static str, usize>,
    pub phase_index: HashMap<&'static str, usize>,
    /// The national corporation (1867's CN), corp index.
    pub national: Option<usize>,

    // -- tiles ---------------------------------------------------------------
    /// Tiles in the title's manifest order (`tile_counts()` = the descriptor
    /// vocabulary's `tile`); `hex_ids[1]` is index + 1 (0 = no laid tile,
    /// len + 1 = a tile not in this list).
    pub tile_ids: Vec<&'static str>,
    pub tile_index: HashMap<&'static str, usize>,
    /// `[tile, TILE_FEATURES]`.
    pub tile_feat: Vec<f32>,

    // -- normalizers -----------------------------------------------------------
    pub bank_cash: f32,
    pub max_market_price: f32,
    pub market_rows: f32,
    pub market_cols: f32,
    pub max_stop_revenue: f32,
    pub max_lay_cost: f32,
    pub max_train_price: f32,
    pub max_company_value: f32,
    pub max_company_revenue: f32,
    pub max_token_price: f32,
    pub num_loans: f32,
    pub loan_value: f32,
    pub loan_interest: f32,
    pub max_loans_major: f32,
    pub max_loans_minor: f32,
    pub market_pool_limit: f32,

    // -- entity layout ---------------------------------------------------------
    /// Entity index of the first corporation / company / train type / phase;
    /// the global token is 0 and seat k is 1 + k.
    pub e_corp0: usize,
    pub e_company0: usize,
    pub e_train0: usize,
    pub e_phase0: usize,
    pub num_entities: usize,
}

impl RlSpec {
    pub fn num_hexes(&self) -> usize {
        self.hex_coords.len()
    }
    pub fn num_tiles(&self) -> usize {
        self.tile_ids.len()
    }
    pub fn seat_entity(seat: usize) -> usize {
        1 + seat
    }

    /// The entity token a descriptor's entity field names (`docs/rl_descriptors.md`
    /// §2, §4): `kind` an `rl::EntKind` id, `index` into that kind's title
    /// vocabulary. Corporations, companies and train types sit at a fixed base
    /// (`base + index`); a player (seating-order index) is the seat token after
    /// the encoding's rotation (`meta[5]`). -1 for none / out of range.
    pub fn descriptor_entity(&self, kind: i32, index: i32, rotation: usize, num_seats: usize) -> i32 {
        use crate::rl::descriptor::{EntKind, ENT_KINDS};
        if index < 0 {
            return PTR_NONE;
        }
        let i = index as usize;
        match ENT_KINDS.get(kind.max(0) as usize).filter(|_| kind >= 0) {
            Some(EntKind::Player) if i < num_seats => (1 + (i + num_seats - rotation % num_seats) % num_seats) as i32,
            Some(EntKind::Corporation) if i < self.corps.len() => (self.e_corp0 + i) as i32,
            Some(EntKind::Company) if i < self.companies.len() => (self.e_company0 + i) as i32,
            Some(EntKind::Train) if i < self.trains.len() => (self.e_train0 + i) as i32,
            _ => PTR_NONE,
        }
    }

    /// Entity names, in entity order: "global", "seat0".., "corp:SYM",
    /// "company:SYM", "train:NAME", "phase:NAME".
    pub fn entity_names(&self) -> Vec<String> {
        let mut out = vec!["global".to_string()];
        out.extend((0..MAX_SEATS).map(|s| format!("seat{s}")));
        out.extend(self.corps.iter().map(|c| format!("corp:{}", c.sym)));
        out.extend(self.companies.iter().map(|c| format!("company:{}", c.sym)));
        out.extend(self.trains.iter().map(|t| format!("train:{}", t.name)));
        out.extend(self.phases.iter().map(|p| format!("phase:{}", p.name)));
        out
    }

    /// The static entity types (seats as players: a position masks the
    /// absent ones).
    pub fn entity_types(&self) -> Vec<i32> {
        let mut out = vec![ENT_GLOBAL];
        out.extend(std::iter::repeat(ENT_PLAYER).take(MAX_SEATS));
        out.extend(self.corps.iter().map(|c| c.ent_type));
        out.extend(self.companies.iter().map(|c| c.ent_type));
        out.extend(std::iter::repeat(ENT_TRAIN).take(self.trains.len()));
        out.extend(std::iter::repeat(ENT_PHASE).take(self.phases.len()));
        out
    }

    fn build(title: &'static dyn GameTitle) -> RlSpec {
        let name = title.name();
        let layout = title.hex_layout();
        let is_g1822_family = title.is_g1822_family();
        let family = title.g1822_family();

        // Entity, hex and tile orders are the descriptor vocabularies'
        // (`rl::Vocab`, docs/rl_descriptors.md §4), so a descriptor's
        // ent0 / ent1 / hex / tile index addresses a token directly.
        let vocab = crate::rl::Vocab::for_title(title);

        // -- map -----------------------------------------------------------
        let hex_defs = title.hex_definitions();
        let hex_coords: Vec<&'static str> = vocab.hexes.clone();
        let hex_index: HashMap<&'static str, usize> =
            hex_coords.iter().enumerate().map(|(i, c)| (*c, i)).collect();
        let parsed: Vec<(i32, i32)> = hex_coords.iter().map(|c| crate::title::parse_coord(c)).collect();
        // (x across, y down) in doubled coordinates.
        let xy: Vec<(i32, i32)> = parsed
            .iter()
            .map(|&(letter, number)| match layout {
                HexLayout::Pointy => (number, letter),
                HexLayout::Flat => (letter, number),
            })
            .collect();
        let (min_x, max_x) = (xy.iter().map(|p| p.0).min().unwrap_or(0), xy.iter().map(|p| p.0).max().unwrap_or(0));
        let (min_y, max_y) = (xy.iter().map(|p| p.1).min().unwrap_or(0), xy.iter().map(|p| p.1).max().unwrap_or(0));
        let norm = |v: i32, lo: i32, hi: i32| if hi > lo { (v - lo) as f32 / (hi - lo) as f32 } else { 0.0 };
        let hex_pos = xy.iter().map(|&(x, y)| [norm(x, min_x, max_x), norm(y, min_y, max_y)]).collect();
        let parity = xy.first().map_or(0, |&(x, y)| (x - y).rem_euclid(2));
        let hex_axial = xy
            .iter()
            .map(|&(x, y)| match layout {
                // doubled width: x + y has one parity on a map
                HexLayout::Pointy => [(x - y - parity).div_euclid(2), y],
                // doubled height
                HexLayout::Flat => [x, (y - x - parity).div_euclid(2)],
            })
            .collect();
        let adjacency = crate::title::compute_adjacency(&hex_coords, layout);
        let hex_neighbors = hex_coords
            .iter()
            .map(|c| {
                let mut n = [PTR_NONE; 6];
                if let Some(m) = adjacency.get(*c) {
                    for (&dir, other) in m {
                        if let Some(&j) = hex_index.get(other.as_str()) {
                            n[dir as usize % 6] = j as i32;
                        }
                    }
                }
                n
            })
            .collect();
        let walls_map = crate::title::title_walls(title);
        let hex_walls = hex_coords
            .iter()
            .map(|c| {
                let mut w = [false; 6];
                for &e in walls_map.get(*c).map(|v| v.as_slice()).unwrap_or(&[]) {
                    w[e as usize % 6] = true;
                }
                w
            })
            .collect();

        // -- phases and trains -------------------------------------------------
        let phase_defs = title.phases();
        let mut train_defs = title.trains();
        // one type per name, first definition kept (the vocabulary's rule)
        let mut seen_names: Vec<&'static str> = Vec::new();
        train_defs.retain(|t| {
            let new = !seen_names.contains(&t.name);
            seen_names.push(t.name);
            new
        });
        let phase_index: HashMap<&'static str, usize> =
            phase_defs.iter().enumerate().map(|(i, p)| (p.name, i)).collect();
        let train_index: HashMap<&'static str, usize> =
            train_defs.iter().enumerate().map(|(i, t)| (t.name, i)).collect();
        let phase_of = |n: Option<&str>| n.and_then(|n| phase_index.get(n)).map_or(PTR_NONE, |&i| i as i32);
        let trains: Vec<TrainSpec> = train_defs
            .iter()
            .map(|t| TrainSpec {
                name: t.name,
                distance: t.distance,
                price: t.price,
                count: t.count,
                multiplier: t.multiplier.max(1),
                rusts_on: phase_of(t.rusts_on),
                obsolete_on: phase_of(t.obsolete_on),
                available_on: phase_of(t.available_on),
                max_discount: t.discount.iter().map(|(_, d)| *d).max().unwrap_or(0),
                n_events: t.events.len(),
            })
            .collect();
        let phases: Vec<PhaseSpec> = phase_defs
            .iter()
            .map(|p| {
                let mut colors = [false; 4];
                for c in p.tiles {
                    match *c {
                        "yellow" => colors[0] = true,
                        "green" => colors[1] = true,
                        "brown" => colors[2] = true,
                        "gray" => colors[3] = true,
                        _ => {}
                    }
                }
                PhaseSpec {
                    name: p.name,
                    on_train: p.on.and_then(|t| train_index.get(t)).map_or(PTR_NONE, |&i| i as i32),
                    train_limit: p.train_limit,
                    minor_train_limit: p.minor_train_limit,
                    operating_rounds: p.operating_rounds,
                    colors,
                    statuses: p
                        .status
                        .iter()
                        .map(|s| vocab_index(PHASE_STATUSES, s).map_or(-1, |i| i as i32))
                        .collect(),
                }
            })
            .collect();

        // -- corporations ------------------------------------------------------
        let corp_defs = title.corporations();
        let corp_index: HashMap<&'static str, usize> =
            corp_defs.iter().enumerate().map(|(i, c)| (c.sym, i)).collect();
        let concessions: &[(&str, &str)] = family.map_or(&[], |f| f.concessions);
        let corps: Vec<CorpSpec> = corp_defs
            .iter()
            .map(|c| CorpSpec {
                sym: c.sym,
                ent_type: match c.corp_type {
                    CorpType::Major => ENT_MAJOR,
                    CorpType::Minor => ENT_MINOR,
                    CorpType::National => ENT_NATIONAL,
                },
                corp_type: c.corp_type,
                home_hex: hex_index.get(c.home_hex).map_or(PTR_NONE, |&i| i as i32),
                home_city: c.home_city_index as usize,
                max_token_price: c.token_prices.iter().copied().max().unwrap_or(0),
                concession: concessions.iter().find(|(_, m)| *m == c.sym).map(|(co, _)| co.to_string()),
            })
            .collect();
        let national = corps.iter().position(|c| c.corp_type == CorpType::National);

        // -- companies ---------------------------------------------------------
        let company_defs = title.companies();
        let company_index: HashMap<&'static str, usize> =
            company_defs.iter().enumerate().map(|(i, c)| (c.sym, i)).collect();
        let family_abilities = family.map(|f| (f.abilities)());
        let companies: Vec<CompanySpec> = company_defs
            .iter()
            .map(|c| {
                let mut ab = [false; 8];
                let mut ability_hex = PTR_NONE;
                let mut hex_bonus = Vec::new();
                let first_hex = |hexes: &[&str], ability_hex: &mut i32| {
                    if *ability_hex < 0 {
                        if let Some(&i) = hexes.first().and_then(|h| hex_index.get(h)) {
                            *ability_hex = i as i32;
                        }
                    }
                };
                for a in c.abilities {
                    match a {
                        AbilityDef::BlocksHexes { hexes, .. } => {
                            ab[0] = true;
                            first_hex(hexes, &mut ability_hex);
                        }
                        AbilityDef::TileLay { hexes, .. } => {
                            ab[1] = true;
                            first_hex(hexes, &mut ability_hex);
                        }
                        AbilityDef::Teleport { hexes, .. } => {
                            ab[2] = true;
                            first_hex(hexes, &mut ability_hex);
                        }
                        AbilityDef::Exchange { .. } => ab[3] = true,
                        AbilityDef::NoBuy => ab[4] = true,
                        AbilityDef::Shares { .. } => ab[5] = true,
                        AbilityDef::HexBonus { hexes, amount, .. } => {
                            ab[6] = true;
                            first_hex(hexes, &mut ability_hex);
                            for h in hexes.iter() {
                                if let Some(&i) = hex_index.get(h) {
                                    hex_bonus.push((i, *amount));
                                }
                            }
                        }
                        AbilityDef::Close { .. } => ab[7] = true,
                    }
                }
                if let Some(abs) = family_abilities.as_ref().and_then(|m| m.get(c.sym)) {
                    for a in abs {
                        match a.kind {
                            crate::g1822::track::AbKind::TileLay => ab[1] = true,
                            crate::g1822::track::AbKind::Teleport => ab[2] = true,
                            crate::g1822::track::AbKind::TileDiscount => {}
                        }
                        if ability_hex < 0 {
                            if let Some(&i) = a.hexes.first().and_then(|h| hex_index.get(h.as_str())) {
                                ability_hex = i as i32;
                            }
                        }
                    }
                }
                let (ent_type, link) = if is_g1822_family {
                    let first = c.sym.chars().next().unwrap_or(' ');
                    let rest = &c.sym[first.len_utf8()..];
                    match first {
                        // M<n> stands for minor <n> (`minor_of_company`)
                        'M' if corp_index.contains_key(rest) => (ENT_MINOR_PROXY, Some(rest.to_string())),
                        'C' if rest.chars().all(|ch| ch.is_ascii_digit()) && !rest.is_empty() => (
                            ENT_CONCESSION,
                            concessions.iter().find(|(co, _)| *co == c.sym).map(|(_, m)| m.to_string()),
                        ),
                        _ => (ENT_PRIVATE, None),
                    }
                } else {
                    (ENT_PRIVATE, None)
                };
                CompanySpec {
                    sym: c.sym,
                    ent_type,
                    value: c.value,
                    revenue: c.revenue,
                    auctionable: c.auctionable,
                    abilities: ab,
                    ability_hex,
                    link,
                    hex_bonus,
                }
            })
            .collect();

        // -- tiles -------------------------------------------------------------
        let catalog = title.tile_catalog();
        let counts: HashMap<&str, u32> = title.tile_counts().into_iter().collect();
        let tile_ids: Vec<&'static str> = vocab.tiles.clone();
        let tile_index: HashMap<&'static str, usize> = tile_ids.iter().enumerate().map(|(i, t)| (*t, i)).collect();

        // -- normalizers: the title's maxima ------------------------------------
        let grid = title.market_grid();
        let max_market_price = grid.iter().flatten().flatten().map(|c| c.price).max().unwrap_or(1).max(1) as f32;
        let market_rows = grid.len().max(1) as f32;
        let market_cols = grid.iter().map(|r| r.len()).max().unwrap_or(1).max(1) as f32;
        let mut max_stop = 1;
        let mut max_cost = 1;
        let note_tile = |t: &TileDef, max_stop: &mut i32, max_cost: &mut i32| {
            for c in &t.cities {
                *max_stop = (*max_stop).max(c.revenue);
            }
            for c in &t.towns {
                *max_stop = (*max_stop).max(c.revenue);
            }
            for o in &t.offboards {
                let tiers = [Some(o.yellow_revenue), Some(o.brown_revenue), o.green_revenue, o.gray_revenue];
                for v in tiers.into_iter().flatten() {
                    *max_stop = (*max_stop).max(v);
                }
            }
            for u in &t.upgrades {
                *max_cost = (*max_cost).max(u.cost);
            }
        };
        for t in catalog.values() {
            note_tile(t, &mut max_stop, &mut max_cost);
        }
        for d in &hex_defs {
            max_cost = max_cost.max(d.terrain_cost);
            for b in d.borders {
                max_cost = max_cost.max(b.cost.unwrap_or(0));
            }
            match d.hex_type {
                HexType::City { revenue, .. } | HexType::Town { revenue } | HexType::DoubleCity { revenue } => {
                    max_stop = max_stop.max(revenue)
                }
                HexType::Offboard { yellow_revenue, brown_revenue } => {
                    max_stop = max_stop.max(yellow_revenue).max(brown_revenue)
                }
                _ => {}
            }
            if let Some((dsl, color)) = title.preprinted_hex_dsl(d.coord) {
                let t = crate::tiles::parse_preprinted_tile(d.coord, dsl, color_of(color));
                note_tile(&t, &mut max_stop, &mut max_cost);
            }
        }
        if let Some(f) = family {
            // 1822 water borders are kept in the family's hex DSL table.
            for (coord, dsl, color) in f.hex_dsl_all {
                let t = crate::tiles::parse_preprinted_tile(coord, dsl, color_of(color));
                note_tile(&t, &mut max_stop, &mut max_cost);
                for part in dsl.split(';') {
                    if let Some(rest) = part.trim().strip_prefix("border=") {
                        for kv in rest.split(',') {
                            if let Some(("cost", v)) = kv.split_once(':') {
                                max_cost = max_cost.max(v.trim().parse().unwrap_or(0));
                            }
                        }
                    }
                }
            }
        }
        let max_stop_revenue = max_stop as f32;
        let max_lay_cost = max_cost as f32;

        let mut tile_feat = vec![0.0f32; tile_ids.len() * F_TILE];
        for (i, id) in tile_ids.iter().enumerate() {
            if let Some(def) = catalog.get(*id) {
                let supply = match counts.get(id).copied().unwrap_or(0) {
                    u32::MAX => 1.0,
                    n => (n as f32 / 10.0).min(1.0),
                };
                write_tile_features(
                    def,
                    max_stop_revenue,
                    max_lay_cost,
                    supply,
                    &mut tile_feat[i * F_TILE..(i + 1) * F_TILE],
                );
            }
        }

        let e_corp0 = 1 + MAX_SEATS;
        let e_company0 = e_corp0 + corps.len();
        let e_train0 = e_company0 + companies.len();
        let e_phase0 = e_train0 + trains.len();
        let num_entities = e_phase0 + phases.len();

        RlSpec {
            title: name,
            title_id: title_id(name),
            layout,
            is_g1822_family,
            hex_coords,
            hex_index,
            hex_pos,
            hex_axial,
            hex_neighbors,
            hex_walls,
            corps,
            trains,
            phases,
            corp_index,
            company_index,
            train_index,
            phase_index,
            national,
            companies,
            tile_ids,
            tile_index,
            tile_feat,
            bank_cash: title.bank_cash().max(1) as f32,
            max_market_price,
            market_rows,
            market_cols,
            max_stop_revenue,
            max_lay_cost,
            max_train_price: train_defs.iter().map(|t| t.price).max().unwrap_or(1).max(1) as f32,
            max_company_value: company_defs.iter().map(|c| c.value).max().unwrap_or(1).max(1) as f32,
            max_company_revenue: company_defs.iter().map(|c| c.revenue).max().unwrap_or(1).max(1) as f32,
            max_token_price: corp_defs
                .iter()
                .flat_map(|c| c.token_prices.iter().copied())
                .max()
                .unwrap_or(1)
                .max(1) as f32,
            num_loans: title.num_loans() as f32,
            loan_value: title.loan_value() as f32,
            loan_interest: title.loan_interest_rate() as f32,
            max_loans_major: title.max_loans(CorpType::Major) as f32,
            max_loans_minor: title.max_loans(CorpType::Minor) as f32,
            market_pool_limit: title.market_pool_limit() as f32,
            e_corp0,
            e_company0,
            e_train0,
            e_phase0,
            num_entities,
        }
    }
}

pub(crate) fn color_of(name: &str) -> TileColor {
    match name {
        "yellow" => TileColor::Yellow,
        "green" => TileColor::Green,
        "brown" => TileColor::Brown,
        "gray" => TileColor::Gray,
        "red" => TileColor::Red,
        _ => TileColor::White,
    }
}

pub(crate) fn color_slot(c: TileColor) -> usize {
    match c {
        TileColor::White => 0,
        TileColor::Yellow => 1,
        TileColor::Green => 2,
        TileColor::Brown => 3,
        TileColor::Gray => 4,
        TileColor::Red => 5,
    }
}

/// Index of the unordered edge pair (i < j) among the 15 pairs, in the order
/// (0,1) (0,2) .. (0,5) (1,2) .. (4,5).
#[inline]
pub(crate) fn pair_index(i: usize, j: usize) -> usize {
    let (i, j) = if i < j { (i, j) } else { (j, i) };
    i * 5 - i * (i.saturating_sub(1)) / 2 + (j - i - 1)
}

/// The 49 track features of a tile's (rotated) paths: 15 direct edge-edge
/// pairs, 15 edge-edge pairs joined through a stop or junction, per edge
/// whether it reaches a city / town / offboard, and a junction flag.
pub(crate) fn write_track_features(paths: &[PathDef], out: &mut [f32]) {
    debug_assert_eq!(out.len(), 49);
    // node key -> edges touching it (up to 8 nodes: cities, towns, offboards, junction)
    let mut node_edges: [(u8, u8, u8); 16] = [(0, 0, 0); 16]; // (kind, index, edge bitmask)
    let mut n_nodes = 0usize;
    let mut junction = false;
    let note = |kind: u8, index: u8, edge: u8, node_edges: &mut [(u8, u8, u8); 16], n_nodes: &mut usize| {
        for k in 0..*n_nodes {
            if node_edges[k].0 == kind && node_edges[k].1 == index {
                node_edges[k].2 |= 1 << edge;
                return;
            }
        }
        if *n_nodes < node_edges.len() {
            node_edges[*n_nodes] = (kind, index, 1 << edge);
            *n_nodes += 1;
        }
    };
    for p in paths {
        let (a, b) = (&p.a, &p.b);
        match (a, b) {
            (PathEndpoint::Edge(x), PathEndpoint::Edge(y)) => {
                let (x, y) = (*x as usize % 6, *y as usize % 6);
                if x != y {
                    out[pair_index(x, y)] = 1.0;
                }
            }
            (PathEndpoint::Edge(e), node) | (node, PathEndpoint::Edge(e)) => {
                let e = *e % 6;
                let (kind, index) = match node {
                    PathEndpoint::City(i) => (0u8, *i as u8),
                    PathEndpoint::Town(i) => (1u8, *i as u8),
                    PathEndpoint::Offboard(i) => (2u8, *i as u8),
                    PathEndpoint::Junction => {
                        junction = true;
                        (3u8, 0u8)
                    }
                    PathEndpoint::Edge(_) => unreachable!(),
                };
                if kind < 3 {
                    out[30 + e as usize * 3 + kind as usize] = 1.0;
                }
                note(kind, index, e, &mut node_edges, &mut n_nodes);
            }
            (PathEndpoint::Junction, _) | (_, PathEndpoint::Junction) => junction = true,
            _ => {}
        }
    }
    for &(_, _, mask) in &node_edges[..n_nodes] {
        for i in 0..6 {
            if mask & (1 << i) == 0 {
                continue;
            }
            for j in (i + 1)..6 {
                if mask & (1 << j) != 0 {
                    out[15 + pair_index(i, j)] = 1.0;
                }
            }
        }
    }
    out[48] = if junction { 1.0 } else { 0.0 };
}

fn write_tile_features(t: &TileDef, max_stop: f32, max_cost: f32, supply: f32, out: &mut [f32]) {
    debug_assert_eq!(out.len(), TILE_FEATURES.len());
    out[color_slot(t.color)] = 1.0;
    out[6] = if t.label.is_some() { 1.0 } else { 0.0 };
    out[7] = t.cities.len() as f32 / 6.0;
    out[8] = t.cities.iter().map(|c| c.slots as f32).sum::<f32>() / 6.0;
    out[9] = t.towns.len() as f32 / 2.0;
    out[10] = if t.offboards.is_empty() { 0.0 } else { 1.0 };
    out[11] = t.cities.iter().map(|c| c.revenue).max().unwrap_or(0) as f32 / max_stop;
    out[12] = t.towns.iter().map(|c| c.revenue).max().unwrap_or(0) as f32 / max_stop;
    out[13] = t.upgrades.iter().map(|u| u.cost).max().unwrap_or(0) as f32 / max_cost;
    write_track_features(&t.paths, &mut out[14..63]);
    out[63] = supply;
}

/// The token-encoder spec of a registered title (memoized; panics on an
/// unknown title string -- engine-internal, as `title::resolve`).
pub fn rl_spec_for(title_name: &str) -> &'static RlSpec {
    static SPECS: OnceLock<HashMap<&'static str, RlSpec>> = OnceLock::new();
    SPECS
        .get_or_init(|| crate::title::all_titles().iter().map(|t| (t.name(), RlSpec::build(*t))).collect())
        .get(title_name)
        .unwrap_or_else(|| panic!("no token-encoder spec for title {title_name}"))
}
