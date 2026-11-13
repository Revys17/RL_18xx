//! Cargo tests of the token encoder: schema and vocabulary coverage per
//! title, geometry, and invariants over random walks of 1830 (2-6 players),
//! 1867 (3-6) and the 1822 family (their player ranges up to 6). The
//! throughput comparison with the 1830 encoder is `bench_*` (ignored; run
//! with `--release --ignored --nocapture`).

use std::collections::{HashMap, HashSet};

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use super::schema::*;
use super::spec::*;
use super::*;
use crate::game::BaseGame;
use crate::title::CrossPlayerPolicy;

// ---------------------------------------------------------------------------
// games and walks
// ---------------------------------------------------------------------------

fn new_game(title: &str, players: usize, seed: u64) -> BaseGame {
    // 1822-family corporations are named "1".."30": players need ids above.
    let ids: Vec<u32> = (0..players as u32).map(|i| 1001 + i).collect();
    let names: HashMap<u32, String> = ids.iter().map(|&i| (i, format!("Player {}", i - 1000))).collect();
    BaseGame::build_seeded(title, ids, names, Vec::new(), Some(seed))
}

fn player_range(title: &str) -> std::ops::RangeInclusive<usize> {
    match title {
        "1830" => 2..=6,
        "1867" => 3..=6,
        t => {
            // every title targets 3-6 players
            let (lo, hi) = crate::title::resolve(t).g1822_family().map_or((3, 6), |f| f.player_range);
            (lo as usize).max(3)..=(hi as usize).min(MAX_SEATS)
        }
    }
}

/// One random legal move applied; false when the game is over or stuck.
fn step(g: &mut BaseGame, rng: &mut StdRng) -> bool {
    if g.finished {
        return false;
    }
    let pick = |n: usize, rng: &mut StdRng| rng.gen_range(0..n);
    match g.title.as_str() {
        "1830" => {
            let choices = g.get_factored_choices_impl();
            let applicable: Vec<_> = choices
                .iter()
                .filter(|c| !(c.action_type == "PlaceToken" && c.params.get("slot").map_or(false, |s| s.is_null())))
                .collect();
            if applicable.is_empty() {
                return false;
            }
            // non-pass moves favoured so walks reach the late game
            let non_pass: Vec<_> = applicable.iter().filter(|c| c.action_type != "Pass").collect();
            let la = if !non_pass.is_empty() && rng.gen::<f64>() < 0.75 {
                *non_pass[pick(non_pass.len(), rng)]
            } else {
                applicable[pick(applicable.len(), rng)]
            };
            let Some(idx) = crate::action_index::legal_action_to_index(la) else { return false };
            let price = la.price_range.map(|(lo, hi)| rng.gen_range(lo..=hi));
            let company_offset = crate::action_index::layout().action_offsets["CompanyBuyShares"];
            match g.build_action(la, idx, idx >= company_offset, price) {
                Ok(a) => g.process_action_internal(&a).is_ok(),
                Err(_) => false,
            }
        }
        "1867" => {
            let moves = g.g1867_legal_moves_with(CrossPlayerPolicy::RESTRICTED);
            if moves.is_empty() {
                return false;
            }
            let m = crate::g1867::fuzz::pick(&moves, rng);
            g.g1867_apply_move(&m).is_ok()
        }
        _ => {
            let moves = g.g1822_legal_moves_with(CrossPlayerPolicy::RESTRICTED);
            if moves.is_empty() {
                return false;
            }
            let non_pass: Vec<_> = moves.iter().filter(|m| m.action.action_type() != "pass").collect();
            let m = if !non_pass.is_empty() && rng.gen::<f64>() < 0.75 {
                non_pass[pick(non_pass.len(), rng)].clone()
            } else {
                moves[pick(moves.len(), rng)].clone()
            };
            let m = match &m.price {
                Some(r) if r.hi > r.lo => {
                    let k = (r.hi - r.lo) / r.step.max(1);
                    m.with_price(r.lo + rng.gen_range(0..=k.min(40)) * r.step)
                }
                _ => m,
            };
            g.g1822_apply_move(&m).is_ok()
        }
    }
}

// ---------------------------------------------------------------------------
// invariants
// ---------------------------------------------------------------------------

#[derive(Default, Debug)]
struct Stats {
    positions: usize,
    finished_games: usize,
    relations: usize,
    max_relations: usize,
    step_kinds: HashSet<&'static str>,
    round_kinds: HashSet<&'static str>,
    tokens_seen: usize,
    reservations_seen: usize,
    /// 1867: step kind disagreed with every generated move's step.
    step_mismatch: usize,
    step_checked: usize,
    mismatch_kinds: HashMap<String, usize>,
}

fn check(g: &BaseGame, enc: &TokenEncoding, stats: &mut Stats) {
    let spec = rl_spec_for(&g.title);
    let (h, e) = (spec.num_hexes(), spec.num_entities);
    let np = g.players.len();
    let ctx = format!("{} {}p move {}", g.title, np, g.action_log.len());
    assert_eq!(enc.num_hexes, h);
    assert_eq!(enc.num_entities, e);
    assert_eq!(enc.hex_feat.len(), h * F_HEX);
    assert_eq!(enc.hex_ptr.len(), h * S_MAX);
    assert_eq!(enc.hex_res.len(), h * S_MAX);
    assert_eq!(enc.hex_slot_city.len(), h * S_MAX);
    assert_eq!(enc.hex_ids.len(), h * 2);
    assert_eq!(enc.ent_type.len(), e);
    assert_eq!(enc.ent_feat.len(), e * F_ENT);
    assert_eq!(enc.ent_ptr.len(), e * P_ENT);
    assert_eq!(enc.ent_hex.len(), e * P_HEX);
    assert_eq!(enc.rel.len(), enc.rel_val.len() * 3);
    assert_eq!(enc.truncated_slots, 0, "{ctx}: tokens beyond S_MAX");

    for (k, v) in enc.hex_feat.iter().chain(&enc.ent_feat).chain(&enc.rel_val).enumerate() {
        assert!(v.is_finite() && v.abs() <= FEATURE_CLIP, "{ctx}: feature {k} = {v}");
    }

    // entity types: static, seats beyond the player count padded
    let types = spec.entity_types();
    for j in 0..e {
        let expect = if (1..=MAX_SEATS).contains(&j) && j - 1 >= np { ENT_PAD } else { types[j] };
        assert_eq!(enc.ent_type[j], expect, "{ctx}: entity {j} type");
    }
    // padded tokens are all zero / none
    for j in 0..e {
        if enc.ent_type[j] == ENT_PAD {
            assert!(enc.ent_feat[j * F_ENT..(j + 1) * F_ENT].iter().all(|&v| v == 0.0));
            assert!(enc.ent_ptr[j * P_ENT..(j + 1) * P_ENT].iter().all(|&v| v == PTR_NONE));
        }
        // features beyond the type's schema stay zero
        let n = entity_features(enc.ent_type[j]).len();
        assert!(enc.ent_feat[j * F_ENT + n..(j + 1) * F_ENT].iter().all(|&v| v == 0.0), "{ctx}: entity {j} tail");
    }
    let is_corp = |x: i32| x >= 0 && (x as usize) < e && matches!(enc.ent_type[x as usize], ENT_MAJOR | ENT_MINOR | ENT_NATIONAL);
    let is_seat = |x: i32| x >= 1 && (x as usize) <= MAX_SEATS && enc.ent_type[x as usize] == ENT_PLAYER;
    let valid = |x: i32| x >= 0 && (x as usize) < e && enc.ent_type[x as usize] != ENT_PAD;

    // hex pointers
    for i in 0..h {
        let ncity = {
            let gi = g.hex_idx[spec.hex_coords[i]];
            g.hexes[gi].tile.cities.len() as i32
        };
        for s in 0..S_MAX {
            let k = i * S_MAX + s;
            let (p, r, c) = (enc.hex_ptr[k], enc.hex_res[k], enc.hex_slot_city[k]);
            assert!(p == PTR_NONE || p == PTR_NEUTRAL || is_corp(p), "{ctx}: hex {i} slot {s} ptr {p}");
            assert!(r == PTR_NONE || is_corp(r), "{ctx}: hex {i} slot {s} reservation {r}");
            assert!(c >= -1 && c < ncity, "{ctx}: hex {i} slot {s} city {c} of {ncity}");
            if c < 0 {
                assert_eq!(p, PTR_NONE);
                assert_eq!(r, PTR_NONE);
            }
            if p >= 0 || p == PTR_NEUTRAL {
                stats.tokens_seen += 1;
            }
            if r >= 0 {
                stats.reservations_seen += 1;
            }
        }
        let (label, tile) = (enc.hex_ids[i * 2], enc.hex_ids[i * 2 + 1]);
        assert!(label >= 0 && (label as usize) < LABELS.len() + 2);
        assert!(tile >= 0 && (tile as usize) <= spec.num_tiles() + 1);
        assert_ne!(label, 1, "{ctx}: hex {} label not in the vocabulary", spec.hex_coords[i]);
    }
    // every token on the map shows up exactly once
    let placed: usize = g.hexes.iter().flat_map(|hx| hx.tile.cities.iter()).map(|c| c.all_tokens().count()).sum();
    let ptrs = enc.hex_ptr.iter().filter(|&&p| p != PTR_NONE).count();
    assert_eq!(ptrs, placed, "{ctx}: tokens on the map vs pointers");

    // entity pointers
    for j in 0..e {
        for col in 0..P_ENT {
            let p = enc.ent_ptr[j * P_ENT + col];
            assert!(p == PTR_NONE || valid(p), "{ctx}: entity {j} ptr[{col}] = {p}");
        }
        for col in 0..P_HEX {
            let p = enc.ent_hex[j * P_HEX + col];
            assert!(p == PTR_NONE || (p >= 0 && (p as usize) < h), "{ctx}: entity {j} hex[{col}] = {p}");
        }
        let owner = enc.ent_ptr[j * P_ENT];
        match enc.ent_type[j] {
            ENT_MAJOR | ENT_MINOR | ENT_NATIONAL => assert!(owner == PTR_NONE || is_seat(owner)),
            ENT_PRIVATE | ENT_CONCESSION | ENT_MINOR_PROXY => assert!(owner == PTR_NONE || is_seat(owner) || is_corp(owner)),
            ENT_TRAIN => {
                for col in 2..4 {
                    let p = enc.ent_ptr[j * P_ENT + col];
                    assert!(p == PTR_NONE || enc.ent_type[p as usize] == ENT_PHASE);
                }
            }
            ENT_PHASE => {
                let p = enc.ent_ptr[j * P_ENT + 2];
                assert!(p == PTR_NONE || enc.ent_type[p as usize] == ENT_TRAIN);
            }
            _ => {}
        }
    }

    // relations
    let n_rel = enc.num_relations();
    stats.relations += n_rel;
    stats.max_relations = stats.max_relations.max(n_rel);
    let mut seen = HashSet::new();
    for r in 0..n_rel {
        let (s, d, k) = (enc.rel[r * 3], enc.rel[r * 3 + 1], enc.rel[r * 3 + 2]);
        assert!(valid(s) && valid(d), "{ctx}: relation {r} ({s}, {d}, {k})");
        assert!(k > REL_PAD && (k as usize) < RELATION_KINDS.len());
        assert!(seen.insert((s, d, k)), "{ctx}: duplicate relation ({s}, {d}, {k})");
        match k {
            REL_SHARES => assert!(is_corp(d) && (is_seat(s) || is_corp(s))),
            REL_PRESIDENT => assert!(is_seat(s) && is_corp(d)),
            REL_OWNS => assert!(matches!(enc.ent_type[d as usize], ENT_PRIVATE | ENT_CONCESSION | ENT_MINOR_PROXY)),
            REL_TRAINS => assert!(is_corp(s) && enc.ent_type[d as usize] == ENT_TRAIN),
            REL_BID | REL_SOLD => assert!(is_seat(s)),
            _ => {}
        }
        let v = enc.rel_val[r];
        // a standing bid may be $0 (1822's starting bid-box entries)
        assert!(v > 0.0 || (k == REL_BID && v == 0.0), "{ctx}: relation {k} ({s} -> {d}) value {v}");
    }
    // shares: each corporation's holdings add up to its certificates
    for (gi, c) in g.corporations.iter().enumerate() {
        let total: u32 = c.shares.iter().map(|s| s.percent as u32).sum();
        let ent = spec.e_corp0 + spec.corp_index[c.sym.as_str()];
        let f = &enc.ent_feat[ent * F_ENT..(ent + 1) * F_ENT];
        let col = |name: &str| CORPORATION_FEATURES.iter().position(|x| *x == name).unwrap();
        let seats: f32 = (0..MAX_SEATS).map(|s| f[col("pct_seat_0") + s]).sum();
        let other = f[col("other_units")] * f[col("total_units")] * 10.0 * f[col("share_unit_pct")];
        let sum = seats + f[col("pct_market")] + f[col("pct_ipo")] + other;
        assert!((sum * 100.0 - total as f32).abs() < 0.5, "{ctx}: corp {} holdings {sum} vs {total}", c.sym);
        let _ = gi;
    }

    // meta
    let m = enc.meta;
    assert_eq!(m[0], spec.title_id);
    assert_eq!(m[4] as usize, np);
    assert!((m[5] as usize) < np);
    assert_eq!(m[6], g.finished as i32);
    assert!(m[1] >= 0 && (m[1] as usize) < ROUND_KINDS.len());
    assert!(m[2] >= 0 && (m[2] as usize) < STEP_KINDS.len());
    if !g.finished {
        assert!(m[1] > 0, "{ctx}: no round kind");
        assert!(m[2] > 0, "{ctx}: no step kind ({})", g.rl_step_kind());
        assert!(m[3] == PTR_NONE || valid(m[3]), "{ctx}: acting entity {}", m[3]);
        stats.step_kinds.insert(STEP_KINDS[m[2] as usize]);
        stats.round_kinds.insert(ROUND_KINDS[m[1] as usize]);
    }
    // cert counts match the engine's
    let certs_col = PLAYER_FEATURES.iter().position(|x| *x == "certs").unwrap();
    let mut ids: Vec<u32> = g.players.iter().map(|p| p.id).collect();
    ids.sort_unstable();
    for s in 0..np {
        let pid = ids[(s + m[5] as usize) % np];
        let want = (g.num_certs_internal(pid) as f32 / (g.cert_limit as f32).max(1.0)).clamp(-FEATURE_CLIP, FEATURE_CLIP);
        assert_eq!(enc.ent_feat[(1 + s) * F_ENT + certs_col], want, "{ctx}: seat {s} certs");
    }
    // seat 0 is the mover
    if let Some(mover) = g.rl_mover_id() {
        let mut ids: Vec<u32> = g.players.iter().map(|p| p.id).collect();
        ids.sort_unstable();
        assert_eq!(ids[m[5] as usize], mover);
        let is_mover = PLAYER_FEATURES.iter().position(|x| *x == "is_mover").unwrap();
        assert_eq!(enc.ent_feat[F_ENT + is_mover], 1.0);
    }
    stats.positions += 1;
}

fn walk_and_check(title: &str, players: usize, seed: u64, max_moves: usize, stats: &mut Stats) {
    let mut g = new_game(title, players, seed);
    let mut rng = StdRng::seed_from_u64(seed * 7919 + players as u64);
    for _ in 0..max_moves {
        let enc = g.encode_tokens();
        check(&g, &enc, stats);
        // determinism: again, and on a clone
        assert_eq!(enc, g.encode_tokens(), "{title} {players}p: encoding not deterministic");
        assert_eq!(enc, g.clone_for_search().encode_tokens(), "{title} {players}p: clone encodes differently");
        if title == "1867" && !g.finished {
            let moves = g.g1867_legal_moves_with(CrossPlayerPolicy::RESTRICTED);
            let kind = g.rl_step_kind();
            stats.step_checked += 1;
            // the in-stock-round minor auction is a mode of BuySellParShares
            let kind = if kind == "BuySellParSharesViaBid" { "BuySellParShares" } else { kind };
            if !moves.is_empty() && !moves.iter().any(|m| format!("{:?}", m.step) == kind) {
                stats.step_mismatch += 1;
                let mut gen: Vec<String> = moves.iter().map(|m| format!("{:?}", m.step)).collect();
                gen.sort();
                gen.dedup();
                *stats.mismatch_kinds.entry(format!("{kind} vs {gen:?}")).or_insert(0) += 1;
            }
        }
        if !step(&mut g, &mut rng) {
            break;
        }
    }
    if g.finished {
        stats.finished_games += 1;
        let enc = g.encode_tokens();
        check(&g, &enc, stats);
    }
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

#[test]
fn schema_is_consistent() {
    for list in [
        HEX_FEATURES,
        TILE_FEATURES,
        GLOBAL_FEATURES,
        PLAYER_FEATURES,
        CORPORATION_FEATURES,
        COMPANY_FEATURES,
        TRAIN_FEATURES,
        PHASE_FEATURES,
    ] {
        let set: HashSet<_> = list.iter().collect();
        assert_eq!(set.len(), list.len(), "duplicate feature names");
        assert!(list.len() <= F_ENT || list == HEX_FEATURES || list == TILE_FEATURES);
    }
    // phase status columns follow PHASE_STATUSES
    let first = PHASE_FEATURES.iter().position(|x| x.starts_with("st_")).unwrap();
    for (i, s) in PHASE_STATUSES.iter().enumerate() {
        assert_eq!(PHASE_FEATURES[first + i], format!("st_{s}"));
    }
    assert_eq!(PHASE_FEATURES[first + PHASE_STATUSES.len()], "st_other");
    // terrain and zone columns follow their vocabularies
    let t0 = HEX_FEATURES.iter().position(|x| *x == "terrain_mountain").unwrap();
    for (i, t) in TERRAINS.iter().enumerate() {
        assert_eq!(HEX_FEATURES[t0 + i], format!("terrain_{t}"));
    }
    let z0 = CORPORATION_FEATURES.iter().position(|x| *x == "zone_par").unwrap();
    for (i, z) in MARKET_ZONES.iter().enumerate() {
        assert_eq!(CORPORATION_FEATURES[z0 + i], format!("zone_{z}"));
    }
    assert_eq!(META_COLUMNS.len(), META_LEN);
    assert_eq!(ENT_PTR_COLUMNS.len(), P_ENT);
    assert_eq!(ENT_HEX_COLUMNS.len(), P_HEX);
    assert_eq!(entity_features(ENT_PAD).len(), 0);
    // track block layout
    let tr = HEX_FEATURES.iter().position(|x| *x == "track_0_1").unwrap();
    assert_eq!(HEX_FEATURES[tr + 48], "junction");
    for (n, (i, j)) in (0..6).flat_map(|i| ((i + 1)..6).map(move |j| (i, j))).enumerate() {
        assert_eq!(pair_index(i, j), n);
        assert_eq!(HEX_FEATURES[tr + n], format!("track_{i}_{j}"));
        assert_eq!(HEX_FEATURES[tr + 15 + n], format!("via_{i}_{j}"));
    }
    for name in STEP_KINDS.iter().chain(ROUND_KINDS) {
        assert!(!name.is_empty());
    }
}

/// Every label, terrain, phase status, market zone and title of the
/// registered titles is in the global vocabularies, and no hex can hold
/// more tokens than `S_MAX` slots.
#[test]
fn vocab_covers_every_title() {
    let mut missing: Vec<String> = Vec::new();
    for t in crate::title::all_titles() {
        let name = t.name();
        if name.starts_with("TEST") {
            continue;
        }
        if title_id(name) == 0 {
            missing.push(format!("title {name}"));
        }
        let mut tiles: Vec<crate::tiles::TileDef> = t.tile_catalog().values().cloned().collect();
        for d in t.hex_definitions() {
            if let Some((dsl, color)) = t.preprinted_hex_dsl(d.coord) {
                tiles.push(crate::tiles::parse_preprinted_tile(d.coord, dsl, color_of(color)));
            }
            if let Some(l) = t.hex_label(d.coord) {
                if label_id(Some(l)) == 1 {
                    missing.push(format!("{name} label {l}"));
                }
            }
            if let Some((l, _)) = t.hex_future_label(d.coord) {
                if label_id(Some(l)) == 1 {
                    missing.push(format!("{name} future label {l}"));
                }
            }
        }
        if let Some(f) = t.g1822_family() {
            for (coord, dsl, color) in f.hex_dsl_all {
                tiles.push(crate::tiles::parse_preprinted_tile(coord, dsl, color_of(color)));
            }
        }
        let mut seen: Vec<String> = tiles.iter().filter_map(|t| t.label.clone()).collect();
        let mut terr: Vec<String> = tiles.iter().flat_map(|t| t.upgrades.iter().map(|u| u.terrain.clone())).collect();
        seen.sort();
        seen.dedup();
        terr.sort();
        terr.dedup();
        eprintln!("{name}: labels {seen:?} terrains {terr:?}");
        for tile in &tiles {
            if let Some(l) = &tile.label {
                if label_id(Some(l)) == 1 {
                    missing.push(format!("{name} tile {} label {l}", tile.name));
                }
            }
            for u in &tile.upgrades {
                if !u.terrain.is_empty() && vocab_index(TERRAINS, &u.terrain).is_none() {
                    missing.push(format!("{name} terrain {}", u.terrain));
                }
            }
            let slots: usize = tile.cities.iter().map(|c| c.slots as usize).sum();
            assert!(slots <= S_MAX - 1, "{name} tile {} has {slots} slots", tile.name);
        }
        for p in t.phases() {
            for s in p.status {
                if vocab_index(PHASE_STATUSES, s).is_none() {
                    missing.push(format!("{name} status {s}"));
                }
            }
        }
        for row in t.market_grid() {
            for cell in row.into_iter().flatten() {
                for z in cell.zones {
                    if z != crate::title::MarketZone::Normal && vocab_index(MARKET_ZONES, z.type_str()).is_none() {
                        missing.push(format!("{name} zone {}", z.type_str()));
                    }
                }
            }
        }
    }
    missing.sort();
    missing.dedup();
    assert!(missing.is_empty(), "not in the vocabularies: {missing:#?}");
}

/// Axial coordinates agree with the engine's adjacency, both ways.
#[test]
fn spec_geometry_matches_adjacency() {
    const DIRS: [[i32; 2]; 6] = [[1, 0], [1, -1], [0, -1], [-1, 0], [-1, 1], [0, 1]];
    for t in crate::title::all_titles() {
        let spec = rl_spec_for(t.name());
        assert_eq!(spec.hex_coords.len(), t.hex_definitions().len(), "{}", t.name());
        for i in 0..spec.num_hexes() {
            for d in 0..6 {
                let j = spec.hex_neighbors[i][d];
                if j < 0 {
                    continue;
                }
                let j = j as usize;
                let dq = [spec.hex_axial[j][0] - spec.hex_axial[i][0], spec.hex_axial[j][1] - spec.hex_axial[i][1]];
                assert!(DIRS.contains(&dq), "{} {} -> {} not adjacent in axial", t.name(), spec.hex_coords[i], spec.hex_coords[j]);
                assert_eq!(spec.hex_neighbors[j][(d + 3) % 6], i as i32, "{} adjacency not symmetric", t.name());
            }
            assert!((0.0..=1.0).contains(&spec.hex_pos[i][0]) && (0.0..=1.0).contains(&spec.hex_pos[i][1]));
        }
        // entity layout
        assert_eq!(spec.entity_names().len(), spec.num_entities);
        assert_eq!(spec.entity_types().len(), spec.num_entities);
        assert_eq!(spec.tile_feat.len(), spec.num_tiles() * F_TILE);
    }
    // 1830 keeps its frozen action-layout hex order
    let s30 = rl_spec_for("1830");
    assert_eq!(s30.hex_coords, crate::title::resolve("1830").action_hex_order());
    assert_eq!(s30.num_entities, 1 + MAX_SEATS + 8 + 6 + 6 + 6);
}

/// The 1822 family's companies split into privates, concessions and minor
/// proxies, each proxy / concession linked to its corporation.
#[test]
fn g1822_company_kinds() {
    let s = rl_spec_for("1822");
    let kinds = |k: i32| s.companies.iter().filter(|c| c.ent_type == k).count();
    assert_eq!(kinds(ENT_PRIVATE), 21);
    assert_eq!(kinds(ENT_CONCESSION), 10);
    assert_eq!(kinds(ENT_MINOR_PROXY), 30);
    for c in &s.companies {
        if c.ent_type != ENT_PRIVATE {
            let link = c.link.as_deref().unwrap_or_else(|| panic!("{} unlinked", c.sym));
            assert!(s.corp_index.contains_key(link), "{} -> {link}", c.sym);
        }
    }
    // 1830's CS / CA are privates, not concessions
    assert!(rl_spec_for("1830").companies.iter().all(|c| c.ent_type == ENT_PRIVATE));
}

#[test]
fn random_walks_1830() {
    let mut stats = Stats::default();
    for players in player_range("1830") {
        for seed in 1..=2 {
            walk_and_check("1830", players, seed, 1500, &mut stats);
        }
    }
    eprintln!("1830: {stats:?}");
    assert!(stats.positions > 3000);
    assert!(stats.tokens_seen > 0 && stats.reservations_seen > 0);
    for k in ["WaterfallAuction", "BuySellParShares", "Track", "BuyTrain"] {
        assert!(stats.step_kinds.contains(k), "1830 walks never reached {k}");
    }
}

#[test]
fn random_walks_1867() {
    let mut stats = Stats::default();
    for players in player_range("1867") {
        for seed in 1..=2 {
            walk_and_check("1867", players, seed, 1500, &mut stats);
        }
    }
    eprintln!("1867: {stats:?}");
    assert!(stats.positions > 3000);
    assert!(stats.tokens_seen > 0);
    for k in ["SingleItemAuction", "BuySellParShares", "Track", "BuyTrain", "Merge"] {
        assert!(stats.step_kinds.contains(k), "1867 walks never reached {k}");
    }
    assert!(stats.round_kinds.contains("Merger"));
    // the step kind read off the round state is a step the generator offers
    // moves from
    assert_eq!(stats.step_mismatch, 0, "1867 step kind disagreed with the generator {} / {} times", stats.step_mismatch, stats.step_checked);
}

#[test]
fn random_walks_1822_family() {
    for title in ["1822", "1822PNW", "1822MX", "1822CA"] {
        let mut stats = Stats::default();
        let range = player_range(title);
        for players in [*range.start(), *range.end()] {
            walk_and_check(title, players, 3, 600, &mut stats);
        }
        eprintln!("{title}: {stats:?}");
        assert!(stats.positions > 500, "{title}: {} positions", stats.positions);
        assert!(stats.step_kinds.contains("BuySellParShares"));
        assert!(stats.reservations_seen > 0, "{title}: no reservations seen");
    }
}

/// Mixed titles and player counts stack into one padded batch.
#[test]
fn batch_stacks_mixed_titles() {
    let games = [new_game("1830", 4, 1), new_game("1867", 5, 1), new_game("1822", 3, 1)];
    let encs: Vec<TokenEncoding> = games.iter().map(|g| g.encode_tokens()).collect();
    let b = py::Batch::stack(&encs, false);
    assert_eq!(b.b, 3);
    assert_eq!(b.h, encs.iter().map(|x| x.num_hexes).max().unwrap());
    assert_eq!(b.e, encs.iter().map(|x| x.num_entities).max().unwrap());
    for (i, x) in encs.iter().enumerate() {
        let n_mask = b.hex_mask[i * b.h..(i + 1) * b.h].iter().filter(|v| **v).count();
        assert_eq!(n_mask, x.num_hexes);
        let n_ent = b.ent_mask[i * b.e..(i + 1) * b.e].iter().filter(|v| **v).count();
        assert_eq!(n_ent, x.ent_type.iter().filter(|t| **t != ENT_PAD).count());
        assert_eq!(&b.meta[i * META_LEN..(i + 1) * META_LEN], &x.meta);
    }
    // de-duplicated maps: positions of one map share a block
    let mut g = new_game("1830", 4, 2);
    let mut rng = StdRng::seed_from_u64(5);
    let mut encs = vec![g.encode_tokens()];
    for _ in 0..5 {
        step(&mut g, &mut rng);
        encs.push(g.encode_tokens());
    }
    let d = py::Batch::stack(&encs, true);
    assert!(d.m < d.b, "stock-round positions share a map");
    let mi = d.map_index.as_ref().unwrap();
    for (i, x) in encs.iter().enumerate() {
        let k = mi[i] as usize;
        assert_eq!(&d.hex_feat[k * d.h * F_HEX..(k * d.h + x.num_hexes) * F_HEX], &x.hex_feat[..]);
        assert_eq!(&d.hex_ptr[k * d.h * S_MAX..(k * d.h + x.num_hexes) * S_MAX], &x.hex_ptr[..]);
    }
}

/// Seats rotate with the mover, corporations do not: the same position seen
/// by another mover keeps the hex block and moves the seat tokens.
#[test]
fn seat_rotation_moves_seats_only() {
    let mut g = new_game("1830", 4, 3);
    let mut rng = StdRng::seed_from_u64(3);
    // reach a stock round with shares held
    for _ in 0..400 {
        if matches!(g.round, crate::rounds::Round::Stock(_)) && g.corporations.iter().any(|c| c.floated) {
            break;
        }
        step(&mut g, &mut rng);
    }
    let enc = g.encode_tokens();
    let rot = enc.meta[5] as usize;
    // per-seat share percents equal the players' holdings, in rotated order
    let mut ids: Vec<u32> = g.players.iter().map(|p| p.id).collect();
    ids.sort_unstable();
    let spec = rl_spec_for("1830");
    let col = CORPORATION_FEATURES.iter().position(|x| *x == "pct_seat_0").unwrap();
    for (ci, c) in g.corporations.iter().enumerate() {
        let ent = spec.e_corp0 + ci;
        for s in 0..4 {
            let pid = ids[(s + rot) % 4];
            let pct = g.player_percent_of(pid, ci) as f32 / 100.0;
            assert_eq!(enc.ent_feat[ent * F_ENT + col + s], pct, "{} seat {s}", c.sym);
        }
    }
    // cash of seat s = player (s + rot)
    for s in 0..4 {
        let pid = ids[(s + rot) % 4];
        let cash = g.players.iter().find(|p| p.id == pid).unwrap().cash as f32 / g.starting_cash as f32;
        assert_eq!(enc.ent_feat[(1 + s) * F_ENT + 2], cash);
    }
}

// ---------------------------------------------------------------------------
// throughput (ignored): `cargo test --release --no-default-features
// rl_encode::tests::bench -- --ignored --nocapture`
// ---------------------------------------------------------------------------

fn collect_positions(title: &str, players: usize, seeds: u64, per_game: usize) -> Vec<BaseGame> {
    let mut out = Vec::new();
    for seed in 1..=seeds {
        let mut g = new_game(title, players, seed);
        let mut rng = StdRng::seed_from_u64(seed);
        for n in 0..per_game {
            if n % 3 == 0 {
                out.push(g.clone_for_search());
            }
            if !step(&mut g, &mut rng) {
                break;
            }
        }
    }
    out
}

fn time_per_position<F: FnMut(&BaseGame)>(games: &[BaseGame], reps: usize, mut f: F) -> f64 {
    let t = std::time::Instant::now();
    for _ in 0..reps {
        for g in games {
            f(g);
        }
    }
    t.elapsed().as_secs_f64() / (reps * games.len()) as f64
}

#[test]
#[ignore]
fn bench_encoders() {
    for (title, players) in [("1830", 4), ("1867", 4), ("1822", 4), ("1822CA", 5)] {
        let games = collect_positions(title, players, 3, 1500);
        let tok = time_per_position(&games, 5, |g| {
            std::hint::black_box(g.encode_tokens());
        });
        let mut line = format!(
            "{title} {players}p: {} positions, token encoder {:.1} us ({:.0}/s)",
            games.len(),
            tok * 1e6,
            1.0 / tok
        );
        if title == "1830" {
            let legacy = time_per_position(&games, 5, |g| {
                std::hint::black_box(g.encode_canonical());
            });
            line += &format!(", 1830 encoder {:.1} us ({:.0}/s)", legacy * 1e6, 1.0 / legacy);
        }
        let stack = {
            let encs: Vec<TokenEncoding> = games.iter().take(256).map(|g| g.encode_tokens()).collect();
            let t = std::time::Instant::now();
            for _ in 0..20 {
                std::hint::black_box(py::Batch::stack(&encs, true));
            }
            t.elapsed().as_secs_f64() / (20 * encs.len()) as f64
        };
        let stack_plain = {
            let encs: Vec<TokenEncoding> = games.iter().take(256).map(|g| g.encode_tokens()).collect();
            let t = std::time::Instant::now();
            for _ in 0..20 {
                std::hint::black_box(py::Batch::stack(&encs, false));
            }
            t.elapsed().as_secs_f64() / (20 * encs.len()) as f64
        };
        line += &format!(", stacking {:.1} us (maps deduped) / {:.1} us (plain)", stack * 1e6, stack_plain * 1e6);
        eprintln!("{line}");
    }
}
