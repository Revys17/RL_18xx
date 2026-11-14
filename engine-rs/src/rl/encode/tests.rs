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
    let is_family = g.g1822.is_some();
    let n_rel = enc.num_relations();
    stats.relations += n_rel;
    stats.max_relations = stats.max_relations.max(n_rel);
    let mut seen = HashSet::new();
    for r in 0..n_rel {
        let (s, d, k) = (enc.rel[r * 3], enc.rel[r * 3 + 1], enc.rel[r * 3 + 2]);
        assert!(valid(s) && valid(d), "{ctx}: relation {r} ({s}, {d}, {k})");
        assert!(k > REL_PAD && (k as usize) < RELATION_KINDS.len());
        assert!(seen.insert((s, d, k)), "{ctx}: duplicate relation ({s}, {d}, {k})");
        let is_company = |x: i32| matches!(enc.ent_type[x as usize], ENT_PRIVATE | ENT_CONCESSION | ENT_MINOR_PROXY);
        match k {
            // a company holds shares only as the 1822 family's tax haven
            REL_SHARES => assert!(is_corp(d) && (is_seat(s) || is_corp(s) || (is_family && is_company(s)))),
            REL_PRESIDENT => assert!(is_seat(s) && is_corp(d)),
            REL_OWNS => assert!(is_company(d)),
            // a company holds a train only as an 1822 train private
            REL_TRAINS => assert!((is_corp(s) || (is_family && is_company(s))) && enc.ent_type[d as usize] == ENT_TRAIN),
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
    let ids: Vec<u32> = g.players.iter().map(|p| p.id).collect();
    // seating order (BaseGame.players), as the descriptor vocabulary
    for s in 0..np {
        let pid = ids[(s + m[5] as usize) % np];
        // the 1822 family counts its own way (`g1822_num_certs`)
        let n = if is_family { g.g1822_num_certs(pid) as f32 } else { g.num_certs_internal(pid) as f32 };
        let want = (n / (g.cert_limit as f32).max(1.0)).clamp(-FEATURE_CLIP, FEATURE_CLIP);
        assert_eq!(enc.ent_feat[(1 + s) * F_ENT + certs_col], want, "{ctx}: seat {s} certs");
    }
    // seat 0 is the mover
    if let Some(mover) = g.rl_mover_id() {
        let ids: Vec<u32> = g.players.iter().map(|p| p.id).collect();
        // seating order (BaseGame.players)
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
    let s30 = rl_spec_for("1830");
    assert_eq!(s30.num_entities, 1 + MAX_SEATS + 8 + 6 + 6 + 6);
}

/// Tokens use the descriptor vocabularies' orders (`rl::Vocab`,
/// docs/rl_descriptors.md §4): hexes in `hex_definitions()` order (=
/// `BaseGame.hexes`), tiles in `tile_counts()` order, corporations,
/// companies and train types in title-data order.
#[test]
fn token_orders_are_the_descriptor_vocabularies() {
    for t in crate::title::all_titles() {
        let spec = rl_spec_for(t.name());
        let vocab = crate::rl::Vocab::for_title(*t);
        assert_eq!(spec.hex_coords, vocab.hexes, "{} hexes", t.name());
        assert_eq!(spec.tile_ids, vocab.tiles, "{} tiles", t.name());
        assert_eq!(spec.corps.iter().map(|c| c.sym).collect::<Vec<_>>(), vocab.corporations, "{}", t.name());
        assert_eq!(spec.companies.iter().map(|c| c.sym).collect::<Vec<_>>(), vocab.companies, "{}", t.name());
        assert_eq!(spec.trains.iter().map(|c| c.name).collect::<Vec<_>>(), vocab.trains, "{}", t.name());
        // BaseGame.hexes is that order too
        if !t.name().starts_with("TEST") {
            let g = new_game(t.name(), 4, 1);
            let ids: Vec<&str> = g.hexes.iter().map(|h| h.id.as_str()).collect();
            assert_eq!(ids, spec.hex_coords, "{} BaseGame.hexes", t.name());
        }
    }
}

/// Every descriptor of the frozen 1830 layout addresses the token of the
/// entity / hex / tile its vocabulary names.
#[test]
fn descriptors_index_the_tokens() {
    use crate::rl::descriptor::{EntKind, NONE};
    let layout = crate::rl::rl_layout("1830").expect("1830 layout");
    let spec = rl_spec_for("1830");
    let names = spec.entity_names();
    let types = spec.entity_types();
    let (mut ents, mut hexes, mut tiles) = (0, 0, 0);
    for d in &layout.descriptors {
        for (kind, index) in [(d.ent0_kind, d.ent0), (d.ent1_kind, d.ent1)] {
            if index == NONE {
                continue;
            }
            let tok = spec.descriptor_entity(kind, index, 0, 4);
            assert!(tok >= 0, "{d:?}");
            let tok = tok as usize;
            let want = if kind == EntKind::Corporation.id() {
                format!("corp:{}", layout.vocab.corporations[index as usize])
            } else if kind == EntKind::Company.id() {
                format!("company:{}", layout.vocab.companies[index as usize])
            } else if kind == EntKind::Train.id() {
                format!("train:{}", layout.vocab.trains[index as usize])
            } else {
                assert_eq!(types[tok], ENT_PLAYER);
                continue;
            };
            assert_eq!(names[tok], want);
            ents += 1;
        }
        if d.via != NONE {
            let tok = spec.descriptor_entity(EntKind::Company.id(), d.via, 0, 4) as usize;
            assert_eq!(names[tok], format!("company:{}", layout.vocab.companies[d.via as usize]));
        }
        if d.hex != NONE {
            assert_eq!(spec.hex_coords[d.hex as usize], layout.vocab.hexes[d.hex as usize]);
            hexes += 1;
        }
        if d.tile != NONE {
            assert_eq!(spec.tile_ids[d.tile as usize], layout.vocab.tiles[d.tile as usize]);
            tiles += 1;
        }
    }
    assert!(ents > 0 && hexes > 25_000 && tiles > 25_000, "{ents} {hexes} {tiles}");
    // a player index follows the rotation: seat 0 is the mover
    assert_eq!(spec.descriptor_entity(EntKind::Player.id(), 2, 2, 4), 1);
    assert_eq!(spec.descriptor_entity(EntKind::Player.id(), 1, 2, 4), 4);
    assert_eq!(spec.descriptor_entity(EntKind::Player.id(), 4, 0, 4), PTR_NONE);
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

/// Version 2's 1822-family features on longer walks: which ones the walks
/// reach (printed), their consistency with the engine (bid boxes, scores,
/// the subject pointer, the tax haven's shares, train privates), and the
/// ones every title must show.
#[test]
fn g1822_family_v2_features() {
    let col = |list: &[&str], name: &str| list.iter().position(|x| *x == name).unwrap();
    let v1 = |t: i32| v1_entity_width(t);
    for title in ["1822", "1822PNW", "1822MX", "1822CA"] {
        let spec = rl_spec_for(title);
        let mut seen: HashMap<String, usize> = HashMap::new();
        let mut note = |k: String| *seen.entry(k).or_insert(0) += 1;
        let mut positions = 0usize;
        let range = player_range(title);
        for players in [*range.start(), *range.end()] {
            for seed in 1..=2u64 {
                let mut g = new_game(title, players, seed + 20);
                let mut rng = StdRng::seed_from_u64(seed * 31 + players as u64);
                for _ in 0..2500 {
                    let enc = g.encode_tokens();
                    positions += 1;
                    let ctx = format!("{title} {players}p seed {seed} move {}", g.action_log.len());
                    for i in 0..enc.num_hexes {
                        for c in V1_F_HEX..F_HEX {
                            if enc.hex_feat[i * F_HEX + c] != 0.0 {
                                note(format!("hex.{}", HEX_FEATURES[c]));
                            }
                        }
                    }
                    for j in 0..enc.num_entities {
                        let t = enc.ent_type[j];
                        let names = entity_features(t);
                        for c in v1(t)..names.len() {
                            if enc.ent_feat[j * F_ENT + c] != 0.0 {
                                note(format!("{}.{}", ENTITY_TYPES[t as usize], names[c]));
                            }
                        }
                    }
                    for r in 0..enc.num_relations() {
                        let (s, d, k) = (enc.rel[r * 3] as usize, enc.rel[r * 3 + 1] as usize, enc.rel[r * 3 + 2]);
                        let company = |x: usize| matches!(enc.ent_type[x], ENT_PRIVATE | ENT_CONCESSION | ENT_MINOR_PROXY);
                        if k == REL_SHARES && company(s) {
                            note("rel.tax_haven_shares".into());
                            assert_eq!(spec.companies[s - spec.e_company0].sym, g.fam().company_osth, "{ctx}");
                        }
                        if k == REL_TRAINS && company(s) {
                            note("rel.company_train".into());
                        }
                        if k == REL_BID && matches!(enc.ent_type[d], ENT_MAJOR | ENT_MINOR | ENT_NATIONAL) {
                            note("rel.bid_on_corporation".into());
                        }
                    }
                    // the lot pointer names a subject
                    let lot = enc.ent_ptr[3];
                    if lot >= 0 {
                        note("global.lot".into());
                        let t = enc.ent_type[lot as usize];
                        let c = col(entity_features(t), "subject");
                        assert_eq!(enc.ent_feat[lot as usize * F_ENT + c], 1.0, "{ctx}: lot without the subject flag");
                    }
                    // a company is up for auction exactly when it sits in a bid box
                    let in_sr = g.g1822.as_ref().unwrap().round.kind == crate::g1822::RoundKind::Stock;
                    for k in 0..spec.companies.len() {
                        let j = spec.e_company0 + k;
                        let f = &enc.ent_feat[j * F_ENT..(j + 1) * F_ENT];
                        let up = f[col(COMPANY_FEATURES, "up_for_auction")];
                        let slot = f[col(COMPANY_FEATURES, "bidbox_slot")];
                        if in_sr {
                            assert_eq!(up > 0.0, slot > 0.0, "{ctx}: {} up {up} slot {slot}", spec.companies[k].sym);
                        }
                        assert!((0.0..=1.0).contains(&slot));
                    }
                    // the player's net worth is the engine's score
                    let rot = enc.meta[5] as usize;
                    let np = g.players.len();
                    for s in 0..np {
                        let pid = g.players[(s + rot) % np].id;
                        let want = (g.g1822_player_value(pid) as f32 / g.starting_cash as f32 / 4.0).clamp(-FEATURE_CLIP, FEATURE_CLIP);
                        let got = enc.ent_feat[(1 + s) * F_ENT + col(PLAYER_FEATURES, "net_worth")];
                        assert!((got - want).abs() < 1e-5, "{ctx}: seat {s} net worth {got} vs {want}");
                        let left = enc.ent_feat[(1 + s) * F_ENT + col(PLAYER_FEATURES, "bid_tokens_left")];
                        assert!((0.0..=1.0).contains(&left), "{ctx}: bid tokens left {left}");
                    }
                    if !step(&mut g, &mut rng) {
                        break;
                    }
                }
            }
        }
        let mut keys: Vec<_> = seen.iter().collect();
        keys.sort();
        eprintln!("{title}: {positions} positions; v2 features seen (positions): {keys:?}");
        for k in [
            "global.debt_interest",
            "global.bid_tokens",
            "global.sr_idle",
            "player.bid_tokens_left",
            "player.certs_with_bids",
            "minor_proxy.bidbox_slot",
            "minor_proxy.bidbox_first",
            "private.uses_left",
        ] {
            assert!(seen.contains_key(k), "{title}: the walks never showed {k}");
        }
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
    let ids: Vec<u32> = g.players.iter().map(|p| p.id).collect();
    // seating order (BaseGame.players), as the descriptor vocabulary
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
// version 1's columns of 1830 and 1867 stay bit-identical
// ---------------------------------------------------------------------------

/// Version 1's widths: hex features, entity features per type, entity and hex
/// pointers, meta. Columns appended since are zero (pointers -1) for 1830 and
/// 1867.
const V1_F_HEX: usize = 106;
const V1_P_ENT: usize = 4;
const V1_P_HEX: usize = 2;
const V1_META: usize = 8;
fn v1_entity_width(t: i32) -> usize {
    match t {
        ENT_GLOBAL => 29,
        ENT_PLAYER => 16,
        ENT_MAJOR | ENT_MINOR | ENT_NATIONAL => 53,
        ENT_PRIVATE | ENT_CONCESSION | ENT_MINOR_PROXY => 28,
        ENT_TRAIN => 17,
        ENT_PHASE => 22,
        _ => 0,
    }
}

struct Digest(u64);
impl Digest {
    fn new() -> Self {
        Digest(0xcbf2_9ce4_8422_2325)
    }
    fn u(&mut self, v: u64) {
        for b in v.to_le_bytes() {
            self.0 = (self.0 ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    fn f(&mut self, v: f32) {
        self.u(v.to_bits() as u64)
    }
    fn i(&mut self, v: i32) {
        self.u(v as u32 as u64)
    }
}

/// Digest of version 1's columns of one encoding; asserts the columns
/// appended since are zero / -1 (1830 and 1867 have none of the 1822 family's
/// state).
fn v1_digest(enc: &TokenEncoding, d: &mut Digest) {
    let (h, e) = (enc.num_hexes, enc.num_entities);
    d.i(enc.title_id);
    for i in 0..h {
        let row = &enc.hex_feat[i * F_HEX..(i + 1) * F_HEX];
        row[..V1_F_HEX].iter().for_each(|v| d.f(*v));
        assert!(row[V1_F_HEX..].iter().all(|v| *v == 0.0), "hex {i}: appended columns not zero");
    }
    for v in [&enc.hex_ptr, &enc.hex_res, &enc.hex_slot_city, &enc.hex_ids] {
        v.iter().for_each(|x| d.i(*x));
    }
    for j in 0..e {
        let t = enc.ent_type[j];
        d.i(t);
        let row = &enc.ent_feat[j * F_ENT..(j + 1) * F_ENT];
        let w = v1_entity_width(t);
        row[..w].iter().for_each(|v| d.f(*v));
        assert!(row[w..].iter().all(|v| *v == 0.0), "entity {j}: appended columns not zero");
        let p = &enc.ent_ptr[j * P_ENT..(j + 1) * P_ENT];
        p[..V1_P_ENT].iter().for_each(|x| d.i(*x));
        assert!(p[V1_P_ENT..].iter().all(|x| *x == PTR_NONE));
        let p = &enc.ent_hex[j * P_HEX..(j + 1) * P_HEX];
        p[..V1_P_HEX].iter().for_each(|x| d.i(*x));
        assert!(p[V1_P_HEX..].iter().all(|x| *x == PTR_NONE));
    }
    d.u(enc.num_relations() as u64);
    enc.rel.iter().for_each(|x| d.i(*x));
    enc.rel_val.iter().for_each(|v| d.f(*v));
    enc.meta[..V1_META].iter().for_each(|x| d.i(*x));
    assert!(enc.meta[V1_META..].iter().all(|x| *x == 0));
}

fn walk_digest(title: &str, players: usize, seed: u64, max_moves: usize) -> (u64, usize) {
    let mut g = new_game(title, players, seed);
    let mut rng = StdRng::seed_from_u64(seed * 7919 + players as u64);
    let mut d = Digest::new();
    let mut n = 0;
    for _ in 0..max_moves {
        v1_digest(&g.encode_tokens(), &mut d);
        n += 1;
        if !step(&mut g, &mut rng) {
            break;
        }
    }
    v1_digest(&g.encode_tokens(), &mut d);
    (d.0, n + 1)
}

/// 1830's and 1867's encodings are version 1's, bit for bit, on fixed random
/// walks (the digests were taken from version 1's encoder, 1757bbf / 46c879b);
/// every column appended since is zero (pointers -1) for them. A rules or
/// generator change that moves these walks changes the digests too: re-take
/// them from the previous encoder then.
#[test]
fn v1_columns_of_1830_and_1867_unchanged() {
    let cases: &[(&str, usize, u64, u64)] = &[
        ("1830", 3, 11, 0xf13b5fd4c3b71fee), // 1201 positions
        ("1830", 4, 12, 0xeb716bac08b0d1ee), // 1201 positions
        ("1830", 6, 13, 0xfb108a95c06046b3), // 572 positions
        ("1867", 3, 11, 0x6a9d8cfe150c499d), // 1102 positions
        ("1867", 4, 12, 0xebde5d1d8daac446), // 1056 positions
        ("1867", 6, 13, 0xf704ab16dc35a2f7), // 1174 positions
    ];
    let mut got = Vec::new();
    for &(title, players, seed, want) in cases {
        let (digest, n) = walk_digest(title, players, seed, 1200);
        eprintln!("(\"{title}\", {players}, {seed}, 0x{digest:016x}), // {n} positions");
        got.push((title, players, seed, digest, want));
    }
    for (title, players, seed, digest, want) in got {
        assert_eq!(digest, want, "{title} {players}p seed {seed}: version 1's columns changed");
    }
}

// ---------------------------------------------------------------------------
// throughput (ignored): `cargo test --release --no-default-features
// rl::encode::tests::bench -- --ignored --nocapture`
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

fn g1822_family(title: &str) -> bool {
    crate::title::resolve(title).is_g1822_family()
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
    for (title, players) in [("1830", 4), ("1867", 4), ("1822", 4), ("1822PNW", 4), ("1822MX", 4), ("1822CA", 5)] {
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
        if g1822_family(title) {
            // the part of an encoding that evaluates Ruby's active step
            let st = time_per_position(&games, 5, |g| {
                std::hint::black_box(g.g1822_active_step());
            });
            line += &format!(" [active step {:.1} us]", st * 1e6);
        }
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
