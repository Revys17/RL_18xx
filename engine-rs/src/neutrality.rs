//! Training neutrality of 1830 (docs/SITE_ENGINE_RS.md §8.4): fixed-seed walks of 1830 through the RL index API
//! exactly as self-play drives it (`factored_legal_indices` -> `price_range_for_index` -> `apply_action_index`, i.e.
//! the factored enumeration, the frozen 1830 layout, `decode_index` and `process_action_native`), each walk hashed:
//! every position's sorted legal index set, the index and price chosen, the engine's action log (the 18xx.games JSON
//! `process_action_native` writes) and the final result.
//!
//! The pinned digests were computed on RL_18xx master a615a40, BEFORE any site-layer change (the first commit of
//! branch site/engine-rs-1830). Site work (src/site/**, the `site` / `wasm` features, `site_rules` branches) must
//! leave every digest unchanged: with `site_rules` off the RL engine's legal moves, actions and outcomes are
//! byte-identical. An intended change to 1830 self-play on master (a rules fix, e.g. X2 / X3) moves them: re-pin
//! from master itself, never from a branch carrying site changes, and say so in the commit. Each walk prints its
//! `(players, seed, variant, pick, digest)` line to paste:
//!
//!   cargo test --release --no-default-features neutrality -- --nocapture                      (master today)
//!   cargo test --release --no-default-features --features python neutrality -- --nocapture    (once pyo3 is optional)
//!   cargo test --release --no-default-features --features site neutrality -- --nocapture      (the site layer's build)
//!
//! The walk is the spike's `scripts/wasm_spike/selfplay_checksum.py` as a cargo test, so it runs under every feature
//! set the crate builds with and needs no wheel. Nothing here may depend on hash-map iteration order: the legal
//! indices are sorted, an index counts as a pass only if every entry behind it is one, the action log is serde_json
//! (sorted keys) and the result is collected into a BTreeMap.
//!
//! 1867's twin (the policy gradient's 1867 engine: the generator layout `1867-v1`) is at the end of the file
//! (docs/SITE_ENGINE_RS_1867.md §7).

use std::collections::{BTreeMap, HashMap};

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::game::BaseGame;

/// FNV-1a, 64 bits: stable across Rust versions and platforms (std's hasher isn't promised to be).
struct Fnv(u64);
impl Fnv {
    fn new() -> Self {
        Fnv(0xcbf2_9ce4_8422_2325)
    }
    fn bytes(&mut self, b: &[u8]) {
        for &x in b {
            self.0 = (self.0 ^ x as u64).wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    fn u64(&mut self, v: u64) {
        self.bytes(&v.to_le_bytes())
    }
    fn i64(&mut self, v: i64) {
        self.u64(v as u64)
    }
    fn str(&mut self, s: &str) {
        self.u64(s.len() as u64);
        self.bytes(s.as_bytes());
    }
}

/// The self-play variants a walk plays under (the generators' `start_positions.new_game` turns the RL rules on).
#[derive(Clone, Copy, Debug)]
enum Variant {
    /// Ruby's rules as the engine has them, no optional rules
    Plain,
    /// `no_buyback_rule` and `auction_unlock` on (self-play's rule variants)
    RlRules,
    /// the RL rules plus both optional rules the site's 1830 tables use
    RlRulesOptional,
}

/// How a walk picks among the legal indices.
#[derive(Clone, Copy, Debug)]
enum Pick {
    /// uniformly (the spike's checksum)
    Uniform,
    /// a non-pass move 80% of the time, so walks reach the late game (phases 5-D, rusting, bankruptcy, the end)
    Busy,
}

fn new_game(players: u32, variant: Variant) -> BaseGame {
    let ids: Vec<u32> = (1..=players).collect();
    let names: HashMap<u32, String> = ids.iter().map(|&i| (i, format!("P{i}"))).collect();
    let rules: Vec<String> = match variant {
        Variant::RlRulesOptional => {
            vec!["optional_6_train".into(), "multiple_brown_from_ipo".into()]
        }
        _ => Vec::new(),
    };
    let mut g = BaseGame::build_with_rules("1830", ids, names, rules);
    if !matches!(variant, Variant::Plain) {
        g.no_buyback_rule = true;
        g.auction_unlock = true;
    }
    g
}

/// `factored_legal_indices`: the de-duplicated, sorted flat indices of the factored enumeration in the game's layout,
/// each with whether it is a pass (every entry behind the index is a `Pass`).
fn legal_indices(g: &mut BaseGame) -> Vec<(u32, bool)> {
    let lo = g.factored_layout().expect("1830 has a layout");
    let mut by_index: BTreeMap<u32, bool> = BTreeMap::new();
    for la in g.get_factored_choices_impl() {
        if let Some(i) = crate::action_index::legal_action_to_index_in(lo, &la) {
            let pass = la.action_type == "Pass";
            by_index.entry(i).and_modify(|p| *p &= pass).or_insert(pass);
        }
    }
    by_index.into_iter().collect()
}

/// One walk: its digest, the number of moves applied, and where it ended (finished?, the phase).
fn walk(
    players: u32,
    seed: u64,
    variant: Variant,
    pick: Pick,
    max_moves: usize,
) -> (u64, usize, bool, String) {
    let mut g = new_game(players, variant);
    let mut rng = StdRng::seed_from_u64(seed);
    let mut h = Fnv::new();
    let mut moves = 0usize;
    while moves < max_moves && !g.finished {
        let legal = legal_indices(&mut g);
        if legal.is_empty() {
            break;
        }
        h.u64(legal.len() as u64);
        for &(i, pass) in &legal {
            h.u64(i as u64);
            h.u64(pass as u64);
        }
        let idx = match pick {
            Pick::Uniform => legal[rng.gen_range(0..legal.len())].0,
            Pick::Busy => {
                let busy: Vec<u32> = legal
                    .iter()
                    .filter(|(_, pass)| !pass)
                    .map(|(i, _)| *i)
                    .collect();
                if !busy.is_empty() && rng.gen::<f64>() < 0.8 {
                    busy[rng.gen_range(0..busy.len())]
                } else {
                    legal[rng.gen_range(0..legal.len())].0
                }
            }
        };
        // `price_range_for_index`, then a price drawn in it as self-play's sampler would
        let range = g
            .legal_action_for_index(idx)
            .expect("index API")
            .and_then(|la| la.price_range);
        let price = range.map(|(lo, hi)| if hi <= lo { lo } else { rng.gen_range(lo..=hi) });
        h.u64(idx as u64);
        h.i64(price.unwrap_or(i64::MIN));
        // `apply_action_index`
        let action = g.decode_index(idx, price).unwrap_or_else(|e| {
            panic!(
                "{players}p seed {seed} {variant:?}: index {idx} offered but not decodable: {}",
                e.message
            )
        });
        g.process_action_native(&action).unwrap_or_else(|e| {
            panic!(
                "{players}p seed {seed} {variant:?}: index {idx} offered but refused: {}",
                e.message
            )
        });
        moves += 1;
    }
    // the engine's action log: what self-play's games are stored and replayed from
    h.u64(g.action_log.len() as u64);
    for a in &g.action_log {
        h.str(&serde_json::to_string(a).expect("action JSON"));
    }
    h.u64(g.finished as u64);
    let result: BTreeMap<u32, i32> = g.calculate_results().into_iter().collect();
    for (p, v) in result {
        h.u64(p as u64);
        h.i64(v as i64);
    }
    (h.0, moves, g.finished, g.phase.name.clone())
}

const MAX_MOVES: usize = 2500;
/// 60 walks: ~70k moves, a few seconds in release.
const WALKS: u64 = 60;

/// (players, seed, variant, pick, digest): pinned on master a615a40 (see the module doc to re-pin).
#[rustfmt::skip]
const PINNED: &[(u32, u64, Variant, Pick, u64)] = {
    use Pick::*;
    use Variant::*;
    &[
        (2, 1830, Plain, Busy, 0xd273ba0d4fcd1ef5), // 1608 moves, finished in phase D
        (3, 1831, RlRules, Uniform, 0xf6fe94f698d02307), // 1391 moves, finished in phase D
        (4, 1832, RlRulesOptional, Busy, 0x6deb0a9eeac02a3c), // 1050 moves, finished in phase D
        (5, 1833, Plain, Uniform, 0x9697b3b7cf070830), // 2276 moves, finished in phase D
        (6, 1834, RlRules, Busy, 0x7ffa367fc0b4c986), // 987 moves, finished in phase 6
        (2, 1835, RlRulesOptional, Uniform, 0x2f944e7c514d6398), // 1386 moves, finished in phase D
        (3, 1836, Plain, Busy, 0xba9f2aab13559425), // 1412 moves, finished in phase D
        (4, 1837, RlRules, Uniform, 0xcc0cf2001c4c4b7b), // 382 moves, finished in phase 4
        (5, 1838, RlRulesOptional, Busy, 0x9bfa1caa87db6787), // 268 moves, finished in phase 4
        (6, 1839, Plain, Uniform, 0xa25ab8694abc9ea8), // 945 moves, finished in phase 6
        (2, 1840, RlRules, Busy, 0xc0a6452fde1667fa), // 973 moves, finished in phase D
        (3, 1841, RlRulesOptional, Uniform, 0x30c3f266609f1ee9), // 1447 moves, finished in phase D
        (4, 1842, Plain, Busy, 0x080a8a0a1c13cf07), // 1609 moves, finished in phase D
        (5, 1843, RlRules, Uniform, 0x5e9a03c5b2966480), // 1304 moves, finished in phase D
        (6, 1844, RlRulesOptional, Busy, 0x3f3d25772f96e5fc), // 1435 moves, finished in phase D
        (2, 1845, Plain, Uniform, 0x88999fabcbb2a730), // 1601 moves, finished in phase D
        (3, 1846, RlRules, Busy, 0xbc64c50f4ea43a47), // 454 moves, finished in phase 6
        (4, 1847, RlRulesOptional, Uniform, 0xc9ea14d7a49a04bd), // 979 moves, finished in phase D
        (5, 1848, Plain, Busy, 0xd9dbd73a525cf256), // 1072 moves, finished in phase D
        (6, 1849, RlRules, Uniform, 0x9712cde95d24f5c9), // 1351 moves, finished in phase D
        (2, 1850, RlRulesOptional, Busy, 0xd215370a5fc16ba0), // 1305 moves, finished in phase D
        (3, 1851, Plain, Uniform, 0x250cf8249e8ec326), // 980 moves, finished in phase D
        (4, 1852, RlRules, Busy, 0x51b8b4f3889b8db3), // 947 moves, finished in phase D
        (5, 1853, RlRulesOptional, Uniform, 0x1870211aec94693c), // 934 moves, finished in phase 6
        (6, 1854, Plain, Busy, 0xb064cf3759e57b53), // 1402 moves, finished in phase D
        (2, 1855, RlRules, Uniform, 0x94551dc5c6fa57db), // 1485 moves, finished in phase D
        (3, 1856, RlRulesOptional, Busy, 0xdbd6e2a4a4bf5c22), // 1533 moves, finished in phase D
        (4, 1857, Plain, Uniform, 0xb1bb809e12d113d6), // 901 moves, finished in phase D
        (5, 1858, RlRules, Busy, 0x231c4e649ae090a7), // 909 moves, finished in phase 6
        (6, 1859, RlRulesOptional, Uniform, 0x9f6776b0a7294ea2), // 397 moves, finished in phase 4
        (2, 1860, Plain, Busy, 0xd3c05882ff4260f6), // 1395 moves, finished in phase D
        (3, 1861, RlRules, Uniform, 0x7047b4799ef3c7d5), // 949 moves, finished in phase D
        (4, 1862, RlRulesOptional, Busy, 0x01a30f7eca972a7c), // 208 moves, finished in phase 4
        (5, 1863, Plain, Uniform, 0x4c74b167e6ab2b48), // 1777 moves, finished in phase D
        (6, 1864, RlRules, Busy, 0x60e626129b434fe8), // 635 moves, finished in phase 6
        (2, 1865, RlRulesOptional, Uniform, 0x302db4b817d6f3c9), // 948 moves, finished in phase D
        (3, 1866, Plain, Busy, 0xf863f72d34bae185), // 1352 moves, finished in phase D
        (4, 1867, RlRules, Uniform, 0x045191508983bfd2), // 1276 moves, finished in phase D
        (5, 1868, RlRulesOptional, Busy, 0x484fe8d25f99dc83), // 1075 moves, finished in phase D
        (6, 1869, Plain, Uniform, 0x49e4565882ebbaec), // 2226 moves, finished in phase D
        (2, 1870, RlRules, Busy, 0x07e37dc937ad6e26), // 667 moves, finished in phase D
        (3, 1871, RlRulesOptional, Uniform, 0x4a8cbb5b0af4dc4d), // 1499 moves, finished in phase D
        (4, 1872, Plain, Busy, 0x9bbdde4a5bd8db08), // 2500 moves, stopped in phase 2
        (5, 1873, RlRules, Uniform, 0xbd53aede0582328c), // 617 moves, finished in phase 4
        (6, 1874, RlRulesOptional, Busy, 0xea9107b3f10d2d55), // 1272 moves, finished in phase D
        (2, 1875, Plain, Uniform, 0x556b60a2c1e81f6e), // 1492 moves, finished in phase D
        (3, 1876, RlRules, Busy, 0x1cf8ce43f0a61edc), // 340 moves, finished in phase 4
        (4, 1877, RlRulesOptional, Uniform, 0xd9813a1c30fc4bfd), // 960 moves, finished in phase D
        (5, 1878, Plain, Busy, 0x432728e0b96566fe), // 1841 moves, finished in phase D
        (6, 1879, RlRules, Uniform, 0x3ed843a69caf1e00), // 1439 moves, finished in phase 6
        (2, 1880, RlRulesOptional, Busy, 0x714d365ae235c5a4), // 1012 moves, finished in phase 5
        (3, 1881, Plain, Uniform, 0x89e60f39767a3562), // 2500 moves, stopped in phase 2
        (4, 1882, RlRules, Busy, 0xf995406f3600fd78), // 695 moves, finished in phase D
        (5, 1883, RlRulesOptional, Uniform, 0xe94c2e06ccfd8fad), // 1408 moves, finished in phase D
        (6, 1884, Plain, Busy, 0xf0a5beb55e0a0466), // 1821 moves, finished in phase D
        (2, 1885, RlRules, Uniform, 0x8af5067cc0bd8b21), // 1395 moves, finished in phase D
        (3, 1886, RlRulesOptional, Busy, 0xb21774f654ee5ccd), // 1440 moves, finished in phase D
        (4, 1887, Plain, Uniform, 0x150f42acd3595747), // 1118 moves, finished in phase D
        (5, 1888, RlRules, Busy, 0xcea1f09c68e012d4), // 1884 moves, finished in phase D
        (6, 1889, RlRulesOptional, Uniform, 0x2ce729988fb8bb6c), // 1642 moves, finished in phase D
    ]
};

#[test]
fn training_neutrality_1830_walks() {
    use Pick::*;
    use Variant::*;
    let cases: Vec<(u32, u64, Variant, Pick)> = (0..WALKS)
        .map(|k| {
            let players = 2 + (k % 5) as u32;
            let variant = [Plain, RlRules, RlRulesOptional][(k % 3) as usize];
            let pick = if k % 2 == 0 { Busy } else { Uniform };
            (players, 1830 + k, variant, pick)
        })
        .collect();
    let mut got = Vec::new();
    let (mut total, mut ended) = (0, 0);
    for &(players, seed, variant, pick) in &cases {
        let (digest, n, finished, phase) = walk(players, seed, variant, pick, MAX_MOVES);
        total += n;
        ended += finished as usize;
        let end = if finished { "finished" } else { "stopped" };
        eprintln!("    ({players}, {seed}, {variant:?}, {pick:?}, 0x{digest:016x}), // {n} moves, {end} in phase {phase}");
        got.push((players, seed, variant, pick, digest));
    }
    eprintln!(
        "training neutrality: {} walks, {total} moves, {ended} finished",
        cases.len()
    );
    assert_eq!(
        got.len(),
        PINNED.len(),
        "the pinned list and the cases differ: re-pin from master (module doc)"
    );
    for (g, p) in got.iter().zip(PINNED) {
        assert_eq!((g.0, g.1), (p.0, p.1), "the pinned list's order changed");
        assert_eq!(
            g.4, p.4,
            "1830 self-play changed: {}p seed {} {:?} {:?}. With site_rules off the RL engine must be byte-identical \
             to master (docs/SITE_ENGINE_RS.md §8.4)",
            g.0, g.1, g.2, g.3
        );
    }
}

/// The same walk twice gives the same digest (the enumeration's hash maps must not leak into the walk).
#[test]
fn training_neutrality_walk_is_deterministic() {
    let a = walk(4, 7, Variant::RlRules, Pick::Busy, 600);
    let b = walk(4, 7, Variant::RlRules, Pick::Busy, 600);
    assert_eq!((a.0, a.1), (b.0, b.1));
}

// =====================================================================================================================
// 1867 (docs/SITE_ENGINE_RS_1867.md §7): the same guard for the 1867 policy gradient's engine, pinned BEFORE any 1867
// site change (the first commit of branch site/engine-rs-1867, on site/engine-rs-1830 f6a9fe5, whose 1867 digests
// equal RL_18xx origin/master cd97116's: checked on a `git archive` of each). Fixed-seed walks of 1867 through the
// generator layout `1867-v1` exactly as the PG workers drive it (`rl_legal_moves_json` / `rl_apply_index`: the
// legal-move generator under a cross-player policy, the index of each move, the price on the move's range, the
// router's run, `process_action_internal`). Each walk hashes every position's legal moves (index, price range, pass),
// the index and price chosen, the generic token encoder's output at every 5th position (`encode_tokens`, what the
// model reads), the engine's action log (the wire dicts) and the result. Site work must leave every digest unchanged.
// Re-pin only from RL_18xx master itself, for an intended change to 1867 there, and say so in the commit:
//
//   cargo test --release --no-default-features --features site neutrality_1867 -- --nocapture

use crate::title::CrossPlayerPolicy;

fn hash_encoding(h: &mut Fnv, g: &BaseGame) {
    let e = g.encode_tokens();
    h.i64(e.title_id as i64);
    h.u64(e.num_hexes as u64);
    h.u64(e.num_entities as u64);
    for v in [&e.hex_feat, &e.ent_feat, &e.rel_val] {
        h.u64(v.len() as u64);
        for x in v.iter() {
            h.u64(x.to_bits() as u64);
        }
    }
    for v in [&e.hex_ptr, &e.hex_res, &e.hex_slot_city, &e.hex_ids, &e.ent_type, &e.ent_ptr, &e.ent_hex, &e.rel] {
        h.u64(v.len() as u64);
        for x in v.iter() {
            h.i64(*x as i64);
        }
    }
    for x in e.meta.iter() {
        h.i64(*x as i64);
    }
    h.u64(e.truncated_slots as u64);
}

/// One 1867 walk: its digest, the number of moves applied, and where it ended (finished?, the phase).
fn walk_1867(players: u32, seed: u64, policy: CrossPlayerPolicy, pick: Pick, max_moves: usize) -> (u64, usize, bool, String) {
    let ids: Vec<u32> = (1..=players).collect();
    let names: HashMap<u32, String> = ids.iter().map(|&i| (i, format!("Player {i}"))).collect();
    let mut g = BaseGame::build_titled("1867", ids, names);
    let mut rng = StdRng::seed_from_u64(seed);
    let mut h = Fnv::new();
    let mut moves = 0usize;
    while moves < max_moves && !g.finished {
        let legal = g.rl_indexed_moves(policy).unwrap_or_else(|e| panic!("{players}p seed {seed}: index API: {}", e.message));
        if legal.is_empty() {
            break;
        }
        h.u64(legal.len() as u64);
        for m in &legal {
            h.u64(m.index as u64);
            h.u64(m.mv.action.action_type().len() as u64);
            match m.mv.price {
                Some(r) => {
                    h.i64(r.lo as i64);
                    h.i64(r.hi as i64);
                    h.i64(r.step as i64);
                }
                None => h.i64(i64::MIN),
            }
            h.u64((m.mv.action.action_type() == "pass") as u64);
        }
        if moves % 5 == 0 {
            hash_encoding(&mut h, &g);
        }
        let k = match pick {
            Pick::Uniform => rng.gen_range(0..legal.len()),
            Pick::Busy => {
                let busy: Vec<usize> =
                    (0..legal.len()).filter(|&i| legal[i].mv.action.action_type() != "pass").collect();
                if !busy.is_empty() && rng.gen::<f64>() < 0.8 {
                    busy[rng.gen_range(0..busy.len())]
                } else {
                    rng.gen_range(0..legal.len())
                }
            }
        };
        let idx = legal[k].index;
        // a price on the move's range and step, as the PG sampler draws one
        let price = legal[k].mv.price.map(|r| {
            let n = ((r.hi - r.lo).max(0) / r.step.max(1)) as i64;
            r.lo + (rng.gen_range(0..=n) as i32) * r.step.max(1)
        });
        h.u64(idx as u64);
        h.i64(price.map_or(i64::MIN, |p| p as i64));
        g.rl_apply_index(idx, price, policy).unwrap_or_else(|e| {
            panic!("{players}p seed {seed}: index {idx} offered but refused: {}", e.message)
        });
        moves += 1;
    }
    h.u64(g.action_log.len() as u64);
    for a in &g.action_log {
        h.str(&serde_json::to_string(a).expect("action JSON"));
    }
    h.u64(g.finished as u64);
    let result: BTreeMap<u32, i32> = g.calculate_results().into_iter().collect();
    for (p, v) in result {
        h.u64(p as u64);
        h.i64(v as i64);
    }
    (h.0, moves, g.finished, g.phase.name.clone())
}

const MAX_MOVES_1867: usize = 3000;
/// 24 walks: 3-6 players, the title's policy (self-play's) twice for each of Ruby's (the importer's matching set).
const WALKS_1867: u64 = 24;

fn cases_1867() -> Vec<(u32, u64, bool, Pick)> {
    (0..WALKS_1867)
        .map(|k| {
            let players = 3 + (k % 4) as u32;
            let rules = k % 3 == 2;
            let pick = if k % 2 == 0 { Pick::Busy } else { Pick::Uniform };
            (players, 1867 + k, rules, pick)
        })
        .collect()
}

/// (players, seed, rules policy?, pick, digest): pinned before any 1867 site change (the 1867 note above), on
/// site/engine-rs-1830 f6a9fe5 = RL_18xx origin/master cd97116 for 1867.
#[rustfmt::skip]
const PINNED_1867: &[(u32, u64, bool, Pick, u64)] = {
    use Pick::*;
    &[
        (3, 1867, false, Busy, 0x7db4f9779a9b08db), // 1101 moves, finished in phase 8
        (4, 1868, false, Uniform, 0x376ce18523e651a8), // 1187 moves, finished in phase 8
        (5, 1869, true, Busy, 0x6887c359d4626858), // 2099 moves, finished in phase 8
        (6, 1870, false, Uniform, 0xd56b0baa95c19e11), // 1596 moves, finished in phase 8
        (3, 1871, false, Busy, 0x1f7a57bf6e45093d), // 1126 moves, finished in phase 8
        (4, 1872, true, Uniform, 0xdaa5b31860f9394b), // 1477 moves, finished in phase 8
        (5, 1873, false, Busy, 0x4837b8b9e063061d), // 1093 moves, finished in phase 8
        (6, 1874, false, Uniform, 0x728db9de668b1221), // 1469 moves, finished in phase 8
        (3, 1875, true, Busy, 0xae71d5b0433eefa2), // 1303 moves, finished in phase 8
        (4, 1876, false, Uniform, 0x0e079ef1c44fa84e), // 1124 moves, finished in phase 8
        (5, 1877, false, Busy, 0xc95c5f94cb505817), // 1213 moves, finished in phase 8
        (6, 1878, true, Uniform, 0x180b7add8fdcc5b3), // 1491 moves, finished in phase 8
        (3, 1879, false, Busy, 0xcd655fe7d16ec0a2), // 1013 moves, finished in phase 8
        (4, 1880, false, Uniform, 0xb32b9e275eba6be5), // 1339 moves, finished in phase 8
        (5, 1881, true, Busy, 0xda4aeff83553765e), // 1583 moves, finished in phase 8
        (6, 1882, false, Uniform, 0xbe618483e1a2379c), // 1216 moves, finished in phase 8
        (3, 1883, false, Busy, 0xe4ff25dfa295daed), // 938 moves, finished in phase 8
        (4, 1884, true, Uniform, 0x44fea0494724e0dc), // 1588 moves, finished in phase 8
        (5, 1885, false, Busy, 0x3da3efa405e71b1a), // 1191 moves, finished in phase 8
        (6, 1886, false, Uniform, 0xb13b31b73276163c), // 1526 moves, finished in phase 8
        (3, 1887, true, Busy, 0xe9bc439861f4544d), // 1327 moves, finished in phase 8
        (4, 1888, false, Uniform, 0x22d23815933efc22), // 1073 moves, finished in phase 8
        (5, 1889, false, Busy, 0xef94e199cbaf37c3), // 1163 moves, finished in phase 8
        (6, 1890, true, Uniform, 0x93b4a1efcd8b4707), // 1814 moves, finished in phase 8
    ]
};

#[test]
fn training_neutrality_1867_walks() {
    let mut got = Vec::new();
    let (mut total, mut ended) = (0, 0);
    for (players, seed, rules, pick) in cases_1867() {
        let policy = if rules { CrossPlayerPolicy::RULES } else { CrossPlayerPolicy::RESTRICTED };
        let (digest, n, finished, phase) = walk_1867(players, seed, policy, pick, MAX_MOVES_1867);
        total += n;
        ended += finished as usize;
        let end = if finished { "finished" } else { "stopped" };
        eprintln!("    ({players}, {seed}, {rules}, {pick:?}, 0x{digest:016x}), // {n} moves, {end} in phase {phase}");
        got.push((players, seed, rules, pick, digest));
    }
    eprintln!("training neutrality 1867: {} walks, {total} moves, {ended} finished", got.len());
    assert_eq!(got.len(), PINNED_1867.len(), "the pinned 1867 list and the cases differ: re-pin from master (module doc)");
    for (g, p) in got.iter().zip(PINNED_1867) {
        assert_eq!((g.0, g.1, g.2), (p.0, p.1, p.2), "the pinned 1867 list's order changed");
        assert_eq!(
            g.4, p.4,
            "1867 self-play changed: {}p seed {} rules {} {:?}. With site_rules off the RL engine must be byte-identical \
             to master (docs/SITE_ENGINE_RS_1867.md §7)",
            g.0, g.1, g.2, g.3
        );
    }
}

/// The same 1867 walk twice gives the same digest (no hash-map order in the generator, the router or the encoder).
#[test]
fn training_neutrality_1867_walk_is_deterministic() {
    let a = walk_1867(4, 7, CrossPlayerPolicy::RESTRICTED, Pick::Busy, 400);
    let b = walk_1867(4, 7, CrossPlayerPolicy::RESTRICTED, Pick::Busy, 400);
    assert_eq!((a.0, a.1), (b.0, b.1));
}
