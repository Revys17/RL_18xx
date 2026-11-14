//! In-crate random-walk fuzzing of the 1822 legal-move generator
//! (RL-readiness contract, item 3): at every decision the generator must
//! offer at least one move until the game ends, every generated move must
//! apply, and nothing may panic. Walks run at 3-6 players, base rules and
//! 1822+ (plus_expansion), with seeded RNGs so failures reproduce.

use std::collections::HashMap;

use crate::game::BaseGame;

use super::legal::{Move, PriceRange};
use crate::title::CrossPlayerPolicy;

/// xorshift64* — deterministic, seedable.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    pub fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    pub fn below(&mut self, n: usize) -> usize {
        (self.next() % n.max(1) as u64) as usize
    }
}

pub fn new_game(players: usize, seed: u64, rules: &[&str]) -> BaseGame {
    new_game_titled("1822", players, seed, rules)
}

/// A fresh game of any 1822-family title.
pub fn new_game_titled(title: &str, players: usize, seed: u64, rules: &[&str]) -> BaseGame {
    let ids: Vec<u32> = (0..players as u32).map(|i| 1001 + i).collect();
    let names: HashMap<u32, String> = ids.iter().map(|&i| (i, format!("Player {}", i - 1000))).collect();
    BaseGame::build_seeded(title, ids, names, rules.iter().map(|s| s.to_string()).collect(), Some(seed))
}

/// A price inside a range (the ends favoured).
pub fn pick_price(r: &PriceRange, rng: &mut Rng) -> i32 {
    let steps = (r.hi - r.lo) / r.step.max(1);
    match rng.below(4) {
        0 => r.lo,
        1 => r.lo + steps * r.step,
        _ => r.lo + (rng.below(steps as usize + 1) as i32) * r.step,
    }
}

pub struct WalkReport {
    pub moves: usize,
    pub finished: bool,
    pub error: Option<String>,
    pub log: Vec<Move>,
}

/// Pick a move: non-pass moves 3:1 so walks reach the late game.
pub fn choose(moves: &[Move], rng: &mut Rng) -> Move {
    let non_pass: Vec<&Move> = moves.iter().filter(|m| m.action.action_type() != "pass").collect();
    let m = if !non_pass.is_empty() && rng.below(4) != 0 {
        non_pass[rng.below(non_pass.len())].clone()
    } else {
        moves[rng.below(moves.len())].clone()
    };
    match &m.price {
        Some(r) => m.with_price(pick_price(r, rng)),
        None => m,
    }
}

/// One random walk from a fresh game, generating under `policy`.
pub fn walk(players: usize, seed: u64, rules: &[&str], policy: CrossPlayerPolicy, max_moves: usize) -> WalkReport {
    walk_titled("1822", players, seed, rules, policy, max_moves)
}

/// [`walk`] for any 1822-family title.
pub fn walk_titled(
    title: &str,
    players: usize,
    seed: u64,
    rules: &[&str],
    policy: CrossPlayerPolicy,
    max_moves: usize,
) -> WalkReport {
    let mut g = new_game_titled(title, players, seed, rules);
    let mut rng = Rng::new(seed ^ 0xA5A5);
    let mut log = Vec::new();
    for n in 0..max_moves {
        if g.finished {
            return WalkReport { moves: n, finished: true, error: None, log };
        }
        let moves = g.g1822_legal_moves_with(policy);
        if moves.is_empty() && ruby_dead_end(&mut g) {
            // Ruby itself cannot continue (it raises on the only way on).
            return WalkReport { moves: n, finished: false, error: None, log };
        }
        if moves.is_empty() {
            let snap = g.g1822_snapshot().unwrap_or_default();
            return WalkReport {
                moves: n,
                finished: false,
                error: Some(format!("no legal moves ({players}p seed {seed} move {n}): {snap}")),
                log,
            };
        }
        let m = choose(&moves, &mut rng);
        if let Err(e) = g.g1822_apply_move(&m) {
            return WalkReport {
                moves: n,
                finished: false,
                error: Some(format!("generated move rejected ({players}p seed {seed} move {n}): {:?}: {e}", m.action)),
                log,
            };
        }
        log.push(m);
    }
    WalkReport { moves: max_moves, finished: g.finished, error: None, log }
}

/// A stuck state where Ruby would raise too: every way on fails with an
/// error the engine mirrors from a Ruby crash ("(Ruby raises)"), or the
/// pending London token of an acquired minor 14 the major cannot pay for.
pub fn ruby_dead_end(g: &mut BaseGame) -> bool {
    let cands = g.g1822_candidates_debug();
    !cands.is_empty()
        && cands.iter().any(|(_, e)| e.as_deref().map_or(false, |e| e.contains("(Ruby raises)")))
        || cands.iter().all(|(a, e)| a.starts_with("PlaceToken") && e.as_deref().map_or(false, |e| e.contains("cannot spend")))
            && !cands.is_empty()
}

fn panic_text(p: &Box<dyn std::any::Any + Send>) -> String {
    p.downcast_ref::<String>()
        .cloned()
        .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_else(|| "?".to_string())
}

/// Ad-hoc probe: `G1822_FUZZ="players,seed,moves[,plus]" cargo test ...
/// g1822::fuzz::probe -- --nocapture --ignored` prints progress, timing and
/// the move-type mix every 200 moves.
#[test]
#[ignore]
fn probe() {
    let spec = std::env::var("G1822_FUZZ").unwrap_or_else(|_| "4,1,600".to_string());
    let p: Vec<&str> = spec.split(',').collect();
    let players: usize = p[0].parse().unwrap();
    let seed: u64 = p[1].parse().unwrap();
    let max: usize = p[2].parse().unwrap();
    let rules: Vec<&str> = if p.get(3) == Some(&"plus") { vec!["plus_expansion"] } else { vec![] };
    let mut g = new_game(players, seed, &rules);
    let mut rng = Rng::new(seed ^ 0xA5A5);
    let t0 = std::time::Instant::now();
    let mut gen_time = std::time::Duration::ZERO;
    let mut mix: std::collections::BTreeMap<String, usize> = Default::default();
    for n in 0..max {
        if g.finished {
            eprintln!("finished after {n} moves");
            break;
        }
        let t = std::time::Instant::now();
        let moves = g.g1822_legal_moves();
        gen_time += t.elapsed();
        if moves.is_empty() {
            eprintln!("NO MOVES at {n}: {}", g.g1822_snapshot().unwrap_or_default());
            for (a, e) in g.g1822_candidates_debug() {
                eprintln!("  cand {a} -> {e:?}");
            }
            break;
        }
        let m = choose(&moves, &mut rng);
        *mix.entry(m.action.action_type().to_string()).or_default() += 1;
        if let Err(e) = g.g1822_apply_move(&m) {
            eprintln!("REJECTED at {n}: {:?}: {e}", m.action);
            break;
        }
        if n % 200 == 199 {
            let snap: serde_json::Value = serde_json::from_str(&g.g1822_snapshot().unwrap_or_default()).unwrap_or_default();
            eprintln!(
                "{n}: {:?}s gen {:?}s round {} turn {} phase {} | {:?}",
                t0.elapsed().as_secs_f32(),
                gen_time.as_secs_f32(),
                snap["round"],
                snap["turn"],
                snap["phase"],
                mix
            );
        }
    }
    eprintln!("total {:?}s, gen {:?}s", t0.elapsed().as_secs_f32(), gen_time.as_secs_f32());
    use super::legal::stats;
    eprintln!(
        "cands {} validate {:.2}s lays {:.2}s restrict {:.2}s ({} calls)",
        stats::get(&stats::CANDS),
        stats::get(&stats::VALIDATE_NS) as f64 / 1e9,
        stats::get(&stats::LAYS_NS) as f64 / 1e9,
        stats::get(&stats::RESTRICT_NS) as f64 / 1e9,
        stats::get(&stats::RESTRICT_N)
    );
}

#[test]
fn random_walks_3_to_6_players() {
    let mut failures = Vec::new();
    let mut finished = 0;
    let mut total_moves = 0;
    for players in 3..=6usize {
        for k in 1..=4u64 {
            let seed = k * 7919 + players as u64;
            let rules: &[&str] = if k % 2 == 0 { &["plus_expansion"] } else { &[] };
            // seeds 1-2: the title's cross-player policy; 3-4: the rules'
            let policy = if k <= 2 { CrossPlayerPolicy::RESTRICTED } else { CrossPlayerPolicy::RULES };
            match std::panic::catch_unwind(|| walk(players, seed, rules, policy, 3000)) {
                Ok(rep) => {
                    total_moves += rep.moves;
                    finished += rep.finished as usize;
                    if let Some(e) = rep.error {
                        failures.push(e);
                    }
                }
                Err(p) => failures.push(format!("panic ({players}p seed {seed}): {}", panic_text(&p))),
            }
        }
    }
    eprintln!("1822 fuzz: {total_moves} moves, {finished}/16 games finished");
    assert!(failures.is_empty(), "{} failures:\n{}", failures.len(), failures.join("\n"));
}

/// 1822PNW (3-5 players): the same walks, both purchase policies.
#[test]
fn pnw_random_walks_3_to_5_players() {
    let mut failures = Vec::new();
    let mut finished = 0;
    let mut total_moves = 0;
    for players in 3..=5usize {
        for k in 1..=4u64 {
            let seed = k * 7919 + players as u64;
            let policy = if k <= 2 { CrossPlayerPolicy::RESTRICTED } else { CrossPlayerPolicy::RULES };
            match std::panic::catch_unwind(|| walk_titled("1822PNW", players, seed, &[], policy, 3000)) {
                Ok(rep) => {
                    total_moves += rep.moves;
                    finished += rep.finished as usize;
                    if let Some(e) = rep.error {
                        failures.push(e);
                    }
                }
                Err(p) => failures.push(format!("panic ({players}p seed {seed}): {}", panic_text(&p))),
            }
        }
    }
    eprintln!("1822PNW fuzz: {total_moves} moves, {finished}/12 games finished");
    assert!(failures.is_empty(), "{} failures:\n{}", failures.len(), failures.join("\n"));
}

/// 1822MX (3-5 players; even seeds with higher_cert_limit): the same walks,
/// both purchase policies.
#[test]
fn mx_random_walks_3_to_5_players() {
    let mut failures = Vec::new();
    let mut finished = 0;
    let mut total_moves = 0;
    for players in 3..=5usize {
        for k in 1..=4u64 {
            let seed = k * 7919 + players as u64;
            let rules: &[&str] = if k % 2 == 0 { &["higher_cert_limit"] } else { &[] };
            let policy = if k <= 2 { CrossPlayerPolicy::RESTRICTED } else { CrossPlayerPolicy::RULES };
            match std::panic::catch_unwind(|| walk_titled("1822MX", players, seed, rules, policy, 3000)) {
                Ok(rep) => {
                    total_moves += rep.moves;
                    finished += rep.finished as usize;
                    if let Some(e) = rep.error {
                        failures.push(e);
                    }
                }
                Err(p) => failures.push(format!("panic ({players}p seed {seed}): {}", panic_text(&p))),
            }
        }
    }
    eprintln!("1822MX fuzz: {total_moves} moves, {finished}/12 games finished");
    assert!(failures.is_empty(), "{} failures:\n{}", failures.len(), failures.join("\n"));
}

/// 1822CA (3-6 players), both purchase policies.
#[test]
fn ca_random_walks_3_to_6_players() {
    let mut failures = Vec::new();
    let mut finished = 0;
    let mut total_moves = 0;
    for players in 3..=6usize {
        for k in 1..=4u64 {
            let seed = k * 7919 + players as u64;
            let policy = if k <= 2 { CrossPlayerPolicy::RESTRICTED } else { CrossPlayerPolicy::RULES };
            match std::panic::catch_unwind(|| walk_titled("1822CA", players, seed, &[], policy, 3000)) {
                Ok(rep) => {
                    total_moves += rep.moves;
                    finished += rep.finished as usize;
                    if let Some(e) = rep.error {
                        failures.push(e);
                    }
                }
                Err(p) => failures.push(format!("panic ({players}p seed {seed}): {}", panic_text(&p))),
            }
        }
    }
    eprintln!("1822CA fuzz: {total_moves} moves, {finished}/16 games finished");
    assert!(failures.is_empty(), "{} failures:\n{}", failures.len(), failures.join("\n"));
}

// ---------------------------------------------------------------------------
// Differential walks: the generator against the reference generator
// ---------------------------------------------------------------------------

/// What a differential walk saw.
pub struct DiffReport {
    pub states: usize,
    pub moves_compared: usize,
    pub finished: bool,
    pub error: Option<String>,
}

/// A random walk under `policy` that compares, at every state, the
/// generator with the reference generator ([`super::legal::reference`])
/// under BOTH cross-player policies (`BaseGame::g1822_reference_diff_with`:
/// sorted move lists, RL index rows, the state left as it was).
pub fn diff_walk(
    title: &str,
    players: usize,
    seed: u64,
    rules: &[&str],
    policy: CrossPlayerPolicy,
    max_moves: usize,
) -> DiffReport {
    let mut g = new_game_titled(title, players, seed, rules);
    let mut rng = Rng::new(seed ^ 0xA5A5);
    let (mut states, mut compared) = (0, 0);
    for n in 0..max_moves {
        if g.finished {
            return DiffReport { states, moves_compared: compared, finished: true, error: None };
        }
        states += 1;
        if let Some(d) = g.g1822_graph_check_with() {
            return DiffReport {
                states,
                moves_compared: compared,
                finished: false,
                error: Some(format!("{title} {players}p seed {seed} move {n}: {d}")),
            };
        }
        for p in [CrossPlayerPolicy::RESTRICTED, CrossPlayerPolicy::RULES] {
            let (k, diff) = g.g1822_reference_diff_with(p);
            compared += k;
            if let Some(d) = diff {
                return DiffReport {
                    states,
                    moves_compared: compared,
                    finished: false,
                    error: Some(format!("{title} {players}p seed {seed} move {n} policy {p:?}: {d}")),
                };
            }
        }
        let moves = g.g1822_legal_moves_with(policy);
        if moves.is_empty() {
            let dead = ruby_dead_end(&mut g);
            return DiffReport {
                states,
                moves_compared: compared,
                finished: false,
                error: (!dead).then(|| format!("{title} {players}p seed {seed} move {n}: no legal moves")),
            };
        }
        let m = choose(&moves, &mut rng);
        if let Err(e) = g.g1822_apply_move(&m) {
            // (a price inside a move's range can lead into a Ruby dead end
            // the engine mirrors: Ruby raises there too)
            let dead_end = e.to_string().contains("(Ruby raises)");
            return DiffReport {
                states,
                moves_compared: compared,
                finished: false,
                error: (!dead_end)
                    .then(|| format!("{title} {players}p seed {seed} move {n}: {:?} rejected: {e}", m.action)),
            };
        }
    }
    DiffReport { states, moves_compared: compared, finished: g.finished, error: None }
}

/// The titles' player counts (the RL pipeline's: 1822 / CA 3-6, PNW / MX
/// 3-5).
fn diff_players(title: &str) -> Vec<usize> {
    match title {
        "1822PNW" | "1822MX" => vec![3, 4, 5],
        _ => vec![3, 4, 5, 6],
    }
}

/// The optional rules, alternated over seeds (1822+, MX's higher cert
/// limit).
fn diff_rules(title: &str, seed: u64) -> &'static [&'static str] {
    match (title, seed % 2) {
        ("1822", 0) => &["plus_expansion"],
        ("1822MX", 0) => &["higher_cert_limit"],
        _ => &[],
    }
}

fn run_diff(title: &str, players: &[usize], seeds: std::ops::RangeInclusive<u64>, policy: CrossPlayerPolicy, max: usize) {
    let mut failures = Vec::new();
    let (mut states, mut moves, mut finished, mut walks) = (0, 0, 0, 0);
    for &p in players {
        for seed in seeds.clone() {
            walks += 1;
            let rules = diff_rules(title, seed);
            match std::panic::catch_unwind(|| diff_walk(title, p, seed, rules, policy, max)) {
                Ok(rep) => {
                    states += rep.states;
                    moves += rep.moves_compared;
                    finished += rep.finished as usize;
                    if let Some(e) = rep.error {
                        failures.push(e);
                    }
                }
                Err(e) => failures.push(format!("{title} {p}p seed {seed}: PANIC {}", panic_text(&e))),
            }
        }
    }
    eprintln!(
        "{title} diff walks ({policy:?}): {walks} walks, {finished} finished, {states} states, {moves} moves compared \
         (each state under both policies)"
    );
    assert!(failures.is_empty(), "{} failures:\n{}", failures.len(), failures.join("\n"));
}

/// The generator's move lists equal the reference generator's at every
/// state of three walks per title (4 players under the title's policy, the
/// smallest and largest count under the rules'; both policies compared at
/// every state). The long run (`g1822_diff_long`) covers every count.
#[test]
fn generator_matches_reference_1822() {
    run_diff("1822", &[4], 101..=101, CrossPlayerPolicy::RESTRICTED, 3000);
    run_diff("1822", &[3, 6], 102..=102, CrossPlayerPolicy::RULES, 3000);
}

#[test]
fn generator_matches_reference_1822pnw() {
    run_diff("1822PNW", &[4], 101..=101, CrossPlayerPolicy::RESTRICTED, 3000);
    run_diff("1822PNW", &[3, 5], 102..=102, CrossPlayerPolicy::RULES, 3000);
}

#[test]
fn generator_matches_reference_1822mx() {
    run_diff("1822MX", &[4], 101..=101, CrossPlayerPolicy::RESTRICTED, 3000);
    run_diff("1822MX", &[3, 5], 102..=102, CrossPlayerPolicy::RULES, 3000);
}

#[test]
fn generator_matches_reference_1822ca() {
    run_diff("1822CA", &[4], 101..=101, CrossPlayerPolicy::RESTRICTED, 3000);
    run_diff("1822CA", &[3, 6], 102..=102, CrossPlayerPolicy::RULES, 3000);
}

/// The long differential run: `G1822_DIFF_TITLES` (default all four),
/// `G1822_DIFF_SEEDS` seeds (default 5) from `G1822_DIFF_FIRST` (201) per
/// player count (the rules' policy on seeds + 1000), `G1822_DIFF_POLICY`
/// (`restricted` / `rules`; default both), `G1822_DIFF_PLAYERS` (default
/// the title's), `G1822_DIFF_MOVES` (6000).
/// `cargo test --release --no-default-features g1822_diff_long -- --ignored --nocapture`.
#[test]
#[ignore]
fn g1822_diff_long() {
    let env = |k: &str| std::env::var(k).ok();
    let seeds: u64 = env("G1822_DIFF_SEEDS").and_then(|s| s.parse().ok()).unwrap_or(5);
    let first: u64 = env("G1822_DIFF_FIRST").and_then(|s| s.parse().ok()).unwrap_or(201);
    let max: usize = env("G1822_DIFF_MOVES").and_then(|s| s.parse().ok()).unwrap_or(6000);
    let titles: Vec<String> = env("G1822_DIFF_TITLES")
        .map(|s| s.split(',').map(str::to_string).collect())
        .unwrap_or_else(|| ["1822", "1822PNW", "1822MX", "1822CA"].iter().map(|s| s.to_string()).collect());
    for t in &titles {
        let players: Vec<usize> = env("G1822_DIFF_PLAYERS")
            .map(|s| s.split(',').filter_map(|x| x.parse().ok()).collect())
            .unwrap_or_else(|| diff_players(t));
        let pol = env("G1822_DIFF_POLICY");
        if pol.as_deref() != Some("rules") {
            run_diff(t, &players, first..=first + seeds - 1, CrossPlayerPolicy::RESTRICTED, max);
        }
        if pol.as_deref() != Some("restricted") {
            run_diff(t, &players, first + 1000..=first + 1000 + seeds - 1, CrossPlayerPolicy::RULES, max);
        }
    }
}

/// Bench: what a validation clone costs, with and without the action log,
/// at a late state of a walk (`G1822_CLONE_BENCH="title,players,seed,moves"`).
#[test]
#[ignore]
fn clone_bench() {
    let spec = std::env::var("G1822_CLONE_BENCH").unwrap_or_else(|_| "1822,4,1,1500".to_string());
    let p: Vec<&str> = spec.split(',').collect();
    let (title, players, seed, max): (&str, usize, u64, usize) =
        (p[0], p[1].parse().unwrap(), p[2].parse().unwrap(), p[3].parse().unwrap());
    let mut g = new_game_titled(title, players, seed, &[]);
    let mut rng = Rng::new(seed ^ 0xA5A5);
    for _ in 0..max {
        if g.finished {
            break;
        }
        let moves = g.g1822_legal_moves();
        if moves.is_empty() {
            break;
        }
        let m = choose(&moves, &mut rng);
        let wire = g.g1822_move_wire(&m);
        g.g1822_apply_move(&m).unwrap();
        g.action_log.push(wire);
    }
    let n = 300;
    let t = std::time::Instant::now();
    for _ in 0..n {
        std::hint::black_box(g.clone_for_search());
    }
    let with_log = t.elapsed() / n;
    let t = std::time::Instant::now();
    for _ in 0..n {
        std::hint::black_box(g.clone_for_search_without_log());
    }
    let without = t.elapsed() / n;
    let t = std::time::Instant::now();
    for _ in 0..n {
        std::hint::black_box(g.hexes.clone());
    }
    let hexes = t.elapsed() / n;
    let t = std::time::Instant::now();
    for _ in 0..n {
        std::hint::black_box(g.g1822.clone());
    }
    let st = t.elapsed() / n;
    let t = std::time::Instant::now();
    for _ in 0..n {
        std::hint::black_box(g.g1822_graph_fast(&g.corporations[0].sym.clone()));
    }
    let graph = t.elapsed() / n;
    let t = std::time::Instant::now();
    for _ in 0..n {
        std::hint::black_box(g.g1822_graph_reference(&g.corporations[0].sym.clone()));
    }
    let graph_ref = t.elapsed() / n;
    let t = std::time::Instant::now();
    for _ in 0..n {
        std::hint::black_box(g.g1822_neighbors());
    }
    let nb = t.elapsed() / n;
    eprintln!("graph reference walk {graph_ref:?}");
    eprintln!(
        "{title} {players}p after {} actions: clone {with_log:?}, without log {without:?} (hexes {hexes:?}, g1822 {st:?}); \
         graph {graph:?}, neighbors {nb:?}",
        g.action_log.len()
    );
}

// ---------------------------------------------------------------------------
// Tile instances: the generator offers Ruby's UI's instance
// ---------------------------------------------------------------------------

/// Ruby's supply (`@tiles`) as a list, kept beside a walk: the initial
/// instances (`init_tile`: `<name>-0..` in TILES order, one for an unlimited
/// tile) and `update_tile_lists` at every lay (an unlimited tile adds an
/// instance one past its highest, the laid instance leaves, the old tile --
/// unless preprinted -- goes last). Ruby's UI offers the first instance of a
/// name (`potential_tiles`' `uniq(&:name)`).
struct SupplyModel {
    tiles: Vec<String>,
    unlimited: std::collections::HashSet<String>,
    max_index: HashMap<String, u32>,
}

impl SupplyModel {
    fn new(g: &BaseGame) -> SupplyModel {
        let mut tiles = Vec::new();
        let mut unlimited = std::collections::HashSet::new();
        let mut max_index = HashMap::new();
        for (name, count) in g.title_def().tile_counts() {
            let n = if count == u32::MAX {
                unlimited.insert(name.to_string());
                1
            } else {
                count
            };
            for i in 0..n {
                tiles.push(format!("{name}-{i}"));
            }
            max_index.insert(name.to_string(), n.saturating_sub(1));
        }
        SupplyModel { tiles, unlimited, max_index }
    }

    fn first(&self, name: &str) -> Option<&String> {
        self.tiles.iter().find(|t| t.rsplit_once('-').map_or(false, |(b, _)| b == name))
    }

    /// A lay of `laid` replacing `old` (None: a preprinted tile).
    fn lay(&mut self, laid: &str, old: Option<&str>) {
        let name = laid.rsplit_once('-').map_or(laid, |(b, _)| b).to_string();
        if self.unlimited.contains(&name) {
            let m = self.max_index.entry(name.clone()).or_insert(0);
            *m += 1;
            self.tiles.push(format!("{name}-{m}"));
        }
        self.tiles.retain(|t| t != laid);
        if let Some(o) = old {
            self.tiles.push(o.to_string());
        }
    }
}

/// At every state of random walks (each title, both policies), every tile
/// lay the generator offers uses the instance Ruby's UI would: the first of
/// its name in the independently kept supply list ([`SupplyModel`]).
fn instance_walks(title: &str, players: &[usize], seeds: std::ops::RangeInclusive<u64>, max: usize) -> (usize, usize) {
    let (mut lays_checked, mut changed) = (0, 0);
    for &p in players {
        for seed in seeds.clone() {
            let policy = if seed % 2 == 0 { CrossPlayerPolicy::RULES } else { CrossPlayerPolicy::RESTRICTED };
            let mut g = new_game_titled(title, p, seed, diff_rules(title, seed));
            let mut model = SupplyModel::new(&g);
            let mut rng = Rng::new(seed ^ 0xA5A5);
            for _ in 0..max {
                if g.finished {
                    break;
                }
                let moves = g.g1822_legal_moves_with(policy);
                if moves.is_empty() {
                    break;
                }
                for m in &moves {
                    let crate::actions::Action::LayTile { tile_id, .. } = &m.action else { continue };
                    let name = tile_id.rsplit_once('-').map_or(tile_id.as_str(), |(b, _)| b);
                    if name == "BC" {
                        continue;
                    }
                    let want = model.first(name);
                    assert_eq!(Some(tile_id), want, "{title} {p}p seed {seed}: offered {tile_id}, Ruby's supply {want:?}");
                    lays_checked += 1;
                    // (the rule before 2026-11-14: the lowest-numbered
                    // instance not on the map)
                    let on_map: std::collections::HashSet<&str> = g.hexes.iter().map(|h| h.tile.id.as_str()).collect();
                    let old_rule = (0..).map(|n| format!("{name}-{n}")).find(|id| !on_map.contains(id.as_str()));
                    if old_rule.as_ref() != Some(tile_id) {
                        changed += 1;
                    }
                }
                let m = choose(&moves, &mut rng);
                let lay = match &m.action {
                    crate::actions::Action::LayTile { hex_id, tile_id, .. } if !tile_id.starts_with("BC") => {
                        let hi = g.hex_idx[hex_id.as_str()];
                        let old = &g.hexes[hi].tile;
                        let old_base = old.name.split('-').next().unwrap_or(&old.name).to_string();
                        let returns = !old.id.starts_with("preprinted_") && !old_base.is_empty() && old_base != *hex_id;
                        Some((hi, returns.then(|| old.id.clone())))
                    }
                    _ => None,
                };
                if let Err(e) = g.g1822_apply_move(&m) {
                    panic!("{title} {p}p seed {seed}: {:?} rejected: {e}", m.action);
                }
                if let Some((hi, old)) = lay {
                    let laid = g.hexes[hi].tile.id.clone();
                    model.lay(&laid, old.as_deref());
                }
            }
        }
    }
    (lays_checked, changed)
}

#[test]
fn offered_tile_instance_is_rubys() {
    for (title, players) in [("1822", &[3usize, 5][..]), ("1822PNW", &[3, 4][..]), ("1822MX", &[4][..]), ("1822CA", &[4][..])] {
        let (n, changed) = instance_walks(title, players, 301..=302, 2500);
        eprintln!("{title}: {n} offered lays checked against Ruby's supply order ({changed} where the old rule differs)");
        assert!(n > 0);
    }
}

/// Every generated tile lay applies (`process_action` on a clone accepts
/// it): random walks of `G1822_LAYS_TITLE` (default 1822CA) at 3-6 players,
/// `G1822_LAYS_SEEDS` seeds (default 6) from `G1822_LAYS_FIRST` (1) under
/// each policy; prints every refused lay. A corporation's own lays skip the
/// generator's apply filter (pre-validated), so a refusal is a generator
/// bug. `cargo test --release --no-default-features every_generated_lay_applies -- --ignored --nocapture`.
#[test]
#[ignore]
fn every_generated_lay_applies() {
    let env = |k: &str| std::env::var(k).ok();
    let title = env("G1822_LAYS_TITLE").unwrap_or_else(|| "1822CA".to_string());
    let seeds: u64 = env("G1822_LAYS_SEEDS").and_then(|s| s.parse().ok()).unwrap_or(6);
    let first: u64 = env("G1822_LAYS_FIRST").and_then(|s| s.parse().ok()).unwrap_or(1);
    let players: Vec<usize> = if title == "1822PNW" || title == "1822MX" { vec![3, 4, 5] } else { vec![3, 4, 5, 6] };
    let (mut states, mut lays, mut refused) = (0usize, 0usize, Vec::new());
    for &p in &players {
        for seed in first..first + seeds {
            for policy in [CrossPlayerPolicy::RESTRICTED, CrossPlayerPolicy::RULES] {
                let mut g = new_game_titled(&title, p, seed, diff_rules(&title, seed));
                let mut rng = Rng::new(seed ^ 0x5A5A ^ (p as u64));
                for n in 0..4000 {
                    if g.finished {
                        break;
                    }
                    let moves = g.g1822_legal_moves_with(policy);
                    if moves.is_empty() {
                        break;
                    }
                    states += 1;
                    for m in moves.iter().filter(|m| matches!(m.action, crate::actions::Action::LayTile { .. })) {
                        lays += 1;
                        let mut c = g.clone_for_search();
                        if let Err(e) = c.g1822_apply_move(m) {
                            refused.push(format!("{title} {p}p seed {seed} {policy:?} move {n}: {:?} validated={}: {e}", m.action, m.validated));
                        }
                    }
                    let m = choose(&moves, &mut rng);
                    if g.g1822_apply_move(&m).is_err() {
                        break;
                    }
                }
            }
        }
    }
    eprintln!("{title}: {states} states, {lays} generated lays applied, {} refused", refused.len());
    for r in refused.iter().take(30) {
        eprintln!("  {r}");
    }
    assert!(refused.is_empty());
}

/// The policy-gradient path between the generator and the apply: the RL
/// legal rows (`rl_g1822_indexed_moves`, their wires) and the encoder
/// (`encode_tokens`) leave the state as it was, and every generated lay
/// then applies. Same walks and env as [`every_generated_lay_applies`].
#[test]
#[ignore]
fn rl_path_keeps_state() {
    use crate::rl::generator::GenMove;
    let env = |k: &str| std::env::var(k).ok();
    let title = env("G1822_LAYS_TITLE").unwrap_or_else(|| "1822CA".to_string());
    let seeds: u64 = env("G1822_LAYS_SEEDS").and_then(|s| s.parse().ok()).unwrap_or(6);
    let first: u64 = env("G1822_LAYS_FIRST").and_then(|s| s.parse().ok()).unwrap_or(1);
    let players: Vec<usize> = if title == "1822PNW" || title == "1822MX" { vec![3, 4, 5] } else { vec![3, 4, 5, 6] };
    let (mut states, mut lays, mut bad) = (0usize, 0usize, Vec::new());
    for &p in &players {
        for seed in first..first + seeds {
            let policy = CrossPlayerPolicy::RESTRICTED;
            let mut g = new_game_titled(&title, p, seed, diff_rules(&title, seed));
            let mut rng = Rng::new(seed ^ 0xA5A5 ^ (p as u64));
            for n in 0..4000 {
                if g.finished {
                    break;
                }
                let f0 = g.g1822_state_fingerprint();
                let Ok(rows) = g.rl_g1822_indexed_moves(policy) else { break };
                if rows.is_empty() {
                    break;
                }
                let _wires: Vec<_> = rows.iter().map(|x| Move::wire(&g, &x.mv)).collect();
                let f1 = g.g1822_state_fingerprint();
                let _enc = g.encode_tokens();
                let f2 = g.g1822_state_fingerprint();
                if f1 != f0 {
                    bad.push(format!("{title} {p}p seed {seed} move {n}: the RL legal rows changed the state"));
                }
                if f2 != f1 {
                    bad.push(format!("{title} {p}p seed {seed} move {n}: the encoder changed the state"));
                }
                states += 1;
                for x in rows.iter().filter(|x| matches!(x.mv.action, crate::actions::Action::LayTile { .. })) {
                    lays += 1;
                    let mut c = g.clone_for_search();
                    if let Err(e) = c.g1822_apply_move(&x.mv) {
                        bad.push(format!("{title} {p}p seed {seed} move {n}: {:?}: {e}", x.mv.action));
                    }
                }
                let moves: Vec<Move> = rows.into_iter().map(|x| x.mv).collect();
                let m = choose(&moves, &mut rng);
                if g.g1822_apply_move(&m).is_err() {
                    break;
                }
            }
        }
    }
    eprintln!("{title}: {states} states, {lays} lays, {} problems", bad.len());
    for r in bad.iter().take(30) {
        eprintln!("  {r}");
    }
    assert!(bad.is_empty());
}
