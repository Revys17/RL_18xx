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
