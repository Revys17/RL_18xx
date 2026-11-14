//! In-crate random walks of the 1867 legal-move generator (RL-readiness
//! contract, item 3): at every decision the generator's moves are taken
//! as the legal set; a walk fails on a panic, a generated move that does
//! not apply, or an unfinished state with no moves.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::HashMap;

use crate::game::BaseGame;
use crate::title::CrossPlayerPolicy;

use super::legal::Move;

pub fn new_game(players: usize) -> BaseGame {
    let ids: Vec<u32> = (1..=players as u32).collect();
    let names: HashMap<u32, String> = ids.iter().map(|&i| (i, format!("Player {i}"))).collect();
    BaseGame::build_titled("1867", ids, names)
}

/// One walk's outcome.
#[derive(Debug, Default)]
pub struct WalkReport {
    pub moves: usize,
    pub finished: bool,
    pub error: Option<String>,
    /// Move counts by step kind / action type.
    pub by_type: HashMap<String, usize>,
}

/// Pick a move: non-pass moves favoured; a random price in a range.
pub fn pick(moves: &[Move], rng: &mut StdRng) -> Move {
    let non_pass: Vec<&Move> = moves.iter().filter(|m| m.action.action_type() != "pass").collect();
    let m = if !non_pass.is_empty() && rng.gen::<f64>() < 0.75 {
        non_pass[rng.gen_range(0..non_pass.len())]
    } else {
        &moves[rng.gen_range(0..moves.len())]
    };
    match m.price {
        Some(r) if r.hi > r.lo => {
            let k = (r.hi - r.lo) / r.step.max(1);
            let x = rng.gen::<f64>();
            let p = if x < 0.4 {
                r.lo
            } else if x < 0.5 {
                r.lo + k * r.step
            } else {
                r.lo + rng.gen_range(0..=k.min(40)) * r.step
            };
            m.with_price(p)
        }
        _ => m.clone(),
    }
}

/// A random walk of up to `max_moves` decisions.
pub fn walk(players: usize, seed: u64, max_moves: usize, policy: CrossPlayerPolicy) -> WalkReport {
    let mut g = new_game(players);
    let mut rng = StdRng::seed_from_u64(seed * 1_000_003 + players as u64);
    let mut rep = WalkReport::default();
    let mut recent: Vec<String> = Vec::new();
    for n in 0..max_moves {
        if g.finished {
            rep.finished = true;
            break;
        }
        let moves = g.g1867_legal_moves_with(policy);
        if moves.is_empty() {
            let offers: Vec<String> =
                g.step_offers().iter().map(|o| format!("{:?}/{:?}/{:?}", o.kind, o.entity, o.types)).collect();
            let cands: Vec<String> =
                g.g1867_candidates_debug().into_iter().take(12).map(|(a, e)| format!("{a} -> {e:?}")).collect();
            rep.error = Some(format!(
                "move {n}: no legal moves ({})\n    offers: {offers:?}\n    candidates: {cands:#?}\n    recent: {recent:#?}",
                g.g1867_debug_state()
            ));
            break;
        }
        let m = pick(&moves, &mut rng);
        recent.push(format!("{:?}", m.action));
        if recent.len() > 6 {
            recent.remove(0);
        }
        *rep.by_type.entry(format!("{:?}/{}", m.step, m.action.action_type())).or_insert(0) += 1;
        if let Err(e) = g.g1867_apply_move(&m) {
            rep.error = Some(format!("move {n}: generated move rejected: {:?}: {e}", m.action));
            break;
        }
        rep.moves += 1;
    }
    rep.finished = g.finished;
    rep
}

/// One differential walk's outcome.
#[derive(Debug, Default)]
pub struct DiffReport {
    pub states: usize,
    /// States × policies compared.
    pub compared: usize,
    pub moves_compared: usize,
    pub finished: bool,
    /// The first difference (state, policy, the two lists' symmetric
    /// difference), or a walk error.
    pub error: Option<String>,
}

/// A random walk (as [`walk`], moves of `policy`) that, at EVERY state,
/// compares the generator's move list with the reference generator's
/// (`g1867_legal_moves_reference`: the code before the 2026-11-13
/// speed-ups) under both cross-player policies — step, action, price range
/// and certificates of every move, in order. Stops at the first difference.
pub fn diff_walk(players: usize, seed: u64, max_moves: usize, policy: CrossPlayerPolicy) -> DiffReport {
    let mut g = new_game(players);
    let mut rng = StdRng::seed_from_u64(seed * 1_000_003 + players as u64);
    let mut rep = DiffReport::default();
    for n in 0..max_moves {
        if g.finished {
            break;
        }
        rep.states += 1;
        let mut chosen = None;
        for p in [CrossPlayerPolicy::RESTRICTED, CrossPlayerPolicy::RULES] {
            let fast = g.g1867_legal_moves_with(p);
            let slow = g.g1867_legal_moves_reference(p);
            let (a, b) = (BaseGame::g1867_moves_strings(&fast), BaseGame::g1867_moves_strings(&slow));
            rep.compared += 1;
            rep.moves_compared += a.len();
            if a != b {
                let only_fast: Vec<&String> = a.iter().filter(|x| !b.contains(x)).collect();
                let only_slow: Vec<&String> = b.iter().filter(|x| !a.contains(x)).collect();
                rep.error = Some(format!(
                    "move {n} ({p:?}): {} vs {} moves; only new: {only_fast:#?}; only reference: {only_slow:#?}; {}",
                    a.len(),
                    b.len(),
                    g.g1867_debug_state()
                ));
                return rep;
            }
            if p == policy {
                chosen = Some(fast);
            }
        }
        let moves = chosen.unwrap_or_default();
        if moves.is_empty() {
            rep.error = Some(format!("move {n}: no legal moves ({})", g.g1867_debug_state()));
            return rep;
        }
        let m = pick(&moves, &mut rng);
        if let Err(e) = g.g1867_apply_move(&m) {
            rep.error = Some(format!("move {n}: generated move rejected: {:?}: {e}", m.action));
            return rep;
        }
    }
    rep.finished = g.finished;
    rep
}

impl BaseGame {
    /// One-line state summary for failure messages.
    pub fn g1867_debug_state(&self) -> String {
        format!(
            "round={} phase={} active={:?} round_detail={:?}",
            self.round.round_type_str(),
            self.phase.name,
            self.round_state.active_entity_id,
            match &self.round {
                crate::rounds::Round::Operating(s) => format!("{:?} corp={:?}", s.step, s.current_corp_sym()),
                crate::rounds::Round::Stock(s) => format!(
                    "stock cur={} passed={:?} bought={} parred={} acted={} auction={:?} pending={:?} \
                     cash={:?}",
                    s.current_player_id(),
                    s.player_passed,
                    s.bought_this_turn,
                    s.parred_this_turn,
                    s.acted_this_turn,
                    s.bid_auction.as_ref().map(|a| (&a.corporation_sym, &a.active_bidders, &a.bids)),
                    s.pending_home_tokens,
                    self.players.iter().map(|p| p.cash).collect::<Vec<_>>()
                ),
                crate::rounds::Round::Merger(s) => format!(
                    "merger idx={} ents={:?} conv={:?} converting={} merging={:?} major={}",
                    s.entity_index, s.entities, s.converted, s.converting, s.merging, s.merge_major
                ),
                _ => String::new(),
            }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(players: &[usize], seeds: std::ops::RangeInclusive<u64>, max_moves: usize, policy: CrossPlayerPolicy) {
        let mut failures = Vec::new();
        let mut finished = 0;
        let mut total = 0;
        for &p in players {
            for seed in seeds.clone() {
                let r = std::panic::catch_unwind(|| walk(p, seed, max_moves, policy));
                total += 1;
                match r {
                    Ok(rep) => {
                        if rep.finished {
                            finished += 1;
                        }
                        if let Some(e) = rep.error {
                            failures.push(format!("{p}p seed {seed}: {e}"));
                        }
                    }
                    Err(_) => failures.push(format!("{p}p seed {seed}: PANIC")),
                }
            }
        }
        eprintln!("{total} walks, {finished} finished, {} failing", failures.len());
        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }

    #[test]
    fn random_walks_3_to_6_players() {
        run(&[3, 4, 5, 6], 1..=2, 4000, CrossPlayerPolicy::RESTRICTED);
    }

    /// T6: `check_phase_advance` panics (`unimplemented!`) on an unknown
    /// train event; every event in 1867's train data is one it handles,
    /// so self-play can never reach it.
    #[test]
    fn every_1867_train_event_is_handled() {
        const HANDLED: &[&str] = &[
            "green_minors_available",
            "majors_can_ipo",
            "minors_cannot_start",
            "train_trade_allowed",
            "nationalize_companies",
            "trainless_nationalization",
            "minors_nationalized",
        ];
        for td in crate::title::g1867::trains() {
            for e in td.events {
                assert!(HANDLED.contains(e), "{} event {e} has no handler", td.name);
            }
        }
    }

    /// Profiling target: `perf record <test binary> perf_walk --ignored`.
    #[test]
    #[ignore]
    fn perf_walk() {
        let seeds: u64 = std::env::var("G1867_PERF_SEEDS").ok().and_then(|s| s.parse().ok()).unwrap_or(3);
        for seed in 1..=seeds {
            let t = std::time::Instant::now();
            let r = walk(4, seed, 4000, CrossPlayerPolicy::RESTRICTED);
            eprintln!("seed {seed}: {} moves {:?} finished={} err={:?}", r.moves, t.elapsed(), r.finished, r.error);
        }
        use super::super::legal::stats;
        let ms = |c: &std::sync::atomic::AtomicU64| stats::get(c) / 1_000_000;
        eprintln!(
            "offers {}ms gen {}ms validate {}ms | lay graph {}ms search {}ms check {}ms | cands {} rejected {}",
            ms(&stats::OFFERS_NS),
            ms(&stats::GEN_NS),
            ms(&stats::VALIDATE_NS),
            ms(&stats::LAY_GRAPH_NS),
            ms(&stats::LAY_SEARCH_NS),
            ms(&stats::LAY_CHECK_NS),
            stats::get(&stats::CANDS),
            stats::get(&stats::REJECTED)
        );
    }

    fn run_diff(players: &[usize], seeds: std::ops::RangeInclusive<u64>, max_moves: usize, policy: CrossPlayerPolicy) {
        let mut failures = Vec::new();
        let (mut states, mut moves, mut finished) = (0, 0, 0);
        for &p in players {
            for seed in seeds.clone() {
                match std::panic::catch_unwind(|| diff_walk(p, seed, max_moves, policy)) {
                    Ok(rep) => {
                        states += rep.states;
                        moves += rep.moves_compared;
                        finished += rep.finished as usize;
                        if let Some(e) = rep.error {
                            failures.push(format!("{p}p seed {seed}: {e}"));
                        }
                    }
                    Err(_) => failures.push(format!("{p}p seed {seed}: PANIC")),
                }
            }
        }
        eprintln!("diff walks {policy:?}: {states} states, {moves} moves compared x2 policies, {finished} finished");
        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }

    /// The generator's move lists equal the reference generator's at every
    /// state of a walk (both policies compared at each state; the walk
    /// follows the title's policy, then the rules'), 3-6 players.
    #[test]
    fn generator_matches_reference_on_walks() {
        run_diff(&[3, 4, 5, 6], 11..=11, 4000, CrossPlayerPolicy::RESTRICTED);
        run_diff(&[3, 4, 5, 6], 12..=12, 4000, CrossPlayerPolicy::RULES);
    }

    /// The long differential run: `G1867_DIFF_SEEDS` seeds (default 10) per
    /// player count and policy. `cargo test --release g1867_diff_long --
    /// --ignored`.
    #[test]
    #[ignore]
    fn g1867_diff_long() {
        let seeds: u64 = std::env::var("G1867_DIFF_SEEDS").ok().and_then(|s| s.parse().ok()).unwrap_or(10);
        let first: u64 = std::env::var("G1867_DIFF_FIRST").ok().and_then(|s| s.parse().ok()).unwrap_or(1);
        let players: Vec<usize> = std::env::var("G1867_DIFF_PLAYERS")
            .ok()
            .map(|s| s.split(',').filter_map(|x| x.parse().ok()).collect())
            .unwrap_or_else(|| vec![3, 4, 5, 6]);
        let policy = match std::env::var("G1867_DIFF_POLICY").as_deref() {
            Ok("rules") => CrossPlayerPolicy::RULES,
            _ => CrossPlayerPolicy::RESTRICTED,
        };
        run_diff(&players, first..=first + seeds - 1, 4000, policy);
    }

    #[test]
    fn random_walks_rules_policy() {
        run(&[3, 4, 5, 6], 3..=3, 4000, CrossPlayerPolicy::RULES);
    }
}
