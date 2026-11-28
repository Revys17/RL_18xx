//! The 1822 family's site layer on the site's fixtures (rl18xx-next `public/fixtures/{1822,1822CA}`, vendored in
//! `site/fixtures/g1822/`): replays, results, keys, views, menus against the generator's RESTRICTED set, the
//! authority's verdicts on the recorded moves, the seat map, the phase statuses (docs/SITE_ENGINE_RS_1822.md §4.5) and
//! a pinned 1822CA walk.

use super::super::game::SiteGame;
use crate::title::CrossPlayerPolicy;
use serde_json::{json, Value};
use std::collections::BTreeSet;

const FIXTURES: &[&str] = &[
    "1822_game_end_stock_market",
    "1822_33845",
    "1822_discard_trains_after_sr_export",
    "1822_hs_lxemeslq_30797",
    "1822CA_1",
    "1822CA_2",
    "1822CA_3",
    "1822CA_4",
    "1822CA_game_end_bank",
    "1822CA_game_end_stock_market",
    "1822CA_loan_payments_preserve_bank",
];

fn fixture(name: &str) -> Value {
    let p = format!("{}/site/fixtures/g1822/{name}.json", env!("CARGO_MANIFEST_DIR"));
    serde_json::from_str(&std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{p}: {e}"))).unwrap()
}

#[test]
fn fixtures_replay_to_their_results() {
    for name in FIXTURES {
        let data = fixture(name);
        let mut g = SiteGame::load(&data).unwrap_or_else(|e| panic!("{name}: {e}"));
        g.run_to(None);
        assert!(g.stopped.is_none(), "{name}: stopped at {:?}", g.stopped);
        assert!(g.finished(), "{name}: not finished");
        let want: Vec<(String, i64)> = data["result"].as_object().unwrap().iter().map(|(k, v)| (k.clone(), v.as_i64().unwrap())).collect();
        let mut got: Vec<(String, i64)> = g.result().into_iter().map(|(k, v)| (k.to_string(), v as i64)).collect();
        let mut want = want;
        want.sort();
        got.sort();
        assert_eq!(got, want, "{name}: result");
    }
}

/// A move's site action as a comparable string.
fn canon(v: &Value) -> String {
    let mut v = v.clone();
    if let Some(o) = v.as_object_mut() {
        o.remove("created_at");
        o.remove("user");
    }
    v.to_string()
}

/// The actions every fact of a menu names (outside the flat `moves` list).
fn fact_actions(v: &Value, out: &mut BTreeSet<String>, top: bool) {
    match v {
        Value::Object(m) => {
            if let Some(a) = m.get("action").filter(|a| a.get("type").is_some()) {
                out.insert(canon(a));
            }
            for (k, x) in m {
                if top && k == "moves" {
                    continue;
                }
                if k != "action" {
                    fact_actions(x, out, false);
                }
            }
        }
        Value::Array(xs) => xs.iter().for_each(|x| fact_actions(x, out, false)),
        _ => {}
    }
}

/// The binding check of the RL session (docs/SITE_ENGINE_RS_1822.md §3.4): at every position of every fixture, the
/// site's menus (the players to act, own-only, no fallback) against engine-rs's RESTRICTED legal set, both directions;
/// the only difference allowed is where the generator fell back to Ruby's set because the restricted one was empty
/// (`CROSS_FALLBACK`). Also: every fact's action is one of the menu's moves and every move has a fact; the key, the
/// snapshot and the bot view build everywhere.
#[test]
fn menus_equal_the_restricted_set_at_every_fixture_position() {
    let mut positions = 0usize;
    let mut equal = 0usize;
    let mut fallback = 0usize;
    let mut menu_only: Vec<String> = Vec::new();
    let mut engine_only: Vec<String> = Vec::new();
    let mut uncovered: std::collections::BTreeMap<String, usize> = Default::default();
    let mut foreign: Vec<String> = Vec::new();
    for name in FIXTURES {
        let data = fixture(name);
        let mut g = SiteGame::load(&data).unwrap();
        g.site_log = false;
        loop {
            if !g.finished() {
                positions += 1;
                // the engine's RESTRICTED set (its fallback marked)
                let before = crate::g1822::legal::stats::get(&crate::g1822::legal::stats::CROSS_FALLBACK);
                let moves = g.game.g1822_legal_moves_with(CrossPlayerPolicy::RESTRICTED);
                let fell_back = crate::g1822::legal::stats::get(&crate::g1822::legal::stats::CROSS_FALLBACK) > before;
                let engine: BTreeSet<String> = moves.iter().map(|m| canon(&super::to_site(&g, &g.game.g1822_move_wire(m)))).collect();
                let mut site: BTreeSet<String> = BTreeSet::new();
                for pid in super::active_players(&g) {
                    let m = super::menu::menu(&mut g, pid);
                    let listed: BTreeSet<String> =
                        m.get("moves").and_then(Value::as_array).map(|v| v.iter().map(|x| canon(&x["action"])).collect()).unwrap_or_default();
                    let mut facts = BTreeSet::new();
                    fact_actions(&m, &mut facts, true);
                    for f in facts.difference(&listed) {
                        foreign.push(format!("{name} @{}: {f}", g.current_action_id));
                    }
                    let mut lay_hexes: BTreeSet<String> = BTreeSet::new();
                    for h in m.get("tileHexes").and_then(Value::as_array).into_iter().flatten() {
                        lay_hexes.extend(h.as_str().map(str::to_string));
                    }
                    for a in m.get("abilities").and_then(Value::as_array).into_iter().flatten() {
                        for h in a["hexes"].as_array().into_iter().flatten() {
                            lay_hexes.extend(h.as_str().map(str::to_string));
                        }
                    }
                    for x in m.get("moves").and_then(Value::as_array).cloned().unwrap_or_default() {
                        // (a lay: its hex is in `tileHexes` / a private's `abilities`, the tiles come from tileOptions; a run:
                        // `routes.pending`, the run from autoRoute)
                        let covered = match x["type"].as_str() {
                            Some("lay_tile") => x["action"]["hex"].as_str().is_some_and(|h| lay_hexes.contains(h)),
                            Some("run_routes") => m["routes"]["pending"] == json!(true),
                            _ => false,
                        };
                        if !covered && !facts.contains(&canon(&x["action"])) {
                            *uncovered.entry(x["type"].as_str().unwrap_or("").to_string()).or_default() += 1;
                        }
                    }
                    site.extend(listed);
                }
                if site == engine {
                    equal += 1;
                } else if fell_back {
                    fallback += 1;
                } else {
                    for x in site.difference(&engine).take(3) {
                        menu_only.push(format!("{name} @{}: {x}", g.current_action_id));
                    }
                    for x in engine.difference(&site).take(3) {
                        engine_only.push(format!("{name} @{}: {x}", g.current_action_id));
                    }
                }
                let _ = super::key::key(&mut g, None);
                let _ = super::view::snapshot(&g, g.current_action_id);
                let _ = super::super::authority::bot_view(&mut g);
            }
            if g.done() {
                break;
            }
            while !g.done() && g.process_next().is_none() {}
        }
    }
    eprintln!(
        "1822 menus vs RESTRICTED: {positions} positions, {equal} equal, {fallback} where the generator fell back to Ruby's set, \
         {} menu-only and {} engine-only moves; facts outside the moves {}; moves without a fact by type {:?}",
        menu_only.len(),
        engine_only.len(),
        foreign.len(),
        uncovered
    );
    for x in menu_only.iter().chain(engine_only.iter()).chain(foreign.iter()).take(20) {
        eprintln!("  {x}");
    }
    assert!(menu_only.is_empty() && engine_only.is_empty(), "the site's menus and the RESTRICTED set differ");
    assert!(foreign.is_empty(), "a fact offers an action the menu's moves don't");
    assert!(uncovered.is_empty(), "moves without a fact: {uncovered:?}");
    assert_eq!(equal + fallback, positions);
}

/// Every recorded move decided at its position as `site_engine move` decides it: all accepted but the documented
/// classes (runs other than the router's, out-of-order discards: docs/SITE_ENGINE_RS_1822.md §2.3 G3).
#[test]
fn recorded_moves_pass_the_authority() {
    let (mut moves, mut accepted, mut runs_refused, mut discards) = (0u64, 0u64, 0u64, 0u64);
    let (mut autos, mut autos_eq) = (0u64, 0u64);
    for name in FIXTURES {
        let data = fixture(name);
        let mut g = SiteGame::load(&data).unwrap();
        g.site_log = false;
        let r = super::super::authority::corpus_check(&mut g);
        moves += r["moves"].as_u64().unwrap();
        accepted += r["accepted"].as_u64().unwrap();
        autos += r["autos"]["checked"].as_u64().unwrap();
        autos_eq += r["autos"]["equal"].as_u64().unwrap();
        for x in r["refused"].as_array().unwrap() {
            match x["action"]["type"].as_str() {
                Some("run_routes") => runs_refused += 1,
                Some("discard_train") => discards += 1,
                _ => panic!("{name}: refused {x}"),
            }
        }
    }
    eprintln!("1822 authority on the fixtures: {moves} moves, {accepted} accepted, {runs_refused} runs not the router's, {discards} out-of-order discards; auto actions {autos_eq} / {autos} equal");
    assert_eq!(discards, 2);
}

/// Site ids that are minors' syms (2, 3, 4) seat at 1002-1004 in the engine and come back as the site's everywhere.
#[test]
fn site_ids_colliding_with_minors_are_seated() {
    let data = json!({"id": 9001, "title": "1822", "settings": {"seed": 1, "optional_rules": []},
                      "players": [{"id": 2, "name": "A"}, {"id": 3, "name": "B"}, {"id": 4, "name": "C"}], "actions": []});
    let mut g = SiteGame::load(&data).unwrap();
    g.run_to(None);
    assert_eq!(g.game.players.iter().map(|p| p.id).collect::<Vec<_>>(), vec![1002, 1003, 1004]);
    let acting = super::active_players(&g);
    assert_eq!(acting, vec![2]);
    let m = super::menu::menu(&mut g, 2);
    assert_eq!(m["actor"]["id"], json!(2));
    let bid = m["bidBoxes"]["boxes"][0]["lots"][0]["action"].clone();
    assert_eq!(bid["entity"], json!(2));
    let (posted, _) = g.act(&bid).unwrap();
    assert_eq!(posted["entity"], json!(2));
    let key = super::key::key(&mut g, None);
    assert!(key["players"].as_array().unwrap().iter().all(|p| [2, 3, 4].contains(&p["id"].as_u64().unwrap())));
    assert_eq!(key["auction"]["bids"][0][1], json!(2));
    // a player's pass after the bid moves the turn on
    let pass = json!({"type": "pass", "entity": 2, "entity_type": "player"});
    g.act(&pass).unwrap();
    assert_eq!(super::active_players(&g), vec![3]);
}

/// docs/SITE_ENGINE_RS_1822.md §4.5: the family's phase statuses are Ruby's (no site table rows) and none opens
/// purchases between players.
#[test]
fn phase_statuses_are_rubys_and_close_interplayer_purchases() {
    for title in super::TITLES {
        let data = json!({"id": 1, "title": title, "settings": {"seed": 3}, "players": [{"id": 1, "name": "A"}, {"id": 2, "name": "B"}, {"id": 3, "name": "C"}], "actions": []});
        let mut g = SiteGame::load(&data).unwrap();
        for p in g.game.title_def().phases() {
            g.game.phase.name = p.name.to_string();
            assert_eq!(g.game.ruby_phase_status(), p.status.to_vec(), "{title} phase {}", p.name);
            assert!(!p.status.contains(&"can_buy_companies_from_other_players"));
            for pl in g.game.players.clone() {
                assert!(!g.game.site_interplayer_purchase_open(pl.id, false), "{title} phase {}", p.name);
            }
        }
    }
}

/// FNV-1a 64.
struct Fnv(u64);
impl Fnv {
    fn str(&mut self, s: &str) {
        for &x in (s.len() as u64).to_le_bytes().iter().chain(s.as_bytes()) {
            self.0 = (self.0 ^ x as u64).wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
}

/// A fixed-seed 1822CA walk through the generator (`policy`), every position's moves and the router's run hashed.
fn walk_ca(players: u32, seed: u64, policy: CrossPlayerPolicy, max_moves: usize) -> (u64, usize, bool) {
    let ids: Vec<u32> = (1..=players).map(|i| 1000 + i).collect();
    let names = ids.iter().map(|&i| (i, format!("Player {}", i - 1000))).collect();
    let mut g = crate::game::BaseGame::build_seeded("1822CA", ids, names, Vec::new(), Some(seed));
    let mut h = Fnv(0xcbf2_9ce4_8422_2325);
    let mut rng = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    let mut n = 0;
    while !g.finished && n < max_moves {
        let moves = g.g1822_legal_moves_with(policy);
        if moves.is_empty() {
            break;
        }
        for m in &moves {
            h.str(&crate::game::BaseGame::g1822_move_string(m));
        }
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        // (prefer anything but a pass, so the walk goes somewhere)
        let busy: Vec<usize> = (0..moves.len()).filter(|&i| moves[i].action.action_type() != "pass").collect();
        let pool = if busy.is_empty() || (rng >> 33) % 5 == 0 { (0..moves.len()).collect::<Vec<_>>() } else { busy };
        let pick = &moves[pool[((rng >> 33) as usize) % pool.len()]];
        let mut mv = pick.clone();
        if let Some(r) = pick.price {
            let k = ((rng >> 17) as i32).rem_euclid(((r.hi - r.lo) / r.step.max(1)) + 1);
            mv = pick.with_price(r.lo + k * r.step.max(1));
        }
        g.g1822_apply_move(&mv).unwrap_or_else(|e| panic!("1822CA {players}p seed {seed}: {}", e.message));
        n += 1;
    }
    h.str(&format!("{}|{}|{}", g.finished, g.phase.name, g.bank.cash));
    (h.0, n, g.finished)
}

/// 1822CA walks pinned before the P5 edit outside src/site (docs/SITE_ENGINE_RS_1822.md §4.4: the timing calls of the
/// generator and router, which 1822CA shares): with site_rules off nothing of the site layer may move them.
/// Re-pin (and say so) only for an intended change of the 1822CA generator or router:
///   cargo test --release --no-default-features --features site ca_walks_are_pinned -- --nocapture
#[test]
fn ca_walks_are_pinned() {
    let cases: &[(u32, u64, bool)] = &[(3, 1822, false), (4, 1823, false), (5, 1824, true), (6, 1825, false)];
    let mut got = Vec::new();
    for &(p, seed, rules) in cases {
        let policy = if rules { CrossPlayerPolicy::RULES } else { CrossPlayerPolicy::RESTRICTED };
        let (d, n, fin) = walk_ca(p, seed, policy, 700);
        eprintln!("1822CA walk {p}p seed {seed} rules {rules}: {d:#018x} ({n} moves, finished {fin})");
        got.push(d);
    }
    let pinned: &[u64] = PINNED_CA;
    if pinned.is_empty() {
        return;
    }
    assert_eq!(got, pinned, "1822CA walks changed");
}

/// The digests of `ca_walks_are_pinned` (computed on site/engine-rs-1822 before P5, fbad02a's parent).
const PINNED_CA: &[u64] = &[0xf3992ff745f9075c, 0xc645efc85ef47ac4, 0x160839b73374634a, 0x486dd74f0ab54232];

/// Debugging aid: a fixture's whole log, one step per line group (G1822_LOG_FIXTURE, G1822_LOG_OUT).
#[test]
#[ignore]
fn print_fixture_log() {
    let name = std::env::var("G1822_LOG_FIXTURE").unwrap_or_else(|_| "1822_game_end_stock_market".into());
    let out = std::env::var("G1822_LOG_OUT").unwrap_or_else(|_| "/dev/stdout".into());
    let data = fixture(&name);
    let mut g = SiteGame::load(&data).unwrap();
    let mut text: Vec<String> = g.start_log.clone();
    while !g.done() {
        if let Some(s) = g.process_next() {
            text.push(format!("# {} {}", s.action_id, s.action["type"]));
            text.extend(s.log);
        }
    }
    std::fs::write(out, text.join("\n")).unwrap();
}
