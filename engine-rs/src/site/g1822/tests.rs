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

/// The binding check of the RL session (`check.rs`) on every fixture: the only difference allowed is where the
/// generator fell back to Ruby's set (none on the fixtures).
#[test]
fn menus_equal_the_restricted_set_at_every_fixture_position() {
    let mut r = super::check::Report::default();
    for name in FIXTURES {
        super::check::check_game(name, &fixture(name), &mut r).unwrap();
    }
    eprintln!("1822 menus vs RESTRICTED on the fixtures: {}", r.json());
    assert!(r.menu_only.is_empty() && r.engine_only.is_empty(), "the site's menus and the RESTRICTED set differ");
    assert!(r.foreign.is_empty(), "a fact offers an action the menu's moves don't");
    assert!(r.uncovered.is_empty(), "moves without a fact: {:?}", r.uncovered);
    assert_eq!(r.equal + r.fallback, r.positions);
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

/// The opening position's key: no laid tile, every company in the stacks or the bid boxes, no bid.
#[test]
fn opening_key_is_rubys_start() {
    for title in super::TITLES {
        let data = json!({"id": 5, "title": title, "settings": {"seed": 11}, "players": [{"id": 1, "name": "A"}, {"id": 2, "name": "B"}, {"id": 3, "name": "C"}, {"id": 4, "name": "D"}], "actions": []});
        let mut g = SiteGame::load(&data).unwrap();
        let k = super::key::key(&mut g, None);
        assert_eq!(k["hexes"], json!({}), "{title}");
        assert_eq!(k["round"], "Stock Round");
        assert_eq!(k["step"], "BuySellParShares");
        assert_eq!(k["acting"], json!([1]));
        assert!(k.get("auction").is_none());
        let codes: BTreeSet<String> = k["companies"].as_object().unwrap().values().map(|v| v.as_str().unwrap().to_string()).collect();
        assert_eq!(codes, ["bank".to_string()].into_iter().collect::<BTreeSet<_>>(), "{title}");
        assert_eq!(k["bidboxes"]["minors"].as_array().unwrap().len(), 4);
    }
}

/// Debugging aid: the menus at the positions of a fixture where a fact appears (G1822_FACT, G1822_LOG_FIXTURE).
#[test]
#[ignore]
fn print_fact_examples() {
    let name = std::env::var("G1822_LOG_FIXTURE").unwrap_or_else(|_| "1822_game_end_stock_market".into());
    let fact = std::env::var("G1822_FACT").unwrap_or_else(|_| "acquireMinor".into());
    let data = fixture(&name);
    let mut g = SiteGame::load(&data).unwrap();
    g.site_log = false;
    let mut shown = 0;
    while !g.done() && shown < 3 {
        for pid in super::active_players(&g) {
            let m = super::menu::menu(&mut g, pid);
            if let Some(f) = m.get(&fact) {
                let mut f = f.clone();
                if let Some(o) = f.as_object_mut() {
                    o.remove("moves");
                }
                eprintln!("@{} {}: {}", g.current_action_id, m["description"], serde_json::to_string(&f).unwrap().chars().take(900).collect::<String>());
                shown += 1;
            }
        }
        while !g.done() && g.process_next().is_none() {}
    }
}

/// Three key fields rl18xx-next's Ruby side (`lib/engine_shadow.rb`, step 4 package B) found differing on the
/// fixtures: an unsold minor's shares are its IPO's (Ruby `ipo_shares`: 100, the engine keeps them unowned until
/// issued), 1822CA's P10 token sits in its Winnipeg city as an extra token (Ruby `City#extra_tokens`), and a
/// DiscardTrain step's acting player is the first crowded company's president only (`crowded_corps.take(1)`).
#[test]
fn key_follows_rubys_ipo_extra_tokens_and_discard_turn() {
    let data = json!({"id": 7, "title": "1822", "settings": {"seed": 5}, "players": [{"id": 1, "name": "A"}, {"id": 2, "name": "B"}, {"id": 3, "name": "C"}], "actions": []});
    let mut g = SiteGame::load(&data).unwrap();
    let k = super::key::key(&mut g, None);
    assert_eq!(k["corps"]["1"]["ipo"], json!(100));
    assert_eq!(k["corps"]["1"]["market"], json!(0));

    let at = |name: &str, upto: i64| {
        let data = super::super::cli::truncate(&fixture(name), Some(upto));
        let mut g = SiteGame::load(&data).unwrap();
        g.run_to(None);
        assert!(g.stopped.is_none(), "{name} @{upto}: {:?}", g.stopped);
        let last = data["actions"].as_array().unwrap().last().cloned();
        super::key::key(&mut g, last.as_ref())
    };
    // P10's place_token for GT in W3's city 1 (1822CA_game_end_bank @746)
    let k = at("1822CA_game_end_bank", 746);
    assert_eq!(k["corps"]["GT"]["tokens"], json!(["AC21-1", "N16-1"]));
    // two crowded companies of two presidents: the first one's president acts (1822_discard_trains_after_sr_export @583)
    let k = at("1822_discard_trains_after_sr_export", 583);
    assert_eq!(k["step"], json!("DiscardTrain"));
    assert_eq!(k["acting"], json!([4035]));
}

/// A run of one OO hex's two cities (minor 24's L train on D35, 1822_discard_trains_after_sr_export @44): the router's
/// route lists the hex once, and the stored run has Ruby's `Route#hexes` from the track (RunRoutes#to_h: one entry per
/// stop, rl18xx-next's `h_to_args` for a route with fewer hexes than stops), as the site's Ruby stores it.
#[test]
fn a_run_inside_one_hex_is_stored_with_rubys_hexes() {
    let data = super::super::cli::truncate(&fixture("1822_discard_trains_after_sr_export"), Some(43));
    let mut g = SiteGame::load(&data).unwrap();
    g.run_to(None);
    let ar = super::menu::auto_route(&mut g);
    let routes = ar["action"]["routes"].as_array().cloned().unwrap_or_default();
    let oo = routes.iter().find(|r| r["nodes"].as_array().is_some_and(|n| n.len() == 2) && r["revenue_str"] == json!("D35-D35"));
    let Some(oo) = oo else { panic!("no D35 run in {ar}") };
    let (stored, _) = g.act(&ar["action"]).unwrap();
    let mine = stored["routes"].as_array().unwrap().iter().find(|r| r["train"] == oo["train"]).unwrap().clone();
    assert_eq!(mine["hexes"], json!(["D35", "D35"]));
    assert_eq!(mine["revenue_str"], json!("D35-D35"));
}

/// 1822CA's ICR reaching one of several Québec cities: Ruby's DestinationToken auto action is a PlaceToken with
/// `slot: city.get_slot(entity)`, nil when the city has no free slot, so its stored form names none (1822CA_1 @291).
#[test]
fn icrs_destination_auto_token_names_rubys_slot() {
    let full = fixture("1822CA_1");
    let data = super::super::cli::truncate(&full, Some(290));
    let mut g = SiteGame::load(&data).unwrap();
    g.run_to(None);
    let mut lay = full["actions"].as_array().unwrap().iter().find(|a| a["id"] == json!(291)).unwrap().clone();
    if let Some(m) = lay.as_object_mut() {
        for k in ["id", "auto_actions", "created_at"] {
            m.remove(k);
        }
    }
    let (stored, _) = g.act(&lay).unwrap();
    let autos = stored["auto_actions"].as_array().cloned().unwrap_or_default();
    let token = autos.iter().find(|a| a["type"] == json!("place_token")).unwrap_or_else(|| panic!("no token in {autos:?}"));
    assert_eq!(token["city"], json!("Q1-0-0"));
    assert_eq!(token["token_type"], json!("destination"));
    assert!(token.get("slot").is_none(), "{token}");
}
