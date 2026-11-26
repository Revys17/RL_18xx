//! The site layer's own tests (docs/SITE_ENGINE_RS.md §8): anonymised site-format 1830 games kept beside the crate
//! (`site/fixtures/`: copies of RL_18xx `tests/fixtures/1830/`, and two scratch-stack games of the verification), the
//! expected values from the site's Ruby (the reference key, `scripts/site_engine/ruby/keys_reference.rb`) and Opal
//! (its goldens, living-board test/rust/golden.cjs).

use super::cli;
use crate::entities::EntityId;
use super::game::SiteGame;
use super::shadow;
use serde_json::{json, Value};

fn fixture(name: &str) -> Value {
    let text = match name {
        "router_d_trains" => include_str!("../../site/fixtures/router_d_trains.json"),
        "consent_log_undo" => include_str!("../../site/fixtures/consent_log_undo.json"),
        // scratch-stack games of the step-1 verification (their players renamed), cut short: 18 to action 80, 17 to 246
        "programs_exchange_ca" => include_str!("../../site/fixtures/programs_exchange_ca.json"),
        "dh_teleport_no_route" => include_str!("../../site/fixtures/dh_teleport_no_route.json"),
        _ => panic!("no fixture {name}"),
    };
    serde_json::from_str(text).unwrap()
}

#[test]
fn a_whole_game_replays_and_ends_as_in_ruby() {
    let mut g = SiteGame::load(&fixture("router_d_trains")).unwrap();
    g.run_to(None);
    assert!(g.stopped.is_none(), "{:?}", g.stopped);
    assert!(g.finished());
    assert_eq!(g.end_reason(), Some("bank"));
    assert_eq!(g.result(), vec![(1, 13578), (2, 10741)]);
    let k = shadow::key_now(&mut g);
    assert_eq!(k["v"], 1);
    assert_eq!(k["finished"], true);
    assert_eq!(k["acting"], json!([]));
    assert_eq!(k["entity"], Value::Null);
    assert_eq!(k["step"], Value::Null);
    assert_eq!(k["actions"], json!([]));
    assert_eq!(k["result"], json!({"1": 13578, "2": 10741}));
    assert_eq!(k["manually_ended"], false);
    assert_eq!(k["game_end_reason"], "bank");
    // both on New York's tile once G19 has been upgraded (no longer swapped)
    assert_eq!(k["corps"]["NYNH"]["tokens"], json!(["G19-1", "F22-0"]));
}

#[test]
fn g19_while_preprinted_is_in_rubys_numbering() {
    // NYNH's home token on preprinted G19: Ruby's city 0, the engine's city 1 (never renumbered: the policy layout)
    let data = cli::truncate(&fixture("router_d_trains"), Some(34));
    let mut g = SiteGame::load(&data).unwrap();
    g.run_to(None);
    let k = shadow::key_now(&mut g);
    assert_eq!(k["corps"]["NYNH"]["tokens"], json!(["G19-0"]));
    assert!(k["hexes"].get("G19").is_none());
    let h = g.game.hex_ref("G19").unwrap();
    let engine_city = h.tile.cities.iter().position(|c| c.tokens.iter().flatten().any(|t| t.corporation_id == "NYNH"));
    assert_eq!(engine_city, Some(1));
}

#[test]
fn undo_and_programs_replay_and_the_turn_is_rubys() {
    let mut g = SiteGame::load(&fixture("consent_log_undo")).unwrap();
    g.run_to(None);
    assert!(g.stopped.is_none(), "{:?}", g.stopped);
    let k = shadow::key_now(&mut g);
    assert_eq!(k["round"], "Stock Round");
    assert_eq!(k["acting"], json!([3]));
    assert_eq!(k["step"], "BuySellParShares");
}

#[test]
fn every_stored_run_is_priced_as_stored() {
    let data = fixture("router_d_trains");
    let (code, out) = cli::keys(&json!({"game": data}));
    assert_eq!(code, cli::EXIT_OK);
    let by_id: std::collections::HashMap<i64, &Value> =
        data["actions"].as_array().unwrap().iter().map(|a| (a["id"].as_i64().unwrap(), a)).collect();
    let mut runs = 0;
    for p in out["positions"].as_array().unwrap() {
        let la = &p["state"]["last_action"];
        if la.get("revenues").is_some() {
            let a = by_id[&p["actionId"].as_i64().unwrap()];
            let stored: Vec<Value> = a["routes"].as_array().unwrap().iter().map(|r| r["revenue"].clone()).collect();
            assert_eq!(la["revenues"], json!(stored), "action {}", p["actionId"]);
            runs += 1;
        }
    }
    assert!(runs > 50, "{runs}");
}

#[test]
fn the_cli_keeps_its_contract() {
    let (code, v) = cli::run("version", "");
    assert_eq!(code, 0);
    assert_eq!((v["api"].clone(), v["key"].clone(), v["engine"].clone()), (json!(2), json!(1), json!("engine-rs")));
    assert_eq!(cli::run("state", "not json").0, cli::EXIT_BAD_INPUT);
    assert_eq!(cli::run("state", "{}").0, cli::EXIT_BAD_INPUT);
    assert_eq!(cli::run("nope", "{}").0, cli::EXIT_BAD_INPUT);
    let other = json!({"game": {"title": "1889", "players": [{"id": 1}, {"id": 2}], "actions": []}});
    assert_eq!(cli::run("state", &other.to_string()).0, cli::EXIT_UNSUPPORTED);
    let data = fixture("consent_log_undo");
    let auto = json!({"game": data, "action": {"type": "pass", "entity": 3, "entity_type": "player"}, "add_auto_actions": true});
    assert_eq!(cli::act(&auto).0, cli::EXIT_UNSUPPORTED);
    // a move of the player to act: stored with the next id
    let (code, out) = cli::act(&json!({"game": data, "action": {"type": "pass", "entity": 3, "entity_type": "player"}}));
    assert_eq!(code, 0);
    let last = data["actions"].as_array().unwrap().last().unwrap()["id"].as_i64().unwrap();
    assert_eq!(out["action"]["id"], last + 1);
    assert_eq!(out["error"], Value::Null);
    // someone else's: refused, in the engine's words, the position unchanged
    let (code, out) = cli::act(&json!({"game": data, "action": {"type": "pass", "entity": 1, "entity_type": "player"}}));
    assert_eq!(code, 0);
    assert_eq!(out["action"], Value::Null);
    assert!(out["error"].as_str().unwrap().starts_with("refused:"), "{}", out["error"]);
    assert_eq!(out["state"]["acting"], json!([3]));
    // `upto` cuts the raw actions before undo/redo are resolved
    let (_, s) = cli::state(&json!({"game": fixture("router_d_trains"), "upto": 34}));
    assert_eq!(s["state"]["corps"]["NYNH"]["tokens"], json!(["G19-0"]));
    assert_eq!(s["stopped"], Value::Null);
}

#[test]
fn replay_digests_are_stable() {
    let data = fixture("router_d_trains");
    let a = shadow::replay_digest(&data).unwrap();
    let b = shadow::replay_digest(&data).unwrap();
    assert_eq!(a, b);
    assert_eq!(a["steps"], 763);
}

#[test]
fn site_rules_only_in_site_games() {
    let names = (1..=4u32).map(|i| (i, format!("P{i}"))).collect();
    let g = crate::game::BaseGame::build_with_rules("1830", vec![1, 2, 3, 4], names, vec![]);
    assert!(!g.site_rules && g.site.is_none());
    assert!(!g.clone_for_search().site_rules);
    let s = SiteGame::load(&fixture("consent_log_undo")).unwrap();
    assert!(s.game.site_rules && s.game.site.is_some());
}

fn step_at(g: &mut SiteGame, id: i64) -> super::game::StepOut {
    g.run_to(Some(id - 1));
    let step = g.process_next().expect("a step");
    assert_eq!(step.action_id, id);
    step
}

#[test]
fn a_stock_round_ending_through_an_empty_operating_round_drops_its_programs_and_logs_the_round() {
    // 18@68: an auto pass ends SR 2; OR 2.1 has no company to operate (only the privates pay); SR 3 begins. Ruby runs
    // check_programmed_actions at each transition: player 14's "for this stock round" pass (set at 40) ends in OR 2.1
    let mut g = SiteGame::load(&fixture("programs_exchange_ca")).unwrap();
    g.run_to(Some(67));
    assert!(g.programs.contains_key(&14));
    let step = g.process_next().unwrap();
    assert_eq!(step.action_id, 68);
    assert!(g.programs.is_empty(), "{:?}", g.programs);
    assert_eq!(
        step.log,
        vec![
            "Player 4 passes",
            "Player 1 has priority deal",
            "-- Operating Round 2.1 (of 1) --",
            "Player 1 collects $5 from Schuylkill Valley",
            "Player 1 collects $25 from Camden & Amboy",
            "Player 3 collects $10 from Champlain & St.Lawrence",
            "Player 2 collects $30 from Baltimore & Ohio",
            "Player 4 collects $15 from Delaware & Hudson",
            "-- Stock Round 3 --",
        ]
    );
    // a program set in the new stock round stays (70: player 13's, its auto pass leaves the round going)
    g.run_to(Some(70));
    assert_eq!(g.programs.keys().copied().collect::<Vec<_>>(), vec![13]);
}

#[test]
fn one_program_per_player() {
    // ALLOW_MULTIPLE_PROGRAMS false: enabling another program replaces the player's one
    let mut g = SiteGame::load(&fixture("programs_exchange_ca")).unwrap();
    g.run_to(Some(70));
    g.apply(&json!({"type": "program_buy_shares", "entity": 13, "entity_type": "player", "corporation": "PRR", "until_condition": "float", "from_market": false})).unwrap();
    let mine: Vec<&str> = g.programs[&13].iter().map(|p| p["type"].as_str().unwrap()).collect();
    assert_eq!(mine, vec!["program_buy_shares"]);
    g.apply(&json!({"type": "program_disable", "entity": 13, "entity_type": "player", "reason": "user"})).unwrap();
    assert!(!g.programs.contains_key(&13));
}

#[test]
fn m_and_h_exchange_starts_no_turn() {
    // 18@23: player 15 exchanges M&H at the start of their turn; Ruby's turn began at 22 (Opal: 22 / 21)
    let mut g = SiteGame::load(&fixture("programs_exchange_ca")).unwrap();
    let step = step_at(&mut g, 23);
    assert_eq!(step.log, vec!["Player 3 exchanges Mohawk & Hudson for a 10% share of NYC from the IPO"]);
    let r = super::details::restart_info(&g);
    assert_eq!((r["turnStart"].clone(), r["lastTurnStart"].clone()), (json!(22), json!(21)));
    // the turn's own actions are BuySellParShares's: the player's later pass is "passes" (Ruby log_pass)
    let step = step_at(&mut g, 24);
    assert_eq!(step.log.first().map(String::as_str), Some("Player 3 buys a 10% share of C&O from the IPO for $67"));
}

#[test]
fn camden_and_amboy_between_players_moves_its_prr_share_as_ruby() {
    // after 18@24: stock round 2, player 14 to act, C&A held by player 13 with PRR_1 (its share from the auction)
    let mut g = SiteGame::load(&fixture("programs_exchange_ca")).unwrap();
    g.run_to(Some(24));
    let buy = json!({"type": "buy_company", "entity": 14, "entity_type": "player", "company": "CA", "price": 1});
    let cash = |g: &SiteGame, id: u32| g.game.players.iter().find(|p| p.id == id).unwrap().cash;
    let (c13, c14) = (cash(&g, 13), cash(&g, 14));
    // a player holds PRR_1: Ruby's after_buy_company raises 'Cannot buy share from player'; nothing moves
    let err = g.apply_engine(&buy).unwrap_err();
    assert!(err.message.contains("Cannot buy share from player"), "{}", err.message);
    assert_eq!((cash(&g, 13), cash(&g, 14)), (c13, c14));
    assert_eq!(g.game.companies[g.game.company_idx["CA"]].owner, EntityId::player(13));
    // PRR_1 in the market: the buyer takes it for nothing
    let prr = g.game.corp_idx["PRR"];
    let si = g.game.corporations[prr].shares.iter().position(|s| s.index == 1).unwrap();
    g.game.corporations[prr].set_share_owner(si, EntityId::market());
    let before = g.game.corporations[prr].percent_owned_by(&EntityId::player(14));
    g.apply_engine(&buy).unwrap();
    assert_eq!(g.game.corporations[prr].shares[si].owner, EntityId::player(14));
    assert_eq!(g.game.corporations[prr].percent_owned_by(&EntityId::player(14)), before + 10);
    assert_eq!(g.game.companies[g.game.company_idx["CA"]].owner, EntityId::player(14));
    assert_eq!((cash(&g, 13), cash(&g, 14)), (c13 + 1, c14 - 1));
    let log = g.take_log();
    assert!(log.ends_with(&["-- Player 2 buys Camden & Amboy from Player 1 for $1".to_string(), "Player 2 receives a 10% share of PRR".to_string()]), "{log:?}");
}

#[test]
fn d_and_h_teleport_without_a_route_withholds_on_the_record() {
    // 17@246: CPR (trains, no route) places D&H's teleport token; its run is skipped and the dividend withheld
    // (Ruby Dividend#skip!: operating_history[[4, 1]] a withhold with no routes)
    let mut g = SiteGame::load(&fixture("dh_teleport_no_route")).unwrap();
    let step = step_at(&mut g, 246);
    assert_eq!(super::view::last_run(&g, "CPR"), json!({"revenue": 0, "kind": "no_run", "or": "4.1"}));
    assert_eq!(
        step.log,
        vec![
            "CPR (DH) places a token on F16 (Scranton) ",
            "CPR skips lay track",
            "CPR skips place a token",
            "CPR skips run routes",
            "CPR does not run",
            "CPR's share price moves left from $30 to $20",
        ]
    );
}

