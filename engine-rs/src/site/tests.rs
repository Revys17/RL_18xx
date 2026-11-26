//! The site layer's own tests (docs/SITE_ENGINE_RS.md §8): two anonymised site-format 1830 games kept beside the
//! crate (`site/fixtures/`, copies of RL_18xx `tests/fixtures/1830/`), the expected values from the site's Ruby (the
//! reference key, `scripts/site_engine/ruby/keys_reference.rb`).

use super::cli;
use super::game::SiteGame;
use super::shadow;
use serde_json::{json, Value};

fn fixture(name: &str) -> Value {
    let text = match name {
        "router_d_trains" => include_str!("../../site/fixtures/router_d_trains.json"),
        "consent_log_undo" => include_str!("../../site/fixtures/consent_log_undo.json"),
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
