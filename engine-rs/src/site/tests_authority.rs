//! Step 3a's tests (docs/SITE_ENGINE_RS_STEP3.md §12.1): `move` / `start` / `view` (K3'), the 1830 move check, the
//! route check, the Programmer, on the site fixtures.

use super::authority;
use super::cli;
use super::game::SiteGame;
use serde_json::{json, Value};

fn fixture(name: &str) -> Value {
    let text = match name {
        "router_d_trains" => include_str!("../../site/fixtures/router_d_trains.json"),
        "consent_log_undo" => include_str!("../../site/fixtures/consent_log_undo.json"),
        "programs_exchange_ca" => include_str!("../../site/fixtures/programs_exchange_ca.json"),
        "dh_teleport_no_route" => include_str!("../../site/fixtures/dh_teleport_no_route.json"),
        "bankrupt_voluntary" => include_str!("../../site/fixtures/bankrupt_voluntary.json"),
        "bank_broken" => include_str!("../../site/fixtures/bank_broken.json"),
        "g1867_21268" => include_str!("../../site/fixtures/g1867_21268.json"),
        "g1867_nationalization_cash" => include_str!("../../site/fixtures/g1867_nationalization_cash.json"),
        _ => panic!("no fixture {name}"),
    };
    serde_json::from_str(text).unwrap()
}

/// The game with its raw actions before action `id`, replayed (the position the server holds when action `id` is
/// posted). By position: router_d_trains, an 18xx.games export, numbers one early action 764.
fn before(data: &Value, id: i64) -> SiteGame {
    let mut d = data.clone();
    let list = d["actions"].as_array().unwrap().clone();
    let at = list.iter().position(|a| a["id"] == json!(id)).unwrap();
    d["actions"] = json!(list[..at].to_vec());
    let mut g = SiteGame::load(&d).unwrap();
    g.site_log = false;
    g.run_to(None);
    assert!(g.stopped.is_none());
    g
}

fn action(data: &Value, id: i64) -> Value {
    data["actions"].as_array().unwrap().iter().find(|a| a["id"] == json!(id)).cloned().unwrap()
}

fn decide(g: &SiteGame, a: &Value) -> Result<authority::Decision, String> {
    let mut c = g.clone();
    authority::decide(&mut c, a, a.get("user").and_then(Value::as_u64), 1_800_000_000)
}

#[test]
fn every_fixture_move_is_decided_as_stored() {
    for name in [
        "router_d_trains", "consent_log_undo", "programs_exchange_ca", "dh_teleport_no_route", "bankrupt_voluntary", "bank_broken",
        "g1867_21268", "g1867_nationalization_cash",
    ] {
        let mut g = SiteGame::load(&fixture(name)).unwrap();
        g.site_log = false;
        let r = authority::corpus_check(&mut g);
        assert_eq!(r["stopped"], Value::Null, "{name}");
        assert_eq!(r["refused"], json!([]), "{name}: {}", r["refused"]);
        assert_eq!(r["moves"], r["accepted"], "{name}");
        // (the 18xx.games exports, router_d_trains and the g1867 ones, carry no auto actions: their Ruby stored none)
        if !name.starts_with("g1867") {
            assert_eq!(r["autos"]["checked"], r["autos"]["equal"], "{name}: {}", r["autos"]["differ"]);
        }
        // revenues as stored (the fixtures' older runs keep their old run strings and node orders)
        for d in r["runs"]["differ"].as_array().unwrap() {
            let rec: Vec<Value> = d["recorded"].as_array().unwrap().iter().map(|x| x[1].clone()).collect();
            let rust: Vec<Value> = d["rust"].as_array().unwrap().iter().map(|x| x[1].clone()).collect();
            assert_eq!(rec, rust, "{name} {}", d["id"]);
        }
    }
    // the programs' auto actions are reproduced (programs_exchange_ca: share passes, buys, disables)
    let mut g = SiteGame::load(&fixture("programs_exchange_ca")).unwrap();
    g.site_log = false;
    let r = authority::corpus_check(&mut g);
    assert!(r["autos"]["equal"].as_u64().unwrap() >= 5, "{r}");
}

#[test]
fn the_1830_check_refuses_what_no_menu_offers() {
    let data = fixture("router_d_trains");
    // 35: NYNH lays 1 on F20
    let lay = action(&data, 35);
    let mut g = before(&data, 35);
    assert!(decide(&g, &lay).is_ok());
    let offered = super::menu::tile_options(&mut g, "F20", None).unwrap();
    let rots: Vec<i64> = offered.as_array().unwrap().iter().filter(|o| o["name"] == "1").map(|o| o["rotation"].as_i64().unwrap()).collect();
    let bad_rot = (0..6).find(|r| !rots.contains(r)).unwrap();
    let mut a = lay.clone();
    a["rotation"] = json!(bad_rot);
    let e = decide(&g, &a).err().unwrap();
    assert!(e.starts_with("refused: Not a legal lay tile"), "{e}");
    let mut a = lay.clone();
    a["tile"] = json!("57-0");
    assert!(decide(&g, &a).is_err());
    let mut a = lay.clone();
    a["hex"] = json!("A9");
    assert!(decide(&g, &a).is_err());
    let mut a = lay.clone();
    a["entity"] = json!("PRR");
    assert!(decide(&g, &a).is_err());
    // a bid out of turn, a pass for someone else
    let e = decide(&g, &json!({"type": "pass", "entity": 2, "entity_type": "player"})).err().unwrap();
    assert!(e.starts_with("refused:"), "{e}");
}

#[test]
fn any_valid_run_is_accepted_and_priced_and_bad_runs_are_refused() {
    let data = fixture("router_d_trains");
    // 81: B&M runs three trains
    let run = action(&data, 81);
    let g = before(&data, 81);
    let d = decide(&g, &run).unwrap();
    assert!(d.repriced.is_empty());
    for (x, y) in d.action["routes"].as_array().unwrap().iter().zip(run["routes"].as_array().unwrap()) {
        assert_eq!((&x["revenue"], &x["revenue_str"], &x["hexes"], &x["nodes"]), (&y["revenue"], &y["revenue_str"], &y["hexes"], &y["nodes"]));
    }
    // a posted revenue that isn't the engine's: stored at the engine's, listed as repriced
    let mut a = run.clone();
    a["routes"][0]["revenue"] = json!(999);
    let d = decide(&g, &a).unwrap();
    assert_eq!(d.action["routes"][0]["revenue"], 100);
    assert_eq!(d.repriced, vec![json!({"path": "routes[0].revenue", "posted": 999, "stored": 100})]);
    // a run drawn by hand that isn't the router's (one train, a shorter route): accepted at its own price
    let mut a = run.clone();
    a["routes"] = json!([{"train": "3-0", "connections": [["F22", "F20"]], "hexes": ["F22", "F20"], "nodes": ["F22-0", "F20-1"]}]);
    let d = decide(&g, &a).unwrap();
    assert!(d.action["routes"][0]["revenue"].as_i64().unwrap() > 0);
    assert_eq!(d.action["routes"][0]["revenue_str"], "F22-F20");
    let refused = |routes: Value| -> String {
        let mut a = run.clone();
        a["routes"] = routes;
        decide(&g, &a).err().unwrap_or_default()
    };
    // a train twice, track twice, too many stops, a chain that isn't on the map, another company's train
    assert!(refused(json!([{"train": "2-4", "connections": [["F22", "F24"]]}, {"train": "2-4", "connections": [["E23", "F24"]]}])).contains("Cannot run train twice"));
    // (Ruby rebuilds the later route first and leaves the earlier one no track of its own: no route, no token)
    assert!(refused(json!([{"train": "2-4", "connections": [["F22", "F24"]]}, {"train": "2-5", "connections": [["F22", "F24"]]}])).contains("Route must contain token"));
    assert!(refused(json!([{"train": "2-4", "connections": [["F22", "F20"], ["F20", "G19"]]}])).contains("too many stops"));
    assert!(refused(json!([{"train": "3-0", "connections": [["F22", "H14"]]}])).contains("Route must contain token"));
    assert!(refused(json!([{"train": "9-9", "connections": [["F22", "F24"]]}])).contains("another corporation's train"));
}

#[test]
fn move_start_and_view_keep_their_contract() {
    let data = fixture("consent_log_undo");
    let last = data["actions"].as_array().unwrap().last().unwrap()["id"].as_i64().unwrap();
    // the player to act passes: accepted, stored with the next id, no `user` (it is the entity's player)
    let (code, out) = authority::move_json(&json!({"game": data, "action": {"type": "pass", "entity": 3, "entity_type": "player"}, "user": 3, "now": 1_800_000_000}));
    assert_eq!(code, cli::EXIT_OK);
    assert_eq!(out["verdict"], "accepted");
    assert_eq!(out["action"]["id"], last + 1);
    assert_eq!(out["action"]["created_at"], 1_800_000_000);
    assert!(out["action"].get("user").is_none());
    assert_eq!(out["actors"], json!([3]));
    assert_eq!(out["state"]["finished"], false);
    assert_eq!(out["key"]["v"], 2);
    // made for the seat by someone else (master mode): `user` kept, the actors say whose move it was
    let (_, out) = authority::move_json(&json!({"game": data, "action": {"type": "pass", "entity": 3, "entity_type": "player"}, "user": 1}));
    assert_eq!(out["action"]["user"], 1);
    assert_eq!(out["actors"], json!([3]));
    // someone else's move: refused in the engine's words, exit 0, nothing stored
    let (code, out) = authority::move_json(&json!({"game": data, "action": {"type": "pass", "entity": 1, "entity_type": "player"}, "user": 1}));
    assert_eq!(code, cli::EXIT_OK);
    assert_eq!(out["verdict"], "refused");
    assert_eq!(out["action"], Value::Null);
    assert!(out["error"].as_str().unwrap().len() > 3);
    // chat, an undo: accepted unchecked
    let (_, out) = authority::move_json(&json!({"game": data, "action": {"type": "message", "entity": 1, "entity_type": "player", "message": "hi"}, "user": 1}));
    assert_eq!((out["verdict"].clone(), out["action"]["message"].clone()), (json!("accepted"), json!("hi")));
    let (_, out) = authority::move_json(&json!({"game": data, "action": {"type": "undo", "entity": 3, "entity_type": "player"}, "user": 3}));
    assert_eq!(out["verdict"], "accepted");
    assert_eq!(out["action"]["type"], "undo");
    // bad input, an unsupported game
    assert_eq!(cli::run("move", "{}").0, cli::EXIT_BAD_INPUT);
    let other = json!({"game": {"title": "1889", "players": [{"id": 1}, {"id": 2}], "actions": []}, "action": {"type": "pass"}});
    assert_eq!(cli::run("move", &other.to_string()).0, cli::EXIT_UNSUPPORTED);
    // start: the opening position
    let (code, out) = cli::run("start", &json!({"game": data}).to_string());
    assert_eq!(code, cli::EXIT_OK);
    assert_eq!(out["state"]["round"], "Auction Round");
    assert_eq!(out["state"]["acting"].as_array().unwrap().len(), 1);
    // view: the bot view's facts
    let mut rdt = fixture("router_d_trains");
    let list = rdt["actions"].as_array().unwrap().clone();
    let at = list.iter().position(|a| a["id"] == json!(81)).unwrap();
    rdt["actions"] = json!(list[..at].to_vec());
    let (code, out) = cli::run("view", &json!({"game": rdt}).to_string());
    assert_eq!(code, cli::EXIT_OK);
    let v = &out["view"];
    assert_eq!((v["finished"].clone(), v["error"].clone()), (json!(false), Value::Null));
    assert!(v["entity"]["id"].is_string() || v["entity"]["id"].is_number());
    assert!(v["step"].is_string() && v["actions"].is_array() && v["trains"].is_array() && v["depot"].is_array());
}

#[test]
fn details_carry_rubys_liquidity_and_the_supply_count() {
    // 1830: cash alone before the second stock round (sellable_turn?), then cash and the dearest dumpable bundle
    let data = fixture("router_d_trains");
    let mut g = before(&data, 35);
    let d = super::details::details(&mut g, &json!(["sheet", "tiles"])).unwrap();
    for p in g.game.players.clone() {
        let l = d["sheet"]["liquidity"][p.id.to_string()].as_i64().unwrap();
        assert!(l >= p.cash as i64, "{l} < {}", p.cash);
    }
    let mut g = SiteGame::load(&data).unwrap();
    g.site_log = false;
    g.run_to(None);
    let rich = g.game.players.iter().map(|p| super::details::liquidity(&g, p.id) - p.cash).max().unwrap();
    assert!(rich > 0, "some shares are worth selling at the end");
    // 1867: an unlimited tile's count is the supply's (one, more once one is upgraded over)
    let mut g = SiteGame::load(&fixture("g1867_21268")).unwrap();
    g.site_log = false;
    g.run_to(None);
    let d = super::details::details(&mut g, &json!(["tiles"])).unwrap();
    let seven = d["tiles"]["tiles"].as_array().unwrap().iter().find(|t| t["name"] == "7").unwrap()["left"].as_u64().unwrap();
    assert!((1..100).contains(&seven), "{seven}");
}
