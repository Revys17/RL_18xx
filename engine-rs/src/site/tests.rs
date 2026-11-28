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
        // 1830 polish (§11.2): a scratch-stack game of the verification (15, whole, players renamed), and :9301's bot
        // game 44 (the bots renamed), where the bank breaks in OR 15.1
        "bankrupt_voluntary" => include_str!("../../site/fixtures/bankrupt_voluntary.json"),
        "bank_broken" => include_str!("../../site/fixtures/bank_broken.json"),
        // 1867 (docs/SITE_ENGINE_RS_1867.md): copies of RL_18xx tests/fixtures/1867 (anonymised 18xx.games games)
        "g1867_21268" => include_str!("../../site/fixtures/g1867_21268.json"),
        "g1867_nationalization_cash" => include_str!("../../site/fixtures/g1867_nationalization_cash.json"),
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
    assert_eq!(k["v"], 2);
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
    assert_eq!((v["api"].clone(), v["key"].clone(), v["engine"].clone()), (json!(3), json!(2), json!("engine-rs")));
    for f in ["programs", "route_check", "authority"] {
        assert!(v["features"].as_array().unwrap().contains(&json!(f)), "{f}");
    }
    assert_eq!(cli::run("state", "not json").0, cli::EXIT_BAD_INPUT);
    assert_eq!(cli::run("state", "{}").0, cli::EXIT_BAD_INPUT);
    assert_eq!(cli::run("nope", "{}").0, cli::EXIT_BAD_INPUT);
    let other = json!({"game": {"title": "1889", "players": [{"id": 1}, {"id": 2}], "actions": []}});
    assert_eq!(cli::run("state", &other.to_string()).0, cli::EXIT_UNSUPPORTED);
    let data = fixture("consent_log_undo");
    // step 3a: Rust makes the auto actions itself (add_auto_actions is accepted and ignored)
    let auto = json!({"game": data, "action": {"type": "pass", "entity": 3, "entity_type": "player"}, "add_auto_actions": true});
    assert_eq!(cli::act(&auto).0, cli::EXIT_OK);
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


#[test]
fn d_and_h_teleport_without_a_route_operates_the_company_in_every_mode() {
    // H5, unconditional part: the D&H branch's inline withhold is Ruby's Dividend#skip! -> process_dividend, so the
    // company has operated (ever_operated), whether or not site_rules is on. CPR has operated before 17@246 (OR 4.1),
    // so the flag is cleared first to see the branch set it.
    let data = fixture("dh_teleport_no_route");
    let a246 = data["actions"].as_array().unwrap().iter().find(|a| a["id"] == 246).unwrap().clone();
    for site_rules in [true, false] {
        let mut g = SiteGame::load(&data).unwrap();
        g.run_to(Some(245));
        let cpr = g.game.corp_idx["CPR"];
        g.game.corporations[cpr].ever_operated = false;
        g.game.site_rules = site_rules;
        let price = g.game.corporations[cpr].share_price.as_ref().map(|p| p.price);
        g.apply_engine(&a246).unwrap();
        assert!(g.game.corporations[cpr].ever_operated, "site_rules {site_rules}");
        // the rest of the branch as before: withheld (price left), on to Buy Trains
        assert_eq!((price, g.game.corporations[cpr].share_price.as_ref().map(|p| p.price)), (Some(30), Some(20)));
        assert_eq!(g.game.active_step_type(), "BuyTrain", "site_rules {site_rules}");
    }
}

#[test]
fn voluntary_bankruptcy_is_offered_where_the_sites_hand_offers_it() {
    // 15@597: NYC (no train, no route: Game#must_buy_train? is nil) in Buy Trains with $160, its president with $26;
    // the cheapest depot train costs $630. The site's hand (Opal) offers bankruptcy (Ruby's nil is true there) and
    // Ruby accepts it; the printed rule (the bots' enumeration) does not offer it: NYC needn't buy a train.
    let data = fixture("bankrupt_voluntary");
    let mut g = SiteGame::load(&cli::truncate(&data, Some(597))).unwrap();
    g.run_to(None);
    assert_eq!(g.game.active_step_type(), "BuyTrain");
    assert!(!g.game.must_buy_train("NYC"));
    let types = g.game.legal_action_types();
    assert!(!types.iter().any(|t| t == "bankrupt"), "{types:?}");
    let m = super::menu::menu(&mut g, 15);
    // Opal's golden at 15@597 (living-board test/rust/golden.cjs), created_at aside
    assert_eq!(
        m["bankrupt"],
        json!({"action": {"type": "bankrupt", "entity": "NYC", "entity_type": "corporation"}, "player": "Player 3",
               "corp": "NYC", "min": 630, "power": 246, "minText": "$630", "powerText": "$246",
               "ends": " The game ends immediately when any player is bankrupt."})
    );
    assert!(super::menu::legal_actions(&mut g).as_array().unwrap().iter().any(|a| a["type"] == "bankrupt"));
    // taken: the engine accepts it (Ruby's process_bankrupt checks can_go_bankrupt? only) and the game ends
    let (posted, _) = g.act(&m["bankrupt"]["action"]).unwrap();
    assert_eq!(posted["id"], 598);
    assert!(g.finished());
    assert_eq!(g.end_reason(), Some("bankrupt"));
    // the next position (598: NYC bought ERIE's 5 for $1, so it has a train): nothing offered
    let mut g = SiteGame::load(&cli::truncate(&data, Some(598))).unwrap();
    g.run_to(None);
    assert!(super::menu::menu(&mut g, 15).get("bankrupt").is_none());
    // the game's own end, 613: B&M, which must buy a train, goes bankrupt (offered under the printed rule too)
    let mut g = SiteGame::load(&data).unwrap();
    g.run_to(Some(612));
    assert!(g.game.legal_action_types().iter().any(|t| t == "bankrupt"));
    assert!(super::menu::menu(&mut g, 13)["bankrupt"].is_object());
    g.run_to(None);
    assert!(g.stopped.is_none(), "{:?}", g.stopped);
    assert_eq!(g.end_reason(), Some("bankrupt"));
}

#[test]
fn the_snapshot_says_when_the_game_ends_as_ruby_does() {
    // bank_broken (:9301's 44): the bank breaks at 794 in OR 15.1 of 3; :full_or, so the game ends after OR 15.3 (881)
    let mut g = SiteGame::load(&fixture("bank_broken")).unwrap();
    let snap_at = |g: &mut SiteGame, id: i64| {
        g.run_to(Some(id));
        super::view::snapshot(g, id)
    };
    assert_eq!(snap_at(&mut g, 793)["ending"], Value::Null);
    // Opal's (adapter.ts endingOf over game_ending_description) at every position from 794 to 880
    let want = json!({"text": "Bank Broken : Game Ends at conclusion of OR 15.3", "when": "after OR 15.3", "why": "bank broken"});
    assert_eq!(snap_at(&mut g, 794)["ending"], want);
    assert_eq!(snap_at(&mut g, 880)["ending"], want);
    let last = snap_at(&mut g, 881);
    assert_eq!((last["finished"].clone(), last["ending"].clone()), (json!(true), Value::Null));
    // (every snapshot has the key: Opal's always does)
    assert!(super::view::snapshot(&g, 881).as_object().unwrap().contains_key("ending"));
}

#[test]
fn the_snapshot_names_the_company_up_for_auction() {
    // 18@5: two bids on C&A while M&H is still first (no auction yet); 6: M&H bought, C&A goes up for auction (Ruby
    // Auctioner#auctioning, adapter.ts auctionedNow); 7: C&A sold
    let mut g = SiteGame::load(&fixture("programs_exchange_ca")).unwrap();
    let mut at = |id: i64| {
        g.run_to(Some(id));
        super::view::snapshot(&g, id).get("auctioning").cloned()
    };
    assert_eq!(at(5), None);
    assert_eq!(at(6), Some(json!("CA")));
    assert_eq!(at(7), None);
}

// ------------------------------------------------------------------------------------------------ 1867

#[test]
fn g1867_a_whole_game_replays_and_ends_as_in_ruby() {
    let data = fixture("g1867_21268");
    let mut g = SiteGame::load(&data).unwrap();
    g.run_to(None);
    assert!(g.stopped.is_none(), "{:?}", g.stopped);
    assert!(g.finished());
    // Ruby's game_end_check: the final phase's trigger outranks the bank's (GAME_END_TIMING_PRIORITY)
    assert_eq!(g.end_reason(), Some("final_phase"));
    let site: Vec<(u32, i32)> = data["result"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(k, v)| (k.parse().unwrap(), v.as_i64().unwrap() as i32))
        .collect();
    for (id, v) in site {
        assert!(g.result().contains(&(id, v)), "player {id}: {:?}", g.result());
    }
    let k = shadow::key_now(&mut g);
    assert_eq!(k["v"], 2);
    // the v2 fields 1867 has (and only those): loans, the national, trainless majors
    assert!(k["loans_left"].is_u64());
    assert!(k["national"]["tokens"].is_array());
    assert!(k["trainless"].is_array());
    assert!(k.get("merger").is_none());
    assert!(k["corps"].as_object().unwrap().values().all(|c| c["loans"].is_u64()));
    // Ruby's @corporations at the end: the majors only (every minor was nationalized in phase 8), never the national
    let corps: Vec<&String> = k["corps"].as_object().unwrap().keys().collect();
    assert!(!corps.iter().any(|c| *c == "CN"));
    assert_eq!(corps.len(), 8, "{corps:?}");
}

#[test]
fn g1867_two_players_and_grid_market_are_unsupported() {
    let mut data = fixture("g1867_21268");
    data["players"] = json!(data["players"].as_array().unwrap()[..2].to_vec());
    let e = SiteGame::load(&data).err().unwrap();
    assert_eq!(e, "unsupported: 1867 for 2 players");
    let mut data = fixture("g1867_21268");
    data["settings"] = json!({"optional_rules": ["grid_market"]});
    assert_eq!(SiteGame::load(&data).err().unwrap(), "unsupported: optional rule grid_market");
    assert!(super::TITLES.contains(&"1867"));
    assert_eq!(cli::version()["titles"], json!(["1830", "1867", "1822", "1822CA"]));
    assert_eq!(cli::version()["key"], 2);
}

/// The game up to (and without) action `id`, and that action as recorded.
fn g1867_before(name: &str, id: i64) -> (SiteGame, Value) {
    let data = fixture(name);
    let a = data["actions"].as_array().unwrap().iter().find(|a| a["id"] == json!(id)).unwrap().clone();
    let mut g = SiteGame::load(&cli::truncate(&data, Some(id - 1))).unwrap();
    g.run_to(None);
    assert!(g.stopped.is_none());
    (g, a)
}

#[test]
fn g1867_act_adds_rubys_step_auto_actions() {
    // NO's run (a minor with no station to place: G1867::Step::Token passes for it), and CA's train ending the OR
    // (the merger round's first minors pass: G1867::Step::Merge with nothing to convert or merge)
    for id in [170, 181] {
        let (mut g, a) = g1867_before("g1867_nationalization_cash", id);
        let recorded: Vec<Value> = a["auto_actions"].as_array().unwrap().iter().map(|x| json!({"type": x["type"], "entity": x["entity"]})).collect();
        // (a page posts the exact router's run, not the recorded one: it pays at least as much)
        let bare = if a["type"] == "run_routes" {
            super::menu::auto_route(&mut g)["action"].clone()
        } else {
            super::json::strip_meta(&a)
        };
        let (posted, _) = g.act(&bare).unwrap_or_else(|e| panic!("action {id}: {e}"));
        let mine: Vec<Value> = posted["auto_actions"].as_array().cloned().unwrap_or_default().iter().map(|x| json!({"type": x["type"], "entity": x["entity"]})).collect();
        assert_eq!(mine, recorded, "action {id}");
        // the position after a recorded move equals the replay's
        if a["type"] != "run_routes" {
            let (mut r, _) = g1867_before("g1867_nationalization_cash", id + 1);
            assert_eq!(shadow::key_now(&mut g)["corps"], shadow::key_now(&mut r)["corps"], "action {id}");
        }
    }
}

#[test]
fn g1867_act_refuses_what_the_generator_does_not_offer() {
    // the opening auction: a bid off the $5 grid, and one below the minimum
    let first = fixture("g1867_21268")["actions"].as_array().unwrap().iter().find(|a| a["type"] == "bid").unwrap()["id"].as_i64().unwrap();
    let (mut g, a) = g1867_before("g1867_21268", first);
    assert_eq!(a["type"], "bid");
    let mut off = super::json::strip_meta(&a);
    off["price"] = json!(a["price"].as_i64().unwrap() + 2);
    let e = g.clone().act(&off).err().unwrap();
    assert!(e.starts_with("refused: Price must be a multiple of $5"), "{e}");
    let mut low = super::json::strip_meta(&a);
    low["price"] = json!(5);
    let e = g.clone().act(&low).err().unwrap();
    assert!(e.starts_with("refused: Price must be between"), "{e}");
    // the recorded bid is accepted, and posted as the site stores it
    let (posted, _) = g.act(&super::json::strip_meta(&a)).unwrap();
    assert_eq!(posted["price"], a["price"]);
    assert!(posted.get("auto_actions").is_none());
}

#[test]
fn site_interplayer_purchases_read_rubys_phase_status() {
    // Ruby's PHASES status (the site's g_1830/game.rb and g_1867/game.rb): can_buy_companies_from_other_players in
    // 1830's phases 2-4 only, never in 1867
    let g30 = SiteGame::load(&fixture("router_d_trains")).unwrap();
    let ruby_1830: &[(&str, &[&str])] = &[
        ("2", &["can_buy_companies_from_other_players"]),
        ("3", &["can_buy_companies", "can_buy_companies_from_other_players"]),
        ("4", &["can_buy_companies", "can_buy_companies_from_other_players"]),
        ("5", &[]),
        ("6", &[]),
        ("D", &[]),
    ];
    let mut game = g30.game.clone_for_search();
    for (phase, want) in ruby_1830 {
        game.phase.name = phase.to_string();
        let mut have = game.ruby_phase_status();
        have.sort();
        let mut want = want.to_vec();
        want.sort();
        assert_eq!(have, want, "1830 phase {phase}");
    }
    let g67 = SiteGame::load(&fixture("g1867_21268")).unwrap();
    let mut game = g67.game.clone_for_search();
    for phase in ["2", "3", "4", "5", "6", "7", "8"] {
        game.phase.name = phase.to_string();
        assert!(!game.ruby_phase_status().contains(&"can_buy_companies_from_other_players"), "1867 phase {phase}");
    }
}

#[test]
fn key_v2_lists_the_1830_auction_bids() {
    // the waterfall auction of router_d_trains: after its first bid on a later private, the key has it
    let data = fixture("router_d_trains");
    let first_bid = data["actions"].as_array().unwrap().iter().find(|a| a["type"] == "bid").unwrap()["id"].as_i64().unwrap();
    let mut g = SiteGame::load(&cli::truncate(&data, Some(first_bid))).unwrap();
    g.run_to(None);
    let k = shadow::key_now(&mut g);
    let a = &k["auction"];
    assert!(a["bids"].as_array().is_some_and(|b| !b.is_empty()) || !a["lot"].is_null(), "{a}");
    assert!(a.get("dutch").is_none());
}

#[test]
fn g1867_menus_train_cards_are_moves_act_accepts() {
    // the train part of every acting player's menu at every position of the 1867 fixtures: each depot train it
    // enables (`short` 0) and each trade-in the treasury pays is a move act's check accepts (the generator's room
    // check: a company at its limit is offered no depot train), and no trade-in is listed at $0 or less (a 5+5E for
    // a 2+2: Spender#spend refuses it)
    for name in ["g1867_21268", "g1867_nationalization_cash"] {
        let data = fixture(name);
        let ids: Vec<i64> = data["actions"].as_array().unwrap().iter().filter_map(|a| a["id"].as_i64()).collect();
        let mut g = SiteGame::load(&data).unwrap();
        let (mut cards, mut exchanges) = (0, 0);
        for id in ids {
            g.run_to(Some(id));
            if g.finished() || g.stopped.is_some() {
                break;
            }
            let k = shadow::key_now(&mut g);
            for pid in k["acting"].as_array().cloned().unwrap_or_default().iter().filter_map(Value::as_u64) {
                let m = super::menu::menu(&mut g, pid as u32);
                for t in m["trains"].as_array().cloned().unwrap_or_default() {
                    if t["short"] == json!(0) {
                        cards += 1;
                        let r = super::g1867::act::check_move(&mut g.clone(), &t["action"]);
                        assert!(r.is_ok(), "{name}@{id}: {} refused: {:?}", t["action"], r);
                    }
                }
                let power = m["trainBuy"]["buyingPower"].as_i64().unwrap_or(0);
                for x in m["trainBuy"]["exchanges"].as_array().cloned().unwrap_or_default() {
                    let price = x["price"].as_i64().unwrap();
                    assert!(price > 0, "{name}@{id}: trade-in at {price}: {}", x["action"]);
                    if price <= power {
                        exchanges += 1;
                        let r = super::g1867::act::check_move(&mut g.clone(), &x["action"]);
                        assert!(r.is_ok(), "{name}@{id}: {} refused: {:?}", x["action"], r);
                    }
                }
            }
        }
        assert!(cards > 20, "{name}: {cards} train cards checked");
        let _ = exchanges;
    }
}
