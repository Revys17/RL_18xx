//! Step 3a: engine-rs decides a move (docs/SITE_ENGINE_RS_STEP3.md §3, contract K3'). The server's
//! `record_action` hands the stored game and the posted move to `site_engine move`; the answer is the verdict, the
//! action as Ruby would have stored it (id, `user`, `created_at`, Rust's auto actions, a run's revenues and strings),
//! who may make the move (Ruby `valid_actors`, master mode), and the game's state and shadow key after it. `start`
//! answers the opening position, `view` the bot API's view facts (§8.1, K12).
//!
//! The replay of the stored game trusts the record as Ruby's `load` does (§2.5: the replay-only allowance); only the
//! posted move is checked: 1830 by the site menu of an acting seat or Ruby's cross-player set (`check1830.rs`), 1867 by
//! the legal-move generator (`g1867/act.rs`), runs by the route check (`route_check.rs`), and Ruby's auto actions are
//! Rust's (`programs.rs`).

use super::game::{refused, SiteGame};
use serde_json::{json, Map, Value};

fn s(v: &Value, k: &str) -> Option<String> {
    v.get(k).filter(|x| !x.is_null()).map(|x| x.as_str().map(str::to_string).unwrap_or_else(|| x.to_string()))
}

fn atype(a: &Value) -> String {
    a.get("type").and_then(Value::as_str).unwrap_or("").to_string()
}

/// The kinds of moves the site layer handles itself (not engine actions).
pub fn is_free(t: &str) -> bool {
    matches!(t, "message" | "log" | "end_game" | "undo" | "redo")
}

/// Ruby's `Action::<Type>#args_to_h` keys per type (what a stored action keeps besides the metadata), for the
/// types 1830 and 1867 use. None: keep every posted key.
fn ruby_arg_keys(t: &str) -> Option<&'static [&'static str]> {
    Some(match t {
        "pass" | "end_game" | "convert" | "redo" => &[],
        "undo" => &["action_id"],
        "message" | "log" => &["message"],
        "bid" => &["company", "corporation", "minor", "price"],
        "par" => &["corporation", "share_price", "slot", "purchase_for_type", "purchase_for", "borrow_from_type", "borrow_from"],
        "buy_shares" => &[
            "shares", "percent", "share_price", "swap", "purchase_for_type", "purchase_for", "borrow_from_type", "borrow_from",
            "total_price", "discounter",
        ],
        "sell_shares" => &["shares", "percent", "share_price", "swap"],
        "buy_company" => &["company", "price"],
        "lay_tile" => &["hex", "tile", "rotation", "combo_entities"],
        "place_token" => &["city", "slot", "cost", "tokener", "token_type"],
        "remove_token" => &["city", "slot"],
        "run_routes" => &["routes", "extra_revenue", "subsidy"],
        "dividend" => &["kind", "amount"],
        "buy_train" => &["train", "price", "variant", "exchange", "shell", "slots", "extra_due", "warranties"],
        "discard_train" => &["train"],
        "bankrupt" => &["option"],
        "merge" => &["corporation", "minor"],
        "choose" => &["choice", "target", "target_type"],
        "take_loan" => &["loan"],
        "payoff_loan" => &["loan"],
        "program_buy_shares" => &["corporation", "until_condition", "from_market", "auto_pass_after"],
        "program_share_pass" => &["unconditional", "indefinite"],
        "program_merger_pass" => &["corporations_by_round", "options"],
        "program_disable" => &["reason", "original_type"],
        _ => return None,
    })
}

/// The posted move without its metadata and without null values (Ruby's `to_h` drops nils).
pub fn bare(a: &Value) -> Value {
    let mut out = Map::new();
    if let Some(m) = a.as_object() {
        for (k, v) in m {
            if super::json::META_KEYS.contains(&k.as_str()) || v.is_null() {
                continue;
            }
            out.insert(k.clone(), v.clone());
        }
    }
    Value::Object(out)
}

/// Ruby `action.entity.player`: Some(Some(id)) a player of the game, Some(None) an entity whose player isn't one of
/// the game's (a stranger's id: Ruby's `Player.new(nil, ...)`), None an entity without a player (a corporation
/// without a president, a company the bank holds).
pub fn entity_player(g: &SiteGame, a: &Value) -> Option<Option<u32>> {
    let et = s(a, "entity_type").unwrap_or_default();
    let ent = s(a, "entity").unwrap_or_default();
    let game = &g.game;
    let corp_owner = |sym: &str| -> Option<u32> {
        let &ci = game.corp_idx.get(sym)?;
        let c = &game.corporations[ci];
        c.owner_id.player_id().or_else(|| c.president_id())
    };
    match et.as_str() {
        "corporation" | "minor" => corp_owner(&ent).map(Some),
        "company" => {
            let &mi = game.company_idx.get(ent.as_str())?;
            let owner = &game.companies[mi].owner;
            match owner.player_id() {
                Some(p) => Some(Some(p)),
                None => owner.corp_sym().and_then(corp_owner).map(Some),
            }
        }
        // a player (Ruby's `game.get` of an unknown type is nil: `Player.new(nil, entity)`)
        _ => Some(ent.parse::<u32>().ok().filter(|p| g.players.iter().any(|x| x.0 == *p))),
    }
}

/// Ruby `active_players_id` (the round's active entities' players; every player not bankrupt when there are none).
pub fn ruby_active_players(g: &mut SiteGame) -> Vec<u32> {
    let acting = if g.game.title == "1867" {
        super::g1867::active_players(&mut g.game)
    } else {
        g.game.acting_player_id_pub().into_iter().collect()
    };
    if acting.is_empty() {
        super::view::player_order(g).into_iter().filter(|p| g.game.players.iter().any(|x| x.id == *p && !x.bankrupt)).collect()
    } else {
        acting
    }
}

/// Ruby `valid_actors(action)` on the position before the move: the entity's player, else the active players.
/// (`record_action`'s master mode: a move is someone else's when the user isn't among these and it isn't free.)
pub fn actors(g: &mut SiteGame, a: &Value) -> Vec<u32> {
    match entity_player(g, a) {
        Some(Some(p)) => vec![p],
        Some(None) => Vec::new(),
        None => ruby_active_players(g),
    }
}

/// The game's state as `set_game_state` stores it (§3.2): who acts, the round, the result, the end.
pub fn state(g: &mut SiteGame, key: &Value) -> Value {
    json!({
        "acting": key["acting"],
        "round": key["round"],
        "turn": key["turn"],
        "round_num": key["round_num"],
        "finished": key["finished"],
        "result": key["result"],
        "manually_ended": key["manually_ended"],
        "game_end_reason": key["game_end_reason"],
        "last_game_action_id": g.last_game_action_id,
        "stopped": g.stopped.as_ref().map(|x| json!({"actionId": x.action_id, "message": x.message})),
    })
}

/// A share id's percent ("PRR_3" -> 10).
fn share_percent(g: &SiteGame, id: &str) -> Option<u32> {
    let (corp, idx) = id.rsplit_once('_')?;
    let idx: usize = idx.parse().ok()?;
    let &ci = g.game.corp_idx.get(corp)?;
    g.game.corporations[ci].shares.iter().find(|x| x.index == idx).map(|x| x.percent as u32)
}

/// The action as Ruby's `to_h` stores it: type, entity, entity_type, id, user (only when it isn't the entity's
/// player: `Action::Base.from_h`), created_at, auto_actions (when any), then the type's own arguments, nils left out.
/// `g` is the position before the move.
pub fn ruby_shape(g: &SiteGame, posted: &Value, id: i64, user: Option<u64>, created_at: i64, autos: &[Value]) -> Value {
    let t = atype(posted);
    let mut m = Map::new();
    m.insert("type".into(), json!(t));
    if let Some(e) = posted.get("entity").filter(|x| !x.is_null()) {
        m.insert("entity".into(), e.clone());
    }
    if let Some(e) = posted.get("entity_type").filter(|x| !x.is_null()) {
        m.insert("entity_type".into(), e.clone());
    }
    m.insert("id".into(), json!(id));
    if let Some(u) = user {
        let keep = match entity_player(g, posted) {
            Some(Some(p)) => p as u64 != u,
            Some(None) => true,
            None => false,
        };
        if keep {
            m.insert("user".into(), json!(u));
        }
    }
    m.insert("created_at".into(), json!(created_at));
    if !autos.is_empty() {
        m.insert("auto_actions".into(), json!(autos));
    }
    let b = bare(posted);
    let keys: Vec<String> = match ruby_arg_keys(&t) {
        Some(ks) => ks.iter().map(|k| k.to_string()).collect(),
        None => b.as_object().map(|o| o.keys().filter(|k| !matches!(k.as_str(), "type" | "entity" | "entity_type")).cloned().collect()).unwrap_or_default(),
    };
    for k in keys {
        if let Some(v) = b.get(&k) {
            m.insert(k, v.clone());
        }
    }
    // a bundle's percent (ShareBundle#percent: the shares' sum when none was posted)
    if (t == "buy_shares" || t == "sell_shares") && !m.contains_key("percent") {
        let pct: u32 = posted
            .get("shares")
            .and_then(Value::as_array)
            .map(|xs| xs.iter().filter_map(|x| x.as_str()).filter_map(|x| share_percent(g, x)).sum())
            .unwrap_or(0);
        m.insert("percent".into(), json!(pct));
    }
    // ExtraDue: `true` or nothing
    if t == "buy_train" {
        match m.get("extra_due") {
            Some(Value::Bool(true)) => {}
            Some(_) => {
                m.remove("extra_due");
            }
            None => {}
        }
    }
    Value::Object(m)
}

/// An auto action as Ruby stores it (no id, no user; `created_at` the time it was made).
pub fn auto_shape(a: &Value, created_at: i64) -> Value {
    let t = atype(a);
    let mut m = Map::new();
    m.insert("type".into(), json!(t));
    if let Some(e) = a.get("entity") {
        m.insert("entity".into(), e.clone());
    }
    if let Some(e) = a.get("entity_type") {
        m.insert("entity_type".into(), e.clone());
    }
    m.insert("created_at".into(), json!(created_at));
    if let Some(o) = a.as_object() {
        for (k, v) in o {
            if matches!(k.as_str(), "type" | "entity" | "entity_type" | "id" | "user" | "created_at" | "auto_actions") || v.is_null() {
                continue;
            }
            m.insert(k.clone(), v.clone());
        }
    }
    Value::Object(m)
}

/// What `decide` found.
pub struct Decision {
    /// The action as stored (with its id and auto actions)
    pub action: Value,
    /// The posted auto actions when they differ from Rust's (`ignore_posted_autos`: logged by the server)
    pub posted_autos: Option<Value>,
    /// A posted revenue Rust priced otherwise: [{path, posted, stored}] (lib/run_revenue.rb's `mismatches`)
    pub repriced: Vec<Value>,
    /// The stored action's step on the game (the board's)
    pub step: Option<super::game::StepOut>,
}

/// The player-time checks of a posted engine move on the position `g` (replayed to its end): Ok(the action to
/// process, a run with Rust's prices) or Err(refused: words).
pub fn check(g: &mut SiteGame, a: &Value) -> Result<Value, String> {
    let t = atype(a);
    if t == "run_routes" {
        return super::route_check::checked_run(g, a);
    }
    if g.game.title == "1867" {
        super::g1867::act::check_move(g, a)?;
    } else {
        super::check1830::check_move(g, a)?;
    }
    Ok(a.clone())
}

/// The checks of a `program_*` move (Ruby's Step::Program: a player's, a type the title offers, and
/// `program_buy_shares` with its `until_condition`).
fn check_program(g: &SiteGame, a: &Value) -> Result<(), String> {
    let t = atype(a);
    let pid = s(a, "entity").and_then(|e| e.parse::<u32>().ok());
    if s(a, "entity_type").as_deref().unwrap_or("player") != "player" || !pid.is_some_and(|p| g.players.iter().any(|x| x.0 == p)) {
        return Err(refused("Programs are a player's"));
    }
    let offered: &[&str] = if g.game.title == "1867" {
        &["program_merger_pass", "program_buy_shares", "program_share_pass", "program_disable"]
    } else {
        &["program_buy_shares", "program_share_pass", "program_disable"]
    };
    if !offered.contains(&t.as_str()) {
        return Err(refused(format!("{t} is not offered in {}", g.game.title)));
    }
    if t == "program_buy_shares" {
        if a.get("until_condition").is_none_or(Value::is_null) {
            return Err(refused("Until condition is unset"));
        }
        let corp = s(a, "corporation").unwrap_or_default();
        if !g.game.corp_idx.contains_key(corp.as_str()) {
            return Err(refused(format!("No corporation {corp}")));
        }
    }
    Ok(())
}

/// Decide a posted move on `g` (the stored game replayed to its end, not stopped). On acceptance `g` is the
/// position after it (the stored action appended) and the Decision says what to store; on refusal `g` is unchanged
/// and the error is the engine's words (`refused: ...`). Undo and redo are the caller's (`decide_json`).
pub fn decide(g: &mut SiteGame, posted: &Value, user: Option<u64>, now: i64) -> Result<Decision, String> {
    let t = atype(posted);
    if t.is_empty() {
        return Err(refused("an action needs a type"));
    }
    let created_at = posted.get("created_at").and_then(Value::as_i64).unwrap_or(now);
    let posted_autos = posted.get("auto_actions").filter(|x| x.as_array().is_some_and(|v| !v.is_empty())).cloned();
    let mut a = bare(posted);
    let mut repriced = Vec::new();
    let engine_move = !is_free(&t) && !t.starts_with("program_");
    if t.starts_with("program_") {
        check_program(g, &a)?;
    } else if t == "end_game" && g.finished() {
        return Err(refused("Game is already finished"));
    } else if engine_move {
        if g.finished() {
            return Err(refused("The game is over"));
        }
        let checked = check(g, &a)?;
        if t == "run_routes" {
            repriced = super::route_check::mismatches(&a, &checked);
        }
        a = checked;
    }
    // the move and Ruby's auto actions after it (`process_action`'s loop), on a copy
    let id = g.current_action_id + 1;
    let mut copy = g.clone();
    copy.site_log = false;
    let mut probe = a.clone();
    probe["id"] = json!(id);
    if let Some(ca) = posted.get("created_at") {
        probe["created_at"] = ca.clone();
    } else {
        probe["created_at"] = json!(now);
    }
    copy.append(&probe)?;
    let autos = super::programs::run_autos(&mut copy, now)?;
    let stored = ruby_shape(g, &a, id, user, created_at, &autos);
    // the stored action on the real game: it must replay as the record will
    let mut with_user = stored.clone();
    if let Some(u) = user {
        with_user["user"] = json!(u);
    }
    let step = g.append(&with_user).map_err(|e| format!("internal: the stored action does not replay: {e}"))?;
    let posted_autos = posted_autos.filter(|p| {
        let strip = |v: &Value| -> Vec<Value> {
            v.as_array().map(|xs| xs.iter().map(|x| bare(x)).collect()).unwrap_or_default()
        };
        strip(p) != autos.iter().map(bare).collect::<Vec<_>>()
    });
    Ok(Decision { action: stored, posted_autos, repriced, step })
}

/// The game JSON with one more raw action.
fn with_action(game: &Value, a: &Value) -> Value {
    let mut out = game.clone();
    match out.get_mut("actions").and_then(Value::as_array_mut) {
        Some(list) => list.push(a.clone()),
        None => out["actions"] = json!([a]),
    }
    out
}

/// `move`'s answer on parsed input (contract K3'): (exit code, JSON).
pub fn move_json(input: &Value) -> (i32, Value) {
    use super::cli::{fail, EXIT_BAD_INPUT, EXIT_OK, EXIT_UNSUPPORTED};
    let t0 = std::time::Instant::now();
    let Some(game) = input.get("game").filter(|g| g.is_object()) else {
        return fail(EXIT_BAD_INPUT, "bad input: no `game` object");
    };
    let Some(posted) = input.get("action").filter(|a| a.is_object()) else {
        return fail(EXIT_BAD_INPUT, "bad input: no `action` object");
    };
    let user = input.get("user").and_then(Value::as_u64).or_else(|| posted.get("user").and_then(Value::as_u64));
    let now = input.get("now").and_then(Value::as_i64).unwrap_or_else(now_secs);
    let mut g = match SiteGame::load(game) {
        Ok(g) => g,
        Err(e) => {
            let code = if e.starts_with("unsupported:") { EXIT_UNSUPPORTED } else { EXIT_BAD_INPUT };
            return fail(code, e);
        }
    };
    g.site_log = false;
    g.run_to(None);
    if let Some(st) = &g.stopped {
        return fail(EXIT_UNSUPPORTED, format!("unsupported: the replay stopped at action {}: {}", st.action_id, st.message));
    }
    let actors = actors(&mut g, posted);
    let t = atype(posted);
    if t == "undo" || t == "redo" {
        // Ruby's process_action takes them unchecked and reloads the game (`clone(@raw_actions)`)
        let a = bare(posted);
        let created_at = posted.get("created_at").and_then(Value::as_i64).unwrap_or(now);
        let stored = ruby_shape(&g, &a, g.current_action_id + 1, user, created_at, &[]);
        let mut with_user = stored.clone();
        if let Some(u) = user {
            with_user["user"] = json!(u);
        }
        let mut after = match SiteGame::load(&with_action(game, &with_user)) {
            Ok(x) => x,
            Err(e) => return fail(EXIT_UNSUPPORTED, e),
        };
        after.site_log = false;
        after.run_to(None);
        let key = super::shadow::key_now(&mut after);
        let state = state(&mut after, &key);
        return (EXIT_OK, json!({"verdict": "accepted", "error": null, "action": stored, "posted_autos": null, "repriced": [],
                                "actors": actors, "state": state, "key": key, "ms": t0.elapsed().as_millis() as u64}));
    }
    match decide(&mut g, posted, user, now) {
        Ok(d) => {
            let key = super::shadow::key_now(&mut g);
            let state = state(&mut g, &key);
            (EXIT_OK, json!({"verdict": "accepted", "error": null, "action": d.action, "posted_autos": d.posted_autos,
                             "repriced": d.repriced, "actors": actors, "state": state, "key": key,
                             "ms": t0.elapsed().as_millis() as u64}))
        }
        Err(e) if e.starts_with("internal:") || e.starts_with("unsupported:") => {
            let code = if e.starts_with("unsupported:") { EXIT_UNSUPPORTED } else { super::cli::EXIT_INTERNAL };
            fail(code, e)
        }
        Err(e) => {
            let words = e.strip_prefix("refused: ").unwrap_or(&e).to_string();
            let key = super::shadow::key_now(&mut g);
            let state = state(&mut g, &key);
            (EXIT_OK, json!({"verdict": "refused", "error": words, "action": null, "posted_autos": null, "repriced": [],
                             "actors": actors, "state": state, "key": key, "ms": t0.elapsed().as_millis() as u64}))
        }
    }
}

/// `start {"game"}`: the opening position (Ruby's `start_game!` loads the game with no actions).
pub fn start_json(input: &Value) -> (i32, Value) {
    use super::cli::{fail, EXIT_BAD_INPUT, EXIT_OK, EXIT_UNSUPPORTED};
    let t0 = std::time::Instant::now();
    let Some(game) = input.get("game").filter(|g| g.is_object()) else {
        return fail(EXIT_BAD_INPUT, "bad input: no `game` object");
    };
    let mut bare_game = game.clone();
    bare_game["actions"] = json!([]);
    let mut g = match SiteGame::load(&bare_game) {
        Ok(g) => g,
        Err(e) => {
            let code = if e.starts_with("unsupported:") { EXIT_UNSUPPORTED } else { EXIT_BAD_INPUT };
            return fail(code, e);
        }
    };
    g.site_log = false;
    let key = super::shadow::key_now(&mut g);
    let state = state(&mut g, &key);
    (EXIT_OK, json!({"ok": true, "state": state, "key": key, "ms": t0.elapsed().as_millis() as u64}))
}

/// The bot API's view facts (§8.1, contract K12): the values of rl18xx-next `routes/bot.rb` `bot_view` but
/// `last_id` and `finished`'s `status` part, which are the server's. `trains` is Ruby's
/// `engine.corporations.reject { trains.empty? }` as [[sym, [{id, name}]]] pairs in its order (a Ruby hash keeps
/// it; JSON objects here don't), `depot` Ruby's `depot.depot_trains`.
pub fn bot_view(g: &mut SiteGame) -> Value {
    let error = g.stopped.as_ref().map(|x| x.message.clone());
    let finished = g.game.finished;
    let acting = if finished { finished_acting(g) } else { ruby_active_players(g) };
    let mut out = json!({"finished": finished, "error": error, "entity": null, "step": null, "actions": [], "acting": acting,
                         "trains": null, "depot": null});
    if finished || error.is_some() {
        return out;
    }
    let is_1867 = g.game.title == "1867";
    let (entity, step, actions) = if is_1867 {
        let active = super::g1867::active_step(&mut g.game);
        let actions = super::g1867::ruby_actions(&mut g.game);
        (active.as_ref().map(|(_, e)| super::g1867::entity_id(e)), active.map(|(k, _)| super::g1867::step_class(k).to_string()), actions)
    } else {
        (super::shadow::ruby_entity(g), super::shadow::ruby_step_class(g).map(str::to_string), super::shadow::ruby_actions(g))
    };
    let Some(entity) = entity else { return out };
    let game = &g.game;
    let (etype, player) = if game.corp_idx.contains_key(entity.as_str()) {
        let ci = game.corp_idx[entity.as_str()];
        let c = &game.corporations[ci];
        ("corporation", c.owner_id.player_id().or_else(|| c.president_id()))
    } else if let Some(&mi) = game.company_idx.get(entity.as_str()) {
        let owner = &game.companies[mi].owner;
        let p = owner.player_id().or_else(|| {
            owner.corp_sym().and_then(|c| game.corp_idx.get(c)).and_then(|&ci| {
                let c = &game.corporations[ci];
                c.owner_id.player_id().or_else(|| c.president_id())
            })
        });
        ("company", p)
    } else if let Ok(p) = entity.parse::<u32>() {
        ("player", Some(p))
    } else {
        ("other", None)
    };
    let id_val = if etype == "player" { json!(entity.parse::<u32>().unwrap_or(0)) } else { json!(entity) };
    out["entity"] = json!({"id": id_val, "type": etype, "player": player});
    if step.is_some() {
        out["step"] = json!(step);
        out["actions"] = json!(actions);
    }
    // Ruby's engine.corporations, in its order (1867: its own membership)
    let members: Vec<usize> = if is_1867 {
        super::g1867::ruby_corporations(game)
    } else {
        (0..game.corporations.len()).filter(|&i| !game.corporations[i].closed).collect()
    };
    let trains: Vec<Value> = members
        .iter()
        .map(|&i| &game.corporations[i])
        .filter(|c| !c.trains.is_empty())
        .map(|c| json!([c.sym, c.trains.iter().map(|t| json!({"id": t.id, "name": t.name})).collect::<Vec<_>>()]))
        .collect();
    out["trains"] = json!(trains);
    let depot: Vec<Value> = super::menu::depot_trains(g).into_iter().map(|(id, name, _, _)| json!({"id": id, "name": name})).collect();
    out["depot"] = json!(depot);
    out
}

/// Ruby's `active_players_id` once the game is over (bot_view's `acting` of a finished game): the round is left as
/// it was, so its active entities' players, or every player not bankrupt. Best effort: the players who were to act
/// when it ended (`SiteGame::acting_at_end`), else every player not bankrupt.
fn finished_acting(g: &SiteGame) -> Vec<u32> {
    if let Some(a) = &g.acting_at_end {
        if !a.is_empty() {
            return a.clone();
        }
    }
    super::view::player_order(g).into_iter().filter(|p| g.game.players.iter().any(|x| x.id == *p && !x.bankrupt)).collect()
}

/// `view {"game"}`: the bot view facts and the key of the position the stored game replays to.
pub fn view_json(input: &Value) -> (i32, Value) {
    use super::cli::{fail, EXIT_BAD_INPUT, EXIT_OK, EXIT_UNSUPPORTED};
    let t0 = std::time::Instant::now();
    let Some(game) = input.get("game").filter(|g| g.is_object()) else {
        return fail(EXIT_BAD_INPUT, "bad input: no `game` object");
    };
    let upto = input.get("upto").and_then(Value::as_i64);
    let mut g = match SiteGame::load(&super::cli::truncate(game, upto)) {
        Ok(g) => g,
        Err(e) => {
            let code = if e.starts_with("unsupported:") { EXIT_UNSUPPORTED } else { EXIT_BAD_INPUT };
            return fail(code, e);
        }
    };
    g.site_log = false;
    g.run_to(None);
    let view = bot_view(&mut g);
    let key = super::shadow::key_now(&mut g);
    (EXIT_OK, json!({"ok": true, "view": view, "key": key, "ms": t0.elapsed().as_millis() as u64}))
}

/// The time now in seconds (Ruby's `Time.now.to_i`): the system's natively, the page's `Date.now()` in the wasm.
pub fn now_secs() -> i64 {
    #[cfg(not(target_arch = "wasm32"))]
    {
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs() as i64).unwrap_or(0)
    }
    #[cfg(all(target_arch = "wasm32", feature = "wasm"))]
    {
        (js_date_now() / 1000.0) as i64
    }
    #[cfg(all(target_arch = "wasm32", not(feature = "wasm")))]
    {
        0
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen::prelude::wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = Date, js_name = now)]
    fn js_date_now() -> f64;
}

/// The kept engine moves of a game checked one by one as `move` would decide them (`site_replay --authority`,
/// §12.1 A1): each at its position, the decision compared with the record (auto actions, a run's revenues and
/// strings). `{moves, accepted, refused: [...], autos: {checked, equal, differ: [...]}, runs: {...}, stopped}`.
pub fn corpus_check(g: &mut SiteGame) -> Value {
    let (mut moves, mut ok) = (0u32, 0u32);
    let (mut auto_checked, mut auto_eq) = (0u32, 0u32);
    let (mut runs, mut runs_eq) = (0u32, 0u32);
    let mut bad = Vec::new();
    let mut auto_bad = Vec::new();
    let mut run_bad = Vec::new();
    while !g.done() {
        let i = g.pos;
        let a = g.raw[i].clone();
        let t = atype(&a);
        let kept = g.kept[i];
        let checkable = kept && t != "undo" && t != "redo" && !(is_free(&t) && t != "message") && !(t == "message" && a.get("auto_actions").is_none());
        if checkable && !(g.game.finished && !is_free(&t) && !t.starts_with("program_")) {
            moves += 1;
            let mut copy = g.clone();
            copy.site_log = false;
            // the position as the server held it: nothing after this action
            copy.raw.truncate(copy.pos);
            copy.kept.truncate(copy.pos);
            let created = a.get("created_at").and_then(Value::as_i64).unwrap_or(0);
            let user = a.get("user").and_then(Value::as_u64);
            match decide(&mut copy, &a, user, created) {
                Ok(d) => {
                    ok += 1;
                    let rec: Vec<Value> = a.get("auto_actions").and_then(Value::as_array).cloned().unwrap_or_default();
                    let mine: Vec<Value> = d.action.get("auto_actions").and_then(Value::as_array).cloned().unwrap_or_default();
                    if !rec.is_empty() || !mine.is_empty() {
                        auto_checked += 1;
                        let norm = |v: &[Value]| -> Vec<Value> {
                            v.iter()
                                .map(|x| {
                                    let mut b = bare(x);
                                    if let Some(m) = b.as_object_mut() {
                                        // program_disable's original_type: Ruby's own disables leave it nil
                                        m.remove("created_at");
                                    }
                                    b
                                })
                                .collect()
                        };
                        if norm(&rec) == norm(&mine) {
                            auto_eq += 1;
                        } else if auto_bad.len() < 60 {
                            auto_bad.push(json!({"id": a.get("id"), "type": t, "recorded": norm(&rec), "rust": norm(&mine)}));
                        }
                    }
                    if t == "run_routes" {
                        runs += 1;
                        let fields = |v: &Value| -> Vec<Value> {
                            v.get("routes")
                                .and_then(Value::as_array)
                                .map(|rs| {
                                    rs.iter()
                                        .map(|r| json!([r.get("train"), r.get("revenue"), r.get("revenue_str"), r.get("hexes"), r.get("nodes")]))
                                        .collect()
                                })
                                .unwrap_or_default()
                        };
                        if fields(&a) == fields(&d.action) {
                            runs_eq += 1;
                        } else if run_bad.len() < 60 {
                            run_bad.push(json!({"id": a.get("id"), "recorded": fields(&a), "rust": fields(&d.action)}));
                        }
                    }
                }
                Err(e) => {
                    if bad.len() < 60 {
                        bad.push(json!({"id": a.get("id"), "action": bare(&a), "error": e}));
                    }
                }
            }
        }
        g.process_next();
    }
    json!({"moves": moves, "accepted": ok, "refused": bad, "autos": {"checked": auto_checked, "equal": auto_eq, "differ": auto_bad},
           "runs": {"checked": runs, "equal": runs_eq, "differ": run_bad},
           "stopped": g.stopped.as_ref().map(|x| json!({"id": x.action_id, "message": x.message}))})
}
