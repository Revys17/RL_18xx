//! The `site_engine` CLI's commands (docs/SITE_ENGINE_RS.md §5.4, contract K3) as plain functions: one JSON object
//! in, one JSON value out, and the process exit code. `src/bin/site_engine.rs` reads stdin and prints; the tests and
//! the corpus tools call these directly.
//!
//! Exit codes: 0 whenever a verdict was reached (a Rust refusal included), 2 bad input, 3 unsupported (title,
//! optional rule), 101 a panic (Rust's own exit code for one).

use super::game::SiteGame;
use serde_json::{json, Value};

pub const EXIT_OK: i32 = 0;
pub const EXIT_BAD_INPUT: i32 = 2;
pub const EXIT_UNSUPPORTED: i32 = 3;
/// A bug the engine caught itself (an `internal:` error; a panic is 101)
pub const EXIT_INTERNAL: i32 = 4;

/// What `version` prints (and `WasmGame.version()` returns).
pub fn version() -> Value {
    let mut features = vec!["site"];
    if cfg!(feature = "wasm") {
        features.push("wasm");
    }
    if cfg!(feature = "site-titles") {
        features.push("site-titles");
    }
    // step 3a (docs/SITE_ENGINE_RS_STEP3.md §10, K1'): `act` makes the programs' auto actions and accepts any valid
    // run; the CLI has `move`, `start`, `view` (K3')
    features.extend(["programs", "route_check", "authority"]);
    json!({"engine": "engine-rs", "commit": super::COMMIT, "api": super::API_VERSION, "key": super::KEY_VERSION,
           "titles": super::TITLES, "features": features})
}

pub fn fail(code: i32, msg: impl Into<String>) -> (i32, Value) {
    (code, json!({"ok": false, "error": msg.into()}))
}

fn stop_json(g: &SiteGame) -> Value {
    match &g.stopped {
        Some(s) => json!({"actionId": s.action_id, "index": s.index, "message": s.message}),
        None => Value::Null,
    }
}

/// The game JSON with only the raw actions up to `upto` (an action id; None: all), as the server holds it at that
/// action: undo/redo are resolved over exactly these.
pub fn truncate(game: &Value, upto: Option<i64>) -> Value {
    let mut g = game.clone();
    if let (Some(u), Some(Value::Array(actions))) = (upto, g.get_mut("actions")) {
        actions.retain(|a| a.get("id").and_then(Value::as_i64).map_or(true, |id| id <= u));
    }
    g
}

fn load(game: &Value) -> Result<SiteGame, (i32, Value)> {
    if !game.is_object() {
        return Err(fail(EXIT_BAD_INPUT, "bad input: `game` must be the site's game JSON"));
    }
    SiteGame::load(game).map_err(|e| {
        let code = if e.starts_with("unsupported:") { EXIT_UNSUPPORTED } else { EXIT_BAD_INPUT };
        fail(code, e)
    })
}

fn ms_since(t: std::time::Instant) -> u64 {
    t.elapsed().as_millis() as u64
}

/// `state {"game", "upto"}` -> `{"ok": true, "state": <key v1>, "stopped": null | {actionId, index, message}, "ms"}`
pub fn state(input: &Value) -> (i32, Value) {
    let t = std::time::Instant::now();
    let Some(game) = input.get("game") else { return fail(EXIT_BAD_INPUT, "bad input: no `game`") };
    let upto = input.get("upto").and_then(Value::as_i64);
    let mut g = match load(&truncate(game, upto)) {
        Ok(g) => g,
        Err(e) => return e,
    };
    g.site_log = false;
    g.run_to(None);
    let key = super::shadow::key_now(&mut g);
    (EXIT_OK, json!({"ok": true, "state": key, "stopped": stop_json(&g), "ms": ms_since(t)}))
}

/// `keys {"game"}` -> `{"ok": true, "positions": [{"actionId", "state"}...], "stopped", "ms"}`: the key at the start
/// (actionId 0) and after every kept action, in one replay (the corpus mode, §8.2).
pub fn keys(input: &Value) -> (i32, Value) {
    let t = std::time::Instant::now();
    let Some(game) = input.get("game") else { return fail(EXIT_BAD_INPUT, "bad input: no `game`") };
    let mut g = match load(game) {
        Ok(g) => g,
        Err(e) => return e,
    };
    g.site_log = false;
    let mut positions = vec![json!({"actionId": 0, "state": super::shadow::key(&mut g, None)})];
    while !g.done() {
        if let Some(step) = g.process_next() {
            let k = super::shadow::key(&mut g, Some(&step.action));
            positions.push(json!({"actionId": step.action_id, "state": k}));
        }
    }
    (EXIT_OK, json!({"ok": true, "positions": positions, "stopped": stop_json(&g), "ms": ms_since(t)}))
}

/// `act {"game", "action"}` -> `{"action": <the site action as Rust would store it, its auto actions Rust's> | null,
/// "error": null | "<words>", "state": <key after, or now when refused>}` (step 3a: the authority's checks and
/// Programmer, `SiteGame::act`; `add_auto_actions` is ignored: Rust always makes them).
pub fn act(input: &Value) -> (i32, Value) {
    let Some(game) = input.get("game") else { return fail(EXIT_BAD_INPUT, "bad input: no `game`") };
    let Some(action) = input.get("action").filter(|a| a.is_object()) else {
        return fail(EXIT_BAD_INPUT, "bad input: no `action` object");
    };
    let mut g = match load(game) {
        Ok(g) => g,
        Err(e) => return e,
    };
    g.site_log = false;
    g.run_to(None);
    if let Some(s) = &g.stopped {
        let msg = format!("stopped: the replay stopped at action {}: {}", s.action_id, s.message);
        let key = super::shadow::key_now(&mut g);
        return (EXIT_OK, json!({"action": Value::Null, "error": msg, "state": key}));
    }
    match g.act(action) {
        Ok((posted, _step)) => {
            let key = super::shadow::key_now(&mut g);
            (EXIT_OK, json!({"action": posted, "error": Value::Null, "state": key}))
        }
        Err(e) => {
            let key = super::shadow::key_now(&mut g);
            (EXIT_OK, json!({"action": Value::Null, "error": e, "state": key}))
        }
    }
}

/// One command on its JSON input (`version` ignores it).
pub fn run(cmd: &str, input: &str) -> (i32, Value) {
    if cmd == "version" {
        return (EXIT_OK, version());
    }
    let parsed: Value = match serde_json::from_str(input) {
        Ok(v) => v,
        Err(e) => return fail(EXIT_BAD_INPUT, format!("bad input: not JSON: {e}")),
    };
    match cmd {
        "state" => state(&parsed),
        "keys" => keys(&parsed),
        "act" => act(&parsed),
        "move" => super::authority::move_json(&parsed),
        "start" => super::authority::start_json(&parsed),
        "view" => super::authority::view_json(&parsed),
        other => fail(EXIT_BAD_INPUT, format!("bad input: unknown command {other:?} (version, state, keys, act, move, start, view)")),
    }
}
