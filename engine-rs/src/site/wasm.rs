//! `WasmGame`, API 2: the site layer as a wasm-bindgen class (docs/SITE_ENGINE_RS.md §3.2, contract K1; the
//! client's view of it is living-board `src/engine-rs/api.ts`). Every structured value crosses as a JSON string;
//! errors are thrown as JS `Error`s whose message starts with a code: `unsupported:` (fall back to Opal),
//! `refused:` (the engine refuses the action: its words), `resync:` (re-open the game), `internal:` (a bug).

use super::game::SiteGame;
use serde_json::{json, Value};
use wasm_bindgen::prelude::*;

fn err(msg: impl Into<String>) -> JsError {
    let m: String = msg.into();
    if m.starts_with("unsupported:") || m.starts_with("refused:") || m.starts_with("internal:") || m.starts_with("resync:") {
        JsError::new(&m)
    } else {
        JsError::new(&format!("internal: {m}"))
    }
}

fn parse(s: &str) -> Result<Value, JsError> {
    serde_json::from_str(s).map_err(|e| err(format!("internal: bad JSON: {e}")))
}

/// The most checkpoints a game keeps for `at`.
const MAX_CHECKPOINTS: usize = 8;

#[wasm_bindgen]
pub struct WasmGame {
    inner: SiteGame,
    /// checkpoints for `at`: (position after raw index, the game there)
    checkpoints: Vec<SiteGame>,
}

#[wasm_bindgen]
impl WasmGame {
    /// The site's game JSON (GET /api/game/<id>). Nothing processed yet; undo/redo already resolved over all raw
    /// actions. Throws `unsupported:` for a title other than 1830 or an optional rule the engine doesn't implement.
    pub fn load(game_json: &str) -> Result<WasmGame, JsError> {
        let data = parse(game_json)?;
        let inner = SiteGame::load(&data).map_err(err)?;
        Ok(WasmGame { inner, checkpoints: Vec::new() })
    }

    /// The digest of a whole replay (debugging, the native-vs-wasm check of §8.2): {digest, steps}, equal to
    /// `site_replay --digest`'s.
    #[wasm_bindgen(js_name = debugDigest)]
    pub fn debug_digest(game_json: &str) -> Result<String, JsError> {
        let data = parse(game_json)?;
        super::shadow::replay_digest(&data).map(|v| v.to_string()).map_err(err)
    }

    /// The engine's build: {engine: "engine-rs", commit, api: 2, key: 1, titles: ["1830"], features}. The worker
    /// falls back to Opal unless api === 2.
    pub fn version() -> String {
        super::cli::version().to_string()
    }

    #[wasm_bindgen(js_name = staticInfo)]
    pub fn static_info(&self) -> String {
        super::view::static_info(&self.inner).to_string()
    }

    /// The log lines written before the first action (step 0's).
    #[wasm_bindgen(js_name = startLog)]
    pub fn start_log(&self) -> String {
        json!(self.inner.start_log).to_string()
    }

    /// Process the next raw actions and return their steps (at most `max_steps`; up to `until_action_id`):
    /// {steps: [{index, actionId, action, log, snap}], loaded, total, done, redo, stopped?}
    pub fn replay(&mut self, max_steps: u32, until_action_id: Option<f64>) -> String {
        let until = until_action_id.map(|x| x as i64);
        let mut steps = Vec::new();
        while !self.inner.done() && (steps.len() as u32) < max_steps {
            if let Some(u) = until {
                let next = self.inner.raw[self.inner.pos].get("id").and_then(Value::as_i64).unwrap_or(0);
                if next > u {
                    break;
                }
            }
            // `at`'s checkpoints: at most 8, spaced by the game's length (at least 100 actions apart); each without
            // its raw actions (`at` takes this game's), so a long game holds ~10 MB, not ~100 (§10)
            let spacing = self.inner.total().div_ceil(MAX_CHECKPOINTS).max(100);
            if self.inner.pos > 0 && self.inner.pos % spacing == 0 && self.checkpoints.last().map_or(true, |c| c.pos < self.inner.pos) {
                let mut c = self.inner.clone();
                c.raw = Vec::new();
                c.kept = Vec::new();
                self.checkpoints.push(c);
            }
            if let Some(s) = self.inner.process_next() {
                steps.push(super::view::step_json(&self.inner, &s));
            }
        }
        let mut out = json!({
            "steps": steps, "loaded": self.inner.pos, "total": self.inner.total(), "done": self.inner.done(),
            "redo": self.inner.redo,
        });
        if let Some(s) = &self.inner.stopped {
            out["stopped"] = json!({"actionId": s.action_id, "index": s.index, "message": s.message});
        }
        out.to_string()
    }

    /// The snapshot of the position now.
    pub fn view(&self) -> String {
        super::view::snapshot(&self.inner, self.inner.current_action_id).to_string()
    }

    #[wasm_bindgen(js_name = currentActionId)]
    pub fn current_action_id(&self) -> f64 {
        self.inner.current_action_id as f64
    }

    /// A live action as the site stores it (with its id): its step, or "null" for a chat message.
    pub fn append(&mut self, action_json: &str) -> Result<String, JsError> {
        let a = parse(action_json)?;
        let step = self.inner.append(&a).map_err(err)?;
        Ok(step.map_or(Value::Null, |s| super::view::step_json(&self.inner, &s)).to_string())
    }

    /// A move as the page would post it: {posted, step}.
    pub fn act(&mut self, action_json: &str) -> Result<String, JsError> {
        let a = parse(action_json)?;
        let (posted, step) = self.inner.act(&a).map_err(err)?;
        let step = step.map_or(Value::Null, |s| super::view::step_json(&self.inner, &s));
        Ok(json!({"posted": posted, "step": step}).to_string())
    }

    /// As `act`, on a copy: {posted} (the site action to post, no auto actions), or throws `refused:`. Nothing
    /// changes here: the worker posts it, then appends the stored action the server returns (§7.1).
    pub fn check(&self, action_json: &str) -> Result<String, JsError> {
        let a = parse(action_json)?;
        let mut copy = self.inner.clone();
        copy.site_log = false;
        let (posted, _step) = copy.act(&a).map_err(err)?;
        Ok(json!({"posted": posted}).to_string())
    }

    /// The Auto tab's facts for a player (living-board ProgramInfo), or "null".
    #[wasm_bindgen(js_name = programInfo)]
    pub fn program_info(&self, player_id: u32) -> String {
        super::details::program_info(&self.inner, player_id).to_string()
    }

    /// The shadow key v1 of the position now (§5.3): the same JSON `site_engine state` prints as `state`.
    #[wasm_bindgen(js_name = shadowKey)]
    pub fn shadow_key(&mut self) -> String {
        super::shadow::key_now(&mut self.inner).to_string()
    }

    /// The move menu (RawMenu) of a player, or "null" when they needn't act.
    pub fn menu(&mut self, player_id: u32) -> String {
        super::menu::menu(&mut self.inner, player_id).to_string()
    }

    #[wasm_bindgen(js_name = tileOptions)]
    pub fn tile_options(&mut self, hex: &str, via: Option<String>) -> Result<String, JsError> {
        super::menu::tile_options(&mut self.inner, hex, via.as_deref()).map(|v| v.to_string()).map_err(err)
    }

    #[wasm_bindgen(js_name = autoRoute)]
    pub fn auto_route(&mut self) -> String {
        super::menu::auto_route(&mut self.inner).to_string()
    }

    pub fn details(&mut self, parts: &str) -> Result<String, JsError> {
        let p = parse(parts)?;
        super::details::details(&mut self.inner, &p).map(|v| v.to_string()).map_err(err)
    }

    #[wasm_bindgen(js_name = restartInfo)]
    pub fn restart_info(&self) -> String {
        super::details::restart_info(&self.inner).to_string()
    }

    #[wasm_bindgen(js_name = lastRun)]
    pub fn last_run(&self, sym: Option<String>) -> String {
        super::details::last_run_info(&self.inner, sym.as_deref()).to_string()
    }

    pub fn meta(&self) -> String {
        super::details::meta(&self.inner).to_string()
    }

    #[wasm_bindgen(js_name = legalActions)]
    pub fn legal_actions(&mut self) -> String {
        super::menu::legal_actions(&mut self.inner).to_string()
    }

    /// A copy at a past position (from the nearest checkpoint, then replay).
    pub fn at(&self, action_id: f64) -> Result<WasmGame, JsError> {
        let id = action_id as i64;
        let base = self
            .checkpoints
            .iter()
            .rev()
            .find(|c| c.current_action_id <= id && c.pos <= self.inner.pos)
            .cloned();
        let mut g = match base {
            Some(c) => c,
            None => {
                let mut data = self.inner.meta.clone();
                data["actions"] = json!(self.inner.raw);
                SiteGame::load(&data).map_err(err)?
            }
        };
        // replay the raw actions up to `id`, but never past what this game has seen
        g.raw = self.inner.raw.clone();
        g.kept = self.inner.kept.clone();
        g.run_to(Some(id));
        Ok(WasmGame { inner: g, checkpoints: Vec::new() })
    }
}
