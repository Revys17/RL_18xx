//! The native harness's comparisons with the Opal goldens (B's `test/rust/golden.cjs`, docs/WASM_ENGINE_SPIKE.md
//! §4.3): field-path diffs of snapshots, actions (tracks) and menus, and log lines, counted per path with the first
//! example. Not compiled for wasm32.

use super::game::{SiteGame, StepOut};
use serde_json::{json, Map, Value};
use std::collections::BTreeMap;

#[derive(Default)]
pub struct Stats {
    pub games: usize,
    pub positions: usize,
    pub snaps_equal: usize,
    pub actions: usize,
    pub actions_equal: usize,
    pub tracks: usize,
    pub tracks_equal: usize,
    pub log_steps: usize,
    pub log_steps_equal: usize,
    pub log_lines: usize,
    pub log_lines_equal: usize,
    pub menus: usize,
    pub menus_equal: usize,
    pub tile_opts: usize,
    pub tile_opts_equal: usize,
    pub routes: usize,
    pub routes_equal: usize,
    pub routes_rust_higher: usize,
    pub routes_rust_lower: usize,
    pub routes_recorded_higher: usize,
    pub misaligned: usize,
    pub static_infos: usize,
    pub static_infos_equal: usize,
    pub restarts: usize,
    pub restarts_equal: usize,
    /// field path -> (count, first example)
    pub diffs: BTreeMap<String, (usize, String)>,
    pub log_diffs: BTreeMap<String, (usize, String)>,
    /// the lines Rust writes that Opal's step doesn't (shape -> count, first example)
    pub log_extra: BTreeMap<String, (usize, String)>,
}

impl Stats {
    fn note(map: &mut BTreeMap<String, (usize, String)>, path: String, example: String) {
        let e = map.entry(path).or_insert((0, example));
        e.0 += 1;
    }

    pub fn print(&self) {
        let pct = |a: usize, b: usize| if b == 0 { 100.0 } else { 100.0 * a as f64 / b as f64 };
        println!("goldens: {} games, {} positions ({} misaligned)", self.games, self.positions, self.misaligned);
        println!("  snapshots equal: {}/{} ({:.2}%)", self.snaps_equal, self.positions, pct(self.snaps_equal, self.positions));
        println!("  actions equal:   {}/{} ({:.2}%)", self.actions_equal, self.actions, pct(self.actions_equal, self.actions));
        println!("  tracks equal:    {}/{} ({:.2}%)", self.tracks_equal, self.tracks, pct(self.tracks_equal, self.tracks));
        println!("  menus equal:     {}/{} ({:.2}%)", self.menus_equal, self.menus, pct(self.menus_equal, self.menus));
        println!("  tile options equal: {}/{} ({:.2}%)", self.tile_opts_equal, self.tile_opts, pct(self.tile_opts_equal, self.tile_opts));
        println!("  restart info equal: {}/{} ({:.2}%)", self.restarts_equal, self.restarts, pct(self.restarts_equal, self.restarts));
        println!("  auto routes: {} compared, {} equal revenue, {} Rust higher, {} Rust lower; recorded run above Rust's: {}",
            self.routes, self.routes_equal, self.routes_rust_higher, self.routes_rust_lower, self.routes_recorded_higher);
        println!("  log steps equal: {}/{} ({:.2}%), lines {}/{} ({:.2}%)", self.log_steps_equal, self.log_steps,
            pct(self.log_steps_equal, self.log_steps), self.log_lines_equal, self.log_lines, pct(self.log_lines_equal, self.log_lines));
        let mut v: Vec<_> = self.diffs.iter().collect();
        v.sort_by(|a, b| b.1 .0.cmp(&a.1 .0));
        println!("  mismatches by field path:");
        for (k, (n, ex)) in v.iter().take(40) {
            println!("    {n:6}  {k}   {}", ex.chars().take(160).collect::<String>());
        }
        let mut l: Vec<_> = self.log_diffs.iter().collect();
        l.sort_by(|a, b| b.1 .0.cmp(&a.1 .0));
        println!("  log line mismatches (Opal's line shape):");
        for (k, (n, ex)) in l.iter().take(40) {
            println!("    {n:6}  {k}   {}", ex.chars().take(160).collect::<String>());
        }
        let mut x: Vec<_> = self.log_extra.iter().collect();
        x.sort_by(|a, b| b.1 .0.cmp(&a.1 .0));
        println!("  lines Rust writes that Opal's step doesn't ({}):", x.iter().map(|e| e.1 .0).sum::<usize>());
        for (k, (n, ex)) in x.iter().take(40) {
            println!("    {n:6}  {k}   {}", ex.chars().take(160).collect::<String>());
        }
    }

    pub fn to_json(&self) -> Value {
        let m = |x: &BTreeMap<String, (usize, String)>| -> Value {
            let mut v: Vec<_> = x.iter().collect();
            v.sort_by(|a, b| b.1 .0.cmp(&a.1 .0));
            json!(v.iter().map(|(k, (n, ex))| json!({"path": k, "n": n, "example": ex})).collect::<Vec<_>>())
        };
        json!({
            "games": self.games, "positions": self.positions, "snapsEqual": self.snaps_equal,
            "actions": self.actions, "actionsEqual": self.actions_equal, "tracks": self.tracks, "tracksEqual": self.tracks_equal,
            "menus": self.menus, "menusEqual": self.menus_equal,
            "tileOptions": self.tile_opts, "tileOptionsEqual": self.tile_opts_equal, "routes": self.routes, "routesEqual": self.routes_equal,
            "routesRustHigher": self.routes_rust_higher, "routesRustLower": self.routes_rust_lower, "recordedAboveRust": self.routes_recorded_higher, "logSteps": self.log_steps, "logStepsEqual": self.log_steps_equal,
            "logLines": self.log_lines, "logLinesEqual": self.log_lines_equal, "misaligned": self.misaligned,
            "staticInfos": self.static_infos, "staticInfosEqual": self.static_infos_equal,
            "diffs": m(&self.diffs), "logDiffs": m(&self.log_diffs), "logExtra": m(&self.log_extra),
        })
    }
}

/// The golden of a game id (index.json maps keys to games: a golden stands for every copy of its action stream).
pub fn load_golden(dir: &str, gid: &str) -> Option<Value> {
    let idx: Value = serde_json::from_str(&std::fs::read_to_string(format!("{dir}/index.json")).ok()?).ok()?;
    let games = idx.get("games")?.as_object()?;
    let key = games.iter().find(|(_, g)| g.get("gid").map(|v| v.to_string().trim_matches('"').to_string()) == Some(gid.to_string()))?.0;
    let out = std::process::Command::new("gzip").arg("-dc").arg(format!("{dir}/{key}.json.gz")).output().ok()?;
    serde_json::from_slice(&out.stdout).ok()
}

/// A line's shape: names and numbers left out ("PRR runs a 4 train for $60: D14-E11" -> "# runs a # train for $#: #").
pub fn line_shape(s: &str) -> String {
    let mut out = String::new();
    let mut prev_hash = false;
    for w in s.split(' ') {
        let alnum = w.chars().any(|c| c.is_ascii_digit());
        let tok = if alnum { "#".to_string() } else { w.to_string() };
        if tok == "#" && prev_hash {
            continue;
        }
        prev_hash = tok == "#";
        if !out.is_empty() {
            out.push(' ');
        }
        out.push_str(&tok);
    }
    out
}

fn key_of(v: &Value) -> Option<String> {
    for k in ["sym", "id", "company", "corp", "kind", "hex", "train"] {
        if let Some(x) = v.get(k) {
            return Some(x.to_string().trim_matches('"').to_string());
        }
    }
    None
}

/// Field-path differences between `a` (Rust) and `b` (Opal), keyed by stable element keys where arrays hold objects.
pub fn diff(path: &str, a: &Value, b: &Value, out: &mut Vec<(String, String)>) {
    match (a, b) {
        (Value::Object(x), Value::Object(y)) => {
            let mut keys: Vec<&String> = x.keys().chain(y.keys()).collect();
            keys.sort();
            keys.dedup();
            for k in keys {
                let p = if path.is_empty() { k.clone() } else { format!("{path}.{k}") };
                match (x.get(k), y.get(k)) {
                    (Some(u), Some(v)) => diff(&p, u, v, out),
                    (Some(u), None) => {
                        if !u.is_null() {
                            out.push((p, format!("rust {} / opal absent", trunc(u))))
                        }
                    }
                    (None, Some(v)) => {
                        if !v.is_null() {
                            out.push((p, format!("rust absent / opal {}", trunc(v))))
                        }
                    }
                    _ => {}
                }
            }
        }
        (Value::Array(x), Value::Array(y)) => {
            let keyed = x.iter().chain(y.iter()).all(|v| v.is_object() && key_of(v).is_some()) && !x.is_empty() && !y.is_empty();
            if keyed && x.len() == y.len() {
                let xk: Vec<Option<String>> = x.iter().map(key_of).collect();
                let yk: Vec<Option<String>> = y.iter().map(key_of).collect();
                if xk == yk {
                    for (i, (u, v)) in x.iter().zip(y.iter()).enumerate() {
                        let k = xk[i].clone().unwrap_or_default();
                        diff(&format!("{path}[{k}]"), u, v, out);
                    }
                    return;
                }
            }
            if x.len() != y.len() || x != y {
                if x.len() == y.len() && !keyed {
                    for (i, (u, v)) in x.iter().zip(y.iter()).enumerate() {
                        let p = if u.is_object() || u.is_array() { format!("{path}[]") } else { format!("{path}[]") };
                        let _ = i;
                        diff(&p, u, v, out);
                    }
                } else {
                    out.push((path.to_string(), format!("rust {} / opal {}", trunc(a), trunc(b))));
                }
            }
        }
        _ => {
            let num_eq = a.as_f64().zip(b.as_f64()).is_some_and(|(p, q)| p == q);
            if a != b && !num_eq {
                out.push((path.to_string(), format!("rust {} / opal {}", trunc(a), trunc(b))));
            }
        }
    }
}

fn trunc(v: &Value) -> String {
    let s = v.to_string();
    if s.len() > 120 {
        format!("{}...", &s[..120])
    } else {
        s
    }
}

pub struct GameCompare<'a> {
    golden: &'a Value,
    steps: Vec<&'a Value>,
}

impl<'a> GameCompare<'a> {
    pub fn new(golden: &'a Value) -> Self {
        let steps = golden.get("steps").and_then(Value::as_array).map(|v| v.iter().collect()).unwrap_or_default();
        GameCompare { golden, steps }
    }

    fn position(&self, g: &mut SiteGame, gold: &Value, snap: &Value, stats: &mut Stats, gid: &str) {
        stats.positions += 1;
        let mut d = Vec::new();
        if let Some(gs) = gold.get("snap") {
            diff("snap", snap, gs, &mut d);
        }
        if d.is_empty() {
            stats.snaps_equal += 1;
        }
        for (p, ex) in d {
            Stats::note(&mut stats.diffs, normalize(&p), format!("{gid}@{}: {ex}", gold.get("actionId").unwrap_or(&Value::Null)));
        }
        if let Some(Value::Array(tiles)) = gold.get("tiles") {
            for t in tiles {
                let hex = t.get("hex").and_then(Value::as_str).unwrap_or("");
                let via = t.get("via").and_then(Value::as_str);
                stats.tile_opts += 1;
                let mine = super::menu::tile_options(g, hex, via).unwrap_or_else(|e| json!({"error": e}));
                let mut theirs = t.get("options").cloned().unwrap_or(Value::Null);
                strip_pack_text(&mut theirs);
                let mut d = Vec::new();
                diff("tiles", &mine, &theirs, &mut d);
                if d.is_empty() {
                    stats.tile_opts_equal += 1;
                }
                for (p, ex) in d {
                    Stats::note(&mut stats.diffs, normalize(&p), format!("{gid}@{} {hex}: {ex}", gold.get("actionId").unwrap_or(&Value::Null)));
                }
            }
        }
        if let Some(r) = gold.get("routes") {
            let mine = super::menu::auto_route(g);
            let theirs = r.get("revenue").and_then(Value::as_i64);
            let ours = mine.get("revenue").and_then(Value::as_i64);
            let recorded = r.get("recorded").and_then(Value::as_i64);
            if let (Some(t), Some(o)) = (theirs, ours) {
                stats.routes += 1;
                if t == o {
                    stats.routes_equal += 1;
                } else if o > t {
                    stats.routes_rust_higher += 1;
                } else {
                    stats.routes_rust_lower += 1;
                    Stats::note(&mut stats.diffs, "routes.revenue(lower)".into(), format!("{gid}@{}: rust {o} / opal {t}", gold.get("actionId").unwrap_or(&Value::Null)));
                }
            }
            if let (Some(rec), Some(o)) = (recorded, ours) {
                if rec > o {
                    stats.routes_recorded_higher += 1;
                    Stats::note(&mut stats.diffs, "routes.recorded>rust".into(), format!("{gid}@{}: recorded {rec} / rust {o}", gold.get("actionId").unwrap_or(&Value::Null)));
                }
            }
        }
        if let Some(gr) = gold.get("restart") {
            stats.restarts += 1;
            let mine = super::details::restart_info(g);
            let mut d = Vec::new();
            diff("restart", &mine, gr, &mut d);
            if d.is_empty() {
                stats.restarts_equal += 1;
            }
            for (p, ex) in d {
                Stats::note(&mut stats.diffs, normalize(&p), format!("{gid}@{}: {ex}", gold.get("actionId").unwrap_or(&Value::Null)));
            }
        }
        if let Some(Value::Object(menus)) = gold.get("menus") {
            for (pid, gm) in menus {
                let Ok(pid) = pid.parse::<u32>() else { continue };
                stats.menus += 1;
                let mine = super::menu::menu(g, pid);
                let mut gm = gm.clone();
                strip_pack_text(&mut gm);
                let mut md = Vec::new();
                diff("menu", &mine, &gm, &mut md);
                if md.is_empty() {
                    stats.menus_equal += 1;
                }
                for (p, ex) in md {
                    Stats::note(&mut stats.diffs, normalize(&p), format!("{gid}@{}: {ex}", gold.get("actionId").unwrap_or(&Value::Null)));
                }
            }
        }
    }

    fn logs(&self, mine: &[String], gold: &Value, stats: &mut Stats, gid: &str) {
        let theirs: Vec<String> = gold.get("log").and_then(Value::as_array).map(|v| v.iter().filter_map(|x| x.as_str().map(str::to_string)).collect()).unwrap_or_default();
        stats.log_steps += 1;
        if mine == theirs.as_slice() {
            stats.log_steps_equal += 1;
        }
        for line in &theirs {
            stats.log_lines += 1;
            if mine.contains(line) {
                stats.log_lines_equal += 1;
            } else {
                Stats::note(&mut stats.log_diffs, line_shape(line), format!("{gid}@{}: {line} | rust: {:?}", gold.get("actionId").unwrap_or(&Value::Null), mine));
            }
        }
        // SITE_LOG_DIFFS=<file>: every step whose lines differ, as JSON lines (a tool's view, nothing reads it back)
        if mine != theirs.as_slice() {
            if let Ok(path) = std::env::var("SITE_LOG_DIFFS") {
                use std::io::Write;
                if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
                    let _ = writeln!(f, "{}", serde_json::json!({"gid": gid, "at": gold.get("actionId"), "opal": theirs, "rust": mine}));
                }
            }
        }
        for line in mine.iter().filter(|l| !theirs.contains(l)) {
            Stats::note(&mut stats.log_extra, line_shape(line), format!("{gid}@{}: {line} | opal: {:?}", gold.get("actionId").unwrap_or(&Value::Null), theirs));
        }
    }

    pub fn start(&mut self, g: &mut SiteGame, stats: &mut Stats) {
        stats.games += 1;
        let gid = self.golden.get("gid").map(|v| v.to_string()).unwrap_or_default();
        // StaticInfo: the engine's part (the pack's names, colours and texts aside)
        if let Some(info) = self.golden.get("info") {
            let mine = super::view::static_info(g);
            let mut diffs = Vec::new();
            for k in ["market", "trainTypes", "phases", "bankStart", "certLimit"] {
                diff(&format!("info.{k}"), &mine[k], &info[k], &mut diffs);
            }
            let pick = |v: &Value, keys: &[&str]| -> Value {
                json!(v.as_array().map(|xs| xs.iter().map(|x| keys.iter().map(|k| (k.to_string(), x[*k].clone())).collect::<Map<String, Value>>()).collect::<Vec<_>>()))
            };
            diff("info.companies", &pick(&mine["companies"], &["sym", "value", "revenue", "blocks", "comesWith"]),
                 &pick(&info["companies"], &["sym", "value", "revenue", "blocks", "comesWith"]), &mut diffs);
            diff("info.players", &pick(&mine["players"], &["id", "name"]), &pick(&info["players"], &["id", "name"]), &mut diffs);
            stats.static_infos += 1;
            if diffs.is_empty() {
                stats.static_infos_equal += 1;
            }
            for (path, ex) in diffs {
                Stats::note(&mut stats.diffs, normalize(&path), format!("{gid}: {ex}"));
            }
        }
        if let Some(s0) = self.steps.first().copied() {
            let snap = super::view::snapshot(g, 0);
            let log = g.start_log.clone();
            self.logs(&log, s0, stats, &gid);
            self.position(g, s0, &snap, stats, &gid);
        }
    }

    pub fn step(&mut self, g: &mut SiteGame, step: &StepOut, snap: &Value, stats: &mut Stats) {
        let gid = self.golden.get("gid").map(|v| v.to_string()).unwrap_or_default();
        let Some(gold) = self.steps.get(step.index).copied() else { return };
        if gold.get("actionId").and_then(Value::as_i64) != Some(step.action_id) {
            stats.misaligned += 1;
            return;
        }
        self.logs(&step.log, gold, stats, &gid);
        // the action as the board shows it (tracks included)
        let mine = super::view::action_lite(g, &step.action);
        if let Some(ga) = gold.get("action") {
            stats.actions += 1;
            let mut d = Vec::new();
            diff("action", &mine, ga, &mut d);
            if d.is_empty() {
                stats.actions_equal += 1;
            }
            if ga.get("type").and_then(Value::as_str) == Some("run_routes") {
                for (i, r) in ga.get("routes").and_then(Value::as_array).cloned().unwrap_or_default().iter().enumerate() {
                    if let Some(t) = r.get("track") {
                        stats.tracks += 1;
                        if mine.pointer(&format!("/routes/{i}/track")) == Some(t) {
                            stats.tracks_equal += 1;
                        }
                    }
                }
            }
            for (p, ex) in d {
                Stats::note(&mut stats.diffs, normalize(&p), format!("{gid}@{}: {ex}", step.action_id));
            }
        }
        self.position(g, gold, snap, stats, &gid);
    }
}

/// A field path with its element keys generalised where they vary (players' ids, hexes).
fn normalize(p: &str) -> String {
    let mut out = String::new();
    let mut chars = p.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '[' {
            let mut k = String::new();
            for d in chars.by_ref() {
                if d == ']' {
                    break;
                }
                k.push(d);
            }
            if k.chars().all(|c| c.is_ascii_digit()) && !k.is_empty() {
                out.push_str("[#]");
            } else {
                out.push('[');
                out.push_str(&k);
                out.push(']');
            }
        } else {
            out.push(c);
        }
    }
    // hexes.<id> -> hexes.*
    let parts: Vec<&str> = out.split('.').collect();
    let mut res = Vec::new();
    let mut i = 0;
    while i < parts.len() {
        res.push(parts[i].to_string());
        if (parts[i] == "hexes" || parts[i].ends_with("certs") || parts[i].ends_with("certCount") || parts[i] == "menus" || parts[i] == "sold" || parts[i] == "programs" || parts[i] == "result") && i + 1 < parts.len() {
            res.push("*".into());
            i += 1;
        }
        i += 1;
    }
    let _ = Map::<String, Value>::new();
    res.join(".")
}

/// Words the title pack supplies (a private's description) and the Opal action's own timestamp: not the engine's.
fn strip_pack_text(v: &mut Value) {
    match v {
        Value::Object(m) => {
            m.remove("desc");
            if m.contains_key("type") {
                m.remove("created_at");
            }
            for (_, x) in m.iter_mut() {
                strip_pack_text(x);
            }
        }
        Value::Array(xs) => xs.iter_mut().for_each(strip_pack_text),
        _ => {}
    }
}
