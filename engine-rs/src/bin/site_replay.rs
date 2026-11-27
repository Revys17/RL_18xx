//! The spike's native harness (docs/WASM_ENGINE_SPIKE.md §4.3): replays site games strictly in the site layer and
//! reports where they stop, whether finished games end with the site's result, and how long it takes.
//!
//! cargo run --release --no-default-features --features site --bin site_replay -- [<corpus dir or game .json>...]
//!     [--report <json>] [--views] [--goldens <dir>] [--verbose] [--digest] [--title 1830|1867]
//!     [--acts <cases.jsonl> [--acts-every N] [--acts-tiles N]]   (§8.3, K11)
//!     [--act-check]   (1867: every recorded move checked by the generator, the step autos predicted)
//!     [--authority]   (step 3a: every kept move decided at its position as `site_engine move` decides it)
//!     --generate N [--seed S] [--moves M] [--out DIR]   (1867 generated games, SITE_ENGINE_RS_1867.md §10.4)
//!
//! `--views`: also build every step's snapshot (the cost of a replay with views). `--goldens <dir>`: compare with
//! the Opal goldens (B's `test/rust/golden.cjs`) where they exist.

use engine_rs::site::game::SiteGame;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

fn corpus_files(args: &[String]) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for a in args {
        let p = Path::new(a);
        if p.is_dir() {
            for e in std::fs::read_dir(p).unwrap().flatten() {
                let q = e.path();
                if q.extension().and_then(|x| x.to_str()) == Some("json") {
                    out.push(q);
                }
            }
        } else {
            out.push(p.to_path_buf());
        }
    }
    let num = |p: &PathBuf| p.file_stem().and_then(|s| s.to_str()).and_then(|s| s.parse::<i64>().ok()).unwrap_or(i64::MAX);
    out.sort_by_key(|p| (num(p), p.clone()));
    out
}

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut inputs = Vec::new();
    let mut report: Option<String> = None;
    let mut goldens: Option<String> = None;
    let mut views = false;
    let mut digest = false;
    let mut acts: Option<String> = None;
    let mut acts_every = 10usize;
    let mut acts_tiles = 5usize;
    let mut verbose = false;
    let mut only_title: Option<String> = None;
    let mut dump: Option<i64> = None;
    let mut act_check = false;
    let mut authority = false;
    let mut generate: Option<usize> = None;
    let mut gen_seed: u64 = 1;
    let mut gen_moves: usize = 3000;
    let mut gen_out: Option<String> = None;
    let mut gen_dead_ends = false;
    let mut i = 0;
    while i < argv.len() {
        match argv[i].as_str() {
            "--report" => {
                report = argv.get(i + 1).cloned();
                i += 1;
            }
            "--goldens" => {
                goldens = argv.get(i + 1).cloned();
                i += 1;
            }
            "--views" => views = true,
            "--digest" => digest = true,
            "--acts" => {
                acts = argv.get(i + 1).cloned();
                i += 1;
            }
            "--acts-every" => {
                acts_every = argv.get(i + 1).and_then(|x| x.parse().ok()).unwrap_or(10);
                i += 1;
            }
            "--acts-tiles" => {
                acts_tiles = argv.get(i + 1).and_then(|x| x.parse().ok()).unwrap_or(5);
                i += 1;
            }
            "--verbose" => verbose = true,
            "--act-check" => act_check = true,
            "--authority" => authority = true,
            "--ruby-dead-ends" => gen_dead_ends = true,
            "--generate" => {
                generate = argv.get(i + 1).and_then(|x| x.parse::<usize>().ok());
                i += 1;
            }
            "--seed" => {
                gen_seed = argv.get(i + 1).and_then(|x| x.parse::<u64>().ok()).unwrap_or(1);
                i += 1;
            }
            "--moves" => {
                gen_moves = argv.get(i + 1).and_then(|x| x.parse::<usize>().ok()).unwrap_or(3000);
                i += 1;
            }
            "--out" => {
                gen_out = argv.get(i + 1).cloned();
                i += 1;
            }
            "--dump" => {
                dump = argv.get(i + 1).and_then(|x| x.parse::<i64>().ok());
                i += 1;
            }
            "--title" => {
                only_title = argv.get(i + 1).cloned();
                i += 1;
            }
            a => inputs.push(a.to_string()),
        }
        i += 1;
    }
    if let Some(n) = generate {
        // docs/SITE_ENGINE_RS_1867.md §10.4: generated 1867 games as site JSON (players 3-6 in turn)
        let out = gen_out.clone().unwrap_or_else(|| ".".into());
        std::fs::create_dir_all(&out).unwrap();
        for k in 0..n {
            let seed = gen_seed + k as u64;
            let players = 3 + (k % 4);
            let t0 = Instant::now();
            let game = engine_rs::site::g1867::generate::generate(seed, players, gen_moves, 0.8, gen_dead_ends);
            let info = game["generated"].clone();
            std::fs::write(format!("{out}/gen-{seed}.json"), game.to_string()).unwrap();
            println!("{}", json!({"seed": seed, "players": players, "ms": t0.elapsed().as_millis() as u64, "info": info}));
        }
        return;
    }
    if inputs.is_empty() {
        inputs.push(std::env::var("CORPUS").unwrap_or_else(|_| {
            "/tmp/claude-1000/-home-revys-workspace/252285c2-5545-52b0-bcda-165a3e4e9ceb/scratchpad/wasm-spike/g9301".into()
        }));
    }
    let files = corpus_files(&inputs);
    // --acts <cases.jsonl>: the act-level cases of §8.3 (K11) of the distinct games
    if let Some(path) = acts {
        use std::io::Write;
        let mut f = std::io::BufWriter::new(std::fs::File::create(&path).expect("cases file"));
        let sampling = engine_rs::site::acts::Sampling { every: acts_every, tile_every: acts_tiles };
        let mut seen = std::collections::HashSet::new();
        let (mut games_n, mut cases_n, mut dup) = (0usize, 0usize, 0usize);
        let mut verdicts: BTreeMap<String, usize> = BTreeMap::new();
        for path in &files {
            let Ok(text) = std::fs::read_to_string(path) else { continue };
            let Ok(data) = serde_json::from_str::<Value>(&text) else { continue };
            let title = data.get("title").and_then(Value::as_str).unwrap_or("");
            if !engine_rs::site::TITLES.contains(&title) || only_title.as_deref().is_some_and(|t| t != title) {
                continue;
            }
            // distinct games: the players, rules and action stream (ids, users and times aside)
            let mut stream = data.get("actions").cloned().unwrap_or(json!([]));
            if let Some(xs) = stream.as_array_mut() {
                for a in xs.iter_mut() {
                    if let Some(m) = a.as_object_mut() {
                        for k in ["id", "user", "created_at"] {
                            m.remove(k);
                        }
                    }
                }
            }
            let sig = json!([data.get("players").map(|ps| ps.as_array().map(|v| v.iter().map(|p| p["id"].clone()).collect::<Vec<_>>())), data.pointer("/settings/optional_rules"), stream]).to_string();
            if !seen.insert(sig) {
                dup += 1;
                continue;
            }
            let gid = path.file_stem().and_then(|s| s.to_str()).unwrap_or("?").to_string();
            let mut emit = |v: Value| {
                *verdicts.entry(v["rust"]["verdict"].as_str().unwrap_or("?").to_string()).or_default() += 1;
                writeln!(f, "{v}").unwrap();
            };
            match engine_rs::site::acts::game_cases(&data, &gid, &sampling, &mut emit) {
                Ok(n) => {
                    games_n += 1;
                    cases_n += n;
                }
                Err(e) => eprintln!("{gid}: {e}"),
            }
        }
        println!("acts: {games_n} distinct games ({dup} duplicates), {cases_n} cases {verdicts:?} -> {path}");
        return;
    }
    // --digest: each game's replay digest (§8.2 native vs wasm; scripts/site_engine/pkg_smoke.cjs --digest)
    if digest {
        for path in &files {
            let Ok(text) = std::fs::read_to_string(path) else { continue };
            let Ok(data) = serde_json::from_str::<Value>(&text) else { continue };
            let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("?");
            match engine_rs::site::shadow::replay_digest(&data) {
                Ok(d) => println!("{name} {} {}", d["digest"].as_str().unwrap_or(""), d["steps"]),
                Err(e) => println!("{name} error {}", e.split(':').next().unwrap_or("")),
            }
        }
        return;
    }
    // --at <gid>:<action id>: the engine's state after that action (debugging a stop)
    if let Some(at) = std::env::var("SITE_AT").ok() {
        let (gid, aid) = at.split_once(':').unwrap();
        let path = files.iter().find(|p| p.file_stem().and_then(|s| s.to_str()) == Some(gid)).expect("game");
        let data: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        let mut g = SiteGame::load(&data).unwrap();
        g.run_to(Some(aid.parse().unwrap()));
        let snap = engine_rs::site::view::snapshot(&g, 0);
        println!("stopped: {:?}", g.stopped);
        println!("round {} step {:?} current {} active {}", snap["round"], snap["stepLabel"], snap["current"], snap["activePlayers"]);
        println!("legal types: {:?}", engine_rs::site::debug_legal_types(&mut g));
        println!("players: {}", snap["players"]);
        if let Ok(hex) = std::env::var("SITE_TILES") {
            println!("tiles {hex}: {:?}", engine_rs::site::menu::tile_options(&mut g, &hex, None));
            println!("upgradeable: {:?}", engine_rs::site::debug_upgradeable(&g, &hex));
        }
        for pid in engine_rs::site::view::active_players(&g) {
            println!("menu {pid}: {}", engine_rs::site::menu::menu(&mut g, pid));
        }
        return;
    }

    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut stops: BTreeMap<String, (usize, Vec<String>)> = BTreeMap::new();
    let mut results: BTreeMap<String, usize> = BTreeMap::new();
    let mut result_diffs: Vec<Value> = Vec::new();
    let mut games: Vec<Value> = Vec::new();
    let mut total_actions = 0usize;
    let mut total_steps = 0usize;
    let mut engine_ms = 0f64;
    let mut slowest: Vec<(f64, String, usize)> = Vec::new();
    let mut golden_stats = engine_rs::site::compare::Stats::default();

    for path in &files {
        let text = match std::fs::read_to_string(path) {
            Ok(t) => t,
            Err(_) => continue,
        };
        let data: Value = match serde_json::from_str(&text) {
            Ok(v) => v,
            Err(e) => {
                *counts.entry("unreadable".into()).or_default() += 1;
                if verbose {
                    eprintln!("{}: {e}", path.display());
                }
                continue;
            }
        };
        let gid = data.get("id").map(|v| v.to_string()).unwrap_or_else(|| path.display().to_string());
        let title = data.get("title").and_then(Value::as_str).unwrap_or("");
        if !engine_rs::site::TITLES.contains(&title) || only_title.as_deref().is_some_and(|t| t != title) {
            *counts.entry("other title".into()).or_default() += 1;
            continue;
        }
        // a golden's game id: the site's id, or "<corpus dir>:<file stem>" (production and fixture games)
        let alt_gid = format!(
            "{}:{}",
            path.parent().and_then(|p| p.file_name()).and_then(|x| x.to_str()).unwrap_or(""),
            path.file_stem().and_then(|x| x.to_str()).unwrap_or("")
        );
        let t0 = Instant::now();
        let mut g = match SiteGame::load(&data) {
            Ok(g) => g,
            Err(e) => {
                *counts.entry(format!("load: {e}")).or_default() += 1;
                continue;
            }
        };
        if authority {
            // step 3a (docs/SITE_ENGINE_RS_STEP3.md §12.1 A1): every kept move decided at its position as `move`
            // decides it, against the record (auto actions, a run's fields)
            g.site_log = false;
            let r = engine_rs::site::authority::corpus_check(&mut g);
            println!("{}", json!({"game": gid, "file": path.display().to_string(), "check": r}));
            continue;
        }
        if act_check {
            // 1867 (docs/SITE_ENGINE_RS_1867.md §3.2-§3.3): every recorded move the generator must accept on the position
            // before it, and Ruby's recorded step auto actions after it = the predicted ones (games without programs)
            let r = engine_rs::site::g1867::act::corpus_check(&mut g);
            println!("{}", json!({"game": gid, "file": path.display().to_string(), "check": r}));
            continue;
        }
        if let Some(at) = dump {
            // debugging: the position after action `at` (0: the start) as the board and the hands see it
            g.run_to(Some(at));
            let pids: Vec<u32> = g.players.iter().map(|p| p.0).collect();
            let mut menus = serde_json::Map::new();
            for pid in pids {
                menus.insert(pid.to_string(), engine_rs::site::menu::menu(&mut g, pid));
            }
            let snap = engine_rs::site::view::snapshot(&g, g.current_action_id);
            let key = engine_rs::site::shadow::key_now(&mut g);
            let moves = engine_rs::site::g1867::debug_moves(&mut g);
            let details = engine_rs::site::details::details(&mut g, &json!(["info", "tiles", "sheet"])).unwrap_or(Value::Null);
            let mut last = serde_json::Map::new();
            let lr0 = engine_rs::site::details::last_run_info(&g, None);
            for op in lr0.get("operators").and_then(Value::as_array).cloned().unwrap_or_default() {
                if let Some(sym) = op.get("sym").and_then(Value::as_str) {
                    last.insert(sym.to_string(), engine_rs::site::details::last_run_info(&g, Some(sym)));
                }
            }
            last.insert(String::new(), lr0);
            let mut programs = serde_json::Map::new();
            for p in g.players.iter().map(|p| p.0).collect::<Vec<_>>() {
                programs.insert(p.to_string(), engine_rs::site::details::program_info(&g, p));
            }
            println!("{}", json!({"snap": snap, "menus": menus, "key": key, "info": engine_rs::site::view::static_info(&g), "moves": moves,
                "details": details, "lastRun": last, "programs": programs, "meta": engine_rs::site::details::meta(&g),
                "restart": engine_rs::site::details::restart_info(&g)}));
            continue;
        }
        let golden = goldens.as_ref().and_then(|d| {
            engine_rs::site::compare::load_golden(d, &alt_gid).or_else(|| engine_rs::site::compare::load_golden(d, &gid))
        });
        let mut steps = 0usize;
        if views || golden.is_some() {
            let mut cmp = golden.as_ref().map(|gd| engine_rs::site::compare::GameCompare::new(gd));
            if let Some(c) = cmp.as_mut() {
                c.start(&mut g, &mut golden_stats);
            }
            while !g.done() {
                if let Some(step) = g.process_next() {
                    steps += 1;
                    let snap = engine_rs::site::view::snapshot(&g, step.action_id);
                    if let Some(c) = cmp.as_mut() {
                        c.step(&mut g, &step, &snap, &mut golden_stats);
                    }
                }
            }
        } else {
            g.run_to(None);
            steps = g.index;
        }
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        engine_ms += ms;
        total_actions += g.raw.len();
        total_steps += steps;
        slowest.push((ms, gid.clone(), g.raw.len()));

        let has_programs = g.raw.iter().any(|a| a.get("type").and_then(Value::as_str).is_some_and(|t| t.starts_with("program_")));
        let status = if let Some(s) = &g.stopped {
            let t = g.raw.iter().find(|a| a.get("id").and_then(Value::as_i64) == Some(s.action_id)).and_then(|a| a.get("type")).and_then(Value::as_str).unwrap_or("?").to_string();
            let msg: String = s.message.chars().take(90).collect();
            let key = format!("{t}: {msg}");
            let e = stops.entry(key).or_default();
            e.0 += 1;
            if e.1.len() < 6 {
                e.1.push(format!("{gid}@{}", s.action_id));
            }
            "stopped"
        } else {
            "ok"
        };
        *counts.entry(format!("{status}{}", if has_programs { " +programs" } else { "" })).or_default() += 1;

        // finished games: the site's stored result
        let site_finished = data.get("status").and_then(Value::as_str) == Some("finished");
        let site_result = data.get("result").filter(|r| r.as_object().is_some_and(|o| !o.is_empty()));
        if site_finished {
            let key = if site_result.is_none() {
                "no stored result".to_string()
            } else if g.stopped.is_some() {
                "stopped early".to_string()
            } else if !g.finished() {
                "not finished in Rust".to_string()
            } else {
                let mine: BTreeMap<String, i64> = g.result().iter().map(|(k, v)| (k.to_string(), *v as i64)).collect();
                let theirs: BTreeMap<String, i64> = site_result
                    .and_then(|r| r.as_object())
                    .map(|o| o.iter().map(|(k, v)| (k.clone(), v.as_i64().unwrap_or(0))).collect())
                    .unwrap_or_default();
                if mine == theirs {
                    format!("equal ({})", g.end_reason().unwrap_or("?"))
                } else {
                    result_diffs.push(json!({"game": gid, "rust": mine, "site": theirs}));
                    "differ".to_string()
                }
            };
            *results.entry(key).or_default() += 1;
        }
        games.push(json!({
            "id": gid, "actions": g.raw.len(), "steps": steps, "ms": (ms * 100.0).round() / 100.0,
            "stopped": g.stopped.as_ref().map(|s| json!({"actionId": s.action_id, "message": s.message})),
            "finished": g.finished(),
        }));
    }

    slowest.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    println!("games: {}", games.len());
    for (k, v) in &counts {
        println!("  {k}: {v}");
    }
    println!("first stops (grouped):");
    let mut sv: Vec<_> = stops.iter().collect();
    sv.sort_by(|a, b| b.1 .0.cmp(&a.1 .0));
    for (k, (n, ex)) in sv.iter().take(30) {
        println!("  {n:4}  {k}   e.g. {}", ex.join(" "));
    }
    println!("finished games vs the site's result:");
    for (k, v) in &results {
        println!("  {k}: {v}");
    }
    println!(
        "time: {:.0} ms for {} raw actions ({} steps{}), {:.1} us/action",
        engine_ms,
        total_actions,
        total_steps,
        if views { ", with snapshots" } else { "" },
        engine_ms * 1000.0 / total_actions.max(1) as f64
    );
    for (ms, gid, n) in slowest.iter().take(5) {
        println!("  slowest: game {gid}: {n} actions in {ms:.1} ms");
    }
    if goldens.is_some() {
        golden_stats.print();
    }
    if let Some(r) = report {
        let out = json!({
            "games": games.len(), "counts": counts, "stops": stops.iter().map(|(k, (n, ex))| json!({"key": k, "n": n, "examples": ex})).collect::<Vec<_>>(),
            "results": results, "resultDiffs": result_diffs, "engineMs": engine_ms, "rawActions": total_actions,
            "steps": total_steps, "views": views, "perGame": games, "goldens": golden_stats.to_json(),
        });
        std::fs::write(&r, serde_json::to_string_pretty(&out).unwrap()).unwrap();
        println!("report: {r}");
    }
}
