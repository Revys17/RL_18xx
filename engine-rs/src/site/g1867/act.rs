//! 1867's act and check (docs/SITE_ENGINE_RS_1867.md §3.2): the legal-move generator decides. Ruby checks a live
//! move with `@game.loading` false (tile legality, rotations, home locations, merge candidates, trade-in timing, ...),
//! which engine-rs's replay handler skips like Ruby's `load`; the generator (`g1867/legal.rs`) applies them. So a live
//! move must match a move of `g1867_legal_moves_with(RULES)` (Ruby's whole set: the site's own-only rule is the
//! menus'), on the importer's outcome key, with a ranged move's price on its range and step. Then it is processed
//! with the step auto actions Ruby would add (`autos.rs`).

use super::super::game::{refused, SiteGame};
use crate::g1867::legal::{Move, PriceRange};
use crate::title::CrossPlayerPolicy;
use serde_json::{json, Value};

fn s(v: &Value, k: &str) -> Option<String> {
    v.get(k).filter(|x| !x.is_null()).map(|x| x.as_str().map(str::to_string).unwrap_or_else(|| x.to_string()))
}

/// A tile or train instance's name ("57-1" -> "57", "5+5E-0" -> "5+5E").
fn base(id: &str) -> String {
    match id.rsplit_once('-') {
        Some((n, i)) if i.parse::<u32>().is_ok() => n.to_string(),
        _ => id.to_string(),
    }
}

/// "<tile id>-<city index>" as hex + city index (a laid tile's instance may be another of the same name).
fn city_norm(c: &str) -> String {
    let p: Vec<&str> = c.split('-').collect();
    if p.len() >= 3 {
        format!("{}-{}", p[0], p[p.len() - 1])
    } else {
        c.to_string()
    }
}

/// Where a train is (its holder's entity id, "depot" for the depot's), and its name.
fn train_place(g: &SiteGame, id: &str) -> Option<(String, String)> {
    let game = &g.game;
    if let Some(t) = game.depot.trains.iter().chain(game.depot.discarded.iter()).find(|t| t.id == id) {
        return Some(("depot".into(), t.name.clone()));
    }
    game.corporations.iter().find_map(|c| c.trains.iter().find(|t| t.id == id).map(|t| (c.sym.clone(), t.name.clone())))
}

/// Whether the site action `a` is the generated move `w` (its wire form; `r` its price range): the importer's
/// outcome key (type, entity; company / corporation; a bundle's corporation, percent and certificates; hex, tile name
/// and rotation; city by hex and index; a depot train by id, another company's by type and holder; kind; choice).
/// The price is checked apart (`price_ok`).
fn same_move(g: &SiteGame, a: &Value, w: &Value) -> bool {
    if s(a, "type") != s(w, "type") || s(a, "entity") != s(w, "entity") {
        return false;
    }
    for k in ["corporation", "company", "hex", "kind", "choice", "exchange"] {
        if let Some(wv) = s(w, k) {
            if s(a, k) != Some(wv) {
                return false;
            }
        }
    }
    if let Some(wp) = w.get("share_price") {
        let price = |v: &Value| -> Option<i64> {
            v.as_i64().or_else(|| v.as_str().and_then(|x| x.split(',').next()).and_then(|x| x.parse().ok()))
        };
        if a.get("share_price").and_then(price) != price(wp) {
            return false;
        }
    }
    if let Some(wv) = w.get("rotation").and_then(Value::as_i64) {
        if a.get("rotation").and_then(Value::as_i64).unwrap_or(0) != wv {
            return false;
        }
    }
    if let Some(wt) = s(w, "tile") {
        if s(a, "tile").map(|x| base(&x)) != Some(base(&wt)) {
            return false;
        }
    }
    if let Some(wt) = s(w, "train") {
        let Some(at) = s(a, "train") else { return false };
        if at != wt {
            // another company's train: any of the same type that company holds (the depot's: that one)
            match (train_place(g, &at), train_place(g, &wt)) {
                (Some((ha, na)), Some((hw, nw))) if ha != "depot" && ha == hw && na == nw => {}
                _ => return false,
            }
        }
        if let Some(wv) = s(w, "variant") {
            if s(a, "variant").unwrap_or_else(|| base(&at)) != wv {
                return false;
            }
        }
    }
    if let Some(wc) = s(w, "city") {
        if city_norm(&s(a, "city").unwrap_or_default()) != city_norm(&wc) {
            return false;
        }
        if s(w, "type").as_deref() == Some("remove_token") && s(a, "slot").unwrap_or_else(|| "0".into()) != s(w, "slot").unwrap_or_else(|| "0".into()) {
            return false;
        }
    }
    if let Some(Value::Array(ws)) = w.get("shares") {
        let mut x: Vec<String> = a.get("shares").and_then(Value::as_array).map(|v| v.iter().filter_map(|y| y.as_str().map(str::to_string)).collect()).unwrap_or_default();
        let mut y: Vec<String> = ws.iter().filter_map(|v| v.as_str().map(str::to_string)).collect();
        x.sort();
        y.sort();
        if x != y {
            // the same corporation and percent, other certificates of the same holder: the same bundle
            let corp = |l: &[String]| l.first().and_then(|z| z.rsplit_once('_').map(|q| q.0.to_string()));
            if a.get("percent") != w.get("percent") || corp(&x) != corp(&y) || x.len() != y.len() {
                return false;
            }
        }
    }
    true
}

/// A ranged price on its range and step, else the move's own.
fn price_ok(a: &Value, w: &Value, r: Option<PriceRange>) -> Result<(), String> {
    let Some(wp) = w.get("price").and_then(Value::as_i64) else { return Ok(()) };
    let Some(ap) = a.get("price").and_then(Value::as_i64) else { return Err("the move needs a price".into()) };
    match r {
        Some(r) => {
            if ap < r.lo as i64 || ap > r.hi as i64 {
                return Err(format!("Price must be between ${} and ${}", r.lo, r.hi));
            }
            if r.step > 1 && (ap - r.lo as i64) % r.step as i64 != 0 {
                return Err(format!("Price must be a multiple of ${}", r.step));
            }
            Ok(())
        }
        None if ap != wp => Err(format!("The price is ${wp}")),
        None => Ok(()),
    }
}

/// The generator's verdict on a live 1867 move (`a` without its metadata): Ok, or the refusal's words.
pub fn check_move(g: &mut SiteGame, a: &Value) -> Result<(), String> {
    let t = s(a, "type").unwrap_or_default();
    let moves: Vec<Move> = g.game.g1867_legal_moves_with(CrossPlayerPolicy::RULES);
    let wires: Vec<(Value, Option<PriceRange>)> = moves.iter().map(|m| (g.game.g1867_move_wire(m), m.price)).collect();
    let mut price_err = None;
    for (w, r) in &wires {
        if same_move(g, a, w) {
            match price_ok(a, w, *r) {
                Ok(()) => return Ok(()),
                Err(e) => price_err = Some(e),
            }
        }
    }
    if let Some(e) = price_err {
        return Err(refused(e));
    }
    let same_type = wires.iter().any(|(w, _)| s(w, "type").as_deref() == Some(t.as_str()));
    let who = s(a, "entity").unwrap_or_default();
    Err(refused(if same_type {
        format!("Not a legal {} for {who} now", t.replace('_', " "))
    } else {
        format!("{who} cannot {} now", t.replace('_', " "))
    }))
}

/// The corpus check (`site_replay --act-check`): replaying the game, every recorded engine move is checked by the
/// generator on the position before it (`check_move`), and after it the predicted step auto actions are compared with
/// the recorded ones (where no player has a program). `{moves, accepted, refused: [...], autos, autos_equal,
/// autos_differ: [...]}`.
pub fn corpus_check(g: &mut SiteGame) -> Value {
    let (mut moves, mut ok, mut autos, mut autos_eq) = (0u32, 0u32, 0u32, 0u32);
    let mut bad = Vec::new();
    let mut auto_bad = Vec::new();
    while !g.done() {
        let i = g.pos;
        let a = g.raw[i].clone();
        let t = s(&a, "type").unwrap_or_default();
        let engine_move = g.kept[i]
            && !(t == "message" || t == "log" || t == "end_game" || t == "undo" || t == "redo" || t.starts_with("program_"));
        if engine_move && !g.game.finished {
            moves += 1;
            let bare = super::super::json::strip_meta(&a);
            let verdict = if t == "run_routes" { Ok(()) } else { check_move(g, &bare) };
            match verdict {
                Ok(()) => ok += 1,
                Err(e) => {
                    if bad.len() < 40 {
                        bad.push(json!({"id": a.get("id"), "action": bare, "error": e}));
                    }
                }
            }
        }
        let programs = g.programs_enabled();
        // the position right after the move alone (its recorded auto actions left out), for the prediction
        let predicted = (engine_move && !programs && !g.game.finished).then(|| {
            let mut c = g.clone();
            c.site_log = false;
            let mut bare = a.clone();
            if let Some(m) = bare.as_object_mut() {
                m.remove("auto_actions");
            }
            match c.apply(&bare) {
                Ok(()) => super::autos::predict(&c.game)
                    .into_iter()
                    .map(|x| json!({"type": s(&x, "type"), "entity": s(&x, "entity")}))
                    .collect::<Vec<_>>(),
                Err(_) => vec![json!({"type": "refused"})],
            }
        });
        g.process_next();
        if let (Some(predicted), None) = (predicted, &g.stopped) {
            // Ruby's auto actions after the move: the recorded ones that aren't programs'
            let recorded: Vec<Value> = a
                .get("auto_actions")
                .and_then(Value::as_array)
                .map(|v| v.iter().filter(|x| !s(x, "type").unwrap_or_default().starts_with("program_")).map(|x| json!({"type": s(x, "type"), "entity": s(x, "entity")})).collect())
                .unwrap_or_default();
            autos += 1;
            if predicted == recorded {
                autos_eq += 1;
            } else if auto_bad.len() < 40 {
                auto_bad.push(json!({"id": a.get("id"), "type": t, "recorded": recorded, "predicted": predicted}));
            }
        }
    }
    json!({"moves": moves, "accepted": ok, "refused": bad, "autos": autos, "autos_equal": autos_eq, "autos_differ": auto_bad,
           "stopped": g.stopped.as_ref().map(|x| json!({"id": x.action_id, "message": x.message}))})
}
