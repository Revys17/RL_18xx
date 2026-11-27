//! The site's action JSON <-> the engine's [`Action`] (docs/WASM_ENGINE_SPIKE.md §3.2, boundary translations).
//!
//! `action_from_json` is the serde_json twin of `Action::from_py_dict` (the same keys, defaults and shapes: the
//! importers' Python dicts ARE the site's JSON). The translations between Ruby's and the engine's numbering live
//! here too:
//! - G19's two preprinted cities are numbered the other way round from Ruby while the hex is preprinted
//!   (`place_token.city` "G19-0-<i>" and `routes[].nodes` "G19-<i>"), as `autoplay.py` `swap_city_refs` does;
//! - the next tile instance of a name follows Ruby's supply list ([`TileSupply`]).

use crate::actions::{Action, DividendKind, GameError, RouteData};
use serde_json::{json, Map, Value};

/// Preprinted hexes whose two cities engine-rs numbers the other way round from Ruby (until a tile is laid there).
pub const SWAPPED_PREPRINTED_CITIES: &[&str] = &["G19"];

fn get<'a>(a: &'a Value, k: &str) -> Option<&'a Value> {
    a.get(k).filter(|v| !v.is_null())
}

fn s(a: &Value, k: &str) -> Result<String, GameError> {
    match get(a, k) {
        Some(Value::String(x)) => Ok(x.clone()),
        Some(_) => Err(GameError::new(format!("'{k}' must be a string"))),
        None => Err(GameError::new(format!("Missing '{k}' key"))),
    }
}

fn opt_s(a: &Value, k: &str) -> Result<Option<String>, GameError> {
    match get(a, k) {
        Some(Value::String(x)) => Ok(Some(x.clone())),
        Some(_) => Err(GameError::new(format!("'{k}' must be a string"))),
        None => Ok(None),
    }
}

fn int(v: &Value) -> Option<i64> {
    v.as_i64().or_else(|| v.as_f64().filter(|f| f.fract() == 0.0).map(|f| f as i64))
}

fn i(a: &Value, k: &str) -> Result<i32, GameError> {
    get(a, k)
        .ok_or_else(|| GameError::new(format!("Missing '{k}' key")))
        .and_then(|v| int(v).map(|x| x as i32).ok_or_else(|| GameError::new(format!("'{k}' must be an integer"))))
}

fn opt_i(a: &Value, k: &str) -> Result<Option<i32>, GameError> {
    match get(a, k) {
        None => Ok(None),
        Some(v) => int(v).map(|x| Some(x as i32)).ok_or_else(|| GameError::new(format!("'{k}' must be an integer"))),
    }
}

fn str_list(v: Option<&Value>) -> Vec<String> {
    match v {
        Some(Value::Array(xs)) => xs.iter().filter_map(|x| x.as_str().map(str::to_string)).collect(),
        _ => Vec::new(),
    }
}

/// The action's entity as the engine names it: a player's site id ("13") or a corporation / company sym.
pub fn entity_of(a: &Value) -> Result<String, GameError> {
    match get(a, "entity") {
        Some(Value::String(x)) => Ok(x.clone()),
        Some(v) => int(v).map(|x| x.to_string()).ok_or_else(|| GameError::new("'entity' must be int or string")),
        None => Err(GameError::new("Missing 'entity' key")),
    }
}

/// `"57-0-0"` (tile instance, city) or `"E5-0-1"` (a preprinted hex's tile) -> the engine's `(__tile:<instance>, city)`.
fn parse_city_ref(city: &str) -> (String, u8) {
    let parts: Vec<&str> = city.split('-').collect();
    if parts.len() >= 3 {
        (format!("__tile:{}-{}", parts[0], parts[1]), parts[2].parse().unwrap_or(0))
    } else if parts.len() == 2 {
        (format!("__tile:{}", parts[0]), parts[1].parse().unwrap_or(0))
    } else {
        (format!("__tile:{city}"), 0)
    }
}

fn shares_info(a: &Value) -> Result<(String, Vec<usize>), GameError> {
    let ids = str_list(get(a, "shares"));
    let corp = match opt_s(a, "corporation")? {
        Some(c) => c,
        None => match ids.first().and_then(|id| id.rfind('_').map(|p| id[..p].to_string())) {
            Some(c) => c,
            None => return Err(GameError::new("Cannot determine corporation from shares or corporation field")),
        },
    };
    let idx = ids.iter().filter_map(|id| id.rsplit('_').next()?.parse::<usize>().ok()).collect();
    Ok((corp, idx))
}

/// The site's action JSON as an engine [`Action`] (`Action::from_py_dict`'s rules).
pub fn action_from_json(a: &Value) -> Result<Action, GameError> {
    let kind = s(a, "type")?;
    let entity_id = entity_of(a)?;
    Ok(match kind.as_str() {
        "pass" => Action::Pass { entity_id },
        "bid" => {
            let price = i(a, "price")?;
            if let Some(corporation_sym) = opt_s(a, "corporation")? {
                Action::CorporationBid { entity_id, corporation_sym, price }
            } else {
                Action::Bid { entity_id, company_sym: s(a, "company")?, price }
            }
        }
        "par" => {
            let share_price = match get(a, "share_price") {
                Some(Value::String(x)) => x
                    .split(',')
                    .next()
                    .and_then(|p| p.parse::<i32>().ok())
                    .ok_or_else(|| GameError::new(format!("Invalid share_price format: {x}")))?,
                Some(v) => int(v).map(|x| x as i32).ok_or_else(|| GameError::new("bad share_price"))?,
                None => return Err(GameError::new("Missing 'share_price' key")),
            };
            Action::Par { entity_id, corporation_sym: s(a, "corporation")?, share_price }
        }
        "buy_shares" => {
            let (corporation_sym, share_indices) = shares_info(a)?;
            Action::BuyShares {
                entity_id,
                corporation_sym,
                shares: Vec::new(),
                percent: opt_i(a, "percent")?.unwrap_or(10) as u8,
                source: opt_s(a, "source")?.unwrap_or_else(|| "auto".into()),
                share_indices,
            }
        }
        "sell_shares" => {
            let (corporation_sym, share_indices) = shares_info(a)?;
            Action::SellShares {
                entity_id,
                corporation_sym,
                shares: Vec::new(),
                percent: opt_i(a, "percent")?.unwrap_or(10) as u8,
                share_indices,
            }
        }
        "lay_tile" => Action::LayTile {
            entity_id,
            hex_id: s(a, "hex")?,
            tile_id: s(a, "tile")?,
            rotation: opt_i(a, "rotation")?.unwrap_or(0) as u8,
        },
        "place_token" => {
            let (hex_id, city_index) = match opt_s(a, "hex")? {
                Some(h) if get(a, "city").is_none() => (h, opt_i(a, "city_index")?.unwrap_or(0) as u8),
                _ => parse_city_ref(&s(a, "city")?),
            };
            Action::PlaceToken { entity_id, hex_id, city_index }
        }
        "run_routes" => {
            let mut routes = Vec::new();
            if let Some(Value::Array(rs)) = get(a, "routes") {
                for r in rs {
                    routes.push(RouteData {
                        train_name: s(r, "train")?,
                        hexes: str_list(get(r, "hexes")),
                        revenue: opt_i(r, "revenue")?,
                        connections: match get(r, "connections") {
                            Some(Value::Array(cs)) => cs.iter().map(|c| str_list(Some(c))).collect(),
                            _ => Vec::new(),
                        },
                        nodes: str_list(get(r, "nodes")),
                    });
                }
            }
            Action::RunRoutes { entity_id, routes, extra_revenue: opt_i(a, "extra_revenue")?.unwrap_or(0) }
        }
        "dividend" => Action::Dividend { entity_id, kind: DividendKind::parse(&s(a, "kind")?)? },
        "buy_train" => Action::BuyTrain {
            entity_id,
            train_name: s(a, "train")?,
            price: i(a, "price")?,
            from: opt_s(a, "from")?.unwrap_or_else(|| "depot".into()),
            variant: opt_s(a, "variant")?,
            exchange: opt_s(a, "exchange")?,
        },
        "discard_train" => Action::DiscardTrain { entity_id, train_name: s(a, "train")? },
        "buy_company" => Action::BuyCompany { entity_id, company_sym: s(a, "company")?, price: i(a, "price")? },
        "bankrupt" => Action::Bankrupt { entity_id },
        "choose" => Action::Choose { entity_id, choice: s(a, "choice")? },
        // 1867's merger round (docs/SITE_ENGINE_RS_1867.md §3.1): one wire type `merge` for adding a minor, naming
        // the major of a merger or of a conversion (Ruby's step state tells them apart); `convert`; `remove_token`
        // (ReduceTokens: a city by id and its slot)
        "merge" => Action::Merge { entity_id, corporation_sym: s(a, "corporation")? },
        "convert" => Action::Convert { entity_id },
        "remove_token" => {
            let (hex_id, city_index) = parse_city_ref(&s(a, "city")?);
            Action::RemoveToken { entity_id, hex_id, city_index, slot: opt_i(a, "slot")?.unwrap_or(0) as u8 }
        }
        other => return Err(GameError::new(format!("unsupported action type: {other}"))),
    })
}

/// The index swap of one G19 reference (`sep_count` dashes: 2 for a city id "G19-0-1", 1 for a node "G19-1").
fn swap_index(r: &str, hex: &str, sep_count: usize) -> String {
    let mut parts: Vec<&str> = r.split('-').collect();
    if parts.first() != Some(&hex) || parts.len() != sep_count + 1 {
        return r.to_string();
    }
    let last = match *parts.last().unwrap() {
        "0" => "1",
        "1" => "0",
        _ => return r.to_string(),
    };
    *parts.last_mut().unwrap() = last;
    parts.join("-")
}

/// `action` (site JSON) with its references to the cities of `hexes` (still preprinted) in the other engine's
/// numbering: a station's `city` and the routes' `nodes`. Its own inverse.
pub fn swap_city_refs(action: &Value, hexes: &[&str]) -> Value {
    let mut out = action.clone();
    for hx in hexes {
        let kind = out.get("type").and_then(Value::as_str).unwrap_or("").to_string();
        if kind == "place_token" {
            if let Some(c) = out.get("city").and_then(Value::as_str) {
                if c.starts_with(&format!("{hx}-0-")) {
                    out["city"] = json!(swap_index(c, hx, 2));
                }
            }
        }
        if kind == "run_routes" {
            if let Some(Value::Array(routes)) = out.get_mut("routes") {
                for r in routes.iter_mut() {
                    if let Some(Value::Array(nodes)) = r.get_mut("nodes") {
                        for n in nodes.iter_mut() {
                            if let Some(x) = n.as_str() {
                                *n = json!(swap_index(x, hx, 1));
                            }
                        }
                    }
                }
            }
        }
    }
    out
}

/// Ruby's tile supply order (`@game.tiles`): the next tile of a name to lay is the first in the list; a laid tile
/// leaves it, and a tile upgraded over goes back *at the end*. engine-rs's own enumeration takes the lowest free
/// instance (`decode.rs` `available_tile_instance_id`), so a site-mode move names the instance Ruby would, and the
/// tile options list the names in the list's order.
#[derive(Clone, Debug, Default)]
pub struct TileSupply {
    list: Vec<String>,
    /// Ruby's unlimited tiles (1867's 7, 8, 9) with the highest instance index made so far
    unlimited: Vec<(String, u32)>,
}

impl TileSupply {
    /// The title's tile counts (`name -> count`, the instances are "<name>-<i>"), in the title's TILES order. A count
    /// of `u32::MAX` is Ruby's `'unlimited'`: one instance in the list at a time (`Game::Base#add_extra_tile`).
    pub fn new(counts: &[(String, u32)]) -> Self {
        let mut list = Vec::new();
        let mut unlimited = Vec::new();
        for (name, n) in counts {
            if *n == u32::MAX {
                unlimited.push((name.clone(), 0));
                list.push(format!("{name}-0"));
                continue;
            }
            for i in 0..*n {
                list.push(format!("{name}-{i}"));
            }
        }
        TileSupply { list, unlimited }
    }

    /// Ruby's choice of the next `name` tile, if any is left.
    pub fn next(&self, name: &str) -> Option<&str> {
        self.list.iter().find(|t| tile_name(t) == Some(name)).map(String::as_str)
    }

    /// Where a tile name first appears in the supply (the order of the tile options).
    pub fn rank(&self, name: &str) -> usize {
        self.list.iter().position(|t| tile_name(t) == Some(name)).unwrap_or(usize::MAX)
    }

    /// A lay of tile instance `laid` over `old` (the hex's tile before: an instance id, or a preprinted hex's name).
    /// Ruby `update_tile_lists`: an unlimited tile's next instance is made first and goes at the end (`add_extra_tile`:
    /// the highest index made so far + 1), then the laid one leaves, then the tile upgraded over comes back at the end.
    pub fn laid(&mut self, laid: &str, old: &str) {
        if let Some(name) = tile_name(laid) {
            if let Some(u) = self.unlimited.iter_mut().find(|u| u.0 == name) {
                let idx = laid.rsplit_once('-').and_then(|(_, i)| i.parse::<u32>().ok()).unwrap_or(0);
                u.1 = u.1.max(idx) + 1;
                self.list.push(format!("{name}-{}", u.1));
            }
        }
        self.list.retain(|t| t != laid);
        if tile_name(old).is_some() && !self.list.iter().any(|t| t == old) {
            self.list.push(old.to_string());
        }
    }
}

/// The tile name of an instance id ("57-1" -> "57"); None for a preprinted hex's own tile ("E5").
pub fn tile_name(instance: &str) -> Option<&str> {
    let (name, idx) = instance.rsplit_once('-')?;
    idx.parse::<u32>().ok().map(|_| name)
}

/// The keys of a site action that the engine takes no notice of but the site stores (kept when re-serialising).
pub const META_KEYS: &[&str] = &["id", "user", "created_at", "auto_actions"];

/// A site action without its metadata (`id`, `user`, `created_at`, `auto_actions`), keys in insertion order.
pub fn strip_meta(a: &Value) -> Value {
    match a {
        Value::Object(m) => {
            let mut out = Map::new();
            for (k, v) in m {
                if !META_KEYS.contains(&k.as_str()) {
                    out.insert(k.clone(), v.clone());
                }
            }
            Value::Object(out)
        }
        v => v.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn g19_refs_swap_both_ways() {
        let a = json!({"type": "place_token", "entity": "NYC", "city": "G19-0-1"});
        let b = swap_city_refs(&a, &["G19"]);
        assert_eq!(b["city"], "G19-0-0");
        assert_eq!(swap_city_refs(&b, &["G19"]), a);
        let r = json!({"type": "run_routes", "entity": "NYC", "routes": [{"train": "2-0", "nodes": ["G19-0", "F20-0"]}]});
        assert_eq!(swap_city_refs(&r, &["G19"])["routes"][0]["nodes"], json!(["G19-1", "F20-0"]));
        // a laid tile's city ids are not touched
        let c = json!({"type": "place_token", "entity": "NYC", "city": "54-0-1"});
        assert_eq!(swap_city_refs(&c, &["G19"]), c);
    }

    #[test]
    fn tile_supply_follows_ruby_order() {
        let mut s = TileSupply::new(&[("8".into(), 3), ("16".into(), 1)]);
        assert_eq!(s.next("8"), Some("8-0"));
        s.laid("8-0", "F14");
        assert_eq!(s.next("8"), Some("8-1"));
        s.laid("16-0", "8-0"); // 8-0 upgraded over: back at the end
        assert_eq!(s.next("8"), Some("8-1"));
        s.laid("8-1", "G13");
        s.laid("8-2", "H12");
        assert_eq!(s.next("8"), Some("8-0"));
        assert_eq!(s.next("16"), None);
    }

    #[test]
    fn actions_parse_like_the_python_dicts() {
        let a = action_from_json(&json!({"type": "par", "entity": 4, "corporation": "B&O", "share_price": "100,0,6"}))
            .unwrap();
        assert!(matches!(a, Action::Par { share_price: 100, .. }));
        let b = action_from_json(&json!({"type": "buy_shares", "entity": 9, "shares": ["C&O_1"], "percent": 10}))
            .unwrap();
        match b {
            Action::BuyShares { corporation_sym, share_indices, entity_id, .. } => {
                assert_eq!((corporation_sym.as_str(), share_indices, entity_id.as_str()), ("C&O", vec![1], "9"));
            }
            _ => panic!(),
        }
    }
}
