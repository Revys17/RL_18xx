//! Engine-computed revenue for RECORDED routes.
//!
//! 18xx.games exports serialize a `run_routes` action as node-to-node hex
//! chains (`connections`) — 1830 records also carry the computed `revenue`,
//! but 1867 hotseat records (and Ruby fixtures) may carry connections only.
//! Replaying those requires the engine to price the recorded route itself:
//! `GameTitle::recorded_route_revenue` dispatches here.
//!
//! This is NOT the route optimizer (router.rs): the route is already chosen,
//! we only price its stops under the title's rules. The bucketed-distance
//! model below ports Ruby 1867's `compute_stops` (g_1867/game.rb:706-739):
//! a train pays up to `pay` counted stops (cities+offboards; offboards only
//! for the 5+5E), runs through any number of uncounted stops (towns) free,
//! and fills spare paying capacity with the best uncounted stops.

use crate::actions::GameError;
use crate::game::BaseGame;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StopKind {
    City,
    Town,
    Offboard,
}

/// A revenue center visited by a recorded route, priced for the current
/// phase.
#[derive(Clone, Debug)]
pub struct Stop {
    pub hex: String,
    pub kind: StopKind,
    pub revenue: i32,
    /// Whether the operating corporation has a token here (5+5E's
    /// tokened-counted-stop requirement).
    pub tokened: bool,
}

/// Title-specific revenue bonuses applied on top of the per-stop sum.
#[derive(Clone, Debug, Default)]
pub struct RouteBonuses {
    /// `hex_bonus` company abilities held by the operating corporation:
    /// +amount per visited hex in the list (flat — NOT multiplied).
    pub hex_bonuses: Vec<(Vec<String>, i32)>,
    /// Visiting any of `capital_hexes` AND `capital_partner` in one route
    /// pays +`capital_amount` × multiplier (1867: Toronto/Montreal/Quebec
    /// with Timmins D2, +$40).
    pub capital_hexes: Vec<String>,
    pub capital_partner: Option<String>,
    pub capital_amount: i32,
}

/// The hexes a recorded route stops at: the endpoints of each connection
/// chain (interior hexes are plain track), deduplicated in first-seen order.
/// Revenue is order-independent, so no chain re-orientation is needed.
pub fn stop_hexes_from_connections(
    connections: &[Vec<String>],
) -> Result<Vec<String>, GameError> {
    let mut out: Vec<String> = Vec::new();
    for chain in connections {
        if chain.len() < 2 {
            return Err(GameError::new(format!(
                "recorded connection chain has fewer than 2 hexes: {:?}",
                chain
            )));
        }
        for hex in [&chain[0], &chain[chain.len() - 1]] {
            if !out.iter().any(|h| h == hex) {
                out.push(hex.clone());
            }
        }
    }
    Ok(out)
}

/// Classify the revenue center the route stops at on `hex_id`, priced for
/// the game's current phase. Offboards take precedence (red hexes), then
/// cities, then towns — a connection endpoint always carries exactly one of
/// them on the traversed path.
pub fn classify_stop(
    game: &BaseGame,
    hex_id: &str,
    corp_sym: &str,
) -> Result<Stop, GameError> {
    let hex = game
        .hex_ref(hex_id)
        .ok_or_else(|| GameError::new(format!("recorded route stop on unknown hex {}", hex_id)))?;
    let tile = &hex.tile;

    if let Some(ob) = tile.offboards.first() {
        return Ok(Stop {
            hex: hex_id.to_string(),
            kind: StopKind::Offboard,
            revenue: ob.phase_revenue(&game.phase.tiles),
            tokened: false,
        });
    }
    if !tile.cities.is_empty() {
        let revenue = tile.cities.iter().map(|c| c.revenue).max().unwrap_or(0);
        let tokened = tile.cities.iter().any(|c| {
            c.tokens
                .iter()
                .flatten()
                .any(|t| t.corporation_id == corp_sym)
        });
        return Ok(Stop {
            hex: hex_id.to_string(),
            kind: StopKind::City,
            revenue,
            tokened,
        });
    }
    if let Some(town) = tile.towns.iter().max_by_key(|t| t.revenue) {
        return Ok(Stop {
            hex: hex_id.to_string(),
            kind: StopKind::Town,
            revenue: town.revenue,
            tokened: false,
        });
    }
    Err(GameError::new(format!(
        "recorded route stop on {} but its tile {} has no revenue center",
        hex_id, tile.id
    )))
}

/// Price a recorded route under bucketed-distance rules (Ruby 1867
/// `compute_stops` + `revenue_for`): counted stops (`counted_kinds`) pay up
/// to `pay` of them; uncounted stops fill any spare capacity, best-first;
/// at least one chosen stop must be tokened when no counted stop is
/// (the 5+5E constraint — trivially satisfied for normal trains, whose
/// route legality already runs through a token).
pub fn price_bucketed_stops(
    stops: &[Stop],
    pay: usize,
    multiplier: i32,
    counted_kinds: &[StopKind],
    bonuses: &RouteBonuses,
) -> Result<i32, GameError> {
    let (mandatory, optional): (Vec<&Stop>, Vec<&Stop>) = stops
        .iter()
        .partition(|s| counted_kinds.contains(&s.kind));

    if mandatory.len() > pay {
        return Err(GameError::new(format!(
            "recorded route visits {} counted stops but the train pays only {}",
            mandatory.len(),
            pay
        )));
    }

    if mandatory.len() == pay {
        return Ok(revenue_for(&mandatory, multiplier, bonuses));
    }

    // Spare paying capacity: fill with the best combination of uncounted
    // stops (Ruby tries every combination because bonuses make per-stop
    // revenue non-additive; stop counts are tiny, so we do the same).
    let remaining = (pay - mandatory.len()).min(optional.len());
    let need_token = !mandatory.iter().any(|s| s.tokened);

    let mut best: Option<i32> = None;
    for combo in combinations(&optional, remaining) {
        if need_token && !combo.iter().any(|s| s.tokened) {
            continue;
        }
        let mut chosen = mandatory.clone();
        chosen.extend(combo);
        let rev = revenue_for(&chosen, multiplier, bonuses);
        if best.map_or(true, |b| rev > b) {
            best = Some(rev);
        }
    }

    best.ok_or_else(|| {
        GameError::new("recorded route has no legal stop allocation (token requirement)")
    })
}

/// Ruby 1867 `revenue_for`: per-stop revenue × train multiplier, plus flat
/// hex_bonus ability amounts, plus the capitals bonus × multiplier.
fn revenue_for(stops: &[&Stop], multiplier: i32, bonuses: &RouteBonuses) -> i32 {
    let mut revenue: i32 = stops.iter().map(|s| s.revenue).sum::<i32>() * multiplier;

    for (hexes, amount) in &bonuses.hex_bonuses {
        revenue += stops
            .iter()
            .filter(|s| hexes.iter().any(|h| h == &s.hex))
            .count() as i32
            * amount;
    }

    if let Some(partner) = &bonuses.capital_partner {
        let capitals = stops
            .iter()
            .any(|s| bonuses.capital_hexes.iter().any(|h| h == &s.hex));
        let has_partner = stops.iter().any(|s| &s.hex == partner);
        if capitals && has_partner {
            revenue += bonuses.capital_amount * multiplier;
        }
    }

    revenue
}

/// All k-element combinations of `items` (k tiny: bounded by train size).
fn combinations<'a>(items: &[&'a Stop], k: usize) -> Vec<Vec<&'a Stop>> {
    if k == 0 {
        return vec![Vec::new()];
    }
    if items.len() < k {
        // Ruby: `combinations = [optional_stops] if combinations.empty?` —
        // fewer fillers than capacity means take them all.
        return vec![items.to_vec()];
    }
    let mut out = Vec::new();
    for (i, first) in items.iter().enumerate() {
        for mut rest in combinations(&items[i + 1..], k - 1) {
            rest.insert(0, first);
            out.push(rest);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn city(hex: &str, revenue: i32) -> Stop {
        Stop { hex: hex.into(), kind: StopKind::City, revenue, tokened: false }
    }
    fn tokened_city(hex: &str, revenue: i32) -> Stop {
        Stop { hex: hex.into(), kind: StopKind::City, revenue, tokened: true }
    }
    fn town(hex: &str, revenue: i32) -> Stop {
        Stop { hex: hex.into(), kind: StopKind::Town, revenue, tokened: false }
    }
    fn offboard(hex: &str, revenue: i32) -> Stop {
        Stop { hex: hex.into(), kind: StopKind::Offboard, revenue, tokened: false }
    }

    const CITY_OFFBOARD: &[StopKind] = &[StopKind::City, StopKind::Offboard];
    const OFFBOARD_ONLY: &[StopKind] = &[StopKind::Offboard];

    #[test]
    fn stop_hexes_dedup_chain_junctions() {
        let conns = vec![
            vec!["J12".to_string(), "K13".to_string()],
            vec!["K13".to_string(), "L12".to_string()],
        ];
        assert_eq!(stop_hexes_from_connections(&conns).unwrap(), vec!["J12", "K13", "L12"]);
    }

    #[test]
    fn stop_hexes_interior_hexes_are_not_stops() {
        let conns = vec![vec!["J12".to_string(), "K11".to_string(), "L10".to_string()]];
        assert_eq!(stop_hexes_from_connections(&conns).unwrap(), vec!["J12", "L10"]);
    }

    #[test]
    fn full_train_excludes_towns() {
        // 2-train over city-town-city: towns drop out when counted == pay.
        let stops = vec![city("A1", 30), town("B2", 10), city("C3", 40)];
        let rev = price_bucketed_stops(&stops, 2, 1, CITY_OFFBOARD, &RouteBonuses::default()).unwrap();
        assert_eq!(rev, 70);
    }

    #[test]
    fn spare_capacity_pays_best_towns() {
        // 3-train, one city, two towns: city + both towns fit (pay 3).
        // (The city is tokened — a legal route always runs through the
        // corporation's token, and it is a counted stop for normal trains.)
        let stops = vec![tokened_city("A1", 30), town("B2", 10), town("C3", 20)];
        let rev = price_bucketed_stops(&stops, 3, 1, CITY_OFFBOARD, &RouteBonuses::default()).unwrap();
        assert_eq!(rev, 60);
        // 2-train: city + the better town only.
        let rev = price_bucketed_stops(&stops, 2, 1, CITY_OFFBOARD, &RouteBonuses::default()).unwrap();
        assert_eq!(rev, 50);
    }

    #[test]
    fn too_many_counted_stops_is_an_error() {
        let stops = vec![city("A1", 30), city("B2", 30), city("C3", 30)];
        assert!(price_bucketed_stops(&stops, 2, 1, CITY_OFFBOARD, &RouteBonuses::default()).is_err());
    }

    #[test]
    fn multiplier_scales_stops_and_capitals_but_not_hex_bonus() {
        let stops = vec![tokened_city("D2", 30), city("L12", 40)];
        let bonuses = RouteBonuses {
            hex_bonuses: vec![(vec!["L12".to_string()], 10)],
            capital_hexes: vec!["F16".into(), "L12".into(), "O7".into()],
            capital_partner: Some("D2".into()),
            capital_amount: 40,
        };
        // (30+40)*2 + 10 (flat) + 40*2 (capitals) = 230
        let rev = price_bucketed_stops(&stops, 2, 2, CITY_OFFBOARD, &bonuses).unwrap();
        assert_eq!(rev, 230);
    }

    #[test]
    fn capitals_bonus_needs_both_ends() {
        let stops = vec![tokened_city("L12", 40), city("K13", 30)];
        let bonuses = RouteBonuses {
            capital_hexes: vec!["F16".into(), "L12".into(), "O7".into()],
            capital_partner: Some("D2".into()),
            capital_amount: 40,
            ..Default::default()
        };
        let rev = price_bucketed_stops(&stops, 2, 1, CITY_OFFBOARD, &bonuses).unwrap();
        assert_eq!(rev, 70);
    }

    #[test]
    fn offboard_only_train_needs_a_tokened_fill() {
        // 5+5E: offboards mandatory, cities optional fill; no offboard is
        // tokened, so a chosen city must be.
        let stops = vec![
            offboard("A7", 40),
            offboard("P8", 40),
            city("X1", 50),
            tokened_city("X2", 20),
        ];
        let rev = price_bucketed_stops(&stops, 5, 2, OFFBOARD_ONLY, &RouteBonuses::default()).unwrap();
        // All four fit within pay 5 (2 mandatory + 2 fills) — token present.
        assert_eq!(rev, (40 + 40 + 50 + 20) * 2);

        // pay 3: only one fill slot — must take the TOKENED city even though
        // the untokened one pays more.
        let rev = price_bucketed_stops(&stops, 3, 2, OFFBOARD_ONLY, &RouteBonuses::default()).unwrap();
        assert_eq!(rev, (40 + 40 + 20) * 2);
    }
}
