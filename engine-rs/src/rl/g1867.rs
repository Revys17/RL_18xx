//! 1867's RL layout (`1867-v1`, `docs/rl_descriptors.md` §8): a registry
//! layout whose blocks and axes come from the title data, and the index
//! function that maps a move of the legal-move generator
//! (`g1867/legal.rs`) to its index.
//!
//! Every generated move maps to exactly one index; a state's moves map to
//! distinct indices (the fuzz in the tests below and the corpus check of
//! `scripts/g1867_completeness.py --index` verify both). A price-bearing
//! move is one index whose legal range comes from the move
//! (`PriceRange { lo, hi, step }`), priced by a price head
//! (`PriceKind`), never a menu of prices. What a move needs beyond its
//! index is state: a token's distance price, a lay's slot rules and cost,
//! whose pass it is, which minor a Merge adds or which major it names.

use std::collections::HashSet;

use super::descriptor::{EntKind, Family, PriceKind};
use super::layout::{Axis, BlockSpec, RlLayout, Vocab};
use crate::actions::{Action, DividendKind, GameError};
use crate::g1867::legal::Move;
use crate::game::BaseGame;
use crate::steps::StepKind;
use crate::title::{CorpType, GameTitle};

/// The layout version. Bump it (and re-convert data) whenever an index
/// changes meaning; the fingerprint test pins it.
pub const VERSION: &str = "1867-v1";

/// The dividend kinds the 1867 Dividend step offers (generator order).
const DIVIDENDS: [&str; 3] = ["dividend:payout", "dividend:half", "dividend:withhold"];
/// A trade-in's new train: from the depot or from the discard pool.
const TRAIN_SOURCES: [&str; 2] = ["train_source:depot", "train_source:pool"];

/// The block specs of `1867-v1`, in index order (the small blocks first,
/// the tile lays last).
pub fn block_specs(title: &dyn GameTitle) -> Result<Vec<BlockSpec>, String> {
    use Axis::*;
    let majors = Axis::corporations(0, title, Some(CorpType::Major));
    let minors = Axis::corporations(0, title, Some(CorpType::Minor));
    // Majors and minors in title order (the CN only holds tokens: it never
    // sells a train, merges or is merged).
    let operating: Vec<&'static str> = title
        .corporations()
        .iter()
        .filter(|c| c.corp_type != CorpType::National)
        .map(|c| c.sym)
        .collect();
    let companies = Axis::companies(0, title, true);
    let trains = || Axis::trains(0, title);

    // Share percents: one share unit up to the market pool limit (a sale or
    // a redemption never moves more than the pool holds).
    let unit = title
        .corporations()
        .iter()
        .filter(|c| c.corp_type == CorpType::Major)
        .flat_map(|c| c.shares.iter().copied())
        .min()
        .ok_or("no major shares")? as i32;
    let percents: Vec<i32> = (1..=title.market_pool_limit() as i32 / unit).map(|k| k * unit).collect();

    // Trade-ins: the trains with a discount table (new) x the train types
    // those tables discount (traded), in train order.
    let tdefs = title.trains();
    let train_order = Vocab::for_title(title).trains;
    let new_trains: Vec<&'static str> =
        train_order.iter().copied().filter(|n| tdefs.iter().any(|t| t.name == *n && !t.discount.is_empty())).collect();
    let traded: Vec<&'static str> = train_order
        .iter()
        .copied()
        .filter(|n| tdefs.iter().any(|t| t.discount.iter().any(|(d, _)| d == n)))
        .collect();
    if new_trains.is_empty() || traded.is_empty() {
        return Err("no trade-in tables in the train data".into());
    }

    let cities = city_slots(title);
    Ok(vec![
        BlockSpec::new("Pass", Family::Pass, vec![]),
        BlockSpec::new("Bid.company", Family::BidCompany, vec![companies.clone()]).priced(PriceKind::Bid),
        // Minor bids price by range-relative cells (the price census: all-in
        // bids are common, openings run past a 20-step ladder).
        BlockSpec::new("Bid.corporation", Family::BidCorporation, vec![minors]).priced(PriceKind::BidRange),
        BlockSpec::new("Par", Family::Par, vec![majors.clone(), Axis::par_prices(title, CorpType::Major)]),
        BlockSpec::new(
            "BuyShares",
            Family::BuyShares,
            vec![majors.clone(), Choice(vec!["source:ipo", "source:market"])],
        ),
        BlockSpec::new("SellShares", Family::SellShares, vec![majors, Param(percents.clone())]),
        BlockSpec::new("RedeemShares", Family::RedeemShares, vec![Param(percents)]),
        BlockSpec::new("PlaceToken", Family::PlaceToken, vec![City(cities.clone())]),
        BlockSpec::new("RemoveToken", Family::RemoveToken, vec![City(cities)]),
        BlockSpec::new("BuyTrain.depot", Family::BuyTrainDepot, vec![trains()]),
        BlockSpec::new("BuyTrain.pool", Family::BuyTrainPool, vec![trains()]),
        BlockSpec::new(
            "BuyTrain.exchange",
            Family::BuyTrainExchange,
            vec![
                Ent { slot: 0, kind: EntKind::Train, syms: new_trains },
                Ent { slot: 1, kind: EntKind::Train, syms: traded },
                Choice(TRAIN_SOURCES.to_vec()),
            ],
        ),
        BlockSpec::new(
            "BuyTrain.corp",
            Family::BuyTrainCorp,
            vec![
                Ent { slot: 0, kind: EntKind::Corporation, syms: operating.clone() },
                Axis::trains(1, title),
            ],
        )
        .priced(PriceKind::TrainTransfer),
        BlockSpec::new("DiscardTrain", Family::DiscardTrain, vec![trains()]),
        BlockSpec::new("Dividend", Family::Dividend, vec![Choice(DIVIDENDS.to_vec())]),
        BlockSpec::new("BuyCompany", Family::BuyCompany, vec![companies]).priced(PriceKind::CompanyPurchase),
        BlockSpec::new("RunRoutes", Family::RunRoutes, vec![]),
        BlockSpec::new(
            "Merge",
            Family::Merge,
            vec![Ent { slot: 0, kind: EntKind::Corporation, syms: operating }],
        ),
        BlockSpec::new("Convert", Family::Convert, vec![]),
        BlockSpec::new("Choose", Family::Choose, vec![Choice(vec!["choose:nationalize"])]),
        BlockSpec::new("LayTile", Family::LayTile, vec![Axis::map_hexes(title), Axis::tiles(title), Rotation]),
    ])
}

/// The layout.
pub fn registry(title: &'static dyn GameTitle) -> Result<RlLayout, String> {
    RlLayout::build(title.name(), VERSION, Vocab::for_title(title), block_specs(title)?)
}

/// City slots `(hex, city)` in map order: every city index a hex's tile
/// can have, i.e. up to the most cities over the hex's upgrade closure (its
/// initial tile and every manifest tile reachable from it by Ruby's
/// `upgrades_to?`, phases and tile supply ignored). L12 (Montreal) reaches
/// three. A token's `city_index` is the city's position on the hex's
/// current tile, so `(hex, city)` names one city of the live map.
pub fn city_slots(title: &dyn GameTitle) -> Vec<(&'static str, usize)> {
    let ids: Vec<u32> = vec![1, 2, 3];
    let names = ids.iter().map(|&i| (i, format!("Player {i}"))).collect();
    let mut g = BaseGame::build_titled(title.name(), ids, names);
    let manifest: Vec<&'static str> = title.tile_counts().iter().map(|(t, _)| *t).collect();
    let mut out = Vec::new();
    for (hi, def) in title.hex_definitions().iter().enumerate() {
        assert_eq!(g.hexes[hi].id, def.coord, "BaseGame.hexes follows the title's hex order");
        let initial = g.hexes[hi].tile.clone();
        let mut most = initial.cities.len();
        let mut seen: HashSet<&'static str> = HashSet::new();
        let mut frontier = vec![initial.clone()];
        while let Some(t) = frontier.pop() {
            g.hexes[hi].tile = t;
            for name in &manifest {
                if seen.contains(name) {
                    continue;
                }
                let d = g.tile_catalog[*name].clone();
                if g.g1867_upgrades_to(hi, &d) {
                    seen.insert(name);
                    let next = BaseGame::tile_from_def(&d, 0);
                    most = most.max(next.cities.len());
                    frontier.push(next);
                }
            }
        }
        g.hexes[hi].tile = initial;
        out.extend((0..most).map(|c| (def.coord, c)));
    }
    out
}

/// A tile instance id's tile name ("57-3" -> "57"; the generator's
/// instances are "<name>-<n>").
fn tile_name(id: &str) -> &str {
    match id.rsplit_once('-') {
        Some((name, n)) if !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()) => name,
        _ => id,
    }
}

/// Where a train (by id) is: the depot's upcoming trains, the discard pool,
/// or a corporation's.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum TrainAt {
    Depot,
    Pool,
    Corp,
    Unknown,
}

fn train_lookup(g: &BaseGame, id: &str) -> (Option<String>, TrainAt) {
    if let Some(t) = g.depot.trains.iter().find(|t| t.id == id) {
        return (Some(t.name.clone()), TrainAt::Depot);
    }
    if let Some(t) = g.depot.discarded.iter().find(|t| t.id == id) {
        return (Some(t.name.clone()), TrainAt::Pool);
    }
    for c in &g.corporations {
        if let Some(t) = c.trains.iter().find(|t| t.id == id) {
            return (Some(t.name.clone()), TrainAt::Corp);
        }
    }
    (None, TrainAt::Unknown)
}

/// The block and axis entries of a move: the index function of `1867-v1`
/// (with [`RlLayout::index_of`]). Reads the state for what the action
/// names by id (a train's type and whether it sits in the depot or the
/// pool, a traded train's type).
pub fn move_key(g: &BaseGame, m: &Move) -> Result<(&'static str, Vec<String>), GameError> {
    use Action as A;
    let err = |what: &str| GameError::new(format!("1867-v1 cannot index {:?}: {what}", m.action));
    let name_of = |id: &str, variant: &Option<String>| -> Result<(String, TrainAt), GameError> {
        let (name, at) = train_lookup(g, id);
        match name.or_else(|| variant.clone()) {
            Some(n) => Ok((n, at)),
            None => Err(err("unknown train")),
        }
    };
    Ok(match &m.action {
        A::Pass { .. } => ("Pass", vec![]),
        A::Bid { company_sym, .. } => ("Bid.company", vec![company_sym.clone()]),
        A::CorporationBid { corporation_sym, .. } => ("Bid.corporation", vec![corporation_sym.clone()]),
        A::Par { corporation_sym, share_price, .. } => ("Par", vec![corporation_sym.clone(), share_price.to_string()]),
        // A corporation buying its own pool shares is a redemption (1867's
        // only corporation buy).
        A::BuyShares { entity_id, corporation_sym, percent, source, .. } => {
            if entity_id == corporation_sym || m.step == StepKind::RedeemShares {
                ("RedeemShares", vec![percent.to_string()])
            } else {
                let choice = match source.as_str() {
                    "ipo" => "source:ipo",
                    "market" => "source:market",
                    _ => return Err(err("share source")),
                };
                ("BuyShares", vec![corporation_sym.clone(), choice.to_string()])
            }
        }
        A::SellShares { corporation_sym, percent, .. } => {
            ("SellShares", vec![corporation_sym.clone(), percent.to_string()])
        }
        A::PlaceToken { hex_id, city_index, .. } => ("PlaceToken", vec![format!("{hex_id}.{city_index}")]),
        A::RemoveToken { hex_id, city_index, .. } => ("RemoveToken", vec![format!("{hex_id}.{city_index}")]),
        A::LayTile { hex_id, tile_id, rotation, .. } => {
            ("LayTile", vec![hex_id.clone(), tile_name(tile_id).to_string(), rotation.to_string()])
        }
        A::BuyTrain { entity_id, train_name, from, variant, exchange, .. } => {
            let (name, at) = name_of(train_name, variant)?;
            if from != "depot" {
                ("BuyTrain.corp", vec![from.clone(), name])
            } else if let Some(x) = exchange {
                let traded = g
                    .corp_idx
                    .get(entity_id.as_str())
                    .and_then(|&ci| g.corporations[ci].trains.iter().find(|t| t.id == *x))
                    .map(|t| t.name.clone())
                    .ok_or_else(|| err("the traded train is not the buyer's"))?;
                let source = match at {
                    TrainAt::Depot => TRAIN_SOURCES[0],
                    TrainAt::Pool => TRAIN_SOURCES[1],
                    _ => return Err(err("a trade-in's new train is neither in the depot nor in the pool")),
                };
                ("BuyTrain.exchange", vec![name, traded, source.to_string()])
            } else {
                match at {
                    TrainAt::Pool => ("BuyTrain.pool", vec![name]),
                    TrainAt::Depot => ("BuyTrain.depot", vec![name]),
                    _ => return Err(err("a depot purchase of a train outside the depot")),
                }
            }
        }
        A::DiscardTrain { train_name, .. } => ("DiscardTrain", vec![name_of(train_name, &None)?.0]),
        A::Dividend { kind, .. } => {
            let k = match kind {
                DividendKind::Payout => DIVIDENDS[0],
                DividendKind::Half => DIVIDENDS[1],
                DividendKind::Withhold => DIVIDENDS[2],
            };
            ("Dividend", vec![k.to_string()])
        }
        A::BuyCompany { company_sym, .. } => ("BuyCompany", vec![company_sym.clone()]),
        A::RunRoutes { .. } => ("RunRoutes", vec![]),
        A::Merge { corporation_sym, .. } => ("Merge", vec![corporation_sym.clone()]),
        A::Convert { .. } => ("Convert", vec![]),
        A::Choose { choice, .. } => ("Choose", vec![format!("choose:{choice}")]),
        _ => return Err(err("no 1867 block for this action type")),
    })
}

/// The index of a generator move in `1867-v1`.
pub fn move_index(lo: &RlLayout, g: &BaseGame, m: &Move) -> Result<u32, GameError> {
    let (block, entries) = move_key(g, m)?;
    let refs: Vec<&str> = entries.iter().map(String::as_str).collect();
    lo.index_of(block, &refs)
        .ok_or_else(|| GameError::new(format!("1867-v1: {block} {entries:?} is not in the layout ({:?})", m.action)))
}

#[cfg(test)]
mod tests;
