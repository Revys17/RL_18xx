//! The 1822 family's RL layouts (`1822-v1`, `1822PNW-v1`, `1822MX-v1`,
//! `1822CA-v1`; `docs/rl_descriptors.md` §9): registry layouts whose blocks
//! and axes come from each title's data (its `GameTitle` and its 1822
//! family table: the companies' tile-lay abilities and combos, the
//! companies named for each ability role, the extra trains), and the index
//! function that maps a move of the family's legal-move generator
//! (`g1822/legal.rs` and the variants' `*_legal.rs`) to its index.
//!
//! Every generated move maps to exactly one index; a state's moves map to
//! distinct indices (the fuzz in the tests and the corpus check of
//! `scripts/g1822_completeness.py --index` verify both). A price-bearing
//! move is ONE index whose legal range comes from the move (`PriceRange {
//! lo, hi, step }`); its price goes to a price head (`PriceKind`), never to
//! a menu of prices. What a move needs beyond its index is state: which
//! minor an acquisition choice is about, whose pass it is, a token's cost.
//!
//! Schema extensions this family uses (all append-only, §9 of the doc):
//!
//! * **Company lays and combos**: a company's lay has `via` = the company;
//!   a lay that combines a second company's ability (Ruby
//!   `combo_entities`) has that company in `ent0` (kind company). One block
//!   per lay company and per (company, partner) combo, over the hexes and
//!   tiles the abilities allow (all of them where an ability names none).
//! * **Choices that index a state list**: the pullman / grain attachments
//!   pick a train of the corporation's: `ent0` = its type, `param` = its
//!   rank among the corporation's trains of that type (the choice's
//!   position in the state's list is not in the descriptor).
//! * **A bid on a token the state names** (1822MX's NdeM auction):
//!   family `Bid.token` with the token's `hex` / `city`.
//! * **Pseudo-tiles**: builder cubes (`BC`) and port tiles (`P1`, `P2`)
//!   are in the PNW / MX tile manifests, so the tile vocabulary has them.
//! * **Named choices** appended to the global choice vocabulary.

use std::collections::BTreeSet;

use super::descriptor::{flags, EntKind, Family, PriceKind};
use super::layout::{Axis, BlockSpec, RlLayout, Vocab};
use crate::actions::{Action, DividendKind, GameError};
use crate::g1822::family::{Family as G1822Family, Variant};
use crate::g1822::legal::Move;
use crate::g1822::track::AbKind;
use crate::g1822::{MergeState, Step};
use crate::game::BaseGame;
use crate::title::{CorpType, GameTitle, MarketZone};

/// The layout version of each title. Bump it (and re-convert data)
/// whenever an index changes meaning; the fingerprint test pins it.
pub fn version(title: &str) -> String {
    format!("{title}-v1")
}

const DIVIDENDS: [&str; 3] = ["dividend:payout", "dividend:half", "dividend:withhold"];
const SOURCES: [&str; 2] = ["source:ipo", "source:market"];
/// The pullman / grain train's rank among the corporation's trains of its
/// type (a corporation never holds more than its train limit, 4, of one
/// type; 6 ranks leave room for an over-limit state).
const RANKS: [i32; 6] = [0, 1, 2, 3, 4, 5];
/// Share counts of 1822PNW's conversion and merger choices
/// (`possible_exchanged_shares`: 2..=6).
const SHARE_COUNTS: [i32; 6] = [1, 2, 3, 4, 5, 6];

fn fam(title: &dyn GameTitle) -> Result<&'static G1822Family, String> {
    title
        .g1822_family()
        .ok_or_else(|| format!("{}: not an 1822-family title", title.name()))
}

/// A company's tile-lay abilities (tile_lay / teleport), merged: the hexes
/// and tiles they allow (empty = any) and their combo partners.
#[derive(Clone, Debug, Default)]
struct LayAbility {
    hexes: Vec<String>,
    tiles: Vec<String>,
    combo: Vec<String>,
}

fn lay_abilities(f: &G1822Family) -> Vec<(String, LayAbility)> {
    let mut out = Vec::new();
    for (co, abs) in (f.abilities)() {
        let lays: Vec<_> = abs
            .iter()
            .filter(|a| matches!(a.kind, AbKind::TileLay | AbKind::Teleport))
            .collect();
        if lays.is_empty() {
            continue;
        }
        let any_hex = lays.iter().any(|a| a.hexes.is_empty());
        let any_tile = lays.iter().any(|a| a.tiles.is_empty());
        let mut la = LayAbility::default();
        for a in &lays {
            if !any_hex {
                la.hexes.extend(a.hexes.iter().cloned());
            }
            if !any_tile {
                la.tiles.extend(a.tiles.iter().cloned());
            }
            la.combo.extend(a.combo.iter().cloned());
        }
        la.hexes.sort();
        la.hexes.dedup();
        let mut seen = BTreeSet::new();
        la.combo.retain(|c| seen.insert(c.clone()));
        out.push((co, la));
    }
    out
}

/// Companies whose lays only place a builder cube (`BC`): 1822PNW's P10,
/// 1822MX's P10 / P11 (their generators special-case them).
fn cube_only(f: &G1822Family, co: &str) -> bool {
    match f.variant {
        Variant::Pnw => co == "P10",
        Variant::Mx => BaseGame::g1822_mx_cube_company(co),
        _ => false,
    }
}

/// Companies whose lays the variant's generator special-cases (they never
/// take the shared path's extras: `PNW5` over a coal mine, a builder cube).
fn special_lay(f: &G1822Family, co: &str) -> bool {
    match f.variant {
        Variant::Pnw => matches!(co, "P10" | "P12" | "P13" | "P16" | "P18" | "P19"),
        Variant::Mx => BaseGame::g1822_mx_cube_company(co) || BaseGame::g1822_mx_port_company(co),
        _ => false,
    }
}

/// The tiles a lay by `co` (with `partner`'s ability combined) can place,
/// in manifest order: Ruby `SpecialTrack#potential_tiles` takes the union
/// of the abilities' tile lists, or the default tiles when no ability
/// names any; cube companies place only `BC`; the shared path may add
/// `PNW5` (over a PNW4 coal mine) and a cube (`BC`, an extra-track
/// company) in the titles that have them.
fn lay_tiles(
    f: &G1822Family,
    manifest: &[&'static str],
    co: &str,
    abilities: &[&LayAbility],
    partner: Option<&str>,
) -> Vec<&'static str> {
    if cube_only(f, co) {
        return manifest.iter().copied().filter(|t| *t == "BC").collect();
    }
    let named: BTreeSet<&str> = abilities
        .iter()
        .flat_map(|a| a.tiles.iter().map(String::as_str))
        .collect();
    if named.is_empty() {
        return manifest.to_vec();
    }
    let extras: &[&str] = if special_lay(f, co) || partner.map_or(false, |p| special_lay(f, p)) {
        &[]
    } else {
        &["BC", "PNW5"]
    };
    manifest
        .iter()
        .copied()
        .filter(|t| named.contains(t) || extras.contains(t))
        .collect()
}

/// City slots `(hex, city)` in map order: every hex gets as many slots as
/// the most cities any tile of the title has (its manifest and every
/// preprinted tile), so no upgrade, special lay or new token city (a
/// company's extra slot is a token in one of the city's slots) can name a
/// slot the layout lacks. A token's `city` is the city's position on the
/// hex's current tile.
pub fn city_slots(title: &dyn GameTitle) -> Vec<(&'static str, usize)> {
    let ids: Vec<u32> = vec![1001, 1002, 1003];
    let names = ids.iter().map(|&i| (i, format!("Player {i}"))).collect();
    let g = BaseGame::build_titled(title.name(), ids, names);
    let catalog = title.tile_catalog();
    let mut most = 1usize;
    for (name, _) in title.tile_counts() {
        if let Some(d) = catalog.get(name) {
            most = most.max(BaseGame::tile_from_def(d, 0).cities.len());
        }
    }
    for h in &g.hexes {
        most = most.max(h.tile.cities.len());
    }
    let mut out = Vec::new();
    for def in title.hex_definitions() {
        out.extend((0..most).map(|c| (def.coord, c)));
    }
    out
}

/// Whether a company is a minor's (the bid boxes' "M<minor>": 1822's M1..M30,
/// PNW's regionals MA..MC), given whether a sym names a minor corporation.
fn is_minor_company(co: &str, is_minor: impl Fn(&str) -> bool) -> bool {
    co.len() > 1 && co.starts_with('M') && is_minor(&co[1..])
}

/// The market's par prices (cells of any par type), ascending.
fn par_prices(title: &dyn GameTitle) -> Vec<i32> {
    let mut v: Vec<i32> = title
        .market_grid()
        .iter()
        .flatten()
        .flatten()
        .filter(|c| {
            c.zones
                .iter()
                .any(|z| matches!(z, MarketZone::Par | MarketZone::Par1 | MarketZone::Par2))
        })
        .map(|c| c.price)
        .collect();
    v.sort_unstable();
    v.dedup();
    v
}

fn nonempty(v: &[&'static str]) -> Vec<&'static str> {
    let mut out: Vec<&'static str> = Vec::new();
    for s in v {
        if !s.is_empty() && !out.contains(s) {
            out.push(s);
        }
    }
    out
}

/// The block specs of the title's layout, in index order (small blocks
/// first, the tile lays last).
pub fn block_specs(title: &dyn GameTitle) -> Result<Vec<BlockSpec>, String> {
    use Axis::*;
    let f = fam(title)?;
    let vocab = Vocab::for_title(title);
    let majors = Axis::corporations(0, title, Some(CorpType::Major));
    let minors = Axis::corporations(0, title, Some(CorpType::Minor));
    let corps = || Axis::corporations(0, title, None);
    let companies = || Axis::companies(0, title, false);
    let trains = |slot: u8| Axis::trains(slot, title);
    let has_train = |n: &str| vocab.trains.contains(&n);
    let unit = title
        .corporations()
        .iter()
        .filter(|c| c.corp_type == CorpType::Major)
        .flat_map(|c| c.shares.iter().copied())
        .min()
        .ok_or("no major shares")? as i32;
    // Every multiple of a share unit up to 200%: 1822MX's NdeM gains a
    // share for every minor it replaces (it redeemed 110% in the fuzz);
    // the others never move more than 100% (the pool limit keeps them
    // lower).
    let percents: Vec<i32> = (1..=200 / unit).map(|k| k * unit).collect();
    let cities = city_slots(title);
    let hexes = || Axis::map_hexes(title);
    let pars = par_prices(title);
    let v = f.variant;

    // Bid boxes (step $5 from the move): a concession's or a private's bid
    // prices on the "Bid" ladder (99.9% of them within 20 steps), a
    // minor's on range-relative cells (9% of minor bids run past a 20-step
    // ladder; the price census, docs/rl_descriptors.md §9).
    let minors_of_title: BTreeSet<&str> = title
        .corporations()
        .iter()
        .filter(|c| c.corp_type == CorpType::Minor)
        .map(|c| c.sym)
        .collect();
    let (minor_cos, other_cos): (Vec<&'static str>, Vec<&'static str>) = vocab
        .companies
        .iter()
        .copied()
        .partition(|co| is_minor_company(co, |m| minors_of_title.contains(m)));
    let company_ent = |syms: Vec<&'static str>| Ent { slot: 0, kind: EntKind::Company, syms };

    let mut s: Vec<BlockSpec> = vec![
        BlockSpec::new("Pass", Family::Pass, vec![]),
        BlockSpec::new("Bid.company", Family::BidCompany, vec![company_ent(other_cos)]).priced(PriceKind::Bid),
        BlockSpec::new("Bid.minor", Family::BidCompany, vec![company_ent(minor_cos)]).priced(PriceKind::BidRange),
        BlockSpec::new("Par", Family::Par, vec![majors.clone(), Param(pars.clone())]),
        BlockSpec::new(
            "BuyShares",
            Family::BuyShares,
            vec![majors.clone(), Choice(SOURCES.to_vec())],
        ),
    ];
    // 1822MX: the presidentless NdeM's first certificate is a 10% share the
    // market offers beside its plain ones (two moves, two indices).
    if v == Variant::Mx {
        s.push(
            BlockSpec::new(
                "BuyShares.president",
                Family::BuyShares,
                vec![
                    Ent {
                        slot: 0,
                        kind: EntKind::Corporation,
                        syms: vec![crate::g1822::mx::NDEM],
                    },
                    Choice(SOURCES.to_vec()),
                ],
            )
            .flagged(flags::PRESIDENT_CERT),
        );
    }
    s.extend([
        BlockSpec::new(
            "SellShares",
            Family::SellShares,
            vec![majors.clone(), Param(percents.clone())],
        ),
        BlockSpec::new("RedeemShares", Family::RedeemShares, vec![Param(percents.clone())]),
        BlockSpec::new("IssueShares", Family::IssueShares, vec![Param(percents)]),
        BlockSpec::new("PayoffPlayerDebt", Family::PayoffPlayerDebt, vec![]),
        BlockSpec::new("PayoffPlayerDebtPartial", Family::PayoffPlayerDebtPartial, vec![]).priced(PriceKind::Amount),
    ]);
    // The tax haven (1822 P16, CA P11): a share of a major (ent0) into the
    // haven, from the IPO or the market.
    if !f.company_osth.is_empty() {
        s.push(BlockSpec::new(
            "ChooseAbility.tax_haven",
            Family::ChooseAbility,
            vec![Via(vec![f.company_osth]), majors.clone(), Choice(SOURCES.to_vec())],
        ));
    }
    // A bank company (ent0) to the top / bottom of its stack (1822 LUR P17,
    // PNW P21).
    if !f.company_lur.is_empty() {
        s.push(BlockSpec::new(
            "ChooseAbility.stack",
            Family::ChooseAbility,
            vec![
                Via(vec![f.company_lur]),
                companies(),
                Choice(vec!["stack:top", "stack:bottom"]),
            ],
        ));
    }
    // The other ability choices (G1822#company_choices and the variants'
    // overrides): an exchange token (1822 LCDR / CHPR / LUR, CA CHPR, PNW
    // P21), MX P16's closure (MX's company choices are P7's double alone).
    let (ability_cos, ability_choices) = match v {
        Variant::G1822 | Variant::Ca => (
            nonempty(&[f.company_lcdr, f.company_chpr, f.company_lur]),
            vec!["ability:exchange", "ability:available"],
        ),
        Variant::Pnw => (nonempty(&[f.company_lur]), vec!["ability:exchange"]),
        Variant::Mx => (vec!["P16"], vec!["ability:close"]),
    };
    if !ability_cos.is_empty() {
        s.push(BlockSpec::new(
            "ChooseAbility",
            Family::ChooseAbility,
            vec![Via(ability_cos), Choice(ability_choices)],
        ));
    }
    // The choices round's double cash (PNW's company choices are P21's
    // alone: no double).
    if !f.company_double_cash.is_empty() && v != Variant::Pnw {
        s.push(BlockSpec::new(
            "Choose.double",
            Family::Choose,
            vec![Via(vec![f.company_double_cash]), Choice(vec!["choose:double"])],
        ));
    }
    // EGR's (1822 P8) token-or-discount choice (the variants' company
    // choices have none).
    if !f.company_egr.is_empty() && v == Variant::G1822 {
        s.push(BlockSpec::new(
            "Choose.egr",
            Family::Choose,
            vec![
                Via(vec![f.company_egr]),
                Choice(vec!["ability:token", "ability:discount"]),
            ],
        ));
    }
    // Tokens.
    s.push(BlockSpec::new(
        "PlaceToken",
        Family::PlaceToken,
        vec![City(cities.clone())],
    ));
    // A company's token: LCDR (1822 P5) in the English Channel, CA P10 in
    // Winnipeg.
    let mut token_cos: Vec<(&'static str, &'static str)> = Vec::new();
    if !f.company_lcdr.is_empty() && !f.english_channel_hex.is_empty() {
        token_cos.push((f.company_lcdr, f.english_channel_hex));
    }
    if v == Variant::Ca {
        token_cos.push((crate::g1822::ca::COMPANY_WINNIPEG_TOKEN, "N16"));
    }
    for (co, hex) in token_cos {
        let slots: Vec<(&'static str, usize)> = cities.iter().copied().filter(|(h, _)| *h == hex).collect();
        s.push(BlockSpec::new(
            format!("PlaceToken:{co}"),
            Family::PlaceToken,
            vec![Via(vec![co]), City(slots)],
        ));
    }
    // Destination tokens: 1822's hex token, CA's city choice.
    s.push(BlockSpec::new(
        "HexToken",
        Family::HexToken,
        vec![hexes(), Choice(vec!["token:destination"])],
    ));
    if v == Variant::Ca {
        s.push(BlockSpec::new(
            "PlaceToken.destination",
            Family::PlaceToken,
            vec![City(cities.clone()), Choice(vec!["token:destination"])],
        ));
    }
    // 1822MX: the NdeM's token auction -- the chooser's token, the bid on it.
    if v == Variant::Mx {
        s.push(BlockSpec::new(
            "RemoveToken",
            Family::RemoveToken,
            vec![City(cities.clone())],
        ));
        s.push(BlockSpec::new("Bid.token", Family::BidToken, vec![City(cities.clone())]).priced(PriceKind::Bid));
    }
    // Trains.
    s.push(BlockSpec::new("BuyTrain.depot", Family::BuyTrainDepot, vec![trains(0)]));
    s.push(BlockSpec::new("BuyTrain.pool", Family::BuyTrainPool, vec![trains(0)]));
    // The L -> 2 upgrade of an owned L train.
    if has_train("L") && has_train("2") {
        s.push(BlockSpec::new(
            "BuyTrain.exchange",
            Family::BuyTrainExchange,
            vec![
                Ent {
                    slot: 0,
                    kind: EntKind::Train,
                    syms: vec!["2"],
                },
                Ent {
                    slot: 1,
                    kind: EntKind::Train,
                    syms: vec!["L"],
                },
            ],
        ));
    }
    s.push(
        BlockSpec::new("BuyTrain.corp", Family::BuyTrainCorp, vec![corps(), trains(1)])
            .priced(PriceKind::TrainTransfer),
    );
    s.push(BlockSpec::new("DiscardTrain", Family::DiscardTrain, vec![trains(0)]));
    // Dividends: the run, and the permanent extra train's own part.
    s.push(BlockSpec::new(
        "Dividend",
        Family::Dividend,
        vec![Choice(DIVIDENDS.to_vec())],
    ));
    let extra: Vec<&'static str> = f
        .extra_train_permanents
        .iter()
        .copied()
        .filter(|t| has_train(t))
        .collect();
    if !extra.is_empty() {
        s.push(BlockSpec::new(
            "Dividend.extra",
            Family::Dividend,
            vec![
                Ent {
                    slot: 0,
                    kind: EntKind::Train,
                    syms: extra,
                },
                Choice(DIVIDENDS.to_vec()),
            ],
        ));
    }
    s.push(BlockSpec::new("RunRoutes", Family::RunRoutes, vec![]));
    // The Route step's attachments: the pullman (and CA's grain train) to a
    // train (type, rank), or CA's skip.
    s.push(BlockSpec::new(
        "Choose.pullman",
        Family::Choose,
        vec![Choice(vec!["route:pullman"]), trains(0), Param(RANKS.to_vec())],
    ));
    if v == Variant::Ca {
        s.push(BlockSpec::new(
            "Choose.grain",
            Family::Choose,
            vec![Choice(vec!["route:grain"]), trains(0), Param(RANKS.to_vec())],
        ));
        s.push(BlockSpec::new(
            "Choose.route_skip",
            Family::Choose,
            vec![Choice(vec!["route:skip"])],
        ));
    }
    // Acquisitions: a private (AcquireCompany), a minor (MinorAcquisition,
    // PNW's Merger round) or the NdeM winner's major (Merge), and the
    // acquisition's pay / token choices.
    s.push(BlockSpec::new(
        "AcquireCompany",
        Family::AcquireCompany,
        vec![companies()],
    ));
    s.push(BlockSpec::new("Merge", Family::Merge, vec![corps()]));
    s.push(BlockSpec::new(
        "Choose.acquire",
        Family::Choose,
        vec![Choice(vec![
            "merge_pay:money",
            "merge_pay:one_share",
            "merge_pay:two_shares",
            "merge_token:replace",
            "merge_token:exchange",
            "merge_token:remove",
        ])],
    ));
    if v == Variant::Pnw {
        s.push(BlockSpec::new(
            "Choose.convert",
            Family::Choose,
            vec![Choice(vec!["convert:shares"]), Param(SHARE_COUNTS.to_vec())],
        ));
        s.push(BlockSpec::new(
            "Choose.merge_par",
            Family::Choose,
            vec![Choice(vec!["merge:par"]), Param(pars)],
        ));
        s.push(BlockSpec::new(
            "Choose.merge_shares",
            Family::Choose,
            vec![Choice(vec!["merge:shares"]), Param(SHARE_COUNTS.to_vec())],
        ));
        s.push(BlockSpec::new(
            "Choose.replace_minor",
            Family::Choose,
            vec![Choice(vec!["choose:replace_minor"]), minors],
        ));
    }
    // Assign: PNW's paper mill / ski haus, CA's sawmill.
    let assign_cos: Vec<&'static str> = match v {
        Variant::Pnw => vec!["P15", "P17"],
        Variant::Ca => vec![crate::g1822::ca::COMPANY_SAWMILL],
        _ => vec![],
    };
    if !assign_cos.is_empty() {
        s.push(BlockSpec::new("Assign", Family::Assign, vec![Via(assign_cos), hexes()]));
    }
    if v == Variant::Ca {
        s.push(BlockSpec::new(
            "Choose.sawmill",
            Family::Choose,
            vec![
                Via(vec![crate::g1822::ca::COMPANY_SAWMILL]),
                Choice(vec!["sawmill:open", "sawmill:closed"]),
            ],
        ));
    }
    // Tile lays: the operating corporation's, then each company's and each
    // (company, partner) combo's over the hexes and tiles its abilities
    // allow.
    s.push(BlockSpec::new(
        "LayTile",
        Family::LayTile,
        vec![hexes(), Axis::tiles(title), Rotation],
    ));
    let manifest = vocab.tiles.clone();
    let abilities = lay_abilities(f);
    let lay_of = |co: &str| abilities.iter().find(|(c, _)| c == co).map(|(_, a)| a);
    for (co, la) in &abilities {
        let co: &'static str = vocab
            .companies
            .iter()
            .copied()
            .find(|c| c == co)
            .ok_or_else(|| format!("lay company {co} is not in the company vocabulary"))?;
        let hex_axis = |la: &LayAbility| -> Result<Axis, String> {
            if la.hexes.is_empty() {
                return Ok(hexes());
            }
            let hs: BTreeSet<&str> = la.hexes.iter().map(String::as_str).collect();
            Ok(Hex(vocab.hexes.iter().copied().filter(|h| hs.contains(h)).collect()))
        };
        let tiles = lay_tiles(f, &manifest, co, &[la], None);
        s.push(BlockSpec::new(
            format!("LayTile:{co}"),
            Family::LayTile,
            vec![Via(vec![co]), hex_axis(la)?, Tile(tiles), Rotation],
        ));
        for p in &la.combo {
            let Some(pla) = lay_of(p) else { continue };
            let p: &'static str = vocab
                .companies
                .iter()
                .copied()
                .find(|c| c == p)
                .ok_or_else(|| format!("combo partner {p} is not in the company vocabulary"))?;
            let tiles = lay_tiles(f, &manifest, co, &[la, pla], Some(p));
            s.push(BlockSpec::new(
                format!("LayTile:{co}+{p}"),
                Family::LayTile,
                vec![
                    Via(vec![co]),
                    Ent {
                        slot: 0,
                        kind: EntKind::Company,
                        syms: vec![p],
                    },
                    hex_axis(la)?,
                    Tile(tiles),
                    Rotation,
                ],
            ));
        }
    }
    Ok(s)
}

/// The layout.
pub fn registry(title: &'static dyn GameTitle) -> Result<RlLayout, String> {
    RlLayout::build(
        title.name(),
        version(title.name()),
        Vocab::for_title(title),
        block_specs(title)?,
    )
}

/// A tile instance id's tile name ("57-3" -> "57", "BC-0" -> "BC").
pub(crate) fn tile_name(id: &str) -> &str {
    match id.rsplit_once('-') {
        Some((name, n)) if !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()) => name,
        _ => id,
    }
}

/// The type a train name stands for in the descriptors: 1822CA's grain
/// suffixes removed. A grain name can outlive its attachment (Ruby's
/// Route#setup drops `@grain_train` without detaching it when the attachment
/// left the corporation without a run: `g1822_ca_route_setup`), and the train
/// is then offered under it -- to the grain or the pullman, from another
/// corporation, to discard (a 2026-11-14 policy game: "Choose.grain
/// ["route:grain", "LP-G", "0"] is not in the layout").
pub(crate) fn train_type(g: &BaseGame, name: &str) -> String {
    if g.g1822_is_ca() {
        BaseGame::g1822_ca_ungrained(name).to_string()
    } else {
        name.to_string()
    }
}

/// The type of a train by id ([`train_type`] of its name): a corporation's,
/// the depot's or the pool's.
fn train_name(g: &BaseGame, id: &str) -> Option<String> {
    for c in &g.corporations {
        if let Some(t) = c.trains.iter().find(|t| t.id == id) {
            return Some(train_type(g, &t.name));
        }
    }
    g.depot
        .trains
        .iter()
        .chain(g.depot.discarded.iter())
        .find(|t| t.id == id)
        .map(|t| train_type(g, &t.name))
}

/// A train's type in the Route step: an attachment renames its train for the
/// run (the pullman "<name>+" / CA "<name>+<2d>", CA's grain train
/// "<name>G" / "<name>-G"; corpus 255116 @1481 offers the pullman to a
/// grain-attached "LP-G") -- the type is the name before it ([`train_type`]).
pub(crate) fn route_train_type(g: &BaseGame, id: &str) -> Option<String> {
    let r = g.rnd();
    if r.pullman_train.as_deref() == Some(id) {
        if let Some((name, _)) = &r.pullman_original {
            return Some(train_type(g, name));
        }
    }
    if r.ca.grain_train.as_deref() == Some(id) {
        if let Some(name) = &r.ca.grain_original {
            return Some(train_type(g, name));
        }
    }
    train_name(g, id)
}

/// The (type, rank) of the `i`-th train of a choice list of train ids: its
/// type ([`route_train_type`]) and how many earlier entries share it.
pub(crate) fn train_rank(g: &BaseGame, list: &[String], i: usize) -> Option<(String, usize)> {
    let name = route_train_type(g, list.get(i)?)?;
    let rank = list[..i]
        .iter()
        .filter(|id| route_train_type(g, id).as_deref() == Some(name.as_str()))
        .count();
    Some((name, rank))
}

/// The block and axis entries of a move: the index function of the
/// family's layouts (with [`RlLayout::index_of`]). Reads the state for what
/// the action names by id or position (a train's type and whether it sits
/// in the pool, a pullman choice's train, the token up for auction, the
/// step a choice belongs to).
pub fn move_key(g: &BaseGame, m: &Move) -> Result<(String, Vec<String>), GameError> {
    use Action as A;
    let err = |what: &str| GameError::new(format!("{}-v1 cannot index {:?}: {what}", g.title, m.action));
    let step = g.g1822_active_step().map(|i| g.rnd().steps[i]);
    let f = g.fam();
    let is_company = |id: &str| g.co_idx(id).is_some() && !g.corp_idx.contains_key(id);
    let k = |b: &str, e: Vec<String>| (b.to_string(), e);
    Ok(match &m.action {
        A::Pass { .. } => k("Pass", vec![]),
        A::Bid { company_sym, .. } if company_sym.is_empty() => {
            // 1822MX: the NdeM token up for auction.
            let a = g.rnd().ndem_auction.as_ref().ok_or_else(|| err("no NdeM auction"))?;
            let t =
                a.up.and_then(|u| a.available.get(u))
                    .ok_or_else(|| err("no token up for auction"))?;
            k("Bid.token", vec![format!("{}.{}", t.hex, t.city)])
        }
        A::Bid { company_sym, .. } => {
            let minor = is_minor_company(company_sym, |m| {
                g.corp_idx.get(m).map_or(false, |&ci| g.corporations[ci].corp_type == CorpType::Minor)
            });
            k(if minor { "Bid.minor" } else { "Bid.company" }, vec![company_sym.clone()])
        }
        A::Par {
            corporation_sym,
            share_price,
            ..
        } => k("Par", vec![corporation_sym.clone(), share_price.to_string()]),
        A::BuyShares {
            entity_id,
            corporation_sym,
            percent,
            source,
            share_indices,
            ..
        } => {
            if entity_id == corporation_sym {
                k("RedeemShares", vec![percent.to_string()])
            } else {
                let src = match source.as_str() {
                    "ipo" => SOURCES[0],
                    "market" => SOURCES[1],
                    _ => return Err(err("share source")),
                };
                let president = g.corp_idx.get(corporation_sym.as_str()).map_or(false, |&ci| {
                    share_indices
                        .iter()
                        .any(|&i| g.corporations[ci].shares.get(i).map_or(false, |s| s.president))
                });
                let block = if president { "BuyShares.president" } else { "BuyShares" };
                k(block, vec![corporation_sym.clone(), src.to_string()])
            }
        }
        A::SellShares {
            entity_id,
            corporation_sym,
            percent,
            ..
        } => {
            if entity_id == corporation_sym {
                k("IssueShares", vec![percent.to_string()])
            } else {
                k("SellShares", vec![corporation_sym.clone(), percent.to_string()])
            }
        }
        A::PayoffPlayerDebt { .. } => k("PayoffPlayerDebt", vec![]),
        A::PayoffPlayerDebtPartial { .. } => k("PayoffPlayerDebtPartial", vec![]),
        A::ChooseAbility { entity_id, choice } => {
            if let Some((target, src)) = choice.rsplit_once('_').filter(|(_, s)| matches!(*s, "ipo" | "market")) {
                if entity_id != f.company_osth {
                    return Err(err("a share choice of a company that is not the tax haven"));
                }
                k(
                    "ChooseAbility.tax_haven",
                    vec![entity_id.clone(), target.to_string(), format!("source:{src}")],
                )
            } else if let Some((target, end)) = choice.rsplit_once('_').filter(|(_, s)| matches!(*s, "top" | "bottom"))
            {
                k(
                    "ChooseAbility.stack",
                    vec![entity_id.clone(), target.to_string(), format!("stack:{end}")],
                )
            } else {
                let c = match choice.as_str() {
                    "exchange" => "ability:exchange",
                    "available" => "ability:available",
                    "close_p16" => "ability:close",
                    _ => return Err(err("unknown ability choice")),
                };
                k("ChooseAbility", vec![entity_id.clone(), c.to_string()])
            }
        }
        A::Choose { entity_id, choice } => match step {
            Some(Step::Choose) if choice == "double" => k(
                "Choose.double",
                vec![f.company_double_cash.to_string(), "choose:double".into()],
            ),
            Some(Step::BuySellParShares) => k("Choose.convert", vec!["convert:shares".into(), choice.clone()]),
            Some(Step::SpecialChoose) => {
                let co = g
                    .g1822_special_choose_company(entity_id)
                    .ok_or_else(|| err("no special-choose company"))?;
                let c = match choice.as_str() {
                    "token" => "ability:token",
                    "discount" => "ability:discount",
                    _ => return Err(err("unknown special choice")),
                };
                k("Choose.egr", vec![co, c.to_string()])
            }
            Some(Step::Route) => {
                if choice == crate::g1822::ca_route::CHOICE_SKIP {
                    return Ok(k("Choose.route_skip", vec!["route:skip".into()]));
                }
                let i: usize = choice
                    .parse()
                    .map_err(|_| err("a route choice that is not a position"))?;
                let grain = g.g1822_is_ca()
                    && g.g1822_ca_choosing(entity_id) == Some(crate::g1822::ca_route::CaChoosing::Grain);
                let list = if grain {
                    g.g1822_ca_grain_choices(entity_id)
                } else {
                    g.g1822_pullman_choices(entity_id)
                };
                let (name, rank) = train_rank(g, &list, i).ok_or_else(|| err("a route choice off the list"))?;
                if grain {
                    k("Choose.grain", vec!["route:grain".into(), name, rank.to_string()])
                } else {
                    k("Choose.pullman", vec!["route:pullman".into(), name, rank.to_string()])
                }
            }
            Some(Step::Dividend) => {
                let tid = g
                    .g1822_find_extra_train(entity_id)
                    .ok_or_else(|| err("no extra train"))?;
                let name = train_name(g, &tid).ok_or_else(|| err("unknown extra train"))?;
                k("Dividend.extra", vec![name, format!("dividend:{choice}")])
            }
            Some(Step::MinorAcquisition) => k(
                "Choose.acquire",
                vec![acquire_choice(choice).ok_or_else(|| err("acquisition choice"))?],
            ),
            Some(Step::Merge) => match g.rnd().merge_state {
                MergeState::SelectingPar => k("Choose.merge_par", vec!["merge:par".into(), choice.clone()]),
                MergeState::SelectingShares => k("Choose.merge_shares", vec!["merge:shares".into(), choice.clone()]),
                _ => k(
                    "Choose.acquire",
                    vec![acquire_choice(choice).ok_or_else(|| err("merger token choice"))?],
                ),
            },
            Some(Step::AcquireCompany) => k(
                "Choose.replace_minor",
                vec!["choose:replace_minor".into(), choice.clone()],
            ),
            Some(Step::AssignSawmill) => k(
                "Choose.sawmill",
                vec![
                    crate::g1822::ca::COMPANY_SAWMILL.to_string(),
                    format!("sawmill:{choice}"),
                ],
            ),
            _ => return Err(err("a choice of no known step")),
        },
        A::LayTile {
            entity_id,
            hex_id,
            tile_id,
            rotation,
        } => {
            let place = vec![hex_id.clone(), tile_name(tile_id).to_string(), rotation.to_string()];
            if is_company(entity_id) {
                let mut e = vec![entity_id.clone()];
                let block = match m.combo.as_slice() {
                    [] => format!("LayTile:{entity_id}"),
                    [p] => {
                        e.push(p.clone());
                        format!("LayTile:{entity_id}+{p}")
                    }
                    _ => return Err(err("a lay combining more than two companies")),
                };
                e.extend(place);
                (block, e)
            } else {
                k("LayTile", place)
            }
        }
        A::PlaceToken {
            entity_id,
            hex_id,
            city_index,
        } => {
            let slot = format!("{hex_id}.{city_index}");
            if is_company(entity_id) {
                (format!("PlaceToken:{entity_id}"), vec![entity_id.clone(), slot])
            } else if g.g1822_is_ca() && step == Some(Step::DestinationToken) {
                k("PlaceToken.destination", vec![slot, "token:destination".into()])
            } else {
                k("PlaceToken", vec![slot])
            }
        }
        A::RemoveToken { hex_id, city_index, .. } => k("RemoveToken", vec![format!("{hex_id}.{city_index}")]),
        A::HexToken { hex_id, .. } => k("HexToken", vec![hex_id.clone(), "token:destination".into()]),
        A::BuyTrain {
            train_name: id,
            from,
            variant,
            exchange,
            ..
        } => {
            let name = variant
                .as_deref()
                .map(|v| train_type(g, v))
                .or_else(|| train_name(g, id))
                .ok_or_else(|| err("unknown train"))?;
            if from != "depot" {
                k("BuyTrain.corp", vec![from.clone(), name])
            } else if let Some(x) = exchange {
                let traded = train_name(g, x).ok_or_else(|| err("unknown traded train"))?;
                k("BuyTrain.exchange", vec![name, traded])
            } else if step != Some(Step::FirstTurnHousekeeping) && g.depot.discarded.iter().any(|t| t.id == *id) {
                k("BuyTrain.pool", vec![name])
            } else {
                k("BuyTrain.depot", vec![name])
            }
        }
        A::DiscardTrain { train_name: id, .. } => k(
            "DiscardTrain",
            vec![train_name(g, id).ok_or_else(|| err("unknown train"))?],
        ),
        A::Dividend { kind, .. } => {
            let d = match kind {
                DividendKind::Payout => DIVIDENDS[0],
                DividendKind::Half => DIVIDENDS[1],
                DividendKind::Withhold => DIVIDENDS[2],
            };
            k("Dividend", vec![d.to_string()])
        }
        A::RunRoutes { .. } => k("RunRoutes", vec![]),
        A::Merge { corporation_sym, .. } => k("Merge", vec![corporation_sym.clone()]),
        A::AcquireCompany { company_sym, .. } => k("AcquireCompany", vec![company_sym.clone()]),
        A::Assign {
            entity_id,
            target,
            target_type,
        } => {
            if target_type != "hex" {
                return Err(err("an assignment to something other than a hex"));
            }
            k("Assign", vec![entity_id.clone(), target.clone()])
        }
        _ => return Err(err("no 1822-family block for this action type")),
    })
}

/// A minor acquisition's / merger's pay or token choice.
fn acquire_choice(choice: &str) -> Option<String> {
    Some(
        match choice {
            "money" => "merge_pay:money",
            "one_share" => "merge_pay:one_share",
            "two_shares" => "merge_pay:two_shares",
            "replace" => "merge_token:replace",
            "exchange" => "merge_token:exchange",
            "remove" => "merge_token:remove",
            _ => return None,
        }
        .to_string(),
    )
}

/// The index of a generator move in the title's layout.
pub fn move_index(lo: &RlLayout, g: &BaseGame, m: &Move) -> Result<u32, GameError> {
    let (block, entries) = move_key(g, m)?;
    let refs: Vec<&str> = entries.iter().map(String::as_str).collect();
    lo.index_of(&block, &refs).ok_or_else(|| {
        GameError::new(format!(
            "{}: {block} {entries:?} is not in the layout ({:?}, combo {:?})",
            lo.version, m.action, m.combo
        ))
    })
}

#[cfg(test)]
mod tests;
