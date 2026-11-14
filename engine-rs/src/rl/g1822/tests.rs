//! The 1822 family's layouts from title data, and the index function
//! against the family's legal-move generator (design §2.8): every generated
//! move has an index, a state's indices are distinct, each index's
//! descriptor says what its move does, decode(index) gives back the move,
//! and applying by index leaves the same state as applying the move.

use std::collections::{BTreeMap, HashSet};

use super::super::descriptor::{Descriptor, EntKind, Family, PriceKind, CHOICES, NONE};
use super::super::generator::{priced, IndexedMove};
use super::super::rl_layout;
use super::super::RlLayout;
use super::{city_slots, move_key, tile_name};
use crate::actions::{Action, DividendKind};
use crate::g1822::fuzz::{new_game_titled, ruby_dead_end, Rng};
use crate::g1822::legal::Move;
use crate::game::BaseGame;
use crate::title::CrossPlayerPolicy;

const TITLES: [&str; 4] = ["1822", "1822PNW", "1822MX", "1822CA"];

fn lo(title: &str) -> &'static RlLayout {
    rl_layout(title).unwrap_or_else(|e| panic!("{title} is bridge-ready: {e}"))
}

/// Every title's blocks: contiguous, every descriptor distinct, nothing
/// dead, price kinds exactly on the price-bearing blocks.
#[test]
fn layout_blocks() {
    for title in TITLES {
        let lo = lo(title);
        assert_eq!(lo.version, format!("{title}-v1"));
        let mut offset = 0;
        for b in &lo.blocks {
            assert_eq!(b.offset, offset, "{title} {}", b.spec.name);
            offset += b.size;
        }
        assert_eq!(lo.total, offset);
        assert_eq!(lo.descriptors.len(), lo.total as usize);
        let rows: HashSet<[i32; 14]> = lo.descriptors.iter().map(|d| d.to_row()).collect();
        assert_eq!(rows.len(), lo.total as usize, "{title}: descriptors are unique");
        assert!(lo.descriptors.iter().all(|d| !d.is_dead()));
        for b in &lo.blocks {
            let priced = matches!(
                b.spec.name.as_str(),
                "Bid.company" | "Bid.minor" | "Bid.token" | "BuyTrain.corp" | "PayoffPlayerDebtPartial"
            );
            assert_eq!(b.spec.price.is_some(), priced, "{title} {}", b.spec.name);
        }
        let lays: u32 = lo
            .blocks
            .iter()
            .filter(|b| b.spec.family == Family::LayTile)
            .map(|b| b.size)
            .sum();
        eprintln!(
            "{title} {}: {} indices ({} lays, {:.1}%), {} blocks, fingerprint {}",
            lo.version,
            lo.total,
            lays,
            100.0 * lays as f64 / lo.total as f64,
            lo.blocks.len(),
            lo.fingerprint_hex()
        );
        for b in &lo.blocks {
            eprintln!(
                "  {:28} {:22} {:>8} {:>7}",
                b.spec.name,
                b.spec.family.name(),
                b.offset,
                b.size
            );
        }
        let slots = city_slots(crate::title::resolve(title));
        eprintln!(
            "  city slots: {} ({} per hex)",
            slots.len(),
            slots.len() / lo.vocab.hexes.len()
        );
    }
}

/// `index_of` against the descriptors, by hand.
#[test]
fn index_of_matches_the_descriptors() {
    let lo = lo("1822");
    let v = &lo.vocab;
    let i = lo
        .index_of("LayTile:P12+P10", &["P12", "P10", "I10", "57", "3"])
        .unwrap();
    let d = lo.descriptors[i as usize];
    assert_eq!(
        (
            v.companies[d.via as usize],
            d.ent0_kind,
            v.companies[d.ent0 as usize],
            v.hexes[d.hex as usize],
            v.tiles[d.tile as usize],
            d.rot
        ),
        ("P12", EntKind::Company.id(), "P10", "I10", "57", 3)
    );
    let i = lo.index_of("Choose.pullman", &["route:pullman", "3", "1"]).unwrap();
    let d = lo.descriptors[i as usize];
    assert_eq!(
        (CHOICES[d.choice as usize], v.trains[d.ent0 as usize], d.param),
        ("route:pullman", "3", 1)
    );
    let i = lo
        .index_of("ChooseAbility.tax_haven", &["P16", "LNWR", "source:market"])
        .unwrap();
    let d = lo.descriptors[i as usize];
    assert_eq!(
        (
            v.companies[d.via as usize],
            v.corporations[d.ent0 as usize],
            CHOICES[d.choice as usize]
        ),
        ("P16", "LNWR", "source:market")
    );
    let i = lo.index_of("Dividend.extra", &["2P", "dividend:half"]).unwrap();
    let d = lo.descriptors[i as usize];
    assert_eq!((d.family, v.trains[d.ent0 as usize]), (Family::Dividend.id(), "2P"));
    assert!(
        lo.index_of("LayTile:P5", &["P5", "I10", "57", "0"]).is_none(),
        "P5 only teleports to P43"
    );
    let mx = super::super::rl_layout("1822MX").unwrap();
    let i = mx.index_of("Bid.token", &["N23.0"]).unwrap();
    let d = mx.descriptors[i as usize];
    assert_eq!(
        (d.family, d.city, d.price_kind),
        (Family::BidToken.id(), 0, PriceKind::Bid.id())
    );
    assert!(mx.index_of("LayTile:P10", &["P10", "N23", "BC", "0"]).is_some());
    assert!(
        mx.index_of("LayTile:P10", &["P10", "N23", "57", "0"]).is_none(),
        "P10 lays builder cubes only"
    );
}

/// The 1830 factored path refuses the family; the generator API takes it.
#[test]
fn the_index_paths_stay_apart() {
    for title in TITLES {
        let mut g = new_game_titled(title, 4, 1, &[]);
        let msg = |e: crate::actions::GameError| e.message;
        assert!(msg(g.factored_layout().err().unwrap()).contains("generator"), "{title}");
        assert!(g.rl_generator_layout().is_ok());
        assert!(!g
            .rl_g1822_indexed_moves(CrossPlayerPolicy::RESTRICTED)
            .unwrap()
            .is_empty());
        assert!(
            g.rl_indexed_moves(CrossPlayerPolicy::RESTRICTED).is_err(),
            "the 1867 move type refuses {title}"
        );
    }
}

// ---------------------------------------------------------------------------
// The index function against the generator
// ---------------------------------------------------------------------------

/// A train's type: its name, but an 1822CA grain name left on it
/// (`super::train_type`) names its card's type.
fn train_name(g: &BaseGame, id: &str) -> Option<String> {
    g.corporations
        .iter()
        .flat_map(|c| c.trains.iter())
        .chain(g.depot.trains.iter())
        .chain(g.depot.discarded.iter())
        .find(|t| t.id == id)
        .map(|t| super::train_type(g, &t.name))
}

/// Whether descriptor `d` (of the index `m` maps to) says what `m` does.
fn agrees(lo: &RlLayout, d: &Descriptor, g: &BaseGame, m: &Move) -> Result<(), String> {
    let v = &lo.vocab;
    let ent = |kind: i32, i: i32| -> Option<&str> {
        let list = match kind {
            k if k == EntKind::Corporation.id() => &v.corporations,
            k if k == EntKind::Company.id() => &v.companies,
            k if k == EntKind::Train.id() => &v.trains,
            _ => return None,
        };
        (i >= 0).then(|| list[i as usize])
    };
    let e0 = ent(d.ent0_kind, d.ent0);
    let e1 = ent(d.ent1_kind, d.ent1);
    let via = (d.via >= 0).then(|| v.companies[d.via as usize]);
    let hex = (d.hex >= 0).then(|| v.hexes[d.hex as usize]);
    let tile = (d.tile >= 0).then(|| v.tiles[d.tile as usize]);
    let choice = (d.choice >= 0).then(|| CHOICES[d.choice as usize]);
    let check = |ok: bool, what: &str| if ok { Ok(()) } else { Err(what.to_string()) };
    check(
        (d.price_kind != NONE) == m.price.is_some(),
        "price kind iff the move has a range",
    )?;
    let is_company = |id: &str| g.co_idx(id).is_some() && !g.corp_idx.contains_key(id);
    use Action as A;
    match (d.family_enum(), &m.action) {
        (Family::Pass, A::Pass { .. }) => Ok(()),
        (Family::BidCompany, A::Bid { company_sym, .. }) => {
            // a minor's company ("M<minor>") bids on range-relative cells
            let minor = company_sym.starts_with('M')
                && g.corp_idx.get(&company_sym[1..]).map_or(false, |&ci| {
                    g.corporations[ci].corp_type == crate::title::CorpType::Minor
                });
            let kind = if minor { PriceKind::BidRange } else { PriceKind::Bid };
            check(e0 == Some(company_sym.as_str()) && d.price_kind == kind.id(), "bid company")
        }
        (Family::BidToken, A::Bid { company_sym, .. }) => {
            let a = g.rnd().ndem_auction.as_ref().ok_or("no auction")?;
            let t = &a.available[a.up.ok_or("nothing up")?];
            check(
                company_sym.is_empty() && hex == Some(t.hex.as_str()) && d.city == t.city as i32,
                "bid token",
            )
        }
        (
            Family::Par,
            A::Par {
                corporation_sym,
                share_price,
                ..
            },
        ) => check(e0 == Some(corporation_sym.as_str()) && d.param == *share_price, "par"),
        (
            Family::BuyShares,
            A::BuyShares {
                entity_id,
                corporation_sym,
                source,
                share_indices,
                ..
            },
        ) => {
            let president = share_indices.iter().any(|&i| {
                g.corporations[g.corp_idx[corporation_sym.as_str()]]
                    .shares
                    .get(i)
                    .map_or(false, |s| s.president)
            });
            check(
                entity_id != corporation_sym
                    && e0 == Some(corporation_sym.as_str())
                    && choice == Some(if source == "ipo" { "source:ipo" } else { "source:market" })
                    && (d.flags & super::super::descriptor::flags::PRESIDENT_CERT != 0) == president,
                "buy shares",
            )
        }
        (
            Family::RedeemShares,
            A::BuyShares {
                entity_id,
                corporation_sym,
                percent,
                ..
            },
        ) => check(entity_id == corporation_sym && d.param == *percent as i32, "redeem"),
        (
            Family::SellShares,
            A::SellShares {
                entity_id,
                corporation_sym,
                percent,
                ..
            },
        ) => check(
            entity_id != corporation_sym && e0 == Some(corporation_sym.as_str()) && d.param == *percent as i32,
            "sell",
        ),
        (
            Family::IssueShares,
            A::SellShares {
                entity_id,
                corporation_sym,
                percent,
                ..
            },
        ) => check(entity_id == corporation_sym && d.param == *percent as i32, "issue"),
        (Family::PayoffPlayerDebt, A::PayoffPlayerDebt { .. }) => Ok(()),
        (Family::PayoffPlayerDebtPartial, A::PayoffPlayerDebtPartial { .. }) => {
            check(d.price_kind == PriceKind::Amount.id(), "payoff amount")
        }
        (Family::ChooseAbility, A::ChooseAbility { entity_id, choice: c }) => {
            check(via == Some(entity_id.as_str()), "the company choosing is via")?;
            let want = if let Some((t, s)) = c.rsplit_once('_').filter(|(_, s)| matches!(*s, "ipo" | "market")) {
                (Some(t.to_string()), format!("source:{s}"))
            } else if let Some((t, s)) = c.rsplit_once('_').filter(|(_, s)| matches!(*s, "top" | "bottom")) {
                (Some(t.to_string()), format!("stack:{s}"))
            } else {
                let x = if c == "close_p16" { "close" } else { c.as_str() };
                (None, format!("ability:{x}"))
            };
            check(
                e0.map(str::to_string) == want.0 && choice == Some(want.1.as_str()),
                "ability choice",
            )
        }
        (Family::Choose, A::Choose { entity_id, choice: c }) => {
            let ch = choice.ok_or("a choose without a choice")?;
            match ch {
                "route:pullman" | "route:grain" => {
                    let i: usize = c.parse().map_err(|_| "a route choice that is no position")?;
                    let list = if ch == "route:grain" {
                        g.g1822_ca_grain_choices(entity_id)
                    } else {
                        g.g1822_pullman_choices(entity_id)
                    };
                    // (the type before an attachment renamed the train for the run)
                    let name = super::route_train_type(g, &list[i]);
                    let rank = list[..i].iter().filter(|id| super::route_train_type(g, id) == name).count();
                    check(e0 == name.as_deref() && d.param == rank as i32, "route attachment")
                }
                "route:skip" => check(c == "None", "skip"),
                "convert:shares" | "merge:par" | "merge:shares" => {
                    check(c.parse::<i32>().ok() == Some(d.param), "a number choice")
                }
                "choose:replace_minor" => check(e0 == Some(c.as_str()), "P20's minor"),
                _ => {
                    let tail = ch.rsplit_once(':').map(|(_, t)| t).unwrap_or(ch);
                    check(tail == c.as_str(), "named choice")
                }
            }
        }
        (Family::Dividend, A::Choose { entity_id, choice: c }) => {
            let extra = g.g1822_find_extra_train(entity_id).and_then(|id| train_name(g, &id));
            check(
                e0 == extra.as_deref() && choice == Some(format!("dividend:{c}").as_str()),
                "extra dividend",
            )
        }
        (Family::Dividend, A::Dividend { kind, .. }) => check(
            e0.is_none()
                && choice
                    == Some(match kind {
                        DividendKind::Payout => "dividend:payout",
                        DividendKind::Half => "dividend:half",
                        DividendKind::Withhold => "dividend:withhold",
                    }),
            "dividend",
        ),
        (
            Family::LayTile,
            A::LayTile {
                entity_id,
                hex_id,
                tile_id,
                rotation,
            },
        ) => {
            let by = if is_company(entity_id) {
                Some(entity_id.as_str())
            } else {
                None
            };
            check(
                via == by
                    && e0 == m.combo.first().map(String::as_str)
                    && hex == Some(hex_id.as_str())
                    && tile == Some(tile_name(tile_id))
                    && d.rot == *rotation as i32,
                "lay",
            )
        }
        (
            Family::PlaceToken,
            A::PlaceToken {
                entity_id,
                hex_id,
                city_index,
            },
        ) => {
            let by = if is_company(entity_id) {
                Some(entity_id.as_str())
            } else {
                None
            };
            check(
                via == by && hex == Some(hex_id.as_str()) && d.city == *city_index as i32,
                "token city",
            )
        }
        (Family::RemoveToken, A::RemoveToken { hex_id, city_index, .. }) => check(
            hex == Some(hex_id.as_str()) && d.city == *city_index as i32,
            "removed token",
        ),
        (Family::HexToken, A::HexToken { hex_id, .. }) => check(
            hex == Some(hex_id.as_str()) && choice == Some("token:destination"),
            "hex token",
        ),
        (
            Family::BuyTrainDepot | Family::BuyTrainPool,
            A::BuyTrain {
                train_name: id,
                from,
                variant,
                exchange: None,
                ..
            },
        ) => {
            let pool = g.depot.discarded.iter().any(|t| t.id == *id);
            let fam_ok = (d.family_enum() == Family::BuyTrainPool) == pool
                || g.g1822_active_step().map(|i| g.rnd().steps[i]) == Some(crate::g1822::Step::FirstTurnHousekeeping);
            let ty = variant.as_deref().map(|v| super::train_type(g, v));
            check(from == "depot" && fam_ok && e0 == ty.as_deref(), "depot / pool train")
        }
        (
            Family::BuyTrainExchange,
            A::BuyTrain {
                variant,
                exchange: Some(x),
                ..
            },
        ) => check(
            e0 == variant.as_deref() && e1 == train_name(g, x).as_deref(),
            "trade-in",
        ),
        (
            Family::BuyTrainCorp,
            A::BuyTrain {
                train_name: id, from, ..
            },
        ) => check(
            e0 == Some(from.as_str()) && e1 == train_name(g, id).as_deref(),
            "corp train",
        ),
        (Family::DiscardTrain, A::DiscardTrain { train_name: id, .. }) => {
            check(e0 == train_name(g, id).as_deref(), "discard")
        }
        (Family::RunRoutes, A::RunRoutes { .. }) => Ok(()),
        (Family::Merge, A::Merge { corporation_sym, .. }) => check(e0 == Some(corporation_sym.as_str()), "merge"),
        (Family::AcquireCompany, A::AcquireCompany { company_sym, .. }) => {
            check(e0 == Some(company_sym.as_str()), "acquire company")
        }
        (Family::Assign, A::Assign { entity_id, target, .. }) => check(
            via == Some(entity_id.as_str()) && hex == Some(target.as_str()),
            "assign",
        ),
        (f, a) => Err(format!("family {f:?} for {a:?}")),
    }
}

/// Sort the arrays a hash set serializes to (the graph memos' node, hex
/// and path sets), so iteration order does not count.
fn canon(v: &mut serde_json::Value) {
    match v {
        serde_json::Value::Object(m) => {
            for (k, x) in m.iter_mut() {
                if matches!(k.as_str(), "connected_nodes" | "reachable_hexes" | "connected_paths") {
                    if let serde_json::Value::Array(a) = x {
                        a.sort_by_key(|e| e.to_string());
                    }
                }
                canon(x);
            }
        }
        serde_json::Value::Array(a) => a.iter_mut().for_each(canon),
        _ => {}
    }
}

/// The state, for "apply by index == apply by move" (serde_json values:
/// maps with sorted keys, so hash-map order does not count; hash sets
/// sorted by [`canon`]).
fn digest(g: &BaseGame) -> String {
    let tiles: Vec<&crate::graph::Tile> = g.hexes.iter().map(|h| &h.tile).collect();
    let mut v = serde_json::json!({
        "players": g.players,
        "corporations": g.corporations,
        "companies": g.companies,
        "tiles": tiles,
        "depot": g.depot,
        "bank": g.bank,
        "phase": g.phase.name,
        "finished": g.finished,
        "g1822": g.g1822,
    });
    canon(&mut v);
    v.to_string()
}

#[derive(Default, Debug)]
pub(crate) struct Stats {
    walks: usize,
    finished: usize,
    ruby_dead_ends: usize,
    states: usize,
    moves_indexed: usize,
    max_legal: usize,
    decoded: usize,
    prices_checked: usize,
    ruby_dead_end_prices: usize,
    by_block: BTreeMap<String, usize>,
    played_by_block: BTreeMap<String, usize>,
}

impl Stats {
    fn merge(&mut self, o: Stats) {
        self.walks += o.walks;
        self.finished += o.finished;
        self.ruby_dead_ends += o.ruby_dead_ends;
        self.states += o.states;
        self.moves_indexed += o.moves_indexed;
        self.max_legal = self.max_legal.max(o.max_legal);
        self.decoded += o.decoded;
        self.prices_checked += o.prices_checked;
        self.ruby_dead_end_prices += o.ruby_dead_end_prices;
        for (k, v) in o.by_block {
            *self.by_block.entry(k).or_default() += v;
        }
        for (k, v) in o.played_by_block {
            *self.played_by_block.entry(k).or_default() += v;
        }
    }
}

/// The block family name a block's statistics are kept under ("LayTile:P12"
/// -> "LayTile:<company>").
fn block_key(name: &str) -> String {
    match name.split_once(':') {
        Some((b, rest)) if rest.contains('+') => format!("{b}:<company>+<company>"),
        Some((b, _)) if b == "LayTile" || b == "PlaceToken" => format!("{b}:<company>"),
        _ => name.to_string(),
    }
}

/// A random price of a ranged move (the ends favoured).
fn random_price(m: &Move, rng: &mut Rng) -> Option<i32> {
    let r = m.price?;
    Some(crate::g1822::fuzz::pick_price(&r, rng))
}

/// One walk through the index API: at every decision, the state's indexed
/// moves (every move indexed, indices distinct), each descriptor against
/// its move, both ends of every price range applied on clones, decode of
/// the chosen move through the API, and apply-by-index against
/// apply-by-move. The walk advances by index.
pub(crate) fn index_walk(
    title: &str,
    players: usize,
    seed: u64,
    rules: &[&str],
    max_moves: usize,
    policy: CrossPlayerPolicy,
) -> Result<Stats, String> {
    let lo = lo(title);
    let mut g = new_game_titled(title, players, seed, rules);
    let mut rng = Rng::new(seed ^ 0x5A5A_1822);
    let mut st = Stats {
        walks: 1,
        ..Stats::default()
    };
    let at = |n: usize| format!("{title} {players}p seed {seed} {rules:?} {policy:?} move {n}");
    for n in 0..max_moves {
        if g.finished {
            break;
        }
        let indexed: Vec<IndexedMove<Move>> = g
            .rl_g1822_indexed_moves(policy)
            .map_err(|e| format!("{}: {e}", at(n)))?;
        if indexed.is_empty() {
            if ruby_dead_end(&mut g) {
                st.ruby_dead_ends += 1;
                return Ok(st);
            }
            return Err(format!("{}: no legal moves", at(n)));
        }
        st.states += 1;
        st.max_legal = st.max_legal.max(indexed.len());
        for x in &indexed {
            let d = lo.descriptors[x.index as usize];
            agrees(lo, &d, &g, &x.mv)
                .map_err(|e| format!("{}: index {} {:?}: {e}\n{d:?}", at(n), x.index, x.mv.action))?;
            let block = lo.block_of(x.index).unwrap().spec.name.clone();
            *st.by_block.entry(block_key(&block)).or_default() += 1;
            if let Some(r) = x.mv.price {
                for p in [r.lo, r.hi] {
                    let mut c = g.clone_for_search();
                    match c.g1822_apply_move(&x.mv.with_price(p)) {
                        Ok(()) => {}
                        // A Ruby dead end the engine mirrors.
                        Err(e) if e.message.contains("(Ruby raises)") => st.ruby_dead_end_prices += 1,
                        Err(e) => {
                            return Err(format!(
                                "{}: {:?} at price {p} of [{}, {}]: {e}",
                                at(n),
                                x.mv.action,
                                r.lo,
                                r.hi
                            ))
                        }
                    }
                    st.prices_checked += 1;
                }
            }
        }
        st.moves_indexed += indexed.len();
        let non_pass: Vec<&IndexedMove<Move>> = indexed
            .iter()
            .filter(|x| !matches!(x.mv.action, Action::Pass { .. }))
            .collect();
        let x = if !non_pass.is_empty() && rng.below(4) != 0 {
            non_pass[rng.below(non_pass.len())]
        } else {
            &indexed[rng.below(indexed.len())]
        };
        let price = random_price(&x.mv, &mut rng);
        let chosen = priced(x.mv.clone(), price).map_err(|e| e.message)?;
        let decoded = g
            .rl_g1822_move_for_index(x.index, price, policy)
            .map_err(|e| format!("{}: decode: {e}", at(n)))?;
        if format!("{:?}{:?}", decoded.action, decoded.combo) != format!("{:?}{:?}", chosen.action, chosen.combo) {
            return Err(format!(
                "{}: decode({}) = {:?}, not {:?}",
                at(n),
                x.index,
                decoded.action,
                chosen.action
            ));
        }
        st.decoded += 1;
        let (block, _) = move_key(&g, &chosen).map_err(|e| e.message)?;
        *st.played_by_block.entry(block_key(&block)).or_default() += 1;
        let wire = g.g1822_move_wire(&chosen);
        let mut by_move = g.clone_for_search();
        match by_move.g1822_apply_move(&chosen) {
            Ok(()) => {}
            // The chosen price runs into a Ruby dead end the engine mirrors:
            // Ruby stops too.
            Err(e) if e.message.contains("(Ruby raises)") => {
                st.ruby_dead_ends += 1;
                return Ok(st);
            }
            Err(e) => return Err(format!("{}: apply {:?}: {e}", at(n), chosen.action)),
        }
        let logged = g
            .rl_g1822_apply_index(x.index, price, policy)
            .map_err(|e| format!("{}: apply index: {e}", at(n)))?;
        if logged != wire || g.action_log.last() != Some(&wire) {
            return Err(format!("{}: logged {logged} for {wire}", at(n)));
        }
        let (a, b) = (digest(&g), digest(&by_move));
        if a != b {
            let i = a
                .bytes()
                .zip(b.bytes())
                .position(|(x, y)| x != y)
                .unwrap_or(a.len().min(b.len()));
            let ctx = |s: &str| s[i.saturating_sub(200)..(i + 200).min(s.len())].to_string();
            return Err(format!(
                "{}: apply by index != apply by move for {:?}\n  by index: ...{}...\n  by move:  ...{}...",
                at(n),
                chosen.action,
                ctx(&a),
                ctx(&b)
            ));
        }
    }
    st.finished = g.finished as usize;
    Ok(st)
}

type Walk = (&'static str, usize, u64, &'static [&'static str], CrossPlayerPolicy);

/// Walks in parallel threads; all must pass.
pub(crate) fn run(walks: Vec<Walk>, max_moves: usize) -> Stats {
    let results: Vec<Result<Stats, String>> = std::thread::scope(|s| {
        let handles: Vec<_> = walks
            .iter()
            .map(|&(t, p, seed, rules, pol)| s.spawn(move || index_walk(t, p, seed, rules, max_moves, pol)))
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().unwrap_or_else(|_| Err("PANIC".into())))
            .collect()
    });
    let mut total = Stats::default();
    let mut failures = Vec::new();
    for r in results {
        match r {
            Ok(s) => total.merge(s),
            Err(e) => failures.push(e),
        }
    }
    assert!(
        failures.is_empty(),
        "{} failures:\n{}",
        failures.len(),
        failures.join("\n")
    );
    total
}

/// The player counts the pipeline trains each title at.
fn players(title: &str) -> std::ops::RangeInclusive<usize> {
    match title {
        "1822PNW" | "1822MX" => 3..=5,
        _ => 3..=6,
    }
}

/// A title's optional rules for walk `k` (1822+ / MX's higher cert limit on
/// every other seed).
fn rules(title: &str, k: u64) -> &'static [&'static str] {
    match (title, k % 2) {
        ("1822", 0) => &["plus_expansion"],
        ("1822MX", 0) => &["higher_cert_limit"],
        _ => &[],
    }
}

fn walks(title: &'static str, seeds: u64) -> Vec<Walk> {
    let mut w = Vec::new();
    for p in players(title) {
        for k in 1..=seeds {
            w.push((
                title,
                p,
                k * 104_729 + p as u64,
                rules(title, k),
                CrossPlayerPolicy::RESTRICTED,
            ));
            w.push((
                title,
                p,
                7_000 + k * 104_729 + p as u64,
                rules(title, k + 1),
                CrossPlayerPolicy::RULES,
            ));
        }
    }
    w
}

#[test]
fn index_fuzz_1822() {
    let st = run(walks("1822", 1), 3000);
    eprintln!("1822 index fuzz: {st:#?}");
    for block in [
        "Pass",
        "Bid.company",
        "Par",
        "BuyShares",
        "SellShares",
        "PlaceToken",
        "BuyTrain.depot",
        "Dividend",
        "RunRoutes",
        "LayTile",
    ] {
        assert!(
            st.by_block.get(block).copied().unwrap_or(0) > 0,
            "never offered: {block}: {:?}",
            st.by_block
        );
    }
}

#[test]
fn index_fuzz_1822pnw() {
    let st = run(walks("1822PNW", 1), 3000);
    eprintln!("1822PNW index fuzz: {st:#?}");
}

#[test]
fn index_fuzz_1822mx() {
    let st = run(walks("1822MX", 1), 3000);
    eprintln!("1822MX index fuzz: {st:#?}");
}

#[test]
fn index_fuzz_1822ca() {
    let st = run(walks("1822CA", 1), 3000);
    eprintln!("1822CA index fuzz: {st:#?}");
}

/// The verification run behind `docs/rl_descriptors.md`'s numbers:
/// `G1822_INDEX_TITLE=1822 G1822_INDEX_SEEDS=5 cargo test --release
/// --no-default-features rl::g1822::tests::index_fuzz_many -- --ignored
/// --nocapture`.
#[test]
#[ignore]
fn index_fuzz_many() {
    let seeds: u64 = std::env::var("G1822_INDEX_SEEDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);
    let only = std::env::var("G1822_INDEX_TITLE").ok();
    for title in TITLES {
        if only.as_deref().map_or(false, |t| t != title) {
            continue;
        }
        let st = run(walks(title, seeds), 6000);
        eprintln!("{title} index fuzz ({seeds} seeds x 2 policies per player count): {st:#?}");
    }
}
