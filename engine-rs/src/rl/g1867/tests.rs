//! `1867-v1`: the layout from title data, and the index function against
//! the legal-move generator (design §2.8): every generated move has an
//! index, a state's indices are distinct, each index's descriptor says what
//! its move does, decode(index) gives back the move, and applying by index
//! leaves the same state as applying the move.

use std::collections::{BTreeMap, HashSet};

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use super::super::descriptor::{Descriptor, EntKind, Family, NONE};
use super::super::rl_layout;
use super::super::RlLayout;
use super::{city_slots, move_key, tile_name};
use crate::actions::{Action, DividendKind};
use crate::g1867::fuzz::new_game;
use crate::g1867::legal::Move;
use crate::game::BaseGame;
use crate::title::CrossPlayerPolicy;

fn lo() -> &'static RlLayout {
    rl_layout("1867").expect("1867 is bridge-ready")
}

/// The blocks of `1867-v1`: (name, family, offset, size).
#[test]
fn layout_blocks() {
    let lo = lo();
    assert_eq!(lo.version, "1867-v1");
    let got: Vec<(&str, &str, u32, u32)> = lo
        .blocks
        .iter()
        .map(|b| (b.spec.name.as_str(), b.spec.family.name(), b.offset, b.size))
        .collect();
    eprintln!("1867-v1 blocks: {got:?} total {}", lo.total);
    let sizes: Vec<(&str, u32)> = got.iter().map(|&(n, _, _, s)| (n, s)).collect();
    let cities = city_slots(crate::title::resolve("1867")).len() as u32;
    assert_eq!(
        sizes,
        vec![
            ("Pass", 1),
            ("Bid.company", 5),
            ("Bid.corporation", 16),
            ("Par", 8 * 11),
            ("BuyShares", 8 * 2),
            ("SellShares", 8 * 5),
            ("RedeemShares", 5),
            ("PlaceToken", cities),
            ("RemoveToken", cities),
            ("BuyTrain.depot", 9),
            ("BuyTrain.pool", 9),
            ("BuyTrain.exchange", 3 * 6 * 2),
            ("BuyTrain.corp", 24 * 9),
            ("DiscardTrain", 9),
            ("Dividend", 3),
            ("BuyCompany", 5),
            ("RunRoutes", 1),
            ("Merge", 24),
            ("Convert", 1),
            ("Choose", 1),
            ("LayTile", 94 * 68 * 6),
        ]
    );
    assert_eq!(lo.total, sizes.iter().map(|(_, s)| s).sum::<u32>());
    assert_eq!(lo.descriptors.len(), lo.total as usize);
    // Every index means something different, and nothing is dead.
    let rows: HashSet<[i32; 14]> = lo.descriptors.iter().map(|d| d.to_row()).collect();
    assert_eq!(rows.len(), lo.total as usize, "descriptors are unique");
    assert!(lo.descriptors.iter().all(|d| !d.is_dead()));
    // Price kinds exactly on the four price-bearing blocks.
    for b in &lo.blocks {
        let priced = matches!(b.spec.name.as_str(), "Bid.company" | "Bid.corporation" | "BuyTrain.corp" | "BuyCompany");
        assert_eq!(b.spec.price.is_some(), priced, "{}", b.spec.name);
    }
    // The axes from title data.
    let entries = |name: &str, axis: usize| lo.block(name).unwrap().spec.axes[axis].entries();
    assert_eq!(entries("Bid.company", 0), vec!["C&SL", "NFB", "MB", "QB", "SCT"]);
    eprintln!("par cells: {:?}", entries("Par", 1));
    assert_eq!(entries("SellShares", 1), vec!["10", "20", "30", "40", "50"]);
    assert_eq!(entries("BuyTrain.exchange", 0), vec!["8", "2+2", "5+5E"]);
    assert_eq!(entries("BuyTrain.exchange", 1), vec!["5", "6", "7", "8", "2+2", "5+5E"]);
    assert_eq!(entries("Dividend", 0), vec!["dividend:payout", "dividend:half", "dividend:withhold"]);
    assert!(!entries("Merge", 0).contains(&"CN".to_string()));
    // The upgrade closure: Montreal (L12) reaches three cities, Ottawa's
    // and Quebec's OO hexes two.
    let slots = city_slots(crate::title::resolve("1867"));
    let per_hex = |h: &str| slots.iter().filter(|(x, _)| *x == h).count();
    eprintln!("city slots: {} over {} hexes", slots.len(), slots.iter().map(|(h, _)| *h).collect::<HashSet<_>>().len());
    assert_eq!(per_hex("L12"), 3, "Montreal's triple city");
}

#[test]
fn index_of_matches_the_descriptors() {
    let lo = lo();
    let v = &lo.vocab;
    let i = lo.index_of("LayTile", &["L12", "X1", "4"]);
    if let Some(i) = i {
        let d = lo.descriptors[i as usize];
        assert_eq!((v.hexes[d.hex as usize], v.tiles[d.tile as usize], d.rot), ("L12", "X1", 4));
    }
    let i = lo.index_of("BuyTrain.exchange", &["2+2", "5", "train_source:pool"]).unwrap();
    let d = lo.descriptors[i as usize];
    assert_eq!(
        (v.trains[d.ent0 as usize], v.trains[d.ent1 as usize], super::super::descriptor::CHOICES[d.choice as usize]),
        ("2+2", "5", "train_source:pool")
    );
    let i = lo.index_of("SellShares", &["CPR", "30"]).unwrap();
    let d = lo.descriptors[i as usize];
    assert_eq!((v.corporations[d.ent0 as usize], d.param), ("CPR", 30));
    assert!(lo.index_of("SellShares", &["CPR", "35"]).is_none());
    assert!(lo.index_of("SellShares", &["CPR"]).is_none());
    assert!(lo.index_of("NoSuchBlock", &[]).is_none());
}

/// Price cells in step units: a minor bid's range ($100..$300 on the $5
/// grid) has 41 legal prices; BidRange's cells cover exactly those, with
/// atoms at the opening bid, the all-in bid and one step below it.
#[test]
fn price_cells_work_in_step_units() {
    use super::super::generator::{price_at, price_cells, price_step};
    use super::super::PriceKind;
    use crate::g1867::legal::PriceRange;
    let r = PriceRange { lo: 100, hi: 300, step: 5 };
    let cells = price_cells(PriceKind::BidRange, r).unwrap();
    assert_eq!(cells.counts.iter().sum::<i64>(), 41, "one cell member per legal price");
    for p in [100, 300, 295] {
        let k = price_step(r, p).unwrap();
        assert_eq!(price_at(r, k), p);
        assert_eq!(cells.counts[cells.cell_of(k).unwrap()], 1, "${p} is an atom");
    }
    assert_eq!(price_step(r, 102), None, "off the $5 grid");
    assert_eq!(price_step(r, 305), None, "above the range");
    // The company auction's "Bid" ladder: each of the first 20 steps is its
    // own cell.
    let bid = price_cells(PriceKind::Bid, PriceRange { lo: 20, hi: 315, step: 5 }).unwrap();
    assert!((0..20).all(|k| bid.counts[bid.cell_of(k).unwrap()] == 1));
    let lo = lo();
    let b = lo.block("Bid.corporation").unwrap();
    assert_eq!(lo.descriptors[b.offset as usize].price_kind, PriceKind::BidRange.id());
}

/// The 1830 factored path refuses 1867 (its moves are indexed by the
/// generator layout); the generator API refuses 1830.
#[test]
fn the_two_index_paths_stay_apart() {
    let mut g = new_game(4);
    let msg = |e: crate::actions::GameError| e.message;
    assert!(msg(g.factored_layout().err().unwrap()).contains("generator"));
    assert!(msg(g.decode_index(0, None).unwrap_err()).contains("generator"));
    assert!(g.rl_generator_layout().is_ok());
    assert!(!g.rl_indexed_moves(CrossPlayerPolicy::RESTRICTED).unwrap().is_empty());
    let ids: Vec<u32> = (1..=4).collect();
    let names = ids.iter().map(|&i| (i, format!("P{i}"))).collect();
    let g30 = BaseGame::build_titled("1830", ids, names);
    assert!(msg(g30.rl_generator_layout().unwrap_err()).contains("factored_legal_indices"));
}

// ---------------------------------------------------------------------------
// The index function against the generator
// ---------------------------------------------------------------------------

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
    let hex = (d.hex >= 0).then(|| v.hexes[d.hex as usize]);
    let tile = (d.tile >= 0).then(|| v.tiles[d.tile as usize]);
    let choice = (d.choice >= 0).then(|| super::super::descriptor::CHOICES[d.choice as usize]);
    let check = |ok: bool, what: &str| if ok { Ok(()) } else { Err(what.to_string()) };
    check((d.price_kind != NONE) == m.price.is_some(), "price kind iff the move has a range")?;
    let train_at = |id: &str| super::train_lookup(g, id);
    use Action as A;
    match (d.family_enum(), &m.action) {
        (Family::Pass, A::Pass { .. }) => Ok(()),
        (Family::BidCompany, A::Bid { company_sym, .. }) => check(e0 == Some(company_sym.as_str()), "bid company"),
        (Family::BidCorporation, A::CorporationBid { corporation_sym, .. }) => {
            check(e0 == Some(corporation_sym.as_str()), "bid minor")
        }
        (Family::Par, A::Par { corporation_sym, share_price, .. }) => {
            check(e0 == Some(corporation_sym.as_str()) && d.param == *share_price, "par")
        }
        (Family::BuyShares, A::BuyShares { entity_id, corporation_sym, source, .. }) => check(
            entity_id != corporation_sym
                && e0 == Some(corporation_sym.as_str())
                && choice == Some(if source == "ipo" { "source:ipo" } else { "source:market" }),
            "buy shares",
        ),
        (Family::RedeemShares, A::BuyShares { entity_id, corporation_sym, percent, .. }) => {
            check(entity_id == corporation_sym && d.param == *percent as i32 && e0.is_none(), "redeem")
        }
        (Family::SellShares, A::SellShares { corporation_sym, percent, .. }) => {
            check(e0 == Some(corporation_sym.as_str()) && d.param == *percent as i32, "sell")
        }
        (Family::PlaceToken, A::PlaceToken { hex_id, city_index, .. })
        | (Family::RemoveToken, A::RemoveToken { hex_id, city_index, .. }) => {
            check(hex == Some(hex_id.as_str()) && d.city == *city_index as i32, "token city")
        }
        (Family::LayTile, A::LayTile { hex_id, tile_id, rotation, .. }) => check(
            hex == Some(hex_id.as_str()) && tile == Some(tile_name(tile_id)) && d.rot == *rotation as i32,
            "lay",
        ),
        (Family::BuyTrainDepot, A::BuyTrain { train_name, from, exchange: None, .. }) => {
            let (name, at) = train_at(train_name);
            check(from == "depot" && at == super::TrainAt::Depot && e0 == name.as_deref(), "depot train")
        }
        (Family::BuyTrainPool, A::BuyTrain { train_name, from, exchange: None, .. }) => {
            let (name, at) = train_at(train_name);
            check(from == "depot" && at == super::TrainAt::Pool && e0 == name.as_deref(), "pool train")
        }
        (Family::BuyTrainExchange, A::BuyTrain { entity_id, train_name, exchange: Some(x), .. }) => {
            let (name, at) = train_at(train_name);
            let traded = g.corporations[g.corp_idx[entity_id.as_str()]].trains.iter().find(|t| t.id == *x).map(|t| t.name.as_str());
            let src = match at {
                super::TrainAt::Depot => Some("train_source:depot"),
                super::TrainAt::Pool => Some("train_source:pool"),
                _ => None,
            };
            check(e0 == name.as_deref() && e1 == traded && choice == src, "trade-in")
        }
        (Family::BuyTrainCorp, A::BuyTrain { train_name, from, .. }) => {
            let (name, at) = train_at(train_name);
            check(at == super::TrainAt::Corp && e0 == Some(from.as_str()) && e1 == name.as_deref(), "corp train")
        }
        (Family::DiscardTrain, A::DiscardTrain { train_name, .. }) => {
            check(e0 == train_at(train_name).0.as_deref(), "discard")
        }
        (Family::Dividend, A::Dividend { kind, .. }) => check(
            choice
                == Some(match kind {
                    DividendKind::Payout => "dividend:payout",
                    DividendKind::Half => "dividend:half",
                    DividendKind::Withhold => "dividend:withhold",
                }),
            "dividend",
        ),
        (Family::BuyCompany, A::BuyCompany { company_sym, .. }) => check(e0 == Some(company_sym.as_str()), "company"),
        (Family::RunRoutes, A::RunRoutes { .. }) | (Family::Convert, A::Convert { .. }) => Ok(()),
        (Family::Merge, A::Merge { corporation_sym, .. }) => check(e0 == Some(corporation_sym.as_str()), "merge"),
        (Family::Choose, A::Choose { choice: c, .. }) => {
            check(choice.map(|x| x.trim_start_matches("choose:")) == Some(c.as_str()), "choose")
        }
        (f, a) => Err(format!("family {f:?} for {a:?}")),
    }
}

/// The state, for "apply by index == apply by move" (serde_json: maps
/// with sorted keys, so hash-map order does not count).
fn digest(g: &BaseGame) -> String {
    let tiles: Vec<&crate::graph::Tile> = g.hexes.iter().map(|h| &h.tile).collect();
    let v = serde_json::json!({
        "players": g.players,
        "corporations": g.corporations,
        "companies": g.companies,
        "tiles": tiles,
        "depot": g.depot,
        "bank": g.bank,
        "round": g.round,
        "phase": g.phase.name,
        "finished": g.finished,
    });
    v.to_string()
}

#[derive(Default, Debug)]
struct Stats {
    walks: usize,
    finished: usize,
    states: usize,
    moves_indexed: usize,
    decoded: usize,
    prices_checked: usize,
    by_block: BTreeMap<String, usize>,
    played_by_block: BTreeMap<String, usize>,
}

impl Stats {
    fn merge(&mut self, o: Stats) {
        self.walks += o.walks;
        self.finished += o.finished;
        self.states += o.states;
        self.moves_indexed += o.moves_indexed;
        self.decoded += o.decoded;
        self.prices_checked += o.prices_checked;
        for (k, v) in o.by_block {
            *self.by_block.entry(k).or_default() += v;
        }
        for (k, v) in o.played_by_block {
            *self.played_by_block.entry(k).or_default() += v;
        }
    }
}

/// A random price of a ranged move (as `g1867::fuzz::pick`).
fn random_price(m: &Move, rng: &mut StdRng) -> Option<i32> {
    let r = m.price?;
    let k = (r.hi - r.lo) / r.step.max(1);
    let x = rng.gen::<f64>();
    Some(if x < 0.4 {
        r.lo
    } else if x < 0.5 {
        r.lo + k * r.step
    } else {
        r.lo + rng.gen_range(0..=k.min(40)) * r.step
    })
}

/// One walk through the index API: at every decision, the state's indexed
/// moves (every move indexed, indices distinct), each descriptor against
/// its move, decode(index) of the chosen move through the API, its lowest
/// and highest prices applied on clones, and apply-by-index against
/// apply-by-move. The walk itself advances by index.
fn index_walk(players: usize, seed: u64, max_moves: usize, policy: CrossPlayerPolicy) -> Result<Stats, String> {
    let lo = lo();
    let mut g = new_game(players);
    let mut rng = StdRng::seed_from_u64(seed * 1_000_003 + players as u64);
    let mut st = Stats { walks: 1, ..Stats::default() };
    let at = |n: usize| format!("{players}p seed {seed} move {n}");
    for n in 0..max_moves {
        if g.finished {
            break;
        }
        let indexed = g.rl_indexed_moves(policy).map_err(|e| format!("{}: {e}", at(n)))?;
        if indexed.is_empty() {
            return Err(format!("{}: no legal moves ({})", at(n), g.g1867_debug_state()));
        }
        st.states += 1;
        for x in &indexed {
            let d = lo.descriptors[x.index as usize];
            agrees(lo, &d, &g, &x.mv).map_err(|e| format!("{}: index {} {:?}: {e}\n{d:?}", at(n), x.index, x.mv.action))?;
            let block = lo.block_of(x.index).unwrap().spec.name.clone();
            if block == "BuyTrain.exchange" {
                let src = super::super::descriptor::CHOICES[d.choice as usize];
                *st.by_block.entry(format!("{block}[{src}]")).or_default() += 1;
            }
            *st.by_block.entry(block).or_default() += 1;
            // Every price on the range applies at its ends (on clones).
            if let Some(r) = x.mv.price {
                for p in [r.lo, r.hi] {
                    let mut c = g.clone_for_search();
                    c.g1867_apply_move(&x.mv.with_price(p))
                        .map_err(|e| format!("{}: {:?} at price {p} of [{}, {}]: {e}", at(n), x.mv.action, r.lo, r.hi))?;
                    st.prices_checked += 1;
                }
            }
        }
        st.moves_indexed += indexed.len();
        // The trade-in ambiguity the source axis resolves: one (new,
        // traded) pair offered from the depot AND from the pool.
        let pairs: Vec<(i32, i32, i32)> = indexed
            .iter()
            .map(|x| lo.descriptors[x.index as usize])
            .filter(|d| d.family == Family::BuyTrainExchange.id())
            .map(|d| (d.ent0, d.ent1, d.choice))
            .collect();
        if pairs.iter().any(|&(a, b, c)| pairs.iter().any(|&(a2, b2, c2)| a == a2 && b == b2 && c != c2)) {
            *st.by_block.entry("(state with a trade-in from depot and pool)".into()).or_default() += 1;
        }
        // Pick a move (non-pass favoured) and a price.
        let non_pass: Vec<&super::super::generator::IndexedMove> =
            indexed.iter().filter(|x| !matches!(x.mv.action, Action::Pass { .. })).collect();
        let x = if !non_pass.is_empty() && rng.gen::<f64>() < 0.75 {
            non_pass[rng.gen_range(0..non_pass.len())]
        } else {
            &indexed[rng.gen_range(0..indexed.len())]
        };
        let price = random_price(&x.mv, &mut rng);
        let chosen = super::super::generator::priced(x.mv.clone(), price).map_err(|e| e.message)?;
        // decode(index) through the API.
        let decoded = g.rl_move_for_index(x.index, price, policy).map_err(|e| format!("{}: decode: {e}", at(n)))?;
        if format!("{:?}", decoded.action) != format!("{:?}", chosen.action) {
            return Err(format!("{}: decode({}) = {:?}, not {:?}", at(n), x.index, decoded.action, chosen.action));
        }
        st.decoded += 1;
        let (block, _) = move_key(&g, &chosen).map_err(|e| e.message)?;
        *st.played_by_block.entry(block.to_string()).or_default() += 1;
        // Apply by move (a clone) vs by index (the walk).
        let wire = g.g1867_move_wire(&chosen);
        let mut by_move = g.clone_for_search();
        by_move.g1867_apply_move(&chosen).map_err(|e| format!("{}: apply {:?}: {e}", at(n), chosen.action))?;
        let logged = g.rl_apply_index(x.index, price, policy).map_err(|e| format!("{}: apply index: {e}", at(n)))?;
        if logged != wire || g.action_log.last() != Some(&wire) {
            return Err(format!("{}: logged {logged} for {wire}", at(n)));
        }
        let (a, b) = (digest(&g), digest(&by_move));
        if a != b {
            let i = a.bytes().zip(b.bytes()).position(|(x, y)| x != y).unwrap_or(a.len().min(b.len()));
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

/// Walks in parallel threads; all must pass.
fn run(walks: Vec<(usize, u64, CrossPlayerPolicy)>, max_moves: usize) -> Stats {
    let results: Vec<Result<Stats, String>> = std::thread::scope(|s| {
        let handles: Vec<_> =
            walks.iter().map(|&(p, seed, pol)| s.spawn(move || index_walk(p, seed, max_moves, pol))).collect();
        handles.into_iter().map(|h| h.join().unwrap_or_else(|_| Err("PANIC".into()))).collect()
    });
    let mut total = Stats::default();
    let mut failures = Vec::new();
    for r in results {
        match r {
            Ok(s) => total.merge(s),
            Err(e) => failures.push(e),
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    total
}

#[test]
fn index_fuzz_3_to_6_players() {
    let mut walks: Vec<(usize, u64, CrossPlayerPolicy)> =
        (3..=6).map(|p| (p, 1, CrossPlayerPolicy::RESTRICTED)).collect();
    walks.extend([(3, 2, CrossPlayerPolicy::RULES), (6, 2, CrossPlayerPolicy::RULES)]);
    let st = run(walks, 4000);
    eprintln!("1867 index fuzz: {st:#?}");
    assert!(st.finished >= 4, "most walks reach the game end: {st:?}");
    for block in [
        "Pass",
        "Bid.company",
        "Bid.corporation",
        "Par",
        "BuyShares",
        "SellShares",
        "PlaceToken",
        "BuyTrain.depot",
        "BuyTrain.corp",
        "Dividend",
        "RunRoutes",
        "Merge",
        "LayTile",
    ] {
        assert!(st.by_block.get(block).copied().unwrap_or(0) > 0, "never offered: {block}: {:?}", st.by_block);
    }
}

/// The verification run behind `docs/rl_descriptors.md`'s numbers:
/// `G1867_INDEX_SEEDS=10 cargo test --release --no-default-features
/// rl::g1867::tests::index_fuzz_many -- --ignored --nocapture`.
#[test]
#[ignore]
fn index_fuzz_many() {
    let seeds: u64 = std::env::var("G1867_INDEX_SEEDS").ok().and_then(|s| s.parse().ok()).unwrap_or(5);
    let mut walks = Vec::new();
    for p in 3..=6 {
        for seed in 1..=seeds {
            walks.push((p, seed, CrossPlayerPolicy::RESTRICTED));
            walks.push((p, 1000 + seed, CrossPlayerPolicy::RULES));
        }
    }
    let st = run(walks, 5000);
    eprintln!("1867 index fuzz: {st:#?}");
}
