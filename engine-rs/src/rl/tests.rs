//! The RL layouts: the frozen 1830 registry against the flat slot layout
//! and the index function, fingerprints per title, and the "not
//! bridge-ready" seam for the titles without a layout.

use std::collections::{BTreeMap, HashMap};

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use super::descriptor::{flags, Descriptor, EntKind, Family, PriceKind, NONE};
use super::{rl_layout, RlLayout};
use crate::actions::{Action, DividendKind};
use crate::factored::LegalAction;
use crate::game::BaseGame;

/// The titles registered without an RL layout (rules-first).
const NOT_READY: [&str; 5] = ["1867", "1822", "1822PNW", "1822MX", "1822CA"];

fn game(title: &str, n: u32) -> BaseGame {
    // The 1822 family's minors have numeric syms: seat ids above 100 there.
    let first = if crate::title::resolve(title).is_g1822_family() {
        1001
    } else {
        1
    };
    let ids: Vec<u32> = (first..first + n).collect();
    let players: HashMap<u32, String> = ids.iter().map(|&i| (i, format!("P{i}"))).collect();
    BaseGame::build_titled(title, ids, players)
}

// ---------------------------------------------------------------------------
// The frozen 1830 layout
// ---------------------------------------------------------------------------

/// Every block of the 1830 v1 registry: (name, family, offset, size). The
/// offsets of the named slot-layout blocks are the frozen constants
/// (`action_index::layout_matches_frozen_1830_constants`).
#[test]
fn frozen_1830_registry_blocks() {
    let lo = rl_layout("1830").expect("1830 is bridge-ready");
    assert_eq!(lo.total, crate::action_index::POLICY_SIZE);
    assert_eq!(lo.descriptors.len(), lo.total as usize);
    assert_eq!(lo.version, "1830-v1");
    let got: Vec<(&str, &str, u32, u32)> = lo
        .blocks
        .iter()
        .map(|b| (b.spec.name.as_str(), b.spec.family.name(), b.offset, b.size))
        .collect();
    assert_eq!(
        got,
        vec![
            ("Pass", "Pass", 0, 1),
            ("Bid", "Bid.company", 1, 6),
            ("Par", "Par", 7, 48),
            ("BuyShares", "BuyShares", 55, 16),
            ("SellShares", "SellShares", 71, 40),
            ("PlaceToken", "PlaceToken", 111, 28),
            ("LayTile", "LayTile", 139, 93 * 46 * 6),
            ("BuyTrain", "BuyTrain.depot", 25807, 1),
            ("BuyTrain.pool", "BuyTrain.pool", 25808, 6),
            ("BuyTrain.corp", "BuyTrain.corp", 25814, 8 * 6 * 14),
            ("DiscardTrain", "DiscardTrain", 26486, 6),
            ("Dividend", "Dividend", 26492, 2),
            ("BuyCompany", "BuyCompany", 26494, 12),
            ("Bankrupt", "Bankrupt", 26506, 1),
            ("RunRoutes", "RunRoutes", 26507, 1),
            ("CompanyBuyShares", "ExchangeShares", 26508, 2),
            ("CompanyLayTile:DH", "LayTile", 26510, 6),
            ("CompanyLayTile:CS", "LayTile", 26516, 18),
            ("CompanyPlaceToken:DH", "PlaceToken", 26534, 1),
            ("BuyTrainDFull", "BuyTrain.depot", 26535, 1),
            ("BuyTrainDTradeIn", "BuyTrain.exchange", 26536, 1),
        ]
    );
    // Dead slots: 13 of 14 legacy price slots per (corp, train), the "max"
    // BuyCompany slot per company.
    let dead = lo.descriptors.iter().filter(|d| d.is_dead()).count();
    assert_eq!(dead, 8 * 6 * 13 + 6);
    // Spot checks against ActionMapper.init_actions.
    let v = &lo.vocab;
    let d = lo.descriptors[7 + 3 * 6 + 2]; // Par [B&O, 76]
    assert_eq!(
        (d.ent0_kind, v.corporations[d.ent0 as usize], d.param),
        (EntKind::Corporation.id(), "B&O", 76)
    );
    let d = lo.descriptors[71 + 5 + 2]; // SellShares [NYC, 3 shares]
    assert_eq!((v.corporations[d.ent0 as usize], d.param), ("NYC", 30));
    let d = lo.descriptors[111 + 17]; // PlaceToken ("E5", 1)
    assert_eq!((v.hexes[d.hex as usize], d.city), ("E5", 1));
    // LayTile [hex 1 = I1, tile 2 = "16", rot 5] in ActionMapper order.
    let d = lo.descriptors[139 + 46 * 6 + 2 * 6 + 5];
    assert_eq!(
        (v.hexes[d.hex as usize], v.tiles[d.tile as usize], d.rot),
        ("I1", "16", 5)
    );
    let d = lo.descriptors[26508 + 1]; // MH -> NYC from the market
    assert_eq!(
        (
            v.companies[d.via as usize],
            v.corporations[d.ent0 as usize],
            super::descriptor::CHOICES[d.choice as usize]
        ),
        ("MH", "NYC", "source:market")
    );
    let d = lo.descriptors[26516 + 2 * 6 + 3]; // CS lays tile 58 on B20 at rot 3
    assert_eq!(
        (
            v.companies[d.via as usize],
            v.hexes[d.hex as usize],
            v.tiles[d.tile as usize],
            d.rot
        ),
        ("CS", "B20", "58", 3)
    );
    let d = lo.descriptors[26534];
    assert_eq!(
        (v.companies[d.via as usize], v.hexes[d.hex as usize], d.city),
        ("DH", "F16", 0)
    );
    let d = lo.descriptors[26536];
    assert_eq!(
        (d.family, v.trains[d.ent0 as usize], d.ent1, d.flags),
        (Family::BuyTrainExchange.id(), "D", NONE, flags::TRAIN_FROM_STATE)
    );
    // Price kinds only on the three price-bearing blocks' live slots.
    let priced: BTreeMap<i32, usize> = lo
        .descriptors
        .iter()
        .filter(|d| d.price_kind != NONE && !d.is_dead())
        .fold(BTreeMap::new(), |mut m, d| {
            *m.entry(d.price_kind).or_default() += 1;
            m
        });
    assert_eq!(
        priced,
        BTreeMap::from([
            (PriceKind::Bid.id(), 6),
            (PriceKind::TrainTransfer.id(), 48),
            (PriceKind::CompanyPurchase.id(), 6)
        ])
    );
}

/// Each title with a layout has its fingerprint pinned: a change to what
/// any index means (block, axis entry, vocabulary order, descriptor field)
/// must show up here, with a layout version bump.
#[test]
fn layout_fingerprints_are_pinned_per_title() {
    let pinned: &[(&str, &str, u32)] = &[
        ("1830", "e721aa6520116e2a", 26537),
        ("TEST-5SHARE", "7196801649141472", 26513),
    ];
    for (title, fp, total) in pinned {
        let lo = rl_layout(title).unwrap_or_else(|e| panic!("{title}: {e}"));
        assert_eq!(lo.total, *total, "{title}");
        assert_eq!(
            lo.fingerprint_hex(),
            *fp,
            "{title}: layout fingerprint moved (bump the version)"
        );
    }
    // Every title with a layout is pinned above; every other is not ready.
    for t in crate::title::all_titles() {
        let ready = t.rl_layout_ready();
        assert_eq!(ready, pinned.iter().any(|(n, _, _)| *n == t.name()), "{}", t.name());
        assert_eq!(rl_layout(t.name()).is_ok(), ready, "{}", t.name());
    }
}

// ---------------------------------------------------------------------------
// The "not bridge-ready" seam
// ---------------------------------------------------------------------------

#[test]
fn titles_without_a_layout_refuse_every_index_path() {
    for title in NOT_READY {
        let t = crate::title::resolve(title);
        assert!(!t.rl_layout_ready() && !t.rl_encoder_ready(), "{title}");
        let msg = |e: crate::actions::GameError| e.message;
        assert!(
            msg(rl_layout(title).unwrap_err()).contains("not bridge-ready"),
            "{title}"
        );
        assert!(msg(crate::action_index::try_layout_for(title).err().unwrap()).contains("not bridge-ready"));
        let mut g = game(title, 4);
        assert!(
            msg(g.factored_layout().err().unwrap()).contains("not bridge-ready"),
            "{title}"
        );
        assert!(
            msg(g.legal_action_for_index(0).unwrap_err()).contains("not bridge-ready"),
            "{title}"
        );
        assert!(
            msg(g.decode_index(0, None).unwrap_err()).contains("not bridge-ready"),
            "{title}"
        );
        // The tree takes its layout from the root game (the same seam).
        assert!(
            crate::mcts::RustMCTSPlayer::from_game(g, None, None, None, None).is_err(),
            "{title}"
        );
    }
    assert!(rl_layout("NO-SUCH").unwrap_err().message.contains("unknown game title"));
    // The 1830 seam resolves to the very same layout as the global one.
    let g = game("1830", 4);
    assert!(std::ptr::eq(
        g.factored_layout().unwrap(),
        crate::action_index::layout()
    ));
}

// ---------------------------------------------------------------------------
// Descriptors against the index function and the native decode
// ---------------------------------------------------------------------------

fn s<'a>(m: &'a HashMap<String, serde_json::Value>, k: &str) -> Option<&'a str> {
    m.get(k).and_then(|v| v.as_str())
}

fn n(m: &HashMap<String, serde_json::Value>, k: &str) -> Option<i64> {
    m.get(k).and_then(|v| v.as_i64())
}

fn base_tile(id: &str) -> &str {
    id.rsplit_once('-').map_or(id, |(b, _)| b)
}

/// Whether descriptor `d` (of the index `la` maps to) says what `la` (the
/// factored enumeration's action) and `action` (its native decode) do.
fn agrees(lo: &RlLayout, d: &Descriptor, la: &LegalAction, action: &Action) -> Result<(), String> {
    let v = &lo.vocab;
    let ent = |kind: EntKind, i: i32| -> Option<&str> {
        let list = match kind {
            EntKind::Corporation => &v.corporations,
            EntKind::Company => &v.companies,
            EntKind::Train => &v.trains,
            EntKind::Player => return None,
        };
        (i >= 0).then(|| list[i as usize])
    };
    let corp0 = || ent(EntKind::Corporation, d.ent0);
    let company0 = || ent(EntKind::Company, d.ent0);
    let train = |i: i32| ent(EntKind::Train, i);
    let via = || (d.via >= 0).then(|| v.companies[d.via as usize]);
    let hex = || (d.hex >= 0).then(|| v.hexes[d.hex as usize]);
    let tile = || (d.tile >= 0).then(|| v.tiles[d.tile as usize]);
    let choice = || (d.choice >= 0).then(|| super::descriptor::CHOICES[d.choice as usize]);
    let check = |ok: bool, what: &str| if ok { Ok(()) } else { Err(what.to_string()) };

    check(!d.is_dead(), "a legal index is DEAD")?;
    // Prices: a price kind exactly where the enumeration has a real range.
    match la.price_range {
        Some((a, b)) if a != b => check(d.price_kind != NONE, "a price range on an unpriced index")?,
        _ => {}
    }
    if d.price_kind != NONE {
        check(la.price_range.is_some(), "a priced index without a range")?;
    }
    let t = la.action_type.as_str();
    match d.family_enum() {
        Family::Pass => check(t == "Pass" && matches!(action, Action::Pass { .. }), "pass"),
        Family::BidCompany => check(
            t == "Bid"
                && s(&la.entity, "private") == company0()
                && matches!(action, Action::Bid { company_sym, .. } if Some(company_sym.as_str()) == company0()),
            "bid",
        ),
        Family::Par => check(
            t == "Par"
                && s(&la.entity, "corp") == corp0()
                && n(&la.params, "par_price") == Some(d.param as i64)
                && matches!(action, Action::Par { corporation_sym, share_price, .. }
                    if Some(corporation_sym.as_str()) == corp0() && *share_price == d.param),
            "par",
        ),
        Family::BuyShares => check(
            t == "BuyShares"
                && s(&la.entity, "private").is_none()
                && s(&la.entity, "corp") == corp0()
                && choice() == s(&la.params, "source").map(|x| if x == "ipo" { "source:ipo" } else { "source:market" })
                && matches!(action, Action::BuyShares { corporation_sym, .. } if Some(corporation_sym.as_str()) == corp0()),
            "buy shares",
        ),
        Family::ExchangeShares => check(
            (t == "CompanyBuyShares" || t == "BuyShares")
                && s(&la.entity, "private") == via()
                && s(&la.entity, "corp") == corp0()
                && choice() == s(&la.params, "source").map(|x| if x == "ipo" { "source:ipo" } else { "source:market" })
                && matches!(action, Action::BuyShares { entity_id, corporation_sym, .. }
                    if Some(entity_id.as_str()) == via() && Some(corporation_sym.as_str()) == corp0()),
            "exchange",
        ),
        Family::SellShares => check(
            t == "SellShares"
                && s(&la.entity, "corp") == corp0()
                && n(&la.params, "percent") == Some(d.param as i64)
                && matches!(action, Action::SellShares { corporation_sym, percent, .. }
                    if Some(corporation_sym.as_str()) == corp0() && *percent as i32 == d.param),
            "sell shares",
        ),
        Family::PlaceToken => check(
            t == "PlaceToken"
                && s(&la.entity, "private") == via()
                && s(&la.params, "hex") == hex()
                && n(&la.params, "city").unwrap_or(0) == d.city as i64
                && matches!(action, Action::PlaceToken { entity_id, hex_id, city_index }
                    if Some(hex_id.as_str()) == hex() && *city_index as i32 == d.city
                        && via().map_or(true, |c| entity_id == c)),
            "place token",
        ),
        // A teleport company's lay with a tile outside its block falls
        // through to the regular LayTile slots (Python parity), so a slot
        // without `via` may carry a company lay.
        Family::LayTile => check(
            t == "LayTile"
                && (via().is_none() || s(&la.entity, "private") == via())
                && s(&la.params, "hex") == hex()
                && s(&la.params, "tile") == tile()
                && n(&la.params, "rotation") == Some(d.rot as i64)
                && matches!(action, Action::LayTile { entity_id, hex_id, tile_id, rotation }
                    if Some(hex_id.as_str()) == hex() && Some(base_tile(tile_id)) == tile()
                        && *rotation as i32 == d.rot && via().map_or(true, |c| entity_id == c)),
            "lay tile",
        ),
        Family::BuyTrainDepot => check(
            t == "BuyTrain"
                && s(&la.entity, "source") == Some("depot")
                && la.entity.get("exchange").is_none()
                && !la.depot_discarded
                && (d.flags & flags::TRAIN_FROM_STATE != 0 || s(&la.entity, "train") == train(d.ent0))
                && matches!(action, Action::BuyTrain { from, exchange: None, .. } if from == "depot"),
            "depot train",
        ),
        // The frozen layout's discard slots also take a trade-in of a train
        // whose name is in the pool (Python's check order): no exchange
        // check here.
        Family::BuyTrainPool => check(
            t == "BuyTrain"
                && (s(&la.entity, "source") == Some("discard") || la.depot_discarded)
                && s(&la.entity, "train") == train(d.ent0),
            "pool train",
        ),
        Family::BuyTrainExchange => check(
            t == "BuyTrain"
                && s(&la.entity, "source") == Some("depot")
                && la.entity.get("exchange").is_some()
                && s(&la.entity, "train") == train(d.ent0)
                && matches!(action, Action::BuyTrain { exchange: Some(_), .. }),
            "trade-in",
        ),
        Family::BuyTrainCorp => check(
            t == "BuyTrain"
                && s(&la.entity, "source") == corp0()
                && s(&la.entity, "train") == train(d.ent1)
                && matches!(action, Action::BuyTrain { from, .. } if Some(from.as_str()) == corp0()),
            "corp train",
        ),
        Family::DiscardTrain => check(
            t == "DiscardTrain" && s(&la.params, "train") == train(d.ent0),
            "discard",
        ),
        Family::Dividend => check(
            t == "Dividend"
                && choice().map(|c| c.trim_start_matches("dividend:")) == s(&la.params, "kind")
                && matches!(action, Action::Dividend { kind, .. }
                    if Some(match kind { DividendKind::Payout => "dividend:payout", DividendKind::Withhold => "dividend:withhold", DividendKind::Half => "dividend:half" }) == choice()),
            "dividend",
        ),
        Family::BuyCompany => check(
            t == "BuyCompany"
                && s(&la.entity, "private") == company0()
                && matches!(action, Action::BuyCompany { company_sym, .. } if Some(company_sym.as_str()) == company0()),
            "buy company",
        ),
        Family::Bankrupt => check(t == "Bankrupt" && matches!(action, Action::Bankrupt { .. }), "bankrupt"),
        Family::RunRoutes => check(t == "RunRoutes" && matches!(action, Action::RunRoutes { .. }), "routes"),
        other => Err(format!("family {other:?} in a factored layout")),
    }
}

/// Random walks: at every state, every legal factored action's index has a
/// descriptor that says what the action and its native decode do. Returns
/// the number of moves played and of operating-round decisions seen.
fn walk(
    title: &str,
    players: u32,
    seed: u64,
    max_actions: usize,
    seen: &mut BTreeMap<&'static str, usize>,
) -> (usize, usize) {
    let lo = rl_layout(title).unwrap();
    let flat = crate::action_index::layout_for(title);
    let company_offset = flat.action_offsets["CompanyBuyShares"];
    let mut g = game(title, players);
    let mut rng = StdRng::seed_from_u64(seed);
    let (mut moves, mut or_moves) = (0, 0);
    for step in 0..max_actions {
        if g.is_finished_pub() {
            break;
        }
        if matches!(g.round, crate::rounds::Round::Operating(_)) {
            or_moves += 1;
        }
        let choices: Vec<LegalAction> = g
            .get_factored_choices_impl()
            .into_iter()
            .filter(|c| !(c.action_type == "PlaceToken" && c.params.get("slot").map_or(false, |x| x.is_null())))
            .collect();
        if choices.is_empty() {
            break;
        }
        let mut scratch = g.clone_for_search();
        let mut indexed = Vec::new();
        for la in &choices {
            let idx = crate::action_index::legal_action_to_index_in(flat, la)
                .unwrap_or_else(|| panic!("{title} seed {seed} step {step}: unindexable {la:?}"));
            let d = lo.descriptors[idx as usize];
            let price = la.price_range.map(|(a, _)| a);
            let action = scratch
                .build_action(la, idx, idx >= company_offset, price)
                .unwrap_or_else(|e| panic!("{title} seed {seed} step {step}: decode {idx}: {e}"));
            if let Err(e) = agrees(lo, &d, la, &action) {
                panic!(
                    "{title} seed {seed} step {step}: index {idx} ({}): {e}\n{d:?}\n{la:?}\n{action:?}",
                    lo.block_of(idx).unwrap().spec.name
                );
            }
            *seen.entry(d.family_enum().name()).or_default() += 1;
            indexed.push((la, idx));
        }
        let (la, idx) = indexed[rng.gen_range(0..indexed.len())];
        let price = la.price_range.map(|(a, b)| rng.gen_range(a..=b));
        let action = g.build_action(la, idx, idx >= company_offset, price).unwrap();
        g.process_action_internal(&action)
            .unwrap_or_else(|e| panic!("{title} seed {seed} step {step}: process {action:?}: {e}"));
        moves += 1;
    }
    (moves, or_moves)
}

#[test]
fn descriptors_agree_with_the_index_function_and_decode_on_1830_walks() {
    let mut seen = BTreeMap::new();
    let mut checked = 0;
    for (i, players) in [3u32, 4, 5, 6, 4, 2].into_iter().enumerate() {
        let (moves, or_moves) = walk("1830", players, 100 + i as u64, 2500, &mut seen);
        assert!(
            moves > 150 && or_moves > 50,
            "walk {i} stalled: {moves} moves, {or_moves} in operating rounds"
        );
        checked += moves;
    }
    eprintln!("1830 walks: {checked} states, legal actions checked per family: {seen:?}");
    for fam in [
        "Pass",
        "Bid.company",
        "Par",
        "BuyShares",
        "SellShares",
        "PlaceToken",
        "LayTile",
        "BuyTrain.depot",
        "BuyTrain.corp",
        "Dividend",
        "RunRoutes",
    ] {
        assert!(
            seen.get(fam).copied().unwrap_or(0) > 0,
            "walks never offered {fam}: {seen:?}"
        );
    }
}

#[test]
fn descriptors_agree_on_the_synthetic_title() {
    let mut seen = BTreeMap::new();
    let (moves, or_moves) = walk("TEST-5SHARE", 4, 7, 2000, &mut seen);
    assert!(
        moves > 150 && or_moves > 50,
        "walk stalled: {moves} moves, {or_moves} in operating rounds"
    );
    assert!(seen.get("SellShares").copied().unwrap_or(0) > 0, "{seen:?}");
    // 5-share corps: a sale moves 20% units.
    let lo = rl_layout("TEST-5SHARE").unwrap();
    let b = lo.block("SellShares").unwrap();
    let params: Vec<i32> = (0..2).map(|j| lo.descriptors[(b.offset + j) as usize].param).collect();
    assert_eq!(params, vec![20, 40]);
}

/// Diagnostic (ignored): the axis sizes, from title data, that the layout
/// sketches of `docs/rl_descriptors.md` quote for the titles without a
/// layout yet. `cargo test --release --no-default-features
/// rl::tests::print_title_axis_sizes -- --ignored --nocapture`
#[test]
#[ignore]
fn print_title_axis_sizes() {
    use super::layout::Axis;
    use crate::title::CorpType;
    for name in ["1830", "1867", "1822", "1822PNW", "1822MX", "1822CA"] {
        let t = crate::title::resolve(name);
        let hexes = Axis::map_hexes(t).len();
        let tiles = Axis::tiles(t).len();
        let g = game(name, 4);
        let city_slots: usize = g.hexes.iter().map(|h| h.tile.cities.len()).sum();
        let kinds = [CorpType::Major, CorpType::Minor, CorpType::National];
        let corps: Vec<(String, usize)> = kinds
            .iter()
            .map(|k| (format!("{k:?}"), Axis::corporations(0, t, Some(*k)).len()))
            .filter(|(_, n)| *n > 0)
            .collect();
        let pars: Vec<(String, usize)> = kinds
            .iter()
            .filter(|k| !Axis::corporations(0, t, Some(**k)).is_empty())
            .map(|k| (format!("{k:?}"), Axis::par_prices(t, *k).len()))
            .collect();
        let unit = t
            .corporations()
            .iter()
            .flat_map(|c| c.shares.iter().copied())
            .min()
            .unwrap_or(10);
        eprintln!(
            "{name}: hexes {hexes} tiles {tiles} lay slots {} initial city slots {city_slots} companies {} \
             (auctionable {}) corps {corps:?} par cells {pars:?} trains {:?} sell units {}",
            hexes * tiles * 6,
            Axis::companies(0, t, false).len(),
            Axis::companies(0, t, true).len(),
            Axis::trains(0, t).entries(),
            t.market_pool_limit() / unit,
        );
    }
}
