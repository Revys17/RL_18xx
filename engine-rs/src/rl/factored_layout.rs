//! The registry form of a `Factored` title's flat slot layout — 1830's
//! FROZEN v1 layout (and the test-only TEST-5SHARE): the same indices
//! `action_index::build_layout` places, walked block by block, so every one
//! of them gets a descriptor.
//!
//! Nothing here computes an index: the index function stays
//! `action_index::legal_action_to_index_in`. The builder reads the
//! `SlotLayout`'s lists (company / corporation / train / hex / tile orders,
//! city tables, company sub-blocks), and refuses to build unless every
//! block lands at the `SlotLayout`'s own offset and the totals agree.
//!
//! What the frozen layout's indices mean, block by block (offsets for
//! 1830): see `docs/rl_descriptors.md` "The frozen 1830 layout".

use super::descriptor::{flags, EntKind, Family, PriceKind};
use super::layout::{Axis, BlockSpec, RlLayout, Vocab};
use crate::action_index::SlotLayout;
use crate::title::GameTitle;

/// The layout version of a factored title's registry (`"1830-v1"`).
pub fn factored_version(title: &str) -> String {
    format!("{title}-v1")
}

/// The block specs of a factored title's slot layout, in index order.
pub fn factored_block_specs(title: &dyn GameTitle, lo: &SlotLayout) -> Vec<BlockSpec> {
    use Axis::*;
    let ent = |slot: u8, kind: EntKind, syms: &[&'static str]| Ent {
        slot,
        kind,
        syms: syms.to_vec(),
    };
    let corps = &lo.corporation_offsets;
    let companies = &lo.company_offsets;
    let trains = &lo.train_type_offsets;
    let sources: Vec<&'static str> = lo.share_location_offsets.iter().map(|s| source_choice(s)).collect();

    // SellShares counts are share units (the factored `count`); the
    // descriptor's param is the percent sold.
    let unit = title
        .corporations()
        .iter()
        .flat_map(|c| c.shares.iter().copied())
        .min()
        .unwrap_or(10) as i32;
    let sell_percents: Vec<i32> = (1..=lo.sell_count_slots as i32).map(|n| n * unit).collect();

    // PlaceToken slots in offset order.
    let mut cities: Vec<(&'static str, usize, u32)> = lo.city_offsets.iter().map(|(&(h, c), &o)| (h, c, o)).collect();
    cities.sort_by_key(|&(_, _, o)| o);

    let mut specs = vec![
        BlockSpec::new("Pass", Family::Pass, vec![]),
        BlockSpec::new("Bid", Family::BidCompany, vec![ent(0, EntKind::Company, companies)]).priced(PriceKind::Bid),
        BlockSpec::new(
            "Par",
            Family::Par,
            vec![ent(0, EntKind::Corporation, corps), Param(lo.par_price_offsets.clone())],
        ),
        BlockSpec::new(
            "BuyShares",
            Family::BuyShares,
            vec![ent(0, EntKind::Corporation, corps), Choice(sources.clone())],
        ),
        BlockSpec::new(
            "SellShares",
            Family::SellShares,
            vec![ent(0, EntKind::Corporation, corps), Param(sell_percents)],
        ),
        BlockSpec::new(
            "PlaceToken",
            Family::PlaceToken,
            vec![City(cities.iter().map(|&(h, c, _)| (h, c)).collect())],
        ),
        BlockSpec::new(
            "LayTile",
            Family::LayTile,
            vec![Hex(lo.hex_offsets.clone()), Tile(lo.tile_offsets.clone()), Rotation],
        ),
        // The single depot slot: the depot's next train, whatever it is.
        BlockSpec::new("BuyTrain", Family::BuyTrainDepot, vec![]).flagged(flags::TRAIN_FROM_STATE),
        // Discarded trains at face value, per train type.
        BlockSpec::new(
            "BuyTrain.pool",
            Family::BuyTrainPool,
            vec![ent(0, EntKind::Train, trains)],
        ),
        // Another corporation's train; only the first legacy price slot is live.
        BlockSpec::new(
            "BuyTrain.corp",
            Family::BuyTrainCorp,
            vec![
                ent(0, EntKind::Corporation, corps),
                ent(1, EntKind::Train, trains),
                LegacyPrice(lo.train_price_offsets.clone()),
            ],
        )
        .priced(PriceKind::TrainTransfer),
        BlockSpec::new(
            "DiscardTrain",
            Family::DiscardTrain,
            vec![ent(0, EntKind::Train, trains)],
        ),
        BlockSpec::new(
            "Dividend",
            Family::Dividend,
            vec![Choice(lo.dividend_offsets.iter().map(|k| dividend_choice(k)).collect())],
        ),
        BlockSpec::new(
            "BuyCompany",
            Family::BuyCompany,
            vec![
                ent(0, EntKind::Company, companies),
                LegacyPrice(lo.buy_company_price_offsets.clone()),
            ],
        )
        .priced(PriceKind::CompanyPurchase),
        BlockSpec::new("Bankrupt", Family::Bankrupt, vec![]),
        BlockSpec::new("RunRoutes", Family::RunRoutes, vec![]),
    ];

    // A company exchanged for a share (1830: MH -> NYC), from the source
    // (ipo / market). The company and corporation are pinned when the title
    // has exactly one such exchange (the index does not name them).
    let exchanges: Vec<(&'static str, &'static str)> = companies
        .iter()
        .filter_map(|&c| crate::abilities::exchange(title.name(), c).map(|(corps, _)| (c, corps)))
        .flat_map(|(c, corps)| corps.iter().map(move |&k| (c, k)))
        .collect();
    let mut axes = Vec::new();
    if let [(company, corp)] = exchanges[..] {
        axes.push(Via(vec![company]));
        axes.push(ent(0, EntKind::Corporation, &[corp]));
    }
    axes.push(Choice(sources));
    specs.push(BlockSpec::new("CompanyBuyShares", Family::ExchangeShares, axes));

    // Company special lays (teleport first, then bonus lays, as placed).
    for b in &lo.company_lay_tile_blocks {
        let mut axes = vec![Via(vec![b.sym])];
        if let [hex] = b.hexes {
            axes.push(Hex(vec![*hex]));
        }
        axes.push(Tile(b.tiles.to_vec()));
        axes.push(Rotation);
        specs.push(BlockSpec::new(
            format!("CompanyLayTile:{}", b.sym),
            Family::LayTile,
            axes,
        ));
    }
    // A teleport company's token (city 0 of its teleport hex).
    for &(sym, _) in &lo.company_place_token_offsets {
        let mut axes = vec![Via(vec![sym])];
        if let Some(([hex], _)) = crate::abilities::teleport(title.name(), sym) {
            axes.push(City(vec![(*hex, 0)]));
        }
        specs.push(BlockSpec::new(
            format!("CompanyPlaceToken:{}", sym),
            Family::PlaceToken,
            axes,
        ));
    }
    if let Some(d) = lo.discount_train {
        specs.push(BlockSpec::new(
            "BuyTrainDFull",
            Family::BuyTrainDepot,
            vec![ent(0, EntKind::Train, &[d])],
        ));
        // The traded train is chosen by decode (any eligible owned train).
        specs.push(
            BlockSpec::new(
                "BuyTrainDTradeIn",
                Family::BuyTrainExchange,
                vec![ent(0, EntKind::Train, &[d])],
            )
            .flagged(flags::TRAIN_FROM_STATE),
        );
    }
    specs
}

fn source_choice(s: &str) -> &'static str {
    match s {
        "ipo" => "source:ipo",
        "market" => "source:market",
        other => panic!("unknown share source {other:?}"),
    }
}

fn dividend_choice(k: &str) -> &'static str {
    match k {
        "payout" => "dividend:payout",
        "withhold" => "dividend:withhold",
        "half" => "dividend:half",
        other => panic!("unknown dividend kind {other:?}"),
    }
}

/// The registry of a factored title's slot layout. Errs unless every block
/// starts where the `SlotLayout` places it and the totals agree.
pub fn factored_registry(title: &'static dyn GameTitle, lo: &SlotLayout) -> Result<RlLayout, String> {
    let layout = RlLayout::build(
        title.name(),
        factored_version(title.name()),
        Vocab::for_title(title),
        factored_block_specs(title, lo),
    )?;
    if layout.total != lo.total {
        return Err(format!(
            "{}: registry total {} != slot layout total {}",
            title.name(),
            layout.total,
            lo.total
        ));
    }
    // Every SlotLayout block start is a registry block start.
    let starts = [
        ("Pass", "Pass"),
        ("Bid", "Bid"),
        ("Par", "Par"),
        ("BuyShares", "BuyShares"),
        ("SellShares", "SellShares"),
        ("PlaceToken", "PlaceToken"),
        ("LayTile", "LayTile"),
        ("BuyTrain", "BuyTrain"),
        ("DiscardTrain", "DiscardTrain"),
        ("Dividend", "Dividend"),
        ("BuyCompany", "BuyCompany"),
        ("Bankrupt", "Bankrupt"),
        ("RunRoutes", "RunRoutes"),
        ("CompanyBuyShares", "CompanyBuyShares"),
        ("BuyTrainDFull", "BuyTrainDFull"),
        ("BuyTrainDTradeIn", "BuyTrainDTradeIn"),
    ];
    for (key, block) in starts {
        let Some(&want) = lo.action_offsets.get(key) else {
            continue;
        };
        let got = layout.block(block).map(|b| b.offset);
        if got != Some(want) {
            return Err(format!(
                "{}: block {block} at {got:?}, slot layout {key} at {want}",
                title.name()
            ));
        }
    }
    // The company sub-blocks.
    let lay0 = lo.action_offsets["CompanyLayTile"];
    for b in &lo.company_lay_tile_blocks {
        let got = layout.block(&format!("CompanyLayTile:{}", b.sym)).map(|x| x.offset);
        if got != Some(lay0 + b.offset) {
            return Err(format!(
                "{}: CompanyLayTile:{} at {got:?}, slot layout at {}",
                title.name(),
                b.sym,
                lay0 + b.offset
            ));
        }
    }
    let tok0 = lo.action_offsets["CompanyPlaceToken"];
    for &(sym, off) in &lo.company_place_token_offsets {
        let got = layout.block(&format!("CompanyPlaceToken:{sym}")).map(|x| x.offset);
        if got != Some(tok0 + off) {
            return Err(format!(
                "{}: CompanyPlaceToken:{sym} at {got:?}, slot layout at {}",
                title.name(),
                tok0 + off
            ));
        }
    }
    Ok(layout)
}
