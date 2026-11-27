//! The shadow key v2's fields (docs/SITE_ENGINE_RS_1867.md §6, contract K4): what v2 adds to v1, present only where
//! the title has it: per corporation `loans` and (in an operating round) `interest`; `loans_left`; `national`;
//! `trainless`; `merger` (a merger round); and every title's `auction` (the bids in progress). The Ruby twin is
//! rl18xx-next `lib/engine_shadow.rb` (and `scripts/site_engine/ruby/keys_reference.rb`).

use super::super::game::SiteGame;
use crate::rounds::Round;
use serde_json::{json, Map, Value};

/// Ruby's `@interest[corporation]` in an operating round (`loans_due_interest`: the loans held when the OR began, or
/// when the company's LoanOperations step re-counted them).
pub fn interest(g: &SiteGame, sym: &str) -> Option<u32> {
    match &g.game.round {
        Round::Operating(s) => {
            // a nationalized major is a new Ruby object (`reset_corporation`): nothing due until the next OR counts it
            let reset = g.game.corp_idx.get(sym).is_some_and(|&ci| g.game.corporations[ci].ipo_price.is_none());
            Some(if reset { 0 } else { s.interest_snapshot.get(sym).copied().unwrap_or(0) })
        }
        _ => None,
    }
}

/// A corporation's v2 fields (1867: every corporation has loans).
pub fn corp_fields(g: &SiteGame, sym: &str, m: &mut Map<String, Value>) {
    let Some(&ci) = g.game.corp_idx.get(sym) else { return };
    m.insert("loans".into(), json!(g.game.corporations[ci].loans));
    if let Some(i) = interest(g, sym) {
        m.insert("interest".into(), json!(i));
    }
}

/// "<hex>-<city index>" of each used token of the national (Ruby `national.tokens.select(&:used)`; the neutral tokens
/// on D2 and L12 are not the national's own).
fn national_tokens(g: &SiteGame) -> Vec<String> {
    let Some(&ci) = g.game.corp_idx.get("CN") else { return vec![] };
    let mut out = Vec::new();
    for t in g.game.corporations[ci].tokens.iter().filter(|t| t.used) {
        let hex = &t.city_hex_id;
        let idx = g.game.hex_ref(hex).and_then(|h| {
            h.tile.cities.iter().position(|c| c.tokens.iter().flatten().any(|x| x.corporation_id == "CN" && x.token_type != "neutral"))
        });
        out.push(format!("{hex}-{}", idx.map_or("?".to_string(), |i| i.to_string())));
    }
    out
}

/// The game's v2 fields (1867).
pub fn game_fields(g: &SiteGame, out: &mut Map<String, Value>) {
    let game = &g.game;
    out.insert("loans_left".into(), json!(game.loans_remaining));
    let mut companies: Vec<String> = game
        .companies
        .iter()
        .filter(|c| !c.closed && c.owner.corp_sym() == Some("CN") && c.owner.is_corporation())
        .map(|c| c.sym.clone())
        .collect();
    companies.sort();
    out.insert(
        "national".into(),
        json!({"tokens": national_tokens(g), "companies": companies, "reservations": game.national_reservations}),
    );
    out.insert("trainless".into(), json!(game.trainless_major));
    if let Round::Merger(s) = &game.round {
        let removing = s.corporations_removing_tokens.as_ref().and_then(|l| l.first().cloned());
        // the share dealing's lists only while a new major's shares are dealt (Ruby keeps the last ones after)
        let dealing: &[u32] = if s.converted.is_some() { &s.share_dealing_players } else { &[] };
        let multiple: &[u32] = if s.converted.is_some() { &s.share_dealing_multiple } else { &[] };
        out.insert(
            "merger".into(),
            json!({
                "converted": s.converted,
                "merge_type": g.merger.merge_type,
                "converts": g.merger.converts,
                "dealing": dealing,
                "multiple": multiple,
                "merging": s.merging,
                "converting": s.converting,
                "merge_major": s.merge_major,
                "removing": removing,
            }),
        );
    }
}

/// The bids in progress (every title): `{lot, bids: [[item, player, price]...] sorted (item, price, player), dutch?}`,
/// None when nothing is up and no bid is open, or when the active step is not an auction. 1830: the waterfall's bids
/// on every private, `lot` the private with more than one bid (Ruby `@auctioning`). 1867: the opening auction's lot
/// (`SingleItemAuction`, with `dutch`), and a stock round's minor auction (`BuySellParSharesViaBid`).
pub fn auction(g: &mut SiteGame) -> Option<Value> {
    if g.game.finished {
        return None;
    }
    let mut bids: Vec<(String, i32, u32)> = Vec::new();
    let mut lot: Option<String> = None;
    let mut dutch: Option<bool> = None;
    match &g.game.round {
        Round::Auction(s) => {
            if s.pending_par.is_some() {
                return None;
            }
            for (&ci, list) in &s.bids {
                let Some(c) = g.game.companies.get(ci) else { continue };
                for b in list {
                    bids.push((c.sym.clone(), b.price, b.player_id));
                }
            }
            match &s.single_item {
                Some(si) => {
                    lot = s.auctioning.and_then(|ci| g.game.companies.get(ci)).map(|c| c.sym.clone());
                    dutch = Some(si.dutch_mode);
                }
                None => {
                    lot = g.game.auctioning_company().filter(|c| g.game.auction_bids(c).len() > 1);
                }
            }
        }
        Round::Stock(_) => {
            // the active step must be the share step (a pending home station is HomeToken's)
            if super::is_1867(g) {
                let active = super::active_step(&mut g.game).map(|x| x.0);
                if active != Some(crate::steps::StepKind::BuySellParShares) {
                    return None;
                }
            }
            let Round::Stock(s) = &g.game.round else { return None };
            if let Some(a) = &s.bid_auction {
                lot = Some(a.corporation_sym.clone());
                for b in &a.bids {
                    bids.push((a.corporation_sym.clone(), b.price, b.player_id));
                }
            }
        }
        _ => return None,
    }
    if lot.is_none() && bids.is_empty() {
        return None;
    }
    bids.sort();
    let mut m = Map::new();
    m.insert("lot".into(), json!(lot));
    m.insert("bids".into(), json!(bids.iter().map(|(c, price, p)| json!([c, p, price])).collect::<Vec<_>>()));
    if let Some(d) = dutch {
        m.insert("dutch".into(), json!(d));
    }
    Some(Value::Object(m))
}
