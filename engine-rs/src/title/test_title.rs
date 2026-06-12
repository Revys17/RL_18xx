//! A synthetic second title, registered ONLY in test builds — the pre-1867
//! smoke test for the Phase 0.5 title abstraction.
//!
//! `TEST-5SHARE` is 1830 (same map, tiles, companies, trains, phases, market)
//! except every corporation is a 5-SHARE corp (40% president + three 20%
//! certificates, share unit 20%) and OR sets always run at least 2 ORs — the
//! two cheapest knobs that exercise the parameterized seams end-to-end:
//! share-unit math (par cost, float treasury, dividends, buy/sell pricing,
//! bundle construction, decode), the per-title round-flow hook, the per-title
//! ability keying, and the CLEAN default action-layout/encoder derivations
//! (this title does NOT override `action_hex_order`/`action_tile_order`, so
//! the trait defaults get coverage that frozen-1830 cannot give them).
//!
//! Registering it in the test registry also turns every frozen-1830 pin into
//! an ISOLATION test: two titles share the memoized ability/layout/encoder
//! maps, and 1830's numbers must not move.

use std::collections::HashMap;
use std::sync::Arc;

use super::{g1830, CompanyDef, CorporationDef, GameTitle, HexDef, MarketCell, MarketMovement, PhaseDef, TrainDef};
use crate::steps::{FinishedRound, RoundTransition, StepDesc};
use crate::tiles::TileDef;

/// 5-share structure: 40% president + 3 × 20% (the 1867-major shape).
const SHARES_5: &[u8] = &[40, 20, 20, 20];

pub struct Test5Share;

impl GameTitle for Test5Share {
    fn name(&self) -> &'static str {
        "TEST-5SHARE"
    }
    fn auction_steps(&self) -> &'static [StepDesc] {
        g1830::auction_steps()
    }
    fn stock_steps(&self) -> &'static [StepDesc] {
        g1830::stock_steps()
    }
    fn operating_steps(&self) -> &'static [StepDesc] {
        g1830::operating_steps()
    }
    /// A flow that DIFFERS from 1830: OR sets always run at least 2 ORs.
    fn next_round(&self, finished: FinishedRound, phase_operating_rounds: u8) -> RoundTransition {
        g1830::next_round(finished, phase_operating_rounds.max(2))
    }
    fn starting_cash(&self, num_players: u8) -> i32 {
        g1830::starting_cash(num_players)
    }
    fn cert_limit(&self, num_players: u8) -> u8 {
        g1830::cert_limit(num_players)
    }
    fn bank_cash(&self) -> i32 {
        g1830::BANK_CASH
    }
    fn companies(&self) -> Vec<CompanyDef> {
        g1830::companies()
    }
    fn corporations(&self) -> Vec<CorporationDef> {
        g1830::corporations()
            .into_iter()
            .map(|mut cd| {
                cd.shares = SHARES_5;
                cd
            })
            .collect()
    }
    fn trains(&self) -> Vec<TrainDef> {
        g1830::trains()
    }
    fn phases(&self) -> Vec<PhaseDef> {
        g1830::phases()
    }
    fn hex_definitions(&self) -> Vec<HexDef> {
        g1830::hex_definitions()
    }
    fn preprinted_hex_dsl(&self, coord: &str) -> Option<(&'static str, &'static str)> {
        g1830::preprinted_hex_dsl(coord)
    }
    fn tile_counts(&self) -> Vec<(&'static str, u32)> {
        g1830::tile_counts()
    }
    fn tile_cities(&self, tile_id: &str) -> Option<Vec<u8>> {
        g1830::tile_cities(tile_id)
    }
    fn tile_catalog(&self) -> Arc<HashMap<String, TileDef>> {
        crate::tiles::tile_catalog_1830()
    }
    fn market_grid(&self) -> Vec<Vec<Option<MarketCell>>> {
        g1830::market_grid()
    }
    fn market_movement(&self) -> MarketMovement {
        MarketMovement::TwoDimensional
    }
    // NOTE: no action_hex_order / action_tile_order overrides — this title
    // exercises the trait's clean default derivations.
}

// ---------------------------------------------------------------------------
// The smoke tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use crate::game::BaseGame;
    use crate::title::Capitalization;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use std::collections::HashMap;

    const TITLE: &str = "TEST-5SHARE";

    fn new_4p_game() -> BaseGame {
        let mut players = HashMap::new();
        players.insert(1, "Alice".to_string());
        players.insert(2, "Bob".to_string());
        players.insert(3, "Carol".to_string());
        players.insert(4, "Dave".to_string());
        BaseGame::build_titled(TITLE, vec![1, 2, 3, 4], players)
    }

    /// Construction picks up the 5-share structure end-to-end.
    #[test]
    fn five_share_corporations_built_from_title_data() {
        let game = new_4p_game();
        assert_eq!(game.title, TITLE);
        for corp in &game.corporations {
            assert_eq!(corp.shares.len(), 4, "{}", corp.sym);
            assert_eq!(corp.shares[0].percent, 40);
            assert!(corp.shares[0].president);
            assert_eq!(corp.share_unit_percent, 20);
            assert_eq!(corp.share_unit(), 20);
            assert_eq!(corp.num_share_units(), 5);
            assert_eq!(corp.president_percent(), 40);
            assert_eq!(corp.president_share_units(), 2);
            assert_eq!(corp.float_percent, 60);
            assert_eq!(corp.capitalization, Capitalization::Full);
        }
        // 1830 games built side-by-side keep the 10-share structure.
        let mut players = HashMap::new();
        players.insert(1, "A".to_string());
        players.insert(2, "B".to_string());
        players.insert(3, "C".to_string());
        players.insert(4, "D".to_string());
        let g1830 = BaseGame::build(vec![1, 2, 3, 4], players);
        assert_eq!(g1830.title, "1830");
        assert_eq!(g1830.corporations[0].num_share_units(), 10);
    }

    /// The per-title layout builds from the CLEAN default derivations and is
    /// internally consistent (this is what 1867 will get by default).
    #[test]
    fn test_title_layout_uses_clean_derivations() {
        let lo = crate::action_index::layout_for(TITLE);
        let title = crate::title::resolve(TITLE);

        // Clean defaults: map order / tile-counts order, not the 1830
        // frozen-artifact orders.
        let hex_defs: Vec<&str> = title.hex_definitions().iter().map(|d| d.coord).collect();
        assert_eq!(lo.hex_offsets, hex_defs);
        let tile_ids: Vec<&str> = title.tile_counts().iter().map(|(id, _)| *id).collect();
        assert_eq!(lo.tile_offsets, tile_ids);
        assert_ne!(
            lo.hex_offsets,
            crate::action_index::layout_for("1830").hex_offsets,
            "1830's frozen artifact order must differ from the clean default"
        );

        // Share-unit-derived: pool limit 50 / unit 20 = 2 sell slots.
        assert_eq!(lo.sell_count_slots, 2);
        // Same market, same par prices; same trains, same discount train.
        assert_eq!(lo.par_price_offsets, vec![67, 71, 76, 82, 90, 100]);
        assert_eq!(lo.discount_train, Some("D"));
        // A different total than 1830 (different sell block + city tables
        // ordering is irrelevant to the size, but the sell block shrinks).
        assert_ne!(lo.total, crate::action_index::layout_for("1830").total);
        // City tables: same map, so the same number of PlaceToken slots.
        assert_eq!(
            lo.city_offsets.len(),
            crate::action_index::layout_for("1830").city_offsets.len()
        );

        // The encoder spec derives per-title and the state encodes at the
        // spec's size.
        let spec = crate::encoder::spec_for(TITLE);
        assert_eq!(spec.num_hexes(), 93);
        let game = new_4p_game();
        let (gs, nf) = game.encode_state();
        assert_eq!(gs.len(), spec.encoding_size(4));
        assert_eq!(nf.len(), spec.num_hexes() * spec.num_node_features());
    }

    /// Total cash in the closed system (players + corporations + bank) must
    /// equal the title's bank cash at every step — the canary for share-unit
    /// money-math bugs.
    fn total_cash(game: &BaseGame) -> i64 {
        let p: i64 = game.players.iter().map(|p| p.cash as i64).sum();
        let c: i64 = game.corporations.iter().map(|c| c.cash as i64).sum();
        p + c + game.bank.cash as i64
    }

    /// Enumerate-apply self-consistency walks on the synthetic title: every
    /// enumerated action must index, build, and apply cleanly (the same
    /// contract the 1830 differential walk enforces), with cash conservation
    /// as a global invariant. This is the fuzzing leg of the per-title
    /// validation strategy, exercised before 1867 exists.
    fn run_walk(seed: u64, max_actions: usize) {
        let mut game = new_4p_game();
        let mut rng = StdRng::seed_from_u64(seed);
        let lo = crate::action_index::layout_for(TITLE);
        let company_offset = lo.action_offsets["CompanyBuyShares"];
        let bank_total = total_cash(&game);

        for step in 0..max_actions {
            if game.is_finished_pub() {
                break;
            }
            let choices = game.get_factored_choices_impl();
            if choices.is_empty() {
                break;
            }
            let applicable: Vec<&crate::factored::LegalAction> = choices
                .iter()
                .filter(|c| {
                    !(c.action_type == "PlaceToken"
                        && c.params.get("slot").map_or(false, |s| s.is_null()))
                })
                .collect();
            if applicable.is_empty() {
                break;
            }
            let la = applicable[rng.gen_range(0..applicable.len())];
            let Some(idx) = crate::action_index::legal_action_to_index_in(lo, la) else {
                panic!("seed {} step {}: unindexable {:?}", seed, step, la);
            };
            let price = la.price_range.map(|(lo, hi)| rng.gen_range(lo..=hi));
            let action = game
                .build_action(la, idx, idx >= company_offset, price)
                .unwrap_or_else(|e| panic!("seed {} step {}: build_action: {}", seed, step, e));
            game.process_action_internal(&action)
                .unwrap_or_else(|e| panic!("seed {} step {}: process {:?}: {}", seed, step, action, e));
            assert_eq!(
                total_cash(&game),
                bank_total,
                "seed {} step {}: cash leaked applying {:?}",
                seed,
                step,
                action
            );
        }
    }

    #[test]
    fn enumerate_apply_walks_on_synthetic_title() {
        for seed in 0..8u64 {
            run_walk(seed, 3000);
        }
    }

    /// Guard against the walk silently going vacuous: a fresh walk must get
    /// deep into the game (past the auction, through operating rounds), and
    /// the per-title flow hook must be in effect (OR sets of >= 2).
    #[test]
    fn walks_make_real_progress() {
        let mut game = new_4p_game();
        let mut rng = StdRng::seed_from_u64(0);
        let lo = crate::action_index::layout_for(TITLE);
        let company_offset = lo.action_offsets["CompanyBuyShares"];
        let mut saw_operating = false;
        let mut max_total_ors = 0u8;
        let mut moves = 0u32;
        for _ in 0..3000 {
            if game.is_finished_pub() {
                break;
            }
            if let crate::rounds::Round::Operating(s) = &game.round {
                saw_operating = true;
                max_total_ors = max_total_ors.max(s.total_ors);
            }
            let choices = game.get_factored_choices_impl();
            let applicable: Vec<&crate::factored::LegalAction> = choices
                .iter()
                .filter(|c| {
                    !(c.action_type == "PlaceToken"
                        && c.params.get("slot").map_or(false, |s| s.is_null()))
                })
                .collect();
            if applicable.is_empty() {
                break;
            }
            let la = applicable[rng.gen_range(0..applicable.len())];
            let idx = crate::action_index::legal_action_to_index_in(lo, la).unwrap();
            let price = la.price_range.map(|(lo, hi)| rng.gen_range(lo..=hi));
            let action = game.build_action(la, idx, idx >= company_offset, price).unwrap();
            game.process_action_internal(&action).unwrap();
            moves += 1;
        }
        assert!(saw_operating, "walk never reached an operating round");
        assert!(moves > 100, "walk stalled after {} moves", moves);
        // The per-title flow hook held: OR sets run >= 2 ORs even in phase 2
        // (1830's flow would have run 1).
        assert!(max_total_ors >= 2, "flow hook ignored: max total_ors {}", max_total_ors);
    }

    /// Float mechanics under the 5-share structure: parring costs 2 units,
    /// floating at 60% pays par x 5 into the treasury (full capitalization).
    #[test]
    fn five_share_float_pays_par_times_five() {
        let mut game = new_4p_game();
        let ci = game.corp_idx["PRR"];
        let par = game.stock_market.par_price(67).expect("par 67");
        game.corporations[ci].ipo_price = Some(par.clone());
        game.corporations[ci].share_price = Some(par.clone());
        let ipo = crate::entities::EntityId::ipo("PRR");
        let n = game.corporations[ci].shares.len();
        for i in 0..n {
            game.corporations[ci].set_share_owner(i, ipo.clone());
        }
        // President (40%) to player 1: 40 < 60 — not floated yet.
        game.corporations[ci].set_share_owner(0, crate::entities::EntityId::player(1));
        assert!(!game.corporations[ci].check_floated());
        // One 20% cert to player 2: 60% sold — floats.
        game.corporations[ci].set_share_owner(1, crate::entities::EntityId::player(2));
        assert!(game.corporations[ci].check_floated());
        let bank_before = game.bank.cash;
        game.check_float(ci);
        assert!(game.corporations[ci].floated);
        assert_eq!(game.corporations[ci].cash, 67 * 5);
        assert_eq!(game.bank.cash, bank_before - 67 * 5);
    }

    /// Per-title ability keying: both titles resolve the same syms through
    /// their own maps, and an unknown title resolves to nothing.
    #[test]
    fn ability_lookups_are_title_keyed() {
        assert!(crate::abilities::teleport("1830", "DH").is_some());
        assert!(crate::abilities::teleport(TITLE, "DH").is_some());
        assert!(crate::abilities::teleport("NO-SUCH-TITLE", "DH").is_none());
        assert_eq!(crate::abilities::company_syms(TITLE).len(), 6);
    }
}
