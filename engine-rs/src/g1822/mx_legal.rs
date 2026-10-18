//! 1822MX legal-move generation: what the 1822 generator (`legal.rs`) does
//! not know — the NdeM's token auction (`AuctionNdemTokens`) and cash-out,
//! P16's close_p16, builder cubes (the Track step's cube lays, P10/P11's
//! cubes, P9's extra-lay cube), P17/P18's port tiles and MX's Track /
//! SpecialTrack `available_hex` / `potential_tiles`. Every candidate the
//! generator does not pre-validate is applied to a clone before it is
//! offered (legal.rs), so the process-side rules filter; the generator's
//! own filters are the UI's offers.

use crate::actions::Action;
use crate::game::BaseGame;

use super::legal::{Move, PriceRange};
use super::mx::NDEM;
use super::tile_rules::RotRule;
use super::Ent;

/// The builder-cube "tile".
const CUBE_TILE: &str = "BC-0";

impl BaseGame {
    // =====================================================================
    // AuctionNdemTokens / CashOutNdem
    // =====================================================================

    /// AuctionNdemTokens: the chooser's tokens up for auction (or pass),
    /// the bidder's bids (a range; pass once someone bid), the winner's
    /// buying corporation. When Ruby's `auto_actions` would pass, only the
    /// pass.
    pub(crate) fn gen_mx_auction(&self, ent: &Ent) -> Vec<Move> {
        let mut out = Vec::new();
        let Ent::Corp(c) = ent else { return out };
        if c != NDEM {
            return out;
        }
        let pass = || Move::plain(Action::Pass { entity_id: NDEM.to_string() });
        if self.g1822_mx_auction_auto_pass() {
            out.push(pass());
            return out;
        }
        let Some(a) = self.rnd().ndem_auction.clone() else { return out };
        if let Some(w) = a.winner {
            for corp in self.g1822_mx_mergeable(w) {
                out.push(Move::plain(Action::Merge { entity_id: NDEM.to_string(), corporation_sym: corp }));
            }
            return out;
        }
        match a.up {
            None => {
                let Some(&p) = a.remaining_choosers.first() else { return out };
                for t in &a.available {
                    if !self.g1822_mx_player_can_purchase(p, t) {
                        continue;
                    }
                    let Some(&hi) = self.hex_idx.get(t.hex.as_str()) else { continue };
                    let slot = self.hexes[hi].tile.cities[t.city]
                        .tokens
                        .iter()
                        .position(|x| x.as_ref().is_some_and(|k| k.corporation_id == NDEM))
                        .unwrap_or(0);
                    out.push(Move::plain(Action::RemoveToken {
                        entity_id: NDEM.to_string(),
                        hex_id: t.hex.clone(),
                        city_index: t.city as u8,
                        slot: slot as u8,
                    }));
                }
                out.push(pass());
            }
            Some(u) => {
                let Some(&bidder) = a.remaining_bidders.first() else { return out };
                let lo = if a.high_bid == 0 { a.available[u].cost } else { a.high_bid + 10 };
                let max = self.g1822_mx_max_bid(bidder);
                // bids are whole multiples of 10 (`price % 10`)
                let lo = (lo + 9).div_euclid(10) * 10;
                let hi = max.div_euclid(10) * 10;
                if lo <= hi {
                    let mut m = Move::plain(Action::Bid { entity_id: NDEM.to_string(), company_sym: String::new(), price: lo });
                    m.price = Some(PriceRange { lo, hi, step: 10 });
                    out.push(m);
                }
                if a.high_bidder.is_some() {
                    out.push(pass());
                }
            }
        }
        out
    }

    // =====================================================================
    // Track (G1822MX::Step::Track)
    // =====================================================================

    /// The Track step's lays for an MX corporation (the NdeM's are its
    /// current tile layer's): Track#available_hex makes a cube-holding
    /// connected hex available whatever the lay slot, and potential_tiles
    /// adds the cube there.
    pub(crate) fn gen_mx_corp_lays(&mut self, c: &str) -> Vec<Move> {
        let graph = self.g1822_graph(c);
        let nbrs = self.g1822_neighbors();
        let Some((lay, upgrade, no_reuse)) = self.g1822_tile_lay(c, false) else { return Vec::new() };
        let mut out = Vec::new();
        let mut his: Vec<usize> = graph.connected_hexes.keys().copied().collect();
        his.sort_unstable();
        for hi in his {
            let edges = graph.connected_hexes[&hi].clone();
            if self.g1822_mx_can_hold_cubes(hi) && lay {
                out.push(self.mx_cube_move(c, hi));
            }
            let white = self.hexes[hi].tile.color == crate::tiles::TileColor::White;
            if (white && !lay) || (!white && !upgrade) {
                continue;
            }
            if !white && no_reuse && self.rnd().laid_hexes.contains(&self.hexes[hi].id) {
                continue;
            }
            let tiles = self.g1822_potential_tiles(c, hi);
            out.extend(self.lays_for(c, c, hi, tiles, &edges, &nbrs, RotRule::Normal, &[], true, &graph.connected_nodes));
        }
        out
    }

    fn mx_cube_move(&self, entity: &str, hi: usize) -> Move {
        Move::plain(Action::LayTile {
            entity_id: entity.to_string(),
            hex_id: self.hexes[hi].id.clone(),
            tile_id: CUBE_TILE.to_string(),
            rotation: 0,
        })
    }

    // =====================================================================
    // SpecialTrack (G1822MX::Step::SpecialTrack)
    // =====================================================================

    /// The MX companies' own lays (None: the 1822 generator's path, with
    /// [`BaseGame::mx_company_extras`] added): P10/P11 place a cube on a
    /// cube-holding hex the owner reaches; P17/P18 lay their port tile on a
    /// blue spike without one (no route needed), turned to the spike.
    pub(crate) fn gen_mx_company_lays(&mut self, co: &str) -> Option<Vec<Move>> {
        let cube = Self::g1822_mx_cube_company(co);
        let port = Self::g1822_mx_port_company(co);
        if !cube && !port {
            return None;
        }
        let mut out = Vec::new();
        let Some((_, ab)) = self.g1822_track_ability(co) else { return Some(out) };
        let Some(owner) = self.co_owner(co).corp_sym().map(str::to_string) else { return Some(out) };
        if cube {
            let graph = self.g1822_graph(&owner);
            let mut his: Vec<usize> = graph.connected_hexes.keys().copied().collect();
            his.sort_unstable();
            for hi in his {
                if self.g1822_mx_can_hold_cubes(hi) {
                    out.push(self.mx_cube_move(co, hi));
                }
            }
            return Some(out);
        }
        let nbrs = self.g1822_neighbors();
        for h in &ab.hexes {
            let Some(&hi) = self.hex_idx.get(h) else { continue };
            let id = &self.hexes[hi].tile.id;
            if super::mx::PORT_TILES.contains(&id.as_str()) || self.g1822_tile_color_name(hi) != "blue" {
                continue;
            }
            let names: Vec<String> = ab
                .tiles
                .iter()
                .filter(|t| self.tile_counts_remaining.get(t.as_str()).copied().unwrap_or(0) > 0)
                .cloned()
                .collect();
            let exits: Vec<u8> = self.hex_adjacency.get(h).map(|m| m.keys().copied().collect()).unwrap_or_default();
            let nodes = std::collections::HashSet::new();
            out.extend(self.lays_for(co, &owner, hi, names, &exits, &nbrs, RotRule::MxPort, &[], false, &nodes));
        }
        Some(out)
    }

    /// P9 (extra tile lay) may spend its lay on a cube instead: a cube on
    /// a cube-holding hex its owner reaches (SpecialTrack#potential_tiles
    /// adds the cube tile; only an extra-track company's cube is legal).
    pub(crate) fn mx_company_extras(&self, co: &str) -> Vec<Move> {
        if !self.g1822_company_ability_extra_track(co)
            || Self::g1822_mx_port_company(co)
            || self.g1822_track_ability(co).is_none()
        {
            return Vec::new();
        }
        let Some(owner) = self.co_owner(co).corp_sym().map(str::to_string) else { return Vec::new() };
        let graph = self.g1822_graph(&owner);
        let mut his: Vec<usize> = graph.connected_hexes.keys().copied().collect();
        his.sort_unstable();
        his.into_iter()
            .filter(|&hi| self.g1822_mx_can_hold_cubes(hi))
            .map(|hi| self.mx_cube_move(co, hi))
            .collect()
    }

    // =====================================================================
    // SpecialChoose: P16's close_p16
    // =====================================================================

    pub(crate) fn gen_mx_special_choose(&self, co: &str) -> Vec<Move> {
        if co != "P16" || !self.g1822_mx_p16_usable() {
            return Vec::new();
        }
        vec![Move::plain(Action::ChooseAbility { entity_id: "P16".to_string(), choice: "close_p16".to_string() })]
    }
}
