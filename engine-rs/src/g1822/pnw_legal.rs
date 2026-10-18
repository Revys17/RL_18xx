//! 1822PNW legal-move generation: what the 1822 generator (`legal.rs`)
//! does not know — the Merger round, P20's associated-minor choice, the
//! Assign step (P15 paper mill, P17 ski haus), builder cubes, the special
//! tile companies (P10 cubes, P12/P13 ports, P16 portage, P18 boom town,
//! P19 coal mine) and PNW's Track / SpecialTrack `available_hex` /
//! `potential_tiles`. Like the rest of the generator, every candidate is
//! applied to a clone before it is offered (PNW lays are never
//! pre-validated), so the process-side rules filter; the generator's own
//! filters are the UI's offers (Ruby `available_hex`, `potential_tiles`,
//! the Assign step's `available_hex`).

use crate::actions::Action;
use crate::game::BaseGame;

use super::legal::Move;
use super::tile_rules::RotRule;
use super::Ent;

/// The builder-cube "tile" (`cube_tile`: `tile_by_id('BC-0')`).
pub(crate) const CUBE_TILE: &str = "BC-0";

/// `tile.exits`: the edges its paths reach.
fn tile_exits(t: &crate::graph::Tile) -> Vec<u8> {
    let mut v: Vec<u8> = t.paths.iter().flat_map(|p| [p.a.edge_num(), p.b.edge_num()]).flatten().collect();
    v.sort_unstable();
    v.dedup();
    v
}

impl BaseGame {
    // =====================================================================
    // Merger round (step/merge.rb) and AcquireCompany's P20 choice
    // =====================================================================

    /// Merge#actions: `choose` mid-merge, else `merge` (one per mergeable
    /// minor) and `pass`.
    pub(crate) fn gen_pnw_merge(&mut self, ent: &Ent) -> Vec<Move> {
        let Ent::Corp(c) = ent else { return Vec::new() };
        let types = self.g1822_pnw_merge_actions(ent);
        let mut out = Vec::new();
        if types.contains(&"choose") {
            for choice in self.g1822_pnw_merge_choices() {
                out.push(Move::plain(Action::Choose { entity_id: c.clone(), choice }));
            }
        }
        if types.contains(&"merge") {
            for m in self.g1822_pnw_mergeable(c) {
                out.push(Move::plain(Action::Merge { entity_id: c.clone(), corporation_sym: m }));
            }
        }
        if types.contains(&"pass") {
            out.push(Move::plain(Action::Pass { entity_id: c.clone() }));
        }
        out
    }

    /// AcquireCompany after P20: `choose` one associated minor to replace.
    pub(crate) fn gen_pnw_p20_choices(&self, c: &str) -> Vec<Move> {
        self.rnd()
            .p20_choices
            .iter()
            .map(|t| Move::plain(Action::Choose { entity_id: c.to_string(), choice: t.clone() }))
            .collect()
    }

    // =====================================================================
    // Assign (P15 paper mill, P17 ski haus): Assign#available_hex
    // =====================================================================

    pub(crate) fn gen_pnw_assigns(&self, co: &str) -> Vec<Move> {
        let mut out = Vec::new();
        let nbrs = self.g1822_neighbors();
        for &h in Self::g1822_pnw_assign_hexes(co) {
            let Some(&hi) = self.hex_idx.get(h) else { continue };
            if self.g1822_pnw_coal_hex(h) {
                continue;
            }
            let ok = match co {
                // a city whose track meets a forest-assigned neighbour's
                "P15" => {
                    !self.g1822_pnw_assigned(h, "P17")
                        && tile_exits(&self.hexes[hi].tile).iter().any(|&e| {
                            nbrs.get(&hi).and_then(|m| m.get(&e)).map_or(false, |&n| {
                                tile_exits(&self.hexes[n].tile).contains(&((e + 3) % 6))
                                    && self.g1822_pnw_assigned(&self.hexes[n].id, "forest")
                            })
                        })
                }
                "P17" => !self.g1822_pnw_assigned(h, "P15") && !self.g1822_pnw_assigned(h, "P17"),
                _ => false,
            };
            if ok {
                out.push(Move::plain(Action::Assign {
                    entity_id: co.to_string(),
                    target: h.to_string(),
                    target_type: "hex".to_string(),
                }));
            }
        }
        out
    }

    // =====================================================================
    // Track step lays (G1822PNW::Step::Track)
    // =====================================================================

    /// Whether the hex's preprinted tile carries a mountain icon
    /// (`hex.tile.icons.any? { name.start_with?('mountain') }`).
    pub(crate) fn pnw_mountain_icon(&self, hi: usize) -> bool {
        let hex = &self.hexes[hi];
        hex.tile.id.starts_with("preprinted_")
            && (self.fam().hex_dsl)(&hex.id).map_or(false, |(d, _)| d.contains("icon=image:mountain"))
    }

    /// PNW potential_tiles additions/filters on top of the 1822 list:
    /// PNW5 over the coal mine, the token cities' sequence rule.
    pub(crate) fn pnw_potential_tiles(&self, hi: usize, mut tiles: Vec<String>) -> Vec<String> {
        let hex = &self.hexes[hi];
        let cur = hex.tile.name.split('-').next().unwrap_or("");
        if cur == "PNW4" && !tiles.iter().any(|t| t == "PNW5") && self.tile_counts_remaining.get("PNW5").copied().unwrap_or(0) > 0 {
            tiles.push("PNW5".to_string());
        }
        if Self::g1822_pnw_tokencity(&hex.id) {
            // tokencity_potential_tiles: the first of the sequence, or the
            // second when the other token city already shows its colour
            if tiles.is_empty() {
                return tiles;
            }
            let color0 = self.tile_catalog.get(&tiles[0]).map(|d| d.color);
            let other_same = ["O8", "H11"].iter().filter(|&&h| h != hex.id).any(|h| {
                self.hex_idx.get(*h).map_or(false, |&o| Some(self.hexes[o].tile.color) == color0)
            });
            return if other_same { tiles.get(1).cloned().into_iter().collect() } else { vec![tiles[0].clone()] };
        }
        tiles
    }

    /// The Track step's lays for a PNW corporation (Track#available_hex:
    /// cube-holding connected hexes are available, mountain-icon hexes
    /// are not; the cube is a candidate wherever a cube fits).
    pub(crate) fn gen_pnw_corp_lays(&mut self, c: &str) -> Vec<Move> {
        let graph = self.g1822_graph(c);
        let nbrs = self.g1822_neighbors();
        let Some((lay, upgrade, no_reuse)) = self.g1822_tile_lay(c, false) else { return Vec::new() };
        let mut out = Vec::new();
        for (&hi, edges) in &graph.connected_hexes {
            let cubes = self.g1822_pnw_can_hold_cubes(hi);
            if cubes && lay {
                out.push(self.pnw_cube_move(c, hi));
            }
            if !cubes && self.pnw_mountain_icon(hi) {
                continue;
            }
            // lay_tile: 'Cannot upgrade forests'
            if self.g1822_pnw_assigned(&self.hexes[hi].id, "forest") {
                continue;
            }
            let white = self.hexes[hi].tile.color == crate::tiles::TileColor::White;
            if (white && !lay) || (!white && !upgrade) {
                continue;
            }
            if !white && no_reuse && self.rnd().laid_hexes.contains(&self.hexes[hi].id) {
                continue;
            }
            let tiles = self.g1822_potential_tiles(c, hi);
            let tiles = self.pnw_potential_tiles(hi, tiles);
            out.extend(self.lays_for(c, c, hi, tiles, edges, &nbrs, RotRule::Normal, &[], true, &graph.connected_nodes));
        }
        out
    }

    fn pnw_cube_move(&self, entity: &str, hi: usize) -> Move {
        Move::plain(Action::LayTile {
            entity_id: entity.to_string(),
            hex_id: self.hexes[hi].id.clone(),
            tile_id: CUBE_TILE.to_string(),
            rotation: 0,
        })
    }

    // =====================================================================
    // SpecialTrack company lays (G1822PNW::Step::SpecialTrack)
    // =====================================================================

    /// The PNW special companies' lays (None: the 1822 generator's path,
    /// with [`BaseGame::pnw_company_extras`] added).
    pub(crate) fn gen_pnw_company_lays(&mut self, co: &str) -> Option<Vec<Move>> {
        if !matches!(co, "P10" | "P12" | "P13" | "P16" | "P18" | "P19") {
            return None;
        }
        let mut out = Vec::new();
        let Some((_, ab)) = self.g1822_track_ability(co) else { return Some(out) };
        let Some(owner) = self.co_owner(co).corp_sym().map(str::to_string) else { return Some(out) };
        let nbrs = self.g1822_neighbors();
        let all_nb = |g: &BaseGame, hi: usize| -> Vec<u8> {
            g.hex_adjacency.get(&g.hexes[hi].id).map(|m| m.keys().copied().collect()).unwrap_or_default()
        };
        let graph = self.g1822_graph(&owner);
        let lay_tiles = |g: &mut BaseGame, hi: usize, rule: RotRule, tiles: &[&str], connected: &[u8]| -> Vec<Move> {
            let names: Vec<String> = tiles
                .iter()
                .filter(|t| g.tile_counts_remaining.get(**t).copied().unwrap_or(0) > 0)
                .map(|t| t.to_string())
                .collect();
            let nodes = std::collections::HashSet::new();
            g.lays_for(co, &owner, hi, names, connected, &nbrs, rule, &[], false, &nodes)
        };
        match co {
            // cube company: a cube on a cube-holding hex the owner reaches
            "P10" => {
                for &hi in graph.connected_hexes.keys() {
                    if self.g1822_pnw_can_hold_cubes(hi) {
                        out.push(self.pnw_cube_move(co, hi));
                    }
                }
            }
            // ports: a blue ability hex without a port tile yet
            "P12" | "P13" => {
                for h in &ab.hexes {
                    let Some(&hi) = self.hex_idx.get(h) else { continue };
                    let id = &self.hexes[hi].tile.id;
                    if id == "P1-0" || id == "P2-0" || !self.g1822_hex_is_blue(h) || !id.starts_with("preprinted_") {
                        continue;
                    }
                    let exits = all_nb(self, hi);
                    let tiles: Vec<&str> = ab.tiles.iter().map(String::as_str).collect();
                    out.extend(lay_tiles(self, hi, RotRule::PnwPort, &tiles, &exits));
                }
            }
            // portage: an ability hex that is not a port tile
            "P16" => {
                for h in &ab.hexes {
                    let Some(&hi) = self.hex_idx.get(h) else { continue };
                    let n = self.hexes[hi].tile.name.split('-').next().unwrap_or("");
                    if matches!(n, "P1" | "P2") {
                        continue;
                    }
                    let exits = all_nb(self, hi);
                    let tiles: Vec<&str> = ab.tiles.iter().map(String::as_str).collect();
                    out.extend(lay_tiles(self, hi, RotRule::PnwPortage, &tiles, &exits));
                }
            }
            // boom town: any plain yellow 7 / 8 / 9
            "P18" => {
                let his: Vec<usize> = (0..self.hexes.len())
                    .filter(|&hi| matches!(self.hexes[hi].tile.name.split('-').next(), Some("7" | "8" | "9")))
                    .collect();
                for hi in his {
                    let exits = all_nb(self, hi);
                    let tiles: Vec<&str> = ab.tiles.iter().map(String::as_str).collect();
                    out.extend(lay_tiles(self, hi, RotRule::PnwBoomtown, &tiles, &exits));
                }
            }
            // coal mine: an ability hex the owner reaches (not the mill or
            // ski haus hex)
            "P19" => {
                for h in &ab.hexes {
                    let Some(&hi) = self.hex_idx.get(h) else { continue };
                    if self.g1822_pnw_assigned(h, "P15") || self.g1822_pnw_assigned(h, "P17") {
                        continue;
                    }
                    if !graph.connected_hexes.contains_key(&hi) {
                        continue;
                    }
                    let exits = all_nb(self, hi);
                    let tiles: Vec<&str> = ab.tiles.iter().map(String::as_str).collect();
                    out.extend(lay_tiles(self, hi, RotRule::PnwCoal, &tiles, &exits));
                }
            }
            _ => {}
        }
        Some(out)
    }

    /// The other companies' extras on top of the 1822 company lays: a cube
    /// for an extra-track company (P11) on a cube-holding hex the owner
    /// reaches (SpecialTrack#potential_tiles adds the cube tile).
    pub(crate) fn pnw_company_extras(&self, co: &str) -> Vec<Move> {
        if !self.g1822_company_ability_extra_track(co) || self.g1822_track_ability(co).is_none() {
            return Vec::new();
        }
        let Some(owner) = self.co_owner(co).corp_sym().map(str::to_string) else { return Vec::new() };
        let graph = self.g1822_graph(&owner);
        graph
            .connected_hexes
            .keys()
            .filter(|&&hi| self.g1822_pnw_can_hold_cubes(hi))
            .map(|&hi| self.pnw_cube_move(co, hi))
            .collect()
    }

    /// Whether exchanging the associated minor for `n` shares works out:
    /// the player keeps the cash their open bids commit, and when the
    /// shares are worth less than the minor the major (holding the minor's
    /// cash) can pay the difference — Ruby raises otherwise. `at`: (par,
    /// minor value, minor cash); None: the conversion in progress.
    pub(crate) fn pnw_exchange_affordable(&self, pid: u32, n: i32, at: Option<(i32, i32, i32)>) -> bool {
        let (par, value, cash) = match at {
            Some(v) => v,
            None => {
                let Some(major) = self.rnd().converting_major.clone() else { return true };
                let Some(minor) = self.g1822_pnw_assoc_minor(&major) else { return true };
                let m = &self.corporations[self.ci(&minor)];
                (self.rnd().convert_par, m.share_price.as_ref().map_or(0, |sp| sp.price) * 2, m.cash)
            }
        };
        let shares_value = n * par;
        let pay = (shares_value - value).max(0);
        self.player_cash(pid) - pay >= self.g1822_committed_cash(pid) && (shares_value >= value || cash >= value - shares_value)
    }
}
