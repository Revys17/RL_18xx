//! 1822MX (The Railways of Mexico): the rules Ruby's `G1822MX::Game`
//! overrides (lib/engine/game/g_1822_mx/), on top of the G1822 step machine.
//!
//! * The NdeM (Ferrocarriles Nacionales de Mexico): a presidentless major
//!   floated at setup (par 100, its shares in the pool, outside the cert
//!   limit) that takes over a minor's home (the first of M14/M15/M17 in the
//!   shuffled stack, then each first-bid-box minor nobody bids on), gets a
//!   pool share for each, receives the L trains exported by unbid minors,
//!   operates last (its shareholders lay its track in turn, its revenue
//!   pays out automatically), and at the first 7-train is privatized:
//!   it runs once more right away, its shares are cashed out
//!   (`CashOutNdem`) and its tokens are auctioned to the majors
//!   (`AuctionNdemTokens`), then it closes.
//! * C1 (the FCM concession) comes with minor 18, floated on purchase.
//! * P7 double cash (G1822's double-cash code, renamed), P16 stock drop
//!   (`close_p16`), P10/P11 builder cubes, P17/P18 ports, P12/P13 town
//!   removal, P8 advanced tile lay, the 3/2P train, M22's Mexico City
//!   connection rule, the one-row market's sell movement.
//!
//! Each function names the Ruby method it ports.

use serde::{Deserialize, Serialize};

use crate::actions::{Action, GameError};
use crate::entities::{EntityId, Share, Token, Train};
use crate::game::BaseGame;

use super::shares::{Bundle, Dir};
use super::{err, Ent};

pub const NDEM: &str = "NDEM";

/// The minors whose earliest stack position becomes the NdeM's first home.
const NDEM_FIRST_MINORS: &[&str] = &["M14", "M15", "M17"];

/// MinorAcquisition#connected_to_port?'s port hexes.
pub const PORT_HEXES: &[&str] = &["F7", "G8", "H9", "I10", "J11"];

/// SpecialTrack::PORT_TILES.
pub const PORT_TILES: &[&str] = &["P1-0", "P2-0"];

/// M22's neighbours by exit (`@m22_adjacent_hexes`).
const M22_ADJACENT: &[(u8, &str)] = &[(0, "N21"), (1, "M20"), (2, "L21"), (3, "L23"), (4, "M24")];

/// `@ndem_state`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum NdemState {
    #[default]
    Open,
    Closing,
    Closed,
}

/// One of the NdeM's tokens up for auction (`@available_ndem_tokens`):
/// where it sits and what it costs (`ndem_token_cost` at the step's
/// setup, raised to the high bid + 10 by each bid).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NdemTok {
    pub hex: String,
    pub city: usize,
    pub cost: i32,
}

/// AuctionNdemTokens' state (set up when the closing NdeM starts its turn).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct NdemAuction {
    pub available: Vec<NdemTok>,
    pub player_step_order: Vec<u32>,
    pub remaining_choosers: Vec<u32>,
    /// Index into `available` of the token up for bid.
    pub up: Option<usize>,
    pub remaining_bidders: Vec<u32>,
    pub high_bid: i32,
    pub high_bidder: Option<u32>,
    pub winner: Option<u32>,
}

impl BaseGame {
    pub(crate) fn g1822_is_mx(&self) -> bool {
        self.g1822.is_some() && self.fam().variant == super::family::Variant::Mx
    }

    /// `ndem` (the corporation), if the game has one.
    pub(crate) fn g1822_mx_is_ndem(&self, c: &str) -> bool {
        c == NDEM && self.g1822_is_mx()
    }

    // =====================================================================
    // Setup (game.rb setup_companies, setup_ndem)
    // =====================================================================

    /// MX `setup_companies`'s stack order: minors (M18 last), concessions
    /// (C1 first), privates (P1 first), each in shuffled order.
    pub(crate) fn g1822_mx_order_companies(shuffled: &[String]) -> Vec<String> {
        let by_prefix = |p: char| -> Vec<String> { shuffled.iter().filter(|s| s.starts_with(p)).cloned().collect() };
        let mut minors = by_prefix('M');
        let mut concessions = by_prefix('C');
        let mut privates = by_prefix('P');
        for (list, first) in [(&mut concessions, "C1"), (&mut privates, "P1")] {
            if let Some(pos) = list.iter().position(|s| s == first) {
                let s = list.remove(pos);
                list.insert(0, s);
            }
        }
        if let Some(pos) = minors.iter().position(|s| s == "M18") {
            let s = minors.remove(pos);
            minors.push(s);
        }
        let mut out = minors;
        out.extend(concessions);
        out.extend(privates);
        out
    }

    /// MX `setup_ndem` (setup_game_specific): the NdeM's shares go to the
    /// pool outside the cert limit, the earliest of M14/M15/M17 in the
    /// stack is replaced by the NdeM, the bid boxes are set up again, and
    /// the NdeM pars at 100.
    pub(crate) fn g1822_mx_setup_ndem(&mut self) {
        self.gm().ndem_shares = 3;
        self.gm().ndem_state = NdemState::Open;
        let ci = self.ci(NDEM);
        let n = self.corporations[ci].shares.len();
        for i in 0..n {
            self.corporations[ci].set_share_owner(i, EntityId::market());
        }
        let first = self
            .g()
            .company_order
            .iter()
            .enumerate()
            .filter(|(_, c)| NDEM_FIRST_MINORS.contains(&c.as_str()))
            .map(|(i, _)| i)
            .min()
            .map(|i| self.g().company_order[i].clone());
        if let Some(company) = first {
            self.g1822_mx_replace_minor_with_ndem(&company);
        }
        self.setup_bidboxes();
        let sp = self
            .stock_market
            .share_prices_with_types(&["par", "par_1"])
            .into_iter()
            .find(|p| p.price == 100)
            .expect("a 100 par price");
        self.g1822_par_corporation(NDEM, sp);
        let ci = self.ci(NDEM);
        // ndem.owner = @share_pool; floated? holds (its treasury is empty)
        self.corporations[ci].owner_id = EntityId::market();
        self.corporations[ci].floated = true;
    }

    /// MX `replace_minor_with_ndem(company)`: the minor's company leaves the
    /// stacks (deleted, not closed), its home reservation goes, an NdeM
    /// token takes the city unless the NdeM is already there, and a new 10%
    /// NdeM share enters the pool.
    pub(crate) fn g1822_mx_replace_minor_with_ndem(&mut self, company: &str) {
        self.delete_company(company);
        let minor = Self::minor_of_company(company);
        let cd = self.title_def().corporations().into_iter().find(|d| d.sym == minor);
        if let Some(cd) = cd {
            let hex = cd.home_hex.to_string();
            if let Some(&hi) = self.hex_idx.get(hex.as_str()) {
                let n = self.hexes[hi].tile.cities.len();
                let city = (0..n).find(|&c| self.g1822_city_res(&hex, c).iter().any(|r| r.as_deref() == Some(minor.as_str())));
                if let Some(c) = city {
                    // city.remove_reservation!(corporation)
                    if let Some(v) = self.gm().city_reservations.get_mut(&format!("{hex}#{c}")) {
                        for r in v.iter_mut() {
                            if r.as_deref() == Some(minor.as_str()) {
                                *r = None;
                            }
                        }
                    }
                    if !self.g1822_city_tokened_by(hi, c, NDEM) {
                        if let Some(t) = self.g1822_find_token(NDEM, "normal") {
                            let _ = self.g1822_put_token(NDEM, t, hi, c, false);
                            self.g1822_ruby_graph_clear();
                        }
                    }
                }
            }
        }
        // a new 10% share (index @number_ndem_shares), into the pool
        let ci = self.ci(NDEM);
        let index = self.corporations[ci].shares.len();
        self.corporations[ci].shares.push(Share {
            corporation_id: NDEM.to_string(),
            percent: 10,
            president: false,
            index,
            owner: EntityId::none(),
            acquired_seq: 0,
        });
        self.corporations[ci].set_share_owner(index, EntityId::market());
        self.gm().ndem_shares += 1;
    }

    /// MX `send_train_to_ndem(train)`: free to the NdeM (an L in phase 2
    /// flips to its 2 side), the phase sees the purchase, and the NdeM's
    /// oldest trains rust while it holds more than its limit.
    pub(crate) fn g1822_mx_send_train_to_ndem(&mut self, mut train: Train) {
        if train.name == "L" && self.phase.name == "2" {
            if let Some((name, dist, price, _)) = (self.fam().train_variant)("L", "2") {
                train.name = name.to_string();
                train.distance = dist;
                train.price = price;
            }
        }
        train.owner = EntityId::corporation(NDEM);
        train.operated = false;
        let ci = self.ci(NDEM);
        self.corporations[ci].trains.push(train.clone());
        self.g1822_phase_buying_train(&train, None);
        loop {
            let ci = self.ci(NDEM);
            let limit = self.g1822_train_limit(NDEM) as usize;
            if self.corporations[ci].trains.len() <= limit {
                break;
            }
            // rust(t): gone for good
            self.corporations[ci].trains.remove(0);
        }
    }

    // =====================================================================
    // Stock round (round/stock.rb)
    // =====================================================================

    /// MX Round::Stock#buy_company's C1 part: minor 18 floats for the
    /// buyer (par 50, £100 from the bank, the 100% certificate handed
    /// over); its company stays with the player outside their companies.
    pub(crate) fn g1822_mx_after_buy_company(&mut self, pid: u32, company: &str) {
        if company != "C1" {
            return;
        }
        if let Some(i) = self.co_idx("M18") {
            self.companies[i].owner = EntityId::player(pid);
            self.gm().bank_owned.retain(|c| c != "M18");
            if !self.g().owner_only.iter().any(|c| c == "M18") {
                self.gm().owner_only.push("M18".to_string());
            }
        }
        let minor = "18";
        let sp = self
            .stock_market
            .share_prices_with_types(&["par", "par_1"])
            .into_iter()
            .find(|p| p.price == 50)
            .expect("a 50 par price");
        self.g1822_par_corporation(minor, sp);
        self.g1822_bank_pay(&EntityId::corporation(minor), 100);
        let b = Bundle { corp: minor.to_string(), shares: vec![0], percent: 100, price_override: None };
        let was_floated = self.corporations[self.ci(minor)].floated;
        self.g1822_transfer_shares(&b, &EntityId::player(pid), true);
        self.g1822_check_float(minor, was_floated);
    }

    /// MX `remove_l_trains(count)`: the L trains go to the NdeM.
    pub(crate) fn g1822_mx_remove_l_trains(&mut self, mut count: u32) {
        while count > 0 && self.depot.trains.first().map_or(false, |t| t.name == "L") {
            let t = self.depot.trains.remove(0);
            self.g1822_mx_send_train_to_ndem(t);
            count -= 1;
        }
    }

    /// MX `remove_minor_and_first_train(company)`: while the NdeM is open
    /// the next train goes to it and it takes the minor's place; afterwards
    /// the minor just closes (its company deleted, not closed).
    pub(crate) fn g1822_mx_remove_minor_and_first_train(&mut self, company: &str) {
        if self.g().ndem_state == NdemState::Open {
            if !self.depot.trains.is_empty() {
                let t = self.depot.trains.remove(0);
                self.g1822_mx_send_train_to_ndem(t);
            }
            self.g1822_mx_replace_minor_with_ndem(company);
        } else {
            let minor = Self::minor_of_company(company);
            self.g1822_close_corporation(&minor);
            self.delete_company(company);
        }
    }

    // =====================================================================
    // NdeM shares (share_pool.rb, step/buy_sell_par_shares.rb, game.rb)
    // =====================================================================

    /// `corporation.counts_for_limit` / `share.counts_for_limit`: the NdeM
    /// and its shares never count.
    pub(crate) fn g1822_mx_counts_for_limit(&self, sym: &str) -> bool {
        !self.g1822_mx_is_ndem(sym)
    }

    /// MX `sell_shares_and_change_price`: the NdeM's price never moves on
    /// sales (and no presidency changes); other corporations move left per
    /// 10% sold by their president, else one (2 players: left per share
    /// sold by the president).
    pub(crate) fn g1822_mx_sell_shares_and_change_price(&mut self, b: &Bundle) {
        if self.g1822_mx_is_ndem(&b.corp) {
            self.g1822_pool_sell_shares(b, false);
            return;
        }
        let ci = self.ci(&b.corp);
        let owner = self.corporations[ci].shares[b.shares[0]].owner.clone();
        let was_president = self.corporations[ci].owner_id == owner && !owner.is_none();
        let spaces = if self.players.len() == 2 {
            if was_president {
                let unit = self.corporations[ci].share_unit_percent as u32;
                (b.percent as u32).div_ceil(unit)
            } else {
                0
            }
        } else if was_president {
            b.percent as u32 / 10
        } else {
            1
        };
        self.g1822_pool_sell_shares(b, true);
        for _ in 0..spaces {
            self.g1822_move_price(&b.corp, Dir::Left);
        }
    }

    /// MX BSPS#process_buy_shares for the NdeM: no presidency, ends the
    /// turn like any buy (counts as something sold).
    pub(crate) fn g1822_mx_is_ndem_bundle(&self, b: &Bundle) -> bool {
        self.g1822_mx_is_ndem(&b.corp)
    }

    // =====================================================================
    // Operating round
    // =====================================================================

    /// MX `operating_order`: the NdeM last while open, first while closing,
    /// gone once closed.
    pub(crate) fn g1822_mx_operating_order(&self, order: Vec<String>) -> Vec<String> {
        let (ndem, others): (Vec<String>, Vec<String>) = order.into_iter().partition(|c| c == NDEM);
        match self.g().ndem_state {
            NdemState::Closing => ndem.into_iter().chain(others).collect(),
            NdemState::Closed => others,
            NdemState::Open => others.into_iter().chain(ndem).collect(),
        }
    }

    /// MX `event_close_ndem!`: the NdeM is privatized; during an OR it
    /// moves from the end of the order to right after the current entity.
    pub(crate) fn g1822_mx_event_close_ndem(&mut self) {
        self.gm().ndem_state = NdemState::Closing;
        if self.rnd().kind != super::RoundKind::Operating {
            return;
        }
        let r = self.rndm();
        if let Some(last) = r.entities.pop() {
            let at = (r.entity_index + 1).min(r.entities.len());
            r.entities.insert(at, last);
        }
    }

    /// MX `payout_companies` after G1822's: a corporation owning P16 pays
    /// the bank £10, or P16 closes and its owner drops a space.
    pub(crate) fn g1822_mx_p16_payment(&mut self) {
        let owner = self.co_owner("P16");
        let Some(c) = owner.corp_sym().map(str::to_string) else { return };
        if self.co_closed("P16") {
            return;
        }
        let ci = self.ci(&c);
        if self.corporations[ci].cash >= 10 {
            self.corporations[ci].cash -= 10;
            self.g1822_bank_receive(10);
        } else {
            self.g1822_mx_close_p16();
        }
    }

    /// P16's choose_ability (`when: owning_corp_or_turn`): open, owned by
    /// the operating corporation.
    pub(crate) fn g1822_mx_p16_usable(&self) -> bool {
        !self.co_closed("P16")
            && self.rnd().kind == super::RoundKind::Operating
            && self.co_owner("P16").corp_sym().is_some()
            && self.co_owner("P16").corp_sym() == self.g1822_operator().as_deref()
    }

    /// MX `close_p16`: the owner's share price moves left, P16 closes.
    pub(crate) fn g1822_mx_close_p16(&mut self) {
        if let Some(c) = self.co_owner("P16").corp_sym().map(str::to_string) {
            self.g1822_move_price(&c, Dir::Left);
        }
        self.close_company("P16");
    }

    /// MX `company_bought`: the train privates flip, P7 pays its corporate
    /// revenue, P16 costs £10 a round, the cube and mail companies stop
    /// paying.
    pub(crate) fn g1822_mx_company_bought(&mut self, company: &str) {
        let Some(i) = self.co_idx(company) else { return };
        match company {
            "P7" => self.companies[i].revenue = self.fam().company_double_cash_revenue[self.phase_num() as usize],
            "P16" => self.companies[i].revenue = -10,
            "P10" | "P11" | "P14" | "P15" => self.companies[i].revenue = 0,
            _ => {}
        }
    }

    /// The NdeM's acting player outside its own steps (`acting_for_entity`):
    /// the shareholder in priority order, else the priority player.
    pub(crate) fn g1822_mx_ndem_shareholder_player(&self) -> Option<u32> {
        let ci = self.ci(NDEM);
        self.players
            .iter()
            .find(|p| self.corporations[ci].percent_owned_by(&EntityId::player(p.id)) > 0)
            .or_else(|| self.players.first())
            .map(|p| p.id)
    }

    /// The players holding NdeM shares, in player order (Track#setup's
    /// `@ndem_tile_layers`; CashOutNdem#players_hold_shares?).
    pub(crate) fn g1822_mx_ndem_holders(&self) -> Vec<u32> {
        let ci = self.ci(NDEM);
        self.players
            .iter()
            .filter(|p| self.corporations[ci].percent_owned_by(&EntityId::player(p.id)) > 0)
            .map(|p| p.id)
            .collect()
    }

    /// The acting player of the NdeM's current decision (Ruby
    /// `active_players` / `acting_for_entity` for the NdeM): the tile layer
    /// in the Track step, the auction's chooser / bidder / winner, else the
    /// first shareholder in priority (the priority player if none).
    pub(crate) fn g1822_mx_ndem_acting_player(&self) -> Option<u32> {
        if let Some(a) = &self.rnd().ndem_auction {
            if let Some(w) = a.winner {
                return Some(w);
            }
            if a.up.is_some() {
                return a.remaining_bidders.first().copied();
            }
            if let Some(p) = a.remaining_choosers.first() {
                return Some(*p);
            }
        }
        if let Some(i) = self.g1822_active_step() {
            if self.rnd().steps[i] == super::Step::Track {
                if let Some(p) = self.rnd().ndem_layers.first() {
                    return Some(*p);
                }
            }
        }
        self.g1822_mx_ndem_shareholder_player()
    }

    // =====================================================================
    // Track (step/track.rb): the NdeM's shareholders lay in turn
    // =====================================================================

    /// Track#setup for the NdeM.
    pub(crate) fn g1822_mx_track_setup(&mut self) {
        let layers = self.g1822_mx_ndem_holders();
        let r = self.rndm();
        r.ndem_layers = layers;
        r.ndem_layers_set = true;
    }

    /// Track#pass! for the NdeM: the next layer starts with a fresh
    /// allowance; the step passes once every layer has.
    pub(crate) fn g1822_mx_track_pass(&mut self, i: usize) {
        let r = self.rndm();
        if !r.ndem_layers.is_empty() {
            r.ndem_layers.remove(0);
        }
        if r.ndem_layers.is_empty() {
            r.passed[i] = true;
        } else {
            r.num_laid_track = 0;
            r.upgraded_track = false;
            r.laid_hexes.clear();
        }
    }

    /// Track#pass! (MX): the NdeM's next layer, or the step passes.
    pub(crate) fn g1822_mx_track_pass_any(&mut self, i: usize, c: &str) {
        if c == NDEM {
            self.g1822_mx_track_pass(i);
        } else {
            self.rndm().passed[i] = true;
        }
    }

    /// MX SpecialTrack#process_lay_tile's cube branches: a cube company
    /// (P10/P11) places a cube; an extra-track company (P9/P17/P18) laying
    /// the cube tile places one with its extra lay. None: G1822's lay.
    pub(crate) fn g1822_mx_special_cube(&mut self, co: &str, hex_id: &str, tile_id: &str) -> Option<Result<(), GameError>> {
        let cube = Self::g1822_mx_cube_company(co);
        let extra_cube = !cube && self.g1822_company_ability_extra_track(co) && tile_id == "BC-0";
        if !cube && !extra_cube {
            return None;
        }
        Some(self.g1822_mx_special_cube_inner(co, hex_id, cube))
    }

    fn g1822_mx_special_cube_inner(&mut self, co: &str, hex_id: &str, cube: bool) -> Result<(), GameError> {
        let (idx, ab) = self
            .g1822_track_ability(co)
            .ok_or_else(|| GameError::new(format!("{co} has no tile lay ability (Ruby raises)")))?;
        if !self.hex_idx.contains_key(hex_id) {
            return err(format!("Unknown hex {hex_id}"));
        }
        self.g1822_pnw_place_cube(hex_id);
        if cube {
            // ability.use!; close when its count hits zero
            self.g1822_use_ability(co, idx, false)?;
            let left = self.g1822_abilities(co).get(idx).and_then(|a| a.count).unwrap_or(0);
            if left == 0 && ab.closed_when_used_up {
                self.close_company(co);
            }
            return Ok(());
        }
        let owner = self.co_owner(co).corp_sym().map(str::to_string).unwrap_or_default();
        self.g1822_use_ability(co, idx, false)?;
        if self.is_minor(&owner) {
            self.g1822_use_ability(co, idx, false)?;
        } else {
            self.rndm().extra_laided_track = true;
        }
        let left = self.g1822_abilities(co).get(idx).and_then(|a| a.count).unwrap_or(0);
        if ab.kind == super::track::AbKind::TileLay && left <= 0 && ab.closed_when_used_up {
            self.close_company(co);
        }
        // handle_extra_tile_lay_company
        if ab.must_lay_together {
            if self.rnd().num_laid_track == 1 && !ab.consume_tile_lay {
                self.rndm().num_laid_track += 1;
            }
            self.rndm().special_track_company = if left > 0 && !self.co_closed(co) { Some(co.to_string()) } else { None };
        } else {
            self.rndm().special_track_company = None;
        }
        Ok(())
    }

    // =====================================================================
    // Builder cubes (game.rb terrain? / max_builder_cubes)
    // =====================================================================

    /// MX `max_builder_cubes(tile)`: 2 for a mountain, 1 for a hill, 1 for
    /// a river (the tile's own terrain upgrades).
    pub(crate) fn g1822_mx_max_cubes(&self, hi: usize) -> u32 {
        let ups = &self.hexes[hi].tile.upgrades;
        let has = |t: &str| ups.iter().any(|u| u.terrain == t);
        let mut n = 0;
        if has("mountain") {
            n += 2;
        }
        if has("hill") {
            n += 1;
        }
        if has("river") {
            n += 1;
        }
        n
    }

    pub(crate) fn g1822_mx_can_hold_cubes(&self, hi: usize) -> bool {
        self.g1822_pnw_cubes(&self.hexes[hi].id) < self.g1822_mx_max_cubes(hi)
    }

    /// MX `upgrade_cost`: G1822's, less 80 for two cubes on a mountain (or
    /// 40 for one on a hill), and 20 for a remaining cube on a river.
    pub(crate) fn g1822_mx_cube_discount(&self, hi: usize) -> i32 {
        let ups = &self.hexes[hi].tile.upgrades;
        let has = |t: &str| ups.iter().any(|u| u.terrain == t);
        let mut cubes = self.g1822_pnw_cubes(&self.hexes[hi].id) as i32;
        let mut d = 0;
        if cubes >= 2 && has("mountain") {
            cubes -= 2;
            d += 80;
        } else if cubes >= 1 && has("hill") {
            cubes -= 1;
            d += 40;
        }
        if cubes >= 1 && has("river") {
            d += 20;
        }
        d
    }

    // =====================================================================
    // Tile rules (game.rb upgrades_to?, legal_tile_rotation?)
    // =====================================================================

    pub(crate) fn g1822_mx_cube_company(co: &str) -> bool {
        co == "P10" || co == "P11"
    }

    pub(crate) fn g1822_mx_port_company(co: &str) -> bool {
        co == "P17" || co == "P18"
    }

    /// MX `upgrades_to?`: blue upgrades to blue; a cube company's only
    /// "tile" is the cube. None: fall through.
    pub(crate) fn g1822_mx_upgrades_to(&self, hi: usize, to_base: &str, selected: Option<&str>) -> Option<bool> {
        if self.g1822_tile_color_name(hi) == "blue" && self.g1822_def_color_name(to_base) == "blue" {
            return Some(true);
        }
        if selected.map_or(false, Self::g1822_mx_cube_company) && to_base != "BC" {
            return Some(false);
        }
        None
    }

    /// MX `must_connect_mc?(hex)`: M22 while nobody owns C2 or presides MC.
    pub(crate) fn g1822_mx_must_connect_mc(&self, hi: usize) -> bool {
        if self.hexes[hi].id != "M22" {
            return false;
        }
        let c2 = !self.co_closed("C2") && self.co_owner("C2").is_player();
        let mc = self.g1822_corp_president("MC").is_some();
        !c2 && !mc
    }

    /// MX `connects_mc?(tile)` for the tile `to` (rotated) laid on M22: its
    /// edge-5 path's town has another exit whose neighbour's tile exits
    /// back toward M22. Ruby raises (nil.edges) when that town has no other
    /// path — Err.
    pub(crate) fn g1822_mx_connects_mc(&self, to: &super::tile_rules::TV) -> Result<bool, GameError> {
        use crate::tiles::PathEndpoint;
        let edge_of = |p: &crate::tiles::PathDef| -> Option<u8> {
            [&p.a, &p.b].iter().find_map(|e| if let PathEndpoint::Edge(n) = e { Some(*n) } else { None })
        };
        let town_of = |p: &crate::tiles::PathDef| -> Option<usize> {
            [&p.a, &p.b].iter().find_map(|e| if let PathEndpoint::Town(n) = e { Some(*n) } else { None })
        };
        let Some(pi) = to.paths.iter().position(|p| edge_of(p) == Some(5)) else { return Ok(false) };
        let town = town_of(&to.paths[pi]);
        let Some(other) = to.paths.iter().enumerate().find(|(j, p)| *j != pi && town_of(p) == town).map(|(_, p)| p) else {
            return err("connects_mc?: no other path from the Mexico City town (Ruby raises)");
        };
        let Some(exit_out) = edge_of(other) else { return err("connects_mc?: no exit (Ruby raises)") };
        let Some(nid) = M22_ADJACENT.iter().find(|(e, _)| *e == exit_out).map(|(_, h)| *h) else {
            return err("connects_mc?: no neighbour (Ruby raises)");
        };
        let Some(&ni) = self.hex_idx.get(nid) else { return Ok(false) };
        let back = (exit_out + 3) % 6;
        let tv = super::tile_rules::TV::from_tile(&self.hexes[ni].tile);
        Ok(tv.exits().contains(&back))
    }

    // =====================================================================
    // Steps the NdeM runs (token, dividend, cash-out, auction)
    // =====================================================================

    /// MX Dividend#share_price_change for the NdeM: left on no revenue,
    /// otherwise right once per full share price earned.
    pub(crate) fn g1822_mx_ndem_price_change(&mut self, revenue: i32) {
        let ci = self.ci(NDEM);
        let Some(sp) = self.corporations[ci].share_price.clone() else { return };
        if revenue <= 0 {
            self.g1822_move_price(NDEM, Dir::Left);
            return;
        }
        let times = (revenue as f64 / sp.price as f64).floor() as i32;
        for _ in 0..times.max(0) {
            self.g1822_move_price(NDEM, Dir::Right);
        }
    }

    /// CashOutNdem#players_hold_shares?.
    pub(crate) fn g1822_mx_players_hold_ndem(&self) -> bool {
        !self.g1822_mx_ndem_holders().is_empty()
    }

    fn g1822_mx_ndem_closing_now(&self) -> bool {
        self.g().ndem_state == NdemState::Closing && self.g1822_operator().as_deref() == Some(NDEM)
    }

    /// CashOutNdem#active? (= blocking?).
    pub(crate) fn g1822_mx_cash_out_active(&self) -> bool {
        self.g1822_mx_ndem_closing_now() && self.g1822_mx_players_hold_ndem()
    }

    /// CashOutNdem#process_pass: every player's NdeM shares are sold to
    /// the pool at the NdeM's price (no price change).
    pub(crate) fn g1822_mx_cash_out(&mut self) {
        let ci = self.ci(NDEM);
        for pid in self.g1822_mx_ndem_holders() {
            let shares = self.corporations[ci].shares_owned_by_in_order(&EntityId::player(pid));
            if shares.is_empty() {
                continue;
            }
            let percent: u32 = shares.iter().map(|&s| self.corporations[ci].shares[s].percent as u32).sum();
            let b = Bundle { corp: NDEM.to_string(), shares, percent: percent as u8, price_override: None };
            self.g1822_pool_sell_shares(&b, false);
        }
    }

    /// AuctionNdemTokens#active? (= blocking?).
    pub(crate) fn g1822_mx_auction_active(&self) -> bool {
        self.g1822_mx_ndem_closing_now()
    }

    /// AuctionNdemTokens#setup (when the closing NdeM starts operating).
    pub(crate) fn g1822_mx_auction_setup(&mut self) {
        let mut available = Vec::new();
        for (hi, hex) in self.hexes.iter().enumerate() {
            for (ci, city) in hex.tile.cities.iter().enumerate() {
                if city.tokens.iter().flatten().any(|t| t.corporation_id == NDEM) {
                    available.push(NdemTok { hex: hex.id.clone(), city: ci, cost: self.g1822_mx_token_cost(hi, ci) });
                }
            }
        }
        // `ndem.tokens.select(&:used)` order: the NdeM's token order.
        let order = self.g1822_mx_ndem_token_order();
        available.sort_by_key(|t| order.iter().position(|(h, c)| *h == t.hex && *c == t.city).unwrap_or(usize::MAX));
        let players: Vec<u32> = self.players.iter().map(|p| p.id).collect();
        self.rndm().ndem_auction = Some(NdemAuction {
            available,
            player_step_order: players.clone(),
            remaining_choosers: players,
            ..Default::default()
        });
    }

    /// The (hex, city) of each used NdeM token, in the NdeM's token order.
    fn g1822_mx_ndem_token_order(&self) -> Vec<(String, usize)> {
        let ci = self.ci(NDEM);
        let mut out: Vec<(String, usize)> = Vec::new();
        for t in self.corporations[ci].tokens.iter().filter(|t| t.used) {
            let Some(&hi) = self.hex_idx.get(t.city_hex_id.as_str()) else { continue };
            for (c, city) in self.hexes[hi].tile.cities.iter().enumerate() {
                if city.tokens.iter().flatten().any(|x| x.corporation_id == NDEM)
                    && !out.iter().any(|(h, cc)| *h == t.city_hex_id && *cc == c)
                {
                    out.push((t.city_hex_id.clone(), c));
                    break;
                }
            }
        }
        out
    }

    /// `ndem_token_cost(token)`: the city's revenue in the latest phase
    /// colour it has, at least 10.
    fn g1822_mx_token_cost(&self, hi: usize, ci: usize) -> i32 {
        let rev = self
            .g1822_tiered_city_revenue(hi)
            .unwrap_or_else(|| self.hexes[hi].tile.cities[ci].revenue);
        rev.max(10)
    }

    /// `max_bid_for_token(player)`: the cash of the player's richest major
    /// still holding an exchange token.
    pub(crate) fn g1822_mx_max_bid(&self, pid: u32) -> i32 {
        self.g1822_mx_token_corps(pid).iter().map(|c| self.corporations[self.ci(c)].cash).max().unwrap_or(0)
    }

    /// The player's presidencies with an exchange token.
    fn g1822_mx_token_corps(&self, pid: u32) -> Vec<String> {
        self.corporations
            .iter()
            .filter(|c| !c.closed && c.owner_id == EntityId::player(pid) && self.g1822_exchange_tokens(&c.sym) > 0)
            .map(|c| c.sym.clone())
            .collect()
    }

    /// `corp_can_purchase_token?(corporation, token)`.
    fn g1822_mx_corp_can_purchase(&self, c: &str, t: &NdemTok) -> bool {
        let Some(&hi) = self.hex_idx.get(t.hex.as_str()) else { return false };
        self.corporations[self.ci(c)].cash >= t.cost
            && self.g1822_exchange_tokens(c) > 0
            && !(0..self.hexes[hi].tile.cities.len()).any(|ci| self.g1822_city_tokened_by(hi, ci, c))
    }

    pub(crate) fn g1822_mx_player_can_purchase(&self, pid: u32, t: &NdemTok) -> bool {
        self.g1822_mx_presidencies(pid).iter().any(|c| self.g1822_mx_corp_can_purchase(c, t))
    }

    /// `player.presidencies` (open corporations the player presides).
    fn g1822_mx_presidencies(&self, pid: u32) -> Vec<String> {
        self.corporations
            .iter()
            .filter(|c| !c.closed && c.owner_id == EntityId::player(pid))
            .map(|c| c.sym.clone())
            .collect()
    }

    fn g1822_mx_auction(&self) -> &NdemAuction {
        self.rnd().ndem_auction.as_ref().expect("NdeM auction")
    }

    /// `auction_over?`.
    fn g1822_mx_auction_over(a: &NdemAuction) -> bool {
        a.remaining_bidders.is_empty() || (a.remaining_bidders.len() == 1 && Some(a.remaining_bidders[0]) == a.high_bidder)
    }

    /// AuctionNdemTokens#actions for the NdeM.
    pub(crate) fn g1822_mx_auction_actions(&self, ent: &Ent) -> Vec<&'static str> {
        match ent {
            Ent::Corp(c) if c == NDEM && self.g1822_operator().as_deref() == Some(NDEM) => {}
            _ => return vec!["pass"],
        }
        let Some(a) = self.rnd().ndem_auction.as_ref() else { return vec!["pass"] };
        if a.up.is_none() {
            return vec!["remove_token", "pass"];
        }
        if a.winner.is_some() {
            return vec!["merge"];
        }
        if a.high_bidder.is_some() {
            return vec!["bid", "pass"];
        }
        vec!["bid"]
    }

    /// AuctionNdemTokens#auto_actions: the pass Ruby takes by itself.
    pub(crate) fn g1822_mx_auction_auto_pass(&self) -> bool {
        let Some(a) = self.rnd().ndem_auction.as_ref() else { return false };
        if a.remaining_choosers.is_empty() {
            return true;
        }
        match a.up {
            None => {
                let p = a.remaining_choosers[0];
                !a.available.iter().any(|t| self.g1822_mx_player_can_purchase(p, t))
            }
            Some(u) => {
                !Self::g1822_mx_auction_over(a)
                    && a.remaining_bidders.first().map_or(false, |&b| !self.g1822_mx_player_can_purchase(b, &a.available[u]))
            }
        }
    }

    pub(crate) fn g1822_mx_auction_process(&mut self, action: &Action) -> Result<(), GameError> {
        match action {
            Action::Pass { .. } => {
                let a = self.g1822_mx_auction().clone();
                if a.remaining_choosers.is_empty() {
                    // process_pass_end_round
                    self.gm().ndem_state = NdemState::Closed;
                    self.g1822_mx_close_ndem()?;
                } else if a.up.is_none() {
                    self.rndm().ndem_auction.as_mut().unwrap().remaining_choosers.remove(0);
                } else {
                    let au = self.rndm().ndem_auction.as_mut().unwrap();
                    if !au.remaining_bidders.is_empty() {
                        au.remaining_bidders.remove(0);
                    }
                    self.g1822_mx_check_auction_over()?;
                }
                Ok(())
            }
            Action::RemoveToken { hex_id, city_index, .. } => {
                let a = self.g1822_mx_auction().clone();
                if a.up.is_some() {
                    return err("A token is already up for bid");
                }
                let player = *a.remaining_choosers.first().ok_or_else(|| GameError::new("nobody to choose"))?;
                let hex = self.g1822_resolve_hex(hex_id)?;
                let Some(u) = a.available.iter().position(|t| t.hex == hex && t.city == *city_index as usize) else {
                    // the token there is not an available NdeM token:
                    // player_can_purchase_token?(player, nil) is false
                    return err(format!("No company can purchase a token at {hex}"));
                };
                if !self.g1822_mx_player_can_purchase(player, &a.available[u]) {
                    return err(format!("Player {player} does not have a company that can purchase a token at {hex}"));
                }
                let order: Vec<u32> = self.players.iter().map(|p| p.id).collect();
                let start = order.iter().position(|&p| p == player).unwrap_or(0);
                let mut bidders = order[start..].to_vec();
                bidders.extend_from_slice(&order[..start]);
                let au = self.rndm().ndem_auction.as_mut().unwrap();
                au.up = Some(u);
                au.remaining_bidders = bidders;
                au.high_bid = 0;
                au.high_bidder = None;
                au.winner = None;
                Ok(())
            }
            Action::Bid { price, .. } => {
                let a = self.g1822_mx_auction().clone();
                let Some(u) = a.up else { return err("No token is up for bid") };
                let bidder = *a.remaining_bidders.first().ok_or_else(|| GameError::new("no bidder"))?;
                let min = if a.high_bid == 0 { a.available[u].cost } else { a.high_bid + 10 };
                if *price < min {
                    return err(format!("Minimum bid is {min}"));
                }
                if *price > self.g1822_mx_max_bid(bidder) {
                    return err(format!("Maximum bid is {}", self.g1822_mx_max_bid(bidder)));
                }
                if price % 10 != 0 {
                    return err("Bids must be in increments of 10");
                }
                let au = self.rndm().ndem_auction.as_mut().unwrap();
                au.high_bid = *price;
                au.available[u].cost = price + 10;
                au.high_bidder = Some(bidder);
                let b = au.remaining_bidders.remove(0);
                au.remaining_bidders.push(b);
                self.g1822_mx_check_auction_over()
            }
            Action::Merge { corporation_sym, .. } => {
                let a = self.g1822_mx_auction().clone();
                let (Some(u), Some(w)) = (a.up, a.winner) else { return err("No auction winner") };
                let c = corporation_sym.clone();
                if !self.g1822_mx_mergeable(w).contains(&c) {
                    return err(format!("{c} cannot buy the NdeM token"));
                }
                let t = a.available[u].clone();
                let hi = *self.hex_idx.get(t.hex.as_str()).ok_or_else(|| GameError::new("unknown hex"))?;
                // @token_up_for_bid.remove!
                self.g1822_remove_token_at(hi, t.city, NDEM, None);
                self.g1822_remove_exchange_token(&c);
                let cix = self.ci(&c);
                self.corporations[cix].tokens.push(Token::new(c.clone(), 0));
                let tidx = self.corporations[cix].tokens.len() - 1;
                self.g1822_put_token(&c, tidx, hi, t.city, false)?;
                self.corp_spend(&c, a.high_bid)?;
                self.g1822_ruby_graph_clear();
                let au = self.rndm().ndem_auction.as_mut().unwrap();
                au.available.remove(u);
                au.up = None;
                au.winner = None;
                if !au.player_step_order.is_empty() {
                    let f = au.player_step_order.remove(0);
                    au.player_step_order.push(f);
                }
                au.remaining_choosers = au.player_step_order.clone();
                Ok(())
            }
            _ => err("AuctionNdemTokens cannot process this action"),
        }
    }

    /// check_auction_over: the high bidder wins (Ruby raises when nobody
    /// bid: `nil.name`).
    fn g1822_mx_check_auction_over(&mut self) -> Result<(), GameError> {
        let a = self.g1822_mx_auction().clone();
        if !Self::g1822_mx_auction_over(&a) {
            return Ok(());
        }
        let Some(w) = a.high_bidder else { return err("NdeM auction ended without a bid (Ruby raises)") };
        self.rndm().ndem_auction.as_mut().unwrap().winner = Some(w);
        Ok(())
    }

    /// `mergeable`: the winner's majors with an exchange token and the cash
    /// for the high bid.
    pub(crate) fn g1822_mx_mergeable(&self, winner: u32) -> Vec<String> {
        let bid = self.rnd().ndem_auction.as_ref().map_or(0, |a| a.high_bid);
        self.g1822_mx_presidencies(winner)
            .into_iter()
            .filter(|c| self.g1822_exchange_tokens(c) > 0 && self.corporations[self.ci(c)].cash >= bid)
            .collect()
    }

    /// `ndem.close!` (Corporation#close!: tokens stay on the map, the
    /// shares where they are; it leaves the market and stops operating).
    fn g1822_mx_close_ndem(&mut self) -> Result<(), GameError> {
        let ci = self.ci(NDEM);
        if self.corporations[ci].cash != 0 {
            return err(format!("NDEM cannot close with @cash={} (Ruby raises)", self.corporations[ci].cash));
        }
        if let Some(sp) = self.corporations[ci].share_price.clone() {
            if let Some(v) = self.market_cell_corps.get_mut(&(sp.row, sp.column)) {
                v.retain(|c| c != NDEM);
            }
        }
        let co = &mut self.corporations[ci];
        co.closed = true;
        co.floated = false;
        co.owner_id = EntityId::none();
        // Corporation#share_price is nil once closed
        co.share_price = None;
        Ok(())
    }

    // =====================================================================
    // MinorAcquisition (step/minor_acquisition.rb)
    // =====================================================================

    /// `connected_to_port?(entity)`.
    pub(crate) fn g1822_mx_connected_to_port(&self, c: &str) -> bool {
        let g = self.g1822_graph(c);
        PORT_HEXES.iter().any(|h| self.hex_idx.get(*h).map_or(false, |hi| g.connected_hexes.contains_key(hi)))
    }
}
