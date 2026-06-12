//! 1867 CN nationalization (Ruby G1867 game.rb:556-660, 913-930,
//! 1015-1045 + step/major_trainless.rb).
//!
//! The CN national corporation never operates and holds no shares — it
//! exists as a token holder. Corporations are nationalized into it when
//! they are trainless after the 4/6/8 train events
//! (`postevent_trainless_nationalization`: minors immediately, majors by
//! choice via the MajorTrainless step), when they cannot pay loan
//! interest (LoanOperations), and — for all remaining minors — on the
//! first 8-train (`minors_nationalized`).

use crate::actions::{Action, GameError};
use crate::entities::{Corporation, EntityId, Share, Token};
use crate::game::BaseGame;
use crate::title::{CorpType, CorporationDef};

/// Build a fresh corporation from its title def — the same construction
/// `BaseGame::new_titled` performs, reused by `reset_corporation` (Ruby
/// resets a nationalized major by swapping in `init_corporations`' copy).
pub(crate) fn build_corporation_from_def(cd: &CorporationDef) -> Corporation {
    let tokens: Vec<Token> = cd
        .token_prices
        .iter()
        .map(|&price| Token::new(cd.sym.to_string(), price))
        .collect();
    let mut shares = Vec::with_capacity(cd.shares.len());
    for (si, &pct) in cd.shares.iter().enumerate() {
        let mut s = Share::new(cd.sym.to_string(), pct, si == 0);
        s.index = si;
        shares.push(s);
    }
    let mut corp = Corporation::new(cd.sym.to_string(), cd.name.to_string(), tokens, shares);
    // Ruby Corporation#share_percent: the second cert's percent, or HALF
    // the president's when the president cert is the only one.
    corp.share_unit_percent = cd
        .shares
        .get(1)
        .copied()
        .unwrap_or_else(|| cd.shares.first().copied().unwrap_or(20) / 2);
    corp.float_percent = cd.float_percent;
    corp.capitalization = cd.capitalization;
    corp.corp_type = cd.corp_type;
    corp.max_ownership_percent = cd.max_ownership_percent;
    corp.always_market_price = cd.always_market_price;
    corp
}

impl BaseGame {
    /// Ruby G1867 `setup` + `add_neutral_tokens` (game.rb:913-930,
    /// 956-971): strip the national's nominal cert, seed the reserved-hex
    /// list (done in `new_titled`), and place its setup tokens — neutral
    /// green placeholders on D2/L12 (removed at `green_minors_available`)
    /// and the CN's first real token on F16. No-op for titles without a
    /// national.
    pub(crate) fn setup_national(&mut self) {
        let Some(ns) = self.title_def().national_setup() else {
            return;
        };
        if let Some(&ci) = self.corp_idx.get(ns.sym) {
            // Ruby `@national.shares.clear` — no certificates exist.
            self.corporations[ci].shares.clear();
        }
        for spot in ns.tokens {
            let Some(&hi) = self.hex_idx.get(spot.hex) else {
                continue;
            };
            let n_cities = self.hexes[hi].tile.cities.len();
            if n_cities == 0 {
                continue;
            }
            let city_idx = if spot.last_city { n_cities - 1 } else { 0 };
            let mut token = Token::new(ns.sym.to_string(), 0);
            token.used = true;
            token.city_hex_id = spot.hex.to_string();
            if spot.neutral {
                // Standalone placeholder (Ruby builds these OUTSIDE the
                // national's token list); a city holding one never blocks
                // route traversal.
                token.token_type = "neutral".to_string();
            } else if let Some(&ci) = self.corp_idx.get(ns.sym) {
                if let Some(ti) = self.corporations[ci].next_token_index() {
                    self.corporations[ci].tokens[ti].used = true;
                    self.corporations[ci].tokens[ti].city_hex_id = spot.hex.to_string();
                }
            }
            let city = &mut self.hexes[hi].tile.cities[city_idx];
            if let Some(slot) = city.tokens.iter().position(|t| t.is_none()) {
                city.tokens[slot] = Some(token);
            }
        }
    }

    /// Remove every neutral placeholder token from the map (Ruby
    /// `@green_tokens.map(&:remove!)` at `event_green_minors_available!`).
    pub(crate) fn remove_neutral_tokens(&mut self) {
        for hex in &mut self.hexes {
            for city in &mut hex.tile.cities {
                for slot in city.tokens.iter_mut() {
                    if slot.as_ref().is_some_and(|t| t.token_type == "neutral") {
                        *slot = None;
                    }
                }
            }
        }
        self.clear_graph_cache();
    }

    /// Sort corporation syms in operating order (Ruby `Corporation#<=>`:
    /// price desc, rightmost column, lowest row, earlier cell arrival,
    /// name) — the comparator behind `compute_operating_order`, applied to
    /// an arbitrary subset. Corps without a share price sort first (Ruby
    /// `<=>` returns -1 for a key-less self), which only arises for
    /// unstarted minors whose handling is order-independent.
    pub(crate) fn sort_syms_operating_order(&self, syms: Vec<String>) -> Vec<String> {
        let mut keyed: Vec<(bool, i32, u8, u8, usize, String, String)> = syms
            .into_iter()
            .map(|sym| {
                let (has_price, price, col, row, pos) = self
                    .corp_idx
                    .get(sym.as_str())
                    .and_then(|&ci| self.corporations[ci].share_price.as_ref().map(|sp| (ci, sp)))
                    .map_or((false, 0, 0, 0, 0), |(_, sp)| {
                        (
                            true,
                            sp.price,
                            sp.column,
                            sp.row,
                            self.market_cell_position(&sym, sp.row, sp.column),
                        )
                    });
                let name = self
                    .corp_idx
                    .get(sym.as_str())
                    .map_or(String::new(), |&ci| self.corporations[ci].name.clone());
                (has_price, price, col, row, pos, name, sym)
            })
            .collect();
        keyed.sort_by(|a, b| {
            a.0.cmp(&b.0) // price-less first (Ruby self-without-key => -1)
                .then(b.1.cmp(&a.1)) // highest price first
                .then(b.2.cmp(&a.2)) // rightmost column first
                .then(a.3.cmp(&b.3)) // lowest row first
                .then(a.4.cmp(&b.4)) // earlier cell arrival first
                .then(a.5.cmp(&b.5)) // alphabetical name
        });
        keyed.into_iter().map(|(.., sym)| sym).collect()
    }

    /// Ruby G1867 `nationalize!` (game.rb:568-643). Loans are repaid while
    /// cash covers the face value; the share price moves left once plus
    /// twice per remaining loan; remaining cash goes to the bank; players
    /// are paid the (post-move) price per share from the bank; the corp's
    /// map tokens are replaced by CN tokens in
    /// highest-city-revenue-then-hex-id order (skipping tiles the CN
    /// already tokened, honoring the Montreal reservation); minors close,
    /// majors reset to their unstarted state.
    pub(crate) fn nationalize_corporation(&mut self, sym: &str) {
        let Some(&ci) = self.corp_idx.get(sym) else {
            return;
        };
        {
            let c = &self.corporations[ci];
            // Ruby's `return if !corporation.floated? ||
            // !@corporations.include?(corporation)`.
            if !c.floated || c.closed || c.corp_type == CorpType::National {
                return;
            }
        }
        let face = self.title_def().loan_value();

        // 1. Forced repayment while cash covers a loan.
        while self.corporations[ci].loans > 0 && self.corporations[ci].cash >= face {
            self.repay_loan(ci);
        }

        // 2. Price: left once, plus twice per loan still outstanding
        // (`nationalization_loan_movement`).
        let moves = 1 + 2 * self.corporations[ci].loans;
        for _ in 0..moves {
            if let Some(sp) = self.corporations[ci].share_price.clone() {
                let (nr, nc) = self.stock_market.move_left(sp.row, sp.column);
                if let Some(new_sp) = self.stock_market.share_price_at(nr, nc) {
                    self.corporations[ci].share_price = Some(new_sp);
                    self.update_market_cell(sym, sp.row, sp.column, nr, nc);
                }
            }
        }

        // 3. Treasury to the bank (`nationalization_transfer_assets`).
        let cash = self.corporations[ci].cash;
        self.corporations[ci].cash = 0;
        self.bank.cash += cash;

        // 4. Players paid price × shares from the bank (pool/treasury
        // certs get nothing — Ruby iterates `@players` only).
        let per_share = self.corporations[ci]
            .share_price
            .as_ref()
            .map_or(0, |sp| sp.price);
        let unit = self.corporations[ci].share_unit();
        let payouts: Vec<(usize, i32)> = self
            .players
            .iter()
            .enumerate()
            .filter_map(|(pi, p)| {
                let pct = self.corporations[ci].percent_owned_by(&EntityId::player(p.id)) as i32;
                (pct > 0).then_some((pi, (pct / unit) * per_share))
            })
            .collect();
        for (pi, amount) in payouts {
            self.players[pi].cash += amount;
            self.bank.cash -= amount;
        }

        // 5. Map tokens become CN tokens.
        self.replace_tokens_with_national(ci);

        // 6. Minors close; majors reset to their unstarted state.
        if self.corporations[ci].corp_type == CorpType::Minor {
            self.close_corporation(ci);
        } else {
            self.reset_corporation(ci);
        }
        self.clear_graph_cache();
    }

    /// The token-replacement half of `nationalize!` (game.rb:604-632):
    /// each placed token is removed in highest-city-revenue-then-hex-id
    /// order; a CN token replaces it unless the tile already carries a
    /// (non-neutral) CN token, the CN is out of tokens, or the CN's
    /// remaining tokens are all reserved (Montreal) — placing on a
    /// reserved hex consumes its reservation.
    fn replace_tokens_with_national(&mut self, corp_ci: usize) {
        let sym = self.corporations[corp_ci].sym.clone();
        let national_ci = self
            .title_def()
            .national_setup()
            .and_then(|ns| self.corp_idx.get(ns.sym).copied());

        // Collect the corp's placed tokens city-side: (hex, city, slot,
        // city revenue). Ruby sorts ascending by [max_revenue, hex.id] and
        // iterates reversed.
        let mut placed: Vec<(String, usize, usize, i32)> = Vec::new();
        for hex in &self.hexes {
            for (cidx, city) in hex.tile.cities.iter().enumerate() {
                for (slot, tok) in city.tokens.iter().enumerate() {
                    if tok.as_ref().is_some_and(|t| t.corporation_id == sym) {
                        placed.push((hex.id.clone(), cidx, slot, city.revenue));
                    }
                }
            }
        }
        placed.sort_by(|a, b| a.3.cmp(&b.3).then(a.0.cmp(&b.0)));

        for (hex_id, cidx, slot, _) in placed.into_iter().rev() {
            let &hi = self.hex_idx.get(hex_id.as_str()).expect("placed token hex");

            // Remove the corp's token (city slot + corp-side entry).
            self.hexes[hi].tile.cities[cidx].tokens[slot] = None;
            if let Some(t) = self.corporations[corp_ci]
                .tokens
                .iter_mut()
                .find(|t| t.used && t.city_hex_id == hex_id)
            {
                t.used = false;
                t.city_hex_id = String::new();
            }

            let Some(nci) = national_ci else {
                continue;
            };
            let national_sym = self.corporations[nci].sym.clone();

            // Skip if any city on this tile already holds a real CN token.
            let tile_has_national = self.hexes[hi].tile.cities.iter().any(|c| {
                c.tokens.iter().flatten().any(|t| {
                    t.corporation_id == national_sym && t.token_type != "neutral"
                })
            });
            if tile_has_national {
                continue;
            }

            let Some(ti) = self.corporations[nci].next_token_index() else {
                continue;
            };

            // Reservation bookkeeping (game.rb:624-628): placing on a
            // reserved hex consumes the reservation; otherwise, when only
            // reserved tokens remain, hold them back.
            if let Some(pos) = self.national_reservations.iter().position(|h| h == &hex_id) {
                self.national_reservations.remove(pos);
            } else {
                let unused = self.corporations[nci].tokens.iter().filter(|t| !t.used).count();
                if unused == self.national_reservations.len() {
                    continue;
                }
            }

            self.corporations[nci].tokens[ti].used = true;
            self.corporations[nci].tokens[ti].city_hex_id = hex_id.clone();
            let placed_token = self.corporations[nci].tokens[ti].clone();
            let city = &mut self.hexes[hi].tile.cities[cidx];
            if let Some(open) = city.tokens.iter().position(|t| t.is_none()) {
                city.tokens[open] = Some(placed_token);
            }
        }
    }

    /// Ruby base `close_corporation` for a nationalized minor
    /// (base.rb:1896-1942): any remaining map tokens vanish, cash goes to
    /// the bank, trains leave play (CLOSED_CORP_TRAINS_REMOVED), its
    /// companies close, and the certs/market presence are wiped
    /// (`merger_close_minor` — shared with the merger round's closes).
    pub(crate) fn close_corporation(&mut self, ci: usize) {
        let sym = self.corporations[ci].sym.clone();
        let eid = EntityId::corporation(&sym);
        for hex in &mut self.hexes {
            for city in &mut hex.tile.cities {
                for slot in city.tokens.iter_mut() {
                    if slot.as_ref().is_some_and(|t| t.corporation_id == sym) {
                        *slot = None;
                    }
                }
            }
        }
        let cash = self.corporations[ci].cash;
        if cash > 0 {
            self.corporations[ci].cash = 0;
            self.bank.cash += cash;
        }
        self.corporations[ci].trains.clear();
        // Outstanding loans leave play WITHOUT returning to the bank pool
        // (Ruby's close_corporation drops the Loan objects).
        self.corporations[ci].loans = 0;
        for company in &mut self.companies {
            if company.owner == eid {
                company.closed = true;
            }
        }
        self.merger_close_minor(ci);
    }

    /// Ruby `reset_corporation` (base.rb:1948-1968 + the G1867 override):
    /// a nationalized major is replaced by its unstarted self — certs back
    /// in the IPO, no price, fresh tokens, its companies closed. Loans it
    /// still held leave play WITHOUT returning to the bank pool (faithful
    /// to Ruby, which drops the Loan objects).
    pub(crate) fn reset_corporation(&mut self, ci: usize) {
        let sym = self.corporations[ci].sym.clone();
        let eid = EntityId::corporation(&sym);
        for company in &mut self.companies {
            if company.owner == eid {
                company.closed = true;
            }
        }
        // Vacate the market cell.
        if let Some(sp) = self.corporations[ci].share_price.clone() {
            if let Some(corps) = self.market_cell_corps.get_mut(&(sp.row, sp.column)) {
                corps.retain(|c| c != &sym);
            }
        }
        // Defensive: the replacement loop removed all placed tokens; clear
        // any straggler city-side copies so the fresh corp starts clean.
        for hex in &mut self.hexes {
            for city in &mut hex.tile.cities {
                for slot in city.tokens.iter_mut() {
                    if slot.as_ref().is_some_and(|t| t.corporation_id == sym) {
                        *slot = None;
                    }
                }
            }
        }
        let fresh = self
            .title_def()
            .corporations()
            .iter()
            .find(|cd| cd.sym == sym)
            .map(build_corporation_from_def)
            .expect("reset corp def");
        self.corporations[ci] = fresh;
    }

    /// Ruby `postevent_trainless_nationalization!` (game.rb:1030-1045):
    /// every OPERATED corp without a train, in operating order — minors
    /// nationalize immediately, majors queue for the MajorTrainless
    /// choice.
    pub(crate) fn postevent_trainless_nationalization(&mut self) {
        let trainless: Vec<String> = self
            .corporations
            .iter()
            .filter(|c| {
                c.ever_operated
                    && c.floated
                    && !c.closed
                    && c.trains.is_empty()
                    && c.corp_type != CorpType::National
            })
            .map(|c| c.sym.clone())
            .collect();
        let trainless = self.sort_syms_operating_order(trainless);
        let mut majors: Vec<String> = Vec::new();
        for sym in trainless {
            let ci = self.corp_idx[sym.as_str()];
            match self.corporations[ci].corp_type {
                CorpType::Minor => self.nationalize_corporation(&sym),
                CorpType::Major => majors.push(sym),
                CorpType::National => {}
            }
        }
        // Ruby re-sorts the queue; minors' closes don't move major prices,
        // so this is the same order — kept literal (game.rb:1043).
        self.trainless_major = self.sort_syms_operating_order(majors);
    }

    /// Ruby G1867 `event_minors_nationalized!` (game.rb:1015-1023, first
    /// 8-train): every remaining minor leaves play — floated ones are
    /// nationalized (in operating order), unstarted ones simply close.
    pub(crate) fn event_minors_nationalized(&mut self) {
        let minors: Vec<String> = self
            .corporations
            .iter()
            .filter(|c| c.corp_type == CorpType::Minor && !c.closed)
            .map(|c| c.sym.clone())
            .collect();
        for sym in self.sort_syms_operating_order(minors) {
            let ci = self.corp_idx[sym.as_str()];
            if self.corporations[ci].floated {
                self.nationalize_corporation(&sym);
            } else {
                self.corporations[ci].closed = true;
            }
        }
    }

    /// Ruby `place_639_token` (game.rb:651-660): laying the gray Montreal
    /// tile claims the CN's reserved token — placed on the tile's (single)
    /// city if the reservation is still outstanding and the CN isn't
    /// already there.
    pub(crate) fn place_639_token(&mut self, hex_id: &str) {
        if self.national_reservations.is_empty() {
            return;
        }
        let Some(ns) = self.title_def().national_setup() else {
            return;
        };
        let Some(&nci) = self.corp_idx.get(ns.sym) else {
            return;
        };
        let Some(&hi) = self.hex_idx.get(hex_id) else {
            return;
        };
        let already = self.hexes[hi].tile.cities.iter().any(|c| {
            c.tokens
                .iter()
                .flatten()
                .any(|t| t.corporation_id == ns.sym && t.token_type != "neutral")
        });
        if already {
            return;
        }
        let Some(ti) = self.corporations[nci].next_token_index() else {
            return;
        };
        self.national_reservations.retain(|h| h != hex_id);
        self.corporations[nci].tokens[ti].used = true;
        self.corporations[nci].tokens[ti].city_hex_id = hex_id.to_string();
        let placed = self.corporations[nci].tokens[ti].clone();
        let city = &mut self.hexes[hi].tile.cities[0];
        if let Some(open) = city.tokens.iter().position(|t| t.is_none()) {
            city.tokens[open] = Some(placed);
        }
        self.clear_graph_cache();
    }

    /// The MajorTrainless step's action handling
    /// (step/major_trainless.rb): a queued major passes (declines) or
    /// chooses `nationalize`. Returns Ok(true) when the action was this
    /// step's — the caller skips normal round dispatch. After the queue
    /// drains, a frozen operating turn resumes via the caller.
    pub(crate) fn try_process_trainless_choice(
        &mut self,
        action: &Action,
    ) -> Result<bool, GameError> {
        if self.trainless_major.is_empty() {
            return Ok(false);
        }
        let (entity_id, choice) = match action {
            Action::Pass { entity_id } => (entity_id, None),
            Action::Choose { entity_id, choice } => (entity_id, Some(choice.as_str())),
            _ => return Ok(false),
        };
        let Some(pos) = self.trainless_major.iter().position(|m| m == entity_id) else {
            return Ok(false);
        };
        match choice {
            None => {
                // Declines nationalization.
                self.trainless_major.remove(pos);
            }
            Some("nationalize") => {
                let sym = self.trainless_major.remove(pos);
                self.nationalize_corporation(&sym);
            }
            Some(other) => {
                return Err(GameError::new(format!(
                    "Unknown MajorTrainless choice: {}",
                    other
                )));
            }
        }
        Ok(true)
    }
}

// ---------------------------------------------------------------------------
// Tests — the fixtures exercise minor nationalization end-to-end (the first
// 4-train), but the MajorTrainless choose/pass path sits beyond the
// train-export seam, so it is pinned here directly.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::BaseGame;
    use std::collections::HashMap;

    fn new_4p_game() -> BaseGame {
        let mut players = HashMap::new();
        players.insert(1, "Alice".to_string());
        players.insert(2, "Bob".to_string());
        players.insert(3, "Carol".to_string());
        players.insert(4, "Dave".to_string());
        BaseGame::build_titled("1867", vec![1, 2, 3, 4], players)
    }

    /// Trainless decisions only arise from stock/operating/merger rounds —
    /// park the fresh game in a stock round so the dispatch gate walks a
    /// step list that carries MajorTrainless.
    fn enter_stock_round(game: &mut BaseGame) {
        let order = game.player_order.clone();
        game.round = crate::rounds::Round::Stock(crate::rounds::StockState::new(&order, order[0]));
        game.update_round_state();
    }

    /// Float `sym` by surgery: price at market column `col`, all certs to
    /// player 1, a placed token on `hex` city 0.
    fn rig_floated(game: &mut BaseGame, sym: &str, col: u8, hex: &str) {
        let ci = game.corp_idx[sym];
        let sp = game.stock_market.share_price_at(0, col).unwrap();
        game.corporations[ci].share_price = Some(sp.clone());
        game.corporations[ci].ipo_price = Some(sp.clone());
        game.update_market_cell(sym, 0, col, 0, col);
        game.corporations[ci].floated = true;
        game.corporations[ci].ever_operated = true;
        let n = game.corporations[ci].shares.len();
        for i in 0..n {
            game.corporations[ci].set_share_owner(i, EntityId::player(1));
        }
        let hi = game.hex_idx[hex];
        let ti = game.corporations[ci].next_token_index().unwrap();
        game.corporations[ci].tokens[ti].used = true;
        game.corporations[ci].tokens[ti].city_hex_id = hex.to_string();
        let tok = game.corporations[ci].tokens[ti].clone();
        let city = &mut game.hexes[hi].tile.cities[0];
        let slot = city.tokens.iter().position(|t| t.is_none()).unwrap();
        city.tokens[slot] = Some(tok);
    }

    fn cn_tokens_on(game: &BaseGame, hex: &str) -> usize {
        let hi = game.hex_idx[hex];
        game.hexes[hi]
            .tile
            .cities
            .iter()
            .flat_map(|c| c.tokens.iter().flatten())
            .filter(|t| t.corporation_id == "CN" && t.token_type != "neutral")
            .count()
    }

    /// Minor nationalization end to end: forced loan repayment, one price
    /// step left plus two per unrepaid loan, treasury to the bank, the
    /// owner paid 2 × post-move price (a minor's 100% cert is two shares),
    /// the map token replaced by a CN token, the corp closed.
    #[test]
    fn nationalize_minor_repays_pays_and_replaces_token() {
        let mut game = new_4p_game();
        let sym = "BO";
        rig_floated(&mut game, sym, 10, "M9");
        let ci = game.corp_idx[sym];
        game.corporations[ci].loans = 2;
        game.corporations[ci].cash = 60; // covers ONE $50 repayment
        let col = game.corporations[ci].share_price.as_ref().unwrap().column;
        let price_at = |game: &BaseGame, c: u8| {
            game.stock_market.share_price_at(0, c).unwrap().price
        };
        let bank0 = game.bank.cash;
        let p1_0 = game.players[0].cash;

        game.nationalize_corporation(sym);

        let ci = game.corp_idx[sym];
        // One loan repaid ($50 of the $60), one left → price 1 + 2×1 = 3
        // steps left; the leftover loan is wiped on close.
        assert_eq!(game.corporations[ci].loans, 0);
        let expected_price = price_at(&game, col - 3);
        let payout = 2 * expected_price;
        // Owner: paid price × 2 shares.
        assert_eq!(game.players[0].cash, p1_0 + payout);
        // Bank: +$50 repay +$10 remaining treasury − payout.
        assert_eq!(game.bank.cash, bank0 + 50 + 10 - payout);
        // Corp closed, certs gone, off the market.
        assert!(game.corporations[ci].closed);
        assert!(!game.corporations[ci].floated);
        assert!(game.corporations[ci].share_price.is_none());
        assert!(game.corporations[ci].shares.iter().all(|s| s.owner.is_none()));
        // Its token became a CN token.
        assert_eq!(cn_tokens_on(&game, "M9"), 1);
    }

    /// A nationalized major resets to its unstarted self (Ruby
    /// reset_corporation): certs back in IPO, no price, fresh tokens —
    /// and the CN holds back its last unreserved tokens for Montreal.
    #[test]
    fn nationalize_major_resets_corporation() {
        let mut game = new_4p_game();
        let sym = "GTR";
        rig_floated(&mut game, sym, 14, "J12");
        let ci = game.corp_idx[sym];
        game.corporations[ci].cash = 30; // below face: no repayment possible
        game.corporations[ci].loans = 0;
        let col = game.corporations[ci].share_price.as_ref().unwrap().column;
        let p1_0 = game.players[0].cash;

        game.nationalize_corporation(sym);

        let ci = game.corp_idx[sym];
        let expected_price = game.stock_market.share_price_at(0, col - 1).unwrap().price;
        // 10 share units, all player 1's.
        assert_eq!(game.players[0].cash, p1_0 + 10 * expected_price);
        // Fresh unstarted major.
        let corp = &game.corporations[ci];
        assert!(!corp.closed && !corp.floated && !corp.ever_operated);
        assert!(corp.share_price.is_none() && corp.ipo_price.is_none());
        assert_eq!(corp.tokens.len(), 3);
        assert!(corp.tokens.iter().all(|t| !t.used));
        assert!(corp.shares.iter().all(|s| s.owner.is_none()));
        assert_eq!(cn_tokens_on(&game, "J12"), 1);
    }

    /// game.rb:626-628: when the CN's unused tokens are all spoken for by
    /// reservations (Montreal), a nationalized token is removed but NOT
    /// replaced off the reserved hex.
    #[test]
    fn national_reservation_holds_back_last_token() {
        let mut game = new_4p_game();
        let cn = game.corp_idx["CN"];
        // Burn CN tokens until exactly ONE is unused (reservations = [L12]).
        while game.corporations[cn].tokens.iter().filter(|t| !t.used).count() > 1 {
            let ti = game.corporations[cn].next_token_index().unwrap();
            game.corporations[cn].tokens[ti].used = true;
            game.corporations[cn].tokens[ti].city_hex_id = "OFFMAP".into();
        }
        let sym = "BO";
        rig_floated(&mut game, sym, 10, "M9");
        game.nationalize_corporation(sym);
        // Token removed, slot empty, CN did NOT spend its reserved token.
        assert_eq!(cn_tokens_on(&game, "M9"), 0);
        assert_eq!(
            game.corporations[cn].tokens.iter().filter(|t| !t.used).count(),
            1
        );
        assert_eq!(game.national_reservations, vec!["L12".to_string()]);
    }

    /// postevent_trainless_nationalization: trainless minors nationalize
    /// immediately; majors queue for the MajorTrainless step, whose
    /// choose/pass are processed via the normal action path (blocking
    /// everything else while queued).
    #[test]
    fn trainless_event_minors_close_majors_choose() {
        let mut game = new_4p_game();
        rig_floated(&mut game, "BO", 10, "M9");
        rig_floated(&mut game, "GTR", 14, "J12");
        rig_floated(&mut game, "CPR", 12, "D16");
        // CPR keeps a train — not trainless.
        let cpr = game.corp_idx["CPR"];
        let train = crate::entities::Train::new("4".into(), 4, 350);
        game.corporations[cpr].trains.push(train);

        enter_stock_round(&mut game);
        game.trainless_nationalization_pending = true;
        game.post_train_buy();

        // The minor closed immediately; only the trainless major queued.
        assert!(game.corporations[game.corp_idx["BO"]].closed);
        assert_eq!(game.trainless_major, vec!["GTR".to_string()]);
        assert!(!game.trainless_nationalization_pending);

        // While queued: a stranger's action is rejected by the gate, the
        // queued major's choose nationalizes it.
        let err = game.process_action_internal(&Action::Pass {
            entity_id: "1".to_string(),
        });
        assert!(err.is_err(), "queue must block other actions");
        game.process_action_internal(&Action::Choose {
            entity_id: "GTR".to_string(),
            choice: "nationalize".to_string(),
        })
        .unwrap();
        assert!(game.trainless_major.is_empty());
        let gtr = &game.corporations[game.corp_idx["GTR"]];
        assert!(!gtr.floated && gtr.share_price.is_none());
        assert_eq!(cn_tokens_on(&game, "J12"), 1);
    }

    // -- train export (game.rs or_round_finished; Ruby g_1867 game.rb:858-864
    //    + depot.rb:20-25). The OR-end export and the game-end timing it can
    //    latch are pinned here next to the nationalization machinery they
    //    feed. --

    /// Pin the game's phase to a named PhaseDef (test surgery — fixtures
    /// reach phases through purchases).
    fn set_phase(game: &mut BaseGame, name: &str) {
        let pd = game
            .title_def()
            .phases()
            .into_iter()
            .find(|p| p.name == name)
            .unwrap();
        game.phase = crate::core::Phase::new(
            pd.name.to_string(),
            pd.operating_rounds,
            pd.train_limit,
            pd.tiles.iter().map(|s| s.to_string()).collect(),
        );
    }

    /// Discard depot trains until `head` is the next to sell (simulates the
    /// earlier tiers having been bought).
    fn drain_depot_until(game: &mut BaseGame, head: &str) {
        while game.depot.trains.first().map(|t| t.name.as_str()) != Some(head) {
            game.depot.trains.remove(0);
        }
    }

    /// No 'export_train' status (phases 2/3/8) → or_round_finished no-ops.
    #[test]
    fn export_only_in_export_phases() {
        let mut game = new_4p_game();
        let n0 = game.depot.trains.len();
        game.or_round_finished(); // phase 2
        assert_eq!(game.depot.trains.len(), n0);
        assert_eq!(game.phase.name, "2");
    }

    /// Depot#export!: the depot head leaves the game AS IF PURCHASED — the
    /// phase changes and the train's first-instance events fire.
    #[test]
    fn export_removes_head_and_changes_phase() {
        let mut game = new_4p_game();
        set_phase(&mut game, "4");
        drain_depot_until(&mut game, "5");
        game.or_round_finished();
        assert_eq!(game.phase.name, "5");
        assert_eq!(game.depot.trains.first().unwrap().id, "5-1");
        assert!(game
            .events_fired
            .iter()
            .any(|e| e == "5:minors_cannot_start"));
        assert!(!game.game_end_triggered);
    }

    /// An exported 8 runs the full purchase machinery: phase 8, the 4s
    /// rust, post_train_buy queues the now-trainless major, and
    /// game_end_check latches final_phase with final_turn = turn + 1
    /// BEFORE the round transition bumps the turn.
    #[test]
    fn export_eight_rusts_fours_queues_majors_and_latches_end() {
        let mut game = new_4p_game();
        set_phase(&mut game, "7");
        rig_floated(&mut game, "GTR", 14, "J12");
        let gtr = game.corp_idx["GTR"];
        game.corporations[gtr]
            .trains
            .push(crate::entities::Train::new("4".into(), 4, 350));
        drain_depot_until(&mut game, "8");
        let turn0 = game.turn;

        game.or_round_finished();

        assert_eq!(game.phase.name, "8");
        assert!(game.corporations[gtr].trains.is_empty(), "the 4 rusts");
        assert_eq!(game.trainless_major, vec!["GTR".to_string()]);
        assert!(game.game_end_triggered);
        assert_eq!(game.final_turn, Some(turn0 + 1));
        // game_end_set_final_turn! (game.rb:787-790): the final OR set has
        // 3 rounds — the next SR→OR transition builds it.
        use crate::steps::{FinishedRound, RoundStart};
        let t = game.title_next_round(FinishedRound::Stock);
        assert_eq!(
            t.start,
            RoundStart::Operating {
                round_num: 1,
                total_ors: 3
            }
        );
    }

    /// GAME_END_CHECK bank: :current_or — a broken bank ends the game with
    /// the NEXT operating round to finish, mid-set included.
    #[test]
    fn bank_break_ends_with_current_or() {
        let mut game = new_4p_game();
        let mut s = crate::rounds::OperatingState::new(1, 2, Vec::new());
        s.finished = true;
        game.round = crate::rounds::Round::Operating(s);
        assert!(!game.should_end_now());
        game.bank.cash = 0;
        game.check_game_end();
        assert!(game.bank_broken && game.game_end_triggered);
        assert!(game.should_end_now(), "OR 1 of 2 already ends the game");
    }

    /// GAME_END_CHECK final_phase: :one_more_full_or_set — the final set
    /// of turn `final_turn` completes first (and bank, when also latched,
    /// wins with its earlier :current_or timing).
    #[test]
    fn final_phase_ends_after_one_more_full_or_set() {
        let mut game = new_4p_game();
        set_phase(&mut game, "8");
        game.check_game_end();
        assert_eq!(game.final_turn, Some(game.turn + 1));
        let mut s = crate::rounds::OperatingState::new(3, 3, Vec::new());
        s.finished = true;
        game.round = crate::rounds::Round::Operating(s);
        // Same turn: not yet — the FINAL set belongs to final_turn.
        assert!(!game.should_end_now());
        game.turn += 1;
        assert!(game.should_end_now());
        // A bank break supersedes: :current_or ends even a mid-set OR.
        game.turn += 1; // past final_turn — one_more_full_or_set alone says no
        assert!(!game.should_end_now());
        game.bank.cash = -1;
        game.check_game_end();
        assert!(game.should_end_now());
    }

    /// The pass arm: a queued major declines and stays on the map.
    #[test]
    fn trainless_major_may_decline() {
        let mut game = new_4p_game();
        rig_floated(&mut game, "GTR", 14, "J12");
        enter_stock_round(&mut game);
        game.trainless_nationalization_pending = true;
        game.post_train_buy();
        assert_eq!(game.trainless_major, vec!["GTR".to_string()]);
        game.process_action_internal(&Action::Pass {
            entity_id: "GTR".to_string(),
        })
        .unwrap();
        assert!(game.trainless_major.is_empty());
        let gtr = &game.corporations[game.corp_idx["GTR"]];
        assert!(gtr.floated && gtr.share_price.is_some());
    }
}
