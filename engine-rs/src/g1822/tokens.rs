//! 1822 station tokens: City#get_slot / tokenable? / place_token with the
//! positional reservation model, home tokens (G1822#place_home_token incl.
//! minor 14's pending London choice and the LNWR's destination), the
//! PendingToken, DestinationToken, Token and SpecialToken steps, exchange
//! tokens and `remove_extra_tokens!`.

use crate::actions::{Action, GameError};
use crate::entities::Token;
use crate::game::BaseGame;

use super::{err, Ent, PendingTok};

fn res_key(hex: &str, city: usize) -> String {
    format!("{hex}#{city}")
}

impl BaseGame {
    // -- reservations -----------------------------------------------------

    pub(crate) fn g1822_city_res(&self, hex: &str, city: usize) -> Vec<Option<String>> {
        self.g().city_reservations.get(&res_key(hex, city)).cloned().unwrap_or_default()
    }

    fn g1822_tile_res(&self, hex: &str) -> Vec<String> {
        self.g().tile_reservations.get(hex).cloned().unwrap_or_default()
    }

    /// Tile#add_reservation! (setup).
    pub(crate) fn g1822_add_reservation(&mut self, hex: &str, city: Option<usize>, corp: &str) {
        let Some(&hi) = self.hex_idx.get(hex) else { return };
        let n = self.hexes[hi].tile.cities.len();
        let city = if n == 1 { Some(0) } else { city };
        match city {
            Some(c) => {
                let slot = self.g1822_get_slot(hi, c, corp, false).unwrap_or(0);
                let v = self.gm().city_reservations.entry(res_key(hex, c)).or_default();
                // Array#insert(slot, entity): pads with nil past the end.
                while v.len() < slot {
                    v.push(None);
                }
                v.insert(slot, Some(corp.to_string()));
            }
            None => self
                .gm()
                .tile_reservations
                .entry(hex.to_string())
                .or_default()
                .push(corp.to_string()),
        }
    }

    /// CLOSED_CORP_RESERVATIONS_REMOVED (city.reservations.delete — the
    /// array shrinks — and the tile's).
    pub(crate) fn g1822_remove_reservations(&mut self, corp: &str) {
        let st = self.gm();
        for v in st.city_reservations.values_mut() {
            v.retain(|r| r.as_deref() != Some(corp));
        }
        for v in st.tile_reservations.values_mut() {
            v.retain(|r| r != corp);
        }
    }

    /// City#place_token's `remove_reservation!` (the slot becomes nil) and
    /// `tile.reservations.delete`.
    fn g1822_consume_reservation(&mut self, hex: &str, city: usize, corp: &str) {
        let st = self.gm();
        if let Some(v) = st.city_reservations.get_mut(&res_key(hex, city)) {
            if let Some(pos) = v.iter().position(|r| r.as_deref() == Some(corp)) {
                v[pos] = None;
            }
        }
        if let Some(v) = st.tile_reservations.get_mut(hex) {
            v.retain(|r| r != corp);
        }
    }

    /// City#get_slot(corporation, cheater:).
    pub(crate) fn g1822_get_slot(&self, hi: usize, ci: usize, corp: &str, cheater: bool) -> Option<usize> {
        let hex = &self.hexes[hi];
        let city = &hex.tile.cities[ci];
        let res = self.g1822_city_res(&hex.id, ci);
        let reservation = res.iter().position(|r| r.as_deref() == Some(corp));
        let open = (0..city.tokens.len()).find(|&i| city.tokens[i].is_none() && res.get(i).map_or(true, |r| r.is_none()));
        if cheater {
            return Some(open.unwrap_or(city.tokens.len()));
        }
        reservation.or(open)
    }

    /// City#available_slots.
    pub(crate) fn g1822_available_slots(&self, hi: usize, ci: usize) -> usize {
        let hex = &self.hexes[hi];
        let city = &hex.tile.cities[ci];
        let res = self.g1822_city_res(&hex.id, ci);
        (0..city.tokens.len())
            .filter(|&i| city.tokens[i].is_none() && res.get(i).map_or(true, |r| r.is_none()))
            .count()
    }

    pub(crate) fn g1822_city_tokened_by(&self, hi: usize, ci: usize, corp: &str) -> bool {
        self.hexes[hi].tile.cities[ci]
            .tokens
            .iter()
            .flatten()
            .any(|t| t.corporation_id == corp)
    }

    /// City#tokenable?(corporation, free:, tokens: tokens_by_type, …).
    pub(crate) fn g1822_city_tokenable(&self, corp: &str, hi: usize, ci: usize, free: bool, same_hex_allowed: bool) -> bool {
        let Some(&cix) = self.corp_idx.get(corp) else { return false };
        let co = &self.corporations[cix];
        // tokens_by_type: unused tokens, one per type (G1822 Token steps
        // exclude destination tokens; the city check sees all types).
        let mut types_seen: Vec<String> = Vec::new();
        let tokens: Vec<&Token> = co
            .tokens
            .iter()
            .filter(|t| !t.used)
            .filter(|t| {
                if types_seen.contains(&t.token_type) {
                    false
                } else {
                    types_seen.push(t.token_type.clone());
                    true
                }
            })
            .collect();
        if tokens.is_empty() {
            return false;
        }
        let hex_id = self.hexes[hi].id.clone();
        tokens.iter().any(|t| {
            if self.g1822_get_slot(hi, ci, corp, false).is_none() {
                return false;
            }
            if !free && t.price > co.cash {
                return false;
            }
            if !same_hex_allowed
                && (0..self.hexes[hi].tile.cities.len()).any(|c| self.g1822_city_tokened_by(hi, c, corp))
            {
                return false;
            }
            let res = self.g1822_city_res(&hex_id, ci);
            if res.iter().any(|r| r.as_deref() == Some(corp)) {
                return true;
            }
            // Tile#token_blocked_by_reservation? (:never blocking)
            let tres = self.g1822_tile_res(&hex_id);
            if !tres.is_empty() {
                let others = tres.iter().filter(|r| *r != corp).count();
                let avail: usize = (0..self.hexes[hi].tile.cities.len())
                    .map(|c| self.g1822_available_slots(hi, c))
                    .sum();
                if others >= avail {
                    return false;
                }
            }
            true
        })
    }

    /// Put the corp token `token_idx` into city (hi, ci) (City#exchange_
    /// token: get_slot, cheater appends) and consume the reservation.
    pub(crate) fn g1822_put_token(&mut self, corp: &str, token_idx: usize, hi: usize, ci: usize, cheater: bool) -> Result<(), GameError> {
        let slot = self
            .g1822_get_slot(hi, ci, corp, cheater)
            .ok_or_else(|| GameError::new(format!("{corp} cannot lay token - no token slots available")))?;
        let hex_id = self.hexes[hi].id.clone();
        let cix = self.ci(corp);
        if self.g1822_token_identity() && self.corporations[cix].tokens[token_idx].serial == 0 {
            self.gm().token_serial += 1;
            let s = self.g().token_serial;
            self.corporations[cix].tokens[token_idx].serial = s;
        }
        let mut tok = self.corporations[cix].tokens[token_idx].clone();
        tok.used = true;
        tok.city_hex_id = hex_id.clone();
        self.corporations[cix].tokens[token_idx].used = true;
        self.corporations[cix].tokens[token_idx].city_hex_id = hex_id.clone();
        let city = &mut self.hexes[hi].tile.cities[ci];
        while city.tokens.len() <= slot {
            city.tokens.push(None);
        }
        city.tokens[slot] = Some(tok);
        self.g1822_consume_reservation(&hex_id, ci, corp);
        if token_idx == 0 {
            self.gm().home_city.insert(corp.to_string(), (hex_id.clone(), ci));
        }
        self.clear_graph_cache();
        Ok(())
    }

    /// Token#remove! for the corp token placed in (hi, ci).
    pub(crate) fn g1822_remove_token_at(&mut self, hi: usize, ci: usize, corp: &str, token_type: Option<&str>) -> Option<Token> {
        let hex_id = self.hexes[hi].id.clone();
        let city = &mut self.hexes[hi].tile.cities[ci];
        let pos = city.tokens.iter().position(|t| {
            t.as_ref().is_some_and(|tok| {
                tok.corporation_id == corp && token_type.map_or(true, |ty| tok.token_type == ty)
            })
        })?;
        let tok = city.tokens[pos].take()?;
        if let Some(&cix) = self.corp_idx.get(corp) {
            // (a stamped token is that very token; else the first used one
            // of its type on the hex)
            if let Some(ct) = self.corporations[cix].tokens.iter_mut().find(|t| {
                if tok.serial != 0 {
                    t.serial == tok.serial
                } else {
                    t.used && t.city_hex_id == hex_id && t.token_type == tok.token_type
                }
            }) {
                ct.used = false;
                ct.city_hex_id.clear();
            }
        }
        self.clear_graph_cache();
        Some(tok)
    }

    /// The first unused token of a type (Operator#find_token_by_type).
    pub(crate) fn g1822_find_token(&self, corp: &str, ty: &str) -> Option<usize> {
        self.corporations[self.ci(corp)]
            .tokens
            .iter()
            .position(|t| !t.used && t.token_type == ty)
    }

    /// Where a corp's (first used) token of a type sits.
    pub(crate) fn g1822_token_city(&self, corp: &str, ty: Option<&str>) -> Option<(usize, usize)> {
        for (hi, hex) in self.hexes.iter().enumerate() {
            for (ci, city) in hex.tile.cities.iter().enumerate() {
                if city.tokens.iter().flatten().any(|t| t.corporation_id == corp && ty.map_or(true, |y| t.token_type == y)) {
                    return Some((hi, ci));
                }
            }
        }
        None
    }

    // -- home tokens --------------------------------------------------------

    /// G1822#place_home_token (+ base): at the start of a corp's turn.
    pub(crate) fn g1822_place_home_token(&mut self, corp: &str) {
        let cix = self.ci(corp);
        if self.corporations[cix].tokens.first().map_or(true, |t| t.used) {
            return;
        }
        if self.rnd().pending_tokens.iter().any(|p| p.corp == corp) {
            return;
        }
        let cd = self
            .title_def()
            .corporations()
            .into_iter()
            .find(|c| c.sym == corp)
            .expect("corp def");
        let Some(&hi) = self.hex_idx.get(cd.home_hex) else { return };
        let hex_id = cd.home_hex.to_string();
        // tile.reserved_by?(corp) && paths → the corp chooses the city.
        let tile_reserved = self.g1822_tile_res(&hex_id).iter().any(|r| r == corp);
        if tile_reserved && !self.hexes[hi].tile.paths.is_empty() {
            self.rndm().pending_tokens.push(PendingTok {
                corp: corp.to_string(),
                hexes: vec![hex_id],
                token_idx: 0,
            });
            return;
        }
        let n = self.hexes[hi].tile.cities.len();
        if n == 0 {
            return;
        }
        let ci = (0..n)
            .find(|&c| self.g1822_city_res(&hex_id, c).iter().any(|r| r.as_deref() == Some(corp)))
            .unwrap_or(0);
        let Some(tidx) = self.g1822_find_token(corp, "normal") else { return };
        // City#tokenable?(corporation, tokens: token) — not free.
        let price = self.corporations[cix].tokens[tidx].price;
        // multiple_tokens_allowed_on_home_hex? (1822MX: Mexico City's homes)
        let same_hex_allowed = self.g1822_is_mx();
        if self.g1822_city_tokenable(corp, hi, ci, false, same_hex_allowed) {
            // City#place_token spends nothing (1822 never reaches a priced
            // home token here: minor 14's London token is a pending one)
            // (1822MX's minor 14 has a £20 home token: placed free)
            if self.g1822_put_token(corp, tidx, hi, ci, false).is_ok()
                && price > 0
                && self.variant() == super::family::Variant::G1822
            {
                let _ = self.corp_spend(corp, price);
            }
        } else if self.g1822_is_pnw() {
            // PNW home_token_can_be_cheater
            let _ = self.g1822_put_token(corp, tidx, hi, ci, true);
        }
        // LNWR places its destination token too (no bonus until connected).
        if corp == self.fam().two_home_corporation {
            if let Some(dest) = self.g1822_destination(corp) {
                if let Some(t) = self.g1822_find_token(corp, "destination") {
                    let _ = self.g1822_place_destination(corp, &dest, t);
                }
            }
        }
    }

    /// place_destination_token: first city, free, unchecked, cheater.
    fn g1822_place_destination(&mut self, corp: &str, hex: &str, tidx: usize) -> Result<(), GameError> {
        let hi = *self.hex_idx.get(hex).ok_or_else(|| GameError::new("unknown hex"))?;
        self.g1822_put_token(corp, tidx, hi, 0, true)?;
        self.g1822_ruby_graph_clear();
        self.gm().destination_ability.retain(|c| c != corp);
        Ok(())
    }

    pub(crate) fn g1822_unplaced_destination_token(&self, corp: &str) -> Option<usize> {
        self.g1822_find_token(corp, "destination")
    }

    /// DestinationToken#process_hex_token (the record's token type is
    /// ignored; connectivity is not re-checked when loading).
    pub(crate) fn g1822_process_destination_token(&mut self, corp: &str, hex: &str) -> Result<(), GameError> {
        let tidx = self
            .g1822_unplaced_destination_token(corp)
            .ok_or_else(|| GameError::new("Corporation does not have a destination token unused"))?;
        if self.g1822_destination(corp).as_deref() != Some(hex) {
            return err(format!("{hex} is not {corp}'s destination"));
        }
        self.g1822_place_destination(corp, hex, tidx)?;
        let hi = self.hex_idx[hex];
        self.g1822_remove_extra_tokens(hi);
        Ok(())
    }

    /// G1822#remove_extra_tokens!(tile): a corp with several tokens in one
    /// city keeps one (destination tokens are never the ones removed;
    /// the home token is kept — removal pops the LAST by corp token order).
    pub(crate) fn g1822_remove_extra_tokens(&mut self, hi: usize) {
        if self.g1822_token_identity() {
            return self.g1822_remove_extra_tokens_by_identity(hi);
        }
        let n = self.hexes[hi].tile.cities.len();
        for ci in 0..n {
            let city = &self.hexes[hi].tile.cities[ci];
            let mut per_corp: Vec<(String, usize, usize)> = Vec::new(); // corp, removable count, total
            for t in city.tokens.iter().flatten() {
                match per_corp.iter_mut().find(|(c, _, _)| c == &t.corporation_id) {
                    Some(e) => {
                        e.2 += 1;
                        if t.token_type != "destination" {
                            e.1 += 1;
                        }
                    }
                    None => per_corp.push((
                        t.corporation_id.clone(),
                        (t.token_type != "destination") as usize,
                        1,
                    )),
                }
            }
            let mut removed_any = false;
            for (corp, _, total) in per_corp {
                if total <= 1 {
                    continue;
                }
                for _ in 0..(total - 1) {
                    // tokens_to_remove sorted by corp.tokens index; pop the
                    // last → the highest-index non-destination token.
                    let cix = self.ci(&corp);
                    let hex_id = self.hexes[hi].id.clone();
                    let idx = self.corporations[cix]
                        .tokens
                        .iter()
                        .enumerate()
                        .filter(|(_, t)| t.used && t.city_hex_id == hex_id && t.token_type != "destination")
                        .map(|(i, _)| i)
                        .last();
                    let Some(idx) = idx else { break };
                    let ty = self.corporations[cix].tokens[idx].token_type.clone();
                    // remove from the city (non-destination)
                    let city = &mut self.hexes[hi].tile.cities[ci];
                    if let Some(pos) = city.tokens.iter().rposition(|t| {
                        t.as_ref().is_some_and(|tok| tok.corporation_id == corp && tok.token_type == ty)
                    }) {
                        city.tokens[pos] = None;
                    }
                    self.corporations[cix].tokens[idx].used = false;
                    self.corporations[cix].tokens[idx].city_hex_id.clear();
                    self.corporations[cix].tokens[idx].price = self.fam().token_price;
                    removed_any = true;
                }
            }
            if removed_any {
                self.g1822_remove_cheater_slot(hi, ci);
            }
        }
        self.clear_graph_cache();
    }

    /// Whether map tokens carry their corporation token's identity (the
    /// variants whose rules see which token object sits where: 1822MX).
    pub(crate) fn g1822_token_identity(&self) -> bool {
        self.g1822_is_mx()
    }

    /// G1822#remove_extra_tokens! with token identity: of a corporation's
    /// non-destination tokens in one city, the ones latest in its token list
    /// (`sort_by corp.tokens.index`, popped) return to its charter at £100.
    fn g1822_remove_extra_tokens_by_identity(&mut self, hi: usize) {
        let n = self.hexes[hi].tile.cities.len();
        for ci in 0..n {
            let city = &self.hexes[hi].tile.cities[ci];
            // corp → (its non-destination tokens here: (slot, corp index), total)
            let mut per_corp: Vec<(String, Vec<(usize, usize)>, usize)> = Vec::new();
            for (slot, t) in city.tokens.iter().enumerate() {
                let Some(t) = t else { continue };
                let corp_idx = self.corp_idx.get(&t.corporation_id).and_then(|&cix| {
                    self.corporations[cix].tokens.iter().position(|ct| t.serial != 0 && ct.serial == t.serial)
                });
                let e = match per_corp.iter_mut().position(|(c, _, _)| *c == t.corporation_id) {
                    Some(i) => &mut per_corp[i],
                    None => {
                        per_corp.push((t.corporation_id.clone(), Vec::new(), 0));
                        per_corp.last_mut().unwrap()
                    }
                };
                e.2 += 1;
                if t.token_type != "destination" {
                    e.1.push((slot, corp_idx.unwrap_or(usize::MAX)));
                }
            }
            let mut removed_any = false;
            for (corp, mut toks, total) in per_corp {
                if total <= 1 {
                    continue;
                }
                toks.sort_by_key(|&(_, k)| k);
                for _ in 0..(total - 1) {
                    let Some((slot, k)) = toks.pop() else { break };
                    self.hexes[hi].tile.cities[ci].tokens[slot] = None;
                    let cix = self.ci(&corp);
                    let price = self.fam().token_price;
                    if let Some(ct) = self.corporations[cix].tokens.get_mut(k) {
                        ct.used = false;
                        ct.city_hex_id.clear();
                        ct.price = price;
                    }
                    removed_any = true;
                }
            }
            if removed_any {
                self.g1822_remove_cheater_slot(hi, ci);
            }
        }
        self.clear_graph_cache();
    }

    /// remove_cheater_slot!(city).
    fn g1822_remove_cheater_slot(&mut self, hi: usize, ci: usize) {
        let hex_id = self.hexes[hi].id.clone();
        let res = self.g1822_city_res(&hex_id, ci);
        let city = &mut self.hexes[hi].tile.cities[ci];
        if city.tokens.len() <= city.slots as usize {
            return;
        }
        let Some(empty) = (0..city.tokens.len()).find(|&i| city.tokens[i].is_none() && res.get(i).map_or(true, |r| r.is_none())) else {
            return;
        };
        if let Some(last) = city.tokens.pop() {
            if empty < city.tokens.len() {
                city.tokens[empty] = last;
            }
        }
    }

    // -- exchange tokens ------------------------------------------------------

    pub(crate) fn g1822_exchange_tokens(&self, corp: &str) -> u32 {
        self.g().exchange_tokens.get(corp).copied().unwrap_or(0)
    }

    /// remove_exchange_token (ability.use!).
    pub(crate) fn g1822_remove_exchange_token(&mut self, corp: &str) {
        if let Some(n) = self.gm().exchange_tokens.get_mut(corp) {
            *n = n.saturating_sub(1);
        }
    }

    /// move_exchange_token: one exchange token becomes a normal £100 token.
    pub(crate) fn g1822_move_exchange_token(&mut self, corp: &str) {
        self.g1822_remove_exchange_token(corp);
        let cix = self.ci(corp);
        let price = self.fam().token_price;
        self.corporations[cix].tokens.push(Token::new(corp.to_string(), price));
    }

    // -- steps --------------------------------------------------------------

    /// PendingToken#process_place_token.
    pub(crate) fn g1822_process_pending_token(&mut self, hex_id: &str, city_index: u8) -> Result<(), GameError> {
        let p = self.rnd().pending_tokens.first().cloned().ok_or_else(|| GameError::new("no pending token"))?;
        let hex = self.g1822_resolve_hex(hex_id)?;
        if !p.hexes.contains(&hex) {
            return err(format!("Cannot place token on {hex} as the hex is not available"));
        }
        let hi = self.hex_idx[&hex];
        let ci = city_index as usize;
        if ci >= self.hexes[hi].tile.cities.len() {
            return err("Invalid city");
        }
        if self.g1822_city_tokened_by(hi, ci, &p.corp) {
            return err(format!("{} cant have 2 tokens in the same city", p.corp));
        }
        if p.corp == self.fam().minor_14_id {
            self.hexes[hi].tile.cities[ci].tokens.push(None);
        }
        if self.g1822_get_slot(hi, ci, &p.corp, false).is_none() {
            self.hexes[hi].tile.cities[ci].tokens.push(None);
        }
        let tidx = p.token_idx;
        let price = self.corporations[self.ci(&p.corp)].tokens[tidx].price;
        // Tokener#place_token (connected: false for M14, extra action,
        // unchecked) — the token ability price/consumption applies.
        let price = self.g1822_token_ability_price(&p.corp, &hex, price);
        self.g1822_put_token(&p.corp, tidx, hi, ci, false)?;
        self.g1822_ruby_graph_clear();
        if price > 0 {
            self.corp_spend(&p.corp, price)?;
        }
        self.rndm().pending_tokens.remove(0);
        if hex == self.fam().minor_14_home_hex {
            // after_place_pending_token: remember the city's exit edge.
            let exit = self.hexes[hi].tile.paths.iter().find_map(|pd| match (&pd.a, &pd.b) {
                (crate::tiles::PathEndpoint::City(c), crate::tiles::PathEndpoint::Edge(e))
                | (crate::tiles::PathEndpoint::Edge(e), crate::tiles::PathEndpoint::City(c))
                    if *c == ci =>
                {
                    Some(*e)
                }
                _ => None,
            });
            self.gm().minor_14_city_exit = exit;
        }
        Ok(())
    }

    /// adjust_token_price_ability!: the corp's own token abilities (SECR's
    /// free English Channel token; the £20 any-hex token a major gets from
    /// acquiring minor 14 from the bank) — consumed when they apply.
    fn g1822_token_ability_price(&mut self, corp: &str, hex: &str, price: i32) -> i32 {
        for (i, hexes, ab_price) in self.g1822_usable_token_abilities(corp) {
            if !hexes.is_empty() && !hexes.iter().any(|h| h == hex) {
                continue;
            }
            let reachable = self
                .hex_idx
                .get(hex)
                .map_or(false, |&hi| self.g1822_graph(corp).reachable_hexes.contains(&hi));
            let new_price = if reachable { ab_price } else { price };
            if let Some(v) = self.gm().corp_token_abilities.get_mut(corp) {
                v.remove(i);
            }
            return new_price;
        }
        price
    }

    /// Resolve a PlaceToken hex id (the `__tile:<instance>` encoding).
    pub(crate) fn g1822_resolve_hex(&self, hex_id: &str) -> Result<String, GameError> {
        if let Some(inst) = hex_id.strip_prefix("__tile:") {
            let base = inst.split('-').next().unwrap_or(inst);
            // Preprinted: "<HEX>-0" → hex; laid tile instance "57-3".
            if self.hex_idx.contains_key(base) {
                return Ok(base.to_string());
            }
            return self
                .hexes
                .iter()
                .find(|h| h.tile.id == inst)
                .map(|h| h.id.clone())
                .ok_or_else(|| GameError::new(format!("No hex with tile {inst}")));
        }
        Ok(hex_id.to_string())
    }

    /// G1822 Token#actions.
    pub(crate) fn g1822_token_actions(&self, ent: &Ent) -> Vec<&'static str> {
        let mut a: Vec<&'static str> = Vec::new();
        match ent {
            Ent::Corp(c) if Some(c) == self.g1822_operator().as_ref() => {
                if self.g1822_can_place_token(c) {
                    a.push("place_token");
                    a.push("pass");
                }
                if a.is_empty()
                    && (self.g1822_owns(c, self.fam().company_chpr) || self.g1822_owns(c, self.fam().company_lcdr))
                    && self.g1822_exchange_tokens(c) > 0
                {
                    a.push("choose_ability");
                }
            }
            Ent::Company(co) => {
                if !self.g1822_company_choices(co, "token").is_empty() {
                    a.push("choose_ability");
                }
            }
            _ => {}
        }
        if !a.is_empty() && !a.contains(&"pass") {
            a.push("pass");
        }
        a
    }

    /// The corporation's own :token abilities that Ruby's `abilities(corp,
    /// :token)` still returns: `ability_usable?` drops a hex-bound token
    /// ability once the corporation has a token in every one of its hexes
    /// (SECR's free English Channel token after P5 placed SECR there).
    /// (index into `corp_token_abilities`, hexes, price)
    pub(crate) fn g1822_usable_token_abilities(&self, c: &str) -> Vec<(usize, Vec<String>, i32)> {
        let Some(v) = self.g().corp_token_abilities.get(c) else { return Vec::new() };
        let tokened: Vec<&str> = self
            .corp_idx
            .get(c)
            .map(|&i| {
                self.corporations[i]
                    .tokens
                    .iter()
                    .filter(|t| t.used)
                    .map(|t| t.city_hex_id.as_str())
                    .collect()
            })
            .unwrap_or_default();
        v.iter()
            .enumerate()
            .filter(|(_, (hexes, _))| hexes.is_empty() || hexes.iter().any(|h| !tokened.contains(&h.as_str())))
            .map(|(i, (hexes, p))| (i, hexes.clone(), *p))
            .collect()
    }

    /// P5's `token` ability as `abilities(corp, :token)` sees it (Corporation
    /// #all_abilities includes owned companies'): live, the company open
    /// and corporation-owned, and `ability_usable?` — the corporation has
    /// no token in the English Channel yet. (Its `when: token` holds
    /// whenever the Token step is the one asking.)
    pub(crate) fn g1822_lcdr_token_usable(&self, c: &str) -> bool {
        let lcdr = self.fam().company_lcdr;
        if !self.g().lcdr_token_ability || !self.g1822_ability_owner_ok(lcdr) {
            return false;
        }
        if self.co_owner(lcdr).corp_sym() != Some(c) {
            return false;
        }
        let Some(&cix) = self.corp_idx.get(c) else { return false };
        !self.corporations[cix]
            .tokens
            .iter()
            .any(|t| t.used && t.city_hex_id == self.fam().english_channel_hex)
    }

    /// G1822 Token#can_place_token? (+ Tokener's).
    fn g1822_can_place_token(&self, c: &str) -> bool {
        // MX Token#can_place_token?: never the NdeM
        if self.g1822_mx_is_ndem(c) {
            return false;
        }
        let cix = self.ci(c);
        let available: Vec<&Token> = self.corporations[cix]
            .tokens
            .iter()
            .filter(|t| !t.used && t.token_type == "normal")
            .collect();
        if available.is_empty() || self.rnd().tokened {
            return false;
        }
        let usable = self.g1822_usable_token_abilities(c);
        if !usable.is_empty() || self.g1822_lcdr_token_usable(c) {
            return true;
        }
        let min_price = available[0].price;
        let mut prices = vec![min_price];
        for (_, _, p) in usable {
            prices.push(p);
        }
        let min = prices.into_iter().min().unwrap_or(min_price);
        min <= self.corporations[cix].cash && self.g1822_can_token(c)
    }

    pub(crate) fn g1822_token_process(&mut self, i: usize, ent: &Ent, action: &Action) -> Result<(), GameError> {
        match action {
            Action::PlaceToken { hex_id, city_index, .. } => {
                let Ent::Corp(c) = ent else { return err("token by a non-corporation") };
                let hex = self.g1822_resolve_hex(hex_id)?;
                let hi = *self.hex_idx.get(&hex).ok_or_else(|| GameError::new("unknown hex"))?;
                let ci = *city_index as usize;
                if ci >= self.hexes[hi].tile.cities.len() {
                    return err("Invalid city");
                }
                if self.is_major(c) && self.g1822_city_tokened_by(hi, ci, c) {
                    return err(format!("{c} cant have 2 tokens in the same city"));
                }
                if self.rnd().tokened {
                    return err("Token already placed this turn");
                }
                let tidx = self
                    .g1822_find_token(c, "normal")
                    .ok_or_else(|| GameError::new("no tokens"))?;
                let base_price = self.corporations[self.ci(c)].tokens[tidx].price;
                let price = self.g1822_token_ability_price(c, &hex, base_price);
                // Tokener#check_connected computes the corporation's graph.
                self.g1822_note_graph(c);
                let check = hex != self.fam().london_hex;
                if check && !self.g1822_city_tokenable(c, hi, ci, price <= 0, false) {
                    return err(format!("{c} cannot lay token on {hex}"));
                }
                self.g1822_put_token(c, tidx, hi, ci, false)?;
                self.g1822_ruby_graph_clear();
                if price > 0 {
                    self.corp_spend(c, price)?;
                }
                self.rndm().tokened = true;
                self.rndm().passed[i] = true;
                Ok(())
            }
            Action::ChooseAbility { choice, .. } => {
                let Ent::Company(co) = ent else { return err("choose_ability by a non-company") };
                self.g1822_company_made_choice(co, choice, "token")
            }
            Action::Pass { .. } => {
                self.rndm().passed[i] = true;
                Ok(())
            }
            _ => err("Token cannot process this action"),
        }
    }

    /// Ruby SpecialToken#can_token_after_teleport?.
    pub(crate) fn g1822_can_token_after_teleport(&self) -> bool {
        let Some(co) = &self.rnd().teleported else { return false };
        // available_tokens(LCDR): an exchange token of its owner.
        self.co_owner(co)
            .corp_sym()
            .map_or(false, |c| self.g1822_exchange_tokens(c) > 0)
    }

    /// G1822 SpecialToken#actions: P5's English Channel exchange token,
    /// while the active step can place tokens.
    pub(crate) fn g1822_special_token_actions(&self, ent: &Ent) -> Vec<&'static str> {
        let Ent::Company(co) = ent else { return vec![] };
        if co != self.fam().company_lcdr || self.co_closed(co) {
            return vec![];
        }
        let Some(owner) = self.co_owner(co).corp_sym().map(str::to_string) else { return vec![] };
        // ability(entity): the token ability, or a USED teleport.
        let has_token_ability = self.g1822_lcdr_token_usable(&owner);
        let used_teleport = self.rnd().teleported.as_deref() == Some(co.as_str());
        if !(has_token_ability || used_teleport) {
            return vec![];
        }
        if self.g1822_exchange_tokens(&owner) == 0 {
            return vec![];
        }
        let active_places = self.g1822_active_step().map_or(false, |a| {
            matches!(
                self.rnd().steps[a],
                super::Step::Token | super::Step::PendingToken | super::Step::SpecialToken
            )
        });
        if !active_places {
            return vec![];
        }
        let mut a = vec!["place_token"];
        if used_teleport {
            a.push("pass");
        }
        a
    }

    pub(crate) fn g1822_special_token_process(&mut self, ent: &Ent, action: &Action) -> Result<(), GameError> {
        let Ent::Company(co) = ent else { return err("special token by a non-company") };
        let owner = self
            .co_owner(co)
            .corp_sym()
            .map(str::to_string)
            .ok_or_else(|| GameError::new("company not corporation-owned"))?;
        match action {
            Action::PlaceToken { hex_id, city_index, .. } => {
                let hex = self.g1822_resolve_hex(hex_id)?;
                if hex != self.fam().english_channel_hex {
                    return err(format!("Cannot place token on {hex}"));
                }
                let hi = self.hex_idx[&hex];
                let ci = *city_index as usize;
                // A fresh token (from_owner false), free.
                let cix = self.ci(&owner);
                self.corporations[cix].tokens.push(Token::new(owner.clone(), 0));
                let tidx = self.corporations[cix].tokens.len() - 1;
                self.g1822_put_token(&owner, tidx, hi, ci, false)?;
                self.g1822_ruby_graph_clear();
                if self.rnd().teleported.is_none() {
                    // the token ability is used up (count 1)
                    self.gm().lcdr_token_ability = false;
                }
                self.rndm().teleported = None;
                self.g1822_remove_exchange_token(&owner);
                self.close_company(co);
                Ok(())
            }
            Action::Pass { .. } => {
                self.rndm().teleported = None;
                Ok(())
            }
            _ => err("SpecialToken cannot process this action"),
        }
    }
}
