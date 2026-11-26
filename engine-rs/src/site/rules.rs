//! The engine half of site mode (docs/SITE_ENGINE_RS.md §4.2): helpers the `site_rules` branches in the engine
//! call. Compiled in every build (the branches are in the engine's own files); only ever reached with
//! `BaseGame::site_rules` on, which nothing but the site layer sets.

use crate::actions::GameError;
use crate::entities::EntityId;
use crate::game::BaseGame;
use crate::title::AbilityDef;

impl BaseGame {
    /// Ruby `Engine::Game::G1830::Step::BuySellParShares#purchasable_companies` (with
    /// `Engine::Game::Base#purchasable_companies`): a player may buy another player's private in a stock round after
    /// the first, while the phase has `can_buy_companies_from_other_players` (1830's phases 2-4), before buying
    /// anything this turn, with cash. Ruby's step then lists `buy_company` and waits for the player (it never skips
    /// them). The RL engine's own rule never offers or expects this purchase; only site mode asks.
    pub fn site_interplayer_purchase_open(&self, player_id: u32, bought: bool) -> bool {
        if self.title != "1830" || bought || self.turn == 1 {
            return false;
        }
        if !matches!(self.phase.name.as_str(), "2" | "3" | "4") {
            return false;
        }
        let me = EntityId::player(player_id);
        let cash = self.player_index(player_id).map_or(0, |i| self.players[i].cash);
        cash > 0
            && self.companies.iter().any(|c| {
                !c.closed && c.owner.is_player() && c.owner != me && !crate::abilities::no_buy(&self.title, &c.sym)
            })
    }

    /// Ruby `Engine::Game::Base#after_buy_company`, which `Engine::Step::BuySellParShares#process_buy_company` calls
    /// for a player who buys a private (here: another player's): the private's `shares` ability runs again, as at
    /// its first purchase (1830's C&A: the PRR share it named, PRR_1), through `SharePool#buy_shares(player, share,
    /// exchange: :free)`. A share a player holds can't be taken ('Cannot buy share from player': Ruby refuses the
    /// whole purchase, so this answers before anything moves); one in the market or the IPO goes to the buyer for
    /// nothing, and the presidency follows (`transfer_shares`). A president's share (B&O's, `companies_pending_par`)
    /// is left alone: B&O can't be bought between players (`no_buy`). Called only with `site_rules`, before the
    /// purchase itself is applied (its cash and the private's owner don't bear on the share).
    pub fn site_after_buy_company(&mut self, company_sym: &str, player_id: u32) -> Result<(), GameError> {
        let grants: Vec<(&'static str, u8)> = crate::abilities::company_abilities(&self.title, company_sym)
            .iter()
            .filter_map(|a| match a {
                AbilityDef::Shares { corporation, share_index } if *share_index > 0 => Some((*corporation, *share_index)),
                _ => None,
            })
            .collect();
        let mut moves = Vec::new();
        for (corp, index) in grants {
            let Some(&ci) = self.corp_idx.get(corp) else { continue };
            let Some(si) = self.corporations[ci].shares.iter().position(|s| s.index == index as usize) else { continue };
            let owner = &self.corporations[ci].shares[si].owner;
            if owner.is_player() {
                return Err(GameError::new("Cannot buy share from player"));
            }
            moves.push((ci, si));
        }
        let me = EntityId::player(player_id);
        for (ci, si) in moves {
            let pre: Vec<EntityId> = self.corporations[ci].shares.iter().map(|s| s.owner.clone()).collect();
            self.corporations[ci].set_share_owner(si, me.clone());
            self.check_president_change_with_snapshot(ci, None, pre);
        }
        Ok(())
    }
}
