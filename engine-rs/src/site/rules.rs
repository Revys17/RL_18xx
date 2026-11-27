//! The engine half of site mode (docs/SITE_ENGINE_RS.md §4.2): helpers the `site_rules` branches in the engine
//! call. Compiled in every build (the branches are in the engine's own files); only ever reached with
//! `BaseGame::site_rules` on, which nothing but the site layer sets.

use crate::actions::GameError;
use crate::entities::EntityId;
use crate::game::BaseGame;
use crate::title::AbilityDef;

/// Ruby's phase status flags the RL title data leaves out (only site mode reads them), by title and phase: the site's
/// Ruby PHASES (g_1830/game.rb: phase 2 `['can_buy_companies_from_other_players']`, 3 and 4 `%w[can_buy_companies
/// can_buy_companies_from_other_players]`; `can_buy_companies` is in the engine's own `PhaseDef.status`). 1867 has
/// none. Not added to `PhaseDef.status`: the token encoder reads every phase status into its features
/// (docs/SITE_ENGINE_RS_1867.md §5.2).
const RUBY_SITE_PHASE_STATUS: &[(&str, &str, &[&str])] = &[
    ("1830", "2", &["can_buy_companies_from_other_players"]),
    ("1830", "3", &["can_buy_companies_from_other_players"]),
    ("1830", "4", &["can_buy_companies_from_other_players"]),
];

/// The titles whose BuySellParShares refuses a purchase between players in the first stock round
/// (`G1830::Step::BuySellParShares#purchasable_companies`' `@game.turn == 1`).
const NO_PURCHASE_ON_TURN_ONE: &[&str] = &["1830"];

impl BaseGame {
    /// Ruby's phase status now: the engine's `PhaseDef.status` and the flags only the site reads.
    pub(crate) fn ruby_phase_status(&self) -> Vec<&'static str> {
        let mut out: Vec<&'static str> = self
            .title_def()
            .phases()
            .iter()
            .find(|p| p.name == self.phase.name)
            .map(|p| p.status.to_vec())
            .unwrap_or_default();
        for (title, phase, flags) in RUBY_SITE_PHASE_STATUS {
            if *title == self.title && *phase == self.phase.name {
                for f in *flags {
                    if !out.contains(f) {
                        out.push(f);
                    }
                }
            }
        }
        out
    }

    /// Ruby `Engine::Step::BuySellParShares#purchasable_companies` (with `Engine::Game::Base#purchasable_companies`;
    /// 1830's step adds `@game.turn == 1`): a player may buy another player's private in a stock round while the
    /// phase's status has `can_buy_companies_from_other_players` (Ruby's phase data, `ruby_phase_status`), before
    /// buying anything this turn, with cash. Ruby's step then lists `buy_company` and waits for the player (it never
    /// skips them). The RL engine's own rule never offers or expects this purchase; only site mode asks.
    pub fn site_interplayer_purchase_open(&self, player_id: u32, bought: bool) -> bool {
        if bought || (self.turn == 1 && NO_PURCHASE_ON_TURN_ONE.contains(&self.title.as_str())) {
            return false;
        }
        if !self.ruby_phase_status().contains(&"can_buy_companies_from_other_players") {
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
