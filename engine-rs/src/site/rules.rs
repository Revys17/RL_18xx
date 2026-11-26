//! The engine half of site mode (docs/SITE_ENGINE_RS.md §4.2): helpers the `site_rules` branches in the engine
//! call. Compiled in every build (the branches are in the engine's own files); only ever reached with
//! `BaseGame::site_rules` on, which nothing but the site layer sets.

use crate::entities::EntityId;
use crate::game::BaseGame;

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
}
