import logging
import random
from typing import List, Dict, Set

import e30
from e30.actions.stock_market_buy_action import StockMarketBuyAction
from e30.actions.stock_market_sell_action import StockMarketSellAction, StockMarketSellActionType
from e30.exceptions.exceptions import InvalidOperationException

log = logging.getLogger(__name__)

RANDOM_NAMES: List[str] = ['CHIMERA_ANT_ARC_LOVER', 'GREED_ISLAND_ARC_LOVER', 'HEAVENS_ARENA_ARC_LOVER',
                           'ZOLDYCK_FAMILY_ARC_LOVER', 'YORKNEW_CITY_ARC_LOVER', 'ELECTION_ARC_LOVER']


class Player:
    def __init__(self, index: int, starting_money: int, agent: 'e30.agent.Agent'):
        self.index: int = index
        self.money: int = starting_money
        # company short name to # shares owned, this doesn't include presidency certs, implied when a company's
        # short name is in the player's 'presiding_companies' list
        self.share_map: Dict[str, int] = {}
        self.presiding_companies: List[str] = []
        self.privates: List[e30.private_company.Private] = []
        self.name: str = random.choice(RANDOM_NAMES) + str(self.index)
        self.agent = agent
        # sold companies cannot be bought by a player in the same stock round
        self.buy_restricted_companies: Set[str] = set()

    def get_bid_buy_action(self, game_state: 'e30.game_state.GameState') -> e30.actions.bid_buy_action.BidBuyAction:
        # needs to return the action the player wants to take
        # this is how we link to our AI
        log.info("Current bid buy player turn: {}".format(self.name))
        log.info("Available money: {}".format(self.money))
        return self.agent.get_bid_buy_action(game_state)

    def get_bid_resolution_action(self, game_state: 'e30.game_state.GameState') -> \
            e30.actions.bid_resolution_action.BidResolutionAction:
        # needs to return the action the player wants to take
        # this is how we link to our AI
        log.info("Current bid resolution player turn: {}".format(self.name))
        log.info("Available money: {}".format(self.money))
        return self.agent.get_bid_resolution_action(game_state)

    def pay_private_income(self) -> None:
        for private in self.privates:
            log.info("Player {} gained {} from private company {}".format(self.name, private.revenue,
                                                                          private.short_name))
            self.money += private.revenue

    def add_money(self, money: int) -> None:
        self.money += money

    def add_share(self, company: 'e30.company.Company') -> None:
        pass

    def get_name(self) -> str:
        return self.name

    def has_share(self, company: 'e30.company.Company') -> bool:
        if company.short_name in self.share_map and self.share_map[company.short_name] > 0:
            return True
        return False

    def sell_shares(self, company: 'e30.company.Company', num_shares: int) -> int:
        value_gained = company.current_share_price.get_value()[0] * num_shares
        self.add_money(value_gained)
        self.share_map[company.short_name] -= num_shares
        if self.share_map[company.short_name] == 0:
            del self.share_map[company.short_name]
            company.owning_players.remove(self)
        return value_gained

    def set_aside_money(self, bid: int) -> None:
        self.remove_money(bid)

    def return_money(self, money: int) -> None:
        self.add_money(money)

    def remove_money(self, money: int) -> None:
        # TODO: is it possible to go negative? Should any actions or considerations be taken into account if that happens?
        self.money -= money

    def get_stock_market_sell_action(self, game_state: 'e30.game_state.GameState') -> StockMarketSellAction:
        sell_action: StockMarketSellAction = self.agent.get_stock_market_sell_action(game_state)
        valid_action_types = [t for t in StockMarketSellActionType]
        if sell_action.action_type not in valid_action_types:
            raise InvalidOperationException(f"Invalid sell action type {sell_action.action_type}")
        if sell_action.action_type is StockMarketSellActionType.SELL and len(sell_action.sell_map.keys()) == 0:
            raise InvalidOperationException(f"Sell action with empty sell map")
        return sell_action

    def get_stock_market_buy_action(self, game_state: 'e30.game_state.GameState') -> StockMarketBuyAction:
        return self.agent.get_stock_market_buy_action(game_state)

    def reset_restricted_companies_buy_list(self):
        self.buy_restricted_companies.clear()
