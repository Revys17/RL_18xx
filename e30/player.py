import logging
import random
from typing import List

import e30
from e30.actions.stock_market_buy_action import StockMarketBuyAction
from e30.actions.stock_market_sell_action import StockMarketSellAction

log = logging.getLogger(__name__)

RANDOM_NAMES: List[str] = ['CHIMERA_ANT_ARC_LOVER', 'GREED_ISLAND_ARC_LOVER', 'HEAVENS_ARENA_ARC_LOVER',
                           'ZOLDYCK_FAMILY_ARC_LOVER', 'YORKNEW_CITY_ARC_LOVER', 'ELECTION_ARC_LOVER']


class Player:
    def __init__(self, index: int, starting_money: int, agent: 'e30.agent.Agent'):
        self.index: int = index
        self.money: int = starting_money
        self.shares: List = []
        self.charters: List = []
        self.privates: List[e30.private_company.Private] = []
        self.name: str = random.choice(RANDOM_NAMES) + str(self.index)
        self.agent = agent

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
        # TODO
        return True

    def sell_shares(self, company: 'e30.company.Company', num_shares: int) -> None:
        pass

    def set_aside_money(self, bid: int) -> None:
        self.remove_money(bid)

    def return_money(self, money: int) -> None:
        self.add_money(money)

    def remove_money(self, money: int) -> None:
        # TODO: is it possible to go negative? Should any actions or considerations be taken into account if that happens?
        self.money -= money

    def get_stock_market_sell_action(self, game_state: 'e30.game_state.GameState') -> StockMarketSellAction:
        return self.agent.get_stock_market_sell_action(game_state)

    def get_stock_market_buy_action(self, game_state: 'e30.game_state.GameState') -> StockMarketBuyAction:
        return self.agent.get_stock_market_buy_action(game_state)
