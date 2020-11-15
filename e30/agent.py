import logging
from typing import List

from e30.actions.bid_buy_action import BidBuyAction, BidBuyActionType
from e30.actions.bid_resolution_action import BidResolutionAction, BidResolutionActionType
from e30.actions.stock_market_buy_action import StockMarketBuyAction
from e30.actions.stock_market_sell_action import StockMarketSellAction
from e30.exceptions.exceptions import InvalidOperationException
import e30.game_state
from e30.private_company import Private

log = logging.getLogger(__name__)


class Agent(object):
    def get_bid_buy_action(self, game_state: e30.game_state.GameState) -> BidBuyAction:
        pass

    def get_bid_resolution_action(self, game_state: e30.game_state.GameState):
        pass

    def get_stock_market_buy_action(self, game_state: e30.game_state.GameState):
        pass

    def get_stock_market_sell_action(self, game_state: e30.game_state.GameState):
        pass


class HumanAgent(Agent):
    def get_bid_buy_action(self, game_state: e30.game_state.GameState) -> BidBuyAction:
        game_state.print_bid_buy_turn_game_state()

        directions: str = """
        Bid buy turn.
        Your turn to bid/buy/pass.
        '1' to give up your turn and pass.
        '2' to buy the lowest face value private company
        '3 (index) (bid amount)' to bid for an unowned private company
        """
        max_private_company_index: int = len(game_state.privates) - 1
        while True:
            user_input: str = input(directions)
            user_input_split: List = user_input.split(" ")
            if len(user_input_split) < 1:
                log.error("Invalid input, try again")
                continue
            action: str = user_input_split[0]
            if action.lower().startswith("1"):
                return BidBuyAction(BidBuyActionType.PASS)
            elif action.lower().startswith("2"):
                return BidBuyAction(BidBuyActionType.BUY)
            elif action.lower().startswith("3"):
                if len(user_input_split) >= 3:
                    try:
                        private_company_index: int = int(user_input_split[1])
                        bid_amount: int = int(user_input_split[2])
                        if private_company_index < 0 or private_company_index > max_private_company_index:
                            raise InvalidOperationException("Invalid private company index to bid on: " +
                                                            str(private_company_index))
                        private: Private = game_state.privates[private_company_index]
                        if bid_amount < private.price + 5 or bid_amount < private.current_winning_bid + 5:
                            log.info("Private company price: {}, current winning bid: {}"
                                     .format(private.price, private.current_winning_bid))
                            raise InvalidOperationException("Invalid bid amount: " + str(bid_amount))
                        return BidBuyAction(BidBuyActionType.BID, private_company_index, bid_amount)
                    except Exception as e:
                        log.error(e)

            log.error("Invalid input, try again")

    def get_bid_resolution_action(self, game_state: e30.game_state.GameState) -> BidResolutionAction:
        game_state.print_bid_resolution_turn_game_state()

        directions: str = """
        Bid buy resolution.
        Your turn to bid/pass.
        '1' to give up your turn and pass.
        '2 (bid amount)' to set the winning bid for the private company 
        """
        while True:
            user_input: str = input(directions)
            user_input_split: List = user_input.split(" ")
            if len(user_input_split) < 1:
                log.error("Invalid input, try again")
                continue
            action: str = user_input_split[0]
            if action.lower().startswith("1"):
                return BidResolutionAction(BidResolutionActionType.PASS)
            elif action.lower().startswith("2"):
                if len(user_input_split) >= 2:
                    try:
                        bid_amount: int = int(user_input_split[1])
                        private: Private = game_state.privates[0]
                        if bid_amount < private.price + 5 or bid_amount < private.current_winning_bid + 5:
                            log.info("Private company price: {}, current winning bid: {}"
                                     .format(private.price, private.current_winning_bid))
                            raise InvalidOperationException("Invalid bid amount: " + str(bid_amount))
                        return BidResolutionAction(BidResolutionActionType.BID, bid_amount)
                    except Exception as e:
                        log.error(e)

            log.error("Invalid input, try again")

    def get_stock_market_buy_action(self, game_state: e30.game_state.GameState) -> StockMarketBuyAction:
        directions: str = """
        Stock market buy action.
        Your turn to buy 1 company certificate or pass.
        """
        user_input: str = input(directions)
        return StockMarketBuyAction()

    def get_stock_market_sell_action(self, game_state: e30.game_state.GameState) -> StockMarketSellAction:
        directions: str = """
        Stock market sell action.
        Your turn to sell company certificates or pass.
        """
        user_input: str = input(directions)
        return StockMarketSellAction()


class AIAgent(Agent):
    def get_bid_buy_action(self, game_state: e30.game_state.GameState):
        pass

    def get_bid_resolution_action(self, game_state: e30.game_state.GameState):
        pass

    def get_stock_market_buy_action(self, game_state: e30.game_state.GameState):
        pass

    def get_stock_market_sell_action(self, game_state: e30.game_state.GameState):
        pass
