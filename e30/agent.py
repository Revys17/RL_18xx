import logging
from collections import OrderedDict
from typing import List

from e30.actions.bid_buy_action import BidBuyAction, BidBuyActionType
from e30.actions.bid_resolution_action import BidResolutionAction, BidResolutionActionType
from e30.actions.stock_market_buy_action import StockMarketBuyAction, StockMarketBuyActionType, \
    StockMarketBuyActionBuyType
from e30.actions.stock_market_sell_action import StockMarketSellAction, StockMarketSellActionType
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

    def get_bo_par_value_action(self, game_state: e30.game_state.GameState):
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
        Your turn to pass or buy a president cert, an IPO share, a market share, or any number of market shares if the
        stock value is brown
        '1' to give up your turn and pass.
        '2 <company shortname> <X>' to buy the president cert  
        '3 <company shortname> <X>' to buy X bank pool shares
        '4 <company shortname>' to buy 1 IPO share
        """
        while True:
            user_input: str = input(directions)
            user_input_split: List = user_input.split(" ")
            if len(user_input_split) < 1:
                log.error("Invalid input, try again")
                continue
            action: str = user_input_split[0]
            if action.lower().startswith("1"):
                return StockMarketBuyAction()
            elif action.lower().startswith("2") or action.lower().startswith("3"):
                if len(user_input_split) != 3:
                    log.error("Invalid input, try again")
                    continue
                company_name: str = user_input_split[1]
                num: int = int(user_input_split[2])
                if company_name not in game_state.companies_map:
                    log.error(f"Invalid company name {company_name}, try again")
                    continue
                if action.lower().startswith("2"):
                    if num not in game_state.stock_market.par_locations:
                        log.error(f"Invalid par value {num} must be in "
                                  f"{list(game_state.stock_market.par_locations.keys())}, try again")
                        continue
                    return StockMarketBuyAction(StockMarketBuyActionType.BUY, company_name,
                                                StockMarketBuyActionBuyType.PRESIDENT_CERT, 1, num)
                else:
                    if num > 8:
                        log.error(f"Cannot purchase {num} market shares, try again")
                        continue
                    return StockMarketBuyAction(StockMarketBuyActionType.BUY, company_name,
                                                StockMarketBuyActionBuyType.BANK_POOL, num)
            elif action.lower().startswith("4"):
                if len(user_input_split) != 2:
                    log.error("Invalid input, try again")
                    continue
                company_name: str = user_input_split[1]
                if company_name not in game_state.companies_map:
                    log.error(f"Invalid company name {company_name}, try again")
                    continue

                return StockMarketBuyAction(StockMarketBuyActionType.BUY, company_name,
                                            StockMarketBuyActionBuyType.IPO, 1)
            log.error("Invalid input, try again")

    def get_stock_market_sell_action(self, game_state: e30.game_state.GameState) -> StockMarketSellAction:
        directions: str = """
        Stock market sell action.
        Your turn to sell any owned certs or pass.
        '1' to give up your turn and pass.
        '2 [<company shortname>:<num certs to sell>,...]' to specify companies and num certs to sell, in that order 
        """
        while True:
            user_input: str = input(directions)
            user_input_split: List = user_input.split(" ")
            if len(user_input_split) < 1:
                log.error("Invalid input, try again")
                continue
            action: str = user_input_split[0]
            if action.lower().startswith("1"):
                return StockMarketSellAction()
            elif action.lower().startswith("2"):
                if len(user_input_split) != 2:
                    log.error("Invalid input, try again")
                    continue
                # comma separated pairs
                company_cert_pairs = user_input_split[1].split(",")
                sell_map: OrderedDict[str, int] = {}
                for pair in company_cert_pairs:
                    # colon separated key value pair
                    key_value_pair = pair.split(":")
                    if len(key_value_pair) != 2:
                        log.error("Invalid key value pair, try again")
                        continue
                    key = key_value_pair[0]
                    value = int(key_value_pair[1])
                    if value < 1 or value > 8:
                        log.error(f"Cannot sell {value} shares")
                        continue
                    sell_map.update({key: value})
                return StockMarketSellAction(StockMarketSellActionType.SELL, sell_map)

            log.error("Invalid input, try again")

    def get_bo_par_value_action(self, game_state: e30.game_state.GameState) -> int:
        directions: str = """
        As the owner of the private BO, you start with the president cert of B&O.
        Set a par value for B&O.
        '<X>' to set X as the par value 
        """
        while True:
            par_value: int = int(input(directions))
            if par_value not in game_state.stock_market.par_locations:
                log.error(f"Invalid par value {par_value} must be in "
                          f"{list(game_state.stock_market.par_locations.keys())}, try again")
                continue
            return par_value


class AIAgent(Agent):
    def get_bid_buy_action(self, game_state: e30.game_state.GameState):
        pass

    def get_bid_resolution_action(self, game_state: e30.game_state.GameState):
        pass

    def get_stock_market_buy_action(self, game_state: e30.game_state.GameState):
        pass

    def get_stock_market_sell_action(self, game_state: e30.game_state.GameState):
        pass
