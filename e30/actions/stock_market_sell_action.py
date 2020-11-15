from enum import Enum


class StockMarketSellActionType(Enum):
    PASS = 0
    SELL = 1


class StockMarketSellAction(object):
    def __init__(self, action_type: StockMarketSellActionType = StockMarketSellActionType.PASS):
        self.action_type = action_type
