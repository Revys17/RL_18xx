from enum import Enum


class StockMarketBuyActionType(Enum):
    PASS = 0
    BUY = 1


class StockMarketBuyAction(object):
    def __init__(self, action_type: StockMarketBuyActionType = StockMarketBuyActionType.PASS):
        self.action_type = action_type
