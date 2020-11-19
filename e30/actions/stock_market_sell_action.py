from enum import Enum
from typing import OrderedDict


class StockMarketSellActionType(Enum):
    PASS = 0
    SELL = 1


class StockMarketSellAction(object):
    def __init__(self, action_type: StockMarketSellActionType = StockMarketSellActionType.PASS,
                 sell_map: OrderedDict[str, int] = {}):
        self.action_type = action_type
        self.sell_map = sell_map
