from enum import Enum


class StockMarketBuyActionType(Enum):
    PASS = 0
    BUY = 1


class StockMarketBuyActionBuyType(Enum):
    PRESIDENT_CERT = 0
    IPO = 1
    BANK_POOL = 2


class StockMarketBuyAction(object):
    def __init__(self, action_type: StockMarketBuyActionType = StockMarketBuyActionType.PASS, company: str = "",
                 buy_type: StockMarketBuyActionBuyType = StockMarketBuyActionBuyType.PRESIDENT_CERT,
                 num_buy: int = 0, par_value: int = 0):
        self.action_type = action_type
        self.company = company
        self.buy_type = buy_type
        self.num_buy = num_buy
        self.par_value = par_value

    def __str__(self):
        return f'Action type: {self.action_type}, company: {self.company}, buy type: {self.buy_type}, ' \
               f'num to buy: {self.num_buy}, par value to set: {self.par_value}'
