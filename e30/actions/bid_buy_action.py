from enum import Enum
import logging

log = logging.getLogger(__name__)


class BidBuyActionType(Enum):
    PASS = 0
    BID = 1
    BUY = 2


class BidBuyAction(object):
    def __init__(self, type: BidBuyActionType = BidBuyActionType.PASS, private_company_index: int = 0, bid: int = 0):
        self.type = type
        self.private_company_index = private_company_index
        self.bid = bid

    def print(self):
        log.info("{} {} {}".format(self.type, self.private_company_index, self.bid))
