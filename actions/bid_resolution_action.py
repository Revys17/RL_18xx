from enum import Enum


class BidResolutionActionType(Enum):
    PASS = 0
    BID = 1


class BidResolutionAction(object):
    def __init__(self, type: BidResolutionActionType = BidResolutionActionType.PASS, bid: int = 0):
        self.type = type
        self.bid = bid
