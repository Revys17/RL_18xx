import logging
from collections import OrderedDict
from typing import List

from e30.actions.bid_resolution_action import BidResolutionAction, BidResolutionActionType
from e30.exceptions.exceptions import InvalidOperationException
from e30 import player, game_state

log = logging.getLogger(__name__)


class Private:
    def __init__(self, name: str, short_name: str, description: str, price: int, revenue: int, location: str):
        self.name: str = name
        self.short_name: str = short_name
        self.description: str = description
        self.price: int = price
        self.revenue: int = revenue
        self.location: str = location
        self.owner: player.Player
        # bids are ordered by player with the lowest bid. Only bids with greater values are ever added
        self.bids: OrderedDict[player.Player, int] = OrderedDict()
        self.current_winning_bid: int = 0

    def add_bid(self, player: 'player.Player', bid: int) -> None:
        log.info("Player {} placing {} bid on private company {}".format(player.name, bid, self.short_name))
        if bid < self.price + 5:
            raise InvalidOperationException("Bid must be a minimum of $5 more than price")

        if bid < self.current_winning_bid + 5:
            raise InvalidOperationException("Bid must be a minimum of $5 more than previous bid")

        # check if the player already has an existing bid. Players can add to the existing bid
        existing_bid: int = 0 if player not in self.bids else self.bids[player]

        if player.money + existing_bid < bid:
            raise InvalidOperationException("Player does not have enough available money")

        self.current_winning_bid = bid

        if existing_bid > 0:
            self.bids.pop(player)
        self.bids[player] = bid
        player.set_aside_money(bid - existing_bid)

    def should_resolve_bid(self) -> bool:
        if self.bids is None:
            return False
        return len(self.bids.keys()) > 0

    def resolve_bid(self, game_state: 'game_state.GameState') -> None:
        lowest_bidder: player.Player = list(self.bids.keys())[0]
        bidders: List[player.Player] = list(self.bids.keys())
        # sort by player index
        bidders.sort(key=lambda p: p.index)
        lowest_bidder_index: int = bidders.index(lowest_bidder)
        # action begins with the lowest_bidder and moves clockwise. Players that pass can still bid on their next turn
        # if the bid resolution has not ended by then
        bid_order: List[player.Player] = bidders[lowest_bidder_index:] + bidders[:lowest_bidder_index]

        while True:
            if len(self.bids.keys()) == 1:
                winning_player: player.Player = list(self.bids.keys())[0]
                log.info("Bid resolution winner: {}".format(winning_player.name))
                self.buy_private(winning_player)
                break

            current_bidder: player.Player = bid_order.pop(0)
            retry: bool = True

            while retry:
                retry = False
                bid_resolution_action: BidResolutionAction = current_bidder.get_bid_resolution_action(game_state)
                log.info("{} {} {}".format(current_bidder.get_name(), bid_resolution_action.type,
                                           bid_resolution_action.bid))

                if bid_resolution_action.type == BidResolutionActionType.PASS:
                    # release bid money when lower bidders pass
                    if current_bidder in self.bids:
                        current_bidder.return_money(self.bids[current_bidder])
                        self.bids.pop(current_bidder)
                else:
                    # if it's not a pass, it's a bid
                    try:
                        self.add_bid(current_bidder, int(bid_resolution_action.bid))
                    except InvalidOperationException as e:
                        log.error(e)
                        retry = True

            # schedule current bidder's next action
            bid_order.append(current_bidder)

    def lower_price(self, lower_amount: int) -> None:
        self.price -= lower_amount

    def buy_private(self, player: 'player.Player'):
        log.info("Player {} buying private company {}".format(player.name, self.short_name))
        # player hasn't bid on the private company yet, buy it outright. Otherwise, bid money was already taken from
        # the player when the bid was placed and is consumed
        if player not in self.bids:
            if player.money < self.price:
                raise InvalidOperationException("Player does not have enough available money")
            player.remove_money(self.price)

        self.bids = None
        self.owner = player
        player.privates.append(self)

    def do_special_action(self, game_state: 'game_state.GameState'):
        # implement special action in subclasses
        pass
