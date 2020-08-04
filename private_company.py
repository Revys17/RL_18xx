import logging
from collections import OrderedDict
from typing import List

from action_blob import ActionBlob
from exceptions.exceptions import InvalidOperationException
import game_state
import player

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
        if bid < self.price + 5:
            raise InvalidOperationException("Bid must be a minimum of $5 more than price")

        if bid < self.current_winning_bid + 5:
            raise InvalidOperationException("Bid must be a minimum of $5 more than previous bid")

        # check if the player already has an existing bid. Players can add to the existing bid
        existing_bid: int = 0 if self.bids[player] is None else self.bids[player]

        if player.money + existing_bid < bid:
            raise InvalidOperationException("Player does not have enough available money")

        self.current_winning_bid = bid

        if existing_bid > 0:
            self.bids.pop(player)
        self.bids[player] = bid
        player.set_aside_money(bid - existing_bid)

    def should_resolve_bid(self) -> bool:
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
                self.buy_private(winning_player)
                break

            current_bidder: player.Player = bid_order.pop(0)
            is_highest_bidder: bool = current_bidder is list(self.bids.keys())[-1]
            retry: bool = True

            while retry:
                # TODO: define and use a bid resolution action type
                retry = False
                action_blob: ActionBlob = current_bidder.get_action_blob(game_state)

                if is_highest_bidder and action_blob.action == "pass":
                    # Do nothing when the highest bidder passes
                    continue
                elif action_blob.action == "pass":
                    # release bid money when lower bidders pass
                    current_bidder.return_money(self.bids[current_bidder])
                    self.bids.pop(current_bidder)
                else:
                    # if it's not a pass, it's a bid
                    try:
                        self.add_bid(current_bidder, int(action_blob.bid))
                    except InvalidOperationException as e:
                        log.error(e)
                        retry = True

            # schedule current bidder's next action
            bid_order.append(current_bidder)

    def lower_price(self, lower_amount: int) -> None:
        self.price -= lower_amount

    def buy_private(self, player: 'player.Player'):
        # player hasn't bid on the private company yet, buy it outright. Otherwise, bid money was already taken from
        # the player when the bid was placed and is consumed
        if self.bids[player] is None:
            if player.money < self.price:
                raise InvalidOperationException("Player does not have enough available money")
            player.remove_money(self.price)

        self.bids = None
        self.owner = player
        player.privates.append(self)

    def do_special_action(self, game_state: 'game_state.GameState'):
        # implement special action in subclasses
        pass
