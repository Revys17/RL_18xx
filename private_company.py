from typing import List, Dict

from exceptions.exceptions import InvalidOperationException
from game_state import GameState
from player import Player


class Private:
    def __init__(self, name: str, short_name: str, description: str, price: int, revenue: int, location: str):
        self.name: str = name
        self.short_name: str = short_name
        self.description: str = description
        self.price: int = price
        self.revenue: int = revenue
        self.location: str = location
        self.owner: Player
        self.all_bidders: List[Player] = []
        self.bids: Dict[Player, int] = {}
        self.current_winning_bid: int = 0

    def add_bid(self, player: Player, bid: int) -> None:
        if bid < self.price + 5:
            raise InvalidOperationException("Bid must be a minimum of $5 more than price")

        if bid < self.current_winning_bid + 5:
            raise InvalidOperationException("Bid must be a minimum of $5 more than previous bid")

        if player.available_money < bid:
            raise InvalidOperationException("Player does not have enough available money")

        try:
            self.all_bidders.remove(player)
        except ValueError:
            pass

        self.all_bidders.append(player)
        self.bids[player] = bid

        player.set_aside_money(bid)

    def resolve_bid(self) -> None:
        # hold auction, available to all bidders
        while True:
            if len(self.all_bidders) == 1:
                winning_player: Player = self.all_bidders[0]
                winning_player.return_money(self.bids[winning_player])
                self.buy_private(winning_player)
                break

            current_bidder: Player = self.all_bidders[0]
            action_blob = current_bidder.get_private_mini_auction_bid()

            if action_blob.action == "pass":
                self.all_bidders.remove(current_bidder)
                current_bidder.return_money(self.bids[current_bidder])
                self.bids[current_bidder] = None
            elif action_blob.action == "bid":
                self.add_bid(current_bidder, int(action_blob.bid))
            else:
                raise InvalidOperationException("Must pass or bid")

    def lower_price(self, lower_amount: int) -> None:
        self.price -= lower_amount

    def buy_private(self, player: Player):
        self.owner = player

        if self.bids[player] is not None:
            if player.available_money < self.bids[player]:
                raise InvalidOperationException("Player does not have enough available money")
            player.remove_money(self.bids[player])
        else:
            if player.available_money < self.price:
                raise InvalidOperationException("Player does not have enough available money")
            player.remove_money(self.price)

        self.all_bidders = None
        self.bids = None

    def do_special_action(self, game_state: GameState):
        # implement special action in subclasses
        pass
