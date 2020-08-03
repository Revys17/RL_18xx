from typing import List, Dict

from action_blob import ActionBlob
from exceptions.exceptions import InvalidOperationException
import game_state
import player

class Private:
    def __init__(self, name: str, short_name: str, description: str, price: int, revenue: int, location: str):
        self.name: str = name
        self.short_name: str = short_name
        self.description: str = description
        self.price: int = price
        self.revenue: int = revenue
        self.location: str = location
        self.owner: player.Player
        self.bids: Dict[player.Player, int] = {}
        self.current_winning_bid: int = 0

    def add_bid(self, player: 'player.Player', bid: int) -> None:
        # TODO: audit this logic, current_winning_bid should be modified with bids coming in
        if bid < self.price + 5:
            raise InvalidOperationException("Bid must be a minimum of $5 more than price")

        if bid < self.current_winning_bid + 5:
            raise InvalidOperationException("Bid must be a minimum of $5 more than previous bid")

        if player.available_money < bid:
            raise InvalidOperationException("Player does not have enough available money")

        self.bids[player] = bid

        player.set_aside_money(bid)

    def should_resolve_bid(self) -> bool:
        return len(self.bids.keys()) > 0

    def resolve_bid(self, game_state: 'game_state.GameState') -> None:
        # hold auction, available to all bidders
        while True:
            if len(self.bids.keys()) == 1:
                winning_player: player.Player = list(self.bids.keys())[0]
                winning_player.return_money(self.bids[winning_player])
                self.buy_private(winning_player)
                break

            current_bidder: player.Player = list(self.bids.keys())[0]
            # TODO: define and use a bid resolution action type
            action_blob: ActionBlob = current_bidder.get_action_blob(game_state)

            if action_blob.action == "pass":
                current_bidder.return_money(self.bids[current_bidder])
                self.bids.pop(current_bidder)
            else:
                # if it's not a pass, it's a bid
                self.add_bid(current_bidder, int(action_blob.bid))

    def lower_price(self, lower_amount: int) -> None:
        self.price -= lower_amount

    def buy_private(self, player: 'player.Player'):
        # TODO: audit the logic, maybe move this to the player
        if self.bids[player] is None:
            if player.available_money < self.price:
                raise InvalidOperationException("Player does not have enough available money")
            player.remove_money(self.price)
        else:
            if player.available_money < self.bids[player]:
                raise InvalidOperationException("Player does not have enough available money")
            player.remove_money(self.bids[player])

        self.bids = None
        self.owner = player
        player.privates.append(self)

    def do_special_action(self, game_state: 'game_state.GameState'):
        # implement special action in subclasses
        pass
