from typing import List

import company
import game_state
from action_blob import ActionBlob
import agent
from private_company import Private


class Player:
    def __init__(self, starting_money: int, agent: 'agent.Agent'):
        self.number: int = 0
        self.money: int = starting_money
        self.shares: List = []
        self.charters: List = []
        self.privates: List[Private] = []
        self.name: str = "TODO"
        self.available_money: int = 0
        self.agent = agent

    def get_action_blob(self, game_state: 'game_state.GameState') -> ActionBlob:
        # needs to return the action the player wants to take
        # this is how we link to our AI
        return self.agent.get_action_blob(game_state)

    def pay_private_income(self) -> None:
        for private in self.privates:
            self.money += private.revenue

    def add_share(self, company: 'company.Company') -> None:
        pass

    def get_name(self) -> str:
        return self.name

    def has_share(self, company: 'company.Company') -> bool:
        # TODO
        return True

    def sell_shares(self, company: 'company.Company', num_shares: int) -> None:
        pass

    def set_aside_money(self, bid: int) -> None:
        pass

    def return_money(self, money: int) -> None:
        pass

    def get_private_mini_auction_bid(self) -> ActionBlob:
        pass

    def remove_money(self, money: int) -> None:
        pass
