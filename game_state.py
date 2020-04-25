from typing import List

from company import Company
from main import get_players, determine_first_player, get_companies, get_privates
from player import Player
from private_company import Private
from stock_market import StockMarket
from bank import Bank


class GameState:
    # TODO: variable players
    def __init__(self, num_players: int = 4):
        self.game_in_progress: bool = True
        self.num_players: int = num_players
        self.players: List[Player] = get_players(num_players)
        self.priority_deal: int = determine_first_player(num_players)
        self.current_player_index: int = self.priority_deal
        self.companies: List[Company] = get_companies()
        self.privates: List[Private] = get_privates()
        self.bank: Bank = Bank()
        self.stock_market: StockMarket = StockMarket()
        #self.train_market = TrainMarket()
        #self.tile_bank = TileBank()
        #self.map = Map()

    def get_next_player(self) -> Player:
        self.current_player_index = (self.current_player_index + 1) % self.num_players
        return self.players[self.current_player_index]

    def reset_next_player(self) -> None:
        self.current_player_index = self.priority_deal

    def pay_private_revenue(self) -> None:
        for player in self.players:
            player.pay_private_income()
