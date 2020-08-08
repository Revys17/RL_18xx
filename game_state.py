from typing import List, Tuple

from company import Company
from enums.phase import Phase
from enums.round import Round
from main import get_players, determine_first_player_index, get_companies, get_privates
import agent
from player import Player
from private_company import Private
from stock_market import StockMarket
from bank import Bank


class GameState:
    # TODO: variable players
    def __init__(self, agents: List['agent.Agent']):
        self.game_in_progress: bool = True
        self.num_players: int = len(agents)
        self.players: List[Player] = get_players(self.num_players, agents)
        self.priority_deal: Player = self.players[determine_first_player_index(self.num_players)]
        self.current_player: Player = self.priority_deal
        self.companies: List[Company] = get_companies()
        self.privates: List[Private] = get_privates()
        self.bank: Bank = Bank()
        self.stock_market: StockMarket = StockMarket()
        self.progression: Tuple[Phase, Round] = (Phase.PRIVATE_AUCTION, Round.BID_BUY)
        #self.train_market = TrainMarket()
        #self.tile_bank = TileBank()
        #self.map = Map()

    def get_priority_deal_player(self) -> Player:
        return self.players[self.priority_deal.index]

    def set_next_as_priority_deal(self, player: Player) -> None:
        self.priority_deal = self.get_next_player(player)

    def get_next_player(self, player: Player) -> Player:
        next_player_index: int = (player.index + 1) % self.num_players
        return self.players[next_player_index]

    def set_next_as_current_player(self, player: Player) -> None:
        self.current_player = self.get_next_player(player)

    def set_current_player_as_priority_deal(self) -> None:
        self.current_player = self.priority_deal

    def pay_private_revenue(self) -> None:
        for player in self.players:
            player.pay_private_income()

    def increment_progression(self) -> None:
        pass
