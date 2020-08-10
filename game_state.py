from typing import List, Tuple
import logging

from company import Company
from enums.phase import Phase
from enums.round import Round
from main import get_players, determine_first_player_index, get_companies, get_privates
import agent
from player import Player
from private_company import Private
from stock_market import StockMarket
from bank import Bank

log = logging.getLogger(__name__)


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

    def print_bid_buy_turn_game_state(self) -> None:
        log.info("Current priority deal player: {}".format(self.priority_deal.name))
        log.info("Current bids and unowned private companies:")
        log.info("Name | index | price | revenue | current winning bid | bids:")
        for i, private in enumerate(self.privates):
            log.info("{} | {} | {} | {} | {} | {}".format(private.short_name, str(i), str(private.price),
                     str(private.revenue), str(private.current_winning_bid), str(private.bids)))

    def print_bid_resolution_turn_game_state(self) -> None:
        bid_target: Private = self.privates[0]
        log.info("Currently bidding for: {}".format(bid_target.short_name))
        log.info("Price | revenue | current winning bid | bids:")
        log.info("{} | {} | {} | {}".format(str(bid_target.price), str(bid_target.revenue),
                 str(bid_target.current_winning_bid), str(bid_target.bids)))
