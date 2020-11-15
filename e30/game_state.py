import heapq
from typing import List, Tuple, Dict
import logging

import e30
from e30.player import Player

log = logging.getLogger(__name__)


class GameState:
    # TODO: variable players
    def __init__(self, agents: List['e30.agent.Agent']):
        self.game_in_progress: bool = True
        self.num_players: int = len(agents)
        self.players: List[e30.player.Player] = e30.main.get_players(self.num_players, agents)
        self.priority_deal: e30.player.Player = self.players[e30.main.determine_first_player_index(self.num_players)]
        self.current_player: e30.player.Player = self.priority_deal
        self.companies_pq: List[e30.company.Company] = e30.main.get_companies()
        # make companies list a PQ, ordered from highest stock price to lowest stock price
        heapq.heapify(e30.main.get_companies())
        self.re_sort_companies()
        self.privates: List[e30.private_company.Private] = e30.main.get_privates()
        self.bank: e30.bank.Bank = e30.bank.Bank()
        self.stock_market: e30.stock_market.StockMarket = e30.stock_market.StockMarket()
        self.phase: e30.enums.phase.Phase = e30.enums.phase.Phase.PRIVATE_AUCTION
        self.progression: Tuple[e30.enums.round.Round, int] = (e30.enums.round.Round.BID_BUY, 0)
        self.num_player_to_total_cert_limit_map: List = [0, 0, 28, 20, 16, 13, 11, 11]
        self.player_total_cert_limit: int = self.num_player_to_total_cert_limit_map[self.num_players]
        #self.train_market = TrainMarket()
        #self.tile_bank = TileBank()
        #self.map = Map()

    def re_sort_companies(self) -> None:
        """
        must be called when company stock prices change, so that the companies pq remains a max heap based on stock
        price
        :return: None
        """
        # heapq are by default a minheap, maintain a maxheap by reverse sorting
        self.companies_pq.sort(reverse=True)

    def get_priority_deal_player(self) -> e30.player.Player:
        return self.players[self.priority_deal.index]

    def set_next_as_priority_deal(self, player: e30.player.Player) -> None:
        self.priority_deal = self.get_next_player(player)

    def get_next_player(self, player: e30.player.Player) -> e30.player.Player:
        next_player_index: int = (player.index + 1) % self.num_players
        return self.players[next_player_index]

    def set_next_as_current_player(self, player: e30.player.Player) -> None:
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
        log.info("Name | index | price | revenue | current winning bid")
        for i, private in enumerate(self.privates):
            log.info("{} | {} | {} | {} | {}".format(private.short_name, str(i), str(private.price),
                     str(private.revenue), str(private.current_winning_bid)))
            [log.info("Player: {} bid amount: {}".format(player.name, bid_amount))
             for player, bid_amount in private.bids.items()]

    def print_bid_resolution_turn_game_state(self) -> None:
        bid_target: e30.private_company.Private = self.privates[0]
        log.info("Currently bidding for: {}".format(bid_target.short_name))
        log.info("Price | revenue | current winning bid")
        log.info("{} | {} | {}".format(str(bid_target.price), str(bid_target.revenue),
                 str(bid_target.current_winning_bid)))
        [log.info("Player: {} bid amount: {}".format(player.name, bid_amount))
         for player, bid_amount in bid_target.bids.items()]

    def print_game_progression(self) -> None:
        log.info("Phase {}, Round {}, Round number {}".format(self.phase, self.progression[0], self.progression[1]))

    def print_priority_deal_player(self) -> None:
        log.info("Priority deal player: {}".format(self.get_priority_deal_player().get_name()))

    def print_stock_market_turn_game_state(self) -> None:
        [print(company) for company in self.companies_pq]
