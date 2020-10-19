from typing import List, Tuple
import logging

import e30

log = logging.getLogger(__name__)


class GameState:
    # TODO: variable players
    def __init__(self, agents: List['e30.agent.Agent']):
        self.game_in_progress: bool = True
        self.num_players: int = len(agents)
        self.players: List[e30.player.Player] = e30.main.get_players(self.num_players, agents)
        self.priority_deal: e30.player.Player = self.players[e30.main.determine_first_player_index(self.num_players)]
        self.current_player: e30.player.Player = self.priority_deal
        self.companies: List[e30.company.Company] = e30.main.get_companies()
        self.privates: List[e30.private_company.Private] = e30.main.get_privates()
        self.bank: e30.bank.Bank = e30.bank.Bank()
        self.stock_market: e30.stock_market.StockMarket = e30.stock_market.StockMarket()
        self.progression: Tuple[e30.enums.phase.Phase, e30.enums.round.Round] = (e30.enums.phase.Phase.PRIVATE_AUCTION,
                                                                                 e30.enums.round.Round.BID_BUY)
        #self.train_market = TrainMarket()
        #self.tile_bank = TileBank()
        #self.map = Map()

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
