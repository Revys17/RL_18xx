from typing import List

from exceptions.exceptions import InvalidOperationException
from player import Player
from private_company import Private
from stock_market import StockMarket
from stock_market_slot import StockMarketSlot


class Company:
    def __init__(self, name: str, initials: str, num_station_tokens: int, starting_city_name: str,
                 starting_city_location: str):
        # company info
        # TODO: stock market maybe shouldn't be a field here
        self.stock_market: StockMarket = StockMarket()
        self.name: str = name
        self.short_name: str = initials
        self.president: int = 0
        self.owning_players: List[int] = []
        self.par_value: int = 0
        self.current_share_price: StockMarketSlot

        # share info
        self.presidency_share: int = 1
        self.ipo_shares: int = 8
        self.market_shares: int = 0
        self.company_shares: int = 0

        # operations info
        self.operating: bool = False
        self.num_station_tokens: int = num_station_tokens
        self.starting_city_name: str = starting_city_name
        self.starting_city_location: str = starting_city_location
        self.trains: List = []
        self.money: int = 0
        self.privates: List[Private] = []
        self.tokens: List = []

    # TODO: where is this used? Can it be initialized in the constructor?
    def start_company(self, president: int, par_value: int) -> None:
        if self.presidency_share == 0:
            raise InvalidOperationException("Company " + self.name + " already started")

        self.president: int = president
        self.owning_players[president] = 2
        self.presidency_share: int = 0
        self.par_value: int = par_value
        self.current_share_price: StockMarketSlot = self.stock_market.get_par_value_slot(par_value)

    def buy_ipo_share(self, player: Player) -> None:
        if self.ipo_shares <= 0:
            raise InvalidOperationException("No IPO shares available for company " + self.name)

        # add check for number of shares owned
        if not self.can_purchase_share(player):
            raise InvalidOperationException("Player " + player.name + " cannot purchase more shares of company " +
                                            self.name)

        self.ipo_shares -= 1
        player.add_share(self)
        self.owning_players[player.number] = self.owning_players[player.number] + 1

        # add check for change of presidency

        if self.ipo_shares == 4:
            self.float_company()

    def can_purchase_share(self, player: Player) -> bool:
        if self.current_share_price.get_color() == "orange" or self.current_share_price.get_color() == "brown":
            return True

        return self.owning_players[player.number] < 6

    def float_company(self) -> None:
        self.money = 10 * self.par_value
        self.operating = True

    def buy_market_share(self, player: Player) -> None:
        if self.market_shares <= 0:
            raise InvalidOperationException("No Market shares available for company " + self.name)

        # add check for number of shares owned

        self.market_shares -= 1
        player.add_share(self)
        self.owning_players[player.number] = self.owning_players[player.number] + 1

        # add check for change of presidency

    def sell_share(self, player: Player, num_shares: int) -> None:
        if not player.has_share(self):
            raise InvalidOperationException("Player " + player.get_name() + " does not have any shares of company " +
                                            self.name + " to sell!")

        if self.market_shares + num_shares >= 5:
            raise InvalidOperationException("Company " + self.name + " has too many shares in the open market!")

        self.market_shares += num_shares
        player.sell_shares(self, num_shares)
        # TODO: current_share_price is a StockMarketSlot or int?
        current_share_price: int = self.stock_market.get_share_price_after_sale(self.current_share_price, num_shares)
