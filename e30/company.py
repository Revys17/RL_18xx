from typing import List

import e30
from e30.exceptions.exceptions import InvalidOperationException
from e30.player import Player
from e30.private_company import Private
from e30.stock_market_slot import StockMarketSlot


class Company:
    def __init__(self, name: str, initials: str, num_station_tokens: int, starting_city_name: str,
                 starting_city_location: str):
        self.name: str = name
        self.short_name: str = initials
        self.president: Player
        self.owning_players: List[Player] = []
        self.par_value: int = 0
        self.current_share_price: StockMarketSlot

        # share info
        self.presidency_share: int = 1
        self.ipo_shares: int = 8
        self.market_shares: int = 0
        # what are company shares?
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

    # initialize all the companies, then set the start when they're bought. Or initialize when bought.

    # TODO: where is this used? Can it be initialized in the constructor?
    def start_company(self, president: Player, par_value: int) -> None:
        if self.presidency_share == 0:
            raise InvalidOperationException("Company " + self.name + " already started")

        self.president: Player = president
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

    def __str__(self):
        president = self.president.get_name() if hasattr(self, 'president') else None
        share_price = '{}{} {}'.format(str(self.current_share_price.get_value()[0]),
                                       self.current_share_price.get_value()[1],
                                       self.current_share_price.get_color())\
            if hasattr(self, 'current_share_price') else None

        return f'Company: {self.short_name}, president: {president}, owning players: {self.owning_players}, ' \
               f'par value: {self.par_value}, current share price: {share_price}, ' \
               f'ipo shares: {self.ipo_shares}, market shares: {self.market_shares}, operating: {self.operating}, ' \
               f'money: {self.money}'

    def __lt__(self, other_company: 'e30.company.Company'):
        # do nothing on ties
        if not hasattr(other_company, 'current_share_price'):
            return False
        if not hasattr(self, 'current_share_price'):
            return True
        # compare int component, if equal, compare str component
        if self.current_share_price.get_value()[0] == other_company.current_share_price.get_value()[0]:
            return self.current_share_price.get_value()[1] < other_company.current_share_price.get_value()[1]
        return self.current_share_price.get_value()[0] < other_company.current_share_price.get_value()[0]
