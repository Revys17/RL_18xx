from typing import List

import e30
from e30.bank import Bank
from e30.exceptions.exceptions import InvalidOperationException
from e30.player import Player
from e30.private_company import Private
from e30.stock_market import StockMarket
from e30.stock_market_slot import StockMarketSlot, StockMarketSlotColor


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
        self.ipo_shares: int = 8
        self.market_shares: int = 0

        # operations info
        self.operating: bool = False
        self.num_station_tokens: int = num_station_tokens
        self.starting_city_name: str = starting_city_name
        self.starting_city_location: str = starting_city_location
        self.trains: List = []
        self.money: int = 0
        self.privates: List[Private] = []
        self.tokens: List = []

    def start_company(self, president: Player, par_value: int, stock_market: StockMarket, bank: Bank) -> None:
        if hasattr(self, 'president'):
            raise InvalidOperationException("Company " + self.name + " already started")

        self.president: Player = president
        self.owning_players.append(president)
        self.par_value: int = par_value
        self.current_share_price: StockMarketSlot = stock_market.get_par_value_slot(par_value)
        president.money -= par_value * 2
        bank.money += par_value * 2

    def buy_ipo_share(self, player: Player, bank: Bank) -> None:
        if self.ipo_shares <= 0:
            raise InvalidOperationException("No IPO shares available for company " + self.name)

        self.ipo_shares -= 1
        self.add_share(bank, player, 1, self.par_value)

        if self.ipo_shares == 4:
            self.float_company()

    def add_share(self, bank: Bank, player: Player, num_shares: int, share_price: int):
        player.money -= num_shares * share_price
        bank.money += num_shares * share_price
        if player not in self.owning_players:
            self.owning_players.append(player)
        if self.short_name not in player.share_map:
            player.share_map[self.short_name] = num_shares
        else:
            player.share_map[self.short_name] += num_shares

    def float_company(self) -> None:
        self.money = 10 * self.par_value
        self.operating = True

    def buy_market_share(self, player: Player, num_buy: int, bank: Bank) -> None:
        if self.market_shares < num_buy:
            raise InvalidOperationException("Not enough market shares available for company " + self.name)

        self.market_shares -= num_buy
        self.add_share(bank, player, num_buy, self.current_share_price.get_value()[0])

    def sell_share(self, player: Player, num_shares: int, bank: Bank) -> None:
        if not player.has_share(self):
            raise InvalidOperationException("Player " + player.get_name() + " does not have any shares of company " +
                                            self.name + " to sell!")

        if self.market_shares + num_shares >= 5:
            raise InvalidOperationException("Company " + self.name + " has too many shares in the open market!")

        self.market_shares += num_shares
        value_gained = player.sell_shares(self, num_shares)
        # TODO: when can the game end? Bank could have negative money at this point
        bank.money -= value_gained
        self.current_share_price = self.current_share_price.get_share_price_after_sale(num_shares)

    def transfer_presidency(self, new_president: Player):
        if not hasattr(self, 'president'):
            raise InvalidOperationException(self.short_name + " has no president. Can't transfer to "
                                            + new_president.get_name())
        # trade president's certificate for 2 regular certs
        self.president.presiding_companies.remove(self.short_name)
        self.president.share_map[self.short_name] += 2
        self.president = new_president
        self.president.presiding_companies.append(self.short_name)
        self.president.share_map[self.short_name] -= 2

    def __str__(self):
        president = self.president.get_name() if hasattr(self, 'president') else None
        share_price = '{}{} {}'.format(str(self.current_share_price.get_value()[0]),
                                       self.current_share_price.get_value()[1],
                                       self.current_share_price.get_color())\
            if hasattr(self, 'current_share_price') else None

        return f'Company: {self.short_name}, president: {president}, owning players: ' \
               f'{[p.get_name() for p in self.owning_players]}, ' \
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

