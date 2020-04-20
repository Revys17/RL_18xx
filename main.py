#!/usr/bin/env python3
import random

"""
Module Docstring
"""

__author__ = "Caleb Larson"
__version__ = "0.1.0"
__license__ = "MIT"


def main():
    """ Main entry point of the app """
    game = initialize()
    run_game(game)
    exit()


def initialize():
    # init
    game_state = GameState()
    return game_state


def run_game(game_state):
    # do game stuff
    do_private_auction(game_state)

    while game_state.game_in_progress:
        do_stock_round(game_state)
        do_operating_rounds(game_state)
    return


def do_private_auction(game_state):
    auction_completed = False
    privates = game_state.privates
    current_private = 0
    consecutive_passes = 0

    while not auction_completed:
        current_player = game_state.get_next_player()
        action_blob = current_player.get_action_blob(game_state)

        if action_blob.action == "pass":
            # check if everyone passed
            consecutive_passes += 1
            if consecutive_passes == game_state.num_players:
                # everyone passed
                # if private is the SV
                if current_private == 0:
                    # if price is already 0, force player to purchase
                    if privates[current_private].price == 0:
                        current_private = do_run_off_auction(privates, current_player, current_private)
                        if current_private == -1:
                            auction_completed = True
                    # otherwise, lower price and continue
                    else:
                        privates[current_private].lower_price(5)
                # if private is not SV, pay private revenue and resume with priority deal
                else:
                    game_state.pay_private_revenue()
                    game_state.reset_next_player()
            continue
        elif action_blob.action == "buy":
            current_private = do_run_off_auction(privates, current_player, current_private)
            if current_private == -1:
                auction_completed = True
        elif action_blob.action == "bid":
            privates[action_blob.private].add_bid(current_player, action_blob.bid)
        consecutive_passes = 0


def do_run_off_auction(privates, starting_player, current_private):
    privates[current_private].buy_private(starting_player)
    for i in range(current_private + 1, len(privates)):
        if privates[i].current_bid is not None:
            privates[i].resolve_bid()
        else:
            return i
    return -1


def do_stock_round(game_state):
    pass


def do_operating_rounds(game_state):
    pass


class GameState:
    def __init__(self, num_players=4):
        self.game_in_progress = True
        self.num_players = num_players
        self.players = get_players(num_players)
        self.priority_deal = determine_first_player(num_players)
        self.current_player_index = self.priority_deal
        self.companies = get_companies()
        self.privates = get_privates()
        self.bank = Bank()
        self.stock_market = StockMarket()
        #self.train_market = TrainMarket()
        #self.tile_bank = TileBank()
        #self.map = Map()

    def get_next_player(self):
        self.current_player_index = (self.current_player_index + 1) % self.num_players
        return self.players[self.current_player_index]

    def reset_next_player(self):
        self.current_player_index = self.priority_deal

    def pay_private_revenue(self):
        for player in self.players:
            player.pay_private_income()


class Player:
    def __init__(self, starting_money):
        self.money = starting_money
        self.shares = []
        self.charters = []
        self.privates = []

    def get_action_blob(self, game_state):
        # needs to return the action the player wants to take
        # this is how we link to our AI
        pass

    def pay_private_income(self):
        for private in self.privates:
            self.money += private.income


class Bank:
    def __init__(self):
        # starting bank size in 1830
        self.money = 9600


class Company:
    def __init__(self, name, initials, num_station_tokens, starting_city_name, starting_city_location):
        # company info
        self.name = name
        self.short_name = initials
        self.president = None
        self.owning_players = {}
        self.par_value = None
        self.current_share_price = None

        # share info
        self.presidency_share = 1
        self.ipo_shares = 8
        self.market_shares = 0
        self.company_shares = 0

        # operations info
        self.operating = False
        self.num_station_tokens = num_station_tokens
        self.starting_city_name = starting_city_name
        self.starting_city_location = starting_city_location
        self.trains = []
        self.money = []
        self.privates = []
        self.tokens = []

    def start_company(self, president, par_value):
        if self.presidency_share == 0:
            raise InvalidOperationException("Company " + self.name + " already started")

        self.president = president
        self.owning_players[president] = 2
        self.presidency_share = 0
        self.par_value = par_value
        self.current_share_price = StockMarket.get_par_value_slot(par_value)

    def buy_ipo_share(self, player):
        if self.ipo_shares <= 0:
            raise InvalidOperationException("No IPO shares available for company " + self.name)

        # add check for number of shares owned
        if not self.can_purchase_share(player):
            raise InvalidOperationException("Player " + player.name + " cannot purchase more shares of company " +
                                            self.name)

        self.ipo_shares -= 1
        player.add_share(self)
        self.owning_players[player] = self.owning_players[player] + 1

        # add check for change of presidency

        if self.ipo_shares == 4:
            self.float_company()

    def can_purchase_share(self, player):
        if self.current_share_price.color == "orange" or self.current_share_price.color == "brown":
            return True

        return self.owning_players[player] < 6

    def float_company(self):
        self.money = 10 * self.par_value
        self.operating = True

    def buy_market_share(self, player):
        if self.market_shares <= 0:
            raise InvalidOperationException("No Market shares available for company " + self.name)

        # add check for number of shares owned

        self.market_shares -= 1
        player.addShare(self)
        self.owning_players[player] = self.owning_players[player] + 1

        # add check for change of presidency

    def sell_share(self, player, num_shares):
        if not player.has_share(self):
            raise InvalidOperationException("Player " + player.get_name() + " does not have any shares of company " +
                                            self.name + " to sell!")

        if self.market_shares + num_shares >= 5:
            raise InvalidOperationException("Company " + self.name + " has too many shares in the open market!")

        self.market_shares += num_shares
        player.sell_shares(self, num_shares)
        self.current_share_price = self.stock_market.get_share_price_after_sale(self.current_share_price, num_shares)


class StockMarket:
    def __init__(self):
        market = [
            ["60A", "67A", "71A", "76A", "82A", "90A", "100A", "112A", "126A", "142A", "160A", "180A", "200A", "225A", "250A", "275A", "300A", "325A", "350A"],
            ["53B", "60B", "66B", "70B", "76B", "82B", "90B",  "100B", "112B", "126B", "142B", "160B", "180B", "200B", "220B", "240B", "260B", "280B", "300B"],
            ["46C", "55C", "60C", "65C", "70C", "76C", "82C",  "90C",  "100C", "111C", "125C", "140C", "155C", "170C", "185C", "200C", None,   None,   None  ],
            ["39D", "48D", "54D", "60D", "66D", "71D", "76D",  "82D",  "90D",  "100D", "110D", "120D", "130D", None,   None,   None,   None,   None,   None  ],
            ["32E", "41E", "48E", "55E", "62E", "67E", "71E",  "76E" , "82E",  "90E",  "100E", None,   None,   None,   None,   None,   None,   None,   None  ],
            ["25F", "34F", "42F", "50F", "58F", "65F", "67F",  "71F",  "75F",  "80F",  None,   None,   None,   None,   None,   None,   None,   None,   None  ],
            ["18G", "27G", "36G", "45G", "54G", "63G", "67G",  "69G",  "70G",  None,   None,   None,   None,   None,   None,   None,   None,   None,   None  ],
            ["10H", "20H", "30H", "40H", "50H", "60H", "67H",  "68H",  None,   None,   None,   None,   None,   None,   None,   None,   None,   None,   None  ],
            [None,  "10I", "20I", "30I", "40I", "50I", "60I",  None,   None,   None,   None,   None,   None,   None,   None,   None,   None,   None,   None  ],
            [None,  None,  "10J", "20J", "30J", "40J", "50J",  None,   None,   None,   None,   None,   None,   None,   None,   None,   None,   None,   None  ],
            [None,  None,  None,  "10K", "20K", "30K", "40K",  None,   None,   None,   None,   None,   None,   None,   None,   None,   None,   None,   None  ]
        ]

        self.node_map = {}
        market_slots = []

        for i in range(len(market)):
            market_slots_row = []
            for j in range(market[i]):
                if market[i][j] is None:
                    market_slots.append(None)
                else:
                    slot = StockMarketSlot(market[i][j])
                    self.node_map[market[i][j]] = slot
                    market_slots.append(slot)
            market_slots.append(market_slots_row)

        for i in range(len(market_slots)):
            for j in range(len(market_slots[i])):
                slot = market_slots[i][j]
                if slot is None:
                    continue

                if i == 0:
                    up = "self"
                else:
                    up = market_slots[i-1][j]
                slot.set_up(up)

                if j == len(market_slots[i]) - 1:
                    right = up
                else:
                    right = market_slots[i][j+1]
                slot.set_right(right)

                if i == len(market_slots) - 1:
                    down = "self"
                else:
                    down = market_slots[i+1][j]
                slot.set_down(down)

                if j == 0:
                    left = down
                else:
                    left = market_slots[i][j-1]
                slot.set_left(left)

        self.par_locations = {
            100: self.node_map["100A"],
            90:  self.node_map["90B"],
            82:  self.node_map["82C"],
            76:  self.node_map["76D"],
            71:  self.node_map["71E"],
            67:  self.node_map["67F"],
        }

    def get_par_value_slot(self, value):
        return self.par_locations[value]

    def get_share_price_after_sale(self, current_price, num_shares_sold):
        for x in num_shares_sold:
            current_price = current_price.down
        return current_price

    def get_price_after_dividends(self, current_price, dividends_paid):
        if dividends_paid:
            return current_price.right
        return current_price.left

    def get_price_for_fully_owned(self, current_price, fully_owned):
        if fully_owned:
            return current_price.up
        return current_price


class StockMarketSlot:
    def __init__(self, location):
        self.location = location
        self.value = self.get_value(location)
        self.up = self
        self.right = self
        self.down = self
        self.left = self

    def set_up(self, node):
        self.up = self.set_dir(node)

    def set_right(self, node):
        self.right = self.set_dir(node)

    def set_down(self, node):
        self.down = self.set_dir(node)

    def set_left(self, node):
        self.left = self.set_dir(node)

    def set_dir(self, node):
        if node is None or node == "self":
            return self
        return node

    def get_value(self, location):
        return int(location[:-1])

    def get_color(self):
        if self.value <= 30:
            return "brown"
        if self.value <= 45:
            return "orange"
        if self.value <= 60:
            return "yellow"
        return "white"


class Private:
    def __init__(self, name, short_name, description, price, revenue, location):
        self.name = name
        self.short_name = short_name
        self.description = description
        self.price = price
        self.revenue = revenue
        self.location = location
        self.owner = None
        self.all_bidders = []
        self.bids = {}

    def add_bid(self, player, bid):
        if bid < self.price + 5:
            raise InvalidOperationException("Bid must be a minimum of $5 more than price")

        if bid < self.current_winning_bid + 5:
            raise InvalidOperationException("Bid must be a minimum of $5 more than previous bid")

        if player.available_money < bid:
            raise InvalidOperationException("Player does not have enough available money")

        try:
            self.all_bidders.remove(player)
        except ValueError:
            pass

        self.all_bidders.append(player)
        self.bids[player] = bid

        player.set_aside_money(bid)

    def resolve_bid(self):
        # hold auction, available to all bidders
        while True:
            if len(self.all_bidders) == 1:
                winning_player = self.all_bidders[0]
                winning_player.return_money(self.bids[winning_player])
                self.buy_private(winning_player)
                break

            current_bidder = self.all_bidders[0]
            action_blob = current_bidder.get_private_mini_auction_bid()

            if action_blob.action == "pass":
                self.all_bidders.remove(current_bidder)
                current_bidder.return_money(self.bids[current_bidder])
                self.bids[current_bidder] = None
            elif action_blob.action == "bid":
                self.add_bid(current_bidder, int(action_blob.bid))
            else:
                raise InvalidOperationException("Must pass or bid")

    def lower_price(self, lower_amount):
        self.price -= lower_amount

    def buy_private(self, player):
        self.owner = player

        if self.bids[player] is not None:
            if player.available_money < self.bids[player]:
                raise InvalidOperationException("Player does not have enough available money")
            player.remove_money(self.bids[player])
        else:
            if player.available_money < self.price:
                raise InvalidOperationException("Player does not have enough available money")
            player.remove_money(self.price)

        self.all_bidders = None
        self.bids = None


class InvalidOperationException:
    def __init__(self, message):
        self.message = message


def determine_first_player(num_players):
    return random.randrange(0, num_players)


def get_players(num_players):
    starting_money = get_starting_money(num_players)
    return [Player(starting_money) for x in range(0, num_players)]


def get_companies():
    companies = [
        Company("Pennsylvania", "PRR", 4, "Altoona", "H-12"),
        Company("New York Central", "NYC", 4, "albany", "E-19"),
        Company("Canadian Pacific", "CPR", 4, "Montreal", "A-19"),
        Company("Baltimore & Ohio", "B&O", 3, "Baltimore", "I-15"),
        Company("Chesapeake & Ohio", "C&O", 3, "Cleveland(Richmond)", "F-6 (K-13)"),
        Company("Erie", "Erie", 3, "Buffalo", "E-11"),
        Company("New York, New Haven, & Hartford", "NNH", 2, "New York", "G-19"),
        Company("Boston & Maine", "B&M", 2, "Boston", "E-23")
    ]

    return companies


def get_privates():
    privates = [
        Private("Schuylkill Valley", "SV", "No special ability", 20, 5, "G-15"),
        Private("Champlain & St. Lawrence", "CS", "Additional track lay on home tile. No connecting track needed.", 40,
                10, "B-20"),
        Private("Delaware & Hudson", "DH", "Allows lay of track and station on DH's hex. Lay costs regular amount " +
                "but station is free. Not an additional tile lay. Does not require route. Station must be placed " +
                "immediately to utilize track-free allowance.", 70, 15, "F-16"),
        Private("Mohawk & Hudson", "MH", "May exchange for a 10% share of NYC (max 60% and shares must be available). " +
                "May be done during player's stock turn or between other players' turns at any time. Closes the MH.",
                110, 20, "D-18"),
        Private("Camden & Amboy", "CA", "Purchaser receives 10% of the PRR.", 160, 25, "H-18"),
        Private("Baltimore & Ohio", "BO", "Purchaser receives the president's certificate of the B&O railroad and " +
                "immediately sets the par value. This company may not be sold or traded. Closes when the BO buys " +
                "its first train.", 220, 30, "I-15")
    ]

    return privates


def get_starting_money(num_players):
    # 1830 rules, up to 6 players
    return 2400 / num_players


if __name__ == "__main__":
    """ This is executed when run from the command line """
    main()
