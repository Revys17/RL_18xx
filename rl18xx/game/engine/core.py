__all__ = [
    "PUBLISHER_INFO",
    "pascal_to_snake",
    "snake_to_pascal",
    "GameError",
    "OptionError",
    "NoToken",
    "RouteTooShort",
    "RouteTooLong",
    "ReusesCity",
    "GameLog",
    "Assignable",
    "Entity",
    "Item",
    "Ownable",
    "Passer",
    "SharePrice",
    "ShareHolder",
    "Spender",
    "BaseMovement",
    "TwoDimensionalMovement",
    "OneDimensionalMovement",
    "ZigZagMovement",
    "StockMarket",
    "Phase",
]


import copy
from collections import defaultdict
import re


def pascal_to_snake(pascal_str):
    return re.sub(r"(?<!^)(?=[A-Z])", "_", pascal_str).lower()


def snake_to_pascal(snake_str):
    return "".join(word.capitalize() for word in snake_str.split("_"))


class GameError(RuntimeError):
    pass


class OptionError(RuntimeError):
    pass


class NoToken(GameError):
    pass


class RouteTooShort(GameError):
    pass


class RouteTooLong(GameError):
    pass


class ReusesCity(GameError):
    pass


class GameLog(list):
    def __init__(self, game):
        super().__init__()
        self.game = game

    def append(self, message):
        """Overrides the append method to add a log entry."""
        if not isinstance(message, GameLog.Entry):
            message = GameLog.Entry(message, self.game.current_action_id)
        super().append(message)

    class Entry:
        """A log entry storing a message and an action ID."""

        def __init__(self, message, action_id):
            self.message = message
            self.action_id = action_id

        def __repr__(self):
            """Representation of the log entry."""
            return f"<Entry message='{self.message}', action_id={self.action_id}>"


class Assignable:
    assignments = {}

    @classmethod
    def assigned(cls, assignable, key):
        return key in cls.assignments.get(assignable, {})

    @classmethod
    def assign(cls, assignable, key, value=True):
        if assignable not in cls.assignments:
            cls.assignments[assignable] = {}
        cls.assignments[assignable][key] = value

    @classmethod
    def remove_assignment(cls, assignable, key):
        if assignable in cls.assignments:
            cls.assignments[assignable].pop(key, None)

    @classmethod
    def remove_from_all(cls, assignables, key):
        for assignable in assignables:
            if cls.assigned(assignable, key):
                cls.remove_assignment(assignable, key)

class Entity:
    def is_company(self):
        return False

    def is_corporation(self):
        return False

    def is_minor(self):
        return False

    def is_system(self):
        return False

    def is_operator(self):
        return False

    def is_player(self):
        return False

    def is_receivership(self):
        return False

    def is_share_pool(self):
        return False

    def is_closed(self):
        return False
    
    def __deepcopy__(self, memo):
        cls = self.__class__
        result = memo.get(id(self))
        if result:
            return result

        result = cls.__new__(cls)
        memo[id(self)] = result

        # Copy the attributes used in __hash__ first
        for attr in ["id", "name"]:
            if hasattr(self, attr):
                setattr(result, attr, copy.deepcopy(getattr(self, attr), memo))

        # Copy the rest of the attributes
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))
        return result


class Item:
    def __init__(self, description="", cost=0):
        self.description = description
        self.cost = cost

    def __eq__(self, other):
        return self.description == other.description and self.cost == other.cost


class Ownable:
    def __init__(self):
        self._owner = None

    @property
    def owner(self):
        return self._owner

    @owner.setter
    def owner(self, value):
        self._owner = value

    def owned_by(self, entity):
        if not entity:
            return False
        return (
            self._owner == entity
            or getattr(self._owner, "owner", None) == entity
            or self._owner == getattr(entity, "owner", None)
        )

    def player(self):
        if self._owner.is_player():
            return self._owner
        return self._owner.player()

    def corporation(self):
        if self.is_corporation():
            return self
        if getattr(self._owner, "corporation", None):
            return self._owner.corporation()
        return None

    def owned_by_corporation(self):
        return self._owner and self._owner.is_corporation()

    def owned_by_player(self):
        return self._owner and self._owner.is_player()

    def is_corporation(self):
        return False


class Passer:
    def __init__(self):
        self._passed = False

    @property
    def passed(self):
        return self._passed

    @property
    def active(self):
        return not self._passed

    def pass_(self):
        self._passed = True

    def unpass(self):
        self._passed = False


import re


class SharePrice:
    TYPE_MAP = {
        "p": "par",
        "e": "endgame",
        "c": "close",
        "b": "multiple_buy",
        "o": "unlimited",
        "y": "no_cert_limit",
        "l": "liquidation",
        "a": "acquisition",
        "r": "repar",
        "i": "ignore_one_sale",
        "j": "ignore_two_sales",
        "s": "safe_par",
        "P": "par_overlap",
        "x": "par_1",
        "z": "par_2",
        "w": "par_3",
        "C": "convert_range",
        "m": "max_price",
        "n": "max_price_1",
        "u": "phase_limited",
        "t": "type_limited",
        "B": "pays_bonus",
        "W": "pays_bonus_1",
        "X": "pays_bonus_2",
        "Y": "pays_bonus_3",
        "Z": "pays_bonus_4",
    }

    NON_HIGHLIGHT_TYPES = [
        "par",
        "safe_par",
        "par_1",
        "par_2",
        "par_3",
        "par_overlap",
        "safe_par",
        "convert_range",
        "max_price",
        "max_price_1",
        "repar",
        "type_limited",
    ]

    PAR_TYPES = ["par", "par_overlap", "par_1", "par_2", "par_3"]

    def __init__(
        self,
        code,
        row,
        column,
        unlimited_types=None,
        multiple_buy_types=None,
    ):
        m = re.match(r"(\d*)([a-zA-Z]*)", code)
        types = [self.TYPE_MAP[char] for char in m.group(2)]

        self.coordinates = (row, column)
        self.price = int(m.group(1))
        self.type = types[0] if types else None
        self.types = types or []
        self.corporations = []
        self.can_buy_multiple = self.type in (multiple_buy_types or [])
        self.limited = self.type not in (unlimited_types or [])

    def __eq__(self, other):
        return isinstance(other, SharePrice) and self.price == other.price and self.coordinates == other.coordinates

    @property
    def id(self):
        return f"{self.price},{','.join(map(str, self.coordinates))}"

    def counts_for_limit(self):
        return self.limited

    def buy_multiple(self):
        return self.can_buy_multiple

    def can_par(self):
        return self.type in self.PAR_TYPES

    def end_game_trigger(self):
        return self.type == "endgame"

    def liquidation(self):
        return self.type == "liquidation"

    def acquisition(self):
        return self.type == "acquisition"

    def highlight(self):
        return self.type and self.type not in self.NON_HIGHLIGHT_TYPES

    def normal_movement(self):
        return self.type != "liquidation"

    def remove_par(self):
        self.types = [t for t in self.types if t not in self.PAR_TYPES]
        self.type = self.types[0] if self.types else None

    def __str__(self):
        return f"Share Price - coordinates: {self.coordinates}, price: {self.price}, types: {self.types}, corporations: {self.corporations}"

    def __repr__(self):
        return self.__str__()


class ShareHolder:
    def __init__(self):
        self._shares_by_corporation = defaultdict(list)

    @property
    def shares(self):
        return [share for shares in self._shares_by_corporation.values() for share in shares]

    @property
    def shares_by_corporation(self):
        return self._shares_by_corporation

    @property
    def shares_by_corporation_sorted(self):
        return dict(sorted(self._shares_by_corporation.items()))

    def shares_of(self, corporation):
        return self._shares_by_corporation.get(corporation, [])

    def delete_share(self, share):
        self._shares_by_corporation.get(share.corporation, []).remove(share)

    def certs_of(self, corporation):
        return self.shares_of(corporation)

    def percent_of(self, corporation):
        shares = self._shares_by_corporation.get(corporation, [])
        return sum(share.percent for share in shares)

    def common_percent_of(self, corporation):
        shares = [share for share in self._shares_by_corporation.get(corporation, []) if not share.preferred]
        return sum(share.percent for share in shares)

    def presidencies(self):
        return [
            corporation
            for corporation, shares in self._shares_by_corporation.items()
            if any(share.president for share in shares)
        ]

    def num_shares_of(self, corporation, ceil=True):
        percent = self.percent_of(corporation)
        num = percent / corporation.share_percent
        return int(num) if ceil else num


class Spender:
    def __init__(self):
        self.cash = 0

    def check_cash(self, amount, borrow_from=None):
        available = self.cash + (borrow_from.cash if borrow_from else 0)
        if (available - amount) < 0:
            raise GameError(f"{self.name} has {self.cash} and cannot spend {amount}")

    def check_positive(self, amount):
        if amount <= 0:
            raise GameError(f"{amount} is not valid to spend")

    def spend(self, cash, receiver, check_cash=True, check_positive=True, borrow_from=None):
        if check_cash:
            self.check_cash(cash, borrow_from=borrow_from)
        if check_positive:
            self.check_positive(cash)

        # Check if we need to borrow from our borrow_from target
        if borrow_from and (cash > self.cash):
            amount_borrowed = cash - self.cash
            self.cash = 0
            borrow_from.cash -= amount_borrowed
        else:
            self.cash -= cash

        receiver.cash += cash


class BaseMovement:
    def __init__(self, market):
        self.market = market

    def share_price(self, coordinates):
        row, column = coordinates
        return (
            self.market.market[row][column]
            if row < len(self.market.market) and column < len(self.market.market[row])
            else None
        )

    def left(self, corporation, coordinates):
        raise NotImplementedError

    def right(self, corporation, coordinates):
        raise NotImplementedError

    def down(self, corporation, coordinates):
        raise NotImplementedError

    def up(self, corporation, coordinates):
        raise NotImplementedError


class TwoDimensionalMovement(BaseMovement):
    def left(self, corporation, coordinates):
        r, c = coordinates
        if c > 0 and self.share_price([r, c - 1]):
            return [r, c - 1]
        else:
            return self.down(corporation, coordinates)

    def right(self, corporation, coordinates):
        r, c = coordinates
        if c + 1 >= len(self.market.market[r]):
            return self.up(corporation, coordinates)
        else:
            return [r, c + 1]

    def down(self, _corporation, coordinates):
        r, c = coordinates
        if r + 1 < len(self.market.market):
            r += 1
        return [r, c]

    def up(self, _corporation, coordinates):
        r, c = coordinates
        if r - 1 >= 0:
            r -= 1
        return [r, c]


class OneDimensionalMovement(BaseMovement):
    def left(self, _corporation, coordinates):
        r, c = coordinates
        if c - 1 >= 0:
            c -= 1
        return [r, c]

    def right(self, _corporation, coordinates):
        r, c = coordinates
        if c + 1 < len(self.market.market[r]):
            c += 1
        return [r, c]

    def down(self, corporation, coordinates):
        return self.left(corporation, coordinates)

    def up(self, corporation, coordinates):
        return self.right(corporation, coordinates)


class ZigZagMovement(BaseMovement):
    def __init__(self, market, ledge_movement):
        self.ledge_movement = ledge_movement
        super().__init__(market)

    def left(self, _corporation, coordinates):
        r, c = coordinates
        if self.ledge_movement:
            c -= 2
            c = 0 if c < 0 else c
        elif c - 2 >= 0:
            c -= 2
        return [r, c]

    def right(self, _corporation, coordinates):
        r, c = coordinates
        if self.ledge_movement:
            c += 2
            c = len(self.market.market[r]) - 1 if c >= len(self.market.market[r]) else c
        elif c + 2 < len(self.market.market[r]):
            c += 2
        return [r, c]

    def down(self, _corporation, coordinates):
        r, c = coordinates
        if c > 0:
            c -= 1
        return [r, c]

    def up(self, _corporation, coordinates):
        r, c = coordinates
        if c + 1 < len(self.market.market[r]):
            c += 1
        return [r, c]


class StockMarket:
    def __init__(
        self,
        market,
        unlimited_types,
        multiple_buy_types=None,
        zigzag=None,
        ledge_movement=None,
    ):
        self.par_prices = []
        self.has_close_cell = False
        self.max_reached = False
        self.zigzag = zigzag
        self.market = [
            [
                SharePrice(code, r_index, c_index, unlimited_types, multiple_buy_types) if code != "" else None
                for c_index, code in enumerate(row)
            ]
            for r_index, row in enumerate(market)
        ]

        for row in self.market:
            for price in row:
                if price and price.can_par():
                    self.par_prices.append(price)
                if price and price.type == "close":
                    self.has_close_cell = True

        self.par_prices.sort(key=lambda p: (p.price, p.coordinates[1], p.coordinates[0]), reverse=True)

        if self.zigzag:
            self.movement = ZigZagMovement(self, ledge_movement)
        elif self.one_d():
            self.movement = OneDimensionalMovement(self)
        else:
            self.movement = TwoDimensionalMovement(self)

    def one_d(self):
        return all(len(row) == 1 for row in self.market)

    def set_par(self, corporation, share_price):
        share_price.corporations.append(corporation)
        corporation.share_price = share_price
        corporation._par_price = share_price
        corporation.original_par_price = share_price

    def right_ledge(self, coordinates):
        row, col = coordinates
        return col + 1 == len(self.market[row])

    def move_right(self, corporation):
        coordinates = self.right(corporation, corporation.share_price.coordinates)
        self.move(corporation, coordinates)

    def right(self, corporation, coordinates):
        return self.movement.right(corporation, coordinates)

    def move_up(self, corporation):
        coordinates = self.up(corporation, corporation.share_price.coordinates)
        self.move(corporation, coordinates)

    def up(self, corporation, coordinates):
        return self.movement.up(corporation, coordinates)

    def move_down(self, corporation):
        coordinates = self.down(corporation, corporation.share_price.coordinates)
        self.move(corporation, coordinates)

    def down(self, corporation, coordinates):
        return self.movement.down(corporation, coordinates)

    def move_left(self, corporation):
        coordinates = self.left(corporation, corporation.share_price.coordinates)
        self.move(corporation, coordinates)

    def left(self, corporation, coordinates):
        return self.movement.left(corporation, coordinates)

    def find_share_price(self, corporation, directions):
        return self.find_relative_share_price(corporation.share_price, corporation, directions)

    def find_relative_share_price(self, share, corporation, directions):
        coordinates = share.coordinates
        price = self.share_price(coordinates)

        for direction in directions:
            if direction == "left":
                coordinates = self.left(corporation, coordinates)
            elif direction == "right":
                coordinates = self.right(corporation, coordinates)
            elif direction == "down":
                coordinates = self.down(corporation, coordinates)
            elif direction == "up":
                coordinates = self.up(corporation, coordinates)

            price = self.share_price(coordinates) or price

        return price

    def move(self, corporation, coordinates, force=False):
        share_price = self.share_price(coordinates)

        if not share_price or share_price == corporation.share_price:
            return

        if not force and not share_price.normal_movement():
            return

        corporation.share_price.corporations.remove(corporation)
        corporation.share_price = share_price
        self.max_reached = True if share_price.end_game_trigger() else False
        share_price.corporations.append(corporation)

    def share_prices_with_types(self, types):
        return sorted(
            [price for row in self.market for price in row if price and any(t in types for t in price.types)],
            key=lambda p: p.price,
            reverse=True,
        )

    def share_price(self, coordinates):
        row, column = coordinates

        if row < len(self.market) and column < len(self.market[row]):
            return self.market[row][column]

    def remove_par(self, price):
        self.par_prices.remove(price)
        price.remove_par()


class Phase:
    def __init__(self, phases, game):
        self.index = 0
        self.phases = phases
        self.game = game
        self.depot = game.depot
        self.log = game.log
        self.setup_phase()

    def buying_train(self, entity, train, source):
        while train.sym in self.next_on:
            self.next_phase()

        self.game.rust_trains(train, entity)
        self.depot.depot_trains(clear=True)

        for event in train.events:
            getattr(self.game, f"event_{event['type']}")()
        train.events.clear()
        self.game.after_buying_train(train, source)

    @property
    def previous(self):
        return self.phases[: self.index]

    @property
    def current(self):
        return self.phases[self.index]

    @property
    def upcoming(self):
        return self.phases[self.index + 1] if self.index + 1 < len(self.phases) else None

    def train_limit(self, entity):
        if isinstance(self._train_limit, dict):
            return self._train_limit.get(entity.type, 0)
        else:
            return self._train_limit

    def available(self, phase_name):
        if not phase_name:
            return False
        index = next(
            (i for i, phase in enumerate(self.phases) if phase["name"] == phase_name),
            -1,
        )
        return index <= self.index

    def setup_phase(self):
        phase = self.phases[self.index]

        self.name = phase["name"]
        self.operating_rounds = phase.get("operating_rounds")
        self._train_limit = phase.get("train_limit")
        self.tiles = list(phase.get("tiles", []))
        self.events = phase.get("events", [])
        self.status = phase.get("status", [])
        self.corporation_sizes = phase.get("corporation_sizes")
        self.next_on = list(self.phases[self.index + 1]["on"]) if self.index + 1 < len(self.phases) else []

        log_msg = f"-- Phase {self.name} ("
        if self.operating_rounds:
            log_msg += f"Operating Rounds: {self.operating_rounds} | "
        log_msg += f"Train Limit: {self.train_limit_to_str(self._train_limit)}"
        log_msg += f" | Available Tiles: {', '.join(map(str.capitalize, self.tiles))})"
        self.log.append(log_msg)
        self.trigger_events()

    def trigger_events(self):
        for company in self.game.companies:
            if company.owner:
                for ability in self.game.abilities(company, "revenue_change", on_phase=self.name):
                    company.revenue = ability.revenue

                for _ in self.game.abilities(company, "close", on_phase=self.name):
                    self.log.append(f"Company {company.name} closes")
                    company.close()

        for entity in self.game.companies + self.game.corporations:
            entity.remove_ability_when(self.name)

    def next_phase(self):
        self.index += 1
        self.setup_phase()

    def train_limit_to_str(self, train_limit):
        if isinstance(train_limit, dict):
            return ", ".join(f"{type}: {limit}" for type, limit in train_limit.items())
        else:
            return str(train_limit)

    def __str__(self):
        return f"<Phase: {self.name}>"

    def __repr__(self):
        return self.__str__()


PUBLISHER_INFO = {
    "all_aboard_games": {
        "name": "All-Aboard Games",
        "url": "https://all-aboardgames.com/",
    },
    "deep_thought_games": {
        "name": "Deep Thought Games",
        "url": "https://boardgamegeek.com/boardgamepublisher/4192/deep-thought-games-llc",
        "hidden": True,
    },
    "gmt_games": {
        "name": "GMT Games",
        "url": "https://www.gmtgames.com/",
    },
    "golden_spike": {
        "name": "Golden Spike Games",
        "url": "https://goldenspikegames.com/",
    },
    "grand_trunk_games": {
        "name": "Grand Trunk Games",
        "url": "https://www.grandtrunkgames.com/",
    },
    "lonny_games": {
        "name": "Lonny Games",
        "url": "https://www.lonnygames.com/",
    },
    "lookout": {
        "name": "Lookout Games",
        "url": "https://lookout-spiele.de/",
        "hidden": True,
    },
    "loserdogs": {
        "name": "Loserdogs",
        "url": "http://tanisan.com/ld/",
        "hidden": True,
    },
    "marflow_games": {
        "name": "Marflow Games",
        "url": "https://18xx-marflow-games.de/",
    },
    "mayfair": {
        "name": "Mayfair Games",
        "url": "https://boardgamegeek.com/boardgamepublisher/10/mayfair-games",
        "hidden": True,
    },
    "mercury": {
        "name": "Mercury Games",
        "url": "https://www.mercurygames.com/",
        "hidden": True,
    },
    "oo_games": {
        "name": "Double-O Games",
        "url": "http://ohley.de/english/",
        "hidden": True,
    },
    "seahorse": {
        "name": "Seahorse Laser & Design",
        "url": "https://www.etsy.com/shop/SeahorseLaserDesign?section_id=24360565",
        "hidden": True,
    },
    "self_published": {
        "name": "Self-published",
        "url": "https://18xx.games",
        "hidden": True,
    },
    "traxx": {
        "name": "TraXX",
        "url": "https://traxx-denver.com/games/",
    },
    "zman_games": {
        "name": "Z-MAN Games",
        "url": "https://zmangames.com/",
        "hidden": True,
    },
}
