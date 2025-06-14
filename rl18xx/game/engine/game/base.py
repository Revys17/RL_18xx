__all__ = ["Meta", "BaseGame"]

from rl18xx.game.engine.core import (
    GameError,
    GameLog,
    NoToken,
    OptionError,
    Phase,
    RouteTooLong,
    StockMarket,
    pascal_to_snake,
)
from rl18xx.game.engine.entities import (
    Bank,
    Company,
    Corporation,
    Depot,
    Minor,
    Train,
    Player,
    PlayerInfo,
    ShareBundle,
    SharePool,
)
from rl18xx.game.engine.abilities import (
    Token as TokenAbility,
)
from rl18xx.game.engine.actions import (
    BaseAction,
    ProgramBuyShares,
    ProgramSharePass,
    Undo,
    Redo,
    Message,
    Pass,
)
from rl18xx.game.engine.graph import (
    Hex,
    Graph,
    Tile,
)
from rl18xx.game.engine.round import (
    SpecialTrack as SpecialTrackStep,
    BuySellParShares,
    CompanyPendingPar,
    Bankrupt as BankruptStep,
    Exchange as ExchangeStep,
    Track as TrackStep,
    Token as TokenStep,
    Route as RouteStep,
    Dividend as DividendStep,
    DiscardTrain as DiscardTrainStep,
    BuyTrain as BuyTrainStep,
    BuyCompany as BuyCompanyStep,
    WaterfallAuction,
    Tracker as TrackerStep,
    Auction as AuctionRound,
    Stock as StockRound,
    Operating as OperatingRound,
)

import re
from random import choice, randint
from collections import defaultdict
from itertools import combinations
import logging
import copy

LOGGER = logging.getLogger(__name__)

from ..core import PUBLISHER_INFO


class Meta:
    # platform-relevant metadata
    DEV_STAGES = ["production", "beta", "alpha", "prealpha"]
    DEV_STAGE = "prealpha"
    PROTOTYPE = False
    DEPENDS_ON = None
    AUTOROUTE = True

    # game title variations
    GAME_TITLE = None  # canonical title stored in database, defaults to '18xx' part of 'G18xx' module name
    GAME_DISPLAY_TITLE = None  # defaults to GAME_TITLE; used in UI on game cards, new game dropdown, game page
    GAME_SUBTITLE = None
    GAME_FULL_TITLE = (
        None  # defaults to "GAME_DISPLAY_TITLE", then "GAME_TITLE: GAME_SUBTITLE"; used in "Game Info" section
    )
    GAME_DROPDOWN_TITLE = (
        None  # new game dropdown, defaults to GAME_DISPLAY_TITLE + location and dev stage if applicable
    )
    GAME_ISSUE_LABEL = None  # the GitHub label used to organize issues for this title, defaults to GAME_TITLE

    # real game metadata
    GAME_DESIGNER = None
    GAME_IMPLEMENTER = None
    GAME_INFO_URL = None
    GAME_LOCATION = None
    GAME_PUBLISHER = None
    GAME_RULES_URL = None
    GAME_ALIASES = []
    GAME_VARIANTS = []
    GAME_IS_VARIANT_OF = None

    # rules data that needs to be known to the engine without loading in the
    # full game class
    PLAYER_RANGE = None
    OPTIONAL_RULES = []
    MUTEX_RULES = []

    # terms to match with on the create games page; see keywords function for
    # values automatically considered as keywords
    KEYWORDS = []

    @classmethod
    def title(cls):
        if not hasattr(cls, "_title"):
            name = cls.__name__
            if name in ["Game", "Meta"]:
                if cls.__module__ == "__main__":
                    name = "Meta"
                else:
                    name = cls.__module__.split(".")[-1][1:]
            cls._title = name
        return cls._title

    @classmethod
    def display_title(cls):
        if not hasattr(cls, "_display_title"):
            cls._display_title = cls.GAME_DISPLAY_TITLE or cls.title()
        return cls._display_title

    @classmethod
    def full_title(cls):
        if not hasattr(cls, "_full_title"):
            cls._full_title = cls.GAME_FULL_TITLE or cls.GAME_DISPLAY_TITLE or f"{cls.title()}:{cls.GAME_SUBTITLE}"
        return cls._full_title

    @classmethod
    def label(cls):
        if not hasattr(cls, "_label"):
            label = cls.GAME_ISSUE_LABEL or cls.title()
            cls._label = f'"{label}"' if " " in label else label
        return cls._label

    @classmethod
    def known_issues_url(cls):
        return f"https://github.com/tobymao/18xx/issues?q=is%3Aissue+is%3Aopen+label%3A{cls.label()}"

    @classmethod
    def fs_name(cls):
        if not hasattr(cls, "_fs_name"):
            parts = cls.__name__.split("::")
            last = parts[-1]
            part = parts[-2] if last in ["Game", "Meta"] else last
            cls._fs_name = part[1:].replace("G", "g_").replace("_", " ").lower()
        return cls._fs_name

    @classmethod
    def meta(cls):
        return cls

    @classmethod
    def game_instance(cls):
        return False

    @classmethod
    def game_variants(cls):
        if not hasattr(cls, "_game_variants"):
            cls._game_variants = {v["sym"]: {**v, "meta": meta_by_title(v["title"])} for v in cls.GAME_VARIANTS}
        return cls._game_variants

    @classmethod
    def keywords(cls):
        if not hasattr(cls, "_keywords"):
            cls._keywords = (
                cls.KEYWORDS
                + [cls.title(), cls.full_title(), cls.display_title()]
                + cls.GAME_ALIASES
                + [cls.DEPENDS_ON, cls.GAME_LOCATION]
                + [v["title"] for v in cls.GAME_VARIANTS]
                + [o_r["short_name"] for o_r in cls.OPTIONAL_RULES]
                + [
                    cls.DEV_STAGE,
                    "PROTOTYPE" if cls.PROTOTYPE else None,
                    cls.GAME_DESIGNER,
                ]
                + [(PUBLISHER_INFO[pub]["name"] if pub in PUBLISHER_INFO else pub) for pub in [cls.GAME_PUBLISHER]]
                + [cls.GAME_IMPLEMENTER]
            )
            cls._keywords = list(set(keyword.upper() for keyword in cls._keywords if keyword))
        return cls._keywords

    @classmethod
    def check_options(cls, options, min_players, max_players):
        pass


import json
import os
from ...gamemap import GameMap


class BaseGame:
    @classmethod
    def load(
        cls,
        data,
        at_action=None,
        actions=None,
        pin=None,
        seed=None,
        optional_rules=None,
        user=None,
        **kwargs,
    ):
        if isinstance(data, str):
            if os.path.exists(data):
                with open(data, "r") as file:
                    data = json.load(file)
            else:
                data = json.loads(data)
            return cls.load(
                data,
                at_action=at_action,
                actions=actions,
                pin=pin or data.get("settings", {}).get("pin"),
                seed=seed or data.get("settings", {}).get("seed"),
                optional_rules=optional_rules,
                user=user,
                **kwargs,
            )

        elif isinstance(data, dict):
            title = data["title"]
            names = {p.get("id"): p["name"] for p in data["players"]}
            id_ = data["id"]
            actions = actions or data.get("actions", [])
            pin = pin or data.get("settings", {}).get("pin")
            seed = seed or data.get("settings", {}).get("seed")
            optional_rules = optional_rules or data.get("settings", {}).get("optional_rules", [])

        elif isinstance(data, BaseGame):
            title = data.title
            names = {u.id: u.name for u in data.ordered_players}
            id_ = data.id
            actions = actions or [action.to_dict() for action in data.actions]
            pin = pin or data.settings.get("pin")
            seed = seed or data.settings.get("seed")
            optional_rules = optional_rules or data.settings.get("optional_rules", [])

        game = GameMap().game_by_title(title)
        return game(
            names,
            id=id_,
            actions=actions,
            at_action=at_action,
            pin=pin,
            seed=seed,
            optional_rules=optional_rules,
            user=user,
            **kwargs,
        )

    def __init__(
        self,
        names,
        metadata,
        entities,
        map,
        id=0,
        actions=None,
        at_action=None,
        pin=None,
        strict=False,
        optional_rules=None,
        user=None,
        seed=None,
    ):
        self.metadata = metadata()
        self.entities = entities()
        self.map = map()

        self.title = metadata.title()

        self.id = id
        self.turn = 1
        self.final_turn = None
        self.loading = False
        self.strict = strict
        self.finished = False
        self.log = GameLog(self)
        self.queued_log = []
        self.actions = [action.copy() for action in actions] if actions else []
        self.raw_actions = []
        self.turn_start_action_id = 0
        self.last_turn_start_action_id = 0
        self.exception = None

        if isinstance(names, dict):
            self.names = names.copy()
        else:
            self.names = dict((n, n) for n in names)

        self.players = [Player(player_id, name) for player_id, name in self.names.items()]

        self.user = user
        self.programmed_actions = {}
        self.round = None
        self.round_counter = 0

        self.optional_rules = self.init_optional_rules(optional_rules)

        self.initialize_seed(seed)

        dev_stage = self.metadata.DEV_STAGE
        if dev_stage == "prealpha":
            self.log.append(f"{self.title} is in prealpha state, no support is provided at all")
        elif dev_stage == "alpha":
            self.log.append(
                f"{self.title} is currently considered 'alpha',"
                " the rules implementation is likely to not be complete."
            )
            self.log.append(
                "As the implementation improves, games that are not compatible"
                " with the latest version will be archived without notice."
            )
            self.log.append("We suggest that any alpha quality game is concluded within 7 days.")
        elif dev_stage == "beta":
            self.log.append(
                f"{self.title} is currently considered 'beta'," " the rules implementation may allow illegal moves."
            )
            self.log.append(
                "As the implementation improves, games that are not compatible"
                " with the latest version will be pinned but may be archived after 7 days."
            )
            self.log.append("Because of this, we suggest not playing games that may take months to complete.")

        if self.metadata.PROTOTYPE:
            self.log.append(
                f"{self.title} is currently a prototype game,"
                " the design is not final, and so may change at any time."
            )
            if dev_stage != "alpha":
                self.log.append("If the game is modified due to a design change, games will be pinned")

        self.companies = self.init_companies(self.players)
        self.stock_market = self.init_stock_market()
        self.minors = self.init_minors()
        self.loans = self.init_loans()
        self.total_loans = len(self.loans)
        self.corporations = self.init_corporations(self.stock_market)
        self.closing_queue = {}
        self.corporations_are_closing = False
        self._axes = None
        self._crowded_corps = None
        self.bank = self.init_bank()
        self.tiles = self.init_tiles()
        self.all_tiles = self.init_tiles()
        self.optional_tiles()
        self.tile_groups = []
        self._cert_limit = self.init_cert_limit()
        self.removals = []
        self.pin = pin

        self.depot = self.init_train_handler()
        self.init_starting_cash(self.players, self.bank)
        self.share_pool = self.init_share_pool()
        self.hexes = self.init_hexes(self.companies, self.corporations)
        self.graph = self.init_graph()

        self.cities = [city for tile in ([hex.tile for hex in self.hexes] + self.tiles) for city in tile.cities]

        self.phase = self.init_phase()
        self.operating_rounds = self.phase.operating_rounds if self.phase else None

        self.round_history = []
        self.setup_preround()
        self.cache_objects()
        self.connect_hexes()

        self.init_company_abilities()
        self.round = self.init_round()
        self.initial_round_type = self.round.__class__

        self.check_optional_rules()
        self.log_optional_rules()
        self.setup_optional_rules()
        self.setup()
        self.round.setup()
        self.temp_allow_cross_company_purchase = True
        self.initialize_actions(actions, at_action=at_action)
        self.temp_allow_cross_company_purchase = False

        if pin:
            self.log.append("----")
            self.log.append("Your game was unable to be upgraded to the latest version of 18xx.games.")
            self.log.append(f"It is pinned to version {pin}.")
            self.log.append("Please do not submit bug reports for pinned games. Pinned games cannot be debugged.")
            if self.metadata.DEV_STAGE == "beta":
                self.log.append("Please note, pinned games may be deleted after 7 days.")
            self.log.append("----")

    # Game end check is described as a dictionary
    # with reason => after
    #   reason: What kind of game end check to do
    #   after: When game should end if check triggered
    # Leave out a reason if the game does not support that.
    # Allowed reasons:
    #  bankrupt, stock_market, bank, final_train, final_phase, custom
    # Allowed after:
    #  immediate - ends in the current turn
    #  current_round - ends at the end of the current round
    #  current_or - ends at the next end of an OR
    #  full_or - ends at the next end of a complete OR set
    #  one_more_full_or_set - finish the current OR set, then
    #                         end after the next complete OR set
    GAME_END_CHECK = {"bankrupt": "immediate", "bank": "full_or"}
    BANKRUPTCY_ALLOWED = True
    # How many players does bankruptcy cause to end the game
    # one - as soon as any player goes bankrupt
    # all_but_one - all but one
    BANKRUPTCY_ENDS_GAME_AFTER = "one"
    BANK_CASH = 12000
    CURRENCY_FORMAT_STR = "${}"
    FORMAT_UPGRADES_ON_HEXES = False
    STARTING_CASH = {}
    HEXES = {}
    LAYOUT = None
    AXES = None
    TRAINS = []
    CERT_LIMIT_TYPES = ["multiple_buy", "unlimited", "no_cert_limit"]
    # Does the cert limit decrease when a player becomes bankrupt?
    CERT_LIMIT_CHANGE_ON_BANKRUPTCY = False
    CERT_LIMIT_INCLUDES_PRIVATES = True
    # Does the cert limit care about how many players started the game or how
    # many remain?
    CERT_LIMIT_COUNTS_BANKRUPTED = False
    PRESIDENT_SALES_TO_MARKET = False
    MULTIPLE_BUY_TYPES = ["multiple_buy"]
    MULTIPLE_BUY_ONLY_FROM_MARKET = False
    STOCKMARKET_COLORS = {
        "par": "red",
        "endgame": "blue",
        "close": "black",
        "multiple_buy": "brown",
        "unlimited": "orange",
        "no_cert_limit": "yellow",
        "liquidation": "red",
        "acquisition": "yellow",
        "repar": "gray",
        "ignore_one_sale": "green",
        "safe_par": "white",
        "max_price": "purple",
    }
    MIN_BID_INCREMENT = 5
    MUST_BID_INCREMENT_MULTIPLE = False
    ONLY_HIGHEST_BID_COMMITTED = False
    CAPITALIZATION = "full"
    # Must sell all shares of a company in one action per turn
    MUST_SELL_IN_BLOCKS = False
    # Percent of one company you are allowed to sell in one turn. None means
    # unlimited and is the default
    TURN_SELL_LIMIT = None
    # when can a shareholder sell shares
    # first           -- after the first stock round
    # after_ipo       -- after the stock round in which the company is opened
    # operate         -- after operation
    # full_or_turn    -- after the corporation completes a full OR turn
    # p_any_operate   -- president any time, shareholders after operation
    # any_time        -- at any time
    # round           -- after the stock round the share was purchased in
    SELL_AFTER = "first"
    # down_share -- down one row per share
    # down_per_10 -- down one row per 10% sold
    # down_block -- down one row per block
    # left_share -- left one column per share
    # left_share_pres -- left one column per share if president
    # left_block -- one row per block
    # down_block_pres -- down one row per block if president
    # left_block_pres -- left one column per block if president
    # left_per_10_if_pres_else_left_one -- left_share_pres + left_block
    # none -- don't drop price
    SELL_MOVEMENT = "down_share"
    # Order in which shares are sold and bought
    # :sell_buy_or_buy_sell
    # :sell_buy
    # :sell_buy_sell
    SELL_BUY_ORDER = "sell_buy_or_buy_sell"
    # Do shares in the pool drop the price?
    # none, down_block, left_block, down_share
    POOL_SHARE_DROP = "none"
    # Do sold out shares increase the price?
    SOLD_OUT_INCREASE = True
    # Player order in the next stock round
    # :after_last_to_act -- player after the last to act goes first. Order remains the same.
    # :first_to_pass -- players ordered by when they first started passing.
    NEXT_SR_PLAYER_ORDER = "after_last_to_act"
    # Do tile reservations completely block other companies?
    # :never -- token can be placed as long as there is a city space for existing tile reservations
    # :always -- token cannot be placed until tile reservation resolved
    # :single_slot_cities -- token cannot be placed if tile contains any single slot cities
    TILE_RESERVATION_BLOCKS_OTHERS = "never"
    # List of companies in the game
    COMPANIES = []
    # Class for defining a company
    COMPANY_CLASS = Company
    # Class for defining a corporation
    CORPORATION_CLASS = Corporation
    # Class for defining a train
    TRAIN_CLASS = Train
    # Class for defining a depot
    DEPOT_CLASS = Depot
    # Class for defining a player
    PLAYER_CLASS = Player
    # List of minor companies in the game
    MINORS = []
    # List of phases in the game
    PHASES = []
    # Names of locations on the game board
    LOCATION_NAMES = {}
    # Hexes that hide location names
    HEXES_HIDE_LOCATION_NAMES = {}
    # Track restriction type
    TRACK_RESTRICTION = "semi_restrictive"
    # Allow presidential swaps of other corporations when ebuying
    EBUY_PRES_SWAP = True
    # Allow ebuying other corp trains for up to face value
    EBUY_OTHER_VALUE = False
    # If ebuying from depot, must buy cheapest train
    EBUY_DEPOT_TRAIN_MUST_BE_CHEAPEST = True
    # Corporation must issue shares before ebuy (if possible)
    MUST_EMERGENCY_ISSUE_BEFORE_EBUY = False
    # Corporation may continue to sell shares even though enough funds
    EBUY_SELL_MORE_THAN_NEEDED = False
    # True if a player can sell shares for ebuy
    EBUY_CAN_SELL_SHARES = True
    # Owner of ebuying entity is on the hook
    EBUY_OWNER_MUST_HELP = False
    # If sold more than needed then cannot then buy a cheaper train in the depot
    EBUY_SELL_MORE_THAN_NEEDED_LIMITS_DEPOT_TRAIN = False
    # Loans taken during ebuy can lead to receivership
    EBUY_CORP_LOANS_RECEIVERSHIP = False
    # Where should sold shares go to?
    # :bank - bank pool
    # :corporation - back to corporation/ipo
    SOLD_SHARES_DESTINATION = "bank"
    # When is the home token placed?
    # on par, float, operating_round (start of next OR), operate (corporation's first OR turn)
    HOME_TOKEN_TIMING = "operate"
    # How to handle discarded trains
    DISCARDED_TRAINS = "discard"
    # Percent discount for discarded trains
    DISCARDED_TRAIN_DISCOUNT = 0
    # Remove trains of closed corporations
    CLOSED_CORP_TRAINS_REMOVED = True
    # Remove tokens of closed corporations
    CLOSED_CORP_TOKENS_REMOVED = True
    # Remove reservations of closed corporations
    CLOSED_CORP_RESERVATIONS_REMOVED = True
    # When must the company buy a train if it doesn't have one
    # route, never, always
    MUST_BUY_TRAIN = "route"
    # Allow train buy from other corporations
    ALLOW_TRAIN_BUY_FROM_OTHERS = True
    # Allow train buy from other player's corporations
    ALLOW_TRAIN_BUY_FROM_OTHER_PLAYERS = False
    # Allow obsolete trains to be bought from other corporations
    ALLOW_OBSOLETE_TRAIN_BUY = False
    # Default tile lay configuration
    TILE_LAYS = [{"lay": True, "upgrade": True, "cost": 0}]
    # Tile type of the game
    TILE_TYPE = "normal"
    # Minors can own shares
    MINORS_CAN_OWN_SHARES = False
    # Must an upgrade use the maximum number of exits
    TILE_UPGRADES_MUST_USE_MAX_EXITS = []
    # Cost of tiles
    TILE_COST = 0
    # Colors of impassable hexes
    IMPASSABLE_HEX_COLORS = ["blue", "gray", "red"]
    # Text for game events
    EVENTS_TEXT = {
        "close_companies": [
            "Companies Close",
            "All companies unless otherwise noted are discarded from the game",
        ],
    }
    # Text for game status
    STATUS_TEXT = {
        "can_buy_companies": [
            "Can Buy Companies",
            "All corporations can buy companies from players",
        ],
    }
    # Text for market conditions
    MARKET_TEXT = {
        "par": "Par value",
        "no_cert_limit": "Corporation shares do not count towards cert limit",
        "unlimited": "Corporation shares can be held above 60%",
        "multiple_buy": "Can buy more than one share in the corporation per turn",
        "close": "Corporation closes",
        "endgame": "End game trigger",
        "liquidation": "Liquidation",
        "repar": "Par value after bankruptcy",
        "ignore_one_sale": "Ignore first share sold when moving price",
    }
    # Text for game end reasons
    GAME_END_REASONS_TEXT = {
        "bankrupt": "player is bankrupt",
        "bank": "The bank runs out of money",
        "stock_market": "Corporation enters end game trigger on stock market",
        "final_train": "The final train is purchased",
        "final_phase": "The final phase is entered",
        "custom": "Unknown custom reason",
    }
    # Text for game end reasons timing
    GAME_END_REASONS_TIMING_TEXT = {
        "immediate": "Immediately",
        "current_round": "End of the current round",
        "current_or": "Next end of an OR",
        "full_or": "Next end of a complete OR set",
        "one_more_full_or_set": "End of the next complete OR set after the current one",
    }
    # Text for game end description reasons mapping
    GAME_END_DESCRIPTION_REASON_MAP_TEXT = {
        "bank": "Bank Broken",
        "bankrupt": "Bankruptcy",
        "stock_market": "Company hit max stock value",
        "final_train": "Final train was purchased",
        "final_phase": "Final phase was reached",
    }
    # Assignment tokens mapping
    ASSIGNMENT_TOKENS = {}
    # Name of the operating round
    OPERATING_ROUND_NAME = "Operating"
    # Short name for operation rounds
    OPERATION_ROUND_SHORT_NAME = "ORs"
    # Market share limit in percent
    MARKET_SHARE_LIMIT = 50
    # Whether all companies are assignable
    ALL_COMPANIES_ASSIGNABLE = False
    # Whether obsolete trains count for limit
    OBSOLETE_TRAINS_COUNT_FOR_LIMIT = False
    # Allow corporate buy share for a single corp only
    CORPORATE_BUY_SHARE_SINGLE_CORP_ONLY = False
    # Allow corporate buy share from the president
    CORPORATE_BUY_SHARE_ALLOW_BUY_FROM_PRESIDENT = False
    # Allow buying shares from other players
    BUY_SHARE_FROM_OTHER_PLAYER = False
    # Use variable float percentages
    VARIABLE_FLOAT_PERCENTAGES = False
    # Whether corporation cards should show percentage ownership breakdown for players
    SHOW_SHARE_PERCENT_OWNERSHIP = False
    # Allow removing towns from the map
    ALLOW_REMOVING_TOWNS = False
    # Allow multiple outstanding programmed actions
    ALLOW_MULTIPLE_PROGRAMS = False
    # Cachable items
    CACHABLE = [
        ("players", "player"),
        ("corporations", "corporation"),
        ("companies", "company"),
        ("trains", "train"),
        ("hexes", "hex"),
        ("tiles", "tile"),
        ("shares", "share"),
        ("share_prices", "share_price"),
        ("cities", "city"),
        ("minors", "minor"),
        ("loans", "loan"),
    ]
    # Parameters for random number generation
    RAND_A = 1103515245
    RAND_C = 12345
    RAND_M = 2**31

    def setup_preround(self):
        pass

    def setup(self):
        pass

    def init_optional_rules(self, optional_rules):
        optional_rules = [rule for rule in (optional_rules or [])]
        for rule in self.metadata.OPTIONAL_RULES:
            if rule.get("players") and len(rule["players"]) > 0:
                optional_rules.remove(rule["sym"])
        return optional_rules

    def check_optional_rules(self):
        min_players = len(self.players)
        max_players = len(self.players)
        error = self.metadata.check_options(self.optional_rules, min_players, max_players)
        if error:
            raise OptionError(error.get("error"))

    def setup_optional_rules(self):
        pass

    def log_optional_rules(self):
        if not self.optional_rules:
            return

        self.log.append("Optional rules used in this game:")
        for optional_rule in self.metadata.OPTIONAL_RULES:
            if optional_rule["sym"] in self.optional_rules:
                self.log.append(f" * {optional_rule['short_name']}: {optional_rule['desc']}")

    def optional_hexes(self):
        return self.game_hexes()

    def game_hexes(self):
        return self.map.HEXES

    def hex_neighbor(self, hex, edge):
        if hex.neighbors.get(edge):
            return hex.neighbors[edge]

        letter = re.match(r"(\D+)(\d+)", hex.id).group(1)
        number = int(re.match(r"(\D+)(\d+)", hex.id).group(2))

        flip_axes = True if self.layout == "flat" and self.axes == {"x": "number", "y": "letter"} else False

        d_letter, d_number = 0, 0
        if (self.layout == "flat" and edge in [0, 1, 2, 3, 4, 5]) or (
            self.layout == "pointy" and edge in [0, 1, 2, 3, 4, 5]
        ):
            if edge == 0 or edge == 4:
                d_number, d_letter = 2, 0
            elif edge == 1 or edge == 3:
                d_number, d_letter = 1, -1
            elif edge == 2 or edge == 2:
                d_number, d_letter = -1, -1
            elif edge == 3 or edge == 1:
                d_number, d_letter = -2, 0
            elif edge == 4 or edge == 0:
                d_number, d_letter = -1, 1
            elif edge == 5 or edge == 5:
                d_number, d_letter = 1, 1

        if flip_axes:
            d_letter, d_number = d_number, d_letter

        letter_index = Hex.LETTERS.index(letter)
        new_letter_index = (letter_index + d_letter) % len(Hex.LETTERS)
        new_letter = Hex.LETTERS[new_letter_index]
        number += d_number

        return self.hex_by_id(f"{new_letter}{number}")

    def location_name(self, coord):
        return self.map.LOCATION_NAMES.get(coord)

    def optional_tiles(self):
        pass

    @property
    def move_number(self):
        return len(self.raw_actions)

    def to_dict(self):
        return {
            "title": self.title,
            "players": [{"id": u.id, "name": u.name} for u in sorted(self.players, key=lambda x: x.id)],
            "actions": [action if isinstance(action, dict) else action.to_dict() for action in self.raw_actions],
            "id": self.id,
            "finished": self.finished,
            "move_number": self.move_number,
            "result": self.result(),
        }

    @classmethod
    def register_colors(cls, **colors):
        cls.COLORS = colors

    def meta(self):
        return self.metadata

    def game_instance(self):
        return True

    def initialize_seed(self, seed):
        # hotseat games created without the seed field being set
        seed = None if seed == "" else seed
        id_digits = [int(digit) for digit in re.findall(r"\d+", str(self.id))]
        seed = seed or (id_digits[0] if id_digits else 0)
        self.seed = seed
        self.rand = self.seed % self.RAND_M

    def random(self):
        self.rand = ((self.RAND_A * self.rand) + self.RAND_C) % self.RAND_M
        return self.rand

    def inspect(self):
        player_names = [player.name for player in self.players]
        return f'{self.__class__.__name__} - {self.metadata.title} {", ".join(player_names)}'

    def result_players(self):
        return self.players

    def result(self):
        result_data = [(player.id, self.player_value(player)) for player in self.result_players()]
        sorted_result = sorted(result_data, key=lambda x: x[1], reverse=True)
        return dict(sorted_result)

    def turn_round_num(self):
        return [self.turn, self.round.round_num]

    @property
    def current_entity(self):
        return self.round.active_step().current_entity if self.round.active_step() else self.actions[-1].entity

    def pass_entity(self, user):
        return self.current_entity

    def active_players(self):
        players_ = [self.acting_for_player(e.player()) for e in self.round.active_entities if e and e.player()]
        players_ = [player for player in players_ if player]  # Remove None values

        if not players_:
            return [player for player in self.players if not player.bankrupt]
        else:
            return players_

    def active_step(self):
        return self.round.active_step()

    def active_players_id(self):
        return [player.id for player in self.active_players()]

    def valid_actors(self, action):
        player = action.entity.player if action.entity and action.entity.player else None
        actor = self.acting_for_player(player)
        return [actor] if player and actor else self.active_players()

    def acting_for_entity(self, entity):
        return entity.owner if entity else None

    def acting_for_player(self, player):
        return player

    def player_log(self, entity, msg):
        if entity and entity.id == self.user:
            self.log.append(f"-- {msg}")

    def available_programmed_actions(self):
        # By default assume normal 1830esk buy shares
        return [ProgramBuyShares, ProgramSharePass]

    # Note: this class expects actions in to_dict() form
    @staticmethod
    def filtered_actions(actions):
        # set_trace()
        if not actions:
            return [], []

        active_undos = []
        filtered_actions = [None] * len(actions)

        for index, action in enumerate(actions):
            action_type = action["type"]
            if action_type == "undo":
                # set_trace()
                undo_to = None
                if "action_id" in action:
                    action_id = action["action_id"]
                    if action_id == 0:
                        undo_to = 0
                    else:
                        for i, a in enumerate(actions):
                            if a.get("id") == action_id:
                                undo_to = i + 1
                                break
                else:
                    undo_to = 0
                    for i, a in enumerate(filtered_actions[index::-1]):
                        if a and a["type"] != "message":
                            undo_to = index - i
                            break

                undos_to_clear = []
                for i in range(undo_to, index):
                    if filtered_actions[i] and filtered_actions[i]["type"] != "message":
                        undos_to_clear.append((filtered_actions[i], i))
                        filtered_actions[i] = None
                if undos_to_clear:
                    active_undos.append(undos_to_clear)
            elif action_type == "redo" and active_undos:
                # set_trace()
                for undo_action, pos in active_undos.pop():
                    filtered_actions[pos] = undo_action
            else:
                if action_type != "message":
                    active_undos.clear()
                filtered_actions[index] = action

        # Ensure the output matches Ruby's behavior, including any None values as they are part of the logic.
        return filtered_actions, active_undos

    def initialize_actions(self, actions, at_action=None):
        self.loading = True if not self.strict else False
        self._filtered_actions, active_undos = self.filtered_actions(actions)

        # Store all actions for history navigation
        self.raw_all_actions = [action for action in actions] if actions else []

        if actions:
            self.process_to_action(at_action or actions[-1]["id"])
        self.undo_possible = False
        if active_undos:
            self.redo_possible = True
        self.loading = False

    def able_to_operate(self, entity, train, name):
        return True

    def process_action(self, action, add_auto_actions=False, validate_auto_actions=False):
        # LOGGER.debug(f"Processing action: {action}")
        if isinstance(action, dict):
            action = BaseAction.action_from_dict(action, self)

        action.id = self.current_action_id + 1
        self.raw_actions.append(action.to_dict())

        if isinstance(action, Undo) or isinstance(action, Redo):
            return self.clone(self.raw_actions)

        self.actions.append(action)

        try:
            self.process_single_action(action)
        except Exception as e:
            # if the action is a pass, let's try skipping it.
            if not isinstance(action, Pass):
                raise e

        if not isinstance(action, Message):
            self.redo_possible = False
            self.undo_possible = True
            self.last_game_action_id = action.id

        if add_auto_actions or validate_auto_actions:
            auto_actions = []
            while True:
                actions = self.round.auto_actions or []
                if not actions:
                    break
                for a in actions:
                    self.process_single_action(a)
                auto_actions.extend(actions)
            if validate_auto_actions:
                if not self.auto_actions_match(action.auto_actions, auto_actions):
                    raise GameError("Auto actions do not match")
            else:
                action.clear_cache()
                action.auto_actions = auto_actions
                self.raw_actions[-1] = action.to_dict()
        else:
            for a in action.auto_actions:
                self.process_single_action(a)

        self.last_processed_action = action.id

        return self

    def process_single_action(self, action):
        if (
            action.user
            and action.user != self.acting_for_player(action.entity.player).id
            and not action.instance_of(Message)
        ):
            self.log.append(
                f'• Action({action.type}) via Master Mode by: {self.player_by_id(action.user).name if self.player_by_id(action.user) else "Owner"}'
            )

        self.preprocess_action(action)
        self.round.process_action(action)
        self.action_processed(action)

        end_timing = self.game_end_check()[-1] if self.game_end_check() else None
        if end_timing == "immediate":
            self.end_game()
        while self.round.finished() and not self.finished:
            for entity in self.round.entities:
                entity.unpass()
            if self.end_now(end_timing):
                self.end_game()
            else:
                self.transition_to_next_round()

    def rescue_exception(self, e, action):
        self.raw_actions.pop()
        self.actions.pop()
        self.exception = e
        self.broken_action = action

    def transition_to_next_round(self):
        # set_trace()
        self.store_player_info()
        self.next_round()
        self.check_programmed_actions()
        self.finalize_round_setup()

    def finalize_round_setup(self):
        self.round.at_start = True
        self.round.setup()
        self.round_history.append(self.current_action_id)

    def maybe_raise(self):
        if self.exception:
            exception = self.exception
            self.exception = None
            self.broken_action = None
            raise exception
        return self

    def auto_actions_match(self, actions_a, actions_b):
        if len(actions_a) != len(actions_b):
            return False
        return all(
            a.to_dict(exclude=["created_at"]) == b.to_dict(exclude=["created_at"]) for a, b in zip(actions_a, actions_b)
        )

    def store_player_info(self):
        if not self.round.show_in_history:
            return
        for p in self.players:
            p.history.append(
                PlayerInfo(
                    self.round.short_name,
                    self.turn,
                    self.round.round_num,
                    self.player_value(p),
                )
            )

    def preprocess_action(self, action):
        pass

    def all_corporations(self):
        return self.corporations

    def sorted_corporations(self):
        ipoed = [corp for corp in self.corporations if corp.ipoed]
        others = [corp for corp in self.corporations if not corp.ipoed]
        return sorted(ipoed) + others

    @property
    def operating_order(self):
        return [minor for minor in self.minors if minor.floated()] + sorted(
            [corp for corp in self.corporations if corp.floated()]
        )

    def operated_operators(self):
        return [entity for entity in (self.corporations + self.minors) if entity.operated]

    @property
    def current_action_id(self):
        return self.raw_actions[-1].get("id", 0) if self.raw_actions and self.raw_actions[-1] else 0

    def last_game_action_id(self):
        return self.last_game_action_id if hasattr(self, "last_game_action_id") else 0

    def previous_action_id_from(self, action_id):
        filtered_actions_rev = reversed(self._filtered_actions) if hasattr(self, "_filtered_actions") else []
        for action in filtered_actions_rev:
            if action and action["id"] < action_id and action["type"] != "message":
                return action["id"]
        return 0

    def next_action_id_from(self, action_id):
        for action in self._filtered_actions if hasattr(self, "_filtered_actions") else []:
            if action and action["id"] > action_id and action["type"] != "message":
                return action["id"]

    def process_to_action(self, id):
        last_processed_action_id = self.raw_actions[-1]["id"] if self.raw_actions and self.raw_actions[-1] else 0
        for index, action in enumerate(self.raw_all_actions):
            if self.exception:
                continue
            if action["id"] <= last_processed_action_id:
                continue
            if action["id"] > id:
                break
            if self._filtered_actions[index]:
                self.process_action(action)
                self.raw_actions[-1]["id"] = action["id"]
                self.last_processed_action = action["id"]
            else:
                self.raw_actions.append(action)

    def next_turn(self):
        if self.turn_start_action_id != self.current_action_id:
            self.last_turn_start_action_id = self.turn_start_action_id
            self.turn_start_action_id = self.current_action_id

    def clone(self, actions):
        return self.__class__(
            self.names,
            id=self.id,
            pin=self.pin,
            seed=self.seed,
            actions=[action.copy() for action in actions] if actions else None,
            optional_rules=self.optional_rules,
        )

    def deep_copy_clone(self) -> 'BaseGame':
        try:
            return copy.deepcopy(self)
        except Exception as e:
            logging.error(f"Error during deepcopy of {self.__class__.__name__}: {e}", exc_info=True)
            raise
    
    def manual_big_clone(self) -> 'BaseGame':
        memo = {}
        new_game = self.__class__.__new__(self.__class__)
        memo[id(self)] = new_game

        # Copy all of the immutable objects and unused properties
        new_game.metadata = self.metadata
        new_game.entities = self.entities
        new_game.map = self.map
        new_game.id = self.id
        new_game.turn = self.turn
        new_game.final_turn = self.final_turn
        new_game.strict = self.strict
        new_game.finished = self.finished
        new_game.log = self.log
        new_game.queued_log = self.queued_log
        new_game.turn_start_action_id = self.turn_start_action_id
        new_game.last_turn_start_action_id = self.last_turn_start_action_id
        new_game.exception = self.exception
        new_game.names = self.names
        new_game.user = self.user
        new_game.programmed_actions = self.programmed_actions
        new_game.round_counter = self.round_counter
        new_game.optional_rules = self.optional_rules
        new_game.seed = self.seed
        new_game.rand = copy.deepcopy(self.rand, memo)
        new_game.corporations_are_closing = self.corporations_are_closing
        new_game._axes = self._axes
        new_game.tile_groups = self.tile_groups
        new_game._cert_limit = self._cert_limit
        new_game.removals = self.removals
        new_game.initial_round_type = self.initial_round_type
        new_game.loans = self.loans
        new_game.total_loans = self.total_loans

        # Copy the mutable collections
        new_game.actions = [action.copy() for action in self.raw_all_actions]
        new_game.raw_actions = [action.copy() for action in self.raw_all_actions]
        new_game.raw_all_actions = [action.copy() for action in self.raw_all_actions]
        new_game._filtered_actions = [action.copy() for action in self.raw_all_actions]
        new_game.round_history = [copy.deepcopy(rh, memo) for rh in self.round_history]

        # Now, let's copy all the stuff that has state.
        new_game.players = [copy.deepcopy(player, memo) for player in self.players]
        new_game.corporations = [copy.deepcopy(corp, memo) for corp in self.corporations]
        new_game.companies = [copy.deepcopy(comp, memo) for comp in self.companies]
        new_game.minors = [copy.deepcopy(minor, memo) for minor in self.minors]
        new_game.bank = copy.deepcopy(self.bank, memo)
        new_game.share_pool = copy.deepcopy(self.share_pool, memo)
        new_game.stock_market = copy.deepcopy(self.stock_market, memo)
        new_game.depot = copy.deepcopy(self.depot, memo)
        new_game.phase = copy.deepcopy(self.phase, memo)
        new_game.round = copy.deepcopy(self.round, memo)
        new_game.closing_queue = [copy.deepcopy(item, memo) for item in self.closing_queue]
        if self._crowded_corps is not None:
            new_game._crowded_corps = [copy.deepcopy(corp, memo) for corp in self._crowded_corps]
        else:
            new_game._crowded_corps = None
        new_game.tiles = [copy.deepcopy(tile, memo) for tile in self.tiles]
        new_game.all_tiles = [copy.deepcopy(tile, memo) for tile in self.all_tiles] # May overlap with new_game.tiles
        new_game.cities = [copy.deepcopy(city, memo) for city in self.cities]
        new_game.hexes = [copy.deepcopy(hex_obj, memo) for hex_obj in self.hexes]
        new_game.graph = copy.deepcopy(self.graph, memo)
        new_game.operating_rounds = copy.deepcopy(self.operating_rounds, memo)
        new_game.cache_objects()
        return new_game


    def load_from_dict_wip(self) -> 'BaseGame':
        # Let's write a method to save all of the relevant game data to a dict.
        game_dict = {}
        # First, let's copy all the stuff that has no state or links to other objects.
        game_dict["class"] = self.__class__
        game_dict["metadata"] = self.metadata
        game_dict["entities"] = self.entities
        game_dict["map"] = self.map
        game_dict["id"] = self.id
        game_dict["turn"] = self.turn
        game_dict["final_turn"] = self.final_turn
        game_dict["strict"] = self.strict
        game_dict["finished"] = self.finished
        game_dict["log"] = self.log
        game_dict["queued_log"] = self.queued_log
        game_dict["turn_start_action_id"] = self.turn_start_action_id
        game_dict["last_turn_start_action_id"] = self.last_turn_start_action_id
        game_dict["exception"] = self.exception
        game_dict["names"] = self.names
        game_dict["user"] = self.user
        game_dict["programmed_actions"] = self.programmed_actions
        game_dict["round_counter"] = self.round_counter
        game_dict["optional_rules"] = self.optional_rules
        game_dict["seed"] = self.seed
        game_dict["rand"] = self.rand
        game_dict["corporations_are_closing"] = self.corporations_are_closing
        game_dict["axes"] = self._axes
        game_dict["tile_groups"] = self.tile_groups
        game_dict["_cert_limit"] = self._cert_limit
        game_dict["removals"] = self.removals
        game_dict["round_history"] = self.round_history
        game_dict["initial_round_type"] = self.initial_round_type
        game_dict["actions"] = self.raw_all_actions
        game_dict["loans"] = self.loans
        game_dict["total_loans"] = self.total_loans

        # Then, let's copy all the stuff that has state. This is gonna be complicated lol.
        # I think the best approach is to define a to_dict method for each class that needs it.
        # Then, we can define a from_dict method that will take a game and a dict and create
        # the object. From_dict will need to enforce the order of operations such that we can
        # load the game in the correct order.

        # Let's start with the players.
        game_dict["players"] = [player.to_dict() for player in self.players]

        # All of the things requiring a deep copy:
        game_dict["players"] = self.players
        game_dict["round"] = self.round
        game_dict["companies"] = self.companies
        game_dict["stock_market"] = self.stock_market
        game_dict["minors"] = self.minors
        game_dict["corporations"] = self.corporations
        game_dict["closing_queue"] = self.closing_queue
        game_dict["crowded_corps"] = self._crowded_corps
        game_dict["bank"] = self.bank
        game_dict["tiles"] = self.tiles
        game_dict["all_tiles"] = self.all_tiles
        game_dict["cities"] = self.cities
        game_dict["depot"] = self.depot
        game_dict["share_pool"] = self.share_pool
        game_dict["hexes"] = self.hexes
        game_dict["graph"] = self.graph
        game_dict["phase"] = self.phase
        game_dict["operating_rounds"] = self.operating_rounds

        new_game = game_dict["class"].__new__(game_dict["class"])
        # We now have a fresh game object.
        # Let's load the rest of the game from the dict.
        new_game.metadata = game_dict["metadata"]
        new_game.entities = game_dict["entities"]
        new_game.map = game_dict["map"]
        new_game.title = new_game.metadata.title()
        new_game.id = game_dict["id"]
        new_game.turn = game_dict["turn"]
        new_game.final_turn = game_dict["final_turn"]
        new_game.loading = False
        new_game.strict = game_dict["strict"]
        new_game.finished = game_dict["finished"]
        new_game.log = game_dict["log"]
        new_game.queued_log = game_dict["queued_log"]
        new_game.turn_start_action_id = game_dict["turn_start_action_id"]
        new_game.last_turn_start_action_id = game_dict["last_turn_start_action_id"]
        new_game.exception = game_dict["exception"]
        new_game.names = game_dict["names"]
        new_game.players = game_dict["players"]
        new_game.user = game_dict["user"]
        new_game.programmed_actions = game_dict["programmed_actions"]
        new_game.round = game_dict["round"]
        new_game.round_counter = game_dict["round_counter"]
        new_game.optional_rules = game_dict["optional_rules"]
        new_game.seed = game_dict["seed"]
        new_game.rand = game_dict["rand"]
        new_game.companies = game_dict["companies"]
        new_game.stock_market = game_dict["stock_market"]
        new_game.minors = game_dict["minors"]
        new_game.loans = game_dict["loans"]
        new_game.total_loans = game_dict["total_loans"]
        new_game.corporations = game_dict["corporations"]
        new_game.closing_queue = game_dict["closing_queue"]
        new_game.corporations_are_closing = game_dict["corporations_are_closing"]
        new_game._axes = game_dict["axes"]
        new_game._crowded_corps = game_dict["crowded_corps"]
        new_game.bank = game_dict["bank"]
        new_game.tiles = game_dict["tiles"]
        new_game.all_tiles = game_dict["all_tiles"]
        new_game.tile_groups = game_dict["tile_groups"]
        new_game._cert_limit = game_dict["_cert_limit"]
        new_game.removals = game_dict["removals"]
        new_game.depot = game_dict["depot"]
        new_game.share_pool = game_dict["share_pool"]
        new_game.hexes = game_dict["hexes"]
        new_game.graph = game_dict["graph"]
        new_game.cities = game_dict["cities"]
        new_game.phase = game_dict["phase"]
        new_game.operating_rounds = game_dict["operating_rounds"]
        new_game.round_history = game_dict["round_history"]
        new_game.cache_objects()
        new_game.initial_round_type = game_dict["initial_round_type"]
        new_game.actions = game_dict["actions"]
        new_game.raw_actions = game_dict["actions"]
        new_game._filtered_actions = game_dict["actions"]
        new_game.raw_all_actions = game_dict["actions"]

        return new_game

    @property
    def trains(self):
        return self.depot.trains

    def train_limit(self, entity):
        return self.phase.train_limit(entity) + sum(
            ability.increase for ability in self.abilities(entity, "train_limit")
        )

    def train_owner(self, train):
        return train.owner

    def route_trains(self, entity):
        return entity.runnable_trains

    def discarded_train_placement(self):
        return self.DISCARDED_TRAINS

    def should_rust(self, train, purchased_train):
        return train.rusts_on == purchased_train.sym or (
            train.obsolete_on == purchased_train.sym and train in self.depot.discarded
        )

    def obsolete(self, train, purchased_train):
        return train.obsolete_on == purchased_train.sym

    @property
    def shares(self):
        return (
            [share for corp in self.corporations for share in corp.shares]
            + [share for player in self.players for share in player.shares]
            + self.share_pool.shares
        )

    @property
    def share_prices(self):
        return self.stock_market.par_prices

    @property
    def layout(self):
        return self.map.LAYOUT

    @property
    def axes(self):
        if self._axes:
            return self._axes
        elif self.layout == "flat":
            self._axes = {"x": "letter", "y": "number"}
        elif self.layout == "pointy":
            self._axes = {"x": "number", "y": "letter"}
        return self._axes

    def format_currency(self, val):
        return self.CURRENCY_FORMAT_STR.format(val)

    def format_revenue_currency(self, val):
        return self.format_currency(val)

    def routes_subsidy(self, routes):
        return 0

    def submit_revenue_str(self, routes, show_subsidy):
        revenue_str = self.format_revenue_currency(self.routes_revenue(routes))
        subsidy = self.routes_subsidy(routes)
        subsidy_str = f" + {self.format_currency(subsidy)} (subsidy)" if show_subsidy or subsidy > 0 else ""
        return revenue_str + subsidy_str

    def purchasable_companies(self, entity=None):
        return [
            company
            for company in self.companies
            if company.owner
            and company.owner.is_player()
            and entity != company.owner
            and not self.abilities(company, "no_buy")
        ]

    @property
    def buyable_bank_owned_companies(self):
        return [company for company in self.companies if not company.is_closed() and company.owner == self.bank]

    def after_buy_company(self, player, company, price):
        for ability in self.abilities(company, "shares"):
            for share in ability.shares:
                if share.president:
                    self.round.companies_pending_par.append(company)
                else:
                    self.share_pool.buy_shares(player, share, exchange="free")
        for ability in self.abilities(company, "acquire_company"):
            acquired_company = self.company_by_id(ability.company)
            acquired_company.owner = player
            player.companies.append(acquired_company)
            self.log.append(f"{player.name} receives {acquired_company.name}")
            self.after_buy_company(player, acquired_company, 0)

    def after_sell_company(self, buyer, company, price, seller):
        pass

    def player_value(self, player):
        return player.value

    def liquidity(self, player, emergency=False):
        if not self.sellable_turn():
            return player.cash

        value = player.cash
        if emergency:
            if not self.round:
                return self.liquidity(player)

            value += sum(
                self.value_for_sellable(player, corporation)
                for corporation, shares in player.shares_by_corporation.items()
                if shares
            )
        else:
            for corporation, _ in player.shares_by_corporation.items():
                if self.SELL_AFTER == "operate":
                    if not corporation.operated():
                        continue
                elif self.SELL_AFTER == "p_any_operate":
                    if not corporation.operated() and not corporation.president(player):
                        continue

                value += self.value_for_dumpable(player, corporation)

        return value

    def check_sale_timing(self, entity, bundle):
        corporation = bundle.corporation

        if self.SELL_AFTER == "first":
            return self.turn > 1 or (self.round and self.round.operating)
        elif self.SELL_AFTER == "after_ipo":
            return corporation.operated() or (self.round and self.round.operating)
        elif self.SELL_AFTER == "operate":
            return corporation.operated()
        elif self.SELL_AFTER == "p_any_operate":
            return corporation.operated() or corporation.president(entity)
        elif self.SELL_AFTER == "full_or_turn":
            if self.round and self.round.operating and corporation == self.round.current_operator:
                return len(corporation.operating_history) > 1
            else:
                return corporation.operated()
        elif self.SELL_AFTER == "round":
            if self.round and self.round.stock():
                return (
                    corporation.share_holders[entity] - self.round.players_bought[entity][corporation]
                ) >= bundle.percent
            else:
                return corporation.operated()
        elif self.SELL_AFTER == "any_time":
            return True
        else:
            raise NotImplementedError

    def value_for_sellable(self, player, corporation):
        max_bundle = max(
            self.sellable_bundles(player, corporation),
            key=lambda bundle: bundle.price,
            default=None,
        )
        return max_bundle.price if max_bundle else 0

    def value_for_dumpable(self, player, corporation):
        if self.PRESIDENT_SALES_TO_MARKET:
            return self.value_for_sellable(player, corporation)

        dumpable_bundles = [
            bundle
            for bundle in self.bundles_for_corporation(player, corporation)
            if bundle.can_dump(player) and (self.share_pool is None or self.share_pool.fit_in_bank(bundle))
        ]
        max_bundle = max(dumpable_bundles, key=lambda bundle: bundle.price, default=None)
        return max_bundle.price if max_bundle else 0

    def issuable_shares(self, entity):
        return []

    def redeemable_shares(self, entity):
        return []

    def sellable_bundles(self, player, corporation):
        if not hasattr(self.round.active_step(), "can_sell"):
            return []

        bundles = self.bundles_for_corporation(player, corporation)
        return [bundle for bundle in bundles if self.round.active_step().can_sell(player, bundle)]

    def bundles_for_corporation(self, share_holder, corporation, shares=None):
        return self.all_bundles_for_corporation(share_holder, corporation, shares=shares)

    def all_bundles_for_corporation(self, share_holder, corporation, shares=None):
        if not corporation.ipoed:
            return []

        if shares is None:
            shares = share_holder.shares_of(corporation)

        if not shares:
            return []

        shares.sort(key=lambda h: (1 if h.president else 0, h.percent))
        bundle = []
        percent = 0
        all_bundles = []
        for share in shares:
            bundle.append(share)
            percent += share.percent
            all_bundles.append(ShareBundle(bundle[:], percent))

        if shares[-1].president:
            all_bundles += self.partial_bundles_for_presidents_share(corporation, bundle[:], percent)

        return sorted(all_bundles, key=lambda bundle: bundle.percent)

    def partial_bundles_for_presidents_share(self, corporation, bundle, percent):
        normal_percent = corporation.share_percent
        difference = corporation.presidents_percent - normal_percent
        num_partial_bundles = difference / normal_percent
        return [ShareBundle(bundle[:], percent - (normal_percent * n)) for n in range(1, int(num_partial_bundles) + 1)]

    def can_buy_presidents_share_directly_from_market(self, corporation):
        return False

    def can_swap_for_presidents_share_directly_from_corporation(self):
        return True

    def shares_for_presidency_swap(self, shares, num_shares):
        return shares[:num_shares]

    def num_certs(self, entity):
        certs = sum(
            s.cert_size if s.corporation().counts_for_limit() and s.counts_for_limit else 0 for s in entity.shares
        )
        if self.CERT_LIMIT_INCLUDES_PRIVATES:
            certs += len(entity.companies)
        return certs

    def sellable_turn(self):
        return self.SELL_AFTER != "first" or (self.turn > 1 or not (self.round and self.round.stock()))

    @property
    def sell_movement(self):
        return self.SELL_MOVEMENT

    def sell_shares_and_change_price(self, bundle, allow_president_change=True, swap=None, movement=None):
        corporation = bundle.corporation
        old_price = corporation.share_price
        was_president = corporation.president(bundle.owner)
        self.share_pool.sell_shares(bundle, allow_president_change=allow_president_change, swap=swap)
        movement = movement or self.sell_movement
        if movement == "down_share":
            for _ in range(bundle.num_shares()):
                self.stock_market.move_down(corporation)
        elif movement == "down_per_10":
            percent = bundle.percent
            if swap:
                percent -= swap.percent
            for _ in range(int(percent / 10)):
                self.stock_market.move_down(corporation)
        elif movement == "down_block":
            self.stock_market.move_down(corporation)
        elif movement == "left_share":
            for _ in range(bundle.num_shares()):
                self.stock_market.move_left(corporation)
        elif movement == "left_share_pres":
            for _ in range(bundle.num_shares()):
                if was_president:
                    self.stock_market.move_left(corporation)
        elif movement == "left_block":
            self.stock_market.move_left(corporation)
        elif movement == "down_block_pres":
            if was_president:
                self.stock_market.move_down(corporation)
        elif movement == "left_block_pres":
            if was_president:
                self.stock_market.move_left(corporation)
        elif movement == "left_per_10_if_pres_else_left_one":
            spaces = int((percent - (swap.percent if swap else 0)) / 10)
            for _ in range(spaces):
                self.stock_market.move_left(corporation)
        elif movement == "none":
            pass
        else:
            raise NotImplementedError

        if movement != "none":
            self.log_share_price(corporation, old_price)

    def sold_out_increase(self, corporation):
        return self.SOLD_OUT_INCREASE

    def log_share_price(self, entity, from_price, steps=None, log_steps=False):
        to = entity.share_price
        to_price = to.price
        if from_price != to:
            jumps = ""
            if steps and (steps > 1 or log_steps):
                jumps = f" ({steps} step{'s' if steps != 1 else ''})"
            r1, c1 = from_price.coordinates
            r2, c2 = to.coordinates
            dirs = []
            if r2 < r1:
                dirs.append("up")
            if r2 > r1:
                dirs.append("down")
            if c2 < c1:
                dirs.append("left")
            if c2 > c1:
                dirs.append("right")
            dir_str = " and ".join(dirs)
            self.log.append(
                f"{entity.name}'s share price moves {dir_str} from {self.format_currency(from_price.price)} to {self.format_currency(to_price)}{jumps}"
            )

    def consenter_for_buy_shares(self, entity, bundle):
        return None

    def consenter_for_choice(self, entity, choice, label):
        return None

    def can_run_route(self, entity):
        return (
            self.graph_for_entity(entity) is not None
            and self.graph_for_entity(entity).route_info(entity) is not None
            and self.graph_for_entity(entity).route_info(entity).get("route_available")
        )

    def must_buy_train(self, entity):
        return (
            not entity.trains
            and self.depot.depot_trains()
            and (
                self.MUST_BUY_TRAIN == "always"
                or (self.MUST_BUY_TRAIN == "route" and self.graph.route_info(entity) and self.graph.route_info(entity).get("route_train_purchase"))
            )
        )

    def can_buy_train_from_others(self):
        if self.temp_allow_cross_company_purchase:
            return True
        return self.ALLOW_TRAIN_BUY_FROM_OTHERS

    def discard_discount(self, train, price):
        if not self.DISCARDED_TRAIN_DISCOUNT or train not in self.depot.discarded:
            return price
        return int(price * (100.0 - float(self.DISCARDED_TRAIN_DISCOUNT)) / 100.0)

    def end_game(self, player_initiated=False):
        if self.finished:
            return

        self.finished = True
        self.manually_ended = player_initiated
        self.store_player_info()
        self.round_counter += 1
        scores = [
            f"{self.player_by_id(id).name} ({self.format_currency(value)})" for id, value in self.result().items()
        ]
        self.log.append(f"-- Game over: {', '.join(scores)} --")

    def revenue_for(self, route, stops):
        return sum(stop.route_revenue(route.phase, route.train) for stop in stops)

    def revenue_str(self, route):
        return "-".join(hex.name for hex in route.hexes)

    def float_str(self, entity):
        if entity.is_corporation() and entity.floatable:
            return f"{entity.percent_to_float}% to float"

    def route_distance_str(self, route):
        return str(self.route_distance(route))

    def route_distance(self, route):
        return sum(stop.visit_cost for stop in route.visited_stops)

    def routes_revenue(self, routes):
        return sum(route.revenue() for route in routes)

    def extra_revenue(self, entity, routes):
        return 0

    def compute_other_paths(self, routes, route):
        return [path for r in routes if r != route for path in r.paths]

    def city_tokened_by(self, city, entity):
        return city.tokened_by(entity)

    def check_route_token(self, route, token):
        if not token:
            raise NoToken("Route must contain token")

    def check_overlap(self, routes):
        tracks = {}

        def check(key):
            if key in tracks:
                raise GameError(f"Route cannot reuse track on {key[0].id}")
            tracks[key] = True

        for route in routes:
            if route is None:
                continue
            for path in route.paths:
                a = path.a
                b = path.b

                if a.is_edge():
                    check((path.hex, a.num, path.lanes[0][1]))
                if b.is_edge():
                    check((path.hex, b.num, path.lanes[1][1]))

                if b.is_edge() and a.is_town():
                    nedge = a.tile.preferred_city_town_edges.get(a)
                    if nedge and nedge != b.num:
                        check((path.hex, a, path.lanes[0][1]))
                if a.is_edge() and b.is_town():
                    nedge = b.tile.preferred_city_town_edges.get(b)
                    if nedge and nedge != a.num:
                        check((path.hex, b, path.lanes[1][1]))

                if len(path.nodes) > 1:
                    check((path.hex, path))

    def check_connected(self, route, corporation):
        if not all(a.connects_to(b, corporation) for a, b in zip(route.ordered_paths, route.ordered_paths[1:])):
            raise GameError("Route is not connected")

    def check_distance(self, route, visits, train=None):
        train = train or route.train
        distance = train.distance
        if isinstance(distance, int):
            route_distance = sum(visit.visit_cost for visit in visits)
            if distance < route_distance:
                raise RouteTooLong(f"{route_distance} is too many stops for {distance} train")
            return

        type_info = defaultdict(list)

        for h in distance:
            pay = h["pay"]
            visit = h["visit"] or pay
            info = {"pay": pay, "visit": visit}
            for stop_type in h["nodes"]:
                type_info[stop_type].append(info)

        grouped = defaultdict(list)

        for visit in visits:
            grouped[stop_type(visit, train)].append(visit)

        grouped = dict(sorted(grouped.items(), key=lambda x: len(type_info[x[0]])))

        for stop_type, group in grouped.items():
            num = sum(visit.visit_cost for visit in group)

            for info in type_info[stop_type]:
                if info["visit"] > 0:
                    if num <= info["visit"]:
                        info["visit"] -= num
                        num = 0
                    else:
                        num -= info["visit"]
                        info["visit"] = 0
                    if not num > 0:
                        break

            if num > 0:
                raise RouteTooLong("Route has too many stops")

    def check_other(self, route):
        pass

    def compute_stops(self, route, train=None):
        train = train or route.train
        visits = self.revenue_stops(route)
        distance = train.distance
        if isinstance(distance, int):
            return visits if visits else []

        max_num_stops = min(sum(h["pay"] for h in distance), len(visits))
        for num_stops in range(max_num_stops, 0, -1):
            for stops_combination in combinations(visits, num_stops):
                if train.requires_token and not any(stop.tokened_by(route.corporation) for stop in stops_combination):
                    continue
                types_used = [0] * len(distance)
                valid_stops = True
                for stop in stops_combination:
                    found_row = None
                    for i, h in enumerate(distance):
                        if self.stop_type(stop, train) in h["nodes"] and types_used[i] < h["pay"]:
                            found_row = i
                            break
                    if found_row is not None:
                        types_used[found_row] += 1
                    else:
                        valid_stops = False
                        break
                if valid_stops:
                    return stops_combination
        return []

    def stop_type(self, stop, train):
        return stop.type

    def visited_stops(self, route):
        return list(
            set(
                [c["left"] for c in route.connection_data if c["left"] is not None]
                + [c["right"] for c in route.connection_data if c["right"] is not None]
            )
        )

    def revenue_stops(self, route):
        return self.visited_stops(route)

    def get(self, type, id):
        if not type or id is None:
            return None
        return getattr(self, f"{type}_by_id")(id)

    def all_companies_with_ability(self, ability_type):
        for company in self.companies:
            for ability in self.abilities(company, ability_type):
                yield company, ability

    def payout_companies(self, ignore=[]):
        companies = [c for c in self.companies if c.owner and c.revenue > 0 and c.id not in ignore]

        companies.sort(
            key=lambda company: (0, self.players.index(company.owner))
            if company.owned_by_player()
            else (1, company.owner, company.name)
        )

        for company in companies:
            owner = company.owner
            if owner != self.bank:
                revenue = company.revenue
                self.bank.spend(revenue, owner)
                self.log.append(f"{owner.name} collects {self.format_currency(revenue)} from {company.name}")

    def init_round_finished(self):
        pass

    def or_round_finished(self):
        pass

    def or_set_finished(self):
        pass

    def home_token_locations(self, corporation):
        raise NotImplementedError

    def home_token_can_be_cheater(self):
        return False

    def place_home_token(self, corporation):
        # set_trace()
        if not corporation.next_token():
            return
        if corporation.tokens[0].used:
            return

        hex = self.hex_by_id(corporation.coordinates)
        tile = hex.tile if hex else None

        if not tile or (tile.reserved_by(corporation) and any(tile.paths)):
            if any(p["entity"] == corporation for p in self.round.pending_tokens):
                self.round.clear_cache()
                return

            hexes = [hex] if hex else self.home_token_locations(corporation)
            if not hexes:
                return

            self.log.append(f"{corporation.name} must choose city for home token")
            self.round.pending_tokens.append(
                {
                    "entity": corporation,
                    "hexes": hexes,
                    "token": corporation.find_token_by_type(),
                }
            )

            self.round.clear_cache()
            return
        else:
            cities = tile.cities
            city = next(
                (c for c in cities if c.reserved_by(corporation)),
                cities[0] if cities else None,
            )
            token = corporation.find_token_by_type()
            if city and city.tokenable(corporation, tokens=[token]):
                self.log.append(f"{corporation.name} places a token on {hex.name}")
                city.place_token(corporation, token)
                self.graph.clear_graph_for(corporation)
            elif self.home_token_can_be_cheater():
                self.log.append(f"{corporation.name} places a token on {hex.name}")
                city.place_token(corporation, token, cheater=True)
                self.graph.clear_graph_for(corporation)

    def graph_for_entity(self, entity):
        return self.graph

    def token_graph_for_entity(self, entity):
        return self.graph

    def clear_graph(self):
        self.graph.clear()

    def clear_graph_for_entity(self, entity):
        self.graph_for_entity(entity).clear()

    def clear_token_graph_for_entity(self, entity):
        self.token_graph_for_entity(entity).clear()

    def graph_skip_paths(self, entity):
        return None

    def upgrade_cost(self, tile, hex, entity, spender):
        if not entity.is_corporation() and entity.owner and entity.owner.corporation:
            entity = entity.owner
        ability = next(
            (a for a in entity.all_abilities if a.type == "tile_discount" and (not a.hexes or hex.name in a.hexes)),
            None,
        )

        discount = ability.discount if ability and ability.discounts_tile(tile) else 0
        self.log_cost_discount(spender, ability, discount)

        return sum([upgrade.cost for upgrade in tile.upgrades]) - discount

    def tile_cost_with_discount(self, tile, hex, entity, spender, cost):
        if not entity.is_corporation() and entity.owner and entity.owner.corporation:
            entity = entity.owner
        ability = next(
            (
                a
                for a in entity.all_abilities
                if a.type == "tile_discount" and (not a.terrain) and (not a.hexes or hex.name in a.hexes)
            ),
            None,
        )

        if not ability:
            return cost

        discount = min(cost, ability.discount)
        self.log_cost_discount(spender, ability, discount)

        return cost - discount

    def log_cost_discount(self, spender, abilities, discount):
        if discount <= 0:
            return

        owners = ", ".join([a.owner.name for a in abilities])
        self.log.append(f"{spender.name} receives a discount of {self.format_currency(discount)} from {owners}")

    def declare_bankrupt(self, player):
        if player.bankrupt:
            msg = f"{player.name} is already bankrupt, cannot declare bankruptcy again."
            raise GameError(msg)

        player.bankrupt = True
        if self.CERT_LIMIT_CHANGE_ON_BANKRUPTCY:
            self._cert_limit = self.init_cert_limit()

    def tile_lays(self, entity):
        return self.TILE_LAYS

    def upgrades_to(self, from_tile, to_tile, special=False, selected_company=None):
        if not self.upgrades_to_correct_color(from_tile, to_tile, selected_company=selected_company):
            return False

        if not from_tile.paths_are_subset_of(to_tile.paths):
            return False

        if special:
            return True

        if not self.upgrades_to_correct_label(from_tile, to_tile):
            return False

        if not self.upgrades_to_correct_city_town(from_tile, to_tile):
            return False

        return True

    def upgrade_ignore_num_cities(self, from_tile):
        return False

    def upgrades_to_correct_color(self, from_tile, to_tile, selected_company=None):
        return Tile.COLORS.index(to_tile.color) == (Tile.COLORS.index(from_tile.color) + 1)

    def upgrades_to_correct_label(self, from_tile, to_tile):
        if from_tile.future_label and to_tile.color == from_tile.future_label.color:
            return from_tile.future_label.label == str(to_tile.label) if to_tile.label else False
        return from_tile.label == to_tile.label

    def upgrades_to_correct_city_town(self, from_tile, to_tile):
        if len(from_tile.towns) != len(to_tile.towns):
            return False
        if (
            not from_tile.label
            and len(from_tile.cities) != len(to_tile.cities)
            and not self.upgrade_ignore_num_cities(from_tile)
        ):
            return False
        if (
            len(from_tile.cities) > 1
            and len(to_tile.cities) > 1
            and not from_tile.city_town_edges_are_subset_of(to_tile.city_town_edges)
        ):
            return False
        if from_tile.label and not from_tile.cities and to_tile.cities:
            return False
        if from_tile.color == "white" and str(from_tile.label) == "OO" and len(from_tile.cities) != len(to_tile.cities):
            return False
        return True

    def legal_tile_rotation(self, entity, hex, tile):
        return True

    def can_par(self, corporation, parrer):
        if corporation.par_via_exchange and corporation.par_via_exchange.owner != parrer:
            return False
        if corporation.needs_token_to_par and not corporation.tokens:
            return False
        if any(ability.type == "unparrable" for ability in corporation.all_abilities):
            return False
        return not corporation.ipoed

    def company_sellable(self, company):
        return not isinstance(company.owner, Corporation)

    def unowned_purchasable_companies(self, entity):
        return []

    def multiple_buy_only_from_market(self):
        return self.MULTIPLE_BUY_ONLY_FROM_MARKET

    def float_corporation(self, corporation):
        self.log.append(f"{corporation.name} floats")
        if corporation.capitalization not in ["incremental", "none"]:
            self.bank.spend(corporation.par_price().price * corporation.total_shares, corporation)
            self.log.append(f"{corporation.name} receives {self.format_currency(corporation.cash)}")

    def total_shares_to_float(self, corporation, price):
        return corporation.percent_to_float / corporation.share_percent

    def close_corporation(self, corporation, quiet=False):
        if not quiet:
            self.log.append(f"{corporation.name} closes")

        for hex in self.hexes:
            for city in hex.tile.cities:
                tokens = [t for t in city.tokens if t and t.corporation == corporation]
                for token in tokens:
                    token.remove()
                if self.CLOSED_CORP_TOKENS_REMOVED:
                    city.tokens = [t for t in city.tokens if not (t and t.corporation == corporation)]

                if self.CLOSED_CORP_RESERVATIONS_REMOVED and corporation in city.reservations:
                    city.reservations.remove(corporation)

            if self.CLOSED_CORP_RESERVATIONS_REMOVED and corporation in hex.tile.reservations:
                hex.tile.reservations.remove(corporation)

        if corporation.cash > 0:
            corporation.spend(corporation.cash, self.bank)

        if self.CLOSED_CORP_TRAINS_REMOVED:
            for train in corporation.trains:
                train.buyable = False
        else:
            for train in corporation.trains[:]:
                self.depot.reclaim_train(train)

        if corporation.companies:
            self.log.append(
                f"{corporation.name}'s companies close: {', '.join([c.sym for c in corporation.companies])}"
            )
            for company in corporation.companies[:]:
                company.close()

        if self.round.current_entity == corporation:
            self.round.force_next_entity()

        if corporation.corporation:
            for share_holder in corporation.share_holders.keys():
                share_holder.shares_by_corporation.pop(corporation, None)

            self.share_pool.shares_by_corporation.pop(corporation, None)
            if corporation.share_price:
                corporation.share_price.corporations.discard(corporation)
            self.corporations.discard(corporation)
        else:
            self.minors.discard(corporation)

        corporation.close()
        self._cert_limit = self.init_cert_limit()

        self.close_corporations_in_close_cell()

    def shares_for_corporation(self, corporation):
        return [share for share in self._shares.values() if share.corporation == corporation]

    def reset_corporation(self, corporation):
        for share_id, share in list(self._shares.items()):
            if share.corporation == corporation:
                share.owner.shares_by_corporation[corporation].clear()
                del self._shares[share_id]

        for company in list(corporation.companies):
            company.close()

        if corporation.share_price:
            corporation.share_price.corporations.discard(corporation)

        new_corporation = next(
            (c for c in self.init_corporations(self.stock_market) if c.id == corporation.id),
            None,
        )

        if new_corporation:
            self.corporations = [new_corporation if c.id == new_corporation.id else c for c in self.corporations]
            self._corporations[new_corporation.id] = new_corporation
            for share in new_corporation.shares:
                self._shares[share.id] = share

        return new_corporation

    def emergency_issuable_bundles(self, corporation):
        return []

    def emergency_issuable_cash(self, corporation):
        return max(
            (bundle.num_shares() * bundle.price for bundle in self.emergency_issuable_bundles(corporation)),
            default=0,
        )

    def can_go_bankrupt(self, player, corporation):
        if not self.BANKRUPTCY_ALLOWED:
            return False

        return self.total_emr_buying_power(player, corporation) < self.depot.min_depot_price

    def total_emr_buying_power(self, player, corporation):
        buying_power = self.liquidity(player, emergency=True)
        if corporation:
            buying_power += corporation.cash + self.emergency_issuable_cash(corporation)
        return buying_power

    def buying_power(self, entity, **kwargs):
        return entity.cash

    def company_sale_price(self, company):
        raise NotImplementedError

    def two_player(self):
        return len(self.players) == 2

    def add_extra_tile(self, tile):
        if not tile.unlimited:
            raise GameError("Add extra tile only works if unlimited")

        tiles = [t for t in self._tiles.values() if t.name == tile.name]
        new_tile = max(tiles, key=lambda t: t.index).copy()
        self.tiles.append(new_tile)
        self._tiles[new_tile.id] = new_tile
        extra_cities = new_tile.cities
        self.cities.extend(extra_cities)
        for city in extra_cities:
            self._cities[city.id] = city
        return new_tile

    def find_share_price(self, price):
        return next(
            (sp for sp in reversed(self.stock_market.market[0]) if sp.price <= price),
            None,
        )

    def after_par(self, corporation):
        if corporation.capitalization == "incremental":
            for company, ability in self.all_companies_with_ability("shares"):
                if corporation.name == ability.shares[0].corporation.name:
                    amount = sum(corporation.par_price().price * share.num_shares() for share in ability.shares)
                    self.bank.spend(amount, corporation)
                    self.log.append(f"{corporation.name} receives {self.format_currency(amount)} from {company.name}")

        self.close_companies_on_event(corporation, "par")
        if self.HOME_TOKEN_TIMING == "par":
            self.place_home_token(corporation)

    def close_companies_on_event(self, entity, event):
        for company in self.companies:
            if not company.is_closed():
                for ability in self.abilities(company, "close", time=event):
                    if entity and entity.name != ability.corporation:
                        continue
                    company.close()
                    if not ability.silent:
                        self.log.append(f"{company.name} closes")

    def train_help(self, entity, runnable_trains, routes):
        return []

    def queue_log(self):
        old_size = len(self.log)
        yield
        self.queued_log = self.log[old_size:]
        self.log = self.log[:old_size]

    def flush_log(self):
        self.log.extend(self.queued_log)
        self.queued_log = []

    def company_bought(self, company, buyer):
        pass

    def ipo_name(self, entity=None):
        return "IPO"

    def ipo_verb(self, entity=None):
        return "pars"

    def ipo_reserved_name(self, entity=None):
        return "IPO Reserved"

    def share_flags(self, shares):
        return None

    def corporation_show_loans(self, corporation):
        return True

    def corporation_show_shares(self, corporation):
        return not corporation.minor

    def corporation_show_individual_reserved_shares(self, corporation):
        return True

    def abilities(
        self,
        entity,
        type=None,
        time=[],
        on_phase=None,
        passive_ok=None,
        strict_time=None,
        callable=False,
    ):
        if not entity:
            return []

        return [
            ability
            for ability in entity.all_abilities
            if self.ability_right_type(ability, type)
            and self.ability_right_owner(ability.owner, ability)
            and self.ability_usable_this_or(ability)
            and self.ability_right_time(
                ability,
                time,
                on_phase,
                passive_ok if passive_ok is not None else True,
                strict_time if strict_time is not None else True,
            )
            and self.ability_usable(ability)
        ]

    def ability_combo_entities(self, entity):
        if not entity.is_company():
            return []

        combo_entities = []
        for ability in self.abilities(entity, "tile_lay"):
            for company_id in ability.combo_entities:
                company = self.company_by_id(company_id)
                if company and company.owner == entity.corporation and self.abilities(company, "tile_lay"):
                    combo_entities.append(company)
        return combo_entities

    def valid_combos(self, companies):
        if len(companies) < 2:
            return True

        companies = [self.company_by_id(c) if isinstance(c, str) else c for c in companies]

        return all(
            all(c in self.ability_combo_entities(company) for company in companies[index + 1 :])
            for index, c in enumerate(companies)
        )

    def entity_can_use_company(self, entity, company):
        return True

    def buy_train(self, operator, train, price=None):
        if price is not None and price != "free":
            operator.spend(price if price != "free" else train.price, train.owner)
        self.remove_train(train)
        train.owner = operator
        operator.trains.append(train)
        self._crowded_corps = None
        self.close_companies_on_event(operator, "bought_train")

    def discountable_trains_for(self, corporation):
        discountable_trains = [
            train
            for train in self.depot.depot_trains()
            if train.discount or any(v.get("discount") for v in train.variants.values())
        ]

        discount_info = []
        for train in corporation.trains:
            for discount_train in discountable_trains:
                # Calculate discounted price for the base version
                base_discounted_price = discount_train.get_price(train)
                if discount_train.price > base_discounted_price:
                    discount_info.append(
                        [
                            train,
                            discount_train,
                            discount_train.name,
                            base_discounted_price,
                        ]
                    )

                # Add variants with discounts, excluding those with the same name as the base version
                for variant in discount_train.variants.values():
                    if variant["name"] != discount_train.name:
                        variant_discounted_price = discount_train.price(train, variant=variant)
                        if variant["price"] > variant_discounted_price:
                            discount_info.append(
                                [
                                    train,
                                    discount_train,
                                    variant["name"],
                                    variant_discounted_price,
                                ]
                            )

        return discount_info

    def remove_train(self, train):
        if train.owner:
            if train.from_depot():
                self.depot.remove_train(train)
            else:
                train.owner.trains.remove(train)
            self._crowded_corps = None

    def rust(self, train):
        train.rusted = True
        self.remove_train(train)
        train.owner = None

    def num_corp_trains(self, entity):
        return (
            sum(1 for t in entity.trains if not t.obsolete)
            if not self.OBSOLETE_TRAINS_COUNT_FOR_LIMIT
            else len(entity.trains)
        )

    @property
    def crowded_corps(self):
        if self._crowded_corps:
            return self._crowded_corps
        self._crowded_corps = [
            c for c in self.minors + self.corporations if self.num_corp_trains(c) > self.train_limit(c)
        ]
        return self._crowded_corps

    def transfer(self, ownable_type, from_entity, to_entity):
        ownables = getattr(from_entity, ownable_type)
        to_ownables = getattr(to_entity, ownable_type)

        self._crowded_corps = None if ownable_type == "trains" else self._crowded_corps

        transferred = list(ownables)
        ownables.clear()

        for ownable in transferred:
            ownable.owner = to_entity
            to_ownables.append(ownable)

        return transferred

    def exchange_for_partial_presidency(self):
        return False

    def exchange_partial_percent(self, share):
        return None

    def exchange_corporations(self, exchange_ability):
        if exchange_ability.corporations == "any":
            return [c for c in self.corporations if not c.is_closed()]
        if exchange_ability.corporations == "ipoed":
            return [c for c in self.corporations if c.ipoed and not c.is_closed()]
        return [
            self.corporation_by_id(c)
            for c in exchange_ability.corporations
            if not self.corporation_by_id(c).is_closed()
        ]

    def round_start(self):
        return self.last_game_action_id == self.round_history[-1]

    def can_hold_above_corp_limit(self, entity):
        return False

    def show_game_cert_limit(self):
        return True

    def cannot_pay_interest_str(self):
        return None

    def hex_blocked_by_ability(self, entity, ability, hex, tile=None):
        return hex.id in ability.hexes

    def rust_trains(self, train, entity):
        obsolete_trains = []
        removed_obsolete_trains = []
        rusted_trains = []
        owners = defaultdict(int)

        for t in self.trains:
            if t.obsolete or not self.obsolete(t, train):
                continue
            obsolete_trains.append(t.name)
            t.obsolete = True

        for t in self.trains:
            if t.rusted or not self.should_rust(t, train):
                continue
            if t.obsolete and t.owner == self.depot:
                removed_obsolete_trains.append(t.name)
            else:
                rusted_trains.append(t.name)
                owners[t.owner.name] += 1
            self.rust(t)

        self._crowded_corps = None

        if obsolete_trains:
            self.log.append(f"-- Event: {', '.join(set(obsolete_trains))} trains are obsolete --")
        if removed_obsolete_trains:
            self.log.append(
                f"-- Event: obsolete {', '.join(set(removed_obsolete_trains))} trains are removed from The Depot --"
            )

        if rusted_trains:
            self.log.append(
                f"-- Event: {', '.join(set(rusted_trains))} trains rust ({', '.join([f'{c} x{t}' for c, t in owners.items()])}) --"
            )

    def show_progress_bar(self):
        return False

    def progress_information(self):
        pass

    def assignment_tokens(self, assignment, simple_logos=False):
        if isinstance(assignment, Corporation):
            return assignment.simple_logo if simple_logos and assignment.simple_logo else assignment.logo
        return self.ASSIGNMENT_TOKENS.get(assignment)

    def bankruptcy_limit_reached(self):
        if self.BANKRUPTCY_ENDS_GAME_AFTER == "one":
            return any(player.bankrupt for player in self.players)
        elif self.BANKRUPTCY_ENDS_GAME_AFTER == "all_but_one":
            return sum(1 for player in self.players if not player.bankrupt) == 1

    def update_tile_lists(self, tile, old_tile):
        if tile.unlimited:
            self.add_extra_tile(tile)

        if tile.hex and tile.hex == self.hex_by_id(tile.hex.id):
            raise GameError(f"Cannot lay tile {tile.id}; it is already on hex {tile.hex.id}")

        self.tiles.remove(tile)
        if not old_tile.preprinted:
            self.tiles.append(old_tile)

    def local_length(self):
        return 2

    def skip_route_track_type(self, train):
        pass

    def tile_valid_for_phase(self, tile, hex=None, phase_color_cache=None):
        if not phase_color_cache:
            phase_color_cache = self.phase.tiles
        return tile.color in phase_color_cache

    def token_owner(self, entity):
        return entity.owner if entity and entity.is_company() else entity

    def company_header(self, company):
        return "PRIVATE COMPANY"

    def market_share_limit(self, corporation=None):
        return self.MARKET_SHARE_LIMIT

    def cert_limit(self, player=None):
        return self._cert_limit

    def corporation_show_interest(self):
        return True

    def after_buying_train(self, train, source):
        pass

    def sold_shares_destination(self, entity):
        return self.SOLD_SHARES_DESTINATION

    def corporations_can_ipo(self):
        return False

    @property
    def possible_presidents(self):
        return [player for player in self.players if not player.bankrupt]

    def receivership_corporations(self):
        return [corporation for corporation in self.corporations if corporation.receivership]

    def bankruptcy_options(self, entity):
        return []

    @property
    def initial_auction_companies(self):
        return self.companies

    def player_debt(self, player):
        return 0

    def render_hex_reservation(self, corporation):
        return True

    def init_graph(self):
        return Graph(self)

    def init_bank(self):
        cash = self.BANK_CASH
        if isinstance(cash, dict):
            cash = cash[len([player for player in self.players if not player.bankrupt])]
        return Bank(cash, log=self.log, check="bank" in self.game_end_check_values)

    def init_cert_limit(self):
        cert_limit = self.game_cert_limit
        if isinstance(cert_limit, dict):
            player_count = len([player for player in self.players if not player.bankrupt])
            _, default = list(cert_limit.items())[0]
            cert_limit = cert_limit.get(player_count, default)
        if isinstance(cert_limit, dict):
            cert_limit = min((k, v) for k, v in cert_limit.items() if k >= len(self.corporations))[1] or next(
                iter(cert_limit.values())
            )
        return cert_limit or self._cert_limit

    @property
    def game_cert_limit(self):
        return self.CERT_LIMIT

    def init_phase(self):
        return Phase(self.game_phases, self)

    @property
    def game_phases(self):
        return self.PHASES

    def init_round(self):
        return self.new_auction_round()

    def init_stock_market(self):
        return StockMarket(
            self.game_market,
            self.CERT_LIMIT_TYPES,
            multiple_buy_types=self.MULTIPLE_BUY_TYPES,
        )

    @property
    def game_market(self):
        return self.MARKET

    def init_companies(self, players):
        return [Company(**company) for company in self.game_companies if len(players) >= company.get("min_players", 0)]

    @property
    def game_companies(self):
        return self.entities.COMPANIES

    def init_train_handler(self):
        trains = []
        for train in self.game_trains:
            num = train.get("num", self.num_trains(train))
            for index in range(num):
                trains.append(self.TRAIN_CLASS(**train, index=index))
        return self.DEPOT_CLASS(trains, self)

    @property
    def game_trains(self):
        return self.TRAINS

    def num_trains(self, train):
        raise NotImplementedError

    def init_minors(self):
        return [Minor(**minor) for minor in self.game_minors]

    @property
    def game_minors(self):
        return self.MINORS

    def init_loans(self):
        return []

    def loans_taken(self):
        return self.total_loans - len(self.loans)

    def maximum_loans(self, entity):
        return 0

    def loan_value(self, entity=None):
        return 0

    def num_emergency_loans(self, entity, debt):
        return 0

    def corporation_opts(self):
        return {}

    def init_corporations(self, stock_market):
        return [
            self.CORPORATION_CLASS(
                min_price=min(stock_market.par_prices, key=lambda x: x.price).price,
                capitalization=self.CAPITALIZATION,
                **corporation,
                **self.corporation_opts(),
            )
            for corporation in self.game_corporations
        ]

    @property
    def game_corporations(self):
        return self.entities.CORPORATIONS

    def init_hexes(self, companies, corporations):
        blockers = defaultdict(list)
        for company in companies + self.minors + corporations:
            for ability in self.abilities(company, "blocks_hexes") + self.abilities(company, "blocks_hexes_consent"):
                for hex_id in ability.hexes:
                    blockers[hex_id].append([company, ability.hidden])

        partition_blockers = {}
        for company in self.partition_companies():
            for ability in self.abilities(company, "blocks_partition"):
                partition_blockers[ability.partition_type] = company

        reservations = defaultdict(list)
        for c in self.reservation_corporations():
            if not isinstance(c.coordinates, list):
                coords = [c.coordinates]
            else:
                coords = c.coordinates
            for idx, coord in enumerate(coords):
                city = c.city[idx] if isinstance(c.city, list) else c.city
                reservations[coord].append({"entity": c, "city": city})

        for c in corporations + companies:
            for ability in self.abilities(c, "reservation"):
                for hex_item in ability.hex:
                    reservations[hex_item].append(
                        {
                            "entity": c,
                            "city": int(ability.city),
                            "slot": int(ability.slot),
                            "ability": ability,
                        }
                    )

        optional_hexes = self.optional_hexes()
        hexes = []

        for color, hex_list in optional_hexes.items():
            for coords, tile_string in hex_list.items():
                for idx, coord in enumerate(coords):
                    if color == "empty":
                        hexes.append(Hex(coord, layout=self.layout, axes=self.axes, empty=True))
                        continue

                    try:
                        tile = Tile.for_tile(tile_string, preprinted=True, index=idx)
                    except GameError:
                        tile = Tile.from_code(coord, color, tile_string, preprinted=True, index=idx)

                    for blocker, hidden in blockers[coord]:
                        tile.add_blocker(blocker, hidden=hidden)

                    for partition in tile.partitions:
                        if partition.type in partition_blockers:
                            partition.add_blocker(partition_blockers[partition.type])

                    # set_trace()
                    for res in reservations[coord]:
                        if res.get("ability"):
                            res["ability"].tile = tile
                        tile.add_reservation(res["entity"], res["city"], res.get("slot"))

                    location_name = self.location_name(coord)

                    hexes.append(
                        Hex(
                            coord,
                            layout=self.layout,
                            axes=self.axes,
                            tile=tile,
                            location_name=location_name,
                            hide_location_name=self.HEXES_HIDE_LOCATION_NAMES.get(coord),
                        )
                    )

        return hexes

    def partition_companies(self):
        return self.companies

    def reservation_corporations(self):
        return self.corporations

    def init_tiles(self):
        return [item for name, val in self.game_tiles.items() for item in self.init_tile(name, val)]

    @property
    def game_tiles(self):
        return self.map.TILES

    def unique_tile_types(self):
        return list(set([tile.name for tile in self.all_tiles]))

    def get_available_tile_with_name(self, name):
        possible_tiles = sorted([tile for tile in self.tiles if tile.name == name], key=lambda x: x.index)
        if len(possible_tiles) == 0:
            return None
        return possible_tiles[0]

    def init_tile(self, name, val):
        if isinstance(val, int) or val == "unlimited":
            count = 1 if val == "unlimited" else val
            return [
                Tile.for_tile(
                    name,
                    index=i,
                    reservation_blocks=self.TILE_RESERVATION_BLOCKS_OTHERS,
                    unlimited=(val == "unlimited"),
                )
                for i in range(count)
            ]
        else:
            count = 1 if val["count"] == "unlimited" else val["count"]
            color = val["color"]
            code = val["code"]
            hidden = bool(val["hidden"])
            return [
                Tile.from_code(
                    name,
                    color,
                    code,
                    index=i,
                    reservation_blocks=self.TILE_RESERVATION_BLOCKS_OTHERS,
                    unlimited=(val["count"] == "unlimited"),
                    hidden=hidden,
                )
                for i in range(count)
            ]

    def init_starting_cash(self, players, bank):
        cash = self.STARTING_CASH
        if isinstance(cash, dict):
            cash = cash[len(players)]

        for player in players:
            bank.spend(cash, player)

    def init_company_abilities(self):
        for company in self.companies:
            abilities = self.abilities(company, "shares")
            ability = abilities[0] if abilities else None
            if not ability:
                continue

            real_shares = []
            for share in ability.shares:
                if share in ["random_president", "first_president"]:
                    idx = 0 if share == "first_president" else randint(0, len(self.corporations) - 1)
                    corporation = self.corporations[idx]
                    share = corporation.shares[0]
                    real_shares.append(share)
                    company.desc = f"Purchasing player takes a president's share (20%) of {corporation.name} \
                        and immediately sets its par value. {company.desc}"
                    self.log.append(f"{company.name} comes with the president's share of {corporation.name}")
                elif share == "random_share":
                    corporations = (
                        [self.corporation_by_id(id) for id in ability.corporations]
                        if ability.corporations
                        else self.corporations
                    )
                    corporation = choice(corporations)
                    share = next((s for s in corporation.shares if not s.president), None)
                    if share:
                        real_shares.append(share)
                        company.desc += f" The random corporation in this game is {corporation.name}."
                        self.log.append(f"{company.name} comes with a {share.percent}% share of {corporation.name}")
                else:
                    self.log.append(f"adding share {self.share_by_id(share)} for id {share} to ability {ability}")
                    real_shares.append(self.share_by_id(share))

            self.log.append(f"setting {ability}'s shares to {real_shares}")
            ability.shares = real_shares

    def init_share_pool(self):
        return SharePool(self, allow_president_sale=self.PRESIDENT_SALES_TO_MARKET)

    def connect_hexes(self):
        coordinates = {tuple([h.x, h.y]): h for h in self.hexes}

        for hex in self.hexes:
            for xy, direction in Hex.DIRECTIONS[hex.layout].items():
                x, y = xy
                neighbor = coordinates.get((hex.x + x, hex.y + y))
                if not neighbor:
                    continue

                hex.all_neighbors[direction] = neighbor

                # set_trace()
                if (neighbor.tile.color in self.IMPASSABLE_HEX_COLORS and not neighbor.targeting(hex)) or any(
                    border.edge == direction and border.type == "impassable" for border in hex.tile.borders
                ):
                    continue

                hex.neighbors[direction] = neighbor

    def total_rounds(self, name):
        if name == self.OPERATING_ROUND_NAME:
            return self.operating_rounds

    def next_round(self):
        if isinstance(self.round, StockRound):
            self.operating_rounds = self.phase.operating_rounds
            self.reorder_players()
            self.round = self.new_operating_round()
        elif isinstance(self.round, OperatingRound):
            if self.round.round_num < self.operating_rounds:
                self.or_round_finished()
                self.round = self.new_operating_round(self.round.round_num + 1)
            else:
                self.turn += 1
                self.or_round_finished()
                self.or_set_finished()
                self.round = self.new_stock_round()
        elif isinstance(self.round, self.initial_round_type):
            self.init_round_finished()
            self.reorder_players()
            self.round = self.new_stock_round()

    def clear_programmed_actions(self):
        self.programmed_actions.clear()

    def check_programmed_actions(self):
        for entity, action_list in list(self.programmed_actions.items()):
            self.programmed_actions[entity] = [action for action in action_list if not action.disable(self)]
            for action in action_list:
                if action.disable(self):
                    self.player_log(
                        entity,
                        f"Programmed action '{action}' removed due to round change",
                    )

    @property
    def game_end_check_values(self):
        return self.GAME_END_CHECK

    def custom_end_game_reached(self):
        return False

    def game_end_check(self):
        triggers = {
            "bankrupt": self.bankruptcy_limit_reached(),
            "bank": self.bank.is_broken(),
            "stock_market": self.stock_market.max_reached,
            "final_train": self.depot.empty(),
            "final_phase": self.phase.phases[-1] == self.phase.current if self.phase and self.phase.phases else False,
            "custom": self.custom_end_game_reached(),
        }

        # Filter the dictionary to keep only truthy values
        triggers = {reason: after for reason, after in triggers.items() if after}

        for after in [
            "immediate",
            "current_round",
            "current_or",
            "full_or",
            "one_more_full_or_set",
        ]:
            for reason in triggers:
                if self.game_end_check_values.get(reason, "") == after:
                    if after == "one_more_full_or_set":
                        self.final_turn = getattr(self, "final_turn", None) or self.turn + 1
                    return reason, after

        return None

    def final_or_in_set(self, round):
        return round.round_num == self.operating_rounds

    def end_now(self, after):
        if not after:
            return False

        if after == "immediate":
            return True

        if after == "current_round":
            return True

        if not isinstance(self.round, self.round_end()):
            return False

        final_or_in_set = self.final_or_in_set(self.round)

        if final_or_in_set and after == "one_more_full_or_set":
            return self.turn == self.final_turn

        return final_or_in_set

    def round_end(self):
        return OperatingRound

    def final_operating_rounds(self):
        return self.phase.operating_rounds

    def game_ending_description(self):
        reason, after, final_turn = self.game_end_check()

        if not after or self.finished:
            return None

        after_text = ""

        if not self.finished:
            if after == "immediate":
                after_text = " : Game Ends immediately"
            elif after == "current_round":
                if isinstance(self.round, OperatingRound):
                    after_text = f" : Game Ends at conclusion of this OR ({self.turn}.{self.round.round_num})"
                else:
                    after_text = f" : Game Ends at conclusion of this round ({self.turn})"
            elif after == "current_or":
                if isinstance(self.round, OperatingRound):
                    after_text = f" : Game Ends at conclusion of this OR ({self.turn}.{self.round.round_num})"
                else:
                    after_text = f" : Game Ends at conclusion of the next OR ({self.turn}.{self.round.round_num})"
            elif after == "full_or":
                if isinstance(self.round, OperatingRound):
                    after_text = f" : Game Ends at conclusion of {self.round_end().short_name} {self.turn}.{self.operating_rounds}"
                else:
                    after_text = f" : Game Ends at conclusion of {self.round_end().short_name} {self.turn}.{self.phase.operating_rounds}"
            elif after == "one_more_full_or_set":
                after_text = f" : Game Ends at conclusion of {self.round_end().short_name} {self.final_turn}.{self.final_operating_rounds()}"

        return f"{self.GAME_END_DESCRIPTION_REASON_MAP_TEXT[reason]}{after_text}"

    def additional_ending_after_text(self):
        return ""

    def action_processed(self, action):
        self.close_corporations_in_close_cell()

    def close_corporations_in_close_cell(self):
        if not self.stock_market.has_close_cell:
            return

        for corp in self.corporations:
            if not corp.is_closed() and corp.share_price and corp.share_price.type == "close":
                self.closing_queue[corp] = True

        if not self.corporations_are_closing:
            self.corporations_are_closing = True
            while self.closing_queue:
                corp = next(iter(self.closing_queue.keys()))
                self.closing_queue.pop(corp)
                self.close_corporation(corp)
            self.corporations_are_closing = False

    def show_priority_deal_player(self, order):
        return order == "after_last_to_act"

    def priority_deal_player(self):
        players = [player for player in self.players if not player.bankrupt]

        if self.round.current_entity and self.round.current_entity.is_player():
            last_to_act = self.round.last_to_act
            if last_to_act:
                priority_idx = (players.index(last_to_act) + 1) % len(players)
            else:
                priority_idx = 0
            return players[priority_idx]
        else:
            return players[0]

    def next_sr_position(self, entity):
        player_order = []
        if self.round.current_entity and self.round.current_entity.is_player():
            next_sr_order = self.next_sr_player_order
            if next_sr_order == "first_to_pass":
                player_order = self.round.pass_order if self.round.pass_order else []
            elif next_sr_order == "most_cash":
                player_order = sorted(
                    self.players,
                    key=lambda p: (p.cash, self.players.index(p)),
                    reverse=True,
                )
            elif next_sr_order == "least_cash":
                player_order = sorted(self.players, key=lambda p: (p.cash, self.players.index(p)))

        return player_order.index(entity) if entity in player_order else None

    @property
    def next_sr_player_order(self):
        return self.NEXT_SR_PLAYER_ORDER

    def reorder_players(self, order=None, log_player_order=False):
        # set_trace()
        order = order or self.next_sr_player_order

        if order == "after_last_to_act":
            self.players = self.players[self.round.entity_index :] + self.players[: self.round.entity_index]
        elif order == "first_to_pass":
            self.players = round.pass_order if round.pass_order else self.players
        elif order == "most_cash":
            current_order = self.players[:]
            current_order.reverse()
            self.players.sort(key=lambda p: (p.cash, current_order.index(p)), reverse=True)
        elif order == "least_cash":
            current_order = self.players[:]
            self.players.sort(key=lambda p: (p.cash, current_order.index(p)))

        if log_player_order:
            player_names = ", ".join(p.name for p in self.players if not p.bankrupt)
            log_message = f"Priority order: {player_names}"
        else:
            log_message = f"{self.players[0].name} has priority deal"

        self.log.append(log_message)

    def new_auction_round(self):
        return AuctionRound(
            self,
            [
                CompanyPendingPar,
                WaterfallAuction,
            ],
        )

    def new_stock_round(self):
        self.log.append(f"-- {self.round_description('Stock')} --")
        self.round_counter += 1
        return self.stock_round()

    def stock_round(self):
        return StockRound(
            self,
            [
                DiscardTrainStep,
                ExchangeStep,
                SpecialTrackStep,
                BuySellParShares,
            ],
        )

    def new_operating_round(self, round_num=1):
        self.log.append(f"-- {self.round_description(self.OPERATING_ROUND_NAME, round_num)} --")
        self.round_counter += 1
        return self.operating_round(round_num)

    def operating_round(self, round_num):
        return OperatingRound(
            self,
            [
                BankruptStep,
                ExchangeStep,
                SpecialTrackStep,
                BuyCompanyStep,
                TrackStep,
                TokenStep,
                RouteStep,
                DividendStep,
                DiscardTrainStep,
                BuyTrainStep,
                [BuyCompanyStep, {"blocks": True}],
            ],
            round_num=round_num,
        )

    def event_close_companies(self):
        self.log.append("-- Event: Private companies close --")
        for company in self.companies:
            abilities = self.abilities(company, "close", on_phase="any")
            ability = abilities[0] if abilities else None
            if (
                ability
                and ability.on_phase != "never"
                and any(phase["name"] == ability.on_phase for phase in self.phase.phases)
            ):
                continue
            company.close()

    def cache_objects(self):
        for type, name in self.CACHABLE:
            ivar = f"_{type}"
            cache = {x.id: x for x in getattr(self, type)}
            setattr(self, ivar, cache)

            def make_method(ivar):
                return lambda self, id, ivar=ivar: getattr(self, ivar).get(id)

            method = make_method(ivar)
            setattr(self.__class__, f"{name}_by_id", method)

    def update_cache(self, type):
        if type in self.CACHABLE:
            ivar = f"_{type}"
            cache = {x.id: x for x in getattr(self, type)()}
            setattr(self, ivar, cache)

    def bank_cash(self):
        return self.bank.cash

    def all_potential_upgrades(self, tile, tile_manifest=False, selected_company=None):
        colors = list(self.phase.phases[-1]["tiles"])
        return [
            t
            for t in self.all_tiles
            if self.tile_valid_for_phase(t, phase_color_cache=colors)
            and self.upgrades_to(tile, t, selected_company=selected_company)
            and not t.blocks_lay
        ]

    def interest_paid(self, entity):
        return True

    def interest_rate(self):
        pass

    def president_assisted_buy(self, corporation, train, price):
        return [0, 0]

    def round_description(self, name, round_number=None):
        round_number = round_number or self.round.round_num
        description = f"{name} Round "

        total = self.total_rounds(name)

        if not self.turn == 0:
            description += str(self.turn)
        if total and not self.turn == 0:
            description += "."
        if total:
            description += f"{round_number} (of {total})"

        return description.strip()

    def corporation_available(self, entity):
        return True

    def or_description_short(self, turn, round):
        return f"{turn}.{round}"

    def corporation_size(self, entity):
        return "small"

    def corporation_size_name(self, entity):
        pass

    def company_status_str(self, company):
        pass

    def status_str(self, corporation):
        pass

    def status_array(self, corporation):
        pass

    def par_price_str(self, share_price):
        return self.format_currency(share_price.price)

    def timeline(self):
        return []

    def count_available_tokens(self, corporation):
        return sum(1 for t in corporation.tokens if not t.used)

    def token_string(self, corporation):
        return f"{self.count_available_tokens(corporation)}/{len(corporation.tokens)}"

    def highlight_token(self, token):
        return False

    def show_value_of_companies(self, entity):
        return entity.player if entity else False

    def company_table_header(self):
        return "Company"

    def player_card_minors(self, player):
        return []

    def player_entities(self):
        return self.players

    def player_sort(self, entities):
        return sorted(
            entities,
            key=lambda entity: (
                self.operating_order.index(entity) if entity in self.operating_order else float("inf"),
                entity.name,
            ),
        )

    def bank_sort(self, corporations):
        return sorted(corporations, key=lambda corp: corp.name)

    def info_train_name(self, train):
        return ", ".join(train.names_to_prices.keys())

    def info_available_train(self, first_train, train):
        return train.sym == first_train.sym

    def info_train_price(self, train):
        return ", ".join(self.format_currency(price) for price in train.names_to_prices.values())

    def info_on_trains(self, phase):
        return phase["on"][0] if phase.get("on") else None

    def ability_right_type(self, ability, type):
        return not type or ability.type == type

    def ability_right_owner(self, entity, ability):
        correct_owner_type = True

        if ability.owner_type == "player":
            correct_owner_type = not entity.owner or entity.owner.is_player()
        elif ability.owner_type == "corporation":
            correct_owner_type = entity.owner and entity.owner.is_corporation()

        return correct_owner_type

    def ability_usable_this_or(self, ability):
        return not ability.count_per_or or ability.count_this_or < ability.count_per_or

    def ability_right_time(self, ability, time, on_phase, passive_ok, strict_time):
        if not self.round:
            return True

        if ability.on_phase and on_phase not in ["any", ability.on_phase]:
            return False

        if ability.after_phase and ability.after_phase not in [phase["name"] for phase in self.phase.previous]:
            return False

        if time == "any" or "any" in ability.when:
            return True

        if ability.passive and not passive_ok:
            return False

        if ability.passive and not ability.when:
            return True

        current_step = self.ability_blocking_step()
        current_step_name = pascal_to_snake(current_step.__class__.__name__) if current_step else None

        if ability.type == "tile_lay" and ability.must_lay_all and isinstance(current_step, SpecialTrackStep):
            return current_step.company == ability.owner

        if not isinstance(time, list):
            time = [time]
        times = [t if t != "%current_step%" else current_step_name for t in time]
        times = [t for t in times if t is not None]
        if not times:
            times_to_check = ability.when
            default = False
        else:
            times_to_check = list(set(ability.when) & set(times))
            default = True
            if times_to_check and not strict_time:
                return True

        return any(
            self.ability_check_time(ability_when, ability, current_step, current_step_name, default)
            for ability_when in times_to_check
        )

    def ability_check_time(self, ability_when, ability, current_step, current_step_name, default):
        if ability_when == current_step_name:
            if self.round.operating:
                return (
                    self.round.current_operator == ability.corporation()
                    or self.round.current_entity == ability.player()
                )
            elif self.round.stock:
                return self.round.current_entity == ability.player()

        if ability_when == "owning_corp_or_turn":
            return self.round.operating and self.round.current_operator == ability.corporation()

        if ability_when == "owning_player_or_turn":
            return self.round.operating and self.round.current_operator.player == ability.player()

        if ability_when == "owning_player_track":
            return (
                self.round.operating
                and self.round.current_operator.player == ability.player()
                and isinstance(current_step, TrackStep)
            )

        if ability_when == "owning_player_sr_turn":
            return self.round.stock and self.round.current_entity == ability.player()

        if ability_when == "or_between_turns":
            return self.round.operating and not self.round.current_operator_acted

        if ability_when == "or_start":
            return self.ability_time_is_or_start()

        if ability_when == "stock_round":
            return self.round.stock

        return default

    def ability_time_is_or_start(self):
        return self.round.operating and self.round.at_start

    def ability_blocking_step(self):
        supported_steps = (TrackerStep, TokenStep, RouteStep, BuyTrainStep)
        return next(
            (
                step
                for step in self.round.steps
                if isinstance(step, supported_steps) and not step.passed and step.active and step.blocks
            ),
            None,
        )

    def ability_usable(self, ability):
        if isinstance(ability, TokenAbility):
            if not ability.hexes:
                return True

            corporation = None
            if isinstance(ability.owner, Corporation):
                corporation = ability.owner
            elif isinstance(ability.owner.owner, Corporation):
                corporation = ability.owner.owner

            if not corporation:
                return True

            if not self.token_ability_from_owner_usable(ability, corporation):
                return False

            tokened_hexes = [token.city.hex.id for token in corporation.tokens if token.used]

            return bool(set(ability.hexes) - set(tokened_hexes))

        return True

    def token_ability_from_owner_usable(self, ability, corporation):
        return ability.from_owner or corporation.find_token_by_type(ability.type)

    def separate_treasury(self):
        return False

    def decorate_marker(self, icon):
        return None

    def adjustable_train_list(self, entity):
        return False

    def adjustable_train_sizes(self, entity):
        return False

    def reset_adjustable_trains(self, entity, routes):
        pass

    def operation_round_short_name(self):
        return self.OPERATION_ROUND_SHORT_NAME

    def operation_round_name(self):
        return self.OPERATING_ROUND_NAME

    def trains_str(self, corporation):
        if corporation.system:
            corps = corporation.shells
        else:
            corps = [corporation]

        result = []
        for corp in corps:
            if not corp.trains:
                result.append("None")
            else:
                train_names = [f"({t.name})" if t.obsolete else t.name for t in corp.trains]
                result.append(" ".join(train_names))

        return result

    def on_train_header(self):
        return "On Train"

    def train_limit_header(self):
        return "Train Limit"

    def train_power(self):
        return False

    def show_map_legend(self):
        return False

    def train_purchase_name(self, train):
        return train.name

    def train_actions_always_use_operating_round_view(self):
        return False

    def nav_bar_color(self):
        return self.phase.current["tiles"][-1]

    def round_phase_string(self):
        return f"Phase {self.phase.name}"

    def phase_valid(self):
        return True

    def market_par_bars(self, price):
        return []

    def show_player_percent(self, player):
        return True

    def companies_sort(self, companies):
        return companies

    def stock_round_name(self):
        return "Stock Round"

    def force_unconditional_stock_pass(self):
        return False

    def second_icon(self, corporation):
        return None
