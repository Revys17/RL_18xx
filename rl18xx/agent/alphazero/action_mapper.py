from rl18xx.game.engine.game import BaseGame
from rl18xx.game import ActionHelper
from rl18xx.game.engine.actions import BaseAction
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from rl18xx.game.engine.actions import (
    Bankrupt,
    Bid,
    BuyCompany,
    BuyShares,
    BuyTrain,
    Dividend,
    LayTile,
    Par,
    Pass,
    PlaceToken,
    SellShares,
    DiscardTrain,
    RunRoutes,
)
from rl18xx.game.engine.round import Exchange as ExchangeStep, BuyTrain as BuyTrainStep
from rl18xx.game.factored_action_helper import FactoredActionHelper, LegalAction
from rl18xx.shared.singleton import Singleton

import logging

LOGGER = logging.getLogger(__name__)


class ActionMapper(metaclass=Singleton):
    def __init__(self):
        self.init_actions()
        self.action_encoding_size = len(self.actions)
        self.mask_size = np.array(len(self.actions), dtype=np.float32)

    def init_actions(self):
        self.company_offsets = {
            "SV": 0,
            "CS": 1,
            "DH": 2,
            "MH": 3,
            "CA": 4,
            "BO": 5,
        }
        self.corporation_offsets = {
            "PRR": 0,
            "NYC": 1,
            "CPR": 2,
            "B&O": 3,
            "C&O": 4,
            "ERIE": 5,
            "NYNH": 6,
            "B&M": 7,
        }
        self.par_price_offsets = {
            67: 0,
            71: 1,
            76: 2,
            82: 3,
            90: 4,
            100: 5,
        }
        self.share_location_offsets = {
            "ipo": 0,
            "market": 1,
        }
        self.train_price_offsets = {
            "1": 0,
            "20": 1,
            "50": 2,
            "100": 3,
            "200": 4,
            "300": 5,
            "400": 6,
            "500": 7,
            "600": 8,
            "700": 9,
            "800": 10,
            "900": 11,
            "all-but-one": 12,
            "all": 13,
        }
        self.dividend_offsets = {
            "payout": 0,
            "withhold": 1,
        }
        self.buy_company_price_offsets = {
            "min": 0,
            "max": 1,
        }
        self.train_type_offsets = {
            "2": 0,
            "3": 1,
            "4": 2,
            "5": 3,
            "6": 4,
            "D": 5,
        }

        self.hex_offsets = {
            "F2": 0,
            "I1": 1,
            "J2": 2,
            "A9": 3,
            "A11": 4,
            "K13": 5,
            "B24": 6,
            "D2": 7,
            "F6": 8,
            "E9": 9,
            "H12": 10,
            "D14": 11,
            "C15": 12,
            "K15": 13,
            "A17": 14,
            "A19": 15,
            "I19": 16,
            "F24": 17,
            "D24": 18,
            "F4": 19,
            "J14": 20,
            "F22": 21,
            "E7": 22,
            "F8": 23,
            "C11": 24,
            "C13": 25,
            "D12": 26,
            "B16": 27,
            "C17": 28,
            "B20": 29,
            "D4": 30,
            "F10": 31,
            "I13": 32,
            "D18": 33,
            "B12": 34,
            "B14": 35,
            "B22": 36,
            "C7": 37,
            "C9": 38,
            "C23": 39,
            "D8": 40,
            "D16": 41,
            "D20": 42,
            "E3": 43,
            "E13": 44,
            "E15": 45,
            "F12": 46,
            "F14": 47,
            "F18": 48,
            "G3": 49,
            "G5": 50,
            "G9": 51,
            "G11": 52,
            "H2": 53,
            "H6": 54,
            "H8": 55,
            "H14": 56,
            "I3": 57,
            "I5": 58,
            "I7": 59,
            "I9": 60,
            "J4": 61,
            "J6": 62,
            "J8": 63,
            "G15": 64,
            "C21": 65,
            "D22": 66,
            "E17": 67,
            "E21": 68,
            "G13": 69,
            "I11": 70,
            "J10": 71,
            "J12": 72,
            "E19": 73,
            "H4": 74,
            "B10": 75,
            "H10": 76,
            "H16": 77,
            "F16": 78,
            "G7": 79,
            "G17": 80,
            "F20": 81,
            "D6": 82,
            "I17": 83,
            "B18": 84,
            "C19": 85,
            "E5": 86,
            "D10": 87,
            "E11": 88,
            "H18": 89,
            "I15": 90,
            "G19": 91,
            "E23": 92,
        }
        self.tile_offsets = {
            "42": 0,
            "4": 1,
            "16": 2,
            "70": 3,
            "23": 4,
            "7": 5,
            "18": 6,
            "24": 7,
            "3": 8,
            "55": 9,
            "61": 10,
            "54": 11,
            "9": 12,
            "41": 13,
            "26": 14,
            "68": 15,
            "57": 16,
            "45": 17,
            "1": 18,
            "56": 19,
            "44": 20,
            "62": 21,
            "63": 22,
            "64": 23,
            "40": 24,
            "66": 25,
            "20": 26,
            "27": 27,
            "39": 28,
            "19": 29,
            "59": 30,
            "25": 31,
            "46": 32,
            "28": 33,
            "65": 34,
            "43": 35,
            "2": 36,
            "53": 37,
            "58": 38,
            "14": 39,
            "47": 40,
            "8": 41,
            "29": 42,
            "69": 43,
            "15": 44,
            "67": 45,
        }
        self.city_count = {
            "D2": 1,
            "F6": 1,
            "H12": 1,
            "D14": 1,
            "K15": 1,
            "A19": 1,
            "F4": 1,
            "J14": 1,
            "F22": 1,
            "B16": 1,
            "E19": 1,
            "H4": 1,
            "B10": 1,
            "H10": 1,
            "H16": 1,
            "F16": 1,
            "E5": 2,
            "D10": 2,
            "E11": 2,
            "H18": 2,
            "I15": 1,
            "G19": 2,
            "E23": 1,
        }
        self.city_offsets = {
            ("D2", 0): 0,
            ("F6", 0): 1,
            ("H12", 0): 2,
            ("D14", 0): 3,
            ("K15", 0): 4,
            ("A19", 0): 5,
            ("F4", 0): 6,
            ("J14", 0): 7,
            ("F22", 0): 8,
            ("B16", 0): 9,
            ("E19", 0): 10,
            ("H4", 0): 11,
            ("B10", 0): 12,
            ("H10", 0): 13,
            ("H16", 0): 14,
            ("F16", 0): 15,
            ("E5", 0): 16,
            ("E5", 1): 17,
            ("D10", 0): 18,
            ("D10", 1): 19,
            ("E11", 0): 20,
            ("E11", 1): 21,
            ("H18", 0): 22,
            ("H18", 1): 23,
            ("I15", 0): 24,
            ("G19", 0): 25,
            ("G19", 1): 26,
            ("E23", 0): 27,
        }

        # Action Encoding
        self.actions: List[Tuple[BaseAction, List[Any]]] = []
        self.action_offsets = {}
        self.action_offsets["Pass"] = len(self.actions)
        self.action_offsets["CompanyPass"] = len(self.actions)
        self.actions.append((Pass, []))

        # Bid
        self.action_offsets["Bid"] = len(self.actions)
        for company in self.company_offsets.keys():
            self.actions.append((Bid, [company]))

        # Par
        self.action_offsets["Par"] = len(self.actions)
        for corp in self.corporation_offsets.keys():
            for price in self.par_price_offsets.keys():
                self.actions.append((Par, [corp, price]))

        # BuyShares
        self.action_offsets["BuyShares"] = len(self.actions)
        for corp in self.corporation_offsets.keys():
            for share_type in self.share_location_offsets.keys():
                self.actions.append((BuyShares, [corp, share_type]))

        # SellShares
        self.action_offsets["SellShares"] = len(self.actions)
        for corp in self.corporation_offsets.keys():
            for num_shares in range(1, 6):
                self.actions.append((SellShares, [corp, num_shares]))

        # PlaceToken
        self.action_offsets["PlaceToken"] = len(self.actions)
        for hex_id in self.hex_offsets.keys():
            if hex_id in self.city_count:
                for num_cities in range(self.city_count[hex_id]):
                    self.actions.append((PlaceToken, [hex_id, num_cities]))

        # LayTile
        self.action_offsets["LayTile"] = len(self.actions)
        for hex_id in self.hex_offsets.keys():
            for tile_name in self.tile_offsets.keys():
                for rotation in range(6):
                    self.actions.append((LayTile, [tile_name, hex_id, rotation]))

        # BuyTrain
        self.action_offsets["BuyTrain"] = len(self.actions)
        self.actions.append((BuyTrain, ["depot"]))
        for train_type in self.train_type_offsets.keys():
            self.actions.append((BuyTrain, [train_type, "market"]))
        for i, corp in enumerate(self.corporation_offsets.keys()):
            for train_type in self.train_type_offsets.keys():
                for price in self.train_price_offsets.keys():
                    self.actions.append((BuyTrain, [corp, train_type, price]))

        # DiscardTrain
        self.action_offsets["DiscardTrain"] = len(self.actions)
        for train in self.train_type_offsets.keys():
            self.actions.append((DiscardTrain, [train]))

        # Dividend
        self.action_offsets["Dividend"] = len(self.actions)
        for type in self.dividend_offsets.keys():
            self.actions.append((Dividend, [type]))

        # BuyCompany
        self.action_offsets["BuyCompany"] = len(self.actions)
        for i, company in enumerate(self.company_offsets.keys()):
            for price in self.buy_company_price_offsets.keys():
                self.actions.append((BuyCompany, [company, price]))

        # Bankrupt
        self.action_offsets["Bankrupt"] = len(self.actions)
        self.actions.append((Bankrupt, []))

        # RunRoutes
        self.action_offsets["RunRoutes"] = len(self.actions)
        self.actions.append((RunRoutes, []))

        # Company actions:
        # Company BuyShares - Only MH can be exchanged for NY
        self.action_offsets["CompanyBuyShares"] = len(self.actions)
        for location in self.share_location_offsets.keys():
            self.actions.append((BuyShares, ["MH", "NYC", location]))

        # Company LayTile
        self.action_offsets["CompanyLayTile"] = len(self.actions)
        # CS can lay a tile on Hex B20, DH can lay a tile on Hex F16
        self.company_tile_offsets = {
            "3": 0,
            "4": 1,
            "58": 2,
        }

        for rotation in range(6):
            self.actions.append((LayTile, ["DH", "57", "F16", rotation]))
        for tile in self.company_tile_offsets.keys():
            for rotation in range(6):
                self.actions.append((LayTile, ["CS", tile, "B20", rotation]))

        # Company PlaceToken
        self.action_offsets["CompanyPlaceToken"] = len(self.actions)
        # Only DH places a token (on F16)
        self.actions.append((PlaceToken, ["DH", "F16", 0]))

        # D-train depot slots — appended at the END so existing 26535 indices stay
        # stable for already-trained checkpoints. Disambiguates phase-6 depot
        # state where 6-trains and D-trains coexist (the legacy ["depot"] slot
        # only ever resolves to the cheapest depot train, i.e. the 6).
        self.action_offsets["BuyTrainDFull"] = len(self.actions)
        self.actions.append((BuyTrain, ["depot", "D", "full"]))
        self.action_offsets["BuyTrainDTradeIn"] = len(self.actions)
        self.actions.append((BuyTrain, ["depot", "D", "trade-in"]))

        # LOGGER.debug(f"Action section sizes:")
        # for item1, item2 in zip(self.action_offsets.items(), list(self.action_offsets.items())[1:]):
        #     if item2:
        #         key1, value1 = item1
        #         key2, value2 = item2
        #         LOGGER.debug(f"{key1}: {value2 - value1}")
        #     else:
        #         LOGGER.debug(f"{key1}: {len(self.actions) - value1}")

        # LOGGER.debug(f"Action encoding size: {len(self.actions)}")

    def get_index_for_action(self, action: BaseAction, state: BaseGame) -> int:
        action_type = action.__class__.__name__
        if action.entity.__class__.__name__ == "Company":
            action_type = f"Company{action_type}"

        action_offset = self.action_offsets[action_type]
        if action_type in ["Pass", "CompanyPass"]:
            return action_offset

        if action_type == "Bid":
            if not action.company:
                raise ValueError(f"Company is None for bid action: {action}")
            return action_offset + self.company_offsets[action.company.id]

        if action_type == "Par":
            if not action.corporation:
                raise ValueError(f"Corporation is None for par action: {action}")
            if not action.share_price:
                raise ValueError(f"Share price is None for par action: {action}")
            return (
                action_offset
                + self.corporation_offsets[action.corporation.id] * len(self.par_price_offsets)
                + self.par_price_offsets[action.share_price.price]
            )

        if action_type == "BuyShares":
            if not action.bundle:
                raise ValueError(f"Bundle is None for buy shares action: {action}")
            if not action.bundle.corporation:
                raise ValueError(f"Corporation is None for buy shares action: {action}")
            if action.bundle.owner.name == "The Bank":
                location = "ipo"
            elif action.bundle.owner.name == "Market":
                location = "market"
            else:
                LOGGER.error(f"Unknown share owner: {action.bundle.owner}")
                raise ValueError(f"Unknown owner for buy shares action: {action}")
            return (
                action_offset
                + self.corporation_offsets[action.bundle.corporation.id] * len(self.share_location_offsets)
                + self.share_location_offsets[location]
            )

        if action_type == "SellShares":
            if not action.bundle:
                raise ValueError(f"Bundle is None for sell shares action: {action}")
            if not action.bundle.corporation:
                raise ValueError(f"Corporation is None for sell shares action: {action}")
            return (
                action_offset
                + self.corporation_offsets[action.bundle.corporation.id] * 5
                + action.bundle.num_shares()
                - 1
            )

        if action_type == "PlaceToken":
            if not action.city:
                raise ValueError(f"City is None for place token action: {action}")

            hex = action.city.tile.hex
            city_idx = hex.tile.cities.index(action.city)
            return action_offset + self.city_offsets[(hex.id, city_idx)]

        if action_type == "LayTile":
            if not action.hex:
                raise ValueError(f"Hex is None for lay tile action: {action}")
            if not action.tile:
                raise ValueError(f"Tile is None for lay tile action: {action}")
            if action.rotation is None:
                raise ValueError(f"Rotation is None for lay tile action: {action}")
            return (
                action_offset
                + self.hex_offsets[action.hex.id] * len(self.tile_offsets) * 6
                + self.tile_offsets[action.tile.name] * 6
                + action.rotation
            )

        if action_type == "BuyTrain":
            if not action.train:
                raise ValueError(f"Train is None for buy train action: {action}")
            if action.train.owner.name == "The Depot":
                if action.train in action.train.owner.discarded:
                    return action_offset + 1 + self.train_type_offsets[action.train.name]
                # Disambiguate D-trains from the legacy cheapest-depot slot.
                if action.exchange is not None:
                    return self.action_offsets["BuyTrainDTradeIn"]
                if action.train.name == "D":
                    return self.action_offsets["BuyTrainDFull"]
                return action_offset

            corporation_id = action.train.owner.id
            price = action.price

            if action.entity.cash == 0 and state.round.active_step().president_may_contribute(action.entity):
                max = action.entity.cash + action.entity.owner.cash
            else:
                max = action.entity.cash

            if price == max - 1:
                price = "all-but-one"
            elif price == max:
                price = "all"
            # Only allow prices defined in train_price_offsets
            if str(price) not in self.train_price_offsets.keys():
                raise ValueError(f"Disallowed price for buy train action: {price}")
            if action.train.name not in self.train_type_offsets.keys():
                raise ValueError(f"Disallowed train type for buy train action: {action.train.name}")
            return (
                action_offset
                + 1
                + len(self.train_type_offsets)
                + self.corporation_offsets[corporation_id]
                * len(self.train_price_offsets)
                * len(self.train_type_offsets)
                + self.train_type_offsets[action.train.name] * len(self.train_price_offsets)
                + self.train_price_offsets[str(price)]
            )

        if action_type == "DiscardTrain":
            if not action.train:
                raise ValueError(f"Train is None for discard train action: {action}")
            return action_offset + self.train_type_offsets[action.train.name]

        if action_type == "Dividend":
            if not action.kind:
                raise ValueError(f"Kind is None for dividend action: {action}")
            return action_offset + self.dividend_offsets[action.kind]

        if action_type == "BuyCompany":
            if not action.company:
                raise ValueError(f"Company is None for buy company action: {action}")
            if not action.price:
                raise ValueError(f"Price is None for buy company action: {action}")
            # Only allow prices defined in buy_company_price_offsets
            if action.price == action.company.min_price:
                price = "min"
            elif action.price == action.company.max_price or action.price == action.entity.cash:
                price = "max"
            else:
                raise ValueError(f"Disallowed price for buy company action: {action.price}")
            return (
                action_offset
                + self.company_offsets[action.company.id] * len(self.buy_company_price_offsets)
                + self.buy_company_price_offsets[price]
            )

        if action_type == "CompanyBuyShares":
            if action.entity.sym != "MH":
                raise ValueError(f"Company is not MH for company buy shares action: {action}")

            # Either exchange MH for NYC from IPO or market
            if action.bundle.owner.name == "The Bank":
                location = "ipo"
            elif action.bundle.owner.name == "Market":
                location = "market"
            else:
                raise ValueError(f"Unknown owner for company buy shares action: {action.bundle.owner.name}")
            return action_offset + self.share_location_offsets[location]

        if action_type == "CompanyLayTile":
            if action.entity.sym != "CS" and action.entity.sym != "DH":
                raise ValueError(f"Company is not CS or DH for company lay tile action: {action}")

            if action.entity.sym == "DH":
                if action.rotation not in range(6):
                    raise ValueError(f"Rotation is not in range for company lay tile action: {action}")
                return action_offset + action.rotation

            tile = action.tile
            if tile.name not in self.company_tile_offsets.keys():
                raise ValueError(f"Tile {tile.name} is not in range for company lay tile action: {action}")
            return action_offset + 6 + self.company_tile_offsets[tile.name] * 6 + action.rotation

        if action_type == "CompanyPlaceToken":
            return action_offset

        if action_type == "Bankrupt":
            return action_offset

        if action_type == "RunRoutes":
            return action_offset

        raise ValueError(f"Unknown action type: {type(action)}")

    def get_lay_tile_index_info(self) -> dict:
        """Returns layout information for the LayTile action block.

        Returns a dict with:
            offset: start index of LayTile actions in the flat action vector
            num_hexes: number of hex positions (93)
            num_tiles: number of tile types (46)
            num_rotations: always 6
            num_lay_tile: total LayTile actions (num_hexes * num_tiles * num_rotations)
        """
        offset = self.action_offsets["LayTile"]
        num_hexes = len(self.hex_offsets)
        num_tiles = len(self.tile_offsets)
        num_rotations = 6
        return {
            "offset": offset,
            "num_hexes": num_hexes,
            "num_tiles": num_tiles,
            "num_rotations": num_rotations,
            "num_lay_tile": num_hexes * num_tiles * num_rotations,
        }

    def decompose_lay_tile_index(self, flat_index: int) -> Tuple[int, int, int]:
        """Given a flat action index that falls in the LayTile range, return (hex_idx, tile_idx, rotation).

        Args:
            flat_index: an index into the full action vector that corresponds to a LayTile action.

        Returns:
            Tuple of (hex_idx, tile_idx, rotation) where each is a zero-based index.

        Raises:
            ValueError: if flat_index is outside the LayTile range.
        """
        info = self.get_lay_tile_index_info()
        offset = info["offset"]
        num_tiles = info["num_tiles"]
        num_rotations = info["num_rotations"]
        num_lay_tile = info["num_lay_tile"]

        relative = flat_index - offset
        if relative < 0 or relative >= num_lay_tile:
            raise ValueError(
                f"flat_index {flat_index} is outside the LayTile range [{offset}, {offset + num_lay_tile})"
            )

        hex_idx = relative // (num_tiles * num_rotations)
        tile_idx = (relative % (num_tiles * num_rotations)) // num_rotations
        rotation = relative % num_rotations
        return (hex_idx, tile_idx, rotation)

    def convert_indices_to_mask(self, indices: List[int]) -> np.ndarray:
        mask = np.zeros(self.action_encoding_size, dtype=np.float32)
        mask[indices] = 1.0
        return mask

    def get_legal_action_mask(self, state: BaseGame) -> np.ndarray:
        indices = self.get_legal_action_indices(state)
        return self.convert_indices_to_mask(indices)

    # Slot layout for the ContinuousPriceHead (kept in sync with
    # ``model_transformer.ContinuousPriceHead``). Exposed here so the
    # pretraining pipeline can resolve (action_type, entity) → slot index
    # without importing model code.
    _PRICE_HEAD_COMPANIES = ("SV", "CS", "DH", "MH", "CA", "BO")
    _PRICE_HEAD_CORPORATIONS = ("PRR", "NYC", "CPR", "B&O", "C&O", "ERIE", "NYNH", "B&M")
    _PRICE_HEAD_TRAIN_TYPES = ("2", "3", "4", "5", "6", "D")

    def price_head_slot_for_action(
        self, action: BaseAction, state: BaseGame
    ) -> Optional[Tuple[str, int, int, int, int]]:
        """Resolve the ``ContinuousPriceHead`` slot for a price-bearing action.

        Returns ``(action_type, slot_index, observed_price, price_min,
        price_max)`` for ``Bid`` / cross-corp ``BuyTrain`` / ``BuyCompany``;
        returns ``None`` for categorical-only actions or fixed-price slots
        (depot trains, exchange trains) that don't reach the head.

        The legal ``[price_min, price_max]`` range comes from ``state``'s
        active step (mirroring :class:`FactoredActionHelper`); the
        ``observed_price`` is the human's raw chosen price (passed through
        unchanged so the NLL loss learns to predict actual human bidding /
        train-trade / private-purchase decisions).

        Used by ``pretraining.convert_game_to_training_data`` to attach a
        price NLL target alongside the categorical pi target.
        """
        name = action.__class__.__name__
        if name == "Bid":
            if action.company is None:
                return None
            try:
                slot = self._PRICE_HEAD_COMPANIES.index(action.company.id)
            except ValueError:
                return None
            step = state.active_step()
            try:
                price_min = int(step.min_bid(action.company))
                price_max = int(step.max_bid(action.entity, action.company))
            except (AttributeError, TypeError, ValueError) as e:
                LOGGER.warning(
                    "price_head_slot_for_action: Bid price-range lookup failed "
                    "for company=%s entity=%s (%s: %s); falling back to fixed price.",
                    action.company.id, getattr(action.entity, "id", action.entity),
                    type(e).__name__, e,
                )
                price_min = int(action.price)
                price_max = int(action.price)
            return ("Bid", slot, int(action.price), price_min, price_max)

        if name == "BuyCompany":
            if action.company is None or action.entity is None:
                return None
            try:
                slot = (
                    len(self._PRICE_HEAD_COMPANIES)
                    + len(self._PRICE_HEAD_CORPORATIONS) * len(self._PRICE_HEAD_TRAIN_TYPES)
                    + self._PRICE_HEAD_COMPANIES.index(action.company.id)
                )
            except ValueError:
                return None
            price_min = int(action.company.min_price)
            buying_power = int(getattr(state, "buying_power", lambda x: x.cash)(action.entity))
            price_max = int(min(action.company.max_price, buying_power))
            return ("BuyCompany", slot, int(action.price), price_min, price_max)

        if name == "BuyTrain":
            train = action.train
            if train is None:
                return None
            # ``train.owner`` is an entity object on the Python engine but a
            # bare string (the corp sym, ``"The Depot"``, or a player id) on
            # the Rust adapter. Normalize both to a string symbol so the
            # depot check and the corp-index lookup work in either world.
            owner = train.owner
            if isinstance(owner, str):
                owner_sym = owner
            else:
                owner_sym = (
                    getattr(owner, "sym", None)
                    or getattr(owner, "name", None)
                    or getattr(owner, "id", None)
                )
            if owner_sym == "The Depot":
                return None  # depot trains are fixed price; not modelled.
            try:
                corp_idx = self._PRICE_HEAD_CORPORATIONS.index(owner_sym)
                train_idx = self._PRICE_HEAD_TRAIN_TYPES.index(train.name)
            except ValueError:
                return None
            slot = (
                len(self._PRICE_HEAD_COMPANIES)
                + corp_idx * len(self._PRICE_HEAD_TRAIN_TYPES)
                + train_idx
            )
            try:
                step = state.round.active_step()
                price_min, price_max = step.spend_minmax(action.entity, train)
                price_min = int(price_min)
                price_max = int(price_max)
            except (AttributeError, TypeError, ValueError) as e:
                LOGGER.warning(
                    "price_head_slot_for_action: BuyTrain price-range lookup failed "
                    "for entity=%s train=%s (%s: %s); falling back to (1, entity.cash).",
                    getattr(action.entity, "id", action.entity), train.name,
                    type(e).__name__, e,
                )
                price_min = 1
                price_max = int(action.entity.cash)
            return ("BuyTrain", slot, int(action.price), price_min, price_max)

        return None

    def canonical_index_for_action(self, action: BaseAction, state: BaseGame) -> int:
        """Return the price-collapsed canonical flat index for a concrete action.

        Counterpart to :meth:`get_index_for_action` but with the price
        dimension folded out for price-bearing types: a ``BuyCompany``
        action at any legal price maps to the same canonical slot per
        (company); a cross-corp ``BuyTrain`` at any legal price maps to
        the same canonical slot per (corp, train_type); and so on.

        Used by pretraining to produce a one-hot categorical pi target
        on the (type, entity) slot while emitting the raw observed price
        as a separate ``price_target`` for the ``ContinuousPriceHead``.
        """
        action_type = action.__class__.__name__
        if action.entity.__class__.__name__ == "Company":
            action_type = f"Company{action_type}"

        if action_type == "BuyTrain":
            if not action.train:
                raise ValueError(f"Train is None for buy train action: {action}")
            offset = self.action_offsets["BuyTrain"]
            if action.train.owner.name == "The Depot":
                if action.train in action.train.owner.discarded:
                    return offset + 1 + self.train_type_offsets[action.train.name]
                if action.exchange is not None:
                    return self.action_offsets["BuyTrainDTradeIn"]
                if action.train.name == "D":
                    return self.action_offsets["BuyTrainDFull"]
                return offset
            corporation_id = action.train.owner.id
            # Cross-corp: collapse the price dimension to the "1" canonical
            # slot regardless of the human's chosen price.
            return (
                offset
                + 1
                + len(self.train_type_offsets)
                + self.corporation_offsets[corporation_id]
                * len(self.train_price_offsets)
                * len(self.train_type_offsets)
                + self.train_type_offsets[action.train.name] * len(self.train_price_offsets)
                + 0  # canonical price slot
            )

        if action_type == "BuyCompany":
            if not action.company:
                raise ValueError(f"Company is None for buy company action: {action}")
            return (
                self.action_offsets["BuyCompany"]
                + self.company_offsets[action.company.id] * len(self.buy_company_price_offsets)
                + 0  # canonical "min" slot
            )

        # Bid and all categorical types: identical to the legacy mapper.
        return self.get_index_for_action(action, state)

    def index_for_factored(self, la: LegalAction, state: BaseGame) -> int:
        """Map a :class:`LegalAction` directly to its canonical flat-policy index.

        For categorical-only types this is a direct lookup. For price-bearing
        types (Bid / BuyTrain / BuyCompany) the price dimension is collapsed:
        every legal (type, entity) pair maps to a single canonical index in
        the existing flat layout. MCTS's continuous-price progressive widening
        recovers the price as a per-node grandchild keyed off the
        :attr:`LegalAction.price_range` metadata; the network's
        ``ContinuousPriceHead`` emits ``(μ, log σ)`` for each (type, entity)
        slot.

        The flat layout itself is unchanged from the legacy enumerated
        ``(action, price)`` layout — we just no longer consume the price-bearing
        slots, so a ``Bid`` legal action maps to its single per-company slot,
        a ``BuyTrain`` legal action maps to a canonical per-(corp, train_type)
        slot, etc. Old expanded slots remain in the policy vector (no resize)
        but are simply not surfaced as legal by the mapper.

        See docs/step1_review.md "Continuous-price action space via progressive
        widening" and "FactoredActionHelper output schema" for the design
        context.
        """
        t = la.type

        if t == "Pass":
            # Pass collapses to a single slot regardless of acting entity; the
            # legacy mapper distinguished CompanyPass via the action_offsets
            # map but both pointed at the same slot (0), so this is consistent.
            return self.action_offsets["Pass"]

        if t == "Bankrupt":
            return self.action_offsets["Bankrupt"]

        if t == "RunRoutes":
            return self.action_offsets["RunRoutes"]

        if t == "Bid":
            company_sym = la.entity.get("private")
            if company_sym is None:
                raise ValueError(f"Bid LegalAction missing 'private' entity: {la}")
            return self.action_offsets["Bid"] + self.company_offsets[company_sym]

        if t == "Par":
            corp_sym = la.entity.get("corp")
            price = la.params.get("par_price")
            if corp_sym is None or price is None:
                raise ValueError(f"Par LegalAction missing corp/par_price: {la}")
            return (
                self.action_offsets["Par"]
                + self.corporation_offsets[corp_sym] * len(self.par_price_offsets)
                + self.par_price_offsets[price]
            )

        if t == "BuyShares":
            corp_sym = la.entity.get("corp")
            source = la.params.get("source")  # "ipo" or "market"
            # Distinguish "private acts as actor" (CompanyBuyShares — only MH→NYC) from
            # the regular player BuyShares.
            if la.entity.get("private") == "MH" and corp_sym == "NYC":
                return self.action_offsets["CompanyBuyShares"] + self.share_location_offsets[source]
            if corp_sym is None or source is None:
                raise ValueError(f"BuyShares LegalAction missing corp/source: {la}")
            return (
                self.action_offsets["BuyShares"]
                + self.corporation_offsets[corp_sym] * len(self.share_location_offsets)
                + self.share_location_offsets[source]
            )

        if t == "CompanyBuyShares":
            # Only MH → NYC exchange today.
            source = la.params.get("source")
            return self.action_offsets["CompanyBuyShares"] + self.share_location_offsets[source]

        if t == "SellShares":
            corp_sym = la.entity.get("corp")
            count = int(la.params.get("count"))
            if corp_sym is None:
                raise ValueError(f"SellShares LegalAction missing corp: {la}")
            return self.action_offsets["SellShares"] + self.corporation_offsets[corp_sym] * 5 + count - 1

        if t == "PlaceToken":
            # PlaceToken slot index is keyed off (hex_id, city_index). The
            # factored helper exposes both; the legacy mapper indexes via the
            # ``city_offsets`` table.
            if la.entity.get("private") == "DH":
                return self.action_offsets["CompanyPlaceToken"]
            hex_id = la.params.get("hex")
            city_idx = la.params.get("city") or 0
            return self.action_offsets["PlaceToken"] + self.city_offsets[(hex_id, city_idx)]

        if t == "LayTile":
            # Company-special tile lays use a different slot block; detect via
            # the entity descriptor.
            private = la.entity.get("private")
            tile_name = la.params.get("tile")
            hex_id = la.params.get("hex")
            rotation = la.params.get("rotation")
            if private == "DH" and hex_id == "F16" and tile_name == "57":
                return self.action_offsets["CompanyLayTile"] + rotation
            if private == "CS" and hex_id == "B20":
                if tile_name not in self.company_tile_offsets:
                    raise ValueError(f"CS company LayTile uses unsupported tile {tile_name}: {la}")
                return self.action_offsets["CompanyLayTile"] + 6 + self.company_tile_offsets[tile_name] * 6 + rotation
            return (
                self.action_offsets["LayTile"]
                + self.hex_offsets[hex_id] * len(self.tile_offsets) * 6
                + self.tile_offsets[tile_name] * 6
                + rotation
            )

        if t == "BuyTrain":
            return self._index_for_factored_buy_train(la, state)

        if t == "DiscardTrain":
            train_name = la.params.get("train")
            return self.action_offsets["DiscardTrain"] + self.train_type_offsets[train_name]

        if t == "Dividend":
            kind = la.params.get("kind")
            return self.action_offsets["Dividend"] + self.dividend_offsets[kind]

        if t == "BuyCompany":
            company_sym = la.entity.get("private")
            if company_sym is None:
                raise ValueError(f"BuyCompany LegalAction missing 'private': {la}")
            # Collapse the price dimension: canonical slot = the "min" entry.
            return self.action_offsets["BuyCompany"] + self.company_offsets[company_sym] * len(
                self.buy_company_price_offsets
            )

        raise ValueError(f"Unknown LegalAction type: {la.type}")

    def _index_for_factored_buy_train(self, la: LegalAction, state: BaseGame) -> int:
        """Resolve the canonical flat index for a factored BuyTrain action.

        The legacy layout has three sub-blocks:
          - depot-cheapest (1 slot)
          - market discarded per train_type (6 slots)
          - cross-corp per (corp, train_type, price) (8 * 6 * 14 = 672 slots)

        We collapse the cross-corp price dimension to its first slot. For
        depot trains we distinguish "fresh-from-depot" (use the single depot
        slot) from "discarded" (per train_type market slot) by consulting the
        game's depot state.
        """
        offset = self.action_offsets["BuyTrain"]
        source = la.entity.get("source")
        train_name = la.entity.get("train")

        if source == "depot":
            # Disambiguate fresh vs discarded by checking the depot. Use the
            # game.depot accessor; for RustGameAdapter the .depot proxy exposes
            # the same API.
            depot = getattr(state, "depot", None)
            is_discarded = False
            if depot is not None:
                discarded = getattr(depot, "discarded", []) or []
                for t in discarded:
                    if getattr(t, "name", None) == train_name:
                        is_discarded = True
                        break
            if is_discarded:
                return offset + 1 + self.train_type_offsets[train_name]
            # Disambiguate D-trains so phase-6 doesn't collapse 6 and D onto
            # the same cheapest-depot slot. ``exchange`` is present on the
            # LegalAction descriptor only for trade-in BuyTrain entries.
            if train_name == "D":
                if la.entity.get("exchange") is not None:
                    return self.action_offsets["BuyTrainDTradeIn"]
                return self.action_offsets["BuyTrainDFull"]
            return offset
        # Cross-corp: collapse the price dim to its canonical (first-price) slot.
        if source not in self.corporation_offsets:
            raise ValueError(f"BuyTrain LegalAction has unknown source {source!r}: {la}")
        first_price_offset = 0  # the "1" price slot, which is always present.
        return (
            offset
            + 1
            + len(self.train_type_offsets)
            + self.corporation_offsets[source] * len(self.train_price_offsets) * len(self.train_type_offsets)
            + self.train_type_offsets[train_name] * len(self.train_price_offsets)
            + first_price_offset
        )

    def get_legal_actions_factored(
        self, state: BaseGame
    ) -> Tuple[List[int], Dict[int, Tuple[int, int]], Dict[int, str]]:
        """Return factored legal actions for ``state``.

        Returns:
            indices: sorted unique list of canonical flat-policy indices.
            price_ranges_by_idx: ``{flat_index: (min, max)}`` for price-bearing
                slots; ``min == max`` for fixed-price entries (depot trains).
                Categorical-only slots are absent.
            action_types_by_idx: ``{flat_index: action_type_name}`` for every
                legal index. MCTS PW consults this to look up the price-grid
                step (``mcts.PRICE_GRID``) and the ``ContinuousPriceHead`` slot
                key.
        """
        if state is None:
            raise ValueError("State is None")

        choices = self._get_factored_choices(state)

        indices: List[int] = []
        price_ranges: Dict[int, Tuple[int, int]] = {}
        action_types: Dict[int, str] = {}
        seen = set()
        for la in choices:
            try:
                idx = self.index_for_factored(la, state)
            except (KeyError, ValueError) as exc:
                LOGGER.warning(f"Unmappable factored LegalAction: {la} ({exc})")
                continue
            if idx in seen:
                # Multiple LegalAction entries can collapse to the same flat
                # slot (e.g., a discarded depot train and the fresh cheapest
                # train of the same name). Merge price ranges by union.
                if la.price_range is not None:
                    existing = price_ranges.get(idx)
                    if existing is None:
                        price_ranges[idx] = la.price_range
                    else:
                        price_ranges[idx] = (min(existing[0], la.price_range[0]), max(existing[1], la.price_range[1]))
                continue
            seen.add(idx)
            indices.append(idx)
            if la.price_range is not None:
                price_ranges[idx] = la.price_range
            action_types[idx] = la.type

        indices.sort()
        return indices, price_ranges, action_types

    def _get_factored_choices(self, state: BaseGame) -> List[LegalAction]:
        """Return factored legal actions, preferring Rust when available.

        ``RustGameAdapter`` exposes ``get_factored_choices`` mirroring the
        Python :class:`FactoredActionHelper`. For Python ``BaseGame`` instances
        we fall back to the in-tree helper.
        """
        if hasattr(state, "get_factored_choices") and callable(state.get_factored_choices):
            return state.get_factored_choices()
        return FactoredActionHelper().get_choices(state)

    def get_legal_action_indices(self, state: BaseGame) -> List[int]:
        """Return the sorted list of legal flat-policy indices for ``state``.

        Backed by :meth:`get_legal_actions_factored` — the price-bearing slots
        are collapsed to a single canonical index per (type, entity). Callers
        that need the price-range metadata (e.g., MCTS PW) should use
        :meth:`get_legal_actions_factored` directly.
        """
        if state is None:
            raise ValueError("State is None")
        indices, _, _ = self.get_legal_actions_factored(state)
        LOGGER.debug(f"Indices: {indices}")
        return indices

    def map_index_to_action(self, index: int, state: BaseGame) -> BaseAction:
        if not (0 <= index < self.action_encoding_size):
            raise IndexError(f"Action index {index} out of bounds (0-{self.action_encoding_size-1})")

        if state is None:
            raise ValueError("State is None")

        action_type, args = self.actions[index]
        entity = state.current_entity
        is_company_action = index >= self.action_offsets["CompanyBuyShares"]

        if action_type is Pass:
            return Pass(entity)

        if action_type is Bid:
            if len(args) != 1:
                raise ValueError(f"Bid action expects 1 argument, got {len(args)}")
            company_id = args[0]
            company = state.company_by_id(company_id)
            if company is None:
                raise ValueError(f"Company '{company_id}' not found in state for Bid action")
            bid_price = state.active_step().min_bid(company)
            return Bid(entity, bid_price, company=company)

        if action_type is Par:
            if len(args) != 2:
                raise ValueError(f"Par action expects 2 arguments, got {len(args)}")
            corporation_id = args[0]
            price = args[1]
            corporation = state.corporation_by_id(corporation_id)
            if corporation is None:
                raise ValueError(f"Corporation '{corporation_id}' not found in state for Par action")
            for par_price in state.stock_market.par_prices:
                if par_price.price == price:
                    return Par(entity, corporation, par_price)
            raise ValueError(f"Share price {price} not found in state for Par action")

        if action_type is BuyShares:
            if is_company_action:
                entity = state.company_by_id(args[0])
                corp = state.corporation_by_id(args[1])
                bank = state.bank
                location = args[2]
                exchange_step = [step for step in state.round.steps if isinstance(step, ExchangeStep)][0]
                shares = exchange_step.exchangeable_shares(entity)

                if location == "ipo":
                    owner = bank
                elif location == "market":
                    owner = state.share_pool
                else:
                    raise ValueError(f"Unknown location for buy shares action: {location}")
                matching = [share for share in shares if share.owner == owner]
                if not matching:
                    # Factored helper enumerated this as legal but the engine's
                    # actual state has no matching share. Raise a typed error
                    # so MCTS can mask the action and retry instead of crashing.
                    raise ValueError(
                        f"No exchangeable share found for entity={entity}, "
                        f"corp={corp}, location={location}"
                    )
                share = matching[0]
            else:
                corp = state.corporation_by_id(args[0])
                location = args[1]
                if location == "ipo":
                    pool = corp.ipo_shares
                elif location == "market":
                    pool = corp.market_shares
                else:
                    raise ValueError(f"Unknown location for buy shares action: {location}")
                if not pool:
                    raise ValueError(
                        f"BuyShares enumerated for {corp}/{location} but pool is empty"
                    )
                share = pool[0]
            return BuyShares(entity, share)

        if action_type is SellShares:
            corp_id, num_shares = args
            sellable_shares = state.active_step().sellable_shares(entity)
            if isinstance(state.active_step(), BuyTrainStep):
                sellable_shares = [share for share in sellable_shares if state.active_step().sellable_bundle(share)]

            possible_bundle = [
                bundle
                for bundle in sellable_shares
                if bundle.corporation.id == corp_id and bundle.num_shares() == num_shares
            ]
            if not possible_bundle:
                raise ValueError(f"No sellable shares found for SellShares action: {args}")
            bundle = possible_bundle[0]
            return SellShares(bundle.owner, bundle)

        if action_type is PlaceToken:
            if is_company_action:
                entity = state.company_by_id(args[0])
                city = state.hex_by_id(args[1]).tile.cities[args[2]]
            else:
                city = state.hex_by_id(args[0]).tile.cities[args[1]]
            return PlaceToken(entity, city, city.get_slot(entity))

        if action_type is LayTile:
            if is_company_action:
                entity = state.company_by_id(args[0])
                tile_name, hex_id, rotation = args[1:]
            else:
                tile_name, hex_id, rotation = args
            tile = state.get_available_tile_with_name(tile_name)
            if tile is None:
                raise ValueError(f"Tile '{tile_name}' not found in state for LayTile action")
            hex = state.hex_by_id(hex_id)
            return LayTile(entity, tile, hex, rotation)

        if action_type is BuyTrain:
            if len(args) == 1:
                # Buy train from depot
                trains = state.depot.depot_trains(entity)
                if len(trains) == 0:
                    raise ValueError("No trains available in depot for BuyTrain action")

                train = None
                i = 0
                while not train:
                    train = trains[i]
                    if train not in state.depot.discarded:
                        break
                    i += 1
                if train is None:
                    raise ValueError("No non-discarded trains available in depot for BuyTrain action")
                return BuyTrain(entity, train, train.price)

            if len(args) == 3 and args[0] == "depot" and args[1] == "D":
                # D-train depot slot: full price or trade-in discount.
                d_trains = [
                    t for t in state.depot.depot_trains(entity)
                    if t.name == "D" and t not in state.depot.discarded
                ]
                if not d_trains:
                    raise ValueError("No non-discarded D-train available in depot for BuyTrain action")
                d_train = d_trains[0]
                if args[2] == "full":
                    return BuyTrain(entity, d_train, d_train.price)
                if args[2] == "trade-in":
                    # Auto-pick lowest-tier donor (4 first, then 5, then 6).
                    donor = None
                    for donor_name in ("4", "5", "6"):
                        matches = [t for t in entity.trains if t.name == donor_name]
                        if matches:
                            donor = matches[0]
                            break
                    if donor is None:
                        raise ValueError(
                            "No 4/5/6 owned by entity for D-trade-in BuyTrain action"
                        )
                    # Look up the canonical discounted price for this (donor, target).
                    discount_info = state.discountable_trains_for(entity)
                    discounted_price = None
                    for donor_t, target_t, _name, price in discount_info:
                        if donor_t is donor and target_t.name == "D":
                            discounted_price = price
                            break
                    if discounted_price is None:
                        # Fall back: any entry for D-target works since donor 4/5/6
                        # all yield the same $300 discount in 1830.
                        for _donor_t, target_t, _name, price in discount_info:
                            if target_t.name == "D":
                                discounted_price = price
                                break
                    if discounted_price is None:
                        raise ValueError(
                            "No discounted-D entry in discountable_trains_for for entity"
                        )
                    return BuyTrain(entity, d_train, int(discounted_price), exchange=donor)
                raise ValueError(f"Unknown D-train depot variant: {args[2]!r}")

            if len(args) == 2:
                train_type = args[0]
                # Buy discarded train from the market
                trains = [train for train in state.depot.discarded if train.name == train_type]
                if len(trains) == 0:
                    raise ValueError("No discarded trains available in depot for BuyTrain action")
                train = trains[0]
                return BuyTrain(entity, train, train.price)

            # Buy train from other corporation
            corp_id, train_type, price = args
            if price == "all-but-one":
                price = entity.cash - 1
            elif price == "all":
                price = entity.cash
            corp = state.corporation_by_id(corp_id)
            if corp is None:
                raise ValueError(f"Corporation '{corp_id}' not found in state for BuyTrain action")
            possible_train = [train for train in corp.trains if train.name == train_type]
            if not possible_train:
                raise ValueError(f"Train '{train_type}' not found in state for BuyTrain action")
            train = possible_train[0]
            return BuyTrain(entity, train, int(price))

        if action_type is DiscardTrain:
            train_type = args[0]
            trains = [train for train in entity.trains if train.name == train_type]
            if not trains:
                raise ValueError(f"Train '{train_type}' not found in state for DiscardTrain action")
            train = trains[0]
            return DiscardTrain(entity, train)

        # RunRoutes is handled automatically

        if action_type is Dividend:
            kind = args[0]
            return Dividend(entity, kind)

        if action_type is BuyCompany:
            company_id = args[0]
            company = state.company_by_id(company_id)
            if company is None:
                raise ValueError(f"Company '{company_id}' not found in state for BuyCompany action")
            price = args[1]
            if price == "min":
                price = company.min_price
            elif price == "max":
                price = company.max_price
            else:
                raise ValueError(f"Invalid price for BuyCompany action: {price}")
            return BuyCompany(entity, company, price)

        if action_type is Bankrupt:
            return Bankrupt(entity)

        if action_type is RunRoutes:
            action_helper = ActionHelper()
            return action_helper.auto_route_action(state)[0]

        raise TypeError(f"Cannot create action for type: {action_type}")

    def map_index_to_action_with_price(
        self, index: int, state: BaseGame, price: int
    ) -> BaseAction:
        """Materialize a price-bearing action with an explicit sampled price.

        Counterpart to :meth:`map_index_to_action` for the continuous-price
        progressive widening flow: MCTS samples a price from the
        ``ContinuousPriceHead`` truncated to a slot's legal range, snaps to
        the legal grid, and calls this method to construct the concrete game
        action. Pretraining uses the same primitive to materialize human
        actions with their raw observed prices.

        For non-price-bearing slots this delegates to
        :meth:`map_index_to_action` and ignores ``price``.
        """
        if not (0 <= index < self.action_encoding_size):
            raise IndexError(f"Action index {index} out of bounds (0-{self.action_encoding_size-1})")
        if state is None:
            raise ValueError("State is None")

        action_type, args = self.actions[index]
        entity = state.current_entity

        if action_type is Bid:
            company_id = args[0]
            company = state.company_by_id(company_id)
            if company is None:
                raise ValueError(f"Company '{company_id}' not found in state for Bid action")
            return Bid(entity, int(price), company=company)

        if action_type is BuyCompany:
            company_id = args[0]
            company = state.company_by_id(company_id)
            if company is None:
                raise ValueError(f"Company '{company_id}' not found in state for BuyCompany action")
            return BuyCompany(entity, company, int(price))

        if action_type is BuyTrain:
            if len(args) == 1:
                # Depot train — price is fixed; ignore the sampled price.
                return self.map_index_to_action(index, state)
            if len(args) == 3 and args[0] == "depot" and args[1] == "D":
                # D-train depot slots — price is fixed (full or trade-in).
                return self.map_index_to_action(index, state)
            if len(args) == 2:
                # Market discarded train — price is the train's face price.
                return self.map_index_to_action(index, state)
            corp_id, train_type, _ = args
            corp = state.corporation_by_id(corp_id)
            if corp is None:
                raise ValueError(f"Corporation '{corp_id}' not found in state for BuyTrain action")
            possible_train = [t for t in corp.trains if t.name == train_type]
            if not possible_train:
                raise ValueError(f"Train '{train_type}' not found in state for BuyTrain action")
            return BuyTrain(entity, possible_train[0], int(price))

        # Non-price-bearing — delegate.
        return self.map_index_to_action(index, state)
