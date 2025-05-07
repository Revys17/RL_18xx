from rl18xx.game.engine.game import BaseGame
from rl18xx.game import ActionHelper
from rl18xx.game.engine.actions import BaseAction
import torch
from typing import List, Tuple, Any
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
from rl18xx.game.engine.round import Exchange as ExchangeStep

import logging

LOGGER = logging.getLogger(__name__)


class ActionMapper:
    def __init__(self, initial_game_state: BaseGame):
        self.init_actions(initial_game_state)
        self.action_encoding_size = len(self.actions)
        self.mask_size = torch.tensor(len(self.actions), dtype=torch.float32)

    def init_actions(self, initial_game_state: BaseGame):
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
        self.city_offsets = {}
        self.action_offsets["PlaceToken"] = len(self.actions)
        city_count = 0
        for hex in initial_game_state.hexes:
            for i, city in enumerate(hex.tile.cities):
                self.city_offsets[(hex.id, i)] = city_count
                city_count += 1
                self.actions.append((PlaceToken, [hex.id, i]))

        # LayTile
        self.hex_offsets = {}
        self.tile_offsets = {}
        self.action_offsets["LayTile"] = len(self.actions)
        for i, hex in enumerate(initial_game_state.hexes):
            self.hex_offsets[hex.id] = i
            for j, tile_name in enumerate(initial_game_state.unique_tile_types()):
                self.tile_offsets[tile_name] = j
                for rotation in range(6):
                    self.actions.append((LayTile, [tile_name, hex.id, rotation]))

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

        # RunRoutes is handled automatically
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

        LOGGER.debug(f"Action section sizes:")
        for item1, item2 in zip(self.action_offsets.items(), list(self.action_offsets.items())[1:]):
            if item2:
                key1, value1 = item1
                key2, value2 = item2
                LOGGER.debug(f"{key1}: {value2 - value1}")
            else:
                LOGGER.debug(f"{key1}: {len(self.actions) - value1}")

        LOGGER.debug(f"Action encoding size: {len(self.actions)}")

    def _get_index_for_action(self, action: BaseAction) -> int:
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
            if action.bundle.owner == action.bundle.corporation:
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
                + action.bundle.num_shares() - 1
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
                return action_offset

            corporation_id = action.train.owner.id
            price = action.price
            if price == action.entity.cash - 1:
                price = "all-but-one"
            elif price == action.entity.cash:
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

        # RunRoutes is handled automatically
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
            elif action.price == action.company.max_price:
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
            if action.bundle.owner == action.bundle.corporation:
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

    def get_legal_action_mask(self, state: BaseGame) -> torch.Tensor:
        if state is None:
            raise ValueError("State is None")
        helper = ActionHelper(state)
        legal_actions = helper.get_all_choices_limited()

        LOGGER.debug(f"Legal actions: {legal_actions}")

        indices = []
        for action in legal_actions:
            try:
                indices.append(self._get_index_for_action(action))
            except ValueError as e:
                LOGGER.warning(f"Warning: Unmappable action from ActionHelper: {action} ({e})")
                raise e

        LOGGER.debug(f"Indices: {indices}")

        mask = torch.zeros(self.action_encoding_size, dtype=torch.float32)
        if not indices:
            raise ValueError("No legal actions found")
        indices_tensor = torch.tensor(indices, dtype=torch.long)
        mask.scatter_(0, indices_tensor, 1.0)
        return mask

    def map_index_to_action(self, index: int, state: BaseGame) -> BaseAction:
        if not (0 <= index < self.action_encoding_size):
            raise IndexError(f"Action index {index} out of bounds (0-{self.action_encoding_size-1})")

        if state is None:
            raise ValueError("State is None")

        action_helper = ActionHelper(state)
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
                location = args[2]
                exchange_step = [step for step in state.round.steps if isinstance(step, ExchangeStep)][0]
                shares = exchange_step.exchangeable_shares(entity)
                
                if location == "ipo":
                    owner = corp
                elif location == "market":
                    owner = state.share_pool
                else:
                    raise ValueError(f"Unknown location for buy shares action: {location}")
                share = [share for share in shares if share.owner == owner][0]
            else:
                corp = state.corporation_by_id(args[0])
                location = args[1]
                if location == "ipo":
                    share = corp.ipo_shares[0]
                elif location == "market":
                    share = corp.market_shares[0]
                else:
                    raise ValueError(f"Unknown location for buy shares action: {location}")
            return BuyShares(entity, share, share.price)

        if action_type is SellShares:
            corp_id, num_shares = args
            corp = state.corporation_by_id(corp_id)
            if corp is None:
                raise ValueError(f"Corporation '{corp_id}' not found in state for SellShares action")
            bundle = [
                bundle
                for bundle in state.bundles_for_corporation(entity, corp)
                if bundle.num_shares() == num_shares
            ][0]
            return SellShares(entity, bundle)

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

            if len(args) == 2:
                train_type = args[0]
                # Buy discarded train from the market
                trains = [train for train in state.depot.discarded(entity) if train.name == train_type]
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
            return action_helper.auto_route_action()[0]

        # TODO:
        # 3. Add tests for all of this

        raise TypeError(f"Cannot create action for type: {action_type}")
