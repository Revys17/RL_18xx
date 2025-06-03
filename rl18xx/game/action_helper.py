__all__ = ["ActionHelper"]

from rl18xx.shared.singleton import Singleton
import importlib
import os
import json
from .engine.entities import Company
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
    RunRoutes,
    SellShares,
    DiscardTrain,
)

from .engine.abilities import Shares as SharesAbility
from .engine.autorouter import AutoRouter
from rl18xx.game.engine.round import (
    BuyTrain as BuyTrainStep,
    Exchange as ExchangeStep,
    SpecialTrack as SpecialTrackStep,
    SpecialToken as SpecialTokenStep,
    WaterfallAuction as WaterfallAuctionStep,
)

from collections import defaultdict
import logging

LOGGER = logging.getLogger(__name__)

class ActionHelper(metaclass=Singleton):
    def __init__(self, print_enabled=False):
        self.print_enabled = print_enabled

    def get_state(self, game):
        state = {}
        state["players"] = {}
        state["corporations"] = {}
        for player in game.players:
            shares = {
                corporation.name: sum(share.percent for share in shares)
                for corporation, shares in player.shares_by_corporation.items()
            }
            state["players"][player.name] = {
                "cash": player.cash,
                "shares": shares,
                "companies": [company.sym for company in player.companies],
            }

        for corp in game.corporations:
            state["corporations"][corp.name] = {
                "cash": corp.cash,
                "companies": [company.sym for company in corp.companies],
                "trains": [train.name for train in corp.trains],
                "share_price": corp.share_price.price if corp.share_price else None,
            }
        return state

    def print_summary(self, game, json_format=False):
        if not self.print_enabled:
            return

        if json_format:
            state = game.get_state()
            print_str = (
                '\n{\n    "players": {\n'
                + ",\n".join(
                    [
                        '        "' + player + '": ' + json.dumps(info)
                        for player, info in sorted(state["players"].items())
                    ]
                )
                + "\n    },\n"
            )
            print_str += (
                '    "corporations": {\n'
                + ",\n".join(
                    ['        "' + corp + '": ' + json.dumps(info) for corp, info in state["corporations"].items()]
                )
                + "\n    },\n"
            )
            print_str += "}\n"
            print(print_str)
            return

        for player in game.players:
            shares2 = {
                corporation.name: sum(share.percent for share in shares)
                for corporation, shares in player.shares_by_corporation.items()
            }
            print(player.name)
            print(f"    cash: {player.cash}")
            print(f"    shares: {shares2}")
            print(f"    companies: {[company.sym for company in player.companies]}")

        print("")
        for corp in game.corporations:
            print(corp.name)
            print(f"    cash: {corp.cash}")
            print(f"    companies: {[company.sym for company in corp.companies]}")
            print(f"    trains: {[train.name for train in corp.trains]}")
            print(f"    ipo price: {corp.par_price().price if corp.par_price() else None}")
            print(f"    share price: {corp.share_price.price if corp.share_price else None}")
            print(f"    president: {corp.owner}")
            print(f"    num IPO shares available: {corp.num_ipo_shares()}")
            print(f"    num market shares available: {corp.num_market_shares()}")

    def sort_actions(self, actions, instances=False):
        if instances:
            return sorted(
                actions,
                key=lambda x: x.__class__.__name__
                if x.__class__.__name__ != "Pass"
                else "zzzzzzzzzzzzzzzzzzzzzzzzzzzzzz",
            )

        return sorted(
            actions,
            key=lambda x: x.__name__ if x.__name__ != "Pass" else "zzzzzzzzzzzzzzzzzzzzzzzzzzzzzz",
        )

    def get_current_actions(self, game):
        if isinstance(game.current_entity, Company):
            if Pass in game.round.actions_for(game.current_entity):
                return [Pass]
            return []
        return self.sort_actions(game.round.actions_for(game.current_entity))

    def get_company_actions(self, game):
        company_actions = {}
        if isinstance(game.current_entity, Company):
            company_actions[game.current_entity] = self.sort_actions(game.round.actions_for(game.current_entity))
        elif hasattr(game.current_entity, "companies"):
            for company in game.current_entity.companies:
                company_actions[company] = self.sort_actions(game.round.actions_for(company))

        return company_actions

    def get_corp_for_company(self, company):
        for ability in company.abilities:
            if isinstance(ability, SharesAbility):
                return ability.shares[0].corporation()
        return None

    def get_pass_actions(self, game, reduced_actions=False):
        # If reduced_actions AND
        # Is a waterfall auction AND
        # There is no active auction AND
        # The last 3 players passed AND
        # It's not the player's only choice
        # THEN don't allow a pass
        if not reduced_actions:
            return [Pass(game.current_entity)]

        if not isinstance(game.active_step(), WaterfallAuctionStep):
            return [Pass(game.current_entity)]

        if game.active_step().auctioning_company():
            return [Pass(game.current_entity)]

        # Force the player to not pass if the last 3 players passed
        if len(game.raw_actions) < 3:
            return [Pass(game.current_entity)]

        if len(self.get_bid_actions(game, min_bid_only=True)) == 0:
            return [Pass(game.current_entity)]

        last_3_actions = game.raw_actions[-3:]
        if len(set(action['entity'] for action in last_3_actions)) == 3 and all(action["type"] == "pass" for action in last_3_actions):
            return []

        return [Pass(game.current_entity)]

    def get_bid_actions(self, game, min_bid_only=False):
        if game.active_step().auctioning_company():
            company = game.active_step().auctioning_company()
            if min_bid_only:
                bid_values = [game.active_step().min_bid(company)]
            else:
                bid_values = list(
                    range(
                        game.active_step().min_bid(company),
                        game.active_step().max_bid(game.current_entity, company) + 1,
                        5,
                    )
                )
            return [
                Bid(game.current_entity, bid_value, company)
                for bid_value in bid_values
                if bid_value <= game.active_step().max_bid(game.current_entity, company)
            ]

        bids = []
        companies = game.active_step().companies
        if game.active_step().max_bid(game.current_entity, companies[0]) >= game.active_step().min_bid(
            companies[0]
        ):
            bids.append(
                [
                    Bid(
                        game.current_entity,
                        game.active_step().min_bid(companies[0]),
                        companies[0],
                    )
                ]
            )
        for company in companies[1:]:
            if min_bid_only:
                bid_values = [game.active_step().min_bid(company)]
            else:
                min_bid = game.active_step().min_bid(company)
                max_bid = game.active_step().max_bid(game.current_entity, company)
                bid_values = list(range(min_bid - (min_bid % 5), (max_bid + 5) - ((max_bid + 5) % 5), 5))

            bids.append(
                [
                    Bid(game.current_entity, bid_value, company)
                    for bid_value in bid_values
                    if bid_value <= game.active_step().max_bid(game.current_entity, company)
                ]
            )
        return sorted([item for sublist in bids for item in sublist])

    def get_par_actions(self, game):
        if hasattr(game.active_step(), "companies_pending_par"):
            par_values = game.share_prices
            return [
                Par(game.current_entity, self.get_corp_for_company(company), price)
                for company in game.active_step().companies_pending_par
                for price in par_values
            ]

        parable_corporations = sorted(
            [corp for corp in game.corporations if game.can_par(corp, game.current_entity)],
            key=lambda corporation: corporation.name,
        )
        par_values = game.share_prices
        buying_power = game.buying_power(game.current_entity)
        return sorted(
            [
                Par(game.current_entity, corp, price)
                for corp in parable_corporations
                for price in par_values
                if 2 * price.price <= buying_power
            ]
        )

    def get_buy_shares_actions(self, game):
        buyable_shares = game.active_step().buyable_shares(game.current_entity)
        # buyable_shares is a list of lists. Each list represents a group of shares for a company:owner pair
        # we want to get the share with the lowest index for each group and combine the lists
        unique_buyable_shares = sorted(
            [min(share_list, key=lambda share: share.index) for share_list in buyable_shares],
            key=lambda share: (share.corporation(), share.owner.__class__.__name__),
        )
        return sorted([BuyShares(game.current_entity, share, share.price) for share in unique_buyable_shares])

    def get_company_buy_shares_actions(self, game, company):
        exchange_step = [step for step in game.round.steps if isinstance(step, ExchangeStep)][0]
        shares = exchange_step.exchangeable_shares(company)
        return sorted([BuyShares(company, share, share.price) for share in shares])

    def get_sell_shares_actions(self, game):
        sellable_shares = game.active_step().sellable_shares(game.current_entity)
        if isinstance(game.active_step(), BuyTrainStep):
            sellable_shares = [share for share in sellable_shares if game.active_step().sellable_bundle(share)]
            return [SellShares(bundle.owner, bundle) for bundle in sellable_shares]
        else:
            return sorted([SellShares(game.current_entity, bundle) for bundle in sellable_shares])

    def get_place_token_actions(self, game):
        if hasattr(game.active_step(), "pending_token") and game.active_step().pending_token:
            hexes = game.active_step().pending_token.get("hexes")
            return sorted(
                [
                    PlaceToken(
                        game.current_entity,
                        city,
                        city.get_slot(game.current_entity),
                    )
                    for hex in hexes
                    for city in hex._tile.cities
                ]
            )

        return sorted(
            [
                PlaceToken(game.current_entity, city, city.get_slot(game.current_entity))
                for city in game.graph.connected_nodes(game.current_entity)
                if city.tokenable(game.current_entity)
            ]
        )

    def get_company_place_token_actions(self, game, company):
        special_token_step = [step for step in game.round.steps if isinstance(step, SpecialTokenStep)][0]
        ability = special_token_step.ability(company)

        actions = []
        for hex in ability.hexes:
            actions.extend(
                [
                    PlaceToken(company, city, city.get_slot(game.current_entity))
                    for city in game.hex_by_id(hex).tile.cities
                ]
            )
        return actions

    def get_lay_tile_actions(self, game):
        moves = defaultdict(dict)
        for hex in game.graph.connected_hexes(game.current_entity):
            layable_tiles = []
            layable_tiles.extend(
                [
                    tile
                    for tile in game.active_step().upgradeable_tiles(game.current_entity, hex)
                    if game.upgrade_cost(hex.tile, hex, game.current_entity, game.current_entity)
                    <= game.buying_power(game.current_entity)
                    and not game.active_step().ability_blocking_hex(game.current_entity, hex)
                ]
            )

            for tile in layable_tiles:
                rotations = game.active_step().legal_tile_rotations(game.current_entity, hex, tile)
                moves[hex][tile] = rotations
        return sorted(
            [
                LayTile(game.current_entity, tile, hex, rotation)
                for hex in moves
                for tile in moves[hex]
                for rotation in moves[hex][tile]
            ],
            key=lambda x: (x.hex.id, x.tile.name, x.rotation),
        )

    def get_company_lay_tile_actions(self, game, company):
        special_track_step = [step for step in game.round.steps if isinstance(step, SpecialTrackStep)][0]
        abilities = special_track_step.abilities(company)

        actions = []
        for ability in abilities:
            hexes = [game.hex_by_id(hex) for hex in ability.hexes]
            for hex in hexes:
                tiles = special_track_step.potential_tiles(company, hex)
                if not tiles:
                    LOGGER.info(f"No tiles found for {company.name}'s special track lay on {hex.id}")
                for tile in tiles:
                    rotations = special_track_step.legal_tile_rotations(company, hex, tile)
                    actions.extend(
                        [
                            LayTile(company, tile, hex, rotation)
                            for rotation in rotations
                            if game.upgrade_cost(hex.tile, hex, company.owner, company.owner)
                            <= game.buying_power(company.owner)
                        ]
                    )

        return actions

    def get_unique_trains(self, trains):
        unique_trains = {}
        for train in trains:
            key = (train.name, train.owner)
            if key not in unique_trains:
                unique_trains[key] = train
        return unique_trains.values()

    def get_valid_cross_company_train_prices(self, entity, limited_price_options=False):
        if limited_price_options:
            return sorted(list(set([1, 20, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, entity.cash - 1, entity.cash])))
        else:
            return range(1, entity.cash + 1)

    def get_buy_train_actions(self, game, limited_price_options=False):
        if not game.active_step().room(game.current_entity):
            return []

        actions = []

        # Add e-buy action
        # Only allow e-buy of the cheapest depot train if forced to purchase
        if (
            game.current_entity.cash < game.depot.min_depot_price and
            game.active_step().must_buy_train(game.current_entity)
        ):
            train = game.depot.min_depot_train
            price = train.min_price()
            if price <= game.buying_power(game.current_entity) + game.buying_power(game.current_entity.owner):
                actions.append(BuyTrain(game.current_entity, train, price))
        else:
            depot_trains = self.get_unique_trains(game.active_step().buyable_trains(game.current_entity))
            for train in depot_trains:
                price = train.min_price()
                if price <= game.buying_power(game.current_entity):
                    actions.append(BuyTrain(game.current_entity, train, price))

        corp_trains = self.get_unique_trains(game.depot.other_trains(game.current_entity))
        for train in corp_trains:
            valid_prices = self.get_valid_cross_company_train_prices(game.current_entity, limited_price_options)
            for price in valid_prices:
                if price > 0 and price <= game.buying_power(game.current_entity):
                    actions.append(BuyTrain(game.current_entity, train, price))

        if not limited_price_options:
            actions.append(self.get_exchange_train_actions(game))

        if actions:
            return actions

        if game.can_go_bankrupt(game.current_entity.owner, game.current_entity):
            return [Bankrupt(game.current_entity)]

        return []

    def get_discard_train_actions(self, game):
        unique_trains = self.get_unique_trains(game.current_entity.trains)
        return [DiscardTrain(game.current_entity, train) for train in unique_trains]

    def get_exchange_train_actions(self, game):
        discounted_trains = game.discountable_trains_for(game.current_entity)
        if not discounted_trains:
            return []

        unique_trains = self.get_unique_trains([discount[1] for discount in discounted_trains])
        unique_discounts = [discount for discount in discounted_trains if discount[1] in unique_trains]
        return [
            BuyTrain(game.current_entity, discount[1], discount[3], exchange=discount[0])
            for discount in unique_discounts
        ]

    def auto_route_action(self, game):
        router = AutoRouter(game)
        best_routes = router.compute(game.current_entity)
        return [
            RunRoutes(
                game.current_entity,
                [route for route in best_routes if route is not None],
            )
        ]

    def get_dividend_actions(self, game):
        options = game.active_step().dividend_options(game.current_entity)
        actions = []
        for option in options.keys():
            actions.append(Dividend(game.current_entity, option))
        return sorted(actions)

    def get_buy_company_actions(self, game, limited_price_options=False):
        # Only buy from president
        companies = game.current_entity.owner.companies
        actions = []
        for company in companies:
            if not game.abilities(company, "no_buy"):
                if limited_price_options:
                    actions.extend(
                        [
                            BuyCompany(game.current_entity, company, price)
                            for price in [company.min_price, company.max_price]
                        ]
                    )
                else:
                    actions.extend(
                        [
                            BuyCompany(game.current_entity, company, price)
                            for price in range(company.min_price, company.max_price + 1)
                        ]
                    )
        # Remove prices above buying power
        actions = [action for action in actions if action.price <= game.buying_power(game.current_entity)]
        return actions

    def get_choices_for_action(self, game, action, reduced_actions=False):
        if action == Pass:
            return self.get_pass_actions(game, reduced_actions=reduced_actions)
        elif action == Bid:
            return self.get_bid_actions(game, min_bid_only=reduced_actions)
        elif action == Par:
            return self.get_par_actions(game)
        elif action == BuyShares:
            return self.get_buy_shares_actions(game)
        elif action == SellShares:
            return self.get_sell_shares_actions(game)
        elif action == PlaceToken:
            return self.get_place_token_actions(game)
        elif action == LayTile:
            return self.get_lay_tile_actions(game)
        elif action == BuyTrain:
            return self.get_buy_train_actions(game, limited_price_options=reduced_actions)
        elif action == DiscardTrain:
            return self.get_discard_train_actions(game)
        elif action == RunRoutes:
            return self.auto_route_action(game)
        elif action == Dividend:
            return self.get_dividend_actions(game)
        elif action == BuyCompany:
            return self.get_buy_company_actions(game, limited_price_options=reduced_actions)
        else:
            return []

    def get_company_choices(self, game):
        company_actions = self.get_company_actions(game)
        if not company_actions.values():
            return []

        choices = []
        for company, actions in company_actions.items():
            for action in actions:
                if action == BuyShares:
                    choices.extend(self.get_company_buy_shares_actions(game, company))
                elif action == LayTile:
                    choices.extend(self.get_company_lay_tile_actions(game, company))
                elif action == PlaceToken:
                    choices.extend(self.get_company_place_token_actions(game, company))
        return choices

    def get_all_choices(self, game):
        if game.finished:
            return []
        choices = [
            choices
            for action in self.get_current_actions(game)
            for choices in self.get_choices_for_action(game, action)
        ]
        choices.extend(self.get_company_choices(game))
        return self.sort_actions(choices, instances=True)

    def get_all_choices_with_index(self, game):
        if game.finished:
            return {}
        return {index: value for index, value in enumerate(self.get_all_choices(game))}

    def get_all_choices_limited(self, game):
        if game.finished:
            return []
        choices = [
            choices
            for action in self.get_current_actions(game)
            for choices in self.get_choices_for_action(game, action, reduced_actions=True)
        ]
        choices.extend(self.get_company_choices(game))

        if not choices:
            if game.can_go_bankrupt(game.current_entity.owner, game.current_entity):
                return [Bankrupt(game.current_entity)]
            return []
        return self.sort_actions(choices, instances=True)
