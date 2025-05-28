__all__ = ["ActionHelper"]


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


class ActionHelper:
    def __init__(self, game, print_enabled=False):
        self.g = game
        self.router = AutoRouter(game)
        self.print_enabled = print_enabled

    def get_state(self):
        state = {}
        state["players"] = {}
        state["corporations"] = {}
        for player in self.g.players:
            shares = {
                corporation.name: sum(share.percent for share in shares)
                for corporation, shares in player.shares_by_corporation.items()
            }
            state["players"][player.name] = {
                "cash": player.cash,
                "shares": shares,
                "companies": [company.sym for company in player.companies],
            }

        for corp in self.g.corporations:
            state["corporations"][corp.name] = {
                "cash": corp.cash,
                "companies": [company.sym for company in corp.companies],
                "trains": [train.name for train in corp.trains],
                "share_price": corp.share_price.price if corp.share_price else None,
            }
        return state

    def print_summary(self, json_format=False):
        if self.print_enabled:
            if json_format:
                state = self.get_state()
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

            for player in self.g.players:
                shares2 = {
                    corporation.name: sum(share.percent for share in shares)
                    for corporation, shares in player.shares_by_corporation.items()
                }
                print(player.name)
                print(f"    cash: {player.cash}")
                print(f"    shares: {shares2}")
                print(f"    companies: {[company.sym for company in player.companies]}")

            print("")
            for corp in self.g.corporations:
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

    def get_current_actions(self):
        if isinstance(self.g.current_entity, Company):
            if Pass in self.g.round.actions_for(self.g.current_entity):
                return [Pass]
            return []
        return self.sort_actions(self.g.round.actions_for(self.g.current_entity))

    def get_company_actions(self):
        company_actions = {}
        if isinstance(self.g.current_entity, Company):
            company_actions[self.g.current_entity] = self.sort_actions(self.g.round.actions_for(self.g.current_entity))
        elif hasattr(self.g.current_entity, "companies"):
            for company in self.g.current_entity.companies:
                company_actions[company] = self.sort_actions(self.g.round.actions_for(company))

        return company_actions

    def get_corp_for_company(self, company):
        for ability in company.abilities:
            if isinstance(ability, SharesAbility):
                return ability.shares[0].corporation()
        return None

    def get_pass_actions(self, reduced_actions=False):
        # If reduced_actions AND
        # Is a waterfall auction AND
        # There is no active auction AND
        # The last 3 players passed
        # THEN don't allow a pass
        if not reduced_actions:
            return [Pass(self.g.current_entity)]

        if not isinstance(self.g.active_step(), WaterfallAuctionStep):
            return [Pass(self.g.current_entity)]

        if self.g.active_step().auctioning_company():
            return [Pass(self.g.current_entity)]

        # Force the player to not pass if the last 3 players passed
        last_3_actions = self.g.raw_actions[-3:]
        all_passes = all(action["type"] == "pass" for action in last_3_actions)
        if all_passes:
            return []

        return [Pass(self.g.current_entity)]

    def get_bid_actions(self, min_bid_only=False):
        if self.g.active_step().auctioning_company():
            company = self.g.active_step().auctioning_company()
            if min_bid_only:
                bid_values = [self.g.active_step().min_bid(company)]
            else:
                bid_values = list(
                    range(
                        self.g.active_step().min_bid(company),
                        self.g.active_step().max_bid(self.g.current_entity, company) + 1,
                        5,
                    )
                )
            return [
                Bid(self.g.current_entity, bid_value, company)
                for bid_value in bid_values
                if bid_value <= self.g.active_step().max_bid(self.g.current_entity, company)
            ]

        bids = []
        companies = self.g.active_step().companies
        if self.g.active_step().max_bid(self.g.current_entity, companies[0]) >= self.g.active_step().min_bid(
            companies[0]
        ):
            bids.append(
                [
                    Bid(
                        self.g.current_entity,
                        self.g.active_step().min_bid(companies[0]),
                        companies[0],
                    )
                ]
            )
        for company in companies[1:]:
            if min_bid_only:
                bid_values = [self.g.active_step().min_bid(company)]
            else:
                min_bid = self.g.active_step().min_bid(company)
                max_bid = self.g.active_step().max_bid(self.g.current_entity, company)
                bid_values = list(range(min_bid - (min_bid % 5), (max_bid + 5) - ((max_bid + 5) % 5), 5))

            bids.append(
                [
                    Bid(self.g.current_entity, bid_value, company)
                    for bid_value in bid_values
                    if bid_value <= self.g.active_step().max_bid(self.g.current_entity, company)
                ]
            )
        return sorted([item for sublist in bids for item in sublist])

    def get_par_actions(self):
        if hasattr(self.g.active_step(), "companies_pending_par"):
            par_values = self.g.share_prices
            return [
                Par(self.g.current_entity, self.get_corp_for_company(company), price)
                for company in self.g.active_step().companies_pending_par
                for price in par_values
            ]

        parable_corporations = sorted(
            [corp for corp in self.g.corporations if self.g.can_par(corp, self.g.current_entity)],
            key=lambda corporation: corporation.name,
        )
        par_values = self.g.share_prices
        buying_power = self.g.buying_power(self.g.current_entity)
        return sorted(
            [
                Par(self.g.current_entity, corp, price)
                for corp in parable_corporations
                for price in par_values
                if 2 * price.price <= buying_power
            ]
        )

    def get_buy_shares_actions(self):
        buyable_shares = self.g.active_step().buyable_shares(self.g.current_entity)
        # buyable_shares is a list of lists. Each list represents a group of shares for a company:owner pair
        # we want to get the share with the lowest index for each group and combine the lists
        unique_buyable_shares = sorted(
            [min(share_list, key=lambda share: share.index) for share_list in buyable_shares],
            key=lambda share: (share.corporation(), share.owner.__class__.__name__),
        )
        return sorted([BuyShares(self.g.current_entity, share, share.price) for share in unique_buyable_shares])

    def get_company_buy_shares_actions(self, company):
        exchange_step = [step for step in self.g.round.steps if isinstance(step, ExchangeStep)][0]
        shares = exchange_step.exchangeable_shares(company)
        return sorted([BuyShares(company, share, share.price) for share in shares])

    def get_sell_shares_actions(self):
        sellable_shares = self.g.active_step().sellable_shares(self.g.current_entity)
        if isinstance(self.g.active_step(), BuyTrainStep):
            sellable_shares = [share for share in sellable_shares if self.g.active_step().sellable_bundle(share)]
            return [SellShares(bundle.owner, bundle) for bundle in sellable_shares]
        else:
            return sorted([SellShares(self.g.current_entity, bundle) for bundle in sellable_shares])

    def get_place_token_actions(self):
        if hasattr(self.g.active_step(), "pending_token") and self.g.active_step().pending_token:
            hexes = self.g.active_step().pending_token.get("hexes")
            return sorted(
                [
                    PlaceToken(
                        self.g.current_entity,
                        city,
                        city.get_slot(self.g.current_entity),
                    )
                    for hex in hexes
                    for city in hex._tile.cities
                ]
            )

        return sorted(
            [
                PlaceToken(self.g.current_entity, city, city.get_slot(self.g.current_entity))
                for city in self.g.graph.connected_nodes(self.g.current_entity)
                if city.tokenable(self.g.current_entity)
            ]
        )

    def get_company_place_token_actions(self, company):
        special_token_step = [step for step in self.g.round.steps if isinstance(step, SpecialTokenStep)][0]
        ability = special_token_step.ability(company)

        actions = []
        for hex in ability.hexes:
            actions.extend(
                [
                    PlaceToken(company, city, city.get_slot(self.g.current_entity))
                    for city in self.g.hex_by_id(hex).tile.cities
                ]
            )
        return actions

    def get_lay_tile_actions(self):
        moves = defaultdict(dict)
        for hex in self.g.graph.connected_hexes(self.g.current_entity):
            layable_tiles = []
            layable_tiles.extend(
                [
                    tile
                    for tile in self.g.active_step().upgradeable_tiles(self.g.current_entity, hex)
                    if self.g.upgrade_cost(hex.tile, hex, self.g.current_entity, self.g.current_entity)
                    <= self.g.buying_power(self.g.current_entity)
                    and not self.g.active_step().ability_blocking_hex(self.g.current_entity, hex)
                ]
            )

            for tile in layable_tiles:
                rotations = self.g.active_step().legal_tile_rotations(self.g.current_entity, hex, tile)
                moves[hex][tile] = rotations
        return sorted(
            [
                LayTile(self.g.current_entity, tile, hex, rotation)
                for hex in moves
                for tile in moves[hex]
                for rotation in moves[hex][tile]
            ],
            key=lambda x: (x.hex.id, x.tile.name, x.rotation),
        )

    def get_company_lay_tile_actions(self, company):
        special_track_step = [step for step in self.g.round.steps if isinstance(step, SpecialTrackStep)][0]
        abilities = special_track_step.abilities(company)

        actions = []
        for ability in abilities:
            hexes = [self.g.hex_by_id(hex) for hex in ability.hexes]
            for hex in hexes:
                tiles = special_track_step.potential_tiles(company, hex)
                for tile in tiles:
                    rotations = special_track_step.legal_tile_rotations(company, hex, tile)
                    actions.extend(
                        [
                            LayTile(company, tile, hex, rotation)
                            for rotation in rotations
                            if self.g.upgrade_cost(hex.tile, hex, company.owner, company.owner)
                            <= self.g.buying_power(company.owner)
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

    def get_buy_train_actions(self, limited_price_options=False):
        if not self.g.active_step().room(self.g.current_entity):
            return []

        trains = self.get_unique_trains(self.g.depot.available(self.g.current_entity))
        # print(trains)
        actions = []
        for train in trains:
            min_spend, max_spend = self.g.active_step().spend_minmax(self.g.current_entity, train)
            # print(min_spend, max_spend)
            min_price = train.min_price()
            if min_price > self.g.buying_power(self.g.current_entity):
                if self.g.active_step().must_buy_train(self.g.current_entity):
                    if min_price > self.g.buying_power(self.g.current_entity) + self.g.buying_power(
                        self.g.current_entity.owner
                    ):
                        continue
                else:
                    continue
            max_price = train.price if train.from_depot() else self.g.current_entity.cash
            if limited_price_options:
                price_values = [
                    price
                    for price in sorted(
                        list(
                            set(
                                [
                                    1,
                                    20,
                                    50,
                                    100,
                                    200,
                                    300,
                                    400,
                                    500,
                                    600,
                                    700,
                                    800,
                                    900,
                                    self.g.current_entity.cash - 1,
                                    max_price,
                                ]
                            )
                        )
                    )
                    if price >= min_price and price <= max_price and price <= self.g.buying_power(self.g.current_entity)
                ]
            else:
                price_values = list(range(min_price, int(max_price) + 1))
            actions.append([BuyTrain(self.g.current_entity, train, price) for price in price_values])

        if not limited_price_options:
            actions.append(self.get_exchange_train_actions())

        all_actions = sorted([item for sublist in actions for item in sublist])
        if all_actions:
            return all_actions

        if self.g.can_go_bankrupt(self.g.current_entity.owner, self.g.current_entity):
            return [Bankrupt(self.g.current_entity)]

        return []

    def get_discard_train_actions(self):
        unique_trains = self.get_unique_trains(self.g.current_entity.trains)
        return [DiscardTrain(self.g.current_entity, train) for train in unique_trains]

    def get_exchange_train_actions(self):
        discounted_trains = self.g.discountable_trains_for(self.g.current_entity)
        if not discounted_trains:
            return []

        unique_trains = self.get_unique_trains([discount[1] for discount in discounted_trains])
        unique_discounts = [discount for discount in discounted_trains if discount[1] in unique_trains]
        return [
            BuyTrain(self.g.current_entity, discount[1], discount[3], exchange=discount[0])
            for discount in unique_discounts
        ]

    def auto_route_action(self):
        best_routes = self.router.compute(self.g.current_entity)
        return [
            RunRoutes(
                self.g.current_entity,
                [route for route in best_routes if route is not None],
            )
        ]

    def get_dividend_actions(self):
        options = self.g.active_step().dividend_options(self.g.current_entity)
        actions = []
        for option in options.keys():
            actions.append(Dividend(self.g.current_entity, option))
        return sorted(actions)

    def get_buy_company_actions(self, limited_price_options=False):
        # Only buy from president
        companies = self.g.current_entity.owner.companies
        actions = []
        for company in companies:
            if not self.g.abilities(company, "no_buy"):
                if limited_price_options:
                    actions.extend(
                        [
                            BuyCompany(self.g.current_entity, company, price)
                            for price in [company.min_price, company.max_price]
                        ]
                    )
                else:
                    actions.extend(
                        [
                            BuyCompany(self.g.current_entity, company, price)
                            for price in range(company.min_price, company.max_price + 1)
                        ]
                    )
        # Remove prices above buying power
        actions = [action for action in actions if action.price <= self.g.buying_power(self.g.current_entity)]
        return actions

    def get_choices_for_action(self, action, reduced_actions=False):
        if action == Pass:
            return self.get_pass_actions()
        elif action == Bid:
            return self.get_bid_actions(min_bid_only=reduced_actions)
        elif action == Par:
            return self.get_par_actions()
        elif action == BuyShares:
            return self.get_buy_shares_actions()
        elif action == SellShares:
            return self.get_sell_shares_actions()
        elif action == PlaceToken:
            return self.get_place_token_actions()
        elif action == LayTile:
            return self.get_lay_tile_actions()
        elif action == BuyTrain:
            return self.get_buy_train_actions(limited_price_options=reduced_actions)
        elif action == DiscardTrain:
            return self.get_discard_train_actions()
        elif action == RunRoutes:
            return self.auto_route_action()
        elif action == Dividend:
            return self.get_dividend_actions()
        elif action == BuyCompany:
            return self.get_buy_company_actions(limited_price_options=reduced_actions)
        else:
            return []

    def get_company_choices(self):
        company_actions = self.get_company_actions()
        if not company_actions.values():
            return []

        choices = []
        for company, actions in company_actions.items():
            for action in actions:
                if action == BuyShares:
                    choices.extend(self.get_company_buy_shares_actions(company))
                elif action == LayTile:
                    choices.extend(self.get_company_lay_tile_actions(company))
                elif action == PlaceToken:
                    choices.extend(self.get_company_place_token_actions(company))
        return choices

    def get_all_choices(self):
        if self.g.finished:
            return []
        choices = [choices for action in self.get_current_actions() for choices in self.get_choices_for_action(action)]
        choices.extend(self.get_company_choices())
        return self.sort_actions(choices, instances=True)

    def get_all_choices_with_index(self):
        if self.g.finished:
            return {}
        return {index: value for index, value in enumerate(self.get_all_choices())}

    def get_all_choices_limited(self):
        if self.g.finished:
            return []
        choices = [
            choices
            for action in self.get_current_actions()
            for choices in self.get_choices_for_action(action, reduced_actions=True)
        ]
        choices.extend(self.get_company_choices())

        if not choices:
            return [Bankrupt(self.g.current_entity)]
        return self.sort_actions(choices, instances=True)
