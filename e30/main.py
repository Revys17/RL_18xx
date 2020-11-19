#!/usr/bin/env python3
import argparse
import logging
import math
import random
import sys
from collections import dict_keys
from typing import List, OrderedDict, Set, Dict

import e30.agent
import e30.game_state
from e30.actions.bid_buy_action import BidBuyAction, BidBuyActionType
from e30.actions.stock_market_buy_action import StockMarketBuyAction, StockMarketBuyActionType
from e30.actions.stock_market_sell_action import StockMarketSellAction, StockMarketSellActionType
from e30.company import Company
from e30.enums.round import Round
from e30.exceptions.exceptions import InvalidOperationException
from e30.player import Player
from e30.private_company import Private
from e30.privates.BO import BO
from e30.privates.CA import CA
from e30.privates.CS import CS
from e30.privates.DH import DH
from e30.privates.MH import MH
from e30.privates.SV import SV
from e30.stock_market_slot import StockMarketSlot

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)
TOTAL_STARTING_MONEY = 2400

"""
Something Spicy
"""

__author__ = "Caleb Larson"
__version__ = "0.1.0"
__license__ = "MIT"


def print_help() -> None:
    print('main.py -n <number of players> -a <human or AI agent>')
    sys.exit(1)


def init_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        usage="%(prog)s [-h] [--num_players] n [--agent_type] a",
        description="Something spicy"
    )
    parser.add_argument("-n", "--num_players", nargs=1, default=4, help="number of players", type=int)
    parser.add_argument("-a", "--agent_type", nargs=1, default="human", help="agent type of 'human' or 'ai'", type=str)
    return parser


def main():
    """ Main entry point of the app """
    parser = init_argparse()
    args = parser.parse_args()

    agents: List[e30.agent.Agent]
    agent_type: str = args.agent_type
    num_players: int = args.num_players

    if num_players < 4 or num_players > 7:
        raise RuntimeError("Invalid number of players: " + str(num_players))

    if agent_type != 'human' and agent_type != 'ai':
        raise RuntimeError("Invalid agent type: " + agent_type)

    # TODO: agents should be able to be a mix of human and AI agents
    if agent_type == 'human':
        agents = [e30.agent.HumanAgent() for _ in range(num_players)]
    else:
        agents = [e30.agent.AIAgent() for _ in range(num_players)]

    log.debug("Running the game with agents:")
    [log.debug(type(agent).__name__) for agent in agents]
    game: e30.game_state.GameState = initialize(agents)
    run_game(game)
    exit()


def initialize(agents: List['e30.agent.Agent']) -> 'e30.game_state.GameState':
    # init
    return e30.game_state.GameState(agents)


def run_game(game_state: 'e30.game_state.GameState') -> None:
    # do game stuff
    log.info("Starting private company auction")
    do_private_auction(game_state)

    num_round: int = 0
    while game_state.game_in_progress:
        log.info("Starting stock round")
        game_state.progression = (Round.STOCK_MARKET, num_round)
        do_stock_round(game_state)
        log.info("Starting operating round")
        game_state.progression = (Round.OPERATING_1, num_round)
        do_operating_rounds(game_state)
        num_round = num_round + 1
    return


def do_private_auction(game_state: 'e30.game_state.GameState') -> None:
    # private companies are ordered by lowest face value
    unowned_privates: List[Private] = game_state.privates
    lowest_face_value_private: Private
    consecutive_passes: int = 0

    # private company auction ends when all the private companies are bought
    while len(unowned_privates) > 0:
        log.info("Num unowned privates: {}".format(len(unowned_privates)))
        current_player: Player = game_state.current_player
        log.info("Current player: {}".format(current_player.get_name()))
        lowest_face_value_private: Private = unowned_privates[0]
        non_full_cycle_pass: bool = False

        game_state.progression = (Round.BID_BUY, 0)
        retry: bool = True

        while retry:
            retry = False
            try:
                bid_buy_action: BidBuyAction = current_player.get_bid_buy_action(game_state)
                bid_buy_action.print()

                if bid_buy_action.type is BidBuyActionType.PASS:
                    # check if everyone passed
                    consecutive_passes += 1
                    log.info("Consecutive passes: {}".format(consecutive_passes))
                    if consecutive_passes == game_state.num_players:
                        # everyone passed
                        # if private is the SV
                        if lowest_face_value_private.short_name == 'SV':
                            # if price is already 0, force player to purchase
                            if lowest_face_value_private.price == 0:
                                log.info("SV price reached 0. Purchase is forced for next player")
                                complete_purchase(game_state, game_state.get_next_player(current_player),
                                                  lowest_face_value_private, unowned_privates)
                                # set current player as the forced purchaser - this is the execution of the purchase
                                current_player = game_state.get_next_player(current_player)
                            # otherwise, lower price and continue
                            else:
                                lowest_face_value_private.lower_price(5)
                                log.info("SV price reduced to: {}".format(lowest_face_value_private.price))
                        # if private is not SV, pay private revenue and resume with priority deal
                        else:
                            log.info("All players passed, private company revenue pays out")
                            game_state.pay_private_revenue()
                    else:
                        # player passed, but it's not a full pass cycle yet
                        non_full_cycle_pass = True
                        game_state.set_next_as_current_player(current_player)
                        continue
                elif bid_buy_action.type is BidBuyActionType.BUY:
                    # only the lowest face value private company can be bought outright
                    complete_purchase(game_state, current_player, lowest_face_value_private, unowned_privates)
                elif bid_buy_action.type is BidBuyActionType.BID:
                    try:
                        unowned_privates[bid_buy_action.private_company_index]
                    except IndexError:
                        raise InvalidOperationException("Index " + str(bid_buy_action.private_company_index) +
                                                        " does not refer to a valid unowned private company")
                    unowned_privates[bid_buy_action.private_company_index].add_bid(current_player,
                                                                                   bid_buy_action.bid)
            except InvalidOperationException as e:
                log.error(e)
                retry = True

        if lowest_face_value_private.should_resolve_bid():
            game_state.progression = (Round.BID_BUY_RESOLUTION, 0)
            log.info("Resolving bid for: {}".format(lowest_face_value_private.short_name))
            lowest_face_value_private.resolve_bid(game_state)
            unowned_privates.remove(lowest_face_value_private)

        if not non_full_cycle_pass:
            game_state.set_next_as_current_player(current_player)
            # reset passes after every action except passes that don't complete full pass cycles
            consecutive_passes = 0


def complete_purchase(game_state: 'e30.game_state.GameState', player: Player, private: Private,
                      unowned_privates: List[Private]):
    log.info("Player {} buying {}".format(player.get_name(), private.short_name))
    private.buy_private(player)
    # private is now owned, remove from the unowned list
    unowned_privates.remove(private)
    game_state.set_next_as_priority_deal(player)


def get_clockwise_distance_from_player(src_index, other_index, num_players):
    diff = other_index - src_index
    if diff < 0:
        return diff + num_players


def process_stock_round_sell_action(game_state: 'e30.game_state.GameState', player: Player) -> bool:
    retry: bool = True

    while retry:
        retry = False
        try:
            game_state.print_stock_market_turn_game_state()
            sell_action: StockMarketSellAction = player.get_stock_market_sell_action(game_state)

            if sell_action.action_type is StockMarketSellActionType.PASS:
                return False

            # does the player actually own shares of the companies in the request?
            requested_sales: OrderedDict[str, int] = sell_action.sell_map
            requested_sale_names: Set[str] = set(requested_sales.keys())
            owned_share_names: Set[str] = set(player.share_map.keys())
            if not requested_sale_names.issubset(owned_share_names):
                raise InvalidOperationException("Can't sell shares from unowned companies")

            # does the player actually have enough shares to sell?
            # does it uphold the required maximum number of shares in the bank pool?
            for company_name, num_sell in requested_sales:
                if num_sell > player.share_map[company_name]:
                    raise InvalidOperationException("Can't sell more shares of " + company_name + " than owned")
                if num_sell + game_state.companies_map[company_name].market_shares > 5:
                    raise InvalidOperationException(company_name + " can't have more than 5 bank pool shares")

            # would the sale cause a change in presidency?
            new_presidents: Dict[str, Player] = {} # get new presidents if there would be any changes
            for company_name, num_sell in requested_sales:
                if company_name in player.presiding_companies:
                    # total shares: 2x president cert + other certs - shares to sell
                    num_shares_after_sale = 2 + player.share_map[company_name] - num_sell
                    other_players = list(filter(lambda pl: pl is not player,
                                                game_state.companies_map[company_name].owning_players))

                    contenders: Dict[Player, int] = {}
                    for p in other_players:
                        # must have at least 2 shares to become the new president
                        if company_name in p.share_map and p.share_map[company_name] > num_shares_after_sale \
                                and p.share_map[company_name] >= 2:
                            contenders[p] = p.share_map[company_name]

                    if num_shares_after_sale < 2 and len(contenders.keys()) == 0:
                        raise InvalidOperationException("No player able to receive presidency of " + company_name)

                    highest_shares = 0
                    closest_clockwise_player_distance = 999 # set to a number higher than the max index of players
                    for p, num_shares in contenders:
                        distance_from_selling_player = get_clockwise_distance_from_player(player.index, p.index,
                                                                                          game_state.num_players)
                        if num_shares > highest_shares:
                            highest_shares = num_shares
                            closest_clockwise_player_distance = distance_from_selling_player
                            new_presidents[company_name] = p
                        elif num_shares == highest_shares and distance_from_selling_player <\
                                closest_clockwise_player_distance:
                            closest_clockwise_player_distance = distance_from_selling_player
                            new_presidents[company_name] = p

            # would the player be over any certificate limits after the sale?
            # create a view of shares owned after sales
            # create a view of the share price of companies after sales
            # count total certs, exclude shares in orange, brown, yellow slots, add 1 for presidency transfers
            new_shares_map = player.share_map.copy()
            for company_name, num_shares in requested_sales:
                new_shares_map[company_name] = new_shares_map[company_name] - num_shares

            new_slots_map: Dict[str, StockMarketSlot] = {}
            for company_name in owned_share_names:
                new_slots_map[company_name] = game_state.companies_map[company_name].current_share_price.copy()
            # apply slot movement from selling shares
            for company_name, num_sell in requested_sales:
                for _ in range(num_sell):
                    new_slots_map[company_name].set_down(new_slots_map[company_name].down)

            cert_limit_excluded_companies = []
            for company_name, slot in new_slots_map:
                if slot.get_color() is not "white":
                    cert_limit_excluded_companies.append(company_name)

            total_num_certs_after_sale = 0
            for company_name, num_shares in new_shares_map:
                if company_name not in cert_limit_excluded_companies:
                    total_num_certs_after_sale += num_shares

            # transferring the president's cert for a company means gaining a regular share in cert conversion
            for company_name in new_presidents:
                if company_name not in cert_limit_excluded_companies:
                    total_num_certs_after_sale += 1

            if total_num_certs_after_sale > game_state.player_total_cert_limit:
                raise InvalidOperationException(
                    "Must sell more shares. Shares after sale {} would still be over the {} limit".format(
                        total_num_certs_after_sale, game_state.player_total_cert_limit))

            # apply presidency changes
            for company_name, new_president in new_presidents:
                game_state.companies_map[company_name].transfer_presidency(new_president)

            # ordered selling of companies
            for company_name, num_shares in requested_sales:
                game_state.companies_map[company_name].sell_share(player, num_shares, game_state.bank)
                # re-sort on stock value changes, order of changes matter
                game_state.re_sort_companies()
                player.buy_restricted_companies.add(company_name)

            return True
        except InvalidOperationException as e:
            log.error(e)
            retry = True


def process_stock_round_buy_action(game_state: 'e30.game_state.GameState', player: Player) -> bool:
    retry: bool = True

    while retry:
        retry = False
        try:
            game_state.print_stock_market_turn_game_state()
            sell_action: StockMarketBuyAction = player.get_stock_market_buy_action(game_state)

            if sell_action.action_type is StockMarketBuyActionType.PASS:
                return False

            return True
        except InvalidOperationException as e:
            log.error(e)
            retry = True


def do_stock_round(game_state: 'e30.game_state.GameState') -> None:
    consecutive_passes: int = 0
    [player.reset_restricted_companies_buy_list() for player in game_state.players]

    # stock round ends when a full consecutive pass cycle occurs
    while consecutive_passes < game_state.num_players:
        turn_has_action: bool = False
        game_state.print_game_progression()
        current_player: Player = game_state.get_priority_deal_player()
        game_state.print_priority_deal_player()

        # sell
        if game_state.progression is not (Round.STOCK_MARKET, 0):
            is_sell: bool = process_stock_round_sell_action(game_state, current_player)

            if is_sell:
                turn_has_action = True

        # buy
        is_buy: bool = process_stock_round_buy_action(game_state, current_player)

        if is_buy:
            turn_has_action = True

        # sell
        if game_state.progression is not (Round.STOCK_MARKET, 0):
            is_sell: bool = process_stock_round_sell_action(game_state, current_player)

            if is_sell:
                turn_has_action = True

        if turn_has_action:
            consecutive_passes = 0
            game_state.set_next_as_priority_deal(current_player)
        else:
            consecutive_passes += 1
            log.info("No sell or buy, consecutive passes: {}".format(consecutive_passes))

    # end of stock round actions


def do_operating_rounds(game_state: 'e30.game_state.GameState') -> None:
    pass


def determine_first_player_index(num_players: int) -> int:
    return random.randrange(0, num_players)


def get_players(num_players: int, agents: List['e30.agent.Agent']) -> List[Player]:
    starting_money: int = get_starting_money(num_players)
    return [Player(i, starting_money, agents[i]) for i in range(num_players)]


def get_companies() -> List[Company]:
    return [
        Company("Pennsylvania", "PRR", 4, "Altoona", "H-12"),
        Company("New York Central", "NYC", 4, "Albany", "E-19"),
        Company("Canadian Pacific", "CPR", 4, "Montreal", "A-19"),
        Company("Baltimore & Ohio", "B&O", 3, "Baltimore", "I-15"),
        Company("Chesapeake & Ohio", "C&O", 3, "Cleveland(Richmond)", "F-6 (K-13)"),
        Company("Erie", "Erie", 3, "Buffalo", "E-11"),
        Company("New York, New Haven, & Hartford", "NNH", 2, "New York", "G-19"),
        Company("Boston & Maine", "B&M", 2, "Boston", "E-23")
    ]


def get_privates() -> List[Private]:
    return [
        SV(), CS(), DH(), MH(), CA(), BO()
    ]


def get_starting_money(num_players: int) -> int:
    # 1830 rules, up to 6 players
    # TODO: hardcode starting money depending on the number of players?
    return math.floor(TOTAL_STARTING_MONEY / num_players)


if __name__ == "__main__":
    """ This is executed when run from the command line """
    main()
