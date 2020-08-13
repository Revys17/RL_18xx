#!/usr/bin/env python3
import argparse
import logging
import math
import random
import sys
from typing import List

import agent
import game_state
from actions.bid_buy_action import BidBuyAction, BidBuyActionType
from company import Company
from enums.phase import Phase
from enums.round import Round
from exceptions.exceptions import InvalidOperationException
from player import Player
from private_company import Private
from privates.BO import BO
from privates.CA import CA
from privates.CS import CS
from privates.DH import DH
from privates.MH import MH
from privates.SV import SV

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

    agents: List[agent.Agent]
    agent_type: str = args.agent_type
    num_players: int = args.num_players

    if num_players < 4 or num_players > 7:
        raise RuntimeError("Invalid number of players: " + str(num_players))

    if agent_type != 'human' and agent_type != 'ai':
        raise RuntimeError("Invalid agent type: " + agent_type)

    # TODO: agents should be able to be a mix of human and AI agents
    if agent_type == 'human':
        agents = [agent.HumanAgent() for _ in range(num_players)]
    else:
        agents = [agent.AIAgent() for _ in range(num_players)]

    log.debug("Running the game with agents:")
    [log.debug(type(agent).__name__) for agent in agents]
    game: game_state.GameState = initialize(agents)
    run_game(game)
    exit()


def initialize(agents: List['agent.Agent']) -> 'game_state.GameState':
    # init
    return game_state.GameState(agents)


def run_game(game_state: 'game_state.GameState') -> None:
    # do game stuff
    log.info("Starting private company auction")
    do_private_auction(game_state)

    while game_state.game_in_progress:
        log.info("Starting stock round")
        do_stock_round(game_state)
        log.info("Starting operating round")
        do_operating_rounds(game_state)
    return


def do_private_auction(game_state: 'game_state.GameState') -> None:
    # private companies are ordered by lowest face value
    unowned_privates: List[Private] = game_state.privates
    lowest_face_value_private: Private
    consecutive_passes: int = 0

    # private company auction ends when all the private companies are bought
    while len(unowned_privates) > 0:
        log.info("Num unowned privates: {}".format(len(unowned_privates)))
        current_player: Player = game_state.current_player
        lowest_face_value_private: Private = unowned_privates[0]
        non_full_cycle_pass: bool = False

        if lowest_face_value_private.should_resolve_bid():
            game_state.progression = (Phase.PRIVATE_AUCTION, Round.BID_BUY_RESOLUTION)
            log.info("Resolving bid for: {}".format(lowest_face_value_private.short_name))
            lowest_face_value_private.resolve_bid(game_state)
            unowned_privates.remove(lowest_face_value_private)
        else:
            game_state.progression = (Phase.PRIVATE_AUCTION, Round.BID_BUY)
            retry: bool = True

            while retry:
                retry = False
                bid_buy_action: BidBuyAction = current_player.get_bid_buy_action(game_state)

                try:
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
                                    log.info("SV price reached 0. Purchase is forced")
                                    complete_purchase(game_state, current_player, lowest_face_value_private,
                                                      unowned_privates)
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

        if not non_full_cycle_pass:
            game_state.set_next_as_current_player(current_player)
            # reset passes after every action except passes that don't complete full pass cycles
            consecutive_passes = 0


def complete_purchase(game_state: 'game_state.GameState', player: Player, private: Private,
                      unowned_privates: List[Private]):
    private.buy_private(player)
    # private is now owned, remove from the unowned list
    unowned_privates.remove(private)
    game_state.set_next_as_priority_deal(player)


def do_stock_round(game_state: 'game_state.GameState') -> None:
    pass


def do_operating_rounds(game_state: 'game_state.GameState') -> None:
    pass


def determine_first_player_index(num_players: int) -> int:
    return random.randrange(0, num_players)


def get_players(num_players: int, agents: List['agent.Agent']) -> List[Player]:
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
