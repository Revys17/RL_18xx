#!/usr/bin/env python3
import getopt
import math
import random
import sys
from typing import List

from agent import Agent, HumanAgent, AIAgent
from company import Company
from game_state import GameState
from player import Player
from private_company import Private
from privates.BO import BO
from privates.CA import CA
from privates.CS import CS
from privates.DH import DH
from privates.MH import MH
from privates.SV import SV

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


def main(argv):
    """ Main entry point of the app """
    num_players: int = 4
    agent_type: str = "human"
    agents: List[Agent]

    try:
        opts, args = getopt.getopt(argv, "hn:a:")
    except getopt.GetoptError:
        print_help()

    for opt, arg in opts:
        if opt == '-h':
            print_help()
        elif opt == '-n':
            num_players = arg
            if num_players < 4 or num_players > 7:
                raise RuntimeError("Invalid number of players: " + arg)
        elif opt == '-a':
            if arg != 'human' or arg != 'ai':
                raise RuntimeError("Invalid agent type: " + arg)
            agent_type = arg

    # TODO: agents should be able to be a mix of human and AI agents
    if agent_type == 'human':
        agents = [HumanAgent() for _ in range(num_players)]
    else:
        agents = [AIAgent() for _ in range(num_players)]

    game: GameState = initialize(agents)
    run_game(game)
    exit()


def initialize(agents: List[Agent]) -> GameState:
    # init
    return GameState(agents)


def run_game(game_state) -> None:
    # do game stuff
    do_private_auction(game_state)

    while game_state.game_in_progress:
        do_stock_round(game_state)
        do_operating_rounds(game_state)
    return


def do_private_auction(game_state) -> None:
    auction_completed: bool = False
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


def do_stock_round(game_state: GameState) -> None:
    pass


def do_operating_rounds(game_state: GameState) -> None:
    pass


def determine_first_player(num_players) -> int:
    return random.randrange(0, num_players)


def get_players(num_players: int, agents: List[Agent]) -> List[Player]:
    starting_money: int = get_starting_money(num_players)
    return [Player(starting_money, agents[i]) for i in range(num_players)]


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
