from rl18xx.game.gamemap import GameMap
from rl18xx.game.action_helper import ActionHelper
from rl18xx.game.engine.game.base import BaseGame


def test_1830_manual_actions():
    game_map = GameMap()
    game = game_map.game_by_title("1830")
    g = game({"1": "Player 1", "2": "Player 2", "3": "Player 3", "4": "Player 4"})
    action_helper = ActionHelper(g)
    # action_helper.print_enabled = True

    # Waterfall auction
    g.process_action(action_helper.get_all_choices()[-1])  # pass
    g.process_action(action_helper.get_all_choices()[1])  # bid 45 on CS
    g.process_action(action_helper.get_all_choices()[1])  # bid 50 on CS
    g.process_action(action_helper.get_all_choices()[-77])  # bid 225 on BO
    g.process_action(action_helper.get_all_choices()[0])  # buy SV
    g.process_action(action_helper.get_all_choices()[-1])  # pass
    g.process_action(action_helper.get_all_choices()[0])  # buy DH
    g.process_action(action_helper.get_all_choices()[0])  # buy MH
    g.process_action(action_helper.get_all_choices()[0])  # buy CA
    g.process_action(action_helper.get_all_choices()[0])  # Par B&O at 100
    action_helper.print_summary(json_format=True)
    expected_state = {
        "players": {
            "Player 1": {"cash": 580, "shares": {}, "companies": ["SV"]},
            "Player 2": {"cash": 530, "shares": {}, "companies": ["DH"]},
            "Player 3": {"cash": 440, "shares": {}, "companies": ["CS", "MH"]},
            "Player 4": {"cash": 215, "shares": {"PRR": 10, "B&O": 20}, "companies": ["CA", "BO"]},
        },
        "corporations": {
            "PRR": {"cash": 0, "companies": [], "trains": [], "share_price": None},
            "NYC": {"cash": 0, "companies": [], "trains": [], "share_price": None},
            "CPR": {"cash": 0, "companies": [], "trains": [], "share_price": None},
            "B&O": {"cash": 0, "companies": [], "trains": [], "share_price": 100},
            "C&O": {"cash": 0, "companies": [], "trains": [], "share_price": None},
            "ERIE": {"cash": 0, "companies": [], "trains": [], "share_price": None},
            "NYNH": {"cash": 0, "companies": [], "trains": [], "share_price": None},
            "B&M": {"cash": 0, "companies": [], "trains": [], "share_price": None},
        },
    }
    assert action_helper.get_state() == expected_state, "State does not match expected state after initial auction"

    # Stock Round 1
    g.process_action(action_helper.get_all_choices()[-2])  # Par PRR
    g.process_action(action_helper.get_all_choices()[-1])  # Pass
    g.process_action(action_helper.get_all_choices()[-8])  # Par NYC
    g.process_action(action_helper.get_all_choices()[1])  # Buy PRR
    g.process_action(action_helper.get_all_choices()[1])  # Buy PRR
    g.process_action(action_helper.get_all_choices()[14])  # Par C&O
    g.process_action(action_helper.get_all_choices()[2])  # Buy NYC
    g.process_action(action_helper.get_all_choices()[1])  # Buy PRR
    g.process_action(action_helper.get_all_choices()[1])  # Buy PRR
    g.process_action(action_helper.get_all_choices()[3])  # Buy C&O
    g.process_action(action_helper.get_all_choices()[2])  # Buy NYC
    g.process_action(action_helper.get_all_choices()[0])  # Buy PRR
    g.process_action(action_helper.get_all_choices()[1])  # Buy PRR
    g.process_action(action_helper.get_all_choices()[3])  # Buy C&O
    g.process_action(action_helper.get_all_choices()[2])  # Buy NYC
    g.process_action(action_helper.get_all_choices()[1])  # Buy PRR
    g.process_action(action_helper.get_all_choices()[2])  # Buy C&O
    g.process_action(action_helper.get_all_choices()[1])  # Buy NYC
    g.process_action(action_helper.get_all_choices()[1])  # Buy NYC
    g.process_action(action_helper.get_all_choices()[2])  # Buy C&O
    g.process_action(action_helper.get_all_choices()[1])  # Buy NYC
    g.process_action(action_helper.get_all_choices()[1])  # Buy NYC
    action_helper.print_summary(json_format=True)

    expected_state = {
        "players": {
            "Player 1": {"cash": 49, "shares": {"PRR": 60, "NYC": 20}, "companies": ["SV"]},
            "Player 2": {"cash": 76, "shares": {"C&O": 60, "NYC": 10}, "companies": ["DH"]},
            "Player 3": {"cash": 68, "shares": {"NYC": 60}, "companies": ["CS", "MH"]},
            "Player 4": {"cash": 69, "shares": {"PRR": 40, "B&O": 20}, "companies": ["CA", "BO"]},
        },
        "corporations": {
            "PRR": {"cash": 670, "companies": [], "trains": [], "share_price": 71},
            "NYC": {"cash": 670, "companies": [], "trains": [], "share_price": 67},
            "CPR": {"cash": 0, "companies": [], "trains": [], "share_price": None},
            "B&O": {"cash": 0, "companies": [], "trains": [], "share_price": 100},
            "C&O": {"cash": 670, "companies": [], "trains": [], "share_price": 67},
            "ERIE": {"cash": 0, "companies": [], "trains": [], "share_price": None},
            "NYNH": {"cash": 0, "companies": [], "trains": [], "share_price": None},
            "B&M": {"cash": 0, "companies": [], "trains": [], "share_price": None},
        },
    }
    assert action_helper.get_state() == expected_state, "State does not match expected state after stock round 1"
    # Operating Round 1
    # PRR
    g.process_action(action_helper.get_all_choices()[0])  # lays tile #57 with rotation 1 on H10
    g.process_action(action_helper.get_all_choices()[-1])  # passes place token
    g.process_action(action_helper.get_all_choices()[0])  # buys a 2 train
    g.process_action(action_helper.get_all_choices()[0])  # buys a 2 train
    g.process_action(action_helper.get_all_choices()[-1])  # passes trains

    # NYC
    g.process_action(action_helper.get_all_choices()[0])  # lays tile #57 with rotation 0 on E19
    g.process_action(action_helper.get_all_choices()[0])  # buys a 2 train
    g.process_action(action_helper.get_all_choices()[-1])  # passes trains

    # C&O
    g.process_action(action_helper.get_all_choices()[-5])
    g.process_action(action_helper.get_all_choices()[0])
    g.process_action(action_helper.get_all_choices()[-1])
    action_helper.print_summary(json_format=True)
    expected_state = {
        "players": {
            "Player 1": {"cash": 49, "shares": {"PRR": 60, "NYC": 20}, "companies": ["SV"]},
            "Player 2": {"cash": 76, "shares": {"C&O": 60, "NYC": 10}, "companies": ["DH"]},
            "Player 3": {"cash": 68, "shares": {"NYC": 60}, "companies": ["CS", "MH"]},
            "Player 4": {"cash": 69, "shares": {"PRR": 40, "B&O": 20}, "companies": ["CA", "BO"]},
        },
        "corporations": {
            "PRR": {"cash": 510, "companies": [], "trains": ["2", "2"], "share_price": 67},
            "NYC": {"cash": 590, "companies": [], "trains": ["2"], "share_price": 65},
            "CPR": {"cash": 0, "companies": [], "trains": [], "share_price": None},
            "B&O": {"cash": 0, "companies": [], "trains": [], "share_price": 100},
            "C&O": {"cash": 590, "companies": [], "trains": ["2"], "share_price": 65},
            "ERIE": {"cash": 0, "companies": [], "trains": [], "share_price": None},
            "NYNH": {"cash": 0, "companies": [], "trains": [], "share_price": None},
            "B&M": {"cash": 0, "companies": [], "trains": [], "share_price": None},
        },
    }
    assert action_helper.get_state() == expected_state, "State does not match expected state after operating round 1"

    # Stock Round 2
    g.process_action(action_helper.get_all_choices()[-2])  # sell nyc
    g.process_action(action_helper.get_all_choices()[-3])  # par nynh
    g.process_action(action_helper.get_all_choices()[0])  # buy C&O
    g.process_action(action_helper.get_all_choices()[-1])  # pass sell
    g.process_action(action_helper.get_all_choices()[0])  # Buy NYC
    g.process_action(action_helper.get_all_choices()[-1])  # pass sell
    g.process_action(action_helper.get_all_choices()[0])  # Buy NYNH
    g.process_action(action_helper.get_all_choices()[-1])  # pass sell
    g.process_action(action_helper.get_all_choices()[1])  # Buy NYNH
    g.process_action(action_helper.get_all_choices()[-1])  # pass sell
    g.process_action(action_helper.get_all_choices()[-1])  # pass
    g.process_action(action_helper.get_all_choices()[-1])  # pass
    g.process_action(action_helper.get_all_choices()[-1])  # pass
    g.process_action(action_helper.get_all_choices()[1])  # Buy NYNH
    g.process_action(action_helper.get_all_choices()[-1])  # pass sell
    g.process_action(action_helper.get_all_choices()[-1])  # pass
    g.process_action(action_helper.get_all_choices()[-1])  # pass
    g.process_action(action_helper.get_all_choices()[-1])  # pass
    g.process_action(action_helper.get_all_choices()[1])  # Buy NYNH
    g.process_action(action_helper.get_all_choices()[-1])  # pass sell
    g.process_action(action_helper.get_all_choices()[-1])  # pass
    g.process_action(action_helper.get_all_choices()[-1])  # pass
    g.process_action(action_helper.get_all_choices()[-1])  # pass
    g.process_action(action_helper.get_all_choices()[-1])  # pass
    action_helper.print_summary(json_format=True)
    expected_state = {
        "players": {
            "Player 1": {"cash": 24, "shares": {"PRR": 60, "NYC": 30}, "companies": ["SV"]},
            "Player 2": {"cash": 20, "shares": {"C&O": 60, "NYC": 10, "NYNH": 10}, "companies": ["DH"]},
            "Player 3": {"cash": 68, "shares": {"NYC": 10, "NYNH": 50}, "companies": ["CS", "MH"]},
            "Player 4": {"cash": 57, "shares": {"PRR": 40, "B&O": 20, "C&O": 10}, "companies": ["CA", "BO"]},
        },
        "corporations": {
            "PRR": {"cash": 510, "companies": [], "trains": ["2", "2"], "share_price": 71},
            "NYC": {"cash": 590, "companies": [], "trains": ["2"], "share_price": 30},
            "CPR": {"cash": 0, "companies": [], "trains": [], "share_price": None},
            "B&O": {"cash": 0, "companies": [], "trains": [], "share_price": 100},
            "C&O": {"cash": 590, "companies": [], "trains": ["2"], "share_price": 65},
            "ERIE": {"cash": 0, "companies": [], "trains": [], "share_price": None},
            "NYNH": {"cash": 710, "companies": [], "trains": [], "share_price": 71},
            "B&M": {"cash": 0, "companies": [], "trains": [], "share_price": None},
        },
    }
    assert action_helper.get_state() == expected_state, "State does not match expected state after stock round 2"

    # Operating Round 2
    # NYNH
    g.process_action(action_helper.get_all_choices()[0])  # lay #1 with rotation 0 on F20
    g.process_action(action_helper.get_all_choices()[0])  # buy 2 train
    g.process_action(action_helper.get_all_choices()[-1])  # pass trains

    # PRR
    g.process_action(action_helper.get_all_choices()[4])  # lay tile #9 with rotation 1 on H8
    g.process_action(action_helper.get_all_choices()[-1])  # pass token
    g.process_action(action_helper.get_all_choices()[0])  # auto trains & run
    g.process_action(action_helper.get_all_choices()[0])  # pay out
    g.process_action(action_helper.get_all_choices()[-1])  # pass trains

    # C&O
    g.process_action(action_helper.get_all_choices()[-3])  # lay tile #8 with rotation 2 on G3
    g.process_action(action_helper.get_all_choices()[0])  # auto trains & run
    g.process_action(action_helper.get_all_choices()[1])  # withhold
    g.process_action(action_helper.get_all_choices()[0])  # buy a 2 train
    g.process_action(action_helper.get_all_choices()[0])  # buy a 3 train
    g.process_action(action_helper.get_all_choices()[-1])  # pass trains
    g.process_action(action_helper.get_all_choices()[-2])  # buy DH from Player 2 for $140
    g.process_action(action_helper.get_all_choices()[0])  # pass buy companies

    # NYC
    g.process_action(action_helper.get_all_choices()[-10])  # lay tile #8 with rotation 3 on F18
    g.process_action(action_helper.get_all_choices()[31])  # buy 3 train
    g.process_action(action_helper.get_all_choices()[-1])  # pass trains
    g.process_action(action_helper.get_all_choices()[-2])  # buys Schuylkill Valley from Player 1 for $40
    g.process_action(action_helper.get_all_choices()[-1])  # pass buy companies
    action_helper.print_summary(json_format=True)
    expected_state = {
        "players": {
            "Player 1": {"cash": 82, "shares": {"PRR": 60, "NYC": 30}, "companies": []},
            "Player 2": {"cash": 160, "shares": {"C&O": 60, "NYC": 10, "NYNH": 10}, "companies": []},
            "Player 3": {"cash": 68, "shares": {"NYC": 10, "NYNH": 50}, "companies": ["CS", "MH"]},
            "Player 4": {"cash": 69, "shares": {"PRR": 40, "B&O": 20, "C&O": 10}, "companies": ["CA", "BO"]},
        },
        "corporations": {
            "PRR": {"cash": 510, "companies": [], "trains": ["2", "2"], "share_price": 76},
            "NYC": {"cash": 370, "companies": ["SV"], "trains": ["2", "3"], "share_price": 20},
            "CPR": {"cash": 0, "companies": [], "trains": [], "share_price": None},
            "B&O": {"cash": 0, "companies": [], "trains": [], "share_price": 100},
            "C&O": {"cash": 260, "companies": ["DH"], "trains": ["2", "2", "3"], "share_price": 58},
            "ERIE": {"cash": 0, "companies": [], "trains": [], "share_price": None},
            "NYNH": {"cash": 630, "companies": [], "trains": ["2"], "share_price": 67},
            "B&M": {"cash": 0, "companies": [], "trains": [], "share_price": None},
        },
    }
    assert action_helper.get_state() == expected_state, "State does not match expected state after operating round 2"

    # Stock Round 3
    g.process_action(action_helper.get_all_choices()[2])  # Buy NYC
    g.process_action(action_helper.get_all_choices()[-1])  # Pass Sell

    g.process_action(action_helper.get_all_choices()[3])  # Buy NYC
    g.process_action(action_helper.get_all_choices()[-1])  # Pass Sell

    g.process_action(action_helper.get_all_choices()[3])  # Buy NYC
    g.process_action(action_helper.get_all_choices()[-1])  # Pass Sell

    g.process_action(action_helper.get_all_choices()[3])  # Exchange for NYC
    g.process_action(action_helper.get_all_choices()[1])  # Buy last NYC
    g.process_action(action_helper.get_all_choices()[-1])  # Pass sell

    g.process_action(action_helper.get_all_choices()[-1])  # pass

    g.process_action(action_helper.get_all_choices()[-1])  # pass

    g.process_action(action_helper.get_all_choices()[1])  # Buy NYNH
    g.process_action(action_helper.get_all_choices()[-1])  # pass

    g.process_action(action_helper.get_all_choices()[-1])  # pass

    g.process_action(action_helper.get_all_choices()[-1])  # pass

    g.process_action(action_helper.get_all_choices()[-1])  # pass

    g.process_action(action_helper.get_all_choices()[-1])  # pass
    action_helper.print_summary(json_format=True)
    expected_state = {
        "players": {
            "Player 1": {"cash": 62, "shares": {"PRR": 60, "NYC": 40}, "companies": []},
            "Player 2": {"cash": 69, "shares": {"C&O": 60, "NYC": 20, "NYNH": 20}, "companies": []},
            "Player 3": {"cash": 58, "shares": {"NYC": 30, "NYNH": 50}, "companies": ["CS"]},
            "Player 4": {
                "cash": 104,
                "shares": {"PRR": 40, "B&O": 20, "C&O": 10, "NYC": 10},
                "companies": ["CA", "BO"],
            },
        },
        "corporations": {
            "PRR": {"cash": 510, "companies": [], "trains": ["2", "2"], "share_price": 82},
            "NYC": {"cash": 375, "companies": ["SV"], "trains": ["2", "3"], "share_price": 30},
            "CPR": {"cash": 0, "companies": [], "trains": [], "share_price": None},
            "B&O": {"cash": 0, "companies": [], "trains": [], "share_price": 100},
            "C&O": {"cash": 275, "companies": ["DH"], "trains": ["2", "2", "3"], "share_price": 58},
            "ERIE": {"cash": 0, "companies": [], "trains": [], "share_price": None},
            "NYNH": {"cash": 630, "companies": [], "trains": ["2"], "share_price": 67},
            "B&M": {"cash": 0, "companies": [], "trains": [], "share_price": None},
        },
    }
    assert action_helper.get_state() == expected_state, "State does not match expected state after stock round 3"

    # Operating Round 3
    # PRR
    g.process_action(action_helper.get_all_choices()[-11])  # lay tile 8 with rotation 2 on H6
    g.process_action(action_helper.get_all_choices()[0])  # Token Pittsburgh
    g.process_action(action_helper.get_all_choices()[0])  # Run 2 train for 30
    g.process_action(action_helper.get_all_choices()[0])  # pay out
    g.process_action(action_helper.get_all_choices()[-1])  # pass trains
    g.process_action(action_helper.get_all_choices()[-1])  # pass companies

    # NYNH
    g.process_action(action_helper.get_all_choices()[-5])  # Buy C&S
    g.process_action(action_helper.get_all_choices()[0])  # lay tile 54 with rotation 0 on G19
    g.process_action(action_helper.get_all_choices()[-1])  # run routes
    g.process_action(action_helper.get_all_choices()[0])  # pay out
    g.process_action(action_helper.get_all_choices()[0])  # buy 3 train
    g.process_action(action_helper.get_all_choices()[-1])  # pass trains
    g.process_action(action_helper.get_all_choices()[-4])  # lay C&S tile
    g.process_action(action_helper.get_all_choices()[-1])  # pass companies

    # C&O
    g.process_action(action_helper.get_all_choices()[-6])  # Lay tile 57 with rotation 1 on F16
    g.process_action(action_helper.get_all_choices()[0])  # Token teleport
    g.process_action(action_helper.get_all_choices()[0])  # run routes
    g.process_action(action_helper.get_all_choices()[0])  # pay out
    g.process_action(action_helper.get_all_choices()[-1])  # pass companies

    # NYC
    g.process_action(action_helper.get_all_choices()[8])  # lay tile 19 with rotation 1 on F18
    g.process_action(action_helper.get_all_choices()[0])  # run trains
    g.process_action(action_helper.get_all_choices()[0])  # pay out
    g.process_action(action_helper.get_all_choices()[-1])  # pass trains
    g.process_action(action_helper.get_all_choices()[-1])  # pass companies

    # PRR
    g.process_action(action_helper.get_all_choices()[-3])  # lay tile 25 with rotation 1 on g5
    g.process_action(action_helper.get_all_choices()[0])  # run routes
    g.process_action(action_helper.get_all_choices()[0])  # pay out
    g.process_action(action_helper.get_all_choices()[-1])  # pass trains
    g.process_action(action_helper.get_all_choices()[-1])  # pass companies

    # NYNH
    g.process_action(action_helper.get_all_choices()[0])  # lay tile 14 with rotation 0 on E19
    g.process_action(action_helper.get_all_choices()[0])  # place a token on E19
    g.process_action(action_helper.get_all_choices()[0])  # run trains
    g.process_action(action_helper.get_all_choices()[0])  # pay out
    g.process_action(action_helper.get_all_choices()[-1])  # pass trains
    g.process_action(action_helper.get_all_choices()[-1])  # pass companies

    # C&O
    g.process_action(action_helper.get_all_choices()[5])  # lay tile 15 with rotation 4 on F16
    g.process_action(action_helper.get_all_choices()[0])  # run trains
    g.process_action(action_helper.get_all_choices()[0])  # pay out
    g.process_action(action_helper.get_all_choices()[-1])  # pass companies

    # NYC
    g.process_action(action_helper.get_all_choices()[-7])  # lay tile #7 with rotation 0 on E21
    g.process_action(action_helper.get_all_choices()[0])  # place token on F16
    g.process_action(action_helper.get_all_choices()[0])  # run trains
    g.process_action(action_helper.get_all_choices()[0])  # pay out
    g.process_action(action_helper.get_all_choices()[-1])  # pass trains
    g.process_action(action_helper.get_all_choices()[-1])  # pass companies
    action_helper.print_summary(json_format=True)
    expected_state = {
        "players": {
            "Player 1": {"cash": 230, "shares": {"PRR": 60, "NYC": 40}, "companies": []},
            "Player 2": {"cash": 273, "shares": {"C&O": 60, "NYC": 20, "NYNH": 20}, "companies": []},
            "Player 3": {"cash": 330, "shares": {"NYC": 30, "NYNH": 50}, "companies": []},
            "Player 4": {
                "cash": 249,
                "shares": {"PRR": 40, "B&O": 20, "C&O": 10, "NYC": 10},
                "companies": ["CA", "BO"],
            },
        },
        "corporations": {
            "PRR": {"cash": 470, "companies": [], "trains": ["2", "2"], "share_price": 100},
            "NYC": {"cash": 220, "companies": ["SV"], "trains": ["2", "3"], "share_price": 50},
            "CPR": {"cash": 0, "companies": [], "trains": [], "share_price": None},
            "B&O": {"cash": 0, "companies": [], "trains": [], "share_price": 100},
            "C&O": {"cash": 170, "companies": ["DH"], "trains": ["2", "2", "3"], "share_price": 67},
            "ERIE": {"cash": 0, "companies": [], "trains": [], "share_price": None},
            "NYNH": {"cash": 260, "companies": ["CS"], "trains": ["2", "3"], "share_price": 76},
            "B&M": {"cash": 0, "companies": [], "trains": [], "share_price": None},
        },
    }
    assert action_helper.get_state() == expected_state, "State does not match expected state after operating round 3"

    # Stock Round 4
    g.process_action(action_helper.get_all_choices()[0])  # buy B&O
    g.process_action(action_helper.get_all_choices()[-1])  # pass sell

    g.process_action(action_helper.get_all_choices()[0])  # buy B&O
    g.process_action(action_helper.get_all_choices()[-1])  # pass sell

    g.process_action(action_helper.get_all_choices()[2])  # buy C&O
    g.process_action(action_helper.get_all_choices()[-1])  # pass sell

    g.process_action(action_helper.get_all_choices()[1])  # buy NYNH
    g.process_action(action_helper.get_all_choices()[-1])  # pass sell

    g.process_action(action_helper.get_all_choices()[0])  # buy B&O
    g.process_action(action_helper.get_all_choices()[-1])  # pass sell

    g.process_action(action_helper.get_all_choices()[0])  # buy B&O
    g.process_action(action_helper.get_all_choices()[-1])  # pass sell

    g.process_action(action_helper.get_all_choices()[2])  # buy C&O
    g.process_action(action_helper.get_all_choices()[-1])  # pass sell

    g.process_action(action_helper.get_all_choices()[1])  # buy NYNH
    g.process_action(action_helper.get_all_choices()[-1])  # pass sell

    g.process_action(action_helper.get_all_choices()[0])  # buy B&O
    g.process_action(action_helper.get_all_choices()[-1])  # pass sell

    g.process_action(action_helper.get_all_choices()[-1])  # pass

    g.process_action(action_helper.get_all_choices()[1])  # buy C&O
    g.process_action(action_helper.get_all_choices()[-1])  # pass sell

    g.process_action(action_helper.get_all_choices()[1])  # buy NYNH
    g.process_action(action_helper.get_all_choices()[-1])  # pass sell

    g.process_action(action_helper.get_all_choices()[-1])  # pass
    g.process_action(action_helper.get_all_choices()[-1])  # pass
    g.process_action(action_helper.get_all_choices()[-1])  # pass
    g.process_action(action_helper.get_all_choices()[-1])  # pass
    action_helper.print_summary(json_format=True)
    expected_state = {
        "players": {
            "Player 1": {"cash": 29, "shares": {"PRR": 60, "NYC": 40, "C&O": 30}, "companies": []},
            "Player 2": {"cash": 60, "shares": {"C&O": 60, "NYC": 20, "NYNH": 50}, "companies": []},
            "Player 3": {"cash": 30, "shares": {"NYC": 30, "NYNH": 50, "B&O": 30}, "companies": []},
            "Player 4": {
                "cash": 104,
                "shares": {"PRR": 40, "B&O": 40, "C&O": 10, "NYC": 10},
                "companies": ["CA", "BO"],
            },
        },
        "corporations": {
            "PRR": {"cash": 470, "companies": [], "trains": ["2", "2"], "share_price": 112},
            "NYC": {"cash": 225, "companies": ["SV"], "trains": ["2", "3"], "share_price": 60},
            "CPR": {"cash": 0, "companies": [], "trains": [], "share_price": None},
            "B&O": {"cash": 1000, "companies": [], "trains": [], "share_price": 100},
            "C&O": {"cash": 185, "companies": ["DH"], "trains": ["2", "2", "3"], "share_price": 71},
            "ERIE": {"cash": 0, "companies": [], "trains": [], "share_price": None},
            "NYNH": {"cash": 270, "companies": ["CS"], "trains": ["2", "3"], "share_price": 82},
            "B&M": {"cash": 0, "companies": [], "trains": [], "share_price": None},
        },
    }
    assert action_helper.get_state() == expected_state, "State does not match expected state after stock round 4"

    # Operating Round 4
    # PRR
    g.process_action(action_helper.get_all_choices()[9])  # lay tile #8 with rotation 5 on H14
    g.process_action(action_helper.get_all_choices()[0])  # Run trains
    g.process_action(action_helper.get_all_choices()[0])  # pay out
    g.process_action(action_helper.get_all_choices()[0])  # buy 3 train
    g.process_action(action_helper.get_all_choices()[-1])  # pass trains
    g.process_action(action_helper.get_all_choices()[-1])  # pass companies

    # B&O
    g.process_action(action_helper.get_all_choices()[-12])  # buy C&A for max
    g.process_action(action_helper.get_all_choices()[5])  # lay tile #8 with rotation 1 on I17
    g.process_action(action_helper.get_all_choices()[0])  # buy 3 train
    g.process_action(action_helper.get_all_choices()[0])  # buy 4 train

    # NYNH
    g.process_action(action_helper.get_all_choices()[-3])  # lay tile #57 with rotation 1 on F22
    g.process_action(action_helper.get_all_choices()[0])  # run trains
    g.process_action(action_helper.get_all_choices()[0])  # pay out

    # C&O
    g.process_action(action_helper.get_all_choices()[-20])  # lay tile #56 with rotation 1 on G17
    g.process_action(action_helper.get_all_choices()[0])  # place token on G19
    g.process_action(action_helper.get_all_choices()[0])  # run trains
    g.process_action(action_helper.get_all_choices()[0])  # pay out

    # NYC
    g.process_action(action_helper.get_all_choices()[13])  # lay tile #26 with rotation 1 on E21
    g.process_action(action_helper.get_all_choices()[0])  # run trains
    g.process_action(action_helper.get_all_choices()[0])  # pay out
    g.process_action(action_helper.get_all_choices()[109])  # buy a 3 train for $110 from PRR

    # PRR
    g.process_action(action_helper.get_all_choices()[12])  # lay tile #53 with rotation 0 on I15
    g.process_action(action_helper.get_all_choices()[0])  # buy 4 train
    g.process_action(action_helper.get_all_choices()[-1])  # pass trains

    # NYNH
    g.process_action(action_helper.get_all_choices()[12])  # tile #53 with rotation 1 on E23
    g.process_action(action_helper.get_all_choices()[0])  # runs a 3 train for $140: G19-E19-E23
    g.process_action(action_helper.get_all_choices()[1])  # withholds $140
    g.process_action(action_helper.get_all_choices()[0])  # buys a 4 train for $300 from The Depot

    # B&O
    g.process_action(action_helper.get_all_choices()[-2])  # lay tile #59 with rotation 0 on H18
    g.process_action(action_helper.get_all_choices()[-1])  # skip token
    g.process_action(action_helper.get_all_choices()[0])  # run trains
    g.process_action(action_helper.get_all_choices()[0])  # pay out

    # C&O
    g.process_action(action_helper.get_all_choices()[8])  # lay tile #69 with rotation 0 on G7
    g.process_action(action_helper.get_all_choices()[0])  # run trains
    g.process_action(action_helper.get_all_choices()[0])  # pay out

    # NYC
    g.process_action(action_helper.get_all_choices()[-3])  # lay tile 9 with rotation 1 on F14
    g.process_action(action_helper.get_all_choices()[-1])  # skip token
    g.process_action(action_helper.get_all_choices()[0])  # run trains
    g.process_action(action_helper.get_all_choices()[0])  # pay out
    g.process_action(action_helper.get_all_choices()[-1])  # pass trains
    action_helper.print_summary(json_format=True)
    expected_state = {
        "players": {
            "Player 1": {"cash": 279, "shares": {"PRR": 60, "NYC": 40, "C&O": 30}, "companies": []},
            "Player 2": {"cash": 298, "shares": {"C&O": 60, "NYC": 20, "NYNH": 50}, "companies": []},
            "Player 3": {"cash": 230, "shares": {"NYC": 30, "NYNH": 50, "B&O": 30}, "companies": []},
            "Player 4": {"cash": 578, "shares": {"PRR": 40, "B&O": 40, "C&O": 10, "NYC": 10}, "companies": []},
        },
        "corporations": {
            "PRR": {"cash": 100, "companies": [], "trains": ["4"], "share_price": 112},
            "NYC": {"cash": 120, "companies": ["SV"], "trains": ["3", "3"], "share_price": 68},
            "CPR": {"cash": 0, "companies": [], "trains": [], "share_price": None},
            "B&O": {"cash": 145, "companies": ["CA"], "trains": ["3", "4"], "share_price": 100},
            "C&O": {"cash": 100, "companies": ["DH"], "trains": ["3"], "share_price": 82},
            "ERIE": {"cash": 0, "companies": [], "trains": [], "share_price": None},
            "NYNH": {"cash": 40, "companies": ["CS"], "trains": ["3", "4"], "share_price": 82},
            "B&M": {"cash": 0, "companies": [], "trains": [], "share_price": None},
        },
    }
    assert action_helper.get_state() == expected_state, "State does not match expected state after operating round 4"

    # Stock Round 5
    g.process_action(
        action_helper.get_all_choices()[0]
    )  # [18:00] Player 3 buys a 10% share of B&O from the IPO for $100
    g.process_action(action_helper.get_all_choices()[-1])  # [18:00] Player 3 declines to sell shares
    g.process_action(
        action_helper.get_all_choices()[0]
    )  # [18:00] Player 4 buys a 10% share of B&O from the IPO for $100
    g.process_action(action_helper.get_all_choices()[-1])  # [18:00] Player 4 declines to sell shares
    g.process_action(
        action_helper.get_all_choices()[0]
    )  # [18:00] Player 1 buys a 10% share of B&O from the IPO for $100
    g.process_action(action_helper.get_all_choices()[-1])  # [18:00] Player 1 declines to sell shares
    g.process_action(action_helper.get_all_choices()[12])  # [18:01] Player 2 pars ERIE at $100
    # [18:01] Player 2 buys a 20% share of ERIE from the IPO for $200
    # [18:01] Player 2 becomes the president of ERIE
    g.process_action(action_helper.get_all_choices()[-1])  # [18:01] Player 2 declines to sell shares
    g.process_action(
        action_helper.get_all_choices()[0]
    )  # [18:01] Player 3 buys a 10% share of ERIE from the IPO for $100
    g.process_action(action_helper.get_all_choices()[-1])  # [18:04] Player 3 declines to sell shares
    g.process_action(
        action_helper.get_all_choices()[0]
    )  # [18:04] Player 4 buys a 10% share of ERIE from the IPO for $100
    g.process_action(action_helper.get_all_choices()[-1])  # [18:04] Player 4 declines to sell shares
    g.process_action(
        action_helper.get_all_choices()[0]
    )  # [18:04] Player 1 buys a 10% share of ERIE from the IPO for $100
    g.process_action(action_helper.get_all_choices()[-1])  # [18:04] Player 1 declines to sell shares
    g.process_action(action_helper.get_all_choices()[0])  # [18:05] Player 2 sells a 10% share of NYC and receives $68
    g.process_action(
        action_helper.get_all_choices()[0]
    )  # [18:05] Player 2 buys a 10% share of ERIE from the IPO for $100
    # [18:05] ERIE floats
    # [18:05] ERIE receives $1000
    g.process_action(action_helper.get_all_choices()[0])  # [18:05] Player 2 sells a 10% share of NYC and receives $68
    g.process_action(action_helper.get_all_choices()[-1])  # [18:05] Player 2 passes
    g.process_action(action_helper.get_all_choices()[-1])  # [18:05] Player 3 passes
    g.process_action(
        action_helper.get_all_choices()[0]
    )  # [18:05] Player 4 buys a 10% share of ERIE from the IPO for $100
    g.process_action(action_helper.get_all_choices()[-1])  # [18:05] Player 4 declines to sell shares
    g.process_action(
        action_helper.get_all_choices()[0]
    )  # [18:05] Player 1 buys a 10% share of NYC from the market for $68
    g.process_action(action_helper.get_all_choices()[-1])  # [18:05] Player 1 declines to sell shares
    g.process_action(
        action_helper.get_all_choices()[0]
    )  # [18:06] Player 2 buys a 10% share of ERIE from the IPO for $100
    g.process_action(action_helper.get_all_choices()[-1])  # [18:06] Player 2 declines to sell shares
    g.process_action(action_helper.get_all_choices()[-1])  # [18:06] Player 3 passes
    g.process_action(
        action_helper.get_all_choices()[0]
    )  # [18:06] Player 4 buys a 10% share of ERIE from the IPO for $100
    g.process_action(action_helper.get_all_choices()[-1])  # [18:06] Player 4 declines to sell shares
    g.process_action(action_helper.get_all_choices()[-1])  # [18:06] Player 1 passes
    g.process_action(action_helper.get_all_choices()[-1])  # [18:06] Player 2 passes
    g.process_action(action_helper.get_all_choices()[-1])  # [18:06] Player 3 passes
    g.process_action(
        action_helper.get_all_choices()[0]
    )  # [18:06] Player 4 buys a 10% share of ERIE from the IPO for $100
    g.process_action(action_helper.get_all_choices()[-1])  # [18:06] Player 4 declines to sell shares
    g.process_action(action_helper.get_all_choices()[-1])  # [18:06] Player 1 passes
    g.process_action(action_helper.get_all_choices()[-1])  # [18:06] Player 2 passes
    g.process_action(action_helper.get_all_choices()[-1])  # [18:06] Player 3 passes
    g.process_action(
        action_helper.get_all_choices()[0]
    )  # [18:06] Player 4 buys a 10% share of NYC from the market for $68
    g.process_action(action_helper.get_all_choices()[-1])  # [18:06] Player 4 declines to sell shares
    g.process_action(action_helper.get_all_choices()[-1])  # pass
    g.process_action(action_helper.get_all_choices()[-1])  # pass
    g.process_action(action_helper.get_all_choices()[-1])  # pass
    g.process_action(action_helper.get_all_choices()[-1])  # pass
    action_helper.print_summary(json_format=True)
    expected_state = {
        "players": {
            "Player 1": {
                "cash": 11,
                "shares": {"PRR": 60, "NYC": 50, "C&O": 30, "B&O": 10, "ERIE": 10},
                "companies": [],
            },
            "Player 2": {"cash": 34, "shares": {"C&O": 60, "NYC": 0, "NYNH": 50, "ERIE": 40}, "companies": []},
            "Player 3": {"cash": 30, "shares": {"NYC": 30, "NYNH": 50, "B&O": 40, "ERIE": 10}, "companies": []},
            "Player 4": {
                "cash": 10,
                "shares": {"PRR": 40, "B&O": 50, "C&O": 10, "NYC": 20, "ERIE": 40},
                "companies": [],
            },
        },
        "corporations": {
            "PRR": {"cash": 100, "companies": [], "trains": ["4"], "share_price": 126},
            "NYC": {"cash": 125, "companies": ["SV"], "trains": ["3", "3"], "share_price": 69},
            "CPR": {"cash": 0, "companies": [], "trains": [], "share_price": None},
            "B&O": {"cash": 170, "companies": ["CA"], "trains": ["3", "4"], "share_price": 100},
            "C&O": {"cash": 115, "companies": ["DH"], "trains": ["3"], "share_price": 90},
            "ERIE": {"cash": 1000, "companies": [], "trains": [], "share_price": 100},
            "NYNH": {"cash": 50, "companies": ["CS"], "trains": ["3", "4"], "share_price": 90},
            "B&M": {"cash": 0, "companies": [], "trains": [], "share_price": None},
        },
    }
    assert action_helper.get_state() == expected_state, "State does not match expected state after stock round 5"

    # Operating Round 5
    # [18:06] B&O collects $25 from Camden & Amboy
    # [18:06] C&O collects $15 from Delaware & Hudson
    # [18:06] NYNH collects $10 from Champlain & St.Lawrence
    # [18:06] NYC collects $5 from Schuylkill Valley

    # [18:06] Player 1 operates PRR
    g.process_action(
        action_helper.get_all_choices()[4]
    )  # [18:12] PRR lays tile #15 with rotation 1 on H10 (Pittsburgh)
    # [18:12] PRR skips place a token
    g.process_action(action_helper.get_all_choices()[0])  # [18:12] PRR runs a 4 train for $130: F2-H10-H12-I15
    g.process_action(action_helper.get_all_choices()[0])  # [18:12] PRR pays out 13 per share (52 to Player 4)
    # [18:12] PRR's share price moves right from 142
    g.process_action(action_helper.get_all_choices()[-1])  # [18:12] PRR passes buy trains
    # [18:12] PRR skips buy companies

    # [18:12] Player 4 operates B&O
    g.process_action(
        action_helper.get_all_choices()[0]
    )  # [18:13] B&O spends $80 and lays tile #57 with rotation 0 on J14 (Washington)
    g.process_action(action_helper.get_all_choices()[1])  # [18:13] B&O places a token on H10 (Pittsburgh) for $40
    g.process_action(action_helper.get_all_choices()[0])  # [18:13] B&O runs a 4 train for $140: K13-J14-I15-H18
    # [18:13] B&O runs a 3 train for $120: I15-H10-F2
    g.process_action(
        action_helper.get_all_choices()[0]
    )  # [18:13] B&O pays out 26 per share (104 to Player 3, $26 to Player 1)
    # [18:13] B&O's share price moves right from 112
    # [18:13] B&O passes buy trains
    # [18:13] B&O skips buy companies

    # [18:13] Player 2 operates ERIE
    # [18:13] ERIE places a token on E11
    g.process_action(
        action_helper.get_all_choices()[2]
    )  # [18:13] ERIE lays tile #59 with rotation 3 on E11 (Dunkirk & Buffalo)
    # [18:13] ERIE must choose city for token
    g.process_action(action_helper.get_all_choices()[1])  # [18:13] ERIE places a token on E11 (Dunkirk & Buffalo)
    # [18:13] ERIE skips place a token
    # [18:13] ERIE skips run routes
    # [18:13] ERIE does not run
    # [18:13] ERIE's share price moves left from 90
    g.process_action(action_helper.get_all_choices()[0])  # [18:13] ERIE buys a 4 train for $300 from The Depot
    g.process_action(action_helper.get_all_choices()[0])  # [18:13] ERIE buys a 5 train for $450 from The Depot
    # [18:13] -- Phase 5 (Operating Rounds: 3 | Train Limit: 2 | Available Tiles: Yellow, Green, Brown) --
    # [18:13] -- Event: Private companies close --
    # [18:13] ERIE skips buy companies

    # [18:13] Player 2 operates C&O
    g.process_action(
        action_helper.get_all_choices()[-4]
    )  # [18:16] C&O lays tile #66 with rotation 0 on H18 (Philadelphia & Trenton)
    # [18:16] C&O skips place a token
    g.process_action(action_helper.get_all_choices()[0])  # [18:16] C&O runs a 3 train for $160: G19-H18-I15
    g.process_action(action_helper.get_all_choices()[1])  # [18:16] C&O withholds $160
    # [18:16] C&O's share price moves left from 82
    g.process_action(action_helper.get_all_choices()[474])  # [18:17] C&O buys a 5 train for $200 from ERIE
    # [18:17] C&O skips buy companies

    # [18:17] Player 3 operates NYNH
    g.process_action(action_helper.get_all_choices()[0])  # [18:19] NYNH lays tile #63 with rotation 0 on E19 (Albany)
    # [18:19] NYNH skips place a token
    g.process_action(action_helper.get_all_choices()[0])  # [18:19] NYNH runs a 4 train for $100: G19-F20-F22-F24
    # [18:19] NYNH runs a 3 train for $150: G19-E19-E23
    g.process_action(action_helper.get_all_choices()[0])  # [18:21] NYNH pays out 25 per share (125 to Player 3)
    # [18:21] NYNH's share price moves right from 100
    # [18:21] NYNH skips buy trains
    # [18:21] NYNH skips buy companies

    # [18:21] Player 1 operates NYC
    g.process_action(
        action_helper.get_all_choices()[8]
    )  # [18:21] NYC lays tile #62 with rotation 0 on G19 (New York & Newark)
    g.process_action(
        action_helper.get_all_choices()[3]
    )  # [18:21] NYC places a token on H18 (Philadelphia & Trenton) for $100
    g.process_action(action_helper.get_all_choices()[0])  # [18:21] NYC runs a 3 train for $170: G19-E19-E23
    # [18:21] NYC runs a 3 train for $180: G19-H18-I15
    g.process_action(action_helper.get_all_choices()[1])  # [18:21] NYC withholds $350
    # [18:21] NYC's share price moves left from 67
    # [18:21] NYC skips buy trains
    # [18:21] NYC skips buy companies

    # [18:21] -- Operating Round 5.2 (of 2) --
    # [18:21] Player 1 operates PRR
    g.process_action(action_helper.get_all_choices()[-3])  # [18:21] PRR lays tile #9 with rotation 0 on G11
    # [18:21] PRR skips place a token
    g.process_action(action_helper.get_all_choices()[0])  # [18:21] PRR runs a 4 train for $160: F2-H10-H12-I15
    g.process_action(action_helper.get_all_choices()[1])  # [18:21] PRR withholds $160
    # [18:21] PRR's share price moves left from 126
    g.process_action(action_helper.get_all_choices()[-1])  # [18:21] PRR passes buy trains
    # [18:21] PRR skips buy companies

    # [18:21] Player 4 operates B&O
    g.process_action(action_helper.get_all_choices()[6])  # [18:22] B&O lays tile #61 with rotation 0 on I15 (Baltimore)
    # [18:22] B&O skips place a token
    g.process_action(action_helper.get_all_choices()[0])  # [18:22] B&O runs a 4 train for $170: K13-J14-I15-H18
    # [18:22] B&O runs a 3 train for $160: I15-H10-F2
    g.process_action(action_helper.get_all_choices()[1])  # [18:22] B&O withholds $330
    # [18:22] B&O's share price moves left from 100
    # [18:22] B&O skips buy trains
    # [18:22] B&O skips buy companies

    # [18:22] Player 3 operates NYNH
    g.process_action(action_helper.get_all_choices()[5])  # [18:22] NYNH lays tile #8 with rotation 5 on D18
    # [18:22] NYNH skips place a token
    g.process_action(action_helper.get_all_choices()[0])  # [18:22] NYNH runs a 4 train for $120: G19-F20-F22-F24
    # [18:22] NYNH runs a 3 train for $170: G19-E19-E23
    g.process_action(action_helper.get_all_choices()[0])  # [18:22] NYNH pays out 29 per share (145 to Player 3)
    # [18:22] NYNH's share price moves right from 111
    # [18:22] NYNH skips buy trains
    # [18:22] NYNH skips buy companies

    # [18:22] Player 2 operates ERIE
    g.process_action(action_helper.get_all_choices()[5])  # [18:22] ERIE lays tile #8 with rotation 2 on F12
    # [18:22] ERIE skips place a token
    g.process_action(action_helper.get_all_choices()[0])  # [18:22] ERIE runs a 4 train for $70: E11-F16
    g.process_action(
        action_helper.get_all_choices()[0]
    )  # [18:23] ERIE pays out 7 per share (28 to Player 4, 7 to Player 3)
    # [18:23] ERIE's share price moves right from 100
    g.process_action(action_helper.get_all_choices()[0])  # [18:23] ERIE buys a 5 train for $450 from The Depot
    # [18:23] ERIE skips buy companies

    # [18:23] Player 2 operates C&O
    g.process_action(
        action_helper.get_all_choices()[-3]
    )  # [18:23] C&O lays tile #68 with rotation 2 on E11 (Dunkirk & Buffalo)
    # [18:23] C&O skips place a token
    g.process_action(action_helper.get_all_choices()[0])  # [18:23] C&O runs a 5 train for $220: E11-F16-G17-G19-H18
    # [18:23] C&O runs a 3 train for $110: F2-F6-G7
    g.process_action(
        action_helper.get_all_choices()[0]
    )  # [18:23] C&O pays out 33 per share (99 to Player 1, $33 to Player 4)
    # [18:23] C&O's share price moves right from 90
    # [18:23] C&O skips buy trains
    # [18:23] C&O skips buy companies

    # [18:23] Player 1 operates NYC
    g.process_action(action_helper.get_all_choices()[41])  # [18:23] NYC lays tile #9 with rotation 1 on D16
    g.process_action(action_helper.get_all_choices()[-1])  # [18:23] NYC passes place a token
    g.process_action(action_helper.get_all_choices()[0])  # [18:23] NYC runs a 3 train for $170: G19-E19-E23
    # [18:23] NYC runs a 3 train for $190: G19-H18-I15
    g.process_action(action_helper.get_all_choices()[1])  # [18:23] NYC withholds $360
    # [18:23] NYC's share price moves left from 63
    # [18:23] NYC skips buy trains
    # [18:23] NYC skips buy companies
    action_helper.print_summary(json_format=True)
    expected_state = {
        "players": {
            "Player 1": {
                "cash": 221,
                "shares": {"PRR": 60, "NYC": 50, "C&O": 30, "B&O": 10, "ERIE": 10},
                "companies": [],
            },
            "Player 2": {"cash": 530, "shares": {"C&O": 60, "NYC": 0, "NYNH": 50, "ERIE": 40}, "companies": []},
            "Player 3": {"cash": 411, "shares": {"NYC": 30, "NYNH": 50, "B&O": 40, "ERIE": 10}, "companies": []},
            "Player 4": {
                "cash": 253,
                "shares": {"PRR": 40, "B&O": 50, "C&O": 10, "NYC": 20, "ERIE": 40},
                "companies": [],
            },
        },
        "corporations": {
            "PRR": {"cash": 260, "companies": [], "trains": ["4"], "share_price": 126},
            "NYC": {"cash": 735, "companies": [], "trains": ["3", "3"], "share_price": 63},
            "CPR": {"cash": 0, "companies": [], "trains": [], "share_price": None},
            "B&O": {"cash": 380, "companies": [], "trains": ["3", "4"], "share_price": 100},
            "C&O": {"cash": 75, "companies": [], "trains": ["3", "5"], "share_price": 90},
            "ERIE": {"cash": 0, "companies": [], "trains": ["4", "5"], "share_price": 100},
            "NYNH": {"cash": 50, "companies": [], "trains": ["3", "4"], "share_price": 111},
            "B&M": {"cash": 0, "companies": [], "trains": [], "share_price": None},
        },
    }
    assert action_helper.get_state() == expected_state, "State does not match expected state after operating round 5"

    # Stock Round 6
    g.process_action(action_helper.get_all_choices()[0])  # [19:11] Player 1 pars B&M at $100
    # [19:11] Player 1 buys a 20% share of B&M from the IPO for $200
    # [19:11] Player 1 becomes the president of B&M
    g.process_action(action_helper.get_all_choices()[-1])  # [19:11] Player 1 declines to sell shares
    g.process_action(action_helper.get_all_choices()[1])  # [19:12] Player 2 pars CPR at $100
    # [19:12] Player 2 buys a 20% share of CPR from the IPO for $200
    # [19:12] Player 2 becomes the president of CPR
    g.process_action(action_helper.get_all_choices()[9])  # [19:13] Player 2 sells a 10% share of NYNH and receives $111
    # [19:13] NYNH's share price moves down from 100
    g.process_action(action_helper.get_all_choices()[-1])
    g.process_action(
        action_helper.get_all_choices()[2]
    )  # [19:14] Player 3 buys a 10% share of CPR from the IPO for $100
    g.process_action(action_helper.get_all_choices()[-1])  # [19:14] Player 3 declines to sell shares
    g.process_action(
        action_helper.get_all_choices()[2]
    )  # [19:14] Player 4 buys a 10% share of CPR from the IPO for $100
    g.process_action(action_helper.get_all_choices()[-1])  # [19:14] Player 4 declines to sell shares
    g.process_action(action_helper.get_all_choices()[2])  # [19:15] Player 1 sells 3 shares of PRR and receives $378
    # [19:15] Player 4 becomes the president of PRR
    # [19:15] PRR's share price moves down from 90
    g.process_action(
        action_helper.get_all_choices()[-2]
    )  # [19:15] Player 1 sells a 10% share of ERIE and receives $100
    # [19:15] ERIE's share price moves down from 90
    g.process_action(
        action_helper.get_all_choices()[1]
    )  # [19:15] Player 1 buys a 10% share of B&M from the IPO for $100
    g.process_action(action_helper.get_all_choices()[-1])
    g.process_action(
        action_helper.get_all_choices()[3]
    )  # [19:15] Player 2 buys a 10% share of ERIE from the market for $90
    g.process_action(action_helper.get_all_choices()[-1])  # [19:15] Player 2 declines to sell shares
    g.process_action(
        action_helper.get_all_choices()[0]
    )  # [19:15] Player 3 buys a 10% share of NYNH from the market for $100
    g.process_action(action_helper.get_all_choices()[-1])  # [19:15] Player 3 declines to sell shares
    g.process_action(
        action_helper.get_all_choices()[1]
    )  # [19:15] Player 4 buys a 10% share of CPR from the IPO for $100
    g.process_action(action_helper.get_all_choices()[-1])  # [19:15] Player 4 declines to sell shares
    g.process_action(
        action_helper.get_all_choices()[0]
    )  # [19:15] Player 1 buys a 10% share of B&M from the IPO for $100
    g.process_action(action_helper.get_all_choices()[-1])  # [19:16] Player 1 declines to sell shares
    g.process_action(
        action_helper.get_all_choices()[1]
    )  # [19:16] Player 2 buys a 10% share of CPR from the IPO for $100
    # [19:16] CPR floats
    # [19:16] CPR receives $1000
    g.process_action(action_helper.get_all_choices()[-1])  # [19:16] Player 2 declines to sell shares
    g.process_action(
        action_helper.get_all_choices()[2]
    )  # [19:16] Player 3 buys a 10% share of PRR from the market for $90
    g.process_action(action_helper.get_all_choices()[-1])  # [19:16] Player 3 declines to sell shares
    g.process_action(action_helper.get_all_choices()[-1])  # [19:16] Player 4 passes
    g.process_action(
        action_helper.get_all_choices()[0]
    )  # [19:16] Player 1 buys a 10% share of B&M from the IPO for $100
    g.process_action(action_helper.get_all_choices()[-1])  # [19:16] Player 1 declines to sell shares
    g.process_action(
        action_helper.get_all_choices()[1]
    )  # [19:16] Player 2 buys a 10% share of CPR from the IPO for $100
    g.process_action(action_helper.get_all_choices()[-1])  # [19:17] Player 2 declines to sell shares
    g.process_action(
        action_helper.get_all_choices()[1]
    )  # [19:18] Player 3 buys a 10% share of CPR from the IPO for $100
    g.process_action(action_helper.get_all_choices()[-1])  # [19:18] Player 3 declines to sell shares
    g.process_action(action_helper.get_all_choices()[-1])  # [19:18] Player 4 passes
    g.process_action(
        action_helper.get_all_choices()[0]
    )  # [19:18] Player 1 buys a 10% share of B&M from the IPO for $100
    # [19:18] B&M floats
    # [19:18] B&M receives $1000
    g.process_action(action_helper.get_all_choices()[-1])  # [19:18] Player 1 declines to sell shares
    g.process_action(action_helper.get_all_choices()[-4])  # [19:18] Player 2 sells 2 shares of NYNH and receives $200
    # [19:18] NYNH's share price moves down from 80
    g.process_action(
        action_helper.get_all_choices()[1]
    )  # [19:18] Player 2 buys a 10% share of CPR from the IPO for $100
    g.process_action(action_helper.get_all_choices()[-1])
    g.process_action(action_helper.get_all_choices()[-1])  # [19:19] Player 3 passes
    g.process_action(action_helper.get_all_choices()[-1])  # [19:19] Player 4 passes
    g.process_action(action_helper.get_all_choices()[-1])  # [19:19] Player 1 passes
    g.process_action(
        action_helper.get_all_choices()[1]
    )  # [19:19] Player 2 buys a 10% share of CPR from the IPO for $100
    g.process_action(action_helper.get_all_choices()[-1])  # [19:19] Player 2 declines to sell shares
    g.process_action(action_helper.get_all_choices()[-1])  # [19:19] Player 3 passes
    g.process_action(action_helper.get_all_choices()[-1])  # [19:19] Player 4 passes
    g.process_action(action_helper.get_all_choices()[-1])  # [19:19] Player 1 passes
    g.process_action(action_helper.get_all_choices()[-1])  # [19:19] Player 2 passes
    action_helper.print_summary(json_format=True)
    expected_state = {
        "players": {
            "Player 1": {
                "cash": 99,
                "shares": {"PRR": 30, "NYC": 50, "C&O": 30, "B&O": 10, "ERIE": 0, "B&M": 60},
                "companies": [],
            },
            "Player 2": {
                "cash": 151,
                "shares": {"C&O": 60, "NYC": 0, "NYNH": 20, "ERIE": 50, "CPR": 60},
                "companies": [],
            },
            "Player 3": {
                "cash": 21,
                "shares": {"NYC": 30, "NYNH": 60, "B&O": 40, "ERIE": 10, "CPR": 20, "PRR": 10},
                "companies": [],
            },
            "Player 4": {
                "cash": 53,
                "shares": {"PRR": 40, "B&O": 50, "C&O": 10, "NYC": 20, "ERIE": 40, "CPR": 20},
                "companies": [],
            },
        },
        "corporations": {
            "PRR": {"cash": 260, "companies": [], "trains": ["4"], "share_price": 90},
            "NYC": {"cash": 735, "companies": [], "trains": ["3", "3"], "share_price": 65},
            "CPR": {"cash": 1000, "companies": [], "trains": [], "share_price": 100},
            "B&O": {"cash": 380, "companies": [], "trains": ["3", "4"], "share_price": 100},
            "C&O": {"cash": 75, "companies": [], "trains": ["3", "5"], "share_price": 100},
            "ERIE": {"cash": 0, "companies": [], "trains": ["4", "5"], "share_price": 100},
            "NYNH": {"cash": 50, "companies": [], "trains": ["3", "4"], "share_price": 80},
            "B&M": {"cash": 1000, "companies": [], "trains": [], "share_price": 100},
        },
    }
    assert action_helper.get_state() == expected_state, "State does not match expected state after stock round 6"

    # Operating Round 6
    # [19:19] Player 2 operates C&O
    g.process_action(action_helper.get_all_choices()[10])  # [19:21] C&O lays tile #9 with rotation 1 on G9
    # [19:21] C&O skips place a token
    g.process_action(action_helper.get_all_choices()[0])  # [19:21] C&O runs a 5 train for $220: E11-F16-G17-G19-H18
    # [19:21] C&O runs a 3 train for $110: F2-F6-G7
    g.process_action(
        action_helper.get_all_choices()[0]
    )  # [19:21] C&O pays out 33 per share (99 to Player 1, $33 to Player 4)
    # [19:21] C&O's share price moves right from 111
    # [19:21] C&O skips buy trains
    # [19:21] C&O skips buy companies

    # [19:21] Player 4 operates B&O
    g.process_action(action_helper.get_all_choices()[28])  # [19:22] B&O lays tile #46 with rotation 5 on G5
    g.process_action(action_helper.get_all_choices()[-1])  # [19:22] B&O passes place a token
    g.process_action(action_helper.get_all_choices()[0])  # [19:22] B&O runs a 4 train for $170: K13-J14-I15-H18
    # [19:22] B&O runs a 3 train for $160: I15-H10-F2
    g.process_action(action_helper.get_all_choices()[1])  # [19:22] B&O withholds $330
    # [19:22] B&O's share price moves left from 90
    # [19:22] B&O skips buy trains
    # [19:22] B&O skips buy companies

    # [19:22] Player 1 operates B&M
    # [19:22] B&M places a token on E23
    g.process_action(action_helper.get_all_choices()[11])  # [19:24] B&M lays tile #45 with rotation 4 on E21
    g.process_action(action_helper.get_all_choices()[-1])  # [19:24] B&M passes place a token
    # [19:24] B&M skips run routes
    # [19:24] B&M does not run
    # [19:24] B&M's share price moves left from 90
    g.process_action(action_helper.get_all_choices()[0])  # [19:24] B&M buys a 5 train for $450 from The Depot
    g.process_action(action_helper.get_all_choices()[-1])  # [19:24] B&M passes buy trains
    # [19:24] B&M skips buy companies

    # [19:24] Player 2 operates CPR
    # [19:24] CPR places a token on A19
    g.process_action(
        action_helper.get_all_choices()[-3]
    )  # [19:24] CPR spends $80 and lays tile #9 with rotation 0 on B18
    # [19:24] CPR skips place a token
    # [19:24] CPR skips run routes
    # [19:24] CPR does not run
    # [19:24] CPR's share price moves left from 90
    g.process_action(action_helper.get_all_choices()[0])  # [19:24] CPR buys a 6 train for $630 from The Depot
    # [19:24] -- Phase 6 (Operating Rounds: 3 | Train Limit: 2 | Available Tiles: Yellow, Green, Brown) --
    # [19:24] -- Event: 3 trains rust ( C&O x1, NYC x2, NYNH x1, B&O x1) --
    g.process_action(action_helper.get_all_choices()[-1])  # [19:25] CPR passes buy trains
    # [19:25] CPR skips buy companies

    # [19:25] Player 2 operates ERIE
    g.process_action(action_helper.get_all_choices()[6])  # [19:25] ERIE lays tile #25 with rotation 2 on F12
    # [19:25] ERIE skips place a token
    g.process_action(action_helper.get_all_choices()[0])  # [19:25] ERIE runs a 5 train for $80: E11-H10
    g.process_action(action_helper.get_all_choices()[1])  # [19:25] ERIE withholds $80
    # [19:25] ERIE's share price moves left from 90
    # [19:25] ERIE skips buy trains
    # [19:25] ERIE skips buy companies

    # [19:25] Player 4 operates PRR
    g.process_action(action_helper.get_all_choices()[8])  # [19:26] PRR lays tile #23 with rotation 1 on H14
    # [19:26] PRR skips place a token
    g.process_action(action_helper.get_all_choices()[0])  # [19:26] PRR runs a 4 train for $170: F2-H10-H12-I15
    g.process_action(action_helper.get_all_choices()[1])  # [19:27] PRR withholds $170
    # [19:27] PRR's share price moves left from 82
    g.process_action(action_helper.get_all_choices()[-1])  # [19:27] PRR passes buy trains
    # [19:27] PRR skips buy companies

    # [19:27] Player 3 operates NYNH
    g.process_action(action_helper.get_all_choices()[20])  # [19:27] NYNH lays tile #8 with rotation 4 on D12
    # [19:27] NYNH skips place a token
    g.process_action(action_helper.get_all_choices()[0])  # [19:27] NYNH runs a 4 train for $190: G19-E19-D14-E11
    g.process_action(action_helper.get_all_choices()[1])  # [19:27] NYNH withholds $190
    # [19:27] NYNH's share price moves left from 75
    # [19:27] NYNH passes buy trains
    # [19:27] NYNH skips buy companies

    # [19:27] Player 1 operates NYC
    g.process_action(action_helper.get_all_choices()[-10])  # [19:27] NYC lays tile #40 with rotation 0 on F12
    g.process_action(action_helper.get_all_choices()[-1])  # [19:27] NYC passes place a token
    # [19:27] NYC skips run routes
    # [19:27] NYC does not run
    # [19:27] NYC's share price moves left from 58
    g.process_action(action_helper.get_all_choices()[0])  # [19:27] NYC buys a 6 train for $630 from The Depot
    g.process_action(action_helper.get_all_choices()[-1])  # [19:28] NYC passes buy trains

    # [19:28] NYC skips buy companies
    # [19:28] Player 2 operates C&O
    g.process_action(action_helper.get_all_choices()[24])  # [20:27] C&O lays tile #23 with rotation 3 on G11
    # [20:27] C&O skips place a token
    g.process_action(action_helper.get_all_choices()[0])  # [20:27] C&O runs a 5 train for $220: E11-F16-G17-G19-H18
    g.process_action(
        action_helper.get_all_choices()[0]
    )  # [20:27] C&O pays out 22 per share (66 to Player 1, $22 to Player 4)
    # [20:27] C&O's share price moves right from 125
    g.process_action(action_helper.get_all_choices()[-1])  # [20:27] C&O passes buy trains
    # [20:27] C&O skips buy companies

    # [20:27] Player 4 operates B&O
    g.process_action(action_helper.get_all_choices()[42])  # [20:27] B&O lays tile #43 with rotation 0 on G11
    g.process_action(action_helper.get_all_choices()[-1])  # [20:28] B&O passes place a token
    g.process_action(action_helper.get_all_choices()[0])  # [20:28] B&O runs a 4 train for $210: F2-H10-I15-H18
    g.process_action(action_helper.get_all_choices()[1])  # [20:28] B&O withholds $210
    # [20:28] B&O's share price moves left from 82
    g.process_action(
        action_helper.get_all_choices()[-2]
    )  # [20:28] B&O exchanges a 4 for a D train for $800 from The Depot
    # [20:28] -- Phase D (Operating Rounds: 3 | Train Limit: 2 | Available Tiles: Yellow, Green, Brown) --
    # [20:28] -- Event: 4 trains rust ( The Depot x1, PRR x1, NYNH x1, ERIE x1) --
    # [20:28] B&O passes buy trains
    # [20:28] B&O skips buy companies

    # [20:28] Player 1 operates B&M
    g.process_action(action_helper.get_all_choices()[11])  # [20:28] B&M lays tile #8 with rotation 5 on D20
    g.process_action(action_helper.get_all_choices()[-1])  # [20:28] B&M passes place a token
    g.process_action(action_helper.get_all_choices()[0])  # [20:28] B&M runs a 5 train for $170: E23-F24-F22-F20-G19
    g.process_action(action_helper.get_all_choices()[0])  # [20:28] B&M pays out 17 per share ($102 to Player 1)
    # [20:28] B&M's share price moves right from 100
    g.process_action(action_helper.get_all_choices()[-1])  # [20:28] B&M passes buy trains
    # [20:28] B&M skips buy companies

    # [20:28] Player 2 operates CPR
    g.process_action(
        action_helper.get_all_choices()[-3]
    )  # [20:28] CPR spends $120 and lays tile #9 with rotation 0 on C17
    # [20:28] CPR skips place a token
    g.process_action(action_helper.get_all_choices()[0])  # [20:29] CPR runs a 6 train for $50: A19-B20
    g.process_action(
        action_helper.get_all_choices()[0]
    )  # [20:29] CPR pays out 5 per share (10 to Player 3, $10 to Player 4)
    # [20:29] CPR's share price moves right from 100
    g.process_action(action_helper.get_all_choices()[-1])  # [20:29] CPR passes buy trains
    # [20:29] CPR skips buy companies

    # [20:29] Player 2 operates ERIE
    g.process_action(
        action_helper.get_all_choices()[0]
    )  # [20:29] ERIE spends $80 and lays tile #59 with rotation 3 on D10 (Hamilton & Toronto)
    # [20:29] ERIE skips place a token
    g.process_action(action_helper.get_all_choices()[0])  # [20:29] ERIE runs a 5 train for $130: D10-E11-G7-F6
    g.process_action(
        action_helper.get_all_choices()[0]
    )  # [20:29] ERIE pays out 13 per share (52 to Player 4, $13 to Player 3)
    # [20:29] ERIE's share price moves right from 100
    # [20:29] ERIE skips buy trains
    # [20:29] ERIE skips buy companies

    # [20:29] Player 4 operates PRR
    g.process_action(
        action_helper.get_all_choices()[11]
    )  # [20:29] PRR lays tile #57 with rotation 1 on H16 (Lancaster)
    g.process_action(action_helper.get_all_choices()[-1])  # [20:29] PRR passes place a token
    # [20:29] PRR skips run routes
    # [20:29] PRR does not run
    # [20:29] PRR's share price moves left from 76
    g.process_action(action_helper.get_all_choices()[440])  # [20:30] Player 4 sells 4 shares of ERIE and receives $400
    # [20:30] ERIE's share price moves down from 71
    g.process_action(
        action_helper.get_all_choices()[433]
    )  # [20:30] Player 4 sells a 10% share of CPR and receives $100
    # [20:30] CPR's share price moves down from 90
    g.process_action(action_helper.get_all_choices()[0])  # [20:30] Player 4 contributes $670
    # [20:30] PRR buys a D train for $1100 from The Depot
    # [20:30] PRR skips buy companies

    # [20:30] Player 3 operates NYNH
    g.process_action(action_helper.get_all_choices()[37])  # [20:30] NYNH lays tile #58 with rotation 3 on F10 (Erie)
    # [20:30] NYNH skips place a token
    # [20:30] NYNH skips run routes
    # [20:30] NYNH does not run
    # [20:30] NYNH's share price moves left from 71

    g.process_action(action_helper.get_all_choices()[9])  # [20:31] Player 3 sells 4 shares of B&O and receives $328
    # [20:31] B&O's share price moves down from 62
    g.process_action(action_helper.get_all_choices()[5])  # [20:31] Player 3 sells 2 shares of CPR and receives $180
    # [20:31] CPR's share price moves down from 76
    g.process_action(action_helper.get_all_choices()[4])  # [20:31] Player 3 sells a 10% share of ERIE and receives $71
    # [20:31] ERIE's share price moves down from 67
    g.process_action(action_helper.get_all_choices()[3])  # [20:31] Player 3 sells 3 shares of NYC and receives $174
    # [20:31] NYC's share price moves down from 40
    g.process_action(action_helper.get_all_choices()[0])  # [20:31] Player 3 sells a 10% share of PRR and receives $76
    # [20:31] PRR's share price moves down from 71
    g.process_action(action_helper.get_all_choices()[0])  # [20:31] Player 3 contributes $860
    # [20:31] NYNH buys a D train for $1100 from The Depot
    # [20:31] NYNH skips buy companies

    # [20:31] Player 1 operates NYC
    g.process_action(action_helper.get_all_choices()[56])  # [20:31] NYC lays tile #24 with rotation 4 on G9
    g.process_action(
        action_helper.get_all_choices()[3]
    )  # [20:32] NYC places a token on E11 (Dunkirk & Buffalo) for $100
    g.process_action(action_helper.get_all_choices()[0])  # [20:32] NYC runs a 6 train for $280: I15-H18-G19-G17-F16-E11
    g.process_action(
        action_helper.get_all_choices()[0]
    )  # [20:32] NYC pays out 28 per share (84 to NYC, $56 to Player 4)
    # [20:32] NYC's share price moves right from 50
    g.process_action(action_helper.get_all_choices()[-1])  # [20:32] NYC passes buy trains
    # [20:32] NYC skips buy companies

    # [20:32] Player 2 operates C&O
    g.process_action(action_helper.get_all_choices()[52])  # [21:35] C&O lays tile #24 with rotation 1 on H8
    # [21:35] C&O skips place a token
    # router.debug = True
    # router.verbose = True
    g.process_action(action_helper.get_all_choices()[0])  # [21:36] C&O runs a 5 train for $240: F2-F16-G17-G19-H18
    g.process_action(
        action_helper.get_all_choices()[0]
    )  # [21:36] C&O pays out 24 per share (72 to Player 1, $24 to Player 4)
    # [21:36] C&O's share price moves right from 140
    g.process_action(action_helper.get_all_choices()[-1])  # [21:36] C&O passes buy trains
    # [21:36] C&O skips buy companies

    # [21:36] Player 1 operates B&M
    g.process_action(action_helper.get_all_choices()[17])  # [21:36] B&M lays tile #23 with rotation 1 on D18
    g.process_action(action_helper.get_all_choices()[0])  # [21:36] B&M places a token on D14 (Rochester) for $40
    g.process_action(action_helper.get_all_choices()[0])  # [21:36] B&M runs a 5 train for $170: E23-F24-F22-F20-G19
    g.process_action(action_helper.get_all_choices()[0])  # [21:36] B&M pays out 17 per share ($102 to Player 1)
    # [21:36] B&M's share price moves right from 112
    g.process_action(action_helper.get_all_choices()[-1])  # [21:36] B&M passes buy trains
    # [21:36] B&M skips buy companies

    # [21:36] Player 2 operates CPR
    g.process_action(action_helper.get_all_choices()[27])  # [21:36] CPR lays tile #20 with rotation 0 on D16
    # [21:36] CPR skips place a token
    g.process_action(action_helper.get_all_choices()[0])  # [21:36] CPR runs a 6 train for $50: A19-B20
    g.process_action(action_helper.get_all_choices()[0])  # [21:36] CPR pays out 5 per share (15 to CPR, $5 to Player 4)
    # [21:36] CPR's share price moves right from 82
    g.process_action(action_helper.get_all_choices()[-1])  # [21:36] CPR passes buy trains
    # [21:36] CPR skips buy companies

    # [21:36] Player 3 operates NYNH
    g.process_action(action_helper.get_all_choices()[9])  # [21:37] NYNH lays tile #47 with rotation 0 on D16
    # [21:37] NYNH skips place a token
    g.process_action(
        action_helper.get_all_choices()[0]
    )  # [21:37] NYNH runs a D train for $250: E23-F24-F22-F20-G19-E19-F20-F16
    g.process_action(
        action_helper.get_all_choices()[0]
    )  # [21:37] NYNH pays out 25 per share (50 to Player 2, $50 to NYNH)
    # [21:37] NYNH's share price moves right from 75
    # [21:37] NYNH passes buy trains
    # [21:37] NYNH skips buy companies

    # [21:37] Player 4 operates PRR
    g.process_action(action_helper.get_all_choices()[35])  # [21:37] PRR lays tile #24 with rotation 1 on F14
    # [21:37] PRR skips place a token
    g.process_action(action_helper.get_all_choices()[0])  # [21:37] PRR runs a D train for $180: F2-H10-H12-H16-H18
    g.process_action(
        action_helper.get_all_choices()[0]
    )  # [21:37] PRR pays out 18 per share (54 to Player 1, $54 to PRR)
    # [21:37] PRR's share price moves right from 76
    g.process_action(action_helper.get_all_choices()[-1])  # [21:37] PRR passes buy trains
    # [21:37] PRR skips buy companies

    # [21:37] Player 2 operates ERIE
    g.process_action(
        action_helper.get_all_choices()[7]
    )  # [21:38] ERIE lays tile #67 with rotation 3 on D10 (Hamilton & Toronto)
    # [21:38] ERIE skips place a token
    g.process_action(action_helper.get_all_choices()[0])  # [21:38] ERIE runs a 5 train for $170: F2-E11-D10
    g.process_action(action_helper.get_all_choices()[0])  # [21:38] ERIE pays out 17 per share (85 to ERIE)
    # [21:38] ERIE's share price moves right from 71
    g.process_action(action_helper.get_all_choices()[-1])  # [21:38] ERIE passes buy trains
    # [21:38] ERIE skips buy companies

    # [21:38] Player 4 operates B&O
    g.process_action(action_helper.get_all_choices()[36])  # [21:38] B&O lays tile #9 with rotation 0 on E15
    g.process_action(action_helper.get_all_choices()[-1])  # [21:39] B&O passes place a token
    g.process_action(action_helper.get_all_choices()[0])  # [21:39] B&O runs a D train for $220: F2-H10-I15-J14-K13
    g.process_action(
        action_helper.get_all_choices()[0]
    )  # [21:39] B&O pays out 22 per share (88 to B&O, $22 to Player 1)
    # [21:39] B&O's share price moves right from 67
    g.process_action(action_helper.get_all_choices()[-1])  # [21:39] B&O passes buy trains
    # [21:39] B&O skips buy companies

    # [21:39] Player 1 operates NYC
    g.process_action(action_helper.get_all_choices()[20])  # [21:39] NYC lays tile #28 with rotation 0 on D12
    # [21:39] NYC skips place a token
    g.process_action(action_helper.get_all_choices()[0])  # [21:39] NYC runs a 6 train for $300: F2-F16-G17-G19-H18-I15
    g.process_action(
        action_helper.get_all_choices()[0]
    )  # [21:39] NYC pays out 30 per share (90 to NYC, $60 to Player 4)
    # [21:39] NYC's share price moves right from 60
    g.process_action(action_helper.get_all_choices()[-1])  # [21:39] NYC passes buy trains
    # [21:39] NYC skips buy companies
    action_helper.print_summary(json_format=True)
    expected_state = {
        "players": {
            "Player 1": {
                "cash": 906,
                "shares": {"PRR": 30, "NYC": 50, "C&O": 30, "B&O": 10, "ERIE": 0, "B&M": 60},
                "companies": [],
            },
            "Player 2": {
                "cash": 885,
                "shares": {"C&O": 60, "NYC": 0, "NYNH": 20, "ERIE": 50, "CPR": 60},
                "companies": [],
            },
            "Player 3": {
                "cash": 163,
                "shares": {"NYC": 0, "NYNH": 60, "B&O": 0, "ERIE": 0, "CPR": 0, "PRR": 0},
                "companies": [],
            },
            "Player 4": {
                "cash": 327,
                "shares": {"PRR": 40, "B&O": 50, "C&O": 10, "NYC": 20, "ERIE": 0, "CPR": 10},
                "companies": [],
            },
        },
        "corporations": {
            "PRR": {"cash": 54, "companies": [], "trains": ["D"], "share_price": 76},
            "NYC": {"cash": 179, "companies": [], "trains": ["6"], "share_price": 60},
            "CPR": {"cash": 185, "companies": [], "trains": ["6"], "share_price": 82},
            "B&O": {"cash": 208, "companies": [], "trains": ["D"], "share_price": 67},
            "C&O": {"cash": 75, "companies": [], "trains": ["5"], "share_price": 140},
            "ERIE": {"cash": 85, "companies": [], "trains": ["5"], "share_price": 71},
            "NYNH": {"cash": 50, "companies": [], "trains": ["D"], "share_price": 75},
            "B&M": {"cash": 510, "companies": [], "trains": ["5"], "share_price": 112},
        },
    }
    assert action_helper.get_state() == expected_state, "State does not match expected state after operating round 6"

    # Stock Round 7
    g.process_action(
        action_helper.get_all_choices()[5]
    )  # [21:46] Player 3 buys a 10% share of NYC from the market for $60
    g.process_action(action_helper.get_all_choices()[-1])  # [21:46] Player 3 declines to sell shares

    g.process_action(
        action_helper.get_all_choices()[6]
    )  # [21:46] Player 4 buys a 10% share of NYC from the market for $60
    g.process_action(action_helper.get_all_choices()[-1])  # [21:47] Player 4 declines to sell shares

    g.process_action(
        action_helper.get_all_choices()[5]
    )  # [21:47] Player 1 buys a 10% share of NYC from the market for $60
    g.process_action(action_helper.get_all_choices()[-1])  # [21:47] Player 1 declines to sell shares

    g.process_action(action_helper.get_all_choices()[-1])  # [21:47] Player 2 passes

    g.process_action(
        action_helper.get_all_choices()[4]
    )  # [21:47] Player 3 buys a 10% share of B&O from the market for $67
    g.process_action(action_helper.get_all_choices()[-1])  # [21:47] Player 3 declines to sell shares

    g.process_action(
        action_helper.get_all_choices()[5]
    )  # [21:47] Player 4 buys a 10% share of B&O from the market for $67
    g.process_action(action_helper.get_all_choices()[-1])  # [21:47] Player 4 declines to sell shares

    g.process_action(
        action_helper.get_all_choices()[4]
    )  # [21:47] Player 1 buys a 10% share of B&O from the market for $67
    g.process_action(action_helper.get_all_choices()[-1])  # [21:47] Player 1 declines to sell shares

    g.process_action(action_helper.get_all_choices()[-1])  # [21:48] Player 2 passes

    g.process_action(action_helper.get_all_choices()[-1])  # [21:48] Player 3 passes

    g.process_action(
        action_helper.get_all_choices()[2]
    )  # [21:48] Player 4 buys a 10% share of PRR from the market for $76
    g.process_action(action_helper.get_all_choices()[-1])  # [21:48] Player 4 declines to sell shares

    g.process_action(
        action_helper.get_all_choices()[4]
    )  # [21:48] Player 1 buys a 10% share of B&O from the market for $67
    g.process_action(action_helper.get_all_choices()[-1])  # [21:48] Player 1 declines to sell shares

    g.process_action(action_helper.get_all_choices()[-1])  # [21:48] Player 2 passes

    g.process_action(action_helper.get_all_choices()[-1])  # [21:48] Player 3 passes

    g.process_action(
        action_helper.get_all_choices()[2]
    )  # [21:48] Player 4 buys a 10% share of PRR from the market for $76
    g.process_action(action_helper.get_all_choices()[-1])  # [21:48] Player 4 declines to sell shares

    g.process_action(
        action_helper.get_all_choices()[3]
    )  # [21:48] Player 1 buys a 10% share of ERIE from the market for $71
    g.process_action(action_helper.get_all_choices()[-1])  # [21:48] Player 1 declines to sell shares

    g.process_action(action_helper.get_all_choices()[-1])  # [21:48] Player 2 passes

    g.process_action(action_helper.get_all_choices()[-1])  # [21:48] Player 3 passes

    g.process_action(action_helper.get_all_choices()[-1])  # [21:48] Player 4 passes

    g.process_action(
        action_helper.get_all_choices()[3]
    )  # [21:49] Player 1 buys a 10% share of ERIE from the market for $71
    g.process_action(action_helper.get_all_choices()[-1])  # [21:49] Player 1 declines to sell shares
    g.process_action(action_helper.get_all_choices()[-1])  # [21:49] Player 2 passes
    g.process_action(action_helper.get_all_choices()[-1])  # [21:49] Player 3 passes
    g.process_action(action_helper.get_all_choices()[-1])  # [21:49] Player 4 passes
    g.process_action(action_helper.get_all_choices()[-1])  # [21:49] Player 1 passes
    action_helper.print_summary(json_format=True)
    expected_state = {
        "players": {
            "Player 1": {
                "cash": 570,
                "shares": {"PRR": 30, "NYC": 60, "C&O": 30, "B&O": 30, "ERIE": 20, "B&M": 60},
                "companies": [],
            },
            "Player 2": {
                "cash": 885,
                "shares": {"C&O": 60, "NYC": 0, "NYNH": 20, "ERIE": 50, "CPR": 60},
                "companies": [],
            },
            "Player 3": {
                "cash": 36,
                "shares": {"NYC": 10, "NYNH": 60, "B&O": 10, "ERIE": 0, "CPR": 0, "PRR": 0},
                "companies": [],
            },
            "Player 4": {
                "cash": 48,
                "shares": {"PRR": 60, "B&O": 60, "C&O": 10, "NYC": 30, "ERIE": 0, "CPR": 10},
                "companies": [],
            },
        },
        "corporations": {
            "PRR": {"cash": 54, "companies": [], "trains": ["D"], "share_price": 76},
            "NYC": {"cash": 179, "companies": [], "trains": ["6"], "share_price": 67},
            "CPR": {"cash": 185, "companies": [], "trains": ["6"], "share_price": 82},
            "B&O": {"cash": 208, "companies": [], "trains": ["D"], "share_price": 71},
            "C&O": {"cash": 75, "companies": [], "trains": ["5"], "share_price": 160},
            "ERIE": {"cash": 85, "companies": [], "trains": ["5"], "share_price": 71},
            "NYNH": {"cash": 50, "companies": [], "trains": ["D"], "share_price": 75},
            "B&M": {"cash": 510, "companies": [], "trains": ["5"], "share_price": 112},
        },
    }
    assert action_helper.get_state() == expected_state, "State does not match expected state after stock round 7"

    # Rest of game
    # pass until game over
    while not g.finished:
        g.process_action(action_helper.get_all_choices()[-1])
    action_helper.print_summary(json_format=True)
    expected_state = {
        "players": {
            "Player 1": {
                "cash": 570,
                "shares": {"PRR": 30, "NYC": 60, "C&O": 30, "B&O": 30, "ERIE": 20, "B&M": 60},
                "companies": [],
            },
            "Player 2": {
                "cash": 885,
                "shares": {"C&O": 60, "NYC": 0, "NYNH": 20, "ERIE": 50, "CPR": 60},
                "companies": [],
            },
            "Player 3": {
                "cash": 36,
                "shares": {"NYC": 10, "NYNH": 60, "B&O": 10, "ERIE": 0, "CPR": 0, "PRR": 0},
                "companies": [],
            },
            "Player 4": {
                "cash": 48,
                "shares": {"PRR": 60, "B&O": 60, "C&O": 10, "NYC": 30, "ERIE": 0, "CPR": 10},
                "companies": [],
            },
        },
        "corporations": {
            "PRR": {"cash": 1134, "companies": [], "trains": ["D"], "share_price": 41},
            "NYC": {"cash": 1979, "companies": [], "trains": ["6"], "share_price": 18},
            "CPR": {"cash": 905, "companies": [], "trains": ["6"], "share_price": 48},
            "B&O": {"cash": 1528, "companies": [], "trains": ["D"], "share_price": 39},
            "C&O": {"cash": 1515, "companies": [], "trains": ["5"], "share_price": 90},
            "ERIE": {"cash": 1105, "companies": [], "trains": ["5"], "share_price": 34},
            "NYNH": {"cash": 1730, "companies": [], "trains": ["D"], "share_price": 42},
            "B&M": {"cash": 1530, "companies": [], "trains": ["5"], "share_price": 67},
        },
    }
    assert action_helper.get_state() == expected_state, "State does not match expected state at the end of the game"
    assert g.finished == True, "Game is not finished"
    player_results = {player.name: player.value for player in g.players}
    expected_results = {"Player 1": 1658, "Player 2": 1967, "Player 3": 345, "Player 4": 720}
    assert player_results == expected_results, "Player results do not match expected results"
    assert next(iter(g.result().items()))[0] == "2", "Player 2 should be the winner"


def test_1830_from_import():
    with open("tests/test_games/game.json", "r") as f:
        browser_game = f.read()
    g2 = BaseGame.load(browser_game)
    assert g2.finished == True
    player_results = {player.name: player.value for player in g2.players}
    assert player_results == {
        "Player 1": 7576,
        "Player 2": 7453,
        "Player 3": 4074,
        "Player 4": 5855,
    }, "Player results do not match expected results"
    assert str(next(iter(g2.result().items()))[0]) == "0", "Player 1 should be the winner"


def test_1830_manual_bankrupcy():
    game_map = GameMap()
    game = game_map.game_by_title("1830")
    g = game({"1": "Player 1", "2": "Player 2", "3": "Player 3", "4": "Player 4"})
    action_helper = ActionHelper(g)
    # action_helper.print_enabled = True

    # Auction
    g.process_action(
        action_helper.get_all_choices()[-2]
    )  # [20:39] -- Phase 2 (Operating Rounds: 1 | Train Limit: 4 | Available Tiles: Yellow) --
    # [20:39] Player 1 bids $600 for Baltimore & Ohio
    g.process_action(action_helper.get_all_choices()[0])  # [20:39] Player 2 buys Schuylkill Valley for $20
    g.process_action(action_helper.get_all_choices()[0])  # [20:39] Player 3 buys Champlain & St.Lawrence for $40
    g.process_action(action_helper.get_all_choices()[0])  # [20:39] Player 4 buys Delaware & Hudson for $70
    g.process_action(action_helper.get_all_choices()[0])  # [20:39] Player 1 passes bidding
    g.process_action(action_helper.get_all_choices()[0])  # [20:39] Player 2 buys Mohawk & Hudson for $110
    g.process_action(action_helper.get_all_choices()[0])  # [20:39] Player 3 buys Camden & Amboy for $160
    # [20:39] Player 3 receives a 10% share of PRR
    # [20:39] Player 1 wins the auction for Baltimore & Ohio with the only bid of $600
    g.process_action(action_helper.get_all_choices()[-1])  # [20:39] Player 1 pars B&O at $67
    # [20:39] Player 1 receives a 20% share of B&O
    # [20:39] Player 1 becomes the president of B&O
    # [20:39] Player 4 has priority deal
    action_helper.print_summary(json_format=True)
    expected_state = {
        "players": {
            "Player 1": {"cash": 0, "shares": {"B&O": 20}, "companies": ["BO"]},
            "Player 2": {"cash": 470, "shares": {}, "companies": ["SV", "MH"]},
            "Player 3": {"cash": 400, "shares": {"PRR": 10}, "companies": ["CS", "CA"]},
            "Player 4": {"cash": 530, "shares": {}, "companies": ["DH"]},
        },
        "corporations": {
            "PRR": {"cash": 0, "companies": [], "trains": [], "share_price": None},
            "NYC": {"cash": 0, "companies": [], "trains": [], "share_price": None},
            "CPR": {"cash": 0, "companies": [], "trains": [], "share_price": None},
            "B&O": {"cash": 0, "companies": [], "trains": [], "share_price": 67},
            "C&O": {"cash": 0, "companies": [], "trains": [], "share_price": None},
            "ERIE": {"cash": 0, "companies": [], "trains": [], "share_price": None},
            "NYNH": {"cash": 0, "companies": [], "trains": [], "share_price": None},
            "B&M": {"cash": 0, "companies": [], "trains": [], "share_price": None},
        },
    }
    assert action_helper.get_state() == expected_state, "State does not match expected state after auction"
    # [20:39] -- Stock Round 1 --
    g.process_action(
        action_helper.get_all_choices()[0]
    )  # [20:39] Player 4 buys a 10% share of B&O from the IPO for $67
    # [20:39] Player 1 has no valid actions and passes
    g.process_action(
        action_helper.get_all_choices()[0]
    )  # [21:13] Player 2 buys a 10% share of B&O from the IPO for $67
    g.process_action(
        action_helper.get_all_choices()[0]
    )  # [21:13] Player 3 buys a 10% share of B&O from the IPO for $67
    g.process_action(
        action_helper.get_all_choices()[0]
    )  # [21:13] Player 4 buys a 10% share of B&O from the IPO for $67
    # [21:13] B&O floats
    # [21:13] B&O receives $670
    # [21:13] Player 1 has no valid actions and passes
    g.process_action(action_helper.get_all_choices()[-2])  # [21:13] Player 2 pars PRR at $67
    # [21:13] Player 2 buys a 20% share of PRR from the IPO for $134
    # [21:13] Player 2 becomes the president of PRR
    g.process_action(
        action_helper.get_all_choices()[1]
    )  # [21:13] Player 3 buys a 10% share of PRR from the IPO for $67
    g.process_action(
        action_helper.get_all_choices()[1]
    )  # [21:13] Player 4 buys a 10% share of PRR from the IPO for $67
    # [21:13] Player 1 has no valid actions and passes
    g.process_action(
        action_helper.get_all_choices()[1]
    )  # [21:13] Player 2 buys a 10% share of PRR from the IPO for $67
    # [21:13] PRR floats
    # [21:13] PRR receives $670
    g.process_action(
        action_helper.get_all_choices()[1]
    )  # [21:13] Player 3 buys a 10% share of PRR from the IPO for $67
    g.process_action(
        action_helper.get_all_choices()[0]
    )  # [21:14] Player 4 buys a 10% share of B&O from the IPO for $67
    # [21:14] Player 4 becomes the president of B&O
    # [21:14] Player 1 has no valid actions and passes
    g.process_action(action_helper.get_all_choices()[-1])  # [21:14] Player 2 passes
    g.process_action(action_helper.get_all_choices()[-1])  # [21:14] Player 3 passes
    g.process_action(
        action_helper.get_all_choices()[1]
    )  # [21:14] Player 4 buys a 10% share of PRR from the IPO for $67
    # [21:14] Player 1 has no valid actions and passes
    g.process_action(action_helper.get_all_choices()[-1])  # [21:14] Player 2 passes
    g.process_action(action_helper.get_all_choices()[-1])  # [21:14] Player 3 passes
    g.process_action(
        action_helper.get_all_choices()[1]
    )  # [21:14] Player 4 buys a 10% share of PRR from the IPO for $67
    # [21:14] Player 1 has no valid actions and passes
    g.process_action(action_helper.get_all_choices()[-1])  # [21:14] Player 2 passes
    g.process_action(action_helper.get_all_choices()[-1])  # [21:14] Player 3 passes
    g.process_action(
        action_helper.get_all_choices()[1]
    )  # [21:14] Player 4 buys a 10% share of PRR from the IPO for $67
    # [21:14] Player 4 becomes the president of PRR
    # [21:14] Player 1 has no valid actions and passes
    g.process_action(action_helper.get_all_choices()[-1])  # [21:15] Player 2 passes
    g.process_action(action_helper.get_all_choices()[-1])  # [21:15] Player 3 passes
    # [21:15] Player 4 has no valid actions and passes
    # [21:15] PRR's share price moves up from 71
    # [21:15] Player 1 has priority deal
    action_helper.print_summary(json_format=True)
    expected_state = {
        "players": {
            "Player 1": {"cash": 30, "shares": {"B&O": 20}, "companies": ["BO"]},
            "Player 2": {"cash": 227, "shares": {"B&O": 10, "PRR": 30}, "companies": ["SV", "MH"]},
            "Player 3": {"cash": 234, "shares": {"PRR": 30, "B&O": 10}, "companies": ["CS", "CA"]},
            "Player 4": {"cash": 76, "shares": {"B&O": 30, "PRR": 40}, "companies": ["DH"]},
        },
        "corporations": {
            "PRR": {"cash": 670, "companies": [], "trains": [], "share_price": 71},
            "NYC": {"cash": 0, "companies": [], "trains": [], "share_price": None},
            "CPR": {"cash": 0, "companies": [], "trains": [], "share_price": None},
            "B&O": {"cash": 670, "companies": [], "trains": [], "share_price": 67},
            "C&O": {"cash": 0, "companies": [], "trains": [], "share_price": None},
            "ERIE": {"cash": 0, "companies": [], "trains": [], "share_price": None},
            "NYNH": {"cash": 0, "companies": [], "trains": [], "share_price": None},
            "B&M": {"cash": 0, "companies": [], "trains": [], "share_price": None},
        },
    }
    assert action_helper.get_state() == expected_state, "State does not match expected state after stock round 1"

    # [21:15] -- Operating Round 1.1 (of 1) --
    # [21:15] Player 1 collects $30 from Baltimore & Ohio
    # [21:15] Player 2 collects $5 from Schuylkill Valley
    # [21:15] Player 2 collects $20 from Mohawk & Hudson
    # [21:15] Player 3 collects $10 from Champlain & St.Lawrence
    # [21:15] Player 3 collects $25 from Camden & Amboy
    # [21:15] Player 4 collects $15 from Delaware & Hudson
    # [21:15] Player 4 operates PRR
    # [21:15] PRR places a token on H12
    g.process_action(
        action_helper.get_all_choices()[0]
    )  # [21:16] PRR lays tile #57 with rotation 1 on H10 (Pittsburgh)
    g.process_action(action_helper.get_all_choices()[-1])  # [21:16] PRR passes place a token
    # [21:16] PRR skips run routes
    # [21:16] PRR does not run
    # [21:16] PRR's share price moves left from 67
    g.process_action(action_helper.get_all_choices()[0])  # [21:16] PRR buys a 2 train for $80 from The Depot
    g.process_action(action_helper.get_all_choices()[0])  # [21:16] PRR buys a 2 train for $80 from The Depot
    g.process_action(action_helper.get_all_choices()[-1])  # [21:17] PRR passes buy trains
    # [21:17] PRR skips buy companies
    # [21:17] Player 4 operates B&O
    # [21:17] B&O places a token on I15
    g.process_action(
        action_helper.get_all_choices()[-3]
    )  # [21:17] B&O spends $80 and lays tile #57 with rotation 0 on J14 (Washington)
    g.process_action(action_helper.get_all_choices()[-1])  # [21:17] B&O passes place a token
    # [21:17] B&O skips run routes
    # [21:17] B&O does not run
    # [21:17] B&O's share price moves left from 65
    g.process_action(action_helper.get_all_choices()[-1])  # [21:22] B&O buys a 2 train for $590 from PRR
    # [21:22] Baltimore & Ohio closes
    # [21:22] B&O skips buy companies
    action_helper.print_summary(json_format=True)
    expected_state = {
        "players": {
            "Player 1": {"cash": 30, "shares": {"B&O": 20}, "companies": []},
            "Player 2": {"cash": 227, "shares": {"B&O": 10, "PRR": 30}, "companies": ["SV", "MH"]},
            "Player 3": {"cash": 234, "shares": {"PRR": 30, "B&O": 10}, "companies": ["CS", "CA"]},
            "Player 4": {"cash": 76, "shares": {"B&O": 30, "PRR": 40}, "companies": ["DH"]},
        },
        "corporations": {
            "PRR": {"cash": 1100, "companies": [], "trains": ["2"], "share_price": 67},
            "NYC": {"cash": 0, "companies": [], "trains": [], "share_price": None},
            "CPR": {"cash": 0, "companies": [], "trains": [], "share_price": None},
            "B&O": {"cash": 0, "companies": [], "trains": ["2"], "share_price": 65},
            "C&O": {"cash": 0, "companies": [], "trains": [], "share_price": None},
            "ERIE": {"cash": 0, "companies": [], "trains": [], "share_price": None},
            "NYNH": {"cash": 0, "companies": [], "trains": [], "share_price": None},
            "B&M": {"cash": 0, "companies": [], "trains": [], "share_price": None},
        },
    }
    assert action_helper.get_state() == expected_state, "State does not match expected state after operating round 1"

    # [21:22] -- Stock Round 2 --
    g.process_action(action_helper.get_all_choices()[-1])  # [21:23] Player 1 passes
    # [23:26] Player 2 pars NYC at $67
    g.process_action(
        action_helper.get_all_choices()[31]
    )  # [23:26] Player 2 buys a 20% share of NYC from the IPO for $134
    # [23:26] Player 2 becomes the president of NYC
    g.process_action(
        action_helper.get_all_choices()[0]
    )  # [23:26] Player 2 exchanges Mohawk & Hudson from the IPO for a 10% share of NYC
    g.process_action(action_helper.get_all_choices()[-1])  # [23:26] Player 2 declines to sell shares
    g.process_action(action_helper.get_all_choices()[13])  # [23:26] Player 3 pars C&O at $67
    # [23:26] Player 3 buys a 20% share of C&O from the IPO for $134
    # [23:26] Player 3 becomes the president of C&O
    g.process_action(action_helper.get_all_choices()[-1])  # [23:26] Player 3 declines to sell shares
    g.process_action(action_helper.get_all_choices()[-2])  # [23:26] Player 4 sells 3 shares of B&O and receives $195
    # [23:26] Player 1 becomes the president of B&O
    # [23:26] B&O's share price moves down from 50
    g.process_action(
        action_helper.get_all_choices()[0]
    )  # [23:27] Player 4 buys a 10% share of NYC from the IPO for $67
    g.process_action(action_helper.get_all_choices()[-1])
    # [23:27] Player 1 has no valid actions and passes
    g.process_action(
        action_helper.get_all_choices()[0]
    )  # [23:27] Player 2 buys a 10% share of NYC from the IPO for $67
    g.process_action(action_helper.get_all_choices()[-1])  # [23:27] Player 2 declines to sell shares
    g.process_action(
        action_helper.get_all_choices()[1]
    )  # [23:27] Player 3 buys a 10% share of C&O from the IPO for $67
    g.process_action(action_helper.get_all_choices()[-1])  # [23:27] Player 3 declines to sell shares
    g.process_action(
        action_helper.get_all_choices()[0]
    )  # [23:27] Player 4 buys a 10% share of NYC from the IPO for $67
    # [23:27] NYC floats
    # [23:27] NYC receives $670
    g.process_action(action_helper.get_all_choices()[-1])  # [23:27] Player 4 declines to sell shares
    # [23:27] Player 1 has no valid actions and passes
    g.process_action(action_helper.get_all_choices()[2])  # [23:27] Player 2 sells 3 shares of PRR and receives $201
    # [23:27] PRR's share price moves down from 60
    g.process_action(
        action_helper.get_all_choices()[1]
    )  # [23:27] Player 2 buys a 10% share of C&O from the IPO for $67
    g.process_action(action_helper.get_all_choices()[-1])
    g.process_action(action_helper.get_all_choices()[1])  # [23:27] Player 3 sells 2 shares of PRR and receives $120
    # [23:27] PRR's share price moves down from 40
    g.process_action(
        action_helper.get_all_choices()[1]
    )  # [23:27] Player 3 buys a 10% share of C&O from the IPO for $67
    g.process_action(action_helper.get_all_choices()[-1])
    g.process_action(
        action_helper.get_all_choices()[1]
    )  # [23:27] Player 4 buys a 10% share of C&O from the IPO for $67
    # [23:27] C&O floats
    # [23:27] C&O receives $670
    g.process_action(action_helper.get_all_choices()[-1])  # [23:35] Player 4 declines to sell shares
    # [23:35] Player 1 has no valid actions and passes
    g.process_action(action_helper.get_all_choices()[20])  # [23:35] Player 2 sells a 10% share of B&O and receives $50
    # [23:35] B&O's share price moves down from 40
    g.process_action(action_helper.get_all_choices()[-1])  # [23:35] Player 2 declines to buy shares
    g.process_action(action_helper.get_all_choices()[4])  # [23:35] Player 3 sells a 10% share of B&O and receives $40
    # [23:35] B&O's share price moves down from 30
    g.process_action(action_helper.get_all_choices()[-1])  # [23:35] Player 3 declines to buy shares
    g.process_action(action_helper.get_all_choices()[-1])  # [23:35] Player 4 passes
    g.process_action(action_helper.get_all_choices()[-1])  # [23:35] Player 1 passes
    g.process_action(action_helper.get_all_choices()[-1])  # [23:35] Player 2 passes
    g.process_action(action_helper.get_all_choices()[-1])  # [23:35] Player 3 passes
    # [23:35] Player 4 has priority deal
    action_helper.print_summary(json_format=True)
    expected_state = {
        "players": {
            "Player 1": {"cash": 30, "shares": {"B&O": 20}, "companies": []},
            "Player 2": {"cash": 215, "shares": {"B&O": 0, "PRR": 0, "NYC": 40, "C&O": 10}, "companies": ["SV"]},
            "Player 3": {"cash": 161, "shares": {"PRR": 10, "B&O": 0, "C&O": 40}, "companies": ["CS", "CA"]},
            "Player 4": {"cash": 85, "shares": {"B&O": 0, "PRR": 40, "NYC": 20, "C&O": 10}, "companies": ["DH"]},
        },
        "corporations": {
            "PRR": {"cash": 1100, "companies": [], "trains": ["2"], "share_price": 40},
            "NYC": {"cash": 670, "companies": [], "trains": [], "share_price": 67},
            "CPR": {"cash": 0, "companies": [], "trains": [], "share_price": None},
            "B&O": {"cash": 0, "companies": [], "trains": ["2"], "share_price": 30},
            "C&O": {"cash": 670, "companies": [], "trains": [], "share_price": 67},
            "ERIE": {"cash": 0, "companies": [], "trains": [], "share_price": None},
            "NYNH": {"cash": 0, "companies": [], "trains": [], "share_price": None},
            "B&M": {"cash": 0, "companies": [], "trains": [], "share_price": None},
        },
    }
    assert action_helper.get_state() == expected_state, "State does not match expected state after stock round 2"

    # [23:35] -- Operating Round 2.1 (of 1) --
    # [23:35] Player 4 collects $15 from Delaware & Hudson
    # [23:35] Player 2 collects $5 from Schuylkill Valley
    # [23:35] Player 3 collects $10 from Champlain & St.Lawrence
    # [23:35] Player 3 collects $25 from Camden & Amboy
    # [23:35] Player 2 operates NYC
    # [23:35] NYC places a token on E19
    g.process_action(action_helper.get_all_choices()[-1])  # [23:35] NYC passes lay/upgrade track
    # [23:35] NYC skips place a token
    # [23:35] NYC skips run routes
    # [23:35] NYC does not run
    # [23:35] NYC's share price moves left from 65
    g.process_action(action_helper.get_all_choices()[0])  # [23:35] NYC buys a 2 train for $80 from The Depot
    g.process_action(action_helper.get_all_choices()[0])  # [23:35] NYC buys a 2 train for $80 from The Depot
    g.process_action(action_helper.get_all_choices()[0])  # [23:35] NYC buys a 2 train for $80 from The Depot
    g.process_action(action_helper.get_all_choices()[0])  # [23:36] NYC buys a 2 train for $80 from The Depot
    # [23:36] NYC skips buy companies
    # [23:36] Player 3 operates C&O
    # [23:36] C&O places a token on F6
    g.process_action(action_helper.get_all_choices()[-1])  # [23:36] C&O passes lay/upgrade track
    # [23:36] C&O skips place a token
    # [23:36] C&O skips run routes
    # [23:36] C&O does not run
    # [23:36] C&O's share price moves left from 65
    g.process_action(action_helper.get_all_choices()[0])  # [23:36] C&O buys a 3 train for $180 from The Depot
    # [23:36] -- Phase 3 (Operating Rounds: 2 | Train Limit: 4 | Available Tiles: Yellow, Green) --
    g.process_action(action_helper.get_all_choices()[-2])  # [23:36] C&O buys a 3 train for $180 from The Depot
    g.process_action(action_helper.get_all_choices()[-2])  # [23:36] C&O buys a 3 train for $180 from The Depot
    g.process_action(action_helper.get_all_choices()[-1])  # [23:36] C&O passes buy trains
    # [23:36] C&O passes buy companies
    # [23:36] Player 4 operates PRR
    g.process_action(action_helper.get_all_choices()[-1])  # [23:36] PRR passes lay/upgrade track
    g.process_action(action_helper.get_all_choices()[-1])  # [23:36] PRR passes place a token
    g.process_action(action_helper.get_all_choices()[-1])  # [23:36] PRR runs a 2 train for $30: H12-H10
    g.process_action(
        action_helper.get_all_choices()[-1]
    )  # [23:36] PRR pays out 3 per share (12 to Player 4, $3 to Player 3)
    # [23:36] PRR's share price moves right from 50
    g.process_action(action_helper.get_all_choices()[-2])  # [23:36] PRR buys a 3 train for $180 from The Depot
    g.process_action(action_helper.get_all_choices()[-2])  # [23:36] PRR buys a 3 train for $180 from The Depot
    g.process_action(action_helper.get_all_choices()[-2])  # [23:36] PRR buys a 4 train for $300 from The Depot
    # [23:36] -- Phase 4 (Operating Rounds: 2 | Train Limit: 3 | Available Tiles: Yellow, Green) --
    # [23:36] -- Event: 2 trains rust ( B&O x1, PRR x1, NYC x4) --
    g.process_action(action_helper.get_all_choices()[-1])  # [23:36] PRR passes buy companies
    # [23:36] Player 1 operates B&O
    g.process_action(action_helper.get_all_choices()[-1])  # [23:36] B&O passes lay/upgrade track
    # [23:36] B&O skips place a token
    # [23:36] B&O skips run routes
    # [23:36] B&O does not run
    # [23:36] B&O's share price moves left from 20
    g.process_action(
        action_helper.get_all_choices()[0]
    )  # [23:36] -- Player 1 goes bankrupt and sells remaining shares --
    # [23:36] -- Game over: Player 3 (562), Player 2 (40) --
    action_helper.print_summary(json_format=True)
    expected_state = {
        "players": {
            "Player 1": {"cash": 0, "shares": {"B&O": 20}, "companies": []},
            "Player 2": {"cash": 215, "shares": {"B&O": 0, "PRR": 0, "NYC": 40, "C&O": 10}, "companies": ["SV"]},
            "Player 3": {"cash": 161, "shares": {"PRR": 10, "B&O": 0, "C&O": 40}, "companies": ["CS", "CA"]},
            "Player 4": {"cash": 85, "shares": {"B&O": 0, "PRR": 40, "NYC": 20, "C&O": 10}, "companies": ["DH"]},
        },
        "corporations": {
            "PRR": {"cash": 470, "companies": [], "trains": ["3", "3", "4"], "share_price": 30},
            "NYC": {"cash": 350, "companies": [], "trains": [], "share_price": 65},
            "CPR": {"cash": 0, "companies": [], "trains": [], "share_price": None},
            "B&O": {"cash": 0, "companies": [], "trains": [], "share_price": 20},
            "C&O": {"cash": 130, "companies": [], "trains": ["3", "3", "3"], "share_price": 65},
            "ERIE": {"cash": 0, "companies": [], "trains": [], "share_price": None},
            "NYNH": {"cash": 0, "companies": [], "trains": [], "share_price": None},
            "B&M": {"cash": 0, "companies": [], "trains": [], "share_price": None},
        },
    }
    assert action_helper.get_state() == expected_state, "State does not match expected state at end of game"

    assert g.finished == True
    assert g.player_by_id("1").bankrupt, "Player 1 should be bankrupt"
    player_results = {player.name: player.value for player in g.players}
    assert player_results == {
        "Player 1": 40,
        "Player 2": 560,
        "Player 3": 651,
        "Player 4": 470,
    }, "Player results do not match expected results"
    assert next(iter(g.result().items()))[0] == "3", "Player 3 should be the winner"
