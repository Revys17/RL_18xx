import pytest
from pdb import set_trace
import numpy as np

# --- Imports from rl18xx library ---
# Adjust paths as necessary based on your project structure
from rl18xx.agent.alphazero.action_mapper import ActionMapper
from rl18xx.game.engine.actions import (
    Pass,
    Bid,
    Par,
    BaseAction,
    BuyShares,
    SellShares,
    PlaceToken,
    LayTile,
    BuyTrain,
    DiscardTrain,
    Dividend,
    BuyCompany,
    Bankrupt,
)
from rl18xx.game.engine.game import BaseGame
from rl18xx.game.gamemap import GameMap
from rl18xx.game.engine.entities import Player, Company, Corporation
from rl18xx.game import ActionHelper

# --- Test Fixtures ---


@pytest.fixture
def initial_game_state():
    game_map = GameMap()
    game = game_map.game_by_title("1830")
    return game({1: "Player 1", 2: "Player 2", 3: "Player 3", 4: "Player 4"})


@pytest.fixture
def stock_round_1_game_state(initial_game_state):
    action_helper = ActionHelper()
    initial_game_state.process_action(action_helper.get_all_choices(initial_game_state)[-1])  # pass
    initial_game_state.process_action(action_helper.get_all_choices(initial_game_state)[1])  # bid 45 on CS
    initial_game_state.process_action(action_helper.get_all_choices(initial_game_state)[1])  # bid 50 on CS
    initial_game_state.process_action(action_helper.get_all_choices(initial_game_state)[-77])  # bid 225 on BO
    initial_game_state.process_action(action_helper.get_all_choices(initial_game_state)[0])  # buy SV
    initial_game_state.process_action(action_helper.get_all_choices(initial_game_state)[-1])  # pass
    initial_game_state.process_action(action_helper.get_all_choices(initial_game_state)[0])  # buy DH
    initial_game_state.process_action(action_helper.get_all_choices(initial_game_state)[0])  # buy MH
    initial_game_state.process_action(action_helper.get_all_choices(initial_game_state)[0])  # buy CA
    initial_game_state.process_action(action_helper.get_all_choices(initial_game_state)[0])  # Par B&O at 100
    return initial_game_state


@pytest.fixture
def operating_round_1_game_state(stock_round_1_game_state):
    action_helper = ActionHelper()
    stock_round_1_game_state.process_action(action_helper.get_all_choices(stock_round_1_game_state)[-2])  # Par PRR
    stock_round_1_game_state.process_action(action_helper.get_all_choices(stock_round_1_game_state)[-1])  # Pass
    stock_round_1_game_state.process_action(action_helper.get_all_choices(stock_round_1_game_state)[-8])  # Par NYC
    stock_round_1_game_state.process_action(action_helper.get_all_choices(stock_round_1_game_state)[1])  # Buy PRR
    stock_round_1_game_state.process_action(action_helper.get_all_choices(stock_round_1_game_state)[1])  # Buy PRR
    stock_round_1_game_state.process_action(action_helper.get_all_choices(stock_round_1_game_state)[14])  # Par C&O
    stock_round_1_game_state.process_action(action_helper.get_all_choices(stock_round_1_game_state)[2])  # Buy NYC
    stock_round_1_game_state.process_action(action_helper.get_all_choices(stock_round_1_game_state)[1])  # Buy PRR
    stock_round_1_game_state.process_action(action_helper.get_all_choices(stock_round_1_game_state)[1])  # Buy PRR
    stock_round_1_game_state.process_action(action_helper.get_all_choices(stock_round_1_game_state)[3])  # Buy C&O
    stock_round_1_game_state.process_action(action_helper.get_all_choices(stock_round_1_game_state)[2])  # Buy NYC
    stock_round_1_game_state.process_action(action_helper.get_all_choices(stock_round_1_game_state)[0])  # Buy PRR
    stock_round_1_game_state.process_action(action_helper.get_all_choices(stock_round_1_game_state)[1])  # Buy PRR
    stock_round_1_game_state.process_action(action_helper.get_all_choices(stock_round_1_game_state)[3])  # Buy C&O
    stock_round_1_game_state.process_action(action_helper.get_all_choices(stock_round_1_game_state)[2])  # Buy NYC
    stock_round_1_game_state.process_action(action_helper.get_all_choices(stock_round_1_game_state)[1])  # Buy PRR
    stock_round_1_game_state.process_action(action_helper.get_all_choices(stock_round_1_game_state)[2])  # Buy C&O
    stock_round_1_game_state.process_action(action_helper.get_all_choices(stock_round_1_game_state)[1])  # Buy NYC
    stock_round_1_game_state.process_action(action_helper.get_all_choices(stock_round_1_game_state)[1])  # Buy NYC
    stock_round_1_game_state.process_action(action_helper.get_all_choices(stock_round_1_game_state)[2])  # Buy C&O
    stock_round_1_game_state.process_action(action_helper.get_all_choices(stock_round_1_game_state)[1])  # Buy NYC
    stock_round_1_game_state.process_action(action_helper.get_all_choices(stock_round_1_game_state)[1])  # Buy NYC
    return stock_round_1_game_state


@pytest.fixture
def stock_round_2_game_state(operating_round_1_game_state):
    action_helper = ActionHelper()
    # PRR
    operating_round_1_game_state.process_action(
        action_helper.get_all_choices(operating_round_1_game_state)[0]
    )  # lays tile #57 with rotation 1 on H10
    operating_round_1_game_state.process_action(action_helper.get_all_choices(operating_round_1_game_state)[-1])  # passes place token
    operating_round_1_game_state.process_action(action_helper.get_all_choices(operating_round_1_game_state)[0])  # buys a 2 train
    operating_round_1_game_state.process_action(action_helper.get_all_choices(operating_round_1_game_state)[0])  # buys a 2 train
    operating_round_1_game_state.process_action(action_helper.get_all_choices(operating_round_1_game_state)[-1])  # passes trains

    # NYC
    operating_round_1_game_state.process_action(
        action_helper.get_all_choices(operating_round_1_game_state)[0]
    )  # lays tile #57 with rotation 0 on E19
    operating_round_1_game_state.process_action(action_helper.get_all_choices(operating_round_1_game_state)[0])  # buys a 2 train
    operating_round_1_game_state.process_action(action_helper.get_all_choices(operating_round_1_game_state)[-1])  # passes trains

    # C&O
    operating_round_1_game_state.process_action(action_helper.get_all_choices(operating_round_1_game_state)[2])
    operating_round_1_game_state.process_action(action_helper.get_all_choices(operating_round_1_game_state)[0])  # Buys a 2 train
    operating_round_1_game_state.process_action(action_helper.get_all_choices(operating_round_1_game_state)[-1])  # passes trains
    return operating_round_1_game_state


@pytest.fixture
def operating_round_2_game_state(stock_round_2_game_state):
    action_helper = ActionHelper()
    stock_round_2_game_state.process_action(action_helper.get_all_choices(stock_round_2_game_state)[-2])  # sell 50% nyc
    stock_round_2_game_state.process_action(action_helper.get_all_choices(stock_round_2_game_state)[-3])  # par nynh at 71
    stock_round_2_game_state.process_action(action_helper.get_all_choices(stock_round_2_game_state)[0])  # buy C&O
    stock_round_2_game_state.process_action(action_helper.get_all_choices(stock_round_2_game_state)[-1])  # pass sell
    stock_round_2_game_state.process_action(action_helper.get_all_choices(stock_round_2_game_state)[0])  # Buy NYC
    stock_round_2_game_state.process_action(action_helper.get_all_choices(stock_round_2_game_state)[-1])  # pass sell
    stock_round_2_game_state.process_action(action_helper.get_all_choices(stock_round_2_game_state)[0])  # Buy NYNH
    stock_round_2_game_state.process_action(action_helper.get_all_choices(stock_round_2_game_state)[-1])  # pass sell
    stock_round_2_game_state.process_action(action_helper.get_all_choices(stock_round_2_game_state)[1])  # Buy NYNH
    stock_round_2_game_state.process_action(action_helper.get_all_choices(stock_round_2_game_state)[-1])  # pass sell
    stock_round_2_game_state.process_action(action_helper.get_all_choices(stock_round_2_game_state)[-1])  # pass
    stock_round_2_game_state.process_action(action_helper.get_all_choices(stock_round_2_game_state)[-1])  # pass
    stock_round_2_game_state.process_action(action_helper.get_all_choices(stock_round_2_game_state)[-1])  # pass
    stock_round_2_game_state.process_action(action_helper.get_all_choices(stock_round_2_game_state)[1])  # Buy NYNH
    stock_round_2_game_state.process_action(action_helper.get_all_choices(stock_round_2_game_state)[-1])  # pass sell
    stock_round_2_game_state.process_action(action_helper.get_all_choices(stock_round_2_game_state)[-1])  # pass
    stock_round_2_game_state.process_action(action_helper.get_all_choices(stock_round_2_game_state)[-1])  # pass
    stock_round_2_game_state.process_action(action_helper.get_all_choices(stock_round_2_game_state)[-1])  # pass
    stock_round_2_game_state.process_action(action_helper.get_all_choices(stock_round_2_game_state)[1])  # Buy NYNH
    stock_round_2_game_state.process_action(action_helper.get_all_choices(stock_round_2_game_state)[-1])  # pass sell
    stock_round_2_game_state.process_action(action_helper.get_all_choices(stock_round_2_game_state)[-1])  # pass
    stock_round_2_game_state.process_action(action_helper.get_all_choices(stock_round_2_game_state)[-1])  # pass
    stock_round_2_game_state.process_action(action_helper.get_all_choices(stock_round_2_game_state)[-1])  # pass
    stock_round_2_game_state.process_action(action_helper.get_all_choices(stock_round_2_game_state)[-1])  # pass
    return stock_round_2_game_state


@pytest.fixture
def bankruptcy_game_state(initial_game_state):
    action_helper = ActionHelper()
    # Auction
    initial_game_state.process_action(
        action_helper.get_all_choices(initial_game_state)[-2]
    )  # [20:39] -- Phase 2 (Operating Rounds: 1 | Train Limit: 4 | Available Tiles: Yellow) --
    # [20:39] Player 1 bids $600 for Baltimore & Ohio
    initial_game_state.process_action(
        action_helper.get_all_choices(initial_game_state)[0]
    )  # [20:39] Player 2 buys Schuylkill Valley for $20
    initial_game_state.process_action(
        action_helper.get_all_choices(initial_game_state)[0]
    )  # [20:39] Player 3 buys Champlain & St.Lawrence for $40
    initial_game_state.process_action(
        action_helper.get_all_choices(initial_game_state)[0]
    )  # [20:39] Player 4 buys Delaware & Hudson for $70
    initial_game_state.process_action(action_helper.get_all_choices(initial_game_state)[0])  # [20:39] Player 1 passes bidding
    initial_game_state.process_action(
        action_helper.get_all_choices(initial_game_state)[0]
    )  # [20:39] Player 2 buys Mohawk & Hudson for $110
    initial_game_state.process_action(
        action_helper.get_all_choices(initial_game_state)[0]
    )  # [20:39] Player 3 buys Camden & Amboy for $160
    # [20:39] Player 3 receives a 10% share of PRR
    # [20:39] Player 1 wins the auction for Baltimore & Ohio with the only bid of $600
    initial_game_state.process_action(action_helper.get_all_choices(initial_game_state)[-1])  # [20:39] Player 1 pars B&O at $67
    # [20:39] Player 1 receives a 20% share of B&O
    # [20:39] Player 1 becomes the president of B&O
    # [20:39] Player 4 has priority deal
    # [20:39] -- Stock Round 1 --
    initial_game_state.process_action(
        action_helper.get_all_choices(initial_game_state)[0]
    )  # [20:39] Player 4 buys a 10% share of B&O from the IPO for $67
    # [20:39] Player 1 has no valid actions and passes
    initial_game_state.process_action(
        action_helper.get_all_choices(initial_game_state)[0]
    )  # [21:13] Player 2 buys a 10% share of B&O from the IPO for $67
    initial_game_state.process_action(
        action_helper.get_all_choices(initial_game_state)[0]
    )  # [21:13] Player 3 buys a 10% share of B&O from the IPO for $67
    initial_game_state.process_action(
        action_helper.get_all_choices(initial_game_state)[0]
    )  # [21:13] Player 4 buys a 10% share of B&O from the IPO for $67
    # [21:13] B&O floats
    # [21:13] B&O receives $670
    # [21:13] Player 1 has no valid actions and passes
    initial_game_state.process_action(action_helper.get_all_choices(initial_game_state)[-2])  # [21:13] Player 2 pars PRR at $67
    # [21:13] Player 2 buys a 20% share of PRR from the IPO for $134
    # [21:13] Player 2 becomes the president of PRR
    initial_game_state.process_action(
        action_helper.get_all_choices(initial_game_state)[1]
    )  # [21:13] Player 3 buys a 10% share of PRR from the IPO for $67
    initial_game_state.process_action(
        action_helper.get_all_choices(initial_game_state)[1]
    )  # [21:13] Player 4 buys a 10% share of PRR from the IPO for $67
    # [21:13] Player 1 has no valid actions and passes
    initial_game_state.process_action(
        action_helper.get_all_choices(initial_game_state)[1]
    )  # [21:13] Player 2 buys a 10% share of PRR from the IPO for $67
    # [21:13] PRR floats
    # [21:13] PRR receives $670
    initial_game_state.process_action(
        action_helper.get_all_choices(initial_game_state)[1]
    )  # [21:13] Player 3 buys a 10% share of PRR from the IPO for $67
    initial_game_state.process_action(
        action_helper.get_all_choices(initial_game_state)[0]
    )  # [21:14] Player 4 buys a 10% share of B&O from the IPO for $67
    # [21:14] Player 4 becomes the president of B&O
    # [21:14] Player 1 has no valid actions and passes
    initial_game_state.process_action(action_helper.get_all_choices(initial_game_state)[-1])  # [21:14] Player 2 passes
    initial_game_state.process_action(action_helper.get_all_choices(initial_game_state)[-1])  # [21:14] Player 3 passes
    initial_game_state.process_action(
        action_helper.get_all_choices(initial_game_state)[1]
    )  # [21:14] Player 4 buys a 10% share of PRR from the IPO for $67
    # [21:14] Player 1 has no valid actions and passes
    initial_game_state.process_action(action_helper.get_all_choices(initial_game_state)[-1])  # [21:14] Player 2 passes
    initial_game_state.process_action(action_helper.get_all_choices(initial_game_state)[-1])  # [21:14] Player 3 passes
    initial_game_state.process_action(
        action_helper.get_all_choices(initial_game_state)[1]
    )  # [21:14] Player 4 buys a 10% share of PRR from the IPO for $67
    # [21:14] Player 1 has no valid actions and passes
    initial_game_state.process_action(action_helper.get_all_choices(initial_game_state)[-1])  # [21:14] Player 2 passes
    initial_game_state.process_action(action_helper.get_all_choices(initial_game_state)[-1])  # [21:14] Player 3 passes
    initial_game_state.process_action(
        action_helper.get_all_choices(initial_game_state)[1]
    )  # [21:14] Player 4 buys a 10% share of PRR from the IPO for $67
    # [21:14] Player 4 becomes the president of PRR
    # [21:14] Player 1 has no valid actions and passes
    initial_game_state.process_action(action_helper.get_all_choices(initial_game_state)[-1])  # [21:15] Player 2 passes
    initial_game_state.process_action(action_helper.get_all_choices(initial_game_state)[-1])  # [21:15] Player 3 passes
    # [21:15] Player 4 has no valid actions and passes
    # [21:15] PRR's share price moves up from 71
    # [21:15] Player 1 has priority deal
    # [21:15] -- Operating Round 1.1 (of 1) --
    # [21:15] Player 1 collects $30 from Baltimore & Ohio
    # [21:15] Player 2 collects $5 from Schuylkill Valley
    # [21:15] Player 2 collects $20 from Mohawk & Hudson
    # [21:15] Player 3 collects $10 from Champlain & St.Lawrence
    # [21:15] Player 3 collects $25 from Camden & Amboy
    # [21:15] Player 4 collects $15 from Delaware & Hudson
    # [21:15] Player 4 operates PRR
    # [21:15] PRR places a token on H12
    initial_game_state.process_action(
        action_helper.get_all_choices(initial_game_state)[0]
    )  # [21:16] PRR lays tile #57 with rotation 1 on H10 (Pittsburgh)
    initial_game_state.process_action(action_helper.get_all_choices(initial_game_state)[-1])  # [21:16] PRR passes place a token
    # [21:16] PRR skips run routes
    # [21:16] PRR does not run
    # [21:16] PRR's share price moves left from 67
    initial_game_state.process_action(
        action_helper.get_all_choices(initial_game_state)[0]
    )  # [21:16] PRR buys a 2 train for $80 from The Depot
    initial_game_state.process_action(
        action_helper.get_all_choices(initial_game_state)[0]
    )  # [21:16] PRR buys a 2 train for $80 from The Depot
    initial_game_state.process_action(action_helper.get_all_choices(initial_game_state)[-1])  # [21:17] PRR passes buy trains
    # [21:17] PRR skips buy companies
    # [21:17] Player 4 operates B&O
    # [21:17] B&O places a token on I15
    initial_game_state.process_action(
        action_helper.get_all_choices(initial_game_state)[4]
    )  # [21:17] B&O spends $80 and lays tile #57 with rotation 0 on J14 (Washington)
    initial_game_state.process_action(action_helper.get_all_choices(initial_game_state)[-1])  # [21:17] B&O passes place a token
    # [21:17] B&O skips run routes
    # [21:17] B&O does not run
    # [21:17] B&O's share price moves left from 65
    initial_game_state.process_action(
        action_helper.get_all_choices(initial_game_state)[-1]
    )  # [21:22] B&O buys a 2 train for $590 from PRR
    # [21:22] Baltimore & Ohio closes
    # [21:22] B&O skips buy companies
    # [21:22] -- Stock Round 2 --
    initial_game_state.process_action(action_helper.get_all_choices(initial_game_state)[-1])  # [21:23] Player 1 passes
    # [23:26] Player 2 pars NYC at $67
    initial_game_state.process_action(
        action_helper.get_all_choices(initial_game_state)[31]
    )  # [23:26] Player 2 buys a 20% share of NYC from the IPO for $134
    # [23:26] Player 2 becomes the president of NYC
    initial_game_state.process_action(
        action_helper.get_all_choices(initial_game_state)[0]
    )  # [23:26] Player 2 exchanges Mohawk & Hudson from the IPO for a 10% share of NYC
    initial_game_state.process_action(action_helper.get_all_choices(initial_game_state)[-1])  # [23:26] Player 2 declines to sell shares
    initial_game_state.process_action(action_helper.get_all_choices(initial_game_state)[13])  # [23:26] Player 3 pars C&O at $67
    # [23:26] Player 3 buys a 20% share of C&O from the IPO for $134
    # [23:26] Player 3 becomes the president of C&O
    initial_game_state.process_action(action_helper.get_all_choices(initial_game_state)[-1])  # [23:26] Player 3 declines to sell shares
    initial_game_state.process_action(
        action_helper.get_all_choices(initial_game_state)[-2]
    )  # [23:26] Player 4 sells 3 shares of B&O and receives $195
    # [23:26] Player 1 becomes the president of B&O
    # [23:26] B&O's share price moves down from 50
    initial_game_state.process_action(
        action_helper.get_all_choices(initial_game_state)[0]
    )  # [23:27] Player 4 buys a 10% share of NYC from the IPO for $67
    initial_game_state.process_action(action_helper.get_all_choices(initial_game_state)[-1])
    # [23:27] Player 1 has no valid actions and passes
    initial_game_state.process_action(
        action_helper.get_all_choices(initial_game_state)[0]
    )  # [23:27] Player 2 buys a 10% share of NYC from the IPO for $67
    initial_game_state.process_action(action_helper.get_all_choices(initial_game_state)[-1])  # [23:27] Player 2 declines to sell shares
    initial_game_state.process_action(
        action_helper.get_all_choices(initial_game_state)[1]
    )  # [23:27] Player 3 buys a 10% share of C&O from the IPO for $67
    initial_game_state.process_action(action_helper.get_all_choices(initial_game_state)[-1])  # [23:27] Player 3 declines to sell shares
    initial_game_state.process_action(
        action_helper.get_all_choices(initial_game_state)[0]
    )  # [23:27] Player 4 buys a 10% share of NYC from the IPO for $67
    # [23:27] NYC floats
    # [23:27] NYC receives $670
    initial_game_state.process_action(action_helper.get_all_choices(initial_game_state)[-1])  # [23:27] Player 4 declines to sell shares
    # [23:27] Player 1 has no valid actions and passes
    initial_game_state.process_action(
        action_helper.get_all_choices(initial_game_state)[2]
    )  # [23:27] Player 2 sells 3 shares of PRR and receives $201
    # [23:27] PRR's share price moves down from 60
    initial_game_state.process_action(
        action_helper.get_all_choices(initial_game_state)[1]
    )  # [23:27] Player 2 buys a 10% share of C&O from the IPO for $67
    initial_game_state.process_action(action_helper.get_all_choices(initial_game_state)[-1])
    initial_game_state.process_action(
        action_helper.get_all_choices(initial_game_state)[1]
    )  # [23:27] Player 3 sells 2 shares of PRR and receives $120
    # [23:27] PRR's share price moves down from 40
    initial_game_state.process_action(
        action_helper.get_all_choices(initial_game_state)[1]
    )  # [23:27] Player 3 buys a 10% share of C&O from the IPO for $67
    initial_game_state.process_action(action_helper.get_all_choices(initial_game_state)[-1])
    initial_game_state.process_action(
        action_helper.get_all_choices(initial_game_state)[1]
    )  # [23:27] Player 4 buys a 10% share of C&O from the IPO for $67
    # [23:27] C&O floats
    # [23:27] C&O receives $670
    initial_game_state.process_action(action_helper.get_all_choices(initial_game_state)[-1])  # [23:35] Player 4 declines to sell shares
    # [23:35] Player 1 has no valid actions and passes
    initial_game_state.process_action(
        action_helper.get_all_choices(initial_game_state)[20]
    )  # [23:35] Player 2 sells a 10% share of B&O and receives $50
    # [23:35] B&O's share price moves down from 40
    initial_game_state.process_action(action_helper.get_all_choices(initial_game_state)[-1])  # [23:35] Player 2 declines to buy shares
    initial_game_state.process_action(
        action_helper.get_all_choices(initial_game_state)[4]
    )  # [23:35] Player 3 sells a 10% share of B&O and receives $40
    # [23:35] B&O's share price moves down from 30
    initial_game_state.process_action(action_helper.get_all_choices(initial_game_state)[-1])  # [23:35] Player 3 declines to buy shares
    initial_game_state.process_action(action_helper.get_all_choices(initial_game_state)[-1])  # [23:35] Player 4 passes
    initial_game_state.process_action(action_helper.get_all_choices(initial_game_state)[-1])  # [23:35] Player 1 passes
    initial_game_state.process_action(action_helper.get_all_choices(initial_game_state)[-1])  # [23:35] Player 2 passes
    initial_game_state.process_action(action_helper.get_all_choices(initial_game_state)[-1])  # [23:35] Player 3 passes
    # [23:35] Player 4 has priority deal
    # [23:35] -- Operating Round 2.1 (of 1) --
    # [23:35] Player 4 collects $15 from Delaware & Hudson
    # [23:35] Player 2 collects $5 from Schuylkill Valley
    # [23:35] Player 3 collects $10 from Champlain & St.Lawrence
    # [23:35] Player 3 collects $25 from Camden & Amboy
    # [23:35] Player 2 operates NYC
    # [23:35] NYC places a token on E19
    initial_game_state.process_action(action_helper.get_all_choices(initial_game_state)[-1])  # [23:35] NYC passes lay/upgrade track
    # [23:35] NYC skips place a token
    # [23:35] NYC skips run routes
    # [23:35] NYC does not run
    # [23:35] NYC's share price moves left from 65
    initial_game_state.process_action(
        action_helper.get_all_choices(initial_game_state)[0]
    )  # [23:35] NYC buys a 2 train for $80 from The Depot
    initial_game_state.process_action(
        action_helper.get_all_choices(initial_game_state)[0]
    )  # [23:35] NYC buys a 2 train for $80 from The Depot
    initial_game_state.process_action(
        action_helper.get_all_choices(initial_game_state)[0]
    )  # [23:35] NYC buys a 2 train for $80 from The Depot
    initial_game_state.process_action(
        action_helper.get_all_choices(initial_game_state)[0]
    )  # [23:36] NYC buys a 2 train for $80 from The Depot
    # [23:36] NYC skips buy companies
    # [23:36] Player 3 operates C&O
    # [23:36] C&O places a token on F6
    initial_game_state.process_action(action_helper.get_all_choices(initial_game_state)[-1])  # [23:36] C&O passes lay/upgrade track
    # [23:36] C&O skips place a token
    # [23:36] C&O skips run routes
    # [23:36] C&O does not run
    # [23:36] C&O's share price moves left from 65
    initial_game_state.process_action(
        action_helper.get_all_choices(initial_game_state)[0]
    )  # [23:36] C&O buys a 3 train for $180 from The Depot
    # [23:36] -- Phase 3 (Operating Rounds: 2 | Train Limit: 4 | Available Tiles: Yellow, Green) --
    initial_game_state.process_action(
        action_helper.get_all_choices(initial_game_state)[-2]
    )  # [23:36] C&O buys a 3 train for $180 from The Depot
    initial_game_state.process_action(
        action_helper.get_all_choices(initial_game_state)[-2]
    )  # [23:36] C&O buys a 3 train for $180 from The Depot
    initial_game_state.process_action(action_helper.get_all_choices(initial_game_state)[-1])  # [23:36] C&O passes buy trains
    # [23:36] C&O passes buy companies
    # [23:36] Player 4 operates PRR
    initial_game_state.process_action(action_helper.get_all_choices(initial_game_state)[-1])  # [23:36] PRR passes lay/upgrade track
    initial_game_state.process_action(action_helper.get_all_choices(initial_game_state)[-1])  # [23:36] PRR passes place a token
    initial_game_state.process_action(
        action_helper.get_all_choices(initial_game_state)[-1]
    )  # [23:36] PRR runs a 2 train for $30: H12-H10
    initial_game_state.process_action(
        action_helper.get_all_choices(initial_game_state)[-1]
    )  # [23:36] PRR pays out 3 per share (12 to Player 4, $3 to Player 3)
    # [23:36] PRR's share price moves right from 50
    initial_game_state.process_action(
        action_helper.get_all_choices(initial_game_state)[-2]
    )  # [23:36] PRR buys a 3 train for $180 from The Depot
    initial_game_state.process_action(
        action_helper.get_all_choices(initial_game_state)[-2]
    )  # [23:36] PRR buys a 3 train for $180 from The Depot
    initial_game_state.process_action(
        action_helper.get_all_choices(initial_game_state)[-2]
    )  # [23:36] PRR buys a 4 train for $300 from The Depot
    # [23:36] -- Phase 4 (Operating Rounds: 2 | Train Limit: 3 | Available Tiles: Yellow, Green) --
    # [23:36] -- Event: 2 trains rust ( B&O x1, PRR x1, NYC x4) --
    initial_game_state.process_action(action_helper.get_all_choices(initial_game_state)[-1])  # [23:36] PRR passes buy companies
    # [23:36] Player 1 operates B&O
    initial_game_state.process_action(action_helper.get_all_choices(initial_game_state)[-1])  # [23:36] B&O passes lay/upgrade track
    # [23:36] B&O skips place a token
    # [23:36] B&O skips run routes
    # [23:36] B&O does not run
    # [23:36] B&O's share price moves left from 20
    return initial_game_state


# --- Test Helpers ---
def get_expected_index_for_action(action_mapper, action):
    if isinstance(action, Pass):
        return 0
    elif isinstance(action, Bid):
        # All bid actions get mapped to the same action index representing the min bid on that company
        return action_mapper.action_offsets["Bid"] + action_mapper.company_offsets[action.company.id]
    elif isinstance(action, Par):
        return (
            action_mapper.action_offsets["Par"]
            + action_mapper.corporation_offsets[action.corporation.id] * len(action_mapper.par_price_offsets)
            + action_mapper.par_price_offsets[action.share_price.price]
        )
    elif isinstance(action, BuyShares):
        if action.bundle.owner.name == "Market":
            location = "market"
        elif action.bundle.owner == action.bundle.corporation:
            location = "ipo"
        else:
            raise ValueError(f"Unknown share owner: {action.bundle.owner}")

        if action.entity.is_company():
            return action_mapper.action_offsets["CompanyBuyShares"] + action_mapper.share_location_offsets[location]

        return (
            action_mapper.action_offsets["BuyShares"]
            + action_mapper.corporation_offsets[action.bundle.corporation.id]
            * len(action_mapper.share_location_offsets)
            + action_mapper.share_location_offsets[location]
        )
    elif isinstance(action, SellShares):
        return (
            action_mapper.action_offsets["SellShares"]
            + action_mapper.corporation_offsets[action.bundle.corporation.id] * 5
            + action.bundle.num_shares()
            - 1
        )
    elif isinstance(action, PlaceToken):
        if action.entity.is_company():
            return action_mapper.action_offsets["CompanyPlaceToken"]
        return (
            action_mapper.action_offsets["PlaceToken"]
            + action_mapper.city_offsets[(action.city.tile.hex.id, action.city.tile.cities.index(action.city))]
        )
    elif isinstance(action, LayTile):
        if action.entity.is_company():
            if action.entity.sym == "DH":
                return action_mapper.action_offsets["CompanyLayTile"] + action.rotation
            elif action.entity.sym == "CS":
                return (
                    action_mapper.action_offsets["CompanyLayTile"]
                    + 6
                    + action_mapper.company_tile_offsets[action.tile.name] * 6
                    + action.rotation
                )
            else:
                raise ValueError(f"Unknown company: {action.entity}")
        return (
            action_mapper.action_offsets["LayTile"]
            + action_mapper.hex_offsets[action.hex.id] * len(action_mapper.tile_offsets) * 6
            + action_mapper.tile_offsets[action.tile.name] * 6
            + action.rotation
        )
    elif isinstance(action, BuyTrain):
        if action.train.owner.name == "The Depot":
            if action.train in action.train.owner.discarded:
                return (
                    action_mapper.action_offsets["BuyTrain"] + 1 + action_mapper.train_type_offsets[action.train.name]
                )
            return action_mapper.action_offsets["BuyTrain"]

        if int(action.price) == action.entity.cash - 1:
            price = "all-but-one"
        elif int(action.price) == action.entity.cash:
            price = "all"
        else:
            price = str(action.price)
        return (
            action_mapper.action_offsets["BuyTrain"]
            + 1
            + len(action_mapper.train_type_offsets)
            + action_mapper.corporation_offsets[action.train.owner.id]
            * len(action_mapper.train_price_offsets)
            * len(action_mapper.train_type_offsets)
            + action_mapper.train_type_offsets[action.train.name] * len(action_mapper.train_price_offsets)
            + action_mapper.train_price_offsets[price]
        )
    elif isinstance(action, DiscardTrain):
        return action_mapper.action_offsets["DiscardTrain"] + action_mapper.train_type_offsets[action.train.name]
    elif isinstance(action, Dividend):
        return action_mapper.action_offsets["Dividend"] + action_mapper.dividend_offsets[action.kind]
    elif isinstance(action, BuyCompany):
        if action.price == action.company.min_price:
            price = "min"
        elif action.price == action.company.max_price:
            price = "max"
        else:
            raise ValueError(f"Invalid price for BuyCompany action: {action.price}")
        return (
            action_mapper.action_offsets["BuyCompany"]
            + action_mapper.company_offsets[action.company.id] * len(action_mapper.buy_company_price_offsets)
            + action_mapper.buy_company_price_offsets[price]
        )
    elif isinstance(action, Bankrupt):
        return action_mapper.action_offsets["Bankrupt"]
    else:
        assert False, f"Unknown action: {action}"


def check_action_in_all_actions(action, all_actions):
    assertion = False
    for a in all_actions:
        if action.__class__ != a.__class__:
            continue
        if action.entity.id != a.entity.id:
            continue
        if action.args_to_dict() != a.args_to_dict():
            continue
        assertion = True
    if not assertion:
        print(f"action class: {action.__class__}")
        print(f"action entity: {action.entity}")
        print(f"action args: {action.args_to_dict()}")
        for a in all_actions:
            print(f"a class: {a.__class__}")
            print(f"a entity: {a.entity}")
            print(f"a args: {a.args_to_dict()}")
    assert assertion, f"Action {action} not in all_actions"


# --- Test Cases ---


def test_auction(initial_game_state):
    action_mapper = ActionMapper()
    action_helper = ActionHelper()

    all_actions = action_helper.get_all_choices_limited(initial_game_state)
    for action in all_actions:
        assert action_mapper.get_index_for_action(action) == get_expected_index_for_action(action_mapper, action)

    mask = action_mapper.get_legal_action_mask(initial_game_state)
    assert mask.shape == (26535,)
    assert mask.dtype == np.float32
    assert mask[0] == 1.0
    assert mask[1] == 1.0
    assert mask[2] == 1.0
    assert mask[3] == 1.0
    assert mask[4] == 1.0
    assert mask[5] == 1.0
    assert mask[6] == 1.0
    assert sum(mask) == 7.0

    # Check the other direction
    indices = [i for i, mask in enumerate(mask) if mask == 1.0]
    mapped_actions = [action_mapper.map_index_to_action(i, initial_game_state) for i in indices]
    for action in mapped_actions:
        check_action_in_all_actions(action, all_actions)

    # Take some actions:
    initial_game_state.process_action(action_helper.get_all_choices(initial_game_state)[-1])  # pass
    initial_game_state.process_action(action_helper.get_all_choices(initial_game_state)[1])  # bid 45 on CS
    initial_game_state.process_action(action_helper.get_all_choices(initial_game_state)[1])  # bid 50 on CS
    initial_game_state.process_action(action_helper.get_all_choices(initial_game_state)[-77])  # bid 225 on BO
    initial_game_state.process_action(action_helper.get_all_choices(initial_game_state)[0])  # buy SV

    all_actions = action_helper.get_all_choices_limited(initial_game_state)
    for action in all_actions:
        assert action_mapper.get_index_for_action(action) == get_expected_index_for_action(action_mapper, action)

    mask = action_mapper.get_legal_action_mask(initial_game_state)
    assert mask.shape == (26535,)
    assert mask.dtype == np.float32
    assert mask[0] == 1.0
    assert mask[1] == 0.0
    assert mask[2] == 1.0
    assert mask[3] == 0.0
    assert mask[4] == 0.0
    assert mask[5] == 0.0
    assert mask[6] == 0.0
    assert sum(mask) == 2.0

    # Check the other direction
    indices = [i for i, mask in enumerate(mask) if mask == 1.0]
    mapped_actions = [action_mapper.map_index_to_action(i, initial_game_state) for i in indices]
    for action in mapped_actions:
        check_action_in_all_actions(action, all_actions)

    initial_game_state.process_action(action_helper.get_all_choices(initial_game_state)[-1])  # pass
    initial_game_state.process_action(action_helper.get_all_choices(initial_game_state)[0])  # buy DH
    initial_game_state.process_action(action_helper.get_all_choices(initial_game_state)[0])  # buy MH

    all_actions = action_helper.get_all_choices_limited(initial_game_state)
    for action in all_actions:
        assert action_mapper.get_index_for_action(action) == get_expected_index_for_action(action_mapper, action)

    mask = action_mapper.get_legal_action_mask(initial_game_state)
    assert mask.shape == (26535,)
    assert mask.dtype == np.float32
    assert mask[0] == 1.0
    assert mask[5] == 1.0
    assert mask[6] == 1.0
    assert sum(mask) == 3.0

    initial_game_state.process_action(action_helper.get_all_choices(initial_game_state)[0])  # buy CA

    all_actions = action_helper.get_all_choices_limited(initial_game_state)
    for action in all_actions:
        assert action_mapper.get_index_for_action(action) == get_expected_index_for_action(action_mapper, action)

    mask = action_mapper.get_legal_action_mask(initial_game_state)
    assert mask.shape == (26535,)
    assert mask.dtype == np.float32
    assert mask[25] == 1.0
    assert mask[26] == 1.0
    assert mask[27] == 1.0
    assert mask[28] == 1.0
    assert mask[29] == 1.0
    assert mask[30] == 1.0
    assert sum(mask) == 6.0

    # Check the other direction
    indices = [i for i, mask in enumerate(mask) if mask == 1.0]
    mapped_actions = [action_mapper.map_index_to_action(i, initial_game_state) for i in indices]
    for action in mapped_actions:
        check_action_in_all_actions(action, all_actions)


def test_stock_round_1_game_state(stock_round_1_game_state):
    action_mapper = ActionMapper()
    action_helper = ActionHelper()

    # Test initial stock round 1 state
    all_actions = action_helper.get_all_choices_limited(stock_round_1_game_state)
    for action in all_actions:
        assert action_mapper.get_index_for_action(action) == get_expected_index_for_action(action_mapper, action)

    mask = action_mapper.get_legal_action_mask(stock_round_1_game_state)
    assert mask.shape == (26535,)
    assert mask.dtype == np.float32
    assert mask[0] == 1.0
    # Par PRR
    assert mask[7] == 1.0
    assert mask[8] == 1.0
    assert mask[9] == 1.0
    assert mask[10] == 1.0
    assert mask[11] == 1.0
    assert mask[12] == 1.0
    # Par NYC
    assert mask[13] == 1.0
    assert mask[14] == 1.0
    assert mask[15] == 1.0
    assert mask[16] == 1.0
    assert mask[17] == 1.0
    assert mask[18] == 1.0
    # Par CPR
    assert mask[19] == 1.0
    assert mask[20] == 1.0
    assert mask[21] == 1.0
    assert mask[22] == 1.0
    assert mask[23] == 1.0
    assert mask[24] == 1.0
    # Can't Par B&O
    assert mask[25] == 0.0
    assert mask[26] == 0.0
    assert mask[27] == 0.0
    assert mask[28] == 0.0
    assert mask[29] == 0.0
    assert mask[30] == 0.0
    # Par C&O
    assert mask[31] == 1.0
    assert mask[32] == 1.0
    assert mask[33] == 1.0
    assert mask[34] == 1.0
    assert mask[35] == 1.0
    assert mask[36] == 1.0
    # Par ERIE
    assert mask[37] == 1.0
    assert mask[38] == 1.0
    assert mask[39] == 1.0
    assert mask[40] == 1.0
    assert mask[41] == 1.0
    assert mask[42] == 1.0
    # Par NYNH
    assert mask[43] == 1.0
    assert mask[44] == 1.0
    assert mask[45] == 1.0
    assert mask[46] == 1.0
    assert mask[47] == 1.0
    assert mask[48] == 1.0
    # Par B&M
    assert mask[49] == 1.0
    assert mask[50] == 1.0
    assert mask[51] == 1.0
    assert mask[52] == 1.0
    assert mask[53] == 1.0
    assert mask[54] == 1.0
    # Buy PRR
    assert mask[55] == 0.0
    assert mask[56] == 0.0
    # Buy NYC
    assert mask[57] == 0.0
    assert mask[58] == 0.0
    # Buy CPR
    assert mask[59] == 0.0
    assert mask[60] == 0.0
    # Buy B&O (ipo only)
    assert mask[61] == 1.0
    assert mask[62] == 0.0
    # Buy C&O
    assert mask[63] == 0.0
    assert mask[64] == 0.0
    # Buy ERIE
    assert mask[65] == 0.0
    assert mask[66] == 0.0
    # Buy NYNH
    assert mask[67] == 0.0
    assert mask[68] == 0.0
    # Buy B&M
    assert mask[69] == 0.0
    assert mask[70] == 0.0
    # Can't sell shares in SR1
    assert sum(mask) == 44.0

    # Check the other direction
    indices = [i for i, mask in enumerate(mask) if mask == 1.0]
    mapped_actions = [action_mapper.map_index_to_action(i, stock_round_1_game_state) for i in indices]
    for action in mapped_actions:
        check_action_in_all_actions(action, all_actions)

    # Test after some stock purchases
    stock_round_1_game_state.process_action(action_helper.get_all_choices(stock_round_1_game_state)[-2])  # Par PRR
    stock_round_1_game_state.process_action(action_helper.get_all_choices(stock_round_1_game_state)[-1])  # Pass
    stock_round_1_game_state.process_action(action_helper.get_all_choices(stock_round_1_game_state)[-8])  # Par NYC
    stock_round_1_game_state.process_action(action_helper.get_all_choices(stock_round_1_game_state)[1])  # Buy PRR
    stock_round_1_game_state.process_action(action_helper.get_all_choices(stock_round_1_game_state)[1])  # Buy PRR
    stock_round_1_game_state.process_action(action_helper.get_all_choices(stock_round_1_game_state)[14])  # Par C&O
    stock_round_1_game_state.process_action(action_helper.get_all_choices(stock_round_1_game_state)[2])  # Buy NYC
    stock_round_1_game_state.process_action(action_helper.get_all_choices(stock_round_1_game_state)[1])  # Buy PRR
    stock_round_1_game_state.process_action(action_helper.get_all_choices(stock_round_1_game_state)[1])  # Buy PRR
    stock_round_1_game_state.process_action(action_helper.get_all_choices(stock_round_1_game_state)[3])  # Buy C&O
    stock_round_1_game_state.process_action(action_helper.get_all_choices(stock_round_1_game_state)[2])  # Buy NYC
    stock_round_1_game_state.process_action(action_helper.get_all_choices(stock_round_1_game_state)[0])  # Buy PRR
    stock_round_1_game_state.process_action(action_helper.get_all_choices(stock_round_1_game_state)[1])  # Buy PRR
    stock_round_1_game_state.process_action(action_helper.get_all_choices(stock_round_1_game_state)[3])  # Buy C&O
    stock_round_1_game_state.process_action(action_helper.get_all_choices(stock_round_1_game_state)[2])  # Buy NYC
    stock_round_1_game_state.process_action(action_helper.get_all_choices(stock_round_1_game_state)[1])  # Buy PRR
    stock_round_1_game_state.process_action(action_helper.get_all_choices(stock_round_1_game_state)[2])  # Buy C&O
    stock_round_1_game_state.process_action(action_helper.get_all_choices(stock_round_1_game_state)[1])  # Buy NYC
    stock_round_1_game_state.process_action(action_helper.get_all_choices(stock_round_1_game_state)[1])  # Buy NYC
    stock_round_1_game_state.process_action(action_helper.get_all_choices(stock_round_1_game_state)[2])  # Buy C&O
    stock_round_1_game_state.process_action(action_helper.get_all_choices(stock_round_1_game_state)[1])  # Buy NYC

    all_actions = action_helper.get_all_choices_limited(stock_round_1_game_state)
    print("\n".join([str(action) for action in all_actions]))
    for action in all_actions:
        assert action_mapper.get_index_for_action(action) == get_expected_index_for_action(action_mapper, action)

    mask = action_mapper.get_legal_action_mask(stock_round_1_game_state)
    assert mask.shape == (26535,)
    assert mask.dtype == np.float32
    assert mask[0] == 1.0
    # Buy NYC (ipo only)
    assert mask[57] == 1.0
    assert mask[58] == 0.0
    # Buy B&O (ipo only)
    assert mask[61] == 1.0
    assert mask[62] == 0.0
    # Can't sell shares in SR1
    assert sum(mask) == 3.0

    # Check the other direction
    indices = [i for i, mask in enumerate(mask) if mask == 1.0]
    mapped_actions = [action_mapper.map_index_to_action(i, stock_round_1_game_state) for i in indices]
    for action in mapped_actions:
        check_action_in_all_actions(action, all_actions)


def test_operating_round_1_game_state(operating_round_1_game_state):
    action_mapper = ActionMapper()
    action_helper = ActionHelper()

    # Test initial operating round 1 state
    all_actions = action_helper.get_all_choices_limited(operating_round_1_game_state)
    for action in all_actions:
        assert action_mapper.get_index_for_action(action) == get_expected_index_for_action(action_mapper, action)

    mask = action_mapper.get_legal_action_mask(operating_round_1_game_state)
    assert mask.shape == (26535,)
    assert mask.dtype == np.float32
    assert mask[0] == 1.0

    h10_city_lay_idx = (
        action_mapper.action_offsets["LayTile"]
        + action_mapper.hex_offsets["H10"] * 6 * len(action_mapper.tile_offsets)
        + action_mapper.tile_offsets["57"] * 6
        + 1
    )
    assert mask[h10_city_lay_idx] == 1.0
    assert mask[h10_city_lay_idx + 3] == 1.0

    h14_7_lay_idx = (
        action_mapper.action_offsets["LayTile"]
        + action_mapper.hex_offsets["H14"] * 6 * len(action_mapper.tile_offsets)
        + action_mapper.tile_offsets["7"] * 6
    )
    assert mask[h14_7_lay_idx] == 1.0
    assert mask[h14_7_lay_idx + 1] == 1.0

    h14_8_lay_idx = (
        action_mapper.action_offsets["LayTile"]
        + action_mapper.hex_offsets["H14"] * 6 * len(action_mapper.tile_offsets)
        + action_mapper.tile_offsets["8"] * 6
        + 1
    )
    assert mask[h14_8_lay_idx] == 1.0
    assert mask[h14_8_lay_idx + 4] == 1.0

    h14_9_lay_idx = (
        action_mapper.action_offsets["LayTile"]
        + action_mapper.hex_offsets["H14"] * 6 * len(action_mapper.tile_offsets)
        + action_mapper.tile_offsets["9"] * 6
        + 1
    )
    assert mask[h14_9_lay_idx] == 1.0
    assert mask[h14_9_lay_idx + 3] == 1.0

    assert sum(mask) == 9.0

    # Check the other direction
    indices = [i for i, mask in enumerate(mask) if mask == 1.0]
    mapped_actions = [action_mapper.map_index_to_action(i, operating_round_1_game_state) for i in indices]
    for action in mapped_actions:
        check_action_in_all_actions(action, all_actions)

    # Test tile lay
    operating_round_1_game_state.process_action(
        action_helper.get_all_choices(operating_round_1_game_state)[0]
    )  # lays tile #57 with rotation 1 on H10

    # Test token options
    all_actions = action_helper.get_all_choices_limited(operating_round_1_game_state)
    for action in all_actions:
        assert action_mapper.get_index_for_action(action) == get_expected_index_for_action(action_mapper, action)

    mask = action_mapper.get_legal_action_mask(operating_round_1_game_state)
    assert mask.shape == (26535,)
    assert mask.dtype == np.float32
    assert mask[0] == 1.0
    token_idx = action_mapper.action_offsets["PlaceToken"] + action_mapper.city_offsets[("H10", 0)]
    assert mask[token_idx] == 1.0
    assert sum(mask) == 2.0

    # Check the other direction
    indices = [i for i, mask in enumerate(mask) if mask == 1.0]
    mapped_actions = [action_mapper.map_index_to_action(i, operating_round_1_game_state) for i in indices]
    for action in mapped_actions:
        check_action_in_all_actions(action, all_actions)

    # Move to trains
    operating_round_1_game_state.process_action(action_helper.get_all_choices(operating_round_1_game_state)[-1])  # passes place token

    # Test train options
    all_actions = action_helper.get_all_choices_limited(operating_round_1_game_state)
    for action in all_actions:
        assert action_mapper.get_index_for_action(action) == get_expected_index_for_action(action_mapper, action)

    mask = action_mapper.get_legal_action_mask(operating_round_1_game_state)
    assert mask.shape == (26535,)
    assert mask.dtype == np.float32
    assert mask[0] == 0.0
    train_idx = action_mapper.action_offsets["BuyTrain"] + action_mapper.train_type_offsets["2"]
    assert mask[train_idx] == 1.0
    assert sum(mask) == 1.0

    # Check the other direction
    indices = [i for i, mask in enumerate(mask) if mask == 1.0]
    mapped_actions = [action_mapper.map_index_to_action(i, operating_round_1_game_state) for i in indices]
    for action in mapped_actions:
        check_action_in_all_actions(action, all_actions)

    operating_round_1_game_state.process_action(action_helper.get_all_choices(operating_round_1_game_state)[0])  # buys a 2 train

    # Check again
    all_actions = action_helper.get_all_choices_limited(operating_round_1_game_state)
    for action in all_actions:
        assert action_mapper.get_index_for_action(action) == get_expected_index_for_action(action_mapper, action)

    mask = action_mapper.get_legal_action_mask(operating_round_1_game_state)
    assert mask.shape == (26535,)
    assert mask.dtype == np.float32
    assert mask[0] == 1.0
    train_idx = action_mapper.action_offsets["BuyTrain"] + action_mapper.train_type_offsets["2"]
    assert mask[train_idx] == 1.0
    assert sum(mask) == 2.0

    # Check the other direction
    indices = [i for i, mask in enumerate(mask) if mask == 1.0]
    mapped_actions = [action_mapper.map_index_to_action(i, operating_round_1_game_state) for i in indices]
    for action in mapped_actions:
        check_action_in_all_actions(action, all_actions)

    # Move to NYC
    operating_round_1_game_state.process_action(action_helper.get_all_choices(operating_round_1_game_state)[0])  # buys a 2 train
    operating_round_1_game_state.process_action(action_helper.get_all_choices(operating_round_1_game_state)[-1])  # passes trains

    # NYC
    all_actions = action_helper.get_all_choices_limited(operating_round_1_game_state)
    for action in all_actions:
        assert action_mapper.get_index_for_action(action) == get_expected_index_for_action(action_mapper, action)

    mask = action_mapper.get_legal_action_mask(operating_round_1_game_state)
    assert mask.shape == (26535,)
    assert mask.dtype == np.float32
    assert mask[0] == 1.0

    # Check the other direction
    indices = [i for i, mask in enumerate(mask) if mask == 1.0]
    mapped_actions = [action_mapper.map_index_to_action(i, operating_round_1_game_state) for i in indices]
    for action in mapped_actions:
        check_action_in_all_actions(action, all_actions)

    e19_city_lay_idx = (
        action_mapper.action_offsets["LayTile"]
        + action_mapper.hex_offsets["E19"] * 6 * len(action_mapper.tile_offsets)
        + action_mapper.tile_offsets["57"] * 6
    )
    assert mask[e19_city_lay_idx] == 1.0
    assert mask[e19_city_lay_idx + 1] == 1.0
    assert mask[e19_city_lay_idx + 2] == 1.0
    assert mask[e19_city_lay_idx + 3] == 1.0
    assert mask[e19_city_lay_idx + 4] == 1.0
    assert mask[e19_city_lay_idx + 5] == 1.0

    assert sum(mask) == 7.0

    # Check the other direction
    indices = [i for i, mask in enumerate(mask) if mask == 1.0]
    mapped_actions = [action_mapper.map_index_to_action(i, operating_round_1_game_state) for i in indices]
    for action in mapped_actions:
        check_action_in_all_actions(action, all_actions)

    # Don't need to test C&O


def test_stock_round_2_game_state(stock_round_2_game_state):
    action_mapper = ActionMapper()
    action_helper = ActionHelper()

    # Test initial stock round 2 state
    all_actions = action_helper.get_all_choices_limited(stock_round_2_game_state)
    for action in all_actions:
        assert action_mapper.get_index_for_action(action) == get_expected_index_for_action(action_mapper, action)

    mask = action_mapper.get_legal_action_mask(stock_round_2_game_state)
    assert mask.shape == (26535,)
    assert mask.dtype == np.float32
    # Legal actions: Pass, Sell NYC (1-5), Buy C%O IPO
    assert mask[0] == 1.0

    # Buy C&O (ipo only)
    assert mask[63] == 1.0
    assert mask[64] == 0.0

    # Sell NYC (1-5)
    assert mask[76] == 1.0
    assert mask[77] == 1.0
    assert mask[78] == 1.0
    assert mask[79] == 1.0
    assert mask[80] == 1.0

    assert sum(mask) == 7.0

    # Check the other direction
    indices = [i for i, mask in enumerate(mask) if mask == 1.0]
    mapped_actions = [action_mapper.map_index_to_action(i, stock_round_2_game_state) for i in indices]
    for action in mapped_actions:
        check_action_in_all_actions(action, all_actions)

    # Sell 2 NYC
    stock_round_2_game_state.process_action(action_helper.get_all_choices(stock_round_2_game_state)[2])  # sells 2 NYC

    # Test MH exchange
    all_actions = action_helper.get_all_choices_limited(stock_round_2_game_state)
    for action in all_actions:
        assert action_mapper.get_index_for_action(action) == get_expected_index_for_action(action_mapper, action)

    mask = action_mapper.get_legal_action_mask(stock_round_2_game_state)
    assert mask.shape == (26535,)
    assert mask.dtype == np.float32
    # Legal actions: Pass, Sell NYC (1-3), Buy C&O IPO
    assert mask[0] == 1.0

    # Par all un-parred companies (not at 100 share price)
    # Par CPR
    assert mask[19] == 1.0
    assert mask[20] == 1.0
    assert mask[21] == 1.0
    assert mask[22] == 1.0
    assert mask[23] == 1.0
    # Par ERIE
    assert mask[37] == 1.0
    assert mask[38] == 1.0
    assert mask[39] == 1.0
    assert mask[40] == 1.0
    assert mask[41] == 1.0
    # Par NYNH
    assert mask[43] == 1.0
    assert mask[44] == 1.0
    assert mask[45] == 1.0
    assert mask[46] == 1.0
    assert mask[47] == 1.0
    # Par B&M
    assert mask[49] == 1.0
    assert mask[50] == 1.0
    assert mask[51] == 1.0
    assert mask[52] == 1.0
    assert mask[53] == 1.0

    # Buy B&O (IPO)
    assert mask[61] == 1.0
    assert mask[62] == 0.0

    # Buy C&O (ipo only)
    assert mask[63] == 1.0
    assert mask[64] == 0.0

    # Sell NYC (1-5)
    assert mask[76] == 1.0
    assert mask[77] == 1.0
    assert mask[78] == 1.0
    assert mask[79] == 0.0
    assert mask[80] == 0.0

    # Exchange MH for NYC
    idx = action_mapper.action_offsets["CompanyBuyShares"]
    assert mask[idx] == 1.0
    assert mask[idx + 1] == 1.0

    assert sum(mask) == 28.0

    # Check the other direction
    indices = [i for i, mask in enumerate(mask) if mask == 1.0]
    mapped_actions = [action_mapper.map_index_to_action(i, stock_round_2_game_state) for i in indices]
    for action in mapped_actions:
        check_action_in_all_actions(action, all_actions)

    # Test MH exchange
    stock_round_2_game_state.process_action(action_helper.get_all_choices(stock_round_2_game_state)[2])  # exchange MH for NYC IPO

    all_actions = action_helper.get_all_choices_limited(stock_round_2_game_state)
    for action in all_actions:
        assert action_mapper.get_index_for_action(action) == get_expected_index_for_action(action_mapper, action)

    mask = action_mapper.get_legal_action_mask(stock_round_2_game_state)
    assert mask.shape == (26535,)
    assert mask.dtype == np.float32
    # Legal actions: Pass, Sell NYC (1-3), Buy C&O IPO
    assert mask[0] == 1.0

    # Par all un-parred companies (not at 100 share price)
    # Par CPR
    assert mask[19] == 1.0
    assert mask[20] == 1.0
    assert mask[21] == 1.0
    assert mask[22] == 1.0
    assert mask[23] == 1.0
    # Par ERIE
    assert mask[37] == 1.0
    assert mask[38] == 1.0
    assert mask[39] == 1.0
    assert mask[40] == 1.0
    assert mask[41] == 1.0
    # Par NYNH
    assert mask[43] == 1.0
    assert mask[44] == 1.0
    assert mask[45] == 1.0
    assert mask[46] == 1.0
    assert mask[47] == 1.0
    # Par B&M
    assert mask[49] == 1.0
    assert mask[50] == 1.0
    assert mask[51] == 1.0
    assert mask[52] == 1.0
    assert mask[53] == 1.0

    # Buy B&O (IPO)
    assert mask[61] == 1.0
    assert mask[62] == 0.0

    # Buy C&O (ipo only)
    assert mask[63] == 1.0
    assert mask[64] == 0.0

    # Sell NYC (1-5)
    assert mask[76] == 1.0
    assert mask[77] == 1.0
    assert mask[78] == 1.0
    assert mask[79] == 0.0
    assert mask[80] == 0.0

    assert sum(mask) == 26.0

    # Check the other direction
    indices = [i for i, mask in enumerate(mask) if mask == 1.0]
    mapped_actions = [action_mapper.map_index_to_action(i, stock_round_2_game_state) for i in indices]
    for action in mapped_actions:
        check_action_in_all_actions(action, all_actions)

    # Don't really need to check the rest


def test_operating_round_2_game_state(operating_round_2_game_state):
    action_mapper = ActionMapper()
    action_helper = ActionHelper()

    # Test initial operating round 2 state
    all_actions = action_helper.get_all_choices_limited(operating_round_2_game_state)
    for action in all_actions:
        assert action_mapper.get_index_for_action(action) == get_expected_index_for_action(action_mapper, action)

    mask = action_mapper.get_legal_action_mask(operating_round_2_game_state)
    assert mask.shape == (26535,)
    assert mask.dtype == np.float32
    # Legal actions: Pass, Lay tile on F20
    assert mask[0] == 1.0

    # can lay #1, #2, #55, #56, and #69 on F20
    lay_1_idx = (
        action_mapper.action_offsets["LayTile"]
        + action_mapper.hex_offsets["F20"] * 6 * len(action_mapper.tile_offsets)
        + action_mapper.tile_offsets["1"] * 6
    )
    lay_2_idx = (
        action_mapper.action_offsets["LayTile"]
        + action_mapper.hex_offsets["F20"] * 6 * len(action_mapper.tile_offsets)
        + action_mapper.tile_offsets["2"] * 6
    )
    lay_55_idx = (
        action_mapper.action_offsets["LayTile"]
        + action_mapper.hex_offsets["F20"] * 6 * len(action_mapper.tile_offsets)
        + action_mapper.tile_offsets["55"] * 6
    )
    lay_56_idx = (
        action_mapper.action_offsets["LayTile"]
        + action_mapper.hex_offsets["F20"] * 6 * len(action_mapper.tile_offsets)
        + action_mapper.tile_offsets["56"] * 6
    )
    lay_69_idx = (
        action_mapper.action_offsets["LayTile"]
        + action_mapper.hex_offsets["F20"] * 6 * len(action_mapper.tile_offsets)
        + action_mapper.tile_offsets["69"] * 6
    )

    assert mask[lay_1_idx] == 1.0
    assert mask[lay_1_idx + 3] == 1.0
    assert mask[lay_2_idx] == 1.0
    assert mask[lay_55_idx] == 1.0
    assert mask[lay_55_idx + 3] == 1.0
    assert mask[lay_56_idx] == 1.0
    assert mask[lay_69_idx] == 1.0
    assert mask[lay_69_idx + 4] == 1.0

    assert sum(mask) == 9.0

    # Check the other direction
    indices = [i for i, mask in enumerate(mask) if mask == 1.0]
    mapped_actions = [action_mapper.map_index_to_action(i, operating_round_2_game_state) for i in indices]
    for action in mapped_actions:
        check_action_in_all_actions(action, all_actions)

    # NYNH
    operating_round_2_game_state.process_action(action_helper.get_all_choices(operating_round_2_game_state)[0])  # lay #1 with rotation 0 on F20
    operating_round_2_game_state.process_action(action_helper.get_all_choices(operating_round_2_game_state)[0])  # buy 2 train
    operating_round_2_game_state.process_action(action_helper.get_all_choices(operating_round_2_game_state)[-1])  # pass trains

    # PRR
    operating_round_2_game_state.process_action(
        action_helper.get_all_choices(operating_round_2_game_state)[10]
    )  # lay tile #9 with rotation 1 on H8
    operating_round_2_game_state.process_action(action_helper.get_all_choices(operating_round_2_game_state)[-1])  # pass token
    operating_round_2_game_state.process_action(action_helper.get_all_choices(operating_round_2_game_state)[0])  # auto trains & run

    # Test dividend
    all_actions = action_helper.get_all_choices_limited(operating_round_2_game_state)
    for action in all_actions:
        assert action_mapper.get_index_for_action(action) == get_expected_index_for_action(action_mapper, action)

    mask = action_mapper.get_legal_action_mask(operating_round_2_game_state)
    assert mask.shape == (26535,)
    assert mask.dtype == np.float32
    # Legal actions: Pay out or withhold

    dividend_idx = action_mapper.action_offsets["Dividend"]
    assert mask[dividend_idx] == 1.0
    assert mask[dividend_idx + 1] == 1.0

    assert sum(mask) == 2.0

    # Check the other direction
    indices = [i for i, mask in enumerate(mask) if mask == 1.0]
    mapped_actions = [action_mapper.map_index_to_action(i, operating_round_2_game_state) for i in indices]
    for action in mapped_actions:
        check_action_in_all_actions(action, all_actions)

    # Move to C&O buy 3 step
    operating_round_2_game_state.process_action(action_helper.get_all_choices(operating_round_2_game_state)[0])  # pay out
    operating_round_2_game_state.process_action(action_helper.get_all_choices(operating_round_2_game_state)[-1])  # pass trains

    # C&O
    operating_round_2_game_state.process_action(action_helper.get_all_choices(operating_round_2_game_state)[2])  # lay tile #8 with rotation 2 on G3
    operating_round_2_game_state.process_action(action_helper.get_all_choices(operating_round_2_game_state)[0])  # auto trains & run
    operating_round_2_game_state.process_action(action_helper.get_all_choices(operating_round_2_game_state)[1])  # withhold
    operating_round_2_game_state.process_action(action_helper.get_all_choices(operating_round_2_game_state)[0])  # buy a 2 train

    # Test 3 train purchase
    all_actions = action_helper.get_all_choices_limited(operating_round_2_game_state)
    for action in all_actions:
        assert action_mapper.get_index_for_action(action) == get_expected_index_for_action(action_mapper, action)

    mask = action_mapper.get_legal_action_mask(operating_round_2_game_state)
    assert mask.shape == (26535,)
    assert mask.dtype == np.float32
    # Legal actions: Pass, Buy 3 train
    assert mask[0] == 1.0
    buy_train_idx = action_mapper.action_offsets["BuyTrain"]
    assert mask[buy_train_idx] == 1.0
    assert sum(mask) == 2.0

    # Check the other direction
    indices = [i for i, mask in enumerate(mask) if mask == 1.0]
    mapped_actions = [action_mapper.map_index_to_action(i, operating_round_2_game_state) for i in indices]
    for action in mapped_actions:
        check_action_in_all_actions(action, all_actions)

    # Move on
    operating_round_2_game_state.process_action(action_helper.get_all_choices(operating_round_2_game_state)[0])  # buy a 3 train
    operating_round_2_game_state.process_action(action_helper.get_all_choices(operating_round_2_game_state)[-1])  # pass trains

    # Test buy company
    all_actions = action_helper.get_all_choices_limited(operating_round_2_game_state)
    for action in all_actions:
        assert action_mapper.get_index_for_action(action) == get_expected_index_for_action(action_mapper, action)

    mask = action_mapper.get_legal_action_mask(operating_round_2_game_state)
    assert mask.shape == (26535,)
    assert mask.dtype == np.float32
    # Legal actions: Buy
    assert mask[0] == 1.0
    buy_company_idx = action_mapper.action_offsets["BuyCompany"]
    assert (
        mask[buy_company_idx + action_mapper.company_offsets["DH"] * len(action_mapper.buy_company_price_offsets)]
        == 1.0
    )  # min
    assert (
        mask[buy_company_idx + action_mapper.company_offsets["DH"] * len(action_mapper.buy_company_price_offsets) + 1]
        == 1.0
    )  # max
    assert sum(mask) == 3.0

    # Check the other direction
    indices = [i for i, mask in enumerate(mask) if mask == 1.0]
    mapped_actions = [action_mapper.map_index_to_action(i, operating_round_2_game_state) for i in indices]
    for action in mapped_actions:
        check_action_in_all_actions(action, all_actions)

    # Move to NYC
    operating_round_2_game_state.process_action(action_helper.get_all_choices(operating_round_2_game_state)[-2])  # buy DH from Player 2 for $140
    operating_round_2_game_state.process_action(action_helper.get_all_choices(operating_round_2_game_state)[0])  # pass buy companies

    # Skip to next OR
    operating_round_2_game_state.process_action(
        action_helper.get_all_choices(operating_round_2_game_state)[46]
    )  # lay tile #8 with rotation 3 on F18
    operating_round_2_game_state.process_action(action_helper.get_all_choices(operating_round_2_game_state)[31])  # buy 3 train
    operating_round_2_game_state.process_action(action_helper.get_all_choices(operating_round_2_game_state)[31])  # buy 3 train
    operating_round_2_game_state.process_action(action_helper.get_all_choices(operating_round_2_game_state)[31])  # buy 3 train
    operating_round_2_game_state.process_action(action_helper.get_all_choices(operating_round_2_game_state)[-1])  # pass
    # SR 3
    operating_round_2_game_state.process_action(action_helper.get_all_choices(operating_round_2_game_state)[-1])  # pass
    operating_round_2_game_state.process_action(action_helper.get_all_choices(operating_round_2_game_state)[-1])  # pass
    operating_round_2_game_state.process_action(action_helper.get_all_choices(operating_round_2_game_state)[-1])  # pass
    operating_round_2_game_state.process_action(action_helper.get_all_choices(operating_round_2_game_state)[-1])  # pass

    operating_round_2_game_state.process_action(action_helper.get_all_choices(operating_round_2_game_state)[45])  # lay tile 8 rot 2 on H6
    operating_round_2_game_state.process_action(action_helper.get_all_choices(operating_round_2_game_state)[-1])  # skip token
    operating_round_2_game_state.process_action(action_helper.get_all_choices(operating_round_2_game_state)[-1])  # auto routes
    operating_round_2_game_state.process_action(action_helper.get_all_choices(operating_round_2_game_state)[-2])  # pay out

    # Check cross-company BuyTrain
    all_actions = action_helper.get_all_choices_limited(operating_round_2_game_state)
    for action in all_actions:
        assert action_mapper.get_index_for_action(action) == get_expected_index_for_action(action_mapper, action)

    mask = action_mapper.get_legal_action_mask(operating_round_2_game_state)
    assert mask.shape == (26535,)
    assert mask.dtype == np.float32
    # Legal actions:
    # Buy SV min/max,
    # Buy NYC 2 at all prices up to 500 (and all-but-one and all),
    # Buy NYC 3 at all prices up to 500 (and all-but-one and all),
    # Pass
    assert mask[0] == 1.0
    buy_company_idx = action_mapper.action_offsets["BuyCompany"]
    assert (
        mask[buy_company_idx + action_mapper.company_offsets["SV"] * len(action_mapper.buy_company_price_offsets)]
        == 1.0
    )  # min
    assert (
        mask[buy_company_idx + action_mapper.company_offsets["SV"] * len(action_mapper.buy_company_price_offsets) + 1]
        == 1.0
    )  # max

    buy_train_idx = (
        action_mapper.action_offsets["BuyTrain"]
        + 1
        + len(action_mapper.train_type_offsets)
        + action_mapper.corporation_offsets["NYC"]
        * len(action_mapper.train_type_offsets)
        * len(action_mapper.train_price_offsets)
    )
    assert mask[buy_train_idx] == 1.0

    buy_2_train_idx = buy_train_idx + action_mapper.train_type_offsets["2"] * len(action_mapper.train_price_offsets)
    assert mask[buy_2_train_idx + action_mapper.train_price_offsets["1"]] == 1.0
    assert mask[buy_2_train_idx + action_mapper.train_price_offsets["20"]] == 1.0
    assert mask[buy_2_train_idx + action_mapper.train_price_offsets["50"]] == 1.0
    assert mask[buy_2_train_idx + action_mapper.train_price_offsets["100"]] == 1.0
    assert mask[buy_2_train_idx + action_mapper.train_price_offsets["200"]] == 1.0
    assert mask[buy_2_train_idx + action_mapper.train_price_offsets["300"]] == 1.0
    assert mask[buy_2_train_idx + action_mapper.train_price_offsets["400"]] == 1.0
    assert mask[buy_2_train_idx + action_mapper.train_price_offsets["500"]] == 1.0
    assert mask[buy_2_train_idx + action_mapper.train_price_offsets["all-but-one"]] == 1.0
    assert mask[buy_2_train_idx + action_mapper.train_price_offsets["all"]] == 1.0

    buy_3_train_idx = buy_train_idx + action_mapper.train_type_offsets["3"] * len(action_mapper.train_price_offsets)
    assert mask[buy_3_train_idx + action_mapper.train_price_offsets["1"]] == 1.0
    assert mask[buy_3_train_idx + action_mapper.train_price_offsets["20"]] == 1.0
    assert mask[buy_3_train_idx + action_mapper.train_price_offsets["50"]] == 1.0
    assert mask[buy_3_train_idx + action_mapper.train_price_offsets["100"]] == 1.0
    assert mask[buy_3_train_idx + action_mapper.train_price_offsets["200"]] == 1.0
    assert mask[buy_3_train_idx + action_mapper.train_price_offsets["300"]] == 1.0
    assert mask[buy_3_train_idx + action_mapper.train_price_offsets["400"]] == 1.0
    assert mask[buy_3_train_idx + action_mapper.train_price_offsets["500"]] == 1.0
    assert mask[buy_3_train_idx + action_mapper.train_price_offsets["all-but-one"]] == 1.0
    assert mask[buy_3_train_idx + action_mapper.train_price_offsets["all"]] == 1.0
    assert sum(mask) == 24.0

    # Check the other direction
    indices = [i for i, mask in enumerate(mask) if mask == 1.0]
    mapped_actions = [action_mapper.map_index_to_action(i, operating_round_2_game_state) for i in indices]
    for action in mapped_actions:
        check_action_in_all_actions(action, all_actions)

    operating_round_2_game_state.process_action(action_helper.get_all_choices_limited(operating_round_2_game_state)[11])  # Buy NYC 2 509
    operating_round_2_game_state.process_action(action_helper.get_all_choices(operating_round_2_game_state)[-1])  # pass trains
    operating_round_2_game_state.process_action(
        action_helper.get_all_choices(operating_round_2_game_state)[4]
    )  # NYNH spends $80 and lays tile #57 with rotation 1 on F22 (Providence)
    operating_round_2_game_state.process_action(action_helper.get_all_choices(operating_round_2_game_state)[-1])  # skip token
    operating_round_2_game_state.process_action(action_helper.get_all_choices(operating_round_2_game_state)[-1])  # auto routes
    operating_round_2_game_state.process_action(action_helper.get_all_choices_limited(operating_round_2_game_state)[2])  # pay out

    # Check company tile lay
    all_actions = action_helper.get_all_choices_limited(operating_round_2_game_state)
    for action in all_actions:
        assert action_mapper.get_index_for_action(action) == get_expected_index_for_action(action_mapper, action)

    mask = action_mapper.get_legal_action_mask(operating_round_2_game_state)
    assert mask.shape == (26535,)
    assert mask.dtype == np.float32
    # Legal actions:
    # Buy MH min/max,
    # Buy 3 train from depot,
    # Lay tile 3,4, 58 on B20 with various rotations,
    # Pass
    assert mask[0] == 1.0

    buy_company_idx = action_mapper.action_offsets["BuyCompany"]
    assert (
        mask[buy_company_idx + action_mapper.company_offsets["MH"] * len(action_mapper.buy_company_price_offsets)]
        == 1.0
    )  # min
    assert (
        mask[buy_company_idx + action_mapper.company_offsets["MH"] * len(action_mapper.buy_company_price_offsets) + 1]
        == 1.0
    )  # max

    assert mask[action_mapper.action_offsets["BuyTrain"]] == 1.0

    company_lay_tile_idx = action_mapper.action_offsets["CompanyLayTile"] + 6
    tile_3_idx = company_lay_tile_idx + action_mapper.company_tile_offsets["3"] * 6
    assert mask[tile_3_idx] == 1.0
    assert mask[tile_3_idx + 1] == 1.0
    assert mask[tile_3_idx + 4] == 1.0
    assert mask[tile_3_idx + 5] == 1.0

    tile_4_idx = company_lay_tile_idx + action_mapper.company_tile_offsets["4"] * 6
    assert mask[tile_4_idx + 1] == 1.0
    assert mask[tile_4_idx + 2] == 1.0
    assert mask[tile_4_idx + 4] == 1.0
    assert mask[tile_4_idx + 5] == 1.0
    tile_58_idx = company_lay_tile_idx + action_mapper.company_tile_offsets["58"] * 6
    assert mask[tile_58_idx] == 1.0
    assert mask[tile_58_idx + 2] == 1.0
    assert mask[tile_58_idx + 4] == 1.0
    assert mask[tile_58_idx + 5] == 1.0

    assert sum(mask) == 16.0

    # Check the other direction
    indices = [i for i, mask in enumerate(mask) if mask == 1.0]
    mapped_actions = [action_mapper.map_index_to_action(i, operating_round_2_game_state) for i in indices]
    for action in mapped_actions:
        check_action_in_all_actions(action, all_actions)

    operating_round_2_game_state.process_action(action_helper.get_all_choices(operating_round_2_game_state)[-1])  # skip trains
    operating_round_2_game_state.process_action(action_helper.get_all_choices(operating_round_2_game_state)[-1])  # skip companies

    # Check company tile lay again
    all_actions = action_helper.get_all_choices_limited(operating_round_2_game_state)
    for action in all_actions:
        assert action_mapper.get_index_for_action(action) == get_expected_index_for_action(action_mapper, action)

    mask = action_mapper.get_legal_action_mask(operating_round_2_game_state)
    assert mask.shape == (26535,)
    assert mask.dtype == np.float32
    # Legal actions:
    # Lay tile 16, 19, 23, 25, 28, 29 on G3
    # Lay tile 16, 19, 23, 24, 25, 28, 29 on G5
    # Lay tile 2, 55, 56, 69 on G7
    # DH Lay 57 on F16
    # Pass
    assert mask[0] == 1.0

    lay_tile_idx = action_mapper.action_offsets["LayTile"]
    hex_g3_idx = lay_tile_idx + action_mapper.hex_offsets["G3"] * len(action_mapper.tile_offsets) * 6
    hex_g5_idx = lay_tile_idx + action_mapper.hex_offsets["G5"] * len(action_mapper.tile_offsets) * 6
    hex_g7_idx = lay_tile_idx + action_mapper.hex_offsets["G7"] * len(action_mapper.tile_offsets) * 6
    assert mask[hex_g3_idx + action_mapper.tile_offsets["16"] * 6 + 2] == 1.0
    assert mask[hex_g3_idx + action_mapper.tile_offsets["19"] * 6] == 1.0
    assert mask[hex_g3_idx + action_mapper.tile_offsets["24"] * 6 + 2] == 1.0
    assert mask[hex_g3_idx + action_mapper.tile_offsets["25"] * 6 + 2] == 1.0
    assert mask[hex_g3_idx + action_mapper.tile_offsets["25"] * 6 + 4] == 1.0
    assert mask[hex_g3_idx + action_mapper.tile_offsets["28"] * 6 + 4] == 1.0
    assert mask[hex_g3_idx + action_mapper.tile_offsets["29"] * 6 + 2] == 1.0

    assert mask[hex_g5_idx + action_mapper.tile_offsets["16"] * 6] == 1.0
    assert mask[hex_g5_idx + action_mapper.tile_offsets["16"] * 6 + 1] == 1.0
    assert mask[hex_g5_idx + action_mapper.tile_offsets["19"] * 6 + 5] == 1.0
    assert mask[hex_g5_idx + action_mapper.tile_offsets["23"] * 6 + 3] == 1.0
    assert mask[hex_g5_idx + action_mapper.tile_offsets["24"] * 6 + 1] == 1.0
    assert mask[hex_g5_idx + action_mapper.tile_offsets["25"] * 6 + 1] == 1.0
    assert mask[hex_g5_idx + action_mapper.tile_offsets["25"] * 6 + 3] == 1.0
    assert mask[hex_g5_idx + action_mapper.tile_offsets["28"] * 6 + 3] == 1.0
    assert mask[hex_g5_idx + action_mapper.tile_offsets["29"] * 6 + 1] == 1.0

    assert mask[hex_g7_idx + action_mapper.tile_offsets["2"] * 6] == 1.0
    assert mask[hex_g7_idx + action_mapper.tile_offsets["2"] * 6 + 1] == 1.0
    assert mask[hex_g7_idx + action_mapper.tile_offsets["2"] * 6 + 2] == 1.0
    assert mask[hex_g7_idx + action_mapper.tile_offsets["2"] * 6 + 5] == 1.0
    assert mask[hex_g7_idx + action_mapper.tile_offsets["55"] * 6 + 1] == 1.0
    assert mask[hex_g7_idx + action_mapper.tile_offsets["55"] * 6 + 2] == 1.0
    assert mask[hex_g7_idx + action_mapper.tile_offsets["55"] * 6 + 4] == 1.0
    assert mask[hex_g7_idx + action_mapper.tile_offsets["55"] * 6 + 5] == 1.0
    assert mask[hex_g7_idx + action_mapper.tile_offsets["56"] * 6] == 1.0
    assert mask[hex_g7_idx + action_mapper.tile_offsets["56"] * 6 + 1] == 1.0
    assert mask[hex_g7_idx + action_mapper.tile_offsets["56"] * 6 + 2] == 1.0
    assert mask[hex_g7_idx + action_mapper.tile_offsets["56"] * 6 + 5] == 1.0
    assert mask[hex_g7_idx + action_mapper.tile_offsets["69"] * 6] == 1.0
    assert mask[hex_g7_idx + action_mapper.tile_offsets["69"] * 6 + 2] == 1.0
    assert mask[hex_g7_idx + action_mapper.tile_offsets["69"] * 6 + 4] == 1.0
    assert mask[hex_g7_idx + action_mapper.tile_offsets["69"] * 6 + 5] == 1.0

    company_lay_tile_idx = action_mapper.action_offsets["CompanyLayTile"]
    assert mask[company_lay_tile_idx] == 1.0
    assert mask[company_lay_tile_idx + 1] == 1.0
    assert mask[company_lay_tile_idx + 2] == 1.0
    assert mask[company_lay_tile_idx + 3] == 1.0
    assert mask[company_lay_tile_idx + 4] == 1.0
    assert mask[company_lay_tile_idx + 5] == 1.0
    assert sum(mask) == 39.0

    operating_round_2_game_state.process_action(
        action_helper.get_all_choices(operating_round_2_game_state)[34]
    )  # [17:12] C&O (DH) spends $120 and lays tile #57 with rotation 2 on F16 (Scranton)

    # Check company token placement
    all_actions = action_helper.get_all_choices_limited(operating_round_2_game_state)
    for action in all_actions:
        assert action_mapper.get_index_for_action(action) == get_expected_index_for_action(action_mapper, action)

    mask = action_mapper.get_legal_action_mask(operating_round_2_game_state)
    assert mask.shape == (26535,)
    assert mask.dtype == np.float32
    # Legal actions:
    # DH Place token F16
    # Pass
    assert mask[0] == 1.0

    company_place_token_idx = action_mapper.action_offsets["CompanyPlaceToken"]
    assert mask[company_place_token_idx] == 1.0
    assert sum(mask) == 2.0

    operating_round_2_game_state.process_action(
        action_helper.get_all_choices(operating_round_2_game_state)[0]
    )  # [17:13] C&O (DH) places a token on F16 (Scranton)

    operating_round_2_game_state.process_action(action_helper.get_all_choices(operating_round_2_game_state)[-1])  # auto routes
    operating_round_2_game_state.process_action(action_helper.get_all_choices(operating_round_2_game_state)[0])  # pay out
    operating_round_2_game_state.process_action(action_helper.get_all_choices(operating_round_2_game_state)[-1])  # skip trains

    # Test tile upgrade
    all_actions = action_helper.get_all_choices_limited(operating_round_2_game_state)
    for action in all_actions:
        assert action_mapper.get_index_for_action(action) == get_expected_index_for_action(action_mapper, action)

    mask = action_mapper.get_legal_action_mask(operating_round_2_game_state)
    assert mask.shape == (26535,)
    assert mask.dtype == np.float32
    # Legal actions:
    # Buy SV min/max,
    # On D20, lay 7, 8, or 9
    # On E19, lay 14 or 15
    # On F18, lay 16, 19, 23, 24, 25, 28, 29
    # On G19, lay 54
    # Pass
    assert mask[0] == 1.0

    buy_company_idx = action_mapper.action_offsets["BuyCompany"]
    assert (
        mask[buy_company_idx + action_mapper.company_offsets["SV"] * len(action_mapper.buy_company_price_offsets)]
        == 1.0
    )  # min
    assert (
        mask[buy_company_idx + action_mapper.company_offsets["SV"] * len(action_mapper.buy_company_price_offsets) + 1]
        == 1.0
    )  # max

    lay_tile_idx = action_mapper.action_offsets["LayTile"]
    hex_d20_idx = lay_tile_idx + action_mapper.hex_offsets["D20"] * len(action_mapper.tile_offsets) * 6
    hex_e19_idx = lay_tile_idx + action_mapper.hex_offsets["E19"] * len(action_mapper.tile_offsets) * 6
    hex_f18_idx = lay_tile_idx + action_mapper.hex_offsets["F18"] * len(action_mapper.tile_offsets) * 6
    hex_g19_idx = lay_tile_idx + action_mapper.hex_offsets["G19"] * len(action_mapper.tile_offsets) * 6
    assert mask[hex_d20_idx + action_mapper.tile_offsets["7"] * 6 + 0] == 1.0
    assert mask[hex_d20_idx + action_mapper.tile_offsets["7"] * 6 + 5] == 1.0
    assert mask[hex_d20_idx + action_mapper.tile_offsets["8"] * 6 + 0] == 1.0
    assert mask[hex_d20_idx + action_mapper.tile_offsets["8"] * 6 + 4] == 1.0
    assert mask[hex_d20_idx + action_mapper.tile_offsets["9"] * 6 + 0] == 1.0
    assert mask[hex_d20_idx + action_mapper.tile_offsets["9"] * 6 + 3] == 1.0

    assert mask[hex_e19_idx + action_mapper.tile_offsets["14"] * 6 + 0] == 1.0
    assert mask[hex_e19_idx + action_mapper.tile_offsets["14"] * 6 + 2] == 1.0
    assert mask[hex_e19_idx + action_mapper.tile_offsets["14"] * 6 + 3] == 1.0
    assert mask[hex_e19_idx + action_mapper.tile_offsets["14"] * 6 + 5] == 1.0
    assert mask[hex_e19_idx + action_mapper.tile_offsets["15"] * 6 + 0] == 1.0
    assert mask[hex_e19_idx + action_mapper.tile_offsets["15"] * 6 + 3] == 1.0

    assert mask[hex_f18_idx + action_mapper.tile_offsets["16"] * 6 + 2] == 1.0
    assert mask[hex_f18_idx + action_mapper.tile_offsets["16"] * 6 + 3] == 1.0
    assert mask[hex_f18_idx + action_mapper.tile_offsets["19"] * 6 + 1] == 1.0
    assert mask[hex_f18_idx + action_mapper.tile_offsets["23"] * 6 + 5] == 1.0
    assert mask[hex_f18_idx + action_mapper.tile_offsets["24"] * 6 + 3] == 1.0
    assert mask[hex_f18_idx + action_mapper.tile_offsets["25"] * 6 + 3] == 1.0
    assert mask[hex_f18_idx + action_mapper.tile_offsets["25"] * 6 + 5] == 1.0
    assert mask[hex_f18_idx + action_mapper.tile_offsets["28"] * 6 + 5] == 1.0
    assert mask[hex_f18_idx + action_mapper.tile_offsets["29"] * 6 + 3] == 1.0

    assert mask[hex_g19_idx + action_mapper.tile_offsets["54"] * 6 + 0] == 1.0

    assert sum(mask) == 25.0

    operating_round_2_game_state.process_action(
        action_helper.get_all_choices_limited(operating_round_2_game_state)[23]
    )  # [17:13] NYC spends $80 and lays tile #54 with rotation 0 on G19 (New York & Newark)
    operating_round_2_game_state.process_action(action_helper.get_all_choices(operating_round_2_game_state)[-1])  # auto routes
    operating_round_2_game_state.process_action(action_helper.get_all_choices_limited(operating_round_2_game_state)[2])  # pay out
    operating_round_2_game_state.process_action(action_helper.get_all_choices_limited(operating_round_2_game_state)[2])  # Buy 3 train
    operating_round_2_game_state.process_action(action_helper.get_all_choices(operating_round_2_game_state)[-1])  # skip companies

    # PRR
    operating_round_2_game_state.process_action(action_helper.get_all_choices_limited(operating_round_2_game_state)[-1])  # skip track
    operating_round_2_game_state.process_action(action_helper.get_all_choices_limited(operating_round_2_game_state)[-1])  # run trains
    operating_round_2_game_state.process_action(action_helper.get_all_choices_limited(operating_round_2_game_state)[0])  # pay out
    operating_round_2_game_state.process_action(action_helper.get_all_choices_limited(operating_round_2_game_state)[-1])  # skip trains

    # NYNH
    operating_round_2_game_state.process_action(action_helper.get_all_choices_limited(operating_round_2_game_state)[-1])  # skip track
    operating_round_2_game_state.process_action(action_helper.get_all_choices_limited(operating_round_2_game_state)[-1])  # skip token
    operating_round_2_game_state.process_action(action_helper.get_all_choices_limited(operating_round_2_game_state)[0])  # run trains
    operating_round_2_game_state.process_action(action_helper.get_all_choices_limited(operating_round_2_game_state)[0])  # pay out
    operating_round_2_game_state.process_action(action_helper.get_all_choices_limited(operating_round_2_game_state)[0])  # buy a 4 train

    # Test discard train
    all_actions = action_helper.get_all_choices_limited(operating_round_2_game_state)
    for action in all_actions:
        assert action_mapper.get_index_for_action(action) == get_expected_index_for_action(action_mapper, action)

    mask = action_mapper.get_legal_action_mask(operating_round_2_game_state)
    assert mask.shape == (26535,)
    assert mask.dtype == np.float32
    # Legal actions:
    # Discard a 3 train

    discard_idx = action_mapper.action_offsets["DiscardTrain"]
    assert mask[discard_idx + action_mapper.train_type_offsets["3"]] == 1.0
    assert sum(mask) == 1.0

    operating_round_2_game_state.process_action(action_helper.get_all_choices_limited(operating_round_2_game_state)[0])  # NYC discard train

    # Test purchase discarded train
    all_actions = action_helper.get_all_choices_limited(operating_round_2_game_state)
    for action in all_actions:
        assert action_mapper.get_index_for_action(action) == get_expected_index_for_action(action_mapper, action)

    mask = action_mapper.get_legal_action_mask(operating_round_2_game_state)
    assert mask.shape == (26535,)
    assert mask.dtype == np.float32
    # Legal actions:
    # CS Lay tile 3, 4, 58 on B20 with various rotations
    # Buy 3 train from open market
    # Pass
    assert mask[0] == 1.0

    buy_train_idx = action_mapper.action_offsets["BuyTrain"]
    assert mask[buy_train_idx + 1 + action_mapper.train_type_offsets["3"]] == 1.0

    company_lay_tile_idx = action_mapper.action_offsets["CompanyLayTile"] + 6
    tile_3_idx = company_lay_tile_idx + action_mapper.company_tile_offsets["3"] * 6
    assert mask[tile_3_idx] == 1.0
    assert mask[tile_3_idx + 1] == 1.0
    assert mask[tile_3_idx + 4] == 1.0
    assert mask[tile_3_idx + 5] == 1.0

    tile_4_idx = company_lay_tile_idx + action_mapper.company_tile_offsets["4"] * 6
    assert mask[tile_4_idx + 1] == 1.0
    assert mask[tile_4_idx + 2] == 1.0
    assert mask[tile_4_idx + 4] == 1.0
    assert mask[tile_4_idx + 5] == 1.0
    tile_58_idx = company_lay_tile_idx + action_mapper.company_tile_offsets["58"] * 6
    assert mask[tile_58_idx] == 1.0
    assert mask[tile_58_idx + 2] == 1.0
    assert mask[tile_58_idx + 4] == 1.0
    assert mask[tile_58_idx + 5] == 1.0

    assert sum(mask) == 14.0


def test_bankruptcy_game_state(bankruptcy_game_state):
    action_mapper = ActionMapper()
    action_helper = ActionHelper()

    # Test bankrupcy action availability
    all_actions = action_helper.get_all_choices_limited(bankruptcy_game_state)
    for action in all_actions:
        assert action_mapper.get_index_for_action(action) == get_expected_index_for_action(action_mapper, action)

    mask = action_mapper.get_legal_action_mask(bankruptcy_game_state)
    assert mask.shape == (26535,)
    assert mask.dtype == np.float32
    assert mask[action_mapper.action_offsets["Bankrupt"]] == 1.0
    assert sum(mask) == 1.0

    # Check the other direction
    indices = [i for i, mask in enumerate(mask) if mask == 1.0]
    mapped_actions = [action_mapper.map_index_to_action(i, bankruptcy_game_state) for i in indices]
    for action in mapped_actions:
        check_action_in_all_actions(action, all_actions)
