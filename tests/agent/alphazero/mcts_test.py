import pytest
import torch
import numpy as np
from typing import List, Any, Tuple
from unittest.mock import MagicMock
# Your project imports
from rl18xx.game.engine.game import BaseGame
from rl18xx.game.gamemap import GameMap
from rl18xx.game.action_helper import ActionHelper
from rl18xx.agent.alphazero.model import Model
from rl18xx.agent.alphazero.encoder import Encoder_1830
from rl18xx.agent.alphazero.action_mapper import ActionMapper
from rl18xx.agent.alphazero.mcts import MCTS, Node

# --- Test Configuration ---
# Model parameters for faster testing
TEST_GAME_STATE_SIZE = 377 # Match your encoder output
TEST_NUM_MAP_NODES = 93    # Match your encoder output
TEST_MAP_NODE_FEATURES = 50 # Match your encoder output
NUM_EDGES = 470 # Number of edges in the graph
TEST_POLICY_OUTPUT_SIZE = 26535
TEST_VALUE_OUTPUT_SIZE = 4 # Assuming a 4-player game
GNN_EDGE_CATEGORIES = 6 # Example: 6 types of hex edges
GNN_EDGE_EMBEDDING_DIM = 32 # Must match a potential model config

# MCTS parameters
TEST_NUM_SIMULATIONS = 8 # Small number for quick tests
TEST_C_PUCT = 5.0

GameObjects = Tuple[BaseGame, ActionHelper, ActionMapper]
# --- Fixtures ---

@pytest.fixture(scope="module")
def model():
    """Instantiates a small version of the Model for testing."""
    model = Model(
        game_state_size=TEST_GAME_STATE_SIZE,
        map_node_features=TEST_MAP_NODE_FEATURES,
        policy_size=TEST_POLICY_OUTPUT_SIZE,
        value_size=TEST_VALUE_OUTPUT_SIZE,
        mlp_hidden_dim=32,
        gnn_node_proj_dim=16,
        gnn_hidden_dim_per_head=8,
        gnn_layers=1,
        gnn_heads=1,
        gnn_output_embed_dim=16,
        gnn_edge_categories=GNN_EDGE_CATEGORIES,
        gnn_edge_embedding_dim=GNN_EDGE_EMBEDDING_DIM,
        shared_trunk_hidden_dim=32,
        num_res_blocks=1,
        dropout_rate=0.0
    )
    model.eval() # Set to eval mode
    return model

@pytest.fixture
def game_objects() -> GameObjects:
    game_map = GameMap()
    game_class = game_map.game_by_title("1830")
    players = {"1": "Player 1", "2": "Player 2", "3": "Player 3", "4": "Player 4"}
    game_instance = game_class(players)
    return (game_instance, ActionHelper(game_instance), ActionMapper(), Encoder_1830())

@pytest.fixture
def terminal_game_state(game_objects) -> GameObjects:
    """
    Provides a BaseGame instance in a known TERMINAL state.
    You'll need to implement this to put your game into a finished state.
    """
    initial_game_state, action_helper, action_mapper, encoder = game_objects
    # Auction
    initial_game_state.process_action(
        action_helper.get_all_choices()[-2]
    )  # [20:39] -- Phase 2 (Operating Rounds: 1 | Train Limit: 4 | Available Tiles: Yellow) --
    # [20:39] Player 1 bids $600 for Baltimore & Ohio
    initial_game_state.process_action(action_helper.get_all_choices()[0])  # [20:39] Player 2 buys Schuylkill Valley for $20
    initial_game_state.process_action(action_helper.get_all_choices()[0])  # [20:39] Player 3 buys Champlain & St.Lawrence for $40
    initial_game_state.process_action(action_helper.get_all_choices()[0])  # [20:39] Player 4 buys Delaware & Hudson for $70
    initial_game_state.process_action(action_helper.get_all_choices()[0])  # [20:39] Player 1 passes bidding
    initial_game_state.process_action(action_helper.get_all_choices()[0])  # [20:39] Player 2 buys Mohawk & Hudson for $110
    initial_game_state.process_action(action_helper.get_all_choices()[0])  # [20:39] Player 3 buys Camden & Amboy for $160
    # [20:39] Player 3 receives a 10% share of PRR
    # [20:39] Player 1 wins the auction for Baltimore & Ohio with the only bid of $600
    initial_game_state.process_action(action_helper.get_all_choices()[-1])  # [20:39] Player 1 pars B&O at $67
    # [20:39] Player 1 receives a 20% share of B&O
    # [20:39] Player 1 becomes the president of B&O
    # [20:39] Player 4 has priority deal
    # [20:39] -- Stock Round 1 --
    initial_game_state.process_action(
        action_helper.get_all_choices()[0]
    )  # [20:39] Player 4 buys a 10% share of B&O from the IPO for $67
    # [20:39] Player 1 has no valid actions and passes
    initial_game_state.process_action(
        action_helper.get_all_choices()[0]
    )  # [21:13] Player 2 buys a 10% share of B&O from the IPO for $67
    initial_game_state.process_action(
        action_helper.get_all_choices()[0]
    )  # [21:13] Player 3 buys a 10% share of B&O from the IPO for $67
    initial_game_state.process_action(
        action_helper.get_all_choices()[0]
    )  # [21:13] Player 4 buys a 10% share of B&O from the IPO for $67
    # [21:13] B&O floats
    # [21:13] B&O receives $670
    # [21:13] Player 1 has no valid actions and passes
    initial_game_state.process_action(action_helper.get_all_choices()[-2])  # [21:13] Player 2 pars PRR at $67
    # [21:13] Player 2 buys a 20% share of PRR from the IPO for $134
    # [21:13] Player 2 becomes the president of PRR
    initial_game_state.process_action(
        action_helper.get_all_choices()[1]
    )  # [21:13] Player 3 buys a 10% share of PRR from the IPO for $67
    initial_game_state.process_action(
        action_helper.get_all_choices()[1]
    )  # [21:13] Player 4 buys a 10% share of PRR from the IPO for $67
    # [21:13] Player 1 has no valid actions and passes
    initial_game_state.process_action(
        action_helper.get_all_choices()[1]
    )  # [21:13] Player 2 buys a 10% share of PRR from the IPO for $67
    # [21:13] PRR floats
    # [21:13] PRR receives $670
    initial_game_state.process_action(
        action_helper.get_all_choices()[1]
    )  # [21:13] Player 3 buys a 10% share of PRR from the IPO for $67
    initial_game_state.process_action(
        action_helper.get_all_choices()[0]
    )  # [21:14] Player 4 buys a 10% share of B&O from the IPO for $67
    # [21:14] Player 4 becomes the president of B&O
    # [21:14] Player 1 has no valid actions and passes
    initial_game_state.process_action(action_helper.get_all_choices()[-1])  # [21:14] Player 2 passes
    initial_game_state.process_action(action_helper.get_all_choices()[-1])  # [21:14] Player 3 passes
    initial_game_state.process_action(
        action_helper.get_all_choices()[1]
    )  # [21:14] Player 4 buys a 10% share of PRR from the IPO for $67
    # [21:14] Player 1 has no valid actions and passes
    initial_game_state.process_action(action_helper.get_all_choices()[-1])  # [21:14] Player 2 passes
    initial_game_state.process_action(action_helper.get_all_choices()[-1])  # [21:14] Player 3 passes
    initial_game_state.process_action(
        action_helper.get_all_choices()[1]
    )  # [21:14] Player 4 buys a 10% share of PRR from the IPO for $67
    # [21:14] Player 1 has no valid actions and passes
    initial_game_state.process_action(action_helper.get_all_choices()[-1])  # [21:14] Player 2 passes
    initial_game_state.process_action(action_helper.get_all_choices()[-1])  # [21:14] Player 3 passes
    initial_game_state.process_action(
        action_helper.get_all_choices()[1]
    )  # [21:14] Player 4 buys a 10% share of PRR from the IPO for $67
    # [21:14] Player 4 becomes the president of PRR
    # [21:14] Player 1 has no valid actions and passes
    initial_game_state.process_action(action_helper.get_all_choices()[-1])  # [21:15] Player 2 passes
    initial_game_state.process_action(action_helper.get_all_choices()[-1])  # [21:15] Player 3 passes
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
        action_helper.get_all_choices()[0]
    )  # [21:16] PRR lays tile #57 with rotation 1 on H10 (Pittsburgh)
    initial_game_state.process_action(action_helper.get_all_choices()[-1])  # [21:16] PRR passes place a token
    # [21:16] PRR skips run routes
    # [21:16] PRR does not run
    # [21:16] PRR's share price moves left from 67
    initial_game_state.process_action(action_helper.get_all_choices()[0])  # [21:16] PRR buys a 2 train for $80 from The Depot
    initial_game_state.process_action(action_helper.get_all_choices()[0])  # [21:16] PRR buys a 2 train for $80 from The Depot
    initial_game_state.process_action(action_helper.get_all_choices()[-1])  # [21:17] PRR passes buy trains
    # [21:17] PRR skips buy companies
    # [21:17] Player 4 operates B&O
    # [21:17] B&O places a token on I15
    initial_game_state.process_action(
        action_helper.get_all_choices()[4]
    )  # [21:17] B&O spends $80 and lays tile #57 with rotation 0 on J14 (Washington)
    initial_game_state.process_action(action_helper.get_all_choices()[-1])  # [21:17] B&O passes place a token
    # [21:17] B&O skips run routes
    # [21:17] B&O does not run
    # [21:17] B&O's share price moves left from 65
    initial_game_state.process_action(action_helper.get_all_choices()[-1])  # [21:22] B&O buys a 2 train for $590 from PRR
    # [21:22] Baltimore & Ohio closes
    # [21:22] B&O skips buy companies
    # [21:22] -- Stock Round 2 --
    initial_game_state.process_action(action_helper.get_all_choices()[-1])  # [21:23] Player 1 passes
    # [23:26] Player 2 pars NYC at $67
    initial_game_state.process_action(
        action_helper.get_all_choices()[31]
    )  # [23:26] Player 2 buys a 20% share of NYC from the IPO for $134
    # [23:26] Player 2 becomes the president of NYC
    initial_game_state.process_action(
        action_helper.get_all_choices()[0]
    )  # [23:26] Player 2 exchanges Mohawk & Hudson from the IPO for a 10% share of NYC
    initial_game_state.process_action(action_helper.get_all_choices()[-1])  # [23:26] Player 2 declines to sell shares
    initial_game_state.process_action(action_helper.get_all_choices()[13])  # [23:26] Player 3 pars C&O at $67
    # [23:26] Player 3 buys a 20% share of C&O from the IPO for $134
    # [23:26] Player 3 becomes the president of C&O
    initial_game_state.process_action(action_helper.get_all_choices()[-1])  # [23:26] Player 3 declines to sell shares
    initial_game_state.process_action(action_helper.get_all_choices()[-2])  # [23:26] Player 4 sells 3 shares of B&O and receives $195
    # [23:26] Player 1 becomes the president of B&O
    # [23:26] B&O's share price moves down from 50
    initial_game_state.process_action(
        action_helper.get_all_choices()[0]
    )  # [23:27] Player 4 buys a 10% share of NYC from the IPO for $67
    initial_game_state.process_action(action_helper.get_all_choices()[-1])
    # [23:27] Player 1 has no valid actions and passes
    initial_game_state.process_action(
        action_helper.get_all_choices()[0]
    )  # [23:27] Player 2 buys a 10% share of NYC from the IPO for $67
    initial_game_state.process_action(action_helper.get_all_choices()[-1])  # [23:27] Player 2 declines to sell shares
    initial_game_state.process_action(
        action_helper.get_all_choices()[1]
    )  # [23:27] Player 3 buys a 10% share of C&O from the IPO for $67
    initial_game_state.process_action(action_helper.get_all_choices()[-1])  # [23:27] Player 3 declines to sell shares
    initial_game_state.process_action(
        action_helper.get_all_choices()[0]
    )  # [23:27] Player 4 buys a 10% share of NYC from the IPO for $67
    # [23:27] NYC floats
    # [23:27] NYC receives $670
    initial_game_state.process_action(action_helper.get_all_choices()[-1])  # [23:27] Player 4 declines to sell shares
    # [23:27] Player 1 has no valid actions and passes
    initial_game_state.process_action(action_helper.get_all_choices()[2])  # [23:27] Player 2 sells 3 shares of PRR and receives $201
    # [23:27] PRR's share price moves down from 60
    initial_game_state.process_action(
        action_helper.get_all_choices()[1]
    )  # [23:27] Player 2 buys a 10% share of C&O from the IPO for $67
    initial_game_state.process_action(action_helper.get_all_choices()[-1])
    initial_game_state.process_action(action_helper.get_all_choices()[1])  # [23:27] Player 3 sells 2 shares of PRR and receives $120
    # [23:27] PRR's share price moves down from 40
    initial_game_state.process_action(
        action_helper.get_all_choices()[1]
    )  # [23:27] Player 3 buys a 10% share of C&O from the IPO for $67
    initial_game_state.process_action(action_helper.get_all_choices()[-1])
    initial_game_state.process_action(
        action_helper.get_all_choices()[1]
    )  # [23:27] Player 4 buys a 10% share of C&O from the IPO for $67
    # [23:27] C&O floats
    # [23:27] C&O receives $670
    initial_game_state.process_action(action_helper.get_all_choices()[-1])  # [23:35] Player 4 declines to sell shares
    # [23:35] Player 1 has no valid actions and passes
    initial_game_state.process_action(action_helper.get_all_choices()[20])  # [23:35] Player 2 sells a 10% share of B&O and receives $50
    # [23:35] B&O's share price moves down from 40
    initial_game_state.process_action(action_helper.get_all_choices()[-1])  # [23:35] Player 2 declines to buy shares
    initial_game_state.process_action(action_helper.get_all_choices()[4])  # [23:35] Player 3 sells a 10% share of B&O and receives $40
    # [23:35] B&O's share price moves down from 30
    initial_game_state.process_action(action_helper.get_all_choices()[-1])  # [23:35] Player 3 declines to buy shares
    initial_game_state.process_action(action_helper.get_all_choices()[-1])  # [23:35] Player 4 passes
    initial_game_state.process_action(action_helper.get_all_choices()[-1])  # [23:35] Player 1 passes
    initial_game_state.process_action(action_helper.get_all_choices()[-1])  # [23:35] Player 2 passes
    initial_game_state.process_action(action_helper.get_all_choices()[-1])  # [23:35] Player 3 passes
    # [23:35] Player 4 has priority deal
    # [23:35] -- Operating Round 2.1 (of 1) --
    # [23:35] Player 4 collects $15 from Delaware & Hudson
    # [23:35] Player 2 collects $5 from Schuylkill Valley
    # [23:35] Player 3 collects $10 from Champlain & St.Lawrence
    # [23:35] Player 3 collects $25 from Camden & Amboy
    # [23:35] Player 2 operates NYC
    # [23:35] NYC places a token on E19
    initial_game_state.process_action(action_helper.get_all_choices()[-1])  # [23:35] NYC passes lay/upgrade track
    # [23:35] NYC skips place a token
    # [23:35] NYC skips run routes
    # [23:35] NYC does not run
    # [23:35] NYC's share price moves left from 65
    initial_game_state.process_action(action_helper.get_all_choices()[0])  # [23:35] NYC buys a 2 train for $80 from The Depot
    initial_game_state.process_action(action_helper.get_all_choices()[0])  # [23:35] NYC buys a 2 train for $80 from The Depot
    initial_game_state.process_action(action_helper.get_all_choices()[0])  # [23:35] NYC buys a 2 train for $80 from The Depot
    initial_game_state.process_action(action_helper.get_all_choices()[0])  # [23:36] NYC buys a 2 train for $80 from The Depot
    # [23:36] NYC skips buy companies
    # [23:36] Player 3 operates C&O
    # [23:36] C&O places a token on F6
    initial_game_state.process_action(action_helper.get_all_choices()[-1])  # [23:36] C&O passes lay/upgrade track
    # [23:36] C&O skips place a token
    # [23:36] C&O skips run routes
    # [23:36] C&O does not run
    # [23:36] C&O's share price moves left from 65
    initial_game_state.process_action(action_helper.get_all_choices()[0])  # [23:36] C&O buys a 3 train for $180 from The Depot
    # [23:36] -- Phase 3 (Operating Rounds: 2 | Train Limit: 4 | Available Tiles: Yellow, Green) --
    initial_game_state.process_action(action_helper.get_all_choices()[-2])  # [23:36] C&O buys a 3 train for $180 from The Depot
    initial_game_state.process_action(action_helper.get_all_choices()[-2])  # [23:36] C&O buys a 3 train for $180 from The Depot
    initial_game_state.process_action(action_helper.get_all_choices()[-1])  # [23:36] C&O passes buy trains
    # [23:36] C&O passes buy companies
    # [23:36] Player 4 operates PRR
    initial_game_state.process_action(action_helper.get_all_choices()[-1])  # [23:36] PRR passes lay/upgrade track
    initial_game_state.process_action(action_helper.get_all_choices()[-1])  # [23:36] PRR passes place a token
    initial_game_state.process_action(action_helper.get_all_choices()[-1])  # [23:36] PRR runs a 2 train for $30: H12-H10
    initial_game_state.process_action(
        action_helper.get_all_choices()[-1]
    )  # [23:36] PRR pays out 3 per share (12 to Player 4, $3 to Player 3)
    # [23:36] PRR's share price moves right from 50
    initial_game_state.process_action(action_helper.get_all_choices()[-2])  # [23:36] PRR buys a 3 train for $180 from The Depot
    initial_game_state.process_action(action_helper.get_all_choices()[-2])  # [23:36] PRR buys a 3 train for $180 from The Depot
    initial_game_state.process_action(action_helper.get_all_choices()[-2])  # [23:36] PRR buys a 4 train for $300 from The Depot
    # [23:36] -- Phase 4 (Operating Rounds: 2 | Train Limit: 3 | Available Tiles: Yellow, Green) --
    # [23:36] -- Event: 2 trains rust ( B&O x1, PRR x1, NYC x4) --
    initial_game_state.process_action(action_helper.get_all_choices()[-1])  # [23:36] PRR passes buy companies
    # [23:36] Player 1 operates B&O
    initial_game_state.process_action(action_helper.get_all_choices()[-1])  # [23:36] B&O passes lay/upgrade track
    # [23:36] B&O skips place a token
    # [23:36] B&O skips run routes
    # [23:36] B&O does not run
    # [23:36] B&O's share price moves left from 20
    initial_game_state.process_action(action_helper.get_all_choices()[0])  # [23:36] Player bankrupts
    return (initial_game_state, action_helper, action_mapper, encoder)

@pytest.fixture
def game_state_single_action(game_objects) -> GameObjects:
    initial_game_state, action_helper, action_mapper, encoder = game_objects
    initial_game_state.process_action(action_helper.get_all_choices()[-1])  # pass
    initial_game_state.process_action(action_helper.get_all_choices()[1])  # bid 45 on CS
    initial_game_state.process_action(action_helper.get_all_choices()[1])  # bid 50 on CS
    initial_game_state.process_action(action_helper.get_all_choices()[-77])  # bid 225 on BO
    initial_game_state.process_action(action_helper.get_all_choices()[0])  # buy SV
    initial_game_state.process_action(action_helper.get_all_choices()[-1])  # pass
    initial_game_state.process_action(action_helper.get_all_choices()[0])  # buy DH
    initial_game_state.process_action(action_helper.get_all_choices()[0])  # buy MH
    initial_game_state.process_action(action_helper.get_all_choices()[0])  # buy CA
    initial_game_state.process_action(action_helper.get_all_choices()[0])  # Par B&O at 100
    initial_game_state.process_action(action_helper.get_all_choices()[-2])  # Par PRR
    initial_game_state.process_action(action_helper.get_all_choices()[-1])  # Pass
    initial_game_state.process_action(action_helper.get_all_choices()[-8])  # Par NYC
    initial_game_state.process_action(action_helper.get_all_choices()[1])  # Buy PRR
    initial_game_state.process_action(action_helper.get_all_choices()[1])  # Buy PRR
    initial_game_state.process_action(action_helper.get_all_choices()[14])  # Par C&O
    initial_game_state.process_action(action_helper.get_all_choices()[2])  # Buy NYC
    initial_game_state.process_action(action_helper.get_all_choices()[1])  # Buy PRR
    initial_game_state.process_action(action_helper.get_all_choices()[1])  # Buy PRR
    initial_game_state.process_action(action_helper.get_all_choices()[3])  # Buy C&O
    initial_game_state.process_action(action_helper.get_all_choices()[2])  # Buy NYC
    initial_game_state.process_action(action_helper.get_all_choices()[0])  # Buy PRR
    initial_game_state.process_action(action_helper.get_all_choices()[1])  # Buy PRR
    initial_game_state.process_action(action_helper.get_all_choices()[3])  # Buy C&O
    initial_game_state.process_action(action_helper.get_all_choices()[2])  # Buy NYC
    initial_game_state.process_action(action_helper.get_all_choices()[1])  # Buy PRR
    initial_game_state.process_action(action_helper.get_all_choices()[2])  # Buy C&O
    initial_game_state.process_action(action_helper.get_all_choices()[1])  # Buy NYC
    initial_game_state.process_action(action_helper.get_all_choices()[1])  # Buy NYC
    initial_game_state.process_action(action_helper.get_all_choices()[2])  # Buy C&O
    initial_game_state.process_action(action_helper.get_all_choices()[1])  # Buy NYC
    initial_game_state.process_action(action_helper.get_all_choices()[1])  # Buy NYC
    # PRR
    initial_game_state.process_action(action_helper.get_all_choices()[0])  # lays tile #57 with rotation 1 on H10
    initial_game_state.process_action(action_helper.get_all_choices()[-1])  # passes place token
    initial_game_state.process_action(action_helper.get_all_choices()[0])  # buys a 2 train
    initial_game_state.process_action(action_helper.get_all_choices()[0])  # buys a 2 train
    initial_game_state.process_action(action_helper.get_all_choices()[-1])  # passes trains

    # NYC
    initial_game_state.process_action(action_helper.get_all_choices()[0])  # lays tile #57 with rotation 0 on E19
    initial_game_state.process_action(action_helper.get_all_choices()[0])  # buys a 2 train
    initial_game_state.process_action(action_helper.get_all_choices()[-1])  # passes trains

    # C&O
    initial_game_state.process_action(action_helper.get_all_choices()[2])
    initial_game_state.process_action(action_helper.get_all_choices()[0])  # Buys a 2 train
    initial_game_state.process_action(action_helper.get_all_choices()[-1])  # passes trains

    # SR2
    initial_game_state.process_action(action_helper.get_all_choices()[-2])  # sell 50% nyc
    initial_game_state.process_action(action_helper.get_all_choices()[-3])  # par nynh at 71
    initial_game_state.process_action(action_helper.get_all_choices()[0])  # buy C&O
    initial_game_state.process_action(action_helper.get_all_choices()[-1])  # pass sell
    initial_game_state.process_action(action_helper.get_all_choices()[0])  # Buy NYC
    initial_game_state.process_action(action_helper.get_all_choices()[-1])  # pass sell
    initial_game_state.process_action(action_helper.get_all_choices()[0])  # Buy NYNH
    initial_game_state.process_action(action_helper.get_all_choices()[-1])  # pass sell
    initial_game_state.process_action(action_helper.get_all_choices()[1])  # Buy NYNH
    initial_game_state.process_action(action_helper.get_all_choices()[-1])  # pass sell
    initial_game_state.process_action(action_helper.get_all_choices()[-1])  # pass
    initial_game_state.process_action(action_helper.get_all_choices()[-1])  # pass
    initial_game_state.process_action(action_helper.get_all_choices()[-1])  # pass
    initial_game_state.process_action(action_helper.get_all_choices()[1])  # Buy NYNH
    initial_game_state.process_action(action_helper.get_all_choices()[-1])  # pass sell
    initial_game_state.process_action(action_helper.get_all_choices()[-1])  # pass
    initial_game_state.process_action(action_helper.get_all_choices()[-1])  # pass
    initial_game_state.process_action(action_helper.get_all_choices()[-1])  # pass
    initial_game_state.process_action(action_helper.get_all_choices()[1])  # Buy NYNH
    initial_game_state.process_action(action_helper.get_all_choices()[-1])  # pass sell
    initial_game_state.process_action(action_helper.get_all_choices()[-1])  # pass
    initial_game_state.process_action(action_helper.get_all_choices()[-1])  # pass
    initial_game_state.process_action(action_helper.get_all_choices()[-1])  # pass
    initial_game_state.process_action(action_helper.get_all_choices()[-1])  # pass
    # NYNH
    initial_game_state.process_action(action_helper.get_all_choices()[0])  # lay #1 with rotation 0 on F20
    initial_game_state.process_action(action_helper.get_all_choices()[0])  # buy 2 train
    initial_game_state.process_action(action_helper.get_all_choices()[-1])  # pass trains
    # PRR
    initial_game_state.process_action(action_helper.get_all_choices()[10])  # lay tile #9 with rotation 1 on H8
    initial_game_state.process_action(action_helper.get_all_choices()[-1])  # pass token
    return (initial_game_state, action_helper, action_mapper, encoder)

@pytest.fixture
def mcts_instance(game_objects, model) -> Tuple[MCTS, GameObjects]:
    initial_game_state, _, action_mapper, encoder = game_objects
    mcts = MCTS(
        root_state=initial_game_state,
        model=model,
        encoder=encoder,
        action_mapper=action_mapper,
        num_simulations=TEST_NUM_SIMULATIONS,
        c_puct=TEST_C_PUCT,
        alpha=0.0, # No noise for basic tests unless specified
        noise_factor=0.0,
        device=torch.device("cpu")
    )
    return mcts, game_objects

@pytest.fixture
def mcts_instance_with_noise(game_objects, model) -> Tuple[MCTS, GameObjects]:
    initial_game_state, _, action_mapper, encoder = game_objects
    mcts = MCTS(
        root_state=initial_game_state,
        model=model,
        encoder=encoder,
        action_mapper=action_mapper,
        num_simulations=TEST_NUM_SIMULATIONS,
        c_puct=TEST_C_PUCT,
        alpha=0.3, # Enable Dirichlet noise
        noise_factor=0.25,
        device=torch.device("cpu")
    )
    return mcts, game_objects


# --- Node Tests ---
def test_node_initialization(game_objects):
    initial_game_state, _, action_mapper, _ = game_objects
    num_players = len(initial_game_state.players)
    action_enc_size = action_mapper.action_encoding_size

    root_node = Node(
        parent=None,
        state=initial_game_state,
        num_players=num_players,
        action_encoding_size=action_enc_size,
        action_index=None
    )
    assert root_node.parent is None
    assert root_node.state == initial_game_state
    assert root_node.num_players == num_players
    assert root_node.action_encoding_size == action_enc_size
    assert root_node.action_index is None
    assert root_node.node_total_visits == 0
    assert len(root_node.children) == 0
    assert np.all(root_node.child_prior_probabilities == 0)
    assert np.all(root_node.child_visit_counts == 0)
    assert np.all(root_node.child_total_action_values == 0)
    assert np.all(root_node.child_average_action_values == 0)

    # Child node
    # Parent first expands the child, setting the prior in parent.child_prior_probabilities
    # Then, the child node is created.
    action_idx_to_child = 0
    prior_for_child = 0.5
    
    # Simulate parent expanding this child
    root_node.child_prior_probabilities[action_idx_to_child] = prior_for_child
    # Other child stats (visits, values) remain 0 until backpropagation for this action

    child_node = Node(
        parent=root_node,
        state=initial_game_state.clone(initial_game_state.raw_actions),
        num_players=num_players,
        action_encoding_size=action_enc_size,
        action_index=action_idx_to_child
    )
    root_node.children[action_idx_to_child] = child_node

    assert child_node.parent == root_node
    assert child_node.action_index == action_idx_to_child
    assert child_node.state.raw_actions == initial_game_state.raw_actions
    assert child_node.node_total_visits == 0
    # The prior for the edge *to* child_node is stored in root_node
    assert root_node.child_prior_probabilities[action_idx_to_child] == prior_for_child
    # The visit count for the edge *to* child_node is also in root_node
    assert root_node.child_visit_counts[action_idx_to_child] == 0


def test_node_is_leaf(game_objects, model):
    initial_game_state, _, action_mapper, _ = game_objects
    num_players = len(initial_game_state.players)
    action_enc_size = action_mapper.action_encoding_size
    
    node = Node(None, initial_game_state, num_players, action_enc_size)
    assert node.is_leaf()
    
    # Simulate expansion by adding a child (simplified)
    # Proper expansion involves MCTS.expand_and_evaluate
    child_state_dummy = initial_game_state.clone(initial_game_state.raw_actions)
    # In real expansion, parent calls expand_child which sets prior and creates node
    node.expand_child(action_idx=0, child_state=child_state_dummy)
    assert not node.is_leaf()

def test_node_creation_and_attributes(game_objects):
    initial_game_state, _, action_mapper, _ = game_objects
    root_node = Node(parent=None, state=initial_game_state, num_players=4, action_encoding_size=action_mapper.action_encoding_size)
    assert root_node.parent is None
    assert root_node.state == initial_game_state
    assert root_node.node_total_visits == 0
    assert len(root_node.children) == 0
    assert root_node.is_leaf()
    assert root_node.action_index is None
    assert root_node.child_prior_probabilities.shape == (action_mapper.action_encoding_size,)
    assert np.all(root_node.child_prior_probabilities == 0)
    assert root_node.child_visit_counts.shape == (action_mapper.action_encoding_size,)
    assert np.all(root_node.child_visit_counts == 0)
    assert root_node.child_total_action_values.shape == (action_mapper.action_encoding_size, 4)
    assert np.all(root_node.child_total_action_values == 0)
    assert root_node.child_average_action_values.shape == (action_mapper.action_encoding_size, 4)
    assert np.all(root_node.child_average_action_values == 0)


    mock_game_state_child = initial_game_state.clone(initial_game_state.raw_actions)
    child_node = root_node.expand_child(action_idx=0, child_state=mock_game_state_child)

    assert child_node.parent == root_node
    assert child_node.state == mock_game_state_child
    assert child_node.action_index == 0
    assert not root_node.is_leaf()
    assert 0 in root_node.children
    assert root_node.children[0] == child_node

def test_node_expand_child_and_initial_stats(game_objects):
    """
    Tests that Node.expand_child correctly creates a child,
    and that the parent's child-related stats arrays are initialized (as zeros by default).
    Priors are set by MCTS.expand_and_evaluate, not Node.expand_child.
    """
    initial_game_state, _, action_mapper, _ = game_objects
    parent_node = Node(parent=None, state=initial_game_state, num_players=4, action_encoding_size=action_mapper.action_encoding_size)
    
    mock_game_state_child = initial_game_state.clone(initial_game_state.raw_actions)
    action_to_child = 0
    
    # Expand the child
    child_node = parent_node.expand_child(action_idx=action_to_child, child_state=mock_game_state_child)

    assert action_to_child in parent_node.children
    assert parent_node.children[action_to_child] == child_node
    assert child_node.parent == parent_node
    assert child_node.action_index == action_to_child

    # Check initial stats on parent for this new child's edge
    # Priors are set by MCTS.expand_and_evaluate, so parent_node.child_prior_probabilities[action_to_child] would still be 0 here.
    assert parent_node.child_prior_probabilities[action_to_child] == 0.0
    assert parent_node.child_visit_counts[action_to_child] == 0
    assert np.all(parent_node.child_total_action_values[action_to_child, :] == 0)
    assert np.all(parent_node.child_average_action_values[action_to_child, :] == 0)

    # Child node itself should have its own stats initialized
    assert child_node.node_total_visits == 0
    assert child_node.is_leaf()


# --- MCTS Tests ---
def test_mcts_initialization(mcts_instance):
    mcts, game_objects = mcts_instance
    initial_game_state, _, action_mapper, _ = game_objects
    
    assert mcts.root is not None
    assert mcts.root.parent is None
    assert mcts.root.state == initial_game_state
    assert mcts.root.num_players == len(initial_game_state.players)
    assert mcts.root.action_encoding_size == action_mapper.action_encoding_size
    assert mcts.root.action_index is None # Root has no action leading to it

    assert mcts.root.node_total_visits == 0
    assert len(mcts.root.children) == 0
    assert np.all(mcts.root.child_prior_probabilities == 0) # Initialized to zeros
    assert np.all(mcts.root.child_visit_counts == 0)
    assert np.all(mcts.root.child_total_action_values == 0)
    assert np.all(mcts.root.child_average_action_values == 0)

def test_mcts_search_runs(mcts_instance):
    mcts, _ = mcts_instance
    try:
        mcts.search()
    except Exception as e:
        # Print more info if search fails
        print(f"MCTS search failed: {e}")
        import traceback
        print(traceback.format_exc())
        # You might want to log mcts.root or its state here
        raise

    assert mcts.root.node_total_visits == TEST_NUM_SIMULATIONS
    assert len(mcts.root.children) > 0, "Root should have children after search"

    # Sum of visit counts for edges from root should equal total simulations
    # (assuming each simulation explores one child from the root)
    total_child_edge_visits = np.sum(mcts.root.child_visit_counts[list(mcts.root.children.keys())])
    assert total_child_edge_visits == TEST_NUM_SIMULATIONS

    # Each child node's own visit count (node_total_visits) should reflect how many sims passed through it.
    # The sum of child_node.node_total_visits might not be TEST_NUM_SIMULATIONS if sims revisit same children.
    # But sum of root.child_visit_counts for *expanded* children should be TEST_NUM_SIMULATIONS.


def test_get_policy_temperature_one(mcts_instance):
    mcts, _ = mcts_instance
    mcts.search()

    policy_dict, policy_vector = mcts.get_policy(temperature=1.0)

    assert np.isclose(sum(policy_dict.values()), 1.0), "Policy probabilities should sum to 1.0"
    assert np.isclose(np.sum(policy_vector), 1.0), "Policy vector sum should be 1.0"
    assert policy_vector.shape == (mcts.action_mapper.action_encoding_size,)

    # Check if probabilities are proportional to visit counts (N^(1/T), T=1 => N)
    child_action_indices = list(mcts.root.children.keys())
    
    # child_visits_from_root[action_idx] = N(root, action_idx)
    child_visits_from_root = mcts.root.child_visit_counts[child_action_indices]

    total_visits = np.sum(child_visits_from_root)
    if total_visits == 0 :
        # This case is handled by get_policy to use priors or uniform.
        # For this test, we assume search resulted in some visits.
        # If not, the policy dict might be based on priors, not visits.
        # Let's check if any child has visits. If not, this check is not meaningful.
        if not np.any(child_visits_from_root > 0):
             pytest.fail("No child visits after search, cannot verify visit-based policy proportionality.")


    for action_idx, prob in policy_dict.items():
        expected_prob = mcts.root.child_visit_counts[action_idx] / total_visits if total_visits > 0 else 0
        assert np.isclose(prob, expected_prob), \
            f"Prob for action {action_idx} mismatch: got {prob}, expected {expected_prob}"
        assert np.isclose(policy_vector[action_idx], expected_prob)


def test_get_policy_temperature_zero(mcts_instance):
    mcts, _ = mcts_instance
    if mcts.root.state.finished:
        pytest.fail("Skipping policy test for terminal root state.")
    mcts.search()

    if not mcts.root.children:
        pytest.fail("Root has no children after search, cannot test policy.")

    policy_dict, policy_vector = mcts.get_policy(temperature=0.0)

    assert np.isclose(sum(policy_dict.values()), 1.0)
    assert np.isclose(np.sum(policy_vector), 1.0)

    num_ones = sum(1 for p in policy_dict.values() if np.isclose(p, 1.0))
    num_zeros = sum(1 for p in policy_dict.values() if np.isclose(p, 0.0))
    
    assert num_ones == 1, f"Expected one action with prob 1.0, got {num_ones}. Policy: {policy_dict}"
    assert num_zeros == len(policy_dict) - 1

    # Find the action index with max visits from root's perspective
    child_action_indices = list(mcts.root.children.keys())
    if not child_action_indices: # Should be caught by earlier check
        pytest.fail("No children to determine max visits from.")

    child_visits_from_root = mcts.root.child_visit_counts[child_action_indices]
    
    # Handle cases where all children might have 0 visits (e.g. very few sims, or all paths terminate early)
    if np.sum(child_visits_from_root) == 0:
        # Policy with temp=0 and no visits might pick one based on priors or first available.
        # The current get_policy falls back to priors, then uniform if priors are zero.
        # If it falls back to uniform over N children, then picks one randomly.
        # This test expects a single max_visit action. If all are 0, it's ambiguous.
        # For now, let's assume search populates some visits. If not, this test might need adjustment
        # based on the exact tie-breaking of the fallback in get_policy.
        # The test for `get_policy` handles the zero-visit case by checking against priors/uniform.
        # Here, we expect a greedy choice based on visits.
        pytest.fail("All children have zero visits, temp=0 policy is ambiguous for this specific check.")


    max_visit_child_idx_in_slice = np.argmax(child_visits_from_root)
    max_visit_action_idx = child_action_indices[max_visit_child_idx_in_slice]
    
    assert np.isclose(policy_dict.get(max_visit_action_idx, 0.0), 1.0)
    assert np.isclose(policy_vector[max_visit_action_idx], 1.0)


def test_get_policy_no_children_error(mcts_instance):
    mcts, game_objects = mcts_instance
    # Ensure root has no children (it shouldn't before simulations if num_simulations is 0 for MCTS init)
    # Or if MCTS is initialized with 0 simulations.
    # The fixture mcts_instance has TEST_NUM_SIMULATIONS > 0, so search would run.
    # To test this, we need an MCTS instance that hasn't run search or has 0 sims.
    
    initial_game_state, _, action_mapper, encoder = game_objects
    model_fixture = mcts.model # Get model from the fixture
    
    mcts_no_sim = MCTS(
        root_state=initial_game_state,
        model=model_fixture,
        encoder=encoder,
        action_mapper=action_mapper,
        num_simulations=0, # Key: 0 simulations
        c_puct=TEST_C_PUCT,
        device=torch.device("cpu")
    )
    assert not mcts_no_sim.root.children
    
    # If num_simulations is 0, search() doesn't run, root has no children.
    # The get_policy method has a check for this.
    # If root is terminal, it also might not have children.
    if mcts_no_sim.root.state.finished:
         # If root is terminal, expand_and_evaluate returns value, no children expanded.
         # get_policy would then correctly return empty/zero policy.
         policy_d, policy_v = mcts_no_sim.get_policy()
         assert len(policy_d) == 0
         assert np.sum(policy_v) == 0
    else:
        # If not terminal and no sims, it should raise ValueError as per current code.
        # The code was: `raise ValueError("Root node has no children after search")`
        # Let's check the updated code:
        # `if not self.root.children: ... if self.num_simulations > 0 and not self.root.state.finished: raise ValueError`
        # So if num_simulations is 0, it should return empty policy.
        policy_d, policy_v = mcts_no_sim.get_policy()
        assert len(policy_d) == 0
        assert np.sum(policy_v) == 0
        # To test the ValueError, we need num_simulations > 0 and still no children (e.g. if expansion failed)
        # This specific test is for "no children" generally. The current behavior for num_sims=0 is fine.


def test_dirichlet_noise_application(mcts_instance_with_noise, model):
    mcts_noisy, game_objects = mcts_instance_with_noise
    initial_game_state, _, action_mapper, encoder = game_objects
    root = mcts_noisy.root

    if root.state.finished:
        pytest.fail("Root state is terminal, noise application test might not be meaningful as expansion differs.")

    # --- Get raw policy from network (masked and normalized) for comparison ---
    game_state_cpu, (map_nodes_cpu, raw_edge_input_tensor_cpu) = encoder.encode(root.state)
    game_state_dev = game_state_cpu.to(mcts_noisy.device)
    map_nodes_dev = map_nodes_cpu.float().to(mcts_noisy.device)
    raw_edge_input_tensor_dev = raw_edge_input_tensor_cpu.to(mcts_noisy.device)
    edge_index_dev = raw_edge_input_tensor_dev[0:2, :].long()
    edge_attr_categorical_dev = raw_edge_input_tensor_dev[2, :].long()
    num_nodes_graph = map_nodes_dev.shape[0]
    node_batch_idx_dev = torch.zeros(num_nodes_graph, dtype=torch.long, device=mcts_noisy.device)

    with torch.no_grad():
        policy_logits, _ = model(
            x_game_state=game_state_dev,
            x_map_nodes_batched=map_nodes_dev,
            edge_index_batched=edge_index_dev,
            node_batch_idx=node_batch_idx_dev,
            edge_attr_categorical_batched=edge_attr_categorical_dev
        )
    raw_policy_tensor = torch.softmax(policy_logits, dim=1).squeeze(0).cpu()
    legal_actions_mask_np = action_mapper.get_legal_action_mask(root.state)
    num_legal_actions_from_mask = np.sum(legal_actions_mask_np)
    
    if num_legal_actions_from_mask == 0:
        pytest.fail("No legal actions, cannot test noise application effectively.")

    legal_actions_mask_torch = torch.from_numpy(legal_actions_mask_np).to(raw_policy_tensor.device)
    masked_raw_policy = raw_policy_tensor * legal_actions_mask_torch
    raw_policy_sum = torch.sum(masked_raw_policy)
    # --- End of block for policy_sum and masked_policy calculation ---

    # Run expand_and_evaluate, which applies noise if it's the root and alpha/factor > 0
    _ = mcts_noisy.expand_and_evaluate(root) 
    
    if not root.children and num_legal_actions_from_mask > 0 :
        pytest.fail("Root not expanded after expand_and_evaluate (but had legal actions), cannot test noise on priors.")
    
    # Priors are stored in root.child_prior_probabilities
    # We should only consider priors for legal actions.
    legal_action_indices = np.where(legal_actions_mask_np)[0]
    stored_priors_for_legal_actions = root.child_prior_probabilities[legal_action_indices]
    sum_stored_priors = np.sum(stored_priors_for_legal_actions)

    if num_legal_actions_from_mask > 0:
        assert np.isclose(sum_stored_priors, 1.0, atol=1e-5), \
            f"Priors stored in root for its legal children should sum to ~1. Sum: {sum_stored_priors}. All priors: {root.child_prior_probabilities[legal_actions_mask_np]}"
    else: # No legal actions
        assert np.isclose(sum_stored_priors, 0.0, atol=1e-5), \
            f"Sum of priors should be 0 if no legal actions. Sum: {sum_stored_priors}"

    if raw_policy_sum > 1e-8 and num_legal_actions_from_mask > 0:
        normalized_masked_raw_policy_np = (masked_raw_policy / raw_policy_sum).numpy()
        
        # Compare stored priors (with noise) to the raw network policy for legal actions
        mcts_priors_for_legal = root.child_prior_probabilities[legal_action_indices]
        raw_policy_for_legal = normalized_masked_raw_policy_np[legal_action_indices]

        if len(mcts_priors_for_legal) > 1: # Only meaningful if multiple actions to compare
            differences_found = not np.allclose(mcts_priors_for_legal, raw_policy_for_legal, atol=1e-3) # Higher atol due to noise
            
            is_raw_policy_uniform_for_legal = len(np.unique(np.round(raw_policy_for_legal, decimals=5))) <= 1

            if not differences_found and not is_raw_policy_uniform_for_legal and mcts_noisy.noise_factor > 0 and mcts_noisy.alpha > 0:
                 print(f"Warning: Noise was active, but MCTS priors seem to match a non-uniform raw network policy for expanded children.")
                 print(f"MCTS priors (legal): {mcts_priors_for_legal}")
                 print(f"Normalized raw policy (legal): {raw_policy_for_legal}")
    elif mcts_noisy.noise_factor > 0 and mcts_noisy.alpha > 0 and num_legal_actions_from_mask > 0:
        print(f"Note: Raw policy sum was low. MCTS likely used uniform base for {num_legal_actions_from_mask} legal actions before applying noise.")
        # Check if stored priors are somewhat uniform (if noise didn't make them too skewed)
        # or at least different from zero.
        assert np.all(stored_priors_for_legal_actions > -1e-6), "Priors should not be negative."


def test_mcts_with_single_legal_action(game_state_single_action, model):
    """Tests MCTS behavior when only one action is legal."""
    single_action_state, _, action_mapper, encoder = game_state_single_action

    mcts_single = MCTS(
        root_state=single_action_state,
        model=model,
        encoder=encoder,
        action_mapper=action_mapper,
        num_simulations=TEST_NUM_SIMULATIONS, 
        c_puct=TEST_C_PUCT,
        alpha=0.0, # No noise for this specific behavior test
        noise_factor=0.0,
        device=torch.device("cpu")
    )

    # Verify only one legal action
    legal_mask_np = action_mapper.get_legal_action_mask(single_action_state)
    assert np.sum(legal_mask_np) == 1, "Test setup error: game state does not have a single legal action."
    
    try:
        mcts_single.search()
    except Exception as e:
        print(f"Error in search: {e}")
        import traceback
        print(traceback.format_exc())
        raise e

    assert len(mcts_single.root.children) == 1, "MCTS should have expanded only one child for a single legal action."
    
    single_child_action_idx = list(mcts_single.root.children.keys())[0]
    single_child_node = mcts_single.root.children[single_child_action_idx]

    assert mcts_single.root.child_visit_counts[single_child_action_idx] == TEST_NUM_SIMULATIONS, \
        f"The edge to the single child should have N_sa = num_simulations. Expected {TEST_NUM_SIMULATIONS}, got {mcts_single.root.child_visit_counts[single_child_action_idx]}."

    assert single_child_node.node_total_visits == TEST_NUM_SIMULATIONS, \
        f"The single child node (state) should have N_s = num_simulations. Expected {TEST_NUM_SIMULATIONS}, got {single_child_node.node_total_visits}."
    
    assert mcts_single.root.node_total_visits == TEST_NUM_SIMULATIONS, \
        f"The root node (state) should have N_s = num_simulations. Expected {TEST_NUM_SIMULATIONS}, got {mcts_single.root.node_total_visits}."

    policy_dict, policy_vector = mcts_single.get_policy(temperature=1.0)
    assert len(policy_dict) == 1, "Policy should be for a single action."
    assert np.isclose(list(policy_dict.values())[0], 1.0), "The single legal action should have probability 1.0."
    assert np.isclose(policy_vector[single_child_action_idx], 1.0)
    assert np.isclose(np.sum(policy_vector), 1.0)

def test_select_child_chooses_highest_puct(game_objects, model):
    initial_game_state, _, action_mapper, encoder = game_objects
    root_node = Node(parent=None, state=initial_game_state, num_players=4, action_encoding_size=action_mapper.action_encoding_size)

    # Mock game states for children
    mock_game_state_child1 = initial_game_state.clone(initial_game_state.raw_actions)
    mock_game_state_child2 = initial_game_state.clone(initial_game_state.raw_actions)

    # Expand two children
    action1 = 0
    action2 = 1
    child1_node = root_node.expand_child(action_idx=action1, child_state=mock_game_state_child1)
    child2_node = root_node.expand_child(action_idx=action2, child_state=mock_game_state_child2)

    # Setup root node's child stats
    root_node.child_prior_probabilities[action1] = 0.6
    root_node.child_prior_probabilities[action2] = 0.4
    root_node.child_visit_counts[action1] = 10
    root_node.child_visit_counts[action2] = 10 # Same visits initially
    root_node.node_total_visits = 20 # Sum of child visits for PUCT calculation

    # Child 1 has higher Q value
    root_node.child_average_action_values[action1, :] = np.array([0.8, 0, 0, 0], dtype=np.float32)
    root_node.child_total_action_values[action1, :] = root_node.child_average_action_values[action1, :] * root_node.child_visit_counts[action1]
    
    # Child 2 has lower Q value
    root_node.child_average_action_values[action2, :] = np.array([0.2, 0, 0, 0], dtype=np.float32)
    root_node.child_total_action_values[action2, :] = root_node.child_average_action_values[action2, :] * root_node.child_visit_counts[action2]


    mcts_instance = MCTS(
        root_state=initial_game_state, # Not directly used by select_child if root_node is passed
        model=model,
        encoder=encoder,
        action_mapper=action_mapper,
        num_simulations=1, # Not relevant for direct select_child test
        c_puct=1.0,
        device=torch.device("cpu")
    )
    
    # Assuming current player is player 0 for Q value perspective
    # For this test, we directly use child_average_action_values which should be from player 0's perspective if Q is for current player

    selected_action, _ = mcts_instance.select_child(root_node)
    assert selected_action == action1, "Should select child with higher Q + U (higher Q here, U is similar)"

    # Test case: Child 2 has much higher prior
    root_node.child_prior_probabilities[action1] = 0.1
    root_node.child_prior_probabilities[action2] = 0.9
    # Q values remain: child1 Q=0.8, child2 Q=0.2
    # U for child1 will be lower, U for child2 will be higher.
    # PUCT = Q + c_puct * P * sqrt(N_parent) / (1 + N_child)
    # P1=0.1, P2=0.9. N_child1=10, N_child2=10. N_parent=20
    # U1_factor = 0.1 * sqrt(20)/11 = 0.1 * 4.47/11 = 0.0406
    # U2_factor = 0.9 * sqrt(20)/11 = 0.9 * 4.47/11 = 0.3656
    # PUCT1 = 0.8 + 1.0 * 0.0406 = 0.8406
    # PUCT2 = 0.2 + 1.0 * 0.3656 = 0.5656
    # Action 1 should still be chosen if Q dominates strongly. Let's make Qs closer or priors more extreme.

    # Let Qs be equal, priors different
    root_node.child_average_action_values[action1, :] = np.array([0.5, 0, 0, 0], dtype=np.float32)
    root_node.child_total_action_values[action1, :] = root_node.child_average_action_values[action1, :] * root_node.child_visit_counts[action1]
    root_node.child_average_action_values[action2, :] = np.array([0.5, 0, 0, 0], dtype=np.float32)
    root_node.child_total_action_values[action2, :] = root_node.child_average_action_values[action2, :] * root_node.child_visit_counts[action2]
    # Priors: P1=0.1, P2=0.9. Visits N1=10, N2=10. N_parent=20
    # Q1=0.5, Q2=0.5
    # PUCT1 = 0.5 + 1.0 * 0.1 * sqrt(20)/(1+10) = 0.5 + 0.0406 = 0.5406
    # PUCT2 = 0.5 + 1.0 * 0.9 * sqrt(20)/(1+10) = 0.5 + 0.3656 = 0.8656
    selected_action, _ = mcts_instance.select_child(root_node)
    assert selected_action == action2, "Should select child with higher prior when Qs are equal"


def test_select_child_handles_no_visits_on_new_nodes(game_objects, model):
    initial_game_state, _, action_mapper, encoder = game_objects
    root_node = Node(parent=None, state=initial_game_state, num_players=4, action_encoding_size=action_mapper.action_encoding_size)

    mock_game_state_child = initial_game_state.clone(initial_game_state.raw_actions)
    test_action = 0
    child_node = root_node.expand_child(action_idx=test_action, child_state=mock_game_state_child)

    # Setup root node's child stats for the new child
    root_node.child_prior_probabilities[test_action] = 0.5 # Has a prior
    root_node.child_visit_counts[test_action] = 0 # No visits yet
    root_node.child_average_action_values[test_action, :] = np.array([0,0,0,0], dtype=np.float32) # Q is 0
    root_node.child_total_action_values[test_action, :] = np.array([0,0,0,0], dtype=np.float32)
    
    root_node.node_total_visits = 0 # Root itself has no visits yet for PUCT exploration term

    mcts_instance = MCTS(
        root_state=initial_game_state,
        model=model,
        encoder=encoder,
        action_mapper=action_mapper,
        num_simulations=1,
        c_puct=1.0,
        device=torch.device("cpu")
    )
    # PUCT = Q + c_puct * P * sqrt(N_parent) / (1 + N_child)
    # Q = 0, N_child = 0, N_parent = 0
    # U = c_puct * P * sqrt(0) / (1+0) = 0 if N_parent is 0.
    # If N_parent is 0, the exploration term is often P or P * c_puct.
    # AlphaZero paper: U = c_puct * P(s,a) * (sqrt(sum_b N(s,b))) / (1 + N(s,a))
    # If sum_b N(s,b) (i.e. root_node.node_total_visits) is 0, U is 0.
    # This means selection might be arbitrary or based on P if Qs are all 0.
    # Let's give root_node.node_total_visits a small value to make U non-zero.
    root_node.node_total_visits = 1 # Pretend one simulation has passed through root to other branches

    selected_action, _ = mcts_instance.select_child(root_node)
    assert selected_action == test_action, "Should select the unvisited child due to exploration bonus from prior"

def test_mcts_expand_and_evaluate_normal_node(game_objects):
    initial_game_state, _, action_mapper, encoder =     game_objects
    node = Node(parent=None, state=initial_game_state, num_players=4, action_encoding_size=action_mapper.action_encoding_size)

    # Mock model to return specific policy and value
    model_mocker = MagicMock(Model)
    mock_policy_logits = torch.randn(1, action_mapper.action_encoding_size)
    mock_value = torch.randn(1, 4)
    model_mocker.return_value = (mock_policy_logits, mock_value)

    mcts = MCTS(
        root_state=initial_game_state,
        model=model_mocker, # Use the mocked model
        encoder=encoder,    # Use the real encoder
        action_mapper=action_mapper,
        num_simulations=1, # Not relevant for this specific test
        device=torch.device("cpu")
    )

    # Ensure the game state has some legal actions
    legal_actions_mask = action_mapper.get_legal_action_mask(initial_game_state)
    if np.sum(legal_actions_mask) == 0:
        pytest.fail("Initial game state has no legal actions, cannot test normal expansion.")

    value_vector = mcts.expand_and_evaluate(node)

    assert value_vector is not None
    assert value_vector.shape == (4,)
    assert not node.is_leaf() if np.sum(legal_actions_mask) > 0 else node.is_leaf() # Node is expanded if legal actions exist
    
    # Check that priors are stored in node.child_prior_probabilities
    legal_action_indices = np.where(legal_actions_mask)[0]
    assert np.any(node.child_prior_probabilities != 0) if np.sum(legal_actions_mask) > 0 else True
    assert np.isclose(np.sum(node.child_prior_probabilities[legal_action_indices]), 1.0) if np.sum(legal_actions_mask) > 0 else np.isclose(np.sum(node.child_prior_probabilities), 0.0)

    # Check that children Node objects are created
    for i, is_legal in enumerate(legal_actions_mask):
        if is_legal:
            assert i in node.children
            assert isinstance(node.children[i], Node)
            assert node.children[i].parent == node
            assert node.children[i].action_index == i
        else:
            assert i not in node.children

def test_mcts_backpropagation(game_objects, model):
    initial_game_state, _, action_mapper, encoder = game_objects
    num_players = 4
    action_enc_size = action_mapper.action_encoding_size

    # Create a path: root -> child1 -> child2 (leaf)
    root = Node(parent=None, state=initial_game_state, num_players=num_players, action_encoding_size=action_enc_size)
    
    # Mock game states for children
    child1_state = initial_game_state.clone(initial_game_state.raw_actions) 
    # Simulate action 0 taken from root
    # child1_state.play_action(action_mapper.get_action_from_index(0, initial_game_state)) # If state needs update

    child2_state = child1_state.clone(child1_state.raw_actions)
    # Simulate action 1 taken from child1
    # child2_state.play_action(action_mapper.get_action_from_index(1, child1_state)) # If state needs update

    # Fix: Removed prior_p from expand_child calls
    child1 = root.expand_child(action_idx=0, child_state=child1_state)
    child2 = child1.expand_child(action_idx=1, child_state=child2_state) # child2 is the leaf

    # Manually set visit counts as if selection path was root -> action 0 -> child1 -> action 1 -> child2
    # These are set by MCTS.search during selection
    root.child_visit_counts[0] = 1
    root.node_total_visits = 1
    child1.child_visit_counts[1] = 1
    child1.node_total_visits = 1
    # child2.node_total_visits will be incremented by backprop itself as it's the leaf

    # Mock MCTS instance (only need a few params for backprop)
    mcts_dummy = MCTS(root_state=initial_game_state, model=model, encoder=encoder, action_mapper=action_mapper, num_simulations=1, device=torch.device("cpu"))
    mcts_dummy.apply_virtual_loss = False # Test basic backprop first

    value_vector = np.array([0.5, -0.5, 0.1, -0.1], dtype=np.float32) # Value from leaf (child2)

    # --- Perform backpropagation ---
    mcts_dummy.backpropagate(child2, value_vector)

    # --- Assertions ---
    # Leaf node (child2)
    assert child2.node_total_visits == 1, "Leaf's N(s) should be incremented by backprop."

    # Parent of leaf (child1)
    assert child1.node_total_visits == 1, "Child1's N(s) was set during selection, not changed by backprop for its own state."
    assert child1.child_visit_counts[1] == 1, "N(s,a) for (child1, action_to_child2) was set during selection."
    assert np.allclose(child1.child_total_action_values[1, :], value_vector)
    assert np.allclose(child1.child_average_action_values[1, :], value_vector / 1.0)

    # Root node
    assert root.node_total_visits == 1, "Root's N(s) was set during selection."
    assert root.child_visit_counts[0] == 1, "N(s,a) for (root, action_to_child1) was set during selection."
    assert np.allclose(root.child_total_action_values[0, :], value_vector)
    assert np.allclose(root.child_average_action_values[0, :], value_vector / 1.0)

    # Test with virtual loss
    mcts_dummy.apply_virtual_loss = True
    mcts_dummy.virtual_loss_c = 1.0

    # Reset stats for the path before re-testing with virtual loss
    root.child_total_action_values[0, :] = 0.0
    root.child_average_action_values[0, :] = 0.0
    child1.child_total_action_values[1, :] = 0.0
    child1.child_average_action_values[1, :] = 0.0
    # Visit counts N(s,a) and N(s) are assumed to be set during selection and are not reset here
    # Leaf N(s) also needs reset if we are re-running backprop on same leaf
    child2.node_total_visits = 0
    
    # Simulate virtual loss application during selection (W -= VL_c)
    # This would have happened in MCTS.search before backprop
    # We need to know which player was active at the parent node when the action was chosen
    # The MCTS get_player_index uses state.active_players()[0].id
    
    # Mock active players for player index determination
    # Player 0 at root, Player 0 at child1 (simplification for test)
    mock_root_active_player = MagicMock()
    mock_root_active_player.id = mcts_dummy.player_idx_to_id[0]
    root.state.active_players = MagicMock(return_value=[mock_root_active_player])

    mock_child1_active_player = MagicMock()
    mock_child1_active_player.id = mcts_dummy.player_idx_to_id[0] # Assuming player 0 also for child1's decision
    child1.state.active_players = MagicMock(return_value=[mock_child1_active_player])

    # Apply virtual loss as if it happened during selection
    player_idx_at_root = mcts_dummy.get_player_index(root.state)
    root.child_total_action_values[0, player_idx_at_root] -= mcts_dummy.virtual_loss_c
    
    player_idx_at_child1 = mcts_dummy.get_player_index(child1.state)
    child1.child_total_action_values[1, player_idx_at_child1] -= mcts_dummy.virtual_loss_c


    mcts_dummy.backpropagate(child2, value_vector)

    # Leaf node (child2)
    assert child2.node_total_visits == 1, "Leaf's N(s) should be incremented (VL test)."

    # Parent of leaf (child1)
    # Expected W = initial_W_after_VL_application + value_vector + VL_c_reverted
    # initial_W_after_VL_application for player 0 was -1.0. For others, 0.
    expected_W_child1 = np.copy(value_vector)
    expected_W_child1[0] += -mcts_dummy.virtual_loss_c # This was the state before backprop
    expected_W_child1[0] += mcts_dummy.virtual_loss_c  # VL reverted
    # So expected_W_child1 should be just value_vector after VL logic
    assert np.allclose(child1.child_total_action_values[1, :], value_vector), \
        f"Expected {value_vector}, got {child1.child_total_action_values[1, :]} (VL test for child1)"
    assert np.allclose(child1.child_average_action_values[1, :], value_vector / 1.0)

    # Root node
    expected_W_root = np.copy(value_vector)
    expected_W_root[0] += -mcts_dummy.virtual_loss_c # State before backprop
    expected_W_root[0] += mcts_dummy.virtual_loss_c # VL reverted
    assert np.allclose(root.child_total_action_values[0, :], value_vector), \
        f"Expected {value_vector}, got {root.child_total_action_values[0, :]} (VL test for root)"
    assert np.allclose(root.child_average_action_values[0, :], value_vector / 1.0)


# I. Virtual Loss Mechanics
def test_mcts_virtual_loss_application_in_selection_and_reversion_in_backprop(game_objects, model):
    initial_game_state, _, action_mapper, encoder = game_objects
    num_players = 4
    action_enc_size = action_mapper.action_encoding_size
    virtual_loss_val = 2.0 # Use a distinct value

    mcts = MCTS(
        root_state=initial_game_state, model=model, encoder=encoder, action_mapper=action_mapper,
        num_simulations=1, device=torch.device("cpu"),
        apply_virtual_loss=True, virtual_loss_c=virtual_loss_val
    )

    # Setup root and one child
    root = mcts.root
    
    # Mock root state for consistent player_idx
    mock_root_active_player = MagicMock()
    mock_root_active_player.id = mcts.player_idx_to_id[0] # Assume player 0 is active
    root.state.active_players = MagicMock(return_value=[mock_root_active_player])
    player_idx_at_root = 0


    # Expand root first (to get priors) - this is done in mcts.search normally
    # For this test, we'll manually set up a child and its prior
    # Let's assume action 0 is a valid child
    action_to_select = 0
    mock_child_state = initial_game_state.clone(initial_game_state.raw_actions)
    # Ensure action_to_select is considered "expanded" for select_child to pick it
    child_node = root.expand_child(action_to_select, mock_child_state)
    
    root.child_prior_probabilities[action_to_select] = 0.8 
    root.child_visit_counts[action_to_select] = 0 # Initially 0 visits
    root.child_total_action_values[action_to_select, :] = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32) # Initial W
    root.child_average_action_values[action_to_select, :] = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32) # Q is initially 0
    root.node_total_visits = 1 # Give root some visits so sqrt_N_s is not 0 for UCB

    initial_W_sa = np.copy(root.child_total_action_values[action_to_select, :])

    # 1. Simulate Selection step (part of mcts.search loop)
    #    select_child is called, then N(s,a) and N(s) are incremented, then VL is applied.
    
    # Manually increment counts as search would
    root.child_visit_counts[action_to_select] += 1
    root.node_total_visits +=1 # This N_s is for the parent (root)

    # Apply virtual loss (as done in MCTS.search)
    root.child_total_action_values[action_to_select, player_idx_at_root] -= virtual_loss_val
    
    # Update Q(s,a) after VL and N(s,a) increment (as done in MCTS.search)
    root.child_average_action_values[action_to_select, :] = \
        root.child_total_action_values[action_to_select, :] / \
        root.child_visit_counts[action_to_select]

    expected_W_after_select = np.copy(initial_W_sa)
    expected_W_after_select[player_idx_at_root] -= virtual_loss_val
    assert np.allclose(root.child_total_action_values[action_to_select, :], expected_W_after_select), \
        "W(s,a) after VL application in selection is incorrect."
    
    expected_Q_after_select = expected_W_after_select / root.child_visit_counts[action_to_select]
    assert np.allclose(root.child_average_action_values[action_to_select, :], expected_Q_after_select), \
        "Q(s,a) after VL application in selection is incorrect."

    # 2. Simulate Backpropagation
    #    The 'child_node' is the leaf of this path.
    value_from_leaf = np.array([0.5, -0.5, 0.2, -0.2], dtype=np.float32)
    
    # Backpropagate from child_node (which is root.children[action_to_select])
    # The backprop logic will update stats for the edge (root, action_to_select)
    
    # First, leaf's N(s) is incremented
    child_node.node_total_visits = 0 # Reset for clarity
    mcts.backpropagate(child_node, value_from_leaf)
    assert child_node.node_total_visits == 1

    # Check stats on root for action_to_select
    # Expected W = (initial_W_sa - VL) + value_from_leaf + VL_reverted
    # So, W = initial_W_sa + value_from_leaf
    expected_W_after_backprop = initial_W_sa + value_from_leaf
    assert np.allclose(root.child_total_action_values[action_to_select, :], expected_W_after_backprop), \
        "W(s,a) after backpropagation with VL is incorrect."

    # N(s,a) for (root, action_to_select) was already 1 from selection.
    # Q(s,a) = W_after_backprop / N(s,a)
    expected_Q_after_backprop = expected_W_after_backprop / root.child_visit_counts[action_to_select]
    assert np.allclose(root.child_average_action_values[action_to_select, :], expected_Q_after_backprop), \
        "Q(s,a) after backpropagation with VL is incorrect."

    # Clean up mocked method
    if hasattr(root.state, 'active_players'): delattr(root.state, 'active_players')


# II. expand_and_evaluate Edge Cases

def test_expand_evaluate_policy_sum_zero_fallback(game_objects, model):
    initial_game_state, _, action_mapper, encoder = game_objects
    leaf_node = Node(parent=None, state=initial_game_state, num_players=4, action_encoding_size=action_mapper.action_encoding_size)

    mcts = MCTS(
        root_state=initial_game_state, model=model, encoder=encoder, action_mapper=action_mapper,
        num_simulations=1, device=torch.device("cpu")
    )

    # Mock model to return policy that sums to zero after masking
    mock_policy_logits = torch.ones(1, action_mapper.action_encoding_size) * -100 # Very small probabilities
    mock_value = torch.randn(1, 4)
    mcts.model = MagicMock(return_value=(mock_policy_logits, mock_value))

    # Mock legal actions: e.g., actions 0, 1, 2 are legal
    legal_mask = np.zeros(action_mapper.action_encoding_size, dtype=bool)
    legal_indices = [0, 1, 2]
    if not legal_indices or max(legal_indices) >= len(legal_mask): # Ensure valid indices
        pytest.skip("Action encoding size too small for test legal_indices.")
    for i in legal_indices: legal_mask[i] = True
    
    num_legal_moves = np.sum(legal_mask)
    if num_legal_moves == 0:
        pytest.skip("Test setup error: No legal moves for policy sum zero fallback test.")

    mcts.action_mapper.get_legal_action_mask = MagicMock(return_value=legal_mask)
    
    # Mock state cloning and processing for child expansion
    initial_game_state.clone = MagicMock(return_value=initial_game_state)
    initial_game_state.process_action = MagicMock()


    value_vector = mcts.expand_and_evaluate(leaf_node)

    assert value_vector is not None # Should still get a value from NN
    expected_prior = 1.0 / num_legal_moves if num_legal_moves > 0 else 0
    
    for i in range(action_mapper.action_encoding_size):
        if i in legal_indices:
            assert np.isclose(leaf_node.child_prior_probabilities[i], expected_prior)
            assert i in leaf_node.children # Children should be expanded
        else:
            assert np.isclose(leaf_node.child_prior_probabilities[i], 0.0)
            assert i not in leaf_node.children
    
    assert np.isclose(np.sum(leaf_node.child_prior_probabilities), 1.0) if num_legal_moves > 0 else np.isclose(np.sum(leaf_node.child_prior_probabilities), 0.0)


def test_expand_evaluate_terminal_state_value_vector_detailed(game_objects, model):
    initial_game_state, _, action_mapper, encoder = game_objects
    num_players = 4 # Must match game_objects setup

    mcts = MCTS(
        root_state=initial_game_state, model=model, encoder=encoder, action_mapper=action_mapper,
        num_simulations=1, device=torch.device("cpu")
    )
    # Ensure player_id_to_idx is set up for 4 players (0, 1, 2, 3)
    # The fixture game_objects creates players "1", "2", "3", "4"
    # MCTS constructor sorts them by ID, so "1"->0, "2"->1, "3"->2, "4"->3 if IDs are strings.
    # Let's verify this assumption or make it explicit for the test.
    expected_player_ids = sorted([p.id for p in initial_game_state.players])


    scenarios = [
        ({"1": 100, "2": 50, "3": 0, "4": 0}, np.array([1.0, -1.0, -1.0, -1.0])), # P1 wins
        ({"1": 75, "2": 75, "3": 0, "4": 0}, np.array([1.0, 1.0, -1.0, -1.0])),  # P1, P2 draw win
        ({"1": 10, "2": 20, "3": 5, "4": 30}, np.array([-1.0, -1.0, -1.0, 1.0])),# P4 wins
        ({"1": 50, "2": 50, "3": 50, "4": 50}, np.array([1.0, 1.0, 1.0, 1.0])), # All draw
    ]

    for game_result, expected_value_vector in scenarios:
        mock_terminal_state = MagicMock(spec=BaseGame)
        mock_terminal_state.finished = True
        mock_terminal_state.result = MagicMock(return_value=game_result)
        # For player_id_to_idx mapping in MCTS.expand_and_evaluate
        mock_terminal_state.players = initial_game_state.players 
        
        leaf_node = Node(parent=None, state=mock_terminal_state, num_players=num_players, action_encoding_size=action_mapper.action_encoding_size)
        
        # Need to ensure mcts.player_id_to_idx is correctly mapping the keys in game_result
        # The MCTS instance is initialized with initial_game_state, so its player_id_to_idx is based on that.
        # We must use player IDs consistent with initial_game_state.players for game_result keys.
        
        # Re-map game_result keys if they are generic like "player_0" to actual player IDs from mcts
        # For this test, the keys "1", "2", "3", "4" in scenarios are fine as they match fixture.

        value_vector = mcts.expand_and_evaluate(leaf_node)
        assert np.allclose(value_vector, expected_value_vector), \
            f"Terminal state result {game_result} produced {value_vector}, expected {expected_value_vector}"


# III. select_child Edge Cases

@pytest.mark.parametrize("is_root", [True, False])
def test_select_child_dirichlet_noise_application_at_root_only(game_objects, model, is_root):
    initial_game_state, _, action_mapper, encoder = game_objects
    alpha = 0.3
    noise_factor = 0.25
    c_puct = 1.0

    mcts = MCTS(
        root_state=initial_game_state, model=model, encoder=encoder, action_mapper=action_mapper,
        num_simulations=1, c_puct=c_puct, alpha=alpha, noise_factor=noise_factor,
        device=torch.device("cpu")
    )

    # Mock active player for get_player_index
    mock_active_player = MagicMock()
    mock_active_player.id = mcts.player_idx_to_id[0]
    initial_game_state.active_players = MagicMock(return_value=[mock_active_player])


    parent_node = Node(None, initial_game_state, mcts.num_players, mcts.action_encoding_size)
    if not is_root: # If testing a non-root node, give it a dummy parent to distinguish from mcts.root
        dummy_grandparent = Node(None, initial_game_state, mcts.num_players, mcts.action_encoding_size)
        parent_node.parent = dummy_grandparent
    else: # If testing root, parent_node should be mcts.root
        mcts.root = parent_node


    # Children A (idx 0) and B (idx 1)
    action_A, action_B = 0, 1
    child_A_state = initial_game_state.clone(initial_game_state.raw_actions)
    child_B_state = initial_game_state.clone(initial_game_state.raw_actions)
    parent_node.expand_child(action_A, child_A_state)
    parent_node.expand_child(action_B, child_B_state)

    # Stats: P(A)=0.6, P(B)=0.4. Q=0, N_child=0 for both. N_parent=1 (to make sqrt_N_s = 1)
    parent_node.child_prior_probabilities[action_A] = 0.6
    parent_node.child_prior_probabilities[action_B] = 0.4
    parent_node.child_visit_counts[action_A] = 0
    parent_node.child_visit_counts[action_B] = 0
    parent_node.child_average_action_values[:, :] = 0.0
    parent_node.node_total_visits = 1 # So sqrt_N_s is 1

    # PUCT = Q + c_puct * P * sqrt(N_s) / (1 + N_sa)
    # Here Q=0, N_sa=0, N_s=1, c_puct=1.0. So PUCT = P_noisy or P_orig.

    original_dirichlet = np.random.dirichlet
    try:
        if is_root:
            # Mock dirichlet to favor action_B (original P(B)=0.4)
            # Noise: e.g., [0.1 for A, 0.9 for B]
            # P_noisy_A = (1-noise_factor)*0.6 + noise_factor*0.1 = 0.75*0.6 + 0.25*0.1 = 0.45 + 0.025 = 0.475
            # P_noisy_B = (1-noise_factor)*0.4 + noise_factor*0.9 = 0.75*0.4 + 0.25*0.9 = 0.30 + 0.225 = 0.525
            # So B should be chosen
            np.random.dirichlet = MagicMock(return_value=np.array([0.1, 0.9], dtype=np.float32))
            expected_action = action_B
        else:
            # No noise for non-root, A (P=0.6) should be chosen over B (P=0.4)
            expected_action = action_A

        selected_action, _ = mcts.select_child(parent_node)
        assert selected_action == expected_action
    finally:
        np.random.dirichlet = original_dirichlet # Restore
        if hasattr(initial_game_state, 'active_players'): delattr(initial_game_state, 'active_players')


def test_select_child_parent_node_total_visits_zero(game_objects, model):
    initial_game_state, _, action_mapper, encoder = game_objects
    mcts = MCTS(
        root_state=initial_game_state, model=model, encoder=encoder, action_mapper=action_mapper,
        num_simulations=1, c_puct=1.0, device=torch.device("cpu")
    )
    
    # Mock active player for get_player_index
    mock_active_player = MagicMock()
    mock_active_player.id = mcts.player_idx_to_id[0]
    initial_game_state.active_players = MagicMock(return_value=[mock_active_player])

    parent_node = Node(None, initial_game_state, mcts.num_players, mcts.action_encoding_size)
    parent_node.node_total_visits = 0 # Key condition for this test

    action_A, action_B = 0, 1
    child_A_state = initial_game_state.clone(initial_game_state.raw_actions)
    child_B_state = initial_game_state.clone(initial_game_state.raw_actions)
    parent_node.expand_child(action_A, child_A_state)
    parent_node.expand_child(action_B, child_B_state)

    parent_node.child_prior_probabilities[action_A] = 0.3
    parent_node.child_prior_probabilities[action_B] = 0.7 # B has higher prior
    parent_node.child_visit_counts[action_A] = 0
    parent_node.child_visit_counts[action_B] = 0
    parent_node.child_average_action_values[:, :] = 0.0 # Q values are zero

    # PUCT = Q + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
    # If N(s) = 0, sqrt_N_s in MCTS.py is 1.0.
    # So U = c_puct * P(s,a) * 1.0 / (1 + 0) = c_puct * P(s,a)
    # Selection should be based on highest P(s,a) when Qs are 0.
    selected_action, _ = mcts.select_child(parent_node)
    assert selected_action == action_B, "Should select child with highest prior when N(s)=0 and Qs are 0."
    
    if hasattr(initial_game_state, 'active_players'): delattr(initial_game_state, 'active_players')


# IV. get_policy Edge Cases

@pytest.mark.parametrize("priors_exist", [True, False])
def test_get_policy_root_children_all_zero_visits(game_objects, model, priors_exist):
    initial_game_state, _, action_mapper, encoder = game_objects
    mcts = MCTS(
        root_state=initial_game_state, model=model, encoder=encoder, action_mapper=action_mapper,
        num_simulations=0, # No simulations run, so visits will be 0
        device=torch.device("cpu")
    )
    # Manually expand root children (as if expand_and_evaluate was called once on root)
    # This part is tricky because mcts.search() calls expand_and_evaluate if root is leaf.
    # For this test, let's directly manipulate mcts.root after MCTS init.
    
    # Simulate root expansion
    child_action_indices = [0, 1] # Assume actions 0 and 1 are expanded
    if max(child_action_indices) >= mcts.action_encoding_size:
        pytest.skip("Action encoding size too small for test legal_indices.")

    for action_idx in child_action_indices:
        mock_child_state = initial_game_state.clone(initial_game_state.raw_actions)
        mcts.root.expand_child(action_idx, mock_child_state)
        mcts.root.child_visit_counts[action_idx] = 0 # Explicitly zero visits
        if priors_exist:
            mcts.root.child_prior_probabilities[action_idx] = 0.2 + action_idx * 0.1 # e.g. P(0)=0.2, P(1)=0.3
        else:
            mcts.root.child_prior_probabilities[action_idx] = 0.0

    if priors_exist:
        # Normalize the manually set priors for expectation
        current_priors = mcts.root.child_prior_probabilities[child_action_indices]
        expected_probs_dict = {idx: p/np.sum(current_priors) for idx, p in zip(child_action_indices, current_priors)}
    else: # Fallback to uniform over explored children
        num_explored = len(child_action_indices)
        expected_probs_dict = {idx: 1.0/num_explored for idx in child_action_indices}


    policy_dict, _ = mcts.get_policy(temperature=1.0)

    assert len(policy_dict) == len(child_action_indices)
    for action_idx, prob in expected_probs_dict.items():
        assert action_idx in policy_dict
        assert np.isclose(policy_dict[action_idx], prob)


def test_get_policy_temperature_zero_with_ties(game_objects, model):
    initial_game_state, _, action_mapper, encoder = game_objects
    mcts = MCTS(
        root_state=initial_game_state, model=model, encoder=encoder, action_mapper=action_mapper,
        num_simulations=10, device=torch.device("cpu") # Sims to populate visits
    )

    # Manually set up root children and visit counts with ties
    tied_actions = [0, 1]
    other_action = 2
    if max(tied_actions + [other_action]) >= mcts.action_encoding_size:
        pytest.skip("Action encoding size too small for test.")

    for action_idx in tied_actions + [other_action]:
        mock_child_state = initial_game_state.clone(initial_game_state.raw_actions)
        mcts.root.expand_child(action_idx, mock_child_state)
    
    mcts.root.child_visit_counts[tied_actions[0]] = 10
    mcts.root.child_visit_counts[tied_actions[1]] = 10 # Tie
    mcts.root.child_visit_counts[other_action] = 5

    policy_dict, _ = mcts.get_policy(temperature=0.0)

    assert len(policy_dict) == 1 # Only one action should have prob 1.0
    chosen_action = list(policy_dict.keys())[0]
    assert chosen_action in tied_actions # Must be one of the actions with max visits
    assert np.isclose(policy_dict[chosen_action], 1.0)


# V. MCTS and Node General Robustness

def test_mcts_get_player_index_invalid_player(game_objects, model):
    initial_game_state, _, action_mapper, encoder = game_objects
    mcts = MCTS(
        root_state=initial_game_state, model=model, encoder=encoder, action_mapper=action_mapper,
        num_simulations=1, device=torch.device("cpu")
    )

    mock_state_with_invalid_player = MagicMock(spec=BaseGame)
    invalid_player = MagicMock()
    invalid_player.id = "INVALID_PLAYER_ID_XYZ" # ID not in mcts.player_id_to_idx
    mock_state_with_invalid_player.active_players = MagicMock(return_value=[invalid_player])

    with pytest.raises(ValueError, match="Player ID 'INVALID_PLAYER_ID_XYZ' not found"):
        mcts.get_player_index(mock_state_with_invalid_player)

def test_expand_and_evaluate_error_during_child_state_generation(game_objects, model):
    initial_game_state, _, action_mapper, encoder = game_objects
    leaf_node = Node(parent=None, state=initial_game_state, num_players=4, action_encoding_size=action_mapper.action_encoding_size)

    mcts = MCTS(
        root_state=initial_game_state, model=model, encoder=encoder, action_mapper=action_mapper,
        num_simulations=1, device=torch.device("cpu")
    )

    # Mock model to return a policy that tries to expand action 0
    mock_policy_logits = torch.zeros(1, action_mapper.action_encoding_size)
    if 0 < action_mapper.action_encoding_size :
        mock_policy_logits[0, 0] = 10 # High logit for action 0
    else:
        pytest.skip("Action encoding size is 0.")

    mock_value = torch.randn(1, 4)
    mcts.model = MagicMock(return_value=(mock_policy_logits, mock_value))

    # Mock legal mask to make action 0 legal
    legal_mask = np.zeros(action_mapper.action_encoding_size, dtype=bool)
    if 0 < action_mapper.action_encoding_size : legal_mask[0] = True
    mcts.action_mapper.get_legal_action_mask = MagicMock(return_value=legal_mask)
    
    # Mock action_mapper.map_index_to_action to raise an error for action 0
    class CustomExpansionError(Exception): pass
    def faulty_map_index_to_action(action_index, state):
        if action_index == 0:
            raise CustomExpansionError("Failed to map action 0")
        # Fallback for other actions if any (though policy focuses on 0)
        return MagicMock() 
    mcts.action_mapper.map_index_to_action = MagicMock(side_effect=faulty_map_index_to_action)
    
    # Mock state cloning
    initial_game_state.clone = MagicMock(return_value=initial_game_state)


    with pytest.raises(CustomExpansionError, match="Failed to map action 0"):
        mcts.expand_and_evaluate(leaf_node)


def test_node_repr_robustness(game_objects):
    initial_game_state, _, action_mapper, _ = game_objects
    parent_node = Node(None, initial_game_state, 4, action_mapper.action_encoding_size)
    
    # Child with an action_index that is out of bounds for parent's arrays
    # Case 1: action_index too large
    child_node_invalid_idx_large = Node(parent_node, initial_game_state, 4, action_mapper.action_encoding_size, action_index=action_mapper.action_encoding_size + 5)
    
    # Case 2: action_index is negative (though Optional[int] might make None more likely for root)
    child_node_invalid_idx_neg = Node(parent_node, initial_game_state, 4, action_mapper.action_encoding_size, action_index=-1)

    # Case 3: action_index is None (like root, but this child has a parent)
    # The __repr__ already handles action_index is None by not trying to access parent stats.
    # The problematic case is when action_index is not None but invalid for array access.

    try:
        repr_large = repr(child_node_invalid_idx_large)
        assert "InvalidActionIdx" in repr_large
        assert "Q_edge=InvalidActionIdx" in repr_large
        assert "P_edge=InvalidActionIdx" in repr_large
        assert "N_edge=InvalidActionIdx" in repr_large

        repr_neg = repr(child_node_invalid_idx_neg)
        assert "InvalidActionIdx" in repr_neg
        assert "Q_edge=InvalidActionIdx" in repr_neg
        assert "P_edge=InvalidActionIdx" in repr_neg
        assert "N_edge=InvalidActionIdx" in repr_neg

    except IndexError:
        pytest.fail("Node.__repr__ raised IndexError with invalid action_index.")

# --- End of new tests ---
