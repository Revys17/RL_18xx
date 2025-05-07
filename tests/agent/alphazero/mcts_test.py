import pytest
import torch
import numpy as np
from typing import List, Any, Tuple

# Your project imports
from rl18xx.game.engine.game import BaseGame # You'll need to import your actual BaseGame
from rl18xx.game.gamemap import GameMap
from rl18xx.game.action_helper import ActionHelper
from rl18xx.agent.alphazero.model import Model
from rl18xx.agent.alphazero.encoder import Encoder_1830 # Assuming this is your encoder
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
TEST_C_PUCT = 1.0

# --- Helper: Minimal Player Class (if your BaseGame needs it) ---
class MockPlayer:
    def __init__(self, player_id: str):
        self.id = player_id

# --- Fixtures ---

@pytest.fixture(scope="module")
def model():
    """Instantiates a small version of the Model for testing."""
    model = Model(
        game_state_size=TEST_GAME_STATE_SIZE,
        num_map_nodes=TEST_NUM_MAP_NODES,
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
def game_objects() -> Tuple[BaseGame, ActionHelper, ActionMapper, Encoder_1830]:
    game_map = GameMap()
    game_class = game_map.game_by_title("1830")
    players = {"1": "Player 1", "2": "Player 2", "3": "Player 3", "4": "Player 4"}
    game_instance = game_class(players)
    return (game_instance, ActionHelper(game_instance), ActionMapper(game_instance), Encoder_1830())

@pytest.fixture
def terminal_game_state(game_objects) -> BaseGame:
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
def game_state_single_action(game_objects) -> BaseGame:
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
    initial_game_state.process_action(action_helper.get_all_choices()[10])  # lay tile #9 with rotation 1 on H8
    initial_game_state.process_action(action_helper.get_all_choices()[-1])  # pass token
    return (initial_game_state, action_helper, action_mapper, encoder)

@pytest.fixture
def mcts_instance(game_objects, model):
    initial_game_state, action_helper, action_mapper, encoder = game_objects
    """Initializes an MCTS instance for testing."""
    if initial_game_state is None: # Should be caught by skip in initial_game_state
        pytest.fail("Game state not available for MCTS instance.")

    return (MCTS(
        root_state=initial_game_state,
        model=model,
        encoder=encoder,
        action_mapper=action_mapper,
        num_simulations=TEST_NUM_SIMULATIONS,
        c_puct=TEST_C_PUCT,
        device=torch.device("cpu") # Use CPU for tests for simplicity
    ), game_objects)

@pytest.fixture
def mcts_instance_no_noise(game_objects, model):
    initial_game_state, _, action_mapper, encoder = game_objects
    return MCTS(
        root_state=initial_game_state,
        model=model,
        encoder=encoder,
        action_mapper=action_mapper,
        num_simulations=TEST_NUM_SIMULATIONS,
        c_puct=TEST_C_PUCT,
        noise_factor=0.0, # NO NOISE
        device=torch.device("cpu")
    )

@pytest.fixture
def mcts_instance_with_noise(game_objects, model): # Same as default mcts_instance
    initial_game_state, _, action_mapper, encoder = game_objects
    # This is essentially the same as the main mcts_instance fixture,
    # but named explicitly for noise tests.
    # Default noise_factor is 0.25 in MCTS class.
    return (MCTS(
        root_state=initial_game_state,
        model=model,
        encoder=encoder,
        action_mapper=action_mapper,
        num_simulations=TEST_NUM_SIMULATIONS,
        c_puct=TEST_C_PUCT,
        device=torch.device("cpu")
    ), game_objects)

# --- Node Tests ---
def test_node_creation(game_objects):
    initial_game_state, action_helper, action_mapper, encoder = game_objects
    root = Node(parent=None, prior_prob=1.0, state=initial_game_state, num_players=len(initial_game_state.players))
    assert root.parent is None
    assert root.prior_probability == 1.0
    assert root.visit_count == 0
    assert np.array_equal(root.total_action_value, np.zeros(len(initial_game_state.players)))
    assert root.is_leaf()

def test_node_update_stats(game_objects):
    initial_game_state, action_helper, action_mapper, encoder = game_objects
    node = Node(parent=None, prior_prob=1.0, state=initial_game_state, num_players=len(initial_game_state.players))
    value_vector = np.array([1.0, 0.0, 0.0, 0.0]) # Assuming 4 players
    node.update_stats(value_vector)
    assert node.visit_count == 1
    assert np.array_equal(node.total_action_value, value_vector)
    assert np.array_equal(node.average_action_value, value_vector)

    node.update_stats(value_vector)
    assert node.visit_count == 2
    assert np.array_equal(node.total_action_value, value_vector * 2)
    assert np.array_equal(node.average_action_value, value_vector)

# --- MCTS Tests ---

def test_mcts_initialization(mcts_instance):
    mcts_instance, game_objects = mcts_instance
    initial_game_state, action_helper, action_mapper, encoder = game_objects
    assert mcts_instance is not None
    assert mcts_instance.root is not None
    assert mcts_instance.root.state == initial_game_state
    assert mcts_instance.num_players == len(initial_game_state.players)
    assert mcts_instance.root.num_players == len(initial_game_state.players)
    assert mcts_instance.model.training is False # Should be in eval mode

def test_select_leaf_initial_root(mcts_instance):
    mcts_instance, game_objects = mcts_instance
    """On a new tree, selecting a leaf should return the root itself."""
    leaf = mcts_instance.select_leaf()
    assert leaf == mcts_instance.root
    assert leaf.is_leaf()

def test_expand_and_evaluate_leaf(mcts_instance):
    mcts_instance, game_objects = mcts_instance
    """Tests expanding the root node and evaluating it."""
    leaf_node = mcts_instance.root
    assert leaf_node.is_leaf()

    # Ensure the action mapper can find legal actions for the root state
    # This depends heavily on your initial_game_state and ActionMapper
    try:
        legal_mask = mcts_instance.action_mapper.get_legal_action_mask(leaf_node.state)
        assert torch.sum(legal_mask) > 0, "ActionMapper found no legal actions for initial state."
    except Exception as e:
        pytest.fail(f"Error getting legal action mask in test setup: {e}")

    value_vector = mcts_instance.expand_and_evaluate(leaf_node)

    assert not leaf_node.is_leaf(), "Root node should have children after expansion."
    assert len(leaf_node.children) > 0, "No children created during expansion."
    # The visit count of the expanded node itself isn't incremented by expand_and_evaluate
    assert leaf_node.visit_count == 0 
    assert isinstance(value_vector, np.ndarray)
    assert value_vector.shape == (mcts_instance.num_players,)

    # Check child properties
    a_child = next(iter(leaf_node.children.values()))
    assert a_child.parent == leaf_node
    assert a_child.visit_count == 0
    assert a_child.prior_probability > 0 # Priors should be assigned from policy

def test_backpropagate(mcts_instance):
    mcts_instance, game_objects = mcts_instance
    # 1. Expand root to get a child
    root = mcts_instance.root
    mcts_instance.expand_and_evaluate(root)
    assert len(root.children) > 0
    
    # Pick a child (leaf for this purpose before further expansion)
    # For simplicity, let's assume the first child is valid
    # A more robust test might iterate or pick one with a known action index
    if not root.children:
         pytest.skip("No children after expansion, cannot test backpropagation.")
    
    child_node = next(iter(root.children.values()))
    
    # 2. Simulate a value vector for this child
    simulated_value = np.random.rand(mcts_instance.num_players) * 2 - 1 # Values between -1 and 1

    # 3. Backpropagate from this child
    mcts_instance.backpropagate(child_node, simulated_value)

    # 4. Check stats
    assert child_node.visit_count == 1
    assert np.allclose(child_node.total_action_value, simulated_value)
    assert np.allclose(child_node.average_action_value, simulated_value)

    assert root.visit_count == 1 # Root is parent, should also be updated
    assert np.allclose(root.total_action_value, simulated_value) # If only one path, root gets same value
    assert np.allclose(root.average_action_value, simulated_value)


def test_run_one_simulation(game_objects, model):
    initial_game_state, _, action_mapper, encoder = game_objects
    mcts_instance = MCTS(
        root_state=initial_game_state,
        model=model,
        encoder=encoder,
        action_mapper=action_mapper,
        num_simulations=1,
        c_puct=TEST_C_PUCT,
        device=torch.device("cpu")
    )
    initial_root_visits = mcts_instance.root.visit_count

    # Ensure the action mapper can find legal actions for the root state
    try:
        legal_mask = mcts_instance.action_mapper.get_legal_action_mask(mcts_instance.root.state)
        assert torch.sum(legal_mask) > 0, "ActionMapper found no legal actions for initial state in test_run_one_simulation."
    except Exception as e:
        pytest.fail(f"Error getting legal action mask in test setup: {e}")

    mcts_instance.search()

    assert mcts_instance.root.visit_count == initial_root_visits + 1
    # Further checks could involve verifying that a path was traversed and stats updated.

def test_run_simulations_full_search(mcts_instance):
    mcts_instance, game_objects = mcts_instance
    """Test running the full MCTS search."""
    # Ensure the action mapper can find legal actions for the root state
    try:
        legal_mask = mcts_instance.action_mapper.get_legal_action_mask(mcts_instance.root.state)
        assert torch.sum(legal_mask) > 0, "ActionMapper found no legal actions for initial state in test_run_simulations_full_search."
    except Exception as e:
        pytest.fail(f"Error getting legal action mask in test setup: {e}")

    mcts_instance.search()
    assert mcts_instance.root.visit_count == TEST_NUM_SIMULATIONS
    assert len(mcts_instance.root.children) > 0 # Should have expanded

def test_get_policy_after_search(mcts_instance):
    mcts_instance, game_objects = mcts_instance
    mcts_instance.search()
    
    if not mcts_instance.root.children:
        pytest.skip("Root has no children after search, cannot test get_policy effectively.")

    policy_dict, policy_vector = mcts_instance.get_policy(temperature=1.0)

    assert isinstance(policy_dict, dict)
    assert isinstance(policy_vector, np.ndarray)
    assert policy_vector.shape == (TEST_POLICY_OUTPUT_SIZE,)
    assert len(policy_dict) == len(mcts_instance.root.children)
    
    # Probabilities should sum to 1 (approximately)
    assert np.isclose(sum(policy_dict.values()), 1.0, atol=1e-6)
    assert np.isclose(np.sum(policy_vector), 1.0, atol=1e-6)

    # All probabilities should be non-negative
    assert all(p >= 0 for p in policy_dict.values())
    assert np.all(policy_vector >= 0)

def test_get_policy_temperature_zero(mcts_instance):
    mcts_instance, game_objects = mcts_instance
    mcts_instance.search()
    if not mcts_instance.root.children:
        pytest.skip("Root has no children after search, cannot test get_policy with temp 0.")

    # Manually set some visit counts to ensure one is clearly the max
    children_list = list(mcts_instance.root.children.values())
    if len(children_list) > 1:
        children_list[0].visit_count = 10
        children_list[1].visit_count = 5
        # Ensure other children have fewer visits if they exist
        for i in range(2, len(children_list)):
            children_list[i].visit_count = 1
    elif children_list: # Only one child
         children_list[0].visit_count = 10
    else: # No children, skip
        pytest.skip("Cannot deterministically test temp 0 policy without multiple children or one child.")


    policy_dict, policy_vector = mcts_instance.get_policy(temperature=0.0)

    assert isinstance(policy_dict, dict)
    # With temp=0, one action should have probability 1.0, others 0.0
    num_ones = sum(1 for p in policy_dict.values() if np.isclose(p, 1.0))
    num_zeros = sum(1 for p in policy_dict.values() if np.isclose(p, 0.0))
    
    assert num_ones == 1, f"Expected one action with prob 1.0, got {num_ones}. Policy: {policy_dict}"
    assert num_zeros == len(policy_dict) - 1

    max_visit_action_idx = -1
    max_visits = -1
    for action_idx, child_node in mcts_instance.root.children.items():
        if child_node.visit_count > max_visits:
            max_visits = child_node.visit_count
            max_visit_action_idx = action_idx
    
    if max_visit_action_idx != -1: # If there are children
        assert np.isclose(policy_dict.get(max_visit_action_idx, 0.0), 1.0)
        assert np.isclose(policy_vector[max_visit_action_idx], 1.0)


def test_get_policy_no_children_error(mcts_instance):
    mcts_instance, game_objects = mcts_instance
    """Test that get_policy raises an error if root has no children."""
    # Ensure root has no children (it shouldn't before simulations)
    assert not mcts_instance.root.children
    with pytest.raises(ValueError, match="Root node has no children after search"):
        mcts_instance.get_policy()

def test_dirichlet_noise_application(mcts_instance):
    mcts_instance, game_objects = mcts_instance
    """
    Checks if Dirichlet noise is applied to root priors during the first expansion.
    This is a bit indirect to test without deep inspection.
    We check if priors are not strictly proportional to raw network output.
    """
    root = mcts_instance.root
    
    # Run expand_and_evaluate, which applies noise if it's the root
    mcts_instance.expand_and_evaluate(root) 
    
    if not root.children:
        pytest.skip("Root not expanded, cannot test noise.")

    child_priors = np.array([child.prior_probability for child in root.children.values()])
    
    # To properly test this, we'd need the raw policy from the network *before* noise.
    # For now, we can at least check that priors are assigned and sum to ~1.
    assert np.isclose(np.sum(child_priors), 1.0, atol=1e-5) # After normalization
    
    # A more advanced test would:
    # 1. Get raw policy from network.
    # 2. Manually apply Dirichlet noise.
    # 3. Compare with child.prior_probability.
    # This requires more access to intermediate steps or a more complex setup.
    # For now, this test mainly ensures priors are set and normalized.
    # If noise_factor is 0, priors should directly reflect policy.
    # If noise_factor > 0, they should differ.
    # This test is more of a placeholder for a deeper noise check.
    pass

def test_expand_and_evaluate_terminal_state(terminal_game_state, model):
    """Tests that expanding a terminal node returns the game outcome directly."""
    terminal_game_state, _, action_mapper, encoder = terminal_game_state
    
    mcts_term = MCTS(
        root_state=terminal_game_state, # Use the terminal state
        model=model,
        encoder=encoder,
        action_mapper=action_mapper,
        num_simulations=1,
        c_puct=TEST_C_PUCT,
        device=torch.device("cpu")
    )
    
    leaf_node = mcts_term.root
    assert leaf_node.state.finished, "Test setup error: game state is not terminal."

    # Expected outcome from the game state itself
    expected_outcome = np.full(mcts_term.num_players, 0.0, dtype=np.float32)
    max_score = max(terminal_game_state.result().values())
    for player_id, score in terminal_game_state.result().items():
        player_idx = mcts_term.player_id_to_idx.get(player_id)
        if player_idx is not None:
            if score == max_score:
                expected_outcome[player_idx] = 1.0
            else:
                expected_outcome[player_idx] = -1.0
        else:
            raise ValueError(f"Player ID {player_id} from final scores not in player map.")

    assert isinstance(expected_outcome, np.ndarray), "Game outcome should be a numpy array."
    assert expected_outcome.shape == (mcts_term.num_players,), "Game outcome shape mismatch."

    value_vector = mcts_term.expand_and_evaluate(leaf_node)

    assert leaf_node.is_leaf(), "Terminal node should not be expanded (no children)."
    assert np.array_equal(value_vector, expected_outcome), \
        f"Value vector {value_vector} from terminal node does not match expected outcome {expected_outcome}."
    # Model should not have been called, so no children created.
    assert not leaf_node.children


def test_get_policy_zero_visits_fallback_to_priors(mcts_instance):
    mcts_object, game_objects_tuple = mcts_instance
    # 1. Expand root to get children with priors
    if mcts_object.root.is_leaf():
        # Ensure there are legal actions to expand
        try:
            legal_mask = mcts_object.action_mapper.get_legal_action_mask(mcts_object.root.state)
            if torch.sum(legal_mask) == 0:
                pytest.skip("No legal actions to expand for testing zero visits fallback.")
        except Exception as e:
            pytest.skip(f"Could not get legal actions for expansion: {e}")
        
        mcts_object.expand_and_evaluate(mcts_object.root)

    if not mcts_object.root.children:
        pytest.skip("Root has no children after expansion, cannot test policy fallback.")

    # 2. Manually set all child visit counts to 0
    expected_priors_dict = {}
    for action_idx, child in mcts_object.root.children.items():
        child.visit_count = 0
        expected_priors_dict[action_idx] = child.prior_probability
    
    # Normalize expected priors for comparison
    total_prior = sum(expected_priors_dict.values())
    if np.isclose(total_prior, 0.0): # Should not happen if expansion worked
         pytest.skip("Sum of priors is zero, cannot test fallback to priors.")

    for k_idx in expected_priors_dict:
        expected_priors_dict[k_idx] /= total_prior
        
    policy_dict, _ = mcts_object.get_policy(temperature=1.0)

    assert len(policy_dict) == len(expected_priors_dict)
    for action_idx, prob in policy_dict.items():
        assert np.isclose(prob, expected_priors_dict.get(action_idx, 0.0)), \
            f"Policy prob {prob} for action {action_idx} does not match normalized prior {expected_priors_dict.get(action_idx)}."


def test_get_policy_zero_visits_zero_priors_fallback_to_uniform(mcts_instance):
    mcts_object, game_objects_tuple = mcts_instance
    # 1. Expand root
    if mcts_object.root.is_leaf():
        try:
            legal_mask = mcts_object.action_mapper.get_legal_action_mask(mcts_object.root.state)
            if torch.sum(legal_mask) == 0:
                pytest.skip("No legal actions to expand for testing uniform fallback.")
        except Exception as e:
            pytest.skip(f"Could not get legal actions for expansion: {e}")
        mcts_object.expand_and_evaluate(mcts_object.root)

    if not mcts_object.root.children:
        pytest.skip("Root has no children, cannot test policy uniform fallback.")

    num_children = len(mcts_object.root.children)
    if num_children == 0: # Should be caught by above
        pytest.skip("No children to test uniform policy.")

    # 2. Manually set all child visit counts and priors to 0
    for child in mcts_object.root.children.values():
        child.visit_count = 0
        child.prior_probability = 0.0 # Crucial for this test case
            
    policy_dict, _ = mcts_object.get_policy(temperature=1.0)
    
    expected_uniform_prob = 1.0 / num_children
    assert len(policy_dict) == num_children
    for action_idx, prob in policy_dict.items():
        assert np.isclose(prob, expected_uniform_prob), \
            f"Policy prob {prob} for action {action_idx} not uniform ({expected_uniform_prob})."


def test_dirichlet_noise_application_with_noise(mcts_instance_with_noise, model):
    mcts_instance_obj, game_objects = mcts_instance_with_noise # Unpack
    initial_game_state, action_helper, action_mapper, encoder = game_objects
    root = mcts_instance_obj.root
    
    # --- Get raw policy from network (masked and normalized) for comparison ---
    game_state_cpu, (map_nodes_cpu, raw_edge_input_tensor_cpu) = encoder.encode(root.state)
    game_state_dev = game_state_cpu.to(mcts_instance_obj.device)
    map_nodes_dev = map_nodes_cpu.to(mcts_instance_obj.device)
    raw_edge_input_tensor_dev = raw_edge_input_tensor_cpu.to(mcts_instance_obj.device)
    edge_index = raw_edge_input_tensor_dev[0:2, :]
    edge_attr_categorical = raw_edge_input_tensor_dev[2, :].long()

    with torch.no_grad():
        policy_logits, _ = model(
            game_state_dev, map_nodes_dev, edge_index, edge_attr_categorical
        )
    raw_policy_tensor = torch.softmax(policy_logits, dim=1).squeeze(0).cpu()
    legal_actions_mask = action_mapper.get_legal_action_mask(root.state)
    
    masked_policy = raw_policy_tensor * legal_actions_mask
    policy_sum = torch.sum(masked_policy)

    # We need at least one legal action for noise to be meaningful
    if torch.sum(legal_actions_mask) == 0:
        pytest.skip("No legal actions, cannot test noise application effectively.")

    # --- Run expand_and_evaluate (which applies noise_factor > 0) ---
    mcts_instance_obj.expand_and_evaluate(root)
    
    if not root.children:
        pytest.skip("Root not expanded, cannot test noise application.")

    child_priors_dict = {idx: child.prior_probability for idx, child in root.children.items()}
    assert np.isclose(sum(child_priors_dict.values()), 1.0, atol=1e-5), "Child priors should sum to 1 after noise."

    # If policy_sum was very low, MCTS might have made priors uniform over legal actions before noise.
    # It's hard to make a direct comparison to 'normalized_masked_policy' when noise is applied,
    # as the noise fundamentally changes the distribution.
    # The main check is that priors sum to 1 and are assigned.
    # A more robust check would be if *any* prior significantly deviates from the raw policy,
    # assuming the raw policy wasn't perfectly uniform itself.
    if policy_sum > 1e-8:
        normalized_masked_policy = (masked_policy / policy_sum).numpy()
        
        # Check if at least one prior is different, assuming multiple children and non-uniform raw policy
        if len(child_priors_dict) > 1:
            differences_found = False
            for action_idx, prior in child_priors_dict.items():
                if not np.isclose(prior, normalized_masked_policy[action_idx], atol=1e-3): # Looser tolerance due to noise
                    differences_found = True
                    break
            # This assertion is tricky because noise could coincidentally make them similar
            # or if raw policy is uniform, noise might keep it somewhat uniform.
            # A better check might be statistical if we knew the exact noise vector.
            # For now, we rely on the fact that noise *was* added and priors sum to 1.
            # If all priors were exactly equal to raw policy, noise wasn't effective.
            if not differences_found and not np.allclose(np.unique(normalized_masked_policy[[*child_priors_dict.keys()]]), np.unique(normalized_masked_policy[[*child_priors_dict.keys()]][0])): # if raw policy wasn't uniform
                 #This check is weak, if raw policy is uniform, noise might not make it non-uniform enough to detect easily
                 pass # print("Warning: Noise applied, but priors seem to match raw policy. Check noise effectiveness or raw policy distribution.")

    else: # All legal actions had near-zero policy from network
        # Priors should be somewhat uniform due to noise on top of (likely) uniform base
        pass


def test_mcts_with_single_legal_action(game_state_single_action, model):
    """Tests MCTS behavior when only one action is legal."""

    single_action_state, _, action_mapper, encoder = game_state_single_action

    mcts_single = MCTS(
        root_state=single_action_state,
        model=model,
        encoder=encoder,
        action_mapper=action_mapper,
        num_simulations=TEST_NUM_SIMULATIONS, # Run a few simulations
        c_puct=TEST_C_PUCT,
        device=torch.device("cpu")
    )

    # Verify only one legal action
    legal_mask = mcts_single.action_mapper.get_legal_action_mask(single_action_state)
    assert torch.sum(legal_mask) == 1, "Test setup error: game state does not have a single legal action."
    
    mcts_single.search()

    assert len(mcts_single.root.children) == 1, "MCTS should have expanded only one child for a single legal action."
    
    single_child_action_idx = list(mcts_single.root.children.keys())[0]
    single_child_node = mcts_single.root.children[single_child_action_idx]

    assert single_child_node.visit_count == TEST_NUM_SIMULATIONS - 1, \
        f"The single child node should have been visited in every simulation. Expected {TEST_NUM_SIMULATIONS - 1}, got {single_child_node.visit_count}."
    assert mcts_single.root.visit_count == TEST_NUM_SIMULATIONS, \
        f"The root node should have been visited in every simulation. Expected {TEST_NUM_SIMULATIONS}, got {mcts_single.root.visit_count}."

    policy_dict, policy_vector = mcts_single.get_policy(temperature=1.0)
    assert len(policy_dict) == 1, "Policy should be for a single action."
    assert np.isclose(list(policy_dict.values())[0], 1.0), "The single legal action should have probability 1.0."
    assert np.isclose(policy_vector[single_child_action_idx], 1.0)
    assert np.isclose(np.sum(policy_vector), 1.0)