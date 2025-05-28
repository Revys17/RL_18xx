
import os
import tempfile
from rl18xx.game.gamemap import GameMap
from rl18xx.agent.alphazero.config import ModelConfig
from rl18xx.agent.alphazero.model import AlphaZeroModel

# --- Define constants based on your model's expected inputs/outputs ---
GAME_STATE_SIZE = 377
NUM_MAP_NODES = 93
MAP_NODE_FEATURES = 50
NUM_EDGES = 470
POLICY_OUTPUT_SIZE = 26535
VALUE_OUTPUT_SIZE = 4
GNN_EDGE_CATEGORIES = 6
GNN_EDGE_EMBEDDING_DIM = 32

# Default MLP and GNN parameters (can be overridden in specific tests if needed)
MLP_HIDDEN_DIM = 128
GNN_NODE_PROJ_DIM = 64
GNN_HIDDEN_DIM_PER_HEAD = 32
GNN_LAYERS = 2
GNN_HEADS = 2
GNN_OUTPUT_EMBED_DIM = 128
SHARED_TRUNK_HIDDEN_DIM = 256
NUM_RES_BLOCKS = 2

def get_fresh_game_state():
    game_map = GameMap()
    game_class = game_map.game_by_title("1830")
    players = {"1": "Player 1", "2": "Player 2", "3": "Player 3", "4": "Player 4"}
    game_instance = game_class(players)
    return game_instance

def get_model():
    return AlphaZeroModel(ModelConfig())

def test_run_single():
    model = get_model()
    model.eval()
    game_state = get_fresh_game_state()
    probs, log_probs, value = model.run(game_state)
    assert probs.shape == (POLICY_OUTPUT_SIZE,)
    assert log_probs.shape == (POLICY_OUTPUT_SIZE,)
    assert value.shape == (VALUE_OUTPUT_SIZE,)

def test_run_single_encoded():
    model = get_model()
    model.eval()
    game_state = get_fresh_game_state()
    encoded_game_state = model.encoder.encode(game_state)
    probs, log_probs, value = model.run_encoded(encoded_game_state)
    assert probs.shape == (POLICY_OUTPUT_SIZE,)
    assert log_probs.shape == (POLICY_OUTPUT_SIZE,)
    assert value.shape == (VALUE_OUTPUT_SIZE,)

def test_run_batch():
    model = get_model()
    model.eval()
    game_states = [get_fresh_game_state() for _ in range(4)]
    probs, log_probs, values = model.run_many(game_states)
    assert probs.shape == (4, POLICY_OUTPUT_SIZE)
    assert log_probs.shape == (4, POLICY_OUTPUT_SIZE)
    assert values.shape == (4, VALUE_OUTPUT_SIZE)

def test_run_batch_encoded():
    model = get_model()
    model.eval()
    game_states = [get_fresh_game_state() for _ in range(4)]
    encoded_game_states = [model.encoder.encode(game_state) for game_state in game_states]
    probs, log_probs, values = model.run_many_encoded(encoded_game_states)
    assert probs.shape == (4, POLICY_OUTPUT_SIZE)
    assert log_probs.shape == (4, POLICY_OUTPUT_SIZE)
    assert values.shape == (4, VALUE_OUTPUT_SIZE)