
import os
import tempfile
from rl18xx.game.gamemap import GameMap
from rl18xx.agent.alphazero.v2.config import MegaConfig
from rl18xx.agent.alphazero.v2.model import AlphaZeroModel

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
    model_config =  {
        "game_state_size": GAME_STATE_SIZE,
        "map_node_features": MAP_NODE_FEATURES,
        "policy_size": POLICY_OUTPUT_SIZE,
        "value_size": VALUE_OUTPUT_SIZE,
        "mlp_hidden_dim": MLP_HIDDEN_DIM,
        "gnn_node_proj_dim": GNN_NODE_PROJ_DIM,
        "gnn_hidden_dim_per_head": GNN_HIDDEN_DIM_PER_HEAD,
        "gnn_layers": GNN_LAYERS,
        "gnn_heads": GNN_HEADS,
        "gnn_output_embed_dim": GNN_OUTPUT_EMBED_DIM,
        "gnn_edge_categories": GNN_EDGE_CATEGORIES,
        "gnn_edge_embedding_dim": GNN_EDGE_EMBEDDING_DIM,
        "shared_trunk_hidden_dim": SHARED_TRUNK_HIDDEN_DIM,
        "num_res_blocks": NUM_RES_BLOCKS,
        "dropout_rate": 0.0
    }
    config = MegaConfig(**model_config)
    return AlphaZeroModel(config)

def test_run_single():
    model = get_model()
    model.eval()
    game_state = get_fresh_game_state()
    probs, log_probs, value = model.run(game_state)
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
