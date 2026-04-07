import os
import tempfile
import torch
from rl18xx.game.gamemap import GameMap
from rl18xx.agent.alphazero.config import ModelConfig
from rl18xx.agent.alphazero.model import AlphaZeroGNNModel, FactoredPolicyHead

# --- Define constants based on your model's expected inputs/outputs ---
GAME_STATE_SIZE = 390
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
    players = {1: "Player 1", 2: "Player 2", 3: "Player 3", 4: "Player 4"}
    game_instance = game_class(players)
    return game_instance


def get_model():
    return AlphaZeroGNNModel(ModelConfig())


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


def test_factored_policy_head_output_shape():
    """The factored head must produce the same output size as the original flat head."""
    from rl18xx.agent.alphazero.action_mapper import ActionMapper

    action_mapper = ActionMapper()
    lay_tile_info = action_mapper.get_lay_tile_index_info()

    trunk_dim = SHARED_TRUNK_HIDDEN_DIM
    head = FactoredPolicyHead(trunk_dim, GNN_OUTPUT_EMBED_DIM, POLICY_OUTPUT_SIZE, lay_tile_info)
    head.eval()

    x = torch.randn(3, trunk_dim)
    logits = head(x)
    assert logits.shape == (3, POLICY_OUTPUT_SIZE)


def test_factored_policy_head_finite_logits():
    """All logits produced by the factored head should be finite (no NaN or Inf)."""
    from rl18xx.agent.alphazero.action_mapper import ActionMapper

    action_mapper = ActionMapper()
    lay_tile_info = action_mapper.get_lay_tile_index_info()

    trunk_dim = SHARED_TRUNK_HIDDEN_DIM
    head = FactoredPolicyHead(trunk_dim, GNN_OUTPUT_EMBED_DIM, POLICY_OUTPUT_SIZE, lay_tile_info)
    head.eval()

    x = torch.randn(2, trunk_dim)
    logits = head(x)
    assert torch.isfinite(logits).all(), "Factored policy head produced non-finite logits"
