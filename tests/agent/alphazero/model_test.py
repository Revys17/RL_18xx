import pytest
from rl18xx.agent.alphazero.model import Model
import torch
import torch.nn as nn

# --- Define constants based on your model's expected inputs/outputs ---
GAME_STATE_SIZE = 377
NUM_MAP_NODES = 93
MAP_NODE_FEATURES = 50
NUM_EDGES = 470 # Number of edges in the graph
POLICY_OUTPUT_SIZE = 26535
VALUE_OUTPUT_SIZE = 4
GNN_EDGE_CATEGORIES = 6 # Example: 6 types of hex edges
GNN_EDGE_EMBEDDING_DIM = 32 # Must match a potential model config

# Default MLP and GNN parameters (can be overridden in specific tests if needed)
MLP_HIDDEN_DIM = 128 # Smaller for faster tests, adjust if needed
GNN_NODE_PROJ_DIM = 64
GNN_HIDDEN_DIM_PER_HEAD = 32
GNN_LAYERS = 2
GNN_HEADS = 2
GNN_OUTPUT_EMBED_DIM = 128
SHARED_TRUNK_HIDDEN_DIM = 256
NUM_RES_BLOCKS = 2


@pytest.fixture
def model_default_config():
    """Provides default configuration parameters for the model."""
    return {
        "game_state_size": GAME_STATE_SIZE,
        "num_map_nodes": NUM_MAP_NODES,
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
        "dropout_rate": 0.0 # Disable dropout for deterministic tests
    }

@pytest.fixture
def model_with_edge_features(model_default_config):
    """Model instance configured to use edge features."""
    return Model(**model_default_config)

@pytest.fixture
def model_without_edge_features(model_default_config):
    """Model instance configured NOT to use edge features."""
    config = model_default_config.copy()
    config["gnn_edge_categories"] = 0
    config["gnn_edge_embedding_dim"] = 0
    return Model(**config)


@pytest.fixture(params=[1, 4])  # Test with different batch sizes
def dummy_inputs(request):
    """Provides a dictionary of dummy input tensors with varying batch sizes."""
    batch_size = request.param
    
    x_game_state = torch.randn(batch_size, GAME_STATE_SIZE)
    x_map_nodes = torch.randn(batch_size, NUM_MAP_NODES, MAP_NODE_FEATURES)
    
    # Create a valid edge_index: shape (2, NUM_EDGES)
    # Ensure node indices are within [0, NUM_MAP_NODES - 1]
    src_nodes = torch.randint(0, NUM_MAP_NODES, (NUM_EDGES,))
    dst_nodes = torch.randint(0, NUM_MAP_NODES, (NUM_EDGES,))
    # Ensure no self-loops if GATv2Conv handles them implicitly or if not desired for test
    # For simplicity, we'll allow them here as GAT can handle them.
    edge_index = torch.stack([src_nodes, dst_nodes], dim=0)
    
    # Create edge_attr_categorical: shape (NUM_EDGES,), type long
    edge_attr_categorical = torch.randint(0, GNN_EDGE_CATEGORIES, (NUM_EDGES,)).long()
    
    return {
        "x_game_state": x_game_state,
        "x_map_nodes": x_map_nodes,
        "edge_index": edge_index,
        "edge_attr_categorical": edge_attr_categorical,
        "batch_size": batch_size
    }

@pytest.fixture(params=[2, 4]) # Use batch sizes > 1 for training mode tests
def dummy_inputs_for_train_tests(request):
    """
    Provides a dictionary of dummy input tensors with batch sizes > 1,
    suitable for tests involving model.train().
    """
    batch_size = request.param
    
    x_game_state = torch.randn(batch_size, GAME_STATE_SIZE)
    x_map_nodes = torch.randn(batch_size, NUM_MAP_NODES, MAP_NODE_FEATURES)
    
    src_nodes = torch.randint(0, NUM_MAP_NODES, (NUM_EDGES,))
    dst_nodes = torch.randint(0, NUM_MAP_NODES, (NUM_EDGES,))
    edge_index = torch.stack([src_nodes, dst_nodes], dim=0)
    
    edge_attr_categorical = torch.randint(0, GNN_EDGE_CATEGORIES, (NUM_EDGES,)).long()
    
    return {
        "x_game_state": x_game_state,
        "x_map_nodes": x_map_nodes,
        "edge_index": edge_index,
        "edge_attr_categorical": edge_attr_categorical,
        "batch_size": batch_size # Though batch_size itself isn't directly used by the test logic here
    }


def test_model_instantiation(model_with_edge_features):
    """Tests if the model can be instantiated."""
    assert model_with_edge_features is not None
    assert isinstance(model_with_edge_features, nn.Module)

def test_model_forward_pass_shapes_with_edge_features(model_with_edge_features, dummy_inputs):
    """Tests the forward pass and output shapes when using edge features."""
    model = model_with_edge_features
    batch_size = dummy_inputs["batch_size"]
    
    model.eval()
    with torch.no_grad():
        policy_log_probs, value = model(
            x_game_state=dummy_inputs["x_game_state"],
            x_map_nodes=dummy_inputs["x_map_nodes"],
            edge_index=dummy_inputs["edge_index"],
            edge_attr_categorical=dummy_inputs["edge_attr_categorical"]
        )

    assert policy_log_probs.shape == (batch_size, POLICY_OUTPUT_SIZE), \
        f"Policy shape mismatch for batch size {batch_size}"
    assert value.shape == (batch_size, VALUE_OUTPUT_SIZE), \
        f"Value shape mismatch for batch size {batch_size}"

def test_model_forward_pass_shapes_without_edge_features(model_without_edge_features, dummy_inputs):
    """Tests the forward pass and output shapes when NOT using edge features."""
    model = model_without_edge_features
    batch_size = dummy_inputs["batch_size"]

    model.eval()
    with torch.no_grad():
        policy_log_probs, value = model(
            x_game_state=dummy_inputs["x_game_state"],
            x_map_nodes=dummy_inputs["x_map_nodes"],
            edge_index=dummy_inputs["edge_index"],
            edge_attr_categorical=None # Explicitly pass None
        )

    assert policy_log_probs.shape == (batch_size, POLICY_OUTPUT_SIZE), \
        f"Policy shape mismatch (no edge features) for batch size {batch_size}"
    assert value.shape == (batch_size, VALUE_OUTPUT_SIZE), \
        f"Value shape mismatch (no edge features) for batch size {batch_size}"


def test_model_output_types(model_with_edge_features, dummy_inputs):
    """Tests the data types of the outputs."""
    model = model_with_edge_features
    model.eval()
    with torch.no_grad():
        policy_log_probs, value = model(
            x_game_state=dummy_inputs["x_game_state"],
            x_map_nodes=dummy_inputs["x_map_nodes"],
            edge_index=dummy_inputs["edge_index"],
            edge_attr_categorical=dummy_inputs["edge_attr_categorical"]
        )

    assert isinstance(policy_log_probs, torch.Tensor)
    assert isinstance(value, torch.Tensor)
    assert policy_log_probs.dtype == torch.float32
    assert value.dtype == torch.float32


def test_model_policy_output_probs(model_with_edge_features, dummy_inputs):
    """Tests if policy outputs are valid log probabilities (exp sum ~ 1)."""
    model = model_with_edge_features
    model.eval()
    with torch.no_grad():
        policy_log_probs, _ = model(
            x_game_state=dummy_inputs["x_game_state"],
            x_map_nodes=dummy_inputs["x_map_nodes"],
            edge_index=dummy_inputs["edge_index"],
            edge_attr_categorical=dummy_inputs["edge_attr_categorical"]
        )

    policy_probs = torch.exp(policy_log_probs)
    sum_probs = torch.sum(policy_probs, dim=1)
    assert torch.allclose(sum_probs, torch.ones_like(sum_probs), atol=1e-5), \
        f"Policy probabilities do not sum to 1. Sums: {sum_probs}"


def test_model_value_output_range(model_with_edge_features, dummy_inputs):
    """Tests if the value output is within the tanh range [-1, 1]."""
    model = model_with_edge_features
    model.eval()
    with torch.no_grad():
        _, value = model(
            x_game_state=dummy_inputs["x_game_state"],
            x_map_nodes=dummy_inputs["x_map_nodes"],
            edge_index=dummy_inputs["edge_index"],
            edge_attr_categorical=dummy_inputs["edge_attr_categorical"]
        )

    assert torch.all(value >= -1.0 - 1e-6), f"Value output below -1: {value}" # Add tolerance for float precision
    assert torch.all(value <= 1.0 + 1e-6), f"Value output above 1: {value}"  # Add tolerance for float precision


def test_model_forward_pass_value_error_missing_edge_attr(model_with_edge_features, dummy_inputs):
    """
    Tests that a ValueError is raised if edge_attr_categorical is missing
    when the model is configured to use edge features.
    """
    model = model_with_edge_features # This model expects edge features
    model.eval()
    with torch.no_grad():
        with pytest.raises(ValueError, match="Model configured to use edge features .* but edge_attr_categorical was not provided"):
            model(
                x_game_state=dummy_inputs["x_game_state"],
                x_map_nodes=dummy_inputs["x_map_nodes"],
                edge_index=dummy_inputs["edge_index"],
                edge_attr_categorical=None # Intentionally missing
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_model_device_movement(model_with_edge_features, dummy_inputs):
    """Tests if the model and data can be moved to CUDA and run."""
    device = torch.device("cuda")
    model = model_with_edge_features.to(device)

    # Move all parts of dummy_inputs to CUDA
    inputs_cuda = {
        "x_game_state": dummy_inputs["x_game_state"].to(device),
        "x_map_nodes": dummy_inputs["x_map_nodes"].to(device),
        "edge_index": dummy_inputs["edge_index"].to(device),
        "edge_attr_categorical": dummy_inputs["edge_attr_categorical"].to(device)
    }

    model.eval()
    with torch.no_grad():
        try:
            policy_log_probs, value = model(
                x_game_state=inputs_cuda["x_game_state"],
                x_map_nodes=inputs_cuda["x_map_nodes"],
                edge_index=inputs_cuda["edge_index"],
                edge_attr_categorical=inputs_cuda["edge_attr_categorical"]
            )
            assert policy_log_probs.device.type == "cuda"
            assert value.device.type == "cuda"
            assert policy_log_probs.shape[0] == dummy_inputs["batch_size"]
            assert value.shape[0] == dummy_inputs["batch_size"]
        except Exception as e:
            pytest.fail(f"Model forward pass failed on CUDA: {e}")


def test_model_runs_in_train_mode(model_with_edge_features, dummy_inputs_for_train_tests):
    """Tests that the model can execute a forward pass in training mode."""
    model = model_with_edge_features
    # Use the new fixture that guarantees batch_size > 1
    inputs = dummy_inputs_for_train_tests 
    model.train() # Set to training mode
    try:
        _, _ = model(
            x_game_state=inputs["x_game_state"],
            x_map_nodes=inputs["x_map_nodes"],
            edge_index=inputs["edge_index"],
            edge_attr_categorical=inputs["edge_attr_categorical"]
        )
    except Exception as e:
        pytest.fail(f"Model forward pass failed in train() mode: {e}")
    finally:
        model.eval() # Reset to eval mode

def test_model_basic_gradient_flow(model_with_edge_features, dummy_inputs_for_train_tests):
    """Tests basic gradient flow through the model."""
    model = model_with_edge_features
    # Use the new fixture that guarantees batch_size > 1
    inputs = dummy_inputs_for_train_tests
    model.train() # Ensure gradients are computed and layers are in training mode

    # Get some parameters to check gradients for
    params_to_check = [
        model.fc_game_state1.weight,
        model.policy_head.weight
    ]
    if model.gnn_layers_modulelist: # If GNN layers exist
        # GATv2Conv has lin_l, lin_r, att_l, att_r, bias, etc.
        # We'll check the first GAT layer's left linear projection
        if hasattr(model.gnn_layers_modulelist[0], 'lin_l') and \
           hasattr(model.gnn_layers_modulelist[0].lin_l, 'weight'):
            params_to_check.append(model.gnn_layers_modulelist[0].lin_l.weight)


    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad()

    policy_log_probs, value_estimates = model(
        x_game_state=inputs["x_game_state"],
        x_map_nodes=inputs["x_map_nodes"],
        edge_index=inputs["edge_index"],
        edge_attr_categorical=inputs["edge_attr_categorical"]
    )

    # Create a dummy loss
    # Summing all outputs to create a single scalar for loss
    dummy_loss = policy_log_probs.sum() + value_estimates.sum()
    dummy_loss.backward()

    for i, param in enumerate(params_to_check):
        assert param.grad is not None, f"Gradient is None for parameter {i}"
        assert torch.sum(torch.abs(param.grad)) > 0, f"Gradient is all zeros for parameter {i}"

    optimizer.zero_grad() # Clean up gradients
    model.eval() # Reset to eval mode
