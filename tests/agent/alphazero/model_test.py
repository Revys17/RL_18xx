import pytest
from rl18xx.agent.alphazero.model import Model
import torch
import torch.nn as nn

# --- Define constants based on your model's expected inputs/outputs ---
GAME_STATE_SIZE = 377
NUM_MAP_NODES = 93
MAP_NODE_FEATURES = 50
NUM_EDGES = 470
POLICY_OUTPUT_SIZE = 26535
VALUE_OUTPUT_SIZE = 4
GNN_EDGE_CATEGORIES = 6 # Example: 6 types of hex edges
GNN_EDGE_EMBEDDING_DIM = 32

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


# Helper function to create PyG-style batched inputs (similar to a previous version)
def _create_batched_graph_inputs(batch_size, num_nodes_per_graph, num_edges_per_graph, node_features, edge_categories, device=torch.device("cpu")):
    """Helper to create PyG-style batched graph inputs."""
    x_map_nodes_list = []
    edge_index_list = []
    edge_attr_list = []
    node_batch_idx_list = []
    current_node_offset = 0

    # If edge_index is fixed per graph, define it once
    # For this test, we generate it per graph to simulate distinct graphs in a batch
    fixed_src_nodes_template = torch.randint(0, num_nodes_per_graph, (num_edges_per_graph,))
    fixed_dst_nodes_template = torch.randint(0, num_nodes_per_graph, (num_edges_per_graph,))
    edge_index_template = torch.stack([fixed_src_nodes_template, fixed_dst_nodes_template], dim=0)


    for i in range(batch_size):
        x_map_nodes_list.append(torch.randn(num_nodes_per_graph, node_features))
        
        # Use the template and offset it
        edge_index_list.append(edge_index_template.clone() + current_node_offset)
        
        if edge_categories > 0:
            edge_attr_list.append(torch.randint(0, edge_categories, (num_edges_per_graph,)).long())
        
        node_batch_idx_list.append(torch.full((num_nodes_per_graph,), fill_value=i, dtype=torch.long))
        current_node_offset += num_nodes_per_graph

    x_map_nodes_batched = torch.cat(x_map_nodes_list, dim=0).to(device)
    edge_index_batched = torch.cat(edge_index_list, dim=1).to(device)
    node_batch_idx = torch.cat(node_batch_idx_list, dim=0).to(device)
    
    edge_attr_categorical_batched = None
    if edge_categories > 0 and edge_attr_list:
        edge_attr_categorical_batched = torch.cat(edge_attr_list, dim=0).to(device)
    
    return x_map_nodes_batched, edge_index_batched, node_batch_idx, edge_attr_categorical_batched


@pytest.fixture(params=[1, 4])
def dummy_inputs(request):
    batch_size = request.param
    x_game_state = torch.randn(batch_size, GAME_STATE_SIZE)

    x_map_nodes_batched, edge_index_batched, \
    node_batch_idx, edge_attr_categorical_batched = _create_batched_graph_inputs(
        batch_size, NUM_MAP_NODES, NUM_EDGES, MAP_NODE_FEATURES, GNN_EDGE_CATEGORIES
    )
    
    return {
        "x_game_state": x_game_state,
        "x_map_nodes_batched": x_map_nodes_batched,
        "edge_index_batched": edge_index_batched,
        "node_batch_idx": node_batch_idx,
        "edge_attr_categorical_batched": edge_attr_categorical_batched,
        "batch_size": batch_size # Keep for assertions on output batch size
    }

@pytest.fixture(params=[2, 4]) 
def dummy_inputs_for_train_tests(request): # For tests needing batch_size > 1
    batch_size = request.param
    x_game_state = torch.randn(batch_size, GAME_STATE_SIZE)
    x_map_nodes_batched, edge_index_batched, \
    node_batch_idx, edge_attr_categorical_batched = _create_batched_graph_inputs(
        batch_size, NUM_MAP_NODES, NUM_EDGES, MAP_NODE_FEATURES, GNN_EDGE_CATEGORIES
    )
    return {
        "x_game_state": x_game_state,
        "x_map_nodes_batched": x_map_nodes_batched,
        "edge_index_batched": edge_index_batched,
        "node_batch_idx": node_batch_idx,
        "edge_attr_categorical_batched": edge_attr_categorical_batched,
        "batch_size": batch_size
    }


def test_model_instantiation(model_with_edge_features):
    """Tests if the model can be instantiated."""
    assert model_with_edge_features is not None
    assert isinstance(model_with_edge_features, nn.Module)

def test_model_forward_pass_with_edge_features(model_with_edge_features, dummy_inputs):
    """Tests a basic forward pass with edge features enabled."""
    model = model_with_edge_features
    model.eval() # Set to evaluation mode
    with torch.no_grad(): # Disable gradient calculations
        policy_logits, value = model(
            x_game_state=dummy_inputs["x_game_state"],
            x_map_nodes_batched=dummy_inputs["x_map_nodes_batched"],
            edge_index_batched=dummy_inputs["edge_index_batched"],
            node_batch_idx=dummy_inputs["node_batch_idx"],
            edge_attr_categorical_batched=dummy_inputs["edge_attr_categorical_batched"]
        )

    assert policy_logits.shape == (dummy_inputs["batch_size"], POLICY_OUTPUT_SIZE), \
        f"Policy shape mismatch for batch size {dummy_inputs['batch_size']}"
    assert value.shape == (dummy_inputs["batch_size"], VALUE_OUTPUT_SIZE), \
        f"Value shape mismatch for batch size {dummy_inputs['batch_size']}"

def test_model_forward_pass_without_edge_features(model_without_edge_features, dummy_inputs):
    """Tests a basic forward pass with edge features disabled."""
    model = model_without_edge_features
    model.eval()
    # Create inputs without edge attributes for this specific test
    x_map_nodes_batched_no_ea, edge_index_batched_no_ea, \
    node_batch_idx_no_ea, _ = _create_batched_graph_inputs(
        dummy_inputs["batch_size"], NUM_MAP_NODES, NUM_EDGES, MAP_NODE_FEATURES, 0 # 0 edge categories
    )

    with torch.no_grad():
        policy_logits, value = model(
            x_game_state=dummy_inputs["x_game_state"],
            x_map_nodes_batched=x_map_nodes_batched_no_ea,
            edge_index_batched=edge_index_batched_no_ea,
            node_batch_idx=node_batch_idx_no_ea,
            edge_attr_categorical_batched=None # Explicitly pass None
        )
    assert policy_logits.shape == (dummy_inputs["batch_size"], POLICY_OUTPUT_SIZE)
    assert value.shape == (dummy_inputs["batch_size"], VALUE_OUTPUT_SIZE)


def test_model_output_types(model_with_edge_features, dummy_inputs):
    """Tests the data types of the outputs."""
    model = model_with_edge_features
    model.eval()
    with torch.no_grad():
        policy_logits, value = model(
            x_game_state=dummy_inputs["x_game_state"],
            x_map_nodes_batched=dummy_inputs["x_map_nodes_batched"],
            edge_index_batched=dummy_inputs["edge_index_batched"],
            node_batch_idx=dummy_inputs["node_batch_idx"],
            edge_attr_categorical_batched=dummy_inputs["edge_attr_categorical_batched"]
        )

    assert isinstance(policy_logits, torch.Tensor)
    assert isinstance(value, torch.Tensor)
    assert policy_logits.dtype == torch.float32
    assert value.dtype == torch.float32


def test_model_policy_output_probs(model_with_edge_features, dummy_inputs):
    """Tests if policy outputs are valid log probabilities (exp sum ~ 1)."""
    model = model_with_edge_features
    model.eval()
    with torch.no_grad():
        policy_logits, _ = model(
            x_game_state=dummy_inputs["x_game_state"],
            x_map_nodes_batched=dummy_inputs["x_map_nodes_batched"],
            edge_index_batched=dummy_inputs["edge_index_batched"],
            node_batch_idx=dummy_inputs["node_batch_idx"],
            edge_attr_categorical_batched=dummy_inputs["edge_attr_categorical_batched"]
        )

    policy_probs = torch.exp(policy_logits)
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
            x_map_nodes_batched=dummy_inputs["x_map_nodes_batched"],
            edge_index_batched=dummy_inputs["edge_index_batched"],
            node_batch_idx=dummy_inputs["node_batch_idx"],
            edge_attr_categorical_batched=dummy_inputs["edge_attr_categorical_batched"]
        )

    assert torch.all(value >= -1.0 - 1e-6), f"Value output below -1: {value}" # Add tolerance for float precision
    assert torch.all(value <= 1.0 + 1e-6), f"Value output above 1: {value}"  # Add tolerance for float precision


def test_model_forward_pass_value_error_missing_edge_attr(model_with_edge_features, dummy_inputs):
    """
    Tests that a ValueError is raised if edge_attr_categorical_batched is missing
    when the model is configured to use edge features.
    """
    model = model_with_edge_features # This model expects edge features
    model.eval()
    with torch.no_grad():
        with pytest.raises(ValueError, match="Model configured to use edge features .* but edge_attr_categorical_batched was not provided"):
            model(
                x_game_state=dummy_inputs["x_game_state"],
                x_map_nodes_batched=dummy_inputs["x_map_nodes_batched"],
                edge_index_batched=dummy_inputs["edge_index_batched"],
                node_batch_idx=dummy_inputs["node_batch_idx"],
                edge_attr_categorical_batched=None # Intentionally missing
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_model_device_movement(model_with_edge_features, dummy_inputs):
    """Tests if the model and data can be moved to CUDA and run."""
    device = torch.device("cuda")
    model = model_with_edge_features.to(device)

    # Move all parts of dummy_inputs to device
    inputs_cuda = {
        "x_game_state": dummy_inputs["x_game_state"].to(device),
        "x_map_nodes_batched": dummy_inputs["x_map_nodes_batched"].to(device),
        "edge_index_batched": dummy_inputs["edge_index_batched"].to(device),
        "node_batch_idx": dummy_inputs["node_batch_idx"].to(device),
        "edge_attr_categorical_batched": dummy_inputs["edge_attr_categorical_batched"].to(device) if dummy_inputs["edge_attr_categorical_batched"] is not None else None,
        "batch_size": dummy_inputs["batch_size"]
    }

    model.eval()
    with torch.no_grad():
        try:
            policy_logits, value = model(
                x_game_state=inputs_cuda["x_game_state"],
                x_map_nodes_batched=inputs_cuda["x_map_nodes_batched"],
                edge_index_batched=inputs_cuda["edge_index_batched"],
                node_batch_idx=inputs_cuda["node_batch_idx"],
                edge_attr_categorical_batched=inputs_cuda["edge_attr_categorical_batched"]
            )
            assert policy_logits.device.type == "cuda"
            assert value.device.type == "cuda"
            assert policy_logits.shape[0] == dummy_inputs["batch_size"]
            assert value.shape[0] == dummy_inputs["batch_size"]
        except Exception as e:
            pytest.fail(f"Model forward pass failed on CUDA: {e}")


def test_model_runs_in_train_mode(model_with_edge_features, dummy_inputs_for_train_tests):
    """Tests that the model can execute a forward pass in training mode."""
    model = model_with_edge_features
    inputs = dummy_inputs_for_train_tests 
    model.train() 
    try:
        _, _ = model(
            x_game_state=inputs["x_game_state"],
            x_map_nodes_batched=inputs["x_map_nodes_batched"],
            edge_index_batched=inputs["edge_index_batched"],
            node_batch_idx=inputs["node_batch_idx"],
            edge_attr_categorical_batched=inputs["edge_attr_categorical_batched"]
        )
    except Exception as e:
        pytest.fail(f"Model forward pass failed in train() mode: {e}")
    finally:
        model.eval()

def test_model_basic_gradient_flow(model_with_edge_features, dummy_inputs_for_train_tests):
    """Tests basic gradient flow through the model."""
    model = model_with_edge_features
    inputs = dummy_inputs_for_train_tests
    model.train()

    params_to_check = [
        model.fc_game_state1.weight,
        model.policy_head.weight
    ]
    if model.gnn_layers_modulelist: 
        if hasattr(model.gnn_layers_modulelist[0], 'lin_l') and \
           hasattr(model.gnn_layers_modulelist[0].lin_l, 'weight'):
            params_to_check.append(model.gnn_layers_modulelist[0].lin_l.weight)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad()

    policy_logits, value_estimates = model(
        x_game_state=inputs["x_game_state"],
        x_map_nodes_batched=inputs["x_map_nodes_batched"],
        edge_index_batched=inputs["edge_index_batched"],
        node_batch_idx=inputs["node_batch_idx"],
        edge_attr_categorical_batched=inputs["edge_attr_categorical_batched"]
    )

    dummy_loss = policy_logits.sum() + value_estimates.sum()
    dummy_loss.backward()

    for i, param in enumerate(params_to_check):
        assert param.grad is not None, f"Gradient is None for parameter {i}"
        assert torch.sum(torch.abs(param.grad)) > 0, f"Gradient is all zeros for parameter {i}"

    optimizer.zero_grad() # Clean up gradients
    model.eval() # Reset to eval mode
