import pytest
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader as PyG_DataLoader
import os
import tempfile
import pickle
import logging

from rl18xx.agent.alphazero.train import (
    SelfPlayDataset,
    load_training_data,
)
from rl18xx.agent.alphazero.model import Model

# --- Constants for dummy data (matching user's observed shapes) ---
DUMMY_GAME_STATE_SIZE = 377
DUMMY_MAP_NODE_FEATURES = 50
DUMMY_NUM_MAP_NODES = 93
DUMMY_NUM_EDGES = 470
DUMMY_POLICY_SIZE = 26535
DUMMY_VALUE_SIZE = 4
DUMMY_GNN_EDGE_CATEGORIES = 6 # Default in Model, ensure consistency

# --- Fixtures ---

@pytest.fixture
def dummy_training_example_raw():
    """Generates one raw training example as it would be loaded from a file."""
    game_state = torch.randn(1, DUMMY_GAME_STATE_SIZE, dtype=torch.float32)
    map_nodes = torch.randn(DUMMY_NUM_MAP_NODES, DUMMY_MAP_NODE_FEATURES, dtype=torch.float32)
    
    edge_index_from = torch.randint(0, DUMMY_NUM_MAP_NODES, (DUMMY_NUM_EDGES,))
    edge_index_to = torch.randint(0, DUMMY_NUM_MAP_NODES, (DUMMY_NUM_EDGES,))
    # Ensure edge_attr_cat are valid category indices
    edge_attr_cat = torch.randint(0, DUMMY_GNN_EDGE_CATEGORIES, (DUMMY_NUM_EDGES,))
    # raw_edge_tensor stores these as floats, SelfPlayDataset will convert to long
    raw_edge_tensor = torch.stack([edge_index_from.float(), edge_index_to.float(), edge_attr_cat.float()], dim=0)

    policy_target_np = np.random.rand(DUMMY_POLICY_SIZE).astype(np.float32)
    # For CrossEntropyLoss with soft labels, targets should be probabilities (sum to 1)
    policy_target_np = policy_target_np / np.sum(policy_target_np) 
    
    value_target_np = np.random.rand(DUMMY_VALUE_SIZE).astype(np.float32)
    
    state_tuple = (game_state, map_nodes, raw_edge_tensor)
    return state_tuple, policy_target_np, value_target_np

@pytest.fixture
def dummy_training_examples_raw(dummy_training_example_raw):
    """A list of raw training examples."""
    # Using 5 examples: enough for a batch of 2 and some leftover
    return [dummy_training_example_raw for _ in range(5)] 

@pytest.fixture
def dummy_self_play_dataset(dummy_training_examples_raw):
    """SelfPlayDataset initialized with dummy raw examples."""
    return SelfPlayDataset(dummy_training_examples_raw)

@pytest.fixture
def device():
    """PyTorch device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture
def dummy_model(device): # Add device fixture
    """A dummy Model instance, using smaller dimensions for speed."""
    model = Model(
        game_state_size=DUMMY_GAME_STATE_SIZE,
        map_node_features=DUMMY_MAP_NODE_FEATURES,
        policy_size=DUMMY_POLICY_SIZE,
        value_size=DUMMY_VALUE_SIZE,
        gnn_edge_categories=DUMMY_GNN_EDGE_CATEGORIES,
        # Smaller architectural params for faster test model initialization/runs
        mlp_hidden_dim=32,
        gnn_node_proj_dim=32,
        gnn_hidden_dim_per_head=16,
        gnn_layers=1, # Minimal GNN layers
        gnn_heads=1,  # Minimal heads
        gnn_output_embed_dim=32,
        shared_trunk_hidden_dim=32,
        num_res_blocks=1, # Minimal ResBlocks
        dropout_rate=0.0 # Disable dropout for test predictability if needed
    ).to(device)
    return model

@pytest.fixture
def temp_data_dir(dummy_training_examples_raw):
    """
    Creates a temporary directory with dummy .pkl training data.
    Each .pkl file will contain a LIST of training examples.
    In this test setup, each file will contain a list with just ONE example,
    to match the structure where dummy_training_examples_raw is a flat list of examples.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # dummy_training_examples_raw is a list of 5 individual example tuples.
        # We want to save each tuple into its own file, but wrapped in a list.
        for i, single_example_tuple in enumerate(dummy_training_examples_raw):
            # Wrap the single example tuple in a list before pickling
            data_to_pickle = [single_example_tuple] 
            with open(os.path.join(tmpdir, f"example_{i}.pkl"), "wb") as f:
                pickle.dump(data_to_pickle, f)
        yield tmpdir

# --- Test Cases ---

def test_load_training_data(temp_data_dir, dummy_training_examples_raw):
    """Tests loading of training data from .pkl files."""
    loaded_examples = load_training_data(temp_data_dir)
    assert len(loaded_examples) == len(dummy_training_examples_raw)
    
    # Compare one example thoroughly
    original_state_tuple, original_policy_np, original_value_np = dummy_training_examples_raw[0]
    loaded_state_tuple, loaded_policy_np, loaded_value_np = loaded_examples[0]

    assert torch.equal(original_state_tuple[0], loaded_state_tuple[0]) # game_state
    assert torch.equal(original_state_tuple[1], loaded_state_tuple[1]) # map_nodes
    assert torch.equal(original_state_tuple[2], loaded_state_tuple[2]) # raw_edge_tensor
    assert np.array_equal(original_policy_np, loaded_policy_np)
    assert np.array_equal(original_value_np, loaded_value_np)

def test_self_play_dataset_item(dummy_self_play_dataset):
    """Tests the __getitem__ method of SelfPlayDataset for correct Data object creation."""
    assert len(dummy_self_play_dataset) > 0, "Dataset should not be empty"
    single_data_object = dummy_self_play_dataset[0]

    assert isinstance(single_data_object, Data), "Output should be a PyG Data object"
    
    # Assert shapes (as per your confirmed correct shapes after previous fixes)
    assert single_data_object.x.shape == (DUMMY_NUM_MAP_NODES, DUMMY_MAP_NODE_FEATURES)
    assert single_data_object.edge_index.shape == (2, DUMMY_NUM_EDGES)
    assert single_data_object.edge_attr.shape == (DUMMY_NUM_EDGES,)
    assert single_data_object.game_state.shape == (1, DUMMY_GAME_STATE_SIZE)
    assert single_data_object.y_policy.shape == (1, DUMMY_POLICY_SIZE)
    assert single_data_object.y_value.shape == (1, DUMMY_VALUE_SIZE)

    # Assert dtypes
    assert single_data_object.x.dtype == torch.float32
    assert single_data_object.edge_index.dtype == torch.long
    assert single_data_object.edge_attr.dtype == torch.long # Converted to long in Data object
    assert single_data_object.game_state.dtype == torch.float32
    assert single_data_object.y_policy.dtype == torch.float32
    assert single_data_object.y_value.dtype == torch.float32

    # Validate graph consistency (PyG 2.3+ returns None on success, older versions True)
    validation_result = single_data_object.validate(raise_on_error=False)
    assert validation_result is None or validation_result is True


def test_dataloader_batching(dummy_self_play_dataset):
    """Tests batching behavior with PyG_DataLoader."""
    batch_size = 2
    assert len(dummy_self_play_dataset) >= batch_size, "Not enough dummy examples for batch size"
    
    dataloader = PyG_DataLoader(dummy_self_play_dataset, batch_size=batch_size, shuffle=False)
    first_batch = next(iter(dataloader))

    assert isinstance(first_batch, Batch), "Dataloader should yield PyG Batch objects"
    assert first_batch.num_graphs == batch_size

    # Assert shapes for batched data
    assert first_batch.x.shape == (batch_size * DUMMY_NUM_MAP_NODES, DUMMY_MAP_NODE_FEATURES)
    assert first_batch.edge_index.shape == (2, batch_size * DUMMY_NUM_EDGES)
    assert first_batch.edge_attr.shape == (batch_size * DUMMY_NUM_EDGES,)
    assert first_batch.game_state.shape == (batch_size, DUMMY_GAME_STATE_SIZE)
    assert first_batch.y_policy.shape == (batch_size, DUMMY_POLICY_SIZE)
    assert first_batch.y_value.shape == (batch_size, DUMMY_VALUE_SIZE)
    
    assert first_batch.batch.shape == (batch_size * DUMMY_NUM_MAP_NODES,)
    assert first_batch.ptr.shape == (batch_size + 1,)

    # Check content of batch.batch (node-to-graph mapping)
    expected_batch_tensor = torch.tensor(
        [i for i in range(batch_size) for _ in range(DUMMY_NUM_MAP_NODES)], 
        dtype=torch.long
    )
    assert torch.equal(first_batch.batch, expected_batch_tensor)


def test_single_training_step(dummy_model, dummy_self_play_dataset, device):
    """Tests a single forward pass, backward pass, and optimizer step."""
    batch_size = 2
    assert len(dummy_self_play_dataset) >= batch_size, "Not enough dummy examples for batch size"

    dataloader = PyG_DataLoader(dummy_self_play_dataset, batch_size=batch_size, shuffle=True)
    # Use a fresh optimizer for the test model instance
    optimizer = optim.AdamW(dummy_model.parameters(), lr=0.001) 
    
    # Loss functions as potentially used in train_model
    policy_loss_fn = nn.CrossEntropyLoss() # Assumes policy_logits and policy_targets are compatible
    value_loss_fn = nn.MSELoss()

    pyg_batch = next(iter(dataloader))
    pyg_batch = pyg_batch.to(device) # Model is already on device from fixture
    
    dummy_model.train() # Ensure model is in training mode

    # Extract data for the model from the PyG Batch object
    game_states_batch = pyg_batch.game_state
    map_nodes_batch = pyg_batch.x
    edge_indices_batch = pyg_batch.edge_index
    node_to_graph_idx_batch = pyg_batch.batch
    edge_attrs_batch = pyg_batch.edge_attr
    
    policy_targets_batch = pyg_batch.y_policy
    value_targets_batch = pyg_batch.y_value

    # --- Training Step ---
    optimizer.zero_grad()
    
    policy_logits, value_preds = dummy_model(
        game_states_batch,
        map_nodes_batch, 
        edge_indices_batch,
        node_to_graph_idx_batch, 
        edge_attrs_batch 
    )
    
    # Assert output shapes from model
    assert policy_logits.shape == (batch_size, DUMMY_POLICY_SIZE)
    assert value_preds.shape == (batch_size, DUMMY_VALUE_SIZE)

    loss_policy = policy_loss_fn(policy_logits, policy_targets_batch)
    loss_value = value_loss_fn(value_preds, value_targets_batch)
    
    total_loss = loss_policy + loss_value 
    
    total_loss.backward()
    optimizer.step()

    # Assert losses are scalar tensors
    assert loss_policy.ndim == 0, "Policy loss should be a scalar"
    assert loss_value.ndim == 0, "Value loss should be a scalar"
    assert total_loss.ndim == 0, "Total loss should be a scalar"

    # Check if gradients exist for some key parameters (indicates backward pass worked)
    assert dummy_model.policy_head.weight.grad is not None
    assert dummy_model.value_head.weight.grad is not None
    # Check a GNN layer's parameter if GNN layers exist
    if hasattr(dummy_model, 'gnn_layers_modulelist') and len(dummy_model.gnn_layers_modulelist) > 0:
        first_gnn_layer_component = dummy_model.gnn_layers_modulelist[0] # This is GATv2Conv
        # GATv2Conv has 'lin_l', 'lin_r', 'att_l', 'att_r' which are Linear layers
        assert first_gnn_layer_component.lin_l.weight.grad is not None
