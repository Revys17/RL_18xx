import pytest
from rl18xx.agent.alphazero.model import Model, INPUT_SIZE, POLICY_OUTPUT_SIZE, VALUE_OUTPUT_SIZE
import torch
import torch.nn as nn


@pytest.fixture
def model():
    return Model()


@pytest.fixture(params=[1, 4])  # Test with different batch sizes
def input_batch(request):
    """Provides dummy input tensors with varying batch sizes."""
    batch_size = request.param
    return torch.randn(batch_size, INPUT_SIZE)


def test_model_instantiation(model):
    """Tests if the model can be instantiated."""
    assert model is not None
    assert isinstance(model, nn.Module)


def test_model_forward_pass_shapes(model, input_batch):
    """Tests the forward pass and output shapes."""
    batch_size = input_batch.shape[0]
    model.eval()  # Set to evaluation mode for consistent BatchNorm behavior
    with torch.no_grad():  # Disable gradient calculation for testing
        policy_log_probs, value = model(input_batch)

    assert policy_log_probs.shape == (
        batch_size,
        POLICY_OUTPUT_SIZE,
    ), f"Policy shape mismatch for batch size {batch_size}"
    assert value.shape == (batch_size, VALUE_OUTPUT_SIZE), f"Value shape mismatch for batch size {batch_size}"


def test_model_output_types(model, input_batch):
    """Tests the data types of the outputs."""
    model.eval()
    with torch.no_grad():
        policy_log_probs, value = model(input_batch)

    assert isinstance(policy_log_probs, torch.Tensor)
    assert isinstance(value, torch.Tensor)
    # Check dtype if necessary, should be float32 by default
    assert policy_log_probs.dtype == torch.float32
    assert value.dtype == torch.float32


def test_model_policy_output_probs(model, input_batch):
    """Tests if policy outputs are valid log probabilities (exp sum ~ 1)."""
    model.eval()
    with torch.no_grad():
        policy_log_probs, _ = model(input_batch)

    # Convert log probabilities to probabilities
    policy_probs = torch.exp(policy_log_probs)

    # Check that probabilities sum to approximately 1 along the action dimension
    sum_probs = torch.sum(policy_probs, dim=1)
    assert torch.allclose(
        sum_probs, torch.ones_like(sum_probs), atol=1e-6
    ), f"Policy probabilities do not sum to 1. Sums: {sum_probs}"


def test_model_value_output_range(model, input_batch):
    """Tests if the value output is within the tanh range [-1, 1]."""
    model.eval()
    with torch.no_grad():
        _, value = model(input_batch)

    assert torch.all(value >= -1.0), f"Value output below -1: {value}"
    assert torch.all(value <= 1.0), f"Value output above 1: {value}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_model_device_movement(model, input_batch):
    """Tests if the model and data can be moved to CUDA and run."""
    device = torch.device("cuda")
    model.to(device)
    input_batch_cuda = input_batch.to(device)

    model.eval()
    with torch.no_grad():
        try:
            policy_log_probs, value = model(input_batch_cuda)
            # Basic checks to ensure it ran on CUDA
            assert policy_log_probs.device.type == "cuda"
            assert value.device.type == "cuda"
            assert policy_log_probs.shape[0] == input_batch.shape[0]
            assert value.shape[0] == input_batch.shape[0]
        except Exception as e:
            pytest.fail(f"Model forward pass failed on CUDA: {e}")
