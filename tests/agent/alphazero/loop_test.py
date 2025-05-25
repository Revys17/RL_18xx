import pytest
import torch
import os
import tempfile
from rl18xx.agent.alphazero.loop import train_loop

def test_minimal_training_loop():
    """Test that a minimal training loop can complete successfully."""
    # Create temporary directories for test data
    with tempfile.TemporaryDirectory() as temp_dir:
        # Configure minimal parameters
        config = {
            "game_title": "1830",
            "player_options": {"1": "P1", "2": "P2", "3": "P3", "4": "P4"},
            "num_loop_iterations": 1,  # Just one iteration
            "num_self_play_games_per_iteration": 1,  # Just one game
            "mcts_simulations": 10,  # Minimal MCTS simulations
            "mcts_initial_temperature": 1.0,
            "mcts_temp_decay_steps": 1,
            "mcts_c_puct": 1.0,
            "mcts_dirichlet_alpha": 0.03,
            "mcts_dirichlet_noise_factor": 0.25,
            "training_epochs_per_iteration": 1,  # Just one epoch
            "training_batch_size": 2,
            "training_learning_rate": 0.001,
            "policy_loss_weight": 1.0,
            "value_loss_weight": 1.0,
            "base_checkpoint_dir": os.path.join(temp_dir, "checkpoints"),
            "base_training_data_dir": os.path.join(temp_dir, "training_data"),
            "base_log_dir": os.path.join(temp_dir, "logs"),
            "device": torch.device("cpu")
        }

        # Run the training loop
        try:
            train_loop(**config)
            # If we get here, the loop completed without errors
            assert True
        except Exception as e:
            pytest.fail(f"Training loop failed with error: {str(e)}")

        # Verify that expected files were created
        assert os.path.exists(config["base_checkpoint_dir"])
        assert os.path.exists(config["base_training_data_dir"])
        assert os.path.exists(config["base_log_dir"])
        
        # Check for specific files we expect
        checkpoint_files = os.listdir(config["base_checkpoint_dir"])
        assert len(checkpoint_files) > 0, "No checkpoint files were created"
        
        training_data_files = os.listdir(config["base_training_data_dir"])
        assert len(training_data_files) > 0, "No training data files were created"