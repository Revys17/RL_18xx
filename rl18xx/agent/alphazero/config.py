from typing import Optional, Any, Union
from torch import device
import torch
from datetime import datetime
from pathlib import Path


class ModelConfig:
    def __init__(
        self,
        device: Optional[device] = None,
        game_state_size: int = 377,
        map_node_features: int = 50,
        policy_size: int = 26535,
        value_size: int = 4,
        mlp_hidden_dim: int = 256,
        gnn_node_proj_dim: int = 128,
        gnn_hidden_dim_per_head: int = 64,
        gnn_layers: int = 3,
        gnn_heads: int = 4,
        gnn_output_embed_dim: int = 256,
        gnn_edge_categories: int = 6,
        gnn_edge_embedding_dim: int = 32,
        shared_trunk_hidden_dim: int = 512,
        num_res_blocks: int = 5,
        dropout_rate: float = 0.1,
        model_checkpoint_file: Optional[str] = None,
        timestamp: Optional[str] = None,
    ):
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device
        self.game_state_size = game_state_size
        self.map_node_features = map_node_features
        self.policy_size = policy_size
        self.value_size = value_size
        self.mlp_hidden_dim = mlp_hidden_dim
        self.gnn_node_proj_dim = gnn_node_proj_dim
        self.gnn_hidden_dim_per_head = gnn_hidden_dim_per_head
        self.gnn_layers = gnn_layers
        self.gnn_heads = gnn_heads
        self.gnn_output_embed_dim = gnn_output_embed_dim
        self.gnn_edge_categories = gnn_edge_categories
        self.gnn_edge_embedding_dim = gnn_edge_embedding_dim
        self.shared_trunk_hidden_dim = shared_trunk_hidden_dim
        self.num_res_blocks = num_res_blocks
        self.dropout_rate = dropout_rate
        self.model_checkpoint_file = model_checkpoint_file
        self.timestamp = timestamp if timestamp is not None else datetime.now().strftime("%Y%m%d_%H%M%S")

    def to_json(self):
        return {
            "game_state_size": self.game_state_size,
            "map_node_features": self.map_node_features,
            "policy_size": self.policy_size,
            "value_size": self.value_size,
            "mlp_hidden_dim": self.mlp_hidden_dim,
            "gnn_node_proj_dim": self.gnn_node_proj_dim,
            "gnn_hidden_dim_per_head": self.gnn_hidden_dim_per_head,
            "gnn_layers": self.gnn_layers,
            "gnn_heads": self.gnn_heads,
            "gnn_output_embed_dim": self.gnn_output_embed_dim,
            "gnn_edge_categories": self.gnn_edge_categories,
            "gnn_edge_embedding_dim": self.gnn_edge_embedding_dim,
            "shared_trunk_hidden_dim": self.shared_trunk_hidden_dim,
            "num_res_blocks": self.num_res_blocks,
            "dropout_rate": self.dropout_rate,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_json(cls, json_data):
        return cls(**json_data)


class TrainingConfig:
    def __init__(
        self,
        root_dir: str = "training_examples",
        train_dir: Optional[str] = None,
        val_dir: Optional[str] = None,
        model_checkpoint_dir: str = "model_checkpoints",
        train_batch_size: int = 256,
        lr: float = 1e-3,
        num_epochs: int = 1,
        weight_decay: float = 1e-4,
        shuffle_examples: bool = True,
        value_loss_weight: float = 1.0,
    ):
        self.root_dir = Path(root_dir)
        self.model_checkpoint_dir = model_checkpoint_dir
        self.train_batch_size = train_batch_size
        self.lr = lr
        self.num_epochs = num_epochs
        self.weight_decay = weight_decay
        self.shuffle_examples = shuffle_examples
        self.value_loss_weight = value_loss_weight

        if train_dir is None and val_dir is None:
            return

        if train_dir is None or val_dir is None:
            raise ValueError("train_dir and val_dir must both be provided or both be None")

        self.train_dir = self.root_dir / train_dir
        self.val_dir = self.root_dir / val_dir


class SelfPlayConfig:
    def __init__(
        self,
        # MCTS
        max_game_length: int = 1000,
        c_puct_base: float = 19652,
        c_puct_init: float = 1.25,
        dirichlet_noise_alpha: float = 0.03,
        dirichlet_noise_weight: float = 0.25,
        # Self Play
        softpick_move_cutoff: int = 500,
        num_readouts: int = 200,
        parallel_readouts: int = 8,
        network: Any = None,
        selfplay_dir: str = "selfplay",
        holdout_dir: str = "holdout",
        holdout_pct: float = 0.05,
    ):
        assert softpick_move_cutoff % 2 == 0
        assert num_readouts > 0

        # MCTS
        self.max_game_length = max_game_length
        self.c_puct_base = c_puct_base
        self.c_puct_init = c_puct_init
        self.dirichlet_noise_alpha = dirichlet_noise_alpha
        self.dirichlet_noise_weight = dirichlet_noise_weight

        # MCTS Player
        self.softpick_move_cutoff = softpick_move_cutoff
        self.num_readouts = num_readouts
        self.parallel_readouts = parallel_readouts
        self.network = network

        # Self Play
        root_dir = Path("training_examples")
        self.selfplay_dir = root_dir / selfplay_dir
        self.holdout_dir = root_dir / holdout_dir
        self.holdout_pct = holdout_pct
