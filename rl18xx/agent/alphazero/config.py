from typing import Optional, Any, Union
import uuid
from torch import device
import torch
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, fields
from torch.utils.tensorboard import SummaryWriter


@dataclass
class ModelConfig:
    device: Optional[torch.device] = None
    game_state_size: int = 377
    map_node_features: int = 50
    policy_size: int = 26535
    value_size: int = 4
    mlp_hidden_dim: int = 256
    gnn_node_proj_dim: int = 128
    gnn_hidden_dim_per_head: int = 64
    gnn_layers: int = 3
    gnn_heads: int = 4
    gnn_output_embed_dim: int = 256
    gnn_edge_categories: int = 6
    gnn_edge_embedding_dim: int = 32
    shared_trunk_hidden_dim: int = 512
    num_res_blocks: int = 5
    dropout_rate: float = 0.1
    model_checkpoint_file: Optional[str] = None
    timestamp: Optional[str] = None

    def __post_init__(self):
        if self.device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        
        if self.timestamp is None:
            self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

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


@dataclass
class TrainingConfig:
    root_dir: str = "training_examples"
    train_dir: Optional[str] = None
    val_dir: Optional[str] = None
    model_checkpoint_dir: str = "model_checkpoints"
    train_batch_size: int = 256
    lr: float = 1e-3
    num_epochs: int = 1
    weight_decay: float = 1e-4
    shuffle_examples: bool = True
    value_loss_weight: float = 1.0
    learning_rate: float = 0.001
    batch_size: int = 256
    writer: Optional[SummaryWriter] = None
    global_step: int = 0

    def __post_init__(self):
        self.root_dir = Path(self.root_dir)
        self.model_checkpoint_dir = self.model_checkpoint_dir
        self.train_batch_size = self.train_batch_size
        self.lr = self.lr
        self.num_epochs = self.num_epochs
        self.weight_decay = self.weight_decay
        self.shuffle_examples = self.shuffle_examples
        self.value_loss_weight = self.value_loss_weight

        if self.train_dir is None and self.val_dir is None:
            return

        if self.train_dir is None or self.val_dir is None:
            raise ValueError("train_dir and val_dir must both be provided or both be None")

        self.train_dir = self.root_dir / self.train_dir
        self.val_dir = self.root_dir / self.val_dir


@dataclass
class SelfPlayConfig:
    max_game_length: int = 1000
    c_puct_base: float = 19652
    c_puct_init: float = 1.25
    dirichlet_noise_alpha: float = 0.03
    dirichlet_noise_weight: float = 0.25
    softpick_move_cutoff: int = 500
    num_readouts: int = 200
    parallel_readouts: int = 8
    network: Any = None
    writer: Optional[SummaryWriter] = None
    global_step: int = 0
    game_id: Optional[str] = None
    selfplay_dir: str = "selfplay"
    holdout_dir: str = "holdout"
    holdout_pct: float = 0.05

    def __post_init__(self):
        assert self.softpick_move_cutoff % 2 == 0
        assert self.num_readouts > 0
        if self.game_id is None:
            self.game_id = str(uuid.uuid4())

        root_dir = Path("training_examples")
        self.selfplay_dir = root_dir / self.selfplay_dir
        self.holdout_dir = root_dir / self.holdout_dir
        self.holdout_pct = self.holdout_pct
