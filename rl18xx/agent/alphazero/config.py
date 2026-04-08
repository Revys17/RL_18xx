from typing import Optional, Any, Union
import uuid
from torch import device
import torch
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, fields
from torch.utils.tensorboard import SummaryWriter

from rl18xx.agent.alphazero.metrics import Metrics


def _select_best_device() -> torch.device:
    """Select the best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class ModelConfig:
    device: Optional[torch.device] = None
    game_state_size: int = 390
    map_node_features: int = 50
    policy_size: int = 26535
    value_size: int = 4
    mlp_hidden_dim: int = 256
    gnn_node_proj_dim: int = 128
    gnn_hidden_dim_per_head: int = 64
    gnn_layers: int = 4
    gnn_heads: int = 8
    gnn_output_embed_dim: int = 256
    gnn_edge_categories: int = 6
    gnn_edge_embedding_dim: int = 32
    shared_trunk_hidden_dim: int = 512
    num_res_blocks: int = 7
    dropout_rate: float = 0.0
    num_round_types: int = 3  # Stock, Operating, Auction
    film_embed_dim: int = 32
    value_head_layers: int = 3
    aux_loss_weight: float = 0.1
    model_checkpoint_file: Optional[str] = None
    timestamp: Optional[str] = None

    def __post_init__(self):
        if self.device is None:
            self.device = _select_best_device()

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
            "num_round_types": self.num_round_types,
            "film_embed_dim": self.film_embed_dim,
            "value_head_layers": self.value_head_layers,
            "aux_loss_weight": self.aux_loss_weight,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_json(cls, json_data):
        return cls(**json_data)


@dataclass
class ModelV2Config:
    """Configuration for the v2 architecture (Hex Transformer + Economic Transformer)."""

    device: Optional[torch.device] = None
    game_state_size: int = 390
    map_node_features: int = 50
    policy_size: int = 26535
    value_size: int = 4  # number of players
    num_hexes: int = 93
    num_tiles: int = 46
    num_rotations: int = 6

    # Economic State Transformer
    d_entity: int = 128
    econ_transformer_layers: int = 2
    econ_transformer_heads: int = 4
    econ_transformer_ff_dim: int = 256  # d_entity * 2

    # Hex Map Transformer
    d_map: int = 256
    hex_transformer_layers: int = 4
    hex_transformer_heads: int = 8
    hex_transformer_ff_dim: int = 512  # d_map * 2
    max_hex_distance: int = 12

    # Cross-modal Fusion
    cross_attn_heads: int = 4

    # Phase-conditioned Trunk
    d_trunk: int = 512
    num_res_blocks: int = 6
    film_embed_dim: int = 32
    num_round_types: int = 3  # Stock, Operating, Auction
    num_game_phases: int = 7  # for aux phase predictor

    # Heads
    value_head_layers: int = 3
    aux_loss_weight: float = 0.01
    phase_aux_loss_weight: float = 0.01

    model_checkpoint_file: Optional[str] = None
    timestamp: Optional[str] = None

    def __post_init__(self):
        if self.device is None:
            self.device = _select_best_device()

        if self.timestamp is None:
            self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def to_json(self):
        return {f.name: getattr(self, f.name) for f in fields(self) if f.name not in ("device", "model_checkpoint_file")}

    @classmethod
    def from_json(cls, json_data):
        return cls(**json_data)


@dataclass
class TrainingConfig:
    train_dir: Optional[Union[str, Path]] = None
    val_dir: Optional[Union[str, Path]] = None
    batch_size: int = 256
    lr: float = 1e-3
    num_epochs: int = 3
    weight_decay: float = 1e-4
    shuffle_examples: bool = True
    value_loss_weight: float = 1.0
    entropy_weight: float = 0.01  # Phase 6.6: policy entropy bonus weight
    gradient_accumulation_steps: int = 1  # gradient accumulation steps (1 = no accumulation)
    max_training_window: int = 0  # 0 means no windowing (use all data)

    def __post_init__(self):
        if self.train_dir is not None:
            self.train_dir = Path(self.train_dir)
        if self.val_dir is not None:
            self.val_dir = Path(self.val_dir)

    def to_json(self):
        return {
            "train_dir": str(self.train_dir) if self.train_dir else None,
            "val_dir": str(self.val_dir) if self.val_dir else None,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "num_epochs": self.num_epochs,
            "weight_decay": self.weight_decay,
            "shuffle_examples": self.shuffle_examples,
            "value_loss_weight": self.value_loss_weight,
            "entropy_weight": self.entropy_weight,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "max_training_window": self.max_training_window,
        }

    @classmethod
    def from_json(cls, json_data):
        return cls(**json_data)


@dataclass
class SelfPlayConfig:
    max_game_length: int = 1000
    c_puct_base: float = 19652
    c_puct_init: float = 1.25
    c_puct_by_round: Optional[dict] = None  # Phase 6.2: per-round-type c_puct overrides
    dirichlet_noise_alpha: float = 0.03  # Kept for backward compatibility; inject_noise() now uses 10/num_legal_actions
    dirichlet_noise_weight: float = 0.25
    softpick_move_cutoff: int = 500
    num_readouts: int = 200
    min_readouts: int = 50
    parallel_readouts: int = 32
    backup_discount: float = 0.995  # Phase 6.1: depth-discounted value backup
    pw_c: float = 1.0  # Phase 6.3: progressive widening constant
    pw_alpha: float = 0.5  # Phase 6.3: progressive widening exponent
    use_score_values: bool = True  # Phase 6.4: use normalized score fractions instead of win/loss
    use_fp16_inference: bool = True  # Phase 6.5: FP16 inference during self-play
    network: Any = None
    metrics: Optional[Metrics] = None
    global_step: int = 0
    game_idx_in_iteration: int = 0
    game_id: Optional[str] = None
    selfplay_dir: str = "selfplay"

    def __post_init__(self):
        assert self.softpick_move_cutoff % 2 == 0
        assert self.num_readouts > 0
        if self.game_id is None:
            self.game_id = str(uuid.uuid4())
        if self.c_puct_by_round is None:
            self.c_puct_by_round = {
                "Auction": 1.5,
                "WaterfallAuction": 1.5,
                "Stock": 1.25,
                "Operating": 1.0,
            }

        root_dir = Path("training_examples")
        self.selfplay_dir = root_dir / self.selfplay_dir
