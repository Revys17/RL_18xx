from typing import Optional, Any, Union
import uuid
import random
import warnings
from torch import device
import torch
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, fields
from torch.utils.tensorboard import SummaryWriter

from rl18xx.agent.alphazero.metrics import Metrics

_V1_CONFIG_DEPRECATION_MESSAGE = (
    "ModelConfig is deprecated; it configures the legacy GNN AlphaZeroGNNModel "
    "(v1). Use ModelV2Config with AlphaZeroV2Model — the transformer "
    "architecture is the default. ModelConfig is kept only for loading legacy "
    "checkpoints."
)


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
    seed: Optional[int] = None

    def __post_init__(self):
        warnings.warn(_V1_CONFIG_DEPRECATION_MESSAGE, DeprecationWarning, stacklevel=3)
        if self.device is None:
            self.device = _select_best_device()

        if self.timestamp is None:
            self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if self.seed is None:
            self.seed = random.randint(0, 2**31 - 1)

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
            "seed": self.seed,
        }

    @classmethod
    def from_json(cls, json_data):
        filtered = {k: v for k, v in json_data.items() if k in {f.name for f in fields(cls)} and k != "device"}
        return cls(**filtered)


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
    seed: Optional[int] = None

    def __post_init__(self):
        if self.device is None:
            self.device = _select_best_device()

        if self.timestamp is None:
            self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if self.seed is None:
            self.seed = random.randint(0, 2**31 - 1)

    def to_json(self):
        return {f.name: getattr(self, f.name) for f in fields(self) if f.name not in ("device", "model_checkpoint_file")}

    @classmethod
    def from_json(cls, json_data):
        filtered = {k: v for k, v in json_data.items() if k in {f.name for f in fields(cls)} and k != "device"}
        return cls(**filtered)


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
    value_lr_multiplier: float = 3.0  # multiplier for value head learning rate relative to config.lr
    use_fp16_training: bool = True  # use mixed-precision (FP16) training on CUDA

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
            "value_lr_multiplier": self.value_lr_multiplier,
            "use_fp16_training": self.use_fp16_training,
        }

    @classmethod
    def from_json(cls, json_data):
        return cls(**json_data)


def _default_c_puct_by_round() -> dict:
    return {
        "Auction": 1.5,
        "WaterfallAuction": 1.5,
        "Stock": 1.25,
        "Operating": 1.0,
    }


@dataclass(frozen=True)
class SelfPlayHyperparams:
    """Immutable MCTS/self-play hyperparameters.

    These describe *how* self-play runs and never change for the lifetime of a
    SelfPlayConfig. Mutable runtime state (network, metrics, game id) lives on
    SelfPlayRuntime instead.
    """

    max_game_length: int = 1000
    c_puct_base: float = 19652
    c_puct_init: float = 1.25
    c_puct_by_round: dict = field(default_factory=_default_c_puct_by_round)
    dirichlet_noise_alpha: float = 0.03
    dirichlet_noise_weight: float = 0.25
    softpick_move_cutoff: int = 500
    num_readouts: int = 200
    min_readouts: int = 50
    parallel_readouts: int = 32
    backup_discount: float = 0.995
    pw_c: float = 1.0
    pw_alpha: float = 0.5
    use_score_values: bool = True
    use_fp16_inference: bool = True
    adaptive_readout_threshold: int = 5

    def __post_init__(self):
        assert self.softpick_move_cutoff % 2 == 0
        assert self.num_readouts > 0


@dataclass
class SelfPlayRuntime:
    """Mutable runtime state attached to a self-play session.

    Lives alongside SelfPlayHyperparams inside a SelfPlayConfig bundle.
    """

    network: Any = None
    metrics: Optional[Metrics] = None
    global_step: int = 0
    game_idx_in_iteration: int = 0
    game_id: Optional[str] = None
    selfplay_dir: Any = "selfplay"

    def __post_init__(self):
        if self.game_id is None:
            self.game_id = str(uuid.uuid4())
        if isinstance(self.selfplay_dir, str):
            self.selfplay_dir = Path("training_examples") / self.selfplay_dir


_HYPER_FIELDS = frozenset(f.name for f in fields(SelfPlayHyperparams))
_RUNTIME_FIELDS = frozenset(f.name for f in fields(SelfPlayRuntime))


class SelfPlayConfig:
    """Bundle of immutable hyperparams and mutable runtime context for self-play.

    Hyperparameters live on ``config.hyperparams`` (frozen); runtime fields
    (``network``, ``metrics``, ``global_step``, ``game_idx_in_iteration``,
    ``game_id``, ``selfplay_dir``) live on ``config.runtime`` and are
    individually mutable.

    For source compatibility, the hyperparameter and runtime names are also
    accessible as flat attributes on the config itself (e.g.
    ``config.num_readouts`` and ``config.network``). Writing to a runtime
    attribute mutates the runtime object; hyperparameter attributes are
    read-only via this proxy (the frozen dataclass enforces immutability).
    """

    __slots__ = ("hyperparams", "runtime")

    def __init__(self, **kwargs):
        hyper_kwargs = {k: kwargs.pop(k) for k in list(kwargs) if k in _HYPER_FIELDS}
        runtime_kwargs = {k: kwargs.pop(k) for k in list(kwargs) if k in _RUNTIME_FIELDS}
        if kwargs:
            raise TypeError(f"Unknown SelfPlayConfig kwargs: {sorted(kwargs)}")
        object.__setattr__(self, "hyperparams", SelfPlayHyperparams(**hyper_kwargs))
        object.__setattr__(self, "runtime", SelfPlayRuntime(**runtime_kwargs))

    def __getattr__(self, name):
        # Only called when normal lookup fails (i.e., name is not in __slots__).
        if name in _HYPER_FIELDS:
            return getattr(self.hyperparams, name)
        if name in _RUNTIME_FIELDS:
            return getattr(self.runtime, name)
        raise AttributeError(f"SelfPlayConfig has no attribute {name!r}")

    def __setattr__(self, name, value):
        if name in _RUNTIME_FIELDS:
            setattr(self.runtime, name, value)
        elif name in _HYPER_FIELDS:
            raise AttributeError(
                f"Cannot set hyperparameter {name!r} after construction — "
                f"SelfPlayHyperparams is frozen."
            )
        elif name in ("hyperparams", "runtime"):
            object.__setattr__(self, name, value)
        else:
            raise AttributeError(f"SelfPlayConfig has no attribute {name!r}")
