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

_GNN_CONFIG_DEPRECATION_MESSAGE = (
    "ModelGNNConfig configures the legacy AlphaZeroGNNModel. Use "
    "ModelTransformerConfig with AlphaZeroTransformerModel for new training "
    "runs. ModelGNNConfig is kept only for loading legacy checkpoints and "
    "for the GNN-vs-Transformer benchmarks."
)


def _select_best_device() -> torch.device:
    """Select the best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class ModelGNNConfig:
    device: Optional[torch.device] = None
    game_state_size: int = 390
    map_node_features: int = 50
    policy_size: int = 26537
    # Number of player slots the value head emits. The legacy GNN architecture
    # defaults to 4 (the original AlphaZero-style 1830 setup); the active
    # Transformer architecture uses ``MAX_PLAYERS = 6``. ``value_size`` is kept
    # as a synonym (mirrored in ``__post_init__``) for legacy code that reads
    # the old name.
    num_players: int = 4
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
        warnings.warn(_GNN_CONFIG_DEPRECATION_MESSAGE, DeprecationWarning, stacklevel=3)
        if self.device is None:
            self.device = _select_best_device()

        if self.timestamp is None:
            self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if self.seed is None:
            self.seed = random.randint(0, 2**31 - 1)

        # Keep ``value_size`` in sync with ``num_players`` so legacy callers
        # that still read ``value_size`` (e.g., pretraining validation
        # bookkeeping) see the right size.
        if self.value_size != self.num_players:
            self.value_size = self.num_players

    def to_json(self):
        return {
            "game_state_size": self.game_state_size,
            "map_node_features": self.map_node_features,
            "policy_size": self.policy_size,
            "num_players": self.num_players,
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
class ModelTransformerConfig:
    """Configuration for the Transformer architecture (Hex Transformer + Economic Transformer).

    A single checkpoint supports the full 2..6 player-count range. The model's
    fixed-shape buffers (id embedding, gather indices, value-head output) are
    built for ``max_players``; shorter games pad their player feature regions
    with zeros and mask out the unused slots in attention via a per-sample
    ``key_padding_mask`` derived from each example's actual ``num_players``.
    """

    device: Optional[torch.device] = None
    # ``game_state_size`` is the *padded* (max-N) layout size — the encoder
    # always emits a state of this length, regardless of the actual player
    # count, so the model's gather indices have a single fixed layout. The
    # post_init below recomputes this from ``max_players`` so callers don't
    # have to know the precomputed sizes.
    game_state_size: int = 408  # max_players=6 layout size; recomputed in __post_init__.
    map_node_features: int = 50
    policy_size: int = 26537
    # Number of player slots the value head emits — one logit per player slot
    # (active first via canonicalization, padded slots last). Loss masking
    # handles the padded slots. This is the canonical knob for the value head
    # output dimension; ``value_size`` is kept as a synonym (mirrored in
    # ``__post_init__``) for downstream callers that still read the old name.
    num_players: int = 6
    value_size: int = 6
    # Maximum number of player slots the model supports (2..6). A single
    # checkpoint trained with ``max_players = 6`` consumes games of any count
    # in [2, max_players] by padding shorter games' player features to
    # ``max_players`` slots and masking the unused slots in attention.
    max_players: int = 6
    num_hexes: int = 93
    num_tiles: int = 46
    num_rotations: int = 6

    # Economic State Transformer
    d_entity: int = 128
    econ_transformer_layers: int = 2
    econ_transformer_heads: int = 4
    econ_transformer_ff_dim: int = 256  # d_entity * 2

    # Map encoder selection: "transformer" (HexTransformerMapEncoder) or
    # "resnet" (HexResNetMapEncoder). Both expose the same
    # ``(per_hex_embeddings, map_pool)`` outputs to the rest of the model; only
    # the inductive bias of the encoder changes.
    map_encoder: str = "transformer"

    # Hex Map Transformer (used when ``map_encoder == "transformer"``)
    d_map: int = 256
    hex_transformer_layers: int = 4
    hex_transformer_heads: int = 8
    hex_transformer_ff_dim: int = 512  # d_map * 2
    max_hex_distance: int = 12

    # Hex ResNet Map Encoder (used when ``map_encoder == "resnet"``).
    # The ResNet operates on an offset-grid projection of the 93 hex coords
    # (~11x13 for 1830) at ``resnet_channels`` channels for ``resnet_layers``
    # residual blocks. Receptive field after ~10 layers covers the full map.
    resnet_channels: int = 128
    resnet_layers: int = 10

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

        # Keep ``num_players`` / ``value_size`` in sync with ``max_players``.
        # The value head emits one logit per player slot, padded slots
        # included; loss masking zeros the gradient on slots beyond a sample's
        # actual player count. ``value_size`` is a synonym retained for legacy
        # callers that still read the old name.
        if self.num_players != self.max_players:
            self.num_players = self.max_players
        if self.value_size != self.num_players:
            self.value_size = self.num_players
        # Layout-dependent default: recompute ``game_state_size`` from the
        # encoder's max-N layout. The encoder always emits a state of this
        # length (shorter games pad their player feature regions with zeros)
        # so the model's gather indices have a single fixed layout.
        from rl18xx.agent.alphazero.encoder import Encoder_GNN as _Encoder
        _, computed_size = _Encoder.compute_section_layout(self.max_players)
        if self.game_state_size != computed_size:
            self.game_state_size = computed_size
        assert 2 <= self.max_players <= 6, (
            f"ModelTransformerConfig.max_players must be 2..6, got {self.max_players}"
        )
        assert self.map_encoder in ("transformer", "resnet"), (
            f"ModelTransformerConfig.map_encoder must be 'transformer' or 'resnet', "
            f"got {self.map_encoder!r}"
        )

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
    # NLL weight for the continuous price head. Multiplies the per-example
    # ``Normal(mean, exp(log_std)).log_prob(price)`` loss summed across
    # price-bearing legal actions. Default kept small (0.1) so the price
    # gradient nudges the trunk without dominating the structural policy
    # losses while the head warms up.
    price_loss_weight: float = 0.1
    # KataGo-style dual value head: the score head (MSE on normalized net-worth
    # fractions at game end) is auxiliary — it gives the trunk a dense gradient
    # signal but is not consumed by MCTS, which backs up the win-loss head only.
    # Default small (0.1) relative to the primary policy/win-loss losses so the
    # auxiliary signal nudges representations without dominating training.
    score_loss_weight: float = 0.1
    entropy_weight: float = 0.01  # Phase 6.6: policy entropy bonus weight
    gradient_accumulation_steps: int = 1  # gradient accumulation steps (1 = no accumulation)
    # 0 = no windowing; default is approximately 5 iterations worth of examples.
    max_training_window: int = 250_000
    value_lr_multiplier: float = 3.0  # multiplier for value head learning rate relative to config.lr
    use_fp16_training: bool = True  # use mixed-precision (FP16) training on CUDA
    pretrain_label_smoothing: float = 0.03  # epsilon for smoothing policy targets during pretraining
    pretrain_validation_percentage: float = 0.05  # per-game probability of routing a game to validation

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
            "price_loss_weight": self.price_loss_weight,
            "score_loss_weight": self.score_loss_weight,
            "entropy_weight": self.entropy_weight,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "max_training_window": self.max_training_window,
            "value_lr_multiplier": self.value_lr_multiplier,
            "use_fp16_training": self.use_fp16_training,
            "pretrain_label_smoothing": self.pretrain_label_smoothing,
            "pretrain_validation_percentage": self.pretrain_validation_percentage,
        }

    @classmethod
    def from_json(cls, json_data):
        filtered = {k: v for k, v in json_data.items() if k in {f.name for f in fields(cls)}}
        return cls(**filtered)


def _default_c_puct_by_round() -> dict:
    return {
        "Auction": 1.5,
        "WaterfallAuction": 1.5,
        "Stock": 1.25,
        "Operating": 1.0,
    }


@dataclass(frozen=True)
class TraceConfig:
    """Configuration for Phase 1 PlayoutTrace instrumentation.

    Defaults stay off (``trace_game_rate=0.0``). When enabled, ``MCTSPlayer``
    decides at game start (via a single coin flip) whether to record traces
    for the entire game — traces are whole games, not isolated moves with no
    surrounding context. Within a traced game, ``trace_every_n_moves``
    selects which moves get traced and ``traces_per_move`` caps the number
    of leaves recorded per traced move.

    Trace output is one JSONL file per traced game at
    ``output_dir/{iteration}/{game_id}.jsonl`` with a header line carrying a
    config snapshot + player ids.
    """

    trace_game_rate: float = 0.0
    trace_every_n_moves: int = 1
    traces_per_move: int = 4
    output_dir: Path = field(default_factory=lambda: Path("traces"))


def _default_trace_config() -> "TraceConfig":
    return TraceConfig()


def _default_player_count_distribution() -> dict:
    """Default sampling distribution over self-play player counts.

    Heavily weighted toward 4-player (the bulk of human data and the most
    common testing target); the other counts are sampled with smaller
    probabilities so that the model is exposed to variable-length entity
    sequences during training. Keys are ``int`` player counts; values are
    probability weights (need not sum to 1 — they are normalized at sample
    time).
    """
    return {2: 0.05, 3: 0.2, 4: 0.6, 5: 0.1, 6: 0.05}


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
    # Concentration constant for the per-action Dirichlet noise: the symmetric
    # alpha is computed as ``dirichlet_noise_concentration / num_legal_actions``
    # so larger action spaces get a flatter prior. Default of 10 matches the
    # value used historically by AlphaZero-style 1830 self-play.
    dirichlet_noise_concentration: float = 10.0
    softpick_move_cutoff: int = 500
    num_readouts: int = 200
    min_readouts: int = 50
    parallel_readouts: int = 32
    backup_discount: float = 0.995
    # Progressive widening schedule for the continuous price head:
    # ``target_children(N) = ceil(pw_c * N^pw_alpha)``. With the defaults
    # ``c=1.0, alpha=0.5`` the slot accumulates ~sqrt(N) grandchildren —
    # i.e. ~14 grandchildren at the 200-readout default. Tune lower if
    # ``MCTS/Price/Grandchildren_Per_PW_Slot`` shows we're over-exploring
    # the continuous dimension at the expense of categorical PUCT, higher
    # if the price head's posterior collapses to a single mode too quickly.
    pw_c: float = 1.0
    pw_alpha: float = 0.5
    # Minimum number of price children to materialize on a price-bearing
    # categorical node before progressive widening kicks in. Ensures every
    # such node has at least one sampled price even at very low visit counts.
    min_price_children: int = 1
    use_score_values: bool = True
    use_fp16_inference: bool = True
    adaptive_readout_threshold: int = 5
    # Variable player count support: each self-play game samples a player count
    # from this distribution. Use ``{4: 1.0}`` to lock self-play to 4-player
    # games (the legacy behaviour). Keys = player counts in 2..6; values =
    # weights normalized at sample time.
    player_count_distribution: dict = field(default_factory=_default_player_count_distribution)
    # Phase 1 PlayoutTrace instrumentation (debug only — off by default).
    trace: TraceConfig = field(default_factory=_default_trace_config)
    # Phase 2 multiplayer consensus resign. End decisively-lost games early
    # based on a rolling window of root Q vectors. See
    # docs/mcts_improvements_plan.md "Phase 2 — Multiplayer consensus resign".
    enable_resign: bool = True
    # ``K`` in the spec — number of recent moves the leader/gap conditions
    # must hold over.
    resign_window: int = 8
    # Minimum ``min_over_window(Q_leader)`` required to resign. Holdout
    # calibration adjusts this between iterations.
    resign_high_threshold: float = 0.65
    # Minimum ``min_over_window(Q_leader - Q_second)`` required to resign.
    # Held fixed by the calibration (margin, not confidence claim).
    resign_gap_threshold: float = 0.30
    # Fraction of games that are flagged as no-resign holdouts at game start.
    # Holdouts play to completion regardless of resign signal; the would-have-
    # resigned moment is recorded for false-positive-rate calibration.
    noresign_holdout_rate: float = 0.10
    # Lower clamp on auto-calibrated ``resign_high_threshold``.
    resign_high_threshold_min: float = 0.45
    # Phase 3 cross-process inference server. Defaults stay off — the
    # legacy in-process inference path remains the production path until
    # a real training run verifies the server's parity + GPU contention
    # behaviour. When ``True``, ``MCTSPlayer`` routes both ``run_encoded``
    # (first-node expansion) and ``run_many_encoded`` (tree_search batch)
    # through ``SelfPlayConfig.inference_client`` instead of the local
    # model. See docs/mcts_improvements_plan.md "Phase 3".
    use_inference_server: bool = False
    inference_batch_size: int = 64
    inference_batch_timeout_ms: float = 2.0
    # Phase 4b Rust MCTS. Defaults off until parity is verified on a full
    # training run. When True, ``SelfPlay`` instantiates ``RustMCTSPlayer``
    # from rust_mcts_player.py instead of the Python ``MCTSPlayer``.
    # Categorical-only — Bid/BuyTrain/BuyCompany slots are treated as
    # fixed-price at price_range[0]. PW + continuous prices land in 4c.
    use_rust_mcts: bool = False

    def __post_init__(self):
        assert self.softpick_move_cutoff % 2 == 0
        assert self.num_readouts > 0
        # Defensive validation: keys must be 2..6 ints and weights non-negative.
        for k, v in self.player_count_distribution.items():
            assert isinstance(k, int) and 2 <= k <= 6, (
                f"player_count_distribution keys must be ints in 2..6, got {k!r}"
            )
            assert v >= 0, f"player_count_distribution weights must be >= 0, got {v}"
        assert sum(self.player_count_distribution.values()) > 0, (
            "player_count_distribution must have at least one positive weight"
        )


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
    # Phase 3 cross-process inference client. When set on a worker, MCTS
    # routes inference through this object instead of ``network``.
    inference_client: Any = None

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
