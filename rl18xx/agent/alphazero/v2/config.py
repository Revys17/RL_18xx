from typing import Optional, Any
from torch import device
import torch

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

class MegaConfig:
    def __init__(
        self,
        # MCTS
        max_game_length: int=1000,
        c_puct_base: float=19652,
        c_puct_init: float=1.25,
        dirichlet_noise_alpha: float=0.03,
        dirichlet_noise_weight: float=0.25,
        # Self Play
        softpick_move_cutoff: int = 500,
        num_readouts: int = 32,
        parallel_readouts: int = 8,
        network: Any = None,
        selfplay_dir: str = None,
        holdout_dir: str = None,
        holdout_pct: float = 0.05,
        # Training
        train_batch_size: int = 1024,
        lr: float = 0.001,
        weight_decay: float = 0.0,
        shuffle_buffer_size: int = 2000,
        shuffle_examples: bool = True,
        steps_to_train: int = None,
        num_examples: int = None,
        window_size: int = 500000,
        filter_amount: float = 1.0,
        export_path: str = None,
        freeze: bool = False,
        use_trt: bool = False,
        trt_max_batch_size: int = None,
        trt_precision: str = 'fp32',
        value_loss_weight: float = 1.0,
    ):
        assert softpick_move_cutoff % 2 == 0
        assert num_readouts > 0
        assert not use_trt or trt_max_batch_size, 'trt_max_batch_size must be set if use_trt is true'
        assert not num_examples or steps_to_train == 0 and filter_amount == 1.0, '`num_examples` requires `steps_to_train==0` and `filter_amount==1.0`'

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
        self.selfplay_dir = selfplay_dir
        self.holdout_dir = holdout_dir
        self.holdout_pct = holdout_pct

        # Training
        self.train_batch_size = train_batch_size
        self.shuffle_buffer_size = shuffle_buffer_size
        self.shuffle_examples = shuffle_examples
        self.steps_to_train = steps_to_train
        self.num_examples = num_examples
        self.window_size = window_size
        self.filter_amount = filter_amount
        self.export_path = export_path
        self.freeze = freeze
        self.use_trt = use_trt
        self.trt_max_batch_size = trt_max_batch_size
        self.trt_precision = trt_precision
        self.lr = lr
        self.weight_decay = weight_decay
        self.value_loss_weight = value_loss_weight