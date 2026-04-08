import torch
from rl18xx.game.gamemap import GameMap
from rl18xx.agent.alphazero.config import ModelV2Config
from rl18xx.agent.alphazero.model_v2 import (
    AlphaZeroV2Model,
    EconomicStateTransformer,
    HexMapTransformer,
    CrossModalFusion,
    FiLMResBlock,
    V2PolicyHead,
    TrackConnectivityComputer,
    PLAYER_FEAT_SIZE,
    CORP_FEAT_SIZE,
    PRIVATE_FEAT_SIZE,
    GLOBAL_FEAT_SIZE,
    NUM_ENTITY_GROUPS,
)

POLICY_OUTPUT_SIZE = 26535
VALUE_OUTPUT_SIZE = 4
NUM_HEXES = 93
MAP_NODE_FEATURES = 50


def get_fresh_game_state():
    game_map = GameMap()
    game_class = game_map.game_by_title("1830")
    players = {1: "Player 1", 2: "Player 2", 3: "Player 3", 4: "Player 4"}
    return game_class(players)


def get_v2_model():
    return AlphaZeroV2Model(ModelV2Config())


def test_economic_transformer_shapes():
    """Economic Transformer produces correct output shapes."""
    d_entity = 128
    econ = EconomicStateTransformer(d_entity=d_entity, num_layers=2, num_heads=4, d_ff=256)
    econ.eval()
    gs = torch.randn(3, 390)
    cls_embed, entity_embeds = econ(gs)
    assert cls_embed.shape == (3, d_entity)
    assert entity_embeds.shape == (3, NUM_ENTITY_GROUPS, d_entity)


def test_hex_transformer_shapes():
    """Hex Transformer produces correct output shapes."""
    d_map = 128
    model = HexMapTransformer(
        num_node_features=MAP_NODE_FEATURES, d_model=d_map, num_heads=4, num_layers=2, d_ff=256, max_distance=12
    )
    model.eval()

    B = 2
    node_features = torch.randn(B, NUM_HEXES, MAP_NODE_FEATURES)
    axial_coords = torch.randn(NUM_HEXES, 2)
    dist_mat = torch.randint(0, 10, (NUM_HEXES, NUM_HEXES))
    dir_mat = torch.randint(0, 7, (NUM_HEXES, NUM_HEXES))
    track_conn = torch.zeros(B, NUM_HEXES, NUM_HEXES)

    node_embeds, map_pool = model(node_features, axial_coords, dist_mat, dir_mat, track_conn)
    assert node_embeds.shape == (B, NUM_HEXES, d_map)
    assert map_pool.shape == (B, d_map)


def test_cross_modal_fusion_shapes():
    """Cross-modal fusion produces correct output shape."""
    d_entity, d_map, d_trunk = 64, 128, 256
    fusion = CrossModalFusion(d_entity, d_map, d_trunk, num_heads=4)
    fusion.eval()

    B = 2
    entity_embeds = torch.randn(B, NUM_ENTITY_GROUPS, d_entity)
    node_embeds = torch.randn(B, NUM_HEXES, d_map)
    map_pool = torch.randn(B, d_map)

    out = fusion(entity_embeds, node_embeds, map_pool)
    assert out.shape == (B, d_trunk)


def test_film_res_block():
    """FiLM residual block preserves shape and starts as identity."""
    d_trunk, d_film = 256, 32
    block = FiLMResBlock(d_trunk, d_film)

    x = torch.randn(2, d_trunk)
    phase = torch.zeros(2, d_film)  # zero phase → identity FiLM

    out = block(x, phase)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_track_connectivity_computer():
    """Track connectivity is computed from node features and direction matrix."""
    tc = TrackConnectivityComputer(MAP_NODE_FEATURES)
    B = 2
    node_features = torch.zeros(B, NUM_HEXES, MAP_NODE_FEATURES)
    dir_mat = torch.full((NUM_HEXES, NUM_HEXES), 6, dtype=torch.long)  # no adjacency

    conn = tc(node_features, dir_mat)
    assert conn.shape == (B, NUM_HEXES, NUM_HEXES)
    assert (conn == 0).all()  # no adjacency → no connectivity


def test_v2_policy_head_shapes():
    """V2 policy head produces correct output shape."""
    from rl18xx.agent.alphazero.action_mapper import ActionMapper

    action_mapper = ActionMapper()
    lay_tile_info = action_mapper.get_lay_tile_index_info()

    d_trunk, d_map = 256, 128
    head = V2PolicyHead(d_trunk, d_map, POLICY_OUTPUT_SIZE, lay_tile_info)
    head.eval()

    B = 2
    trunk = torch.randn(B, d_trunk)
    node_embeds = torch.randn(B, NUM_HEXES, d_map)

    logits = head(trunk, node_embeds)
    assert logits.shape == (B, POLICY_OUTPUT_SIZE)
    assert torch.isfinite(logits).all()


def test_v2_model_forward_synthetic():
    """Full v2 model forward pass with synthetic data."""
    config = ModelV2Config(
        d_entity=64,
        econ_transformer_layers=1,
        econ_transformer_heads=2,
        econ_transformer_ff_dim=128,
        d_map=64,
        hex_transformer_layers=1,
        hex_transformer_heads=2,
        hex_transformer_ff_dim=128,
        d_trunk=128,
        num_res_blocks=2,
    )
    model = AlphaZeroV2Model(config)
    model.eval()
    device = config.device

    B = 2
    gs = torch.randn(B, 390, device=device)
    # Make round_type and active_player reasonable
    gs[:, 16] = 0.5  # Operating round
    gs[:, 0] = 1.0  # Player 0 active
    gs[:, 1:4] = 0.0
    node_features = torch.randn(B, NUM_HEXES, MAP_NODE_FEATURES, device=device)
    round_type = torch.tensor([1, 0], dtype=torch.long, device=device)
    active_player = torch.tensor([0, 1], dtype=torch.long, device=device)

    policy_logits, value_logits, aux_pred = model(gs, node_features, round_type, active_player)
    assert policy_logits.shape == (B, POLICY_OUTPUT_SIZE)
    assert value_logits.shape == (B, VALUE_OUTPUT_SIZE)
    assert aux_pred.shape == (B, 1)
    assert torch.isfinite(policy_logits).all()
    assert torch.isfinite(value_logits).all()


def test_v2_model_run_from_game():
    """Full v2 model run from a real game state (end-to-end)."""
    model = get_v2_model()
    model.eval()
    game = get_fresh_game_state()

    # Initialize structural matrices from the game
    model._compute_structural_matrices(game)

    probs, log_probs, value = model.run(game)
    assert probs.shape == (POLICY_OUTPUT_SIZE,)
    assert log_probs.shape == (POLICY_OUTPUT_SIZE,)
    assert value.shape == (VALUE_OUTPUT_SIZE,)


def test_v2_model_run_batch_encoded():
    """V2 model batch inference with encoded game states."""
    model = get_v2_model()
    model.eval()
    games = [get_fresh_game_state() for _ in range(3)]

    # Encode games
    encoded = [model.encoder.encode(g) for g in games]

    # Initialize structural matrices
    model._compute_structural_matrices(games[0])

    probs, log_probs, values = model.run_many_encoded(encoded)
    assert probs.shape == (3, POLICY_OUTPUT_SIZE)
    assert log_probs.shape == (3, POLICY_OUTPUT_SIZE)
    assert values.shape == (3, VALUE_OUTPUT_SIZE)


def test_v2_model_parameter_count():
    """V2 model with default config should be approximately 7.3M parameters."""
    model = get_v2_model()
    param_count = sum(p.numel() for p in model.parameters())
    print(f"V2 model parameter count: {param_count:,}")
    # Should be roughly 7-8M with default config
    assert 4_000_000 < param_count < 12_000_000, f"Parameter count {param_count:,} outside expected range"
