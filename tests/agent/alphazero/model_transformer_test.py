import torch
from rl18xx.game.gamemap import GameMap
from rl18xx.agent.alphazero.config import ModelTransformerConfig
from rl18xx.agent.alphazero.model_transformer import (
    AlphaZeroTransformerModel,
    EconomicStateTransformer,
    HexMapTransformer,
    HexResNetMapEncoder,
    HexTransformerMapEncoder,
    CrossModalFusion,
    FiLMResBlock,
    HierarchicalPolicyHead,
    TrackConnectivityComputer,
    PLAYER_FEAT_SIZE,
    CORP_FEAT_SIZE,
    PRIVATE_FEAT_SIZE,
    GLOBAL_FEAT_SIZE,
    NUM_ENTITY_GROUPS,
)
from rl18xx.agent.alphazero.encoder import HEX_COORDS_ORDERED

POLICY_OUTPUT_SIZE = 26537
# Single-checkpoint multi-N: value head emits ``max_players`` logits regardless
# of the actual game's player count. Loss masking covers padded slots.
VALUE_OUTPUT_SIZE = 6
NUM_HEXES = 93
MAP_NODE_FEATURES = 50
# Flat game-state vector size for the model's max-N layout (max_players=6).
# Encoder emits shorter games padded to this layout via the model's
# ``_pad_state_to_max_players`` helper.
GAME_STATE_SIZE = 442


def get_fresh_game_state():
    game_map = GameMap()
    game_class = game_map.game_by_title("1830")
    players = {1: "Player 1", 2: "Player 2", 3: "Player 3", 4: "Player 4"}
    return game_class(players)


def get_transformer_model():
    return AlphaZeroTransformerModel(ModelTransformerConfig())


def test_economic_transformer_shapes():
    """Economic Transformer produces correct output shapes."""
    d_entity = 128
    econ = EconomicStateTransformer(d_entity=d_entity, num_layers=2, num_heads=4, d_ff=256)
    econ.eval()
    # State vector sized for the module's ``max_players`` layout (default 6).
    gs = torch.randn(3, GAME_STATE_SIZE)
    entity_summary, entity_embeds, key_padding_mask = econ(gs)
    assert entity_summary.shape == (3, d_entity)
    assert entity_embeds.shape == (3, NUM_ENTITY_GROUPS, d_entity)
    # Default (no per-sample ``num_players``) treats every sample as fully
    # populated, so the padding mask is all-False.
    assert key_padding_mask.shape == (3, NUM_ENTITY_GROUPS)
    assert not key_padding_mask.any()


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


def test_hex_transformer_map_encoder_alias():
    """``HexTransformerMapEncoder`` is the doc-canonical name for ``HexMapTransformer``."""
    assert HexTransformerMapEncoder is HexMapTransformer


def test_resnet_map_encoder_shapes():
    """Hex ResNet map encoder produces correct (per_hex, map_pool) shapes."""
    d_map = 128
    channels = 64
    model = HexResNetMapEncoder(
        num_node_features=MAP_NODE_FEATURES,
        d_map=d_map,
        num_layers=4,
        channels=channels,
        hex_coords=HEX_COORDS_ORDERED,
    )
    model.eval()

    B = 2
    node_features = torch.randn(B, NUM_HEXES, MAP_NODE_FEATURES)

    per_hex, map_pool = model(node_features)
    assert per_hex.shape == (B, NUM_HEXES, d_map)
    assert map_pool.shape == (B, d_map)
    assert torch.isfinite(per_hex).all()
    assert torch.isfinite(map_pool).all()

    # Offset-grid mapping must accommodate all 93 hexes with no collisions.
    assert model.grid_rows * model.grid_cols >= NUM_HEXES
    assert int(model.grid_mask.sum().item()) == NUM_HEXES


def test_resnet_map_encoder_in_model():
    """Full model with ``map_encoder='resnet'`` runs end-to-end."""
    config = ModelTransformerConfig(
        map_encoder="resnet",
        d_entity=64,
        econ_transformer_layers=1,
        econ_transformer_heads=2,
        econ_transformer_ff_dim=128,
        d_map=64,
        resnet_channels=64,
        resnet_layers=4,
        d_trunk=128,
        num_res_blocks=2,
    )
    model = AlphaZeroTransformerModel(config)
    model.eval()
    device = config.device

    B = 2
    gs = torch.randn(B, GAME_STATE_SIZE, device=device)
    gs[:, 16] = 0.5
    gs[:, 0] = 1.0
    gs[:, 1:4] = 0.0
    node_features = torch.randn(B, NUM_HEXES, MAP_NODE_FEATURES, device=device)
    round_type = torch.tensor([1, 0], dtype=torch.long, device=device)
    active_player = torch.tensor([0, 1], dtype=torch.long, device=device)

    policy_logits, win_loss_logits, score_pred, aux_pred = model(
        gs, node_features, round_type, active_player
    )
    assert policy_logits.shape == (B, POLICY_OUTPUT_SIZE)
    assert win_loss_logits.shape == (B, VALUE_OUTPUT_SIZE)
    assert score_pred.shape == (B, VALUE_OUTPUT_SIZE)
    assert aux_pred.shape == (B, 1)
    assert torch.isfinite(policy_logits).all()
    assert torch.isfinite(win_loss_logits).all()
    assert torch.isfinite(score_pred).all()


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
    head = HierarchicalPolicyHead(d_trunk, d_map, POLICY_OUTPUT_SIZE, lay_tile_info)
    head.eval()

    B = 2
    trunk = torch.randn(B, d_trunk)
    node_embeds = torch.randn(B, NUM_HEXES, d_map)

    logits, components = head(trunk, node_embeds)
    assert logits.shape == (B, POLICY_OUTPUT_SIZE)
    assert torch.isfinite(logits).all()
    # Structured components must agree with the lay_tile_info layout.
    assert components["hex_logits"].shape == (B, lay_tile_info["num_hexes"])
    assert components["tile_logits"].shape == (
        B, lay_tile_info["num_hexes"], lay_tile_info["num_tiles"],
    )
    assert components["rotation_logits"].shape == (
        B, lay_tile_info["num_hexes"], lay_tile_info["num_tiles"], lay_tile_info["num_rotations"],
    )
    # PlaceToken components
    assert components["place_token_hex_logits"].shape == (B, lay_tile_info["num_hexes"])
    assert components["place_token_slot_logits"].shape[0] == B


def test_v2_model_forward_synthetic():
    """Full v2 model forward pass with synthetic data."""
    config = ModelTransformerConfig(
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
    model = AlphaZeroTransformerModel(config)
    model.eval()
    device = config.device

    B = 2
    gs = torch.randn(B, GAME_STATE_SIZE, device=device)
    # Make round_type and active_player reasonable
    gs[:, 16] = 0.5  # Operating round
    gs[:, 0] = 1.0  # Player 0 active
    gs[:, 1:4] = 0.0
    node_features = torch.randn(B, NUM_HEXES, MAP_NODE_FEATURES, device=device)
    round_type = torch.tensor([1, 0], dtype=torch.long, device=device)
    active_player = torch.tensor([0, 1], dtype=torch.long, device=device)

    policy_logits, win_loss_logits, score_pred, aux_pred = model(
        gs, node_features, round_type, active_player
    )
    assert policy_logits.shape == (B, POLICY_OUTPUT_SIZE)
    assert win_loss_logits.shape == (B, VALUE_OUTPUT_SIZE)
    assert score_pred.shape == (B, VALUE_OUTPUT_SIZE)
    assert aux_pred.shape == (B, 1)
    assert torch.isfinite(policy_logits).all()
    assert torch.isfinite(win_loss_logits).all()
    assert torch.isfinite(score_pred).all()


def test_v2_model_run_from_game():
    """Full v2 model run from a real game state (end-to-end)."""
    model = get_transformer_model()
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
    model = get_transformer_model()
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
    model = get_transformer_model()
    param_count = sum(p.numel() for p in model.parameters())
    print(f"V2 model parameter count: {param_count:,}")
    # Should be roughly 7-8M with default config
    assert 4_000_000 < param_count < 12_000_000, f"Parameter count {param_count:,} outside expected range"


# ---------------------------------------------------------------------------
# Variable-N padding tests (encoder emits shorter games; model pads to max_players)
# ---------------------------------------------------------------------------


def _get_game_state_for_players(n: int):
    """Construct a fresh ``BaseGame`` with ``n`` players."""
    game_map = GameMap()
    game_class = game_map.game_by_title("1830")
    players = {i + 1: f"Player {i + 1}" for i in range(n)}
    return game_class(players)


def test_encoder_pad_to_max_players_3_player_to_6_layout():
    """Encoder emits a shorter state vector for a 3-player game; the model's
    ``_pad_state_to_max_players`` helper must remap it into the 442-dim
    max-N (max_players=6) layout."""
    from rl18xx.agent.alphazero.encoder import Encoder_1830Graph
    from rl18xx.agent.alphazero.model_transformer import _pad_state_to_max_players

    # 3-player layout size
    _, size_3 = Encoder_1830Graph.compute_section_layout(3)
    # 6-player (max-N) layout size
    _, size_6 = Encoder_1830Graph.compute_section_layout(6)
    assert size_6 == GAME_STATE_SIZE

    # Build a synthetic state vector at the 3-player size and pad it.
    state_3p = torch.arange(size_3, dtype=torch.float32)
    padded = _pad_state_to_max_players(state_3p, num_players=3, max_players=6)
    assert padded.shape == (size_6,), (
        f"Padded state must match max-N layout {size_6}; got {padded.shape}"
    )


def test_encoder_pad_no_op_when_num_players_equals_max():
    """When num_players == max_players the padding helper must be a no-op
    (returns the input identically) — no copy, no remap."""
    from rl18xx.agent.alphazero.model_transformer import _pad_state_to_max_players

    state_6p = torch.randn(GAME_STATE_SIZE)
    out = _pad_state_to_max_players(state_6p, num_players=6, max_players=6)
    assert out is state_6p


def test_econ_transformer_padding_mask_blanks_padded_slots():
    """When the EconomicStateTransformer is given a 3-player game padded to
    a 6-player layout, the key-padding mask must mark slots 3-5 as padded
    AND the output for those slots must be zero (padded entries are blanked
    after the transformer)."""
    d_entity = 64
    econ = EconomicStateTransformer(d_entity=d_entity, num_layers=1, num_heads=2, d_ff=128)
    econ.eval()

    B = 2
    gs = torch.randn(B, GAME_STATE_SIZE)
    # First sample: 3 real players; second: 4 real players.
    num_players = torch.tensor([3, 4], dtype=torch.long)

    entity_summary, entity_embeds, key_padding_mask = econ(gs, num_players=num_players)
    assert key_padding_mask.shape == (B, NUM_ENTITY_GROUPS)

    # Player slots 3..max_players-1 must be padded for sample 0 (3 real players).
    # Slots 4..max_players-1 must be padded for sample 1 (4 real players).
    for b, n in enumerate(num_players.tolist()):
        for slot in range(VALUE_OUTPUT_SIZE):  # max_players==VALUE_OUTPUT_SIZE
            expected_padded = slot >= n
            assert bool(key_padding_mask[b, slot]) == expected_padded, (
                f"sample {b}, slot {slot}: padded={key_padding_mask[b, slot]} "
                f"vs expected={expected_padded} (n={n})"
            )
        # Padded player rows must be zero post-attention.
        for slot in range(n, VALUE_OUTPUT_SIZE):
            assert torch.allclose(entity_embeds[b, slot], torch.zeros(d_entity)), (
                f"padded slot {slot} of sample {b} should be zero, got {entity_embeds[b, slot]}"
            )
        # Real player rows should not be uniformly zero (something non-trivial happened).
        any_nonzero = any(entity_embeds[b, slot].abs().sum().item() > 0 for slot in range(n))
        assert any_nonzero, f"All real player rows of sample {b} are zero — attention is dead"


def test_model_end_to_end_with_3_player_game_matches_max_layout():
    """Full model forward from a real 3-player game state. The encoder emits a
    smaller state and the model pads it internally; output shapes still match
    the max-N convention (policy: 26537, value: 6)."""
    model = get_transformer_model()
    model.eval()
    game = _get_game_state_for_players(3)

    model._compute_structural_matrices(game)

    probs, log_probs, value = model.run(game)
    assert probs.shape == (POLICY_OUTPUT_SIZE,)
    assert log_probs.shape == (POLICY_OUTPUT_SIZE,)
    # Value head always emits VALUE_OUTPUT_SIZE=6 logits regardless of
    # actual player count (padded slots are masked at loss time).
    assert value.shape == (VALUE_OUTPUT_SIZE,)


def test_model_batch_mixed_player_counts():
    """Batch inference with mixed player counts (3p and 4p in the same batch);
    each game's encoded state has a different raw size but the model pads
    them all into the same max-N layout."""
    model = get_transformer_model()
    model.eval()

    games = [
        _get_game_state_for_players(3),
        _get_game_state_for_players(4),
        _get_game_state_for_players(6),
    ]
    encoded = [model.encoder.encode(g) for g in games]

    # Encoded states should have different sizes per the variable-N encoder.
    state_sizes = {e[0].numel() for e in encoded}
    assert len(state_sizes) > 1, (
        f"Expected mixed encoded sizes for 3p/4p/6p games; got {state_sizes}"
    )

    model._compute_structural_matrices(games[0])
    probs, log_probs, values = model.run_many_encoded(encoded)
    assert probs.shape == (3, POLICY_OUTPUT_SIZE)
    assert values.shape == (3, VALUE_OUTPUT_SIZE)
    assert torch.isfinite(values).all()
