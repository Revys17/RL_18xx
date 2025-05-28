import pytest
import numpy as np
from torch import float32, long
from typing import Tuple
import torch

# Assuming the encoder and game engine classes are importable
from rl18xx.agent.alphazero.encoder import (
    G1830_STARTING_CASH,
    Encoder_1830,
    ROUND_TYPE_MAP,
    MAX_ROUND_TYPE_IDX,
    PHASE_NAMES_ORDERED,
    G1830_BANK_CASH,
    G1830_TRAIN_COUNT,
    CORPORATION_IDS_ORDERED,
    MAX_SHARE_PRICE,
)
from rl18xx.game.engine.game.base import BaseGame
from rl18xx.game.engine.game.title.g1830 import Game as Game_1830, Map as Map_1830
from rl18xx.game.engine.round import WaterfallAuction, Stock
from rl18xx.game.engine.core import Phase
from rl18xx.game.gamemap import GameMap
from rl18xx.game.action_helper import ActionHelper


# Helper function for float comparisons
def assert_float(expected, actual, desc="", tolerance=1e-4):
    assert abs(actual - expected) < tolerance, f"{desc}: Expected {expected:.4f}, got {actual:.4f}"


# Helper to get section slice and offset
def get_section_slice(encoder: Encoder_1830, section_name: str, current_offset: int) -> tuple[slice, int]:
    size = encoder._get_section_size(section_name)
    section_slice = slice(current_offset, current_offset + size)
    next_offset = current_offset + size
    return section_slice, next_offset


# Helper to get map feature index by name
def get_feature_index(encoder: Encoder_1830, feature_name: str) -> int:
    try:
        return encoder.map_node_features.index(feature_name)
    except ValueError:
        raise ValueError(
            f"Feature '{feature_name}' not found in encoder.map_node_features: {encoder.map_node_features}"
        )


@pytest.fixture(scope="module")
def encoder_1830():
    """Provides a reusable Encoder_1830 instance."""
    return Encoder_1830()


@pytest.fixture
def test_game_1830_4p():
    """Provides a fresh 4-player 1830 game instance at the start of the private auction."""
    game_map = GameMap()
    game_class = game_map.game_by_title("1830")
    players = {"1": "Player 1", "2": "Player 2", "3": "Player 3", "4": "Player 4"}
    game_instance = game_class(players)
    return game_instance, ActionHelper(game_instance)


@pytest.fixture
def operating_round_2_game_state(test_game_1830_4p):
    test_game_1830_4p, action_helper = test_game_1830_4p
    # Auction
    test_game_1830_4p.process_action(action_helper.get_all_choices()[-1])  # pass
    test_game_1830_4p.process_action(action_helper.get_all_choices()[1])  # bid 45 on CS
    test_game_1830_4p.process_action(action_helper.get_all_choices()[1])  # bid 50 on CS
    test_game_1830_4p.process_action(action_helper.get_all_choices()[-77])  # bid 225 on BO
    test_game_1830_4p.process_action(action_helper.get_all_choices()[0])  # buy SV
    test_game_1830_4p.process_action(action_helper.get_all_choices()[-1])  # pass
    test_game_1830_4p.process_action(action_helper.get_all_choices()[0])  # buy DH
    test_game_1830_4p.process_action(action_helper.get_all_choices()[0])  # buy MH
    test_game_1830_4p.process_action(action_helper.get_all_choices()[0])  # buy CA
    test_game_1830_4p.process_action(action_helper.get_all_choices()[0])  # Par B&O at 100
    # SR 1
    test_game_1830_4p.process_action(action_helper.get_all_choices()[-2])  # Par PRR
    test_game_1830_4p.process_action(action_helper.get_all_choices()[-1])  # Pass
    test_game_1830_4p.process_action(action_helper.get_all_choices()[-8])  # Par NYC
    test_game_1830_4p.process_action(action_helper.get_all_choices()[1])  # Buy PRR
    test_game_1830_4p.process_action(action_helper.get_all_choices()[1])  # Buy PRR
    test_game_1830_4p.process_action(action_helper.get_all_choices()[14])  # Par C&O
    test_game_1830_4p.process_action(action_helper.get_all_choices()[2])  # Buy NYC
    test_game_1830_4p.process_action(action_helper.get_all_choices()[1])  # Buy PRR
    test_game_1830_4p.process_action(action_helper.get_all_choices()[1])  # Buy PRR
    test_game_1830_4p.process_action(action_helper.get_all_choices()[3])  # Buy C&O
    test_game_1830_4p.process_action(action_helper.get_all_choices()[2])  # Buy NYC
    test_game_1830_4p.process_action(action_helper.get_all_choices()[0])  # Buy PRR
    test_game_1830_4p.process_action(action_helper.get_all_choices()[1])  # Buy PRR
    test_game_1830_4p.process_action(action_helper.get_all_choices()[3])  # Buy C&O
    test_game_1830_4p.process_action(action_helper.get_all_choices()[2])  # Buy NYC
    test_game_1830_4p.process_action(action_helper.get_all_choices()[1])  # Buy PRR
    test_game_1830_4p.process_action(action_helper.get_all_choices()[2])  # Buy C&O
    test_game_1830_4p.process_action(action_helper.get_all_choices()[1])  # Buy NYC
    test_game_1830_4p.process_action(action_helper.get_all_choices()[1])  # Buy NYC
    test_game_1830_4p.process_action(action_helper.get_all_choices()[2])  # Buy C&O
    test_game_1830_4p.process_action(action_helper.get_all_choices()[1])  # Buy NYC
    test_game_1830_4p.process_action(action_helper.get_all_choices()[1])  # Buy NYC
    # OR 1
    test_game_1830_4p.process_action(action_helper.get_all_choices()[0])  # lays tile #57 with rotation 1 on H10
    test_game_1830_4p.process_action(action_helper.get_all_choices()[-1])  # passes place token
    test_game_1830_4p.process_action(action_helper.get_all_choices()[0])  # buys a 2 train
    test_game_1830_4p.process_action(action_helper.get_all_choices()[0])  # buys a 2 train
    test_game_1830_4p.process_action(action_helper.get_all_choices()[-1])  # passes trains
    test_game_1830_4p.process_action(action_helper.get_all_choices()[0])  # lays tile #57 with rotation 0 on E19
    test_game_1830_4p.process_action(action_helper.get_all_choices()[0])  # buys a 2 train
    test_game_1830_4p.process_action(action_helper.get_all_choices()[-1])  # passes trains
    test_game_1830_4p.process_action(action_helper.get_all_choices()[2])
    test_game_1830_4p.process_action(action_helper.get_all_choices()[0])  # Buys a 2 train
    test_game_1830_4p.process_action(action_helper.get_all_choices()[-1])  # passes trains
    # SR 2
    test_game_1830_4p.process_action(action_helper.get_all_choices()[-2])  # sell 50% nyc
    test_game_1830_4p.process_action(action_helper.get_all_choices()[-3])  # par nynh at 71
    test_game_1830_4p.process_action(action_helper.get_all_choices()[0])  # buy C&O
    test_game_1830_4p.process_action(action_helper.get_all_choices()[-1])  # pass sell
    test_game_1830_4p.process_action(action_helper.get_all_choices()[0])  # Buy NYC
    test_game_1830_4p.process_action(action_helper.get_all_choices()[-1])  # pass sell
    test_game_1830_4p.process_action(action_helper.get_all_choices()[0])  # Buy NYNH
    test_game_1830_4p.process_action(action_helper.get_all_choices()[-1])  # pass sell
    test_game_1830_4p.process_action(action_helper.get_all_choices()[1])  # Buy NYNH
    test_game_1830_4p.process_action(action_helper.get_all_choices()[-1])  # pass sell
    test_game_1830_4p.process_action(action_helper.get_all_choices()[-1])  # pass
    test_game_1830_4p.process_action(action_helper.get_all_choices()[-1])  # pass
    test_game_1830_4p.process_action(action_helper.get_all_choices()[-1])  # pass
    test_game_1830_4p.process_action(action_helper.get_all_choices()[1])  # Buy NYNH
    test_game_1830_4p.process_action(action_helper.get_all_choices()[-1])  # pass sell
    test_game_1830_4p.process_action(action_helper.get_all_choices()[-1])  # pass
    test_game_1830_4p.process_action(action_helper.get_all_choices()[-1])  # pass
    test_game_1830_4p.process_action(action_helper.get_all_choices()[-1])  # pass
    test_game_1830_4p.process_action(action_helper.get_all_choices()[1])  # Buy NYNH
    test_game_1830_4p.process_action(action_helper.get_all_choices()[-1])  # pass sell
    test_game_1830_4p.process_action(action_helper.get_all_choices()[-1])  # pass
    test_game_1830_4p.process_action(action_helper.get_all_choices()[-1])  # pass
    test_game_1830_4p.process_action(action_helper.get_all_choices()[-1])  # pass
    test_game_1830_4p.process_action(action_helper.get_all_choices()[-1])  # pass
    return (test_game_1830_4p, action_helper)


# --- Helper function for specific edge checks (optional, for the USER ACTION section) ---
def check_edge_exists_in_tensor(idx1: int, idx2: int, edge_index_tensor: torch.Tensor) -> bool:
    """Checks if an undirected edge between idx1 and idx2 exists in the edge_index tensor."""
    for i in range(edge_index_tensor.shape[1]):
        col = edge_index_tensor[:, i]
        u, v = col[0].item(), col[1].item()
        if (u == idx1 and v == idx2) or (u == idx2 and v == idx1):
            return True
    return False


def test_edge_index(encoder_1830: Encoder_1830, test_game_1830_4p):
    test_game_1830_4p, action_helper = test_game_1830_4p

    # Get and check the edge_index created on the initial game state:
    # Call the method to be tested
    # If this method is part of __init__, this test might need refactoring
    # to check the state after encoder initialization.
    # Assuming it's a distinct method callable after __init__.
    encoder_1830._precompute_adjacency(test_game_1830_4p)

    # Basic assertions about the output
    assert hasattr(encoder_1830, 'base_edge_index'), "edge_index attribute was not set on encoder_instance"
    assert hasattr(encoder_1830, 'base_edge_attributes'), "edge_index attribute was not set on encoder_instance"
    assert encoder_1830.base_edge_index is not None, "base_edge_index was not set (is None)"
    assert encoder_1830.base_edge_attributes is not None, "base_edge_attributes was not set (is None)"
    assert isinstance(encoder_1830.base_edge_index, torch.Tensor), "base_edge_index is not a torch.Tensor"
    assert isinstance(encoder_1830.base_edge_attributes, torch.Tensor), "base_edge_attributes is not a torch.Tensor"
    assert encoder_1830.base_edge_index.dtype == torch.long, "base_edge_index dtype is not torch.long"
    assert encoder_1830.base_edge_attributes.dtype == torch.long, "base_edge_attributes dtype is not torch.long"
    assert encoder_1830.base_edge_index.ndim == 2, "base_edge_index should be 2-dimensional"
    assert encoder_1830.base_edge_attributes.ndim == 1, "base_edge_attributes should be 1-dimensional"
    assert encoder_1830.base_edge_index.shape[0] == 2, "base_edge_index should have 3 rows (source, target, direction)"
    assert encoder_1830.base_edge_index.shape[1] == 470, "base_edge_index should have 470 columns"
    assert encoder_1830.base_edge_attributes.shape[0] == 470, "base_edge_attributes should have 470 elements"

    # Spot check adjacencies
    adjacency_list = {
        "B10": ["A9", "A11", "B12", "C11", "C9"],
        "E7": ["E5", "D6", "D8", "E9", "F8", "F6"],
        "E15": ["E13", "D14", "D16", "E17", "F16", "F14"],
        "G19": ["G17", "H18", "F18", "F20"],
        "K13": ["J12", "J14", "K15"],
        "J2": ["I1", "I3", "J4"],
        "D24": ["C23", "D22", "E23"],
    }

    for hex_id1, neighbor_list in adjacency_list.items():
        for hex_id2 in encoder_1830.hex_coord_to_idx.keys():
            idx1 = encoder_1830.hex_coord_to_idx[hex_id1]
            idx2 = encoder_1830.hex_coord_to_idx[hex_id2]
            are_neighbors = check_edge_exists_in_tensor(idx1, idx2, encoder_1830.base_edge_index)

            if are_neighbors and hex_id2 not in neighbor_list:
                raise ValueError(f"Edge exists between {hex_id1} and {hex_id2}, but {hex_id2} is not in {neighbor_list}")
            elif not are_neighbors and hex_id2 in neighbor_list:
                raise ValueError(f"Edge does not exist between {hex_id1} and {hex_id2}, but {hex_id2} is in {neighbor_list}")


def test_initial_encoding_structure(encoder_1830: Encoder_1830, test_game_1830_4p):
    """Verify the basic structure and size of the initial encoding."""
    g, action_helper = test_game_1830_4p
    num_players = len(g.players)
    assert num_players == 4, "Test requires a 4-player game"

    game_state_tensor, node_features_tensor, edge_index, edge_attributes = encoder_1830.encode(g)

    expected_size = encoder_1830.GAME_STATE_ENCODING_SIZE
    assert game_state_tensor.shape == (
        1,
        expected_size,
    ), f"Expected shape (1, {expected_size}), got {game_state_tensor.shape}"
    assert game_state_tensor.dtype == float32, "Expected dtype float32"

    expected_node_features_size = (encoder_1830.NUM_HEXES, encoder_1830.num_map_node_features)
    assert (
        node_features_tensor.shape == expected_node_features_size
    ), f"Expected shape {expected_node_features_size}, got {node_features_tensor.shape}"
    assert node_features_tensor.dtype == float32, "Expected dtype float32"

    expected_edge_index_size = (2, 470)
    assert (
        edge_index.shape == expected_edge_index_size
    ), f"Expected shape {expected_edge_index_size}, got {edge_index.shape}"
    assert edge_index.dtype == long, "Expected dtype long"

    expected_edge_attributes_size = (470,)
    assert (
        edge_attributes.shape == expected_edge_attributes_size
    ), f"Expected shape {expected_edge_attributes_size}, got {edge_attributes.shape}"
    assert edge_attributes.dtype == long, "Expected dtype long"

def test_initial_game_state_encoding(encoder_1830: Encoder_1830, test_game_1830_4p):
    """Verify the encoding of general game state at the start."""
    g, action_helper = test_game_1830_4p
    game_state_tensor, node_features_tensor, edge_index, edge_attributes = encoder_1830.encode(g)
    encoding = game_state_tensor.squeeze(0).numpy()  # Work with numpy array

    offset = 0
    player_map = encoder_1830.player_id_to_idx
    corp_map = encoder_1830.corp_id_to_idx
    private_map = encoder_1830.private_id_to_idx
    num_players = encoder_1830.num_players
    num_corps = encoder_1830.NUM_CORPORATIONS
    num_privates = encoder_1830.NUM_PRIVATES
    num_phases = encoder_1830.NUM_PHASES
    # --- Section 1: Active Entity ---
    s_slice, offset = get_section_slice(encoder_1830, "active_entity", offset)
    active_player_id = g.round.active_entities[0].id
    active_idx = player_map[active_player_id]
    expected = np.zeros(num_players + num_corps)
    expected[active_idx] = 1.0
    np.testing.assert_allclose(expected, encoding[s_slice], atol=1e-6, err_msg="Section 1: Active Entity")

    # --- Section 2: Active President ---
    s_slice, offset = get_section_slice(encoder_1830, "active_president", offset)
    expected = np.zeros(num_players)
    np.testing.assert_allclose(expected, encoding[s_slice], atol=1e-6, err_msg="Section 2: Active President")

    # --- Section 3: Round Type ---
    s_slice, offset = get_section_slice(encoder_1830, "round_type", offset)
    expected_round_idx = float(ROUND_TYPE_MAP.get(type(g.round).__name__, -1.0)) / MAX_ROUND_TYPE_IDX
    assert expected_round_idx != -1.0, "Initial round type not found in map"
    assert_float(expected_round_idx, encoding[s_slice].item(), "Section 3: Round Type")

    # --- Section 4: Game Phase ---
    s_slice, offset = get_section_slice(encoder_1830, "game_phase", offset)
    phase_name = g.phase.name
    expected_phase_idx = PHASE_NAMES_ORDERED.index(phase_name) / (num_phases - 1)
    assert_float(expected_phase_idx, encoding[s_slice].item(), "Section 4: Game Phase")

    # --- Section 5: Priority Deal Player ---
    s_slice, offset = get_section_slice(encoder_1830, "priority_deal_player", offset)
    priority_player_id = g.priority_deal_player().id
    priority_idx = player_map[priority_player_id]
    expected = np.zeros(num_players)
    expected[priority_idx] = 1.0
    np.testing.assert_allclose(encoding[s_slice], expected, atol=1e-6, err_msg="Section 5: Priority Deal")

    # --- Section 6: Bank Cash ---
    s_slice, offset = get_section_slice(encoder_1830, "bank_cash", offset)
    expected = float(g.bank.cash) / G1830_BANK_CASH
    assert_float(expected, encoding[s_slice].item(), "Section 6: Bank Cash")

    # --- Section 7: Player Certs Remaining ---
    s_slice, offset = get_section_slice(encoder_1830, "player_certs_remaining", offset)
    expected = np.ones(num_players)
    np.testing.assert_allclose(encoding[s_slice], expected, atol=1e-6, err_msg="Section 7: Player Certs Remaining")

    # --- Section 8: Player Cash ---
    s_slice, offset = get_section_slice(encoder_1830, "player_cash", offset)
    start_cash = G1830_STARTING_CASH[num_players]
    expected = np.array([p.cash / start_cash for p in g.players])
    np.testing.assert_allclose(encoding[s_slice], expected, atol=1e-6, err_msg="Section 8: Player Cash")
    assert_float(encoding[s_slice][0], 1.0, "Section 8: Player 1 Cash Initial")  # Specific check

    # --- Section 9: Player Shares ---
    s_slice, offset = get_section_slice(encoder_1830, "player_shares", offset)
    expected = np.zeros(num_players * num_corps)  # No shares at start
    np.testing.assert_allclose(encoding[s_slice], expected, atol=1e-6, err_msg="Section 9: Player Shares")

    # --- Section 10: Player Privates ---
    s_slice, offset = get_section_slice(encoder_1830, "private_ownership", offset)
    expected = np.zeros((num_players + num_corps) * num_privates)  # No privates owned at start
    np.testing.assert_allclose(encoding[s_slice], expected, atol=1e-6, err_msg="Section 10: Player Privates")

    # --- Section 11: Private Revenue ---
    s_slice, offset = get_section_slice(encoder_1830, "private_revenue", offset)
    max_rev = max(comp.revenue for comp in g.companies) if g.companies else 1  # Avoid div by zero
    expected = np.zeros(num_privates)
    for priv_id, priv_idx in private_map.items():
        comp = g.company_by_id(priv_id)
        expected[priv_idx] = float(comp.revenue) / max_rev
    np.testing.assert_allclose(encoding[s_slice], expected, atol=1e-6, err_msg="Section 11: Private Revenue")

    # --- Section 12: Corp Floated ---
    s_slice, offset = get_section_slice(encoder_1830, "corp_floated", offset)
    expected = np.zeros(num_corps)  # No corps floated at start
    np.testing.assert_allclose(encoding[s_slice], expected, atol=1e-6, err_msg="Section 12: Corp Floated")

    # --- Section 13: Corp Cash ---
    s_slice, offset = get_section_slice(encoder_1830, "corp_cash", offset)
    expected = np.zeros(num_corps)  # No corps have cash at start
    np.testing.assert_allclose(encoding[s_slice], expected, atol=1e-6, err_msg="Section 13: Corp Cash")

    # --- Section 14: Corp Trains ---
    s_slice, offset = get_section_slice(encoder_1830, "corp_trains", offset)
    expected = np.zeros(num_corps * encoder_1830.NUM_TRAIN_TYPES)  # No corps have trains at start
    np.testing.assert_allclose(encoding[s_slice], expected, atol=1e-6, err_msg="Section 14: Corp Trains")

    # --- Section 15: Corp Tokens Remaining ---
    s_slice, offset = get_section_slice(encoder_1830, "corp_tokens_remaining", offset)
    expected = np.zeros(num_corps)
    for corp_id, corp_idx in corp_map.items():
        corp = g.corporation_by_id(corp_id)
        expected[corp_idx] = float(len(corp.unplaced_tokens())) / len(corp.tokens)
    np.testing.assert_allclose(
        encoding[s_slice], expected, atol=1e-6, err_msg="Section 15: Corp Tokens Remaining (Expected 0 before par)"
    )

    # --- Section 16: Corp Share Price ---
    s_slice, offset = get_section_slice(encoder_1830, "corp_share_price", offset)
    expected = np.zeros(num_corps * 2)  # No price before par
    np.testing.assert_allclose(encoding[s_slice], expected, atol=1e-6, err_msg="Section 16: Corp Share Price")

    # --- Section 17: Corp Shares ---
    s_slice, offset = get_section_slice(encoder_1830, "corp_shares", offset)
    expected = np.ones(num_corps * 2)
    # Every other value is 0.0
    expected[1::2] = 0.0
    np.testing.assert_allclose(encoding[s_slice], expected, atol=1e-6, err_msg="Section 17: Corp Shares")

    # --- Section 18: Corp Market Zone ---
    s_slice, offset = get_section_slice(encoder_1830, "corp_market_zone", offset)
    # Assuming -1 indicates off-board/not placed
    expected = np.zeros(num_corps * 4)
    np.testing.assert_allclose(encoding[s_slice], expected, atol=1e-6, err_msg="Section 18: Corp Market Zone")

    # --- Section 19: Depot Trains ---
    s_slice, offset = get_section_slice(encoder_1830, "depot_trains", offset)
    expected = np.zeros(encoder_1830.NUM_TRAIN_TYPES)
    for train_name, train_idx in encoder_1830.train_name_to_idx.items():
        initial_count = G1830_TRAIN_COUNT.get(train_name, 0)
        current_count = sum(1 for t in g.depot.trains if t.name == train_name)
        expected[train_idx] = float(current_count) / initial_count
    np.testing.assert_allclose(encoding[s_slice], expected, atol=1e-6, err_msg="Section 19: Depot Trains")
    np.testing.assert_allclose(
        encoding[s_slice],
        np.ones(encoder_1830.NUM_TRAIN_TYPES),
        atol=1e-6,
        err_msg="Section 21: Depot Trains (All initially 1.0)",
    )

    # --- Section 20: Market Pool Trains ---
    s_slice, offset = get_section_slice(encoder_1830, "market_pool_trains", offset)
    expected = np.zeros(encoder_1830.NUM_TRAIN_TYPES)  # No discarded trains at start
    np.testing.assert_allclose(encoding[s_slice], expected, atol=1e-6, err_msg="Section 20: Market Pool Trains")

    # --- Section 21: Depot Tiles ---
    s_slice, offset = get_section_slice(encoder_1830, "depot_tiles", offset)
    num_initial_tiles = sum(Map_1830.TILES.values())
    assert np.sum(encoding[s_slice]) > 0, "Section 21: Depot Tiles (Should have some tiles)"

    # --- Auction Sections ---
    assert isinstance(g.round.active_step(), WaterfallAuction), "Game should start in WaterfallAuctionRound"
    auction_step = g.round.active_step()

    # --- Section A1: Auction Bids ---
    s_slice, offset = get_section_slice(encoder_1830, "auction_bids", offset)
    expected = np.zeros(num_privates * num_players)  # No bids at start
    np.testing.assert_allclose(encoding[s_slice], expected, atol=1e-6, err_msg="Section A1: Auction Bids")

    # --- Section A2: Auction Min Bid ---
    s_slice, offset = get_section_slice(encoder_1830, "auction_min_bid", offset)
    expected = np.zeros(num_privates)
    start_cash = encoder_1830.starting_cash
    for priv_id, priv_idx in private_map.items():
        company = g.company_by_id(priv_id)
        min_bid = auction_step.min_bid(company)  # Assumes method exists
        expected[priv_idx] = float(min_bid) / start_cash
    np.testing.assert_allclose(encoding[s_slice], expected, atol=1e-4, err_msg="Section A2: Auction Min Bid")

    # --- Section A3: Auction Available ---
    s_slice, offset = get_section_slice(encoder_1830, "auction_available", offset)
    expected = np.zeros(num_privates)
    available_comp = auction_step.companies[0]
    if available_comp:
        expected[private_map[available_comp.id]] = 1.0
    np.testing.assert_allclose(encoding[s_slice], expected, atol=1e-6, err_msg="Section A3: Auction Available")
    # Check SV is available first
    assert_float(encoding[s_slice][private_map["SV"]], 1.0, "Section A3: SV Available")

    # --- Section A4: Auction Face Value ---
    s_slice, offset = get_section_slice(encoder_1830, "auction_face_value", offset)
    expected = np.zeros(num_privates)
    start_cash = encoder_1830.starting_cash
    for priv_id, priv_idx in private_map.items():
        company = g.company_by_id(priv_id)
        expected[priv_idx] = float(company.value) / start_cash
    np.testing.assert_allclose(encoding[s_slice], expected, atol=1e-4, err_msg="Section A4: Auction Face Value")

    # --- Final Offset Check ---
    assert (
        offset == encoder_1830.ENCODING_SIZE
    ), f"Final offset {offset} does not match expected size {encoder_1830.ENCODING_SIZE}"


def test_encoding_after_bid(encoder_1830: Encoder_1830, test_game_1830_4p):
    """Verify encoding changes correctly after a player makes a bid."""
    g, action_helper = test_game_1830_4p

    # Player 1 bids on B&O (assuming B&O is the last private, ID 'BO')
    # Find the bid action for B&O by Player 1
    bo_private = g.company_by_id("BO")  # Check ID
    player1 = g.players[0]
    bid_amount = g.round.active_step().min_bid(bo_private)
    bid_action = None
    for action in action_helper.get_all_choices_limited():
        if action.__class__.__name__ == "Bid":
            if action.company == bo_private and action.price == bid_amount:
                bid_action = action
                break
    assert bid_action is not None, "Could not find bid action for P1 on B&O"

    g.process_action(bid_action)

    game_state_tensor, node_features_tensor, edge_index, edge_attributes = encoder_1830.encode(g)
    encoding = game_state_tensor.squeeze(0).numpy()
    offset = 0
    player_map = encoder_1830.player_id_to_idx
    private_map = encoder_1830.private_id_to_idx
    num_players = encoder_1830.num_players
    num_privates = encoder_1830.NUM_PRIVATES
    start_cash = encoder_1830.starting_cash

    # --- Check Active Player (Section 1) ---
    s_slice, offset = get_section_slice(encoder_1830, "active_entity", offset)
    active_player_id = g.round.active_entities[0].id
    active_idx = player_map[active_player_id]
    assert active_idx == 1, "Active player should be Player 2 (index 1)"
    assert_float(encoding[s_slice][active_idx], 1.0, "Section 1: Player 2 Active after P1 bid")
    assert_float(encoding[s_slice][0], 0.0, "Section 1: Player 1 Inactive after P1 bid")

    # Skip unchanged sections quickly
    offset += sum(
        encoder_1830._get_section_size(name)
        for name in [
            "active_president",
            "round_type",
            "game_phase",
            "priority_deal_player",
            "bank_cash",
            "player_certs_remaining",
            "player_cash",
            "player_shares",
            "private_ownership",
            "private_revenue",
            "corp_floated",
            "corp_cash",
            "corp_trains",
            "corp_tokens_remaining",
            "corp_share_price",
            "corp_shares",
            "corp_market_zone",
            "depot_trains",
            "market_pool_trains",
            "depot_tiles",
        ]
    )

    # --- Check Auction Bids (Section A1) ---
    s_slice, offset = get_section_slice(encoder_1830, "auction_bids", offset)
    bo_idx = private_map["BO"]
    p1_idx = player_map[player1.id]
    expected_bid_norm = float(bid_amount) / start_cash
    # Calculate the flat index for P1's bid on B&O
    bid_flat_idx = bo_idx * num_players + p1_idx
    assert_float(encoding[s_slice][bid_flat_idx], expected_bid_norm, "Section A1: P1 Bid on B&O")
    # Check other bids are still 0
    assert np.sum(encoding[s_slice]) == expected_bid_norm, "Section A1: Only P1 B&O bid should be non-zero"

    # --- Check Auction Min Bid (Section A2) ---
    s_slice, offset = get_section_slice(encoder_1830, "auction_min_bid", offset)
    # Min bid for B&O should increase
    new_min_bid = g.round.active_step().min_bid(bo_private)
    expected_min_bid_norm = float(new_min_bid) / start_cash
    assert_float(encoding[s_slice][bo_idx], expected_min_bid_norm, "Section A2: Min Bid B&O increased")

    # --- Check Auction Available (Section A3) ---
    s_slice, offset = get_section_slice(encoder_1830, "auction_available", offset)
    # Still SV should be available
    sv_idx = private_map["SV"]
    assert_float(encoding[s_slice][sv_idx], 1.0, "Section A3: SV still available")

    # --- Check Auction Face Value (Section A4) ---
    s_slice, offset = get_section_slice(encoder_1830, "auction_face_value", offset)
    # Face values should not change
    expected_fv_norm = float(bo_private.value) / start_cash
    assert_float(encoding[s_slice][bo_idx], expected_fv_norm, "Section A4: Face Value B&O unchanged")

    # --- Final Offset Check ---
    assert offset == encoder_1830.ENCODING_SIZE, f"Final offset {offset} after bid test"


def test_encoding_after_purchase(encoder_1830: Encoder_1830, test_game_1830_4p):
    """Verify encoding changes correctly after a player buys a private."""
    g, action_helper = test_game_1830_4p

    # Player 1 passes, Player 2 buys SV
    player1 = g.players[0]
    player2 = g.players[1]
    sv_private = g.company_by_id("SV")
    buy_price = g.round.active_step().min_bid(sv_private)

    # P1 Passes
    pass_action_p1 = None
    for action in action_helper.get_all_choices_limited():
        if action.__class__.__name__ == "Pass":
            pass_action_p1 = action
            break
    assert pass_action_p1 is not None, "Could not find pass action for P1"
    g.process_action(pass_action_p1)

    # P2 Buys SV
    buy_action_p2 = None
    # Update action helper for P2's turn
    action_helper = ActionHelper(g)  # Re-create or update helper
    for action in action_helper.get_all_choices_limited():
        if action.__class__.__name__ == "Bid":
            if action.company == sv_private and action.price == buy_price:
                buy_action_p2 = action
                break
    assert buy_action_p2 is not None, "Could not find buy action for P2 on SV"
    g.process_action(buy_action_p2)

    game_state_tensor, node_features_tensor, edge_index, edge_attributes = encoder_1830.encode(g)
    encoding = game_state_tensor.squeeze(0).numpy()
    offset = 0
    player_map = encoder_1830.player_id_to_idx
    private_map = encoder_1830.private_id_to_idx
    num_players = encoder_1830.num_players
    num_privates = encoder_1830.NUM_PRIVATES
    num_corps = encoder_1830.NUM_CORPORATIONS
    start_cash = encoder_1830.starting_cash
    sv_idx = private_map["SV"]
    p2_idx = player_map[player2.id]

    # --- Check Active Player (Section 1) ---
    s_slice, offset = get_section_slice(encoder_1830, "active_entity", offset)
    active_player_id = g.round.active_entities[0].id
    active_idx = player_map[active_player_id]
    assert active_idx == 2, "Active player should be Player 3 (index 2)"
    assert_float(encoding[s_slice][active_idx], 1.0, "Section 1: Player 3 Active after P2 buy")

    # Skip sections until Player Cash
    offset += sum(
        encoder_1830._get_section_size(name)
        for name in [
            "active_president",
            "round_type",
            "game_phase",
            "priority_deal_player",
            "bank_cash",
            "player_certs_remaining",
        ]
    )

    # --- Check Player Cash (Section 8) ---
    s_slice, offset = get_section_slice(encoder_1830, "player_cash", offset)
    expected_p2_cash = float(start_cash - buy_price) / start_cash
    assert_float(encoding[s_slice][p2_idx], expected_p2_cash, "Section 8: Player 2 Cash reduced")
    assert_float(encoding[s_slice][0], 1.0, "Section 8: Player 1 Cash unchanged")  # P1 passed

    # Skip Player Shares
    offset += encoder_1830._get_section_size("player_shares")

    # --- Check Private Ownership (Section 10) ---
    s_slice, offset = get_section_slice(encoder_1830, "private_ownership", offset)
    # Calculate the flat index for P2 owning SV
    owner_flat_idx = sv_idx * (num_players + num_corps) + p2_idx
    assert_float(1.0, encoding[s_slice][owner_flat_idx], "Section 10: Player 2 owns SV")
    assert np.sum(encoding[s_slice]) == 1.0, "Section 10: Only P2 should own SV"

    # Skip unchanged sections until Auction state
    offset += sum(
        encoder_1830._get_section_size(name)
        for name in [
            "private_revenue",
            "corp_floated",
            "corp_cash",
            "corp_trains",
            "corp_tokens_remaining",
            "corp_share_price",
            "corp_shares",
            "corp_market_zone",
            "depot_trains",
            "market_pool_trains",
            "depot_tiles",
            "auction_bids",
        ]
    )
    # TODO: Check if there are any bids
    s_slice_bids, _ = get_section_slice(
        encoder_1830, "auction_bids", offset - encoder_1830._get_section_size("auction_bids")
    )
    assert np.sum(encoding[s_slice_bids]) == 0, "Bids should clear after purchase"

    # --- Check Auction Min Bid (Section A2) ---
    s_slice, offset = get_section_slice(encoder_1830, "auction_min_bid", offset)
    # Min bid for SV should be irrelevant now, maybe 0 or -1? Encoder uses min_bid() which might error.
    # Let's assume the encoder handles sold privates gracefully (e.g., returns 0)
    # Or check the next available private's min bid
    next_private = g.company_by_id("CS")  # Assuming CSL is next
    csl_idx = private_map["CS"]
    expected_min_bid = float(g.round.active_step().min_bid(next_private)) / start_cash
    assert_float(encoding[s_slice][csl_idx], expected_min_bid, "Section A2: Min Bid for CSL")

    # --- Check Auction Available (Section A3) ---
    s_slice, offset = get_section_slice(encoder_1830, "auction_available", offset)
    # CSL should be available now
    assert_float(encoding[s_slice][csl_idx], 1.0, "Section A3: CSL Available")
    assert_float(encoding[s_slice][sv_idx], 0.0, "Section A3: SV Not Available")

    # --- Check Auction Face Value (Section A4) ---
    s_slice, offset = get_section_slice(encoder_1830, "auction_face_value", offset)
    # Face values should not change
    expected_fv_norm = float(sv_private.value) / start_cash
    assert_float(encoding[s_slice][sv_idx], expected_fv_norm, "Section A4: Face Value SV unchanged")

    # --- Final Offset Check ---
    assert offset == encoder_1830.ENCODING_SIZE, f"Final offset {offset} after purchase test"


def test_initial_map_node_features(encoder_1830: Encoder_1830, test_game_1830_4p):
    """Verify the node features encoding for specific hexes at the start."""
    g, _ = test_game_1830_4p
    _, node_features_tensor, edge_index, edge_attributes = encoder_1830.encode(g)
    node_features = node_features_tensor.numpy()

    # --- Check G19 (New York - City) ---
    hex_coord_ny = "G19"
    hex_idx_ny = encoder_1830.hex_coord_to_idx[hex_coord_ny]
    features_ny = node_features[hex_idx_ny, :]
    # Expected features for NY (Tile 57, rotation 0 initially?)
    # Revenue: 30 (Phase 1) -> 30 / MAX_HEX_REVENUE (80) = 0.375
    # Type: City=1, OO=0, Town=0, Offboard=0
    # Rotation: 0
    # Tokens: All 0 initially
    assert_float(0.5, features_ny[get_feature_index(encoder_1830, "revenue")], "Map: NY Revenue")
    assert_float(1.0, features_ny[get_feature_index(encoder_1830, "is_city")], "Map: NY is_city")
    assert_float(1.0, features_ny[get_feature_index(encoder_1830, "is_oo")], "Map: NY is_oo")
    assert_float(0.0, features_ny[get_feature_index(encoder_1830, "is_town")], "Map: NY is_town")
    assert_float(0.0, features_ny[get_feature_index(encoder_1830, "is_offboard")], "Map: NY is_offboard")
    assert_float(80.0 / 120.0, features_ny[get_feature_index(encoder_1830, "upgrade_cost")], "Map: NY upgrade_cost")
    assert_float(0.0, features_ny[get_feature_index(encoder_1830, "rotation")], "Map: NY rotation")
    # No edge connectivity
    # Only port 0 connects to revenue 0
    assert_float(
        1.0, features_ny[get_feature_index(encoder_1830, "port_0_connects_revenue_0")], "Map: NY port_0_rev_0"
    )
    # Only port 3 connects to revenue 1
    assert_float(
        1.0, features_ny[get_feature_index(encoder_1830, "port_3_connects_revenue_1")], "Map: NY port_3_rev_1"
    )
    assert_float(4.5 + 80.0 / 120.0, sum(features_ny), "Map: NY node features sum")

    # --- Check F2 (Offboard - West) ---
    hex_coord_off = "F2"
    hex_idx_off = encoder_1830.hex_coord_to_idx[hex_coord_off]
    features_off = node_features[hex_idx_off, :]
    # Expected features for F2 (Tile 1, rotation 0?)
    # Revenue: 30 (Phase 1) -> 30 / 80 = 0.375
    # Type: City=0, OO=0, Town=0, Offboard=1
    # Rotation: 0
    # Tokens: N/A (All 0)
    # Connectivity: Tile X1 connects edge 3 to revenue 0.
    assert_float(0.5, features_off[get_feature_index(encoder_1830, "revenue")], "Map: Offboard Revenue")
    assert_float(0.0, features_off[get_feature_index(encoder_1830, "is_city")], "Map: Offboard is_city")
    assert_float(0.0, features_off[get_feature_index(encoder_1830, "is_oo")], "Map: Offboard is_oo")
    assert_float(0.0, features_off[get_feature_index(encoder_1830, "is_town")], "Map: Offboard is_town")
    assert_float(1.0, features_off[get_feature_index(encoder_1830, "is_offboard")], "Map: Offboard is_offboard")
    assert_float(0.0, features_off[get_feature_index(encoder_1830, "upgrade_cost")], "Map: Offboard upgrade_cost")
    assert_float(0.0, features_off[get_feature_index(encoder_1830, "rotation")], "Map: Offboard rotation")
    # No tokens
    # Edge connectivity - None
    assert_float(0.0, features_off[get_feature_index(encoder_1830, "connects_5_0")], "Map: Offboard connects_5_0")
    assert_float(0.0, features_off[get_feature_index(encoder_1830, "connects_5_1")], "Map: Offboard connects_5_1")
    assert_float(0.0, features_off[get_feature_index(encoder_1830, "connects_5_2")], "Map: Offboard connects_5_2")
    assert_float(0.0, features_off[get_feature_index(encoder_1830, "connects_5_3")], "Map: Offboard connects_5_3")
    assert_float(0.0, features_off[get_feature_index(encoder_1830, "connects_5_4")], "Map: Offboard connects_5_4")
    assert_float(0.0, features_off[get_feature_index(encoder_1830, "connects_4_0")], "Map: Offboard connects_4_0")
    assert_float(0.0, features_off[get_feature_index(encoder_1830, "connects_4_1")], "Map: Offboard connects_4_1")
    assert_float(0.0, features_off[get_feature_index(encoder_1830, "connects_4_2")], "Map: Offboard connects_4_2")
    assert_float(0.0, features_off[get_feature_index(encoder_1830, "connects_4_3")], "Map: Offboard connects_4_3")
    assert_float(0.0, features_off[get_feature_index(encoder_1830, "connects_3_0")], "Map: Offboard connects_3_0")
    assert_float(0.0, features_off[get_feature_index(encoder_1830, "connects_3_1")], "Map: Offboard connects_3_1")
    assert_float(0.0, features_off[get_feature_index(encoder_1830, "connects_3_2")], "Map: Offboard connects_3_2")
    assert_float(0.0, features_off[get_feature_index(encoder_1830, "connects_2_0")], "Map: Offboard connects_2_0")
    assert_float(0.0, features_off[get_feature_index(encoder_1830, "connects_2_1")], "Map: Offboard connects_2_1")
    assert_float(0.0, features_off[get_feature_index(encoder_1830, "connects_1_0")], "Map: Offboard connects_1_0")
    # Port Connectivity - 2, 3, 4 connect to revenue 0
    assert_float(
        0.0,
        features_off[get_feature_index(encoder_1830, "port_0_connects_revenue_0")],
        "Map: Offboard port_0_rev_0 (False)",
    )
    assert_float(
        0.0,
        features_off[get_feature_index(encoder_1830, "port_1_connects_revenue_0")],
        "Map: Offboard port_1_rev_0 (False)",
    )
    assert_float(
        0.0,
        features_off[get_feature_index(encoder_1830, "port_2_connects_revenue_0")],
        "Map: Offboard port_2_rev_0 (False)",
    )
    assert_float(
        1.0,
        features_off[get_feature_index(encoder_1830, "port_3_connects_revenue_0")],
        "Map: Offboard port_3_rev_0 (False)",
    )
    assert_float(
        1.0,
        features_off[get_feature_index(encoder_1830, "port_4_connects_revenue_0")],
        "Map: Offboard port_4_rev_0 (False)",
    )
    assert_float(
        1.0,
        features_off[get_feature_index(encoder_1830, "port_5_connects_revenue_0")],
        "Map: Offboard port_5_rev_0 (False)",
    )
    # No revenue 1
    assert_float(
        0.0,
        features_off[get_feature_index(encoder_1830, "port_0_connects_revenue_1")],
        "Map: Offboard port_0_rev_1 (False)",
    )
    assert_float(
        0.0,
        features_off[get_feature_index(encoder_1830, "port_1_connects_revenue_1")],
        "Map: Offboard port_1_rev_1 (False)",
    )
    assert_float(
        0.0,
        features_off[get_feature_index(encoder_1830, "port_2_connects_revenue_1")],
        "Map: Offboard port_2_rev_1 (False)",
    )
    assert_float(
        0.0,
        features_off[get_feature_index(encoder_1830, "port_3_connects_revenue_1")],
        "Map: Offboard port_3_rev_1 (False)",
    )
    assert_float(
        0.0,
        features_off[get_feature_index(encoder_1830, "port_4_connects_revenue_1")],
        "Map: Offboard port_4_rev_1 (False)",
    )
    assert_float(
        0.0,
        features_off[get_feature_index(encoder_1830, "port_5_connects_revenue_1")],
        "Map: Offboard port_5_rev_1 (False)",
    )

    # --- Check E9 (Plain Grey Track) ---
    hex_coord_track = "E9"
    hex_idx_track = encoder_1830.hex_coord_to_idx[hex_coord_track]
    features_track = node_features[hex_idx_track, :]
    # Expected features for E9 (Tile E9, rotation 0?)
    # Revenue: 0 -> 0.0
    # Type: City=0, OO=0, Town=0, Offboard=0
    # Rotation: 0
    # Tokens: N/A (All 0)
    # Connectivity: Tile 7 connects edges 0-3.
    assert_float(0.0, features_track[get_feature_index(encoder_1830, "revenue")], "Map: Track Revenue")
    assert_float(0.0, features_track[get_feature_index(encoder_1830, "is_city")], "Map: Track is_city")
    assert_float(0.0, features_track[get_feature_index(encoder_1830, "is_oo")], "Map: Track is_oo")
    assert_float(0.0, features_track[get_feature_index(encoder_1830, "is_town")], "Map: Track is_town")
    assert_float(0.0, features_track[get_feature_index(encoder_1830, "is_offboard")], "Map: Track is_offboard")
    assert_float(0.0, features_track[get_feature_index(encoder_1830, "upgrade_cost")], "Map: Track upgrade_cost")
    assert_float(0.0, features_track[get_feature_index(encoder_1830, "rotation")], "Map: Track rotation")
    # Connectivity: Tile 7 connects edges 2-3.
    assert_float(0.0, features_track[get_feature_index(encoder_1830, "connects_5_0")], "Map: Track connects_5_0")
    assert_float(0.0, features_track[get_feature_index(encoder_1830, "connects_5_1")], "Map: Track connects_5_1")
    assert_float(0.0, features_track[get_feature_index(encoder_1830, "connects_5_2")], "Map: Track connects_5_2")
    assert_float(0.0, features_track[get_feature_index(encoder_1830, "connects_5_3")], "Map: Track connects_5_3")
    assert_float(0.0, features_track[get_feature_index(encoder_1830, "connects_5_4")], "Map: Track connects_5_4")
    assert_float(0.0, features_track[get_feature_index(encoder_1830, "connects_4_0")], "Map: Track connects_4_0")
    assert_float(0.0, features_track[get_feature_index(encoder_1830, "connects_4_1")], "Map: Track connects_4_1")
    assert_float(0.0, features_track[get_feature_index(encoder_1830, "connects_4_2")], "Map: Track connects_4_2")
    assert_float(0.0, features_track[get_feature_index(encoder_1830, "connects_4_3")], "Map: Track connects_4_3")
    assert_float(0.0, features_track[get_feature_index(encoder_1830, "connects_3_0")], "Map: Track connects_3_0")
    assert_float(0.0, features_track[get_feature_index(encoder_1830, "connects_3_1")], "Map: Track connects_3_1")
    assert_float(1.0, features_track[get_feature_index(encoder_1830, "connects_3_2")], "Map: Track connects_3_2")
    assert_float(0.0, features_track[get_feature_index(encoder_1830, "connects_2_0")], "Map: Track connects_2_0")
    assert_float(0.0, features_track[get_feature_index(encoder_1830, "connects_2_1")], "Map: Track connects_2_1")
    assert_float(0.0, features_track[get_feature_index(encoder_1830, "connects_1_0")], "Map: Track connects_1_0")
    # No revenue
    assert_float(
        0.0,
        features_track[get_feature_index(encoder_1830, "port_0_connects_revenue_0")],
        "Map: Track port_0_rev_0 (False)",
    )
    assert_float(
        0.0,
        features_track[get_feature_index(encoder_1830, "port_1_connects_revenue_0")],
        "Map: Track port_1_rev_0 (False)",
    )
    assert_float(
        0.0,
        features_track[get_feature_index(encoder_1830, "port_2_connects_revenue_0")],
        "Map: Track port_2_rev_0 (False)",
    )
    assert_float(
        0.0,
        features_track[get_feature_index(encoder_1830, "port_3_connects_revenue_0")],
        "Map: Track port_3_rev_0 (False)",
    )
    assert_float(
        0.0,
        features_track[get_feature_index(encoder_1830, "port_4_connects_revenue_0")],
        "Map: Track port_4_rev_0 (False)",
    )
    assert_float(
        0.0,
        features_track[get_feature_index(encoder_1830, "port_5_connects_revenue_0")],
        "Map: Track port_5_rev_0 (False)",
    )
    # No revenue 1
    assert_float(
        0.0,
        features_track[get_feature_index(encoder_1830, "port_0_connects_revenue_1")],
        "Map: Track port_0_rev_1 (False)",
    )
    assert_float(
        0.0,
        features_track[get_feature_index(encoder_1830, "port_1_connects_revenue_1")],
        "Map: Track port_1_rev_1 (False)",
    )
    assert_float(
        0.0,
        features_track[get_feature_index(encoder_1830, "port_2_connects_revenue_1")],
        "Map: Track port_2_rev_1 (False)",
    )
    assert_float(
        0.0,
        features_track[get_feature_index(encoder_1830, "port_3_connects_revenue_1")],
        "Map: Track port_3_rev_1 (False)",
    )
    assert_float(
        0.0,
        features_track[get_feature_index(encoder_1830, "port_4_connects_revenue_1")],
        "Map: Track port_4_rev_1 (False)",
    )
    assert_float(
        0.0,
        features_track[get_feature_index(encoder_1830, "port_5_connects_revenue_1")],
        "Map: Track port_5_rev_1 (False)",
    )


def test_encoding_after_par(encoder_1830: Encoder_1830, test_game_1830_4p):
    """Verify encoding changes correctly after a corporation is parred."""
    g, action_helper = test_game_1830_4p
    player1 = g.players[0]
    player2 = g.players[1]
    player3 = g.players[2]
    player4 = g.players[3]
    sv_comp = g.company_by_id("SV")
    cs_comp = g.company_by_id("CS")
    dh_comp = g.company_by_id("DH")
    mh_comp = g.company_by_id("MH")
    ca_comp = g.company_by_id("CA")
    bo_comp = g.company_by_id("BO")
    erie_corp = g.corporation_by_id("ERIE")
    bo_corp = g.corporation_by_id("B&O")
    prr_corp = g.corporation_by_id("PRR")
    par_price = 100  # B&O par price

    # --- Manually advance game state to BO purchase ---
    g.process_action(action_helper.get_all_choices_limited()[0])
    g.process_action(action_helper.get_all_choices_limited()[0])
    g.process_action(action_helper.get_all_choices_limited()[0])
    g.process_action(action_helper.get_all_choices_limited()[0])
    g.process_action(action_helper.get_all_choices_limited()[0])
    g.process_action(action_helper.get_all_choices_limited()[0])
    # --- End Manual Setup ---

    # Find the Par action for Player 1 parring B&O at 100
    par_action = None
    for action in action_helper.get_all_choices_limited():
        if action.__class__.__name__ == "Par":
            if action.corporation == bo_corp and action.share_price.price == par_price:
                par_action = action
                break
    assert par_action is not None, f"Could not find Par action for P1 on B&O at {par_price}"

    # Process the action
    g.process_action(par_action)

    # Encode the new state
    game_state_tensor, node_features_tensor, edge_index, edge_attributes = encoder_1830.encode(g)
    encoding = game_state_tensor.squeeze(0).numpy()
    offset = 0
    player_map = encoder_1830.player_id_to_idx
    corp_map = encoder_1830.corp_id_to_idx
    comp_map = encoder_1830.private_id_to_idx
    num_players = encoder_1830.num_players
    num_corps = encoder_1830.NUM_CORPORATIONS
    start_cash = encoder_1830.starting_cash
    # players
    p1_idx = player_map[player1.id]
    p2_idx = player_map[player2.id]
    p3_idx = player_map[player3.id]
    p4_idx = player_map[player4.id]
    # companies
    cs_idx = comp_map[cs_comp.id]
    sv_idx = comp_map[sv_comp.id]
    dh_idx = comp_map[dh_comp.id]
    mh_idx = comp_map[mh_comp.id]
    ca_idx = comp_map[ca_comp.id]
    bo_idx = comp_map[bo_comp.id]

    # corps
    erie_idx = corp_map[erie_corp.id]
    bno_idx = corp_map[bo_corp.id]
    prr_idx = corp_map[prr_corp.id]

    # Section 1: Active Entity
    # Should be P3's turn in the stock round
    s_slice, offset = get_section_slice(encoder_1830, "active_entity", offset)
    assert_float(1.0, encoding[s_slice][p3_idx], "Section 1: P3 Active Entity")

    # --- Section 2: Active President ---
    s_slice, offset = get_section_slice(encoder_1830, "active_president", offset)
    # No active president
    assert_float(0.0, encoding[s_slice][p1_idx], "Section 2: Player 1 Active President after Par")
    assert_float(0.0, encoding[s_slice][p2_idx], "Section 2: Player 2 Active President after Par")
    assert_float(0.0, encoding[s_slice][p3_idx], "Section 2: Player 3 Active President after Par")
    assert_float(0.0, encoding[s_slice][p4_idx], "Section 2: Player 4 Active President after Par")

    # --- Section 3: Round Type ---
    s_slice, offset = get_section_slice(encoder_1830, "round_type", offset)
    expected_round_idx = float(ROUND_TYPE_MAP["Stock"]) / MAX_ROUND_TYPE_IDX
    assert_float(expected_round_idx, encoding[s_slice].item(), "Section 3: Round Type Stock")

    # --- Section 4: Game Phase ---
    s_slice, offset = get_section_slice(encoder_1830, "game_phase", offset)
    phase_name = g.phase.name
    expected_phase_idx = PHASE_NAMES_ORDERED.index(phase_name) / (encoder_1830.NUM_PHASES - 1)
    assert phase_name == "2", "Game should be in Phase 2 after par"
    assert_float(expected_phase_idx, encoding[s_slice].item(), "Section 4: Game Phase 2")

    # --- Section 5: Priority Deal ---
    s_slice, offset = get_section_slice(encoder_1830, "priority_deal_player", offset)
    assert_float(0.0, encoding[s_slice][p1_idx], "Section 5: Player 1 Priority Deal")
    assert_float(0.0, encoding[s_slice][p2_idx], "Section 5: Player 2 Priority Deal")
    assert_float(1.0, encoding[s_slice][p3_idx], "Section 5: Player 3 Priority Deal")
    assert_float(0.0, encoding[s_slice][p4_idx], "Section 5: Player 4 Priority Deal")

    # --- Section 6: Bank Cash ---
    s_slice, offset = get_section_slice(encoder_1830, "bank_cash", offset)
    assert_float(10220.0 / 12000.0, encoding[s_slice].item(), "Section 6: Bank Cash")

    # Skip certs
    offset += encoder_1830._get_section_size("player_certs_remaining")

    # --- Section 8: Player Cash ---
    s_slice, offset = get_section_slice(encoder_1830, "player_cash", offset)
    expected_p1_cash = 420.0 / start_cash
    assert_float(expected_p1_cash, encoding[s_slice][p1_idx], "Section 8: Player 1 Cash reduced by Par")
    expected_p2_cash = 340.0 / start_cash
    assert_float(expected_p2_cash, encoding[s_slice][p2_idx], "Section 8: Player 2 Cash")
    expected_p3_cash = 530.0 / start_cash
    assert_float(expected_p3_cash, encoding[s_slice][p3_idx], "Section 8: Player 3 Cash")
    expected_p4_cash = 490.0 / start_cash
    assert_float(expected_p4_cash, encoding[s_slice][p4_idx], "Section 8: Player 4 Cash")

    # --- Section 9: Player Shares ---
    # P1 has 10% PRR, P2 has 20% B&O, others have none
    s_slice, offset = get_section_slice(encoder_1830, "player_shares", offset)
    p1_prr_share_flat_idx = p1_idx * num_corps + prr_idx
    p2_bo_share_flat_idx = p2_idx * num_corps + bno_idx
    p3_prr_share_flat_idx = p3_idx * num_corps + prr_idx
    p4_prr_share_flat_idx = p4_idx * num_corps + prr_idx
    assert_float(0.1, encoding[s_slice][p1_prr_share_flat_idx], "Section 9: Player 1 owns 10% PRR")
    assert_float(0.2, encoding[s_slice][p2_bo_share_flat_idx], "Section 9: Player 2 owns 20% B&O")
    assert_float(0.0, encoding[s_slice][p3_prr_share_flat_idx], "Section 9: Player 3 owns 0% PRR")
    assert_float(0.0, encoding[s_slice][p4_prr_share_flat_idx], "Section 9: Player 4 owns 0% PRR")

    assert (
        np.sum(encoding[s_slice][p1_idx * num_corps : (p1_idx + 1) * num_corps]) == 0.1
    ), "Section 9: P1 only owns PRR"
    assert (
        np.sum(encoding[s_slice][p2_idx * num_corps : (p2_idx + 1) * num_corps]) == 0.2
    ), "Section 9: P2 only owns B&O"
    assert np.sum(encoding[s_slice][p3_idx * num_corps : (p3_idx + 1) * num_corps]) == 0.0, "Section 9: P3 owns nothing"
    assert np.sum(encoding[s_slice][p4_idx * num_corps : (p4_idx + 1) * num_corps]) == 0.0, "Section 9: P4 owns nothing"

    # --- Section 10: Private Ownership ---
    s_slice, offset = get_section_slice(encoder_1830, "private_ownership", offset)
    # P1 owns SV and CA, P2 owns CS & BO, P3 owns DH, P4 owns MH
    assert_float(1.0, encoding[s_slice][sv_idx * (num_players + num_corps) + p1_idx], "Section 10: Player 1 owns SV")
    assert_float(1.0, encoding[s_slice][ca_idx * (num_players + num_corps) + p1_idx], "Section 10: Player 1 owns CA")
    assert_float(1.0, encoding[s_slice][cs_idx * (num_players + num_corps) + p2_idx], "Section 10: Player 2 owns CS")
    assert_float(1.0, encoding[s_slice][bo_idx * (num_players + num_corps) + p2_idx], "Section 10: Player 2 owns BO")
    assert_float(1.0, encoding[s_slice][dh_idx * (num_players + num_corps) + p3_idx], "Section 10: Player 3 owns DH")
    assert_float(1.0, encoding[s_slice][mh_idx * (num_players + num_corps) + p4_idx], "Section 10: Player 4 owns MH")

    # Skip Section 11: Private Revenue because it's constant
    offset += encoder_1830._get_section_size("private_revenue")
    # --- Section 12: Corp Floated ---
    # No corps floated
    s_slice, offset = get_section_slice(encoder_1830, "corp_floated", offset)
    assert_float(0.0, encoding[s_slice][bno_idx], "Section 12: B&O not floated")
    assert_float(0.0, encoding[s_slice][prr_idx], "Section 12: PRR not floated")

    # --- Section 13: Corp Cash ---
    # No corp cash
    s_slice, offset = get_section_slice(encoder_1830, "corp_cash", offset)
    assert_float(0.0, encoding[s_slice][bno_idx], "Section 13: B&O Cash")
    assert_float(0.0, encoding[s_slice][prr_idx], "Section 13: PRR Cash")

    # --- Section 14: Corp Trains ---
    # No corp trains
    s_slice, offset = get_section_slice(encoder_1830, "corp_trains", offset)
    assert_float(0.0, encoding[s_slice][bno_idx * 6], "Section 14: B&O Trains")
    assert_float(0.0, encoding[s_slice][prr_idx * 6], "Section 14: PRR Trains")

    # --- Section 15: Corp Tokens ---
    # No tokens placed = 100% remaining
    s_slice, offset = get_section_slice(encoder_1830, "corp_tokens_remaining", offset)
    assert_float(1.0, encoding[s_slice][bno_idx], "Section 15: B&O Tokens")
    assert_float(1.0, encoding[s_slice][prr_idx], "Section 15: PRR Tokens")

    # --- Section 16: Corp Share Price ---
    # B&O has been parred, PRR has not
    s_slice, offset = get_section_slice(encoder_1830, "corp_share_price", offset)
    assert_float(100.0 / 350.0, encoding[s_slice][bno_idx * 2], "Section 16: B&O IPO Price")
    assert_float(100.0 / 350.0, encoding[s_slice][bno_idx * 2 + 1], "Section 16: B&O Share Price")
    assert_float(0.0, encoding[s_slice][prr_idx * 2], "Section 16: PRR IPO Price")
    assert_float(0.0, encoding[s_slice][prr_idx * 2 + 1], "Section 16: PRR Share Price")

    # --- Section 17: Corp Shares Remaining ---
    # B&O has 80% remaining, PRR has 90% remaining, rest have 100%
    s_slice, offset = get_section_slice(encoder_1830, "corp_shares", offset)
    assert_float(0.8, encoding[s_slice][2 * bno_idx], "Section 17: B&O IPO Shares Remaining")
    assert_float(0.0, encoding[s_slice][2 * bno_idx + 1], "Section 17: B&O Market Shares Remaining")
    assert_float(0.9, encoding[s_slice][2 * prr_idx], "Section 17: PRR IPO Shares Remaining")
    assert_float(0.0, encoding[s_slice][2 * prr_idx + 1], "Section 17: PRR Market Shares Remaining")
    assert_float(1.0, encoding[s_slice][2 * erie_idx], "Section 17: ERIE IPO Shares Remaining")
    assert_float(0.0, encoding[s_slice][2 * erie_idx + 1], "Section 17: ERIE Market Shares Remaining")

    # --- Section 18: Corp Market Zone ---
    # Only B&O has a share price and it is grey
    s_slice, offset = get_section_slice(encoder_1830, "corp_market_zone", offset)
    assert_float(1.0, encoding[s_slice][bno_idx * 4], "Section 17: B&O Market Zone")
    assert sum(encoding[s_slice]) == 1.0

    # --- Section 19: Depot Trains ---
    # All depot trains available
    s_slice, offset = get_section_slice(encoder_1830, "depot_trains", offset)
    assert_float(1.0, encoding[s_slice][0], "Section 18: Depot 2 Trains")
    assert_float(1.0, encoding[s_slice][1], "Section 18: Depot 3 Trains")
    assert_float(1.0, encoding[s_slice][2], "Section 18: Depot 4 Trains")
    assert_float(1.0, encoding[s_slice][3], "Section 18: Depot 5 Trains")
    assert_float(1.0, encoding[s_slice][4], "Section 18: Depot 6 Trains")
    assert_float(1.0, encoding[s_slice][5], "Section 18: Depot D Trains")

    # --- Section 20: Market Pool Trains ---
    # No market pool trains
    s_slice, offset = get_section_slice(encoder_1830, "market_pool_trains", offset)
    assert_float(0.0, encoding[s_slice][0], "Section 20: Market Pool 2 Trains")
    assert_float(0.0, encoding[s_slice][1], "Section 20: Market Pool 3 Trains")
    assert_float(0.0, encoding[s_slice][2], "Section 20: Market Pool 4 Trains")
    assert_float(0.0, encoding[s_slice][3], "Section 20: Market Pool 5 Trains")
    assert_float(0.0, encoding[s_slice][4], "Section 20: Market Pool 6 Trains")
    assert_float(0.0, encoding[s_slice][5], "Section 20: Market Pool D Trains")

    # --- Section 21: Depot Tiles ---
    # All depot tiles
    s_slice, offset = get_section_slice(encoder_1830, "depot_tiles", offset)
    assert_float(1.0, encoding[s_slice][0], "Section 21: Depot 1 Tile")
    assert_float(1.0, encoding[s_slice][1], "Section 21: Depot 2 Tile")
    assert_float(1.0, encoding[s_slice][2], "Section 21: Depot 3 Tile")
    assert_float(1.0, encoding[s_slice][3], "Section 21: Depot 4 Tile")
    assert_float(1.0, encoding[s_slice][4], "Section 21: Depot 7 Tile")
    assert_float(1.0, encoding[s_slice][5], "Section 21: Depot 8 Tile")
    assert_float(1.0, encoding[s_slice][6], "Section 21: Depot 9 Tile")
    assert_float(1.0, encoding[s_slice][7], "Section 21: Depot 14 Tile")
    assert_float(1.0, encoding[s_slice][8], "Section 21: Depot 15 Tile")
    assert_float(1.0, encoding[s_slice][9], "Section 21: Depot 16 Tile")
    assert_float(1.0, encoding[s_slice][10], "Section 21: Depot 18 Tile")
    assert_float(1.0, encoding[s_slice][11], "Section 21: Depot 19 Tile")
    assert_float(1.0, encoding[s_slice][12], "Section 21: Depot 23 Tile")
    # ...

    # Section A: Auction: No auction data (except private face value)
    # --- Section A1: Auction Bids ---
    s_slice, offset = get_section_slice(encoder_1830, "auction_bids", offset)
    # Whole slice is 0
    assert np.sum(encoding[s_slice]) == 0.0, "Section A: Auction Bids"
    # --- Section A2: Auction Min Bid ---
    s_slice, offset = get_section_slice(encoder_1830, "auction_min_bid", offset)
    # Whole slice is 0
    assert np.sum(encoding[s_slice]) == 0.0, "Section A: Auction Min Bid"
    # --- Section A3: Auction Available ---
    s_slice, offset = get_section_slice(encoder_1830, "auction_available", offset)
    # Whole slice is 0
    assert np.sum(encoding[s_slice]) == 0.0, "Section A: Auction Available"
    # --- Section A4: Auction Face Value ---
    s_slice, offset = get_section_slice(encoder_1830, "auction_face_value", offset)
    # Manual check
    assert_float(sv_comp.value / 600.0, encoding[s_slice][sv_idx], "Section A: SV Face Value")
    assert_float(cs_comp.value / 600.0, encoding[s_slice][cs_idx], "Section A: CS Face Value")
    assert_float(dh_comp.value / 600.0, encoding[s_slice][dh_idx], "Section A: DH Face Value")
    assert_float(mh_comp.value / 600.0, encoding[s_slice][mh_idx], "Section A: MH Face Value")
    assert_float(ca_comp.value / 600.0, encoding[s_slice][ca_idx], "Section A: CA Face Value")
    assert_float(bo_comp.value / 600.0, encoding[s_slice][bo_idx], "Section A: BO Face Value")

    # --- Final Offset Check ---
    assert offset == encoder_1830.ENCODING_SIZE, f"Final offset {offset} after par test"

    # --- Manual Setup Part 2: Float B&O, then move to Operating Round ---
    g.process_action(action_helper.get_all_choices_limited()[0])
    g.process_action(action_helper.get_all_choices_limited()[0])
    g.process_action(action_helper.get_all_choices_limited()[0])
    g.process_action(action_helper.get_all_choices_limited()[0])
    g.process_action(action_helper.get_all_choices_limited()[-1])
    g.process_action(action_helper.get_all_choices_limited()[-1])
    g.process_action(action_helper.get_all_choices_limited()[-1])
    g.process_action(action_helper.get_all_choices_limited()[-1])

    game_state_tensor, node_features_tensor, edge_index, edge_attributes = encoder_1830.encode(g)
    encoding = game_state_tensor.squeeze(0).numpy()
    offset = 0

    # Section 1: Active Entity
    # Should be B&O's turn
    s_slice, offset = get_section_slice(encoder_1830, "active_entity", offset)
    assert_float(1.0, encoding[s_slice][num_players + bno_idx], "Section 1: B&O Active Entity")

    # --- Section 2: Active President ---
    s_slice, offset = get_section_slice(encoder_1830, "active_president", offset)
    # P2 is acting as president
    assert_float(0.0, encoding[s_slice][p1_idx], "Section 2: Player 1 Active President after Par")
    assert_float(1.0, encoding[s_slice][p2_idx], "Section 2: Player 2 Active President after Par")
    assert_float(0.0, encoding[s_slice][p3_idx], "Section 2: Player 3 Active President after Par")
    assert_float(0.0, encoding[s_slice][p4_idx], "Section 2: Player 4 Active President after Par")

    # --- Section 3: Round Type ---
    s_slice, offset = get_section_slice(encoder_1830, "round_type", offset)
    expected_round_idx = float(ROUND_TYPE_MAP["Operating"]) / MAX_ROUND_TYPE_IDX
    assert_float(expected_round_idx, encoding[s_slice].item(), "Section 3: Operating Type Stock")

    # --- Section 4: Game Phase ---
    s_slice, offset = get_section_slice(encoder_1830, "game_phase", offset)
    phase_name = g.phase.name
    expected_phase_idx = PHASE_NAMES_ORDERED.index(phase_name) / (encoder_1830.NUM_PHASES - 1)
    assert phase_name == "2", "Game should be in Phase 2 after par"
    assert_float(expected_phase_idx, encoding[s_slice].item(), "Section 4: Game Phase 2")

    # --- Section 5: Priority Deal ---
    s_slice, offset = get_section_slice(encoder_1830, "priority_deal_player", offset)
    assert_float(0.0, encoding[s_slice][p1_idx], "Section 5: Player 1 Priority Deal")
    assert_float(0.0, encoding[s_slice][p2_idx], "Section 5: Player 2 Priority Deal")
    assert_float(1.0, encoding[s_slice][p3_idx], "Section 5: Player 3 Priority Deal")
    assert_float(0.0, encoding[s_slice][p4_idx], "Section 5: Player 4 Priority Deal")

    # --- Section 6: Bank Cash ---
    s_slice, offset = get_section_slice(encoder_1830, "bank_cash", offset)
    assert_float(9515.0 / 12000.0, encoding[s_slice].item(), "Section 6: Bank Cash")

    # --- Section 7: Player Certs ---
    s_slice, offset = get_section_slice(encoder_1830, "player_certs_remaining", offset)
    assert_float(12.0 / 16.0, encoding[s_slice][p1_idx], "Section 7: Player 1 Certs")
    assert_float(12.0 / 16.0, encoding[s_slice][p2_idx], "Section 7: Player 2 Certs")
    assert_float(14.0 / 16.0, encoding[s_slice][p3_idx], "Section 7: Player 3 Certs")
    assert_float(14.0 / 16.0, encoding[s_slice][p4_idx], "Section 7: Player 4 Certs")

    # --- Section 8: Player Cash ---
    s_slice, offset = get_section_slice(encoder_1830, "player_cash", offset)
    expected_p1_cash = 350.0 / start_cash
    assert_float(expected_p1_cash, encoding[s_slice][p1_idx], "Section 8: Player 1 Cash reduced by Par")
    expected_p2_cash = 280.0 / start_cash
    assert_float(expected_p2_cash, encoding[s_slice][p2_idx], "Section 8: Player 2 Cash")
    expected_p3_cash = 445.0 / start_cash
    assert_float(expected_p3_cash, encoding[s_slice][p3_idx], "Section 8: Player 3 Cash")
    expected_p4_cash = 410.0 / start_cash
    assert_float(expected_p4_cash, encoding[s_slice][p4_idx], "Section 8: Player 4 Cash")

    # --- Section 9: Player Shares ---
    # P1 has 10% PRR, P2 has 20% B&O, others have none
    s_slice, offset = get_section_slice(encoder_1830, "player_shares", offset)
    p1_prr_share_flat_idx = p1_idx * num_corps + prr_idx
    p1_bno_share_flat_idx = p1_idx * num_corps + bno_idx
    p2_bo_share_flat_idx = p2_idx * num_corps + bno_idx
    p3_bno_share_flat_idx = p3_idx * num_corps + bno_idx
    p4_bno_share_flat_idx = p4_idx * num_corps + bno_idx

    assert_float(0.1, encoding[s_slice][p1_prr_share_flat_idx], "Section 9: Player 1 owns 10% PRR")
    assert_float(0.1, encoding[s_slice][p1_bno_share_flat_idx], "Section 9: Player 1 owns 10% B&O")
    assert_float(0.3, encoding[s_slice][p2_bo_share_flat_idx], "Section 9: Player 2 owns 30% B&O")
    assert_float(0.1, encoding[s_slice][p3_bno_share_flat_idx], "Section 9: Player 3 owns 10% B&O")
    assert_float(0.1, encoding[s_slice][p4_bno_share_flat_idx], "Section 9: Player 4 owns 10% B&O")

    # --- Section 10: Private Ownership ---
    s_slice, offset = get_section_slice(encoder_1830, "private_ownership", offset)
    # P1 owns SV and CA, P2 owns CS & BO, P3 owns DH, P4 owns MH
    assert_float(1.0, encoding[s_slice][sv_idx * (num_players + num_corps) + p1_idx], "Section 10: Player 1 owns SV")
    assert_float(1.0, encoding[s_slice][ca_idx * (num_players + num_corps) + p1_idx], "Section 10: Player 1 owns CA")
    assert_float(1.0, encoding[s_slice][cs_idx * (num_players + num_corps) + p2_idx], "Section 10: Player 2 owns CS")
    assert_float(1.0, encoding[s_slice][bo_idx * (num_players + num_corps) + p2_idx], "Section 10: Player 2 owns BO")
    assert_float(1.0, encoding[s_slice][dh_idx * (num_players + num_corps) + p3_idx], "Section 10: Player 3 owns DH")
    assert_float(1.0, encoding[s_slice][mh_idx * (num_players + num_corps) + p4_idx], "Section 10: Player 4 owns MH")

    # Skip Section 11: Private Revenue because it's constant
    offset += encoder_1830._get_section_size("private_revenue")

    # --- Section 12: Corp Floated ---
    # B&O floated
    s_slice, offset = get_section_slice(encoder_1830, "corp_floated", offset)
    assert_float(1.0, encoding[s_slice][bno_idx], "Section 12: B&O floated")
    assert_float(0.0, encoding[s_slice][prr_idx], "Section 12: PRR not floated")

    # --- Section 13: Corp Cash ---
    # B&O has 1000
    s_slice, offset = get_section_slice(encoder_1830, "corp_cash", offset)
    assert_float(1000.0 / start_cash, encoding[s_slice][bno_idx], "Section 13: B&O Cash")
    assert_float(0.0, encoding[s_slice][prr_idx], "Section 13: PRR Cash")

    # --- Section 14: Corp Trains ---
    # No corp trains
    s_slice, offset = get_section_slice(encoder_1830, "corp_trains", offset)
    assert_float(0.0, encoding[s_slice][bno_idx * 6], "Section 14: B&O Trains")
    assert_float(0.0, encoding[s_slice][prr_idx * 6], "Section 14: PRR Trains")

    # --- Section 15: Corp Tokens ---
    # B&O has placed one of 3 tokens
    s_slice, offset = get_section_slice(encoder_1830, "corp_tokens_remaining", offset)
    assert_float(0.6667, encoding[s_slice][bno_idx], "Section 15: B&O Tokens")
    assert_float(1.0, encoding[s_slice][prr_idx], "Section 15: PRR Tokens")

    # --- Section 16: Corp Share Price ---
    # B&O has been parred, PRR has not
    s_slice, offset = get_section_slice(encoder_1830, "corp_share_price", offset)
    assert_float(100.0 / 350.0, encoding[s_slice][bno_idx * 2], "Section 16: B&O IPO Price")
    assert_float(100.0 / 350.0, encoding[s_slice][bno_idx * 2 + 1], "Section 16: B&O Share Price")
    assert_float(0.0, encoding[s_slice][prr_idx * 2], "Section 16: PRR IPO Price")
    assert_float(0.0, encoding[s_slice][prr_idx * 2 + 1], "Section 16: PRR Share Price")

    # --- Section 17: Corp Shares Remaining ---
    # B&O has 80% remaining, PRR has 90% remaining, rest have 100%
    s_slice, offset = get_section_slice(encoder_1830, "corp_shares", offset)
    assert_float(0.4, encoding[s_slice][2 * bno_idx], "Section 17: B&O IPO Shares Remaining")
    assert_float(0.0, encoding[s_slice][2 * bno_idx + 1], "Section 17: B&O Market Shares Remaining")
    assert_float(0.9, encoding[s_slice][2 * prr_idx], "Section 17: PRR IPO Shares Remaining")
    assert_float(0.0, encoding[s_slice][2 * prr_idx + 1], "Section 17: PRR Market Shares Remaining")
    assert_float(1.0, encoding[s_slice][2 * erie_idx], "Section 17: ERIE IPO Shares Remaining")
    assert_float(0.0, encoding[s_slice][2 * erie_idx + 1], "Section 17: ERIE Market Shares Remaining")

    # --- Section 18: Corp Market Zone ---
    # B&O has been parred, PRR has not. However, both are in normal market zone (0)
    s_slice, offset = get_section_slice(encoder_1830, "corp_market_zone", offset)
    assert_float(1.0, encoding[s_slice][bno_idx * 4], "Section 17: B&O Market Zone")
    assert sum(encoding[s_slice]) == 1.0

    # --- Section 19: Depot Trains ---
    # All depot trains available
    s_slice, offset = get_section_slice(encoder_1830, "depot_trains", offset)
    assert_float(1.0, encoding[s_slice][0], "Section 18: Depot 2 Trains")
    assert_float(1.0, encoding[s_slice][1], "Section 18: Depot 3 Trains")
    assert_float(1.0, encoding[s_slice][2], "Section 18: Depot 4 Trains")
    assert_float(1.0, encoding[s_slice][3], "Section 18: Depot 5 Trains")
    assert_float(1.0, encoding[s_slice][4], "Section 18: Depot 6 Trains")
    assert_float(1.0, encoding[s_slice][5], "Section 18: Depot D Trains")

    # --- Section 20: Market Pool Trains ---
    # No market pool trains
    s_slice, offset = get_section_slice(encoder_1830, "market_pool_trains", offset)
    assert_float(0.0, encoding[s_slice][0], "Section 20: Market Pool 2 Trains")
    assert_float(0.0, encoding[s_slice][1], "Section 20: Market Pool 3 Trains")
    assert_float(0.0, encoding[s_slice][2], "Section 20: Market Pool 4 Trains")
    assert_float(0.0, encoding[s_slice][3], "Section 20: Market Pool 5 Trains")
    assert_float(0.0, encoding[s_slice][4], "Section 20: Market Pool 6 Trains")
    assert_float(0.0, encoding[s_slice][5], "Section 20: Market Pool D Trains")

    # --- Section 21: Depot Tiles ---
    # All depot tiles
    s_slice, offset = get_section_slice(encoder_1830, "depot_tiles", offset)
    assert_float(1.0, encoding[s_slice][0], "Section 21: Depot 1 Tile")
    assert_float(1.0, encoding[s_slice][1], "Section 21: Depot 2 Tile")
    assert_float(1.0, encoding[s_slice][2], "Section 21: Depot 3 Tile")
    assert_float(1.0, encoding[s_slice][3], "Section 21: Depot 4 Tile")
    assert_float(1.0, encoding[s_slice][4], "Section 21: Depot 7 Tile")
    assert_float(1.0, encoding[s_slice][5], "Section 21: Depot 8 Tile")
    assert_float(1.0, encoding[s_slice][6], "Section 21: Depot 9 Tile")
    assert_float(1.0, encoding[s_slice][7], "Section 21: Depot 14 Tile")
    assert_float(1.0, encoding[s_slice][8], "Section 21: Depot 15 Tile")
    assert_float(1.0, encoding[s_slice][9], "Section 21: Depot 16 Tile")
    assert_float(1.0, encoding[s_slice][10], "Section 21: Depot 18 Tile")
    assert_float(1.0, encoding[s_slice][11], "Section 21: Depot 19 Tile")
    assert_float(1.0, encoding[s_slice][12], "Section 21: Depot 23 Tile")
    # ...

    # Section A: Auction: No auction data (except private face value)
    # --- Section A1: Auction Bids ---
    s_slice, offset = get_section_slice(encoder_1830, "auction_bids", offset)
    # Whole slice is 0
    assert np.sum(encoding[s_slice]) == 0.0, "Section A: Auction Bids"
    # --- Section A2: Auction Min Bid ---
    s_slice, offset = get_section_slice(encoder_1830, "auction_min_bid", offset)
    # Whole slice is 0
    assert np.sum(encoding[s_slice]) == 0.0, "Section A: Auction Min Bid"
    # --- Section A3: Auction Available ---
    s_slice, offset = get_section_slice(encoder_1830, "auction_available", offset)
    # Whole slice is 0
    assert np.sum(encoding[s_slice]) == 0.0, "Section A: Auction Available"
    # --- Section A4: Auction Face Value ---
    s_slice, offset = get_section_slice(encoder_1830, "auction_face_value", offset)
    # Manual check
    assert_float(sv_comp.value / start_cash, encoding[s_slice][sv_idx], "Section A: SV Face Value")
    assert_float(cs_comp.value / start_cash, encoding[s_slice][cs_idx], "Section A: CS Face Value")
    assert_float(dh_comp.value / start_cash, encoding[s_slice][dh_idx], "Section A: DH Face Value")
    assert_float(mh_comp.value / start_cash, encoding[s_slice][mh_idx], "Section A: MH Face Value")
    assert_float(ca_comp.value / start_cash, encoding[s_slice][ca_idx], "Section A: CA Face Value")
    assert_float(bo_comp.value / start_cash, encoding[s_slice][bo_idx], "Section A: BO Face Value")

    # --- Final Offset Check ---
    assert offset == encoder_1830.ENCODING_SIZE, f"Final offset {offset} after par test"

    # --- Manual Setup Part 2: Do the B&O operating round, then check one last time ---
    g.process_action(action_helper.get_all_choices_limited()[0])
    g.process_action(action_helper.get_all_choices_limited()[0])
    g.process_action(action_helper.get_all_choices_limited()[-1])

    game_state_tensor, node_features_tensor, edge_index, edge_attributes = encoder_1830.encode(g)
    encoding = game_state_tensor.squeeze(0).numpy()
    offset = 0

    # Section 1: Active Entity
    # Should be P3's turn
    s_slice, offset = get_section_slice(encoder_1830, "active_entity", offset)
    assert_float(1.0, encoding[s_slice][p3_idx], "Section 1: P3 Active Entity")

    # --- Section 2: Active President ---
    s_slice, offset = get_section_slice(encoder_1830, "active_president", offset)
    # No president
    assert_float(0.0, encoding[s_slice][p1_idx], "Section 2: Player 1 Active President after Par")
    assert_float(0.0, encoding[s_slice][p2_idx], "Section 2: Player 2 Active President after Par")
    assert_float(0.0, encoding[s_slice][p3_idx], "Section 2: Player 3 Active President after Par")
    assert_float(0.0, encoding[s_slice][p4_idx], "Section 2: Player 4 Active President after Par")

    # --- Section 3: Round Type ---
    s_slice, offset = get_section_slice(encoder_1830, "round_type", offset)
    expected_round_idx = float(ROUND_TYPE_MAP["Stock"]) / MAX_ROUND_TYPE_IDX
    assert_float(expected_round_idx, encoding[s_slice].item(), "Section 3: Round Type Stock")

    # --- Section 4: Game Phase ---
    s_slice, offset = get_section_slice(encoder_1830, "game_phase", offset)
    phase_name = g.phase.name
    expected_phase_idx = PHASE_NAMES_ORDERED.index(phase_name) / (encoder_1830.NUM_PHASES - 1)
    assert phase_name == "2", "Game should be in Phase 2"
    assert_float(expected_phase_idx, encoding[s_slice].item(), "Section 4: Game Phase 2")

    # --- Section 5: Priority Deal ---
    s_slice, offset = get_section_slice(encoder_1830, "priority_deal_player", offset)
    assert_float(0.0, encoding[s_slice][p1_idx], "Section 5: Player 1 Priority Deal")
    assert_float(0.0, encoding[s_slice][p2_idx], "Section 5: Player 2 Priority Deal")
    assert_float(1.0, encoding[s_slice][p3_idx], "Section 5: Player 3 Priority Deal")
    assert_float(0.0, encoding[s_slice][p4_idx], "Section 5: Player 4 Priority Deal")

    # --- Section 6: Bank Cash ---
    s_slice, offset = get_section_slice(encoder_1830, "bank_cash", offset)
    assert_float(9675.0 / 12000.0, encoding[s_slice].item(), "Section 6: Bank Cash")

    # --- Section 7: Player Certs ---
    s_slice, offset = get_section_slice(encoder_1830, "player_certs_remaining", offset)
    assert_float(12.0 / 16.0, encoding[s_slice][p1_idx], "Section 7: Player 1 Certs")
    assert_float(13.0 / 16.0, encoding[s_slice][p2_idx], "Section 7: Player 2 Certs")
    assert_float(14.0 / 16.0, encoding[s_slice][p3_idx], "Section 7: Player 3 Certs")
    assert_float(14.0 / 16.0, encoding[s_slice][p4_idx], "Section 7: Player 4 Certs")

    # --- Section 8: Player Cash ---
    s_slice, offset = get_section_slice(encoder_1830, "player_cash", offset)
    expected_p1_cash = 350.0 / start_cash
    assert_float(expected_p1_cash, encoding[s_slice][p1_idx], "Section 8: Player 1 Cash reduced by Par")
    expected_p2_cash = 280.0 / start_cash
    assert_float(expected_p2_cash, encoding[s_slice][p2_idx], "Section 8: Player 2 Cash")
    expected_p3_cash = 445.0 / start_cash
    assert_float(expected_p3_cash, encoding[s_slice][p3_idx], "Section 8: Player 3 Cash")
    expected_p4_cash = 410.0 / start_cash
    assert_float(expected_p4_cash, encoding[s_slice][p4_idx], "Section 8: Player 4 Cash")

    # --- Section 9: Player Shares ---
    # P1 has 10% PRR and 10% B&O, P2 has 30% B&O, others have 10% B&O
    s_slice, offset = get_section_slice(encoder_1830, "player_shares", offset)
    p1_prr_share_flat_idx = p1_idx * num_corps + prr_idx
    p1_bno_share_flat_idx = p1_idx * num_corps + bno_idx
    p2_bo_share_flat_idx = p2_idx * num_corps + bno_idx
    p3_bno_share_flat_idx = p3_idx * num_corps + bno_idx
    p4_bno_share_flat_idx = p4_idx * num_corps + bno_idx

    assert_float(0.1, encoding[s_slice][p1_prr_share_flat_idx], "Section 9: Player 1 owns 10% PRR")
    assert_float(0.1, encoding[s_slice][p1_bno_share_flat_idx], "Section 9: Player 1 owns 10% B&O")
    assert_float(0.3, encoding[s_slice][p2_bo_share_flat_idx], "Section 9: Player 2 owns 30% B&O")
    assert_float(0.1, encoding[s_slice][p3_bno_share_flat_idx], "Section 9: Player 3 owns 10% B&O")
    assert_float(0.1, encoding[s_slice][p4_bno_share_flat_idx], "Section 9: Player 4 owns 10% B&O")

    # --- Section 10: Private Ownership ---
    s_slice, offset = get_section_slice(encoder_1830, "private_ownership", offset)
    # P1 owns SV and CA, P2 owns CS, P3 owns DH, P4 owns MH, No one owns BO
    assert_float(1.0, encoding[s_slice][sv_idx * (num_players + num_corps) + p1_idx], "Section 10: Player 1 owns SV")
    assert_float(1.0, encoding[s_slice][ca_idx * (num_players + num_corps) + p1_idx], "Section 10: Player 1 owns CA")
    assert_float(1.0, encoding[s_slice][cs_idx * (num_players + num_corps) + p2_idx], "Section 10: Player 2 owns CS")
    assert_float(0.0, encoding[s_slice][bo_idx * (num_players + num_corps) + p2_idx], "Section 10: Player 2 owns BO")
    assert_float(1.0, encoding[s_slice][dh_idx * (num_players + num_corps) + p3_idx], "Section 10: Player 3 owns DH")
    assert_float(1.0, encoding[s_slice][mh_idx * (num_players + num_corps) + p4_idx], "Section 10: Player 4 owns MH")

    # Skip Section 11: Private Revenue because it's constant
    offset += encoder_1830._get_section_size("private_revenue")

    # --- Section 12: Corp Floated ---
    # B&O floated
    s_slice, offset = get_section_slice(encoder_1830, "corp_floated", offset)
    assert_float(1.0, encoding[s_slice][bno_idx], "Section 12: B&O floated")
    assert_float(0.0, encoding[s_slice][prr_idx], "Section 12: PRR not floated")

    # --- Section 13: Corp Cash ---
    # B&O has 840
    s_slice, offset = get_section_slice(encoder_1830, "corp_cash", offset)
    assert_float(840.0 / start_cash, encoding[s_slice][bno_idx], "Section 13: B&O Cash")
    assert_float(0.0, encoding[s_slice][prr_idx], "Section 13: PRR Cash")

    # --- Section 14: Corp Trains ---
    # B&O has a 2
    s_slice, offset = get_section_slice(encoder_1830, "corp_trains", offset)
    assert_float(1.0 / 6, encoding[s_slice][bno_idx * 6], "Section 14: B&O Trains")
    assert_float(0.0, encoding[s_slice][prr_idx * 6], "Section 14: PRR Trains")

    # --- Section 15: Corp Tokens ---
    # B&O has placed one of 3 tokens
    s_slice, offset = get_section_slice(encoder_1830, "corp_tokens_remaining", offset)
    assert_float(0.6667, encoding[s_slice][bno_idx], "Section 15: B&O Tokens")
    assert_float(1.0, encoding[s_slice][prr_idx], "Section 15: PRR Tokens")

    # --- Section 16: Corp Share Price ---
    # B&O has moved left, PRR has not
    s_slice, offset = get_section_slice(encoder_1830, "corp_share_price", offset)
    assert_float(90.0 / 350.0, encoding[s_slice][bno_idx * 2 + 1], "Section 16: B&O IPO Price")
    assert_float(100.0 / 350.0, encoding[s_slice][bno_idx * 2], "Section 16: B&O Share Price")
    assert_float(0.0, encoding[s_slice][prr_idx * 2], "Section 16: PRR Share Price")

    # --- Section 17: Corp Shares Remaining ---
    # B&O has 40% remaining, PRR has 90% remaining, rest have 100%
    s_slice, offset = get_section_slice(encoder_1830, "corp_shares", offset)
    assert_float(0.4, encoding[s_slice][2 * bno_idx], "Section 17: B&O IPO Shares Remaining")
    assert_float(0.0, encoding[s_slice][2 * bno_idx + 1], "Section 17: B&O Market Shares Remaining")
    assert_float(0.9, encoding[s_slice][2 * prr_idx], "Section 17: PRR IPO Shares Remaining")
    assert_float(0.0, encoding[s_slice][2 * prr_idx + 1], "Section 17: PRR Market Shares Remaining")
    assert_float(1.0, encoding[s_slice][2 * erie_idx], "Section 17: ERIE IPO Shares Remaining")
    assert_float(0.0, encoding[s_slice][2 * erie_idx + 1], "Section 17: ERIE Market Shares Remaining")

    # --- Section 18: Corp Market Zone ---
    # B&O has been parred, PRR has not. However, both are in normal market zone (0)
    s_slice, offset = get_section_slice(encoder_1830, "corp_market_zone", offset)
    assert_float(1.0, encoding[s_slice][bno_idx * 4], "Section 17: B&O Market Zone")
    assert sum(encoding[s_slice]) == 1.0

    # --- Section 19: Depot Trains ---
    # All depot trains available
    s_slice, offset = get_section_slice(encoder_1830, "depot_trains", offset)
    assert_float(5.0 / 6.0, encoding[s_slice][0], "Section 18: Depot 2 Trains")
    assert_float(1.0, encoding[s_slice][1], "Section 18: Depot 3 Trains")
    assert_float(1.0, encoding[s_slice][2], "Section 18: Depot 4 Trains")
    assert_float(1.0, encoding[s_slice][3], "Section 18: Depot 5 Trains")
    assert_float(1.0, encoding[s_slice][4], "Section 18: Depot 6 Trains")
    assert_float(1.0, encoding[s_slice][5], "Section 18: Depot D Trains")

    # --- Section 20: Market Pool Trains ---
    # No market pool trains
    s_slice, offset = get_section_slice(encoder_1830, "market_pool_trains", offset)
    assert_float(0.0, encoding[s_slice][0], "Section 20: Market Pool 2 Trains")
    assert_float(0.0, encoding[s_slice][1], "Section 20: Market Pool 3 Trains")
    assert_float(0.0, encoding[s_slice][2], "Section 20: Market Pool 4 Trains")
    assert_float(0.0, encoding[s_slice][3], "Section 20: Market Pool 5 Trains")
    assert_float(0.0, encoding[s_slice][4], "Section 20: Market Pool 6 Trains")
    assert_float(0.0, encoding[s_slice][5], "Section 20: Market Pool D Trains")

    # --- Section 21: Depot Tiles ---
    # All depot tiles
    s_slice, offset = get_section_slice(encoder_1830, "depot_tiles", offset)
    assert_float(1.0, encoding[s_slice][0], "Section 21: Depot 1 Tile")
    assert_float(1.0, encoding[s_slice][1], "Section 21: Depot 2 Tile")
    assert_float(1.0, encoding[s_slice][2], "Section 21: Depot 3 Tile")
    assert_float(1.0, encoding[s_slice][3], "Section 21: Depot 4 Tile")
    assert_float(3.0 / 4.0, encoding[s_slice][4], "Section 21: Depot 7 Tile")
    assert_float(1.0, encoding[s_slice][5], "Section 21: Depot 8 Tile")
    assert_float(1.0, encoding[s_slice][6], "Section 21: Depot 9 Tile")
    assert_float(1.0, encoding[s_slice][7], "Section 21: Depot 14 Tile")
    assert_float(1.0, encoding[s_slice][8], "Section 21: Depot 15 Tile")
    assert_float(1.0, encoding[s_slice][9], "Section 21: Depot 16 Tile")
    assert_float(1.0, encoding[s_slice][10], "Section 21: Depot 18 Tile")
    assert_float(1.0, encoding[s_slice][11], "Section 21: Depot 19 Tile")
    assert_float(1.0, encoding[s_slice][12], "Section 21: Depot 23 Tile")
    # ...

    # Section A: Auction: No auction data (except private face value)
    # --- Section A1: Auction Bids ---
    s_slice, offset = get_section_slice(encoder_1830, "auction_bids", offset)
    # Whole slice is 0
    assert np.sum(encoding[s_slice]) == 0.0, "Section A: Auction Bids"
    # --- Section A2: Auction Min Bid ---
    s_slice, offset = get_section_slice(encoder_1830, "auction_min_bid", offset)
    # Whole slice is 0
    assert np.sum(encoding[s_slice]) == 0.0, "Section A: Auction Min Bid"
    # --- Section A3: Auction Available ---
    s_slice, offset = get_section_slice(encoder_1830, "auction_available", offset)
    # Whole slice is 0
    assert np.sum(encoding[s_slice]) == 0.0, "Section A: Auction Available"
    # --- Section A4: Auction Face Value ---
    s_slice, offset = get_section_slice(encoder_1830, "auction_face_value", offset)
    # Manual check
    assert_float(sv_comp.value / start_cash, encoding[s_slice][sv_idx], "Section A: SV Face Value")
    assert_float(cs_comp.value / start_cash, encoding[s_slice][cs_idx], "Section A: CS Face Value")
    assert_float(dh_comp.value / start_cash, encoding[s_slice][dh_idx], "Section A: DH Face Value")
    assert_float(mh_comp.value / start_cash, encoding[s_slice][mh_idx], "Section A: MH Face Value")
    assert_float(ca_comp.value / start_cash, encoding[s_slice][ca_idx], "Section A: CA Face Value")
    assert_float(bo_comp.value / start_cash, encoding[s_slice][bo_idx], "Section A: BO Face Value")

    # --- Final Offset Check ---
    assert offset == encoder_1830.ENCODING_SIZE, f"Final offset {offset} after par test"

    # Check map data as well
    node_features = node_features_tensor.squeeze(0).numpy()

    # Check node features
    # Hex I17 has a new tile that connects edges 1 and 2
    hex_coord_track = "I17"
    hex_idx_track = encoder_1830.hex_coord_to_idx[hex_coord_track]
    features_track = node_features[hex_idx_track, :]
    # Expected features for I17 (Tile I17, rotation 0)
    # Revenue: 0 -> 0.0
    # Type: City=0, OO=0, Town=0, Offboard=0
    # Rotation: 0
    # Tokens: N/A (All 0)
    assert_float(0.0, features_track[get_feature_index(encoder_1830, "revenue")], "Map: Track Revenue")
    assert_float(0.0, features_track[get_feature_index(encoder_1830, "is_city")], "Map: Track is_city")
    assert_float(0.0, features_track[get_feature_index(encoder_1830, "is_oo")], "Map: Track is_oo")
    assert_float(0.0, features_track[get_feature_index(encoder_1830, "is_town")], "Map: Track is_town")
    assert_float(0.0, features_track[get_feature_index(encoder_1830, "is_offboard")], "Map: Track is_offboard")
    assert_float(0.0, features_track[get_feature_index(encoder_1830, "upgrade_cost")], "Map: Track upgrade_cost")
    assert_float(1.0, features_track[get_feature_index(encoder_1830, "rotation")], "Map: Track rotation")
    # Connectivity: Tile 7 connects edges 1 and 2.
    assert_float(0.0, features_track[get_feature_index(encoder_1830, "connects_5_0")], "Map: Track connects_5_0")
    assert_float(0.0, features_track[get_feature_index(encoder_1830, "connects_5_1")], "Map: Track connects_5_1")
    assert_float(0.0, features_track[get_feature_index(encoder_1830, "connects_5_2")], "Map: Track connects_5_2")
    assert_float(0.0, features_track[get_feature_index(encoder_1830, "connects_5_3")], "Map: Track connects_5_3")
    assert_float(0.0, features_track[get_feature_index(encoder_1830, "connects_5_4")], "Map: Track connects_5_4")
    assert_float(0.0, features_track[get_feature_index(encoder_1830, "connects_4_0")], "Map: Track connects_4_0")
    assert_float(0.0, features_track[get_feature_index(encoder_1830, "connects_4_1")], "Map: Track connects_4_1")
    assert_float(0.0, features_track[get_feature_index(encoder_1830, "connects_4_2")], "Map: Track connects_4_2")
    assert_float(0.0, features_track[get_feature_index(encoder_1830, "connects_4_3")], "Map: Track connects_4_3")
    assert_float(0.0, features_track[get_feature_index(encoder_1830, "connects_3_0")], "Map: Track connects_3_0")
    assert_float(0.0, features_track[get_feature_index(encoder_1830, "connects_3_1")], "Map: Track connects_3_1")
    assert_float(0.0, features_track[get_feature_index(encoder_1830, "connects_3_2")], "Map: Track connects_3_2")
    assert_float(0.0, features_track[get_feature_index(encoder_1830, "connects_2_0")], "Map: Track connects_2_0")
    assert_float(1.0, features_track[get_feature_index(encoder_1830, "connects_2_1")], "Map: Track connects_2_1")
    assert_float(0.0, features_track[get_feature_index(encoder_1830, "connects_1_0")], "Map: Track connects_1_0")


def test_operating_round_2_encoding(encoder_1830: Encoder_1830, operating_round_2_game_state: Tuple[BaseGame, ActionHelper]):
    operating_round_2_game_state, action_helper = operating_round_2_game_state

    # Test initial encoding
    game_state_tensor, node_features_tensor, edge_index_tensor, edge_attributes_tensor = encoder_1830.encode(operating_round_2_game_state)
    encoding = game_state_tensor.squeeze(0).numpy()
    offset = 0
    # --------------------- Check game_state ---------------------
    # # Section 1: Active Entity
    # Should be NYNH's turn in the operating round
    s_slice, offset = get_section_slice(encoder_1830, "active_entity", offset)
    assert_float(1.0, encoding[s_slice][4 + encoder_1830.corp_id_to_idx["NYNH"]], "Section 1: NYNH Active Entity")
    assert sum(encoding[s_slice]) == 1.0, "Section 1: NYNH Active Entity"
    # # Section 2: Active President
    # P3 is president of NYNH
    s_slice, offset = get_section_slice(encoder_1830, "active_president", offset)
    assert_float(1.0, encoding[s_slice][encoder_1830.player_id_to_idx["3"]], "Section 2: Player 3 Active President")
    assert sum(encoding[s_slice]) == 1.0, "Section 2: Player 3 Active President"
    # # Section 3: Round Type
    # Operating round
    s_slice, offset = get_section_slice(encoder_1830, "round_type", offset)
    expected_round_idx = float(ROUND_TYPE_MAP["Operating"]) / MAX_ROUND_TYPE_IDX
    assert_float(expected_round_idx, encoding[s_slice].item(), "Section 3: Round Type Operating")
    # --- Section 4: Game Phase ---
    s_slice, offset = get_section_slice(encoder_1830, "game_phase", offset)
    phase_name = operating_round_2_game_state.phase.name
    expected_phase_idx = PHASE_NAMES_ORDERED.index(phase_name) / (encoder_1830.NUM_PHASES - 1)
    assert phase_name == "2", "Game should be in Phase 2 in OR 2"
    assert_float(expected_phase_idx, encoding[s_slice].item(), "Section 4: Game Phase 2")
    # --- Section 5: Priority Deal ---
    s_slice, offset = get_section_slice(encoder_1830, "priority_deal_player", offset)
    assert_float(1.0, encoding[s_slice][encoder_1830.player_id_to_idx["4"]], "Section 5: Player 4 Priority Deal")
    assert sum(encoding[s_slice]) == 1.0, "Section 5: Player 4 Priority Deal"
    # --- Section 6: Bank Cash ---
    s_slice, offset = get_section_slice(encoder_1830, "bank_cash", offset)
    assert_float(9431.0 / 12000.0, encoding[s_slice].item(), "Section 6: Bank Cash")
    # --- Section 7: Player Certs ---
    s_slice, offset = get_section_slice(encoder_1830, "player_certs_remaining", offset)
    assert_float(10.0 / 16.0, encoding[s_slice][encoder_1830.player_id_to_idx["1"]], "Section 7: Player 1 Certs")
    assert_float(9.0 / 16.0, encoding[s_slice][encoder_1830.player_id_to_idx["2"]], "Section 7: Player 2 Certs")
    assert_float(10.0 / 16.0, encoding[s_slice][encoder_1830.player_id_to_idx["3"]], "Section 7: Player 3 Certs")
    assert_float(8.0 / 16.0, encoding[s_slice][encoder_1830.player_id_to_idx["4"]], "Section 7: Player 4 Certs")
    # --- Section 8: Player Cash ---
    s_slice, offset = get_section_slice(encoder_1830, "player_cash", offset)
    expected_p1_cash = 24.0 / 600.0
    assert_float(expected_p1_cash, encoding[s_slice][encoder_1830.player_id_to_idx["1"]], "Section 8: Player 1 Cash reduced by Par")
    expected_p2_cash = 20.0 / 600.0
    assert_float(expected_p2_cash, encoding[s_slice][encoder_1830.player_id_to_idx["2"]], "Section 8: Player 2 Cash")
    expected_p3_cash = 68.0 / 600.0
    assert_float(expected_p3_cash, encoding[s_slice][encoder_1830.player_id_to_idx["3"]], "Section 8: Player 3 Cash")
    expected_p4_cash = 57.0 / 600.0
    assert_float(expected_p4_cash, encoding[s_slice][encoder_1830.player_id_to_idx["4"]], "Section 8: Player 4 Cash")
    # --- Section 9: Player Shares ---
    # P1 has 10% PRR, P2 has 20% B&O, others have none
    s_slice, offset = get_section_slice(encoder_1830, "player_shares", offset)
    assert_float(0.6, encoding[s_slice][encoder_1830.player_id_to_idx["1"] * 8 + encoder_1830.corp_id_to_idx["PRR"]], "Section 9: Player 1 owns 60% PRR")
    assert_float(0.3, encoding[s_slice][encoder_1830.player_id_to_idx["1"] * 8 + encoder_1830.corp_id_to_idx["NYC"]], "Section 9: Player 1 owns 30% NYC")
    assert_float(0.6, encoding[s_slice][encoder_1830.player_id_to_idx["2"] * 8 + encoder_1830.corp_id_to_idx["C&O"]], "Section 9: Player 2 owns 60% C&O")
    assert_float(0.1, encoding[s_slice][encoder_1830.player_id_to_idx["2"] * 8 + encoder_1830.corp_id_to_idx["NYNH"]], "Section 9: Player 2 owns 10% NYNH")
    assert_float(0.1, encoding[s_slice][encoder_1830.player_id_to_idx["2"] * 8 + encoder_1830.corp_id_to_idx["NYC"]], "Section 9: Player 2 owns 10% NYC")
    assert_float(0.5, encoding[s_slice][encoder_1830.player_id_to_idx["3"] * 8 + encoder_1830.corp_id_to_idx["NYNH"]], "Section 9: Player 3 owns 50% NYNH")
    assert_float(0.1, encoding[s_slice][encoder_1830.player_id_to_idx["3"] * 8 + encoder_1830.corp_id_to_idx["NYC"]], "Section 9: Player 3 owns 10% NYC")
    assert_float(0.4, encoding[s_slice][encoder_1830.player_id_to_idx["4"] * 8 + encoder_1830.corp_id_to_idx["PRR"]], "Section 9: Player 4 owns 40% PRR")
    assert_float(0.2, encoding[s_slice][encoder_1830.player_id_to_idx["4"] * 8 + encoder_1830.corp_id_to_idx["B&O"]], "Section 9: Player 4 owns 20% B&O")
    assert_float(0.1, encoding[s_slice][encoder_1830.player_id_to_idx["4"] * 8 + encoder_1830.corp_id_to_idx["C&O"]], "Section 9: Player 4 owns 10% C&O")
    assert abs(sum(encoding[s_slice]) - 3.0) < 1e-6, f"Section 9 sum: expected 3.0, got {sum(encoding[s_slice])}"
    # --- Section 10: Private Ownership ---
    s_slice, offset = get_section_slice(encoder_1830, "private_ownership", offset)
    # P1 owns SV and CA, P2 owns CS & BO, P3 owns DH, P4 owns MH
    assert_float(1.0, encoding[s_slice][encoder_1830.private_id_to_idx["SV"] * (4 + 8) + encoder_1830.player_id_to_idx["1"]], "Section 10: Player 1 owns SV")
    assert_float(1.0, encoding[s_slice][encoder_1830.private_id_to_idx["CS"] * (4 + 8) + encoder_1830.player_id_to_idx["3"]], "Section 10: Player 3 owns CS")
    assert_float(1.0, encoding[s_slice][encoder_1830.private_id_to_idx["DH"] * (4 + 8) + encoder_1830.player_id_to_idx["2"]], "Section 10: Player 2 owns DH")
    assert_float(1.0, encoding[s_slice][encoder_1830.private_id_to_idx["MH"] * (4 + 8) + encoder_1830.player_id_to_idx["3"]], "Section 10: Player 3 owns MH")
    assert_float(1.0, encoding[s_slice][encoder_1830.private_id_to_idx["CA"] * (4 + 8) + encoder_1830.player_id_to_idx["4"]], "Section 10: Player 4 owns CA")
    assert_float(1.0, encoding[s_slice][encoder_1830.private_id_to_idx["BO"] * (4 + 8) + encoder_1830.player_id_to_idx["4"]], "Section 10: Player 4 owns BO")
    assert sum(encoding[s_slice]) == 6.0, "Section 10 sum"
    # Skip Section 11: Private Revenue because it's constant
    offset += encoder_1830._get_section_size("private_revenue")
    # --- Section 12: Corp Floated ---
    # B&O floated
    s_slice, offset = get_section_slice(encoder_1830, "corp_floated", offset)
    assert_float(1.0, encoding[s_slice][encoder_1830.corp_id_to_idx["PRR"]], "Section 12: PRR floated")
    assert_float(1.0, encoding[s_slice][encoder_1830.corp_id_to_idx["C&O"]], "Section 12: C&O floated")
    assert_float(1.0, encoding[s_slice][encoder_1830.corp_id_to_idx["NYNH"]], "Section 12: NYNH floated")
    assert_float(1.0, encoding[s_slice][encoder_1830.corp_id_to_idx["NYC"]], "Section 12: NYC floated")
    assert sum(encoding[s_slice]) == 4.0, "Section 12 sum"
    # --- Section 13: Corp Cash ---
    # B&O has 1000
    s_slice, offset = get_section_slice(encoder_1830, "corp_cash", offset)
    assert_float(0.0, encoding[s_slice][encoder_1830.corp_id_to_idx["B&O"]], "Section 13: B&O Cash")
    assert_float(510.0 / 600.0, encoding[s_slice][encoder_1830.corp_id_to_idx["PRR"]], "Section 13: PRR Cash")
    assert_float(590.0 / 600.0, encoding[s_slice][encoder_1830.corp_id_to_idx["NYC"]], "Section 13: NYC Cash")
    assert_float(590.0 / 600.0, encoding[s_slice][encoder_1830.corp_id_to_idx["C&O"]], "Section 13: C&O Cash")
    assert_float(710.0 / 600.0, encoding[s_slice][encoder_1830.corp_id_to_idx["NYNH"]], "Section 13: NYNH Cash")
    assert_float(0.0, encoding[s_slice][encoder_1830.corp_id_to_idx["B&M"]], "Section 13: B&M Cash")
    # --- Section 14: Corp Trains ---
    s_slice, offset = get_section_slice(encoder_1830, "corp_trains", offset)
    assert_float(0.0, encoding[s_slice][encoder_1830.corp_id_to_idx["B&O"] * 6], "Section 14: B&O Trains")
    assert_float(2.0 / 6.0, encoding[s_slice][encoder_1830.corp_id_to_idx["PRR"] * 6], "Section 14: PRR 2 Trains")
    assert_float(1.0 / 6.0, encoding[s_slice][encoder_1830.corp_id_to_idx["NYC"] * 6], "Section 14: NYC Trains")
    assert_float(1.0 / 6.0, encoding[s_slice][encoder_1830.corp_id_to_idx["C&O"] * 6], "Section 14: C&O Trains")
    assert_float(0.0, encoding[s_slice][encoder_1830.corp_id_to_idx["NYNH"] * 6], "Section 14: NYNH Trains")
    assert sum(encoding[s_slice]) == 4.0 / 6.0, "Section 14 sum"
    # --- Section 15: Corp Tokens ---
    s_slice, offset = get_section_slice(encoder_1830, "corp_tokens_remaining", offset)
    assert_float(1.0, encoding[s_slice][encoder_1830.corp_id_to_idx["B&O"]], "Section 15: B&O Tokens")
    assert_float(3.0 / 4.0, encoding[s_slice][encoder_1830.corp_id_to_idx["PRR"]], "Section 15: PRR Tokens")
    assert_float(3.0 / 4.0, encoding[s_slice][encoder_1830.corp_id_to_idx["NYC"]], "Section 15: NYC Tokens")
    assert_float(2.0 / 3.0, encoding[s_slice][encoder_1830.corp_id_to_idx["C&O"]], "Section 15: C&O Tokens")
    assert_float(1.0 / 2.0, encoding[s_slice][encoder_1830.corp_id_to_idx["NYNH"]], "Section 15: NYNH Tokens")
    # --- Section 16: Corp Share Price ---
    s_slice, offset = get_section_slice(encoder_1830, "corp_share_price", offset)
    assert_float(100.0 / 350.0, encoding[s_slice][encoder_1830.corp_id_to_idx["B&O"] * 2], "Section 16: B&O IPO Price")
    assert_float(100.0 / 350.0, encoding[s_slice][encoder_1830.corp_id_to_idx["B&O"] * 2 + 1], "Section 16: B&O Share Price")
    assert_float(67.0 / 350.0, encoding[s_slice][encoder_1830.corp_id_to_idx["PRR"] * 2], "Section 16: PRR IPO Price")
    assert_float(71.0 / 350.0, encoding[s_slice][encoder_1830.corp_id_to_idx["PRR"] * 2 + 1], "Section 16: PRR Share Price")
    assert_float(67.0 / 350.0, encoding[s_slice][encoder_1830.corp_id_to_idx["NYC"] * 2], "Section 16: NYC IPO Price")
    assert_float(30.0 / 350.0, encoding[s_slice][encoder_1830.corp_id_to_idx["NYC"] * 2 + 1], "Section 16: NYC Share Price")
    assert_float(67.0 / 350.0, encoding[s_slice][encoder_1830.corp_id_to_idx["C&O"] * 2], "Section 16: C&O IPO Price")
    assert_float(65.0 / 350.0, encoding[s_slice][encoder_1830.corp_id_to_idx["C&O"] * 2 + 1], "Section 16: C&O Share Price")
    assert_float(71.0 / 350.0, encoding[s_slice][encoder_1830.corp_id_to_idx["NYNH"] * 2], "Section 16: NYNH IPO Price")
    assert_float(71.0 / 350.0, encoding[s_slice][encoder_1830.corp_id_to_idx["NYNH"] * 2 + 1], "Section 16: NYNH Share Price")
    # --- Section 17: Corp Shares ---
    s_slice, offset = get_section_slice(encoder_1830, "corp_shares", offset)
    assert_float(0.8, encoding[s_slice][encoder_1830.corp_id_to_idx["B&O"] * 2], "Section 17: B&O IPO Shares Remaining")
    assert_float(0.0, encoding[s_slice][encoder_1830.corp_id_to_idx["B&O"] * 2 + 1], "Section 17: B&O Market Shares Remaining")
    assert_float(0.0, encoding[s_slice][encoder_1830.corp_id_to_idx["PRR"] * 2], "Section 17: PRR IPO Shares Remaining")
    assert_float(0.0, encoding[s_slice][encoder_1830.corp_id_to_idx["PRR"] * 2 + 1], "Section 17: PRR Market Shares Remaining")
    assert_float(0.1, encoding[s_slice][encoder_1830.corp_id_to_idx["NYC"] * 2], "Section 17: NYC IPO Shares Remaining")
    assert_float(0.4, encoding[s_slice][encoder_1830.corp_id_to_idx["NYC"] * 2 + 1], "Section 17: NYC Market Shares Remaining")
    assert_float(0.3, encoding[s_slice][encoder_1830.corp_id_to_idx["C&O"] * 2], "Section 17: C&O IPO Shares Remaining")
    assert_float(0.0, encoding[s_slice][encoder_1830.corp_id_to_idx["C&O"] * 2 + 1], "Section 17: C&O Market Shares Remaining")
    assert_float(0.4, encoding[s_slice][encoder_1830.corp_id_to_idx["NYNH"] * 2], "Section 17: NYNH IPO Shares Remaining")
    assert_float(0.0, encoding[s_slice][encoder_1830.corp_id_to_idx["NYNH"] * 2 + 1], "Section 17: NYNH Market Shares Remaining")
    # ERIE is not parred yet
    assert_float(1.0, encoding[s_slice][encoder_1830.corp_id_to_idx["ERIE"] * 2], "Section 17: ERIE IPO Shares Remaining")
    assert_float(0.0, encoding[s_slice][encoder_1830.corp_id_to_idx["ERIE"] * 2 + 1], "Section 17: ERIE Market Shares Remaining")
    # --- Section 18: Corp Market Zone ---
    s_slice, offset = get_section_slice(encoder_1830, "corp_market_zone", offset)
    assert_float(1.0, encoding[s_slice][encoder_1830.corp_id_to_idx["B&O"] * 4], "Section 18: B&O Market Zone")
    assert_float(1.0, encoding[s_slice][encoder_1830.corp_id_to_idx["PRR"] * 4], "Section 18: PRR Market Zone")
    # NYC is in brown
    assert_float(1.0, encoding[s_slice][encoder_1830.corp_id_to_idx["NYC"] * 4 + 3], "Section 18: NYC Market Zone")
    assert_float(1.0, encoding[s_slice][encoder_1830.corp_id_to_idx["C&O"] * 4], "Section 18: C&O Market Zone")
    assert_float(1.0, encoding[s_slice][encoder_1830.corp_id_to_idx["NYNH"] * 4], "Section 18: NYNH Market Zone")
    assert sum(encoding[s_slice]) == 5.0, "Section 18 sum"
    # --- Section 19: Depot Trains ---
    s_slice, offset = get_section_slice(encoder_1830, "depot_trains", offset)
    assert_float(2.0 / 6.0, encoding[s_slice][0], "Section 19: Depot 2 Trains")
    assert_float(1.0, encoding[s_slice][1], "Section 19: Depot 3 Trains")
    assert_float(1.0, encoding[s_slice][2], "Section 19: Depot 4 Trains")
    assert_float(1.0, encoding[s_slice][3], "Section 19: Depot 5 Trains")
    assert_float(1.0, encoding[s_slice][4], "Section 19: Depot 6 Trains")
    assert_float(1.0, encoding[s_slice][5], "Section 19: Depot D Trains")
    # --- Section 20: Market Pool Trains ---
    # No market pool trains
    s_slice, offset = get_section_slice(encoder_1830, "market_pool_trains", offset)
    assert_float(0.0, encoding[s_slice][0], "Section 20: Market Pool 2 Trains")
    assert_float(0.0, encoding[s_slice][1], "Section 20: Market Pool 3 Trains")
    assert_float(0.0, encoding[s_slice][2], "Section 20: Market Pool 4 Trains")
    assert_float(0.0, encoding[s_slice][3], "Section 20: Market Pool 5 Trains")
    assert_float(0.0, encoding[s_slice][4], "Section 20: Market Pool 6 Trains")
    assert_float(0.0, encoding[s_slice][5], "Section 20: Market Pool D Trains")
    # --- Section 21: Depot Tiles ---
    s_slice, offset = get_section_slice(encoder_1830, "depot_tiles", offset)
    assert_float(7.0 / 8.0, encoding[s_slice][encoder_1830.tile_id_to_idx["8"]], "Section 21: Depot 8 Tile")
    assert_float(2.0 / 4.0, encoding[s_slice][encoder_1830.tile_id_to_idx["57"]], "Section 21: Depot 57 Tile")
    expected_normalized_tile_count = len(encoder_1830.tile_id_to_idx) - 0.5 - (1.0 / 8.0)
    assert_float(expected_normalized_tile_count, sum(encoding[s_slice]), "Section 21: Normalized tile count")
    # Section A: Auction: No auction data (except private face value)
    # --- Section A1: Auction Bids ---
    s_slice, offset = get_section_slice(encoder_1830, "auction_bids", offset)
    # Whole slice is 0
    assert np.sum(encoding[s_slice]) == 0.0, "Section A: Auction Bids"
    # --- Section A2: Auction Min Bid ---
    s_slice, offset = get_section_slice(encoder_1830, "auction_min_bid", offset)
    # Whole slice is 0
    assert np.sum(encoding[s_slice]) == 0.0, "Section A: Auction Min Bid"
    # --- Section A3: Auction Available ---
    s_slice, offset = get_section_slice(encoder_1830, "auction_available", offset)
    # Whole slice is 0
    assert np.sum(encoding[s_slice]) == 0.0, "Section A: Auction Available"
    # --- Section A4: Auction Face Value ---
    s_slice, offset = get_section_slice(encoder_1830, "auction_face_value", offset)
    # Manual check
    assert_float(operating_round_2_game_state.company_by_id("SV").value / 600.0, encoding[s_slice][encoder_1830.private_id_to_idx["SV"]], "Section A: SV Face Value")
    assert_float(operating_round_2_game_state.company_by_id("CS").value / 600.0, encoding[s_slice][encoder_1830.private_id_to_idx["CS"]], "Section A: CS Face Value")
    assert_float(operating_round_2_game_state.company_by_id("DH").value / 600.0, encoding[s_slice][encoder_1830.private_id_to_idx["DH"]], "Section A: DH Face Value")
    assert_float(operating_round_2_game_state.company_by_id("MH").value / 600.0, encoding[s_slice][encoder_1830.private_id_to_idx["MH"]], "Section A: MH Face Value")
    assert_float(operating_round_2_game_state.company_by_id("CA").value / 600.0, encoding[s_slice][encoder_1830.private_id_to_idx["CA"]], "Section A: CA Face Value")
    assert_float(operating_round_2_game_state.company_by_id("BO").value / 600.0, encoding[s_slice][encoder_1830.private_id_to_idx["BO"]], "Section A: BO Face Value")

    # --- Final Offset Check ---
    assert offset == encoder_1830.ENCODING_SIZE, f"Final offset {offset} after par test"

    # --------------------- Check node_features ---------------------
    node_features = node_features_tensor.squeeze(0).numpy()
    # Hex H10 had a #57 added with rotation 1
    hex_coord_track = "H10"
    hex_idx_track = encoder_1830.hex_coord_to_idx[hex_coord_track]
    features_track = node_features[hex_idx_track, :]
    # Expected features for H10 (Tile 57, rotation 1)
    # Revenue: 0 -> 20.0 / 80.0
    # Type: City=1, OO=0, Town=0, Offboard=0
    # Rotation: 1 (or possibly 4)
    # Tokens: N/A (All 0)
    assert_float(20.0 / 80.0, features_track[get_feature_index(encoder_1830, "revenue")], "Map: Track Revenue")
    assert_float(1.0, features_track[get_feature_index(encoder_1830, "is_city")], "Map: Track is_city")
    assert_float(0.0, features_track[get_feature_index(encoder_1830, "is_oo")], "Map: Track is_oo")
    assert_float(0.0, features_track[get_feature_index(encoder_1830, "is_town")], "Map: Track is_town")
    assert_float(0.0, features_track[get_feature_index(encoder_1830, "is_offboard")], "Map: Track is_offboard")
    assert_float(0.0, features_track[get_feature_index(encoder_1830, "upgrade_cost")], "Map: Track upgrade_cost")
    assert_float(1.0, features_track[get_feature_index(encoder_1830, "rotation")], "Map: Track rotation")
    # No edge connectivity.
    assert_float(0.0, features_track[get_feature_index(encoder_1830, "connects_5_0")], "Map: Track connects_5_0")
    assert_float(0.0, features_track[get_feature_index(encoder_1830, "connects_5_1")], "Map: Track connects_5_1")
    assert_float(0.0, features_track[get_feature_index(encoder_1830, "connects_5_2")], "Map: Track connects_5_2")
    assert_float(0.0, features_track[get_feature_index(encoder_1830, "connects_5_3")], "Map: Track connects_5_3")
    assert_float(0.0, features_track[get_feature_index(encoder_1830, "connects_5_4")], "Map: Track connects_5_4")
    assert_float(0.0, features_track[get_feature_index(encoder_1830, "connects_4_0")], "Map: Track connects_4_0")
    assert_float(0.0, features_track[get_feature_index(encoder_1830, "connects_4_1")], "Map: Track connects_4_1")
    assert_float(0.0, features_track[get_feature_index(encoder_1830, "connects_4_2")], "Map: Track connects_4_2")
    assert_float(0.0, features_track[get_feature_index(encoder_1830, "connects_4_3")], "Map: Track connects_4_3")
    assert_float(0.0, features_track[get_feature_index(encoder_1830, "connects_3_0")], "Map: Track connects_3_0")
    assert_float(0.0, features_track[get_feature_index(encoder_1830, "connects_3_1")], "Map: Track connects_3_1")
    assert_float(0.0, features_track[get_feature_index(encoder_1830, "connects_3_2")], "Map: Track connects_3_2")
    assert_float(0.0, features_track[get_feature_index(encoder_1830, "connects_2_0")], "Map: Track connects_2_0")
    assert_float(0.0, features_track[get_feature_index(encoder_1830, "connects_2_1")], "Map: Track connects_2_1")
    assert_float(0.0, features_track[get_feature_index(encoder_1830, "connects_1_0")], "Map: Track connects_1_0")
    # Ports 1 and 4 connect to revenue 0
    assert_float(
        0.0, features_track[get_feature_index(encoder_1830, "port_0_connects_revenue_0")], "Map: H10 port_0_rev_0 (False)"
    )
    assert_float(
        1.0, features_track[get_feature_index(encoder_1830, "port_1_connects_revenue_0")], "Map: H10 port_1_rev_0"
    )
    assert_float(
        0.0, features_track[get_feature_index(encoder_1830, "port_2_connects_revenue_0")], "Map: H10 port_2_rev_0 (False)"
    )
    assert_float(0.0, features_track[get_feature_index(encoder_1830, "port_3_connects_revenue_0")], "Map: H10 port_3_rev_0 (False)")
    assert_float(
        1.0, features_track[get_feature_index(encoder_1830, "port_4_connects_revenue_0")], "Map: H10 port_4_rev_0"
    )
    assert_float(
        0.0, features_track[get_feature_index(encoder_1830, "port_5_connects_revenue_0")], "Map: H10 port_5_rev_0 (False)"
    )
    assert_float(4.0 + 20.0/80.0, sum(features_track), "Map: Track sum for E19")

    # Hex E19 had a #57 added with rotation 0
    hex_coord_track = "E19"
    hex_idx_track = encoder_1830.hex_coord_to_idx[hex_coord_track]
    features_track = node_features[hex_idx_track, :]
    # Expected features for E19 (Tile 57, rotation 1)
    # Revenue: 0 -> 20.0 / 80.0
    # Type: City=1, OO=0, Town=0, Offboard=0
    # Rotation: 0
    # Tokens: NYC
    assert_float(20.0 / 80.0, features_track[get_feature_index(encoder_1830, "revenue")], "Map: Track Revenue")
    assert_float(1.0, features_track[get_feature_index(encoder_1830, "is_city")], "Map: Track is_city")
    assert_float(0.0, features_track[get_feature_index(encoder_1830, "is_oo")], "Map: Track is_oo")
    assert_float(0.0, features_track[get_feature_index(encoder_1830, "is_town")], "Map: Track is_town")
    assert_float(0.0, features_track[get_feature_index(encoder_1830, "is_offboard")], "Map: Track is_offboard")
    assert_float(0.0, features_track[get_feature_index(encoder_1830, "upgrade_cost")], "Map: Track upgrade_cost")
    assert_float(0.0, features_track[get_feature_index(encoder_1830, "rotation")], "Map: Track rotation")
    assert_float(1.0, features_track[get_feature_index(encoder_1830, "token_NYC_revenue_0")], "Map: NYC Token")
    # No edge connectivity.
    assert_float(0.0, features_track[get_feature_index(encoder_1830, "connects_5_0")], "Map: Track connects_5_0")
    assert_float(0.0, features_track[get_feature_index(encoder_1830, "connects_5_1")], "Map: Track connects_5_1")
    assert_float(0.0, features_track[get_feature_index(encoder_1830, "connects_5_2")], "Map: Track connects_5_2")
    assert_float(0.0, features_track[get_feature_index(encoder_1830, "connects_5_3")], "Map: Track connects_5_3")
    assert_float(0.0, features_track[get_feature_index(encoder_1830, "connects_5_4")], "Map: Track connects_5_4")
    assert_float(0.0, features_track[get_feature_index(encoder_1830, "connects_4_0")], "Map: Track connects_4_0")
    assert_float(0.0, features_track[get_feature_index(encoder_1830, "connects_4_1")], "Map: Track connects_4_1")
    assert_float(0.0, features_track[get_feature_index(encoder_1830, "connects_4_2")], "Map: Track connects_4_2")
    assert_float(0.0, features_track[get_feature_index(encoder_1830, "connects_4_3")], "Map: Track connects_4_3")
    assert_float(0.0, features_track[get_feature_index(encoder_1830, "connects_3_0")], "Map: Track connects_3_0")
    assert_float(0.0, features_track[get_feature_index(encoder_1830, "connects_3_1")], "Map: Track connects_3_1")
    assert_float(0.0, features_track[get_feature_index(encoder_1830, "connects_3_2")], "Map: Track connects_3_2")
    assert_float(0.0, features_track[get_feature_index(encoder_1830, "connects_2_0")], "Map: Track connects_2_0")
    assert_float(0.0, features_track[get_feature_index(encoder_1830, "connects_2_1")], "Map: Track connects_2_1")
    assert_float(0.0, features_track[get_feature_index(encoder_1830, "connects_1_0")], "Map: Track connects_1_0")
    # Ports 0 and 3 connect to revenue 0
    assert_float(
        1.0, features_track[get_feature_index(encoder_1830, "port_0_connects_revenue_0")], "Map: H10 port_0_rev_0"
    )
    assert_float(
        0.0, features_track[get_feature_index(encoder_1830, "port_1_connects_revenue_0")], "Map: H10 port_1_rev_0 (False)"
    )
    assert_float(
        0.0, features_track[get_feature_index(encoder_1830, "port_2_connects_revenue_0")], "Map: H10 port_2_rev_0 (False)"
    )
    assert_float(1.0, features_track[get_feature_index(encoder_1830, "port_3_connects_revenue_0")], "Map: H10 port_3_rev_0")
    assert_float(
        0.0, features_track[get_feature_index(encoder_1830, "port_4_connects_revenue_0")], "Map: H10 port_4_rev_0 (False)"
    )
    assert_float(
        0.0, features_track[get_feature_index(encoder_1830, "port_5_connects_revenue_0")], "Map: H10 port_5_rev_0 (False)"
    )
    assert_float(4.0 + 20.0/80.0, sum(features_track), "Map: Track sum for E19")

    # Hex G5 had a #8 added with rotation 1
    hex_coord_track = "G5"
    hex_idx_track = encoder_1830.hex_coord_to_idx[hex_coord_track]
    features_track = node_features[hex_idx_track, :]
    # Expected features for G5 (Tile 8, rotation 1)
    # Revenue: 0 -> 0
    # Type: City=0, OO=0, Town=0, Offboard=0
    # Rotation: 1
    # Tokens: None
    assert_float(0.0, features_track[get_feature_index(encoder_1830, "revenue")], "Map: Track Revenue")
    assert_float(0.0, features_track[get_feature_index(encoder_1830, "is_city")], "Map: Track is_city")
    assert_float(0.0, features_track[get_feature_index(encoder_1830, "is_oo")], "Map: Track is_oo")
    assert_float(0.0, features_track[get_feature_index(encoder_1830, "is_town")], "Map: Track is_town")
    assert_float(0.0, features_track[get_feature_index(encoder_1830, "is_offboard")], "Map: Track is_offboard")
    assert_float(0.0, features_track[get_feature_index(encoder_1830, "upgrade_cost")], "Map: Track upgrade_cost")
    assert_float(1.0, features_track[get_feature_index(encoder_1830, "rotation")], "Map: Track rotation")
    # Edge from 1 to 3
    assert_float(0.0, features_track[get_feature_index(encoder_1830, "connects_5_0")], "Map: Track connects_5_0")
    assert_float(0.0, features_track[get_feature_index(encoder_1830, "connects_5_1")], "Map: Track connects_5_1")
    assert_float(0.0, features_track[get_feature_index(encoder_1830, "connects_5_2")], "Map: Track connects_5_2")
    assert_float(0.0, features_track[get_feature_index(encoder_1830, "connects_5_3")], "Map: Track connects_5_3")
    assert_float(0.0, features_track[get_feature_index(encoder_1830, "connects_5_4")], "Map: Track connects_5_4")
    assert_float(0.0, features_track[get_feature_index(encoder_1830, "connects_4_0")], "Map: Track connects_4_0")
    assert_float(0.0, features_track[get_feature_index(encoder_1830, "connects_4_1")], "Map: Track connects_4_1")
    assert_float(0.0, features_track[get_feature_index(encoder_1830, "connects_4_2")], "Map: Track connects_4_2")
    assert_float(0.0, features_track[get_feature_index(encoder_1830, "connects_4_3")], "Map: Track connects_4_3")
    assert_float(0.0, features_track[get_feature_index(encoder_1830, "connects_3_0")], "Map: Track connects_3_0")
    assert_float(1.0, features_track[get_feature_index(encoder_1830, "connects_3_1")], "Map: Track connects_3_1")
    assert_float(0.0, features_track[get_feature_index(encoder_1830, "connects_3_2")], "Map: Track connects_3_2")
    assert_float(0.0, features_track[get_feature_index(encoder_1830, "connects_2_0")], "Map: Track connects_2_0")
    assert_float(0.0, features_track[get_feature_index(encoder_1830, "connects_2_1")], "Map: Track connects_2_1")
    assert_float(0.0, features_track[get_feature_index(encoder_1830, "connects_1_0")], "Map: Track connects_1_0")
    assert_float(2.0, sum(features_track), "Map: Track sum for G5")

    # Should also check the token on New York & Newark (G19)
    hex_coord_track = "G19"
    hex_idx_track = encoder_1830.hex_coord_to_idx[hex_coord_track]
    features_track = node_features[hex_idx_track, :]
    # Expected features for G5 (Tile 8, rotation 1)
    # Revenue: 0 -> 0
    # Type: City=0, OO=0, Town=0, Offboard=0
    # Rotation: 1
    # Tokens: None
    assert_float(40.0 / 80.0, features_track[get_feature_index(encoder_1830, "revenue")], "Map: Track Revenue")
    assert_float(1.0, features_track[get_feature_index(encoder_1830, "is_city")], "Map: Track is_city")
    assert_float(1.0, features_track[get_feature_index(encoder_1830, "is_oo")], "Map: Track is_oo")
    assert_float(0.0, features_track[get_feature_index(encoder_1830, "is_town")], "Map: Track is_town")
    assert_float(0.0, features_track[get_feature_index(encoder_1830, "is_offboard")], "Map: Track is_offboard")
    assert_float(80.0 / 120.0, features_track[get_feature_index(encoder_1830, "upgrade_cost")], "Map: Track upgrade_cost")
    assert_float(0.0, features_track[get_feature_index(encoder_1830, "rotation")], "Map: Track rotation")
    assert_float(1.0, features_track[get_feature_index(encoder_1830, "token_NYNH_revenue_1")], "Map: NYNH Token")
    # No edge connectivity.
    # Only port 0 connects to revenue 0
    assert_float(1.0, features_track[get_feature_index(encoder_1830, "port_0_connects_revenue_0")], "Map: NY port_0_rev_0")
    # Only port 3 connects to revenue 1
    assert_float(1.0, features_track[get_feature_index(encoder_1830, "port_3_connects_revenue_1")], "Map: NY port_3_rev_1")
    assert_float(5.0 + 40.0/80.0 + 80.0/120.0, sum(features_track), "Map: Track sum for G19")
    
    # MOVE TO NEXT INTERESTING POINT
    # NYNH
    operating_round_2_game_state.process_action(action_helper.get_all_choices()[0])  # lay #1 with rotation 0 on F20
    operating_round_2_game_state.process_action(action_helper.get_all_choices()[0])  # buy 2 train
    operating_round_2_game_state.process_action(action_helper.get_all_choices()[-1])  # pass trains
    # PRR
    operating_round_2_game_state.process_action(action_helper.get_all_choices()[10])  # lay tile #9 with rotation 1 on H8
    operating_round_2_game_state.process_action(action_helper.get_all_choices()[-1])  # pass token
    operating_round_2_game_state.process_action(action_helper.get_all_choices()[0])  # auto trains & run
    operating_round_2_game_state.process_action(action_helper.get_all_choices()[0])  # pay out
    operating_round_2_game_state.process_action(action_helper.get_all_choices()[-1])  # pass trains
    # C&O
    operating_round_2_game_state.process_action(action_helper.get_all_choices()[2])  # lay tile #8 with rotation 2 on G3
    operating_round_2_game_state.process_action(action_helper.get_all_choices()[0])  # auto trains & run
    operating_round_2_game_state.process_action(action_helper.get_all_choices()[1])  # withhold
    operating_round_2_game_state.process_action(action_helper.get_all_choices()[0])  # buy a 2 train
    operating_round_2_game_state.process_action(action_helper.get_all_choices()[0])  # buy a 3 train

    # Test that a 3 has been purchased
    game_state_tensor, node_features_tensor, edge_index, edge_attributes = encoder_1830.encode(operating_round_2_game_state)
    encoding = game_state_tensor.squeeze(0).numpy()
    offset = 0
    # --------------------- Check game_state ---------------------
    # # Section 1: Active Entity
    # Should be NYNH's turn in the operating round
    s_slice, offset = get_section_slice(encoder_1830, "active_entity", offset)
    assert_float(1.0, encoding[s_slice][4 + encoder_1830.corp_id_to_idx["C&O"]], "Section 1: C&O Active Entity")
    assert sum(encoding[s_slice]) == 1.0, "Section 1: C&O Active Entity"
    # # Section 2: Active President
    # P3 is president of NYNH
    s_slice, offset = get_section_slice(encoder_1830, "active_president", offset)
    assert_float(1.0, encoding[s_slice][encoder_1830.player_id_to_idx["2"]], "Section 2: Player 2 Active President")
    assert sum(encoding[s_slice]) == 1.0, "Section 2: Player 2 Active President"
    # # Section 3: Round Type
    # Operating round
    s_slice, offset = get_section_slice(encoder_1830, "round_type", offset)
    expected_round_idx = float(ROUND_TYPE_MAP["Operating"]) / MAX_ROUND_TYPE_IDX
    assert_float(expected_round_idx, encoding[s_slice].item(), "Section 3: Round Type Operating")
    # --- Section 4: Game Phase ---
    s_slice, offset = get_section_slice(encoder_1830, "game_phase", offset)
    phase_name = operating_round_2_game_state.phase.name
    expected_phase_idx = PHASE_NAMES_ORDERED.index(phase_name) / (encoder_1830.NUM_PHASES - 1)
    assert phase_name == "3", "Game should be in Phase 3 in OR 2"
    assert_float(expected_phase_idx, encoding[s_slice].item(), "Section 4: Game Phase 3")
    # --- Section 5: Priority Deal ---
    s_slice, offset = get_section_slice(encoder_1830, "priority_deal_player", offset)
    assert_float(1.0, encoding[s_slice][encoder_1830.player_id_to_idx["4"]], "Section 5: Player 4 Priority Deal")
    assert sum(encoding[s_slice]) == 1.0, "Section 5: Player 4 Priority Deal"
    # --- Section 6: Bank Cash ---
    s_slice, offset = get_section_slice(encoder_1830, "bank_cash", offset)
    assert_float(9671.0 / 12000.0, encoding[s_slice].item(), "Section 6: Bank Cash")
    # --- Section 7: Player Certs ---
    s_slice, offset = get_section_slice(encoder_1830, "player_certs_remaining", offset)
    assert_float(10.0 / 16.0, encoding[s_slice][encoder_1830.player_id_to_idx["1"]], "Section 7: Player 1 Certs")
    assert_float(14.0 / 16.0, encoding[s_slice][encoder_1830.player_id_to_idx["2"]], "Section 7: Player 2 Certs")
    assert_float(10.0 / 16.0, encoding[s_slice][encoder_1830.player_id_to_idx["3"]], "Section 7: Player 3 Certs")
    assert_float(9.0 / 16.0, encoding[s_slice][encoder_1830.player_id_to_idx["4"]], "Section 7: Player 4 Certs")
    # --- Section 8: Player Cash ---
    s_slice, offset = get_section_slice(encoder_1830, "player_cash", offset)
    assert_float(42.0 / 600.0, encoding[s_slice][encoder_1830.player_id_to_idx["1"]], "Section 8: Player 1 Cash")
    assert_float(20.0 / 600.0, encoding[s_slice][encoder_1830.player_id_to_idx["2"]], "Section 8: Player 2 Cash")
    assert_float(68.0 / 600.0, encoding[s_slice][encoder_1830.player_id_to_idx["3"]], "Section 8: Player 3 Cash")
    assert_float(69.0 / 600.0, encoding[s_slice][encoder_1830.player_id_to_idx["4"]], "Section 8: Player 4 Cash")
    # --- Section 9: Player Shares ---
    # P1 has 10% PRR, P2 has 20% B&O, others have none
    s_slice, offset = get_section_slice(encoder_1830, "player_shares", offset)
    assert_float(0.6, encoding[s_slice][encoder_1830.player_id_to_idx["1"] * 8 + encoder_1830.corp_id_to_idx["PRR"]], "Section 9: Player 1 owns 60% PRR")
    assert_float(0.3, encoding[s_slice][encoder_1830.player_id_to_idx["1"] * 8 + encoder_1830.corp_id_to_idx["NYC"]], "Section 9: Player 1 owns 30% NYC")
    assert_float(0.6, encoding[s_slice][encoder_1830.player_id_to_idx["2"] * 8 + encoder_1830.corp_id_to_idx["C&O"]], "Section 9: Player 2 owns 60% C&O")
    assert_float(0.1, encoding[s_slice][encoder_1830.player_id_to_idx["2"] * 8 + encoder_1830.corp_id_to_idx["NYNH"]], "Section 9: Player 2 owns 10% NYNH")
    assert_float(0.1, encoding[s_slice][encoder_1830.player_id_to_idx["2"] * 8 + encoder_1830.corp_id_to_idx["NYC"]], "Section 9: Player 2 owns 10% NYC")
    assert_float(0.5, encoding[s_slice][encoder_1830.player_id_to_idx["3"] * 8 + encoder_1830.corp_id_to_idx["NYNH"]], "Section 9: Player 3 owns 50% NYNH")
    assert_float(0.1, encoding[s_slice][encoder_1830.player_id_to_idx["3"] * 8 + encoder_1830.corp_id_to_idx["NYC"]], "Section 9: Player 3 owns 10% NYC")
    assert_float(0.4, encoding[s_slice][encoder_1830.player_id_to_idx["4"] * 8 + encoder_1830.corp_id_to_idx["PRR"]], "Section 9: Player 4 owns 40% PRR")
    assert_float(0.2, encoding[s_slice][encoder_1830.player_id_to_idx["4"] * 8 + encoder_1830.corp_id_to_idx["B&O"]], "Section 9: Player 4 owns 20% B&O")
    assert_float(0.1, encoding[s_slice][encoder_1830.player_id_to_idx["4"] * 8 + encoder_1830.corp_id_to_idx["C&O"]], "Section 9: Player 4 owns 10% C&O")
    assert abs(sum(encoding[s_slice]) - 3.0) < 1e-6, f"Section 9 sum: expected 3.0, got {sum(encoding[s_slice])}"
    # --- Section 10: Private Ownership ---
    s_slice, offset = get_section_slice(encoder_1830, "private_ownership", offset)
    # P1 owns SV and CA, P2 owns CS & BO, P3 owns DH, P4 owns MH
    assert_float(1.0, encoding[s_slice][encoder_1830.private_id_to_idx["SV"] * (4 + 8) + encoder_1830.player_id_to_idx["1"]], "Section 10: Player 1 owns SV")
    assert_float(1.0, encoding[s_slice][encoder_1830.private_id_to_idx["CS"] * (4 + 8) + encoder_1830.player_id_to_idx["3"]], "Section 10: Player 3 owns CS")
    assert_float(1.0, encoding[s_slice][encoder_1830.private_id_to_idx["DH"] * (4 + 8) + encoder_1830.player_id_to_idx["2"]], "Section 10: Player 2 owns DH")
    assert_float(1.0, encoding[s_slice][encoder_1830.private_id_to_idx["MH"] * (4 + 8) + encoder_1830.player_id_to_idx["3"]], "Section 10: Player 3 owns MH")
    assert_float(1.0, encoding[s_slice][encoder_1830.private_id_to_idx["CA"] * (4 + 8) + encoder_1830.player_id_to_idx["4"]], "Section 10: Player 4 owns CA")
    assert_float(1.0, encoding[s_slice][encoder_1830.private_id_to_idx["BO"] * (4 + 8) + encoder_1830.player_id_to_idx["4"]], "Section 10: Player 4 owns BO")
    assert sum(encoding[s_slice]) == 6.0, "Section 10 sum"
    # Skip Section 11: Private Revenue because it's constant
    offset += encoder_1830._get_section_size("private_revenue")
    # --- Section 12: Corp Floated ---
    # B&O floated
    s_slice, offset = get_section_slice(encoder_1830, "corp_floated", offset)
    assert_float(1.0, encoding[s_slice][encoder_1830.corp_id_to_idx["PRR"]], "Section 12: PRR floated")
    assert_float(1.0, encoding[s_slice][encoder_1830.corp_id_to_idx["C&O"]], "Section 12: C&O floated")
    assert_float(1.0, encoding[s_slice][encoder_1830.corp_id_to_idx["NYNH"]], "Section 12: NYNH floated")
    assert_float(1.0, encoding[s_slice][encoder_1830.corp_id_to_idx["NYC"]], "Section 12: NYC floated")
    assert sum(encoding[s_slice]) == 4.0, "Section 12 sum"
    # --- Section 13: Corp Cash ---
    # B&O has 1000
    s_slice, offset = get_section_slice(encoder_1830, "corp_cash", offset)
    assert_float(0.0, encoding[s_slice][encoder_1830.corp_id_to_idx["B&O"]], "Section 13: B&O Cash")
    assert_float(510.0 / 600.0, encoding[s_slice][encoder_1830.corp_id_to_idx["PRR"]], "Section 13: PRR Cash")
    assert_float(590.0 / 600.0, encoding[s_slice][encoder_1830.corp_id_to_idx["NYC"]], "Section 13: NYC Cash")
    assert_float(400.0 / 600.0, encoding[s_slice][encoder_1830.corp_id_to_idx["C&O"]], "Section 13: C&O Cash")
    assert_float(630.0 / 600.0, encoding[s_slice][encoder_1830.corp_id_to_idx["NYNH"]], "Section 13: NYNH Cash")
    assert_float(0.0, encoding[s_slice][encoder_1830.corp_id_to_idx["B&M"]], "Section 13: B&M Cash")
    # --- Section 14: Corp Trains ---
    s_slice, offset = get_section_slice(encoder_1830, "corp_trains", offset)
    assert_float(0.0, encoding[s_slice][encoder_1830.corp_id_to_idx["B&O"] * 6], "Section 14: B&O Trains")
    assert_float(2.0 / 6.0, encoding[s_slice][encoder_1830.corp_id_to_idx["PRR"] * 6], "Section 14: PRR 2 Trains")
    assert_float(1.0 / 6.0, encoding[s_slice][encoder_1830.corp_id_to_idx["NYC"] * 6], "Section 14: NYC 2 Trains")
    assert_float(2.0 / 6.0, encoding[s_slice][encoder_1830.corp_id_to_idx["C&O"] * 6], "Section 14: C&O 2 Trains")
    assert_float(1.0 / 5.0, encoding[s_slice][encoder_1830.corp_id_to_idx["C&O"] * 6 + 1], "Section 14: C&O 3 Trains")
    assert_float(1.0 / 6.0, encoding[s_slice][encoder_1830.corp_id_to_idx["NYNH"] * 6], "Section 14: NYNH 2 Trains")
    assert_float(1.2, sum(encoding[s_slice]), "Section 14 sum")
    # --- Section 15: Corp Tokens ---
    s_slice, offset = get_section_slice(encoder_1830, "corp_tokens_remaining", offset)
    assert_float(1.0, encoding[s_slice][encoder_1830.corp_id_to_idx["B&O"]], "Section 15: B&O Tokens")
    assert_float(3.0 / 4.0, encoding[s_slice][encoder_1830.corp_id_to_idx["PRR"]], "Section 15: PRR Tokens")
    assert_float(3.0 / 4.0, encoding[s_slice][encoder_1830.corp_id_to_idx["NYC"]], "Section 15: NYC Tokens")
    assert_float(2.0 / 3.0, encoding[s_slice][encoder_1830.corp_id_to_idx["C&O"]], "Section 15: C&O Tokens")
    assert_float(1.0 / 2.0, encoding[s_slice][encoder_1830.corp_id_to_idx["NYNH"]], "Section 15: NYNH Tokens")
    # --- Section 16: Corp Share Price ---
    s_slice, offset = get_section_slice(encoder_1830, "corp_share_price", offset)
    assert_float(100.0 / 350.0, encoding[s_slice][encoder_1830.corp_id_to_idx["B&O"] * 2], "Section 16: B&O IPO Price")
    assert_float(100.0 / 350.0, encoding[s_slice][encoder_1830.corp_id_to_idx["B&O"] * 2 + 1], "Section 16: B&O Share Price")
    assert_float(67.0 / 350.0, encoding[s_slice][encoder_1830.corp_id_to_idx["PRR"] * 2], "Section 16: PRR IPO Price")
    assert_float(76.0 / 350.0, encoding[s_slice][encoder_1830.corp_id_to_idx["PRR"] * 2 + 1], "Section 16: PRR Share Price")
    assert_float(67.0 / 350.0, encoding[s_slice][encoder_1830.corp_id_to_idx["NYC"] * 2], "Section 16: NYC IPO Price")
    assert_float(30.0 / 350.0, encoding[s_slice][encoder_1830.corp_id_to_idx["NYC"] * 2 + 1], "Section 16: NYC Share Price")
    assert_float(67.0 / 350.0, encoding[s_slice][encoder_1830.corp_id_to_idx["C&O"] * 2], "Section 16: C&O IPO Price")
    assert_float(58.0 / 350.0, encoding[s_slice][encoder_1830.corp_id_to_idx["C&O"] * 2 + 1], "Section 16: C&O Share Price")
    assert_float(71.0 / 350.0, encoding[s_slice][encoder_1830.corp_id_to_idx["NYNH"] * 2], "Section 16: NYNH IPO Price")
    assert_float(67.0 / 350.0, encoding[s_slice][encoder_1830.corp_id_to_idx["NYNH"] * 2 + 1], "Section 16: NYNH Share Price")
    # --- Section 17: Corp Shares ---
    s_slice, offset = get_section_slice(encoder_1830, "corp_shares", offset)
    assert_float(0.8, encoding[s_slice][encoder_1830.corp_id_to_idx["B&O"] * 2], "Section 17: B&O IPO Shares Remaining")
    assert_float(0.0, encoding[s_slice][encoder_1830.corp_id_to_idx["B&O"] * 2 + 1], "Section 17: B&O Market Shares Remaining")
    assert_float(0.0, encoding[s_slice][encoder_1830.corp_id_to_idx["PRR"] * 2], "Section 17: PRR IPO Shares Remaining")
    assert_float(0.0, encoding[s_slice][encoder_1830.corp_id_to_idx["PRR"] * 2 + 1], "Section 17: PRR Market Shares Remaining")
    assert_float(0.1, encoding[s_slice][encoder_1830.corp_id_to_idx["NYC"] * 2], "Section 17: NYC IPO Shares Remaining")
    assert_float(0.4, encoding[s_slice][encoder_1830.corp_id_to_idx["NYC"] * 2 + 1], "Section 17: NYC Market Shares Remaining")
    assert_float(0.3, encoding[s_slice][encoder_1830.corp_id_to_idx["C&O"] * 2], "Section 17: C&O IPO Shares Remaining")
    assert_float(0.0, encoding[s_slice][encoder_1830.corp_id_to_idx["C&O"] * 2 + 1], "Section 17: C&O Market Shares Remaining")
    assert_float(0.4, encoding[s_slice][encoder_1830.corp_id_to_idx["NYNH"] * 2], "Section 17: NYNH IPO Shares Remaining")
    assert_float(0.0, encoding[s_slice][encoder_1830.corp_id_to_idx["NYNH"] * 2 + 1], "Section 17: NYNH Market Shares Remaining")
    # ERIE is not parred yet
    assert_float(1.0, encoding[s_slice][encoder_1830.corp_id_to_idx["ERIE"] * 2], "Section 17: ERIE IPO Shares Remaining")
    assert_float(0.0, encoding[s_slice][encoder_1830.corp_id_to_idx["ERIE"] * 2 + 1], "Section 17: ERIE Market Shares Remaining")
    # --- Section 18: Corp Market Zone ---
    s_slice, offset = get_section_slice(encoder_1830, "corp_market_zone", offset)
    assert_float(1.0, encoding[s_slice][encoder_1830.corp_id_to_idx["B&O"] * 4], "Section 18: B&O Market Zone")
    assert_float(1.0, encoding[s_slice][encoder_1830.corp_id_to_idx["PRR"] * 4], "Section 18: PRR Market Zone")
    # NYC is in brown
    assert_float(1.0, encoding[s_slice][encoder_1830.corp_id_to_idx["NYC"] * 4 + 3], "Section 18: NYC Market Zone")
    # C&O is in yellow
    assert_float(1.0, encoding[s_slice][encoder_1830.corp_id_to_idx["C&O"] * 4 + 1], "Section 18: C&O Market Zone")
    assert_float(1.0, encoding[s_slice][encoder_1830.corp_id_to_idx["NYNH"] * 4], "Section 18: NYNH Market Zone")
    assert sum(encoding[s_slice]) == 5.0, "Section 18 sum"
    # --- Section 19: Depot Trains ---
    s_slice, offset = get_section_slice(encoder_1830, "depot_trains", offset)
    assert_float(0.0, encoding[s_slice][0], "Section 19: Depot 2 Trains")
    assert_float(4.0 / 5.0, encoding[s_slice][1], "Section 19: Depot 3 Trains")
    assert_float(1.0, encoding[s_slice][2], "Section 19: Depot 4 Trains")
    assert_float(1.0, encoding[s_slice][3], "Section 19: Depot 5 Trains")
    assert_float(1.0, encoding[s_slice][4], "Section 19: Depot 6 Trains")
    assert_float(1.0, encoding[s_slice][5], "Section 19: Depot D Trains")
    # --- Section 20: Market Pool Trains ---
    # No market pool trains
    s_slice, offset = get_section_slice(encoder_1830, "market_pool_trains", offset)
    assert_float(0.0, encoding[s_slice][0], "Section 20: Market Pool 2 Trains")
    assert_float(0.0, encoding[s_slice][1], "Section 20: Market Pool 3 Trains")
    assert_float(0.0, encoding[s_slice][2], "Section 20: Market Pool 4 Trains")
    assert_float(0.0, encoding[s_slice][3], "Section 20: Market Pool 5 Trains")
    assert_float(0.0, encoding[s_slice][4], "Section 20: Market Pool 6 Trains")
    assert_float(0.0, encoding[s_slice][5], "Section 20: Market Pool D Trains")
    # --- Section 21: Depot Tiles ---
    s_slice, offset = get_section_slice(encoder_1830, "depot_tiles", offset)
    assert_float(0.0, encoding[s_slice][encoder_1830.tile_id_to_idx["1"]], "Section 21: Depot 1 Tile")
    assert_float(6.0 / 8.0, encoding[s_slice][encoder_1830.tile_id_to_idx["8"]], "Section 21: Depot 8 Tile")
    assert_float(6.0 / 7.0, encoding[s_slice][encoder_1830.tile_id_to_idx["9"]], "Section 21: Depot 9 Tile")
    assert_float(2.0 / 4.0, encoding[s_slice][encoder_1830.tile_id_to_idx["57"]], "Section 21: Depot 57 Tile")
    expected_normalized_tile_count = len(encoder_1830.tile_id_to_idx) - 1.5 - (2.0 / 8.0) - (1.0 / 7.0)
    assert_float(expected_normalized_tile_count, sum(encoding[s_slice]), "Section 21: Normalized tile count")
    # Section A: Auction: No auction data (except private face value)
    # --- Section A1: Auction Bids ---
    s_slice, offset = get_section_slice(encoder_1830, "auction_bids", offset)
    # Whole slice is 0
    assert np.sum(encoding[s_slice]) == 0.0, "Section A: Auction Bids"
    # --- Section A2: Auction Min Bid ---
    s_slice, offset = get_section_slice(encoder_1830, "auction_min_bid", offset)
    # Whole slice is 0
    assert np.sum(encoding[s_slice]) == 0.0, "Section A: Auction Min Bid"
    # --- Section A3: Auction Available ---
    s_slice, offset = get_section_slice(encoder_1830, "auction_available", offset)
    # Whole slice is 0
    assert np.sum(encoding[s_slice]) == 0.0, "Section A: Auction Available"
    # --- Section A4: Auction Face Value ---
    s_slice, offset = get_section_slice(encoder_1830, "auction_face_value", offset)
    # Manual check
    assert_float(operating_round_2_game_state.company_by_id("SV").value / 600.0, encoding[s_slice][encoder_1830.private_id_to_idx["SV"]], "Section A: SV Face Value")
    assert_float(operating_round_2_game_state.company_by_id("CS").value / 600.0, encoding[s_slice][encoder_1830.private_id_to_idx["CS"]], "Section A: CS Face Value")
    assert_float(operating_round_2_game_state.company_by_id("DH").value / 600.0, encoding[s_slice][encoder_1830.private_id_to_idx["DH"]], "Section A: DH Face Value")
    assert_float(operating_round_2_game_state.company_by_id("MH").value / 600.0, encoding[s_slice][encoder_1830.private_id_to_idx["MH"]], "Section A: MH Face Value")
    assert_float(operating_round_2_game_state.company_by_id("CA").value / 600.0, encoding[s_slice][encoder_1830.private_id_to_idx["CA"]], "Section A: CA Face Value")
    assert_float(operating_round_2_game_state.company_by_id("BO").value / 600.0, encoding[s_slice][encoder_1830.private_id_to_idx["BO"]], "Section A: BO Face Value")

    # --- Final Offset Check ---
    assert offset == encoder_1830.ENCODING_SIZE, f"Final offset {offset} after test"

    # MOVE TO NEXT INTERESTING POINT
    operating_round_2_game_state.process_action(action_helper.get_all_choices()[-1])  # pass trains
    # Move to NYC
    operating_round_2_game_state.process_action(action_helper.get_all_choices()[-2])  # buy DH from Player 2 for $140
    operating_round_2_game_state.process_action(action_helper.get_all_choices()[0])  # pass buy companies
    # Skip to next OR
    operating_round_2_game_state.process_action(action_helper.get_all_choices()[46])  # lay tile #8 with rotation 3 on F18
    operating_round_2_game_state.process_action(action_helper.get_all_choices()[31])  # buy 3 train
    operating_round_2_game_state.process_action(action_helper.get_all_choices()[31])  # buy 3 train
    operating_round_2_game_state.process_action(action_helper.get_all_choices()[31])  # buy 3 train
    operating_round_2_game_state.process_action(action_helper.get_all_choices()[-1])  # pass
    # SR 3
    operating_round_2_game_state.process_action(action_helper.get_all_choices()[-1])  # pass
    operating_round_2_game_state.process_action(action_helper.get_all_choices()[-1])  # pass
    operating_round_2_game_state.process_action(action_helper.get_all_choices()[-1])  # pass
    operating_round_2_game_state.process_action(action_helper.get_all_choices()[-1])  # pass
    # OR 3
    operating_round_2_game_state.process_action(action_helper.get_all_choices()[45])  # lay tile 8 rot 2 on H6
    operating_round_2_game_state.process_action(action_helper.get_all_choices()[-1])  # skip token
    operating_round_2_game_state.process_action(action_helper.get_all_choices()[-1])  # auto routes
    operating_round_2_game_state.process_action(action_helper.get_all_choices()[-2])  # pay out
    operating_round_2_game_state.process_action(action_helper.get_all_choices_limited()[11])  # Buy NYC 2 509
    operating_round_2_game_state.process_action(action_helper.get_all_choices()[-1])  # pass trains
    operating_round_2_game_state.process_action(action_helper.get_all_choices_limited()[4])  # NYNH spends $80 and lays tile #57 with rotation 1 on F22 (Providence)
    operating_round_2_game_state.process_action(action_helper.get_all_choices()[-1])  # skip token
    operating_round_2_game_state.process_action(action_helper.get_all_choices()[-1])  # auto routes
    operating_round_2_game_state.process_action(action_helper.get_all_choices_limited()[4])  # pay out
    operating_round_2_game_state.process_action(action_helper.get_all_choices()[-1])  # skip trains
    operating_round_2_game_state.process_action(action_helper.get_all_choices()[-1])  # skip companies
    operating_round_2_game_state.process_action(action_helper.get_all_choices()[34])  # [17:12] C&O (DH) spends $120 and lays tile #57 with rotation 2 on F16 (Scranton)
    operating_round_2_game_state.process_action(action_helper.get_all_choices()[0])  # [17:13] C&O (DH) places a token on F16 (Scranton)

    # Check post-company action encoding
    game_state_tensor, node_features_tensor, edge_index, edge_attributes = encoder_1830.encode(operating_round_2_game_state)
    encoding = game_state_tensor.squeeze(0).numpy()
    offset = 0
    # --------------------- Check game_state ---------------------
    # # Section 1: Active Entity
    # Should be NYNH's turn in the operating round
    s_slice, offset = get_section_slice(encoder_1830, "active_entity", offset)
    assert_float(1.0, encoding[s_slice][4 + encoder_1830.corp_id_to_idx["C&O"]], "Section 1: C&O Active Entity")
    assert sum(encoding[s_slice]) == 1.0, "Section 1: C&O Active Entity"
    # # Section 2: Active President
    # P3 is president of NYNH
    s_slice, offset = get_section_slice(encoder_1830, "active_president", offset)
    assert_float(1.0, encoding[s_slice][encoder_1830.player_id_to_idx["2"]], "Section 2: Player 2 Active President")
    assert sum(encoding[s_slice]) == 1.0, "Section 2: Player 2 Active President"
    # # Section 3: Round Type
    # Operating round
    s_slice, offset = get_section_slice(encoder_1830, "round_type", offset)
    expected_round_idx = float(ROUND_TYPE_MAP["Operating"]) / MAX_ROUND_TYPE_IDX
    assert_float(expected_round_idx, encoding[s_slice].item(), "Section 3: Round Type Operating")
    # --- Section 4: Game Phase ---
    s_slice, offset = get_section_slice(encoder_1830, "game_phase", offset)
    phase_name = operating_round_2_game_state.phase.name
    expected_phase_idx = PHASE_NAMES_ORDERED.index(phase_name) / (encoder_1830.NUM_PHASES - 1)
    assert phase_name == "3", "Game should be in Phase 3 in OR 3"
    assert_float(expected_phase_idx, encoding[s_slice].item(), "Section 4: Game Phase 3")
    # --- Section 5: Priority Deal ---
    s_slice, offset = get_section_slice(encoder_1830, "priority_deal_player", offset)
    assert_float(1.0, encoding[s_slice][encoder_1830.player_id_to_idx["4"]], "Section 5: Player 4 Priority Deal")
    assert sum(encoding[s_slice]) == 1.0, "Section 5: Player 4 Priority Deal"
    # --- Section 6: Bank Cash ---
    s_slice, offset = get_section_slice(encoder_1830, "bank_cash", offset)
    assert_float(10246.0 / 12000.0, encoding[s_slice].item(), "Section 6: Bank Cash")
    # --- Section 7: Player Certs ---
    s_slice, offset = get_section_slice(encoder_1830, "player_certs_remaining", offset)
    assert_float(10.0 / 16.0, encoding[s_slice][encoder_1830.player_id_to_idx["1"]], "Section 7: Player 1 Certs")
    assert_float(15.0 / 16.0, encoding[s_slice][encoder_1830.player_id_to_idx["2"]], "Section 7: Player 2 Certs")
    assert_float(10.0 / 16.0, encoding[s_slice][encoder_1830.player_id_to_idx["3"]], "Section 7: Player 3 Certs")
    assert_float(9.0 / 16.0, encoding[s_slice][encoder_1830.player_id_to_idx["4"]], "Section 7: Player 4 Certs")
    # --- Section 8: Player Cash ---
    s_slice, offset = get_section_slice(encoder_1830, "player_cash", offset)
    assert_float(65.0 / 600.0, encoding[s_slice][encoder_1830.player_id_to_idx["1"]], "Section 8: Player 1 Cash reduced by Par")
    assert_float(165.0 / 600.0, encoding[s_slice][encoder_1830.player_id_to_idx["2"]], "Section 8: Player 2 Cash")
    assert_float(123.0 / 600.0, encoding[s_slice][encoder_1830.player_id_to_idx["3"]], "Section 8: Player 3 Cash")
    assert_float(136.0 / 600.0, encoding[s_slice][encoder_1830.player_id_to_idx["4"]], "Section 8: Player 4 Cash")
    # --- Section 9: Player Shares ---
    # P1 has 10% PRR, P2 has 20% B&O, others have none
    s_slice, offset = get_section_slice(encoder_1830, "player_shares", offset)
    assert_float(0.6, encoding[s_slice][encoder_1830.player_id_to_idx["1"] * 8 + encoder_1830.corp_id_to_idx["PRR"]], "Section 9: Player 1 owns 60% PRR")
    assert_float(0.3, encoding[s_slice][encoder_1830.player_id_to_idx["1"] * 8 + encoder_1830.corp_id_to_idx["NYC"]], "Section 9: Player 1 owns 30% NYC")
    assert_float(0.6, encoding[s_slice][encoder_1830.player_id_to_idx["2"] * 8 + encoder_1830.corp_id_to_idx["C&O"]], "Section 9: Player 2 owns 60% C&O")
    assert_float(0.1, encoding[s_slice][encoder_1830.player_id_to_idx["2"] * 8 + encoder_1830.corp_id_to_idx["NYNH"]], "Section 9: Player 2 owns 10% NYNH")
    assert_float(0.1, encoding[s_slice][encoder_1830.player_id_to_idx["2"] * 8 + encoder_1830.corp_id_to_idx["NYC"]], "Section 9: Player 2 owns 10% NYC")
    assert_float(0.5, encoding[s_slice][encoder_1830.player_id_to_idx["3"] * 8 + encoder_1830.corp_id_to_idx["NYNH"]], "Section 9: Player 3 owns 50% NYNH")
    assert_float(0.1, encoding[s_slice][encoder_1830.player_id_to_idx["3"] * 8 + encoder_1830.corp_id_to_idx["NYC"]], "Section 9: Player 3 owns 10% NYC")
    assert_float(0.4, encoding[s_slice][encoder_1830.player_id_to_idx["4"] * 8 + encoder_1830.corp_id_to_idx["PRR"]], "Section 9: Player 4 owns 40% PRR")
    assert_float(0.2, encoding[s_slice][encoder_1830.player_id_to_idx["4"] * 8 + encoder_1830.corp_id_to_idx["B&O"]], "Section 9: Player 4 owns 20% B&O")
    assert_float(0.1, encoding[s_slice][encoder_1830.player_id_to_idx["4"] * 8 + encoder_1830.corp_id_to_idx["C&O"]], "Section 9: Player 4 owns 10% C&O")
    assert_float(3.0, sum(encoding[s_slice]), "Section 9 sum")
    # --- Section 10: Private Ownership ---
    s_slice, offset = get_section_slice(encoder_1830, "private_ownership", offset)
    # P1 owns SV and CA, P2 owns CS & BO, P3 owns DH, P4 owns MH
    assert_float(1.0, encoding[s_slice][encoder_1830.private_id_to_idx["SV"] * (4 + 8) + encoder_1830.player_id_to_idx["1"]], "Section 10: Player 1 owns SV")
    assert_float(1.0, encoding[s_slice][encoder_1830.private_id_to_idx["CS"] * (4 + 8) + encoder_1830.player_id_to_idx["3"]], "Section 10: Player 3 owns CS")
    assert_float(1.0, encoding[s_slice][encoder_1830.private_id_to_idx["DH"] * (4 + 8) + 4 + encoder_1830.corp_id_to_idx["C&O"]], "Section 10: C&O owns DH")
    assert_float(1.0, encoding[s_slice][encoder_1830.private_id_to_idx["MH"] * (4 + 8) + encoder_1830.player_id_to_idx["3"]], "Section 10: Player 3 owns MH")
    assert_float(1.0, encoding[s_slice][encoder_1830.private_id_to_idx["CA"] * (4 + 8) + encoder_1830.player_id_to_idx["4"]], "Section 10: Player 4 owns CA")
    assert_float(1.0, encoding[s_slice][encoder_1830.private_id_to_idx["BO"] * (4 + 8) + encoder_1830.player_id_to_idx["4"]], "Section 10: Player 4 owns BO")
    assert sum(encoding[s_slice]) == 6.0, "Section 10 sum"
    # Skip Section 11: Private Revenue because it's constant
    offset += encoder_1830._get_section_size("private_revenue")
    # --- Section 12: Corp Floated ---
    # B&O floated
    s_slice, offset = get_section_slice(encoder_1830, "corp_floated", offset)
    assert_float(1.0, encoding[s_slice][encoder_1830.corp_id_to_idx["PRR"]], "Section 12: PRR floated")
    assert_float(1.0, encoding[s_slice][encoder_1830.corp_id_to_idx["C&O"]], "Section 12: C&O floated")
    assert_float(1.0, encoding[s_slice][encoder_1830.corp_id_to_idx["NYNH"]], "Section 12: NYNH floated")
    assert_float(1.0, encoding[s_slice][encoder_1830.corp_id_to_idx["NYC"]], "Section 12: NYC floated")
    assert sum(encoding[s_slice]) == 4.0, "Section 12 sum"
    # --- Section 13: Corp Cash ---
    # B&O has 1000
    s_slice, offset = get_section_slice(encoder_1830, "corp_cash", offset)
    assert_float(0.0, encoding[s_slice][encoder_1830.corp_id_to_idx["B&O"]], "Section 13: B&O Cash")
    assert_float(1.0 / 600.0, encoding[s_slice][encoder_1830.corp_id_to_idx["PRR"]], "Section 13: PRR Cash")
    assert_float(559.0 / 600.0, encoding[s_slice][encoder_1830.corp_id_to_idx["NYC"]], "Section 13: NYC Cash")
    assert_float(155.0 / 600.0, encoding[s_slice][encoder_1830.corp_id_to_idx["C&O"]], "Section 13: C&O Cash")
    assert_float(550.0 / 600.0, encoding[s_slice][encoder_1830.corp_id_to_idx["NYNH"]], "Section 13: NYNH Cash")
    assert_float(0.0, encoding[s_slice][encoder_1830.corp_id_to_idx["B&M"]], "Section 13: B&M Cash")
    # --- Section 14: Corp Trains ---
    s_slice, offset = get_section_slice(encoder_1830, "corp_trains", offset)
    assert_float(0.0, encoding[s_slice][encoder_1830.corp_id_to_idx["B&O"] * 6], "Section 14: B&O Trains")
    assert_float(3.0 / 6.0, encoding[s_slice][encoder_1830.corp_id_to_idx["PRR"] * 6], "Section 14: PRR 2 Trains")
    assert_float(3.0 / 5.0, encoding[s_slice][encoder_1830.corp_id_to_idx["NYC"] * 6 + 1], "Section 14: NYC 3 Trains")
    assert_float(2.0 / 6.0, encoding[s_slice][encoder_1830.corp_id_to_idx["C&O"] * 6], "Section 14: C&O 2 Trains")
    assert_float(1.0 / 5.0, encoding[s_slice][encoder_1830.corp_id_to_idx["C&O"] * 6 + 1], "Section 14: C&O 3 Trains")
    assert_float(1.0 / 6.0, encoding[s_slice][encoder_1830.corp_id_to_idx["NYNH"] * 6], "Section 14: NYNH 2 Trains")
    assert_float(1.8, sum(encoding[s_slice]), "Section 14 sum")
    # --- Section 15: Corp Tokens ---
    s_slice, offset = get_section_slice(encoder_1830, "corp_tokens_remaining", offset)
    assert_float(1.0, encoding[s_slice][encoder_1830.corp_id_to_idx["B&O"]], "Section 15: B&O Tokens")
    assert_float(3.0 / 4.0, encoding[s_slice][encoder_1830.corp_id_to_idx["PRR"]], "Section 15: PRR Tokens")
    assert_float(3.0 / 4.0, encoding[s_slice][encoder_1830.corp_id_to_idx["NYC"]], "Section 15: NYC Tokens")
    assert_float(1.0 / 3.0, encoding[s_slice][encoder_1830.corp_id_to_idx["C&O"]], "Section 15: C&O Tokens")
    assert_float(1.0 / 2.0, encoding[s_slice][encoder_1830.corp_id_to_idx["NYNH"]], "Section 15: NYNH Tokens")
    # --- Section 16: Corp Share Price ---
    s_slice, offset = get_section_slice(encoder_1830, "corp_share_price", offset)
    assert_float(100.0 / 350.0, encoding[s_slice][encoder_1830.corp_id_to_idx["B&O"] * 2], "Section 16: B&O IPO Price")
    assert_float(100.0 / 350.0, encoding[s_slice][encoder_1830.corp_id_to_idx["B&O"] * 2 + 1], "Section 16: B&O Share Price")
    assert_float(67.0 / 350.0, encoding[s_slice][encoder_1830.corp_id_to_idx["PRR"] * 2], "Section 16: PRR IPO Price")
    assert_float(90.0 / 350.0, encoding[s_slice][encoder_1830.corp_id_to_idx["PRR"] * 2 + 1], "Section 16: PRR Share Price")
    assert_float(67.0 / 350.0, encoding[s_slice][encoder_1830.corp_id_to_idx["NYC"] * 2], "Section 16: NYC IPO Price")
    assert_float(20.0 / 350.0, encoding[s_slice][encoder_1830.corp_id_to_idx["NYC"] * 2 + 1], "Section 16: NYC Share Price")
    assert_float(67.0 / 350.0, encoding[s_slice][encoder_1830.corp_id_to_idx["C&O"] * 2], "Section 16: C&O IPO Price")
    assert_float(58.0 / 350.0, encoding[s_slice][encoder_1830.corp_id_to_idx["C&O"] * 2 + 1], "Section 16: C&O Share Price")
    assert_float(71.0 / 350.0, encoding[s_slice][encoder_1830.corp_id_to_idx["NYNH"] * 2], "Section 16: NYNH IPO Price")
    assert_float(71.0 / 350.0, encoding[s_slice][encoder_1830.corp_id_to_idx["NYNH"] * 2 + 1], "Section 16: NYNH Share Price")
    # --- Section 17: Corp Shares ---
    s_slice, offset = get_section_slice(encoder_1830, "corp_shares", offset)
    assert_float(0.8, encoding[s_slice][encoder_1830.corp_id_to_idx["B&O"] * 2], "Section 17: B&O IPO Shares Remaining")
    assert_float(0.0, encoding[s_slice][encoder_1830.corp_id_to_idx["B&O"] * 2 + 1], "Section 17: B&O Market Shares Remaining")
    assert_float(0.0, encoding[s_slice][encoder_1830.corp_id_to_idx["PRR"] * 2], "Section 17: PRR IPO Shares Remaining")
    assert_float(0.0, encoding[s_slice][encoder_1830.corp_id_to_idx["PRR"] * 2 + 1], "Section 17: PRR Market Shares Remaining")
    assert_float(0.1, encoding[s_slice][encoder_1830.corp_id_to_idx["NYC"] * 2], "Section 17: NYC IPO Shares Remaining")
    assert_float(0.4, encoding[s_slice][encoder_1830.corp_id_to_idx["NYC"] * 2 + 1], "Section 17: NYC Market Shares Remaining")
    assert_float(0.3, encoding[s_slice][encoder_1830.corp_id_to_idx["C&O"] * 2], "Section 17: C&O IPO Shares Remaining")
    assert_float(0.0, encoding[s_slice][encoder_1830.corp_id_to_idx["C&O"] * 2 + 1], "Section 17: C&O Market Shares Remaining")
    assert_float(0.4, encoding[s_slice][encoder_1830.corp_id_to_idx["NYNH"] * 2], "Section 17: NYNH IPO Shares Remaining")
    assert_float(0.0, encoding[s_slice][encoder_1830.corp_id_to_idx["NYNH"] * 2 + 1], "Section 17: NYNH Market Shares Remaining")
    # ERIE is not parred yet
    assert_float(1.0, encoding[s_slice][encoder_1830.corp_id_to_idx["ERIE"] * 2], "Section 17: ERIE IPO Shares Remaining")
    assert_float(0.0, encoding[s_slice][encoder_1830.corp_id_to_idx["ERIE"] * 2 + 1], "Section 17: ERIE Market Shares Remaining")
    # --- Section 18: Corp Market Zone ---
    s_slice, offset = get_section_slice(encoder_1830, "corp_market_zone", offset)
    assert_float(1.0, encoding[s_slice][encoder_1830.corp_id_to_idx["B&O"] * 4], "Section 18: B&O Market Zone")
    assert_float(1.0, encoding[s_slice][encoder_1830.corp_id_to_idx["PRR"] * 4], "Section 18: PRR Market Zone")
    # NYC is in brown
    assert_float(1.0, encoding[s_slice][encoder_1830.corp_id_to_idx["NYC"] * 4 + 3], "Section 18: NYC Market Zone")
    # C&O is in yellow
    assert_float(1.0, encoding[s_slice][encoder_1830.corp_id_to_idx["C&O"] * 4 + 1], "Section 18: C&O Market Zone")
    assert_float(1.0, encoding[s_slice][encoder_1830.corp_id_to_idx["NYNH"] * 4], "Section 18: NYNH Market Zone")
    assert sum(encoding[s_slice]) == 5.0, "Section 18 sum"
    # --- Section 19: Depot Trains ---
    s_slice, offset = get_section_slice(encoder_1830, "depot_trains", offset)
    assert_float(0.0, encoding[s_slice][0], "Section 19: Depot 2 Trains")
    assert_float(1.0 / 5.0, encoding[s_slice][1], "Section 19: Depot 3 Trains")
    assert_float(1.0, encoding[s_slice][2], "Section 19: Depot 4 Trains")
    assert_float(1.0, encoding[s_slice][3], "Section 19: Depot 5 Trains")
    assert_float(1.0, encoding[s_slice][4], "Section 19: Depot 6 Trains")
    assert_float(1.0, encoding[s_slice][5], "Section 19: Depot D Trains")
    # --- Section 20: Market Pool Trains ---
    # No market pool trains
    s_slice, offset = get_section_slice(encoder_1830, "market_pool_trains", offset)
    assert_float(0.0, encoding[s_slice][0], "Section 20: Market Pool 2 Trains")
    assert_float(0.0, encoding[s_slice][1], "Section 20: Market Pool 3 Trains")
    assert_float(0.0, encoding[s_slice][2], "Section 20: Market Pool 4 Trains")
    assert_float(0.0, encoding[s_slice][3], "Section 20: Market Pool 5 Trains")
    assert_float(0.0, encoding[s_slice][4], "Section 20: Market Pool 6 Trains")
    assert_float(0.0, encoding[s_slice][5], "Section 20: Market Pool D Trains")
    # --- Section 21: Depot Tiles ---
    # Skip because I can't be bothered
    offset += encoder_1830._get_section_size("depot_tiles")
    # Section A: Auction: No auction data (except private face value)
    # --- Section A1: Auction Bids ---
    s_slice, offset = get_section_slice(encoder_1830, "auction_bids", offset)
    # Whole slice is 0
    assert np.sum(encoding[s_slice]) == 0.0, "Section A: Auction Bids"
    # --- Section A2: Auction Min Bid ---
    s_slice, offset = get_section_slice(encoder_1830, "auction_min_bid", offset)
    # Whole slice is 0
    assert np.sum(encoding[s_slice]) == 0.0, "Section A: Auction Min Bid"
    # --- Section A3: Auction Available ---
    s_slice, offset = get_section_slice(encoder_1830, "auction_available", offset)
    # Whole slice is 0
    assert np.sum(encoding[s_slice]) == 0.0, "Section A: Auction Available"
    # --- Section A4: Auction Face Value ---
    s_slice, offset = get_section_slice(encoder_1830, "auction_face_value", offset)
    # Manual check
    assert_float(operating_round_2_game_state.company_by_id("SV").value / 600.0, encoding[s_slice][encoder_1830.private_id_to_idx["SV"]], "Section A: SV Face Value")
    assert_float(operating_round_2_game_state.company_by_id("CS").value / 600.0, encoding[s_slice][encoder_1830.private_id_to_idx["CS"]], "Section A: CS Face Value")
    assert_float(operating_round_2_game_state.company_by_id("DH").value / 600.0, encoding[s_slice][encoder_1830.private_id_to_idx["DH"]], "Section A: DH Face Value")
    assert_float(operating_round_2_game_state.company_by_id("MH").value / 600.0, encoding[s_slice][encoder_1830.private_id_to_idx["MH"]], "Section A: MH Face Value")
    assert_float(operating_round_2_game_state.company_by_id("CA").value / 600.0, encoding[s_slice][encoder_1830.private_id_to_idx["CA"]], "Section A: CA Face Value")
    assert_float(operating_round_2_game_state.company_by_id("BO").value / 600.0, encoding[s_slice][encoder_1830.private_id_to_idx["BO"]], "Section A: BO Face Value")

    # --- Final Offset Check ---
    assert offset == encoder_1830.ENCODING_SIZE, f"Final offset {offset}"

    # Not checking map

    # MOVE TO NEXT INTERESTING POINT
    operating_round_2_game_state.process_action(action_helper.get_all_choices()[-1])  # auto routes
    operating_round_2_game_state.process_action(action_helper.get_all_choices()[0])  # pay out
    operating_round_2_game_state.process_action(action_helper.get_all_choices()[-1])  # skip trains
    
    operating_round_2_game_state.process_action(action_helper.get_all_choices_limited()[23]) # [17:13] NYC spends $80 and lays tile #54 with rotation 0 on G19 (New York & Newark)
    operating_round_2_game_state.process_action(action_helper.get_all_choices()[-1])  # auto routes
    operating_round_2_game_state.process_action(action_helper.get_all_choices_limited()[2])  # pay out
    operating_round_2_game_state.process_action(action_helper.get_all_choices_limited()[2]) # Buy 3 train
    operating_round_2_game_state.process_action(action_helper.get_all_choices()[-1])  # skip companies
    # PRR
    operating_round_2_game_state.process_action(action_helper.get_all_choices_limited()[-1])  # skip track
    operating_round_2_game_state.process_action(action_helper.get_all_choices_limited()[-1])  # run trains
    operating_round_2_game_state.process_action(action_helper.get_all_choices_limited()[0])  # pay out
    operating_round_2_game_state.process_action(action_helper.get_all_choices_limited()[-1])  # skip trains
    # NYNH
    operating_round_2_game_state.process_action(action_helper.get_all_choices_limited()[-1])  # lay tile
    operating_round_2_game_state.process_action(action_helper.get_all_choices_limited()[-1])  # skip token
    operating_round_2_game_state.process_action(action_helper.get_all_choices_limited()[-1])  # run trains
    operating_round_2_game_state.process_action(action_helper.get_all_choices_limited()[4])  # pay out
    operating_round_2_game_state.process_action(action_helper.get_all_choices_limited()[4])  # buy a 4 train
    operating_round_2_game_state.process_action(action_helper.get_all_choices_limited()[0])  # NYC discard train

    # Check upgraded tile and discarded train encodings
    game_state_tensor, node_features_tensor, edge_index, edge_attributes = encoder_1830.encode(operating_round_2_game_state)
    encoding = game_state_tensor.squeeze(0).numpy()
    offset = 0
    # --------------------- Check game_state ---------------------
    # # Section 1: Active Entity
    # Should be NYNH's turn in the operating round
    s_slice, offset = get_section_slice(encoder_1830, "active_entity", offset)
    assert_float(1.0, encoding[s_slice][4 + encoder_1830.corp_id_to_idx["NYNH"]], "Section 1: NYNH Active Entity")
    assert sum(encoding[s_slice]) == 1.0, "Section 1: NYNH Active Entity"
    # # Section 2: Active President
    # P3 is president of NYNH
    s_slice, offset = get_section_slice(encoder_1830, "active_president", offset)
    assert_float(1.0, encoding[s_slice][encoder_1830.player_id_to_idx["3"]], "Section 2: Player 3 Active President")
    assert sum(encoding[s_slice]) == 1.0, "Section 2: Player 3 Active President"
    # # Section 3: Round Type
    # Operating round
    s_slice, offset = get_section_slice(encoder_1830, "round_type", offset)
    expected_round_idx = float(ROUND_TYPE_MAP["Operating"]) / MAX_ROUND_TYPE_IDX
    assert_float(expected_round_idx, encoding[s_slice].item(), "Section 3: Round Type Operating")
    # --- Section 4: Game Phase ---
    s_slice, offset = get_section_slice(encoder_1830, "game_phase", offset)
    phase_name = operating_round_2_game_state.phase.name
    expected_phase_idx = PHASE_NAMES_ORDERED.index(phase_name) / (encoder_1830.NUM_PHASES - 1)
    assert phase_name == "4", f"Game should be in Phase 4 in OR 3, got {phase_name}"
    assert_float(expected_phase_idx, encoding[s_slice].item(), "Section 4: Game Phase 4")
    # --- Section 5: Priority Deal ---
    s_slice, offset = get_section_slice(encoder_1830, "priority_deal_player", offset)
    assert_float(1.0, encoding[s_slice][encoder_1830.player_id_to_idx["4"]], "Section 5: Player 4 Priority Deal")
    assert sum(encoding[s_slice]) == 1.0, "Section 5: Player 4 Priority Deal"
    # --- Section 6: Bank Cash ---
    s_slice, offset = get_section_slice(encoder_1830, "bank_cash", offset)
    assert_float(10502.0 / 12000.0, encoding[s_slice].item(), "Section 6: Bank Cash")
    # --- Section 7: Player Certs ---
    s_slice, offset = get_section_slice(encoder_1830, "player_certs_remaining", offset)
    assert_float(10.0 / 16.0, encoding[s_slice][encoder_1830.player_id_to_idx["1"]], "Section 7: Player 1 Certs")
    assert_float(10.0 / 16.0, encoding[s_slice][encoder_1830.player_id_to_idx["2"]], "Section 7: Player 2 Certs")
    assert_float(10.0 / 16.0, encoding[s_slice][encoder_1830.player_id_to_idx["3"]], "Section 7: Player 3 Certs")
    assert_float(8.0 / 16.0, encoding[s_slice][encoder_1830.player_id_to_idx["4"]], "Section 7: Player 4 Certs")
    # --- Section 8: Player Cash ---
    s_slice, offset = get_section_slice(encoder_1830, "player_cash", offset)
    assert_float(112.0 / 600.0, encoding[s_slice][encoder_1830.player_id_to_idx["1"]], "Section 8: Player 1 Cash reduced by Par")
    assert_float(223.0 / 600.0, encoding[s_slice][encoder_1830.player_id_to_idx["2"]], "Section 8: Player 2 Cash")
    assert_float(201.0 / 600.0, encoding[s_slice][encoder_1830.player_id_to_idx["3"]], "Section 8: Player 3 Cash")
    assert_float(210.0 / 600.0, encoding[s_slice][encoder_1830.player_id_to_idx["4"]], "Section 8: Player 4 Cash")
    # --- Section 9: Player Shares ---
    # P1 has 10% PRR, P2 has 20% B&O, others have none
    s_slice, offset = get_section_slice(encoder_1830, "player_shares", offset)
    assert_float(0.6, encoding[s_slice][encoder_1830.player_id_to_idx["1"] * 8 + encoder_1830.corp_id_to_idx["PRR"]], "Section 9: Player 1 owns 60% PRR")
    assert_float(0.3, encoding[s_slice][encoder_1830.player_id_to_idx["1"] * 8 + encoder_1830.corp_id_to_idx["NYC"]], "Section 9: Player 1 owns 30% NYC")
    assert_float(0.6, encoding[s_slice][encoder_1830.player_id_to_idx["2"] * 8 + encoder_1830.corp_id_to_idx["C&O"]], "Section 9: Player 2 owns 60% C&O")
    assert_float(0.1, encoding[s_slice][encoder_1830.player_id_to_idx["2"] * 8 + encoder_1830.corp_id_to_idx["NYNH"]], "Section 9: Player 2 owns 10% NYNH")
    assert_float(0.1, encoding[s_slice][encoder_1830.player_id_to_idx["2"] * 8 + encoder_1830.corp_id_to_idx["NYC"]], "Section 9: Player 2 owns 10% NYC")
    assert_float(0.5, encoding[s_slice][encoder_1830.player_id_to_idx["3"] * 8 + encoder_1830.corp_id_to_idx["NYNH"]], "Section 9: Player 3 owns 50% NYNH")
    assert_float(0.1, encoding[s_slice][encoder_1830.player_id_to_idx["3"] * 8 + encoder_1830.corp_id_to_idx["NYC"]], "Section 9: Player 3 owns 10% NYC")
    assert_float(0.4, encoding[s_slice][encoder_1830.player_id_to_idx["4"] * 8 + encoder_1830.corp_id_to_idx["PRR"]], "Section 9: Player 4 owns 40% PRR")
    assert_float(0.2, encoding[s_slice][encoder_1830.player_id_to_idx["4"] * 8 + encoder_1830.corp_id_to_idx["B&O"]], "Section 9: Player 4 owns 20% B&O")
    assert_float(0.1, encoding[s_slice][encoder_1830.player_id_to_idx["4"] * 8 + encoder_1830.corp_id_to_idx["C&O"]], "Section 9: Player 4 owns 10% C&O")
    assert abs(sum(encoding[s_slice]) - 3.0) < 1e-6, f"Section 9 sum: expected 3.0, got {sum(encoding[s_slice])}"
    # --- Section 10: Private Ownership ---
    s_slice, offset = get_section_slice(encoder_1830, "private_ownership", offset)
    # P1 owns SV and CA, P2 owns CS & BO, P3 owns DH, P4 owns MH
    assert_float(1.0, encoding[s_slice][encoder_1830.private_id_to_idx["SV"] * (4 + 8) + encoder_1830.player_id_to_idx["1"]], "Section 10: Player 1 owns SV")
    assert_float(1.0, encoding[s_slice][encoder_1830.private_id_to_idx["CS"] * (4 + 8) + encoder_1830.player_id_to_idx["3"]], "Section 10: Player 3 owns CS")
    assert_float(1.0, encoding[s_slice][encoder_1830.private_id_to_idx["DH"] * (4 + 8) + 4 + encoder_1830.corp_id_to_idx["C&O"]], "Section 10: C&O owns DH")
    assert_float(1.0, encoding[s_slice][encoder_1830.private_id_to_idx["MH"] * (4 + 8) + encoder_1830.player_id_to_idx["3"]], "Section 10: Player 3 owns MH")
    assert_float(1.0, encoding[s_slice][encoder_1830.private_id_to_idx["CA"] * (4 + 8) + encoder_1830.player_id_to_idx["4"]], "Section 10: Player 4 owns CA")
    assert_float(1.0, encoding[s_slice][encoder_1830.private_id_to_idx["BO"] * (4 + 8) + encoder_1830.player_id_to_idx["4"]], "Section 10: Player 4 owns BO")
    assert sum(encoding[s_slice]) == 6.0, "Section 10 sum"
    # Skip Section 11: Private Revenue because it's constant
    offset += encoder_1830._get_section_size("private_revenue")
    # --- Section 12: Corp Floated ---
    # B&O floated
    s_slice, offset = get_section_slice(encoder_1830, "corp_floated", offset)
    assert_float(1.0, encoding[s_slice][encoder_1830.corp_id_to_idx["PRR"]], "Section 12: PRR floated")
    assert_float(1.0, encoding[s_slice][encoder_1830.corp_id_to_idx["C&O"]], "Section 12: C&O floated")
    assert_float(1.0, encoding[s_slice][encoder_1830.corp_id_to_idx["NYNH"]], "Section 12: NYNH floated")
    assert_float(1.0, encoding[s_slice][encoder_1830.corp_id_to_idx["NYC"]], "Section 12: NYC floated")
    assert sum(encoding[s_slice]) == 4.0, "Section 12 sum"
    # --- Section 13: Corp Cash ---
    # B&O has 1000
    s_slice, offset = get_section_slice(encoder_1830, "corp_cash", offset)
    assert_float(0.0, encoding[s_slice][encoder_1830.corp_id_to_idx["B&O"]], "Section 13: B&O Cash")
    assert_float(1.0 / 600.0, encoding[s_slice][encoder_1830.corp_id_to_idx["PRR"]], "Section 13: PRR Cash")
    assert_float(331.0 / 600.0, encoding[s_slice][encoder_1830.corp_id_to_idx["NYC"]], "Section 13: NYC Cash")
    assert_float(170.0 / 600.0, encoding[s_slice][encoder_1830.corp_id_to_idx["C&O"]], "Section 13: C&O Cash")
    assert_float(250.0 / 600.0, encoding[s_slice][encoder_1830.corp_id_to_idx["NYNH"]], "Section 13: NYNH Cash")
    assert_float(0.0, encoding[s_slice][encoder_1830.corp_id_to_idx["B&M"]], "Section 13: B&M Cash")
    # --- Section 14: Corp Trains ---
    s_slice, offset = get_section_slice(encoder_1830, "corp_trains", offset)
    assert_float(0.0, encoding[s_slice][encoder_1830.corp_id_to_idx["B&O"] * 6], "Section 14: B&O Trains")
    assert_float(3.0 / 5.0, encoding[s_slice][encoder_1830.corp_id_to_idx["NYC"] * 6 + 1], "Section 14: NYC 3 Trains")
    assert_float(1.0 / 5.0, encoding[s_slice][encoder_1830.corp_id_to_idx["C&O"] * 6 + 1], "Section 14: C&O 3 Trains")
    assert_float(1.0 / 4.0, encoding[s_slice][encoder_1830.corp_id_to_idx["NYNH"] * 6 + 2], "Section 14: NYNH 4 Trains")
    assert sum(encoding[s_slice]) == 1.05, "Section 14 sum"
    # --- Section 15: Corp Tokens ---
    s_slice, offset = get_section_slice(encoder_1830, "corp_tokens_remaining", offset)
    assert_float(1.0, encoding[s_slice][encoder_1830.corp_id_to_idx["B&O"]], "Section 15: B&O Tokens")
    assert_float(3.0 / 4.0, encoding[s_slice][encoder_1830.corp_id_to_idx["PRR"]], "Section 15: PRR Tokens")
    assert_float(3.0 / 4.0, encoding[s_slice][encoder_1830.corp_id_to_idx["NYC"]], "Section 15: NYC Tokens")
    assert_float(1.0 / 3.0, encoding[s_slice][encoder_1830.corp_id_to_idx["C&O"]], "Section 15: C&O Tokens")
    assert_float(1.0 / 2.0, encoding[s_slice][encoder_1830.corp_id_to_idx["NYNH"]], "Section 15: NYNH Tokens")
    # --- Section 16: Corp Share Price ---
    s_slice, offset = get_section_slice(encoder_1830, "corp_share_price", offset)
    assert_float(100.0 / 350.0, encoding[s_slice][encoder_1830.corp_id_to_idx["B&O"] * 2], "Section 16: B&O IPO Price")
    assert_float(100.0 / 350.0, encoding[s_slice][encoder_1830.corp_id_to_idx["B&O"] * 2 + 1], "Section 16: B&O Share Price")
    assert_float(67.0 / 350.0, encoding[s_slice][encoder_1830.corp_id_to_idx["PRR"] * 2], "Section 16: PRR IPO Price")
    assert_float(100.0 / 350.0, encoding[s_slice][encoder_1830.corp_id_to_idx["PRR"] * 2 + 1], "Section 16: PRR Share Price")
    assert_float(67.0 / 350.0, encoding[s_slice][encoder_1830.corp_id_to_idx["NYC"] * 2], "Section 16: NYC IPO Price")
    assert_float(30.0 / 350.0, encoding[s_slice][encoder_1830.corp_id_to_idx["NYC"] * 2 + 1], "Section 16: NYC Share Price")
    assert_float(67.0 / 350.0, encoding[s_slice][encoder_1830.corp_id_to_idx["C&O"] * 2], "Section 16: C&O IPO Price")
    assert_float(65.0 / 350.0, encoding[s_slice][encoder_1830.corp_id_to_idx["C&O"] * 2 + 1], "Section 16: C&O Share Price")
    assert_float(71.0 / 350.0, encoding[s_slice][encoder_1830.corp_id_to_idx["NYNH"] * 2], "Section 16: NYNH IPO Price")
    assert_float(76.0 / 350.0, encoding[s_slice][encoder_1830.corp_id_to_idx["NYNH"] * 2 + 1], "Section 16: NYNH Share Price")
    # --- Section 17: Corp Shares ---
    s_slice, offset = get_section_slice(encoder_1830, "corp_shares", offset)
    assert_float(0.8, encoding[s_slice][encoder_1830.corp_id_to_idx["B&O"] * 2], "Section 17: B&O IPO Shares Remaining")
    assert_float(0.0, encoding[s_slice][encoder_1830.corp_id_to_idx["B&O"] * 2 + 1], "Section 17: B&O Market Shares Remaining")
    assert_float(0.0, encoding[s_slice][encoder_1830.corp_id_to_idx["PRR"] * 2], "Section 17: PRR IPO Shares Remaining")
    assert_float(0.0, encoding[s_slice][encoder_1830.corp_id_to_idx["PRR"] * 2 + 1], "Section 17: PRR Market Shares Remaining")
    assert_float(0.1, encoding[s_slice][encoder_1830.corp_id_to_idx["NYC"] * 2], "Section 17: NYC IPO Shares Remaining")
    assert_float(0.4, encoding[s_slice][encoder_1830.corp_id_to_idx["NYC"] * 2 + 1], "Section 17: NYC Market Shares Remaining")
    assert_float(0.3, encoding[s_slice][encoder_1830.corp_id_to_idx["C&O"] * 2], "Section 17: C&O IPO Shares Remaining")
    assert_float(0.0, encoding[s_slice][encoder_1830.corp_id_to_idx["C&O"] * 2 + 1], "Section 17: C&O Market Shares Remaining")
    assert_float(0.4, encoding[s_slice][encoder_1830.corp_id_to_idx["NYNH"] * 2], "Section 17: NYNH IPO Shares Remaining")
    assert_float(0.0, encoding[s_slice][encoder_1830.corp_id_to_idx["NYNH"] * 2 + 1], "Section 17: NYNH Market Shares Remaining")
    # ERIE is not parred yet
    assert_float(1.0, encoding[s_slice][encoder_1830.corp_id_to_idx["ERIE"] * 2], "Section 17: ERIE IPO Shares Remaining")
    assert_float(0.0, encoding[s_slice][encoder_1830.corp_id_to_idx["ERIE"] * 2 + 1], "Section 17: ERIE Market Shares Remaining")
    # --- Section 18: Corp Market Zone ---
    s_slice, offset = get_section_slice(encoder_1830, "corp_market_zone", offset)
    assert_float(1.0, encoding[s_slice][encoder_1830.corp_id_to_idx["B&O"] * 4], "Section 18: B&O Market Zone")
    assert_float(1.0, encoding[s_slice][encoder_1830.corp_id_to_idx["PRR"] * 4], "Section 18: PRR Market Zone")
    # NYC is in brown
    assert_float(1.0, encoding[s_slice][encoder_1830.corp_id_to_idx["NYC"] * 4 + 3], "Section 18: NYC Market Zone")
    assert_float(1.0, encoding[s_slice][encoder_1830.corp_id_to_idx["C&O"] * 4], "Section 18: C&O Market Zone")
    assert_float(1.0, encoding[s_slice][encoder_1830.corp_id_to_idx["NYNH"] * 4], "Section 18: NYNH Market Zone")
    assert sum(encoding[s_slice]) == 5.0, "Section 18 sum"
    # --- Section 19: Depot Trains ---
    s_slice, offset = get_section_slice(encoder_1830, "depot_trains", offset)
    assert_float(0.0, encoding[s_slice][0], "Section 19: Depot 2 Trains")
    assert_float(0.0, encoding[s_slice][1], "Section 19: Depot 3 Trains")
    assert_float(3.0 / 4.0, encoding[s_slice][2], "Section 19: Depot 4 Trains")
    assert_float(1.0, encoding[s_slice][3], "Section 19: Depot 5 Trains")
    assert_float(1.0, encoding[s_slice][4], "Section 19: Depot 6 Trains")
    assert_float(1.0, encoding[s_slice][5], "Section 19: Depot D Trains")
    # --- Section 20: Market Pool Trains ---
    # No market pool trains
    s_slice, offset = get_section_slice(encoder_1830, "market_pool_trains", offset)
    assert_float(0.0, encoding[s_slice][0], "Section 20: Market Pool 2 Trains")
    assert_float(1.0 / 5.0, encoding[s_slice][1], "Section 20: Market Pool 3 Trains")
    assert_float(0.0, encoding[s_slice][2], "Section 20: Market Pool 4 Trains")
    assert_float(0.0, encoding[s_slice][3], "Section 20: Market Pool 5 Trains")
    assert_float(0.0, encoding[s_slice][4], "Section 20: Market Pool 6 Trains")
    assert_float(0.0, encoding[s_slice][5], "Section 20: Market Pool D Trains")
    # --- Section 21: Depot Tiles ---
    s_slice, offset = get_section_slice(encoder_1830, "depot_tiles", offset)
    assert_float(0.0, encoding[s_slice][encoder_1830.tile_id_to_idx["1"]], "Section 21: Depot 1 Tile")
    assert_float(4.0 / 8.0, encoding[s_slice][encoder_1830.tile_id_to_idx["8"]], "Section 21: Depot 8 Tile")
    assert_float(6.0 / 7.0, encoding[s_slice][encoder_1830.tile_id_to_idx["9"]], "Section 21: Depot 9 Tile")
    assert_float(0.0, encoding[s_slice][encoder_1830.tile_id_to_idx["57"]], "Section 21: Depot 54 Tile")
    assert_float(0.0, encoding[s_slice][encoder_1830.tile_id_to_idx["57"]], "Section 21: Depot 57 Tile")
    expected_normalized_tile_count = len(encoder_1830.tile_id_to_idx) - 3.5 - (1.0 / 7.0)
    assert_float(expected_normalized_tile_count, sum(encoding[s_slice]), "Section 21: Normalized tile count")
    # Section A: Auction: No auction data (except private face value)
    # --- Section A1: Auction Bids ---
    s_slice, offset = get_section_slice(encoder_1830, "auction_bids", offset)
    # Whole slice is 0
    assert np.sum(encoding[s_slice]) == 0.0, "Section A: Auction Bids"
    # --- Section A2: Auction Min Bid ---
    s_slice, offset = get_section_slice(encoder_1830, "auction_min_bid", offset)
    # Whole slice is 0
    assert np.sum(encoding[s_slice]) == 0.0, "Section A: Auction Min Bid"
    # --- Section A3: Auction Available ---
    s_slice, offset = get_section_slice(encoder_1830, "auction_available", offset)
    # Whole slice is 0
    assert np.sum(encoding[s_slice]) == 0.0, "Section A: Auction Available"
    # --- Section A4: Auction Face Value ---
    s_slice, offset = get_section_slice(encoder_1830, "auction_face_value", offset)
    # Manual check
    assert_float(operating_round_2_game_state.company_by_id("SV").value / 600.0, encoding[s_slice][encoder_1830.private_id_to_idx["SV"]], "Section A: SV Face Value")
    assert_float(operating_round_2_game_state.company_by_id("CS").value / 600.0, encoding[s_slice][encoder_1830.private_id_to_idx["CS"]], "Section A: CS Face Value")
    assert_float(operating_round_2_game_state.company_by_id("DH").value / 600.0, encoding[s_slice][encoder_1830.private_id_to_idx["DH"]], "Section A: DH Face Value")
    assert_float(operating_round_2_game_state.company_by_id("MH").value / 600.0, encoding[s_slice][encoder_1830.private_id_to_idx["MH"]], "Section A: MH Face Value")
    assert_float(operating_round_2_game_state.company_by_id("CA").value / 600.0, encoding[s_slice][encoder_1830.private_id_to_idx["CA"]], "Section A: CA Face Value")
    assert_float(operating_round_2_game_state.company_by_id("BO").value / 600.0, encoding[s_slice][encoder_1830.private_id_to_idx["BO"]], "Section A: BO Face Value")

    # --- Final Offset Check ---
    assert offset == encoder_1830.ENCODING_SIZE, f"Final offset {offset}"

    # --------------------- Check node_features ---------------------
    node_features = node_features_tensor.squeeze(0).numpy()
    # Hex G19 was upgraded to tile 54
    hex_coord_track = "G19"
    hex_idx_track = encoder_1830.hex_coord_to_idx[hex_coord_track]
    features_track = node_features[hex_idx_track, :]
    # Expected features for G19 (Tile 54, rotation 0)
    # Revenue: 0 -> 60.0 / 80.0
    # Type: City=1, OO=1, Town=0, Offboard=0
    # Rotation: 0
    # Tokens: NYNH in 0
    assert_float(60.0 / 80.0, features_track[get_feature_index(encoder_1830, "revenue")], "Map: Track Revenue")
    assert_float(1.0, features_track[get_feature_index(encoder_1830, "is_city")], "Map: Track is_city")
    assert_float(1.0, features_track[get_feature_index(encoder_1830, "is_oo")], "Map: Track is_oo")
    assert_float(0.0, features_track[get_feature_index(encoder_1830, "is_town")], "Map: Track is_town")
    assert_float(0.0, features_track[get_feature_index(encoder_1830, "is_offboard")], "Map: Track is_offboard")
    assert_float(0.0, features_track[get_feature_index(encoder_1830, "upgrade_cost")], "Map: Track upgrade_cost")
    assert_float(0.0, features_track[get_feature_index(encoder_1830, "rotation")], "Map: Track rotation")
    assert_float(1.0, features_track[get_feature_index(encoder_1830, "token_NYNH_revenue_1")], "Map: NYNH Token")
    # No edge connectivity.
    # Ports 0 and 1 connect to revenue 0
    assert_float(
        1.0, features_track[get_feature_index(encoder_1830, "port_0_connects_revenue_0")], "Map: G19 port_0_rev_0"
    )
    assert_float(1.0, features_track[get_feature_index(encoder_1830, "port_1_connects_revenue_0")], "Map: G19 port_1_rev_0")
    # Ports 2 and 3 connect to revenue 1
    assert_float(
        1.0, features_track[get_feature_index(encoder_1830, "port_2_connects_revenue_1")], "Map: G19 port_2_rev_1"
    )
    assert_float(1.0, features_track[get_feature_index(encoder_1830, "port_3_connects_revenue_1")], "Map: G19 port_3_rev_1")
    assert_float(7.0 + 60.0/80.0, sum(features_track), "Map: Track sum for G19")

    # Check OO town on F20
    hex_coord_track = "F20"
    hex_idx_track = encoder_1830.hex_coord_to_idx[hex_coord_track]
    features_track = node_features[hex_idx_track, :]
    # Expected features for F20 (Tile 1, rotation 0)
    # Revenue: 0 -> 10.0 / 80.0
    # Type: City=0, OO=1, Town=1, Offboard=0
    # Rotation: 0
    # Tokens: None
    assert_float(10.0 / 80.0, features_track[get_feature_index(encoder_1830, "revenue")], "Map: Track Revenue")
    assert_float(0.0, features_track[get_feature_index(encoder_1830, "is_city")], "Map: Track is_city")
    assert_float(1.0, features_track[get_feature_index(encoder_1830, "is_oo")], "Map: Track is_oo")
    assert_float(1.0, features_track[get_feature_index(encoder_1830, "is_town")], "Map: Track is_town")
    assert_float(0.0, features_track[get_feature_index(encoder_1830, "is_offboard")], "Map: Track is_offboard")
    assert_float(0.0, features_track[get_feature_index(encoder_1830, "upgrade_cost")], "Map: Track upgrade_cost")
    assert_float(0.0, features_track[get_feature_index(encoder_1830, "rotation")], "Map: Track rotation")
    # No edge connectivity.
    # Ports 1 and 3 connect to revenue 0
    assert_float(
        1.0, features_track[get_feature_index(encoder_1830, "port_1_connects_revenue_0")], "Map: G19 port_1_rev_0"
    )
    assert_float(1.0, features_track[get_feature_index(encoder_1830, "port_3_connects_revenue_0")], "Map: G19 port_3_rev_0")
    # Ports 0 and 4 connect to revenue 1
    assert_float(
        1.0, features_track[get_feature_index(encoder_1830, "port_0_connects_revenue_1")], "Map: G19 port_0_rev_1"
    )
    assert_float(1.0, features_track[get_feature_index(encoder_1830, "port_4_connects_revenue_1")], "Map: G19 port_4_rev_1")
    assert_float(6.0 + 10.0/80.0, sum(features_track), "Map: Track sum for F20")

    # MOVE TO END
    operating_round_2_game_state.process_action(action_helper.get_all_choices_limited()[-2])  # Buy discarded train

    # Check final state
    game_state_tensor, node_features_tensor, edge_index, edge_attributes = encoder_1830.encode(operating_round_2_game_state)
    encoding = game_state_tensor.squeeze(0).numpy()
    offset = 0
    # --------------------- Check game_state ---------------------
    # # Section 1: Active Entity
    # Should be NYNH's turn in the operating round
    s_slice, offset = get_section_slice(encoder_1830, "active_entity", offset)
    assert_float(1.0, encoding[s_slice][4 + encoder_1830.corp_id_to_idx["NYNH"]], "Section 1: NYNH Active Entity")
    assert sum(encoding[s_slice]) == 1.0, "Section 1: NYNH Active Entity"
    # # Section 2: Active President
    # P3 is president of NYNH
    s_slice, offset = get_section_slice(encoder_1830, "active_president", offset)
    assert_float(1.0, encoding[s_slice][encoder_1830.player_id_to_idx["3"]], "Section 2: Player 3 Active President")
    assert sum(encoding[s_slice]) == 1.0, "Section 2: Player 3 Active President"
    # # Section 3: Round Type
    # Operating round
    s_slice, offset = get_section_slice(encoder_1830, "round_type", offset)
    expected_round_idx = float(ROUND_TYPE_MAP["Operating"]) / MAX_ROUND_TYPE_IDX
    assert_float(expected_round_idx, encoding[s_slice].item(), "Section 3: Round Type Operating")
    # --- Section 4: Game Phase ---
    s_slice, offset = get_section_slice(encoder_1830, "game_phase", offset)
    phase_name = operating_round_2_game_state.phase.name
    expected_phase_idx = PHASE_NAMES_ORDERED.index(phase_name) / (encoder_1830.NUM_PHASES - 1)
    assert phase_name == "4", "Game should be in Phase 4 in OR 3"
    assert_float(expected_phase_idx, encoding[s_slice].item(), "Section 4: Game Phase 4")
    # --- Section 5: Priority Deal ---
    s_slice, offset = get_section_slice(encoder_1830, "priority_deal_player", offset)
    assert_float(1.0, encoding[s_slice][encoder_1830.player_id_to_idx["4"]], "Section 5: Player 4 Priority Deal")
    assert sum(encoding[s_slice]) == 1.0, "Section 5: Player 4 Priority Deal"
    # --- Section 6: Bank Cash ---
    s_slice, offset = get_section_slice(encoder_1830, "bank_cash", offset)
    assert_float(10682.0 / 12000.0, encoding[s_slice].item(), "Section 6: Bank Cash")
    # --- Section 7: Player Certs ---
    s_slice, offset = get_section_slice(encoder_1830, "player_certs_remaining", offset)
    assert_float(10.0 / 16.0, encoding[s_slice][encoder_1830.player_id_to_idx["1"]], "Section 7: Player 1 Certs")
    assert_float(10.0 / 16.0, encoding[s_slice][encoder_1830.player_id_to_idx["2"]], "Section 7: Player 2 Certs")
    assert_float(10.0 / 16.0, encoding[s_slice][encoder_1830.player_id_to_idx["3"]], "Section 7: Player 3 Certs")
    assert_float(8.0 / 16.0, encoding[s_slice][encoder_1830.player_id_to_idx["4"]], "Section 7: Player 4 Certs")
    # --- Section 8: Player Cash ---
    s_slice, offset = get_section_slice(encoder_1830, "player_cash", offset)
    assert_float(112.0 / 600.0, encoding[s_slice][encoder_1830.player_id_to_idx["1"]], "Section 8: Player 1 Cash")
    assert_float(223.0 / 600.0, encoding[s_slice][encoder_1830.player_id_to_idx["2"]], "Section 8: Player 2 Cash")
    assert_float(201.0 / 600.0, encoding[s_slice][encoder_1830.player_id_to_idx["3"]], "Section 8: Player 3 Cash")
    assert_float(210.0 / 600.0, encoding[s_slice][encoder_1830.player_id_to_idx["4"]], "Section 8: Player 4 Cash")
    # --- Section 9: Player Shares ---
    # P1 has 10% PRR, P2 has 20% B&O, others have none
    s_slice, offset = get_section_slice(encoder_1830, "player_shares", offset)
    assert_float(0.6, encoding[s_slice][encoder_1830.player_id_to_idx["1"] * 8 + encoder_1830.corp_id_to_idx["PRR"]], "Section 9: Player 1 owns 60% PRR")
    assert_float(0.3, encoding[s_slice][encoder_1830.player_id_to_idx["1"] * 8 + encoder_1830.corp_id_to_idx["NYC"]], "Section 9: Player 1 owns 30% NYC")
    assert_float(0.6, encoding[s_slice][encoder_1830.player_id_to_idx["2"] * 8 + encoder_1830.corp_id_to_idx["C&O"]], "Section 9: Player 2 owns 60% C&O")
    assert_float(0.1, encoding[s_slice][encoder_1830.player_id_to_idx["2"] * 8 + encoder_1830.corp_id_to_idx["NYNH"]], "Section 9: Player 2 owns 10% NYNH")
    assert_float(0.1, encoding[s_slice][encoder_1830.player_id_to_idx["2"] * 8 + encoder_1830.corp_id_to_idx["NYC"]], "Section 9: Player 2 owns 10% NYC")
    assert_float(0.5, encoding[s_slice][encoder_1830.player_id_to_idx["3"] * 8 + encoder_1830.corp_id_to_idx["NYNH"]], "Section 9: Player 3 owns 50% NYNH")
    assert_float(0.1, encoding[s_slice][encoder_1830.player_id_to_idx["3"] * 8 + encoder_1830.corp_id_to_idx["NYC"]], "Section 9: Player 3 owns 10% NYC")
    assert_float(0.4, encoding[s_slice][encoder_1830.player_id_to_idx["4"] * 8 + encoder_1830.corp_id_to_idx["PRR"]], "Section 9: Player 4 owns 40% PRR")
    assert_float(0.2, encoding[s_slice][encoder_1830.player_id_to_idx["4"] * 8 + encoder_1830.corp_id_to_idx["B&O"]], "Section 9: Player 4 owns 20% B&O")
    assert_float(0.1, encoding[s_slice][encoder_1830.player_id_to_idx["4"] * 8 + encoder_1830.corp_id_to_idx["C&O"]], "Section 9: Player 4 owns 10% C&O")
    assert abs(sum(encoding[s_slice]) - 3.0) < 1e-6, f"Section 9 sum: expected 3.0, got {sum(encoding[s_slice])}"
    # --- Section 10: Private Ownership ---
    s_slice, offset = get_section_slice(encoder_1830, "private_ownership", offset)
    # P1 owns SV and CA, P2 owns CS & BO, P3 owns DH, P4 owns MH
    assert_float(1.0, encoding[s_slice][encoder_1830.private_id_to_idx["SV"] * (4 + 8) + encoder_1830.player_id_to_idx["1"]], "Section 10: Player 1 owns SV")
    assert_float(1.0, encoding[s_slice][encoder_1830.private_id_to_idx["CS"] * (4 + 8) + encoder_1830.player_id_to_idx["3"]], "Section 10: Player 3 owns CS")
    assert_float(1.0, encoding[s_slice][encoder_1830.private_id_to_idx["DH"] * (4 + 8) + 4 + encoder_1830.corp_id_to_idx["C&O"]], "Section 10: C&O owns DH")
    assert_float(1.0, encoding[s_slice][encoder_1830.private_id_to_idx["MH"] * (4 + 8) + encoder_1830.player_id_to_idx["3"]], "Section 10: Player 3 owns MH")
    assert_float(1.0, encoding[s_slice][encoder_1830.private_id_to_idx["CA"] * (4 + 8) + encoder_1830.player_id_to_idx["4"]], "Section 10: Player 4 owns CA")
    assert_float(1.0, encoding[s_slice][encoder_1830.private_id_to_idx["BO"] * (4 + 8) + encoder_1830.player_id_to_idx["4"]], "Section 10: Player 4 owns BO")
    assert sum(encoding[s_slice]) == 6.0, "Section 10 sum"
    # Skip Section 11: Private Revenue because it's constant
    offset += encoder_1830._get_section_size("private_revenue")
    # --- Section 12: Corp Floated ---
    # B&O floated
    s_slice, offset = get_section_slice(encoder_1830, "corp_floated", offset)
    assert_float(1.0, encoding[s_slice][encoder_1830.corp_id_to_idx["PRR"]], "Section 12: PRR floated")
    assert_float(1.0, encoding[s_slice][encoder_1830.corp_id_to_idx["C&O"]], "Section 12: C&O floated")
    assert_float(1.0, encoding[s_slice][encoder_1830.corp_id_to_idx["NYNH"]], "Section 12: NYNH floated")
    assert_float(1.0, encoding[s_slice][encoder_1830.corp_id_to_idx["NYC"]], "Section 12: NYC floated")
    assert sum(encoding[s_slice]) == 4.0, "Section 12 sum"
    # --- Section 13: Corp Cash ---
    # B&O has 1000
    s_slice, offset = get_section_slice(encoder_1830, "corp_cash", offset)
    assert_float(0.0, encoding[s_slice][encoder_1830.corp_id_to_idx["B&O"]], "Section 13: B&O Cash")
    assert_float(1.0 / 600.0, encoding[s_slice][encoder_1830.corp_id_to_idx["PRR"]], "Section 13: PRR Cash")
    assert_float(331.0 / 600.0, encoding[s_slice][encoder_1830.corp_id_to_idx["NYC"]], "Section 13: NYC Cash")
    assert_float(170.0 / 600.0, encoding[s_slice][encoder_1830.corp_id_to_idx["C&O"]], "Section 13: C&O Cash")
    assert_float(70.0 / 600.0, encoding[s_slice][encoder_1830.corp_id_to_idx["NYNH"]], "Section 13: NYNH Cash")
    assert_float(0.0, encoding[s_slice][encoder_1830.corp_id_to_idx["B&M"]], "Section 13: B&M Cash")
    # --- Section 14: Corp Trains ---
    s_slice, offset = get_section_slice(encoder_1830, "corp_trains", offset)
    assert_float(0.0, encoding[s_slice][encoder_1830.corp_id_to_idx["B&O"] * 6], "Section 14: B&O Trains")
    assert_float(3.0 / 5.0, encoding[s_slice][encoder_1830.corp_id_to_idx["NYC"] * 6 + 1], "Section 14: NYC 3 Trains")
    assert_float(1.0 / 5.0, encoding[s_slice][encoder_1830.corp_id_to_idx["C&O"] * 6 + 1], "Section 14: C&O 3 Trains")
    assert_float(1.0 / 5.0, encoding[s_slice][encoder_1830.corp_id_to_idx["NYNH"] * 6 + 1], "Section 14: NYNH 3 Trains")
    assert_float(1.0 / 4.0, encoding[s_slice][encoder_1830.corp_id_to_idx["NYNH"] * 6 + 2], "Section 14: NYNH 4 Trains")
    assert sum(encoding[s_slice]) == 1.25, f"Section 14 sum: Expected 1.25, got {sum(encoding[s_slice])}"
    # --- Section 15: Corp Tokens ---
    s_slice, offset = get_section_slice(encoder_1830, "corp_tokens_remaining", offset)
    assert_float(1.0, encoding[s_slice][encoder_1830.corp_id_to_idx["B&O"]], "Section 15: B&O Tokens")
    assert_float(3.0 / 4.0, encoding[s_slice][encoder_1830.corp_id_to_idx["PRR"]], "Section 15: PRR Tokens")
    assert_float(3.0 / 4.0, encoding[s_slice][encoder_1830.corp_id_to_idx["NYC"]], "Section 15: NYC Tokens")
    assert_float(1.0 / 3.0, encoding[s_slice][encoder_1830.corp_id_to_idx["C&O"]], "Section 15: C&O Tokens")
    assert_float(1.0 / 2.0, encoding[s_slice][encoder_1830.corp_id_to_idx["NYNH"]], "Section 15: NYNH Tokens")
    # --- Section 16: Corp Share Price ---
    s_slice, offset = get_section_slice(encoder_1830, "corp_share_price", offset)
    assert_float(100.0 / 350.0, encoding[s_slice][encoder_1830.corp_id_to_idx["B&O"] * 2], "Section 16: B&O IPO Price")
    assert_float(100.0 / 350.0, encoding[s_slice][encoder_1830.corp_id_to_idx["B&O"] * 2 + 1], "Section 16: B&O Share Price")
    assert_float(67.0 / 350.0, encoding[s_slice][encoder_1830.corp_id_to_idx["PRR"] * 2], "Section 16: PRR IPO Price")
    assert_float(100.0 / 350.0, encoding[s_slice][encoder_1830.corp_id_to_idx["PRR"] * 2 + 1], "Section 16: PRR Share Price")
    assert_float(67.0 / 350.0, encoding[s_slice][encoder_1830.corp_id_to_idx["NYC"] * 2], "Section 16: NYC IPO Price")
    assert_float(30.0 / 350.0, encoding[s_slice][encoder_1830.corp_id_to_idx["NYC"] * 2 + 1], "Section 16: NYC Share Price")
    assert_float(67.0 / 350.0, encoding[s_slice][encoder_1830.corp_id_to_idx["C&O"] * 2], "Section 16: C&O IPO Price")
    assert_float(65.0 / 350.0, encoding[s_slice][encoder_1830.corp_id_to_idx["C&O"] * 2 + 1], "Section 16: C&O Share Price")
    assert_float(71.0 / 350.0, encoding[s_slice][encoder_1830.corp_id_to_idx["NYNH"] * 2], "Section 16: NYNH IPO Price")
    assert_float(76.0 / 350.0, encoding[s_slice][encoder_1830.corp_id_to_idx["NYNH"] * 2 + 1], "Section 16: NYNH Share Price")
    # --- Section 17: Corp Shares ---
    s_slice, offset = get_section_slice(encoder_1830, "corp_shares", offset)
    assert_float(0.8, encoding[s_slice][encoder_1830.corp_id_to_idx["B&O"] * 2], "Section 17: B&O IPO Shares Remaining")
    assert_float(0.0, encoding[s_slice][encoder_1830.corp_id_to_idx["B&O"] * 2 + 1], "Section 17: B&O Market Shares Remaining")
    assert_float(0.0, encoding[s_slice][encoder_1830.corp_id_to_idx["PRR"] * 2], "Section 17: PRR IPO Shares Remaining")
    assert_float(0.0, encoding[s_slice][encoder_1830.corp_id_to_idx["PRR"] * 2 + 1], "Section 17: PRR Market Shares Remaining")
    assert_float(0.1, encoding[s_slice][encoder_1830.corp_id_to_idx["NYC"] * 2], "Section 17: NYC IPO Shares Remaining")
    assert_float(0.4, encoding[s_slice][encoder_1830.corp_id_to_idx["NYC"] * 2 + 1], "Section 17: NYC Market Shares Remaining")
    assert_float(0.3, encoding[s_slice][encoder_1830.corp_id_to_idx["C&O"] * 2], "Section 17: C&O IPO Shares Remaining")
    assert_float(0.0, encoding[s_slice][encoder_1830.corp_id_to_idx["C&O"] * 2 + 1], "Section 17: C&O Market Shares Remaining")
    assert_float(0.4, encoding[s_slice][encoder_1830.corp_id_to_idx["NYNH"] * 2], "Section 17: NYNH IPO Shares Remaining")
    assert_float(0.0, encoding[s_slice][encoder_1830.corp_id_to_idx["NYNH"] * 2 + 1], "Section 17: NYNH Market Shares Remaining")
    # ERIE is not parred yet
    assert_float(1.0, encoding[s_slice][encoder_1830.corp_id_to_idx["ERIE"] * 2], "Section 17: ERIE IPO Shares Remaining")
    assert_float(0.0, encoding[s_slice][encoder_1830.corp_id_to_idx["ERIE"] * 2 + 1], "Section 17: ERIE Market Shares Remaining")
    # --- Section 18: Corp Market Zone ---
    s_slice, offset = get_section_slice(encoder_1830, "corp_market_zone", offset)
    assert_float(1.0, encoding[s_slice][encoder_1830.corp_id_to_idx["B&O"] * 4], "Section 18: B&O Market Zone")
    assert_float(1.0, encoding[s_slice][encoder_1830.corp_id_to_idx["PRR"] * 4], "Section 18: PRR Market Zone")
    # NYC is in brown
    assert_float(1.0, encoding[s_slice][encoder_1830.corp_id_to_idx["NYC"] * 4 + 3], "Section 18: NYC Market Zone")
    assert_float(1.0, encoding[s_slice][encoder_1830.corp_id_to_idx["C&O"] * 4], "Section 18: C&O Market Zone")
    assert_float(1.0, encoding[s_slice][encoder_1830.corp_id_to_idx["NYNH"] * 4], "Section 18: NYNH Market Zone")
    assert sum(encoding[s_slice]) == 5.0, "Section 18 sum"
    # --- Section 19: Depot Trains ---
    s_slice, offset = get_section_slice(encoder_1830, "depot_trains", offset)
    assert_float(0.0, encoding[s_slice][0], "Section 19: Depot 2 Trains")
    assert_float(0.0, encoding[s_slice][1], "Section 19: Depot 3 Trains")
    assert_float(3.0 / 4.0, encoding[s_slice][2], "Section 19: Depot 4 Trains")
    assert_float(1.0, encoding[s_slice][3], "Section 19: Depot 5 Trains")
    assert_float(1.0, encoding[s_slice][4], "Section 19: Depot 6 Trains")
    assert_float(1.0, encoding[s_slice][5], "Section 19: Depot D Trains")
    # --- Section 20: Market Pool Trains ---
    # No market pool trains
    s_slice, offset = get_section_slice(encoder_1830, "market_pool_trains", offset)
    assert_float(0.0, encoding[s_slice][0], "Section 20: Market Pool 2 Trains")
    assert_float(0.0, encoding[s_slice][1], "Section 20: Market Pool 3 Trains")
    assert_float(0.0, encoding[s_slice][2], "Section 20: Market Pool 4 Trains")
    assert_float(0.0, encoding[s_slice][3], "Section 20: Market Pool 5 Trains")
    assert_float(0.0, encoding[s_slice][4], "Section 20: Market Pool 6 Trains")
    assert_float(0.0, encoding[s_slice][5], "Section 20: Market Pool D Trains")
    # --- Section 21: Depot Tiles ---
    s_slice, offset = get_section_slice(encoder_1830, "depot_tiles", offset)
    assert_float(0.0, encoding[s_slice][encoder_1830.tile_id_to_idx["1"]], "Section 21: Depot 1 Tile")
    assert_float(4.0 / 8.0, encoding[s_slice][encoder_1830.tile_id_to_idx["8"]], "Section 21: Depot 8 Tile")
    assert_float(6.0 / 7.0, encoding[s_slice][encoder_1830.tile_id_to_idx["9"]], "Section 21: Depot 9 Tile")
    assert_float(0.0, encoding[s_slice][encoder_1830.tile_id_to_idx["57"]], "Section 21: Depot 54 Tile")
    assert_float(0.0, encoding[s_slice][encoder_1830.tile_id_to_idx["57"]], "Section 21: Depot 57 Tile")
    expected_normalized_tile_count = len(encoder_1830.tile_id_to_idx) - 3.5 - (1.0 / 7.0)
    assert_float(expected_normalized_tile_count, sum(encoding[s_slice]), "Section 21: Normalized tile count")
    # Section A: Auction: No auction data (except private face value)
    # --- Section A1: Auction Bids ---
    s_slice, offset = get_section_slice(encoder_1830, "auction_bids", offset)
    # Whole slice is 0
    assert np.sum(encoding[s_slice]) == 0.0, "Section A: Auction Bids"
    # --- Section A2: Auction Min Bid ---
    s_slice, offset = get_section_slice(encoder_1830, "auction_min_bid", offset)
    # Whole slice is 0
    assert np.sum(encoding[s_slice]) == 0.0, "Section A: Auction Min Bid"
    # --- Section A3: Auction Available ---
    s_slice, offset = get_section_slice(encoder_1830, "auction_available", offset)
    # Whole slice is 0
    assert np.sum(encoding[s_slice]) == 0.0, "Section A: Auction Available"
    # --- Section A4: Auction Face Value ---
    s_slice, offset = get_section_slice(encoder_1830, "auction_face_value", offset)
    # Manual check
    assert_float(operating_round_2_game_state.company_by_id("SV").value / 600.0, encoding[s_slice][encoder_1830.private_id_to_idx["SV"]], "Section A: SV Face Value")
    assert_float(operating_round_2_game_state.company_by_id("CS").value / 600.0, encoding[s_slice][encoder_1830.private_id_to_idx["CS"]], "Section A: CS Face Value")
    assert_float(operating_round_2_game_state.company_by_id("DH").value / 600.0, encoding[s_slice][encoder_1830.private_id_to_idx["DH"]], "Section A: DH Face Value")
    assert_float(operating_round_2_game_state.company_by_id("MH").value / 600.0, encoding[s_slice][encoder_1830.private_id_to_idx["MH"]], "Section A: MH Face Value")
    assert_float(operating_round_2_game_state.company_by_id("CA").value / 600.0, encoding[s_slice][encoder_1830.private_id_to_idx["CA"]], "Section A: CA Face Value")
    assert_float(operating_round_2_game_state.company_by_id("BO").value / 600.0, encoding[s_slice][encoder_1830.private_id_to_idx["BO"]], "Section A: BO Face Value")

    # --- Final Offset Check ---
    assert offset == encoder_1830.ENCODING_SIZE, f"Final offset {offset}"