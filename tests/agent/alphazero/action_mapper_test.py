import pytest
import torch
from unittest.mock import MagicMock, patch

# --- Imports from rl18xx library ---
# Adjust paths as necessary based on your project structure
from rl18xx.agent.alphazero.action_mapper import ActionMapper, ACTION_ENCODING_SIZE
from rl18xx.game.engine.actions import Pass, Bid, Par, BaseAction
from rl18xx.game.engine.game import BaseGame
from rl18xx.game.gamemap import GameMap
from rl18xx.game.engine.entities import Player, Company, Corporation
from rl18xx.game import ActionHelper

# --- Test Fixtures ---


@pytest.fixture
def action_mapper():
    """Provides an ActionMapper instance for testing."""
    return ActionMapper()


@pytest.fixture
def real_game_state():
    game_map = GameMap()
    game = game_map.game_by_title("1830")
    return game({"1": "Player 1", "2": "Player 2", "3": "Player 3", "4": "Player 4"})


@pytest.fixture
def mock_game_state():
    """Creates a mock BaseGame state."""
    state = MagicMock(spec=BaseGame)
    state.current_entity = MagicMock(spec=Player)

    # Mock companies and corporations needed for map_index_to_action
    mock_sv_priv = Company(sym="SV", value=20)
    mock_cs_priv = Company(sym="CS", value=45)
    mock_dh_priv = Company(sym="DH", value=75)
    mock_mh_priv = Company(sym="MH", value=115)
    mock_ca_priv = Company(sym="CA", value=165)
    mock_bo_priv = Company(sym="BO", value=225)
    state.companies = [mock_sv_priv, mock_cs_priv, mock_dh_priv, mock_mh_priv, mock_ca_priv, mock_bo_priv]
    company_map = {
        "SV": mock_sv_priv,
        "CS": mock_cs_priv,
        "DH": mock_dh_priv,
        "MH": mock_mh_priv,
        "CA": mock_ca_priv,
        "BO": mock_bo_priv,
    }

    mock_prr_corp = Corporation(sym="PRR")
    mock_nyc_corp = Corporation(sym="NYC")
    mock_cpr_corp = Corporation(sym="CPR")
    mock_b_o_corp = Corporation(sym="B&O")
    mock_c_o_corp = Corporation(sym="C&O")
    mock_erie_corp = Corporation(sym="ERIE")
    mock_nynh_corp = Corporation(sym="NYNH")
    mock_b_m_corp = Corporation(sym="B&M")
    state.corporations = [
        mock_prr_corp,
        mock_nyc_corp,
        mock_cpr_corp,
        mock_b_o_corp,
        mock_c_o_corp,
        mock_erie_corp,
        mock_nynh_corp,
        mock_b_m_corp,
    ]
    corporation_map = {
        "PRR": mock_prr_corp,
        "NYC": mock_nyc_corp,
        "CPR": mock_cpr_corp,
        "B&O": mock_b_o_corp,
        "C&O": mock_c_o_corp,
        "ERIE": mock_erie_corp,
        "NYNH": mock_nynh_corp,
        "B&M": mock_b_m_corp,
    }
    # Mock active_step and min_bid methods
    mock_step = MagicMock()

    def min_bid(company):
        return company.value

    mock_step.min_bid = MagicMock(side_effect=min_bid)
    state.active_step = MagicMock(return_value=mock_step)

    # Mock get_by_id
    def company_by_id(company_id):
        return company_map.get(company_id, Company(sym="UNKNOWN"))

    state.company_by_id = MagicMock(side_effect=company_by_id)

    def corporation_by_id(corp_id):
        return corporation_map.get(corp_id, Corporation(sym="UNKNOWN"))

    state.corporation_by_id = MagicMock(side_effect=corporation_by_id)
    return state


# Helper to create actions for testing _get_index_for_action
def create_action(game_state, action_class, target_id=None, price=None):
    # Use correct attribute names based on action type
    if action_class is Bid:
        company = game_state.company_by_id(target_id)
        action = action_class(game_state.current_entity, price, company=company)
    elif action_class is Par:
        corporation = game_state.corporation_by_id(target_id)
        action = action_class(game_state.current_entity, corporation, price)
    elif action_class is Pass:
        action = action_class(game_state.current_entity)
    else:
        raise ValueError(f"Unknown action type: {action_class}")

    return action


# --- Test Cases ---


# Test with real game
def test_get_index_for_action_real_game(action_mapper, real_game_state):
    action_helper = ActionHelper(real_game_state)
    purchase_action = action_helper.get_all_choices_limited()[0]
    bid_action = action_helper.get_all_choices_limited()[1]
    pass_action = action_helper.get_all_choices_limited()[-1]

    assert action_mapper._get_index_for_action(purchase_action) == 1
    assert action_mapper._get_index_for_action(bid_action) == 2
    assert action_mapper._get_index_for_action(pass_action) == 0

    fake_purchase_action = create_action(real_game_state, Bid, "SV")
    fake_bid_action = create_action(real_game_state, Bid, "CS")
    fake_pass_action = create_action(real_game_state, Pass)
    assert action_mapper._get_index_for_action(fake_purchase_action) == 1
    assert action_mapper._get_index_for_action(fake_bid_action) == 2
    assert action_mapper._get_index_for_action(fake_pass_action) == 0

    mask = action_mapper.get_legal_action_mask(real_game_state)
    assert mask.shape == (ACTION_ENCODING_SIZE,)
    assert mask.dtype == torch.float32
    assert mask[0] == 1.0
    assert mask[1] == 1.0
    assert mask[2] == 1.0
    assert mask[3] == 1.0
    assert mask[4] == 1.0
    assert mask[5] == 1.0
    assert mask[6] == 1.0
    assert mask[7] == 0.0
    assert mask[8] == 0.0
    assert mask[9] == 0.0
    assert mask[10] == 0.0
    assert mask[11] == 0.0
    assert mask[12] == 0.0


# Test _get_index_for_action (internal method, but crucial)
@pytest.mark.parametrize(
    "action_details, expected_index",
    [
        ((Pass,), 0),
        ((Bid, "SV"), 1),
        ((Bid, "CS"), 2),
        ((Bid, "DH"), 3),
        ((Bid, "MH"), 4),
        ((Bid, "CA"), 5),
        ((Bid, "BO"), 6),  # Bid on private B&O
        ((Par, "B&O", 67), 7),  # Par public B&O
        ((Par, "B&O", 71), 8),
        ((Par, "B&O", 76), 9),
        ((Par, "B&O", 82), 10),
        ((Par, "B&O", 90), 11),
        ((Par, "B&O", 100), 12),
    ],
)
def test_get_index_for_action_valid(action_mapper, mock_game_state, action_details, expected_index):
    """Tests mapping valid action objects to their correct indices."""
    action_class = action_details[0]
    params = action_details[1:]
    mock_action = create_action(mock_game_state, action_class, *params)
    # Add assertion for debugging
    assert action_mapper._get_index_for_action(mock_action) == expected_index


@pytest.mark.parametrize(
    "action_details",
    [
        (Bid, "Unknown"),  # Unknown Bid company
        (Par, "Unknown", 100),  # Unknown Par company
        (Par, "B&O", 99),  # Unknown Par price for B&O
        (MagicMock,),  # Completely unknown action type (using MagicMock as placeholder)
    ],
)
def test_get_index_for_action_invalid(mock_game_state, action_mapper, action_details):
    """Tests that invalid or unknown actions raise ValueError."""
    action_class = action_details[0]
    params = action_details[1:]
    # Handle case where action_class might not be a real class spec
    if action_class is MagicMock:
        # Create a generic mock that won't match known types
        mock_action = MagicMock()
        mock_action.__class__ = MagicMock  # Explicitly set class
    else:
        mock_action = create_action(mock_game_state, action_class, *params)

    with pytest.raises(ValueError):
        action_mapper._get_index_for_action(mock_action)


# Test map_index_to_action
@pytest.mark.parametrize(
    "index, expected_action_type, expected_params",
    [
        (0, Pass, ()),
        (1, Bid, ("SV", 20)),
        (6, Bid, ("BO", 225)),
        (7, Par, ("B&O", 67)),
        (12, Par, ("B&O", 100)),
    ],
)
def test_map_index_to_action_valid(action_mapper, mock_game_state, index, expected_action_type, expected_params):
    """Tests mapping valid indices back to action objects."""
    # No need to patch constructors anymore, we check the returned object
    action_obj = action_mapper.map_index_to_action(index, mock_game_state)

    assert isinstance(action_obj, expected_action_type)
    assert action_obj.entity == mock_game_state.current_entity

    # Check specific attributes based on type
    if expected_action_type is Bid:
        assert action_obj.company.id == expected_params[0]
        assert action_obj.price == expected_params[1]  # Check the price used
    elif expected_action_type is Par:
        assert action_obj.corporation.id == expected_params[0]
        assert action_obj.share_price == expected_params[1]
    elif expected_action_type is Pass:
        pass  # Already checked entity and type


def test_map_index_to_action_invalid(action_mapper, mock_game_state):
    """Tests that out-of-bounds indices raise IndexError."""
    # Check upper bound
    with pytest.raises(IndexError):
        action_mapper.map_index_to_action(ACTION_ENCODING_SIZE, mock_game_state)
    # Check lower bound
    with pytest.raises(IndexError):
        action_mapper.map_index_to_action(-1, mock_game_state)


# Test get_legal_action_mask
@pytest.mark.parametrize(
    "legal_actions_details, expected_mask_indices",
    [
        # Scenario 1: Only Pass legal
        ([(Pass,)], [0]),
        # Scenario 2: Pass and Bid on CS legal
        ([(Pass,), (Bid, "CS")], [0, 2]),
        # Scenario 3: Pass and Par B&O at 90 legal
        ([(Pass,), (Par, "B&O", 90)], [0, 11]),
        # Scenario 4: Multiple Bids legal
        ([(Pass,), (Bid, "DH"), (Bid, "CA")], [0, 3, 5]),
        # Scenario 5: All Par actions legal (hypothetical)
        ([(Par, "B&O", p) for p in [67, 71, 76, 82, 90, 100]], [7, 8, 9, 10, 11, 12]),
        # Scenario 6: Empty list
        ([], []),
        # Scenario 7: Only Bids legal
        ([(Bid, "SV"), (Bid, "MH")], [1, 4]),
    ],
)
def test_get_legal_action_mask(action_mapper, mock_game_state, mocker, legal_actions_details, expected_mask_indices):
    """Tests generating the action mask based on mocked legal actions."""
    mock_actions = [create_action(mock_game_state, details[0], *details[1:]) for details in legal_actions_details]

    # --- Mocking the ActionHelper ---
    # Patch the ActionHelper class *instantiation* within the mapper module
    mock_helper_instance = MagicMock(spec=ActionHelper)
    mock_helper_instance.get_all_choices.return_value = mock_actions
    # Make sure the path matches where ActionHelper is imported/used in action_mapper.py
    mocker.patch("rl18xx.agent.alphazero.action_mapper.ActionHelper", return_value=mock_helper_instance)
    # --- End Mocking ---

    mask = action_mapper.get_legal_action_mask(mock_game_state)

    expected_mask = torch.zeros(ACTION_ENCODING_SIZE, dtype=torch.float32)
    if expected_mask_indices:
        indices_tensor = torch.tensor(expected_mask_indices, dtype=torch.long)
        # Use float 1.0 for src, should match mask dtype
        expected_mask.scatter_(0, indices_tensor, 1.0)

    assert torch.equal(mask, expected_mask), f"Mask mismatch.\nExpected: {expected_mask}\nGot: {mask}"
    assert mask.shape == (ACTION_ENCODING_SIZE,)
    assert mask.dtype == torch.float32
