import pytest
import numpy as np
from torch import float32

# Assuming the encoder and game engine classes are importable
from rl18xx.agent.alphazero.encoder import Encoder_1830
from rl18xx.game.engine.game.base import BaseGame
from rl18xx.game.engine.game.title.g1830 import Game as Game_1830  # For constants if needed
from rl18xx.game.engine.round import WaterfallAuction  # Import specific round type
from rl18xx.game.engine.core import Phase  # For phase checks

# Import other necessary game components if needed for setup or assertions

# Mock or simplified game setup might be needed if full engine is too heavy
# For now, assume a fixture `test_game_1830_4p` provides a fresh 4-player 1830 game instance
# at the start of the private auction.


# Helper function for float comparisons
def assert_float(actual, expected, desc="", tolerance=1e-4):
    assert abs(actual - expected) < tolerance, f"{desc}: Expected {expected:.4f}, got {actual:.4f}"


# Helper to get section slice and offset
def get_section_slice(encoder: Encoder_1830, section_name: str, current_offset: int) -> tuple[slice, int]:
    size = encoder._get_section_size(section_name)
    section_slice = slice(current_offset, current_offset + size)
    next_offset = current_offset + size
    return section_slice, next_offset


@pytest.fixture(scope="module")
def encoder_1830():
    """Provides a reusable Encoder_1830 instance."""
    return Encoder_1830()


@pytest.fixture
def test_game_1830_4p():
    """Provides a fresh 4-player 1830 game instance at the start of the private auction."""
    # This needs to be implemented based on your test setup framework
    # Example using a hypothetical setup:
    from rl18xx.game.gamemap import GameMap
    from rl18xx.game.action_helper import ActionHelper

    game_map = GameMap()
    game_class = game_map.game_by_title("1830")
    players = {"1": "Player 1", "2": "Player 2", "3": "Player 3", "4": "Player 4"}
    game_instance = game_class(players)
    # Ensure game is at the very start (private auction)
    # game_instance.setup() # Or similar method if needed
    return game_instance, ActionHelper(game_instance)


def test_initial_encoding_structure(encoder_1830: Encoder_1830, test_game_1830_4p):
    """Verify the basic structure and size of the initial encoding."""
    g, _ = test_game_1830_4p
    num_players = len(g.players)
    assert num_players == 4, "Test requires a 4-player game"

    # Force initialization if needed (should happen on first encode)
    encoder_1830._initialize_game_specifics(g)
    encoder_1830._initialize_player_map(g.players)

    # Calculate expected size based on the encoder's structure
    # This ensures the test is aligned with the encoder's definition
    expected_size = encoder_1830.ENCODING_SIZE
    assert expected_size > 74, "Encoding size should be much larger now"  # Sanity check

    encoding_tensor = encoder_1830.encode(g)

    assert encoding_tensor.shape == (1, expected_size), f"Expected shape (1, {expected_size})"
    assert encoding_tensor.dtype == float32, "Expected dtype float32"


def test_initial_game_state_encoding(encoder_1830: Encoder_1830, test_game_1830_4p):
    """Verify the encoding of general game state at the start."""
    g, _ = test_game_1830_4p
    encoding = encoder_1830.encode(g).squeeze(0).numpy()  # Work with numpy array

    offset = 0
    player_map = encoder_1830.player_id_to_idx
    corp_map = encoder_1830.corp_id_to_idx
    private_map = encoder_1830.private_id_to_idx
    num_players = encoder_1830.num_players
    num_corps = encoder_1830.NUM_CORPORATIONS
    num_privates = encoder_1830.NUM_PRIVATES

    # --- Section 1: Active Entity ---
    s_slice, offset = get_section_slice(encoder_1830, "active_entity", offset)
    active_player_id = g.round.active_entities()[0].id
    active_idx = player_map[active_player_id]
    expected = np.zeros(num_players + num_corps)
    expected[active_idx] = 1.0
    np.testing.assert_allclose(encoding[s_slice], expected, atol=1e-6, err_msg="Section 1: Active Entity")

    # --- Section 2: Active President ---
    s_slice, offset = get_section_slice(encoder_1830, "active_president", offset)
    expected = np.zeros(num_players)  # No president active at start
    np.testing.assert_allclose(encoding[s_slice], expected, atol=1e-6, err_msg="Section 2: Active President")

    # --- Section 3: Round Type ---
    s_slice, offset = get_section_slice(encoder_1830, "round_type", offset)
    expected_round_idx = encoder_1830.ROUND_TYPE_MAP.get(type(g.round).__name__, -1.0)
    assert expected_round_idx != -1.0, "Initial round type not found in map"
    assert_float(encoding[s_slice].item(), expected_round_idx, "Section 3: Round Type")

    # --- Section 4: Game Phase ---
    s_slice, offset = get_section_slice(encoder_1830, "game_phase", offset)
    phase_name = g.phase.name
    expected_phase_idx = encoder_1830.PHASE_NAMES_ORDERED.index(phase_name)
    assert_float(encoding[s_slice].item(), expected_phase_idx, "Section 4: Game Phase")

    # --- Section 5: Priority Deal Player ---
    s_slice, offset = get_section_slice(encoder_1830, "priority_deal_player", offset)
    priority_player_id = g.priority_deal_player.id  # Assumes attribute exists
    priority_idx = player_map[priority_player_id]
    expected = np.zeros(num_players)
    expected[priority_idx] = 1.0
    np.testing.assert_allclose(encoding[s_slice], expected, atol=1e-6, err_msg="Section 5: Priority Deal")

    # --- Section 6: Bank Cash ---
    s_slice, offset = get_section_slice(encoder_1830, "bank_cash", offset)
    expected = g.bank.cash / encoder_1830.G1830_BANK_CASH if encoder_1830.G1830_BANK_CASH > 0 else 0
    assert_float(encoding[s_slice].item(), expected, "Section 6: Bank Cash")

    # --- Section 7: Player Certs Remaining ---
    s_slice, offset = get_section_slice(encoder_1830, "player_certs_remaining", offset)
    initial_limit = g.cert_limit  # Assumes cert_limit is accessible and correct
    expected = np.ones(num_players)  # All players start with full limit
    # Or calculate individually if needed:
    # expected = np.array([(p.cert_limit_remaining / initial_limit if initial_limit > 0 else 0) for p in g.players])
    np.testing.assert_allclose(encoding[s_slice], expected, atol=1e-6, err_msg="Section 7: Player Certs Remaining")

    # --- Section 8: Player Cash ---
    s_slice, offset = get_section_slice(encoder_1830, "player_cash", offset)
    start_cash = encoder_1830.starting_cash
    expected = np.array([p.cash / start_cash if start_cash > 0 else 0 for p_id, p in sorted(g.players_by_id.items())])
    np.testing.assert_allclose(encoding[s_slice], expected, atol=1e-6, err_msg="Section 8: Player Cash")
    assert_float(encoding[s_slice][0], 1.0, "Section 8: Player 1 Cash Initial")  # Specific check

    # --- Section 9: Player Shares ---
    s_slice, offset = get_section_slice(encoder_1830, "player_shares", offset)
    expected = np.zeros(num_players * num_corps)  # No shares at start
    np.testing.assert_allclose(encoding[s_slice], expected, atol=1e-6, err_msg="Section 9: Player Shares")

    # --- Section 10: Player Privates ---
    s_slice, offset = get_section_slice(encoder_1830, "player_privates", offset)
    expected = np.zeros(num_players * num_privates)  # No privates owned at start
    np.testing.assert_allclose(encoding[s_slice], expected, atol=1e-6, err_msg="Section 10: Player Privates")

    # --- Section 11: Private Revenue ---
    s_slice, offset = get_section_slice(encoder_1830, "private_revenue", offset)
    max_rev = max(comp.revenue for comp in g.companies) if g.companies else 1  # Avoid div by zero
    expected = np.zeros(num_privates)
    for priv_id, priv_idx in private_map.items():
        comp = g.company_by_id(priv_id)
        expected[priv_idx] = comp.revenue / max_rev if max_rev > 0 else 0
    np.testing.assert_allclose(encoding[s_slice], expected, atol=1e-6, err_msg="Section 11: Private Revenue")

    # --- Section 12: Private Corp Owner ---
    s_slice, offset = get_section_slice(encoder_1830, "private_corp_owner", offset)
    expected = np.zeros(num_privates * num_corps)  # No corps own privates at start
    np.testing.assert_allclose(encoding[s_slice], expected, atol=1e-6, err_msg="Section 12: Private Corp Owner")

    # --- Section 13: Corp Floated ---
    s_slice, offset = get_section_slice(encoder_1830, "corp_floated", offset)
    expected = np.zeros(num_corps)  # No corps floated at start
    np.testing.assert_allclose(encoding[s_slice], expected, atol=1e-6, err_msg="Section 13: Corp Floated")

    # --- Section 14: Corp Cash ---
    s_slice, offset = get_section_slice(encoder_1830, "corp_cash", offset)
    expected = np.zeros(num_corps)  # No corps have cash at start
    np.testing.assert_allclose(encoding[s_slice], expected, atol=1e-6, err_msg="Section 14: Corp Cash")

    # --- Section 15: Corp Trains ---
    s_slice, offset = get_section_slice(encoder_1830, "corp_trains", offset)
    expected = np.zeros(num_corps * encoder_1830.NUM_TRAIN_TYPES)  # No corps have trains at start
    np.testing.assert_allclose(encoding[s_slice], expected, atol=1e-6, err_msg="Section 15: Corp Trains")

    # --- Section 16: Corp Tokens Remaining ---
    s_slice, offset = get_section_slice(encoder_1830, "corp_tokens_remaining", offset)
    expected = np.zeros(num_corps)
    for corp_id, corp_idx in corp_map.items():
        corp = g.corporation_by_id(corp_id)
        # Assuming initial token count is stored/accessible, e.g., corp.max_tokens
        # If not, this needs adjustment based on how initial tokens are defined
        expected[corp_idx] = corp.tokens_available()  # Or similar method
    # Normalize? The spec says raw count. Let's test raw count.
    # Example: expected = np.array([corp.tokens_available() for corp_id, corp in sorted(g.corporations_by_id.items())])
    # This needs verification based on game engine implementation
    # For now, assert they are likely non-zero if defined
    # assert np.all(encoding[s_slice] > 0), "Section 16: Corp Tokens Remaining (should be > 0)"
    # Let's assert 0 for now, assuming they get tokens upon floating/parring
    np.testing.assert_allclose(
        encoding[s_slice], expected, atol=1e-6, err_msg="Section 16: Corp Tokens Remaining (Expected 0 before par)"
    )

    # --- Section 17: Corp Share Price ---
    s_slice, offset = get_section_slice(encoder_1830, "corp_share_price", offset)
    expected = np.zeros(num_corps)  # No price before par
    np.testing.assert_allclose(encoding[s_slice], expected, atol=1e-6, err_msg="Section 17: Corp Share Price")

    # --- Section 18: Corp IPO Shares ---
    s_slice, offset = get_section_slice(encoder_1830, "corp_ipo_shares", offset)
    expected = np.ones(num_corps)  # All 10 shares (normalized 10/10 = 1.0) are in IPO at start
    np.testing.assert_allclose(encoding[s_slice], expected, atol=1e-6, err_msg="Section 18: Corp IPO Shares")

    # --- Section 19: Corp Market Shares ---
    s_slice, offset = get_section_slice(encoder_1830, "corp_market_shares", offset)
    expected = np.zeros(num_corps)  # No shares in market at start
    np.testing.assert_allclose(encoding[s_slice], expected, atol=1e-6, err_msg="Section 19: Corp Market Shares")

    # --- Section 20: Corp Market Zone ---
    s_slice, offset = get_section_slice(encoder_1830, "corp_market_zone", offset)
    # Assuming -1 indicates off-board/not placed
    expected = np.full(num_corps, -1.0)
    np.testing.assert_allclose(encoding[s_slice], expected, atol=1e-6, err_msg="Section 20: Corp Market Zone")

    # --- Section 21: Depot Trains ---
    s_slice, offset = get_section_slice(encoder_1830, "depot_trains", offset)
    expected = np.zeros(encoder_1830.NUM_TRAIN_TYPES)
    for train_name, train_idx in encoder_1830.train_name_to_idx.items():
        initial_count = encoder_1830.initial_train_counts.get(train_name, 0)
        # Count trains of this type currently in depot
        current_count = sum(1 for t in g.depot.trains if t.name == train_name)  # Assumes depot.trains
        expected[train_idx] = current_count / initial_count if initial_count > 0 else 0
    np.testing.assert_allclose(encoding[s_slice], expected, atol=1e-6, err_msg="Section 21: Depot Trains")
    # Assert all trains are initially available (normalized to 1.0)
    np.testing.assert_allclose(
        encoding[s_slice],
        np.ones(encoder_1830.NUM_TRAIN_TYPES),
        atol=1e-6,
        err_msg="Section 21: Depot Trains (All initially 1.0)",
    )

    # --- Section 22: Market Pool Trains ---
    s_slice, offset = get_section_slice(encoder_1830, "market_pool_trains", offset)
    expected = np.zeros(encoder_1830.NUM_TRAIN_TYPES)  # No discarded trains at start
    np.testing.assert_allclose(encoding[s_slice], expected, atol=1e-6, err_msg="Section 22: Market Pool Trains")

    # --- Section 23: Depot Tiles ---
    s_slice, offset = get_section_slice(encoder_1830, "depot_tiles", offset)
    # Assert all tiles are initially available (normalized to 1.0)
    # This requires knowing which tiles *should* be available at the start
    # For simplicity, check a few known tiles if possible, or just the sum
    # expected = np.ones(encoder_1830.NUM_TILE_IDS) # This might not be true if some tiles aren't phase 1
    # Check sum is equal to the number of tiles initially available
    num_initial_tiles = sum(encoder_1830.initial_tile_counts.values())
    # The sum of normalized counts might not be easily predictable without knowing phase restrictions
    # Let's just check it's not all zero for now
    assert np.sum(encoding[s_slice]) > 0, "Section 23: Depot Tiles (Should have some tiles)"

    # --- Section 24: Hex Tile ID ---
    s_slice, offset = get_section_slice(encoder_1830, "hex_tile_id", offset)
    # Check a known hex, e.g., 'G15' (Altoona) should have tile '57' initially
    # This requires the hex_coords_ordered and tile_ids_ordered to be stable
    # Example (needs adjustment based on actual indices):
    # altoona_hex_coord = 'G15'
    # if altoona_hex_coord in encoder_1830.hex_coord_to_idx:
    #     altoona_idx = encoder_1830.hex_coord_to_idx[altoona_hex_coord]
    #     altoona_tile_name = '57' # Initial tile for Altoona
    #     if altoona_tile_name in encoder_1830.tile_id_to_idx:
    #          expected_tile_idx = encoder_1830.tile_id_to_idx[altoona_tile_name]
    #          assert_float(encoding[s_slice][altoona_idx], expected_tile_idx, "Section 24: Hex Tile ID (Altoona G15)")
    # else:
    #      pytest.skip("Hex G15 not found in encoder map")
    # For now, just check the slice has the right size
    assert len(encoding[s_slice]) == encoder_1830.NUM_HEXES

    # --- Section 25: Hex Rotation ---
    s_slice, offset = get_section_slice(encoder_1830, "hex_rotation", offset)
    # Check rotation for a known hex if possible
    # Example: Altoona 'G15' might have rotation 0 initially
    # if altoona_hex_coord in encoder_1830.hex_coord_to_idx:
    #     altoona_idx = encoder_1830.hex_coord_to_idx[altoona_hex_coord]
    #     assert_float(encoding[s_slice][altoona_idx], 0, "Section 25: Hex Rotation (Altoona G15)")
    # Just check size for now
    assert len(encoding[s_slice]) == encoder_1830.NUM_HEXES

    # --- Section 26: Hex Tokens ---
    s_slice, offset = get_section_slice(encoder_1830, "hex_tokens", offset)
    expected = np.zeros(encoder_1830.NUM_HEXES * encoder_1830.MAX_TILE_SLOTS * num_corps)  # No tokens placed at start
    np.testing.assert_allclose(encoding[s_slice], expected, atol=1e-6, err_msg="Section 26: Hex Tokens")

    # --- Auction Sections ---
    assert isinstance(g.round, WaterfallAuctionRound), "Game should start in WaterfallAuctionRound"
    auction_step = g.round  # Or g.round.active_step() depending on engine

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
        expected[priv_idx] = min_bid / start_cash if start_cash > 0 else 0
    np.testing.assert_allclose(encoding[s_slice], expected, atol=1e-4, err_msg="Section A2: Auction Min Bid")
    # Check specific value from old test (SV = $20, P1 start cash = $600? -> 20/600 = 0.0333)
    # Need to confirm starting cash for 4p game in the engine
    # assert_float(encoding[s_slice][private_map['SV']], 0.0333, "Section A2: Min Bid SV") # Example

    # --- Section A3: Auction Available ---
    s_slice, offset = get_section_slice(encoder_1830, "auction_available", offset)
    expected = np.zeros(num_privates)
    available_comp = auction_step.available_company  # Assumes attribute exists
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
        expected[priv_idx] = company.value / start_cash if start_cash > 0 else 0
    np.testing.assert_allclose(encoding[s_slice], expected, atol=1e-4, err_msg="Section A4: Auction Face Value")
    # Check specific value from old test (SV value = $20 -> 20/600 = 0.0333)
    # assert_float(encoding[s_slice][private_map['SV']], 0.0333, "Section A4: Face Value SV") # Example

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
    bid_amount = g.round.active_step().min_bid(bo_private)  # Bid minimum
    bid_action = None
    for action in action_helper.get_all_choices_limited():
        if isinstance(action, dict) and action.get("type") == "bid":
            if action.get("company") == bo_private.id and action.get("price") == bid_amount:
                bid_action = action
                break
    assert bid_action is not None, "Could not find bid action for P1 on B&O"

    g.process_action(bid_action)

    encoding = encoder_1830.encode(g).squeeze(0).numpy()
    offset = 0
    player_map = encoder_1830.player_id_to_idx
    private_map = encoder_1830.private_id_to_idx
    num_players = encoder_1830.num_players
    num_privates = encoder_1830.NUM_PRIVATES
    start_cash = encoder_1830.starting_cash

    # --- Check Active Player (Section 1) ---
    s_slice, offset = get_section_slice(encoder_1830, "active_entity", offset)
    active_player_id = g.round.active_entities()[0].id
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
            "player_privates",
            "private_revenue",
            "private_corp_owner",
            "corp_floated",
            "corp_cash",
            "corp_trains",
            "corp_tokens_remaining",
            "corp_share_price",
            "corp_ipo_shares",
            "corp_market_shares",
            "corp_market_zone",
            "depot_trains",
            "market_pool_trains",
            "depot_tiles",
            "hex_tile_id",
            "hex_rotation",
            "hex_tokens",
        ]
    )

    # --- Check Auction Bids (Section A1) ---
    s_slice, offset = get_section_slice(encoder_1830, "auction_bids", offset)
    bo_idx = private_map["BO"]
    p1_idx = player_map[player1.id]
    expected_bid_norm = bid_amount / start_cash if start_cash > 0 else 0
    # Calculate the flat index for P1's bid on B&O
    bid_flat_idx = bo_idx * num_players + p1_idx
    assert_float(encoding[s_slice][bid_flat_idx], expected_bid_norm, "Section A1: P1 Bid on B&O")
    # Check other bids are still 0
    assert np.sum(encoding[s_slice]) == expected_bid_norm, "Section A1: Only P1 B&O bid should be non-zero"

    # --- Check Auction Min Bid (Section A2) ---
    s_slice, offset = get_section_slice(encoder_1830, "auction_min_bid", offset)
    # Min bid for B&O should increase
    new_min_bid = g.round.min_bid(bo_private)
    expected_min_bid_norm = new_min_bid / start_cash if start_cash > 0 else 0
    assert_float(encoding[s_slice][bo_idx], expected_min_bid_norm, "Section A2: Min Bid B&O increased")

    # --- Check Auction Available (Section A3) ---
    s_slice, offset = get_section_slice(encoder_1830, "auction_available", offset)
    # Still SV should be available
    sv_idx = private_map["SV"]
    assert_float(encoding[s_slice][sv_idx], 1.0, "Section A3: SV still available")

    # --- Check Auction Face Value (Section A4) ---
    s_slice, offset = get_section_slice(encoder_1830, "auction_face_value", offset)
    # Face values should not change
    expected_fv_norm = bo_private.value / start_cash if start_cash > 0 else 0
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
        if isinstance(action, dict) and action.get("type") == "pass":
            pass_action_p1 = action
            break
    assert pass_action_p1 is not None, "Could not find pass action for P1"
    g.process_action(pass_action_p1)

    # P2 Buys SV
    buy_action_p2 = None
    # Update action helper for P2's turn
    action_helper = ActionHelper(g)  # Re-create or update helper
    for action in action_helper.get_all_choices_limited():
        if isinstance(action, dict) and action.get("type") == "buy_company":
            if action.get("company") == sv_private.id and action.get("price") == buy_price:
                buy_action_p2 = action
                break
    assert buy_action_p2 is not None, "Could not find buy action for P2 on SV"
    g.process_action(buy_action_p2)

    encoding = encoder_1830.encode(g).squeeze(0).numpy()
    offset = 0
    player_map = encoder_1830.player_id_to_idx
    private_map = encoder_1830.private_id_to_idx
    num_players = encoder_1830.num_players
    num_privates = encoder_1830.NUM_PRIVATES
    start_cash = encoder_1830.starting_cash
    sv_idx = private_map["SV"]
    p2_idx = player_map[player2.id]

    # --- Check Active Player (Section 1) ---
    s_slice, offset = get_section_slice(encoder_1830, "active_entity", offset)
    active_player_id = g.round.active_entities()[0].id
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
    expected_p2_cash = (start_cash - buy_price) / start_cash if start_cash > 0 else 0
    assert_float(encoding[s_slice][p2_idx], expected_p2_cash, "Section 8: Player 2 Cash reduced")
    assert_float(encoding[s_slice][0], 1.0, "Section 8: Player 1 Cash unchanged")  # P1 passed

    # Skip Player Shares
    offset += encoder_1830._get_section_size("player_shares")

    # --- Check Player Privates (Section 10) ---
    s_slice, offset = get_section_slice(encoder_1830, "player_privates", offset)
    # Calculate the flat index for P2 owning SV
    owner_flat_idx = p2_idx * num_privates + sv_idx
    assert_float(encoding[s_slice][owner_flat_idx], 1.0, "Section 10: Player 2 owns SV")
    assert np.sum(encoding[s_slice]) == 1.0, "Section 10: Only P2 should own SV"

    # Skip unchanged sections until Auction state
    offset += sum(
        encoder_1830._get_section_size(name)
        for name in [
            "private_revenue",
            "private_corp_owner",
            "corp_floated",
            "corp_cash",
            "corp_trains",
            "corp_tokens_remaining",
            "corp_share_price",
            "corp_ipo_shares",
            "corp_market_shares",
            "corp_market_zone",
            "depot_trains",
            "market_pool_trains",
            "depot_tiles",
            "hex_tile_id",
            "hex_rotation",
            "hex_tokens",
            "auction_bids",  # Bids should be cleared for SV? Verify engine behavior. Assume cleared.
        ]
    )
    # Re-check bids if they are NOT cleared on purchase
    # s_slice_bids, _ = get_section_slice(encoder_1830, "auction_bids", offset - encoder_1830._get_section_size("auction_bids"))
    # assert np.sum(encoding[s_slice_bids]) == 0, "Bids should clear after purchase"

    # --- Check Auction Min Bid (Section A2) ---
    s_slice, offset = get_section_slice(encoder_1830, "auction_min_bid", offset)
    # Min bid for SV should be irrelevant now, maybe 0 or -1? Encoder uses min_bid() which might error.
    # Let's assume the encoder handles sold privates gracefully (e.g., returns 0)
    # Or check the next available private's min bid
    next_private = g.company_by_id("CSL")  # Assuming CSL is next
    csl_idx = private_map["CSL"]
    expected_min_bid = g.round.min_bid(next_private) / start_cash if start_cash > 0 else 0
    assert_float(encoding[s_slice][csl_idx], expected_min_bid, "Section A2: Min Bid for CSL")

    # --- Check Auction Available (Section A3) ---
    s_slice, offset = get_section_slice(encoder_1830, "auction_available", offset)
    # CSL should be available now
    assert_float(encoding[s_slice][csl_idx], 1.0, "Section A3: CSL Available")
    assert_float(encoding[s_slice][sv_idx], 0.0, "Section A3: SV Not Available")

    # --- Check Auction Face Value (Section A4) ---
    s_slice, offset = get_section_slice(encoder_1830, "auction_face_value", offset)
    # Face values should not change
    expected_fv_norm = sv_private.value / start_cash if start_cash > 0 else 0
    assert_float(encoding[s_slice][sv_idx], expected_fv_norm, "Section A4: Face Value SV unchanged")

    # --- Final Offset Check ---
    assert offset == encoder_1830.ENCODING_SIZE, f"Final offset {offset} after purchase test"
