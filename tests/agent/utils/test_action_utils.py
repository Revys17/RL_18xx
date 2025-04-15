"""Tests for action encoding utilities."""
import unittest
import numpy as np
from unittest.mock import Mock, patch

from rl18xx.agent.utils.action_utils import (
    ActionSpace,
    STOCK_ACTIONS,
    OPERATING_ACTIONS,
    SPECIAL_ACTIONS,
    ActionDecodingError
)

class TestActionUtils(unittest.TestCase):
    def create_mock_with_numeric(self, value):
        """Create a mock that supports numeric operations."""
        mock = Mock()
        # Basic arithmetic
        mock.__add__ = Mock(side_effect=lambda x: float(value) + float(x))
        mock.__radd__ = Mock(side_effect=lambda x: float(x) + float(value))
        mock.__sub__ = Mock(side_effect=lambda x: float(value) - float(x))
        mock.__rsub__ = Mock(side_effect=lambda x: float(x) - float(value))
        mock.__mul__ = Mock(side_effect=lambda x: float(value) * float(x))
        mock.__rmul__ = Mock(side_effect=lambda x: float(x) * float(value))
        mock.__truediv__ = Mock(side_effect=lambda x: float(value) / float(x))
        mock.__rtruediv__ = Mock(side_effect=lambda x: float(x) / float(value))
        mock.__floordiv__ = Mock(side_effect=lambda x: float(value) // float(x))
        mock.__mod__ = Mock(side_effect=lambda x: float(value) % float(x))
        
        # Comparison
        mock.__eq__ = Mock(side_effect=lambda x: float(value) == float(x))
        mock.__ne__ = Mock(side_effect=lambda x: float(value) != float(x))
        mock.__lt__ = Mock(side_effect=lambda x: float(value) < float(x))
        mock.__le__ = Mock(side_effect=lambda x: float(value) <= float(x))
        mock.__gt__ = Mock(side_effect=lambda x: float(value) > float(x))
        mock.__ge__ = Mock(side_effect=lambda x: float(value) >= float(x))
        
        # Conversion
        mock.__int__ = Mock(return_value=int(value))
        mock.__float__ = Mock(return_value=float(value))
        mock.__str__ = Mock(return_value=str(value))
        mock.__format__ = Mock(side_effect=lambda fmt: format(float(value), fmt))
        
        # For numpy operations
        mock.__array__ = Mock(side_effect=lambda dtype=None: np.array(float(value), dtype=dtype))
        mock.__array_ufunc__ = None  # Let numpy handle ufuncs
        
        # Store original value
        mock.value = value
        return mock

    def setUp(self):
        """Set up test fixtures."""
        self.action_space = ActionSpace()
        
        # Create mock game state
        self.game = Mock()
        self.game.current_entity = Mock()
        self.game.current_entity.cash = self.create_mock_with_numeric(1000)
        
        # Mock corporation with proper share setup
        self.corporation = Mock()
        self.corporation.shares = []
        self.corporation.index = 0
        
        # Mock share with proper owner setup
        self.share = Mock()
        self.share.percent = self.create_mock_with_numeric(10)
        self.share.price = self.create_mock_with_numeric(100)
        self.share.owner = self.corporation  # Set owner to corporation for availability
        self.share.corporation = Mock(return_value=self.corporation)
        
        # Add share to corporation
        self.corporation.shares.append(self.share)
        
        # Set up game corporations
        self.game.corporations = [self.corporation]
        
        # Mock buy shares action
        self.buy_action = Mock(spec=['__class__', 'share', 'corporation'])
        buy_shares_class = Mock()
        buy_shares_class.__name__ = 'BuyShares'
        self.buy_action.__class__ = buy_shares_class
        self.buy_action.share = self.share
        self.buy_action.corporation = self.corporation
        
        # Mock tile laying actions
        self.hex = Mock()
        self.hex.x = self.create_mock_with_numeric(5)
        self.hex.y = self.create_mock_with_numeric(10)
        
        # Mock tile
        self.tile = Mock()
        self.lay_tile = Mock(spec=['__class__', 'hex', 'tile', 'rotation'])
        lay_tile_class = Mock()
        lay_tile_class.__name__ = 'LayTile'
        self.lay_tile.__class__ = lay_tile_class
        self.lay_tile.hex = self.hex
        self.lay_tile.tile = self.tile
        self.lay_tile.rotation = self.create_mock_with_numeric(2)
        
        # Mock token actions
        self.city = Mock()
        self.city.hex = self.hex
        self.city.index = self.create_mock_with_numeric(1)
        self.city.get_slot = Mock(return_value=0)
        
        # Mock cities list with proper length
        cities = Mock()
        cities.__len__ = Mock(return_value=2)
        cities.__getitem__ = Mock(return_value=self.city)
        self.hex.tile.cities = cities
        
        # Mock hex_by_coordinates to return proper hex
        self.game.hex_by_coordinates = Mock(return_value=self.hex)
        
        # Set up game state for action processing
        self.game.corporations = [self.corporation]
        self.game.share_pool = Mock()
        self.game.share_pool.can_buy = Mock(return_value=True)
        self.game.graph = Mock()
        self.game.graph.connected_nodes = Mock(return_value=[self.city])
        self.game.can_buy = Mock(return_value=True)
        
        # Ensure numeric values are properly handled
        self.hex.x = self.create_mock_with_numeric(5)
        self.hex.y = self.create_mock_with_numeric(10)
        self.lay_tile.rotation = self.create_mock_with_numeric(2)
        
        # Add proper length methods to mocks
        self.hex.tile.cities = Mock()
        self.hex.tile.cities.__len__ = Mock(return_value=2)
        
        # Mock place token action
        self.place_token = Mock(spec=['__class__', 'city', 'corporation'])
        place_token_class = Mock()
        place_token_class.__name__ = 'PlaceToken'
        self.place_token.__class__ = place_token_class
        self.place_token.city = self.city
        self.place_token.corporation = self.corporation
        
    def test_action_space_initialization(self):
        """Test action space initialization action dimension."""
        self.assertEqual(self.action_space.action_dim, 76)  # Updated dimension
        
    def test_action_space_initialization(self):
        """Test action space initialization action types."""
        self.assertTrue(all(k in self.action_space.action_types 
                          for k in ['Pass', 'BuyShares', 'LayTile']))
        
    def test_encode_action_buy_shares(self):
        """Test encoding of buy shares action type."""
        encoded = self.action_space.encode_action(self.buy_action)
        action_type_idx = STOCK_ACTIONS['BuyShares']
        self.assertEqual(encoded[action_type_idx], 1)
        
    def test_encode_action_buy_shares_corporation(self):
        """Test encoding of buy shares corporation."""
        encoded = self.action_space.encode_action(self.buy_action)
        self.assertEqual(encoded[13], 1)  # First corporation (index 0 at offset 13)
        
    def test_encode_action_buy_shares_percent(self):
        """Test encoding of buy shares percent."""
        encoded = self.action_space.encode_action(self.buy_action)
        self.assertEqual(encoded[21], 0.1)  # 10% normalized
        
    def test_encode_action_sell_shares(self):
        """Test encoding of sell shares action."""
        sell_action = Mock(__class__=Mock(__name__='SellShares'))
        sell_action.share = self.share
        sell_action.corporation = self.corporation
        
        encoded = self.action_space.encode_action(sell_action)
        action_type_idx = STOCK_ACTIONS['SellShares']
        self.assertEqual(encoded[action_type_idx], 1)
        
    def test_encode_action_lay_tile_action_type(self):
        """Test encoding of lay tile action type."""
        encoded = self.action_space.encode_action(self.lay_tile)
        action_type_idx = OPERATING_ACTIONS['LayTile']
        self.assertEqual(encoded[action_type_idx], 1)
        
    def test_encode_action_lay_tile_x_coordinate(self):
        """Test encoding of lay tile x coordinate."""
        encoded = self.action_space.encode_action(self.lay_tile)
        self.assertEqual(encoded[35], 5/100)  # x coordinate normalized at index 35
        
    def test_encode_action_lay_tile_y_coordinate(self):
        """Test encoding of lay tile y coordinate."""
        encoded = self.action_space.encode_action(self.lay_tile)
        self.assertEqual(encoded[36], 10/100)  # y coordinate normalized at index 36
        
    def test_encode_action_lay_tile_rotation(self):
        """Test encoding of lay tile rotation."""
        encoded = self.action_space.encode_action(self.lay_tile)
        self.assertEqual(encoded[44 + 2], 1)  # Rotation 2 one-hot encoding at index 46
        
    def test_encode_action_place_token_action_type(self):
        """Test encoding of place token action type."""
        encoded = self.action_space.encode_action(self.place_token)
        action_type_idx = OPERATING_ACTIONS['PlaceToken']
        self.assertEqual(encoded[action_type_idx], 1)
        
    def test_encode_action_place_token_city_slot(self):
        """Test encoding of place token city slot."""
        encoded = self.action_space.encode_action(self.place_token)
        self.assertEqual(encoded[29], 1)  # City slot 1
        
    def test_encode_action_pass(self):
        """Test encoding of pass action."""
        pass_action = Mock(__class__=Mock(__name__='Pass'))
        encoded = self.action_space.encode_action(pass_action)
        action_type_idx = SPECIAL_ACTIONS['Pass']
        self.assertEqual(encoded[action_type_idx], 1)
        
    def test_encode_action_bankrupt(self):
        """Test encoding of bankrupt action."""
        bankrupt_action = Mock(__class__=Mock(__name__='Bankrupt'))
        encoded = self.action_space.encode_action(bankrupt_action)
        action_type_idx = SPECIAL_ACTIONS['Bankrupt']
        self.assertEqual(encoded[action_type_idx], 1)
        
    def test_get_action_mask_dimensions(self):
        """Test action mask dimensions."""
        valid_actions = [self.buy_action, self.lay_tile]
        mask = self.action_space.get_action_mask(valid_actions)
        self.assertEqual(len(mask), self.action_space.action_dim)
        
    def test_get_action_mask_valid_actions(self):
        """Test action mask for valid actions."""
        valid_actions = [self.buy_action, self.lay_tile]
        mask = self.action_space.get_action_mask(valid_actions)
        buy_idx = STOCK_ACTIONS['BuyShares']
        lay_idx = OPERATING_ACTIONS['LayTile']
        self.assertTrue(mask[buy_idx])
        self.assertTrue(mask[lay_idx])
        
    def test_get_action_mask_invalid_actions(self):
        """Test action mask for invalid actions."""
        valid_actions = [self.buy_action, self.lay_tile]
        mask = self.action_space.get_action_mask(valid_actions)
        pass_idx = SPECIAL_ACTIONS['Pass']
        self.assertFalse(mask[pass_idx])
        
    def test_decode_action_buy_shares(self):
        """Test decoding of buy shares action."""
        encoded = self.action_space.encode_action(self.buy_action)
        decoded = self.action_space.decode_action(encoded, self.game)
        self.assertIsNotNone(decoded)
        self.assertEqual(decoded.__class__.__name__, 'BuyShares')
        if decoded is not None:  # For type checking
            self.assertEqual(decoded.share.percent.value, 10)
            self.assertEqual(decoded.corporation.index, 0)
            
    def test_decode_action_lay_tile(self):
        """Test decoding of lay tile action."""
        encoded = self.action_space.encode_action(self.lay_tile)
        decoded = self.action_space.decode_action(encoded, self.game)
        self.assertIsNotNone(decoded)
        self.assertEqual(decoded.__class__.__name__, 'LayTile')
        if decoded is not None:  # For type checking
            self.assertEqual(decoded.hex.x.value, 5)
            self.assertEqual(decoded.hex.y.value, 10)
            self.assertEqual(decoded.rotation.value, 2)
            
    def test_decode_action_place_token(self):
        """Test decoding of place token action."""
        encoded = self.action_space.encode_action(self.place_token)
        decoded = self.action_space.decode_action(encoded, self.game)
        self.assertIsNotNone(decoded)
        self.assertEqual(decoded.__class__.__name__, 'PlaceToken')
        if decoded is not None:  # For type checking
            self.assertEqual(decoded.city.index.value, 1)
            
    def test_decode_action_invalid(self):
        """Test decoding of invalid action."""
        invalid_encoded = np.zeros(self.action_space.action_dim)
        # Set a valid action type to avoid KeyError
        invalid_encoded[SPECIAL_ACTIONS['Pass']] = 1
        decoded = self.action_space.decode_action(invalid_encoded, self.game)
        self.assertIsNone(decoded)
        
    def test_encode_action_invalid_type(self):
        """Test encoding of invalid action type."""
        invalid_action = Mock()
        invalid_action.__class__ = Mock(__name__='InvalidAction')
        invalid_action.__class__.__name__ = 'InvalidAction'
        encoded = self.action_space.encode_action(invalid_action)
        self.assertTrue(all(v == 0 for v in encoded))  # All values should be 0
        
    def test_encode_action_missing_attributes(self):
        """Test encoding of action with missing attributes."""
        broken_action = Mock()
        broken_action.__class__ = Mock(__name__='BuyShares')
        broken_action.__class__.__name__ = 'BuyShares'
        # Don't set any other attributes
        encoded = self.action_space.encode_action(broken_action)
        # All fields except action type should be 0
        self.assertEqual(encoded[STOCK_ACTIONS['BuyShares']], 1)  # Action type set
        self.assertTrue(all(v == 0 for i, v in enumerate(encoded) 
                          if i != STOCK_ACTIONS['BuyShares']))  # Rest should be 0
        
    def test_action_space_coverage(self):
        """Test that all game actions are covered."""
        all_actions = {**STOCK_ACTIONS, **OPERATING_ACTIONS, **SPECIAL_ACTIONS}
        
        for action_name in all_actions:
            # Create mock action
            action = Mock(__class__=Mock(__name__=action_name))
            
            # Try encoding
            encoded = self.action_space.encode_action(action)
            
            # Verify basic encoding worked
            self.assertEqual(len(encoded), self.action_space.action_dim)
            action_idx = all_actions[action_name]
            self.assertEqual(encoded[action_idx], 1)
        
    def test_encode_action_buy_shares_price_and_amount(self):
        """Test encoding of buy shares with separate price and amount."""
        encoded = self.action_space.encode_action(self.buy_action)
        self.assertEqual(encoded[33], 100/1000)  # Price normalized
        self.assertEqual(encoded[34], 0.1)  # Amount (10%) normalized
        
    def test_encode_action_sell_shares_bundle(self):
        """Test encoding of sell shares bundle."""
        sell_action = Mock(__class__=Mock(__name__='SellShares'))
        sell_action.shares = [self.share, self.share]  # Bundle of 2 shares
        sell_action.corporation = self.corporation
        
        encoded = self.action_space.encode_action(sell_action)
        self.assertEqual(encoded[34], 2/10)  # 2 shares normalized
        self.assertEqual(encoded[33], 100/1000)  # Price normalized
        
    def test_encode_action_buy_company(self):
        """Test encoding of buy company action."""
        buy_company = Mock(__class__=Mock(__name__='BuyCompany'))
        buy_company.company = Mock(index=0)
        buy_company.price = self.create_mock_with_numeric(500)
        buy_company.seller = Mock(index=1)
        
        encoded = self.action_space.encode_action(buy_company)
        self.assertEqual(encoded[OPERATING_ACTIONS['BuyCompany']], 1)
        self.assertEqual(encoded[13], 1)  # Target company
        self.assertEqual(encoded[33], 500/1000)  # Price normalized
        self.assertEqual(encoded[24], 1)  # Seller index
        
    def test_encode_action_run_routes(self):
        """Test encoding of run routes with detailed route information."""
        route = Mock()
        route.train = Mock(index=0)
        route.start_hex = Mock(x=5, y=10)
        route.end_hex = Mock(x=8, y=12)
        route.revenue = self.create_mock_with_numeric(300)
        route.path_encoding = 1
        
        run_routes = Mock(__class__=Mock(__name__='RunRoutes'))
        run_routes.routes = [route]
        
        encoded = self.action_space.encode_action(run_routes)
        self.assertEqual(encoded[50], 1)  # Route exists
        self.assertEqual(encoded[51], 0)  # Train index
        self.assertEqual(encoded[52:54].tolist(), [5/100, 10/100])  # Start hex
        self.assertEqual(encoded[54:56].tolist(), [8/100, 12/100])  # End hex
        self.assertEqual(encoded[56], 300/1000)  # Revenue normalized
        self.assertEqual(encoded[57], 1)  # Path encoding
        
    def test_decode_action_error_handling(self):
        """Test error handling in decode_action."""
        # Create invalid action encoding
        invalid_encoded = np.zeros(self.action_space.action_dim)
        invalid_encoded[STOCK_ACTIONS['BuyShares']] = 1
        invalid_encoded[13] = 1  # Target corporation index out of range
        
        with self.assertRaises(ActionDecodingError):
            self.action_space.decode_action(invalid_encoded, self.game)
            
    def test_get_action_mask_continuous_values(self):
        """Test action mask handling of continuous values."""
        buy_action1 = Mock(__class__=Mock(__name__='BuyShares'))
        buy_action1.corporation = self.corporation
        buy_action1.share = Mock(price=self.create_mock_with_numeric(100))
        
        buy_action2 = Mock(__class__=Mock(__name__='BuyShares'))
        buy_action2.corporation = self.corporation
        buy_action2.share = Mock(price=self.create_mock_with_numeric(200))
        
        valid_actions = [buy_action1, buy_action2]
        mask = self.action_space.get_action_mask(valid_actions)
        
        # Verify the BuyShares action type is valid
        self.assertTrue(mask[STOCK_ACTIONS['BuyShares']])
        # Verify the price dimension is valid
        self.assertTrue(mask[33])  # Make sure this index matches the actual price encoding index
        
    def test_encode_decode_roundtrip(self):
        """Test that encoding and decoding preserves action properties."""
        # Test with BuyShares action
        encoded = self.action_space.encode_action(self.buy_action)
        decoded = self.action_space.decode_action(encoded, self.game)
        
        self.assertIsNotNone(decoded)
        self.assertEqual(decoded.__class__.__name__, 'BuyShares')
        if decoded is not None:  # For type checking
            self.assertEqual(decoded.share.price.value, self.buy_action.share.price.value)
            self.assertEqual(decoded.corporation.index, self.buy_action.corporation.index)

if __name__ == '__main__':
    unittest.main()
