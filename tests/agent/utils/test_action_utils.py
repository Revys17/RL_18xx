"""Tests for action encoding utilities."""
import unittest
import numpy as np
from unittest.mock import Mock, patch

from rl18xx.agent.utils.action_utils import (
    ActionSpace,
    STOCK_ACTIONS,
    OPERATING_ACTIONS,
    SPECIAL_ACTIONS
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
        
        # Mock share actions
        self.share = Mock()
        self.share.percent = self.create_mock_with_numeric(10)
        self.share.price = self.create_mock_with_numeric(100)
        
        # Mock corporation
        self.corporation = Mock()
        share_price_mock = Mock()
        share_price_mock.price = self.create_mock_with_numeric(100)
        self.corporation.share_price = share_price_mock
        self.corporation.index = 0
        
        # Set up corporation state
        self.corporation.ipo_owner = Mock()
        self.corporation.owner = Mock()
        self.share.owner = self.corporation.ipo_owner
        self.share.corporation = Mock(return_value=self.corporation)
        self.corporation.shares = [self.share]
        
        # Mock buy shares action
        buy_shares_class = Mock()
        buy_shares_class.__name__ = 'BuyShares'
        self.buy_action = Mock()
        self.buy_action.__class__ = buy_shares_class
        self.buy_action.share = self.share
        self.buy_action.corporation = self.corporation
        
        # Mock tile laying actions
        self.hex = Mock()
        self.hex.x = self.create_mock_with_numeric(5)
        self.hex.y = self.create_mock_with_numeric(10)
        
        # Mock tile
        self.tile = Mock()
        lay_tile_class = Mock()
        lay_tile_class.__name__ = 'LayTile'
        self.lay_tile = Mock()
        self.lay_tile.__class__ = lay_tile_class
        self.lay_tile.hex = self.hex
        self.lay_tile.tile = self.tile
        self.lay_tile.rotation = self.create_mock_with_numeric(2)
        
        # Mock token actions
        self.city = Mock()
        self.city.hex = self.hex
        self.city.index = self.create_mock_with_numeric(1)
        self.city.get_slot = Mock(return_value=0)
        
        place_token_class = Mock()
        place_token_class.__name__ = 'PlaceToken'
        self.place_token = Mock()
        self.place_token.__class__ = place_token_class
        self.place_token.city = self.city
        self.place_token.corporation = self.corporation
        
        # Set up game state for action processing
        self.game.corporations = [self.corporation]
        self.game.share_pool = Mock()
        self.game.share_pool.can_buy = Mock(return_value=True)
        self.game.graph = Mock()
        self.game.graph.connected_nodes = Mock(return_value=[self.city])
        self.game.can_buy = Mock(return_value=True)
        
    def test_action_space_initialization(self):
        """Test action space initialization."""
        self.assertEqual(self.action_space.action_dim, 43)
        self.assertTrue(all(k in self.action_space.action_types 
                          for k in ['Pass', 'BuyShares', 'LayTile']))
        
    def test_encode_stock_actions(self):
        """Test encoding of stock round actions."""
        # Test buy shares encoding
        encoded = self.action_space.encode_action(self.buy_action)
        
        # Check action type one-hot
        action_type_idx = STOCK_ACTIONS['BuyShares']
        self.assertEqual(encoded[action_type_idx], 1)
        
        # Check corporation encoding
        self.assertEqual(encoded[1], 1)  # First corporation
        
        # Check share percent encoding
        self.assertEqual(encoded[21], 0.1)  # 10% normalized
        
        # Test sell shares encoding
        sell_action = Mock(__class__=Mock(__name__='SellShares'))
        sell_action.share = self.share
        sell_action.corporation = self.corporation
        
        encoded = self.action_space.encode_action(sell_action)
        action_type_idx = STOCK_ACTIONS['SellShares']
        self.assertEqual(encoded[action_type_idx], 1)
        
    def test_encode_operating_actions(self):
        """Test encoding of operating round actions."""
        # Test lay tile encoding
        encoded = self.action_space.encode_action(self.lay_tile)
        
        # Check action type
        action_type_idx = OPERATING_ACTIONS['LayTile']
        self.assertEqual(encoded[action_type_idx], 1)
        
        # Check hex coordinates
        self.assertEqual(encoded[22], 5/100)  # x coordinate normalized
        self.assertEqual(encoded[23], 10/100)  # y coordinate normalized
        
        # Check rotation
        self.assertEqual(encoded[33], 1)  # Rotation 2 one-hot encoding
        
        # Test place token encoding
        encoded = self.action_space.encode_action(self.place_token)
        action_type_idx = OPERATING_ACTIONS['PlaceToken']
        self.assertEqual(encoded[action_type_idx], 1)
        self.assertEqual(encoded[29], 1)  # City slot 1
        
    def test_encode_special_actions(self):
        """Test encoding of special actions."""
        # Test pass action
        pass_action = Mock(__class__=Mock(__name__='Pass'))
        encoded = self.action_space.encode_action(pass_action)
        action_type_idx = SPECIAL_ACTIONS['Pass']
        self.assertEqual(encoded[action_type_idx], 1)
        
        # Test bankrupt action
        bankrupt_action = Mock(__class__=Mock(__name__='Bankrupt'))
        encoded = self.action_space.encode_action(bankrupt_action)
        action_type_idx = SPECIAL_ACTIONS['Bankrupt']
        self.assertEqual(encoded[action_type_idx], 1)
        
    def test_action_masking(self):
        """Test action masking functionality."""
        valid_actions = [self.buy_action, self.lay_tile]
        mask = self.action_space.get_action_mask(valid_actions)
        
        # Check mask dimensions
        self.assertEqual(len(mask), self.action_space.action_dim)
        
        # Check valid actions are unmasked
        buy_idx = STOCK_ACTIONS['BuyShares']
        lay_idx = OPERATING_ACTIONS['LayTile']
        self.assertTrue(mask[buy_idx])
        self.assertTrue(mask[lay_idx])
        
        # Check invalid actions are masked
        pass_idx = SPECIAL_ACTIONS['Pass']
        self.assertFalse(mask[pass_idx])
        
    def test_decode_actions(self):
        """Test action decoding functionality."""
        # Test buy shares decoding
        encoded = self.action_space.encode_action(self.buy_action)
        decoded = self.action_space.decode_action(encoded, self.game)
        self.assertIsNotNone(decoded)
        self.assertEqual(decoded.__class__.__name__, 'BuyShares')
        if decoded is not None:  # For type checking
            self.assertEqual(decoded.share.percent.value, 10)
            self.assertEqual(decoded.corporation.index, 0)
            
        # Test lay tile decoding
        encoded = self.action_space.encode_action(self.lay_tile)
        decoded = self.action_space.decode_action(encoded, self.game)
        self.assertIsNotNone(decoded)
        self.assertEqual(decoded.__class__.__name__, 'LayTile')
        if decoded is not None:  # For type checking
            self.assertEqual(decoded.hex.x.value, 5)
            self.assertEqual(decoded.hex.y.value, 10)
            self.assertEqual(decoded.rotation.value, 2)
            
        # Test place token decoding
        encoded = self.action_space.encode_action(self.place_token)
        decoded = self.action_space.decode_action(encoded, self.game)
        self.assertIsNotNone(decoded)
        self.assertEqual(decoded.__class__.__name__, 'PlaceToken')
        if decoded is not None:  # For type checking
            self.assertEqual(decoded.city.index.value, 1)
            
        # Test invalid action decoding
        invalid_encoded = np.zeros(self.action_space.action_dim)
        decoded = self.action_space.decode_action(invalid_encoded, self.game)
        self.assertIsNone(decoded)
        
    def test_error_handling(self):
        """Test error handling in action encoding/decoding."""
        # Test with invalid action type
        invalid_action = Mock()
        invalid_action.__class__ = Mock(__name__='InvalidAction')
        invalid_action.__class__.__name__ = 'InvalidAction'
        encoded = self.action_space.encode_action(invalid_action)
        self.assertTrue(all(v == 0 for v in encoded))  # All values should be 0
        
        # Test with missing attributes
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
        # Check that all action types have encoders
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

if __name__ == '__main__':
    unittest.main()