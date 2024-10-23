"""Tests for reward calculation utilities."""
import unittest
import numpy as np
from unittest.mock import Mock, patch

from rl18xx.agent.utils.reward_utils import RewardCalculator

class TestRewardUtils(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        # Create reward calculator
        self.reward_calc = RewardCalculator(
            net_worth_weight=1.0,
            relative_position_weight=0.5,
            dividend_weight=0.3,
            route_weight=0.2,
            connectivity_weight=0.2,
            bankruptcy_penalty=-1.0
        )
        
        # Mock game state
        self.game = Mock()
        self.game.players = [Mock() for _ in range(4)]
        self.game.corporations = [Mock() for _ in range(3)]
        
        # Set up player
        self.player = Mock()
        self.player.cash = 1000
        self.player.shares = [Mock() for _ in range(3)]
        self.player.companies = [Mock(value=100) for _ in range(2)]
        self.player.shares_by_corporation = {}
        
        # Set up corporation
        self.corporation = Mock()
        self.corporation.cash = 500
        self.corporation.trains = [Mock(price=200) for _ in range(2)]
        self.corporation.share_price = Mock(price=100)
        self.corporation.tokens = [Mock() for _ in range(3)]
        self.corporation.operating_history = {1: 100, 2: 150}
        self.corporation.owner = self.player
        self.corporation.placed_tokens = Mock(return_value=[Mock()])
        
        # Set up actions
        self.dividend_action = Mock(__class__=Mock(__name__='Dividend'))
        self.dividend_action.kind = 'full'
        
        self.lay_tile_action = Mock(__class__=Mock(__name__='LayTile'))
        self.bankrupt_action = Mock(__class__=Mock(__name__='Bankrupt'))
        
    def test_net_worth_calculation(self):
        """Test net worth calculation."""
        # Basic net worth
        net_worth = self.reward_calc.calculate_net_worth(self.player, self.game)
        expected = (
            1000 +  # Cash
            200 +   # Company values
            0      # No shares yet
        )
        self.assertEqual(net_worth, expected)
        
        # Add share holdings
        corp = Mock()
        corp.share_price = Mock(price=100)
        self.player.shares_by_corporation[corp] = [Mock(percent=10) for _ in range(2)]
        
        net_worth = self.reward_calc.calculate_net_worth(self.player, self.game)
        expected += 20  # 20% of share price
        self.assertEqual(net_worth, expected)
        
        # Test with corporation presidency
        self.corporation.owner = self.player
        net_worth = self.reward_calc.calculate_net_worth(self.player, self.game)
        expected += (
            500 +   # Corporation cash
            200    # Half value of trains
        )
        self.assertEqual(net_worth, expected)
        
    def test_relative_position(self):
        """Test relative position calculation."""
        # Set up player net worths
        players = self.game.players
        worths = [1000, 2000, 1500, 500]
        for player, worth in zip(players, worths):
            player.cash = worth
            player.shares = []
            player.companies = []
            player.shares_by_corporation = {}
        
        # Test player positions
        for player, worth in zip(players, worths):
            position = self.reward_calc.calculate_relative_position(player, self.game)
            # Position should be between -1 and 1
            self.assertTrue(-1 <= position <= 1)
            # Higher worth should give higher position
            if worth == max(worths):
                self.assertGreater(position, 0)
            if worth == min(worths):
                self.assertLess(position, 0)
                
    def test_route_value(self):
        """Test route value calculation."""
        # Test basic route value
        value = self.reward_calc.calculate_route_value(self.corporation, self.game)
        self.assertEqual(value, 150)  # Latest revenue
        
        # Test with no operating history
        self.corporation.operating_history = {}
        value = self.reward_calc.calculate_route_value(self.corporation, self.game)
        self.assertEqual(value, 0)
        
    def test_connectivity(self):
        """Test connectivity calculation."""
        # Mock graph connections
        self.game.graph = Mock()
        self.game.graph.connected = Mock(return_value=True)
        
        # Create mock tokens and cities
        token1, token2 = Mock(), Mock()
        city1, city2 = Mock(), Mock()
        city1.id, city2.id = 1, 2
        token1.city, token2.city = city1, city2
        
        self.corporation.placed_tokens.return_value = [token1, token2]
        
        # Test connectivity value
        value = self.reward_calc.calculate_connectivity(self.corporation, self.game)
        self.assertEqual(value, 1)  # One connected pair
        
        # Test with no connections
        self.game.graph.connected = Mock(return_value=False)
        value = self.reward_calc.calculate_connectivity(self.corporation, self.game)
        self.assertEqual(value, 0)
        
    def test_action_rewards(self):
        """Test immediate action rewards."""
        # Test dividend action
        reward = self.reward_calc.calculate_action_reward(self.dividend_action, self.game)
        self.assertEqual(reward, 1.0)  # Full dividend
        
        # Test half dividend
        self.dividend_action.kind = 'half'
        reward = self.reward_calc.calculate_action_reward(self.dividend_action, self.game)
        self.assertEqual(reward, 0.5)  # Half dividend
        
        # Test lay tile
        reward = self.reward_calc.calculate_action_reward(self.lay_tile_action, self.game)
        self.assertEqual(reward, 0.1)  # Small positive reward
        
        # Test bankruptcy
        reward = self.reward_calc.calculate_action_reward(self.bankrupt_action, self.game)
        self.assertEqual(reward, -1.0)  # Bankruptcy penalty
        
    def test_full_reward_calculation(self):
        """Test full reward calculation pipeline."""
        # Calculate initial reward
        reward = self.reward_calc.calculate_reward(
            self.player,
            self.game,
            self.dividend_action,
            False
        )
        
        # Should include:
        # - Net worth change
        # - Relative position
        # - Action reward
        # - Route value
        # - Connectivity
        self.assertIsInstance(reward, float)
        
        # Test terminal state
        reward = self.reward_calc.calculate_reward(
            self.player,
            self.game,
            None,
            True
        )
        
        # Should include terminal reward
        self.assertIsInstance(reward, float)
        
    def test_reward_tracking(self):
        """Test reward calculator state tracking."""
        # Calculate reward twice
        state1 = self.reward_calc.calculate_reward(self.player, self.game, None, False)
        
        # Change player state
        self.player.cash += 500
        
        state2 = self.reward_calc.calculate_reward(self.player, self.game, None, False)
        
        # Second reward should reflect the change
        self.assertNotEqual(state1, state2)
        
        # Previous state should be tracked
        self.assertEqual(
            self.reward_calc.previous_net_worth[self.player],
            self.reward_calc.calculate_net_worth(self.player, self.game)
        )

if __name__ == '__main__':
    unittest.main()