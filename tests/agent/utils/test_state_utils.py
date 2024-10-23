"""Tests for state encoding utilities."""
import unittest
import numpy as np
from unittest.mock import Mock, patch

from rl18xx.agent.utils.state_utils import (
    encode_player_state,
    encode_corporation_state,
    encode_market_state,
    encode_game_phase,
    encode_full_state
)

class TestStateUtils(unittest.TestCase):
    def create_mock_with_numeric(self, value):
        """Create a mock that supports numeric operations."""
        mock = Mock()
        # Division
        mock.__truediv__ = Mock(side_effect=lambda x: value / x)
        mock.__div__ = Mock(side_effect=lambda x: value / x)
        mock.__rtruediv__ = Mock(side_effect=lambda x: x / value)
        mock.__rdiv__ = Mock(side_effect=lambda x: x / value)
        # Addition
        mock.__add__ = Mock(side_effect=lambda x: value + x)
        mock.__radd__ = Mock(side_effect=lambda x: x + value)
        mock.__iadd__ = Mock(side_effect=lambda x: value + x)
        # Subtraction
        mock.__sub__ = Mock(side_effect=lambda x: value - x)
        mock.__rsub__ = Mock(side_effect=lambda x: x - value)
        mock.__isub__ = Mock(side_effect=lambda x: value - x)
        # Multiplication
        mock.__mul__ = Mock(side_effect=lambda x: value * x)
        mock.__rmul__ = Mock(side_effect=lambda x: x * value)
        mock.__imul__ = Mock(side_effect=lambda x: value * x)
        # Division
        mock.__floordiv__ = Mock(side_effect=lambda x: value // x)
        mock.__rfloordiv__ = Mock(side_effect=lambda x: x // value)
        # Modulo
        mock.__mod__ = Mock(side_effect=lambda x: value % x)
        mock.__rmod__ = Mock(side_effect=lambda x: x % value)
        # Power
        mock.__pow__ = Mock(side_effect=lambda x: value ** x)
        mock.__rpow__ = Mock(side_effect=lambda x: x ** value)
        # Bitwise
        mock.__and__ = Mock(side_effect=lambda x: value & x)
        mock.__or__ = Mock(side_effect=lambda x: value | x)
        mock.__xor__ = Mock(side_effect=lambda x: value ^ x)
        # Comparison
        mock.__lt__ = Mock(side_effect=lambda x: value < x)
        mock.__le__ = Mock(side_effect=lambda x: value <= x)
        mock.__gt__ = Mock(side_effect=lambda x: value > x)
        mock.__ge__ = Mock(side_effect=lambda x: value >= x)
        mock.__eq__ = Mock(side_effect=lambda x: value == x)
        mock.__ne__ = Mock(side_effect=lambda x: value != x)
        # Conversion
        mock.__int__ = Mock(return_value=int(value))
        mock.__float__ = Mock(return_value=float(value))
        mock.__str__ = Mock(return_value=str(value))
        mock.__index__ = Mock(return_value=int(value))
        mock.__round__ = Mock(return_value=round(value))
        mock.__floor__ = Mock(return_value=int(value))
        mock.__ceil__ = Mock(return_value=int(value) + (1 if value > int(value) else 0))
        mock.__trunc__ = Mock(return_value=int(value))
        # Protocols
        mock.__iter__ = Mock(return_value=iter([value]))
        mock.__len__ = Mock(return_value=1)
        mock.__bool__ = Mock(return_value=bool(value))
        # Store original value for comparison
        mock.value = value
        return mock

    def setUp(self):
        """Set up test fixtures."""
        # Create mock game state
        self.game = Mock()
        
        # Create corporations with proper mock configuration
        self.corporations = []
        for i in range(10):
            corp = Mock()
            # Financial state
            share_price_mock = Mock()
            share_price_mock.price = self.create_mock_with_numeric(100)
            corp.share_price = share_price_mock
            corp.cash = self.create_mock_with_numeric(500)
            
            # Shares and ownership
            corp.shares = [Mock() for _ in range(10)]
            corp.ipo_owner = Mock()
            par_price_mock = Mock()
            par_price_mock.price = self.create_mock_with_numeric(100)
            corp.par_price = Mock(return_value=par_price_mock)
            
            # Trains
            train_mocks = [Mock() for _ in range(2)]
            for train in train_mocks:
                train.name = "2"
                train.price = 200
            corp.trains = train_mocks
            
            # Tokens with cities
            city_mock = Mock()
            revenue_mock = self.create_mock_with_numeric(30)
            revenue_mock.__iadd__ = Mock(side_effect=lambda x: self.create_mock_with_numeric(revenue_mock.value + x))
            city_mock.revenue = revenue_mock
            city_mock.hex = Mock()
            city_mock.hex.x = self.create_mock_with_numeric(5)
            city_mock.hex.y = self.create_mock_with_numeric(10)
            
            token_mock = Mock()
            token_mock.city = city_mock
            token_mock.corporation = corp
            
            unplaced_token1 = Mock()
            unplaced_token1.city = None
            unplaced_token2 = Mock()
            unplaced_token2.city = None
            
            # Set up tokens list with proper iteration support
            tokens_list = [token_mock]
            tokens_mock = Mock()
            tokens_mock.__iter__ = Mock(return_value=iter(tokens_list))
            tokens_mock.__len__ = Mock(return_value=len(tokens_list))
            tokens_mock.__getitem__ = Mock(side_effect=tokens_list.__getitem__)
            
            unplaced_tokens = [unplaced_token1, unplaced_token2]
            unplaced_tokens_mock = Mock()
            unplaced_tokens_mock.__iter__ = Mock(return_value=iter(unplaced_tokens))
            unplaced_tokens_mock.__len__ = Mock(return_value=len(unplaced_tokens))
            unplaced_tokens_mock.__getitem__ = Mock(side_effect=unplaced_tokens.__getitem__)
            
            corp.placed_tokens = Mock(return_value=tokens_mock)
            corp.unplaced_tokens = Mock(return_value=unplaced_tokens_mock)
            
            # Status flags
            corp.ipoed = True
            corp.floated = Mock(return_value=True)
            corp.operated = Mock(return_value=True)
            corp.receivership = False
            
            # Collections that need items() method
            share_holders_mock = Mock()
            mock_holders = {self.game: 60, Mock(): 40}  # Use game as a stable mock key
            share_holders_mock.items = Mock(return_value=mock_holders.items())
            corp.share_holders = share_holders_mock
            
            operating_history_mock = Mock()
            mock_history = {1: 100, 2: 150}
            operating_history_mock.items = Mock(return_value=mock_history.items())
            corp.operating_history = operating_history_mock
            
            self.corporations.append(corp)
            
        self.game.corporations = self.corporations
        
        # Create players with proper mock configuration
        self.players = []
        for i in range(4):
            player = Mock()
            # Financial state
            player.cash = self.create_mock_with_numeric(1000)
            player.value = self.create_mock_with_numeric(2000)
            
            # Holdings
            player.shares = [Mock() for _ in range(5)]
            company_mocks = [Mock() for _ in range(2)]
            for company in company_mocks:
                company.value = 100
            player.companies = company_mocks
            
            # Share holdings
            player.percent_of = Mock(return_value=10)  # 10% share holding
            mock_shares = {}
            for i, corp in enumerate(self.corporations[:3]):
                share_mock = Mock()
                share_mock.percent = self.create_mock_with_numeric(10)
                share_mock.price = self.create_mock_with_numeric(100)
                share_mock.corporation = Mock(return_value=corp)
                mock_shares[corp] = [share_mock]
            # Create a dict-like mock that supports iteration
            shares_by_corp_mock = Mock()
            shares_by_corp_mock.items = Mock(return_value=list(mock_shares.items()))
            shares_by_corp_mock.__iter__ = Mock(return_value=iter(mock_shares))
            shares_by_corp_mock.__getitem__ = Mock(side_effect=mock_shares.__getitem__)
            shares_by_corp_mock.get = Mock(side_effect=mock_shares.get)
            shares_by_corp_mock.keys = Mock(return_value=mock_shares.keys())
            shares_by_corp_mock.values = Mock(return_value=mock_shares.values())
            player.shares_by_corporation = shares_by_corp_mock
            
            self.players.append(player)
            
        self.game.players = self.players
        self.game.current_entity = self.players[0]
        self.game.share_pool = Mock()
        
        # Bank setup
        bank_mock = Mock()
        bank_mock.cash = self.create_mock_with_numeric(10000)
        self.game.bank = bank_mock
        
        # Game phase setup
        phase_mock = Mock()
        phase_mock.name = "3"
        phase_mock.__str__ = Mock(return_value="3")
        self.game.phase = phase_mock
        
        # Round setup
        round_mock = Mock()
        round_mock.__class__ = Mock(__name__="OperatingRound")
        step_mock = Mock()
        step_mock.__class__ = Mock(__name__="Track")
        round_mock.current_step = step_mock
        round_mock.turns_remaining = self.create_mock_with_numeric(3)
        self.game.round = round_mock
        
        # Operation order setup
        operation_order_mock = Mock()
        operation_order_mock.__iter__ = Mock(side_effect=lambda: iter(self.corporations[:3]))
        operation_order_mock.index = Mock(side_effect=lambda x: self.corporations[:3].index(x))
        operation_order_mock.__len__ = Mock(return_value=3)
        self.game.operation_order = operation_order_mock
        
        # Game counters and limits
        self.game.round_counter = self.create_mock_with_numeric(5)
        self.game.stock_round_number = self.create_mock_with_numeric(2)
        self.game.cert_limit = Mock(return_value=40)
        self.game.train_limit = Mock(return_value=4)
        
        # Stock market setup
        self.game.share_prices = [Mock(price=self.create_mock_with_numeric(p)) for p in [50, 75, 100, 150]]
        
        # Players list setup
        players_mock = Mock()
        players_mock.__iter__ = Mock(side_effect=lambda: iter(self.players))
        players_mock.index = Mock(side_effect=lambda x: self.players.index(x))
        players_mock.__len__ = Mock(return_value=len(self.players))
        players_mock.__getitem__ = Mock(side_effect=lambda i: self.players[i])
        self.game.players = players_mock
        
        # Priority deal
        self.game.priority_deal_player = self.players[0]
        
        # Set up references for individual tests
        self.player = self.players[0]
        self.corporation = self.corporations[0]
        self.corporation.owner = self.player
        
    def test_encode_player_state(self):
        """Test player state encoding."""
        state = encode_player_state(self.player, self.game)
        
        # Check dimensions
        self.assertEqual(len(state), 45)  # Adjust based on actual dimensions
        
        # Check normalization
        self.assertTrue(all(-1 <= x <= 1 for x in state))
        
        # Check specific encodings
        self.assertEqual(state[0], 1000 / 10000)  # Cash normalization
        self.assertEqual(state[2], 0.1)  # Share percentage for first corporation
        
        # Test with no shares or companies
        self.player.shares = []
        self.player.companies = []
        state = encode_player_state(self.player, self.game)
        self.assertEqual(state[42], 0.0)  # Certificate count should be 0
        
    def test_encode_corporation_state(self):
        """Test corporation state encoding."""
        state = encode_corporation_state(self.corporation, self.game)
        
        # Check dimensions
        self.assertEqual(len(state), 34)  # Adjust based on actual dimensions
        
        # Check normalization
        self.assertTrue(all(-1 <= x <= 1 for x in state))
        
        # Check specific encodings
        self.assertEqual(state[0], 100 / 10000)  # Share price normalization
        self.assertEqual(state[2], 2 / 4)  # Number of trains normalization
        self.assertEqual(state[5], 1)  # IPO status
        self.assertEqual(state[6], 1)  # Float status
        
        # Test with no trains
        self.corporation.trains = []
        state = encode_corporation_state(self.corporation, self.game)
        self.assertEqual(state[33], 1)  # Should need mandatory train
        
    def test_encode_market_state(self):
        """Test market state encoding."""
        state = encode_market_state(self.game)
        
        # Check dimensions
        self.assertEqual(len(state), 57)  # Adjust based on actual dimensions
        
        # Check normalization
        self.assertTrue(all(-1 <= x <= 1 for x in state))
        
        # Check specific encodings
        self.assertEqual(state[51], 3/6)  # Phase normalization
        self.assertEqual(state[52], 2/3)  # Operating round normalization
        
    def test_encode_game_phase(self):
        """Test game phase encoding."""
        state = encode_game_phase(self.game)
        
        # Check dimensions
        self.assertEqual(len(state), 14)  # Adjust based on actual dimensions
        
        # Check normalization
        self.assertTrue(all(-1 <= x <= 1 for x in state))
        
        # Check round type encoding
        round_type = state[2:6]  # One-hot encoding of round type
        self.assertEqual(round_type[1], 1)  # Operating round should be active
        self.assertEqual(sum(round_type), 1)  # Only one round type should be active
        
    def test_encode_full_state(self):
        """Test full state encoding."""
        with patch('rl18xx.agent.utils.state_utils.encode_board_state') as mock_board:
            mock_board.return_value = np.zeros((20, 20, 54))
            state = encode_full_state(self.game)
            
            # Check all components present
            self.assertTrue(all(k in state for k in [
                'board', 'players', 'corporations', 'market', 'phase'
            ]))
            
            # Check component shapes
            self.assertEqual(state['board'].shape, (20, 20, 54))
            self.assertEqual(state['players'].shape, (4, 45))
            self.assertEqual(state['corporations'].shape, (10, 34))
            self.assertEqual(state['market'].shape, (57,))
            self.assertEqual(state['phase'].shape, (14,))
            
    def test_error_handling(self):
        """Test error handling in encoding functions."""
        # Test with invalid phase name
        self.game.phase.name = "Invalid"
        state = encode_market_state(self.game)
        self.assertEqual(state[51], 1/6)  # Should default to phase 1
        
        # Test with missing attributes
        delattr(self.game, 'round_counter')
        state = encode_game_phase(self.game)
        self.assertEqual(state[0], 0)  # Should handle missing round counter
        
    def test_special_states(self):
        """Test handling of special game states."""
        # Test emergency fund raising
        self.game.emergency_fund_raising = True
        state = encode_market_state(self.game)
        self.assertEqual(state[56], 1)  # Emergency flag should be set
        
        # Test train export pending
        self.game.train_export_pending = True
        state = encode_market_state(self.game)
        self.assertEqual(state[54], 1)  # Train export flag should be set
        
        # Test phase change pending
        self.game.phase_change_triggered = True
        state = encode_market_state(self.game)
        self.assertEqual(state[55], 1)  # Phase change flag should be set

if __name__ == '__main__':
    unittest.main()