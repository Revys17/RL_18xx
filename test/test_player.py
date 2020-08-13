import unittest
from unittest.mock import MagicMock, Mock

from src.actions.bid_buy_action import BidBuyAction
from src.game_state import GameState
from src.agent import Agent
from src.player import Player


class PlayerTest(unittest.TestCase):

    def setUp(self):
        self.agent_mock: Agent = Mock()
        self.game_state_mock: GameState = Mock()

        self.test_player: Player = Player(0, 1000, self.agent_mock)

    def test_get_bid_buy_action(self):
        expected_action: BidBuyAction = BidBuyAction()
        self.agent_mock.get_bid_buy_action = MagicMock()
        self.agent_mock.get_bid_buy_action.return_value = expected_action

        # call
        actual: BidBuyAction = self.test_player.get_bid_buy_action(self.game_state_mock)

        self.assertEqual(expected_action, actual)
        self.agent_mock.get_bid_buy_action.assert_called_once_with()








if __name__ == '__main__':
    unittest.main()
