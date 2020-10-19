import unittest
from unittest.mock import MagicMock, Mock

import e30


class PlayerTest(unittest.TestCase):

    def setUp(self):
        self.agent_mock: e30.agent.Agent = Mock()
        self.game_state_mock: e30.game_state.GameState = Mock()

        self.test_player: e30.player.Player = e30.player.Player(0, 1000, self.agent_mock)

    def test_get_bid_buy_action(self):
        expected_action: e30.actions.bid_buy_action.BidBuyAction = e30.actions.bid_buy_action.BidBuyAction()
        self.agent_mock.get_bid_buy_action = MagicMock()
        self.agent_mock.get_bid_buy_action.return_value = expected_action

        # call
        actual: e30.actions.bid_buy_action.BidBuyAction = self.test_player.get_bid_buy_action(self.game_state_mock)

        self.assertEqual(expected_action, actual)
        self.agent_mock.get_bid_buy_action.assert_called_once_with(self.game_state_mock)








if __name__ == '__main__':
    unittest.main()
