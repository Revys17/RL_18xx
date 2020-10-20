import unittest
from unittest.mock import MagicMock, Mock

import e30


class PlayerTest(unittest.TestCase):

    def setUp(self):
        self.agent_mock: e30.agent.Agent = Mock()
        self.game_state_mock: e30.game_state.GameState = Mock()

        self.test_player: e30.player.Player = e30.player.Player(0, 1000, self.agent_mock)
        self.test_player.privates = [e30.privates.BO.BO(), e30.privates.CA.CA()]

    def test_get_bid_buy_action(self):
        expected_action: e30.actions.bid_buy_action.BidBuyAction = e30.actions.bid_buy_action.BidBuyAction()
        self.agent_mock.get_bid_buy_action = MagicMock()
        self.agent_mock.get_bid_buy_action.return_value = expected_action

        # call
        actual: e30.actions.bid_buy_action.BidBuyAction = self.test_player.get_bid_buy_action(self.game_state_mock)

        self.assertEqual(expected_action, actual)
        self.agent_mock.get_bid_buy_action.assert_called_once_with(self.game_state_mock)

    def test_get_bid_resolution_action(self):
        expected_action: e30.actions.bid_resolution_action.BidResolutionAction = \
            e30.actions.bid_resolution_action.BidResolutionAction()
        self.agent_mock.get_bid_resolution_action = MagicMock()
        self.agent_mock.get_bid_resolution_action.return_value = expected_action

        # call
        actual: e30.actions.bid_resolution_action.BidResolutionAction = self.test_player.get_bid_resolution_action(
            self.game_state_mock)

        self.assertEqual(expected_action, actual)
        self.agent_mock.get_bid_resolution_action.assert_called_once_with(self.game_state_mock)

    def test_pay_private_income(self):
        # starting + privates revenue
        expected_total = 1000 + 30 + 25

        self.test_player.pay_private_income()

        self.assertEqual(expected_total, self.test_player.money)

    def test_add_money(self):
        self.test_player.add_money(1000)

        # starting + added money
        self.assertEqual(1000 + 1000, self.test_player.money)

    def test_get_name(self):
        self.assertEqual(self.test_player.name, self.test_player.get_name())

    def test_has_share(self):
        self.assertEqual(True, self.test_player.has_share(MagicMock()))

    def test_set_aside_money(self):
        # call
        self.test_player.set_aside_money(500)

        self.assertEqual(500, self.test_player.money)

    def test_return_money(self):
        # call
        self.test_player.return_money(500)

        self.assertEqual(1500, self.test_player.money)

    def test_remove_money(self):
        # call
        self.test_player.remove_money(500)

        self.assertEqual(500, self.test_player.money)


if __name__ == '__main__':
    unittest.main()
