import unittest
from unittest.mock import MagicMock

from e30.agent import HumanAgent
from e30.game_state import GameState


class GameStateTest(unittest.TestCase):

    def setUp(self):
        self.test_game_state = GameState([HumanAgent(), HumanAgent()])

        for p in self.test_game_state.players:
            p.pay_private_income = MagicMock()

        self.player = self.test_game_state.current_player
        self.player2 = self.test_game_state.get_next_player(self.player)

    def test_get_priority_deal_player(self):
        self.assertEqual(self.player, self.test_game_state.get_priority_deal_player())

    def test_set_next_as_priority_deal(self):
        self.assertEqual(self.player, self.test_game_state.get_priority_deal_player())

        # call
        self.test_game_state.set_next_as_priority_deal(self.player)

        self.assertEqual(self.player2, self.test_game_state.get_priority_deal_player())

    def test_get_next_player(self):
        self.assertEqual(self.player2, self.test_game_state.get_next_player(self.player))
        self.assertEqual(self.player, self.test_game_state.get_next_player(self.player2))

    def test_set_next_as_current_player(self):
        self.assertEqual(self.player, self.test_game_state.current_player)

        # call
        self.test_game_state.set_next_as_current_player(self.player)

        self.assertEqual(self.player2, self.test_game_state.current_player)

        # call
        self.test_game_state.set_next_as_current_player(self.player2)

        self.assertEqual(self.player, self.test_game_state.current_player)

    def test_set_current_player_as_priority_deal(self):
        self.test_game_state.set_next_as_priority_deal(self.player)
        self.assertEqual(self.player2, self.test_game_state.get_priority_deal_player())
        self.assertEqual(self.player, self.test_game_state.current_player)

        # call
        self.test_game_state.set_current_player_as_priority_deal()

        self.assertEqual(self.player2, self.test_game_state.get_priority_deal_player())
        self.assertEqual(self.player2, self.test_game_state.current_player)

    def test_pay_private_revenue(self):
        # call
        self.test_game_state.pay_private_revenue()

        [p.pay_private_income.assert_called_once_with() for p in self.test_game_state.players]


if __name__ == '__main__':
    unittest.main()
