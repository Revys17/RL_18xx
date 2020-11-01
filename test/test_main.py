import unittest
from unittest.mock import MagicMock, call

import e30
from e30.actions.bid_buy_action import BidBuyAction, BidBuyActionType
from e30.agent import HumanAgent
from e30.exceptions.exceptions import InvalidOperationException
from e30.game_state import GameState
from e30.privates.BO import BO
from e30.privates.SV import SV


class MainTest(unittest.TestCase):

    def setUp(self):
        self.game_state = GameState([HumanAgent(), HumanAgent()])

        for p in self.game_state.players:
            p.get_bid_buy_action = MagicMock()
            p.get_bid_resolution_action = MagicMock()
            p.pay_private_income = MagicMock()

    def test_do_private_auction_no_unowned_privates_exit(self):
        player = self.game_state.current_player
        player2 = self.game_state.get_next_player(player)
        # no unowned privates
        self.game_state.privates = []

        # call
        e30.main.do_private_auction(self.game_state)

        # assert nothing happened
        [p.get_bid_buy_action.assert_not_called() for p in self.game_state.players]
        [p.get_bid_resolution_action.assert_not_called() for p in self.game_state.players]

        self.assertEqual(player, self.game_state.priority_deal)
        self.assertEqual(player, self.game_state.current_player)

    def test_do_private_auction_bid_on_lowest_resolves(self):
        private = SV()
        private.add_bid = MagicMock()
        self.game_state.privates = [private]
        player = self.game_state.current_player
        player2 = self.game_state.get_next_player(player)
        player.get_bid_buy_action.return_value = BidBuyAction(BidBuyActionType.BID, 0, 25)
        private.should_resolve_bid = MagicMock()
        private.should_resolve_bid.side_effect = [True]
        private.resolve_bid = MagicMock()

        # call
        e30.main.do_private_auction(self.game_state)

        private.add_bid.assert_called_once_with(player, 25)
        self.assertEqual(1, private.should_resolve_bid.call_count)
        private.resolve_bid.assert_called_once_with(self.game_state)
        player.get_bid_buy_action.assert_called_once_with(self.game_state)
        player2.get_bid_buy_action.assert_not_called()
        self.assertEqual(0, len(self.game_state.privates))
        self.assertEqual(player, self.game_state.priority_deal)
        self.assertEqual(player2, self.game_state.current_player)

    def test_do_private_auction_all_pass_SV_0_forced_buy(self):
        private = SV()
        private.price = 5
        self.game_state.privates = [private]
        player = self.game_state.current_player
        player2 = self.game_state.get_next_player(player)
        player.get_bid_buy_action.return_value = BidBuyAction()  # pass, pass, price = 0, pass, pass, forced buy
        player2.get_bid_buy_action.return_value = BidBuyAction()
        private.should_resolve_bid = MagicMock()
        private.should_resolve_bid.side_effect = [False, False, False, False]
        private.buy_private = MagicMock()

        # call
        e30.main.do_private_auction(self.game_state)

        private.buy_private.assert_called_once_with(player)
        self.assertEqual(4, private.should_resolve_bid.call_count)
        player.get_bid_buy_action.assert_has_calls([call(self.game_state), call(self.game_state)])
        player2.get_bid_buy_action.assert_has_calls([call(self.game_state), call(self.game_state)])
        self.assertEqual(0, len(self.game_state.privates))
        self.assertEqual(player2, self.game_state.priority_deal)
        self.assertEqual(player2, self.game_state.current_player)

    def test_do_private_auction_all_pass_not_SV_revenue_paid(self):
        private = BO()
        self.game_state.privates = [private]
        player = self.game_state.current_player
        player2 = self.game_state.get_next_player(player)
        # all pass, then player buys
        player.get_bid_buy_action.side_effect = [BidBuyAction(), BidBuyAction(BidBuyActionType.BUY, 0, 0)]
        player2.get_bid_buy_action.return_value = BidBuyAction()
        private.should_resolve_bid = MagicMock()
        private.should_resolve_bid.side_effect = [False, False, False]
        private.buy_private = MagicMock()

        # call
        e30.main.do_private_auction(self.game_state)

        private.buy_private.assert_called_once_with(player)
        self.assertEqual(3, private.should_resolve_bid.call_count)
        # private income paid for all passing
        [p.pay_private_income.assert_called_once_with() for p in self.game_state.players]
        player.get_bid_buy_action.assert_has_calls([call(self.game_state), call(self.game_state)])
        player2.get_bid_buy_action.assert_called_once_with(self.game_state)
        self.assertEqual(0, len(self.game_state.privates))
        self.assertEqual(player2, self.game_state.priority_deal)
        self.assertEqual(player2, self.game_state.current_player)

    def test_do_private_auction_buy_lowest_resolves(self):
        private = SV()
        self.game_state.privates = [private]
        player = self.game_state.current_player
        player2 = self.game_state.get_next_player(player)
        player.get_bid_buy_action.return_value = BidBuyAction(BidBuyActionType.BUY, 0, 0)
        private.should_resolve_bid = MagicMock()
        private.should_resolve_bid.side_effect = [False]
        private.buy_private = MagicMock()

        # call
        e30.main.do_private_auction(self.game_state)

        private.buy_private.assert_called_once_with(player)
        self.assertEqual(1, private.should_resolve_bid.call_count)
        player.get_bid_buy_action.assert_called_once_with(self.game_state)
        player2.get_bid_buy_action.assert_not_called()
        self.assertEqual(0, len(self.game_state.privates))
        self.assertEqual(player2, self.game_state.priority_deal)
        self.assertEqual(player2, self.game_state.current_player)

    def test_do_private_auction_bid_buy_action_error_retries(self):
        private = SV()
        self.game_state.privates = [private]
        player = self.game_state.current_player
        player2 = self.game_state.get_next_player(player)
        # error then retry buy, resolves
        player.get_bid_buy_action.side_effect = [InvalidOperationException("hello"),
                                                 BidBuyAction(BidBuyActionType.BUY, 0, 0)]
        private.should_resolve_bid = MagicMock()
        private.should_resolve_bid.side_effect = [False]
        private.buy_private = MagicMock()

        # call
        e30.main.do_private_auction(self.game_state)

        private.buy_private.assert_called_once_with(player)
        self.assertEqual(1, private.should_resolve_bid.call_count)
        player.get_bid_buy_action.assert_has_calls([call(self.game_state), call(self.game_state)])
        player2.get_bid_buy_action.assert_not_called()
        self.assertEqual(0, len(self.game_state.privates))
        self.assertEqual(player2, self.game_state.priority_deal)
        self.assertEqual(player2, self.game_state.current_player)

    def test_do_private_auction_bid_on_lowest_index_error_retries_resolves(self):
        private = SV()
        private.add_bid = MagicMock()
        self.game_state.privates = [private]
        player = self.game_state.current_player
        player2 = self.game_state.get_next_player(player)
        # index out of bounds, retry, valid, resolves
        player.get_bid_buy_action.side_effect = [BidBuyAction(BidBuyActionType.BID, 1, 25),
                                                 BidBuyAction(BidBuyActionType.BID, 0, 25)]
        private.should_resolve_bid = MagicMock()
        private.should_resolve_bid.side_effect = [True]
        private.resolve_bid = MagicMock()

        # call
        e30.main.do_private_auction(self.game_state)

        private.add_bid.assert_called_once_with(player, 25)
        self.assertEqual(1, private.should_resolve_bid.call_count)
        private.resolve_bid.assert_called_once_with(self.game_state)
        player.get_bid_buy_action.assert_has_calls([call(self.game_state), call(self.game_state)])
        player2.get_bid_buy_action.assert_not_called()
        self.assertEqual(0, len(self.game_state.privates))
        self.assertEqual(player, self.game_state.priority_deal)
        self.assertEqual(player2, self.game_state.current_player)

    def test_complete_purchase(self):
        player = self.game_state.current_player
        player2 = self.game_state.get_next_player(player)
        private = SV()
        private.buy_private = MagicMock()
        self.game_state.privates = [private]

        # call
        e30.main.complete_purchase(self.game_state, player, private, self.game_state.privates)

        private.buy_private.assert_called_once_with(player)
        self.assertEqual(0, len(self.game_state.privates))
        self.assertEqual(player2, self.game_state.priority_deal)


if __name__ == '__main__':
    unittest.main()
