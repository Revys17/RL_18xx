import unittest
from unittest.mock import MagicMock

import e30
from e30.actions.bid_resolution_action import BidResolutionAction, BidResolutionActionType


class PrivateCompanyTest(unittest.TestCase):

    def setUp(self):
        self.player = e30.player.Player(index=0, starting_money=1000, agent=MagicMock())
        self.player2 = e30.player.Player(index=1, starting_money=1000, agent=MagicMock())
        self.game_state = MagicMock()

        #220 cost
        self.test_private_company = e30.privates.BO.BO()

    def test_add_bid_no_existing_bid(self):

        self.assertEqual(0, len(self.test_private_company.bids.keys()))

        # call
        self.test_private_company.add_bid(self.player, 225)

        self.assertEqual(1, len(self.test_private_company.bids.keys()))
        self.assertEqual(225, self.test_private_company.bids[self.player])
        self.assertEqual(1000 - 225, self.player.money)
        self.assertEqual(225, self.test_private_company.current_winning_bid)

    def test_add_bid_existing_bid(self):

        self.assertEqual(0, len(self.test_private_company.bids.keys()))

        # call
        self.test_private_company.add_bid(self.player, 225)
        self.test_private_company.add_bid(self.player, 230)

        self.assertEqual(1, len(self.test_private_company.bids.keys()))
        self.assertEqual(230, self.test_private_company.bids[self.player])
        self.assertEqual(1000 - 230, self.player.money)
        self.assertEqual(230, self.test_private_company.current_winning_bid)

        # add another player's bid
        self.test_private_company.add_bid(self.player2, 235)

        self.assertEqual(2, len(self.test_private_company.bids.keys()))
        self.assertEqual(230, self.test_private_company.bids[self.player])
        self.assertEqual(235, self.test_private_company.bids[self.player2])
        self.assertEqual(1000 - 230, self.player.money)
        self.assertEqual(1000 - 235, self.player2.money)
        self.assertEqual(235, self.test_private_company.current_winning_bid)

    def test_add_bid_less_than_price_and_minimum(self):
        # call
        self.assertRaises(e30.exceptions.exceptions.InvalidOperationException,
                          self.test_private_company.add_bid, self.player, 224)

    def test_add_bid_less_than_winning_bid_and_minimum(self):
        self.test_private_company.current_winning_bid = 500
        self.assertTrue(self.test_private_company.price + 5 < 500)

        # call
        self.assertRaises(e30.exceptions.exceptions.InvalidOperationException,
                          self.test_private_company.add_bid, self.player, 504)

    def test_add_bid_player_not_enough_money_no_existing_bid(self):
        self.assertEqual(1000, self.player.money)

        # call
        self.assertRaises(e30.exceptions.exceptions.InvalidOperationException,
                          self.test_private_company.add_bid, self.player, 1001)

    def test_add_bid_player_not_enough_money_existing_bid(self):
        self.assertEqual(1000, self.player.money)

        self.assertEqual(0, len(self.test_private_company.bids.keys()))

        # call
        self.test_private_company.add_bid(self.player, 225)

        self.assertEqual(1, len(self.test_private_company.bids.keys()))
        self.assertEqual(225, self.test_private_company.bids[self.player])
        self.assertEqual(1000 - 225, self.player.money)
        self.assertEqual(225, self.test_private_company.current_winning_bid)

        # call
        self.assertRaises(e30.exceptions.exceptions.InvalidOperationException,
                          self.test_private_company.add_bid, self.player, 1001)

    def test_should_resolve_bid(self):
        # call
        self.assertFalse(self.test_private_company.should_resolve_bid())

        self.test_private_company.add_bid(self.player, 225)

        # call
        self.assertTrue(self.test_private_company.should_resolve_bid())

    def test_resolve_bid_1_bidder(self):
        self.test_private_company.add_bid(self.player, 225)

        # call
        self.test_private_company.resolve_bid(self.game_state)

        self.assertEqual(1000 - 225, self.player.money)
        self.assertEqual(self.player, self.test_private_company.owner)
        self.assertIsNone(self.test_private_company.bids)

    def test_resolve_bid_2_bidders_lower_bidder_bids_and_wins(self):
        self.test_private_company.add_bid(self.player, 225)
        self.test_private_company.add_bid(self.player2, 230)

        self.player2.get_bid_resolution_action = MagicMock()
        # highest bidder passes first round
        self.player2.get_bid_resolution_action.return_value = BidResolutionAction()

        self.player.get_bid_resolution_action = MagicMock()
        # lower bidder increases bid first round, wins the bid
        self.player.get_bid_resolution_action.return_value = BidResolutionAction(BidResolutionActionType.BID, 235)

        # call
        self.test_private_company.resolve_bid(self.game_state)

        self.assertEqual(1000, self.player2.money)
        self.assertEqual(1000 - 235, self.player.money)
        self.assertEqual(self.player, self.test_private_company.owner)
        self.assertEqual([self.test_private_company], self.player.privates)
        self.assertEqual([], self.player2.privates)
        self.assertIsNone(self.test_private_company.bids)

    def test_resolve_bid_2_bidders_lower_bidder_passes_loses(self):
        self.test_private_company.add_bid(self.player, 225)
        self.test_private_company.add_bid(self.player2, 230)

        self.player.get_bid_resolution_action = MagicMock()
        # lower bidder passes, higher bidder wins
        self.player.get_bid_resolution_action.return_value = BidResolutionAction()

        self.player2.get_bid_resolution_action = MagicMock()

        # call
        self.test_private_company.resolve_bid(self.game_state)

        self.assertEqual(1000 - 230, self.player2.money)
        self.assertEqual(1000, self.player.money)
        self.assertEqual(self.player2, self.test_private_company.owner)
        self.assertEqual([self.test_private_company], self.player2.privates)
        self.assertEqual([], self.player.privates)
        self.assertIsNone(self.test_private_company.bids)
        self.player2.get_bid_resolution_action.assert_not_called()

    def test_resolve_bid_2_bidders_bid_twice_then_pass(self):
        self.test_private_company.add_bid(self.player, 225)
        self.test_private_company.add_bid(self.player2, 230)

        self.player.get_bid_resolution_action = MagicMock()
        # lower bidder bids first round, bids second round, passes
        self.player.get_bid_resolution_action.side_effect = [BidResolutionAction(BidResolutionActionType.BID, 235),
                                                             BidResolutionAction(BidResolutionActionType.BID, 245),
                                                             BidResolutionAction()]

        self.player2.get_bid_resolution_action = MagicMock()
        # higher bidder bids first round, bids second round, wins
        self.player2.get_bid_resolution_action.side_effect = [BidResolutionAction(BidResolutionActionType.BID, 240),
                                                              BidResolutionAction(BidResolutionActionType.BID, 250)]

        # call
        self.test_private_company.resolve_bid(self.game_state)

        self.assertEqual(1000 - 250, self.player2.money)
        self.assertEqual(1000, self.player.money)
        self.assertEqual(self.player2, self.test_private_company.owner)
        self.assertEqual([self.test_private_company], self.player2.privates)
        self.assertEqual([], self.player.privates)
        self.assertIsNone(self.test_private_company.bids)

    def test_resolve_bid_2_bidders_bad_bids_retry(self):
        self.test_private_company.add_bid(self.player, 225)
        self.test_private_company.add_bid(self.player2, 230)

        self.player.get_bid_resolution_action = MagicMock()
        # lower bidder bids first round, bids second round, passes
        self.player.get_bid_resolution_action.side_effect = [BidResolutionAction(BidResolutionActionType.BID, 230),
                                                             BidResolutionAction(BidResolutionActionType.BID, 235),
                                                             BidResolutionAction()]

        self.player2.get_bid_resolution_action = MagicMock()
        # higher bidder bids first round, bids second round, wins
        self.player2.get_bid_resolution_action.side_effect = [BidResolutionAction(BidResolutionActionType.BID, 230),
                                                              BidResolutionAction(BidResolutionActionType.BID, 240)]

        # call
        self.test_private_company.resolve_bid(self.game_state)

        self.assertEqual(1000 - 240, self.player2.money)
        self.assertEqual(1000, self.player.money)
        self.assertEqual(self.player2, self.test_private_company.owner)
        self.assertEqual([self.test_private_company], self.player2.privates)
        self.assertEqual([], self.player.privates)
        self.assertIsNone(self.test_private_company.bids)

    def test_lower_price(self):
        self.assertEqual(220, self.test_private_company.price)

        # call
        self.test_private_company.lower_price(100)

        self.assertEqual(220 - 100, self.test_private_company.price)

    def test_buy_private_player_in_bids(self):
        self.assertEqual(1000, self.player.money)
        self.test_private_company.add_bid(self.player, 225)
        self.assertEqual(1000 - 225, self.player.money)

        # call
        self.test_private_company.buy_private(self.player)

        self.assertEqual(1000 - 225, self.player.money)
        self.assertEqual(self.player, self.test_private_company.owner)
        self.assertEqual([self.test_private_company], self.player.privates)

    def test_buy_private_player_not_in_bids(self):
        self.assertEqual(1000, self.player.money)

        # call
        self.test_private_company.buy_private(self.player)

        self.assertEqual(1000 - 220, self.player.money)
        self.assertEqual(self.player, self.test_private_company.owner)
        self.assertEqual([self.test_private_company], self.player.privates)

    def test_buy_private_player_not_in_bids_not_enough(self):
        self.player.money = 219

        # call
        self.assertRaises(e30.exceptions.exceptions.InvalidOperationException,
                          self.test_private_company.buy_private, self.player)

        self.assertEqual(219, self.player.money)
        self.assertFalse(hasattr(self.test_private_company, 'owner'))
        self.assertEqual([], self.player.privates)


if __name__ == '__main__':
    unittest.main()
