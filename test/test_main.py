import unittest
from collections import OrderedDict
from unittest.mock import MagicMock, call

import e30
from e30.actions.bid_buy_action import BidBuyAction, BidBuyActionType
from e30.actions.stock_market_sell_action import StockMarketSellAction, StockMarketSellActionType
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
            p.get_stock_market_sell_action = MagicMock()

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

    def test_process_stock_round_sell_action_pass(self):
        player = self.game_state.get_priority_deal_player()
        # pass
        player.get_stock_market_sell_action.return_value = StockMarketSellAction()
        self.assertEqual({}, player.share_map)
        self.assertEqual([], player.presiding_companies)
        self.assertEqual(1200, player.money)

        # call
        e30.main.process_stock_round_sell_action(self.game_state, player)

        self.assertEqual({}, player.share_map)
        self.assertEqual([], player.presiding_companies)
        self.assertEqual(1200, player.money)

    def test_process_stock_round_sell_action_requested_sales_not_owned(self):
        player = self.game_state.get_priority_deal_player()
        # sell unowned company
        player.get_stock_market_sell_action.return_value =\
            StockMarketSellAction(StockMarketSellActionType.SELL, OrderedDict([('PRR', 1)]))
        self.assertEqual({}, player.share_map)
        self.assertEqual([], player.presiding_companies)
        self.assertEqual(1200, player.money)

        # call
        self.assertRaises(InvalidOperationException, e30.main.process_stock_round_sell_action, self.game_state,
                          player)
        self.assertEqual({}, player.share_map)
        self.assertEqual([], player.presiding_companies)
        self.assertEqual(1200, player.money)

    def test_process_stock_round_sell_action_cannot_sell_president_certs(self):
        player = self.game_state.get_priority_deal_player()
        player.presiding_companies = ['PRR']
        self.game_state.companies_map['PRR'].owning_players = [player]
        self.game_state.companies_map['PRR'].president = player
        # sell unowned company
        player.get_stock_market_sell_action.return_value = \
            StockMarketSellAction(StockMarketSellActionType.SELL, OrderedDict([('PRR', 1)]))
        self.assertEqual({}, player.share_map)
        self.assertEqual(1200, player.money)

        # call
        self.assertRaises(InvalidOperationException, e30.main.process_stock_round_sell_action, self.game_state,
                          player)
        self.assertEqual({}, player.share_map)
        self.assertEqual(1200, player.money)

    def test_process_stock_round_sell_action_requested_sales_more_than_owned(self):
        player = self.game_state.get_priority_deal_player()
        player.share_map = {'PRR': 1}
        # sell more than owned
        player.get_stock_market_sell_action.return_value = \
            StockMarketSellAction(StockMarketSellActionType.SELL, OrderedDict([('PRR', 2)]))
        self.assertEqual([], player.presiding_companies)
        self.assertEqual(1200, player.money)

        # call
        self.assertRaises(InvalidOperationException, e30.main.process_stock_round_sell_action, self.game_state,
                          player)
        self.assertEqual({'PRR': 1}, player.share_map)
        self.assertEqual([], player.presiding_companies)
        self.assertEqual(1200, player.money)

    def test_process_stock_round_sell_action_requested_sales_over_bank_pool_limit(self):
        player = self.game_state.get_priority_deal_player()
        player.share_map = {'PRR': 1}
        self.game_state.companies_map['PRR'].market_shares = 5
        # sell such that bank pool has greater than limit of 5
        player.get_stock_market_sell_action.return_value = \
            StockMarketSellAction(StockMarketSellActionType.SELL, OrderedDict([('PRR', 1)]))
        self.assertEqual([], player.presiding_companies)
        self.assertEqual(1200, player.money)

        # call
        self.assertRaises(InvalidOperationException, e30.main.process_stock_round_sell_action, self.game_state,
                          player)
        self.assertEqual({'PRR': 1}, player.share_map)
        self.assertEqual([], player.presiding_companies)
        self.assertEqual(1200, player.money)

    def test_process_stock_round_sell_action_over_total_cert_limit(self):
        # 2 player limit is 28
        player = self.game_state.get_priority_deal_player()
        # player owns 30 certs
        player.share_map = {'PRR': 5, 'NYC': 5, 'CPR': 5, 'B&O': 5, 'C&O': 5, 'Erie': 5}
        companies = ['PRR', 'NYC', 'CPR', 'B&O', 'C&O', 'Erie']
        # non-count excluded stock market slots
        for c in companies:
            self.game_state.companies_map[c].current_share_price = self.game_state.stock_market.node_map["350A"]
        # selling 1, still over 28 certs
        player.get_stock_market_sell_action.return_value = \
            StockMarketSellAction(StockMarketSellActionType.SELL, OrderedDict([('PRR', 1)]))
        self.assertEqual(1200, player.money)

        # call
        self.assertRaises(InvalidOperationException, e30.main.process_stock_round_sell_action, self.game_state,
                          player)
        self.assertEqual({'PRR': 5, 'NYC': 5, 'CPR': 5, 'B&O': 5, 'C&O': 5, 'Erie': 5}, player.share_map)
        self.assertEqual([], player.presiding_companies)
        self.assertEqual(1200, player.money)
        for c in companies:
            self.assertEqual(self.game_state.stock_market.node_map["350A"],
                             self.game_state.companies_map[c].current_share_price)

    def test_process_stock_round_sell_action_over_total_cert_limit_with_presidential_certs(self):
        # 2 player limit is 28
        player = self.game_state.get_priority_deal_player()
        # player owns 30 certs including president certs
        player.share_map = {'PRR': 4, 'NYC': 4, 'CPR': 4, 'B&O': 4, 'C&O': 4, 'Erie': 4}
        player.presiding_companies = ['PRR', 'NYC', 'CPR', 'B&O', 'C&O', 'Erie']
        companies = ['PRR', 'NYC', 'CPR', 'B&O', 'C&O', 'Erie']
        # non-count excluded stock market slots
        for c in companies:
            self.game_state.companies_map[c].current_share_price = self.game_state.stock_market.node_map["350A"]
            self.game_state.companies_map[c].president = player
            self.game_state.companies_map[c].owning_players = [player]
        # selling 1, still over 28 certs
        player.get_stock_market_sell_action.return_value = \
            StockMarketSellAction(StockMarketSellActionType.SELL, OrderedDict([('PRR', 1)]))
        self.assertEqual(1200, player.money)

        # call
        self.assertRaises(InvalidOperationException, e30.main.process_stock_round_sell_action, self.game_state,
                          player)
        self.assertEqual({'PRR': 4, 'NYC': 4, 'CPR': 4, 'B&O': 4, 'C&O': 4, 'Erie': 4}, player.share_map)
        self.assertEqual(['PRR', 'NYC', 'CPR', 'B&O', 'C&O', 'Erie'], player.presiding_companies)
        self.assertEqual(1200, player.money)
        for c in companies:
            self.assertEqual(self.game_state.stock_market.node_map["350A"],
                             self.game_state.companies_map[c].current_share_price)

    def test_process_stock_round_sell_action_over_total_cert_limit_presidential_transfer(self):
        # 2 player limit is 28
        player = self.game_state.get_priority_deal_player()
        player2 = self.game_state.get_next_player(player)
        # player owns 29 certs, sell 1, transfer president of 1, still over the limit
        player.share_map = {'PRR': 3, 'NYC': 4, 'CPR': 4, 'B&O': 4, 'C&O': 4, 'Erie': 4}
        player2.share_map = {'PRR': 5}
        player.presiding_companies = ['PRR', 'NYC', 'CPR', 'B&O', 'C&O', 'Erie']
        companies = ['PRR', 'NYC', 'CPR', 'B&O', 'C&O', 'Erie']
        # non-count excluded stock market slots
        for c in companies:
            self.game_state.companies_map[c].current_share_price = self.game_state.stock_market.node_map["350A"]
            self.game_state.companies_map[c].president = player
            self.game_state.companies_map[c].owning_players = [player]
        self.game_state.companies_map['PRR'].owning_players = [player, player2]
        player.get_stock_market_sell_action.return_value = \
            StockMarketSellAction(StockMarketSellActionType.SELL, OrderedDict([('PRR', 1)]))
        self.assertEqual(1200, player.money)
        self.assertEqual(1200, player2.money)

        # call
        self.assertRaises(InvalidOperationException, e30.main.process_stock_round_sell_action, self.game_state,
                          player)
        self.assertEqual({'PRR': 3, 'NYC': 4, 'CPR': 4, 'B&O': 4, 'C&O': 4, 'Erie': 4}, player.share_map)
        self.assertEqual({'PRR': 5}, player2.share_map)
        self.assertEqual(['PRR', 'NYC', 'CPR', 'B&O', 'C&O', 'Erie'], player.presiding_companies)
        self.assertEqual([player, player2], self.game_state.companies_map['PRR'].owning_players)
        self.assertEqual(player, self.game_state.companies_map['PRR'].president)
        self.assertEqual(1200, player.money)
        self.assertEqual(1200, player2.money)
        for c in companies:
            self.assertEqual(self.game_state.stock_market.node_map["350A"],
                             self.game_state.companies_map[c].current_share_price)

    def test_process_stock_round_sell_action_price_presidential_transfer_drop_color_exclusion_success(self):
        # 2 player limit is 28
        player = self.game_state.get_priority_deal_player()
        player2 = self.game_state.get_next_player(player)
        # player owns 29 certs, sell 1, transfer president of 1, still over the limit
        player.share_map = {'PRR': 3, 'NYC': 4, 'CPR': 4, 'B&O': 4, 'C&O': 4, 'Erie': 4}
        player2.share_map = {'PRR': 5}
        player.presiding_companies = ['PRR', 'NYC', 'CPR', 'B&O', 'C&O', 'Erie']
        companies = ['PRR', 'NYC', 'CPR', 'B&O', 'C&O', 'Erie']
        for c in companies:
            self.game_state.companies_map[c].current_share_price = self.game_state.stock_market.node_map["350A"]
            self.game_state.companies_map[c].president = player
            self.game_state.companies_map[c].owning_players = [player]
        # 1 drop brings color into count exclusion
        self.game_state.companies_map['PRR'].current_share_price = self.game_state.stock_market.node_map["67H"]
        self.game_state.companies_map['PRR'].owning_players = [player, player2]
        player.get_stock_market_sell_action.return_value = \
            StockMarketSellAction(StockMarketSellActionType.SELL, OrderedDict([('PRR', 1)]))
        self.assertEqual(9600, self.game_state.bank.money)
        self.assertEqual(1200, player.money)
        self.assertEqual(1200, player2.money)

        # call
        e30.main.process_stock_round_sell_action(self.game_state, player)

        self.assertEqual({'PRR': 4, 'NYC': 4, 'CPR': 4, 'B&O': 4, 'C&O': 4, 'Erie': 4}, player.share_map)
        self.assertEqual({'PRR': 3}, player2.share_map)
        self.assertEqual(['NYC', 'CPR', 'B&O', 'C&O', 'Erie'], player.presiding_companies)
        self.assertEqual(['PRR'], player2.presiding_companies)
        self.assertEqual([player, player2], self.game_state.companies_map['PRR'].owning_players)
        self.assertEqual(player2, self.game_state.companies_map['PRR'].president)
        self.assertEqual(1, self.game_state.companies_map['PRR'].market_shares)
        self.assertEqual(self.game_state.stock_market.node_map["60I"],
                         self.game_state.companies_map['PRR'].current_share_price)
        self.assertEqual(9600 - 67, self.game_state.bank.money)
        self.assertEqual(1200 + 67, player.money)
        self.assertEqual(1200, player2.money)
        companies.remove('PRR')
        for c in companies:
            self.assertEqual(self.game_state.stock_market.node_map["350A"],
                             self.game_state.companies_map[c].current_share_price)

    def test_process_stock_round_sell_action_price_drop_color_exclusion_success(self):
        # 2 player limit is 28
        player = self.game_state.get_priority_deal_player()
        # player owns 30 certs, sell 1, still over the limit
        player.share_map = {'PRR': 4, 'NYC': 4, 'CPR': 4, 'B&O': 4, 'C&O': 4, 'Erie': 4}
        player.presiding_companies = ['PRR', 'NYC', 'CPR', 'B&O', 'C&O', 'Erie']
        companies = ['PRR', 'NYC', 'CPR', 'B&O', 'C&O', 'Erie']
        for c in companies:
            self.game_state.companies_map[c].current_share_price = self.game_state.stock_market.node_map["350A"]
            self.game_state.companies_map[c].president = player
            self.game_state.companies_map[c].owning_players = [player]
        # 1 drop brings color into count exclusion
        self.game_state.companies_map['PRR'].current_share_price = self.game_state.stock_market.node_map["67H"]
        self.game_state.companies_map['PRR'].owning_players = [player]
        player.get_stock_market_sell_action.return_value = \
            StockMarketSellAction(StockMarketSellActionType.SELL, OrderedDict([('PRR', 1)]))
        self.assertEqual(9600, self.game_state.bank.money)
        self.assertEqual(1200, player.money)

        # call
        e30.main.process_stock_round_sell_action(self.game_state, player)

        self.assertEqual({'PRR': 3, 'NYC': 4, 'CPR': 4, 'B&O': 4, 'C&O': 4, 'Erie': 4}, player.share_map)
        self.assertEqual(['PRR', 'NYC', 'CPR', 'B&O', 'C&O', 'Erie'], player.presiding_companies)
        self.assertEqual([player], self.game_state.companies_map['PRR'].owning_players)
        self.assertEqual(player, self.game_state.companies_map['PRR'].president)
        self.assertEqual(1, self.game_state.companies_map['PRR'].market_shares)
        self.assertEqual(self.game_state.stock_market.node_map["60I"],
                         self.game_state.companies_map['PRR'].current_share_price)
        self.assertEqual(9600 - 67, self.game_state.bank.money)
        self.assertEqual(1200 + 67, player.money)
        companies.remove('PRR')
        for c in companies:
            self.assertEqual(self.game_state.stock_market.node_map["350A"],
                             self.game_state.companies_map[c].current_share_price)

    def test_process_stock_round_sell_action_price_success(self):
        player = self.game_state.get_priority_deal_player()
        # player owns 30 certs, sell 1, still over the limit
        player.share_map = {'PRR': 3}
        self.game_state.companies_map['PRR'].current_share_price = self.game_state.stock_market.node_map["350A"]
        self.game_state.companies_map['PRR'].owning_players = [player]
        player.get_stock_market_sell_action.return_value = \
            StockMarketSellAction(StockMarketSellActionType.SELL, OrderedDict([('PRR', 1)]))
        self.assertEqual(9600, self.game_state.bank.money)
        self.assertEqual(1200, player.money)

        # call
        e30.main.process_stock_round_sell_action(self.game_state, player)

        self.assertEqual({'PRR': 2}, player.share_map)
        self.assertEqual([], player.presiding_companies)
        self.assertEqual([player], self.game_state.companies_map['PRR'].owning_players)
        self.assertEqual(1, self.game_state.companies_map['PRR'].market_shares)
        self.assertEqual(self.game_state.stock_market.node_map["300B"],
                         self.game_state.companies_map['PRR'].current_share_price)
        self.assertEqual(9600 - 350, self.game_state.bank.money)
        self.assertEqual(1200 + 350, player.money)


if __name__ == '__main__':
    unittest.main()
