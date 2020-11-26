import unittest
from typing import Dict
from unittest.mock import MagicMock

from e30.agent import HumanAgent
from e30.exceptions.exceptions import InvalidOperationException
from e30.game_state import GameState
from e30.stock_market_slot import StockMarketSlot


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

    def test_companies_pq(self):
        [self.assertFalse(hasattr(company, 'current_share_price')) for company in self.test_game_state.companies_pq]

        # heapq without values has default order
        self.assertEqual(["PRR", "NYC", "CPR", "B&O", "C&O", "Erie", "NNH", "B&M"], self.get_company_names())

        # sort descending order should not change order
        self.test_game_state.re_sort_companies()
        self.assertEqual(["PRR", "NYC", "CPR", "B&O", "C&O", "Erie", "NNH", "B&M"], self.get_company_names())

        # set values and re-sort and assert order
        company_value_map: Dict[str, StockMarketSlot] = {
            # "PRR" - remains undefined, order maintained - 7
            # "NYC" - remains undefined, order maintained - 8
            "CPR": StockMarketSlot('100C'),  # third highest value, same int component, lower str component - 6
            "B&O": StockMarketSlot('100D'),  # third highest value, same int component, higher str component - 5
            "C&O": StockMarketSlot('200B'),  # second highest value tied, order maintained - 2
            "Erie": StockMarketSlot('200B'),  # second highest value tied, order maintained - 3
            "NNH": StockMarketSlot('200B'),  # second highest value tied, order maintained - 4
            "B&M": StockMarketSlot('350A')  # highest value - 1
        }

        for company in self.test_game_state.companies_pq:
            if company.short_name in company_value_map:
                company.current_share_price = company_value_map.get(company.short_name)

        # apply heap sort
        self.test_game_state.re_sort_companies()
        self.assertEqual(["B&M", "C&O", "Erie", "NNH", "B&O", "CPR", "PRR", "NYC"], self.get_company_names())

    def get_company_names(self):
        return [company.short_name for company in self.test_game_state.companies_pq]

    def test_get_total_num_non_excluded_certs_none_excluded(self):
        self.player.share_map = {'PRR': 4, 'NYC': 4}
        self.player.presiding_companies = ['PRR', 'B&O']
        self.test_game_state.companies_map['PRR'].current_share_price = \
            self.test_game_state.stock_market.node_map["350A"]
        self.test_game_state.companies_map['NYC'].current_share_price = \
            self.test_game_state.stock_market.node_map["350A"]
        self.test_game_state.companies_map['B&O'].current_share_price = \
            self.test_game_state.stock_market.node_map["350A"]

        # call
        # PRR president, B&O president, + shares
        self.assertEqual(1 + 1 + 4 + 4, self.test_game_state.get_total_num_non_excluded_certs(self.player))

    def test_get_total_num_non_excluded_certs_company_excluded(self):
        self.player.share_map = {'PRR': 4, 'NYC': 4}
        self.player.presiding_companies = ['PRR', 'B&O']
        self.test_game_state.companies_map['PRR'].current_share_price = \
            self.test_game_state.stock_market.node_map["60I"]
        self.test_game_state.companies_map['NYC'].current_share_price = \
            self.test_game_state.stock_market.node_map["350A"]
        self.test_game_state.companies_map['B&O'].current_share_price = \
            self.test_game_state.stock_market.node_map["350A"]

        # call
        # B&O president + shares, PRR excluded
        self.assertEqual(1 + 4, self.test_game_state.get_total_num_non_excluded_certs(self.player))

    def test_has_presidency_transfer_already_president(self):
        self.player.presiding_companies = ['PRR']

        # call
        self.assertFalse(self.test_game_state.has_presidency_transfer('PRR', 0, self.player))

    def test_has_presidency_transfer_equal_ownership_no_transfer(self):
        self.player.share_map = {'PRR': 4}
        self.player2.share_map = {'PRR': 3}
        self.test_game_state.companies_map['PRR'].president = self.player2

        # call
        self.assertFalse(self.test_game_state.has_presidency_transfer('PRR', 1, self.player))

    def test_has_presidency_transfer(self):
        self.player.share_map = {'PRR': 5}
        self.player2.share_map = {'PRR': 2}
        self.test_game_state.companies_map['PRR'].president = self.player2

        # call
        self.assertTrue(self.test_game_state.has_presidency_transfer('PRR', 1, self.player))

    def test_has_presidency_transfer_on_num_buy(self):
        self.player2.share_map = {'PRR': 2}
        self.test_game_state.companies_map['PRR'].president = self.player2

        # call
        self.assertTrue(self.test_game_state.has_presidency_transfer('PRR', 5, self.player))

    def validate_ownership_percent_and_total_cert_limit_ignore_all_limits(self):
        self.test_game_state.companies_map['PRR'].current_share_price = \
            self.test_game_state.stock_market.node_map["45G"]

        # call
        self.test_game_state.validate_ownership_percent_and_total_cert_limit('PRR', self.player, True)

    def validate_ownership_percent_and_total_cert_limit_under_all_limits(self):
        companies = ['PRR', 'NYC', 'CPR', 'B&O', 'C&O', 'Erie']
        for c in companies:
            self.test_game_state.companies_map[c].current_share_price = \
                self.test_game_state.stock_market.node_map["350A"]
        self.player.presiding_companies = ['PRR']
        # 27 certs, 1 under the limit, limit reached when buying
        self.player.share_map = {'PRR': 3, 'NYC': 3, 'CPR': 5, 'B&O': 5, 'C&O': 5, 'Erie': 5}

        # call
        self.test_game_state.validate_ownership_percent_and_total_cert_limit('PRR', self.player, False)

    def validate_ownership_percent_and_total_cert_limit_over_ownership_percent_limit(self):
        self.test_game_state.companies_map['PRR'].current_share_price = \
            self.test_game_state.stock_market.node_map["350A"]
        self.player.presiding_companies = ['PRR']
        self.player.share_map = {'PRR': 4}

        # call
        self.assertRaises(
            InvalidOperationException, self.test_game_state.validate_ownership_percent_and_total_cert_limit, 'PRR',
            self.player, False)

    def validate_ownership_percent_and_total_cert_limit_over_total_cert_limit(self):
        companies = ['PRR', 'NYC', 'CPR', 'B&O', 'C&O', 'Erie']
        for c in companies:
            self.test_game_state.companies_map[c].current_share_price = \
                self.test_game_state.stock_market.node_map["350A"]
        self.player.presiding_companies = ['PRR']
        # 28 certs, limit exceeded when buying
        self.player.share_map = {'PRR': 3, 'NYC': 4, 'CPR': 5, 'B&O': 5, 'C&O': 5, 'Erie': 5}

        # call
        self.assertRaises(
            InvalidOperationException, self.test_game_state.validate_ownership_percent_and_total_cert_limit, 'PRR',
            self.player, False)

    def validate_ownership_percent_and_total_cert_limit_over_total_cert_limit_but_presidential_transfer(self):
        companies = ['PRR', 'NYC', 'CPR', 'B&O', 'C&O', 'Erie']
        for c in companies:
            self.test_game_state.companies_map[c].current_share_price = \
                self.test_game_state.stock_market.node_map["350A"]
        # 28 certs, limit exceeded when buying, but presidential transfer brings it back
        self.player.share_map = {'PRR': 3, 'NYC': 4, 'CPR': 5, 'B&O': 5, 'C&O': 5, 'Erie': 5}

        # call
        self.test_game_state.validate_ownership_percent_and_total_cert_limit('PRR', self.player, True)


if __name__ == '__main__':
    unittest.main()
