import unittest

import e30.bank as bank


class BankTest(unittest.TestCase):

    def test_bank_money(self):
        self.assertEqual(9600, bank.Bank().money)


if __name__ == '__main__':
    unittest.main()
