import unittest

from src.bank import Bank


class BankTest(unittest.TestCase):

    def test_bank_money(self):
        self.assertEqual(9600, Bank().money)


if __name__ == '__main__':
    unittest.main()
