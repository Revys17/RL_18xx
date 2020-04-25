from typing import List, Dict

from stock_market_slot import StockMarketSlot


class StockMarket:
    def __init__(self):
        market: List[List[str]] = [
            ["60A", "67A", "71A", "76A", "82A", "90A", "100A", "112A", "126A", "142A", "160A", "180A", "200A", "225A", "250A", "275A", "300A", "325A", "350A"],
            ["53B", "60B", "66B", "70B", "76B", "82B", "90B",  "100B", "112B", "126B", "142B", "160B", "180B", "200B", "220B", "240B", "260B", "280B", "300B"],
            ["46C", "55C", "60C", "65C", "70C", "76C", "82C",  "90C",  "100C", "111C", "125C", "140C", "155C", "170C", "185C", "200C", None,   None,   None  ],
            ["39D", "48D", "54D", "60D", "66D", "71D", "76D",  "82D",  "90D",  "100D", "110D", "120D", "130D", None,   None,   None,   None,   None,   None  ],
            ["32E", "41E", "48E", "55E", "62E", "67E", "71E",  "76E" , "82E",  "90E",  "100E", None,   None,   None,   None,   None,   None,   None,   None  ],
            ["25F", "34F", "42F", "50F", "58F", "65F", "67F",  "71F",  "75F",  "80F",  None,   None,   None,   None,   None,   None,   None,   None,   None  ],
            ["18G", "27G", "36G", "45G", "54G", "63G", "67G",  "69G",  "70G",  None,   None,   None,   None,   None,   None,   None,   None,   None,   None  ],
            ["10H", "20H", "30H", "40H", "50H", "60H", "67H",  "68H",  None,   None,   None,   None,   None,   None,   None,   None,   None,   None,   None  ],
            [None,  "10I", "20I", "30I", "40I", "50I", "60I",  None,   None,   None,   None,   None,   None,   None,   None,   None,   None,   None,   None  ],
            [None,  None,  "10J", "20J", "30J", "40J", "50J",  None,   None,   None,   None,   None,   None,   None,   None,   None,   None,   None,   None  ],
            [None,  None,  None,  "10K", "20K", "30K", "40K",  None,   None,   None,   None,   None,   None,   None,   None,   None,   None,   None,   None  ]
        ]

        self.node_map: Dict[str, StockMarketSlot] = {}
        market_slots = []

        for i in range(len(market)):
            market_slots_row = []
            for j in range(market[i]):
                if market[i][j] is None:
                    market_slots.append(None)
                else:
                    slot = StockMarketSlot(market[i][j])
                    self.node_map[market[i][j]] = slot
                    market_slots.append(slot)
            market_slots.append(market_slots_row)

        for i in range(len(market_slots)):
            for j in range(len(market_slots[i])):
                slot = market_slots[i][j]
                if slot is None:
                    continue

                if i == 0:
                    up = "self"
                else:
                    up = market_slots[i-1][j]
                slot.set_up(up)

                if j == len(market_slots[i]) - 1:
                    right = up
                else:
                    right = market_slots[i][j+1]
                slot.set_right(right)

                if i == len(market_slots) - 1:
                    down = "self"
                else:
                    down = market_slots[i+1][j]
                slot.set_down(down)

                if j == 0:
                    left = down
                else:
                    left = market_slots[i][j-1]
                slot.set_left(left)

        self.par_locations = {
            100: self.node_map["100A"],
            90:  self.node_map["90B"],
            82:  self.node_map["82C"],
            76:  self.node_map["76D"],
            71:  self.node_map["71E"],
            67:  self.node_map["67F"],
        }

    def get_par_value_slot(self, value: int) -> StockMarketSlot:
        return self.par_locations[value]

    def get_share_price_after_sale(self, current_price, num_shares_sold):
        for x in num_shares_sold:
            current_price = current_price.down
        return current_price

    def get_price_after_dividends(self, current_price, dividends_paid):
        if dividends_paid:
            return current_price.right
        return current_price.left

    def get_price_for_fully_owned(self, current_price, fully_owned):
        if fully_owned:
            return current_price.up
        return current_price