from enum import Enum
from typing import Optional, Union, Tuple


class StockMarketSlotColor(Enum):
    BROWN = 0
    ORANGE = 1
    YELLOW = 2
    WHITE = 3


class StockMarketSlot:
    def __init__(self, location: str, up=None, right=None, down=None, left=None):
        self.location: str = location
        # value has an int component, and a str component to break ties
        self.value: Tuple[int, str] = (int(location[:-1]), location[-1])
        self.up: StockMarketSlot = up
        self.right: StockMarketSlot = right
        self.down: StockMarketSlot = down
        self.left: StockMarketSlot = left

    def set_up(self, node: Union[str, 'StockMarketSlot']) -> None:
        self.up = self.set_dir(node)

    def set_right(self, node: Union[str, 'StockMarketSlot']) -> None:
        self.right = self.set_dir(node)

    def set_down(self, node: Union[str, 'StockMarketSlot']) -> None:
        self.down = self.set_dir(node)

    def set_left(self, node: Union[str, 'StockMarketSlot']) -> None:
        self.left = self.set_dir(node)

    def set_dir(self, node: Union[str, 'StockMarketSlot']) -> Optional['StockMarketSlot']:
        if node is None or node == "self":
            return self
        return node

    def get_value(self) -> Tuple[int, str]:
        return self.value

    def get_color(self) -> StockMarketSlotColor:
        if self.value[0] <= 30:
            return StockMarketSlotColor.BROWN
        if self.value[0] <= 45:
            return StockMarketSlotColor.BROWN
        if self.value[0] <= 60:
            return StockMarketSlotColor.YELLOW
        return StockMarketSlotColor.WHITE

    def copy(self) -> 'StockMarketSlot':
        return StockMarketSlot(self.location, self.up, self.right, self.down, self.left)

    def get_share_price_after_sale(self, num_shares_sold: int) -> 'StockMarketSlot':
        share_price = self
        for _ in range(num_shares_sold):
            share_price = self.down
        return share_price

    def ignore_total_cert_limit(self) -> bool:
        return self.get_color() is not StockMarketSlotColor.WHITE

    def can_buy_multiple_certs(self) -> bool:
        return self.get_color() is StockMarketSlotColor.BROWN

    def ignore_company_ownership_percent_limit(self) -> bool:
        return self.get_color() is StockMarketSlotColor.BROWN or self.get_color() is StockMarketSlotColor.ORANGE
