from typing import Optional, Union


class StockMarketSlot:
    def __init__(self, location: str):
        self.location: str = location
        self.value: int = self.get_value(location)
        self.up: StockMarketSlot = self
        self.right: StockMarketSlot = self
        self.down: StockMarketSlot = self
        self.left: StockMarketSlot = self

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

    def get_value(self, location: str) -> int:
        return int(location[:-1])

    def get_color(self) -> str:
        if self.value <= 30:
            return "brown"
        if self.value <= 45:
            return "orange"
        if self.value <= 60:
            return "yellow"
        return "white"
