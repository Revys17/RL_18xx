class StockMarketSlot:
    def __init__(self, location):
        self.location = location
        self.value = self.get_value(location)
        self.up = self
        self.right = self
        self.down = self
        self.left = self

    def set_up(self, node):
        self.up = self.set_dir(node)

    def set_right(self, node):
        self.right = self.set_dir(node)

    def set_down(self, node):
        self.down = self.set_dir(node)

    def set_left(self, node):
        self.left = self.set_dir(node)

    def set_dir(self, node):
        if node is None or node == "self":
            return self
        return node

    def get_value(self, location):
        return int(location[:-1])

    def get_color(self):
        if self.value <= 30:
            return "brown"
        if self.value <= 45:
            return "orange"
        if self.value <= 60:
            return "yellow"
        return "white"