from game_state import GameState
from private_company import Private


class CA(Private):
    def __init__(self):
        super().__init__("Camden & Amboy", "CA", "Purchaser receives 10% of the PRR.", 160, 25, "H-18")

    def do_special_action(self, game_state: GameState):
        pass
