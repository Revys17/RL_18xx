from game_state import GameState
from private_company import Private


class SV(Private):
    def __init__(self):
        super().__init__("Schuylkill Valley", "SV", "No special ability", 20, 5, "G-15")

    def do_special_action(self, game_state: GameState):
        pass
