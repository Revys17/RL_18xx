from e30 import game_state
from e30.private_company import Private


class CS(Private):
    def __init__(self):
        super().__init__("Champlain & St. Lawrence", "CS", "Additional track lay on home tile. No connecting track " +
                                                           "needed.", 40, 10, "B-20")

    def do_special_action(self, game_state: 'game_state.GameState'):
        game_state.board.lay_additional_track(self.location)
