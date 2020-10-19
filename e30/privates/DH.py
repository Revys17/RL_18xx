from e30 import game_state
from e30.private_company import Private


class DH(Private):
    def __init__(self):
        super().__init__("Delaware & Hudson", "DH", "Allows lay of track and station on DH's hex. Lay costs regular " +
                         "amount but station is free. Not an additional tile lay. Does not require route. Station " +
                         "must be placed immediately to utilize track-free allowance.", 70, 15, "F-16")

    def do_special_action(self, game_state: 'game_state.GameState'):
        game_state.board.lay_track(self.location)
        # TODO: add optional station lay
