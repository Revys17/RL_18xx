import game_state
from private_company import Private


class MH(Private):
    def __init__(self):
        super().__init__("Mohawk & Hudson", "MH", "May exchange for a 10% share of NYC (max 60% and shares must be " +
                         "available). May be done during player's stock turn or between other players' turns at any " +
                         "time. Closes the MH.", 110, 20, "D-18")

    def do_special_action(self, game_state: 'game_state.GameState'):
        pass
        # TODO: add private action
