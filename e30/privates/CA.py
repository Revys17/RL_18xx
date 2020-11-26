from e30 import game_state
from e30.private_company import Private


class CA(Private):
    def __init__(self):
        super().__init__("Camden & Amboy", "CA", "Purchaser receives 10% of the PRR.", 160, 25, "H-18")

    def do_special_action(self, game_state: 'game_state.GameState'):
        # after the private auction 10% shares of PRR go to the owner at no cost
        self.owner.share_map = {'PRR': 1}
        game_state.companies_map['PRR'].owning_players.append(self.owner)
        game_state.companies_map['PRR'].ipo_shares -= 1
