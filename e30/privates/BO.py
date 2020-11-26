from e30 import game_state
from e30.private_company import Private


class BO(Private):
    def __init__(self):
        super().__init__("Baltimore & Ohio", "BO", "Purchaser receives the president's certificate of the B&O " +
                         "railroad and immediately sets the par value. This company may not be sold or traded. " +
                         "Closes when the BO buys its first train.", 220, 30, "I-15")

    def do_special_action(self, game_state: 'game_state.GameState'):
        # after the private auction presidency of B&O goes to the owner at no cost
        # set par value and initial share price elsewhere
        self.owner.presiding_companies = ['B&O']
        game_state.companies_map['B&O'].owning_players.append(self.owner)
        game_state.companies_map['B&O'].president = self.owner
