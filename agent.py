from actions.action_blob import ActionBlob
from game_state import GameState


class Agent(object):
    def get_action_blob(self, game_state: GameState):
        pass

    def get_bid_buy_action(self, game_state: GameState):
        pass

    def get_bid_resolution_action(self, game_state: GameState):
        pass


class HumanAgent(Agent):
    def get_action_blob(self, game_state: GameState):
        # TODO prompt command line for decision
        return ActionBlob

    def get_bid_buy_action(self, game_state: GameState):
        # TODO prompt command line for decision
        # passing in unowned privates, give some choices for actions and possibilities, descriptions and guidelines
        # unit tests should be able to mock this action without defining another mock agent type for unit tests
        pass

    def get_bid_resolution_action(self, game_state: GameState):
        # TODO prompt command line for decision
        pass


class AIAgent(Agent):
    def get_action_blob(self, game_state: GameState):
        # TODO game_state -> feature map -> AI processing -> ActionBlob
        return ActionBlob
