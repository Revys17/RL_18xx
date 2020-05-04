from action_blob import ActionBlob
from game_state import GameState


class Agent(object):
    def get_action_blob(self, game_state: GameState):
        pass


class HumanAgent(Agent):
    def get_action_blob(self, game_state: GameState):
        # TODO prompt command line for decision
        return ActionBlob


class AIAgent(Agent):
    def get_action_blob(self, game_state: GameState):
        # TODO game_state -> feature map -> AI processing -> ActionBlob
        return ActionBlob
