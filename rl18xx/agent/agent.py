from abc import ABC, abstractmethod
from rl18xx.game.engine.game import BaseGame

class Agent(ABC):
    @abstractmethod
    def initialize_game(self, game_state: BaseGame):
        pass

    @abstractmethod
    def get_game_state(self):
        pass

    @abstractmethod
    def suggest_move(self):
        pass

    @abstractmethod
    def play_move(self, action_index: int):
        pass