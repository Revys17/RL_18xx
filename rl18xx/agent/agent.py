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

    @abstractmethod
    def committed_action_dicts(self) -> list[dict]:
        """The engine action dicts the last ``play_move`` applied: the chosen
        action with its concrete price, then any forced-chain actions the agent
        auto-advanced through."""
        pass

    @abstractmethod
    def play_action_dicts(self, action_index: int, action_dicts: list[dict]):
        """Advance through a move another agent committed.

        ``action_dicts`` is that agent's ``committed_action_dicts()`` and must be
        applied verbatim — re-deriving the move from ``action_index`` alone
        loses the price of price-bearing actions (Bid, BuyTrain, BuyCompany).
        """
        pass
