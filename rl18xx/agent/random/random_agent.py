from rl18xx.agent.agent import Agent
from rl18xx.agent.alphazero.action_mapper import ActionMapper
from rl18xx.game.engine.game import BaseGame
from rl18xx.game.gamemap import GameMap

import random
from typing import Optional

class RandomPlayer(Agent):
    def __init__(self, game_state: Optional[BaseGame] = None):
        self.game_state = game_state
        self.action_mapper = ActionMapper()
        if self.game_state is None:
            self.game_state = self.get_fresh_game_state()
        self.legal_action_indices = self.action_mapper.get_legal_action_indices(self.game_state)

    def __str__(self):
        return f"RandomPlayer"
    
    def __repr__(self):
        return self.__str__()

    def get_fresh_game_state(self):
        game_map = GameMap()
        game_class = game_map.game_by_title("1830")
        players = {1: "Player 1", 2: "Player 2", 3: "Player 3", 4: "Player 4"}
        return game_class(players)

    def initialize_game(self, game_state: BaseGame):
        self.game_state = game_state
        self.legal_action_indices = self.action_mapper.get_legal_action_indices(self.game_state)

    def get_game_state(self):
        return self.game_state

    def suggest_move(self):
        if self.legal_action_indices is None:
            raise ValueError("Legal action indices not set. Call initialize_game first.")
        return random.choice(self.legal_action_indices)

    def play_move(self, action_index):
        if self.legal_action_indices is None:
            raise ValueError("Legal action indices not set. Call initialize_game first.")
        new_position = self.game_state.pickle_clone()
        # Serialize before processing: a Rust-adapter LayTile resolves its tile
        # instance lazily, so after the lay it would name the next free copy.
        self._last_action_dict = self.action_mapper.map_index_to_action(action_index, new_position).to_dict()
        new_position.process_action(self._last_action_dict)
        self.game_state = new_position
        self.legal_action_indices = self.action_mapper.get_legal_action_indices(self.game_state)

    def committed_action_dicts(self) -> list[dict]:
        return [self._last_action_dict]

    def play_action_dicts(self, action_index: int, action_dicts: list[dict]):
        new_position = self.game_state.pickle_clone()
        for action_dict in action_dicts:
            new_position.process_action(action_dict)
        self.game_state = new_position
        self.legal_action_indices = self.action_mapper.get_legal_action_indices(self.game_state)
