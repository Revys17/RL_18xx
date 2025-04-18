
__all__ = ['TITLE_DIR', 'TITLE_MODULE', 'GameMap']


import importlib
import os
from pathlib import Path


TITLE_DIR = "engine/game/title"
TITLE_MODULE = "rl18xx.game.engine.game.title"


class GameMap:
    def __init__(self):
        self.game_modules = self._import_game_modules()
        self.game_meta = self._collect_game_meta()
        self.games = self._load_games()

    def _import_game_modules(self):
        game_modules = {}
        for filename in os.listdir(Path(__file__).parent/TITLE_DIR):
            if filename.endswith(".ipynb"):
                module_name = filename[3:-6]
                module_path = f"{TITLE_MODULE}.{module_name}"
                game_modules[module_name] = importlib.import_module(module_path)
            elif filename.endswith(".py"):
                module_name = filename[0:-3]
                module_path = f"{TITLE_MODULE}.{module_name}"
                game_modules[module_name] = importlib.import_module(module_path)
        return game_modules

    def _collect_game_meta(self):
        game_meta_by_title = {}
        for name, module in self.game_modules.items():
            if hasattr(module, "Meta"):
                meta = getattr(module, "Meta")
                game_meta_by_title[meta.title()] = meta
        return game_meta_by_title

    def _load_games(self):
        games = {}
        for title in self.game_meta:
            if hasattr(self.game_modules["g" + title], "Game"):
                games[title] = getattr(self.game_modules["g" + title], "Game")
        return games

    def meta_by_title(self, title):
        return self.game_meta.get(title)

    def game_by_title(self, title):
        return self.games.get(title)
