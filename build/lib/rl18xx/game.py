
__all__ = ['TITLE_DIR', 'TITLE_MODULE', 'game_modules', 'import_game_modules', 'Engine']


import importlib
import os


TITLE_DIR = "engine/game/titles"
TITLE_MODULE = "game.engine.game.title"


def import_game_modules():
    game_modules = {}
    for filename in os.listdir(TITLE_DIR):
        if filename.endswith('.ipynb'):
            module_name = filename[3:-6]
            module_path = f"{TITLE_MODULE}.{module_name}"
            game_modules[module_name] = importlib.import_module(module_path)
    return game_modules

game_modules = import_game_modules()


class Engine:
    def __init__(self):
        self.games = {}
        self.game_meta_by_title = self._collect_game_meta()

    def _collect_game_meta(self):
        game_meta_by_title = {}
        for name, module in game_modules.items():
            if hasattr(module, "Meta"):
                meta = getattr(module, "Meta")
                game_meta_by_title[meta.title] = meta
        return game_meta_by_title

    def game_by_title(self, title):
        # Directly return the game object by its title
        if title in self.game_meta_by_title:
            game_meta = self.game_meta_by_title[title]
            # Assuming a game class or factory method is defined in the module
            if hasattr(game_modules[game_meta.module_name], "Game"):
                game_class = getattr(game_modules[game_meta.module_name], "Game")
                if title not in self.games:
                    self.games[
                        title
                    ] = game_class()  # Initialize the game if not already done
                return self.games[title]
        return None
