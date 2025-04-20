from rl18xx.agent.alphazero.encoder import Encoder_1830
from rl18xx.game.gamemap import GameMap
from rl18xx.game.actionfinder import ActionHelper
from rl18xx.game.engine.game.base import BaseGame

def test_encoder():
    game_map = GameMap()
    game = game_map.game_by_title("1830")
    g = game({"1": "Player 1", "2": "Player 2", "3": "Player 3", "4": "Player 4"})
    action_helper = ActionHelper(g)
    encoder = Encoder_1830()
    encoding = encoder.encode(g)
    print(encoding)
    print(encoding.shape)
