# Game Usage

This file contains detailed instructions and examples for how to use the game engine implementation.

## Game Initialization

Initialize a new game using the following code:
```python
from rl18xx.game.gamemap import GameMap
game_map = GameMap()
game_class = game_map.game_by_title("1830")
players = {"1": "Player 1", "2": "Player 2", "3": "Player 3", "4": "Player 4"}
game = game_class(players)
```

To load a game from a set of actions, use the following code instead:
```python
from rl18xx.game.engine.game.base import BaseGame
game_actions = [] # This should contain your list of actions, either in object or dict form.
game = BaseGame.load(
    {
        "id": "hs_inzqxyla_1708318031",
        "players": [{"name": "Player 1", "id": "1"}, {"name": "Player 2", "id": "2"}, {"name": "Player 3", "id": "3"}, {"name": "Player 4", "id": "4"}],
        "title": "1830",
        "description": "",
        "min_players": "4",
        "max_players": "4",
        "settings": {
            "optional_rules": [],
            "seed": ""
        },
        "actions": game_actions
    })
```

Finally, if you want to fork a game to try multiple actions, you can use the `clone` method:
```python
game_clone = game.clone(game.raw_actions)
```

## Game Actions

Game actions are most easily created by using the `ActionHelper` class.

```python
from rl18xx.game import ActionHelper
action_helper = ActionHelper()
all_actions = action_helper.get_all_choices(game)
# This contains a list of all possible legal actions for the current player
# To play an action, use `game.process_action(action)`
game.process_action(all_actions[0])
```

## File Structure

### Base Game

The base game implementation is located in `rl18xx/game/engine/game/base.py`. This is a very long and complex file (unfortunately), so it may be difficult to navigate. Here are the key pieces you need to know, however:

- `BaseGame` is the base class for all game implementations.
- Game-specific data is stored in `rl18xx/game/engine/game/title/`. For example, `g1830.py` is the implementation for 1830.
- You can see the current round by using `game.round`.
- You can see the current step by using `game.active_step()` or `game.round.active_step()`
- You can see the current active entity (player, company, or corporation) by using `game.current_entity`
- You can see the active player by using `game.active_players()[0]`
- You can get actions from ActionHelper as shown in the previous section and apply them using `game.process_action(action)`.
- If you want to see all the actions taken in the game, you can use `game.raw_actions`
- You can see how the game class is encoded into a model-friendly tensor by looking at `rl18xx/agent/alphazero/encoder.py`.
- You can see how actions are encoded into a model-friendly tensor by looking at `rl18xx/agent/alphazero/action_mapper.py`.

### Action Helper

The `ActionHelper` class is used to get all possible legal actions for the current player. It is located in `rl18xx/game/action_helper.py`.

### Game Rounds and Steps

The game class contains most of the core logic for the game itself, but each game step has custom logic that is implemented in `rl18xx/game/engine/round.py`. For example, the game begins during an `Auction` round and `WaterfallAuction` step, and the logic for which actions are possible (Bid, Par, Pass) and how to process each action is implemented in `round.py`.

### Game Entities

Various game entities that are used across the game implementation are defined in two places: `rl18xx/game/engine/entities.py` and `rl18xx/game/engine/core.py`. This contains the base classes for players, companies, corporations, the share pool, the train depot, and so forth.

### Game Graph

The game graph is defined in `rl18xx/game/engine/graph.py`. This contains all of the data structures used for creating the game map, including Hexes, Tiles, and tile Parts such as revenue locations and paths. The methods in Hex, Tile, Node, and Path are used extensively for the operating round logic of the game to determine which tiles are available, which locations and rotations are legal, which paths and nodes are connected, and so forth. When it comes to route finding, `Node` and `Path` are the most important classes.

### Auto-router

The auto-router is defined in `rl18xx/game/engine/auto_router.py`. This contains the logic for finding the most optimal set of routes for a given corporation's trains. It relies on the `Node` and `Path` classes for route finding, using their `walk` method in particular.

## Sample Game

A sample game can be found in the test file `tests/game_test.py` under the test `test_1830_manual_actions`. This test contains a full game played out with manual actions (until at least SR 7 at which point the players all pass until the game ends). The game is played out using the `ActionHelper` class to get all possible actions and then applying them manually.
