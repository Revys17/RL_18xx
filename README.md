# 18XX Reinforcement Learning Agent

This repository contains a Python implementation of the 1830 board game engine (ported from the Ruby implementation in [tobymao/18xx](https://github.com/tobymao/18xx)) and an AlphaZero-style agent trained via self-play reinforcement learning.

## Setup

```bash
# Requires Python 3.11+ and uv
uv sync
```

## Usage

All commands are run through `main.py`:

```bash
# Train the agent (self-play + network training loop)
uv run python main.py train --iterations 5 --games 25 --threads 2 --readouts 64

# Pre-train from human game data
uv run python main.py pretrain --data-dir human_games --epochs 10

# Run an arena match between agents
uv run python main.py arena --agents mcts mcts random random --readouts 200

# Start the training dashboard (http://localhost:5001)
uv run python main.py dashboard

# Replay a game log in the browser
uv run python main.py replay path/to/game.log
```

Run `uv run python main.py --help` or `uv run python main.py <command> --help` for full options.

### Services

`startup.sh` launches both TensorBoard (port 6006) and the training dashboard (port 5001) as background processes.

### Game Engine

The game engine can also be used directly:

```python
from rl18xx.game.gamemap import GameMap
from rl18xx.game.action_helper import ActionHelper

game = GameMap().game_by_title("1830")({1: "P1", 2: "P2", 3: "P3", 4: "P4"})
action_helper = ActionHelper()

# Get all legal actions and play one
actions = action_helper.get_all_choices(game)
game.process_action(actions[0])
```

See [game_usage.md](game_usage.md) for detailed engine documentation.

## Running Tests

```bash
uv run pytest tests/                           # All tests
uv run pytest tests/agent/alphazero/mcts_test.py  # Specific file
uv run pytest tests/agent/alphazero/mcts_test.py::test_select_leaf  # Specific test
```

## Project Structure

```
rl18xx/
  game/engine/       Game engine (rules, entities, graph, actions, rounds)
  agent/alphazero/   AlphaZero agent (model, MCTS, encoder, self-play, training)
  agent/arena.py     Agent vs agent matches
  agent/dashboard/   Flask training dashboard
  client/            Integration with 18xx.games online platform
tests/               Test suite
main.py              CLI entry point
startup.sh           Service launcher (TensorBoard + dashboard)
```
