from rl18xx.game.engine.game.base import BaseGame
from rl18xx.client.game_sync import GameSync
import ast

PREFIX_STR = "rl18xx.agent.alphazero.self_play - INFO - Game actions: "

def parse_array_from_str(s: str) -> list[str]:
    game_actions = ast.literal_eval(s)
    return game_actions

def get_game_actions_from_log_file(log_file_path: str) -> list[str]:
    with open(log_file_path, "r") as f:
        log_file_content = f.readlines()

    for line in log_file_content:
        if PREFIX_STR in line:
            game_actions_str = line.split(PREFIX_STR)[1].strip()
            game_actions = parse_array_from_str(game_actions_str)
            return game_actions

    raise ValueError(f"Prefix string {PREFIX_STR} not found in log file {log_file_path}")

def get_game_from_actions(game_actions: list[str]) -> BaseGame:
    g = BaseGame.load(
        {
            "id": "hs_inzqxyla_1708318031",
            "players": [{"name": "Player 1","id": 1},{"name": "Player 2","id": 2},{"name": "Player 3","id": 3},{"name": "Player 4","id": 4}],
            "title": "1830",
            "description": "",
            "min_players": "4",
            "max_players": "4",
            "settings": {"optional_rules": [],"seed": 0},
            "actions": game_actions
        })
    return g

def replay_game_from_log_file(log_file_path: str):
    game_actions = get_game_actions_from_log_file(log_file_path)
    game_object = get_game_from_actions(game_actions)
    game_sync = GameSync(game_object)
    print(f"Replicated game from log file {log_file_path} to online game {game_sync.game_id}")
    return game_sync


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_file_path", type=str, required=True)
    args = parser.parse_args()
    replay_game_from_log_file(args.log_file_path)
