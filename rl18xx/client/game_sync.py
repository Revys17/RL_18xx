import random
import string
from typing import Optional, Union

import numpy
from rl18xx.client.ruby_backend_api_client import ApiClient
from rl18xx.game.engine.game import BaseGame
from rl18xx.game.engine.actions import BaseAction
import logging
from rl18xx.agent.alphazero.action_mapper import ActionMapper
from rl18xx.game.gamemap import GameMap
from typing import List

LOGGER = logging.getLogger(__name__)


test_accounts = [
    {
        "name": "a",
        "email": "a@a.a",
        "password": "a"
    },
    {
        "name": "b",
        "email": "b@b.b",
        "password": "b"
    },
    {
        "name": "c",
        "email": "c@c.c",
        "password": "c"
    },
    {
        "name": "z",
        "email": "z@z.z",
        "password": "z"
    }
]

class GameSync:
    def __init__(self, game: Optional[BaseGame] = None):
        self.api_client = ApiClient()
        self.local_game = self.get_fresh_game_state()
        self.action_mapper = ActionMapper()
        self.setup_new_online_game()
        if game is not None:
            LOGGER.info(f"Game already has actions, syncing them")
            self.replay_game(game.raw_actions)

    def get_fresh_game_state(self):
        game_map = GameMap()
        game_class = game_map.game_by_title("1830")
        players = {1: "Player 1", 2: "Player 2", 3: "Player 3", 4: "Player 4"}
        return game_class(players)

    def replay_game(self, actions_to_replay: List[dict]):
        for action in actions_to_replay:
            self.take_synced_action(action)

    def setup_new_online_game(self):
        local_player_order = self.local_game.players
        LOGGER.info(local_player_order)

        auth_token_name_map = {}
        for account in test_accounts:
            _, auth_token = self.api_client.login(account["email"], account["password"])
            LOGGER.info(f"{account['name']} -> {auth_token}")
            auth_token_name_map[account["name"]] = auth_token

        online_game = self.api_client.create_game(auth_token_name_map["a"], self.local_game.seed)
        self.game_id = online_game["id"]
    
        self.api_client.join_game(self.game_id, auth_token_name_map["b"])
        self.api_client.join_game(self.game_id, auth_token_name_map["c"])
        self.api_client.join_game(self.game_id, auth_token_name_map["z"])
        online_game = self.api_client.start_game(self.game_id, auth_token_name_map["a"])
        LOGGER.info(online_game)

        online_player_order = online_game["players"]
        for i, player in enumerate(online_player_order):
            LOGGER.info(f"Player {player['name']} has online id {player['id']} and position {i}")

        self.auth_token_map = {}
        for i, player in enumerate(online_player_order):
            self.auth_token_map[player["id"]] = auth_token_name_map[player["name"]]

        self.local_id_to_online_id_map = {}
        for i, player in enumerate(local_player_order):
            LOGGER.info(f"Player {player.name} has local id {player.id} and position {i}")
            LOGGER.info(f"Mapping {player.id} to {online_player_order[i]['id']}")
            self.local_id_to_online_id_map[player.id] = online_player_order[i]["id"]

    def take_synced_action(
        self,
        action: Union[int, numpy.int64, dict, BaseAction]
    ):
        # If more than one specified, raise an error
        if isinstance(action, int) or isinstance(action, numpy.int64):
            action = self.action_mapper.map_index_to_action(action, self.local_game).to_dict()
        elif isinstance(action, dict):
            action = action
        elif isinstance(action, BaseAction):
            action = action.to_dict()
        else:
            raise ValueError(f"Invalid action type: {type(action)}")

        LOGGER.info(f"Taking action: {action}")

        self.take_online_action(action.copy())
        self.take_local_action(action)

    def take_local_action(self, action_dict: dict):
        LOGGER.info(f"Taking local action: {action_dict}")
        self.local_game.process_action(action_dict)

    def map_action_player_to_online_player(self, action_dict: dict):
        LOGGER.info(f"Mapping action player to online player: {action_dict}")
        local_entity_id = action_dict.get("entity")
        local_entity = getattr(self.local_game, f"{action_dict.get('entity_type')}_by_id")(local_entity_id)
        local_player = local_entity.player()
        LOGGER.info(f"Local player: {local_player}")
        online_player_id = self.local_id_to_online_id_map[local_player.id]
        LOGGER.info(f"Online player id: {online_player_id}")
        return online_player_id

    def update_local_action_to_work_online(self, action_dict: dict, online_player_id: int):
        if action_dict.get("entity_type") == "player":
            LOGGER.info(f"Updating action entity from {action_dict.get('entity')} to {online_player_id}")
            action_dict["entity"] = online_player_id
        return action_dict

    def take_online_action(self, action_dict: dict):
        online_player_id = self.map_action_player_to_online_player(action_dict)
        mapped_action_dict = self.update_local_action_to_work_online(action_dict, online_player_id)
        auth_token = self.auth_token_map[online_player_id]

        try:
            self.api_client.take_action(self.game_id, mapped_action_dict, auth_token)
        except Exception as e:
            LOGGER.error(e)
            LOGGER.error(f"Game actions at this point: {self.local_game.raw_actions}")
            raise e
