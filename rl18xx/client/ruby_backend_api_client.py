import requests
from typing import Optional
import logging

LOGGER = logging.getLogger(__name__)

class ApiClient:
    def __init__(self, base_url: str = "http://localhost:9292"):
        self.base_url = base_url

    def login(self, email: str, password: str) -> tuple[dict, str]:
        response = requests.post(f"{self.base_url}/api/user/login", json={"email": email, "password": password})
        return response.json(), response.cookies.get_dict()["auth_token"]

    def create_game(self, auth_token: str, seed: Optional[int] = None):
        json = {
            "Multiplayer": True,
            "Hotseat": False,
            "Import hotseat game": False,
            "description": "",
            "min_players": "4",
            "max_players": "4",
            "async": True,
            "live": False,
            "unlisted": False,
            "auto_routing": False,
            "seed": seed,
            "keywords": "",
            "title": "1830",
            "optional_rules": [],
        }

        response = requests.post(f"{self.base_url}/api/game", json=json, cookies={"auth_token": auth_token})
        return response.json()

    def join_game(self, game_id: int, auth_token: str):
        response = requests.post(f"{self.base_url}/api/game/{game_id}/join", cookies={"auth_token": auth_token})
        return response.json()

    def start_game(self, game_id: int, auth_token: str):
        response = requests.post(f"{self.base_url}/api/game/{game_id}/start", cookies={"auth_token": auth_token})
        return response.json()

    def get_game_state(self, game_id: int):
        response = requests.get(f"{self.base_url}/api/game/{game_id}")
        return response.json()

    def take_action(self, game_id: int, action: dict, auth_token: str):
        response = requests.post(f"{self.base_url}/api/game/{game_id}/action", json=action, cookies={"auth_token": auth_token})
        if response.json().get("error", None) is not None:
            LOGGER.error(response.json()["error"])
            raise Exception(response.json()["error"])

        return response.json()
