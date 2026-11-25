"""The bots on the local copy of 18xx.games (:mod:`bot_hub`'s ``local`` connector).

The copy is the ``rl18xx`` branch of tobymao/18xx (~/workspace/18xx-local, run in docker): a game
created there with bot players seats bot accounts, and its bot API (``routes/bot.rb``, authorized by
the site's ``BOT_API_TOKEN``) lists the games with bots, gives one with the site engine's view of who
must act -- what the extension's page bridge reads from the site's engine in a browser
(:class:`~rl18xx.agent.advisor.autoplay.SiteView`) -- and takes a bot's move, adding the auto actions
it sets off. Its engine also serves the 18xx.games connector (:meth:`SiteClient.engine`).

Every look at the site moves each game waiting on a bot. A move the site refuses where Ruby still
offers a step our 1830 skips (cross-player train and private purchases) becomes that step's pass, as
the extension's autoplay does; any other refusal, or a game the model can't follow, leaves the game
alone until it changes (a human's undo, say) or the admin page says retry.

Games of the other titles the hub has a policy for (1867: ``main.py advisor --title-policy``,
:mod:`titled`) are played the same way, each move the policy's own. Our 1867 engine skips no step
Ruby waits in (its cross-player rule only removes moves, so the bot passes or declines among its
own), so a refused 1867 move is never turned into a pass: the engines disagree, and the game is left
alone with the reason.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

from rl18xx.agent.advisor.autoplay import SKIPPED_PURCHASE_STEPS
from rl18xx.agent.advisor.bot_hub import LOCAL, CannotMove, Connector
from rl18xx.agent.advisor.live_game import SUPPORTED_TITLE

POLL_SECONDS = 1.0  # between looks when no bot has a move to make
IMPLICIT_PASS_STEPS = SKIPPED_PURCHASE_STEPS


class SiteError(RuntimeError):
    """The site answered with an error (``status``: its HTTP status, 0 if it didn't answer)."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


class SiteClient:
    """The local site's bot API."""

    def __init__(self, url: str, token: str, timeout: float = 120.0):
        self.url = url.rstrip("/")
        self.token = token
        self.timeout = timeout

    def _request(self, path: str, body: Optional[dict] = None) -> dict:
        request = urllib.request.Request(
            f"{self.url}/api/bot/{path}",
            data=None if body is None else json.dumps(body).encode(),
            headers={"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"},
            method="GET" if body is None else "POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as e:
            try:
                message = json.loads(e.read()).get("error") or e.reason
            except ValueError:
                message = e.reason
            raise SiteError(e.code, str(message)) from None
        except (urllib.error.URLError, OSError) as e:
            raise SiteError(0, f"no answer from {self.url}: {e}") from None

    def games(self) -> list:
        """The new and active games with a bot (the site's game data without actions, plus ``bots``)."""
        return self._request("games")["games"]

    def game(self, game_id) -> dict:
        """``{game, bots, view}``: the game with its actions, its bots' user ids, the engine's view."""
        return self._request(f"game/{game_id}")

    def post(self, game_id, user_id, last_id: int, action: dict) -> dict:
        return self._request(f"game/{game_id}/action", {"user_id": user_id, "last_id": last_id, "action": action})

    def engine(self, game: dict, action: Optional[dict] = None) -> dict:
        """Another site's game through this site's engine: ``{view}``, and with ``action`` that move as
        the engine stores it (``action``: numbered, with its auto actions) or the engine's ``error``."""
        body = {"game": game}
        if action is not None:
            body["action"] = action
        return self._request("engine", body)


def read_token(token_file) -> str:
    token = Path(token_file).expanduser().read_text().strip()
    if not token:
        raise ValueError(f"{token_file} is empty: it holds the local site's BOT_API_TOKEN")
    return token


def client_for(settings: dict) -> SiteClient:
    return SiteClient(settings["url"], read_token(settings["token_file"]))


def waiting_on(game: dict) -> str:
    names = {p["id"]: p["name"] for p in game.get("players") or []}
    return ", ".join(str(names.get(i, i)) for i in game.get("acting") or [])


class LocalSite(Connector):
    site = LOCAL

    def __init__(self, hub, client: Optional[SiteClient] = None):
        super().__init__(hub)
        self._client = client

    @property
    def client(self) -> SiteClient:
        if self._client is None or self._client.url != self.settings["url"].rstrip("/"):
            self._client = client_for(self.settings)
        return self._client

    def tick(self) -> float:
        games = self.client.games()
        self.last_look = time.time()
        listed = set()
        played = False
        for game in games:
            if self.stopping:
                break
            listed.add(game["id"])
            bots = set(game.get("bots") or [])
            bot_turn = game["status"] == "active" and bool(bots & set(game.get("acting") or []))
            self.hub.update_game(
                self.site,
                game["id"],
                url=f"{self.client.url}/game/{game['id']}",
                title=game.get("title", ""),
                status=game["status"],
                round=f"{game.get('round') or ''} {game.get('turn') or ''}".strip(),
                players=[{"name": p["name"], "bot": p["id"] in bots} for p in game.get("players") or []],
                waiting_on=waiting_on(game),
                bot_turn=bot_turn,
            )
            if bot_turn:
                played = self.play(game["id"]) or played
        gone = [s.game_id for s in list(self.hub.games.values()) if s.site == self.site and s.game_id not in listed]
        self.hub.finished(self.site, gone)
        return 0.0 if played else POLL_SECONDS

    def play(self, game_id) -> bool:
        """The model's move in ``game_id`` if a bot must act there; whether one was posted."""
        data = self.client.game(game_id)
        game, view, bots = data["game"], data["view"], set(data["bots"])
        last_id = int(view.get("last_id") or 0)
        if self.hub.stuck(self.site, game_id, last_id):
            return False
        self.hub.unstick(self.site, game_id)
        if view.get("error"):
            self.hub.give_up(self.site, game_id, last_id, f"the site's engine can't load it: {view['error']}")
            return False
        entity = view.get("entity") or {}
        if view.get("finished") or entity.get("player") not in bots:
            return False
        try:
            result = self.hub.search_move(self.site, game, view, bots, self.settings["readouts"])
        except CannotMove as e:
            self.hub.give_up(self.site, game_id, last_id, str(e))
            return False
        if result is None:
            return False  # another search is running: on the next look
        player = entity["player"]
        try:
            self.client.post(game_id, player, last_id, result["action"])
        except SiteError as e:
            if e.status in (0, 409):  # no answer / the game moved on: look again
                return False
            skips = str(game.get("title")) == SUPPORTED_TITLE  # only our 1830 skips steps Ruby waits in
            if not skips or not self._implicit_pass(game_id, view, player, last_id, result["action"]):
                self.hub.give_up(self.site, game_id, last_id, f'the site refused "{result["description"]}": {e}')
                return False
            self.hub.moved(self.site, game_id, result["actor"], f"Pass ({view.get('step')}): the site refused {e}")
            return True
        self.hub.moved(self.site, game_id, result["actor"], result["description"])
        return True

    def _implicit_pass(self, game_id, view: dict, player, last_id: int, action: dict) -> bool:
        """Where Ruby waits in a purchase step our 1830 skips, the bot passes it; whether it did."""
        if action.get("type") == "pass" or "pass" not in (view.get("actions") or []):
            return False
        if view.get("step") not in IMPLICIT_PASS_STEPS:
            return False
        entity = view["entity"]
        try:
            self.client.post(
                game_id, player, last_id, {"type": "pass", "entity": entity["id"], "entity_type": entity["type"]}
            )
        except SiteError:
            return False
        return True
